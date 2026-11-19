import pytest

def calculate_kv_cache_bytes(layers, kv_heads, head_dim, context_len, bytes_per_element=2):
    return 2 * layers * kv_heads * head_dim * context_len * bytes_per_element

def plan_memory(model_size_bytes, layers, kv_heads, head_dim, requested_context, available_ram_bytes):
    workspace = 128 * 1024 * 1024       # 128 MB
    safety_margin = 512 * 1024 * 1024   # 512 MB
    
    # Check requested context
    kv = calculate_kv_cache_bytes(layers, kv_heads, head_dim, requested_context)
    total = model_size_bytes + kv + workspace + safety_margin
    if total <= available_ram_bytes:
        return {"fits": True, "context": requested_context, "kv_bytes": kv, "total_bytes": total}
        
    # Check fallbacks
    for fallback in [2048, 1024, 512]:
        if fallback >= requested_context:
            continue
        kv = calculate_kv_cache_bytes(layers, kv_heads, head_dim, fallback)
        total = model_size_bytes + kv + workspace + safety_margin
        if total <= available_ram_bytes:
            return {"fits": True, "context": fallback, "kv_bytes": kv, "total_bytes": total}
            
    return {"fits": False, "context": 0, "kv_bytes": kv, "total_bytes": total}

def compute_scheduler_policy(physical_cores, power_profile, thermal_state):
    # power_profile: 0=Balanced, 1=Saver, 2=Performance
    # thermal_state: 0=Normal, 1=Warm, 2=Hot, 3=Critical
    if thermal_state == 3:  # Critical
        return {"allow": False, "threads": 1, "batch": 128}
        
    cores = max(1, physical_cores)
    if power_profile == 1:    # Saver
        threads = max(1, cores // 2)
        batch = 256
    elif power_profile == 2:  # Performance
        threads = cores
        batch = 512
    else:                     # Balanced
        threads = max(2, min(cores, 6))
        batch = 512
        
    if thermal_state == 1:    # Warm
        threads = max(1, (threads * 3) // 4)
    elif thermal_state == 2:  # Hot
        threads = max(1, threads // 2)
        batch = 128
        
    return {"allow": True, "threads": threads, "batch": batch}

def test_kv_cache_calculation():
    # 32 layers, 8 KV heads (GQA), 128 head dim, 2048 context, F16 (2 bytes)
    # 2 * 32 * 8 * 128 * 2048 * 2 = 268,435,456 bytes (256 MB)
    kv = calculate_kv_cache_bytes(layers=32, kv_heads=8, head_dim=128, context_len=2048, bytes_per_element=2)
    assert kv == 256 * 1024 * 1024

def test_memory_planning_fits_requested():
    model_size = 4 * 1024 * 1024 * 1024  # 4 GB model (e.g., Q4_K_M 7B)
    available_ram = 16 * 1024 * 1024 * 1024 # 16 GB RAM
    res = plan_memory(model_size, layers=32, kv_heads=8, head_dim=128, requested_context=4096, available_ram_bytes=available_ram)
    assert res["fits"] is True
    assert res["context"] == 4096

def test_memory_planning_downgrades_on_pressure():
    model_size = 3 * 1024 * 1024 * 1024  # 3 GB model
    available_ram = 3800 * 1024 * 1024   # 3.8 GB available
    # 4096 context needs 512MB KV + 128MB ws + 512MB margin = 1152 MB -> total > 4.1 GB (fails)
    # Downgrades to 2048 or 1024
    res = plan_memory(model_size, layers=32, kv_heads=8, head_dim=128, requested_context=4096, available_ram_bytes=available_ram)
    assert res["fits"] is True
    assert res["context"] in [2048, 1024, 512]

def test_memory_planning_fails_safely_when_model_too_large():
    model_size = 14 * 1024 * 1024 * 1024  # 14 GB model
    available_ram = 4 * 1024 * 1024 * 1024  # 4 GB RAM available
    res = plan_memory(model_size, layers=32, kv_heads=8, head_dim=128, requested_context=2048, available_ram_bytes=available_ram)
    assert res["fits"] is False
    assert res["context"] == 0

def test_scheduler_power_profiles():
    # 16 physical cores
    p_perf = compute_scheduler_policy(physical_cores=16, power_profile=2, thermal_state=0)
    assert p_perf["threads"] == 16
    assert p_perf["allow"] is True
    
    p_saver = compute_scheduler_policy(physical_cores=16, power_profile=1, thermal_state=0)
    assert p_saver["threads"] == 8
    
    p_balanced = compute_scheduler_policy(physical_cores=16, power_profile=0, thermal_state=0)
    assert p_balanced["threads"] == 6

def test_scheduler_thermal_throttling():
    # In Warm state, Balanced profile threads reduce from 6 to 4
    p_warm = compute_scheduler_policy(physical_cores=16, power_profile=0, thermal_state=1)
    assert p_warm["threads"] == 4
    
    # In Hot state, Balanced threads reduce to 3 and batch size is halved
    p_hot = compute_scheduler_policy(physical_cores=16, power_profile=0, thermal_state=2)
    assert p_hot["threads"] == 3
    assert p_hot["batch"] == 128
    
    # In Critical state, generation is disallowed to protect device
    p_critical = compute_scheduler_policy(physical_cores=16, power_profile=0, thermal_state=3)
    assert p_critical["allow"] is False
