import os
import sys
import pytest
import struct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sdk.python.qerlio import QerlioEngine, GenerationConfig
from tests.test_gguf_integrity import create_synthetic_gguf

def test_full_lifecycle_streaming(tmp_path):
    model_file = tmp_path / "test_model.gguf"
    create_synthetic_gguf(str(model_file), magic=b"GGUF", version=3, tensor_count=50, kv_count=12)

    # 1. Initialize Engine
    engine = QerlioEngine(max_concurrent_sessions=2)
    assert len(engine.loaded_models) == 0

    # 2. Load Model
    model = engine.load_model(str(model_file), context_length=2048)
    assert model.header_info["tensor_count"] == 50

    # 3. Create Session
    session = model.create_session()
    assert session.is_generating is False
    assert session.is_cancelled is False

    # 4. Stream Generation
    stream = session.generate_stream("What is local AI?", GenerationConfig(max_output_tokens=8))
    collected_tokens = []
    for chunk in stream:
        collected_tokens.append(chunk)

    assert len(collected_tokens) == 8
    assert "".join(collected_tokens).strip() != ""

    # 5. Clean teardown
    engine.shutdown()
    assert len(engine.loaded_models) == 0

def test_cancellation_mid_stream(tmp_path):
    model_file = tmp_path / "test_cancel_model.gguf"
    create_synthetic_gguf(str(model_file), magic=b"GGUF", version=3)

    engine = QerlioEngine()
    model = engine.load_model(str(model_file))
    session = model.create_session()

    # Cancel via callback after 3 tokens
    collected_tokens = []
    def stop_callback(token):
        collected_tokens.append(token)
        return len(collected_tokens) < 3  # Return False to cancel when reach 3

    stream = session.generate_stream("Prompt", GenerationConfig(max_output_tokens=15), callback=stop_callback)
    list(stream)

    assert len(collected_tokens) == 3
    assert session.is_cancelled is True

    engine.shutdown()
