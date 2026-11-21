import os
import sys
import time
import json
import uuid
import psutil
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Header, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sdk.python.qerlio import QerlioEngine, GenerationConfig, PowerProfile
from core.compression.quant_matrix import QUANT_CATALOG, compute_layer_adaptive_mixed_precision, calculate_sink_aware_kv_cache
from core.model_registry.architecture_catalog import ArchitectureCatalog
from core.agent.agent_runtime import AgentRuntime
from core.agent.tools import ToolRegistry

app = FastAPI(
    title="QERLIO Universal Multi-API Gateway",
    version="0.2.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE = QerlioEngine(max_concurrent_sessions=8)
ACTIVE_MODELS: Dict[str, Any] = {}
SERVER_API_KEY = os.environ.get("QERLIO_API_KEY", "")
ARCH_CATALOG = ArchitectureCatalog()
TOOL_REGISTRY = ToolRegistry()
AGENT_RUNTIME = AgentRuntime(tool_registry=TOOL_REGISTRY)


class OpenAIChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""


class OpenAIChatRequest(BaseModel):
    model: str = "qerlio-adaptive-balanced"
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    power_profile: Optional[str] = "balanced"
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class AnthropicRequest(BaseModel):
    model: str = "qerlio-adaptive-balanced"
    messages: List[AnthropicMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    system: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None


class OllamaChatRequest(BaseModel):
    model: str = "qerlio-adaptive-balanced"
    messages: List[OpenAIChatMessage]
    stream: Optional[bool] = False


class OllamaGenerateRequest(BaseModel):
    model: str = "qerlio-adaptive-balanced"
    prompt: str
    stream: Optional[bool] = False


class AgentExecuteRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    prompt: str
    model: Optional[str] = "zaqeris/zqn-12b"
    max_iterations: Optional[int] = 3


def verify_token(authorization: Optional[str] = Header(None)):
    if not SERVER_API_KEY:
        return True
    if not authorization:
        return True
    token = authorization.replace("Bearer ", "").strip()
    if token != SERVER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized"
        )
    return True


@app.get("/v1/health")
@app.get("/healthz")
def get_health():
    ram = psutil.virtual_memory()
    return {
        "status": "healthy",
        "runtime": "QERLIO Universal Gateway 0.2.2",
        "cpu_usage_pct": psutil.cpu_percent(),
        "ram_available_gb": round(ram.available / (1024**3), 2),
        "gateways": ["openai", "anthropic", "ollama"],
        "supported_architectures_count": ARCH_CATALOG.total_count,
        "tested_architectures_count": ARCH_CATALOG.tested_count,
        "experimental_architectures_count": ARCH_CATALOG.experimental_count,
        "agent_mode_ready": True
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>QERLIO Dashboard</h1>"


@app.get("/v1/telemetry/hud")
def get_telemetry_hud():
    ram = psutil.virtual_memory()
    # The gateway currently serves simulated responses, so there is no live
    # inference to measure. Report null instead of fixed placeholder numbers.
    return {
        "measurement_status": "no_live_measurement",
        "throughput_tps": None,
        "ttft_ms": None,
        "model_memory_gb": None,
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_available_gb": round(ram.available / (1024**3), 2),
        "thermal_state": None,
        "quant_schemes_count": len(QUANT_CATALOG),
        "supported_architectures_count": ARCH_CATALOG.total_count,
        "tested_architectures_count": ARCH_CATALOG.tested_count,
        "experimental_architectures_count": ARCH_CATALOG.experimental_count,
        "architectures_families": ARCH_CATALOG.list_families(),
        "adaptive_hybrid_7b": compute_layer_adaptive_mixed_precision(7.0),
        "sink_aware_16k": calculate_sink_aware_kv_cache(context_tokens=16384)
    }


@app.get("/v1/models")
def list_openai_models(_: bool = Depends(verify_token)):
    base_models = [
        {"id": "zaqeris/zaqeris-7b", "object": "model", "created": int(time.time()), "owned_by": "zaqeris-ai"},
        {"id": "zaqeris/zqn-12b", "object": "model", "created": int(time.time()), "owned_by": "zaqeris-ai"},
        {"id": "zaqeris/zaqeris-axiom", "object": "model", "created": int(time.time()), "owned_by": "zaqeris-ai"},
        {"id": "NousResearch/Hermes-4.3-36B", "object": "model", "created": int(time.time()), "owned_by": "nousresearch"},
        {"id": "qerlio-adaptive-balanced", "object": "model", "created": int(time.time()), "owned_by": "zaqeris-ai"},
        {"id": "qerlio-reference-cpu", "object": "model", "created": int(time.time()), "owned_by": "zaqeris-ai"}
    ]
    return {
        "object": "list",
        "data": base_models
    }


@app.get("/v1/architectures")
def list_architectures(family: Optional[str] = None, search: Optional[str] = None, status: Optional[str] = None):
    if family:
        items = ARCH_CATALOG.list_by_family(family)
    elif search:
        items = ARCH_CATALOG.search(search)
    else:
        items = ARCH_CATALOG.list_all()
    if status:
        items = [i for i in items if i.get("support_status", "").lower() == status.lower()]
    return {
        "total_count": len(items),
        "global_supported_count": ARCH_CATALOG.total_count,
        "tested_count": ARCH_CATALOG.tested_count,
        "experimental_count": ARCH_CATALOG.experimental_count,
        "families": ARCH_CATALOG.list_families(),
        "data": items
    }


@app.get("/v1/architectures/{arch_id}")
def get_architecture(arch_id: str):
    item = ARCH_CATALOG.get(arch_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Architecture '{arch_id}' not found")
    return item


@app.post("/v1/agents/execute")
def agent_execute_tool(req: AgentExecuteRequest, _: bool = Depends(verify_token)):
    res = TOOL_REGISTRY.execute(req.tool, req.arguments)
    return res


@app.post("/v1/agents/run")
def agent_run_step(req: AgentRunRequest, _: bool = Depends(verify_token)):
    simulated_step = f'<think>\nDeliberating prompt: {req.prompt}\n</think>\n<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "100 / 4"}}}}\n</tool_call>'
    step_res = AGENT_RUNTIME.run_step(simulated_step)
    return {
        "model": req.model,
        "prompt": req.prompt,
        "simulated": True,
        "step": step_res
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatRequest, _: bool = Depends(verify_token)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if req.tools and len(req.tools) > 0:
        first_tool = req.tools[0]
        tool_name = first_tool.get("function", {}).get("name", "calculator")
        tool_call_payload = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps({"expression": "42 * 2"})
            }
        }
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "simulated": True,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call_payload]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": None
        }

    if req.stream:
        def sse_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': True, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            chunks = ["Hello", " from", " QERLIO", " Universal", " API", " Gateway", "!"]
            for chunk in chunks:
                time.sleep(0.01)
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "simulated": True,
                    "choices": [{"index": 0, "delta": {"content": " " + chunk}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(data)}\n\n"

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': True, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    content = "Hello from QERLIO Universal API Gateway! Your request was processed securely."
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "simulated": True,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }
        ],
        "usage": None
    }


@app.post("/v1/messages")
def anthropic_messages(req: AnthropicRequest, _: bool = Depends(verify_token)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    msg_id = f"msg_{uuid.uuid4().hex[:16]}"
    text_out = "QERLIO Anthropic-compatible gateway response."

    if req.tools and len(req.tools) > 0:
        first_tool = req.tools[0]
        tool_name = first_tool.get("name", "calculator")
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "simulated": True,
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": {"expression": "25 * 4"}
                }
            ],
            "model": req.model,
            "stop_reason": "tool_use",
            "usage": None
        }

    if req.stream:
        def anthropic_sse():
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'simulated': True, 'content': [], 'model': req.model}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            for word in text_out.split():
                time.sleep(0.01)
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': ' ' + word}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return StreamingResponse(anthropic_sse(), media_type="text/event-stream")

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "simulated": True,
        "content": [{"type": "text", "text": text_out}],
        "model": req.model,
        "stop_reason": "end_turn",
        "usage": None
    }


@app.get("/api/tags")
def ollama_tags():
    return {
        "models": [
            {
                "name": "zaqeris-7b:latest",
                "model": "zaqeris-7b:latest",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": None,
                "digest": None,
                "details": {"format": "gguf", "family": "zaqeris_7b", "parameter_size": "7B", "quantization_level": "Q4_K_M"}
            },
            {
                "name": "zqn-12b:latest",
                "model": "zqn-12b:latest",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": None,
                "digest": None,
                "details": {"format": "gguf", "family": "zqn_12b", "parameter_size": "12B", "quantization_level": "M3_GPTQ"}
            },
            {
                "name": "qerlio:latest",
                "model": "qerlio:latest",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": None,
                "digest": None,
                "details": {"format": "gguf", "family": "llama", "parameter_size": "1B", "quantization_level": "Q4_K_M"}
            }
        ]
    }


@app.post("/api/generate")
def ollama_generate(req: OllamaGenerateRequest):
    if req.stream:
        def ndjson_stream():
            words = ["QERLIO", "Ollama", "gateway", "streaming", "output."]
            for word in words:
                time.sleep(0.01)
                yield json.dumps({"model": req.model, "response": " " + word, "done": False, "simulated": True}) + "\n"
            yield json.dumps({"model": req.model, "response": "", "done": True, "simulated": True, "total_duration": None}) + "\n"

        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": "Hello from QERLIO Ollama-compatible endpoint.",
        "done": True,
        "simulated": True,
        "total_duration": None
    }


@app.post("/api/chat")
def ollama_chat(req: OllamaChatRequest):
    if req.stream:
        def ndjson_chat():
            words = ["Hello", "from", "QERLIO", "chat."]
            for word in words:
                time.sleep(0.01)
                yield json.dumps({"model": req.model, "message": {"role": "assistant", "content": " " + word}, "done": False, "simulated": True}) + "\n"
            yield json.dumps({"model": req.model, "done": True, "simulated": True}) + "\n"

        return StreamingResponse(ndjson_chat(), media_type="application/x-ndjson")

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": "Hello from QERLIO Ollama chat endpoint."},
        "done": True,
        "simulated": True
    }


def start_server(host: str = "127.0.0.1", port: int = 8787):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
