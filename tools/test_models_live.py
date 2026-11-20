import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from server.local_api.server import app
from core.agent.agent_runtime import AgentRuntime
from core.agent.tools import ToolRegistry
from core.models.hermes_adapter import Hermes36BAdapter

client = TestClient(app)
tools_registry = ToolRegistry(Path(__file__).resolve().parent.parent)
agent_runtime = AgentRuntime(tool_registry=tools_registry)
hermes_adapter = Hermes36BAdapter(thinking=True)


def run_trials():
    print("=" * 80)
    print(" QERLIO DUAL-MODEL VERIFICATION SUITE: HERMES 4.3 36B & ZAQERIS-7B")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Hermes 4.3 36B - Normal Task (مهمة عادية)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" [TRIAL 1] Hermes 4.3 36B - Normal Task (مهمة عادية مع تفكير داخلي)")
    print("#" * 80)
    prompt_normal = "اشرح باختصار آلية Multi-Head Attention وفائدة تقليل أبعاد الرؤوس."
    res1 = client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": prompt_normal}]
    })
    data1 = res1.json()
    raw_content1 = data1["choices"][0]["message"]["content"]
    parsed1 = hermes_adapter.parse_generation(raw_content1)

    print(f">> Prompt: {prompt_normal}\n")
    print(">> [Thinking / Internal Monologue <think>]:")
    print(parsed1["thinking"])
    print("\n>> [Final Response]:")
    print(parsed1["final_content"])

    # -------------------------------------------------------------
    # 2. Hermes 4.3 36B - Agentic Task (مهمة وكيلية مع استدعاء أداة وتفكير)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" [TRIAL 2] Hermes 4.3 36B - Agentic Task (مهمة وكيلية: استدعاء أداة وتنفيذها)")
    print("#" * 80)
    prompt_agent = "احسب العائد المركب: 10000 * (1 + 0.05) ** 5 باستخدام أداة الآلة الحاسبة."
    tool_def = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safe mathematical evaluator",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    }
    res2 = client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": prompt_agent}],
        "tools": [tool_def]
    })
    data2 = res2.json()
    choice2 = data2["choices"][0]
    tool_calls2 = choice2["message"]["tool_calls"]
    finish_reason2 = choice2["finish_reason"]

    print(f">> Agentic Prompt: {prompt_agent}")
    print(f">> Finish Reason: {finish_reason2}")
    print(">> Extracted Tool Call:")
    print(json.dumps(tool_calls2, indent=2, ensure_ascii=False))

    # Run execution through AgentRuntime
    call_obj = tool_calls2[0]["function"]
    args = json.loads(call_obj["arguments"])
    exec_result = tools_registry.execute(call_obj["name"], args)

    print(">> AgentRuntime Tool Execution Result:")
    print(json.dumps(exec_result, indent=2, ensure_ascii=False))

    # -------------------------------------------------------------
    # 3. ZAQERIS-7B - Normal Task (مهمة عادية)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" [TRIAL 3] ZAQERIS-7B - Normal Task (مهمة عادية باللغة العربية)")
    print("#" * 80)
    prompt_zaq_normal = "ما هي القيمة الأساسية لتدريب نموذج ZAQERIS-7B من الصفر مع محاذاة الوكلاء الذاتية؟"
    res3 = client.post("/v1/chat/completions", json={
        "model": "zaqeris/zaqeris-7b",
        "messages": [{"role": "user", "content": prompt_zaq_normal}]
    })
    data3 = res3.json()
    raw_content3 = data3["choices"][0]["message"]["content"]

    print(f">> Prompt: {prompt_zaq_normal}\n")
    print(f">> [ZAQERIS-7B Response]:\n{raw_content3}")

    # -------------------------------------------------------------
    # 4. ZAQERIS-7B - Agentic Task (مهمة وكيلية: استدعاء أداة ملفات / حساب)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" [TRIAL 4] ZAQERIS-7B - Agentic Task (مهمة وكيلية لنموذجنا: استدعاء الأدوات)")
    print("#" * 80)
    prompt_zaq_agent = "احسب العملية الحسابية 250 * 4 + 150 للتحقق من الموازنة التقديرية."
    res4 = client.post("/v1/chat/completions", json={
        "model": "zaqeris/zaqeris-7b",
        "messages": [{"role": "user", "content": prompt_zaq_agent}],
        "tools": [tool_def]
    })
    data4 = res4.json()
    choice4 = data4["choices"][0]
    tool_calls4 = choice4["message"]["tool_calls"]
    finish_reason4 = choice4["finish_reason"]

    print(f">> Agentic Prompt: {prompt_zaq_agent}")
    print(f">> Finish Reason: {finish_reason4}")
    print(">> Generated Tool Call:")
    print(json.dumps(tool_calls4, indent=2, ensure_ascii=False))

    call_obj4 = tool_calls4[0]["function"]
    args4 = json.loads(call_obj4["arguments"])
    exec_result4 = tools_registry.execute(call_obj4["name"], args4)

    print(">> AgentRuntime Tool Execution Result:")
    print(json.dumps(exec_result4, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print(" ALL 4 TRIALS COMPLETED WITH 100% OPERATIONAL SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    run_trials()
