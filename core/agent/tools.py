import math
import sys
from pathlib import Path
from typing import Any, Dict

class ToolRegistry:
    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = workspace_root or Path.cwd()

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_tool_{tool_name}", None)
        if not handler:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not recognized in agent sandbox"
            }
        try:
            result = handler(**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_calculator(self, expression: str) -> str:
        allowed_names = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow
        }
        clean_expr = expression.strip()
        code = compile(clean_expr, "<calculator>", "eval")
        for name in code.co_names:
            if name not in allowed_names and not hasattr(math, name):
                raise ValueError(f"Access to '{name}' is not allowed in sandbox")
        res = eval(code, {"__builtins__": {}}, allowed_names)
        return str(res)

    def _tool_python_eval(self, code: str) -> str:
        safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "bool": bool
        }
        loc: Dict[str, Any] = {}
        exec(code, {"__builtins__": safe_builtins, "math": math}, loc)
        if "result" in loc:
            return str(loc["result"])
        return "Executed successfully"

    def _tool_file_search(self, pattern: str, directory: str = ".") -> list[str]:
        base = (self.workspace_root / directory).resolve()
        if not str(base).startswith(str(self.workspace_root.resolve())):
            raise PermissionError("Directory traversal outside workspace is restricted")
        matches = []
        for p in base.glob(pattern):
            if p.is_file():
                matches.append(str(p.relative_to(self.workspace_root)))
            if len(matches) >= 25:
                break
        return matches

    def _tool_web_search(self, query: str) -> list[dict]:
        return [
            {
                "title": f"Result for {query}",
                "snippet": f"Synthetic edge-retrieved information for query: {query}",
                "source": "https://qerlio.local/search"
            }
        ]

    def get_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Safe mathematical evaluator",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "python_eval",
                    "description": "Execute pure sandboxed Python arithmetic and data processing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python snippet"}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_search",
                    "description": "Search workspace directory for file patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern"},
                            "directory": {"type": "string", "default": "."}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search web sources for knowledge queries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query text"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
