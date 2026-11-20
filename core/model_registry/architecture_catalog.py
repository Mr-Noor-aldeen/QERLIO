import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class ArchitectureCatalog:
    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path:
            self.registry_file = Path(registry_path)
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.registry_file = base_dir / "registry" / "model_architectures.json"
        
        self._data: Dict[str, Any] = {}
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_file.is_file():
            raise FileNotFoundError(f"Architectures file not found: {self.registry_file}")
        with open(self.registry_file, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        for item in self._data.get("architectures", []):
            self._by_id[item["id"]] = item

    @property
    def total_count(self) -> int:
        return len(self._by_id)

    @property
    def tested_count(self) -> int:
        return len(self.list_by_status("tested"))

    @property
    def experimental_count(self) -> int:
        return len(self.list_by_status("experimental"))

    def list_by_status(self, status: str) -> List[Dict[str, Any]]:
        return [item for item in self._by_id.values() if item.get("support_status", "").lower() == status.lower()]

    def get(self, arch_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(arch_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._by_id.values())

    def list_by_family(self, family: str) -> List[Dict[str, Any]]:
        return [item for item in self._by_id.values() if item.get("family", "").lower() == family.lower()]

    def list_families(self) -> List[str]:
        return sorted(list(set(item["family"] for item in self._by_id.values())))

    def search(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        return [
            item for item in self._by_id.values()
            if q in item["id"].lower() or q in item["name"].lower() or q in item["developer"].lower() or q in item["family"].lower()
        ]
