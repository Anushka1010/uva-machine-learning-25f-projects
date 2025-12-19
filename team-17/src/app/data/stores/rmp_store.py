"""RMP data store - caches professor reviews as JSON."""

import json
import os
from typing import List


class RMPStore:
    """Simple JSON cache for RateMyProfessor reviews."""

    def __init__(self, base_path: str):
        """Initialize store.
        
        Args:
            base_path: Base path for cache files (e.g., "data/cache/rmp_reviews_1277")
        """
        self.path = base_path + ".json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def has_reviews(self, tid: int) -> bool:
        return str(tid) in self.data
    
    def append_reviews(self, tid: int, professor_name: str, reviews: List[dict]) -> None:
        self.data[str(tid)] = {
            "professor_name": professor_name,
            "reviews": reviews,
        }
        self._save()
    
    def get_reviews(self, tid: int, limit: int = 200) -> List[dict]:
        entry = self.data.get(str(tid))
        if not entry:
            return []
        return entry.get("reviews", [])[:limit]
    
    def clear(self) -> None:
        self.data = {}
        self._save()
