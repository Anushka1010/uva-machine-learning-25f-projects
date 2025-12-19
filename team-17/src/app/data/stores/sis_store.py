"""SIS API data store - caches course data as JSON."""

import json
import os
from typing import Any, List


class SISStore:
    """Simple JSON cache for SIS course data."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _path(self, term: str) -> str:
        return os.path.join(self.cache_dir, f"sis_courses_{term}.json")
    
    def has(self, term: str) -> bool:
        return os.path.exists(self._path(term))
    
    def save(self, term: str, courses: List[dict]) -> None:
        with open(self._path(term), "w", encoding="utf-8") as f:
            json.dump(courses, f, ensure_ascii=False, indent=2)
    
    def load(self, term: str) -> List[dict]:
        if not self.has(term):
            return []
        with open(self._path(term), "r", encoding="utf-8") as f:
            return json.load(f)
    
    def clear(self, term: str = None) -> None:
        if term:
            path = self._path(term)
            if os.path.exists(path):
                os.remove(path)
        else:
            for f in os.listdir(self.cache_dir):
                if f.startswith("sis_courses_"):
                    os.remove(os.path.join(self.cache_dir, f))
