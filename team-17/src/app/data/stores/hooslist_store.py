"""Hooslist API data store - caches course descriptions as JSON."""

import json
import os
from typing import Optional


class HooslistStore:
    """Simple JSON cache for Hooslist descriptions."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "hooslist_descriptions.json")
        self.data = self._load()
    
    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def _key(self, subject: str, catalog_number: str) -> str:
        return f"{subject.upper()}_{catalog_number}"
    
    def has(self, subject: str, catalog_number: str) -> bool:
        return self._key(subject, catalog_number) in self.data
    
    def save(self, subject: str, catalog_number: str, description: str, prerequisites: str = "") -> None:
        self.data[self._key(subject, catalog_number)] = {
            "description": description,
            "prerequisites": prerequisites,
        }
        self._save()
    
    def load(self, subject: str, catalog_number: str) -> Optional[dict]:
        return self.data.get(self._key(subject, catalog_number))
    
    def clear(self) -> None:
        self.data = {}
        self._save()
