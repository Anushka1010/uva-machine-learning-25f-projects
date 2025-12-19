"""TCF Reviews Loader - loads instructor reviews from cached JSON."""

import json
import os
from typing import Optional
from functools import lru_cache


class TCFReviewsLoader:
    """Loads TCF instructor reviews from cache and matches by course + instructor."""
    
    def __init__(self, cache_path: str = "data/cache/tcf_instructor_reviews.json"):
        self.cache_path = cache_path
        self.data = self._load()
    
    def _load(self) -> dict:
        """Load TCF reviews from JSON file."""
        if not os.path.exists(self.cache_path):
            print(f"[TCF] Cache not found: {self.cache_path}")
            return {}
        
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[TCF] Error loading cache: {e}")
            return {}
    
    def _normalize_name(self, name: str) -> str:
        """Normalize instructor name for matching."""
        return name.lower().strip()
    
    def _key(self, subject: str, catalog_number: str) -> str:
        """Generate cache key for a course."""
        return f"{subject.upper()}_{catalog_number}"
    
    def get_course_instructors(self, subject: str, catalog_number: str) -> list[dict]:
        """Get all instructors for a course.
        
        Args:
            subject: Course subject (e.g., "CS")
            catalog_number: Course number (e.g., "4774")
            
        Returns:
            List of instructor dicts with reviews
        """
        key = self._key(subject, catalog_number)
        course_data = self.data.get(key, {})
        return course_data.get("instructors", [])
    
    def get_reviews_for_instructor(
        self,
        subject: str,
        catalog_number: str,
        instructor_name: str,
        limit: int = 5,
    ) -> Optional[dict]:
        """Get TCF data for a specific instructor teaching a specific course.
        
        Args:
            subject: Course subject (e.g., "CS")
            catalog_number: Course number (e.g., "4774")
            instructor_name: Instructor's full name
            limit: Max number of review texts to return
            
        Returns:
            Dict with instructor stats and reviews, or None if not found
        """
        instructors = self.get_course_instructors(subject, catalog_number)
        
        if not instructors:
            return None
        
        # Normalize the search name
        search_name = self._normalize_name(instructor_name)
        search_parts = search_name.split()
        
        for inst in instructors:
            tcf_name = self._normalize_name(inst.get("instructor_name", ""))
            
            # Try exact match first
            if tcf_name == search_name:
                return self._format_result(inst, limit)
            
            # Try matching by last name if first + last provided
            if len(search_parts) >= 2:
                search_last = search_parts[-1]
                tcf_parts = tcf_name.split()
                if tcf_parts and tcf_parts[-1] == search_last:
                    # Also check first name initial or first name
                    search_first = search_parts[0]
                    tcf_first = tcf_parts[0] if tcf_parts else ""
                    if tcf_first.startswith(search_first) or search_first.startswith(tcf_first):
                        return self._format_result(inst, limit)
        
        return None
    
    def _format_result(self, inst: dict, limit: int) -> dict:
        """Format instructor data for output."""
        reviews = inst.get("reviews", [])
        
        # Filter out spam/test reviews
        valid_reviews = []
        for r in reviews:
            text = r.get("text", "")
            if len(text) > 50 and "app app app" not in text.lower():
                valid_reviews.append(text)
        
        return {
            "instructor_name": inst.get("instructor_name", ""),
            "rating": inst.get("rating"),
            "difficulty": inst.get("difficulty"),
            "gpa": inst.get("gpa"),
            "review_count": inst.get("review_count", len(reviews)),
            "sample_reviews": valid_reviews[:limit],
        }
    
    def summarize_instructor(self, subject: str, catalog_number: str, instructor_name: str) -> Optional[dict]:
        """Get a summary of an instructor for a specific course.
        
        Args:
            subject: Course subject
            catalog_number: Course number
            instructor_name: Instructor name
            
        Returns:
            Summary dict or None
        """
        result = self.get_reviews_for_instructor(subject, catalog_number, instructor_name)
        if not result:
            return None
        
        return {
            "instructor_name": result["instructor_name"],
            "rating": result["rating"],
            "difficulty": result["difficulty"],
            "gpa": result["gpa"],
            "review_count": result["review_count"],
            "sample_reviews": result["sample_reviews"][:3],
        }


# Singleton pattern for efficiency
_tcf_loader: Optional[TCFReviewsLoader] = None


def get_tcf_loader() -> TCFReviewsLoader:
    """Get the singleton TCF reviews loader."""
    global _tcf_loader
    if _tcf_loader is None:
        _tcf_loader = TCFReviewsLoader()
    return _tcf_loader

