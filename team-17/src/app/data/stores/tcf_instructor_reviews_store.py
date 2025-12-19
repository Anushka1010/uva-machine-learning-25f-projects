"""TCF instructor review store - caches per-course instructor reviews as JSON."""

import json
import os
from typing import List, Optional, Dict


class TCFInstructorReviewsStore:
    """Simple JSON cache for TheCourseForum instructor reviews per course."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "tcf_instructor_reviews.json")
        self.data = self._load()
    
    def _load(self) -> dict:
        """Load existing cache from disk."""
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save(self) -> None:
        """Save cache to disk."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def _key(self, subject: str, catalog_nbr: str) -> str:
        """Generate cache key for a course."""
        return f"{subject.upper()}_{catalog_nbr}"
    
    def has(self, subject: str, catalog_nbr: str) -> bool:
        """Check if reviews exist for this course."""
        return self._key(subject, catalog_nbr) in self.data
    
    def save(self, subject: str, catalog_nbr: str, course_data: dict) -> None:
        """Save instructor reviews for a course.
        
        Args:
            subject: Course subject (e.g., "CS")
            catalog_nbr: Course catalog number (e.g., "2100")
            course_data: Dictionary from scrape_all_course_reviews() with structure:
                {
                    "subject": str,
                    "catalog_nbr": str,
                    "course_id": int,
                    "instructors": [
                        {
                            "instructor_name": str,
                            "profile_url": str,
                            "reviews": [{"review_id": str, "text": str}, ...],
                            "review_count": int,
                            ...
                        },
                        ...
                    ]
                }
        """
        self.data[self._key(subject, catalog_nbr)] = course_data
        self._save()
    
    def load(self, subject: str, catalog_nbr: str) -> Optional[dict]:
        """Load all instructor reviews for a course.
        
        Returns:
            Course data dictionary or None if not cached
        """
        return self.data.get(self._key(subject, catalog_nbr))
    
    def get_instructor_reviews(self, subject: str, catalog_nbr: str, instructor_name: str) -> List[dict]:
        """Get reviews for a specific instructor in a course.
        
        Args:
            subject: Course subject
            catalog_nbr: Course catalog number
            instructor_name: Instructor's name (case-insensitive match)
            
        Returns:
            List of review dictionaries or empty list if not found
        """
        course_data = self.load(subject, catalog_nbr)
        if not course_data:
            return []
        
        # Find instructor (case-insensitive)
        instructor_name_lower = instructor_name.lower().strip()
        for instructor in course_data.get("instructors", []):
            if instructor.get("instructor_name", "").lower().strip() == instructor_name_lower:
                return instructor.get("reviews", [])
        
        return []
    
    def get_all_instructors(self, subject: str, catalog_nbr: str) -> List[str]:
        """Get list of all instructor names for a course.
        
        Returns:
            List of instructor names or empty list if course not cached
        """
        course_data = self.load(subject, catalog_nbr)
        if not course_data:
            return []
        
        return [
            instructor.get("instructor_name", "Unknown")
            for instructor in course_data.get("instructors", [])
        ]
    
    def get_instructor_summary(self, subject: str, catalog_nbr: str, instructor_name: str) -> Optional[Dict]:
        """Get summary stats for an instructor's reviews.
        
        Returns:
            Dictionary with review_count and sample reviews, or None if not found
        """
        course_data = self.load(subject, catalog_nbr)
        if not course_data:
            return None
        
        instructor_name_lower = instructor_name.lower().strip()
        for instructor in course_data.get("instructors", []):
            if instructor.get("instructor_name", "").lower().strip() == instructor_name_lower:
                return {
                    "instructor_name": instructor.get("instructor_name"),
                    "review_count": instructor.get("review_count", 0),
                    "profile_url": instructor.get("profile_url"),
                    "sample_reviews": instructor.get("reviews", [])[:5]  # First 5 reviews
                }
        
        return None
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.data = {}
        self._save()
    
    def get_stats(self) -> dict:
        """Get statistics about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        total_courses = len(self.data)
        total_instructors = sum(
            len(course.get("instructors", [])) 
            for course in self.data.values()
        )
        total_reviews = sum(
            sum(
                instructor.get("review_count", 0)
                for instructor in course.get("instructors", [])
            )
            for course in self.data.values()
        )
        
        return {
            "total_courses": total_courses,
            "total_instructors": total_instructors,
            "total_reviews": total_reviews
        }