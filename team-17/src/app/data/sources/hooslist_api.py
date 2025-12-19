"""Hooslist API client for course descriptions and prerequisites.

Hooslist provides detailed course descriptions and prerequisite information
that isn't available in the main SIS API.
"""

import httpx
from typing import Optional

from app.data.stores import HooslistStore


class HooslistApi:
    """Client for the Hooslist course description API with caching."""
    
    BASE_URL = "https://hooslist.virginia.edu/ClassSchedule/_GetCourseDescription"
    
    def __init__(self, timeout: float = 10.0, cache_dir: str = "data/cache"):
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        self.store = HooslistStore(cache_dir)
    
    def get_description(
        self, 
        subject: str, 
        catalog_number: str,
        use_cache: bool = True,
    ) -> dict:
        """Fetch course description and prerequisites with caching.
        
        Args:
            subject: Subject code (e.g., "CS")
            catalog_number: Course number (e.g., "4774")
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with 'description' and 'prerequisites' keys
        """
        # Check cache first
        if use_cache and self.store.has(subject, catalog_number):
            cached = self.store.load(subject, catalog_number)
            if cached:
                return {
                    "description": cached.get("description", ""),
                    "prerequisites": cached.get("prerequisites", ""),
                }
        
        # Fetch from API
        try:
            response = self.client.get(
                self.BASE_URL,
                params={"subject": subject.upper(), "courseNum": catalog_number},
            )
            response.raise_for_status()
            
            text = response.text.strip()
            result = self._parse_response(text)
            
            # Save to cache
            self.store.save(
                subject, 
                catalog_number, 
                result["description"], 
                result["prerequisites"]
            )
            
            return result
            
        except Exception:
            return {"description": "", "prerequisites": ""}
    
    def _parse_response(self, text: str) -> dict:
        """Parse the API response text into description and prerequisites.
        
        Args:
            text: Raw response text
            
        Returns:
            Dictionary with 'description' and 'prerequisites' keys
        """
        description = ""
        prerequisites = ""
        
        if "Prerequisites:" in text:
            parts = text.split("Prerequisites:", 1)
            description = parts[0].strip()
            prerequisites = parts[1].strip()
        elif "Prerequisite:" in text:
            parts = text.split("Prerequisite:", 1)
            description = parts[0].strip()
            prerequisites = parts[1].strip()
        else:
            description = text
        
        return {
            "description": description,
            "prerequisites": prerequisites,
        }
    
    def fetch_batch(
        self,
        courses: list[tuple[str, str]],
        on_progress: callable = None,
        use_cache: bool = True,
    ) -> dict:
        """Fetch descriptions for multiple courses with caching.
        
        Args:
            courses: List of (subject, catalog_number) tuples
            on_progress: Optional callback(index, total, subject, catalog_number)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping "SUBJECT_CATALOG" to description info
        """
        descriptions = {}
        total = len(courses)
        cache_hits = 0
        
        for idx, (subject, catalog_number) in enumerate(courses, 1):
            key = f"{subject}_{catalog_number}"
            
            # Check if already cached
            was_cached = use_cache and self.store.has(subject, catalog_number)
            if was_cached:
                cache_hits += 1
            
            descriptions[key] = self.get_description(subject, catalog_number, use_cache)
            
            if on_progress:
                on_progress(idx, total, subject, catalog_number)
        
        return descriptions
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
