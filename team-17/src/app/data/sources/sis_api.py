"""UVA SIS (Student Information System) API client.

Fetches course catalog, sections, and enrollment data from the official UVA SIS API.
API Documentation: https://s23.cs3240.org/sis-api.html

Term codes:
    Format: 1 + [2-digit year] + [semester code]
    Semester codes: 2 = Spring, 6 = Summer, 8 = Fall
    Examples: 1262 = Spring 2026, 1252 = Spring 2025, 1248 = Fall 2024
"""

import httpx
from typing import Optional

from app.data.stores import SISStore


class SISApi:
    """Client for the UVA SIS course search API with caching."""
    
    BASE_URL = "https://sisuva.admin.virginia.edu/psc/ihprd/UVSS/SA/s/WEBLIB_HCX_CM.H_CLASS_SEARCH.FieldFormula.IScript_ClassSearch"
    OPTIONS_URL = "https://sisuva.admin.virginia.edu/psc/ihprd/UVSS/SA/s/WEBLIB_HCX_CM.H_CLASS_SEARCH.FieldFormula.IScript_ClassSearchOptions"
    
    def __init__(self, timeout: float = 30.0, cache_dir: str = "data/cache"):
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        self.store = SISStore(cache_dir)
    
    def search(
        self,
        subject: Optional[str] = None,
        catalog_number: Optional[str] = None,
        keyword: Optional[str] = None,
        instructor: Optional[str] = None,
        term: str = "1262",
        page: int = 1,
    ) -> dict:
        """Search for courses in SIS.
        
        Args:
            subject: Subject code (e.g., "CS", "MATH")
            catalog_number: Course number (e.g., "4774")
            keyword: Keyword to search in course titles
            instructor: Instructor last name
            term: Academic term code (default: Spring 2026)
            page: Results page number
            
        Returns:
            Raw API response dictionary
        """
        params = {
            "institution": "UVA01",
            "term": term,
            "page": page,
        }
        
        if subject:
            params["subject"] = subject.upper()
        if catalog_number:
            params["catalog_nbr"] = catalog_number
        if keyword:
            params["keyword"] = keyword
        if instructor:
            params["instructor_name"] = instructor
        
        response = self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_classes_list(self, api_response) -> list:
        """Extract classes list from API response.
        
        The API can return either a list directly or a dict with 'classes' key.
        
        Args:
            api_response: Raw API response
            
        Returns:
            List of class dictionaries
        """
        if isinstance(api_response, list):
            return api_response
        elif isinstance(api_response, dict):
            return api_response.get("classes", [])
        return []
    
    def fetch_all_courses(
        self,
        subjects: list[str],
        term: str = "1262",
        on_progress: callable = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """Fetch all courses for given subjects with caching.
        
        Args:
            subjects: List of subject codes to fetch
            term: Academic term code
            on_progress: Optional callback(subject, page, count) for progress updates
            use_cache: Whether to use cached data if available
            
        Returns:
            List of all course dictionaries
        """
        # Check cache first
        if use_cache and self.store.has(term):
            courses = self.store.load(term)
            if on_progress:
                on_progress("CACHE", 0, len(courses))
            return courses
        
        # Fetch from API
        all_courses = []
        
        for subject in subjects:
            page = 1
            while True:
                try:
                    response = self.search(subject=subject, term=term, page=page)
                    classes = self.get_classes_list(response)
                    
                    if not classes:
                        break
                    
                    all_courses.extend(classes)
                    
                    if on_progress:
                        on_progress(subject, page, len(classes))
                    
                    page += 1
                    
                    # Safety limit
                    if page > 20:
                        break
                        
                except Exception as e:
                    if on_progress:
                        on_progress(subject, page, 0, error=str(e))
                    break
        
        # Save to cache
        if all_courses:
            self.store.save(term, all_courses)
        
        return all_courses
    
    def extract_instructor(self, course: dict) -> str:
        """Extract primary instructor name from course data."""
        instructors = course.get("instructors", [])
        if instructors:
            return instructors[0].get("name", "Staff")
        return "Staff"
    
    def format_time(self, time_str: str) -> str:
        """Format SIS time string to readable format.
        
        Converts "09.00.00.000000" to "9am", "14.30.00.000000" to "2:30pm"
        
        Args:
            time_str: Time string in format "HH.MM.SS.ffffff"
            
        Returns:
            Formatted time string like "9am" or "2:30pm"
        """
        if not time_str:
            return ""
        
        try:
            parts = time_str.split(".")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            
            period = "am" if hour < 12 else "pm"
            if hour == 0:
                hour = 12
            elif hour > 12:
                hour -= 12
            
            if minute == 0:
                return f"{hour}{period}"
            else:
                return f"{hour}:{minute:02d}{period}"
        except (ValueError, IndexError):
            return time_str
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

