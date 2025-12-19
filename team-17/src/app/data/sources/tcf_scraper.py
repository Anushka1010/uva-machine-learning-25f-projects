"""TheCourseForum scraper for instructor reviews and ratings.

Scrapes instructor ratings, difficulty scores, and GPA data from TheCourseForum.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional

from app.data.stores import TCFStore


class TCFScraper:
    """Scraper for TheCourseForum instructor data with caching."""
    
    BASE_URL = "https://thecourseforum.com/course"
    
    HEADERS = {
        "User-Agent": "UVA-Course-Advising-Project/1.0 (academic use)"
    }
    
    def __init__(self, timeout: float = 10.0, cache_dir: str = "data/cache"):
        self.timeout = timeout
        self.store = TCFStore(cache_dir)
    
    def get_course_reviews(
        self, 
        subject: str, 
        catalog_number: str,
        use_cache: bool = True,
    ) -> list[dict]:
        """Get all instructor reviews for a course with caching.
        
        Args:
            subject: Subject code (e.g., "CS")
            catalog_number: Course number (e.g., "4774")
            use_cache: Whether to use cached data if available
            
        Returns:
            List of instructor review dictionaries
        """
        # Check cache first
        if use_cache and self.store.has(subject, catalog_number):
            return self.store.load(subject, catalog_number)
        
        # Fetch from web
        url = f"{self.BASE_URL}/{subject}/{catalog_number}/All"
        reviews = self._scrape_course(url)
        
        # Save to cache
        self.store.save(subject, catalog_number, reviews)
        
        return reviews
    
    def _scrape_course(self, url: str) -> list[dict]:
        """Scrape course reviews from a TCF URL.
        
        Args:
            url: Full URL to the course page
            
        Returns:
            List of instructor review dictionaries
        """
        try:
            soup = self._fetch_page(url)
            instructors = []
            
            for li in self._get_instructor_cards(soup):
                data = self._parse_instructor(li)
                if data["instructor_name"]:
                    instructors.append(data)
            
            return instructors
        except Exception:
            return []
    
    def _fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object
        """
        resp = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    
    def _get_instructor_cards(self, soup: BeautifulSoup) -> list:
        """Extract instructor card elements from page.
        
        Args:
            soup: Parsed page
            
        Returns:
            List of instructor list item elements
        """
        ul = soup.find("ul", class_="instructor-list")
        if not ul:
            return []
        return ul.find_all("li", class_="instructor")
    
    def _parse_instructor(self, li) -> dict:
        """Parse instructor data from a list item element.
        
        Args:
            li: BeautifulSoup list item element
            
        Returns:
            Dictionary with instructor data
        """
        def safe_text(el):
            return el.get_text(strip=True) if el else None
        
        # Name
        name = safe_text(li.find("h3", id="title"))
        
        # Profile link
        a_tag = li.find("a", href=True)
        profile_url = f"https://thecourseforum.com{a_tag['href']}" if a_tag else None
        
        # Stats
        rating = safe_text(li.find("p", id="rating"))
        difficulty = safe_text(li.find("p", id="difficulty"))
        gpa = safe_text(li.find("p", id="gpa"))
        
        return {
            "instructor_name": name,
            "profile_url": profile_url,
            "rating": rating if rating else None,
            "difficulty": difficulty if difficulty else None,
            "gpa": gpa if gpa else None,
        }
    
    def fetch_batch(
        self,
        courses: list[tuple[str, str]],
        on_progress: callable = None,
        use_cache: bool = True,
    ) -> dict:
        """Fetch reviews for multiple courses with caching.
        
        Args:
            courses: List of (subject, catalog_number) tuples
            on_progress: Optional callback(index, total, subject, catalog_number, success)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping "SUBJECT_CATALOG" to reviews list
        """
        reviews_map = {}
        total = len(courses)
        
        for idx, (subject, catalog_number) in enumerate(courses, 1):
            key = f"{subject}_{catalog_number}"
            reviews = self.get_course_reviews(subject, catalog_number, use_cache)
            reviews_map[key] = reviews
            
            if on_progress:
                on_progress(idx, total, subject, catalog_number, len(reviews) > 0)
        
        return reviews_map


# Backwards compatibility - expose as module-level function
BASE_URL = TCFScraper.BASE_URL

def scrape_course(course_url: str) -> list[dict]:
    """Legacy function for backwards compatibility.
    
    Args:
        course_url: URL like "https://thecourseforum.com/course/CS/4774/"
        
    Returns:
        List of instructor review dictionaries
    """
    scraper = TCFScraper()
    return scraper._scrape_course(course_url)
