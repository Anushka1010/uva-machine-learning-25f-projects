"""RateMyProfessor API client for professor reviews and ratings.

Fetches professor reviews from RateMyProfessor with caching support.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.data.stores import RMPStore


# =============================================================================
# Data Classes
# =============================================================================

class Professor:
    """Represents a professor from RateMyProfessor."""
    
    def __init__(
        self,
        ratemyprof_id: int,
        first_name: str,
        last_name: str,
        num_of_ratings: int,
        overall_rating,
    ):
        self.ratemyprof_id = int(ratemyprof_id)
        self.name = f"{first_name} {last_name}".strip()
        self.first_name = first_name
        self.last_name = last_name
        self.num_of_ratings = int(num_of_ratings or 0)

        if self.num_of_ratings < 1:
            self.overall_rating = 0.0
        else:
            try:
                self.overall_rating = float(overall_rating)
            except (TypeError, ValueError):
                self.overall_rating = 0.0


class ProfessorNotFound(Exception):
    """Raised when a professor is not found."""
    
    def __init__(self, search_argument: str, search_parameter: str = "Name"):
        self.search_argument = search_argument
        self.search_parameter = search_parameter

    def __str__(self) -> str:
        return (
            "Professor not found. "
            f"The search argument '{self.search_argument}' did not match "
            f"any professor's {self.search_parameter}."
        )


# =============================================================================
# Low-level RateMyProfessor GraphQL API
# =============================================================================

def _global_id(type_name: str, legacy_id: str | int) -> str:
    """Create a global ID for GraphQL queries."""
    raw = f"{type_name}-{legacy_id}"
    return base64.b64encode(raw.encode("utf-8")).decode("utf-8")


class RateMyProfApi:
    """Low-level RateMyProfessor GraphQL API client."""
    
    GRAPHQL_URL = "https://www.ratemyprofessors.com/graphql"

    def __init__(self, school_id: str = "1277", testing: bool = False):
        self.UniversityId = str(school_id)
        self.testing = testing

        folder = "SchoolID_" + self.UniversityId
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": "https://www.ratemyprofessors.com",
            "Referer": "https://www.ratemyprofessors.com/",
            "Authorization": "Basic dGVzdDp0ZXN0",
        })

        self.school_global_id = _global_id("School", self.UniversityId)
        self.professors: Dict[int, Professor] = self.scrape_professors(testing=self.testing)

    def _graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        resp = self.session.post(
            self.GRAPHQL_URL,
            json={"query": query, "variables": variables},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload and payload["errors"]:
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        return payload.get("data", {})

    def scrape_professors(self, testing: bool = False) -> Dict[int, Professor]:
        """Fetch all professors for the school via GraphQL pagination."""
        query = """
        query TeacherSearch($query: TeacherSearchQuery!, $after: String) {
          newSearch {
            teachers(query: $query, first: 50, after: $after) {
              edges {
                cursor
                node {
                  legacyId
                  firstName
                  lastName
                  avgRating
                  numRatings
                }
              }
              pageInfo { hasNextPage endCursor }
              resultCount
            }
          }
        }
        """

        teacher_query = {
            "text": "",
            "schoolID": self.school_global_id,
            "fallback": True,
            "departmentID": None,
        }

        professors: Dict[int, Professor] = {}
        after: Optional[str] = None
        page_count = 0

        while True:
            data = self._graphql(query, {"query": teacher_query, "after": after})
            teachers_block = (data.get("newSearch") or {}).get("teachers") or {}
            edges = teachers_block.get("edges") or []

            for edge in edges:
                node = (edge or {}).get("node") or {}
                legacy_id = node.get("legacyId")
                if legacy_id is None:
                    continue

                tid = int(legacy_id)
                prof = Professor(
                    tid,
                    node.get("firstName", "") or "",
                    node.get("lastName", "") or "",
                    node.get("numRatings", 0) or 0,
                    node.get("avgRating", 0) or 0,
                )
                professors[tid] = prof

            page_info = teachers_block.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break

            after = page_info.get("endCursor")
            page_count += 1

            if testing and page_count >= 1:
                break

        return professors

    def create_reviews_list(self, tid: int, max_pages: int = 2) -> List[Dict[str, Any]]:
        """Fetch reviews for a professor via GraphQL."""
        rich_query = """
        query TeacherRatings($id: ID!, $after: String) {
        node(id: $id) {
            ... on Teacher {
            ratings(first: 50, after: $after) {
                edges {
                cursor
                node {
                    class
                    comment
                    date
                    difficultyRating
                    clarityRating
                    helpfulRating
                    wouldTakeAgain
                }
                }
                pageInfo { hasNextPage endCursor }
            }
            }
        }
        }
        """

        minimal_query = """
        query TeacherRatings($id: ID!, $after: String) {
        node(id: $id) {
            ... on Teacher {
            ratings(first: 50, after: $after) {
                edges {
                cursor
                node {
                    class
                    comment
                    date
                    difficultyRating
                    wouldTakeAgain
                }
                }
                pageInfo { hasNextPage endCursor }
            }
            }
        }
        }
        """

        teacher_id = _global_id("Teacher", tid)
        reviews: List[Dict[str, Any]] = []
        after: Optional[str] = None
        page_count = 0
        query_to_use = rich_query
        tried_fallback = False

        while True:
            try:
                data = self._graphql(query_to_use, {"id": teacher_id, "after": after})
            except RuntimeError as e:
                if (not tried_fallback) and ("Cannot query field" in str(e)):
                    query_to_use = minimal_query
                    tried_fallback = True
                    continue
                raise

            node = data.get("node") or {}
            ratings = node.get("ratings") or {}
            edges = ratings.get("edges") or []

            for edge in edges:
                r = (edge or {}).get("node") or {}
                reviews.append({
                    "rClass": r.get("class") or "",
                    "rComments": r.get("comment") or "",
                    "rDate": r.get("date"),
                    "rEasy": r.get("difficultyRating"),
                    "rClarity": r.get("clarityRating"),
                    "rHelpful": r.get("helpfulRating"),
                    "rWouldTakeAgain": r.get("wouldTakeAgain"),
                })

            page_info = ratings.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break

            after = page_info.get("endCursor")
            page_count += 1
            if page_count >= max_pages:
                break

        return reviews

    def get_professor_by_last_name(self, last_name: str) -> Professor:
        """Find a professor by last name."""
        last_name = last_name.lower().strip()
        for prof in self.professors.values():
            if prof.last_name.lower() == last_name:
                return prof
        raise ProfessorNotFound(last_name, "Last Name")


# =============================================================================
# High-level RMP API with caching
# =============================================================================

class RMPApi:
    """Client for RateMyProfessor API with caching."""
    
    UVA_SCHOOL_ID = "1277"
    
    def __init__(
        self,
        school_id: str = UVA_SCHOOL_ID,
        testing: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize RMP API client.
        
        Args:
            school_id: RateMyProfessor school ID (default: UVA)
            testing: Use testing mode for API
            cache_dir: Directory for cache files
        """
        self.school_id = str(school_id)
        self.testing = testing
        
        if cache_dir is None:
            cache_dir = os.path.join("app", "data", "cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        base = os.path.join(self.cache_dir, f"rmp_reviews_{self.school_id}")
        self.reviews_store = RMPStore(base)
        
        self._api: Optional[RateMyProfApi] = None
    
    @property
    def api(self) -> RateMyProfApi:
        """Lazy-load the underlying API client."""
        if self._api is None:
            self._api = RateMyProfApi(school_id=self.school_id, testing=self.testing)
        return self._api
    
    @staticmethod
    def normalize_course(raw: Any) -> Optional[str]:
        """Normalize course code to UVA format (e.g., "CS 1110")."""
        if not raw or not isinstance(raw, str):
            return None
        
        s = raw.strip().upper()
        if not s:
            return None
        
        compact = s.replace(" ", "")
        m = re.match(r"^([A-Z]{2,4})(\d{4})$", compact)
        if not m:
            return None
        
        return f"{m.group(1)} {m.group(2)}"
    
    def _cache_path(self, name: str, max_pages: int) -> str:
        """Get cache file path."""
        suffix = "testing" if self.testing else "full"
        return os.path.join(
            self.cache_dir,
            f"rmp_{name}_{self.school_id}_{suffix}_pages{max_pages}.json",
        )
    
    def _get_reviews_with_retries(
        self,
        tid: int,
        max_pages: int,
        retries: int = 4,
        backoff: float = 1.6,
    ) -> List[Dict[str, Any]]:
        """Fetch reviews with retry logic."""
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return self.api.create_reviews_list(tid, max_pages=max_pages)
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
                RuntimeError,
            ) as e:
                last_err = e
                time.sleep(backoff ** attempt)
        raise last_err  # type: ignore
    
    def get_professor_reviews(
        self,
        tid: int,
        max_pages: int = 2,
        uva_only: bool = True,
        force_refresh: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Get reviews for a professor with caching.
        
        Args:
            tid: Professor ID
            max_pages: Max pages to fetch from API
            uva_only: Filter to UVA courses only
            force_refresh: Force re-fetch from API
            limit: Max reviews to return
            
        Returns:
            List of review dictionaries
        """
        if not force_refresh and self.reviews_store.has_reviews(tid):
            return self.reviews_store.get_reviews(tid, limit=limit)
        
        reviews = self._get_reviews_with_retries(tid, max_pages=max_pages)
        
        if uva_only:
            filtered: List[Dict[str, Any]] = []
            for r in reviews:
                norm = self.normalize_course(r.get("rClass"))
                if not norm:
                    continue
                r2 = dict(r)
                r2["rClass"] = norm
                filtered.append(r2)
            reviews = filtered
        
        professor = self.api.professors.get(tid)
        prof_name = professor.name if professor else "UNKNOWN"
        
        self.reviews_store.append_reviews(
            tid=tid,
            professor_name=prof_name,
            reviews=reviews,
        )
        
        return reviews[:limit]
    
    def build_course_professor_map(
        self,
        max_pages: int = 1,
        batch_size: int = 50,
        sleep_between_requests: float = 0.05,
        force_restart: bool = False,
    ) -> Dict[str, List[dict]]:
        """Build a mapping of courses to professors incrementally.
        
        Args:
            max_pages: Max pages per professor
            batch_size: Professors to process per call
            sleep_between_requests: Delay between API calls
            force_restart: Start fresh ignoring progress
            
        Returns:
            Dict mapping course codes to list of professor info
        """
        course_map_path = self._cache_path("course_map", max_pages)
        progress_path = self._cache_path("progress", max_pages)
        
        if os.path.exists(course_map_path) and not force_restart:
            with open(course_map_path, "r", encoding="utf-8") as f:
                course_map: Dict[str, List[dict]] = json.load(f)
        else:
            course_map = {}
        
        if os.path.exists(progress_path) and not force_restart:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
            processed: set[int] = set(progress.get("processed_tids", []))
        else:
            processed = set()
        
        course_map_dd: Dict[str, List[dict]] = defaultdict(list, course_map)
        professor_items = list(self.api.professors.items())
        
        to_process: List[Tuple[int, Any]] = []
        for tid, prof in professor_items:
            if tid in processed:
                continue
            to_process.append((tid, prof))
            if len(to_process) >= batch_size:
                break
        
        if not to_process:
            with open(course_map_path, "w", encoding="utf-8") as f:
                json.dump(dict(course_map_dd), f, ensure_ascii=False)
            return dict(course_map_dd)
        
        for tid, prof in to_process:
            if sleep_between_requests > 0:
                time.sleep(sleep_between_requests)
            
            try:
                reviews = self.get_professor_reviews(
                    tid,
                    max_pages=max_pages,
                    uva_only=True,
                    force_refresh=False,
                )
            except Exception:
                processed.add(tid)
                continue
            
            seen_courses = set()
            for review in reviews:
                course = review.get("rClass")
                if not course or course in seen_courses:
                    continue
                seen_courses.add(course)
                
                already = any(
                    p.get("professor_name") == prof.name
                    for p in course_map_dd[course]
                )
                if already:
                    continue
                
                course_map_dd[course].append({
                    "professor_name": prof.name,
                    "overall_rating": prof.overall_rating,
                    "num_ratings": prof.num_of_ratings,
                })
            
            processed.add(tid)
            
            with open(course_map_path, "w", encoding="utf-8") as f:
                json.dump(dict(course_map_dd), f, ensure_ascii=False)
            
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"processed_tids": sorted(processed)}, f, ensure_ascii=False)
        
        return dict(course_map_dd)
    
    def get_professors_for_course(
        self,
        course_code: str,
        max_pages: int = 1,
        batch_size: int = 50,
    ) -> List[dict]:
        """Get professors who have taught a course.
        
        Args:
            course_code: Course code (e.g., "CS 1110")
            max_pages: Max pages per professor
            batch_size: Professors to process per call
            
        Returns:
            List of professor info dictionaries
        """
        norm = self.normalize_course(course_code)
        if not norm:
            return []
        
        mapping = self.build_course_professor_map(
            max_pages=max_pages,
            batch_size=batch_size,
        )
        return mapping.get(norm, [])


# Backwards compatibility alias
RateMyProfessorService = RMPApi
