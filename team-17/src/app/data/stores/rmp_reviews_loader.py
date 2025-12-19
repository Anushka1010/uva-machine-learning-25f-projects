"""RMP Reviews Loader - loads and searches RateMyProfessor reviews.

Provides functionality to search RMP reviews by professor name and 
generate AI-powered summaries of reviews.
"""

import json
import os
from typing import Optional


class RMPReviewsLoader:
    """Loads and searches RateMyProfessor reviews from cache."""
    
    def __init__(self, cache_path: str = "data/cache/rmp_reviews.json"):
        self.cache_path = cache_path
        self._reviews: list[dict] = []
        self._by_professor: dict[str, list[dict]] = {}
        self._loaded = False
    
    def _load(self):
        """Load reviews from JSONL file."""
        if self._loaded:
            return
        
        if not os.path.exists(self.cache_path):
            self._loaded = True
            return
        
        # The file is JSONL (one JSON per line)
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    review = json.loads(line)
                    self._reviews.append(review)
                    
                    # Index by professor name (lowercase)
                    prof_name = review.get("professor_name", "").lower().strip()
                    if prof_name:
                        if prof_name not in self._by_professor:
                            self._by_professor[prof_name] = []
                        self._by_professor[prof_name].append(review)
                except json.JSONDecodeError:
                    continue
        
        self._loaded = True
    
    def get_reviews_for_professor(
        self,
        professor_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get reviews for a professor by exact name match.
        
        Args:
            professor_name: Professor's full name (e.g., "Mark Sherriff")
            limit: Maximum number of reviews to return
            
        Returns:
            List of review dictionaries
        """
        self._load()
        
        name_lower = professor_name.lower().strip()
        
        # Exact match only
        if name_lower in self._by_professor:
            return self._by_professor[name_lower][:limit]
        
        return []
    
    def get_reviews_for_course_instructor(
        self,
        course_code: str,
        instructor_name: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get reviews for a specific instructor teaching a specific course.
        
        Args:
            course_code: Course code (e.g., "CS 4774" or "CS4774")
            instructor_name: Instructor's name
            limit: Maximum reviews to return
            
        Returns:
            Reviews for this instructor that mention this course
        """
        self._load()
        
        # Normalize course code
        course_normalized = course_code.upper().replace(" ", "")
        
        # Get all reviews for this professor
        all_reviews = self.get_reviews_for_professor(instructor_name, limit=100)
        
        # Filter to those mentioning this course
        matching = []
        for review in all_reviews:
            r_class = review.get("rClass", "").upper().replace(" ", "")
            if r_class == course_normalized:
                matching.append(review)
                if len(matching) >= limit:
                    break
        
        # If no course-specific reviews, return general reviews for this prof
        if not matching:
            return all_reviews[:limit]
        
        return matching
    
    def format_reviews_for_document(
        self,
        reviews: list[dict],
        max_reviews: int = 5,
    ) -> str:
        """Format reviews as text for inclusion in a document.
        
        Args:
            reviews: List of review dictionaries
            max_reviews: Maximum number of reviews to include
            
        Returns:
            Formatted text string
        """
        if not reviews:
            return ""
        
        lines = []
        for i, review in enumerate(reviews[:max_reviews]):
            comment = review.get("rComments", "").strip()
            if not comment:
                continue
            
            # Truncate long comments
            if len(comment) > 300:
                comment = comment[:300] + "..."
            
            rating = review.get("rClarity") or review.get("rHelpful")
            difficulty = review.get("rEasy")
            
            line = f"  - \"{comment}\""
            if rating:
                line += f" (Rating: {rating}/5"
                if difficulty:
                    line += f", Difficulty: {difficulty}/5"
                line += ")"
            
            lines.append(line)
        
        if not lines:
            return ""
        
        return "\n".join(lines)
    
    def summarize_reviews(
        self,
        reviews: list[dict],
        instructor_name: str,
    ) -> dict:
        """Generate summary statistics for reviews.
        
        Args:
            reviews: List of review dictionaries
            instructor_name: Professor's name
            
        Returns:
            Dictionary with summary stats
        """
        if not reviews:
            return {
                "instructor_name": instructor_name,
                "review_count": 0,
                "avg_clarity": None,
                "avg_helpful": None,
                "avg_difficulty": None,
                "would_take_again_pct": None,
                "sample_comments": [],
            }
        
        clarity_scores = []
        helpful_scores = []
        difficulty_scores = []
        would_take_again = []
        comments = []
        
        for review in reviews:
            if review.get("rClarity"):
                clarity_scores.append(review["rClarity"])
            if review.get("rHelpful"):
                helpful_scores.append(review["rHelpful"])
            if review.get("rEasy"):
                difficulty_scores.append(review["rEasy"])
            if review.get("rWouldTakeAgain") is not None:
                would_take_again.append(1 if review["rWouldTakeAgain"] else 0)
            if review.get("rComments"):
                comments.append(review["rComments"][:200])
        
        def avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else None
        
        return {
            "instructor_name": instructor_name,
            "review_count": len(reviews),
            "avg_clarity": avg(clarity_scores),
            "avg_helpful": avg(helpful_scores),
            "avg_difficulty": avg(difficulty_scores),
            "would_take_again_pct": round(100 * avg(would_take_again)) if would_take_again else None,
            "sample_comments": comments[:3],
        }


# Singleton instance
_loader: Optional[RMPReviewsLoader] = None


def get_rmp_loader() -> RMPReviewsLoader:
    """Get the singleton RMP reviews loader."""
    global _loader
    if _loader is None:
        _loader = RMPReviewsLoader()
    return _loader

