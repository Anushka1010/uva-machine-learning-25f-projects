import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": "UVA-Course-Advising-Project/1.0 (academic use)"
}

def fetch_page(url: str) -> BeautifulSoup:
    """Fetch a page and return BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def extract_review_text(review_div):
    """Extract the review comment text from a review card div."""
    # The review text is in div with class "review-text-full"
    review_text_div = review_div.find("div", class_="review-text-full")
    
    if not review_text_div:
        return None
    
    # Get all paragraph tags within the review text
    paragraphs = review_text_div.find_all("p")
    
    if not paragraphs:
        # Fallback: get all text from the div
        text = review_text_div.get_text(strip=True)
        return text if text and len(text) > 0 else None
    
    # Join all paragraphs with newlines
    text_parts = []
    for p in paragraphs:
        p_text = p.get_text(strip=True)
        if p_text:
            text_parts.append(p_text)
    
    full_text = "\n".join(text_parts)
    return full_text if full_text else None

def scrape_instructor_reviews(course_id: int, instructor_id: int, max_pages: int = 10):
    """Scrape all reviews for a specific instructor teaching a specific course.
    
    Args:
        course_id: TheCourseForum course ID
        instructor_id: TheCourseForum instructor ID
        max_pages: Maximum number of pages to scrape (default 10)
        
    Returns:
        List of review dictionaries with text content
    """
    base_url = f"https://thecourseforum.com/course/{course_id}/{instructor_id}/"
    reviews = []
    page = 1
    
    while page <= max_pages:
        # Construct URL with page parameter
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}?page={page}#reviews"
        
        try:
            soup = fetch_page(url)
            
            # Find all review cards
            review_cards = soup.find_all("div", class_="review")
            
            # If no reviews found, we've reached the end
            if not review_cards:
                break
            
            page_reviews = 0
            for card in review_cards:
                # Extract review ID from the card's id attribute
                review_id = card.get("id", "").replace("review", "")
                
                # Extract review text
                review_text = extract_review_text(card)
                
                if review_text:
                    reviews.append({
                        "review_id": review_id,
                        "text": review_text,
                    })
                    page_reviews += 1
            
            # If we got no reviews on this page, stop
            if page_reviews == 0:
                break
            
            # Check if there's a next page
            # Look for pagination controls
            pagination = soup.find("ul", class_="pagination")
            if not pagination:
                break
            
            # Check if there's a "next" button or if current page is last
            next_button = pagination.find("a", string="Next") or pagination.find("a", {"aria-label": "Next"})
            if not next_button or "disabled" in next_button.get("class", []):
                break
            
            page += 1
            
            # Be polite - add a small delay between requests
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Warning: Error fetching page {page} for course {course_id}, instructor {instructor_id}: {e}")
            break
    
    return reviews

def scrape_reviews_for_instructor_data(instructor_data: dict, course_id: int):
    """Scrape reviews for an instructor given their data from the course page.
    
    Args:
        instructor_data: Dictionary with instructor info (from your existing scraper)
        course_id: TheCourseForum course ID
        
    Returns:
        Dictionary with instructor info and reviews
    """
    # Extract instructor ID from profile URL
    # URL format: https://thecourseforum.com/course/14940/4710/
    profile_url = instructor_data.get("profile_url", "")
    if not profile_url:
        return {
            **instructor_data,
            "reviews": []
        }
    
    # Parse instructor ID from URL
    import re
    match = re.search(r'/course/\d+/(\d+)/', profile_url)
    if not match:
        return {
            **instructor_data,
            "reviews": []
        }
    
    instructor_id = int(match.group(1))
    
    # Scrape reviews
    reviews = scrape_instructor_reviews(course_id, instructor_id)
    
    return {
        **instructor_data,
        "reviews": reviews,
        "review_count": len(reviews)
    }

def scrape_all_course_reviews(subject: str, catalog_nbr: str, course_id: int, instructors: list):
    """Scrape reviews for all instructors of a course.
    
    Args:
        subject: Course subject (e.g., "CS")
        catalog_nbr: Course catalog number (e.g., "2100")
        course_id: TheCourseForum course ID
        instructors: List of instructor dictionaries from scrape_course()
        
    Returns:
        Dictionary formatted for storage
    """
    results = {
        "subject": subject,
        "catalog_nbr": catalog_nbr,
        "course_id": course_id,
        "instructors": []
    }
    
    for instructor in instructors:
        print(f"  Scraping reviews for {instructor.get('instructor_name')}...", end=" ")
        
        instructor_with_reviews = scrape_reviews_for_instructor_data(instructor, course_id)
        results["instructors"].append(instructor_with_reviews)
        
        print(f"{instructor_with_reviews['review_count']} reviews")
    
    return results