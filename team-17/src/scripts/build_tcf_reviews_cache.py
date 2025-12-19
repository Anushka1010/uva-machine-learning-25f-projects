"""Script to scrape all instructor reviews and store them in JSON cache.

Run from project root: python src/scripts/build_tcf_reviews_cache.py
"""

import json
import time
from app.data.sources.tcf_review_scraper import scrape_all_course_reviews
from app.data.stores.tcf_instructor_reviews_store import TCFInstructorReviewsStore

def parse_course_id_from_url(profile_url: str) -> int:
    """Extract course_id from profile URL.
    
    Args:
        profile_url: URL like "https://thecourseforum.com/course/594/13208/"
        
    Returns:
        course_id (e.g., 594)
    """
    import re
    match = re.search(r'/course/(\d+)/\d+/', profile_url)
    if match:
        return int(match.group(1))
    return None


def load_existing_course_cache(cache_path: str = "data/cache/tcf_reviews.json"):
    """Load the existing course cache with instructor data.
    
    Returns:
        Dictionary with course keys and instructor lists
    """
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scrape_and_cache_all_reviews(store, existing_cache, delay=1.0):
    """Scrape reviews for all courses in existing cache.
    
    Args:
        store: TCFInstructorReviewStore instance
        existing_cache: Dictionary from tcf_reviews.json
        delay: Delay in seconds between course scrapes (be polite to server)
    """
    total = len(existing_cache)
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, (course_key, instructors) in enumerate(existing_cache.items(), 1):
        # Parse subject and catalog number from key (e.g., "CS_1110")
        parts = course_key.split("_")
        if len(parts) != 2:
            print(f"[{idx}/{total}] Invalid course key format: {course_key}, skipping")
            errors += 1
            continue
        
        subject, catalog_nbr = parts
        
        # Skip if already cached
        if store.has(subject, catalog_nbr):
            print(f"[{idx}/{total}] {subject} {catalog_nbr}: Already cached, skipping")
            skipped += 1
            continue
        
        # Get course_id from first instructor's profile URL
        if not instructors or len(instructors) == 0:
            print(f"[{idx}/{total}] {subject} {catalog_nbr}: No instructors found, skipping")
            errors += 1
            continue
        
        course_id = parse_course_id_from_url(instructors[0].get("profile_url", ""))
        if not course_id:
            print(f"[{idx}/{total}] {subject} {catalog_nbr}: Could not parse course_id, skipping")
            errors += 1
            continue
        
        print(f"\n[{idx}/{total}] Scraping {subject} {catalog_nbr} (course_id: {course_id})")
        print(f"  Found {len(instructors)} instructors")
        
        try:
            # Scrape all reviews for this course
            course_data = scrape_all_course_reviews(
                subject=subject,
                catalog_nbr=catalog_nbr,
                course_id=course_id,
                instructors=instructors
            )
            
            # Save to cache
            store.save(subject, catalog_nbr, course_data)
            
            # Print summary
            total_reviews = sum(
                inst.get("review_count", 0) 
                for inst in course_data.get("instructors", [])
            )
            print(f"  ✓ Saved {len(course_data['instructors'])} instructors, {total_reviews} total reviews")
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error scraping {subject} {catalog_nbr}: {e}")
            errors += 1
            continue
        
        # Be polite - delay between courses
        if idx < total:
            time.sleep(delay)
    
    # Print final statistics
    print("\n" + "="*60)
    print("Scraping complete!")
    print(f"Processed: {processed} courses")
    print(f"Skipped (already cached): {skipped} courses")
    print(f"Errors: {errors} courses")
    print("-"*60)
    
    stats = store.get_stats()
    print(f"Total courses in cache: {stats['total_courses']}")
    print(f"Total instructors: {stats['total_instructors']}")
    print(f"Total reviews: {stats['total_reviews']}")
    print("="*60)


def main():
    """Main function to orchestrate the scraping process."""
    
    # Initialize store
    store = TCFInstructorReviewsStore()
    
    # Load existing course cache
    cache_path = "data/cache/tcf_reviews.json"
    
    try:
        print(f"Loading existing course cache from {cache_path}...")
        existing_cache = load_existing_course_cache(cache_path)
        print(f"Found {len(existing_cache)} courses in existing cache\n")
    except FileNotFoundError:
        print(f"Error: Could not find cache file at {cache_path}")
        print("Please make sure you've run your course scraper first!")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON from {cache_path}: {e}")
        return
    
    # Optional: Preview what will be scraped
    print("Courses to scrape:")
    for course_key in list(existing_cache.keys())[:5]:
        subject, catalog_nbr = course_key.split("_")
        status = "✓ cached" if store.has(subject, catalog_nbr) else "○ needs scraping"
        print(f"  {status} {subject} {catalog_nbr}")
    
    if len(existing_cache) > 5:
        print(f"  ... and {len(existing_cache) - 5} more")
    
    # Ask for confirmation
    print("\n" + "="*60)
    response = input("Start scraping? This may take a while. (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run the scraper
    print("\nStarting scrape...\n")
    scrape_and_cache_all_reviews(store, existing_cache, delay=1.0)


if __name__ == "__main__":
    main()