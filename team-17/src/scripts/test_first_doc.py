"""Test script to save the first indexed document to a text file."""

import json
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.sources import SISApi, HooslistApi, TCFScraper
from app.data.document_builder import DocumentBuilder


def main():
    print("Building first document for inspection...\n")
    
    # Initialize
    sis_api = SISApi()
    hooslist_api = HooslistApi()
    tcf_scraper = TCFScraper()
    doc_builder = DocumentBuilder()
    
    # Fetch first page of CS courses
    print("Fetching CS courses from SIS...")
    response = sis_api.search(subject="CS", term="1262", page=1)
    courses = sis_api.get_classes_list(response)
    
    if not courses:
        print("No courses found!")
        return
    
    course = courses[0]
    subject = course.get("subject", "")
    catalog_nbr = course.get("catalog_nbr", "")
    print(f"First course: {subject} {catalog_nbr} - {course.get('descr', '')}")
    
    # Fetch Hooslist description
    print(f"Fetching Hooslist description...")
    hooslist_info = hooslist_api.get_description(subject, catalog_nbr)
    
    # Fetch TCF reviews
    print(f"Fetching TCF reviews...")
    try:
        all_reviews = tcf_scraper.get_course_reviews(subject, catalog_nbr)
    except Exception as e:
        print(f"  TCF error: {e}")
        all_reviews = []
    
    # Match reviews to instructors
    matched_reviews = doc_builder.match_reviews_to_instructors(course, all_reviews)
    
    # Build document
    print("Building document...")
    doc_text = doc_builder.build_document(course, hooslist_info, matched_reviews)
    metadata = doc_builder.build_metadata(course, hooslist_info, matched_reviews)
    
    # Save to file
    output_dir = Path("data/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc_path = output_dir / "first_doc.txt"
    meta_path = output_dir / "first_doc_meta.json"
    raw_path = output_dir / "first_doc_raw.json"
    
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc_text)
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "course": course,
            "hooslist_info": hooslist_info,
            "all_reviews": all_reviews,
            "matched_reviews": matched_reviews,
        }, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Saved to:")
    print(f"  Document: {doc_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Raw data: {raw_path}")
    print(f"{'='*60}")
    
    print(f"\n--- DOCUMENT CONTENT ---\n")
    print(doc_text)
    print(f"\n--- METADATA ---\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

