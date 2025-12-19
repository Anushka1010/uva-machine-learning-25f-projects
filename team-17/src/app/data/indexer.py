"""Course indexer - orchestrates data fetching and vector indexing.

This is the main entry point for loading and indexing course data from all sources:
- SIS API (course catalog, sections, enrollment)
- Hooslist (descriptions, prerequisites)
- TheCourseForum (instructor reviews, ratings)

The indexer fetches data, builds documents, and stores them in the vector database.
"""

import json
import time
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.data.sources import SISApi, HooslistApi, TCFScraper
from app.data.document_builder import DocumentBuilder
from app.data.vector_store import VectorStore


class CourseIndexer:
    """Orchestrates course data fetching and vector indexing."""
    
    # Default subjects to index
    DEFAULT_SUBJECTS = ["CS", "DS", "STAT", "MATH", "STS"]
    
    def __init__(self):
        self.settings = get_settings()
        
        # Data sources
        self.sis_api = SISApi()
        self.hooslist_api = HooslistApi()
        self.tcf_scraper = TCFScraper()
        
        # Document builder and vector store
        self.doc_builder = DocumentBuilder()
        self.vector_store = VectorStore()
        
        # Cache directory
        self.cache_dir = Path(self.settings.data_dir) / "courses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def index_courses(
        self,
        term: str = "1262",
        subjects: Optional[list[str]] = None,
        force_refresh: bool = False,
    ) -> int:
        """Index courses from all data sources.
        
        Args:
            term: Academic term code (default: Spring 2026)
            subjects: List of subjects to index (default: CS, DS, STAT, MATH, STS)
            force_refresh: Force re-fetch from APIs even if cached
            
        Returns:
            Number of documents indexed
        """
        subjects = subjects or self.DEFAULT_SUBJECTS
        
        # Step 1: Fetch courses from SIS (with caching)
        courses = self._fetch_sis_courses(term, subjects, force_refresh)
        
        # Step 2: Fetch descriptions from Hooslist
        hooslist_data = self._fetch_hooslist_descriptions(courses)
        
        # Step 3: Fetch reviews from TheCourseForum
        tcf_data = self._fetch_tcf_reviews(courses)
        
        # Step 4: Build documents and index
        return self._index_documents(courses, hooslist_data, tcf_data)
    
    def _fetch_sis_courses(
        self,
        term: str,
        subjects: list[str],
        force_refresh: bool,
    ) -> list[dict]:
        """Fetch courses from SIS API (with caching via SISStore).
        
        Args:
            term: Academic term code
            subjects: Subjects to fetch
            force_refresh: Force re-fetch
            
        Returns:
            List of course dictionaries
        """
        # Check cache first (using SISStore)
        if not force_refresh and self.sis_api.store.has(term):
            print(f"\n{'='*50}")
            print("SIS COURSES")
            print(f"{'='*50}")
            courses = self.sis_api.store.load(term)
            print(f"[CACHE] Loaded {len(courses)} courses from cache")
            return courses
        
        # Fetch from API
        print(f"\n{'='*50}")
        print("SIS COURSES")
        print(f"{'='*50}")
        print(f"Subjects: {', '.join(subjects)}")
        print(f"Term: {term}")
        print()
        
        all_courses = []
        total_subjects = len(subjects)
        start_time = time.time()
        
        for idx, subject in enumerate(subjects, 1):
            subject_start = time.time()
            subject_courses = 0
            page = 1
            
            print(f"[{idx}/{total_subjects}] Fetching {subject}...", end=" ", flush=True)
            
            while True:
                try:
                    response = self.sis_api.search(subject=subject, term=term, page=page)
                    classes = self.sis_api.get_classes_list(response)
                    
                    if not classes:
                        break
                    
                    all_courses.extend(classes)
                    subject_courses += len(classes)
                    page += 1
                    
                    if page > 20:  # Safety limit
                        break
                        
                except Exception as e:
                    print(f"Error: {e}")
                    break
            
            elapsed = time.time() - subject_start
            print(f"{subject_courses} sections ({elapsed:.1f}s)")
        
        total_time = time.time() - start_time
        print(f"\nFetched {len(all_courses)} total sections in {total_time:.1f}s")
        
        # Cache the results (using SISStore)
        self.sis_api.store.save(term, all_courses)
        print(f"[CACHE] Saved to cache")
        
        return all_courses
    
    def _fetch_hooslist_descriptions(self, courses: list[dict]) -> dict:
        """Fetch descriptions from Hooslist for unique courses.
        
        Args:
            courses: List of course dictionaries
            
        Returns:
            Dictionary mapping "SUBJECT_CATALOG" to description info
        """
        # Get unique courses
        unique_courses = self._get_unique_courses(courses)
        total = len(unique_courses)
        
        # Count cached vs uncached
        cached_count = sum(
            1 for info in unique_courses.values()
            if self.hooslist_api.store.has(info["subject"], info["catalog_nbr"])
        )
        to_fetch = total - cached_count
        
        print(f"\n{'='*50}")
        print("HOOSLIST DESCRIPTIONS")
        print(f"{'='*50}")
        print(f"Total courses: {total}")
        print(f"  - Cached: {cached_count}")
        print(f"  - To fetch: {to_fetch}")
        
        if to_fetch == 0:
            print("\n[CACHE] All descriptions loaded from cache!")
        else:
            print()
        
        descriptions = {}
        start_time = time.time()
        fetched = 0
        
        for idx, (key, info) in enumerate(unique_courses.items(), 1):
            subject = info["subject"]
            catalog_nbr = info["catalog_nbr"]
            
            was_cached = self.hooslist_api.store.has(subject, catalog_nbr)
            descriptions[key] = self.hooslist_api.get_description(subject, catalog_nbr)
            
            if not was_cached:
                fetched += 1
                if fetched % 10 == 0 or fetched == to_fetch:
                    elapsed = time.time() - start_time
                    rate = fetched / elapsed if elapsed > 0 else 0
                    eta = (to_fetch - fetched) / rate if rate > 0 else 0
                    print(f"  Fetching [{fetched}/{to_fetch}] ({elapsed:.1f}s, ETA: {eta:.0f}s)", flush=True)
        
        # Stats
        with_desc = sum(1 for d in descriptions.values() if d.get("description"))
        with_prereq = sum(1 for d in descriptions.values() if d.get("prerequisites"))
        
        total_time = time.time() - start_time
        print(f"\nLoaded {len(descriptions)} descriptions in {total_time:.1f}s")
        print(f"  - With descriptions: {with_desc}")
        print(f"  - With prerequisites: {with_prereq}")
        
        return descriptions
    
    def _fetch_tcf_reviews(self, courses: list[dict]) -> dict:
        """Fetch reviews from TheCourseForum for unique courses.
        
        Args:
            courses: List of course dictionaries
            
        Returns:
            Dictionary mapping "SUBJECT_CATALOG" to reviews list
        """
        unique_courses = self._get_unique_courses(courses)
        total = len(unique_courses)
        
        # Count cached vs uncached
        cached_count = sum(
            1 for info in unique_courses.values()
            if self.tcf_scraper.store.has(info["subject"], info["catalog_nbr"])
        )
        to_fetch = total - cached_count
        
        print(f"\n{'='*50}")
        print("THECOURSEFORUM REVIEWS")
        print(f"{'='*50}")
        print(f"Total courses: {total}")
        print(f"  - Cached: {cached_count}")
        print(f"  - To fetch: {to_fetch}")
        
        if to_fetch == 0:
            print("\n[CACHE] All reviews loaded from cache!")
        else:
            print()
        
        reviews_map = {}
        start_time = time.time()
        fetched = 0
        successful = 0
        failed = 0
        
        for idx, (key, info) in enumerate(unique_courses.items(), 1):
            subject = info["subject"]
            catalog_nbr = info["catalog_nbr"]
            
            was_cached = self.tcf_scraper.store.has(subject, catalog_nbr)
            
            try:
                reviews = self.tcf_scraper.get_course_reviews(subject, catalog_nbr)
                reviews_map[key] = reviews
                if reviews:
                    successful += 1
                else:
                    failed += 1
            except Exception:
                reviews_map[key] = []
                failed += 1
            
            if not was_cached:
                fetched += 1
                if fetched % 5 == 0 or fetched == to_fetch:
                    elapsed = time.time() - start_time
                    rate = fetched / elapsed if elapsed > 0 else 0
                    eta = (to_fetch - fetched) / rate if rate > 0 else 0
                    print(f"  Scraping [{fetched}/{to_fetch}] ({elapsed:.1f}s, ETA: {eta:.0f}s)", flush=True)
        
        total_time = time.time() - start_time
        print(f"\nLoaded {len(reviews_map)} courses in {total_time:.1f}s")
        print(f"  - With reviews: {successful}")
        print(f"  - No reviews: {failed}")
        
        return reviews_map
    
    def _index_documents(
        self,
        courses: list[dict],
        hooslist_data: dict,
        tcf_data: dict,
    ) -> int:
        """Build documents and index into vector store.
        
        Args:
            courses: List of course dictionaries
            hooslist_data: Hooslist descriptions by course key
            tcf_data: TCF reviews by course key
            
        Returns:
            Number of documents indexed
        """
        print(f"\n{'='*50}")
        print("INDEXING INTO VECTOR DATABASE")
        print(f"{'='*50}")
        
        # Clear existing index
        print("Clearing existing index...", end=" ", flush=True)
        self.vector_store.clear()
        print("Done")
        
        # Build documents
        print("Preparing documents...", end=" ", flush=True)
        documents = []
        metadatas = []
        ids = []
        seen = set()
        
        for course in courses:
            course_id = f"{course.get('subject', '')}_{course.get('catalog_nbr', '')}_{course.get('class_nbr', '')}"
            
            if course_id in seen:
                continue
            seen.add(course_id)
            
            # Get Hooslist data for this course
            key = f"{course.get('subject', '')}_{course.get('catalog_nbr', '')}"
            hooslist_info = hooslist_data.get(key, {})
            
            # Build document (TCF + RMP reviews are loaded internally by document builder)
            doc_text = self.doc_builder.build_document(course, hooslist_info)
            metadata = self.doc_builder.build_metadata(course, hooslist_info)
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(course_id)
        
        print(f"{len(documents)} unique sections")
        
        # Batch index
        self._batch_index(documents, metadatas, ids)
        
        return len(documents)
    
    def _batch_index(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
        batch_size: int = 50,
    ):
        """Index documents in batches with progress.
        
        Args:
            documents: Document texts
            metadatas: Document metadata
            ids: Document IDs
            batch_size: Number of documents per batch
        """
        total_batches = (len(documents) + batch_size - 1) // batch_size
        start_time = time.time()
        
        print(f"\nIndexing {len(documents)} documents ({total_batches} batches):")
        print(f"  [Estimated time: ~{total_batches * 2}-{total_batches * 4} seconds]")
        print()
        
        for batch_num, i in enumerate(range(0, len(documents), batch_size), 1):
            batch_start = time.time()
            
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            progress = (batch_num / total_batches) * 100
            docs_done = min(i + batch_size, len(documents))
            
            print(f"  Batch {batch_num}/{total_batches} [{progress:5.1f}%] - {docs_done}/{len(documents)} docs...", end=" ", flush=True)
            
            self.vector_store.add_documents(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            
            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            remaining = total_batches - batch_num
            eta = (elapsed / batch_num) * remaining if batch_num > 0 else 0
            
            print(f"Done ({batch_time:.1f}s) | ETA: {eta:.0f}s")
        
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print("INDEXING COMPLETE!")
        print(f"  - Indexed: {len(documents)} documents")
        print(f"  - Time: {total_time:.1f} seconds")
        print(f"{'='*50}\n")
    
    def _get_unique_courses(self, courses: list[dict]) -> dict:
        """Get unique courses by subject + catalog number.
        
        Args:
            courses: List of course dictionaries
            
        Returns:
            Dictionary mapping "SUBJECT_CATALOG" to course info
        """
        unique = {}
        for course in courses:
            key = f"{course.get('subject', '')}_{course.get('catalog_nbr', '')}"
            if key not in unique:
                unique[key] = {
                    "subject": course.get("subject", ""),
                    "catalog_nbr": course.get("catalog_nbr", ""),
                }
        return unique
    
    def get_status(self) -> dict:
        """Get current indexing status.
        
        Returns:
            Status dictionary with counts
        """
        return {
            "indexed_count": self.vector_store.count(),
            "cache_files": [str(f) for f in self.cache_dir.glob("*.json")],
        }




