"""
Incrementally build the RateMyProfessor course cache.

Auto-stops when there are no more professors left to process.
Safe to run multiple times (resumes from progress file).

Run from project root: python src/scripts/build_rmp_cache.py
"""

import time
from app.data.sources import RMPApi


def main():
    # === CONFIG ===
    SCHOOL_ID = "1277"      # UVA
    TESTING = False         # False = full professor list
    MAX_PAGES = 1           # rating pages per professor (keep small!)
    BATCH_SIZE = 50         # professors processed per loop
    SLEEP_BETWEEN_RUNS = 2  # seconds between batches

    print("Starting RateMyProfessor cache builder")
    print(f"School ID: {SCHOOL_ID}")
    print(f"Batch size: {BATCH_SIZE}, Rating pages: {MAX_PAGES}")
    print("This will stop automatically when finished.\n")

    rmp = RMPApi(
        school_id=SCHOOL_ID,
        testing=TESTING,
    )

    # total professors (uses the already-working professor list)
    total_profs = len(rmp.api.professors)

    try:
        while True:
            # read processed count before
            progress_path = rmp._cache_path("progress", MAX_PAGES)
            before_processed = 0
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    before_processed = len(__import__("json").load(f).get("processed_tids", []))
            except FileNotFoundError:
                before_processed = 0

            # Do one incremental batch
            course_map = rmp.build_course_professor_map(
                max_pages=MAX_PAGES,
                batch_size=BATCH_SIZE,
            )

            # Read processed count after
            after_processed = 0
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    after_processed = len(__import__("json").load(f).get("processed_tids", []))
            except FileNotFoundError:
                after_processed = 0

            newly_processed = after_processed - before_processed

            print(
                f"Courses so far: {len(course_map)} | "
                f"Processed professors: {after_processed}/{total_profs} | "
                f"New this batch: {newly_processed}"
            )

            if after_processed >= total_profs:
                print("\nDone: all professors processed. Stopping.")
                break

            if newly_processed <= 0:
                print("\nDone: no new professors processed this batch. Stopping.")
                break

            time.sleep(SLEEP_BETWEEN_RUNS)

    except KeyboardInterrupt:
        print("\nStopped by user. Progress has been saved safely.")


if __name__ == "__main__":
    main()
