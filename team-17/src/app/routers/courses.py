"""Courses router for browsing and searching courses."""

from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.config import get_settings
from app.data.sources import SISApi
from app.data.stores import SISStore

router = APIRouter(prefix="/courses", tags=["courses"])
settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))

# Cache store for faster searching
sis_store = SISStore()


def get_user_schedule(user_id: str) -> list[dict]:
    """Get user's schedule (lazy import to avoid circular dependency)."""
    from app.routers.schedule import user_schedules
    return user_schedules.get(user_id, [])


def format_sis_time(time_str: str) -> str:
    """Format SIS time string to readable format.
    
    Converts "09.00.00.000000" to "9:00 AM", "14.30.00.000000" to "2:30 PM"
    """
    if not time_str:
        return ""
    
    try:
        parts = time_str.split(".")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        
        period = "AM" if hour < 12 else "PM"
        if hour == 0:
            hour = 12
        elif hour > 12:
            hour -= 12
        
        return f"{hour}:{minute:02d} {period}"
    except (ValueError, IndexError):
        return time_str


# Add custom filter to templates
templates.env.filters["format_time"] = format_sis_time


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def courses_page(
    request: Request,
    subject: str = Query(default=""),
    keyword: str = Query(default=""),
    term: str = Query(default="1262"),
    page: int = Query(default=1),
    user_id: str = Query(default="default"),
    searched: str = Query(default=""),
):
    """Render the course browser page."""
    courses = []
    total_count = 0
    error_message = None
    from_cache = False
    
    # Get user's current schedule to mark already-added courses
    scheduled = get_user_schedule(user_id)
    scheduled_keys = {(item["course_id"], item["section_id"]) for item in scheduled}
    
    # Check if we have cached data for this term
    has_cache = sis_store.has(term)
    
    # Search if user clicked search button (searched=true) or has specific filters
    if searched or subject or keyword:
        try:
            if has_cache:
                cached_courses = sis_store.load(term)
                from_cache = True
                
                # Filter courses based on search criteria
                filtered = []
                subject_upper = subject.upper() if subject else ""
                keyword_lower = keyword.lower() if keyword else ""
                
                for course in cached_courses:
                    # Filter by subject (skip if "All Subjects" selected)
                    if subject_upper and course.get("subject", "").upper() != subject_upper:
                        continue
                    
                    # Filter by keyword in title/description
                    if keyword_lower:
                        title = course.get("descr", "").lower()
                        subject_code = course.get("subject", "").lower()
                        catalog = course.get("catalog_nbr", "").lower()
                        if keyword_lower not in title and keyword_lower not in subject_code and keyword_lower not in catalog:
                            continue
                    
                    filtered.append(course)
                
                courses = filtered
                total_count = len(courses)
            else:
                # Fall back to API search (requires at least subject or keyword)
                if subject or keyword:
                    sis_api = SISApi()
                    response = sis_api.search(
                        subject=subject if subject else None,
                        keyword=keyword if keyword else None,
                        term=term,
                        page=page,
                    )
                    courses = sis_api.get_classes_list(response)
                    total_count = len(courses)
            
        except Exception as e:
            error_message = f"Error searching courses: {str(e)}"
    
    return templates.TemplateResponse(
        "courses.html",
        {
            "request": request,
            "courses": courses,
            "subject": subject,
            "keyword": keyword,
            "term": term,
            "page": page,
            "total_count": total_count,
            "error_message": error_message,
            "title": "Courses - HoosAdvisor",
            "subjects": get_common_subjects(),
            "scheduled_keys": scheduled_keys,
            "from_cache": from_cache,
            "user_id": user_id,
        }
    )


@router.post("/search", response_class=HTMLResponse)
async def search_courses(
    subject: str = Form(default=""),
    keyword: str = Form(default=""),
    term: str = Form(default="1262"),
    searched: str = Form(default=""),
):
    """Handle course search form submission."""
    params = ["searched=true"]  # Always mark as searched
    if subject:
        params.append(f"subject={subject}")
    if keyword:
        params.append(f"keyword={keyword}")
    if term:
        params.append(f"term={term}")
    
    query_string = "&".join(params)
    
    return RedirectResponse(
        url=f"/courses?{query_string}",
        status_code=303,
    )


def get_common_subjects() -> list[dict]:
    """Get list of common subject codes."""
    return [
        {"code": "CS", "name": "Computer Science"},
        {"code": "DS", "name": "Data Science"},
        {"code": "MATH", "name": "Mathematics"},
        {"code": "STAT", "name": "Statistics"},
        {"code": "STS", "name": "Science, Technology & Society"},
        {"code": "ENGR", "name": "Engineering"},
        {"code": "APMA", "name": "Applied Mathematics"},
        {"code": "ECE", "name": "Electrical & Computer Engineering"},
        {"code": "PHYS", "name": "Physics"},
        {"code": "ECON", "name": "Economics"},
    ]

