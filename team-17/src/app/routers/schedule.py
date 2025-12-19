"""Schedule router for schedule building."""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.config import get_settings

router = APIRouter(prefix="/schedule", tags=["schedule"])
settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))


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

# In-memory schedule storage (for demo purposes)
user_schedules: dict[str, list[dict]] = {}


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def schedule_page(request: Request, user_id: str = "default"):
    """Render the schedule builder page."""
    schedule = user_schedules.get(user_id, [])
    
    # Organize by day for calendar view
    calendar = organize_by_day(schedule)
    
    return templates.TemplateResponse(
        "schedule.html",
        {
            "request": request,
            "user_id": user_id,
            "schedule": schedule,
            "calendar": calendar,
            "title": "Schedule - HoosAdvisor",
            "time_slots": generate_time_slots(),
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        }
    )


@router.post("/add", response_class=HTMLResponse)
async def add_to_schedule(
    user_id: str = Form(default="default"),
    course_id: str = Form(...),
    section_id: str = Form(...),
    title: str = Form(...),
    days: str = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    location: str = Form(default=""),
    instructor: str = Form(default=""),
):
    """Add a course to the schedule."""
    if user_id not in user_schedules:
        user_schedules[user_id] = []
    
    # Check for conflicts
    new_item = {
        "course_id": course_id,
        "section_id": section_id,
        "title": title,
        "days": days,
        "start_time": start_time,
        "end_time": end_time,
        "location": location,
        "instructor": instructor,
    }
    
    # Add to schedule
    user_schedules[user_id].append(new_item)
    
    return RedirectResponse(
        url=f"/schedule?user_id={user_id}",
        status_code=303,
    )


@router.post("/remove", response_class=HTMLResponse)
async def remove_from_schedule(
    user_id: str = Form(default="default"),
    course_id: str = Form(...),
    section_id: str = Form(...),
):
    """Remove a course from the schedule."""
    if user_id in user_schedules:
        user_schedules[user_id] = [
            item for item in user_schedules[user_id]
            if not (item["course_id"] == course_id and item["section_id"] == section_id)
        ]
    
    return RedirectResponse(
        url=f"/schedule?user_id={user_id}",
        status_code=303,
    )


@router.post("/clear", response_class=HTMLResponse)
async def clear_schedule(user_id: str = Form(default="default")):
    """Clear entire schedule."""
    if user_id in user_schedules:
        user_schedules[user_id] = []
    
    return RedirectResponse(
        url=f"/schedule?user_id={user_id}",
        status_code=303,
    )


def organize_by_day(schedule: list[dict]) -> dict[str, list[dict]]:
    """Organize schedule items by day of week."""
    day_map = {
        "Mo": "Monday",
        "Tu": "Tuesday",
        "We": "Wednesday",
        "Th": "Thursday",
        "Fr": "Friday",
    }
    
    calendar = {day: [] for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}
    
    for item in schedule:
        days_str = item.get("days", "")
        for abbrev, full_day in day_map.items():
            if abbrev in days_str:
                calendar[full_day].append(item)
    
    return calendar


def generate_time_slots() -> list[str]:
    """Generate time slots for schedule grid."""
    slots = []
    for hour in range(8, 22):  # 8 AM to 10 PM
        period = "AM" if hour < 12 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        slots.append(f"{display_hour}:00 {period}")
    return slots

