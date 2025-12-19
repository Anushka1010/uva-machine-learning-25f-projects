"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.services import RAGEngine
from app.config import get_settings
from app.routers import chat_router, courses_router, schedule_router
from app.data.indexer import CourseIndexer


settings = get_settings()


def format_sis_time(time_str: str) -> str:
    """Format SIS time string to readable format.
    
    Converts "09.00.00.000000" to "9:00 AM", "14.30.00.000000" to "2:30 PM"
    
    Args:
        time_str: Time string in format "HH.MM.SS.ffffff"
        
    Returns:
        Formatted time string like "9:00 AM" or "2:30 PM"
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Initialize data if needed
    print("Starting HoosAdvisor...")
    
    # Check if we have indexed data
    try:
        indexer = CourseIndexer()
        status = indexer.get_status()
        print(f"Indexed courses: {status['indexed_count']}")
    except Exception as e:
        print(f"Could not check index status: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_title,
    description="AI-powered course planning assistant for UVA students",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Add custom filters to templates
templates.env.filters["format_time"] = format_sis_time

# Include routers
app.include_router(chat_router)
app.include_router(courses_router)
app.include_router(schedule_router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "HoosAdvisor",
        }
    )


@app.get("/admin/index", response_class=HTMLResponse)
async def admin_index(request: Request):
    """Admin page to trigger course indexing."""
    return templates.TemplateResponse(
        "admin_index.html",
        {
            "request": request,
            "title": "Admin - Index Courses",
        }
    )


@app.post("/admin/index/run")
async def run_indexing(request: Request):
    """Run course indexing from SIS API."""
    try:
        indexer = CourseIndexer()
        count = indexer.index_courses(term="1262", force_refresh=False)
        
        return templates.TemplateResponse(
            "admin_index.html",
            {
                "request": request,
                "title": "Admin - Index Courses",
                "success_message": f"Successfully indexed {count} courses!",
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "admin_index.html",
            {
                "request": request,
                "title": "Admin - Index Courses",
                "error_message": f"Error indexing courses: {str(e)}",
            }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app": settings.app_title}

