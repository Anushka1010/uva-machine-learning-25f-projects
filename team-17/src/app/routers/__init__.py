"""API routers."""

from .chat import router as chat_router
from .courses import router as courses_router
from .schedule import router as schedule_router

__all__ = [
    "chat_router",
    "courses_router",
    "schedule_router",
]

