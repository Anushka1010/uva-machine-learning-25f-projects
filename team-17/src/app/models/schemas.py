"""Pydantic models for data validation."""

from typing import Optional
from pydantic import BaseModel, Field


class Course(BaseModel):
    """Course information model."""
    
    subject: str = Field(..., description="Subject code (e.g., CS)")
    catalog_number: str = Field(..., description="Course number (e.g., 4774)")
    title: str = Field(..., description="Course title")
    description: Optional[str] = Field(None, description="Course description")
    units: Optional[str] = Field(None, description="Credit hours")
    prerequisites: Optional[str] = Field(None, description="Prerequisites")
    
    @property
    def course_id(self) -> str:
        """Return formatted course ID."""
        return f"{self.subject} {self.catalog_number}"


class Section(BaseModel):
    """Course section model."""
    
    class_number: str = Field(..., description="Section class number")
    section: str = Field(..., description="Section identifier")
    instructor: Optional[str] = Field(None, description="Instructor name")
    days: Optional[str] = Field(None, description="Meeting days")
    start_time: Optional[str] = Field(None, description="Start time")
    end_time: Optional[str] = Field(None, description="End time")
    location: Optional[str] = Field(None, description="Room location")
    enrollment_total: Optional[int] = Field(None, description="Current enrollment")
    enrollment_cap: Optional[int] = Field(None, description="Enrollment capacity")
    waitlist_total: Optional[int] = Field(None, description="Waitlist count")


class ChatMessage(BaseModel):
    """Single chat message."""
    
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request from user."""
    
    message: str = Field(..., description="User's message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")


class ChatResponse(BaseModel):
    """Chat response from assistant."""
    
    response: str = Field(..., description="Assistant's response")
    sources: list[str] = Field(default_factory=list, description="Source documents used")
    conversation_id: str = Field(..., description="Conversation ID")


class CourseSearchRequest(BaseModel):
    """Course search parameters."""
    
    subject: Optional[str] = Field(None, description="Subject filter")
    keyword: Optional[str] = Field(None, description="Keyword search")
    term: str = Field(default="1262", description="Term code (e.g., 1262 for Spring 2025)")


class ScheduleItem(BaseModel):
    """Item in user's schedule."""
    
    course_id: str = Field(..., description="Course identifier")
    section_id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Course title")
    days: str = Field(..., description="Meeting days")
    start_time: str = Field(..., description="Start time")
    end_time: str = Field(..., description="End time")
    location: Optional[str] = Field(None, description="Room location")
    instructor: Optional[str] = Field(None, description="Instructor name")

