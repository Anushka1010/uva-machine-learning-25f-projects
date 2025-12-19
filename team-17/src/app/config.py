"""Application configuration settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


# =============================================================================
# Course Clusters - UVA CS Degree Requirements
# https://records.ureg.virginia.edu/preview_program.php?catoid=67&poid=10221
# =============================================================================

COURSE_CLUSTERS: dict[str, list[str]] = {
    # ===== BA/BS in Computer Science =====

    # Prerequisites to declare the major (7 credits total)
    "cs_prerequisites": [
        "CS 1110", "CS 1111", "CS 1112", "CS 1113",  # choose ONE (Intro to Programming)
        "CS 2100",  # Data Structures and Algorithms 1
    ],

    # Required CS courses in the major (20 credits total) - HIGH PRIORITY
    "CS_REQUIRED_COURSES": [
        "CS 2120",  # Discrete Mathematics and Theory 1
        "CS 2130",  # Computer Systems and Organization 1
        "CS 3100",  # Data Structures and Algorithms 2
        "CS 3120",  # Discrete Mathematics and Theory 2
        "CS 3130",  # Computer Systems and Organization 2
        "CS 3140",  # Software Development Essentials
    ],

    # Restricted electives (pick THREE courses = 9 credits)
    # Note: at most 3 credits of CS 4993 may count toward this requirement.
    "cs_restricted_electives": [
        "CS 3205",  # HCI in Software Development
        "CS 3240",  # Software Engineering
        "CS 3250",  # Software Testing
        "CS 3501",  # Special Topics in Computer Science (1-3)
        "CS 3710",  # Introduction to Cybersecurity
        "CS 4260",  # Internet Scale Applications
        "CS 4330",  # Advanced Computer Architecture
        "CS 4414",  # Operating Systems
        "CS 4434",  # Dependable Computing Systems
        "CS 4444",  # Introduction to Parallel Computing
        "CS 4457",  # Computer Networks
        "CS 4501",  # Special Topics in Computer Science (1-3)
        "CS 4610",  # Programming Languages
        "CS 4620",  # Compilers
        "CS 4630",  # Defense Against the Dark Arts
        "CS 4640",  # Programming Languages for Web Applications
        "CS 4710",  # Artificial Intelligence
        "CS 4720",  # Mobile Application Development
        "CS 4730",  # Computer Game Design
        "CS 4740",  # Cloud Computing
        "CS 4750",  # Database Systems
        "CS 4760",  # Network Security
        "CS 4770",  # Natural Language Processing
        "CS 4771",  # Reinforcement Learning
        "CS 4774",  # Machine Learning
        "CS 4790",  # Cryptocurrency
        "CS 4810",  # Introduction to Computer Graphics
        "CS 4993",  # Independent Study (1-3)
    ],

    # Distinguished Majors Program thesis course (6 credits total across two semesters)
    "cs_distinguished_majors": [
        "CS 4998",
    ],
}

CLUSTER_DESCRIPTIONS: dict[str, str] = {
    "cs_prerequisites": "Prereqs to declare the CS BA/BS major: one intro programming course (CS 1110/1111/1112/1113) + CS 2100.",
    "CS_REQUIRED_COURSES": "REQUIRED CS MAJOR COURSES (20 credits): CS 2120 Discrete Math I, CS 2130 Systems I, CS 3100 DSA2, CS 3120 Discrete Math II, CS 3130 Systems II, CS 3140 Software Dev Essentials.",
    "cs_restricted_electives": "Pick 3 (9 credits) from the prescribed CS restricted-elective list (CS 4993 counts for at most 3 credits).",
    "cs_integration_electives": "Pick 12 credits of approved non-CS integration electives (College of Arts & Sciences list in the catalog).",
    "cs_distinguished_majors": "Thesis/research course for Distinguished Majors (CS 4998 taken for two semesters).",
}

# Cluster weights - how many times to repeat cluster info in embeddings
# Higher = more weight in semantic search
CLUSTER_WEIGHTS: dict[str, int] = {
    "CS_REQUIRED_COURSES": 5,  # HIGH priority - required courses
    "cs_prerequisites": 2,
    "cs_restricted_electives": 2,
    "cs_distinguished_majors": 1,
}

def get_course_clusters(course_code: str) -> list[str]:
    """Get all clusters that a course belongs to.

    Args:
        course_code: Course code like "CS 4774"

    Returns:
        List of cluster names this course belongs to
    """
    clusters = []
    for cluster_name, courses in COURSE_CLUSTERS.items():
        if course_code in courses:
            clusters.append(cluster_name)
    return clusters


def get_cluster_description(cluster_name: str) -> str:
    """Get human-readable description of a cluster.

    Args:
        cluster_name: Name of the cluster

    Returns:
        Description string
    """
    return CLUSTER_DESCRIPTIONS.get(cluster_name, cluster_name)


def get_cluster_summary() -> str:
    """Get a formatted summary of all clusters for the AI system prompt.
    
    Returns:
        Formatted string describing all clusters and their courses
    """
    lines = ["UVA CS DEGREE REQUIREMENTS AND COURSE CLUSTERS:"]
    for cluster_id, courses in COURSE_CLUSTERS.items():
        desc = CLUSTER_DESCRIPTIONS.get(cluster_id, cluster_id)
        lines.append(f"\n{desc}:")
        lines.append(f"  Courses: {', '.join(courses)}")
    return "\n".join(lines)


# =============================================================================
# Application Settings
# =============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App Settings
    app_title: str = "HoosAdvisor Assistant"
    debug: bool = False
    
    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    
    # Vector Database
    chroma_persist_dir: str = "./data/chroma"
    
    # Embedding Weights (higher = more influence on similarity)
    embed_weight_description: int = 3
    embed_weight_title: int = 2
    embed_weight_prerequisites: int = 2
    embed_weight_subject: int = 1
    embed_weight_cluster: int = 2
    embed_weight_instructor: int = 0
    embed_weight_schedule: int = 0
    
    # SIS API
    sis_api_base_url: str = "https://sisuva.admin.virginia.edu/psc/ihprd/UVSS/SA/s/WEBLIB_HCX_CM.H_CLASS_SEARCH.FieldFormula.IScript_ClassSearch"
    
    # Paths
    # src/ directory (contains app/, templates/, static/)
    src_dir: Path = Path(__file__).parent.parent
    templates_dir: Path = src_dir / "templates"
    static_dir: Path = src_dir / "static"
    # data/ is at project root (same level as src/)
    data_dir: Path = src_dir.parent / "data"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
