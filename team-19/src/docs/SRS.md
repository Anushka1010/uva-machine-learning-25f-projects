# Software Requirements Specification (SRS) for canvasGPT

**Version:** 1.0
**Date:** 2025-12-17
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose
The purpose of this document is to define the software requirements for **canvasGPT**, a desktop application designed to help students manage their academic life. canvasGPT aggregates data from the Canvas Learning Management System (LMS) and external course resources, synthesizes this information into a coherent knowledge base, and provides an AI-powered interface for querying and interaction.

### 1.2 Scope
canvasGPT is a local-first desktop application built with Electron. It integrates with the Canvas API to fetch course data (assignments, announcements, files) and uses AI agents to discover and ingest external resources (course websites, PDFs). The system uses a local vector database (LanceDB) and a relational database (SQLite) to store and synthesize this data, enabling features like semantic search, intelligent deadlines tracking, and a conversational course assistant.

### 1.3 Definitions and Acronyms
*   **LMS:** Learning Management System (specifically Instructure Canvas).
*   **IPC:** Inter-Process Communication (between Electron Main and Renderer processes).
*   **MCP:** Model Context Protocol (standard for AI agent interactions).
*   **Universal Item:** A normalized data unit representing any course content (assignment, file, reading, etc.).
*   **Entity:** A synthesized, deduplicated representation of a distinct course object (e.g., "Homework 1" combining the Canvas assignment and a PDF spec).
*   **Triage:** The process of reviewing low-confidence or incomplete data items.

---

## 2. System Architecture

### 2.1 Technology Stack
*   **Runtime:** Electron (Chromium + Node.js)
*   **Frontend:** React, TypeScript, Tailwind CSS, Vite
*   **Backend (Main Process):** Node.js, LangChain/LangGraph
*   **Database (Relational):** SQLite (via `better-sqlite3`)
*   **Database (Vector):** LanceDB
*   **AI/ML:**
    *   **LLM:** OpenAI GPT-4o-mini (via LangChain)
    *   **Embeddings:** `Xenova/all-MiniLM-L6-v2` (local ONNX runtime)
    *   **Cross-Encoder:** `Xenova/ms-marco-TinyBERT-L-2-v2` (for clustering)
    *   **QA Model:** `Xenova/distilbert-base-cased-distilled-squad` (for metadata extraction)

### 2.2 High-Level Architecture
The application follows a standard Electron multi-process architecture:
1.  **Renderer Process:** Hosts the React UI. Interacts with the Main process via a secure `contextBridge` and IPC channels.
2.  **Main Process:** Orchestrates the application lifecycle, manages databases, runs the local AI models, and executes the data pipeline agents.
3.  **Data Pipeline:** A series of services (`SyncManager`, `DiscoveryAgent`, `SynthesisManager`) that ingest, process, and refine data.

---

## 3. User Features

### 3.1 Authentication & Setup
*   **Canvas Integration:** Users must provide a Canvas API Token and Domain.
*   **Secure Storage:** Credentials are encrypted and stored locally using the OS keychain (via `electron-store` or similar mechanism).
*   **Data Wipe:** Users can delete all local data and keys via the Settings interface.

### 3.2 Dashboard
*   **Overview:** Displays a summary of active courses.
*   **Upcoming Items:** A widget showing the next 5-10 deadlines across all courses.
*   **Triage Status:** Indicators for items requiring user attention.

### 3.3 Course Management
*   **Course View:** Detailed view of a specific course, listing its synthesized entities (Assignments, Materials, etc.).
*   **Discovery:** Users can trigger a "Discovery Agent" to scan for external course websites or resources linked within Canvas but hosted externally.
*   **Sync:** Manual trigger to synchronize data from Canvas and external sources.

### 3.4 Content Synthesis & Triage
*   **Triage Queue:** A dedicated view for items with low confidence scores, missing metadata (e.g., no due date), or broken links.
*   **Review Actions:** Users can manually edit metadata, merge duplicates, or dismiss items.

### 3.5 Chat Assistant
*   **Conversational Interface:** A chat window allowing users to ask natural language questions about their courses (e.g., "What is due next week?", "Summarize the reading for History 101").
*   **Context-Aware:** The assistant uses RAG (Retrieval-Augmented Generation) to pull relevant "Universal Items" from the local vector store to answer queries accurately.

---

## 4. Data Pipeline Requirements

### 4.1 Data Ingestion
*   **Sources:**
    *   **Canvas API:** Assignments, Quizzes, Announcements, Modules, Files, Pages.
    *   **Web Scraper:** External HTML pages (via regex/CSS selectors defined in `IngestionRules`).
    *   **PDF Parser:** Local or remote PDFs (syllabus parsing).
*   **Rate Limiting:** Must respect Canvas API rate limits (concurrent request throttling).
*   **Deduplication:** Prevent duplicate entries based on `content_url`.

### 4.2 Synthesis & Clustering
*   **Universal Items:** All ingested data is normalized into `universal_items` records.
*   **Clustering:** The system uses ML-based semantic similarity (embeddings + cross-encoders) to group related items (e.g., a Canvas assignment entry + a PDF uploaded to Files).
*   **Entity Creation:** Clusters are resolved into `realised_entities`, selecting the best metadata from the grouped items.

### 4.3 Vector Search
*   **Embeddings:** All text content (titles, descriptions) is embedded using a local model (`all-MiniLM-L6-v2`) to ensure privacy and offline capability.
*   **Search:** Supports semantic query retrieval for the Chat Assistant.

---

## 5. Non-Functional Requirements

### 5.1 Privacy & Security
*   **Local-First:** All course data, embeddings, and database records reside on the user's machine. No data is sent to external servers except for:
    *   Canvas API calls (direct to LMS).
    *   LLM Inference (anonymized context sent to OpenAI API).
*   **Encryption:** API tokens must be stored securely.

### 5.2 Performance
*   **UI Responsiveness:** The UI should remain responsive during background sync operations.
*   **Search Latency:** Vector search results should return in < 200ms.
*   **Offline Access:** Users must be able to view cached data and chat (limited to RAG context) without an internet connection (though LLM calls require internet).

### 5.3 Reliability
*   **Error Handling:** The system must gracefully handle network failures (Canvas API downtime) and parsing errors (malformed HTML/PDFs).
*   **Data Integrity:** Foreign key constraints and transaction safety in SQLite to prevent data corruption.

---

## 6. Interface Requirements

### 6.1 IPC API (Main -> Renderer)
The application exposes a `window.canvasGPT` object with methods including:
*   `login(tokens)`
*   `askQuestion(query)`
*   `getCourses()` / `getCourseDetails(id)`
*   `getUpcomingItems()`
*   `getTriageItems()`
*   `syncAll()`
*   `discoverCourses()`

---

**Appendix A: Data Models**
*   See `DATA_PIPELINE_SPECIFICATION.md` for detailed SQL schemas and JSON structures.
