# CanvasGPT

**Team ID:** 19 
**Members:**
- Ethan Zhang (dnq8kp)
- Paris Phan (auj4yx)

**Demo Video:** https://www.youtube.com/watch?v=IBh6QZ9l06A


---

## Overview

CanvasGPT is an intelligent desktop application that enhances the Canvas LMS experience by providing AI-powered course management and semantic search capabilities. Built with Electron and React, it syncs your Canvas courses locally and uses machine learning to intelligently cluster assignments, extract metadata from PDFs, and provide a conversational interface for querying course materials.

**Key Features:**
- **Local-First Architecture:** All your course data is synced and stored locally in SQLite with vector embeddings in LanceDB
- **AI-Powered Synthesis:** Automatically clusters related course materials (assignments, PDFs, announcements) using semantic similarity and ML models
- **Conversational Interface:** Chat with "Van Gogh" - an in-app LLM agent that understands your courses and answers questions using RAG (Retrieval-Augmented Generation)
- **PDF Intelligence:** Automatic text extraction and embedding from course PDFs for comprehensive search
- **MCP Integration:** Exposes course data to Claude Desktop and other MCP-compatible AI tools via the Model Context Protocol
- **Smart Metadata Extraction:** Uses transformer models to extract due dates, points, and other metadata from unstructured content

More information can be found in the docs/ directory
- SRS.md - Software Requirements Specifications
- TRS.md - Technical Requirements Specifications
- testing_setup.md - our canvas API endpoint testing courses suite
---

## Usage

### Prerequisites

- **Node.js** v24+ (or v22+)
- **Canvas LMS** account with API access
- **OpenAI API Key** (for embeddings and chat)

### Quick Start

1. **Clone and Install**
   ```bash
   git clone https://github.com/paris-phan/canvasGPT.git
   cd canvasGPT
   npm install
   ```

2. **Run the Application**
   ```bash
   npm run dev
   ```

3. **Initial Setup in the App**
   - Enter your Canvas domain (e.g., `canvas.instructure.com`)
   - Enter your Canvas API token
   - Enter your Notion API Key
   - Enter your Notion Database ID
   - Enter your OpenAI API key
   - Click "Connect" to authenticate

4. **Sync Your Courses**
   - Click the sync button to pull all your courses, assignments, and materials
   - The app will automatically:
     - Extract text from PDFs
     - Generate embeddings for semantic search
     - Cluster related materials using ML
     - Populate the vector database

5. **Start Using**
   - Browse your courses and assignments in the dashboard
   - Use the chat interface to ask questions about your coursework
   - View synthesized "Realized Entities" that group related materials

### Core Features

**Dashboard:** View all active courses with color-coded cards and quick stats

**Assignments View:** See upcoming assignments, filter by course, mark as complete

**Chat with Van Gogh:** Ask questions like:
- "What courses am I enrolled in?"
- "What's due this week?"
- "Find lecture notes about derivatives"
- "Summarize the requirements for my Machine Learning project"

**Course Materials:** Browse and search through all synced materials with full-text search

**Synthesis:** View automatically clustered "Realized Entities" that intelligently group:
- Assignment PDFs with their Canvas assignment
- Related readings and resources
- Lecture slides and supplementary materials

---

## Setup

### Environment Configuration

The app uses local storage for configuration:
- **macOS:** `~/Library/Application Support/canvasgpt/`
- **Linux:** `~/.config/canvasgpt/`
- **Windows:** `%APPDATA%/canvasgpt/`

Contains:
- `database.sqlite` - Course data and metadata
- `lancedb/` - Vector embeddings for semantic search
- `config.json` - API keys and settings

### Getting Canvas API Token

1. Log into Canvas
2. Go to Account → Settings
3. Scroll to "Approved Integrations"
4. Click "+ New Access Token"
5. Set purpose: "CanvasGPT Desktop App"
6. Copy the generated token

### Optional: MCP Server Setup (Prototype)

> **Note:** The MCP integration is currently implemented as a **separate standalone script** for development convenience and currently only runs on Macbook. In a production app, this would be integrated directly into the Electron application as a dual-mode binary.

**Current Architecture:**
- The MCP server (`mcp-server-standalone.ts`) runs as a **separate Node.js process**
- Launched independently by Claude Desktop (not by the Electron app)
- Reads the same SQLite database that the Electron app writes to
- Uses `sql.js` instead of `better-sqlite3` to avoid native module compatibility issues

**Setup:**

1. **Ensure Node.js is installed** (v24+ required for the MCP script)

2. **Configure Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
   ```json
   {
     "mcpServers": {
       "canvasGPT": {
         "command": "node",
         "args": ["/absolute/path/to/canvasGPT/bin/canvasgpt-mcp"]
       }
     }
   }
   ```
   
   **Important:** Replace `/absolute/path/to/canvasGPT` with your actual project directory path (use `pwd` command in the project folder to get it).

3. **Restart Claude Desktop** completely (Quit and reopen)

4. **Ask Claude questions about your courses:**
   - "What courses am I taking?"
   - "What's due this week?"
   - "Find materials about linear algebra"

**How It Works:**
- When you ask Claude a question, Claude launches `node bin/canvasgpt-mcp` as a subprocess
- The MCP server reads your SQLite database and responds via stdio (standard input/output)
- Your main Electron app and the MCP server are completely independent processes
- They only share data through the SQLite database file

See [MCP-SETUP.md](./MCP-SETUP.md) for detailed instructions and troubleshooting.

---

## Project Architecture

### System Architecture

```
┌─────────────────────────────────────────────┐
│         Electron App (Canvas GPT)      │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  Van Gogh LLM                      │    │
│  │  - Answers user questions          │    │
│  │  - Uses RAG from LanceDB           │    │
│  └────────────────────────────────────┘    │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  Ingestion Service LangGraph Agent |    │
│  │  - Syncs Canvas API → SQLite       │    │
│  │  - LangGraph agents cluster/embed  │    │
│  │  - Populates LanceDB vector store  │    │
│  └────────────────────────────────────┘    │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  Data Layer                        │    │
│  │  - SQLite (structured data)        │    │
│  │  - LanceDB (embeddings)            │    │
│  └────────────────────────────────────┘    │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  MCP Server (optional feature)     │    │
│  │  - Exposes tools to external AIs   │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                  ↑
                  │ MCP calls (if exposed)
                  │
         ┌────────┴────────┐
         │                 │
    Claude Desktop      Cline
    (external AI)   (external AI)
```

### Technology Stack

**Frontend:**
- React 18 + TypeScript
- Tailwind CSS for styling
- Vite for build tooling

**Backend (Electron Main Process):**
- Node.js with TypeScript
- SQLite (better-sqlite3) for structured data
- LanceDB for vector embeddings
- LangGraph for agentic workflows

**ML/AI:**
- Xenova/transformers.js for local embeddings (all-MiniLM-L6-v2)
- OpenAI API for chat (gpt-4o-mini)
- Cosine Similarity Clustering of embeddings for content grouping
- Question-answering models for metadata extraction

**Key Components:**
- **Sync Manager:** Orchestrates Canvas API sync and data ingestion
- **Synthesis Manager:** Clusters related items using semantic similarity
- **Vector Store:** LanceDB-based semantic search
- **Van Gogh Agent:** LangGraph-based conversational agent with RAG
- **Discovery Agent:** Automatically finds external course resources
- **MCP Server:** Exposes tools via Model Context Protocol

### Data Flow

1. **Ingestion:** Canvas API → SQLite (universal_items table)
2. **Synthesis:** Clustering → Realized Entities
3. **Embedding:** Items → Local embeddings → LanceDB
4. **Query:** User question → Embedding → Vector search → LLM → Answer

---

## Development

### Project Structure

```
canvasGPT/
├── electron/              # Main process code
│   ├── main.ts           # Electron entry point
│   ├── preload.ts        # IPC bridge
│   ├── database/         # SQLite & LanceDB
│   ├── services/         # Canvas API, sync, synthesis
│   ├── graph/            # LangGraph agents
│   └── mcp/              # MCP server implementation
├── src/                  # Renderer process (React)
│   ├── components/       # UI components
│   └── lib/              # Shared types & utils
├── bin/                  # MCP server launcher
└── scripts/              # Utilities & migrations
```

### Available Scripts

- `npm install` - install dependencies
- `npm run dev` - Start development server
- `npm run build` - Build production app
- `npm run lint` - Run ESLint
- `npm run wipe` - Clear all local data

---

## Troubleshooting

**"NODE_MODULE_VERSION" error:**
- Run `npx electron-rebuild` to recompile native modules
- Ensure Node.js version matches (v24+)

**Sync fails:**
- Verify Canvas API token is valid
- Check Canvas domain is correct (no https:// prefix needed)
- Ensure network connection

**Chat doesn't work:**
- Verify OpenAI API key is configured
- Check that sync has completed (data must exist)
- Restart app after initial sync

**Empty search results:**
- Sync must complete first (embeddings are generated during sync)
- Check LanceDB directory exists in app data folder

---

## License

This project is part of an academic course project.

---

## Acknowledgments

Built with:
- [Electron](https://www.electronjs.org/)
- [LangChain](https://www.langchain.com/)
- [LanceDB](https://lancedb.com/)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [OpenAI API](https://openai.com/) 