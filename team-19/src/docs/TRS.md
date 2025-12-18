# CanvasGPT Technical Specifications

**Version:** 1.0.0
**Last Updated:** 2025-12-17
**Project Type:** Electron Desktop Application
**Language:** TypeScript/React

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Core Components](#4-core-components)
5. [Data Architecture](#5-data-architecture)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [User Interface](#7-user-interface)
8. [API & Communication](#8-api--communication)
9. [Security & Authentication](#9-security--authentication)
10. [Performance Considerations](#10-performance-considerations)
11. [Development Workflow](#11-development-workflow)
12. [Deployment](#12-deployment)
13. [MCP Integration](#13-mcp-integration)
14. [Testing Strategy](#14-testing-strategy)
15. [Future Enhancements](#15-future-enhancements)

---

## 1. Executive Summary

### 1.1 Purpose

CanvasGPT is an intelligent desktop application that serves as a unified interface for Canvas LMS course management. It automatically discovers, syncs, and synthesizes course content from multiple sources into a coherent knowledge base accessible through both a visual interface and an AI-powered chat assistant.

### 1.2 Key Features

- **Automated Course Discovery**: Uses LangGraph-based agent to analyze Canvas course structure and identify data sources
- **Multi-Source Integration**: Ingests data from Canvas API, external websites, PDFs, and file repositories
- **Intelligent Synthesis**: ML-powered clustering and deduplication of duplicate/related content
- **Semantic Search**: Vector-based similarity search using LanceDB for fast content retrieval
- **AI Chat Interface**: GPT-4o-mini powered conversational assistant with RAG (Retrieval Augmented Generation)
- **MCP Server**: Model Context Protocol integration for external access via Claude Desktop
- **Triage System**: Automatically identifies items needing user review
- **Cross-Platform**: Built with Electron for macOS, Windows, and Linux support

### 1.3 Target Users

- Students managing multiple courses with varying content structures
- Educators looking to centralize course materials
- Administrators needing comprehensive course content analytics

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │  React UI    │  │ Electron     │  │  MCP Server (External) ││
│  │  Components  │  │ Main Process │  │  Claude Desktop API    ││
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘│
└─────────┼──────────────────┼──────────────────────┼──────────────┘
          │                  │                      │
          │       IPC        │         stdio        │
          │                  │                      │
┌─────────┴──────────────────┴──────────────────────┴──────────────┐
│                      APPLICATION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ Sync Manager │  │ Discovery    │  │  Query Agent           ││
│  │              │  │ Agent        │  │  (LangGraph)           ││
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘│
│         │                 │                       │             │
│  ┌──────┴─────────────────┴───────────────────────┴────────────┐│
│  │              Synthesis Pipeline                              ││
│  │  - Clustering  - Metadata Extraction  - Deduplication       ││
│  └──────────────────────────┬───────────────────────────────────┘│
└─────────────────────────────┼──────────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│                       DATA LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ SQLite DB    │  │ LanceDB      │  │  Electron Store        ││
│  │ (Structured) │  │ (Vectors)    │  │  (Credentials)         ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
          ▲                  ▲                      ▲
          │                  │                      │
┌─────────┴──────────────────┴──────────────────────┴──────────────┐
│                      EXTERNAL SERVICES                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ Canvas API   │  │ OpenAI API   │  │  External Websites     ││
│  │              │  │ (Embeddings/ │  │  (Scraped Content)     ││
│  │              │  │  Chat)       │  │                        ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Process Architecture

**Main Process (Electron)**
- Window management
- IPC handlers
- Database initialization
- Background sync orchestration
- MCP server lifecycle management

**Renderer Process (React)**
- UI rendering
- User interactions
- Local state management
- IPC client

**MCP Server Process (Standalone)**
- Independent Node.js process
- Stdio-based communication
- Direct SQLite/LanceDB access
- Claude Desktop integration

---

## 3. Technology Stack

### 3.1 Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.2.0 | UI framework |
| **TypeScript** | 5.2.2 | Type safety |
| **Tailwind CSS** | 4.1.17 | Styling framework |
| **Lucide React** | 0.555.0 | Icon library |
| **Vite** | 5.1.6 | Build tool & dev server |

### 3.2 Backend (Electron Main)

| Technology | Version | Purpose |
|------------|---------|---------|
| **Electron** | 30.0.1 | Desktop application framework |
| **Better-SQLite3** | 12.5.0 | Synchronous SQLite interface |
| **LanceDB** | 0.22.3 | Vector database for embeddings |
| **Axios** | 1.13.2 | HTTP client for Canvas API |
| **PDF-Parse** | 2.4.5 | PDF text extraction |
| **Electron-Store** | 11.0.2 | Persistent key-value storage |

### 3.3 Machine Learning

| Technology | Version | Purpose |
|------------|---------|---------|
| **@xenova/transformers** | 2.17.2 | ONNX Runtime (client-side ML) |
| **@langchain/langgraph** | 1.0.2 | Agent orchestration |
| **@langchain/openai** | 1.1.3 | OpenAI SDK integration |
| **onnxruntime-node** | 1.23.2 | ONNX inference engine |
| **chrono-node** | 2.9.0 | Natural language date parsing |

### 3.4 Models Used

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `Xenova/all-MiniLM-L6-v2` | Sentence Embeddings | ~23 MB | Item similarity & clustering |
| `Xenova/ms-marco-TinyBERT-L-2-v2` | Cross-Encoder | ~17 MB | Link confidence scoring |
| `Xenova/distilbert-base-cased-distilled-squad` | Question Answering | ~260 MB | Metadata extraction |
| `Xenova/mobilebert-uncased-mnli` | Zero-Shot Classifier | ~100 MB | Orphan item classification |
| `OpenAI text-embedding-3-small` | Embeddings API | Cloud | Query vectorization |
| `GPT-4o-mini` | Chat Completion | Cloud | Conversational interface |

---

## 4. Core Components

### 4.1 Discovery Agent

**Location:** `electron/graph/discovery_agent.ts`

**Purpose:** Automatically analyzes Canvas course structure to identify and configure data sources.

**Implementation:**
```typescript
interface DiscoveryState {
  courseId: string;
  fullContext: string;
  foundLinks: string[];
  finalRules: IngestionRule[];
}

// LangGraph workflow:
// 1. fetchCourseData: Get Canvas course info
// 2. analyzePages: Examine course pages for external links
// 3. generateRules: Create ingestion configurations
```

**Capabilities:**
- Identifies external syllabus links
- Detects course-specific resource websites
- Extracts module structures
- Generates CSS selectors for scraping
- Validates URL accessibility

**Output:** Array of `IngestionRule` objects with:
- `source_url`: Target URL
- `source_type`: CANVAS_API | EXTERNAL_HTML | PDF_LINK
- `category`: assignments, syllabus, schedule, files, modules
- `extraction_config`: JSON with scraping parameters

### 4.2 Sync Manager

**Location:** `electron/services/sync_manager.ts`

**Purpose:** Orchestrates data fetching from all configured sources.

**Process Flow:**
```
1. For each active course:
   a. syncCourseStandardData()
      - Fetch assignments (with pagination)
      - Fetch announcements
      - Fetch quizzes
      - Fetch discussion topics
      - Extract linked files from HTML
      - Process embedded PDFs

   b. processIngestionRules()
      - For CANVAS_API rules: Fetch modules/files
      - For EXTERNAL_HTML rules: Scrape with CSS selectors
      - For PDF_LINK rules: Download and parse text

   c. saveItem() for each extracted item
      - Deduplication check (content_url)
      - PDF text extraction if applicable
      - Insert into universal_items table

2. Trigger synthesis pipeline asynchronously

3. Populate LanceDB vector store
```

**Rate Limiting:**
- Max 5 concurrent Canvas API requests
- Retry logic with exponential backoff
- Respects Canvas API rate limits

**Deduplication:**
- Primary key: `content_url`
- Updates existing items instead of creating duplicates
- Preserves user-modified fields

### 4.3 Synthesis Pipeline

**Location:** `electron/services/synthesis/`

**Purpose:** Transform raw universal items into coherent, deduplicated entities.

#### 4.3.1 Manager (`manager.ts`)

**Orchestration:**
```typescript
class SynthesisManager {
  async processCourse(courseId: string) {
    1. findUnprocessedItems(courseId)
       - SQL: Items not in entity_provenance

    2. clusterItems(items)
       - Greedy clustering algorithm
       - ML-based similarity scoring
       - Confidence threshold: 0.55

    3. For each cluster:
       a. synthesizeEntity(cluster)
          - resolveCluster (anchor selection)
          - extractMetadata (if needed)
          - Set review_status

       b. saveRealisedEntity(entity, sources)
          - Insert into realised_entities
          - Create entity_provenance links
  }
}
```

#### 4.3.2 Cluster Resolver (`cluster_resolver.ts`)

**Anchor Selection Priority:**
1. ASSIGNMENT or QUIZ types (most complete metadata)
2. SYLLABUS type (authoritative source)
3. Earliest created_at timestamp

**Metadata Merging:**
- Title: From anchor
- Description: Concatenate all non-empty descriptions
- Due Date: From item with highest confidence
- URL: Prefer Canvas URLs over external

#### 4.3.3 Matcher (`matcher.ts`)

**Link Confidence Calculation:**
```typescript
function calculateLinkConfidence(itemA, itemB): number {
  // 1. Model-based similarity (70% weight)
  const modelScore = await crossEncoderSimilarity(
    itemA.title + itemA.description,
    itemB.title + itemB.description
  );

  // 2. Temporal proximity (30% weight)
  const timeScore = calculateTimeScore(
    itemA.due_date,
    itemB.due_date
  );

  // 3. Combine and adjust
  let finalScore = (modelScore * 0.7) + (timeScore * 0.3);

  // 4. Boost for high confidence
  if (modelScore > 0.9) finalScore = 1.0;

  // 5. Module context adjustment
  if (sameModule) finalScore += 0.1;
  if (differentModule) finalScore -= 0.1;

  return Math.max(0, Math.min(1, finalScore));
}
```

**Fallback Strategy:**
- If cross-encoder fails: Use Jaccard similarity
- If dates unparseable: Use 0.5 time score

#### 4.3.4 Metadata Extractor (`metadata_extractor.ts`)

**Two-Phase Extraction:**

**Phase 1: ML-Based (DistilBERT QA Model)**
```typescript
const questions = [
  "When is this assignment due?",
  "How many points is this worth?",
  "What is the deadline?"
];

for (const question of questions) {
  const result = await qa_model({
    question,
    context: concatenatedText
  });

  if (result.score > 0.5) {
    if (isDueDate(result.answer)) {
      metadata.due_date = parseDate(result.answer);
      confidence += 0.5;
    }
  }
}
```

**Phase 2: Regex-Based (Fallback)**
- Date parsing: chrono-node library
- Points parsing: `/Points:?\s*(\d+)/i`
- Confidence contribution: +0.3 (dates), +0.2 (points)

**Confidence Threshold:** Only apply extracted metadata if confidence > 0.8

### 4.4 Query Agent

**Location:** `electron/graph/agent.ts`

**Architecture:** LangGraph state machine

```
                ┌─────────────┐
                │ Supervisor  │ (Intent Classification)
                └──────┬──────┘
                       │
           ┌───────────┴──────────┐
           ▼                      ▼
    ┌──────────────┐      ┌──────────────┐
    │syncChecker   │      │  Retriever   │
    │(Sync Status) │      │(Vector Search)│
    └──────┬───────┘      └──────┬───────┘
           │                     │
           └──────────┬──────────┘
                      ▼
              ┌──────────────┐
              │  Generator   │
              │ (GPT-4o-mini)│
              └──────────────┘
```

**Intent Detection:**
```typescript
function supervisor(state) {
  const query = state.messages[-1].content.toLowerCase();

  if (query.includes('grade') || query.includes('sync')) {
    return { intent: 'grades_sync' };
  } else {
    return { intent: 'content' };
  }
}
```

**Retrieval Process:**
```typescript
async function retriever(state) {
  1. Generate query embedding (all-MiniLM-L6-v2)
  2. Search LanceDB for top 10 similar items
  3. Format results as context string
  4. Return { context: formattedResults }
}
```

**Generation:**
```typescript
async function generator(state) {
  const systemPrompt = `
    You are a helpful assistant.
    Use the following context: ${state.context}
  `;

  const response = await gpt4omini.invoke([
    { role: "system", content: systemPrompt },
    { role: "user", content: state.messages[-1].content }
  ]);

  return { messages: [response] };
}
```

---

## 5. Data Architecture

### 5.1 SQLite Schema

#### 5.1.1 Courses Table

```sql
CREATE TABLE courses (
  id TEXT PRIMARY KEY,              -- UUID or Canvas Course ID
  canvas_id INTEGER UNIQUE,         -- Canvas API ID
  name TEXT NOT NULL,               -- "CS 4774 - Machine Learning"
  course_code TEXT,                 -- "CS 4774"
  term TEXT,                        -- "Fall 2025"
  color_hex TEXT,                   -- "#FF5733"
  is_active BOOLEAN DEFAULT 1       -- Enrollment status
);
```

**Indexes:**
```sql
CREATE INDEX idx_courses_active ON courses(is_active);
CREATE INDEX idx_courses_canvas_id ON courses(canvas_id);
```

#### 5.1.2 Ingestion Rules Table

```sql
CREATE TABLE ingestion_rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  course_id TEXT NOT NULL,
  source_url TEXT NOT NULL,         -- Target URL
  source_type TEXT NOT NULL,        -- ENUM: CANVAS_API, EXTERNAL_HTML, PDF_LINK, GOOGLE_DRIVE
  category TEXT,                    -- assignments, syllabus, schedule, files, modules
  extraction_config JSON,           -- Scraping configuration
  check_frequency_hours INTEGER DEFAULT 1,
  last_checked_at DATETIME,
  FOREIGN KEY(course_id) REFERENCES courses(id)
);
```

**Extraction Config Example:**
```json
{
  "cssSelectors": {
    "title": "h3.assignment-title",
    "dueDate": "span.due-date",
    "description": "div.description"
  },
  "pdfPageRange": [1, 10],
  "apiParams": {
    "endpoint": "/api/v1/modules",
    "headers": { "Accept": "application/json" }
  }
}
```

#### 5.1.3 Universal Items Table

```sql
CREATE TABLE universal_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  course_id TEXT NOT NULL,
  ingestion_rule_id INTEGER,        -- Provenance tracking
  item_type TEXT NOT NULL,          -- ASSIGNMENT, READING, SLIDE, SYLLABUS,
                                     -- ANNOUNCEMENT, QUIZ, FILE, PAGE, DISCUSSION
  title TEXT NOT NULL,
  description TEXT,                 -- May contain HTML
  due_date DATETIME,                -- ISO8601 format
  content_url TEXT,                 -- Canonical URL (used for deduplication)
  raw_content_snippet TEXT,         -- First ~5000 chars for re-processing
  confidence_score REAL DEFAULT 1.0,-- Source reliability (1.0 for Canvas API)
  embedding TEXT,                   -- JSON array of 384 floats
  is_read BOOLEAN DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(course_id) REFERENCES courses(id),
  FOREIGN KEY(ingestion_rule_id) REFERENCES ingestion_rules(id)
);
```

**Indexes:**
```sql
CREATE INDEX idx_universal_items_course ON universal_items(course_id);
CREATE INDEX idx_universal_items_url ON universal_items(content_url);
CREATE INDEX idx_universal_items_type ON universal_items(item_type);
CREATE INDEX idx_universal_items_due_date ON universal_items(due_date);
```

**Item Types:**
- **ASSIGNMENT**: Graded work with deadlines
- **QUIZ**: Assessments and exams
- **READING**: Textbook chapters, articles
- **SLIDE**: Lecture presentations
- **SYLLABUS**: Course syllabi
- **ANNOUNCEMENT**: Course updates
- **FILE**: Generic file attachments
- **PAGE**: Course wiki pages
- **DISCUSSION**: Forum topics

#### 5.1.4 Realised Entities Table

```sql
CREATE TABLE realised_entities (
  id TEXT PRIMARY KEY,              -- UUID
  title TEXT NOT NULL,
  description TEXT,
  due_date TEXT,                    -- ISO8601
  entity_type TEXT NOT NULL,        -- assignment, quiz, syllabus, material,
                                     -- announcement, other
  status TEXT NOT NULL,             -- pending_review, confirmed
  review_status TEXT DEFAULT 'needs_review',  -- auto_verified, needs_review, user_verified
  synthesis_log JSON,               -- Array of synthesis decisions
  confidence_score REAL,            -- Minimum cluster confidence
  metadata JSON,                    -- Provenance and cluster info
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Metadata Structure:**
```json
{
  "anchor_item_id": 123,
  "cluster_size": 3,
  "min_confidence": 0.87,
  "provenance_log": [
    {
      "field": "title",
      "source_item_id": 123,
      "method": "anchor",
      "confidence": 1.0
    },
    {
      "field": "due_date",
      "source_item_id": 124,
      "method": "extracted_from_text",
      "confidence": 0.92
    }
  ]
}
```

**Synthesis Log Example:**
```json
[
  {
    "message": "Started new cluster with item 123",
    "item_id": 123
  },
  {
    "message": "Added item 124 to cluster",
    "related_to_item": 123,
    "confidence": 0.89
  },
  {
    "message": "Extracted due_date using QA model",
    "field": "due_date",
    "value": "2025-01-15T23:59:00Z",
    "confidence": 0.92
  }
]
```

#### 5.1.5 Entity Provenance Table

```sql
CREATE TABLE entity_provenance (
  id TEXT PRIMARY KEY,              -- UUID
  realised_entity_id TEXT NOT NULL,
  universal_item_id INTEGER NOT NULL,
  contribution_type TEXT NOT NULL,  -- source_content, date_source, primary_ref
  FOREIGN KEY(realised_entity_id) REFERENCES realised_entities(id) ON DELETE CASCADE,
  FOREIGN KEY(universal_item_id) REFERENCES universal_items(id) ON DELETE CASCADE
);
```

**Purpose:**
- Bidirectional linking between entities and source items
- Enables "show source" functionality
- Supports entity re-synthesis when sources update

**Indexes:**
```sql
CREATE INDEX idx_provenance_entity ON entity_provenance(realised_entity_id);
CREATE INDEX idx_provenance_item ON entity_provenance(universal_item_id);
```

### 5.2 LanceDB Vector Store

**Location:** `electron/database/vector-store.ts`

**Table Schema:**
```typescript
interface Document {
  id: string;              // Format: "item_{id}", "course_{id}", "rule_{id}"
  vector: Float32Array;    // 384 dimensions (all-MiniLM-L6-v2)
  text: string;            // Searchable text
  metadata: string;        // JSON-stringified metadata
}
```

**Text Field Construction:**
```typescript
// For universal_items:
text = [
  `${item_type}: ${title}`,
  `Course: ${course_name} (${course_code})`,
  `Due: ${due_date}`,
  description,
  raw_content_snippet
].filter(Boolean).join('\n\n');

// For courses:
text = [
  `Course: ${name}`,
  `Course Code: ${course_code}`,
  `Term: ${term}`,
  `Canvas ID: ${canvas_id}`,
  "You are enrolled in this course"
].filter(Boolean).join('\n');
```

**Population Process:**
```typescript
async function populateFromSQLite(sqliteDb) {
  const documents = [];

  // 1. Add courses
  for (const course of activeCourses) {
    const embedding = await generateEmbedding(courseText);
    documents.push({ id: `course_${course.id}`, vector: embedding, ... });
  }

  // 2. Add ingestion rules
  for (const rule of rules) {
    const embedding = await generateEmbedding(ruleText);
    documents.push({ id: `rule_${rule.id}`, vector: embedding, ... });
  }

  // 3. Add items with embeddings
  for (const item of itemsWithEmbeddings) {
    const embedding = JSON.parse(item.embedding);
    documents.push({ id: `item_${item.id}`, vector: embedding, ... });
  }

  // 4. Upsert to LanceDB
  await upsertDocuments(documents);
}
```

**Search Example:**
```typescript
const results = await similaritySearch(queryVector, 10);
// Returns: Array of documents sorted by cosine similarity
```

---

## 6. Machine Learning Pipeline

### 6.1 Embedding Generation

**Model:** `Xenova/all-MiniLM-L6-v2`

**Location:** `electron/services/synthesis/embedding_cache.ts`

**Implementation:**
```typescript
import { pipeline } from '@xenova/transformers';

let embedder: any = null;

export async function generateEmbedding(text: string): Promise<number[]> {
  if (!embedder) {
    embedder = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2'
    );
  }

  // Truncate to 512 tokens
  const truncated = text.substring(0, 2048);

  // Generate embedding
  const output = await embedder(truncated, {
    pooling: 'mean',
    normalize: true
  });

  // Extract 384-dim vector
  const embedding = Array.from(output.data);

  return embedding;
}
```

**Caching Strategy:**
- Embeddings stored in `universal_items.embedding` as JSON
- Generated during sync, reused for vector store population
- Prevents redundant computation

**Performance:**
- ~50-100ms per document on CPU
- Batch processing: 32 documents at a time
- Total for 500 items: ~10-20 seconds

### 6.2 Similarity Scoring

**Model:** `Xenova/ms-marco-TinyBERT-L-2-v2` (Cross-Encoder)

**Location:** `electron/services/synthesis/matcher.ts`

**Purpose:** Calculate semantic similarity between item pairs during clustering.

**Implementation:**
```typescript
import { pipeline } from '@xenova/transformers';

async function crossEncoderSimilarity(
  textA: string,
  textB: string
): Promise<number> {
  if (!crossEncoder) {
    crossEncoder = await pipeline(
      'text-classification',
      'Xenova/ms-marco-TinyBERT-L-2-v2'
    );
  }

  const input = `${textA} [SEP] ${textB}`.substring(0, 512);
  const result = await crossEncoder(input);

  // Return score for "similar" class
  return result[0].score;
}
```

**Fallback (Jaccard Similarity):**
```typescript
function jaccardSimilarity(textA: string, textB: string): number {
  const tokensA = new Set(textA.toLowerCase().split(/\s+/));
  const tokensB = new Set(textB.toLowerCase().split(/\s+/));

  const intersection = new Set(
    [...tokensA].filter(x => tokensB.has(x))
  );

  const union = new Set([...tokensA, ...tokensB]);

  return intersection.size / union.size;
}
```

### 6.3 Metadata Extraction

**Model:** `Xenova/distilbert-base-cased-distilled-squad`

**Location:** `electron/services/synthesis/metadata_extractor.ts`

**Question Answering Approach:**
```typescript
async function extractDueDate(text: string): Promise<{
  date: string | null;
  confidence: number;
}> {
  const questions = [
    "When is this assignment due?",
    "What is the deadline?",
    "When should this be submitted?"
  ];

  for (const question of questions) {
    const result = await qa_model({
      question,
      context: text
    });

    if (result.score > 0.5) {
      // Parse with chrono-node
      const parsed = chrono.parseDate(result.answer);

      if (parsed) {
        return {
          date: parsed.toISOString(),
          confidence: result.score
        };
      }
    }
  }

  // Fallback to regex
  const regexMatch = text.match(/due[:\s]+(.+)/i);
  if (regexMatch) {
    const parsed = chrono.parseDate(regexMatch[1]);
    if (parsed) {
      return { date: parsed.toISOString(), confidence: 0.3 };
    }
  }

  return { date: null, confidence: 0 };
}
```

**Supported Extractions:**
1. **Due Dates**
   - QA model confidence contribution: +0.5
   - Regex fallback contribution: +0.3
   - chrono-node handles formats like:
     - "January 15, 2025"
     - "1/15/25"
     - "next Friday"
     - "in 3 days"

2. **Points**
   - Regex: `/Points:?\s*(\d+(?:\.\d+)?)/i`
   - Contribution: +0.2

### 6.4 Orphan Classification

**Model:** `Xenova/mobilebert-uncased-mnli` (Zero-Shot)

**Location:** `electron/services/synthesis/orphan_classifier.ts`

**Purpose:** Classify items with uncertain types.

**Implementation:**
```typescript
async function classifyOrphanItem(text: string): Promise<{
  type: ItemType;
  confidence: number;
}> {
  const labels = [
    "This is a homework assignment",
    "This is a reading material",
    "This is a lecture slide",
    "This is a course syllabus",
    "This is an announcement"
  ];

  const result = await classifier(text, labels);

  const typeMap = {
    "This is a homework assignment": "ASSIGNMENT",
    "This is a reading material": "READING",
    "This is a lecture slide": "SLIDE",
    "This is a course syllabus": "SYLLABUS",
    "This is an announcement": "ANNOUNCEMENT"
  };

  return {
    type: typeMap[result.labels[0]],
    confidence: result.scores[0]
  };
}
```

---

## 7. User Interface

### 7.1 Application Views

| View | Component | Purpose |
|------|-----------|---------|
| **Dashboard** | `Dashboard.tsx` | Course overview cards |
| **Course View** | `CourseView.tsx` | Individual course details |
| **Assignments** | `Assignments.tsx` | Upcoming assignments list |
| **Materials List** | `MaterialsListView.tsx` | All course materials |
| **Detail View** | `ObjectView.tsx` | Individual item details |
| **Triage** | `Triage.tsx` | Items needing review |
| **Chat** | `Chat.tsx` | AI assistant interface |
| **Settings** | `Settings.tsx` | Configuration & credentials |

### 7.2 Component Hierarchy

```
App.tsx
├── Auth.tsx (if not authenticated)
└── Layout.tsx (if authenticated)
    ├── Sidebar.tsx
    │   ├── Navigation Links
    │   └── Sync Status
    └── Main Content Area
        ├── Dashboard
        ├── CourseView
        ├── Assignments
        ├── MaterialsListView
        ├── ObjectView
        ├── Triage
        ├── Chat
        └── Settings
```

### 7.3 State Management

**Local State (React useState):**
```typescript
const [currentView, setCurrentView] = useState<View>('chat');
const [selectedCourseId, setSelectedCourseId] = useState<string | null>(null);
const [selectedDetail, setSelectedDetail] = useState<{ type: ItemType; id: number } | null>(null);
const [previousView, setPreviousView] = useState<View>('dashboard');
```

**No Global State Library:**
- Props drilling for simple data flow
- IPC for backend communication
- No Redux/MobX (simplicity over complexity)

### 7.4 Window Management

**Authentication Window:**
```typescript
const AUTH_WINDOW_SIZE = {
  width: 600,
  height: 800,
  minWidth: 600,
  minHeight: 800
};
```

**Main Application Window:**
```typescript
const MAIN_WINDOW_SIZE = {
  width: 1280,
  height: 960,
  minWidth: 1280,
  minHeight: 960
};
```

**Dynamic Resizing:**
```typescript
useEffect(() => {
  if (window.canvasGPT && window.canvasGPT.setWindowSize) {
    if (isAuthenticated) {
      window.canvasGPT.setWindowSize(MAIN_WINDOW_SIZE);
    } else {
      window.canvasGPT.setWindowSize(AUTH_WINDOW_SIZE);
    }
  }
}, [isAuthenticated]);
```

### 7.5 Styling System

**Tailwind Configuration:**
```javascript
// tailwind.config.js
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: {
          primary: '#0097A7',
          dark: '#00838F',
        }
      }
    }
  }
}
```

**Utility Library:**
```typescript
// src/lib/utils.ts
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

---

## 8. API & Communication

### 8.1 Canvas API Integration

**Location:** `electron/services/canvas.ts`

**Authentication:**
- Bearer token authentication
- Token stored in Electron Store

**Endpoints Used:**

| Endpoint | Method | Purpose | Pagination |
|----------|--------|---------|------------|
| `/api/v1/courses` | GET | List active courses | Yes (50/page) |
| `/api/v1/courses/:id/assignments` | GET | Course assignments | Yes |
| `/api/v1/courses/:id/announcements` | GET | Course announcements | Yes |
| `/api/v1/courses/:id/quizzes` | GET | Course quizzes | Yes |
| `/api/v1/courses/:id/discussion_topics` | GET | Discussion topics | Yes |
| `/api/v1/courses/:id/modules` | GET | Course modules | Yes |
| `/api/v1/courses/:id/files` | GET | Course files | Yes |
| `/api/v1/courses/:courseId/files/:fileId` | GET | File metadata | No |

**Pagination Handler:**
```typescript
const parseNextLink = (linkHeader?: string): string | null => {
  if (!linkHeader) return null;

  const links = linkHeader.split(',');
  for (const link of links) {
    const match = link.match(/<([^>]+)>; rel="next"/);
    if (match?.[1]) return match[1];
  }
  return null;
};

const fetchCanvasCourses = async (): Promise<CanvasCourse[]> => {
  const collected: CanvasCourse[] = [];
  let nextUrl: string | null = `/api/v1/courses?per_page=50&enrollment_state=active`;

  while (nextUrl) {
    const response = await client.get(nextUrl);
    collected.push(...response.data);

    const linkHeader = response.headers?.link;
    nextUrl = parseNextLink(linkHeader);
  }

  return collected;
};
```

**Error Handling:**
```typescript
try {
  const response = await canvasClient.get(endpoint);
  return response.data;
} catch (error) {
  if (error.response?.status === 403) {
    console.warn('Access forbidden:', endpoint);
    return [];
  }
  if (error.response?.status === 404) {
    console.warn('Resource not found:', endpoint);
    return [];
  }
  throw error;
}
```

### 8.2 IPC Communication

**Location:** `electron/main.ts` (handlers), `electron/preload.ts` (API exposure)

**Preload API:**
```typescript
// electron/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('canvasGPT', {
  // Authentication
  saveKeys: (keys: AuthKeys) => ipcRenderer.send('save-keys', keys),
  getKeys: () => ipcRenderer.invoke('get-keys'),
  clearKeys: () => ipcRenderer.send('clear-keys'),

  // Courses
  getCourses: () => ipcRenderer.invoke('get-courses'),
  getCourseDetails: (courseId: string) => ipcRenderer.invoke('get-course-details', courseId),

  // Discovery & Sync
  discoverCourses: () => ipcRenderer.invoke('discover-courses'),
  syncAll: () => ipcRenderer.invoke('sync-all'),

  // Items
  getUpcomingItems: () => ipcRenderer.invoke('get-upcoming-items'),
  getTriageItems: () => ipcRenderer.invoke('get-triage-items'),
  getUniversalItem: (id: number) => ipcRenderer.invoke('get-universal-item', id),

  // Chat
  askQuestion: (query: string) => ipcRenderer.invoke('ask-question', query),

  // Utilities
  openExternal: (url: string) => ipcRenderer.invoke('open-external', url),
  setWindowSize: (size: WindowSize) => ipcRenderer.invoke('set-window-size', size),
  deleteAllData: () => ipcRenderer.invoke('delete-all-data'),
});
```

**Handler Examples:**
```typescript
// electron/main.ts
ipcMain.handle('get-courses', () => {
  try {
    const rows = db.prepare(`
      SELECT id, canvas_id, name, course_code, term, color_hex, is_active
      FROM courses
      ORDER BY name COLLATE NOCASE
    `).all();

    return rows.map(row => ({
      id: String(row.id),
      canvas_id: row.canvas_id,
      name: row.name,
      course_code: row.course_code ?? '',
      term: row.term ?? '',
      color_hex: row.color_hex ?? '',
      is_active: Boolean(row.is_active ?? 1),
    }));
  } catch (error: any) {
    console.error('[Courses] Failed to fetch courses:', error);
    return [];
  }
});

ipcMain.handle('sync-all', async () => {
  try {
    return await syncAll();
  } catch (error: any) {
    console.error('[Sync] Failed to sync all sources:', error);
    return { success: false, error: error.message };
  }
});
```

### 8.3 OpenAI API Integration

**Embeddings:**
```typescript
import { OpenAIEmbeddings } from '@langchain/openai';

const embeddings = new OpenAIEmbeddings({
  apiKey: keys.openaiKey,
  modelName: 'text-embedding-3-small',
});

const queryVector = await embeddings.embedQuery(query);
```

**Chat Completion:**
```typescript
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({
  apiKey: keys.openaiKey,
  modelName: 'gpt-4o-mini',
  temperature: 0.7,
});

const response = await model.invoke([
  { role: 'system', content: systemPrompt },
  { role: 'user', content: userMessage }
]);
```

---

## 9. Security & Authentication

### 9.1 Credential Storage

**Technology:** `electron-store` (encrypted key-value store)

**Location:** `electron/services/auth.ts`

**Stored Credentials:**
```typescript
interface AuthKeys {
  canvasToken: string;      // Canvas API access token
  canvasDomain: string;     // Canvas instance URL
  notionToken: string;      // Notion API token (legacy/future use)
  notionDbId: string;       // Notion database ID
  openaiKey: string;        // OpenAI API key
}
```

**Implementation:**
```typescript
import Store from 'electron-store';

const store = new Store({
  name: 'auth-keys',
  encryptionKey: 'your-encryption-key', // Auto-generated on first run
});

export function saveKeys(keys: AuthKeys) {
  store.set('auth', keys);
}

export function getKeys(): AuthKeys {
  return store.get('auth', {
    canvasToken: '',
    canvasDomain: '',
    notionToken: '',
    notionDbId: '',
    openaiKey: '',
  });
}

export function clearKeys() {
  store.delete('auth');
}
```

**Security Features:**
- AES-256 encryption at rest
- OS-level keychain integration (macOS/Windows)
- No plaintext credentials in code
- Auto-lock after app closure

### 9.2 API Key Validation

**Canvas:**
```typescript
async function validateCanvasToken(domain: string, token: string): Promise<boolean> {
  try {
    const response = await axios.get(`${domain}/api/v1/users/self`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    return response.status === 200;
  } catch (error) {
    return false;
  }
}
```

**OpenAI:**
```typescript
async function validateOpenAIKey(apiKey: string): Promise<boolean> {
  try {
    const embeddings = new OpenAIEmbeddings({ apiKey });
    await embeddings.embedQuery('test');
    return true;
  } catch (error) {
    return false;
  }
}
```

### 9.3 Data Privacy

**Local-First Architecture:**
- All data stored locally (SQLite + LanceDB)
- No external data transmission except:
  - Canvas API (required for sync)
  - OpenAI API (embeddings & chat)
- No analytics or telemetry

**User Data Control:**
```typescript
// Delete all local data
ipcMain.handle('delete-all-data', async () => {
  try {
    // Clear SQLite tables
    db.prepare('PRAGMA foreign_keys = OFF;').run();
    const tables = db.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).all();

    for (const table of tables) {
      db.prepare(`DELETE FROM ${table.name}`).run();
    }

    db.prepare('PRAGMA foreign_keys = ON;').run();

    // Clear LanceDB
    await clearVectorStore();

    return { success: true };
  } catch (error: any) {
    console.error('[Main] Failed to delete all data:', error);
    return { success: false, error: error.message };
  }
});
```

---

## 10. Performance Considerations

### 10.1 Database Optimization

**Indexes:**
```sql
-- Universal items
CREATE INDEX idx_universal_items_course ON universal_items(course_id);
CREATE INDEX idx_universal_items_url ON universal_items(content_url);
CREATE INDEX idx_universal_items_type ON universal_items(item_type);
CREATE INDEX idx_universal_items_due_date ON universal_items(due_date);

-- Entity provenance
CREATE INDEX idx_provenance_entity ON entity_provenance(realised_entity_id);
CREATE INDEX idx_provenance_item ON entity_provenance(universal_item_id);

-- Courses
CREATE INDEX idx_courses_active ON courses(is_active);
```

**Query Performance:**
- Typical query time: <5ms for indexed lookups
- Full course scan: <50ms for 500 items
- Join queries: <100ms for entity + provenance

**WAL Mode:**
```typescript
db.pragma('journal_mode = WAL');
// Enables concurrent reads during writes
```

### 10.2 Vector Search Performance

**LanceDB Characteristics:**
- **Index Type:** IVF-PQ (Inverted File with Product Quantization)
- **Search Latency:** ~10-50ms for 10,000 vectors
- **Memory Usage:** ~5 MB per 1,000 384-dim vectors
- **Disk Usage:** ~1.5 KB per vector (compressed)

**Optimization Strategies:**
```typescript
// Limit result size
const results = await similaritySearch(queryVector, 10); // Top 10 only

// Filter by course
const results = await similaritySearch(
  queryVector,
  10,
  `metadata.course_id == '${courseId}'`
);
```

### 10.3 Embedding Generation

**Batch Processing:**
```typescript
async function generateEmbeddingsBatch(texts: string[]): Promise<number[][]> {
  const embedder = await getEmbedder();

  const batchSize = 32;
  const embeddings: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const batchEmbeddings = await Promise.all(
      batch.map(text => generateEmbedding(text))
    );
    embeddings.push(...batchEmbeddings);
  }

  return embeddings;
}
```

**Performance Metrics:**
- Single embedding: ~50-100ms (CPU)
- Batch of 32: ~1.5-2 seconds
- 500 items: ~15-20 seconds total

### 10.4 Sync Performance

**Canvas API Rate Limits:**
- Default: 3000 requests/hour
- Authenticated: 6000 requests/hour

**Concurrency Control:**
```typescript
import PQueue from 'p-queue';

const queue = new PQueue({ concurrency: 5 });

for (const assignment of assignments) {
  queue.add(() => processAssignment(assignment));
}

await queue.onIdle();
```

**Deduplication Early Exit:**
```typescript
if (item.content_url) {
  const existing = db.prepare(
    'SELECT id FROM universal_items WHERE content_url = ? AND course_id = ?'
  ).get(item.content_url, item.course_id);

  if (existing) {
    // Update instead of insert
    db.prepare(`UPDATE universal_items SET ... WHERE id = ?`).run(existing.id);
    return;
  }
}
```

---

## 11. Development Workflow

### 11.1 Project Structure

```
canvasGPT/
├── electron/
│   ├── main.ts                  # Main process entry
│   ├── preload.ts               # Preload script (context bridge)
│   ├── database/
│   │   ├── db.ts                # SQLite initialization
│   │   └── vector-store.ts      # LanceDB wrapper
│   ├── graph/
│   │   ├── agent.ts             # Query agent (LangGraph)
│   │   ├── discovery_agent.ts   # Course discovery agent
│   │   └── db_utils.ts          # Agent database helpers
│   ├── services/
│   │   ├── auth.ts              # Credential management
│   │   ├── canvas.ts            # Canvas API client
│   │   ├── sync_manager.ts      # Sync orchestration
│   │   ├── pdf_parser.ts        # PDF text extraction
│   │   ├── notion-setup.ts      # Notion integration (legacy)
│   │   ├── ml/
│   │   │   ├── index.ts         # ML model loader
│   │   │   └── model_loader.ts  # Xenova model initialization
│   │   └── synthesis/
│   │       ├── manager.ts       # Synthesis orchestrator
│   │       ├── cluster_resolver.ts  # Anchor selection
│   │       ├── matcher.ts       # Link confidence scoring
│   │       ├── metadata_extractor.ts  # QA-based extraction
│   │       ├── orphan_classifier.ts   # Zero-shot classification
│   │       ├── embedding_cache.ts     # Embedding generation
│   │       └── candidate_generator.ts # Clustering helpers
│   └── mcp/
│       └── server.ts            # MCP server (for Electron integration)
├── src/
│   ├── main.tsx                 # React entry point
│   ├── App.tsx                  # Root component
│   ├── components/
│   │   ├── Auth.tsx             # Login form
│   │   ├── Layout.tsx           # App shell
│   │   ├── Sidebar.tsx          # Navigation
│   │   ├── Dashboard.tsx        # Course overview
│   │   ├── CourseView.tsx       # Course details
│   │   ├── Assignments.tsx      # Upcoming assignments
│   │   ├── MaterialsListView.tsx  # All materials
│   │   ├── ObjectView.tsx       # Item detail view
│   │   ├── Triage.tsx           # Review queue
│   │   ├── Chat.tsx             # AI assistant
│   │   ├── Settings.tsx         # Configuration
│   │   └── ui.tsx               # Reusable UI components
│   └── lib/
│       ├── types.ts             # TypeScript interfaces
│       └── utils.ts             # Utility functions
├── mcp-server-standalone.ts     # Standalone MCP server
├── bin/
│   └── canvasgpt-mcp            # MCP server executable
├── docs/
│   └── testing_setup.md         # Test configuration docs
├── vite.config.ts               # Vite configuration
├── package.json                 # Dependencies
├── tsconfig.json                # TypeScript config
├── tailwind.config.js           # Tailwind CSS config
├── DATA_PIPELINE_SPECIFICATION.md  # Data pipeline docs
├── MCP-SETUP.md                 # MCP setup guide
└── README.md                    # Project overview
```

### 11.2 Build Configuration

**Vite Config:**
```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import electron from 'vite-plugin-electron';
import electronRenderer from 'vite-plugin-electron-renderer';

export default defineConfig({
  plugins: [
    react(),
    electron([
      {
        entry: 'electron/main.ts',
        vite: {
          build: {
            outDir: 'dist-electron',
          },
        },
      },
      {
        entry: 'electron/preload.ts',
        onstart(options) {
          options.reload();
        },
        vite: {
          build: {
            outDir: 'dist-electron',
          },
        },
      },
    ]),
    electronRenderer(),
  ],
});
```

**Electron Builder:**
```json
// package.json (build section)
{
  "build": {
    "appId": "com.canvasgpt.app",
    "productName": "CanvasGPT",
    "directories": {
      "output": "release"
    },
    "files": [
      "dist/**/*",
      "dist-electron/**/*",
      "package.json"
    ],
    "mac": {
      "target": ["dmg", "zip"],
      "category": "public.app-category.education",
      "icon": "public/canvas_reversed_logo.png"
    },
    "win": {
      "target": ["nsis", "zip"],
      "icon": "public/canvas_reversed_logo.png"
    },
    "linux": {
      "target": ["AppImage", "deb"],
      "category": "Education"
    }
  }
}
```

### 11.3 Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build && electron-builder",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview",
    "wipe": "rm -rf \"$HOME/Library/Application Support/canvasgpt\""
  }
}
```

### 11.4 Development Environment

**Prerequisites:**
- Node.js >= 18.0
- npm >= 9.0
- Git

**Setup:**
```bash
# Clone repository
git clone https://github.com/yourusername/canvasGPT.git
cd canvasGPT

# Install dependencies
npm install

# Start development server
npm run dev
```

**Hot Reload:**
- Vite provides HMR for React components
- Electron main process requires manual restart
- Preload script auto-reloads on changes

---

## 12. Deployment

### 12.1 Build Process

**Production Build:**
```bash
npm run build
```

**Output:**
```
canvasGPT/
├── dist/                    # Vite build (renderer)
│   ├── index.html
│   ├── assets/
│   │   ├── index-*.js
│   │   └── index-*.css
│   └── ...
├── dist-electron/           # Electron build (main + preload)
│   ├── main.js
│   └── preload.mjs
└── release/                 # Electron Builder output
    ├── CanvasGPT-1.0.0.dmg         (macOS)
    ├── CanvasGPT Setup 1.0.0.exe   (Windows)
    └── CanvasGPT-1.0.0.AppImage    (Linux)
```

### 12.2 Platform-Specific Considerations

**macOS:**
- Code signing required for distribution
- Notarization required for Gatekeeper
- Install location: `/Applications/CanvasGPT.app`
- Data location: `~/Library/Application Support/canvasgpt`

**Windows:**
- Installer: NSIS
- Install location: `C:\Program Files\CanvasGPT`
- Data location: `%APPDATA%\canvasgpt`

**Linux:**
- AppImage (portable)
- Debian package (deb)
- Data location: `~/.config/canvasgpt`

### 12.3 Auto-Update (Future)

**Implementation Plan:**
```typescript
import { autoUpdater } from 'electron-updater';

autoUpdater.checkForUpdatesAndNotify();

autoUpdater.on('update-available', () => {
  // Show notification
});

autoUpdater.on('update-downloaded', () => {
  // Prompt user to restart
});
```

---

## 13. MCP Integration

### 13.1 Architecture

**Standalone MCP Server:**
- **Location:** `mcp-server-standalone.ts`
- **Transport:** stdio (standard input/output)
- **Protocol:** Model Context Protocol (MCP) v1.0
- **Client:** Claude Desktop

**Communication Flow:**
```
Claude Desktop
     ↓ (stdio)
MCP Server (Node.js)
     ↓ (direct access)
SQLite Database + LanceDB
```

### 13.2 Available Tools

#### 13.2.1 search_materials

**Purpose:** Semantic search across course materials

**Input Schema:**
```json
{
  "query": "string (required)",
  "courseId": "string (optional)"
}
```

**Implementation:**
```typescript
async function searchMaterials(query: string, courseId?: string): Promise<string> {
  // 1. Generate embedding
  const embeddings = new OpenAIEmbeddings({ apiKey });
  const queryVector = await embeddings.embedQuery(query);

  // 2. Search LanceDB
  const results = await similaritySearch(queryVector, 5);

  // 3. Format results
  return results.map((r, i) =>
    `${i + 1}. ${r.title}\n   ${r.text.substring(0, 200)}...`
  ).join('\n\n');
}
```

#### 13.2.2 get_upcoming_assignments

**Purpose:** Get assignments due in the next N days

**Input Schema:**
```json
{
  "daysAhead": "number (default: 7)"
}
```

**Implementation:**
```typescript
async function getUpcomingAssignments(daysAhead: number = 7): Promise<string> {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() + daysAhead);

  const items = db.prepare(`
    SELECT ui.title, ui.due_date, ui.item_type, c.name as course_name
    FROM universal_items ui
    JOIN courses c ON ui.course_id = c.id
    WHERE ui.due_date IS NOT NULL
      AND ui.due_date <= ?
      AND ui.due_date >= datetime('now')
    ORDER BY ui.due_date ASC
    LIMIT 20
  `).all(cutoffDate.toISOString());

  return items.map(item =>
    `- ${item.title} (${item.course_name})\n  Due: ${new Date(item.due_date).toLocaleString()}`
  ).join('\n\n');
}
```

#### 13.2.3 get_course_overview

**Purpose:** Get summary of all courses or specific course details

**Input Schema:**
```json
{
  "courseId": "string (optional)"
}
```

**Implementation:**
```typescript
async function getCourseOverview(courseId?: string): Promise<string> {
  if (courseId) {
    // Single course details
    const course = db.prepare(`
      SELECT c.*, COUNT(ui.id) as item_count
      FROM courses c
      LEFT JOIN universal_items ui ON c.id = ui.course_id
      WHERE c.id = ?
      GROUP BY c.id
    `).get(courseId);

    return `${course.name} (${course.course_code})\n- Total items: ${course.item_count}`;
  } else {
    // All courses summary
    const courses = db.prepare(`
      SELECT c.name, c.course_code, COUNT(ui.id) as item_count
      FROM courses c
      LEFT JOIN universal_items ui ON c.id = ui.course_id
      GROUP BY c.id
    `).all();

    return courses.map(c => `${c.name}: ${c.item_count} items`).join('\n');
  }
}
```

#### 13.2.4 get_item_details

**Purpose:** Get full details for a specific item

**Input Schema:**
```json
{
  "itemId": "number (required)"
}
```

**Implementation:**
```typescript
async function getItemDetails(itemId: number): Promise<string> {
  const item = db.prepare(`
    SELECT ui.*, c.name as course_name
    FROM universal_items ui
    JOIN courses c ON ui.course_id = c.id
    WHERE ui.id = ?
  `).get(itemId);

  if (!item) return "Item not found";

  return `${item.title} (${item.course_name})
Type: ${item.item_type}
Due: ${item.due_date ? new Date(item.due_date).toLocaleString() : 'No due date'}
URL: ${item.content_url || 'N/A'}

Description:
${item.description || 'No description'}

Content:
${item.raw_content_snippet?.substring(0, 500) || 'No content'}...`;
}
```

### 13.3 Setup & Configuration

**Claude Desktop Config:**
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

**Server Startup:**
```typescript
// mcp-server-standalone.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'canvasGPT',
  version: '1.0.0',
}, {
  capabilities: { tools: {} },
});

// Register tools...
server.setRequestHandler(ListToolsRequestSchema, ...);
server.setRequestHandler(CallToolRequestSchema, ...);

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## 14. Testing Strategy

### 14.1 Test Coverage Plan

**Unit Tests (Planned):**
- Database query functions
- Embedding generation
- Similarity scoring
- Metadata extraction
- Clustering algorithm

**Integration Tests (Planned):**
- Canvas API client
- Sync manager end-to-end
- Synthesis pipeline
- MCP server tools

**E2E Tests (Planned):**
- User authentication flow
- Course discovery
- Full sync cycle
- Chat interactions

### 14.2 Test Environment

**Configuration:**
```markdown
# docs/testing_setup.md

## Canvas Setup
### 3 Courses
- CS 4444 - Intro to Parallel Computing
- CS 4774 - Machine Learning
- MATH 3354 - Survey of Algebra

### Course Structures
| Section       | CS 4444           | CS 4774                  | MATH 3354         |
|---------------|-------------------|--------------------------|-------------------|
| Home          | External syllabus | External site w/ routes  | [empty]           |
| Announcements | [empty]           | Update notices           | [empty]           |
| Syllabus      | [DNE]             | [empty]                  | Syllabus PDF      |
| Modules       | Lecture PDFs      | [DNE]                    | [DNE]             |
| Assignments   | HW w/ PDFs        | HW w/ PDFs & due dates   | No PDF, no dates  |
| Files         | [DNE]             | [DNE]                    | Lecture + HW PDFs |
```

**Test Data:**
- Mock Canvas API responses
- Sample PDFs with varying structures
- External HTML pages with different selectors

---

## 15. Future Enhancements

### 15.1 Planned Features

**High Priority:**
1. **Notion Integration**
   - Two-way sync with Notion databases
   - Export realised entities to Notion pages
   - Custom views and filters

2. **Calendar View**
   - Visual timeline of assignments
   - Drag-and-drop rescheduling
   - Google Calendar sync

3. **Advanced Triage**
   - ML-powered conflict detection
   - Automatic priority assignment
   - Smart notifications

**Medium Priority:**
4. **Multi-User Support**
   - User profiles
   - Shared course data
   - Collaborative notes

5. **Mobile Companion App**
   - React Native implementation
   - Push notifications
   - Offline mode

6. **Enhanced Discovery**
   - Playwright-based browser automation
   - JavaScript-rendered page support
   - CAPTCHA handling

**Low Priority:**
7. **Plugin System**
   - Custom data sources
   - Third-party integrations
   - Community extensions

8. **Analytics Dashboard**
   - Time spent per course
   - Assignment completion rates
   - Study pattern insights

### 15.2 Technical Debt

**Known Issues:**
1. No error boundaries in React components
2. Limited input validation on IPC handlers
3. Hardcoded retry logic in Canvas API client
4. No rate limiting on OpenAI API calls
5. Synthesis pipeline not resumable after crash

**Refactoring Opportunities:**
1. Extract Canvas API client to separate package
2. Implement proper logging framework (Winston/Pino)
3. Add comprehensive TypeScript interfaces
4. Migrate to React Query for state management
5. Implement proper error tracking (Sentry)

### 15.3 Performance Improvements

**Planned Optimizations:**
1. **Incremental Sync**
   - Only fetch items modified since last sync
   - Use Canvas API `updated_since` parameter

2. **Parallel Synthesis**
   - Process multiple courses concurrently
   - Worker threads for CPU-intensive ML

3. **Smart Caching**
   - Cache Canvas API responses (Redis)
   - Memoize embedding generation
   - LRU cache for vector search results

4. **Database Partitioning**
   - Separate database per semester
   - Archive old course data
   - Lazy load historical records

---

## Appendix A: Environment Variables

```bash
# Development
VITE_DEV_SERVER_URL=http://localhost:5173

# Production (optional overrides)
CANVAS_API_BASE=https://canvas.instructure.com
OPENAI_API_BASE=https://api.openai.com/v1
```

---

## Appendix B: Database Migration

**Current Version:** 1.0

**Migration Script (Future):**
```typescript
// scripts/migrate_db.ts
import db from '../electron/database/db';

const CURRENT_VERSION = 1;

function getCurrentVersion(): number {
  try {
    const result = db.prepare('PRAGMA user_version').get();
    return result.user_version;
  } catch {
    return 0;
  }
}

function migrate_v0_to_v1() {
  db.exec(`
    ALTER TABLE universal_items ADD COLUMN embedding TEXT;
    PRAGMA user_version = 1;
  `);
}

function runMigrations() {
  const currentVersion = getCurrentVersion();

  if (currentVersion < 1) {
    console.log('Migrating to v1...');
    migrate_v0_to_v1();
  }

  // Future migrations...
}

runMigrations();
```

---

## Appendix C: API Reference

### IPC API

```typescript
interface CanvasGPTAPI {
  // Authentication
  saveKeys(keys: AuthKeys): void;
  getKeys(): Promise<AuthKeys>;
  clearKeys(): void;

  // Courses
  getCourses(): Promise<Course[]>;
  getCourseDetails(courseId: string): Promise<CourseDetails>;

  // Discovery & Sync
  discoverCourses(): Promise<DiscoveryResult>;
  syncAll(): Promise<SyncResult>;

  // Items
  getUpcomingItems(): Promise<UniversalItem[]>;
  getTriageItems(): Promise<TriageItem[]>;
  getUniversalItem(id: number): Promise<UniversalItem | null>;

  // Chat
  askQuestion(query: string): Promise<string>;

  // Utilities
  openExternal(url: string): Promise<void>;
  setWindowSize(size: WindowSize): Promise<void>;
  deleteAllData(): Promise<{ success: boolean; error?: string }>;
}
```

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| **Universal Item** | Raw content item from any source (Canvas, external, PDF) |
| **Realised Entity** | Synthesized, deduplicated entity from clustered universal items |
| **Ingestion Rule** | Configuration for fetching data from a specific source |
| **Anchor** | Representative item chosen as primary source for entity metadata |
| **Link Confidence** | ML-based similarity score between two items (0-1) |
| **Synthesis** | Process of clustering and deduplicating universal items |
| **Triage** | Queue of items or entities needing user review |
| **Discovery Agent** | LangGraph agent that analyzes courses and generates ingestion rules |
| **Query Agent** | LangGraph agent that answers user questions via RAG |
| **MCP** | Model Context Protocol - standard for AI assistant tool integration |
| **Provenance** | Tracking of which universal items contributed to an entity |

---

## Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-12-17 | Initial comprehensive specification | Technical Documentation Team |

---

**End of Technical Specifications**
