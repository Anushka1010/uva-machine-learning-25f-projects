// @ts-ignore
import Database from 'better-sqlite3';
import { app } from 'electron';
import path from 'node:path';
import fs from 'node:fs';

const userDataPath = app.getPath('userData');

// Ensure userData directory exists
if (!fs.existsSync(userDataPath)) {
  fs.mkdirSync(userDataPath, { recursive: true });
}

const dbPath = path.join(userDataPath, 'database.sqlite');

const db = new Database(dbPath);
db.pragma('journal_mode = WAL');

// Initialize your database schema here
const createTableQuery = `
  CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY,          -- UUID or Canvas Course ID
    canvas_id INTEGER UNIQUE,     -- The actual ID from Canvas API
    name TEXT NOT NULL,
    course_code TEXT,             -- e.g., "CS 3100"
    term TEXT,                    -- e.g., "Fall 2025"
    color_hex TEXT,               -- For UI theming
    is_active BOOLEAN DEFAULT 1
  );

  CREATE TABLE IF NOT EXISTS ingestion_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id TEXT NOT NULL,
    
    -- Where should the app look?
    source_url TEXT NOT NULL,     -- The URL found by the agent (e.g., external syllabus link)
    source_type TEXT NOT NULL,    -- ENUM: 'CANVAS_API', 'EXTERNAL_HTML', 'PDF_LINK', 'GOOGLE_DRIVE'
    category TEXT,                -- e.g., 'assignments', 'syllabus', 'schedule/lecture slides'
    
    -- What allows the agent to navigate this source?
    -- This JSON blob stores agent-discovered context. 
    -- Example: { "css_selector": "div.homework-list", "pdf_page_range": "2-4" }
    extraction_config JSON,       
    
    -- How often should we check this?
    check_frequency_hours INTEGER DEFAULT 1,
    last_checked_at DATETIME,
    
    FOREIGN KEY(course_id) REFERENCES courses(id)
  );

  CREATE TABLE IF NOT EXISTS universal_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id TEXT NOT NULL,
    ingestion_rule_id INTEGER,    -- Link back to which rule found this item
    
    -- The ML model's classification results
    item_type TEXT NOT NULL,      -- ENUM: 'ASSIGNMENT', 'READING', 'SLIDE', 'SYLLABUS'
    title TEXT NOT NULL,
    description TEXT,
    
    -- Normalized data points
    due_date DATETIME,            -- Extracted by ML from raw text
    content_url TEXT,             -- Link to the actual resource (local or remote)
    
    -- For "Foundational ML" Context
    raw_content_snippet TEXT,     -- The text the ML used to classify this (for debugging/re-indexing)
    confidence_score REAL,        -- How sure is the model this is an 'Assignment'?
    
    -- Cached embedding for similarity comparison (JSON array of floats)
    embedding TEXT,               -- Serialized embedding vector for clustering
    
    is_read BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY(course_id) REFERENCES courses(id),
    FOREIGN KEY(ingestion_rule_id) REFERENCES ingestion_rules(id)
  );

  CREATE TABLE IF NOT EXISTS agent_checkpoints (
    course_id TEXT PRIMARY KEY,
    thread_id TEXT,               -- LangGraph thread identifier
    state_blob BLOB,              -- Serialized state of the graph
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(course_id) REFERENCES courses(id)
  );

  CREATE TABLE IF NOT EXISTS realised_entities (
    id TEXT PRIMARY KEY,          -- UUID
    title TEXT NOT NULL,
    description TEXT,
    due_date TEXT,                -- ISO8601 Date String
    entity_type TEXT NOT NULL,    -- e.g., 'assignment', 'exam', 'reading'
    status TEXT NOT NULL,         -- e.g., 'pending_review', 'confirmed'
    review_status TEXT DEFAULT 'needs_review', -- 'auto_verified', 'needs_review', 'user_verified'
    synthesis_log JSON,           -- Log of synthesis decisions
    confidence_score REAL,        -- 0.0 to 1.0
    metadata JSON,                -- Additional flexible fields
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS entity_provenance (
    id TEXT PRIMARY KEY,          -- UUID
    realised_entity_id TEXT NOT NULL,
    universal_item_id INTEGER NOT NULL,
    contribution_type TEXT NOT NULL, -- e.g., 'source_content', 'date_source', 'primary_ref'
    FOREIGN KEY(realised_entity_id) REFERENCES realised_entities(id) ON DELETE CASCADE,
    FOREIGN KEY(universal_item_id) REFERENCES universal_items(id) ON DELETE CASCADE
  );
`;

db.exec(createTableQuery);

// Migration: Add embedding column if it doesn't exist
try {
  const columns = db.pragma('table_info(universal_items)');
  const hasEmbedding = columns.some((col: any) => col.name === 'embedding');
  
  if (!hasEmbedding) {
    console.log('[DB] Adding embedding column to universal_items table...');
    db.exec('ALTER TABLE universal_items ADD COLUMN embedding TEXT');
    console.log('[DB] Embedding column added successfully');
  }
} catch (error) {
  console.error('[DB] Error checking/adding embedding column:', error);
}

export default db;
