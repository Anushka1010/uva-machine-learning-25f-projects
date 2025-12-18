import * as lancedb from '@lancedb/lancedb';
import { app } from 'electron';
import path from 'node:path';
import fs from 'node:fs';

const userDataPath = app.getPath('userData');
const vectorDbPath = path.join(userDataPath, 'lancedb');

// Ensure userData directory exists
if (!fs.existsSync(userDataPath)) {
  fs.mkdirSync(userDataPath, { recursive: true });
}

let dbInstance: lancedb.Connection | null = null;

const getDb = async () => {
  if (!dbInstance) {
    dbInstance = await lancedb.connect(vectorDbPath);
  }
  return dbInstance;
};

interface Document {
  id: string;
  vector: number[];
  text: string;
  metadata: string; // JSON string
  [key: string]: unknown;
}

/* Checks if a document with the given ID exists in the vector store. */
export const documentExists = async (id: string): Promise<boolean> => {
  const db = await getDb();
  try {
    const table = await db.openTable('course_knowledge');
    console.log(`[VectorStore] Checking if document exists: ${id}`);
    
    // Use SQL-like filter syntax for LanceDB
    const results = await table.query().filter(`id == '${id}'`).limit(1).toArray();
    console.log(`[VectorStore] Found ${results.length} matching documents`);
    
    return results.length > 0;
  } catch (e) {
    console.log(`[VectorStore] documentExists error:`, e);
    // Table doesn't exist yet or filter failed
    return false;
  }
};

/* Upserts documents to the vector store (insert or update based on ID).
   @param documents - Array of documents to upsert. */
export const upsertDocuments = async (documents: Document[]) => {
  const db = await getDb();
  let table;
  
  try {
    // Try to open existing table
    table = await db.openTable('course_knowledge');
    console.log('[VectorStore] Upserting into existing table');
    
    // Use mergeInsert for upsert behavior
    await table
      .mergeInsert('id')
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .execute(documents);
      
    console.log('[VectorStore] Documents upserted successfully');
  } catch (e) {
    // Table doesn't exist, create it
    console.log('[VectorStore] Creating new table');
    table = await db.createTable('course_knowledge', documents);
    console.log('[VectorStore] Table created with initial documents');
  }
};

/* Adds documents to the vector store (deprecated - use upsertDocuments).
   @param documents - Array of documents to add. */
export const addDocuments = async (documents: Document[]) => {
  // Delegate to upsert for consistency
  return upsertDocuments(documents);
};

/* Searches for similar documents.
   @param queryVector - The vector representation of the query.
   @param limit - Number of results to return.
   @returns Array of matching documents with distance. */
export const similaritySearch = async (queryVector: number[], limit: number = 5, filter?: string) => {
  const db = await getDb();
  try {
    const table = await db.openTable('course_knowledge');
    console.log('[VectorStore] Searching with vector length:', queryVector.length);
    
    let query = table.search(queryVector).limit(limit);
    
    if (filter) {
      query = query.filter(filter);
    }
    
    const results = await query.toArray();
    console.log('[VectorStore] Raw results:', results.length, 'items');
    
    return results;
  } catch (e) {
    console.error('[VectorStore] Error searching:', e);
    return [];
  }
};

/* Retrieves all documents from the vector store.
   @returns Array of all documents with their id, text, and metadata. */
export const getAllDocuments = async () => {
  const db = await getDb();
  try {
    const table = await db.openTable('course_knowledge');
    console.log('[VectorStore] Retrieving all documents');
    const results = await table.query().select(["id", "text", "metadata"]).toArray();
    // Parse metadata back to object
    return results.map(doc => ({
      ...doc,
      metadata: JSON.parse(doc.metadata as string)
    }));
  } catch (e) {
    console.error('[VectorStore] Error retrieving all documents:', e);
    return [];
  }
};

/* Clears all documents from the vector store by dropping the table. */
export const clearVectorStore = async () => {
  const db = await getDb();
  try {
    console.log('[VectorStore] Dropping table course_knowledge...');
    await db.dropTable('course_knowledge');
    console.log('[VectorStore] Table dropped successfully.');
  } catch (e: any) {
    // Ignore "table not found" errors (expected if never synced with embeddings)
    if (e.message?.includes('was not found') || e.code === 'GenericFailure') {
      console.log('[VectorStore] Table does not exist (nothing to clear).');
    } else {
      console.error('[VectorStore] Error clearing vector store:', e);
    }
  }
};

/* Populates LanceDB vector store from SQLite embeddings.
   This should be called after sync to make items searchable by the chat agent. */
export const populateFromSQLite = async (sqliteDb: any) => {
  console.log('[VectorStore] Populating LanceDB from SQLite embeddings...');
  
  try {
    const documents: Document[] = [];
    
    // Import the embedding generator
    const { generateEmbedding } = await import('../services/synthesis/embedding_cache.js');
    
    // 1. Add course information
    const courses = sqliteDb.prepare(`
      SELECT id, canvas_id, name, course_code, term, color_hex, is_active
      FROM courses
      WHERE is_active = 1
    `).all() as Array<{
      id: string;
      canvas_id: number;
      name: string;
      course_code?: string;
      term?: string;
      color_hex?: string;
      is_active: number;
    }>;
    
    console.log(`[VectorStore] Found ${courses.length} active courses`);
    
    // Add courses with generated embeddings
    for (const course of courses) {
      const text = [
        `Course: ${course.name}`,
        course.course_code ? `Course Code: ${course.course_code}` : '',
        course.term ? `Term: ${course.term}` : '',
        `Canvas ID: ${course.canvas_id}`,
        `You are enrolled in this course`
      ].filter(Boolean).join('\n');
      
      const embedding = await generateEmbedding(text);
      
      documents.push({
        id: `course_${course.id}`,
        vector: embedding,
        text,
        metadata: JSON.stringify({
          type: 'COURSE',
          id: course.id,
          canvas_id: course.canvas_id,
          name: course.name,
          course_code: course.course_code,
          term: course.term
        })
      });
    }
    
    // 2. Add ingestion rules (external sites, modules, etc.)
    const rules = sqliteDb.prepare(`
      SELECT ir.*, c.name as course_name
      FROM ingestion_rules ir
      JOIN courses c ON ir.course_id = c.id
      WHERE c.is_active = 1
    `).all() as Array<{
      id: number;
      course_id: string;
      source_url: string;
      source_type: string;
      category?: string;
      course_name: string;
    }>;
    
    console.log(`[VectorStore] Found ${rules.length} ingestion rules`);
    
    for (const rule of rules) {
      const text = [
        `Resource: ${rule.category || 'External Link'}`,
        `Course: ${rule.course_name}`,
        `URL: ${rule.source_url}`,
        `Type: ${rule.source_type}`,
        rule.category ? `Category: ${rule.category}` : ''
      ].filter(Boolean).join('\n');
      
      const embedding = await generateEmbedding(text);
      
      documents.push({
        id: `rule_${rule.id}`,
        vector: embedding,
        text,
        metadata: JSON.stringify({
          type: 'RESOURCE',
          id: rule.id,
          course_id: rule.course_id,
          url: rule.source_url,
          category: rule.category
        })
      });
    }
    
    // 3. Get all items with embeddings from SQLite
    const items = sqliteDb.prepare(`
      SELECT ui.*, c.name as course_name, c.course_code
      FROM universal_items ui
      JOIN courses c ON ui.course_id = c.id
      WHERE ui.embedding IS NOT NULL
    `).all() as Array<{
      id: number;
      title: string;
      description?: string;
      content_url?: string;
      item_type: string;
      course_id: string;
      course_name: string;
      course_code?: string;
      due_date?: string;
      embedding: string;
      raw_content_snippet?: string;
    }>;
    
    console.log(`[VectorStore] Found ${items.length} items with embeddings`);
    
    // Transform items to LanceDB format - enrich with course context
    for (const item of items) {
      const embedding = JSON.parse(item.embedding);
      const text = [
        `${item.item_type}: ${item.title}`,
        `Course: ${item.course_name}${item.course_code ? ` (${item.course_code})` : ''}`,
        item.due_date ? `Due: ${new Date(item.due_date).toLocaleDateString()}` : '',
        item.description || '',
        item.raw_content_snippet || ''
      ].filter(Boolean).join('\n\n');
      
      documents.push({
        id: `item_${item.id}`,
        vector: embedding,
        text,
        metadata: JSON.stringify({
          id: item.id,
          title: item.title,
          type: item.item_type,
          course_id: item.course_id,
          course_name: item.course_name,
          due_date: item.due_date,
          url: item.content_url
        })
      });
    }
    
    if (documents.length === 0) {
      console.log('[VectorStore] No documents to index');
      return { success: true, count: 0 };
    }
    
    // Upsert all documents to LanceDB
    await upsertDocuments(documents);
    
    console.log(`[VectorStore] Successfully populated LanceDB with ${documents.length} documents (${courses.length} courses + ${rules.length} resources + ${items.length} items)`);
    return { success: true, count: documents.length };
  } catch (error) {
    console.error('[VectorStore] Error populating from SQLite:', error);
    throw error;
  }
};
