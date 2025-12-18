#!/usr/bin/env node

/* CanvasGPT MCP Server - Standalone Version
   Runs outside of Electron, directly accessing the SQLite database */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import initSqlJs from 'sql.js';
import { OpenAIEmbeddings } from "@langchain/openai";
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';

// Get database path (same location as Electron app uses)
const dbPath = path.join(
  os.homedir(),
  'Library', 
  'Application Support',
  'canvasgpt',
  'database.sqlite'
);

if (!fs.existsSync(dbPath)) {
  console.error(`[MCP Error] Database not found at: ${dbPath}`);
  console.error(`[MCP Error] Please run the CanvasGPT app first to create the database`);
  process.exit(1);
}

// Initialize sql.js
const SQL = await initSqlJs();
const buffer = fs.readFileSync(dbPath);
const db = new SQL.Database(buffer);

// Helper function to execute queries (sql.js has different API than better-sqlite3)
function query(sql: string, params: any[] = []): any[] {
  const stmt = db.prepare(sql);
  if (params.length > 0) {
    stmt.bind(params);
  }
  const results: any[] = [];
  while (stmt.step()) {
    const row = stmt.getAsObject();
    results.push(row);
  }
  stmt.free();
  return results;
}

function queryOne(sql: string, params: any[] = []): any | null {
  const results = query(sql, params);
  return results.length > 0 ? results[0] : null;
}

// Initialize MCP Server
const server = new Server(
  {
    name: "canvasGPT",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Helper: Get OpenAI key from database
function getOpenAIKey(): string | null {
  try {
    // The auth service stores keys in electron-store, which is a separate JSON file
    const storePath = path.join(
      os.homedir(),
      'Library',
      'Application Support',
      'canvasgpt',
      'config.json'
    );
    
    if (!fs.existsSync(storePath)) {
      return null;
    }
    
    const config = JSON.parse(fs.readFileSync(storePath, 'utf-8'));
    return config.OPENAI_API_KEY || null;
  } catch (error) {
    console.error('[MCP] Error reading API key:', error);
    return null;
  }
}

// Helper: Calculate cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Tool 1: Search Materials
async function searchMaterials(searchQuery: string, courseId?: string): Promise<string> {
  try {
    const apiKey = getOpenAIKey();
    if (!apiKey) {
      return "Error: OpenAI API key not configured. Please set it up in the CanvasGPT app.";
    }

    // Generate embedding for the query
    const embeddings = new OpenAIEmbeddings({
      apiKey,
      modelName: "text-embedding-3-small",
    });

    const queryVector = await embeddings.embedQuery(searchQuery);

    // Get all items with embeddings
    let sql = `
      SELECT id, title, course_id, item_type, content_url, embedding
      FROM universal_items
      WHERE embedding IS NOT NULL
    `;
    
    if (courseId) {
      sql += ` AND course_id = ?`;
    }
    
    const items = courseId ? query(sql, [courseId]) : query(sql);

    if (!items || items.length === 0) {
      return "No materials found in the database.";
    }

    // Calculate similarities
    const results = items
      .map((item: any) => {
        const embedding = JSON.parse(item.embedding);
        const similarity = cosineSimilarity(queryVector, embedding);
        return { ...item, similarity };
      })
      .filter((item: any) => item.similarity > 0.5)
      .sort((a: any, b: any) => b.similarity - a.similarity)
      .slice(0, 5);

    if (results.length === 0) {
      return `No materials found matching "${searchQuery}"`;
    }

    // Format results
    const formatted = results
      .map((r: any, i: number) => {
        const lines = [
          `${i + 1}. **${r.title || "Untitled"}** (${r.item_type})`,
          `   Course ID: ${r.course_id}`,
          `   URL: ${r.content_url || "N/A"}`,
          `   Similarity: ${(r.similarity * 100).toFixed(1)}%`,
        ];
        return lines.join("\n");
      })
      .join("\n\n");

    return `Found ${results.length} materials matching "${searchQuery}":\n\n${formatted}`;
  } catch (error) {
    console.error("[MCP] Error in searchMaterials:", error);
    return `Error searching materials: ${error}`;
  }
}

// Tool 2: Get Upcoming Assignments
async function getUpcomingAssignments(days: number = 7): Promise<string> {
  try {
    const now = new Date();
    const futureDate = new Date(now.getTime() + days * 24 * 60 * 60 * 1000);

    const assignments = query(`
      SELECT title, course_id, due_date, content_url
      FROM universal_items
      WHERE item_type = 'ASSIGNMENT'
        AND due_date IS NOT NULL
        AND datetime(due_date) BETWEEN datetime('now') AND datetime(?, 'unixepoch')
      ORDER BY datetime(due_date) ASC
    `, [Math.floor(futureDate.getTime() / 1000)]);

    if (!assignments || assignments.length === 0) {
      return `No assignments due in the next ${days} days.`;
    }

    const formatted = assignments
      .map((a: any, i: number) => {
        const dueDate = new Date(a.due_date);
        const lines = [
          `${i + 1}. **${a.title}**`,
          `   Course ID: ${a.course_id}`,
          `   Due: ${dueDate.toLocaleString()}`,
          `   URL: ${a.content_url || "N/A"}`,
        ];
        return lines.join("\n");
      })
      .join("\n\n");

    return `Found ${assignments.length} upcoming assignments:\n\n${formatted}`;
  } catch (error) {
    console.error("[MCP] Error in getUpcomingAssignments:", error);
    return `Error getting assignments: ${error}`;
  }
}

// Tool 3: Get Course Overview
async function getCourseOverview(courseId?: string): Promise<string> {
  try {
    let courses;

    if (courseId) {
      courses = query(`
        SELECT c.id, c.name, c.course_code,
               COUNT(DISTINCT CASE WHEN ui.item_type = 'ASSIGNMENT' THEN ui.id END) as assignment_count,
               COUNT(DISTINCT CASE WHEN ui.item_type = 'READING' THEN ui.id END) as reading_count,
               COUNT(DISTINCT CASE WHEN ui.item_type = 'SLIDE' THEN ui.id END) as slide_count
        FROM courses c
        LEFT JOIN universal_items ui ON ui.course_id = c.id
        WHERE c.id = ?
        GROUP BY c.id
      `, [courseId]);
    } else {
      courses = query(`
        SELECT c.id, c.name, c.course_code,
               COUNT(DISTINCT CASE WHEN ui.item_type = 'ASSIGNMENT' THEN ui.id END) as assignment_count,
               COUNT(DISTINCT CASE WHEN ui.item_type = 'READING' THEN ui.id END) as reading_count,
               COUNT(DISTINCT CASE WHEN ui.item_type = 'SLIDE' THEN ui.id END) as slide_count
        FROM courses c
        LEFT JOIN universal_items ui ON ui.course_id = c.id
        GROUP BY c.id
      `);
    }

    if (!courses || courses.length === 0) {
      return courseId
        ? `No course found with ID: ${courseId}`
        : "No courses found in database.";
    }

    const formatted = courses
      .map((c: any, i: number) => {
        const lines = [
          `${i + 1}. **${c.name}** (${c.course_code || 'N/A'})`,
          `   Course ID: ${c.id}`,
          `   Assignments: ${c.assignment_count}`,
          `   Readings: ${c.reading_count}`,
          `   Slides: ${c.slide_count}`,
        ];
        return lines.join("\n");
      })
      .join("\n\n");

    return courseId
      ? `Course overview:\n\n${formatted}`
      : `Found ${courses.length} courses:\n\n${formatted}`;
  } catch (error) {
    console.error("[MCP] Error in getCourseOverview:", error);
    return `Error getting course overview: ${error}`;
  }
}

// Tool 4: Get Item Details
async function getItemDetails(itemId: string): Promise<string> {
  try {
    const item = queryOne(`
      SELECT *
      FROM universal_items
      WHERE id = ?
    `, [itemId]);

    if (!item) {
      return `No item found with ID: ${itemId}`;
    }

    const lines = [
      `**${item.title || "Untitled"}**`,
      `Type: ${item.item_type}`,
      `Course ID: ${item.course_id}`,
      `URL: ${item.content_url || "N/A"}`,
    ];

    if (item.due_date) {
      lines.push(`Due: ${new Date(item.due_date).toLocaleString()}`);
    }

    if (item.confidence_score) {
      lines.push(`Confidence: ${(item.confidence_score * 100).toFixed(1)}%`);
    }

    if (item.description) {
      lines.push(`\nDescription:\n${item.description}`);
    }

    if (item.raw_content_snippet) {
      lines.push(`\nContent Preview:\n${item.raw_content_snippet.substring(0, 500)}...`);
    }

    return lines.join("\n");
  } catch (error) {
    console.error("[MCP] Error in getItemDetails:", error);
    return `Error getting item details: ${error}`;
  }
}

// Register tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "search_materials",
        description: "Search for course materials (assignments, pages, files) using semantic search",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query (e.g., 'linear algebra homework', 'lecture notes on sorting')",
            },
            courseId: {
              type: "string",
              description: "Optional: Filter by specific course ID",
            },
          },
          required: ["query"],
        },
      },
      {
        name: "get_upcoming_assignments",
        description: "Get assignments due in the next N days",
        inputSchema: {
          type: "object",
          properties: {
            days: {
              type: "number",
              description: "Number of days to look ahead (default: 7)",
              default: 7,
            },
          },
        },
      },
      {
        name: "get_course_overview",
        description: "Get overview of all courses or a specific course",
        inputSchema: {
          type: "object",
          properties: {
            courseId: {
              type: "string",
              description: "Optional: Get overview of a specific course",
            },
          },
        },
      },
      {
        name: "get_item_details",
        description: "Get detailed information about a specific item by ID",
        inputSchema: {
          type: "object",
          properties: {
            itemId: {
              type: "string",
              description: "The ID of the item to retrieve",
            },
          },
          required: ["itemId"],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "search_materials": {
        if (!args || typeof args.query !== "string") {
          throw new Error("query parameter is required");
        }
        const result = await searchMaterials(
          args.query,
          args.courseId as string | undefined
        );
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "get_upcoming_assignments": {
        const days = args?.days as number | undefined;
        const result = await getUpcomingAssignments(days);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "get_course_overview": {
        const courseId = args?.courseId as string | undefined;
        const result = await getCourseOverview(courseId);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "get_item_details": {
        if (!args || typeof args.itemId !== "string") {
          throw new Error("itemId parameter is required");
        }
        const result = await getItemDetails(args.itemId);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    console.error(`[MCP] Error executing tool ${name}:`, error);
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[MCP] CanvasGPT server started successfully");
  console.error(`[MCP] Database: ${dbPath}`);
  console.error(`[MCP] API Key configured: ${getOpenAIKey() ? "Yes" : "No"}`);
}

main().catch((error) => {
  console.error("[MCP] Fatal error:", error);
  process.exit(1);
});
