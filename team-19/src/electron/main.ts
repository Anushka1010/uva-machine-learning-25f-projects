import 'dotenv/config'
import { app, BrowserWindow, ipcMain, shell } from 'electron'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import axios from 'axios'
import { saveKeys, getKeys, clearKeys, AuthKeys } from './services/auth'
import { agent } from './graph/agent'
import { discoveryAgent } from './graph/discovery_agent'
import { saveResultsToDb } from './graph/db_utils'
import { HumanMessage } from "@langchain/core/messages";
import { clearVectorStore} from './database/vector-store';
import db from './database/db';
import { syncAll } from './services/sync_manager';

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// The built directory structure
//
// ├─┬─┬ dist
// │ │ └── index.html
// │ │
// │ ├─┬ dist-electron
// │ │ ├── main.js
// │ │ └── preload.mjs
// │
process.env.APP_ROOT = path.join(__dirname, '..')

// 🚧 Use ['ENV_NAME'] avoid vite:define plugin - Vite@2.x
export const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
export const MAIN_DIST = path.join(process.env.APP_ROOT, 'dist-electron')
export const RENDERER_DIST = path.join(process.env.APP_ROOT, 'dist')

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL ? path.join(process.env.APP_ROOT, 'public') : RENDERER_DIST

let win: BrowserWindow | null

function createWindow() {
  win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, 'canvas_reversed_logo.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
      
    },
    'minWidth': 600,
    'minHeight': 800,
    'width': 600,
    'height': 800,
    show: false,
    titleBarStyle: 'hidden',
    // titleBarOverlay: {
    //   color: '#2d2d30',
    //   symbolColor: '#74b1be',
    // },
  })

  win.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url);
    return { action: 'deny' };
  });

  win.on('ready-to-show', () => {
    win?.show()
  })

  // Test active push message to Renderer-process.
  win.webContents.on('did-finish-load', () => {
    win?.webContents.send('main-process-message', (new Date).toLocaleString())
  })

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL)
  } else {
    // win.loadFile('dist/index.html')
    win.loadFile(path.join(RENDERER_DIST, 'index.html'))
  }
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
    win = null
  }
})

app.on('activate', () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

ipcMain.on('save-keys', (_event, keys: AuthKeys) => {
  saveKeys(keys)
})

ipcMain.handle('get-keys', () => {
  return getKeys()
})

ipcMain.on('clear-keys', () => {
  clearKeys()
})

ipcMain.handle('ask-question', async (_event, query: string) => {
  try {
    const response = await agent.invoke({
      messages: [new HumanMessage(query)],
    });

    const lastMessage = response.messages[response.messages.length - 1];
    return typeof lastMessage.content === 'string' ? lastMessage.content : 'No response generated.';
  } catch (error: any) {
    console.error('Agent failed:', error);
    return `Error: ${error.message}`;
  }
})

ipcMain.handle('set-window-size', (_event, { width, height, minWidth, minHeight }) => {
  if (win) {
    win.setMinimumSize(minWidth, minHeight);
    win.setSize(width, height);
    win.center(); // Center the window after resizing
  }
});

ipcMain.handle('delete-all-data', async () => {
  try {
    console.log('[Main] Deleting all data...');

    // 1. Clear SQLite tables

    // Disable foreign keys to avoid constraint violations during deletion
    db.prepare('PRAGMA foreign_keys = OFF;').run();

    //get all table names
    const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';").all().map((row: any) => row.name);
    for (const table of tables) {
      db.prepare(`DELETE FROM ${table}`).run();
    }
    
    // Re-enable foreign keys
    db.prepare('PRAGMA foreign_keys = ON;').run();

    console.log('[Main] SQLite tables cleared.');

    // 2. Clear LanceDB
    await clearVectorStore();
    console.log('[Main] LanceDB cleared.');

    return { success: true };
  } catch (error: any) {
    console.error('[Main] Failed to delete all data:', error);
    return { success: false, error: error.message };
  }
});

type CanvasCourse = {
  id: number;
  name: string;
  course_code?: string;
  term?: { name?: string } | null;
  course_color?: string | null;
};

type UniversalItemRow = {
  id: number;
  course_id: string;
  ingestion_rule_id?: number | null;
  item_type: string;
  title: string;
  description?: string | null;
  due_date?: string | null;
  content_url?: string | null;
  raw_content_snippet?: string | null;
  confidence_score?: number | null;
  is_read?: number | boolean | null;
  created_at?: string | null;
};

const parseNextLink = (linkHeader?: string): string | null => {
  if (!linkHeader) return null;

  const links = linkHeader.split(',');
  for (const link of links) {
    const match = link.match(/<([^>]+)>; rel="next"/);
    if (match?.[1]) {
      return match[1];
    }
  }
  return null;
};

const fetchCanvasCourses = async (): Promise<CanvasCourse[]> => {
  const { canvasToken, canvasDomain } = getKeys();
  if (!canvasToken || !canvasDomain) {
    throw new Error('Canvas credentials are missing. Please add your Canvas token and domain.');
  }

  const baseUrl = canvasDomain.startsWith('http') ? canvasDomain : `https://${canvasDomain}`;
  const client = axios.create({
    baseURL: baseUrl,
    headers: {
      Authorization: `Bearer ${canvasToken}`,
    },
  });

  const collected: CanvasCourse[] = [];
  let nextUrl: string | null = `/api/v1/courses?per_page=50&enrollment_state=active`;

  while (nextUrl) {
    const response = await client.get(nextUrl);
    const data = Array.isArray(response.data) ? response.data : [];
    collected.push(...(data as CanvasCourse[]));

    const linkHeader = response.headers?.link as string | undefined;
    nextUrl = parseNextLink(linkHeader);

    // If Canvas returns an absolute URL for the next page, strip the domain so axios keeps the baseURL.
    if (nextUrl && nextUrl.startsWith(baseUrl)) {
      nextUrl = nextUrl.replace(baseUrl, '');
    }
  }

  return collected;
};

const upsertCourses = (courses: CanvasCourse[]) => {
  const stmt = db.prepare(`
    INSERT INTO courses (id, canvas_id, name, course_code, term, color_hex, is_active)
    VALUES (@id, @canvas_id, @name, @course_code, @term, @color_hex, 1)
    ON CONFLICT(id) DO UPDATE SET
      canvas_id=excluded.canvas_id,
      name=excluded.name,
      course_code=excluded.course_code,
      term=excluded.term,
      color_hex=excluded.color_hex,
      is_active=1
  `);

  const tx = db.transaction((rows: CanvasCourse[]) => {
    for (const course of rows) {
      stmt.run({
        id: String(course.id),
        canvas_id: course.id,
        name: course.name ?? `Course ${course.id}`,
        course_code: course.course_code ?? null,
        term: course.term?.name ?? null,
        color_hex: course.course_color ?? null,
      });
    }
  });

  tx(courses);
};

const parseExtractionConfig = (raw: unknown) => {
  if (raw === null || raw === undefined) return {};
  if (typeof raw === "object") return raw as Record<string, unknown>;
  if (typeof raw === "string" && raw.trim() === "") return {};
  try {
    return JSON.parse(raw as string);
  } catch {
    return {};
  }
};

ipcMain.handle('get-courses', () => {
  try {
    const rows = db
      .prepare(
        `
        SELECT id, canvas_id, name, course_code, term, color_hex, is_active
        FROM courses
        ORDER BY name COLLATE NOCASE
      `,
      )
      .all();

    return rows.map((row: any) => ({
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

ipcMain.handle('get-course-details', (_event, courseId: string) => {
  try {
    const course = db
      .prepare(
        `
        SELECT id, canvas_id, name, course_code, term, color_hex, is_active
        FROM courses
        WHERE id = ?
      `,
      )
      .get(courseId);

    const rules = db
      .prepare(
        `
        SELECT id, source_url, source_type, category, extraction_config, check_frequency_hours, last_checked_at
        FROM ingestion_rules
        WHERE course_id = ?
        ORDER BY id DESC
      `,
      )
      .all(courseId)
      .map((row: any) => ({
        ...row,
        extraction_config: parseExtractionConfig(row.extraction_config),
      }));

    const items = db
      .prepare(
        `
        SELECT
          ui.*,
          c.name as course_name,
          c.course_code as course_code
        FROM universal_items ui
        LEFT JOIN courses c ON ui.course_id = c.id
        WHERE ui.course_id = ?
        ORDER BY
          CASE WHEN ui.due_date IS NULL THEN 1 ELSE 0 END,
          ui.due_date ASC,
          ui.created_at DESC
      `,
      )
      .all(courseId)
      .map((row: any) => ({
        ...row,
        is_read: Boolean(row.is_read ?? 0),
      })) as Array<UniversalItemRow & { course_name?: string; course_code?: string }>;

    return { course, rules, items };
  } catch (error: any) {
    console.error('[Course] Failed to fetch course details:', error);
    return { course: null, rules: [], items: [] };
  }
});

ipcMain.handle('get-upcoming-items', () => {
  try {
    const rows = db
      .prepare(
        `
        SELECT
          ui.*,
          c.name as course_name,
          c.course_code as course_code
        FROM universal_items ui
        JOIN courses c ON ui.course_id = c.id
        WHERE ui.due_date IS NOT NULL
        ORDER BY ui.due_date ASC
        LIMIT 50
      `,
      )
      .all();

    return rows.map((row: any) => ({
      id: row.id,
      course_id: row.course_id,
      item_type: row.item_type,
      title: row.title,
      date: row.due_date,
      course_name: row.course_name,
      course_code: row.course_code,
      description: row.description ?? '',
      content_url: row.content_url ?? '',
      confidence_score: row.confidence_score ?? null,
    }));
  } catch (error: any) {
    console.error('[Upcoming] Failed to fetch upcoming items:', error);
    return [];
  }
});

ipcMain.handle('get-triage-items', () => {
  try {
    const rows = db
      .prepare(
        `
        SELECT
          ui.*,
          c.name as course_name,
          c.course_code as course_code
        FROM universal_items ui
        LEFT JOIN courses c ON ui.course_id = c.id
        ORDER BY ui.created_at DESC
        LIMIT 100
      `,
      )
      .all() as Array<UniversalItemRow & { course_name?: string; course_code?: string }>;

    const triageItems = rows
      .map((row) => {
        const reasons: string[] = [];
        if (!row.due_date && row.item_type === 'ASSIGNMENT') {
          reasons.push('Missing due date');
        }
        if (!row.content_url) {
          reasons.push('Missing source link');
        }
        if ((row.confidence_score ?? 1) < 0.6) {
          reasons.push('Low confidence');
        }
        if (!row.description && !row.raw_content_snippet) {
          reasons.push('No description');
        }

        return {
          id: row.id,
          item_type: row.item_type,
          title: row.title,
          course_name: (row.course_name ?? 'Unknown course') as string,
          course_code: (row.course_code ?? '') as string,
          reasons,
        };
      })
      .filter((item) => item.reasons.length > 0);

    return triageItems;
  } catch (error: any) {
    console.error('[Triage] Failed to fetch triage items:', error);
    return [];
  }
});

ipcMain.handle('get-universal-item', (_event, itemId: number) => {
  try {
    const row = db
      .prepare(
        `
        SELECT
          ui.*,
          c.name as course_name,
          c.course_code as course_code
        FROM universal_items ui
        LEFT JOIN courses c ON ui.course_id = c.id
        WHERE ui.id = ?
      `,
      )
      .get(itemId) as (UniversalItemRow & { course_name?: string; course_code?: string }) | undefined;

    if (!row) return null;

    return {
      ...row,
      is_read: Boolean(row.is_read ?? 0),
    };
  } catch (error: any) {
    console.error('[Item] Failed to fetch universal item:', error);
    return null;
  }
});

ipcMain.handle('discover-courses', async () => {
  try {
    // Always refresh the local course catalog from Canvas before running discovery.
    const canvasCourses = await fetchCanvasCourses();
    if (canvasCourses.length) {
      upsertCourses(canvasCourses);
    }

    const courses = db
      .prepare("SELECT id, name FROM courses")
      .all() as Array<{ id: string | number; name?: string }>;

    if (!courses.length) {
      return { success: false, message: "No courses stored. Please add courses before running discovery." };
    }

    const results: Array<{
      courseId: string;
      name?: string;
      archetype: string | null;
      foundLinks: string[];
      savedRules: number;
    }> = [];

    for (const course of courses) {
      const courseId = String(course.id);
      const state = await discoveryAgent.invoke({
        courseId,
        fullContext: "",
      });

      const rules = Array.isArray(state.finalRules) ? state.finalRules : [];
      const savedRules = rules.length ? saveResultsToDb(courseId, rules) : 0;

      results.push({
        courseId,
        name: course.name,
        archetype: null,
        foundLinks: [],
        savedRules,
      });
    }

    return { success: true, results };
  } catch (error: any) {
    console.error('[Discovery] Failed to run discovery agent:', error);
    return { success: false, message: error.message };
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

ipcMain.handle('open-external', async (_event, url: string) => {
  try {
    await shell.openExternal(url);
  } catch (error) {
    console.error('Failed to open external URL:', error);
    throw error;
  }
});

app.whenReady().then(createWindow)
