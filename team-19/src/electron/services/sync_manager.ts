import db from '../database/db';
import { getKeys } from './auth';
import * as canvas from './canvas';
import { synthesisManager } from './synthesis';
import { extractPdfText } from './pdf_parser';
import { populateFromSQLite } from '../database/vector-store';

// Types from database schema
interface IngestionRule {
  id: number;
  course_id: string;
  source_url: string;
  source_type: 'CANVAS_API' | 'EXTERNAL_HTML' | 'PDF_LINK' | 'GOOGLE_DRIVE';
  category: string;
  extraction_config: any; // JSON
}

interface UniversalItem {
  course_id: string;
  ingestion_rule_id?: number | null;
  item_type: 'ASSIGNMENT' | 'READING' | 'SLIDE' | 'SYLLABUS' | 'ANNOUNCEMENT' | 'QUIZ' | 'FILE' | 'PAGE' | 'DISCUSSION';
  title: string;
  description?: string;
  due_date?: string | null;
  content_url?: string;
  raw_content_snippet?: string;
  confidence_score: number;
  is_read?: boolean;
}

let syncStatus = "Idle";
let lastSyncTime: Date | null = null;

export const getSyncStatus = () => {
    if (syncStatus === "Idle" && lastSyncTime) {
        return `Last synced at ${lastSyncTime.toLocaleTimeString()}`;
    }
    return syncStatus;
};

// Mock function for external sites
const mockExternalSync = async (rule: IngestionRule): Promise<UniversalItem[]> => {
    console.log(`[Sync] Mocking external sync for rule ${rule.id} (${rule.source_url})`);
    return [];
};

const saveItem = async (item: UniversalItem) => {
    const stmt = db.prepare(`
        INSERT INTO universal_items (
            course_id, ingestion_rule_id, item_type, title, description,
            due_date, content_url, raw_content_snippet, confidence_score, is_read
        ) VALUES (
            @course_id, @ingestion_rule_id, @item_type, @title, @description,
            @due_date, @content_url, @raw_content_snippet, @confidence_score, 0
        )
    `);
    
    // Simple deduplication check
    if (item.content_url) {
        const existing = db.prepare('SELECT id FROM universal_items WHERE content_url = ? AND course_id = ?').get(item.content_url, item.course_id);
        if (existing) {
             // Update? For now, skip or update specific fields.
             db.prepare(`
                UPDATE universal_items 
                SET title = @title, description = @description, due_date = @due_date, confidence_score = @confidence_score, raw_content_snippet = @raw_content_snippet
                WHERE id = @id
             `).run({
                 ...item,
                 id: (existing as any).id
             });
             return;
        }
    }

    // Extract PDF text if this is a PDF file
    let rawContentSnippet = item.raw_content_snippet ?? null;
    const isPdf = item.item_type === 'FILE' && (
        item.title.toLowerCase().endsWith('.pdf') || 
        (item.content_url && item.content_url.toLowerCase().includes('.pdf'))
    );
    
    if (isPdf && item.content_url) {
        console.log(`[Sync] Detected PDF file: ${item.title}, extracting text...`);
        const pdfText = await extractPdfText(item.content_url);
        if (pdfText) {
            rawContentSnippet = pdfText;
            console.log(`[Sync] Extracted ${pdfText.length} characters from ${item.title}`);
        } else {
            console.warn(`[Sync] Failed to extract text from ${item.title}`);
        }
    }

    stmt.run({
        ...item,
        ingestion_rule_id: item.ingestion_rule_id ?? null,
        description: item.description ?? null,
        due_date: item.due_date ?? null,
        content_url: item.content_url ?? null,
        raw_content_snippet: rawContentSnippet
    });
};

const extractLinkedFiles = (html: string): { courseId: string, fileId: string }[] => {
    if (!html) return [];
    // Matches /courses/:courseId/files/:fileId
    // Captures group 1: courseId, group 2: fileId
    const regex = /courses\/(\d+)\/files\/(\d+)/g;
    const matches = [...html.matchAll(regex)];
    return matches.map(m => ({ courseId: m[1], fileId: m[2] }));
};

const processLinkedFiles = async (currentCourseId: string, html: string, processedFiles: Set<string>) => {
    if (!html) return;
    const links = extractLinkedFiles(html);
    const keys = getKeys();
    if (!keys.canvasToken || !keys.canvasDomain) return;

    for (const { courseId, fileId } of links) {
        // Use the linked courseId if found, otherwise fallback to current
        // (Regex enforces finding a courseId, but good to be explicit)
        const targetCourseId = courseId || currentCourseId;
        const uniqueKey = `${targetCourseId}:${fileId}`;

        if (processedFiles.has(uniqueKey)) continue;
        processedFiles.add(uniqueKey);

        try {
             // Check if we already have this file content_url in DB to avoid fetch?
             // Since we don't know the exact download URL without fetching metadata, 
             // and checking strict content_url requires it.
             // However, we can check if we have a FILE item for this course with a content_url containing the fileId?
             // That's tricky. Let's just fetch metadata. It's lightweight compared to downloading content.
             
             const file = await canvas.fetchFile(keys.canvasDomain, keys.canvasToken, targetCourseId, fileId);
             if (file) {
                 await saveItem({
                     course_id: targetCourseId,
                     item_type: 'FILE',
                     title: file.display_name,
                     description: '',
                     due_date: null,
                     content_url: file.url,
                     confidence_score: 1.0,
                     raw_content_snippet: ''
                 });
             }
        } catch (e) {
            console.error(`[Sync] Error processing linked file ${fileId} in course ${courseId}:`, e);
        }
    }
};

export const syncCourseStandardData = async (courseId: string) => {
    const keys = getKeys();
    if (!keys.canvasToken || !keys.canvasDomain) {
        console.warn('[Sync] Missing Canvas credentials');
        return;
    }

    console.log(`[Sync] Syncing standard data for course ${courseId}`);
    const processedFiles = new Set<string>();

    // 1. Assignments
    try {
        const assignments = await canvas.fetchAssignments(keys.canvasDomain, keys.canvasToken, courseId);
        for (const a of assignments) {
            await saveItem({
                course_id: courseId,
                item_type: 'ASSIGNMENT',
                title: a.name,
                description: a.description || '',
                due_date: a.due_at,
                content_url: a.html_url,
                confidence_score: 1.0,
                raw_content_snippet: '',
                ingestion_rule_id: null
            });
            await processLinkedFiles(courseId, a.description, processedFiles);
        }
    } catch (e) {
        console.error(`[Sync] Error fetching assignments for ${courseId}:`, e);
    }

    // 2. Announcements
    try {
        const announcements = await canvas.fetchAnnouncements(keys.canvasDomain, keys.canvasToken, courseId);
        for (const a of announcements) {
             await saveItem({
                course_id: courseId,
                item_type: 'ANNOUNCEMENT',
                title: a.title,
                description: a.message || '',
                due_date: a.posted_at, 
                content_url: a.html_url,
                confidence_score: 1.0,
                raw_content_snippet: '',
                ingestion_rule_id: null
            });
            await processLinkedFiles(courseId, a.message, processedFiles);
        }
    } catch (e) {
        console.error(`[Sync] Error fetching announcements for ${courseId}:`, e);
    }
    
    // 3. Quizzes
    try {
        const quizzes = await canvas.fetchQuizzes(keys.canvasDomain, keys.canvasToken, courseId);
        for (const q of quizzes) {
             await saveItem({
                course_id: courseId,
                item_type: 'QUIZ',
                title: q.title,
                description: q.description || '',
                due_date: q.due_at,
                content_url: q.html_url,
                confidence_score: 1.0,
                raw_content_snippet: '',
                ingestion_rule_id: null
            });
            if (q.description) {
                await processLinkedFiles(courseId, q.description, processedFiles);
            }
        }
    } catch (e) {
        console.error(`[Sync] Error fetching quizzes for ${courseId}:`, e);
    }

    // 4. Discussions
    try {
        const discussions = await canvas.fetchDiscussionTopics(keys.canvasDomain, keys.canvasToken, courseId);
        for (const d of discussions) {
             await saveItem({
                course_id: courseId,
                item_type: 'DISCUSSION',
                title: d.title,
                description: d.message || '',
                due_date: d.posted_at,
                content_url: d.html_url,
                confidence_score: 1.0,
                raw_content_snippet: '',
                ingestion_rule_id: null
            });
            if (d.message) {
                await processLinkedFiles(courseId, d.message, processedFiles);
            }
        }
    } catch (e) {
        console.error(`[Sync] Error fetching discussions for ${courseId}:`, e);
    }
};

export const processIngestionRules = async () => {
    const rules = db.prepare('SELECT * FROM ingestion_rules').all() as IngestionRule[];
    const keys = getKeys();
    
    console.log(`[Sync] Processing ${rules.length} ingestion rules`);

    for (const rule of rules) {
        try {
            if (rule.source_type === 'CANVAS_API') {
                if (!keys.canvasToken || !keys.canvasDomain) continue;

                const courseId = rule.course_id;
                
                if (rule.category === 'files' || rule.category === 'resources') {
                    const files = await canvas.fetchFiles(keys.canvasDomain, keys.canvasToken, courseId);
                    for (const f of files) {
                        await saveItem({
                            course_id: courseId,
                            ingestion_rule_id: rule.id,
                            item_type: 'FILE',
                            title: f.display_name,
                            description: '',
                            due_date: null,
                            content_url: f.url,
                            confidence_score: 1.0,
                            raw_content_snippet: ''
                        });
                    }
                } else if (rule.category === 'modules') {
                    const modules = await canvas.fetchModules(keys.canvasDomain, keys.canvasToken, courseId);
                    for (const m of modules) {
                        for (const item of m.items) {
                             await saveItem({
                                course_id: courseId,
                                ingestion_rule_id: rule.id,
                                item_type: 'READING', 
                                title: item.title,
                                description: '',
                                due_date: null,
                                content_url: item.html_url,
                                confidence_score: 0.9,
                                raw_content_snippet: ''
                            });
                        }
                    }
                }
            } else {
                const items = await mockExternalSync(rule);
                for (const item of items) {
                    await saveItem({ ...item, course_id: rule.course_id, ingestion_rule_id: rule.id });
                }
            }
            
            db.prepare('UPDATE ingestion_rules SET last_checked_at = CURRENT_TIMESTAMP WHERE id = ?').run(rule.id);
            
        } catch (error) {
            console.error(`[Sync] Error processing rule ${rule.id}:`, error);
        }
    }
};

export const syncAll = async () => {
    if (syncStatus === "Syncing") return { success: false, message: "Sync already in progress" };
    syncStatus = "Syncing";
    
    try {
        const courses = db.prepare('SELECT id FROM courses WHERE is_active = 1').all() as { id: string }[];
        
        console.log('[Sync] Starting full sync...');
        
        for (const course of courses) {
            await syncCourseStandardData(course.id);
        }
        
        await processIngestionRules();

        // Trigger Synthesis Asynchronously (don't await)
        (async () => {
            console.log('[Sync] Triggering background synthesis...');
            for (const course of courses) {
                await synthesisManager.processCourse(course.id);
            }
            
            // After synthesis completes, populate LanceDB for chat queries
            console.log('[Sync] Populating LanceDB vector store...');
            try {
                await populateFromSQLite(db);
                console.log('[Sync] LanceDB population completed');
            } catch (error) {
                console.error('[Sync] Failed to populate LanceDB:', error);
            }
        })();
        
        console.log('[Sync] Full sync completed.');
        lastSyncTime = new Date();
        syncStatus = "Idle";
        return { success: true };
    } catch (e: any) {
        syncStatus = "Error";
        console.error('[Sync] Error:', e);
        return { success: false, error: e.message };
    }
};