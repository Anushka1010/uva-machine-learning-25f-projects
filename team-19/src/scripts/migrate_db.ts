import Database from 'better-sqlite3';
import path from 'node:path';
import fs from 'node:fs';
import { app } from 'electron';

// We need to find the user data path. Since we are running this as a script, we can't easily use 'electron'.
// We will assume the default path for macOS for now based on the prompt info or just try to locate it.
// The user provided the temporary directory but not the full userdata path in the prompt context,
// but the 'wipe' script in package.json gives a hint: "$HOME/Library/Application Support/canvasgpt"

const dbPath = path.join(process.env.HOME || '', 'Library/Application Support/canvasgpt/database.sqlite');

if (fs.existsSync(dbPath)) {
    console.log(`Migrating database at ${dbPath}`);
    const db = new Database(dbPath);
    try {
        db.prepare("ALTER TABLE realised_entities ADD COLUMN review_status TEXT DEFAULT 'needs_review'").run();
        console.log("Added review_status column.");
    } catch (e: any) {
        if (!e.message.includes('duplicate column name')) {
            console.error("Error adding review_status:", e);
        } else {
            console.log("review_status column already exists.");
        }
    }

    try {
        db.prepare("ALTER TABLE realised_entities ADD COLUMN synthesis_log JSON").run();
        console.log("Added synthesis_log column.");
    } catch (e: any) {
        if (!e.message.includes('duplicate column name')) {
            console.error("Error adding synthesis_log:", e);
        } else {
             console.log("synthesis_log column already exists.");
        }
    }
} else {
    console.log("Database not found at default path, skipping migration (tables will be created fresh on next run).");
}
