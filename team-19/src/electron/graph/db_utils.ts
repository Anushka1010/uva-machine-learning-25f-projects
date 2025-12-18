import db from "../database/db";

export type IngestionRuleInput = {
  source_url: string;
  source_type: string;
  category?: string;
  extraction_config?: Record<string, unknown>;
  check_frequency_hours?: number;
};

export type IngestionRuleRow = {
  id: number;
  course_id: string;
  source_url: string;
  source_type: string;
  category?: string | null;
  extraction_config: Record<string, unknown>;
  check_frequency_hours?: number;
  last_checked_at: string | null;
};

export const getDatabasePath = (): string => {
  // better-sqlite3 exposes the underlying filename via the `name` property.
  const name = (db as unknown as { name?: string }).name;
  return name ?? "";
};

const ensureTableExists = (): void => {
  // Matches the schema defined in electron/database/db.ts
  const createTableSql = `
    CREATE TABLE IF NOT EXISTS ingestion_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id TEXT NOT NULL,
        source_url TEXT NOT NULL,
        source_type TEXT NOT NULL,
        category TEXT,
        extraction_config JSON,
        check_frequency_hours INTEGER DEFAULT 1,
        last_checked_at DATETIME,
        FOREIGN KEY(course_id) REFERENCES courses(id)
    );
  `;

  db.exec(createTableSql);

  // Backfill: add category column if the table already exists without it.
  const columns = db.prepare(`PRAGMA table_info(ingestion_rules)`).all() as Array<{ name: string }>;
  const hasCategory = columns.some((col) => col.name === "category");
  if (!hasCategory) {
    db.exec(`ALTER TABLE ingestion_rules ADD COLUMN category TEXT;`);
  }
};

export const saveResultsToDb = (courseId: string, rules: IngestionRuleInput[]): number => {
  ensureTableExists();

  const insertSql = `
    INSERT INTO ingestion_rules (
      course_id,
      source_url,
      source_type,
      category,
      extraction_config,
      check_frequency_hours
    ) VALUES (@course_id, @source_url, @source_type, @category, @extraction_config, @check_frequency_hours)
  `;

  const insertStmt = db.prepare(insertSql);
  let insertedCount = 0;

  const runTransaction = db.transaction((ruleList: IngestionRuleInput[]) => {
    for (const rule of ruleList) {
      insertStmt.run({
        course_id: courseId,
        source_url: rule.source_url ?? "",
        source_type: rule.source_type ?? "EXTERNAL_HTML",
        category: rule.category ?? null,
        extraction_config: JSON.stringify(rule.extraction_config ?? {}),
        check_frequency_hours: rule.check_frequency_hours ?? 1,
      });
      insertedCount += 1;
    }
  });

  runTransaction(rules);
  return insertedCount;
};

export const getRulesForCourse = (courseId: string): IngestionRuleRow[] => {
  ensureTableExists();

  const rows = db
    .prepare(
      `
      SELECT
        id,
        course_id,
        source_url,
        source_type,
        category,
        extraction_config,
        check_frequency_hours,
        last_checked_at
      FROM ingestion_rules
      WHERE course_id = ?
    `,
    )
    .all(courseId) as Array<Omit<IngestionRuleRow, "extraction_config"> & { extraction_config: string | null }>;

  return rows.map((row) => ({
    ...row,
    extraction_config: row.extraction_config ? JSON.parse(row.extraction_config) : {},
    category: row.category ?? null,
  }));
};

export const deleteRulesForCourse = (courseId: string): number => {
  ensureTableExists();

  const result = db.prepare("DELETE FROM ingestion_rules WHERE course_id = ?").run(courseId);
  return result.changes;
};
