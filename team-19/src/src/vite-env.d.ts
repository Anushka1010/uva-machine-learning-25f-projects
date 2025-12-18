/// <reference types="vite/client" />

interface Window {
  ipcRenderer: {
    on: (channel: string, func: (...args: any[]) => void) => () => void
    send: (channel: string, ...args: any[]) => void
    invoke: (channel: string, ...args: any[]) => Promise<any>
  }
  canvasGPT: {
    login: (tokens: { canvasToken: string; notionToken: string; canvasDomain: string; notionDbId: string; openaiKey?: string }) => void
    getCourses: () => Promise<Array<{
      id: string;
      canvas_id?: number;
      name: string;
      course_code?: string;
      term?: string;
      color_hex?: string;
      is_active?: boolean;
    }>>
    getCourseDetails: (courseId: string) => Promise<{
      course: {
        id: string;
        canvas_id?: number;
        name: string;
        course_code?: string;
        term?: string;
        color_hex?: string;
        is_active?: boolean;
      } | null;
      rules: Array<{
        id: number;
        source_url: string;
        source_type: string;
        category?: string | null;
        extraction_config: Record<string, unknown>;
        check_frequency_hours?: number;
        last_checked_at?: string | null;
      }>;
      items: Array<{
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
        is_read?: boolean | null;
        created_at?: string | null;
        course_name?: string;
        course_code?: string;
      }>;
    }>
    discoverCourses: () => Promise<{
      success: boolean;
      message?: string;
      results?: Array<{
        courseId: string;
        name?: string;
        archetype: string | null;
        foundLinks: string[];
        savedRules: number;
      }>;
    }>
    askQuestion: (query: string) => Promise<string>
    getKeys: () => Promise<{ canvasToken: string; notionToken: string; canvasDomain: string; notionDbId: string; openaiKey?: string }>
    clearKeys: () => void
    setWindowSize: (options: { width: number, height: number, minWidth: number, minHeight: number }) => Promise<void>
    getUpcomingItems: () => Promise<Array<{
      id: number;
      course_id: string;
      item_type: string;
      title: string;
      date?: string | null;
      course_name?: string;
      course_code?: string;
      description?: string;
      content_url?: string;
      confidence_score?: number | null;
    }>>;
    getTriageItems: () => Promise<Array<{
      id: number;
      item_type: string;
      title: string;
      course_name: string;
      course_code: string;
      reasons: string[];
    }>>;
    getUniversalItem: (id: number) => Promise<any>;
    syncAll: () => Promise<{ success: boolean; message?: string; error?: string }>;
    openExternal: (url: string) => Promise<void>;
  }
}
