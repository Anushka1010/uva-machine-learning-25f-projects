export type ItemType = 'ASSIGNMENT' | 'READING' | 'SLIDE' | 'SYLLABUS' | 'ANNOUNCEMENT' | 'QUIZ' | 'FILE' | 'PAGE' | 'DISCUSSION' | 'UNKNOWN';

export type UniversalItem = {
  id: number;
  course_id: string;
  ingestion_rule_id?: number | null;
  item_type: ItemType | string;
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
};

export type View = 'dashboard' | 'assignments' | 'chat' | 'settings' | 'course' | 'detail_view' | 'triage' | 'materials_list';

export type RealisedEntity = {
  id: string;
  title: string;
  description?: string | null;
  due_date?: string | null;
  entity_type: string;
  status: string;
  review_status?: 'auto_verified' | 'needs_review' | 'user_verified';
  synthesis_log?: any;
  confidence_score?: number | null;
  metadata?: Record<string, any> | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type EntityProvenance = {
  id: string;
  realised_entity_id: string;
  universal_item_id: number;
  contribution_type: string;
};

export type RealisedEntityWithSources = RealisedEntity & {
  sources: UniversalItem[];
};