import { randomUUID } from 'node:crypto';
import type { UniversalItem, RealisedEntity } from '../../../src/lib/types';
import { extractMetadata } from './metadata_extractor';

interface ProvenanceEntry {
  field: string;
  source_item_id: number;
  method: string; 
  confidence: number;
}

/*
 Resolves a cluster of UniversalItems into a single RealisedEntity.
 Applies heuristic rules to select an anchor item and overlay metadata.
*/
export async function resolveCluster(cluster: UniversalItem[]): Promise<RealisedEntity> {
  if (cluster.length === 0) {
    throw new Error('Cannot resolve empty cluster');
  }

  // 1. Anchor Selection
  const anchor = selectAnchor(cluster);
  
  // Initialize Provenance Log
  const provenanceLog: ProvenanceEntry[] = [];

  // 2. Base Entity Construction from Anchor
  const entity: RealisedEntity = {
    id: randomUUID(),
    title: anchor.title,
    description: anchor.description || null,
    // Initialize with anchor's due date (if any)
    due_date: anchor.due_date || null,
    entity_type: mapToEntityType(anchor.item_type as string),
    status: 'pending_review',
    confidence_score: 0.8, // Default confidence
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    metadata: {},
  };

  // Log Anchor contributions
  provenanceLog.push({
    field: 'title',
    source_item_id: anchor.id,
    method: 'anchor',
    confidence: 1.0,
  });

  if (anchor.due_date) {
    provenanceLog.push({
      field: 'due_date',
      source_item_id: anchor.id,
      method: 'anchor',
      confidence: 1.0,
    });
  }

  // If Anchor is missing due_date, check other items
  if (!entity.due_date) {
    for (const item of cluster) {
      if (item.id === anchor.id) continue;

      // Extract metadata from raw content or description
      const textToScan = [item.raw_content_snippet, item.description].filter(Boolean).join('\n');
      if (!textToScan) continue;

      const extracted = await extractMetadata(textToScan);

      // Rule: If confidence > 0.8, use it
      if (extracted.dueDate && extracted.confidence > 0.8) {
        entity.due_date = extracted.dueDate.toISOString();
        
        provenanceLog.push({
          field: 'due_date',
          source_item_id: item.id,
          method: 'extracted_from_text',
          confidence: extracted.confidence,
        });
        
        // Break after finding a high-confidence date? 
        // The requirement says "use the PDF's date", implying first valid match is okay.
        break; 
      }
    }
  }

  // Store provenance in metadata
  entity.metadata = {
    provenance_log: provenanceLog,
    anchor_item_id: anchor.id,
  };

  return entity;
}

/*
 Selects the "Anchor" item based on priority rules.
 Priority 1: ASSIGNMENT or QUIZ
 Priority 2: SYLLABUS
 Priority 3: Earliest created_at
*/
function selectAnchor(cluster: UniversalItem[]): UniversalItem {
  // Sort function
  const sorted = [...cluster].sort((a, b) => {
    const scoreA = getPriorityScore(a);
    const scoreB = getPriorityScore(b);

    if (scoreA !== scoreB) {
      return scoreB - scoreA; // Higher score first
    }

    // Tie-breaker: Earliest created_at
    const dateA = a.created_at ? new Date(a.created_at).getTime() : Number.MAX_SAFE_INTEGER;
    const dateB = b.created_at ? new Date(b.created_at).getTime() : Number.MAX_SAFE_INTEGER;
    
    return dateA - dateB;
  });

  return sorted[0];
}

function getPriorityScore(item: UniversalItem): number {
  const type = (item.item_type || '').toString().toUpperCase();
  
  if (type === 'ASSIGNMENT' || type === 'QUIZ') return 3;
  if (type === 'SYLLABUS') return 2;
  return 1;
}

function mapToEntityType(itemType: string): string {
  const type = itemType.toUpperCase();
  if (type === 'ASSIGNMENT') return 'assignment';
  if (type === 'QUIZ') return 'quiz';
  if (type === 'SYLLABUS') return 'syllabus';
  if (type === 'READING' || type === 'FILE' || type === 'PAGE') return 'material';
  if (type === 'ANNOUNCEMENT') return 'announcement';
  return 'other';
}
