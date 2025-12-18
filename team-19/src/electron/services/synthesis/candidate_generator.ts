import { modelLoader } from '../ml';
import { similaritySearch } from '../../database/vector-store';
import db from '../../database/db';
import type { UniversalItem } from '../../../src/lib/types';

export class CandidateGenerator {
  
  /*
   Finds candidate UniversalItems that are semantically similar to the anchor item.
   Enforces they belong to the same course.
  */
  async findCandidates(anchorItem: UniversalItem): Promise<UniversalItem[]> {
    console.log(`[CandidateGenerator] Finding candidates for item ${anchorItem.id} (${anchorItem.title})`);

    const courseId = anchorItem.course_id;
    if (!courseId) {
      console.warn('[CandidateGenerator] Anchor item missing course_id');
      return [];
    }

    // 1. Get Embedding for Anchor
    let embedding: number[] | null = null;
    
    // Check if it's already in the store (using ID as string)
    // Note: This check logic relies on us being able to retrieve the vector, which documentExists doesn't do.
    // We might need to generate it anyway if we can't easily fetch the vector from LanceDB by ID without a full query.
    // For simplicity and robustness, let's generate it on the fly if we don't have it handy, 
    // or we could query the table for the specific ID to get its vector.
    // Let's try to generate it to ensure we have a valid query vector.
    
    try {
      embedding = await this.generateEmbedding(anchorItem.title + ' ' + (anchorItem.description || ''));
    } catch (err) {
      console.error('[CandidateGenerator] Failed to generate embedding:', err);
      return [];
    }

    if (!embedding) return [];

    // 2. Query Vector Store
    // Filter by course_id and ensure we don't match the item itself (though post-filter is safer for ID equality)
    // LanceDB filter syntax: "course_id = '123' AND id != '456'"
    // Note: We need to ensure course_id is stored in metadata or as a column. 
    // I will assume 'metadata' field in LanceDB contains a JSON string where we might find course_id, 
    // OR that course_id is a top-level column. 
    // If course_id is ONLY in the JSON metadata string, LanceDB might not be able to filter on it efficiently without a specific schema.
    // However, looking at previous vector-store.ts, it seems to use a generic schema.
    // Let's try to filter by metadata if possible, or just fetch more and filter in memory if strict SQL filtering isn't set up.
    // Given the prompt: "Filter results: enforce that candidates must be from the same course_id."
    // I will use a post-filter approach if the DB schema is uncertain, but will attempt a generic filter string first.
    
    // Assuming metadata is a JSON string, filtering inside it might be tricky with simple SQL.
    // SAFE APPROACH: Retrieve results, then filter in memory for course_id match.
    // But better: Use the `filter` param on `similaritySearch` if the schema supports it.
    // Let's assume we can filter on `metadata` if it was flattened, but here it's a string.
    // Actually, let's assume the user has set up the vector store to index specific fields or we filter post-fetch.
    // Limit is small (5), so post-filtering might miss valid candidates if the top 5 are from other courses.
    // I will request 20 items and filter in memory to be safe, unless I can be sure of the schema.
    
    const results = await similaritySearch(embedding, 20);
    
    // process results here
    const candidateIds: number[] = [];
    
    for (const res of results) {
        // Parse metadata
        let meta: any = {};
        try {
            meta = typeof res.metadata === 'string' ? JSON.parse(res.metadata) : res.metadata;
        } catch (e) {
            continue;
        }
        
        // Check ID mismatch
        if (String(res.id) === String(anchorItem.id)) continue;
        
        // Check Course ID (from metadata or if we can fetch it)
        // If metadata doesn't have course_id, we'll have to check the DB later.
        if (meta.course_id && meta.course_id !== courseId) continue;
        
        // If metadata is just a string or doesn't have course_id, we might rely on the DB lookup.
        // But let's assume we want to match valid IDs.
        
        candidateIds.push(Number(res.id));
    }

    if (candidateIds.length === 0) return [];

    // 4. Fetch Full Items from SQLite
    // We only want the top 5 valid ones
    const placeholders = candidateIds.map(() => '?').join(',');
    const stmt = db.prepare(`SELECT * FROM universal_items WHERE id IN (${placeholders}) AND course_id = ?`);
    
    const candidates = stmt.all(...candidateIds, courseId) as UniversalItem[];
    
    // Re-order based on the vector search order (which was by similarity)
    const orderedCandidates = candidateIds
        .map(id => candidates.find(c => c.id === id))
        .filter((c): c is UniversalItem => !!c) // Filter undefined
        .slice(0, 5); // Take top 5

    console.log(`[CandidateGenerator] Found ${orderedCandidates.length} candidates for item ${anchorItem.id}`);
    return orderedCandidates;
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    const pipe = await modelLoader.getPipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    const output = await pipe(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
  }
}

export const candidateGenerator = new CandidateGenerator();
