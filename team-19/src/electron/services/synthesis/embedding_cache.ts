import db from '../../database/db';
import { modelLoader } from '../ml';
import type { UniversalItem } from '../../../src/lib/types';

/*
 Get or generate embedding for an item.
 Checks database cache first, generates and stores if missing.
 */
export async function getOrCreateEmbedding(item: UniversalItem): Promise<number[]> {
  // Check if embedding exists in database
  const row = db.prepare('SELECT embedding FROM universal_items WHERE id = ?').get(item.id) as { embedding?: string } | undefined;
  
  if (row?.embedding) {
    try {
      return JSON.parse(row.embedding);
    } catch (err) {
      console.warn(`[EmbeddingCache] Failed to parse cached embedding for item ${item.id}, regenerating`);
    }
  }
  
  // Generate new embedding
  const text = `${item.title} ${item.description || ''} ${item.raw_content_snippet || ''}`.trim();
  const embedding = await generateEmbedding(text);
  
  // Store in database
  db.prepare('UPDATE universal_items SET embedding = ? WHERE id = ?').run(JSON.stringify(embedding), item.id);
  
  console.log(`[EmbeddingCache] Generated and cached embedding for item ${item.id}`);
  return embedding;
}

/* Generate embedding vector for text using all-MiniLM-L6-v2.
   Exported for use by other services (e.g., chat agent). */
export async function generateEmbedding(text: string): Promise<number[]> {
  if (!text.trim()) {
    // Return zero vector for empty text
    return new Array(384).fill(0); // all-MiniLM-L6-v2 has 384 dimensions
  }
  
  const extractor = await modelLoader.getPipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  
  // @ts-ignore
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  
  return Array.from(output.data);
}

/*
 Calculate cosine similarity between two embedding vectors.
*/
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}

/*
  Currently not working well
  TODO: FIX THIS
 */
export function extractAssignmentNumber(title: string): string | null {
  const lower = title.toLowerCase();
  
  let match = lower.match(/pset[_\s-]?(\d+)/);
  if (match) return parseInt(match[1], 10).toString();
  
  match = lower.match(/(?:homework|hw)[_\s-]?(\d+)/);
  if (match) return parseInt(match[1], 10).toString();
  
  match = lower.match(/practice[_\s-]?(\d+)/);
  if (match) return parseInt(match[1], 10).toString();
  
  match = lower.match(/assignment[_\s-]?(\d+)/);
  if (match) return parseInt(match[1], 10).toString();
  
  match = lower.match(/^(\d+)[\._-]/);
  if (match) return parseInt(match[1], 10).toString();
  
  return null;
}
