import type { UniversalItem } from '../../../src/lib/types';
import { getOrCreateEmbedding, cosineSimilarity, extractAssignmentNumber } from './embedding_cache';

/* Predicts similarity between two items using cached embeddings.
   Returns a normalized score between 0.0 and 1.0. */
async function predictSimilarity(itemA: UniversalItem, itemB: UniversalItem): Promise<number> {
  try {
    // Get cached or generate embeddings
    const [embeddingA, embeddingB] = await Promise.all([
      getOrCreateEmbedding(itemA),
      getOrCreateEmbedding(itemB)
    ]);
    
    // Calculate base cosine similarity
    let similarity = cosineSimilarity(embeddingA, embeddingB);
    
    // Boost score if assignment numbers match
    const assignmentA = extractAssignmentNumber(itemA.title);
    const assignmentB = extractAssignmentNumber(itemB.title);
    
    if (assignmentA && assignmentB && assignmentA === assignmentB) {
      console.log(`[Matcher] Assignment number match! ${itemA.title} (${assignmentA}) <-> ${itemB.title} (${assignmentB})`);
      // Boost by 0.25 to strongly encourage clustering of same assignment
      similarity = Math.min(1.0, similarity + 0.25);
    }
    
    // Normalize to 0-1 range
    const score = Math.max(0, Math.min(1, similarity));
    
    return score;
  } catch (err) {
    console.error('[Matcher] Model prediction failed:', err);
    // Fallback to Jaccard if ML fails
    const textA = `${itemA.title} ${itemA.description || ''} ${itemA.raw_content_snippet || ''}`.trim();
    const textB = `${itemB.title} ${itemB.description || ''} ${itemB.raw_content_snippet || ''}`.trim();
    return jaccardSimilarity(textA, textB);
  }
}

/* Calculates the Jaccard Similarity between two strings.
   Tokenizes by splitting on whitespace and converting to lowercase. */
function jaccardSimilarity(str1: string, str2: string): number {
  const tokenize = (s: string) => new Set(s.toLowerCase().trim().split(/\s+/));
  
  const set1 = tokenize(str1);
  const set2 = tokenize(str2);
  
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);
  
  if (union.size === 0) return 0;
  return intersection.size / union.size;
}

/* Calculates a time difference score (0.0 - 1.0).
   1.0 = Same time. Drops off as difference increases. */
function calculateTimeScore(itemA: UniversalItem, itemB: UniversalItem): number {
    const dateA = itemA.due_date ? new Date(itemA.due_date).getTime() : (itemA.created_at ? new Date(itemA.created_at).getTime() : null);
    const dateB = itemB.due_date ? new Date(itemB.due_date).getTime() : (itemB.created_at ? new Date(itemB.created_at).getTime() : null);

    if (!dateA || !dateB) return 0.5; // Neutral if missing dates

    const diffMs = Math.abs(dateA - dateB);
    const diffDays = diffMs / (1000 * 60 * 60 * 24);

    // Decay function: 1 / (1 + diffDays)
    // If 0 days diff -> 1.0
    // If 1 day diff -> 0.5
    // If 7 days diff -> 0.125
    return 1 / (1 + diffDays);
}

/* Calculates a confidence score (0.0 - 1.0) indicating how likely two items are related.
   Uses cached embeddings (70%) and Time Difference (30%). */
export async function calculateLinkConfidence(itemA: UniversalItem, itemB: UniversalItem): Promise<number> {
  // Get model score using cached embeddings
  const modelScorePromise = predictSimilarity(itemA, itemB);
  const timeScore = calculateTimeScore(itemA, itemB);
  
  const modelScore = await modelScorePromise;

  // Weighted Combination
  // 70% Model, 30% Time
  let finalScore = (modelScore * 0.7) + (timeScore * 0.3);

  // Boost Rule: If Model is very confident (>0.9), force high confidence.
  if (modelScore > 0.9) {
      finalScore = 1.0;
  }
  
  // Module ID Boost/Penalty (Legacy Logic preserved/adapted)
  const modA = (itemA as any).module_id;
  const modB = (itemB as any).module_id;
  if (modA && modB) {
      if (modA === modB) finalScore = Math.min(finalScore + 0.1, 1.0);
      else finalScore = Math.max(finalScore - 0.1, 0.0);
  }

  // LOGGING: Breakdown of scores
  console.log(`[Matcher] Comparing '${itemA.title}' vs '${itemB.title}'`);
  console.log(`          Model Score: ${modelScore.toFixed(2)} | Time Score: ${timeScore.toFixed(2)}`);
  console.log(`          Initial Weighted: ${((modelScore * 0.7) + (timeScore * 0.3)).toFixed(2)} | Final: ${finalScore.toFixed(2)}`);

  return Math.min(Math.max(finalScore, 0), 1);
}
