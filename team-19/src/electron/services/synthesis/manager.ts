import db from '../../database/db';
import { randomUUID } from 'node:crypto';
import type { UniversalItem, RealisedEntity } from '../../../src/lib/types';
import { resolveCluster } from './cluster_resolver';
import { calculateLinkConfidence } from './matcher';

interface ClusterResult {
  items: UniversalItem[];
  minConfidence: number;
  synthesisLog: any[];
}

export class SynthesisManager {
  /*
   Queries universal_items for items that do not yet have an entry in entity_provenance.
  */
  findUnprocessedItems(courseId: string): UniversalItem[] {
    const stmt = db.prepare(`
      SELECT * FROM universal_items
      WHERE course_id = ?
      AND id NOT IN (SELECT universal_item_id FROM entity_provenance)
    `);
    return stmt.all(courseId) as UniversalItem[];
  }

  /* Clusters items using greedy strategy based on calculateLinkConfidence.
     Returns clusters along with their minimum link confidence. */
  async clusterItems(items: UniversalItem[]): Promise<ClusterResult[]> {
    const clusters: ClusterResult[] = [];

    // Sort by created_at to have a stable order, or importance
    const sortedItems = [...items].sort((a, b) => (a.id - b.id));

    for (const item of sortedItems) {
      console.log(`[Synthesis] Processing item ${item.id}: "${item.title}"`);
      let bestClusterIndex = -1;
      let bestConfidence = -1;
      let bestMatchItemId = -1;

      // Try to find a matching cluster
      for (let i = 0; i < clusters.length; i++) {
        let maxClusterConfidence = -1;
        let bestItemInCluster = -1;
        
        // Compare against ALL items in this cluster (not just the first)
        for (const clusterItem of clusters[i].items) {
          const confidence = await calculateLinkConfidence(clusterItem, item);
          
          if (confidence > maxClusterConfidence) {
            maxClusterConfidence = confidence;
            bestItemInCluster = clusterItem.id;
          }
        }

        // Use the best match from this cluster
        // Increased threshold from 0.4 to 0.55 to avoid over-clustering
        if (maxClusterConfidence >= 0.55 && maxClusterConfidence > bestConfidence) {
          bestConfidence = maxClusterConfidence;
          bestClusterIndex = i;
          bestMatchItemId = bestItemInCluster;
        }
      }

      if (bestClusterIndex !== -1) {
        // Add to existing cluster
        console.log(`[Synthesis]   -> Matched Cluster ${bestClusterIndex} (Confidence: ${bestConfidence.toFixed(2)}, matched item ${bestMatchItemId})`);
        clusters[bestClusterIndex].items.push(item);
        clusters[bestClusterIndex].minConfidence = Math.min(clusters[bestClusterIndex].minConfidence, bestConfidence);
        clusters[bestClusterIndex].synthesisLog.push({
          message: `Added item ${item.id} to cluster`,
          related_to_item: bestMatchItemId,
          confidence: bestConfidence
        });
      } else {
        // Start new cluster
        console.log(`[Synthesis]   -> Starting New Cluster ${clusters.length}`);
        clusters.push({
          items: [item],
          minConfidence: 1.0, // Self-match is perfect
          synthesisLog: [{
            message: `Started new cluster with item ${item.id}`,
            item_id: item.id
          }]
        });
      }
    }

    return clusters;
  }

  /* Synthesizes a RealisedEntity from a cluster. */
  async synthesizeEntity(clusterResult: ClusterResult): Promise<RealisedEntity> {
    const { items, minConfidence, synthesisLog } = clusterResult;
    
    // Use the cluster resolver to build the core entity
    const entity = await resolveCluster(items);
    console.log(`[Synthesis] Synthesizing Entity: "${entity.title}" (Items: ${items.length}, MinConf: ${minConfidence.toFixed(2)})`);

    // Set review status based on confidence thresholds
    // If minConfidence is 1.0 (single item), it's auto_verified (or strictly, it's just a raw item promoted).
    // The prompt implies we are looking at link confidence.
    
    if (items.length === 1) {
       // Single item is always verified as itself
       entity.review_status = 'auto_verified';
       entity.confidence_score = 1.0;
    } else if (minConfidence >= 0.85) {
      entity.review_status = 'auto_verified';
      entity.confidence_score = minConfidence;
    } else if (minConfidence >= 0.4) {
      entity.review_status = 'needs_review';
      entity.confidence_score = minConfidence;
    } else {
      // Should not happen given clustering logic, but fallback
      entity.review_status = 'needs_review';
      entity.confidence_score = minConfidence;
    }

    entity.synthesis_log = synthesisLog;
    
    return entity;
  }

  /* Performs a transaction to save the RealisedEntity and link its sources. */
  saveRealisedEntity(entity: RealisedEntity, sources: UniversalItem[]): void {
    console.log(`[Synthesis] Saving Realised Entity: ${entity.id} ("${entity.title}")`);
    const insertEntity = db.prepare(`
      INSERT INTO realised_entities (
        id, title, description, due_date, entity_type, status, review_status, synthesis_log, confidence_score, metadata, created_at, updated_at
      ) VALUES (
        @id, @title, @description, @due_date, @entity_type, @status, @review_status, @synthesis_log, @confidence_score, @metadata, @created_at, @updated_at
      )
    `);

    const insertProv = db.prepare(`
      INSERT INTO entity_provenance (
        id, realised_entity_id, universal_item_id, contribution_type
      ) VALUES (
        @id, @realised_entity_id, @universal_item_id, @contribution_type
      )
    `);

    const transaction = db.transaction(() => {
      insertEntity.run({
        ...entity,
        synthesis_log: JSON.stringify(entity.synthesis_log || []),
        metadata: JSON.stringify(entity.metadata || {})
      });

      for (const source of sources) {
        insertProv.run({
          id: randomUUID(),
          realised_entity_id: entity.id,
          universal_item_id: source.id,
          contribution_type: 'source_content'
        });
      }
    });

    transaction();
  }

  /* Orchestrates the synthesis process for a single course. */
  async processCourse(courseId: string): Promise<void> {
    console.log(`[Synthesis] Starting synthesis for course ${courseId}`);
    try {
      const unprocessedItems = this.findUnprocessedItems(courseId);
      if (unprocessedItems.length === 0) {
        console.log(`[Synthesis] No unprocessed items found for course ${courseId}`);
        return;
      }

      console.log(`[Synthesis] Found ${unprocessedItems.length} unprocessed items`);
      // Update: Await the async clusterItems
      const clusters = await this.clusterItems(unprocessedItems);
      console.log(`[Synthesis] Formed ${clusters.length} clusters`);

      for (const clusterResult of clusters) {
        try {
          const entity = await this.synthesizeEntity(clusterResult);
          this.saveRealisedEntity(entity, clusterResult.items);
        } catch (err) {
          console.error(`[Synthesis] Error processing cluster:`, err);
        }
      }
      console.log(`[Synthesis] Completed synthesis for course ${courseId}`);
    } catch (error) {
      console.error(`[Synthesis] Error during synthesis for course ${courseId}:`, error);
    }
  }
}

export const synthesisManager = new SynthesisManager();
