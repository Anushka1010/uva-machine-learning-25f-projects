import { modelLoader } from '../ml';

export interface ClassificationResult {
  label: string;
  confidence: number;
  isPotentialEntity: boolean;
}

export class OrphanClassifier {
  private readonly CANDIDATE_LABELS = [
    'assignment', 
    'quiz', 
    'announcement', 
    'syllabus', 
    'administrative'
  ];

  /* Classifies a text snippet to determine its semantic type.
     Uses Zero-Shot Classification. */
  async classifyItem(text: string): Promise<ClassificationResult> {
    if (!text || text.trim().length === 0) {
      return { label: 'unknown', confidence: 0, isPotentialEntity: false };
    }

    try {
      const pipe = await modelLoader.getPipeline('zero-shot-classification', 'Xenova/mobilebert-uncased-mnli');
      
      // @ts-ignore - transformers.js types might differ slightly
      const output = await pipe(text, this.CANDIDATE_LABELS);
      
      // Output format is usually { sequence: string, labels: string[], scores: number[] }
      const topLabel = output.labels[0];
      const topScore = output.scores[0];

      // Check criteria: 'assignment' or 'quiz' with > 0.9 confidence
      const isHighConfidenceEntity = 
        (topLabel === 'assignment' || topLabel === 'quiz') && 
        topScore > 0.9;

      return {
        label: topLabel,
        confidence: topScore,
        isPotentialEntity: isHighConfidenceEntity
      };
    } catch (err) {
      console.error('[OrphanClassifier] Classification failed:', err);
      return { label: 'error', confidence: 0, isPotentialEntity: false };
    }
  }
}

export const orphanClassifier = new OrphanClassifier();
