import * as chrono from 'chrono-node';
import { modelLoader } from '../ml';

export interface ExtractedMetadata {
  dueDate: Date | null;
  points: number | null;
  confidence: number;
}

export class MetadataExtractor {
  
  /* Uses a QA model to find an answer to a question within a context. */
  async findAnswer(context: string, question: string): Promise<{ answer: string; score: number }> {
    try {
      const pipe = await modelLoader.getPipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad');
      
      const q = String(question);
      const c = String(context);

      if (!c.trim()) {
        return { answer: '', score: 0 };
      }
      
      // Pass as separate arguments: question, context
      // @ts-ignore
      const result = await pipe(q, c);
      
      return {
        answer: result.answer,
        score: result.score
      };
    } catch (err) {
      console.error('[MetadataExtractor] QA Model failed:', err);
      return { answer: '', score: 0 };
    }
  }

  /* Extracts metadata using QA model with fallback to regex/rules. */
  async extract(text: string): Promise<ExtractedMetadata> {
    const result: ExtractedMetadata = {
      dueDate: null,
      points: null,
      confidence: 0,
    };

    if (!text || text.trim().length === 0) return result;

    // 1. Due Date Extraction
    // Try QA Model first
    const dateQA = await this.findAnswer(text, "When is this assignment due?");
    let dateFound = false;

    if (dateQA.score > 0.5) {
      const parsed = chrono.parseDate(dateQA.answer);
      if (parsed) {
        result.dueDate = parsed;
        result.confidence += 0.5; // High confidence from QA
        dateFound = true;
      }
    }

    // Fallback for Date
    if (!dateFound) {
      const parsedDates = chrono.parse(text);
      if (parsedDates.length > 0) {
        result.dueDate = parsedDates[0].start.date();
        // Lower confidence contribution if purely rule-based fallback (or maintain previous baseline)
        result.confidence += 0.3; 
      }
    }

    // 2. Points Extraction
    // Try QA Model first
    const pointsQA = await this.findAnswer(text, "How many points is this worth?");
    let pointsFound = false;

    if (pointsQA.score > 0.5) {
      const p = this.parsePoints(pointsQA.answer);
      if (p !== null) {
        result.points = p;
        result.confidence += 0.4;
        pointsFound = true;
      }
    }

    // Fallback for Points
    if (!pointsFound) {
      const p = this.parsePoints(text); // Use regex on full text
      if (p !== null) {
        result.points = p;
        // Lower confidence if fallback
        result.confidence += 0.2;
      }
    }

    // Cap confidence
    result.confidence = Math.min(result.confidence, 1.0);

    return result;
  }

  private parsePoints(text: string): number | null {
    // Regex matches "Points: 10", "10 pts", "10.5 points"
    const pointsRegex = /(?:Points:?\s*(\d+(?:\.\d+)?))|(\d+(?:\.\d+)?)\s*pts?/i;
    const match = text.match(pointsRegex);
    if (match) {
      const val = match[1] || match[2];
      return parseFloat(val);
    }
    // Also try simple number parsing if the QA answer was just "10"
    if (/^\d+(\.\d+)?$/.test(text.trim())) {
        return parseFloat(text.trim());
    }
    return null;
  }
}

export const metadataExtractor = new MetadataExtractor();
export const extractMetadata = (text: string) => metadataExtractor.extract(text);
