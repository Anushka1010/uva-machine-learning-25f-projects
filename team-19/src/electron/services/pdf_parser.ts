import { PDFParse } from 'pdf-parse';
import path from 'node:path';
import { app } from 'electron';

// Configure PDF.js worker for Electron
// Use app.getAppPath() to get the correct base directory
const appPath = app.getAppPath();
const workerPath = path.join(appPath, 'node_modules/pdfjs-dist/legacy/build/pdf.worker.mjs');
PDFParse.setWorker(workerPath);

/*
 Downloads and extracts text content from a PDF file.
 Uses pdf-parse library for local extraction (fast, free, no API costs).
 Returns the extracted text or null if parsing fails.
 */
export async function extractPdfText(pdfUrl: string): Promise<string | null> {
  try {
    console.log(`[PDF Parser] Downloading PDF from: ${pdfUrl}`);
    
    // Download the PDF
    const response = await fetch(pdfUrl);
    if (!response.ok) {
      console.error(`[PDF Parser] Failed to download PDF: ${response.statusText}`);
      return null;
    }

    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    
    // Check size (reasonable limit: 10MB for performance)
    const sizeMB = buffer.length / (1024 * 1024);
    if (sizeMB > 10) {
      console.warn(`[PDF Parser] PDF too large (${sizeMB.toFixed(2)}MB), skipping`);
      return null;
    }

    console.log(`[PDF Parser] PDF downloaded (${sizeMB.toFixed(2)}MB), extracting text...`);

    // Parse PDF using pdf-parse
    const parser = new PDFParse({ data: buffer });
    const result = await parser.getText();
    
    const extractedText = result.text.trim();
    
    if (!extractedText) {
      console.warn('[PDF Parser] No text content found in PDF');
      return null;
    }

    console.log(`[PDF Parser] Successfully extracted ${extractedText.length} characters from ${result.pages.length} pages`);
    
    // Truncate if too long (keep first 10000 chars for clustering)
    const truncated = extractedText.length > 10000 
      ? extractedText.substring(0, 10000) + '...' 
      : extractedText;
    
    return truncated;

  } catch (error: any) {
    console.error('[PDF Parser] Error extracting PDF text:', error.message);
    return null;
  }
}
