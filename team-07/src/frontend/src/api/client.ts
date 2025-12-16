/**
 * API client for backend communication.
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 second timeout for LLM responses
});

export interface UploadResponse {
  filename: string;
  chunks_created: number;
  file_type: string;
}

export interface FileInfo {
  filename: string;
  file_type: string;
  num_chunks: number;
}

export interface SourceInfo {
  filename: string;
  chunk: string;
  score: number;
  chunk_index: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceInfo[];
}

export interface QueryRequest {
  question: string;
  active_files?: string[];
  top_k?: number;
}

/**
 * Upload a file to be indexed.
 */
export const uploadFile = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post<UploadResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

/**
 * Get list of all indexed files.
 */
export const getFiles = async (): Promise<FileInfo[]> => {
  const response = await api.get<FileInfo[]>('/files');
  return response.data;
};

/**
 * Query the indexed documents.
 */
export const queryDocuments = async (request: QueryRequest): Promise<QueryResponse> => {
  const response = await api.post<QueryResponse>('/query', request);
  return response.data;
};

/**
 * Check backend health.
 */
export const checkHealth = async (): Promise<any> => {
  const response = await api.get('/');
  return response.data;
};
