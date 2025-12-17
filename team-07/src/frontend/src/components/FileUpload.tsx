/**
 * File upload component.
 * 
 * Allows users to select and upload PDF, TXT, MD, or DOCX files.
 */

import React, { useState } from 'react';
import { uploadFile } from '../api/client';

interface FileUploadProps {
  onUploadSuccess: () => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await uploadFile(file);
      setSuccess(`Uploaded ${response.filename} (${response.chunks_created} chunks created)`);
      onUploadSuccess();
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccess(null), 5000);
      
      // Reset file input
      event.target.value = '';
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold mb-4">Upload Document</h2>
      
      <div className="mb-4">
        <label className="block mb-2 text-sm font-medium text-gray-700">
          Select a file to index (PDF, TXT, MD, DOCX)
        </label>
        <input
          type="file"
          accept=".pdf,.txt,.md,.markdown,.docx"
          onChange={handleFileSelect}
          disabled={uploading}
          className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none p-2"
        />
      </div>

      {uploading && (
        <div className="flex items-center text-blue-600 mb-2">
          <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          Processing and indexing file...
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-2">
          {error}
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded mb-2">
          {success}
        </div>
      )}
    </div>
  );
};
