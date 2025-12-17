/**
 * File list component.
 * 
 * Displays all indexed files with their metadata.
 * Allows selecting which files are active for search.
 */

import React, { useEffect, useState } from 'react';
import { getFiles, FileInfo } from '../api/client';

interface FileListProps {
  refresh: number;
  onActiveFilesChange: (files: string[]) => void;
}

export const FileList: React.FC<FileListProps> = ({ refresh, onActiveFilesChange }) => {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeFiles, setActiveFiles] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadFiles();
  }, [refresh]);

  const loadFiles = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const fileList = await getFiles();
      setFiles(fileList);
      
      // Initially select all files
      const allFilenames = new Set(fileList.map(f => f.filename));
      setActiveFiles(allFilenames);
      onActiveFilesChange(Array.from(allFilenames));
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const toggleFile = (filename: string) => {
    const newActiveFiles = new Set(activeFiles);
    if (newActiveFiles.has(filename)) {
      newActiveFiles.delete(filename);
    } else {
      newActiveFiles.add(filename);
    }
    setActiveFiles(newActiveFiles);
    onActiveFilesChange(Array.from(newActiveFiles));
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Indexed Files</h2>
        <div className="text-gray-500">Loading files...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Indexed Files</h2>
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold mb-4">Indexed Files ({files.length})</h2>
      
      {files.length === 0 ? (
        <div className="text-gray-500 text-center py-4">
          No files indexed yet. Upload a document to get started.
        </div>
      ) : (
        <div className="space-y-2">
          {files.map((file) => (
            <div
              key={file.filename}
              className={`border rounded-lg p-3 cursor-pointer transition-colors ${
                activeFiles.has(file.filename)
                  ? 'bg-blue-50 border-blue-300'
                  : 'bg-gray-50 border-gray-200'
              }`}
              onClick={() => toggleFile(file.filename)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={activeFiles.has(file.filename)}
                    onChange={() => toggleFile(file.filename)}
                    className="w-4 h-4 text-blue-600 cursor-pointer"
                  />
                  <div>
                    <div className="font-medium text-gray-900">{file.filename}</div>
                    <div className="text-sm text-gray-500">
                      {file.num_chunks} chunks • {file.file_type.toUpperCase()}
                    </div>
                  </div>
                  <span className="text-xs font-semibold text-blue-700 bg-blue-100 px-2 py-1 rounded">
                    {file.file_type.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {files.length > 0 && (
        <div className="mt-4 text-sm text-gray-600">
          Tip: Click files to toggle them for search. Only selected files will be queried.
        </div>
      )}
    </div>
  );
};
