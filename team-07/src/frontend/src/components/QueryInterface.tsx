/**
 * Query interface component.
 * 
 * Handles user queries and displays results with sources.
 */

import React, { useState } from 'react';
import { queryDocuments, QueryResponse } from '../api/client';

interface QueryInterfaceProps {
  activeFiles: string[];
}

export const QueryInterface: React.FC<QueryInterfaceProps> = ({ activeFiles }) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [showSources, setShowSources] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    if (activeFiles.length === 0) {
      setError('Please select at least one file to search');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await queryDocuments({
        question: question.trim(),
        active_files: activeFiles,
        top_k: 5,
      });
      setResult(response);
      setShowSources(false);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Query failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold mb-4">Ask a Question</h2>
      
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="mb-4">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question here..."
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
        </div>
        
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600">
            Searching in {activeFiles.length} file{activeFiles.length !== 1 ? 's' : ''}
          </div>
          <button
            type="submit"
            disabled={loading || !question.trim() || activeFiles.length === 0}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8">
          <svg className="animate-spin h-8 w-8 text-blue-600" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span className="ml-3 text-gray-600">Generating answer...</span>
        </div>
      )}

      {result && (
        <div className="border-t pt-6">
          <div className="mb-4">
            <h3 className="text-lg font-semibold mb-2">Answer</h3>
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 whitespace-pre-wrap">
              {result.answer}
            </div>
          </div>

          {result.sources.length > 0 && (
            <div>
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center text-sm font-medium text-blue-600 hover:text-blue-800 mb-3"
              >
                <span>
                  {showSources ? 'Hide sources' : 'Show sources'} ({result.sources.length})
                </span>
              </button>

              {showSources && (
                <div className="space-y-3">
                  {result.sources.map((source, idx) => (
                    <div
                      key={idx}
                      className="bg-blue-50 border border-blue-200 rounded-lg p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-medium text-sm text-gray-900">
                          {source.filename} (chunk {source.chunk_index})
                        </div>
                        <div className="text-sm text-gray-600">
                          Score: {(source.score * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="text-sm text-gray-700 whitespace-pre-wrap">
                        {source.chunk}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
