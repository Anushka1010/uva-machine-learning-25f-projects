/**
 * Main application component.
 */

import React, { useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { FileList } from './components/FileList';
import { QueryInterface } from './components/QueryInterface';

function App() {
  const [refreshFiles, setRefreshFiles] = useState(0);
  const [activeFiles, setActiveFiles] = useState<string[]>([]);

  const handleUploadSuccess = () => {
    setRefreshFiles((prev: number) => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-blue-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">Local RAG System</h1>
          <p className="text-blue-100 mt-2">Retrieval-augmented search for your documents</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Upload & Files */}
          <div className="space-y-6">
            <FileUpload onUploadSuccess={handleUploadSuccess} />
            <FileList 
              refresh={refreshFiles} 
              onActiveFilesChange={setActiveFiles}
            />
          </div>

          {/* Right Column: Query Interface */}
          <div className="lg:col-span-2">
            <QueryInterface activeFiles={activeFiles} />
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-8 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-bold mb-3">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="font-semibold text-blue-600 mb-1">1. Upload</div>
              <div className="text-gray-600">
                Upload PDF, TXT, MD, or DOCX files. Documents are chunked and embedded locally.
              </div>
            </div>
            <div>
              <div className="font-semibold text-blue-600 mb-1">2. Index</div>
              <div className="text-gray-600">
                Text is split into chunks and converted to vectors using sentence-transformers.
              </div>
            </div>
            <div>
              <div className="font-semibold text-blue-600 mb-1">3. Query</div>
              <div className="text-gray-600">
                Ask questions and get answers grounded in your documents.
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 mt-12">
        <div className="container mx-auto px-4 py-6 text-center text-sm">
          <p>Local RAG System • Built with FastAPI, React, FAISS & Ollama</p>
          <p className="mt-1">ML Final Project 2024</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
