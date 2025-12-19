# Frontend - Local RAG System

React + Vite frontend for the Local RAG system.

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development
- **Tailwind CSS** for styling
- **Axios** for API communication

## Features

### 1. File Upload
- Drag-and-drop or click to upload
- Supported formats: PDF, TXT, MD, DOCX
- Real-time upload progress
- Success/error feedback

### 2. File Management
- List all indexed files
- Toggle files for search (active/inactive)
- View file metadata (type, chunk count)
- Visual indicators for selected files

### 3. Query Interface
- Natural language question input
- Search across selected files only
- Display AI-generated answers
- Show retrieved source chunks with similarity scores
- Collapsible sources panel

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts          # API client (axios)
│   ├── components/
│   │   ├── FileUpload.tsx     # Upload component
│   │   ├── FileList.tsx       # File list with selection
│   │   └── QueryInterface.tsx # Query input & results
│   ├── App.tsx                # Main app component
│   ├── main.tsx               # Entry point
│   └── index.css              # Tailwind imports
├── index.html
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run Development Server
```bash
npm run dev
```

Frontend will start on `http://localhost:5173`

### 3. Build for Production
```bash
npm run build
```

## API Integration

The frontend communicates with the backend through three main endpoints:

### Upload File
```typescript
POST /api/upload
Content-Type: multipart/form-data
```

### List Files
```typescript
GET /api/files
Response: FileInfo[]
```

### Query Documents
```typescript
POST /api/query
Body: {
  question: string,
  active_files?: string[],
  top_k?: number
}
Response: {
  answer: string,
  sources: SourceInfo[]
}
```

## UI Components

### FileUpload
- Handles file selection and upload
- Shows upload progress and status
- Triggers file list refresh on success

### FileList
- Displays all indexed files
- Checkbox selection for active files
- Passes selected files to query component
- Visual feedback for selection state

### QueryInterface
- Question input textarea
- Submit button with loading state
- Answer display area
- Collapsible sources section
- Source chunks with metadata

## Design Decisions

### State Management
- Used React hooks (useState, useEffect) for simplicity
- No Redux/Context - state is simple enough for prop drilling
- Component-level state for UI concerns
- Lifted state for shared data (active files)

### Styling
- Tailwind CSS for rapid development
- Utility-first approach
- Responsive grid layout
- Color-coded states (blue=active, gray=inactive, red=error, green=success)

### User Experience
- Immediate feedback on all actions
- Loading states for async operations
- Error messages with helpful details
- Success messages that auto-dismiss
- Collapsible sections to reduce clutter

### Type Safety
- TypeScript for all components
- Interfaces for API types
- Strict type checking enabled
- Props validation

## Future Improvements

- [ ] Add file deletion
- [ ] Implement query history
- [ ] Add export functionality (save Q&A)
- [ ] Improve mobile responsiveness
- [ ] Add dark mode
- [ ] Implement streaming responses
- [ ] Add query suggestions
- [ ] Show indexing progress in real-time
- [ ] Add file preview
- [ ] Implement pagination for large file lists
