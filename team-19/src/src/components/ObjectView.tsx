import { useEffect, useState } from 'react';
import { ChevronLeft, Calendar, Clock, FileText, ListTodo, Paperclip, File, ExternalLink, X } from 'lucide-react';
import { Loader2 } from 'lucide-react';
import { ItemType } from '../lib/types';

interface ObjectViewProps {
  type: ItemType;
  id: number;
  onBack: () => void;
}

export default function ObjectView({ type, id, onBack }: ObjectViewProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [previewFile, setPreviewFile] = useState<any>(null);

  useEffect(() => {
    loadData();
  }, [type, id]);

  const loadData = async () => {
    setLoading(true);
    try {
      const result = await window.canvasGPT.getUniversalItem(id);
      setData(result);
    } catch (error) {
      console.error(`Failed to load ${type} ${id}:`, error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'No date';
    return new Date(dateString).toLocaleDateString(undefined, {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric'
    });
  };

  const formatItemType = (itemType: string) => {
    switch (itemType) {
      case 'ASSIGNMENT':
        return 'Assignment';
      case 'READING':
        return 'Reading';
      case 'SLIDE':
        return 'Slide';
      case 'SYLLABUS':
        return 'Syllabus';
      default:
        return itemType;
    }
  };

  const resolvedType = (data?.item_type as ItemType) || type || 'UNKNOWN';
  const typeLabel = formatItemType(resolvedType);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center bg-white">
        <Loader2 className="h-8 w-8 animate-spin text-slate-300" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex flex-col h-full items-center justify-center bg-white">
        <p className="text-slate-500">Item not found.</p>
        <button onClick={onBack} className="mt-4 text-blue-600 hover:underline">
          Go Back
        </button>
      </div>
    );
  }

  const renderHeaderIcon = () => {
    switch (resolvedType) {
      case 'ASSIGNMENT': return <ListTodo className="h-6 w-6 text-blue-500" />;
      case 'READING': return <FileText className="h-6 w-6 text-emerald-600" />;
      case 'SLIDE': return <FileText className="h-6 w-6 text-indigo-600" />;
      case 'SYLLABUS': return <FileText className="h-6 w-6 text-slate-500" />;
      default: return <FileText className="h-6 w-6 text-slate-500" />;
    }
  };

  const renderHeader = () => (
    <div className="border-b border-slate-200 bg-slate-50 p-6 relative">
      <button 
        onClick={onBack}
        className="flex items-center text-sm text-slate-500 hover:text-slate-900 mb-4 transition-colors"
      >
        <ChevronLeft className="w-4 h-4 mr-1" />
        Back
      </button>

      <span className="absolute top-6 right-6 inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-500/10">
        {typeLabel}
      </span>

      <div className="flex items-start gap-4">
        <div className="p-3 bg-white rounded-lg shadow-sm border border-slate-100">
            {renderHeaderIcon()}
        </div>
        <div className="flex-1">
            <h1 className="text-2xl font-bold text-slate-900 leading-tight">
                {data.title || data.name}
            </h1>
            
            <div className="flex flex-wrap items-center gap-4 mt-3 text-sm text-slate-500">
                {data.due_date && (
                    <div className="flex items-center gap-1.5">
                        <Calendar size={15} />
                        <span>Due: {formatDate(data.due_date)}</span>
                    </div>
                )}
                {data.course_name && (
                    <div className="flex items-center gap-1.5">
                        <Clock size={15} />
                        <span>{data.course_name}</span>
                    </div>
                )}
            </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <div className="flex flex-col h-full bg-white relative z-0">
        {renderHeader()}
        <div className="flex-1 overflow-auto p-8">
          <div className="max-w-4xl mx-auto">
             {data.content_url && (
              <div className="mb-4">
                <a
                  href={data.content_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  <ExternalLink className="h-4 w-4" />
                  Open Source
                </a>
              </div>
             )}
             <div className="prose prose-slate prose-headings:font-semibold prose-a:text-blue-600 hover:prose-a:text-blue-500">
                <div dangerouslySetInnerHTML={{ __html: data.description || data.raw_content_snippet || '<p>No content.</p>' }} />
             </div>

             {data.files && data.files.length > 0 && (
              <div className="mt-8 border-t border-slate-200 pt-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                  <Paperclip className="h-5 w-5 text-slate-500" />
                  Attachments
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {data.files.map((file: any) => (
                                  <button
                                    key={file.id}
                                    onClick={() => setPreviewFile(file)}
                                    className="group flex items-center p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-all bg-white text-left focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 cursor-pointer" 
                                    title={`Preview ${file.display_name}`}
                                  >
                                    <File className="h-8 w-8 text-blue-500 bg-blue-100 p-1.5 rounded-md mr-3" />
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm font-medium text-slate-900 truncate group-hover:text-blue-700">
                                        {file.display_name}
                                      </p>
                                      <p className="text-xs text-slate-500">
                                         {file.processing_state === 'processed' ? 'Processed' : 
                                          file.processing_state === 'download_failed' ? 'Download Failed' : 
                                          file.processing_state === 'extraction_failed' ? 'Preview Unavailable' : 'Processing...'}
                                      </p>
                                    </div>
                                    {/* The ExternalLink is now only in the modal */}
                                  </button>
                                ))}                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {previewFile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="flex h-[85vh] w-full max-w-4xl flex-col rounded-xl bg-white shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
             <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50 px-6 py-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white rounded-lg border border-slate-200 shadow-sm">
                    <File className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-slate-900 leading-tight">{previewFile.display_name}</h3>
                    <p className="text-xs text-slate-500 mt-0.5">Previewing extracted content</p>
                  </div>
                </div>
                <button 
                  onClick={() => setPreviewFile(null)} 
                  className="rounded-full p-2 text-slate-400 hover:bg-slate-200 hover:text-slate-600 transition-colors"
                >
                   <X className="h-5 w-5" />
                </button>
             </div>
             
             <div className="flex-1 overflow-auto bg-white p-8">
                {previewFile.extracted_text ? (
                  <div className="mx-auto max-w-3xl">
                    <pre className="whitespace-pre-wrap font-mono text-sm text-slate-700 leading-relaxed bg-slate-50 p-6 rounded-lg border border-slate-100 shadow-sm">
                       {previewFile.extracted_text}
                    </pre>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-4">
                    <FileText className="h-16 w-16 text-slate-300" />
                    <div className="text-center">
                      <p className="text-lg font-medium text-slate-900">No Preview Available</p>
                      <p className="text-sm mt-1">We couldn't extract text from this file for preview.</p>
                      {previewFile.processing_state === 'download_failed' && (
                        <p className="text-xs text-red-500 mt-2 font-medium">Reason: Download from Canvas failed.</p>
                      )}
                    </div>
                  </div>
                )}
             </div>

             <div className="border-t border-slate-100 bg-slate-50 px-6 py-4 flex justify-between items-center">
                 <span className="text-xs text-slate-500">
                    ID: {previewFile.id}
                 </span>
                 {previewFile.url && (
                    <a 
                      href={previewFile.url} 
                      target="_blank" 
                      rel="noreferrer" 
                      className="inline-flex items-center gap-2 rounded-md bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm ring-1 ring-inset ring-slate-300 hover:bg-slate-50"
                    >
                       <ExternalLink className="h-4 w-4" />
                       Open Original in Canvas
                    </a>
                 )}
             </div>
          </div>
        </div>
      )}
    </>
  );
}
