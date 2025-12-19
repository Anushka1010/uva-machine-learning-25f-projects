import { useEffect, useState } from 'react';
import { ChevronLeft, ListTodo, FileText, Calendar, BookOpen, AlertCircle, Percent, Link as LinkIcon, ArrowRight } from 'lucide-react';
import { Card } from './ui';
import { ItemType, UniversalItem } from '../lib/types';

interface CourseViewProps {
  courseId: string;
  onBack: () => void;
  onSelect: (type: ItemType, id: number) => void;
  onViewAllMaterials: (courseId: string) => void; // New prop
}

export default function CourseView({ courseId, onBack, onSelect, onViewAllMaterials }: CourseViewProps) {
  const [course, setCourse] = useState<any>(null);
  const [rules, setRules] = useState<any[]>([]);
  const [items, setItems] = useState<UniversalItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [canvasDomain, setCanvasDomain] = useState<string>('');

  useEffect(() => {
    loadAllData();
  }, [courseId]);

  const loadAllData = async () => {
    setLoading(true);
    try {
      const [data, keys] = await Promise.all([
        window.canvasGPT.getCourseDetails(courseId),
        window.canvasGPT.getKeys()
      ]);
      setCourse(data?.course ?? null);
      setRules(data?.rules ?? []);
      setItems((data?.items as UniversalItem[]) || []);
      setCanvasDomain(keys.canvasDomain);
    } catch (error) {
      console.error('Failed to load course data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSourceLink = (url: string) => {
    if (!url) return '#';
    let finalUrl = url;
    
    // Handle canvas:// protocol
    if (finalUrl.startsWith('canvas://')) {
         finalUrl = finalUrl.substring('canvas://'.length);
    }
    
    // If it starts with api/, make it /api/ for consistent replacement
    if (finalUrl.startsWith('api/')) {
        finalUrl = '/' + finalUrl;
    }
    
    // If it's an API url, try to convert to UI url
    finalUrl = finalUrl.replace(/\/api\/v1\//, '/').replace(/\/api\//, '/');
    
    // If it doesn't start with http, and we have a domain, prepend it
    if (!finalUrl.startsWith('http') && canvasDomain) {
        const domain = canvasDomain.startsWith('http') ? canvasDomain : `https://${canvasDomain}`;
        try {
          // Handle cases where finalUrl might be relative path or just path
          if (!finalUrl.startsWith('/')) finalUrl = '/' + finalUrl;
          finalUrl = new URL(finalUrl, domain).toString();
        } catch (e) {
          // Fallback if URL construction fails
          finalUrl = `${domain}${finalUrl.startsWith('/') ? '' : '/'}${finalUrl}`;
        }
    }
    
    return finalUrl;
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'No date';
    return new Date(dateString).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric'
    });
  };

  const SectionHeader = ({ icon: Icon, title, count }: { icon: any, title: string, count?: number }) => (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2 text-slate-700">
        <Icon size={18} />
        <h2 className="font-semibold">{title}</h2>
      </div>
      {count !== undefined && (
        <span className="bg-slate-100 text-slate-500 text-xs px-2 py-0.5 rounded-full font-medium">
          {count}
        </span>
      )}
    </div>
  );

  const syllabusItem = items.find((item) => item.item_type === 'SYLLABUS');
  const assignmentItems = items.filter((item) => item.item_type === 'ASSIGNMENT');
  const otherItems = items.filter((item) => item.item_type !== 'ASSIGNMENT' && item.item_type !== 'SYLLABUS');

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

  if (loading) {
     return (
        <div className="flex h-full items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
     );
  }

  return (
    <div className="flex flex-col h-full bg-slate-50">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-8 py-6 flex-shrink-0 z-10">
        <button 
          onClick={onBack}
          className="flex items-center text-sm text-slate-500 hover:text-slate-900 mb-2 transition-colors"
        >
          <ChevronLeft className="w-4 h-4 mr-1" />
          Back to Dashboard
        </button>
        
        <h1 className="text-2xl font-bold text-slate-900">
            {course ? course.name : 'Loading...'}
        </h1>
        <div className="flex items-center gap-4 mt-1">
             <p className="text-slate-500 text-sm">
                {[course?.course_code, course?.term].filter(Boolean).join(' • ')}
            </p>
            <span className="text-xs text-blue-700 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-100">
              {rules.length} source{rules.length === 1 ? '' : 's'}
            </span>
        </div>
      </div>

      {/* Bento Grid Content */}
      <div className="flex-1 overflow-auto p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto">
            
            {/* Left Column: Syllabus & Sources */}
            <div className="lg:col-span-8 flex flex-col gap-6">
                
                {/* Syllabus Card */}
                <Card className="p-6 border-slate-200 shadow-sm flex-1 min-h-[300px]">
                    <SectionHeader icon={BookOpen} title="Syllabus" />
                    <div className="prose prose-sm prose-slate max-w-none h-full overflow-y-auto max-h-[400px] pr-2 custom-scrollbar">
                        {syllabusItem?.description || syllabusItem?.raw_content_snippet ? (
                            <div dangerouslySetInnerHTML={{ __html: syllabusItem.description || syllabusItem.raw_content_snippet || '' }} />
                        ) : (
                            <div className="flex flex-col items-center justify-center h-40 text-slate-400">
                                <AlertCircle size={32} className="mb-2 opacity-50" />
                                <p>No syllabus content saved yet.</p>
                            </div>
                        )}
                    </div>
                </Card>

                {/* Ingestion Sources */}
                <Card className="p-6 border-slate-200 shadow-sm">
                  <SectionHeader icon={LinkIcon} title="Ingestion Sources" count={rules.length} />
                  {rules.length === 0 ? (
                    <p className="text-sm text-slate-500">No sources stored for this course yet.</p>
                  ) : (
                    <div className="space-y-3">
                      {rules.map((rule) => (
                        <div key={rule.id} className="flex items-start justify-between gap-3 rounded-lg border border-slate-200 p-3 bg-white">
                          <div className="min-w-0">
                            <a 
                              href={getSourceLink(rule.source_url)} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="text-sm font-semibold text-slate-900 truncate hover:text-blue-600 hover:underline block"
                            >
                              {getSourceLink(rule.source_url).replace(/^https?:\/\//, '')}
                            </a>
                            <p className="text-xs text-slate-500 mt-1">
                              Check every {rule.check_frequency_hours || 24}h
                              {rule.last_checked_at ? ` • Last checked ${formatDate(rule.last_checked_at)}` : ''}
                            </p>
                          </div>
                          <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-500/10 uppercase">
                            {rule.category}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>

                {/* Other Materials */}
                <Card className="p-6 border-slate-200 shadow-sm">
                  <SectionHeader icon={FileText} title="Materials" count={otherItems.length} />
                  {otherItems.length === 0 ? (
                    <p className="text-sm text-slate-500">No materials found for this course yet.</p>
                  ) : (
                    <>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {otherItems.slice(0, 4).map((item) => ( // Display fewer items here
                          <Card
                            key={item.id}
                            onClick={() => onSelect((item.item_type as ItemType) || 'UNKNOWN', item.id)}
                            className="p-4 hover:shadow-md transition-all cursor-pointer border-slate-200 bg-white"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <h3 className="font-medium text-slate-900 truncate pr-2" title={item.title}>{item.title}</h3>
                              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-semibold uppercase">
                                {formatItemType(item.item_type)}
                              </span>
                            </div>
                            <p className="text-xs text-slate-500 line-clamp-3">{item.description || item.raw_content_snippet || 'No description saved.'}</p>
                          </Card>
                        ))}
                      </div>
                      {otherItems.length > 4 && ( // Only show button if there are more items
                        <button
                          onClick={() => onViewAllMaterials(courseId)}
                          className="mt-4 flex items-center justify-center w-full px-4 py-2 bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors font-medium text-sm"
                        >
                          View All Materials ({otherItems.length})
                          <ArrowRight className="w-4 h-4 ml-2" />
                        </button>
                      )}
                    </>
                  )}
                </Card>

            </div>

            {/* Right Column: Assignments */}
            <div className="lg:col-span-4 flex flex-col gap-8">
                
                {/* Assignments */}
                <div>
                    <SectionHeader icon={ListTodo} title="Upcoming Assignments" count={assignmentItems.length} />
                    <div className="flex flex-col gap-3">
                        {assignmentItems.slice(0, 6).map((assignment) => (
                            <Card 
                                key={assignment.id} 
                                onClick={() => onSelect('ASSIGNMENT', assignment.id)}
                                className="p-3 hover:shadow-md transition-all cursor-pointer border-slate-200 bg-white group"
                            >
                                <div className="flex justify-between items-start mb-1">
                                    <h3 className="font-medium text-slate-900 line-clamp-2 text-sm group-hover:text-blue-600 transition-colors">
                                        {assignment.title}
                                    </h3>
                                </div>
                                <div className="flex items-center justify-between mt-2">
                                     <div className="flex items-center gap-1 text-xs text-slate-500 bg-slate-50 px-1.5 py-0.5 rounded">
                                        <Calendar size={12} />
                                        <span>{assignment.due_date ? formatDate(assignment.due_date) : 'No due date'}</span>
                                     </div>
                                     {assignment.confidence_score !== null && assignment.confidence_score !== undefined && (
                                         <span className="text-xs font-medium text-slate-400">{Math.round((assignment.confidence_score || 0) * 100)}% confidence</span>
                                     )}
                                </div>
                            </Card>
                        ))}
                        {assignmentItems.length === 0 && <p className="text-slate-400 text-sm italic">No assignments found.</p>}
                    </div>
                </div>

                {/* Quick summary */}
                <Card className="p-4 border-slate-200 bg-white">
                    <SectionHeader icon={Percent} title="Course Snapshot" />
                    <div className="space-y-2 text-sm text-slate-600">
                      <p><span className="font-semibold text-slate-900">{items.length}</span> items saved</p>
                      <p><span className="font-semibold text-slate-900">{rules.length}</span> ingestion sources</p>
                      <p><span className="font-semibold text-slate-900">{assignmentItems.length}</span> assignments tracked</p>
                    </div>
                </Card>

            </div>
        </div>
      </div>
    </div>
  );
}
