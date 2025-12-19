import { useEffect, useState } from 'react';
import { ChevronLeft, Calendar } from 'lucide-react';
import { Card } from './ui';
import { ItemType, UniversalItem } from '../lib/types';

interface MaterialsListViewProps {
  courseId: string;
  onBack: () => void;
  onSelect: (type: ItemType, id: number) => void;
}

export default function MaterialsListView({ courseId, onBack, onSelect }: MaterialsListViewProps) {
  const [courseName, setCourseName] = useState('');
  const [materials, setMaterials] = useState<UniversalItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadMaterials = async () => {
      setLoading(true);
      try {
        const data = await window.canvasGPT.getCourseDetails(courseId);
        setCourseName(data?.course?.name || 'Course Materials');
        // Filter out assignments and syllabus, similar to otherItems in CourseView
        const filteredMaterials = (data?.items || []).filter(
          (item) => item.item_type !== 'ASSIGNMENT' && item.item_type !== 'SYLLABUS'
        );
        setMaterials(filteredMaterials);
      } catch (error) {
        console.error('Failed to load materials:', error);
      } finally {
        setLoading(false);
      }
    };

    loadMaterials();
  }, [courseId]);

  const formatDate = (dateString: string) => {
    if (!dateString) return 'No date';
    return new Date(dateString).toLocaleDateString(undefined, {
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
      case 'ANNOUNCEMENT':
        return 'Announcement';
      case 'QUIZ':
        return 'Quiz';
      case 'FILE':
        return 'File';
      case 'PAGE':
        return 'Page';
      case 'DISCUSSION':
        return 'Discussion';
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
    <div className="flex flex-col h-full ">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-8 py-6 flex-shrink-0 z-10">
        <button
          onClick={onBack}
          className="flex items-center text-sm text-slate-500 hover:text-slate-900 mb-2 transition-colors"
        >
          <ChevronLeft className="w-4 h-4 mr-1" />
          Back to Course
        </button>
        <h1 className="text-2xl font-bold text-slate-900">{courseName} Materials</h1>
      </div>

      {/* Materials List */}
      <div className="flex-1 overflow-auto p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {materials.length === 0 ? (
            <p className="text-slate-500 text-lg md:col-span-3 lg:col-span-3">No additional materials found for this course.</p>
          ) : (
            materials.map((item) => (
              <Card
                key={item.id}
                onClick={() => onSelect((item.item_type as ItemType) || 'UNKNOWN', item.id)}
                className="p-4 hover:shadow-md transition-all cursor-pointer border-slate-200 bg-white"
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-slate-900 truncate pr-2" title={item.title}>
                    {item.title}
                  </h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-semibold uppercase">
                    {formatItemType(item.item_type)}
                  </span>
                </div>
                <p className="text-xs text-slate-500 line-clamp-3">
                  {item.description || item.raw_content_snippet || 'No description saved.'}
                </p>
                {item.due_date && (
                  <div className="flex items-center gap-1 text-xs text-slate-500 bg-slate-50 px-1.5 py-0.5 rounded mt-2 w-fit">
                    <Calendar size={12} />
                    <span>{formatDate(item.due_date)}</span>
                  </div>
                )}
              </Card>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
