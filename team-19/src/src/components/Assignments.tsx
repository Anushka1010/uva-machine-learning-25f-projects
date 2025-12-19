import { useEffect, useState } from 'react';
import { ListTodo, Clock, CheckCircle2, FileText, ExternalLink } from 'lucide-react';
import { Card } from './ui';
import { cn } from '../lib/utils';
import { ItemType } from '../lib/types';

interface UpcomingItem {
  id: number;
  course_id: string;
  title: string;
  date?: string | null;
  item_type: ItemType | string;
  course_name?: string;
  course_code?: string;
  description?: string;
  content_url?: string;
  confidence_score?: number | null;
}

interface AssignmentsProps {
  onSelect: (type: ItemType, id: number) => void;
}

export default function Assignments({ onSelect }: AssignmentsProps) {
  const [items, setItems] = useState<UpcomingItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchItems();
    
    // @ts-ignore
    const cleanup = window.ipcRenderer.on('data-updated', () => {
        fetchItems();
    });
    return () => cleanup();
  }, []);

  const fetchItems = async () => {
    try {
      setLoading(true);
      const data = await window.canvasGPT.getUpcomingItems();
      setItems(data);
    } catch (error) {
      console.error('Failed to fetch upcoming items:', error);
    } finally {
      setLoading(false);
    }
  };

  const groupItemsByDate = (items: UpcomingItem[]) => {
    const groups: { [key: string]: UpcomingItem[] } = {};
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    items.forEach(item => {
      if (!item.date) {
        if (!groups['Undated']) groups['Undated'] = [];
        groups['Undated'].push(item);
        return;
      }

      const date = new Date(item.date);
      date.setHours(0, 0, 0, 0);
      
      let key = date.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' });
      
      if (date.getTime() === today.getTime()) key = 'Today';
      else if (date.getTime() === tomorrow.getTime()) key = 'Tomorrow';
      else if (date.getTime() < today.getTime()) key = 'Overdue / Past';

      if (!groups[key]) groups[key] = [];
      groups[key].push(item);
    });

    return groups;
  };

  const groupedItems = groupItemsByDate(items);

  const formatTime = (dateString: string) => {
    if (!dateString) return '';
    return new Date(dateString).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
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

  return (
    <div className="space-y-8 h-full flex flex-col">
      <div className="flex items-center justify-between pb-6 border-b border-slate-100 flex-shrink-0">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Upcoming</h1>
          <p className="text-slate-500 text-sm">Your timeline of assignments and announcements.</p>
        </div>
      </div>

      <div className="flex-1 overflow-auto -mx-6 px-6">
        {loading ? (
           <div className="space-y-4">
               {[1, 2, 3].map(i => (
                   <div key={i} className="h-24 bg-slate-50 rounded-xl animate-pulse" />
               ))}
           </div>
        ) : items.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
                <div className="bg-green-50 p-4 rounded-full mb-4">
                    <CheckCircle2 className="h-8 w-8 text-green-500" />
                </div>
                <h3 className="text-lg font-medium text-slate-900">All caught up!</h3>
                <p className="text-slate-500 mt-1">No upcoming assignments or recent announcements found.</p>
            </div>
        ) : (
            <div className="space-y-8 pb-8">
                {Object.entries(groupedItems).map(([dateLabel, groupItems]) => (
                    <div key={dateLabel}>
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 sticky top-0 bg-white py-2 z-10 border-b border-white/0">
                            {dateLabel}
                        </h3>
                        <div className="space-y-3">
                            {groupItems.map(item => (
                                <Card 
                                    key={`${item.item_type}-${item.id}`} 
                                    onClick={() => onSelect((item.item_type as ItemType) || 'UNKNOWN', item.id)}
                                    className="flex items-start p-4 hover:shadow-md transition-shadow border-slate-200 cursor-pointer"
                                >
                                    <div className={cn(
                                        "p-2 rounded-lg mr-4 flex-shrink-0",
                                        item.item_type === 'ASSIGNMENT' ? "bg-blue-50 text-blue-600" : "bg-amber-50 text-amber-600"
                                    )}>
                                        {item.item_type === 'ASSIGNMENT' ? <ListTodo size={20} /> : <FileText size={20} />}
                                    </div>
                                    
                                    <div className="flex-1 min-w-0">
                                        <div className="flex justify-between items-start">
                                            <h4 className="text-base font-semibold text-slate-900 truncate pr-2">
                                                {item.title}
                                            </h4>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs font-medium text-slate-500 whitespace-nowrap bg-slate-50 px-2 py-1 rounded">
                                                    {formatItemType(item.item_type)}
                                                </span>
                                                {item.content_url && (
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            if (item.content_url) {
                                                                window.canvasGPT.openExternal(item.content_url);
                                                            }
                                                        }}
                                                        className="p-1.5 rounded hover:bg-blue-50 text-slate-400 hover:text-blue-600 transition-colors"
                                                        title="Open in Canvas"
                                                    >
                                                        <ExternalLink size={14} />
                                                    </button>
                                                )}
                                            </div>
                                        </div>
                                        
                                        <p className="text-xs text-slate-500 font-medium mt-0.5">
                                            {item.course_name || 'Unknown course'}
                                        </p>

                                        <div className="flex items-center gap-4 mt-3 text-xs text-slate-400">
                                            {item.date && (
                                                <div className="flex items-center gap-1">
                                                    <Clock size={12} />
                                                    <span>Due {formatTime(item.date)}</span>
                                                </div>
                                            )}
                                            {item.confidence_score !== null && item.confidence_score !== undefined && (
                                                <span className="truncate">{Math.round((item.confidence_score || 0) * 100)}% confidence</span>
                                            )}
                                        </div>
                                    </div>
                                </Card>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        )}
      </div>
    </div>
  );
}
