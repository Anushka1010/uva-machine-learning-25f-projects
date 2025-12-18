import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, ListTodo, FileText } from 'lucide-react';
import { Card } from './ui';
import { cn } from '../lib/utils';
import { ItemType } from '../lib/types';

interface TriageItem {
  item_type: ItemType | string;
  id: number;
  title: string;
  course_name: string;
  course_code?: string;
  reasons: string[];
}

interface TriageProps {
  onSelect: (type: ItemType, id: number) => void;
}

export default function Triage({ onSelect }: TriageProps) {
  const [items, setItems] = useState<TriageItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchItems();
  }, []);

  const fetchItems = async () => {
    try {
      setLoading(true);
      const data = await window.canvasGPT.getTriageItems();
      setItems(data);
    } catch (error) {
      console.error('Failed to fetch triage items:', error);
    } finally {
      setLoading(false);
    }
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'ASSIGNMENT': return <ListTodo size={20} />;
      case 'READING':
      case 'SLIDE':
      case 'SYLLABUS':
        return <FileText size={20} />;
      default: return <AlertTriangle size={20} />;
    }
  };

  return (
    <div className="space-y-8 h-full flex flex-col">
      <div className="flex items-center justify-between pb-6 border-b border-slate-100 flex-shrink-0">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Triage</h1>
          <p className="text-slate-500 text-sm">Items needing attention or with missing details.</p>
        </div>
        <div className="bg-amber-50 text-amber-700 px-3 py-1 rounded-full text-xs font-medium border border-amber-100 flex items-center gap-1">
            <AlertTriangle size={12} />
            {items.length} Issues Found
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
                <h3 className="text-lg font-medium text-slate-900">No issues found!</h3>
                <p className="text-slate-500 mt-1">Everything seems to have complete information.</p>
            </div>
        ) : (
            <div className="grid grid-cols-1 gap-4 pb-8">
                {items.map((item) => (
                    <Card 
                        key={`${item.item_type}-${item.id}`} 
                        onClick={() => onSelect((item.item_type as ItemType) || 'UNKNOWN', item.id)}
                        className="flex items-start p-4 hover:shadow-md transition-shadow border-slate-200 cursor-pointer group"
                    >
                        <div className={cn(
                            "p-2 rounded-lg mr-4 flex-shrink-0 transition-colors",
                            item.item_type === 'ASSIGNMENT'
                              ? "bg-slate-100 text-slate-500 group-hover:bg-blue-50 group-hover:text-blue-600"
                              : "bg-slate-100 text-slate-500 group-hover:bg-slate-200 group-hover:text-slate-700"
                        )}>
                            {getIcon(item.item_type)}
                        </div>
                        
                        <div className="flex-1 min-w-0">
                            <div className="flex justify-between items-start mb-1">
                                <h4 className="text-base font-semibold text-slate-900 truncate pr-2">
                                    {item.title || 'Untitled'}
                                </h4>
                            </div>
                            
                            <p className="text-xs text-slate-500 font-medium mb-3">
                                {item.course_name || 'Unknown course'}
                            </p>

                            <div className="flex flex-wrap gap-2">
                                {item.reasons.map((reason, idx) => (
                                    <span key={idx} className="inline-flex items-center rounded-md bg-red-50 px-2 py-1 text-xs font-medium text-red-700 ring-1 ring-inset ring-red-600/10">
                                        {reason}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </Card>
                ))}
            </div>
        )}
      </div>
    </div>
  );
}
