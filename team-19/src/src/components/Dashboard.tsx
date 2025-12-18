import { useEffect, useState } from 'react';
import { Card } from './ui';
import { BookOpen } from 'lucide-react';

interface Course {
  id: string;
  name: string;
  course_code?: string;
  term?: string;
  color_hex?: string;
  is_active?: boolean;
}

interface DashboardProps {
  onCourseSelect: (courseId: string) => void;
}

export default function Dashboard({ onCourseSelect }: DashboardProps) {
  const [courses, setCourses] = useState<Course[]>([]);

  useEffect(() => {
    fetchCourses();

    const cleanup = window.ipcRenderer.on('data-updated', () => {
      fetchCourses();
    });

    return () => {
      cleanup();
    };
  }, []);

  const fetchCourses = async () => {
    try {
      // @ts-ignore
      const data = await window.canvasGPT.getCourses();
      setCourses((data as Course[]) || []);
    } catch (e) {
      console.error("Failed to fetch courses", e);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between pb-6 border-b border-slate-100">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Dashboard</h1>
          <p className="text-slate-500 text-sm">Your active courses.</p>
        </div>
      </div>

      {/* Stats / Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {courses.length === 0 ? (
            <div className="col-span-full flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50/50 p-16 text-center">
                <div className="rounded-full bg-slate-100 p-3 mb-4">
                    <BookOpen className="h-6 w-6 text-slate-400" />
                </div>
                <h3 className="text-base font-semibold text-slate-900">No courses found</h3>
                <p className="mt-1 text-sm text-slate-500 max-w-sm">
                    We couldn't find any active courses. Check your Canvas connection or ensure you are enrolled.
                </p>
            </div>
        ) : (
            courses.map((course) => (
                <Card 
                    key={course.id} 
                    onClick={() => onCourseSelect(course.id)}
                    className="group relative overflow-hidden p-6 hover:shadow-lg transition-all hover:-translate-y-1 cursor-pointer"
                >
                    <div className="flex flex-col space-y-3">
                        <div className="flex justify-between items-start">
                            <div
                              className="h-10 w-10 rounded-lg flex items-center justify-center text-white font-bold"
                              style={{ backgroundColor: course.color_hex || '#2563eb' }}
                            >
                                <BookOpen className="h-5 w-5" />
                            </div>
                            <span className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ring-1 ring-inset ${
                              course.is_active === false
                                ? "bg-slate-100 text-slate-600 ring-slate-200"
                                : "bg-green-50 text-green-700 ring-green-600/20"
                            }`}>
                                {course.is_active === false ? 'Inactive' : 'Active'}
                            </span>
                        </div>
                        <div>
                            <h3 className="font-semibold tracking-tight text-slate-900 line-clamp-1" title={course.name}>
                                {course.name}
                            </h3>
                            <p className="text-sm text-slate-500 mt-1">
                              {[course.course_code, course.term].filter(Boolean).join(' • ')}
                            </p>
                        </div>
                    </div>
                </Card>
            ))
        )}
      </div>
    </div>
  );
}
