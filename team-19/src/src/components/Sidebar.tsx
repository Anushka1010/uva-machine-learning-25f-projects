import { cn } from '../lib/utils';
import { LayoutDashboard, Palette, ListTodo, Settings, LogOut, AlertTriangle} from 'lucide-react';
import clsx from 'clsx';
import { View } from '../lib/types';

interface SidebarProps {
  currentView: View;
  onChangeView: (view: View) => void;
  onLogout: () => void;
}

export default function Sidebar({ currentView, onChangeView, onLogout }: SidebarProps) {
  const navItems = [
    { id: 'triage', label: 'Triage', icon: AlertTriangle },
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'assignments', label: 'Upcoming', icon: ListTodo },
    { id: 'chat', label: 'Van Gogh', icon:  Palette},
  ] as const;

  return (
    <div className="flex w-64 flex-col border-r border-slate-200 bg-slate-50/50 pt-8 backdrop-blur-xl">
      <div className="px-6 mb-8 drag-region">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500">CanvasGPT</h2>
      </div>
      
      <nav className="flex-1 space-y-1 px-3">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onChangeView(item.id)}
            className={clsx(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors no-drag",
              currentView === item.id
                ? "bg-white text-blue-600 shadow-sm ring-1 ring-slate-200"
                : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
            )}
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </button>
        ))}
      </nav>

      <div className="border-t border-slate-200 p-3 space-y-1">
        <button
            onClick={() => onChangeView('settings')}
            className={cn(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors no-drag",
              currentView === 'settings'
                ? "bg-white text-blue-600 shadow-sm ring-1 ring-slate-200"
                : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
            )}
          >
            <Settings className="h-4 w-4" />
            Settings
        </button>
        <button
          onClick={onLogout}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 no-drag"
        >
          <LogOut className="h-4 w-4" />
          Disconnect
        </button>
      </div>
    </div>
  );
}
