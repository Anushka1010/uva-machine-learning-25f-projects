import React from 'react';
import Sidebar from './Sidebar';
import { View } from '../lib/types';

interface LayoutProps {
  children: React.ReactNode;
  currentView: View;
  onChangeView: (view: View) => void;
  onLogout: () => void;
}

export default function Layout({ children, currentView, onChangeView, onLogout }: LayoutProps) {
  return (
    <div className="flex-1 overflow-auto bg-white relative">
      <div className="h-8 w-full absolute top-0 left-0 titlebar-drag-region z-50 pointer-events-none" />
      <div className="flex h-screen w-full overflow-hidden bg-white text-slate-900">
        <Sidebar currentView={currentView} onChangeView={onChangeView} onLogout={onLogout} />
        <main className="flex-1 overflow-auto bg-white relative">
          <div className="h-full p-8 pt-10">
              {children}
          </div>
        </main>
      </div>
    </div>
  );
}
