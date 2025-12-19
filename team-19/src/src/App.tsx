import { useEffect, useState } from 'react';
import Auth from './components/Auth';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import Chat from './components/Chat';
import Assignments from './components/Assignments';
import Settings from './components/Settings';
import CourseView from './components/CourseView';
import ObjectView from './components/ObjectView';
import Triage from './components/Triage';
import MaterialsListView from './components/MaterialsListView'; // New import
import { Loader2 } from 'lucide-react';
import { ItemType, View } from './lib/types'; // Updated import

function App() {
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentView, setCurrentView] = useState<View>('chat');
  const [selectedCourseId, setSelectedCourseId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<{ type: ItemType; id: number } | null>(null);
  const [previousView, setPreviousView] = useState<View>('dashboard');

  const AUTH_WINDOW_SIZE = { width: 600, height: 800, minWidth: 600, minHeight: 800 };
  const MAIN_WINDOW_SIZE = { width: 1280, height: 960, minWidth: 1280, minHeight: 960 };

  useEffect(() => {
    checkAuth();
  }, []);

  // Effect to handle window resizing based on authentication state
  useEffect(() => {
    // @ts-ignore
    if (window.canvasGPT && window.canvasGPT.setWindowSize) {
      if (isAuthenticated) {
        // Set main window size
        // @ts-ignore
        window.canvasGPT.setWindowSize(MAIN_WINDOW_SIZE);
      } else {
        // Set auth window size
        // @ts-ignore
        window.canvasGPT.setWindowSize(AUTH_WINDOW_SIZE);
      }
    }
  }, [isAuthenticated]);

  const checkAuth = async () => {
    setLoading(true);
    try {
      // @ts-ignore
      const keys = await window.canvasGPT.getKeys();
      if (keys.canvasToken && keys.notionToken && keys.notionDbId) {
        setIsAuthenticated(true);
      } else {
        setIsAuthenticated(false);
      }
    } catch (e) {
      console.error(e);
      setIsAuthenticated(false);
    } finally {
      setLoading(false);
    }
  };

  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
    checkAuth(); // Re-check auth state after login (will trigger size change)
  };

  const handleLogout = () => {
    // @ts-ignore
    window.canvasGPT.clearKeys();
    setIsAuthenticated(false); // This will trigger the size change in useEffect
    setCurrentView('dashboard');
    setSelectedCourseId(null);
    setSelectedDetail(null);
  };

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <Dashboard
            onCourseSelect={(id) => {
              setSelectedCourseId(id);
              setCurrentView('course');
            }}
          />
        );
      case 'course':
        if (!selectedCourseId) return <Dashboard onCourseSelect={(id) => { setSelectedCourseId(id); setCurrentView('course'); }} />;
        return (
          <CourseView
            courseId={selectedCourseId}
            onBack={() => setCurrentView('dashboard')}
            onSelect={(type, id) => {
              setSelectedDetail({ type, id });
              setPreviousView('course');
              setCurrentView('detail_view');
            }}
            onViewAllMaterials={(courseId: string) => { // New prop
              setSelectedCourseId(courseId);
              setPreviousView('course');
              setCurrentView('materials_list');
            }}
          />
        );
      case 'materials_list': // New case
        if (!selectedCourseId) return <Dashboard onCourseSelect={(id) => { setSelectedCourseId(id); setCurrentView('course'); }} />;
        return (
          <MaterialsListView
            courseId={selectedCourseId}
            onBack={() => setCurrentView(previousView)} // Go back to previous view (course)
            onSelect={(type, id) => {
              setSelectedDetail({ type, id });
              setPreviousView('materials_list');
              setCurrentView('detail_view');
            }}
          />
        );
      case 'detail_view':
        if (!selectedDetail) return <Dashboard onCourseSelect={(id) => { setSelectedCourseId(id); setCurrentView('course'); }} />;
        return (
          <ObjectView
            type={selectedDetail.type}
            id={selectedDetail.id}
            onBack={() => setCurrentView(previousView)}
          />
        );
      case 'assignments':
        return (
          <Assignments
            onSelect={(type, id) => {
              setSelectedDetail({ type, id });
              setPreviousView('assignments');
              setCurrentView('detail_view');
            }}
          />
        );
      case 'triage':
        return (
          <Triage
            onSelect={(type, id) => {
              setSelectedDetail({ type, id });
              setPreviousView('triage');
              setCurrentView('detail_view');
            }}
          />
        );
      case 'chat':
        return <Chat />;
      case 'settings':
        return <Settings onLogout={handleLogout} />;
      default:
        return (
          <Dashboard
            onCourseSelect={(id) => {
              setSelectedCourseId(id);
              setCurrentView('course');
            }}
          />
        );
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-white">
        <Loader2 className="h-8 w-8 animate-spin text-slate-300" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Auth onLogin={handleLoginSuccess} />;
  }

  return (
    <Layout
      currentView={currentView}
      onChangeView={setCurrentView}
      onLogout={handleLogout}
    >
      {renderView()}
    </Layout>
  );
}

export default App;
