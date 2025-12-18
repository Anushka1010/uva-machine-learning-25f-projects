import { useState } from 'react';
import { Trash2, LogOut, AlertTriangle, Sparkles, RefreshCw } from 'lucide-react';

interface SettingsProps {
  onLogout: () => void;
}

export default function Settings({ onLogout }: SettingsProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [message, setMessage] = useState('');
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [discoveryMessage, setDiscoveryMessage] = useState('');
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState('');

  const handleDeleteAllData = async () => {
    setIsDeleting(true);
    setMessage('Deleting all data...');
    try {
      // @ts-ignore
      const result = await window.ipcRenderer.invoke('delete-all-data');
      if (result.success) {
        setMessage('All data deleted successfully.');
        setTimeout(() => {
            setShowConfirm(false);
            setMessage('');
        }, 2000);
      } else {
        setMessage(`Error: ${result.error}`);
      }
    } catch (e: any) {
      setMessage(`Error: ${e.message}`);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDiscoverCourses = async () => {
    setIsDiscovering(true);
    setDiscoveryMessage('Running discovery agent...');
    try {
      // @ts-ignore
      const result = await window.canvasGPT.discoverCourses();

      if (result.success) {
        const summaries = (result.results || []).map((item: any) => {
          const ruleLabel = `${item.savedRules} rule${item.savedRules === 1 ? '' : 's'}`;
          const archetype = item.archetype || 'unknown';
          return `${item.name || item.courseId}: ${archetype} (${ruleLabel})`;
        });

        setDiscoveryMessage(
          summaries.length
            ? `Discovery complete:\n${summaries.join('\n')}`
            : 'Discovery completed, but no rules were saved.'
        );
      } else {
        setDiscoveryMessage(result.message || 'Discovery failed to run.');
      }
    } catch (e: any) {
      setDiscoveryMessage(`Error: ${e.message || e}`);
    } finally {
      setIsDiscovering(false);
    }
  };

  const handleManualSync = async () => {
    setIsSyncing(true);
    setSyncMessage('Syncing data from all sources...');
    try {
      // @ts-ignore
      const result = await window.canvasGPT.syncAll();
      
      if (result.success) {
        setSyncMessage('Sync completed successfully.');
      } else {
        setSyncMessage(`Sync failed: ${result.message || result.error || 'Unknown error'}`);
      }
    } catch (e: any) {
      setSyncMessage(`Error: ${e.message}`);
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Settings</h2>
        <p className="text-slate-500 mt-1">Manage your application data and account.</p>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <h3 className="text-lg font-medium text-slate-900">Manual Sync</h3>
          <p className="text-sm text-slate-500 mt-1">
            Force a sync of all ingestion rules and Canvas data.
          </p>
        </div>
        <div className="p-6 bg-slate-50/50 space-y-3">
          <button
            onClick={handleManualSync}
            disabled={isSyncing}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors font-medium disabled:opacity-60"
          >
            <RefreshCw className={`h-4 w-4 ${isSyncing ? 'animate-spin' : ''}`} />
            {isSyncing ? 'Syncing...' : 'Sync Now'}
          </button>
          {syncMessage && (
            <pre className="whitespace-pre-wrap text-sm text-slate-600 bg-white border border-slate-200 rounded-md p-3">
              {syncMessage}
            </pre>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <h3 className="text-lg font-medium text-slate-900">Discovery Setup</h3>
          <p className="text-sm text-slate-500 mt-1">
            Run the LangGraph discovery agent to inspect your courses and seed ingestion rules.
          </p>
        </div>
        <div className="p-6 bg-slate-50/50 space-y-3">
          <button
            onClick={handleDiscoverCourses}
            disabled={isDiscovering}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors font-medium disabled:opacity-60"
          >
            <Sparkles className="h-4 w-4" />
            {isDiscovering ? 'Running discovery...' : 'Initialize Discovery'}
          </button>
          {discoveryMessage && (
            <pre className="whitespace-pre-wrap text-sm text-slate-600 bg-white border border-slate-200 rounded-md p-3">
              {discoveryMessage}
            </pre>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <h3 className="text-lg font-medium text-slate-900">Data Management</h3>
          <p className="text-sm text-slate-500 mt-1">
            Clear all locally cached data from Canvas and Notion. This includes downloaded files, vector embeddings, and the knowledge graph. 
            Your API keys and authentication tokens will NOT be deleted.
          </p>
        </div>
        <div className="p-6 bg-slate-50/50">
          {!showConfirm ? (
            <button
              onClick={() => setShowConfirm(true)}
              className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 rounded-md hover:bg-red-100 transition-colors font-medium border border-red-200"
            >
              <Trash2 className="h-4 w-4" />
              Delete All Data
            </button>
          ) : (
            <div className="bg-white p-4 rounded-md border border-red-200 shadow-sm max-w-md">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-red-100 rounded-full text-red-600">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="font-medium text-slate-900">Are you sure?</h4>
                  <p className="text-sm text-slate-500 mt-1">
                    This action cannot be undone. You will need to re-sync your data to use the application effectively.
                  </p>
                  
                  {message && (
                    <p className={`text-sm mt-2 ${message.includes('Error') ? 'text-red-600' : 'text-green-600'}`}>
                        {message}
                    </p>
                  )}

                  <div className="flex gap-3 mt-4">
                    <button
                      onClick={handleDeleteAllData}
                      disabled={isDeleting}
                      className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm font-medium disabled:opacity-50"
                    >
                      {isDeleting ? 'Deleting...' : 'Yes, Delete Everything'}
                    </button>
                    <button
                      onClick={() => { setShowConfirm(false); setMessage(''); }}
                      disabled={isDeleting}
                      className="px-3 py-2 bg-white text-slate-700 border border-slate-300 rounded-md hover:bg-slate-50 text-sm font-medium"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <h3 className="text-lg font-medium text-slate-900">Account</h3>
          <p className="text-sm text-slate-500 mt-1">
            Sign out of your current session. This will remove your API keys from this device.
          </p>
        </div>
        <div className="p-6 bg-slate-50/50">
          <button
            onClick={onLogout}
            className="flex items-center gap-2 px-4 py-2 bg-white text-slate-700 border border-slate-300 rounded-md hover:bg-slate-50 transition-colors font-medium shadow-sm"
          >
            <LogOut className="h-4 w-4" />
            Disconnect & Logout
          </button>
        </div>
      </div>
    </div>
  );
}
