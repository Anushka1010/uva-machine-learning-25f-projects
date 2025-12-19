import { ipcRenderer, contextBridge } from 'electron'
import { AuthKeys } from './services/auth'

// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld('ipcRenderer', {
  on(channel: string, func: (...args: any[]) => void) {
    const subscription = (_event: any, ...args: any[]) => func(...args)
    ipcRenderer.on(channel, subscription)
    return () => {
      ipcRenderer.removeListener(channel, subscription)
    }
  },
  send(...args: Parameters<typeof ipcRenderer.send>) {
    const [channel, ...omit] = args
    return ipcRenderer.send(channel, ...omit)
  },
  invoke(...args: Parameters<typeof ipcRenderer.invoke>) {
    const [channel, ...omit] = args
    return ipcRenderer.invoke(channel, ...omit)
  },
})

contextBridge.exposeInMainWorld('canvasGPT', {
  login: (tokens: AuthKeys) => ipcRenderer.send('save-keys', tokens),
  askQuestion: (query: string) => ipcRenderer.invoke('ask-question', query),
  getKeys: () => ipcRenderer.invoke('get-keys'),
  clearKeys: () => ipcRenderer.send('clear-keys'),
  setWindowSize: (options: { width: number, height: number, minWidth: number, minHeight: number }) => ipcRenderer.invoke('set-window-size', options),
  getCourses: () => ipcRenderer.invoke('get-courses'),
  getCourseDetails: (courseId: string) => ipcRenderer.invoke('get-course-details', courseId),
  getUpcomingItems: () => ipcRenderer.invoke('get-upcoming-items'),
  getTriageItems: () => ipcRenderer.invoke('get-triage-items'),
  getUniversalItem: (id: number) => ipcRenderer.invoke('get-universal-item', id),
  discoverCourses: () => ipcRenderer.invoke('discover-courses'),
  syncAll: () => ipcRenderer.invoke('sync-all'),
  openExternal: (url: string) => ipcRenderer.invoke('open-external', url),
})
