/// <reference types="vite-plugin-electron/electron-env" />

declare namespace NodeJS {
  interface ProcessEnv {
    /* The built directory structure

       ```tree
       ├─┬─┬ dist
       │ │ └── index.html
       │ │
       │ ├─┬ dist-electron
       │ │ ├── main.js
       │ │ └── preload.js
       │
       ``` */
    APP_ROOT: string
    /* /dist/ or /public/ */
    VITE_PUBLIC: string
  }
}

// Used in Renderer process, expose in `preload.ts`
interface Window {
  ipcRenderer: import('electron').IpcRenderer
  canvasGPT: {
    login: (tokens: { canvasToken: string; notionToken: string; canvasDomain: string; notionDbId: string }) => void
    startSync: () => Promise<{ success: boolean; message: string }>
    getCourses: () => Promise<any[]>
    discoverCourses: () => Promise<{
      success: boolean;
      message?: string;
      results?: Array<{
        courseId: string;
        name?: string;
        archetype: string | null;
        foundLinks: string[];
        savedRules: number;
      }>;
    }>
    askQuestion: (query: string) => Promise<string>
    getKeys: () => Promise<{ canvasToken: string; notionToken: string; canvasDomain: string; notionDbId: string }>
    clearKeys: () => void
  }
}
