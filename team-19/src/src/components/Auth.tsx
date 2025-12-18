import React, { useState } from 'react';
import { Button, Input, Label, Card } from './ui';
import { Sparkles } from 'lucide-react';

interface AuthProps {
  onLogin: () => void;
}

export default function Auth({ onLogin }: AuthProps) {
  const [formData, setFormData] = useState({
    canvasToken: '',
    notionToken: '',
    canvasDomain: 'canvas.instructure.com',
    notionDbId: '',
    openaiKey: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    window.canvasGPT.login(formData);
    onLogin();
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-slate-50 p-4 font-sans text-slate-900 selection:bg-blue-100">
      <div className="h-8 w-full absolute top-0 left-0 titlebar-drag-region z-50 pointer-events-none" />
      <div className="mb-8 flex flex-col items-center space-y-2">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-600/20">
            <Sparkles className="h-6 w-6" />
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">Welcome to CanvasGPT</h1>
        <p className="text-sm text-slate-500">Sync your academic life with AI.</p>
      </div>

      <Card className="w-full max-w-md p-8 shadow-xl shadow-slate-200/50 border-slate-200/60 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="domain">Canvas Domain</Label>
            <Input 
              id="domain" 
              placeholder="e.g. canvas.instructure.com" 
              value={formData.canvasDomain}
              onChange={(e) => setFormData({...formData, canvasDomain: e.target.value})}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="canvasToken">Canvas Access Token</Label>
            <Input 
              id="canvasToken" 
              type="password" 
              placeholder="Your Canvas API Token" 
              value={formData.canvasToken}
              onChange={(e) => setFormData({...formData, canvasToken: e.target.value})}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="notionToken">Notion Integration Token</Label>
            <Input 
              id="notionToken" 
              type="password" 
              placeholder="secret_..." 
              value={formData.notionToken}
              onChange={(e) => setFormData({...formData, notionToken: e.target.value})}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="notionDb">Notion Database ID</Label>
            <Input 
              id="notionDb" 
              placeholder="Database ID from Notion URL" 
              value={formData.notionDbId}
              onChange={(e) => setFormData({...formData, notionDbId: e.target.value})}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="openaiKey">OpenAI API Key <span className="text-slate-400 font-normal">Local Models Coming Soon</span></Label>
            <Input 
              id="openaiKey" 
              type="password" 
              placeholder="sk-..." 
              value={formData.openaiKey}
              onChange={(e) => setFormData({...formData, openaiKey: e.target.value})}
            />
          </div>
          <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 h-10 text-base shadow-md shadow-blue-600/10">
            Connect Accounts
          </Button>
        </form>
      </Card>
      
      <p className="mt-8 text-center text-xs text-slate-400">
        Your keys are stored locally and securely on your device.
      </p>
    </div>
  );
}
