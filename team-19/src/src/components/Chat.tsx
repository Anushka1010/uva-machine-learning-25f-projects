import { useState } from 'react';
import { Input } from './ui';
import { Loader2, Bot } from 'lucide-react';
import { cn } from '../lib/utils';

export default function Chat() {
  const [messages, setMessages] = useState<Array<{role: 'user' | 'assistant', content: string}>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!input.trim()) return;
    
    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);
    
    try {
      // @ts-ignore - window.canvasGPT is defined in preload
      const response = await window.canvasGPT.askQuestion(userMsg);
      setMessages(prev => [...prev, { role: 'assistant', content: response }]);
    } catch (error: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex items-center justify-between pb-4 border-b border-slate-100">
        <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">Van Gogh</h1>
            <p className="text-slate-500 text-sm">like painting on canvas?</p>
        </div>
      </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-400 space-y-2">
                <Bot className="h-12 w-12 opacity-50" />
                <p>No messages yet. Ask something!</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={cn("flex gap-3", msg.role === 'user' ? 'flex-row-reverse' : 'flex-row')}>
                <div className={cn(
                  "rounded-2xl px-4 py-2 text-sm max-w-[80%] bg-white rounded-tl-none",
                  msg.role === 'user' 
                    ? 'text-slate-400'
                    : 'text-slate-700'
                )}>
                  <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="flex gap-3">
                    <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
                    <span className="text-xs text-slate-400">Thinking...</span>
            </div>
          )}
        </div>`
        <Input
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
          disabled={loading}
          className="bg-white shadow-sm border-slate-200 focus:ring-blue-500/20"
        />
    </div>
  );
}
