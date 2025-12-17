"use client";

import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Badge from "@/components/ui/Badge";
import { useEffect, useMemo, useState } from "react";
import { api_listEvents, api_scanIn, api_scanOut, mockInfer } from "@/lib/mockDb";
import { formatDate } from "@/lib/utils";
import { listCategoryImages } from "@/lib/categoryImages";

export default function ScanPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [last, setLast] = useState<any>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [manualLabel, setManualLabel] = useState<string>("");

  const classes = Object.keys(listCategoryImages());
  const FLASK_API_URL = "http://127.0.0.1:5000"; // Ideally put this in .env

  async function loadEvents() {
    const ev = await api_listEvents();
    setEvents(ev);
  }

  useEffect(() => { loadEvents(); }, []);

  const detections = useMemo(() => {
    if (!file) return [];
    return mockInfer(file.name);
  }, [file]);

  function onPick(f: File | null) {
    setFile(f);
    setLast(null);
    if (!f) return setPreview("");
    const url = URL.createObjectURL(f);
    setPreview(url);
  }

  async function scanIn() {
    if (!file) return alert("Upload an image first.");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("type", "IN"); 

      const response = await fetch(`${FLASK_API_URL}/api/scan`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Server error");

      const data = await response.json();

      setLast(data);
      await loadEvents(); 
    } catch (error) {
      console.error("Scan failed:", error);
      alert("Failed to send data to server.");
    }
  }

  async function scanOut() {
    if (!file) return alert("Upload an image first.");
    const res = await api_scanOut({ name: file.name, imageUrl: preview });
    setLast(res);
    await loadEvents();
  }

  async function addManual() {
    if (!manualLabel) return alert("Select an item first.");

    try {
      const response = await fetch(`${FLASK_API_URL}/api/manual-add`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          label: manualLabel,
          type: "IN"
        }),
      });

      const data = await response.json();
      setLast(data);
      await loadEvents();
      setManualLabel("");
    } catch (error) {
      console.error("Manual add failed:", error);
    }
  }

  return (
    <LayoutShell>
      <div className="space-y-4">
        <div>
          <h1 className="text-2xl font-semibold">Scan & Events</h1>
          <p className="text-sm text-gray-600">
            Frontend demo: upload a snapshot, preview detections, then run Scan In/Out.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card className="p-4 space-y-3">
            <div className="text-sm font-medium">Snapshot</div>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => onPick(e.target.files?.[0] ?? null)}
              className="text-sm border border-gray-300 rounded px-3 py-2 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />

            {preview && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="" className="w-full max-h-[280px] object-cover rounded border" />
            )}

            <div className="flex gap-2">
              <Button onClick={scanIn}>Run Scan In</Button>
              <Button variant="secondary" onClick={scanOut}>Run Scan Out</Button>
            </div>

            <div className="text-xs text-gray-600">
              Detections are mocked from filename for now; later we’ll call your real model.
            </div>
          </Card>

          <Card className="p-4 space-y-3">
            <div className="text-sm font-medium">Detections Preview</div>
            <div className="flex flex-wrap gap-2">
              {detections.map((d) => (
                <Badge key={d.label}>
                  {d.label} ({Math.round(d.confidence * 100)}%)
                </Badge>
              ))}
              {detections.length === 0 && (
                <div className="text-sm text-gray-600">Upload an image to see detections.</div>
              )}
            </div>

            <div className="text-sm font-medium pt-2">Last Result</div>
            {!last ? (
              <div className="text-sm text-gray-600">Run a scan to see results.</div>
            ) : (
              <pre className="text-xs bg-gray-50 border rounded p-3 overflow-auto max-h-[320px]">
                {JSON.stringify(last, null, 2)}
              </pre>
            )}
          </Card>
        </div>

        <Card className="p-4 space-y-3">
          <div className="text-sm font-medium">Manual Add</div>
          <div className="text-sm text-gray-600">
            Select an item from the list to manually add it to the fridge.
          </div>
          <div className="flex gap-2 items-center">
            <input
              list="item-list"
              value={manualLabel}
              onChange={(e) => setManualLabel(e.target.value)}
              placeholder="Search and select item..."
              className="flex-1 px-3 py-2 border rounded text-sm"
            />
            <datalist id="item-list">
              {classes.map((cls) => (
                <option key={cls} value={cls} />
              ))}
            </datalist>
            <Button onClick={addManual}>Add to Fridge</Button>
          </div>
        </Card>

        <Card className="overflow-hidden">
          <div className="p-3 font-medium text-sm border-b">Event Log</div>
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left p-3">Time</th>
                <th className="text-left p-3">Type</th>
                <th className="text-left p-3">Summary</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id} className="border-t">
                  <td className="p-3">{formatDate(e.timestamp)}</td>
                  <td className="p-3"><Badge>{e.type}</Badge></td>
                  <td className="p-3">{e.resultSummary}</td>
                </tr>
              ))}
              {events.length === 0 && (
                <tr><td className="p-3 text-sm text-gray-600" colSpan={3}>No events yet.</td></tr>
              )}
            </tbody>
          </table>
        </Card>
      </div>
    </LayoutShell>
  );
}
