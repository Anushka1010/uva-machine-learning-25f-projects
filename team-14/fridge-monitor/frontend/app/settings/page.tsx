"use client";

import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Input from "@/components/ui/Input";
import { useEffect, useState } from "react";
import { api_getExpiryDefaults, api_setExpiryDefaults } from "@/lib/mockDb";
import Link from "next/link";

export default function SettingsPage() {
  const [defaults, setDefaults] = useState<Record<string, number>>({});

  async function load() {
    const d = await api_getExpiryDefaults();
    setDefaults(d);
  }

  useEffect(() => { load(); }, []);

  async function save() {
    await api_setExpiryDefaults(defaults);
    alert("Saved!");
  }

  return (
    <LayoutShell>
      <div className="space-y-4">
        <div>
          <h1 className="text-2xl font-semibold">Settings</h1>
          <p className="text-sm text-gray-600">Edit category expiration defaults (days).</p>
        </div>

        <Card className="p-4 space-y-3">
          <div className="font-medium text-sm">Expiration Defaults</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {Object.entries(defaults).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between border rounded p-3">
                <div className="text-sm">{k}</div>
                <div className="w-24">
                  <Input
                    value={v}
                    onChange={(e) => setDefaults({ ...defaults, [k]: Number(e.target.value) })}
                  />
                </div>
              </div>
            ))}
          </div>
          <Button onClick={save}>Save</Button>
        </Card>

      </div>
    </LayoutShell>
  );
}
