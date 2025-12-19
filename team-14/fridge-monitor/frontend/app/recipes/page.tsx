"use client";

import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Badge from "@/components/ui/Badge";
import Input from "@/components/ui/Input";
import { useEffect, useState } from "react";
import { api_recommendations } from "@/lib/mockDb";

export default function RecipesPage() {
  const [target, setTarget] = useState({ calories: 600, protein: 35, carbs: 60, fat: 15 });
  const [minCoverage, setMinCoverage] = useState(0.3);
  const [recs, setRecs] = useState<any[]>([]);

  async function load() {
    const r = await api_recommendations({
      target,
      minCoverage,
      prioritizeExpiring: true,
    });
    setRecs(r);
  }

  useEffect(() => { load(); }, []);

  return (
    <LayoutShell>
      <div className="space-y-4">
        <div>
          <h1 className="text-2xl font-semibold">Recipes</h1>
          <p className="text-sm text-gray-600">Recommendations based on expiring items + macros.</p>
        </div>

        <Card className="p-4 space-y-3">
          <div className="font-medium text-sm">Macro Targets</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {(["calories","protein","carbs","fat"] as const).map((k) => (
              <div key={k} className="space-y-1">
                <div className="text-xs text-gray-600">{k}</div>
                <Input
                  value={(target as any)[k]}
                  onChange={(e) => setTarget({ ...target, [k]: Number(e.target.value) })}
                />
              </div>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <div className="text-xs text-gray-600">Min coverage</div>
            <div className="w-24">
              <Input value={minCoverage} onChange={(e) => setMinCoverage(Number(e.target.value))} />
            </div>
            <Button variant="secondary" onClick={load}>Recompute</Button>
          </div>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {recs.map((r) => (
            <Card key={r.recipe.id} className="p-4">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="font-medium">{r.recipe.name}</div>
                  <div className="text-xs text-gray-600">
                    Score: {r.score} • Coverage: {(r.coverage * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              <div className="mt-3">
                <div className="text-xs text-gray-600">Have</div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {r.have.map((x: string) => <Badge key={x}>{x}</Badge>)}
                  {r.have.length === 0 && <span className="text-sm text-gray-600">None</span>}
                </div>
              </div>

              <div className="mt-3">
                <div className="text-xs text-gray-600">Missing</div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {r.missing.map((x: string) => <Badge key={x} className="bg-white">{x}</Badge>)}
                  {r.missing.length === 0 && <span className="text-sm text-gray-600">None</span>}
                </div>
              </div>

              <div className="mt-3 text-xs text-gray-600">
                {r.reasons.join(" • ")}
              </div>
            </Card>
          ))}
        </div>

        {recs.length === 0 && (
          <Card className="p-4 text-sm text-gray-600">
            No recommendations yet. Add items via <b>Scan In</b>.
          </Card>
        )}
      </div>
    </LayoutShell>
  );
}
