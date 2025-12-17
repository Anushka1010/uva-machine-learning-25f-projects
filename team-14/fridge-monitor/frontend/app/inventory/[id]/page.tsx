"use client";

import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Badge from "@/components/ui/Badge";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  api_getItem,
  api_patchItem,
  api_relatedRecipesForItem,
  api_getItemMacros,
  api_setItemMacros,
  api_resetItemMacros,
} from "@/lib/mockDb";
import { formatDate } from "@/lib/utils";
import { getCategoryImage } from "@/lib/categoryImages";
import { daysLeft } from "@/lib/utils";

export default function ItemDetailPage() {
  const params = useParams<{ id: string }>();
  const itemId = params?.id;

  const [item, setItem] = useState<any>(null);
  const [macros, setMacros] = useState<any>(null);
  const [related, setRelated] = useState<any[]>([]);
  const [err, setErr] = useState("");

  async function load(id: string) {
    try {
      const it = await api_getItem(id);
      if (!it) {
        setItem(null);
        setMacros(null);
        setRelated([]);
        setErr("Item not found.");
        return;
      }
      setItem(it);

      const m = await api_getItemMacros(it.label);
      setMacros(m);

      const rel = await api_relatedRecipesForItem(it.label);
      setRelated(rel);

      setErr("");
    } catch (e: any) {
      setErr(e.message || "Failed to load item");
    }
  }

  useEffect(() => {
    if (!itemId) return;
    load(itemId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [itemId]);

  async function editExpiry() {
    if (!item) return;
    const val = prompt(
      "Enter expiration date (YYYY-MM-DD) or ISO string:",
      item?.expiresAt ?? ""
    );
    if (!val) return;
    const d = new Date(val);
    if (isNaN(d.getTime())) return alert("Invalid date");
    await api_patchItem(item.id, { expiresAt: d.toISOString() });
    await load(item.id);
  }

  async function markRemoved() {
    if (!item) return;
    await api_patchItem(item.id, { status: "removed" });
    await load(item.id);
  }

  return (
    <LayoutShell>
      <div className="space-y-4">
        <div className="flex items-end justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Item Details</h1>
            <p className="text-sm text-gray-600">Details, macros, and related recipes.</p>
          </div>
        </div>

        {err && <Card className="p-4 text-sm text-red-600">{err}</Card>}

        {item && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Item info */}
            <Card className="p-4 lg:col-span-1 space-y-3">
              <div className="flex gap-3 items-start">
                <div className="w-24 h-24 rounded bg-gray-100 overflow-hidden border">
                  {getCategoryImage(item.label) ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={getCategoryImage(item.label)!}
                      alt={item.label}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-gray-500">
                      No class image
                    </div>
                  )}
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-lg">{item.label}</div>
                  <div className="text-sm text-gray-600">{item.category}</div>
                  <div className="mt-2 flex gap-2 flex-wrap">
                    <Badge>Status: {item.status}</Badge>
                    <Badge>Conf: {Math.round(item.confidence * 100)}%</Badge>
                  </div>
                </div>
              </div>

              <div className="text-sm space-y-1">
                <div>
                  <span className="text-gray-600">Added:</span> {formatDate(item.createdAt)}
                </div>
                <div>
                  <span className="text-gray-600">Expires:</span> {formatDate(item.expiresAt)}
                </div>
              </div>

              <div className="flex gap-2">
                <Button variant="secondary" onClick={editExpiry}>Edit expiry</Button>
                {item.status === "in_fridge" && (
                  <Button variant="secondary" onClick={markRemoved}>Mark removed</Button>
                )}
              </div>
            </Card>

            {/* Editable macros */}
            <Card className="p-4 lg:col-span-1 space-y-3">
              <div className="font-medium">Macros (Editable)</div>

              {!macros ? (
                <div className="text-sm text-gray-600">No macro data for this item label yet.</div>
              ) : (
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="text-xs text-gray-600">Serving</div>
                    <input
                      className="border rounded px-3 py-2 text-sm w-full"
                      value={macros.serving}
                      onChange={(e) => setMacros({ ...macros, serving: e.target.value })}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    {(["calories", "protein", "carbs", "fat"] as const).map((k) => (
                      <div key={k} className="space-y-1">
                        <div className="text-xs text-gray-600">{k}</div>
                        <input
                          className="border rounded px-3 py-2 text-sm w-full"
                          value={macros[k]}
                          onChange={(e) => setMacros({ ...macros, [k]: Number(e.target.value) })}
                        />
                      </div>
                    ))}
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="secondary"
                      onClick={async () => {
                        await api_setItemMacros(item.label, macros);
                        alert("Saved macros!");
                      }}
                    >
                      Save macros
                    </Button>

                    <Button
                      variant="secondary"
                      onClick={async () => {
                        await api_resetItemMacros(item.label);
                        const fresh = await api_getItemMacros(item.label);
                        setMacros(fresh);
                      }}
                    >
                      Reset
                    </Button>
                  </div>

                  <div className="text-xs text-gray-600">
                    Saved per label (e.g., all “eggs” share these macros). Stored in localStorage.
                  </div>
                </div>
              )}
            </Card>

            {/* Related recipes */}
            <Card className="p-4 lg:col-span-1 space-y-3">
              <div className="font-medium">Related Recipes</div>

              {related.length === 0 ? (
                <div className="text-sm text-gray-600">
                  No recipes found containing <b>{item.label}</b>.
                </div>
              ) : (
                <div className="space-y-2">
                  {related.slice(0, 6).map((r: any) => (
                    <div key={r.recipe.id} className="border rounded p-3">
                      <div className="font-medium">{r.recipe.name}</div>
                      <div className="text-xs text-gray-600">
                        Coverage: {(r.coverage * 100).toFixed(0)}% • Have {r.have.length}/{r.recipe.ingredients.length}
                      </div>

                      <div className="mt-2 flex flex-wrap gap-1">
                        {r.have.map((x: string) => <Badge key={x}>{x}</Badge>)}
                      </div>

                      {r.missing.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {r.missing.map((x: string) => (
                            <Badge key={x} className="bg-white">{x}</Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>
        )}
      </div>
    </LayoutShell>
  );
}
