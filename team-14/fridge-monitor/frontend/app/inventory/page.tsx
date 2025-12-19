"use client";

import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Input from "@/components/ui/Input";
import Select from "@/components/ui/Select";
import Badge from "@/components/ui/Badge";
import { useEffect, useMemo, useState } from "react";
import { formatDate } from "@/lib/utils";
import type { Item } from "@/lib/types";
import Link from "next/link";
import { getCategoryImage } from "@/lib/categoryImages";

const FLASK_API_URL = "http://127.0.0.1:5000";

export default function InventoryPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<"in_fridge" | "removed">("in_fridge");

  // get img source
  function getDriveImageSrc(fileId: string) {
    return `https://drive.google.com/thumbnail?id=${fileId}&sz=w1000`;
  }

  // fetch inventory items from backend 
  async function load() {
    console.log(`Fetching items with status: ${status}`);
    try {
      const response = await fetch(`${FLASK_API_URL}/api/inventory`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setItems(data as Item[]);
    } catch (error) {
      console.error("Failed to load inventory from server:", error);
      setItems([]);
    }
  }

  // send patch request to edit item
  async function patchItem(itemId: string, updateData: Partial<Item>) {
    try {
      const response = await fetch(`${FLASK_API_URL}/api/inventory/${itemId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(`Update failed: ${response.status} - ${errorBody.error || errorBody.message}`);
      }

      console.log(`Item ${itemId} updated successfully.`);
      return true;
    } catch (error) {
      console.error("Error updating item:", error);
      return false;
    }
  }


  useEffect(() => { load(); }, [status]);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    return items.filter(i =>
      !s || i.name.toLowerCase().includes(s) || i.category?.toLowerCase().includes(s)
    );
  }, [items, q]);

  async function editExpiry(item: Item) {
    const val = window.prompt(`Enter expiration date (YYYY-MM-DD) for ${item.name}:`, item.expiration_date?.substring(0, 10) ?? "");
    if (!val) return;

    // date validation
    const d = new Date(val);
    if (isNaN(d.getTime())) {
      console.error("Invalid date entered.");
      return;
    }

    const isoString = d.toISOString();

    // call patch function
    const success = await patchItem(item._id, { expiration_date: isoString });
    if (success) {
      load();
    }
  }

  async function markRemoved(item: Item) {
    // call patch function
    const success = await patchItem(item._id, { status: "removed" });
    if (success) {
      load();
    }
  }

  return (
    <LayoutShell>
      <div className="space-y-4">
        <div className="flex items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Inventory</h1>
            <p className="text-sm text-gray-600">Search, edit expiration, and manage items.</p>
          </div>
          <Button variant="secondary" onClick={load}>Refresh</Button>
        </div>

        <div className="flex flex-wrap gap-2 items-center">
          <div className="w-64"><Input placeholder="Search items…" value={q} onChange={(e) => setQ(e.target.value)} /></div>
          <Select value={status} onChange={(e) => setStatus(e.target.value as "in_fridge" | "removed")}>
            <option value="in_fridge">In fridge</option>
            <option value="removed">Removed</option>
          </Select>
          <Badge>{filtered.length} shown</Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filtered.map((i) => (
            <Card key={i._id} className="p-3">
              <Link href={`/inventory/${i._id}`} className="block">
                <div className="flex gap-3">
                  <div className="w-16 h-16 rounded bg-gray-100 overflow-hidden border flex items-center justify-center relative">
                    {i.image && i.image.file_id ? (
                      <img
                        src={getDriveImageSrc(i.image.file_id)}
                        alt={i.name}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    ) : (
                      getCategoryImage(i.name) ? (
                        <img src={getCategoryImage(i.name)!} alt={i.name} className="w-full h-full object-cover" />
                      ) : (
                        <span className="text-[10px] text-gray-500 text-center px-1">
                          No image
                        </span>
                      )
                    )}
                  </div>

                  <div className="flex-1">
                    <div className="font-medium">{i.name}</div>
                    <div className="text-xs text-gray-600 mt-1">
                      Added: {formatDate(i.date_placed)}
                    </div>
                    <div className="text-sm mt-1 font-medium text-orange-700">
                      Exp: {formatDate(i.expiration_date)}
                    </div>
                  </div>
                </div>
              </Link>

              <div className="mt-3 flex gap-2">
                <Button variant="secondary" className="text-xs px-2 py-1" onClick={(e) => { e.preventDefault(); editExpiry(i); }}>
                  Edit expiry
                </Button>
                {status === "in_fridge" && (
                  <Button variant="secondary" className="text-xs px-2 py-1" onClick={(e) => { e.preventDefault(); markRemoved(i); }}>
                    Mark removed
                  </Button>
                )}
              </div>
            </Card>
          ))}

          {filtered.length === 0 && (
            <Card className="p-4 text-sm text-gray-600">
              No items match. Try running <b>Scan In</b>.
            </Card>
          )}
        </div>
      </div>
    </LayoutShell>
  );
}
