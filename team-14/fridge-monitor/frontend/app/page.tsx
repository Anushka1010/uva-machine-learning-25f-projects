"use client";

import ExpirationBar from "@/components/ExpirationBar";
import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import { useEffect, useMemo, useState } from "react";
import { api_listItems } from "@/lib/mockDb";
import { daysLeft, formatDate } from "@/lib/utils";
import type { Item } from "@/lib/types";
import { getCategoryImage } from "@/lib/categoryImages";

const FLASK_API_URL = "http://127.0.0.1:5000";

export default function Page() {
  const [items, setItems] = useState<any[]>([]);

  async function load() {
    try {
      const invRes = await fetch(`${FLASK_API_URL}/api/inventory`);
      const invData = await invRes.json();
      setItems(invData);
    } catch (error) {
      console.error("Failed to load data:", error);
      setItems([]);
    }
  }

  useEffect(() => { load(); }, []);

  const alerts = useMemo(() => {
    const now = new Date();
    const out: Array<{ type: string; message: string; itemId: string }> = [];

    for (const it of items) {
      if (it.status !== "in_fridge" || !it.expiration_date) continue;

      const created = new Date(it.date_placed);
      const expires = new Date(it.expiration_date);

      const total = expires.getTime() - created.getTime();
      const used = now.getTime() - created.getTime();
      const pct = total > 0 ? used / total : 0;
      const daysLeft = (expires.getTime() - now.getTime()) / 86400000;

      if (pct >= 0.66 && pct <= 0.75) {
        out.push({
          type: "shelf_life",
          message: `${it.name} is ~${Math.round(pct * 100)}% through shelf life`,
          itemId: it._id,
        });
      }
      if (daysLeft >= 0 && daysLeft <= 1.1) {
        out.push({
          type: "one_day",
          message: `${it.name} expires in ~1 day`,
          itemId: it._id,
        });
      }
      if (daysLeft < 0) {
        out.push({
          type: "expired",
          message: `${it.name} has expired`,
          itemId: it._id,
        });
      }
    }

    return out;
  }, [items]);

  const expSoon = useMemo(() => {
    return [...items]
      .filter((i) => i.expiration_date)
      .sort((a, b) => (a.expiration_date! < b.expiration_date! ? -1 : 1))
      .slice(0, 8);
  }, [items]);

  return (
    <LayoutShell>
      <div className="space-y-6">
        <div className="flex items-end justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Dashboard</h1>
            <p className="text-sm text-gray-600">Expirations, alerts, and recent activity.</p>
          </div>
          <Button variant="secondary" onClick={load}>Refresh</Button>
        </div>

        <section className="space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Expiring Soon</h2>
            <Badge>{items.length} item(s) in fridge</Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
  {expSoon.length > 0 ? (
    expSoon.map((i) => (
      <Card key={i._id} className="p-3">
        <div className="flex gap-3">
          {/* Item Image */}
          <div className="w-14 h-14 rounded bg-gray-100 overflow-hidden border flex items-center justify-center">
            {getCategoryImage(i.name) ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={getCategoryImage(i.name)!}
                alt={i.name}
                className="w-full h-full object-cover"
              />
            ) : (
              <span className="text-[8px] text-gray-500 text-center px-1">
                No image
              </span>
            )}
          </div>

          {/* Item Info */}
          <div className="flex-1 flex flex-col justify-between">
            <div>
              <div className="font-medium">{i.name}</div>
              <div className="text-xs text-gray-600">{i.category}</div>

              {/* Expiration Text */}
              {i.expiration_date && (
                <div className="text-sm mt-1">
                  {(() => {
                    const days = daysLeft(i.expiration_date);
                    if (days === null) return "No expiry set";
                    return days >= 0
                      ? `Expires in ${days} day(s)`
                      : `Expired ${Math.abs(days)} day(s) ago`;
                  })()}
                </div>
              )}
              
              {/* Expiration Progress Bar */}
              {i.expiration_date && (
                <div className="mt-2" title={`Expires on ${formatDate(i.expiration_date)}`}>
                  <ExpirationBar expiresAt={i.expiration_date} />
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    ))
  ) : (
    <Card className="p-4 text-sm text-gray-600">
      No items yet. Go to <b>Scan & Events</b> and run Scan In.
    </Card>
  )}
</div>

        </section>

        <section className="space-y-2">
          <h2 className="font-semibold">Alerts</h2>
          <Card className="divide-y">
            {alerts.map((a: any, idx: number) => (
              <div key={idx} className="p-3 text-sm">{a.message}</div>
            ))}
            {alerts.length === 0 && (
              <div className="p-3 text-sm text-gray-600">No alerts right now.</div>
            )}
          </Card>
        </section>
      </div>
    </LayoutShell>
  );
}
