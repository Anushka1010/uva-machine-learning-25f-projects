"use client";

import ExpirationBar from "@/components/ExpirationBar";
import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import { useEffect, useMemo, useState } from "react";
import { api_alerts, api_listEvents, api_listItems } from "@/lib/mockDb";
import { daysLeft, formatDate } from "@/lib/utils";
import type { Item } from "@/lib/types";
import { getCategoryImage } from "@/lib/categoryImages";

export default function Page() {
  const [items, setItems] = useState<Item[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);

  async function load() {
    const [it, ev, al] = await Promise.all([
      api_listItems("in_fridge"),
      api_listEvents(),
      api_alerts(),
    ]);
    setItems(it);
    setEvents(ev.slice(0, 10));
    setAlerts(al);
  }

  useEffect(() => { load(); }, []);

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

        <section className="space-y-2">
          <h2 className="font-semibold">Recent Activity</h2>
          <Card className="overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left p-3">Time</th>
                  <th className="text-left p-3">Type</th>
                  <th className="text-left p-3">Summary</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e: any) => (
                  <tr key={e.id} className="border-t">
                    <td className="p-3 text-gray-700">{formatDate(e.timestamp)}</td>
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
        </section>
      </div>
    </LayoutShell>
  );
}
