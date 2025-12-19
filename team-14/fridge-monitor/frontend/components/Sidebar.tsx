"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "@/lib/utils";

const nav = [
  { href: "/", label: "Dashboard" },
  { href: "/inventory", label: "Inventory" },
  { href: "/scan", label: "Scan & Events" },
  { href: "/recipes", label: "Recipes" },
  { href: "/settings", label: "Settings" },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="w-64 border-r bg-white">
      <div className="p-4">
        <div className="text-lg font-semibold">Fridge Monitor</div>
        <div className="text-xs text-gray-600">Welcome!</div>
      </div>
      <nav className="px-2 space-y-1">
        {nav.map((n) => {
          const active = path === n.href;
          return (
            <Link
              key={n.href}
              href={n.href}
              className={clsx(
                "block rounded px-3 py-2 text-sm",
                active ? "bg-gray-900 text-white" : "hover:bg-gray-100"
              )}
            >
              {n.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
