import { clsx } from "@/lib/utils";

export default function Badge({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={clsx("text-xs border rounded px-2 py-1 bg-gray-50", className)}>
      {children}
    </span>
  );
}
