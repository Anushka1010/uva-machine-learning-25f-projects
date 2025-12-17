"use client";

type Props = {
  expiresAt: string;     // expiration date
  addedAt?: string;      // optional: date item was added
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export default function ExpirationBar({ expiresAt, addedAt }: Props) {
  if (!expiresAt) return null; // safety check

  const now = new Date();
  const exp = new Date(expiresAt);
  const start = addedAt ? new Date(addedAt) : new Date(exp.getTime() - 7 * 24 * 60 * 60 * 1000);

  const totalMs = exp.getTime() - start.getTime();
  const leftMs = exp.getTime() - now.getTime();

  // Calculate percentage remaining
  let percent = clamp((leftMs / totalMs) * 100, 0, 100);

  // Determine bar color
  let color = "bg-green-500"; // default: plenty of time
  if (percent < 40) color = "bg-yellow-500"; // expiring soon
  if (percent < 15) color = "bg-red-500";    // almost expired

  // If item is expired, force the bar to be full red
  if (leftMs <= 0) {
    percent = 100;
    color = "bg-red-500";
  }

  return (
    <div className="w-full h-2 bg-gray-200 rounded overflow-hidden">
      <div
        className={`h-full transition-all ${color}`}
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}
