export function id() {
  return Math.random().toString(16).slice(2) + Math.random().toString(16).slice(2);
}

export function nowISO() {
  return new Date().toISOString();
}

export function daysFromNowISO(days: number) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString();
}

export function formatDate(s?: string) {
  if (!s) return "—";
  const d = new Date(s);
  return d.toLocaleString();
}

export function daysLeft(expires?: string) {
  if (!expires) return null;
  const ms = new Date(expires).getTime() - Date.now();
  return Math.ceil(ms / 86400000);
}

export function macroDistance(a: any, b: any) {
  const keys = ["calories", "protein", "carbs", "fat"];
  return keys.reduce((sum, k) => sum + Math.abs((a?.[k] ?? 0) - (b?.[k] ?? 0)), 0);
}

export function clsx(...xs: Array<string | false | undefined | null>) {
  return xs.filter(Boolean).join(" ");
}
