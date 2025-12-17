import Badge from "./ui/Badge";

export default function Topbar() {
  return (
    <header className="h-14 border-b bg-white flex items-center justify-end px-4 gap-2">
      <Badge>Camera: Demo</Badge>
      <Badge>Model: Stub</Badge>
      <Badge>Server: Frontend-only</Badge>
    </header>
  );
}
