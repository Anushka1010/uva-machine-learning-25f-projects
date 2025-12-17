import { clsx } from "@/lib/utils";

export default function Card({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={clsx("border rounded bg-white", className)}>{children}</div>;
}
