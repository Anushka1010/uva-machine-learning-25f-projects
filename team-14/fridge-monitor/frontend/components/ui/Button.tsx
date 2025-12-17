import { clsx } from "@/lib/utils";

export default function Button({
  children,
  onClick,
  variant = "default",
  className,
  type,
  disabled,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "default" | "secondary" | "ghost";
  className?: string;
  type?: "button" | "submit";
  disabled?: boolean;
}) {
  const base =
    "inline-flex items-center justify-center rounded px-3 py-2 text-sm transition border";
  const styles =
    variant === "default"
      ? "bg-gray-900 text-white border-gray-900 hover:bg-black"
      : variant === "secondary"
      ? "bg-white text-gray-900 border-gray-300 hover:bg-gray-100"
      : "bg-transparent text-gray-900 border-transparent hover:bg-gray-100";

  return (
    <button
      type={type ?? "button"}
      onClick={onClick}
      className={clsx(base, styles, className)}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
