import { clsx } from "@/lib/utils";

export default function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={clsx(
        "border rounded px-3 py-2 text-sm bg-white w-full",
        props.className
      )}
    />
  );
}
