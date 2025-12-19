import { clsx } from "@/lib/utils";

export default function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={clsx("border rounded px-3 py-2 text-sm bg-white", props.className)}
    />
  );
}
