import type { ReactNode } from "react";

const VARIANTS: Record<string, string> = {
  neutral: "bg-neutral-100 text-neutral-700 border border-neutral-200",
  coral: "coral-grad text-white",
  green: "bg-emerald-50 text-emerald-700 border border-emerald-200",
  amber: "bg-amber-50 text-amber-700 border border-amber-200",
  red: "bg-red-50 text-red-700 border border-red-200",
};

export function Badge({
  children, variant = "neutral",
}: { children: ReactNode; variant?: keyof typeof VARIANTS }) {
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${VARIANTS[variant]}`}>
      {children}
    </span>
  );
}
