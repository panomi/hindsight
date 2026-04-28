import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "ghost" | "danger";

export function Button({
  children, variant = "primary", className = "", ...rest
}: { children: ReactNode; variant?: Variant; className?: string } & ButtonHTMLAttributes<HTMLButtonElement>) {
  const base = "inline-flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed";
  const styles =
    variant === "primary" ? "coral-grad text-white shadow-md hover:brightness-105" :
    variant === "ghost"   ? "bg-white/70 text-neutral-800 border border-neutral-200/60 hover:bg-white" :
                            "bg-red-50 text-red-700 border border-red-200 hover:bg-red-100";
  return <button className={`${base} ${styles} ${className}`} {...rest}>{children}</button>;
}
