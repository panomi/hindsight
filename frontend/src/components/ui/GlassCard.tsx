import type { HTMLAttributes, ReactNode } from "react";

export function GlassCard({
  children,
  className = "",
  ...rest
}: { children: ReactNode; className?: string } & HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={`glass p-5 ${className}`} {...rest}>
      {children}
    </div>
  );
}
