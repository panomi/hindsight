import { User } from "lucide-react";

export function SubjectChip({ label, active = false, onClick }: {
  label: string; active?: boolean; onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition " +
        (active
          ? "coral-grad text-white shadow-md"
          : "bg-white/80 text-neutral-700 border border-neutral-200/60 hover:bg-white")
      }
    >
      <User size={11} /> {label}
    </button>
  );
}
