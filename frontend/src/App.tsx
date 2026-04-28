import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Activity, FolderOpen, Search, Upload } from "lucide-react";
import Dashboard from "./views/Dashboard";
import Collections from "./views/Collections";
import Ingest from "./views/Ingest";
import Investigate from "./views/Investigate";

function NavLink({ to, icon: Icon, label }: { to: string; icon: any; label: string }) {
  const { pathname } = useLocation();
  const active = pathname === to || pathname.startsWith(to + "/");
  return (
    <Link
      to={to}
      className={
        "flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition " +
        (active
          ? "coral-grad text-white shadow-md"
          : "text-neutral-700 hover:bg-white/80")
      }
    >
      <Icon size={16} />
      {label}
    </Link>
  );
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="glass mx-4 mt-4 px-4 py-3 flex items-center gap-6">
        <div className="font-semibold tracking-tight text-lg flex items-center gap-2">
          <span className="inline-block w-2.5 h-2.5 rounded-full coral-grad" />
          Investigation Platform
        </div>
        <nav className="flex items-center gap-1">
          <NavLink to="/" icon={Activity} label="Dashboard" />
          <NavLink to="/collections" icon={FolderOpen} label="Collections" />
          <NavLink to="/ingest" icon={Upload} label="Ingest" />
        </nav>
        <div className="ml-auto text-xs text-neutral-500 flex items-center gap-1">
          <Search size={12} /> agentic video search
        </div>
      </header>

      <main className="flex-1 p-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/collections" element={<Collections />} />
          <Route path="/ingest" element={<Ingest />} />
          <Route path="/investigate/:id" element={<Investigate />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
