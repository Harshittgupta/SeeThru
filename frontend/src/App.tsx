import { Link, Route, Routes } from "react-router-dom";
import { Upload } from "./routes/Upload";
import { Analyze } from "./routes/Analyze";
import { About } from "./routes/About";

export function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-bold">
            <span className="text-xl">👁️</span> SEETHRU
          </Link>
          <nav className="flex gap-4 text-sm text-slate-600">
            <Link to="/" className="hover:text-slate-900">
              Analyse
            </Link>
            <Link to="/about" className="hover:text-slate-900">
              About
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-8">
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/analyze/:jobId" element={<Analyze />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <footer className="mx-auto max-w-5xl px-4 py-6 text-center text-xs text-slate-400">
        Research tool — not forensic evidence.
      </footer>
    </div>
  );
}
