import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import TopNav from "./components/TopNav.jsx";
import Dashboard from "./views/Dashboard.jsx";
import Explorer from "./views/Explorer.jsx";
import Sandbox from "./views/Sandbox.jsx";
import EmbeddingMap from "./views/EmbeddingMap.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <TopNav />
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/explorer/:clusterId" element={<Explorer />} />
          <Route path="/sandbox" element={<Sandbox />} />
          <Route path="/embedding" element={<EmbeddingMap />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
