import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/explorer", label: "Explorer" },
  { to: "/sandbox", label: "Sandbox" },
  { to: "/embedding", label: "Embedding Map" },
];

export default function TopNav() {
  return (
    <header className="topnav">
      <div className="brand">
        <span className="logo">◈</span>
        <div>
          <h1>NEMESIS</h1>
          <p>Fraud Network Intelligence</p>
        </div>
      </div>
      <nav className="nav-links">
        {LINKS.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
