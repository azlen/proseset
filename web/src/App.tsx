import { AdminPlanner } from "./components/AdminPlanner";
import { GameApp } from "./components/GameApp";
import "./index.css";

export function App() {
  const params = new URLSearchParams(window.location.search);
  const adminMode = params.get("admin") === "1" || window.location.pathname === "/admin";
  return adminMode ? <AdminPlanner /> : <GameApp />;
}

export default App;
