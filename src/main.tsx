import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import ResourceGuardHost from "./features/resourceGuard/ResourceGuardHost";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
    <ResourceGuardHost />
  </StrictMode>,
);
