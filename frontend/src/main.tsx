import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import MindmapView from "./MindmapView";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MindmapView />
  </StrictMode>,
);
