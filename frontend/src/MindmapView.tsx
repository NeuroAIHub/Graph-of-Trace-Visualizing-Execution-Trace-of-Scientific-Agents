import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  MarkerType,
  type Edge as RFEdge,
  type Node as RFNode,
  type ReactFlowInstance,
} from "reactflow";
import "reactflow/dist/style.css";
import {
  applyRepulsion,
  convertGoTToFlow,
  DEFAULT_DAGRE_OPTIONS,
  layoutWithDagre,
  layoutWithD3Force,
  type GoTGraph,
  type GoTNode,
} from "./got-to-reactflow";

/**
 * Where to load got.json from. Resolution order:
 *   1. ?src=<url> query param
 *   2. VITE_GOT_URL build-time env
 *   3. ./got.json (static file served next to the app)
 *
 * Point it at a backend endpoint (e.g. /api/conversations/<id>/got) or a static
 * file the backend writes via the configured output.path_template.
 */
function resolveGotUrl(): string {
  const fromQuery = new URLSearchParams(window.location.search).get("src");
  if (fromQuery) return fromQuery;
  const fromEnv = import.meta.env.VITE_GOT_URL as string | undefined;
  if (fromEnv) return fromEnv;
  return "./got.json";
}

// --- Inlined resizable split panel (replaces OpenHands useResizablePanels) ---
function useResizablePanels(opts: {
  defaultLeftWidth: number;
  minLeftWidth: number;
  maxLeftWidth: number;
  storageKey: string;
}) {
  const { defaultLeftWidth, minLeftWidth, maxLeftWidth, storageKey } = opts;

  const [leftWidth, setLeftWidth] = useState<number>(() => {
    const raw = window.localStorage.getItem(storageKey);
    const parsed = raw ? Number(raw) : NaN;
    return Number.isFinite(parsed) ? parsed : defaultLeftWidth;
  });
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const clampWidth = useCallback(
    (w: number) => Math.max(minLeftWidth, Math.min(maxLeftWidth, w)),
    [minLeftWidth, maxLeftWidth],
  );

  const handleMouseDown = useCallback((e: ReactMouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  useEffect(() => {
    if (!isDragging) return undefined;

    const onMove = (e: globalThis.MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setLeftWidth(clampWidth(pct));
    };
    const onUp = () => {
      setIsDragging(false);
      setLeftWidth((w) => {
        window.localStorage.setItem(storageKey, String(w));
        return w;
      });
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isDragging, clampWidth, storageKey]);

  return {
    leftWidth,
    rightWidth: 100 - leftWidth,
    isDragging,
    containerRef,
    handleMouseDown,
  };
}

function ResizeHandle({
  onMouseDown,
}: {
  onMouseDown: (e: ReactMouseEvent) => void;
}) {
  return (
    <div
      onMouseDown={onMouseDown}
      style={{
        width: 6,
        cursor: "ew-resize",
        background: "#e5e7eb",
        flexShrink: 0,
      }}
      aria-label="Resize panels"
    />
  );
}

function NodeDetails({ node }: { node: GoTNode }) {
  const hasStructuredArtifacts =
    Array.isArray(node.artifacts) && node.artifacts.length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, color: "#000" }}>
      {node.description ? (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600 }}>Description</div>
          <div style={{ fontSize: 14, whiteSpace: "pre-wrap" }}>
            {node.description}
          </div>
        </div>
      ) : null}

      {node.reason ? (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600 }}>Reason</div>
          <div style={{ fontSize: 14, whiteSpace: "pre-wrap" }}>{node.reason}</div>
        </div>
      ) : null}

      {node.context ? (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600 }}>Context</div>
          <div style={{ fontSize: 14, whiteSpace: "pre-wrap" }}>
            {node.context}
          </div>
        </div>
      ) : null}

      {Array.isArray(node.parents) && node.parents.length > 0 ? (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600 }}>Parents</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {node.parents.map((p) => (
              <div key={`${node.id}-${p.id}-${p.relation}`} style={{ fontSize: 14 }}>
                <div>
                  {p.id} <span style={{ color: "#6b7280" }}>({p.relation})</span>
                </div>
                {p.explanation ? (
                  <div
                    style={{ fontSize: 12, color: "#374151", whiteSpace: "pre-wrap" }}
                  >
                    {p.explanation}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {hasStructuredArtifacts ? (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600 }}>Artifacts</div>
          <ul style={{ paddingLeft: 16, fontSize: 14, margin: 0 }}>
            {(node.artifacts ?? []).map((a) => (
              <li
                key={`${typeof a.path === "string" ? a.path : "unknown"}-${typeof a.type === "string" ? a.type : ""}`}
                style={{ wordBreak: "break-all" }}
              >
                {a.path}
                {a.type ? <span style={{ color: "#6b7280" }}> ({a.type})</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {node.status ? (
        <div style={{ fontSize: 12, color: "#374151" }}>status: {node.status}</div>
      ) : null}
    </div>
  );
}

function computeGoTSignature(graph: GoTGraph | null): string {
  const nodes = graph?.nodes ?? [];
  if (nodes.length === 0) return "0";
  const last = nodes[nodes.length - 1];
  return `${nodes.length}:${last.id}:${typeof last.title === "string" ? last.title : ""}`;
}

type Engine = "dagre" | "force";

export default function MindmapView() {
  const gotUrl = useMemo(resolveGotUrl, []);
  const lastSignatureRef = useRef<string>("");
  const unchangedCountRef = useRef<number>(0);

  const [data, setData] = useState<GoTGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [engine, setEngine] = useState<Engine>("dagre");
  const [draggedPositions, setDraggedPositions] = useState<
    Record<string, { x: number; y: number }>
  >({});

  const {
    leftWidth: graphWidth,
    rightWidth: detailsWidth,
    isDragging: isDetailsDragging,
    containerRef: detailsContainerRef,
    handleMouseDown: handleDetailsMouseDown,
  } = useResizablePanels({
    defaultLeftWidth: 70,
    minLeftWidth: 40,
    maxLeftWidth: 85,
    storageKey: "got-details-panel-width",
  });

  const nodes = data?.nodes ?? [];

  const selectedNode = useMemo(() => {
    if (!selectedId) return null;
    return nodes.find((n) => n.id === selectedId) ?? null;
  }, [nodes, selectedId]);

  const reactFlowInstanceRef = useRef<ReactFlowInstance | null>(null);
  const lastFitSignatureRef = useRef<string>("");

  const onNodeClick = useCallback((_e: unknown, n: RFNode) => {
    setSelectedId(n.id);
  }, []);

  const onNodeDragStop = useCallback((_e: unknown, n: RFNode) => {
    setDraggedPositions((prev) => ({
      ...prev,
      [n.id]: { x: n.position.x, y: n.position.y },
    }));
  }, []);

  // --- Polling fetch with adaptive backoff (replaces react-query) ---
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const intervalFor = (unchanged: number): number | null => {
      if (unchanged >= 10) return null;
      if (unchanged >= 6) return 30_000;
      if (unchanged >= 3) return 10_000;
      return 3000;
    };

    const tick = async () => {
      try {
        const res = await fetch(gotUrl, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = (await res.json()) as GoTGraph;
        if (cancelled) return;

        setError(null);
        const sig = computeGoTSignature(json);
        if (sig === lastSignatureRef.current) {
          unchangedCountRef.current += 1;
        } else {
          lastSignatureRef.current = sig;
          unchangedCountRef.current = 0;
          setData(json);
          if (Array.isArray(json?.nodes) && json.nodes.length > 0) {
            setSelectedId((prev) =>
              prev && json.nodes.some((n) => n.id === prev)
                ? prev
                : json.nodes[json.nodes.length - 1].id,
            );
          } else {
            setSelectedId(null);
          }
          const ids = new Set((json?.nodes ?? []).map((n) => n.id));
          setDraggedPositions((prev) => {
            let changed = false;
            const nextPos: Record<string, { x: number; y: number }> = {};
            for (const [id, pos] of Object.entries(prev)) {
              if (ids.has(id)) nextPos[id] = pos;
              else changed = true;
            }
            return changed ? nextPos : prev;
          });
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          unchangedCountRef.current += 1;
        }
      }

      if (cancelled) return;
      const next = intervalFor(unchangedCountRef.current);
      if (next !== null) timer = setTimeout(tick, next);
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [gotUrl]);

  const { nodes: rfNodesRaw, edges: rfEdgesRaw } = useMemo(
    () => convertGoTToFlow(data),
    [data],
  );

  const { nodes: rfNodes, edges: rfEdges } = useMemo(() => {
    const pinnedIds = new Set(Object.keys(draggedPositions));

    if (engine === "dagre") {
      const edgeCount = rfEdgesRaw.length;
      const nodeCount = rfNodesRaw.length;
      const density = edgeCount / Math.max(1, nodeCount);
      const scale = Math.min(1.8, 1 + edgeCount / 80);
      const nodesep = Math.round(40 * scale);
      const sparseFactor = Math.min(1, 0.72 + density * 0.47);
      const ranksep = Math.round(25 + 45 * scale * sparseFactor);

      const gotNodes = data?.nodes ?? [];
      let rootId: string | undefined;
      if (gotNodes.length > 0) {
        const inDegree = new Map<string, number>();
        for (const n of gotNodes) inDegree.set(n.id, 0);
        for (const n of gotNodes) {
          const parents = Array.isArray(n.parents) ? n.parents : [];
          for (const p of parents) {
            if (p && typeof p.id === "string" && inDegree.has(n.id)) {
              inDegree.set(n.id, (inDegree.get(n.id) ?? 0) + 1);
            }
          }
        }
        const candidates = gotNodes
          .filter((n) => (inDegree.get(n.id) ?? 0) === 0)
          .map((n) => n.id);
        if (candidates.length > 0) {
          candidates.sort((a, b) => a.localeCompare(b));
          rootId = candidates[0];
        } else {
          rootId = gotNodes[0].id;
        }
      }

      const dagreResult = layoutWithDagre(rfNodesRaw, rfEdgesRaw, {
        nodesep,
        ranksep,
        edgesep: DEFAULT_DAGRE_OPTIONS.edgesep,
        ranker: DEFAULT_DAGRE_OPTIONS.ranker,
        postLayerRedistribution: "center-pack",
        rootId,
      });

      const nodesAfterRepulsion = applyRepulsion(dagreResult.nodes, {
        pinnedIds,
        yDamping: 0,
        iterations: 12,
        padding: 18,
        maxStep: 18,
        groupById: dagreResult.rankById,
      });

      const yByRank = new Map<number, number>();
      for (const n of nodesAfterRepulsion) {
        const r = dagreResult.rankById.get(n.id);
        if (typeof r !== "number") continue;
        const y = n.position.y;
        const prev = yByRank.get(r);
        if (typeof prev !== "number" || y < prev) yByRank.set(r, y);
      }

      const nodesAfterFinalSnap = nodesAfterRepulsion.map((n) => {
        const r = dagreResult.rankById.get(n.id);
        if (typeof r !== "number") return n;
        const y = yByRank.get(r);
        if (typeof y !== "number") return n;
        return { ...n, position: { ...n.position, y } };
      });

      const nodesAfterRecenterX = (() => {
        const out = nodesAfterFinalSnap.map((n) => ({
          ...n,
          position: { ...n.position },
        }));
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        for (const n of out) {
          const w = (n.width as number) ?? 260;
          minX = Math.min(minX, n.position.x);
          maxX = Math.max(maxX, n.position.x + w);
        }
        if (Number.isFinite(minX) && Number.isFinite(maxX)) {
          const centerX = (minX + maxX) / 2;
          for (const n of out) n.position.x -= centerX;
        }
        return out;
      })();

      const nodesAfterFinalSnap2 = (() => {
        const y2ByRank = new Map<number, number>();
        for (const n of nodesAfterRecenterX) {
          const r = dagreResult.rankById.get(n.id);
          if (typeof r !== "number") continue;
          const y = n.position.y;
          const prev = y2ByRank.get(r);
          if (typeof prev !== "number" || y < prev) y2ByRank.set(r, y);
        }
        return nodesAfterRecenterX.map((n) => {
          const r = dagreResult.rankById.get(n.id);
          if (typeof r !== "number") return n;
          const y = y2ByRank.get(r);
          if (typeof y !== "number") return n;
          return { ...n, position: { ...n.position, y } };
        });
      })();

      return {
        edges: dagreResult.edges,
        nodes: nodesAfterFinalSnap2.map((n) => {
          const pos = draggedPositions[n.id];
          if (!pos) return n;
          return { ...n, position: pos };
        }),
      };
    }

    const forceResult = layoutWithD3Force(rfNodesRaw, rfEdgesRaw, {
      pinnedIds,
      seed: gotUrl,
    });

    return {
      edges: forceResult.edges,
      nodes: forceResult.nodes.map((n) => {
        const pos = draggedPositions[n.id];
        if (!pos) return n;
        return { ...n, position: pos };
      }),
    };
  }, [engine, data?.nodes, draggedPositions, rfEdgesRaw, rfNodesRaw, gotUrl]);

  useEffect(() => {
    const signature = `${rfNodes.length}:${rfEdges.length}`;
    if (lastFitSignatureRef.current === "") {
      lastFitSignatureRef.current = signature;
      reactFlowInstanceRef.current?.fitView({ padding: 0.2 });
    }
  }, [rfEdges.length, rfNodes.length]);

  return (
    <div
      ref={detailsContainerRef}
      style={{
        display: "flex",
        height: "100%",
        width: "100%",
        overflow: "hidden",
        background: "#fff",
      }}
    >
      <div
        style={{
          minWidth: 0,
          height: "100%",
          position: "relative",
          width: `${graphWidth}%`,
          transitionProperty: isDetailsDragging ? "none" : "all",
        }}
      >
        <ReactFlow
          nodes={rfNodes.map((n) => ({
            ...n,
            style: {
              border: "1px solid #111827",
              borderRadius: 6,
              background: "#fff",
              padding: 10,
              fontSize: 12,
              color: "#111827",
              boxShadow: "0 1px 0 rgba(0,0,0,0.05)",
              ...n.style,
            },
          }))}
          edges={rfEdges.map((e) => ({
            ...(e as RFEdge),
            type: "default",
            sourcePosition: "bottom",
            targetPosition: "top",
            markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
            style: {
              ...e.style,
              stroke: e.style?.stroke ?? "#111827",
              strokeWidth: e.style?.strokeWidth ?? 2.4,
            },
          }))}
          onNodeClick={onNodeClick}
          onNodeDragStop={onNodeDragStop}
          onInit={(instance) => {
            reactFlowInstanceRef.current = instance;
          }}
          fitView
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>

        <div
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            zIndex: 10,
            display: "flex",
            gap: 6,
          }}
        >
          <button
            type="button"
            onClick={() => setEngine((p) => (p === "dagre" ? "force" : "dagre"))}
            style={{
              border: "1px solid #d1d5db",
              borderRadius: 4,
              background: "#fff",
              padding: "4px 8px",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            Layout: {engine}
          </button>
        </div>

        <div
          style={{
            position: "absolute",
            bottom: 8,
            left: 8,
            zIndex: 10,
            borderRadius: 4,
            border: "1px solid #e5e7eb",
            background: "rgba(255,255,255,0.95)",
            padding: "6px 10px",
            fontSize: 12,
            color: "#000",
          }}
          aria-label="Node and edge count"
        >
          Nodes: {rfNodes.length} | Edges: {rfEdges.length}
        </div>
      </div>

      <ResizeHandle onMouseDown={handleDetailsMouseDown} />

      <div
        style={{
          flexShrink: 0,
          borderLeft: "1px solid #e5e7eb",
          background: "#fff",
          overflowY: "auto",
          width: `${detailsWidth}%`,
          transitionProperty: isDetailsDragging ? "none" : "all",
        }}
      >
        <div style={{ padding: 12, fontSize: 12, color: "#000" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: 8,
            }}
          >
            <span style={{ fontWeight: 500 }}>Details</span>
          </div>
          {error ? (
            <div style={{ color: "#b91c1c", marginBottom: 8 }}>
              Failed to load GoT: {error}
            </div>
          ) : null}
          {!selectedNode ? (
            <div>{nodes.length === 0 ? "No nodes." : "Select a node."}</div>
          ) : (
            <NodeDetails node={selectedNode} />
          )}
        </div>
      </div>
    </div>
  );
}
