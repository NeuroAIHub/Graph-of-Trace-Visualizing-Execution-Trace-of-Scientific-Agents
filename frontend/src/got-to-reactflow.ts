import dagre from "dagre";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationNodeDatum,
} from "d3-force";
import { MarkerType, type Edge, type Node } from "reactflow";

export type GoTParent = {
  id: string;
  relation: string;
  explanation?: string;
};

export type GoTArtifact = {
  path: string;
  type?: string;
  [key: string]: unknown;
};

export type GoTNode = {
  id: string;
  title?: string;
  type?: string;
  parents?: GoTParent[];
  description?: string;
  reason?: string;
  context?: string;
  artifacts?: GoTArtifact[];
  artifacts_used?: string[];
  artifacts_produced?: string[];
  status?: string;
  [key: string]: unknown;
};

export type GoTGraph = {
  meta?: Record<string, unknown>;
  nodes: GoTNode[];
};

function getNodeTitle(n: GoTNode): string {
  if (typeof n.title === "string" && n.title.trim()) return n.title.trim();
  return n.id;
}

const NODE_WIDTH = 260;
const NODE_HEIGHT = 90;

/** Semantic edge styles for main flow, experiment branches, and backtrack/fix. */
const EDGE_STYLE = {
  main: { strokeWidth: 2.4, stroke: "#111827" },
  experiment: { strokeWidth: 2, stroke: "#94a3b8" },
  fix: { strokeWidth: 2, stroke: "#f97316", strokeDasharray: "6 4" },
} as const;

type EdgeSemantic = keyof typeof EDGE_STYLE;

const MAIN_RELATIONS = new Set(["necessitated_by", "refinement_of"]);
const EXPERIMENT_RELATIONS = new Set([
  "ablation_of",
  "validation_of",
  "comparison_with",
]);

function getEdgeSemantic(
  relation: string,
  sourceId: string,
  targetId: string,
  nodeIndexById: Map<string, number>,
): EdgeSemantic {
  if (MAIN_RELATIONS.has(relation)) return "main";
  if (EXPERIMENT_RELATIONS.has(relation)) {
    const srcIdx = nodeIndexById.get(sourceId);
    const tgtIdx = nodeIndexById.get(targetId);
    if (
      typeof srcIdx === "number" &&
      typeof tgtIdx === "number" &&
      tgtIdx < srcIdx
    ) {
      return "fix";
    }
    return "experiment";
  }
  return "main";
}

function getDagreEdgeAttrs(
  relation: string,
  sourceId: string,
  targetId: string,
  nodes: Node[],
): { weight: number; minlen: number } {
  const nodeIndexById = new Map<string, number>();
  nodes.forEach((n, idx) => nodeIndexById.set(n.id, idx));

  const semantic = getEdgeSemantic(relation, sourceId, targetId, nodeIndexById);

  if (semantic === "main") return { weight: 6, minlen: 1 };
  if (semantic === "experiment") return { weight: 2, minlen: 1 };
  return { weight: 1, minlen: 3 };
}

export function convertGoTToFlow(graph: GoTGraph | null): {
  nodes: Node[];
  edges: Edge[];
} {
  const gotNodes = graph?.nodes ?? [];

  const nodeIds = new Set(gotNodes.map((n) => n.id));
  const nodeIndexById = new Map<string, number>();
  gotNodes.forEach((n, idx) => nodeIndexById.set(n.id, idx));

  const placeholderIds = new Set<string>();
  const edgeEntries: Array<{
    key: string;
    source: string;
    target: string;
    relation: string;
    explanation?: string;
  }> = [];
  const edgeKeys = new Set<string>();

  for (const child of gotNodes) {
    const parents = Array.isArray(child.parents) ? child.parents : [];
    for (const p of parents) {
      if (!p || typeof p.id !== "string" || typeof p.relation !== "string") {
        // eslint-disable-next-line no-continue
        continue;
      }

      if (!nodeIds.has(p.id)) placeholderIds.add(p.id);

      const key = `${p.id}->${child.id}:${p.relation}`;
      if (edgeKeys.has(key)) {
        // eslint-disable-next-line no-continue
        continue;
      }
      edgeKeys.add(key);

      edgeEntries.push({
        key,
        source: p.id,
        target: child.id,
        relation: p.relation,
        explanation: p.explanation,
      });
    }
  }

  Array.from(placeholderIds).forEach((id, i) =>
    nodeIndexById.set(id, gotNodes.length + i),
  );

  const edges: Edge[] = edgeEntries.map((e) => {
    const semantic = getEdgeSemantic(
      e.relation,
      e.source,
      e.target,
      nodeIndexById,
    );
    const style = { ...EDGE_STYLE[semantic] };

    return {
      id: e.key,
      source: e.source,
      target: e.target,
      data: { relation: e.relation, explanation: e.explanation },
      type: "bezier",
      sourcePosition: "bottom",
      targetPosition: "top",
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 16,
        height: 16,
      },
      style,
    };
  });

  const nodes: Node[] = [];

  for (const id of placeholderIds) {
    nodes.push({
      id,
      data: { label: id, isPlaceholder: true },
      position: { x: 0, y: 0 },
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      draggable: false,
      selectable: true,
    });
  }

  for (const n of gotNodes) {
    nodes.push({
      id: n.id,
      data: { label: getNodeTitle(n), gotNode: n },
      position: { x: 0, y: 0 },
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
    });
  }

  return { nodes, edges };
}

export type DagreLayoutOptions = {
  nodesep?: number;
  ranksep?: number;
  edgesep?: number;
  align?: "UL" | "UR" | "DL" | "DR";
  ranker?: "network-simplex" | "tight-tree" | "longest-path";
  /**
   * Post-processing after Dagre assigns nodes to ranks.
   *
   * - off: keep Dagre x positions
   * - center-pack: re-distribute nodes within each rank evenly and center them
   */
  postLayerRedistribution?: "off" | "center-pack";
  rootId?: string;
};

export const DEFAULT_DAGRE_OPTIONS: Required<DagreLayoutOptions> = {
  nodesep: 24,
  ranksep: 42,
  edgesep: 16,
  align: "UL",
  ranker: "network-simplex",
  postLayerRedistribution: "off",
  rootId: "",
};

export function layoutWithDagre(
  nodes: Node[],
  edges: Edge[],
  options?: DagreLayoutOptions,
): {
  nodes: Node[];
  edges: Edge[];
  rankById: Map<string, number>;
  xCenterById: Map<string, number>;
} {
  const opts = { ...DEFAULT_DAGRE_OPTIONS, ...options };
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: "TB",
    nodesep: opts.nodesep,
    ranksep: opts.ranksep,
    align: opts.align,
    edgesep: opts.edgesep,
    ranker: opts.ranker,
  });

  // If a root node is known, force it into the first rank so it appears at the top.
  // (Root anchoring handled after nodes/edges are added.)

  for (const n of nodes) {
    g.setNode(n.id, {
      width: (n.width as number) ?? NODE_WIDTH,
      height: (n.height as number) ?? NODE_HEIGHT,
    });
  }

  for (const e of edges) {
    const relation =
      typeof e.data?.relation === "string" ? e.data.relation : "";
    const attrs = getDagreEdgeAttrs(relation, e.source, e.target, nodes);
    g.setEdge(e.source, e.target, attrs);
  }

  // If a root node is known, force it into the first rank so it appears at the top.
  if (
    typeof opts.rootId === "string" &&
    opts.rootId.trim() &&
    g.hasNode(opts.rootId)
  ) {
    g.setNode("__dagre_root_anchor__", { width: 1, height: 1 });
    g.setEdge("__dagre_root_anchor__", opts.rootId, { weight: 10, minlen: 1 });
    g.setGraph({
      ...g.graph(),
      // eslint-disable-next-line @typescript-eslint/naming-convention
      rank: "min",
    } as unknown as dagre.GraphLabel);
  }

  dagre.layout(g);

  // First pass: apply Dagre positions (center -> top-left)
  const laidOut = nodes.map((n) => {
    const pos = g.node(n.id);
    if (!pos) return n;

    return {
      ...n,
      position: {
        x: pos.x - ((n.width as number) ?? NODE_WIDTH) / 2,
        y: pos.y - ((n.height as number) ?? NODE_HEIGHT) / 2,
      },
    };
  });

  if (laidOut.length === 0) {
    return {
      nodes: laidOut,
      edges,
      rankById: new Map(),
      xCenterById: new Map(),
    };
  }

  if (opts.postLayerRedistribution !== "center-pack") {
    // Still expose rankById/xCenterById for metrics/repulsion grouping.
    const rankById = new Map<string, number>();
    const xCenterById = new Map<string, number>();

    const yCenterById = new Map<string, number>();
    for (const n of laidOut) {
      const pos = g.node(n.id) as { x?: number; y?: number } | undefined;
      if (typeof pos?.x === "number" && Number.isFinite(pos.x))
        xCenterById.set(n.id, pos.x);
      if (typeof pos?.y === "number" && Number.isFinite(pos.y))
        yCenterById.set(n.id, pos.y);
    }
    const uniqueY = Array.from(new Set(yCenterById.values())).sort(
      (a, b) => a - b,
    );
    const rankByY = new Map<number, number>();
    uniqueY.forEach((y, idx) => rankByY.set(y, idx));
    for (const n of laidOut) {
      const yc = yCenterById.get(n.id);
      const rank = typeof yc === "number" ? rankByY.get(yc) : undefined;
      if (typeof rank === "number") rankById.set(n.id, rank);
    }

    return { nodes: laidOut, edges, rankById, xCenterById };
  }

  // Compute global horizontal bounds from Dagre result.
  let globalMinX = Number.POSITIVE_INFINITY;
  let globalMaxX = Number.NEGATIVE_INFINITY;
  for (const n of laidOut) {
    const w = (n.width as number) ?? NODE_WIDTH;
    const left = n.position.x;
    const right = n.position.x + w;
    if (left < globalMinX) globalMinX = left;
    if (right > globalMaxX) globalMaxX = right;
  }

  if (!Number.isFinite(globalMinX) || !Number.isFinite(globalMaxX)) {
    return {
      nodes: laidOut,
      edges,
      rankById: new Map(),
      xCenterById: new Map(),
    };
  }

  // Group nodes into layers by Dagre-computed Y centers.
  // In dagre@0.8.5 (this environment), g.node(id) does not reliably expose
  // rank/order. The ONLY stable source of truth for layering is the set of
  // Y centers after layout.
  //
  // Critical constraint: we never change rank (Y). We will later do y-snap so
  // all nodes in the same inferred rank share identical Y.
  type Layer = Node[];
  const layersByRank = new Map<number, Layer>();
  const rankByNodeId = new Map<string, number>();
  const yCenterByNodeId = new Map<string, number>();
  const xCenterByNodeId = new Map<string, number>();

  // Collect centers.
  for (const n of laidOut) {
    const pos = g.node(n.id) as { x?: number; y?: number } | undefined;
    if (typeof pos?.y === "number" && Number.isFinite(pos.y)) {
      yCenterByNodeId.set(n.id, pos.y);
    }
    if (typeof pos?.x === "number" && Number.isFinite(pos.x)) {
      xCenterByNodeId.set(n.id, pos.x);
    }
  }

  // Build rank mapping from unique Y centers (ascending).
  const uniqueY = Array.from(new Set(yCenterByNodeId.values())).sort(
    (a, b) => a - b,
  );
  const rankByY = new Map<number, number>();
  uniqueY.forEach((y, idx) => rankByY.set(y, idx));

  for (const n of laidOut) {
    const yc = yCenterByNodeId.get(n.id);
    const rank = typeof yc === "number" ? rankByY.get(yc) : undefined;
    if (typeof rank === "number") {
      rankByNodeId.set(n.id, rank);
      const layer = layersByRank.get(rank);
      if (layer) layer.push(n);
      else layersByRank.set(rank, [n]);
      // eslint-disable-next-line no-continue
      continue;
    }

    // Fallback for nodes without a Dagre y-center:
    // assign them a stable rank close to their current y position.
    // This avoids creating a pseudo-layer that breaks global y alignment.
    let bestRank = 0;
    if (uniqueY.length > 0) {
      const yTopLeft = n.position.y + (((n.height as number) ?? NODE_HEIGHT) / 2);
      let bestDist = Number.POSITIVE_INFINITY;
      for (let i = 0; i < uniqueY.length; i += 1) {
        const d = Math.abs(uniqueY[i] - yTopLeft);
        if (d < bestDist) {
          bestDist = d;
          bestRank = i;
        }
      }
    }

    rankByNodeId.set(n.id, bestRank);
    const layer = layersByRank.get(bestRank);
    if (layer) layer.push(n);
    else layersByRank.set(bestRank, [n]);
  }

  const layers: Array<{ rank: number; nodes: Layer }> = [
    ...layersByRank.entries(),
  ]
    .sort((a, b) => a[0] - b[0])
    .map(([rank, layer]) => ({ rank, nodes: layer }));

  // y-snap within each rank to ensure perfect horizontal alignment.
  // (GoTpreview behavior)
  const yByRank = new Map<number, number>();
  for (const n of laidOut) {
    const r = rankByNodeId.get(n.id);
    if (typeof r !== "number") continue;
    const y = n.position.y;
    const prev = yByRank.get(r);
    if (typeof prev !== "number" || y < prev) yByRank.set(r, y);
  }
  for (const n of laidOut) {
    const r = rankByNodeId.get(n.id);
    if (typeof r !== "number") continue;
    const y = yByRank.get(r);
    if (typeof y === "number") n.position.y = y;
  }

  // IMPORTANT: repulsion runs after Dagre layout and may introduce small Y drift.
  // GoTpreview does a final y-snap before returning nodes (using rankById as truth).
  // Here we do a best-effort additional y-snap keyed by the inferred rank.
  for (const n of laidOut) {
    const r = rankByNodeId.get(n.id);
    if (typeof r !== "number") continue;
    const y = yByRank.get(r);
    if (typeof y === "number") n.position.y = y;
  }

  // For each layer, redistribute nodes horizontally so that:
  // - nodes are evenly spaced
  // - the layer as a whole is centered around globalCenterX
  //
  // IMPORTANT: preserve Dagre's crossing-minimization signal for intra-layer ordering.
  // Since rank/order are not exposed, we use Dagre's x center as a stable proxy.
  for (const { nodes: layer } of layers) {
    if (layer.length === 0) {
      // eslint-disable-next-line no-continue
      // eslint-disable-next-line no-continue
      continue;
    }

    const sorted = [...layer].sort((a, b) => a.id.localeCompare(b.id));

    const widths = sorted.map((n) => (n.width as number) ?? NODE_WIDTH);
    const k = sorted.length;
    const spacing = opts.nodesep;
    const totalWidth =
      widths.reduce((acc, w) => acc + w, 0) + spacing * Math.max(0, k - 1);

    // Pack the layer, but center *this layer* to x=0 (GoTpreview behavior).
    let cursorX = -totalWidth / 2;

    for (let i = 0; i < sorted.length; i += 1) {
      const n = sorted[i];
      const w = widths[i];
      // Place node with its left at cursorX
      n.position.x = cursorX;
      cursorX += w + spacing;
    }

    // If a layer has only one node, still keep it centered on x=0.
    // (GoTpreview behavior)
    if (sorted.length === 1) {
      const only = sorted[0];
      const w = (only.width as number) ?? NODE_WIDTH;
      only.position.x = -w / 2;
    }
  }

  // Second pass: re-center the whole graph around x=0.
  // This makes the overall view symmetric even after later post-processing
  // (e.g. applyRepulsion) introduces slight drift.
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  for (const n of laidOut) {
    const w = (n.width as number) ?? NODE_WIDTH;
    const left = n.position.x;
    const right = n.position.x + w;
    if (left < minX) minX = left;
    if (right > maxX) maxX = right;
  }

  if (Number.isFinite(minX) && Number.isFinite(maxX)) {
    const centerX = (minX + maxX) / 2;
    for (const n of laidOut) {
      n.position.x -= centerX;
    }
  }

  return {
    nodes: laidOut,
    edges,
    rankById: rankByNodeId,
    xCenterById: xCenterByNodeId,
  };
}

function getNodeId(v: string | { id: string }): string {
  return typeof v === "string" ? v : v.id;
}

function countAdjacentLayerCrossings(
  edges: Array<{
    source: string | { id: string };
    target: string | { id: string };
  }>,
  rankById: Map<string, number>,
  orderInRankById: Map<string, number>,
): number {
  // Count crossings across all adjacent rank pairs.
  // Two edges (a->b) and (c->d) cross between r and r+1 iff:
  //   order(a) < order(c) but order(b) > order(d)   (in same adjacent ranks)
  const edgesByRankPair = new Map<string, Array<{ s: string; t: string }>>();

  for (const e of edges) {
    const s = getNodeId(e.source);
    const t = getNodeId(e.target);
    const rs = rankById.get(s);
    const rt = rankById.get(t);
    if (typeof rs !== "number" || typeof rt !== "number")
      // eslint-disable-next-line no-continue
      continue;
    if (rt !== rs + 1)
      // eslint-disable-next-line no-continue
      continue;

    const key = `${rs}->${rt}`;
    const arr = edgesByRankPair.get(key);
    const entry = { s, t };
    if (arr) arr.push(entry);
    else edgesByRankPair.set(key, [entry]);
  }

  let total = 0;
  for (const arr of edgesByRankPair.values()) {
    for (let i = 0; i < arr.length; i += 1) {
      for (let j = i + 1; j < arr.length; j += 1) {
        const e1 = arr[i];
        const e2 = arr[j];
        const s1 = orderInRankById.get(e1.s);
        const s2 = orderInRankById.get(e2.s);
        const t1 = orderInRankById.get(e1.t);
        const t2 = orderInRankById.get(e2.t);
        if (
          typeof s1 !== "number" ||
          typeof s2 !== "number" ||
          typeof t1 !== "number" ||
          typeof t2 !== "number"
        ) {
          // eslint-disable-next-line no-continue
          continue;
        }
        if ((s1 < s2 && t1 > t2) || (s1 > s2 && t1 < t2)) total += 1;
      }
    }
  }

  return total;
}

export function buildOrderInRankById(
  nodes: Node[],
  rankById: Map<string, number>,
  xCenterById: Map<string, number>,
): Map<string, number> {
  const nodesByRank = new Map<number, Node[]>();
  for (const n of nodes) {
    const r = rankById.get(n.id);
    if (typeof r !== "number")
      // eslint-disable-next-line no-continue
      continue;
    const arr = nodesByRank.get(r);
    if (arr) arr.push(n);
    else nodesByRank.set(r, [n]);
  }

  const order = new Map<string, number>();
  for (const [r, arr] of nodesByRank.entries()) {
    const sorted = [...arr].sort((a, b) => {
      const xa = xCenterById.get(a.id);
      const xb = xCenterById.get(b.id);
      if (typeof xa === "number" && typeof xb === "number" && xa !== xb)
        return xa - xb;
      if (typeof xa === "number" && typeof xb !== "number") return -1;
      if (typeof xa !== "number" && typeof xb === "number") return 1;
      return a.id.localeCompare(b.id);
    });
    sorted.forEach((n, idx) => order.set(n.id, idx));
    // r is intentionally unused; we iterate over each rank.
    void r; // eslint-disable-line no-void
  }
  return order;
}

export function computeCrossings(
  nodes: Node[],
  edges: Edge[],
  rankById: Map<string, number>,
  xCenterById: Map<string, number>,
): { total: number } {
  const orderInRankById = buildOrderInRankById(nodes, rankById, xCenterById);
  const total = countAdjacentLayerCrossings(edges, rankById, orderInRankById);
  return { total };
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function getRect(n: Node, pad: number) {
  const w = (n.width as number) ?? NODE_WIDTH;
  const h = (n.height as number) ?? NODE_HEIGHT;
  return {
    x1: n.position.x - pad,
    y1: n.position.y - pad,
    x2: n.position.x + w + pad,
    y2: n.position.y + h + pad,
    w,
    h,
  };
}

export type ForceLayoutOptions = {
  iterations?: number;
  linkDistance?: number;
  linkStrength?: number;
  chargeStrength?: number;
  collisionPadding?: number;
  seed?: string | number;
  pinnedIds?: Set<string>;
};

function mulberry32(seed: number): () => number {
  // eslint-disable-next-line no-bitwise
  let state = seed | 0;
  return () => {
    // Mulberry32 PRNG (bitwise ops by design)
    // eslint-disable-next-line no-bitwise
    state = (state + 0x6d2b79f5) | 0;
    // eslint-disable-next-line no-bitwise
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    // eslint-disable-next-line no-bitwise
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    // eslint-disable-next-line no-bitwise
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(seed: string | number): number {
  if (typeof seed === "number" && Number.isFinite(seed)) {
    // eslint-disable-next-line no-bitwise
    return seed | 0;
  }
  const s = String(seed);
  // Simple FNV-1a 32-bit hash (bitwise ops by design)
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    // eslint-disable-next-line no-bitwise
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  // eslint-disable-next-line no-bitwise
  return h | 0;
}

function hashStringToUnitInterval(s: string): number {
  // Simple FNV-1a 32-bit hash (bitwise ops by design)
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    // eslint-disable-next-line no-bitwise
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  // eslint-disable-next-line no-bitwise
  return (h >>> 0) / 4294967296;
}

type ForceNodeDatum = SimulationNodeDatum & {
  id: string;
  width: number;
  height: number;
  fx?: number | null;
  fy?: number | null;
};

export function layoutWithD3Force(
  nodes: Node[],
  edges: Edge[],
  options?: ForceLayoutOptions,
): { nodes: Node[]; edges: Edge[] } {
  const iterations = options?.iterations ?? 250;
  const linkDistance = options?.linkDistance ?? 220;
  const linkStrength = options?.linkStrength ?? 0.6;
  const chargeStrength = options?.chargeStrength ?? -900;
  const collisionPadding = options?.collisionPadding ?? 18;
  const seed = options?.seed ?? "default";
  const pinnedIds = options?.pinnedIds ?? new Set<string>();

  // Stable order for determinism.
  const sortedNodes = [...nodes].sort((a, b) => a.id.localeCompare(b.id));

  const seedInt = hashSeed(seed);
  const rng = mulberry32(seedInt);

  const nodeData: ForceNodeDatum[] = sortedNodes.map((n) => {
    const w = (n.width as number) ?? NODE_WIDTH;
    const h = (n.height as number) ?? NODE_HEIGHT;

    // Deterministic initial placement from (seed, id).
    const ux = hashStringToUnitInterval(`${seedInt}:${n.id}:x`);
    const uy = hashStringToUnitInterval(`${seedInt}:${n.id}:y`);
    const x = (ux - 0.5) * 800;
    const y = (uy - 0.5) * 500;

    return {
      id: n.id,
      width: w,
      height: h,
      x,
      y,
      fx: pinnedIds.has(n.id) ? n.position.x + w / 2 : null,
      fy: pinnedIds.has(n.id) ? n.position.y + h / 2 : null,
    };
  });

  const nodeById = new Map(nodeData.map((d) => [d.id, d] as const));

  const simLinks = edges
    .map((e) => {
      const s = nodeById.get(e.source);
      const t = nodeById.get(e.target);
      if (!s || !t) return null;
      return { source: s, target: t };
    })
    .filter(
      (x): x is { source: ForceNodeDatum; target: ForceNodeDatum } =>
        x !== null,
    );

  const sim = forceSimulation(nodeData)
    .randomSource(rng)
    .force(
      "link",
      forceLink(simLinks)
        .id((d) => (d as ForceNodeDatum).id)
        .distance(linkDistance)
        .strength(linkStrength),
    )
    .force("charge", forceManyBody().strength(chargeStrength))
    .force(
      "collide",
      forceCollide<ForceNodeDatum>().radius(
        (d) => Math.max(d.width, d.height) / 2 + collisionPadding,
      ),
    )
    .force("center", forceCenter(0, 0))
    .stop();

  for (let i = 0; i < iterations; i += 1) sim.tick();

  const outNodes = nodes.map((n) => {
    const d = nodeById.get(n.id);
    const dx = d?.x;
    const dy = d?.y;
    if (!d || !Number.isFinite(dx) || !Number.isFinite(dy)) return n;
    const x = dx as number;
    const y = dy as number;
    const w = (n.width as number) ?? NODE_WIDTH;
    const h = (n.height as number) ?? NODE_HEIGHT;
    return {
      ...n,
      position: {
        x: x - w / 2,
        y: y - h / 2,
      },
    };
  });

  return { nodes: outNodes, edges };
}

export function applyRepulsion(
  nodes: Node[],
  options?: {
    pinnedIds?: Set<string>;
    iterations?: number;
    padding?: number;
    maxStep?: number;
    yDamping?: number;
    /**
     * Optional grouping key: if provided, repulsion is only applied within each group.
     * Useful for preserving Dagre ranks (layering) while de-overlapping in X.
     */
    groupById?: Map<string, string | number>;
  },
): Node[] {
  const pinnedIds = options?.pinnedIds ?? new Set<string>();
  const iterations = options?.iterations ?? 60;
  const padding = options?.padding ?? 26;
  const maxStep = options?.maxStep ?? 40;
  const yDamping = options?.yDamping ?? 0.2;

  const next = nodes.map((n) => ({
    ...n,
    position: { ...n.position },
  }));

  for (let iter = 0; iter < iterations; iter += 1) {
    const deltas = new Map<string, { dx: number; dy: number }>();

    for (let i = 0; i < next.length; i += 1) {
      const a = next[i];
      const aPinned = pinnedIds.has(a.id);
      const ra = getRect(a, padding);

      for (let j = i + 1; j < next.length; j += 1) {
        const b = next[j];
        const bPinned = pinnedIds.has(b.id);

        const groupById = options?.groupById;
        if (groupById) {
          const ga = groupById.get(a.id);
          const gb = groupById.get(b.id);
          if (ga !== gb) {
            // eslint-disable-next-line no-continue
            // eslint-disable-next-line no-continue
            continue;
          }
        }
        if (aPinned && bPinned) {
          // eslint-disable-next-line no-continue
          // eslint-disable-next-line no-continue
          continue;
        }

        const rb = getRect(b, padding);

        const overlapX = Math.min(ra.x2, rb.x2) - Math.max(ra.x1, rb.x1);
        const overlapY = Math.min(ra.y2, rb.y2) - Math.max(ra.y1, rb.y1);
        if (overlapX <= 0 || overlapY <= 0) {
          // eslint-disable-next-line no-continue
          // eslint-disable-next-line no-continue
          continue;
        }

        const acx = (ra.x1 + ra.x2) / 2;
        const bcx = (rb.x1 + rb.x2) / 2;
        const acy = (ra.y1 + ra.y2) / 2;
        const bcy = (rb.y1 + rb.y2) / 2;

        const sx = acx <= bcx ? -1 : 1;
        const sy = acy <= bcy ? -1 : 1;

        // Prefer resolving overlap along X to preserve TB layering.
        const sepX = overlapX + 6;
        const sepY = overlapY + 6;

        if (!aPinned) {
          const d = deltas.get(a.id) ?? { dx: 0, dy: 0 };
          d.dx += (sx * sepX) / 2;
          d.dy += (sy * sepY) / 2;
          deltas.set(a.id, d);
        }

        if (!bPinned) {
          const d = deltas.get(b.id) ?? { dx: 0, dy: 0 };
          d.dx -= (sx * sepX) / 2;
          d.dy -= (sy * sepY) / 2;
          deltas.set(b.id, d);
        }
      }
    }

    for (const n of next) {
      if (pinnedIds.has(n.id)) {
        // eslint-disable-next-line no-continue
        continue;
      }
      const d = deltas.get(n.id);
      if (!d) {
        // eslint-disable-next-line no-continue
        continue;
      }

      const dx = clamp(d.dx ?? 0, -maxStep, maxStep);
      const dy = clamp((d.dy ?? 0) * yDamping, -maxStep, maxStep);

      n.position.x += dx;
      n.position.y += dy;
    }
  }

  return next;
}
