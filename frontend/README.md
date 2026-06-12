# Graph of Trace — Viewer

Standalone React + ReactFlow viewer for `got.json`. Renders the scientific-agent
execution trace as a DAG. No dependency on any specific agent framework.

## Run

```bash
npm install
npm run dev        # http://localhost:4500
npm run build      # typecheck + production bundle into dist/
npm run preview    # serve the built bundle
```

## Loading data

By default the app fetches `./got.json` (served from `public/got.json`; a sample
is included). Override the source without rebuilding:

- `?src=<url>` — e.g. `http://localhost:4500/?src=/api/conversations/<id>/got`
- `VITE_GOT_URL=<url>` at build time
- replace `public/got.json`, or serve the backend's output file at `./got.json`

It polls with adaptive backoff (3s → 10s → 30s → stop once stable), so live
updates appear as the agent records new subtasks.

## Layout

- **Dagre** (default): layered top-down DAG, root anchored to the top rank.
- **D3-force**: force-directed; toggle via the "Layout" button.

All layout/conversion logic lives in `src/got-to-reactflow.ts` (pure functions,
reusable independently of the UI shell in `src/MindmapView.tsx`).

## Data contract

Expects the `got.json` schema documented in the repo root README
(`meta` + `nodes[]` with `id`, `title`, `description`, `parents[]`, `artifacts[]`).
