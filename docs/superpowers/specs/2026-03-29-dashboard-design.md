# Parameter Golf Dashboard — Design Spec

Read-only SvelteKit dashboard on Cloudflare Workers for monitoring the Parameter Golf autoresearch system. Audience: Robin + small group of colleagues/friends. Displays full experiment and research history.

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Framework | SvelteKit | SSR + client hydration, CF Workers adapter |
| Styling | Tailwind v4 | Design system maps directly |
| Charts | LayerCake | Svelte-native, SVG-based |
| Technique graph | d3-force (layout only) | Force-directed math, Svelte renders SVG |
| Markdown | marked | Render program.md / strategy.md |
| Font | BDO Grotesk (local @font-face) | Primary sans-serif, weights 300-900 |
| Code font | Fira Code | Fallback: Cascadia Code, Consolas, monospace |
| Deployment | CF Workers via @sveltejs/adapter-cloudflare | |
| Structured storage | D1 | Experiments, research items, budget |
| Blob storage | KV | Markdown docs, technique map |
| Ingest auth | Bearer token | Simple, sufficient for push-only |

## Design System

Sharp, flat, monochromatic, typography-driven. No border-radius anywhere, no shadows, no decorative elements. Content-first minimalism.

### Tailwind v4 Theme

```css
@theme {
  --radius-*: 0px;
  --font-sans: var(--font-bdo), system-ui, sans-serif;
}
```

### Color System

Four colors only. No accent colors. Only variety is in code syntax highlighting.

```css
:root {
  --surface: hsl(0 0% 90%);
  --on-surface: hsl(0 0% 5%);
  --muted: hsl(0 0% 40%);
  --edge: hsl(0 0% 78%);
}

@media (prefers-color-scheme: dark) {
  :root {
    --surface: hsl(0 0% 5%);
    --on-surface: hsl(0 0% 90%);
    --muted: hsl(0 0% 60%);
    --edge: hsl(0 0% 18%);
  }
}
```

Supports manual override via `data-theme="light"` / `data-theme="dark"` on `<html>`.

### Typography

| Element | Size | Weight | Extra |
|---------|------|--------|-------|
| Logo/title | text-5xl sm:text-8xl | font-semibold | leading-none tracking-tighter |
| h1 (prose) | 1.75rem | default | mb-3 |
| h2 (prose) | 1.375rem | default | mt-8 mb-3 |
| h3 (prose) | 1.125rem | default | mt-8 mb-3 |
| Body | inherited | default | leading-relaxed (1.75 line-height) |
| Section labels | text-xs | font-medium | tracking-wider text-muted |
| Metadata/dates | text-xs or text-sm | default | text-muted |
| Nav links | text-sm | default | text-muted |

No uppercase transforms anywhere. Title case or sentence case only.

### Component Patterns

**Cards:** `border border-edge p-4 transition-opacity duration-150 hover:opacity-70`

**Buttons:** `border border-edge bg-transparent px-3 py-2 text-sm text-on-surface transition-colors hover:bg-edge disabled:opacity-50`

**Inputs:** `border border-edge bg-transparent px-3 py-2 text-sm text-on-surface placeholder:text-muted focus:border-on-surface focus:outline-none`

**Links:** `text-on-surface transition-opacity duration-150 hover:opacity-70`. Prose links add `underline` with `text-underline-offset: 3px`.

**Dividers:** `border-b border-edge` between list items.

**Footer:** Sticky bottom, `border-t border-edge`, flex between.

### Interaction & Animation

- Color changes: `transition-colors duration-150`
- Opacity: `transition-opacity duration-150`
- Hover: `hover:opacity-70` (primary) or `hover:bg-edge` (buttons)
- Inactive state: `opacity-40`
- Page enter: Svelte `fly` — `fly={{ y: 24, duration: 500 }}` with opacity fade
- Nav enter: `fly={{ y: -12, duration: 400 }}`
- Respects `prefers-reduced-motion` via Svelte's `reducedMotion` store

### Key Rules

1. Zero border-radius — everything is sharp rectangles
2. Zero shadows — completely flat
3. Only 4 colors — surface, on-surface, muted, edge
4. Opacity for interaction — hover dims to 70%, inactive items at 40%
5. 150ms transitions — fast, subtle, never flashy
6. 1px borders only — always border-edge color
7. No decorative elements — content speaks for itself
8. Monospace for code only — everything else is sans-serif
9. No uppercase transforms — title case or sentence case only

## Data Model

### D1 Tables

**`experiments`**

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | Commit hash |
| tier | TEXT | 'local' or 'runpod' |
| val_bpb | REAL | Validation bits-per-byte |
| artifact_bytes | INTEGER | Artifact size |
| memory_gb | REAL | Memory used |
| status | TEXT | 'keep', 'discard', or 'crash' |
| promoted | INTEGER | 0 or 1 |
| cost_usd | REAL | RunPod cost |
| description | TEXT | Experiment description |
| source_item | TEXT | Research item that inspired it |
| created_at | TEXT | ISO timestamp |

**`research_items`**

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | e.g. 'arxiv:2401.12345' |
| score | REAL | Total score (0-15) |
| tier | TEXT | 'A', 'B', or 'C' |
| bpb_impact | REAL | Score dimension |
| size_compat | REAL | Score dimension |
| time_compat | REAL | Score dimension |
| implement | REAL | Score dimension |
| novelty | REAL | Score dimension |
| summary | TEXT | Agent-generated summary |
| flags | TEXT | JSON array of flags |
| verified | INTEGER | 0 or 1 |
| graded_at | TEXT | ISO timestamp |
| verified_at | TEXT | ISO timestamp, null if unverified |

**`budget_runs`**

| Column | Type | Description |
|--------|------|-------------|
| run_id | TEXT PRIMARY KEY | RunPod run identifier |
| started_at | TEXT | ISO timestamp |
| duration_s | INTEGER | Run duration in seconds |
| cost_usd | REAL | Run cost |
| val_bpb | REAL | Result bpb |
| artifact_bytes | INTEGER | Result artifact size |
| promoted_from | TEXT | Experiment id |

**`budget_snapshot`** (single row)

| Column | Type | Description |
|--------|------|-------------|
| total_credits | REAL | Total available credits |
| spent | REAL | Total spent |
| min_reserve | REAL | Reserve floor |
| updated_at | TEXT | ISO timestamp |

**`agent_status`**

| Column | Type | Description |
|--------|------|-------------|
| agent | TEXT PRIMARY KEY | 'experiment' or 'research' |
| status | TEXT | 'running', 'idle', or 'crashed' |
| last_activity | TEXT | ISO timestamp |
| restart_count | INTEGER | Number of restarts |

### KV Keys

| Key | Value | Description |
|-----|-------|-------------|
| `doc:program` | Markdown string | Current program.md |
| `doc:strategy` | Markdown string | Current strategy.md |
| `doc:technique_map` | JSON string | technique_map.json |
| `meta:last_push` | ISO timestamp | Last orchestrator push |
| `meta:sota_bpb` | REAL as string | Current SOTA val_bpb target (pushed via heartbeat) |
| `meta:pipeline_counts` | JSON string | `{ fetched, graded, verified, injected }` counts from local caches |

## Ingest API

### `POST /api/ingest`

Bearer token auth via `Authorization: Bearer <DASHBOARD_TOKEN>`.

Request body:
```json
{
  "event": "<event_type>",
  "data": { ... }
}
```

### Event Types

| Event | Data | Action |
|-------|------|--------|
| `experiment_complete` | Experiment row object | Upsert into `experiments` |
| `research_graded` | Array of research items | Upsert into `research_items` |
| `research_verified` | `{ id, verified_at }` | Update row: `verified = 1` |
| `budget_update` | `{ snapshot, run? }` | Upsert `budget_snapshot`, optionally insert `budget_runs` |
| `doc_update` | `{ key, content }` | KV put to `doc:{key}` |
| `heartbeat` | `{ agents, sota_bpb, pipeline_counts }` | Upsert `agent_status` rows, KV put `meta:sota_bpb` + `meta:pipeline_counts`, update `meta:last_push` |

Returns `200 { "ok": true }` on success. `401` for bad token. `400` for invalid event/payload.

## Pages

### Overview (`/`)

At-a-glance page. No charts — key numbers and recent activity.

- **Status bar** — agent status indicators (running/idle/crashed), last push timestamp
- **Key metrics row** — best val_bpb (local + runpod), distance to SOTA, artifact size headroom, budget remaining
- **Recent experiments** — last 10, compact table: description, tier, val_bpb, status
- **Recent research** — last 10 graded items, compact: summary, score, tier

### Experiments (`/experiments`)

Full experiment history with performance chart.

- **Bpb chart** — LayerCake SVG time-series line chart. Best val_bpb over time. Dots: muted for local, on-surface for runpod. Promotion events as vertical lines.
- **Filters** — tier (all/local/runpod), status (all/keep/discard/crash). `<select>` inputs.
- **Table** — full history, sortable columns, paginated (50/page). Columns: description, tier, val_bpb, artifact_bytes, status, cost, source_item, date.

### Research (`/research`)

Research pipeline and findings.

- **Pipeline funnel** — four horizontal bars: fetched → graded → verified → injected (counts from `meta:pipeline_counts`, pushed via heartbeat)
- **Tier breakdown** — count of A/B/C items
- **Table** — all items, sortable/filterable by tier and score. Expandable rows show 5-dimension score breakdown and full summary.

### Budget (`/budget`)

Spend tracking and runway projection.

- **Budget bar** — horizontal bar with three segments: spent, available, reserve. Dollar amounts labeled.
- **Burn rate** — sum of cost over last 7 days / 7 = $/day. Estimated days remaining = (remaining - reserve) / burn_rate.
- **Run history** — table of all RunPod runs: date, duration, cost, val_bpb, promoted_from.

### Strategy (`/strategy`)

Living documents and technique relationships.

- **Technique map** — interactive SVG node-link diagram. d3-force for layout, Svelte renders nodes/edges. Node labels = technique name. Node opacity by status: proven = 100%, exploring = 70%, dead_end = 40%, untried = 70% with dashed border. Hovering a node highlights its connections.
- **Program.md** — rendered markdown from KV
- **Strategy.md** — rendered markdown from KV

### Navigation

Top horizontal nav bar. "Parameter Golf" title on the left (text-5xl sm:text-8xl, font-semibold, leading-none, tracking-tighter). Page links on the right (text-sm, text-muted). Active link at full opacity, inactive at 40%.

### Page Data Loading

All data loaded server-side via `+page.server.ts` load functions querying D1/KV directly. Pages SSR'd with full data — no loading spinners, no client-side fetch.

| Page | Queries |
|------|---------|
| Overview | Latest 10 experiments, latest 10 research, budget snapshot, agent status |
| Experiments | All experiments (paginated 50/page), best bpb per day for chart |
| Research | All research items (paginated), count by tier, count by pipeline stage |
| Budget | Budget snapshot, all budget runs, burn rate (last 7 days) |
| Strategy | KV gets: program, strategy, technique_map |

### Freshness

No polling or websockets. "Last updated: {timestamp}" in footer from `meta:last_push`. Refresh page for new data.

## Orchestrator Integration

New `compute/dashboard.py` module with a `DashboardPusher` class.

### Configuration

Reads from `.env`:
- `DASHBOARD_URL` — the Worker URL (e.g. `https://pgolf-dashboard.<subdomain>.workers.dev`)
- `DASHBOARD_TOKEN` — bearer token matching the Worker's secret

If `DASHBOARD_URL` is unset, all methods are no-ops.

### Push Methods

- `push_experiment(row: dict)` — called after results.tsv append
- `push_research(items: list[dict])` — called after graded_cache.jsonl append
- `push_verified(item_id: str)` — called after verified_cache.jsonl append
- `push_budget(snapshot: dict, run: dict | None)` — called after budget.json write
- `push_doc(key: str, content: str)` — called after program.md / strategy.md / technique_map.json write
- `push_heartbeat(statuses: list[dict], sota_bpb: float, pipeline_counts: dict)` — called every 5 minutes, includes SOTA target and pipeline stage counts from local caches

All methods: single `POST /api/ingest`, 5s timeout, fire-and-forget. Dashboard being down never blocks agents.

## Project Structure

```
dashboard/
├── src/
│   ├── lib/
│   │   ├── server/
│   │   │   ├── db.ts            -- D1 query helpers
│   │   │   └── kv.ts            -- KV read helpers
│   │   ├── components/
│   │   │   ├── Nav.svelte
│   │   │   ├── MetricCard.svelte
│   │   │   ├── DataTable.svelte
│   │   │   ├── BpbChart.svelte
│   │   │   ├── FunnelBar.svelte
│   │   │   ├── BudgetBar.svelte
│   │   │   └── TechGraph.svelte
│   │   ├── styles/
│   │   │   ├── app.css          -- design system tokens, @font-face, tailwind
│   │   │   └── fonts/           -- BDO Grotesk woff2 files
│   │   └── types.ts             -- shared TypeScript types
│   ├── routes/
│   │   ├── +layout.svelte       -- Nav, page transitions, global styles
│   │   ├── +page.svelte         -- Overview
│   │   ├── +page.server.ts
│   │   ├── experiments/
│   │   │   ├── +page.svelte
│   │   │   └── +page.server.ts
│   │   ├── research/
│   │   │   ├── +page.svelte
│   │   │   └── +page.server.ts
│   │   ├── budget/
│   │   │   ├── +page.svelte
│   │   │   └── +page.server.ts
│   │   ├── strategy/
│   │   │   ├── +page.svelte
│   │   │   └── +page.server.ts
│   │   └── api/
│   │       └── ingest/
│   │           └── +server.ts   -- POST endpoint for orchestrator push
│   └── app.html
├── wrangler.toml                -- D1 + KV bindings
├── svelte.config.js
├── tailwind.config.ts
└── package.json
```

Lives inside the existing repo as a `dashboard/` subdirectory.
