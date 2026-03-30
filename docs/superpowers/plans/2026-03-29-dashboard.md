# Parameter Golf Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only SvelteKit dashboard on Cloudflare Workers that displays experiment history, research findings, budget health, and strategy docs for the Parameter Golf autoresearch system.

**Architecture:** SvelteKit with CF Workers adapter. D1 for structured data (experiments, research, budget). KV for blob data (markdown docs, technique map). Orchestrator pushes events to a `POST /api/ingest` endpoint. All pages are SSR'd from D1/KV queries.

**Tech Stack:** SvelteKit, Tailwind v4, LayerCake (charts), d3-force (graph layout), marked (markdown), Cloudflare D1 + KV, Python httpx (pusher)

**Spec:** `docs/superpowers/specs/2026-03-29-dashboard-design.md`

---

## File Structure

```
dashboard/
├── src/
│   ├── app.d.ts                        -- CF platform type declarations
│   ├── app.html                        -- HTML shell
│   ├── lib/
│   │   ├── server/
│   │   │   ├── db.ts                   -- D1 query helpers
│   │   │   └── kv.ts                   -- KV read helpers
│   │   ├── components/
│   │   │   ├── Nav.svelte              -- Top navigation bar
│   │   │   ├── MetricCard.svelte       -- Key metric display
│   │   │   ├── DataTable.svelte        -- Sortable, paginated table
│   │   │   ├── BpbChart.svelte         -- LayerCake time-series chart
│   │   │   ├── FunnelBar.svelte        -- Horizontal pipeline bar
│   │   │   ├── BudgetBar.svelte        -- Three-segment spend bar
│   │   │   └── TechGraph.svelte        -- d3-force technique graph
│   │   ├── styles/
│   │   │   └── app.css                 -- Design system tokens, @font-face, tailwind
│   │   └── types.ts                    -- Shared TypeScript types
│   ├── routes/
│   │   ├── +layout.svelte              -- Global layout with nav + footer
│   │   ├── +layout.server.ts           -- Load shared data (agent status, last push)
│   │   ├── +page.svelte                -- Overview page
│   │   ├── +page.server.ts             -- Overview data loader
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
│   │           └── +server.ts          -- POST endpoint for orchestrator push
│   └── hooks.server.ts                 -- (empty, placeholder for future auth)
├── static/
│   └── fonts/                          -- BDO Grotesk woff2 files
├── migrations/
│   └── 0001_init.sql                   -- D1 schema
├── tests/
│   └── ingest.test.ts                  -- Ingest endpoint tests
├── wrangler.jsonc                      -- D1 + KV bindings
├── svelte.config.js
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── .gitignore

compute/
└── dashboard.py                        -- DashboardPusher class (Python)

tests/
└── test_dashboard_pusher.py            -- Pusher unit tests (Python)
```

---

### Task 1: Scaffold SvelteKit project

**Files:**
- Create: `dashboard/package.json`
- Create: `dashboard/svelte.config.js`
- Create: `dashboard/vite.config.ts`
- Create: `dashboard/tsconfig.json`
- Create: `dashboard/.gitignore`
- Create: `dashboard/wrangler.jsonc`
- Create: `dashboard/src/app.html`
- Create: `dashboard/src/app.d.ts`

- [ ] **Step 1: Create the dashboard directory and initialize the project**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
mkdir -p dashboard
cd dashboard
npm create svelte@latest . -- --template skeleton --types typescript
```

Select: Skeleton project, Yes to TypeScript, No to additional options.

- [ ] **Step 2: Install dependencies**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npm install @sveltejs/adapter-cloudflare
npm install -D tailwindcss @tailwindcss/vite
npm install layercake d3-force d3-shape d3-scale marked
npm install -D @types/d3-force @types/d3-shape @types/d3-scale @cloudflare/workers-types wrangler
```

- [ ] **Step 3: Configure the CF Workers adapter**

`dashboard/svelte.config.js`:
```js
import adapter from '@sveltejs/adapter-cloudflare';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			platformProxy: {
				persist: true
			}
		})
	}
};

export default config;
```

- [ ] **Step 4: Configure Vite with Tailwind**

`dashboard/vite.config.ts`:
```ts
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()]
});
```

- [ ] **Step 5: Write the wrangler config with D1 + KV bindings**

`dashboard/wrangler.jsonc`:
```jsonc
{
	"name": "pgolf-dashboard",
	"main": ".svelte-kit/cloudflare/_worker.js",
	"compatibility_date": "2025-04-01",
	"compatibility_flags": ["nodejs_als"],
	"assets": {
		"binding": "ASSETS",
		"directory": ".svelte-kit/cloudflare"
	},
	"d1_databases": [
		{
			"binding": "DB",
			"database_name": "pgolf-dashboard-db",
			"database_id": "<will-be-set-after-creation>"
		}
	],
	"kv_namespaces": [
		{
			"binding": "KV",
			"id": "<will-be-set-after-creation>"
		}
	],
	"vars": {
		"DASHBOARD_TOKEN": ""
	}
}
```

- [ ] **Step 6: Declare CF platform types**

`dashboard/src/app.d.ts`:
```ts
import type { D1Database, KVNamespace } from '@cloudflare/workers-types';

declare global {
	namespace App {
		interface Platform {
			env: {
				DB: D1Database;
				KV: KVNamespace;
				DASHBOARD_TOKEN: string;
			};
		}
	}
}

export {};
```

- [ ] **Step 7: Write the HTML shell**

`dashboard/src/app.html`:
```html
<!doctype html>
<html lang="en" data-theme="">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />
		<link rel="icon" href="data:," />
		<title>Parameter Golf</title>
		%sveltekit.head%
	</head>
	<body data-sveltekit-preload-data="hover">
		<div style="display: contents">%sveltekit.body%</div>
	</body>
</html>
```

- [ ] **Step 8: Add dashboard-specific entries to .gitignore**

`dashboard/.gitignore`:
```
node_modules/
.svelte-kit/
.cloudflare/
.wrangler/
build/
```

- [ ] **Step 9: Verify the project builds**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npm run build
```

Expected: Build succeeds (may warn about missing pages — that's fine).

- [ ] **Step 10: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/
git commit -m "feat(dashboard): scaffold SvelteKit project with CF Workers adapter"
```

---

### Task 2: Design system — Tailwind v4, fonts, CSS tokens

**Files:**
- Create: `dashboard/src/lib/styles/app.css`
- Create: `dashboard/static/fonts/` (BDO Grotesk woff2 files)
- Modify: `dashboard/src/app.html` — add CSS import

- [ ] **Step 1: Download BDO Grotesk font files**

Download BDO Grotesk woff2 files (weights 300, 400, 500, 600, 700, 900) and place them in `dashboard/static/fonts/`. The font is available from Google Fonts or the BDO website. If unavailable, use Inter as a fallback with the same weight range.

```bash
mkdir -p /Users/robin/dev/parameter-golf-autoresearch/dashboard/static/fonts
# Place woff2 files here: BDOGrotesk-Light.woff2, BDOGrotesk-Regular.woff2,
# BDOGrotesk-Medium.woff2, BDOGrotesk-SemiBold.woff2, BDOGrotesk-Bold.woff2,
# BDOGrotesk-Black.woff2
```

- [ ] **Step 2: Write the design system CSS**

`dashboard/src/lib/styles/app.css`:
```css
@import 'tailwindcss';

/* --- Font faces --- */
@font-face {
	font-family: 'BDO Grotesk';
	font-weight: 300;
	font-display: swap;
	src: url('/fonts/BDOGrotesk-Light.woff2') format('woff2');
}
@font-face {
	font-family: 'BDO Grotesk';
	font-weight: 400;
	font-display: swap;
	src: url('/fonts/BDOGrotesk-Regular.woff2') format('woff2');
}
@font-face {
	font-family: 'BDO Grotesk';
	font-weight: 500;
	font-display: swap;
	src: url('/fonts/BDOGrotesk-Medium.woff2') format('woff2');
}
@font-face {
	font-family: 'BDO Grotesk';
	font-weight: 600;
	font-display: swap;
	src: url('/fonts/BDOGrotesk-SemiBold.woff2') format('woff2');
}
@font-face {
	font-family: 'BDO Grotesk';
	font-weight: 700;
	font-display: swap;
	src: url('/fonts/BDOGrotesk-Bold.woff2') format('woff2');
}
@font-face {
	font-family: 'BDO Grotesk';
	font-weight: 900;
	font-display: swap;
	src: url('/fonts/BDOGrotesk-Black.woff2') format('woff2');
}

/* --- Tailwind v4 theme overrides --- */
@theme {
	--radius-xs: 0px;
	--radius-sm: 0px;
	--radius-md: 0px;
	--radius-lg: 0px;
	--radius-xl: 0px;
	--radius-2xl: 0px;
	--radius-3xl: 0px;
	--radius-4xl: 0px;
	--font-sans: 'BDO Grotesk', system-ui, sans-serif;
	--font-mono: 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
}

/* --- Color tokens --- */
:root {
	--surface: hsl(0 0% 90%);
	--on-surface: hsl(0 0% 5%);
	--muted: hsl(0 0% 40%);
	--edge: hsl(0 0% 78%);
}

@media (prefers-color-scheme: dark) {
	:root:not([data-theme='light']) {
		--surface: hsl(0 0% 5%);
		--on-surface: hsl(0 0% 90%);
		--muted: hsl(0 0% 60%);
		--edge: hsl(0 0% 18%);
	}
}

:root[data-theme='dark'] {
	--surface: hsl(0 0% 5%);
	--on-surface: hsl(0 0% 90%);
	--muted: hsl(0 0% 60%);
	--edge: hsl(0 0% 18%);
}

/* --- Base styles --- */
body {
	background-color: var(--surface);
	color: var(--on-surface);
	font-family: var(--font-sans);
	line-height: 1.75;
}

/* --- Utility classes for design tokens --- */
@utility bg-surface {
	background-color: var(--surface);
}
@utility bg-edge {
	background-color: var(--edge);
}
@utility text-on-surface {
	color: var(--on-surface);
}
@utility text-muted {
	color: var(--muted);
}
@utility border-edge {
	border-color: var(--edge);
}
```

- [ ] **Step 3: Import the CSS in the app**

Add to the top of `dashboard/src/app.html`, inside `<head>` before `%sveltekit.head%`:

This is handled automatically by SvelteKit when we import `app.css` in the layout. We'll do that in Task 8.

- [ ] **Step 4: Verify Tailwind processes the CSS**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npm run build
```

Expected: Build succeeds without CSS errors.

- [ ] **Step 5: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/styles/ dashboard/static/fonts/
git commit -m "feat(dashboard): add design system — Tailwind v4 theme, BDO Grotesk fonts, color tokens"
```

---

### Task 3: D1 schema migration

**Files:**
- Create: `dashboard/migrations/0001_init.sql`

- [ ] **Step 1: Write the migration SQL**

`dashboard/migrations/0001_init.sql`:
```sql
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    val_bpb REAL,
    artifact_bytes INTEGER,
    memory_gb REAL,
    status TEXT NOT NULL DEFAULT 'keep',
    promoted INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    source_item TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX idx_experiments_created_at ON experiments(created_at);
CREATE INDEX idx_experiments_tier ON experiments(tier);
CREATE INDEX idx_experiments_status ON experiments(status);

CREATE TABLE research_items (
    id TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0,
    tier TEXT NOT NULL DEFAULT 'C',
    bpb_impact REAL NOT NULL DEFAULT 0,
    size_compat REAL NOT NULL DEFAULT 0,
    time_compat REAL NOT NULL DEFAULT 0,
    implement REAL NOT NULL DEFAULT 0,
    novelty REAL NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    flags TEXT NOT NULL DEFAULT '[]',
    verified INTEGER NOT NULL DEFAULT 0,
    graded_at TEXT NOT NULL,
    verified_at TEXT
);

CREATE INDEX idx_research_items_tier ON research_items(tier);
CREATE INDEX idx_research_items_score ON research_items(score);

CREATE TABLE budget_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    duration_s INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    val_bpb REAL,
    artifact_bytes INTEGER,
    promoted_from TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_budget_runs_started_at ON budget_runs(started_at);

CREATE TABLE budget_snapshot (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_credits REAL NOT NULL DEFAULT 0,
    spent REAL NOT NULL DEFAULT 0,
    min_reserve REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

INSERT INTO budget_snapshot (id, total_credits, spent, min_reserve, updated_at)
VALUES (1, 500.0, 0.0, 50.0, '');

CREATE TABLE agent_status (
    agent TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'idle',
    last_activity TEXT NOT NULL DEFAULT '',
    restart_count INTEGER NOT NULL DEFAULT 0
);

INSERT INTO agent_status (agent, status) VALUES ('experiment', 'idle');
INSERT INTO agent_status (agent, status) VALUES ('research', 'idle');
```

- [ ] **Step 2: Create the D1 database (local dev)**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler d1 create pgolf-dashboard-db
```

Copy the returned `database_id` into `wrangler.jsonc` to replace `<will-be-set-after-creation>` for the D1 binding.

- [ ] **Step 3: Create the KV namespace (local dev)**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler kv namespace create KV
```

Copy the returned `id` into `wrangler.jsonc` to replace `<will-be-set-after-creation>` for the KV binding.

- [ ] **Step 4: Run the migration locally**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler d1 migrations apply pgolf-dashboard-db --local
```

Expected: Migration applied successfully.

- [ ] **Step 5: Verify tables exist**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler d1 execute pgolf-dashboard-db --local --command "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
```

Expected: Output lists `agent_status`, `budget_runs`, `budget_snapshot`, `experiments`, `research_items`.

- [ ] **Step 6: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/migrations/ dashboard/wrangler.jsonc
git commit -m "feat(dashboard): add D1 schema migration and KV namespace config"
```

---

### Task 4: TypeScript types

**Files:**
- Create: `dashboard/src/lib/types.ts`

- [ ] **Step 1: Write all shared types**

`dashboard/src/lib/types.ts`:
```ts
export interface Experiment {
	id: string;
	tier: 'local' | 'runpod';
	val_bpb: number | null;
	artifact_bytes: number | null;
	memory_gb: number | null;
	status: 'keep' | 'discard' | 'crash';
	promoted: boolean;
	cost_usd: number;
	description: string;
	source_item: string;
	created_at: string;
}

export interface ResearchItem {
	id: string;
	score: number;
	tier: 'A' | 'B' | 'C';
	bpb_impact: number;
	size_compat: number;
	time_compat: number;
	implement: number;
	novelty: number;
	summary: string;
	flags: string[];
	verified: boolean;
	graded_at: string;
	verified_at: string | null;
}

export interface BudgetRun {
	run_id: string;
	started_at: string;
	duration_s: number;
	cost_usd: number;
	val_bpb: number | null;
	artifact_bytes: number | null;
	promoted_from: string;
}

export interface BudgetSnapshot {
	total_credits: number;
	spent: number;
	min_reserve: number;
	updated_at: string;
}

export interface AgentStatus {
	agent: 'experiment' | 'research';
	status: 'running' | 'idle' | 'crashed';
	last_activity: string;
	restart_count: number;
}

export interface PipelineCounts {
	fetched: number;
	graded: number;
	verified: number;
	injected: number;
}

export interface TechniqueNode {
	id: string;
	label: string;
	status: 'proven' | 'exploring' | 'dead_end' | 'untried';
}

export interface TechniqueEdge {
	source: string;
	target: string;
}

export interface TechniqueMap {
	nodes: TechniqueNode[];
	edges: TechniqueEdge[];
}

export type IngestEvent =
	| { event: 'experiment_complete'; data: Experiment }
	| { event: 'research_graded'; data: ResearchItem[] }
	| { event: 'research_verified'; data: { id: string; verified_at: string } }
	| { event: 'budget_update'; data: { snapshot: BudgetSnapshot; run?: BudgetRun } }
	| { event: 'doc_update'; data: { key: string; content: string } }
	| {
			event: 'heartbeat';
			data: {
				agents: AgentStatus[];
				sota_bpb: number;
				pipeline_counts: PipelineCounts;
			};
	  };
```

- [ ] **Step 2: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/types.ts
git commit -m "feat(dashboard): add shared TypeScript types for all data models"
```

---

### Task 5: Ingest API endpoint

**Files:**
- Create: `dashboard/src/routes/api/ingest/+server.ts`
- Create: `dashboard/tests/ingest.test.ts`

- [ ] **Step 1: Write the ingest endpoint**

`dashboard/src/routes/api/ingest/+server.ts`:
```ts
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import type { IngestEvent } from '$lib/types';

export const POST: RequestHandler = async ({ request, platform }) => {
	const token = request.headers.get('authorization')?.replace('Bearer ', '');
	if (!token || token !== platform?.env.DASHBOARD_TOKEN) {
		return error(401, 'Unauthorized');
	}

	const body = (await request.json()) as IngestEvent;
	const db = platform!.env.DB;
	const kv = platform!.env.KV;

	switch (body.event) {
		case 'experiment_complete': {
			const e = body.data;
			await db
				.prepare(
					`INSERT OR REPLACE INTO experiments
					(id, tier, val_bpb, artifact_bytes, memory_gb, status, promoted, cost_usd, description, source_item, created_at)
					VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
				)
				.bind(
					e.id,
					e.tier,
					e.val_bpb,
					e.artifact_bytes,
					e.memory_gb,
					e.status,
					e.promoted ? 1 : 0,
					e.cost_usd,
					e.description,
					e.source_item,
					e.created_at
				)
				.run();
			break;
		}

		case 'research_graded': {
			const items = body.data;
			const stmt = db.prepare(
				`INSERT OR REPLACE INTO research_items
				(id, score, tier, bpb_impact, size_compat, time_compat, implement, novelty, summary, flags, verified, graded_at, verified_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
			);
			await db.batch(
				items.map((r) =>
					stmt.bind(
						r.id,
						r.score,
						r.tier,
						r.bpb_impact,
						r.size_compat,
						r.time_compat,
						r.implement,
						r.novelty,
						r.summary,
						JSON.stringify(r.flags),
						r.verified ? 1 : 0,
						r.graded_at,
						r.verified_at
					)
				)
			);
			break;
		}

		case 'research_verified': {
			const { id, verified_at } = body.data;
			await db
				.prepare('UPDATE research_items SET verified = 1, verified_at = ? WHERE id = ?')
				.bind(verified_at, id)
				.run();
			break;
		}

		case 'budget_update': {
			const { snapshot, run } = body.data;
			await db
				.prepare(
					'UPDATE budget_snapshot SET total_credits = ?, spent = ?, min_reserve = ?, updated_at = ? WHERE id = 1'
				)
				.bind(snapshot.total_credits, snapshot.spent, snapshot.min_reserve, snapshot.updated_at)
				.run();
			if (run) {
				await db
					.prepare(
						`INSERT OR REPLACE INTO budget_runs
						(run_id, started_at, duration_s, cost_usd, val_bpb, artifact_bytes, promoted_from)
						VALUES (?, ?, ?, ?, ?, ?, ?)`
					)
					.bind(
						run.run_id,
						run.started_at,
						run.duration_s,
						run.cost_usd,
						run.val_bpb,
						run.artifact_bytes,
						run.promoted_from
					)
					.run();
			}
			break;
		}

		case 'doc_update': {
			const { key, content } = body.data;
			const validKeys = ['program', 'strategy', 'technique_map'];
			if (!validKeys.includes(key)) {
				return error(400, `Invalid doc key: ${key}`);
			}
			await kv.put(`doc:${key}`, content);
			break;
		}

		case 'heartbeat': {
			const { agents, sota_bpb, pipeline_counts } = body.data;
			const stmt = db.prepare(
				'INSERT OR REPLACE INTO agent_status (agent, status, last_activity, restart_count) VALUES (?, ?, ?, ?)'
			);
			await db.batch(
				agents.map((a) => stmt.bind(a.agent, a.status, a.last_activity, a.restart_count))
			);
			await kv.put('meta:sota_bpb', String(sota_bpb));
			await kv.put('meta:pipeline_counts', JSON.stringify(pipeline_counts));
			await kv.put('meta:last_push', new Date().toISOString());
			break;
		}

		default:
			return error(400, `Unknown event type`);
	}

	return json({ ok: true });
};
```

- [ ] **Step 2: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/routes/api/ingest/
git commit -m "feat(dashboard): add POST /api/ingest endpoint with event routing"
```

---

### Task 6: D1 query helpers

**Files:**
- Create: `dashboard/src/lib/server/db.ts`

- [ ] **Step 1: Write the query helpers**

`dashboard/src/lib/server/db.ts`:
```ts
import type {
	Experiment,
	ResearchItem,
	BudgetRun,
	BudgetSnapshot,
	AgentStatus
} from '$lib/types';
import type { D1Database } from '@cloudflare/workers-types';

function boolFromInt(val: number): boolean {
	return val === 1;
}

function parseExperiment(row: Record<string, unknown>): Experiment {
	return {
		...(row as unknown as Experiment),
		promoted: boolFromInt(row.promoted as number)
	};
}

function parseResearchItem(row: Record<string, unknown>): ResearchItem {
	return {
		...(row as unknown as ResearchItem),
		flags: JSON.parse(row.flags as string),
		verified: boolFromInt(row.verified as number)
	};
}

export async function getRecentExperiments(db: D1Database, limit = 10): Promise<Experiment[]> {
	const { results } = await db
		.prepare('SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?')
		.bind(limit)
		.all();
	return results.map(parseExperiment);
}

export async function getExperimentsPaginated(
	db: D1Database,
	page: number,
	perPage: number,
	tier?: string,
	status?: string
): Promise<{ items: Experiment[]; total: number }> {
	let where = 'WHERE 1=1';
	const binds: unknown[] = [];
	if (tier && tier !== 'all') {
		where += ' AND tier = ?';
		binds.push(tier);
	}
	if (status && status !== 'all') {
		where += ' AND status = ?';
		binds.push(status);
	}

	const countResult = await db
		.prepare(`SELECT COUNT(*) as count FROM experiments ${where}`)
		.bind(...binds)
		.first<{ count: number }>();

	const offset = (page - 1) * perPage;
	const { results } = await db
		.prepare(`SELECT * FROM experiments ${where} ORDER BY created_at DESC LIMIT ? OFFSET ?`)
		.bind(...binds, perPage, offset)
		.all();

	return { items: results.map(parseExperiment), total: countResult?.count ?? 0 };
}

export async function getBestBpbPerDay(
	db: D1Database
): Promise<{ date: string; best_bpb: number; tier: string }[]> {
	const { results } = await db
		.prepare(
			`SELECT
				DATE(created_at) as date,
				MIN(val_bpb) as best_bpb,
				tier
			FROM experiments
			WHERE val_bpb IS NOT NULL AND status = 'keep'
			GROUP BY DATE(created_at), tier
			ORDER BY date ASC`
		)
		.all();
	return results as { date: string; best_bpb: number; tier: string }[];
}

export async function getRecentResearch(db: D1Database, limit = 10): Promise<ResearchItem[]> {
	const { results } = await db
		.prepare('SELECT * FROM research_items ORDER BY graded_at DESC LIMIT ?')
		.bind(limit)
		.all();
	return results.map(parseResearchItem);
}

export async function getResearchPaginated(
	db: D1Database,
	page: number,
	perPage: number,
	tier?: string
): Promise<{ items: ResearchItem[]; total: number }> {
	let where = 'WHERE 1=1';
	const binds: unknown[] = [];
	if (tier && tier !== 'all') {
		where += ' AND tier = ?';
		binds.push(tier);
	}

	const countResult = await db
		.prepare(`SELECT COUNT(*) as count FROM research_items ${where}`)
		.bind(...binds)
		.first<{ count: number }>();

	const offset = (page - 1) * perPage;
	const { results } = await db
		.prepare(`SELECT * FROM research_items ${where} ORDER BY score DESC LIMIT ? OFFSET ?`)
		.bind(...binds, perPage, offset)
		.all();

	return { items: results.map(parseResearchItem), total: countResult?.count ?? 0 };
}

export async function getResearchTierCounts(
	db: D1Database
): Promise<{ tier: string; count: number }[]> {
	const { results } = await db
		.prepare('SELECT tier, COUNT(*) as count FROM research_items GROUP BY tier')
		.all();
	return results as { tier: string; count: number }[];
}

export async function getBudgetSnapshot(db: D1Database): Promise<BudgetSnapshot> {
	const row = await db.prepare('SELECT * FROM budget_snapshot WHERE id = 1').first();
	return row as unknown as BudgetSnapshot;
}

export async function getBudgetRuns(db: D1Database): Promise<BudgetRun[]> {
	const { results } = await db
		.prepare('SELECT * FROM budget_runs ORDER BY started_at DESC')
		.all();
	return results as unknown as BudgetRun[];
}

export async function getBurnRate(db: D1Database): Promise<number> {
	const row = await db
		.prepare(
			`SELECT COALESCE(SUM(cost_usd), 0) as total_cost
			FROM budget_runs
			WHERE started_at >= datetime('now', '-7 days')`
		)
		.first<{ total_cost: number }>();
	return (row?.total_cost ?? 0) / 7;
}

export async function getAgentStatuses(db: D1Database): Promise<AgentStatus[]> {
	const { results } = await db.prepare('SELECT * FROM agent_status').all();
	return results as unknown as AgentStatus[];
}

export async function getPromotionDates(db: D1Database): Promise<{ date: string }[]> {
	const { results } = await db
		.prepare(
			"SELECT DATE(created_at) as date FROM experiments WHERE promoted = 1 GROUP BY DATE(created_at) ORDER BY date ASC"
		)
		.all();
	return results as { date: string }[];
}

export async function getMaxArtifactBytes(db: D1Database): Promise<number | null> {
	const row = await db
		.prepare(
			"SELECT MAX(artifact_bytes) as max_bytes FROM experiments WHERE status = 'keep' AND artifact_bytes IS NOT NULL"
		)
		.first<{ max_bytes: number | null }>();
	return row?.max_bytes ?? null;
}

export async function getBestBpb(db: D1Database): Promise<{ local: number | null; runpod: number | null }> {
	const local = await db
		.prepare(
			"SELECT MIN(val_bpb) as best FROM experiments WHERE tier = 'local' AND status = 'keep' AND val_bpb IS NOT NULL"
		)
		.first<{ best: number | null }>();
	const runpod = await db
		.prepare(
			"SELECT MIN(val_bpb) as best FROM experiments WHERE tier = 'runpod' AND status = 'keep' AND val_bpb IS NOT NULL"
		)
		.first<{ best: number | null }>();
	return { local: local?.best ?? null, runpod: runpod?.best ?? null };
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/server/db.ts
git commit -m "feat(dashboard): add D1 query helpers for all tables"
```

---

### Task 7: KV read helpers

**Files:**
- Create: `dashboard/src/lib/server/kv.ts`

- [ ] **Step 1: Write the KV helpers**

`dashboard/src/lib/server/kv.ts`:
```ts
import type { KVNamespace } from '@cloudflare/workers-types';
import type { TechniqueMap, PipelineCounts } from '$lib/types';

export async function getDoc(kv: KVNamespace, key: string): Promise<string> {
	return (await kv.get(`doc:${key}`)) ?? '';
}

export async function getTechniqueMap(kv: KVNamespace): Promise<TechniqueMap> {
	const raw = await kv.get('doc:technique_map');
	if (!raw) return { nodes: [], edges: [] };
	return JSON.parse(raw) as TechniqueMap;
}

export async function getLastPush(kv: KVNamespace): Promise<string> {
	return (await kv.get('meta:last_push')) ?? '';
}

export async function getSotaBpb(kv: KVNamespace): Promise<number | null> {
	const raw = await kv.get('meta:sota_bpb');
	if (!raw) return null;
	return parseFloat(raw);
}

export async function getPipelineCounts(kv: KVNamespace): Promise<PipelineCounts> {
	const raw = await kv.get('meta:pipeline_counts');
	if (!raw) return { fetched: 0, graded: 0, verified: 0, injected: 0 };
	return JSON.parse(raw) as PipelineCounts;
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/server/kv.ts
git commit -m "feat(dashboard): add KV read helpers for docs and metadata"
```

---

### Task 8: Layout + Nav component

**Files:**
- Create: `dashboard/src/lib/components/Nav.svelte`
- Create: `dashboard/src/routes/+layout.svelte`
- Create: `dashboard/src/routes/+layout.server.ts`

- [ ] **Step 1: Write the Nav component**

`dashboard/src/lib/components/Nav.svelte`:
```svelte
<script lang="ts">
	import { page } from '$app/state';
	import { fly } from 'svelte/transition';

	const links = [
		{ href: '/', label: 'Overview' },
		{ href: '/experiments', label: 'Experiments' },
		{ href: '/research', label: 'Research' },
		{ href: '/budget', label: 'Budget' },
		{ href: '/strategy', label: 'Strategy' }
	];

	function isActive(href: string): boolean {
		if (href === '/') return page.url.pathname === '/';
		return page.url.pathname.startsWith(href);
	}
</script>

<nav
	class="border-b border-edge px-4 sm:px-8 py-4 flex items-baseline justify-between"
	in:fly={{ y: -12, duration: 400 }}
>
	<a href="/" class="text-on-surface no-underline">
		<span class="text-lg font-semibold leading-none tracking-tighter">Parameter Golf</span>
	</a>
	<div class="flex gap-4">
		{#each links as { href, label }}
			<a
				{href}
				class="text-sm text-muted no-underline transition-opacity duration-150"
				class:opacity-40={!isActive(href)}
				class:text-on-surface={isActive(href)}
			>
				{label}
			</a>
		{/each}
	</div>
</nav>
```

- [ ] **Step 2: Write the layout server loader**

`dashboard/src/routes/+layout.server.ts`:
```ts
import type { LayoutServerLoad } from './$types';
import { getAgentStatuses } from '$lib/server/db';
import { getLastPush } from '$lib/server/kv';

export const load: LayoutServerLoad = async ({ platform }) => {
	const db = platform!.env.DB;
	const kv = platform!.env.KV;

	const [agents, lastPush] = await Promise.all([getAgentStatuses(db), getLastPush(kv)]);

	return { agents, lastPush };
};
```

- [ ] **Step 3: Write the layout component**

`dashboard/src/routes/+layout.svelte`:
```svelte
<script lang="ts">
	import '$lib/styles/app.css';
	import Nav from '$lib/components/Nav.svelte';
	import { fly } from 'svelte/transition';
	import type { Snippet } from 'svelte';

	let { data, children }: { data: any; children: Snippet } = $props();
</script>

<div class="min-h-screen flex flex-col bg-surface text-on-surface">
	<Nav />

	<main class="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-8 py-6 sm:py-10">
		<div in:fly={{ y: 24, duration: 500 }}>
			{@render children()}
		</div>
	</main>

	<footer class="border-t border-edge px-4 sm:px-8 py-3 flex justify-between text-xs text-muted">
		<span>Parameter Golf Dashboard</span>
		{#if data.lastPush}
			<span>Last updated: {new Date(data.lastPush).toLocaleString()}</span>
		{:else}
			<span>No data yet</span>
		{/if}
	</footer>
</div>
```

- [ ] **Step 4: Verify the layout renders**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npm run dev
```

Open `http://localhost:5173` in a browser. Expected: See the nav bar with "Parameter Golf" title and the five nav links. The footer shows "No data yet". The page is styled with the design system colors.

Note: D1/KV bindings won't work in plain `npm run dev`. The layout loader may fail. To test with bindings, use:
```bash
npx wrangler dev
```

- [ ] **Step 5: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/components/Nav.svelte dashboard/src/routes/+layout.svelte dashboard/src/routes/+layout.server.ts
git commit -m "feat(dashboard): add layout with nav, footer, and page transitions"
```

---

### Task 9: MetricCard and DataTable components

**Files:**
- Create: `dashboard/src/lib/components/MetricCard.svelte`
- Create: `dashboard/src/lib/components/DataTable.svelte`

- [ ] **Step 1: Write the MetricCard component**

`dashboard/src/lib/components/MetricCard.svelte`:
```svelte
<script lang="ts">
	let { label, value, sub = '' }: { label: string; value: string; sub?: string } = $props();
</script>

<div class="border border-edge p-4">
	<div class="text-xs font-medium tracking-wider text-muted">{label}</div>
	<div class="text-2xl font-semibold mt-1">{value}</div>
	{#if sub}
		<div class="text-xs text-muted mt-1">{sub}</div>
	{/if}
</div>
```

- [ ] **Step 2: Write the DataTable component**

`dashboard/src/lib/components/DataTable.svelte`:
```svelte
<script lang="ts" generics="T extends Record<string, unknown>">
	import type { Snippet } from 'svelte';

	interface Column {
		key: string;
		label: string;
		format?: (val: unknown, row: T) => string;
	}

	let {
		data,
		columns,
		total = data.length,
		page = 1,
		perPage = 50,
		onPageChange,
		expandable = false,
		expandRow
	}: {
		data: T[];
		columns: Column[];
		total?: number;
		page?: number;
		perPage?: number;
		onPageChange?: (page: number) => void;
		expandable?: boolean;
		expandRow?: Snippet<[T]>;
	} = $props();

	let expandedId = $state<string | null>(null);
	let sortKey = $state('');
	let sortAsc = $state(true);

	let sorted = $derived.by(() => {
		if (!sortKey) return data;
		return [...data].sort((a, b) => {
			const av = a[sortKey];
			const bv = b[sortKey];
			if (av == null) return 1;
			if (bv == null) return -1;
			if (av < bv) return sortAsc ? -1 : 1;
			if (av > bv) return sortAsc ? 1 : -1;
			return 0;
		});
	});

	let totalPages = $derived(Math.ceil(total / perPage));

	function toggleSort(key: string) {
		if (sortKey === key) {
			sortAsc = !sortAsc;
		} else {
			sortKey = key;
			sortAsc = true;
		}
	}

	function toggleExpand(id: string) {
		expandedId = expandedId === id ? null : id;
	}
</script>

<div class="overflow-x-auto">
	<table class="w-full text-sm">
		<thead>
			<tr class="border-b border-edge">
				{#each columns as col}
					<th
						class="text-left py-2 pr-4 text-xs font-medium tracking-wider text-muted cursor-pointer transition-opacity duration-150 hover:opacity-70"
						onclick={() => toggleSort(col.key)}
					>
						{col.label}
						{#if sortKey === col.key}
							<span class="ml-1">{sortAsc ? '\u2191' : '\u2193'}</span>
						{/if}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each sorted as row, i (row.id ?? i)}
				<tr
					class="border-b border-edge transition-opacity duration-150 hover:opacity-70"
					class:cursor-pointer={expandable}
					onclick={() => expandable && toggleExpand(String(row.id ?? i))}
				>
					{#each columns as col}
						<td class="py-2 pr-4">
							{col.format ? col.format(row[col.key], row) : (row[col.key] ?? '')}
						</td>
					{/each}
				</tr>
				{#if expandable && expandedId === String(row.id ?? i) && expandRow}
					<tr class="border-b border-edge">
						<td colspan={columns.length} class="py-3 px-4 bg-edge/10">
							{@render expandRow(row)}
						</td>
					</tr>
				{/if}
			{/each}
		</tbody>
	</table>
</div>

{#if totalPages > 1}
	<div class="flex items-center gap-2 mt-4 text-sm text-muted">
		<button
			class="border border-edge bg-transparent px-3 py-1 text-sm text-on-surface transition-colors hover:bg-edge disabled:opacity-50"
			disabled={page <= 1}
			onclick={() => onPageChange?.(page - 1)}
		>
			Prev
		</button>
		<span>{page} / {totalPages}</span>
		<button
			class="border border-edge bg-transparent px-3 py-1 text-sm text-on-surface transition-colors hover:bg-edge disabled:opacity-50"
			disabled={page >= totalPages}
			onclick={() => onPageChange?.(page + 1)}
		>
			Next
		</button>
	</div>
{/if}
```

- [ ] **Step 3: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/components/MetricCard.svelte dashboard/src/lib/components/DataTable.svelte
git commit -m "feat(dashboard): add MetricCard and DataTable reusable components"
```

---

### Task 10: Overview page

**Files:**
- Create: `dashboard/src/routes/+page.server.ts`
- Create: `dashboard/src/routes/+page.svelte`

- [ ] **Step 1: Write the overview data loader**

`dashboard/src/routes/+page.server.ts`:
```ts
import type { PageServerLoad } from './$types';
import { getRecentExperiments, getRecentResearch, getBudgetSnapshot, getBestBpb, getMaxArtifactBytes } from '$lib/server/db';
import { getSotaBpb } from '$lib/server/kv';

export const load: PageServerLoad = async ({ platform }) => {
	const db = platform!.env.DB;
	const kv = platform!.env.KV;

	const [experiments, research, budget, bestBpb, sotaBpb, maxArtifactBytes] = await Promise.all([
		getRecentExperiments(db, 10),
		getRecentResearch(db, 10),
		getBudgetSnapshot(db),
		getBestBpb(db),
		getSotaBpb(kv),
		getMaxArtifactBytes(db)
	]);

	return { experiments, research, budget, bestBpb, sotaBpb, maxArtifactBytes };
};
```

- [ ] **Step 2: Write the overview page**

`dashboard/src/routes/+page.svelte`:
```svelte
<script lang="ts">
	import MetricCard from '$lib/components/MetricCard.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import type { Experiment, ResearchItem } from '$lib/types';

	let { data } = $props();

	let bestOverall = $derived(
		data.bestBpb.runpod ?? data.bestBpb.local
			? Math.min(
					...[data.bestBpb.local, data.bestBpb.runpod].filter((v): v is number => v !== null)
				)
			: null
	);

	let distanceToSota = $derived(
		bestOverall !== null && data.sotaBpb !== null
			? (bestOverall - data.sotaBpb).toFixed(4)
			: '—'
	);

	let budgetRemaining = $derived(data.budget.total_credits - data.budget.spent);

	const MAX_ARTIFACT = 16_000_000;
	let artifactHeadroom = $derived(
		data.maxArtifactBytes !== null
			? ((MAX_ARTIFACT - data.maxArtifactBytes) / 1e6).toFixed(1)
			: null
	);

	const experimentCols = [
		{ key: 'description', label: 'Description' },
		{ key: 'tier', label: 'Tier' },
		{
			key: 'val_bpb',
			label: 'Val bpb',
			format: (v: unknown) => (v != null ? Number(v).toFixed(4) : '—')
		},
		{ key: 'status', label: 'Status' }
	];

	const researchCols = [
		{ key: 'summary', label: 'Summary' },
		{
			key: 'score',
			label: 'Score',
			format: (v: unknown) => Number(v).toFixed(1)
		},
		{ key: 'tier', label: 'Tier' }
	];
</script>

<div class="space-y-8">
	<!-- Agent status -->
	<div class="flex gap-4 text-xs text-muted">
		{#each data.agents as agent}
			<span>
				{agent.agent}:
				<span
					class:text-on-surface={agent.status === 'running'}
					class:opacity-40={agent.status === 'idle'}
				>
					{agent.status}
				</span>
			</span>
		{/each}
	</div>

	<!-- Key metrics -->
	<div class="grid grid-cols-2 sm:grid-cols-5 gap-4">
		<MetricCard
			label="Best bpb"
			value={bestOverall !== null ? bestOverall.toFixed(4) : '—'}
			sub="local: {data.bestBpb.local?.toFixed(4) ?? '—'} / runpod: {data.bestBpb.runpod?.toFixed(4) ?? '—'}"
		/>
		<MetricCard label="Distance to sota" value={distanceToSota} />
		<MetricCard
			label="Artifact headroom"
			value={artifactHeadroom !== null ? `${artifactHeadroom}MB` : '—'}
			sub="of 16MB limit"
		/>
		<MetricCard
			label="Budget remaining"
			value="${budgetRemaining.toFixed(2)}"
			sub="of ${data.budget.total_credits.toFixed(2)}"
		/>
		<MetricCard
			label="Spent"
			value="${data.budget.spent.toFixed(2)}"
			sub="reserve: ${data.budget.min_reserve.toFixed(2)}"
		/>
	</div>

	<!-- Recent experiments -->
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Recent experiments</h2>
		<DataTable data={data.experiments} columns={experimentCols} />
	</section>

	<!-- Recent research -->
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Recent research</h2>
		<DataTable data={data.research} columns={researchCols} />
	</section>
</div>
```

- [ ] **Step 3: Verify the overview page renders**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler dev
```

Open `http://localhost:8787`. Expected: Overview page with empty metric cards (showing "—" for bpb values, $0.00 for budget), empty tables, agent status showing "idle".

- [ ] **Step 4: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/routes/+page.svelte dashboard/src/routes/+page.server.ts
git commit -m "feat(dashboard): add overview page with metrics, recent experiments, recent research"
```

---

### Task 11: BpbChart + Experiments page

**Files:**
- Create: `dashboard/src/lib/components/BpbChart.svelte`
- Create: `dashboard/src/routes/experiments/+page.server.ts`
- Create: `dashboard/src/routes/experiments/+page.svelte`

- [ ] **Step 1: Write the BpbChart component**

`dashboard/src/lib/components/BpbChart.svelte`:
```svelte
<script lang="ts">
	import { LayerCake, Svg } from 'layercake';
	import { line as d3line } from 'd3-shape';
	import { getContext } from 'svelte';

	let {
		data,
		promotions = []
	}: {
		data: { date: string; best_bpb: number; tier: string }[];
		promotions?: { date: string }[];
	} = $props();

	let chartData = $derived(
		data.map((d) => ({
			x: new Date(d.date).getTime(),
			y: d.best_bpb,
			tier: d.tier
		}))
	);
</script>

{#if chartData.length > 0}
	<div class="h-64 w-full">
		<LayerCake
			padding={{ top: 10, right: 10, bottom: 30, left: 50 }}
			x="x"
			y="y"
			data={chartData}
		>
			<Svg>
				{@render chartContent()}
			</Svg>
		</LayerCake>
	</div>
{:else}
	<div class="h-64 w-full flex items-center justify-center text-muted text-sm">
		No experiment data yet
	</div>
{/if}

{#snippet chartContent()}
	{@const { xScale, yScale, width, height } = getContext('LayerCake')}
	<!-- Grid lines -->
	{#each $yScale.ticks(5) as tick}
		<line
			x1={0}
			x2={$width}
			y1={$yScale(tick)}
			y2={$yScale(tick)}
			stroke="var(--edge)"
			stroke-width="1"
			stroke-dasharray="2"
		/>
		<text x={-8} y={$yScale(tick)} dy="4" text-anchor="end" fill="var(--muted)" font-size="11">
			{tick.toFixed(3)}
		</text>
	{/each}

	<!-- X axis labels -->
	{#each $xScale.ticks(6) as tick}
		<text
			x={$xScale(tick)}
			y={$height + 20}
			text-anchor="middle"
			fill="var(--muted)"
			font-size="11"
		>
			{new Date(tick).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
		</text>
	{/each}

	<!-- Promotion vertical lines -->
	{#each promotions as promo}
		{@const px = $xScale(new Date(promo.date).getTime())}
		<line
			x1={px}
			x2={px}
			y1={0}
			y2={$height}
			stroke="var(--edge)"
			stroke-width="1"
			stroke-dasharray="4,2"
		/>
	{/each}

	<!-- Line -->
	{@const pathGen = d3line()
		.x((d) => $xScale(d.x))
		.y((d) => $yScale(d.y))}
	<path d={pathGen(chartData)} fill="none" stroke="var(--on-surface)" stroke-width="1.5" />

	<!-- Dots -->
	{#each chartData as point}
		<circle
			cx={$xScale(point.x)}
			cy={$yScale(point.y)}
			r="3"
			fill={point.tier === 'runpod' ? 'var(--on-surface)' : 'var(--muted)'}
		/>
	{/each}
{/snippet}
```

- [ ] **Step 2: Write the experiments data loader**

`dashboard/src/routes/experiments/+page.server.ts`:
```ts
import type { PageServerLoad } from './$types';
import { getExperimentsPaginated, getBestBpbPerDay, getPromotionDates } from '$lib/server/db';

export const load: PageServerLoad = async ({ platform, url }) => {
	const db = platform!.env.DB;
	const page = Number(url.searchParams.get('page') ?? '1');
	const tier = url.searchParams.get('tier') ?? 'all';
	const status = url.searchParams.get('status') ?? 'all';

	const [experiments, chartData, promotions] = await Promise.all([
		getExperimentsPaginated(db, page, 50, tier, status),
		getBestBpbPerDay(db),
		getPromotionDates(db)
	]);

	return { ...experiments, chartData, promotions, page, tier, status };
};
```

- [ ] **Step 3: Write the experiments page**

`dashboard/src/routes/experiments/+page.svelte`:
```svelte
<script lang="ts">
	import BpbChart from '$lib/components/BpbChart.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import { goto } from '$app/navigation';

	let { data } = $props();

	function updateFilter(key: string, value: string) {
		const url = new URL(window.location.href);
		url.searchParams.set(key, value);
		url.searchParams.set('page', '1');
		goto(url.toString(), { replaceState: true });
	}

	function changePage(p: number) {
		const url = new URL(window.location.href);
		url.searchParams.set('page', String(p));
		goto(url.toString(), { replaceState: true });
	}

	const columns = [
		{ key: 'description', label: 'Description' },
		{ key: 'tier', label: 'Tier' },
		{
			key: 'val_bpb',
			label: 'Val bpb',
			format: (v: unknown) => (v != null ? Number(v).toFixed(4) : '—')
		},
		{
			key: 'artifact_bytes',
			label: 'Size',
			format: (v: unknown) => (v != null ? `${(Number(v) / 1e6).toFixed(1)}MB` : '—')
		},
		{ key: 'status', label: 'Status' },
		{
			key: 'cost_usd',
			label: 'Cost',
			format: (v: unknown) => (Number(v) > 0 ? `$${Number(v).toFixed(2)}` : '—')
		},
		{ key: 'source_item', label: 'Source' },
		{
			key: 'created_at',
			label: 'Date',
			format: (v: unknown) => (v ? new Date(String(v)).toLocaleDateString() : '—')
		}
	];
</script>

<div class="space-y-8">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Val bpb over time</h2>
		<BpbChart data={data.chartData} promotions={data.promotions} />
	</section>

	<section>
		<div class="flex gap-4 mb-4">
			<select
				class="border border-edge bg-transparent px-3 py-2 text-sm text-on-surface focus:border-on-surface focus:outline-none"
				value={data.tier}
				onchange={(e) => updateFilter('tier', e.currentTarget.value)}
			>
				<option value="all">All tiers</option>
				<option value="local">Local</option>
				<option value="runpod">Runpod</option>
			</select>
			<select
				class="border border-edge bg-transparent px-3 py-2 text-sm text-on-surface focus:border-on-surface focus:outline-none"
				value={data.status}
				onchange={(e) => updateFilter('status', e.currentTarget.value)}
			>
				<option value="all">All statuses</option>
				<option value="keep">Keep</option>
				<option value="discard">Discard</option>
				<option value="crash">Crash</option>
			</select>
		</div>

		<DataTable
			data={data.items}
			{columns}
			total={data.total}
			page={data.page}
			perPage={50}
			onPageChange={changePage}
		/>
	</section>
</div>
```

- [ ] **Step 4: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/components/BpbChart.svelte dashboard/src/routes/experiments/
git commit -m "feat(dashboard): add experiments page with bpb chart, filters, and paginated table"
```

---

### Task 12: FunnelBar + Research page

**Files:**
- Create: `dashboard/src/lib/components/FunnelBar.svelte`
- Create: `dashboard/src/routes/research/+page.server.ts`
- Create: `dashboard/src/routes/research/+page.svelte`

- [ ] **Step 1: Write the FunnelBar component**

`dashboard/src/lib/components/FunnelBar.svelte`:
```svelte
<script lang="ts">
	import type { PipelineCounts } from '$lib/types';

	let { counts }: { counts: PipelineCounts } = $props();

	let max = $derived(Math.max(counts.fetched, counts.graded, counts.verified, counts.injected, 1));

	const stages: { key: keyof PipelineCounts; label: string }[] = [
		{ key: 'fetched', label: 'Fetched' },
		{ key: 'graded', label: 'Graded' },
		{ key: 'verified', label: 'Verified' },
		{ key: 'injected', label: 'Injected' }
	];
</script>

<div class="space-y-2">
	{#each stages as { key, label }}
		{@const pct = (counts[key] / max) * 100}
		<div class="flex items-center gap-3">
			<span class="text-xs text-muted w-16 text-right">{label}</span>
			<div class="flex-1 border border-edge h-6 relative">
				<div
					class="h-full bg-on-surface transition-all duration-150"
					style="width: {pct}%; opacity: 0.15"
				></div>
				<span class="absolute inset-0 flex items-center px-2 text-xs">{counts[key]}</span>
			</div>
		</div>
	{/each}
</div>
```

- [ ] **Step 2: Write the research data loader**

`dashboard/src/routes/research/+page.server.ts`:
```ts
import type { PageServerLoad } from './$types';
import { getResearchPaginated, getResearchTierCounts } from '$lib/server/db';
import { getPipelineCounts } from '$lib/server/kv';

export const load: PageServerLoad = async ({ platform, url }) => {
	const db = platform!.env.DB;
	const kv = platform!.env.KV;
	const page = Number(url.searchParams.get('page') ?? '1');
	const tier = url.searchParams.get('tier') ?? 'all';

	const [research, tierCounts, pipelineCounts] = await Promise.all([
		getResearchPaginated(db, page, 50, tier),
		getResearchTierCounts(db),
		getPipelineCounts(kv)
	]);

	return { ...research, tierCounts, pipelineCounts, page, tier };
};
```

- [ ] **Step 3: Write the research page**

`dashboard/src/routes/research/+page.svelte`:
```svelte
<script lang="ts">
	import FunnelBar from '$lib/components/FunnelBar.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import { goto } from '$app/navigation';
	import type { ResearchItem } from '$lib/types';

	let { data } = $props();

	function updateFilter(key: string, value: string) {
		const url = new URL(window.location.href);
		url.searchParams.set(key, value);
		url.searchParams.set('page', '1');
		goto(url.toString(), { replaceState: true });
	}

	function changePage(p: number) {
		const url = new URL(window.location.href);
		url.searchParams.set('page', String(p));
		goto(url.toString(), { replaceState: true });
	}

	const columns = [
		{ key: 'summary', label: 'Summary' },
		{
			key: 'score',
			label: 'Score',
			format: (v: unknown) => Number(v).toFixed(1)
		},
		{ key: 'tier', label: 'Tier' },
		{
			key: 'verified',
			label: 'Verified',
			format: (v: unknown) => (v ? 'Yes' : 'No')
		},
		{
			key: 'graded_at',
			label: 'Graded',
			format: (v: unknown) => (v ? new Date(String(v)).toLocaleDateString() : '—')
		}
	];
</script>

<div class="space-y-8">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Research pipeline</h2>
		<FunnelBar counts={data.pipelineCounts} />
	</section>

	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Tier breakdown</h2>
		<div class="flex gap-6 text-sm">
			{#each data.tierCounts as { tier, count }}
				<span>
					<span class="text-muted">Tier {tier}:</span>
					{count}
				</span>
			{/each}
		</div>
	</section>

	<section>
		<div class="flex gap-4 mb-4">
			<select
				class="border border-edge bg-transparent px-3 py-2 text-sm text-on-surface focus:border-on-surface focus:outline-none"
				value={data.tier}
				onchange={(e) => updateFilter('tier', e.currentTarget.value)}
			>
				<option value="all">All tiers</option>
				<option value="A">Tier A</option>
				<option value="B">Tier B</option>
				<option value="C">Tier C</option>
			</select>
		</div>

		<DataTable
			data={data.items}
			{columns}
			total={data.total}
			page={data.page}
			perPage={50}
			onPageChange={changePage}
			expandable={true}
		>
			{#snippet expandRow(item: ResearchItem)}
				<div class="grid grid-cols-5 gap-4 text-xs">
					<div><span class="text-muted">Bpb impact:</span> {item.bpb_impact}</div>
					<div><span class="text-muted">Size compat:</span> {item.size_compat}</div>
					<div><span class="text-muted">Time compat:</span> {item.time_compat}</div>
					<div><span class="text-muted">Implement:</span> {item.implement}</div>
					<div><span class="text-muted">Novelty:</span> {item.novelty}</div>
				</div>
				{#if item.flags.length > 0}
					<div class="mt-2 text-xs text-muted">
						Flags: {item.flags.join(', ')}
					</div>
				{/if}
			{/snippet}
		</DataTable>
	</section>
</div>
```

- [ ] **Step 4: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/components/FunnelBar.svelte dashboard/src/routes/research/
git commit -m "feat(dashboard): add research page with pipeline funnel, tier breakdown, and expandable table"
```

---

### Task 13: BudgetBar + Budget page

**Files:**
- Create: `dashboard/src/lib/components/BudgetBar.svelte`
- Create: `dashboard/src/routes/budget/+page.server.ts`
- Create: `dashboard/src/routes/budget/+page.svelte`

- [ ] **Step 1: Write the BudgetBar component**

`dashboard/src/lib/components/BudgetBar.svelte`:
```svelte
<script lang="ts">
	import type { BudgetSnapshot } from '$lib/types';

	let { snapshot }: { snapshot: BudgetSnapshot } = $props();

	let remaining = $derived(snapshot.total_credits - snapshot.spent - snapshot.min_reserve);
	let spentPct = $derived((snapshot.spent / snapshot.total_credits) * 100);
	let availablePct = $derived((remaining / snapshot.total_credits) * 100);
	let reservePct = $derived((snapshot.min_reserve / snapshot.total_credits) * 100);
</script>

<div>
	<div class="border border-edge h-8 flex">
		<div
			class="h-full bg-on-surface"
			style="width: {spentPct}%; opacity: 0.3"
			title="Spent: ${snapshot.spent.toFixed(2)}"
		></div>
		<div
			class="h-full bg-on-surface"
			style="width: {availablePct}%; opacity: 0.1"
			title="Available: ${remaining.toFixed(2)}"
		></div>
		<div
			class="h-full border-l border-edge bg-on-surface"
			style="width: {reservePct}%; opacity: 0.05"
			title="Reserve: ${snapshot.min_reserve.toFixed(2)}"
		></div>
	</div>
	<div class="flex justify-between text-xs text-muted mt-1">
		<span>Spent: ${snapshot.spent.toFixed(2)}</span>
		<span>Available: ${remaining.toFixed(2)}</span>
		<span>Reserve: ${snapshot.min_reserve.toFixed(2)}</span>
	</div>
</div>
```

- [ ] **Step 2: Write the budget data loader**

`dashboard/src/routes/budget/+page.server.ts`:
```ts
import type { PageServerLoad } from './$types';
import { getBudgetSnapshot, getBudgetRuns, getBurnRate } from '$lib/server/db';

export const load: PageServerLoad = async ({ platform }) => {
	const db = platform!.env.DB;

	const [snapshot, runs, burnRate] = await Promise.all([
		getBudgetSnapshot(db),
		getBudgetRuns(db),
		getBurnRate(db)
	]);

	const remaining = snapshot.total_credits - snapshot.spent - snapshot.min_reserve;
	const daysRemaining = burnRate > 0 ? remaining / burnRate : null;

	return { snapshot, runs, burnRate, daysRemaining };
};
```

- [ ] **Step 3: Write the budget page**

`dashboard/src/routes/budget/+page.svelte`:
```svelte
<script lang="ts">
	import BudgetBar from '$lib/components/BudgetBar.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';

	let { data } = $props();

	const columns = [
		{
			key: 'started_at',
			label: 'Date',
			format: (v: unknown) => (v ? new Date(String(v)).toLocaleString() : '—')
		},
		{
			key: 'duration_s',
			label: 'Duration',
			format: (v: unknown) => `${Math.round(Number(v) / 60)}m`
		},
		{
			key: 'cost_usd',
			label: 'Cost',
			format: (v: unknown) => `$${Number(v).toFixed(2)}`
		},
		{
			key: 'val_bpb',
			label: 'Val bpb',
			format: (v: unknown) => (v != null ? Number(v).toFixed(4) : '—')
		},
		{ key: 'promoted_from', label: 'From experiment' }
	];
</script>

<div class="space-y-8">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Budget</h2>
		<BudgetBar snapshot={data.snapshot} />
	</section>

	<section>
		<div class="grid grid-cols-2 sm:grid-cols-3 gap-4">
			<MetricCard
				label="Burn rate"
				value="${data.burnRate.toFixed(2)}/day"
			/>
			<MetricCard
				label="Est. days remaining"
				value={data.daysRemaining !== null ? Math.round(data.daysRemaining).toString() : '—'}
			/>
			<MetricCard
				label="Total runs"
				value={data.runs.length.toString()}
			/>
		</div>
	</section>

	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Run history</h2>
		<DataTable data={data.runs} {columns} />
	</section>
</div>
```

- [ ] **Step 4: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/components/BudgetBar.svelte dashboard/src/routes/budget/
git commit -m "feat(dashboard): add budget page with spend bar, burn rate, and run history"
```

---

### Task 14: TechGraph + Strategy page

**Files:**
- Create: `dashboard/src/lib/components/TechGraph.svelte`
- Create: `dashboard/src/routes/strategy/+page.server.ts`
- Create: `dashboard/src/routes/strategy/+page.svelte`

- [ ] **Step 1: Write the TechGraph component**

`dashboard/src/lib/components/TechGraph.svelte`:
```svelte
<script lang="ts">
	import { onMount } from 'svelte';
	import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide } from 'd3-force';
	import type { TechniqueMap } from '$lib/types';

	let { map }: { map: TechniqueMap } = $props();

	interface SimNode {
		id: string;
		label: string;
		status: string;
		x: number;
		y: number;
	}

	interface SimLink {
		source: string | SimNode;
		target: string | SimNode;
	}

	let width = 800;
	let height = 500;
	let nodes = $state<SimNode[]>([]);
	let links = $state<SimLink[]>([]);
	let hoveredId = $state<string | null>(null);

	function statusOpacity(status: string): number {
		switch (status) {
			case 'proven':
				return 1;
			case 'exploring':
				return 0.7;
			case 'dead_end':
				return 0.4;
			case 'untried':
				return 0.7;
			default:
				return 0.5;
		}
	}

	function statusDash(status: string): string {
		return status === 'untried' ? '4,2' : 'none';
	}

	function isConnected(nodeId: string): boolean {
		if (!hoveredId) return true;
		if (nodeId === hoveredId) return true;
		return links.some((l) => {
			const src = typeof l.source === 'string' ? l.source : l.source.id;
			const tgt = typeof l.target === 'string' ? l.target : l.target.id;
			return (src === hoveredId && tgt === nodeId) || (tgt === hoveredId && src === nodeId);
		});
	}

	onMount(() => {
		if (map.nodes.length === 0) return;

		const simNodes: SimNode[] = map.nodes.map((n) => ({
			...n,
			x: width / 2 + Math.random() * 100 - 50,
			y: height / 2 + Math.random() * 100 - 50
		}));

		const simLinks: SimLink[] = map.edges.map((e) => ({
			source: e.source,
			target: e.target
		}));

		const simulation = forceSimulation(simNodes)
			.force(
				'link',
				forceLink(simLinks)
					.id((d: any) => d.id)
					.distance(80)
			)
			.force('charge', forceManyBody().strength(-200))
			.force('center', forceCenter(width / 2, height / 2))
			.force('collide', forceCollide(30));

		simulation.on('tick', () => {
			nodes = [...simNodes];
			links = [...simLinks];
		});

		return () => simulation.stop();
	});
</script>

{#if map.nodes.length > 0}
	<svg {width} {height} class="w-full" viewBox="0 0 {width} {height}">
		<!-- Edges -->
		{#each links as link}
			{@const src = typeof link.source === 'string' ? null : link.source}
			{@const tgt = typeof link.target === 'string' ? null : link.target}
			{#if src && tgt}
				<line
					x1={src.x}
					y1={src.y}
					x2={tgt.x}
					y2={tgt.y}
					stroke="var(--edge)"
					stroke-width="1"
					opacity={hoveredId ? (isConnected(src.id) && isConnected(tgt.id) ? 1 : 0.15) : 1}
				/>
			{/if}
		{/each}

		<!-- Nodes -->
		{#each nodes as node}
			<g
				transform="translate({node.x}, {node.y})"
				opacity={hoveredId ? (isConnected(node.id) ? statusOpacity(node.status) : 0.15) : statusOpacity(node.status)}
				class="transition-opacity duration-150 cursor-pointer"
				onmouseenter={() => (hoveredId = node.id)}
				onmouseleave={() => (hoveredId = null)}
			>
				<rect
					x="-30"
					y="-12"
					width="60"
					height="24"
					fill="var(--surface)"
					stroke="var(--on-surface)"
					stroke-width="1"
					stroke-dasharray={statusDash(node.status)}
				/>
				<text
					text-anchor="middle"
					dy="4"
					fill="var(--on-surface)"
					font-size="10"
					font-family="var(--font-sans)"
				>
					{node.label.length > 8 ? node.label.slice(0, 8) + '...' : node.label}
				</text>
			</g>
		{/each}
	</svg>
{:else}
	<div class="h-64 flex items-center justify-center text-muted text-sm">
		No technique data yet
	</div>
{/if}
```

- [ ] **Step 2: Write the strategy data loader**

`dashboard/src/routes/strategy/+page.server.ts`:
```ts
import type { PageServerLoad } from './$types';
import { getDoc, getTechniqueMap } from '$lib/server/kv';

export const load: PageServerLoad = async ({ platform }) => {
	const kv = platform!.env.KV;

	const [programMd, strategyMd, techniqueMap] = await Promise.all([
		getDoc(kv, 'program'),
		getDoc(kv, 'strategy'),
		getTechniqueMap(kv)
	]);

	return { programMd, strategyMd, techniqueMap };
};
```

- [ ] **Step 3: Write the strategy page**

`dashboard/src/routes/strategy/+page.svelte`:
```svelte
<script lang="ts">
	import TechGraph from '$lib/components/TechGraph.svelte';
	import { marked } from 'marked';

	let { data } = $props();

	let programHtml = $derived(data.programMd ? marked.parse(data.programMd) : '');
	let strategyHtml = $derived(data.strategyMd ? marked.parse(data.strategyMd) : '');
</script>

<div class="space-y-12">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Technique map</h2>
		<TechGraph map={data.techniqueMap} />
		<div class="flex gap-6 mt-3 text-xs text-muted">
			<span>Proven (100%)</span>
			<span class="opacity-70">Exploring (70%)</span>
			<span class="opacity-40">Dead end (40%)</span>
			<span class="opacity-70" style="text-decoration: underline dashed">Untried (dashed)</span>
		</div>
	</section>

	{#if data.programMd}
		<section>
			<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Program</h2>
			<article class="prose-content leading-relaxed">
				{@html programHtml}
			</article>
		</section>
	{/if}

	{#if data.strategyMd}
		<section>
			<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Strategy</h2>
			<article class="prose-content leading-relaxed">
				{@html strategyHtml}
			</article>
		</section>
	{/if}
</div>

<style>
	:global(.prose-content h1) {
		font-size: 1.75rem;
		margin-bottom: 0.75rem;
	}
	:global(.prose-content h2) {
		font-size: 1.375rem;
		margin-top: 2rem;
		margin-bottom: 0.75rem;
	}
	:global(.prose-content h3) {
		font-size: 1.125rem;
		margin-top: 2rem;
		margin-bottom: 0.75rem;
	}
	:global(.prose-content p) {
		margin-bottom: 1rem;
	}
	:global(.prose-content a) {
		color: var(--on-surface);
		text-decoration: underline;
		text-underline-offset: 3px;
		transition: opacity 150ms;
	}
	:global(.prose-content a:hover) {
		opacity: 0.7;
	}
	:global(.prose-content code) {
		font-family: var(--font-mono);
		font-size: 0.875em;
	}
	:global(.prose-content pre) {
		border: 1px solid var(--edge);
		padding: 1rem;
		overflow-x: auto;
		margin-bottom: 1rem;
	}
	:global(.prose-content ul, .prose-content ol) {
		padding-left: 1.5rem;
		margin-bottom: 1rem;
	}
</style>
```

- [ ] **Step 4: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add dashboard/src/lib/components/TechGraph.svelte dashboard/src/routes/strategy/
git commit -m "feat(dashboard): add strategy page with technique graph and rendered markdown docs"
```

---

### Task 15: Python DashboardPusher

**Files:**
- Create: `compute/dashboard.py`
- Create: `tests/test_dashboard_pusher.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dashboard_pusher.py`:
```python
import json
from unittest.mock import patch, MagicMock
import pytest


def test_pusher_noop_when_no_url():
    """DashboardPusher is a no-op when DASHBOARD_URL is not set."""
    with patch.dict("os.environ", {}, clear=True):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()
        # Should not raise, should do nothing
        pusher.push_experiment({"id": "abc", "tier": "local"})


def test_pusher_sends_experiment():
    """DashboardPusher sends experiment_complete event."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch.dict(
        "os.environ",
        {"DASHBOARD_URL": "https://example.com", "DASHBOARD_TOKEN": "tok123"},
    ):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            pusher.push_experiment({"id": "abc", "tier": "local", "val_bpb": 1.21})

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            body = json.loads(call_kwargs.kwargs.get("content", call_kwargs[1].get("content", "")))
            assert body["event"] == "experiment_complete"
            assert body["data"]["id"] == "abc"


def test_pusher_sends_heartbeat():
    """DashboardPusher sends heartbeat with agents, sota, and pipeline counts."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch.dict(
        "os.environ",
        {"DASHBOARD_URL": "https://example.com", "DASHBOARD_TOKEN": "tok123"},
    ):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            pusher.push_heartbeat(
                statuses=[{"agent": "experiment", "status": "running", "last_activity": "", "restart_count": 0}],
                sota_bpb=1.05,
                pipeline_counts={"fetched": 100, "graded": 50, "verified": 10, "injected": 5},
            )

            call_kwargs = mock_post.call_args
            body = json.loads(call_kwargs.kwargs.get("content", call_kwargs[1].get("content", "")))
            assert body["event"] == "heartbeat"
            assert body["data"]["sota_bpb"] == 1.05


def test_pusher_swallows_errors():
    """DashboardPusher never raises — errors are silently swallowed."""
    with patch.dict(
        "os.environ",
        {"DASHBOARD_URL": "https://example.com", "DASHBOARD_TOKEN": "tok123"},
    ):
        from compute.dashboard import DashboardPusher

        pusher = DashboardPusher()

        with patch("httpx.post", side_effect=Exception("network down")):
            # Should not raise
            pusher.push_experiment({"id": "abc"})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
python -m pytest tests/test_dashboard_pusher.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'compute.dashboard'`

- [ ] **Step 3: Write the DashboardPusher implementation**

`compute/dashboard.py`:
```python
"""Fire-and-forget push client for the Parameter Golf dashboard."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

_TIMEOUT_SECONDS = 5


class DashboardPusher:
    """Pushes events to the CF Workers dashboard.

    If DASHBOARD_URL is not set, all methods are silent no-ops.
    All methods swallow exceptions — the dashboard being down never blocks agents.
    """

    def __init__(self) -> None:
        self._url = os.environ.get("DASHBOARD_URL", "")
        self._token = os.environ.get("DASHBOARD_TOKEN", "")

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def _post(self, event: str, data: Any) -> None:
        if not self.enabled:
            return
        try:
            httpx.post(
                f"{self._url}/api/ingest",
                content=json.dumps({"event": event, "data": data}),
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=_TIMEOUT_SECONDS,
            )
        except Exception:
            pass

    def push_experiment(self, row: dict) -> None:
        self._post("experiment_complete", row)

    def push_research(self, items: list[dict]) -> None:
        self._post("research_graded", items)

    def push_verified(self, item_id: str, verified_at: str) -> None:
        self._post("research_verified", {"id": item_id, "verified_at": verified_at})

    def push_budget(self, snapshot: dict, run: dict | None = None) -> None:
        self._post("budget_update", {"snapshot": snapshot, "run": run})

    def push_doc(self, key: str, content: str) -> None:
        self._post("doc_update", {"key": key, "content": content})

    def push_heartbeat(
        self,
        statuses: list[dict],
        sota_bpb: float,
        pipeline_counts: dict,
    ) -> None:
        self._post(
            "heartbeat",
            {
                "agents": statuses,
                "sota_bpb": sota_bpb,
                "pipeline_counts": pipeline_counts,
            },
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
python -m pytest tests/test_dashboard_pusher.py -v
```

Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add compute/dashboard.py tests/test_dashboard_pusher.py
git commit -m "feat(dashboard): add Python DashboardPusher with fire-and-forget event push"
```

---

### Task 16: Integrate DashboardPusher into orchestrator

**Files:**
- Modify: `orchestrate.py` — import pusher, call at event points
- Modify: `research/grade.py` — call pusher after grading
- Modify: `research/verify.py` — call pusher after verification
- Modify: `research/inject.py` — call pusher after doc writes
- Modify: `research/reflect.py` — call pusher after strategy/technique_map writes
- Modify: `compute/budget.py` — call pusher after budget save

This task modifies existing files. The implementer should read each file first and find the exact integration points described below. The line numbers below are approximate — use the function names to locate the correct insertion points.

- [ ] **Step 1: Add pusher import and initialization to orchestrate.py**

At the top of `orchestrate.py`, add:
```python
from compute.dashboard import DashboardPusher
```

Near the other module-level constants, add:
```python
_dashboard = DashboardPusher()
```

- [ ] **Step 2: Push after experiment result append**

In `orchestrate.py`, in the `_append_result()` function, after the `f.write()` line, add:
```python
    _dashboard.push_experiment({
        "id": run_id,
        "tier": tier,
        "val_bpb": result.get(_KEY_VAL_BPB),
        "artifact_bytes": result.get(_KEY_ARTIFACT_BYTES),
        "memory_gb": None,
        "status": status,
        "promoted": tier == "runpod",
        "cost_usd": cost,
        "description": "",
        "source_item": source_item,
        "created_at": _now_iso(),
    })
```

Where `_now_iso` is imported from `agents.shared` or defined as:
```python
from datetime import datetime, timezone

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
```

- [ ] **Step 3: Push heartbeat in the supervisor loop**

In `orchestrate.py`, in the main `while True` loop (the health check loop), add a heartbeat push after the agent status checks. Add a counter to send every 10 poll intervals (~5 min if poll interval is 30s):

```python
_heartbeat_counter = 0
```

Inside the loop, after the agent health checks:
```python
    _heartbeat_counter += 1
    if _heartbeat_counter % 10 == 0:
        _dashboard.push_heartbeat(
            statuses=[
                {
                    "agent": "experiment",
                    "status": "running" if _check_agent_alive(experiment_proc) else "crashed",
                    "last_activity": _now_iso(),
                    "restart_count": experiment_restarts,
                },
                {
                    "agent": "research",
                    "status": "running" if _check_agent_alive(research_proc) else "crashed",
                    "last_activity": _now_iso(),
                    "restart_count": research_restarts,
                },
            ],
            sota_bpb=_read_sota_bpb(),
            pipeline_counts=_read_pipeline_counts(),
        )
```

Add these helper functions to `orchestrate.py`:
```python
def _read_sota_bpb() -> float:
    """Read current SOTA bpb from program.md or return 0."""
    try:
        text = Path("program.md").read_text()
        for line in text.splitlines():
            if "sota" in line.lower() and "bpb" in line.lower():
                import re
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    return float(match.group(1))
    except Exception:
        pass
    return 0.0


def _read_pipeline_counts() -> dict:
    """Count lines in each pipeline cache file."""
    def _count_lines(path: str) -> int:
        try:
            return sum(1 for _ in open(path))
        except FileNotFoundError:
            return 0
    return {
        "fetched": _count_lines("raw_cache.jsonl"),
        "graded": _count_lines("graded_cache.jsonl"),
        "verified": _count_lines("verified_cache.jsonl"),
        "injected": _count_lines("research_results.jsonl"),
    }
```

- [ ] **Step 4: Push after grading in research/grade.py**

In `research/grade.py`, in the `_append_graded()` function, after the file write loop, add:
```python
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_research([asdict(item) for item in items])
```

- [ ] **Step 5: Push after verification in research/verify.py**

In `research/verify.py`, in `_append_verified()`, after the file write, add:
```python
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_verified(item.id, item.verified_at)
```

- [ ] **Step 6: Push after doc writes in research/inject.py and research/reflect.py**

In `research/inject.py`, after each `program_md_path.write_text()` call in `inject_into_program_md()`, add:
```python
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_doc("program", content)
```

In `research/reflect.py`, after `strategy_path.write_text()` in `_write_strategy_md()`, add:
```python
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_doc("strategy", final_content)
```

After `technique_map_path.write_text()` in both `bootstrap_technique_map()` and `merge_technique_updates()`, add:
```python
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_doc("technique_map", json.dumps(data, indent=2))
```

- [ ] **Step 7: Push after budget save in compute/budget.py**

In `compute/budget.py`, in the `_save()` method of `BudgetManager`, after `BUDGET_FILE.write_text()`, add:
```python
        from compute.dashboard import DashboardPusher
        DashboardPusher().push_budget(data)
```

In `record_run()`, after `self._save()`, push the run data:
```python
        from compute.dashboard import DashboardPusher
        DashboardPusher().push_budget(
            {
                _KEY_TOTAL: self.total_credits,
                _KEY_SPENT: round(self.spent, _COST_ROUND_DIGITS),
                _KEY_RESERVE: self.min_reserve,
            },
            run=run_entry,
        )
```

Where `run_entry` is the dict that was appended to `self.runs`.

- [ ] **Step 8: Run existing tests to verify no regressions**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
python -m pytest tests/ -v
```

Expected: All tests pass (new pusher tests + existing tests).

- [ ] **Step 9: Commit**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add orchestrate.py compute/budget.py research/grade.py research/verify.py research/inject.py research/reflect.py
git commit -m "feat(dashboard): integrate DashboardPusher into orchestrator and research pipeline"
```

---

### Task 17: End-to-end smoke test and deploy

**Files:**
- No new files

- [ ] **Step 1: Run the dashboard locally with wrangler**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler dev
```

Expected: Dashboard starts on `http://localhost:8787`. All pages load without errors (data will be empty).

- [ ] **Step 2: Smoke test the ingest endpoint locally**

```bash
curl -X POST http://localhost:8787/api/ingest \
  -H "Authorization: Bearer " \
  -H "Content-Type: application/json" \
  -d '{
    "event": "experiment_complete",
    "data": {
      "id": "test123",
      "tier": "local",
      "val_bpb": 1.21,
      "artifact_bytes": 15000000,
      "memory_gb": null,
      "status": "keep",
      "promoted": false,
      "cost_usd": 0,
      "description": "test experiment",
      "source_item": "manual",
      "created_at": "2026-03-29T12:00:00Z"
    }
  }'
```

Expected: `{"ok":true}`

- [ ] **Step 3: Verify the experiment appears on the overview page**

Refresh `http://localhost:8787`. Expected: The "Recent experiments" table shows the test experiment with val_bpb 1.2100, tier "local", status "keep".

- [ ] **Step 4: Test heartbeat ingestion**

```bash
curl -X POST http://localhost:8787/api/ingest \
  -H "Authorization: Bearer " \
  -H "Content-Type: application/json" \
  -d '{
    "event": "heartbeat",
    "data": {
      "agents": [
        {"agent": "experiment", "status": "running", "last_activity": "2026-03-29T12:05:00Z", "restart_count": 0},
        {"agent": "research", "status": "idle", "last_activity": "2026-03-29T12:00:00Z", "restart_count": 1}
      ],
      "sota_bpb": 1.05,
      "pipeline_counts": {"fetched": 150, "graded": 80, "verified": 15, "injected": 8}
    }
  }'
```

Expected: `{"ok":true}`. Overview page shows agent statuses. Research page shows pipeline funnel with counts.

- [ ] **Step 5: Deploy to Cloudflare**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler d1 migrations apply pgolf-dashboard-db --remote
npx wrangler deploy
```

Expected: Deployment succeeds. Note the deployed URL.

- [ ] **Step 6: Set the dashboard token secret**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch/dashboard
npx wrangler secret put DASHBOARD_TOKEN
```

Enter a secure random token when prompted.

- [ ] **Step 7: Add dashboard URL and token to .env**

Add to the project root `.env`:
```
DASHBOARD_URL=https://pgolf-dashboard.<your-subdomain>.workers.dev
DASHBOARD_TOKEN=<the-token-you-just-set>
```

- [ ] **Step 8: Verify remote deployment**

Open the deployed URL in a browser. Expected: Dashboard loads with the design system applied — nav, footer, empty pages all render correctly.

- [ ] **Step 9: Commit any remaining changes**

```bash
cd /Users/robin/dev/parameter-golf-autoresearch
git add -A dashboard/
git commit -m "feat(dashboard): finalize dashboard for deployment"
```
