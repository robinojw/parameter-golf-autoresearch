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
