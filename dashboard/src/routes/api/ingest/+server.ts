import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import type { IngestEvent } from '$lib/types';

export const POST: RequestHandler = async ({ request, platform }) => {
	const token = request.headers.get('authorization')?.replace('Bearer ', '');
	if (!token || token !== platform?.env.DASHBOARD_TOKEN) {
		throw error(401, 'Unauthorized');
	}

	const body = (await request.json()) as IngestEvent;
	const db = platform?.env?.DB;
	const kv = platform?.env?.KV;

	if (!db || !kv) {
		console.error('D1/KV bindings missing. platform.env keys:', Object.keys(platform?.env ?? {}));
		return json({ error: 'Storage bindings not configured' }, { status: 503 });
	}

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
				throw error(400, `Invalid doc key: ${key}`);
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

		case 'activity': {
			const { agent, action, detail, timestamp } = body.data;
			await db
				.prepare(
					'INSERT INTO activity_log (agent, action, detail, created_at) VALUES (?, ?, ?, ?)'
				)
				.bind(agent, action, detail?.slice(0, 500) ?? '', timestamp)
				.run();
			// Trim old entries — keep last 200
			await db
				.prepare(
					'DELETE FROM activity_log WHERE id NOT IN (SELECT id FROM activity_log ORDER BY id DESC LIMIT 200)'
				)
				.run();
			break;
		}

		default:
			throw error(400, 'Unknown event type');
	}

	return json({ ok: true });
};
