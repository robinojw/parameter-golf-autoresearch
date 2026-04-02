import type { PageServerLoad } from './$types';
import { getBudgetSnapshot, getBudgetRuns, getBurnRate } from '$lib/server/db';

export const load: PageServerLoad = async ({ platform }) => {
	const db = platform?.env?.DB;

	if (!db) {
		return {
			snapshot: { total_credits: 0, spent: 0, min_reserve: 0, updated_at: '' },
			runs: [],
			burnRate: 0,
			daysRemaining: null
		};
	}

	const [snapshot, runs, burnRate] = await Promise.all([
		getBudgetSnapshot(db),
		getBudgetRuns(db),
		getBurnRate(db)
	]);

	const remaining = snapshot.total_credits - snapshot.spent - snapshot.min_reserve;
	const daysRemaining = burnRate > 0 ? remaining / burnRate : null;

	return { snapshot, runs, burnRate, daysRemaining };
};
