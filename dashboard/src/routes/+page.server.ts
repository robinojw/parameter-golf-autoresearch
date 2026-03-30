import type { PageServerLoad } from './$types';
import { getRecentExperiments, getRecentResearch, getBudgetSnapshot, getBestBpb, getMaxArtifactBytes, getRecentActivity } from '$lib/server/db';
import { getSotaBpb } from '$lib/server/kv';

export const load: PageServerLoad = async ({ platform }) => {
	const db = platform?.env?.DB;
	const kv = platform?.env?.KV;

	if (!db || !kv) {
		return {
			experiments: [],
			research: [],
			budget: { total_credits: 0, spent: 0, min_reserve: 0, updated_at: '' },
			bestBpb: { local: null, runpod: null },
			sotaBpb: null,
			maxArtifactBytes: null,
			activity: []
		};
	}

	const [experiments, research, budget, bestBpb, sotaBpb, maxArtifactBytes, activity] = await Promise.all([
		getRecentExperiments(db, 10),
		getRecentResearch(db, 10),
		getBudgetSnapshot(db),
		getBestBpb(db),
		getSotaBpb(kv),
		getMaxArtifactBytes(db),
		getRecentActivity(db, 30)
	]);

	return { experiments, research, budget, bestBpb, sotaBpb, maxArtifactBytes, activity };
};
