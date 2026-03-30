import type { PageServerLoad } from './$types';
import { getResearchPaginated, getResearchTierCounts } from '$lib/server/db';
import { getPipelineCounts } from '$lib/server/kv';

export const load: PageServerLoad = async ({ platform, url }) => {
	const db = platform?.env?.DB;
	const kv = platform?.env?.KV;
	const page = Number(url.searchParams.get('page') ?? '1');
	const tier = url.searchParams.get('tier') ?? 'all';

	if (!db || !kv) {
		return { items: [], total: 0, tierCounts: [], pipelineCounts: { fetched: 0, graded: 0, verified: 0, injected: 0 }, page, tier };
	}

	const [research, tierCounts, pipelineCounts] = await Promise.all([
		getResearchPaginated(db, page, 50, tier),
		getResearchTierCounts(db),
		getPipelineCounts(kv)
	]);

	return { ...research, tierCounts, pipelineCounts, page, tier };
};
