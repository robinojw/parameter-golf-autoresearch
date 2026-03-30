import type { PageServerLoad } from './$types';
import { getExperimentsPaginated, getBestBpbPerDay, getPromotionDates } from '$lib/server/db';

export const load: PageServerLoad = async ({ platform, url }) => {
	const db = platform?.env?.DB;
	const page = Number(url.searchParams.get('page') ?? '1');
	const tier = url.searchParams.get('tier') ?? 'all';
	const status = url.searchParams.get('status') ?? 'all';

	if (!db) {
		return { items: [], total: 0, chartData: [], promotions: [], page, tier, status };
	}

	const [experiments, chartData, promotions] = await Promise.all([
		getExperimentsPaginated(db, page, 50, tier, status),
		getBestBpbPerDay(db),
		getPromotionDates(db)
	]);

	return { ...experiments, chartData, promotions, page, tier, status };
};
