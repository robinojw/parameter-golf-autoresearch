import type { LayoutServerLoad } from './$types';
import { getAgentStatuses } from '$lib/server/db';
import { getLastPush } from '$lib/server/kv';

export const load: LayoutServerLoad = async ({ platform }) => {
	const db = platform?.env?.DB;
	const kv = platform?.env?.KV;

	if (!db || !kv) {
		return { agents: [], lastPush: '' };
	}

	const [agents, lastPush] = await Promise.all([getAgentStatuses(db), getLastPush(kv)]);

	return { agents, lastPush };
};
