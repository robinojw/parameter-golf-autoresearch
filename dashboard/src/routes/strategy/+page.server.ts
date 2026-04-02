import type { PageServerLoad } from './$types';
import { getDoc, getTechniqueMap } from '$lib/server/kv';

export const load: PageServerLoad = async ({ platform }) => {
	const kv = platform?.env?.KV;

	if (!kv) {
		return { programMd: '', strategyMd: '', techniqueMap: { nodes: [], edges: [] } };
	}

	const [programMd, strategyMd, techniqueMap] = await Promise.all([
		getDoc(kv, 'program'),
		getDoc(kv, 'strategy'),
		getTechniqueMap(kv)
	]);

	return { programMd, strategyMd, techniqueMap };
};
