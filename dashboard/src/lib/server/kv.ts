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
