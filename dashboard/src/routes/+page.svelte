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

	<!-- Activity feed -->
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Activity</h2>
		{#if data.activity.length > 0}
			<div class="border border-edge divide-y divide-edge max-h-80 overflow-y-auto">
				{#each data.activity as entry}
					<div class="px-3 py-2 text-sm flex gap-3">
						<span class="text-xs text-muted shrink-0 w-24 font-mono">
							{new Date(entry.created_at).toLocaleTimeString()}
						</span>
						<span class="text-xs text-muted shrink-0 w-32">{entry.agent}</span>
						<span class="text-xs text-muted shrink-0 w-16">{entry.action}</span>
						<span class="text-sm truncate">{entry.detail}</span>
					</div>
				{/each}
			</div>
		{:else}
			<p class="text-sm text-muted">No activity yet</p>
		{/if}
	</section>

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
