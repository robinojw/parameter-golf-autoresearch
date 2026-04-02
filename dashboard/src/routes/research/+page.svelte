<script lang="ts">
	import FunnelBar from '$lib/components/FunnelBar.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import { goto } from '$app/navigation';
	import type { ResearchItem } from '$lib/types';

	let { data } = $props();

	function updateFilter(key: string, value: string) {
		const url = new URL(window.location.href);
		url.searchParams.set(key, value);
		url.searchParams.set('page', '1');
		goto(url.toString(), { replaceState: true });
	}

	function changePage(p: number) {
		const url = new URL(window.location.href);
		url.searchParams.set('page', String(p));
		goto(url.toString(), { replaceState: true });
	}

	const columns = [
		{ key: 'summary', label: 'Summary' },
		{
			key: 'score',
			label: 'Score',
			format: (v: unknown) => Number(v).toFixed(1)
		},
		{ key: 'tier', label: 'Tier' },
		{
			key: 'verified',
			label: 'Verified',
			format: (v: unknown) => (v ? 'Yes' : 'No')
		},
		{
			key: 'graded_at',
			label: 'Graded',
			format: (v: unknown) => (v ? new Date(String(v)).toLocaleDateString() : '—')
		}
	];
</script>

<div class="space-y-8">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Research pipeline</h2>
		<FunnelBar counts={data.pipelineCounts} />
	</section>

	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Tier breakdown</h2>
		<div class="flex gap-6 text-sm">
			{#each data.tierCounts as { tier, count }}
				<span>
					<span class="text-muted">Tier {tier}:</span>
					{count}
				</span>
			{/each}
		</div>
	</section>

	<section>
		<div class="flex gap-4 mb-4">
			<select
				class="border border-edge bg-transparent px-3 py-2 text-sm text-on-surface focus:border-on-surface focus:outline-none"
				value={data.tier}
				onchange={(e) => updateFilter('tier', e.currentTarget.value)}
			>
				<option value="all">All tiers</option>
				<option value="A">Tier A</option>
				<option value="B">Tier B</option>
				<option value="C">Tier C</option>
			</select>
		</div>

		<DataTable
			data={data.items}
			{columns}
			total={data.total}
			page={data.page}
			perPage={50}
			onPageChange={changePage}
			expandable={true}
		>
			{#snippet expandRow(item: ResearchItem)}
				<div class="grid grid-cols-5 gap-4 text-xs">
					<div><span class="text-muted">Bpb impact:</span> {item.bpb_impact}</div>
					<div><span class="text-muted">Size compat:</span> {item.size_compat}</div>
					<div><span class="text-muted">Time compat:</span> {item.time_compat}</div>
					<div><span class="text-muted">Implement:</span> {item.implement}</div>
					<div><span class="text-muted">Novelty:</span> {item.novelty}</div>
				</div>
				{#if item.flags.length > 0}
					<div class="mt-2 text-xs text-muted">
						Flags: {item.flags.join(', ')}
					</div>
				{/if}
			{/snippet}
		</DataTable>
	</section>
</div>
