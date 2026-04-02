<script lang="ts">
	import BudgetBar from '$lib/components/BudgetBar.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import MetricCard from '$lib/components/MetricCard.svelte';

	let { data } = $props();

	const columns = [
		{
			key: 'started_at',
			label: 'Date',
			format: (v: unknown) => (v ? new Date(String(v)).toLocaleString() : '—')
		},
		{
			key: 'duration_s',
			label: 'Duration',
			format: (v: unknown) => `${Math.round(Number(v) / 60)}m`
		},
		{
			key: 'cost_usd',
			label: 'Cost',
			format: (v: unknown) => `$${Number(v).toFixed(2)}`
		},
		{
			key: 'val_bpb',
			label: 'Val bpb',
			format: (v: unknown) => (v != null ? Number(v).toFixed(4) : '—')
		},
		{ key: 'promoted_from', label: 'From experiment' }
	];
</script>

<div class="space-y-8">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Budget</h2>
		<BudgetBar snapshot={data.snapshot} />
	</section>

	<section>
		<div class="grid grid-cols-2 sm:grid-cols-3 gap-4">
			<MetricCard
				label="Burn rate"
				value="${data.burnRate.toFixed(2)}/day"
			/>
			<MetricCard
				label="Est. days remaining"
				value={data.daysRemaining !== null ? Math.round(data.daysRemaining).toString() : '—'}
			/>
			<MetricCard
				label="Total runs"
				value={data.runs.length.toString()}
			/>
		</div>
	</section>

	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Run history</h2>
		<DataTable data={data.runs} {columns} />
	</section>
</div>
