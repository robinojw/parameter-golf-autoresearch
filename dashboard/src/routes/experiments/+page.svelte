<script lang="ts">
	import BpbChart from '$lib/components/BpbChart.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import { goto } from '$app/navigation';

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
		{ key: 'description', label: 'Description' },
		{ key: 'tier', label: 'Tier' },
		{
			key: 'val_bpb',
			label: 'Val bpb',
			format: (v: unknown) => (v != null ? Number(v).toFixed(4) : '—')
		},
		{
			key: 'artifact_bytes',
			label: 'Size',
			format: (v: unknown) => (v != null ? `${(Number(v) / 1e6).toFixed(1)}MB` : '—')
		},
		{ key: 'status', label: 'Status' },
		{
			key: 'cost_usd',
			label: 'Cost',
			format: (v: unknown) => (Number(v) > 0 ? `$${Number(v).toFixed(2)}` : '—')
		},
		{ key: 'source_item', label: 'Source' },
		{
			key: 'created_at',
			label: 'Date',
			format: (v: unknown) => (v ? new Date(String(v)).toLocaleDateString() : '—')
		}
	];
</script>

<div class="space-y-8">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Val bpb over time</h2>
		<BpbChart data={data.chartData} promotions={data.promotions} />
	</section>

	<section>
		<div class="flex gap-4 mb-4">
			<select
				class="border border-edge bg-transparent px-3 py-2 text-sm text-on-surface focus:border-on-surface focus:outline-none"
				value={data.tier}
				onchange={(e) => updateFilter('tier', e.currentTarget.value)}
			>
				<option value="all">All tiers</option>
				<option value="local">Local</option>
				<option value="runpod">Runpod</option>
			</select>
			<select
				class="border border-edge bg-transparent px-3 py-2 text-sm text-on-surface focus:border-on-surface focus:outline-none"
				value={data.status}
				onchange={(e) => updateFilter('status', e.currentTarget.value)}
			>
				<option value="all">All statuses</option>
				<option value="keep">Keep</option>
				<option value="discard">Discard</option>
				<option value="crash">Crash</option>
			</select>
		</div>

		<DataTable
			data={data.items}
			{columns}
			total={data.total}
			page={data.page}
			perPage={50}
			onPageChange={changePage}
		/>
	</section>
</div>
