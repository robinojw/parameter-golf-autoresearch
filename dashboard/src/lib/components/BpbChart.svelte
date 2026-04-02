<script lang="ts">
	import { LayerCake, Svg } from 'layercake';
	import BpbChartInner from './BpbChartInner.svelte';

	let {
		data,
		promotions = []
	}: {
		data: { date: string; best_bpb: number; tier: string }[];
		promotions?: { date: string }[];
	} = $props();

	let chartData = $derived(
		data.map((d) => ({
			x: new Date(d.date).getTime(),
			y: d.best_bpb,
			tier: d.tier
		}))
	);
</script>

{#if chartData.length > 0}
	<div class="h-64 w-full">
		<LayerCake
			padding={{ top: 10, right: 10, bottom: 30, left: 50 }}
			x="x"
			y="y"
			data={chartData}
		>
			<Svg>
				<BpbChartInner {chartData} {promotions} />
			</Svg>
		</LayerCake>
	</div>
{:else}
	<div class="h-64 w-full flex items-center justify-center text-muted text-sm">
		No experiment data yet
	</div>
{/if}
