<script lang="ts">
	import { getContext } from 'svelte';
	import { get, type Readable } from 'svelte/store';
	import { line as d3line } from 'd3-shape';

	let {
		chartData,
		promotions = []
	}: {
		chartData: { x: number; y: number; tier: string }[];
		promotions?: { date: string }[];
	} = $props();

	const ctx: Record<string, Readable<any>> = getContext('LayerCake');
	const xScale = ctx.xScale;
	const yScale = ctx.yScale;
	const width = ctx.width;
	const height = ctx.height;

	let linePath = $derived.by(() => {
		const xs = $xScale;
		const ys = $yScale;
		const gen = d3line<{ x: number; y: number }>()
			.x((d) => xs(d.x))
			.y((d) => ys(d.y));
		return gen(chartData);
	});
</script>

<!-- Grid lines -->
{#each $yScale.ticks(5) as tick}
	<line
		x1={0}
		x2={$width}
		y1={$yScale(tick)}
		y2={$yScale(tick)}
		stroke="var(--edge)"
		stroke-width="1"
		stroke-dasharray="2"
	/>
	<text x={-8} y={$yScale(tick)} dy="4" text-anchor="end" fill="var(--muted)" font-size="11">
		{tick.toFixed(3)}
	</text>
{/each}

<!-- X axis labels -->
{#each $xScale.ticks(6) as tick}
	<text
		x={$xScale(tick)}
		y={$height + 20}
		text-anchor="middle"
		fill="var(--muted)"
		font-size="11"
	>
		{new Date(tick).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
	</text>
{/each}

<!-- Promotion vertical lines -->
{#each promotions as promo}
	{@const px = $xScale(new Date(promo.date).getTime())}
	<line
		x1={px}
		x2={px}
		y1={0}
		y2={$height}
		stroke="var(--edge)"
		stroke-width="1"
		stroke-dasharray="4,2"
	/>
{/each}

<!-- Line -->
<path d={linePath} fill="none" stroke="var(--on-surface)" stroke-width="1.5" />

<!-- Dots -->
{#each chartData as point}
	<circle
		cx={$xScale(point.x)}
		cy={$yScale(point.y)}
		r="3"
		fill={point.tier === 'runpod' ? 'var(--on-surface)' : 'var(--muted)'}
	/>
{/each}
