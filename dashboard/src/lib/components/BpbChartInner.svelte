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

	let localData = $derived(chartData.filter((d) => d.tier === 'local'));
	let runpodData = $derived(chartData.filter((d) => d.tier === 'runpod'));

	let localPath = $derived.by(() => {
		if (localData.length === 0) return '';
		const gen = d3line<{ x: number; y: number }>()
			.x((d) => $xScale(d.x))
			.y((d) => $yScale(d.y));
		return gen(localData);
	});

	let runpodPath = $derived.by(() => {
		if (runpodData.length === 0) return '';
		const gen = d3line<{ x: number; y: number }>()
			.x((d) => $xScale(d.x))
			.y((d) => $yScale(d.y));
		return gen(runpodData);
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

<!-- Local line + dots (dashed, muted) -->
{#if localPath}
	<path d={localPath} fill="none" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4,3" />
{/if}
{#each localData as point}
	<circle
		cx={$xScale(point.x)}
		cy={$yScale(point.y)}
		r="3"
		fill="var(--muted)"
	/>
{/each}

<!-- RunPod line + dots (solid, full contrast) -->
{#if runpodPath}
	<path d={runpodPath} fill="none" stroke="var(--on-surface)" stroke-width="2" />
{/if}
{#each runpodData as point}
	<circle
		cx={$xScale(point.x)}
		cy={$yScale(point.y)}
		r="4"
		fill="var(--on-surface)"
	/>
{/each}

<!-- Legend -->
<g transform="translate({$width - 140}, 0)">
	<line x1="0" x2="16" y1="6" y2="6" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4,3" />
	<text x="20" y="10" fill="var(--muted)" font-size="10">Local (MLX)</text>
	<line x1="0" x2="16" y1="22" y2="22" stroke="var(--on-surface)" stroke-width="2" />
	<text x="20" y="26" fill="var(--on-surface)" font-size="10">RunPod (H100)</text>
</g>
