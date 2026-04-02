<script lang="ts">
	import { onMount } from 'svelte';
	import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide } from 'd3-force';
	import type { TechniqueMap } from '$lib/types';

	let { map }: { map: TechniqueMap } = $props();

	interface SimNode {
		id: string;
		label: string;
		status: string;
		x: number;
		y: number;
	}

	interface SimLink {
		source: string | SimNode;
		target: string | SimNode;
	}

	let width = 800;
	let height = 500;
	let nodes = $state<SimNode[]>([]);
	let links = $state<SimLink[]>([]);
	let hoveredId = $state<string | null>(null);

	function statusOpacity(status: string): number {
		switch (status) {
			case 'proven': return 1;
			case 'exploring': return 0.7;
			case 'dead_end': return 0.4;
			case 'untried': return 0.7;
			default: return 0.5;
		}
	}

	function statusDash(status: string): string {
		return status === 'untried' ? '4,2' : 'none';
	}

	function isConnected(nodeId: string): boolean {
		if (!hoveredId) return true;
		if (nodeId === hoveredId) return true;
		return links.some((l) => {
			const src = typeof l.source === 'string' ? l.source : l.source.id;
			const tgt = typeof l.target === 'string' ? l.target : l.target.id;
			return (src === hoveredId && tgt === nodeId) || (tgt === hoveredId && src === nodeId);
		});
	}

	onMount(() => {
		if (map.nodes.length === 0) return;

		const simNodes: SimNode[] = map.nodes.map((n) => ({
			...n,
			x: width / 2 + Math.random() * 100 - 50,
			y: height / 2 + Math.random() * 100 - 50
		}));

		const simLinks: SimLink[] = map.edges.map((e) => ({
			source: e.source,
			target: e.target
		}));

		const simulation = forceSimulation(simNodes)
			.force(
				'link',
				forceLink(simLinks)
					.id((d: any) => d.id)
					.distance(80)
			)
			.force('charge', forceManyBody().strength(-200))
			.force('center', forceCenter(width / 2, height / 2))
			.force('collide', forceCollide(30));

		simulation.on('tick', () => {
			nodes = [...simNodes];
			links = [...simLinks];
		});

		return () => simulation.stop();
	});
</script>

{#if map.nodes.length > 0}
	<svg {width} {height} class="w-full" viewBox="0 0 {width} {height}">
		<!-- Edges -->
		{#each links as link}
			{@const src = typeof link.source === 'string' ? null : link.source}
			{@const tgt = typeof link.target === 'string' ? null : link.target}
			{#if src && tgt}
				<line
					x1={src.x}
					y1={src.y}
					x2={tgt.x}
					y2={tgt.y}
					stroke="var(--edge)"
					stroke-width="1"
					opacity={hoveredId ? (isConnected(src.id) && isConnected(tgt.id) ? 1 : 0.15) : 1}
				/>
			{/if}
		{/each}

		<!-- Nodes -->
		{#each nodes as node}
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<g
				transform="translate({node.x}, {node.y})"
				opacity={hoveredId ? (isConnected(node.id) ? statusOpacity(node.status) : 0.15) : statusOpacity(node.status)}
				class="transition-opacity duration-150 cursor-pointer"
				role="img"
				aria-label={node.label}
				onmouseenter={() => (hoveredId = node.id)}
				onmouseleave={() => (hoveredId = null)}
			>
				<rect
					x="-30"
					y="-12"
					width="60"
					height="24"
					fill="var(--surface)"
					stroke="var(--on-surface)"
					stroke-width="1"
					stroke-dasharray={statusDash(node.status)}
				/>
				<text
					text-anchor="middle"
					dy="4"
					fill="var(--on-surface)"
					font-size="10"
					font-family="var(--font-sans)"
				>
					{node.label.length > 8 ? node.label.slice(0, 8) + '...' : node.label}
				</text>
			</g>
		{/each}
	</svg>
{:else}
	<div class="h-64 flex items-center justify-center text-muted text-sm">
		No technique data yet
	</div>
{/if}
