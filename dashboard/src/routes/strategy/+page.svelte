<script lang="ts">
	import TechGraph from '$lib/components/TechGraph.svelte';
	import { marked } from 'marked';

	let { data } = $props();

	let programHtml = $derived(data.programMd ? marked.parse(data.programMd) : '');
	let strategyHtml = $derived(data.strategyMd ? marked.parse(data.strategyMd) : '');
</script>

<div class="space-y-12">
	<section>
		<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Technique map</h2>
		<TechGraph map={data.techniqueMap} />
		<div class="flex gap-6 mt-3 text-xs text-muted">
			<span>Proven (100%)</span>
			<span class="opacity-70">Exploring (70%)</span>
			<span class="opacity-40">Dead end (40%)</span>
			<span class="opacity-70" style="text-decoration: underline dashed">Untried (dashed)</span>
		</div>
	</section>

	{#if data.programMd}
		<section>
			<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Program</h2>
			<article class="prose-content leading-relaxed">
				{@html programHtml}
			</article>
		</section>
	{/if}

	{#if data.strategyMd}
		<section>
			<h2 class="text-xs font-medium tracking-wider text-muted mb-3">Strategy</h2>
			<article class="prose-content leading-relaxed">
				{@html strategyHtml}
			</article>
		</section>
	{/if}
</div>

<style>
	:global(.prose-content h1) {
		font-size: 1.75rem;
		margin-bottom: 0.75rem;
	}
	:global(.prose-content h2) {
		font-size: 1.375rem;
		margin-top: 2rem;
		margin-bottom: 0.75rem;
	}
	:global(.prose-content h3) {
		font-size: 1.125rem;
		margin-top: 2rem;
		margin-bottom: 0.75rem;
	}
	:global(.prose-content p) {
		margin-bottom: 1rem;
	}
	:global(.prose-content a) {
		color: var(--on-surface);
		text-decoration: underline;
		text-underline-offset: 3px;
		transition: opacity 150ms;
	}
	:global(.prose-content a:hover) {
		opacity: 0.7;
	}
	:global(.prose-content code) {
		font-family: var(--font-mono);
		font-size: 0.875em;
	}
	:global(.prose-content pre) {
		border: 1px solid var(--edge);
		padding: 1rem;
		overflow-x: auto;
		margin-bottom: 1rem;
	}
	:global(.prose-content ul, .prose-content ol) {
		padding-left: 1.5rem;
		margin-bottom: 1rem;
	}
</style>
