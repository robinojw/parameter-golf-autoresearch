<script lang="ts">
	import '$lib/styles/app.css';
	import Nav from '$lib/components/Nav.svelte';
	import { fly } from 'svelte/transition';
	import type { Snippet } from 'svelte';

	let { data, children }: { data: any; children: Snippet } = $props();
</script>

<div class="min-h-screen flex flex-col bg-surface text-on-surface">
	<Nav />

	<main class="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-8 py-6 sm:py-10">
		<div in:fly={{ y: 24, duration: 500 }}>
			{@render children()}
		</div>
	</main>

	<footer class="border-t border-edge px-4 sm:px-8 py-3 flex justify-between text-xs text-muted">
		<span>Parameter Golf Dashboard</span>
		{#if data.lastPush}
			<span>Last updated: {new Date(data.lastPush).toLocaleString()}</span>
		{:else}
			<span>No data yet</span>
		{/if}
	</footer>
</div>
