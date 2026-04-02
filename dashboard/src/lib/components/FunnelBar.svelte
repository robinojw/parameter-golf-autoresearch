<script lang="ts">
	import type { PipelineCounts } from '$lib/types';

	let { counts }: { counts: PipelineCounts } = $props();

	let max = $derived(Math.max(counts.fetched, counts.graded, counts.verified, counts.injected, 1));

	const stages: { key: keyof PipelineCounts; label: string }[] = [
		{ key: 'fetched', label: 'Fetched' },
		{ key: 'graded', label: 'Graded' },
		{ key: 'verified', label: 'Verified' },
		{ key: 'injected', label: 'Injected' }
	];
</script>

<div class="space-y-2">
	{#each stages as { key, label }}
		{@const pct = (counts[key] / max) * 100}
		<div class="flex items-center gap-3">
			<span class="text-xs text-muted w-16 text-right">{label}</span>
			<div class="flex-1 border border-edge h-6 relative">
				<div
					class="h-full bg-on-surface transition-all duration-150"
					style="width: {pct}%; opacity: 0.15"
				></div>
				<span class="absolute inset-0 flex items-center px-2 text-xs">{counts[key]}</span>
			</div>
		</div>
	{/each}
</div>
