<script lang="ts">
	let { label, value, sub = '' }: { label: string; value: string; sub?: string } = $props();
</script>

<div class="border border-edge p-4">
	<div class="text-xs font-medium tracking-wider text-muted">{label}</div>
	<div class="text-2xl font-semibold mt-1">{value}</div>
	{#if sub}
		<div class="text-xs text-muted mt-1">{sub}</div>
	{/if}
</div>
