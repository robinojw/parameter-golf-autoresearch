<script lang="ts">
	import type { BudgetSnapshot } from '$lib/types';

	let { snapshot }: { snapshot: BudgetSnapshot } = $props();

	let remaining = $derived(snapshot.total_credits - snapshot.spent - snapshot.min_reserve);
	let spentPct = $derived((snapshot.spent / snapshot.total_credits) * 100);
	let availablePct = $derived((remaining / snapshot.total_credits) * 100);
	let reservePct = $derived((snapshot.min_reserve / snapshot.total_credits) * 100);
</script>

<div>
	<div class="border border-edge h-8 flex">
		<div
			class="h-full bg-on-surface"
			style="width: {spentPct}%; opacity: 0.3"
			title="Spent: ${snapshot.spent.toFixed(2)}"
		></div>
		<div
			class="h-full bg-on-surface"
			style="width: {availablePct}%; opacity: 0.1"
			title="Available: ${remaining.toFixed(2)}"
		></div>
		<div
			class="h-full border-l border-edge bg-on-surface"
			style="width: {reservePct}%; opacity: 0.05"
			title="Reserve: ${snapshot.min_reserve.toFixed(2)}"
		></div>
	</div>
	<div class="flex justify-between text-xs text-muted mt-1">
		<span>Spent: ${snapshot.spent.toFixed(2)}</span>
		<span>Available: ${remaining.toFixed(2)}</span>
		<span>Reserve: ${snapshot.min_reserve.toFixed(2)}</span>
	</div>
</div>
