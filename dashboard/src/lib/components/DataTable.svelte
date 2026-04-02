<script lang="ts" generics="T extends Record<string, unknown>">
	import type { Snippet } from 'svelte';

	interface Column {
		key: string;
		label: string;
		format?: (val: unknown, row: T) => string;
	}

	let {
		data,
		columns,
		total = data.length,
		page = 1,
		perPage = 50,
		onPageChange,
		expandable = false,
		expandRow
	}: {
		data: T[];
		columns: Column[];
		total?: number;
		page?: number;
		perPage?: number;
		onPageChange?: (page: number) => void;
		expandable?: boolean;
		expandRow?: Snippet<[T]>;
	} = $props();

	let expandedId = $state<string | null>(null);
	let sortKey = $state('');
	let sortAsc = $state(true);

	let sorted = $derived.by(() => {
		if (!sortKey) return data;
		return [...data].sort((a, b) => {
			const av = a[sortKey];
			const bv = b[sortKey];
			if (av == null) return 1;
			if (bv == null) return -1;
			if (av < bv) return sortAsc ? -1 : 1;
			if (av > bv) return sortAsc ? 1 : -1;
			return 0;
		});
	});

	let totalPages = $derived(Math.ceil(total / perPage));

	function toggleSort(key: string) {
		if (sortKey === key) {
			sortAsc = !sortAsc;
		} else {
			sortKey = key;
			sortAsc = true;
		}
	}

	function toggleExpand(id: string) {
		expandedId = expandedId === id ? null : id;
	}
</script>

<div class="overflow-x-auto">
	<table class="w-full text-sm">
		<thead>
			<tr class="border-b border-edge">
				{#each columns as col}
					<th
						class="text-left py-2 pr-4 text-xs font-medium tracking-wider text-muted cursor-pointer transition-opacity duration-150 hover:opacity-70"
						onclick={() => toggleSort(col.key)}
					>
						{col.label}
						{#if sortKey === col.key}
							<span class="ml-1">{sortAsc ? '\u2191' : '\u2193'}</span>
						{/if}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each sorted as row, i (row.id ?? i)}
				<tr
					class="border-b border-edge transition-opacity duration-150 hover:opacity-70"
					class:cursor-pointer={expandable}
					onclick={() => expandable && toggleExpand(String(row.id ?? i))}
				>
					{#each columns as col}
						<td class="py-2 pr-4">
							{col.format ? col.format(row[col.key], row) : (row[col.key] ?? '')}
						</td>
					{/each}
				</tr>
				{#if expandable && expandedId === String(row.id ?? i) && expandRow}
					<tr class="border-b border-edge">
						<td colspan={columns.length} class="py-3 px-4 bg-edge/10">
							{@render expandRow(row)}
						</td>
					</tr>
				{/if}
			{/each}
		</tbody>
	</table>
</div>

{#if totalPages > 1}
	<div class="flex items-center gap-2 mt-4 text-sm text-muted">
		<button
			class="border border-edge bg-transparent px-3 py-1 text-sm text-on-surface transition-colors hover:bg-edge disabled:opacity-50"
			disabled={page <= 1}
			onclick={() => onPageChange?.(page - 1)}
		>
			Prev
		</button>
		<span>{page} / {totalPages}</span>
		<button
			class="border border-edge bg-transparent px-3 py-1 text-sm text-on-surface transition-colors hover:bg-edge disabled:opacity-50"
			disabled={page >= totalPages}
			onclick={() => onPageChange?.(page + 1)}
		>
			Next
		</button>
	</div>
{/if}
