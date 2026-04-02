<script lang="ts">
	import { page } from '$app/state';
	import { base } from '$app/paths';
	import { fly } from 'svelte/transition';

	const links = [
		{ href: '', label: 'Overview' },
		{ href: '/experiments', label: 'Experiments' },
		{ href: '/research', label: 'Research' },
		{ href: '/budget', label: 'Budget' },
		{ href: '/strategy', label: 'Strategy' }
	];

	function isActive(href: string): boolean {
		const full = base + href;
		if (href === '') return page.url.pathname === base || page.url.pathname === base + '/';
		return page.url.pathname.startsWith(full);
	}
</script>

<nav
	class="border-b border-edge px-4 sm:px-8 py-4 flex items-baseline justify-between"
	in:fly={{ y: -12, duration: 400 }}
>
	<a href="{base}/" class="text-on-surface no-underline">
		<span class="text-lg font-semibold leading-none tracking-tighter">Parameter Golf</span>
	</a>
	<div class="flex gap-4">
		{#each links as { href, label }}
			<a
				href="{base}{href}"
				class="text-sm text-muted no-underline transition-opacity duration-150"
				class:opacity-40={!isActive(href)}
				class:text-on-surface={isActive(href)}
			>
				{label}
			</a>
		{/each}
	</div>
</nav>
