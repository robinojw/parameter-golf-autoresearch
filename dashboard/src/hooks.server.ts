import type { Handle } from '@sveltejs/kit';

const ALLOWED_ORIGINS = new Set([
	'http://localhost:5173',
	'http://localhost:4173',
	'http://127.0.0.1:5173',
	'http://127.0.0.1:4173',
]);

export const handle: Handle = async ({ event, resolve }) => {
	const response = await resolve(event);

	const origin = event.request.headers.get('origin');
	if (origin && ALLOWED_ORIGINS.has(origin)) {
		response.headers.set('Access-Control-Allow-Origin', origin);
		response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
		response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
	}

	return response;
};
