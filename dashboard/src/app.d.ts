import type { D1Database, KVNamespace } from '@cloudflare/workers-types';

declare global {
	namespace App {
		interface Platform {
			env: {
				DB: D1Database;
				KV: KVNamespace;
				DASHBOARD_TOKEN: string;
			};
		}
	}
}

export {};
