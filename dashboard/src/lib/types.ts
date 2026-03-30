export interface Experiment {
	id: string;
	tier: 'local' | 'runpod';
	val_bpb: number | null;
	artifact_bytes: number | null;
	memory_gb: number | null;
	status: 'keep' | 'discard' | 'crash';
	promoted: boolean;
	cost_usd: number;
	description: string;
	source_item: string;
	created_at: string;
}

export interface ResearchItem {
	id: string;
	score: number;
	tier: 'A' | 'B' | 'C';
	bpb_impact: number;
	size_compat: number;
	time_compat: number;
	implement: number;
	novelty: number;
	summary: string;
	flags: string[];
	verified: boolean;
	graded_at: string;
	verified_at: string | null;
}

export interface BudgetRun {
	run_id: string;
	started_at: string;
	duration_s: number;
	cost_usd: number;
	val_bpb: number | null;
	artifact_bytes: number | null;
	promoted_from: string;
}

export interface BudgetSnapshot {
	total_credits: number;
	spent: number;
	min_reserve: number;
	updated_at: string;
}

export interface AgentStatus {
	agent: 'experiment' | 'research';
	status: 'running' | 'idle' | 'crashed';
	last_activity: string;
	restart_count: number;
}

export interface PipelineCounts {
	fetched: number;
	graded: number;
	verified: number;
	injected: number;
}

export interface TechniqueNode {
	id: string;
	label: string;
	status: 'proven' | 'exploring' | 'dead_end' | 'untried';
}

export interface TechniqueEdge {
	source: string;
	target: string;
}

export interface TechniqueMap {
	nodes: TechniqueNode[];
	edges: TechniqueEdge[];
}

export type IngestEvent =
	| { event: 'experiment_complete'; data: Experiment }
	| { event: 'research_graded'; data: ResearchItem[] }
	| { event: 'research_verified'; data: { id: string; verified_at: string } }
	| { event: 'budget_update'; data: { snapshot: BudgetSnapshot; run?: BudgetRun } }
	| { event: 'doc_update'; data: { key: string; content: string } }
	| {
			event: 'heartbeat';
			data: {
				agents: AgentStatus[];
				sota_bpb: number;
				pipeline_counts: PipelineCounts;
			};
	  };
