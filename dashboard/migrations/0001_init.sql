CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    val_bpb REAL,
    artifact_bytes INTEGER,
    memory_gb REAL,
    status TEXT NOT NULL DEFAULT 'keep',
    promoted INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    source_item TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX idx_experiments_created_at ON experiments(created_at);
CREATE INDEX idx_experiments_tier ON experiments(tier);
CREATE INDEX idx_experiments_status ON experiments(status);

CREATE TABLE research_items (
    id TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0,
    tier TEXT NOT NULL DEFAULT 'C',
    bpb_impact REAL NOT NULL DEFAULT 0,
    size_compat REAL NOT NULL DEFAULT 0,
    time_compat REAL NOT NULL DEFAULT 0,
    implement REAL NOT NULL DEFAULT 0,
    novelty REAL NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    flags TEXT NOT NULL DEFAULT '[]',
    verified INTEGER NOT NULL DEFAULT 0,
    graded_at TEXT NOT NULL,
    verified_at TEXT
);

CREATE INDEX idx_research_items_tier ON research_items(tier);
CREATE INDEX idx_research_items_score ON research_items(score);

CREATE TABLE budget_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    duration_s INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    val_bpb REAL,
    artifact_bytes INTEGER,
    promoted_from TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_budget_runs_started_at ON budget_runs(started_at);

CREATE TABLE budget_snapshot (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_credits REAL NOT NULL DEFAULT 0,
    spent REAL NOT NULL DEFAULT 0,
    min_reserve REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

INSERT INTO budget_snapshot (id, total_credits, spent, min_reserve, updated_at)
VALUES (1, 500.0, 0.0, 50.0, '');

CREATE TABLE agent_status (
    agent TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'idle',
    last_activity TEXT NOT NULL DEFAULT '',
    restart_count INTEGER NOT NULL DEFAULT 0
);

INSERT INTO agent_status (agent, status) VALUES ('experiment', 'idle');
INSERT INTO agent_status (agent, status) VALUES ('research', 'idle');
