CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS nba_game_logs (
  id BIGSERIAL PRIMARY KEY,
  game_date date NOT NULL,
  matchup text NOT NULL,
  player_name text NOT NULL,
  stats_summary text NOT NULL,
  search_text TEXT,
  embedding vector(768)
);

CREATE TABLE IF NOT EXISTS nba_evidence_chunks (
  chunk_id text PRIMARY KEY,
  chunk_type text NOT NULL,
  text text NOT NULL,
  player text,
  team text,
  opponent text,
  date date,
  season text,
  game_id text,
  source_type text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  embedding vector(768)
);

CREATE INDEX IF NOT EXISTS idx_nba_evidence_player ON nba_evidence_chunks (player);
CREATE INDEX IF NOT EXISTS idx_nba_evidence_team ON nba_evidence_chunks (team);
CREATE INDEX IF NOT EXISTS idx_nba_evidence_opponent ON nba_evidence_chunks (opponent);
CREATE INDEX IF NOT EXISTS idx_nba_evidence_date ON nba_evidence_chunks (date);
CREATE INDEX IF NOT EXISTS idx_nba_evidence_season ON nba_evidence_chunks (season);
CREATE INDEX IF NOT EXISTS idx_nba_evidence_chunk_type ON nba_evidence_chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_nba_evidence_search ON nba_evidence_chunks USING gin (search_vector);
