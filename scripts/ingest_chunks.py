from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import execute_values
except Exception as exc:
    raise SystemExit("psycopg2 is required for database ingestion. Install psycopg2-binary first.") from exc

from .chunking import DEFAULT_OUTPUT, build_chunks, load_player_logs


DB_CONFIG = {
    "dbname": "postgres",
    "user": "john",
    "password": "",
    "host": "localhost",
    "port": "5432",
}


def ensure_chunks(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build_chunks(load_player_logs()).to_csv(path, index=False)
    return pd.read_csv(path)


def ingest_chunks(path: Path = DEFAULT_OUTPUT) -> None:
    df = ensure_chunks(path)
    rows = []
    for row in df.itertuples(index=False):
        metadata = json.loads(row.metadata_json) if isinstance(row.metadata_json, str) else {}
        rows.append(
            (
                row.chunk_id,
                row.chunk_type,
                row.text,
                None if pd.isna(row.player) else row.player,
                None if pd.isna(row.team) else row.team,
                None if pd.isna(row.opponent) else row.opponent,
                None if pd.isna(row.date) else row.date,
                None if pd.isna(row.season) else row.season,
                None if pd.isna(row.game_id) else str(row.game_id),
                row.source_type,
                json.dumps(metadata),
            )
        )

    sql = """
        INSERT INTO nba_evidence_chunks
        (chunk_id, chunk_type, text, player, team, opponent, date, season, game_id, source_type, metadata)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
          chunk_type = EXCLUDED.chunk_type,
          text = EXCLUDED.text,
          player = EXCLUDED.player,
          team = EXCLUDED.team,
          opponent = EXCLUDED.opponent,
          date = EXCLUDED.date,
          season = EXCLUDED.season,
          game_id = EXCLUDED.game_id,
          source_type = EXCLUDED.source_type,
          metadata = EXCLUDED.metadata
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
    print(f"Upserted {len(rows)} evidence chunks into nba_evidence_chunks.")


if __name__ == "__main__":
    ingest_chunks()
