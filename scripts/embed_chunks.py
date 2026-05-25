from __future__ import annotations

import ollama

try:
    import psycopg2
except Exception as exc:
    raise SystemExit("psycopg2 is required for database embedding. Install psycopg2-binary first.") from exc


DB_CONFIG = {
    "dbname": "postgres",
    "user": "john",
    "password": "",
    "host": "localhost",
    "port": "5432",
}

BATCH_SIZE = 50


def embed_chunks() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            while True:
                cur.execute(
                    "SELECT chunk_id, text FROM nba_evidence_chunks WHERE embedding IS NULL LIMIT %s",
                    (BATCH_SIZE,),
                )
                rows = cur.fetchall()
                if not rows:
                    print("All evidence chunks are embedded.")
                    break
                for chunk_id, text in rows:
                    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
                    cur.execute(
                        "UPDATE nba_evidence_chunks SET embedding = %s WHERE chunk_id = %s",
                        (response["embedding"], chunk_id),
                    )
                conn.commit()
                print(f"Embedded {len(rows)} chunks.")


if __name__ == "__main__":
    embed_chunks()
