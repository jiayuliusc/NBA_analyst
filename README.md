# NBA RAG Project

## Retrieval Redesign

The original baseline stores player game logs in one flat pgvector table and retrieves with naive top-k vector search. That fails for many NBA questions because the search space mixes player rows, team questions, season-average questions, recent-game questions, and matchup questions without enough metadata control.

The improved pipeline adds evidence-unit chunking and metadata-aware retrieval:

- `player-game`: one player box-score line from one game.
- `team-game`: one team rollup for one game with totals and top contributors.
- `game-level`: one full game rollup with leading lines from both teams.
- `season-summary`: one player/team/season average chunk.
- `injury/transaction/event`: event-ready chunk type for future feeds.

Every evidence chunk carries `player`, `team`, `opponent`, `date`, `season`, `game_id`, `source_type`, and `chunk_type`. Retrieval now combines NBA query understanding, metadata filtering, dense scoring, keyword/BM25-style fallback, and an intent reranker.

## Current Evaluation

Run:

```bash
python3 -m scripts.chunking
python3 -m scripts.evaluate_retrieval --write-eval
```

Latest local report: `reports/retrieval_ablation_report.md`

| system | recall@5 | recall@10 | MRR | nDCG@10 |
|:--|--:|--:|--:|--:|
| current baseline | 0.286 | 0.286 | 0.286 | 0.286 |
| baseline + metadata filtering | 0.429 | 0.429 | 0.429 | 0.429 |
| baseline + hybrid retrieval | 0.429 | 0.429 | 0.429 | 0.429 |
| baseline + reranker | 0.357 | 0.357 | 0.357 | 0.341 |
| full system | 0.786 | 0.786 | 0.857 | 0.784 |

The ablation report makes the main effect clear: metadata-aware candidate control and intent reranking produce the material recall improvement over flat top-k search.

## Demo UI

Run:

```bash
python3 -m scripts.demo_app
```

Open `http://127.0.0.1:8000` and compare retrieval modes:

- `baseline`
- `metadata`
- `hybrid`
- `reranker`
- `full`

The UI shows the understood intent, metadata filters, evidence scores, and retrieved chunks so it is easy to inspect why an answer is grounded.

## PostgreSQL + pgvector Setup

The project still supports PostgreSQL + pgvector. The schema now includes the original `nba_game_logs` table and a new `nba_evidence_chunks` table with metadata columns, full-text search, and vector embeddings.

```bash
python3 -m scripts.setup_db
python3 -m scripts.ingest_chunks
python3 -m scripts.embed_chunks
```

The retriever can use the database path via `retrieve_evidence(..., use_db=True)`. If Postgres or `psycopg2` is unavailable, scripts fall back to the local evidence CSV and TF-IDF scoring for demo/evaluation.

## Legacy Setup Notes

# Project Setup Guide

This guide provides all the necessary steps to set up your PostgreSQL 16 database and configure the `pgvector` extension for your project.

---

## Prerequisites

Before starting, ensure you have [Homebrew](https://brew.sh/) installed on your macOS machine.

---

## Setup Instructions

### 1. Install PostgreSQL 16

First, you need to install PostgreSQL 16 using Homebrew. Run the following command in your terminal:

```bash
brew install postgresql@16
```

### 2. Install the `pgvector` Extension

```bash
brew install pgvector
```

### 3. Start PostgreSQL Database Service

```bash
brew services start postgresql@16
```

### 4. Access the PostgreSQL Command Line Interface

```bash
psql postgres
```

### 5. Connect to Your Database

```bash
\c nbadb
```

### 6. Enable the pgvector Extension

```bash
CREATE EXTENSION vector;
```

### 7. Exit PostgreSQL CLI

```bash
\q
```
