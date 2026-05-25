# NBA Retrieval Ablation Report

## Diagnosis

The old pipeline embeds one flat table of player-game rows and orders by vector distance. That loses high-signal NBA structure: player, team, opponent, game date, season, game id, and evidence type. Generic questions such as latest matchup, last night, team performance, or season averages have no reliable way to restrict the candidate pool, so top-k search can return semantically plausible but wrong rows.

## Upgrades

- Evidence-unit chunking: player-game, team-game, game-level, season-summary, and event-ready chunks.
- Metadata-aware query understanding for player stats, team performance, matchup history, recent games, season averages, and injuries/transactions.
- Hybrid retrieval that combines metadata filters, dense scoring, keyword scoring, and a small intent reranker.
- Offline evaluator that runs without Postgres/Ollama, plus database ingestion scripts for pgvector deployment.

## Baseline Comparison

| system                        |   recall@5 |   recall@10 |   MRR |   nDCG@10 |
|:------------------------------|-----------:|------------:|------:|----------:|
| current baseline              |      0.286 |       0.286 | 0.286 |     0.286 |
| baseline + metadata filtering |      0.429 |       0.429 | 0.429 |     0.429 |
| baseline + hybrid retrieval   |      0.429 |       0.429 | 0.429 |     0.429 |
| baseline + reranker           |      0.357 |       0.357 | 0.357 |     0.341 |
| full system                   |      0.786 |       0.786 | 0.857 |     0.784 |

## One-Component Ablations

| system              |   recall@5 |   recall@10 |   MRR |   nDCG@10 |
|:--------------------|-----------:|------------:|------:|----------:|
| full system         |      0.786 |       0.786 | 0.857 |     0.784 |
| no metadata         |      0.357 |       0.357 | 0.357 |     0.341 |
| no keyword fallback |      0.786 |       0.786 | 0.857 |     0.784 |
| no dense score      |      0.786 |       0.786 | 0.857 |     0.784 |
| no reranker         |      0.429 |       0.429 | 0.429 |     0.429 |

## Qualitative Wins

Question: How did LeBron James play in his last game?
- Baseline top result: Player-game evidence. LeBron James played for Lakers against Suns on 2025-10-14 (LAL vs. PHX, game 12500060). He recorded 0 points, 0 rebounds, 0 assists, 0 steals, and 0 blocks. Result: Loss.
- Full-system top result: Player-game evidence. LeBron James played for Lakers against Grizzlies on 2026-01-04 (LAL vs. MEM, game 22500501). He recorded 26 points, 7 rebounds, 10 assists, 1 steals, and 0 blocks. Result: Win.

Question: How did the Cavaliers perform last night?
- Baseline top result: Game-level evidence. Game 22400188 on 2024-11-08 in season 2024-25: Cavaliers vs. Warriors. Leading player lines: Darius Garland (Cavaliers): 27 PTS, 3 REB, 6 AST; Evan Mobley (Cavaliers): 23 PTS, 4 REB, 4 AST; Jonathan Kuminga (Warriors): 21 PTS, 5 REB, 5 AST; Ty Jerome (Cavaliers): 20 PTS, 1 REB, 3 AST; Isaac Okoro (Cavaliers): 16 PTS, 1 REB, 4 A
- Full-system top result: Team-game evidence. Cavaliers faced Pistons on 2026-01-04 (CLE vs. DET, game 22500494, season 2025-26). Team totals in logged player rows: 110 points, 44 rebounds, 23 assists, 11 steals, 10 blocks. Top contributors: Donovan Mitchell: 30 PTS, 6 REB, 3 AST; Darius Garland: 16 PTS, 2 REB, 6 AST; Evan Mobley: 15 PTS, 4 REB, 5 AST; Sam Merrill: 15 PTS, 
