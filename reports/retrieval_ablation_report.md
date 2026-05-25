# NBA Retrieval Ablation Report

## Diagnosis

The old pipeline embeds one flat table of player-game rows and orders by vector distance. That loses high-signal NBA structure: player, team, opponent, game date, season, game id, and evidence type. Generic questions such as latest matchup, last night, team performance, or season averages have no reliable way to restrict the candidate pool, so top-k search can return semantically plausible but wrong rows.

## Upgrades

- Evidence-unit chunking: player-game, team-game, game-level, season-summary, and event-ready chunks.
- Metadata-aware query understanding for player stats, team performance, matchup history, recent games, season averages, and injuries/transactions.
- Hybrid retrieval that combines metadata filters, dense scoring, keyword scoring, and a small intent reranker.
- Offline evaluator that runs without Postgres/Ollama, plus database ingestion scripts for pgvector deployment.

## Evaluation Set

The current labeled evaluation set contains 60 generated-but-deterministic questions sampled from the local NBA evidence index. It covers player stats, season averages, team performance, matchup history, recent games, and exact player/date/opponent lookups. This is still not a substitute for human-labeled production queries, but it avoids over-reading an 8-question smoke test.

## Baseline Comparison

| system                        |   recall@3 |   recall@5 |   recall@10 |   MRR |   nDCG@10 |
|:------------------------------|-----------:|-----------:|------------:|------:|----------:|
| current baseline              |      0.367 |      0.367 |       0.383 | 0.358 |     0.364 |
| baseline + metadata filtering |      0.517 |      0.517 |       0.533 | 0.508 |     0.514 |
| baseline + hybrid retrieval   |      0.517 |      0.517 |       0.533 | 0.508 |     0.514 |
| baseline + reranker           |      0.367 |      0.367 |       0.383 | 0.342 |     0.352 |
| full system                   |      0.833 |      0.833 |       0.867 | 0.892 |     0.854 |

## One-Component Ablations

| system              |   recall@3 |   recall@5 |   recall@10 |   MRR |   nDCG@10 |
|:--------------------|-----------:|-----------:|------------:|------:|----------:|
| full system         |      0.833 |      0.833 |       0.867 | 0.892 |     0.854 |
| no metadata         |      0.367 |      0.367 |       0.383 | 0.358 |     0.364 |
| no keyword fallback |      0.85  |      0.85  |       0.867 | 0.908 |     0.867 |
| no dense score      |      0.9   |      0.9   |       0.928 | 0.908 |     0.907 |
| no reranker         |      0.517 |      0.517 |       0.533 | 0.508 |     0.514 |

## Context Budget and Lost-in-the-Middle Check

This checks whether useful evidence is appearing only in ranks 6-10, where generation can become more expensive and more vulnerable to lost-in-the-middle behavior. When recall@5 and recall@10 are similar, the generator can usually receive fewer chunks.

| category         |   queries |   recall@3 |   recall@5 |   recall@10 |   first_rank<=3 |   middle_hit_4-7 |
|:-----------------|----------:|-----------:|-----------:|------------:|----------------:|-----------------:|
| all              |        60 |      0.833 |      0.833 |       0.867 |           0.917 |            0.017 |
| matchup history  |        12 |      0.583 |      0.583 |       0.667 |           1     |            0     |
| player stats     |        21 |      0.762 |      0.762 |       0.81  |           0.762 |            0.048 |
| season averages  |        15 |      1     |      1     |       1     |           1     |            0     |
| team performance |        12 |      1     |      1     |       1     |           1     |            0     |

Recommendation: start generation with the top 5 chunks. Use top 3 for lower-cost demo answers when recall@3 is close to recall@5 for the target query type, and reserve top 10 for debugging or broad matchup/summary questions.

## Qualitative Wins

Question: How did LeBron James play in his last game?
- Baseline top result: Player-game evidence. LeBron James played for Lakers against Suns on 2025-10-14 (LAL vs. PHX, game 12500060). He recorded 0 points, 0 rebounds, 0 assists, 0 steals, and 0 blocks. Result: Loss.
- Full-system top result: Player-game evidence. LeBron James played for Lakers against Grizzlies on 2026-01-04 (LAL vs. MEM, game 22500501). He recorded 26 points, 7 rebounds, 10 assists, 1 steals, and 0 blocks. Result: Win.

Question: How did Anthony Edwards play in his last game?
- Baseline top result: Player-game evidence. Anthony Edwards played for Timberwolves against Lakers on 2024-12-02 (MIN vs. LAL, game 22400318). He recorded 8 points, 7 rebounds, 3 assists, 2 steals, and 0 blocks. Result: Win.
- Full-system top result: Player-game evidence. Anthony Edwards played for Timberwolves against Wizards on 2026-01-04 (MIN vs. WAS, game 22500498). He recorded 35 points, 6 rebounds, 3 assists, 4 steals, and 0 blocks. Result: Win.
