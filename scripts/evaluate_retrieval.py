from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .improved_retrieval import CHUNKS_PATH, RetrievalConfig, retrieve_evidence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "data" / "retrieval_eval.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_ablation_report.md"
DEFAULT_EVAL_SIZE = 60


@dataclass
class EvalQuestion:
    question: str
    relevant_chunk_ids: list[str]
    category: str
    note: str


CONFIGS = {
    "current baseline": RetrievalConfig(metadata=False, dense=True, keyword=False, rerank=False),
    "baseline + metadata filtering": RetrievalConfig(metadata=True, dense=True, keyword=False, rerank=False),
    "baseline + hybrid retrieval": RetrievalConfig(metadata=True, dense=True, keyword=True, rerank=False),
    "baseline + reranker": RetrievalConfig(metadata=False, dense=True, keyword=False, rerank=True),
    "full system": RetrievalConfig(metadata=True, dense=True, keyword=True, rerank=True),
}

ABLATIONS = {
    "full system": RetrievalConfig(metadata=True, dense=True, keyword=True, rerank=True),
    "no metadata": RetrievalConfig(metadata=False, dense=True, keyword=True, rerank=True),
    "no keyword fallback": RetrievalConfig(metadata=True, dense=True, keyword=False, rerank=True),
    "no dense score": RetrievalConfig(metadata=True, dense=False, keyword=True, rerank=True),
    "no reranker": RetrievalConfig(metadata=True, dense=True, keyword=True, rerank=False),
}


def _config_key(config: RetrievalConfig) -> tuple[bool, bool, bool, bool, int]:
    return (config.metadata, config.dense, config.keyword, config.rerank, config.candidate_limit)


def _choose_player(df: pd.DataFrame, preferred: list[str]) -> str:
    available = set(df["player"].dropna())
    for player in preferred:
        if player in available:
            return player
    return str(df["player"].value_counts().index[0])


def _add_unique(questions: list[EvalQuestion], seen: set[str], item: EvalQuestion) -> None:
    if item.question not in seen and item.relevant_chunk_ids:
        questions.append(item)
        seen.add(item.question)


def build_eval_set(chunks: pd.DataFrame, target_size: int = DEFAULT_EVAL_SIZE) -> list[EvalQuestion]:
    player_chunks = chunks[chunks["chunk_type"] == "player-game"].copy()
    team_chunks = chunks[chunks["chunk_type"] == "team-game"].copy()
    game_chunks = chunks[chunks["chunk_type"] == "game-level"].copy()
    season_chunks = chunks[chunks["chunk_type"] == "season-summary"].copy()

    latest_date = str(player_chunks["date"].max())
    questions: list[EvalQuestion] = []
    seen: set[str] = set()

    player_counts = player_chunks["player"].value_counts()
    frequent_players = [p for p in player_counts.index if player_counts[p] >= 12][:24]
    preferred = ["LeBron James", "Anthony Edwards", "Stephen Curry", "Luka Doncic", "Nikola Jokic", "Jayson Tatum"]
    player_order = []
    for player in preferred + frequent_players:
        if player in set(player_chunks["player"]) and player not in player_order:
            player_order.append(player)

    for player in player_order[:15]:
        latest = player_chunks[player_chunks["player"] == player].sort_values(["date", "game_id"]).iloc[-1]
        _add_unique(
            questions,
            seen,
            EvalQuestion(
                question=f"How did {player} play in his last game?",
                relevant_chunk_ids=[str(latest["chunk_id"])],
                category="player stats",
                note="Player + recency query; should resolve to the latest player-game evidence.",
            ),
        )

    for _, row in season_chunks[season_chunks["player"].isin(player_order)].head(15).iterrows():
        _add_unique(
            questions,
            seen,
            EvalQuestion(
                question=f"What were {row['player']}'s {row['season']} season averages?",
                relevant_chunk_ids=[str(row["chunk_id"])],
                category="season averages",
                note="Requires season-summary evidence instead of arbitrary player-game rows.",
            ),
        )

    latest_team_chunks = team_chunks[team_chunks["date"] == latest_date].sort_values(["team", "game_id"]).head(12)
    for _, row in latest_team_chunks.iterrows():
        _add_unique(
            questions,
            seen,
            EvalQuestion(
                question=f"How did the {row['team']} perform last night?",
                relevant_chunk_ids=[str(row["chunk_id"])],
                category="team performance",
                note="Generic recency wording should resolve to latest-date team-game chunks.",
            ),
        )

    sampled_games = game_chunks.sort_values(["date", "game_id"], ascending=[False, True]).head(12)
    for _, game in sampled_games.iterrows():
        teams = str(game["team"]).split(";") if pd.notna(game["team"]) else []
        if len(teams) < 2:
            continue
        related_team_chunks = team_chunks[team_chunks["game_id"].astype(str) == str(game["game_id"])]["chunk_id"].astype(str).head(2).tolist()
        _add_unique(
            questions,
            seen,
            EvalQuestion(
                question=f"What happened in the {teams[0]} versus {teams[1]} matchup on {game['date']}?",
                relevant_chunk_ids=[str(game["chunk_id"]), *related_team_chunks],
                category="matchup history",
                note="Matchup wording should favor game-level and team-game evidence.",
            ),
        )

    exact_rows = player_chunks.sort_values(["date", "player"], ascending=[False, True]).head(20)
    for _, row in exact_rows.iterrows():
        _add_unique(
            questions,
            seen,
            EvalQuestion(
                question=f"Show me {row['player']}'s stats against the {row['opponent']} on {row['date']}.",
                relevant_chunk_ids=[str(row["chunk_id"])],
                category="player stats",
                note="Exact player/date/opponent metadata should put the right row near the top.",
            ),
        )

    if len(questions) < target_size:
        recent_games = game_chunks.sort_values(["date", "game_id"], ascending=[False, True]).head(target_size - len(questions))
        for _, game in recent_games.iterrows():
            _add_unique(
                questions,
                seen,
                EvalQuestion(
                    question=f"What happened in game {game['game_id']}?",
                    relevant_chunk_ids=[str(game["chunk_id"])],
                    category="recent games",
                    note="Game-id query should retrieve the game-level evidence unit.",
                ),
            )
    return questions[:target_size]


def write_eval_set(path: Path = EVAL_PATH, target_size: int = DEFAULT_EVAL_SIZE) -> list[EvalQuestion]:
    chunks = pd.read_csv(CHUNKS_PATH, dtype={"chunk_id": str, "game_id": str})
    questions = build_eval_set(chunks, target_size=target_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.__dict__) + "\n")
    return questions


def load_eval_set(path: Path = EVAL_PATH) -> list[EvalQuestion]:
    if not path.exists():
        return write_eval_set(path)
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                questions.append(EvalQuestion(**json.loads(line)))
    return questions


def recall_at(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for idx, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1 / idx
    return 0.0


def ndcg_at(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = 0.0
    for idx, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            dcg += 1 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(
    configs: dict[str, RetrievalConfig],
    questions: list[EvalQuestion],
    retrieval_cache: dict[tuple[str, tuple[bool, bool, bool, bool, int]], list[str | object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    retrieval_cache = retrieval_cache if retrieval_cache is not None else {}
    rows = []
    examples: dict[str, list[dict]] = {}
    for name, config in configs.items():
        per_query = []
        for item in questions:
            cache_key = (item.question, _config_key(config))
            if cache_key not in retrieval_cache:
                retrieval_cache[cache_key] = retrieve_evidence(item.question, k=10, config=config, use_db=False)
            results = retrieval_cache[cache_key]
            retrieved = [result.chunk_id for result in results]
            relevant = set(item.relevant_chunk_ids)
            record = {
                "question": item.question,
                "category": item.category,
                "recall@3": recall_at(retrieved, relevant, 3),
                "recall@5": recall_at(retrieved, relevant, 5),
                "recall@10": recall_at(retrieved, relevant, 10),
                "MRR": mrr(retrieved, relevant),
                "nDCG@10": ndcg_at(retrieved, relevant, 10),
                "top_result": results[0].text if results else "",
                "hit": bool(set(retrieved[:10]) & relevant),
            }
            per_query.append(record)
        examples[name] = per_query
        frame = pd.DataFrame(per_query)
        rows.append(
            {
                "system": name,
                "recall@3": frame["recall@3"].mean(),
                "recall@5": frame["recall@5"].mean(),
                "recall@10": frame["recall@10"].mean(),
                "MRR": frame["MRR"].mean(),
                "nDCG@10": frame["nDCG@10"].mean(),
            }
        )
    return pd.DataFrame(rows), examples


def context_budget_analysis(questions: list[EvalQuestion]) -> pd.DataFrame:
    rows = []
    for item in questions:
        results = retrieve_evidence(item.question, k=10, config=CONFIGS["full system"], use_db=False)
        retrieved = [result.chunk_id for result in results]
        relevant = set(item.relevant_chunk_ids)
        first_rank = next((idx for idx, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant), None)
        rows.append(
            {
                "question": item.question,
                "category": item.category,
                "first_relevant_rank": first_rank,
                "recall@3": recall_at(retrieved, relevant, 3),
                "recall@5": recall_at(retrieved, relevant, 5),
                "recall@10": recall_at(retrieved, relevant, 10),
                "middle_hit_rank_4_to_7": bool(first_rank and 4 <= first_rank <= 7),
            }
        )
    frame = pd.DataFrame(rows)
    summary = []
    for category, group in [("all", frame), *frame.groupby("category")]:
        summary.append(
            {
                "category": category,
                "queries": len(group),
                "recall@3": group["recall@3"].mean(),
                "recall@5": group["recall@5"].mean(),
                "recall@10": group["recall@10"].mean(),
                "first_rank<=3": group["first_relevant_rank"].le(3).mean(),
                "middle_hit_4-7": group["middle_hit_rank_4_to_7"].mean(),
            }
        )
    return pd.DataFrame(summary)


def _markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{value:.3f}")
    return table.to_markdown(index=False)


def write_report(
    metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    context_budget: pd.DataFrame,
    examples: dict[str, list[dict]],
    eval_size: int,
    path: Path = REPORT_PATH,
) -> None:
    baseline_examples = examples.get("current baseline", [])
    full_examples = examples.get("full system", [])
    qualitative = []
    for before, after in zip(baseline_examples, full_examples):
        if not before["hit"] and after["hit"]:
            qualitative.append((before, after))
        if len(qualitative) >= 2:
            break

    lines = [
        "# NBA Retrieval Ablation Report",
        "",
        "## Diagnosis",
        "",
        "The old pipeline embeds one flat table of player-game rows and orders by vector distance. That loses high-signal NBA structure: player, team, opponent, game date, season, game id, and evidence type. Generic questions such as latest matchup, last night, team performance, or season averages have no reliable way to restrict the candidate pool, so top-k search can return semantically plausible but wrong rows.",
        "",
        "## Upgrades",
        "",
        "- Evidence-unit chunking: player-game, team-game, game-level, season-summary, and event-ready chunks.",
        "- Metadata-aware query understanding for player stats, team performance, matchup history, recent games, season averages, and injuries/transactions.",
        "- Hybrid retrieval that combines metadata filters, dense scoring, keyword scoring, and a small intent reranker.",
        "- Offline evaluator that runs without Postgres/Ollama, plus database ingestion scripts for pgvector deployment.",
        "",
        "## Evaluation Set",
        "",
        f"The current labeled evaluation set contains {eval_size} generated-but-deterministic questions sampled from the local NBA evidence index. It covers player stats, season averages, team performance, matchup history, recent games, and exact player/date/opponent lookups. This is still not a substitute for human-labeled production queries, but it avoids over-reading an 8-question smoke test.",
        "",
        "## Baseline Comparison",
        "",
        _markdown_table(metrics),
        "",
        "## One-Component Ablations",
        "",
        _markdown_table(ablations),
        "",
        "## Context Budget and Lost-in-the-Middle Check",
        "",
        "This checks whether useful evidence is appearing only in ranks 6-10, where generation can become more expensive and more vulnerable to lost-in-the-middle behavior. When recall@5 and recall@10 are similar, the generator can usually receive fewer chunks.",
        "",
        _markdown_table(context_budget),
        "",
        "Recommendation: start generation with the top 5 chunks. Use top 3 for lower-cost demo answers when recall@3 is close to recall@5 for the target query type, and reserve top 10 for debugging or broad matchup/summary questions.",
        "",
        "## Qualitative Wins",
        "",
    ]
    if qualitative:
        for before, after in qualitative:
            lines.extend(
                [
                    f"Question: {before['question']}",
                    f"- Baseline top result: {before['top_result'][:350]}",
                    f"- Full-system top result: {after['top_result'][:350]}",
                    "",
                ]
            )
    else:
        lines.append("The generated eval set did not produce a clean baseline-miss/full-hit pair; inspect per-query JSONL output for finer-grained differences.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NBA retrieval ablations.")
    parser.add_argument("--write-eval", action="store_true", help="Regenerate the labeled eval JSONL.")
    parser.add_argument("--target-size", type=int, default=DEFAULT_EVAL_SIZE, help="Number of generated eval questions.")
    args = parser.parse_args()

    questions = write_eval_set(target_size=args.target_size) if args.write_eval else load_eval_set()
    retrieval_cache: dict[tuple[str, tuple[bool, bool, bool, bool, int]], list[object]] = {}
    metrics, examples = evaluate(CONFIGS, questions, retrieval_cache)
    ablations, _ = evaluate(ABLATIONS, questions, retrieval_cache)
    context_budget = context_budget_analysis(questions)
    write_report(metrics, ablations, context_budget, examples, len(questions))

    print("Baseline comparison")
    print(metrics.to_string(index=False))
    print("\nOne-component ablations")
    print(ablations.to_string(index=False))
    print("\nContext budget / lost-in-the-middle check")
    print(context_budget.to_string(index=False))
    print(f"\nWrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
