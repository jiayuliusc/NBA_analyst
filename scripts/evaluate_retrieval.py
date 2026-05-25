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


def _choose_player(df: pd.DataFrame, preferred: list[str]) -> str:
    available = set(df["player"].dropna())
    for player in preferred:
        if player in available:
            return player
    return str(df["player"].value_counts().index[0])


def build_eval_set(chunks: pd.DataFrame) -> list[EvalQuestion]:
    player_chunks = chunks[chunks["chunk_type"] == "player-game"].copy()
    team_chunks = chunks[chunks["chunk_type"] == "team-game"].copy()
    game_chunks = chunks[chunks["chunk_type"] == "game-level"].copy()
    season_chunks = chunks[chunks["chunk_type"] == "season-summary"].copy()

    latest_date = str(player_chunks["date"].max())
    latest_players = player_chunks[player_chunks["date"] == latest_date].sort_values("text")
    star = _choose_player(player_chunks, ["LeBron James", "Anthony Edwards", "Stephen Curry", "Luka Doncic"])
    star_latest = player_chunks[player_chunks["player"] == star].sort_values("date").iloc[-1]
    star_season = season_chunks[season_chunks["player"] == star].iloc[-1]
    latest_team = str(team_chunks[team_chunks["date"] == latest_date].iloc[0]["team"])
    latest_team_chunk = team_chunks[(team_chunks["date"] == latest_date) & (team_chunks["team"] == latest_team)].iloc[0]
    latest_game = game_chunks[game_chunks["date"] == latest_date].iloc[0]
    latest_player = latest_players.iloc[0]
    opponent = str(latest_player["opponent"])

    questions = [
        EvalQuestion(
            question=f"How did {star} play in his last game?",
            relevant_chunk_ids=[str(star_latest["chunk_id"])],
            category="player stats",
            note="Requires player + recency metadata, not just semantic similarity.",
        ),
        EvalQuestion(
            question=f"What were {star}'s season averages?",
            relevant_chunk_ids=[str(star_season["chunk_id"])],
            category="season averages",
            note="The answer lives in a season-summary evidence unit.",
        ),
        EvalQuestion(
            question=f"How did the {latest_team} perform last night?",
            relevant_chunk_ids=[str(latest_team_chunk["chunk_id"])],
            category="team performance",
            note="Generic wording needs latest-date interpretation and team-game chunks.",
        ),
        EvalQuestion(
            question="What happened in the latest matchup?",
            relevant_chunk_ids=[str(latest_game["chunk_id"])],
            category="recent games",
            note="No player name appears, so flat player-row search usually drifts.",
        ),
        EvalQuestion(
            question=f"Show me player stats from {latest_player['player']} against the {opponent} on {latest_date}.",
            relevant_chunk_ids=[str(latest_player["chunk_id"])],
            category="player stats",
            note="Exact player/date/opponent metadata should put the right row near the top.",
        ),
        EvalQuestion(
            question=f"Who led the {latest_team} game on {latest_date}?",
            relevant_chunk_ids=[str(latest_team_chunk["chunk_id"]), str(latest_game["chunk_id"])],
            category="team performance",
            note="Team-game and game-level chunks are both useful evidence.",
        ),
        EvalQuestion(
            question=f"Give me the {latest_team} versus {opponent} matchup history from the recent game.",
            relevant_chunk_ids=[str(latest_game["chunk_id"]), str(latest_team_chunk["chunk_id"])],
            category="matchup history",
            note="Matchup intent should favor game-level and team-game evidence.",
        ),
        EvalQuestion(
            question="Any injury or transaction evidence for the latest games?",
            relevant_chunk_ids=list(chunks[chunks["chunk_type"] == "injury/transaction/event"]["chunk_id"].astype(str).head(3)),
            category="injuries / transactions",
            note="This dataset has no real injury feed; a good retriever should expose that evidence is absent.",
        ),
    ]
    return [q for q in questions if q.relevant_chunk_ids]


def write_eval_set(path: Path = EVAL_PATH) -> list[EvalQuestion]:
    chunks = pd.read_csv(CHUNKS_PATH, dtype={"chunk_id": str, "game_id": str})
    questions = build_eval_set(chunks)
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


def evaluate(configs: dict[str, RetrievalConfig], questions: list[EvalQuestion]) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    rows = []
    examples: dict[str, list[dict]] = {}
    for name, config in configs.items():
        per_query = []
        for item in questions:
            results = retrieve_evidence(item.question, k=10, config=config, use_db=False)
            retrieved = [result.chunk_id for result in results]
            relevant = set(item.relevant_chunk_ids)
            record = {
                "question": item.question,
                "category": item.category,
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
                "recall@5": frame["recall@5"].mean(),
                "recall@10": frame["recall@10"].mean(),
                "MRR": frame["MRR"].mean(),
                "nDCG@10": frame["nDCG@10"].mean(),
            }
        )
    return pd.DataFrame(rows), examples


def _markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    for column in table.columns:
        if column != "system":
            table[column] = table[column].map(lambda value: f"{value:.3f}")
    return table.to_markdown(index=False)


def write_report(metrics: pd.DataFrame, ablations: pd.DataFrame, examples: dict[str, list[dict]], path: Path = REPORT_PATH) -> None:
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
        "## Baseline Comparison",
        "",
        _markdown_table(metrics),
        "",
        "## One-Component Ablations",
        "",
        _markdown_table(ablations),
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
    args = parser.parse_args()

    questions = write_eval_set() if args.write_eval else load_eval_set()
    metrics, examples = evaluate(CONFIGS, questions)
    ablations, _ = evaluate(ABLATIONS, questions)
    write_report(metrics, ablations, examples)

    print("Baseline comparison")
    print(metrics.to_string(index=False))
    print("\nOne-component ablations")
    print(ablations.to_string(index=False))
    print(f"\nWrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
