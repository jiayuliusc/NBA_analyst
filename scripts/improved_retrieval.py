from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import ollama
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .query_understanding import QueryIntent, understand_query

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
    RealDictCursor = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "evidence_chunks.csv"

DB_CONFIG = {
    "dbname": "postgres",
    "user": "john",
    "password": "",
    "host": "localhost",
    "port": "5432",
}


@dataclass
class RetrievalConfig:
    metadata: bool = True
    dense: bool = True
    keyword: bool = True
    rerank: bool = True
    candidate_limit: int = 80


@dataclass
class EvidenceResult:
    chunk_id: str
    chunk_type: str
    text: str
    score: float
    player: str | None = None
    team: str | None = None
    opponent: str | None = None
    date: str | None = None
    season: str | None = None
    game_id: str | None = None
    source_type: str | None = None
    debug: dict | None = None


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


@lru_cache(maxsize=1)
def load_chunks() -> pd.DataFrame:
    if not CHUNKS_PATH.exists():
        from .chunking import build_chunks, load_player_logs

        chunks = build_chunks(load_player_logs())
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        chunks.to_csv(CHUNKS_PATH, index=False)
    df = pd.read_csv(CHUNKS_PATH, dtype={"chunk_id": str, "game_id": str})
    for column in ("player", "team", "opponent", "date", "season", "game_id", "source_type"):
        if column in df.columns:
            df[column] = df[column].where(df[column].notna(), None)
    return df


@lru_cache(maxsize=1)
def _tfidf_index():
    df = load_chunks()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(df["text"].fillna(""))
    return vectorizer, matrix


def _metadata_mask(df: pd.DataFrame, intent: QueryIntent) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if intent.players:
        mask &= df["player"].isin(intent.players) | df["text"].str.contains("|".join(map(re.escape, intent.players)), case=False, na=False)
    if intent.teams:
        team_pattern = "|".join(map(re.escape, intent.teams))
        if intent.intent == "team performance" and len(intent.teams) == 1:
            mask &= df["team"].fillna("").str.contains(team_pattern, case=False, regex=True)
        else:
            mask &= (
                df["team"].fillna("").str.contains(team_pattern, case=False, regex=True)
                | df["opponent"].fillna("").str.contains(team_pattern, case=False, regex=True)
                | df["text"].str.contains(team_pattern, case=False, regex=True, na=False)
            )
    if intent.date:
        mask &= df["date"].astype(str).eq(intent.date)
    if intent.season:
        mask &= df["season"].astype(str).eq(intent.season)
    if intent.chunk_types:
        mask &= df["chunk_type"].isin(intent.chunk_types)
    return mask


def _soft_metadata_pool(df: pd.DataFrame, intent: QueryIntent) -> pd.DataFrame:
    strict = df[_metadata_mask(df, intent)]
    if len(strict) >= 10:
        return strict

    mask = pd.Series(True, index=df.index)
    if intent.players:
        mask &= df["player"].isin(intent.players) | df["text"].str.contains("|".join(map(re.escape, intent.players)), case=False, na=False)
    if intent.teams and len(df[mask]) < 10:
        team_pattern = "|".join(map(re.escape, intent.teams))
        if intent.intent == "team performance" and len(intent.teams) == 1:
            mask |= df["team"].fillna("").str.contains(team_pattern, case=False, regex=True)
        else:
            mask |= (
                df["team"].fillna("").str.contains(team_pattern, case=False, regex=True)
                | df["opponent"].fillna("").str.contains(team_pattern, case=False, regex=True)
            )
    if intent.season and len(df[mask]) >= 10:
        mask &= df["season"].astype(str).eq(intent.season)
    if intent.chunk_types and len(df[mask]) >= 10:
        mask &= df["chunk_type"].isin(intent.chunk_types)
    pooled = df[mask]
    return pooled if len(pooled) else df


def _keyword_scores(query: str, candidate_idx: Iterable[int]) -> dict[int, float]:
    vectorizer, matrix = _tfidf_index()
    q_vec = vectorizer.transform([query])
    idx = list(candidate_idx)
    if not idx:
        return {}
    sims = cosine_similarity(q_vec, matrix[idx]).ravel()
    return {row_idx: float(score) for row_idx, score in zip(idx, sims)}


def _heuristic_dense_scores(query: str, candidate_idx: Iterable[int]) -> dict[int, float]:
    return _keyword_scores(query, candidate_idx)


def _metadata_boost(row: pd.Series, intent: QueryIntent) -> float:
    boost = 0.0
    text = str(row["text"]).lower()
    if row["chunk_type"] in intent.chunk_types:
        boost += 0.15
    if intent.players and row.get("player") in intent.players:
        boost += 0.35
    if intent.players and any(player.lower() in text for player in intent.players):
        boost += 0.15
    if intent.teams:
        for team in intent.teams:
            if team.lower() in str(row.get("team", "")).lower():
                boost += 0.25
                break
            if intent.intent != "team performance" and team.lower() in str(row.get("opponent", "")).lower():
                boost += 0.12
                break
    if intent.date and str(row.get("date")) == intent.date:
        boost += 0.3
    if intent.season and str(row.get("season")) == intent.season:
        boost += 0.2
    return boost


def retrieve_evidence(
    query: str,
    k: int = 10,
    config: RetrievalConfig | None = None,
    use_db: bool = False,
) -> list[EvidenceResult]:
    config = config or RetrievalConfig()
    intent = understand_query(query)
    if use_db and psycopg2 is not None:
        try:
            return _retrieve_from_db(query, intent, k, config)
        except Exception:
            pass

    df = load_chunks()
    pool = _soft_metadata_pool(df, intent) if config.metadata else df
    candidate_idx = list(pool.index)
    if config.keyword or config.dense:
        shared_scores = _keyword_scores(query, candidate_idx)
    else:
        shared_scores = {}
    keyword_scores = shared_scores if config.keyword else {}
    dense_scores = shared_scores if config.dense else {}

    if not config.metadata:
        base_scored = []
        for idx in candidate_idx:
            dense_score = dense_scores.get(idx, 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)
            score = (0.65 * dense_score) + (0.35 * keyword_score)
            base_scored.append((score, idx, {"dense": dense_score, "keyword": keyword_score, "intent": intent.intent}))
        base_scored.sort(key=lambda item: item[0], reverse=True)
        candidate_count = config.candidate_limit if config.rerank else k
        candidate_idx = [idx for _, idx, _ in base_scored[:candidate_count]]
        base_debug = {idx: (score, debug) for score, idx, debug in base_scored[:candidate_count]}
    else:
        base_debug = {}

    scored: list[tuple[float, int, dict]] = []
    for idx in candidate_idx:
        row = df.loc[idx]
        if idx in base_debug:
            score, debug = base_debug[idx]
            dense_score = debug["dense"]
            keyword_score = debug["keyword"]
        else:
            dense_score = dense_scores.get(idx, 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)
            score = (0.65 * dense_score) + (0.35 * keyword_score)
            debug = {"dense": dense_score, "keyword": keyword_score, "intent": intent.intent}
        if config.metadata:
            score += _metadata_boost(row, intent)
        if config.rerank:
            score += _rerank_boost(row, intent, query)
        scored.append((score, idx, debug))

    if len(scored) > config.candidate_limit and config.rerank:
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = scored[: config.candidate_limit]
    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[EvidenceResult] = []
    for score, idx, debug in scored[:k]:
        row = df.loc[idx]
        results.append(
            EvidenceResult(
                chunk_id=str(row["chunk_id"]),
                chunk_type=str(row["chunk_type"]),
                text=str(row["text"]),
                score=float(score),
                player=_clean(row.get("player")),
                team=_clean(row.get("team")),
                opponent=_clean(row.get("opponent")),
                date=_clean(row.get("date")),
                season=_clean(row.get("season")),
                game_id=_clean(row.get("game_id")),
                source_type=_clean(row.get("source_type")),
                debug=debug,
            )
        )
    return results


def _rerank_boost(row: pd.Series, intent: QueryIntent, query: str) -> float:
    lowered = query.lower()
    boost = 0.0
    if intent.intent == "season averages" and row["chunk_type"] == "season-summary":
        boost += 0.35
    if intent.intent == "player stats" and row["chunk_type"] == "player-game":
        boost += 0.25
    if intent.intent == "team performance" and row["chunk_type"] == "team-game":
        boost += 0.3
    if intent.intent == "matchup history" and row["chunk_type"] in {"game-level", "team-game"}:
        boost += 0.2
    if any(term in lowered for term in ("points", "scored", "pts")) and "points" in str(row["text"]).lower():
        boost += 0.08
    if any(term in lowered for term in ("rebounds", "boards")) and "rebounds" in str(row["text"]).lower():
        boost += 0.08
    if any(term in lowered for term in ("assists", "ast")) and "assists" in str(row["text"]).lower():
        boost += 0.08
    return boost


def _retrieve_from_db(query: str, intent: QueryIntent, k: int, config: RetrievalConfig) -> list[EvidenceResult]:
    query_vector = None
    if config.dense:
        query_vector = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

    where_parts = []
    params: list[object] = []
    if config.metadata:
        if intent.players:
            where_parts.append("(player = ANY(%s) OR text ILIKE ANY(%s))")
            params.extend([intent.players, [f"%{p}%" for p in intent.players]])
        if intent.teams:
            where_parts.append("(team = ANY(%s) OR opponent = ANY(%s) OR text ILIKE ANY(%s))")
            params.extend([intent.teams, intent.teams, [f"%{t}%" for t in intent.teams]])
        if intent.date:
            where_parts.append("date = %s")
            params.append(intent.date)
        if intent.season:
            where_parts.append("season = %s")
            params.append(intent.season)
        if intent.chunk_types:
            where_parts.append("chunk_type = ANY(%s)")
            params.append(intent.chunk_types)
    where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    if config.dense and config.keyword:
        sql = f"""
            SELECT *, (
                0.65 * (1 - (embedding <=> %s::vector)) +
                0.35 * ts_rank_cd(search_vector, plainto_tsquery('english', %s))
            ) AS score
            FROM nba_evidence_chunks
            {where_sql}
            ORDER BY score DESC
            LIMIT %s
        """
        params = [query_vector, query, *params, max(k, config.candidate_limit)]
    elif config.dense:
        sql = f"""
            SELECT *, (1 - (embedding <=> %s::vector)) AS score
            FROM nba_evidence_chunks
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [query_vector, *params, query_vector, max(k, config.candidate_limit)]
    else:
        sql = f"""
            SELECT *, ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS score
            FROM nba_evidence_chunks
            {where_sql}
            ORDER BY score DESC
            LIMIT %s
        """
        params = [query, *params, max(k, config.candidate_limit)]

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        return []
    if config.rerank:
        df["score"] = df.apply(lambda row: float(row["score"]) + _rerank_boost(row, intent, query) + _metadata_boost(row, intent), axis=1)
        df = df.sort_values("score", ascending=False)
    return [
        EvidenceResult(
            chunk_id=str(row["chunk_id"]),
            chunk_type=str(row["chunk_type"]),
            text=str(row["text"]),
            score=float(row["score"]),
            player=_clean(row.get("player")),
            team=_clean(row.get("team")),
            opponent=_clean(row.get("opponent")),
            date=_clean(row.get("date")),
            season=_clean(row.get("season")),
            game_id=_clean(row.get("game_id")),
            source_type=_clean(row.get("source_type")),
        )
        for _, row in df.head(k).iterrows()
    ]


def results_as_dicts(results: list[EvidenceResult]) -> list[dict]:
    return [asdict(result) for result in results]
