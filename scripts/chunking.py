from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "cleaned_player_statistics.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evidence_chunks.csv"


SUMMARY_RE = re.compile(
    r"Player:\s*(?P<player>.*?)\.\s*"
    r"Team:\s*(?P<team>.*?)\.\s*"
    r"Opponent:\s*(?P<opponent>.*?)\.\s*"
    r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})\.\s*"
    r"Stats:\s*(?P<points>[-\d.]+)\s*PTS,\s*(?P<rebounds>[-\d.]+)\s*REB,\s*(?P<assists>[-\d.]+)\s*AST\.\s*"
    r"Other:\s*(?P<steals>[-\d.]+)\s*STL,\s*(?P<blocks>[-\d.]+)\s*BLK\.\s*"
    r"Outcome:\s*(?P<outcome>Win|Loss)\.",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    chunk_type: str
    text: str
    player: str | None
    team: str | None
    opponent: str | None
    date: str | None
    season: str | None
    game_id: str | None
    source_type: str
    metadata_json: str


def _season_for(date_value: str) -> str:
    year = int(str(date_value)[:4])
    month = int(str(date_value)[5:7])
    start = year if month >= 10 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _chunk_id(*parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _parse_summary(summary: str) -> dict[str, object]:
    match = SUMMARY_RE.search(summary)
    if not match:
        return {}
    values: dict[str, object] = match.groupdict()
    for key in ("points", "rebounds", "assists", "steals", "blocks"):
        values[key] = float(values[key])
    return values


def load_player_logs(input_path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    parsed = df["gameSummary"].map(_parse_summary).apply(pd.Series)
    for column in parsed.columns:
        if column not in df.columns:
            df[column] = parsed[column]
    df["date"] = pd.to_datetime(df["gameDateTimeEst"]).dt.date.astype(str)
    df["season"] = df["date"].map(_season_for)
    df["game_id"] = df["gameId"].astype(str)
    df["player"] = df["name"]
    df["matchup"] = df["matchup"].where(df["matchup"].notna(), df["team"].astype(str) + " vs. " + df["opponent"].astype(str))
    df["chunk_source"] = "cleaned_player_statistics"
    return df


def _metadata(row: pd.Series | dict, **extra: object) -> str:
    base = {
        "player": row.get("player"),
        "team": row.get("team"),
        "opponent": row.get("opponent"),
        "date": row.get("date"),
        "season": row.get("season"),
        "game_id": row.get("game_id"),
        "source_type": row.get("source_type", row.get("chunk_source", "cleaned_player_statistics")),
    }
    base.update(extra)
    return json.dumps(base, sort_keys=True)


def build_player_game_chunks(df: pd.DataFrame) -> Iterable[EvidenceChunk]:
    for row in df.itertuples(index=False):
        values = row._asdict()
        text = (
            f"Player-game evidence. {values['player']} played for {values['team']} against "
            f"{values['opponent']} on {values['date']} ({values['matchup']}, game {values['game_id']}). "
            f"He recorded {values['points']:.0f} points, {values['rebounds']:.0f} rebounds, "
            f"{values['assists']:.0f} assists, {values['steals']:.0f} steals, and {values['blocks']:.0f} blocks. "
            f"Result: {values['outcome']}."
        )
        chunk_id = _chunk_id("player-game", values["game_id"], values["player"], values["team"])
        yield EvidenceChunk(
            chunk_id=chunk_id,
            chunk_type="player-game",
            text=text,
            player=values["player"],
            team=values["team"],
            opponent=values["opponent"],
            date=values["date"],
            season=values["season"],
            game_id=values["game_id"],
            source_type="box_score",
            metadata_json=_metadata(values, points=values["points"], rebounds=values["rebounds"], assists=values["assists"]),
        )


def build_team_game_chunks(df: pd.DataFrame) -> Iterable[EvidenceChunk]:
    for (game_id, team), group in df.groupby(["game_id", "team"], dropna=False):
        first = group.iloc[0]
        totals = group[["points", "rebounds", "assists", "steals", "blocks"]].sum()
        leaders = group.sort_values(["points", "rebounds", "assists"], ascending=False).head(4)
        leader_text = "; ".join(
            f"{r.player}: {r.points:.0f} PTS, {r.rebounds:.0f} REB, {r.assists:.0f} AST"
            for r in leaders.itertuples()
        )
        text = (
            f"Team-game evidence. {team} faced {first['opponent']} on {first['date']} "
            f"({first['matchup']}, game {game_id}, season {first['season']}). "
            f"Team totals in logged player rows: {totals['points']:.0f} points, {totals['rebounds']:.0f} rebounds, "
            f"{totals['assists']:.0f} assists, {totals['steals']:.0f} steals, {totals['blocks']:.0f} blocks. "
            f"Top contributors: {leader_text}."
        )
        yield EvidenceChunk(
            chunk_id=_chunk_id("team-game", game_id, team),
            chunk_type="team-game",
            text=text,
            player=None,
            team=team,
            opponent=first["opponent"],
            date=first["date"],
            season=first["season"],
            game_id=str(game_id),
            source_type="box_score_rollup",
            metadata_json=_metadata(first, player=None, source_type="box_score_rollup", team_points=float(totals["points"])),
        )


def build_game_level_chunks(df: pd.DataFrame) -> Iterable[EvidenceChunk]:
    for game_id, group in df.groupby("game_id", dropna=False):
        first = group.iloc[0]
        teams = sorted(group["team"].dropna().unique())
        matchup = " vs. ".join(teams) if len(teams) == 2 else first["matchup"]
        leaders = group.sort_values(["points", "rebounds", "assists"], ascending=False).head(8)
        leader_text = "; ".join(
            f"{r.player} ({r.team}): {r.points:.0f} PTS, {r.rebounds:.0f} REB, {r.assists:.0f} AST"
            for r in leaders.itertuples()
        )
        text = (
            f"Game-level evidence. Game {game_id} on {first['date']} in season {first['season']}: "
            f"{matchup}. Leading player lines: {leader_text}."
        )
        yield EvidenceChunk(
            chunk_id=_chunk_id("game-level", game_id),
            chunk_type="game-level",
            text=text,
            player=None,
            team=";".join(teams) if teams else None,
            opponent=None,
            date=first["date"],
            season=first["season"],
            game_id=str(game_id),
            source_type="game_rollup",
            metadata_json=_metadata(first, player=None, team=teams, opponent=None, source_type="game_rollup"),
        )


def build_season_summary_chunks(df: pd.DataFrame) -> Iterable[EvidenceChunk]:
    grouped = df.groupby(["season", "player", "team"], dropna=False)
    for (season, player, team), group in grouped:
        games = len(group)
        means = group[["points", "rebounds", "assists", "steals", "blocks"]].mean()
        latest = group.sort_values("date").iloc[-1]
        text = (
            f"Season-summary evidence. In {season}, {player} with {team} has {games} logged games. "
            f"Averages: {means['points']:.1f} points, {means['rebounds']:.1f} rebounds, "
            f"{means['assists']:.1f} assists, {means['steals']:.1f} steals, {means['blocks']:.1f} blocks. "
            f"Most recent logged game: {latest['date']} against {latest['opponent']}."
        )
        yield EvidenceChunk(
            chunk_id=_chunk_id("season-summary", season, player, team),
            chunk_type="season-summary",
            text=text,
            player=player,
            team=team,
            opponent=None,
            date=str(latest["date"]),
            season=season,
            game_id=None,
            source_type="season_rollup",
            metadata_json=_metadata(latest, opponent=None, game_id=None, source_type="season_rollup", games=games),
        )


def build_event_chunks(df: pd.DataFrame) -> Iterable[EvidenceChunk]:
    event_terms = re.compile(r"\b(?:injur|trade|transaction|waived|signed|out|questionable|probable)\b", re.I)
    for row in df[df["gameSummary"].fillna("").str.contains(event_terms, regex=True)].itertuples(index=False):
        values = row._asdict()
        text = (
            f"Event evidence. Possible event mention for {values['player']} on {values['date']} "
            f"before/after {values['team']} vs {values['opponent']}: {values['gameSummary']}"
        )
        yield EvidenceChunk(
            chunk_id=_chunk_id("event", values["game_id"], values["player"], values["date"]),
            chunk_type="injury/transaction/event",
            text=text,
            player=values["player"],
            team=values["team"],
            opponent=values["opponent"],
            date=values["date"],
            season=values["season"],
            game_id=values["game_id"],
            source_type="event_heuristic",
            metadata_json=_metadata(values, source_type="event_heuristic"),
        )


def build_chunks(df: pd.DataFrame) -> pd.DataFrame:
    builders = (
        build_player_game_chunks,
        build_team_game_chunks,
        build_game_level_chunks,
        build_season_summary_chunks,
        build_event_chunks,
    )
    chunks = [asdict(chunk) for builder in builders for chunk in builder(df)]
    return pd.DataFrame(chunks).drop_duplicates("chunk_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata-rich NBA evidence chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_player_logs(args.input)
    chunks = build_chunks(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(args.output, index=False)
    print(f"Wrote {len(chunks)} evidence chunks to {args.output}")
    print(chunks["chunk_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
