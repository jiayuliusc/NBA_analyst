from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_player_statistics.csv"

TEAM_ALIASES = {
    "hawks": "Hawks",
    "celtics": "Celtics",
    "nets": "Nets",
    "hornets": "Hornets",
    "bulls": "Bulls",
    "cavaliers": "Cavaliers",
    "cavs": "Cavaliers",
    "mavericks": "Mavericks",
    "mavs": "Mavericks",
    "nuggets": "Nuggets",
    "pistons": "Pistons",
    "warriors": "Warriors",
    "rockets": "Rockets",
    "pacers": "Pacers",
    "clippers": "Clippers",
    "lakers": "Lakers",
    "grizzlies": "Grizzlies",
    "heat": "Heat",
    "bucks": "Bucks",
    "timberwolves": "Timberwolves",
    "wolves": "Timberwolves",
    "pelicans": "Pelicans",
    "knicks": "Knicks",
    "thunder": "Thunder",
    "magic": "Magic",
    "76ers": "76ers",
    "sixers": "76ers",
    "suns": "Suns",
    "blazers": "Trail Blazers",
    "trail blazers": "Trail Blazers",
    "kings": "Kings",
    "spurs": "Spurs",
    "raptors": "Raptors",
    "jazz": "Jazz",
    "wizards": "Wizards",
}


@dataclass
class QueryIntent:
    query: str
    intent: str = "general"
    players: list[str] = field(default_factory=list)
    teams: list[str] = field(default_factory=list)
    opponents: list[str] = field(default_factory=list)
    date: str | None = None
    season: str | None = None
    recent: bool = False
    chunk_types: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _catalog() -> tuple[list[str], list[str], str | None]:
    if not DATA_PATH.exists():
        return [], [], None
    df = pd.read_csv(DATA_PATH, usecols=["name", "gameDateTimeEst", "gameSummary"])
    players = sorted(df["name"].dropna().unique(), key=len, reverse=True)
    teams = sorted({alias for alias in TEAM_ALIASES.values()})
    latest_date = str(pd.to_datetime(df["gameDateTimeEst"]).dt.date.max())
    return players, teams, latest_date


def _season_for_year(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def understand_query(query: str) -> QueryIntent:
    lowered = query.lower()
    players, _, latest_date = _catalog()
    intent = QueryIntent(query=query)

    for player in players:
        if re.search(rf"\b{re.escape(player.lower())}\b", lowered):
            intent.players.append(player)
            if len(intent.players) >= 3:
                break

    for alias, canonical in TEAM_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered) and canonical not in intent.teams:
            intent.teams.append(canonical)

    date_match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", lowered)
    if date_match:
        y, m, d = map(int, date_match.groups())
        intent.date = date(y, m, d).isoformat()
    elif latest_date and any(term in lowered for term in ("last night", "latest", "recent", "last game", "most recent")):
        latest = date.fromisoformat(latest_date)
        intent.date = latest.isoformat()
        intent.recent = True
    elif latest_date and "yesterday" in lowered:
        intent.date = (date.fromisoformat(latest_date) - timedelta(days=1)).isoformat()
        intent.recent = True

    season_match = re.search(r"\b(20\d{2})[-/](\d{2})\b", lowered)
    if season_match:
        intent.season = f"{season_match.group(1)}-{season_match.group(2)}"
    else:
        year_match = re.search(r"\b(20\d{2})\b", lowered)
        if year_match and not intent.date:
            year = int(year_match.group(1))
            intent.season = _season_for_year(year - 1 if year < 2030 else year)

    if any(term in lowered for term in ("average", "averages", "season", "per game")):
        intent.intent = "season averages"
        intent.chunk_types = ["season-summary", "player-game"]
    elif any(term in lowered for term in ("injury", "injured", "transaction", "trade", "signed", "waived")):
        intent.intent = "injuries / transactions"
        intent.chunk_types = ["injury/transaction/event", "player-game"]
    elif any(term in lowered for term in ("matchup", "against", "versus", " vs ", "head to head")):
        intent.intent = "matchup history"
        intent.chunk_types = ["game-level", "team-game", "player-game"]
    elif intent.players and any(term in lowered for term in ("stat", "score", "points", "rebounds", "assists", "play", "played")):
        intent.intent = "player stats"
        intent.chunk_types = ["player-game", "season-summary"]
    elif intent.teams:
        intent.intent = "team performance"
        intent.chunk_types = ["team-game", "game-level", "player-game"]
    elif intent.recent:
        intent.intent = "recent games"
        intent.chunk_types = ["game-level", "team-game", "player-game"]
    else:
        intent.chunk_types = ["player-game", "team-game", "game-level", "season-summary"]

    if len(intent.teams) >= 2:
        intent.opponents = intent.teams[1:]
    return intent
