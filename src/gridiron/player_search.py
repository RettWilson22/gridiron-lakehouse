"""Smart player search for the cheat sheet's player index.

Matches the way people actually type player names: any case, accents and punctuation,
last name first, partial names ("jeff"), initials ("jsn", "arsb", "kw3"), common nicknames
("cmc"), and small typos ("mccaffery"). Team and position words in the query filter the
results instead of being matched against names, so "lions wr" lists Detroit's receivers.

Standard library only: Streamlit in Snowflake imports this file directly from the app's
stage, next to the app.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

POSITIONS = ("QB", "RB", "WR", "TE")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5"}

# nflverse team codes -> words people use for the team.
TEAMS: dict[str, tuple[str, ...]] = {
    "ARI": ("arizona", "cardinals", "cards"),
    "ATL": ("atlanta", "falcons"),
    "BAL": ("baltimore", "ravens"),
    "BUF": ("buffalo", "bills"),
    "CAR": ("carolina", "panthers"),
    "CHI": ("chicago", "bears"),
    "CIN": ("cincinnati", "bengals"),
    "CLE": ("cleveland", "browns"),
    "DAL": ("dallas", "cowboys"),
    "DEN": ("denver", "broncos"),
    "DET": ("detroit", "lions"),
    "GB": ("green bay", "packers"),
    "HOU": ("houston", "texans"),
    "IND": ("indianapolis", "colts"),
    "JAX": ("jacksonville", "jaguars", "jags"),
    "KC": ("kansas city", "chiefs"),
    "LA": ("los angeles rams", "rams"),
    "LAC": ("los angeles chargers", "chargers"),
    "LV": ("las vegas", "raiders"),
    "MIA": ("miami", "dolphins"),
    "MIN": ("minnesota", "vikings"),
    "NE": ("new england", "patriots", "pats"),
    "NO": ("new orleans", "saints"),
    "NYG": ("new york giants", "giants"),
    "NYJ": ("new york jets", "jets"),
    "PHI": ("philadelphia", "eagles"),
    "PIT": ("pittsburgh", "steelers"),
    "SEA": ("seattle", "seahawks"),
    "SF": ("san francisco", "49ers", "niners"),
    "TB": ("tampa bay", "buccaneers", "bucs"),
    "TEN": ("tennessee", "titans"),
    "WAS": ("washington", "commanders"),
}
# Two-letter codes that are also everyday words only count as a team when they are the
# whole query ("no" alone means the Saints; "no" next to a name is ignored as a filter).
AMBIGUOUS_CODES = {"NO", "NE", "LA"}

# Nicknames that initials alone don't produce.
NICKNAMES: dict[str, str] = {
    "cmc": "christian mccaffrey",
    "dk": "dk metcalf",
    "tmac": "terry mclaurin",
    "scary terry": "terry mclaurin",
    "kittle": "george kittle",
    "hollywood": "marquise brown",
    "chubb": "nick chubb",
    "saquon": "saquon barkley",
    "bijan": "bijan robinson",
    "jj": "justin jefferson",
    "ajb": "aj brown",
    "melt": "drake london",
}

# Limits on what a search reads, so a huge paste cannot tie up the app: every name word is
# compared with every player, with a fuzzy match for each. The search boxes also stop at
# MAX_QUERY_CHARS characters, but the limit is enforced here, server side.
MAX_QUERY_CHARS = 60
MAX_NAME_TERMS = 4

# Match strength, highest first. Ties are broken by how relevant the player is this week.
EXACT, NICKNAME, INITIALS, PREFIX, SUBSTRING = 100.0, 95.0, 90.0, 80.0, 70.0
FUZZY_FLOOR, FUZZY_MIN_RATIO = 50.0, 0.78


@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    position: str
    team: str


@dataclass(frozen=True)
class Query:
    """A search split into name words and the team/position filters it contains."""

    terms: tuple[str, ...]
    teams: frozenset[str]
    positions: frozenset[str]

    @property
    def is_empty(self) -> bool:
        return not (self.terms or self.teams or self.positions)


@dataclass(frozen=True)
class Match:
    player: Player
    score: float
    reason: str


def normalize(text: str) -> str:
    """Lower case, accents and punctuation removed, hyphens and periods become spaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[.\-_/]", " ", text.lower())
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def name_tokens(name: str) -> list[str]:
    return normalize(name).split()


def _without_suffix(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in SUFFIXES] or tokens


def initials(name: str) -> set[str]:
    """Ways people abbreviate a name: "Amon-Ra St. Brown" -> arsb; "Kenneth Walker III" ->
    kw, kw3, kwiii; "A.J. Brown" -> ajb; first + last initial for everyone."""
    tokens = name_tokens(name)
    core = _without_suffix(tokens)
    found = {"".join(t[0] for t in core)}
    if len(core) >= 2:
        found.add(core[0][0] + core[-1][0])
    suffix = [t for t in tokens if t in SUFFIXES]
    for s in suffix:
        base = "".join(t[0] for t in core)
        found |= {base + s, base + ROMAN.get(s, s[0])}
    return {f for f in found if len(f) >= 2}


def last_name(name: str) -> str:
    """The surname used to sort the index ("St. Brown", "Harrison" for "Marvin Harrison Jr.")."""
    # Split on spaces only, so a hyphenated first name ("Amon-Ra") stays one word.
    words = [normalize(w) for w in name.split()]
    core = [w for w in words if w and w not in SUFFIXES] or words
    if len(core) <= 1:
        return core[0] if core else ""
    return " ".join(core[1:])


def index_letter(name: str) -> str:
    surname = last_name(name)
    return surname[:1].upper() if surname[:1].isalpha() else "#"


def parse_query(raw: str) -> Query:
    """Pull team and position words out of the query; what remains is the name.

    Reads at most ``MAX_QUERY_CHARS`` characters and keeps at most ``MAX_NAME_TERMS``
    name words.
    """
    text = normalize(raw[:MAX_QUERY_CHARS])
    teams: set[str] = set()
    for code, words in TEAMS.items():
        for word in sorted(words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(word)}\b", text):
                teams.add(code)
                text = re.sub(rf"\b{re.escape(word)}\b", " ", text)
    tokens = text.split()
    positions = {t.upper() for t in tokens if t.upper() in POSITIONS}
    tokens = [t for t in tokens if t.upper() not in POSITIONS]
    only_token = len(tokens) == 1 and not teams and not positions
    for token in list(tokens):
        code = token.upper()
        if code in TEAMS and (code not in AMBIGUOUS_CODES or only_token):
            teams.add(code)
            tokens.remove(token)
    return Query(tuple(tokens[:MAX_NAME_TERMS]), frozenset(teams), frozenset(positions))


def _token_ratio(term: str, tokens: list[str]) -> float:
    return max((difflib.SequenceMatcher(None, term, t).ratio() for t in tokens), default=0.0)


def score_name(name: str, terms: tuple[str, ...]) -> tuple[float, str] | None:  # noqa: PLR0911
    """How well the name words of a query match one player's name, or None.

    Checked from the strongest kind of match to the weakest; the first that applies wins.
    """
    if not terms:
        return 1.0, "team / position"
    query = " ".join(terms)
    full = normalize(name)
    tokens = name_tokens(name)
    compact = query.replace(" ", "")
    if query == full or query == " ".join(_without_suffix(tokens)):
        return EXACT, "exact"
    if NICKNAMES.get(query) == " ".join(_without_suffix(tokens)):
        return NICKNAME, "nickname"
    if len(terms) == 1 and len(compact) >= 2 and compact in initials(name):
        return INITIALS, "initials"
    if all(any(t.startswith(term) for t in tokens) for term in terms):
        # Starting with the surname, or typing several words, is a stronger signal.
        bonus = 5.0 if tokens and terms[0] != tokens[0] else 0.0
        return PREFIX + bonus + min(len(compact), 10) * 0.5, "prefix"
    if query in full or compact in full.replace(" ", ""):
        return SUBSTRING, "substring"
    ratios = [_token_ratio(term, tokens) for term in terms if len(term) >= 3]
    if ratios and len(ratios) == len(terms) and min(ratios) >= FUZZY_MIN_RATIO:
        return FUZZY_FLOOR + 20.0 * sum(ratios) / len(ratios), "close spelling"
    return None


def search(
    players: Iterable[Player],
    raw_query: str,
    relevance: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[Match]:
    """Players matching the query, best first; ties go to the more relevant player."""
    query = parse_query(raw_query)
    if query.is_empty:
        return []
    relevance = relevance or {}
    matches = []
    for player in players:
        if query.teams and player.team not in query.teams:
            continue
        if query.positions and player.position not in query.positions:
            continue
        scored = score_name(player.name, query.terms)
        if scored is not None:
            matches.append(Match(player, *scored))
    matches.sort(key=lambda m: (-m.score, -relevance.get(m.player.player_id, 0.0), m.player.name))
    return matches[:limit] if limit else matches


def did_you_mean(players: Iterable[Player], raw_query: str) -> str | None:
    """The closest full name when nothing matched, for a "Did you mean" hint."""
    terms = parse_query(raw_query).terms
    if not terms:
        return None
    names = {normalize(p.name): p.name for p in players}
    close = difflib.get_close_matches(" ".join(terms), list(names), n=1, cutoff=0.6)
    return names[close[0]] if close else None
