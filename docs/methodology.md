# Methodology

How the projections are built and how they are judged. Backtest numbers come from
`make model-local` run on the local data on 2026-10-09 (UTC); the full output, including
every metric below, is [`artifacts/backtest_metrics.json`](../artifacts/backtest_metrics.json).

The app's Track record tab shows the Databricks run instead. The sample sizes, baselines and
expert rankings are identical, but the model was trained separately in each environment,
and its numbers differ slightly: MAE by at most 0.024, rank correlation by at most 0.007 and
range coverage by at most 1.3 percentage points.

## Data sources

URLs and schemas were checked on 2026-10-08. Seasons 2018-2026 are used; as of 2026-10-08,
the 2026 season is complete through week 4.

| Dataset | Source | Coverage published | Used for | Terms |
| --- | --- | --- | --- | --- |
| Play-by-play | nflverse-data release `pbp/play_by_play_{season}.parquet` | 1999- | red-zone and goal-line opportunities, team volume | CC-BY-4.0 |
| Weekly player stats | nflverse `stats_player/stats_player_week_{season}.parquet` (the older `player_stats/player_stats_{season}.parquet` stops at 2024) | 1999-2026 | fantasy points and stat lines (targets) | CC-BY-4.0 |
| Snap counts | nflverse `snap_counts/snap_counts_{season}.parquet` (Pro Football Reference ids and origin) | 2012-2026 | snap share | CC-BY-4.0 |
| Weekly rosters | nflverse `weekly_rosters/roster_weekly_{season}.parquet` | 2002-2026 | listed position, years of experience | CC-BY-4.0 |
| Injury reports | nflverse `injuries/injuries_{season}.parquet` | 2009-2026 | game status and practice participation | CC-BY-4.0 |
| Depth charts | nflverse `depth_charts/depth_charts_{season}.parquet`: weekly charts through 2024, timestamped snapshots from 2025 (one or two a day in season) | 2001-2026 | candidate pool and depth rank (both formats harmonized) | CC-BY-4.0 |
| Schedules | nflverse `schedules/games.parquet` with closing or current spread and total | 1999-2026, upcoming weeks once lines post | opponents, kickoff times, implied team totals, the upcoming week | CC-BY-4.0 |
| Players | nflverse `players/players.parquet` | current | names, ids (PFR to GSIS crosswalk), birth dates | CC-BY-4.0 |
| Expected fantasy points | ffverse/ffopportunity release `latest-data/ep_weekly_{season}.parquet` | 2006-2026 | usage-based expected PPR points (features, risers) | GPL-3.0 repository |
| FantasyPros weekly ECR | DynastyProcess `files/db_fpecr.parquet` (every FantasyPros ranking page since 2019; weekly positional pages kept) | weekly snapshots 2020-2026; 2019 has one; 2024 starts in week 4 | the expert benchmark in the backtest | GPL-3.0 repository; the rankings are FantasyPros content |
| FantasyPros-to-GSIS ids | DynastyProcess `files/db_playerids.csv` | current | joining ECR to players | GPL-3.0 repository |

The FantasyPros rankings are used only as a benchmark. The public snapshot leaves out the
per-player ranks and keeps the accuracy comparison against them.

Historical weekly ECR exists, so the backtest uses it directly and no archiving step is
needed. Ingest notes: integer columns are widened at landing because nflverse types drift
between seasons (injury `season` and `week` are doubles before 2021); play-by-play is the
exception and lands as published, because the 20 columns silver reads do not drift. The ECR
history is filtered to weekly positional rankings and split per season so completed seasons
are written once; the id crosswalk arrives as CSV and is landed as Parquet; relocated teams
are normalized in silver (schedules say `OAK` for 2018-2019, box scores say `LV`).

## Candidates

For each team and game: the players on that week's depth chart at QB, RB, WR or TE, plus
anyone who played for the team in its previous two games, minus anyone ruled Out on the
final injury report. Weekly depth charts (through 2024) are used as published for the week;
for 2025 onward the pool uses the last snapshot strictly before the team's kickoff. A
player's position is the one on that week's roster, else that week's depth chart, else the
one he played in the game that put him in the pool.

The pool covers 96.1% of all 2018-2026 player-games with a stat row and 98.9% of those with
10 or more PPR points (`candidate_coverage` in the metrics file).

## Features and leakage rules

Every projection uses only information available before kickoff:

* Player, team and opponent history is attached with an as-of merge on a strictly earlier
  `(season, week)`, so a week-w row cannot see week w or anything later.
* Week-w inputs are limited to what is published before the game: the schedule and betting
  lines, the depth chart, that week's final injury report, and that week's roster. Years of
  experience come from the player's latest roster row in the same season at or before week w.

Features: exponentially weighted usage and production over previous games (snap, target,
carry, air-yards and red-zone shares, red-zone and goal-line carries, expected PPR points,
yards, touchdowns), last-game and last-three averages, season-to-date and last-season
averages, the opponent's PPR allowed to the position over its last six games and that
relative to the league, home or away, implied team total, spread and total, injury status
and practice participation, depth rank, games missed, rookie flag, experience and age.

`tests/test_features.py` checks the rules on fixture data. It scrambles every result from
week w onward (and separately drops them), scrambles other weeks' injury reports and depth
charts and later weeks' rosters, and checks that the week-w candidates and features do not
change. Control tests show the features do react to earlier weeks' results and to the
week's own roster. Writing these tests caught a leak in an early draft (a player's position
was looked up from his latest stats row, which can be in a later week). A later review
caught a second one: years of experience was the minimum over the whole season's rosters,
including later weeks. It now comes from the roster as of the week, and the roster test
perturbs later weeks' experience as well. On the 2018-2026 data the fix changed no feature
value, so the backtest results did not change.

## Model

Per position, one histogram gradient-boosting regressor (scikit-learn) per stat-line
component: passing yards, passing touchdowns, interceptions, rushing yards and touchdowns,
receptions, receiving yards and touchdowns, fumbles lost, two-point conversions (Poisson
loss for counts). Projected points in any format are the scoring rules applied to the
projected stat line, which is also what custom league scoring needs.

Floor and ceiling are the 10th and 90th percentile of actual PPR points among training
player-weeks with a similar projection (20 equal-count bins, interpolated, made monotone),
scaled to half PPR and standard by the ratio of the projections.

The design was settled in exploratory runs on the 2022 season (trained on 2018-2021) before
any test season was scored. Those runs are not included in the repository as a script; the
choices they led to were: project components rather than points (about as accurate, and
needed for custom scoring), use shallow, heavily regularized trees, and use empirical bands
because quantile gradient boosting collapsed toward zero on this zero-inflated target
(players who sit score zero).

## Tiers, start/sit and risers

**Tiers.** Within each position, among the top 24 QBs, 48 RBs, 72 WRs and 24 TEs, tiers are
the optimal one-dimensional clustering of the projections (Jenks natural breaks, solved
exactly by dynamic programming) with the fewest tiers that explain at least 90% of the
spread. Tiers break where the gaps between players are large relative to the position.

**Start / sit.** Positional rank against a 12-team, 1 QB / 2 RB / 3 WR / 1 TE / 1 flex
league: Start within the starters, Flex for the next 12 RBs or WRs (4 TEs), Sit otherwise.

**Risers.** A player's usage over his last three games against his earlier games this
season (or last season early on): snap, target, carry and red-zone shares and expected PPR
points per game. A riser gained at least 2.5 expected points per game and now averages 5 or
more.

## Live weeks

The Databricks job runs Tuesday (results through Monday night), Thursday (final injury
reports for Thursday games) and Saturday (final reports for Sunday and Monday games) at
12:00 UTC. The upcoming week is the first regular-season week with an unplayed game. Each
run replaces projections only for games that have not kicked off, so the stored record is
exactly what was published before each game, and a game that kicked off before its first
run is never projected. `tests/test_projections.py` covers both rules;
[verification.md](verification.md) shows them on real data.

## Backtest

Walk-forward over 2023, 2024 and 2025: each season is projected by a model trained only on
the seasons before it, with every feature taken from before each game. Scored on the players
FantasyPros ranked in the top 24 QBs, 48 RBs, 72 WRs and 24 TEs that week, in the 46 weeks
with a weekly ECR snapshot (2023 weeks 2-17, 2024 weeks 4-17, 2025 weeks 2-17). Rows need a
value from every method, which leaves out rookies before their first game. A projected
player without a stat row for the game (inactive, or active without recording a stat)
scores zero. The model projected 97.6-98.8% of the expert-ranked player-weeks, depending on
position (`pool_coverage`).

MAE and RMSE are in PPR points. Rank correlation is Spearman's within each position-week,
averaged over weeks. ECR has no point values, so it is compared on ranking only.

### 2023-2025 combined

| Position | Player-weeks | MAE model | MAE last 3 | MAE season avg | RMSE model | RMSE last 3 | RMSE season avg | Rank corr. model | Rank corr. last 3 | Rank corr. season avg | Rank corr. ECR | Inside 10th-90th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QB | 1,091 | **6.37** | 6.83 | 6.66 | **7.96** | 8.69 | 8.48 | 0.308 | 0.200 | 0.223 | **0.314** | 77.5% |
| RB | 2,170 | **5.70** | 6.29 | 6.04 | **7.39** | 8.16 | 7.84 | 0.477 | 0.391 | 0.426 | **0.499** | 83.1% |
| WR | 3,221 | **5.78** | 6.50 | 6.17 | **7.50** | 8.29 | 7.98 | 0.434 | 0.321 | 0.364 | **0.458** | 84.2% |
| TE | 1,075 | **5.30** | 5.94 | 5.58 | **6.96** | 7.63 | 7.22 | 0.284 | 0.192 | 0.209 | **0.316** | 78.8% |

### By season

MAE, and rank correlation model / ECR.

| Season | QB | RB | WR | TE |
| --- | --- | --- | --- | --- |
| 2023 MAE model / last 3 / season avg | 6.18 / 6.74 / 6.40 | 5.55 / 6.22 / 6.05 | 5.82 / 6.52 / 6.31 | 5.23 / 5.67 / 5.48 |
| 2023 rank corr. model / ECR | 0.365 / 0.395 | 0.450 / 0.452 | 0.451 / 0.482 | 0.287 / 0.365 |
| 2024 MAE model / last 3 / season avg | 6.34 / 6.59 / 6.32 | 5.64 / 6.12 / 5.97 | 5.91 / 6.65 / 6.07 | 5.28 / 5.98 / 5.52 |
| 2024 rank corr. model / ECR | 0.329 / 0.324 | 0.505 / 0.537 | 0.414 / 0.439 | 0.314 / 0.328 |
| 2025 MAE model / last 3 / season avg | 6.59 / 7.14 / 7.21 | 5.91 / 6.51 / 6.09 | 5.63 / 6.33 / 6.11 | 5.39 / 6.17 / 5.74 |
| 2025 rank corr. model / ECR | 0.231 / 0.225 | 0.479 / 0.512 | 0.434 / 0.452 | 0.255 / 0.257 |

### Reading the results

* **Points.** Over the three seasons the model has the lowest MAE and RMSE at every
  position, by 0.28 to 0.39 points per player-week against the better of the two averages.
  It is not a clean sweep: for 2024 QBs the season-to-date average had a slightly lower MAE
  (6.32 against 6.34), though a higher RMSE.
* **Ranking.** The model ranks players better than both averages at every position, and
  worse than FantasyPros consensus at every position over the three seasons (by 0.006 to
  0.032). It matched or beat ECR only for QBs in 2024 and 2025. Experts see things this
  model does not: news, coaching intent, the end of the week's injury picture.
* **Bias.** In this pool the model projects 0.5 to 1.1 points low on average. The pool is
  players the experts rank highly, and the model is trained on every candidate, including
  backups who often score zero, so it shades expert favorites down.
* **Range.** A calibrated 80% range should contain about 80% of outcomes. The model's
  10th-90th percentile ranges contained 77.5-84.2% depending on position: slightly narrow
  for QBs and TEs, slightly wide for RBs and WRs.

### What the comparison does and does not control

* **Retraining.** The backtest retrains once per season; the live system retrains on every
  run and also learns from the current season's earlier weeks. On this point the backtest
  understates the live model.
* **Information.** Backtest features use closing betting lines and the final injury report
  for every game. The Tuesday and Thursday live runs project Sunday and Monday games with
  the lines and injury reports available at the time; only the Saturday run has the final
  reports for those games. On this point the backtest overstates what a midweek projection
  knows.
* **Expert timing.** The ECR benchmark for a week is the latest FantasyPros scrape on or
  before the week's last game day (the pages are scraped on Fridays, after Thursday's game),
  so ECR can include news a Tuesday projection does not have.
* **Pool.** Scoring on FantasyPros' top N keeps the comparison fair across methods, but it
  scores only weeks with a ranking snapshot (none for week 1, week 18, or 2024 weeks 1-3)
  and leaves out players the experts did not rank highly.

The same record is rebuilt in SQL by dbt (`mart_projection_scorecard`) from the published
projections, and a dbt test (`assert_scorecard_reconciles_with_backtest`) checks that it
reproduces the Python sample sizes, MAE and interval coverage.

## Known gaps

* Floors and ceilings come from PPR and are scaled to other formats by the ratio of the
  projections, including under custom scoring. Yardage bonuses applied to an average stat
  line understate their real value (the app says so).
* The tight end premium and yardage bonuses are supported by the scoring UDF but not
  modelled separately.
* From 2025 on, depth charts are timestamped snapshots used as of the latest one before
  kickoff; for the Tuesday and Thursday runs that is days before the game.
* Kickers and defenses are not modelled.
* Samples are small: five training seasons for the first test season, and 14 to 16 scored
  weeks per test season.
