-- 01_create_udf.sql
-- Run as GRIDIRON_ADMIN with the Snowflake CLI (snow sql -f) from the repository root;
-- PUT needs a client. In Snowsight, upload src/gridiron/scoring.py to the stage instead.
--
-- Registers APP.FANTASY_POINTS(STATS OBJECT, SCORING OBJECT) -> FLOAT, a Python UDF that
-- scores a stat line under any league's settings. The handler is src/gridiron/scoring.py,
-- the same file the Databricks score task and the app's local mode use (standard library
-- only, so no packages are needed).
--
-- STATS keys: passing_yards, passing_tds, passing_interceptions, rushing_yards,
--   rushing_tds, receptions, receiving_yards, receiving_tds, fumbles_lost,
--   two_point_conversions, special_teams_tds, position (missing keys count as zero).
-- SCORING keys: preset ('ppr' (default), 'half' or 'standard') plus any overrides:
--   pass_yd, pass_td, pass_int, rush_yd, rush_td, rec, rec_yd, rec_td, fumble_lost,
--   two_pt, st_td, te_rec_premium, bonus_pass_300, bonus_pass_400, bonus_rush_100,
--   bonus_rush_200, bonus_rec_100, bonus_rec_200. Unknown keys raise an error.

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE SCHEMA APP;
USE WAREHOUSE GRIDIRON_WH;

CREATE STAGE IF NOT EXISTS APP.CODE
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Python UDF code';

PUT file://src/gridiron/scoring.py @APP.CODE AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

CREATE OR REPLACE FUNCTION APP.FANTASY_POINTS(STATS OBJECT, SCORING OBJECT)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'scoring.udf_handler'
IMPORTS = ('@APP.CODE/scoring.py')
COMMENT = 'Fantasy points for a stat line under custom league scoring (gridiron)';

-- 6 catches, 80 yards, a touchdown: 20 in PPR, 17 in half PPR, 14 in standard.
SELECT
  APP.FANTASY_POINTS(OBJECT_CONSTRUCT('receptions', 6, 'receiving_yards', 80, 'receiving_tds', 1), NULL) AS PPR,
  APP.FANTASY_POINTS(OBJECT_CONSTRUCT('receptions', 6, 'receiving_yards', 80, 'receiving_tds', 1),
                     OBJECT_CONSTRUCT('preset', 'half')) AS HALF_PPR,
  APP.FANTASY_POINTS(OBJECT_CONSTRUCT('receptions', 6, 'receiving_yards', 80, 'receiving_tds', 1),
                     OBJECT_CONSTRUCT('preset', 'standard')) AS STANDARD;

-- Last week's actual results rescored for a six-point-passing-touchdown, half PPR league
-- with a 100-yard receiving bonus, next to the default PPR scoring.
SELECT
  PLAYER_NAME, POSITION, FANTASY_POINTS_PPR,
  APP.FANTASY_POINTS(
    OBJECT_CONSTRUCT(
      'passing_yards', PASSING_YARDS, 'passing_tds', PASSING_TDS,
      'passing_interceptions', PASSING_INTERCEPTIONS, 'rushing_yards', RUSHING_YARDS,
      'rushing_tds', RUSHING_TDS, 'receptions', RECEPTIONS, 'receiving_yards', RECEIVING_YARDS,
      'receiving_tds', RECEIVING_TDS, 'fumbles_lost', FUMBLES_LOST,
      'two_point_conversions', TWO_POINT_CONVERSIONS, 'special_teams_tds', SPECIAL_TEAMS_TDS,
      'position', POSITION),
    OBJECT_CONSTRUCT('preset', 'half', 'pass_td', 6, 'bonus_rec_100', 3)
  ) AS CUSTOM_POINTS
FROM GRIDIRON.SYNCED.PLAYER_WEEK
WHERE (SEASON, WEEK) IN (SELECT SEASON, MAX(WEEK) FROM GRIDIRON.SYNCED.PLAYER_WEEK
                        WHERE SEASON = (SELECT MAX(SEASON) FROM GRIDIRON.SYNCED.PLAYER_WEEK)
                        GROUP BY SEASON)
ORDER BY CUSTOM_POINTS DESC
LIMIT 20;
