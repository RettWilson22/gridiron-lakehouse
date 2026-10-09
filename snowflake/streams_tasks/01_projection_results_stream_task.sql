-- 01_projection_results_stream_task.sql
-- Run as GRIDIRON_ADMIN after the first sync. Keeps APP.PROJECTION_RESULTS (every published
-- projection next to what actually happened) current as results land, without waiting for
-- a dbt run, and exposes weekly accuracy as the view APP.WEEKLY_ACCURACY.
--
-- Why a stream + task here:
--   * New results are a naturally incremental workload: each sync after a week's games
--     MERGEs that week's player rows into SYNCED.PLAYER_WEEK, and stat corrections update a
--     few rows. The sync only touches rows that changed, so a standard stream on the table
--     captures exactly the affected weeks, and the task rebuilds only those weeks.
--   * The task runs on GRIDIRON_WH, so the resource monitor covers it, and its WHEN clause
--     skips the run (no warehouse start) when the stream is empty.
--   * SHOW_INITIAL_ROWS makes the first run backfill every existing week.
-- dbt's MARTS.MART_PROJECTION_SCORECARD holds the same record rebuilt in batch (and in
-- DuckDB for CI and the public copy); this table is the in-Snowflake incremental version.

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE WAREHOUSE GRIDIRON_WH;

CREATE TABLE IF NOT EXISTS APP.PROJECTION_RESULTS (
    SEASON NUMBER(10,0),
    WEEK NUMBER(10,0),
    PLAYER_ID VARCHAR,
    PLAYER_NAME VARCHAR,
    POSITION VARCHAR,
    TEAM VARCHAR,
    KIND VARCHAR,
    PROJ_PPR FLOAT,
    FLOOR_PPR FLOAT,
    CEILING_PPR FLOAT,
    BASELINE_LAST3 FLOAT,
    BASELINE_SEASON_AVG FLOAT,
    ECR_RANK NUMBER(19,0),
    ACTUAL_PPR FLOAT,
    PLAYED BOOLEAN,
    ERROR FLOAT,
    INSIDE_RANGE BOOLEAN,
    UPDATED_AT TIMESTAMP_LTZ,
    PRIMARY KEY (SEASON, WEEK, PLAYER_ID)
);

-- Work table holding the weeks touched by the current batch of changes.
CREATE TABLE IF NOT EXISTS APP.RESULTS_CHANGED_WEEKS (
    SEASON NUMBER(10,0),
    WEEK NUMBER(10,0)
);

CREATE STREAM IF NOT EXISTS SYNCED.PLAYER_WEEK_CHANGES
  ON TABLE SYNCED.PLAYER_WEEK
  SHOW_INITIAL_ROWS = TRUE
  COMMENT = 'New and corrected results, consumed by APP.MAINTAIN_PROJECTION_RESULTS';

CREATE OR REPLACE TASK APP.MAINTAIN_PROJECTION_RESULTS
  WAREHOUSE = GRIDIRON_WH
  -- Tuesday 14:00 UTC, after the Databricks job (12:00 UTC) and the sync.
  SCHEDULE = 'USING CRON 0 14 * * 2 UTC'
  COMMENT = 'Rebuild projection results for weeks whose actual results changed'
  WHEN SYSTEM$STREAM_HAS_DATA('GRIDIRON.SYNCED.PLAYER_WEEK_CHANGES')
AS
EXECUTE IMMEDIATE $$
BEGIN
  BEGIN TRANSACTION;

  DELETE FROM APP.RESULTS_CHANGED_WEEKS;

  -- Consuming the stream in a DML statement advances its offset when the transaction
  -- commits. Both the before and after images of updated rows are included.
  INSERT INTO APP.RESULTS_CHANGED_WEEKS (SEASON, WEEK)
  SELECT DISTINCT SEASON, WEEK FROM SYNCED.PLAYER_WEEK_CHANGES;

  DELETE FROM APP.PROJECTION_RESULTS r
  USING APP.RESULTS_CHANGED_WEEKS c
  WHERE r.SEASON = c.SEASON AND r.WEEK = c.WEEK;

  -- Every projection for a finished game, scored against the actual result (zero for a
  -- projected player without a stat row: inactive, or active without recording a stat).
  INSERT INTO APP.PROJECTION_RESULTS
  SELECT
    p.SEASON, p.WEEK, p.PLAYER_ID, p.PLAYER_NAME, p.POSITION, p.TEAM, p.KIND,
    p.PROJ_PPR, p.FLOOR_PPR, p.CEILING_PPR, p.BASELINE_LAST3, p.BASELINE_SEASON_AVG,
    p.ECR_RANK,
    COALESCE(w.FANTASY_POINTS_PPR, 0) AS ACTUAL_PPR,
    w.PLAYER_ID IS NOT NULL AS PLAYED,
    p.PROJ_PPR - COALESCE(w.FANTASY_POINTS_PPR, 0) AS ERROR,
    COALESCE(w.FANTASY_POINTS_PPR, 0) BETWEEN p.FLOOR_PPR AND p.CEILING_PPR AS INSIDE_RANGE,
    CURRENT_TIMESTAMP()
  FROM SYNCED.PROJECTIONS p
  JOIN APP.RESULTS_CHANGED_WEEKS c
    ON p.SEASON = c.SEASON AND p.WEEK = c.WEEK
  JOIN SYNCED.TEAM_WEEK g
    ON p.GAME_ID = g.GAME_ID AND p.TEAM = g.TEAM AND g.IS_FINAL
  LEFT JOIN SYNCED.PLAYER_WEEK w
    ON p.SEASON = w.SEASON AND p.WEEK = w.WEEK AND p.PLAYER_ID = w.PLAYER_ID;

  COMMIT;
END;
$$;

-- Weekly accuracy over the backtest's evaluation pool (FantasyPros positional top N; the
-- sizes in the DECODE are gridiron.tiers.POOL, and a test checks they match).
CREATE OR REPLACE VIEW APP.WEEKLY_ACCURACY AS
SELECT
  SEASON,
  WEEK,
  POSITION,
  ANY_VALUE(KIND) AS KIND,
  COUNT(*) AS PLAYERS,
  AVG(ABS(ERROR)) AS MAE,
  SQRT(AVG(ERROR * ERROR)) AS RMSE,
  AVG(ERROR) AS BIAS,
  AVG(IFF(INSIDE_RANGE, 1, 0)) AS INTERVAL_COVERAGE,
  AVG(ABS(BASELINE_LAST3 - ACTUAL_PPR)) AS MAE_LAST3,
  AVG(ABS(BASELINE_SEASON_AVG - ACTUAL_PPR)) AS MAE_SEASON_AVG
FROM APP.PROJECTION_RESULTS
WHERE ECR_RANK <= DECODE(POSITION, 'QB', 24, 'RB', 48, 'WR', 72, 'TE', 24)
  AND BASELINE_LAST3 IS NOT NULL
  AND BASELINE_SEASON_AVG IS NOT NULL
GROUP BY SEASON, WEEK, POSITION;

-- Tasks are created suspended.
ALTER TASK APP.MAINTAIN_PROJECTION_RESULTS RESUME;

-- Run once now instead of waiting for Tuesday, then inspect:
-- EXECUTE TASK APP.MAINTAIN_PROJECTION_RESULTS;
-- SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(TASK_NAME => 'MAINTAIN_PROJECTION_RESULTS'));
-- SELECT * FROM APP.WEEKLY_ACCURACY ORDER BY SEASON DESC, WEEK DESC, POSITION LIMIT 20;
