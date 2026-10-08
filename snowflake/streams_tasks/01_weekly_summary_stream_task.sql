-- 01_weekly_summary_stream_task.sql
-- Run as GRIDIRON_ADMIN. Incrementally maintains APP.WEEKLY_FOURTH_DOWN_SUMMARY
-- (one row per season, week and team) from changes to SYNCED.FOURTH_DOWN_SCORED.
--
-- Why a stream + task here (and not a dynamic table):
--   * The fallback sync MERGEs into SYNCED.FOURTH_DOWN_SCORED and only touches rows that
--     changed, so a standard stream captures exactly the new or corrected plays of the
--     current week; the task recomputes just the affected (season, week, team) groups.
--   * The task runs on GRIDIRON_WH, so it is covered by the resource monitor, and its
--     WHEN clause skips the run (no warehouse start) when the stream is empty.
--   * A dynamic table is the simpler declarative option and is what this project uses on
--     the Iceberg path, where Databricks rewrites the serving table on every run and a
--     stream would report every row as changed. See 02_weekly_summary_dynamic_table.sql.

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE WAREHOUSE GRIDIRON_WH;

CREATE TABLE IF NOT EXISTS APP.WEEKLY_FOURTH_DOWN_SUMMARY (
    SEASON NUMBER(10,0),
    SEASON_TYPE VARCHAR,
    WEEK NUMBER(10,0),
    TEAM VARCHAR,
    FOURTH_DOWNS NUMBER(19,0),
    GO_ATTEMPTS NUMBER(19,0),
    CONVERSIONS NUMBER(19,0),
    GO_RECOMMENDATIONS NUMBER(19,0),
    FOLLOWED_MODEL NUMBER(19,0),
    EXPECTED_POINTS_LOST FLOAT,
    UPDATED_AT TIMESTAMP_LTZ,
    PRIMARY KEY (SEASON, SEASON_TYPE, WEEK, TEAM)
);

-- Work table holding the groups touched by the current batch of changes.
CREATE TABLE IF NOT EXISTS APP.WEEKLY_SUMMARY_CHANGED_KEYS (
    SEASON NUMBER(10,0),
    SEASON_TYPE VARCHAR,
    WEEK NUMBER(10,0),
    TEAM VARCHAR
);

-- SHOW_INITIAL_ROWS makes the first task run backfill every existing play.
CREATE STREAM IF NOT EXISTS SYNCED.FOURTH_DOWN_SCORED_CHANGES
  ON TABLE SYNCED.FOURTH_DOWN_SCORED
  SHOW_INITIAL_ROWS = TRUE
  COMMENT = 'Changes to scored fourth downs, consumed by APP.MAINTAIN_WEEKLY_SUMMARY';

CREATE OR REPLACE TASK APP.MAINTAIN_WEEKLY_SUMMARY
  WAREHOUSE = GRIDIRON_WH
  SCHEDULE = 'USING CRON 0 14 * * 2 UTC'  -- Tuesday, after the Databricks job and sync
  COMMENT = 'Recompute weekly fourth-down summary rows for groups with changed plays'
  WHEN SYSTEM$STREAM_HAS_DATA('GRIDIRON.SYNCED.FOURTH_DOWN_SCORED_CHANGES')
AS
EXECUTE IMMEDIATE $$
BEGIN
  BEGIN TRANSACTION;

  DELETE FROM APP.WEEKLY_SUMMARY_CHANGED_KEYS;

  -- Consuming the stream in a DML statement advances its offset when the transaction
  -- commits. Both the before and after images of updated rows are included.
  INSERT INTO APP.WEEKLY_SUMMARY_CHANGED_KEYS (SEASON, SEASON_TYPE, WEEK, TEAM)
  SELECT DISTINCT SEASON, SEASON_TYPE, WEEK, POSTEAM
  FROM SYNCED.FOURTH_DOWN_SCORED_CHANGES;

  MERGE INTO APP.WEEKLY_FOURTH_DOWN_SUMMARY t
  USING (
    SELECT
      f.SEASON,
      f.SEASON_TYPE,
      f.WEEK,
      f.POSTEAM AS TEAM,
      COUNT(*) AS FOURTH_DOWNS,
      COUNT_IF(f.DECISION = 'go') AS GO_ATTEMPTS,
      COUNT_IF(f.OUTCOME = 'converted') AS CONVERSIONS,
      COUNT_IF(f.RECOMMENDATION = 'go') AS GO_RECOMMENDATIONS,
      COUNT_IF(f.FOLLOWED_MODEL) AS FOLLOWED_MODEL,
      SUM(f.EXPECTED_POINTS_LOST) AS EXPECTED_POINTS_LOST
    FROM SYNCED.FOURTH_DOWN_SCORED f
    JOIN APP.WEEKLY_SUMMARY_CHANGED_KEYS k
      ON f.SEASON = k.SEASON AND f.SEASON_TYPE = k.SEASON_TYPE
     AND f.WEEK = k.WEEK AND f.POSTEAM = k.TEAM
    GROUP BY 1, 2, 3, 4
  ) s
  ON t.SEASON = s.SEASON AND t.SEASON_TYPE = s.SEASON_TYPE
 AND t.WEEK = s.WEEK AND t.TEAM = s.TEAM
  WHEN MATCHED THEN UPDATE SET
    FOURTH_DOWNS = s.FOURTH_DOWNS,
    GO_ATTEMPTS = s.GO_ATTEMPTS,
    CONVERSIONS = s.CONVERSIONS,
    GO_RECOMMENDATIONS = s.GO_RECOMMENDATIONS,
    FOLLOWED_MODEL = s.FOLLOWED_MODEL,
    EXPECTED_POINTS_LOST = s.EXPECTED_POINTS_LOST,
    UPDATED_AT = CURRENT_TIMESTAMP()
  WHEN NOT MATCHED THEN INSERT
    (SEASON, SEASON_TYPE, WEEK, TEAM, FOURTH_DOWNS, GO_ATTEMPTS, CONVERSIONS,
     GO_RECOMMENDATIONS, FOLLOWED_MODEL, EXPECTED_POINTS_LOST, UPDATED_AT)
  VALUES
    (s.SEASON, s.SEASON_TYPE, s.WEEK, s.TEAM, s.FOURTH_DOWNS, s.GO_ATTEMPTS, s.CONVERSIONS,
     s.GO_RECOMMENDATIONS, s.FOLLOWED_MODEL, s.EXPECTED_POINTS_LOST, CURRENT_TIMESTAMP());

  -- Groups whose plays were all deleted upstream.
  DELETE FROM APP.WEEKLY_FOURTH_DOWN_SUMMARY t
  USING APP.WEEKLY_SUMMARY_CHANGED_KEYS k
  WHERE t.SEASON = k.SEASON AND t.SEASON_TYPE = k.SEASON_TYPE
    AND t.WEEK = k.WEEK AND t.TEAM = k.TEAM
    AND NOT EXISTS (
      SELECT 1 FROM SYNCED.FOURTH_DOWN_SCORED f
      WHERE f.SEASON = k.SEASON AND f.SEASON_TYPE = k.SEASON_TYPE
        AND f.WEEK = k.WEEK AND f.POSTEAM = k.TEAM
    );

  COMMIT;
END;
$$;

-- Tasks are created suspended.
ALTER TASK APP.MAINTAIN_WEEKLY_SUMMARY RESUME;

-- Run once now instead of waiting for Tuesday, then inspect:
-- EXECUTE TASK APP.MAINTAIN_WEEKLY_SUMMARY;
-- SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(TASK_NAME => 'MAINTAIN_WEEKLY_SUMMARY'));
-- SELECT * FROM APP.WEEKLY_FOURTH_DOWN_SUMMARY ORDER BY SEASON DESC, WEEK DESC LIMIT 20;
