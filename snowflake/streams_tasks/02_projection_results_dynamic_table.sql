-- 02_projection_results_dynamic_table.sql
-- Iceberg path only (needs external storage; see the README). Run as GRIDIRON_ADMIN.
-- The equivalent of the stream + task when Snowflake reads the serving tables through
-- Iceberg: Databricks rewrites each serving table on every run (INSERT OVERWRITE), so there
-- is no useful row-level change feed, and a declarative full refresh once a day fits.

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE WAREHOUSE GRIDIRON_WH;

CREATE OR REPLACE DYNAMIC TABLE APP.PROJECTION_RESULTS_DT
  TARGET_LAG = '1 day'
  WAREHOUSE = GRIDIRON_WH
  REFRESH_MODE = FULL
  COMMENT = 'Projections next to actual results, over the Iceberg serving tables'
AS
SELECT
  p.SEASON, p.WEEK, p.PLAYER_ID, p.PLAYER_NAME, p.POSITION, p.TEAM, p.KIND,
  p.PROJ_PPR, p.FLOOR_PPR, p.CEILING_PPR, p.BASELINE_LAST3, p.BASELINE_SEASON_AVG,
  p.ECR_RANK,
  COALESCE(w.FANTASY_POINTS_PPR, 0) AS ACTUAL_PPR,
  w.PLAYER_ID IS NOT NULL AS PLAYED,
  p.PROJ_PPR - COALESCE(w.FANTASY_POINTS_PPR, 0) AS ERROR,
  COALESCE(w.FANTASY_POINTS_PPR, 0) BETWEEN p.FLOOR_PPR AND p.CEILING_PPR AS INSIDE_RANGE
FROM ICEBERG.PROJECTIONS p
JOIN ICEBERG.TEAM_WEEK g
  ON p.GAME_ID = g.GAME_ID AND p.TEAM = g.TEAM AND g.IS_FINAL
LEFT JOIN ICEBERG.PLAYER_WEEK w
  ON p.SEASON = w.SEASON AND p.WEEK = w.WEEK AND p.PLAYER_ID = w.PLAYER_ID;
