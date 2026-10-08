-- 02_weekly_summary_dynamic_table.sql
-- Run as GRIDIRON_ADMIN. The Iceberg-path equivalent of the stream + task: a dynamic table
-- over the ICEBERG view. Databricks rewrites the serving table on each run (INSERT
-- OVERWRITE), so there is no useful row-level change feed to consume; a declarative
-- full refresh once a day is the honest fit. REFRESH_MODE = FULL is explicit because
-- incremental refresh is not expected to apply to an externally managed source.

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE WAREHOUSE GRIDIRON_WH;

CREATE OR REPLACE DYNAMIC TABLE APP.WEEKLY_FOURTH_DOWN_SUMMARY_DT
  TARGET_LAG = '1 day'
  WAREHOUSE = GRIDIRON_WH
  REFRESH_MODE = FULL
  COMMENT = 'Weekly fourth-down summary over the Iceberg serving table'
AS
SELECT
  SEASON,
  SEASON_TYPE,
  WEEK,
  POSTEAM AS TEAM,
  COUNT(*) AS FOURTH_DOWNS,
  COUNT_IF(DECISION = 'go') AS GO_ATTEMPTS,
  COUNT_IF(OUTCOME = 'converted') AS CONVERSIONS,
  COUNT_IF(RECOMMENDATION = 'go') AS GO_RECOMMENDATIONS,
  COUNT_IF(FOLLOWED_MODEL) AS FOLLOWED_MODEL,
  SUM(EXPECTED_POINTS_LOST) AS EXPECTED_POINTS_LOST
FROM ICEBERG.FOURTH_DOWN_SCORED
GROUP BY 1, 2, 3, 4;
