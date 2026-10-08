-- 02_iceberg_tables.sql
-- Run as GRIDIRON_ADMIN after 01_catalog_integration.sql.
-- Externally managed Iceberg tables: Databricks remains the only writer; Snowflake reads
-- the current snapshot through the catalog integration. AUTO_REFRESH polls the catalog
-- (REFRESH_INTERVAL_SECONDS on the integration) for new snapshots after each job run.
-- Table names are the Unity Catalog names, which are lower case.

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE SCHEMA ICEBERG_RAW;
USE WAREHOUSE GRIDIRON_WH;

CREATE ICEBERG TABLE IF NOT EXISTS FOURTH_DOWN_DECISIONS
  CATALOG = 'GRIDIRON_UNITY_CATALOG'
  CATALOG_TABLE_NAME = 'fourth_down_decisions'
  AUTO_REFRESH = TRUE;

CREATE ICEBERG TABLE IF NOT EXISTS FOURTH_DOWN_SCORED
  CATALOG = 'GRIDIRON_UNITY_CATALOG'
  CATALOG_TABLE_NAME = 'fourth_down_scored'
  AUTO_REFRESH = TRUE;

CREATE ICEBERG TABLE IF NOT EXISTS TEAM_SEASON_SUMMARY
  CATALOG = 'GRIDIRON_UNITY_CATALOG'
  CATALOG_TABLE_NAME = 'team_season_summary'
  AUTO_REFRESH = TRUE;

CREATE ICEBERG TABLE IF NOT EXISTS GAME_SUMMARY
  CATALOG = 'GRIDIRON_UNITY_CATALOG'
  CATALOG_TABLE_NAME = 'game_summary'
  AUTO_REFRESH = TRUE;

CREATE ICEBERG TABLE IF NOT EXISTS COACH_AGGRESSIVENESS
  CATALOG = 'GRIDIRON_UNITY_CATALOG'
  CATALOG_TABLE_NAME = 'coach_aggressiveness'
  AUTO_REFRESH = TRUE;

-- Manual refresh, if AUTO_REFRESH is not available on your account:
-- ALTER ICEBERG TABLE FOURTH_DOWN_SCORED REFRESH;

-- Check what Snowflake sees. Unity Catalog column names are lower case;
-- 03_iceberg_views.sql wraps each table in a view with upper-case column names so
-- downstream SQL does not depend on identifier case resolution.
SELECT COUNT(*) FROM FOURTH_DOWN_SCORED;
SELECT SYSTEM$AUTO_REFRESH_STATUS('GRIDIRON.ICEBERG_RAW.FOURTH_DOWN_SCORED');
