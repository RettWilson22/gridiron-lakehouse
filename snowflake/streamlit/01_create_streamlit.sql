-- 01_create_streamlit.sql
-- Run as GRIDIRON_ADMIN with the Snowflake CLI (snow sql -f) from the repository root;
-- PUT needs a client. In Snowsight, upload the four files to the stage instead.
-- Requires the dbt marts (MARTS schema) and APP.FANTASY_POINTS (snowpark/01_create_udf.sql).

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE SCHEMA APP;
USE WAREHOUSE GRIDIRON_WH;

CREATE STAGE IF NOT EXISTS APP.STREAMLIT_STAGE DIRECTORY = (ENABLE = TRUE);

PUT file://snowflake/streamlit/streamlit_app.py @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://snowflake/streamlit/data_access.py @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://snowflake/streamlit/environment.yml @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
-- Tiering helper shared with the Databricks score task (standard library only).
PUT file://src/gridiron/tiers.py @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

CREATE OR REPLACE STREAMLIT APP.FANTASY_CHEAT_SHEET
  FROM '@GRIDIRON.APP.STREAMLIT_STAGE/cheat_sheet'
  MAIN_FILE = 'streamlit_app.py'
  -- Warehouse runtime (packages from environment.yml). New accounts otherwise default to the
  -- container runtime, which needs a compute pool that trial accounts can't always start.
  RUNTIME_NAME = 'SYSTEM$WAREHOUSE_RUNTIME'
  QUERY_WAREHOUSE = GRIDIRON_WH
  TITLE = 'Fantasy Football Cheat Sheet'
  COMMENT = 'Weekly fantasy projections, tiers, start/sit and track record';

-- Viewers need only the reader role.
GRANT USAGE ON STREAMLIT APP.FANTASY_CHEAT_SHEET TO ROLE GRIDIRON_READER;
