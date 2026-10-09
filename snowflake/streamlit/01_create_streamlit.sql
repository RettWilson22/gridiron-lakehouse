-- 01_create_streamlit.sql
-- Run with the Snowflake CLI (snow sql -f) from the repository root, as the user who ran
-- setup/02_database_and_roles.sql (it holds GRIDIRON_ADMIN and GRIDIRON_APP_OWNER); PUT
-- needs a client. In Snowsight, upload the five files to the stage instead.
-- Requires the dbt marts (MARTS schema) and APP.FANTASY_POINTS (snowpark/01_create_udf.sql).
--
-- The app is owned by GRIDIRON_APP_OWNER. Streamlit in Snowflake runs an app's queries with
-- its owner's rights, so the app can read the marts and call the scoring function and
-- nothing else (no SYNCED, STAGING or admin privileges).

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE SCHEMA APP;
USE WAREHOUSE GRIDIRON_WH;

CREATE STAGE IF NOT EXISTS APP.STREAMLIT_STAGE DIRECTORY = (ENABLE = TRUE);
-- The app owner copies the app files from this stage when it creates the app.
GRANT READ ON STAGE APP.STREAMLIT_STAGE TO ROLE GRIDIRON_APP_OWNER;

PUT file://snowflake/streamlit/streamlit_app.py @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://snowflake/streamlit/data_access.py @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://snowflake/streamlit/environment.yml @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
-- Tiering helper shared with the Databricks score task (standard library only).
PUT file://src/gridiron/tiers.py @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://src/gridiron/player_search.py @APP.STREAMLIT_STAGE/cheat_sheet AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

-- Earlier versions created the app as GRIDIRON_ADMIN. GRIDIRON_ADMIN can drop it either way
-- (it inherits GRIDIRON_APP_OWNER), so drop and recreate rather than replace.
DROP STREAMLIT IF EXISTS APP.FANTASY_CHEAT_SHEET;

USE ROLE GRIDIRON_APP_OWNER;
CREATE STREAMLIT APP.FANTASY_CHEAT_SHEET
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
