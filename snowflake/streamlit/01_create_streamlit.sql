-- 01_create_streamlit.sql
-- Run as GRIDIRON_ADMIN from the repository root with SnowSQL or the Snowflake CLI
-- (or upload the three files to the stage in Snowsight and run only the CREATE).
-- Requires the dbt marts (MARTS schema) and the UDF from snowpark/01_create_udf.sql.

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE SCHEMA APP;
USE WAREHOUSE GRIDIRON_WH;

CREATE STAGE IF NOT EXISTS APP.STREAMLIT_STAGE DIRECTORY = (ENABLE = TRUE);

PUT file://snowflake/streamlit/streamlit_app.py @APP.STREAMLIT_STAGE/fourth_down_explorer AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://snowflake/streamlit/data_access.py @APP.STREAMLIT_STAGE/fourth_down_explorer AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://snowflake/streamlit/environment.yml @APP.STREAMLIT_STAGE/fourth_down_explorer AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

CREATE OR REPLACE STREAMLIT APP.FOURTH_DOWN_EXPLORER
  FROM '@GRIDIRON.APP.STREAMLIT_STAGE/fourth_down_explorer'
  MAIN_FILE = 'streamlit_app.py'
  -- Warehouse runtime (packages from environment.yml). New accounts otherwise default to the
  -- container runtime, which needs a compute pool that trial accounts can't always start.
  RUNTIME_NAME = 'SYSTEM$WAREHOUSE_RUNTIME'
  QUERY_WAREHOUSE = GRIDIRON_WH
  TITLE = 'Fourth Down Explorer'
  COMMENT = 'NFL fourth-down decisions vs. an expected-points model';

-- Viewers need only the reader role.
GRANT USAGE ON STREAMLIT APP.FOURTH_DOWN_EXPLORER TO ROLE GRIDIRON_READER;
