-- 01_create_udf.sql
-- Run as GRIDIRON_ADMIN with SnowSQL or the Snowflake CLI from the repository root
-- (PUT needs a client; in Snowsight, upload the two files to the stage through the UI).
--
-- Registers APP.FOURTH_DOWN_RECOMMENDATION, a Python UDF that evaluates the decision model
-- exported by the Databricks job (artifacts/decision_model.json, or download it from the
-- volume path /Volumes/<catalog>/<schema>/landing/models/decision_model.json).

USE ROLE GRIDIRON_ADMIN;
USE DATABASE GRIDIRON;
USE SCHEMA APP;
USE WAREHOUSE GRIDIRON_WH;

CREATE STAGE IF NOT EXISTS APP.MODELS
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Exported decision model and UDF code';

PUT file://snowflake/snowpark/fourth_down_udf.py @APP.MODELS AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
PUT file://artifacts/decision_model.json @APP.MODELS AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

CREATE OR REPLACE FUNCTION APP.FOURTH_DOWN_RECOMMENDATION(
    YDSTOGO FLOAT,
    YARDLINE_100 FLOAT,
    SCORE_DIFFERENTIAL FLOAT,
    GAME_SECONDS_REMAINING FLOAT
)
RETURNS OBJECT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'fourth_down_udf.udf_handler'
IMPORTS = ('@APP.MODELS/fourth_down_udf.py', '@APP.MODELS/decision_model.json')
COMMENT = 'Go / punt / field goal recommendation in expected points (gridiron decision model)';

-- 4th and 2 at the opponent 38, tied, 9 minutes left in the game.
SELECT APP.FOURTH_DOWN_RECOMMENDATION(2, 38, 0, 540) AS RESULT;

-- The model applied to every real play, compared with what the coach did.
SELECT
  GAME_ID, PLAY_ID, POSTEAM, DECISION,
  APP.FOURTH_DOWN_RECOMMENDATION(YDSTOGO, YARDLINE_100, SCORE_DIFFERENTIAL,
                                 GAME_SECONDS_REMAINING):recommendation::VARCHAR AS UDF_RECOMMENDATION,
  RECOMMENDATION AS DATABRICKS_RECOMMENDATION
FROM GRIDIRON.SYNCED.FOURTH_DOWN_SCORED  -- GRIDIRON.ICEBERG.FOURTH_DOWN_SCORED on the Iceberg path
WHERE SEASON = 2025
LIMIT 20;
