-- 01_catalog_integration.sql
-- Iceberg path, for Databricks workspaces with external storage. NOT the path used with
-- Databricks Free Edition: there the metastore uses privilege model 1.0, so EXTERNAL USE
-- SCHEMA is not applicable, and the Unity Catalog Iceberg REST endpoint returns table
-- metadata but no vended storage credentials for tables on Databricks default storage, so
-- Snowflake cannot read the data files. Free Edition uses the sync instead
-- (snowflake/sync/ and scripts/sync_to_snowflake.py). Kept, and documented in the README,
-- for paid workspaces whose serving schema lives on external storage.
--
-- Run as ACCOUNTADMIN. Connects Snowflake to Databricks Unity Catalog's Iceberg REST
-- catalog so Snowflake reads the UniForm serving tables in place (no data copy).
--
-- Databricks side, before running this:
--   1. Metastore setting "External data access" is enabled.
--   2. The principal below has USE CATALOG on <DATABRICKS_CATALOG>, and USE SCHEMA,
--      EXTERNAL USE SCHEMA and SELECT on <DATABRICKS_SERVING_SCHEMA>.
--   3. The `gridiron-refresh` job has run, so the serving tables exist.
--
-- Placeholders:
--   <DATABRICKS_HOST>            workspace host, e.g. dbc-1234abcd-5678.cloud.databricks.com
--   <DATABRICKS_CATALOG>         Unity Catalog catalog
--   <DATABRICKS_SERVING_SCHEMA>  `gridiron_serving` (prod target)
--   <DATABRICKS_PAT>             personal access token of the principal in step 2
--
-- Option A (simplest, works with a personal access token): bearer authentication.

USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE CATALOG INTEGRATION GRIDIRON_UNITY_CATALOG
  CATALOG_SOURCE = ICEBERG_REST
  TABLE_FORMAT = ICEBERG
  CATALOG_NAMESPACE = '<DATABRICKS_SERVING_SCHEMA>'
  REST_CONFIG = (
    CATALOG_URI = 'https://<DATABRICKS_HOST>/api/2.1/unity-catalog/iceberg-rest'
    CATALOG_NAME = '<DATABRICKS_CATALOG>'
    ACCESS_DELEGATION_MODE = VENDED_CREDENTIALS
  )
  REST_AUTHENTICATION = (
    TYPE = BEARER
    BEARER_TOKEN = '<DATABRICKS_PAT>'
  )
  REFRESH_INTERVAL_SECONDS = 300
  ENABLED = TRUE
  COMMENT = 'Databricks Unity Catalog Iceberg REST catalog (gridiron serving tables)';

-- Option B (preferred for anything long-lived): OAuth machine-to-machine with a
-- Databricks service principal, so no personal token is stored in Snowflake.
--
-- CREATE OR REPLACE CATALOG INTEGRATION GRIDIRON_UNITY_CATALOG
--   CATALOG_SOURCE = ICEBERG_REST
--   TABLE_FORMAT = ICEBERG
--   CATALOG_NAMESPACE = '<DATABRICKS_SERVING_SCHEMA>'
--   REST_CONFIG = (
--     CATALOG_URI = 'https://<DATABRICKS_HOST>/api/2.1/unity-catalog/iceberg-rest'
--     CATALOG_NAME = '<DATABRICKS_CATALOG>'
--     ACCESS_DELEGATION_MODE = VENDED_CREDENTIALS
--   )
--   REST_AUTHENTICATION = (
--     TYPE = OAUTH
--     OAUTH_TOKEN_URI = 'https://<DATABRICKS_HOST>/oidc/v1/token'
--     OAUTH_CLIENT_ID = '<SERVICE_PRINCIPAL_APPLICATION_ID>'
--     OAUTH_CLIENT_SECRET = '<SERVICE_PRINCIPAL_OAUTH_SECRET>'
--     OAUTH_ALLOWED_SCOPES = ('all-apis')
--   )
--   REFRESH_INTERVAL_SECONDS = 300
--   ENABLED = TRUE;

GRANT USAGE ON INTEGRATION GRIDIRON_UNITY_CATALOG TO ROLE GRIDIRON_ADMIN;

-- Should return a success payload. If it fails, check external data access, the
-- EXTERNAL USE SCHEMA grant, and the token.
SELECT SYSTEM$VERIFY_CATALOG_INTEGRATION('GRIDIRON_UNITY_CATALOG');
