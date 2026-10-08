-- 01_warehouse_and_monitor.sql
-- Run as ACCOUNTADMIN (resource monitors require it).
--
-- Cost guard rails for a trial account:
--   * one X-Small warehouse (1 credit/hour while running) that suspends after 60 seconds;
--   * a monthly resource monitor that notifies at 50% / 80% and suspends the warehouse at
--     100% of the quota. Resource monitors only cover warehouses, which is why the task
--     in streams_tasks/ runs on this warehouse rather than on serverless compute.

USE ROLE ACCOUNTADMIN;

CREATE RESOURCE MONITOR IF NOT EXISTS GRIDIRON_MONITOR
  WITH CREDIT_QUOTA = 20
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON 50 PERCENT DO NOTIFY
    ON 80 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND
    ON 110 PERCENT DO SUSPEND_IMMEDIATE;

CREATE WAREHOUSE IF NOT EXISTS GRIDIRON_WH
  WAREHOUSE_SIZE = XSMALL
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Gridiron lakehouse: dbt, tasks, Streamlit and ad hoc queries';

ALTER WAREHOUSE GRIDIRON_WH SET RESOURCE_MONITOR = GRIDIRON_MONITOR;

-- Keep runaway queries from burning credits.
ALTER WAREHOUSE GRIDIRON_WH SET STATEMENT_TIMEOUT_IN_SECONDS = 1800;
