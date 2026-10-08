{#
  Use the configured schema name as-is (STAGING, MARTS) instead of dbt's default
  "<target_schema>_<custom_schema>" so the Snowflake grants in setup/ line up.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
