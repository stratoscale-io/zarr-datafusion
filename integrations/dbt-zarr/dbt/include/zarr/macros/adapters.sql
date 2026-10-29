{# DataFusion takes schema names as plain `catalog.schema` strings, so schema
   statements are rendered unquoted. #}

{% macro zarr__create_schema(relation) -%}
  {%- call statement('create_schema') -%}
    create schema if not exists {{ relation.database }}.{{ relation.schema }}
  {%- endcall -%}
{% endmacro %}

{% macro zarr__drop_schema(relation) -%}
  {%- call statement('drop_schema') -%}
    drop schema if exists {{ relation.database }}.{{ relation.schema }} cascade
  {%- endcall -%}
{% endmacro %}

{% macro zarr__list_schemas(database) -%}
  {% set sql %}
    select schema_name
    from information_schema.schemata
    where catalog_name = '{{ database }}'
  {% endset %}
  {{ return(run_query(sql)) }}
{% endmacro %}

{% macro zarr__check_schema_exists(information_schema, schema) -%}
  {% set sql %}
    select count(*)
    from information_schema.schemata
    where catalog_name = '{{ information_schema.database }}'
      and schema_name = '{{ schema }}'
  {% endset %}
  {{ return(run_query(sql)) }}
{% endmacro %}

{% macro zarr__list_relations_without_caching(schema_relation) %}
  {% call statement('list_relations_without_caching', fetch_result=True) -%}
    select
      table_catalog as database,
      table_name as name,
      table_schema as schema,
      case when table_type = 'VIEW' then 'view' else 'table' end as type
    from information_schema.tables
    where table_catalog = '{{ schema_relation.database }}'
      and table_schema = '{{ schema_relation.schema }}'
  {%- endcall %}
  {{ return(load_result('list_relations_without_caching').table) }}
{% endmacro %}

{% macro zarr__get_columns_in_relation(relation) -%}
  {% call statement('get_columns_in_relation', fetch_result=True) %}
    select
      column_name,
      data_type,
      cast(null as bigint) as character_maximum_length,
      cast(null as bigint) as numeric_precision,
      cast(null as bigint) as numeric_scale
    from information_schema.columns
    where table_catalog = '{{ relation.database }}'
      and table_schema = '{{ relation.schema }}'
      and table_name = '{{ relation.identifier }}'
    order by ordinal_position
  {% endcall %}
  {{ return(sql_convert_columns_in_relation(load_result('get_columns_in_relation').table)) }}
{% endmacro %}

{# No CASCADE: DataFusion doesn't support it on DROP VIEW / DROP TABLE.
   zarr-flightsql deregisters views whose upstream is dropped and re-creates
   them when it comes back. #}

{% macro zarr__drop_view(relation) -%}
  drop view if exists {{ relation.render() }}
{%- endmacro %}

{% macro zarr__drop_table(relation) -%}
  drop table if exists {{ relation.render() }}
{%- endmacro %}

{% macro zarr__get_drop_sql(relation) -%}
  {%- if relation.is_view -%}
    {{ drop_view(relation) }}
  {%- else -%}
    {{ drop_table(relation) }}
  {%- endif -%}
{%- endmacro %}

{% macro zarr__rename_relation(from_relation, to_relation) -%}
  {{ exceptions.raise_compiler_error(
      "zarr-datafusion cannot rename relations (" ~ from_relation ~ " -> " ~ to_relation ~ ")") }}
{% endmacro %}

{% macro zarr__truncate_relation(relation) -%}
  {{ exceptions.raise_compiler_error(
      "zarr-datafusion cannot truncate relations (" ~ relation ~ ")") }}
{% endmacro %}

{% macro zarr__current_timestamp() -%}
  now()
{%- endmacro %}

{# Replacing in one statement: the server swaps the relation atomically and
   re-creates the views that read it. #}

{% macro zarr__create_view_as(relation, sql) -%}
  create or replace view {{ relation.render() }} as
  {{ sql }}
{%- endmacro %}

{% macro zarr__create_table_as(temporary, relation, sql) -%}
  create or replace table {{ relation.render() }} as
  {{ sql }}
{%- endmacro %}
