{# Seeds load in one statement (see ZarrAdapter.seed_create_sql): the default
   create-then-insert-with-bindings flow would need prepared statements and a
   TRUNCATE, which the server doesn't have. The default seed materialization
   is kept; only its helpers change. #}

{% macro zarr__create_csv_table(model, agate_table) %}
  {%- set sql = adapter.seed_create_sql(
      this.render(),
      agate_table,
      model['config'].get('column_types', {}),
      model['config'].get('quote_columns', none)) -%}
  {%- call statement('_') -%}
    {{ sql }}
  {%- endcall -%}
  {{ return(sql) }}
{% endmacro %}

{% macro zarr__reset_csv_table(model, full_refresh, old_relation, agate_table) %}
  {{ return(create_csv_table(model, agate_table)) }}
{% endmacro %}

{% macro zarr__load_csv_rows(model, agate_table) %}
  {{ return('') }}
{% endmacro %}
