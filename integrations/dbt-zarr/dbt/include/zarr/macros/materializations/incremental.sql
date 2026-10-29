{# Incremental models, `append` strategy only: the first run (or
   --full-refresh) creates an in-memory table, later runs INSERT the rows the
   model selects when `is_incremental()`.

   Inserted rows are not journaled by the server: after a restart the table is
   rebuilt from its original, full query, i.e. as if fully refreshed. #}

{%- materialization incremental, adapter='zarr' -%}

  {%- set existing_relation = load_cached_relation(this) -%}
  {%- set target_relation = this.incorporate(type='table') -%}
  {%- set strategy = config.get('incremental_strategy') or 'append' -%}
  {%- if strategy != 'append' -%}
    {{ exceptions.raise_compiler_error(
        "zarr-datafusion supports only the 'append' incremental strategy, not '" ~ strategy ~ "'") }}
  {%- endif -%}

  {{ run_hooks(pre_hooks, inside_transaction=False) }}
  {{ run_hooks(pre_hooks, inside_transaction=True) }}

  {% if existing_relation is none or not existing_relation.is_table or should_full_refresh() %}
    {% if existing_relation is not none and not existing_relation.is_table %}
      {{ drop_relation_if_exists(existing_relation) }}
    {% endif %}
    {% set build_sql = get_create_table_as_sql(False, target_relation, sql) %}
  {% else %}
    {%- set columns = adapter.get_columns_in_relation(target_relation) -%}
    {%- set column_list = columns | map(attribute='quoted') | join(', ') -%}
    {% set build_sql %}
      insert into {{ target_relation.render() }} ({{ column_list }})
      select {{ column_list }} from (
        {{ sql }}
      ) as dbt_incremental_rows
    {% endset %}
  {% endif %}

  {% call statement('main') -%}
    {{ build_sql }}
  {%- endcall %}

  {% do persist_docs(target_relation, model) %}

  {{ run_hooks(post_hooks, inside_transaction=True) }}
  {{ adapter.commit() }}
  {{ run_hooks(post_hooks, inside_transaction=False) }}

  {{ return({'relations': [target_relation]}) }}

{%- endmaterialization -%}
