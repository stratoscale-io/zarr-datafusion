{# A `table` is an in-memory table on the server (CREATE OR REPLACE TABLE AS),
   recomputed from its query if the server restarts with a journal. Use it for
   small results that many models read; prefer views otherwise. #}

{%- materialization table, adapter='zarr' -%}

  {%- set existing_relation = load_cached_relation(this) -%}
  {%- set target_relation = this.incorporate(type='table') -%}

  {{ run_hooks(pre_hooks, inside_transaction=False) }}
  {{ run_hooks(pre_hooks, inside_transaction=True) }}

  {% if existing_relation is not none and not existing_relation.is_table %}
    {{ drop_relation_if_exists(existing_relation) }}
  {% endif %}

  {% call statement('main') -%}
    {{ get_create_table_as_sql(False, target_relation, sql) }}
  {%- endcall %}

  {% do persist_docs(target_relation, model) %}

  {{ run_hooks(post_hooks, inside_transaction=True) }}
  {{ adapter.commit() }}
  {{ run_hooks(post_hooks, inside_transaction=False) }}

  {{ return({'relations': [target_relation]}) }}

{%- endmaterialization -%}
