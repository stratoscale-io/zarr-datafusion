{# The default view materialization builds an intermediate relation and
   renames it into place; DataFusion has no RENAME. CREATE OR REPLACE VIEW is
   a single atomic statement on the server instead. #}

{%- materialization view, adapter='zarr' -%}

  {%- set existing_relation = load_cached_relation(this) -%}
  {%- set target_relation = this.incorporate(type='view') -%}

  {{ run_hooks(pre_hooks, inside_transaction=False) }}
  {{ run_hooks(pre_hooks, inside_transaction=True) }}

  {% if existing_relation is not none and not existing_relation.is_view %}
    {{ drop_relation_if_exists(existing_relation) }}
  {% endif %}

  {% call statement('main') -%}
    {{ get_create_view_as_sql(target_relation, sql) }}
  {%- endcall %}

  {% do persist_docs(target_relation, model) %}

  {{ run_hooks(post_hooks, inside_transaction=True) }}
  {{ adapter.commit() }}
  {{ run_hooks(post_hooks, inside_transaction=False) }}

  {{ return({'relations': [target_relation]}) }}

{%- endmaterialization -%}
