{# Metadata for `dbt docs generate`. #}

{% macro zarr__get_catalog(information_schema, schemas) -%}
  {% set query %}
    select
      t.table_catalog as table_database,
      t.table_schema,
      t.table_name,
      t.table_type,
      cast(null as varchar) as table_comment,
      c.column_name,
      c.ordinal_position as column_index,
      c.data_type as column_type,
      cast(null as varchar) as column_comment,
      cast(null as varchar) as table_owner
    from information_schema.tables t
    join information_schema.columns c
      on t.table_catalog = c.table_catalog
     and t.table_schema = c.table_schema
     and t.table_name = c.table_name
    where t.table_catalog = '{{ information_schema.database }}'
      and t.table_schema in (
        {%- for schema in schemas -%}
          '{{ schema }}'{% if not loop.last %}, {% endif %}
        {%- endfor -%}
      )
    order by t.table_schema, t.table_name, c.ordinal_position
  {% endset %}
  {{ return(run_query(query)) }}
{%- endmacro %}
