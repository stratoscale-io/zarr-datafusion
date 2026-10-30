{#
  Register a Zarr store as a table, for use from `on-run-start`:

    on-run-start:
      - "{{ zarr_source('era5', 'gs://bucket/era5.zarr', schema='raw') }}"

  The location is resolved by the server, not by dbt. Declare the same table
  in a sources .yml so `source()` and lineage work.

  The table is replaced on every run, so a changed location takes effect (a
  Zarr table holds only metadata; the server refreshes views reading it).
#}

{% macro zarr_source(name, location, schema=none) %}
  {%- set schema = schema or target.schema -%}
  {%- if execute -%}
    {% do run_query("create schema if not exists " ~ target.database ~ "." ~ schema) %}
    {% do run_query(
        "create or replace external table " ~ target.database ~ "." ~ schema ~ "." ~ name
        ~ " stored as zarr location '" ~ location ~ "'") %}
  {%- endif -%}
{% endmacro %}
