# Resolved run configurations

One JSON per paper run (110 = 80 central + 30 sweep): the complete configuration
the shared runner used, extracted from the `config` object of each run's
`*_metadata.json` in `reference_results/`.

Only the path-valued bookkeeping keys (`_original_config`, `_cyber_config`,
`_experiment_config`, `result_dir`, `config_template_source`) were rewritten to
release-relative paths; all values that affect computation are unchanged. The
verbatim originals, with the run-time container paths, remain in
`reference_results/<group>/.../*_metadata.json`.

`config_template_source` names the earlier experiment directory whose overlay
was used as a template when the run configs were generated. It is a provenance
label and is not read by any code.
