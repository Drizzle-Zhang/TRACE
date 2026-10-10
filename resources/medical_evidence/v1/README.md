# Medical evidence resource bundle v1

This bundle provides endpoint-specific evidence tables, feature mappings, and
disease-name metadata for `MedicalEvidenceStore`.

## Files

| File | Purpose |
| --- | --- |
| `risk_factor_evidence.csv` | Risk-factor statement records. |
| `decision_screening_evidence.csv` | Decision and screening statement records. |
| `consensus_evidence.csv` | Factor and action summary records. |
| `disease_evidence_feature_map.csv` | Evidence-factor to model-feature mappings. |
| `disease_master.csv` | Disease identifiers, names, aliases, and model labels. |
| `compressed_disease_resolution.csv` | Compressed disease names and endpoint-name mappings. |
| `disease_metadata/outcome_filter.csv` | Endpoint `NAME` and `LONGNAME` metadata. |
| `disease_metadata/outcome_test.csv` | Additional endpoint `NAME` and `LONGNAME` metadata. |
| `disease_metadata/disease_map_full.csv` | Original and compressed disease-name mappings. |
| `pipeline_summary.json` | Bundle version and table row counts. |
| `SHA256SUMS` | SHA-256 digests of the distributed CSV and JSON files. |

## Loading

Set `consistency.evidence_bundle` to this directory. The example configurations
`configs/examples/prevalent-consistency.yaml` and
`configs/examples/longitudinal-consistency.yaml` use
`../../resources/medical_evidence/v1` relative to their configuration files.

`MedicalEvidenceStore` discovers the endpoint tables under `disease_metadata/`.
Set `consistency.disease_metadata_dir` to load those tables from another directory.
The loader combines evidence factors with model features using
`disease_evidence_feature_map.csv` and matches endpoint names using the disease
metadata and `compressed_disease_resolution.csv`.

## Metadata columns

The distributed disease metadata uses English column names:

- `disease_map_full.csv`: `original_longname`, `HD_ICD_10_v1`,
  `compressed_disease_name`, `disease_category`, and `independent_card`.
- `disease_master.csv` and `compressed_disease_resolution.csv`:
  `disease_category` stores category labels.
- `outcome_filter.csv` and `outcome_test.csv`: `NAME` identifies the endpoint
  and `LONGNAME` provides its expanded name.

## File verification

Run from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```
