# Source and resource provenance

TRACE code is distributed under the repository's MIT license. Data and model
assets require their own applicable access and licensing terms. See
[methods](../methods.md) for the algorithms and [validation](../validation.md)
for evaluation requirements.

## Source fingerprints

[source_fingerprints.json](source_fingerprints.json) identifies the reference
implementations and resources used by the numerical tests. Each entry states
what was hashed. These records support implementation comparisons; they do not
replace the cohort, preprocessing and configuration needed to reproduce a study.

## Medical evidence and review status

The [medical evidence bundle](../../resources/medical_evidence/v1/README.md)
contains 1,704 template-generated risk-factor and decision/screening statements.
Their institution and guideline labels were assigned by templates; certainty
and strength were imputed. Original citations and medical review have not been
established. The consensus table aggregates individual template statements,
without independent multi-source agreement.

Consistency scores measure agreement with these rules and feature mappings.
They do not establish clinical correctness or current guideline adherence.
Use source-derived, endpoint-appropriate evidence and independent medical review
when making claims about medical consistency.

The [reference annotations](../../resources/medical_evidence/v1/source_annotations.csv)
provide scoped paraphrases and source links. They remain
pending clinical review, do not validate entire template statements, and are
excluded from runtime scoring. Their population, threshold and source-version
boundaries are described in the bundle documentation.

## Resource versions

[SHA256SUMS](../../resources/medical_evidence/v1/SHA256SUMS) identifies the
distributed files. The [translation manifest](../../resources/medical_evidence/v1/translation_manifest.json)
records source and translated versions separately. Translation preserves
endpoint identifiers, model labels, feature mappings and scoring values; it
does not establish medical review.

Saved consistency rules require the matching model and resource files. Retain
that complete set when applying rules to another cohort. After changing the
evidence bundle, recompute consistency and select rules on validation data;
do not replace recorded hashes to bypass a resource mismatch. Metadata-only
translation does not require refitting risk estimators.

## Research data and model records

Participant-level data, identifiers, outcomes, splits, attributions and
predictions are not distributed. Prepare authorized data according to the
[input contracts](../data_contracts.md). Synthetic examples check the workflow
without providing evidence of real-cohort performance.

Keep model manifests, input and split identities, selected features, cohort and
outcome definitions, configuration and environment with experiment results.
For externally prepared scores, confirm label identity and OOF origin upstream;
format checks alone cannot establish how their training data were used.
