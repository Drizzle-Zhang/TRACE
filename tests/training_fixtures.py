"""Publish coherent test artifacts when exercising deeper semantic checks."""

import json
from pathlib import Path

from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact
from trace_agent.runtime.provenance import file_provenance
from trace_agent.training.manifests import write_training_manifest


def rewrite_training_artifact(artifact: TraceModelArtifact, path: str | Path) -> Path:
    """Keep the sidecar valid so axis/upstream tests reach their intended guard."""
    destination = dump_trace_artifact(artifact, path)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        artifact=file_provenance(destination),
        component=artifact.component,
        task=artifact.task.value,
        created_at=artifact.created_at,
        training=dict(artifact.training),
        provenance=dict(artifact.provenance),
        metrics=dict(artifact.metrics),
    )
    write_training_manifest(manifest_path, manifest)
    return destination
