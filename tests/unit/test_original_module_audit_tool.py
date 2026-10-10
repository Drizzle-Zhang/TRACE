"""The source audit must reject changed scripts before executing code."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).parents[2] / "tools/audit_original_module_kernels.py"


def test_source_audit_rejects_unverified_sources_without_top_level_execution(tmp_path):
    sentinel = tmp_path / "executed"
    source = tmp_path / "changed.py"
    source.write_text(f"open({str(sentinel)!r}, 'w').write('executed')\n")
    result = subprocess.run(
        [sys.executable, str(TOOL), "--lightgbm-source", str(source), "--cox-source", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert not sentinel.exists()


def test_ast_extraction_never_runs_top_level_io(tmp_path):
    spec = importlib.util.spec_from_file_location("original_module_audit_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentinel = tmp_path / "executed"
    source = tmp_path / "isolated.py"
    source.write_text(
        f"open({str(sentinel)!r}, 'w').write('executed')\n"
        "def model_training():\n    return 3\n"
        "def pred_train_iter():\n    return 7\n"
    )
    namespace = module.extract_functions(source, hashlib.sha256(source.read_bytes()).hexdigest(), {})
    assert namespace["model_training"]() == 3
    assert namespace["pred_train_iter"]() == 7
    assert not sentinel.exists()
