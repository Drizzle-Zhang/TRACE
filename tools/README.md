# Model packaging

Build archives from explicitly listed local model files. Model files and
generated packages are runtime inputs and outputs; they are supplied separately.

## Package explicitly listed files

`package_model_files.py` builds local archives from a JSON specification:

```json
{
  "output": "./data/model_packages",
  "groups": [
    {
      "name": "example",
      "files": [
        {
          "source": "./data/models/example.joblib",
          "member": "module_models/example.joblib",
          "kind": "model"
        },
        {
          "source": "./reference_code/__init__.py",
          "member": "reference_code/__init__.py",
          "kind": "reference_code"
        }
      ]
    }
  ]
}
```

Save the specification as `data/package_specification.json`. Run from the directory relative to which the specification's paths are defined; the example keeps packages and completion records in ignored `data/`:

```sh
python tools/package_model_files.py data/package_specification.json
```

The importable functions are `package_group(group, output)` and
`package_specification(specification)`. Each file entry supplies a local `source`,
an archive `member`, and a `kind`; an optional `expected_size` checks the source
size before packaging. Files must be nonempty unless their kind is
`reference_code`, which permits empty Python initializers.

Each archive includes `PACKAGE_MANIFEST.json` with schema
`trace.legacy_model_package.v1`, the source paths, sizes, modification times, and
SHA-256 hashes. File bytes are copied unchanged. A group archive and its JSON
completion record are published through temporary files and atomic replacement.
`packages.json` lists the completed group records. A repeated run reuses an
archive when its recorded SHA-256 and model count match. Use a new group name or
output directory to package changed sources.
