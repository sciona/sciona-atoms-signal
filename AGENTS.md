# Agent Instructions for sciona-atoms-signal

## Templated Dataset Confidentiality

Treat templated datasets and their metadata as non-public. Never commit dataset
contents, names, recording or subject identifiers, source paths, filenames,
directory layouts, schemas, channel inventories, excerpts, timestamps,
checksums, URLs, or generated adapters derived from a real template.

Use synthetic committed fixtures. Real-data evaluations must receive their
inputs through local runtime configuration excluded from Git, and committed
benchmark evidence may contain only opaque aliases and aggregate metrics.
Inspect staged evaluation and generated files for identifying metadata before
every commit. If publication appears necessary, stop and request an explicitly
approved public representation.
