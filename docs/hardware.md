# Hardware

`scripts/bootstrap.py` supports `auto`, `cpu`, `cuda12`, `cuda13`, and `tpu` modes. Auto mode falls back to CPU. Explicit accelerator requests should fail loudly when the requested backend cannot be verified in future full training workflows.
