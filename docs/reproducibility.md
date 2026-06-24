# Reproducibility

Runs should record resolved configuration, seed, dataset checksum, Git revision, dirty state, lockfile checksum, package versions, backend metadata, and checkpoint schema. CPU smoke tests are deterministic. Future accelerator training may not be bitwise deterministic across devices or compiler versions.
