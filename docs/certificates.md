# Certificates

Certificates are canonical JSON files containing the initial presentation, primitive moves, intermediate hashes, final presentation, goal mode, seed, and reproducibility metadata. Verification never loads a neural network.

Use:

```bash
uv run --frozen aczero certificate verify runs/smoke/certificates/example.json
```
