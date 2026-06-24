#!/usr/bin/env sh
set -eu
for cert in runs/*/certificates/*.json; do
  [ -e "$cert" ] || continue
  uv run --frozen aczero certificate verify "$cert"
done
