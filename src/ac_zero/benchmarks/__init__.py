"""Benchmark evaluation: the AK/MS presentation catalog and the run that scores a model.

Three pieces, in the order a benchmark run uses them:

* :mod:`catalog` enumerates every Akbulut-Kirby and Miller-Schupp presentation
  under a relator-length bound and writes it as a standalone catalog file.
* :mod:`evaluation` walks a catalog with a trained checkpoint: a cheap classical
  scan over every entry first, then a model-guided search on what the scan missed.
* :mod:`results` shapes the run into the summary + detail documents published to
  Hugging Face under ``benchmarks/``.

These presentations are literature candidates, never training data -- see the
leakage warning carried on every catalog entry.
"""
