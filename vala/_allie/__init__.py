"""Vendored minimal inference code for Allie v1 (the gpt2-based human model).

Source: github.com/ippolito-cmu/allie (MIT), `src/modeling/{moves,data,model}.py`,
from *Human-Aligned Chess With a Bit of Search* (Zhang et al., ICLR 2025).

Only the inference path is vendored. Removed vs upstream:
  - `load_data` + the `datasets`/`omegaconf` imports (training only).
  - `optimum`'s `BetterTransformer.transform` (verified a no-op for loading and
    bit-identical heads on CPU — see vala/allie.py), dropping that heavy dep.

The public entry point is `vala.allie.AlliePredictor`; treat this package as a
frozen black box (the upstream is a research repo, not a maintained library).
"""
