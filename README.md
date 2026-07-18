# ACOR: Affective Context-Aware Omni-Modal Reasoning for Emotion-Centric Human Understanding

## Overview
This repository provides the official implementation of ACOR, an affective context-aware omni-modal reasoning framework for emotion-centric human understanding.

Included:
- ACOR SFT entrypoint
- ACOR reward optimization
- A0–A3 matched reward ablations
- benchmark evaluation
- paper analysis

Not included:
- model weights
- datasets
- interpolation code
- pair_lam025 generation
- private API credentials

## Public Release Scope
This repository is a code-only public release. It preserves the training and evaluation entrypoints needed to reproduce the public pipeline when the required external assets are provided locally.

## Repository Structure
- `configs/sft/`: SFT YAML configs
- `configs/grpo/`: GRPO YAML configs
- `configs/deepspeed/`: DeepSpeed configs
- `scripts/train/`: SFT and A0–A3 training commands
- `scripts/eval/`: benchmark evaluation entrypoints
- `scripts/analysis/`: paper analysis helpers
- `src/open_r1/`: ACOR training, reward, trainer, and multimodal modules

## Installation
Follow your existing Python environment setup for the upstream dependencies listed in `requirements.txt` and `environment.yml`.

## Required Environment Variables
Set the external resources before running any public entrypoint:
- `BASE_MODEL`
- `SFT_CHECKPOINT`
- `DATA_ROOT`
- `OUTPUT_ROOT`
- `INTENTBENCH_ROOT`
- `DAILY_OMNI_ROOT`
- `WORLDSENSE_ROOT`
- `API_BASE`
- `API_KEY`

The training scripts also require strict shell checks for these variables, and the YAML loaders expand `${DATA_ROOT}` at load time.

## SFT Command
```bash
bash scripts/train/run_sft_qwenomni.sh
```

## A0–A3 GRPO Commands
```bash
bash scripts/train/run_acor_ablation_A0_fa_only.sh
bash scripts/train/run_acor_ablation_A1_affctx_only.sh
bash scripts/train/run_acor_ablation_A2_emocons_only.sh
bash scripts/train/run_acor_ablation_A3_full.sh
```

## Evaluation Command
```bash
python scripts/eval/eval_humanomniv2.py
```

## Analysis Command
```bash
python scripts/analysis/acor_paper_stats.py
python scripts/analysis/collect_acor_eval_summary.py
```

## Dataset Preparation
Benchmark datasets are not bundled. Prepare local copies and point the required environment variables to those locations before running the scripts.

## Checkpoint Note
The repository does not include model checkpoints. Use `SFT_CHECKPOINT` to point at your local ACOR-SFT checkpoint.

## Reproducibility Limitation
This release preserves the public code path, but it does not ship the private datasets, checkpoints, or external judge/API credentials required for exact end-to-end reproduction.

## License

This repository is released under the Apache License 2.0.
It contains code derived from HumanOmniV2, whose models and code
are also distributed under the Apache License 2.0. Upstream
copyright and license notices are retained in the derived files.

See [LICENSE](LICENSE) for the full license text.

The repository license applies to the code in this release.
Third-party models, datasets, and other external assets remain
subject to their respective licenses and terms.

ACOR is built on the training and evaluation framework of
HumanOmniV2. We thank the HumanOmniV2 authors for releasing
their code under the Apache License 2.0.

## Citation

The official ACOR citation will be added when the paper metadata is publicly available.
