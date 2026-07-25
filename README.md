# ACOR: Affective Context-Aware Omni-Modal Reasoning

Official implementation of **ACOR: Affective Context-Aware Omni-Modal
Reasoning for Emotion-Centric Human Understanding**.

ACOR extends structured omni-modal reasoning with an affective `<context>`
stage, two emotion-aware rewards, staged GRPO training, and adapter
interpolation/repair. The public release contains code only: no datasets,
model weights, checkpoints, evaluation outputs, logs, or API credentials.

## What is reproducible

1. Affective-context SFT with `<context><think><answer>` targets.
2. Format, accuracy, affective-context, and Emotion Category Consistency rewards.
3. Construction of ACOR-Balanced, ACOR-Affective, ACOR-Repair, and ACOR-A.R.
4. MER2024 output parsing and metrics.
5. IntentBench, Daily-Omni, and WorldSense evaluation through the retained
   HumanOmniV2-compatible evaluator.

## Repository structure

```text
acor/                 prompts, parsers, rewards, adapter utilities
training/             adapter interpolation and soup entrypoint
configs/              SFT/GRPO/repair settings and model manifest
evaluation/           shared metrics and MER2024 evaluation
scripts/              retained training/evaluation launchers
src/open_r1/          HumanOmniV2-compatible SFT and GRPO integration
```

## Environment

- Python 3.10
- CUDA 12.2 or 12.3
- PyTorch and Transformers versions are recorded in `environment.yml`
- Base model: Qwen2.5-Omni-7B

```bash
conda env create -f environment.yml
conda activate humanomni
export PYTHONPATH="$PWD:$PWD/src"
```

The base model and all third-party datasets remain subject to their original
licenses. The complete HumanOmniV2 repository is not vendored here.

## Data preparation

Set local paths through environment variables:

```bash
export BASE_MODEL=/path/to/Qwen2.5-Omni-7B
export SFT_CHECKPOINT=/path/to/acor-sft
export DATA_ROOT=/path/to/training-data
export OUTPUT_ROOT=/path/to/output
export INTENTBENCH_ROOT=/path/to/IntentBench
export DAILY_OMNI_ROOT=/path/to/Daily-Omni
export WORLDSENSE_ROOT=/path/to/WorldSense
```

Raw datasets are not redistributed.

## Affective-context SFT

`acor/prompts.py` exposes `build_affective_prompt`,
`build_structured_target`, and `format_sft_sample`. The retained training
integration is in `src/open_r1/sft.py`.

```bash
bash scripts/train/run_sft_qwenomni.sh
```

## GRPO and rewards

The dependency-light public reward API is in `acor/rewards.py`:

- `format_reward`
- `accuracy_reward`
- `affective_context_reward`
- `emotion_consistency_reward`

The paper name **Emotion Category Consistency Reward** corresponds to
`emotion_consistency_reward`.

The actual GRPO training entrypoint calls the corresponding methods in
`src/open_r1/vlm_modules/qwenomni_module.py`; the smaller public module
preserves the same five-stage affective scoring rule for testing and reuse.

The optional LLM judge reads credentials only from the environment:

```bash
export LLM_JUDGE_API_BASE=https://your-openai-compatible-endpoint/v1/chat/completions
export LLM_JUDGE_API_KEY=...
export LLM_JUDGE_MODEL=qwen-plus
```

Defaults are temperature `0`, maximum output length `16`, and a normalized
0–5 score. The exact prompt and score parser are public in `acor/rewards.py`.

The paper configuration uses 8 generations, completion length 768, 16 video
frames, audio input, micro-batch size 1, and LoRA dropout 0.05. Stage-specific
reward weights are recorded under `configs/`.

```bash
bash scripts/train/run_acor_ablation_A3_full.sh
```

## ACOR variants and ACOR-A.R.

The verified lineage is recorded in `configs/model_manifest.yaml`:

- ACOR-Balanced: emotion-preserving checkpoint B500.
- ACOR-Affective: B500/B700 adapter interpolation with `alpha=0.20`.
- ACOR-Repair: 50-step anti-regression run; checkpoint 20 selected.
- ACOR-A.R.: interpolation of ACOR-Affective and repair checkpoint 20 with
  `lambda=0.25` (`pair_alpha20_repair20_lam025`).

```bash
python training/interpolate_and_soup.py interpolate \
  --source-a /path/to/ACOR-Affective \
  --source-b /path/to/repair/checkpoint-20 \
  --alpha 0.25 \
  --output /path/to/ACOR-A.R.
```

No full 7B weights are required; compatible LoRA adapters are sufficient.

The exact repair entrypoint is:

```bash
bash training/train_repair.sh
```

## MER2024

Evaluate saved free-generation outputs:

```bash
python evaluation/eval_mer2024.py \
  --predictions /path/to/predictions.jsonl \
  --output /path/to/metrics.json
```

This reports Accuracy, Macro-F1, and Parse Valid.

## Omni-modal benchmarks

The retained evaluator supports IntentBench, Daily-Omni, and WorldSense with
the same prompt, generation path, and parser:

```bash
python scripts/eval/eval_humanomniv2.py --dataset ib --model-path /path/to/model
python scripts/eval/eval_humanomniv2.py --dataset daily --model-path /path/to/model
python scripts/eval/eval_humanomniv2.py --dataset world --model-path /path/to/model
```

## Security and release policy

- Never commit raw data, model weights, checkpoints, logs, caches, or outputs.
- Never hard-code API keys, server addresses, usernames, or absolute paths.
- Keep this GitHub repository private until the final citation and public
  adapter release have passed review.

## License

Apache License 2.0. Upstream notices are retained. Third-party models and
datasets follow their own licenses.

## Citation

The official ACOR citation will be added when the paper metadata is public.
