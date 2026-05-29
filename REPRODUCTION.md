# GUDA Reproduction Guide

This guide is the canonical public entry point for reproducing the GUDA paper
experiments from this repository.

The paper reference is the arXiv version:
https://arxiv.org/abs/2601.22651

Full reproduction is GPU-expensive. The repository distributes source code and
UnlearnCanvas prompt/descriptor metadata only. Users must obtain CIFAR-10,
Stable Diffusion 1.5, and UnlearnCanvas images separately under the applicable
terms, then produce all checkpoints, generated images, caches, and score files
locally.

## Artifact Map

External inputs:

- CIFAR-10 images, downloaded by torchvision or provided under `CIFAR10/data/`.
- Stable Diffusion 1.5 weights, resolved by diffusers from the configured model
  ID or local cache.
- UnlearnCanvas images placed as
  `UnlearnCanvas/data/UnlearnCanvas/<Style>/<Object>/<id>.jpg`.
- Distributed UnlearnCanvas metadata:
  `UnlearnCanvas/data/train_prompts_ffsd_very_relaxed.jsonl`,
  `UnlearnCanvas/data/eval_prompts_ffsd_very_relaxed.jsonl`,
  `UnlearnCanvas/data/ffsd_very_relaxed_descriptors.json`, and
  `UnlearnCanvas/data/uc_index.json`.

Generated CIFAR-10 artifacts:

- Full model: `CIFAR10/outputs/models/cifar10/full/`.
- Conditional model for ESD: `CIFAR10/outputs/models/cifar10/conditional/`.
- LOGO models: `CIFAR10/outputs/models/cifar10/logo/exclude_<class>/`.
- Query images: `CIFAR10/results/cifar10/generated_queries/ddpm4000/`.
- GUDA-U models: `CIFAR10/outputs/models/cifar10/guda_retrack/unlearn_class_<class>/`.
- Optional ESD models: `CIFAR10/outputs/models/cifar10/guda_esd/unlearn_class_<class>/`.
- Attribution scores and ranking summaries under `CIFAR10/results/cifar10/`.

Generated UnlearnCanvas artifacts:

- CLIP style prototypes: `UnlearnCanvas/cache/clip_style_prototypes_very_relaxed.pt`.
- All-class SD1.5 checkpoint:
  `UnlearnCanvas/outputs/allclass_sd15_ffsd_very_relaxed/checkpoint_step_10000.pt`.
- LOGO checkpoints:
  `UnlearnCanvas/outputs/logo_sd15_ffsd_very_relaxed/<Style>/checkpoint_step_10000.pt`.
- Query images: `UnlearnCanvas/outputs/evaluation_images_ffsd_very_relaxed/`.
- GUDA-C checkpoints:
  `UnlearnCanvas/outputs/guda_awss/<Style>/checkpoint_step_<step>.pt`.
- LOGOA/GUDA-C scores and ranking summaries under `UnlearnCanvas/outputs/`.

## CIFAR-10 DAG

Run all CIFAR-10 commands from `CIFAR10/` with `PYTHONPATH=.`. The command
snippets in `docs/examples/cifar10_commands.md` show one concrete command for
each stage; `scripts/run_cifar10_reproduction.sh` is a plain shell example that
chains the main stages.

1. Train the full unconditional DDPM on CIFAR-10.
2. Train a class-conditional DDPM only if running the ESD comparison.
3. Train 10 LOGO models by excluding classes `0` through `9`.
4. Generate deterministic query images from the full model.
5. Train GUDA-U ReTrack models for classes `0` through `9`.
6. Optionally train GUDA-U ESD models for classes `0` through `9`.
7. Compute LOGOA scores using the full and LOGO checkpoints.
8. Compute GUDA-U scores using the full and unlearned checkpoints.
9. Compare LOGOA and GUDA rankings.
10. Run ReTrack ablations over unlearning epochs, learning rate,
    forget/preserve weighting, and nearest-neighbor count.

Important checkpoint naming:

- CIFAR trainers save milestone checkpoints as `epoch_XXXX` and resume
  checkpoints as `latest`.
- LOGOA examples use `epoch_2400`; set the checkpoint epoch to whichever
  milestone you actually retained.

## UnlearnCanvas DAG

Run all UnlearnCanvas commands from `UnlearnCanvas/` with `PYTHONPATH=.`. The
command snippets in `docs/examples/unlearncanvas_commands.md` show one concrete
command for each stage; `scripts/run_unlearncanvas_reproduction.sh` is a plain
shell example that chains the main stages.

Use these 16 paper-faithful target styles:

```text
Abstractionism
Artist Sketch
Blossom Season
Blue Blooming
Bricks
Byzantine
Cartoon
Cold Warm
Color Fantasy
Comic Etch
Crayon
Crypto Punks
Cubism
Dadaism
Dapple
Defoliation
```

1. Place UnlearnCanvas images under
   `UnlearnCanvas/data/UnlearnCanvas/<Style>/<Object>/<id>.jpg`.
2. Precompute CLIP style prototypes from the distributed descriptors.
3. Fine-tune the all-class SD1.5 model on the prompt metadata.
4. Train 16 LOGO models, one held-out target style at a time.
5. Generate 320 query images from the all-class model using the 16 target
   styles and 20 evaluation objects.
6. Train 16 GUDA-C AWSS models, one forget style at a time.
7. Compute LOGOA scores using all-class and LOGO checkpoints.
8. Compute GUDA-C scores using all-class and GUDA-C checkpoints.
9. Compare LOGOA and GUDA-C rankings.
10. Run anchor ablations for AWSS, uniform sampling, and style removed.

Important checkpoint naming:

- `train_allclass_prompts_full.py`, `train_logo_prompts_full.py`, and
  `train_retrack_ffsd.py` save checkpoints as `checkpoint_step_<step>.pt`.
- Evaluation scripts accept either that `.pt` form or diffusers-style
  checkpoint directories when supported. The public examples use the `.pt`
  form emitted by this repository.

## Lightweight Validation

These checks do not run training and should be safe on a login node or other
lightweight environment:

```bash
python3 - <<'PY'
from pathlib import Path
bad = []
for root in ["CIFAR10", "UnlearnCanvas"]:
    for path in Path(root).rglob("*.py"):
        try:
            compile(path.read_text(), str(path), "exec")
        except Exception as exc:
            bad.append((str(path), type(exc).__name__, str(exc)))
if bad:
    for item in bad:
        print("BAD", *item)
    raise SystemExit(1)
print("Python syntax OK")
PY
```

```bash
python3 - <<'PY'
import json
from collections import Counter
for path in [
    "UnlearnCanvas/data/train_prompts_ffsd_very_relaxed.jsonl",
    "UnlearnCanvas/data/eval_prompts_ffsd_very_relaxed.jsonl",
]:
    rows = 0
    styles = Counter()
    objects = set()
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            rows += 1
            styles[item["style"]] += 1
            objects.add(item["object"])
    print(path, rows, "rows", len(styles), "styles", len(objects), "objects")
PY
```

Expected metadata counts:

- train prompts: `24000` rows, `60` styles, `20` objects.
- eval prompts: `1200` rows, `60` styles, `20` objects.

Confirm the paper-faithful target style list is consistent:

```bash
python3 - <<'PY'
import json
paths = [
    "UnlearnCanvas/param_configs.json",
    "UnlearnCanvas/param_configs_ablation_sampling_paperfaithful.json",
    "UnlearnCanvas/param_configs_ablation_descriptor_count_paperfaithful.json",
    "UnlearnCanvas/param_configs_ablation_temperature_paperfaithful.json",
]
style_lists = []
for path in paths:
    with open(path) as f:
        style_lists.append(json.load(f)["styles"])
first = style_lists[0]
for path, styles in zip(paths, style_lists):
    if styles != first:
        raise SystemExit(f"style mismatch: {path}")
print("Paper-faithful style list OK:", ", ".join(first))
PY
```

