# CIFAR-10 Commands

Run all commands from `CIFAR10/`:

```bash
export PYTHONPATH=.
```

These are concrete single-stage examples. Full paper reproduction requires
running the class loops described below and in `../../REPRODUCTION.md`.

Train the full unconditional model:

```bash
python scripts/train_cifar10_unconditional.py \
  --dataset cifar10 \
  --data_dir data \
  --output_dir outputs/models/cifar10/full \
  --epochs 2600 \
  --batch_size 256 \
  --timesteps 4000 \
  --beta_schedule squaredcos_cap_v2
```

Train the conditional model needed by the ESD comparison:

```bash
python scripts/train_cifar10_conditional.py \
  --dataset cifar10 \
  --data_dir data \
  --output_dir outputs/models/cifar10/conditional \
  --epochs 400 \
  --batch_size 256 \
  --timesteps 4000 \
  --beta_schedule squaredcos_cap_v2
```

Train one LOGO model. Repeat with `--exclude_classes 0` through
`--exclude_classes 9` and matching `exclude_<class>` output directories:

```bash
python scripts/train_cifar10_unconditional.py \
  --dataset cifar10 \
  --data_dir data \
  --output_dir outputs/models/cifar10/logo/exclude_0 \
  --exclude_classes 0 \
  --epochs 2600 \
  --batch_size 256 \
  --timesteps 4000 \
  --beta_schedule squaredcos_cap_v2
```

Generate deterministic query images from the full model:

```bash
python tools/create_fixed_noise.py \
  --output results/cifar10/fixed_noise/query_2048.pt \
  --num_images 2048 \
  --channels 3 \
  --height 32 \
  --width 32 \
  --seed 42

python tools/sample.py \
  --model_type unconditional \
  --ckpt_dir outputs/models/cifar10/full/epoch_2400 \
  --dataset cifar10 \
  --output_dir results/cifar10/generated_queries/ddpm4000 \
  --num_images 2048 \
  --batch_size 256 \
  --num_inference_steps 4000 \
  --scheduler ddpm \
  --noise_file results/cifar10/fixed_noise/query_2048.pt \
  --seed 42
```

Train one GUDA-U ReTrack model. Repeat for classes `0` through `9`:

```bash
python scripts/train_cifar10_retrack.py \
  --dataset cifar10 \
  --data_dir data \
  --teacher_model_path outputs/models/cifar10/full/latest \
  --unlearn_class 0 \
  --output_dir outputs/models/cifar10/guda_retrack/unlearn_class_0 \
  --epochs 50 \
  --lr 1e-5 \
  --lambda_forget 0.03 \
  --k_neighbors 10 \
  --fast_retrack
```

Train one GUDA-U ESD model. Repeat for classes `0` through `9` if reproducing
the ReTrack-vs-ESD comparison:

```bash
python scripts/train_cifar10_esd.py \
  --dataset cifar10 \
  --data_dir data \
  --teacher_model_path outputs/models/cifar10/full/latest \
  --conditional_model_path outputs/models/cifar10/conditional/latest \
  --unlearn_class 0 \
  --output_dir outputs/models/cifar10/guda_esd/unlearn_class_0 \
  --epochs 50 \
  --lr 3e-5 \
  --lambda_forget 0.3
```

Compute LOGOA scores:

```bash
python evaluation/compute_delta_elbo_logo.py \
  --dataset cifar10 \
  --gen_dir results/cifar10/generated_queries/ddpm4000 \
  --ckpt_all outputs/models/cifar10/full/epoch_2400 \
  --ckpt_base outputs/models/cifar10/logo \
  --ckpt_epoch epoch_2400 \
  --output_dir results/cifar10/logoa \
  --timesteps 4000 \
  --skip 10 \
  --min_t 1
```

Compute GUDA-U attribution scores:

```bash
python evaluation/compute_delta_elbo_unlearned.py \
  --dataset cifar10 \
  --gen_dir results/cifar10/generated_queries/ddpm4000 \
  --ckpt_base outputs/models/cifar10/full/epoch_2400 \
  --ckpt_unlearned outputs/models/cifar10/guda_retrack \
  --ckpt_epoch epoch_0030 \
  --output_dir results/cifar10/guda_retrack_scores \
  --timesteps 4000 \
  --skip 10 \
  --min_t 1
```

Compare rankings:

```bash
python evaluation/plot_multi_correlation_analysis.py \
  --reference_csv results/cifar10/logoa/delta_elbo_loo_sorted.csv \
  --test_csvs results/cifar10/guda_retrack_scores/delta_elbo_unlearned_sorted.csv \
  --labels GUDA_ReTrack \
  --output_dir results/cifar10/ranking_agreement \
  --exp_name logoa_vs_guda
```
