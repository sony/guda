# UnlearnCanvas Commands

Run all commands from `UnlearnCanvas/`:

```bash
export PYTHONPATH=.
```

These are concrete single-stage examples. Full paper reproduction requires
running the 16-style loops described below and in `../../REPRODUCTION.md`.

Preferred local image layout:

```text
data/UnlearnCanvas/<Style>/<Object>/<id>.jpg
```

The prompt files used in the paper are distributed:

```text
data/train_prompts_ffsd_very_relaxed.jsonl
data/eval_prompts_ffsd_very_relaxed.jsonl
data/ffsd_very_relaxed_descriptors.json
```

Precompute CLIP style prototypes for AWSS:

```bash
python evaluation/precompute_clip_prototypes.py \
  --descriptors_file data/ffsd_very_relaxed_descriptors.json \
  --output_file cache/clip_style_prototypes_very_relaxed.pt \
  --clip_model openai/clip-vit-large-patch14 \
  --device cuda
```

Train the SD 1.5 all-style model:

```bash
python machine_unlearning/retrack_latent/src/train_allclass_prompts_full.py \
  --model_id runwayml/stable-diffusion-v1-5 \
  --data_root data/UnlearnCanvas \
  --prompt_file data/train_prompts_ffsd_very_relaxed.jsonl \
  --output_dir outputs/allclass_sd15_ffsd_very_relaxed \
  --num_steps 10000 \
  --lr 1e-6
```

Train one LOGO model. Repeat for the 16 paper-faithful target styles listed in
`../../REPRODUCTION.md`:

```bash
python machine_unlearning/retrack_latent/src/train_logo_prompts_full.py \
  --model_id runwayml/stable-diffusion-v1-5 \
  --data_root data/UnlearnCanvas \
  --prompt_file data/train_prompts_ffsd_very_relaxed.jsonl \
  --exclude_style "Abstractionism" \
  --output_dir "outputs/logo_sd15_ffsd_very_relaxed/Abstractionism" \
  --num_steps 10000 \
  --lr 1e-6
```

Generate the 320 query images for the 16 target styles and 20 evaluation
objects:

```bash
python evaluation/generate_ffsd_evaluation_images.py \
  --allclass_dir outputs/allclass_sd15_ffsd_very_relaxed \
  --checkpoint_step 10000 \
  --eval_prompts_file data/eval_prompts_ffsd_very_relaxed.jsonl \
  --output_dir outputs/evaluation_images_ffsd_very_relaxed \
  --styles \
    "Abstractionism" "Artist Sketch" "Blossom Season" "Blue Blooming" \
    "Bricks" "Byzantine" "Cartoon" "Cold Warm" \
    "Color Fantasy" "Comic Etch" "Crayon" "Crypto Punks" \
    "Cubism" "Dadaism" "Dapple" "Defoliation" \
  --seed 42
```

Train one GUDA-C AWSS model. Repeat for the 16 target styles:

```bash
python machine_unlearning/retrack_latent/src/train_retrack_ffsd.py \
  --allclass_checkpoint outputs/allclass_sd15_ffsd_very_relaxed/checkpoint_step_10000.pt \
  --data_root data/UnlearnCanvas \
  --prompts_file data/train_prompts_ffsd_very_relaxed.jsonl \
  --forget_style "Abstractionism" \
  --anchor_strategy weighted_style_select \
  --clip_prototypes_file cache/clip_style_prototypes_very_relaxed.pt \
  --descriptors_file data/ffsd_very_relaxed_descriptors.json \
  --beta 2.0 \
  --eta_uniform 0.3 \
  --style_sampling_mode weighted \
  --num_anchor_descriptors 3 \
  --output_dir "outputs/guda_awss/Abstractionism" \
  --num_steps 5000 \
  --learning_rate 2e-6
```

Compute LOGOA scores:

```bash
python evaluation/compute_logoa_ffsd.py \
  --allclass_dir outputs/allclass_sd15_ffsd_very_relaxed \
  --checkpoint_step 10000 \
  --logo_dir outputs/logo_sd15_ffsd_very_relaxed \
  --image_dir outputs/evaluation_images_ffsd_very_relaxed \
  --eval_prompts_file data/eval_prompts_ffsd_very_relaxed.jsonl \
  --output_dir outputs/logoa_scores_ffsd_very_relaxed
```

Compute GUDA-C scores:

```bash
python evaluation/compute_una_ffsd.py \
  --allclass_dir outputs/allclass_sd15_ffsd_very_relaxed \
  --checkpoint_step 10000 \
  --retrack_dir outputs/guda_awss \
  --retrack_checkpoint_step 2000 \
  --image_dir outputs/evaluation_images_ffsd_very_relaxed \
  --eval_prompts_file data/eval_prompts_ffsd_very_relaxed.jsonl \
  --output_dir outputs/guda_awss_scores
```

Compare rankings:

```bash
python evaluation/analyze_una_retrack_correlation.py \
  --logoa_dir outputs/logoa_scores_ffsd_very_relaxed \
  --una_dir outputs/guda_awss_scores \
  --output_dir outputs/ranking_agreement_awss
```
