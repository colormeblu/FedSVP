# FedSVP
基于对比对齐的联邦尺度感知视觉提示学习
FedSVP(Federated Scale-aware Visual Prompting with Contrastive Alignment) is a runnable Python project for the algorithm described in `毕业论文算法一.docx`:

- **Fed-MSVP (Federated Multi-Scale Visual Prompting)**: multi-scale visual prompts across ViT layers with adaptive gates (`alpha_l`, initialized to 0).
- **Fed-GCA (Global-Local Contrastive Alignment)**: global-local contrastive alignment with server-maintained class prototypes.
- **Fed-SDWA (Semantic-Distance Weighted Aggregation)**: semantic-distance weighted aggregation based on proxy-data semantic scores.

The project keeps the same scaffold style as `fedcausal_prompt_project`:
`train.py + configs + src/<package> + tests`.

## 1) Install

Use **Python 3.9** (required by `torch==1.9` in this project).

```bash
cd /home/dengyu/FedSVP
pip install -r requirements.txt
pip install -e .
```

## 2) Quick Start

List datasets and algorithms:

```bash
python train.py --list
```

Single run (PACS, target=sketch):

```bash
python train.py --config configs/pacs_fedsvp.json \
  --set dataset.root=/path/to/data
```

Run LODO on all targets in one command:

```bash
python train.py --config configs/pacs_fedsvp.json \
  --set dataset.root=/path/to/data \
  --set dataset.target_domain=ALL
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=2 train.py --config configs/domainnet_fedsvp.json \
  --set dataset.root=/path/to/data \
  --set parallel_mode=ddp \
  --set multi_gpu=true \
  --set gpu_ids=[0,1]
```

Notes:
- DDP requires `torchrun`; do not use plain `python train.py` for multi-GPU.
- `multi_gpu=true` with empty `gpu_ids` uses all visible CUDA devices.
- You can set explicit devices, e.g. `gpu_ids=[1,2,3]`, and then use `--nproc_per_node=3`.
- Keep `device` unset (or set it to `cuda`) unless you need a specific single-GPU run.

## 3) Core Mapping to the Thesis

### Fed-MSVP (Module 1)

- `src/fedsvp/models/fedsvp_clip.py`
- `MultiScaleVisualPrompt`
  - Shallow prompts: layers `0-3`
  - Middle prompts: layers `4-8`
  - Deep prompts: layers `9-11`
- `alpha_l` gate per layer, initialized to `0`
- local loss in `src/fedsvp/algorithms/fedsvp_train.py`:
  - `L_local = CE + lambda_consistency * (1 - cosine(f_prompt, f_frozen)) + lambda_contrastive * L_con`

### Fed-GCA (Module 2)

- client uploads per-class local prototypes after local training
- server aggregates global prototypes with class-count weighting
- next communication round uses InfoNCE-style loss:
  - `L_con = CE( sim(z, G) / contrastive_tau )`
- implementation files:
  - `src/fedsvp/algorithms/fedsvp_train.py`
  - `src/fedsvp/algorithms/agg.py`
  - `src/fedsvp/algorithms/runners.py`

### Fed-SDWA (Module 3)

- semantic anchor construction from class names and templates
- proxy-data scoring per client:
  - `S_k = mean_i max_c cosine(v_k,i, z_anchor,c)`
- semantic softmax weights:
  - `w_k = softmax(S_k / tau)`
- weighted prompt aggregation on server

Implementation files:

- `src/fedsvp/algorithms/runners.py`
- `src/fedsvp/algorithms/agg.py`

## 4) Configs

FedSVP configs:

- `configs/pacs_fedsvp.json`
- `configs/office_fedsvp.json`
- `configs/domainnet_fedsvp.json`

Baseline configs:

- `configs/pacs_fedavg.json`
- `configs/office_fedavg.json`
- `configs/domainnet_fedavg.json`

Ablation grid (matching the thesis table logic):

- `configs/grid_pacs_loo_ablation_fedsvp.json`
  - `standard_prompt_fedavg`
  - `msvp_only`
  - `msvp_gca`
  - `sdwa_only`
  - `fedsvp_full`

Run grid:

```bash
python train.py --grid configs/grid_pacs_loo_ablation_fedsvp.json \
  --set dataset.root=/path/to/data
```

## 5) Dataset Layout

### PACS

```text
<DATA_ROOT>/PACS/
  photo/<class>/*.jpg
  art/<class>/*.jpg
  cartoon/<class>/*.jpg
  sketch/<class>/*.jpg
```

### Office-Home

```text
<DATA_ROOT>/OfficeHome/
  Art/<class>/*.jpg
  Clipart/<class>/*.jpg
  Product/<class>/*.jpg
  RealWorld/<class>/*.jpg
```

### DomainNet

```text
<DATA_ROOT>/DomainNet/
  clipart/<class>/*
  infograph/<class>/*
  painting/<class>/*
  quickdraw/<class>/*
  real/<class>/*
  sketch/<class>/*
```

## 6) Project Layout

```text
FedSVP/
├── configs/
├── src/fedsvp/
│   ├── algorithms/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── registry.py
├── tests/
└── train.py
```
