# On-Policy Distillation

## Installation

```bash
conda create -n verl-vllm python=3.12 -y
conda activate verl-vllm

pip install vllm==0.8.5
pip install -e ".[vllm]"
pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir

pip install polars
```

## Prepare Datasets

```bash
bash examples/opd_trainer/prepare_dapo_data.sh
```

## Run OPD Trainer

distil Qwen3-4B from DAPO-Qwen-32B
```bash
bash examples/opd_trainer/run_opd.sh
```