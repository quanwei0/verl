# On-Policy Distillation

## Installation

```bash
conda create -n verl-vllm python=3.12 -y

pip install vllm==0.11.0
pip install -e ".[vllm]"
pip install flash-attn --no-build-isolation

pip install polars
```

## Prepare Datasets

```bash
bash examples/opd_trainer/prepare_dapo_data.sh
```

## Run OPD Trainer

distil Qwen3 4B from DAPO-Qwen-32B
```bash
bash examples/opd_trainer/run_opd.sh
```