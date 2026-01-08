# On-Policy Distillation

## Installation

```bash
conda create -n verl-vllm python=3.12 -y

pip install vllm==0.11.0
pip install -e ".[vllm]"
pip install flash-attn --no-build-isolation
```

## Run OPD Trainer

distil qwen3 1.7b from qwen3 32b
```bash
bash examples/opd_trainer/run_qwen3-1.7b_opd.sh
```