## Installation

```bash
git clone https://github.com/quanwei0/verl

cd verl

git checkout mrppo


conda create -n verl-vllm python=3.12 -y
conda activate verl-vllm
pip install -e ".[vllm]"

```

## Data Processing

```bash
conda activate verl-vllm
python examples/mrppo_trainer/math_reasoning_datasets.py
```

## Run

```bash
bash examples/mrppo_trainer/run_ppo_1.5b.sh 2>&1 | tee train.log
```
