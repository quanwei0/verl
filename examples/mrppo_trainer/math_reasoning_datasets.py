"""
Preprocess math evaluation datasets to parquet format for verl training/evaluation.

Supported datasets:
  - AIME 2024  (HuggingFaceH4/aime_2024)
  - AIME 2025  (yentinglin/aime_2025)
  - AIME 2026  (MathArena/aime_2026)
  - AMC        (math-ai/amc23)
  - DAPO Math  (BytedTsinghua-SIA/DAPO-Math-17k)  [training]
"""

import argparse
import os

from datasets import load_dataset

SYSTEM_PROMPT = """You are a helpful AI assistant.

For every request, you should carefully think through the math problem step by step, then provide the fianl answer in integer format.

Steps for Each Request:
1. Think: Provide detailed, step-by-step reasoning, calculations, or derivations.
2. Produce Final Answer: After step-by-step reasoning, output the final answer in integer format.

Output Format:
<think>Your thoughts and reasoning</think>
<answer>Final answer in integer format</answer>

Important Notes:
1. You must include your reasoning steps inside <think>.
2. You must always output the Final Answer within <answer> after the reasoning steps is done.
3. You should consistently work through the solution step by step before giving the final answer.
4. The final answer can only be an integer.
"""


def make_record(idx, *, data_source, question, prompt, ground_truth, split):
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": idx, "question": question},
    }


# ---------------------------------------------------------------------------
# AIME 2024
# ---------------------------------------------------------------------------

def build_aime2024(split="test"):
    data_source = "HuggingFaceH4/aime_2024"
    print(f"Loading {data_source} ...", flush=True)
    ds = load_dataset(data_source, split="train")

    def _map(example, idx):
        question = example["problem"]
        ground_truth = str(example["answer"])
        prompt = question.strip()
        return make_record(idx,
                           data_source="aime24",
                           question=question,
                           prompt=prompt,
                           ground_truth=ground_truth,
                           split=split)

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# AIME 2025
# ---------------------------------------------------------------------------

def build_aime2025(split="test"):
    data_source = "yentinglin/aime_2025"
    print(f"Loading {data_source} ...", flush=True)
    ds = load_dataset(data_source, split="train")

    def _map(example, idx):
        question = example["problem"]
        ground_truth = str(example["solution"])
        prompt = question.strip()
        return make_record(idx,
                           data_source="aime25",
                           question=question,
                           prompt=prompt,
                           ground_truth=ground_truth,
                           split=split)

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# AIME 2026
# ---------------------------------------------------------------------------

def build_aime2026(split="test"):
    data_source = "MathArena/aime_2026"
    print(f"Loading {data_source} ...", flush=True)
    ds = load_dataset(data_source, split="train")

    def _map(example, idx):
        question = example["problem"]
        ground_truth = str(example["answer"])
        prompt = question.strip()
        return make_record(idx,
                           data_source="aime26",
                           question=question,
                           prompt=prompt,
                           ground_truth=ground_truth,
                           split=split)

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# AMC  (math-ai/amc23)
# ---------------------------------------------------------------------------

def build_amc(split="test"):
    data_source = "math-ai/amc23"
    print(f"Loading {data_source} ...", flush=True)
    ds = load_dataset(data_source, split="test")

    def _map(example, idx):
        question = example["question"]
        ground_truth = str(example["answer"])
        prompt = question.strip()
        return make_record(idx,
                           data_source="amc",
                           question=question,
                           prompt=prompt,
                           ground_truth=ground_truth,
                           split=split)

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# DAPO Math  (BytedTsinghua-SIA/DAPO-Math-17k)  — training set
# ---------------------------------------------------------------------------

def build_dapo_math(split="train"):
    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    print(f"Loading {data_source} ...", flush=True)
    ds = load_dataset(data_source, split="train")

    def _map(example, idx):
        question = example["prompt"][0]["content"]
        ground_truth = str(example["reward_model"]["ground_truth"])
        prompt = question.strip()
        return make_record(idx,
                           data_source="dapo_math",
                           question=question,
                           prompt=prompt,
                           ground_truth=ground_truth,
                           split=split)

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASET_BUILDERS = {
    "aime24": build_aime2024,
    "aime25": build_aime2025,
    "aime26": build_aime2026,
    "amc": build_amc,
    "dapo_math": build_dapo_math,
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess math eval datasets to parquet.")
    parser.add_argument("--save_dir", default="~/data/math_reasoning",
                        help="Root directory to save parquet files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_BUILDERS.keys()),
        choices=list(DATASET_BUILDERS.keys()),
        help="Which datasets to process (default: all).",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for name in args.datasets:
        ds = DATASET_BUILDERS[name]()
        out = os.path.join(save_dir, f"{name}.parquet")
        ds.to_parquet(out)
        print(f"  Saved {len(ds)} examples -> {out}")


if __name__ == "__main__":
    main()
