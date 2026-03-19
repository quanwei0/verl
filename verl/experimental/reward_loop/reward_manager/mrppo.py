# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def answer_reward(response_str: str, ground_truth: str) -> float:
    extracted = extract_xml_answer(response_str)
    return 1.0 if extracted == str(ground_truth) else 0.0


def int_reward(response_str: str) -> float:
    extracted = extract_xml_answer(response_str)
    return 1.0 if extracted.isdigit() else 0.0


def format_reward(response_str: str) -> float:
    pattern = r"^(?:<think>)?.*?</think>\n+<answer>.*?</answer>$"
    if (
        re.search(pattern, response_str, re.DOTALL)
        and response_str.count("<answer>") == 1
        and response_str.count("</answer>") == 1
    ):
        return 1.0
    return 0.0


@register("mrppo")
class MRPPORewardManager(RewardManagerBase):
    """MRPPO Reward Manager.

    Computes three separate reward components (answer_reward, int_reward,
    format_reward) to support multi-reward PPO advantage estimation via
    algorithm.mrppo_reward_keys.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.use_answer_reward_only = config.reward.get("reward_kwargs", {}).get("use_answer_reward_only", False)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        r_answer = answer_reward(response_str, ground_truth)
        r_int = int_reward(response_str)
        r_format = format_reward(response_str)

        if self.use_answer_reward_only:
            reward = r_answer
        else:
            reward = r_answer + r_int + r_format

        reward_extra_info = {
            "answer_reward": r_answer,
            "int_reward": r_int,
            "format_reward": r_format,
            "total_reward": reward,
        }

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
