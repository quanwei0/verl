export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
export WANDB_ENTITY="rl_agent"

function now() {
    date '+%Y-%m-%d-%H-%M'
}

mkdir -p logs


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# nohup bash examples/sglang_multiturn/search_r1_like/run_grpo_qwen2.5-3b_instruct_search_multiturn.sh \
#     trainer.experiment_name=grpo_qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) \
#     > logs/searchR1-like$(now).log 2>&1 &


bash examples/sglang_multiturn/search_r1_like/run_grpo_qwen2.5-3b_instruct_search_multiturn.sh trainer.experiment_name=grpo_qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now)
