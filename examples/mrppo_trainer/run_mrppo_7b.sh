set -x


export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
export WANDB_ENTITY="rl_agent"

PROJECT_NAME=dapo-math-new
EXPERIMENT_NAME="r1-7b-mrppo"

# export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=8
BASE_DIR=$HOME/experiments/$PROJECT_NAME/$EXPERIMENT_NAME

train_files="['$HOME/data/math_reasoning/dapo_math.parquet']"
test_files="['$HOME/data/math_reasoning/aime24.parquet','$HOME/data/math_reasoning/aime25.parquet','$HOME/data/math_reasoning/aime26.parquet','$HOME/data/math_reasoning/amc.parquet']"

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MAX_RESPONSE_LENGTH=8192

ROLLOUT_IS="token"
ROLLOUT_IS_THRESHOLD=2.0

N_VALUE_HEADS=3
MRPPO_REWARD_KEYS="[answer_reward,int_reward,format_reward]"

SAVE_DATA=false

SAVE_ARGS=()
if [[ "$SAVE_DATA" == true ]]; then
    SAVE_ARGS+=(trainer.default_local_dir=$BASE_DIR/checkpoints)
    SAVE_ARGS+=(trainer.rollout_data_dir=$BASE_DIR/rollout)
    SAVE_ARGS+=(trainer.validation_data_dir=$BASE_DIR/validation)
else
    SAVE_ARGS+=(trainer.default_local_dir=null)
    SAVE_ARGS+=(trainer.rollout_data_dir=null)
    SAVE_ARGS+=(trainer.validation_data_dir=null)
fi


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=mrppo \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS} \
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=$(nproc) \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    critic.enable=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    +critic.model.override_config.n_value_heads=$N_VALUE_HEADS \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    "${SAVE_ARGS[@]}" \
    reward_model.reward_manager=mrppo \
    +reward_model.reward_kwargs.use_answer_reward_only=False \
    "+algorithm.mrppo_reward_keys=$MRPPO_REWARD_KEYS" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]" \
    "critic.checkpoint.save_contents=[model,hf_model,optimizer,extra]" \
    trainer.total_epochs=1 $@
