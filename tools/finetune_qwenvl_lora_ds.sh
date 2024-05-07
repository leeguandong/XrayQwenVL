#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=$(pwd)

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/weights/qwen-vl-caht/" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL"  Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/data/openai-zh-qwenvl-prompt.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --bf16 False \
  --fp16 True \
  --fix_vit True \
  --output_dir output_qwen \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 1024 \
  --lazy_preprocess True \
  --use_lora \
  --gradient_checkpointing \
  --deepspeed finetune/ds_config_zero2.json
