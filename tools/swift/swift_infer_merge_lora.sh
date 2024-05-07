CUDA_VISIBLE_DEVICES=0 swift export \
  --ckpt_dir /home/image_team/image_team_docker_home/lgd/e_commerce_lmm/results/qwenvl_swift_xray/qwen-vl-chat/v1-20240505-042908/checkpoint-990/ \
  --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
  --ckpt_dir /home/image_team/image_team_docker_home/lgd/e_commerce_lmm/results/qwenvl_swift_xray/qwen-vl-chat/v1-20240505-042908/checkpoint-990-merged \
  --load_dataset_config true
