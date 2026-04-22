#!/bin/bash

# 检查是否传入了足够的参数
if [ -z "$1" ]; then
  echo "Error: No task parameter provided."
  echo "Usage: $0 <task> [img_size]"
  exit 1
fi

# 获取传入的任务参数
TASK=$1

# 获取传入的 img 参数_size，如果没有提供，则使用默认值 384
IMG_SIZE=${2:-384}

# 运行原始命令，并将任务参数和 img_size 参数传递进去
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
   --backbone segswin-base segswin-small --lr1 3e-4 --lr2 3e-5 --train_batch 32 --mfusion LSF \
   --log_path ./log/ --decay_epoch1 5 --decay_epoch2 10 --gamma 0.5 --img_size $IMG_SIZE --task $TASK --test_now