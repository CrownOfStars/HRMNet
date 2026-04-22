#!/bin/bash

# 固定参数
GPUS="0,1,2,3"
NPROC=4
PYTHON_CMD="python -m torch.distributed.launch --nproc_per_node=${NPROC} distributed.py"
COMMON_ARGS="--train_batch 32 --log_path ./log/ --gamma 0.5 --img_size 576 --task COD --test_now"

# 超参数搜索空间
freeze_lrs=(1e-4 3e-4 5e-4)
freeze_decays=(3 5 10)
full_lrs=(1e-4 3e-4 5e-4)
full_decays=(5 10 20)

# 冻结/全量阶段总周期
freeze_epochs=40
full_epochs=80

# 启动网格搜索
run_id=0
for lr_freeze in "${freeze_lrs[@]}"; do
  for decay_freeze in "${freeze_decays[@]}"; do
    for lr_full in "${full_lrs[@]}"; do
      for decay_full in "${full_decays[@]}"; do
        ((run_id++))
        echo "Run $run_id: Freeze LR=$lr_freeze, Freeze Decay=$decay_freeze | Full LR=$lr_full, Full Decay=$decay_full"

        CUDA_VISIBLE_DEVICES=$GPUS $PYTHON_CMD \
          --lr $lr_freeze \
          --decay_epoch $decay_freeze \
          --epochs $freeze_epochs \
          --stage freeze \
          $COMMON_ARGS \
          --run_id ${run_id}_freeze

        CUDA_VISIBLE_DEVICES=$GPUS $PYTHON_CMD \
          --lr $lr_full \
          --decay_epoch $decay_full \
          --epochs $full_epochs \
          --stage full \
          $COMMON_ARGS \
          --resume \
          --run_id ${run_id}_full

        echo "===================="
      done
    done
  done
done
