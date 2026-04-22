#CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#   --backbone segswin-base segswin-small --pretrain_batch 64 --finetune_batch 8 --mfusion LSF \
#   --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD --test_now
#
# step_adam: lr/wd 在 optim_presets.STEP_ADAM_CONFIG
# cosine_adamw: lr/wd 在 optim_presets.COSINE_ADAMW_CONFIG
# PYTORCH_ALLOC_CONF 缓解显存碎片化，避免 OOM（见 PyTorch Memory Management 文档）


export PYTORCH_ALLOC_CONF=expandable_segments:True
# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone convnextv2-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD

# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone pvtv2-b5 --optim_preset step_adam_simple \
#    --pretrain_batch 96 --finetune_batch 12 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD



# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone segswin-base --optim_preset step_adam_simple \
#    --pretrain_batch 64 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD


# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone segswin-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD --gpu_id 0 --test_now

# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone segswin-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task ISOD --gpu_id 0 --test_now

# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone hiera-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD --gpu_id 0 --test_now


# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone hiera-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task ISOD --gpu_id 0 --test_now


# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone segswin-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD --gpu_id 0 --test_now

# CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone segswin-base --optim_preset step_adam_simple \
#    --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
#    --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task ISOD --gpu_id 0 --test_now


CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
   --backbone pvtv2_b5 --optim_preset step_adam_simple \
   --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
   --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task COD --gpu_id 0 --test_now

CUDA_VISIBLE_DEVICES=1,2,3,4  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
   --backbone pvtv2_b5 --optim_preset step_adam_simple \
   --pretrain_batch 60 --finetune_batch 8 --mfusion LSF \
   --log_path ./log/ --pretrain_size 384 --finetune_size 576 --task ISOD --gpu_id 0 --test_now
