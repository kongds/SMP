#!/usr/bin/env bash

EXP=$1
LR=2e-2
NAME=$EXP
echo $EXP
TEACHER_PATH=./squad_teacher_model
BATCH_SIZE=32
GRAD_ACC=1
EPOCH=10


args=() # flags for training

GPU=$CUDA_VISIBLE_DEVICES
if [ -z "$GPU" ]; then
  GPU=0
fi

REPORT_TO=wandb

if [[ $EXP == *"_0.03"* ]]; then
  SPARSE=0.03
  EPOCH=20
  args+=(--warmup_threshold_step 22650) # same threshold step with 0.10 sparsity
  args+=(--warmup_regu_step 27660) # 10 epoch regu warmup
elif [[ $EXP == *"_0.10"* ]]; then
  SPARSE=0.10
elif [[ $EXP == *"_0.30"* ]]; then
  SPARSE=0.10
elif [[ $EXP == *"_0.40"* ]]; then
  SPARSE=0.40
elif [[ $EXP == *"_0.50"* ]]; then
  SPARSE=0.50

  if [[ $EXP == *"_mag"* ]] && [[ $EXP != *"_kd"* ]]; then
    # settings for squad_0.50_mag
    args+=(--magnitude_topk_threshold_p 4 --warmup_threshold_step 5000)
  fi
elif [[ $EXP == *"_0.70"* ]]; then
  SPARSE=0.70
  args+=(--warmup_threshold_step 5000)
elif [[ $EXP == *"_0.80"* ]]; then
  SPARSE=0.80
  args+=(--warmup_threshold_step 5000)
fi

if [[ $EXP == *"_kd"* ]]; then
  # use gradient accumulation to avoid OOM for 24GB GPU
  #BATCH_SIZE=16
  #GRAD_ACC=2
  args+=(--teacher_model_name_or_path ${TEACHER_PATH})
fi

if [[ $EXP == *"_b16"* ]]; then
  BATCH_SIZE=16
  GRAD_ACC=2
fi

if [[ $EXP == *"_mag"* ]]; then
    if [ `echo "$SPARSE < 0.7"|bc` -eq 1 ]; then
        args+=(--magnitude_topk_threshold)
    else
        args+=(--magnitude_topk_threshold --magnitude_topk_threshold_p 4)
    fi
fi


export WANDB_PROJECT=MP_SQUAD
TRANSFORMERS_OFFLINE=0  HF_DATASETS_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU \
python run_squad.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_eval \
  --do_train \
  --overwrite_cache True \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACC \
  --num_train_epochs $EPOCH \
  --report_to $REPORT_TO\
  --learning_rate 3e-5 \
  --mask_scores_learning_rate 2e-2 \
  --initial_warmup 0 \
  --final_warmup 1 \
  --initial_threshold 1 \
  --final_threshold ${SPARSE} \
  --pruning_method topK \
  --mask_init constant \
  --mask_scale 0. \
  --regularization l1 \
  --final_lambda 400. \
  --output_dir models/${NAME} \
  --run_name $NAME \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_strategy steps \
  --save_steps 1000 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --per_device_eval_batch_size 32 \
  --warmup_ratio 0.00 \
  --logging_steps 50 \
  --save_total_limit 1 \
  --freeze_bert \
  --use_cls_in_prompt \
  --label_map "{'1':'No','0':'Yes'}" \
  ${args[@]}
