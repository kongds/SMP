#!/usr/bin/env bash
USE_MASK_PT=$1
LR=2e-2
FL=400.

# if USE_MASK_PT not end with .npy and USE_MASK_PT.npy not exist then download
if [[ "${USE_MASK_PT}" == *_kd_mag ]]; then
    if [[ ! -f "${USE_MASK_PT}.npy" ]]; then
        echo "Trying to download ${USE_MASK_PT}.npy"
        wget "https://github.com/kongds/SMP/releases/download/SMP-S/${USE_MASK_PT}.npy.zip"
        unzip "${USE_MASK_PT}.npy.zip"
    fi
    USE_MASK_PT="${USE_MASK_PT}.npy"
fi

echo $USE_MASK_PT

args=() # flags for training

GPU=$CUDA_VISIBLE_DEVICES
if [ -z "$GPU" ]; then
  GPU=0
fi


BERT=bert-base-uncased
if [[ $USE_MASK_PT == "mnli"* ]]; then
    TASK=mnli
    LABEL_MAP="{'2':'No','0':'Yes','1':'Maybe'}"
elif [[ $USE_MASK_PT == "qnli"* ]]; then
    TASK=qnli
    LABEL_MAP="{'1':'No','0':'Yes'}"
elif [[ $USE_MASK_PT == "qqp"* ]]; then
    TASK=qqp
    LABEL_MAP="{'0':'No','1':'Yes'}"
elif [[ $USE_MASK_PT == "sst2"* ]]; then
    LABEL_MAP="{'0':'terrible','1':'great'}"
elif [[ $USE_MASK_PT == "mrpc"* ]]; then
    LABEL_MAP="{'0':'No','1':'Yes'}"
elif [[ $USE_MASK_PT == "cola"* ]]; then
    LABEL_MAP="{'0':'incorrect','1':'correct'}"
elif [[ $USE_MASK_PT == "stsb"* ]]; then
    TASK=stsb
    LABEL_MAP="{'0':'Yes'}"
elif [[ $USE_MASK_PT == "rte"* ]]; then
    TASK=rte
    LABEL_MAP="{'1':'No','0':'Yes'}"
elif [[ $USE_MASK_PT == "wnli"* ]]; then
    TASK=wnli
    LABEL_MAP="{'1':'No','0':'Yes'}"
elif [[ $USE_MASK_PT == "squad"* ]]; then
    export WANDB_PROJECT=MP_SQUAD
    TRANSFORMERS_OFFLINE=0  HF_DATASETS_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU \
    python run_squad.py \
     --model_name_or_path $BERT \
     --dataset_name squad \
     --do_eval \
     --max_seq_length 384 \
     --doc_stride 128 \
     --report_to none \
     --initial_warmup 0 \
     --overwrite_cache True \
     --final_warmup 1 \
     --initial_threshold 1 \
     --pruning_method topK \
     --mask_init constant \
     --mask_scale 0. \
     --regularization l1 \
     --final_lambda 400. \
     --output_dir eval/${USE_MASK_PT/.npy/} \
     --per_device_eval_batch_size 128 \
     --warmup_ratio 0.00 \
     --logging_steps 50 \
     --save_total_limit 1 \
     --freeze_bert \
     --use_cls_in_prompt \
     --label_map "{'1':'No','0':'Yes'}" \
    --use_mask_pt ${USE_MASK_PT}
    exit
fi


TRANSFORMERS_OFFLINE=0  HF_DATASETS_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU  \
python run_glue.py \
 --model_name_or_path ${BERT} \
 --task_name ${TASK} \
 --do_eval \
 --max_seq_length 128 \
 --report_to none \
 --learning_rate 0 \
 --initial_warmup 0 \
 --final_warmup 6 \
 --initial_threshold 1 \
 --overwrite_cache True \
 --pruning_method topK \
 --mask_init constant \
 --mask_scale 0. \
 --regularization l1 \
 --per_device_eval_batch_size 128 \
 --warmup_ratio 0.00 \
 --logging_steps 50 \
 --save_total_limit 1 --freeze_bert \
 --label_map ${LABEL_MAP} \
 --output_dir eval/${USE_MASK_PT/.npy/} \
 --use_mask_pt ${USE_MASK_PT}

