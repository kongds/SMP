#!/usr/bin/env bash

EXP=$1
LR=2e-2
FL=400.
TEACHER_PATH=./dynabert
NAME=$EXP
echo $EXP

args=() # flags for training

GPU=$CUDA_VISIBLE_DEVICES
if [ -z "$GPU" ]; then
  GPU=0
fi

REPORT_TO=wandb


if [[ $EXP == "mnli"* ]]; then
    EPOCH=12
    BATCH_SIZE=64
    TASK=mnli
    METRIC=accuracy
    EVAL_STRATEGY=steps

    LABEL_MAP="{'2':'No','0':'Yes','1':'Maybe'}"
    if [[ $EXP == *"_0.50"* ]] && [[ $EXP != *"_kd"* ]]; then
        args+=(--magnitude_topk_threshold_p 3)
    fi

    export WANDB_PROJECT=SMP_MNLI
elif [[ $EXP == "qnli"* ]]; then
    EPOCH=12
    BATCH_SIZE=64
    TASK=qnli
    METRIC=accuracy
    EVAL_STRATEGY=steps

    LABEL_MAP="{'1':'No','0':'Yes'}"

    export WANDB_PROJECT=SMP_QNLI
elif [[ $EXP == "qqp"* ]]; then
    EPOCH=12
    BATCH_SIZE=64
    TASK=qqp
    METRIC=accuracy
    EVAL_STRATEGY=steps

    LABEL_MAP="{'0':'No','1':'Yes'}"

    export WANDB_PROJECT=SMP_QQP
elif [[ $EXP == "sst2"* ]]; then
    EPOCH=12
    BATCH_SIZE=64
    TASK=sst2
    METRIC=accuracy
    EVAL_STRATEGY=steps

    LABEL_MAP="{'0':'terrible','1':'great'}"

    export WANDB_PROJECT=SMP_SST2
elif [[ $EXP == "mrpc"* ]]; then
    EPOCH=60
    BATCH_SIZE=32
    TASK=mrpc
    METRIC=accuracy
    EVAL_STRATEGY=epoch

    LABEL_MAP="{'0':'No','1':'Yes'}"

    export WANDB_PROJECT=SMP_MRPC
elif [[ $EXP == "cola"* ]]; then
    EPOCH=60
    BATCH_SIZE=32
    TASK=cola
    METRIC=matthews_correlation
    EVAL_STRATEGY=epoch

    LABEL_MAP="{'0':'incorrect','1':'correct'}"

    export WANDB_PROJECT=SMP_COLA
elif [[ $EXP == "stsb"* ]]; then
    EPOCH=60
    BATCH_SIZE=32
    TASK=stsb
    METRIC=pearson
    EVAL_STRATEGY=epoch

    LABEL_MAP="{'0':'Yes'}"
    export WANDB_PROJECT=SMP_STSB
elif [[ $EXP == "rte"* ]]; then
    EPOCH=100
    BATCH_SIZE=32
    TASK=rte
    METRIC=accuracy
    EVAL_STRATEGY=epoch

    LABEL_MAP="{'1':'No','0':'Yes'}"
    export WANDB_PROJECT=SMP_RTE
fi

if [[ $EXP == *"_roberta_"* ]]; then
    BERT=roberta-base
    EXP=`echo $EXP | sed 's/roberta_//'`
else
    BERT=bert-base-uncased
fi


if [[ $EXP == *"_kd"* ]]; then
    args+=(--teacher_model_name_or_path  ${TEACHER_PATH}/${TASK^^})
fi


if [[ $EXP == *"_0.03"* ]]; then
    SPARSE=0.03
elif [[ $EXP == *"_0.10"* ]]; then
    SPARSE=0.10
elif [[ $EXP == *"_0.30"* ]]; then
    SPARSE=0.10
elif [[ $EXP == *"_0.40"* ]]; then
    SPARSE=0.40
elif [[ $EXP == *"_0.50"* ]]; then
    SPARSE=0.50
elif [[ $EXP == *"_0.70"* ]]; then
    # use smaller warmup threshold steps for 0.7 and 0.8
    SPARSE=0.70
    args+=(--warmup_threshold_step 5250)
elif [[ $EXP == *"_0.80"* ]]; then
    SPARSE=0.80
    args+=(--warmup_threshold_step 3500)
fi

if [[ $EXP == *"_mag"* ]]; then
    # SMP-S
    if [ `echo "$SPARSE < 0.7"|bc` -eq 1 ]; then
        args+=(--magnitude_topk_threshold)
    else
        args+=(--magnitude_topk_threshold --magnitude_topk_threshold_p 4)
    fi
fi


TRANSFORMERS_OFFLINE=0  HF_DATASETS_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU  \
python run_glue.py \
 --model_name_or_path ${BERT} \
 --task_name ${TASK} \
 --do_eval \
 --do_train \
 --max_seq_length 128 \
 --per_device_train_batch_size ${BATCH_SIZE} \
 --num_train_epochs ${EPOCH} \
 --report_to $REPORT_TO\
 --overwrite_cache True \
 --learning_rate 0 \
 --mask_scores_learning_rate $LR \
 --initial_warmup 0 \
 --final_warmup 6 \
 --initial_threshold 1 \
 --final_threshold ${SPARSE} \
 --pruning_method topK \
 --mask_init constant \
 --mask_scale 0. \
 --regularization l1 \
 --final_lambda ${FL} \
 --output_dir  ./models/${EXP} \
 --run_name $NAME \
 --evaluation_strategy ${EVAL_STRATEGY} \
 --eval_steps 1000 \
 --save_strategy ${EVAL_STRATEGY}  \
 --save_steps 1000 \
 --metric_for_best_model ${METRIC} \
 --load_best_model_at_end \
 --per_device_eval_batch_size 128 \
 --warmup_ratio 0.00 \
 --logging_steps 50 \
 --save_total_limit 1 --freeze_bert \
 --label_map ${LABEL_MAP} \
 ${args[@]}
