from typing import Optional, Dict
from dataclasses import dataclass, field

import os
import numpy as np

import torch
from torch import nn

from transformers import TrainingArguments, AdamW
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import PredictionOutput

from emmental.modules import TopKBinarizer


def regularization(model: nn.Module, mode: str):
    regu, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                regu += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                regu += torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum() / param.numel()
            else:
                ValueError("Don't know this mode.")
            counter += 1
    return regu / counter

def schedule_threshold(
    step: int,
    total_step: int,
    warmup_steps: int,
    initial_threshold: float,
    final_threshold: float,
    initial_warmup: int,
    final_warmup: int,
    final_lambda: float,
    linear: bool = False,
):
    assert total_step - final_warmup * warmup_steps > 0
    if step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif step > (total_step - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        if linear:
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 1)
        else:
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda

@dataclass
class SMPTrainingArguments(TrainingArguments):
    mask_scores_learning_rate: float = field(
        default=1e-2,
    )
    initial_threshold: float = field(
        default=1.0,
    )
    final_threshold: float = field(
        default=0.7,
    )
    initial_warmup: int = field(
        default=1,
    )
    final_warmup: int = field(
        default=2,
    )

    pruning_method: str = field(
        default='topK',
    )
    mask_init: str = field(
        default='constant',
    )
    mask_scale: float = field(
        default=0.0,
    )
    regularization: Optional[str] = field(
        default=None,
    )
    final_lambda: float = field(
        default=0.0
    )
    global_topk: bool = field(
        default=False
    )
    global_topk_frequency_compute: int = field(
        default=1,
    )

    freeze_bert: bool = field(
        default=True,
    )

    threshold_warmup_steps: int = field(default=5000)

    warmup_threshold_step: Optional[int] = field(default=None)
    warmup_regu_step: Optional[int] = field(default=None)
    linear_threshold: bool = field(default=False)
    save_best_model: bool = field(default=False)

    save_mask: bool = field(default=False)

    temperature: float = field(default=2.0)
    alpha_distil: float = field(default=0.5)
    alpha_ce: float = field(default=0.5)

    magnitude_topk_threshold: bool = field(default=False)
    magnitude_topk_threshold_p: float = field(default=1.0)

    threshold_hold_step: int = field(default=0)

class SMPTrainer(Trainer):
    args: SMPTrainingArguments
    threshold = 1.0
    teacher_model = None
    regu_lambda = 0
    regu_trigger = False
    threshold_mem = None

    log_regu_loss = 0
    log_cls_loss = 0

    use_qa = False

    def set_teacher_model(self, teacher_model):
        self.teacher_model = teacher_model.to(self.args.device)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            if self.args.freeze_bert:
                for n, p in self.model.named_parameters():
                    if 'mask_score' not in n:
                        p.requires_grad = False

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "mask_score" in n and p.requires_grad],
                    "lr": self.args.mask_scores_learning_rate,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "mask_score" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "mask_score" not in n and p.requires_grad and any(nd in n for nd in no_decay)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)   # type: ignore
        return self.optimizer

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):


        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        if self.args.global_topk:
            inputs['threshold'] = self.threshold_mem
        else:
            inputs['threshold'] = self.threshold

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()

        inputs = self._prepare_inputs(inputs)

        if self.args.max_steps < 0:
            import math
            train_dataloader = self.get_train_dataloader()
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
            self.args.max_steps = max_steps
            del train_dataloader
        num_training_steps= self.args.max_steps

        threshold, regu_lambda = schedule_threshold(
            step=self.state.global_step,
            total_step=num_training_steps,
            warmup_steps=self.args.get_warmup_steps(num_training_steps),
            final_threshold=self.args.final_threshold,
            initial_threshold=self.args.initial_threshold,
            final_warmup=self.args.final_warmup,
            initial_warmup=self.args.initial_warmup,
            final_lambda=self.args.final_lambda,
        )


        if 'topK' in self.args.pruning_method:
            if self.args.warmup_threshold_step is not None:
                step = self.state.global_step
                if self.args.threshold_hold_step > 0:
                    step = step // (self.args.threshold_hold_step + 1)
                final_threshold = self.args.final_threshold
                initial_threshold = self.args.initial_threshold
                if step > self.args.warmup_threshold_step:
                    threshold = final_threshold
                else:
                    spars_warmup_steps = 0
                    spars_schedu_steps = self.args.warmup_threshold_step
                    mul_coeff = 1 - (step - spars_warmup_steps) / spars_schedu_steps
                    if self.args.linear_threshold:
                        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 1)
                    else:
                        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
                if self.args.warmup_regu_step:
                    final_threshold = 0.1
                    initial_threshold = 0
                    if step > self.args.warmup_regu_step:
                        _threshold = final_threshold
                    else:
                        spars_warmup_steps = 0
                        spars_schedu_steps = self.args.warmup_regu_step
                        mul_coeff = 1 - (step - spars_warmup_steps) / spars_schedu_steps
                        if self.args.linear_threshold:
                            _threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 1)
                        else:
                            _threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
                    regu_lambda = self.args.final_lambda * _threshold / final_threshold
                else:
                    regu_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                    _, regu_lambda = schedule_threshold(
                        step=self.state.global_step,
                        total_step=num_training_steps,
                        warmup_steps=regu_warmup_steps,
                        final_threshold=0.1,
                        initial_threshold=0,
                        final_warmup=self.args.final_warmup,
                        initial_warmup=self.args.initial_warmup,
                        final_lambda=self.args.final_lambda,
                    )
            else:
                threshold, _ = schedule_threshold(
                    step=self.state.global_step,
                    total_step=num_training_steps,
                    warmup_steps=self.args.threshold_warmup_steps,
                    final_threshold=self.args.final_threshold,
                    initial_threshold=self.args.initial_threshold,
                    final_warmup=self.args.final_warmup,
                    initial_warmup=self.args.initial_warmup,
                    final_lambda=self.args.final_lambda,
                    linear=self.args.linear_threshold,
                )

                _, regu_lambda = schedule_threshold(
                    step=self.state.global_step,
                    total_step=num_training_steps,
                    warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    final_threshold=0.1,
                    initial_threshold=0,
                    final_warmup=self.args.final_warmup,
                    initial_warmup=self.args.initial_warmup,
                    final_lambda=self.args.final_lambda,
                )

            self.threshold = threshold

            if self.args.magnitude_topk_threshold:
                self.set_magnitude_topk_threshold(model, threshold)

        inputs['threshold'] = threshold

        if self.teacher_model is not None:
            if self.use_qa:
                with torch.no_grad():
                    start_logits_tea, end_logits_tea = self.teacher_model(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"],
                        threshold=1.0,
                    )

                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                start_logits_stu, end_logits_stu = outputs[1:]

                loss_start = (
                    nn.functional.kl_div(
                        input=nn.functional.log_softmax(start_logits_stu / self.args.temperature, dim=-1),
                        target=nn.functional.softmax(start_logits_tea / self.args.temperature, dim=-1),
                        reduction="batchmean",
                    )
                    * (self.args.temperature ** 2)
                )
                loss_end = (
                    nn.functional.kl_div(
                        input=nn.functional.log_softmax(end_logits_stu / self.args.temperature, dim=-1),
                        target=nn.functional.softmax(end_logits_tea / self.args.temperature, dim=-1),
                        reduction="batchmean",
                    )
                    * (self.args.temperature ** 2)
                )
                loss_logits = (loss_start + loss_end) / 2.0

                loss = self.args.alpha_distil * loss_logits + self.args.alpha_ce * loss
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits_stu = outputs[1:][0]

                with torch.no_grad():
                    logits_tea = self.teacher_model(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"],
                        threshold=1.0,
                    )[0]

                loss_logits = (
                    nn.functional.kl_div(
                        input=nn.functional.log_softmax(logits_stu / self.args.temperature, dim=-1),
                        target=nn.functional.softmax(logits_tea / self.args.temperature, dim=-1),
                        reduction="batchmean",
                    )
                    * (self.args.temperature ** 2)
                )

                loss = self.args.alpha_distil * loss_logits + self.args.alpha_ce * loss
        else:
            loss = self.compute_loss(model, inputs)

        self.log_cls_loss += loss.item()

        # Regularization
        if self.args.regularization is not None:
            regu_ = regularization(model=model, mode=self.args.regularization)
            self.regu_lambda = regu_lambda
            self.log_regu_loss += regu_.item()
            loss = loss + regu_lambda * regu_

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def set_magnitude_topk_threshold(self, model, threshold):
        with torch.no_grad():
            mag_mask_scores = []
            for i, layer in enumerate(model.bert.encoder.layer):
                p =  1 / self.args.magnitude_topk_threshold_p
                mag_mask_scores.append([
                    (p * layer.attention.self.query.mask_scores).sigmoid().mean().item(),
                    (p * layer.attention.self.key.mask_scores).sigmoid().mean().item(),
                    (p * layer.attention.self.value.mask_scores).sigmoid().mean().item(),
                    (p * layer.attention.output.dense.mask_scores).sigmoid().mean().item(),
                    (p * layer.intermediate.dense.mask_scores).sigmoid().mean().item(),
                    (p * layer.output.dense.mask_scores).sigmoid().mean().item(),
                ])

            sum_mag_mask_scores = [sum(j[i] for j in mag_mask_scores) for i in range(len(mag_mask_scores[0]))]

            t = threshold * len(mag_mask_scores)
            for i, layer in enumerate(model.bert.encoder.layer):
                mag_mask_score = mag_mask_scores[i]
                layer.attention.self.query.local_threshold = mag_mask_score[0] / sum_mag_mask_scores[0] * t if sum_mag_mask_scores[0] != 0 else 0
                layer.attention.self.key.local_threshold = mag_mask_score[1] / sum_mag_mask_scores[1] * t if sum_mag_mask_scores[1] != 0 else 0
                layer.attention.self.value.local_threshold = mag_mask_score[2] / sum_mag_mask_scores[2] * t if sum_mag_mask_scores[2] != 0 else 0
                layer.attention.output.dense.local_threshold = mag_mask_score[3] / sum_mag_mask_scores[3] * t if sum_mag_mask_scores[3] != 0 else 0
                layer.intermediate.dense.local_threshold = mag_mask_score[4] / sum_mag_mask_scores[4] * t if sum_mag_mask_scores[4] != 0 else 0
                layer.output.dense.local_threshold = mag_mask_score[5] / sum_mag_mask_scores[5] * t if sum_mag_mask_scores[5] != 0 else 0

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()  # type: ignore

            # reset tr_loss to zero
            tr_loss -= tr_loss

            self.log_regu_loss  /= self.args.gradient_accumulation_steps
            self.log_cls_loss  /= self.args.gradient_accumulation_steps

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["regu_loss"] = round(self.log_regu_loss / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["cls_loss"] = round(self.log_cls_loss / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs['threshold'] = self.threshold
            logs['regu_lambda'] = self.regu_lambda

            if self.args.magnitude_topk_threshold or self.args.global_topk:
                for i, layer in enumerate(model.bert.encoder.layer):
                    logs[f'layer{i}.query'] = layer.attention.self.query.local_threshold
                    logs[f'layer{i}.key'] = layer.attention.self.key.local_threshold
                    logs[f'layer{i}.value'] = layer.attention.self.value.local_threshold
                    logs[f'layer{i}.output'] = layer.attention.output.dense.local_threshold
                    logs[f'layer{i}.int'] = layer.intermediate.dense.local_threshold
                    logs[f'layer{i}.dense'] = layer.output.dense.local_threshold

            self.log_regu_loss = 0
            self.log_cls_loss = 0

            for name, param in model.named_parameters():
                if "encoder" not in name:
                    continue

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save and self.args.save_mask:
            run_dir = self.args.output_dir
            name = f"mask-{self.state.global_step}.pt"
            save_path = os.path.join(run_dir, name)
            out = {}
            for name, param in model.named_parameters():
                if 'mask_score' in name:
                    mask = TopKBinarizer.apply(param, self.threshold).bool().cpu().clone()
                    out[name] = mask
            torch.save(out, save_path)

        if self.control.should_save and \
           ((abs(self.threshold - self.args.final_threshold) < 0.01 and self.args.pruning_method == 'topK') or
            (self.regu_trigger  and self.args.pruning_method == 'sigmoied_threshold')):
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

class QASMPTrainer(SMPTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.use_qa = True

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
