from transformers.models.roberta.modeling_roberta import RobertaLMHead
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN

import torch
from torch import nn

from .modules import MaskedLinear


from .modeling_roberta_mask import MaskedRobertaPreTrainedModel, MaskedRobertaModel

class MaskedRobertaForSMP(MaskedRobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    mask_token_id = None
    label_map = None

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = MaskedRobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.use_qa = False
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        start_positions=None,
        end_positions=None,
        threshold=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            threshold=threshold,
        )

        if self.use_qa:
            sequence_output = outputs[0]
            logits = self.lm_head(sequence_output)[:, :, self.label_map]
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (
                start_logits.contiguous(),
                end_logits.contiguous(),
            ) + outputs[2:]
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

            return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
        else:
            sequence_output = outputs[0]

            mask_token_mask = input_ids ==  self.mask_token_id
            if (mask_token_mask).sum(-1).bool().sum() != input_ids.shape[0]:
                mask_token_mask[~(input_ids ==  self.mask_token_id).sum(-1).bool(), 0] = True

            prediction_scores = self.lm_head(sequence_output[mask_token_mask])[:, self.label_map]

            masked_lm_loss = None
            if labels is not None:
                if self.config.num_labels == 1:
                    loss_fct = MSELoss()
                    masked_lm_loss = loss_fct(prediction_scores.squeeze(), labels.squeeze())
                else:
                    loss_fct = CrossEntropyLoss()  # -100 index = padding token
                    masked_lm_loss = loss_fct(prediction_scores, labels.view(-1))
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output)
