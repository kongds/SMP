from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from .modeling_bert_masked import MaskedBertPreTrainedModel, MaskedBertModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN

import torch
from torch import nn

from .modules import MaskedLinear

class MaskedBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = MaskedLinear(
            config.hidden_size,
            config.hidden_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, threshold=None) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor, threshold=threshold)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MaskedBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense =  MaskedLinear(
            config.hidden_size,
            config.hidden_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
        )
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, threshold=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, threshold=threshold)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MaskedBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MaskedBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder = MaskedLinear(
            config.hidden_size,
            config.vocab_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
            bias=False,
        )


        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states, threshold=None):
        hidden_states = self.transform(hidden_states, threshold=threshold)
        hidden_states = self.decoder(hidden_states, threshold=threshold)
        return hidden_states


class MaskedBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MaskedBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor, threshold=None) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output, threshold=threshold)
        return prediction_scores

class MaskedBertForSMP(MaskedBertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    mask_token_id = None
    label_map = None

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = MaskedBertModel(config)
        if hasattr(config, 'use_sparse_cls') and config.use_sparse_cls:
            self.cls = MaskedBertOnlyMLMHead(config)
        else:
            self.cls = BertOnlyMLMHead(config)

        self.use_qa = False
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

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

        outputs = self.bert(
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
            logits = self.cls(sequence_output)[:, :, self.label_map]
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

            if type(self.cls) is MaskedBertOnlyMLMHead:
                prediction_scores = self.cls(sequence_output[mask_token_mask], threshold)[:, self.label_map]
            else:
                prediction_scores = self.cls(sequence_output[mask_token_mask])[:, self.label_map]


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
