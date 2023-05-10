# coding=utf-8
# Copyright 2020-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Masked Linear module: A fully connected layer that computes an adaptive binary mask on the fly.
The mask (binary or not) is computed at each forward pass and multiplied against
the weight matrix to prune a portion of the weights.
The pruned weight matrix is then multiplied against the inputs (and if necessary, the bias is added).
"""

import math

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from .binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer, ZeroBinarizer


limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    If needed, a score matrix is created to store the importance of each associated weight.
    """

    global_topk = False
    not_use_mask = False
    custom_mask = None
    local_threshold = -1
    current_mask = None
    more_zero = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Choices: ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
                Default: ``topK``
        """
        super(MaskedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)


        assert pruning_method in ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0", "hardconcrete"]
        self.pruning_method = pruning_method

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0",  "hardconcrete"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            self.mask_scores = nn.Parameter(torch.empty(self.weight.size()))
            self.init_mask()


    def init_mask(self):
        if self.pruning_method == "hardconcrete":
            mean = 2.4
            self.mask_scores.data.normal_(mean, 1e-2)
            return

        if self.mask_init == "constant":
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(self.mask_scores, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))



    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    temperature=2./3.
    magical_number=0.8
    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            import pdb;pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        _soft_mask = soft_mask.view(-1)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(_soft_mask, k=num_zeros, largest=False)
                _soft_mask[indices] = 0.
        return soft_mask

    def get_remain_size(self):
        return torch.sum(1 - self.cdf_qz(0, self.mask_scores))

    def forward(self, input: torch.tensor, threshold: float):
        # Get the mask
        if self.local_threshold > 0 and not self.global_topk:
            threshold = self.local_threshold

        if self.more_zero:
            mask = ZeroBinarizer.apply(self.mask_scores)
        elif self.global_topk:
            threshold = ((self.mask_scores > threshold).sum() / (self.mask_scores >= self.mask_scores.min()).sum()).item()
            threshold = max(threshold, 0.005)
            self.local_threshold = threshold
            mask = TopKBinarizer.apply(self.mask_scores, threshold)
        elif self.pruning_method == "topK":
            mask = TopKBinarizer.apply(self.mask_scores, threshold)
        elif self.pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in self.pruning_method
            mask = ThresholdBinarizer.apply(self.mask_scores, threshold, sig)
        elif self.pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(self.weight, threshold)
        elif self.pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if self.training:
                u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / b)
            else:
                s = torch.sigmoid(self.mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        elif self.pruning_method == 'hardconcrete':
            if self.train:
                mask = self.sample_z(self.mask_scores)
            else:
                mask = self.deterministic_z(self.mask_scores.numel(), self.mask_scores)

        # Mask weights with computed mask
        # if ((mask == 1) + (mask == 0)).sum() != (mask > -100).sum() and not self.train:
            # import pdb;pdb.set_trace()

        if self.custom_mask is not None:
            weight_thresholded = self.custom_mask * self.weight
            # prec = (self.custom_mask.sum() / (mask > -100).sum()).item()
            # print(prec)
        else:
            weight_thresholded = mask * self.weight
            self.current_mask = mask.detach()

        # Compute output (linear layer) with masked weights
        # input @ weight_thresholded.T + self.bias
        if self.not_use_mask:
            return nn.functional.linear(input, self.weight, self.bias)
        else:
            return nn.functional.linear(input, weight_thresholded, self.bias)
