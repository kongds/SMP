import os
import torch
from emmental.modeling_bert_smp import MaskedBertForSMP
from emmental.modeling_roberta_smp import MaskedRobertaForSMP
from emmental.modules.binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', type=str, default='/home/hx/data/bert_base_uncased/')
argparser.add_argument('--threshold', type=float, default=None)
argparser.add_argument('--output_name', type=str)
argparser.add_argument('--magnitude_topk_threshold_p', type=float, default=1)

arg = argparser.parse_args()

is_mag = 'mag' in arg.model_path

if arg.threshold is None:
    arg.threshold = float([i for i in arg.model_path.split('_') if '0.' in i][0])
    print(arg.threshold)
if 'checkpoint' not in arg.model_path:
    m_step = 1000000
    _model_path = arg.model_path
    for path in os.listdir(arg.model_path):
        if 'checkpoint' in path:
            print(path)
            step = int(path.split('checkpoint-')[-1])
            if step < m_step:
                _model_path = os.path.join(arg.model_path, path)
                m_step = step
    arg.model_path = _model_path
if arg.output_name is None:
    arg.output_name = './outs/'+'_'.join(arg.model_path.split('/')[-2].split('_')[1:])+'.pt'
print(arg.model_path, arg.output_name, arg.threshold)
st_0_10 = torch.load(os.path.join(arg.model_path, "pytorch_model.bin"), map_location='cpu')
threshold = arg.threshold


del_name = []
for name in st_0_10:
    if 'mask_score' in name:
        pass
    else:
        del_name.append(name)
for name in del_name:
    del st_0_10[name]

layers = []
for i in range(12):
    one_layer = {}
    for name in [f'bert.encoder.layer.{i}.attention.self.query.mask_scores',
                 f'bert.encoder.layer.{i}.attention.self.key.mask_scores',
                 f'bert.encoder.layer.{i}.attention.self.value.mask_scores',
                 f'bert.encoder.layer.{i}.attention.output.dense.mask_scores',
                 f'bert.encoder.layer.{i}.intermediate.dense.mask_scores',
                 f'bert.encoder.layer.{i}.output.dense.mask_scores']:
        real_name = {f'bert.encoder.layer.{i}.attention.self.query.mask_scores': 'query',
                     f'bert.encoder.layer.{i}.attention.self.key.mask_scores': 'key',
                     f'bert.encoder.layer.{i}.attention.self.value.mask_scores': 'value',
                     f'bert.encoder.layer.{i}.attention.output.dense.mask_scores': 'attn_out',
                     f'bert.encoder.layer.{i}.intermediate.dense.mask_scores': 'intermediate',
                     f'bert.encoder.layer.{i}.output.dense.mask_scores': 'output'}[name]
        one_layer[real_name] = st_0_10[name]
    layers.append(one_layer)

keys = sorted(layers[0].keys())

mag_mask_scores = []
for layer in layers:
    p = 1 / arg.magnitude_topk_threshold_p
    mag_mask_scores.append([
        (p * layer[key]).sigmoid().mean().item() for key in keys
    ])

sum_mag_mask_scores = [sum(j[i] for j in mag_mask_scores) for i in range(len(mag_mask_scores[0]))]

t = threshold * len(mag_mask_scores)
for i, layer in enumerate(layers):
    mag_mask_score = mag_mask_scores[i]
    for j, key in enumerate(keys):
        if is_mag:
            layer[key] = TopKBinarizer.apply(layer[key],
                                            mag_mask_score[j] / sum_mag_mask_scores[j] * t if sum_mag_mask_scores[j] != 0 else 0)
        else:
            layer[key] = TopKBinarizer.apply(layer[key], arg.threshold)

output = {}
for i, layer in enumerate(layers):
    for key in keys:
        name_map = {f'bert.encoder.layer.{i}.attention.self.query.mask_scores': 'query',
                     f'bert.encoder.layer.{i}.attention.self.key.mask_scores': 'key',
                     f'bert.encoder.layer.{i}.attention.self.value.mask_scores': 'value',
                     f'bert.encoder.layer.{i}.attention.output.dense.mask_scores': 'attn_out',
                     f'bert.encoder.layer.{i}.intermediate.dense.mask_scores': 'intermediate',
                     f'bert.encoder.layer.{i}.output.dense.mask_scores': 'output'}
        name_map = {name_map[key]: key for key in name_map}

        output[name_map[key]] = layer[key].bool().clone()

#torch.save(output, f'mnli_0.03_mag_mask_score.pt')
def print_attn(name):
    print(name)
    for i in output:
        if name in i:
            for j in range(0, 768, 64):
                print(round(output[i][j:j+64,:].sum().item() / 64/768 * 100), end=' ')
            print()

a, b = 0, 0
for i in output:
    _a = output[i].sum().item()
    _b = output[i].numel()
    a += _a
    b += _b
    print(i, _a/_b)

print('threshold', a/b)

print_attn('query')

print_attn('key')

print_attn('value')

#
#out = torch.load('mnli_0.10_mag_s1_mask_score.pt') # 88MB
import numpy as np
keys = sorted(output.keys())
flat = np.packbits(torch.cat([output[key].view(-1) for key in keys]).numpy())
np.save(arg.output_name.replace('.pt', '.npy'), flat) # 11MB
