import os
import argparse
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from glob import glob
from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
import onnx
import time
import numpy as np
from tqdm import tqdm

torch.cuda.empty_cache()

# def get_labels(dataset):
#     labels = []
#     path = 'data/{}_objectInfo150.txt'.format(dataset)
#     assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
#     f = open(path, 'r') 
#     lines = f.readlines()      
#     for line in lines: 
#         label = line.strip().split(',')[-1].split(';')[0]
#         labels.append(label)
#     f.close()
#     if dataset in ['ade20k']:
#         labels = labels[1:]
#     return labels


# labels = get_labels(dataset='ade20k')

labels = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate', 'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag']

# print(labels)

from modules.lseg_module import LSegNet

model = LSegNet(
            labels=labels,
            path='/home/jihoon-epitone/Downloads/Pixel_aligned_VLM/Lseg/checkpoints/demo_e200.ckpt',
            backbone='clip_vitl16_384',
            features=256,
            crop_size=384,
            arch_option=0,
            block_depth=0,
            activation='relu',
        )

# print(model)

# 모델을 평가 모드로 설정
model.eval()

# 더미 입력 텐서 정의 (모델 입력 크기에 맞게 조정)
dummy_input = torch.randn(1, 3, 384, 384)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
dummy_input = dummy_input.to(device)

for name, param in model.named_parameters():
    if param.dtype == torch.int64:
        param.data = param.data.to(torch.int32)


# output = model(dummy_input)
# print(output.size())

# # 평균 수행 시간 측정
# num_iterations = 1000

# # torch.cuda.empty_cache()
# start_time = time.time()

# with torch.no_grad():
#     for _ in tqdm(range(num_iterations), desc="Running model"):
#         output = model(dummy_input)

# end_time = time.time()
# average_time = (end_time - start_time) / num_iterations

# print(f'Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB')
# print(f'Memory reserved: {torch.cuda.memory_reserved() / (1024 ** 2)} MB')
# print(f'Average inference time over {num_iterations} iterations: {average_time:.6f} seconds')

# output = model(dummy_input)

# print(output.size())



# 모델을 ONNX로 변환
torch.onnx.export(
    model,
    dummy_input,
    "lseg_model_241105.onnx",
    export_params=True,
    opset_version=13,  # or higher
    do_constant_folding=False,  # Try turning this off if it's causing issues
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("ONNX 모델 변환 완료!")