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

from fusion_util import extract_lseg_img_feature

# 모델 로드
model = LSegModule.load_from_checkpoint(
    checkpoint_path='checkpoints/demo_e200.ckpt',
    data_path='../datasets/',
    dataset='ade20k',
    backbone='clip_vitl16_384',
    aux=False,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0,
    batch_size=1,
    max_epochs=0,
    ignore_index=255,
    dropout=0.0,
    scale_inv=False,
    augment=False,
    no_batchnorm=False,
    widehead=True,
    widehead_hr=False,
    map_location="cuda",  # 오타 수정: map_location
    arch_option=0,
    block_depth=0,
    activation='lrelu',
)

# 모델의 pretrained 부분을 가져옴
# model = model.net.pretrained.model
model = model.eval()

# CUDA가 사용 가능하면 모델을 GPU로 이동
if torch.cuda.is_available():
    model = model.cuda()

print(model)

# model_to_export = model.net.pretrained
model_to_export = model.net

# 더미 입력 생성 및 CUDA 설정
dummy_input = torch.randn(1, 3, 384, 384)

# # 시간 측정
# num_iterations = 100  # 반복 횟수
# total_time = 0.0

# # output = model(dummy_input)
# # print(output.size())

# for _ in tqdm(range(num_iterations), desc="Inference Timing"):
#     start = time.time()
#     with torch.no_grad():  # 평가 모드에서 불필요한 그래디언트 계산 비활성화
#         output = model(dummy_input)
#     end = time.time()
#     total_time += (end - start)

# average_time_per_inference = total_time / num_iterations
# print(f"Average time per inference: {average_time_per_inference:.6f} seconds")


# Move the entire model to CUDA if available
if torch.cuda.is_available():
    print("cuda available!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    model_to_export = model.net.cuda()  # Ensure this is the part you want to export
    dummy_input = dummy_input.cuda()
else:
    model_to_export = model.net

model_to_export = model_to_export.cuda()
for param in model_to_export.parameters():
    param.data = param.data.cuda()
    if param.grad is not None:
        param.grad.data = param.grad.data.cuda()

# Export to ONNX
torch.onnx.export(
    model_to_export,  # Make sure this is on the correct device
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=12,
    do_constant_folding=True
)