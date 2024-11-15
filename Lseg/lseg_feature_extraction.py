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

def get_args():
    parser = argparse.ArgumentParser(description="LSeg Per-Pixel Feature Extraction")
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--img_long_side', type=int, help='The long side of an image, e.g. 1280 for [1280, 960]')
    parser.add_argument('--lseg_model', type=str, default='checkpoints/demo_e200.ckpt', help='Where is the LSeg checkpoint')

    args = parser.parse_args()
    return args

def main(args):   
    seed = 1457
    torch.manual_seed(seed)

    data_dir = args.data_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)


    ##############################
    ##### load the LSeg model ####

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.lseg_model,
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
        map_location="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )


    # model
    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval()
    model = model.cpu()
    # scales = (
    #     [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    #     if args.dataset == "citys"
    #     else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    # )
    scales = ([1])


    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]

    model.crop_size = 2*args.img_long_side
    model.base_size = 2*args.img_long_side

    evaluator = LSeg_MultiEvalModule(
        model, scales=scales, flip=True
    ).cuda()
    evaluator.eval()

    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    files = glob(os.path.join(data_dir, '*'))
    for file in tqdm(files):
        print(file)
        feat = extract_lseg_img_feature(file, transform, evaluator)
        print("feat size: ", feat.size())
        file_name = file.split('/')[-1].split('.')[0]
        torch.save(feat, os.path.join(out_dir, '{}.pt'.format(file_name)))
    

    # model = model.cpu().float()
    # print(model)
    # dummy_input = torch.randn(1, 3, 384, 384, dtype=torch.float32)

    # repeat_num = 1000
    # time_list = np.zeros([repeat_num])

    # model_to_export = model.net.pretrained
    # model_to_export = model.net.clip_pretrained.visual

    # with torch.no_grad(): 
    #     for i in tqdm(range(repeat_num)):
    #         start_time = time.time()
    #         dummy_output = model_to_export(dummy_input)
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time
    #         time_list[i] = elapsed_time
    #         # print(f"Execution time: {elapsed_time:.5f} seconds")

    # print(np.mean(time_list))

    # torch.onnx.export(
    #     model_to_export,
    #     dummy_input,
    #     "model.onnx",
    #     # dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
    #     input_names=["input"],
    #     output_names=["output"]
    # )

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    main(args)
