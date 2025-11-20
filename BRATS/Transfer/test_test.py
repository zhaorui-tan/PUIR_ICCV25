# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import nibabel as nib
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from utils.data_utils import get_nnunet_loader
from utils.utils import dice, resample_3d

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from utils.utils import AverageMeter, distributed_all_gather
from nibabel.nifti1 import Nifti1Image
from models.ssl_head_for_task import SSLHead

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
# parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")

parser.add_argument("--modality", default="PET_CT", type=str, help="PET/CT/PET_CT")
parser.add_argument("--crop_foreground", action="store_true", help="crop_foreground")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--few_shot", default=1.0, type=float, help="few shot rate")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed testing")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")


parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")

from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    EnsureType,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    NormalizeIntensity
)
from monai.data import decollate_batch
from monai.utils.enums import MetricReduction
from monai.metrics import DiceMetric
from tqdm import tqdm
import time
# from metrics import structural_similarity as ssim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def main():
    args = parser.parse_args()
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    args.output_directory = output_directory
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)

import sys
import numpy
from scipy import signal
from scipy import ndimage

import gauss

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()






def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = True
    
    val_loader = get_nnunet_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=96,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )


    model = SSLHead(args)
    model.reform_model_for_substask_BRATS(args)

    model_dict = torch.load(pretrained_pth)
    print(f"load pretrained model from {pretrained_pth}")
    state_dict = model_dict["state_dict"]
    # fix potential differences in state dict keys from pre-training to
    # fine-tuning
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    # We now load model weights, setting param `strict` to False, i.e.:
    # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
    # the decoder weights untouched (CNN UNet decoder).
    model.load_state_dict(state_dict, strict=False)
    print("Using pretrained self-supervised Swin UNETR backbone weights !")
   


    model.eval()
    model.cuda(args.gpu)
    ##
    post_trans_output = Compose([EnsureType(),  NormalizeIntensity(channel_wise=True)])
    ##

    # run_dice = AverageMeter()
    start_time = time.time()
    with torch.no_grad():

        count = 0
        affine = np.eye(4)
        keys =  ['t1n', 't1c', 't2w',  't2f', ]

        for i in keys:
            for j in keys:
                scores[f'{i}_2_{j}_ssim'] = []
                scores[f'{i}_2_{j}_psnr'] = []
                scores[f'{i}_2_{j}_mse'] = []
        
        for batch_data in tqdm(val_loader):
    
            val_outputs = []
            all_modal = []
            for i in ['t1n', 't1c', 't2w',  't2f', ]:
                tmp= batch_data[i].squeeze(1)
                all_modal.append(tmp)

            data = torch.stack(all_modal, dim=1).cuda(args.gpu)
            mask = data.ge(-0.35)
            print(pretrained_pth)

            for i in range(4):
                val_inputs = data[:,i,...].repeat(1,4,1,1,1)
         
                val_outputs_1 = sliding_window_inference(
                    val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
                )
                val_outputs_1 = post_trans_output(val_outputs_1) 
                val_outputs_1 = val_outputs_1*mask + data*(1-mask.int()).bool()
                for j in range(4):
                    current_score_key = f'{keys[i]}_2_{keys[j]}'

                    mse_score = torch.nn.MSELoss(reduction='mean')(val_outputs_1[:,j,...][:,None,...], data[:,j,...][:,None,...]) 
                    scores[f'{current_score_key}_mse'].append(mse_score.item())
                    print(f'{current_score_key}_mse', mse_score.item())
                

                    val_output = val_outputs_1[:,j,...][:,None,...].cpu().numpy()
                    val_labels = data[:,j,...][:,None,...].cpu().numpy()

                    rg = max(val_output.max()-val_output.min(),val_labels.max()-val_labels.min())
                    
                    ssim_score = ssim(np.squeeze(val_output), np.squeeze(val_labels), win_size=None, gradient=False, data_range=rg,  channel_axis=None, multichannel=False, gaussian_weights=False, full=False)
                    scores[f'{current_score_key}_ssim'].append(ssim_score)
                    print(f'{current_score_key}_ssim', ssim_score)
                
                    psnr_score = psnr(np.squeeze(val_output), np.squeeze(val_labels), data_range=rg,)
                    scores[f'{current_score_key}_psnr'].append(psnr_score)
                    print(f'{current_score_key}_psnr', psnr_score)

            count += 1

        for s in scores:
            print(s, np.array(scores[s]).mean(), np.array(scores[s]).std())
            print()


if __name__ == "__main__":
    main()
