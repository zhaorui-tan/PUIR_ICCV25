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

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
from reader import Transpose
import math
import random

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_nnunet_loader(data_dir, json_list):
   
    datalist_json = os.path.join(data_dir, json_list)

    # test transform
    test_transform_list = [transforms.LoadImaged(keys=["image"])]
    test_transform_list+=[transforms.CropForegroundD(keys=["image"], source_key="image",select_fn=lambda x: (x[1,...] >= -4.25)[None,...].repeat(2,axis=0))]
    test_transform_list += [transforms.ToTensord(keys=["image"])]
    test_transform = transforms.Compose(test_transform_list)

    test_files = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_sampler = None
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=test_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    loader = test_loader

    return loader

if __name__ == '__main__':
    from tqdm import tqdm
    import pdb
    import numpy as np
    import torch
    data_dir = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/dataplatform/lifescience/pet_ct_pretrain/'
    json_list = 'dataset_PET_CT_quanjing.json'
    loader = get_nnunet_loader(data_dir,json_list)
    means =[]
    maxs = []
    mins = []
    for batch in tqdm(loader):
        img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

        ori_img = np.load(os.path.join(data_dir,'PET_CT_npy_quanjing',img_name))
        original_shape = batch["image_meta_dict"]["spatial_shape"]
        start_coord = batch["foreground_start_coord"][0]
        end_coord = batch["foreground_end_coord"][0]
        restored_outputs = np.ones(np.array(original_shape[0][1:]).tolist())
        
        restored_outputs[start_coord[0]:end_coord[0], start_coord[1]:end_coord[1], start_coord[2]:end_coord[2]] = 0
        pet_background = ori_img[0][restored_outputs==1]
        if pet_background.size == 0:
            continue
        means.append(pet_background.mean())
        maxs.append(pet_background.max())
        mins.append(pet_background.min())
    print(f'mean {np.mean(means)}, max {np.max(maxs)}, min {np.min(mins)}')
