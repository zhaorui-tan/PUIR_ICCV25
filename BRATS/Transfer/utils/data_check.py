from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final
from typing import Tuple
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg
from monai.transforms.transform import MapTransform


class CheckSized(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        size: Tuple,
        pad_value: float = 0,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.size = size
    def check_size(self,data):
        if len(data.shape) != len(self.size)+1:
            print(f'bad data {self.name} {data.shape}')
            return np.ones(self.size).astype(data.dtype)[None,...].repeat(data.shape[0],axis=0)
        for i,j in zip(data.shape[1:],self.size):
            if i < j:
                print(f'bad data {self.name} {data.shape}')
                return np.ones(self.size).astype(data.dtype)[None,...].repeat(data.shape[0],axis=0)
        return data

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.name = data['image_meta_dict']["filename_or_obj"]
        for key in self.key_iterator(d):
            d[key] = self.check_size(d[key])
        return d