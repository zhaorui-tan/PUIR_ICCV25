import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from pathlib import Path
from monai.utils import ImageMetaKey as Key
from monai.utils.module import look_up_option
from monai.config import DtypeLike, KeysCollection
from monai.data.utils import correct_nifti_header_if_necessary
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import ensure_tuple, ensure_tuple_rep, optional_import
from monai.transforms.transform import MapTransform
from monai.transforms import InvertibleTransform
from monai.data.utils import is_supported_format
import inspect
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
import sys
from monai.data.image_reader import ImageReader, ITKReader, NibabelReader, PILReader, NumpyReader
from monai.utils.enums import TransformBackends
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
class Transpose(MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        axis: tuple=None
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.axis = axis


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if isinstance(d[key], torch.Tensor):
                d[key] = torch.permute(d[key],self.axis)
            else:
                d[key] = np.transpose(d[key],self.axis)
            self.push_transform(d, key)
        return d
