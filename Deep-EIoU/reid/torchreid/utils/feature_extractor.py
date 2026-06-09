from __future__ import absolute_import
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model


class FeatureExtractor(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=False
    ):
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (model_path and check_isfile(model_path)),
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(
                model, (1, 3, image_size[0], image_size[1])
            )
            print('Model: {}'.format(model_name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        device = torch.device(device)
        model.to(device)

        # Normalization constants as (1, 3, 1, 1) tensors so a whole
        # (B, C, H, W) batch can be normalized in one broadcasted op on device.
        mean = torch.tensor(pixel_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(pixel_std, device=device).view(1, 3, 1, 1)

        # Class attributes
        self.model = model
        self.image_size = image_size  # (height, width)
        self.pixel_norm = pixel_norm
        self.mean = mean
        self.std = std
        self.device = device

    def _preprocess(self, crops):
        """Resize + normalize a list of HxWxC numpy crops into one GPU batch.

        Each crop is moved to the GPU as uint8 (a quarter of the bytes of a
        float transfer), resized to ``image_size`` with a bilinear
        ``F.interpolate``, then the whole stack is scaled to [0, 1] and
        normalized as a single broadcasted tensor op. No PIL, no per-crop CPU
        resize/normalize on the critical path.

        Crops have different sizes, so they are resized individually before they
        can be stacked -- but each resize now runs on the GPU, and all the heavy
        work (resize/normalize) happens in one batch.

        Returns a ``(N, 3, H, W)`` float tensor on ``self.device``.
        """
        height, width = self.image_size
        resized = []
        for crop in crops:
            tensor = torch.from_numpy(np.ascontiguousarray(crop)).to(self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
            tensor = F.interpolate(
                tensor, size=(height, width),
                mode='bilinear', align_corners=False
            )
            resized.append(tensor)

        batch = torch.cat(resized, dim=0) / 255.0
        if self.pixel_norm:
            batch = (batch - self.mean) / self.std
        return batch

    def __call__(self, input):
        if isinstance(input, list):
            crops = []
            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')
                    crops.append(np.asarray(image))

                elif isinstance(element, np.ndarray):
                    crops.append(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

            images = self._preprocess(crops)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            images = self._preprocess([np.asarray(image)])

        elif isinstance(input, np.ndarray):
            images = self._preprocess([input])

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features
