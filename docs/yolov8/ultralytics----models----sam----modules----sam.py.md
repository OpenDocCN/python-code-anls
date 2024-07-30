# `.\yolov8\ultralytics\models\sam\modules\sam.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入需要的模块和类
from typing import List

import torch
from torch import nn

# 导入本模块中的子模块
from .decoders import MaskDecoder
from .encoders import ImageEncoderViT, PromptEncoder

# 定义 Sam 类，用于对象分割任务
class Sam(nn.Module):
    """
    Sam (Segment Anything Model) is designed for object segmentation tasks. It uses image encoders to generate image
    embeddings, and prompt encoders to encode various types of input prompts. These embeddings are then used by the mask
    decoder to predict object masks.

    Attributes:
        mask_threshold (float): Threshold value for mask prediction.
        image_format (str): Format of the input image, default is 'RGB'.
        image_encoder (ImageEncoderViT): The backbone used to encode the image into embeddings.
        prompt_encoder (PromptEncoder): Encodes various types of input prompts.
        mask_decoder (MaskDecoder): Predicts object masks from the image and prompt embeddings.
        pixel_mean (List[float]): Mean pixel values for image normalization.
        pixel_std (List[float]): Standard deviation values for image normalization.
    """

    # 默认的阈值用于掩模预测
    mask_threshold: float = 0.0
    # 默认的输入图像格式为 RGB
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = (123.675, 116.28, 103.53),
        pixel_std: List[float] = (58.395, 57.12, 57.375),
    ) -> None:
        """
        Initialize the Sam class to predict object masks from an image and input prompts.

        Note:
            All forward() operations moved to SAMPredictor.

        Args:
            image_encoder (ImageEncoderViT): The backbone used to encode the image into image embeddings.
            prompt_encoder (PromptEncoder): Encodes various types of input prompts.
            mask_decoder (MaskDecoder): Predicts masks from the image embeddings and encoded prompts.
            pixel_mean (List[float], optional): Mean values for normalizing pixels in the input image. Defaults to
                (123.675, 116.28, 103.53).
            pixel_std (List[float], optional): Std values for normalizing pixels in the input image. Defaults to
                (58.395, 57.12, 57.375).
        """
        super().__init__()
        # 设置图像编码器、提示编码器和掩模解码器
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # 注册用于图像归一化的均值和标准差作为缓冲区
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
```