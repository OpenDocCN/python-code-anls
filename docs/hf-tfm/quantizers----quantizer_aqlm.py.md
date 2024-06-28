# `.\quantizers\quantizer_aqlm.py`

```py
# 版权声明及版权许可信息
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的模块和库
import importlib
from typing import TYPE_CHECKING, Optional

# 导入版本管理相关的模块
from packaging import version

# 导入基类 HfQuantizer
from .base import HfQuantizer

# 检查类型，仅在类型检查时导入相关模块
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

# 导入 AQLM 相关函数和类
from ..integrations import replace_with_aqlm_linear
from ..utils import is_accelerate_available, is_aqlm_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin

# 如果 Torch 可用，则导入 Torch 模块
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)


class AqlmHfQuantizer(HfQuantizer):
    """
    AQLM 方法的量化器。支持加载预量化模型。
    """

    # 需要校准
    requires_calibration = True
    # 需要的包
    required_packages = ["aqlm"]
    # 最佳量化器，默认为 None
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        # 检查是否安装了 Accelerate 加速库
        if not is_accelerate_available():
            raise ImportError("Using `aqlm` quantization requires Accelerate: `pip install accelerate`")

        # 检查是否安装了 AQLM 库
        if not is_aqlm_available():
            raise ImportError("Using `aqlm` quantization requires AQLM: `pip install aqlm[gpu,cpu]`")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        # 如果未指定 Torch 的数据类型
        if torch_dtype is None:
            # 如果 CUDA 可用，则默认为 torch.float16
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info(
                    "CUDA available. Assuming AQLM inference on GPU and loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually."
                )
            # 如果 CUDA 不可用，默认为 torch.float32
            else:
                torch_dtype = torch.float32
                logger.info(
                    "CUDA is unavailable. Assuming AQLM inference on CPU and loading the model in `torch.float32`. To overwrite it, set `torch_dtype` manually."
                )
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        # 替换模型中的线性层为 AQLM 线性层
        replace_with_aqlm_linear(
            model,
            quantization_config=self.quantization_config,
            linear_weights_not_to_quantize=self.quantization_config.linear_weights_not_to_quantize,
        )
        # 将量化配置信息保存到模型配置中
        model.config.quantization_config = self.quantization_config
    # 在模型加载权重后处理模型的方法，返回未修改的模型对象
    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    # 属性方法，用于检查模型是否可训练
    # 如果当前安装的 `aqlm` 版本支持训练，返回 True
    # 否则记录警告信息并返回 False
    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        # 检查当前安装的 `aqlm` 版本是否大于等于 1.0.2
        aqlm_supports_training = version.parse(importlib.metadata.version("aqlm")) >= version.parse("1.0.2")
        if aqlm_supports_training:
            return True
        else:
            # 记录警告信息，提示用户更新 `aqlm` 版本以支持训练
            logger.warn(
                f"Currently installed `aqlm` version ({importlib.metadata.version('aqlm')}) doesn't support training. If you wish to train a quantized model, please update `aqlm` with `pip install aqlm>=1.0.2`"
            )
            return False

    # 属性方法，用于检查对象是否可序列化
    @property
    def is_serializable(self):
        return True
```