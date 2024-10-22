# `.\diffusers\pipelines\deprecated\spectrogram_diffusion\pipeline_spectrogram_diffusion.py`

```py
# 版权所有 2022 The Music Spectrogram Diffusion Authors.
# 版权所有 2024 The HuggingFace Team。保留所有权利。
#
# 根据 Apache License, Version 2.0 (以下称为“许可证”)进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 按“原样”分发，不附带任何明示或暗示的担保或条件。
# 请参见许可证以获取有关权限的具体说明和
# 限制。

import math  # 导入数学库以进行数学运算
from typing import Any, Callable, List, Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库以进行数值计算
import torch  # 导入 PyTorch 库以进行深度学习

from ....models import T5FilmDecoder  # 从模型模块导入 T5FilmDecoder 类
from ....schedulers import DDPMScheduler  # 从调度器模块导入 DDPMScheduler 类
from ....utils import is_onnx_available, logging  # 导入实用函数和日志功能
from ....utils.torch_utils import randn_tensor  # 从 PyTorch 实用程序导入随机张量函数

if is_onnx_available():  # 如果 ONNX 可用
    from ...onnx_utils import OnnxRuntimeModel  # 导入 ONNX 运行时模型类

from ...pipeline_utils import AudioPipelineOutput, DiffusionPipeline  # 导入音频管道输出和扩散管道类
from .continuous_encoder import SpectrogramContEncoder  # 从当前模块导入连续编码器
from .notes_encoder import SpectrogramNotesEncoder  # 从当前模块导入音符编码器

logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器，禁用 pylint 名称检查

TARGET_FEATURE_LENGTH = 256  # 定义目标特征长度常量

class SpectrogramDiffusionPipeline(DiffusionPipeline):  # 定义一个继承自 DiffusionPipeline 的类
    r"""  # 开始类文档字符串
    用于无条件音频生成的管道。

    此模型继承自 [`DiffusionPipeline`]. 检查超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    参数：
        notes_encoder ([`SpectrogramNotesEncoder`]):  # 音符编码器参数
        continuous_encoder ([`SpectrogramContEncoder`]):  # 连续编码器参数
        decoder ([`T5FilmDecoder`]):  # 解码器参数
            用于去噪编码音频潜在特征的 [`T5FilmDecoder`]。
        scheduler ([`DDPMScheduler`]):  # 调度器参数
            与 `decoder` 一起使用以去噪编码音频潜在特征的调度器。
        melgan ([`OnnxRuntimeModel`]):  # MELGAN 参数
    """  # 结束类文档字符串

    _optional_components = ["melgan"]  # 定义可选组件列表

    def __init__(  # 构造函数定义
        self,
        notes_encoder: SpectrogramNotesEncoder,  # 音符编码器
        continuous_encoder: SpectrogramContEncoder,  # 连续编码器
        decoder: T5FilmDecoder,  # 解码器
        scheduler: DDPMScheduler,  # 调度器
        melgan: OnnxRuntimeModel if is_onnx_available() else Any,  # MELGAN，如果可用则为 OnnxRuntimeModel
    ) -> None:  # 返回类型为 None
        super().__init__()  # 调用父类构造函数

        # 从 MELGAN
        self.min_value = math.log(1e-5)  # 设置最小值，与 MelGAN 训练匹配
        self.max_value = 4.0  # 设置最大值，适用于大多数示例
        self.n_dims = 128  # 设置维度数

        self.register_modules(  # 注册模块，包括编码器和调度器
            notes_encoder=notes_encoder,  # 注册音符编码器
            continuous_encoder=continuous_encoder,  # 注册连续编码器
            decoder=decoder,  # 注册解码器
            scheduler=scheduler,  # 注册调度器
            melgan=melgan,  # 注册 MELGAN
        )
    # 定义特征缩放函数，将输入特征线性缩放到网络输出范围
        def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
            # 解包输出范围的最小值和最大值
            min_out, max_out = output_range
            # 如果启用剪辑，将特征限制在指定的最小值和最大值之间
            if clip:
                features = torch.clip(features, self.min_value, self.max_value)
            # 将特征缩放到[0, 1]的范围
            zero_one = (features - self.min_value) / (self.max_value - self.min_value)
            # 将特征缩放到[min_out, max_out]的范围并返回
            return zero_one * (max_out - min_out) + min_out
    
        # 定义反向缩放函数，将网络输出线性缩放到特征范围
        def scale_to_features(self, outputs, input_range=(-1.0, 1.0), clip=False):
            # 解包输入范围的最小值和最大值
            min_out, max_out = input_range
            # 如果启用剪辑，将输出限制在指定的范围
            outputs = torch.clip(outputs, min_out, max_out) if clip else outputs
            # 将输出缩放到[0, 1]的范围
            zero_one = (outputs - min_out) / (max_out - min_out)
            # 将输出缩放到[self.min_value, self.max_value]的范围并返回
            return zero_one * (self.max_value - self.min_value) + self.min_value
    
        # 定义编码函数，将输入令牌和连续输入编码为网络的表示
        def encode(self, input_tokens, continuous_inputs, continuous_mask):
            # 创建令牌掩码，标识有效输入
            tokens_mask = input_tokens > 0
            # 使用令牌编码器编码输入令牌和掩码
            tokens_encoded, tokens_mask = self.notes_encoder(
                encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
            )
    
            # 使用连续输入编码器编码连续输入和掩码
            continuous_encoded, continuous_mask = self.continuous_encoder(
                encoder_inputs=continuous_inputs, encoder_inputs_mask=continuous_mask
            )
    
            # 返回编码后的令牌和连续输入
            return [(tokens_encoded, tokens_mask), (continuous_encoded, continuous_mask)]
    
        # 定义解码函数，根据编码和掩码生成输出
        def decode(self, encodings_and_masks, input_tokens, noise_time):
            # 将噪声时间赋值给时间步长
            timesteps = noise_time
            # 如果时间步长不是张量，则转换为张量
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=input_tokens.device)
            # 如果时间步长是张量但维度为0，则增加一个维度
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(input_tokens.device)
    
            # 通过广播使时间步长适应批处理维度，确保与ONNX/Core ML兼容
            timesteps = timesteps * torch.ones(input_tokens.shape[0], dtype=timesteps.dtype, device=timesteps.device)
    
            # 使用解码器生成 logits
            logits = self.decoder(
                encodings_and_masks=encodings_and_masks, decoder_input_tokens=input_tokens, decoder_noise_time=timesteps
            )
            # 返回生成的 logits
            return logits
    
        # 使用torch.no_grad()装饰器，避免计算梯度
        @torch.no_grad()
        def __call__(
            # 定义调用方法的参数，包括输入令牌、生成器和推理步骤等
            input_tokens: List[List[int]],
            generator: Optional[torch.Generator] = None,
            num_inference_steps: int = 100,
            return_dict: bool = True,
            output_type: str = "np",
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            callback_steps: int = 1,
```