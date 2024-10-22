# `.\diffusers\pipelines\unclip\pipeline_unclip.py`

```py
# 版权声明，指明版权归属及许可信息
# Copyright 2024 Kakao Brain and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可协议，使用该文件需要遵循许可条款
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 许可协议的副本可以在以下网址获取
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律规定或书面同意，否则软件按“现状”提供，不提供任何形式的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 具体的许可条款可以查看许可协议
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块，用于获取对象的签名和源代码
import inspect
# 从 typing 模块导入类型注解
from typing import List, Optional, Tuple, Union

# 导入 torch 库以进行张量运算
import torch
# 从 torch.nn.functional 导入常用的函数接口
from torch.nn import functional as F
# 导入 CLIP 文本模型和分词器
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
# 从 CLIP 模型导入文本模型输出类
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

# 从当前目录导入多个模型和调度器
from ...models import PriorTransformer, UNet2DConditionModel, UNet2DModel
from ...schedulers import UnCLIPScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .text_proj import UnCLIPTextProjModel

# 初始化日志记录器，用于记录运行信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个用于文本到图像生成的管道类
class UnCLIPPipeline(DiffusionPipeline):
    """
    使用 unCLIP 进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。有关通用方法的文档请参见超类文档
    (下载、保存、在特定设备上运行等)。

    参数:
        text_encoder ([`~transformers.CLIPTextModelWithProjection`]):
            冻结的文本编码器。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行分词的 `CLIPTokenizer`。
        prior ([`PriorTransformer`]):
            经典的 unCLIP 先验，用于从文本嵌入近似图像嵌入。
        text_proj ([`UnCLIPTextProjModel`]):
            用于准备和组合嵌入的实用类，嵌入会在传递给解码器之前进行处理。
        decoder ([`UNet2DConditionModel`]):
            将图像嵌入反转为图像的解码器。
        super_res_first ([`UNet2DModel`]):
            超分辨率 UNet。在超分辨率扩散过程的所有步骤中使用，除了最后一步。
        super_res_last ([`UNet2DModel`]):
            超分辨率 UNet。在超分辨率扩散过程的最后一步使用。
        prior_scheduler ([`UnCLIPScheduler`]):
            在先验去噪过程中使用的调度器（修改版 [`DDPMScheduler`]).
        decoder_scheduler ([`UnCLIPScheduler`]):
            在解码器去噪过程中使用的调度器（修改版 [`DDPMScheduler`]).
        super_res_scheduler ([`UnCLIPScheduler`]):
            在超分辨率去噪过程中使用的调度器（修改版 [`DDPMScheduler`]).
    """

    # 指定在 CPU 卸载时要排除的组件
    _exclude_from_cpu_offload = ["prior"]

    # 定义先验字段，类型为 PriorTransformer
    prior: PriorTransformer
    # 定义解码器模型类型
    decoder: UNet2DConditionModel
    # 定义文本投影模型类型
    text_proj: UnCLIPTextProjModel
    # 定义文本编码器模型类型
    text_encoder: CLIPTextModelWithProjection
    # 定义分词器类型
    tokenizer: CLIPTokenizer
    # 定义超分辨率模型的第一个部分
    super_res_first: UNet2DModel
    # 定义超分辨率模型的最后一个部分
    super_res_last: UNet2DModel

    # 定义优先级调度器
    prior_scheduler: UnCLIPScheduler
    # 定义解码器调度器
    decoder_scheduler: UnCLIPScheduler
    # 定义超分辨率调度器
    super_res_scheduler: UnCLIPScheduler

    # 定义模型的CPU卸载顺序
    model_cpu_offload_seq = "text_encoder->text_proj->decoder->super_res_first->super_res_last"

    # 初始化函数，定义所需的参数
    def __init__(
        # 定义优先变换器参数
        prior: PriorTransformer,
        # 定义解码器模型参数
        decoder: UNet2DConditionModel,
        # 定义文本编码器参数
        text_encoder: CLIPTextModelWithProjection,
        # 定义分词器参数
        tokenizer: CLIPTokenizer,
        # 定义文本投影模型参数
        text_proj: UnCLIPTextProjModel,
        # 定义超分辨率模型的第一个部分参数
        super_res_first: UNet2DModel,
        # 定义超分辨率模型的最后一个部分参数
        super_res_last: UNet2DModel,
        # 定义优先级调度器参数
        prior_scheduler: UnCLIPScheduler,
        # 定义解码器调度器参数
        decoder_scheduler: UnCLIPScheduler,
        # 定义超分辨率调度器参数
        super_res_scheduler: UnCLIPScheduler,
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 注册模块，存储所需的模型和调度器
        self.register_modules(
            prior=prior,
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            prior_scheduler=prior_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

    # 准备潜在变量的函数
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果没有给定潜在变量，生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果给定的潜在变量形状与预期不符，则抛出异常
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在变量移动到指定设备
            latents = latents.to(device)

        # 将潜在变量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 编码提示的函数
    def _encode_prompt(
        self,
        # 提示内容
        prompt,
        # 设备类型
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否执行分类器自由引导
        do_classifier_free_guidance,
        # 文本模型输出，可选
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        # 文本注意力掩码，可选
        text_attention_mask: Optional[torch.Tensor] = None,
    @torch.no_grad()
    # 调用函数，用于生成图像
    def __call__(
        # 提示内容，可选
        prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: int = 1,
        # 优先模型的推理步骤数量，默认为25
        prior_num_inference_steps: int = 25,
        # 解码器的推理步骤数量，默认为25
        decoder_num_inference_steps: int = 25,
        # 超分辨率的推理步骤数量，默认为7
        super_res_num_inference_steps: int = 7,
        # 随机数生成器，可选
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 优先潜在变量，可选
        prior_latents: Optional[torch.Tensor] = None,
        # 解码器潜在变量，可选
        decoder_latents: Optional[torch.Tensor] = None,
        # 超分辨率潜在变量，可选
        super_res_latents: Optional[torch.Tensor] = None,
        # 文本模型输出，可选
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        # 文本注意力掩码，可选
        text_attention_mask: Optional[torch.Tensor] = None,
        # 优先引导尺度，默认为4.0
        prior_guidance_scale: float = 4.0,
        # 解码器引导尺度，默认为8.0
        decoder_guidance_scale: float = 8.0,
        # 输出类型，默认为"pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式，默认为True
        return_dict: bool = True,
```