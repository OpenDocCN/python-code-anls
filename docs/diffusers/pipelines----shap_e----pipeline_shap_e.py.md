# `.\diffusers\pipelines\shap_e\pipeline_shap_e.py`

```py
# 版权信息，标明该文件的版权归属
# Copyright 2024 Open AI and The HuggingFace Team. All rights reserved.
#
# 按照 Apache License, Version 2.0 进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件只能在遵守许可的情况下使用
# you may not use this file except in compliance with the License.
# 可以通过以下链接获取许可副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，软件按“原样”提供，不提供任何明示或暗示的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可文件以了解特定权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数学模块
import math
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入类型提示相关的类型
from typing import List, Optional, Union

# 导入 numpy 库并命名为 np
import numpy as np
# 导入图像处理库 PIL 的 Image 模块
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIPTextModelWithProjection 和 CLIPTokenizer
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# 从本地模型模块导入 PriorTransformer 类
from ...models import PriorTransformer
# 从调度器模块导入 HeunDiscreteScheduler 类
from ...schedulers import HeunDiscreteScheduler
# 从工具模块导入多个工具类和函数
from ...utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
# 从工具的 torch_utils 模块导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入 DiffusionPipeline 类
from ..pipeline_utils import DiffusionPipeline
# 从渲染器模块导入 ShapERenderer 类
from .renderer import ShapERenderer

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该管道
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from diffusers.utils import export_to_gif

        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        >>> repo = "openai/shap-e"
        >>> pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
        >>> pipe = pipe.to(device)

        >>> guidance_scale = 15.0
        >>> prompt = "a shark"

        >>> images = pipe(
        ...     prompt,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=64,
        ...     frame_size=256,
        ... ).images

        >>> gif_path = export_to_gif(images[0], "shark_3d.gif")
        ```py
"""

# 定义 ShapEPipelineOutput 数据类，继承自 BaseOutput
@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    ShapEPipeline 和 ShapEImg2ImgPipeline 的输出类。

    参数:
        images (`torch.Tensor`)
            生成的 3D 渲染图像列表。
    """

    # 声明一个属性，表示图像可以是多种格式的列表
    images: Union[List[List[PIL.Image.Image]], List[List[np.ndarray]]]

# 定义 ShapEPipeline 类，继承自 DiffusionPipeline
class ShapEPipeline(DiffusionPipeline):
    """
    用于生成 3D 资产的潜在表示并使用 NeRF 方法进行渲染的管道。

    该模型继承自 DiffusionPipeline。请查看超类文档以获取所有管道实现的通用方法
    (下载、保存、在特定设备上运行等)。
    # 文档字符串，描述构造函数的参数及其类型
    Args:
        prior ([`PriorTransformer`]):
            用于近似文本嵌入生成图像嵌入的标准 unCLIP 先验。
        text_encoder ([`~transformers.CLIPTextModelWithProjection`]):
            冻结的文本编码器。
        tokenizer ([`~transformers.CLIPTokenizer`]):
             用于对文本进行分词的 `CLIPTokenizer`。
        scheduler ([`HeunDiscreteScheduler`]):
            用于与 `prior` 模型结合生成图像嵌入的调度器。
        shap_e_renderer ([`ShapERenderer`]):
            Shap-E 渲染器将生成的潜在向量投影到 MLP 的参数中，以使用 NeRF 渲染方法创建 3D 对象。
    """

    # 定义 CPU 卸载顺序，指定先卸载 text_encoder 后卸载 prior
    model_cpu_offload_seq = "text_encoder->prior"
    # 指定不进行 CPU 卸载的模块列表
    _exclude_from_cpu_offload = ["shap_e_renderer"]

    # 初始化方法，接收多个参数用于设置对象状态
    def __init__(
        self,
        prior: PriorTransformer,  # 先验模型
        text_encoder: CLIPTextModelWithProjection,  # 文本编码器
        tokenizer: CLIPTokenizer,  # 文本分词器
        scheduler: HeunDiscreteScheduler,  # 调度器
        shap_e_renderer: ShapERenderer,  # Shap-E 渲染器
    ):
        super().__init__()  # 调用父类的初始化方法

        # 注册各个模块，将其绑定到当前实例
        self.register_modules(
            prior=prior,  # 注册先验模型
            text_encoder=text_encoder,  # 注册文本编码器
            tokenizer=tokenizer,  # 注册文本分词器
            scheduler=scheduler,  # 注册调度器
            shap_e_renderer=shap_e_renderer,  # 注册 Shap-E 渲染器
        )

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline 复制的方法，准备潜在向量
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果未提供潜在向量，则生成随机的潜在向量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供的潜在向量形状不符合预期，则抛出异常
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在向量移动到指定设备
            latents = latents.to(device)

        # 将潜在向量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在向量
        return latents

    # 编码提示文本的方法，接收多个参数用于配置
    def _encode_prompt(
        self,
        prompt,  # 提示文本
        device,  # 指定设备
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否进行无分类器引导
    # 定义一个方法，处理输入的提示文本
        ):
            # 判断 prompt 是否为列表，如果是，返回其长度，否则返回 1
            len(prompt) if isinstance(prompt, list) else 1
    
            # YiYi 注释: 将 pad_token_id 设置为 0，不确定为何无法在配置文件中设置
            self.tokenizer.pad_token_id = 0
            # 获取提示文本的嵌入表示
            text_inputs = self.tokenizer(
                prompt,
                # 在处理时填充到最大长度
                padding="max_length",
                # 设置最大长度为 tokenizer 的最大模型长度
                max_length=self.tokenizer.model_max_length,
                # 如果文本超长，进行截断
                truncation=True,
                # 返回张量格式，类型为 PyTorch
                return_tensors="pt",
            )
            # 获取文本输入的 ID
            text_input_ids = text_inputs.input_ids
            # 获取未截断的 ID，填充方式为最长
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    
            # 检查未截断 ID 的形状是否大于等于文本输入 ID 的形状，并确保它们不相等
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码并记录被截断的文本部分
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "以下输入部分被截断，因为 CLIP 只能处理最长"
                    f" {self.tokenizer.model_max_length} 个 tokens: {removed_text}"
                )
    
            # 将文本输入 ID 转移到设备上并获取编码输出
            text_encoder_output = self.text_encoder(text_input_ids.to(device))
            # 获取文本嵌入
            prompt_embeds = text_encoder_output.text_embeds
    
            # 根据每个提示文本的图像数量重复嵌入
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 在 Shap-E 中，先对 prompt_embeds 进行归一化，然后再重新缩放
            prompt_embeds = prompt_embeds / torch.linalg.norm(prompt_embeds, dim=-1, keepdim=True)
    
            # 如果需要分类自由引导
            if do_classifier_free_guidance:
                # 创建与 prompt_embeds 形状相同的零张量
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    
                # 为了分类自由引导，需要进行两次前向传递
                # 将无条件嵌入和文本嵌入连接到一个批次中，以避免进行两次前向传递
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    
            # 将特征重新缩放为单位方差
            prompt_embeds = math.sqrt(prompt_embeds.shape[1]) * prompt_embeds
    
            # 返回最终的提示嵌入
            return prompt_embeds
    
        # 装饰器，禁用梯度计算
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义调用方法，处理输入参数
        def __call__(
            # 提示文本
            prompt: str,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 推理步骤数量，默认为 25
            num_inference_steps: int = 25,
            # 随机数生成器，可选
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在张量，可选
            latents: Optional[torch.Tensor] = None,
            # 指导比例，默认为 4.0
            guidance_scale: float = 4.0,
            # 帧大小，默认为 64
            frame_size: int = 64,
            # 输出类型，可选，默认为 'pil'
            output_type: Optional[str] = "pil",  # pil, np, latent, mesh
            # 是否返回字典格式，默认为 True
            return_dict: bool = True,
```