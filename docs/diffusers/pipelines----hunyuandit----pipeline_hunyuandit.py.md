# `.\diffusers\pipelines\hunyuandit\pipeline_hunyuandit.py`

```py
# 版权声明，标明文件归属和使用许可信息
# Copyright 2024 HunyuanDiT Authors and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License 2.0 许可使用该文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在遵守许可的情况下使用此文件
# You may not use this file except in compliance with the License.
# 可以在以下网址获取许可副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件以“按现状”提供，不提供任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可以了解权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块以便于代码检查和获取信息
import inspect
# 从 typing 模块导入所需的类型注解
from typing import Callable, Dict, List, Optional, Tuple, Union

# 导入 NumPy 库以进行数值计算
import numpy as np
# 导入 PyTorch 库以进行深度学习模型的构建和训练
import torch
# 从 transformers 库导入所需的模型和分词器
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

# 从 diffusers 库导入稳定扩散管道输出类型
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

# 从回调模块导入多管道回调和单管道回调
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 从图像处理模块导入 VAE 图像处理器
from ...image_processor import VaeImageProcessor
# 从模型模块导入自编码器和 HunyuanDiT2D 模型
from ...models import AutoencoderKL, HunyuanDiT2DModel
# 从嵌入模块导入二维旋转位置嵌入的获取函数
from ...models.embeddings import get_2d_rotary_pos_embed
# 从安全检查器模块导入稳定扩散安全检查器
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 从调度器模块导入 DDPMScheduler
from ...schedulers import DDPMScheduler
# 从工具模块导入各种工具函数
from ...utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
# 从 PyTorch 工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入 DiffusionPipeline
from ..pipeline_utils import DiffusionPipeline

# 检查是否可用 Torch XLA 库以支持 TPU 加速
if is_torch_xla_available():
    # 导入 XLA 模块以进行 TPU 操作
    import torch_xla.core.xla_model as xm

    # 标记 XLA 可用性为 True
    XLA_AVAILABLE = True
else:
    # 标记 XLA 可用性为 False
    XLA_AVAILABLE = False

# 获取日志记录器以进行调试和日志记录
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用 HunyuanDiTPipeline
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import HunyuanDiTPipeline

        >>> pipe = HunyuanDiTPipeline.from_pretrained(
        ...     "Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> # 也可以使用英语提示，因为 HunyuanDiT 支持英语和中文
        >>> # prompt = "An astronaut riding a horse"
        >>> prompt = "一个宇航员在骑马"
        >>> image = pipe(prompt).images[0]
        ```py
"""

# 定义标准宽高比数组
STANDARD_RATIO = np.array(
    [
        1.0,  # 1:1
        4.0 / 3.0,  # 4:3
        3.0 / 4.0,  # 3:4
        16.0 / 9.0,  # 16:9
        9.0 / 16.0,  # 9:16
    ]
)
# 定义标准形状的列表，包括不同宽高比的分辨率
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],  # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],  # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],  # 3:4
    [(1280, 768)],  # 16:9
    [(768, 1280)],  # 9:16
]
# 根据标准形状计算面积
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]
# 定义支持的形状，列出可能的输入分辨率
SUPPORTED_SHAPE = [
    (1024, 1024),
    (1280, 1280),  # 1:1
    (1024, 768),
    (1152, 864),
    (1280, 960),  # 4:3
    (768, 1024),
    (864, 1152),
    (960, 1280),  # 3:4
    (1280, 768),  # 16:9
    (768, 1280),  # 9:16
]

# 定义一个函数，将目标宽高映射到标准形状
def map_to_standard_shapes(target_width, target_height):
    # 计算目标宽高比
        target_ratio = target_width / target_height
        # 找到与目标宽高比最接近的标准宽高比的索引
        closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
        # 找到与目标面积最接近的标准面积的索引
        closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
        # 根据最接近的宽高比和面积索引获取标准形状的宽度和高度
        width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
        # 返回找到的宽度和高度
        return width, height
# 定义获取网格的调整大小和裁剪区域的函数，接受源图像尺寸和目标尺寸
def get_resize_crop_region_for_grid(src, tgt_size):
    # 将目标尺寸赋值给高度和宽度变量
    th = tw = tgt_size
    # 解构源图像尺寸，获取高度和宽度
    h, w = src

    # 计算宽高比
    r = h / w

    # 根据宽高比决定调整大小的方式
    # 如果高度大于宽度
    if r > 1:
        # 将目标高度设为目标尺寸
        resize_height = th
        # 根据目标高度计算目标宽度
        resize_width = int(round(th / h * w))
    else:
        # 否则将目标宽度设为目标尺寸
        resize_width = tw
        # 根据目标宽度计算目标高度
        resize_height = int(round(tw / w * h))

    # 计算裁剪区域的上边界
    crop_top = int(round((th - resize_height) / 2.0))
    # 计算裁剪区域的左边界
    crop_left = int(round((tw - resize_width) / 2.0))

    # 返回裁剪区域的坐标和调整后的尺寸
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制的函数
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 重新调整 `noise_cfg` 的比例。基于论文 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的发现。见第 3.4 节
    """
    # 计算噪声预测文本的标准差
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 通过标准差调整噪声预测以修正过曝问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 根据引导重缩放原始结果，以避免图像过于单调
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回调整后的噪声配置
    return noise_cfg


# 定义 HunyuanDiT 管道类，继承自 DiffusionPipeline
class HunyuanDiTPipeline(DiffusionPipeline):
    r"""
    使用 HunyuanDiT 进行英语/中文到图像生成的管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取库为所有管道实现的通用方法（例如下载或保存，运行在特定设备上等）。

    HunyuanDiT 使用两个文本编码器：[mT5](https://huggingface.co/google/mt5-base) 和 [双语 CLIP](fine-tuned by
    ourselves)

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。我们使用 `sdxl-vae-fp16-fix`。
        text_encoder (Optional[`~transformers.BertModel`, `~transformers.CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
            HunyuanDiT 使用经过微调的 [双语 CLIP]。
        tokenizer (Optional[`~transformers.BertTokenizer`, `~transformers.CLIPTokenizer`]):
            用于文本标记化的 `BertTokenizer` 或 `CLIPTokenizer`。
        transformer ([`HunyuanDiT2DModel`]):
            腾讯 Hunyuan 设计的 HunyuanDiT 模型。
        text_encoder_2 (`T5EncoderModel`):
            mT5 嵌入器。具体为 't5-v1_1-xxl'。
        tokenizer_2 (`MT5Tokenizer`):
            mT5 嵌入器的标记器。
        scheduler ([`DDPMScheduler`]):
            用于与 HunyuanDiT 结合使用的调度器，以去噪编码的图像潜在。
    """
    # 定义模型的 CPU 卸载序列，指定组件的调用顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    # 可选组件的列表，包含模型中可能使用的其他模块
    _optional_components = [
        "safety_checker",  # 安全检查器
        "feature_extractor",  # 特征提取器
        "text_encoder_2",  # 第二个文本编码器
        "tokenizer_2",  # 第二个分词器
        "text_encoder",  # 第一个文本编码器
        "tokenizer",  # 第一个分词器
    ]
    # 指定不参与 CPU 卸载的组件
    _exclude_from_cpu_offload = ["safety_checker"]  # 安全检查器不参与 CPU 卸载
    # 回调张量输入列表，包含输入模型的张量名称
    _callback_tensor_inputs = [
        "latents",  # 潜在变量
        "prompt_embeds",  # 提示嵌入
        "negative_prompt_embeds",  # 负提示嵌入
        "prompt_embeds_2",  # 第二个提示嵌入
        "negative_prompt_embeds_2",  # 第二个负提示嵌入
    ]

    # 初始化方法，设置模型的各个组件
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器
        text_encoder: BertModel,  # 文本编码器，使用 BERT 模型
        tokenizer: BertTokenizer,  # 分词器，使用 BERT 分词器
        transformer: HunyuanDiT2DModel,  # 转换器模型
        scheduler: DDPMScheduler,  # 调度器
        safety_checker: StableDiffusionSafetyChecker,  # 稳定扩散安全检查器
        feature_extractor: CLIPImageProcessor,  # 特征提取器，使用 CLIP 图像处理器
        requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
        text_encoder_2=T5EncoderModel,  # 第二个文本编码器，默认使用 T5 模型
        tokenizer_2=MT5Tokenizer,  # 第二个分词器，默认使用 MT5 分词器
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模型的各个组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            text_encoder_2=text_encoder_2,
        )

        # 检查安全检查器是否为 None，并根据需要发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器和特征提取器的配置，确保正确设置
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 根据 VAE 的配置计算缩放因子
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # 初始化图像处理器
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 注册配置，保存是否需要安全检查器的标志
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        # 根据 transformer 的配置确定默认样本大小
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128  # 默认样本大小为 128
        )
    # 定义一个编码提示的函数，包含多个参数配置
        def encode_prompt(
            self,
            # 输入的提示文本
            prompt: str,
            # 设备类型，可选
            device: torch.device = None,
            # 数据类型，可选
            dtype: torch.dtype = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: int = 1,
            # 是否进行分类器自由引导，默认为True
            do_classifier_free_guidance: bool = True,
            # 负提示文本，默认为None
            negative_prompt: Optional[str] = None,
            # 提示的嵌入表示，默认为None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的嵌入表示，默认为None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 提示的注意力掩码，默认为None
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 负提示的注意力掩码，默认为None
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 最大序列长度，默认为None
            max_sequence_length: Optional[int] = None,
            # 文本编码器索引，默认为0
            text_encoder_index: int = 0,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的函数
        def run_safety_checker(self, image, device, dtype):
            # 检查安全检查器是否存在
            if self.safety_checker is None:
                # 如果不存在，设置nsfw概念为None
                has_nsfw_concept = None
            else:
                # 如果输入是张量，则进行后处理
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果不是张量，则转换为PIL格式
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 获取安全检查器的输入并移动到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器，检查图像的nsfw概念
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和nsfw概念
            return image, has_nsfw_concept
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的函数
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外的关键字参数，因为并非所有调度器具有相同的签名
            # eta (η) 仅在DDIMScheduler中使用，其他调度器将被忽略
            # eta对应于DDIM论文中的η: https://arxiv.org/abs/2010.02502
            # 应在[0, 1]之间
    
            # 检查调度器是否接受eta参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                # 如果接受eta，则将其添加到额外参数中
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受generator参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            if accepts_generator:
                # 如果接受generator，则将其添加到额外参数中
                extra_step_kwargs["generator"] = generator
            # 返回额外的关键字参数字典
            return extra_step_kwargs
    
        # 定义一个检查输入的函数，包含多个参数
        def check_inputs(
            self,
            # 输入的提示文本
            prompt,
            # 图像高度
            height,
            # 图像宽度
            width,
            # 负提示文本，默认为None
            negative_prompt=None,
            # 提示的嵌入表示，默认为None
            prompt_embeds=None,
            # 负提示的嵌入表示，默认为None
            negative_prompt_embeds=None,
            # 提示的注意力掩码，默认为None
            prompt_attention_mask=None,
            # 负提示的注意力掩码，默认为None
            negative_prompt_attention_mask=None,
            # 第二组提示的嵌入表示，默认为None
            prompt_embeds_2=None,
            # 第二组负提示的嵌入表示，默认为None
            negative_prompt_embeds_2=None,
            # 第二组提示的注意力掩码，默认为None
            prompt_attention_mask_2=None,
            # 第二组负提示的注意力掩码，默认为None
            negative_prompt_attention_mask_2=None,
            # 用于步骤结束回调的张量输入，默认为None
            callback_on_step_end_tensor_inputs=None,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的函数
    # 准备潜在变量，定义输入形状
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，考虑缩放因子
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器列表长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    # 抛出错误，提示生成器长度与批量大小不符
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果潜在变量为空，则随机生成
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 否则将潜在变量移动到指定设备
                latents = latents.to(device)
    
            # 将初始噪声按调度器要求的标准差进行缩放
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 返回指导缩放因子
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 返回指导重缩放因子
        @property
        def guidance_rescale(self):
            return self._guidance_rescale
    
        # 定义无分类器引导的属性，依照Imagen论文中的公式定义
        @property
        def do_classifier_free_guidance(self):
            # 判断指导缩放因子是否大于1，决定是否进行无分类器引导
            return self._guidance_scale > 1
    
        # 返回时间步数
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 返回中断状态
        @property
        def interrupt(self):
            return self._interrupt
    
        # 禁用梯度计算，优化内存使用
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，允许使用不同的参数生成输出
    def __call__(
            self,
            # 提示内容，可以是单个字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 输出图像的高度
            height: Optional[int] = None,
            # 输出图像的宽度
            width: Optional[int] = None,
            # 推理步骤的数量，默认是50
            num_inference_steps: Optional[int] = 50,
            # 引导比例，默认值为5.0
            guidance_scale: Optional[float] = 5.0,
            # 负提示内容，可以是单个字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认是1
            num_images_per_prompt: Optional[int] = 1,
            # 影响生成随机性的超参数，默认是0.0
            eta: Optional[float] = 0.0,
            # 随机数生成器，可以是单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在空间的张量，表示生成过程中的潜在表示
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入表示
            prompt_embeds: Optional[torch.Tensor] = None,
            # 第二个提示的嵌入表示
            prompt_embeds_2: Optional[torch.Tensor] = None,
            # 负提示的嵌入表示
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 第二个负提示的嵌入表示
            negative_prompt_embeds_2: Optional[torch.Tensor] = None,
            # 提示的注意力掩码
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 第二个提示的注意力掩码
            prompt_attention_mask_2: Optional[torch.Tensor] = None,
            # 负提示的注意力掩码
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 第二个负提示的注意力掩码
            negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
            # 输出类型，默认为"pil"，表示图像格式
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的输出，默认为True
            return_dict: bool = True,
            # 生成步骤结束时的回调函数，可以是特定格式的函数
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 指定在步骤结束时的张量输入名称，默认为["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 引导重标定的比例，默认值为0.0
            guidance_rescale: float = 0.0,
            # 原始图像的尺寸，默认为(1024, 1024)
            original_size: Optional[Tuple[int, int]] = (1024, 1024),
            # 目标图像的尺寸，默认为None，表示使用原始尺寸
            target_size: Optional[Tuple[int, int]] = None,
            # 裁剪区域的左上角坐标，默认为(0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 是否使用分辨率分箱，默认为True
            use_resolution_binning: bool = True,
```