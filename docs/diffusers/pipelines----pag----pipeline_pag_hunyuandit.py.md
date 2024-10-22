# `.\diffusers\pipelines\pag\pipeline_pag_hunyuandit.py`

```py
# 版权声明，注明代码作者及版权信息
# Copyright 2024 HunyuanDiT Authors and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 授权该文件的使用
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 说明如何获取授权副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 说明在适用法律情况下的免责条款
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 参见许可证关于权限和限制的具体说明
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块以检查对象的签名和文档
import inspect
# 从 typing 模块导入类型提示
from typing import Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 用于数值计算
import numpy as np
# 导入 torch 用于深度学习操作
import torch
# 导入 transformers 中的相关模型和工具
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

# 从 diffusers 导入 StableDiffusionPipelineOutput，用于稳定扩散输出
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

# 导入回调函数相关的模块
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入图像处理器
from ...image_processor import VaeImageProcessor
# 导入模型定义
from ...models import AutoencoderKL, HunyuanDiT2DModel
# 导入注意力处理器
from ...models.attention_processor import PAGCFGHunyuanAttnProcessor2_0, PAGHunyuanAttnProcessor2_0
# 导入嵌入相关功能
from ...models.embeddings import get_2d_rotary_pos_embed
# 导入安全检查器
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 导入调度器
from ...schedulers import DDPMScheduler
# 导入实用工具
from ...utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
# 从工具模块导入随机生成张量的功能
from ...utils.torch_utils import randn_tensor
# 导入扩散管道工具
from ..pipeline_utils import DiffusionPipeline
# 导入 PAGMixin 模块
from .pag_utils import PAGMixin

# 检查是否可用 torch_xla 库，用于分布式训练
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 相关功能

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为 True
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标志为 False

# 设置日志记录器，获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义示例文档字符串，展示如何使用管道
EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipe = AutoPipelineForText2Image.from_pretrained(
        ...     "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        ...     torch_dtype=torch.float16,
        ...     enable_pag=True,
        ...     pag_applied_layers=[14],
        ... ).to("cuda")

        >>> # prompt = "an astronaut riding a horse"
        >>> prompt = "一个宇航员在骑马"
        >>> image = pipe(prompt, guidance_scale=4, pag_scale=3).images[0]
        ```py
"""

# 定义标准比例，包含多种常见的宽高比
STANDARD_RATIO = np.array(
    [
        1.0,  # 1:1
        4.0 / 3.0,  # 4:3
        3.0 / 4.0,  # 3:4
        16.0 / 9.0,  # 16:9
        9.0 / 16.0,  # 9:16
    ]
)
# 定义标准形状，包含不同宽高比下的尺寸组合
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],  # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],  # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],  # 3:4
    [(1280, 768)],  # 16:9
    [(768, 1280)],  # 9:16
]
# 计算每种形状的面积，存储在标准面积列表中
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]
# 定义支持的形状，包含常见的图像尺寸
SUPPORTED_SHAPE = [
    (1024, 1024),
    (1280, 1280),  # 1:1
    (1024, 768),
    (1152, 864),
    (1280, 960),  # 4:3
    (768, 1024),
    # 定义一个宽度为 864，高度为 1152 的尺寸元组
    (864, 1152),
    # 定义一个宽度为 960，高度为 1280 的尺寸元组，比例为 3:4
    (960, 1280),  # 3:4
    # 定义一个宽度为 1280，高度为 768 的尺寸元组，比例为 16:9
    (1280, 768),  # 16:9
    # 定义一个宽度为 768，高度为 1280 的尺寸元组，比例为 9:16
    (768, 1280),  # 9:16
]


# 将目标宽度和高度映射到标准形状
def map_to_standard_shapes(target_width, target_height):
    # 计算目标的宽高比
    target_ratio = target_width / target_height
    # 找到与目标宽高比最接近的标准宽高比的索引
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    # 找到与目标面积最接近的标准面积的索引
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    # 根据索引获取对应的标准宽度和高度
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    # 返回标准宽度和高度
    return width, height


# 根据源图像尺寸和目标尺寸计算裁剪区域
def get_resize_crop_region_for_grid(src, tgt_size):
    # 目标尺寸的高度和宽度
    th = tw = tgt_size
    # 源图像的高度和宽度
    h, w = src

    # 计算源图像的宽高比
    r = h / w

    # 根据宽高比决定如何调整尺寸
    # 如果高度大于宽度，按高度调整
    if r > 1:
        resize_height = th  # 设置调整后的高度为目标高度
        resize_width = int(round(th / h * w))  # 根据比例计算调整后的宽度
    else:
        resize_width = tw  # 设置调整后的宽度为目标宽度
        resize_height = int(round(tw / w * h))  # 根据比例计算调整后的高度

    # 计算裁剪区域的上边和左边
    crop_top = int(round((th - resize_height) / 2.0))  # 计算裁剪区域的上边界
    crop_left = int(round((tw - resize_width) / 2.0))  # 计算裁剪区域的左边界

    # 返回裁剪区域的坐标
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制而来
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 guidance_rescale 调整 `noise_cfg` 的尺度。基于文献 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的研究。见第 3.4 节。
    """
    # 计算噪声预测文本的标准差
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据文本标准差和配置标准差调整噪声预测结果（修复过度曝光）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 将调整后的噪声与原始噪声按指导比例混合，以避免图像“平淡”
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回调整后的噪声配置
    return noise_cfg


class HunyuanDiTPAGPipeline(DiffusionPipeline, PAGMixin):
    r"""
    使用 HunyuanDiT 和 [Perturbed Attention
    Guidance](https://huggingface.co/docs/diffusers/en/using-diffusers/pag) 的中英图像生成管道。

    此模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等），请查看父类文档。

    HunyuanDiT 使用两个文本编码器：[mT5](https://huggingface.co/google/mt5-base) 和 [双语 CLIP](由我们自行微调)。
    # 参数说明
        Args:
            vae ([`AutoencoderKL`]):
                变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。我们使用
                `sdxl-vae-fp16-fix`。
            text_encoder (Optional[`~transformers.BertModel`, `~transformers.CLIPTextModel`]):
                冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
                HunyuanDiT 使用经过微调的 [双语 CLIP]。
            tokenizer (Optional[`~transformers.BertTokenizer`, `~transformers.CLIPTokenizer`]):
                用于分词的 `BertTokenizer` 或 `CLIPTokenizer`。
            transformer ([`HunyuanDiT2DModel`]):
                腾讯 Hunyuan 设计的 HunyuanDiT 模型。
            text_encoder_2 (`T5EncoderModel`):
                mT5 嵌入器，具体为 't5-v1_1-xxl'。
            tokenizer_2 (`MT5Tokenizer`):
                mT5 嵌入器的分词器。
            scheduler ([`DDPMScheduler`]):
                与 HunyuanDiT 结合使用的调度器，用于对编码的图像潜在进行去噪。
        """
    
        # 模型的 CPU 卸载顺序
        model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
        # 可选组件列表
        _optional_components = [
            "safety_checker",
            "feature_extractor",
            "text_encoder_2",
            "tokenizer_2",
            "text_encoder",
            "tokenizer",
        ]
        # 从 CPU 卸载中排除的组件
        _exclude_from_cpu_offload = ["safety_checker"]
        # 回调张量输入列表
        _callback_tensor_inputs = [
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "prompt_embeds_2",
            "negative_prompt_embeds_2",
        ]
    
        # 初始化函数定义
        def __init__(
            self,
            vae: AutoencoderKL,  # 传入的 VAE 模型
            text_encoder: BertModel,  # 文本编码器
            tokenizer: BertTokenizer,  # 文本分词器
            transformer: HunyuanDiT2DModel,  # HunyuanDiT 模型
            scheduler: DDPMScheduler,  # 调度器
            safety_checker: Optional[StableDiffusionSafetyChecker] = None,  # 可选的安全检查器
            feature_extractor: Optional[CLIPImageProcessor] = None,  # 可选的特征提取器
            requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
            text_encoder_2: Optional[T5EncoderModel] = None,  # 可选的第二文本编码器
            tokenizer_2: Optional[MT5Tokenizer] = None,  # 可选的第二分词器
            pag_applied_layers: Union[str, List[str]] = "blocks.1",  # 应用层的字符串或列表
    # 初始化父类构造函数
        ):
            super().__init__()
    
            # 注册各个模块到当前对象中
            self.register_modules(
                vae=vae,  # 注册变分自编码器
                text_encoder=text_encoder,  # 注册文本编码器
                tokenizer=tokenizer,  # 注册分词器
                tokenizer_2=tokenizer_2,  # 注册第二个分词器
                transformer=transformer,  # 注册变换器
                scheduler=scheduler,  # 注册调度器
                safety_checker=safety_checker,  # 注册安全检查器
                feature_extractor=feature_extractor,  # 注册特征提取器
                text_encoder_2=text_encoder_2,  # 注册第二个文本编码器
            )
    
            # 如果安全检查器为 None 且需要安全检查器，则发出警告
            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 如果安全检查器不为 None 但特征提取器为 None，则引发错误
            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 设置 VAE 的缩放因子，若 VAE 存在则取其配置中的通道数
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
            # 创建图像处理器实例
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 将是否需要安全检查器的配置注册到当前对象
            self.register_to_config(requires_safety_checker=requires_safety_checker)
            # 设置默认的样本大小，若变换器存在则使用其配置中的样本大小
            self.default_sample_size = (
                self.transformer.config.sample_size
                if hasattr(self, "transformer") and self.transformer is not None
                else 128
            )
    
            # 设置应用的层及其注意力处理器
            self.set_pag_applied_layers(
                pag_applied_layers, pag_attn_processors=(PAGCFGHunyuanAttnProcessor2_0(), PAGHunyuanAttnProcessor2_0())
            )
    
        # 从 diffusers.pipelines.hunyuandit.pipeline_hunyuandit.HunyuanDiTPipeline 复制的 encode_prompt 方法
        def encode_prompt(
            self,
            prompt: str,  # 要编码的提示文本
            device: torch.device = None,  # 设备类型（如 CPU 或 GPU）
            dtype: torch.dtype = None,  # 数据类型
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool = True,  # 是否执行无分类器引导
            negative_prompt: Optional[str] = None,  # 负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 提示文本的嵌入表示
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示的嵌入表示
            prompt_attention_mask: Optional[torch.Tensor] = None,  # 提示的注意力掩码
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # 负面提示的注意力掩码
            max_sequence_length: Optional[int] = None,  # 最大序列长度
            text_encoder_index: int = 0,  # 文本编码器的索引
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的代码
    def run_safety_checker(self, image, device, dtype):
        # 检查安全检查器是否存在
        if self.safety_checker is None:
            # 如果不存在，设置无 NSFW 概念标志为 None
            has_nsfw_concept = None
        else:
            # 如果输入是张量格式
            if torch.is_tensor(image):
                # 将图像处理后转为 PIL 格式
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入不是张量，则将其转为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理图像并将其移动到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器处理图像并获取是否包含 NSFW 概念的标志
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像及 NSFW 概念标志
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的代码
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器都有相同的参数签名
        # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略。
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 值应在 [0, 1] 范围内

        # 检查调度器的步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果接受 eta，添加到额外参数字典中
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受 generator，添加到额外参数字典中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.hunyuandit.pipeline_hunyuandit.HunyuanDiTPipeline.check_inputs 复制的代码
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        prompt_embeds_2=None,
        negative_prompt_embeds_2=None,
        prompt_attention_mask_2=None,
        negative_prompt_attention_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    # 准备潜在变量，用于模型的生成过程
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，基于输入参数
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器列表的长度是否与批处理大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果没有提供潜在变量，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将现有潜在变量转移到指定设备
            latents = latents.to(device)
    
        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents
    
    # 返回引导缩放的属性值
    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    # 返回引导重缩放的属性值
    @property
    def guidance_rescale(self):
        return self._guidance_rescale
    
    # 根据Imagen论文定义的分类器自由引导的标志
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1
    
    # 返回时间步数的属性值
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    # 返回中断状态的属性值
    @property
    def interrupt(self):
        return self._interrupt
    
    # 关闭梯度计算，优化性能
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的方法，接收多个参数以生成图像
        def __call__(
            self,
            # 提示文本，可以是字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 图像的高度，默认为 None
            height: Optional[int] = None,
            # 图像的宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: Optional[int] = 50,
            # 指导缩放比例，默认为 5.0
            guidance_scale: Optional[float] = 5.0,
            # 负提示文本，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 额外的随机性控制，默认为 0.0
            eta: Optional[float] = 0.0,
            # 随机数生成器，可以是单个或多个
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，可以是一个张量
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，可以是一个张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 第二组提示嵌入，可以是一个张量
            prompt_embeds_2: Optional[torch.Tensor] = None,
            # 负提示嵌入，可以是一个张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 第二组负提示嵌入，可以是一个张量
            negative_prompt_embeds_2: Optional[torch.Tensor] = None,
            # 提示的注意力掩码，可以是一个张量
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 第二组提示的注意力掩码，可以是一个张量
            prompt_attention_mask_2: Optional[torch.Tensor] = None,
            # 负提示的注意力掩码，可以是一个张量
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 第二组负提示的注意力掩码，可以是一个张量
            negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 步骤结束时的回调函数
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 回调时输入的张量名称列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 指导重标定值，默认为 0.0
            guidance_rescale: float = 0.0,
            # 原始图像大小，默认为 (1024, 1024)
            original_size: Optional[Tuple[int, int]] = (1024, 1024),
            # 目标图像大小，默认为 None
            target_size: Optional[Tuple[int, int]] = None,
            # 裁剪区域的左上角坐标，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 是否使用分辨率分箱，默认为 True
            use_resolution_binning: bool = True,
            # 页面缩放比例，默认为 3.0
            pag_scale: float = 3.0,
            # 自适应页面缩放比例，默认为 0.0
            pag_adaptive_scale: float = 0.0,
```