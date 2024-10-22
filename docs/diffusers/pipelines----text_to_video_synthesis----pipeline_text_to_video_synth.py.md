# `.\diffusers\pipelines\text_to_video_synthesis\pipeline_text_to_video_synth.py`

```py
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件受 Apache 2.0 许可证保护，使用需遵循该许可证
# you may not use this file except in compliance with the License.
# 只有在遵循许可证的情况下，才能使用此文件
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 可在此处获取许可证的副本
#
# Unless required by applicable law or agreed to in writing, software
# 除非法律要求或书面协议，软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 按 "原样" 方式分发，不提供任何保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何明示或暗示的保证或条件
# See the License for the specific language governing permissions and
# 查看许可证以了解特定权限和
# limitations under the License.
# 限制条件

import inspect  # 导入 inspect 模块，用于获取对象的成员信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注释相关的类

import torch  # 导入 PyTorch 库，进行张量操作和深度学习
from transformers import CLIPTextModel, CLIPTokenizer  # 从 transformers 库导入 CLIP 模型和分词器

from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器混合类
from ...models import AutoencoderKL, UNet3DConditionModel  # 导入模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 LoRA 权重的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器类
from ...utils import (  # 导入工具函数和常量
    USE_PEFT_BACKEND,  # 表示是否使用 PEFT 后端的常量
    deprecate,  # 用于标记过时的功能
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 用于替换示例文档字符串的函数
    scale_lora_layers,  # 用于缩放 LoRA 层的函数
    unscale_lora_layers,  # 用于还原 LoRA 层的函数
)
from ...utils.torch_utils import randn_tensor  # 导入用于生成随机张量的函数
from ...video_processor import VideoProcessor  # 导入视频处理器类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类
from . import TextToVideoSDPipelineOutput  # 导入文本到视频生成管道的输出类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于记录日志信息

EXAMPLE_DOC_STRING = """  # 定义一个示例文档字符串，提供使用示例
    Examples:
        ```py  # Python 代码块标记
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import TextToVideoSDPipeline  # 从 diffusers 库导入文本到视频管道
        >>> from diffusers.utils import export_to_video  # 导入视频导出工具

        >>> pipe = TextToVideoSDPipeline.from_pretrained(  # 从预训练模型加载管道
        ...     "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"  # 指定模型名称和数据类型
        ... )
        >>> pipe.enable_model_cpu_offload()  # 启用模型的 CPU 内存卸载

        >>> prompt = "Spiderman is surfing"  # 定义生成视频的提示语
        >>> video_frames = pipe(prompt).frames[0]  # 生成视频帧
        >>> video_path = export_to_video(video_frames)  # 导出生成的视频帧为视频文件
        >>> video_path  # 输出视频文件路径
        ```py
"""


class TextToVideoSDPipeline(  # 定义文本到视频生成管道类
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin  # 继承多个混合类
):
    r"""  # 开始类的文档字符串
    Pipeline for text-to-video generation.  # 文档说明：用于文本到视频生成的管道

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    该模型继承自 `DiffusionPipeline`。查阅超类文档以获取通用方法的说明
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).
    实现了所有管道的通用方法（下载、保存、在特定设备上运行等）

    The pipeline also inherits the following loading methods:
    该管道还继承了以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
    # 文档字符串，描述构造函数的参数及其用途
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器模型，用于将图像编码和解码为潜在表示
        text_encoder ([`CLIPTextModel`]):
            # 冻结的文本编码器模型，用于处理文本数据
        tokenizer (`CLIPTokenizer`):
            # 用于将文本标记化的 CLIP 标记器
        unet ([`UNet3DConditionModel`]):
            # UNet 模型，用于对编码的视频潜在空间进行去噪
        scheduler ([`SchedulerMixin`]):
            # 与 UNet 结合使用的调度器，用于去噪编码的图像潜在空间，可以是多种调度器之一
    """

    # 定义模型的计算顺序，依次为文本编码器、UNet 和 VAE
    model_cpu_offload_seq = "text_encoder->unet->vae"

    # 初始化函数，接受多个模型参数
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模型模块
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化视频处理器，不进行缩放
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)

    # 定义私有方法，用于编码输入的提示
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        # 定义弃用消息，提醒用户此方法将在未来版本中删除
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用弃用警告函数
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用编码提示的方法，获取提示的嵌入元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # 连接嵌入以用于向后兼容
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回最终的提示嵌入
        return prompt_embeds

    # 从稳定扩散管道中复制的方法，编码提示
    # 编码提示信息和相关参数以生成图像
    def encode_prompt(
            self,
            prompt,  # 用户输入的提示文本
            device,  # 设备类型（如CPU或GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的LoRA缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪切跳过参数
        # 解码潜在空间中的数据以生成图像
        def decode_latents(self, latents):
            # 使用缩放因子调整潜在数据
            latents = 1 / self.vae.config.scaling_factor * latents
    
            # 获取批次大小、通道数、帧数、高度和宽度
            batch_size, channels, num_frames, height, width = latents.shape
            # 重新排列潜在数据的维度，以适应解码器输入
            latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    
            # 解码潜在数据以生成图像
            image = self.vae.decode(latents).sample
            # 重塑图像以形成视频格式
            video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
            # 转换数据类型为float32，以确保与bfloat16兼容
            video = video.float()
            # 返回生成的视频数据
            return video
    
        # 准备额外的步骤参数，以供调度器使用
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外的关键字参数
            # eta (η) 仅在DDIMScheduler中使用，其他调度器将忽略
            # eta对应于DDIM论文中的η，取值范围应在[0, 1]之间
    
            # 检查调度器是否接受eta参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            # 如果接受eta，添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受generator参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受generator，添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    
        # 检查输入参数的有效性
        def check_inputs(
            self,
            prompt,  # 用户输入的提示文本
            height,  # 生成图像的高度
            width,  # 生成图像的宽度
            callback_steps,  # 回调步骤的数量
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
            callback_on_step_end_tensor_inputs=None,  # 可选的回调张量输入
    # 结束前面的代码块
        ):
            # 检查高度和宽度是否能被8整除，若不能则抛出异常
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    
            # 检查回调步数是否为正整数，若不是则抛出异常
            if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
            # 检查回调结束时的张量输入是否都在指定的输入中，若不在则抛出异常
            if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
            ):
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
                )
    
            # 检查是否同时提供了prompt和prompt_embeds，若是则抛出异常
            if prompt is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查是否都没有提供prompt和prompt_embeds，若是则抛出异常
            elif prompt is None and prompt_embeds is None:
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查prompt的类型是否为字符串或列表，若不是则抛出异常
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查是否同时提供了negative_prompt和negative_prompt_embeds，若是则抛出异常
            if negative_prompt is not None and negative_prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查prompt_embeds和negative_prompt_embeds的形状是否一致，若不一致则抛出异常
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        # 定义准备潜变量的方法，接受多个参数
        def prepare_latents(
            self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 定义形状元组，包括批次大小、通道数、帧数、高度和宽度的缩放
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且其长度与批次大小不匹配，若不匹配则抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果潜在变量为空，则生成新的随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将给定的潜在变量转移到指定设备
            latents = latents.to(device)

        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 禁用梯度计算以提高性能
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 定义输入参数，包括提示、图像尺寸、帧数等
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
```