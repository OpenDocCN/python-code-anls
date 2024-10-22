# `.\diffusers\pipelines\cogvideo\pipeline_cogvideox_image2video.py`

```py
# 版权声明，说明文件的版权归属及使用许可
# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# 授权条款，说明使用该文件的条件
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 允许用户在遵守许可的情况下使用该文件
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非另有协议，否则软件以“原样”方式分发，不提供任何明示或暗示的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入所需的模块
import inspect  # 用于获取对象的活跃信息
import math  # 提供数学函数
from typing import Callable, Dict, List, Optional, Tuple, Union  # 类型提示的支持

# 导入图像处理库
import PIL  # 图像处理库
import torch  # 深度学习框架
from transformers import T5EncoderModel, T5Tokenizer  # 导入 T5 模型和分词器

# 导入自定义回调和处理器
from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 回调相关
from ...image_processor import PipelineImageInput  # 图像输入处理器
from ...models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel  # 模型定义
from ...models.embeddings import get_3d_rotary_pos_embed  # 获取 3D 旋转位置嵌入
from ...pipelines.pipeline_utils import DiffusionPipeline  # 扩散管道
from ...schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler  # 调度器
from ...utils import (
    logging,  # 日志工具
    replace_example_docstring,  # 替换示例文档字符串
)
from ...utils.torch_utils import randn_tensor  # 随机张量生成工具
from ...video_processor import VideoProcessor  # 视频处理器
from .pipeline_output import CogVideoXPipelineOutput  # 管道输出定义


# 创建日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# 示例文档字符串，提供使用示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import CogVideoXImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)  # 从预训练模型创建管道
        >>> pipe.to("cuda")  # 将管道移动到 GPU

        >>> prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."  # 定义生成视频的提示
        >>> image = load_image(  # 加载输入图像
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> video = pipe(image, prompt, use_dynamic_cfg=True)  # 生成视频
        >>> export_to_video(video.frames[0], "output.mp4", fps=8)  # 导出生成的视频
        ```py
"""


# 定义调整图像大小和裁剪区域的函数
# 类似于 diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width  # 目标宽度
    th = tgt_height  # 目标高度
    h, w = src  # 源图像的高度和宽度
    r = h / w  # 计算源图像的纵横比
    # 根据纵横比决定调整后的高度和宽度
    if r > (th / tw):  
        resize_height = th  # 设置调整后的高度为目标高度
        resize_width = int(round(th / h * w))  # 根据比例计算调整后的宽度
    else:
        resize_width = tw  # 设置调整后的宽度为目标宽度
        resize_height = int(round(tw / w * h))  # 根据比例计算调整后的高度

    # 计算裁剪区域的起始位置
    crop_top = int(round((th - resize_height) / 2.0))  # 上边裁剪位置
    crop_left = int(round((tw - resize_width) / 2.0))  # 左边裁剪位置

    # 返回裁剪区域的起始和结束坐标
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 复制而来
def retrieve_timesteps(
    # 调度器对象
    scheduler,
    # 推理步骤的数量（可选）
    num_inference_steps: Optional[int] = None,
    # 设备信息（可选）
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步（可选）
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 值（可选）
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器检索时间步。处理自定义时间步。任何关键字参数将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步应移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递了 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma，用于覆盖调度器的时间步间隔策略。如果传递了 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是调度器的时间步计划，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了自定义时间步和 sigma
    if timesteps is not None and sigmas is not None:
        raise ValueError("只能传递 `timesteps` 或 `sigmas` 中的一个。请选择一个设置自定义值")
    # 如果传递了自定义时间步
    if timesteps is not None:
        # 检查调度器是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，则抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义"
                f" 时间步计划。请检查您是否使用了正确的调度器。"
            )
        # 调用调度器的 `set_timesteps` 方法设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传递了自定义 sigma
    elif sigmas is not None:
        # 检查调度器是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，则抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义"
                f" sigma 计划。请检查您是否使用了正确的调度器。"
            )
        # 调用调度器的 `set_timesteps` 方法设置自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果不是前一个条件的情况，执行以下代码
        else:
            # 设置推理步骤数，并指定设备和其他关键字参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取当前调度器的时间步长
            timesteps = scheduler.timesteps
        # 返回时间步长和推理步骤数
        return timesteps, num_inference_steps
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的代码
def retrieve_latents(
    # 输入的编码器输出，类型为 torch.Tensor
    encoder_output: torch.Tensor, 
    # 可选的随机数生成器，用于采样
    generator: Optional[torch.Generator] = None, 
    # 采样模式，默认为 "sample"
    sample_mode: str = "sample"
):
    # 检查 encoder_output 是否有 latent_dist 属性且模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中采样并返回结果
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否有 latent_dist 属性且模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的模式
        return encoder_output.latent_dist.mode()
    # 检查 encoder_output 是否有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 encoder_output 中的 latents
        return encoder_output.latents
    # 如果以上条件都不满足，抛出 AttributeError
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class CogVideoXImageToVideoPipeline(DiffusionPipeline):
    r"""
    使用 CogVideoX 的图像到视频生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看父类文档以获取库实现的通用方法
    （例如下载或保存，运行在特定设备等）。

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器（VAE）模型，用于将视频编码和解码为潜在表示。
        text_encoder ([`T5EncoderModel`]):
            冻结的文本编码器。CogVideoX 使用
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)；特别是
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) 变体。
        tokenizer (`T5Tokenizer`):
            类的分词器
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer)。
        transformer ([`CogVideoXTransformer3DModel`]):
            一个文本条件的 `CogVideoXTransformer3DModel`，用于去噪编码的视频潜在。
        scheduler ([`SchedulerMixin`]):
            一个调度器，结合 `transformer` 用于去噪编码的视频潜在。
    """

    # 可选组件列表，初始化为空
    _optional_components = []
    # 指定 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 需要回调的张量输入列表
    _callback_tensor_inputs = [
        # 潜在张量
        "latents",
        # 提示嵌入
        "prompt_embeds",
        # 负面提示嵌入
        "negative_prompt_embeds",
    ]

    def __init__(
        # 初始化方法的参数：分词器
        self,
        tokenizer: T5Tokenizer,
        # 文本编码器
        text_encoder: T5EncoderModel,
        # VAE 模型
        vae: AutoencoderKLCogVideoX,
        # 变换模型
        transformer: CogVideoXTransformer3DModel,
        # 调度器
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        # 调用父类的构造函数以初始化基类部分
        super().__init__()

        # 注册各个模块，传入相关参数
        self.register_modules(
            tokenizer=tokenizer,  # 注册分词器
            text_encoder=text_encoder,  # 注册文本编码器
            vae=vae,  # 注册变分自编码器
            transformer=transformer,  # 注册变换器
            scheduler=scheduler,  # 注册调度器
        )
        # 计算空间缩放因子，默认值为8
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # 获取时间压缩比，如果 VAE 存在则使用其配置
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )

        # 创建视频处理器，使用空间缩放因子
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._get_t5_prompt_embeds 复制而来
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,  # 输入提示，支持单个字符串或字符串列表
        num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量
        max_sequence_length: int = 226,  # 最大序列长度
        device: Optional[torch.device] = None,  # 设备类型，默认为 None
        dtype: Optional[torch.dtype] = None,  # 数据类型，默认为 None
    ):
        # 如果未指定设备，则使用执行设备
        device = device or self._execution_device
        # 如果未指定数据类型，则使用文本编码器的数据类型
        dtype = dtype or self.text_encoder.dtype

        # 如果提示是字符串，则将其转为列表
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 获取批处理大小
        batch_size = len(prompt)

        # 对提示进行编码，返回张量，填充到最大长度
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",  # 填充到最大长度
            max_length=max_sequence_length,  # 最大长度限制
            truncation=True,  # 允许截断
            add_special_tokens=True,  # 添加特殊标记
            return_tensors="pt",  # 返回 PyTorch 张量
        )
        # 获取编码后的输入 ID
        text_input_ids = text_inputs.input_ids
        # 获取未截断的输入 ID
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        # 如果未截断的 ID 长度大于等于文本输入 ID 长度且两者不相等，则进行警告
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            # 解码被截断的文本部分并记录警告
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"  # 输出截断的提示文本
            )

        # 获取提示的嵌入表示
        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        # 转换嵌入为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # 复制文本嵌入以生成每个提示的视频，使用适合 MPS 的方法
        _, seq_len, _ = prompt_embeds.shape  # 获取嵌入的形状
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)  # 重复嵌入以适应视频数量
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)  # 变形为合适的形状

        # 返回处理后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.encode_prompt 复制而来
    # 定义一个用于编码提示信息的函数，接受多种参数
        def encode_prompt(
            self,
            prompt: Union[str, List[str]],  # 输入的提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负提示，类似格式
            do_classifier_free_guidance: bool = True,  # 是否启用无分类器引导
            num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入
            max_sequence_length: int = 226,  # 最大序列长度
            device: Optional[torch.device] = None,  # 指定的设备类型
            dtype: Optional[torch.dtype] = None,  # 指定的数据类型
        def prepare_latents(
            self,
            image: torch.Tensor,  # 输入图像的张量
            batch_size: int = 1,  # 每批次的样本数量
            num_channels_latents: int = 16,  # 潜在变量的通道数
            num_frames: int = 13,  # 视频的帧数
            height: int = 60,  # 图像的高度
            width: int = 90,  # 图像的宽度
            dtype: Optional[torch.dtype] = None,  # 指定的数据类型
            device: Optional[torch.device] = None,  # 指定的设备类型
            generator: Optional[torch.Generator] = None,  # 随机数生成器
            latents: Optional[torch.Tensor] = None,  # 可选的潜在变量张量
        ):
            # 计算有效的帧数，以适应 VAE 的时间缩放因子
            num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            # 定义张量的形状，包括批次、帧数和空间维度
            shape = (
                batch_size,
                num_frames,
                num_channels_latents,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )
    
            # 检查生成器列表的长度是否与批次大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 在图像张量中插入一个维度，以适配后续处理
            image = image.unsqueeze(2)  # [B, C, F, H, W]
    
            # 如果生成器是列表，逐个处理每个图像
            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                ]
            else:
                # 使用单一生成器处理所有图像
                image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]
    
            # 合并图像潜在变量，并调整维度
            image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            # 按比例缩放图像潜在变量
            image_latents = self.vae.config.scaling_factor * image_latents
    
            # 定义潜在变量的填充形状
            padding_shape = (
                batch_size,
                num_frames - 1,
                num_channels_latents,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )
            # 创建填充张量
            latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
            # 将填充与图像潜在变量合并
            image_latents = torch.cat([image_latents, latent_padding], dim=1)
    
            # 如果没有提供潜在变量，则生成随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 将提供的潜在变量移动到指定设备
                latents = latents.to(device)
    
            # 按照调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回潜在变量和图像潜在变量
            return latents, image_latents
    
        # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.decode_latents 复制的代码
    # 解码潜在变量并返回张量格式的帧
        def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
            # 重新排列潜在变量的维度为 [batch_size, num_channels, num_frames, height, width]
            latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
            # 将潜在变量缩放为 VAE 配置中的因子
            latents = 1 / self.vae.config.scaling_factor * latents
    
            # 解码潜在变量并获取采样帧
            frames = self.vae.decode(latents).sample
            # 返回解码得到的帧
            return frames
    
        # 从 diffusers.pipelines.animatediff.pipeline_animatediff_video2video 导入的方法
        def get_timesteps(self, num_inference_steps, timesteps, strength, device):
            # 根据 init_timestep 获取原始时间步
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算时间步的起始位置，确保不小于零
            t_start = max(num_inference_steps - init_timestep, 0)
            # 根据调度器的顺序获取相关时间步
            timesteps = timesteps[t_start * self.scheduler.order :]
    
            # 返回过滤后的时间步和剩余的推理步骤数
            return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 导入的方法
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备额外的调度器步骤参数，因为不同调度器的参数不尽相同
            # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
            # eta 对应 DDIM 论文中的 η，范围应在 [0, 1] 之间
    
            # 检查调度器是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            # 如果调度器接受 eta，则将其添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator，则将其添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    
        # 检查输入参数的有效性
        def check_inputs(
            self,
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            video=None,
            latents=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        # 检查 image 是否是合法类型：torch.Tensor、PIL.Image.Image 或 list
        if (
            not isinstance(image, torch.Tensor)  # 如果 image 不是 torch.Tensor
            and not isinstance(image, PIL.Image.Image)  # 并且不是 PIL.Image.Image
            and not isinstance(image, list)  # 并且不是 list
        ):
            # 抛出类型错误，提示 image 的类型不正确
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"  # 显示当前 image 的类型
            )

        # 检查 height 和 width 是否能被 8 整除
        if height % 8 != 0 or width % 8 != 0:
            # 抛出值错误，提示 height 和 width 不符合要求
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调输入是否存在，并且是否全在 _callback_tensor_inputs 中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs  # 确保每个 k 都在 _callback_tensor_inputs 中
        ):
            # 抛出值错误，提示回调输入不符合要求
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        
        # 检查 prompt 和 prompt_embeds 是否同时存在
        if prompt is not None and prompt_embeds is not None:
            # 抛出值错误，提示不能同时提供 prompt 和 prompt_embeds
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都为 None
        elif prompt is None and prompt_embeds is None:
            # 抛出值错误，提示至少提供一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 是否为合法类型
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出值错误，提示 prompt 的类型不正确
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查 prompt 和 negative_prompt_embeds 是否同时存在
        if prompt is not None and negative_prompt_embeds is not None:
            # 抛出值错误，提示不能同时提供 prompt 和 negative_prompt_embeds
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 negative_prompt 和 negative_prompt_embeds 是否同时存在
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出值错误，提示不能同时提供 negative_prompt 和 negative_prompt_embeds
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否都存在
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            # 检查它们的形状是否一致
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出值错误，提示它们的形状不匹配
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查 video 和 latents 是否同时存在
        if video is not None and latents is not None:
            # 抛出值错误，提示只能提供一个
            raise ValueError("Only one of `video` or `latents` should be provided")

    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.fuse_qkv_projections 复制而来
    # 定义一个启用融合 QKV 投影的方法，不返回任何值
    def fuse_qkv_projections(self) -> None:
        # 方法的文档字符串，描述其功能
        r"""Enables fused QKV projections."""
        # 设置属性 fusing_transformer 为 True，表示启用融合
        self.fusing_transformer = True
        # 调用 transformer 对象的方法，进行 QKV 投影融合
        self.transformer.fuse_qkv_projections()

    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.unfuse_qkv_projections 复制的方法
    # 定义一个禁用 QKV 投影融合的方法，不返回任何值
    def unfuse_qkv_projections(self) -> None:
        # 方法的文档字符串，描述其功能
        r"""Disable QKV projection fusion if enabled."""
        # 检查属性 fusing_transformer 是否为 False
        if not self.fusing_transformer:
            # 如果没有融合，记录警告日志
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            # 调用 transformer 对象的方法，解除 QKV 投影的融合
            self.transformer.unfuse_qkv_projections()
            # 设置属性 fusing_transformer 为 False，表示禁用融合
            self.fusing_transformer = False

    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings 复制的方法
    # 定义一个准备旋转位置嵌入的方法，返回两个张量
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 根据输入高度和宽度，计算网格的高度
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        # 根据输入高度和宽度，计算网格的宽度
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        # 计算基础宽度，固定为 720
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        # 计算基础高度，固定为 480
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        # 获取网格裁剪区域的坐标
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        # 获取旋转位置嵌入的余弦和正弦频率
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        # 将余弦频率张量转移到指定设备
        freqs_cos = freqs_cos.to(device=device)
        # 将正弦频率张量转移到指定设备
        freqs_sin = freqs_sin.to(device=device)
        # 返回余弦和正弦频率张量
        return freqs_cos, freqs_sin

    # 定义一个属性，获取指导尺度的值
    @property
    def guidance_scale(self):
        # 返回私有属性 _guidance_scale 的值
        return self._guidance_scale

    # 定义一个属性，获取时间步数的值
    @property
    def num_timesteps(self):
        # 返回私有属性 _num_timesteps 的值
        return self._num_timesteps

    # 定义一个属性，获取中断状态的值
    @property
    def interrupt(self):
        # 返回私有属性 _interrupt 的值
        return self._interrupt

    # 采用无梯度上下文装饰器，避免计算梯度
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用方法，允许实例像函数一样被调用
        def __call__(
            self,
            image: PipelineImageInput,  # 输入图像，类型为PipelineImageInput
            prompt: Optional[Union[str, List[str]]] = None,  # 提示文本，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 负面提示文本，可以是字符串或字符串列表
            height: int = 480,  # 输出图像的高度，默认为480
            width: int = 720,  # 输出图像的宽度，默认为720
            num_frames: int = 49,  # 生成的视频帧数，默认为49
            num_inference_steps: int = 50,  # 推理步骤的数量，默认为50
            timesteps: Optional[List[int]] = None,  # 可选的时间步列表
            guidance_scale: float = 6,  # 引导尺度，影响生成图像的质量，默认为6
            use_dynamic_cfg: bool = False,  # 是否使用动态配置，默认为False
            num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量，默认为1
            eta: float = 0.0,  # 影响采样过程的参数，默认为0.0
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选的随机数生成器
            latents: Optional[torch.FloatTensor] = None,  # 可选的潜在变量，类型为浮点张量
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的提示嵌入，类型为浮点张量
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的负面提示嵌入，类型为浮点张量
            output_type: str = "pil",  # 输出类型，默认为"PIL"格式
            return_dict: bool = True,  # 是否返回字典格式的结果，默认为True
            callback_on_step_end: Optional[  # 在步骤结束时调用的可选回调函数
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # 在步骤结束时的张量输入列表，默认为["latents"]
            max_sequence_length: int = 226,  # 最大序列长度，默认为226
```