# `.\diffusers\pipelines\cogvideo\pipeline_cogvideox.py`

```py
# 版权声明，表明此文件的所有权和使用许可
# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# 根据 Apache 2.0 许可证许可，使用条款
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 你可以在以下地址获得许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面协议另有约定，否则此文件以“按原样”方式分发，不提供任何明示或暗示的担保或条件。
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入用于检查函数和方法的模块
import inspect
# 导入数学库以使用数学函数
import math
# 从 typing 模块导入类型注释工具
from typing import Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 T5 编码器模型和分词器
from transformers import T5EncoderModel, T5Tokenizer

# 从相对路径导入回调相关的类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 从相对路径导入模型相关的类
from ...models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
# 从相对路径导入获取 3D 旋转位置嵌入的函数
from ...models.embeddings import get_3d_rotary_pos_embed
# 从相对路径导入扩散管道工具
from ...pipelines.pipeline_utils import DiffusionPipeline
# 从相对路径导入调度器
from ...schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
# 从相对路径导入日志工具和替换示例文档字符串的函数
from ...utils import logging, replace_example_docstring
# 从相对路径导入生成随机张量的工具
from ...utils.torch_utils import randn_tensor
# 从相对路径导入视频处理器
from ...video_processor import VideoProcessor
# 从当前包导入管道输出相关的类
from .pipeline_output import CogVideoXPipelineOutput

# 创建一个日志记录器，用于记录当前模块的信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该模块的功能
EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoXPipeline
        >>> from diffusers.utils import export_to_video

        >>> # 模型： "THUDM/CogVideoX-2b" 或 "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
        >>> prompt = (
        ...     "一只穿着小红外套和小帽子的熊猫，坐在宁静的竹林中的木凳上。"
        ...     "熊猫的毛茸茸的爪子拨动着微型木吉他，演奏出柔和的旋律。附近，几只其他的熊猫好奇地聚集，"
        ...     "有些还在节奏中鼓掌。阳光透过高大的竹子，洒下柔和的光辉，"
        ...     "照亮了这个场景。熊猫的脸上流露出专注和快乐，随着音乐的演奏而展现。"
        ...     "背景中有一条小溪流和生机勃勃的绿叶，增强了这个独特音乐表演的宁静和魔幻气氛。"
        ... )
        >>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```py
"""

# 定义一个函数，用于计算调整大小和裁剪区域，以适应网格
# 该函数类似于 diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    # 目标宽度赋值给变量 tw
    tw = tgt_width
    # 目标高度赋值给变量 th
    th = tgt_height
    # 从源图像的尺寸中提取高度和宽度
    h, w = src
    # 计算源图像的高宽比
    r = h / w
    # 检查缩放比例 r 是否大于给定阈值 th 和 tw 的比值
        if r > (th / tw):
            # 如果是，则设定新的高度为 th
            resize_height = th
            # 计算对应的宽度，保持宽高比
            resize_width = int(round(th / h * w))
        else:
            # 否则，设定新的宽度为 tw
            resize_width = tw
            # 计算对应的高度，保持宽高比
            resize_height = int(round(tw / w * h))
    
        # 计算裁剪的上边缘位置，以居中显示
        crop_top = int(round((th - resize_height) / 2.0))
        # 计算裁剪的左边缘位置，以居中显示
        crop_left = int(round((tw - resize_width) / 2.0))
    
        # 返回裁剪区域的坐标，包含左上角和右下角
        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制
def retrieve_timesteps(
    # 调度器对象，用于获取时间步
    scheduler,
    # 用于生成样本的推理步骤数（可选）
    num_inference_steps: Optional[int] = None,
    # 指定设备（可选），可以是字符串或 torch.device
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步列表（可选）
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 列表（可选）
    sigmas: Optional[List[float]] = None,
    # 额外的关键字参数，传递给调度器的 set_timesteps 方法
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并从调度器中检索时间步。处理自定义时间步。
    任何关键字参数都将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            在生成样本时使用的扩散步骤数。如果使用，`timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步移动到的设备。如果为 `None`，时间步不会被移动。
        timesteps (`List[int]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义时间步。如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义 sigma。如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是来自调度器的时间步安排，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了时间步和 sigma，若是则抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了自定义时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传递自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传递了自定义 sigma
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传递自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 否则，设置推理步骤数以及相关设备和额外参数
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器中的时间步
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤数
        return timesteps, num_inference_steps
# 定义一个名为 CogVideoXPipeline 的类，继承自 DiffusionPipeline 类
class CogVideoXPipeline(DiffusionPipeline):
    r"""
    使用 CogVideoX 进行文本到视频生成的管道。

    此模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法（例如下载或保存，运行在特定设备等），请查看超类文档。

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于将视频编码和解码为潜在表示。
        text_encoder ([`T5EncoderModel`]):
            冻结的文本编码器。CogVideoX 使用
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)；具体使用
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) 变体。
        tokenizer (`T5Tokenizer`):
            类
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer) 的标记器。
        transformer ([`CogVideoXTransformer3DModel`]):
            一个文本条件的 `CogVideoXTransformer3DModel` 用于去噪编码的视频潜在。
        scheduler ([`SchedulerMixin`]):
            与 `transformer` 结合使用的调度器，用于去噪编码的视频潜在。
    """

    # 定义可选组件的列表，初始化为空
    _optional_components = []
    # 定义模型 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 定义回调张量输入的列表
    _callback_tensor_inputs = [
        "latents",  # 潜在张量输入
        "prompt_embeds",  # 提示嵌入张量输入
        "negative_prompt_embeds",  # 负提示嵌入张量输入
    ]

    # 初始化函数，接受多个参数以构建管道
    def __init__(
        self,
        tokenizer: T5Tokenizer,  # T5 标记器实例
        text_encoder: T5EncoderModel,  # T5 文本编码器实例
        vae: AutoencoderKLCogVideoX,  # 变分自编码器实例
        transformer: CogVideoXTransformer3DModel,  # CogVideoX 3D 转换器实例
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],  # 调度器实例，支持多种类型
    ):
        # 调用超类的初始化函数
        super().__init__()

        # 注册模块，整合各个组件
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        # 根据 VAE 的配置计算空间缩放因子
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # 根据 VAE 的配置计算时间缩放因子
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )

        # 初始化视频处理器，使用空间缩放因子
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # 定义获取 T5 提示嵌入的函数
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,  # 输入的提示，可以是字符串或字符串列表
        num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量，默认为 1
        max_sequence_length: int = 226,  # 最大序列长度，默认为 226
        device: Optional[torch.device] = None,  # 设备类型，默认为 None
        dtype: Optional[torch.dtype] = None,  # 数据类型，默认为 None
    # 处理输入参数，优先使用已设置的设备
        ):
            device = device or self._execution_device
            # 使用已定义的 dtype，默认取文本编码器的 dtype
            dtype = dtype or self.text_encoder.dtype
    
            # 如果输入 prompt 是字符串，则将其转换为列表
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取 prompt 的批大小
            batch_size = len(prompt)
    
            # 使用 tokenizer 处理 prompt，并返回张量格式的输入
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",  # 填充到最大长度
                max_length=max_sequence_length,  # 最大序列长度
                truncation=True,  # 超出部分截断
                add_special_tokens=True,  # 添加特殊标记
                return_tensors="pt",  # 返回 PyTorch 张量
            )
            # 提取输入 ID
            text_input_ids = text_inputs.input_ids
            # 获取未截断的输入 ID
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    
            # 检查是否需要警告用户输入被截断
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被移除的文本并记录警告
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_sequence_length} tokens: {removed_text}"
                )
    
            # 获取文本输入的嵌入表示
            prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
            # 转换嵌入的 dtype 和 device
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
            # 为每个生成的提示重复文本嵌入，使用适合 MPS 的方法
            _, seq_len, _ = prompt_embeds.shape
            # 重复嵌入以匹配视频生成数量
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            # 调整嵌入的形状以符合批处理
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    
            # 返回最终的文本嵌入
            return prompt_embeds
    
        # 定义编码提示的函数
        def encode_prompt(
            self,
            # 输入的提示，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 可选的负提示
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 控制分类器自由引导的开关
            do_classifier_free_guidance: bool = True,
            # 每个提示生成的视频数量
            num_videos_per_prompt: int = 1,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 最大序列长度
            max_sequence_length: int = 226,
            # 可选的设备
            device: Optional[torch.device] = None,
            # 可选的数据类型
            dtype: Optional[torch.dtype] = None,
        # 准备潜在变量的函数
        def prepare_latents(
            self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 定义形状元组，包含批次大小、帧数、通道数、高度和宽度
        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,  # 计算处理后的帧数
            num_channels_latents,  # 潜在通道数
            height // self.vae_scale_factor_spatial,  # 根据空间缩放因子调整高度
            width // self.vae_scale_factor_spatial,  # 根据空间缩放因子调整宽度
        )
        # 检查生成器是否是列表，且长度与批次大小不匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出值错误，说明生成器列表长度与批次大小不匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果潜在变量为 None，则生成随机潜在变量
        if latents is None:
            # 使用给定形状生成随机张量，指定生成器、设备和数据类型
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果潜在变量不为 None，则将其移动到指定设备
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 解码潜在变量，返回解码后的帧
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # 重新排列潜在变量的维度，以适应解码器的输入格式
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        # 根据缩放因子调整潜在变量的值
        latents = 1 / self.vae.config.scaling_factor * latents

        # 解码潜在变量并获取样本帧
        frames = self.vae.decode(latents).sample
        # 返回解码后的帧
        return frames

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并非所有调度器的签名相同
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 范围内

        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数
        return extra_step_kwargs

    # 从 diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs 复制
    def check_inputs(
        self,
        prompt,  # 输入的提示文本
        height,  # 生成图像的高度
        width,   # 生成图像的宽度
        negative_prompt,  # 负提示文本，用于引导生成
        callback_on_step_end_tensor_inputs,  # 每步结束时的回调，用于处理张量输入
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负提示嵌入
    ):
        # 检查高度和宽度是否能被8整除，若不能则抛出错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调输入是否不为空且是否都在已注册的回调输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        # 检查是否同时提供了提示和提示嵌入，若是则抛出错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否同时未提供提示和提示嵌入，若是则抛出错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示的类型是否为字符串或列表，若不是则抛出错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了提示和负提示嵌入，若是则抛出错误
        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查是否同时提供了负提示和负提示嵌入，若是则抛出错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负提示嵌入是否都不为空，且它们的形状是否相同，若不同则抛出错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        # 启用融合 QKV 投影
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        # 调用变换器进行 QKV 投影的融合
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        # 禁用 QKV 投影融合（如果已启用）
        r"""Disable QKV projection fusion if enabled."""
        # 如果没有启用融合，则记录警告信息
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            # 调用变换器进行 QKV 投影的取消融合
            self.transformer.unfuse_qkv_projections()
            # 更新状态为未融合
            self.fusing_transformer = False
    # 准备旋转位置嵌入的函数
        def _prepare_rotary_positional_embeddings(
            self,
            height: int,  # 输入的高度
            width: int,   # 输入的宽度
            num_frames: int,  # 输入的帧数
            device: torch.device,  # 计算设备（如CPU或GPU）
        ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回两个张量的元组
            # 根据 VAE 缩放因子和变换器的补丁大小计算网格高度
            grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
            # 根据 VAE 缩放因子和变换器的补丁大小计算网格宽度
            grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
            # 计算基础宽度大小
            base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
            # 计算基础高度大小
            base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
    
            # 获取网格的裁剪区域坐标
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            # 获取三维旋转位置嵌入的余弦和正弦频率
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,  # 嵌入维度
                crops_coords=grid_crops_coords,  # 裁剪坐标
                grid_size=(grid_height, grid_width),  # 网格大小
                temporal_size=num_frames,  # 时间维度大小
                use_real=True,  # 是否使用实数
            )
    
            # 将余弦频率移动到指定设备
            freqs_cos = freqs_cos.to(device=device)
            # 将正弦频率移动到指定设备
            freqs_sin = freqs_sin.to(device=device)
            # 返回余弦和正弦频率
            return freqs_cos, freqs_sin
    
        # 获取指导缩放比例的属性
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 获取时间步数的属性
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 获取中断状态的属性
        @property
        def interrupt(self):
            return self._interrupt
    
        # 关闭梯度计算并替换文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义调用方法
        def __call__(
            self,
            prompt: Optional[Union[str, List[str]]] = None,  # 输入提示
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 负面提示
            height: int = 480,  # 默认高度
            width: int = 720,  # 默认宽度
            num_frames: int = 49,  # 默认帧数
            num_inference_steps: int = 50,  # 默认推理步骤
            timesteps: Optional[List[int]] = None,  # 可选的时间步
            guidance_scale: float = 6,  # 默认指导缩放比例
            use_dynamic_cfg: bool = False,  # 是否使用动态配置
            num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量
            eta: float = 0.0,  # 控制噪声的参数
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            latents: Optional[torch.FloatTensor] = None,  # 可选的潜变量
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 提示嵌入
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 负面提示嵌入
            output_type: str = "pil",  # 输出类型
            return_dict: bool = True,  # 是否返回字典格式
            callback_on_step_end: Optional[  # 步骤结束时的回调
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # 步骤结束时的张量输入
            max_sequence_length: int = 226,  # 最大序列长度
```