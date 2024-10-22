# `.\diffusers\pipelines\cogvideo\pipeline_cogvideox_video2video.py`

```py
# 版权声明，指明版权归 CogVideoX 团队、清华大学、ZhipuAI 和 HuggingFace 团队所有
# 所有权利保留
#
# 根据 Apache License 2.0（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是以“原样”基础分发，
# 不提供任何形式的担保或条件，无论是明示或暗示的
# 有关许可证的特定权限和限制，请参见许可证

import inspect  # 导入 inspect 模块，用于获取对象的信息
import math  # 导入 math 模块，提供数学函数
from typing import Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类

import torch  # 导入 PyTorch 库，进行深度学习
from PIL import Image  # 从 PIL 库导入 Image，用于图像处理
from transformers import T5EncoderModel, T5Tokenizer  # 导入 T5 模型及其分词器

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入回调相关类
from ...models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel  # 导入模型类
from ...models.embeddings import get_3d_rotary_pos_embed  # 导入获取 3D 旋转位置嵌入的函数
from ...pipelines.pipeline_utils import DiffusionPipeline  # 导入扩散管道类
from ...schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler  # 导入调度器类
from ...utils import (  # 导入工具模块中的函数
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入生成随机张量的函数
from ...video_processor import VideoProcessor  # 导入视频处理器类
from .pipeline_output import CogVideoXPipelineOutput  # 导入管道输出类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

EXAMPLE_DOC_STRING = """  # 示例文档字符串，展示用法
    Examples:  # 示例部分
        ```python  # Python 代码块开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline  # 导入特定模块
        >>> from diffusers.utils import export_to_video, load_video  # 导入工具函数

        >>> # 模型：可以选择 "THUDM/CogVideoX-2b" 或 "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoXVideoToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)  # 加载预训练管道
        >>> pipe.to("cuda")  # 将管道移动到 GPU
        >>> pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)  # 配置调度器

        >>> input_video = load_video(  # 加载输入视频
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"  # 视频链接
        ... )
        >>> prompt = (  # 定义生成视频的提示
        ...     "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
        ...     "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
        ...     "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
        ...     "moons, but the remainder of the scene is mostly realistic."
        ... )

        >>> video = pipe(  # 调用管道生成视频
        ...     video=input_video, prompt=prompt, strength=0.8, guidance_scale=6, num_inference_steps=50  # 传入参数
        ... ).frames[0]  # 获取生成的视频帧
        >>> export_to_video(video, "output.mp4", fps=8)  # 导出生成的视频
        ```py  # Python 代码块结束
"""
# 根据源图像的大小和目标宽高计算缩放裁剪区域
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    # 设置目标宽度和高度
    tw = tgt_width
    th = tgt_height
    # 解构源图像的高度和宽度
    h, w = src
    # 计算源图像的宽高比
    r = h / w
    # 判断源图像的宽高比与目标宽高比的关系
    if r > (th / tw):
        # 如果源图像更高，则以目标高度缩放
        resize_height = th
        # 计算相应的宽度
        resize_width = int(round(th / h * w))
    else:
        # 否则以目标宽度缩放
        resize_width = tw
        # 计算相应的高度
        resize_height = int(round(tw / w * h))

    # 计算裁剪区域的顶部坐标
    crop_top = int(round((th - resize_height) / 2.0))
    # 计算裁剪区域的左侧坐标
    crop_left = int(round((tw - resize_width) / 2.0))

    # 返回裁剪区域的坐标和调整后的尺寸
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制的函数
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器检索时间步。处理自定义时间步。任何额外参数将传递给 `scheduler.set_timesteps`。

    参数：
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步应移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义时间步。如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义 sigma。如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是调度器的时间步计划，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了时间步和 sigma
    if timesteps is not None and sigmas is not None:
        raise ValueError("只能传递 `timesteps` 或 `sigmas` 之一。请选择一个设置自定义值")
    # 如果传递了时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义"
                f" 时间步计划。请检查您是否使用了正确的调度器。"
            )
        # 调用调度器的 set_timesteps 方法
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 检查 sigmas 是否不为 None，即是否提供了自定义 sigma 值
    elif sigmas is not None:
        # 检查当前调度器的 set_timesteps 方法是否接受 sigmas 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受 sigmas，抛出值错误异常，并提示用户
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步长，使用提供的 sigmas、设备和其他参数
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取当前调度器的时间步长
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量，即时间步长的长度
        num_inference_steps = len(timesteps)
    else:
        # 如果没有提供 sigmas，使用推理步骤的数量设置调度器的时间步长
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取当前调度器的时间步长
        timesteps = scheduler.timesteps
    # 返回时间步长和推理步骤的数量
    return timesteps, num_inference_steps
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 模块复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor,  # 输入参数，编码器输出，类型为 torch.Tensor
    generator: Optional[torch.Generator] = None,  # 可选的随机数生成器，用于采样
    sample_mode: str = "sample"  # 采样模式，默认为 "sample"
):
    # 检查 encoder_output 是否具有 latent_dist 属性且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中采样，并返回样本
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否具有 latent_dist 属性且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的众数（最可能的值）
        return encoder_output.latent_dist.mode()
    # 检查 encoder_output 是否具有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 latents 属性
        return encoder_output.latents
    # 如果都不满足，则引发 AttributeError
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class CogVideoXVideoToVideoPipeline(DiffusionPipeline):
    r""" 
    使用 CogVideoX 的视频到视频生成管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档，以获取库实现的所有管道的通用方法 
    （例如下载或保存，运行在特定设备等）。

    Args:
        vae ([`AutoencoderKL`]): 
            用于将视频编码和解码到潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`T5EncoderModel`]): 
            冻结的文本编码器。CogVideoX 使用 
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)；特别是 
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) 变体。
        tokenizer (`T5Tokenizer`): 
            类的标记器 
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer)。
        transformer ([`CogVideoXTransformer3DModel`]): 
            一个文本条件的 `CogVideoXTransformer3DModel`，用于去噪编码的视频潜在表示。
        scheduler ([`SchedulerMixin`]): 
            用于与 `transformer` 结合使用的调度器，以去噪编码的视频潜在表示。
    """

    _optional_components = []  # 可选组件的列表，当前为空
    model_cpu_offload_seq = "text_encoder->transformer->vae"  # 模型的 CPU 卸载顺序

    _callback_tensor_inputs = [  # 用于回调的张量输入列表
        "latents",  # 潜在表示
        "prompt_embeds",  # 提示嵌入
        "negative_prompt_embeds",  # 负面提示嵌入
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,  # 标记器实例
        text_encoder: T5EncoderModel,  # 文本编码器实例
        vae: AutoencoderKLCogVideoX,  # VAE 实例
        transformer: CogVideoXTransformer3DModel,  # 转换器实例
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],  # 调度器实例，可为两种类型之一
    # 初始化父类
        ):
            super().__init__()
    
            # 注册所需模块，包括tokenizer、text_encoder、vae、transformer和scheduler
            self.register_modules(
                tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
            )
            # 计算空间的vae缩放因子，如果vae存在则根据块的输出通道数计算，否则默认为8
            self.vae_scale_factor_spatial = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
            # 计算时间的vae缩放因子，如果vae存在则使用其时间压缩比，否则默认为4
            self.vae_scale_factor_temporal = (
                self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
            )
    
            # 初始化视频处理器，使用空间的vae缩放因子
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
    
        # 从diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._get_t5_prompt_embeds复制的方法
        def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            num_videos_per_prompt: int = 1,
            max_sequence_length: int = 226,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
        ):
            # 设置执行设备，若未指定则使用默认执行设备
            device = device or self._execution_device
            # 设置数据类型，若未指定则使用text_encoder的数据类型
            dtype = dtype or self.text_encoder.dtype
    
            # 将输入的prompt转换为列表格式
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 计算批次大小
            batch_size = len(prompt)
    
            # 使用tokenizer处理文本输入，返回张量格式，并进行填充、截断和添加特殊标记
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            # 获取处理后的输入ID
            text_input_ids = text_inputs.input_ids
            # 获取未截断的ID
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    
            # 检查未截断ID是否大于等于处理后的ID，并且两者不相等
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被截断的文本，并发出警告
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_sequence_length} tokens: {removed_text}"
                )
    
            # 通过text_encoder生成prompt的嵌入表示，并将其移动到指定设备
            prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
            # 转换嵌入的dtype和设备
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
            # 为每个生成的prompt重复文本嵌入，使用适合MPS的方法
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            # 重新调整嵌入的形状，以适应批次大小和生成数量
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    
            # 返回最终的文本嵌入
            return prompt_embeds
    
        # 从diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.encode_prompt复制的方法
    # 定义编码提示的函数，接受多种参数以设置提示信息和生成参数
        def encode_prompt(
            self,
            # 提示内容，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 负提示内容，可选
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 是否进行分类器自由引导
            do_classifier_free_guidance: bool = True,
            # 每个提示生成的视频数量
            num_videos_per_prompt: int = 1,
            # 提示的嵌入向量，可选
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的嵌入向量，可选
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 最大序列长度
            max_sequence_length: int = 226,
            # 设备类型，可选
            device: Optional[torch.device] = None,
            # 数据类型，可选
            dtype: Optional[torch.dtype] = None,
        # 定义准备潜在变量的函数，接受视频和其他参数
        def prepare_latents(
            self,
            # 输入视频，可选
            video: Optional[torch.Tensor] = None,
            # 批次大小
            batch_size: int = 1,
            # 潜在通道数量
            num_channels_latents: int = 16,
            # 视频高度
            height: int = 60,
            # 视频宽度
            width: int = 90,
            # 数据类型，可选
            dtype: Optional[torch.dtype] = None,
            # 设备类型，可选
            device: Optional[torch.device] = None,
            # 随机数生成器，可选
            generator: Optional[torch.Generator] = None,
            # 现有潜在变量，可选
            latents: Optional[torch.Tensor] = None,
            # 时间步长，可选
            timestep: Optional[torch.Tensor] = None,
        ):
            # 计算视频帧数，如果潜在变量未提供则根据视频尺寸计算
            num_frames = (video.size(2) - 1) // self.vae_scale_factor_temporal + 1 if latents is None else latents.size(1)
    
            # 设置潜在变量的形状
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
    
            # 如果未提供潜在变量
            if latents is None:
                # 如果生成器是列表，则检查长度
                if isinstance(generator, list):
                    if len(generator) != batch_size:
                        raise ValueError(
                            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                        )
    
                    # 为每个视频初始化潜在变量
                    init_latents = [
                        retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                    ]
                else:
                    # 单一生成器情况下为视频初始化潜在变量
                    init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]
    
                # 将初始潜在变量连接并转移到目标数据类型，调整维度顺序
                init_latents = torch.cat(init_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                # 通过配置的缩放因子调整潜在变量
                init_latents = self.vae.config.scaling_factor * init_latents
    
                # 生成随机噪声
                noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                # 将噪声添加到初始潜在变量中
                latents = self.scheduler.add_noise(init_latents, noise, timestep)
            else:
                # 如果潜在变量已提供，则将其转移到目标设备
                latents = latents.to(device)
    
            # 根据调度器要求缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回准备好的潜在变量
            return latents
    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.decode_latents 拷贝而来
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # 将输入的张量进行维度变换，排列为 [batch_size, num_channels, num_frames, height, width]
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        # 使用 VAE 的缩放因子对 latents 进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents

        # 解码 latents，生成相应的帧并返回
        frames = self.vae.decode(latents).sample
        # 返回解码后的帧
        return frames

    # 从 diffusers.pipelines.animatediff.pipeline_animatediff_video2video.AnimateDiffVideoToVideoPipeline.get_timesteps 拷贝而来
    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # 根据初始时间步计算原始时间步
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步，确保不小于0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从时间步数组中截取相关部分
        timesteps = timesteps[t_start * self.scheduler.order :]

        # 返回调整后的时间步和剩余的推理步骤
        return timesteps, num_inference_steps - t_start

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 拷贝而来
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器具有相同的参数签名
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略该参数
        # eta 对应于 DDIM 论文中的 η，参考链接：https://arxiv.org/abs/2010.02502
        # eta 应该在 [0, 1] 之间

        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        strength,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        video=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        # 检查高度和宽度是否能被8整除，如果不能则抛出错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查strength的值是否在0到1之间，如果不在范围内则抛出错误
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 检查callback_on_step_end_tensor_inputs是否不为None且是否所有元素都在_callback_tensor_inputs中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 抛出错误，如果callback_on_step_end_tensor_inputs中的某些元素不在_callback_tensor_inputs中
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        
        # 检查prompt和prompt_embeds是否同时不为None
        if prompt is not None and prompt_embeds is not None:
            # 抛出错误，提示不能同时提供prompt和prompt_embeds
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            # 抛出错误，提示必须提供prompt或prompt_embeds其中之一
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出错误，提示prompt的类型必须是str或list
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查prompt和negative_prompt_embeds是否同时不为None
        if prompt is not None and negative_prompt_embeds is not None:
            # 抛出错误，提示不能同时提供prompt和negative_prompt_embeds
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查negative_prompt和negative_prompt_embeds是否同时不为None
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出错误，提示不能同时提供negative_prompt和negative_prompt_embeds
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查prompt_embeds和negative_prompt_embeds是否都不为None
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            # 检查两个embeds的形状是否相同，如果不同则抛出错误
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查video和latents是否同时不为None
        if video is not None and latents is not None:
            # 抛出错误，提示只能提供video或latents其中之一
            raise ValueError("Only one of `video` or `latents` should be provided")

    # 从diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline复制的方法
    def fuse_qkv_projections(self) -> None:
        # 文档字符串，说明该方法启用融合的QKV投影
        r"""Enables fused QKV projections."""
        # 设置fusing_transformer属性为True，表示启用融合
        self.fusing_transformer = True
        # 调用transformer对象的fuse_qkv_projections方法
        self.transformer.fuse_qkv_projections()
    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.unfuse_qkv_projections 复制的代码
    def unfuse_qkv_projections(self) -> None:
        r"""禁用 QKV 投影融合（如果已启用）。"""
        # 检查是否启用了投影融合
        if not self.fusing_transformer:
            # 如果没有启用，记录警告信息
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            # 如果启用了，执行解除 QKV 投影融合操作
            self.transformer.unfuse_qkv_projections()
            # 将融合标志设置为 False
            self.fusing_transformer = False

    # 从 diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings 复制的代码
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,  # 输入的高度
        width: int,   # 输入的宽度
        num_frames: int,  # 输入的帧数
        device: torch.device,  # 指定的设备（CPU 或 GPU）
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回的类型为一对张量
        # 计算网格高度，基于输入高度和其他参数
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        # 计算网格宽度，基于输入宽度和其他参数
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        # 计算基础宽度大小
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        # 计算基础高度大小
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        # 获取用于网格的裁剪区域坐标
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        # 生成三维旋转位置嵌入的余弦和正弦频率
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,  # 嵌入维度
            crops_coords=grid_crops_coords,  # 裁剪坐标
            grid_size=(grid_height, grid_width),  # 网格大小
            temporal_size=num_frames,  # 时间序列大小
        )

        # 将余弦频率张量移动到指定设备
        freqs_cos = freqs_cos.to(device=device)
        # 将正弦频率张量移动到指定设备
        freqs_sin = freqs_sin.to(device=device)
        # 返回余弦和正弦频率张量
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        # 返回指导尺度的值
        return self._guidance_scale

    @property
    def num_timesteps(self):
        # 返回时间步数的值
        return self._num_timesteps

    @property
    def interrupt(self):
        # 返回中断标志的值
        return self._interrupt

    @torch.no_grad()  # 在不计算梯度的上下文中运行
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 替换示例文档字符串
    # 定义可调用的类方法，允许传入多个参数以处理视频生成
    def __call__(
            # 视频图像列表，默认为 None
            self,
            video: List[Image.Image] = None,
            # 生成视频的提示文本，可以是字符串或字符串列表，默认为 None
            prompt: Optional[Union[str, List[str]]] = None,
            # 生成视频的负面提示文本，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 输出视频的高度，默认为 480 像素
            height: int = 480,
            # 输出视频的宽度，默认为 720 像素
            width: int = 720,
            # 进行推断的步骤数量，默认为 50 步
            num_inference_steps: int = 50,
            # 选定的时间步列表，默认为 None
            timesteps: Optional[List[int]] = None,
            # 控制强度的浮点数，默认为 0.8
            strength: float = 0.8,
            # 引导缩放比例，默认为 6
            guidance_scale: float = 6,
            # 是否使用动态配置的布尔值，默认为 False
            use_dynamic_cfg: bool = False,
            # 每个提示生成视频的数量，默认为 1
            num_videos_per_prompt: int = 1,
            # eta 参数，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是 torch.Generator 或其列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在张量，默认为 None
            latents: Optional[torch.FloatTensor] = None,
            # 可选的提示嵌入，默认为 None
            prompt_embeds: Optional[torch.FloatTensor] = None,
            # 可选的负面提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 输出类型，默认为 "pil"
            output_type: str = "pil",
            # 是否返回字典格式，默认为 True
            return_dict: bool = True,
            # 步骤结束时调用的回调函数，可以是单一或多个回调，默认为 None
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 用于步骤结束回调的张量输入列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 最大序列长度，默认为 226
            max_sequence_length: int = 226,
```