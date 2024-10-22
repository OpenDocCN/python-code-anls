# `.\diffusers\pipelines\stable_video_diffusion\pipeline_stable_video_diffusion.py`

```py
# 版权声明，声明此文件的版权属于 HuggingFace 团队
# 
# 根据 Apache 许可证第 2.0 版（"许可证"）授权；
# 除非遵循许可证，否则您不能使用此文件。
# 您可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面同意，按照许可证分发的软件
# 是按 "原样" 基础分发，不提供任何形式的保证或条件。
# 有关许可证所涵盖的特定权限和限制，请参见许可证。

import inspect  # 导入 inspect 模块以检查对象的信息
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器，用于创建数据类
from typing import Callable, Dict, List, Optional, Union  # 导入类型提示相关的类型

import numpy as np  # 导入 numpy 库，用于数组和数值计算
import PIL.Image  # 导入 PIL.Image 模块，用于处理图像
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # 从 transformers 导入 CLIP 相关的处理器和模型

from ...image_processor import PipelineImageInput  # 从当前包导入 PipelineImageInput 类
from ...models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel  # 导入特定模型类
from ...schedulers import EulerDiscreteScheduler  # 导入调度器类
from ...utils import BaseOutput, logging, replace_example_docstring  # 导入工具类和函数
from ...utils.torch_utils import is_compiled_module, randn_tensor  # 导入与 PyTorch 相关的工具函数
from ...video_processor import VideoProcessor  # 导入视频处理器类
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道类

logger = logging.get_logger(__name__)  # 获取日志记录器，使用当前模块的名称

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串
    Examples:  # 示例部分
        ```py
        >>> from diffusers import StableVideoDiffusionPipeline  # 从 diffusers 导入管道
        >>> from diffusers.utils import load_image, export_to_video  # 导入加载图像和导出视频的工具

        >>> pipe = StableVideoDiffusionPipeline.from_pretrained(  # 从预训练模型创建管道
        ...     "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"  # 指定模型名称和类型
        ... )
        >>> pipe.to("cuda")  # 将管道移到 GPU

        >>> image = load_image(  # 加载图像
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd-docstring-example.jpeg"  # 指定图像 URL
        ... )
        >>> image = image.resize((1024, 576))  # 调整图像大小

        >>> frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]  # 生成视频帧
        >>> export_to_video(frames, "generated.mp4", fps=7)  # 导出帧为视频文件
        ```py
"""  # 结束示例文档字符串

def _append_dims(x, target_dims):  # 定义一个私有函数，用于向张量添加维度
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""  # 函数说明，描述其功能
    dims_to_append = target_dims - x.ndim  # 计算需要添加的维度数量
    if dims_to_append < 0:  # 如果目标维度小于当前维度
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")  # 抛出值错误
    return x[(...,) + (None,) * dims_to_append]  # 在张量末尾添加所需数量的维度

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 导入的函数
def retrieve_timesteps(  # 定义一个函数，用于检索时间步
    scheduler,  # 调度器对象
    num_inference_steps: Optional[int] = None,  # 推理步骤的可选参数
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备参数
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表
    sigmas: Optional[List[float]] = None,  # 可选的 sigma 值列表
    **kwargs,  # 接收其他关键字参数
):
    """  # 函数说明，描述其功能
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles  # 调用调度器的 `set_timesteps` 方法并检索时间步
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.  # 处理自定义时间步，其他关键字参数传递给 `scheduler.set_timesteps`
    # 参数说明
        Args:
            scheduler (`SchedulerMixin`):  # 调度器，用于获取时间步
                The scheduler to get timesteps from.
            num_inference_steps (`int`):  # 生成样本时使用的扩散步骤数
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):  # 指定时间步移动的设备
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):  # 自定义时间步以覆盖调度器的时间步间距策略
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):  # 自定义sigmas以覆盖调度器的时间步间距策略
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.
    
        Returns:
            `Tuple[torch.Tensor, int]`: 返回一个元组，包含调度器的时间步调度和推理步骤数
        """
        # 检查是否同时传入了时间步和sigmas
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        # 如果提供了时间步
        if timesteps is not None:
            # 检查当前调度器是否支持自定义时间步
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            # 设置时间步
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            # 获取设置后的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤数
            num_inference_steps = len(timesteps)
        # 如果提供了sigmas
        elif sigmas is not None:
            # 检查当前调度器是否支持自定义sigmas
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            # 设置sigmas
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            # 获取设置后的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤数
            num_inference_steps = len(timesteps)
        # 如果没有提供时间步和sigmas
        else:
            # 根据推理步骤数设置时间步
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取设置后的时间步
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤数
        return timesteps, num_inference_steps
# 定义 StableVideoDiffusionPipelineOutput 类，继承自 BaseOutput
@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    # 定义输出属性 frames，可以是多种类型：嵌套 PIL 图像列表、numpy 数组或 torch 张量
    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


# 定义 StableVideoDiffusionPipeline 类，继承自 DiffusionPipeline
class StableVideoDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder
            ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    # 定义模型的 CPU 卸载顺序，用于优化内存使用
    model_cpu_offload_seq = "image_encoder->unet->vae"
    # 定义需要回调的张量输入
    _callback_tensor_inputs = ["latents"]

    # 初始化方法，定义模型所需的组件
    def __init__(
        self,
        # VAE 模型，用于编码和解码图像
        vae: AutoencoderKLTemporalDecoder,
        # CLIP 图像编码器，被冻结以提取图像特征
        image_encoder: CLIPVisionModelWithProjection,
        # 用于去噪的 UNet 模型
        unet: UNetSpatioTemporalConditionModel,
        # 用于调度的 Euler 离散调度器
        scheduler: EulerDiscreteScheduler,
        # 图像特征提取器
        feature_extractor: CLIPImageProcessor,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 注册模型组件
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建视频处理器实例，启用图像缩放
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

    # 定义图像编码方法
    def _encode_image(
        self,
        # 输入图像，类型为 PipelineImageInput
        image: PipelineImageInput,
        # 设备类型，字符串或 torch.device
        device: Union[str, torch.device],
        # 每个提示生成的视频数量
        num_videos_per_prompt: int,
        # 是否进行无分类器自由引导
        do_classifier_free_guidance: bool,
    # 返回类型为 torch.Tensor 的函数
    ) -> torch.Tensor:
        # 获取图像编码器参数的 dtype
        dtype = next(self.image_encoder.parameters()).dtype
    
        # 如果输入的图像不是 torch.Tensor 类型
        if not isinstance(image, torch.Tensor):
            # 将 PIL 图像转换为 NumPy 数组
            image = self.video_processor.pil_to_numpy(image)
            # 将 NumPy 数组转换为 PyTorch 张量
            image = self.video_processor.numpy_to_pt(image)
    
            # 在调整大小之前对图像进行归一化，以匹配原始实现
            # 然后在调整大小后进行反归一化
            image = image * 2.0 - 1.0
            # 使用抗锯齿算法调整图像大小到 (224, 224)
            image = _resize_with_antialiasing(image, (224, 224))
            # 反归一化图像
            image = (image + 1.0) / 2.0
    
        # 使用 CLIP 输入对图像进行归一化
        image = self.feature_extractor(
            # 输入图像
            images=image,
            # 是否进行归一化
            do_normalize=True,
            # 是否进行中心裁剪
            do_center_crop=False,
            # 是否调整大小
            do_resize=False,
            # 是否重新缩放
            do_rescale=False,
            # 返回的张量类型
            return_tensors="pt",
        ).pixel_values
    
        # 将图像移动到指定设备并设置数据类型
        image = image.to(device=device, dtype=dtype)
        # 使用图像编码器生成图像嵌入
        image_embeddings = self.image_encoder(image).image_embeds
        # 在第 1 维上增加一个维度
        image_embeddings = image_embeddings.unsqueeze(1)
    
        # 针对每个提示的生成复制图像嵌入，使用适合 MPS 的方法
        bs_embed, seq_len, _ = image_embeddings.shape
        # 重复图像嵌入，针对每个提示的生成
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        # 将嵌入形状调整为适合的格式
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
    
        # 如果需要分类器自由引导
        if do_classifier_free_guidance:
            # 创建与图像嵌入相同形状的零张量
            negative_image_embeddings = torch.zeros_like(image_embeddings)
    
            # 对于分类器自由引导，我们需要进行两次前向传播
            # 在这里，我们将无条件和文本嵌入拼接成一个批次
            # 以避免进行两次前向传播
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
    
        # 返回图像嵌入
        return image_embeddings
    
    # 定义一个编码 VAE 图像的函数
    def _encode_vae_image(
        self,
        # 输入图像张量
        image: torch.Tensor,
        # 设备类型
        device: Union[str, torch.device],
        # 每个提示的视频数量
        num_videos_per_prompt: int,
        # 是否进行分类器自由引导
        do_classifier_free_guidance: bool,
    ):
        # 将图像移动到指定设备
        image = image.to(device=device)
        # 使用 VAE 编码器对图像进行编码，获取潜在分布的模式
        image_latents = self.vae.encode(image).latent_dist.mode()
    
        # 针对每个提示的生成复制图像潜在，使用适合 MPS 的方法
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
    
        # 如果需要分类器自由引导
        if do_classifier_free_guidance:
            # 创建与图像潜在相同形状的零张量
            negative_image_latents = torch.zeros_like(image_latents)
    
            # 对于分类器自由引导，我们需要进行两次前向传播
            # 在这里，我们将无条件和文本嵌入拼接成一个批次
            # 以避免进行两次前向传播
            image_latents = torch.cat([negative_image_latents, image_latents])
    
        # 返回图像潜在
        return image_latents
    
    # 定义一个获取附加时间 ID 的函数
    def _get_add_time_ids(
        # 帧率
        fps: int,
        # 动作桶 ID
        motion_bucket_id: int,
        # 噪声增强强度
        noise_aug_strength: float,
        # 数据类型
        dtype: torch.dtype,
        # 批大小
        batch_size: int,
        # 每个提示的视频数量
        num_videos_per_prompt: int,
        # 是否进行分类器自由引导
        do_classifier_free_guidance: bool,
    )
    ):
        # 将 FPS、运动桶 ID 和噪声增强强度放入列表
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        # 计算传入的时间嵌入维度（乘以添加的时间 ID 数量）
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        # 获取模型期望的时间嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查期望的和实际的时间嵌入维度是否匹配
        if expected_add_embed_dim != passed_add_embed_dim:
            # 如果不匹配，则抛出错误，提示配置不正确
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将添加的时间 ID 转换为张量，指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 重复添加时间 ID，生成与批次大小和每个提示的视频数相同的数量
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        # 如果进行无分类器自由引导
        if do_classifier_free_guidance:
            # 将添加的时间 ID 与自身连接以形成双倍数量
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        # 返回最终的添加时间 ID 张量
        return add_time_ids

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14):
        # 将输入的潜在张量形状从 [batch, frames, channels, height, width] 转换为 [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        # 使用 VAE 的缩放因子缩放潜在张量
        latents = 1 / self.vae.config.scaling_factor * latents

        # 确定前向 VAE 函数，如果 VAE 是编译模块，则使用其原始模块的前向方法
        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        # 检查前向函数是否接受 num_frames 参数
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # 每次解码 decode_chunk_size 帧以避免内存不足（OOM）
        frames = []
        # 按照 decode_chunk_size 步长遍历潜在张量
        for i in range(0, latents.shape[0], decode_chunk_size):
            # 获取当前块中的帧数
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            # 如果前向函数接受帧数参数
            if accepts_num_frames:
                # 仅在期望时传递当前帧数
                decode_kwargs["num_frames"] = num_frames_in

            # 解码当前潜在块并获取采样结果
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            # 将解码的帧添加到列表中
            frames.append(frame)
        # 将所有帧合并为一个张量
        frames = torch.cat(frames, dim=0)

        # 将帧的形状从 [batch*frames, channels, height, width] 转换为 [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # 将帧转换为 float32 类型，以确保兼容性并避免显著开销
        frames = frames.float()
        # 返回最终解码的帧
        return frames

    def check_inputs(self, image, height, width):
        # 检查输入图像类型是否有效
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            # 如果无效，则抛出错误，说明类型不匹配
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        # 检查高度和宽度是否为8的倍数
        if height % 8 != 0 or width % 8 != 0:
            # 如果不是，则抛出错误
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    # 准备潜在变量，返回适合输入的张量
        def prepare_latents(
            self,
            # 批次大小
            batch_size: int,
            # 帧数
            num_frames: int,
            # 潜在变量的通道数
            num_channels_latents: int,
            # 图像高度
            height: int,
            # 图像宽度
            width: int,
            # 数据类型
            dtype: torch.dtype,
            # 设备类型（CPU或GPU）
            device: Union[str, torch.device],
            # 随机数生成器
            generator: torch.Generator,
            # 可选的潜在变量张量
            latents: Optional[torch.Tensor] = None,
        ):
            # 定义潜在变量的形状
            shape = (
                batch_size,
                num_frames,
                num_channels_latents // 2,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            # 检查生成器列表的长度是否与批次大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果潜在变量为空，生成随机张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 将给定的潜在变量移动到指定设备
                latents = latents.to(device)
    
            # 根据调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回准备好的潜在变量
            return latents
    
        # 属性：获取引导比例
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # `guidance_scale` 属性定义与Imagen论文中公式(2)的引导权重`w`类似
        # `guidance_scale = 1`表示不进行分类器自由引导
        @property
        def do_classifier_free_guidance(self):
            # 检查引导比例是否为整数或浮点数
            if isinstance(self.guidance_scale, (int, float)):
                return self.guidance_scale > 1
            # 如果是张量，检查最大值是否大于1
            return self.guidance_scale.max() > 1
    
        # 属性：获取时间步数
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 不计算梯度的函数，用于调用模型
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 输入图像，可以是单个图像、图像列表或张量
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
            # 图像高度（默认为576）
            height: int = 576,
            # 图像宽度（默认为1024）
            width: int = 1024,
            # 可选的帧数
            num_frames: Optional[int] = None,
            # 推理步骤数（默认为25）
            num_inference_steps: int = 25,
            # 可选的噪声参数列表
            sigmas: Optional[List[float]] = None,
            # 最小引导比例（默认为1.0）
            min_guidance_scale: float = 1.0,
            # 最大引导比例（默认为3.0）
            max_guidance_scale: float = 3.0,
            # 帧率（默认为7）
            fps: int = 7,
            # 运动桶的ID（默认为127）
            motion_bucket_id: int = 127,
            # 噪声增强强度（默认为0.02）
            noise_aug_strength: float = 0.02,
            # 可选的解码块大小
            decode_chunk_size: Optional[int] = None,
            # 每个提示生成的视频数量（默认为1）
            num_videos_per_prompt: Optional[int] = 1,
            # 可选的随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量张量
            latents: Optional[torch.Tensor] = None,
            # 输出类型（默认为"pil"）
            output_type: Optional[str] = "pil",
            # 每个步骤结束时的回调函数
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 回调函数输入张量的列表（默认为["latents"]）
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 返回字典格式的标志（默认为True）
            return_dict: bool = True,
# 图像缩放工具
# TODO: 稍后清理
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    # 获取输入图像的高度和宽度
    h, w = input.shape[-2:]
    # 计算高度和宽度的缩放因子
    factors = (h / size[0], w / size[1])

    # 首先，我们必须确定 sigma
    # 取自 skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),  # 计算高度方向的 sigma，确保不小于 0.001
        max((factors[1] - 1.0) / 2.0, 0.001),  # 计算宽度方向的 sigma，确保不小于 0.001
    )

    # 现在计算卷积核大小。对于 3 sigma 得到较好的结果，但速度较慢。Pillow 使用 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # 但他们用两次传递，得到更好的结果。暂时尝试 2 sigmas
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # 确保卷积核大小是奇数
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]  # 如果高度为偶数，增加 1 使其为奇数

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1  # 如果宽度为偶数，增加 1 使其为奇数

    # 对输入图像应用高斯模糊
    input = _gaussian_blur2d(input, ks, sigmas)

    # 使用插值方法调整图像大小
    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output  # 返回调整大小后的图像


def _compute_padding(kernel_size):
    """计算填充元组。"""
    # 4 或 6 个整数:  (左填充, 右填充, 上填充, 下填充)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)  # 如果卷积核大小小于 2，抛出异常
    computed = [k - 1 for k in kernel_size]  # 计算每个维度的填充量

    # 对于偶数大小的卷积核，我们需要不对称填充 :(
    out_padding = 2 * len(kernel_size) * [0]  # 初始化填充数组

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]  # 取最后一个计算的填充量

        pad_front = computed_tmp // 2  # 计算前填充
        pad_rear = computed_tmp - pad_front  # 计算后填充

        out_padding[2 * i + 0] = pad_front  # 设置前填充
        out_padding[2 * i + 1] = pad_rear  # 设置后填充

    return out_padding  # 返回填充元组


def _filter2d(input, kernel):
    # 准备卷积核
    b, c, h, w = input.shape  # 获取输入张量的形状
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)  # 转换卷积核到正确的设备和数据类型

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)  # 扩展卷积核以匹配输入通道数

    height, width = tmp_kernel.shape[-2:]  # 获取卷积核的高度和宽度

    padding_shape: List[int] = _compute_padding([height, width])  # 计算所需的填充
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")  # 对输入进行填充

    # 将卷积核和输入张量重塑以对齐逐元素或批处理参数
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)  # 重塑卷积核
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))  # 重塑输入

    # 用卷积核对张量进行卷积
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)  # 重塑输出为原始形状
    return out  # 返回卷积结果


def _gaussian(window_size: int, sigma):
    # 如果 sigma 是浮点数，转化为张量
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]  # 获取 sigma 的批大小

    # 创建一个 x 张量，用于计算高斯窗口
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    # 如果窗口大小为偶数，调整 x 使其中心偏移
    if window_size % 2 == 0:
        x = x + 0.5  # 对于偶数大小，增加 0.5 以使其居中
    # 计算高斯函数值，使用输入 x 和标准差 sigma
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    
        # 归一化高斯函数值，确保总和为 1，保持维度不变
        return gauss / gauss.sum(-1, keepdim=True)
# 定义一个二次元高斯模糊函数，接受输入图像、卷积核大小和标准差
def _gaussian_blur2d(input, kernel_size, sigma):
    # 检查 sigma 是否为元组，如果是，则将其转换为张量
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    # 如果 sigma 不是元组，则将其转换为与输入数据相同的数据类型
    else:
        sigma = sigma.to(dtype=input.dtype)

    # 从 kernel_size 中提取出高和宽，转换为整数
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    # 获取 sigma 的第一个维度，表示批量大小
    bs = sigma.shape[0]
    # 计算 x 方向的高斯核
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    # 计算 y 方向的高斯核
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    # 在 x 方向上应用滤波
    out_x = _filter2d(input, kernel_x[..., None, :])
    # 在 y 方向上应用滤波
    out = _filter2d(out_x, kernel_y[..., None])

    # 返回模糊处理后的输出
    return out
```