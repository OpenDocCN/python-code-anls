# `.\diffusers\pipelines\animatediff\pipeline_animatediff_video2video.py`

```py
# 版权声明，标明版权持有者和年份
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可声明，说明该文件的使用条件
# Licensed under the Apache License, Version 2.0 (the "License");
# 仅在遵循许可条件下使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则本许可下分发的软件是“按现状”提供的，
# 不提供任何明示或暗示的担保或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取关于权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect  # 导入 inspect 模块，用于检查对象信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示所需的类型

import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入 transformers 库中的 CLIP 相关模型和处理器

from ...image_processor import PipelineImageInput  # 从相对路径导入 PipelineImageInput 类
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器相关的混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel  # 导入各种模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整文本编码器 Lora 缩放的函数
from ...models.unets.unet_motion_model import MotionAdapter  # 从 UNet 运动模型中导入 MotionAdapter 类
from ...schedulers import (  # 从调度器模块导入各种调度器类
    DDIMScheduler,  # 导入 DDIM 调度器
    DPMSolverMultistepScheduler,  # 导入多步 DPM 求解器调度器
    EulerAncestralDiscreteScheduler,  # 导入 Euler 祖先离散调度器
    EulerDiscreteScheduler,  # 导入 Euler 离散调度器
    LMSDiscreteScheduler,  # 导入 LMS 离散调度器
    PNDMScheduler,  # 导入 PNDM 调度器
)
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers  # 导入实用工具函数和变量
from ...utils.torch_utils import randn_tensor  # 从 torch_utils 导入生成随机张量的函数
from ...video_processor import VideoProcessor  # 导入视频处理器类
from ..free_init_utils import FreeInitMixin  # 从相对路径导入 FreeInitMixin 类
from ..free_noise_utils import AnimateDiffFreeNoiseMixin  # 从相对路径导入 AnimateDiffFreeNoiseMixin 类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类
from .pipeline_output import AnimateDiffPipelineOutput  # 导入 AnimateDiffPipelineOutput 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于日志记录 # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """  # 定义一个多行字符串，可能用于示例文档或说明
```  
    # 示例代码，展示如何使用 AnimateDiffVideoToVideoPipeline 处理视频
        Examples:
            ```py
            # 导入所需的库
            >>> import imageio
            >>> import requests
            >>> import torch
            >>> from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
            >>> from diffusers.utils import export_to_gif
            >>> from io import BytesIO
            >>> from PIL import Image
    
            # 从预训练模型加载运动适配器
            >>> adapter = MotionAdapter.from_pretrained(
            ...     "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16
            ... )
            # 从预训练模型加载视频到视频的管道，并将其移动到 GPU
            >>> pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            ...     "SG161222/Realistic_Vision_V5.1_noVAE", motion_adapter=adapter
            ... ).to("cuda")
            # 设置调度器的参数
            >>> pipe.scheduler = DDIMScheduler(
            ...     beta_schedule="linear", steps_offset=1, clip_sample=False, timespace_spacing="linspace"
            ... )
    
            # 定义加载视频的函数
            >>> def load_video(file_path: str):
            ...     images = []  # 初始化一个空列表以存储图像帧
    
            ...     # 检查文件路径是否是 URL
            ...     if file_path.startswith(("http://", "https://")):
            ...         # 如果 file_path 是 URL，发送 GET 请求
            ...         response = requests.get(file_path)
            ...         response.raise_for_status()  # 检查请求是否成功
            ...         content = BytesIO(response.content)  # 将响应内容转为字节流
            ...         vid = imageio.get_reader(content)  # 使用字节流读取视频
            ...     else:
            ...         # 假设是本地文件路径
            ...         vid = imageio.get_reader(file_path)  # 从文件路径读取视频
    
            ...     # 遍历视频中的每一帧
            ...     for frame in vid:
            ...         pil_image = Image.fromarray(frame)  # 将帧转换为 PIL 图像
            ...         images.append(pil_image)  # 将图像添加到列表中
    
            ...     return images  # 返回包含所有图像的列表
    
    
            # 加载视频并传入 URL
            >>> video = load_video(
            ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif"
            ... )
            # 处理视频并生成输出，设置提示语和强度
            >>> output = pipe(
            ...     video=video, prompt="panda playing a guitar, on a boat, in the ocean, high quality", strength=0.5
            ... )
            # 获取处理后的视频帧
            >>> frames = output.frames[0]
            # 将帧导出为 GIF 文件
            >>> export_to_gif(frames, "animation.gif")
            ``` 
# """  # 开始多行字符串注释，通常用于文档说明

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents  # 表示该函数是从指定模块复制的
def retrieve_latents(  # 定义函数 retrieve_latents，接受编码器输出、可选生成器和采样模式参数
    encoder_output: torch.Tensor,  # 编码器输出，类型为 torch.Tensor
    generator: Optional[torch.Generator] = None,  # 可选的随机数生成器，默认值为 None
    sample_mode: str = "sample"  # 采样模式，默认为 "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":  # 检查 encoder_output 是否具有 latent_dist 属性，且采样模式为 "sample"
        return encoder_output.latent_dist.sample(generator)  # 从 latent_dist 中采样并返回
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":  # 检查 encoder_output 是否具有 latent_dist 属性，且采样模式为 "argmax"
        return encoder_output.latent_dist.mode()  # 返回 latent_dist 的众数
    elif hasattr(encoder_output, "latents"):  # 检查 encoder_output 是否具有 latents 属性
        return encoder_output.latents  # 返回 latents
    else:  # 如果上述条件都不满足
        raise AttributeError("Could not access latents of provided encoder_output")  # 抛出属性错误，表示无法访问 latents

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps  # 表示该函数是从指定模块复制的
def retrieve_timesteps(  # 定义函数 retrieve_timesteps，接受调度器和其他可选参数
    scheduler,  # 调度器对象
    num_inference_steps: Optional[int] = None,  # 可选的推理步骤数量，默认值为 None
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备类型，默认值为 None
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表，默认值为 None
    sigmas: Optional[List[float]] = None,  # 可选的 sigma 列表，默认值为 None
    **kwargs,  # 接受其他关键字参数
):
    """  # 开始文档字符串注释
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles  # 调用调度器的 set_timesteps 方法并在调用后获取时间步
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.  # 处理自定义时间步，任何关键字参数将传递给 scheduler.set_timesteps

    Args:  # 参数说明
        scheduler (`SchedulerMixin`):  # 调度器类型
            The scheduler to get timesteps from.  # 从中获取时间步的调度器
        num_inference_steps (`int`):  # 推理步骤数量类型
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`  # 生成样本时使用的扩散步骤数量，如果使用该参数，timesteps 必须为 None
            must be `None`.  # timesteps 必须为 None
        device (`str` or `torch.device`, *optional*):  # 设备类型说明
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.  # 将时间步移动到的设备，如果为 None，则不移动
        timesteps (`List[int]`, *optional*):  # 自定义时间步列表说明
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,  # 自定义时间步，用于覆盖调度器的时间步间隔策略，如果传入该参数
            `num_inference_steps` and `sigmas` must be `None`.  # num_inference_steps 和 sigmas 必须为 None
        sigmas (`List[float]`, *optional*):  # 自定义 sigma 列表说明
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,  # 自定义 sigma，用于覆盖调度器的时间步间隔策略，如果传入该参数
            `num_inference_steps` and `timesteps` must be `None`.  # num_inference_steps 和 timesteps 必须为 None

    Returns:  # 返回值说明
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the  # 返回一个元组，包含调度器的时间步序列和推理步骤数量
        second element is the number of inference steps.  # 第二个元素是推理步骤数量
    """  # 结束文档字符串注释
    if timesteps is not None and sigmas is not None:  # 检查 timesteps 和 sigmas 是否都不为 None
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")  # 抛出值错误，提示只能传递一个参数
    # 检查 timesteps 是否为 None
        if timesteps is not None:
            # 判断 scheduler.set_timesteps 方法是否接受 timesteps 参数
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            # 如果不接受，抛出异常并提示用户
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            # 设置 scheduler 的 timesteps，指定设备和其他参数
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            # 获取设置后的 timesteps
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 如果 sigmas 不为 None
        elif sigmas is not None:
            # 判断 scheduler.set_timesteps 方法是否接受 sigmas 参数
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            # 如果不接受，抛出异常并提示用户
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            # 设置 scheduler 的 sigmas，指定设备和其他参数
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            # 获取设置后的 timesteps
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 如果 timesteps 和 sigmas 都为 None
        else:
            # 设置 scheduler 的 timesteps 为推理步骤的数量，指定设备和其他参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取设置后的 timesteps
            timesteps = scheduler.timesteps
        # 返回 timesteps 和推理步骤的数量
        return timesteps, num_inference_steps
# 定义一个名为 AnimateDiffVideoToVideoPipeline 的类，继承自多个基类
class AnimateDiffVideoToVideoPipeline(
    DiffusionPipeline,  # 继承自 DiffusionPipeline，提供通用的管道功能
    StableDiffusionMixin,  # 继承自 StableDiffusionMixin，提供稳定扩散功能
    TextualInversionLoaderMixin,  # 继承自 TextualInversionLoaderMixin，提供文本反转加载功能
    IPAdapterMixin,  # 继承自 IPAdapterMixin，提供 IP 适配器功能
    StableDiffusionLoraLoaderMixin,  # 继承自 StableDiffusionLoraLoaderMixin，提供 LoRA 加载功能
    FreeInitMixin,  # 继承自 FreeInitMixin，提供自由初始化功能
    AnimateDiffFreeNoiseMixin,  # 继承自 AnimateDiffFreeNoiseMixin，提供动画差异无噪声功能
):
    r"""  # 开始文档字符串，描述管道的功能

    Pipeline for video-to-video generation.  # 视频到视频生成的管道

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).  # 提示查看父类文档以获取通用方法

    The pipeline also inherits the following loading methods:  # 列出管道继承的加载方法
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings  # 文本反转嵌入加载方法
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights  # LoRA 权重加载方法
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights  # LoRA 权重保存方法
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters  # IP 适配器加载方法

    Args:  # 参数说明
        vae ([`AutoencoderKL`]):  # VAE 模型，用于图像的编码和解码
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.  # 描述 VAE 的作用
        text_encoder ([`CLIPTextModel`]):  # 冻结的文本编码器
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).  # 指定文本编码器模型
        tokenizer (`CLIPTokenizer`):  # 文本标记器
            A [`~transformers.CLIPTokenizer`] to tokenize text.  # 描述标记器的功能
        unet ([`UNet2DConditionModel`]):  # 用于生成去噪的 UNet 模型
            A [`UNet2DConditionModel`] used to create a UNetMotionModel to denoise the encoded video latents.  # 描述 UNet 的作用
        motion_adapter ([`MotionAdapter`]):  # 动作适配器，用于视频去噪
            A [`MotionAdapter`] to be used in combination with `unet` to denoise the encoded video latents.  # 描述动作适配器的作用
        scheduler ([`SchedulerMixin`]):  # 调度器，用于图像去噪
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of  # 描述调度器的功能和选项
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].  # 可选的调度器类型
    """  # 结束文档字符串

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"  # 定义模型在 CPU 上的卸载顺序
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]  # 定义可选组件的列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]  # 定义回调张量输入的列表

    def __init__(  # 初始化方法
        self,  # 当前实例
        vae: AutoencoderKL,  # VAE 模型参数
        text_encoder: CLIPTextModel,  # 文本编码器参数
        tokenizer: CLIPTokenizer,  # 标记器参数
        unet: UNet2DConditionModel,  # UNet 模型参数
        motion_adapter: MotionAdapter,  # 动作适配器参数
        scheduler: Union[  # 调度器参数，支持多种类型
            DDIMScheduler,  # DDIM 调度器
            PNDMScheduler,  # PNDM 调度器
            LMSDiscreteScheduler,  # LMS 离散调度器
            EulerDiscreteScheduler,  # 欧拉离散调度器
            EulerAncestralDiscreteScheduler,  # 欧拉祖先离散调度器
            DPMSolverMultistepScheduler,  # DPM 多步调度器
        ],
        feature_extractor: CLIPImageProcessor = None,  # 特征提取器参数，默认为 None
        image_encoder: CLIPVisionModelWithProjection = None,  # 图像编码器参数，默认为 None
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 检查传入的 unet 是否为 UNet2DConditionModel 类型
        if isinstance(unet, UNet2DConditionModel):
            # 将 UNet2DConditionModel 转换为 UNetMotionModel
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

        # 注册各个模块到当前对象中
        self.register_modules(
            # 注册变分自编码器
            vae=vae,
            # 注册文本编码器
            text_encoder=text_encoder,
            # 注册分词器
            tokenizer=tokenizer,
            # 注册 UNet 模型
            unet=unet,
            # 注册运动适配器
            motion_adapter=motion_adapter,
            # 注册调度器
            scheduler=scheduler,
            # 注册特征提取器
            feature_extractor=feature_extractor,
            # 注册图像编码器
            image_encoder=image_encoder,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建视频处理器，使用计算出的缩放因子
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor)

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制，参数 num_images_per_prompt 改为 num_videos_per_prompt
    def encode_prompt(
        self,
        # 提示文本
        prompt,
        # 设备类型
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否进行分类器自由引导
        do_classifier_free_guidance,
        # 可选的负面提示文本
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 LORA 缩放因子
        lora_scale: Optional[float] = None,
        # 可选的跳过 CLIP 层数
        clip_skip: Optional[int] = None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入的图像不是张量，则通过特征提取器处理
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像移动到指定设备并设置数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 获取图像编码的隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 重复隐藏状态以匹配每个提示的图像数量
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 获取无条件图像的隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 重复无条件隐藏状态以匹配每个提示的图像数量
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回有条件和无条件的图像隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 获取图像嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 重复图像嵌入以匹配每个提示的图像数量
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建与图像嵌入相同形状的零张量作为无条件嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回有条件和无条件的图像嵌入
            return image_embeds, uncond_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制
    def prepare_ip_adapter_image_embeds(
        # 输入的适配器图像
        self, ip_adapter_image, 
        # 输入的适配器图像嵌入
        ip_adapter_image_embeds, 
        # 设备类型
        device, 
        # 每个提示生成的图像数量
        num_images_per_prompt, 
        # 是否进行分类器自由引导
        do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用分类器自由引导，则初始化一个空列表，用于存储负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果输入适配器图像不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像的长度是否与 IP 适配器的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不相同，则抛出值错误
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个单独的输入适配器图像及其对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断当前图像投影层是否为 ImageProjection 类型
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个适配器图像，获取图像嵌入和负图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到列表中，并在第一维增加维度
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用分类器自由引导，则将负图像嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历输入适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用分类器自由引导，则将图像嵌入分成两部分
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负图像嵌入添加到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储最终的输入适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历每个图像嵌入
        for i, single_image_embeds in enumerate(image_embeds):
            # 根据每个提示需要生成的图像数量，复制图像嵌入
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用分类器自由引导，则处理负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入与正图像嵌入拼接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备上
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到最终列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回最终的输入适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 编码视频的方法，接受视频、生成器和解码块大小参数
    def encode_video(self, video, generator, decode_chunk_size: int = 16) -> torch.Tensor:
        # 初始化一个空列表，用于存储潜在表示
        latents = []
        # 按照解码块大小遍历视频
        for i in range(0, len(video), decode_chunk_size):
            # 获取当前块的视频帧
            batch_video = video[i : i + decode_chunk_size]
            # 编码当前块的视频帧并检索潜在表示
            batch_video = retrieve_latents(self.vae.encode(batch_video), generator=generator)
            # 将编码结果添加到潜在表示列表中
            latents.append(batch_video)
        # 将所有潜在表示拼接成一个张量并返回
        return torch.cat(latents)

    # 从 diffusers.pipelines.animatediff.pipeline_animatediff.AnimateDiffPipeline.decode_latents 复制的代码
    # 解码潜在变量，生成视频
    def decode_latents(self, latents, decode_chunk_size: int = 16):
        # 将潜在变量按比例缩放
        latents = 1 / self.vae.config.scaling_factor * latents

        # 获取潜在变量的形状信息：批量大小、通道数、帧数、高度和宽度
        batch_size, channels, num_frames, height, width = latents.shape
        # 调整潜在变量的维度顺序，并展平帧维度
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        # 初始化视频列表以存储解码后的帧
        video = []
        # 逐批解码潜在变量
        for i in range(0, latents.shape[0], decode_chunk_size):
            # 提取当前批次的潜在变量
            batch_latents = latents[i : i + decode_chunk_size]
            # 使用 VAE 解码当前批次的潜在变量，获取样本
            batch_latents = self.vae.decode(batch_latents).sample
            # 将解码后的帧添加到视频列表中
            video.append(batch_latents)

        # 将所有解码后的帧沿着第一个维度拼接
        video = torch.cat(video)
        # 重塑视频的形状为 (批量大小, 帧数, 其他维度)，并调整维度顺序
        video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
        # 转换视频数据为 float32 类型，确保与 bfloat16 兼容且不会造成显著开销
        video = video.float()
        # 返回解码后的视频数据
        return video

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的函数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外关键字参数，因为并非所有调度器的签名相同
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 并且应该在 [0, 1] 之间

        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤的关键字参数字典
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果接受 eta，添加到额外参数字典中
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受 generator，添加到额外参数字典中
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 检查输入的有效性
    def check_inputs(
        self,
        prompt,
        strength,
        height,
        width,
        video=None,
        latents=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    # 获取时间步
    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # 使用 init_timestep 获取原始时间步
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步，确保不小于 0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从时间步中提取有效时间步
        timesteps = timesteps[t_start * self.scheduler.order :]

        # 返回有效时间步和剩余的推理步骤数
        return timesteps, num_inference_steps - t_start

    # 准备潜在变量
    def prepare_latents(
        self,
        video,
        height,
        width,
        num_channels_latents,
        batch_size,
        timestep,
        dtype,
        device,
        generator,
        latents=None,
        decode_chunk_size: int = 16,
    # 获取引导比例
    @property
    def guidance_scale(self):
        # 返回引导比例的值
        return self._guidance_scale

    @property
    # 定义一个方法，返回当前实例的 clip_skip 属性
    def clip_skip(self):
        return self._clip_skip

    # 这里定义了 `guidance_scale`，它与 Imagen 论文中公式 (2) 的分类权重 `w` 类似
    # `guidance_scale = 1` 表示不进行无分类器引导
    @property
    def do_classifier_free_guidance(self):
        # 返回一个布尔值，指示是否进行无分类器引导，依据是 guidance_scale 是否大于 1
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        # 返回跨注意力的关键字参数
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        # 返回时间步数
        return self._num_timesteps

    # 禁用梯度计算，以节省内存和计算资源
    @torch.no_grad()
    def __call__(
        # 定义调用方法的参数，允许输入视频列表，提示信息，图像的高度和宽度等
        video: List[List[PipelineImageInput]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        # 设置默认的推理步骤数为 50
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        # 默认的引导比例为 7.5
        guidance_scale: float = 7.5,
        # 默认的强度参数为 0.8
        strength: float = 0.8,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的视频数量默认为 1
        num_videos_per_prompt: Optional[int] = 1,
        # 默认的 eta 值为 0.0
        eta: float = 0.0,
        # 随机数生成器，支持单个或多个生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型默认为 "pil"
        output_type: Optional[str] = "pil",
        # 返回字典的布尔标志，默认为 True
        return_dict: bool = True,
        # 允许输入跨注意力的关键字参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # clip_skip 的可选参数
        clip_skip: Optional[int] = None,
        # 逐步结束时的回调函数
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 逐步结束时的张量输入回调名称列表，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 解码时的块大小，默认为 16
        decode_chunk_size: int = 16,
```