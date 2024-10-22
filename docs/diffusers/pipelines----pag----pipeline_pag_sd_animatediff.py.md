# `.\diffusers\pipelines\pag\pipeline_pag_sd_animatediff.py`

```py
# 版权声明，标明版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版授权（“许可证”）；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件按“原样”分发，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 请参见许可证以获取管理权限和
# 限制的具体条款。

import inspect  # 导入 inspect 模块，用于获取活跃对象的信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示，用于增强代码的可读性和可维护性

import torch  # 导入 PyTorch 库，用于深度学习和张量计算
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入 Transformers 库中的 CLIP 相关模型和处理器

from ...image_processor import PipelineImageInput  # 从自定义模块导入图像处理输入类
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器混合类，用于处理不同模型加载
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel  # 导入不同模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入函数，用于调整文本编码器的 LoRA 比例
from ...models.unets.unet_motion_model import MotionAdapter  # 导入运动适配器类，用于处理动态模型
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器类
from ...utils import (  # 导入多个工具函数和常量
    USE_PEFT_BACKEND,  # 用于指示是否使用 PEFT 后端的常量
    logging,  # 导入日志模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入用于缩放 LoRA 层的函数
    unscale_lora_layers,  # 导入用于取消缩放 LoRA 层的函数
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入生成随机张量的函数
from ...video_processor import VideoProcessor  # 导入视频处理器类
from ..animatediff.pipeline_output import AnimateDiffPipelineOutput  # 导入动画扩散管道输出类
from ..free_init_utils import FreeInitMixin  # 导入自由初始化混合类
from ..free_noise_utils import AnimateDiffFreeNoiseMixin  # 导入动画扩散自由噪声混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类
from .pag_utils import PAGMixin  # 导入 PAG 混合类

logger = logging.get_logger(__name__)  # 创建一个记录器，用于日志记录，使用模块的名称
EXAMPLE_DOC_STRING = """  # 定义示例文档字符串的开始
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
    Examples:
        ```py
        # 导入 PyTorch 库
        >>> import torch
        # 从 diffusers 库导入相关的类
        >>> from diffusers import AnimateDiffPAGPipeline, MotionAdapter, DDIMScheduler
        # 从 diffusers.utils 导入 GIF 导出工具
        >>> from diffusers.utils import export_to_gif

        # 定义模型的 ID，用于加载预训练模型
        >>> model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        # 定义运动适配器的 ID，用于加载相应的适配器
        >>> motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-2"
        # 从预训练模型加载运动适配器
        >>> motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id)
        # 从预训练模型加载调度器，设置调度参数
        >>> scheduler = DDIMScheduler.from_pretrained(
        ...     model_id, subfolder="scheduler", beta_schedule="linear", steps_offset=1, clip_sample=False
        ... )
        # 从预训练模型加载动画差异管道，并将其移动到 CUDA 设备
        >>> pipe = AnimateDiffPAGPipeline.from_pretrained(
        ...     model_id,
        ...     motion_adapter=motion_adapter,
        ...     scheduler=scheduler,
        ...     pag_applied_layers=["mid"],
        ...     torch_dtype=torch.float16,
        ... ).to("cuda")

        # 生成视频，设置提示词和参数
        >>> video = pipe(
        ...     prompt="car, futuristic cityscape with neon lights, street, no human",
        ...     negative_prompt="low quality, bad quality",
        ...     num_inference_steps=25,
        ...     guidance_scale=6.0,
        ...     pag_scale=3.0,
        ...     generator=torch.Generator().manual_seed(42),
        ... ).frames[0]  # 获取生成的第一帧

        # 导出生成的视频为 GIF 格式
        >>> export_to_gif(video, "animatediff_pag.gif")
        ```
# 定义 AnimateDiffPAGPipeline 类，继承自多个基类以实现文本到视频的生成
class AnimateDiffPAGPipeline(
    # 继承自 DiffusionPipeline 基类
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin 以便于稳定扩散相关功能
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin 用于加载文本反演嵌入
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin 用于加载 IP 适配器
    IPAdapterMixin,
    # 继承自 StableDiffusionLoraLoaderMixin 用于处理 LoRA 权重
    StableDiffusionLoraLoaderMixin,
    # 继承自 FreeInitMixin 用于初始化功能
    FreeInitMixin,
    # 继承自 AnimateDiffFreeNoiseMixin 以处理动画差异的噪声
    AnimateDiffFreeNoiseMixin,
    # 继承自 PAGMixin 用于应用 Perturbed Attention Guidance
    PAGMixin,
):
    r"""
    文本到视频生成的管道，使用
    [AnimateDiff](https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff) 和 [Perturbed Attention
    Guidance](https://huggingface.co/docs/diffusers/en/using-diffusers/pag)。
    
    该模型继承自 [`DiffusionPipeline`]。请查阅超类文档，了解所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。
    
    该管道还继承了以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    
    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器（VAE）模型，用于编码和解码图像的潜在表示。
        text_encoder ([`CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer (`CLIPTokenizer`):
            [`~transformers.CLIPTokenizer`] 用于文本标记化。
        unet ([`UNet2DConditionModel`]):
            [`UNet2DConditionModel`] 用于创建 UNetMotionModel 来去噪编码的视频潜在。
        motion_adapter ([`MotionAdapter`]):
            [`MotionAdapter`]，与 `unet` 结合使用以去噪编码的视频潜在。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用以去噪编码的图像潜在的调度器，可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]
    # 定义需要回调的张量输入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # 初始化方法，定义构造函数参数
    def __init__(
        # VAE 模型，用于图像的编码和解码
        vae: AutoencoderKL,
        # 文本编码器，用于处理输入文本
        text_encoder: CLIPTextModel,
        # 文本标记化器
        tokenizer: CLIPTokenizer,
        # UNet 模型，处理条件模型或运动模型
        unet: Union[UNet2DConditionModel, UNetMotionModel],
        # 动作适配器，用于去噪处理
        motion_adapter: MotionAdapter,
        # 调度器，用于模型的调度控制
        scheduler: KarrasDiffusionSchedulers,
        # 可选的特征提取器，默认为 None
        feature_extractor: CLIPImageProcessor = None,
        # 可选的图像编码器，默认为 None
        image_encoder: CLIPVisionModelWithProjection = None,
        # 应用的 PAG 层，可以是字符串或字符串列表
        pag_applied_layers: Union[str, List[str]] = "mid_block.*attn1",  # ["mid"], ["down_blocks.1"]
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 检查给定的 unet 是否为 UNet2DConditionModel 类型
        if isinstance(unet, UNet2DConditionModel):
            # 从 UNet2DConditionModel 创建 UNetMotionModel 实例，并传入 motion_adapter
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

        # 注册多个模块，便于后续使用
        self.register_modules(
            # 注册变换自编码器模块
            vae=vae,
            # 注册文本编码器模块
            text_encoder=text_encoder,
            # 注册分词器模块
            tokenizer=tokenizer,
            # 注册 UNet 模块
            unet=unet,
            # 注册运动适配器模块
            motion_adapter=motion_adapter,
            # 注册调度器模块
            scheduler=scheduler,
            # 注册特征提取器模块
            feature_extractor=feature_extractor,
            # 注册图像编码器模块
            image_encoder=image_encoder,
        )
        # 计算 VAE 的缩放因子，基于 VAE 配置中的块输出通道数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化视频处理器，不进行缩放，并设置 VAE 缩放因子
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)

        # 设置应用于 PAG 的层
        self.set_pag_applied_layers(pag_applied_layers)

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制而来，num_images_per_prompt -> num_videos_per_prompt
    def encode_prompt(
        self,
        # 输入的提示文本
        prompt,
        # 指定设备（CPU 或 GPU）
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否进行分类器自由引导
        do_classifier_free_guidance,
        # 可选的负面提示文本
        negative_prompt=None,
        # 可选的提示嵌入张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 Lora 缩放因子
        lora_scale: Optional[float] = None,
        # 可选的剪辑跳过参数
        clip_skip: Optional[int] = None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制而来
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入的图像不是张量，则进行特征提取
        if not isinstance(image, torch.Tensor):
            # 使用特征提取器将图像转换为张量，并返回其像素值
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像张量移动到指定设备，并设置数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 获取图像编码器的隐藏状态的倒数第二个输出
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 根据每个提示的图像数量重复隐藏状态
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 获取无条件图像的隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 同样重复无条件隐藏状态
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回图像和无条件图像的隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 获取图像编码器的图像嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 根据每个提示的图像数量重复图像嵌入
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建与图像嵌入形状相同的全零无条件图像嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回图像嵌入和无条件图像嵌入
            return image_embeds, uncond_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制而来
    def prepare_ip_adapter_image_embeds(
        # 输入的适配器图像
        self, ip_adapter_image, 
        # 输入的适配器图像嵌入
        ip_adapter_image_embeds, 
        # 指定设备（CPU 或 GPU）
        device, 
        # 每个提示生成的图像数量
        num_images_per_prompt, 
        # 是否进行分类器自由引导
        do_classifier_free_guidance
    # 处理图像嵌入和分类器自由引导的逻辑
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用分类器自由引导，初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果输入适配器图像嵌入为空
            if ip_adapter_image_embeds is None:
                # 如果输入适配器图像不是列表，则将其转换为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
                # 检查输入适配器图像的数量是否与 IP 适配器的数量一致
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    # 抛出值错误，说明输入不匹配
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
                # 遍历输入适配器图像和对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 确定输出隐藏状态的标志
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码单个图像，获取图像嵌入和负图像嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
                    # 将单个图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用分类器自由引导，将负图像嵌入添加到列表中
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历已有的图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用分类器自由引导，将嵌入分成负图像和图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化适配器图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入列表，并根据每个提示的图像数量进行重复
            for i, single_image_embeds in enumerate(image_embeds):
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用分类器自由引导，将负图像嵌入重复并连接
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
                # 将图像嵌入移至指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的嵌入添加到适配器图像嵌入列表中
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回最终的适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.animatediff.pipeline_animatediff.AnimateDiffPipeline.decode_latents 复制的代码
    # 解码潜在表示，返回解码后的视频张量
    def decode_latents(self, latents, decode_chunk_size: int = 16):
        # 根据配置的缩放因子对潜在表示进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents
    
        # 获取潜在表示的批次大小、通道数、帧数、高度和宽度
        batch_size, channels, num_frames, height, width = latents.shape
        # 重新排列和调整潜在表示的形状，方便后续处理
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    
        video = []  # 初始化视频列表以存储解码结果
        # 按照解码块大小遍历潜在表示
        for i in range(0, latents.shape[0], decode_chunk_size):
            # 获取当前块的潜在表示
            batch_latents = latents[i : i + decode_chunk_size]
            # 解码当前块的潜在表示，并提取样本
            batch_latents = self.vae.decode(batch_latents).sample
            # 将解码后的块添加到视频列表
            video.append(batch_latents)
    
        # 将所有解码块连接成一个张量
        video = torch.cat(video)
        # 重新调整视频张量的形状，便于后续处理
        video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
        # 将视频张量转换为 float32 类型，以确保兼容性
        video = video.float()
        # 返回解码后的视频张量
        return video
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为不同调度器的参数签名不同
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会被忽略
        # eta 对应 DDIM 论文中的 η，范围应在 [0, 1] 之间
    
        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}  # 初始化额外参数字典
        if accepts_eta:
            # 如果接受 eta，添加到额外参数字典中
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受 generator，添加到额外参数字典中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs
    
    # 从 diffusers.pipelines.pia.pipeline_pia.PIAPipeline.check_inputs 复制
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查高度和宽度是否都是 8 的倍数
        if height % 8 != 0 or width % 8 != 0:
            # 抛出异常，如果条件不满足，显示当前高度和宽度
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调输入是否存在且不全在已注册的回调输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 抛出异常，如果回调输入不在已注册的输入中
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了 prompt 和 prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            # 抛出异常，提示只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否两个参数都未提供
        elif prompt is None and prompt_embeds is None:
            # 抛出异常，提示必须提供其中一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 类型是否为字符串或列表
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出异常，显示实际类型
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 negative_prompt 和 negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出异常，提示只能提供其中一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否存在
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            # 检查它们的形状是否相同
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出异常，显示两个参数的形状不一致
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查是否同时提供了 ip_adapter_image 和 ip_adapter_image_embeds
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            # 抛出异常，提示只能提供其中一个
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        # 检查 ip_adapter_image_embeds 是否存在
        if ip_adapter_image_embeds is not None:
            # 检查其类型是否为列表
            if not isinstance(ip_adapter_image_embeds, list):
                # 抛出异常，显示实际类型
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            # 检查列表中第一个元素的维度是否为 3D 或 4D
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                # 抛出异常，显示实际维度
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # 从 diffusers.pipelines.animatediff.pipeline_animatediff.AnimateDiffPipeline.prepare_latents 复制的代码
    # 准备潜在变量的方法，接收多个参数以配置潜在变量的生成
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 如果启用了 FreeNoise，按照 [FreeNoise](https://arxiv.org/abs/2310.15169) 的公式 (7) 生成潜在变量
        if self.free_noise_enabled:
            # 调用 _prepare_latents_free_noise 方法生成潜在变量
            latents = self._prepare_latents_free_noise(
                batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents
            )

        # 检查生成器的类型和数量是否与请求的批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果生成器列表的长度与批量大小不匹配，则抛出错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 定义潜在变量的形状，依据批量大小和其他参数计算
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        # 如果未提供潜在变量，则生成随机的潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，则将其转移到指定设备
            latents = latents.to(device)

        # 按照调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在变量
        return latents

    # 定义一个属性，返回引导尺度
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义一个属性，返回剪切跳过的值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 定义一个属性，判断是否执行无分类器引导
    # `guidance_scale` 类似于 Imagen 论文中公式 (2) 的引导权重 `w`
    # `guidance_scale = 1` 表示不进行分类器自由引导
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # 定义一个属性，返回交叉注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 定义一个属性，返回时间步数的数量
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 禁用梯度计算，以减少内存消耗和提高计算效率
    @torch.no_grad()
    # 替换示例文档字符串为预定义的文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的特殊方法，用于执行某些操作
    def __call__(
        # 接收的提示信息，可以是字符串或字符串列表，默认为 None
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        # 要生成的帧数，默认为 16
        num_frames: Optional[int] = 16,
        # 输出图像的高度，默认为 None
        height: Optional[int] = None,
        # 输出图像的宽度，默认为 None
        width: Optional[int] = None,
        # 推理步骤的数量，默认为 50
        num_inference_steps: int = 50,
        # 指导比例，默认为 7.5，影响生成图像的质量
        guidance_scale: float = 7.5,
        # 负向提示信息，可以是字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的视频数量，默认为 1
        num_videos_per_prompt: Optional[int] = 1,
        # 控制噪声的强度，默认为 0.0
        eta: float = 0.0,
        # 随机数生成器，可以是单个或多个 torch.Generator，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 生成的潜在变量，可以是 torch.Tensor，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 提示的嵌入表示，可以是 torch.Tensor，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负向提示的嵌入表示，可以是 torch.Tensor，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 图像输入的适配器图像，默认为 None
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 适配器图像的嵌入表示，可以是 torch.Tensor 列表，默认为 None
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型，默认为 "pil"，表示返回 PIL 图像
        output_type: Optional[str] = "pil",
        # 是否返回字典格式，默认为 True
        return_dict: bool = True,
        # 交叉注意力的额外参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 跳过的 CLIP 层数，默认为 None
        clip_skip: Optional[int] = None,
        # 每步结束时的回调函数，默认为 None
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 每步结束时回调时传入的张量输入的名称，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 解码时的块大小，默认为 16
        decode_chunk_size: int = 16,
        # PAG 的缩放比例，默认为 3.0
        pag_scale: float = 3.0,
        # PAG 的自适应缩放，默认为 0.0
        pag_adaptive_scale: float = 0.0,
```