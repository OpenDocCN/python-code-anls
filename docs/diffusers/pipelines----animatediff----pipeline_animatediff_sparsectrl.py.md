# `.\diffusers\pipelines\animatediff\pipeline_animatediff_sparsectrl.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（"许可证"）进行许可；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 在许可证下分发是按“原样”基础，
# 不提供任何明示或暗示的保证或条件。
# 请参见许可证以了解管理权限和
# 限制的具体语言。

# 导入inspect模块，用于获取对象的信息
import inspect
# 导入类型提示所需的各种类型
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入numpy库，用于数值计算
import numpy as np
# 导入PIL库，用于图像处理
import PIL
# 导入torch库，用于深度学习计算
import torch
# 从torch.nn.functional导入功能性操作
import torch.nn.functional as F
# 从transformers库导入CLIP相关模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 导入自定义图像处理模块
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入加载器混合类，用于不同类型的模型加载
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入各种模型，包括自动编码器和UNet
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
# 导入稀疏控制网络模型
from ...models.controlnet_sparsectrl import SparseControlNetModel
# 从lora模块导入调整Lora尺度的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从unet_motion_model模块导入运动适配器
from ...models.unets.unet_motion_model import MotionAdapter
# 导入Karras扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 导入一些工具函数
from ...utils import (
    USE_PEFT_BACKEND,  # 用于选择PEFT后端的常量
    logging,           # 导入日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers, # 调整Lora层比例的函数
    unscale_lora_layers, # 反向调整Lora层比例的函数
)
# 导入与Torch相关的实用函数
from ...utils.torch_utils import is_compiled_module, randn_tensor
# 导入视频处理模块
from ...video_processor import VideoProcessor
# 导入FreeInitMixin类
from ..free_init_utils import FreeInitMixin
# 导入扩散管道和稳定扩散的混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入动画扩散管道输出类
from .pipeline_output import AnimateDiffPipelineOutput

# 创建日志记录器，用于模块的日志记录
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串的初始化
EXAMPLE_DOC_STRING = """
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
    # 示例代码，展示如何使用 AnimateDiffSparseControlNetPipeline
        Examples:
            ```py
            # 导入 PyTorch 和相关的 Diffusers 模型
            >>> import torch
            >>> from diffusers import AnimateDiffSparseControlNetPipeline
            >>> from diffusers.models import AutoencoderKL, MotionAdapter, SparseControlNetModel
            >>> from diffusers.schedulers import DPMSolverMultistepScheduler
            >>> from diffusers.utils import export_to_gif, load_image
    
            # 定义模型和适配器的 ID
            >>> model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            >>> motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
            >>> controlnet_id = "guoyww/animatediff-sparsectrl-scribble"
            >>> lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
            >>> vae_id = "stabilityai/sd-vae-ft-mse"
            >>> device = "cuda"  # 设置设备为 GPU
    
            # 从预训练模型加载运动适配器并转移到指定设备
            >>> motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=torch.float16).to(device)
            # 从预训练模型加载控制网络并转移到指定设备
            >>> controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(device)
            # 从预训练模型加载变换编码器并转移到指定设备
            >>> vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(device)
            # 从预训练模型加载调度器，并设置相关参数
            >>> scheduler = DPMSolverMultistepScheduler.from_pretrained(
            ...     model_id,
            ...     subfolder="scheduler",  # 指定子文件夹
            ...     beta_schedule="linear",  # 设置 beta 调度
            ...     algorithm_type="dpmsolver++",  # 设置算法类型
            ...     use_karras_sigmas=True,  # 使用 Karras sigmas
            ... )
            # 从预训练模型加载动画Diff稀疏控制管道并转移到指定设备
            >>> pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
            ...     model_id,
            ...     motion_adapter=motion_adapter,
            ...     controlnet=controlnet,
            ...     vae=vae,
            ...     scheduler=scheduler,
            ...     torch_dtype=torch.float16,
            ... ).to(device)
            # 加载 LORA 权重
            >>> pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")
            # 融合 LORA 权重，设置比例为 1.0
            >>> pipe.fuse_lora(lora_scale=1.0)
    
            # 定义生成图像的提示词
            >>> prompt = "an aerial view of a cyberpunk city, night time, neon lights, masterpiece, high quality"
            # 定义生成图像的负面提示词
            >>> negative_prompt = "low quality, worst quality, letterboxed"
    
            # 定义条件帧的图像文件列表
            >>> image_files = [
            ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-1.png",
            ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-2.png",
            ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-3.png",
            ... ]
            # 定义条件帧的索引
            >>> condition_frame_indices = [0, 8, 15]
            # 加载条件帧的图像
            >>> conditioning_frames = [load_image(img_file) for img_file in image_files]
    
            # 生成视频，并设置相关参数
            >>> video = pipe(
            ...     prompt=prompt,
            ...     negative_prompt=negative_prompt,
            ...     num_inference_steps=25,  # 设定推理步数
            ...     conditioning_frames=conditioning_frames,
            ...     controlnet_conditioning_scale=1.0,  # 设置控制网络条件比例
            ...     controlnet_frame_indices=condition_frame_indices,  # 设置控制网络帧索引
            ...     generator=torch.Generator().manual_seed(1337),  # 设置随机种子
            ... ).frames[0]  # 获取生成的视频帧
            # 导出视频为 GIF 格式
            >>> export_to_gif(video, "output.gif")
            ``` 
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    # 编码器输出的张量
    encoder_output: torch.Tensor, 
    # 可选的随机数生成器
    generator: Optional[torch.Generator] = None, 
    # 采样模式，默认为 "sample"
    sample_mode: str = "sample"
):
    # 如果 encoder_output 有 latent_dist 属性且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从潜在分布中采样并返回
        return encoder_output.latent_dist.sample(generator)
    # 如果 encoder_output 有 latent_dist 属性且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回潜在分布的众数
        return encoder_output.latent_dist.mode()
    # 如果 encoder_output 有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回潜在变量
        return encoder_output.latents
    # 否则，抛出属性错误
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# 用于受控文本到视频生成的管道
class AnimateDiffSparseControlNetPipeline(
    DiffusionPipeline,  # 继承自 DiffusionPipeline
    StableDiffusionMixin,  # 继承自 StableDiffusionMixin
    TextualInversionLoaderMixin,  # 继承自 TextualInversionLoaderMixin
    IPAdapterMixin,  # 继承自 IPAdapterMixin
    StableDiffusionLoraLoaderMixin,  # 继承自 StableDiffusionLoraLoaderMixin
    FreeInitMixin,  # 继承自 FreeInitMixin
):
    r"""
    基于 [SparseCtrl: Adding Sparse Controls
    to Text-to-Video Diffusion Models](https://arxiv.org/abs/2311.16933) 方法的受控文本到视频生成管道。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道通用方法的文档，请查看超类文档（下载、保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器模型。
        text_encoder ([`CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer (`CLIPTokenizer`):
            用于分词的 [`~transformers.CLIPTokenizer`]。
        unet ([`UNet2DConditionModel`]):
            [`UNet2DConditionModel`] 用于创建 UNetMotionModel，以去噪编码的视频潜在变量。
        motion_adapter ([`MotionAdapter`]):
            用于与 `unet` 结合使用以去噪编码视频潜在变量的 [`MotionAdapter`]。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用以去噪编码图像潜在变量的调度器。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
    """

    # 指定模型 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 可选组件列表
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]
    # 回调张量输入列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化类的构造函数
        def __init__(
            # 自动编码器模型，用于数据压缩和重构
            vae: AutoencoderKL,
            # 文本编码器，负责将文本转换为嵌入表示
            text_encoder: CLIPTextModel,
            # 分词器，用于将文本分解为标记
            tokenizer: CLIPTokenizer,
            # 条件生成模型，可以是 UNet2D 或运动模型
            unet: Union[UNet2DConditionModel, UNetMotionModel],
            # 动作适配器，用于处理运动数据
            motion_adapter: MotionAdapter,
            # 稀疏控制网络模型，增强生成的灵活性
            controlnet: SparseControlNetModel,
            # 调度器，控制生成过程中的时间步长
            scheduler: KarrasDiffusionSchedulers,
            # 特征提取器，可选，用于处理图像特征
            feature_extractor: CLIPImageProcessor = None,
            # 图像编码器，可选，用于将图像转换为嵌入
            image_encoder: CLIPVisionModelWithProjection = None,
        ):
            # 调用父类的构造函数
            super().__init__()
            # 检查 UNet 的类型，如果是 UNet2D，则转换为 UNetMotion
            if isinstance(unet, UNet2DConditionModel):
                unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
    
            # 注册多个模块，使其可以在模型中使用
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                motion_adapter=motion_adapter,
                controlnet=controlnet,
                scheduler=scheduler,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            # 计算 VAE 的缩放因子，基于其配置中的通道数
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建视频处理器实例，不进行缩放，使用 VAE 缩放因子
            self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)
            # 创建图像处理器实例，使用 VAE 缩放因子，进行 RGB 转换，不进行归一化
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
    
        # 从 StableDiffusionPipeline 中复制的编码提示方法，参数中的 num_images_per_prompt 更改为 num_videos_per_prompt
        def encode_prompt(
            # 输入的提示文本
            prompt,
            # 设备信息，用于指定计算的设备（如 CPU 或 GPU）
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否进行无分类器引导
            do_classifier_free_guidance,
            # 可选的负面提示文本
            negative_prompt=None,
            # 可选的提示嵌入，预计算的文本嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入，预计算的负面文本嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 Lora 缩放因子，用于调整影响力
            lora_scale: Optional[float] = None,
            # 可选的跳过剪辑的层数
            clip_skip: Optional[int] = None,
        # 从 StableDiffusionPipeline 中复制的编码图像方法
    # 定义一个编码图像的函数，接收图像、设备、每个提示的图像数量及可选的隐藏状态
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数值类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，则使用特征提取器处理图像，返回张量格式
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并设置数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 按照每个提示的图像数量重复隐藏状态
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对无条件图像编码进行处理，创建零张量作为输入
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 同样按照每个提示的图像数量重复无条件隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码的图像隐藏状态和无条件隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 按照每个提示的图像数量重复图像嵌入
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的函数
        def prepare_ip_adapter_image_embeds(
            # 接收适配器图像、图像嵌入、设备、每个提示的图像数量和分类器自由引导的标志
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用无分类器引导，则初始化一个空列表，用于存储负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果 IP 适配器图像嵌入为 None
        if ip_adapter_image_embeds is None:
            # 检查 ip_adapter_image 是否为列表，如果不是，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查 ip_adapter_image 的长度是否与 IP 适配器数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误，说明图像数量与 IP 适配器数量不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个单独的 IP 适配器图像及其对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 检查图像投影层是否为 ImageProjection，以确定输出隐藏状态的需求
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像以获取图像嵌入和负图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用无分类器引导，则将负图像嵌入也添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历已有的 IP 适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用无分类器引导，则将嵌入拆分为负和正
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负图像嵌入添加到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将正图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储最终的 IP 适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历图像嵌入及其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将每个图像嵌入复制 num_images_per_prompt 次
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用无分类器引导，处理负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入与正图像嵌入连接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入转移到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到最终列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回最终的 IP 适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.text_to_video_synthesis/pipeline_text_to_video_synth.TextToVideoSDPipeline 复制
    def decode_latents(self, latents):
        # 按照 VAE 配置的缩放因子调整潜变量
        latents = 1 / self.vae.config.scaling_factor * latents

        # 获取潜变量的形状信息
        batch_size, channels, num_frames, height, width = latents.shape
        # 调整潜变量的维度顺序并重塑为新的形状
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        # 解码潜变量，获取图像
        image = self.vae.decode(latents).sample
        # 重塑图像以形成视频的形状
        video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
        # 将视频转换为 float32，以确保兼容性
        video = video.float()
        # 返回解码后的视频
        return video
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制而来
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备额外的参数用于调度器步骤，因为并非所有调度器具有相同的参数签名
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 并且应该在 [0, 1] 之间
    
        # 检查调度器的步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个空字典用于存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到额外步骤参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到额外步骤参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
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
        image=None,
        controlnet_conditioning_scale: float = 1.0,
        # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image 复制而来
    # 定义一个检查图像的函数，参数包括图像、提示和提示嵌入
    def check_image(self, image, prompt, prompt_embeds):
        # 检查图像是否为 PIL 图像对象
        image_is_pil = isinstance(image, PIL.Image.Image)
        # 检查图像是否为 PyTorch 张量
        image_is_tensor = isinstance(image, torch.Tensor)
        # 检查图像是否为 NumPy 数组
        image_is_np = isinstance(image, np.ndarray)
        # 检查图像是否为 PIL 图像列表
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        # 检查图像是否为 PyTorch 张量列表
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        # 检查图像是否为 NumPy 数组列表
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        # 如果图像不属于任何支持的类型，则抛出类型错误
        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        # 如果图像是 PIL 图像，设置批处理大小为 1
        if image_is_pil:
            image_batch_size = 1
        else:
            # 否则，图像批处理大小为图像的长度
            image_batch_size = len(image)

        # 如果提示不为 None 且是字符串，设置提示批处理大小为 1
        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        # 如果提示为列表，则设置提示批处理大小为列表长度
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        # 如果提示嵌入不为 None，设置提示批处理大小为提示嵌入的第一个维度
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        # 如果图像批处理大小不为 1 且与提示批处理大小不一致，则抛出值错误
        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents 复制的函数
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 根据输入参数构造拉丁特征的形状
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 如果生成器是列表且其长度与批处理大小不匹配，则抛出值错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果拉丁特征为 None，则生成随机的拉丁特征
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果拉丁特征不为 None，则将其移动到指定设备
            latents = latents.to(device)

        # 将初始噪声按调度器所需的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的拉丁特征
        return latents
    # 定义准备图像的函数，接收图像及其尺寸、设备和数据类型作为参数
    def prepare_image(self, image, width, height, device, dtype):
        # 对输入图像进行预处理，调整为指定的高度和宽度
        image = self.control_image_processor.preprocess(image, height=height, width=width)
        # 增加一个维度，并将图像移动到指定设备上，转换为指定数据类型
        controlnet_images = image.unsqueeze(0).to(device, dtype)
        # 获取控制网图像的批大小、帧数、通道数、高度和宽度
        batch_size, num_frames, channels, height, width = controlnet_images.shape

        # TODO: 移除下面这一行，检查控制网图像的最小值和最大值是否在0到1之间
        assert controlnet_images.min() >= 0 and controlnet_images.max() <= 1

        # 如果使用简化的条件嵌入，则进行形状调整和数据标准化
        if self.controlnet.use_simplified_condition_embedding:
            # 调整控制网图像的形状以便于后续处理
            controlnet_images = controlnet_images.reshape(batch_size * num_frames, channels, height, width)
            # 将图像数据从[0, 1]范围映射到[-1, 1]范围
            controlnet_images = 2 * controlnet_images - 1
            # 编码图像并获取条件帧，乘以配置的缩放因子
            conditioning_frames = retrieve_latents(self.vae.encode(controlnet_images)) * self.vae.config.scaling_factor
            # 将条件帧调整为原始批大小和帧数
            conditioning_frames = conditioning_frames.reshape(
                batch_size, num_frames, 4, height // self.vae_scale_factor, width // self.vae_scale_factor
            )
        else:
            # 否则，条件帧直接使用控制网图像
            conditioning_frames = controlnet_images

        # 重新排列维度以适应后续处理，格式为[b, c, f, h, w]
        conditioning_frames = conditioning_frames.permute(0, 2, 1, 3, 4)
        # 返回处理后的条件帧
        return conditioning_frames

    # 定义准备稀疏控制条件的函数，接收条件帧及其它参数
    def prepare_sparse_control_conditioning(
        self,
        conditioning_frames: torch.Tensor,
        num_frames: int,
        controlnet_frame_indices: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 确保条件帧的帧数大于控制网帧索引的数量
        assert conditioning_frames.shape[2] >= len(controlnet_frame_indices)

        # 获取条件帧的批大小、通道数、高度和宽度
        batch_size, channels, _, height, width = conditioning_frames.shape
        # 创建一个零张量作为控制网条件，形状为[batch_size, channels, num_frames, height, width]
        controlnet_cond = torch.zeros((batch_size, channels, num_frames, height, width), dtype=dtype, device=device)
        # 创建一个零张量作为控制网条件掩码
        controlnet_cond_mask = torch.zeros((batch_size, 1, num_frames, height, width), dtype=dtype, device=device)
        # 将条件帧的对应索引值赋值到控制网条件张量中
        controlnet_cond[:, :, controlnet_frame_indices] = conditioning_frames[:, :, : len(controlnet_frame_indices)]
        # 更新控制网条件掩码的对应索引为1
        controlnet_cond_mask[:, :, controlnet_frame_indices] = 1

        # 返回控制网条件和条件掩码
        return controlnet_cond, controlnet_cond_mask

    # 定义一个属性，用于获取引导缩放的值
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义一个属性，用于获取剪辑跳过的值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 这里的 `guidance_scale` 是根据 Imagen 论文的方程（2）定义的引导权重 `w`
    # 当 `guidance_scale = 1` 时，表示没有进行分类器自由引导
    @property
    def do_classifier_free_guidance(self):
        # 检查引导缩放是否大于1，以确定是否进行分类器自由引导
        return self._guidance_scale > 1

    # 定义一个属性，用于获取交叉注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 定义一个属性，用于获取时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 在不计算梯度的情况下执行下面的装饰器，确保效率
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的类方法，接受多个可选参数
        def __call__(
            # 提示文本，可以是字符串或字符串列表，默认为 None
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            # 生成图像的高度，默认为 None
            height: Optional[int] = None,
            # 生成图像的宽度，默认为 None
            width: Optional[int] = None,
            # 每个提示生成的帧数，默认为 16
            num_frames: int = 16,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 指导缩放因子，默认为 7.5
            guidance_scale: float = 7.5,
            # 负提示文本，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的视频数量，默认为 1
            num_videos_per_prompt: int = 1,
            # eta 参数，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个或多个生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 先前的潜在表示，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入表示，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的嵌入表示，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # IP 适配器的图像输入，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # IP 适配器的图像嵌入表示，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 条件帧的图像输入列表，默认为 None
            conditioning_frames: Optional[List[PipelineImageInput]] = None,
            # 输出类型，默认为 "pil"
            output_type: str = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 交叉注意力的参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # ControlNet 的条件缩放因子，默认为 1.0
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # ControlNet 的帧索引，默认为 [0]
            controlnet_frame_indices: List[int] = [0],
            # 是否启用猜测模式，默认为 False
            guess_mode: bool = False,
            # 跳过的 CLIP 步数，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 步骤结束时的回调函数的张量输入列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```