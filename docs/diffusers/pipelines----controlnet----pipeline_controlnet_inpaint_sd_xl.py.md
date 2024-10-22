# `.\diffusers\pipelines\controlnet\pipeline_controlnet_inpaint_sd_xl.py`

```py
# 版权所有 2024 Harutatsu Akiyama, Jinbin Bai 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，
# 否则根据许可证分发的软件是按“原样”提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以了解管理权限和
# 限制的具体条款。

# 导入 inspect 模块以获取对象的签名、源代码等信息
import inspect
# 从 typing 模块导入类型提示工具
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 库用于数值计算
import numpy as np
# 导入 PIL.Image 模块用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能性接口
import torch.nn.functional as F
# 从 transformers 库中导入 CLIP 相关模型和处理器
from transformers import (
    CLIPImageProcessor,  # 图像处理器
    CLIPTextModel,  # 文本模型
    CLIPTextModelWithProjection,  # 带投影的文本模型
    CLIPTokenizer,  # 分词器
    CLIPVisionModelWithProjection,  # 带投影的视觉模型
)

# 从 callbacks 模块导入多管道回调类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 从 image_processor 模块导入图像输入处理器
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 从 loaders 模块导入各种加载器混合类
from ...loaders import (
    FromSingleFileMixin,  # 从单个文件加载
    IPAdapterMixin,  # IP 适配器混合
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散 XL LoRA 加载器混合
    TextualInversionLoaderMixin,  # 文本反转加载器混合
)
# 从 models 模块导入多种模型类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
# 从注意力处理器模块导入不同版本的注意力处理器
from ...models.attention_processor import (
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
# 从调度器模块导入 Karras 扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从 utils 模块导入实用函数和常量
from ...utils import (
    USE_PEFT_BACKEND,  # 使用 PEFT 后端的标志
    deprecate,  # 用于标记过时功能的装饰器
    is_invisible_watermark_available,  # 检查是否可用隐形水印
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 缩放 LoRA 层的函数
    unscale_lora_layers,  # 取消缩放 LoRA 层的函数
)
# 从 torch_utils 模块导入特定的 PyTorch 实用工具函数
from ...utils.torch_utils import is_compiled_module, randn_tensor
# 从 pipeline_utils 模块导入扩散管道和稳定扩散混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从稳定扩散 XL 的输出模块导入管道输出类
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
# 从 multicontrolnet 模块导入多控制网络模型
from .multicontrolnet import MultiControlNetModel

# 如果可用隐形水印，则导入相应的水印处理器
if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor,  # 输入为编码器输出的张量
    generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
    sample_mode: str = "sample"  # 采样模式，默认设置为 "sample"
):
    # 如果 encoder_output 具有 latent_dist 属性并且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中采样并返回结果
        return encoder_output.latent_dist.sample(generator)
    # 如果 encoder_output 具有 latent_dist 属性并且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的众数
        return encoder_output.latent_dist.mode()
    # 如果 encoder_output 具有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 latents 属性
        return encoder_output.latents
    # 如果没有找到有效属性，则抛出异常
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# 示例文档字符串
EXAMPLE_DOC_STRING = """
``` 
```py 
``` 
```py 
``` 
```py 
    # 示例代码，展示如何使用Diffusers库进行图像处理
    Examples:
        ```py
        >>> # 安装必要的库
        >>> # !pip install transformers accelerate
        >>> # 导入所需的模块
        >>> from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
        >>> from diffusers.utils import load_image
        >>> from PIL import Image
        >>> import numpy as np
        >>> import torch

        >>> # 从指定URL加载初始图像
        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
        ... )
        >>> # 将图像调整为1024x1024像素
        >>> init_image = init_image.resize((1024, 1024))

        >>> # 创建一个生成器并设定随机种子
        >>> generator = torch.Generator(device="cpu").manual_seed(1)

        >>> # 从指定URL加载掩码图像
        >>> mask_image = load_image(
        ...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
        ... )
        >>> # 将掩码图像调整为1024x1024像素
        >>> mask_image = mask_image.resize((1024, 1024))

        >>> # 定义一个函数，用于生成Canny边缘图像
        >>> def make_canny_condition(image):
        ...     # 将图像转换为NumPy数组
        ...     image = np.array(image)
        ...     # 应用Canny边缘检测算法
        ...     image = cv2.Canny(image, 100, 200)
        ...     # 增加一个维度，以适应后续操作
        ...     image = image[:, :, None]
        ...     # 复制图像到三个通道，以便转换为RGB格式
        ...     image = np.concatenate([image, image, image], axis=2)
        ...     # 从数组创建图像对象
        ...     image = Image.fromarray(image)
        ...     return image

        >>> # 生成控制图像
        >>> control_image = make_canny_condition(init_image)

        >>> # 从预训练模型加载ControlNet模型
        >>> controlnet = ControlNetModel.from_pretrained(
        ...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        ... )
        >>> # 从预训练模型加载Stable Diffusion管道
        >>> pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # 启用模型CPU卸载，以节省内存
        >>> pipe.enable_model_cpu_offload()

        >>> # 生成图像
        >>> image = pipe(
        ...     "a handsome man with ray-ban sunglasses",
        ...     num_inference_steps=20,
        ...     generator=generator,
        ...     eta=1.0,
        ...     image=init_image,
        ...     mask_image=mask_image,
        ...     control_image=control_image,
        ... ).images[0]  # 从生成的图像列表中提取第一张图像
        ```py 
"""
# 文档字符串，描述该模块的功能或用法


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制的函数
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 重新缩放 `noise_cfg`。基于[Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)中的发现。参见第 3.4 节
    """
    # 计算 noise_pred_text 的标准差，保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算 noise_cfg 的标准差，保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据标准差重缩放来自指导的结果（修复过度曝光）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 按照指导比例混合原始结果，以避免生成“平淡”的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的 noise_cfg
    return noise_cfg


# 定义一个用于文本到图像生成的 Stable Diffusion XL 控制网络插值管道类
class StableDiffusionXLControlNetInpaintPipeline(
    DiffusionPipeline,  # 继承自 DiffusionPipeline
    StableDiffusionMixin,  # 继承自 StableDiffusionMixin
    StableDiffusionXLLoraLoaderMixin,  # 继承自 StableDiffusionXLLoraLoaderMixin
    FromSingleFileMixin,  # 继承自 FromSingleFileMixin
    IPAdapterMixin,  # 继承自 IPAdapterMixin
    TextualInversionLoaderMixin,  # 继承自 TextualInversionLoaderMixin
):
    r"""
    用于使用 Stable Diffusion XL 进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等）的文档，请查阅超类文档。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    # 定义参数说明的文档字符串
        Args:
            vae ([`AutoencoderKL`]):
                定义用于编码和解码图像的变分自编码器模型，将图像转换为潜在表示。
            text_encoder ([`CLIPTextModel`]):
                冻结的文本编码器。Stable Diffusion XL使用
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)的文本部分，
                特别是[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)变体。
            text_encoder_2 ([` CLIPTextModelWithProjection`]):
                第二个冻结文本编码器。Stable Diffusion XL使用
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection)的文本和池部分，
                特别是[laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)变体。
            tokenizer (`CLIPTokenizer`):
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)类的标记器。
            tokenizer_2 (`CLIPTokenizer`):
                第二个[CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)类的标记器。
            unet ([`UNet2DConditionModel`]): 条件U-Net架构，用于去噪编码的图像潜在表示。
            scheduler ([`SchedulerMixin`]):
                用于与`unet`结合使用的调度器，以去噪编码的图像潜在表示。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`]或[`PNDMScheduler`]之一。
        """
    
        # 定义模型组件的顺序，以便进行CPU卸载
        model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    
        # 定义可选组件的列表
        _optional_components = [
            "tokenizer",  # 标记器
            "tokenizer_2",  # 第二个标记器
            "text_encoder",  # 文本编码器
            "text_encoder_2",  # 第二个文本编码器
            "image_encoder",  # 图像编码器
            "feature_extractor",  # 特征提取器
        ]
        # 定义回调张量输入的列表
        _callback_tensor_inputs = [
            "latents",  # 潜在表示
            "prompt_embeds",  # 提示嵌入
            "negative_prompt_embeds",  # 负提示嵌入
            "add_text_embeds",  # 添加的文本嵌入
            "add_time_ids",  # 添加的时间ID
            "negative_pooled_prompt_embeds",  # 负池化提示嵌入
            "add_neg_time_ids",  # 添加的负时间ID
            "mask",  # 掩码
            "masked_image_latents",  # 被掩码的图像潜在表示
        ]
    
        # 初始化方法定义，接收多个模型和参数
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器模型
            text_encoder: CLIPTextModel,  # 文本编码器
            text_encoder_2: CLIPTextModelWithProjection,  # 第二个文本编码器
            tokenizer: CLIPTokenizer,  # 第一个标记器
            tokenizer_2: CLIPTokenizer,  # 第二个标记器
            unet: UNet2DConditionModel,  # 条件U-Net模型
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],  # 控制网络模型或模型列表
            scheduler: KarrasDiffusionSchedulers,  # Karras调度器
            requires_aesthetics_score: bool = False,  # 是否需要美学评分的布尔值
            force_zeros_for_empty_prompt: bool = True,  # 对空提示强制使用零的布尔值
            add_watermarker: Optional[bool] = None,  # 可选的水印标记布尔值
            feature_extractor: Optional[CLIPImageProcessor] = None,  # 可选的特征提取器
            image_encoder: Optional[CLIPVisionModelWithProjection] = None,  # 可选的图像编码器
    # 初始化父类构造函数
        ):
            super().__init__()
    
            # 检查 controlnet 是否为列表或元组，若是则转换为 MultiControlNetModel 实例
            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)
    
            # 注册各个模块以供使用
            self.register_modules(
                # 注册变分自编码器
                vae=vae,
                # 注册文本编码器
                text_encoder=text_encoder,
                # 注册第二文本编码器
                text_encoder_2=text_encoder_2,
                # 注册标记器
                tokenizer=tokenizer,
                # 注册第二标记器
                tokenizer_2=tokenizer_2,
                # 注册联合网络
                unet=unet,
                # 注册控制网络
                controlnet=controlnet,
                # 注册调度器
                scheduler=scheduler,
                # 注册特征提取器
                feature_extractor=feature_extractor,
                # 注册图像编码器
                image_encoder=image_encoder,
            )
            # 将强制零填充的参数注册到配置中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 将需要美学评分的参数注册到配置中
            self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器实例，使用 VAE 缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 创建掩模处理器实例，配置不同的处理参数
            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
            )
            # 创建控制图像处理器实例，配置不同的处理参数
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
    
            # 确定是否添加水印，若未指定则检查是否可用
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 若需要添加水印，则初始化水印器
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                # 否则水印设置为 None
                self.watermark = None
    
        # 从稳定扩散管道复制的函数，用于编码提示
        def encode_prompt(
            # 提示字符串
            prompt: str,
            # 第二个提示字符串，可选
            prompt_2: Optional[str] = None,
            # 设备类型，可选
            device: Optional[torch.device] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: int = 1,
            # 是否进行分类器自由引导
            do_classifier_free_guidance: bool = True,
            # 负面提示字符串，可选
            negative_prompt: Optional[str] = None,
            # 第二个负面提示字符串，可选
            negative_prompt_2: Optional[str] = None,
            # 提示嵌入张量，可选
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入张量，可选
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 经过处理的提示嵌入张量，可选
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 经过处理的负面提示嵌入张量，可选
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # Lora 缩放因子，可选
            lora_scale: Optional[float] = None,
            # 跳过剪辑的参数，可选
            clip_skip: Optional[int] = None,
        # 从稳定扩散管道复制的函数，用于编码图像
    # 定义一个编码图像的函数，接收图像、设备、每个提示的图像数量以及可选的隐藏状态参数
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype
    
        # 检查输入图像是否为张量，若不是，则使用特征提取器将其转换为张量
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
        # 将图像移动到指定设备，并转换为适当的数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 编码图像并获取倒数第二层的隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 根据每个提示的图像数量重复隐藏状态
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 编码全零图像并获取倒数第二层的隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 根据每个提示的图像数量重复隐藏状态
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回编码后的图像和无条件图像的隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 编码图像并获取图像嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 根据每个提示的图像数量重复图像嵌入
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建与图像嵌入相同形状的全零张量作为无条件图像嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)
    
            # 返回编码后的图像嵌入和无条件图像嵌入
            return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的函数
        def prepare_ip_adapter_image_embeds(
            # 函数参数定义，包括适配器图像、图像嵌入、设备、每个提示的图像数量和分类器自由引导的标志
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    # 函数体开始
        ):
            # 初始化用于存储图像嵌入的列表
            image_embeds = []
            # 如果使用分类器自由引导，则初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 检查适配器图像嵌入是否为 None
            if ip_adapter_image_embeds is None:
                # 确保适配器图像为列表形式
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
    
                # 检查适配器图像的长度是否与 IP 适配器数量匹配
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
    
                # 遍历每个适配器图像和相应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 检查是否需要输出隐藏状态
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 对单个适配器图像进行编码，得到嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将单个图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果使用分类器自由引导，添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 如果已有适配器图像嵌入，遍历这些嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果使用分类器自由引导，拆分负和正图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将单个图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化适配器图像嵌入的列表
            ip_adapter_image_embeds = []
            # 遍历每个图像嵌入，进行处理
            for i, single_image_embeds in enumerate(image_embeds):
                # 将每个图像嵌入复制指定次数以生成多图像嵌入
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果使用分类器自由引导，处理负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 将负图像嵌入与正图像嵌入合并
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的嵌入添加到适配器图像嵌入列表中
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回最终的适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的代码
    # 准备额外的参数用于调度器步骤，因为并不是所有调度器都有相同的参数签名
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 取值应在 [0, 1] 之间
    
            # 检查调度器步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
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
    
        # 检查输入图像及其对应的提示和提示嵌入
        def check_image(self, image, prompt, prompt_embeds):
            # 检查图像是否为 PIL 图像类型
            image_is_pil = isinstance(image, PIL.Image.Image)
            # 检查图像是否为 PyTorch 张量类型
            image_is_tensor = isinstance(image, torch.Tensor)
            # 检查图像是否为 NumPy 数组类型
            image_is_np = isinstance(image, np.ndarray)
            # 检查图像是否为 PIL 图像列表
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            # 检查图像是否为 PyTorch 张量列表
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            # 检查图像是否为 NumPy 数组列表
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
            # 如果图像不属于上述任何一种类型，则引发类型错误
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
    
            # 如果图像为 PIL 图像，则图像批量大小为 1
            if image_is_pil:
                image_batch_size = 1
            else:
                # 否则，图像批量大小为图像列表的长度
                image_batch_size = len(image)
    
            # 检查提示是否为字符串类型
            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            # 检查提示是否为列表类型
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            # 如果提示嵌入不为空，则根据其形状确定提示批量大小
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]
    
            # 如果图像批量大小不为 1，且与提示批量大小不匹配，则引发值错误
            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )
    # 定义检查输入参数的函数，包含多个参数
    def check_inputs(
        self,
        prompt,  # 输入的提示文本
        prompt_2,  # 第二个输入的提示文本
        image,  # 输入的图像
        mask_image,  # 输入的掩码图像
        strength,  # 强度参数
        num_inference_steps,  # 推理步骤数
        callback_steps,  # 回调步骤
        output_type,  # 输出类型
        negative_prompt=None,  # 负面提示文本（可选）
        negative_prompt_2=None,  # 第二个负面提示文本（可选）
        prompt_embeds=None,  # 提示的嵌入表示（可选）
        negative_prompt_embeds=None,  # 负面提示的嵌入表示（可选）
        ip_adapter_image=None,  # IP 适配器图像（可选）
        ip_adapter_image_embeds=None,  # IP 适配器图像嵌入（可选）
        pooled_prompt_embeds=None,  # 池化后的提示嵌入（可选）
        negative_pooled_prompt_embeds=None,  # 负面池化提示嵌入（可选）
        controlnet_conditioning_scale=1.0,  # ControlNet 的条件缩放因子
        control_guidance_start=0.0,  # Control 引导的起始值
        control_guidance_end=1.0,  # Control 引导的结束值
        callback_on_step_end_tensor_inputs=None,  # 步骤结束时的回调张量输入（可选）
        padding_mask_crop=None,  # 填充掩码裁剪（可选）
    # 定义准备控制图像的函数，包含多个参数
    def prepare_control_image(
        self,
        image,  # 输入的图像
        width,  # 图像的宽度
        height,  # 图像的高度
        batch_size,  # 批处理大小
        num_images_per_prompt,  # 每个提示的图像数量
        device,  # 设备类型（CPU/GPU）
        dtype,  # 数据类型
        crops_coords,  # 裁剪坐标
        resize_mode,  # 调整大小的模式
        do_classifier_free_guidance=False,  # 是否进行无分类器引导
        guess_mode=False,  # 是否启用猜测模式
    ):
        # 预处理图像，调整大小和裁剪，并转换为指定的数据类型
        image = self.control_image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        ).to(dtype=torch.float32)  # 转换为浮点型
        # 获取图像的批处理大小
        image_batch_size = image.shape[0]

        # 如果图像批处理大小为1，则重复次数为批处理大小
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # 如果图像批处理大小与提示批处理大小相同
            repeat_by = num_images_per_prompt

        # 按指定维度重复图像
        image = image.repeat_interleave(repeat_by, dim=0)

        # 将图像移动到指定的设备和数据类型
        image = image.to(device=device, dtype=dtype)

        # 如果启用无分类器引导且未启用猜测模式，则复制图像
        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)  # 将图像重复连接

        # 返回处理后的图像
        return image

    # 定义准备潜变量的函数，包含多个参数
    def prepare_latents(
        self,
        batch_size,  # 批处理大小
        num_channels_latents,  # 潜变量的通道数
        height,  # 高度
        width,  # 宽度
        dtype,  # 数据类型
        device,  # 设备类型
        generator,  # 随机数生成器
        latents=None,  # 潜变量（可选）
        image=None,  # 输入图像（可选）
        timestep=None,  # 时间步（可选）
        is_strength_max=True,  # 强度是否达到最大值
        add_noise=True,  # 是否添加噪声
        return_noise=False,  # 是否返回噪声
        return_image_latents=False,  # 是否返回图像潜变量
    ):
        # 定义形状，包括批量大小、通道数和缩放后的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器列表长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果图像或时间步长未提供且强度未达到最大值，抛出错误
        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        # 如果需要返回图像潜变量或潜变量为空且强度未达到最大值
        if return_image_latents or (latents is None and not is_strength_max):
            # 将图像转换为指定设备和数据类型
            image = image.to(device=device, dtype=dtype)

            # 如果图像有四个通道，直接赋值给图像潜变量
            if image.shape[1] == 4:
                image_latents = image
            else:
                # 否则通过 VAE 编码图像以获取潜变量
                image_latents = self._encode_vae_image(image=image, generator=generator)
            # 根据批量大小重复潜变量
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # 如果潜变量为空且需要添加噪声
        if latents is None and add_noise:
            # 创建随机噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 如果强度为 1，则将潜变量初始化为噪声，否则将其初始化为图像和噪声的组合
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # 如果强度为最大值，则根据调度器的初始 sigma 缩放潜变量
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:
            # 如果需要添加噪声，将潜变量转换到指定设备
            noise = latents.to(device)
            # 用调度器的初始 sigma 缩放潜变量
            latents = noise * self.scheduler.init_noise_sigma
        else:
            # 创建随机噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 将图像潜变量转换到指定设备
            latents = image_latents.to(device)

        # 创建输出元组，初始仅包含潜变量
        outputs = (latents,)

        # 如果需要返回噪声，则将其添加到输出中
        if return_noise:
            outputs += (noise,)

        # 如果需要返回图像潜变量，则将其添加到输出中
        if return_image_latents:
            outputs += (image_latents,)

        # 返回最终的输出元组
        return outputs
    # 定义一个私有方法，用于编码变分自编码器（VAE）的图像
        def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
            # 获取输入图像的数值类型
            dtype = image.dtype
            # 如果配置要求强制转换数据类型为浮点型
            if self.vae.config.force_upcast:
                # 将图像转换为浮点型
                image = image.float()
                # 将 VAE 模型转换为浮点32位类型
                self.vae.to(dtype=torch.float32)
    
            # 如果生成器是一个列表
            if isinstance(generator, list):
                # 遍历每个图像，编码并获取对应的潜在变量
                image_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(image.shape[0])  # 遍历图像的每一维
                ]
                # 将所有潜在变量在第0维上连接成一个张量
                image_latents = torch.cat(image_latents, dim=0)
            else:
                # 编码整个图像并获取潜在变量
                image_latents = retrieve_latents(self.vae.encode(image), generator=generator)
    
            # 如果配置要求强制转换数据类型为原始类型
            if self.vae.config.force_upcast:
                # 将 VAE 模型恢复为原始数据类型
                self.vae.to(dtype)
    
            # 将潜在变量转换为原始数据类型
            image_latents = image_latents.to(dtype)
            # 将潜在变量乘以缩放因子
            image_latents = self.vae.config.scaling_factor * image_latents
    
            # 返回编码后的潜在变量
            return image_latents
    
        # 定义一个方法，用于准备掩码潜在变量
        def prepare_mask_latents(
            # 接收掩码、被掩盖的图像、批次大小、高度、宽度、数据类型、设备、生成器和分类器自由引导的标志
            self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # 将掩膜调整为与潜在形状相同，以便在连接掩膜与潜在时使用
        # 在转换为数据类型之前进行此操作，以避免在使用 cpu_offload 和半精度时出现问题
        mask = torch.nn.functional.interpolate(
            # 调整掩膜的大小，使其与潜在相匹配
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        # 将掩膜移动到指定设备，并设置数据类型
        mask = mask.to(device=device, dtype=dtype)

        # 为每个提示生成重复的掩膜和掩膜图像潜在，使用与 MPS 友好的方法
        if mask.shape[0] < batch_size:
            # 检查掩膜数量是否可以被批量大小整除
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    # 抛出错误，提示掩膜与批量大小不匹配
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            # 重复掩膜以匹配批量大小
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        # 如果启用分类器自由引导，重复掩膜两次；否则保持不变
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        masked_image_latents = None
        if masked_image is not None:
            # 将掩膜图像移动到指定设备，并设置数据类型
            masked_image = masked_image.to(device=device, dtype=dtype)
            # 对掩膜图像进行 VAE 编码，生成潜在表示
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
            # 检查掩膜图像潜在数量是否可以被批量大小整除
            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        # 抛出错误，提示图像与批量大小不匹配
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                # 重复掩膜图像潜在以匹配批量大小
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            # 如果启用分类器自由引导，重复潜在表示两次；否则保持不变
            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # 将掩膜图像潜在移动到指定设备，以防在与潜在模型输入连接时出现设备错误
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        # 返回处理后的掩膜和掩膜图像潜在
        return mask, masked_image_latents

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps 复制的
    # 获取时间步长，包含推理步数、强度、设备和可选的去噪开始时间
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # 如果没有提供去噪开始时间，则计算初始时间步
        if denoising_start is None:
            # 计算初始时间步，取强度和推理步数的乘积与推理步数中的最小值
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            # 计算开始时间步，确保不小于 0
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            # 如果提供了去噪开始时间，开始时间步设为 0
            t_start = 0

        # 根据调度器的时间步数组，从开始时间步切片获取时间步
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # 如果有去噪开始时间，强度不再重要；
        # 此时强度由去噪开始时间决定
        if denoising_start is not None:
            # 计算离散时间步的截止值
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            # 统计时间步小于截止值的数量，得到推理步数
            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            # 如果调度器为二阶调度器且推理步数为偶数，可能需要加 1
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # 因为每个时间步（最高的时间步除外）被重复，如果推理步数为偶数，
                # 则可能会在去噪步骤中间切分时间步，导致结果不正确。
                # 加 1 确保去噪过程在调度器的二阶导数步骤后结束
                num_inference_steps = num_inference_steps + 1

            # 因为 t_n+1 >= t_n，从结束开始切片获取时间步
            timesteps = timesteps[-num_inference_steps:]
            # 返回时间步和推理步数
            return timesteps, num_inference_steps

        # 返回时间步和从开始时间步减去的推理步数
        return timesteps, num_inference_steps - t_start

    # 定义获取附加时间 ID 的私有方法，参数包括原始大小、裁剪坐标、目标大小、美学分数等
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        dtype,
        text_encoder_projection_dim=None,
    ):
        # 检查配置是否需要美学评分
        if self.config.requires_aesthetics_score:
            # 创建添加时间 ID 列表，包括原始大小、裁剪坐标和美学评分
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            # 创建添加负美学评分 ID 列表，包括原始大小、裁剪坐标和负美学评分
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            # 创建添加时间 ID 列表，包括原始大小、裁剪坐标和目标大小
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            # 创建添加负时间 ID 列表，包括原始大小、裁剪坐标和目标大小
            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

        # 计算通过的添加嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取期望的添加嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查期望的嵌入维度是否大于实际通过的嵌入维度，并且差值是否等于配置的添加时间嵌入维度
        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出错误，提示嵌入向量长度不匹配
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        # 检查期望的嵌入维度是否小于实际通过的嵌入维度，并且差值是否等于配置的添加时间嵌入维度
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出错误，提示嵌入向量长度不匹配
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        # 检查期望的嵌入维度是否与实际通过的嵌入维度不相等
        elif expected_add_embed_dim != passed_add_embed_dim:
            # 抛出错误，提示模型配置不正确
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将添加时间 ID 转换为张量
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 将添加负时间 ID 转换为张量
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        # 返回添加时间 ID 和添加负时间 ID
        return add_time_ids, add_neg_time_ids

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae 中复制
    # 定义一个函数用于提升 VAE 的类型
        def upcast_vae(self):
            # 获取 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为浮点32位数据类型
            self.vae.to(dtype=torch.float32)
            # 检查当前使用的处理器是否为特定版本或类型
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                ),
            )
            # 如果使用 xformers 或 torch_2_0，注意力模块不需要使用浮点32位，可以节省大量内存
            if use_torch_2_0_or_xformers:
                # 将后量化卷积转换为相应的数据类型
                self.vae.post_quant_conv.to(dtype)
                # 将输入卷积层转换为相应的数据类型
                self.vae.decoder.conv_in.to(dtype)
                # 将中间块转换为相应的数据类型
                self.vae.decoder.mid_block.to(dtype)
    
        # 定义属性以获取指导比例
        @property
        def guidance_scale(self):
            # 返回指导比例的值
            return self._guidance_scale
    
        # 定义属性以获取剪辑跳过的值
        @property
        def clip_skip(self):
            # 返回剪辑跳过的值
            return self._clip_skip
    
        # 定义属性以判断是否进行无分类器引导
        # 这里的 `guidance_scale` 类似于 Imagen 论文中方程 (2) 的引导权重 `w`
        # `guidance_scale = 1` 表示不进行分类器无引导。
        @property
        def do_classifier_free_guidance(self):
            # 返回是否进行无分类器引导的布尔值
            return self._guidance_scale > 1
    
        # 定义属性以获取交叉注意力的参数
        @property
        def cross_attention_kwargs(self):
            # 返回交叉注意力的参数
            return self._cross_attention_kwargs
    
        # 定义属性以获取时间步数
        @property
        def num_timesteps(self):
            # 返回时间步数的值
            return self._num_timesteps
    
        # 禁用梯度计算以节省内存
        @torch.no_grad()
        # 用于替换示例文档字符串的装饰器
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的方法，用于处理输入参数并执行相关操作
        def __call__(
            self,  # 方法本身
            prompt: Union[str, List[str]] = None,  # 输入提示，支持字符串或字符串列表，默认为 None
            prompt_2: Optional[Union[str, List[str]]] = None,  # 第二个输入提示，支持字符串或字符串列表，默认为 None
            image: PipelineImageInput = None,  # 输入图像，默认为 None
            mask_image: PipelineImageInput = None,  # 输入掩码图像，默认为 None
            control_image: Union[  # 控制图像，支持单个或多个输入图像
                PipelineImageInput,
                List[PipelineImageInput],
            ] = None,  # 默认为 None
            height: Optional[int] = None,  # 输出图像的高度，默认为 None
            width: Optional[int] = None,  # 输出图像的宽度，默认为 None
            padding_mask_crop: Optional[int] = None,  # 填充掩码裁剪参数，默认为 None
            strength: float = 0.9999,  # 强度参数，默认为 0.9999
            num_inference_steps: int = 50,  # 推理步骤数量，默认为 50
            denoising_start: Optional[float] = None,  # 去噪开始的值，默认为 None
            denoising_end: Optional[float] = None,  # 去噪结束的值，默认为 None
            guidance_scale: float = 5.0,  # 引导缩放因子，默认为 5.0
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 负向提示，支持字符串或字符串列表，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,  # 第二个负向提示，支持字符串或字符串列表，默认为 None
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量，默认为 1
            eta: float = 0.0,  # 影响随机性的参数，默认为 0.0
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器，默认为 None
            latents: Optional[torch.Tensor] = None,  # 潜在变量，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,  # 提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负向提示嵌入，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,  # 输入适配器图像，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,  # 输入适配器图像嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 池化后的提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 池化后的负向提示嵌入，默认为 None
            output_type: Optional[str] = "pil",  # 输出类型，默认为 "pil"
            return_dict: bool = True,  # 是否返回字典，默认为 True
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 交叉注意力参数，默认为 None
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,  # 控制网条件缩放因子，默认为 1.0
            guess_mode: bool = False,  # 猜测模式开关，默认为 False
            control_guidance_start: Union[float, List[float]] = 0.0,  # 控制引导开始值，默认为 0.0
            control_guidance_end: Union[float, List[float]] = 1.0,  # 控制引导结束值，默认为 1.0
            guidance_rescale: float = 0.0,  # 引导重新缩放因子，默认为 0.0
            original_size: Tuple[int, int] = None,  # 原始图像尺寸，默认为 None
            crops_coords_top_left: Tuple[int, int] = (0, 0),  # 裁剪的左上角坐标，默认为 (0, 0)
            target_size: Tuple[int, int] = None,  # 目标图像尺寸，默认为 None
            aesthetic_score: float = 6.0,  # 美学评分，默认为 6.0
            negative_aesthetic_score: float = 2.5,  # 负向美学评分，默认为 2.5
            clip_skip: Optional[int] = None,  # 剪切跳过参数，默认为 None
            callback_on_step_end: Optional[  # 步骤结束时的回调函数，支持多种类型
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,  # 默认为 None
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # 步骤结束时的张量输入，默认为 ["latents"]
            **kwargs,  # 额外的关键字参数
```