# `.\diffusers\pipelines\controlnet_xs\pipeline_controlnet_xs_sd_xl.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则根据许可证分发的软件
# 是按“原样”基础提供的，不附带任何明示或暗示的担保或条件。
# 请参见许可证以获取有关权限和限制的特定语言。

import inspect  # 导入 inspect 模块，用于获取对象的信息
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 从 typing 模块导入类型注解

import numpy as np  # 导入 numpy 库，通常用于数值计算
import PIL.Image  # 导入 PIL 的 Image 模块，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块，提供各种函数
from transformers import (  # 从 transformers 库导入所需的类和函数
    CLIPImageProcessor,  # 导入 CLIP 图像处理器
    CLIPTextModel,  # 导入 CLIP 文本模型
    CLIPTextModelWithProjection,  # 导入带投影的 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 分词器
)

from diffusers.utils.import_utils import is_invisible_watermark_available  # 导入检查水印可用性的工具

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入回调相关的类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关的类
from ...loaders import FromSingleFileMixin, StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器相关的混合类
from ...models import AutoencoderKL, ControlNetXSAdapter, UNet2DConditionModel, UNetControlNetXSModel  # 导入模型类
from ...models.attention_processor import (  # 从注意力处理器模块导入类
    AttnProcessor2_0,  # 导入版本 2.0 的注意力处理器
    XFormersAttnProcessor,  # 导入 XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 LoRA 规模的文本编码器函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 从 utils 模块导入工具函数和常量
    USE_PEFT_BACKEND,  # 导入用于 PEFT 后端的常量
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放 LoRA 层的函数
    unscale_lora_layers,  # 导入取消缩放 LoRA 层的函数
)
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor  # 导入与 PyTorch 相关的工具函数
from ..pipeline_utils import DiffusionPipeline  # 从管道工具导入 DiffusionPipeline 类
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput  # 从稳定扩散 XL 模块导入管道输出类


if is_invisible_watermark_available():  # 如果隐形水印可用
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker  # 导入 StableDiffusionXLWatermarker 类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，命名为模块名

EXAMPLE_DOC_STRING = """  # 示例文档字符串，通常用于说明函数或类的用法


```  # 文档字符串结束标志
    # 示例代码
        Examples:
            ```py
            >>> # !pip install opencv-python transformers accelerate  # 安装所需的库
            >>> from diffusers import StableDiffusionXLControlNetXSPipeline, ControlNetXSAdapter, AutoencoderKL  # 导入所需的类
            >>> from diffusers.utils import load_image  # 导入加载图像的工具
            >>> import numpy as np  # 导入 NumPy 库以进行数组操作
            >>> import torch  # 导入 PyTorch 库以进行深度学习
    
            >>> import cv2  # 导入 OpenCV 库以进行计算机视觉处理
            >>> from PIL import Image  # 导入 PIL 库以处理图像
    
            >>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"  # 设置生成图像的提示语
            >>> negative_prompt = "low quality, bad quality, sketches"  # 设置生成图像的负面提示
    
            >>> # 下载一张图像
            >>> image = load_image(  # 使用工具加载指定 URL 的图像
            ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            ... )
    
            >>> # 初始化模型和管道
            >>> controlnet_conditioning_scale = 0.5  # 设置 ControlNet 的条件缩放因子
            >>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)  # 加载预训练的自编码器模型
            >>> controlnet = ControlNetXSAdapter.from_pretrained(  # 加载预训练的 ControlNet 模型
            ...     "UmerHA/Testing-ConrolNetXS-SDXL-canny", torch_dtype=torch.float16
            ... )
            >>> pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(  # 加载 Stable Diffusion 管道
            ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
            ... )
            >>> pipe.enable_model_cpu_offload()  # 启用 CPU 内存卸载以优化内存使用
    
            >>> # 获取 Canny 图像
            >>> image = np.array(image)  # 将图像转换为 NumPy 数组
            >>> image = cv2.Canny(image, 100, 200)  # 应用 Canny 边缘检测
            >>> image = image[:, :, None]  # 在第三维添加一个维度以便于后续处理
            >>> image = np.concatenate([image, image, image], axis=2)  # 复制图像以创建 RGB 格式
            >>> canny_image = Image.fromarray(image)  # 将数组转换为 PIL 图像
    
            >>> # 生成图像
            >>> image = pipe(  # 使用管道生成图像
            ...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
            ... ).images[0]  # 获取生成的图像
# 定义一个名为 StableDiffusionXLControlNetXSPipeline 的类，继承多个基类
class StableDiffusionXLControlNetXSPipeline(
    DiffusionPipeline,  # 从 DiffusionPipeline 继承基本功能
    TextualInversionLoaderMixin,  # 从 TextualInversionLoaderMixin 继承加载文本反演的功能
    StableDiffusionXLLoraLoaderMixin,  # 从 StableDiffusionXLLoraLoaderMixin 继承加载 LoRA 权重的功能
    FromSingleFileMixin,  # 从 FromSingleFileMixin 继承从单个文件加载的功能
):
    r"""  # 文档字符串，描述该管道的功能
    Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet-XS guidance.  # 说明管道用于文本到图像的生成
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods  # 指出该模型继承自 DiffusionPipeline
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).  # 说明该模型实现了一些通用方法
    The pipeline also inherits the following loading methods:  # 说明该管道还继承了以下加载方法
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings  # 指出用于加载文本反演嵌入的方法
        - [`loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights  # 指出用于加载 LoRA 权重的方法
        - [`loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files  # 指出用于加载 `.ckpt` 文件的方法
    Args:  # 参数说明部分
        vae ([`AutoencoderKL`]):  # VAE 模型参数，表示变分自编码器
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.  # 描述 VAE 的功能
        text_encoder ([`~transformers.CLIPTextModel`]):  # 文本编码器参数，使用 CLIP 模型
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).  # 描述使用的具体文本编码器
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):  # 第二个文本编码器参数
            Second frozen text-encoder  # 描述第二个文本编码器
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).  # 指出具体的模型
        tokenizer ([`~transformers.CLIPTokenizer`]):  # 第一个分词器参数
            A `CLIPTokenizer` to tokenize text.  # 描述分词器的功能
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):  # 第二个分词器参数
            A `CLIPTokenizer` to tokenize text.  # 描述第二个分词器的功能
        unet ([`UNet2DConditionModel`]):  # UNet 模型参数，用于去噪图像
            A [`UNet2DConditionModel`] used to create a UNetControlNetXSModel to denoise the encoded image latents.  # 描述 UNet 的用途
        controlnet ([`ControlNetXSAdapter`]):  # ControlNet 参数，用于图像去噪
            A [`ControlNetXSAdapter`] to be used in combination with `unet` to denoise the encoded image latents.  # 描述 ControlNet 的用途
        scheduler ([`SchedulerMixin`]):  # 调度器参数，用于去噪处理
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of  # 说明调度器的功能
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].  # 列出可用的调度器类型
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):  # 参数，控制空提示时的处理
            Whether the negative prompt embeddings should always be set to 0. Also see the config of  # 描述该参数的功能
            `stabilityai/stable-diffusion-xl-base-1-0`.  # 指出相关的配置
        add_watermarker (`bool`, *optional*):  # 参数，控制是否使用水印
            Whether to use the [invisible_watermark](https://github.com/ShieldMnt/invisible-watermark/) library to  # 描述水印的功能
            watermark output images. If not defined, it defaults to `True` if the package is installed; otherwise no  # 说明默认值及安装条件
            watermarker is used.  # 说明无水印的条件
    """  # 文档字符串结束
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"  # 定义模型在 CPU 上的卸载顺序
    _optional_components = [  # 定义可选组件列表
        "tokenizer",  # 第一个分词器
        "tokenizer_2",  # 第二个分词器
        "text_encoder",  # 第一个文本编码器
        "text_encoder_2",  # 第二个文本编码器
        "feature_extractor",  # 特征提取器
    ]  # 可选组件列表结束
    # 定义一个包含输入张量名称的列表，用于回调处理
    _callback_tensor_inputs = [
        # 潜在变量
        "latents",
        # 提示词嵌入
        "prompt_embeds",
        # 负面提示词嵌入
        "negative_prompt_embeds",
        # 额外文本嵌入
        "add_text_embeds",
        # 额外时间 ID
        "add_time_ids",
        # 负面池化提示词嵌入
        "negative_pooled_prompt_embeds",
        # 负面额外时间 ID
        "negative_add_time_ids",
    ]

    # 初始化方法，设置类的属性
    def __init__(
        # 变分自编码器
        self,
        vae: AutoencoderKL,
        # 文本编码器
        text_encoder: CLIPTextModel,
        # 第二个文本编码器
        text_encoder_2: CLIPTextModelWithProjection,
        # 第一个分词器
        tokenizer: CLIPTokenizer,
        # 第二个分词器
        tokenizer_2: CLIPTokenizer,
        # U-Net模型，支持两种类型
        unet: Union[UNet2DConditionModel, UNetControlNetXSModel],
        # 控制网适配器
        controlnet: ControlNetXSAdapter,
        # 调度器
        scheduler: KarrasDiffusionSchedulers,
        # 是否为空提示强制使用零
        force_zeros_for_empty_prompt: bool = True,
        # 可选水印参数
        add_watermarker: Optional[bool] = None,
        # 可选特征提取器
        feature_extractor: CLIPImageProcessor = None,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查 U-Net 类型，如果是 UNet2DConditionModel，则转换为 UNetControlNetXSModel
        if isinstance(unet, UNet2DConditionModel):
            unet = UNetControlNetXSModel.from_unet(unet, controlnet)

        # 注册模型模块，便于管理和使用
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，用于处理 VAE 输出的图像
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        # 创建控制图像处理器，用于控制网图像处理
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        # 根据用户输入或默认设置决定是否添加水印
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        # 如果需要添加水印，则初始化水印处理器
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            # 否则将水印设置为 None
            self.watermark = None

        # 注册配置参数，以便后续使用
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

    # 从其他模块复制的提示编码方法
    def encode_prompt(
        # 输入的提示词
        self,
        prompt: str,
        # 可选的第二个提示词
        prompt_2: Optional[str] = None,
        # 可选的设备参数
        device: Optional[torch.device] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: int = 1,
        # 是否进行无分类器引导
        do_classifier_free_guidance: bool = True,
        # 可选的负面提示词
        negative_prompt: Optional[str] = None,
        # 可选的第二个负面提示词
        negative_prompt_2: Optional[str] = None,
        # 可选的提示词嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示词嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的池化提示词嵌入
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面池化提示词嵌入
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 LoRA 缩放因子
        lora_scale: Optional[float] = None,
        # 可选的跳过处理的 clip 层
        clip_skip: Optional[int] = None,
    # 从其他模块复制的额外步骤参数准备方法
    # 准备调度器步骤的额外参数，因为并不是所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta（η）仅在 DDIMScheduler 中使用，其他调度器会忽略它
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 的范围内
    
        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个空字典用于存放额外参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta 参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator 参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    # 检查输入的有效性，包括多个参数
    def check_inputs(
        self,
        prompt,  # 主要的提示文本
        prompt_2,  # 第二个提示文本
        image,  # 输入的图像
        negative_prompt=None,  # 可选的负面提示文本
        negative_prompt_2=None,  # 第二个可选的负面提示文本
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
        pooled_prompt_embeds=None,  # 可选的池化提示嵌入
        negative_pooled_prompt_embeds=None,  # 可选的池化负面提示嵌入
        controlnet_conditioning_scale=1.0,  # 控制网络的条件缩放
        control_guidance_start=0.0,  # 控制指导的起始值
        control_guidance_end=1.0,  # 控制指导的结束值
        callback_on_step_end_tensor_inputs=None,  # 步骤结束时的回调函数输入
    # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image 复制
    # 检查输入的图像类型及其与提示的匹配情况
    def check_image(self, image, prompt, prompt_embeds):
        # 判断图像是否为 PIL 图像类型
        image_is_pil = isinstance(image, PIL.Image.Image)
        # 判断图像是否为 Torch 张量类型
        image_is_tensor = isinstance(image, torch.Tensor)
        # 判断图像是否为 NumPy 数组类型
        image_is_np = isinstance(image, np.ndarray)
        # 判断图像是否为 PIL 图像列表
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        # 判断图像是否为 Torch 张量列表
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        # 判断图像是否为 NumPy 数组列表
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
        # 检查图像是否为有效类型，如果无效则抛出类型错误
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
    
        # 如果图像是 PIL 类型，设置批量大小为 1
        if image_is_pil:
            image_batch_size = 1
        else:
            # 否则，批量大小为图像列表的长度
            image_batch_size = len(image)
    
        # 检查提示的类型，设置对应的批量大小
        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]
    
        # 检查图像批量大小与提示批量大小是否匹配
        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )
    
    # 准备图像，调整其大小和批量
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        # 预处理图像并转换为浮点数张量
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        # 获取图像批量大小
        image_batch_size = image.shape[0]
    
        # 如果批量大小为 1，则重复次数为 batch_size
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # 否则，重复次数为每个提示的图像数量
            repeat_by = num_images_per_prompt
    
        # 根据重复次数扩展图像张量
        image = image.repeat_interleave(repeat_by, dim=0)
    
        # 将图像移动到指定设备和数据类型
        image = image.to(device=device, dtype=dtype)
    
        # 如果使用无分类器自由引导，则将图像复制两次
        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)
    
        # 返回处理后的图像
        return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在向量，设置批处理大小、通道数、高度、宽度等参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在向量的形状，考虑 VAE 的缩放因子
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器的数量是否与批处理大小一致
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        # 如果没有提供潜在向量，则生成新的随机潜在向量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在向量，则将其移动到指定设备
            latents = latents.to(device)
    
        # 将初始噪声按调度器所需的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在向量
        return latents
    
    # 获取附加时间ID，包含原始大小、裁剪坐标和目标大小
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # 计算附加时间ID，由原始大小、裁剪坐标和目标大小组成
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
        # 计算传入的附加嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取期望的附加嵌入维度
        expected_add_embed_dim = self.unet.base_add_embedding.linear_1.in_features
    
        # 检查实际的附加嵌入维度是否与期望一致
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
    
        # 将附加时间ID转换为张量
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 返回附加时间ID
        return add_time_ids
    
    # 从 StableDiffusionUpscalePipeline 复制的函数，用于上升 VAE
    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为浮点32类型
        self.vae.to(dtype=torch.float32)
        # 检查是否使用了 Torch 2.0 或 Xformers 处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # 如果使用了 Xformers 或 Torch 2.0，注意力块不需要为浮点32，这样可以节省大量内存
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)
    
        @property
        # 从 StableDiffusionPipeline 复制的属性，返回引导比例
        def guidance_scale(self):
            return self._guidance_scale
    
        @property
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.clip_skip 复制
    def clip_skip(self):
        # 返回剪辑跳过的设置
        return self._clip_skip

    @property
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.do_classifier_free_guidance 复制
    def do_classifier_free_guidance(self):
        # 判断是否启用分类器自由引导
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.cross_attention_kwargs 复制
    def cross_attention_kwargs(self):
        # 返回交叉注意力的关键字参数
        return self._cross_attention_kwargs

    @property
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.num_timesteps 复制
    def num_timesteps(self):
        # 返回时间步数
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 定义可调用方法，接收多个参数用于图像生成
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            # 定义回调函数在步骤结束时执行
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```