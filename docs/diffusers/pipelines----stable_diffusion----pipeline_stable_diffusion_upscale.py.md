# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion_upscale.py`

```py
# 版权声明，表明该代码的版权所有者及相关权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（"许可证"）授权；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，软件
# 在许可证下分发是以“原样”基础提供的，
# 不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解有关权限和
# 限制的具体信息。

# 导入 inspect 模块，用于获取对象的信息
import inspect
# 导入 warnings 模块，用于发出警告
import warnings
# 从 typing 模块导入类型注解支持
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 PIL 库中的 Image 类，用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 相关类
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 从相对路径导入其他处理器和加载器
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
# 从注意力处理器模块导入相关类
from ...models.attention_processor import (
    AttnProcessor2_0,
    XFormersAttnProcessor,
)
# 从 Lora 模块导入调整 Lora 规模的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从调度器模块导入相关类
from ...schedulers import DDPMScheduler, KarrasDiffusionSchedulers
# 从工具模块导入各种工具函数
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
# 从 Torch 工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入相关类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从当前包导入稳定扩散管道输出
from . import StableDiffusionPipelineOutput

# 创建日志记录器，用于记录当前模块的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# 定义预处理函数，处理输入图像
def preprocess(image):
    # 发出警告，提示该方法已弃用，未来版本将被移除
    warnings.warn(
        "The preprocess method is deprecated and will be removed in a future version. Please"
        " use VaeImageProcessor.preprocess instead",
        FutureWarning,
    )
    # 如果输入是张量，直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，转换为列表
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 检查列表中第一个元素是否为 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽和高
        w, h = image[0].size
        # 将宽高调整为 64 的整数倍
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        # 调整所有图像的大小并转换为 NumPy 数组
        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        # 将所有图像数组沿第一个轴连接
        image = np.concatenate(image, axis=0)
        # 将数组转换为浮点型并归一化到 [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序为 (N, C, H, W)
        image = image.transpose(0, 3, 1, 2)
        # 将像素值范围从 [0, 1] 映射到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果列表中第一个元素是张量，沿着第一个维度连接它们
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image


# 定义 StableDiffusionUpscalePipeline 类，继承多个基类
class StableDiffusionUpscalePipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    FromSingleFileMixin,
):
    r"""
    使用 Stable Diffusion 2 进行文本引导的图像超分辨率的管道。

    此模型继承自 [`DiffusionPipeline`]。请查阅超类文档以获取所有管道通用方法的实现（下载、保存、在特定设备上运行等）。
    # 管道还继承了以下加载方法：
        # - `~loaders.TextualInversionLoaderMixin.load_textual_inversion` 用于加载文本反转嵌入
        # - `~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights` 用于加载 LoRA 权重
        # - `~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights` 用于保存 LoRA 权重
        # - `~loaders.FromSingleFileMixin.from_single_file` 用于加载 `.ckpt` 文件

    # 参数说明
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):  # 冻结的文本编码器
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):  # 用于标记化文本的 CLIPTokenizer
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):  # 用于去噪编码图像潜在的 UNet2DConditionModel
            A `UNet2DConditionModel` to denoise the encoded image latents.
        low_res_scheduler ([`SchedulerMixin`]):  # 用于向低分辨率条件图像添加初始噪声的调度器，必须是 DDPMScheduler 的实例
            A scheduler used to add initial noise to the low resolution conditioning image. It must be an instance of
            [`DDPMScheduler`].
        scheduler ([`SchedulerMixin`]):  # 与 UNet 一起使用的调度器，用于去噪编码图像潜在
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    # 定义 CPU 离线加载的模型顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件的列表
    _optional_components = ["watermarker", "safety_checker", "feature_extractor"]
    # 定义不包含在 CPU 离线加载中的组件
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法
    def __init__(  # 定义初始化方法
        self,
        vae: AutoencoderKL,  # 接收变分自编码器（VAE）模型
        text_encoder: CLIPTextModel,  # 接收文本编码器
        tokenizer: CLIPTokenizer,  # 接收标记化器
        unet: UNet2DConditionModel,  # 接收 UNet 模型
        low_res_scheduler: DDPMScheduler,  # 接收低分辨率调度器
        scheduler: KarrasDiffusionSchedulers,  # 接收调度器
        safety_checker: Optional[Any] = None,  # 可选的安全检查器
        feature_extractor: Optional[CLIPImageProcessor] = None,  # 可选的特征提取器
        watermarker: Optional[Any] = None,  # 可选的水印处理器
        max_noise_level: int = 350,  # 最大噪声级别，默认为350
    ):
        # 初始化父类
        super().__init__()

        # 检查 VAE 是否有配置属性
        if hasattr(
            vae, "config"
        ):  # 检查 VAE 是否具有配置属性 `scaling_factor`，如果未设置为 0.08333，则设置为 0.08333 并发出弃用警告
            # 确定 `scaling_factor` 是否已设置为 0.08333
            is_vae_scaling_factor_set_to_0_08333 = (
                hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor == 0.08333
            )
            # 如果 `scaling_factor` 未设置为 0.08333，则执行以下操作
            if not is_vae_scaling_factor_set_to_0_08333:
                # 创建弃用消息，说明配置问题及建议
                deprecation_message = (
                    "The configuration file of the vae does not contain `scaling_factor` or it is set to"
                    f" {vae.config.scaling_factor}, which seems highly unlikely. If your checkpoint is a fine-tuned"
                    " version of `stabilityai/stable-diffusion-x4-upscaler` you should change 'scaling_factor' to"
                    " 0.08333 Please make sure to update the config accordingly, as not doing so might lead to"
                    " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging"
                    " Face Hub, it would be very nice if you could open a Pull Request for the `vae/config.json` file"
                )
                # 记录弃用警告，并更新 `scaling_factor` 为 0.08333
                deprecate("wrong scaling_factor", "1.0.0", deprecation_message, standard_warn=False)
                vae.register_to_config(scaling_factor=0.08333)

        # 注册各个模块到当前配置中
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            safety_checker=safety_checker,
            watermarker=watermarker,
            feature_extractor=feature_extractor,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器实例，使用双三次插值法
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, resample="bicubic")
        # 将最大噪声水平注册到配置中
        self.register_to_config(max_noise_level=max_noise_level)

    def run_safety_checker(self, image, device, dtype):
        # 如果存在安全检查器，则执行安全检查
        if self.safety_checker is not None:
            # 对输入图像进行后处理以适配安全检查器
            feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            # 将后处理后的图像转换为张量，并移动到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 进行安全检查，获取处理后的图像及检测结果
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
            )
        else:
            # 如果没有安全检查器，则将检测结果设置为 None
            nsfw_detected = None
            watermark_detected = None

            # 如果存在 UNet 的卸载钩子，则执行卸载操作
            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()

        # 返回处理后的图像及检测结果
        return image, nsfw_detected, watermark_detected

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的代码
    # 定义一个私有方法用于编码提示信息
    def _encode_prompt(
        self,  # 方法的第一个参数，表示实例本身
        prompt,  # 要编码的提示信息
        device,  # 设备（如 CPU 或 GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 可选的负面提示信息
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
        **kwargs,  # 其他可选参数
    ):
        # 警告信息，表示此方法已被弃用，未来版本将删除
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 方法记录弃用信息
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法编码提示，返回一个元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 传递提示信息
            device=device,  # 传递设备
            num_images_per_prompt=num_images_per_prompt,  # 传递每个提示的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 传递无分类器引导标志
            negative_prompt=negative_prompt,  # 传递负面提示
            prompt_embeds=prompt_embeds,  # 传递提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,  # 传递负面提示嵌入
            lora_scale=lora_scale,  # 传递 LoRA 缩放因子
            **kwargs,  # 传递其他参数
        )

        # 为向后兼容，将元组中的提示嵌入连接在一起
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回连接后的提示嵌入
        return prompt_embeds

    # 从 StableDiffusionPipeline 类中复制的编码提示方法
    def encode_prompt(
        self,  # 方法的第一个参数，表示实例本身
        prompt,  # 要编码的提示信息
        device,  # 设备（如 CPU 或 GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 可选的负面提示信息
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
        clip_skip: Optional[int] = None,  # 可选的剪切跳过参数
    # 从 StableDiffusionPipeline 类中复制的准备额外步骤参数的方法
    def prepare_extra_step_kwargs(self, generator, eta):  # 准备额外的调度器步骤参数
        # 为调度器步骤准备额外的参数，因为并非所有调度器的签名相同
        # eta (η) 仅在 DDIMScheduler 中使用，对其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 之间

        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}  # 初始化一个空字典以存储额外参数
        if accepts_eta:  # 如果接受 eta
            extra_step_kwargs["eta"] = eta  # 将 eta 添加到字典中

        # 检查调度器步骤是否接受生成器参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:  # 如果接受生成器
            extra_step_kwargs["generator"] = generator  # 将生成器添加到字典中
        # 返回准备好的额外步骤参数
        return extra_step_kwargs

    # 从 StableDiffusionPipeline 类中复制的解码潜在变量的方法
    # 解码潜在表示
    def decode_latents(self, latents):
        # 定义一个弃用提示消息，告知用户该方法即将被移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用函数，记录使用该方法的警告信息
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 根据 VAE 的缩放因子调整潜在表示的值
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在表示，返回的第一项为解码后的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像值缩放到 [0, 1] 范围并进行裁剪
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 格式，以提高兼容性
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image
    
    # 检查输入参数的有效性
    def check_inputs(
        self,
        prompt,  # 输入提示
        image,  # 输入图像
        noise_level,  # 噪声水平
        callback_steps,  # 回调步骤
        negative_prompt=None,  # 可选的负面提示
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
    )
    
    # 准备潜在表示
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在表示的形状
        shape = (batch_size, num_channels_latents, height, width)
        # 如果没有提供潜在表示，随机生成
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供的潜在表示形状不匹配，则引发错误
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在表示移动到指定设备
            latents = latents.to(device)
    
        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在表示
        return latents
    
    # 升级 VAE 的数据类型
    def upcast_vae(self):
        # 获取 VAE 的当前数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 数据类型
        self.vae.to(dtype=torch.float32)
        # 检查是否使用了特定的注意力处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # 如果使用 xformers 或 torch_2_0，则注意力块不需要是 float32，从而节省内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层转换为相同数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将解码器输入卷积层转换为相同数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将解码器中间块转换为相同数据类型
            self.vae.decoder.mid_block.to(dtype)
    
    # 关闭梯度计算以节省内存
    @torch.no_grad()
    # 定义一个可调用的方法，允许传入多个参数进行处理
        def __call__(
            # 用户输入的提示信息，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输入的图像数据，类型为 PipelineImageInput
            image: PipelineImageInput = None,
            # 推理步骤的数量，默认值为 75
            num_inference_steps: int = 75,
            # 指导比例，用于控制生成图像的样式，默认值为 9.0
            guidance_scale: float = 9.0,
            # 噪声级别，影响生成图像的随机性，默认值为 20
            noise_level: int = 20,
            # 负提示信息，可以是字符串或字符串列表，控制生成图像的方向
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 额外的参数，影响生成过程，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个或多个 torch.Generator
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 预先计算的潜在表示，类型为 torch.Tensor
            latents: Optional[torch.Tensor] = None,
            # 预先计算的提示嵌入，类型为 torch.Tensor
            prompt_embeds: Optional[torch.Tensor] = None,
            # 预先计算的负提示嵌入，类型为 torch.Tensor
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"，表示以 PIL 格式返回
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 回调函数，可用于处理生成过程中的状态，返回 None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调函数的调用频率，默认为每一步 1 次
            callback_steps: int = 1,
            # 跨注意力的额外参数，可以为字典类型
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指定跳过的剪切层级，默认为 None
            clip_skip: int = None,
```