# `.\diffusers\pipelines\stable_diffusion_safe\pipeline_stable_diffusion_safe.py`

```py
# 导入 inspect 模块用于检查活跃对象的来源
import inspect
# 导入 warnings 模块以发出警告信息
import warnings
# 导入类型提示相关的类型
from typing import Callable, List, Optional, Union

# 导入 numpy 库以进行数值计算
import numpy as np
# 导入 torch 库用于深度学习
import torch
# 导入版本管理工具
from packaging import version
# 从 transformers 库导入必要的类用于处理图像和文本
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 从上级模块导入 FrozenDict 类
from ...configuration_utils import FrozenDict
# 从上级模块导入 PipelineImageInput 类
from ...image_processor import PipelineImageInput
# 从上级模块导入 IPAdapterMixin 类
from ...loaders import IPAdapterMixin
# 从上级模块导入模型类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
# 从上级模块导入调度器类
from ...schedulers import KarrasDiffusionSchedulers
# 从上级模块导入工具函数
from ...utils import deprecate, logging
# 从工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入 DiffusionPipeline 和 StableDiffusionMixin
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从当前模块导入输出类
from . import StableDiffusionSafePipelineOutput
# 从当前模块导入安全检查器
from .safety_checker import SafeStableDiffusionSafetyChecker

# 创建日志记录器实例以便于记录信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个安全的稳定扩散管道类，继承自 DiffusionPipeline 和其他混合类
class StableDiffusionPipelineSafe(DiffusionPipeline, StableDiffusionMixin, IPAdapterMixin):
    r"""
    基于 [`StableDiffusionPipeline`] 的管道，用于使用安全的潜在扩散进行文本到图像生成。

    此模型继承自 [`DiffusionPipeline`]。查看超类文档以获取所有管道的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行标记的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码图像潜在表示的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码图像潜在表示。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的一个。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            评估生成的图像是否可能被认为具有攻击性或有害的分类模块。
            有关模型潜在危害的更多详细信息，请参阅 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成的图像中提取特征的 `CLIPImageProcessor`；作为输入传递给 `safety_checker`。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 初始化方法，用于创建类的实例
    def __init__(
        self,  # 方法参数列表的开始
        vae: AutoencoderKL,  # 变分自编码器实例
        text_encoder: CLIPTextModel,  # 文本编码器实例
        tokenizer: CLIPTokenizer,  # 分词器实例
        unet: UNet2DConditionModel,  # U-Net 条件模型实例
        scheduler: KarrasDiffusionSchedulers,  # Karras 扩散调度器实例
        safety_checker: SafeStableDiffusionSafetyChecker,  # 安全检查器实例
        feature_extractor: CLIPImageProcessor,  # 特征提取器实例
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,  # 可选的图像编码器实例
        requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
    ):
    
    # 安全概念的属性获取器
    @property
    def safety_concept(self):
        r"""  # 文档字符串，描述获取器的功能
        Getter method for the safety concept used with SLD  # 获取安全概念的方法

        Returns:
            `str`: The text describing the safety concept  # 返回安全概念的文本描述
        """
        return self._safety_text_concept  # 返回安全概念的内部文本

    # 安全概念的属性设置器
    @safety_concept.setter
    def safety_concept(self, concept):
        r"""  # 文档字符串，描述设置器的功能
        Setter method for the safety concept used with SLD  # 设置安全概念的方法

        Args:
            concept (`str`):  # 参数说明，新的安全概念文本
                The text of the new safety concept
        """
        self._safety_text_concept = concept  # 设置新的安全概念文本

    # 编码提示的方法
    def _encode_prompt(
        self,  # 方法参数列表的开始
        prompt,  # 输入的提示文本
        device,  # 运行设备（CPU/GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否执行无分类器引导
        negative_prompt,  # 负提示文本
        enable_safety_guidance,  # 是否启用安全引导
    ):
    
    # 运行安全检查器的方法
    def run_safety_checker(self, image, device, dtype, enable_safety_guidance):
        # 检查安全检查器是否存在
        if self.safety_checker is not None:
            images = image.copy()  # 复制输入图像以供检查
            # 提取特征并转换为张量，准备安全检查器的输入
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            # 运行安全检查器，检查图像是否包含 NSFW 内容
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)  # 传递图像和特征输入
            )
            # 创建一个标记图像的数组
            flagged_images = np.zeros((2, *image.shape[1:]))  # 用于存放被标记的图像
            # 如果检测到 NSFW 概念
            if any(has_nsfw_concept):
                logger.warning(  # 记录警告信息
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead."
                    f"{'You may look at this images in the `unsafe_images` variable of the output at your own discretion.' if enable_safety_guidance else 'Try again with a different prompt and/or seed.'}"
                )
                # 遍历每个图像，检查是否存在 NSFW 概念
                for idx, has_nsfw_concept in enumerate(has_nsfw_concept):
                    if has_nsfw_concept:  # 如果检测到 NSFW 概念
                        flagged_images[idx] = images[idx]  # 保存被标记的图像
                        image[idx] = np.zeros(image[idx].shape)  # 将该图像替换为黑色图像
        else:  # 如果没有安全检查器
            has_nsfw_concept = None  # NSFW 概念为 None
            flagged_images = None  # 被标记的图像为 None
        return image, has_nsfw_concept, flagged_images  # 返回处理后的图像和概念信息

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制的代码
    # 解码潜在变量以生成图像
    def decode_latents(self, latents):
        # 生成关于 decode_latents 方法被弃用的提示信息
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用 deprecate 函数记录方法弃用信息
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 根据缩放因子调整潜在变量的值
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量并获取生成的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像值缩放到 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 格式，并进行必要的形状变换
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回生成的图像
        return image
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备额外的参数以用于调度器步骤，因为并非所有调度器都有相同的签名
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将被忽略
        # eta 对应 DDIM 论文中的 η，范围应在 [0, 1] 之间
    
        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数的字典
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    # 从 diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs 复制
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查高度和宽度是否都能被8整除
        if height % 8 != 0 or width % 8 != 0:
            # 如果不能整除，则抛出值错误，提示高度和宽度的当前值
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步数是否有效
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果回调步数不是正整数，则抛出值错误，提示其当前值和类型
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查给定的回调输入是否在已定义的回调张量输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有不在列表中的输入，则抛出值错误，列出不符合的输入
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了提示和提示嵌入
        if prompt is not None and prompt_embeds is not None:
            # 如果两者都提供，则抛出值错误，说明只允许提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否都未提供提示和提示嵌入
        elif prompt is None and prompt_embeds is None:
            # 如果两者都未定义，则抛出值错误，提示至少需要提供一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示是否为有效类型
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果提示不是字符串或列表，则抛出值错误，提示其当前类型
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了负提示和负提示嵌入
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果两者都提供，则抛出值错误，说明只允许提供其中一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负提示嵌入的形状是否相同
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不匹配，则抛出值错误，说明两者的形状必须相同
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 复制自 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    # 准备潜在变量的函数，定义输入参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，考虑 VAE 的缩放因子
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器的类型及其长度是否与批次大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出值错误，提示生成器数量与批次大小不一致
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在变量，则生成随机的潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，将其转移到指定设备
            latents = latents.to(device)

        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 执行安全引导的函数，定义输入参数
    def perform_safety_guidance(
        self,
        enable_safety_guidance,
        safety_momentum,
        noise_guidance,
        noise_pred_out,
        i,
        sld_guidance_scale,
        sld_warmup_steps,
        sld_threshold,
        sld_momentum_scale,
        sld_mom_beta,
    ):
        # 如果启用了安全引导
        if enable_safety_guidance:
            # 如果安全动量未定义，则初始化为与噪声引导相同形状的零张量
            if safety_momentum is None:
                safety_momentum = torch.zeros_like(noise_guidance)
            # 从噪声预测输出中提取文本噪声和无条件噪声预测
            noise_pred_text, noise_pred_uncond = noise_pred_out[0], noise_pred_out[1]
            noise_pred_safety_concept = noise_pred_out[2]

            # 计算安全引导的比例（公式6）
            scale = torch.clamp(torch.abs((noise_pred_text - noise_pred_safety_concept)) * sld_guidance_scale, max=1.0)

            # 根据阈值计算安全概念的比例（公式6）
            safety_concept_scale = torch.where(
                (noise_pred_text - noise_pred_safety_concept) >= sld_threshold, torch.zeros_like(scale), scale
            )

            # 计算安全噪声引导（公式4）
            noise_guidance_safety = torch.mul((noise_pred_safety_concept - noise_pred_uncond), safety_concept_scale)

            # 将动量加入安全噪声引导（公式7）
            noise_guidance_safety = noise_guidance_safety + sld_momentum_scale * safety_momentum

            # 更新安全动量（公式8）
            safety_momentum = sld_mom_beta * safety_momentum + (1 - sld_mom_beta) * noise_guidance_safety

            # 如果当前步骤超过暖身步骤
            if i >= sld_warmup_steps:  # Warmup
                # 根据安全噪声引导调整总噪声引导（公式3）
                noise_guidance = noise_guidance - noise_guidance_safety
        # 返回调整后的噪声引导和安全动量
        return noise_guidance, safety_momentum

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制的内容
    # 定义一个编码图像的方法，接受图像及其相关参数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量类型
            if not isinstance(image, torch.Tensor):
                # 使用特征提取器将图像转换为张量，并返回张量的像素值
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定的设备，并转换为正确的数据类型
            image = image.to(device=device, dtype=dtype)
            
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态重复指定次数，沿着第0维
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于无条件图像，创建一个与输入图像相同形状的零张量，并编码其隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将无条件图像的隐藏状态重复指定次数，沿着第0维
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的图像和无条件图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入重复指定次数，沿着第0维
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回编码后的图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 使用装饰器禁止梯度计算
        @torch.no_grad()
        # 定义一个可调用的方法，接受多个参数
        def __call__(
            self,
            prompt: Union[str, List[str]],  # 输入提示，可以是字符串或字符串列表
            height: Optional[int] = None,  # 可选的图像高度
            width: Optional[int] = None,  # 可选的图像宽度
            num_inference_steps: int = 50,  # 推理步骤数量，默认为50
            guidance_scale: float = 7.5,  # 引导尺度，默认为7.5
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量，默认为1
            eta: float = 0.0,  # 额外参数，默认为0.0
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选的随机数生成器
            latents: Optional[torch.Tensor] = None,  # 可选的潜在张量
            ip_adapter_image: Optional[PipelineImageInput] = None,  # 可选的适配器图像输入
            output_type: Optional[str] = "pil",  # 输出类型，默认为"pil"
            return_dict: bool = True,  # 是否返回字典格式，默认为True
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选的回调函数
            callback_steps: int = 1,  # 回调步骤，默认为1
            sld_guidance_scale: Optional[float] = 1000,  # 可选的平滑引导尺度，默认为1000
            sld_warmup_steps: Optional[int] = 10,  # 可选的预热步骤，默认为10
            sld_threshold: Optional[float] = 0.01,  # 可选的阈值，默认为0.01
            sld_momentum_scale: Optional[float] = 0.3,  # 可选的动量尺度，默认为0.3
            sld_mom_beta: Optional[float] = 0.4,  # 可选的动量贝塔，默认为0.4
```