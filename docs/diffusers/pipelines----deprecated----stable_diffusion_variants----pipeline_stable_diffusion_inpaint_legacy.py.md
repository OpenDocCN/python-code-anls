# `.\diffusers\pipelines\deprecated\stable_diffusion_variants\pipeline_stable_diffusion_inpaint_legacy.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“现状”提供的，
# 不提供任何形式的担保或条件，无论是明示或暗示的。
# 有关许可证下特定权限和限制的详细信息，请参见许可证。

# 导入 inspect 模块以检查对象
import inspect
# 从 typing 模块导入类型提示相关的类
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库用于数值计算
import numpy as np
# 导入 PIL.Image 以处理图像
import PIL.Image
# 导入 torch 库以进行深度学习计算
import torch
# 导入 version 模块以处理版本比较
from packaging import version
# 导入 HuggingFace 变换器的相关类
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 导入 FrozenDict 类用于不可变字典
from ....configuration_utils import FrozenDict
# 导入 VaeImageProcessor 类处理变分自编码器图像
from ....image_processor import VaeImageProcessor
# 导入不同的加载器混合类
from ....loaders import FromSingleFileMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入自编码器和条件 UNet 模型
from ....models import AutoencoderKL, UNet2DConditionModel
# 导入用于调整 Lora 规模的函数
from ....models.lora import adjust_lora_scale_text_encoder
# 导入 Karras Diffusion 调度器
from ....schedulers import KarrasDiffusionSchedulers
# 导入工具函数和常量
from ....utils import PIL_INTERPOLATION, USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
# 导入用于生成随机张量的函数
from ....utils.torch_utils import randn_tensor
# 导入 DiffusionPipeline 类
from ...pipeline_utils import DiffusionPipeline
# 导入 StableDiffusionPipelineOutput 类
from ...stable_diffusion import StableDiffusionPipelineOutput
# 导入稳定扩散的安全检查器
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义图像预处理函数
def preprocess_image(image, batch_size):
    # 获取图像的宽和高
    w, h = image.size
    # 调整宽和高为8的整数倍
    w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
    # 根据新的尺寸重新调整图像大小，使用 Lanczos 重采样
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    # 将图像转换为浮点数数组并归一化到[0, 1]范围
    image = np.array(image).astype(np.float32) / 255.0
    # 创建一个大小为 batch_size 的图像堆叠
    image = np.vstack([image[None].transpose(0, 3, 1, 2)] * batch_size)
    # 将 NumPy 数组转换为 PyTorch 张量
    image = torch.from_numpy(image)
    # 将图像值缩放到 [-1, 1] 范围
    return 2.0 * image - 1.0

# 定义掩膜预处理函数
def preprocess_mask(mask, batch_size, scale_factor=8):
    # 如果掩膜不是张量，则进行转换
    if not isinstance(mask, torch.Tensor):
        # 将掩膜转换为灰度图像
        mask = mask.convert("L")
        # 获取掩膜的宽和高
        w, h = mask.size
        # 调整宽和高为8的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        # 根据缩放因子调整掩膜大小，使用最近邻重采样
        mask = mask.resize((w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"])
        # 将掩膜转换为浮点数数组并归一化到[0, 1]范围
        mask = np.array(mask).astype(np.float32) / 255.0
        # 将掩膜扩展到4个通道
        mask = np.tile(mask, (4, 1, 1))
        # 创建一个大小为 batch_size 的掩膜堆叠
        mask = np.vstack([mask[None]] * batch_size)
        # 将白色部分重绘为黑色，黑色部分保持不变
        mask = 1 - mask  # repaint white, keep black
        # 将 NumPy 数组转换为 PyTorch 张量
        mask = torch.from_numpy(mask)
        # 返回处理后的掩膜
        return mask
    else:
        # 定义有效的掩码通道大小
        valid_mask_channel_sizes = [1, 3]
        # 如果掩码通道是第四个张量维度，调整维度顺序为 PyTorch 标准 (B, C, H, W)
        if mask.shape[3] in valid_mask_channel_sizes:
            mask = mask.permute(0, 3, 1, 2)
        # 如果掩码的第二个维度大小不在有效通道大小中，抛出异常
        elif mask.shape[1] not in valid_mask_channel_sizes:
            raise ValueError(
                f"Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension,"
                f" but received mask of shape {tuple(mask.shape)}"
            )
        # （可能）将掩码通道维度从 3 减少到 1，以便广播到潜在形状
        mask = mask.mean(dim=1, keepdim=True)
        # 获取掩码的高度和宽度
        h, w = mask.shape[-2:]
        # 将高度和宽度调整为 8 的整数倍
        h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
        # 使用插值调整掩码大小到目标尺寸
        mask = torch.nn.functional.interpolate(mask, (h // scale_factor, w // scale_factor))
        # 返回调整后的掩码
        return mask
# 定义一个名为 StableDiffusionInpaintPipelineLegacy 的类，继承多个混入类以扩展功能
class StableDiffusionInpaintPipelineLegacy(
    # 继承 DiffusionPipeline 和其他混入类以获取其功能
    DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin, FromSingleFileMixin
):
    # 文档字符串，描述此管道的用途和特性
    r"""
    Pipeline for text-guided image inpainting using Stable Diffusion. *This is an experimental feature*.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # 定义一个字符串，表示模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义一个可选组件列表，包含 feature_extractor
    _optional_components = ["feature_extractor"]
    # 定义一个不包含在 CPU 卸载中的组件列表，包含 safety_checker
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法，定义构造函数接受的参数
    def __init__(
        # 接受一个 AutoencoderKL 实例作为 VAE 模型
        self,
        vae: AutoencoderKL,
        # 接受一个 CLIPTextModel 实例作为文本编码器
        text_encoder: CLIPTextModel,
        # 接受一个 CLIPTokenizer 实例作为分词器
        tokenizer: CLIPTokenizer,
        # 接受一个 UNet2DConditionModel 实例作为去噪模型
        unet: UNet2DConditionModel,
        # 接受一个调度器，通常是 KarrasDiffusionSchedulers 的实例
        scheduler: KarrasDiffusionSchedulers,
        # 接受一个 StableDiffusionSafetyChecker 实例作为安全检查器
        safety_checker: StableDiffusionSafetyChecker,
        # 接受一个 CLIPImageProcessor 实例作为特征提取器
        feature_extractor: CLIPImageProcessor,
        # 定义一个布尔值，表示是否需要安全检查器，默认值为 True
        requires_safety_checker: bool = True,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的代码
    def _encode_prompt(
        self,  # 定义一个方法，接受多个参数
        prompt,  # 输入的提示文本
        device,  # 设备类型（例如 CPU 或 GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 Lora 比例
        **kwargs,  # 其他任意参数
    ):
        # 设置弃用消息，告知用户该方法将在未来版本中移除
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用弃用函数，记录弃用信息
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用新的 encode_prompt 方法，并获取提示嵌入的元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 传递提示文本
            device=device,  # 传递设备
            num_images_per_prompt=num_images_per_prompt,  # 传递每个提示的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 传递无分类器引导标志
            negative_prompt=negative_prompt,  # 传递负面提示
            prompt_embeds=prompt_embeds,  # 传递提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,  # 传递负面提示嵌入
            lora_scale=lora_scale,  # 传递 Lora 比例
            **kwargs,  # 传递其他任意参数
        )

        # 将提示嵌入元组中的两个部分连接起来，兼容旧版本
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回连接后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的代码
    def encode_prompt(
        self,  # 定义一个方法，接受多个参数
        prompt,  # 输入的提示文本
        device,  # 设备类型（例如 CPU 或 GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 Lora 比例
        clip_skip: Optional[int] = None,  # 可选的剪辑跳过参数
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的代码
    def run_safety_checker(self, image, device, dtype):  # 定义安全检查方法，接受图像、设备和数据类型
        if self.safety_checker is None:  # 如果安全检查器未定义
            has_nsfw_concept = None  # 设置无敏感内容概念为 None
        else:  # 否则
            if torch.is_tensor(image):  # 如果输入图像是张量
                # 将图像后处理为 PIL 格式
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:  # 如果输入图像不是张量
                # 将 NumPy 数组转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征并将其转移到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，检查图像和提取的特征
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和是否存在敏感内容的概念
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制的代码
    # 解码潜在表示
        def decode_latents(self, latents):
            # 设置弃用警告信息，提示用户该方法将在1.0.0版本中被移除
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 调用弃用函数，记录弃用信息
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 按照缩放因子调整潜在表示的值
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在表示，返回图像的第一个元素
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像值归一化到[0, 1]区间
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为float32格式，以确保兼容bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回处理后的图像
            return image
    
        # 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs复制而来
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为并非所有调度器具有相同的参数签名
            # eta (η) 仅在DDIMScheduler中使用，对于其他调度器将被忽略。
            # eta对应于DDIM论文中的η: https://arxiv.org/abs/2010.02502
            # 应该在[0, 1]之间
    
            # 检查调度器的步骤函数是否接受eta参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            # 如果接受eta参数，则将其添加到字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤函数是否接受generator参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受generator参数，则将其添加到字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 检查输入参数
        def check_inputs(
            self,
            prompt,  # 输入的提示信息
            strength,  # 强度参数
            callback_steps,  # 回调步骤
            negative_prompt=None,  # 可选的负面提示信息
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
            callback_on_step_end_tensor_inputs=None,  # 可选的回调在步骤结束时的张量输入
    ):
        # 检查 strength 是否在 [0, 1] 范围内，若不在则抛出 ValueError
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 检查 callback_steps 是否为正整数，若不是则抛出 ValueError
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 callback_on_step_end_tensor_inputs 中的键是否在 _callback_tensor_inputs 中，若不在则抛出 ValueError
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        # 检查 prompt 和 prompt_embeds 是否同时定义，若是则抛出 ValueError
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都未定义，若是则抛出 ValueError
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否为 str 或 list，若不是则抛出 ValueError
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查 negative_prompt 和 negative_prompt_embeds 是否同时定义，若是则抛出 ValueError
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否都已定义，并检查它们的形状是否相同，若不相同则抛出 ValueError
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 定义获取时间步的函数，参数包括推理步骤数量、strength 和设备
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步，取 num_inference_steps 和 strength 的乘积的最小值
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算起始时间步，确保不小于 0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 获取调度器中的时间步，基于 t_start 和调度器的顺序
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # 返回时间步和剩余推理步骤
        return timesteps, num_inference_steps - t_start
    # 准备潜在向量的函数，输入图像、时间步、每个提示生成的图像数量、数据类型、设备和随机生成器
    def prepare_latents(self, image, timestep, num_images_per_prompt, dtype, device, generator):
        # 将输入图像转换为指定设备和数据类型
        image = image.to(device=device, dtype=dtype)
        # 使用变分自编码器 (VAE) 编码图像，获取潜在分布
        init_latent_dist = self.vae.encode(image).latent_dist
        # 从潜在分布中采样潜在向量
        init_latents = init_latent_dist.sample(generator=generator)
        # 将潜在向量缩放，以适应 VAE 配置的缩放因子
        init_latents = self.vae.config.scaling_factor * init_latents

        # 扩展初始潜在向量，以匹配批处理大小和每个提示生成的图像数量
        init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)
        # 保存原始的初始潜在向量
        init_latents_orig = init_latents

        # 使用时间步向潜在向量添加噪声
        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=dtype)
        # 调度器添加噪声到初始潜在向量中
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        # 将潜在向量赋值给变量 latents
        latents = init_latents
        # 返回潜在向量、原始潜在向量和噪声
        return latents, init_latents_orig, noise

    # 装饰器表示在计算梯度时不会记录该函数的操作
    @torch.no_grad()
    # 定义调用函数，允许多种参数输入
    def __call__(
        # 提示文本，可以是单个字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 输入图像，可以是张量或 PIL 图像
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        # 掩码图像，可以是张量或 PIL 图像
        mask_image: Union[torch.Tensor, PIL.Image.Image] = None,
        # 强度参数，控制图像的强度
        strength: float = 0.8,
        # 推理步骤的数量
        num_inference_steps: Optional[int] = 50,
        # 指导比例，影响生成结果的引导强度
        guidance_scale: Optional[float] = 7.5,
        # 负提示，可以是单个字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: Optional[int] = 1,
        # 是否添加预测的噪声
        add_predicted_noise: Optional[bool] = False,
        # eta 参数，用于控制生成过程
        eta: Optional[float] = 0.0,
        # 随机生成器，可以是单个或多个生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 提示嵌入，允许预先计算的嵌入传入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的结果
        return_dict: bool = True,
        # 回调函数，用于在推理步骤中进行额外操作
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 每多少步调用一次回调
        callback_steps: int = 1,
        # 跨注意力的额外参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 跳过的剪辑层数
        clip_skip: Optional[int] = None,
```