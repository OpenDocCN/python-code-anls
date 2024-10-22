# `.\diffusers\pipelines\deprecated\stable_diffusion_variants\pipeline_stable_diffusion_model_editing.py`

```py
# 版权信息，声明版权和许可协议
# Copyright 2024 TIME Authors and The HuggingFace Team. All rights reserved."
# 根据 Apache License 2.0 许可协议进行许可
# 此文件只能在遵守许可证的情况下使用
# 可通过以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按“现状”分发，不提供任何形式的担保或条件
# 具体许可条款和限制请参见许可证

# 导入复制模块，用于对象复制
import copy
# 导入检查模块，用于检查对象的信息
import inspect
# 导入类型提示相关的模块
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库中导入图像处理器、文本模型和标记器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 从相对路径导入自定义图像处理器
from ....image_processor import VaeImageProcessor
# 从相对路径导入自定义加载器混合类
from ....loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从相对路径导入自定义模型类
from ....models import AutoencoderKL, UNet2DConditionModel
# 从相对路径导入调整 LoRA 规模的函数
from ....models.lora import adjust_lora_scale_text_encoder
# 从相对路径导入调度器类
from ....schedulers import PNDMScheduler
# 从调度器工具导入调度器混合类
from ....schedulers.scheduling_utils import SchedulerMixin
# 从工具库中导入多个功能模块
from ....utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
# 从自定义的 PyTorch 工具库中导入随机张量生成函数
from ....utils.torch_utils import randn_tensor
# 从管道工具导入扩散管道和稳定扩散混合类
from ...pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从稳定扩散相关模块导入管道输出类
from ...stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
# 从稳定扩散安全检查器模块导入安全检查器类
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# 创建一个日志记录器，用于记录当前模块的信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个常量列表，包含不同的图像描述前缀
AUGS_CONST = ["A photo of ", "An image of ", "A picture of "]

# 定义一个稳定扩散模型编辑管道类，继承自多个基类
class StableDiffusionModelEditingPipeline(
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin
):
    r"""
    文本到图像模型编辑的管道。

    该模型继承自 [`DiffusionPipeline`]。请查阅超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
    # 文档字符串，描述类或方法的参数
    Args:
        vae ([`AutoencoderKL`]):
            # Variational Auto-Encoder (VAE) 模型，用于将图像编码和解码为潜在表示
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，使用 CLIP 的大型视觉变换模型
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # 用于文本标记化的 CLIPTokenizer
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            # 用于去噪编码图像潜在的 UNet2DConditionModel
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            # 调度器，与 unet 一起用于去噪编码的图像潜在，可以是 DDIMScheduler、LMSDiscreteScheduler 或 PNDMScheduler
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 分类模块，用于估计生成的图像是否可能被视为冒犯或有害
            Classification module that estimates whether generated images could be considered offensive or harmful.
            # 参阅模型卡以获取有关模型潜在危害的更多详细信息
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # 用于从生成的图像中提取特征的 CLIPImageProcessor；作为输入传递给 safety_checker
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
        with_to_k ([`bool`]):
            # 是否在编辑文本到图像模型时编辑键投影矩阵与值投影矩阵
            Whether to edit the key projection matrices along with the value projection matrices.
        with_augs ([`list`]):
            # 在编辑文本到图像模型时应用的文本增强，设置为 [] 表示不进行增强
            Textual augmentations to apply while editing the text-to-image model. Set to `[]` for no augmentations.
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor"]
    # 定义不参与 CPU 卸载的组件
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法，设置模型和参数
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        with_to_k: bool = True,
        with_augs: list = AUGS_CONST,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的代码
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        # 定义一个弃用消息，提示用户 `_encode_prompt()` 已弃用，建议使用 `encode_prompt()`，并说明输出格式变化
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数记录弃用警告，指定版本和警告信息，标准警告设置为 False
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法，将参数传入以获取提示嵌入的元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 用户输入的提示文本
            device=device,  # 指定运行设备
            num_images_per_prompt=num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=negative_prompt,  # 负面提示文本
            prompt_embeds=prompt_embeds,  # 现有的提示嵌入（如果有的话）
            negative_prompt_embeds=negative_prompt_embeds,  # 负面提示嵌入（如果有的话）
            lora_scale=lora_scale,  # Lora 缩放参数
            **kwargs,  # 额外参数
        )

        # 连接提示嵌入元组的两个部分，适配旧版本的兼容性
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回连接后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制
    def encode_prompt(
        self,
        prompt,  # 用户输入的提示文本
        device,  # 指定运行设备
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器的引导
        negative_prompt=None,  # 负面提示文本，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,  # 现有的提示嵌入，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示嵌入，默认为 None
        lora_scale: Optional[float] = None,  # Lora 缩放参数，默认为 None
        clip_skip: Optional[int] = None,  # 跳过的剪辑层，默认为 None
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    def run_safety_checker(self, image, device, dtype):  # 定义安全检查函数，接收图像、设备和数据类型
        # 检查安全检查器是否存在
        if self.safety_checker is None:
            has_nsfw_concept = None  # 如果没有安全检查器，则 NSFW 概念为 None
        else:
            # 如果图像是张量类型，进行后处理，转换为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果图像是 numpy 数组，直接转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 获取安全检查器的输入，转换为张量并移至指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器检查图像，返回处理后的图像和 NSFW 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 NSFW 概念
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
    # 解码潜在向量的方法
    def decode_latents(self, latents):
        # 警告信息，提示此方法已弃用，将在 1.0.0 中移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用 deprecate 函数记录弃用信息
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 根据配置的缩放因子调整潜在向量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在向量并返回图像数据
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像数据归一化到 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 始终将图像数据转换为 float32 类型，以确保兼容性并降低开销
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像数据
        return image
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，不同调度器的签名不同
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会忽略
        # eta 对应于 DDIM 论文中的 η，范围应在 [0, 1] 之间
    
        # 检查调度器步骤的参数是否接受 eta
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤关键字参数字典
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器步骤的参数是否接受 generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回额外步骤关键字参数字典
        return extra_step_kwargs
    
    # 检查输入参数的方法
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
        # 检查高度和宽度是否能被8整除
        if height % 8 != 0 or width % 8 != 0:
            # 抛出异常，给出高度和宽度的信息
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步数是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 抛出异常，给出回调步数的信息
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查回调输入是否在预期的输入列表中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 抛出异常，列出不在预期列表中的输入
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了提示和提示嵌入
        if prompt is not None and prompt_embeds is not None:
            # 抛出异常，提示只能提供一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否同时未提供提示和提示嵌入
        elif prompt is None and prompt_embeds is None:
            # 抛出异常，提示至少要提供一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示类型是否合法
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出异常，提示类型不符合要求
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了负提示和负提示嵌入
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出异常，提示只能提供一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负提示嵌入的形状是否一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出异常，给出形状不一致的信息
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在变量的函数，接受多个参数以控制形状和生成方式
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，包括批量大小、通道数和调整后的高度宽度
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器列表的长度是否与批量大小一致
            if isinstance(generator, list) and len(generator) != batch_size:
                # 如果不一致，则抛出值错误并提供相关信息
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果潜在变量为空，则生成新的随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果已提供潜在变量，则将其转移到指定设备
                latents = latents.to(device)
    
            # 根据调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回准备好的潜在变量
            return latents
    
        # 装饰器，指示该函数不需要计算梯度
        @torch.no_grad()
        def edit_model(
            self,
            source_prompt: str,
            destination_prompt: str,
            lamb: float = 0.1,
            restart_params: bool = True,
        # 装饰器，指示该函数不需要计算梯度
        @torch.no_grad()
        def __call__(
            self,
            # 允许输入字符串或字符串列表作为提示
            prompt: Union[str, List[str]] = None,
            # 可选参数，指定生成图像的高度
            height: Optional[int] = None,
            # 可选参数，指定生成图像的宽度
            width: Optional[int] = None,
            # 设置推理步骤的默认数量为50
            num_inference_steps: int = 50,
            # 设置引导比例的默认值为7.5
            guidance_scale: float = 7.5,
            # 可选参数，允许输入负面提示
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 可选参数，指定每个提示生成的图像数量，默认值为1
            num_images_per_prompt: Optional[int] = 1,
            # 设置eta的默认值为0.0
            eta: float = 0.0,
            # 可选参数，允许输入生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选参数，允许输入潜在变量的张量
            latents: Optional[torch.Tensor] = None,
            # 可选参数，允许输入提示嵌入的张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选参数，允许输入负面提示嵌入的张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选参数，指定输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 可选参数，控制是否返回字典形式的输出，默认为True
            return_dict: bool = True,
            # 可选回调函数，用于处理生成过程中每一步的信息
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 可选参数，指定回调的步骤数，默认为1
            callback_steps: int = 1,
            # 可选参数，允许传入交叉注意力的关键字参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选参数，允许指定跳过的clip层数
            clip_skip: Optional[int] = None,
```