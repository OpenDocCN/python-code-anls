# `.\diffusers\pipelines\deprecated\alt_diffusion\pipeline_alt_diffusion.py`

```py
# 版权声明，表示该文件的所有权归 HuggingFace 团队所有
# 该文件在 Apache 许可证 2.0 下授权使用
# 许可证的详细信息可以在以下链接获取
# http://www.apache.org/licenses/LICENSE-2.0
# 根据许可证，软件按“原样”提供，不提供任何形式的保证
# 详细的许可证条款可以在下面查看
import inspect  # 导入 inspect 模块，用于检查对象的类型和属性
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注解以增强可读性和静态检查

import torch  # 导入 PyTorch 库，进行深度学习相关操作
from packaging import version  # 导入 version 模块，处理版本信息
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, XLMRobertaTokenizer  # 导入变换器模型和处理器

# 从当前模块的父级导入配置和处理器工具
from ....configuration_utils import FrozenDict  
from ....image_processor import PipelineImageInput, VaeImageProcessor  

# 从加载器模块导入多种混合类
from ....loaders import (
    FromSingleFileMixin,  # 单文件加载混合
    IPAdapterMixin,  # IP 适配器混合
    StableDiffusionLoraLoaderMixin,  # 稳定扩散 Lora 加载混合
    TextualInversionLoaderMixin,  # 文本反转加载混合
)

# 从模型模块导入不同的模型类
from ....models import AutoencoderKL, ImageProjection, UNet2DConditionModel  
from ....models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 比例的函数
from ....schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ....utils import (
    USE_PEFT_BACKEND,  # 指示是否使用 PEFT 后端
    deprecate,  # 导入废弃功能的装饰器
    logging,  # 导入日志工具
    replace_example_docstring,  # 替换示例文档字符串的工具
    scale_lora_layers,  # 调整 Lora 层的比例
    unscale_lora_layers,  # 恢复 Lora 层的比例
)
from ....utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ...pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入安全检查器
from .modeling_roberta_series import RobertaSeriesModelWithTransformation  # 导入 Roberta 系列模型
from .pipeline_output import AltDiffusionPipelineOutput  # 导入备用扩散管道输出

logger = logging.get_logger(__name__)  # 初始化日志记录器，用于记录模块相关信息

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，用于展示如何使用管道
    Examples:  # 示例标题
        ```py  # 示例代码开始
        >>> import torch  # 导入 PyTorch
        >>> from diffusers import AltDiffusionPipeline  # 导入备用扩散管道

        >>> pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9", torch_dtype=torch.float16)  # 从预训练模型加载管道
        >>> pipe = pipe.to("cuda")  # 将管道移至 CUDA 设备

        >>> # "dark elf princess, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and fuji choko and viktoria gavrilenko and hoang lap"
        >>> prompt = "黑暗精灵公主，非常详细，幻想，非常详细，数字绘画，概念艺术，敏锐的焦点，插图"  # 定义提示内容
        >>> image = pipe(prompt).images[0]  # 使用提示生成图像并获取第一张
        ```py  # 示例代码结束
"""

# 从扩散管道中复制的函数，调整噪声配置
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):  # 定义重标定噪声配置的函数
    """
    根据指导重标定噪声配置。基于研究[Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)。见第 3.4 节
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)  # 计算噪声预测文本的标准差
    # 计算噪声配置的标准差，沿指定维度进行
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # 对指导结果进行重新缩放，以修正过度曝光的问题
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # 将重新缩放的噪声与原始指导结果混合，通过引导缩放因子避免“平淡”图像
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        # 返回调整后的噪声配置
        return noise_cfg
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 复制的代码
def retrieve_timesteps(
    # 调度器对象，用于获取时间步
    scheduler,
    # 生成样本时使用的扩散步骤数量，可选参数
    num_inference_steps: Optional[int] = None,
    # 时间步应移动到的设备，字符串或 torch.device 类型
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步的列表，可选参数
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 的列表，可选参数
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中获取时间步。处理自定义时间步。
    任何 kwargs 将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步应移动到的设备。如果为 `None`，则时间步不被移动。
        timesteps (`List[int]`, *可选*):
            自定义时间步，用于覆盖调度器的时间步间距策略。如果传入 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma，用于覆盖调度器的时间步间距策略。如果传入 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是调度器的时间步调度，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传入了时间步和 sigma
    if timesteps is not None and sigmas is not None:
        # 抛出错误，因为不能同时使用这两个参数
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查是否传入了自定义时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义时间步，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传入自定义时间步和设备
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 检查是否传入了自定义 sigma
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义 sigma，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传入自定义 sigma 和设备
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 否则，设置推理步骤和设备，传入额外的参数
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器的时间步列表
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤的数量
        return timesteps, num_inference_steps
# 定义一个名为 AltDiffusionPipeline 的类，继承多个父类以实现文本到图像生成的功能
class AltDiffusionPipeline(
    # 继承自 DiffusionPipeline，提供基础的扩散管道功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin，提供稳定扩散相关的功能
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin，提供文本反转加载功能
    TextualInversionLoaderMixin,
    # 继承自 StableDiffusionLoraLoaderMixin，提供 LoRA 权重的加载和保存功能
    StableDiffusionLoraLoaderMixin,
    # 继承自 IPAdapterMixin，提供 IP 适配器的加载功能
    IPAdapterMixin,
    # 继承自 FromSingleFileMixin，提供从单个文件加载的功能
    FromSingleFileMixin,
):
    # 文档字符串，描述该类的用途和功能
    r"""
    Pipeline for text-to-image generation using Alt Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.RobertaSeriesModelWithTransformation`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.XLMRobertaTokenizer`]):
            A `XLMRobertaTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    # 定义模型在 CPU 上卸载的顺序，确保特定组件在内存中的顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件的列表，这些组件可能在使用时被添加
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义在 CPU 卸载时排除的组件，这些组件不会被卸载
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义需要回调的张量输入列表，这些输入将在特定操作中被处理
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化类的构造函数，接收多个参数以配置模型组件
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器实例
            text_encoder: RobertaSeriesModelWithTransformation,  # 文本编码器实例
            tokenizer: XLMRobertaTokenizer,  # 令牌化工具
            unet: UNet2DConditionModel,  # UNet模型实例
            scheduler: KarrasDiffusionSchedulers,  # 调度器实例
            safety_checker: StableDiffusionSafetyChecker,  # 安全检查器实例
            feature_extractor: CLIPImageProcessor,  # 特征提取器实例
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器实例
            requires_safety_checker: bool = True,  # 是否需要安全检查器的布尔值
        def _encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备类型（如CPU或GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否执行无分类器引导的布尔值
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入张量
            lora_scale: Optional[float] = None,  # 可选的LoRA缩放因子
            **kwargs,  # 额外的关键字参数
        ):
            # 生成弃用警告信息，提示用户使用新方法
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用弃用处理函数，记录弃用警告
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用encode_prompt方法处理提示文本，并返回嵌入的元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 输入的提示文本
                device=device,  # 设备类型
                num_images_per_prompt=num_images_per_prompt,  # 图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 引导选项
                negative_prompt=negative_prompt,  # 负面提示文本
                prompt_embeds=prompt_embeds,  # 提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 负面嵌入
                lora_scale=lora_scale,  # LoRA缩放
                **kwargs,  # 额外参数
            )
    
            # 将返回的嵌入元组拼接为一个张量，兼容旧版本
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回最终的提示嵌入张量
            return prompt_embeds
    
        # 定义新的提示编码方法，接收多个参数以处理输入提示
        def encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备类型（如CPU或GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否执行无分类器引导的布尔值
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入张量
            lora_scale: Optional[float] = None,  # 可选的LoRA缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪裁跳过参数
    # 定义一个方法，用于编码图像，参数包括图像、设备、每个提示的图像数量和可选的隐藏状态输出
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的 dtype
        dtype = next(self.image_encoder.parameters()).dtype

        # 检查传入的图像是否为张量类型，如果不是，则使用特征提取器处理图像
        if not isinstance(image, torch.Tensor):
            # 将图像转换为 PyTorch 张量，并提取像素值
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像移动到指定的设备，并转换为指定的数据类型
        image = image.to(device=device, dtype=dtype)
        
        # 如果请求输出隐藏状态
        if output_hidden_states:
            # 获取图像编码器的隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 重复编码的隐藏状态，以适应每个提示的图像数量
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 对无条件图像进行编码，生成其隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 重复无条件图像编码的隐藏状态
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回图像和无条件图像的编码隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 如果不需要隐藏状态，则直接获取图像嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 重复图像嵌入，以适应每个提示的图像数量
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建与图像嵌入形状相同的零张量作为无条件图像嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回图像嵌入和无条件图像嵌入
            return image_embeds, uncond_image_embeds

    # 定义一个方法，用于运行安全检查器，参数包括图像、设备和数据类型
    def run_safety_checker(self, image, device, dtype):
        # 检查安全检查器是否为 None
        if self.safety_checker is None:
            # 如果是，则无不当内容概念标志为 None
            has_nsfw_concept = None
        else:
            # 检查传入的图像是否为张量类型
            if torch.is_tensor(image):
                # 如果是张量，则将其后处理为 PIL 图像格式
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果不是张量，则将其转换为 PIL 图像
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理后处理后的图像，并将其移动到指定的设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，检查图像和处理后的输入
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回检查后的图像和不当内容概念标志
        return image, has_nsfw_concept

    # 定义一个方法，用于解码潜在变量
    def decode_latents(self, latents):
        # 定义弃用警告消息
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 发出弃用警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 对潜在变量进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents
        # 使用 VAE 解码潜在变量，返回解码后的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 对图像进行归一化和裁剪，以确保值在 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像移动到 CPU，并调整维度顺序，转换为 numpy 数组，保持为 float32 类型
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回解码后的图像
        return image
    # 定义一个方法，用于准备额外的调度器步骤参数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为不同调度器的参数签名可能不同
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略它
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个空字典，用于存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果接受 eta 参数，将其添加到额外步骤参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，将其添加到额外步骤参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    # 定义一个方法，用于检查输入参数的有效性
    def check_inputs(
        self,
        prompt,  # 用户输入的提示文本
        height,  # 输出图像的高度
        width,   # 输出图像的宽度
        callback_steps,  # 用于回调的步骤数量
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds=None,  # 可选的提示嵌入向量
        negative_prompt_embeds=None,  # 可选的负面提示嵌入向量
        callback_on_step_end_tensor_inputs=None,  # 可选的回调输入张量
    ):
        # 检查高度和宽度是否能被8整除
        if height % 8 != 0 or width % 8 != 0:
            # 如果不能整除，抛出值错误
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果不是，抛出值错误
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查回调结束张量输入是否在允许的张量输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果不在，抛出值错误
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查提示和提示嵌入是否同时提供
        if prompt is not None and prompt_embeds is not None:
            # 如果同时提供，抛出值错误
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否同时未提供提示和提示嵌入
        elif prompt is None and prompt_embeds is None:
            # 如果未提供，抛出值错误
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示类型是否为字符串或列表
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果不是，抛出值错误
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查负提示和负提示嵌入是否同时提供
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果同时提供，抛出值错误
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负提示嵌入形状是否匹配
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不匹配，抛出值错误
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
    # 准备潜在变量，根据输入参数设置形状
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，包括批量大小、通道数和缩放后的高度与宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且其长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不匹配，抛出值错误并提示
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果潜在变量未提供，生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，将其转移到指定设备
            latents = latents.to(device)

        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 获取引导比例嵌入的函数
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        查看 https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                在这些时间步生成嵌入向量
            embedding_dim (`int`, *optional*, defaults to 512):
                生成嵌入的维度
            dtype:
                生成嵌入的数据类型

        Returns:
            `torch.Tensor`: 形状为 `(len(timesteps), embedding_dim)` 的嵌入向量
        """
        # 确保输入的 w 是一维的
        assert len(w.shape) == 1
        # 将 w 的值扩大 1000 倍
        w = w * 1000.0

        # 计算半维度
        half_dim = embedding_dim // 2
        # 计算嵌入的基本值
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成指数衰减的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将 w 转换为目标数据类型并进行广播
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦值连接起来形成嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度为奇数，则在末尾进行零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保嵌入的形状符合预期
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回最终的嵌入
        return emb

    # 引导比例的属性，返回存储的引导比例值
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 引导重缩放的属性，返回存储的重缩放值
    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # clip_skip 属性，返回存储的跳过值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 判断是否执行无分类器引导的属性，基于引导比例和条件配置
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 跨注意力参数的属性，返回存储的参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 时间步数的属性，返回存储的时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 使用装饰器标记后续代码不计算梯度
    @torch.no_grad()
    # 用于替换示例文档字符串的装饰器
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义可调用方法，接收多个参数
        def __call__(
            # 用户提供的文本提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 生成图像的高度
            height: Optional[int] = None,
            # 生成图像的宽度
            width: Optional[int] = None,
            # 推理步骤的数量，默认为50
            num_inference_steps: int = 50,
            # 可选的时间步列表
            timesteps: List[int] = None,
            # 可选的标准差列表
            sigmas: List[float] = None,
            # 指导缩放系数，默认为7.5
            guidance_scale: float = 7.5,
            # 可选的负面提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 额外的超参数，默认为0.0
            eta: float = 0.0,
            # 可选的随机数生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量张量
            latents: Optional[torch.Tensor] = None,
            # 可选的提示嵌入张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的适配器输入图像
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输出类型，默认为“pil”格式
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为True
            return_dict: bool = True,
            # 可选的跨注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指导再缩放的系数，默认为0.0
            guidance_rescale: float = 0.0,
            # 可选的跳过的剪辑步数
            clip_skip: Optional[int] = None,
            # 可选的回调函数，在步骤结束时执行
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 在步骤结束时回调的张量输入列表，默认为“latents”
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他可变参数
            **kwargs,
```