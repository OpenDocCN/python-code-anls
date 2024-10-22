# `.\diffusers\pipelines\deprecated\stable_diffusion_variants\pipeline_stable_diffusion_paradigms.py`

```py
# 版权所有 2024 ParaDiGMS 作者和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，依据许可证分发的软件
# 是按“原样”提供的，没有任何形式的明示或暗示的担保或条件。
# 有关许可证的特定权限和限制，请参见许可证。

# 导入 inspect 模块以进行对象检查
import inspect
# 从 typing 模块导入类型提示相关的类
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 PyTorch 库以进行深度学习操作
import torch
# 从 transformers 库导入 CLIP 图像处理器、文本模型和分词器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 导入自定义的图像处理器
from ....image_processor import VaeImageProcessor
# 导入与加载相关的混合类
from ....loaders import FromSingleFileMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入用于自动编码器和条件 UNet 的模型
from ....models import AutoencoderKL, UNet2DConditionModel
# 从 lora 模块导入调整 lora 规模的函数
from ....models.lora import adjust_lora_scale_text_encoder
# 导入 Karras 扩散调度器
from ....schedulers import KarrasDiffusionSchedulers
# 导入实用工具模块中的各种功能
from ....utils import (
    USE_PEFT_BACKEND,  # 用于 PEFT 后端的标志
    deprecate,  # 用于标记弃用功能的装饰器
    logging,  # 日志记录功能
    replace_example_docstring,  # 替换示例文档字符串的功能
    scale_lora_layers,  # 调整 lora 层规模的功能
    unscale_lora_layers,  # 反调整 lora 层规模的功能
)
# 从 torch_utils 模块导入生成随机张量的功能
from ....utils.torch_utils import randn_tensor
# 导入扩散管道和稳定扩散混合类
from ...pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入稳定扩散管道输出类
from ...stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
# 导入稳定扩散安全检查器
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# 创建日志记录器，用于当前模块的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DDPMParallelScheduler
        >>> from diffusers import StableDiffusionParadigmsPipeline

        >>> scheduler = DDPMParallelScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

        >>> pipe = StableDiffusionParadigmsPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", scheduler=scheduler, torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> ngpu, batch_per_device = torch.cuda.device_count(), 5
        >>> pipe.wrapped_unet = torch.nn.DataParallel(pipe.unet, device_ids=[d for d in range(ngpu)])

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt, parallel=ngpu * batch_per_device, num_inference_steps=1000).images[0]
        ```py
"""

# 定义 StableDiffusionParadigmsPipeline 类，继承多个混合类以实现功能
class StableDiffusionParadigmsPipeline(
    DiffusionPipeline,  # 从扩散管道继承
    StableDiffusionMixin,  # 从稳定扩散混合类继承
    TextualInversionLoaderMixin,  # 从文本反转加载混合类继承
    StableDiffusionLoraLoaderMixin,  # 从稳定扩散 lora 加载混合类继承
    FromSingleFileMixin,  # 从单文件加载混合类继承
):
    r"""
    用于文本到图像生成的管道，使用稳定扩散的并行化版本。

    此模型继承自 [`DiffusionPipeline`]。有关通用方法的文档，请查看超类文档
    # 实现所有管道的功能（下载、保存、在特定设备上运行等）。
    
    # 管道还继承以下加载方法：
    # - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
    # - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
    # - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
    # - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
    
    # 参数说明：
    # vae ([`AutoencoderKL`]):
    #    变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示。
    # text_encoder ([`~transformers.CLIPTextModel`]):
    #    冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
    # tokenizer ([`~transformers.CLIPTokenizer`]):
    #    一个 `CLIPTokenizer` 用于对文本进行标记化。
    # unet ([`UNet2DConditionModel`]):
    #    一个 `UNet2DConditionModel` 用于去噪编码的图像潜在。
    # scheduler ([`SchedulerMixin`]):
    #    用于与 `unet` 结合使用的调度器，用于去噪编码的图像潜在。可以是
    #    [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的一个。
    # safety_checker ([`StableDiffusionSafetyChecker`]):
    #    分类模块，估计生成的图像是否可能被认为是冒犯或有害的。
    #    请参考 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 获取更多关于模型潜在危害的详细信息。
    # feature_extractor ([`~transformers.CLIPImageProcessor`]):
    #    一个 `CLIPImageProcessor` 用于从生成的图像中提取特征；作为 `safety_checker` 的输入。
    
    # 定义模型的 CPU 离线加载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor"]
    # 定义排除在 CPU 离线加载之外的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    
    # 初始化方法，接受多个参数
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器模型
        text_encoder: CLIPTextModel,  # 文本编码器
        tokenizer: CLIPTokenizer,  # 文本标记器
        unet: UNet2DConditionModel,  # UNet2D 条件模型
        scheduler: KarrasDiffusionSchedulers,  # 调度器
        safety_checker: StableDiffusionSafetyChecker,  # 安全检查模块
        feature_extractor: CLIPImageProcessor,  # 特征提取器
        requires_safety_checker: bool = True,  # 是否需要安全检查器
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查是否禁用安全检查器，并且需要安全检查器
        if safety_checker is None and requires_safety_checker:
            # 记录警告信息，提醒用户禁用安全检查器的风险
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查是否提供了安全检查器但未提供特征提取器
        if safety_checker is not None and feature_extractor is None:
            # 抛出错误，提示用户需要定义特征提取器
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册各个模块到当前实例
        self.register_modules(
            vae=vae,  # 变分自编码器
            text_encoder=text_encoder,  # 文本编码器
            tokenizer=tokenizer,  # 分词器
            unet=unet,  # U-Net 模型
            scheduler=scheduler,  # 调度器
            safety_checker=safety_checker,  # 安全检查器
            feature_extractor=feature_extractor,  # 特征提取器
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化图像处理器，使用 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 将是否需要安全检查器的配置注册到当前实例
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        # 用于在多个 GPU 上运行多个去噪步骤时，将 unet 包装为 torch.nn.DataParallel
        self.wrapped_unet = self.unet

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的函数
    def _encode_prompt(
        self,
        prompt,  # 输入的提示文本
        device,  # 设备类型（CPU或GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
        **kwargs,  # 其他可选参数
    ):
        # 设置弃用信息，提醒用户该方法即将被移除，建议使用新方法
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数，传递弃用信息和版本号
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法，获取提示的嵌入元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 输入提示文本
            device=device,  # 计算设备
            num_images_per_prompt=num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=negative_prompt,  # 负提示文本
            prompt_embeds=prompt_embeds,  # 提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,  # 负提示嵌入
            lora_scale=lora_scale,  # LORA 缩放因子
            **kwargs,  # 其他额外参数
        )

        # 将返回的嵌入元组进行拼接，以支持向后兼容
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回拼接后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 中复制的 encode_prompt 方法
    def encode_prompt(
        self,
        prompt,  # 输入的提示文本
        device,  # 计算设备
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 负提示文本，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 LORA 缩放因子
        clip_skip: Optional[int] = None,  # 可选的跳过剪辑参数
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的代码
    def run_safety_checker(self, image, device, dtype):
        # 检查是否存在安全检查器
        if self.safety_checker is None:
            has_nsfw_concept = None  # 如果没有安全检查器，则设置无 NSFW 概念为 None
        else:
            # 如果输入是张量格式，则进行后处理为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入不是张量，转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理图像，返回张量形式
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 调用安全检查器，检查图像是否包含 NSFW 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image,  # 输入图像
                clip_input=safety_checker_input.pixel_values.to(dtype)  # 安全检查的特征输入
            )
        # 返回处理后的图像及是否存在 NSFW 概念
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的代码
    # 定义一个方法，用于准备额外的参数以供调度器步骤使用
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并非所有调度器都有相同的签名
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 其值应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个字典以存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    # 定义一个方法，用于检查输入参数的有效性
    def check_inputs(
        self,
        prompt,  # 输入的提示文本
        height,  # 图像的高度
        width,   # 图像的宽度
        callback_steps,  # 回调步骤的频率
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
        callback_on_step_end_tensor_inputs=None,  # 可选的在步骤结束时的回调张量输入
    ):
        # 检查高度和宽度是否为 8 的倍数
        if height % 8 != 0 or width % 8 != 0:
            # 抛出错误，如果高度或宽度不符合要求
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步数是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 抛出错误，如果回调步数不符合要求
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查回调结束时的张量输入是否在允许的输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 抛出错误，如果不在允许的输入中
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了提示和提示嵌入
        if prompt is not None and prompt_embeds is not None:
            # 抛出错误，不能同时提供提示和提示嵌入
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查提示和提示嵌入是否均未定义
        elif prompt is None and prompt_embeds is None:
            # 抛出错误，必须提供一个提示或提示嵌入
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示的类型
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出错误，如果提示不是字符串或列表
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了负提示和负提示嵌入
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出错误，不能同时提供负提示和负提示嵌入
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负提示嵌入的形状是否相同
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出错误，如果形状不一致
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    # 准备潜在变量，设置其形状和属性
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，考虑批量大小、通道数和缩放因子
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器列表的长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        # 如果潜在变量未提供，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将提供的潜在变量移动到指定设备
            latents = latents.to(device)
    
        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents
    
    # 计算输入张量在指定维度上的累积和
    def _cumsum(self, input, dim, debug=False):
        # 如果调试模式开启，则在CPU上执行累积和以确保可重复性
        if debug:
            # cumsum_cuda_kernel没有确定性实现，故在CPU上执行
            return torch.cumsum(input.cpu().float(), dim=dim).to(input.device)
        else:
            # 在指定维度上直接计算累积和
            return torch.cumsum(input, dim=dim)
    
    # 调用方法，接受多种参数以生成输出
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        parallel: int = 10,
        tolerance: float = 0.1,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        clip_skip: int = None,
```