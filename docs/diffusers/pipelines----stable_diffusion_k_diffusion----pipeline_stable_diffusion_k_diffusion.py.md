# `.\diffusers\pipelines\stable_diffusion_k_diffusion\pipeline_stable_diffusion_k_diffusion.py`

```py
# 版权声明，表示该文件由 HuggingFace 团队所有，所有权利保留
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）进行授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件
# 按“原样”分发，没有任何形式的明示或暗示的担保或条件。
# 有关许可证所涵盖的特定权限和
# 限制，请参阅许可证。

import importlib  # 导入模块以动态导入其他模块
import inspect  # 导入用于检查对象的模块
from typing import Callable, List, Optional, Union  # 导入类型注解

import torch  # 导入 PyTorch 库
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser  # 从 k_diffusion 导入去噪模型
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras  # 导入采样相关的函数和类

from ...image_processor import VaeImageProcessor  # 从相对路径导入 VAE 图像处理器
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器混合类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 LoRA 缩放的函数
from ...schedulers import LMSDiscreteScheduler  # 导入 LMS 离散调度器
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers  # 导入工具函数
from ...utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from ..stable_diffusion import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

class ModelWrapper:  # 定义模型包装类
    def __init__(self, model, alphas_cumprod):  # 初始化模型和累积 alpha 参数
        self.model = model  # 将传入的模型赋值给实例变量
        self.alphas_cumprod = alphas_cumprod  # 将传入的累积 alpha 参数赋值给实例变量

    def apply_model(self, *args, **kwargs):  # 定义应用模型的方法，接受可变参数
        if len(args) == 3:  # 如果参数数量为 3
            encoder_hidden_states = args[-1]  # 将最后一个参数作为编码器隐藏状态
            args = args[:2]  # 保留前两个参数
        if kwargs.get("cond", None) is not None:  # 如果关键字参数中有 "cond"
            encoder_hidden_states = kwargs.pop("cond")  # 从关键字参数中移除并赋值给编码器隐藏状态
        return self.model(*args, encoder_hidden_states=encoder_hidden_states, **kwargs).sample  # 调用模型并返回样本

class StableDiffusionKDiffusionPipeline(  # 定义稳定扩散 K 扩散管道类
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin  # 继承多个基类
):
    r"""  # 文档字符串，描述此类的功能
    用于文本到图像生成的管道，使用稳定扩散模型。

    该模型继承自 [`DiffusionPipeline`]. 查看超类文档以获取库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重

    <Tip warning={true}>

        这是一个实验性管道，未来可能会发生变化。

    </Tip>
    # 文档字符串，描述参数的含义
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器模型，用于对图像进行编码和解码，转换为潜在表示
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):  # 冻结的文本编码器，稳定扩散使用 CLIP 的文本部分
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)，具体为
            [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        tokenizer (`CLIPTokenizer`):  # CLIP 的分词器，负责将文本转换为模型可接受的输入格式
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]):  # 条件 U-Net 结构，用于对编码的图像潜在表示去噪
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):  # 用于与 U-Net 结合的调度器，帮助去噪图像潜在表示
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):  # 分类模块，评估生成的图像是否可能具有攻击性或有害
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):  # 模型从生成的图像中提取特征，以作为 `safety_checker` 的输入
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # 定义模型在 CPU 上的卸载顺序，从文本编码器到 U-Net 再到 VAE
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件，包含安全检查器和特征提取器
    _optional_components = ["safety_checker", "feature_extractor"]
    # 定义不包括在 CPU 卸载中的组件，特定为安全检查器
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法，接受多个组件作为参数
    def __init__(
        self,
        vae,  # 传入变分自编码器实例
        text_encoder,  # 传入文本编码器实例
        tokenizer,  # 传入分词器实例
        unet,  # 传入条件 U-Net 实例
        scheduler,  # 传入调度器实例
        safety_checker,  # 传入安全检查器实例
        feature_extractor,  # 传入特征提取器实例
        requires_safety_checker: bool = True,  # 指示是否需要安全检查器的布尔参数，默认为 True
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 记录当前类是实验性管道，可能会在未来发生变化的信息
        logger.info(
            f"{self.__class__} is an experimntal pipeline and is likely to change in the future. We recommend to use"
            " this pipeline for fast experimentation / iteration if needed, but advice to rely on existing pipelines"
            " as defined in https://huggingface.co/docs/diffusers/api/schedulers#implemented-schedulers for"
            " production settings."
        )

        # 从 LMS 配置中获取正确的 sigmas
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        # 注册模型组件，包括变分自编码器、文本编码器、分词器、UNet、调度器、安全检查器和特征提取器
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # 将安全检查器的要求注册到配置中
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器，使用计算出的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 封装模型，将 UNet 和调度器的累积 alpha 传入模型包装器
        model = ModelWrapper(unet, scheduler.alphas_cumprod)
        # 根据预测类型选择合适的去噪模型
        if scheduler.config.prediction_type == "v_prediction":
            self.k_diffusion_model = CompVisVDenoiser(model)
        else:
            self.k_diffusion_model = CompVisDenoiser(model)

    # 设置调度器类型的方法
    def set_scheduler(self, scheduler_type: str):
        # 动态导入 k_diffusion 库
        library = importlib.import_module("k_diffusion")
        # 获取采样模块
        sampling = getattr(library, "sampling")
        try:
            # 尝试获取指定的采样器
            self.sampler = getattr(sampling, scheduler_type)
        except Exception:
            # 如果发生异常，收集有效的采样器名称
            valid_samplers = []
            for s in dir(sampling):
                if "sample_" in s:
                    valid_samplers.append(s)

            # 抛出无效调度器类型的异常，并提供有效的采样器列表
            raise ValueError(f"Invalid scheduler type {scheduler_type}. Please choose one of {valid_samplers}.")

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的方法
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        # 可选的提示嵌入，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示嵌入，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 LORA 缩放因子，默认为 None
        lora_scale: Optional[float] = None,
        # 额外的关键字参数
        **kwargs,
    ):
        # 定义一个警告信息，提示 `_encode_prompt()` 已被弃用，并将在未来版本中移除，建议使用 `encode_prompt()` 替代
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数，记录弃用信息，设置标准警告为 False
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法，生成提示嵌入的元组
        prompt_embeds_tuple = self.encode_prompt(
            # 提供必要的参数给 encode_prompt 方法
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # 连接提示嵌入元组中的两个部分，方便向后兼容
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回合并后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的部分
    def encode_prompt(
        self,
        # 定义参数，分别用于处理提示信息和设备设置等
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的部分
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则设置 NSFW 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入图像是张量格式，进行后处理并转换为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入图像不是张量，则将其转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征，并将其转换为适合设备的张量格式
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器处理图像，并返回处理后的图像及 NSFW 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 NSFW 概念
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制的部分
    # 解码潜在变量的方法
        def decode_latents(self, latents):
            # 设置弃用警告信息，提示用户此方法将在1.0.0版本中移除
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 调用弃用函数，记录弃用信息
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 根据缩放因子调整潜在变量
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在变量，返回的第一个元素是解码后的图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像数据缩放到[0, 1]范围
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像从GPU移动到CPU，并调整维度顺序，转换为float32类型以保持兼容性
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回解码后的图像
            return image
    
        # 检查输入参数的方法
        def check_inputs(
            self,
            # 提示文本
            prompt,
            # 图像高度
            height,
            # 图像宽度
            width,
            # 回调步骤
            callback_steps,
            # 可选的负提示文本
            negative_prompt=None,
            # 可选的提示嵌入
            prompt_embeds=None,
            # 可选的负提示嵌入
            negative_prompt_embeds=None,
            # 可选的在步骤结束时回调的张量输入
            callback_on_step_end_tensor_inputs=None,
    ):
        # 检查高度和宽度是否能被 8 整除，若不能则抛出错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤是否为正整数，若不是则抛出错误
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查回调结束时的张量输入是否有效，若有无效项则抛出错误
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查同时提供了 prompt 和 prompt_embeds，若同时提供则抛出错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都未提供，若都未提供则抛出错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否为 str 或 list，若不是则抛出错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查同时提供了 negative_prompt 和 negative_prompt_embeds，若同时提供则抛出错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 的形状是否一致，若不一致则抛出错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
    # 准备潜在向量的函数，输入参数包括批大小、通道数、高度、宽度等
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 根据输入参数计算潜在向量的形状
        shape = (
            batch_size,  # 批处理的大小
            num_channels_latents,  # 潜在向量的通道数
            int(height) // self.vae_scale_factor,  # 高度经过缩放因子调整
            int(width) // self.vae_scale_factor,  # 宽度经过缩放因子调整
        )
        # 如果没有提供潜在向量，则随机生成一个
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供的潜在向量形状不匹配，则抛出错误
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在向量移动到指定设备
            latents = latents.to(device)
    
        # 根据调度器所需的标准差缩放初始噪声
        return latents
    
    # 该方法用于模型调用，且不计算梯度
    @torch.no_grad()
    def __call__(
        # 输入提示，可能是单个字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 图像的高度，默认为 None
        height: Optional[int] = None,
        # 图像的宽度，默认为 None
        width: Optional[int] = None,
        # 推理步骤的数量，默认为 50
        num_inference_steps: int = 50,
        # 引导尺度，默认为 7.5
        guidance_scale: float = 7.5,
        # 负提示，可能是单个字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 额外的噪声，默认为 0.0
        eta: float = 0.0,
        # 随机数生成器，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 预先生成的潜在向量，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 提示的嵌入，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示的嵌入，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式，默认为 True
        return_dict: bool = True,
        # 回调函数，默认为 None
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调步骤，默认为 1
        callback_steps: int = 1,
        # 是否使用 Karras Sigma，默认为 False
        use_karras_sigmas: Optional[bool] = False,
        # 噪声采样器的种子，默认为 None
        noise_sampler_seed: Optional[int] = None,
        # 跳过的剪辑数量，默认为 None
        clip_skip: int = None,
```