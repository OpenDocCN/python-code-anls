# `.\diffusers\pipelines\semantic_stable_diffusion\pipeline_semantic_stable_diffusion.py`

```py
# 导入 Python 的 inspect 模块，用于获取信息
import inspect
# 从 itertools 模块导入 repeat 函数，用于生成重复元素
from itertools import repeat
# 导入类型提示所需的 Callable, List, Optional, Union
from typing import Callable, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 图像处理器、文本模型和分词器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 从自定义模块导入 VAE 图像处理器
from ...image_processor import VaeImageProcessor
# 从自定义模型中导入 AutoencoderKL 和 UNet2DConditionModel
from ...models import AutoencoderKL, UNet2DConditionModel
# 从安全检查器模块导入 StableDiffusionSafetyChecker
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 从调度器模块导入 KarrasDiffusionSchedulers
from ...schedulers import KarrasDiffusionSchedulers
# 从工具模块导入弃用和日志记录功能
from ...utils import deprecate, logging
# 从 PyTorch 工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入 DiffusionPipeline 和 StableDiffusionMixin
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从管道输出模块导入 SemanticStableDiffusionPipelineOutput
from .pipeline_output import SemanticStableDiffusionPipelineOutput

# 创建一个日志记录器，记录当前模块的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个用于语义稳定扩散的管道类，继承自 DiffusionPipeline 和 StableDiffusionMixin
class SemanticStableDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    使用稳定扩散进行文本到图像生成的管道，支持潜在编辑。

    此模型继承自 [`DiffusionPipeline`]，并基于 [`StableDiffusionPipeline`]。有关所有管道的通用方法的文档，
    请查阅超类文档（下载、保存、在特定设备上运行等）。

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行分词的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码后的图像潜在表示的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合以去噪编码图像潜在表示的调度器，可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
        safety_checker ([`Q16SafetyChecker`]):
            评估生成图像是否可能被视为冒犯或有害的分类模块。
            有关模型潜在危害的更多详细信息，请参阅 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成图像中提取特征的 `CLIPImageProcessor`；用于 `safety_checker` 的输入。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor"]

    # 初始化方法，接受多个参数以配置管道
    def __init__(
        self,
        vae: AutoencoderKL,  # 接受变分自编码器模型
        text_encoder: CLIPTextModel,  # 接受文本编码器模型
        tokenizer: CLIPTokenizer,  # 接受文本分词器
        unet: UNet2DConditionModel,  # 接受去噪网络模型
        scheduler: KarrasDiffusionSchedulers,  # 接受调度器
        safety_checker: StableDiffusionSafetyChecker,  # 接受安全检查器
        feature_extractor: CLIPImageProcessor,  # 接受图像处理器
        requires_safety_checker: bool = True,  # 可选参数，指示是否需要安全检查器
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 如果未提供安全检查器且需要安全检查器，发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                # 输出警告信息，提示用户需要遵守使用条件
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 如果提供了安全检查器但未提供特征提取器，抛出异常
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                # 抛出值错误，提示用户需定义特征提取器
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册多个模块，方便后续调用
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器实例，用于后续图像处理
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 将是否需要安全检查器的配置注册到实例
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的方法
    def run_safety_checker(self, image, device, dtype):
        # 如果没有安全检查器，初始化 nsfw 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入是张量，进行后处理转换为 PIL 图像
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入不是张量，将其转换为 PIL 图像
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 将图像输入特征提取器并转换为适合设备的张量
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，获取处理后的图像和 nsfw 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 nsfw 概念
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制的方法
    # 解码潜在表示的方法
    def decode_latents(self, latents):
        # 提示用户该方法已被弃用，并将在1.0.0中移除，建议使用新的方法
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用警告函数，标记该方法为不推荐使用
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 将潜在表示根据缩放因子进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在表示，返回的第一项是解码后的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 对图像进行归一化处理，将值限制在[0, 1]范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为float32格式，兼容bfloat16，且不会造成显著开销
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回解码后的图像
        return image

    # 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs复制而来
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器具有相同的参数签名
        # eta（η）仅用于DDIMScheduler，其他调度器将忽略该参数
        # eta对应于DDIM论文中的η，范围应在[0, 1]之间

        # 检查调度器的步骤方法是否接受eta参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个字典以存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果接受eta参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤方法是否接受generator参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受generator参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 从diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs复制而来
    def check_inputs(
        # 方法参数包括提示、图像高度、宽度、回调步骤等
        prompt,
        height,
        width,
        callback_steps,
        # 负面提示、提示嵌入和负面提示嵌入可选参数
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查高度和宽度是否能被8整除，若不满足则引发错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步数是否为正整数，若不满足则引发错误
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 检查在步骤结束时的张量输入是否有效，若无效则引发错误
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了提示和提示嵌入，若是则引发错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查提示和提示嵌入是否都未定义，若是则引发错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示的类型是否有效，若无效则引发错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查负面提示和负面提示嵌入是否同时提供，若是则引发错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负面提示嵌入的形状是否一致，若不一致则引发错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 中复制
    # 准备潜在变量的函数，接收一系列参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，包括批量大小、通道数、高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且长度与批量大小不匹配，若不匹配则抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果潜在变量为 None，则生成随机的潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果潜在变量不为 None，则将其移动到指定设备
            latents = latents.to(device)
    
        # 根据调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents
    
    # 禁用梯度计算的上下文装饰器
    @torch.no_grad()
    def __call__(
        # 接收多个参数，包括提示、图像高度和宽度、推理步骤数等
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeddings: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        edit_momentum_scale: Optional[float] = 0.1,
        edit_mom_beta: Optional[float] = 0.4,
        edit_weights: Optional[List[float]] = None,
        sem_guidance: Optional[List[torch.Tensor]] = None,
```