# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion_img2img.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件是按“原样”提供的，
# 不提供任何明示或暗示的担保或条件。
# 有关许可证下权限和限制的具体语言，请参阅许可证。

# 导入 inspect 模块，用于获取对象的签名和其他信息
import inspect
# 从 typing 模块导入类型注解，方便类型提示
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 PIL 库，用于图像处理
import PIL.Image
# 导入 PyTorch 库，用于深度学习
import torch
# 导入版本管理工具，用于版本比较
from packaging import version
# 从 transformers 库导入 CLIP 相关模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 导入回调函数相关的类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入配置相关的工具类
from ...configuration_utils import FrozenDict
# 导入图像处理相关的输入类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入多种加载器混合类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入模型相关的类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
# 导入 Lora 调整函数
from ...models.lora import adjust_lora_scale_text_encoder
# 导入调度器
from ...schedulers import KarrasDiffusionSchedulers
# 导入实用工具函数
from ...utils import (
    PIL_INTERPOLATION,  # PIL 图像插值方法
    USE_PEFT_BACKEND,   # 是否使用 PEFT 后端
    deprecate,          # 用于标记弃用功能
    logging,            # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 调整 Lora 层的比例
    unscale_lora_layers,  # 取消 Lora 层的比例
)
# 导入随机张量生成工具
from ...utils.torch_utils import randn_tensor
# 导入管道相关工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入输出类
from . import StableDiffusionPipelineOutput
# 导入安全检查器
from .safety_checker import StableDiffusionSafetyChecker

# 创建日志记录器实例，用于记录日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import requests  # 导入 requests 库，用于发送 HTTP 请求
        >>> import torch  # 导入 PyTorch 库
        >>> from PIL import Image  # 从 PIL 导入图像处理类
        >>> from io import BytesIO  # 从 io 导入字节流处理类

        >>> from diffusers import StableDiffusionImg2ImgPipeline  # 导入图像到图像的稳定扩散管道

        >>> device = "cuda"  # 指定使用的设备为 GPU
        >>> model_id_or_path = "runwayml/stable-diffusion-v1-5"  # 指定模型 ID 或路径
        >>> pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)  # 从预训练模型创建管道
        >>> pipe = pipe.to(device)  # 将管道转移到指定设备

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"  # 图像 URL

        >>> response = requests.get(url)  # 发送 GET 请求以获取图像
        >>> init_image = Image.open(BytesIO(response.content)).convert("RGB")  # 打开图像并转换为 RGB 格式
        >>> init_image = init_image.resize((768, 512))  # 调整图像大小

        >>> prompt = "A fantasy landscape, trending on artstation"  # 设置生成图像的提示文本

        >>> images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images  # 生成图像
        >>> images[0].save("fantasy_landscape.png")  # 保存生成的图像
        ```py
"""

# 定义一个函数以检索潜在变量
def retrieve_latents(
    encoder_output: torch.Tensor,  # 输入的编码器输出张量
    generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
    sample_mode: str = "sample"  # 采样模式，默认为“sample”
):
    # 检查 encoder_output 是否有 "latent_dist" 属性，并且采样模式为 "sample"
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            # 从 latent_dist 中采样并返回结果
            return encoder_output.latent_dist.sample(generator)
        # 检查 encoder_output 是否有 "latent_dist" 属性，并且采样模式为 "argmax"
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            # 返回 latent_dist 的众数作为结果
            return encoder_output.latent_dist.mode()
        # 检查 encoder_output 是否有 "latents" 属性
        elif hasattr(encoder_output, "latents"):
            # 直接返回 latents 属性的值
            return encoder_output.latents
        # 如果没有找到任何相关属性，抛出属性错误
        else:
            raise AttributeError("Could not access latents of provided encoder_output")
# 定义预处理图像的函数
def preprocess(image):
    # 定义弃用警告信息，说明该方法在未来的版本中将被移除
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 调用弃用函数，传入方法名、版本号、警告信息及标准警告参数
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 检查输入是否为 PyTorch 张量
    if isinstance(image, torch.Tensor):
        # 如果是张量，直接返回
        return image
    # 检查输入是否为 PIL 图像
    elif isinstance(image, PIL.Image.Image):
        # 将单个图像放入列表中
        image = [image]

    # 如果列表中的第一个元素是 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽和高
        w, h = image[0].size
        # 将宽高调整为 8 的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 对每个图像进行调整大小，并转为 numpy 数组，增加维度
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 将所有图像在第一个维度上拼接
        image = np.concatenate(image, axis=0)
        # 将数组转换为浮点型并归一化到 [0, 1] 区间
        image = np.array(image).astype(np.float32) / 255.0
        # 调整维度顺序为 (批量, 通道, 高, 宽)
        image = image.transpose(0, 3, 1, 2)
        # 将值映射到 [-1, 1] 区间
        image = 2.0 * image - 1.0
        # 将 numpy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果列表中的第一个元素是 PyTorch 张量
    elif isinstance(image[0], torch.Tensor):
        # 在第一个维度上拼接所有张量
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image


# 定义从调度器获取时间步的函数
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理自定义时间步。任何关键字参数将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            从中获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步要移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传入 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma，用于覆盖调度器的时间步间隔策略。如果传入 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是调度器的时间步调度，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传入了时间步和 sigma
    if timesteps is not None and sigmas is not None:
        # 如果同时传入，抛出错误
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查 timesteps 是否为 None，确定是否需要使用自定义时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受 timesteps 参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受 timesteps，则抛出 ValueError 异常
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 检查 sigmas 是否为 None，确定是否需要使用自定义 sigmas
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigmas 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受 sigmas，则抛出 ValueError 异常
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，设置自定义 sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果 timesteps 和 sigmas 都为 None
    else:
        # 调用调度器的 set_timesteps 方法，使用推理步骤的数量
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤的数量
    return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionImg2ImgPipeline 的类，继承多个混合类以实现功能
class StableDiffusionImg2ImgPipeline(
    # 继承自 DiffusionPipeline 类
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin 类
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin 类
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin 类
    IPAdapterMixin,
    # 继承自 StableDiffusionLoraLoaderMixin 类
    StableDiffusionLoraLoaderMixin,
    # 继承自 FromSingleFileMixin 类
    FromSingleFileMixin,
):
    # 文档字符串，描述该管道的功能和参数
    r"""
    Pipeline for text-guided image-to-image generation using Stable Diffusion.

    # 说明该模型继承自 DiffusionPipeline，提供通用方法的文档
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    # 说明该管道还继承了多个加载方法
    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    # 定义该类的参数及其功能
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
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

    # 定义一个字符串，指定 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义一个可选组件列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义一个不参与 CPU 卸载的组件列表
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义一个用于回调的张量输入列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化类的构造函数，接收多个参数
        def __init__(
            self,
            # 变分自编码器模型
            vae: AutoencoderKL,
            # 文本编码器模型
            text_encoder: CLIPTextModel,
            # 词汇表处理器
            tokenizer: CLIPTokenizer,
            # 2D 条件生成模型
            unet: UNet2DConditionModel,
            # Karras 扩散调度器
            scheduler: KarrasDiffusionSchedulers,
            # 稳定扩散安全检查器
            safety_checker: StableDiffusionSafetyChecker,
            # 图像处理器
            feature_extractor: CLIPImageProcessor,
            # 可选的图像编码器模型
            image_encoder: CLIPVisionModelWithProjection = None,
            # 是否需要安全检查器，默认值为 True
            requires_safety_checker: bool = True,
        # 从 StableDiffusionPipeline 的 _encode_prompt 方法复制
        def _encode_prompt(
            self,
            # 输入的提示文本
            prompt,
            # 设备类型
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否使用分类器自由引导
            do_classifier_free_guidance,
            # 可选的负面提示文本
            negative_prompt=None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 LoRA 缩放因子
            lora_scale: Optional[float] = None,
            # 其他关键字参数
            **kwargs,
        ):
            # 过时警告信息，提醒用户方法即将被删除
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用过时警告函数
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法，获取提示嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                # 输入的提示文本
                prompt=prompt,
                # 设备类型
                device=device,
                # 每个提示生成的图像数量
                num_images_per_prompt=num_images_per_prompt,
                # 是否使用分类器自由引导
                do_classifier_free_guidance=do_classifier_free_guidance,
                # 可选的负面提示文本
                negative_prompt=negative_prompt,
                # 可选的提示嵌入
                prompt_embeds=prompt_embeds,
                # 可选的负面提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,
                # 可选的 LoRA 缩放因子
                lora_scale=lora_scale,
                # 其他关键字参数
                **kwargs,
            )
    
            # 将提示嵌入元组中的两个部分连接以便向后兼容
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回连接后的提示嵌入
            return prompt_embeds
    
        # 从 StableDiffusionPipeline 的 encode_prompt 方法复制
        def encode_prompt(
            self,
            # 输入的提示文本
            prompt,
            # 设备类型
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否使用分类器自由引导
            do_classifier_free_guidance,
            # 可选的负面提示文本
            negative_prompt=None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 LoRA 缩放因子
            lora_scale: Optional[float] = None,
            # 可选的跳过的 CLIP 层数
            clip_skip: Optional[int] = None,
        # 从 StableDiffusionPipeline 的 encode_image 方法复制
    # 定义编码图像的函数，接受图像、设备、每个提示的图像数量和可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入的图像是否为张量类型
            if not isinstance(image, torch.Tensor):
                # 使用特征提取器将图像转换为张量，并返回像素值
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备，并转换为指定数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 获取图像编码器的隐藏状态的倒数第二层
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态按提示数量进行重复
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 生成与输入图像形状相同的全零张量，并获取其隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将无条件隐藏状态按提示数量进行重复
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回图像和无条件的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 直接获取图像编码器的图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入按提示数量进行重复
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入形状相同的全零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件嵌入
                return image_embeds, uncond_image_embeds
    
        # 从稳定扩散管道复制的函数，用于准备适配器图像嵌入
        def prepare_ip_adapter_image_embeds(
            # 定义函数参数：适配器图像、适配器图像嵌入、设备、每个提示的图像数量和是否使用分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用无分类器自由引导，初始化一个空列表存储负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器的图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 检查输入适配器的图像是否为列表类型，如果不是，则转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器的图像数量与 IP 适配器数量是否匹配
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 如果不匹配，抛出值错误并给出相关信息
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个输入适配器图像和对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断输出隐藏状态是否为图像投影层的实例
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像，获取其嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到列表中，增加一个维度
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用无分类器自由引导，将负嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果输入适配器图像嵌入不为空，遍历每个嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用无分类器自由引导，拆分负嵌入和图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负嵌入添加到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历图像嵌入及其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入重复 num_images_per_prompt 次，并在维度 0 上连接
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用无分类器自由引导，处理负嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负嵌入和正嵌入连接在一起
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备上
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    # 定义一个方法来运行安全检查器，接受图像、设备和数据类型作为参数
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则没有不安全内容的概念
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入图像是一个张量
            if torch.is_tensor(image):
                # 将图像处理为 PIL 格式以供特征提取
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果图像不是张量，则将其从 NumPy 格式转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征，并将其转移到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，返回处理后的图像和是否存在不安全内容的概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和不安全内容的概念
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
    # 定义一个方法来解码潜在向量
    def decode_latents(self, latents):
        # 定义弃用警告信息
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 发出弃用警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 根据 VAE 配置的缩放因子调整潜在向量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在向量，返回字典中的第一个元素（图像）
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像的像素值归一化到 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 格式以兼容 bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    # 定义一个方法来准备额外的调度器步骤关键字参数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并非所有调度器的签名相同
        # eta（η）仅在 DDIMScheduler 中使用，其他调度器将忽略该参数。
        # eta 对应于 DDIM 论文中的 η，范围应在 [0, 1] 之间

        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外的步骤关键字参数字典
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果接受 eta，则将其添加到字典中
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受 generator，则将其添加到字典中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤关键字参数
        return extra_step_kwargs

    # 定义一个方法来检查输入参数的有效性
    def check_inputs(
        self,
        prompt,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    # 定义获取时间步的方法，参数包括推理步骤数量、强度和设备类型
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步，取num_inference_steps和num_inference_steps * strength的最小值
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算时间步开始的位置，确保不小于0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取时间步，从t_start开始到结束
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # 如果调度器有设置开始索引的方法，则调用它
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        # 返回时间步和剩余推理步骤数量
        return timesteps, num_inference_steps - t_start

    # 从指定的文本到图像管道中复制的方法，用于获取引导比例嵌入
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参考文献链接，获取引导比例嵌入。

        参数:
            w (`torch.Tensor`):
                使用指定的引导比例生成嵌入向量，以丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认为512):
                生成的嵌入的维度。
            dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                生成的嵌入的数据类型。

        返回:
            `torch.Tensor`: 形状为`(len(w), embedding_dim)`的嵌入向量。
        """
        # 确保输入张量的形状是一维的
        assert len(w.shape) == 1
        # 将w乘以1000.0以调整比例
        w = w * 1000.0

        # 计算嵌入的半维度
        half_dim = embedding_dim // 2
        # 计算用于嵌入的缩放因子
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成指数衰减的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 扩展w并计算最终的嵌入
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦嵌入连接在一起
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度为奇数，进行零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保嵌入的形状与预期相符
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    # 定义引导比例属性，返回私有变量
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义跳过剪辑属性，返回私有变量
    @property
    def clip_skip(self):
        return self._clip_skip

    # 定义是否进行无分类器引导的属性，基于引导比例和UNet配置
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 定义交叉注意力参数的属性，返回私有变量
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 定义时间步数属性，返回私有变量
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 定义中断属性，返回私有变量
    @property
    def interrupt(self):
        return self._interrupt

    # 该方法不计算梯度，并替换示例文档字符串
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，用于处理生成图像的请求
        def __call__(
            # 提示文本，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输入的图像，通常用于处理图像生成
            image: PipelineImageInput = None,
            # 生成强度的参数，默认值为0.8
            strength: float = 0.8,
            # 推理步骤的数量，默认值为50
            num_inference_steps: Optional[int] = 50,
            # 采样的时间步，通常用于控制生成过程
            timesteps: List[int] = None,
            # 噪声水平的列表，用于控制生成图像的随机性
            sigmas: List[float] = None,
            # 指导比例，控制生成图像的多样性，默认值为7.5
            guidance_scale: Optional[float] = 7.5,
            # 负面提示文本，可以是单个字符串或字符串列表，用于避免某些特征
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认值为1
            num_images_per_prompt: Optional[int] = 1,
            # 生成过程中使用的超参数，默认值为0.0
            eta: Optional[float] = 0.0,
            # 随机数生成器，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 预先计算的提示嵌入，用于加速生成
            prompt_embeds: Optional[torch.Tensor] = None,
            # 预先计算的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 用于图像适配器的输入图像
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 图像适配器的输入图像嵌入列表
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认值为"pil"，表示返回PIL图像
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认值为True
            return_dict: bool = True,
            # 交叉注意力的额外参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 跳过的clip层数，用于调整模型的特征提取
            clip_skip: int = None,
            # 结束步骤时的回调函数，可以是单个函数或多个回调的组合
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 结束步骤时的张量输入的列表，默认值为["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他额外的参数
            **kwargs,
```