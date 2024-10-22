# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion.py`

```py
# 版权声明，表明此文件的版权归 HuggingFace 团队所有
# 
# 根据 Apache License 2.0 许可协议进行授权；
# 除非遵循此许可协议，否则不得使用此文件。
# 可以通过以下网址获取许可的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，根据该许可协议分发的软件在“按原样”基础上分发，
# 不提供任何明示或暗示的担保或条件。
# 详细信息请参见许可协议中关于权限和限制的具体条款。
import inspect  # 导入用于检查对象的模块
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注解工具

import torch  # 导入 PyTorch 库
from packaging import version  # 导入版本管理工具
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入特定的转换器模型

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入多管道回调相关类
from ...configuration_utils import FrozenDict  # 导入不可变字典工具
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入不同模型
from ...models.lora import adjust_lora_scale_text_encoder  # 导入用于调整文本编码器的 LoRA 函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 导入多个实用工具
    USE_PEFT_BACKEND,  # 指定使用 PEFT 后端的标志
    deprecate,  # 导入弃用装饰器
    logging,  # 导入日志工具
    replace_example_docstring,  # 导入替换示例文档字符串的工具
    scale_lora_layers,  # 导入缩放 LoRA 层的工具
    unscale_lora_layers,  # 导入反缩放 LoRA 层的工具
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道相关类
from .pipeline_output import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出类
from .safety_checker import StableDiffusionSafetyChecker  # 导入稳定扩散安全检查器类


logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器，禁止 pylint 检查命名

EXAMPLE_DOC_STRING = """  # 示例文档字符串，展示使用示例
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusionPipeline  # 从 diffusers 导入稳定扩散管道

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)  # 从预训练模型加载管道并设置数据类型
        >>> pipe = pipe.to("cuda")  # 将管道转移到 GPU

        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义文本提示
        >>> image = pipe(prompt).images[0]  # 生成图像并提取第一张图像
        ```py
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):  # 定义噪声配置重标定函数
    """
    根据 `guidance_rescale` 对 `noise_cfg` 进行重标定。基于论文[Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)的发现。参见第 3.4 节
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)  # 计算文本噪声的标准差，保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)  # 计算噪声配置的标准差，保持维度
    # 使用文本标准差调整噪声配置，以修正曝光过度的问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)  # 进行重标定
    # 按照指导比例将重标定的噪声与原始噪声混合，避免生成“平面”图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg  # 更新噪声配置
    # 返回噪声配置对象
        return noise_cfg
# 定义一个函数用于检索时间步，接受多个参数
def retrieve_timesteps(
    # 调度器实例，用于获取时间步
    scheduler,
    # 可选的推理步骤数量，默认为 None
    num_inference_steps: Optional[int] = None,
    # 可选的设备参数，可以是字符串或 torch.device，默认为 None
    device: Optional[Union[str, torch.device]] = None,
    # 可选的自定义时间步，默认为 None
    timesteps: Optional[List[int]] = None,
    # 可选的自定义 sigma 值，默认为 None
    sigmas: Optional[List[float]] = None,
    # 其他可选参数，将传递给调度器的 set_timesteps 方法
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器获取时间步。处理自定义时间步。
    所有 kwargs 将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用此参数，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步应移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            用于覆盖调度器时间步间距策略的自定义时间步。如果传入 `timesteps`，
            `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            用于覆盖调度器时间步间距策略的自定义 sigma 值。如果传入 `sigmas`，
            `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是调度器的时间步调度，
        第二个元素是推理步骤的数量。
    """
    # 检查是否同时传入了自定义时间步和 sigma 值，如果是则抛出异常
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传入了自定义时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出异常
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数量
        num_inference_steps = len(timesteps)
    # 如果传入了自定义 sigma 值
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出异常
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法设置自定义 sigma 值
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数量
        num_inference_steps = len(timesteps)
    # 如果没有传入自定义时间步或 sigma
    else:
        # 调用调度器的 set_timesteps 方法，使用推理步骤数量
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
    # 返回时间步长和推理步骤的数量
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionPipeline 的类，继承多个混入类
class StableDiffusionPipeline(
    # 继承 DiffusionPipeline 基础功能
    DiffusionPipeline,
    # 继承稳定扩散特有的功能
    StableDiffusionMixin,
    # 继承文本反演加载功能
    TextualInversionLoaderMixin,
    # 继承 LoRA 加载功能
    StableDiffusionLoraLoaderMixin,
    # 继承 IP 适配器功能
    IPAdapterMixin,
    # 继承从单一文件加载功能
    FromSingleFileMixin,
):
    # 文档字符串，描述该类用于文本到图像生成
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    # 说明此模型继承自 DiffusionPipeline，并指出可以查看超类文档获取通用方法
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    # 说明该管道也继承了多种加载方法
    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    # 参数说明，定义构造函数需要的各类参数及其类型
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

    # 定义一个字符串，表示模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件的列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义不包含在 CPU 卸载中的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调张量输入的列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化方法，构造类的实例并接受多个参数
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器，用于图像生成
            text_encoder: CLIPTextModel,  # 文本编码器，用于将文本转换为嵌入
            tokenizer: CLIPTokenizer,  # 分词器，用于处理文本数据
            unet: UNet2DConditionModel,  # UNet模型，用于条件生成
            scheduler: KarrasDiffusionSchedulers,  # 调度器，控制生成过程中的步伐
            safety_checker: StableDiffusionSafetyChecker,  # 安全检查器，确保生成内容符合安全标准
            feature_extractor: CLIPImageProcessor,  # 特征提取器，用于处理图像
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选图像编码器，用于图像嵌入
            requires_safety_checker: bool = True,  # 是否需要安全检查器，默认为True
        def _encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备信息，指定运行的硬件
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的Lora缩放因子
            **kwargs,  # 其他可选参数
        ):
            # 生成弃用警告信息，提示用户该方法将被移除
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 发出弃用警告，通知版本号和信息
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用新的编码方法，获取提示嵌入的元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 输入提示
                device=device,  # 设备信息
                num_images_per_prompt=num_images_per_prompt,  # 图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 引导选项
                negative_prompt=negative_prompt,  # 负面提示
                prompt_embeds=prompt_embeds,  # 提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 负面提示嵌入
                lora_scale=lora_scale,  # Lora缩放因子
                **kwargs,  # 其他参数
            )
    
            # 将元组中的提示嵌入连接为一个张量，便于后续处理
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回连接后的提示嵌入
            return prompt_embeds
    
        # 新的编码提示方法，接受多个参数以生成提示嵌入
        def encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备信息，指定运行的硬件
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的Lora缩放因子
            clip_skip: Optional[int] = None,  # 可选的跳过参数，用于调节处理流程
    # 定义一个方法用于编码图像，接受图像、设备、每个提示的图像数量及可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，则通过特征提取器转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备，并转换为正确的数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 通过图像编码器处理图像，获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态重复，以适应每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建一个与输入图像大小相同的零张量，通过图像编码器获取未条件化的隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将未条件化的隐藏状态重复，以适应每个提示的图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的图像隐藏状态和未条件化图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 通过图像编码器处理图像，获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入重复，以适应每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建一个与图像嵌入相同形状的零张量作为未条件化的图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和未条件化图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 定义一个方法用于准备 IP 适配器的图像嵌入，接受 IP 适配器图像、图像嵌入、设备、每个提示的图像数量及是否进行分类自由引导
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化图像嵌入列表
        image_embeds = []
        # 如果启用无分类器自由引导，则初始化负图像嵌入列表
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为 None
        if ip_adapter_image_embeds is None:
            # 如果输入适配器图像不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像的长度是否与 IP 适配器数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误，说明输入适配器图像长度不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历输入适配器图像和对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断是否输出隐藏状态
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像，返回图像嵌入和负图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用无分类器自由引导，则将负图像嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历已存在的输入适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用无分类器自由引导，分离负图像嵌入和图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化适配器图像嵌入列表
        ip_adapter_image_embeds = []
        # 遍历图像嵌入列表
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入复制指定次数以适应每个提示的图像数量
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用无分类器自由引导，复制负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入和图像嵌入拼接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 安全检查器的执行方法
    def run_safety_checker(self, image, device, dtype):
        # 如果没有安全检查器，初始化不适合的概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入图像是张量，则进行后处理
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入图像是 numpy 数组，则转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 将处理后的输入图像提取特征并移动到设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，获取处理后的图像和不适合的概念标志
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和不适合的概念标志
        return image, has_nsfw_concept
    # 解码潜在表示
    def decode_latents(self, latents):
        # 定义弃用提示信息
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用函数，提示用户
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 根据缩放因子调整潜在表示
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在表示，返回图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像值缩放到[0, 1]范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 转换图像为 float32 类型以确保兼容性
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回最终图像
        return image
    
    # 准备额外的步骤参数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因不同调度器的参数签名不同
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应 DDIM 论文中的 η，应在 [0, 1] 之间
    
        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果接受，则将 eta 添加到额外参数中
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受，则将 generator 添加到额外参数中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数
        return extra_step_kwargs
    
    # 检查输入参数的有效性
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 方法体未提供，此处无进一步操作
    
    # 准备潜在表示
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 根据批大小和图像尺寸定义潜在表示的形状
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 如果生成器列表长度与批大小不匹配，抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果未提供潜在表示，则随机生成
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将已提供的潜在表示转移到指定设备
            latents = latents.to(device)
    
        # 按调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在表示
        return latents
    
    # 从 latent_consistency_models 获取指导尺度嵌入的方法复制
    # 定义生成指导缩放嵌入的函数，接受张量 w 和其他参数
    def get_guidance_scale_embedding(
        # 输入参数 w，为一维的张量
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参考链接，提供生成嵌入向量的信息
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                用指定的指导缩放生成嵌入向量，以此丰富时间步嵌入。
            embedding_dim (`int`, *optional*, defaults to 512):
                要生成的嵌入的维度。
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                生成的嵌入的数据类型。

        Returns:
            `torch.Tensor`: 嵌入向量，形状为 `(len(w), embedding_dim)`。
        """
        # 确保输入张量 w 是一维的
        assert len(w.shape) == 1
        # 将 w 的值乘以 1000.0
        w = w * 1000.0

        # 计算嵌入的半维度
        half_dim = embedding_dim // 2
        # 计算常量 emb，用于后续的指数计算
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成半维度的指数衰减嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将 w 转换为指定数据类型，并与 emb 进行广播相乘
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦嵌入拼接在一起
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度是奇数，则在最后添加零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保最终嵌入的形状符合预期
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    # 定义属性，返回指导缩放值
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义属性，返回指导重缩放值
    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # 定义属性，返回跨注意力的值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 定义属性，判断是否进行无分类器引导
    # 这里 `guidance_scale` 是类似于 Imagen 论文中方程 (2) 的指导权重 `w`
    # 当 `guidance_scale = 1` 时，相当于不进行分类器无引导
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 定义属性，返回跨注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 定义属性，返回时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 定义属性，返回中断标志
    @property
    def interrupt(self):
        return self._interrupt

    # 指定在此上下文中不计算梯度
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法
        def __call__(
            # 输入提示，可以是字符串或字符串列表，默认为 None
            self,
            prompt: Union[str, List[str]] = None,
            # 输出图像的高度，默认为 None
            height: Optional[int] = None,
            # 输出图像的宽度，默认为 None
            width: Optional[int] = None,
            # 进行推理的步骤数，默认为 50
            num_inference_steps: int = 50,
            # 指定的时间步列表，默认为 None
            timesteps: List[int] = None,
            # 指定的 sigma 值列表，默认为 None
            sigmas: List[float] = None,
            # 指导尺度，默认为 7.5
            guidance_scale: float = 7.5,
            # 负面提示，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 控制生成过程中的随机性，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个或多个 PyTorch 生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量张量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入的张量，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入的张量，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像的嵌入列表，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"（Python Imaging Library）
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 交叉注意力的额外参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指导重缩放因子，默认为 0.0
            guidance_rescale: float = 0.0,
            # 跳过的剪辑层，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，可以是多种类型，默认为 None
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 步骤结束时的张量输入列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 额外的关键字参数
            **kwargs,
```