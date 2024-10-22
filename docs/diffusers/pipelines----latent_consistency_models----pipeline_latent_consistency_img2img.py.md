# `.\diffusers\pipelines\latent_consistency_models\pipeline_latent_consistency_img2img.py`

```py
# 版权所有 2024 斯坦福大学团队和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件
# 在许可证下分发是按“原样”基础提供的，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 有关特定权限和
# 许可证下的限制，请参见许可证。

# 免责声明：此代码受以下项目强烈影响 https://github.com/pesser/pytorch_diffusion
# 和 https://github.com/hojonathanho/diffusion

import inspect  # 导入用于检查对象及其属性的模块
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注解

import PIL.Image  # 导入用于图像处理的PIL库
import torch  # 导入PyTorch库
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入transformers库中的模型和处理器

from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入自定义图像处理模块
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入自定义加载器
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整Lora文本编码器的函数
from ...schedulers import LCMScheduler  # 导入调度器
from ...utils import (  # 导入工具函数
    USE_PEFT_BACKEND,  # 指示是否使用PEFT后端
    deprecate,  # 标记弃用的函数
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的工具
    scale_lora_layers,  # 缩放Lora层的函数
    unscale_lora_layers,  # 反缩放Lora层的函数
)
from ...utils.torch_utils import randn_tensor  # 导入用于生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道相关类
from ..stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker  # 导入稳定扩散管道输出和安全检查器

logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(  # 定义函数以检索潜在张量
    encoder_output: torch.Tensor,  # 输入：编码器输出的张量
    generator: Optional[torch.Generator] = None,  # 可选：随机数生成器
    sample_mode: str = "sample"  # 可选：采样模式，默认为"sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":  # 检查是否有潜在分布且模式为采样
        return encoder_output.latent_dist.sample(generator)  # 从潜在分布中采样并返回
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":  # 检查是否有潜在分布且模式为最大值
        return encoder_output.latent_dist.mode()  # 返回潜在分布的众数
    elif hasattr(encoder_output, "latents"):  # 检查是否有直接的潜在值
        return encoder_output.latents  # 返回潜在值
    else:  # 如果没有匹配的属性
        raise AttributeError("Could not access latents of provided encoder_output")  # 抛出属性错误

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制的函数
def retrieve_timesteps(  # 定义函数以检索时间步
    scheduler,  # 输入：调度器
    num_inference_steps: Optional[int] = None,  # 可选：推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选：设备类型
    timesteps: Optional[List[int]] = None,  # 可选：时间步列表
    sigmas: Optional[List[float]] = None,  # 可选：sigma值列表
    **kwargs,  # 其他可选参数
):
    """  # 文档字符串：调用调度器的`set_timesteps`方法并从调度器检索时间步
    # 该函数用于设置自定义的时间步，任何额外的关键字参数将传递给 `scheduler.set_timesteps`。

    # 函数参数说明：
    Args:
        scheduler (`SchedulerMixin`):  # 用于获取时间步的调度器
            The scheduler to get timesteps from.
        num_inference_steps (`int`):  # 生成样本时使用的扩散步骤数，使用时 `timesteps` 必须为 `None`
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):  # 指定时间步要移动到的设备，如果为 `None` 则不移动
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):  # 自定义时间步，如果提供则覆盖调度器的时间步间隔策略
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):  # 自定义 sigma 值，如果提供则覆盖调度器的时间步间隔策略
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    # 返回一个包含时间步调度和推理步骤数量的元组
    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    # 检查是否同时提供了 `timesteps` 和 `sigmas`
    if timesteps is not None and sigmas is not None:
        # 如果同时提供，抛出错误
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    
    # 检查是否提供了自定义的时间步
    if timesteps is not None:
        # 检查当前调度器是否支持自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            # 如果不支持，自定义错误信息
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    
    # 检查是否提供了自定义的 sigma 值
    elif sigmas is not None:
        # 检查当前调度器是否支持自定义 sigma 值
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            # 如果不支持，自定义错误信息
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义 sigma 值
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    
    # 如果没有提供自定义的时间步或 sigma 值，使用推理步骤数量设置时间步
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
    
    # 返回时间步和推理步骤数量的元组
    return timesteps, num_inference_steps
# 示例文档字符串，包含用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import AutoPipelineForImage2Image  # 从 diffusers 模块导入图像到图像的自动管道
        >>> import torch  # 导入 PyTorch 库
        >>> import PIL  # 导入 Python Imaging Library (PIL)

        >>> pipe = AutoPipelineForImage2Image.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")  # 从预训练模型加载管道
        >>> # 为了节省 GPU 内存，可以使用 torch.float16，但可能会影响图像质量。
        >>> pipe.to(torch_device="cuda", torch_dtype=torch.float32)  # 将管道移动到 CUDA 设备，并设置数据类型为 float32

        >>> prompt = "High altitude snowy mountains"  # 设置生成图像的提示词
        >>> image = PIL.Image.open("./snowy_mountains.png")  # 打开一张输入图像

        >>> # 可设置为 1~50 步。LCM 支持快速推断，即使步数 <= 4。推荐：1~8 步。
        >>> num_inference_steps = 4  # 设置推断步骤数为 4
        >>> images = pipe(  # 调用管道生成图像
        ...     prompt=prompt, image=image, num_inference_steps=num_inference_steps, guidance_scale=8.0  # 传入提示词、图像、推断步骤数和引导比例
        ... ).images  # 获取生成的图像列表

        >>> images[0].save("image.png")  # 保存生成的第一张图像为 'image.png'
        ```py

"""

# 图像到图像生成的潜在一致性模型管道类
class LatentConsistencyModelImg2ImgPipeline(  
    DiffusionPipeline,  # 继承自扩散管道类
    StableDiffusionMixin,  # 继承自稳定扩散混合类
    TextualInversionLoaderMixin,  # 继承自文本反演加载混合类
    IPAdapterMixin,  # 继承自 IP 适配器混合类
    StableDiffusionLoraLoaderMixin,  # 继承自稳定扩散 LoRA 加载混合类
    FromSingleFileMixin,  # 继承自单文件加载混合类
):
    r"""  
    使用潜在一致性模型进行图像到图像生成的管道。

    此模型继承自 [`DiffusionPipeline`]。查看超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    # 文档字符串，描述各个参数的作用
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器（VAE）模型，用于将图像编码为潜在表示并解码。
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 模型。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # 一个 `CLIPTokenizer` 用于对文本进行标记化。
        unet ([`UNet2DConditionModel`]):
            # 一个 `UNet2DConditionModel` 用于对编码的图像潜在表示进行去噪。
        scheduler ([`SchedulerMixin`]):
            # 调度器，用于与 `unet` 一起去噪编码的图像潜在表示。当前仅支持 [`LCMScheduler`]。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 分类模块，用于评估生成的图像是否可能被视为冒犯或有害。
            # 详情请参考 [模型卡片](https://huggingface.co/runwayml/stable-diffusion-v1-5) 以了解模型的潜在危害。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # 一个 `CLIPImageProcessor` 用于提取生成图像的特征；这些特征用于输入到 `safety_checker`。
        requires_safety_checker (`bool`, *optional*, defaults to `True`):
            # 指示管道是否需要安全检查器组件。

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义不从 CPU 卸载的组件列表
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调张量输入列表
    _callback_tensor_inputs = ["latents", "denoised", "prompt_embeds", "w_embedding"]

    # 构造函数的定义
    def __init__(
        # VAE 模型的参数
        vae: AutoencoderKL,
        # 文本编码器的参数
        text_encoder: CLIPTextModel,
        # 标记器的参数
        tokenizer: CLIPTokenizer,
        # UNet 模型的参数
        unet: UNet2DConditionModel,
        # 调度器的参数
        scheduler: LCMScheduler,
        # 安全检查器的参数
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器的参数
        feature_extractor: CLIPImageProcessor,
        # 可选的图像编码器参数，默认为 None
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        # 指示是否需要安全检查器的布尔参数，默认为 True
        requires_safety_checker: bool = True,
    # 初始化父类
        ):
            super().__init__()
    
            # 注册各个模块，包括 VAE、文本编码器、分词器等
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
    
            # 检查安全检查器是否为 None，并判断是否需要安全检查器
            if safety_checker is None and requires_safety_checker:
                # 记录警告信息，提醒用户禁用安全检查器的后果
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器实例，使用 VAE 缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
        # 从 StableDiffusionPipeline 复制的编码提示方法
        def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
        # 从 StableDiffusionPipeline 复制的编码图像方法
    # 定义一个编码图像的函数，接受图像、设备、每个提示的图像数量及可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入是否为张量，如果不是则使用特征提取器将其转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并转换为相应的数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果要求输出隐藏状态
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 根据每个提示的图像数量重复隐藏状态
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于无条件图像，编码一个零张量并获取其隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 根据每个提示的图像数量重复无条件隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码的图像隐藏状态和无条件图像隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取其嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 根据每个提示的图像数量重复图像嵌入
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回编码的图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 StableDiffusionPipeline 的 prepare_ip_adapter_image_embeds 函数复制而来
        def prepare_ip_adapter_image_embeds(
            # 定义函数接受的参数：适配器图像、图像嵌入、设备、每个提示的图像数量、是否进行无分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化图像嵌入列表
        image_embeds = []
        # 如果启用了分类器自由引导，则初始化负图像嵌入列表
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果输入适配器图像不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像数量是否与 IP 适配器数量一致
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误信息，说明图像数量与适配器数量不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个输入适配器图像和对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 检查输出是否为隐藏状态
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个适配器图像，获取图像嵌入和负图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用了分类器自由引导，则添加负图像嵌入
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历已存在的输入适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用了分类器自由引导，则分离负图像嵌入和图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化输入适配器图像嵌入的列表
        ip_adapter_image_embeds = []
        # 遍历图像嵌入及其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入复制指定次数
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用了分类器自由引导，处理负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回输入适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的代码
    # 定义安全检查器的运行方法，输入为图像、设备和数据类型
    def run_safety_checker(self, image, device, dtype):
        # 检查安全检查器是否存在
        if self.safety_checker is None:
            # 如果不存在，设置不安全内容的概念为 None
            has_nsfw_concept = None
        else:
            # 检查输入的图像是否为张量
            if torch.is_tensor(image):
                # 将图像进行后处理，转换为 PIL 格式
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果不是张量，将 NumPy 数组转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理输入，返回张量格式并移动到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 调用安全检查器，获取处理后的图像和不安全内容概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和不安全内容概念
        return image, has_nsfw_concept
    
    # 从 StableDiffusionImg2ImgPipeline 类中复制的 prepare_latents 方法
    # 从 LatentConsistencyModelPipeline 类中复制的 get_guidance_scale_embedding 方法
    def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            """
            详细信息见 GitHub 上的 VDM 代码链接
    
            参数:
                w (`torch.Tensor`):
                    生成具有指定引导比例的嵌入向量，以丰富时间步嵌入。
                embedding_dim (`int`, *可选*, 默认值为 512):
                    生成的嵌入维度。
                dtype (`torch.dtype`, *可选*, 默认值为 `torch.float32`):
                    生成嵌入的数值类型。
    
            返回:
                `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
            """
            # 确保输入张量是一维的
            assert len(w.shape) == 1
            # 将输入乘以 1000.0 进行缩放
            w = w * 1000.0
    
            # 计算半个嵌入维度
            half_dim = embedding_dim // 2
            # 计算嵌入的底数
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 计算嵌入的指数形式
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 将缩放后的 w 转换为指定 dtype，并生成嵌入
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦嵌入合并
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保嵌入形状正确
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回最终嵌入
            return emb
    
    # 从 StableDiffusionPipeline 类中复制的 prepare_extra_step_kwargs 方法
    # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的参数签名
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta (η) 仅用于 DDIMScheduler，其他调度器将忽略此参数
            # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
            # 并且应在 [0, 1] 之间
    
            # 检查调度器的步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外步骤参数字典
            extra_step_kwargs = {}
            # 如果接受 eta，则将其添加到字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，则将其添加到字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制的函数
        def get_timesteps(self, num_inference_steps, strength, device):
            # 使用 init_timestep 获取原始时间步
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算开始时间步，确保不小于 0
            t_start = max(num_inference_steps - init_timestep, 0)
            # 根据开始时间步获取时间步列表
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器有设置开始索引的方法，则调用它
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步列表和剩余的推理步骤数
            return timesteps, num_inference_steps - t_start
    
        def check_inputs(
            self,
            prompt: Union[str, List[str]],
            strength: float,
            callback_steps: int,
            # 可选的提示嵌入参数，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的图像适配器输入，默认为 None
            ip_adapter_image=None,
            # 可选的图像适配器嵌入，默认为 None
            ip_adapter_image_embeds=None,
            # 可选的回调结束时的张量输入，默认为 None
            callback_on_step_end_tensor_inputs=None,
    ):
        # 检查 strength 的值是否在有效范围 [0.0, 1.0] 内
        if strength < 0 or strength > 1:
            # 如果不在范围内，抛出值错误
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 检查 callback_steps 是否是正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果不是正整数，抛出值错误
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 callback_on_step_end_tensor_inputs 中的键是否在自定义的回调输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有不在列表中的键，抛出值错误
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查 prompt 和 prompt_embeds 是否同时存在
        if prompt is not None and prompt_embeds is not None:
            # 如果同时存在，抛出值错误
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否同时为 None
        elif prompt is None and prompt_embeds is None:
            # 如果都是 None，抛出值错误
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否正确
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果类型不正确，抛出值错误
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查 ip_adapter_image 和 ip_adapter_image_embeds 是否同时存在
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            # 如果同时存在，抛出值错误
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        # 检查 ip_adapter_image_embeds 的类型是否为列表
        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                # 如果类型不正确，抛出值错误
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            # 检查列表中第一个元素的维度是否为 3D 或 4D
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                # 如果维度不正确，抛出值错误
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # 属性方法，返回指导比例
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 属性方法，返回交叉注意力的参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 属性方法，返回剪切跳过的标志
    @property
    def clip_skip(self):
        return self._clip_skip

    # 属性方法，返回是否执行无分类器的自由引导
    @property
    def do_classifier_free_guidance(self):
        return False

    # 属性方法，返回时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 装饰器，禁用梯度计算
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，允许该对象像函数一样被调用
        def __call__(
            # 提示文本，可以是字符串或字符串列表，默认为 None
            self,
            prompt: Union[str, List[str]] = None,
            # 输入图像，类型为 PipelineImageInput，默认为 None
            image: PipelineImageInput = None,
            # 推理步骤的数量，默认为 4
            num_inference_steps: int = 4,
            # 强度参数，影响图像生成的效果，默认为 0.8
            strength: float = 0.8,
            # 原始推理步骤的数量，默认为 None
            original_inference_steps: int = None,
            # 时间步的列表，默认为 None
            timesteps: List[int] = None,
            # 指导比例，控制生成的多样性，默认为 8.5
            guidance_scale: float = 8.5,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 随机数生成器，可以是单个或列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，类型为 torch.Tensor，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，类型为 torch.Tensor，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 图像适配器输入，类型为 PipelineImageInput，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 图像适配器的嵌入列表，类型为 List[torch.Tensor]，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"，表示 PIL 图像格式
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 跨注意力的关键字参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 跳过的 CLIP 步数，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 步骤结束时的张量输入回调，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他关键字参数
            **kwargs,
```