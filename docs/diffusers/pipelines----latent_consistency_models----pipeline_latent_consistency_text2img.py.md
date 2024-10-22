# `.\diffusers\pipelines\latent_consistency_models\pipeline_latent_consistency_text2img.py`

```py
# 版权声明，注明版权归斯坦福大学团队和HuggingFace团队所有
# 
# 根据Apache许可证第2.0版（“许可证”）许可；
# 除非遵守许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，软件以“原样”基础分发，
# 不提供任何明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和限制的具体语言。

# 声明：此代码受到以下项目的强烈影响：https://github.com/pesser/pytorch_diffusion
# 和 https://github.com/hojonathanho/diffusion

import inspect  # 导入inspect模块以获取对象的获取信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示以支持类型注解

import torch  # 导入PyTorch库以进行深度学习操作
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 从transformers库导入相关的CLIP模型

from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入自定义图像处理器
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入不同的加载器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入各种模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入用于调整LoRA缩放的函数
from ...schedulers import LCMScheduler  # 导入调度器类
from ...utils import (  # 从utils模块导入多个工具函数和常量
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor  # 从torch_utils模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from ..stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker  # 导入稳定扩散输出和安全检查器类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，提供使用示例
    Examples:
        ```py
        >>> from diffusers import DiffusionPipeline  # 从diffusers模块导入DiffusionPipeline类
        >>> import torch  # 导入torch库

        >>> pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")  # 从预训练模型创建扩散管道实例
        >>> # 为了节省GPU内存，可以使用torch.float16，但可能会影响图像质量。
        >>> pipe.to(torch_device="cuda", torch_dtype=torch.float32)  # 将管道转移到指定设备并设置数据类型

        >>> prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"  # 定义生成图像的提示

        >>> # 可以设置为1~50步。LCM支持快速推理，即使步数<=4。建议：1~8步。
        >>> num_inference_steps = 4  # 设置推理步骤数
        >>> images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0).images  # 生成图像
        >>> images[0].save("image.png")  # 保存生成的第一张图像
        ```py
"""

# 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion复制的retrieve_timesteps函数
def retrieve_timesteps(  # 定义函数以检索时间步
    scheduler,  # 调度器对象，负责控制时间步
    num_inference_steps: Optional[int] = None,  # 可选的推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备参数，可以是字符串或torch设备对象
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表
    sigmas: Optional[List[float]] = None,  # 可选的sigma值列表
    **kwargs,  # 额外的关键字参数
):
    """
    # 调用调度器的 `set_timesteps` 方法并在调用后从调度器中获取时间步。处理自定义时间步。
    # 任何 kwargs 将被传递给 `scheduler.set_timesteps`。

    # 参数说明:
    # scheduler (`SchedulerMixin`): 从中获取时间步的调度器。
    # num_inference_steps (`int`): 用于生成样本的扩散步骤数。如果使用，则 `timesteps` 必须为 `None`。
    # device (`str` 或 `torch.device`, *可选*): 
    #     时间步应移动到的设备。如果为 `None`，则不移动时间步。
    # timesteps (`List[int]`, *可选*): 
    #     自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
    # sigmas (`List[float]`, *可选*): 
    #     自定义 sigma，用于覆盖调度器的时间步间隔策略。如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    # 返回:
    #     `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是来自调度器的时间步计划，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传入 `timesteps` 和 `sigmas`
    if timesteps is not None and sigmas is not None:
        # 如果同时传入，抛出错误，提示只能选择一个自定义值
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查是否传入了 `timesteps`
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受 `timesteps` 参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误，提示当前调度器不支持自定义时间步
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 检查是否传入了 `sigmas`
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受 `sigmas` 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误，提示当前调度器不支持自定义 sigma
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 如果都没有传入，则使用推理步骤数调用 `set_timesteps`
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤的数量
    return timesteps, num_inference_steps
# 定义一个潜在一致性模型的管道类，继承多个混入类
class LatentConsistencyModelPipeline(
    # 从扩散管道类继承
    DiffusionPipeline,
    # 从稳定扩散混入类继承
    StableDiffusionMixin,
    # 从文本反转加载混入类继承
    TextualInversionLoaderMixin,
    # 从 IP 适配器混入类继承
    IPAdapterMixin,
    # 从稳定扩散 LoRA 加载混入类继承
    StableDiffusionLoraLoaderMixin,
    # 从单文件加载混入类继承
    FromSingleFileMixin,
):
    r"""
    使用潜在一致性模型进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道实现的通用方法的文档，请查看超类文档
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            一个 `CLIPTokenizer` 用于对文本进行标记化。
        unet ([`UNet2DConditionModel`]):
            一个 `UNet2DConditionModel` 用于去噪编码的图像潜在表示。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 一起使用的调度器，用于去噪编码的图像潜在表示。当前仅支持 [`LCMScheduler`]。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，估计生成的图像是否可能被认为是冒犯性或有害的。
            有关模型潜在危害的更多详细信息，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            一个 `CLIPImageProcessor` 用于从生成的图像中提取特征；作为输入用于 `safety_checker`。
        requires_safety_checker (`bool`, *可选*, 默认为 `True`):
            管道是否需要安全检查器组件。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义不参与 CPU 卸载的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调的张量输入
    _callback_tensor_inputs = ["latents", "denoised", "prompt_embeds", "w_embedding"]
    # 初始化方法，用于创建类的实例，接收多个参数以初始化模型组件
    def __init__(
        # VAE（变分自编码器）模型，用于图像生成
        self,
        vae: AutoencoderKL,
        # 文本编码器，用于将文本转换为向量表示
        text_encoder: CLIPTextModel,
        # 分词器，用于将文本拆分为词汇单元
        tokenizer: CLIPTokenizer,
        # UNet模型，用于生成条件图像
        unet: UNet2DConditionModel,
        # 调度器，控制模型训练和生成过程中的学习率等参数
        scheduler: LCMScheduler,
        # 安全检查器，确保生成内容的安全性
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器，处理和提取图像特征
        feature_extractor: CLIPImageProcessor,
        # 可选的图像编码器，用于对图像进行编码
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        # 是否需要安全检查器的标志，默认为 True
        requires_safety_checker: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查是否禁用安全检查器且要求使用安全检查器
        if safety_checker is None and requires_safety_checker:
            # 记录警告信息，提醒用户遵循Stable Diffusion许可证的条件
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
            # 抛出异常，要求用户提供特征提取器
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册模型组件，便于管理和使用
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
        # 计算 VAE 的缩放因子，通常用于图像生成的调整
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化图像处理器，使用上面计算的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 将是否需要安全检查器的配置注册到类的配置中
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从StableDiffusionPipeline复制的方法，用于编码文本提示
    def encode_prompt(
        # 输入的文本提示
        self,
        prompt,
        # 设备类型（如CPU或GPU）
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否执行分类器自由引导
        do_classifier_free_guidance,
        # 可选的负面提示
        negative_prompt=None,
        # 可选的提示嵌入，预先计算的文本嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示嵌入，预先计算的负面文本嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的LoRA缩放因子，用于调节模型的生成效果
        lora_scale: Optional[float] = None,
        # 可选的跳过CLIP模型的某些层
        clip_skip: Optional[int] = None,
    # 从StableDiffusionPipeline复制的方法，用于编码图像
    # 定义一个方法，用于编码图像并返回相应的嵌入
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，则使用特征提取器将其转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像转移到指定设备，并设置数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，则进行隐藏状态的编码
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 根据每个提示重复隐藏状态以匹配数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于无条件输入，编码全零图像并获取隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 同样重复无条件隐藏状态以匹配数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码的图像和无条件图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要输出隐藏状态，则编码图像以获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 根据每个提示重复图像嵌入以匹配数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建一个与图像嵌入形状相同的全零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回编码的图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从稳定扩散管道中复制的方法，用于准备图像嵌入
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    # 定义一个处理图像嵌入的部分
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用分类器自由引导
            if do_classifier_free_guidance:
                # 初始化负图像嵌入列表
                negative_image_embeds = []
            # 如果没有提供图像嵌入
            if ip_adapter_image_embeds is None:
                # 确保输入的图像是列表格式
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
    
                # 检查输入图像数量是否与 IP 适配器数量匹配
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        # 抛出错误提示信息
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
    
                # 遍历每个图像和对应的投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断输出是否为隐藏状态
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码单个图像，获取其嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将单个图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用分类器自由引导，添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历已提供的图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用分类器自由引导，分离负图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化适配器图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入及其索引
            for i, single_image_embeds in enumerate(image_embeds):
                # 将每个图像嵌入复制指定次数
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用分类器自由引导，复制负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 连接负图像嵌入和正图像嵌入
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 添加到适配器图像嵌入列表
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 中复制
    # 定义运行安全检查器的方法，输入为图像、设备和数据类型
    def run_safety_checker(self, image, device, dtype):
        # 检查安全检查器是否存在
        if self.safety_checker is None:
            # 如果不存在，设置无 NSFW 概念为 None
            has_nsfw_concept = None
        else:
            # 检查输入图像是否为 PyTorch 张量
            if torch.is_tensor(image):
                # 将张量图像后处理为 PIL 格式
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 将 NumPy 数组图像转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征并将其转移到指定设备上
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器处理图像，并获取是否存在 NSFW 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 NSFW 概念标志
        return image, has_nsfw_concept
    
    # 从稳定扩散管道准备潜在数据的方法
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在数据的形状
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
    
        # 如果潜在数据为 None，生成新的随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在数据，则将其转移到指定设备上
            latents = latents.to(device)
    
        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在数据
        return latents
    
    # 定义获取引导尺度嵌入的方法，输入为张量、嵌入维度和数据类型
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参考链接：https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        参数：
            w (`torch.Tensor`):
                生成具有指定引导尺度的嵌入向量，以便后续丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认值为 512):
                生成的嵌入的维度。
            dtype (`torch.dtype`, *可选*, 默认值为 `torch.float32`):
                生成的嵌入的数据类型。

        返回：
            `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
        """
        # 确保输入的张量 w 是一维的
        assert len(w.shape) == 1
        # 将 w 的值放大 1000.0，以调整引导尺度
        w = w * 1000.0

        # 计算半维度
        half_dim = embedding_dim // 2
        # 计算嵌入的基础对数缩放因子
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成指数衰减的嵌入值
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将输入的 w 转换为目标 dtype，并进行维度扩展，生成最终的嵌入
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 对嵌入应用正弦和余弦变换
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度为奇数，则在最后一维进行零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保输出的形状为 (w.shape[0], embedding_dim)
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的代码
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并非所有调度器都有相同的签名
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略。
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 并且应在 [0, 1] 范围内

        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 当前的 StableDiffusionPipeline.check_inputs，已移除负提示部分
    def check_inputs(
        self,
        prompt: Union[str, List[str]],  # 提示文本，可以是单个字符串或字符串列表
        height: int,                    # 图像的高度
        width: int,                     # 图像的宽度
        callback_steps: int,            # 回调步骤数
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        ip_adapter_image=None,          # 可选的图像适配器
        ip_adapter_image_embeds=None,   # 可选的图像适配器嵌入
        callback_on_step_end_tensor_inputs=None,  # 可选的回调输入张量
    ):
        # 检查高度和宽度是否能被8整除，如果不能则引发错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤是否为正整数，如果不是则引发错误
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查回调结束时的张量输入是否在允许的输入列表中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查是否同时提供了提示和提示嵌入，如果是则引发错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查提示和提示嵌入是否都未定义，如果是则引发错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示的类型是否为字符串或列表，如果不是则引发错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查适配器图像和嵌入是否同时定义，如果是则引发错误
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        # 检查适配器图像嵌入的类型和维度是否符合要求
        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # 返回指导尺度属性
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 返回交叉注意力关键字参数属性
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 返回跳过剪辑属性
    @property
    def clip_skip(self):
        return self._clip_skip

    # 返回是否执行无分类器引导的属性
    @property
    def do_classifier_free_guidance(self):
        return False

    # 返回时间步数属性
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 在不计算梯度的情况下执行后续装饰器
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，接受多个参数
    def __call__(
        # 输入提示，可以是单个字符串或字符串列表
        self,
        prompt: Union[str, List[str]] = None,
        # 图像高度，默认为 None
        height: Optional[int] = None,
        # 图像宽度，默认为 None
        width: Optional[int] = None,
        # 推理步骤的数量，默认为 4
        num_inference_steps: int = 4,
        # 原始推理步骤的数量，默认为 None
        original_inference_steps: int = None,
        # 指定时间步，默认为 None
        timesteps: List[int] = None,
        # 引导尺度，默认为 8.5
        guidance_scale: float = 8.5,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 随机数生成器，可以是单个或多个生成器，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 预先生成的潜在表示，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 预先计算的提示嵌入，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 图像输入的适配器，默认为 None
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 适配器图像嵌入，默认为 None
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
        # 交叉注意力的关键字参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 跳过的剪辑步骤，默认为 None
        clip_skip: Optional[int] = None,
        # 步骤结束时的回调函数，默认为 None
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 步骤结束时的张量输入回调，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 额外的关键字参数，默认为空
        **kwargs,
```