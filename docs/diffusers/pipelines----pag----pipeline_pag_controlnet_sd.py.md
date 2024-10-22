# `.\diffusers\pipelines\pag\pipeline_pag_controlnet_sd.py`

```py
# 版权声明，标识该文件的版权归 HuggingFace 团队所有
# 按照 Apache 2.0 许可证的条款进行许可
# 除非遵守许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律或书面协议另有规定，软件以 "按现状" 基础分发
# 不提供任何形式的明示或暗示的保证或条件
# 请参阅许可证以了解特定语言下的权限和限制


# 导入 inspect 模块，用于检查对象
import inspect
# 从 typing 模块导入多种类型提示，用于类型注解
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 库，常用于科学计算
import numpy as np
# 导入 PIL.Image，用于处理图像
import PIL.Image
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F
# 从 transformers 库导入 CLIP 模型相关组件
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 导入自定义回调函数类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入图像处理相关类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入多种加载器混合类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入多种模型类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
# 从 LoRA 模块导入调整 LoRA 比例的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 导入 Karras 扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从 utils 模块导入多个实用工具
from ...utils import (
    USE_PEFT_BACKEND,  # 是否使用 PEFT 后端
    logging,  # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 缩放 LoRA 层的函数
    unscale_lora_layers,  # 反缩放 LoRA 层的函数
)
# 从 torch_utils 模块导入多种与 PyTorch 相关的工具函数
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
# 从 MultiControlNet 模块导入多控制网模型类
from ..controlnet.multicontrolnet import MultiControlNetModel
# 从 pipeline_utils 导入扩散管道和稳定扩散混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从 stable_diffusion.pipeline_output 导入稳定扩散管道输出类
from ..stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
# 从 stable_diffusion.safety_checker 导入稳定扩散安全检查器类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 从 pag_utils 导入 PAG 混合类
from .pag_utils import PAGMixin


# 创建一个日志记录器实例，用于记录该模块的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# 示例文档字符串的模板，用于描述示例代码的结构和用途
EXAMPLE_DOC_STRING = """
``` 
```py 
``` 
```py 
``` 
```py 
``` 
    # 示例代码
        Examples:
            ```py
            >>> # !pip install opencv-python transformers accelerate  # 安装必要的库
            >>> from diffusers import AutoPipelineForText2Image, ControlNetModel, UniPCMultistepScheduler  # 导入用于图像生成的库
            >>> from diffusers.utils import load_image  # 导入加载图像的工具
            >>> import numpy as np  # 导入NumPy用于数组操作
            >>> import torch  # 导入PyTorch用于深度学习
    
            >>> import cv2  # 导入OpenCV用于计算机视觉
            >>> from PIL import Image  # 导入PIL用于图像处理
    
            >>> # 下载一张图片
            >>> image = load_image(  # 使用指定的URL下载图像
            ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"  # 图像的URL
            ... )
            >>> image = np.array(image)  # 将下载的图像转换为NumPy数组
    
            >>> # 获取Canny边缘图像
            >>> image = cv2.Canny(image, 100, 200)  # 使用Canny算子检测边缘
            >>> image = image[:, :, None]  # 添加一个新的维度以适配后续操作
            >>> image = np.concatenate([image, image, image], axis=2)  # 将边缘图像复制到三个通道，转换为RGB格式
            >>> canny_image = Image.fromarray(image)  # 将NumPy数组转换回PIL图像
    
            >>> # 加载控制网和稳定扩散模型v1-5
            >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)  # 加载预训练的控制网络
            >>> pipe = AutoPipelineForText2Image.from_pretrained(  # 创建图像生成管道
            ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, enable_pag=True  # 加载稳定扩散模型并启用分页
            ... )
    
            >>> # 通过更快的调度器和内存优化加速扩散过程
            >>> # 如果未安装xformers，可以删除以下行
            >>> pipe.enable_xformers_memory_efficient_attention()  # 启用xformers以提高内存效率
    
            >>> pipe.enable_model_cpu_offload()  # 启用模型CPU卸载以节省GPU内存
    
            >>> # 生成图像
            >>> generator = torch.manual_seed(0)  # 设置随机种子以确保生成图像的可重现性
            >>> image = pipe(  # 调用管道生成图像
            ...     "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",  # 输入生成图像的描述
            ...     guidance_scale=7.5,  # 设置引导比例以控制生成图像的质量
            ...     generator=generator,  # 传入随机生成器
            ...     image=canny_image,  # 使用之前处理的Canny图像作为输入
            ...     pag_scale=10,  # 设置分页比例
            ... ).images[0]  # 获取生成的第一张图像
            ```
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 复制而来
def retrieve_timesteps(
    # 调度器对象
    scheduler,
    # 推理步骤的数量（可选）
    num_inference_steps: Optional[int] = None,
    # 设备类型（可选）
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步（可选）
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 值（可选）
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步。处理自定义时间步。任何关键字参数将传递给 `scheduler.set_timesteps`。
    
    参数：
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`，*可选*):
            要将时间步移动到的设备。如果为 `None`，则时间步不会移动。
        timesteps (`List[int]`，*可选*):
            用于覆盖调度器的时间步间距策略的自定义时间步。如果传递 `timesteps`，`num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`，*可选*):
            用于覆盖调度器的时间步间距策略的自定义 sigma 值。如果传递 `sigmas`，`num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是来自调度器的时间步调度，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了时间步和 sigma
    if timesteps is not None and sigmas is not None:
        # 抛出错误，提示只能传递一个
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查是否传递了时间步
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法，传递自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器中获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数量
        num_inference_steps = len(timesteps)
    # 检查是否传递了 sigma
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法，传递自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器中获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数量
        num_inference_steps = len(timesteps)
    # 如果不满足之前的条件，则执行以下操作
        else:
            # 设置调度器的时间步数，指定推理步骤数量和设备，传入额外参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取当前调度器的时间步列表
            timesteps = scheduler.timesteps
        # 返回时间步列表和推理步骤数量
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionControlNetPAGPipeline 的类，继承自多个基类
class StableDiffusionControlNetPAGPipeline(
    # 继承自 DiffusionPipeline 类
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin 类
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin 类
    TextualInversionLoaderMixin,
    # 继承自 StableDiffusionLoraLoaderMixin 类
    StableDiffusionLoraLoaderMixin,
    # 继承自 IPAdapterMixin 类
    IPAdapterMixin,
    # 继承自 FromSingleFileMixin 类
    FromSingleFileMixin,
    # 继承自 PAGMixin 类
    PAGMixin,
):
    r""" 
    用于文本到图像生成的管道，使用 Stable Diffusion 和 ControlNet 指导。

    此模型继承自 [`DiffusionPipeline`]。请查阅超类文档以获取所有管道的通用方法
    （下载、保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`] ):
            用于对图像进行编码和解码的变分自编码器 (VAE) 模型。
        text_encoder ([`~transformers.CLIPTextModel`] ):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`] ):
            用于文本分词的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`] ):
            用于去噪编码图像潜变量的 `UNet2DConditionModel`。
        controlnet ([`ControlNetModel`] 或 `List[ControlNetModel`] ):
            在去噪过程中为 `unet` 提供额外的条件。如果将多个 ControlNet 设置为列表，则每个 ControlNet 的输出将相加以创建一个组合的额外条件。
        scheduler ([`SchedulerMixin`] ):
            用于与 `unet` 一起去噪编码图像潜变量的调度器。可以是 [`DDIMScheduler`]、[`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`] ):
            分类模块，用于评估生成的图像是否可能被认为是冒犯性或有害的。
            请参考 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 获取有关模型潜在危害的更多信息。
        feature_extractor ([`~transformers.CLIPImageProcessor`] ):
            用于从生成的图像中提取特征的 `CLIPImageProcessor`；用作 `safety_checker` 的输入。
    """

    # 定义 CPU 卸载顺序，确定各组件在 CPU 卸载时的顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件列表，包括安全检查器、特征提取器和图像编码器
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义不参与 CPU 卸载的组件，安全检查器不在其中
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调张量输入，包含潜变量和提示嵌入等
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化方法，设置模型的基本参数和组件
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器，用于图像生成
            text_encoder: CLIPTextModel,  # 文本编码器，将文本转为向量表示
            tokenizer: CLIPTokenizer,  # 分词器，将文本转换为token
            unet: UNet2DConditionModel,  # UNet模型，用于图像处理
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],  # 控制网络，用于条件控制
            scheduler: KarrasDiffusionSchedulers,  # 调度器，控制生成过程的时间步
            safety_checker: StableDiffusionSafetyChecker,  # 安全检查器，确保生成内容符合安全标准
            feature_extractor: CLIPImageProcessor,  # 特征提取器，从图像中提取特征
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器，用于处理图像
            requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
            pag_applied_layers: Union[str, List[str]] = "mid",  # 应用的层的名称，可以是单个或多个层
        ):
            super().__init__()  # 调用父类的初始化方法
    
            # 检查是否禁用安全检查器，并发出警告
            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 如果安全检查器存在但特征提取器为None，抛出错误
            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 如果控制网络是列表或元组，则转换为MultiControlNetModel
            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)
    
            # 注册各个模块，便于管理
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            # 计算VAE缩放因子，基于模型配置
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 初始化图像处理器，设置为RGB转换
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
            # 初始化控制图像处理器，禁用归一化
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            # 注册配置，记录是否需要安全检查器
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
            # 设置应用的层
            self.set_pag_applied_layers(pag_applied_layers)
    
        # 从StableDiffusionPipeline类中复制的编码提示方法
    # 定义编码提示的函数，接受多个参数以处理图像和提示
        def encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 目标设备，例如CPU或GPU
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否执行无分类器引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 提示的嵌入表示，可选
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示的嵌入表示，可选
            lora_scale: Optional[float] = None,  # LoRA缩放因子，可选
            clip_skip: Optional[int] = None,  # 可选的剪切跳过参数
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制的函数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):  # 定义编码图像的函数
            dtype = next(self.image_encoder.parameters()).dtype  # 获取图像编码器参数的数据类型
    
            # 检查输入是否为张量，如果不是则通过特征提取器处理
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并设置数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，则进行以下处理
            if output_hidden_states:
                # 编码图像并获取倒数第二个隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 重复隐藏状态以匹配图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 生成与图像大小相同的零张量，并编码以获取无条件的隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 重复无条件隐藏状态以匹配图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要输出隐藏状态，则直接编码图像
                image_embeds = self.image_encoder(image).image_embeds
                # 重复图像嵌入以匹配图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入大小相同的零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回编码后的图像嵌入和无条件嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的函数
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance  # 准备图像适配器的图像嵌入
    # 开始处理图像嵌入
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用无分类器自由引导，则初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 检查 ip_adapter_image_embeds 是否为 None
            if ip_adapter_image_embeds is None:
                # 确保 ip_adapter_image 是一个列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
    
                # 检查 ip_adapter_image 的长度是否与 IP 适配器数量相同
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        # 抛出值错误，提示图像数量与适配器数量不符
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
    
                # 遍历每个单独的适配器图像及其对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断输出是否为隐藏状态
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码图像，返回单个图像嵌入及其负嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将单个图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用无分类器自由引导，则添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历已存在的图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用无分类器自由引导，则分离负图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 添加单个图像嵌入到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化最终的图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历每个图像嵌入，进行重复操作
            for i, single_image_embeds in enumerate(image_embeds):
                # 将单个图像嵌入重复指定次数
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用无分类器自由引导，则处理负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将最终的图像嵌入添加到列表中
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回所有的图像嵌入
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    # 定义运行安全检查器的方法，接受图像、设备和数据类型作为参数
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未初始化，则设置 nsfw 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 检查输入图像是否为 PyTorch 张量
            if torch.is_tensor(image):
                # 将张量图像后处理为 PIL 格式以供特征提取器使用
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 将 NumPy 数组图像转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理图像并返回 PyTorch 张量，移动到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，获取处理后的图像和 nsfw 概念标志
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 nsfw 概念标志
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的方法，用于准备额外的步骤参数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器具有相同的参数签名
        # eta（η）仅在 DDIMScheduler 中使用，其他调度器将忽略它
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 之间

        # 检查调度器的步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个字典以存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    # 定义检查输入的方法，接受多个参数以验证输入的有效性
    def check_inputs(
        self,
        prompt,
        image,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline 复制的方法，检查图像输入
    # 检查输入图像及其提示是否符合要求
    def check_image(self, image, prompt, prompt_embeds):
        # 判断输入是否为 PIL 图像类型
        image_is_pil = isinstance(image, PIL.Image.Image)
        # 判断输入是否为 PyTorch 张量类型
        image_is_tensor = isinstance(image, torch.Tensor)
        # 判断输入是否为 NumPy 数组类型
        image_is_np = isinstance(image, np.ndarray)
        # 判断输入是否为 PIL 图像的列表
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        # 判断输入是否为 PyTorch 张量的列表
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        # 判断输入是否为 NumPy 数组的列表
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        # 如果输入不属于任何已知的图像类型，则抛出类型错误
        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        # 如果输入为 PIL 图像，则图像批量大小为 1
        if image_is_pil:
            image_batch_size = 1
        else:
            # 否则，图像批量大小为输入列表的长度
            image_batch_size = len(image)

        # 如果提示不为空且为字符串，则提示批量大小为 1
        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        # 如果提示为列表，则提示批量大小为列表的长度
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        # 如果提示嵌入不为空，则提示批量大小为嵌入的第一维长度
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        # 如果图像批量大小不为 1，且与提示批量大小不一致，则抛出值错误
        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image 复制
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        # 预处理图像，并调整为指定的高度和宽度，转换为浮点类型
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        # 获取图像的批量大小
        image_batch_size = image.shape[0]

        # 如果图像批量大小为 1，则根据批量大小重复图像
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # 否则，图像批量大小应与提示批量大小相同
            repeat_by = num_images_per_prompt

        # 根据重复因子重复图像
        image = image.repeat_interleave(repeat_by, dim=0)

        # 将图像移动到指定设备并转换为指定数据类型
        image = image.to(device=device, dtype=dtype)

        # 如果进行无分类器自由引导且不处于猜测模式，则将图像重复连接
        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        # 返回处理后的图像
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在变量，接收批量大小、通道数、图像高度、宽度、数据类型、设备、生成器及潜在变量（可选）
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 根据输入参数计算潜在变量的形状
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,  # 根据 VAE 缩放因子调整高度
            int(width) // self.vae_scale_factor,    # 根据 VAE 缩放因子调整宽度
        )
        # 检查生成器是否为列表且长度与批量大小不匹配，若是则抛出异常
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在变量，则生成随机的潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将提供的潜在变量转移到指定设备
            latents = latents.to(device)

        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 从 latent_consistency_models.pipeline_latent_consistency_text2img 模型中复制的方法，获取引导尺度嵌入
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        查看指定链接中的实现细节

        参数:
            w (`torch.Tensor`):
                生成具有指定引导尺度的嵌入向量，以随后丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认为 512):
                生成嵌入的维度。
            dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                生成的嵌入的数据类型。

        返回:
            `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
        """
        # 确保输入向量 w 为一维
        assert len(w.shape) == 1
        # 将输入向量 w 放大 1000 倍
        w = w * 1000.0

        # 计算嵌入的半维度
        half_dim = embedding_dim // 2
        # 计算嵌入的频率
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成频率嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将 w 转换为指定 dtype 并计算最终的嵌入
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦值连接起来，形成完整的嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度为奇数，则在最后填充一个零
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保嵌入的形状符合预期
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回最终的嵌入
        return emb

    # 返回当前的引导尺度
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 返回当前的剪裁跳过值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 这里 `guidance_scale` 的定义类似于 Imagen 论文中的引导权重 `w`，当 `guidance_scale = 1`
    # 表示没有进行分类器自由引导。
    @property
    # 定义一个方法，用于判断是否使用无分类器自由引导
        def do_classifier_free_guidance(self):
            # 返回判断：如果引导比例大于1且时间条件投影维度为None
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 定义一个属性，返回交叉注意力的关键字参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 定义一个属性，返回时间步数
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 装饰器：无梯度计算
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义一个可调用的方法，接受多个参数
        def __call__(
            # 提示文本，支持字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 输入图像
            image: PipelineImageInput = None,
            # 输出图像的高度
            height: Optional[int] = None,
            # 输出图像的宽度
            width: Optional[int] = None,
            # 推理步骤数
            num_inference_steps: int = 50,
            # 时间步数列表
            timesteps: List[int] = None,
            # sigma值列表
            sigmas: List[float] = None,
            # 引导比例
            guidance_scale: float = 7.5,
            # 负提示文本，支持字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: Optional[int] = 1,
            # eta值
            eta: float = 0.0,
            # 随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在表示
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像嵌入列表
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为"PIL"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式
            return_dict: bool = True,
            # 交叉注意力的关键字参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 控制网条件比例
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # 是否开启猜测模式
            guess_mode: bool = False,
            # 控制引导开始值
            control_guidance_start: Union[float, List[float]] = 0.0,
            # 控制引导结束值
            control_guidance_end: Union[float, List[float]] = 1.0,
            # 跳过剪辑的数量
            clip_skip: Optional[int] = None,
            # 每步结束时的回调
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 每步结束时的张量输入回调
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # pag_scale值
            pag_scale: float = 3.0,
            # pag自适应比例
            pag_adaptive_scale: float = 0.0,
```