# `.\diffusers\pipelines\controlnet\pipeline_controlnet.py`

```py
# 版权信息，指明该代码由 HuggingFace 团队版权所有
# 
# 根据 Apache 2.0 许可证授权，用户需遵循许可证规定使用该文件
# 许可证可以在以下网址获取
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面协议，否则软件在"原样"基础上分发，不提供任何形式的担保
# 参见许可证以获取特定的权限和限制

# 导入 inspect 模块，用于检查活跃的对象
import inspect
# 从 typing 导入多种类型注解
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 库，通常用于数值计算
import numpy as np
# 导入 PIL.Image 用于图像处理
import PIL.Image
# 导入 PyTorch 库及其功能模块
import torch
import torch.nn.functional as F
# 从 transformers 导入多个 CLIP 相关模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 导入回调相关的多种类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入图像处理类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入多种加载器混合类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入多种模型类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
# 从 LoRA 模型导入调整文本编码器的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 导入调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从 utils 导入多个实用功能
from ...utils import (
    USE_PEFT_BACKEND,  # 是否使用 PEFT 后端
    deprecate,  # 用于标记过时的功能
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串
    scale_lora_layers,  # 缩放 LoRA 层的函数
    unscale_lora_layers,  # 取消缩放 LoRA 层的函数
)
# 从 torch_utils 导入多种功能
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
# 导入 DiffusionPipeline 和 StableDiffusionMixin 类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入稳定扩散的输出和安全检查器
from ..stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 导入多控制网络模型
from .multicontrolnet import MultiControlNetModel

# 创建日志记录器，使用模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串
EXAMPLE_DOC_STRING = """
``` 
```py  # 结束示例文档字符串
``` 
```py  # 开始另一个代码块
``` 
```py  # 结束另一个代码块
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
``` 
```py  # 结束
``` 
```py  # 继续其他内容
    # 示例代码展示如何使用控制网与稳定扩散生成图像
        Examples:
            ```py
            >>> # !pip install opencv-python transformers accelerate  # 安装所需的库
            >>> from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler  # 导入必要的类
            >>> from diffusers.utils import load_image  # 导入图像加载工具
            >>> import numpy as np  # 导入 NumPy 库
            >>> import torch  # 导入 PyTorch 库
    
            >>> import cv2  # 导入 OpenCV 库
            >>> from PIL import Image  # 导入 PIL 库用于图像处理
    
            >>> # 下载一张图像
            >>> image = load_image(  # 使用 load_image 函数下载图像
            ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"  # 提供图像的 URL
            ... )
            >>> image = np.array(image)  # 将图像转换为 NumPy 数组
    
            >>> # 获取 Canny 边缘图像
            >>> image = cv2.Canny(image, 100, 200)  # 使用 Canny 算法进行边缘检测
            >>> image = image[:, :, None]  # 增加一个维度以适配颜色通道
            >>> image = np.concatenate([image, image, image], axis=2)  # 复制图像到三个通道形成 RGB 图像
            >>> canny_image = Image.fromarray(image)  # 从数组创建 PIL 图像对象
    
            >>> # 加载控制网和稳定扩散模型 v1-5
            >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)  # 加载预训练控制网模型
            >>> pipe = StableDiffusionControlNetPipeline.from_pretrained(  # 加载稳定扩散管道
            ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16  # 指定控制网和数据类型
            ... )
    
            >>> # 加快扩散过程，使用更快的调度器和内存优化
            >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)  # 设置调度器以优化速度
            >>> # 如果未安装 xformers，请移除以下行
            >>> pipe.enable_xformers_memory_efficient_attention()  # 启用 xformers 的内存高效注意力机制
    
            >>> pipe.enable_model_cpu_offload()  # 启用模型 CPU 卸载以节省内存
    
            >>> # 生成图像
            >>> generator = torch.manual_seed(0)  # 设置随机数种子以确保可重复性
            >>> image = pipe(  # 通过管道生成图像
            ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image  # 指定生成的描述、推理步骤和输入图像
            ... ).images[0]  # 获取生成的第一张图像
            ```py  
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的函数
def retrieve_timesteps(
    # 调度器对象
    scheduler,
    # 可选的推理步骤数
    num_inference_steps: Optional[int] = None,
    # 可选的设备参数
    device: Optional[Union[str, torch.device]] = None,
    # 可选的时间步列表
    timesteps: Optional[List[int]] = None,
    # 可选的 sigma 列表
    sigmas: Optional[List[float]] = None,
    # 其他可选参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器检索时间步。处理自定义时间步。
    任何 kwargs 将传递给 `scheduler.set_timesteps`。

    参数：
        scheduler (`SchedulerMixin`):
            要获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步要移动到的设备。如果为 `None`，则时间步不会移动。
        timesteps (`List[int]`, *可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递了 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma，用于覆盖调度器的时间步间隔策略。如果传递了 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是来自调度器的时间步调度，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了时间步和 sigma
    if timesteps is not None and sigmas is not None:
        raise ValueError("只能传递 `timesteps` 或 `sigmas` 之一。请选一个设置自定义值")
    # 如果传递了时间步
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义时间步调度。"
                f"请检查是否使用了正确的调度器。"
            )
        # 设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 如果传递了 sigma
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义 sigma 调度。"
                f"请检查是否使用了正确的调度器。"
            )
        # 设置自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取调度器中的时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 如果不满足条件，则设置调度器的推理步骤
        else:
            # 使用指定的推理步骤和设备设置调度器，传入额外参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器的时间步信息
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤的元组
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionControlNetPipeline 的类，继承多个父类以实现功能
class StableDiffusionControlNetPipeline(
    # 继承 DiffusionPipeline 类，提供扩散管道的基本功能
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 类，提供与 Stable Diffusion 相关的功能
    StableDiffusionMixin,
    # 继承 TextualInversionLoaderMixin 类，提供文本反演加载功能
    TextualInversionLoaderMixin,
    # 继承 StableDiffusionLoraLoaderMixin 类，提供 LoRA 权重的加载和保存功能
    StableDiffusionLoraLoaderMixin,
    # 继承 IPAdapterMixin 类，提供 IP 适配器的加载功能
    IPAdapterMixin,
    # 继承 FromSingleFileMixin 类，提供从单个文件加载的功能
    FromSingleFileMixin,
):
    # 文档字符串，描述此类的功能及其参数
    r"""
    用于文本到图像生成的管道，使用 Stable Diffusion 结合 ControlNet 指导。

    此模型继承自 [`DiffusionPipeline`]。请查看父类文档以了解所有管道的通用方法
    （下载、保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            用于编码和解码图像与潜在表示之间的变分自编码器 (VAE) 模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行分词的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码的图像潜在表示的 `UNet2DConditionModel`。
        controlnet ([`ControlNetModel`] 或 `List[ControlNetModel]`):
            在去噪过程中为 `unet` 提供额外的条件。如果将多个 ControlNet 作为列表设置，来自每个 ControlNet 的输出将被加在一起，形成一个合并的额外条件。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码的图像潜在表示。可以是以下之一：
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            估计生成图像是否可能被视为冒犯或有害的分类模块。
            有关模型潜在危害的更多详细信息，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成的图像中提取特征的 `CLIPImageProcessor`；用于作为 `safety_checker` 的输入。
    """

    # 定义一个字符串，表示模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义一个列表，包含可选组件的名称
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义一个列表，包含不从 CPU 卸载的组件名称
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义一个列表，包含回调张量输入的名称
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化类的构造函数，设置各种参数和模型
        def __init__(
            self,
            # 自动编码器模型
            vae: AutoencoderKL,
            # 文本编码器模型
            text_encoder: CLIPTextModel,
            # 分词器
            tokenizer: CLIPTokenizer,
            # 条件 U-Net 模型
            unet: UNet2DConditionModel,
            # 控制网模型，可以是多种格式
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            # Karras 扩散调度器
            scheduler: KarrasDiffusionSchedulers,
            # 稳定扩散安全检查器
            safety_checker: StableDiffusionSafetyChecker,
            # CLIP 图像处理器
            feature_extractor: CLIPImageProcessor,
            # 可选的图像编码器
            image_encoder: CLIPVisionModelWithProjection = None,
            # 是否需要安全检查器
            requires_safety_checker: bool = True,
        ):
            # 调用父类构造函数
            super().__init__()
    
            # 检查安全检查器是否未定义且需要安全检查
            if safety_checker is None and requires_safety_checker:
                # 记录警告，提醒用户禁用安全检查器的后果
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 检查安全检查器已定义但特征提取器未定义的情况
            if safety_checker is not None and feature_extractor is None:
                # 抛出错误，确保定义特征提取器
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 如果 controlnet 是列表或元组，则转换为 MultiControlNetModel
            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)
    
            # 注册模块，初始化各种模型和组件
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
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建 VAE 图像处理器，转换为 RGB
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
            # 创建控制图像处理器，设置为不归一化
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            # 将需要的安全检查器配置注册到类中
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的代码
    # 定义一个私有方法用于编码提示信息
        def _encode_prompt(
            self,  # 方法的第一个参数，通常为实例自身
            prompt,  # 用户输入的提示信息
            device,  # 设备类型，如 CPU 或 GPU
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 可选的负提示信息
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入，类型为 Torch 张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入，类型为 Torch 张量
            lora_scale: Optional[float] = None,  # 可选的 Lora 比例因子
            **kwargs,  # 其他可选参数
        ):
            # 创建一个废弃提示，告知用户该方法将在未来版本中移除，建议使用新方法
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用废弃处理函数，记录该方法的废弃信息
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法获取编码的提示嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 传入提示信息
                device=device,  # 传入设备类型
                num_images_per_prompt=num_images_per_prompt,  # 传入每个提示生成的图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 传入无分类器引导的选项
                negative_prompt=negative_prompt,  # 传入负提示信息
                prompt_embeds=prompt_embeds,  # 传入可选的提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 传入可选的负提示嵌入
                lora_scale=lora_scale,  # 传入可选的 Lora 比例因子
                **kwargs,  # 传入其他可选参数
            )
    
            # 连接返回的提示嵌入元组的两个部分以兼容旧版本
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回连接后的提示嵌入
            return prompt_embeds
    
        # 定义一个方法用于编码提示信息，来自于 StableDiffusionPipeline
        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
        def encode_prompt(
            self,  # 方法的第一个参数，通常为实例自身
            prompt,  # 用户输入的提示信息
            device,  # 设备类型，如 CPU 或 GPU
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 可选的负提示信息
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入，类型为 Torch 张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入，类型为 Torch 张量
            lora_scale: Optional[float] = None,  # 可选的 Lora 比例因子
            clip_skip: Optional[int] = None,  # 可选的剪裁跳过参数
        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    # 定义一个编码图像的函数，接受图像及其他参数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，则使用特征提取器进行处理
            if not isinstance(image, torch.Tensor):
                # 将图像转换为张量，并获取像素值
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备，并转换为指定数据类型
            image = image.to(device=device, dtype=dtype)
            
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 编码图像并获取倒数第二个隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态按每个提示的图像数量进行重复
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 生成与输入图像大小相同的全零图像张量
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将无条件隐藏状态按每个提示的图像数量进行重复
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回有条件和无条件的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入按每个提示的图像数量进行重复
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入大小相同的全零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回有条件和无条件的图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 中复制的函数，用于准备图像嵌入
        def prepare_ip_adapter_image_embeds(
            # 定义参数：适配器图像、适配器图像嵌入、设备、每个提示的图像数量、是否进行无分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化图像嵌入的列表
        image_embeds = []
        # 如果启用无分类器自由引导，则初始化负图像嵌入的列表
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果 IP 适配器图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果 ip_adapter_image 不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查 ip_adapter_image 的长度是否与 IP 适配器的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 如果不相同，则抛出值错误，提示长度不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个单独的 IP 适配器图像及其对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断输出隐藏状态是否为真，若图像投影层不是 ImageProjection 类
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像，返回嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用无分类器自由引导，则将负图像嵌入添加到负嵌入列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果已提供 IP 适配器图像嵌入，则遍历它们
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用无分类器自由引导，则将嵌入拆分为负和正
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负图像嵌入添加到负嵌入列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds)

        # 初始化最终的 IP 适配器图像嵌入列表
        ip_adapter_image_embeds = []
        # 遍历图像嵌入及其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 通过复制单个图像嵌入以适应每个提示的图像数量
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用无分类器自由引导，处理负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入与正图像嵌入连接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入转移到指定设备上
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到最终列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回最终的 IP 适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的内容
    # 定义运行安全检查器的方法，接受图像、设备和数据类型作为参数
        def run_safety_checker(self, image, device, dtype):
            # 检查安全检查器是否存在
            if self.safety_checker is None:
                # 如果不存在，设置无敏感内容概念为 None
                has_nsfw_concept = None
            else:
                # 检查图像是否为张量
                if torch.is_tensor(image):
                    # 将图像进行后处理，转换为 PIL 格式
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 将 NumPy 数组转换为 PIL 图像
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 提取特征并转换为指定设备的张量
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器，返回处理后的图像和无敏感内容概念
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和无敏感内容概念
            return image, has_nsfw_concept
    
        # 从 StableDiffusionPipeline 复制的解码潜在值的方法
        def decode_latents(self, latents):
            # 提示该方法已弃用，建议使用 VaeImageProcessor.postprocess(...)
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 调用弃用警告函数
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 根据 VAE 配置的缩放因子缩放潜在值
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在值，返回图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 归一化图像，并将其限制在 [0, 1] 范围内
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为 float32 格式，以兼容 bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回最终图像
            return image
    
        # 从 StableDiffusionPipeline 复制的准备额外步骤参数的方法
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，确保不同调度器具有一致的签名
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将被忽略。
            # eta 对应于 DDIM 论文中的 η，值应在 [0, 1] 范围内
    
            # 检查调度器的步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外步骤参数字典
            extra_step_kwargs = {}
            # 如果接受 eta，将其添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，将其添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回额外步骤参数字典
            return extra_step_kwargs
    
        # 定义输入检查的方法，接受多个参数
        def check_inputs(
            self,
            prompt,
            image,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            callback_on_step_end_tensor_inputs=None,
    # 检查输入的图像及其相关的提示是否符合预期格式
        def check_image(self, image, prompt, prompt_embeds):
            # 判断图像是否为 PIL 图像类型
            image_is_pil = isinstance(image, PIL.Image.Image)
            # 判断图像是否为 Torch 张量类型
            image_is_tensor = isinstance(image, torch.Tensor)
            # 判断图像是否为 NumPy 数组类型
            image_is_np = isinstance(image, np.ndarray)
            # 判断图像是否为 PIL 图像列表
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            # 判断图像是否为 Torch 张量列表
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            # 判断图像是否为 NumPy 数组列表
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
            # 如果图像不符合任何预期类型，则抛出类型错误
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
    
            # 如果图像为 PIL 图像，则图像批次大小设为 1
            if image_is_pil:
                image_batch_size = 1
            else:
                # 否则，图像批次大小为图像的长度
                image_batch_size = len(image)
    
            # 检查提示是否为字符串类型
            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            # 检查提示是否为列表类型
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            # 检查提示嵌入是否不为空
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]
    
            # 如果图像批次大小不为 1 且与提示批次大小不匹配，则抛出值错误
            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )
    
        # 准备图像以进行后续处理
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
            # 使用控制图像处理器对图像进行预处理并转换数据类型为 float32
            image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            # 获取图像的批次大小
            image_batch_size = image.shape[0]
    
            # 如果图像批次大小为 1，则重复次数设为批次大小
            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # 如果图像批次大小与提示批次大小相同，则设定重复次数
                repeat_by = num_images_per_prompt
    
            # 通过重复图像来扩展批次
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像移动到指定设备并转换为指定数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果开启分类器自由引导且未启用猜测模式，则重复图像
            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)
    
            # 返回准备好的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 中复制的内容
    # 准备潜在向量的函数，接受批量大小、通道数、高度、宽度等参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在向量的形状，基于输入参数计算每个维度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,  # 高度经过缩放因子处理
            int(width) // self.vae_scale_factor,    # 宽度经过缩放因子处理
        )
        # 检查生成器列表的长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不匹配，抛出值错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果未提供潜在向量，则生成随机潜在向量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在向量，则将其移动到指定设备
            latents = latents.to(device)

        # 按照调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在向量
        return latents

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 导入的函数
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参考链接: https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        参数:
            w (`torch.Tensor`):
                生成具有指定指导比例的嵌入向量，以丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认为 512):
                要生成的嵌入的维度。
            dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                生成的嵌入的数据类型。

        返回:
            `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
        """
        # 确保输入的张量 w 是一维的
        assert len(w.shape) == 1
        # 将输入张量 w 扩大1000倍
        w = w * 1000.0

        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算用于缩放的常量
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 计算指数衰减的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将 w 转换为指定的数据类型并进行矩阵乘法
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 连接正弦和余弦嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度是奇数，进行零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保嵌入的形状符合预期
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    # 获取指导比例的属性
    @property
    def guidance_scale(self):
        # 返回内部存储的指导比例
        return self._guidance_scale

    # 获取剪辑跳过的属性
    @property
    def clip_skip(self):
        # 返回内部存储的剪辑跳过值
        return self._clip_skip

    # 此处的 `guidance_scale` 与 Imagen 论文中方程 (2) 的指导权重 `w` 类似：
    # https://arxiv.org/pdf/2205.11487.pdf 。`guidance_scale = 1` 表示不进行分类器自由指导。
    @property
    # 定义一个方法，判断是否进行分类器自由引导
    def do_classifier_free_guidance(self):
        # 返回指导比例大于 1 且 UNet 配置中时间条件投影维度为空的布尔值
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 定义一个属性，返回交叉注意力的参数
    @property
    def cross_attention_kwargs(self):
        # 返回当前的交叉注意力参数字典
        return self._cross_attention_kwargs

    # 定义一个属性，返回时间步数
    @property
    def num_timesteps(self):
        # 返回当前的时间步数
        return self._num_timesteps

    # 装饰器，禁用梯度计算，减少内存使用
    @torch.no_grad()
    # 装饰器，替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用方法，处理输入参数
    def __call__(
        # 提示字符串或字符串列表，用于生成内容
        prompt: Union[str, List[str]] = None,
        # 输入图像，类型为 PipelineImageInput
        image: PipelineImageInput = None,
        # 指定输出图像的高度
        height: Optional[int] = None,
        # 指定输出图像的宽度
        width: Optional[int] = None,
        # 推理步骤的数量，默认值为 50
        num_inference_steps: int = 50,
        # 时间步的列表
        timesteps: List[int] = None,
        # sigma 值的列表
        sigmas: List[float] = None,
        # 指导比例，默认值为 7.5
        guidance_scale: float = 7.5,
        # 可选的负提示字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认值为 1
        num_images_per_prompt: Optional[int] = 1,
        # 控制噪声的强度，默认值为 0.0
        eta: float = 0.0,
        # 可选的随机数生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选的潜在张量
        latents: Optional[torch.Tensor] = None,
        # 可选的提示嵌入张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的输入适配器图像
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 可选的输入适配器图像嵌入列表
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 可选的输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
        # 可选的交叉注意力参数字典
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 控制网络条件比例，默认为 1.0
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        # 是否启用猜测模式，默认为 False
        guess_mode: bool = False,
        # 控制引导开始时的比例，默认为 0.0
        control_guidance_start: Union[float, List[float]] = 0.0,
        # 控制引导结束时的比例，默认为 1.0
        control_guidance_end: Union[float, List[float]] = 1.0,
        # 可选的跳过剪辑的参数
        clip_skip: Optional[int] = None,
        # 可选的步骤结束回调函数或回调列表
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        # 步骤结束时的张量输入回调参数列表，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 额外的关键字参数
        **kwargs,
```