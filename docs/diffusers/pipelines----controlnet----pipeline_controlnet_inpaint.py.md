# `.\diffusers\pipelines\controlnet\pipeline_controlnet_inpaint.py`

```py
# 版权声明，表明版权归 HuggingFace 团队所有
# 
# 根据 Apache License, Version 2.0 许可协议进行许可；
# 除非符合许可协议，否则不可使用此文件。
# 可以在以下地址获取许可协议：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件按 "原样" 提供，
# 不提供任何形式的保证或条件，明示或暗示。
# 请参阅许可协议以获取特定语言的权限和
# 限制条款。

# 此模型实现受 https://github.com/haofanwang/ControlNet-for-Diffusers/ 启发

import inspect  # 导入 inspect 模块，用于获取对象信息
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示工具

import numpy as np  # 导入 numpy，用于数值计算
import PIL.Image  # 导入 PIL.Image，用于图像处理
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的功能性神经网络模块
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入 Transformers 库中的 CLIP 相关组件

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 从回调模块导入多管道回调类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从图像处理模块导入图像输入和 VAE 图像处理器
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 从加载器模块导入混合类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel  # 从模型模块导入各种模型
from ...models.lora import adjust_lora_scale_text_encoder  # 从 LoRA 模块导入调整文本编码器的函数
from ...schedulers import KarrasDiffusionSchedulers  # 从调度器模块导入 Karras 采样调度器
from ...utils import (  # 从工具模块导入各种实用函数
    USE_PEFT_BACKEND,  # 使用 PEFT 后端的标志
    deprecate,  # 警告使用过时功能的函数
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 缩放 LoRA 层的函数
    unscale_lora_layers,  # 还原 LoRA 层缩放的函数
)
from ...utils.torch_utils import is_compiled_module, randn_tensor  # 从 PyTorch 工具模块导入相关功能
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从管道工具模块导入扩散管道和稳定扩散混合类
from ..stable_diffusion import StableDiffusionPipelineOutput  # 从稳定扩散模块导入管道输出类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 从稳定扩散安全检查模块导入安全检查器
from .multicontrolnet import MultiControlNetModel  # 从多控制网络模块导入多控制网络模型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例，禁用 pylint 检查

EXAMPLE_DOC_STRING = """  # 定义一个多行字符串变量 EXAMPLE_DOC_STRING
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
```py  # 多行字符串结束符
```  # 多行字符串结束符
    # 示例代码片段，展示如何使用相关库生成图像
        Examples:
            ```py
            >>> # 安装所需的库，transformers 和 accelerate
            >>> from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
            # 从 diffusers 库导入相关类，用于图像处理和生成
            >>> from diffusers.utils import load_image
            # 从 diffusers.utils 导入 load_image 函数，用于加载图像
            >>> import numpy as np
            # 导入 numpy 库，用于数组操作
            >>> import torch
            # 导入 PyTorch 库，用于深度学习
    
            >>> init_image = load_image(
            ...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
            ... )
            # 加载初始图像并存储在 init_image 变量中
            >>> init_image = init_image.resize((512, 512))
            # 将初始图像调整为 512x512 的尺寸
    
            >>> generator = torch.Generator(device="cpu").manual_seed(1)
            # 创建一个 CPU 上的随机数生成器，并设置种子为 1
    
            >>> mask_image = load_image(
            ...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
            ... )
            # 加载掩模图像并存储在 mask_image 变量中
            >>> mask_image = mask_image.resize((512, 512))
            # 将掩模图像调整为 512x512 的尺寸
    
            >>> def make_canny_condition(image):
            ...     image = np.array(image)
            # 将输入图像转换为 numpy 数组
            ...     image = cv2.Canny(image, 100, 200)
            # 使用 Canny 算法进行边缘检测
            ...     image = image[:, :, None]
            # 在数组最后添加一个新维度，使其适应后续操作
            ...     image = np.concatenate([image, image, image], axis=2)
            # 将边缘检测结果复制到三个通道，生成三通道图像
            ...     image = Image.fromarray(image)
            # 将 numpy 数组转换回图像格式
            ...     return image
            # 返回处理后的图像
    
            >>> control_image = make_canny_condition(init_image)
            # 对初始图像应用 Canny 边缘检测，生成控制图像
    
            >>> controlnet = ControlNetModel.from_pretrained(
            ...     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            ... )
            # 从预训练模型加载 ControlNetModel，并设置数据类型为 float16
            >>> pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            ... )
            # 创建一个用于图像生成的管道，使用预训练的 Stable Diffusion 模型和 ControlNet
    
            >>> pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            # 将管道的调度器设置为从配置中加载的 DDIMScheduler
            >>> pipe.enable_model_cpu_offload()
            # 启用模型的 CPU 离线加载，以节省内存
    
            >>> # 生成图像
            >>> image = pipe(
            ...     "a handsome man with ray-ban sunglasses",
            ...     num_inference_steps=20,
            ...     generator=generator,
            ...     eta=1.0,
            ...     image=init_image,
            ...     mask_image=mask_image,
            ...     control_image=control_image,
            ... ).images[0]
            # 使用管道生成一幅图像，传入描述、推理步骤、随机生成器、初始图像、掩模和控制图像
            ``` 
"""
# 文档字符串，通常用于描述模块、类或方法的功能
# 这里没有具体内容，可能是留作注释或文档

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    # 定义函数接收一个张量类型的编码器输出和可选的随机数生成器
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 检查 encoder_output 是否具有 latent_dist 属性并且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中采样并返回结果
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否具有 latent_dist 属性并且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的模式值
        return encoder_output.latent_dist.mode()
    # 检查 encoder_output 是否具有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 latents 属性的值
        return encoder_output.latents
    # 如果以上条件都不满足，则抛出异常
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# 定义一个图像修复管道类，使用带有 ControlNet 指导的 Stable Diffusion
class StableDiffusionControlNetInpaintPipeline(
    # 继承自 DiffusionPipeline 和其他多个混合类
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    # 文档字符串，描述该管道的功能
    r"""
    使用 ControlNet 指导的 Stable Diffusion 进行图像修复的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档，以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    <Tip>

    该管道可以与专门为修复微调的检查点一起使用
    （[runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)）以及
    默认的文本到图像 Stable Diffusion 检查点
    （[runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)）。 默认的文本到图像
    Stable Diffusion 检查点可能更适合已经在这些检查点上微调的 ControlNet，例如
    [lllyasviel/control_v11p_sd15_inpaint](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint)。

    </Tip>
    # 参数说明
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器（VAE）模型，用于对图像进行编码和解码，转换为潜在表示。
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # 用于对文本进行分词的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            # 用于对编码的图像潜在空间进行去噪的 `UNet2DConditionModel`。
        controlnet ([`ControlNetModel`] 或 `List[ControlNetModel]`):
            # 在去噪过程中为 `unet` 提供额外的条件。如果设置多个 ControlNet 作为列表，则每个 ControlNet 的输出会相加，创建一个组合的额外条件。
        scheduler ([`SchedulerMixin`]):
            # 用于与 `unet` 结合，去噪编码图像潜在空间的调度器。可以是 [`DDIMScheduler`]、[`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 分类模块，用于估计生成的图像是否可能被认为是冒犯性或有害的。
            # 有关模型潜在危害的更多详细信息，请参考 [模型卡片](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # `CLIPImageProcessor` 用于从生成的图像中提取特征；作为 `safety_checker` 的输入。
    """

    # 定义模型的 CPU 离线加载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 可选组件列表，包含安全检查器、特征提取器和图像编码器
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 从 CPU 离线加载中排除的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 回调张量输入的列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # 构造函数初始化
    def __init__(
        # 变分自编码器实例
        vae: AutoencoderKL,
        # 文本编码器实例
        text_encoder: CLIPTextModel,
        # 分词器实例
        tokenizer: CLIPTokenizer,
        # UNet 模型实例
        unet: UNet2DConditionModel,
        # ControlNet 模型或模型列表
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        # 调度器实例
        scheduler: KarrasDiffusionSchedulers,
        # 安全检查器实例
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器实例
        feature_extractor: CLIPImageProcessor,
        # 可选的图像编码器实例，默认为 None
        image_encoder: CLIPVisionModelWithProjection = None,
        # 是否需要安全检查器，默认为 True
        requires_safety_checker: bool = True,
    # 结束函数定义，初始化父类
    ):
        super().__init__()

        # 检查安全检查器是否为 None，且需要安全检查器的情况下发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                # 日志警告信息，提醒用户禁用安全检查器的后果
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器不为 None 但特征提取器为 None 时引发错误
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                # 报错信息，提示用户需要定义特征提取器
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 如果 controlnet 是列表或元组，则将其转换为 MultiControlNetModel
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        # 注册模块，传入各种组件
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
        # 创建用于图像处理的 VAE 图像处理器
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 创建用于掩码处理的 VAE 图像处理器，设置不同的处理选项
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        # 创建用于控制图像处理的 VAE 图像处理器
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        # 注册配置，记录是否需要安全检查器
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 StableDiffusionPipeline 复制的编码提示的方法
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    # 结束函数参数列表
        ):
            # 定义弃用消息，告知用户该函数将来会被移除
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用弃用警告函数，标记该方法为过时
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 函数，获取提示的嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                # 输入提示文本
                prompt=prompt,
                # 指定设备
                device=device,
                # 每个提示生成的图像数量
                num_images_per_prompt=num_images_per_prompt,
                # 是否使用无分类器自由引导
                do_classifier_free_guidance=do_classifier_free_guidance,
                # 负面提示文本
                negative_prompt=negative_prompt,
                # 提示嵌入，若有
                prompt_embeds=prompt_embeds,
                # 负面提示嵌入，若有
                negative_prompt_embeds=negative_prompt_embeds,
                # Lora 缩放因子，若有
                lora_scale=lora_scale,
                # 其他关键字参数
                **kwargs,
            )
    
            # 连接提示嵌入元组中的元素，以便向后兼容
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回连接后的提示嵌入
            return prompt_embeds
    
        # 从 diffusers 库中复制的 encode_prompt 方法
        def encode_prompt(
            # 输入参数列表
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
        # 从 diffusers 库中复制的 encode_image 方法
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入图像不是张量，则通过特征提取器转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并设置数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 获取图像的隐藏状态，并按每个提示图像的数量重复
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 获取无条件图像的隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回图像和无条件图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 按每个提示图像的数量重复图像嵌入
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的全零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers 库中复制的 prepare_ip_adapter_image_embeds 方法
    # 准备图像嵌入以供 IP 适配器使用
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用无分类器自由引导，则初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果没有提供图像嵌入，则处理给定的 IP 适配器图像
            if ip_adapter_image_embeds is None:
                # 如果给定的图像不是列表，则将其转换为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
                # 检查给定图像数量与 IP 适配器层数量是否匹配
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
                # 遍历每个 IP 适配器图像和对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断是否需要输出隐藏状态
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 对单个图像进行编码，获取图像嵌入和负图像嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
                    # 将单个图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用无分类器自由引导，将负图像嵌入添加到列表中
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历提供的图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用无分类器自由引导，将图像嵌入拆分为负和正图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds)
            # 初始化最终图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历每个图像嵌入
            for i, single_image_embeds in enumerate(image_embeds):
                # 重复图像嵌入以满足每个提示的图像数量
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用无分类器自由引导，重复负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 将负图像嵌入与正图像嵌入合并
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的图像嵌入添加到最终列表中
                ip_adapter_image_embeds.append(single_image_embeds)
            # 返回最终的图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    # 运行安全检查器，检测图像是否包含不适宜内容
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则设置 NSFW 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入是张量格式，进行图像处理，转为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入是 numpy 格式，转为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理图像，并将结果转移到指定设备
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器检查图像，并返回处理后的图像和 NSFW 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像及是否含有 NSFW 概念
        return image, has_nsfw_concept
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
        # 解码潜在变量，生成对应的图像
        def decode_latents(self, latents):
            # 显示解码方法已弃用的警告信息
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 根据 VAE 配置的缩放因子调整潜在变量
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在变量生成图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像数据归一化到 [0, 1] 范围内
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为 float32 格式，以便与 bfloat16 兼容
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回处理后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
        # 准备额外的参数以供调度器步骤使用
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外的参数，因为不同调度器的签名可能不同
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会忽略
            # eta 在 DDIM 论文中对应于 η，应在 [0, 1] 范围内
    
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
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制
    # 定义获取时间步长的方法，接受推理步骤数、强度和设备作为参数
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步，确保不超过总推理步骤
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步，确保不小于零
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中提取相关时间步
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # 如果调度器有设置开始索引的方法，则调用该方法
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        # 返回时间步和剩余的推理步骤数
        return timesteps, num_inference_steps - t_start

    # 定义检查输入参数的方法，接受多个参数
    def check_inputs(
        self,
        prompt,
        image,
        mask_image,
        height,
        width,
        callback_steps,
        output_type,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
        padding_mask_crop=None,
    # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.copy 的检查图像方法
    def check_image(self, image, prompt, prompt_embeds):
        # 检查输入图像是否为 PIL 图像对象
        image_is_pil = isinstance(image, PIL.Image.Image)
        # 检查输入图像是否为 PyTorch 张量
        image_is_tensor = isinstance(image, torch.Tensor)
        # 检查输入图像是否为 NumPy 数组
        image_is_np = isinstance(image, np.ndarray)
        # 检查输入图像是否为 PIL 图像列表
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        # 检查输入图像是否为张量列表
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        # 检查输入图像是否为 NumPy 数组列表
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        # 检查输入图像类型是否合法，若不合法则抛出类型错误
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

        # 如果图像是 PIL 图像，批量大小为 1
        if image_is_pil:
            image_batch_size = 1
        else:
            # 否则，批量大小为图像列表的长度
            image_batch_size = len(image)

        # 检查提示内容的批量大小
        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        # 如果图像批量大小不为 1，且与提示批量大小不一致，则抛出值错误
        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )
    # 准备控制图像的函数
        def prepare_control_image(
            self,
            image,  # 输入图像
            width,  # 目标宽度
            height,  # 目标高度
            batch_size,  # 批处理大小
            num_images_per_prompt,  # 每个提示生成的图像数量
            device,  # 设备类型（CPU或GPU）
            dtype,  # 数据类型
            crops_coords,  # 裁剪坐标
            resize_mode,  # 调整大小的模式
            do_classifier_free_guidance=False,  # 是否使用无分类器引导
            guess_mode=False,  # 是否启用猜测模式
        ):
            # 预处理图像，包括调整大小和裁剪，并转换为浮点32格式
            image = self.control_image_processor.preprocess(
                image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
            ).to(dtype=torch.float32)
            # 获取图像的批处理大小
            image_batch_size = image.shape[0]
    
            if image_batch_size == 1:  # 如果批处理大小为1
                repeat_by = batch_size  # 设置重复次数为批处理大小
            else:
                # 如果图像批处理大小与提示批处理大小相同
                repeat_by = num_images_per_prompt  # 设置重复次数为每个提示生成的图像数量
    
            # 沿着第0维度重复图像
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像移动到指定的设备并设置数据类型
            image = image.to(device=device, dtype=dtype)
    
            if do_classifier_free_guidance and not guess_mode:  # 如果启用无分类器引导且未启用猜测模式
                # 复制图像并将其连接在一起
                image = torch.cat([image] * 2)
    
            # 返回处理后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline.prepare_latents 复制的函数
        def prepare_latents(
            self,
            batch_size,  # 批处理大小
            num_channels_latents,  # 潜在通道数
            height,  # 高度
            width,  # 宽度
            dtype,  # 数据类型
            device,  # 设备类型
            generator,  # 随机数生成器
            latents=None,  # 潜在变量（可选）
            image=None,  # 输入图像（可选）
            timestep=None,  # 时间步（可选）
            is_strength_max=True,  # 是否最大强度
            return_noise=False,  # 是否返回噪声
            return_image_latents=False,  # 是否返回图像潜在变量
    ):
        # 定义输出形状，包括批处理大小、通道数、高度和宽度
        shape = (
            batch_size,  # 批处理大小
            num_channels_latents,  # 潜在变量的通道数
            int(height) // self.vae_scale_factor,  # 高度缩放后的值
            int(width) // self.vae_scale_factor,  # 宽度缩放后的值
        )
        # 检查生成器是否为列表且其长度与批处理大小不匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不匹配，抛出值错误，提示用户
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 检查图像或时间步是否为 None，且最大强度为 False
        if (image is None or timestep is None) and not is_strength_max:
            # 如果是，则抛出值错误，提示必须提供图像或噪声时间步
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        # 检查是否需要返回图像潜在变量，或者潜在变量为 None 且最大强度为 False
        if return_image_latents or (latents is None and not is_strength_max):
            # 将图像转换为指定设备和数据类型
            image = image.to(device=device, dtype=dtype)

            # 检查图像的通道数是否为 4
            if image.shape[1] == 4:
                # 如果是，则将图像潜在变量设为图像本身
                image_latents = image
            else:
                # 否则，通过 VAE 编码图像生成潜在变量
                image_latents = self._encode_vae_image(image=image, generator=generator)
            # 根据批处理大小重复图像潜在变量以匹配批处理大小
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # 如果潜在变量为 None
        if latents is None:
            # 生成随机噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 如果强度为 1，则初始化潜在变量为噪声，否则为图像和噪声的组合
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # 如果是纯噪声，则将初始化的潜在变量乘以调度器的初始 sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            # 如果潜在变量不为 None，则将其转移到设备上
            noise = latents.to(device)
            # 根据调度器的初始 sigma 缩放潜在变量
            latents = noise * self.scheduler.init_noise_sigma

        # 将潜在变量放入输出元组中
        outputs = (latents,)

        # 如果需要返回噪声，将其添加到输出中
        if return_noise:
            outputs += (noise,)

        # 如果需要返回图像潜在变量，将其添加到输出中
        if return_image_latents:
            outputs += (image_latents,)

        # 返回输出元组
        return outputs

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline.prepare_mask_latents 复制
    def prepare_mask_latents(
        # 定义方法的参数，包括掩码、被遮挡图像、批处理大小、高度、宽度、数据类型、设备、生成器和是否进行分类器自由引导
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    # 函数的闭合部分，处理掩膜和图像的形状和数据类型
        ):
            # 将掩膜调整为与潜在空间的形状，以便在连接时不会出错
            # 在转换数据类型之前进行调整，以避免在使用 cpu_offload 和半精度时出现问题
            mask = torch.nn.functional.interpolate(
                # 将掩膜的大小调整为经过 VAE 缩放因子的高度和宽度
                mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
            )
            # 将掩膜移动到指定设备并转换为指定数据类型
            mask = mask.to(device=device, dtype=dtype)
    
            # 将掩膜图像移动到指定设备并转换为指定数据类型
            masked_image = masked_image.to(device=device, dtype=dtype)
    
            # 如果掩膜图像有四个通道，则直接使用掩膜图像作为潜在表示
            if masked_image.shape[1] == 4:
                masked_image_latents = masked_image
            else:
                # 否则，通过 VAE 编码掩膜图像以获得潜在表示
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
    
            # 为每个提示复制掩膜和潜在图像，使用适合 MPS 的方法
            if mask.shape[0] < batch_size:
                # 如果掩膜数量不能整除批处理大小，则引发错误
                if not batch_size % mask.shape[0] == 0:
                    raise ValueError(
                        "传入的掩膜数量与所需批处理大小不匹配。掩膜应复制到"
                        f" 总批处理大小 {batch_size}，但传入了 {mask.shape[0]} 个掩膜。确保传入的掩膜数量能被总请求的批处理大小整除。"
                    )
                # 复制掩膜以匹配批处理大小
                mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
            # 如果潜在图像数量不能整除批处理大小，则引发错误
            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "传入的图像数量与所需批处理大小不匹配。图像应复制到"
                        f" 总批处理大小 {batch_size}，但传入了 {masked_image_latents.shape[0]} 个图像。确保传入的图像数量能被总请求的批处理大小整除。"
                    )
                # 复制潜在图像以匹配批处理大小
                masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)
    
            # 如果启用无分类器引导，则重复掩膜两次
            mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
            # 如果启用无分类器引导，则重复潜在图像两次
            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )
    
            # 对齐设备，以防在与潜在模型输入连接时出现设备错误
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
            # 返回掩膜和潜在图像
            return mask, masked_image_latents
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline._encode_vae_image 复制的内容
    # 定义一个私有方法，用于编码变分自编码器（VAE）图像
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        # 检查生成器是否为列表
        if isinstance(generator, list):
            # 对每个图像批次进行编码并提取潜在表示
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])  # 遍历图像的每一张
            ]
            # 将潜在表示沿第0维（批次维度）进行拼接
            image_latents = torch.cat(image_latents, dim=0)
        else:
            # 如果生成器不是列表，则对整个图像进行编码并提取潜在表示
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        # 根据 VAE 配置的缩放因子调整潜在表示
        image_latents = self.vae.config.scaling_factor * image_latents

        # 返回最终的潜在表示
        return image_latents

    # 定义一个只读属性，返回指导比例
    @property
    def guidance_scale(self):
        return self._guidance_scale  # 返回内部存储的指导比例

    # 定义一个只读属性，返回剪切跳过的参数
    @property
    def clip_skip(self):
        return self._clip_skip  # 返回内部存储的剪切跳过参数

    # 定义一个只读属性，判断是否进行无分类器引导
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1  # 当指导比例大于1时返回True

    # 定义一个只读属性，返回交叉注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs  # 返回内部存储的交叉注意力参数

    # 定义一个只读属性，返回时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps  # 返回内部存储的时间步数

    # 装饰器，关闭梯度计算以提高效率
    @torch.no_grad()
    # 用于替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义调用方法，处理生成图像的输入参数
    def __call__(
        # 定义提示内容，支持字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 定义输入图像
        image: PipelineImageInput = None,
        # 定义掩膜图像
        mask_image: PipelineImageInput = None,
        # 定义控制图像
        control_image: PipelineImageInput = None,
        # 定义图像高度
        height: Optional[int] = None,
        # 定义图像宽度
        width: Optional[int] = None,
        # 定义填充掩膜裁剪参数
        padding_mask_crop: Optional[int] = None,
        # 定义强度参数
        strength: float = 1.0,
        # 定义推理步骤数
        num_inference_steps: int = 50,
        # 定义指导比例
        guidance_scale: float = 7.5,
        # 定义负提示内容，支持字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 定义每个提示生成的图像数量
        num_images_per_prompt: Optional[int] = 1,
        # 定义η参数
        eta: float = 0.0,
        # 定义生成器，支持单个或多个生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 定义潜在表示
        latents: Optional[torch.Tensor] = None,
        # 定义提示嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 定义负提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 定义图像适配器输入
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 定义图像适配器嵌入
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 定义输出类型，默认为 PIL 图像
        output_type: Optional[str] = "pil",
        # 定义是否返回字典格式的输出
        return_dict: bool = True,
        # 定义交叉注意力的关键字参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 定义控制网络条件缩放比例，默认为0.5
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        # 定义是否使用猜测模式
        guess_mode: bool = False,
        # 定义控制引导开始比例，默认为0.0
        control_guidance_start: Union[float, List[float]] = 0.0,
        # 定义控制引导结束比例，默认为1.0
        control_guidance_end: Union[float, List[float]] = 1.0,
        # 定义剪切跳过的参数
        clip_skip: Optional[int] = None,
        # 定义步骤结束时的回调函数
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        # 定义步骤结束时的张量输入回调参数
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 允许接收其他关键字参数
        **kwargs,
```