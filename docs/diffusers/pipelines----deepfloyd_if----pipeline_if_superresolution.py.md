# `.\diffusers\pipelines\deepfloyd_if\pipeline_if_superresolution.py`

```py
# 导入用于处理 HTML 的模块
import html
# 导入用于获取对象信息的模块
import inspect
# 导入用于正则表达式处理的模块
import re
# 导入 URL 解析模块并命名为 ul
import urllib.parse as ul
# 从 typing 模块导入类型提示相关的类
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 NumPy 库并命名为 np
import numpy as np
# 导入图像处理库 PIL
import PIL.Image
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能性模块
import torch.nn.functional as F
# 从 transformers 库导入图像处理器和 T5 模型及其标记器
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

# 从相对路径导入所需的混合类和模型
from ...loaders import StableDiffusionLoraLoaderMixin
from ...models import UNet2DConditionModel
from ...schedulers import DDPMScheduler
# 从 utils 模块导入多个工具函数和常量
from ...utils import (
    BACKENDS_MAPPING,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
# 从 torch_utils 模块导入生成随机张量的函数
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 导入扩散管道类
from ..pipeline_utils import DiffusionPipeline
# 从 pipeline_output 导入输出类
from .pipeline_output import IFPipelineOutput
# 从安全检查器模块导入安全检查器类
from .safety_checker import IFSafetyChecker
# 从水印模块导入水印处理类
from .watermark import IFWatermarker

# 如果 bs4 可用，则导入 BeautifulSoup 类
if is_bs4_available():
    from bs4 import BeautifulSoup

# 如果 ftfy 可用，则导入该模块
if is_ftfy_available():
    import ftfy

# 创建一个日志记录器以记录模块内的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，提供了使用该类的示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
        >>> from diffusers.utils import pt_to_pil
        >>> import torch

        >>> pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        >>> image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images

        >>> # save intermediate image
        >>> pil_image = pt_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        >>> super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        ... )
        >>> super_res_1_pipe.enable_model_cpu_offload()

        >>> image = super_res_1_pipe(
        ...     image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds
        ... ).images
        >>> image[0].save("./if_stage_II.png")
        ```py
"""

# 定义 IFSuperResolutionPipeline 类，继承自 DiffusionPipeline 和 StableDiffusionLoraLoaderMixin
class IFSuperResolutionPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    # 定义标记器属性，类型为 T5Tokenizer
    tokenizer: T5Tokenizer
    # 定义文本编码器属性，类型为 T5EncoderModel
    text_encoder: T5EncoderModel

    # 定义 UNet 模型属性，类型为 UNet2DConditionModel
    unet: UNet2DConditionModel
    # 定义调度器属性，类型为 DDPMScheduler
    scheduler: DDPMScheduler
    # 定义图像噪声调度器属性，类型为 DDPMScheduler
    image_noising_scheduler: DDPMScheduler

    # 定义可选的特征提取器属性，类型为 CLIPImageProcessor
    feature_extractor: Optional[CLIPImageProcessor]
    # 定义可选的安全检查器属性，类型为 IFSafetyChecker
    safety_checker: Optional[IFSafetyChecker]

    # 定义可选的水印处理器属性，类型为 IFWatermarker
    watermarker: Optional[IFWatermarker]

    # 定义用于匹配不良标点符号的正则表达式
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa
    # 定义可选组件的列表
    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]
    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->unet"
    # 定义不参与 CPU 卸载的组件
    _exclude_from_cpu_offload = ["watermarker"]

    # 初始化方法，定义模型的各个组件
    def __init__(
        self,
        tokenizer: T5Tokenizer,  # 用于文本分词的模型
        text_encoder: T5EncoderModel,  # 文本编码器模型
        unet: UNet2DConditionModel,  # UNet 模型，用于图像生成
        scheduler: DDPMScheduler,  # 调度器，用于控制生成过程
        image_noising_scheduler: DDPMScheduler,  # 图像噪声调度器
        safety_checker: Optional[IFSafetyChecker],  # 安全检查器（可选）
        feature_extractor: Optional[CLIPImageProcessor],  # 特征提取器（可选）
        watermarker: Optional[IFWatermarker],  # 水印器（可选）
        requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查安全检查器是否为 None，且要求使用安全检查器
        if safety_checker is None and requires_safety_checker:
            # 记录警告信息，提醒用户安全检查器未启用
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器存在且特征提取器为 None
        if safety_checker is not None and feature_extractor is None:
            # 抛出值错误，提醒用户需要定义特征提取器
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 检查 UNet 的输入通道数是否为 6
        if unet.config.in_channels != 6:
            # 记录警告，提醒用户所加载的检查点不适合超分辨率
            logger.warning(
                "It seems like you have loaded a checkpoint that shall not be used for super resolution from {unet.config._name_or_path} as it accepts {unet.config.in_channels} input channels instead of 6. Please make sure to pass a super resolution checkpoint as the `'unet'`: IFSuperResolutionPipeline.from_pretrained(unet=super_resolution_unet, ...)."
            )

        # 注册各个组件到模型
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            image_noising_scheduler=image_noising_scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        # 将是否需要安全检查器的信息注册到配置中
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制的内容
    # 定义文本预处理函数，接收文本和可选参数 clean_caption
    def _text_preprocessing(self, text, clean_caption=False):
        # 如果 clean_caption 为真且 bs4 库不可用，则记录警告信息
        if clean_caption and not is_bs4_available():
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            # 记录将 clean_caption 设置为 False 的警告信息
            logger.warning("Setting `clean_caption` to False...")
            # 将 clean_caption 设置为 False
            clean_caption = False

        # 如果 clean_caption 为真且 ftfy 库不可用，则记录警告信息
        if clean_caption and not is_ftfy_available():
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            # 记录将 clean_caption 设置为 False 的警告信息
            logger.warning("Setting `clean_caption` to False...")
            # 将 clean_caption 设置为 False
            clean_caption = False

        # 如果文本不是元组或列表，则将其转换为列表
        if not isinstance(text, (tuple, list)):
            text = [text]

        # 定义处理单个文本的内部函数
        def process(text: str):
            # 如果 clean_caption 为真，执行清理操作
            if clean_caption:
                text = self._clean_caption(text)
                # 再次清理文本
                text = self._clean_caption(text)
            else:
                # 将文本转为小写并去除首尾空白
                text = text.lower().strip()
            # 返回处理后的文本
            return text

        # 对文本列表中的每个文本应用处理函数并返回结果列表
        return [process(t) for t in text]

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制而来
    @torch.no_grad()
    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.encode_prompt 复制而来
    def encode_prompt(
        # 定义提示的类型为字符串或字符串列表
        prompt: Union[str, List[str]],
        # 是否使用无分类器自由引导，默认为 True
        do_classifier_free_guidance: bool = True,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: int = 1,
        # 可选参数，指定设备
        device: Optional[torch.device] = None,
        # 可选参数，负面提示，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 可选参数，提示嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选参数，负面提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 是否清理提示的选项，默认为 False
        clean_caption: bool = False,
    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.run_safety_checker 复制而来
    def run_safety_checker(self, image, device, dtype):
        # 如果存在安全检查器
        if self.safety_checker is not None:
            # 使用特征提取器处理图像并转换为张量
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            # 执行安全检查，返回处理后的图像和检测结果
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
            )
        else:
            # 如果没有安全检查器，设置检测结果为 None
            nsfw_detected = None
            watermark_detected = None

        # 返回处理后的图像及其检测结果
        return image, nsfw_detected, watermark_detected

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.prepare_extra_step_kwargs 复制而来
    # 准备额外的参数用于调度器步骤，因为并非所有调度器具有相同的参数签名
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 应该在 [0, 1] 范围内
    
            # 检查调度器的步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 创建额外参数字典
            extra_step_kwargs = {}
            # 如果接受 eta，添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 检查输入参数的有效性
        def check_inputs(
            self,
            prompt,
            image,
            batch_size,
            noise_level,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.prepare_intermediate_images 复制
        def prepare_intermediate_images(self, batch_size, num_channels, height, width, dtype, device, generator):
            # 定义中间图像的形状
            shape = (batch_size, num_channels, height, width)
            # 检查生成器列表的长度是否与请求的批大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"您传入了长度为 {len(generator)} 的生成器列表，但请求的有效批大小为 {batch_size}。"
                    f" 请确保批大小与生成器的长度匹配。"
                )
    
            # 生成随机噪声张量作为初始中间图像
            intermediate_images = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
            # 按调度器所需的标准差缩放初始噪声
            intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
            # 返回生成的中间图像
            return intermediate_images
    # 预处理图像，使其适合于后续模型处理
    def preprocess_image(self, image, num_images_per_prompt, device):
        # 检查输入是否为张量或列表，若不是，则将其转换为列表
        if not isinstance(image, torch.Tensor) and not isinstance(image, list):
            image = [image]
    
        # 如果输入图像是 PIL 图像，则转换为 NumPy 数组并归一化
        if isinstance(image[0], PIL.Image.Image):
            image = [np.array(i).astype(np.float32) / 127.5 - 1.0 for i in image]
    
            # 将图像列表堆叠为 NumPy 数组
            image = np.stack(image, axis=0)  # to np
            # 转换 NumPy 数组为 PyTorch 张量，并调整维度顺序
            image = torch.from_numpy(image.transpose(0, 3, 1, 2))
        # 如果输入图像是 NumPy 数组，则直接堆叠并处理
        elif isinstance(image[0], np.ndarray):
            image = np.stack(image, axis=0)  # to np
            # 如果数组有五个维度，则只取第一个
            if image.ndim == 5:
                image = image[0]
    
            # 转换 NumPy 数组为 PyTorch 张量，并调整维度顺序
            image = torch.from_numpy(image.transpose(0, 3, 1, 2))
        # 如果输入是张量列表，则根据维度进行堆叠或连接
        elif isinstance(image, list) and isinstance(image[0], torch.Tensor):
            dims = image[0].ndim
    
            # 三维张量则堆叠，四维张量则连接
            if dims == 3:
                image = torch.stack(image, dim=0)
            elif dims == 4:
                image = torch.concat(image, dim=0)
            # 若维度不符合要求，抛出错误
            else:
                raise ValueError(f"Image must have 3 or 4 dimensions, instead got {dims}")
    
        # 将图像移动到指定设备，并设置数据类型
        image = image.to(device=device, dtype=self.unet.dtype)
    
        # 根据每个提示生成的图像数量重复图像
        image = image.repeat_interleave(num_images_per_prompt, dim=0)
    
        # 返回处理后的图像
        return image
    
    # 装饰器，用于禁用梯度计算，节省内存和计算
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的类方法
    def __call__(
        # 提示内容，可以是字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 输出图像的高度
        height: int = None,
        # 输出图像的宽度
        width: int = None,
        # 输入的图像，可以是多种类型
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor] = None,
        # 推理步骤的数量
        num_inference_steps: int = 50,
        # 时间步列表
        timesteps: List[int] = None,
        # 指导缩放因子
        guidance_scale: float = 4.0,
        # 负提示，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: Optional[int] = 1,
        # 噪声级别
        eta: float = 0.0,
        # 随机数生成器，可以是单个或列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 提示嵌入张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，默认为 PIL
        output_type: Optional[str] = "pil",
        # 是否返回字典格式
        return_dict: bool = True,
        # 可选的回调函数
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调步骤的数量
        callback_steps: int = 1,
        # 交叉注意力的参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 噪声级别
        noise_level: int = 250,
        # 是否清理标题
        clean_caption: bool = True,
```