# `.\diffusers\pipelines\deepfloyd_if\pipeline_if_inpainting_superresolution.py`

```py
# 导入 html 模块，用于处理 HTML 文本
import html
# 导入 inspect 模块，用于获取对象的信息
import inspect
# 导入 re 模块，用于正则表达式匹配
import re
# 导入 urllib.parse 模块并重命名为 ul，用于处理 URL 编码
import urllib.parse as ul
# 从 typing 模块导入类型提示相关的类
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库并重命名为 np，用于数组和数学计算
import numpy as np
# 导入 PIL.Image 模块，用于图像处理
import PIL.Image
# 导入 torch 库，用于深度学习操作
import torch
# 从 torch.nn.functional 导入 F，用于神经网络的功能操作
import torch.nn.functional as F
# 从 transformers 库导入 CLIPImageProcessor, T5EncoderModel 和 T5Tokenizer，用于处理图像和文本
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

# 从本地模块导入 StableDiffusionLoraLoaderMixin，用于加载稳定扩散模型
from ...loaders import StableDiffusionLoraLoaderMixin
# 从本地模块导入 UNet2DConditionModel，用于2D条件模型
from ...models import UNet2DConditionModel
# 从本地模块导入 DDPMScheduler，用于扩散调度
from ...schedulers import DDPMScheduler
# 从本地模块导入多个实用工具函数
from ...utils import (
    BACKENDS_MAPPING,        # 后端映射
    PIL_INTERPOLATION,      # PIL 插值方式
    is_bs4_available,       # 检查 BeautifulSoup 是否可用
    is_ftfy_available,      # 检查 ftfy 是否可用
    logging,                # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
)
# 从本地工具模块导入 randn_tensor 函数，用于生成随机张量
from ...utils.torch_utils import randn_tensor
# 从本地模块导入 DiffusionPipeline，用于处理扩散管道
from ..pipeline_utils import DiffusionPipeline
# 从本地模块导入 IFPipelineOutput，用于扩散管道的输出
from .pipeline_output import IFPipelineOutput
# 从本地模块导入 IFSafetyChecker，用于安全检查
from .safety_checker import IFSafetyChecker
# 从本地模块导入 IFWatermarker，用于添加水印
from .watermark import IFWatermarker

# 如果 BeautifulSoup 可用，则导入 BeautifulSoup 类
if is_bs4_available():
    from bs4 import BeautifulSoup

# 如果 ftfy 可用，则导入 ftfy 模块
if is_ftfy_available():
    import ftfy

# 创建一个 logger 实例用于记录日志，命名为当前模块名
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 从 diffusers.pipelines.deepfloyd_if.pipeline_if_img2img.resize 复制的函数
def resize(images: PIL.Image.Image, img_size: int) -> PIL.Image.Image:
    # 获取输入图像的宽度和高度
    w, h = images.size

    # 计算宽高比
    coef = w / h

    # 初始化宽高为目标尺寸
    w, h = img_size, img_size

    # 如果宽高比大于等于 1，则按比例调整宽度
    if coef >= 1:
        w = int(round(img_size / 8 * coef) * 8)  # 调整宽度为 8 的倍数
    else:
        # 如果宽高比小于 1，则按比例调整高度
        h = int(round(img_size / 8 / coef) * 8)  # 调整高度为 8 的倍数

    # 调整图像大小，使用双三次插值法
    images = images.resize((w, h), resample=PIL_INTERPOLATION["bicubic"], reducing_gap=None)

    # 返回调整后的图像
    return images

# 示例文档字符串
EXAMPLE_DOC_STRING = """
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
    # 示例用法
    Examples:
        ```py
        # 导入必要的库和模块
        >>> from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
        >>> from diffusers.utils import pt_to_pil
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from io import BytesIO

        # 定义原始图像的 URL
        >>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
        # 发送 GET 请求以获取图像
        >>> response = requests.get(url)
        # 将响应内容转换为 RGB 格式的图像
        >>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
        # 将原始图像赋值给原始图像变量
        >>> original_image = original_image

        # 定义掩膜图像的 URL
        >>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses_mask.png"
        # 发送 GET 请求以获取掩膜图像
        >>> response = requests.get(url)
        # 将响应内容转换为图像
        >>> mask_image = Image.open(BytesIO(response.content))
        # 将掩膜图像赋值给掩膜图像变量
        >>> mask_image = mask_image

        # 从预训练模型加载图像修复管道
        >>> pipe = IFInpaintingPipeline.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
        ... )
        # 启用模型 CPU 卸载功能以节省内存
        >>> pipe.enable_model_cpu_offload()

        # 定义提示文本
        >>> prompt = "blue sunglasses"

        # 编码提示文本为嵌入向量，包括正面和负面提示
        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
        # 使用原始图像、掩膜图像和提示嵌入生成图像
        >>> image = pipe(
        ...     image=original_image,
        ...     mask_image=mask_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ...     output_type="pt",
        ... ).images

        # 保存中间生成的图像
        >>> pil_image = pt_to_pil(image)
        # 将中间图像保存到文件
        >>> pil_image[0].save("./if_stage_I.png")

        # 从预训练模型加载超分辨率管道
        >>> super_res_1_pipe = IFInpaintingSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        ... )
        # 启用模型 CPU 卸载功能以节省内存
        >>> super_res_1_pipe.enable_model_cpu_offload()

        # 使用超分辨率管道生成最终图像
        >>> image = super_res_1_pipe(
        ...     image=image,
        ...     mask_image=mask_image,
        ...     original_image=original_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ... ).images
        # 将最终图像保存到文件
        >>> image[0].save("./if_stage_II.png")
        ```py
    """
# 定义一个名为 IFInpaintingSuperResolutionPipeline 的类，继承自 DiffusionPipeline 和 StableDiffusionLoraLoaderMixin
class IFInpaintingSuperResolutionPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    # 定义一个 tokenizer 属性，类型为 T5Tokenizer
    tokenizer: T5Tokenizer
    # 定义一个 text_encoder 属性，类型为 T5EncoderModel
    text_encoder: T5EncoderModel

    # 定义一个 unet 属性，类型为 UNet2DConditionModel
    unet: UNet2DConditionModel
    # 定义一个调度器 scheduler，类型为 DDPMScheduler
    scheduler: DDPMScheduler
    # 定义一个图像噪声调度器 image_noising_scheduler，类型为 DDPMScheduler
    image_noising_scheduler: DDPMScheduler

    # 可选的特征提取器，类型为 CLIPImageProcessor
    feature_extractor: Optional[CLIPImageProcessor]
    # 可选的安全检查器，类型为 IFSafetyChecker
    safety_checker: Optional[IFSafetyChecker]

    # 可选的水印处理器，类型为 IFWatermarker
    watermarker: Optional[IFWatermarker]

    # 定义一个正则表达式，用于匹配不良标点
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"  # 包含特定特殊字符
        + r"\)"  # 匹配右括号
        + r"\("  # 匹配左括号
        + r"\]"  # 匹配右中括号
        + r"\["  # 匹配左中括号
        + r"\}"  # 匹配右花括号
        + r"\{"  # 匹配左花括号
        + r"\|"  # 匹配竖线
        + "\\"
        + r"\/"  # 匹配斜杠
        + r"\*"  # 匹配星号
        + r"]{1,}"  # 至少匹配一个以上的字符
    )  # noqa

    # 定义一个字符串，用于表示 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet"
    # 定义一个可选组件列表
    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]
    # 定义一个不参与 CPU 卸载的组件列表
    _exclude_from_cpu_offload = ["watermarker"]

    # 初始化方法，接收多个参数以构建类的实例
    def __init__(
        self,
        # tokenizer 参数，类型为 T5Tokenizer
        tokenizer: T5Tokenizer,
        # text_encoder 参数，类型为 T5EncoderModel
        text_encoder: T5EncoderModel,
        # unet 参数，类型为 UNet2DConditionModel
        unet: UNet2DConditionModel,
        # scheduler 参数，类型为 DDPMScheduler
        scheduler: DDPMScheduler,
        # image_noising_scheduler 参数，类型为 DDPMScheduler
        image_noising_scheduler: DDPMScheduler,
        # 可选的安全检查器参数，类型为 IFSafetyChecker
        safety_checker: Optional[IFSafetyChecker],
        # 可选的特征提取器参数，类型为 CLIPImageProcessor
        feature_extractor: Optional[CLIPImageProcessor],
        # 可选的水印处理器参数，类型为 IFWatermarker
        watermarker: Optional[IFWatermarker],
        # 指示是否需要安全检查器的布尔值，默认为 True
        requires_safety_checker: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查安全检查器是否为 None 且要求使用安全检查器
        if safety_checker is None and requires_safety_checker:
            # 记录警告，提示用户已禁用安全检查器
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器是否不为 None 且特征提取器为 None
        if safety_checker is not None and feature_extractor is None:
            # 抛出值错误，提示必须定义特征提取器
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 检查 UNet 配置的输入通道数是否不等于 6
        if unet.config.in_channels != 6:
            # 记录警告，提示用户加载了不适用于超分辨率的检查点
            logger.warning(
                "It seems like you have loaded a checkpoint that shall not be used for super resolution from {unet.config._name_or_path} as it accepts {unet.config.in_channels} input channels instead of 6. Please make sure to pass a super resolution checkpoint as the `'unet'`: IFSuperResolutionPipeline.from_pretrained(unet=super_resolution_unet, ...)`."
            )

        # 注册各个模块以便后续使用
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
        # 将配置注册到实例中，记录是否需要安全检查器
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制
    # 定义一个文本预处理的私有方法，接收文本和是否清理标题的标志
    def _text_preprocessing(self, text, clean_caption=False):
        # 如果设置了清理标题但 bs4 库不可用，发出警告并将 clean_caption 设置为 False
        if clean_caption and not is_bs4_available():
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        # 如果设置了清理标题但 ftfy 库不可用，发出警告并将 clean_caption 设置为 False
        if clean_caption and not is_ftfy_available():
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        # 如果文本不是元组或列表，则将其包装成列表
        if not isinstance(text, (tuple, list)):
            text = [text]

        # 定义一个处理单个文本的内部函数
        def process(text: str):
            # 如果需要清理标题，则调用清理标题的方法
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                # 否则将文本转换为小写并去除首尾空白
                text = text.lower().strip()
            # 返回处理后的文本
            return text

        # 对文本列表中的每个元素应用处理函数，并返回处理后的结果列表
        return [process(t) for t in text]

    # 禁用梯度计算，以节省内存和提高性能
    @torch.no_grad()
    # 定义编码提示的方法，接收多个参数
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
    # 定义运行安全检查的方法，接收图像、设备和数据类型
    def run_safety_checker(self, image, device, dtype):
        # 如果存在安全检查器
        if self.safety_checker is not None:
            # 使用特征提取器处理图像，并将结果转换为指定设备的张量
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            # 运行安全检查器，检测图像中的不当内容和水印
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
            )
        else:
            # 如果没有安全检查器，则将检测结果设置为 None
            nsfw_detected = None
            watermark_detected = None

        # 返回处理后的图像和检测结果
        return image, nsfw_detected, watermark_detected

    # 这里是从其他模块复制的准备额外步骤参数的方法
    # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略它
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 范围内

        # 检查调度器的 step 方法参数中是否接受 eta
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将 eta 添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法参数中是否接受 generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将 generator 添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs

    # 检查输入参数的有效性
    def check_inputs(
        self,
        prompt,
        image,
        original_image,
        mask_image,
        batch_size,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_img2img.IFImg2ImgPipeline.preprocess_image 复制而来，将 preprocess_image 修改为 preprocess_original_image
    def preprocess_original_image(self, image: PIL.Image.Image) -> torch.Tensor:
        # 如果输入的 image 不是列表，则将其转换为列表
        if not isinstance(image, list):
            image = [image]

        # 定义将 NumPy 数组转换为 PyTorch 张量的函数
        def numpy_to_pt(images):
            # 如果输入图像是 3D 数组，则在最后添加一个维度
            if images.ndim == 3:
                images = images[..., None]

            # 将 NumPy 数组转换为 PyTorch 张量并调整维度顺序
            images = torch.from_numpy(images.transpose(0, 3, 1, 2))
            return images

        # 如果第一个图像是 PIL 图像类型
        if isinstance(image[0], PIL.Image.Image):
            new_image = []

            # 遍历每个图像，进行转换和处理
            for image_ in image:
                # 将图像转换为 RGB 格式
                image_ = image_.convert("RGB")
                # 调整图像大小
                image_ = resize(image_, self.unet.config.sample_size)
                # 将图像转换为 NumPy 数组
                image_ = np.array(image_)
                # 转换数据类型为 float32
                image_ = image_.astype(np.float32)
                # 归一化图像数据到 [-1, 1] 范围
                image_ = image_ / 127.5 - 1
                # 将处理后的图像添加到新图像列表中
                new_image.append(image_)

            # 将新图像列表转换为 NumPy 数组
            image = new_image

            # 将图像堆叠成一个 NumPy 数组
            image = np.stack(image, axis=0)  # 转换为 NumPy 数组
            # 将 NumPy 数组转换为 PyTorch 张量
            image = numpy_to_pt(image)  # 转换为 PyTorch 张量

        # 如果第一个图像是 NumPy 数组
        elif isinstance(image[0], np.ndarray):
            # 如果数组是 4 维，则进行拼接，否则堆叠成一个数组
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            # 将 NumPy 数组转换为 PyTorch 张量
            image = numpy_to_pt(image)

        # 如果第一个图像是 PyTorch 张量
        elif isinstance(image[0], torch.Tensor):
            # 如果张量是 4 维，则进行拼接，否则堆叠成一个张量
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

        # 返回处理后的图像张量
        return image

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_superresolution.IFSuperResolutionPipeline.preprocess_image 复制而来
    # 定义图像预处理函数，接受图像、每个提示的图像数量和设备作为参数，返回处理后的张量
    def preprocess_image(self, image: PIL.Image.Image, num_images_per_prompt, device) -> torch.Tensor:
        # 检查输入的图像是否为张量或列表，如果不是则将其转换为列表
        if not isinstance(image, torch.Tensor) and not isinstance(image, list):
            image = [image]

        # 如果列表中的第一个元素是 PIL 图像
        if isinstance(image[0], PIL.Image.Image):
            # 将 PIL 图像转换为 NumPy 数组，并归一化到 [-1, 1] 范围
            image = [np.array(i).astype(np.float32) / 127.5 - 1.0 for i in image]

            # 将列表中的图像堆叠为 NumPy 数组，增加一个维度
            image = np.stack(image, axis=0)  # to np
            # 将 NumPy 数组转换为 PyTorch 张量，并调整维度顺序
            image = torch.from_numpy(image.transpose(0, 3, 1, 2))
        # 如果列表中的第一个元素是 NumPy 数组
        elif isinstance(image[0], np.ndarray):
            # 将列表中的图像堆叠为 NumPy 数组，增加一个维度
            image = np.stack(image, axis=0)  # to np
            # 如果图像是 5 维，取第一个元素
            if image.ndim == 5:
                image = image[0]

            # 将 NumPy 数组转换为 PyTorch 张量，并调整维度顺序
            image = torch.from_numpy(image.transpose(0, 3, 1, 2))
        # 如果输入是列表且第一个元素是张量
        elif isinstance(image, list) and isinstance(image[0], torch.Tensor):
            # 获取第一个张量的维度
            dims = image[0].ndim

            # 如果是 3 维，沿第 0 维堆叠张量
            if dims == 3:
                image = torch.stack(image, dim=0)
            # 如果是 4 维，沿第 0 维连接张量
            elif dims == 4:
                image = torch.concat(image, dim=0)
            # 如果维度不是 3 或 4，抛出错误
            else:
                raise ValueError(f"Image must have 3 or 4 dimensions, instead got {dims}")

        # 将图像张量移动到指定设备，并设置数据类型
        image = image.to(device=device, dtype=self.unet.dtype)

        # 重复图像以匹配每个提示的图像数量
        image = image.repeat_interleave(num_images_per_prompt, dim=0)

        # 返回处理后的图像张量
        return image

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_inpainting.IFInpaintingPipeline 复制的预处理掩码图像的代码
    # 预处理掩码图像，返回处理后的 PyTorch 张量
    def preprocess_mask_image(self, mask_image) -> torch.Tensor:
        # 检查掩码图像是否为列表，如果不是，则将其包装为列表
        if not isinstance(mask_image, list):
            mask_image = [mask_image]
    
        # 如果掩码图像的第一个元素是 PyTorch 张量
        if isinstance(mask_image[0], torch.Tensor):
            # 根据第一个张量的维度，选择合并（cat）或堆叠（stack）操作
            mask_image = torch.cat(mask_image, axis=0) if mask_image[0].ndim == 4 else torch.stack(mask_image, axis=0)
    
            # 如果处理后的张量是二维
            if mask_image.ndim == 2:
                # 对于单个掩码，增加批次维度和通道维度
                mask_image = mask_image.unsqueeze(0).unsqueeze(0)
            # 如果处理后的张量是三维，且第一维大小为1
            elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
                # 单个掩码，认为第0维是批次大小为1
                mask_image = mask_image.unsqueeze(0)
            # 如果处理后的张量是三维，且第一维大小不为1
            elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
                # 一批掩码，认为第0维是批次维度
                mask_image = mask_image.unsqueeze(1)
    
            # 将掩码图像中小于0.5的值设为0
            mask_image[mask_image < 0.5] = 0
            # 将掩码图像中大于等于0.5的值设为1
            mask_image[mask_image >= 0.5] = 1
    
        # 如果掩码图像的第一个元素是 PIL 图像
        elif isinstance(mask_image[0], PIL.Image.Image):
            new_mask_image = []  # 创建一个新的列表以存储处理后的掩码图像
    
            # 遍历每个掩码图像
            for mask_image_ in mask_image:
                # 将掩码图像转换为灰度模式
                mask_image_ = mask_image_.convert("L")
                # 调整掩码图像的大小
                mask_image_ = resize(mask_image_, self.unet.config.sample_size)
                # 将图像转换为 NumPy 数组
                mask_image_ = np.array(mask_image_)
                # 增加批次和通道维度
                mask_image_ = mask_image_[None, None, :]
                # 将处理后的掩码图像添加到新列表中
                new_mask_image.append(mask_image_)
    
            # 将新处理的掩码图像列表合并为一个张量
            mask_image = new_mask_image
            # 沿第0维连接所有掩码图像
            mask_image = np.concatenate(mask_image, axis=0)
            # 将值转换为浮点数并归一化到[0, 1]
            mask_image = mask_image.astype(np.float32) / 255.0
            # 将掩码图像中小于0.5的值设为0
            mask_image[mask_image < 0.5] = 0
            # 将掩码图像中大于等于0.5的值设为1
            mask_image[mask_image >= 0.5] = 1
            # 将 NumPy 数组转换为 PyTorch 张量
            mask_image = torch.from_numpy(mask_image)
    
        # 如果掩码图像的第一个元素是 NumPy 数组
        elif isinstance(mask_image[0], np.ndarray):
            # 沿第0维连接每个掩码数组，并增加批次和通道维度
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)
    
            # 将掩码图像中小于0.5的值设为0
            mask_image[mask_image < 0.5] = 0
            # 将掩码图像中大于等于0.5的值设为1
            mask_image[mask_image >= 0.5] = 1
            # 将 NumPy 数组转换为 PyTorch 张量
            mask_image = torch.from_numpy(mask_image)
    
        # 返回处理后的掩码图像
        return mask_image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制的代码
        def get_timesteps(self, num_inference_steps, strength):
            # 根据初始时间步计算原始时间步
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算开始时间步，确保不小于0
            t_start = max(num_inference_steps - init_timestep, 0)
            # 从调度器中获取时间步
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器有设置开始索引的方法，则调用它
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步和剩余的推理步骤数量
            return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_inpainting.IFInpaintingPipeline.prepare_intermediate_images 复制的代码
    # 准备中间图像，为图像处理生成噪声并应用遮罩
    def prepare_intermediate_images(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, mask_image, generator=None
        ):
            # 获取输入图像的批量大小、通道数、高度和宽度
            image_batch_size, channels, height, width = image.shape
    
            # 更新批量大小为每个提示生成的图像数量
            batch_size = batch_size * num_images_per_prompt
    
            # 定义新的形状用于生成噪声
            shape = (batch_size, channels, height, width)
    
            # 检查生成器列表的长度是否与请求的批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                # 抛出错误以确保生成器数量与批量大小一致
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 生成与输入形状匹配的随机噪声
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
            # 将输入图像按提示的数量重复，以匹配新批量大小
            image = image.repeat_interleave(num_images_per_prompt, dim=0)
            # 向图像添加噪声
            noised_image = self.scheduler.add_noise(image, noise, timestep)
    
            # 应用遮罩以混合原始图像和带噪声图像
            image = (1 - mask_image) * image + mask_image * noised_image
    
            # 返回处理后的图像
            return image
    
        # 禁用梯度计算以节省内存
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            self,
            # 接受图像，支持多种输入类型
            image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
            # 可选参数：原始图像，支持多种输入类型
            original_image: Union[
                PIL.Image.Image, torch.Tensor, np.ndarray, List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]
            ] = None,
            # 可选参数：遮罩图像，支持多种输入类型
            mask_image: Union[
                PIL.Image.Image, torch.Tensor, np.ndarray, List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]
            ] = None,
            # 设置强度参数，默认值为0.8
            strength: float = 0.8,
            # 可选参数：提示信息，支持字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 设置推理步骤的数量，默认值为100
            num_inference_steps: int = 100,
            # 可选参数：时间步列表
            timesteps: List[int] = None,
            # 设置指导尺度，默认值为4.0
            guidance_scale: float = 4.0,
            # 可选参数：负面提示信息
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 可选参数：每个提示生成的图像数量，默认值为1
            num_images_per_prompt: Optional[int] = 1,
            # 设置η值，默认值为0.0
            eta: float = 0.0,
            # 可选参数：随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选参数：提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选参数：负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选参数：输出类型，默认值为"pil"
            output_type: Optional[str] = "pil",
            # 可选参数：是否返回字典，默认值为True
            return_dict: bool = True,
            # 可选参数：回调函数
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 设置回调步骤的数量，默认值为1
            callback_steps: int = 1,
            # 可选参数：交叉注意力关键字参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 设置噪声水平，默认值为0
            noise_level: int = 0,
            # 可选参数：清理标题的标志，默认值为True
            clean_caption: bool = True,
```