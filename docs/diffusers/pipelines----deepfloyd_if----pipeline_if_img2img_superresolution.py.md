# `.\diffusers\pipelines\deepfloyd_if\pipeline_if_img2img_superresolution.py`

```py
# 导入标准库中的 html 模块，用于处理 HTML 字符串
import html
# 导入 inspect 模块，用于获取对象的信息
import inspect
# 导入正则表达式模块，用于字符串模式匹配
import re
# 导入 urllib.parse 模块并命名为 ul，用于处理 URL
import urllib.parse as ul
# 从 typing 模块导入各种类型提示
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库，常用于数值计算
import numpy as np
# 导入 PIL.Image 模块，用于图像处理
import PIL.Image
# 导入 torch 库，深度学习框架
import torch
# 从 torch.nn.functional 导入 F，提供常用的神经网络功能
import torch.nn.functional as F
# 从 transformers 导入 CLIPImageProcessor, T5EncoderModel, T5Tokenizer，用于自然语言处理和图像处理
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

# 从 loaders 模块导入 StableDiffusionLoraLoaderMixin 类
from ...loaders import StableDiffusionLoraLoaderMixin
# 从 models 模块导入 UNet2DConditionModel 类
from ...models import UNet2DConditionModel
# 从 schedulers 模块导入 DDPMScheduler 类
from ...schedulers import DDPMScheduler
# 从 utils 模块导入多个工具函数和常量
from ...utils import (
    BACKENDS_MAPPING,  # 后端映射
    PIL_INTERPOLATION,  # PIL 插值方式
    is_bs4_available,  # 检查 BeautifulSoup 是否可用
    is_ftfy_available,  # 检查 ftfy 是否可用
    logging,  # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的工具
)
# 从 torch_utils 模块导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 模块导入 DiffusionPipeline 类
from ..pipeline_utils import DiffusionPipeline
# 从 pipeline_output 模块导入 IFPipelineOutput 类
from .pipeline_output import IFPipelineOutput
# 从 safety_checker 模块导入 IFSafetyChecker 类
from .safety_checker import IFSafetyChecker
# 从 watermark 模块导入 IFWatermarker 类
from .watermark import IFWatermarker

# 如果 bs4 可用，导入 BeautifulSoup 类用于解析 HTML 文档
if is_bs4_available():
    from bs4 import BeautifulSoup

# 如果 ftfy 可用，导入 ftfy 模块用于处理文本
if is_ftfy_available():
    import ftfy

# 创建一个 logger 实例，用于日志记录，禁用无效名称警告
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个函数，进行图像的大小调整
def resize(images: PIL.Image.Image, img_size: int) -> PIL.Image.Image:
    # 获取图像的宽和高
    w, h = images.size

    # 计算图像的宽高比
    coef = w / h

    # 将宽和高都设置为目标尺寸
    w, h = img_size, img_size

    # 根据宽高比调整宽或高，使其为 8 的倍数
    if coef >= 1:
        w = int(round(img_size / 8 * coef) * 8)
    else:
        h = int(round(img_size / 8 / coef) * 8)

    # 调整图像大小，使用双立方插值法
    images = images.resize((w, h), resample=PIL_INTERPOLATION["bicubic"], reducing_gap=None)

    # 返回调整后的图像
    return images

# 示例文档字符串，可能用于函数或类的文档说明
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
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
    Examples:
        ```py
        # 导入所需的库和模块
        >>> from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
        >>> from diffusers.utils import pt_to_pil
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from io import BytesIO

        # 定义图片的 URL
        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        # 发送 GET 请求以获取图像
        >>> response = requests.get(url)
        # 将响应内容转换为 PIL 图像并转换为 RGB 模式
        >>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
        # 调整图像大小为 768x512 像素
        >>> original_image = original_image.resize((768, 512))

        # 从预训练模型加载图像到图像转换管道
        >>> pipe = IFImg2ImgPipeline.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0",
        ...     variant="fp16",  # 使用半精度浮点格式
        ...     torch_dtype=torch.float16,  # 设置 PyTorch 的数据类型
        ... )
        # 启用模型的 CPU 离线处理以节省内存
        >>> pipe.enable_model_cpu_offload()

        # 定义生成图像的提示
        >>> prompt = "A fantasy landscape in style minecraft"
        # 对提示进行编码，获取正向和负向的嵌入向量
        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        # 使用管道生成图像
        >>> image = pipe(
        ...     image=original_image,  # 输入的原始图像
        ...     prompt_embeds=prompt_embeds,  # 正向提示嵌入
        ...     negative_prompt_embeds=negative_embeds,  # 负向提示嵌入
        ...     output_type="pt",  # 输出类型设置为 PyTorch 张量
        ... ).images  # 获取生成的图像列表

        # 将生成的中间图像保存为文件
        >>> pil_image = pt_to_pil(image)  # 将 PyTorch 张量转换为 PIL 图像
        >>> pil_image[0].save("./if_stage_I.png")  # 保存第一张图像

        # 从预训练模型加载超分辨率管道
        >>> super_res_1_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0",
        ...     text_encoder=None,  # 不使用文本编码器
        ...     variant="fp16",  # 使用半精度浮点格式
        ...     torch_dtype=torch.float16,  # 设置 PyTorch 的数据类型
        ... )
        # 启用模型的 CPU 离线处理以节省内存
        >>> super_res_1_pipe.enable_model_cpu_offload()

        # 使用超分辨率管道对生成的图像进行处理
        >>> image = super_res_1_pipe(
        ...     image=image,  # 输入的图像
        ...     original_image=original_image,  # 原始图像
        ...     prompt_embeds=prompt_embeds,  # 正向提示嵌入
        ...     negative_prompt_embeds=negative_embeds,  # 负向提示嵌入
        ... ).images  # 获取处理后的图像列表
        # 保存处理后的第一张图像
        >>> image[0].save("./if_stage_II.png")  # 保存为文件
        ```py
"""
# 文档字符串，通常用于描述类的功能和用途
class IFImg2ImgSuperResolutionPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    # 声明类的属性，包括 tokenizer 和 text_encoder 的类型
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    # 声明 UNet 和调度器的类型
    unet: UNet2DConditionModel
    scheduler: DDPMScheduler
    image_noising_scheduler: DDPMScheduler

    # 可选的特征提取器和安全检查器
    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[IFSafetyChecker]

    # 可选的水印处理器
    watermarker: Optional[IFWatermarker]

    # 定义一个正则表达式，用于匹配不良标点符号
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

    # 可选组件列表，包含 tokenizer、text_encoder、safety_checker 和 feature_extractor
    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor"]
    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet"
    # 排除水印器不参与 CPU 卸载
    _exclude_from_cpu_offload = ["watermarker"]

    # 构造函数，初始化类的各个属性
    def __init__(
        self,
        tokenizer: T5Tokenizer,  # tokenizer 用于文本处理
        text_encoder: T5EncoderModel,  # text_encoder 负责编码文本
        unet: UNet2DConditionModel,  # unet 处理图像生成
        scheduler: DDPMScheduler,  # scheduler 控制生成过程的时间步
        image_noising_scheduler: DDPMScheduler,  # image_noising_scheduler 处理图像噪声
        safety_checker: Optional[IFSafetyChecker],  # 可选的安全检查器
        feature_extractor: Optional[CLIPImageProcessor],  # 可选的特征提取器
        watermarker: Optional[IFWatermarker],  # 可选的水印处理器
        requires_safety_checker: bool = True,  # 是否需要安全检查器的布尔标志
    ):
        # 调用父类的构造函数进行初始化
        super().__init__()

        # 检查安全检查器是否为 None 且要求使用安全检查器
        if safety_checker is None and requires_safety_checker:
            # 记录警告信息，提示用户禁用了安全检查器
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
            # 抛出错误，提示用户需要定义特征提取器
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 检查 UNet 配置的输入通道数是否不等于 6
        if unet.config.in_channels != 6:
            # 记录警告信息，提示用户加载的检查点不适用于超分辨率
            logger.warning(
                "It seems like you have loaded a checkpoint that shall not be used for super resolution from {unet.config._name_or_path} as it accepts {unet.config.in_channels} input channels instead of 6. Please make sure to pass a super resolution checkpoint as the `'unet'`: IFSuperResolutionPipeline.from_pretrained(unet=super_resolution_unet, ...)`."
            )

        # 注册多个模块，包括 tokenizer、text_encoder 等
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
        # 将需要的配置注册到对象中
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制的代码
    # 文本预处理方法，接受文本和清理标题的标志
    def _text_preprocessing(self, text, clean_caption=False):
        # 如果需要清理标题且未安装 BeautifulSoup4，则发出警告
        if clean_caption and not is_bs4_available():
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            # 发出警告，提示将清理标题设置为 False
            logger.warning("Setting `clean_caption` to False...")
            # 设置清理标题为 False
            clean_caption = False

        # 如果需要清理标题且未安装 ftfy，则发出警告
        if clean_caption and not is_ftfy_available():
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            # 发出警告，提示将清理标题设置为 False
            logger.warning("Setting `clean_caption` to False...")
            # 设置清理标题为 False
            clean_caption = False

        # 如果输入文本不是元组或列表，则将其转换为列表
        if not isinstance(text, (tuple, list)):
            text = [text]

        # 定义内部处理函数，接受一个字符串文本
        def process(text: str):
            # 如果需要清理标题，调用清理标题的方法
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)  # 再次调用以确保清理完全
            else:
                # 否则，将文本转换为小写并去除前后空格
                text = text.lower().strip()
            # 返回处理后的文本
            return text

        # 对输入文本列表中的每个文本应用处理函数，并返回结果列表
        return [process(t) for t in text]

    # 禁用梯度计算，提升性能
    @torch.no_grad()
    # 定义编码提示的方法，接受多个参数
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 输入的提示，可以是字符串或字符串列表
        do_classifier_free_guidance: bool = True,  # 是否进行无分类器引导
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        device: Optional[torch.device] = None,  # 指定设备
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负提示
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入
        clean_caption: bool = False,  # 是否清理标题
    # 定义运行安全检查的方法，接受图像、设备和数据类型参数
    def run_safety_checker(self, image, device, dtype):
        # 如果存在安全检查器，则进行安全检查
        if self.safety_checker is not None:
            # 使用特征提取器处理图像，转换为适合模型输入的格式
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            # 进行安全检查，返回检查后的图像及检测结果
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
            )
        else:
            # 如果没有安全检查器，则设置检测结果为 None
            nsfw_detected = None
            watermark_detected = None

        # 返回检查后的图像及检测结果
        return image, nsfw_detected, watermark_detected

    # 准备额外步骤参数的方法，供其他方法使用
    # 准备额外的参数用于调度器步骤，不同调度器可能有不同的参数签名
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta (η) 仅用于 DDIMScheduler，其他调度器将忽略该参数
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 应在 [0, 1] 之间
    
            # 检查调度器是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 创建存放额外参数的字典
            extra_step_kwargs = {}
            # 如果调度器接受 eta，添加 eta 到额外参数字典
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator，添加 generator 到额外参数字典
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回额外参数字典
            return extra_step_kwargs
    
        # 检查输入参数的有效性
        def check_inputs(
            self,
            prompt,
            image,
            original_image,
            batch_size,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_img2img.IFImg2ImgPipeline.preprocess_image 复制而来，将 preprocess_image 替换为 preprocess_original_image
        def preprocess_original_image(self, image: PIL.Image.Image) -> torch.Tensor:
            # 如果输入不是列表，则将其转换为列表
            if not isinstance(image, list):
                image = [image]
    
            # 定义将 numpy 数组转换为 PyTorch 张量的函数
            def numpy_to_pt(images):
                # 如果图像是 3 维，则添加一个维度
                if images.ndim == 3:
                    images = images[..., None]
                # 转换图像格式并创建 PyTorch 张量
                images = torch.from_numpy(images.transpose(0, 3, 1, 2))
                return images
    
            # 如果输入图像是 PIL 图像类型
            if isinstance(image[0], PIL.Image.Image):
                new_image = []
                # 遍历每个图像进行处理
                for image_ in image:
                    # 转换图像为 RGB 格式
                    image_ = image_.convert("RGB")
                    # 调整图像大小
                    image_ = resize(image_, self.unet.config.sample_size)
                    # 转换为 numpy 数组
                    image_ = np.array(image_)
                    # 转换数据类型为 float32
                    image_ = image_.astype(np.float32)
                    # 标准化图像数据
                    image_ = image_ / 127.5 - 1
                    # 将处理后的图像添加到新列表中
                    new_image.append(image_)
    
                # 将新图像列表堆叠为 numpy 数组
                image = np.stack(image, axis=0)  # 转换为 numpy 数组
                # 转换为 PyTorch 张量
                image = numpy_to_pt(image)  # 转换为张量
    
            # 如果输入是 numpy 数组类型
            elif isinstance(image[0], np.ndarray):
                # 根据维度合并多个 numpy 数组
                image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
                # 转换为 PyTorch 张量
                image = numpy_to_pt(image)
    
            # 如果输入是 PyTorch 张量类型
            elif isinstance(image[0], torch.Tensor):
                # 根据维度合并多个张量
                image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)
    
            # 返回处理后的图像
            return image
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_superresolution.IFSuperResolutionPipeline.preprocess_image 复制而来
    # 处理输入图像，将其预处理为适合模型的格式
        def preprocess_image(self, image: PIL.Image.Image, num_images_per_prompt, device) -> torch.Tensor:
            # 检查输入是否为张量或列表，如果不是，则将其转换为列表
            if not isinstance(image, torch.Tensor) and not isinstance(image, list):
                image = [image]
    
            # 如果列表中的第一个元素是 PIL 图像，转换为 NumPy 数组并归一化
            if isinstance(image[0], PIL.Image.Image):
                image = [np.array(i).astype(np.float32) / 127.5 - 1.0 for i in image]
    
                # 将列表中的数组堆叠成一个 NumPy 数组
                image = np.stack(image, axis=0)  # to np
                # 将 NumPy 数组转换为 PyTorch 张量，并调整维度顺序
                image = torch.from_numpy(image.transpose(0, 3, 1, 2))
            # 如果列表中的第一个元素是 NumPy 数组，直接堆叠
            elif isinstance(image[0], np.ndarray):
                image = np.stack(image, axis=0)  # to np
                # 如果数组是五维，取第一维
                if image.ndim == 5:
                    image = image[0]
    
                # 将 NumPy 数组转换为 PyTorch 张量，并调整维度顺序
                image = torch.from_numpy(image.transpose(0, 3, 1, 2))
            # 如果列表中的第一个元素是 PyTorch 张量，检查维度
            elif isinstance(image, list) and isinstance(image[0], torch.Tensor):
                dims = image[0].ndim
    
                # 三维张量堆叠
                if dims == 3:
                    image = torch.stack(image, dim=0)
                # 四维张量连接
                elif dims == 4:
                    image = torch.concat(image, dim=0)
                # 维度不匹配时引发错误
                else:
                    raise ValueError(f"Image must have 3 or 4 dimensions, instead got {dims}")
    
            # 将图像移动到指定设备并设置数据类型
            image = image.to(device=device, dtype=self.unet.dtype)
    
            # 重复图像以匹配每个提示的图像数量
            image = image.repeat_interleave(num_images_per_prompt, dim=0)
    
            # 返回处理后的图像张量
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制
        def get_timesteps(self, num_inference_steps, strength):
            # 计算初始时间步长，确保不超过总步长
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算开始时间步
            t_start = max(num_inference_steps - init_timestep, 0)
            # 获取调度器的时间步列表
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器具有设置开始索引的方法，则调用该方法
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步和有效的推理步长
            return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_img2img.IFImg2ImgPipeline.prepare_intermediate_images 复制
        def prepare_intermediate_images(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None
        ):
            # 解构图像的维度信息
            _, channels, height, width = image.shape
    
            # 计算有效的批量大小
            batch_size = batch_size * num_images_per_prompt
    
            # 创建新的形状元组
            shape = (batch_size, channels, height, width)
    
            # 检查生成器列表的长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 生成随机噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
            # 重复图像以匹配每个提示的图像数量
            image = image.repeat_interleave(num_images_per_prompt, dim=0)
            # 将噪声添加到图像中
            image = self.scheduler.add_noise(image, noise, timestep)
    
            # 返回添加噪声后的图像
            return image
    
        # 禁用梯度计算，以减少内存使用
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用类的 __call__ 方法，允许该类实例像函数一样被调用
        def __call__(
            self,
            # 输入的图像，可以是 PIL 图像、NumPy 数组或 PyTorch 张量
            image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
            # 原始图像，可以是多种格式，默认为 None
            original_image: Union[
                PIL.Image.Image, torch.Tensor, np.ndarray, List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]
            ] = None,
            # 图像处理的强度，默认为 0.8
            strength: float = 0.8,
            # 提示词，可以是单个字符串或字符串列表，默认为 None
            prompt: Union[str, List[str]] = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 时间步的列表，默认为 None
            timesteps: List[int] = None,
            # 指导比例，默认为 4.0
            guidance_scale: float = 4.0,
            # 负提示词，可以是单个字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示词生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 控制随机性的参数，默认为 0.0
            eta: float = 0.0,
            # 生成器，用于控制随机性，可以是单个或多个 PyTorch 生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 提示词的嵌入表示，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示词的嵌入表示，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"（PIL 图像）
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的输出，默认为 True
            return_dict: bool = True,
            # 回调函数，在处理过程中可以调用，默认为 None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调函数调用的步数，默认为 1
            callback_steps: int = 1,
            # 跨注意力的关键字参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 噪声水平，默认为 250
            noise_level: int = 250,
            # 是否清理提示词，默认为 True
            clean_caption: bool = True,
```