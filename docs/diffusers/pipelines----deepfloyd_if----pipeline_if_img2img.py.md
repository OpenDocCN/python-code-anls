# `.\diffusers\pipelines\deepfloyd_if\pipeline_if_img2img.py`

```py
# 导入处理 HTML 的库
import html
# 导入用于获取对象的源代码的库
import inspect
# 导入正则表达式库
import re
# 导入用于解析 URL 的库，并简化其引用
import urllib.parse as ul
# 导入类型提示所需的类型
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 NumPy 库，常用于科学计算
import numpy as np
# 导入 PIL 库，用于处理图像
import PIL.Image
# 导入 PyTorch 库，深度学习框架
import torch
# 从 Transformers 库中导入图像处理器和模型
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
    PIL_INTERPOLATION,  # PIL 图像插值方法
    is_bs4_available,  # 检查 BeautifulSoup 库是否可用
    is_ftfy_available,  # 检查 ftfy 库是否可用
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

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 如果 BeautifulSoup 库可用，则导入其模块
if is_bs4_available():
    from bs4 import BeautifulSoup

# 如果 ftfy 库可用，则导入其模块
if is_ftfy_available():
    import ftfy


# 定义调整图像大小的函数，接受图像和目标大小作为参数
def resize(images: PIL.Image.Image, img_size: int) -> PIL.Image.Image:
    # 获取图像的宽度和高度
    w, h = images.size

    # 计算宽高比
    coef = w / h

    # 将宽度和高度初始化为目标大小
    w, h = img_size, img_size

    # 根据宽高比调整宽度或高度，使其为8的倍数
    if coef >= 1:
        w = int(round(img_size / 8 * coef) * 8)
    else:
        h = int(round(img_size / 8 / coef) * 8)

    # 使用指定的插值方法调整图像大小
    images = images.resize((w, h), resample=PIL_INTERPOLATION["bicubic"], reducing_gap=None)

    # 返回调整大小后的图像
    return images


# 示例文档字符串，用于说明功能
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
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
    # 示例代码
    Examples:
        ```py
        # 导入需要的库和模块
        >>> from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
        >>> from diffusers.utils import pt_to_pil
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from io import BytesIO

        # 定义图像的URL
        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        # 发送请求获取图像数据
        >>> response = requests.get(url)
        # 打开图像数据并转换为RGB格式
        >>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
        # 调整图像大小为768x512
        >>> original_image = original_image.resize((768, 512))

        # 从预训练模型加载图像到图像管道
        >>> pipe = IFImg2ImgPipeline.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0",
        ...     variant="fp16",
        ...     torch_dtype=torch.float16,
        ... )
        # 启用模型CPU卸载以节省内存
        >>> pipe.enable_model_cpu_offload()

        # 定义生成图像的提示语
        >>> prompt = "A fantasy landscape in style minecraft"
        # 编码提示语以获取正负嵌入
        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        # 生成图像
        >>> image = pipe(
        ...     image=original_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ...     output_type="pt",
        ... ).images

        # 保存中间生成的图像
        >>> pil_image = pt_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        # 从预训练模型加载超分辨率管道
        >>> super_res_1_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0",
        ...     text_encoder=None,
        ...     variant="fp16",
        ...     torch_dtype=torch.float16,
        ... )
        # 启用模型CPU卸载以节省内存
        >>> super_res_1_pipe.enable_model_cpu_offload()

        # 进行超分辨率处理
        >>> image = super_res_1_pipe(
        ...     image=image,
        ...     original_image=original_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ... ).images
        # 保存最终生成的超分辨率图像
        >>> image[0].save("./if_stage_II.png")
        ```py
# 定义一个图像到图像的扩散管道类，继承自DiffusionPipeline和StableDiffusionLoraLoaderMixin
class IFImg2ImgPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    # 定义用于文本处理的分词器
    tokenizer: T5Tokenizer
    # 定义用于编码文本的模型
    text_encoder: T5EncoderModel

    # 定义条件生成的UNet模型
    unet: UNet2DConditionModel
    # 定义扩散调度器
    scheduler: DDPMScheduler

    # 可选的特征提取器
    feature_extractor: Optional[CLIPImageProcessor]
    # 可选的安全检查器
    safety_checker: Optional[IFSafetyChecker]

    # 可选的水印器
    watermarker: Optional[IFWatermarker]

    # 定义不良标点的正则表达式
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

    # 定义可选组件列表
    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]
    # 定义模型CPU卸载的顺序
    model_cpu_offload_seq = "text_encoder->unet"
    # 定义不参与CPU卸载的组件
    _exclude_from_cpu_offload = ["watermarker"]

    # 初始化方法，接收多个参数以配置模型
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        # 调用父类构造函数
        super().__init__()

        # 检查是否禁用安全检查器并发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查是否缺少特征提取器并引发错误
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册模块，包括分词器、文本编码器、UNet、调度器等
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        # 将需要的安全检查器配置注册到配置中
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 设定不需要梯度计算的上下文
    @torch.no_grad()
    # 定义一个用于编码提示的函数，接收多种输入参数
        def encode_prompt(
            self,
            # 提示内容，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 是否使用无分类器自由引导的标志，默认为 True
            do_classifier_free_guidance: bool = True,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 计算设备，默认为 None
            device: Optional[torch.device] = None,
            # 负面提示内容，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 提示的张量表示，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示的张量表示，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 是否清理标题，默认为 False
            clean_caption: bool = False,
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.run_safety_checker 复制而来
        def run_safety_checker(self, image, device, dtype):
            # 如果存在安全检查器，则执行安全检查
            if self.safety_checker is not None:
                # 将图像转换为 PIL 格式，并提取特征
                safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
                # 使用安全检查器检测图像中的不安全内容和水印
                image, nsfw_detected, watermark_detected = self.safety_checker(
                    images=image,
                    clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
                )
            else:
                # 如果没有安全检查器，则不安全检测返回 None
                nsfw_detected = None
                watermark_detected = None
    
            # 返回经过检查的图像及检测结果
            return image, nsfw_detected, watermark_detected
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.prepare_extra_step_kwargs 复制而来
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为不是所有调度器的签名相同
            # eta (η) 仅用于 DDIMScheduler，其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η，范围在 [0, 1] 之间
    
            # 检查调度器步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            if accepts_eta:
                # 如果接受 eta，则将其添加到额外参数中
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            if accepts_generator:
                # 如果接受 generator，则将其添加到额外参数中
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    
        # 定义一个检查输入参数的函数
        def check_inputs(
            self,
            # 提示内容
            prompt,
            # 输入图像
            image,
            # 批处理大小
            batch_size,
            # 回调步骤
            callback_steps,
            # 负面提示，默认为 None
            negative_prompt=None,
            # 提示的张量表示，默认为 None
            prompt_embeds=None,
            # 负面提示的张量表示，默认为 None
            negative_prompt_embeds=None,
    # 该部分代码用于验证输入参数的有效性
        ):
            # 检查回调步数是否有效
            if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                # 抛出异常，提示回调步数必须为正整数
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
    
            # 检查是否同时提供了 prompt 和 prompt_embeds
            if prompt is not None and prompt_embeds is not None:
                # 抛出异常，提示不能同时提供这两个参数
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查是否两个参数都未提供
            elif prompt is None and prompt_embeds is None:
                # 抛出异常，提示至少需要提供一个参数
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查 prompt 的类型
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 抛出异常，提示 prompt 必须是字符串或列表类型
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查负面提示和其嵌入是否同时提供
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 抛出异常，提示不能同时提供这两个参数
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查嵌入的形状是否匹配
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    # 抛出异常，提示两个嵌入的形状必须一致
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
            # 检查图像类型
            if isinstance(image, list):
                check_image_type = image[0]
            else:
                check_image_type = image
    
            # 验证图像类型是否有效
            if (
                not isinstance(check_image_type, torch.Tensor)
                and not isinstance(check_image_type, PIL.Image.Image)
                and not isinstance(check_image_type, np.ndarray)
            ):
                # 抛出异常，提示图像类型无效
                raise ValueError(
                    "`image` has to be of type `torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is"
                    f" {type(check_image_type)}"
                )
    
            # 根据图像类型确定批量大小
            if isinstance(image, list):
                image_batch_size = len(image)
            elif isinstance(image, torch.Tensor):
                image_batch_size = image.shape[0]
            elif isinstance(image, PIL.Image.Image):
                image_batch_size = 1
            elif isinstance(image, np.ndarray):
                image_batch_size = image.shape[0]
            else:
                # 断言无效的图像类型
                assert False
    
            # 检查批量大小是否一致
            if batch_size != image_batch_size:
                # 抛出异常，提示图像批量大小与提示批量大小不一致
                raise ValueError(f"image batch size: {image_batch_size} must be same as prompt batch size {batch_size}")
    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制的代码
    def _text_preprocessing(self, text, clean_caption=False):
        # 检查是否启用清理字幕，且 bs4 模块不可用
        if clean_caption and not is_bs4_available():
            # 记录警告，提示用户缺少 bs4 模块
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            # 记录警告，自动将 clean_caption 设置为 False
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False  # 更新 clean_caption 状态

        # 检查是否启用清理字幕，且 ftfy 模块不可用
        if clean_caption and not is_ftfy_available():
            # 记录警告，提示用户缺少 ftfy 模块
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            # 记录警告，自动将 clean_caption 设置为 False
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False  # 更新 clean_caption 状态

        # 如果 text 不是元组或列表，转换为列表
        if not isinstance(text, (tuple, list)):
            text = [text]  # 将单一文本包裹成列表

        # 定义处理文本的内部函数
        def process(text: str):
            # 如果启用清理字幕，进行两次清理操作
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                # 否则，将文本转换为小写并去除空白
                text = text.lower().strip()
            return text  # 返回处理后的文本

        # 对列表中的每个文本进行处理，并返回结果
        return [process(t) for t in text]  # 返回处理后的文本列表

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制的代码
    def preprocess_image(self, image: PIL.Image.Image) -> torch.Tensor:
        # 如果输入不是列表，转换为列表
        if not isinstance(image, list):
            image = [image]  # 将单一图像包裹成列表

        # 定义将 NumPy 数组转换为 PyTorch 张量的内部函数
        def numpy_to_pt(images):
            # 如果图像维度为 3，增加最后一个维度
            if images.ndim == 3:
                images = images[..., None]
            # 转换为 PyTorch 张量并调整维度顺序
            images = torch.from_numpy(images.transpose(0, 3, 1, 2))
            return images  # 返回转换后的张量

        # 如果图像是 PIL 图像实例
        if isinstance(image[0], PIL.Image.Image):
            new_image = []  # 创建新的图像列表

            # 遍历每个图像进行处理
            for image_ in image:
                image_ = image_.convert("RGB")  # 转换为 RGB 格式
                image_ = resize(image_, self.unet.config.sample_size)  # 调整图像大小
                image_ = np.array(image_)  # 转换为 NumPy 数组
                image_ = image_.astype(np.float32)  # 转换数据类型为 float32
                image_ = image_ / 127.5 - 1  # 归一化到 [-1, 1] 范围
                new_image.append(image_)  # 添加处理后的图像到列表

            image = new_image  # 更新为处理后的图像列表

            # 将图像列表堆叠为 NumPy 数组
            image = np.stack(image, axis=0)  # to np
            # 转换为 PyTorch 张量
            image = numpy_to_pt(image)  # to pt

        # 如果输入图像是 NumPy 数组
        elif isinstance(image[0], np.ndarray):
            # 根据维度将图像合并或堆叠
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = numpy_to_pt(image)  # 转换为 PyTorch 张量

        # 如果输入图像是 PyTorch 张量
        elif isinstance(image[0], torch.Tensor):
            # 根据维度将图像合并或堆叠
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

        return image  # 返回处理后的图像
    # 获取时间步长的函数，基于推理步骤和强度参数
        def get_timesteps(self, num_inference_steps, strength):
            # 根据初始时间步和强度计算最小的时间步
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算开始的时间步，确保不小于0
            t_start = max(num_inference_steps - init_timestep, 0)
            # 从调度器中获取时间步，从t_start开始，按照调度器的顺序
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器有设置开始索引的方法，则调用它
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步和有效推理步骤的数量
            return timesteps, num_inference_steps - t_start
    
        # 准备中间图像的函数
        def prepare_intermediate_images(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None
        ):
            # 获取输入图像的维度信息
            _, channels, height, width = image.shape
    
            # 计算有效的批量大小
            batch_size = batch_size * num_images_per_prompt
    
            # 设置图像的目标形状
            shape = (batch_size, channels, height, width)
    
            # 检查生成器的长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 生成随机噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
            # 按照每个提示重复输入图像
            image = image.repeat_interleave(num_images_per_prompt, dim=0)
            # 向图像中添加噪声
            image = self.scheduler.add_noise(image, noise, timestep)
    
            # 返回处理后的图像
            return image
    
        # 调用函数，进行图像生成
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            # 输入图像可以是多种格式
            image: Union[
                PIL.Image.Image, torch.Tensor, np.ndarray, List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]
            ] = None,
            # 设置强度参数
            strength: float = 0.7,
            # 设置推理步骤数量
            num_inference_steps: int = 80,
            # 时间步列表，默认值为None
            timesteps: List[int] = None,
            # 设置引导比例
            guidance_scale: float = 10.0,
            # 可选的负面提示
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: Optional[int] = 1,
            # 设置eta参数
            eta: float = 0.0,
            # 生成器参数，可选
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 提示的嵌入表示，可选
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示的嵌入表示，可选
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为'pil'
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果
            return_dict: bool = True,
            # 可选的回调函数
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调的步长设置
            callback_steps: int = 1,
            # 是否清理提示
            clean_caption: bool = True,
            # 交叉注意力的额外参数，可选
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
```