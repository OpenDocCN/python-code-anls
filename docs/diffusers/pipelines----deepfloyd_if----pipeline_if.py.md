# `.\diffusers\pipelines\deepfloyd_if\pipeline_if.py`

```py
# 导入 html 模块，用于处理 HTML 文本
import html
# 导入 inspect 模块，用于获取对象的信息
import inspect
# 导入 re 模块，用于正则表达式操作
import re
# 从 urllib.parse 导入模块，作为 ul，用于处理 URL
import urllib.parse as ul
# 导入类型提示所需的类型
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 图像处理器、T5 编码器模型和 T5 标记器
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
    is_bs4_available,  # 检查 bs4 是否可用
    is_ftfy_available,  # 检查 ftfy 是否可用
    logging,  # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的函数
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

# 创建日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 如果 bs4 可用，导入 BeautifulSoup 类
if is_bs4_available():
    from bs4 import BeautifulSoup

# 如果 ftfy 可用，导入该模块
if is_ftfy_available():
    import ftfy

# 定义示例文档字符串，包含示例代码块
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
        ...     image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
        ... ).images

        >>> # save intermediate image
        >>> pil_image = pt_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        >>> safety_modules = {
        ...     "feature_extractor": pipe.feature_extractor,
        ...     "safety_checker": pipe.safety_checker,
        ...     "watermarker": pipe.watermarker,
        ... }
        >>> super_res_2_pipe = DiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
        ... )
        >>> super_res_2_pipe.enable_model_cpu_offload()

        >>> image = super_res_2_pipe(
        ...     prompt=prompt,
        ...     image=image,
        ... ).images
        >>> image[0].save("./if_stage_II.png")
        ```py
""" 
# 定义一个名为 IFPipeline 的类，继承自 DiffusionPipeline 和 StableDiffusionLoraLoaderMixin
class IFPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    # 定义 tokenizer 属性，类型为 T5Tokenizer
    tokenizer: T5Tokenizer
    # 定义 text_encoder 属性，类型为 T5EncoderModel
    text_encoder: T5EncoderModel

    # 定义 unet 属性，类型为 UNet2DConditionModel
    unet: UNet2DConditionModel
    # 定义 scheduler 属性，类型为 DDPMScheduler
    scheduler: DDPMScheduler

    # 定义 feature_extractor 属性，类型为可选的 CLIPImageProcessor
    feature_extractor: Optional[CLIPImageProcessor]
    # 定义 safety_checker 属性，类型为可选的 IFSafetyChecker
    safety_checker: Optional[IFSafetyChecker]

    # 定义 watermarker 属性，类型为可选的 IFWatermarker
    watermarker: Optional[IFWatermarker]

    # 编译一个正则表达式，用于匹配不良标点符号
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

    # 定义一个可选组件的列表
    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]
    # 定义模型 CPU 卸载的顺序
    model_cpu_offload_seq = "text_encoder->unet"
    # 定义需要从 CPU 卸载中排除的组件
    _exclude_from_cpu_offload = ["watermarker"]

    # 初始化方法，定义类的构造函数
    def __init__(
        self,
        # 接收 tokenizer 参数，类型为 T5Tokenizer
        tokenizer: T5Tokenizer,
        # 接收 text_encoder 参数，类型为 T5EncoderModel
        text_encoder: T5EncoderModel,
        # 接收 unet 参数，类型为 UNet2DConditionModel
        unet: UNet2DConditionModel,
        # 接收 scheduler 参数，类型为 DDPMScheduler
        scheduler: DDPMScheduler,
        # 接收可选的 safety_checker 参数，类型为 IFSafetyChecker
        safety_checker: Optional[IFSafetyChecker],
        # 接收可选的 feature_extractor 参数，类型为 CLIPImageProcessor
        feature_extractor: Optional[CLIPImageProcessor],
        # 接收可选的 watermarker 参数，类型为 IFWatermarker
        watermarker: Optional[IFWatermarker],
        # 接收一个布尔值，表示是否需要安全检查器，默认为 True
        requires_safety_checker: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 如果安全检查器为 None 且需要安全检查器，发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 如果安全检查器不为 None 但特征提取器为 None，抛出值错误
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册各个模块
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

    # 采用 torch.no_grad() 装饰器，表示在此装饰的函数中不计算梯度
    @torch.no_grad()
    # 定义编码提示的函数，接受多个参数以配置行为
        def encode_prompt(
            self,
            prompt: Union[str, List[str]],  # 提示内容，可以是字符串或字符串列表
            do_classifier_free_guidance: bool = True,  # 是否启用无分类器引导
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            device: Optional[torch.device] = None,  # 指定设备（如CPU或GPU）
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入张量
            clean_caption: bool = False,  # 是否清理提示文本
        # 定义运行安全检查器的函数，检查生成的图像是否安全
        def run_safety_checker(self, image, device, dtype):
            # 检查是否存在安全检查器
            if self.safety_checker is not None:
                # 提取图像特征并转换为张量，移动到指定设备
                safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
                # 使用安全检查器处理图像，返回处理后的图像和检测结果
                image, nsfw_detected, watermark_detected = self.safety_checker(
                    images=image,
                    clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
                )
            else:
                # 如果没有安全检查器，设置检测结果为None
                nsfw_detected = None
                watermark_detected = None
    
            # 返回处理后的图像和检测结果
            return image, nsfw_detected, watermark_detected
    
        # 从StableDiffusionPipeline类复制的函数，用于准备额外的调度器步骤参数
        # 准备调度器步骤的额外参数，因不同调度器的签名可能不同
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta（η）仅在DDIMScheduler中使用，对于其他调度器将被忽略
            # eta对应于DDIM论文中的η，范围应在[0, 1]之间
    
            # 检查调度器是否接受eta参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}  # 初始化额外参数字典
            # 如果接受eta，添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受generator参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受generator，添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 定义检查输入的函数，确保参数有效
        def check_inputs(
            self,
            prompt,  # 提示内容
            callback_steps,  # 回调步骤，用于调度
            negative_prompt=None,  # 可选的负面提示
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
    ):
        # 检查 callback_steps 是否为 None 或者不是正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            # 抛出错误，确保 callback_steps 是一个正整数
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查是否同时提供了 prompt 和 prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            # 抛出错误，提示只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都未定义
        elif prompt is None and prompt_embeds is None:
            # 抛出错误，要求提供至少一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否有效
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出错误，提示 prompt 必须是字符串或列表
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 negative_prompt 和 negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出错误，提示只能提供其中一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否都不为 None
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            # 检查这两个张量的形状是否一致
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出错误，提示形状不匹配
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 准备中间图像的函数，接受多个参数
    def prepare_intermediate_images(self, batch_size, num_channels, height, width, dtype, device, generator):
        # 定义图像的形状
        shape = (batch_size, num_channels, height, width)
        # 检查生成器的类型和长度是否符合要求
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出错误，提示生成器长度与批量大小不匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 生成随机张量作为初始图像
        intermediate_images = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 按调度器要求的标准差缩放初始噪声
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        # 返回处理后的中间图像
        return intermediate_images
    # 定义一个文本预处理的方法，接收文本和一个可选的清理标志
    def _text_preprocessing(self, text, clean_caption=False):
        # 如果清理标志为真且 BeautifulSoup 库不可用
        if clean_caption and not is_bs4_available():
            # 记录警告，提示用户该库不可用
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            # 记录另一个警告，说明清理标志将被设置为假
            logger.warning("Setting `clean_caption` to False...")
            # 将清理标志设置为假
            clean_caption = False

        # 如果清理标志为真且 ftfy 库不可用
        if clean_caption and not is_ftfy_available():
            # 记录警告，提示用户该库不可用
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            # 记录另一个警告，说明清理标志将被设置为假
            logger.warning("Setting `clean_caption` to False...")
            # 将清理标志设置为假
            clean_caption = False

        # 如果文本不是元组或列表类型
        if not isinstance(text, (tuple, list)):
            # 将文本转换为单元素列表
            text = [text]

        # 定义一个内部处理函数，接收文本字符串
        def process(text: str):
            # 如果清理标志为真
            if clean_caption:
                # 清理文本标题
                text = self._clean_caption(text)
                # 再次清理文本标题（可能是多次清理）
                text = self._clean_caption(text)
            else:
                # 将文本转换为小写并去除首尾空格
                text = text.lower().strip()
            # 返回处理后的文本
            return text

        # 对文本列表中的每个文本应用处理函数，并返回处理结果的列表
        return [process(t) for t in text]

    # 装饰器，指示在此方法内不需要计算梯度
    @torch.no_grad()
    # 装饰器，替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义调用方法，接收多个参数
    def __call__(
        # 接收一个可选的字符串或字符串列表作为提示
        prompt: Union[str, List[str]] = None,
        # 指定推理步骤的数量，默认为100
        num_inference_steps: int = 100,
        # 接收一个可选的时间步列表
        timesteps: List[int] = None,
        # 指导缩放的浮点数，默认为7.0
        guidance_scale: float = 7.0,
        # 可选的负提示字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: Optional[int] = 1,
        # 可选的图像高度
        height: Optional[int] = None,
        # 可选的图像宽度
        width: Optional[int] = None,
        # eta的浮点值，默认为0.0
        eta: float = 0.0,
        # 可选的生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选的提示嵌入张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型的可选字符串，默认为“pil”
        output_type: Optional[str] = "pil",
        # 返回字典的布尔值，默认为真
        return_dict: bool = True,
        # 可选的回调函数，接收三个参数
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调步骤的整数，默认为1
        callback_steps: int = 1,
        # 清理标志，默认为真
        clean_caption: bool = True,
        # 可选的交叉注意力参数字典
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
```