# `.\diffusers\pipelines\pixart_alpha\pipeline_pixart_alpha.py`

```py
# 版权声明，2024年PixArt-Alpha团队与HuggingFace团队版权所有
# 
# 根据Apache许可证第2.0版（“许可证”）授权；
# 您只能在遵循许可证的情况下使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”提供的，
# 不附有任何形式的明示或暗示的保证或条件。
# 有关许可证的具体权限和限制，请参阅许可证。

import html  # 导入html库，用于处理HTML内容
import inspect  # 导入inspect库，用于获取对象的信息
import re  # 导入re库，用于正则表达式处理
import urllib.parse as ul  # 导入urllib.parse库并简化命名为ul，用于解析URL
from typing import Callable, List, Optional, Tuple, Union  # 导入类型提示相关的工具

import torch  # 导入PyTorch库，用于深度学习
from transformers import T5EncoderModel, T5Tokenizer  # 从transformers库导入T5编码模型和分词器

from ...image_processor import PixArtImageProcessor  # 从上级模块导入PixArt图像处理器
from ...models import AutoencoderKL, PixArtTransformer2DModel  # 从上级模块导入模型类
from ...schedulers import DPMSolverMultistepScheduler  # 从上级模块导入调度器类
from ...utils import (  # 从上级模块导入多个工具函数和常量
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从上级模块导入扩散管道和图像输出类

logger = logging.get_logger(__name__)  # 创建一个logger实例以记录日志，使用当前模块名称

if is_bs4_available():  # 检查BeautifulSoup库是否可用
    from bs4 import BeautifulSoup  # 如果可用，则导入BeautifulSoup类

if is_ftfy_available():  # 检查ftfy库是否可用
    import ftfy  # 如果可用，则导入ftfy库

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串
    Examples:  # 示例部分
        ```py  # Python代码块开始
        >>> import torch  # 导入torch库
        >>> from diffusers import PixArtAlphaPipeline  # 从diffusers模块导入PixArtAlphaPipeline类

        >>> # 你可以用"PixArt-alpha/PixArt-XL-2-512x512"替换检查点ID。
        >>> pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)  # 创建管道实例，加载预训练模型
        >>> # 启用内存优化。
        >>> pipe.enable_model_cpu_offload()  # 启用模型的CPU卸载以优化内存使用

        >>> prompt = "A small cactus with a happy face in the Sahara desert."  # 定义生成图像的提示
        >>> image = pipe(prompt).images[0]  # 使用提示生成图像并获取第一个图像
        ```py  # Python代码块结束
"""

ASPECT_RATIO_1024_BIN = {  # 定义一个字典以存储不同长宽比下的图像尺寸
    "0.25": [512.0, 2048.0],  # 长宽比为0.25时的尺寸
    "0.28": [512.0, 1856.0],  # 长宽比为0.28时的尺寸
    "0.32": [576.0, 1792.0],  # 长宽比为0.32时的尺寸
    "0.33": [576.0, 1728.0],  # 长宽比为0.33时的尺寸
    "0.35": [576.0, 1664.0],  # 长宽比为0.35时的尺寸
    "0.4": [640.0, 1600.0],  # 长宽比为0.4时的尺寸
    "0.42": [640.0, 1536.0],  # 长宽比为0.42时的尺寸
    "0.48": [704.0, 1472.0],  # 长宽比为0.48时的尺寸
    "0.5": [704.0, 1408.0],  # 长宽比为0.5时的尺寸
    "0.52": [704.0, 1344.0],  # 长宽比为0.52时的尺寸
    "0.57": [768.0, 1344.0],  # 长宽比为0.57时的尺寸
    "0.6": [768.0, 1280.0],  # 长宽比为0.6时的尺寸
    "0.68": [832.0, 1216.0],  # 长宽比为0.68时的尺寸
    "0.72": [832.0, 1152.0],  # 长宽比为0.72时的尺寸
    "0.78": [896.0, 1152.0],  # 长宽比为0.78时的尺寸
    "0.82": [896.0, 1088.0],  # 长宽比为0.82时的尺寸
    "0.88": [960.0, 1088.0],  # 长宽比为0.88时的尺寸
    "0.94": [960.0, 1024.0],  # 长宽比为0.94时的尺寸
    "1.0": [1024.0, 1024.0],  # 长宽比为1.0时的尺寸
    "1.07": [1024.0, 960.0],  # 长宽比为1.07时的尺寸
    "1.13": [1088.0, 960.0],  # 长宽比为1.13时的尺寸
    "1.21": [1088.0, 896.0],  # 长宽比为1.21时的尺寸
    "1.29": [1152.0, 896.0],  # 长宽比为1.29时的尺寸
    "1.38": [1152.0, 832.0],  # 长宽比为1.38时的尺寸
    "1.46": [1216.0, 832.0],  # 长宽比为1.46时的尺寸
    "1.67": [1280.0, 768.0],  # 长宽比为1.67时的尺寸
    "1.75": [1344.0, 768.0],  # 长宽比为1.75时的尺寸
    "2.0": [1408.0, 704.0],  # 长宽比为2.0时的尺寸
    "2.09": [1472.0, 704.0],  # 长宽比为2.09时的尺寸
    "2.4": [1536.0, 640.0],  # 长宽比为2.4时的尺寸
    "2.5": [1600.0, 640.0],  # 长宽比为2.5时的尺寸
    "3.0": [1728.0, 576.0],  # 长宽比为3.0时的尺寸
    "4.0": [2048.0, 512.0],  # 长宽比为4.0时的尺寸
}

ASPECT_RATIO_512_BIN = {  # 定义一个字典以存储512宽度下的不同长宽比图像尺寸
    # 定义一个字典的条目，键为字符串类型的数值，值为包含两个浮点数的列表
    "0.25": [256.0, 1024.0],  # 键 "0.25" 对应的值是一个列表，包含 256.0 和 1024.0
    "0.28": [256.0, 928.0],   # 键 "0.28" 对应的值是一个列表，包含 256.0 和 928.0
    "0.32": [288.0, 896.0],   # 键 "0.32" 对应的值是一个列表，包含 288.0 和 896.0
    "0.33": [288.0, 864.0],   # 键 "0.33" 对应的值是一个列表，包含 288.0 和 864.0
    "0.35": [288.0, 832.0],   # 键 "0.35" 对应的值是一个列表，包含 288.0 和 832.0
    "0.4": [320.0, 800.0],    # 键 "0.4" 对应的值是一个列表，包含 320.0 和 800.0
    "0.42": [320.0, 768.0],   # 键 "0.42" 对应的值是一个列表，包含 320.0 和 768.0
    "0.48": [352.0, 736.0],   # 键 "0.48" 对应的值是一个列表，包含 352.0 和 736.0
    "0.5": [352.0, 704.0],    # 键 "0.5" 对应的值是一个列表，包含 352.0 和 704.0
    "0.52": [352.0, 672.0],   # 键 "0.52" 对应的值是一个列表，包含 352.0 和 672.0
    "0.57": [384.0, 672.0],   # 键 "0.57" 对应的值是一个列表，包含 384.0 和 672.0
    "0.6": [384.0, 640.0],    # 键 "0.6" 对应的值是一个列表，包含 384.0 和 640.0
    "0.68": [416.0, 608.0],   # 键 "0.68" 对应的值是一个列表，包含 416.0 和 608.0
    "0.72": [416.0, 576.0],   # 键 "0.72" 对应的值是一个列表，包含 416.0 和 576.0
    "0.78": [448.0, 576.0],   # 键 "0.78" 对应的值是一个列表，包含 448.0 和 576.0
    "0.82": [448.0, 544.0],   # 键 "0.82" 对应的值是一个列表，包含 448.0 和 544.0
    "0.88": [480.0, 544.0],   # 键 "0.88" 对应的值是一个列表，包含 480.0 和 544.0
    "0.94": [480.0, 512.0],   # 键 "0.94" 对应的值是一个列表，包含 480.0 和 512.0
    "1.0": [512.0, 512.0],    # 键 "1.0" 对应的值是一个列表，包含 512.0 和 512.0
    "1.07": [512.0, 480.0],   # 键 "1.07" 对应的值是一个列表，包含 512.0 和 480.0
    "1.13": [544.0, 480.0],   # 键 "1.13" 对应的值是一个列表，包含 544.0 和 480.0
    "1.21": [544.0, 448.0],   # 键 "1.21" 对应的值是一个列表，包含 544.0 和 448.0
    "1.29": [576.0, 448.0],   # 键 "1.29" 对应的值是一个列表，包含 576.0 和 448.0
    "1.38": [576.0, 416.0],   # 键 "1.38" 对应的值是一个列表，包含 576.0 和 416.0
    "1.46": [608.0, 416.0],   # 键 "1.46" 对应的值是一个列表，包含 608.0 和 416.0
    "1.67": [640.0, 384.0],   # 键 "1.67" 对应的值是一个列表，包含 640.0 和 384.0
    "1.75": [672.0, 384.0],   # 键 "1.75" 对应的值是一个列表，包含 672.0 和 384.0
    "2.0": [704.0, 352.0],    # 键 "2.0" 对应的值是一个列表，包含 704.0 和 352.0
    "2.09": [736.0, 352.0],   # 键 "2.09" 对应的值是一个列表，包含 736.0 和 352.0
    "2.4": [768.0, 320.0],    # 键 "2.4" 对应的值是一个列表，包含 768.0 和 320.0
    "2.5": [800.0, 320.0],    # 键 "2.5" 对应的值是一个列表，包含 800.0 和 320.0
    "3.0": [864.0, 288.0],    # 键 "3.0" 对应的值是一个列表，包含 864.0 和 288.0
    "4.0": [1024.0, 256.0],   # 键 "4.0" 对应的值是一个列表，包含 1024.0 和 256.0
# 定义一个常量字典，表示不同宽高比对应的二进制值
ASPECT_RATIO_256_BIN = {
    # 键为宽高比，值为对应的宽度和高度
    "0.25": [128.0, 512.0],
    "0.28": [128.0, 464.0],
    "0.32": [144.0, 448.0],
    "0.33": [144.0, 432.0],
    "0.35": [144.0, 416.0],
    "0.4": [160.0, 400.0],
    "0.42": [160.0, 384.0],
    "0.48": [176.0, 368.0],
    "0.5": [176.0, 352.0],
    "0.52": [176.0, 336.0],
    "0.57": [192.0, 336.0],
    "0.6": [192.0, 320.0],
    "0.68": [208.0, 304.0],
    "0.72": [208.0, 288.0],
    "0.78": [224.0, 288.0],
    "0.82": [224.0, 272.0],
    "0.88": [240.0, 272.0],
    "0.94": [240.0, 256.0],
    "1.0": [256.0, 256.0],
    "1.07": [256.0, 240.0],
    "1.13": [272.0, 240.0],
    "1.21": [272.0, 224.0],
    "1.29": [288.0, 224.0],
    "1.38": [288.0, 208.0],
    "1.46": [304.0, 208.0],
    "1.67": [320.0, 192.0],
    "1.75": [336.0, 192.0],
    "2.0": [352.0, 176.0],
    "2.09": [368.0, 176.0],
    "2.4": [384.0, 160.0],
    "2.5": [400.0, 160.0],
    "3.0": [432.0, 144.0],
    "4.0": [512.0, 128.0],
}

# 定义一个函数，用于从调度器中检索时间步长
def retrieve_timesteps(
    # 调度器对象
    scheduler,
    # 推断步骤数，可选
    num_inference_steps: Optional[int] = None,
    # 指定设备，可选
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步长，可选
    timesteps: Optional[List[int]] = None,
    # 自定义sigma值，可选
    sigmas: Optional[List[float]] = None,
    # 其他可选参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步长。处理自定义时间步长。
    任何kwargs将传递给 `scheduler.set_timesteps`。

    参数：
        scheduler (`SchedulerMixin`):
            用于获取时间步长的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数。如果使用，`timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步长移动到的设备。如果为 `None`，则不移动时间步长。
        timesteps (`List[int]`, *可选*):
            自定义时间步长，用于覆盖调度器的时间步长间距策略。如果传递 `timesteps`，`num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义sigma值，用于覆盖调度器的时间步长间距策略。如果传递 `sigmas`，`num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是调度器的时间步长调度，第二个元素是推断步骤数。
    """
    # 检查是否同时传递了时间步长和sigma值，抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查 timesteps 是否为 None，如果不为 None，表示需要处理时间步
    if timesteps is not None:
        # 检查当前调度器的 set_timesteps 方法是否接受 timesteps 参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受 timesteps 参数，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                # 报告调度器类不支持自定义时间步调度的错误
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传入 timesteps 和其他参数
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取当前的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果 timesteps 为 None，检查 sigmas 是否不为 None
    elif sigmas is not None:
        # 检查当前调度器的 set_timesteps 方法是否接受 sigmas 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受 sigmas 参数，抛出错误
        if not accept_sigmas:
            raise ValueError(
                # 报告调度器类不支持自定义 sigma 调度的错误
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传入 sigmas 和其他参数
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取当前的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果两者都为 None，使用默认推理步骤数设置调度器
    else:
        # 调用调度器的 set_timesteps 方法，传入推理步骤数和其他参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器获取当前的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤的数量
    return timesteps, num_inference_steps
# 定义一个名为 PixArtAlphaPipeline 的类，继承自 DiffusionPipeline
class PixArtAlphaPipeline(DiffusionPipeline):
    r"""
    用于文本到图像生成的 PixArt-Alpha 管道。

    此模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等），请查看超类文档。

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示。
        text_encoder ([`T5EncoderModel`]):
            冻结的文本编码器。PixArt-Alpha 使用
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)，具体为
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) 变体。
        tokenizer (`T5Tokenizer`):
            类的标记器
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer)。
        transformer ([`PixArtTransformer2DModel`]):
            一个文本条件的 `PixArtTransformer2DModel`，用于去噪编码的图像潜在。
        scheduler ([`SchedulerMixin`]):
            一个调度器，用于与 `transformer` 结合使用，以去噪编码的图像潜在。
    """

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

    # 可选组件的列表，包含 tokenizer 和 text_encoder
    _optional_components = ["tokenizer", "text_encoder"]
    # 定义模型 CPU 卸载的顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 初始化方法，定义类的构造函数
    def __init__(
        self,
        tokenizer: T5Tokenizer,  # 输入的标记器
        text_encoder: T5EncoderModel,  # 输入的文本编码器
        vae: AutoencoderKL,  # 输入的变分自编码器模型
        transformer: PixArtTransformer2DModel,  # 输入的 PixArt 转换器模型
        scheduler: DPMSolverMultistepScheduler,  # 输入的调度器
    ):
        super().__init__()  # 调用父类的构造函数

        # 注册模块，包括 tokenizer、text_encoder、vae、transformer 和 scheduler
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # 根据 VAE 的配置计算缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器实例，使用计算得到的缩放因子
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt 中改编而来
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 输入的提示，可以是字符串或字符串列表
        do_classifier_free_guidance: bool = True,  # 是否使用无分类器自由引导
        negative_prompt: str = "",  # 可选的负面提示
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        device: Optional[torch.device] = None,  # 设备参数，默认是 None
        prompt_embeds: Optional[torch.Tensor] = None,  # 提示的嵌入向量，默认是 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示的嵌入向量，默认是 None
        prompt_attention_mask: Optional[torch.Tensor] = None,  # 提示的注意力掩码，默认是 None
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # 负面提示的注意力掩码，默认是 None
        clean_caption: bool = False,  # 是否清理标题，默认是 False
        max_sequence_length: int = 120,  # 最大序列长度，默认是 120
        **kwargs,  # 其他关键字参数
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制而来
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器都有相同的签名
        # eta（η）仅用于 DDIMScheduler，其他调度器将忽略它。
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 并且应在 [0, 1] 之间
    
        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个空字典以存储额外参数
        extra_step_kwargs = {}
        # 如果接受 eta，将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_steps,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制而来
        def _text_preprocessing(self, text, clean_caption=False):
            # 如果需要清理标题但 bs4 不可用，则记录警告并将 clean_caption 设置为 False
            if clean_caption and not is_bs4_available():
                logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
                logger.warning("Setting `clean_caption` to False...")
                clean_caption = False
    
            # 如果需要清理标题但 ftfy 不可用，则记录警告并将 clean_caption 设置为 False
            if clean_caption and not is_ftfy_available():
                logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
                logger.warning("Setting `clean_caption` to False...")
                clean_caption = False
    
            # 如果 text 不是元组或列表，则将其转换为列表
            if not isinstance(text, (tuple, list)):
                text = [text]
    
            # 定义处理文本的内部函数
            def process(text: str):
                # 如果需要清理标题，调用清理方法
                if clean_caption:
                    text = self._clean_caption(text)
                    text = self._clean_caption(text)
                else:
                    # 否则将文本转为小写并去除空白
                    text = text.lower().strip()
                return text
    
            # 返回处理后的文本列表
            return [process(t) for t in text]
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制而来
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制而来
    # 准备潜在向量，接受多种参数以控制生成过程
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在向量的形状，基于输入参数计算
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,  # 通过 VAE 缩放因子调整高度
            int(width) // self.vae_scale_factor,    # 通过 VAE 缩放因子调整宽度
        )
        # 检查生成器是否是列表且其长度与批次大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                # 抛出错误，提示生成器的长度与请求的批次大小不匹配
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在向量，则生成随机潜在向量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将提供的潜在向量转换到指定的设备上
            latents = latents.to(device)

        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在向量
        return latents

    # 禁用梯度计算以减少内存使用
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 定义调用方法的输入参数及其默认值
        prompt: Union[str, List[str]] = None,  # 正向提示
        negative_prompt: str = "",              # 负向提示
        num_inference_steps: int = 20,          # 推理步骤的数量
        timesteps: List[int] = None,            # 时间步长列表
        sigmas: List[float] = None,             # 噪声标准差列表
        guidance_scale: float = 4.5,            # 引导缩放因子
        num_images_per_prompt: Optional[int] = 1, # 每个提示生成的图像数量
        height: Optional[int] = None,           # 输出图像的高度
        width: Optional[int] = None,            # 输出图像的宽度
        eta: float = 0.0,                       # 附加参数
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 生成器
        latents: Optional[torch.Tensor] = None, # 潜在向量
        prompt_embeds: Optional[torch.Tensor] = None,  # 正向提示嵌入
        prompt_attention_mask: Optional[torch.Tensor] = None, # 正向提示注意力掩码
        negative_prompt_embeds: Optional[torch.Tensor] = None, # 负向提示嵌入
        negative_prompt_attention_mask: Optional[torch.Tensor] = None, # 负向提示注意力掩码
        output_type: Optional[str] = "pil",     # 输出类型，默认为 PIL 图像
        return_dict: bool = True,               # 是否返回字典格式的结果
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None, # 回调函数
        callback_steps: int = 1,                 # 每隔多少步调用一次回调
        clean_caption: bool = True,              # 是否清理标题
        use_resolution_binning: bool = True,     # 是否使用分辨率分箱
        max_sequence_length: int = 120,          # 最大序列长度
        **kwargs,                                # 其他可选参数
```