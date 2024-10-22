# `.\diffusers\pipelines\pag\pipeline_pag_pixart_sigma.py`

```py
# 版权声明，说明本文件的版权归 PixArt-Sigma 作者和 HuggingFace 团队所有，保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件在 "按原样" 基础上提供，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证下特定权限和限制的信息，请参见许可证。

import html  # 导入 HTML 处理库
import inspect  # 导入用于检查对象的库
import re  # 导入正则表达式库
import urllib.parse as ul  # 导入用于解析和构建 URL 的库
from typing import Callable, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch 库
from transformers import T5EncoderModel, T5Tokenizer  # 从 transformers 库导入 T5 编码器模型和分词器

from ...image_processor import PixArtImageProcessor  # 从相对路径导入 PixArt 图像处理器
from ...models import AutoencoderKL, PixArtTransformer2DModel  # 从相对路径导入模型
from ...schedulers import KarrasDiffusionSchedulers  # 从相对路径导入 Karras 扩散调度器
from ...utils import (  # 从相对路径导入多个实用工具
    BACKENDS_MAPPING,  # 后端映射
    deprecate,  # 废弃标记
    is_bs4_available,  # 检查 BeautifulSoup 是否可用的函数
    is_ftfy_available,  # 检查 ftfy 是否可用的函数
    logging,  # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的工具
)
from ...utils.torch_utils import randn_tensor  # 从相对路径导入生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从相对路径导入扩散管道和图像输出类
from ..pixart_alpha.pipeline_pixart_alpha import (  # 从相对路径导入 PixArt Alpha 管道中的多个常量
    ASPECT_RATIO_256_BIN,  # 256 维的长宽比常量
    ASPECT_RATIO_512_BIN,  # 512 维的长宽比常量
    ASPECT_RATIO_1024_BIN,  # 1024 维的长宽比常量
)
from ..pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN  # 从相对路径导入 2048 维的长宽比常量
from .pag_utils import PAGMixin  # 从当前模块导入 PAGMixin 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例，禁用 pylint 的无效名称警告

if is_bs4_available():  # 检查是否可用 BeautifulSoup 库
    from bs4 import BeautifulSoup  # 从 bs4 导入 BeautifulSoup 类

if is_ftfy_available():  # 检查是否可用 ftfy 库
    import ftfy  # 导入 ftfy 库

EXAMPLE_DOC_STRING = """  # 示例文档字符串，用于展示如何使用该管道
    Examples:  # 示例部分的标题
        ```py  # Python 代码块开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import AutoPipelineForText2Image  # 从 diffusers 库导入自动文本到图像的管道

        >>> pipe = AutoPipelineForText2Image.from_pretrained(  # 从预训练模型加载管道
        ...     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",  # 指定模型路径
        ...     torch_dtype=torch.float16,  # 设置使用的 PyTorch 数据类型
        ...     pag_applied_layers=["blocks.14"],  # 指定应用 PAG 的层
        ...     enable_pag=True,  # 启用 PAG
        ... )  # 结束管道初始化
        >>> pipe = pipe.to("cuda")  # 将管道移动到 CUDA 设备

        >>> prompt = "A small cactus with a happy face in the Sahara desert"  # 定义提示语
        >>> image = pipe(prompt, pag_scale=4.0, guidance_scale=1.0).images[0]  # 生成图像并获取第一张图像
        ```py  # Python 代码块结束
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的函数
def retrieve_timesteps(  # 定义函数以检索时间步
    scheduler,  # 传入调度器对象
    num_inference_steps: Optional[int] = None,  # 可选参数，推理步骤数量
    device: Optional[Union[str, torch.device]] = None,  # 可选参数，设备信息
    timesteps: Optional[List[int]] = None,  # 可选参数，时间步列表
    sigmas: Optional[List[float]] = None,  # 可选参数，sigma 值列表
    **kwargs,  # 额外的关键字参数
):
    """  # 函数文档字符串
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步。处理
    自定义时间步。任何 kwargs 将被传递给 `scheduler.set_timesteps`。
```  # 函数文档字符串结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
```py  # 代码块结束
```  # 代码块结束
    # 参数说明部分，解释函数输入的参数类型和含义
        Args:
            scheduler (`SchedulerMixin`):  # 调度器，用于获取时间步
                The scheduler to get timesteps from.  # 从调度器中获取时间步
            num_inference_steps (`int`):  # 生成样本时使用的扩散步骤数量
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`  # 如果使用此参数，则 `timesteps` 必须为 `None`
                must be `None`.  
            device (`str` or `torch.device`, *optional*):  # 目标设备，时间步将被移动到该设备上
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.  # 如果为 `None`，则不移动时间步
            timesteps (`List[int]`, *optional*):  # 自定义时间步，用于覆盖调度器的时间步间隔策略
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,  # 如果传入 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`
                `num_inference_steps` and `sigmas` must be `None`.  
            sigmas (`List[float]`, *optional*):  # 自定义 sigma，用于覆盖调度器的时间步间隔策略
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,  # 如果传入 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`
                `num_inference_steps` and `timesteps` must be `None`.  
    
        Returns:  # 返回值说明部分
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the  # 返回一个元组，第一个元素是来自调度器的时间步调度
            second element is the number of inference steps.  # 第二个元素是推理步骤的数量
        """
        if timesteps is not None and sigmas is not None:  # 检查是否同时提供了时间步和 sigma
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")  # 如果同时提供，抛出值错误
        if timesteps is not None:  # 如果提供了时间步
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())  # 检查调度器是否支持自定义时间步
            if not accepts_timesteps:  # 如果不支持
                raise ValueError(  # 抛出值错误，提示不支持自定义时间步
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)  # 设置调度器的时间步
            timesteps = scheduler.timesteps  # 获取调度器的时间步
            num_inference_steps = len(timesteps)  # 计算推理步骤的数量
        elif sigmas is not None:  # 如果提供了 sigma
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())  # 检查调度器是否支持自定义 sigma
            if not accept_sigmas:  # 如果不支持
                raise ValueError(  # 抛出值错误，提示不支持自定义 sigma
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)  # 设置调度器的 sigma
            timesteps = scheduler.timesteps  # 获取调度器的时间步
            num_inference_steps = len(timesteps)  # 计算推理步骤的数量
        else:  # 如果既没有提供时间步也没有提供 sigma
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)  # 使用推理步骤数量设置时间步
            timesteps = scheduler.timesteps  # 获取调度器的时间步
        return timesteps, num_inference_steps  # 返回时间步和推理步骤的数量
# 定义一个名为 PixArtSigmaPAGPipeline 的类，继承自 DiffusionPipeline 和 PAGMixin
class PixArtSigmaPAGPipeline(DiffusionPipeline, PAGMixin):
    r"""
    [PAG pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) for text-to-image generation
    using PixArt-Sigma.
    """  # 文档字符串，描述该管道的用途

    # 定义一个用于匹配不良标点符号的正则表达式
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"  # 匹配特定的不良字符
        + r"\)"  # 匹配右括号
        + r"\("  # 匹配左括号
        + r"\]"  # 匹配右方括号
        + r"\["  # 匹配左方括号
        + r"\}"  # 匹配右花括号
        + r"\{"  # 匹配左花括号
        + r"\|"  # 匹配竖线
        + "\\"  # 匹配反斜杠
        + r"\/"  # 匹配斜杠
        + r"\*"  # 匹配星号
        + r"]{1,}"  # 匹配一个或多个上述字符
    )  # noqa

    # 定义可选组件的列表，包括 tokenizer 和 text_encoder
    _optional_components = ["tokenizer", "text_encoder"]
    # 定义模型的 CPU 离线顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 初始化方法，接收多个参数以构建管道
    def __init__(
        self,
        tokenizer: T5Tokenizer,  # 传入的 tokenizer 对象
        text_encoder: T5EncoderModel,  # 传入的文本编码器对象
        vae: AutoencoderKL,  # 传入的变分自编码器对象
        transformer: PixArtTransformer2DModel,  # 传入的变换器模型对象
        scheduler: KarrasDiffusionSchedulers,  # 传入的调度器对象
        pag_applied_layers: Union[str, List[str]] = "blocks.1",  # 应用 PAG 的层，默认为第一个变换器块
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模块以供后续使用
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 PixArt 图像处理器对象，使用 VAE 缩放因子
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 设置 PAG 应用的层
        self.set_pag_applied_layers(pag_applied_layers)

    # 从 PixArtAlphaPipeline 复制的 encode_prompt 方法，修改了最大序列长度
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 输入的提示文本，可以是字符串或字符串列表
        do_classifier_free_guidance: bool = True,  # 是否进行无分类器引导
        negative_prompt: str = "",  # 可选的负面提示文本
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        device: Optional[torch.device] = None,  # 可选的设备参数
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        prompt_attention_mask: Optional[torch.Tensor] = None,  # 可选的提示注意力掩码
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # 可选的负面提示注意力掩码
        clean_caption: bool = False,  # 是否清理标题
        max_sequence_length: int = 300,  # 最大序列长度，默认为 300
        **kwargs,  # 其他可选参数
    # 从 StableDiffusionPipeline 复制的 prepare_extra_step_kwargs 方法
    # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅用于 DDIMScheduler，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个字典来存储额外参数
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha.PixArtAlphaPipeline.check_inputs 复制而来
    def check_inputs(
        self,
        # 输入的提示信息
        prompt,
        # 输出图像的高度
        height,
        # 输出图像的宽度
        width,
        # 负提示信息
        negative_prompt,
        # 回调步骤
        callback_steps,
        # 可选的提示嵌入
        prompt_embeds=None,
        # 可选的负提示嵌入
        negative_prompt_embeds=None,
        # 可选的提示注意力掩码
        prompt_attention_mask=None,
        # 可选的负提示注意力掩码
        negative_prompt_attention_mask=None,
    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制而来
    def _text_preprocessing(self, text, clean_caption=False):
        # 如果需要清理标题但缺少 bs4 库，则发出警告
        if clean_caption and not is_bs4_available():
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            # 将 clean_caption 设置为 False
            clean_caption = False

        # 如果需要清理标题但缺少 ftfy 库，则发出警告
        if clean_caption and not is_ftfy_available():
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            # 将 clean_caption 设置为 False
            clean_caption = False

        # 如果输入的文本不是元组或列表，则将其转换为列表
        if not isinstance(text, (tuple, list)):
            text = [text]

        # 定义处理文本的函数
        def process(text: str):
            # 如果需要清理标题，则执行清理操作
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                # 否则将文本转换为小写并去掉首尾空白
                text = text.lower().strip()
            # 返回处理后的文本
            return text

        # 返回处理后的所有文本
        return [process(t) for t in text]

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制而来
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制而来
    # 准备潜在变量，定义形状和输入参数
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，包含批次大小、通道数和调整后的高度与宽度
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器的类型和数量是否与批次大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                # 如果不匹配，则抛出值错误并提示信息
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果未提供潜在变量，则生成随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在变量，则将其转移到指定设备上
                latents = latents.to(device)
    
            # 按调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回最终的潜在变量
            return latents
    
        # 禁用梯度计算以提高性能
        @torch.no_grad()
        # 替换文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 接受提示，可以是单个字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 负提示，默认为空字符串
            negative_prompt: str = "",
            # 推理步骤的数量，默认为20
            num_inference_steps: int = 20,
            # 时间步列表，默认为None
            timesteps: List[int] = None,
            # sigma值列表，默认为None
            sigmas: List[float] = None,
            # 指导比例，默认为4.5
            guidance_scale: float = 4.5,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 图像高度，默认为None
            height: Optional[int] = None,
            # 图像宽度，默认为None
            width: Optional[int] = None,
            # eta值，默认为0.0
            eta: float = 0.0,
            # 随机生成器，默认为None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，默认为None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 提示注意力掩码，默认为None
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 负提示嵌入，默认为None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示注意力掩码，默认为None
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为True
            return_dict: bool = True,
            # 回调函数，默认为None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调步骤，默认为1
            callback_steps: int = 1,
            # 是否清理提示，默认为True
            clean_caption: bool = True,
            # 是否使用分辨率分箱，默认为True
            use_resolution_binning: bool = True,
            # 最大序列长度，默认为300
            max_sequence_length: int = 300,
            # pag_scale，默认为3.0
            pag_scale: float = 3.0,
            # pag_adaptive_scale，默认为0.0
            pag_adaptive_scale: float = 0.0,
```