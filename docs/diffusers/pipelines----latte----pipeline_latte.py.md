# `.\diffusers\pipelines\latte\pipeline_latte.py`

```py
# 版权所有 2024 Latte 团队和 HuggingFace 团队。
# 所有权利保留。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，按照许可证分发的软件
# 是按“原样”基础分发，不附带任何明示或暗示的担保或条件。
# 有关许可证的具体权限和限制，请参阅许可证。

import html  # 导入 html 模块，用于处理 HTML 实体和字符串
import inspect  # 导入 inspect 模块，用于获取对象的信息
import re  # 导入 re 模块，用于处理正则表达式
import urllib.parse as ul  # 导入 urllib.parse 模块，简化 URL 解析和构建
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器，用于简化数据类的定义
from typing import Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示，用于类型注解

import torch  # 导入 PyTorch 库，用于张量操作和深度学习
from transformers import T5EncoderModel, T5Tokenizer  # 从 transformers 导入 T5 模型和分词器

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 从本地模块导入多管道回调和单管道回调
from ...models import AutoencoderKL, LatteTransformer3DModel  # 从本地模块导入自动编码器和 Latte 3D 模型
from ...pipelines.pipeline_utils import DiffusionPipeline  # 从本地模块导入扩散管道
from ...schedulers import KarrasDiffusionSchedulers  # 从本地模块导入 Karras 扩散调度器
from ...utils import (  # 从本地模块导入多个工具函数和常量
    BACKENDS_MAPPING,  # 后端映射
    BaseOutput,  # 基础输出类
    is_bs4_available,  # 检查 BeautifulSoup 是否可用的函数
    is_ftfy_available,  # 检查 ftfy 是否可用的函数
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
)
from ...utils.torch_utils import is_compiled_module, randn_tensor  # 从工具模块导入 PyTorch 相关的实用函数
from ...video_processor import VideoProcessor  # 从本地模块导入视频处理器

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，命名为模块名，禁用 pylint 的无效名称警告

if is_bs4_available():  # 如果 BeautifulSoup 可用
    from bs4 import BeautifulSoup  # 导入 BeautifulSoup 库

if is_ftfy_available():  # 如果 ftfy 可用
    import ftfy  # 导入 ftfy 库

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，用于展示如何使用某个功能
    Examples:  # 示例标题
        ```py  # Python 代码块开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import LattePipeline  # 从 diffusers 导入 LattePipeline
        >>> from diffusers.utils import export_to_gif  # 从 diffusers 导入导出 GIF 的工具函数

        >>> # 您也可以将检查点 ID 替换为 "maxin-cn/Latte-1"。
        >>> pipe = LattePipeline.from_pretrained("maxin-cn/Latte-1", torch_dtype=torch.float16).to("cuda")  # 从预训练模型加载管道并转移到 GPU
        >>> # 启用内存优化。
        >>> pipe.enable_model_cpu_offload()  # 启用模型的 CPU 卸载以优化内存使用

        >>> prompt = "A small cactus with a happy face in the Sahara desert."  # 定义提示语
        >>> videos = pipe(prompt).frames[0]  # 使用管道生成视频并提取第一帧
        >>> export_to_gif(videos, "latte.gif")  # 将生成的视频导出为 GIF 文件
        ```py  # Python 代码块结束
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制的函数
def retrieve_timesteps(  # 定义一个名为 retrieve_timesteps 的函数
    scheduler,  # 调度器对象，用于控制时间步长
    num_inference_steps: Optional[int] = None,  # 推理步骤的可选数量
    device: Optional[Union[str, torch.device]] = None,  # 设备类型的可选参数，可以是字符串或 torch.device 对象
    timesteps: Optional[List[int]] = None,  # 自定义时间步长的可选列表
    sigmas: Optional[List[float]] = None,  # 自定义 sigma 值的可选列表
    **kwargs,  # 允许其他关键字参数
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器中检索时间步长。处理自定义时间步长。
    任何 kwargs 将被提供给 `scheduler.set_timesteps`。
```  # 函数文档字符串的开始
```py  # 函数文档字符串的结束
```  # 代码块的结束
    # 参数说明
    Args:
        scheduler (`SchedulerMixin`):  # 接受一个调度器，用于获取时间步
            The scheduler to get timesteps from.
        num_inference_steps (`int`):  # 生成样本时使用的扩散步骤数量
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.  # 如果使用此参数，`timesteps` 必须为 None
        device (`str` or `torch.device`, *optional*):  # 指定时间步要移动到的设备
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.  # 如果为 None，时间步不会移动
        timesteps (`List[int]`, *optional*):  # 自定义时间步，覆盖调度器的时间步间隔策略
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.  # 如果传递了此参数，`num_inference_steps` 和 `sigmas` 必须为 None
        sigmas (`List[float]`, *optional*):  # 自定义 sigma，覆盖调度器的时间步间隔策略
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.  # 如果传递了此参数，`num_inference_steps` 和 `timesteps` 必须为 None

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.  # 返回一个元组，第一个元素是调度器的时间步安排，第二个元素是推断步骤的数量
    """
    # 检查是否同时传递了 time步和 sigma
    if timesteps is not None and sigmas is not None:
        # 抛出值错误，提示只能传递一个自定义值
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出值错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推断步骤数量
        num_inference_steps = len(timesteps)
    # 如果传递了 sigma
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出值错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推断步骤数量
        num_inference_steps = len(timesteps)
    # 如果没有传递时间步和 sigma
    else:
        # 使用推断步骤数量设置时间步
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推断步骤数量
    return timesteps, num_inference_steps
# 定义一个数据类，用于保存拉铁管道的输出帧
@dataclass
class LattePipelineOutput(BaseOutput):
    # 输出帧以张量形式存储
    frames: torch.Tensor


# 定义一个拉铁管道类，用于文本到视频生成
class LattePipeline(DiffusionPipeline):
    r"""
    使用拉铁生成文本到视频的管道。

    该模型继承自 [`DiffusionPipeline`]。查看超类文档以获取所有管道实现的通用方法
    （例如下载或保存，运行在特定设备等）。

    参数：
        vae ([`AutoencoderKL`]):
            用于将视频编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`T5EncoderModel`]):
            冻结的文本编码器。拉铁使用
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)，特别是
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) 变体。
        tokenizer (`T5Tokenizer`):
            类
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer) 的分词器。
        transformer ([`LatteTransformer3DModel`]):
            一个文本条件的 `LatteTransformer3DModel`，用于去噪编码的视频潜在表示。
        scheduler ([`SchedulerMixin`]):
            与 `transformer` 结合使用的调度器，用于去噪编码的视频潜在表示。
    """

    # 定义一个正则表达式，用于匹配坏的标点符号
    bad_punct_regex = re.compile(r"[#®•©™&@·º½¾¿¡§~\)\(\]\[\}\{\|\\/\\*]{1,}")

    # 可选组件列表，包括分词器和文本编码器
    _optional_components = ["tokenizer", "text_encoder"]
    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 定义需要作为回调的张量输入
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    # 初始化方法，接受各个组件作为参数
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: LatteTransformer3DModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 注册各个模块，包括分词器、文本编码器、VAE、变换器和调度器
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建视频处理器实例，传入缩放因子
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor)

    # 从指定链接修改而来，掩蔽文本嵌入
    def mask_text_embeddings(self, emb, mask):
        # 如果嵌入的第一个维度为1
        if emb.shape[0] == 1:
            # 计算保留的索引
            keep_index = mask.sum().item()
            # 返回被掩蔽的嵌入和保留的索引
            return emb[:, :, :keep_index, :], keep_index  # 1, 120, 4096 -> 1 7 4096
        else:
            # 应用掩蔽，生成被掩蔽的特征
            masked_feature = emb * mask[:, None, :, None]  # 1 120 4096
            # 返回被掩蔽的特征和原始嵌入的形状
            return masked_feature, emb.shape[2]

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt 修改而来
    # 定义一个编码提示的函数
    def encode_prompt(
            self,  # 当前实例的引用
            prompt: Union[str, List[str]],  # 输入的提示，支持字符串或字符串列表
            do_classifier_free_guidance: bool = True,  # 是否使用无分类器引导
            negative_prompt: str = "",  # 负提示，用于引导模型避免生成特定内容
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            device: Optional[torch.device] = None,  # 指定设备（如 GPU），默认为 None
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 提示的嵌入表示，默认为 None
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 负提示的嵌入表示，默认为 None
            clean_caption: bool = False,  # 是否清理标题，默认为 False
            mask_feature: bool = True,  # 是否使用特征掩码，默认为 True
            dtype=None,  # 数据类型，默认为 None
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
        def prepare_extra_step_kwargs(self, generator, eta):  # 定义一个准备额外步骤参数的函数
            # 为调度器步骤准备额外的参数，因为并非所有调度器具有相同的参数签名
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将被忽略。
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 并应在 [0, 1] 范围内
    
            # 检查调度器步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}  # 初始化额外步骤参数字典
            if accepts_eta:  # 如果接受 eta
                extra_step_kwargs["eta"] = eta  # 将 eta 添加到额外参数中
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            if accepts_generator:  # 如果接受 generator
                extra_step_kwargs["generator"] = generator  # 将 generator 添加到额外参数中
            return extra_step_kwargs  # 返回准备好的额外参数
    
        # 定义一个检查输入的函数
        def check_inputs(
            self,  # 当前实例的引用
            prompt,  # 输入的提示
            height,  # 图像高度
            width,  # 图像宽度
            negative_prompt,  # 负提示
            callback_on_step_end_tensor_inputs,  # 步骤结束时的回调
            prompt_embeds=None,  # 提示嵌入，默认为 None
            negative_prompt_embeds=None,  # 负提示嵌入，默认为 None
    ):  # 定义函数的结束括号
        # 检查高度和宽度是否为8的倍数
        if height % 8 != 0 or width % 8 != 0:
            # 如果不是，抛出值错误，提示高度和宽度必须是8的倍数
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调的输入是否为None且是否包含在预定义的回调输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果不在，抛出值错误，提示回调输入不合法
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        # 检查是否同时提供了prompt和prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            # 如果是，抛出值错误，提示只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查prompt和prompt_embeds是否都为None
        elif prompt is None and prompt_embeds is None:
            # 如果是，抛出值错误，提示必须提供其中一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查prompt的类型是否合法
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果不合法，抛出值错误，提示类型错误
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了prompt和negative_prompt_embeds
        if prompt is not None and negative_prompt_embeds is not None:
            # 如果是，抛出值错误，提示只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查是否同时提供了negative_prompt和negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果是，抛出值错误，提示只能提供其中一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查prompt_embeds和negative_prompt_embeds是否同时提供且形状一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不一致，抛出值错误，提示形状必须相同
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制的部分
    # 文本预处理方法，接受文本和一个指示是否清理标题的标志
    def _text_preprocessing(self, text, clean_caption=False):
        # 如果需要清理标题且 BeautifulSoup4 不可用，记录警告并禁用清理标志
        if clean_caption and not is_bs4_available():
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        # 如果需要清理标题且 ftfy 不可用，记录警告并禁用清理标志
        if clean_caption and not is_ftfy_available():
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        # 如果文本不是元组或列表，则将其转为列表
        if not isinstance(text, (tuple, list)):
            text = [text]

        # 定义处理单个文本的内部函数
        def process(text: str):
            # 如果需要清理标题，则调用清理方法两次
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                # 否则将文本转换为小写并去除首尾空格
                text = text.lower().strip()
            return text

        # 对列表中的每个文本进行处理并返回结果
        return [process(t) for t in text]

    # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制
    # 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents 复制
    def prepare_latents(
        # 定义方法参数，指定批量大小、通道数、帧数、高度、宽度、数据类型、设备、生成器和潜在变量
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 计算潜在变量的形状
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 检查生成器列表长度是否与批量大小匹配，若不匹配则抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果潜在变量为 None，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 否则将已有潜在变量转换到指定设备
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 定义属性，返回指导比例的值
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 这里定义的 `guidance_scale` 类似于 Imagen 论文中公式（2）中的指导权重 `w`
    # `guidance_scale = 1` 表示不进行无分类器引导
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # 定义属性，返回时间步数的值
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 定义属性，返回中断标志的值
    @property
    def interrupt(self):
        return self._interrupt

    # 在不计算梯度的情况下执行以下方法
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的 __call__ 方法，接受多个参数
        def __call__(
            # 输入提示，可以是字符串或字符串列表，默认为 None
            self,
            prompt: Union[str, List[str]] = None,
            # 负向提示，默认为空字符串
            negative_prompt: str = "",
            # 推理步骤数量，默认为 50
            num_inference_steps: int = 50,
            # 时间步列表，默认为 None
            timesteps: Optional[List[int]] = None,
            # 引导尺度，默认为 7.5
            guidance_scale: float = 7.5,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 视频长度，默认为 16
            video_length: int = 16,
            # 图像高度，默认为 512
            height: int = 512,
            # 图像宽度，默认为 512
            width: int = 512,
            # ETA 值，默认为 0.0
            eta: float = 0.0,
            # 随机生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在空间张量，默认为 None
            latents: Optional[torch.FloatTensor] = None,
            # 提示嵌入张量，默认为 None
            prompt_embeds: Optional[torch.FloatTensor] = None,
            # 负向提示嵌入张量，默认为 None
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 输出类型，默认为 "pil"
            output_type: str = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 步骤结束时的张量输入回调，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 是否清理提示，默认为 True
            clean_caption: bool = True,
            # 是否启用特征掩码，默认为 True
            mask_feature: bool = True,
            # 是否启用时间注意力机制，默认为 True
            enable_temporal_attentions: bool = True,
            # 解码块大小，默认为 None
            decode_chunk_size: Optional[int] = None,
        # 与 diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion.decode_latents 相似
        def decode_latents(self, latents: torch.Tensor, video_length: int, decode_chunk_size: int = 14):
            # 将张量维度调整为 [batch, channels, frames, height, width] 格式
            latents = latents.permute(0, 2, 1, 3, 4).flatten(0, 1)
    
            # 使用缩放因子调整潜在空间张量
            latents = 1 / self.vae.config.scaling_factor * latents
    
            # 获取 VAE 的前向函数，判断其是否编译过
            forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
            # 检查前向函数是否接受帧数参数
            accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
    
            # 每次解码 decode_chunk_size 帧以避免内存溢出
            frames = []
            for i in range(0, latents.shape[0], decode_chunk_size):
                # 当前块中的帧数量
                num_frames_in = latents[i : i + decode_chunk_size].shape[0]
                decode_kwargs = {}
                if accepts_num_frames:
                    # 如果需要，传递当前帧数量
                    decode_kwargs["num_frames"] = num_frames_in
    
                # 解码当前块的潜在张量，获取样本帧
                frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
                frames.append(frame)
            # 将所有帧沿第 0 维拼接
            frames = torch.cat(frames, dim=0)
    
            # 将帧维度调整回 [batch, channels, frames, height, width] 格式
            frames = frames.reshape(-1, video_length, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
    
            # 将帧转换为 float32 类型，以保持兼容性
            frames = frames.float()
            # 返回处理后的帧
            return frames
```