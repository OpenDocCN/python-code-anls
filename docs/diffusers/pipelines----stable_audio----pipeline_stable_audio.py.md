# `.\diffusers\pipelines\stable_audio\pipeline_stable_audio.py`

```py
# 版权声明，指明版权归 Stability AI 和 HuggingFace 团队所有
# 
# 根据 Apache 2.0 许可证授权（"许可证"）；
# 除非遵守许可证，否则不得使用此文件。
# 可在以下地址获得许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有规定，否则根据许可证分发的软件
# 是以“按现状”基础提供，不提供任何明示或暗示的保证或条件。
# 有关许可证的具体条款和权限限制，请参阅许可证文档。

import inspect  # 导入 inspect 模块以进行对象的检查
from typing import Callable, List, Optional, Union  # 从 typing 导入类型提示工具

import torch  # 导入 PyTorch 库
from transformers import (  # 从 transformers 导入相关模型和标记器
    T5EncoderModel,  # 导入 T5 编码器模型
    T5Tokenizer,  # 导入 T5 标记器
    T5TokenizerFast,  # 导入快速 T5 标记器
)

from ...models import AutoencoderOobleck, StableAudioDiTModel  # 从模型中导入特定类
from ...models.embeddings import get_1d_rotary_pos_embed  # 导入获取一维旋转位置嵌入的函数
from ...schedulers import EDMDPMSolverMultistepScheduler  # 导入多步调度器类
from ...utils import (  # 从 utils 导入通用工具
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具中导入生成随机张量的函数
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline  # 导入音频管道输出和扩散管道
from .modeling_stable_audio import StableAudioProjectionModel  # 导入稳定音频投影模型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁止 pylint 检查命名

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，包含代码示例
    Examples:  # 示例部分
        ```py  # 代码块开始
        >>> import scipy  # 导入 scipy 库
        >>> import torch  # 导入 PyTorch 库
        >>> import soundfile as sf  # 导入 soundfile 库以处理音频文件
        >>> from diffusers import StableAudioPipeline  # 从 diffusers 导入稳定音频管道

        >>> repo_id = "stabilityai/stable-audio-open-1.0"  # 定义模型的仓库 ID
        >>> pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)  # 从预训练模型加载管道，设置数据类型为 float16
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU

        >>> # 定义提示语
        >>> prompt = "The sound of a hammer hitting a wooden surface."  # 正面提示语
        >>> negative_prompt = "Low quality."  # 负面提示语

        >>> # 为生成器设置种子
        >>> generator = torch.Generator("cuda").manual_seed(0)  # 创建 GPU 上的随机数生成器并设置种子

        >>> # 执行生成
        >>> audio = pipe(  # 调用管道生成音频
        ...     prompt,  # 传入正面提示语
        ...     negative_prompt=negative_prompt,  # 传入负面提示语
        ...     num_inference_steps=200,  # 设置推理步骤数
        ...     audio_end_in_s=10.0,  # 设置音频结束时间为 10 秒
        ...     num_waveforms_per_prompt=3,  # 每个提示生成三个波形
        ...     generator=generator,  # 传入随机数生成器
        ... ).audios  # 获取生成的音频列表

        >>> output = audio[0].T.float().cpu().numpy()  # 转置第一个音频并转换为 NumPy 数组
        >>> sf.write("hammer.wav", output, pipe.vae.sampling_rate)  # 将输出音频写入文件
        ```py  # 代码块结束
"""
    # 文档字符串，描述参数的作用和类型
    Args:
        vae ([`AutoencoderOobleck`]):
            # 变分自编码器 (VAE) 模型，用于将图像编码为潜在表示并从潜在表示解码图像。
        text_encoder ([`~transformers.T5EncoderModel`]):
            # 冻结的文本编码器。StableAudio 使用
            # [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel) 的编码器，
            # 特别是 [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) 变体。
        projection_model ([`StableAudioProjectionModel`]):
            # 一个经过训练的模型，用于线性投影文本编码器模型的隐藏状态和开始及结束秒数。
            # 编码器的投影隐藏状态和条件秒数被连接，以作为变换器模型的输入。
        tokenizer ([`~transformers.T5Tokenizer`]):
            # 用于为冻结文本编码器进行文本标记化的分词器。
        transformer ([`StableAudioDiTModel`]):
            # 用于去噪编码音频潜在表示的 `StableAudioDiTModel`。
        scheduler ([`EDMDPMSolverMultistepScheduler`]):
            # 结合 `transformer` 使用的调度器，用于去噪编码的音频潜在表示。
    """

    # 定义模型组件的顺序，用于 CPU 内存卸载
    model_cpu_offload_seq = "text_encoder->projection_model->transformer->vae"

    # 初始化方法，接收多个模型和组件作为参数
    def __init__(
        self,
        vae: AutoencoderOobleck,
        text_encoder: T5EncoderModel,
        projection_model: StableAudioProjectionModel,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        transformer: StableAudioDiTModel,
        scheduler: EDMDPMSolverMultistepScheduler,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册多个模块，以便后续使用
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            projection_model=projection_model,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        # 计算旋转嵌入维度，注意力头维度的一半
        self.rotary_embed_dim = self.transformer.config.attention_head_dim // 2

    # 从 diffusers.pipelines.pipeline_utils.StableDiffusionMixin 复制的方法，启用 VAE 切片
    def enable_vae_slicing(self):
        r"""
        # 启用切片 VAE 解码。当启用此选项时，VAE 将输入张量切片以
        # 进行分步解码。这有助于节省内存并允许更大的批处理大小。
        """
        # 调用 VAE 的启用切片方法
        self.vae.enable_slicing()

    # 从 diffusers.pipelines.pipeline_utils.StableDiffusionMixin 复制的方法，禁用 VAE 切片
    def disable_vae_slicing(self):
        r"""
        # 禁用切片 VAE 解码。如果之前启用了 `enable_vae_slicing`，
        # 此方法将返回到一次性解码。
        """
        # 调用 VAE 的禁用切片方法
        self.vae.disable_slicing()
    # 编码提示信息的函数定义
        def encode_prompt(
            self,
            prompt,  # 提示内容
            device,  # 设备（CPU或GPU）
            do_classifier_free_guidance,  # 是否使用无分类器自由引导
            negative_prompt=None,  # 可选的负面提示内容
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            attention_mask: Optional[torch.LongTensor] = None,  # 可选的注意力掩码
            negative_attention_mask: Optional[torch.LongTensor] = None,  # 可选的负面注意力掩码
        # 编码音频持续时间的函数定义
        def encode_duration(
            self,
            audio_start_in_s,  # 音频开始时间（秒）
            audio_end_in_s,  # 音频结束时间（秒）
            device,  # 设备（CPU或GPU）
            do_classifier_free_guidance,  # 是否使用无分类器自由引导
            batch_size,  # 批处理大小
        ):
            # 如果开始时间不是列表，则转换为列表
            audio_start_in_s = audio_start_in_s if isinstance(audio_start_in_s, list) else [audio_start_in_s]
            # 如果结束时间不是列表，则转换为列表
            audio_end_in_s = audio_end_in_s if isinstance(audio_end_in_s, list) else [audio_end_in_s]
    
            # 如果开始时间列表长度为1，则扩展为批处理大小
            if len(audio_start_in_s) == 1:
                audio_start_in_s = audio_start_in_s * batch_size
            # 如果结束时间列表长度为1，则扩展为批处理大小
            if len(audio_end_in_s) == 1:
                audio_end_in_s = audio_end_in_s * batch_size
    
            # 将开始时间转换为浮点数列表
            audio_start_in_s = [float(x) for x in audio_start_in_s]
            # 将开始时间转换为张量并移动到指定设备
            audio_start_in_s = torch.tensor(audio_start_in_s).to(device)
    
            # 将结束时间转换为浮点数列表
            audio_end_in_s = [float(x) for x in audio_end_in_s]
            # 将结束时间转换为张量并移动到指定设备
            audio_end_in_s = torch.tensor(audio_end_in_s).to(device)
    
            # 使用投影模型获取输出
            projection_output = self.projection_model(
                start_seconds=audio_start_in_s,  # 开始时间张量
                end_seconds=audio_end_in_s,  # 结束时间张量
            )
            # 获取开始时间的隐藏状态
            seconds_start_hidden_states = projection_output.seconds_start_hidden_states
            # 获取结束时间的隐藏状态
            seconds_end_hidden_states = projection_output.seconds_end_hidden_states
    
            # 如果使用无分类器自由引导，则需要进行两个前向传递
            # 这里复制音频隐藏状态以避免进行两个前向传递
            if do_classifier_free_guidance:
                # 在第一个维度上重复开始时间的隐藏状态
                seconds_start_hidden_states = torch.cat([seconds_start_hidden_states, seconds_start_hidden_states], dim=0)
                # 在第一个维度上重复结束时间的隐藏状态
                seconds_end_hidden_states = torch.cat([seconds_end_hidden_states, seconds_end_hidden_states], dim=0)
    
            # 返回开始和结束时间的隐藏状态
            return seconds_start_hidden_states, seconds_end_hidden_states
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的代码
    # 准备额外的参数用于调度器步骤，因为并非所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 的取值应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个空的字典，用于存放额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta 参数，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator 参数，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外步骤参数的字典
        return extra_step_kwargs

    # 检查输入参数的有效性
    def check_inputs(
        self,
        prompt,  # 输入提示文本
        audio_start_in_s,  # 音频起始时间（秒）
        audio_end_in_s,  # 音频结束时间（秒）
        callback_steps,  # 回调步骤的间隔
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds=None,  # 可选的提示嵌入向量
        negative_prompt_embeds=None,  # 可选的负面提示嵌入向量
        attention_mask=None,  # 可选的注意力掩码
        negative_attention_mask=None,  # 可选的负面注意力掩码
        initial_audio_waveforms=None,  # 初始音频波形（张量）
        initial_audio_sampling_rate=None,  # 初始音频采样率
    # 准备潜在变量
    def prepare_latents(
        self,
        batch_size,  # 批处理大小
        num_channels_vae,  # VAE 的通道数
        sample_size,  # 样本尺寸
        dtype,  # 数据类型
        device,  # 设备信息（如 CPU 或 GPU）
        generator,  # 随机数生成器
        latents=None,  # 可选的潜在变量
        initial_audio_waveforms=None,  # 初始音频波形（张量）
        num_waveforms_per_prompt=None,  # 每个提示的音频波形数量
        audio_channels=None,  # 音频通道数
    # 禁用梯度计算，以提高性能
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 调用方法，执行推理过程
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,  # 输入的提示文本，可以是字符串或字符串列表
        audio_end_in_s: Optional[float] = None,  # 音频结束时间（秒），可选
        audio_start_in_s: Optional[float] = 0.0,  # 音频起始时间（秒），默认为0.0
        num_inference_steps: int = 100,  # 推理步骤数，默认为100
        guidance_scale: float = 7.0,  # 指导因子，默认为7.0
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示文本
        num_waveforms_per_prompt: Optional[int] = 1,  # 每个提示的音频波形数量，默认为1
        eta: float = 0.0,  # eta 值，默认为0.0
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选的随机数生成器
        latents: Optional[torch.Tensor] = None,  # 可选的潜在变量（张量）
        initial_audio_waveforms: Optional[torch.Tensor] = None,  # 可选的初始音频波形（张量）
        initial_audio_sampling_rate: Optional[torch.Tensor] = None,  # 可选的初始音频采样率
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入向量（张量）
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入向量（张量）
        attention_mask: Optional[torch.LongTensor] = None,  # 可选的注意力掩码（张量）
        negative_attention_mask: Optional[torch.LongTensor] = None,  # 可选的负面注意力掩码（张量）
        return_dict: bool = True,  # 是否返回字典格式的结果，默认为 True
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选的回调函数
        callback_steps: Optional[int] = 1,  # 回调步骤的间隔，默认为1
        output_type: Optional[str] = "pt",  # 输出类型，默认为 "pt"
```