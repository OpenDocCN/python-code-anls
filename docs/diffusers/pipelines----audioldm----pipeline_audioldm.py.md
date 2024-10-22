# `.\diffusers\pipelines\audioldm\pipeline_audioldm.py`

```py
# 版权声明，标识该文件的版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面同意，否则根据许可证分发的软件是以“按现状”基础提供，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证下特定语言管理权限和限制的详细信息，请参见许可证。

import inspect  # 导入 inspect 模块，用于获取对象的内部信息
from typing import Any, Callable, Dict, List, Optional, Union  # 从 typing 模块导入类型注解，支持类型提示

import numpy as np  # 导入 numpy 库，通常用于数组和数值计算
import torch  # 导入 PyTorch 库，主要用于深度学习
import torch.nn.functional as F  # 导入 PyTorch 的神经网络功能模块，简化函数调用
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan  # 从 transformers 库导入特定的模型和标记器

from ...models import AutoencoderKL, UNet2DConditionModel  # 从相对路径导入 AutoencoderKL 和 UNet2DConditionModel 模型
from ...schedulers import KarrasDiffusionSchedulers  # 从相对路径导入 KarrasDiffusionSchedulers，用于调度
from ...utils import logging, replace_example_docstring  # 从相对路径导入工具函数，提供日志和文档字符串替换功能
from ...utils.torch_utils import randn_tensor  # 从相对路径导入 randn_tensor 函数，生成随机张量
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline, StableDiffusionMixin  # 从上一级导入音频管道输出和 DiffusionPipeline 相关类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于记录日志信息

EXAMPLE_DOC_STRING = """  # 示例文档字符串，提供使用示例以指导用户
    Examples:  # 示例部分的开始
        ```py  # 使用代码块标记 Python 示例
        >>> from diffusers import AudioLDMPipeline  # 导入 AudioLDMPipeline 类
        >>> import torch  # 导入 PyTorch 库
        >>> import scipy  # 导入 scipy 库，用于科学计算

        >>> repo_id = "cvssp/audioldm-s-full-v2"  # 定义模型的存储库 ID
        >>> pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)  # 从预训练模型创建管道
        >>> pipe = pipe.to("cuda")  # 将管道移动到 CUDA 设备以加速计算

        >>> prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"  # 定义生成音频的提示
        >>> audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]  # 生成音频并获取第一个音频输出

        >>> # 保存音频样本为 .wav 文件
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)  # 使用 scipy 将音频写入文件
        ```py
"""  # 示例文档字符串的结束


class AudioLDMPipeline(DiffusionPipeline, StableDiffusionMixin):  # 定义 AudioLDMPipeline 类，继承自 DiffusionPipeline 和 StableDiffusionMixin
    r"""  # 文档字符串，描述类的功能
    Pipeline for text-to-audio generation using AudioLDM.  # 说明该管道用于文本到音频生成

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods  # 指明该模型继承自 DiffusionPipeline，建议查看父类文档以了解通用方法
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).  # 提及所有管道通用方法的实现（下载、保存、设备运行等）
    # 参数说明，定义模型及其组件
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器模型，用于将图像编码为潜在表示并解码
        text_encoder ([`~transformers.ClapTextModelWithProjection`]):
            # 冻结的文本编码器，具体为ClapTextModelWithProjection变体
            [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused)。
        tokenizer ([`PreTrainedTokenizer`]):
            # 用于文本分词的RobertaTokenizer
        unet ([`UNet2DConditionModel`]):
            # 用于去噪编码音频潜在表示的UNet2DConditionModel
        scheduler ([`SchedulerMixin`]):
            # 与unet结合使用的调度器，用于去噪音频潜在表示，可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`]或[`PNDMScheduler`]。
        vocoder ([`~transformers.SpeechT5HifiGan`]):
            # SpeechT5HifiGan类的声码器
    """

    # 定义模型的CPU卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"

    def __init__(
        # 构造函数参数定义
        self,
        vae: AutoencoderKL,
        text_encoder: ClapTextModelWithProjection,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        # 调用父类构造函数
        super().__init__()

        # 注册模型组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        # 计算VAE的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _encode_prompt(
        # 编码提示参数定义
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    def decode_latents(self, latents):
        # 解码潜在表示，调整比例并生成梅尔谱
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        # 返回生成的梅尔谱
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        # 如果梅尔谱有四维，去掉多余维度
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        # 使用声码器生成波形
        waveform = self.vocoder(mel_spectrogram)
        # 始终转换为float32，确保与bfloat16兼容且开销不大
        waveform = waveform.cpu().float()
        # 返回生成的波形
        return waveform

    # 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs复制
    # 定义一个方法，准备调度器步骤所需的额外参数
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外参数，因为并非所有调度器都有相同的参数签名
            # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 并且应该在 [0, 1] 之间
    
            # 检查调度器的步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化一个字典用于存储额外参数
            extra_step_kwargs = {}
            # 如果调度器接受 eta，则将其添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator，则将其添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 定义一个方法，检查输入参数的有效性
        def check_inputs(
            self,
            prompt,  # 输入的提示文本
            audio_length_in_s,  # 音频长度（以秒为单位）
            vocoder_upsample_factor,  # 声码器上采样因子
            callback_steps,  # 回调步骤
            negative_prompt=None,  # 可选的负向提示文本
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负向提示嵌入
    ):
        # 计算最小音频长度，以秒为单位，基于重采样因子和 VAE 缩放因子
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        # 检查输入音频长度是否小于最小音频长度
        if audio_length_in_s < min_audio_length_in_s:
            # 抛出值错误，说明音频长度必须大于或等于最小值
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        # 检查频率 bins 数是否能被 VAE 缩放因子整除
        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            # 抛出值错误，说明频率 bins 数必须能被 VAE 缩放因子整除
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        # 检查回调步骤是否为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            # 抛出值错误，说明回调步骤必须是正整数
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查是否同时提供了 `prompt` 和 `prompt_embeds`
        if prompt is not None and prompt_embeds is not None:
            # 抛出值错误，说明只能提供一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否两者都未提供
        elif prompt is None and prompt_embeds is None:
            # 抛出值错误，说明至少需要提供一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 `prompt` 类型是否有效
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 抛出值错误，说明 `prompt` 必须是字符串或列表
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 `negative_prompt` 和 `negative_prompt_embeds`
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 抛出值错误，说明只能提供一个
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 `prompt_embeds` 和 `negative_prompt_embeds` 的形状是否匹配
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 抛出值错误，说明两者形状必须相同
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制，宽度调整为 self.vocoder.config.model_in_dim
    # 准备潜在变量，用于生成模型的输入
        def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，包括批大小、通道数、高度和宽度
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(self.vocoder.config.model_in_dim) // self.vae_scale_factor,
            )
            # 检查生成器是否为列表且其长度是否与批大小一致
            if isinstance(generator, list) and len(generator) != batch_size:
                # 如果不一致，则引发值错误
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果未提供潜在变量，则生成随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在变量，则将其转移到指定设备
                latents = latents.to(device)
    
            # 按调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 禁用梯度计算以节省内存
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 提示信息，可以是字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 音频长度（秒）
            audio_length_in_s: Optional[float] = None,
            # 推理步骤数量
            num_inference_steps: int = 10,
            # 引导缩放因子
            guidance_scale: float = 2.5,
            # 负提示信息，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的波形数量
            num_waveforms_per_prompt: Optional[int] = 1,
            # eta参数
            eta: float = 0.0,
            # 生成器，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 是否返回字典格式的结果
            return_dict: bool = True,
            # 回调函数，接收当前步骤和张量
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调步骤频率
            callback_steps: Optional[int] = 1,
            # 跨注意力参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 输出类型，默认为 numpy 格式
            output_type: Optional[str] = "np",
```