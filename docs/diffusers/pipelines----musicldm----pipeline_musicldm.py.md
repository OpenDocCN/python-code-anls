# `.\diffusers\pipelines\musicldm\pipeline_musicldm.py`

```py
# 版权声明，表明该代码的版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可声明，指明该代码遵循 Apache 许可证 2.0 版本
# Licensed under the Apache License, Version 2.0 (the "License");
# 除非遵循许可证，否则不可使用此文件
# you may not use this file except in compliance with the License.
# 提供许可证获取链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 指出软件按“原样”分发，没有任何明示或暗示的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 参考许可证以了解具体权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块以获取有关对象的信息
import inspect
# 从 typing 模块导入类型提示相关的类型
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库以进行数值计算
import numpy as np
# 导入 PyTorch 库以进行深度学习
import torch
# 从 transformers 库导入多个模型和特征提取器
from transformers import (
    ClapFeatureExtractor,  # 导入 Clap 的特征提取器
    ClapModel,  # 导入 Clap 模型
    ClapTextModelWithProjection,  # 导入带投影的 Clap 文本模型
    RobertaTokenizer,  # 导入 Roberta 分词器
    RobertaTokenizerFast,  # 导入快速 Roberta 分词器
    SpeechT5HifiGan,  # 导入 SpeechT5 的 HiFiGan 模型
)

# 从相对路径导入模型和调度器
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入自编码器和条件 UNet 模型
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (
    is_accelerate_available,  # 检查 accelerate 是否可用
    is_accelerate_version,  # 检查 accelerate 版本
    is_librosa_available,  # 检查 librosa 是否可用
    logging,  # 导入日志记录功能
    replace_example_docstring,  # 导入替换示例文档字符串的工具
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline, StableDiffusionMixin  # 导入音频管道输出和扩散管道类


# 如果 librosa 库可用，则导入它
if is_librosa_available():
    import librosa

# 创建一个日志记录器，用于记录当前模块的信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义示例文档字符串，展示如何使用 MusicLDMPipeline
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import MusicLDMPipeline  # 从 diffusers 导入音乐生成管道
        >>> import torch  # 导入 PyTorch 库
        >>> import scipy  # 导入 SciPy 库用于处理音频

        >>> repo_id = "ucsd-reach/musicldm"  # 定义模型仓库 ID
        >>> pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)  # 从预训练模型创建管道
        >>> pipe = pipe.to("cuda")  # 将管道移至 CUDA 设备

        >>> prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"  # 定义生成音乐的提示
        >>> audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]  # 生成音频

        >>> # 将生成的音频保存为 .wav 文件
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)  # 保存音频文件
        ```py
"""

# 定义 MusicLDMPipeline 类，继承自 DiffusionPipeline 和 StableDiffusionMixin
class MusicLDMPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    用于基于文本生成音频的管道，使用 MusicLDM 模型。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取所有管道实现的通用方法
    (下载、保存、在特定设备上运行等)。
    # 文档字符串，描述构造函数的参数及其类型
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示
        text_encoder ([`~transformers.ClapModel`]):
            # 冻结的文本-音频嵌入模型（`ClapTextModel`），特别是
            # [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) 变体
        tokenizer ([`PreTrainedTokenizer`]):
            # [`~transformers.RobertaTokenizer`] 用于对文本进行分词
        feature_extractor ([`~transformers.ClapFeatureExtractor`]):
            # 特征提取器，用于从音频波形计算梅尔谱图
        unet ([`UNet2DConditionModel`]):
            # `UNet2DConditionModel` 用于去噪编码后的音频潜在表示
        scheduler ([`SchedulerMixin`]):
            # 调度器，与 `unet` 结合使用以去噪编码的音频潜在表示，可以是
            # [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]
        vocoder ([`~transformers.SpeechT5HifiGan`]):
            # `SpeechT5HifiGan` 类的声码器
    """

    # 构造函数，初始化对象的属性
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器模型
        text_encoder: Union[ClapTextModelWithProjection, ClapModel],  # 文本编码器
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],  # 分词器
        feature_extractor: Optional[ClapFeatureExtractor],  # 可选的特征提取器
        unet: UNet2DConditionModel,  # UNet 模型用于去噪
        scheduler: KarrasDiffusionSchedulers,  # 调度器
        vocoder: SpeechT5HifiGan,  # 声码器
    ):
        super().__init__()  # 调用父类的构造函数

        # 注册模块，存储各种组件到对象的属性中
        self.register_modules(
            vae=vae,  # 注册 VAE 模型
            text_encoder=text_encoder,  # 注册文本编码器
            tokenizer=tokenizer,  # 注册分词器
            feature_extractor=feature_extractor,  # 注册特征提取器
            unet=unet,  # 注册 UNet 模型
            scheduler=scheduler,  # 注册调度器
            vocoder=vocoder,  # 注册声码器
        )
        # 计算 VAE 的缩放因子，根据 VAE 配置的块输出通道数计算
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # 编码提示的私有方法，处理输入提示
    def _encode_prompt(
        self,
        prompt,  # 输入提示文本
        device,  # 设备信息
        num_waveforms_per_prompt,  # 每个提示生成的波形数量
        do_classifier_free_guidance,  # 是否进行无分类器引导
        negative_prompt=None,  # 可选的负面提示
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
    # 从 diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline 复制的梅尔谱图到波形的转换方法
    def mel_spectrogram_to_waveform(self, mel_spectrogram):  # 定义方法，将梅尔谱图转换为波形
        # 如果梅尔谱图是四维的，则去掉第一维
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        # 使用声码器将梅尔谱图转换为波形
        waveform = self.vocoder(mel_spectrogram)
        # 始终转换为 float32 类型，因为这不会造成显著的开销，并且与 bfloat16 兼容
        waveform = waveform.cpu().float()  # 将波形移动到 CPU 并转换为 float32 类型
        return waveform  # 返回生成的波形

    # 从 diffusers.pipelines.audioldm2.pipeline_audioldm2.AudioLDM2Pipeline 复制的得分波形的方法
    # 评分音频波形与文本提示之间的匹配度
        def score_waveforms(self, text, audio, num_waveforms_per_prompt, device, dtype):
            # 检查是否安装了 librosa 包
            if not is_librosa_available():
                # 记录信息，提示用户安装 librosa 包以启用自动评分
                logger.info(
                    "Automatic scoring of the generated audio waveforms against the input prompt text requires the "
                    "`librosa` package to resample the generated waveforms. Returning the audios in the order they were "
                    "generated. To enable automatic scoring, install `librosa` with: `pip install librosa`."
                )
                # 返回原始音频
                return audio
            # 对文本进行标记化并返回张量格式的输入
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            # 使用 librosa 对音频进行重采样，调整采样率
            resampled_audio = librosa.resample(
                audio.numpy(), orig_sr=self.vocoder.config.sampling_rate, target_sr=self.feature_extractor.sampling_rate
            )
            # 将重采样后的音频特征提取并转换为指定数据类型
            inputs["input_features"] = self.feature_extractor(
                list(resampled_audio), return_tensors="pt", sampling_rate=self.feature_extractor.sampling_rate
            ).input_features.type(dtype)
            # 将输入数据移动到指定设备上
            inputs = inputs.to(device)
    
            # 使用 CLAP 模型计算音频与文本的相似性得分
            logits_per_text = self.text_encoder(**inputs).logits_per_text
            # 根据文本匹配度对生成的音频进行排序
            indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :num_waveforms_per_prompt]
            # 选择根据排序结果的音频
            audio = torch.index_select(audio, 0, indices.reshape(-1).cpu())
            # 返回排序后的音频
            return audio
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的参数签名
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会被忽略。
            # eta 对应 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 其值应在 [0, 1] 之间
    
            # 检查调度器是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            # 如果接受 eta，则将其添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，则将其添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 从 diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.check_inputs 复制
        def check_inputs(
            self,
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        # 计算最小音频长度（秒），基于声码器上采样因子和 VAE 缩放因子
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        # 如果输入音频长度小于最小音频长度，则抛出错误
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                # 提示音频长度必须大于等于最小音频长度
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        # 检查声码器模型输入维度是否可以被 VAE 缩放因子整除
        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                # 提示频率 bins 数量必须可以被 VAE 缩放因子整除
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        # 检查 callback_steps 是否有效（必须为正整数）
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                # 提示 callback_steps 必须为正整数
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查同时传入 prompt 和 prompt_embeds 是否有效
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                # 提示不能同时提供 prompt 和 prompt_embeds
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否同时为 None
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                # 提示必须提供 prompt 或 prompt_embeds，不能都为空
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否为字符串或列表
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时传入 negative_prompt 和 negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                # 提示不能同时提供 negative_prompt 和 negative_prompt_embeds
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否形状一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    # 提示 prompt_embeds 和 negative_prompt_embeds 形状必须相同
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.prepare_latents 中复制的代码
    # 准备潜在变量，参数包括批次大小、通道数、高度等
        def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，考虑 VAE 的缩放因子
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(self.vocoder.config.model_in_dim) // self.vae_scale_factor,
            )
            # 检查生成器列表的长度是否与批次大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有提供潜在变量，则随机生成
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在变量，将其移动到指定设备
                latents = latents.to(device)
    
            # 根据调度器需要的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 启用模型的 CPU 卸载，以减少内存使用
        def enable_model_cpu_offload(self, gpu_id=0):
            r"""
            Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
            to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
            method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
            `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
            """
            # 检查 accelerate 是否可用且版本是否合适
            if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
                from accelerate import cpu_offload_with_hook
            else:
                # 抛出错误以提示用户需要更新 accelerate
                raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")
    
            # 设置设备为指定的 GPU
            device = torch.device(f"cuda:{gpu_id}")
    
            # 如果当前设备不是 CPU，则将模型移动到 CPU
            if self.device.type != "cpu":
                self.to("cpu", silence_dtype_warnings=True)
                # 清空 GPU 缓存以查看内存节省
                torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)
    
            # 定义需要卸载到 CPU 的模型序列
            model_sequence = [
                self.text_encoder.text_model,
                self.text_encoder.text_projection,
                self.unet,
                self.vae,
                self.vocoder,
                self.text_encoder,
            ]
    
            hook = None
            # 遍历模型序列，逐个卸载到 CPU
            for cpu_offloaded_model in model_sequence:
                _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)
    
            # 手动卸载最后一个模型
            self.final_offload_hook = hook
    
        # 禁用梯度计算，优化性能
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，允许使用不同参数进行推断
    def __call__(
        # 提示内容，可以是单个字符串或字符串列表
        self,
        prompt: Union[str, List[str]] = None,
        # 音频长度，单位为秒，默认为 None 表示不限制
        audio_length_in_s: Optional[float] = None,
        # 推理步骤数量，默认为 200
        num_inference_steps: int = 200,
        # 引导比例，用于控制生成结果的引导强度，默认为 2.0
        guidance_scale: float = 2.0,
        # 负提示，可以是单个字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的波形数量，默认为 1
        num_waveforms_per_prompt: Optional[int] = 1,
        # 采样的 eta 值，默认为 0.0
        eta: float = 0.0,
        # 随机数生成器，可以是单个或多个 PyTorch 生成器，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选的潜在表示，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 可选的提示嵌入，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 是否返回字典形式的结果，默认为 True
        return_dict: bool = True,
        # 可选的回调函数，用于在推理过程中执行
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调执行的步骤间隔，默认为 1
        callback_steps: Optional[int] = 1,
        # 可选的交叉注意力参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 输出类型，默认为 "np"，表示返回 NumPy 数组
        output_type: Optional[str] = "np",
```