# `.\diffusers\pipelines\audioldm2\pipeline_audioldm2.py`

```py
# 版权信息，声明该文件的所有权和使用条款
# Copyright 2024 CVSSP, ByteDance and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License 2.0 许可证使用该文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 仅在遵循许可证的情况下使用该文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律另有规定或书面协议，否则根据许可证分发的软件是按“原样”提供的
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参阅许可证以获取有关权限和限制的具体条款
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect  # 导入 inspect 模块以进行对象检查和获取信息
from typing import Any, Callable, Dict, List, Optional, Union  # 从 typing 模块导入类型注解

import numpy as np  # 导入 numpy 作为 np 以进行数值计算
import torch  # 导入 PyTorch 库以进行深度学习操作
from transformers import (  # 从 transformers 模块导入多个类
    ClapFeatureExtractor,  # 用于提取 Clap 特征的类
    ClapModel,  # Clap 模型类
    GPT2Model,  # GPT-2 模型类
    RobertaTokenizer,  # Roberta 分词器类
    RobertaTokenizerFast,  # 快速 Roberta 分词器类
    SpeechT5HifiGan,  # SpeechT5 HifiGan 类
    T5EncoderModel,  # T5 编码器模型类
    T5Tokenizer,  # T5 分词器类
    T5TokenizerFast,  # 快速 T5 分词器类
    VitsModel,  # VITS 模型类
    VitsTokenizer,  # VITS 分词器类
)

from ...models import AutoencoderKL  # 从上级模块导入 AutoencoderKL 类
from ...schedulers import KarrasDiffusionSchedulers  # 从上级模块导入 KarrasDiffusionSchedulers 类
from ...utils import (  # 从上级模块导入多个工具函数
    is_accelerate_available,  # 检查 accelerate 是否可用的函数
    is_accelerate_version,  # 检查 accelerate 版本的函数
    is_librosa_available,  # 检查 librosa 是否可用的函数
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从上级模块导入 randn_tensor 函数
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline  # 从同级模块导入音频管道输出和扩散管道
from .modeling_audioldm2 import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel  # 从当前模块导入音频 LDM2 模型

if is_librosa_available():  # 如果 librosa 可用
    import librosa  # 导入 librosa 库用于音频处理

logger = logging.get_logger(__name__)  # 创建一个日志记录器，使用当前模块名作为标识

EXAMPLE_DOC_STRING = """  # 示例文档字符串的开始
```  # 示例文档的内容
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
```py  # 示例文档字符串的结束
```  # 示例文档的分隔
    Examples:
        ```py
        >>> import scipy  # 导入 scipy 库，用于处理音频文件
        >>> import torch  # 导入 PyTorch 库，用于深度学习模型的计算
        >>> from diffusers import AudioLDM2Pipeline  # 从 diffusers 库导入 AudioLDM2Pipeline 类，用于音频生成

        >>> repo_id = "cvssp/audioldm2"  # 定义模型的仓库 ID
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)  # 从预训练模型加载管道，并指定数据类型为 float16
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU 上以加速计算

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."  # 定义正向提示语，描述想要生成的音频内容
        >>> negative_prompt = "Low quality."  # 定义负向提示语，表明不希望生成的音频质量

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)  # 创建一个 GPU 上的随机数生成器并设置种子

        >>> # run the generation
        >>> audio = pipe(  # 调用生成管道生成音频
        ...     prompt,  # 使用正向提示语
        ...     negative_prompt=negative_prompt,  # 使用负向提示语
        ...     num_inference_steps=200,  # 设置推理步骤数为 200
        ...     audio_length_in_s=10.0,  # 设置生成音频的时长为 10 秒
        ...     num_waveforms_per_prompt=3,  # 为每个提示生成 3 个波形
        ...     generator=generator,  # 使用之前创建的随机数生成器
        ... ).audios  # 获取生成的音频数据

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])  # 将最佳音频样本（索引 0）保存为 .wav 文件，采样率为 16000
        ```
        ```py
        #Using AudioLDM2 for Text To Speech
        >>> import scipy  # 导入 scipy 库，用于处理音频文件
        >>> import torch  # 导入 PyTorch 库，用于深度学习模型的计算
        >>> from diffusers import AudioLDM2Pipeline  # 从 diffusers 库导入 AudioLDM2Pipeline 类，用于音频生成

        >>> repo_id = "anhnct/audioldm2_gigaspeech"  # 定义 TTS 模型的仓库 ID
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)  # 从预训练模型加载管道，并指定数据类型为 float16
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU 上以加速计算

        >>> # define the prompts
        >>> prompt = "A female reporter is speaking"  # 定义正向提示语，描述想要生成的语音内容
        >>> transcript = "wish you have a good day"  # 定义要生成的语音的转录文本

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)  # 创建一个 GPU 上的随机数生成器并设置种子

        >>> # run the generation
        >>> audio = pipe(  # 调用生成管道生成音频
        ...     prompt,  # 使用正向提示语
        ...     transcription=transcript,  # 使用转录文本
        ...     num_inference_steps=200,  # 设置推理步骤数为 200
        ...     audio_length_in_s=10.0,  # 设置生成音频的时长为 10 秒
        ...     num_waveforms_per_prompt=2,  # 为每个提示生成 2 个波形
        ...     generator=generator,  # 使用之前创建的随机数生成器
        ...     max_new_tokens=512,          #必须将 max_new_tokens 设置为 512 以用于 TTS
        ... ).audios  # 获取生成的音频数据

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("tts.wav", rate=16000, data=audio[0])  # 将最佳音频样本（索引 0）保存为 .wav 文件，采样率为 16000
        ``` 
# 文档字符串，用于描述函数或类的功能
"""


# 定义用于生成输入的函数，接收嵌入和其他参数
def prepare_inputs_for_generation(
    inputs_embeds,  # 输入的嵌入表示
    attention_mask=None,  # 可选的注意力掩码
    past_key_values=None,  # 可选的过去的键值对
    **kwargs,  # 其他可选参数
):
    # 如果提供了过去的键值对
    if past_key_values is not None:
        # 只保留输入嵌入的最后一个 token
        inputs_embeds = inputs_embeds[:, -1:]

    # 返回包含输入嵌入、注意力掩码、过去的键值对及缓存使用标志的字典
    return {
        "inputs_embeds": inputs_embeds,  # 输入的嵌入表示
        "attention_mask": attention_mask,  # 注意力掩码
        "past_key_values": past_key_values,  # 过去的键值对
        "use_cache": kwargs.get("use_cache"),  # 获取使用缓存的标志
    }


# 定义音频生成的管道类，继承自 DiffusionPipeline
class AudioLDM2Pipeline(DiffusionPipeline):
    r"""
    用于基于文本生成音频的管道，使用 AudioLDM2 模型。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解所有管道的通用方法
    （下载、保存、在特定设备上运行等）。
    # 参数说明部分，描述各个参数的用途
        Args:
            vae ([`AutoencoderKL`]):
                # 变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示
            text_encoder ([`~transformers.ClapModel`]):
                # 第一个被冻结的文本编码器。AudioLDM2 使用联合音频-文本嵌入模型
                # [CLAP](https://huggingface.co/docs/transformers/model_doc/clap#transformers.CLAPTextModelWithProjection)，
                # 特别是 [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) 变体。
                # 文本分支用于将文本提示编码为提示嵌入。完整的音频-文本模型用于
                # 通过计算相似度分数来对生成的波形进行排名。
            text_encoder_2 ([`~transformers.T5EncoderModel`, `~transformers.VitsModel`]):
                # 第二个被冻结的文本编码器。AudioLDM2 使用
                # [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel) 的编码器，
                # 特别是 [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) 变体。第二个被冻结的文本编码器
                # 用于文本转语音（TTS）。AudioLDM2 使用
                # [Vits](https://huggingface.co/docs/transformers/model_doc/vits#transformers.VitsModel) 的编码器。
            projection_model ([`AudioLDM2ProjectionModel`]):
                # 一个训练过的模型，用于线性投影第一个和第二个文本编码器模型的隐藏状态，并插入学习到的 SOS 和 EOS 令牌嵌入。
                # 来自两个文本编码器的投影隐藏状态被连接，作为语言模型的输入。
                # 为 Vits 隐藏状态提供学习的位置嵌入。
            language_model ([`~transformers.GPT2Model`]):
                # 自回归语言模型，用于生成一系列基于两个文本编码器的投影输出的隐藏状态。
            tokenizer ([`~transformers.RobertaTokenizer`]):
                # 用于对第一个被冻结的文本编码器进行文本标记化的标记器。
            tokenizer_2 ([`~transformers.T5Tokenizer`, `~transformers.VitsTokenizer`]):
                # 用于对第二个被冻结的文本编码器进行文本标记化的标记器。
            feature_extractor ([`~transformers.ClapFeatureExtractor`]):
                # 特征提取器，用于将生成的音频波形预处理为对数-梅尔谱图，以便进行自动评分。
            unet ([`UNet2DConditionModel`]):
                # 一个 `UNet2DConditionModel`，用于对编码的音频潜在变量进行去噪。
            scheduler ([`SchedulerMixin`]):
                # 调度器，与 `unet` 一起用于去噪编码的音频潜在变量。可以是
                # [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
            vocoder ([`~transformers.SpeechT5HifiGan`]):
                # 类 `SpeechT5HifiGan` 的声码器，用于将梅尔谱图潜在变量转换为最终音频波形。
        """
    # 初始化方法，设置类的属性
    def __init__(
        # VAE（变分自编码器）模型
        self,
        vae: AutoencoderKL,
        # 文本编码器模型
        text_encoder: ClapModel,
        # 第二个文本编码器，可以是 T5 编码器或 Vits 模型
        text_encoder_2: Union[T5EncoderModel, VitsModel],
        # 投影模型，用于音频处理
        projection_model: AudioLDM2ProjectionModel,
        # 语言模型，这里使用 GPT-2 模型
        language_model: GPT2Model,
        # 第一个标记器，可以是 Roberta 标记器或快速版本
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        # 第二个标记器，可以是 T5 标记器、快速版本或 Vits 标记器
        tokenizer_2: Union[T5Tokenizer, T5TokenizerFast, VitsTokenizer],
        # 特征提取器，用于音频特征提取
        feature_extractor: ClapFeatureExtractor,
        # UNet 模型，用于条件生成
        unet: AudioLDM2UNet2DConditionModel,
        # 调度器，用于控制生成过程
        scheduler: KarrasDiffusionSchedulers,
        # 语音合成模型
        vocoder: SpeechT5HifiGan,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册所有模块，将其绑定到当前实例
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            projection_model=projection_model,
            language_model=language_model,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        # 计算 VAE 的缩放因子，基于块输出通道的数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # 启用 VAE 切片解码的方法
    # 当此选项启用时，VAE 将输入张量分成切片进行多步骤解码
    # 有助于节省内存并允许更大的批处理大小
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        # 调用 VAE 的启用切片解码的方法
        self.vae.enable_slicing()

    # 禁用 VAE 切片解码的方法
    # 如果之前启用了 `enable_vae_slicing`，则返回到单步解码
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        # 调用 VAE 的禁用切片解码的方法
        self.vae.disable_slicing()
    # 定义一个方法，用于将所有模型迁移到 CPU，降低内存使用并保持较低的性能影响
    def enable_model_cpu_offload(self, gpu_id=0):
        # 方法的文档字符串，描述其功能和与其他方法的比较
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        # 检查是否可用 accelerate 库，并且版本符合要求
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            # 从 accelerate 库导入 CPU 离线加载函数
            from accelerate import cpu_offload_with_hook
        else:
            # 如果不符合条件，抛出导入错误
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")
    
        # 设置设备为指定的 GPU
        device = torch.device(f"cuda:{gpu_id}")
    
        # 如果当前设备不是 CPU，则迁移模型到 CPU
        if self.device.type != "cpu":
            # 将当前模型迁移到 CPU，抑制数据类型警告
            self.to("cpu", silence_dtype_warnings=True)
            # 清空 CUDA 缓存以释放内存
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)
    
        # 定义一个模型序列，包含需要迁移的各个模型
        model_sequence = [
            self.text_encoder.text_model,
            self.text_encoder.text_projection,
            self.text_encoder_2,
            self.projection_model,
            self.language_model,
            self.unet,
            self.vae,
            self.vocoder,
            self.text_encoder,
        ]
    
        # 初始化钩子变量
        hook = None
        # 遍历模型序列，将每个模型迁移到 CPU，并设置钩子
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)
    
        # 手动离线加载最后一个模型
        self.final_offload_hook = hook
    
    # 定义生成语言模型的方法，接受输入和其他参数
    def generate_language_model(
        self,
        inputs_embeds: torch.Tensor = None,
        max_new_tokens: int = 8,
        **model_kwargs,
    ):
        """
        生成一系列隐藏状态，基于语言模型和嵌入输入进行条件生成。

        参数:
            inputs_embeds (`torch.Tensor` 形状为 `(batch_size, sequence_length, hidden_size)`):
                作为生成提示的序列。
            max_new_tokens (`int`):
                生成的新标记数量。
            model_kwargs (`Dict[str, Any]`, *可选*):
                额外模型特定参数的临时参数化，将传递给模型的 `forward` 函数。

        返回:
            `inputs_embeds (`torch.Tensor` 形状为 `(batch_size, sequence_length, hidden_size)`):
                生成的隐藏状态序列。
        """
        # 如果未指定 max_new_tokens，则使用模型配置中的最大新标记数
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        # 获取输入嵌入的初始缓存位置，并更新模型参数
        model_kwargs = self.language_model._get_initial_cache_position(inputs_embeds, model_kwargs)
        # 循环生成指定数量的新标记
        for _ in range(max_new_tokens):
            # 准备模型输入
            model_inputs = prepare_inputs_for_generation(inputs_embeds, **model_kwargs)

            # 前向传递以获取下一个隐藏状态
            output = self.language_model(**model_inputs, return_dict=True)

            # 获取最后一个隐藏状态
            next_hidden_states = output.last_hidden_state

            # 更新模型输入，将最新的隐藏状态添加到输入嵌入中
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # 更新生成的隐藏状态、模型输入和下一步的长度
            model_kwargs = self.language_model._update_model_kwargs_for_generation(output, model_kwargs)

        # 返回生成的隐藏状态序列中的最后 max_new_tokens 个状态
        return inputs_embeds[:, -max_new_tokens:, :]

    def encode_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        transcription=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generated_prompt_embeds: Optional[torch.Tensor] = None,
        negative_generated_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
    # 从 diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline 复制的函数
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        # 如果梅尔频谱的维度为4，则去掉维度为1的部分
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        # 使用声码器将梅尔频谱转换为波形
        waveform = self.vocoder(mel_spectrogram)
        # 始终转换为 float32，以便与 bfloat16 兼容且不会导致显著的性能开销
        waveform = waveform.cpu().float()
        # 返回转换后的波形
        return waveform
    # 定义一个方法来评分音频波形与文本的相似度
    def score_waveforms(self, text, audio, num_waveforms_per_prompt, device, dtype):
        # 检查是否安装了 librosa 库
        if not is_librosa_available():
            # 如果没有安装，记录信息并返回原始音频
            logger.info(
                "Automatic scoring of the generated audio waveforms against the input prompt text requires the "
                "`librosa` package to resample the generated waveforms. Returning the audios in the order they were "
                "generated. To enable automatic scoring, install `librosa` with: `pip install librosa`."
            )
            return audio
        # 使用 tokenizer 将文本转换为张量并进行填充
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        # 使用 librosa 对音频进行重采样
        resampled_audio = librosa.resample(
            audio.numpy(), orig_sr=self.vocoder.config.sampling_rate, target_sr=self.feature_extractor.sampling_rate
        )
        # 将重采样后的音频转换为输入特征，并设置数据类型
        inputs["input_features"] = self.feature_extractor(
            list(resampled_audio), return_tensors="pt", sampling_rate=self.feature_extractor.sampling_rate
        ).input_features.type(dtype)
        # 将输入转移到指定的设备（CPU/GPU）
        inputs = inputs.to(device)

        # 计算音频与文本的相似度得分，使用 CLAP 模型
        logits_per_text = self.text_encoder(**inputs).logits_per_text
        # 按照与每个提示的匹配程度对生成结果进行排序
        indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :num_waveforms_per_prompt]
        # 根据排序结果选择音频
        audio = torch.index_select(audio, 0, indices.reshape(-1).cpu())
        # 返回选中的音频
        return audio

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的方法
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为不是所有调度器的签名都相同
        # eta（η）仅在 DDIMScheduler 中使用，其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 范围内

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
        # 返回额外的参数字典
        return extra_step_kwargs

    # 定义检查输入参数的方法
    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        transcription=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        generated_prompt_embeds=None,
        negative_generated_prompt_embeds=None,
        attention_mask=None,
        negative_attention_mask=None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的方法，宽度->self.vocoder.config.model_in_dim
    # 准备潜在变量，返回处理后的潜在变量张量
        def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
            # 定义潜在变量的形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(self.vocoder.config.model_in_dim) // self.vae_scale_factor,
            )
            # 检查生成器是否为列表且其长度与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有给定潜在变量，则生成随机张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果给定潜在变量，则将其转移到指定设备
                latents = latents.to(device)
    
            # 根据调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 禁用梯度计算，优化内存使用
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 可选的提示字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 可选的转录字符串或字符串列表
            transcription: Union[str, List[str]] = None,
            # 可选的音频时长，以秒为单位
            audio_length_in_s: Optional[float] = None,
            # 进行推理的步数
            num_inference_steps: int = 200,
            # 引导尺度
            guidance_scale: float = 3.5,
            # 可选的负面提示字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 可选的每个提示生成的波形数量
            num_waveforms_per_prompt: Optional[int] = 1,
            # 控制随机性的参数
            eta: float = 0.0,
            # 可选的生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量张量
            latents: Optional[torch.Tensor] = None,
            # 可选的提示嵌入张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的生成的提示嵌入张量
            generated_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面生成提示嵌入张量
            negative_generated_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的注意力掩码
            attention_mask: Optional[torch.LongTensor] = None,
            # 可选的负面注意力掩码
            negative_attention_mask: Optional[torch.LongTensor] = None,
            # 可选的最大新令牌数量
            max_new_tokens: Optional[int] = None,
            # 是否返回字典格式的输出
            return_dict: bool = True,
            # 可选的回调函数
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 可选的回调步骤
            callback_steps: Optional[int] = 1,
            # 可选的交叉注意力参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的输出类型
            output_type: Optional[str] = "np",
```