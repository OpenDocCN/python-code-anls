# `.\diffusers\pipelines\flux\pipeline_flux.py`

```py
# 版权声明，标明所有权和许可信息
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# 依据 Apache License, Version 2.0 进行授权
# 如果不遵循该许可协议，则不得使用此文件
# 可在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在“按原样”基础上分发，
# 不提供任何明示或暗示的保证或条件
# 查看许可证以获取特定的权限和限制

import inspect  # 导入inspect模块以进行对象的检测
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注解

import numpy as np  # 导入numpy库以进行数值计算
import torch  # 导入PyTorch库以进行深度学习
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast  # 导入transformers库中的模型和分词器

from ...image_processor import VaeImageProcessor  # 从图像处理模块导入变分自编码器图像处理器
from ...loaders import FluxLoraLoaderMixin  # 导入FluxLoraLoaderMixin以处理数据加载
from ...models.autoencoders import AutoencoderKL  # 导入KL自编码器模型
from ...models.transformers import FluxTransformer2DModel  # 导入Flux 2D变换器模型
from ...schedulers import FlowMatchEulerDiscreteScheduler  # 导入调度器以处理时间步进
from ...utils import (  # 导入工具函数
    USE_PEFT_BACKEND,  # PEFT后端的使用标志
    is_torch_xla_available,  # 检查Torch XLA可用性
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的工具函数
    scale_lora_layers,  # 缩放LoRA层的工具函数
    unscale_lora_layers,  # 取消缩放LoRA层的工具函数
)
from ...utils.torch_utils import randn_tensor  # 导入用于生成随机张量的工具函数
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道
from .pipeline_output import FluxPipelineOutput  # 导入Flux管道输出

# 检查Torch XLA是否可用，并相应地导入和设置标志
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入XLA模型以支持分布式训练

    XLA_AVAILABLE = True  # 设置XLA可用标志为True
else:
    XLA_AVAILABLE = False  # 设置XLA可用标志为False

logger = logging.get_logger(__name__)  # 创建日志记录器，以当前模块名作为标识

EXAMPLE_DOC_STRING = """  # 示例文档字符串
    Examples:
        ```py
        >>> import torch  # 导入PyTorch库
        >>> from diffusers import FluxPipeline  # 从diffusers导入Flux管道

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)  # 加载预训练的Flux管道
        >>> pipe.to("cuda")  # 将管道移动到CUDA设备
        >>> prompt = "A cat holding a sign that says hello world"  # 设置生成图像的提示
        >>> # 根据使用的变体，管道调用会略有不同
        >>> # 有关更多详细信息，请参阅管道文档
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]  # 生成图像
        >>> image.save("flux.png")  # 保存生成的图像
        ```py
"""  # 示例文档字符串结束

def calculate_shift(  # 定义计算图像序列长度的偏移函数
    image_seq_len,  # 输入参数：图像序列长度
    base_seq_len: int = 256,  # 基本序列长度，默认为256
    max_seq_len: int = 4096,  # 最大序列长度，默认为4096
    base_shift: float = 0.5,  # 基本偏移量，默认为0.5
    max_shift: float = 1.16,  # 最大偏移量，默认为1.16
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)  # 计算斜率
    b = base_shift - m * base_seq_len  # 计算截距
    mu = image_seq_len * m + b  # 计算偏移量
    return mu  # 返回计算得到的偏移量

# 从稳定扩散的管道中复制的检索时间步的函数
def retrieve_timesteps(  # 定义检索时间步的函数
    scheduler,  # 输入参数：调度器对象
    num_inference_steps: Optional[int] = None,  # 可选参数：推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选参数：设备类型
    timesteps: Optional[List[int]] = None,  # 可选参数：时间步列表
    sigmas: Optional[List[float]] = None,  # 可选参数：标准差列表
    **kwargs,  # 其他关键字参数
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步
```  # 函数文档字符串开始
```py  # 文档字符串结束
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
```py  #
    # 函数文档字符串，描述参数和返回值的用途
    """
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    
        Args:
            scheduler (`SchedulerMixin`):  # 定义一个调度器类的实例，用于获取时间步
                The scheduler to get timesteps from.
            num_inference_steps (`int`):  # 定义推理步骤的数量，用于生成样本
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.  # 如果使用此参数，`timesteps`必须为None
            device (`str` or `torch.device`, *optional*):  # 指定将时间步移动到的设备
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):  # 定义自定义时间步以覆盖调度器的时间步策略
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.  # 如果传递此参数，`num_inference_steps`和`sigmas`必须为None
            sigmas (`List[float]`, *optional*):  # 定义自定义sigma以覆盖调度器的时间步策略
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.  # 如果传递此参数，`num_inference_steps`和`timesteps`必须为None
    
        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.  # 返回一个包含时间步调度和推理步骤数量的元组
        """
        # 检查是否同时提供了自定义时间步和sigma
        if timesteps is not None and sigmas is not None:
            # 抛出错误，提示只能传递一个参数
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        # 如果提供了自定义时间步
        if timesteps is not None:
            # 检查调度器是否接受自定义时间步
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            # 如果不接受，抛出错误
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            # 设置调度器的自定义时间步
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            # 获取调度器的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 如果提供了自定义sigma
        elif sigmas is not None:
            # 检查调度器是否接受自定义sigma
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            # 如果不接受，抛出错误
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            # 设置调度器的自定义sigma
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            # 获取调度器的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 如果没有提供自定义时间步或sigma
        else:
            # 根据推理步骤设置调度器的时间步
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器的时间步
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤数量的元组
        return timesteps, num_inference_steps
# 定义一个名为 FluxPipeline 的类，继承自 DiffusionPipeline 和 FluxLoraLoaderMixin
class FluxPipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    r"""
    Flux 管道用于文本到图像生成。

    参考文献: https://blackforestlabs.ai/announcing-black-forest-labs/

    参数:
        transformer ([`FluxTransformer2DModel`]):
            条件变换器 (MMDiT) 架构，用于去噪编码的图像潜在。
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            与 `transformer` 结合使用的调度器，用于去噪编码的图像潜在。
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)，特别是
            [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel)，特别是
            [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) 变体。
        tokenizer (`CLIPTokenizer`):
            类的分词器
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer)。
        tokenizer_2 (`T5TokenizerFast`):
            类的第二个分词器
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast)。
    """

    # 定义一个序列，用于 CPU 卸载模型组件的顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    # 定义可选组件的空列表
    _optional_components = []
    # 定义用于回调的张量输入名称
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    # 初始化方法，接收多个参数以设置对象
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,  # 调度器参数
        vae: AutoencoderKL,  # VAE 参数
        text_encoder: CLIPTextModel,  # 文本编码器参数
        tokenizer: CLIPTokenizer,  # 第一个分词器参数
        text_encoder_2: T5EncoderModel,  # 第二个文本编码器参数
        tokenizer_2: T5TokenizerFast,  # 第二个分词器参数
        transformer: FluxTransformer2DModel,  # 转换器参数
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册多个模块以供使用
        self.register_modules(
            vae=vae,  # 注册 VAE
            text_encoder=text_encoder,  # 注册文本编码器
            text_encoder_2=text_encoder_2,  # 注册第二个文本编码器
            tokenizer=tokenizer,  # 注册第一个分词器
            tokenizer_2=tokenizer_2,  # 注册第二个分词器
            transformer=transformer,  # 注册转换器
            scheduler=scheduler,  # 注册调度器
        )
        # 计算 VAE 的缩放因子，默认值为 16
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        # 创建 VAE 图像处理器，使用 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 设置分词器的最大长度，默认值为 77
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        # 默认样本大小设置为 64
        self.default_sample_size = 64
    # 定义获取 T5 模型提示嵌入的私有方法
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,  # 提示文本，可以是字符串或字符串列表
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        max_sequence_length: int = 512,  # 提示的最大序列长度
        device: Optional[torch.device] = None,  # 可选的设备，默认为 None
        dtype: Optional[torch.dtype] = None,  # 可选的数据类型，默认为 None
    ):
        # 如果未指定设备，则使用类中的执行设备
        device = device or self._execution_device
        # 如果未指定数据类型，则使用文本编码器的默认数据类型
        dtype = dtype or self.text_encoder.dtype

        # 如果提示是字符串，则将其转换为单元素列表
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 获取提示的批处理大小
        batch_size = len(prompt)

        # 使用分词器将提示转换为张量，进行填充、截断等处理
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",  # 填充到最大长度
            max_length=max_sequence_length,  # 最大长度限制
            truncation=True,  # 允许截断
            return_length=False,  # 不返回长度
            return_overflowing_tokens=False,  # 不返回溢出的标记
            return_tensors="pt",  # 返回 PyTorch 张量
        )
        # 获取输入的 ID
        text_input_ids = text_inputs.input_ids
        # 获取未截断的 ID
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        # 检查未截断的 ID 是否长于输入 ID，并且内容不相等
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            # 解码被截断的文本并记录警告
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"  # 日志记录被截断的文本
            )

        # 获取文本嵌入，不输出隐藏状态
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        # 更新数据类型
        dtype = self.text_encoder_2.dtype
        # 将嵌入转换为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # 获取嵌入的序列长度
        _, seq_len, _ = prompt_embeds.shape

        # 为每个提示生成的图像复制文本嵌入，使用与 MPS 兼容的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 重塑嵌入以适应新的批处理大小
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 返回生成的提示嵌入
        return prompt_embeds

    # 定义获取 CLIP 模型提示嵌入的私有方法
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],  # 提示文本，可以是字符串或字符串列表
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        device: Optional[torch.device] = None,  # 可选的设备，默认为 None
    # 开始一个函数定义，使用括号表示参数
        ):
            # 如果未指定设备，则使用实例的执行设备
            device = device or self._execution_device
    
            # 如果 prompt 是字符串，则将其转换为列表形式
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取 prompt 的批处理大小
            batch_size = len(prompt)
    
            # 使用 tokenizer 对 prompt 进行编码，生成张量格式的输入
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",  # 填充到最大长度
                max_length=self.tokenizer_max_length,  # 最大长度设置
                truncation=True,  # 允许截断
                return_overflowing_tokens=False,  # 不返回溢出令牌
                return_length=False,  # 不返回长度信息
                return_tensors="pt",  # 返回 PyTorch 张量
            )
    
            # 获取编码后的输入 ID
            text_input_ids = text_inputs.input_ids
            # 使用最长填充对原始 prompt 进行编码，获取未截断的 ID
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            # 如果未截断的 ID 长度大于等于输入 ID，且不相等，则处理截断警告
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码并记录被截断的文本
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                # 记录警告信息，提示用户部分输入已被截断
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer_max_length} tokens: {removed_text}"
                )
            # 使用文本编码器生成 prompt 的嵌入
            prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
    
            # 使用 CLIPTextModel 的池化输出
            prompt_embeds = prompt_embeds.pooler_output
            # 将嵌入转换为指定的数据类型和设备
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
    
            # 为每个 prompt 生成重复的文本嵌入，使用适合 mps 的方法
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
            # 调整张量形状以适应批处理大小和图像数量
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    
            # 返回处理后的文本嵌入
            return prompt_embeds
    
        # 定义一个新的函数，名为 encode_prompt，接收多个参数
        def encode_prompt(
            self,
            # 定义 prompt，支持字符串或字符串列表
            prompt: Union[str, List[str]],
            # 定义第二个 prompt，支持字符串或字符串列表
            prompt_2: Union[str, List[str]],
            # 可选设备参数，默认值为 None
            device: Optional[torch.device] = None,
            # 每个 prompt 生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 可选的文本嵌入参数，默认值为 None
            prompt_embeds: Optional[torch.FloatTensor] = None,
            # 可选的池化文本嵌入参数，默认值为 None
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 最大序列长度，默认值为 512
            max_sequence_length: int = 512,
            # 可选的 LoRA 比例，默认值为 None
            lora_scale: Optional[float] = None,
        # 定义另一个函数 check_inputs，接收多个参数
        def check_inputs(
            self,
            # 定义第一个 prompt 参数
            prompt,
            # 定义第二个 prompt 参数
            prompt_2,
            # 定义高度参数
            height,
            # 定义宽度参数
            width,
            # 可选的文本嵌入参数，默认值为 None
            prompt_embeds=None,
            # 可选的池化文本嵌入参数，默认值为 None
            pooled_prompt_embeds=None,
            # 可选的回调参数，默认值为 None
            callback_on_step_end_tensor_inputs=None,
            # 可选的最大序列长度，默认值为 None
            max_sequence_length=None,
    ):
        # 检查高度和宽度是否都能被 8 整除
        if height % 8 != 0 or width % 8 != 0:
            # 如果不满足条件，抛出值错误，说明高度和宽度的具体值
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查是否提供了回调输入，并确保它们都在预定义的回调输入中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有未在预定义输入中的回调，抛出值错误，显示具体未找到的回调
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查 prompt 和 prompt_embeds 是否同时提供
        if prompt is not None and prompt_embeds is not None:
            # 如果都提供了，抛出值错误，提醒只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt_2 和 prompt_embeds 是否同时提供
        elif prompt_2 is not None and prompt_embeds is not None:
            # 如果都提供了，抛出值错误，提醒只能提供其中一个
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都未提供
        elif prompt is None and prompt_embeds is None:
            # 如果都未提供，抛出值错误，提醒必须提供一个
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否为字符串或列表
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果类型不符合，抛出值错误，显示实际类型
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        # 检查 prompt_2 的类型是否为字符串或列表
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            # 如果类型不符合，抛出值错误，显示实际类型
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        # 检查是否提供了 prompt_embeds 但未提供 pooled_prompt_embeds
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            # 如果未提供 pooled_prompt_embeds，抛出值错误，说明需要从相同的文本编码器生成
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        # 检查最大序列长度是否大于 512
        if max_sequence_length is not None and max_sequence_length > 512:
            # 如果大于 512，抛出值错误，说明具体的最大序列长度
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    # 准备潜在图像 ID
        def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
            # 创建一个零的张量，形状为 (height // 2, width // 2, 3)
            latent_image_ids = torch.zeros(height // 2, width // 2, 3)
            # 为第二维度增加行索引，形成潜在图像 ID 的位置
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
            # 为第三维度增加列索引
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    
            # 获取潜在图像 ID 的高度、宽度和通道数
            latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    
            # 将潜在图像 ID 复制到 batch_size 的维度上
            latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
            # 重塑张量为 (batch_size, height_id * width_id, channels)
            latent_image_ids = latent_image_ids.reshape(
                batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
            )
    
            # 返回设备和数据类型调整后的潜在图像 ID
            return latent_image_ids.to(device=device, dtype=dtype)
    
        @staticmethod
        # 打包潜在张量
        def _pack_latents(latents, batch_size, num_channels_latents, height, width):
            # 重塑张量为特定形状
            latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
            # 调整维度顺序
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            # 再次重塑张量
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    
            # 返回打包后的潜在张量
            return latents
    
        @staticmethod
        # 解包潜在张量
        def _unpack_latents(latents, height, width, vae_scale_factor):
            # 获取批量大小、补丁数量和通道数
            batch_size, num_patches, channels = latents.shape
    
            # 根据 VAE 缩放因子调整高度和宽度
            height = height // vae_scale_factor
            width = width // vae_scale_factor
    
            # 重塑潜在张量为特定形状
            latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
            # 调整维度顺序
            latents = latents.permute(0, 3, 1, 4, 2, 5)
    
            # 再次重塑张量为最终形状
            latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)
    
            # 返回解包后的潜在张量
            return latents
    
        # 准备潜在张量
        def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
        ):
            # 根据 VAE 缩放因子调整高度和宽度
            height = 2 * (int(height) // self.vae_scale_factor)
            width = 2 * (int(width) // self.vae_scale_factor)
    
            # 定义张量形状
            shape = (batch_size, num_channels_latents, height, width)
    
            # 如果提供了潜在张量，则准备潜在图像 ID
            if latents is not None:
                latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
                return latents.to(device=device, dtype=dtype), latent_image_ids
    
            # 验证生成器列表的长度与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 创建随机潜在张量
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 打包潜在张量
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
    
            # 准备潜在图像 ID
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
    
            # 返回打包后的潜在张量和潜在图像 ID
            return latents, latent_image_ids
    
        # 获取引导比例
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 属性定义结束
    # 定义获取联合注意力参数的方法
    def joint_attention_kwargs(self):
        # 返回存储的联合注意力参数
        return self._joint_attention_kwargs

    # 定义 num_timesteps 属性
    @property
    def num_timesteps(self):
        # 返回存储的时间步数
        return self._num_timesteps

    # 定义 interrupt 属性
    @property
    def interrupt(self):
        # 返回存储的中断状态
        return self._interrupt

    # 禁用梯度计算，以节省内存和加快计算速度
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义调用方法，接受多种参数
    def __call__(
        # 提示信息，可以是字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 第二个提示信息，可以是字符串或字符串列表，默认为 None
        prompt_2: Optional[Union[str, List[str]]] = None,
        # 图像的高度，默认为 None
        height: Optional[int] = None,
        # 图像的宽度，默认为 None
        width: Optional[int] = None,
        # 推理步骤的数量，默认为 28
        num_inference_steps: int = 28,
        # 时间步列表，默认为 None
        timesteps: List[int] = None,
        # 指导比例，默认为 7.0
        guidance_scale: float = 7.0,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 随机数生成器，可以是单个或多个生成器，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在变量，默认为 None
        latents: Optional[torch.FloatTensor] = None,
        # 提示嵌入，默认为 None
        prompt_embeds: Optional[torch.FloatTensor] = None,
        # 池化的提示嵌入，默认为 None
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式，默认为 True
        return_dict: bool = True,
        # 联合注意力参数，默认为 None
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 每个步骤结束时的回调函数，默认为 None
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 在步骤结束时的张量输入名称列表，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 最大序列长度，默认为 512
        max_sequence_length: int = 512,
```