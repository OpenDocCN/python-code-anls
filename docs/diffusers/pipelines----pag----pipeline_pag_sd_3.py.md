# `.\diffusers\pipelines\pag\pipeline_pag_sd_3.py`

```py
# 版权声明，指明版权归属及相关许可信息
# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可协议授权使用此文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 本文件的使用需遵循该许可协议
# You may not use this file except in compliance with the License.
# 可在以下链接获取许可的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非另有书面协议，否则根据许可分发的软件不提供任何保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不承担任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可协议中关于权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入用于检查对象信息的模块
import inspect
# 导入类型提示相关的类型
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 导入 transformers 库中的相关模型和分词器
from transformers import (
    CLIPTextModelWithProjection,  # 导入 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 分词器
    T5EncoderModel,  # 导入 T5 编码器模型
    T5TokenizerFast,  # 导入快速 T5 分词器
)

# 导入自定义图像处理器
from ...image_processor import VaeImageProcessor
# 导入自定义加载器
from ...loaders import FromSingleFileMixin, SD3LoraLoaderMixin
# 导入自定义注意力处理器
from ...models.attention_processor import PAGCFGJointAttnProcessor2_0, PAGJointAttnProcessor2_0
# 导入自定义自动编码器
from ...models.autoencoders import AutoencoderKL
# 导入自定义变换器模型
from ...models.transformers import SD3Transformer2DModel
# 导入自定义调度器
from ...schedulers import FlowMatchEulerDiscreteScheduler
# 导入各种实用工具函数
from ...utils import (
    USE_PEFT_BACKEND,  # 导入是否使用 PEFT 后端的标识
    is_torch_xla_available,  # 导入检查 XLA 是否可用的函数
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放 LoRA 层的函数
    unscale_lora_layers,  # 导入取消缩放 LoRA 层的函数
)
# 导入 PyTorch 相关的实用工具函数
from ...utils.torch_utils import randn_tensor
# 导入扩散管道的实用工具
from ..pipeline_utils import DiffusionPipeline
# 导入稳定扩散的管道输出
from ..stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
# 导入 PAG 相关的实用工具
from .pag_utils import PAGMixin


# 检查 XLA 是否可用，如果可用则导入 XLA 相关模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 核心模型
    XLA_AVAILABLE = True  # 设置 XLA 可用标识为真
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标识为假


# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，包含使用示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipe = AutoPipelineForText2Image.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers",
        ...     torch_dtype=torch.float16,
        ...     enable_pag=True,
        ...     pag_applied_layers=["blocks.13"],
        ... )
        >>> pipe.to("cuda")  # 将管道移动到 GPU
        >>> prompt = "A cat holding a sign that says hello world"  # 定义生成图像的提示
        >>> image = pipe(prompt, guidance_scale=5.0, pag_scale=0.7).images[0]  # 生成图像
        >>> image.save("sd3_pag.png")  # 保存生成的图像
        ```py
"""


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的函数
def retrieve_timesteps(
    scheduler,  # 调度器实例
    num_inference_steps: Optional[int] = None,  # 可选的推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备参数
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表
    sigmas: Optional[List[float]] = None,  # 可选的 sigma 列表
    **kwargs,  # 其他可选参数
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理自定义时间步。
    Any kwargs will be supplied to `scheduler.set_timesteps`.
    # 参数说明
    Args:
        # 调度器，用于获取时间步的类
        scheduler (`SchedulerMixin`):
            # 从调度器获取时间步
            The scheduler to get timesteps from.
        # 生成样本时的扩散步骤数量
        num_inference_steps (`int`):
            # 使用预训练模型生成样本时的扩散步骤数。如果使用，`timesteps` 必须为 `None`
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        # 可选参数，指定时间步移动的设备
        device (`str` or `torch.device`, *optional*):
            # 如果为 `None`，则时间步不会被移动
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        # 可选参数，自定义时间步以覆盖调度器的时间步间隔策略
        timesteps (`List[int]`, *optional*):
            # 如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        # 可选参数，自定义 sigmas 以覆盖调度器的时间步间隔策略
        sigmas (`List[float]`, *optional*):
            # 如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    # 返回一个元组，包含时间步调度和推理步骤数量
    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    # 如果同时传递了时间步和 sigmas，抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了时间步
    if timesteps is not None:
        # 检查当前调度器是否支持时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取当前时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数量
        num_inference_steps = len(timesteps)
    # 如果传递了 sigmas
    elif sigmas is not None:
        # 检查当前调度器是否支持 sigmas
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的 sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取当前时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数量
        num_inference_steps = len(timesteps)
    # 如果没有传递时间步或 sigmas
    else:
        # 设置调度器的时间步数量
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器获取当前时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤数量
    return timesteps, num_inference_steps
# 定义一个名为 StableDiffusion3PAGPipeline 的类，继承自多个基类
class StableDiffusion3PAGPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, PAGMixin):
    r"""
    # 文档字符串，描述该类的功能和参数
    [PAG pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) for text-to-image generation
    using Stable Diffusion 3.

    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    # 定义一个模型的 CPU 卸载顺序，指定各组件之间的依赖关系
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    # 定义可选组件的空列表，可能在子类中进行扩展
    _optional_components = []
    # 定义回调张量输入的名称列表，这些输入将在处理过程中被回调
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]
    # 初始化类的构造函数
        def __init__(
            self,
            transformer: SD3Transformer2DModel,  # 输入的 2D 转换器模型
            scheduler: FlowMatchEulerDiscreteScheduler,  # 调度器，用于控制模型的生成过程
            vae: AutoencoderKL,  # 自编码器，用于生成图像
            text_encoder: CLIPTextModelWithProjection,  # 文本编码器，将文本转为向量
            tokenizer: CLIPTokenizer,  # 文本分词器，将文本拆分为标记
            text_encoder_2: CLIPTextModelWithProjection,  # 第二个文本编码器
            tokenizer_2: CLIPTokenizer,  # 第二个文本分词器
            text_encoder_3: T5EncoderModel,  # 第三个文本编码器
            tokenizer_3: T5TokenizerFast,  # 第三个文本分词器
            pag_applied_layers: Union[str, List[str]] = "blocks.1",  # 默认应用于第一个变换层
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 注册各个模块，使其可以在模型中使用
            self.register_modules(
                vae=vae,  # 注册自编码器
                text_encoder=text_encoder,  # 注册文本编码器
                text_encoder_2=text_encoder_2,  # 注册第二个文本编码器
                text_encoder_3=text_encoder_3,  # 注册第三个文本编码器
                tokenizer=tokenizer,  # 注册文本分词器
                tokenizer_2=tokenizer_2,  # 注册第二个文本分词器
                tokenizer_3=tokenizer_3,  # 注册第三个文本分词器
                transformer=transformer,  # 注册转换器
                scheduler=scheduler,  # 注册调度器
            )
            # 根据 VAE 配置计算缩放因子
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
            # 初始化图像处理器，使用计算出的缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 获取分词器的最大长度
            self.tokenizer_max_length = (
                self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
            )
            # 获取转换器的默认样本大小
            self.default_sample_size = (
                self.transformer.config.sample_size
                if hasattr(self, "transformer") and self.transformer is not None
                else 128
            )
    
            # 设置应用于 PAG 的层以及注意力处理器
            self.set_pag_applied_layers(
                pag_applied_layers, pag_attn_processors=(PAGCFGJointAttnProcessor2_0(), PAGJointAttnProcessor2_0())
            )
    
        # 从其他模块复制的函数，用于获取 T5 提示嵌入
        def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,  # 输入的提示，字符串或字符串列表
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            max_sequence_length: int = 256,  # 最大序列长度
            device: Optional[torch.device] = None,  # 指定设备（如 CPU 或 GPU）
            dtype: Optional[torch.dtype] = None,  # 指定数据类型
    ):
        # 如果没有指定设备，则使用实例的执行设备
        device = device or self._execution_device
        # 如果没有指定数据类型，则使用文本编码器的数据类型
        dtype = dtype or self.text_encoder.dtype

        # 如果提示词是字符串，则将其放入列表中，否则保持原样
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 获取提示词的批处理大小
        batch_size = len(prompt)

        # 如果文本编码器未初始化，则返回全零的张量
        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    # 张量的第一维为批处理大小乘以每个提示生成的图像数量
                    batch_size * num_images_per_prompt,
                    # 张量的第二维为最大令牌长度
                    self.tokenizer_max_length,
                    # 张量的第三维为变换器的联合注意力维度
                    self.transformer.config.joint_attention_dim,
                ),
                # 指定设备
                device=device,
                # 指定数据类型
                dtype=dtype,
            )

        # 使用 tokenizer_3 编码提示词，返回张量形式的输入
        text_inputs = self.tokenizer_3(
            prompt,
            # 填充到最大长度
            padding="max_length",
            # 设置最大序列长度
            max_length=max_sequence_length,
            # 截断超出部分
            truncation=True,
            # 添加特殊令牌
            add_special_tokens=True,
            # 返回 PyTorch 张量
            return_tensors="pt",
        )
        # 获取编码后的输入 ID
        text_input_ids = text_inputs.input_ids
        # 获取未截断的输入 ID
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        # 检查未截断的 ID 是否比截断的 ID 更长且不相等
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            # 解码被截断的部分并记录警告
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # 将输入 ID 转移到指定设备并通过文本编码器获得提示嵌入
        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        # 获取文本编码器的数据类型
        dtype = self.text_encoder_3.dtype
        # 将提示嵌入转换为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # 解包提示嵌入的形状信息
        _, seq_len, _ = prompt_embeds.shape

        # 为每个生成的图像复制文本嵌入和注意力掩码，使用适合 mps 的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 重塑张量以符合(batch_size * num_images_per_prompt, seq_len, -1)的形状
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 返回处理后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds 复制
    def _get_clip_prompt_embeds(
        self,
        # 输入的提示词，可以是字符串或字符串列表
        prompt: Union[str, List[str]],
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: int = 1,
        # 可选的设备参数
        device: Optional[torch.device] = None,
        # 可选的跳过参数
        clip_skip: Optional[int] = None,
        # 使用的 clip 模型索引，默认为 0
        clip_model_index: int = 0,
        ):
            # 如果未指定设备，则使用执行环境中的默认设备
            device = device or self._execution_device

            # 初始化 CLIP 模型的分词器和文本编码器列表
            clip_tokenizers = [self.tokenizer, self.tokenizer_2]
            clip_text_encoders = [self.text_encoder, self.text_encoder_2]

            # 根据索引选择当前使用的分词器和文本编码器
            tokenizer = clip_tokenizers[clip_model_index]
            text_encoder = clip_text_encoders[clip_model_index]

            # 将提示文本转换为列表形式（如果输入为字符串则转为单元素列表），确定批处理大小
            prompt = [prompt] if isinstance(prompt, str) else prompt
            batch_size = len(prompt)

            # 使用指定的分词器对提示文本进行编码，设置最大长度和填充方式，返回 PyTorch 张量
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )

            # 获取输入文本的 ID
            text_input_ids = text_inputs.input_ids

            # 获取未截断的文本 ID，并进行比较，如果存在截断则记录警告信息
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer_max_length} tokens: {removed_text}"
                )

            # 使用文本编码器对输入文本进行编码，输出包含隐藏状态
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]

            # 根据 clip_skip 参数选择合适的隐藏状态
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            # 将编码结果转换为指定的数据类型和设备类型
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            # 获取文本编码的序列长度
            _, seq_len, _ = prompt_embeds.shape

            # 使用 MPS 友好的方法复制文本嵌入以生成每个提示的生成结果
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # 复制池化后的文本嵌入以生成每个提示的生成结果
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

            # 返回生成的提示文本嵌入结果
            return prompt_embeds, pooled_prompt_embeds

        # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt 复制过来的
    # 定义编码提示的函数，接受多个参数以支持不同的提示输入
    def encode_prompt(
        # 第一个提示，可以是字符串或字符串列表
        self,
        prompt: Union[str, List[str]],
        # 第二个提示，可以是字符串或字符串列表
        prompt_2: Union[str, List[str]],
        # 第三个提示，可以是字符串或字符串列表
        prompt_3: Union[str, List[str]],
        # 可选参数，指定设备（如 GPU 或 CPU）
        device: Optional[torch.device] = None,
        # 每个提示生成的图像数量，默认是 1
        num_images_per_prompt: int = 1,
        # 是否进行无分类器引导，默认是 True
        do_classifier_free_guidance: bool = True,
        # 可选的负面提示，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 可选的第二个负面提示，可以是字符串或字符串列表
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 可选的第三个负面提示，可以是字符串或字符串列表
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        # 可选的提示嵌入，类型为浮点张量
        prompt_embeds: Optional[torch.FloatTensor] = None,
        # 可选的负面提示嵌入，类型为浮点张量
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 可选的池化提示嵌入，类型为浮点张量
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 可选的负面池化提示嵌入，类型为浮点张量
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 可选参数，指定跳过的 CLIP 层数
        clip_skip: Optional[int] = None,
        # 最大序列长度，默认是 256
        max_sequence_length: int = 256,
        # 可选的 LORA 缩放因子
        lora_scale: Optional[float] = None,
    # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.check_inputs 复制而来
    # 定义输入检查的函数，确保输入的有效性
    def check_inputs(
        self,
        # 第一个提示
        prompt,
        # 第二个提示
        prompt_2,
        # 第三个提示
        prompt_3,
        # 图像高度
        height,
        # 图像宽度
        width,
        # 可选的负面提示
        negative_prompt=None,
        # 可选的第二个负面提示
        negative_prompt_2=None,
        # 可选的第三个负面提示
        negative_prompt_3=None,
        # 可选的提示嵌入
        prompt_embeds=None,
        # 可选的负面提示嵌入
        negative_prompt_embeds=None,
        # 可选的池化提示嵌入
        pooled_prompt_embeds=None,
        # 可选的负面池化提示嵌入
        negative_pooled_prompt_embeds=None,
        # 可选的回调，用于步骤结束时的张量输入
        callback_on_step_end_tensor_inputs=None,
        # 可选的最大序列长度
        max_sequence_length=None,
    # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents 复制而来
    # 定义准备潜在张量的函数，生成潜在张量
    def prepare_latents(
        self,
        # 批次大小
        batch_size,
        # 潜在张量的通道数
        num_channels_latents,
        # 图像高度
        height,
        # 图像宽度
        width,
        # 数据类型
        dtype,
        # 设备类型
        device,
        # 随机数生成器
        generator,
        # 可选的潜在张量
        latents=None,
    ):
        # 如果给定了潜在张量，则将其转移到指定设备和数据类型
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # 计算潜在张量的形状
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        # 检查生成器列表的长度是否与批次大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不匹配，抛出值错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 生成随机潜在张量
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 返回生成的潜在张量
        return latents

    # 定义一个只读属性，用于获取引导缩放因子
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义一个只读属性，用于获取 CLIP 跳过的层数
    @property
    def clip_skip(self):
        return self._clip_skip

    # 定义一个只读属性，检查是否进行无分类器引导
    # 依据 Imagen 论文中的公式定义引导缩放
    # guidance_scale = 1 表示没有进行无分类器引导
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # 定义一个只读属性
    # 定义获取联合注意力参数的方法
        def joint_attention_kwargs(self):
            # 返回类属性 _joint_attention_kwargs 的值
            return self._joint_attention_kwargs
    
        # 定义 num_timesteps 属性
        @property
        def num_timesteps(self):
            # 返回类属性 _num_timesteps 的值
            return self._num_timesteps
    
        # 定义 interrupt 属性
        @property
        def interrupt(self):
            # 返回类属性 _interrupt 的值
            return self._interrupt
    
        # 禁用梯度计算并替换示例文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义类的可调用方法
        def __call__(
            # 定义输入提示，默认为 None
            prompt: Union[str, List[str]] = None,
            # 定义第二个提示，默认为 None
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 定义第三个提示，默认为 None
            prompt_3: Optional[Union[str, List[str]]] = None,
            # 定义图像高度，默认为 None
            height: Optional[int] = None,
            # 定义图像宽度，默认为 None
            width: Optional[int] = None,
            # 定义推理步骤数，默认为 28
            num_inference_steps: int = 28,
            # 定义时间步列表，默认为 None
            timesteps: List[int] = None,
            # 定义引导比例，默认为 7.0
            guidance_scale: float = 7.0,
            # 定义负提示，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 定义第二个负提示，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 定义第三个负提示，默认为 None
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            # 定义每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 定义生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 定义潜在变量，默认为 None
            latents: Optional[torch.FloatTensor] = None,
            # 定义提示嵌入，默认为 None
            prompt_embeds: Optional[torch.FloatTensor] = None,
            # 定义负提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 定义池化提示嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 定义负池化提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 定义输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 定义是否返回字典，默认为 True
            return_dict: bool = True,
            # 定义联合注意力参数，默认为 None
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 定义跳过剪辑的数量，默认为 None
            clip_skip: Optional[int] = None,
            # 定义步骤结束时的回调，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 定义步骤结束时张量输入的回调，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 定义最大序列长度，默认为 256
            max_sequence_length: int = 256,
            # 定义页面缩放比例，默认为 3.0
            pag_scale: float = 3.0,
            # 定义自适应页面缩放比例，默认为 0.0
            pag_adaptive_scale: float = 0.0,
```