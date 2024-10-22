# `.\diffusers\pipelines\controlnet_sd3\pipeline_stable_diffusion_3_controlnet.py`

```py
# 版权声明，指出版权所有者及相关信息
# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# 按照 Apache 2.0 许可证授权使用此文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵守许可证的情况下，才可使用此文件
# you may not use this file except in compliance with the License.
# 可以通过以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非在适用法律下另有规定或以书面形式达成协议，否则按“原样”分发的软件不提供任何明示或暗示的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取特定权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块，用于检查活跃的对象和模块
import inspect
# 导入类型提示相关的模块
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库中导入多个模型和分词器
from transformers import (
    CLIPTextModelWithProjection,  # 导入带有投影的 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 分词器
    T5EncoderModel,  # 导入 T5 编码器模型
    T5TokenizerFast,  # 导入快速 T5 分词器
)

# 从本地模块中导入图像处理和加载器相关的类
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from ...models.autoencoders import AutoencoderKL  # 导入自编码器模型
from ...models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel  # 导入控制网络模型
from ...models.transformers import SD3Transformer2DModel  # 导入 2D 变压器模型
from ...schedulers import FlowMatchEulerDiscreteScheduler  # 导入调度器
from ...utils import (
    USE_PEFT_BACKEND,  # 导入使用 PEFT 后端的标志
    is_torch_xla_available,  # 导入检查 XLA 是否可用的函数
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放 LoRA 层的函数
    unscale_lora_layers,  # 导入反缩放 LoRA 层的函数
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道类
from ..stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput  # 导入扩散管道输出类

# 检查是否可用 XLA 支持
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 模型相关的模块

    XLA_AVAILABLE = True  # 设置 XLA 可用的标志
else:
    XLA_AVAILABLE = False  # 设置 XLA 不可用的标志

# 创建日志记录器，用于当前模块
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用管道
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3ControlNetPipeline  # 导入稳定扩散控制网络管道
        >>> from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel  # 导入控制网络模型
        >>> from diffusers.utils import load_image  # 导入加载图像的工具函数

        >>> controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)  # 加载预训练的控制网络模型

        >>> pipe = StableDiffusion3ControlNetPipeline.from_pretrained(  # 从预训练模型创建管道
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")  # 将管道移动到 GPU
        >>> control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")  # 加载控制图像
        >>> prompt = "A girl holding a sign that says InstantX"  # 设置生成图像的提示
        >>> image = pipe(prompt, control_image=control_image, controlnet_conditioning_scale=0.7).images[0]  # 生成图像
        >>> image.save("sd3.png")  # 保存生成的图像
        ```py
"""

# 从稳定扩散管道中复制的函数，用于检索时间步
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(  # 定义函数以检索时间步
    scheduler,  # 调度器对象
    num_inference_steps: Optional[int] = None,  # 可选的推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备参数
    # 可选参数，用于指定时间步的列表
        timesteps: Optional[List[int]] = None,
        # 可选参数，用于指定sigma值的列表
        sigmas: Optional[List[float]] = None,
        # 接受其他任意关键字参数
        **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步。处理
    自定义时间步。任何额外参数将被传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            获取时间步的调度器。
        num_inference_steps (`int`):
            用于生成样本的扩散步骤数。如果使用，则 `timesteps`
            必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步应移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义时间步。如果传入 `timesteps`，
            `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义 sigma。如果传入 `sigmas`，
            `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是来自调度器的时间步计划，
        第二个元素是推理步骤的数量。
    """
    # 检查是否同时传入了 timesteps 和 sigmas
    if timesteps is not None and sigmas is not None:
        # 如果同时传入，抛出值错误，提示只能选择一个自定义值
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查是否传入了 timesteps
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受 timesteps 参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出值错误，提示使用的调度器不支持自定义时间步
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传入 timesteps 和 device，以及其他关键字参数
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 检查是否传入了 sigmas
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigmas 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出值错误，提示使用的调度器不支持自定义 sigma
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 set_timesteps 方法，传入 sigmas 和 device，以及其他关键字参数
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算时间步的数量
        num_inference_steps = len(timesteps)
    # 如果都没有传入，使用 num_inference_steps 作为参数调用 set_timesteps
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤的数量
    return timesteps, num_inference_steps


class StableDiffusion3ControlNetPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    # 函数的参数说明部分
    Args:
        transformer ([`SD3Transformer2DModel`]):
            # 用于去噪编码图像潜变量的条件变换器（MMDiT）架构。
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            # 与 `transformer` 结合使用的调度器，用于去噪编码图像潜变量。
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            # 用于编码和解码图像到潜在表示的变分自编码器（VAE）模型。
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            # 特定于 CLIP 的模型，增加了一个投影层，该层的初始化为与 `hidden_size` 维度相同的对角矩阵。
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            # 特定于 CLIP 的第二个模型。
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            # 冻结的文本编码器，使用 T5 模型的特定变体。
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            # CLIPTokenizer 类的分词器。
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            # 第二个 CLIPTokenizer 类的分词器。
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            # T5Tokenizer 类的分词器。
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        controlnet ([`SD3ControlNetModel`] or `List[SD3ControlNetModel]` or [`SD3MultiControlNetModel`]):
            # 在去噪过程中为 `unet` 提供额外的条件。如果将多个 ControlNet 设置为列表，则每个 ControlNet 的输出将相加以创建一个组合的附加条件。
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    # 定义可选组件的列表，初始化为空
    _optional_components = []
    # 定义回调张量输入的名称列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]
    # 初始化类的构造函数，设置各种模型和调度器
        def __init__(
            # 接收的变换器模型
            self,
            transformer: SD3Transformer2DModel,
            # 调度器，用于控制生成过程
            scheduler: FlowMatchEulerDiscreteScheduler,
            # 自动编码器模型
            vae: AutoencoderKL,
            # 文本编码器模型
            text_encoder: CLIPTextModelWithProjection,
            # 文本标记器
            tokenizer: CLIPTokenizer,
            # 第二个文本编码器模型
            text_encoder_2: CLIPTextModelWithProjection,
            # 第二个文本标记器
            tokenizer_2: CLIPTokenizer,
            # 第三个文本编码器模型
            text_encoder_3: T5EncoderModel,
            # 第三个文本标记器
            tokenizer_3: T5TokenizerFast,
            # 控制网络模型，可以是单个或多个
            controlnet: Union[
                SD3ControlNetModel, List[SD3ControlNetModel], Tuple[SD3ControlNetModel], SD3MultiControlNetModel
            ],
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 注册各种模块以便后续使用
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                text_encoder_3=text_encoder_3,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                tokenizer_3=tokenizer_3,
                transformer=transformer,
                scheduler=scheduler,
                controlnet=controlnet,
            )
            # 计算 VAE 的缩放因子，根据配置的块输出通道数
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
            # 创建图像处理器，使用计算的缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 获取标记器的最大长度，如果不存在则默认值为 77
            self.tokenizer_max_length = (
                self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
            )
            # 获取变换器的默认样本大小，如果不存在则默认值为 128
            self.default_sample_size = (
                self.transformer.config.sample_size
                if hasattr(self, "transformer") and self.transformer is not None
                else 128
            )
    
        # 从现有管道复制的函数，用于获取 T5 的提示嵌入
        def _get_t5_prompt_embeds(
            self,
            # 提示文本，可以是字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: int = 1,
            # 最大序列长度
            max_sequence_length: int = 256,
            # 设备，默认为 None
            device: Optional[torch.device] = None,
            # 数据类型，默认为 None
            dtype: Optional[torch.dtype] = None,
    # 设备默认为执行设备，如果未指定则使用默认设备
        ):
            device = device or self._execution_device
            # 数据类型默认为文本编码器的数据类型，如果未指定则使用默认数据类型
            dtype = dtype or self.text_encoder.dtype
    
            # 如果输入是字符串，则将其转换为列表形式
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取批次大小，即提示的数量
            batch_size = len(prompt)
    
            # 如果文本编码器未定义，返回一个零张量作为占位符
            if self.text_encoder_3 is None:
                return torch.zeros(
                    (
                        batch_size * num_images_per_prompt,  # 每个提示的图像数量乘以批次大小
                        self.tokenizer_max_length,            # 最大的序列长度
                        self.transformer.config.joint_attention_dim,  # 联合注意力维度
                    ),
                    device=device,  # 指定设备
                    dtype=dtype,    # 指定数据类型
                )
    
            # 使用 tokenizer 对提示进行编码，返回张量格式
            text_inputs = self.tokenizer_3(
                prompt,
                padding="max_length",                  # 填充到最大长度
                max_length=max_sequence_length,        # 最大序列长度限制
                truncation=True,                       # 启用截断
                add_special_tokens=True,               # 添加特殊标记
                return_tensors="pt",                  # 返回 PyTorch 张量
            )
            # 获取编码后的输入 ID
            text_input_ids = text_inputs.input_ids
            # 获取未截断的输入 ID
            untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids
    
            # 检查未截断的 ID 是否比截断的 ID 更长，并且两者是否不相等
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被截断的文本并记录警告
                removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_sequence_length} tokens: {removed_text}"  # 显示被截断的文本
                )
    
            # 获取文本嵌入
            prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]
    
            # 确保嵌入的数据类型与文本编码器一致
            dtype = self.text_encoder_3.dtype
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
            # 获取嵌入的形状信息
            _, seq_len, _ = prompt_embeds.shape
    
            # 为每个提示生成重复文本嵌入和注意力掩码，使用适合 mps 的方法
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 调整张量的形状，以适应批次大小和图像数量
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
            # 返回处理后的嵌入
            return prompt_embeds
    
        # 从稳定扩散管道复制的函数，用于获取 CLIP 提示嵌入
        def _get_clip_prompt_embeds(
            self,
            prompt: Union[str, List[str]],  # 输入提示，支持字符串或字符串列表
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            device: Optional[torch.device] = None,  # 设备选择，默认为 None
            clip_skip: Optional[int] = None,  # 可选的跳过步骤
            clip_model_index: int = 0,        # CLIP 模型的索引
    ):
        # 如果未指定设备，则使用默认执行设备
        device = device or self._execution_device

        # 定义 CLIP 模型的两个分词器
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        # 定义 CLIP 模型的两个文本编码器
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        # 根据索引选择对应的分词器
        tokenizer = clip_tokenizers[clip_model_index]
        # 根据索引选择对应的文本编码器
        text_encoder = clip_text_encoders[clip_model_index]

        # 将提示转换为列表，如果是字符串则单独处理
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 计算提示的批处理大小
        batch_size = len(prompt)

        # 对提示进行编码，返回的张量将填充到最大长度
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # 提取编码后的输入 ID
        text_input_ids = text_inputs.input_ids
        # 不截断的输入 ID，使用最长填充方式
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        # 检查是否有输入被截断
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            # 解码并记录被截断的文本
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        # 使用文本编码器生成提示的嵌入，输出隐藏状态
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        # 获取池化后的提示嵌入
        pooled_prompt_embeds = prompt_embeds[0]

        # 根据 clip_skip 的值选择不同的隐藏状态
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        # 将提示嵌入转换为指定的数据类型并放置到正确的设备上
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # 获取提示嵌入的形状信息
        _, seq_len, _ = prompt_embeds.shape
        # 为每个提示的生成重复文本嵌入，使用适合 MPS 的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 重塑为 (批处理大小 * 每个提示生成的图像数, 序列长度, 嵌入维度)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 为池化的提示嵌入重复，调整形状
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # 返回生成的提示嵌入和池化的提示嵌入
        return prompt_embeds, pooled_prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt 复制的代码
    # 定义编码提示的函数，接受多个提示和相关参数
    def encode_prompt(
        self,  # self 参数，表示类的实例
        prompt: Union[str, List[str]],  # 第一个提示，可以是字符串或字符串列表
        prompt_2: Union[str, List[str]],  # 第二个提示，可以是字符串或字符串列表
        prompt_3: Union[str, List[str]],  # 第三个提示，可以是字符串或字符串列表
        device: Optional[torch.device] = None,  # 可选参数，指定设备（如 CPU 或 GPU）
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量，默认为 1
        do_classifier_free_guidance: bool = True,  # 是否执行无分类器引导，默认为 True
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选负提示，可以是字符串或字符串列表
        negative_prompt_2: Optional[Union[str, List[str]]] = None,  # 第二个可选负提示
        negative_prompt_3: Optional[Union[str, List[str]]] = None,  # 第三个可选负提示
        prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选提示嵌入，类型为浮点张量
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选负提示嵌入
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选池化的提示嵌入
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选池化的负提示嵌入
        clip_skip: Optional[int] = None,  # 可选参数，指定剪切的层数
        max_sequence_length: int = 256,  # 最大序列长度，默认为 256
        lora_scale: Optional[float] = None,  # 可选参数，指定 LoRA 的缩放因子
    # 定义检查输入的函数，验证输入参数的有效性
    def check_inputs(
        self,  # self 参数，表示类的实例
        prompt,  # 第一个提示
        prompt_2,  # 第二个提示
        prompt_3,  # 第三个提示
        height,  # 高度参数
        width,  # 宽度参数
        negative_prompt=None,  # 可选负提示
        negative_prompt_2=None,  # 第二个可选负提示
        negative_prompt_3=None,  # 第三个可选负提示
        prompt_embeds=None,  # 可选提示嵌入
        negative_prompt_embeds=None,  # 可选负提示嵌入
        pooled_prompt_embeds=None,  # 可选池化的提示嵌入
        negative_pooled_prompt_embeds=None,  # 可选池化的负提示嵌入
        callback_on_step_end_tensor_inputs=None,  # 可选参数，用于回调
        max_sequence_length=None,  # 可选最大序列长度
    # 定义准备潜在空间的函数，用于生成潜在表示
    def prepare_latents(
        self,  # self 参数，表示类的实例
        batch_size,  # 批次大小
        num_channels_latents,  # 潜在张量的通道数
        height,  # 高度
        width,  # 宽度
        dtype,  # 数据类型
        device,  # 设备类型
        generator,  # 随机数生成器
        latents=None,  # 可选潜在张量
    ):
        # 如果提供了潜在张量，则将其转移到指定设备和数据类型
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # 定义潜在张量的形状，包括批次大小和通道数
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,  # 根据 VAE 缩放因子调整高度
            int(width) // self.vae_scale_factor,  # 根据 VAE 缩放因子调整宽度
        )

        # 检查生成器的类型，确保其与批次大小一致
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 生成随机潜在张量
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 返回生成的潜在张量
        return latents

    # 定义准备图像的函数，接受图像及相关参数
    def prepare_image(
        self,  # self 参数，表示类的实例
        image,  # 输入的图像
        width,  # 图像的宽度
        height,  # 图像的高度
        batch_size,  # 批次大小
        num_images_per_prompt,  # 每个提示生成的图像数量
        device,  # 设备类型
        dtype,  # 数据类型
        do_classifier_free_guidance=False,  # 是否执行无分类器引导，默认为 False
        guess_mode=False,  # 是否启用猜测模式，默认为 False
    ):
        # 检查输入的图像是否为 PyTorch 张量
        if isinstance(image, torch.Tensor):
            # 如果是张量，则不进行处理，直接跳过
            pass
        else:
            # 如果不是张量，则使用图像处理器进行预处理，调整高度和宽度
            image = self.image_processor.preprocess(image, height=height, width=width)

        # 获取图像批次的大小
        image_batch_size = image.shape[0]

        # 如果图像批次大小为 1，则根据批次大小重复图像
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # 否则，图像批次大小与提示批次大小相同
            repeat_by = num_images_per_prompt

        # 根据 repeat_by 在第 0 维度重复图像
        image = image.repeat_interleave(repeat_by, dim=0)

        # 将图像移动到指定设备并转换数据类型
        image = image.to(device=device, dtype=dtype)

        # 如果启用分类器自由引导且不处于猜测模式，则对图像进行复制
        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        # 返回处理后的图像
        return image

    @property
    # 获取指导缩放比例
    def guidance_scale(self):
        return self._guidance_scale

    @property
    # 获取剪切跳过的参数
    def clip_skip(self):
        return self._clip_skip

    # `guidance_scale` 定义类似于 Imagen 论文中方程 (2) 的指导权重 `w`
    # `guidance_scale = 1` 表示不进行分类器自由引导。
    @property
    # 检查是否启用分类器自由引导
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    # 获取联合注意力的参数
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    # 获取时间步数的参数
    def num_timesteps(self):
        return self._num_timesteps

    @property
    # 获取中断标志
    def interrupt(self):
        return self._interrupt

    # 禁用梯度计算的装饰器
    @torch.no_grad()
    # 用于替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的 __call__ 方法
        def __call__(
            # 提示文本，支持字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个提示文本，支持字符串或字符串列表，默认为 None
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 第三个提示文本，支持字符串或字符串列表，默认为 None
            prompt_3: Optional[Union[str, List[str]]] = None,
            # 输出图像的高度，默认为 None
            height: Optional[int] = None,
            # 输出图像的宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤的数量，默认为 28
            num_inference_steps: int = 28,
            # 自定义时间步的列表，默认为 None
            timesteps: List[int] = None,
            # 指导尺度，默认为 7.0
            guidance_scale: float = 7.0,
            # 控制引导开始的尺度，支持浮点数或浮点数列表，默认为 0.0
            control_guidance_start: Union[float, List[float]] = 0.0,
            # 控制引导结束的尺度，支持浮点数或浮点数列表，默认为 1.0
            control_guidance_end: Union[float, List[float]] = 1.0,
            # 控制图像的输入，默认为 None
            control_image: PipelineImageInput = None,
            # 控制网络条件缩放，支持浮点数或浮点数列表，默认为 1.0
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # 控制网络池化投影，默认为 None
            controlnet_pooled_projections: Optional[torch.FloatTensor] = None,
            # 负提示文本，支持字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负提示文本，支持字符串或字符串列表，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 第三个负提示文本，支持字符串或字符串列表，默认为 None
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 随机数生成器，支持生成器或生成器列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在向量，默认为 None
            latents: Optional[torch.FloatTensor] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.FloatTensor] = None,
            # 负提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 池化提示嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 负池化提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的输出，默认为 True
            return_dict: bool = True,
            # 联合注意力的参数，默认为 None
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 跳过的剪辑层数，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 步骤结束时的张量输入回调，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 最大序列长度，默认为 256
            max_sequence_length: int = 256,
```