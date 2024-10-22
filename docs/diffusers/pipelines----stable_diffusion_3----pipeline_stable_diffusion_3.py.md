# `.\diffusers\pipelines\stable_diffusion_3\pipeline_stable_diffusion_3.py`

```py
# 版权声明，表明版权归 2024 Stability AI 和 HuggingFace Team 所有
# 
# 根据 Apache 许可证，版本 2.0（"许可证"）进行许可；
# 除非遵循许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面同意，否则根据许可证分发的软件是以“现状”基础提供的，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 有关许可证下权限和限制的具体条款，请参见许可证。

# 导入 inspect 模块，用于获取对象的各种信息
import inspect
# 从 typing 模块导入类型提示相关的类型
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入相关的模型和分词器
from transformers import (
    CLIPTextModelWithProjection,  # 导入 CLIP 文本模型
    CLIPTokenizer,                 # 导入 CLIP 分词器
    T5EncoderModel,                # 导入 T5 编码器模型
    T5TokenizerFast,               # 导入快速 T5 分词器
)

# 从本地模块导入图像处理器
from ...image_processor import VaeImageProcessor
# 从本地模块导入加载器混合类
from ...loaders import FromSingleFileMixin, SD3LoraLoaderMixin
# 从本地模块导入自动编码器模型
from ...models.autoencoders import AutoencoderKL
# 从本地模块导入变换器模型
from ...models.transformers import SD3Transformer2DModel
# 从本地模块导入调度器
from ...schedulers import FlowMatchEulerDiscreteScheduler
# 从本地模块导入各种工具函数
from ...utils import (
    USE_PEFT_BACKEND,              # 导入 PEFT 后端使用标志
    is_torch_xla_available,        # 导入检查 XLA 可用性的函数
    logging,                       # 导入日志记录工具
    replace_example_docstring,     # 导入替换示例文档字符串的工具
    scale_lora_layers,             # 导入缩放 LoRA 层的工具
    unscale_lora_layers,           # 导入取消缩放 LoRA 层的工具
)
# 从本地模块导入随机张量生成工具
from ...utils.torch_utils import randn_tensor
# 从本地模块导入扩散管道工具
from ..pipeline_utils import DiffusionPipeline
# 从本地模块导入稳定扩散管道输出类
from .pipeline_output import StableDiffusion3PipelineOutput

# 检查是否可用 torch_xla 库，若可用则导入
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 模型相关功能

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为 True
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标志为 False

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，用于说明如何使用 StableDiffusion3Pipeline
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")  # 将管道移动到 GPU
        >>> prompt = "A cat holding a sign that says hello world"  # 定义生成图像的提示
        >>> image = pipe(prompt).images[0]  # 生成图像并提取第一张
        >>> image.save("sd3.png")  # 保存生成的图像
        ```py
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的函数
def retrieve_timesteps(
    scheduler,                       # 调度器对象，用于管理时间步
    num_inference_steps: Optional[int] = None,  # 可选参数，推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选参数，设备类型
    timesteps: Optional[List[int]] = None,  # 可选参数，自定义时间步列表
    sigmas: Optional[List[float]] = None,    # 可选参数，自定义 sigma 值列表
    **kwargs,                           # 额外的关键字参数
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器中检索时间步。处理
    自定义时间步。所有的关键字参数将传递给 `scheduler.set_timesteps`。
    # 定义参数的说明
        Args:
            scheduler (`SchedulerMixin`):  # 调度器，用于获取时间步
                The scheduler to get timesteps from.
            num_inference_steps (`int`):  # 生成样本时使用的扩散步骤数
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):  # 指定时间步要移动到的设备
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):  # 自定义时间步以覆盖调度器的时间步间隔策略
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):  # 自定义sigma以覆盖调度器的时间步间隔策略
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.
    
        Returns:
            `Tuple[torch.Tensor, int]`: 返回一个元组，第一个元素是调度器的时间步调度，第二个元素是推理步骤的数量。
        """
        # 检查是否同时传入了时间步和sigma，若是则抛出错误
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        # 检查是否传入了时间步
        if timesteps is not None:
            # 检查调度器是否支持自定义时间步
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:  # 不支持则抛出错误
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            # 设置调度器的时间步
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            # 获取当前调度器的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 检查是否传入了sigma
        elif sigmas is not None:
            # 检查调度器是否支持自定义sigma
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:  # 不支持则抛出错误
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            # 设置调度器的sigma
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            # 获取当前调度器的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 如果没有传入时间步和sigma
        else:
            # 使用推理步骤数量设置调度器的时间步
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取当前调度器的时间步
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤数量
        return timesteps, num_inference_steps
# 定义 StableDiffusion3Pipeline 类，继承自 DiffusionPipeline、SD3LoraLoaderMixin 和 FromSingleFileMixin
class StableDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""  # 文档字符串，描述类的参数及其作用
    Args:  # 指明构造函数参数的开始
        transformer ([`SD3Transformer2DModel`]):  # 条件 Transformer（MMDiT）架构，用于去噪编码后的图像潜变量
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):  # 与 transformer 结合使用的调度器，用于去噪图像潜变量
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):  # 用于图像编码和解码的变分自编码器模型
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):  # 特定 CLIP 变体的文本编码器，具有额外的投影层
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):  # 第二个特定 CLIP 变体的文本编码器
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):  # 冻结的文本编码器，使用 T5 变体
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):  # CLIPTokenizer 类的标记器
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):  # 第二个 CLIPTokenizer 类的标记器
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):  # T5Tokenizer 类的标记器
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """  # 文档字符串结束

    # 定义模型的 CPU 离线加载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    # 定义可选组件为空列表
    _optional_components = []
    # 定义回调张量输入的列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    # 初始化方法，定义类的构造函数
    def __init__(
        self,
        transformer: SD3Transformer2DModel,  # 接收的 transformer 参数，类型为 SD3Transformer2DModel
        scheduler: FlowMatchEulerDiscreteScheduler,  # 接收的调度器参数，类型为 FlowMatchEulerDiscreteScheduler
        vae: AutoencoderKL,  # 接收的 VAE 参数，类型为 AutoencoderKL
        text_encoder: CLIPTextModelWithProjection,  # 接收的文本编码器参数，类型为 CLIPTextModelWithProjection
        tokenizer: CLIPTokenizer,  # 接收的标记器参数，类型为 CLIPTokenizer
        text_encoder_2: CLIPTextModelWithProjection,  # 接收的第二个文本编码器参数，类型为 CLIPTextModelWithProjection
        tokenizer_2: CLIPTokenizer,  # 接收的第二个标记器参数，类型为 CLIPTokenizer
        text_encoder_3: T5EncoderModel,  # 接收的第三个文本编码器参数，类型为 T5EncoderModel
        tokenizer_3: T5TokenizerFast,  # 接收的第三个标记器参数，类型为 T5TokenizerFast
    # 初始化父类
        ):
            super().__init__()
    
            # 注册多个模块，包括变分自编码器、文本编码器和调度器
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
            )
            # 计算 VAE 缩放因子，基于配置的块输出通道数
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
            # 创建图像处理器，使用计算得出的 VAE 缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 获取最大标记长度，如果有 tokenizer 的话
            self.tokenizer_max_length = (
                self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
            )
            # 设置默认样本大小，基于 transformer 的配置
            self.default_sample_size = (
                self.transformer.config.sample_size
                if hasattr(self, "transformer") and self.transformer is not None
                else 128
            )
    
        # 定义获取 T5 提示嵌入的方法
        def _get_t5_prompt_embeds(
            self,
            # 提示文本，支持单个字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: int = 1,
            # 最大序列长度限制
            max_sequence_length: int = 256,
            # 设备类型选择
            device: Optional[torch.device] = None,
            # 数据类型选择
            dtype: Optional[torch.dtype] = None,
    # 处理设备和数据类型的设置
        ):
            # 如果未指定设备，则使用默认执行设备
            device = device or self._execution_device
            # 如果未指定数据类型，则使用文本编码器的数据类型
            dtype = dtype or self.text_encoder.dtype
    
            # 将输入的 prompt 转换为列表形式（如果是字符串则包裹在列表中）
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取 prompt 的批处理大小
            batch_size = len(prompt)
    
            # 如果没有第三个文本编码器，则返回零张量
            if self.text_encoder_3 is None:
                return torch.zeros(
                    # 创建一个零张量，形状由批处理大小和其他参数决定
                    (
                        batch_size * num_images_per_prompt,
                        self.tokenizer_max_length,
                        self.transformer.config.joint_attention_dim,
                    ),
                    device=device,
                    dtype=dtype,
                )
    
            # 使用第三个文本编码器对 prompt 进行编码，返回张量格式
            text_inputs = self.tokenizer_3(
                prompt,
                padding="max_length",  # 填充到最大长度
                max_length=max_sequence_length,  # 最大序列长度
                truncation=True,  # 启用截断
                add_special_tokens=True,  # 添加特殊标记
                return_tensors="pt",  # 返回 PyTorch 张量
            )
            # 提取输入 ID
            text_input_ids = text_inputs.input_ids
            # 获取未截断的 ID
            untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids
    
            # 检查未截断 ID 是否长于或等于输入 ID，且不相等
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被截断的部分并记录警告
                removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_sequence_length} tokens: {removed_text}"
                )
    
            # 获取文本编码器对输入 ID 的嵌入
            prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]
    
            # 设置数据类型
            dtype = self.text_encoder_3.dtype
            # 将嵌入转换为指定的数据类型和设备
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
            # 获取嵌入的形状信息
            _, seq_len, _ = prompt_embeds.shape
    
            # 为每个 prompt 生成的图像重复文本嵌入和注意力掩码，采用适合 mps 的方法
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 调整嵌入的形状
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
            # 返回最终的文本嵌入
            return prompt_embeds
    
        # 获取 CLIP 的提示嵌入
        def _get_clip_prompt_embeds(
            self,
            # 接受字符串或字符串列表作为提示
            prompt: Union[str, List[str]],
            # 每个提示生成的图像数量，默认值为 1
            num_images_per_prompt: int = 1,
            # 设备选择
            device: Optional[torch.device] = None,
            # CLIP 跳过的层数（可选）
            clip_skip: Optional[int] = None,
            # CLIP 模型索引，默认值为 0
            clip_model_index: int = 0,
    ):
        # 确定设备，如果没有指定，则使用默认执行设备
        device = device or self._execution_device

        # 定义 CLIP 的两个分词器
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        # 定义 CLIP 的两个文本编码器
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        # 根据给定的模型索引选择对应的分词器
        tokenizer = clip_tokenizers[clip_model_index]
        # 根据给定的模型索引选择对应的文本编码器
        text_encoder = clip_text_encoders[clip_model_index]

        # 如果 prompt 是字符串，则将其放入列表中，否则保持原样
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 计算批处理大小，即 prompt 的数量
        batch_size = len(prompt)

        # 使用选定的分词器对 prompt 进行编码，返回张量
        text_inputs = tokenizer(
            prompt,
            padding="max_length",  # 填充到最大长度
            max_length=self.tokenizer_max_length,  # 最大长度限制
            truncation=True,  # 允许截断
            return_tensors="pt",  # 返回 PyTorch 张量
        )

        # 提取编码后的输入 ID
        text_input_ids = text_inputs.input_ids
        # 对原始 prompt 进行长格式编码以获取未截断的 ID
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        # 检查是否需要警告用户输入被截断
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            # 解码被截断的文本
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            # 记录警告，通知用户被截断的部分
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        # 使用选定的文本编码器生成提示的嵌入表示
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        # 提取 pooled 提示嵌入
        pooled_prompt_embeds = prompt_embeds[0]

        # 根据是否指定 clip_skip 来选择对应的隐藏状态
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        # 将嵌入转换为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # 获取提示嵌入的形状
        _, seq_len, _ = prompt_embeds.shape
        # 对每个 prompt 生成多个图像时复制文本嵌入，使用兼容 mps 的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 重塑为批处理大小与生成图像数量的组合
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 复制 pooled 提示嵌入以适应多个生成的图像
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 重塑为批处理大小与生成图像数量的组合
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # 返回提示嵌入和 pooled 提示嵌入
        return prompt_embeds, pooled_prompt_embeds
    # 定义一个方法用于编码提示信息
        def encode_prompt(
            self,  # 方法参数开始
            prompt: Union[str, List[str]],  # 第一个提示，支持字符串或字符串列表
            prompt_2: Union[str, List[str]],  # 第二个提示，支持字符串或字符串列表
            prompt_3: Union[str, List[str]],  # 第三个提示，支持字符串或字符串列表
            device: Optional[torch.device] = None,  # 可选参数，指定设备
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量，默认1
            do_classifier_free_guidance: bool = True,  # 是否执行无分类器引导，默认是
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
            negative_prompt_2: Optional[Union[str, List[str]]] = None,  # 第二个负面提示
            negative_prompt_3: Optional[Union[str, List[str]]] = None,  # 第三个负面提示
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的负面提示嵌入
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的负面池化提示嵌入
            clip_skip: Optional[int] = None,  # 可选参数，指定跳过的剪辑步骤
            max_sequence_length: int = 256,  # 最大序列长度，默认256
        # 定义一个检查输入的方法
        def check_inputs(
            self,  # 方法参数开始
            prompt,  # 第一个提示
            prompt_2,  # 第二个提示
            prompt_3,  # 第三个提示
            height,  # 图像高度
            width,  # 图像宽度
            negative_prompt=None,  # 可选的负面提示
            negative_prompt_2=None,  # 第二个负面提示
            negative_prompt_3=None,  # 第三个负面提示
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
            pooled_prompt_embeds=None,  # 可选的池化提示嵌入
            negative_pooled_prompt_embeds=None,  # 可选的负面池化提示嵌入
            callback_on_step_end_tensor_inputs=None,  # 可选的回调输入
            max_sequence_length=None,  # 可选的最大序列长度
        # 定义一个准备潜在变量的方法
        def prepare_latents(
            self,  # 方法参数开始
            batch_size,  # 批次大小
            num_channels_latents,  # 潜在变量通道数
            height,  # 图像高度
            width,  # 图像宽度
            dtype,  # 数据类型
            device,  # 设备
            generator,  # 随机生成器
            latents=None,  # 可选的潜在变量
        ):
            if latents is not None:  # 检查是否提供了潜在变量
                return latents.to(device=device, dtype=dtype)  # 转移潜在变量到指定设备和数据类型
    
            shape = (  # 定义潜在变量的形状
                batch_size,  # 批次大小
                num_channels_latents,  # 潜在变量通道数
                int(height) // self.vae_scale_factor,  # 高度缩放
                int(width) // self.vae_scale_factor,  # 宽度缩放
            )
    
            if isinstance(generator, list) and len(generator) != batch_size:  # 检查生成器列表的长度
                raise ValueError(  # 如果长度不匹配则抛出错误
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # 生成潜在变量
    
            return latents  # 返回生成的潜在变量
    
        @property
        def guidance_scale(self):  # 定义一个属性用于获取引导比例
            return self._guidance_scale  # 返回引导比例
    
        @property
        def clip_skip(self):  # 定义一个属性用于获取剪辑跳过值
            return self._clip_skip  # 返回剪辑跳过值
    
        # 这里`guidance_scale`定义类似于Imagen论文中方程(2)的引导权重`w`
        # https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # 表示不执行无分类器引导。
        @property
        def do_classifier_free_guidance(self):  # 定义一个属性判断是否执行无分类器引导
            return self._guidance_scale > 1  # 如果引导比例大于1，则返回True
    
        @property
        def joint_attention_kwargs(self):  # 定义一个属性用于获取联合注意力参数
            return self._joint_attention_kwargs  # 返回联合注意力参数
    
        @property
        def num_timesteps(self):  # 定义一个属性用于获取时间步数
            return self._num_timesteps  # 返回时间步数
    
        @property
        def interrupt(self):  # 定义一个属性用于获取中断状态
            return self._interrupt  # 返回中断状态
    # 禁用梯度计算，以节省内存和加快计算速度
    @torch.no_grad()
    # 装饰器，用于替换示例文档字符串为预定义的文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，接收多个参数用于生成图像
    def __call__(
        # 提示文本，可以是字符串或字符串列表，默认为 None
        self,
        prompt: Union[str, List[str]] = None,
        # 第二个提示文本，类型和默认值与 prompt 相同
        prompt_2: Optional[Union[str, List[str]]] = None,
        # 第三个提示文本，类型和默认值与 prompt 相同
        prompt_3: Optional[Union[str, List[str]]] = None,
        # 输出图像的高度，可选参数，默认为 None
        height: Optional[int] = None,
        # 输出图像的宽度，可选参数，默认为 None
        width: Optional[int] = None,
        # 进行推理的步骤数，默认为 28
        num_inference_steps: int = 28,
        # 指定时间步的列表，默认为 None
        timesteps: List[int] = None,
        # 引导强度，默认为 7.0
        guidance_scale: float = 7.0,
        # 负面提示文本，可以是字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 第二个负面提示文本，类型和默认值与 negative_prompt 相同
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 第三个负面提示文本，类型和默认值与 negative_prompt 相同
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 随机数生成器，可以是单个生成器或生成器列表，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在表示，可选的浮点张量，默认为 None
        latents: Optional[torch.FloatTensor] = None,
        # 提示嵌入，可选的浮点张量，默认为 None
        prompt_embeds: Optional[torch.FloatTensor] = None,
        # 负面提示嵌入，可选的浮点张量，默认为 None
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 聚合的提示嵌入，可选的浮点张量，默认为 None
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 负面聚合提示嵌入，可选的浮点张量，默认为 None
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典，默认为 True
        return_dict: bool = True,
        # 联合注意力的额外参数，默认为 None
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 跳过的剪辑步骤，默认为 None
        clip_skip: Optional[int] = None,
        # 每步结束时的回调函数，默认为 None
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 在每步结束时要回调的张量输入列表，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 最大序列长度，默认为 256
        max_sequence_length: int = 256,
```