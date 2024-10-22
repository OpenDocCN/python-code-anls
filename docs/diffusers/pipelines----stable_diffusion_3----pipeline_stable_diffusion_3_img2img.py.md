# `.\diffusers\pipelines\stable_diffusion_3\pipeline_stable_diffusion_3_img2img.py`

```py
# 版权声明，指定版权所有者及保留权利
# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# 在 Apache License, Version 2.0 下授权（“许可证”）；
# 除非遵循许可证的规定，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，
# 否则根据许可证分发的软件是以“原样”基础提供的，
# 不提供任何明示或暗示的保证或条件。
# 有关许可证规定的权限和限制，请参见许可证。

# 导入 inspect 模块，用于获取对象的各种信息
import inspect
# 从 typing 模块导入所需的类型注解
from typing import Callable, Dict, List, Optional, Union

# 导入 PIL.Image 库，用于图像处理
import PIL.Image
# 导入 PyTorch 库，深度学习框架
import torch
# 从 transformers 库导入所需的模型和分词器
from transformers import (
    CLIPTextModelWithProjection,  # CLIP 文本模型
    CLIPTokenizer,                 # CLIP 分词器
    T5EncoderModel,                # T5 编码器模型
    T5TokenizerFast,               # T5 快速分词器
)

# 从本地模块中导入图像处理和模型相关类
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import SD3LoraLoaderMixin  # 导入 SD3 Lora 加载器混合类
from ...models.autoencoders import AutoencoderKL  # 导入自动编码器模型
from ...models.transformers import SD3Transformer2DModel  # 导入 SD3 2D 转换模型
from ...schedulers import FlowMatchEulerDiscreteScheduler  # 导入调度器
from ...utils import (
    USE_PEFT_BACKEND,              # 导入是否使用 PEFT 后端的标识
    is_torch_xla_available,        # 导入检查是否可用 Torch XLA 的函数
    logging,                       # 导入日志模块
    replace_example_docstring,     # 导入替换示例文档字符串的函数
    scale_lora_layers,             # 导入缩放 Lora 层的函数
    unscale_lora_layers,           # 导入取消缩放 Lora 层的函数
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道类
from .pipeline_output import StableDiffusion3PipelineOutput  # 导入稳定扩散 3 的管道输出类


# 检查是否可用 Torch XLA，适用于 TPU
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 核心模块

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为 True
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标志为 False


# 创建日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该模块
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch

        >>> from diffusers import AutoPipelineForImage2Image
        >>> from diffusers.utils import load_image

        >>> device = "cuda"  # 设置设备为 CUDA
        >>> model_id_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"  # 指定模型路径
        >>> pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, torch_dtype=torch.float16)  # 从预训练模型加载管道
        >>> pipe = pipe.to(device)  # 将管道移动到指定设备

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"  # 指定输入图像 URL
        >>> init_image = load_image(url).resize((1024, 1024))  # 加载并调整图像大小

        >>> prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"  # 设置生成的提示语

        >>> images = pipe(prompt=prompt, image=init_image, strength=0.95, guidance_scale=7.5).images[0]  # 生成图像
        ```py
"""


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor,  # 输入的编码器输出，类型为张量
    generator: Optional[torch.Generator] = None,  # 随机数生成器，默认为 None
    sample_mode: str = "sample"  # 采样模式，默认为 "sample"
):
    # 如果 encoder_output 具有 latent_dist 属性且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中进行采样，并返回结果
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否具有 "latent_dist" 属性，并且 sample_mode 为 "argmax"
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            # 返回 encoder_output 中 latent_dist 的众数
            return encoder_output.latent_dist.mode()
        # 检查 encoder_output 是否具有 "latents" 属性
        elif hasattr(encoder_output, "latents"):
            # 返回 encoder_output 中的 latents 属性
            return encoder_output.latents
        # 如果以上条件都不满足，抛出属性错误
        else:
            raise AttributeError("Could not access latents of provided encoder_output")
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的代码
def retrieve_timesteps(
    scheduler,  # 调度器，用于获取时间步
    num_inference_steps: Optional[int] = None,  # 推断步骤数量，默认为 None
    device: Optional[Union[str, torch.device]] = None,  # 设备类型，默认为 None
    timesteps: Optional[List[int]] = None,  # 自定义时间步，默认为 None
    sigmas: Optional[List[float]] = None,  # 自定义 sigma 值，默认为 None
    **kwargs,  # 其他关键字参数
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步。处理
    自定义时间步。所有关键字参数将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数。如果使用，则 `timesteps`
            必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            要将时间步移动到的设备。如果为 `None`，则时间步不移动。
        timesteps (`List[int]`, *可选*):
            用于覆盖调度器的时间步间距策略的自定义时间步。如果传递 `timesteps`，
            `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            用于覆盖调度器的时间步间距策略的自定义 sigma。如果传递 `sigmas`，
            `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是来自调度器的时间步调度，
        第二个元素是推断步骤的数量。
    """
    # 检查是否同时传入了自定义时间步和 sigma，若是则抛出异常
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    
    # 如果传入了自定义时间步
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义时间步，抛出异常
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推断步骤数量
        num_inference_steps = len(timesteps)
    
    # 如果传入了自定义 sigma
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义 sigma，抛出异常
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推断步骤数量
        num_inference_steps = len(timesteps)
    else:  # 如果不满足前面的条件，则执行以下代码
        # 设置调度器的时间步数，传入推理步数和设备信息，以及其他可选参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器的时间步数并赋值给变量 timesteps
        timesteps = scheduler.timesteps
    # 返回时间步数和推理步数
    return timesteps, num_inference_steps
# 定义一个名为 StableDiffusion3Img2ImgPipeline 的类，继承自 DiffusionPipeline
class StableDiffusion3Img2ImgPipeline(DiffusionPipeline):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) 结构，用于对编码后的图像潜变量进行去噪。
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            一个调度器，结合 `transformer` 用于对编码后的图像潜变量进行去噪。
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于在潜在表示和图像之间进行编码和解码。
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            特别是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体，
            并添加了一个投影层，该层使用对角矩阵初始化，维度为 `hidden_size`。
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            特别是
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            变体。
        text_encoder_3 ([`T5EncoderModel`]):
            冻结的文本编码器。Stable Diffusion 3 使用
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)，特别是
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) 变体。
        tokenizer (`CLIPTokenizer`):
            类
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 的标记器。
        tokenizer_2 (`CLIPTokenizer`):
            类
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 的第二个标记器。
        tokenizer_3 (`T5TokenizerFast`):
            类
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer) 的标记器。
    """

    # 定义一个字符串，表示模型组件的加载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    # 定义一个可选组件列表，初始为空
    _optional_components = []
    # 定义一个回调张量输入列表，包含潜变量和提示嵌入等
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    # 初始化方法，定义所需的组件
    def __init__(
        self,
        # 定义 transformer 参数，类型为 SD3Transformer2DModel
        transformer: SD3Transformer2DModel,
        # 定义 scheduler 参数，类型为 FlowMatchEulerDiscreteScheduler
        scheduler: FlowMatchEulerDiscreteScheduler,
        # 定义 vae 参数，类型为 AutoencoderKL
        vae: AutoencoderKL,
        # 定义 text_encoder 参数，类型为 CLIPTextModelWithProjection
        text_encoder: CLIPTextModelWithProjection,
        # 定义 tokenizer 参数，类型为 CLIPTokenizer
        tokenizer: CLIPTokenizer,
        # 定义第二个 text_encoder 参数，类型为 CLIPTextModelWithProjection
        text_encoder_2: CLIPTextModelWithProjection,
        # 定义第二个 tokenizer 参数，类型为 CLIPTokenizer
        tokenizer_2: CLIPTokenizer,
        # 定义第三个 text_encoder 参数，类型为 T5EncoderModel
        text_encoder_3: T5EncoderModel,
        # 定义第三个 tokenizer 参数，类型为 T5TokenizerFast
        tokenizer_3: T5TokenizerFast,
    ):
        # 调用父类的构造函数进行初始化
        super().__init__()

        # 注册多个模块，方便在后续操作中使用
        self.register_modules(
            # 注册变分自编码器模块
            vae=vae,
            # 注册文本编码器模块
            text_encoder=text_encoder,
            # 注册第二个文本编码器模块
            text_encoder_2=text_encoder_2,
            # 注册第三个文本编码器模块
            text_encoder_3=text_encoder_3,
            # 注册标记器模块
            tokenizer=tokenizer,
            # 注册第二个标记器模块
            tokenizer_2=tokenizer_2,
            # 注册第三个标记器模块
            tokenizer_3=tokenizer_3,
            # 注册变换器模块
            transformer=transformer,
            # 注册调度器模块
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子，基于块输出通道的数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，传入 VAE 的缩放因子和潜在通道数
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.vae.config.latent_channels
        )
        # 获取标记器的最大长度
        self.tokenizer_max_length = self.tokenizer.model_max_length
        # 获取变换器的默认样本大小
        self.default_sample_size = self.transformer.config.sample_size

    # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds 复制的方法
    def _get_t5_prompt_embeds(
        self,
        # 输入的提示，可以是字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: int = 1,
        # 最大序列长度
        max_sequence_length: int = 256,
        # 设备类型，可选
        device: Optional[torch.device] = None,
        # 数据类型，可选
        dtype: Optional[torch.dtype] = None,
    # 方法的定义，接受多个参数
        ):
            # 如果没有指定设备，则使用类中定义的执行设备
            device = device or self._execution_device
            # 如果没有指定数据类型，则使用文本编码器的数据类型
            dtype = dtype or self.text_encoder.dtype
    
            # 如果提示为字符串，则转换为列表形式；否则保持原样
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取提示的批处理大小，即提示的数量
            batch_size = len(prompt)
    
            # 如果没有文本编码器 3，则返回一个全零的张量
            if self.text_encoder_3 is None:
                return torch.zeros(
                    # 返回形状为 (批处理大小 * 每个提示的图像数量, 最大序列长度, 联合注意力维度)
                    (
                        batch_size * num_images_per_prompt,
                        self.tokenizer_max_length,
                        self.transformer.config.joint_attention_dim,
                    ),
                    # 指定设备和数据类型
                    device=device,
                    dtype=dtype,
                )
    
            # 使用文本编码器 3 对提示进行编码，返回张量格式
            text_inputs = self.tokenizer_3(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            # 获取输入的 ID
            text_input_ids = text_inputs.input_ids
            # 获取未截断的 ID，用于检测是否有内容被截断
            untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids
    
            # 检查是否未截断 ID 的长度大于或等于输入 ID 的长度，并且两者不相等
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被截断的文本并发出警告
                removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_sequence_length} tokens: {removed_text}"
                )
    
            # 获取文本输入的嵌入
            prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]
    
            # 更新数据类型为文本编码器 3 的数据类型
            dtype = self.text_encoder_3.dtype
            # 将嵌入转换为指定的数据类型和设备
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
            # 获取嵌入的形状信息
            _, seq_len, _ = prompt_embeds.shape
    
            # 为每个提示生成的图像复制文本嵌入
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 调整嵌入形状，以便于处理
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
            # 返回最终的文本嵌入
            return prompt_embeds
    
        # 从 StableDiffusion3Pipeline 类复制的方法，用于获取 CLIP 提示嵌入
        def _get_clip_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            device: Optional[torch.device] = None,
            clip_skip: Optional[int] = None,
            clip_model_index: int = 0,
    # 设备设置，如果未指定，则使用默认执行设备
        ):
            device = device or self._execution_device
    
            # 定义 CLIP 使用的分词器
            clip_tokenizers = [self.tokenizer, self.tokenizer_2]
            # 定义 CLIP 使用的文本编码器
            clip_text_encoders = [self.text_encoder, self.text_encoder_2]
    
            # 根据给定的模型索引选择分词器
            tokenizer = clip_tokenizers[clip_model_index]
            # 根据给定的模型索引选择文本编码器
            text_encoder = clip_text_encoders[clip_model_index]
    
            # 如果 prompt 是字符串，则转为列表形式
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取 prompt 的批处理大小
            batch_size = len(prompt)
    
            # 使用选择的分词器对 prompt 进行编码
            text_inputs = tokenizer(
                prompt,
                padding="max_length",  # 填充到最大长度
                max_length=self.tokenizer_max_length,  # 最大长度限制
                truncation=True,  # 允许截断
                return_tensors="pt",  # 返回 PyTorch 张量
            )
    
            # 提取编码后的输入 ID
            text_input_ids = text_inputs.input_ids
            # 进行最长填充以获取未截断的 ID
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            # 检查未截断的 ID 是否比当前输入 ID 更长且不相等
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被截断的部分，并记录警告
                removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer_max_length} tokens: {removed_text}"
                )
            # 使用文本编码器生成 prompt 的嵌入
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            # 获取池化后的 prompt 嵌入
            pooled_prompt_embeds = prompt_embeds[0]
    
            # 判断是否跳过某些隐藏状态
            if clip_skip is None:
                # 使用倒数第二个隐藏状态作为嵌入
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # 根据 clip_skip 使用相应的隐藏状态
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
    
            # 将嵌入转换为所需的数据类型和设备
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
    
            # 获取嵌入的形状
            _, seq_len, _ = prompt_embeds.shape
            # 针对每个 prompt 复制文本嵌入，使用适应 MPS 的方法
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 重新调整嵌入的形状
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
            # 复制池化后的嵌入
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 重新调整池化嵌入的形状
            pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    
            # 返回最终的 prompt 嵌入和池化后的嵌入
            return prompt_embeds, pooled_prompt_embeds
    
        # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt 复制的内容
    # 定义一个编码提示的函数，接收多个参数
        def encode_prompt(
            # 第一个提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]],
            # 第二个提示，可以是字符串或字符串列表
            prompt_2: Union[str, List[str]],
            # 第三个提示，可以是字符串或字符串列表
            prompt_3: Union[str, List[str]],
            # 设备选项，默认为 None
            device: Optional[torch.device] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 是否进行分类器自由引导，默认为 True
            do_classifier_free_guidance: bool = True,
            # 负提示，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负提示，可以是字符串或字符串列表，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 第三个负提示，可以是字符串或字符串列表，默认为 None
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.FloatTensor] = None,
            # 负提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 池化后的提示嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 池化后的负提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 可选的跳过参数，默认为 None
            clip_skip: Optional[int] = None,
            # 最大序列长度，默认为 256
            max_sequence_length: int = 256,
            # 可选的 Lora 缩放参数，默认为 None
            lora_scale: Optional[float] = None,
        # 定义一个检查输入的函数，接收多个参数
        def check_inputs(
            # 第一个提示
            self,
            prompt,
            # 第二个提示
            prompt_2,
            # 第三个提示
            prompt_3,
            # 强度参数
            strength,
            # 负提示，默认为 None
            negative_prompt=None,
            # 第二个负提示，默认为 None
            negative_prompt_2=None,
            # 第三个负提示，默认为 None
            negative_prompt_3=None,
            # 提示嵌入，默认为 None
            prompt_embeds=None,
            # 负提示嵌入，默认为 None
            negative_prompt_embeds=None,
            # 池化后的提示嵌入，默认为 None
            pooled_prompt_embeds=None,
            # 池化后的负提示嵌入，默认为 None
            negative_pooled_prompt_embeds=None,
            # 步骤结束时的回调输入，默认为 None
            callback_on_step_end_tensor_inputs=None,
            # 最大序列长度，默认为 None
            max_sequence_length=None,
        # 定义获取时间步的函数，接收推理步骤数、强度和设备参数
        def get_timesteps(self, num_inference_steps, strength, device):
            # 计算初始化时间步的原始值
            init_timestep = min(num_inference_steps * strength, num_inference_steps)
    
            # 计算开始的时间步
            t_start = int(max(num_inference_steps - init_timestep, 0))
            # 从调度器获取时间步
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器具有设置开始索引的属性，设置开始索引
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步和剩余的推理步骤数
            return timesteps, num_inference_steps - t_start
    # 准备潜在向量，用于图像生成的前处理
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        # 检查输入的图像类型是否为 torch.Tensor, PIL.Image.Image 或列表
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            # 抛出类型错误，提示用户输入的图像类型不正确
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # 将图像转换为指定设备和数据类型
        image = image.to(device=device, dtype=dtype)

        # 计算有效批次大小
        batch_size = batch_size * num_images_per_prompt
        # 如果图像的通道数与 VAE 的潜在通道数相同，则初始化潜在向量为图像
        if image.shape[1] == self.vae.config.latent_channels:
            init_latents = image

        else:
            # 如果生成器是列表且其长度与批次大小不符，抛出错误
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                # 对每个图像生成潜在向量，并将结果合并成一个张量
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                # 在第0维上拼接所有潜在向量
                init_latents = torch.cat(init_latents, dim=0)
            else:
                # 对单个图像生成潜在向量
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            # 根据 VAE 配置调整潜在向量的值
            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 如果要求的批次大小大于初始化的潜在向量数量且可以整除
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # 为批次大小扩展初始化潜在向量
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            # 通过复制初始化潜在向量来增加批次大小
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        # 如果要求的批次大小大于初始化的潜在向量数量且不能整除
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            # 抛出错误，提示无法复制图像以满足批次大小
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            # 确保潜在向量为二维，方便后续处理
            init_latents = torch.cat([init_latents], dim=0)

        # 获取潜在向量的形状
        shape = init_latents.shape
        # 生成与潜在向量形状相同的随机噪声张量
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 获取潜在向量，通过调度器缩放噪声
        init_latents = self.scheduler.scale_noise(init_latents, timestep, noise)
        # 将潜在向量转换为指定设备和数据类型
        latents = init_latents.to(device=device, dtype=dtype)

        # 返回处理后的潜在向量
        return latents

    # 返回当前的指导比例
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 返回当前的剪辑跳过值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 判断是否进行无分类器引导，依据指导比例
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # 返回当前的时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    # 定义一个方法以返回中断状态
        def interrupt(self):
            # 返回中断标志的值
            return self._interrupt
    
        # 禁用梯度计算以节省内存和计算
        @torch.no_grad()
        # 替换文档字符串以提供示例文档
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义可调用方法，处理各种输入参数
        def __call__(
            # 主提示文本，可以是单个字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 第二个提示文本，可选
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 第三个提示文本，可选
            prompt_3: Optional[Union[str, List[str]]] = None,
            # 输入图像，类型为管道图像输入
            image: PipelineImageInput = None,
            # 强度参数，默认值为0.6
            strength: float = 0.6,
            # 推理步骤数，默认值为50
            num_inference_steps: int = 50,
            # 时间步长列表，可选
            timesteps: List[int] = None,
            # 引导比例，默认值为7.0
            guidance_scale: float = 7.0,
            # 负提示文本，可选
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负提示文本，可选
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 第三个负提示文本，可选
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 随机数生成器，可选
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，类型为FloatTensor，可选
            latents: Optional[torch.FloatTensor] = None,
            # 提示嵌入，类型为FloatTensor，可选
            prompt_embeds: Optional[torch.FloatTensor] = None,
            # 负提示嵌入，类型为FloatTensor，可选
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 池化后的提示嵌入，类型为FloatTensor，可选
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 负池化提示嵌入，类型为FloatTensor，可选
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为True
            return_dict: bool = True,
            # 跳过的剪辑层数，可选
            clip_skip: Optional[int] = None,
            # 在步骤结束时的回调函数，可选
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 步骤结束时的张量输入回调列表，默认包含"latents"
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 最大序列长度，默认为256
            max_sequence_length: int = 256,
```