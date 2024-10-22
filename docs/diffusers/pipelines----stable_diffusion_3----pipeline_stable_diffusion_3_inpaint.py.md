# `.\diffusers\pipelines\stable_diffusion_3\pipeline_stable_diffusion_3_inpaint.py`

```py
# 版权声明，包含版权持有者及其授权信息
# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版进行授权
# 该文件仅可在遵循许可的情况下使用
# 许可证的副本可以在以下地址获得
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面协议另有约定，否则以 "按原样" 基础分发软件，
# 不提供任何形式的担保或条件
# 查看许可证以获取特定语言的权限和限制

import inspect  # 导入 inspect 模块以检查对象
from typing import Callable, Dict, List, Optional, Union  # 导入类型提示相关的类

import torch  # 导入 PyTorch 库
from transformers import (  # 从 transformers 库导入必要的类
    CLIPTextModelWithProjection,  # 导入 CLIP 文本模型类
    CLIPTokenizer,  # 导入 CLIP 词元化工具
    T5EncoderModel,  # 导入 T5 编码器模型
    T5TokenizerFast,  # 导入快速 T5 词元化工具
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入回调相关类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理类
from ...loaders import SD3LoraLoaderMixin  # 导入 Lora 加载器混合类
from ...models.autoencoders import AutoencoderKL  # 导入自动编码器类
from ...models.transformers import SD3Transformer2DModel  # 导入 2D 转换模型
from ...schedulers import FlowMatchEulerDiscreteScheduler  # 导入调度器类
from ...utils import (  # 导入实用工具
    USE_PEFT_BACKEND,  # 导入 PEFT 后端标志
    is_torch_xla_available,  # 导入检查 Torch XLA 可用性的函数
    logging,  # 导入日志模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放 Lora 层的函数
    unscale_lora_layers,  # 导入取消缩放 Lora 层的函数
)
from ...utils.torch_utils import randn_tensor  # 导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道类
from .pipeline_output import StableDiffusion3PipelineOutput  # 导入稳定扩散输出类


# 检查 Torch XLA 是否可用，导入相应模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 模型核心模块

    XLA_AVAILABLE = True  # 设置 XLA 可用标志
else:
    XLA_AVAILABLE = False  # 设置 XLA 不可用标志


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 pylint 命名警告

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串
    Examples:  # 示例说明
        ```py  # 开始代码块
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusion3InpaintPipeline  # 导入稳定扩散修复管道类
        >>> from diffusers.utils import load_image  # 导入加载图像的实用工具

        >>> pipe = StableDiffusion3InpaintPipeline.from_pretrained(  # 从预训练模型加载管道
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16  # 指定模型名称及数据类型
        ... )
        >>> pipe.to("cuda")  # 将管道转移到 CUDA 设备
        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"  # 定义生成图像的提示
        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"  # 定义源图像 URL
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"  # 定义掩模图像 URL
        >>> source = load_image(img_url)  # 加载源图像
        >>> mask = load_image(mask_url)  # 加载掩模图像
        >>> image = pipe(prompt=prompt, image=source, mask_image=mask).images[0]  # 生成修复后的图像
        >>> image.save("sd3_inpainting.png")  # 保存生成的图像
        ```py  # 结束代码块
"""


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(  # 定义函数以检索潜在变量
    encoder_output: torch.Tensor,  # 输入为编码器输出张量
    generator: Optional[torch.Generator] = None,  # 可选参数，指定随机数生成器
    sample_mode: str = "sample"  # 指定采样模式，默认为 "sample"
):
    # 检查 encoder_output 是否有 latent_dist 属性，并且样本模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中采样，使用指定的生成器
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否有 latent_dist 属性，并且样本模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的众数
        return encoder_output.latent_dist.mode()
    # 检查 encoder_output 是否有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 latents 属性的值
        return encoder_output.latents
    # 如果以上条件都不满足，则抛出 AttributeError
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的代码
def retrieve_timesteps(
    # 调度器，用于获取时间步
    scheduler,
    # 推理步骤的数量，默认为 None
    num_inference_steps: Optional[int] = None,
    # 要移动到的设备，默认为 None
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步，默认为 None
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma，默认为 None
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数，传递给调度器的 set_timesteps 方法
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器获取时间步。处理
    自定义时间步。任何 kwargs 将被传递给 `scheduler.set_timesteps`。

    参数：
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            用于生成样本的扩散步骤数。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`，*可选*):
            时间步应移动到的设备。如果为 `None`，则时间步不会移动。
        timesteps (`List[int]`，*可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递 `timesteps`，
            则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`，*可选*):
            自定义 sigma，用于覆盖调度器的时间步间隔策略。如果传递 `sigmas`，
            则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`：一个元组，第一个元素是调度器的时间步调度，
        第二个元素是推理步骤的数量。
    """
    # 如果同时传递了 timesteps 和 sigmas，则抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("只能传递 `timesteps` 或 `sigmas` 之一。请选择一个设置自定义值")
    # 如果传递了 timesteps
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受 timesteps 参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义"
                f" 时间步调度。请检查您是否使用了正确的调度器。"
            )
        # 调用调度器的 set_timesteps 方法，传递自定义时间步和设备
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传递了 sigmas
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigmas 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义"
                f" sigma 调度。请检查您是否使用了正确的调度器。"
            )
        # 调用调度器的 set_timesteps 方法，传递自定义 sigma 和设备
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果不是特定条件，则设置推理步骤的时间步数
        else:
            # 调用调度器设置推理步骤数，并指定设备及其他参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器的时间步数
            timesteps = scheduler.timesteps
        # 返回时间步数和推理步骤数
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusion3InpaintPipeline 的类，继承自 DiffusionPipeline
class StableDiffusion3InpaintPipeline(DiffusionPipeline):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            条件变换器（MMDiT）架构，用于对编码后的图像潜变量进行去噪。
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            用于与 `transformer` 结合使用的调度器，用于去噪编码后的图像潜变量。
        vae ([`AutoencoderKL`]):
            变分自编码器（VAE）模型，用于将图像编码为潜在表示并解码。
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection)，
            特别是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体，
            具有额外的投影层，该层用具有 `hidden_size` 维度的对角矩阵初始化。
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection)，
            特别是 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 变体。
        text_encoder_3 ([`T5EncoderModel`]):
            冻结的文本编码器。Stable Diffusion 3 使用 [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)，
            特别是 [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) 变体。
        tokenizer (`CLIPTokenizer`):
            类 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 的分词器。
        tokenizer_2 (`CLIPTokenizer`):
            第二个类 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 的分词器。
        tokenizer_3 (`T5TokenizerFast`):
            类 [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer) 的分词器。
    """
    
    # 定义模型 CPU 卸载的顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    # 定义可选组件的空列表
    _optional_components = []
    # 定义回调张量输入的列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    # 初始化方法，接收多个参数
    def __init__(
        # 接收条件变换器模型
        self,
        transformer: SD3Transformer2DModel,
        # 接收调度器
        scheduler: FlowMatchEulerDiscreteScheduler,
        # 接收变分自编码器
        vae: AutoencoderKL,
        # 接收文本编码器
        text_encoder: CLIPTextModelWithProjection,
        # 接收分词器
        tokenizer: CLIPTokenizer,
        # 接收第二个文本编码器
        text_encoder_2: CLIPTextModelWithProjection,
        # 接收第二个分词器
        tokenizer_2: CLIPTokenizer,
        # 接收第三个文本编码器
        text_encoder_3: T5EncoderModel,
        # 接收第三个分词器
        tokenizer_3: T5TokenizerFast,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册多个模块，方便后续使用
        self.register_modules(
            # 注册变分自编码器
            vae=vae,
            # 注册文本编码器
            text_encoder=text_encoder,
            # 注册第二个文本编码器
            text_encoder_2=text_encoder_2,
            # 注册第三个文本编码器
            text_encoder_3=text_encoder_3,
            # 注册分词器
            tokenizer=tokenizer,
            # 注册第二个分词器
            tokenizer_2=tokenizer_2,
            # 注册第三个分词器
            tokenizer_3=tokenizer_3,
            # 注册转换器
            transformer=transformer,
            # 注册调度器
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子，基于块输出通道数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化图像处理器，使用 VAE 缩放因子和潜在通道数
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.vae.config.latent_channels
        )
        # 初始化掩码处理器，设置参数以处理图像
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        # 获取分词器的最大长度
        self.tokenizer_max_length = self.tokenizer.model_max_length
        # 获取转换器的默认采样大小
        self.default_sample_size = self.transformer.config.sample_size

    # 从稳定扩散管道复制的方法，获取 T5 提示嵌入
    def _get_t5_prompt_embeds(
        self,
        # 输入提示，可以是字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: int = 1,
        # 最大序列长度
        max_sequence_length: int = 256,
        # 可选的设备参数
        device: Optional[torch.device] = None,
        # 可选的数据类型
        dtype: Optional[torch.dtype] = None,
    # 定义一个方法，接受多个参数，处理输入文本以生成提示嵌入
        ):
            # 如果未指定设备，则使用默认执行设备
            device = device or self._execution_device
            # 如果未指定数据类型，则使用文本编码器的数据类型
            dtype = dtype or self.text_encoder.dtype
    
            # 如果提示是字符串，则将其转换为列表；否则保持原样
            prompt = [prompt] if isinstance(prompt, str) else prompt
            # 获取提示的批处理大小
            batch_size = len(prompt)
    
            # 如果第三个文本编码器为空，则返回一个零张量
            if self.text_encoder_3 is None:
                return torch.zeros(
                    (
                        batch_size * num_images_per_prompt,
                        self.tokenizer_max_length,
                        self.transformer.config.joint_attention_dim,
                    ),
                    device=device,
                    dtype=dtype,
                )
    
            # 使用第三个文本编码器对提示进行标记化，返回张量
            text_inputs = self.tokenizer_3(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            # 提取输入 ID
            text_input_ids = text_inputs.input_ids
            # 获取未截断的 ID
            untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids
    
            # 检查未截断的 ID 是否大于等于输入 ID，并且不相等
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                # 解码被截断的文本并记录警告
                removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_sequence_length} tokens: {removed_text}"
                )
    
            # 获取文本输入的嵌入
            prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]
    
            # 获取文本编码器的数据类型
            dtype = self.text_encoder_3.dtype
            # 将提示嵌入转换为指定的数据类型和设备
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
            # 获取嵌入的形状，提取序列长度
            _, seq_len, _ = prompt_embeds.shape
    
            # 复制文本嵌入和注意力掩码以适应每个提示的生成，使用适合 MPS 的方法
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # 重塑嵌入以匹配批处理大小和生成的图像数量
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    
            # 返回生成的提示嵌入
            return prompt_embeds
    
        # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds 复制的方法
        def _get_clip_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            device: Optional[torch.device] = None,
            clip_skip: Optional[int] = None,
            clip_model_index: int = 0,
    ):
        # 如果没有指定设备，则使用当前对象的执行设备
        device = device or self._execution_device

        # 定义两个 CLIP 分词器的列表
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        # 定义两个 CLIP 文本编码器的列表
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        # 根据所选模型索引选择相应的分词器
        tokenizer = clip_tokenizers[clip_model_index]
        # 根据所选模型索引选择相应的文本编码器
        text_encoder = clip_text_encoders[clip_model_index]

        # 如果 prompt 是字符串，则将其转换为列表，否则保持不变
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 获取 prompt 的批量大小
        batch_size = len(prompt)

        # 使用选择的分词器对 prompt 进行编码，返回张量
        text_inputs = tokenizer(
            prompt,
            padding="max_length",  # 填充到最大长度
            max_length=self.tokenizer_max_length,  # 最大长度限制
            truncation=True,  # 如果超出最大长度则截断
            return_tensors="pt",  # 返回 PyTorch 张量
        )

        # 获取编码后的输入 ID
        text_input_ids = text_inputs.input_ids
        # 获取未截断的输入 ID
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        # 检查未截断的输入 ID 是否超过最大长度，并且与截断的 ID 是否不同
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            # 解码被截断的文本部分
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            # 记录警告，指出被截断的部分
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        # 使用文本编码器对输入 ID 进行编码，并输出隐藏状态
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        # 获取池化后的提示嵌入
        pooled_prompt_embeds = prompt_embeds[0]

        # 如果没有指定跳过层，则使用倒数第二层的嵌入
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # 根据指定的跳过层获取相应的嵌入
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        # 将提示嵌入转换为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # 获取提示嵌入的形状信息
        _, seq_len, _ = prompt_embeds.shape
        # 为每个提示生成多个文本嵌入，使用适合 MPS 的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 调整形状以符合批处理要求
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 重复池化的提示嵌入以匹配生成数量
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 调整形状以符合批处理要求
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # 返回处理后的提示嵌入和池化提示嵌入
        return prompt_embeds, pooled_prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt 复制而来
    # 定义一个方法来编码提示信息
        def encode_prompt(
            self,  # 方法的第一个参数，指向当前实例
            prompt: Union[str, List[str]],  # 第一个提示，可以是字符串或字符串列表
            prompt_2: Union[str, List[str]],  # 第二个提示，可以是字符串或字符串列表
            prompt_3: Union[str, List[str]],  # 第三个提示，可以是字符串或字符串列表
            device: Optional[torch.device] = None,  # 可选参数，指定设备（CPU或GPU）
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量，默认为1
            do_classifier_free_guidance: bool = True,  # 是否使用无分类器引导，默认为True
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示，可以是字符串或字符串列表
            negative_prompt_2: Optional[Union[str, List[str]]] = None,  # 第二个负面提示
            negative_prompt_3: Optional[Union[str, List[str]]] = None,  # 第三个负面提示
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选参数，提示的嵌入表示
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选参数，负面提示的嵌入表示
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选参数，池化后的提示嵌入表示
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选参数，池化后的负面提示嵌入表示
            clip_skip: Optional[int] = None,  # 可选参数，控制剪辑跳过的层数
            max_sequence_length: int = 256,  # 最大序列长度，默认为256
            lora_scale: Optional[float] = None,  # 可选参数，LORA缩放因子
        # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.check_inputs 拷贝而来
        def check_inputs(
            self,  # 当前实例
            prompt,  # 第一个提示
            prompt_2,  # 第二个提示
            prompt_3,  # 第三个提示
            strength,  # 强度参数
            negative_prompt=None,  # 可选的负面提示
            negative_prompt_2=None,  # 第二个负面提示
            negative_prompt_3=None,  # 第三个负面提示
            prompt_embeds=None,  # 可选的提示嵌入表示
            negative_prompt_embeds=None,  # 可选的负面提示嵌入表示
            pooled_prompt_embeds=None,  # 可选的池化提示嵌入表示
            negative_pooled_prompt_embeds=None,  # 可选的池化负面提示嵌入表示
            callback_on_step_end_tensor_inputs=None,  # 可选的步骤结束回调输入
            max_sequence_length=None,  # 可选的最大序列长度
        # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps 拷贝而来
        def get_timesteps(self, num_inference_steps, strength, device):  # 定义获取时间步的方法
            # 使用 init_timestep 获取原始时间步
            init_timestep = min(num_inference_steps * strength, num_inference_steps)  # 计算初始化时间步
    
            t_start = int(max(num_inference_steps - init_timestep, 0))  # 计算起始时间步
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]  # 获取调度器中的时间步
            if hasattr(self.scheduler, "set_begin_index"):  # 检查调度器是否有设置开始索引的方法
                self.scheduler.set_begin_index(t_start * self.scheduler.order)  # 设置调度器的开始索引
    
            return timesteps, num_inference_steps - t_start  # 返回时间步和剩余的推理步骤
    
        def prepare_latents(  # 定义准备潜在变量的方法
            self,  # 当前实例
            batch_size,  # 批处理大小
            num_channels_latents,  # 潜在变量的通道数
            height,  # 图像高度
            width,  # 图像宽度
            dtype,  # 数据类型
            device,  # 设备
            generator,  # 随机数生成器
            latents=None,  # 可选的潜在变量
            image=None,  # 可选的输入图像
            timestep=None,  # 可选的时间步
            is_strength_max=True,  # 强度是否为最大值，默认为True
            return_noise=False,  # 是否返回噪声，默认为False
            return_image_latents=False,  # 是否返回图像潜在变量，默认为False
    ):
        # 定义输出的形状，包含批量大小、通道数、高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,  # 根据 VAE 的缩放因子计算高度
            int(width) // self.vae_scale_factor,    # 根据 VAE 的缩放因子计算宽度
        )
        # 检查生成器是否是列表且长度与批量大小不匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出值错误，提示生成器长度与批量大小不匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 检查图像或时间步是否为 None，且强度未达到最大值
        if (image is None or timestep is None) and not is_strength_max:
            # 抛出值错误，提示必须提供图像或噪声时间步
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        # 检查是否返回图像潜变量，或潜变量为 None 且强度未达到最大值
        if return_image_latents or (latents is None and not is_strength_max):
            # 将图像转换到指定设备和数据类型
            image = image.to(device=device, dtype=dtype)

            # 检查图像的通道数是否为 16
            if image.shape[1] == 16:
                # 如果是，则直接将图像潜变量设置为图像
                image_latents = image
            else:
                # 否则，通过 VAE 编码图像来获取潜变量
                image_latents = self._encode_vae_image(image=image, generator=generator)
            # 根据批量大小重复潜变量，以匹配批量尺寸
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # 检查潜变量是否为 None
        if latents is None:
            # 根据形状生成噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 如果强度为 1，则初始化潜变量为噪声，否则初始化为图像与噪声的组合
            latents = noise if is_strength_max else self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            # 将潜变量移动到指定设备
            noise = latents.to(device)
            # 直接将噪声赋值给潜变量
            latents = noise

        # 创建输出元组，包含潜变量
        outputs = (latents,)

        # 如果需要返回噪声，则将噪声添加到输出中
        if return_noise:
            outputs += (noise,)

        # 如果需要返回图像潜变量，则将其添加到输出中
        if return_image_latents:
            outputs += (image_latents,)

        # 返回输出元组
        return outputs

    # 定义编码 VAE 图像的函数
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        # 检查生成器是否为列表
        if isinstance(generator, list):
            # 遍历图像，编码每个图像并提取潜变量
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            # 将潜变量沿着第一个维度拼接成一个张量
            image_latents = torch.cat(image_latents, dim=0)
        else:
            # 如果不是列表，则直接编码整个图像
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        # 对潜变量进行缩放和偏移处理
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 返回处理后的潜变量
        return image_latents

    # 定义准备掩码潜变量的函数
    def prepare_mask_latents(
        self,
        mask,                       # 掩码张量
        masked_image,              # 被掩盖的图像
        batch_size,                # 批量大小
        num_images_per_prompt,     # 每个提示的图像数量
        height,                    # 图像高度
        width,                     # 图像宽度
        dtype,                     # 数据类型
        device,                    # 设备类型
        generator,                 # 随机生成器
        do_classifier_free_guidance,# 是否进行无分类器引导
    ):
        # 将掩码调整为与潜在向量形状相同，以便在连接掩码和潜在向量时使用
        # 在转换数据类型之前执行此操作，以避免在使用 cpu_offload 和半精度时出现问题
        mask = torch.nn.functional.interpolate(
            # 使用插值方法将掩码调整为指定的高度和宽度
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        # 将掩码移动到指定的设备并转换为指定的数据类型
        mask = mask.to(device=device, dtype=dtype)

        # 计算总的批大小，考虑每个提示生成的图像数量
        batch_size = batch_size * num_images_per_prompt

        # 将遮罩图像移动到指定的设备并转换为指定的数据类型
        masked_image = masked_image.to(device=device, dtype=dtype)

        # 如果掩码图像的形状为 16，直接赋值给潜在图像变量
        if masked_image.shape[1] == 16:
            masked_image_latents = masked_image
        else:
            # 使用 VAE 编码器检索潜在图像
            masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        # 对潜在图像进行归一化处理，减去偏移量并乘以缩放因子
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 为每个提示的生成重复掩码和潜在图像，使用适合 MPS 的方法
        if mask.shape[0] < batch_size:
            # 检查掩码数量是否能整除批大小
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    # 如果掩码数量和批大小不匹配，抛出错误
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            # 根据批大小重复掩码
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        # 检查潜在图像数量是否能整除批大小
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    # 如果潜在图像数量和批大小不匹配，抛出错误
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            # 根据批大小重复潜在图像
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 根据是否使用无分类器自由引导选择重复掩码
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        # 根据是否使用无分类器自由引导选择重复潜在图像
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # 将潜在图像移动到指定的设备并转换为指定的数据类型，以防拼接时出现设备错误
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        # 返回处理后的掩码和潜在图像
        return mask, masked_image_latents

    @property
    def guidance_scale(self):
        # 返回指导缩放因子
        return self._guidance_scale

    @property
    def clip_skip(self):
        # 返回跳过剪辑的参数
        return self._clip_skip

    # 此处 `guidance_scale` 类似于方程 (2) 中的指导权重 `w`
    # 来自 Imagen 论文: https://arxiv.org/pdf/2205.11487.pdf 。`guidance_scale = 1`
    # 表示不进行分类器自由引导的情况
    @property
    def do_classifier_free_guidance(self):
        # 判断引导尺度是否大于1，返回布尔值
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        # 返回时间步的数量
        return self._num_timesteps

    @property
    def interrupt(self):
        # 返回中断状态
        return self._interrupt

    # 该装饰器用于禁止梯度计算，减少内存消耗
    @torch.no_grad()
    # 替换文档字符串为示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 接收的提示文本，可以是字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 第二个提示文本，默认为 None
        prompt_2: Optional[Union[str, List[str]]] = None,
        # 第三个提示文本，默认为 None
        prompt_3: Optional[Union[str, List[str]]] = None,
        # 输入的图像，可以是图像数据
        image: PipelineImageInput = None,
        # 掩码图像，用于特定操作
        mask_image: PipelineImageInput = None,
        # 被掩码的图像潜变量
        masked_image_latents: PipelineImageInput = None,
        # 图像高度，默认为 None
        height: int = None,
        # 图像宽度，默认为 None
        width: int = None,
        # 可选的填充掩码裁剪值，默认为 None
        padding_mask_crop: Optional[int] = None,
        # 强度参数，默认为 0.6
        strength: float = 0.6,
        # 推理步骤的数量，默认为 50
        num_inference_steps: int = 50,
        # 时间步列表，默认为 None
        timesteps: List[int] = None,
        # 引导尺度，默认为 7.0
        guidance_scale: float = 7.0,
        # 可选的负面提示文本，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 第二个负面提示文本，默认为 None
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 第三个负面提示文本，默认为 None
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 随机数生成器，可以是单个生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜变量，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 提示嵌入，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负面提示嵌入，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 聚合的提示嵌入，默认为 None
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 负面聚合提示嵌入，默认为 None
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典，默认为 True
        return_dict: bool = True,
        # 可选的跳过剪辑参数，默认为 None
        clip_skip: Optional[int] = None,
        # 每步结束时的回调函数，默认为 None
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 在每步结束时使用的张量输入回调，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 最大序列长度，默认为 256
        max_sequence_length: int = 256,
```