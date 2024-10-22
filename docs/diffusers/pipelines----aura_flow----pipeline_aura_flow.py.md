# `.\diffusers\pipelines\aura_flow\pipeline_aura_flow.py`

```py
# 版权声明，2024年AuraFlow作者和HuggingFace团队保留所有权利
#
# 根据Apache许可证第2.0版（“许可证”）进行授权；
# 除非遵守该许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件
# 按“原样”分发，不附带任何形式的保证或条件，
# 明示或暗示。有关许可证的具体权限和
# 限制，请参见许可证。
import inspect  # 导入inspect模块，用于获取对象的信息
from typing import List, Optional, Tuple, Union  # 从typing模块导入类型提示工具

import torch  # 导入torch库，用于张量计算和深度学习
from transformers import T5Tokenizer, UMT5EncoderModel  # 从transformers导入T5分词器和UMT5编码模型

from ...image_processor import VaeImageProcessor  # 导入变分自编码器图像处理器
from ...models import AuraFlowTransformer2DModel, AutoencoderKL  # 导入模型类
from ...models.attention_processor import AttnProcessor2_0, FusedAttnProcessor2_0, XFormersAttnProcessor  # 导入注意力处理器类
from ...schedulers import FlowMatchEulerDiscreteScheduler  # 导入调度器类
from ...utils import logging, replace_example_docstring  # 导入日志工具和文档替换工具
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 导入扩散管道和图像输出类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

EXAMPLE_DOC_STRING = """
    示例：
        ```py
        >>> import torch
        >>> from diffusers import AuraFlowPipeline

        >>> pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow", torch_dtype=torch.float16)  # 从预训练模型创建管道
        >>> pipe = pipe.to("cuda")  # 将管道移动到GPU设备
        >>> prompt = "A cat holding a sign that says hello world"  # 定义输入提示
        >>> image = pipe(prompt).images[0]  # 生成图像
        >>> image.save("aura_flow.png")  # 保存生成的图像
        ```py
"""

# 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion导入的函数
def retrieve_timesteps(
    scheduler,  # 调度器对象，用于设置时间步
    num_inference_steps: Optional[int] = None,  # 可选的推理步骤数
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备参数
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表
    sigmas: Optional[List[float]] = None,  # 可选的sigma值列表
    **kwargs,  # 其他参数，传递给调度器
):
    """
    调用调度器的`set_timesteps`方法并在调用后从调度器检索时间步。处理
    自定义时间步。任何kwargs将被传递给`scheduler.set_timesteps`。
    # 定义参数说明
        Args:
            scheduler (`SchedulerMixin`):  # 调度器，用于获取时间步
                The scheduler to get timesteps from.
            num_inference_steps (`int`):  # 用于生成样本的扩散步数
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):  # 指定时间步移动的设备
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):  # 自定义时间步以覆盖调度器的时间步间隔策略
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):  # 自定义 sigma 以覆盖调度器的时间步间隔策略
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.
    
        Returns:
            `Tuple[torch.Tensor, int]`:  # 返回一个元组，第一个元素是调度器的时间步计划，第二个元素是推理步骤数量
            A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        # 检查是否同时传入了 timesteps 和 sigmas
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        # 如果传入了 timesteps
        if timesteps is not None:
            # 检查当前调度器是否支持自定义时间步
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(  # 抛出错误，表示当前调度器不支持自定义时间步
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            # 设置调度器的时间步
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            # 获取调度器的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤数量
            num_inference_steps = len(timesteps)
        # 如果传入了 sigmas
        elif sigmas is not None:
            # 检查当前调度器是否支持自定义 sigma
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(  # 抛出错误，表示当前调度器不支持自定义 sigma
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            # 设置调度器的 sigma
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            # 获取调度器的时间步
            timesteps = scheduler.timesteps
            # 计算推理步骤数量
            num_inference_steps = len(timesteps)
        # 如果没有传入 timesteps 或 sigmas
        else:
            # 设置调度器的默认时间步
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器的时间步
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤数量
        return timesteps, num_inference_steps
# 定义一个名为 AuraFlowPipeline 的类，继承自 DiffusionPipeline
class AuraFlowPipeline(DiffusionPipeline):
    r"""
    参数：
        tokenizer (`T5TokenizerFast`):
            T5Tokenizer 类的分词器
        text_encoder ([`T5EncoderModel`]):
            冻结的文本编码器。AuraFlow 使用 T5，具体是 
            [EleutherAI/pile-t5-xl](https://huggingface.co/EleutherAI/pile-t5-xl) 变体
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器模型
        transformer ([`AuraFlowTransformer2DModel`]):
            条件 Transformer 架构 (MMDiT 和 DiT) 用于去噪编码的图像潜在表示
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            用于与 `transformer` 结合使用的调度器，以去噪编码的图像潜在表示
    """

    # 可选组件的列表，初始化为空
    _optional_components = []
    # 定义模型在 CPU 上卸载的顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 初始化方法，定义类的参数
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKL,
        transformer: AuraFlowTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模块，将各个组件注册到类中
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # 计算 VAE 的缩放因子，根据配置的通道数决定
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # 初始化图像处理器，使用计算得到的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # 检查输入参数的方法
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        # 检查高度和宽度是否为8的倍数，若不是则抛出错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查是否同时提供了提示和提示嵌入，若是则抛出错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查提示和提示嵌入是否都未提供，若是则抛出错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示类型是否为字符串或列表，若不是则抛出错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了提示和负提示嵌入，若是则抛出错误
        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查是否同时提供了负提示和负提示嵌入，若是则抛出错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查如果提供了提示嵌入则必须提供相应的注意力掩码，若不然则抛出错误
        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        # 检查如果提供了负提示嵌入则必须提供相应的注意力掩码，若不然则抛出错误
        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        # 检查提示嵌入和负提示嵌入的形状是否一致，若不一致则抛出错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            # 检查提示和负提示注意力掩码的形状是否一致，若不一致则抛出错误
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )
    # 定义一个编码提示的函数，接受多个参数以构建提示
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 提示文本，可以是字符串或字符串列表
        negative_prompt: Union[str, List[str]] = None,  # 负面提示文本，可以是字符串或字符串列表，默认为 None
        do_classifier_free_guidance: bool = True,  # 是否使用无分类器引导，默认为 True
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量，默认为 1
        device: Optional[torch.device] = None,  # 指定设备（如 CPU 或 GPU），默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,  # 提示的嵌入张量，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示的嵌入张量，默认为 None
        prompt_attention_mask: Optional[torch.Tensor] = None,  # 提示的注意力掩码，默认为 None
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # 负面提示的注意力掩码，默认为 None
        max_sequence_length: int = 256,  # 最大序列长度，默认为 256
    # 从 diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents 复制
    def prepare_latents(
        self,
        batch_size,  # 批处理大小
        num_channels_latents,  # 潜在通道数量
        height,  # 图像高度
        width,  # 图像宽度
        dtype,  # 数据类型
        device,  # 指定设备
        generator,  # 随机数生成器
        latents=None,  # 潜在张量，默认为 None
    ):
        # 如果提供了潜在张量，则将其转换为指定设备和数据类型
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # 定义潜在张量的形状
        shape = (
            batch_size,  # 批处理大小
            num_channels_latents,  # 潜在通道数量
            int(height) // self.vae_scale_factor,  # 计算缩放后的高度
            int(width) // self.vae_scale_factor,  # 计算缩放后的宽度
        )

        # 检查生成器列表的长度是否与批处理大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(  # 抛出错误，提示生成器长度与批处理大小不匹配
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 生成随机潜在张量
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 返回生成的潜在张量
        return latents

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.upcast_vae 复制
    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 数据类型
        self.vae.to(dtype=torch.float32)
        # 检查当前使用的处理器是否为特定类型
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,  # 检查是否为 AttnProcessor2_0 类型
                XFormersAttnProcessor,  # 检查是否为 XFormersAttnProcessor 类型
                FusedAttnProcessor2_0,  # 检查是否为 FusedAttnProcessor2_0 类型
            ),
        )
        # 如果使用了 xformers 或 torch_2_0，则注意力块不需要为 float32，从而节省大量内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层转换为原始数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将输入卷积层转换为原始数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将中间块转换为原始数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 不计算梯度装饰器，通常用于推理时以节省内存和计算
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的方法，支持多种参数配置
        def __call__(
            # 提示文本，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 负面提示文本，可以是单个字符串或字符串列表
            negative_prompt: Union[str, List[str]] = None,
            # 推理步骤的数量，默认为50
            num_inference_steps: int = 50,
            # 时间步列表，用于推理过程
            timesteps: List[int] = None,
            # sigma值列表，控制噪声级别
            sigmas: List[float] = None,
            # 引导缩放因子，默认为3.5
            guidance_scale: float = 3.5,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 生成图像的高度，默认为1024
            height: Optional[int] = 1024,
            # 生成图像的宽度，默认为1024
            width: Optional[int] = 1024,
            # 随机数生成器，可以是单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 初始潜在向量，可以是一个张量
            latents: Optional[torch.Tensor] = None,
            # 提示文本的嵌入向量，可以是一个张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 提示文本的注意力掩码，可以是一个张量
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 负面提示文本的嵌入向量，可以是一个张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示文本的注意力掩码，可以是一个张量
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 最大序列长度，默认为256
            max_sequence_length: int = 256,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的输出，默认为True
            return_dict: bool = True,
```