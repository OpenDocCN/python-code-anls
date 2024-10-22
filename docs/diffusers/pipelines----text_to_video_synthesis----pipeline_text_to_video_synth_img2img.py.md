# `.\diffusers\pipelines\text_to_video_synthesis\pipeline_text_to_video_synth_img2img.py`

```py
# 版权声明，标明此文件的版权归 HuggingFace 团队所有
# 
# 根据 Apache License 2.0 版本（“许可证”）授权；
# 除非遵守该许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，按照许可证分发的软件按“原样”提供，
# 不附带任何形式的明示或暗示的担保或条件。
# 有关特定语言管理权限和
# 限制的更多信息，请参见许可证。

# 导入 inspect 模块以进行对象检查和获取信息
import inspect
# 从 typing 模块导入多种类型，用于类型提示
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 torch 库，用于深度学习相关操作
import torch
# 从 transformers 库导入 CLIP 模型和标记器
from transformers import CLIPTextModel, CLIPTokenizer

# 从 loaders 模块导入所需的混合类
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从 models 模块导入自动编码器和条件 UNet 模型
from ...models import AutoencoderKL, UNet3DConditionModel
# 从 lora 模块导入调整 LORA 规模的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从 schedulers 模块导入 Karras 扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从 utils 模块导入多个实用工具函数和常量
from ...utils import (
    USE_PEFT_BACKEND,  # 用于指定是否使用 PEFT 后端
    deprecate,  # 用于标记已弃用的函数
    logging,  # 用于日志记录
    replace_example_docstring,  # 用于替换示例文档字符串
    scale_lora_layers,  # 用于缩放 LORA 层
    unscale_lora_layers,  # 用于反缩放 LORA 层
)
# 从 torch_utils 模块导入生成随机张量的函数
from ...utils.torch_utils import randn_tensor
# 导入视频处理器类
from ...video_processor import VideoProcessor
# 从 pipeline_utils 模块导入扩散管道和稳定扩散混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入文本到视频的稳定扩散管道输出类
from . import TextToVideoSDPipelineOutput

# 获取当前模块的日志记录器实例
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串的模板，可能用于生成文档
EXAMPLE_DOC_STRING = """
``` 
```py 
``` 
```py 
``` 
    Examples:
        ```py
        # 导入所需的库
        >>> import torch
        >>> from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        >>> from diffusers.utils import export_to_video

        # 从预训练模型加载扩散管道，并指定数据类型为 float16
        >>> pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
        # 设置调度器为多步 DPM 解决器，使用管道当前调度器的配置
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # 将模型移动到 CUDA 设备以加速计算
        >>> pipe.to("cuda")

        # 定义生成视频的提示文本
        >>> prompt = "spiderman running in the desert"
        # 使用提示生成视频帧，设置推理步数、高度、宽度和帧数
        >>> video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames[0]
        # 导出低分辨率视频并保存到指定路径
        >>> # safe low-res video
        >>> video_path = export_to_video(video_frames, output_video_path="./video_576_spiderman.mp4")

        # 将文本到图像的模型移回 CPU
        >>> # let's offload the text-to-image model
        >>> pipe.to("cpu")

        # 重新加载图像到图像的模型
        >>> # and load the image-to-image model
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     "cerspense/zeroscope_v2_XL", torch_dtype=torch.float16, revision="refs/pr/15"
        ... )
        # 设置调度器为多步 DPM 解决器，使用新的调度器配置
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # 启用模型的 CPU 卸载，以节省内存
        >>> pipe.enable_model_cpu_offload()

        # VAE 占用大量内存，确保以切片模式运行以降低内存使用
        >>> # The VAE consumes A LOT of memory, let's make sure we run it in sliced mode
        >>> pipe.vae.enable_slicing()

        # 将视频帧上采样到更高的分辨率
        >>> # now let's upscale it
        >>> video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

        # 使用生成的提示和上采样的视频帧进行去噪处理
        >>> # and denoise it
        >>> video_frames = pipe(prompt, video=video, strength=0.6).frames[0]
        # 导出去噪后的视频并保存到指定路径
        >>> video_path = export_to_video(video_frames, output_video_path="./video_1024_spiderman.mp4")
        # 返回最终视频的路径
        >>> video_path
# 导入所需的模块和类型
"""


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    # 输入参数：编码器输出，类型为 torch.Tensor
    encoder_output: torch.Tensor, 
    # 可选的随机数生成器
    generator: Optional[torch.Generator] = None, 
    # 采样模式，默认为 "sample"
    sample_mode: str = "sample"
):
    # 检查编码器输出是否具有 "latent_dist" 属性，并且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 返回从潜在分布中采样的结果
        return encoder_output.latent_dist.sample(generator)
    # 检查编码器输出是否具有 "latent_dist" 属性，并且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回潜在分布的众数
        return encoder_output.latent_dist.mode()
    # 检查编码器输出是否具有 "latents" 属性
    elif hasattr(encoder_output, "latents"):
        # 返回编码器输出的潜在值
        return encoder_output.latents
    # 如果都不满足，抛出属性错误
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# 定义 VideoToVideoSDPipeline 类，继承多个混入类
class VideoToVideoSDPipeline(
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin
):
    r"""
    用于文本引导的视频到视频生成的管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承了以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器（VAE）模型，用于将视频编码和解码为潜在表示。
        text_encoder ([`CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer (`CLIPTokenizer`):
            [`~transformers.CLIPTokenizer`] 用于对文本进行标记化。
        unet ([`UNet3DConditionModel`]):
            [`UNet3DConditionModel`] 用于去噪编码的视频潜在值。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码的图像潜在值。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"

    # 初始化函数，设置各个组件
    def __init__(
        # 输入参数：变分自编码器
        vae: AutoencoderKL,
        # 输入参数：文本编码器
        text_encoder: CLIPTextModel,
        # 输入参数：标记器
        tokenizer: CLIPTokenizer,
        # 输入参数：去噪模型
        unet: UNet3DConditionModel,
        # 输入参数：调度器
        scheduler: KarrasDiffusionSchedulers,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模型模块
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化视频处理器，禁用调整大小并使用 VAE 缩放因子
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的
    def _encode_prompt(
        self,  # 当前实例的引用
        prompt,  # 要编码的提示文本
        device,  # 设备类型（例如 CPU 或 GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否进行无分类器引导
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
        **kwargs,  # 其他可选参数
    ):
        # 定义弃用消息，告知用户此函数在未来版本中将被移除
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数，记录弃用信息
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法获取提示嵌入的元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 传递提示文本
            device=device,  # 传递设备信息
            num_images_per_prompt=num_images_per_prompt,  # 传递每个提示的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 传递无分类器引导信息
            negative_prompt=negative_prompt,  # 传递负面提示文本
            prompt_embeds=prompt_embeds,  # 传递提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,  # 传递负面提示嵌入
            lora_scale=lora_scale,  # 传递 Lora 缩放因子
            **kwargs,  # 传递其他参数
        )

        # 将提示嵌入元组中的两个元素连接起来以兼容旧版本
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回连接后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的
    def encode_prompt(
        self,  # 当前实例的引用
        prompt,  # 要编码的提示文本
        device,  # 设备类型（例如 CPU 或 GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否进行无分类器引导
        negative_prompt=None,  # 可选的负面提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
        clip_skip: Optional[int] = None,  # 可选的剪辑跳过参数
    # 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.decode_latents 复制的
    def decode_latents(self, latents):  # 解码潜在变量的方法
        # 按照 VAE 配置的缩放因子调整潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents

        # 获取潜在变量的形状，分别表示批量大小、通道数、帧数、高度和宽度
        batch_size, channels, num_frames, height, width = latents.shape
        # 调整潜在变量的维度，以便于解码
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        # 使用 VAE 解码潜在变量并获取样本
        image = self.vae.decode(latents).sample
        # 将图像调整为视频格式，包含批量大小和帧数
        video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
        # 始终将视频转换为 float32 格式，以确保兼容性
        video = video.float()
        # 返回处理后的 video
        return video

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的
    # 定义准备额外参数的函数，用于调度器步骤
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外的关键字参数，因不同调度器的参数签名不同
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略该参数
            # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
            # eta 应在 [0, 1] 范围内
    
            # 检查调度器的步骤函数是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 创建一个空字典以存储额外参数
            extra_step_kwargs = {}
            # 如果调度器接受 eta，则将其添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤函数是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator，则将其添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 定义检查输入参数的函数
        def check_inputs(
            self,
            prompt,  # 输入提示文本
            strength,  # 强度参数
            callback_steps,  # 回调步骤数
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
            callback_on_step_end_tensor_inputs=None,  # 可选的步骤结束时的张量输入
    ):
        # 检查 strength 的值是否在有效范围 [0.0, 1.0] 之间
        if strength < 0 or strength > 1:
            # 如果不在范围内，抛出 ValueError 异常
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 检查 callback_steps 是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果不是，抛出 ValueError 异常
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 callback_on_step_end_tensor_inputs 是否在 _callback_tensor_inputs 中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有不在的项，抛出 ValueError 异常
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        # 检查是否同时提供了 prompt 和 prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            # 如果同时提供，抛出 ValueError 异常
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都未定义
        elif prompt is None and prompt_embeds is None:
            # 如果都未定义，抛出 ValueError 异常
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否正确
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果类型不正确，抛出 ValueError 异常
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 negative_prompt 和 negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果同时提供，抛出 ValueError 异常
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 的形状是否相同
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不匹配，抛出 ValueError 异常
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 中复制的函数
    # 定义获取时间步长的方法，输入为推理步数、强度和设备
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步长，取最小值以确保不超过总推理步数
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始的时间步长，确保不小于0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器获取时间步长，从t_start开始，按照调度器的顺序切片
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # 如果调度器有设置开始索引的方法，则设置为t_start
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        # 返回时间步长和有效推理步数
        return timesteps, num_inference_steps - t_start

    # 定义准备潜在表示的方法，输入为视频、时间步长、批大小、数据类型和设备
    def prepare_latents(self, video, timestep, batch_size, dtype, device, generator=None):
        # 将视频数据转移到指定的设备并转换数据类型
        video = video.to(device=device, dtype=dtype)

        # 改变视频的形状从 (b, c, f, h, w) 到 (b * f, c, w, h)
        bsz, channel, frames, width, height = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

        # 如果视频有4个通道，则初始化潜在表示为视频本身
        if video.shape[1] == 4:
            init_latents = video
        else:
            # 检查生成器是否是列表且其长度与批大小不匹配，抛出异常
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            # 如果生成器是列表，则逐个处理视频并获取潜在表示
            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(video[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                # 将所有潜在表示沿着第0维度连接
                init_latents = torch.cat(init_latents, dim=0)
            else:
                # 如果生成器不是列表，直接处理整个视频获取潜在表示
                init_latents = retrieve_latents(self.vae.encode(video), generator=generator)

            # 对潜在表示进行缩放
            init_latents = self.vae.config.scaling_factor * init_latents

        # 检查批大小是否大于潜在表示的数量且不可整除，抛出异常
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `video` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            # 将潜在表示扩展为批大小
            init_latents = torch.cat([init_latents], dim=0)

        # 获取潜在表示的形状
        shape = init_latents.shape
        # 生成噪声张量，用于添加到潜在表示中
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 获取潜在表示，添加噪声
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        # 重新调整潜在表示的形状以匹配原始视频的维度
        latents = latents[None, :].reshape((bsz, frames, latents.shape[1]) + latents.shape[2:]).permute(0, 2, 1, 3, 4)

        # 返回处理后的潜在表示
        return latents

    # 关闭梯度计算以节省内存
    @torch.no_grad()
    # 替换示例文档字符串为指定文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，支持多种输入参数
        def __call__(
            # 输入提示，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输入视频，可以是图像数组列表或张量
            video: Union[List[np.ndarray], torch.Tensor] = None,
            # 强度参数，默认值为0.6，控制某些特性
            strength: float = 0.6,
            # 推理步骤数量，默认为50，影响生成质量
            num_inference_steps: int = 50,
            # 引导尺度，默认值为15.0，控制生成的引导强度
            guidance_scale: float = 15.0,
            # 负面提示，可以是单个字符串或字符串列表，用于排除不想要的内容
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 额外参数，控制噪声，默认值为0.0
            eta: float = 0.0,
            # 随机生成器，可选，控制随机性
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在张量，可选，影响生成过程
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入张量，可选，直接使用预处理过的提示
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入张量，可选，直接使用预处理过的负面提示
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，可选，默认为"np"，指定返回格式
            output_type: Optional[str] = "np",
            # 是否返回字典格式，默认为True
            return_dict: bool = True,
            # 可选的回调函数，允许在特定步骤执行自定义逻辑
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调执行的步骤间隔，默认为1
            callback_steps: int = 1,
            # 交叉注意力的额外参数，可选，用于细化生成过程
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选，跳过的剪辑步骤
            clip_skip: Optional[int] = None,
```