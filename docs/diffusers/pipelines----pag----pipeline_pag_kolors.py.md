# `.\diffusers\pipelines\pag\pipeline_pag_kolors.py`

```py
# 版权所有 2024 Stability AI, Kwai-Kolors Team 和 The HuggingFace Team。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下位置获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，按照许可证分发的软件均按“原样”提供，
# 不提供任何形式的担保或条件，明示或暗示。
# 请参阅许可证以获取有关权限和限制的具体语言。
import inspect  # 导入 inspect 模块以便进行对象检查
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示以便于类型注释

import torch  # 导入 PyTorch 库
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # 从 transformers 导入图像处理器和模型

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 从回调模块导入多管道回调和单个管道回调
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从图像处理模块导入图像输入和变分自编码器图像处理器
from ...loaders import IPAdapterMixin, StableDiffusionXLLoraLoaderMixin  # 从加载器模块导入适配器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 从模型模块导入自编码器、图像投影和条件模型
from ...models.attention_processor import AttnProcessor2_0, FusedAttnProcessor2_0, XFormersAttnProcessor  # 导入注意力处理器
from ...schedulers import KarrasDiffusionSchedulers  # 从调度器模块导入 Karras 扩散调度器
from ...utils import is_torch_xla_available, logging, replace_example_docstring  # 从工具模块导入工具函数
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具模块导入随机张量生成函数
from ..kolors.pipeline_output import KolorsPipelineOutput  # 从 Kolors 模块导入管道输出类
from ..kolors.text_encoder import ChatGLMModel  # 从 Kolors 模块导入文本编码器模型
from ..kolors.tokenizer import ChatGLMTokenizer  # 从 Kolors 模块导入分词器
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从管道工具模块导入扩散管道和稳定扩散混合类
from .pag_utils import PAGMixin  # 从 PAG 工具模块导入 PAG 混合类

# 检查 XLA 是否可用，如果可用，则导入相关模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 模型相关功能

    XLA_AVAILABLE = True  # 标记 XLA 可用
else:
    XLA_AVAILABLE = False  # 标记 XLA 不可用

# 创建日志记录器实例，用于记录模块内的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，提供用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import AutoPipelineForText2Image  # 从 diffusers 导入自动文本转图像管道

        >>> pipe = AutoPipelineForText2Image.from_pretrained(  # 从预训练模型创建管道实例
        ...     "Kwai-Kolors/Kolors-diffusers",  # 指定模型名称
        ...     variant="fp16",  # 指定变体为 fp16
        ...     torch_dtype=torch.float16,  # 设置 PyTorch 数据类型为 float16
        ...     enable_pag=True,  # 启用 PAG 功能
        ...     pag_applied_layers=["down.block_2.attentions_1", "up.block_0.attentions_1"],  # 指定 PAG 应用层
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 CUDA 设备

        >>> prompt = (  # 定义提示文本
        ...     "A photo of a ladybug, macro, zoom, high quality, film, holding a wooden sign with the text 'KOLORS'"  # 提示内容
        ... )
        >>> image = pipe(prompt, guidance_scale=5.5, pag_scale=1.5).images[0]  # 生成图像并提取第一张
        ```py
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 导入的函数
def retrieve_timesteps(  # 定义函数以检索时间步
    scheduler,  # 调度器对象
    num_inference_steps: Optional[int] = None,  # 可选的推理步数
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备类型
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表
    sigmas: Optional[List[float]] = None,  # 可选的 sigma 值列表
    **kwargs,  # 其他可选参数
):
    """
    # 调用调度器的 `set_timesteps` 方法，并在调用后从调度器中获取时间步
    # 处理自定义时间步。任何关键字参数将被传递给 `scheduler.set_timesteps`。

    # 参数说明：
    # scheduler (`SchedulerMixin`):
    #     要从中获取时间步的调度器。
    # num_inference_steps (`int`):
    #     在使用预训练模型生成样本时使用的扩散步骤数量。如果使用此参数，则 `timesteps`
    #     必须为 `None`。
    # device (`str` 或 `torch.device`, *可选*):
    #     时间步应该移动到的设备。如果为 `None`，则时间步不会被移动。
    # timesteps (`List[int]`, *可选*):
    #     自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递 `timesteps`，
    #     则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
    # sigmas (`List[float]`, *可选*):
    #     自定义 sigmas，用于覆盖调度器的时间步间隔策略。如果传递 `sigmas`，
    #     则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    # 返回:
    #     `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是调度器的时间步调度，
    #     第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了 `timesteps` 和 `sigmas`
    if timesteps is not None and sigmas is not None:
        # 抛出错误，提示只能选择其中一个作为自定义值
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果提供了自定义时间步
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误提示
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法，传入自定义时间步及设备和关键字参数
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果提供了自定义 sigmas
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受 sigmas
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误提示
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法，传入自定义 sigmas 及设备和关键字参数
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果没有提供 `timesteps` 和 `sigmas`
    else:
        # 调用调度器的 `set_timesteps` 方法，传入推理步骤数量及设备和关键字参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤数量
    return timesteps, num_inference_steps
# 定义一个名为 KolorsPAGPipeline 的类，继承自多个基类，用于文本到图像的生成
class KolorsPAGPipeline(
    # 继承 DiffusionPipeline、StableDiffusionMixin、StableDiffusionXLLoraLoaderMixin、IPAdapterMixin 和 PAGMixin
    DiffusionPipeline, StableDiffusionMixin, StableDiffusionXLLoraLoaderMixin, IPAdapterMixin, PAGMixin
):
    r"""
    使用 Kolors 进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解库为所有管道实现的通用方法（如下载或保存、在特定设备上运行等）。

    该管道还继承了以下加载方法：
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。
        text_encoder ([`ChatGLMModel`]):
            冻结的文本编码器。Kolors 使用 [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b)。
        tokenizer (`ChatGLMTokenizer`):
            类的标记器
            [ChatGLMTokenizer](https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py)。
        unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码的图像潜在表示。
        scheduler ([`SchedulerMixin`]):
            结合 `unet` 使用的调度器，用于去噪编码的图像潜在表示。可以是
            [`DDIMScheduler`]、[`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"False"`):
            是否强制负提示嵌入始终设置为 0。另请参见 `Kwai-Kolors/Kolors-diffusers` 的配置。
        pag_applied_layers (`str` or `List[str]``, *optional*, defaults to `"mid"`):
            设置要应用扰动注意力引导的变压器注意力层。可以是字符串或字符串列表，包含 "down"、"mid"、"up"，或整个变压器块或特定变压器块注意力层，例如：
                ["mid"] ["down", "mid"] ["down", "mid", "up.block_1"] ["down", "mid", "up.block_1.attentions_0",
                "up.block_1.attentions_1"]
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件的列表，这些组件在实例化时可以选择性提供
    _optional_components = [
        "image_encoder",
        "feature_extractor",
    ]
    # 定义需要作为张量输入的回调列表，这些输入将用于后续的处理
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]
    # 初始化方法，用于设置类的初始状态
        def __init__(
            self,
            vae: AutoencoderKL,  # VAE模型，用于图像编码
            text_encoder: ChatGLMModel,  # 文本编码器模型，用于处理输入文本
            tokenizer: ChatGLMTokenizer,  # 令牌化工具，将文本转为模型可处理的格式
            unet: UNet2DConditionModel,  # UNet模型，用于生成图像
            scheduler: KarrasDiffusionSchedulers,  # 调度器，用于管理采样过程
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器
            feature_extractor: CLIPImageProcessor = None,  # 可选的特征提取器
            force_zeros_for_empty_prompt: bool = False,  # 是否强制为空提示时输出零
            pag_applied_layers: Union[str, List[str]] = "mid",  # 应用的层，默认设置为“mid”
        ):
            super().__init__()  # 调用父类的初始化方法
    
            # 注册多个模块，以便于管理和访问
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
            # 将配置参数注册到类中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 根据 VAE 的配置计算缩放因子
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
            # 创建图像处理器实例，使用 VAE 的缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 设置默认的样本大小，从 UNet 配置中获取
            self.default_sample_size = self.unet.config.sample_size
    
            # 设置应用的层
            self.set_pag_applied_layers(pag_applied_layers)
    
        # 从其他管道复制的 encode_prompt 方法，用于编码提示
        def encode_prompt(
            self,
            prompt,  # 输入提示文本
            device: Optional[torch.device] = None,  # 可选的设备参数，用于模型计算
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool = True,  # 是否进行无分类器引导
            negative_prompt=None,  # 可选的负提示文本
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化提示嵌入
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 可选的负提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负池化提示嵌入
            max_sequence_length: int = 256,  # 最大序列长度
        # 从其他管道复制的 encode_image 方法
    # 定义一个编码图像的函数，接收图像、设备、每个提示的图像数量和可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，则通过特征提取器处理并返回张量格式
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移到指定设备并转换为适当的数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 获取编码后的隐藏状态的倒数第二个层
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 复制隐藏状态以适应每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于无条件图像，使用全零张量获取隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 复制无条件隐藏状态以适应每个提示的图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码图像和无条件图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 获取编码后的图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 复制图像嵌入以适应每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入形状相同的全零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回编码图像和无条件图像的嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的函数
        def prepare_ip_adapter_image_embeds(
            # 定义输入适配器图像、适配器图像嵌入、设备、每个提示的图像数量和是否进行无分类器引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化图像嵌入列表
        image_embeds = []
        # 如果启用分类器自由引导，初始化负图像嵌入列表
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器的图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 确保输入的图像是列表形式
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入图像的数量与适配器数量是否匹配
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误提示输入图像数量不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个单独的输入适配器图像和对应的投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 确定输出是否为隐藏状态
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 对图像进行编码，返回嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到图像嵌入列表
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用分类器自由引导，将负图像嵌入添加到列表
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历已存在的图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用分类器自由引导，将图像嵌入拆分为负和正
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表
                image_embeds.append(single_image_embeds)

        # 初始化适配器图像嵌入列表
        ip_adapter_image_embeds = []
        # 遍历图像嵌入并处理
        for i, single_image_embeds in enumerate(image_embeds):
            # 根据提示的图像数量重复图像嵌入
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用分类器自由引导，处理负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到适配器图像嵌入列表
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回适配器图像嵌入
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的代码
    # 准备额外的参数用于调度器的步骤，因为并非所有调度器的签名相同
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta（η）仅在 DDIMScheduler 中使用，对其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 其值应在 [0, 1] 之间
    
            # 检查调度器的步骤方法是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外步骤参数字典
            extra_step_kwargs = {}
            # 如果调度器接受 eta，则将其添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤方法是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator，则将其添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外步骤参数字典
            return extra_step_kwargs
    
        # 从 diffusers.pipelines.kolors.pipeline_kolors.KolorsPipeline.check_inputs 复制的函数
        def check_inputs(
            self,
            prompt,
            num_inference_steps,
            height,
            width,
            negative_prompt=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=None,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的函数
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜变量的形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查传入的 generator 列表长度是否与批量大小一致
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"您传入的生成器列表长度为 {len(generator)}，请求的有效批量大小为 {batch_size}。"
                    f" 确保批量大小与生成器长度一致。"
                )
    
            # 如果潜变量为 None，则生成随机潜变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果潜变量不为 None，将其移至指定设备
                latents = latents.to(device)
    
            # 按调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜变量
            return latents
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids 复制的函数
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # 创建一个包含原始尺寸、裁剪左上角坐标和目标尺寸的列表
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        # 计算传入的附加时间嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取模型期望的附加嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 如果期望的嵌入维度与实际的不同，则抛出错误
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将附加时间 ID 转换为张量，指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 返回附加时间 ID 张量
        return add_time_ids

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.upcast_vae 复制的函数
    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 类型
        self.vae.to(dtype=torch.float32)
        # 检查是否使用了 torch 2.0 或 xformers，确定注意力处理器的类型
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # 如果使用了 xformers 或 torch 2.0，则注意力块不需要是 float32，这样可以节省大量内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层转换为指定的数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将输入卷积层转换为指定的数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将中间块转换为指定的数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding 复制的函数
    def get_guidance_scale_embedding(
        # 输入的张量 w 和嵌入维度以及数据类型的默认值
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    # 该函数返回生成的嵌入向量，形状为 (len(w), embedding_dim)
    ) -> torch.Tensor:
            """
            # 引用文档，详细说明函数的实现和参数
            See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            Args:
                w (`torch.Tensor`):
                    # 用于生成嵌入向量的指导权重
                    Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
                embedding_dim (`int`, *optional*, defaults to 512):
                    # 生成的嵌入维度，默认值为512
                    Dimension of the embeddings to generate.
                dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                    # 生成的嵌入的数据类型，默认为torch.float32
                    Data type of the generated embeddings.
    
            Returns:
                `torch.Tensor`: # 返回嵌入向量
                Embedding vectors with shape `(len(w), embedding_dim)`.
            """
            # 确保输入的 w 只有一维
            assert len(w.shape) == 1
            # 将 w 的值扩大1000倍
            w = w * 1000.0
    
            # 计算嵌入的一半维度
            half_dim = embedding_dim // 2
            # 计算缩放因子的对数
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 计算嵌入的指数值
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 将 w 转换为指定 dtype 并进行广播
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 组合正弦和余弦函数的嵌入
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度是奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保生成的嵌入形状符合预期
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回生成的嵌入
            return emb
    
        # 返回指导尺度属性
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 定义指导尺度，与Imagen论文中的指导权重 w 类似
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # 对应于不执行分类器自由指导。
        @property
        def do_classifier_free_guidance(self):
            # 返回是否进行分类器自由指导的布尔值
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 返回交叉注意力的关键字参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 返回去噪结束的属性
        @property
        def denoising_end(self):
            return self._denoising_end
    
        # 返回时间步数的属性
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 返回中断的属性
        @property
        def interrupt(self):
            return self._interrupt
    
        # 禁用梯度计算以提高效率
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义调用方法，允许通过实例调用该类
        def __call__(
            # 输入提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 生成图像的高度，可选
            height: Optional[int] = None,
            # 生成图像的宽度，可选
            width: Optional[int] = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 时间步的列表，可选
            timesteps: List[int] = None,
            # sigma 值的列表，可选
            sigmas: List[float] = None,
            # 去噪结束值，可选
            denoising_end: Optional[float] = None,
            # 引导缩放因子，默认为 5.0
            guidance_scale: float = 5.0,
            # 负面提示，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # eta 值，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可选
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在表示，可选
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，可选
            prompt_embeds: Optional[torch.Tensor] = None,
            # 处理过的提示嵌入，可选
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入，可选
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 处理过的负面提示嵌入，可选
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # IP 适配器图像，可选
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # IP 适配器图像嵌入的列表，可选
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 跨注意力参数的字典，可选
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 原始图像大小，可选
            original_size: Optional[Tuple[int, int]] = None,
            # 裁剪坐标的左上角，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标大小，可选
            target_size: Optional[Tuple[int, int]] = None,
            # 负面图像的原始大小，可选
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负面裁剪坐标的左上角，默认为 (0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负面目标大小，可选
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 在步骤结束时的回调函数，可选
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 在步骤结束时的张量输入回调名称，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # PAG 缩放因子，默认为 3.0
            pag_scale: float = 3.0,
            # 自适应 PAG 缩放，默认为 0.0
            pag_adaptive_scale: float = 0.0,
            # 最大序列长度，默认为 256
            max_sequence_length: int = 256,
```