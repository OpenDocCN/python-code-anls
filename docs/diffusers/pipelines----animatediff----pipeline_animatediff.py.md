# `.\diffusers\pipelines\animatediff\pipeline_animatediff.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按照许可证分发的软件
# 是按“原样”基础分发的，没有任何形式的担保或条件，
# 明示或暗示。
# 请参阅许可证，以了解治理权限和
# 许可证的限制。

import inspect  # 导入 inspect 模块，用于获取活跃对象的信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示所需的类型

import torch  # 导入 PyTorch 库，用于张量计算
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入 transformers 库中的 CLIP 相关类

from ...image_processor import PipelineImageInput  # 从图像处理模块导入 PipelineImageInput 类
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入混合类，用于适配不同加载器
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel  # 导入各种模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整文本编码器 LoRA 比例的函数
from ...models.unets.unet_motion_model import MotionAdapter  # 从 UNet 动作模型导入 MotionAdapter 类
from ...schedulers import (  # 导入不同调度器类
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import (  # 从 utils 模块导入常用工具
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具模块导入随机张量生成函数
from ...video_processor import VideoProcessor  # 导入视频处理类
from ..free_init_utils import FreeInitMixin  # 导入自由初始化混合类
from ..free_noise_utils import AnimateDiffFreeNoiseMixin  # 导入动画差异自由噪声混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道及其混合类
from .pipeline_output import AnimateDiffPipelineOutput  # 导入动画差异管道输出类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，供后续使用

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，以展示使用方法
    Examples:  # 示例部分的起始
        ```py  # 示例代码块开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler  # 导入相关类
        >>> from diffusers.utils import export_to_gif  # 导入 GIF 导出工具

        >>> adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")  # 加载预训练的动作适配器
        >>> pipe = AnimateDiffPipeline.from_pretrained("frankjoshua/toonyou_beta6", motion_adapter=adapter)  # 加载动画差异管道并设置动作适配器
        >>> pipe.scheduler = DDIMScheduler(beta_schedule="linear", steps_offset=1, clip_sample=False)  # 设置调度器为 DDIM，并配置参数
        >>> output = pipe(prompt="A corgi walking in the park")  # 生成输出，输入提示文本
        >>> frames = output.frames[0]  # 提取第一帧
        >>> export_to_gif(frames, "animation.gif")  # 将帧导出为 GIF 文件
        ```py  # 示例代码块结束
"""


class AnimateDiffPipeline(  # 定义 AnimateDiffPipeline 类
    DiffusionPipeline,  # 继承自扩散管道类
    StableDiffusionMixin,  # 混合稳定扩散功能
    TextualInversionLoaderMixin,  # 混合文本反演加载功能
    IPAdapterMixin,  # 混合图像处理适配功能
    StableDiffusionLoraLoaderMixin,  # 混合稳定扩散 LoRA 加载功能
    FreeInitMixin,  # 混合自由初始化功能
    AnimateDiffFreeNoiseMixin,  # 混合动画差异自由噪声功能
):
    r"""  # 定义类文档字符串
    Pipeline for text-to-video generation.  # 描述该类为文本到视频生成的管道

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods  # 指明该模型继承自扩散管道，并建议查看父类文档
```  # 文档字符串结束
    # 该管道实现了所有管道操作（下载、保存、在特定设备上运行等）。

    # 此管道还继承了以下加载方法：
        # [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        # [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        # [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        # [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    # 参数：
        # vae ([`AutoencoderKL`])：变分自编码器 (VAE) 模型，用于编码和解码图像到潜在表示。
        # text_encoder ([`CLIPTextModel`] )：冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        # tokenizer (`CLIPTokenizer`)：用于对文本进行标记化的 [`~transformers.CLIPTokenizer`]。
        # unet ([`UNet2DConditionModel`] )：用于创建 UNetMotionModel 以去噪编码的视频潜在表示的 [`UNet2DConditionModel`]。
        # motion_adapter ([`MotionAdapter`] )：与 `unet` 结合使用的 [`MotionAdapter`]，用于去噪编码的视频潜在表示。
        # scheduler ([`SchedulerMixin`] )：与 `unet` 结合使用的调度器，用于去噪编码的图像潜在表示。可以是
            # [`DDIMScheduler`]，[`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
    # """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 可选组件的列表
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]
    # 回调张量输入的列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # 初始化方法
    def __init__(
        # 初始化所需的变分自编码器
        self,
        vae: AutoencoderKL,
        # 初始化所需的文本编码器
        text_encoder: CLIPTextModel,
        # 初始化所需的标记器
        tokenizer: CLIPTokenizer,
        # 初始化所需的 UNet 模型（可以是 UNet2DConditionModel 或 UNetMotionModel）
        unet: Union[UNet2DConditionModel, UNetMotionModel],
        # 初始化所需的运动适配器
        motion_adapter: MotionAdapter,
        # 初始化所需的调度器（多种选择）
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        # 可选的特征提取器
        feature_extractor: CLIPImageProcessor = None,
        # 可选的图像编码器
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        # 调用父类初始化方法
        super().__init__()
        # 如果 unet 是 UNet2DConditionModel，则将其转换为 UNetMotionModel
        if isinstance(unet, UNet2DConditionModel):
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

        # 注册所有模块，设置相应的属性
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            motion_adapter=motion_adapter,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建视频处理器实例，设置不缩放和 VAE 缩放因子
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的代码，修改了参数 num_images_per_prompt 为 num_videos_per_prompt
    def encode_prompt(
        self,  # 方法所属的类实例
        prompt,  # 输入的提示文本
        device,  # 计算设备（如 GPU 或 CPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 负向提示文本，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负向提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 LORA 缩放因子
        clip_skip: Optional[int] = None,  # 可选的剪辑跳过值
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制的代码
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):  # 方法用于编码图像
        dtype = next(self.image_encoder.parameters()).dtype  # 获取图像编码器参数的数据类型

        if not isinstance(image, torch.Tensor):  # 检查图像是否为 Tensor
            image = self.feature_extractor(image, return_tensors="pt").pixel_values  # 使用特征提取器处理图像并转换为 Tensor

        image = image.to(device=device, dtype=dtype)  # 将图像移动到指定设备并转换为正确的数据类型
        if output_hidden_states:  # 如果要求输出隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]  # 获取倒数第二个隐藏状态
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)  # 根据每个提示的图像数量重复隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(  # 对全零图像编码以获取无条件隐藏状态
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]  # 获取倒数第二个隐藏状态
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(  # 根据每个提示的图像数量重复无条件隐藏状态
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states  # 返回图像和无条件的隐藏状态
        else:  # 如果不要求输出隐藏状态
            image_embeds = self.image_encoder(image).image_embeds  # 编码图像并获取图像嵌入
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)  # 根据每个提示的图像数量重复图像嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)  # 创建与图像嵌入形状相同的全零张量作为无条件嵌入

            return image_embeds, uncond_image_embeds  # 返回图像嵌入和无条件嵌入

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的代码
    def prepare_ip_adapter_image_embeds(  # 方法用于准备图像适配器的图像嵌入
        self,  # 方法所属的类实例
        ip_adapter_image,  # 输入的适配器图像
        ip_adapter_image_embeds,  # 输入的适配器图像嵌入
        device,  # 计算设备
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance  # 是否使用无分类器引导
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用了分类器自由引导，则初始化一个空列表用于负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 检查 ip_adapter_image_embeds 是否为 None
        if ip_adapter_image_embeds is None:
            # 如果 ip_adapter_image 不是列表，则将其转换为单元素列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 确保 ip_adapter_image 的长度与 IP 适配器的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误，提示 ip_adapter_image 的长度与 IP 适配器数量不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历 ip_adapter_image 和对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 确定是否需要输出隐藏状态
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个 IP 适配器图像，获取嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到列表中，使用 None 维度增加维度
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用了分类器自由引导，则将负图像嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果 ip_adapter_image_embeds 已定义，直接使用其中的图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用了分类器自由引导，则分割正负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将正图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储最终的 IP 适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历每个图像嵌入及其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将每个图像嵌入复制 num_images_per_prompt 次
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用了分类器自由引导，则复制负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入和正图像嵌入连接在一起
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到最终列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回最终的 IP 适配器图像嵌入列表
        return ip_adapter_image_embeds
    # 解码潜在向量的函数，接受潜在向量和解码块大小作为参数
        def decode_latents(self, latents, decode_chunk_size: int = 16):
            # 根据 VAE 配置的缩放因子调整潜在向量的值
            latents = 1 / self.vae.config.scaling_factor * latents
    
            # 获取潜在向量的形状，分别为批量大小、通道数、帧数、高度和宽度
            batch_size, channels, num_frames, height, width = latents.shape
            # 重新排列潜在向量的维度，并调整形状以适应解码过程
            latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    
            # 用于存储解码后的帧数据
            video = []
            # 按照指定的解码块大小进行迭代处理潜在向量
            for i in range(0, latents.shape[0], decode_chunk_size):
                # 选取当前块的潜在向量进行解码
                batch_latents = latents[i : i + decode_chunk_size]
                # 调用 VAE 的解码器进行解码，并提取样本数据
                batch_latents = self.vae.decode(batch_latents).sample
                # 将解码后的帧添加到视频列表中
                video.append(batch_latents)
    
            # 将所有解码帧在第一个维度上连接
            video = torch.cat(video)
            # 调整视频的形状以匹配批量大小和帧数，重新排列维度
            video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
            # 始终将视频数据转换为 float32 类型，以保持兼容性且不会造成显著开销
            video = video.float()
            # 返回解码后的视频数据
            return video
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的函数
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为不同调度器的参数签名不同
            # eta（η）仅在 DDIMScheduler 中使用，其他调度器会忽略
            # eta 对应于 DDIM 论文中的 η，范围应在 [0, 1] 之间
    
            # 检查调度器的步骤函数是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            # 如果接受 eta 参数，将其添加到字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤函数是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator 参数，将其添加到字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs 复制的函数
        def check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            callback_on_step_end_tensor_inputs=None,
        # 准备潜在向量的函数，接受多个参数
        def prepare_latents(
            self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 如果启用了 FreeNoise，根据 [FreeNoise](https://arxiv.org/abs/2310.15169) 的公式 (7) 生成潜变量
        if self.free_noise_enabled:
            # 准备 FreeNoise 模式下的潜变量，传入相关参数
            latents = self._prepare_latents_free_noise(
                batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents
            )

        # 检查生成器列表长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不匹配，则引发值错误，提示用户
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 定义潜变量的形状，包含批量大小和其他参数
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        # 如果潜变量为 None，则生成随机潜变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果潜变量已存在，将其转移到指定设备
            latents = latents.to(device)

        # 按照调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回生成的潜变量
        return latents

    @property
    # 返回当前的引导比例
    def guidance_scale(self):
        return self._guidance_scale

    @property
    # 返回当前的剪切跳过值
    def clip_skip(self):
        return self._clip_skip

    # 在此定义 `guidance_scale`，其类似于 Imagen 论文中公式 (2) 的引导权重 `w`： https://arxiv.org/pdf/2205.11487.pdf
    # `guidance_scale = 1` 表示不进行分类器自由引导。
    @property
    # 判断是否进行分类器自由引导，基于引导比例是否大于 1
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    # 返回跨注意力的参数
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    # 返回时间步数的数量
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，允许使用不同参数生成视频或图像
        def __call__(
            # 提供输入提示，类型为字符串或字符串列表，默认为 None
            self,
            prompt: Union[str, List[str]] = None,
            # 设置生成的帧数，默认为 16
            num_frames: Optional[int] = 16,
            # 设置生成图像的高度，默认为 None
            height: Optional[int] = None,
            # 设置生成图像的宽度，默认为 None
            width: Optional[int] = None,
            # 指定推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 设置引导比例，默认为 7.5
            guidance_scale: float = 7.5,
            # 提供负向提示，类型为字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 设置每个提示生成的视频数量，默认为 1
            num_videos_per_prompt: Optional[int] = 1,
            # 设置随机性参数，默认为 0.0
            eta: float = 0.0,
            # 提供随机数生成器，类型为 torch.Generator 或其列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 提供潜在变量，类型为 torch.Tensor，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提供提示的嵌入表示，类型为 torch.Tensor，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 提供负向提示的嵌入表示，类型为 torch.Tensor，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 提供适配器图像，类型为 PipelineImageInput，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 提供适配器图像的嵌入表示，类型为 torch.Tensor 列表，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 设置输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 指定是否返回字典格式，默认为 True
            return_dict: bool = True,
            # 提供交叉注意力的额外参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指定跳过的剪辑层数，默认为 None
            clip_skip: Optional[int] = None,
            # 提供在步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 指定在步骤结束时的张量输入回调，默认为包含 "latents" 的列表
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 设置解码块的大小，默认为 16
            decode_chunk_size: int = 16,
            # 接受任意其他关键字参数
            **kwargs,
```