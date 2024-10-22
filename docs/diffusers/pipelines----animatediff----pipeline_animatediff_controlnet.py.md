# `.\diffusers\pipelines\animatediff\pipeline_animatediff_controlnet.py`

```py
# 版权信息，声明版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则不得使用本文件。
# 可通过以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有规定，软件
# 在“按原样”基础上分发，不提供任何形式的担保或条件。
# 请参阅许可证了解有关权限和
# 限制的具体语言。

# 导入 inspect 模块，用于检查对象
import inspect
# 从 typing 模块导入类型注释所需的类
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能性操作模块
import torch.nn.functional as F
# 从 transformers 库导入与 CLIP 相关的模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 从图像处理模块导入 PipelineImageInput 类
from ...image_processor import PipelineImageInput
# 从 loaders 模块导入多个混合类
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从 models 模块导入多个模型类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel, UNetMotionModel
# 从 LoRA 模块导入调整文本编码器的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从 UNet 动作模型模块导入 MotionAdapter 类
from ...models.unets.unet_motion_model import MotionAdapter
# 从调度器模块导入 KarrasDiffusionSchedulers 类
from ...schedulers import KarrasDiffusionSchedulers
# 从工具模块导入多个实用函数
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
# 从 PyTorch 工具模块导入一些辅助函数
from ...utils.torch_utils import is_compiled_module, randn_tensor
# 从视频处理模块导入 VideoProcessor 类
from ...video_processor import VideoProcessor
# 从控制网络模块导入 MultiControlNetModel 类
from ..controlnet.multicontrolnet import MultiControlNetModel
# 从免费初始化工具模块导入 FreeInitMixin 类
from ..free_init_utils import FreeInitMixin
# 从免费噪声工具模块导入 AnimateDiffFreeNoiseMixin 类
from ..free_noise_utils import AnimateDiffFreeNoiseMixin
# 从管道工具模块导入 DiffusionPipeline 和 StableDiffusionMixin 类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从管道输出模块导入 AnimateDiffPipelineOutput 类
from .pipeline_output import AnimateDiffPipelineOutput

# 获取当前模块的日志记录器实例
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，当前为空
EXAMPLE_DOC_STRING = """
"""

# 定义 AnimateDiffControlNetPipeline 类，继承多个混合类
class AnimateDiffControlNetPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    FreeInitMixin,
    AnimateDiffFreeNoiseMixin,
):
    r"""
    用于基于 ControlNet 指导的文本到视频生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档了解所有管道的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    # 定义参数文档字符串，描述每个参数的用途和类型
        Args:
            vae ([`AutoencoderKL`]):
                Variational Auto-Encoder (VAE) 模型，用于对图像进行编码和解码，转换为潜在表示。
            text_encoder ([`CLIPTextModel`]):
                冻结的文本编码器，使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 模型。
            tokenizer (`CLIPTokenizer`):
                一个 [`~transformers.CLIPTokenizer`] 用于对文本进行分词。
            unet ([`UNet2DConditionModel`]):
                一个 [`UNet2DConditionModel`]，用于创建 UNetMotionModel 来去噪编码的视频潜在表示。
            motion_adapter ([`MotionAdapter`]):
                一个 [`MotionAdapter`]，与 `unet` 一起使用，以去噪编码的视频潜在表示。
            scheduler ([`SchedulerMixin`]):
                一个调度器，与 `unet` 一起使用，以去噪编码的图像潜在表示。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的任意一种。
        """
    
        # 定义模型的 CPU 卸载顺序
        model_cpu_offload_seq = "text_encoder->unet->vae"
        # 可选组件列表
        _optional_components = ["feature_extractor", "image_encoder"]
        # 注册的张量输入列表
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    
        # 初始化方法，设置模型的各个组件
        def __init__(
            self,
            vae: AutoencoderKL,  # VAE 模型
            text_encoder: CLIPTextModel,  # 文本编码器
            tokenizer: CLIPTokenizer,  # 文本分词器
            unet: Union[UNet2DConditionModel, UNetMotionModel],  # UNet 模型
            motion_adapter: MotionAdapter,  # 动作适配器
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],  # 控制网络模型
            scheduler: KarrasDiffusionSchedulers,  # 调度器
            feature_extractor: Optional[CLIPImageProcessor] = None,  # 可选特征提取器
            image_encoder: Optional[CLIPVisionModelWithProjection] = None,  # 可选图像编码器
        ):
            # 调用父类的初始化方法
            super().__init__()
            # 检查 UNet 类型，如果是 UNet2DConditionModel，则转换为 UNetMotionModel
            if isinstance(unet, UNet2DConditionModel):
                unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
    
            # 如果 controlnet 是列表或元组，转换为 MultiControlNetModel
            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)
    
            # 注册各个模型模块
            self.register_modules(
                vae=vae,  # 注册 VAE 模型
                text_encoder=text_encoder,  # 注册文本编码器
                tokenizer=tokenizer,  # 注册文本分词器
                unet=unet,  # 注册 UNet 模型
                motion_adapter=motion_adapter,  # 注册动作适配器
                controlnet=controlnet,  # 注册控制网络模型
                scheduler=scheduler,  # 注册调度器
                feature_extractor=feature_extractor,  # 注册特征提取器
                image_encoder=image_encoder,  # 注册图像编码器
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建视频处理器，使用 VAE 缩放因子
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor)
            # 创建控制视频处理器，带 RGB 转换和归一化选项
            self.control_video_processor = VideoProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制，修改 num_images_per_prompt 为 num_videos_per_prompt
    # 定义一个编码提示的函数，接受多个参数以进行处理
    def encode_prompt(
        self,
        prompt,  # 要编码的提示文本
        device,  # 设备类型（CPU或GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否执行无分类器引导
        negative_prompt=None,  # 可选的负提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
        clip_skip: Optional[int] = None,  # 可选的跳过剪辑的层数
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制的函数
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):  # 定义图像编码函数
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入图像不是张量，则使用特征提取器进行转换
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像数据移到指定设备并转换为正确的数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 编码图像并获取倒数第二层的隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 重复隐藏状态以匹配生成的图像数量
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 对于无条件输入，生成一个与图像形状相同的全零张量
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 同样重复无条件隐藏状态
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回有条件和无条件的隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 如果不需要输出隐藏状态，则直接编码图像
            image_embeds = self.image_encoder(image).image_embeds
            # 重复编码嵌入以匹配生成的图像数量
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建一个与编码嵌入形状相同的全零张量作为无条件嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回有条件和无条件的嵌入
            return image_embeds, uncond_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的函数
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance  # 定义一个函数以准备 IP 适配器的图像嵌入
    ):  # 函数定义结束，开始函数体
        image_embeds = []  # 初始化用于存储图像嵌入的列表
        if do_classifier_free_guidance:  # 检查是否使用无分类器引导
            negative_image_embeds = []  # 初始化用于存储负图像嵌入的列表
        if ip_adapter_image_embeds is None:  # 检查输入适配器图像嵌入是否为 None
            if not isinstance(ip_adapter_image, list):  # 检查输入图像是否为列表
                ip_adapter_image = [ip_adapter_image]  # 将单个图像转换为列表

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):  # 检查图像数量与适配器数量是否匹配
                raise ValueError(  # 抛出值错误
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."  # 错误信息说明
                )

            for single_ip_adapter_image, image_proj_layer in zip(  # 遍历每个图像和对应的投影层
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers  # 通过 zip 函数组合图像和投影层
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)  # 判断是否需要输出隐藏状态
                single_image_embeds, single_negative_image_embeds = self.encode_image(  # 调用 encode_image 函数获取图像嵌入
                    single_ip_adapter_image, device, 1, output_hidden_state  # 传递图像及相关参数
                )

                image_embeds.append(single_image_embeds[None, :])  # 将单个图像嵌入添加到列表中
                if do_classifier_free_guidance:  # 如果使用无分类器引导
                    negative_image_embeds.append(single_negative_image_embeds[None, :])  # 将负图像嵌入添加到列表中
        else:  # 如果输入的适配器图像嵌入不为 None
            for single_image_embeds in ip_adapter_image_embeds:  # 遍历每个输入图像嵌入
                if do_classifier_free_guidance:  # 如果使用无分类器引导
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)  # 拆分为负和正图像嵌入
                    negative_image_embeds.append(single_negative_image_embeds)  # 将负图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)  # 将正图像嵌入添加到列表中

        ip_adapter_image_embeds = []  # 初始化用于存储适配器图像嵌入的列表
        for i, single_image_embeds in enumerate(image_embeds):  # 遍历图像嵌入及其索引
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)  # 复制图像嵌入
            if do_classifier_free_guidance:  # 如果使用无分类器引导
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)  # 复制负图像嵌入
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)  # 合并负和正图像嵌入

            single_image_embeds = single_image_embeds.to(device=device)  # 将图像嵌入移动到指定设备
            ip_adapter_image_embeds.append(single_image_embeds)  # 将处理后的图像嵌入添加到列表中

        return ip_adapter_image_embeds  # 返回适配器图像嵌入列表

    # Copied from diffusers.pipelines.animatediff.pipeline_animatediff.AnimateDiffPipeline.decode_latents  # 复制自其他模块的解码函数
    # 解码潜在向量，将其转换为视频数据
    def decode_latents(self, latents, decode_chunk_size: int = 16):
        # 根据 VAE 的缩放因子调整潜在向量
        latents = 1 / self.vae.config.scaling_factor * latents
    
        # 获取潜在向量的批量大小、通道数、帧数、高度和宽度
        batch_size, channels, num_frames, height, width = latents.shape
        # 重排和调整潜在向量的形状以适应解码器输入
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    
        # 初始化视频列表以存储解码后的帧
        video = []
        # 按解码块大小遍历潜在向量
        for i in range(0, latents.shape[0], decode_chunk_size):
            # 选择当前解码块的潜在向量
            batch_latents = latents[i : i + decode_chunk_size]
            # 解码当前潜在向量，并获取样本
            batch_latents = self.vae.decode(batch_latents).sample
            # 将解码结果添加到视频列表中
            video.append(batch_latents)
    
        # 将所有解码帧合并为一个张量
        video = torch.cat(video)
        # 调整视频张量的形状以匹配期望的输出格式
        video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
        # 将视频数据转换为 float32 格式以确保兼容性
        video = video.float()
        # 返回解码后的视频数据
        return video
    
    # 从稳定扩散管道复制的准备额外步骤参数的函数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的参数签名
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将被忽略
        # eta 在 DDIM 论文中对应于 η，范围应在 [0, 1] 之间
    
        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    # 检查输入参数的函数
    def check_inputs(
        self,
        prompt,
        height,
        width,
        num_frames,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        video=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        # 从动画扩散管道复制的准备潜在向量的函数
        def prepare_latents(
            self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 如果启用了 FreeNoise，按照 [FreeNoise](https://arxiv.org/abs/2310.15169) 的公式 (7) 生成潜在变量
        if self.free_noise_enabled:
            # 准备潜在变量，使用 FreeNoise 方法
            latents = self._prepare_latents_free_noise(
                batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents
            )

        # 检查生成器是否为列表，并且长度是否与批大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果生成器长度与请求的有效批大小不符，抛出错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 定义潜在变量的形状
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        # 如果潜在变量为 None，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将现有的潜在变量移动到指定的设备
            latents = latents.to(device)

        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回生成的潜在变量
        return latents

    def prepare_video(
        self,
        video,
        width,
        height,
        batch_size,
        num_videos_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        # 对输入视频进行预处理，调整大小并转换为指定的数据类型
        video = self.control_video_processor.preprocess_video(video, height=height, width=width).to(
            dtype=torch.float32
        )
        # 重新排列视频的维度，并将前两个维度扁平化
        video = video.permute(0, 2, 1, 3, 4).flatten(0, 1)
        # 获取视频的批大小
        video_batch_size = video.shape[0]

        # 根据视频批大小确定重复次数
        if video_batch_size == 1:
            repeat_by = batch_size
        else:
            # 如果视频批大小与提示批大小相同，则重复次数为每个提示的视频数量
            repeat_by = num_videos_per_prompt

        # 根据计算的重复次数重复视频数据
        video = video.repeat_interleave(repeat_by, dim=0)
        # 将视频移动到指定的设备和数据类型
        video = video.to(device=device, dtype=dtype)

        # 如果启用无分类器引导且不处于猜测模式，则将视频重复拼接
        if do_classifier_free_guidance and not guess_mode:
            video = torch.cat([video] * 2)

        # 返回处理后的视频
        return video

    @property
    def guidance_scale(self):
        # 返回引导缩放因子
        return self._guidance_scale

    @property
    def clip_skip(self):
        # 返回跳过的剪辑数
        return self._clip_skip

    # 此处 `guidance_scale` 定义类似于 Imagen 论文中公式 (2) 的引导权重 `w`
    # https://arxiv.org/pdf/2205.11487.pdf 。 `guidance_scale = 1`
    # 表示不进行无分类器引导。
    @property
    def do_classifier_free_guidance(self):
        # 如果引导缩放因子大于 1，则启用无分类器引导
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        # 返回交叉注意力的关键字参数
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        # 返回时间步数
        return self._num_timesteps

    @torch.no_grad()
    # 定义可调用方法，用于执行某些操作
        def __call__(
            # 提示文本，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 帧数，默认为 16
            num_frames: Optional[int] = 16,
            # 输出视频的高度，默认为 None
            height: Optional[int] = None,
            # 输出视频的宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤数，默认为 50
            num_inference_steps: int = 50,
            # 指导比例，默认为 7.5
            guidance_scale: float = 7.5,
            # 负提示文本，可以是单个字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的视频数量，默认为 1
            num_videos_per_prompt: Optional[int] = 1,
            # 额外参数 eta，默认为 0.0
            eta: float = 0.0,
            # 随机生成器，可以是单个或列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在张量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像嵌入，默认为 None
            ip_adapter_image_embeds: Optional[PipelineImageInput] = None,
            # 条件帧的列表，默认为 None
            conditioning_frames: Optional[List[PipelineImageInput]] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 跨注意力参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 控制网络的条件比例，默认为 1.0
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # 猜测模式，默认为 False
            guess_mode: bool = False,
            # 控制指导开始比例，默认为 0.0
            control_guidance_start: Union[float, List[float]] = 0.0,
            # 控制指导结束比例，默认为 1.0
            control_guidance_end: Union[float, List[float]] = 1.0,
            # 跳过的剪辑，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 步骤结束时张量输入的回调，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 解码块大小，默认为 16
            decode_chunk_size: int = 16,
```