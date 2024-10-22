# `.\diffusers\pipelines\i2vgen_xl\pipeline_i2vgen_xl.py`

```py
# 版权所有 2024 Alibaba DAMO-VILAB 和 The HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件
# 是按“原样”基础分发，不提供任何形式的担保或条件。
# 有关许可证的特定权限和限制，请参见许可证。

import inspect  # 导入 inspect 模块，用于获取对象的信息
from dataclasses import dataclass  # 从 dataclasses 导入 dataclass 装饰器
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示所需的各种类型

import numpy as np  # 导入 numpy 库，通常用于数值计算
import PIL  # 导入 PIL 库，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 从 transformers 导入相关的 CLIP 模型和处理器

from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从自定义模块导入图像处理类
from ...models import AutoencoderKL  # 从自定义模块导入自动编码器模型
from ...models.unets.unet_i2vgen_xl import I2VGenXLUNet  # 从自定义模块导入特定的 UNet 模型
from ...schedulers import DDIMScheduler  # 从自定义模块导入调度器
from ...utils import (  # 从自定义模块导入各种工具
    BaseOutput,  # 基础输出类
    logging,  # 日志记录功能
    replace_example_docstring,  # 替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从自定义模块导入用于生成随机张量的函数
from ...video_processor import VideoProcessor  # 从自定义模块导入视频处理器
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从自定义模块导入扩散管道和稳定扩散混合类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例，方便记录日志信息

EXAMPLE_DOC_STRING = """  # 定义一个示例文档字符串
    Examples:  # 示例部分的标题
        ```py  # Python 代码块的开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import I2VGenXLPipeline  # 从 diffusers 模块导入 I2VGenXLPipeline 类
        >>> from diffusers.utils import export_to_gif, load_image  # 从 utils 模块导入辅助函数

        >>> pipeline = I2VGenXLPipeline.from_pretrained(  # 从预训练模型创建管道实例
        ...     "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"  # 指定模型名称和参数
        ... )
        >>> pipeline.enable_model_cpu_offload()  # 启用模型的 CPU 卸载功能以节省内存

        >>> image_url = (  # 定义图像的 URL
        ...     "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"  # 图像的具体 URL
        ... )
        >>> image = load_image(image_url).convert("RGB")  # 加载图像并转换为 RGB 格式

        >>> prompt = "Papers were floating in the air on a table in the library"  # 定义正向提示
        >>> negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"  # 定义负向提示
        >>> generator = torch.manual_seed(8888)  # 设置随机种子以确保结果可重现

        >>> frames = pipeline(  # 调用管道以生成图像帧
        ...     prompt=prompt,  # 传递正向提示
        ...     image=image,  # 传递输入图像
        ...     num_inference_steps=50,  # 设置推理步骤数
        ...     negative_prompt=negative_prompt,  # 传递负向提示
        ...     guidance_scale=9.0,  # 设置引导比例
        ...     generator=generator,  # 传递随机数生成器
        ... ).frames[0]  # 获取生成的第一帧
        >>> video_path = export_to_gif(frames, "i2v.gif")  # 将生成的帧导出为 GIF 格式的视频
        ```py  # Python 代码块的结束
"""

@dataclass  # 使用 dataclass 装饰器定义一个数据类
class I2VGenXLPipelineOutput(BaseOutput):  # 定义 I2VGenXLPipelineOutput 类，继承自 BaseOutput
    r"""  # 文档字符串，描述该类的作用
     Output class for image-to-video pipeline.  # 说明这是图像到视频管道的输出类
    # 函数参数文档字符串，描述函数的参数及其类型
        Args:
             # frames 参数可以是 torch.Tensor、np.ndarray 或嵌套列表，每个子列表包含去噪的 PIL 图像序列
             frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
                 # 说明 frames 是一个视频输出列表，长度为 batch_size，每个子列表包含 num_frames 长度的去噪图像序列
                 List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
                 denoised
         # 说明 frames 也可以是形状为 (batch_size, num_frames, channels, height, width) 的 NumPy 数组或 Torch 张量
         PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
        `(batch_size, num_frames, channels, height, width)`
        """
    
        # 定义 frames 变量类型，支持多种数据类型的联合
        frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
# 定义图像到视频生成的管道类，继承自 DiffusionPipeline 和 StableDiffusionMixin
class I2VGenXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
):
    r"""
    用于图像到视频生成的管道，如 [I2VGenXL](https://i2vgen-xl.github.io/) 所提议。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道的通用方法（下载、保存、在特定设备上运行等），请查看超类文档。

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer (`CLIPTokenizer`):
            用于标记文本的 [`~transformers.CLIPTokenizer`]。
        unet ([`I2VGenXLUNet`]):
            用于去噪编码视频潜在表示的 [`I2VGenXLUNet`]。
        scheduler ([`DDIMScheduler`]):
            与 `unet` 结合使用以去噪编码图像潜在表示的调度器。
    """

    # 定义模型各组件的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        image_encoder: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
        unet: I2VGenXLUNet,
        scheduler: DDIMScheduler,
    ):
        # 初始化父类
        super().__init__()

        # 注册模型的各个模块
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 设置视频处理器，不进行默认调整大小
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor, do_resize=False)

    # 定义属性，获取指导缩放值
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义属性，确定是否执行无分类器引导
    # 这里的 `guidance_scale` 定义类似于 Imagen 论文中公式（2）的指导权重 `w`：
    # `guidance_scale = 1` 对应于不进行分类器自由引导。
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # 编码提示的函数
    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
    # 定义编码图像的私有方法，接收图像、设备和每个提示的视频数量
    def _encode_image(self, image, device, num_videos_per_prompt):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 检查输入图像是否为 PyTorch 张量
        if not isinstance(image, torch.Tensor):
            # 将 PIL 图像转换为 NumPy 数组
            image = self.video_processor.pil_to_numpy(image)
            # 将 NumPy 数组转换为 PyTorch 张量
            image = self.video_processor.numpy_to_pt(image)

            # 使用 CLIP 训练统计信息对图像进行归一化处理
            image = self.feature_extractor(
                images=image,
                do_normalize=True,  # 是否归一化
                do_center_crop=False,  # 是否中心裁剪
                do_resize=False,  # 是否调整大小
                do_rescale=False,  # 是否重新缩放
                return_tensors="pt",  # 返回 PyTorch 张量
            ).pixel_values  # 获取处理后的图像像素值

        # 将图像移动到指定设备，并转换为指定数据类型
        image = image.to(device=device, dtype=dtype)
        # 使用图像编码器对图像进行编码，获取图像嵌入
        image_embeddings = self.image_encoder(image).image_embeds
        # 在第二个维度上添加一个维度
        image_embeddings = image_embeddings.unsqueeze(1)

        # 为每个提示生成的重复图像嵌入，使用兼容 MPS 的方法
        bs_embed, seq_len, _ = image_embeddings.shape  # 获取嵌入的批量大小和序列长度
        # 重复图像嵌入以适应每个提示的视频数量
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        # 重新调整图像嵌入的形状以合并批量和视频数量
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # 如果启用了无分类器自由引导
        if self.do_classifier_free_guidance:
            # 创建与图像嵌入相同形状的零张量
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            # 将负图像嵌入和正图像嵌入拼接
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        # 返回最终的图像嵌入
        return image_embeddings

    # 定义解码潜在空间的公共方法，接收潜在向量和可选的解码块大小
    def decode_latents(self, latents, decode_chunk_size=None):
        # 使用 VAE 配置的缩放因子对潜在向量进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents

        # 获取潜在向量的批量大小、通道数、帧数、高度和宽度
        batch_size, channels, num_frames, height, width = latents.shape
        # 重新排列潜在向量的维度，以适应 VAE 解码器的输入格式
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        # 如果指定了解码块大小
        if decode_chunk_size is not None:
            frames = []  # 用于存储解码的帧
            # 按照解码块大小逐块解码潜在向量
            for i in range(0, latents.shape[0], decode_chunk_size):
                # 解码当前块的潜在向量，获取采样结果
                frame = self.vae.decode(latents[i : i + decode_chunk_size]).sample
                frames.append(frame)  # 将解码的帧添加到列表中
            # 将所有帧在第一个维度上拼接成一个张量
            image = torch.cat(frames, dim=0)
        else:
            # 如果未指定块大小，直接解码所有潜在向量
            image = self.vae.decode(latents).sample

        # 计算解码后的形状，以适应最终视频的结构
        decode_shape = (batch_size, num_frames, -1) + image.shape[2:]
        # 重新调整图像的形状，以便形成视频结构
        video = image[None, :].reshape(decode_shape).permute(0, 2, 1, 3, 4)

        # 始终将视频转换为 float32 格式，以兼容 bfloat16 并减少开销
        video = video.float()
        # 返回解码后的视频
        return video

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的方法
    # 准备额外的参数以供调度器步骤使用，因为并不是所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略它
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 并且应该在 [0, 1] 范围内

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个字典用于存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 检查输入参数的有效性
    def check_inputs(
        # 提示文本
        prompt,
        # 输入图像
        image,
        # 图像高度
        height,
        # 图像宽度
        width,
        # 可选的负提示文本
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds=None,
        # 可选的负提示嵌入
        negative_prompt_embeds=None,
    # 结束函数参数列表
        ):
            # 检查高度和宽度是否都是8的倍数，若不是则抛出错误
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    
            # 检查是否同时提供了 prompt 和 prompt_embeds，若是则抛出错误
            if prompt is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查 prompt 和 prompt_embeds 是否都未定义，若是则抛出错误
            elif prompt is None and prompt_embeds is None:
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查 prompt 的类型是否为字符串或列表，若不是则抛出错误
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查是否同时提供了 negative_prompt 和 negative_prompt_embeds，若是则抛出错误
            if negative_prompt is not None and negative_prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查 prompt_embeds 和 negative_prompt_embeds 的形状是否一致，若不一致则抛出错误
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
            # 检查 image 的类型是否为 torch.Tensor、PIL.Image.Image 或其列表，若不是则抛出错误
            if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
            ):
                raise ValueError(
                    "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                    f" {type(image)}"
                )
    
        # 定义 prepare_image_latents 函数，准备图像潜变量
        def prepare_image_latents(
            self,
            image,
            device,
            num_frames,
            num_videos_per_prompt,
    ):
        # 将图像移动到指定的设备（如 GPU）
        image = image.to(device=device)
        # 编码图像并从变分自编码器（VAE）获取潜在分布的样本
        image_latents = self.vae.encode(image).latent_dist.sample()
        # 将潜在表示缩放到 VAE 配置的缩放因子
        image_latents = image_latents * self.vae.config.scaling_factor

        # 为潜在图像添加帧维度
        image_latents = image_latents.unsqueeze(2)

        # 为每个后续帧添加位置掩码，初始图像潜在帧之后
        frame_position_mask = []
        # 遍历除第一帧之外的所有帧
        for frame_idx in range(num_frames - 1):
            # 计算当前帧的缩放因子
            scale = (frame_idx + 1) / (num_frames - 1)
            # 将缩放因子应用于与潜在表示相同形状的张量
            frame_position_mask.append(torch.ones_like(image_latents[:, :, :1]) * scale)
        # 如果位置掩码非空，则连接它们
        if frame_position_mask:
            frame_position_mask = torch.cat(frame_position_mask, dim=2)
            # 将位置掩码附加到潜在表示上
            image_latents = torch.cat([image_latents, frame_position_mask], dim=2)

        # 根据每个提示的生成数量复制潜在表示，使用适合 MPS 的方法
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

        # 如果使用无分类器自由引导，则重复潜在表示
        if self.do_classifier_free_guidance:
            image_latents = torch.cat([image_latents] * 2)

        # 返回处理后的潜在表示
        return image_latents

    # 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents 复制
    def prepare_latents(
        # 准备潜在表示的参数，包括批大小、通道数、帧数等
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 定义潜在表示的形状
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 检查生成器列表长度是否与批大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果未提供潜在表示，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在表示，则将其移动到指定设备
            latents = latents.to(device)

        # 根据调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在表示
        return latents

    # 在不计算梯度的上下文中执行
    @torch.no_grad()
    # 用示例文档字符串替换
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，允许实例像函数一样被调用
    def __call__(
        # 输入提示，可以是字符串或字符串列表，默认为 None
        self,
        prompt: Union[str, List[str]] = None,
        # 输入图像，类型为 PipelineImageInput，默认为 None
        image: PipelineImageInput = None,
        # 目标图像高度，默认为 704 像素
        height: Optional[int] = 704,
        # 目标图像宽度，默认为 1280 像素
        width: Optional[int] = 1280,
        # 目标帧率，默认为 16 帧每秒
        target_fps: Optional[int] = 16,
        # 要生成的帧数，默认为 16
        num_frames: int = 16,
        # 推理步骤数量，默认为 50
        num_inference_steps: int = 50,
        # 指导比例，默认为 9.0，控制生成图像的多样性
        guidance_scale: float = 9.0,
        # 负提示，可以是字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 随机噪声的调节因子，默认为 0.0
        eta: float = 0.0,
        # 每个提示生成的视频数量，默认为 1
        num_videos_per_prompt: Optional[int] = 1,
        # 解码时的块大小，默认为 1
        decode_chunk_size: Optional[int] = 1,
        # 随机数生成器，可以是单个或多个 torch.Generator，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在变量张量，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 提示嵌入张量，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示嵌入张量，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"，表示生成 PIL 图像
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
        # 交叉注意力的关键字参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 跳过的剪辑数，默认为 1
        clip_skip: Optional[int] = 1,
# 以下实用工具来自并适应于
# https://github.com/ali-vilab/i2vgen-xl/blob/main/utils/transforms.py.

# 将 PyTorch 张量或张量列表转换为 PIL 图像
def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    # 如果输入是一个张量列表，则将其沿第一个维度拼接成一个张量
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    # 如果输入是一个张量
    if isinstance(image, torch.Tensor):
        # 如果张量是 3 维的，则在第一个维度增加一个维度，使其变为 4 维
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # 将 PyTorch 张量转换为 NumPy 数组
        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        # 将 NumPy 数组转换为 PIL 图像
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        # 更新 image 为 PIL 图像
        image = image_pil

    # 返回转换后的图像
    return image

# 使用双线性插值调整图像大小
def _resize_bilinear(
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # 首先将图像转换为 PIL 格式，以防它们是浮动张量（目前仅与测试相关）
    image = _convert_pt_to_pil(image)

    # 如果输入是图像列表，则对每个图像进行调整大小
    if isinstance(image, list):
        image = [u.resize(resolution, PIL.Image.BILINEAR) for u in image]
    else:
        # 如果是单个图像，直接调整大小
        image = image.resize(resolution, PIL.Image.BILINEAR)
    # 返回调整大小后的图像
    return image

# 进行中心裁剪以适应给定的分辨率
def _center_crop_wide(
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # 首先将图像转换为 PIL 格式，以防它们是浮动张量（目前仅与测试相关）
    image = _convert_pt_to_pil(image)

    # 如果输入是图像列表
    if isinstance(image, list):
        # 计算缩放比例，确保图像适应目标分辨率
        scale = min(image[0].size[0] / resolution[0], image[0].size[1] / resolution[1])
        # 调整每个图像的大小
        image = [u.resize((round(u.width // scale), round(u.height // scale)), resample=PIL.Image.BOX) for u in image]

        # 进行中心裁剪
        x1 = (image[0].width - resolution[0]) // 2  # 计算裁剪区域的左上角 x 坐标
        y1 = (image[0].height - resolution[1]) // 2  # 计算裁剪区域的左上角 y 坐标
        # 裁剪每个图像并返回
        image = [u.crop((x1, y1, x1 + resolution[0], y1 + resolution[1])) for u in image]
        return image
    else:
        # 对于单个图像，计算缩放比例
        scale = min(image.size[0] / resolution[0], image.size[1] / resolution[1])
        # 调整图像大小
        image = image.resize((round(image.width // scale), round(image.height // scale)), resample=PIL.Image.BOX)
        # 计算裁剪区域的左上角坐标
        x1 = (image.width - resolution[0]) // 2
        y1 = (image.height - resolution[1]) // 2
        # 裁剪图像并返回
        image = image.crop((x1, y1, x1 + resolution[0], y1 + resolution[1]))
        return image
```