# `.\diffusers\pipelines\text_to_video_synthesis\pipeline_text_to_video_zero_sdxl.py`

```py
# 导入标准库和第三方库
import copy  # 导入复制对象的模块
import inspect  # 导入检查对象的模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库
import PIL  # 导入 PIL 库
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能模块
from torch.nn.functional import grid_sample  # 导入网格采样功能

# 导入变换器相关的模块
from transformers import (
    CLIPImageProcessor,  # 导入 CLIP 图像处理器
    CLIPTextModel,  # 导入 CLIP 文本模型
    CLIPTextModelWithProjection,  # 导入带有投影的 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 词法分析器
    CLIPVisionModelWithProjection,  # 导入带有投影的 CLIP 视觉模型
)

# 从相对路径导入图像处理器和加载器
from ...image_processor import VaeImageProcessor  # 导入 VAE 图像处理器
from ...loaders import StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin  # 导入稳定扩散和文本反转加载器混合类
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入自编码器和条件 U-Net 模型

# 导入注意力处理器相关的模块
from ...models.attention_processor import (
    AttnProcessor2_0,  # 导入注意力处理器版本 2.0
    FusedAttnProcessor2_0,  # 导入融合注意力处理器版本 2.0
    XFormersAttnProcessor,  # 导入 XFormers 注意力处理器
)

# 从相对路径导入 LoRA 相关功能
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 LoRA 缩放文本编码器的函数

# 导入调度器
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器

# 导入实用工具
from ...utils import (
    USE_PEFT_BACKEND,  # 导入是否使用 PEFT 后端的标志
    BaseOutput,  # 导入基础输出类
    is_invisible_watermark_available,  # 导入检查不可见水印是否可用的函数
    logging,  # 导入日志模块
    scale_lora_layers,  # 导入缩放 LoRA 层的函数
    unscale_lora_layers,  # 导入反缩放 LoRA 层的函数
)

# 从相对路径导入 PyTorch 实用工具
from ...utils.torch_utils import randn_tensor  # 导入随机张量生成函数

# 导入管道工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类

# 如果不可见水印可用，则导入水印模块
if is_invisible_watermark_available():
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker  # 导入稳定扩散 XL 水印器

# 创建日志记录器
logger = logging.get_logger(__name__)  # 根据当前模块名称创建日志记录器，pylint 检查规则

# 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.rearrange_0 中复制的函数
def rearrange_0(tensor, f):
    F, C, H, W = tensor.size()  # 获取张量的尺寸，F:帧数, C:通道数, H:高度, W:宽度
    tensor = torch.permute(torch.reshape(tensor, (F // f, f, C, H, W)), (0, 2, 1, 3, 4))  # 改变张量的形状和维度
    return tensor  # 返回重新排列的张量

# 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.rearrange_1 中复制的函数
def rearrange_1(tensor):
    B, C, F, H, W = tensor.size()  # 获取张量的尺寸，B:批次大小, C:通道数, F:帧数, H:高度, W:宽度
    return torch.reshape(torch.permute(tensor, (0, 2, 1, 3, 4)), (B * F, C, H, W))  # 改变维度并返回张量

# 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.rearrange_3 中复制的函数
def rearrange_3(tensor, f):
    F, D, C = tensor.size()  # 获取张量的尺寸，F:帧数, D:特征维度, C:通道数
    return torch.reshape(tensor, (F // f, f, D, C))  # 改变张量的形状并返回

# 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.rearrange_4 中复制的函数
def rearrange_4(tensor):
    B, F, D, C = tensor.size()  # 获取张量的尺寸，B:批次大小, F:帧数, D:特征维度, C:通道数
    return torch.reshape(tensor, (B * F, D, C))  # 改变张量形状并返回

# 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor 中复制的类
class CrossFrameAttnProcessor:
    """
    跨帧注意力处理器。每帧关注第一帧。

    参数：
        batch_size: 表示实际批次大小的数字，除了帧数以外。
            例如，在使用单个提示和 num_images_per_prompt=1 调用 unet 时，batch_size 应该等于
            2，因分类器自由引导。
    """

    def __init__(self, batch_size=2):
        self.batch_size = batch_size  # 初始化批次大小
    # 定义可调用的前向传递方法，接受注意力机制相关参数
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
            # 获取批次大小、序列长度和隐藏状态的维度
            batch_size, sequence_length, _ = hidden_states.shape
            # 准备注意力掩码以适应批次和序列长度
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # 将隐藏状态转换为查询向量
            query = attn.to_q(hidden_states)
    
            # 判断是否为交叉注意力
            is_cross_attention = encoder_hidden_states is not None
            # 如果没有编码器隐藏状态，则使用隐藏状态本身
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            # 如果需要归一化交叉注意力，则对编码器隐藏状态进行归一化
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
            # 将编码器隐藏状态转换为键向量
            key = attn.to_k(encoder_hidden_states)
            # 将编码器隐藏状态转换为值向量
            value = attn.to_v(encoder_hidden_states)
    
            # 处理交叉帧注意力
            if not is_cross_attention:
                # 计算视频长度并初始化第一个帧索引列表
                video_length = key.size()[0] // self.batch_size
                first_frame_index = [0] * video_length
    
                # 重新排列键以使批次和帧在第一和第二维度
                key = rearrange_3(key, video_length)
                key = key[:, first_frame_index]
                # 重新排列值以使批次和帧在第一和第二维度
                value = rearrange_3(value, video_length)
                value = value[:, first_frame_index]
    
                # 重新排列回原始形状
                key = rearrange_4(key)
                value = rearrange_4(value)
    
            # 将查询、键和值转换为批次维度
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
    
            # 计算注意力得分
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # 使用注意力概率与值进行批次矩阵乘法
            hidden_states = torch.bmm(attention_probs, value)
            # 将隐藏状态转换回头部维度
            hidden_states = attn.batch_to_head_dim(hidden_states)
    
            # 对隐藏状态进行线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # 对隐藏状态应用 dropout
            hidden_states = attn.to_out[1](hidden_states)
    
            # 返回处理后的隐藏状态
            return hidden_states
# 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero 中复制的类
class CrossFrameAttnProcessor2_0:
    """
    使用 PyTorch 2.0 的缩放点积注意力的跨帧注意力处理器。

    Args:
        batch_size: 表示实际批处理大小的数字，而不是帧的数量。
            例如，使用单个提示和 num_images_per_prompt=1 调用 unet 时，batch_size 应该等于
            2，因为无分类器引导。
    """

    # 初始化方法，接受批处理大小的参数，默认为 2
    def __init__(self, batch_size=2):
        # 检查 PyTorch 是否具有缩放点积注意力的功能
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，抛出导入错误，提示需要升级 PyTorch 到 2.0
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # 将传入的批处理大小赋值给实例变量
        self.batch_size = batch_size
    # 定义调用方法，用于执行注意力机制
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # 获取批次大小和序列长度，选择使用的隐藏状态
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # 获取隐藏状态的内维度
        inner_dim = hidden_states.shape[-1]

        # 如果存在注意力掩码
        if attention_mask is not None:
            # 准备注意力掩码，以适应序列长度和批次大小
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention 期望注意力掩码的形状为 (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)

        # 检查是否为交叉注意力
        is_cross_attention = encoder_hidden_states is not None
        # 如果没有交叉隐藏状态，则使用当前的隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要进行交叉注意力的归一化
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 将交叉隐藏状态转换为键和值
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 处理非交叉帧注意力
        if not is_cross_attention:
            # 计算视频长度，确保至少为1
            video_length = max(1, key.size()[0] // self.batch_size)
            first_frame_index = [0] * video_length

            # 重新排列键，使批次和帧位于第1和第2维
            key = rearrange_3(key, video_length)
            key = key[:, first_frame_index]
            # 重新排列值，使批次和帧位于第1和第2维
            value = rearrange_3(value, video_length)
            value = value[:, first_frame_index]

            # 重新排列回原始形状
            key = rearrange_4(key)
            value = rearrange_4(value)

        # 计算每个头的维度
        head_dim = inner_dim // attn.heads
        # 重新排列查询、键和值的形状
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # scaled_dot_product_attention 的输出形状为 (batch, num_heads, seq_len, head_dim)
        # TODO: 在升级到 Torch 2.1 时添加对 attn.scale 的支持
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # 转置并重塑隐藏状态以合并头维度
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # 将隐藏状态转换为查询向量的数据类型
        hidden_states = hidden_states.to(query.dtype)

        # 线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)
        # 返回最终的隐藏状态
        return hidden_states
# 定义一个用于零-shot文本到视频管道的输出类，继承自BaseOutput
@dataclass
class TextToVideoSDXLPipelineOutput(BaseOutput):
    """
    输出类用于零-shot文本到视频管道。

    参数：
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            长度为 `batch_size` 的去噪PIL图像列表或形状为 `(batch_size, height, width,
            num_channels)` 的numpy数组。PIL图像或numpy数组呈现扩散管道的去噪图像。
    """

    # 定义属性images，可以是PIL图像列表或numpy数组
    images: Union[List[PIL.Image.Image], np.ndarray]


# 从diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.coords_grid复制的函数
def coords_grid(batch, ht, wd, device):
    # 从给定的高度和宽度生成网格坐标，适应设备
    # 适配自：https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    # 反转坐标顺序并转换为浮点型
    coords = torch.stack(coords[::-1], dim=0).float()
    # 生成的坐标扩展到指定的batch维度
    return coords[None].repeat(batch, 1, 1, 1)


# 从diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.warp_single_latent复制的函数
def warp_single_latent(latent, reference_flow):
    """
    使用给定的流对单帧的潜在表示进行扭曲

    参数：
        latent: 单帧的潜在代码
        reference_flow: 用于扭曲潜在表示的流

    返回：
        warped: 扭曲后的潜在表示
    """
    # 获取参考流的尺寸，高度和宽度
    _, _, H, W = reference_flow.size()
    # 获取潜在表示的尺寸，高度和宽度
    _, _, h, w = latent.size()
    # 生成坐标网格并转换为潜在表示的dtype
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype)

    # 将参考流加到坐标网格上
    coords_t0 = coords0 + reference_flow
    # 将坐标归一化到[0, 1]范围
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    # 将坐标缩放到[-1, 1]范围
    coords_t0 = coords_t0 * 2.0 - 1.0
    # 对坐标进行双线性插值以调整大小
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear")
    # 调整坐标的维度顺序
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))

    # 使用给定坐标从潜在表示中采样扭曲的结果
    warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection")
    return warped


# 从diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.create_motion_field复制的函数
def create_motion_field(motion_field_strength_x, motion_field_strength_y, frame_ids, device, dtype):
    """
    创建平移运动场

    参数：
        motion_field_strength_x: x轴的运动强度
        motion_field_strength_y: y轴的运动强度
        frame_ids: 正在处理的潜在表示的帧索引。
            当我们执行分块推理时，这一点很重要
        device: 设备
        dtype: 数据类型

    返回：

    """
    # 获取帧的序列长度
    seq_length = len(frame_ids)
    # 初始化参考流，大小为 (seq_length, 2, 512, 512)
    reference_flow = torch.zeros((seq_length, 2, 512, 512), device=device, dtype=dtype)
    # 遍历每一帧，计算对应的运动场
    for fr_idx in range(seq_length):
        # 在x轴上为每一帧应用运动强度
        reference_flow[fr_idx, 0, :, :] = motion_field_strength_x * (frame_ids[fr_idx])
        # 在y轴上为每一帧应用运动强度
        reference_flow[fr_idx, 1, :, :] = motion_field_strength_y * (frame_ids[fr_idx])
    # 返回生成的参考流
    return reference_flow


# 从diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.create_motion_field_and_warp_latents复制的函数
def create_motion_field_and_warp_latents(motion_field_strength_x, motion_field_strength_y, frame_ids, latents):
    """
    # 创建平移运动，并相应地扭曲潜在表示
        """
        # 参数说明
        # motion_field_strength_x: x 轴上的运动强度
        # motion_field_strength_y: y 轴上的运动强度
        # frame_ids: 正在处理的帧的索引。
        #          在进行分块推理时需要此信息
        # latents: 帧的潜在编码
        # 返回值说明
        # warped_latents: 扭曲后的潜在表示
        """
        # 创建运动场，根据给定的运动强度和帧 ID
        motion_field = create_motion_field(
            # 设置 x 轴运动强度
            motion_field_strength_x=motion_field_strength_x,
            # 设置 y 轴运动强度
            motion_field_strength_y=motion_field_strength_y,
            # 传入帧 ID
            frame_ids=frame_ids,
            # 设定设备为潜在表示的设备
            device=latents.device,
            # 设定数据类型为潜在表示的数据类型
            dtype=latents.dtype,
        )
        # 克隆潜在表示以创建扭曲后的潜在表示
        warped_latents = latents.clone().detach()
        # 遍历所有扭曲的潜在表示
        for i in range(len(warped_latents)):
            # 对每个潜在表示应用单个扭曲操作
            warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
        # 返回扭曲后的潜在表示
        return warped_latents
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 调整 `noise_cfg`。基于 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的研究结果。见第 3.4 节
    """
    # 计算 noise_pred_text 在所有维度上的标准差，并保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算 noise_cfg 在所有维度上的标准差，并保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据标准差调整来自指导的结果（修正过度曝光）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 通过 guidance_rescale 因子与来自指导的原始结果混合，以避免“单调”的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回调整后的 noise_cfg
    return noise_cfg


# 定义一个用于零-shot 文本到视频生成的管道类
class TextToVideoZeroSDXLPipeline(
    DiffusionPipeline,  # 继承自 DiffusionPipeline
    StableDiffusionMixin,  # 继承自 StableDiffusionMixin
    StableDiffusionXLLoraLoaderMixin,  # 继承自 StableDiffusionXLLoraLoaderMixin
    TextualInversionLoaderMixin,  # 继承自 TextualInversionLoaderMixin
):
    r"""
    使用 Stable Diffusion XL 进行零-shot 文本到视频生成的管道。

    该模型继承自 [`DiffusionPipeline`]。查看超类文档以获取所有管道的通用方法
    （下载、保存、在特定设备上运行等）。
    # 参数说明
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器（VAE）模型，用于将图像编码为潜在表示，并从潜在表示解码图像。
        text_encoder ([`CLIPTextModel`]):
            # 冻结的文本编码器。稳定扩散 XL 使用 CLIP 的文本部分，
            # 具体来说是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            # 第二个冻结文本编码器。稳定扩散 XL 使用 CLIP 的文本和池部分，
            # 具体是 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 变体。
        tokenizer (`CLIPTokenizer`):
            # CLIPTokenizer 类的分词器。
        tokenizer_2 (`CLIPTokenizer`):
            # 第二个 CLIPTokenizer 类的分词器。
        unet ([`UNet2DConditionModel`]): 
            # 条件 U-Net 架构，用于去噪编码后的图像潜在表示。
        scheduler ([`SchedulerMixin`]):
            # 调度器，用于与 `unet` 结合去噪编码后的图像潜在表示。可以是
            # [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的任意一个。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    # 定义可选组件列表
    _optional_components = [
        # 分词器
        "tokenizer",
        # 第二个分词器
        "tokenizer_2",
        # 第一个文本编码器
        "text_encoder",
        # 第二个文本编码器
        "text_encoder_2",
        # 图像编码器
        "image_encoder",
        # 特征提取器
        "feature_extractor",
    ]

    # 初始化方法定义
    def __init__(
        # 变分自编码器
        self,
        vae: AutoencoderKL,
        # 文本编码器
        text_encoder: CLIPTextModel,
        # 第二个文本编码器
        text_encoder_2: CLIPTextModelWithProjection,
        # 第一个分词器
        tokenizer: CLIPTokenizer,
        # 第二个分词器
        tokenizer_2: CLIPTokenizer,
        # 条件 U-Net
        unet: UNet2DConditionModel,
        # 调度器
        scheduler: KarrasDiffusionSchedulers,
        # 可选的图像编码器，默认为 None
        image_encoder: CLIPVisionModelWithProjection = None,
        # 可选的特征提取器，默认为 None
        feature_extractor: CLIPImageProcessor = None,
        # 用于处理空提示的强制零值，默认为 True
        force_zeros_for_empty_prompt: bool = True,
        # 可选的水印标记，默认为 None
        add_watermarker: Optional[bool] = None,
    # 初始化父类
    ):
        super().__init__()
        # 注册多个模块，包括 VAE、文本编码器和其他组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        # 将配置参数注册到对象中，强制为空提示使用零
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        # 计算 VAE 的缩放因子，基于输出通道的数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，使用 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 设置默认的采样尺寸，从 UNet 配置中获取
        self.default_sample_size = self.unet.config.sample_size

        # 如果 add_watermarker 为 None，使用可用的隐形水印设置
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        # 如果需要添加水印，则初始化水印器，否则设置为 None
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # 从稳定扩散管道中复制的函数，用于准备额外的步骤参数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为并非所有调度器具有相同的签名
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 范围内

        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果接受，则将 eta 添加到额外参数中
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受，则将 generator 添加到额外参数中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数
        return extra_step_kwargs

    # 从稳定扩散 XL 管道中复制的函数，用于提升 VAE 精度
    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 类型
        self.vae.to(dtype=torch.float32)
        # 检查是否使用 torch 2.0 或 xformers
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # 如果使用了 xformers 或 torch 2.0，则不需要将注意力块放在 float32 中，以节省大量内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积转换为原始数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将解码器的输入卷积转换为原始数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将解码器的中间块转换为原始数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 从稳定扩散 XL 管道中复制的函数，用于获取附加时间 ID
    # 定义一个私有方法，用于获取添加时间的 ID
    def _get_add_time_ids(
        # 输入参数：原始大小、裁剪的左上角坐标、目标大小、数据类型和文本编码器的投影维度（可选）
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # 将原始大小、裁剪坐标和目标大小合并为一个列表
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        # 计算通过时间嵌入所需的维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取模型预期的时间嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查计算得到的维度与预期是否一致
        if expected_add_embed_dim != passed_add_embed_dim:
            # 如果不一致，抛出值错误，提示用户检查配置
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将添加时间的 ID 转换为张量，并指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 返回添加时间的 ID 张量
        return add_time_ids

    # 从 StableDiffusionPipeline 复制的方法，用于准备潜在的张量
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 根据输入参数计算潜在张量的形状
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器的长度是否与批量大小一致
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不一致，抛出值错误，提示用户检查生成器长度
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在张量，随机生成一个
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在张量，将其移动到指定设备
            latents = latents.to(device)

        # 根据调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在张量
        return latents

    # 检查输入参数的方法
    def check_inputs(
        # 输入参数：提示、第二个提示、高度、宽度、回调步骤以及可选的负提示和嵌入
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    # 从 StableDiffusionXLPipeline 复制的方法，用于编码提示
    # 定义一个方法用于编码提示信息
        def encode_prompt(
            # 输入的提示字符串
            self,
            prompt: str,
            # 第二个提示字符串，默认为 None
            prompt_2: Optional[str] = None,
            # 设备类型，默认为 None
            device: Optional[torch.device] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 是否使用分类器自由引导，默认为 True
            do_classifier_free_guidance: bool = True,
            # 负提示字符串，默认为 None
            negative_prompt: Optional[str] = None,
            # 第二个负提示字符串，默认为 None
            negative_prompt_2: Optional[str] = None,
            # 提示的张量表示，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的张量表示，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化后的提示张量表示，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化后的负提示张量表示，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # Lora 的缩放因子，默认为 None
            lora_scale: Optional[float] = None,
            # 跳过的 clip 数，默认为 None
            clip_skip: Optional[int] = None,
        # 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoZeroPipeline.forward_loop 复制的代码
        def forward_loop(self, x_t0, t0, t1, generator):
            """
            从 t0 到 t1 执行 DDPM 向前过程，即根据相应方差添加噪声。
    
            Args:
                x_t0:
                    t0 时刻的潜在编码。
                t0:
                    t0 时刻的时间步。
                t1:
                    t1 时刻的时间步。
                generator (`torch.Generator` 或 `List[torch.Generator]`, *可选*):
                    用于生成确定性的 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)。
    
            Returns:
                x_t1:
                    对 x_t0 应用从 t0 到 t1 的向前过程。
            """
            # 生成与 x_t0 相同形状的随机噪声张量
            eps = randn_tensor(x_t0.size(), generator=generator, dtype=x_t0.dtype, device=x_t0.device)
            # 计算在 t0 到 t1 之间的 alpha 向量的乘积
            alpha_vec = torch.prod(self.scheduler.alphas[t0:t1])
            # 计算向前过程结果 x_t1
            x_t1 = torch.sqrt(alpha_vec) * x_t0 + torch.sqrt(1 - alpha_vec) * eps
            # 返回结果 x_t1
            return x_t1
    
        # 定义一个方法用于执行向后过程
        def backward_loop(
            # 潜在变量
            latents,
            # 时间步序列
            timesteps,
            # 提示的张量表示
            prompt_embeds,
            # 引导的缩放因子
            guidance_scale,
            # 回调函数
            callback,
            # 回调步骤
            callback_steps,
            # 预热步骤数量
            num_warmup_steps,
            # 额外步骤的参数
            extra_step_kwargs,
            # 添加文本嵌入
            add_text_embeds,
            # 添加时间 ID
            add_time_ids,
            # 交叉注意力的参数，默认为 None
            cross_attention_kwargs=None,
            # 引导重缩放因子，默认为 0.0
            guidance_rescale: float = 0.0,
        # 不计算梯度的装饰器
        @torch.no_grad()
    # 定义可调用对象的方法，允许使用特定参数生成视频
    def __call__(
            self,
            # 输入提示，支持字符串或字符串列表
            prompt: Union[str, List[str]],
            # 第二个提示，支持字符串或字符串列表，默认为 None
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 视频长度，默认为 8 秒
            video_length: Optional[int] = 8,
            # 视频高度，默认为 None
            height: Optional[int] = None,
            # 视频宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤数，默认为 50
            num_inference_steps: int = 50,
            # 去噪结束阈值，默认为 None
            denoising_end: Optional[float] = None,
            # 引导缩放因子，默认为 7.5
            guidance_scale: float = 7.5,
            # 负提示，支持字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负提示，支持字符串或字符串列表，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的视频数量，默认为 1
            num_videos_per_prompt: Optional[int] = 1,
            # 额外噪声强度，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个或多个 torch.Generator，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 帧 ID 列表，默认为 None
            frame_ids: Optional[List[int]] = None,
            # 提示的嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 聚合提示的嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负聚合提示的嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 潜在变量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 动作场强度 X，默认为 12
            motion_field_strength_x: float = 12,
            # 动作场强度 Y，默认为 12
            motion_field_strength_y: float = 12,
            # 输出类型，默认为 "tensor"
            output_type: Optional[str] = "tensor",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 回调函数，接受特定参数，默认为 None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 每多少步骤调用一次回调，默认为 1
            callback_steps: int = 1,
            # 跨注意力的额外参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 引导重标定，默认为 0.0
            guidance_rescale: float = 0.0,
            # 原始尺寸，默认为 None
            original_size: Optional[Tuple[int, int]] = None,
            # 裁剪坐标的左上角，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标尺寸，默认为 None
            target_size: Optional[Tuple[int, int]] = None,
            # 初始时间 t0，默认为 44
            t0: int = 44,
            # 结束时间 t1，默认为 47
            t1: int = 47,
```