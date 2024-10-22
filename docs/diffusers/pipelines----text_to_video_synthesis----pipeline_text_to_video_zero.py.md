# `.\diffusers\pipelines\text_to_video_synthesis\pipeline_text_to_video_zero.py`

```py
# 导入所需模块和库
import copy  # 复制对象的模块
import inspect  # 检查对象的模块
from dataclasses import dataclass  # 数据类装饰器
from typing import Callable, List, Optional, Union  # 类型提示相关

import numpy as np  # 数组处理库
import PIL.Image  # 图像处理库
import torch  # 深度学习框架
import torch.nn.functional as F  # 神经网络功能模块
from torch.nn.functional import grid_sample  # 网格采样函数
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer  # 处理CLIP模型的类

from ...image_processor import VaeImageProcessor  # VAE图像处理器
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 加载器混合类
from ...models import AutoencoderKL, UNet2DConditionModel  # 模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 调整LoRA比例的函数
from ...schedulers import KarrasDiffusionSchedulers  # Karras扩散调度器
from ...utils import USE_PEFT_BACKEND, BaseOutput, logging, scale_lora_layers, unscale_lora_layers  # 实用工具
from ...utils.torch_utils import randn_tensor  # 随机张量生成工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 管道工具类
from ..stable_diffusion import StableDiffusionSafetyChecker  # 稳定扩散安全检查器

logger = logging.get_logger(__name__)  # 创建模块日志记录器，禁用pylint命名检查


def rearrange_0(tensor, f):
    F, C, H, W = tensor.size()  # 解构张量维度
    tensor = torch.permute(torch.reshape(tensor, (F // f, f, C, H, W)), (0, 2, 1, 3, 4))  # 重排列和调整张量形状
    return tensor  # 返回调整后的张量


def rearrange_1(tensor):
    B, C, F, H, W = tensor.size()  # 解构张量维度
    return torch.reshape(torch.permute(tensor, (0, 2, 1, 3, 4)), (B * F, C, H, W))  # 重排列并调整形状


def rearrange_3(tensor, f):
    F, D, C = tensor.size()  # 解构张量维度
    return torch.reshape(tensor, (F // f, f, D, C))  # 调整张量形状


def rearrange_4(tensor):
    B, F, D, C = tensor.size()  # 解构张量维度
    return torch.reshape(tensor, (B * F, D, C))  # 调整张量形状


class CrossFrameAttnProcessor:
    """
    跨帧注意力处理器。每帧关注第一帧。

    Args:
        batch_size: 表示实际批大小的数字，而不是帧数。
            例如，使用单个提示和num_images_per_prompt=1调用unet时，batch_size应等于
            2，因为使用了无分类器引导。
    """

    def __init__(self, batch_size=2):
        self.batch_size = batch_size  # 初始化批大小
    # 定义可调用方法，处理注意力机制
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # 获取批次大小、序列长度及其维度
        batch_size, sequence_length, _ = hidden_states.shape
        # 准备注意力掩码，以适应批次和序列长度
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)
    
        # 判断是否进行交叉注意力
        is_cross_attention = encoder_hidden_states is not None
        # 如果没有编码器隐藏状态，则使用隐藏状态本身
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要归一化交叉注意力的隐藏状态
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
        # 将编码器隐藏状态转换为键和值
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    
        # 处理交叉帧注意力
        if not is_cross_attention:
            # 计算视频长度和第一个帧索引
            video_length = key.size()[0] // self.batch_size
            first_frame_index = [0] * video_length
    
            # 重新排列键，使批次和帧在前两个维度
            key = rearrange_3(key, video_length)
            key = key[:, first_frame_index]
            # 重新排列值，使批次和帧在前两个维度
            value = rearrange_3(value, video_length)
            value = value[:, first_frame_index]
    
            # 重新排列回原始形状
            key = rearrange_4(key)
            value = rearrange_4(value)
    
        # 将查询、键和值转换为批次维度
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
    
        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 应用注意力分数于值，得到隐藏状态
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态转换回头部维度
        hidden_states = attn.batch_to_head_dim(hidden_states)
    
        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)
    
        # 返回最终的隐藏状态
        return hidden_states
# 定义一个名为 CrossFrameAttnProcessor2_0 的类
class CrossFrameAttnProcessor2_0:
    """
    Cross frame attention processor with scaled_dot_product attention of Pytorch 2.0.

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, calling unet with a single prompt and num_images_per_prompt=1, batch_size should be equal to
            2, due to classifier-free guidance.
    """

    # 初始化方法，设置默认的批次大小为 2
    def __init__(self, batch_size=2):
        # 检查 F 是否具有 scaled_dot_product_attention 属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，抛出 ImportError，提示需要升级到 PyTorch 2.0
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # 将传入的批次大小赋值给实例变量 self.batch_size
        self.batch_size = batch_size
    # 定义可调用对象的方法，接受注意力、隐藏状态和可选的编码器隐藏状态、注意力掩码
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
            # 获取批次大小和序列长度，使用编码器隐藏状态的形状或隐藏状态的形状
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            # 获取隐藏状态的内部维度
            inner_dim = hidden_states.shape[-1]
    
            # 如果提供了注意力掩码
            if attention_mask is not None:
                # 准备注意力掩码，调整形状以匹配序列长度和批次大小
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention 期望注意力掩码的形状为 (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
    
            # 将隐藏状态转换为查询
            query = attn.to_q(hidden_states)
    
            # 检查是否为交叉注意力
            is_cross_attention = encoder_hidden_states is not None
            # 如果没有提供编码器隐藏状态，则使用隐藏状态
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            # 如果需要，对编码器隐藏状态进行归一化
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
            # 将编码器隐藏状态转换为键和值
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
    
            # 处理交叉帧注意力
            if not is_cross_attention:
                # 计算视频长度并初始化第一个帧索引
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
            # 调整查询的形状并转置维度
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # 调整键的形状并转置维度
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # 调整值的形状并转置维度
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    
            # 执行缩放点积注意力，输出形状为 (batch, num_heads, seq_len, head_dim)
            # TODO: 当我们迁移到 Torch 2.1 时，添加对 attn.scale 的支持
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
    
            # 转置输出并调整形状以合并头的维度
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            # 将隐藏状态转换为查询的 dtype
            hidden_states = hidden_states.to(query.dtype)
    
            # 应用线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # 应用 dropout
            hidden_states = attn.to_out[1](hidden_states)
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个数据类，用于表示零-shot文本到视频的输出
@dataclass
class TextToVideoPipelineOutput(BaseOutput):
    r"""
    输出类用于零-shot文本到视频管道。

    参数：
        images (`[List[PIL.Image.Image]`, `np.ndarray`]):
            长度为`batch_size`的去噪PIL图像列表或形状为`(batch_size, height, width, num_channels)`的NumPy数组。
        nsfw_content_detected (`[List[bool]]`):
            列表，指示相应生成的图像是否包含“不安全内容”（nsfw），如果安全检查无法执行则为`None`。
    """

    # 图像字段，支持PIL图像列表或NumPy数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # 检测到的NSFW内容列表
    nsfw_content_detected: Optional[List[bool]]


# 定义一个函数，用于生成坐标网格
def coords_grid(batch, ht, wd, device):
    # 从指定高度和宽度生成网格坐标
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    # 将生成的坐标堆叠并转换为浮点数
    coords = torch.stack(coords[::-1], dim=0).float()
    # 将坐标扩展到指定批量大小
    return coords[None].repeat(batch, 1, 1, 1)


# 定义一个函数，用于根据给定的光流变形单帧的潜在编码
def warp_single_latent(latent, reference_flow):
    """
    使用给定的光流变形单帧的潜在编码

    参数：
        latent: 单帧的潜在编码
        reference_flow: 用于变形潜在编码的光流

    返回：
        warped: 变形后的潜在编码
    """
    # 获取参考光流的高度和宽度
    _, _, H, W = reference_flow.size()
    # 获取潜在编码的高度和宽度
    _, _, h, w = latent.size()
    # 生成坐标网格
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype)

    # 将光流添加到坐标上
    coords_t0 = coords0 + reference_flow
    # 归一化坐标
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    # 将坐标缩放到[-1, 1]范围
    coords_t0 = coords_t0 * 2.0 - 1.0
    # 使用双线性插值调整坐标大小
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear")
    # 重新排列坐标的维度
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))

    # 根据坐标样本获取变形后的潜在编码
    warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection")
    return warped


# 定义一个函数，用于创建平移运动场
def create_motion_field(motion_field_strength_x, motion_field_strength_y, frame_ids, device, dtype):
    """
    创建平移运动场

    参数：
        motion_field_strength_x: x轴的运动强度
        motion_field_strength_y: y轴的运动强度
        frame_ids: 正在处理的潜在帧的索引。
            在进行分块推理时需要此信息
        device: 设备
        dtype: 数据类型

    返回：

    """
    # 获取帧的数量
    seq_length = len(frame_ids)
    # 创建一个全零的参考光流张量
    reference_flow = torch.zeros((seq_length, 2, 512, 512), device=device, dtype=dtype)
    # 遍历每一帧，生成对应的运动场
    for fr_idx in range(seq_length):
        reference_flow[fr_idx, 0, :, :] = motion_field_strength_x * (frame_ids[fr_idx])
        reference_flow[fr_idx, 1, :, :] = motion_field_strength_y * (frame_ids[fr_idx])
    return reference_flow


# 定义一个函数，用于创建运动场并相应地变形潜在编码
def create_motion_field_and_warp_latents(motion_field_strength_x, motion_field_strength_y, frame_ids, latents):
    """
    创建平移运动并相应地变形潜在编码
    # 参数说明
    Args:
        motion_field_strength_x: motion strength along x-axis  # 表示x轴上的运动强度
        motion_field_strength_y: motion strength along y-axis  # 表示y轴上的运动强度
        frame_ids: indexes of the frames the latents of which are being processed.  # 当前处理的帧的索引
            This is needed when we perform chunk-by-chunk inference  # 进行分块推理时需要使用
        latents: latent codes of frames  # 帧的潜在代码

    Returns:
        warped_latents: warped latents  # 返回经过变形处理的潜在代码
    """
    # 创建运动场，结合x轴和y轴的运动强度
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,  # 传入x轴运动强度
        motion_field_strength_y=motion_field_strength_y,  # 传入y轴运动强度
        frame_ids=frame_ids,  # 传入帧索引
        device=latents.device,  # 使用潜在代码的设备信息
        dtype=latents.dtype,  # 使用潜在代码的数据类型
    )
    # 克隆潜在代码，以便进行后续的变形处理
    warped_latents = latents.clone().detach()
    # 遍历每个变形后的潜在代码
    for i in range(len(warped_latents)):
        # 对每个潜在代码进行单独的变形处理
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    # 返回变形后的潜在代码
    return warped_latents
# 定义一个名为 TextToVideoZeroPipeline 的类，继承自多个父类
class TextToVideoZeroPipeline(
    # 继承 DiffusionPipeline 类
    DiffusionPipeline, 
    # 继承 StableDiffusionMixin 类
    StableDiffusionMixin, 
    # 继承 TextualInversionLoaderMixin 类
    TextualInversionLoaderMixin, 
    # 继承 StableDiffusionLoraLoaderMixin 类
    StableDiffusionLoraLoaderMixin
):
    r"""
    用于使用 Stable Diffusion 进行零-shot 文本到视频生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看父类文档以获取所有管道的通用方法
    （下载、保存、在特定设备上运行等）。

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。
        text_encoder ([`CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer (`CLIPTokenizer`):
            用于标记文本的 [`~transformers.CLIPTokenizer`]。
        unet ([`UNet2DConditionModel`]):
            [`UNet3DConditionModel`] 用于去噪编码的视频潜在。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用的调度器，以去噪编码的图像潜在。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，用于估计生成的图像是否可能被视为攻击性或有害。
            有关模型潜在危害的更多详细信息，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`CLIPImageProcessor`]):
            用于从生成图像中提取特征的 [`CLIPImageProcessor`]；作为 `safety_checker` 的输入。
    """

    # 初始化方法，用于创建类的实例
    def __init__(
        # 变分自编码器 (VAE) 实例
        vae: AutoencoderKL,
        # 文本编码器实例
        text_encoder: CLIPTextModel,
        # 标记器实例
        tokenizer: CLIPTokenizer,
        # UNet 实例，用于去噪处理
        unet: UNet2DConditionModel,
        # 调度器实例
        scheduler: KarrasDiffusionSchedulers,
        # 安全检查器实例
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器实例
        feature_extractor: CLIPImageProcessor,
        # 是否需要安全检查器的布尔值，默认值为 True
        requires_safety_checker: bool = True,
    ):
        # 调用父类的构造函数进行初始化
        super().__init__()
        # 注册多个模块，包括 VAE、文本编码器、分词器、UNet、调度器、安全检查器和特征提取器
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # 如果安全检查器为 None 且需要安全检查器，则发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                # 提示用户禁用安全检查器可能带来的风险和使用条款
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )
        # 根据 VAE 的配置计算缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器，使用之前计算的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def forward_loop(self, x_t0, t0, t1, generator):
        """
        执行 DDPM 前向过程，从时间 t0 到 t1。这与添加具有相应方差的噪声相同。

        Args:
            x_t0:
                时间 t0 时的潜在代码。
            t0:
                t0 时的时间步。
            t1:
                t1 时的时间戳。
            generator (`torch.Generator` 或 `List[torch.Generator]`, *可选*):
                一个 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) 用于生成
                确定性结果。

        Returns:
            x_t1:
                应用前向过程后的 x_t0，从时间 t0 到 t1。
        """
        # 生成与 x_t0 大小相同的随机噪声张量
        eps = randn_tensor(x_t0.size(), generator=generator, dtype=x_t0.dtype, device=x_t0.device)
        # 计算在 t0 到 t1 时间步之间的 alpha 向量的乘积
        alpha_vec = torch.prod(self.scheduler.alphas[t0:t1])
        # 计算 t1 时的潜在代码，结合原始潜在代码和随机噪声
        x_t1 = torch.sqrt(alpha_vec) * x_t0 + torch.sqrt(1 - alpha_vec) * eps
        # 返回时间 t1 的潜在代码
        return x_t1

    def backward_loop(
        self,
        latents,
        timesteps,
        prompt_embeds,
        guidance_scale,
        callback,
        callback_steps,
        num_warmup_steps,
        extra_step_kwargs,
        cross_attention_kwargs=None,
    # 从 diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs 复制
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查高度和宽度是否能被8整除，若不能则抛出错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步数是否为正整数，若不是则抛出错误
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        
        # 检查回调结束时的张量输入是否在已定义的输入列表中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # 检查提示和提示嵌入是否同时存在，若同时存在则抛出错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查提示和提示嵌入是否都未提供，若都未提供则抛出错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示类型是否为字符串或列表，若不是则抛出错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查负提示和负提示嵌入是否同时存在，若同时存在则抛出错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查提示嵌入和负提示嵌入是否同时存在且形状是否一致，若不一致则抛出错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的内容
    # 准备潜在变量以进行模型推理
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，包括批量大小和通道数
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器是否为列表且长度与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            # 如果潜在变量未提供，则生成随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 将提供的潜在变量转移到指定设备
                latents = latents.to(device)
            # 根据调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 不需要计算梯度的调用方法
        @torch.no_grad()
        def __call__(
            # 接受多个参数以生成视频
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int] = 8,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            motion_field_strength_x: float = 12,
            motion_field_strength_y: float = 12,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            callback_steps: Optional[int] = 1,
            t0: int = 44,
            t1: int = 47,
            frame_ids: Optional[List[int]] = None,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 中复制
        def run_safety_checker(self, image, device, dtype):
            # 检查是否定义了安全检查器
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                # 如果输入为张量，进行后处理转换
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果输入为 numpy 数组，转换为 PIL 图像
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 使用特征提取器获取安全检查器输入并转移到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器，返回处理后的图像和 NSFW 概念标识
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像及其 NSFW 概念标识
            return image, has_nsfw_concept
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 中复制
    # 准备额外的参数用于调度器步骤，因为并非所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略该参数
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 其值应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数的字典
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的函数
    def encode_prompt(
        self,
        # 要编码的提示文本
        prompt,
        # 要使用的设备
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否执行无分类器自由引导
        do_classifier_free_guidance,
        # 可选的负提示文本
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 LoRA 缩放因子
        lora_scale: Optional[float] = None,
        # 可选的剪切跳过参数
        clip_skip: Optional[int] = None,
    def decode_latents(self, latents):
        # 将潜在变量缩放回原始大小
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，返回图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像像素值归一化到 [0, 1] 范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 始终转换为 float32 类型，以避免显著开销，并与 bfloat16 兼容
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image
```