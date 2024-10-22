# `.\diffusers\pipelines\ledits_pp\pipeline_leditspp_stable_diffusion.py`

```py
# 导入 inspect 模块，用于获取对象的信息
import inspect
# 导入 math 模块，提供数学函数
import math
# 从 itertools 导入 repeat，用于生成重复元素的迭代器
from itertools import repeat
# 从 typing 导入常用类型，用于类型注解
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 torch 库，用于深度学习
import torch
# 从 torch.nn.functional 导入 F，提供各种神经网络功能
import torch.nn.functional as F
# 从 packaging 导入 version，用于版本比较
from packaging import version
# 从 transformers 导入 CLIP 相关组件，用于图像和文本处理
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# 从上级目录导入 FrozenDict，用于不可变字典
from ...configuration_utils import FrozenDict
# 从上级目录导入图像处理相关类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 从上级目录导入加载器混合类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从上级目录导入模型
from ...models import AutoencoderKL, UNet2DConditionModel
# 从注意力处理器模块导入 Attention 和 AttnProcessor
from ...models.attention_processor import Attention, AttnProcessor
# 从 lora 模块导入调整 LORA 比例的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从稳定扩散管道导入安全检查器
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 从调度器导入 DDIM 和 DPMSolver 多步调度器
from ...schedulers import DDIMScheduler, DPMSolverMultistepScheduler
# 从工具模块导入各种工具函数
from ...utils import (
    USE_PEFT_BACKEND,  # 用于是否使用 PEFT 后端的标志
    deprecate,  # 用于标记弃用的函数
    logging,  # 用于日志记录
    replace_example_docstring,  # 用于替换示例文档字符串的函数
    scale_lora_layers,  # 用于缩放 LORA 层的函数
    unscale_lora_layers,  # 用于反缩放 LORA 层的函数
)
# 从 torch_utils 导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入扩散管道类
from ..pipeline_utils import DiffusionPipeline
# 从管道输出模块导入管道输出类
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput

# 创建日志记录器，使用当前模块名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，提供使用示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import PIL  # 导入 PIL 库，用于图像处理
        >>> import requests  # 导入 requests 库，用于 HTTP 请求
        >>> import torch  # 导入 torch 库，用于深度学习
        >>> from io import BytesIO  # 从 io 导入 BytesIO，用于字节流处理

        >>> from diffusers import LEditsPPPipelineStableDiffusion  # 从 diffusers 导入 LEditsPPPipelineStableDiffusion 类
        >>> from diffusers.utils import load_image  # 从 diffusers.utils 导入 load_image 函数

        >>> pipe = LEditsPPPipelineStableDiffusion.from_pretrained(  # 从预训练模型创建管道
        ...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16  # 指定模型路径和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU

        >>> img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/cherry_blossom.png"  # 图像的 URL
        >>> image = load_image(img_url).convert("RGB")  # 加载图像并转换为 RGB 格式

        >>> _ = pipe.invert(image=image, num_inversion_steps=50, skip=0.1)  # 对图像进行反演处理

        >>> edited_image = pipe(  # 使用管道进行图像编辑
        ...     editing_prompt=["cherry blossom"], edit_guidance_scale=10.0, edit_threshold=0.75  # 设置编辑提示和参数
        ... ).images[0]  # 获取编辑后的图像
        ```py
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline 修改的类
class LeditsAttentionStore:
    @staticmethod
    # 静态方法，获取一个空的注意力存储
    def get_empty_store():
        # 返回一个字典，包含不同层次的注意力存储列表
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}
    # 定义一个可调用的方法，接收注意力权重和其他参数
    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts, PnP=False):
        # 注意力权重的形状为：批大小 * 头大小, 序列长度查询, 序列长度键
        if attn.shape[1] <= self.max_size:  # 检查序列长度是否小于或等于最大大小
            bs = 1 + int(PnP) + editing_prompts  # 计算批次大小
            skip = 2 if PnP else 1  # 确定跳过的步骤：如果是 PnP，跳过 2，否则跳过 1
            # 将注意力权重拆分为指定批大小的张量，并调整维度顺序
            attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3)
            source_batch_size = int(attn.shape[1] // bs)  # 计算源批次大小
            # 调用 forward 方法，传入处理后的注意力权重
            self.forward(attn[:, skip * source_batch_size :], is_cross, place_in_unet)

    # 定义 forward 方法，用于处理注意力权重
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 生成键值，用于存储不同位置的注意力
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # 将当前注意力权重添加到对应的存储列表中
        self.step_store[key].append(attn)

    # 定义在步骤之间的操作，可以选择存储当前步骤
    def between_steps(self, store_step=True):
        if store_step:  # 如果需要存储当前步骤
            if self.average:  # 如果启用平均
                if len(self.attention_store) == 0:  # 如果注意力存储为空
                    self.attention_store = self.step_store  # 直接赋值
                else:
                    # 对现有注意力存储进行更新
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:  # 如果不启用平均
                if len(self.attention_store) == 0:  # 如果注意力存储为空
                    self.attention_store = [self.step_store]  # 作为第一个存储项
                else:
                    self.attention_store.append(self.step_store)  # 添加新的存储项

            self.cur_step += 1  # 更新当前步骤计数
        # 重置当前步骤的存储
        self.step_store = self.get_empty_store()

    # 定义获取特定步骤注意力的方法
    def get_attention(self, step: int):
        if self.average:  # 如果启用平均
            # 计算平均注意力，按当前步骤进行归一化
            attention = {
                key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
            }
        else:  # 否则，确保提供了步骤
            assert step is not None  # 断言步骤不为 None
            attention = self.attention_store[step]  # 获取指定步骤的注意力
        return attention  # 返回注意力

    # 定义聚合注意力的方法，处理多个参数
    def aggregate_attention(
        self, attention_maps, prompts, res: Union[int, Tuple[int]], from_where: List[str], is_cross: bool, select: int
    ):
        out = [[] for x in range(self.batch_size)]  # 初始化输出列表
        if isinstance(res, int):  # 如果分辨率为整数
            num_pixels = res**2  # 计算像素数量
            resolution = (res, res)  # 设置分辨率
        else:  # 如果分辨率为元组
            num_pixels = res[0] * res[1]  # 计算像素数量
            resolution = res[:2]  # 设置分辨率

        # 遍历指定来源，提取注意力图
        for location in from_where:
            for bs_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                for batch, item in enumerate(bs_item):
                    if item.shape[1] == num_pixels:  # 检查当前项的形状是否匹配
                        # 重塑注意力图并选择指定项
                        cross_maps = item.reshape(len(prompts), -1, *resolution, item.shape[-1])[select]
                        out[batch].append(cross_maps)  # 将提取的映射添加到输出

        # 合并输出，计算每个批次的平均值
        out = torch.stack([torch.cat(x, dim=0) for x in out])  # 在第一个维度上堆叠
        out = out.sum(1) / out.shape[1]  # 在头维度上取平均
        return out  # 返回聚合后的输出
    # 初始化方法，用于创建类的实例，接受多个参数
        def __init__(self, average: bool, batch_size=1, max_resolution=16, max_size: int = None):
            # 获取一个空的存储结构，用于保存步骤数据
            self.step_store = self.get_empty_store()
            # 初始化注意力存储列表
            self.attention_store = []
            # 当前步骤计数器初始化为 0
            self.cur_step = 0
            # 设置是否计算平均值的标志
            self.average = average
            # 设置批处理大小，默认值为 1
            self.batch_size = batch_size
            # 如果没有指定最大大小，则计算为最大分辨率的平方
            if max_size is None:
                self.max_size = max_resolution**2
            # 如果指定了最大大小且没有指定最大分辨率，则使用指定的最大大小
            elif max_size is not None and max_resolution is None:
                self.max_size = max_size
            # 如果同时指定了最大分辨率和最大大小，则抛出错误
            else:
                raise ValueError("Only allowed to set one of max_resolution or max_size")
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.GaussianSmoothing 修改而来
class LeditsGaussianSmoothing:
    # 初始化函数，接收设备参数
    def __init__(self, device):
        # 定义高斯核的大小
        kernel_size = [3, 3]
        # 定义高斯核的标准差
        sigma = [0.5, 0.5]

        # 高斯核是每个维度高斯函数的乘积
        kernel = 1
        # 创建网格以便于计算高斯核
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        # 遍历每个维度的大小、标准差和网格
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            # 计算高斯核的均值
            mean = (size - 1) / 2
            # 计算高斯核值
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # 确保高斯核的值之和为1
        kernel = kernel / torch.sum(kernel)

        # 将高斯核重塑为深度卷积权重
        kernel = kernel.view(1, 1, *kernel.size())
        # 重复高斯核以适应卷积层的形状
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        # 将权重移动到指定设备上
        self.weight = kernel.to(device)

    # 调用函数，用于应用高斯滤波
    def __call__(self, input):
        """
        参数:
        对输入应用高斯滤波。
            input (torch.Tensor): 需要应用高斯滤波的输入。
        返回:
            filtered (torch.Tensor): 经过滤波的输出。
        """
        # 使用卷积操作应用高斯滤波
        return F.conv2d(input, weight=self.weight.to(input.dtype))


class LEDITSCrossAttnProcessor:
    # 初始化函数，接收多个参数以设置注意力处理器
    def __init__(self, attention_store, place_in_unet, pnp, editing_prompts):
        # 存储注意力的变量
        self.attnstore = attention_store
        # 设置在 UNet 中的位置
        self.place_in_unet = place_in_unet
        # 存储编辑提示
        self.editing_prompts = editing_prompts
        # 存储 PnP 相关的变量
        self.pnp = pnp

    # 调用函数，处理注意力和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        temb=None,
    # 处理输入，确定批量大小和序列长度
        ):
            batch_size, sequence_length, _ = (
                # 如果没有编码器隐藏状态，则使用隐藏状态形状，否则使用编码器隐藏状态形状
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            # 准备注意力掩码，以便后续的注意力计算
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    
            # 将隐藏状态转换为查询向量
            query = attn.to_q(hidden_states)
    
            # 如果没有编码器隐藏状态，使用隐藏状态；否则，如果需要规范化，则规范化编码器隐藏状态
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
            # 将编码器隐藏状态转换为键向量
            key = attn.to_k(encoder_hidden_states)
            # 将编码器隐藏状态转换为值向量
            value = attn.to_v(encoder_hidden_states)
    
            # 将查询向量转换为批处理维度
            query = attn.head_to_batch_dim(query)
            # 将键向量转换为批处理维度
            key = attn.head_to_batch_dim(key)
            # 将值向量转换为批处理维度
            value = attn.head_to_batch_dim(value)
    
            # 计算注意力得分
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # 存储注意力得分，标记为交叉注意力
            self.attnstore(
                attention_probs,
                is_cross=True,
                place_in_unet=self.place_in_unet,
                editing_prompts=self.editing_prompts,
                PnP=self.pnp,
            )
    
            # 通过值向量和注意力得分计算新的隐藏状态
            hidden_states = torch.bmm(attention_probs, value)
            # 将隐藏状态转换回头部维度
            hidden_states = attn.batch_to_head_dim(hidden_states)
    
            # 进行线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # 进行 dropout 操作
            hidden_states = attn.to_out[1](hidden_states)
    
            # 重新缩放输出
            hidden_states = hidden_states / attn.rescale_output_factor
            # 返回最终的隐藏状态
            return hidden_states
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 导入的 rescale_noise_cfg 函数
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 对 `noise_cfg` 进行重新缩放。基于 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的发现。见第 3.4 节
    """
    # 计算文本噪声预测的标准差，保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算配置噪声的标准差，保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据标准差调整指导结果（修正过度曝光）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 按照指导缩放因子混合原始结果，以避免图像“平淡”
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回调整后的噪声配置
    return noise_cfg


# 定义 LEditsPPPipelineStableDiffusion 类，继承多个混合类
class LEditsPPPipelineStableDiffusion(
    DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
):
    """
    使用 LEDits++ 和 Stable Diffusion 进行文本图像编辑的管道。

    此模型继承自 [`DiffusionPipeline`]，并建立在 [`StableDiffusionPipeline`] 之上。查看超类
    文档以了解所有管道实现的通用方法（下载、保存、在特定设备上运行等）。
    # 文档字符串，定义类的参数及其描述
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器模型，用于将图像编码和解码为潜在表示
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，Stable Diffusion 使用 CLIP 的文本部分
            Frozen text-encoder. Stable Diffusion uses the text portion of
            # 指定的 CLIP 模型变体
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            # 指定的 CLIP-ViT 大模型变体
            [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # CLIPTokenizer 类的分词器，用于文本处理
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): 
            # 条件 U-Net 架构，用于对编码的图像潜在表示进行去噪
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]):
            # 调度器，用于与 `unet` 结合去噪编码的图像潜在表示
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            # 允许的调度器类型，若传入其他类型则默认为 DPMSolverMultistepScheduler
            [`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]. If any other scheduler is passed it will
            automatically be set to [`DPMSolverMultistepScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 分类模块，评估生成的图像是否可能被视为冒犯或有害
            Classification module that estimates whether generated images could be considered offensive or harmful.
            # 有关详细信息，请参考模型卡
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # 从生成的图像中提取特征，以作为 `safety_checker` 的输入
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # 定义模型在 CPU 卸载时的顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义在 CPU 卸载时排除的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调张量输入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 定义可选组件
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    # 初始化方法，定义类的构造函数
    def __init__(
        self,
        # 变分自编码器模型实例
        vae: AutoencoderKL,
        # 文本编码器实例
        text_encoder: CLIPTextModel,
        # 分词器实例
        tokenizer: CLIPTokenizer,
        # U-Net 模型实例
        unet: UNet2DConditionModel,
        # 调度器实例
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
        # 安全检查器实例
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器实例
        feature_extractor: CLIPImageProcessor,
        # 指示是否需要安全检查器的布尔值，默认为 True
        requires_safety_checker: bool = True,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的内容
    # 运行安全检查器，对输入图像进行安全性检测
        def run_safety_checker(self, image, device, dtype):
            # 检查安全检查器是否已初始化
            if self.safety_checker is None:
                # 若未初始化，NSFW 概念标志设置为 None
                has_nsfw_concept = None
            else:
                # 检查输入图像是否为张量
                if torch.is_tensor(image):
                    # 如果是张量，则处理图像为 PIL 格式
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果不是张量，则将 NumPy 数组转换为 PIL 图像
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 提取图像特征，返回张量并移动到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 执行安全检查，返回处理后的图像和 NSFW 概念标志
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和 NSFW 概念标志
            return image, has_nsfw_concept
    
        # 从 StableDiffusionPipeline 复制的解码潜变量方法
        def decode_latents(self, latents):
            # 提示解码方法已弃用，未来版本将移除
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 根据 VAE 配置缩放潜变量
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜变量，获取图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像数据归一化并限制在 [0, 1] 范围内
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为 float32 格式，以兼容 bfloat16，且不会造成显著开销
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回处理后的图像
            return image
    
        # 从 StableDiffusionPipeline 复制的准备额外步骤参数的方法
        def prepare_extra_step_kwargs(self, eta, generator=None):
            # 准备调度器步骤的额外参数，不同调度器的参数签名可能不同
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会忽略
            # eta 对应于 DDIM 论文中的 η，范围应在 [0, 1] 之间
    
            # 检查调度器步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            # 如果接受 eta，则将其添加到额外参数中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，则将其添加到额外参数中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    
        # 从 StableDiffusionPipeline 复制的输入检查方法
        def check_inputs(
            self,
            # 定义检查的输入参数，负提示和编辑提示嵌入等
            negative_prompt=None,
            editing_prompt_embeddings=None,
            negative_prompt_embeds=None,
            callback_on_step_end_tensor_inputs=None,
    ):
        # 检查回调输入是否为空，并确保所有输入都在回调张量输入列表中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有输入不在回调张量输入列表中，抛出值错误
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        # 检查负提示和负提示嵌入是否同时存在
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果同时存在，抛出值错误
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查编辑提示嵌入和负提示嵌入是否同时存在
        if editing_prompt_embeddings is not None and negative_prompt_embeds is not None:
            # 检查两者形状是否相同
            if editing_prompt_embeddings.shape != negative_prompt_embeds.shape:
                # 如果形状不匹配，抛出值错误
                raise ValueError(
                    "`editing_prompt_embeddings` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `editing_prompt_embeddings` {editing_prompt_embeddings.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 修改而来
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents):
        # 计算预期的张量形状
        # shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        # 如果输入的 latents 形状与预期形状不匹配，抛出值错误
        # if latents.shape != shape:
        #    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # 将输入的 latents 张量移动到指定设备
        latents = latents.to(device)

        # 根据调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的 latents
        return latents

    def prepare_unet(self, attention_store, PnP: bool = False):
        # 创建一个空字典用于存储注意力处理器
        attn_procs = {}
        # 遍历 UNet 中的注意力处理器键
        for name in self.unet.attn_processors.keys():
            # 根据名称前缀确定其在 UNet 中的位置
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            # 根据名称选择合适的注意力处理器
            if "attn2" in name and place_in_unet != "mid":
                attn_procs[name] = LEDITSCrossAttnProcessor(
                    attention_store=attention_store,
                    place_in_unet=place_in_unet,
                    pnp=PnP,
                    editing_prompts=self.enabled_editing_prompts,
                )
            else:
                attn_procs[name] = AttnProcessor()

        # 将注意力处理器设置到 UNet 中
        self.unet.set_attn_processor(attn_procs)
    # 定义一个编码提示的方法，接收多个参数
    def encode_prompt(
        self,  # 当前对象的引用
        device,  # 设备类型，例如 CPU 或 GPU
        num_images_per_prompt,  # 每个提示生成的图像数量
        enable_edit_guidance,  # 是否启用编辑引导
        negative_prompt=None,  # 可选的负面提示
        editing_prompt=None,  # 可选的编辑提示
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        editing_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的编辑提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
        clip_skip: Optional[int] = None,  # 可选的跳过的剪辑层数
    # 获取指导重缩放的属性
    @property
    def guidance_rescale(self):  
        return self._guidance_rescale  # 返回指导重缩放的值

    # 获取剪辑跳过的属性
    @property
    def clip_skip(self):  
        return self._clip_skip  # 返回剪辑跳过的值

    # 获取交叉注意力参数的属性
    @property
    def cross_attention_kwargs(self):  
        return self._cross_attention_kwargs  # 返回交叉注意力参数

    # 装饰器，指示不计算梯度
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的方法，接收多个可选参数
    def __call__(
        self,  # 方法本身的引用
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选的生成器
        output_type: Optional[str] = "pil",  # 输出类型，默认是 PIL
        return_dict: bool = True,  # 是否返回字典格式的输出
        editing_prompt: Optional[Union[str, List[str]]] = None,  # 可选的编辑提示
        editing_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的编辑提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,  # 是否反向编辑方向
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,  # 编辑引导缩放因子
        edit_warmup_steps: Optional[Union[int, List[int]]] = 0,  # 编辑热身步数
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,  # 编辑冷却步数
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,  # 编辑阈值
        user_mask: Optional[torch.Tensor] = None,  # 可选的用户掩码
        sem_guidance: Optional[List[torch.Tensor]] = None,  # 可选的语义引导
        use_cross_attn_mask: bool = False,  # 是否使用交叉注意力掩码
        use_intersect_mask: bool = True,  # 是否使用交集掩码
        attn_store_steps: Optional[List[int]] = [],  # 存储注意力的步骤列表
        store_averaged_over_steps: bool = True,  # 是否在步骤上存储平均值
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数
        guidance_rescale: float = 0.0,  # 指导重缩放的默认值
        clip_skip: Optional[int] = None,  # 可选的剪辑跳过层数
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,  # 可选的步骤结束回调
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # 默认的步骤结束回调输入
        **kwargs,  # 接收其他关键字参数
    # 装饰器，指示不计算梯度
    @torch.no_grad()
    # 定义反转的方法，接收多个参数
    def invert(
        self,  # 方法本身的引用
        image: PipelineImageInput,  # 输入图像
        source_prompt: str = "",  # 源提示字符串
        source_guidance_scale: float = 3.5,  # 源指导缩放因子
        num_inversion_steps: int = 30,  # 反转步骤数量
        skip: float = 0.15,  # 跳过的比例
        generator: Optional[torch.Generator] = None,  # 可选的生成器
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数
        clip_skip: Optional[int] = None,  # 可选的剪辑跳过层数
        height: Optional[int] = None,  # 可选的图像高度
        width: Optional[int] = None,  # 可选的图像宽度
        resize_mode: Optional[str] = "default",  # 图像调整大小的模式
        crops_coords: Optional[Tuple[int, int, int, int]] = None,  # 可选的裁剪坐标
    # 装饰器，指示不计算梯度
    @torch.no_grad()
    # 定义一个编码图像的函数，接收多个参数以处理图像
    def encode_image(self, image, dtype=None, height=None, width=None, resize_mode="default", crops_coords=None):
        # 使用图像处理器预处理输入图像，调整其高度、宽度和裁剪坐标
        image = self.image_processor.preprocess(
            image=image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        # 使用图像处理器后处理图像，输出类型为 PIL 图像
        resized = self.image_processor.postprocess(image=image, output_type="pil")
    
        # 检查输入图像的最大维度是否超过默认分辨率的 1.5 倍
        if max(image.shape[-2:]) > self.vae.config["sample_size"] * 1.5:
            # 记录警告信息，提示输入图像分辨率过高可能导致输出图像有严重伪影
            logger.warning(
                "Your input images far exceed the default resolution of the underlying diffusion model. "
                "The output images may contain severe artifacts! "
                "Consider down-sampling the input using the `height` and `width` parameters"
            )
        # 将图像转换为指定的数据类型
        image = image.to(dtype)
    
        # 使用 VAE 编码器对图像进行编码，获取潜在分布的模式
        x0 = self.vae.encode(image.to(self.device)).latent_dist.mode()
        # 将编码结果转换为指定的数据类型
        x0 = x0.to(dtype)
        # 将编码结果乘以缩放因子
        x0 = self.vae.config.scaling_factor * x0
        # 返回编码后的结果和处理后的图像
        return x0, resized
# 计算 DDIM 噪声
def compute_noise_ddim(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. 获取前一个时间步的值（t-1）
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. 计算 alphas 和 betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]  # 当前时间步的累积 alpha
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )  # 前一个时间步的累积 alpha，若为负则使用最终 alpha

    beta_prod_t = 1 - alpha_prod_t  # 当前时间步的 beta

    # 3. 从预测噪声计算预测的原始样本，也称为“预测的 x_0”
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. 裁剪“预测的 x_0”
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)  # 将值限制在 -1 到 1 之间

    # 5. 计算方差：“sigma_t(η)” -> 参见公式 (16)
    variance = scheduler._get_variance(timestep, prev_timestep)  # 获取方差
    std_dev_t = eta * variance ** (0.5)  # 计算标准差

    # 6. 计算指向 x_t 的方向，参见公式 (12)
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

    # 修改以返回更新后的 xtm1（避免误差累积）
    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction  # 计算 mu_xt
    if variance > 0.0:
        noise = (prev_latents - mu_xt) / (variance ** (0.5) * eta)  # 计算噪声
    else:
        noise = torch.tensor([0.0]).to(latents.device)  # 方差为零时噪声设为零

    return noise, mu_xt + (eta * variance**0.5) * noise  # 返回噪声和更新后的样本


# 计算 SDE DPM PP 二阶噪声
def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    def first_order_update(model_output, sample):  # 定义一阶更新
        sigma_t, sigma_s = scheduler.sigmas[scheduler.step_index + 1], scheduler.sigmas[scheduler.step_index]  # 获取当前和前一个 sigma
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)  # 将 sigma 转换为 alpha
        alpha_s, sigma_s = scheduler._sigma_to_alpha_sigma_t(sigma_s)  # 将前一个 sigma 转换为 alpha
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)  # 计算 lambda_t
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)  # 计算 lambda_s

        h = lambda_t - lambda_s  # 计算 h

        mu_xt = (sigma_t / sigma_s * torch.exp(-h)) * sample + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output  # 计算 mu_xt

        mu_xt = scheduler.dpm_solver_first_order_update(  # 更新 mu_xt
            model_output=model_output, sample=sample, noise=torch.zeros_like(sample)  # 将噪声设为零
        )

        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))  # 计算 sigma
        if sigma > 0.0:
            noise = (prev_latents - mu_xt) / sigma  # 计算噪声
        else:
            noise = torch.tensor([0.0]).to(sample.device)  # 方差为零时噪声设为零

        prev_sample = mu_xt + sigma * noise  # 计算前一个样本
        return noise, prev_sample  # 返回噪声和前一个样本
    # 定义二阶更新函数，接受模型输出列表和样本
        def second_order_update(model_output_list, sample):  # timestep_list, prev_timestep, sample):
            # 获取当前和前后时刻的 sigma 值
            sigma_t, sigma_s0, sigma_s1 = (
                scheduler.sigmas[scheduler.step_index + 1],  # 当前时刻的 sigma
                scheduler.sigmas[scheduler.step_index],      # 上一时刻的 sigma
                scheduler.sigmas[scheduler.step_index - 1],  # 更早时刻的 sigma
            )
    
            # 将 sigma 转换为 alpha 和 sigma_t
            alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s0, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigma_s0)
            alpha_s1, sigma_s1 = scheduler._sigma_to_alpha_sigma_t(sigma_s1)
    
            # 计算 lambda 值
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)  # 当前时刻的 lambda
            lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)  # 上一时刻的 lambda
            lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)  # 更早时刻的 lambda
    
            # 获取最后两个模型输出
            m0, m1 = model_output_list[-1], model_output_list[-2]
    
            # 计算 h 和 r0 值
            h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1  # h 和 h_0
            r0 = h_0 / h  # r0 的计算
            D0, D1 = m0, (1.0 / r0) * (m0 - m1)  # D0 和 D1 的计算
    
            # 计算 mu_xt
            mu_xt = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample  # 根据样本和 sigma 计算
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0  # 加上 D0 的贡献
                + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1  # 加上 D1 的一半贡献
            )
    
            # 计算 sigma
            sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))  # 最终的 sigma 计算
            # 根据 sigma 计算噪声
            if sigma > 0.0:
                noise = (prev_latents - mu_xt) / sigma  # 正常情况下计算噪声
            else:
                noise = torch.tensor([0.0]).to(sample.device)  # sigma 为零时，噪声为零
    
            # 计算前一个样本
            prev_sample = mu_xt + sigma * noise  # 最终样本的计算
    
            return noise, prev_sample  # 返回噪声和前一个样本
    
        # 初始化 step_index，如果未定义的话
        if scheduler.step_index is None:
            scheduler._init_step_index(timestep)  # 初始化步进索引
    
        # 将模型输出转换为合适格式
        model_output = scheduler.convert_model_output(model_output=noise_pred, sample=latents)
        # 更新模型输出列表
        for i in range(scheduler.config.solver_order - 1):
            scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]  # 向前移动输出
        scheduler.model_outputs[-1] = model_output  # 保存当前模型输出
    
        # 根据 lower_order_nums 选择更新方法
        if scheduler.lower_order_nums < 1:
            noise, prev_sample = first_order_update(model_output, latents)  # 一阶更新
        else:
            noise, prev_sample = second_order_update(scheduler.model_outputs, latents)  # 二阶更新
    
        # 更新 lower_order_nums
        if scheduler.lower_order_nums < scheduler.config.solver_order:
            scheduler.lower_order_nums += 1  # 增加 lower_order_nums
    
        # 完成后增加步进索引
        scheduler._step_index += 1  # 增加步进索引
    
        return noise, prev_sample  # 返回噪声和前一个样本
# 定义一个计算噪声的函数，接收调度器和可变参数
def compute_noise(scheduler, *args):
    # 检查调度器是否为 DDIMScheduler 类型
    if isinstance(scheduler, DDIMScheduler):
        # 调用对应的函数计算 DDIM 噪声
        return compute_noise_ddim(scheduler, *args)
    # 检查调度器是否为 DPMSolverMultistepScheduler，并验证其配置
    elif (
        isinstance(scheduler, DPMSolverMultistepScheduler)
        and scheduler.config.algorithm_type == "sde-dpmsolver++"
        and scheduler.config.solver_order == 2
    ):
        # 调用对应的函数计算 SDE-DPM 噪声（2阶）
        return compute_noise_sde_dpm_pp_2nd(scheduler, *args)
    else:
        # 如果不支持的调度器类型，抛出未实现的错误
        raise NotImplementedError
```