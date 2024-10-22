# `.\diffusers\pipelines\deprecated\stable_diffusion_variants\pipeline_stable_diffusion_pix2pix_zero.py`

```py
# 版权声明，说明版权信息及持有者
# Copyright 2024 Pix2Pix Zero Authors and The HuggingFace Team. All rights reserved.
#
# 使用 Apache License 2.0 许可协议
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件仅在遵循许可协议的情况下使用
# you may not use this file except in compliance with the License.
# 许可协议的获取链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或书面协议，否则软件按“原样”分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可协议中关于权限和限制的详细信息
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect  # 导入 inspect 模块，用于获取对象信息
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示

import numpy as np  # 导入 numpy 模块，常用于数值计算
import PIL.Image  # 导入 PIL.Image 模块，用于图像处理
import torch  # 导入 PyTorch 库，主要用于深度学习
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from transformers import (  # 从 transformers 模块导入以下类
    BlipForConditionalGeneration,  # 导入用于条件生成的 Blip 模型
    BlipProcessor,  # 导入 Blip 处理器
    CLIPImageProcessor,  # 导入 CLIP 图像处理器
    CLIPTextModel,  # 导入 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 分词器
)

from ....image_processor import PipelineImageInput, VaeImageProcessor  # 从自定义模块导入图像处理相关类
from ....loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入稳定扩散和文本反转加载器混合类
from ....models import AutoencoderKL, UNet2DConditionModel  # 导入自动编码器和 UNet 模型
from ....models.attention_processor import Attention  # 导入注意力处理器
from ....models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 文本编码器规模的函数
from ....schedulers import DDIMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler  # 导入多种调度器
from ....schedulers.scheduling_ddim_inverse import DDIMInverseScheduler  # 导入 DDIM 反向调度器
from ....utils import (  # 从自定义工具模块导入实用函数和常量
    PIL_INTERPOLATION,  # 导入 PIL 图像插值方法
    USE_PEFT_BACKEND,  # 导入是否使用 PEFT 后端的常量
    BaseOutput,  # 导入基础输出类
    deprecate,  # 导入废弃标记装饰器
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放 Lora 层的函数
    unscale_lora_layers,  # 导入反缩放 Lora 层的函数
)
from ....utils.torch_utils import randn_tensor  # 从 PyTorch 工具模块导入生成随机张量的函数
from ...pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从管道工具模块导入扩散管道和稳定扩散混合类
from ...stable_diffusion.pipeline_output import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出类
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入稳定扩散安全检查器

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

@dataclass  # 将下面的类定义为数据类
class Pix2PixInversionPipelineOutput(BaseOutput, TextualInversionLoaderMixin):  # 定义输出类，继承基础输出和文本反转加载器混合类
    """
    输出类用于稳定扩散管道。

    参数:
        latents (`torch.Tensor`)
            反转的潜在张量
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            长度为 `batch_size` 的去噪 PIL 图像列表或形状为 `(batch_size, height, width,
            num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组呈现扩散管道的去噪图像。
    """

    latents: torch.Tensor  # 定义潜在张量属性
    images: Union[List[PIL.Image.Image], np.ndarray]  # 定义图像属性，可以是图像列表或 numpy 数组

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串的初始部分
```  # 示例文档字符串的开始
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
```py  # 示例文档字符串的结束
```  # 示例文档字符串的结束
    # 示例代码展示如何使用 Diffusers 库进行图像生成
    Examples:
        ```py
        # 导入所需的库
        >>> import requests  # 用于发送 HTTP 请求
        >>> import torch  # 用于处理张量和深度学习模型

        # 从 Diffusers 库导入必要的类
        >>> from diffusers import DDIMScheduler, StableDiffusionPix2PixZeroPipeline

        # 定义下载嵌入文件的函数
        >>> def download(embedding_url, local_filepath):
        ...     # 发送 GET 请求获取嵌入文件
        ...     r = requests.get(embedding_url)
        ...     # 以二进制模式打开本地文件并写入获取的内容
        ...     with open(local_filepath, "wb") as f:
        ...         f.write(r.content)

        # 定义模型检查点的名称
        >>> model_ckpt = "CompVis/stable-diffusion-v1-4"
        # 从预训练模型加载管道并设置数据类型为 float16
        >>> pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
        # 根据管道配置创建 DDIM 调度器
        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        # 将模型移动到 GPU
        >>> pipeline.to("cuda")

        # 定义文本提示
        >>> prompt = "a high resolution painting of a cat in the style of van gough"
        # 定义源和目标嵌入文件的 URL
        >>> source_emb_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/cat.pt"
        >>> target_emb_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/dog.pt"

        # 遍历源和目标嵌入 URL 进行下载
        >>> for url in [source_emb_url, target_emb_url]:
        ...     # 调用下载函数，将文件保存到本地
        ...     download(url, url.split("/")[-1])

        # 从本地加载源嵌入
        >>> src_embeds = torch.load(source_emb_url.split("/")[-1])
        # 从本地加载目标嵌入
        >>> target_embeds = torch.load(target_emb_url.split("/")[-1])
        # 使用管道生成图像
        >>> images = pipeline(
        ...     prompt,  # 输入的文本提示
        ...     source_embeds=src_embeds,  # 源嵌入
        ...     target_embeds=target_embeds,  # 目标嵌入
        ...     num_inference_steps=50,  # 推理步骤数
        ...     cross_attention_guidance_amount=0.15,  # 跨注意力引导的强度
        ... ).images  # 生成的图像

        # 保存生成的第一张图像
        >>> images[0].save("edited_image_dog.png")  # 将图像保存为 PNG 文件
"""
# 示例文档字符串，提供了使用示例和说明
EXAMPLE_INVERT_DOC_STRING = """
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from transformers import BlipForConditionalGeneration, BlipProcessor  # 从 transformers 导入模型和处理器
        >>> from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline  # 从 diffusers 导入调度器和管道

        >>> import requests  # 导入 requests 库，用于发送网络请求
        >>> from PIL import Image  # 从 PIL 导入 Image 类，用于处理图像

        >>> captioner_id = "Salesforce/blip-image-captioning-base"  # 定义图像说明生成模型的 ID
        >>> processor = BlipProcessor.from_pretrained(captioner_id)  # 从预训练模型加载处理器
        >>> model = BlipForConditionalGeneration.from_pretrained(  # 从预训练模型加载图像说明生成模型
        ...     captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True  # 指定数据类型和低内存使用模式
        ... )

        >>> sd_model_ckpt = "CompVis/stable-diffusion-v1-4"  # 定义稳定扩散模型的检查点 ID
        >>> pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(  # 从预训练模型加载 Pix2Pix 零管道
        ...     sd_model_ckpt,  # 指定检查点
        ...     caption_generator=model,  # 指定图像说明生成器
        ...     caption_processor=processor,  # 指定图像说明处理器
        ...     torch_dtype=torch.float16,  # 指定数据类型
        ...     safety_checker=None,  # 关闭安全检查器
        ... )

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)  # 使用调度器配置初始化 DDIM 调度器
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)  # 使用调度器配置初始化 DDIM 反向调度器
        >>> pipeline.enable_model_cpu_offload()  # 启用模型的 CPU 卸载

        >>> img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"  # 定义要处理的图像 URL

        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))  # 从 URL 加载图像并调整大小
        >>> # 生成说明
        >>> caption = pipeline.generate_caption(raw_image)  # 生成图像的说明

        >>> # "a photography of a cat with flowers and dai dai daie - daie - daie kasaii"  # 生成的说明示例
        >>> inv_latents = pipeline.invert(caption, image=raw_image).latents  # 根据说明和原始图像进行反向处理，获取潜变量
        >>> # 我们需要生成源和目标嵌入

        >>> source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]  # 定义源提示列表

        >>> target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]  # 定义目标提示列表

        >>> source_embeds = pipeline.get_embeds(source_prompts)  # 获取源提示的嵌入表示
        >>> target_embeds = pipeline.get_embeds(target_prompts)  # 获取目标提示的嵌入表示
        >>> # 潜变量可以用于编辑真实图像
        >>> # 在使用稳定扩散 2 或其他使用 v-prediction 的模型时
        >>> # 将 `cross_attention_guidance_amount` 设置为 0.01 或更低，以避免输入潜变量梯度爆炸

        >>> image = pipeline(  # 使用管道生成新的图像
        ...     caption,  # 使用生成的说明
        ...     source_embeds=source_embeds,  # 传递源嵌入
        ...     target_embeds=target_embeds,  # 传递目标嵌入
        ...     num_inference_steps=50,  # 指定推理步骤数量
        ...     cross_attention_guidance_amount=0.15,  # 指定交叉注意力指导量
        ...     generator=generator,  # 使用指定的生成器
        ...     latents=inv_latents,  # 传递潜变量
        ...     negative_prompt=caption,  # 使用生成的说明作为负提示
        ... ).images[0]  # 获取生成的图像
        >>> image.save("edited_image.png")  # 保存生成的图像
        ```py
"""


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 导入的 preprocess 函数
def preprocess(image):  # 定义 preprocess 函数，接受图像作为参数
    # 设置一个警告信息，提示用户 preprocess 方法已被弃用，并将在 diffusers 1.0.0 中移除
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 调用 deprecate 函数，记录弃用信息，设定标准警告为 False
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 检查输入的 image 是否是一个 Torch 张量
    if isinstance(image, torch.Tensor):
        # 如果是，直接返回该张量
        return image
    # 检查输入的 image 是否是一个 PIL 图像
    elif isinstance(image, PIL.Image.Image):
        # 如果是，将其封装为一个单元素列表
        image = [image]

    # 检查列表中的第一个元素是否为 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取第一个图像的宽度和高度
        w, h = image[0].size
        # 将宽度和高度调整为 8 的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 对每个图像进行调整大小，转换为 numpy 数组，并在新维度上增加一维
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 将所有图像在第 0 维上连接成一个大的数组
        image = np.concatenate(image, axis=0)
        # 将数据转换为 float32 类型并归一化到 [0, 1] 范围
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序为 (batch_size, channels, height, width)
        image = image.transpose(0, 3, 1, 2)
        # 将像素值范围从 [0, 1] 转换到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 numpy 数组转换为 Torch 张量
        image = torch.from_numpy(image)
    # 检查列表中的第一个元素是否为 Torch 张量
    elif isinstance(image[0], torch.Tensor):
        # 将多个张量在第 0 维上连接成一个大的张量
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image
# 准备 UNet 模型以执行 Pix2Pix Zero 优化
def prepare_unet(unet: UNet2DConditionModel):
    # 初始化一个空字典，用于存储 Pix2Pix Zero 注意力处理器
    pix2pix_zero_attn_procs = {}
    # 遍历 UNet 的注意力处理器的键
    for name in unet.attn_processors.keys():
        # 将处理器名称中的 ".processor" 替换为空
        module_name = name.replace(".processor", "")
        # 获取 UNet 中对应的子模块
        module = unet.get_submodule(module_name)
        # 如果名称包含 "attn2"
        if "attn2" in name:
            # 将处理器设置为 Pix2Pix Zero 模式
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(is_pix2pix_zero=True)
            # 允许该模块进行梯度更新
            module.requires_grad_(True)
        else:
            # 设置为非 Pix2Pix Zero 模式
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(is_pix2pix_zero=False)
            # 不允许该模块进行梯度更新
            module.requires_grad_(False)

    # 设置 UNet 的注意力处理器为修改后的处理器字典
    unet.set_attn_processor(pix2pix_zero_attn_procs)
    # 返回修改后的 UNet 模型
    return unet


class Pix2PixZeroL2Loss:
    # 初始化损失类
    def __init__(self):
        # 设置初始损失值为 0
        self.loss = 0.0

    # 计算损失的方法
    def compute_loss(self, predictions, targets):
        # 更新损失值为预测值与目标值之间的均方差
        self.loss += ((predictions - targets) ** 2).sum((1, 2)).mean(0)


class Pix2PixZeroAttnProcessor:
    """注意力处理器类，用于存储注意力权重。
    在 Pix2Pix Zero 中，该过程发生在交叉注意力块的计算中。"""

    # 初始化注意力处理器
    def __init__(self, is_pix2pix_zero=False):
        # 记录是否为 Pix2Pix Zero 模式
        self.is_pix2pix_zero = is_pix2pix_zero
        # 如果是 Pix2Pix Zero 模式，初始化参考交叉注意力映射
        if self.is_pix2pix_zero:
            self.reference_cross_attn_map = {}

    # 定义调用方法
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None,
    ):
        # 获取隐藏状态的批次大小和序列长度
        batch_size, sequence_length, _ = hidden_states.shape
        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)

        # 如果没有编码器隐藏状态，则使用隐藏状态本身
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要进行交叉规范化
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 将编码器隐藏状态转换为键和值
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 将查询、键和值转换为批次维度
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 如果是 Pix2Pix Zero 模式且时间步不为 None
        if self.is_pix2pix_zero and timestep is not None:
            # 新的记录以保存注意力权重
            if loss is None:
                self.reference_cross_attn_map[timestep.item()] = attention_probs.detach().cpu()
            # 计算损失
            elif loss is not None:
                # 获取之前的注意力概率
                prev_attn_probs = self.reference_cross_attn_map.pop(timestep.item())
                # 计算损失
                loss.compute_loss(attention_probs, prev_attn_probs.to(attention_probs.device))

        # 将注意力概率与值相乘以获得新的隐藏状态
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态转换回头维度
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 返回新的隐藏状态
        return hidden_states
# 定义一个用于像素级图像编辑的管道类，基于 Stable Diffusion
class StableDiffusionPix2PixZeroPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    使用 Pix2Pix Zero 进行像素级图像编辑的管道。基于 Stable Diffusion。

    该模型继承自 [`DiffusionPipeline`]。请查阅超类文档以获取库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等）。

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码到潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`CLIPTextModel`]):
            冻结的文本编码器。Stable Diffusion 使用
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 的文本部分，
            特别是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        tokenizer (`CLIPTokenizer`):
            类的分词器
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)。
        unet ([`UNet2DConditionModel`]): 用于去噪编码图像潜在的条件 U-Net 架构。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 一起使用以去噪编码图像潜在的调度器。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`] 或 [`DDPMScheduler`] 的之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            估计生成图像是否可能被视为攻击性或有害的分类模块。
            请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 以获取详细信息。
        feature_extractor ([`CLIPImageProcessor`]):
            从生成图像中提取特征的模型，以便作为 `safety_checker` 的输入。
        requires_safety_checker (bool):
            管道是否需要安全检查器。如果您公开使用该管道，我们建议将其设置为 True。
    """

    # 定义 CPU 卸载的模型组件顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 可选组件列表
    _optional_components = [
        "safety_checker",
        "feature_extractor",
        "caption_generator",
        "caption_processor",
        "inverse_scheduler",
    ]
    # 从 CPU 卸载中排除的组件列表
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法，定义管道的参数
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器模型
        text_encoder: CLIPTextModel,  # 文本编码器模型
        tokenizer: CLIPTokenizer,  # 分词器模型
        unet: UNet2DConditionModel,  # 条件 U-Net 模型
        scheduler: Union[DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler],  # 调度器类型
        feature_extractor: CLIPImageProcessor,  # 特征提取器模型
        safety_checker: StableDiffusionSafetyChecker,  # 安全检查器模型
        inverse_scheduler: DDIMInverseScheduler,  # 反向调度器
        caption_generator: BlipForConditionalGeneration,  # 描述生成器
        caption_processor: BlipProcessor,  # 描述处理器
        requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
    # 定义一个构造函数
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 如果没有提供安全检查器且需要安全检查器，发出警告
            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    # 输出关于禁用安全检查器的警告信息
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 如果提供了安全检查器但没有提供特征提取器，抛出错误
            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    # 提示用户必须定义特征提取器以使用安全检查器
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 注册模块，设置各个组件
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                caption_processor=caption_processor,
                caption_generator=caption_generator,
                inverse_scheduler=inverse_scheduler,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器，使用 VAE 缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 将配置项注册到当前实例
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 StableDiffusionPipeline 类复制的编码提示的方法
        def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            # 可选参数，表示提示的嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选参数，表示负面提示的嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选参数，表示 LORA 的缩放因子
            lora_scale: Optional[float] = None,
            # 接收任意额外参数
            **kwargs,
    # 开始定义一个方法，处理已弃用的编码提示功能
        ):
            # 定义弃用信息，说明该方法将被移除，并推荐使用新方法
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用弃用函数，记录弃用警告
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用新的编码提示方法，获取结果元组
            prompt_embeds_tuple = self.encode_prompt(
                # 传入提示文本
                prompt=prompt,
                # 设备类型（CPU/GPU）
                device=device,
                # 每个提示生成的图像数量
                num_images_per_prompt=num_images_per_prompt,
                # 是否进行分类器自由引导
                do_classifier_free_guidance=do_classifier_free_guidance,
                # 负面提示文本
                negative_prompt=negative_prompt,
                # 提示嵌入
                prompt_embeds=prompt_embeds,
                # 负面提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,
                # Lora缩放因子
                lora_scale=lora_scale,
                # 其他可选参数
                **kwargs,
            )
    
            # 将返回的元组中的两个嵌入连接起来以兼容旧版
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回最终的提示嵌入
            return prompt_embeds
    
        # 从指定的管道中复制的 encode_prompt 方法定义
        def encode_prompt(
            # 提示文本
            self,
            prompt,
            # 设备类型
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否进行分类器自由引导
            do_classifier_free_guidance,
            # 负面提示文本（可选）
            negative_prompt=None,
            # 提示嵌入（可选）
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入（可选）
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # Lora缩放因子（可选）
            lora_scale: Optional[float] = None,
            # 跳过的clip层数（可选）
            clip_skip: Optional[int] = None,
        # 从指定的管道中复制的 run_safety_checker 方法定义
        def run_safety_checker(self, image, device, dtype):
            # 检查是否存在安全检查器
            if self.safety_checker is None:
                # 如果没有安全检查器，标记为无概念
                has_nsfw_concept = None
            else:
                # 如果图像是张量，则进行后处理以转换为PIL格式
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果不是张量，则将其转换为PIL格式
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 将处理后的图像提取特征，准备进行安全检查
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 使用安全检查器检查图像，返回图像及其概念状态
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回检查后的图像及其概念状态
            return image, has_nsfw_concept
    
        # 从指定的管道中复制的 decode_latents 方法
    # 解码潜在向量并返回生成的图像
    def decode_latents(self, latents):
        # 警告用户该方法已过时，将在1.0.0版本中删除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用deprecate函数记录该方法的弃用信息
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 使用配置的缩放因子对潜在向量进行缩放
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在向量，返回生成的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像值从[-1, 1]映射到[0, 1]并限制范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为float32格式以确保兼容性，并将其转换为numpy数组
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image
    
    # 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备额外的参数以供调度器步骤使用，因调度器的参数签名可能不同
        # eta (η) 仅在DDIMScheduler中使用，其他调度器将忽略它。
        # eta在DDIM论文中对应于η：https://arxiv.org/abs/2010.02502
        # eta的值应在[0, 1]之间
    
        # 检查调度器步骤是否接受eta参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        # 如果接受eta，则将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器步骤是否接受generator参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受generator，则将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数
        return extra_step_kwargs
    
    def check_inputs(
        self,
        prompt,
        source_embeds,
        target_embeds,
        callback_steps,
        prompt_embeds=None,
    ):
        # 检查callback_steps是否为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # 确保source_embeds和target_embeds不能同时未定义
        if source_embeds is None and target_embeds is None:
            raise ValueError("`source_embeds` and `target_embeds` cannot be undefined.")
    
        # 检查prompt和prompt_embeds不能同时被定义
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查prompt和prompt_embeds不能同时未定义
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 确保prompt的类型为str或list
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    #  从 StableDiffusionPipeline 的 prepare_latents 方法复制的内容
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在张量的形状，包括批次大小、通道数、高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且其长度与批次大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果未提供潜在张量，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在张量，则将其移动到指定设备
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在张量
        return latents

    @torch.no_grad()
    def generate_caption(self, images):
        """为给定图像生成标题。"""
        # 初始化生成标题的文本
        text = "a photography of"

        # 保存当前设备
        prev_device = self.caption_generator.device

        # 获取执行设备
        device = self._execution_device
        # 处理输入图像并转换为张量
        inputs = self.caption_processor(images, text, return_tensors="pt").to(
            device=device, dtype=self.caption_generator.dtype
        )
        # 将标题生成器移动到指定设备
        self.caption_generator.to(device)
        # 生成标题输出
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)

        # 将标题生成器移回先前设备
        self.caption_generator.to(prev_device)

        # 解码输出以获取标题
        caption = self.caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # 返回生成的标题
        return caption

    def construct_direction(self, embs_source: torch.Tensor, embs_target: torch.Tensor):
        """构造用于引导图像生成过程的编辑方向。"""
        # 返回目标和源嵌入的均值之差，并增加一个维度
        return (embs_target.mean(0) - embs_source.mean(0)).unsqueeze(0)

    @torch.no_grad()
    def get_embeds(self, prompt: List[str], batch_size: int = 16) -> torch.Tensor:
        # 获取提示的数量
        num_prompts = len(prompt)
        # 初始化嵌入列表
        embeds = []
        # 分批处理提示
        for i in range(0, num_prompts, batch_size):
            prompt_slice = prompt[i : i + batch_size]

            # 将提示转换为输入 ID，进行填充和截断
            input_ids = self.tokenizer(
                prompt_slice,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            # 将输入 ID 移动到文本编码器设备
            input_ids = input_ids.to(self.text_encoder.device)
            # 获取嵌入并追加到列表
            embeds.append(self.text_encoder(input_ids)[0])

        # 将所有嵌入拼接并计算均值
        return torch.cat(embeds, dim=0).mean(0)[None]
    # 准备图像的潜在表示，接收图像和其他参数，返回潜在向量
        def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
            # 检查输入图像的类型是否为有效类型
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                # 抛出类型错误，提示用户输入类型不正确
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )
    
            # 将图像转换到指定的设备和数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果图像有四个通道，直接将其作为潜在表示
            if image.shape[1] == 4:
                latents = image
    
            else:
                # 检查生成器列表的长度是否与批次大小匹配
                if isinstance(generator, list) and len(generator) != batch_size:
                    # 抛出错误，提示生成器列表长度与批次大小不匹配
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
    
                # 如果生成器是列表，逐个图像编码并生成潜在表示
                if isinstance(generator, list):
                    latents = [
                        self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                    ]
                    # 将潜在表示合并到一个张量中
                    latents = torch.cat(latents, dim=0)
                else:
                    # 使用单个生成器编码图像并生成潜在表示
                    latents = self.vae.encode(image).latent_dist.sample(generator)
    
                # 根据配置的缩放因子调整潜在表示
                latents = self.vae.config.scaling_factor * latents
    
            # 检查潜在表示的批次大小是否与请求的匹配
            if batch_size != latents.shape[0]:
                # 如果可以整除，则扩展潜在表示以匹配批次大小
                if batch_size % latents.shape[0] == 0:
                    # 构建弃用消息，提示用户行为即将被移除
                    deprecation_message = (
                        f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                        " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                        " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                        " your script to pass as many initial images as text prompts to suppress this warning."
                    )
                    # 触发弃用警告，提醒用户修改代码
                    deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                    # 计算每个图像需要复制的次数
                    additional_latents_per_image = batch_size // latents.shape[0]
                    # 将潜在表示按需重复以匹配批次大小
                    latents = torch.cat([latents] * additional_latents_per_image, dim=0)
                else:
                    # 抛出错误，提示无法复制图像以匹配批次大小
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                    )
            else:
                # 将潜在表示封装为一个张量
                latents = torch.cat([latents], dim=0)
    
            # 返回最终的潜在表示
            return latents
    # 定义一个获取epsilon的函数，输入为模型输出、样本和时间步
        def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int):
            # 获取反向调度器的预测类型配置
            pred_type = self.inverse_scheduler.config.prediction_type
            # 计算在当前时间步的累积alpha值
            alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep]
    
            # 计算beta值为1减去alpha值
            beta_prod_t = 1 - alpha_prod_t
    
            # 根据预测类型返回相应的结果
            if pred_type == "epsilon":
                return model_output
            elif pred_type == "sample":
                # 根据样本和模型输出计算返回值
                return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (0.5)
            elif pred_type == "v_prediction":
                # 根据alpha和beta值结合模型输出和样本计算返回值
                return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
            else:
                # 如果预测类型无效，抛出异常
                raise ValueError(
                    f"prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
                )
    
        # 定义一个自动相关损失计算的函数，输入为隐藏状态和可选生成器
        def auto_corr_loss(self, hidden_states, generator=None):
            # 初始化正则化损失为0
            reg_loss = 0.0
            # 遍历隐藏状态的每一个维度
            for i in range(hidden_states.shape[0]):
                for j in range(hidden_states.shape[1]):
                    # 选取当前噪声
                    noise = hidden_states[i : i + 1, j : j + 1, :, :]
                    # 进行循环，直到噪声尺寸小于等于8
                    while True:
                        # 随机生成滚动的位移量
                        roll_amount = torch.randint(noise.shape[2] // 2, (1,), generator=generator).item()
                        # 计算并累加正则化损失
                        reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                        reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2
    
                        # 如果噪声尺寸小于等于8，跳出循环
                        if noise.shape[2] <= 8:
                            break
                        # 对噪声进行2x2的平均池化
                        noise = F.avg_pool2d(noise, kernel_size=2)
            # 返回计算得到的正则化损失
            return reg_loss
    
        # 定义一个计算KL散度的函数，输入为隐藏状态
        def kl_divergence(self, hidden_states):
            # 计算隐藏状态的均值
            mean = hidden_states.mean()
            # 计算隐藏状态的方差
            var = hidden_states.var()
            # 返回KL散度的计算结果
            return var + mean**2 - 1 - torch.log(var + 1e-7)
    
        # 定义调用函数，使用@torch.no_grad()装饰器禁止梯度计算
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 输入参数包括提示、源和目标嵌入、图像的高和宽、推理步骤等
            prompt: Optional[Union[str, List[str]]] = None,
            source_embeds: torch.Tensor = None,
            target_embeds: torch.Tensor = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            cross_attention_guidance_amount: float = 0.1,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
        # 使用@torch.no_grad()和装饰器替换文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_INVERT_DOC_STRING)
    # 定义一个名为 invert 的方法，包含多个可选参数
        def invert(
            # 输入提示，默认为 None
            self,
            prompt: Optional[str] = None,
            # 输入图像，默认为 None
            image: PipelineImageInput = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 指导比例，默认为 1
            guidance_scale: float = 1,
            # 随机数生成器，可以是单个或多个生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 跨注意力引导量，默认为 0.1
            cross_attention_guidance_amount: float = 0.1,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 回调函数，默认为 None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调步骤，默认为 1
            callback_steps: Optional[int] = 1,
            # 跨注意力参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 自动相关的权重，默认为 20.0
            lambda_auto_corr: float = 20.0,
            # KL 散度的权重，默认为 20.0
            lambda_kl: float = 20.0,
            # 正则化步骤的数量，默认为 5
            num_reg_steps: int = 5,
            # 自动相关滚动的数量，默认为 5
            num_auto_corr_rolls: int = 5,
```