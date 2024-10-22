# `.\diffusers\pipelines\ledits_pp\pipeline_leditspp_stable_diffusion_xl.py`

```py
# 版权声明，表示此文件的版权信息及归属
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# 许可条款，声明该文件根据 Apache License 2.0 进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件的使用需遵循许可证的规定
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非在适用法律或书面协议中另有规定，软件按“原样”提供
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示的担保
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参阅许可证以了解有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect  # 导入 inspect 模块，用于获取对象的内部信息
import math  # 导入 math 模块，提供数学函数
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 从 typing 导入类型注解，用于类型提示

import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块，通常用于激活函数等
from transformers import (  # 从 transformers 库导入多个类和函数
    CLIPImageProcessor,  # 导入 CLIP 图像处理器类
    CLIPTextModel,  # 导入 CLIP 文本模型类
    CLIPTextModelWithProjection,  # 导入带有投影的 CLIP 文本模型类
    CLIPTokenizer,  # 导入 CLIP 分词器类
    CLIPVisionModelWithProjection,  # 导入带有投影的 CLIP 视觉模型类
)

from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从相对路径导入图像处理相关类
from ...loaders import (  # 从相对路径导入加载器相关类
    FromSingleFileMixin,  # 单文件混合加载器
    IPAdapterMixin,  # IP 适配器混合类
    StableDiffusionXLLoraLoaderMixin,  # Stable Diffusion XL LoRA 加载器混合类
    TextualInversionLoaderMixin,  # 文本反演加载器混合类
)
from ...models import AutoencoderKL, UNet2DConditionModel  # 从相对路径导入模型类
from ...models.attention_processor import (  # 从相对路径导入注意力处理器相关类
    Attention,  # 注意力机制类
    AttnProcessor,  # 注意力处理器基类
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 从相对路径导入调整 LoRA 比例的函数
from ...schedulers import DDIMScheduler, DPMSolverMultistepScheduler  # 从相对路径导入调度器类
from ...utils import (  # 从相对路径导入实用工具函数
    USE_PEFT_BACKEND,  # 用于 PEFT 后端的常量
    is_invisible_watermark_available,  # 检查是否可用隐形水印的函数
    is_torch_xla_available,  # 检查是否可用 Torch XLA 的函数
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 缩放 LoRA 层的函数
    unscale_lora_layers,  # 反缩放 LoRA 层的函数
)
from ...utils.torch_utils import randn_tensor  # 从相对路径导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline  # 从相对路径导入扩散管道类
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput  # 导入管道输出相关类


# 如果隐形水印可用，则导入相应的水印处理类
if is_invisible_watermark_available():
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker  # 导入 Stable Diffusion XL 水印类

# 检查是否可用 Torch XLA，若可用则导入其核心模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 Torch XLA 核心模型模块

    XLA_AVAILABLE = True  # 设置标志，表示 XLA 可用
else:
    XLA_AVAILABLE = False  # 设置标志，表示 XLA 不可用

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于后续日志记录使用

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串的多行字符串，用于文档或示例说明
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
    # 示例代码，展示如何使用 LE edits PP 管道进行图像编辑
        Examples:
            ```py
            # 导入所需的库
            >>> import torch  # 导入 PyTorch 库
            >>> import PIL  # 导入 Python Imaging Library (PIL)
            >>> import requests  # 导入请求库以获取网络资源
            >>> from io import BytesIO  # 导入 BytesIO 用于字节流处理
    
            # 从 diffusers 库导入 LEditsPPPipelineStableDiffusionXL 类
            >>> from diffusers import LEditsPPPipelineStableDiffusionXL
    
            # 创建一个 LEditsPPPipelineStableDiffusionXL 的实例，使用预训练模型
            >>> pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
            ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
            ... )
            # 将管道移动到 CUDA 设备以加速计算
            >>> pipe = pipe.to("cuda")
    
            # 定义一个函数下载并返回图像
            >>> def download_image(url):
            ...     # 发送 GET 请求获取图像
            ...     response = requests.get(url)
            ...     # 将响应内容转换为 RGB 格式的图像
            ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")
    
            # 图像的 URL
            >>> img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/tennis.jpg"
            # 下载图像并存储
            >>> image = download_image(img_url)
    
            # 使用管道进行图像反转操作，指定反转步骤和跳过比例
            >>> _ = pipe.invert(image=image, num_inversion_steps=50, skip=0.2)
    
            # 使用管道进行图像编辑，定义编辑提示和参数
            >>> edited_image = pipe(
            ...     editing_prompt=["tennis ball", "tomato"],
            ...     reverse_editing_direction=[True, False],
            ...     edit_guidance_scale=[5.0, 10.0],
            ...     edit_threshold=[0.9, 0.85],
            ... ).images[0]  # 获取编辑后的第一张图像
# 文档字符串，通常用于描述类或方法的功能
"""
# 从 diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LeditsAttentionStore 复制的类
class LeditsAttentionStore:
    # 静态方法，返回一个空的注意力存储结构
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    # 使对象可调用的方法，处理注意力矩阵
    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts, PnP=False):
        # attn.shape = batch_size * head_size, seq_len query, seq_len_key
        # 如果注意力矩阵的第二维小于等于最大大小
        if attn.shape[1] <= self.max_size:
            # 计算批次大小，考虑 PnP 和编辑提示
            bs = 1 + int(PnP) + editing_prompts
            skip = 2 if PnP else 1  # 跳过 PnP 和无条件
            # 将注意力矩阵分割并重新排列维度
            attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3)
            # 计算源批次大小
            source_batch_size = int(attn.shape[1] // bs)
            # 调用前向传播方法
            self.forward(attn[:, skip * source_batch_size :], is_cross, place_in_unet)

    # 前向传播方法，存储注意力
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 创建键，用于存储当前层的注意力
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # 将当前注意力添加到步骤存储中
        self.step_store[key].append(attn)

    # 在步骤之间调用的方法，决定是否存储步骤
    def between_steps(self, store_step=True):
        if store_step:
            # 如果需要平均处理
            if self.average:
                # 如果注意力存储为空，初始化
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    # 将步骤存储中的注意力与已有的注意力相加
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                # 如果不平均处理，初始化或追加步骤存储
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)

            # 当前步骤计数加一
            self.cur_step += 1
        # 重置步骤存储为一个空的注意力存储
        self.step_store = self.get_empty_store()

    # 获取特定步骤的注意力数据
    def get_attention(self, step: int):
        # 如果需要平均，计算平均注意力
        if self.average:
            attention = {
                key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
            }
        else:
            # 断言步骤不为空
            assert step is not None
            # 获取指定步骤的注意力
            attention = self.attention_store[step]
        return attention

    # 聚合注意力的方法，处理多个输入
    def aggregate_attention(
        self, attention_maps, prompts, res: Union[int, Tuple[int]], from_where: List[str], is_cross: bool, select: int
    ):
        # 初始化一个列表，包含 self.batch_size 个空列表，用于存储输出
        out = [[] for x in range(self.batch_size)]
        # 检查 res 是否为整数
        if isinstance(res, int):
            # 计算每个像素的数量，假设为 res 的平方
            num_pixels = res**2
            # 设置分辨率为 (res, res)
            resolution = (res, res)
        else:
            # 计算像素数量为 res 的两个维度的乘积
            num_pixels = res[0] * res[1]
            # 设置分辨率为 res 的前两个值
            resolution = res[:2]

        # 遍历 from_where 列表
        for location in from_where:
            # 遍历与当前位置相关的注意力图
            for bs_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                # 枚举当前批次和项目
                for batch, item in enumerate(bs_item):
                    # 检查当前项目的第二维是否等于 num_pixels
                    if item.shape[1] == num_pixels:
                        # 将项目重塑为指定形状，并选择相关的项目
                        cross_maps = item.reshape(len(prompts), -1, *resolution, item.shape[-1])[select]
                        # 将重塑的项目添加到输出的相应批次中
                        out[batch].append(cross_maps)

        # 将每个批次的输出合并成一个张量
        out = torch.stack([torch.cat(x, dim=0) for x in out])
        # 对每个头进行平均
        out = out.sum(1) / out.shape[1]
        # 返回最终的输出
        return out

    def __init__(self, average: bool, batch_size=1, max_resolution=16, max_size: int = None):
        # 初始化一个空的存储，用于保存步骤信息
        self.step_store = self.get_empty_store()
        # 初始化一个空的注意力存储列表
        self.attention_store = []
        # 当前步骤初始化为 0
        self.cur_step = 0
        # 设置平均标志
        self.average = average
        # 设置批次大小
        self.batch_size = batch_size
        # 如果 max_size 为空，计算最大尺寸为 max_resolution 的平方
        if max_size is None:
            self.max_size = max_resolution**2
        # 如果 max_size 不为空且 max_resolution 为空，设置最大尺寸为 max_size
        elif max_size is not None and max_resolution is None:
            self.max_size = max_size
        # 如果两个都被设置，则抛出错误
        else:
            raise ValueError("Only allowed to set one of max_resolution or max_size")
# 从 diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion 导入的 LeditsGaussianSmoothing 类
class LeditsGaussianSmoothing:
    # 初始化函数，接受设备参数
    def __init__(self, device):
        # 定义高斯核的大小
        kernel_size = [3, 3]
        # 定义高斯核的标准差
        sigma = [0.5, 0.5]

        # 高斯核是每个维度的高斯函数的乘积
        kernel = 1
        # 创建网格，用于生成高斯核
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        # 遍历每个维度的大小、标准差和网格
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            # 计算均值
            mean = (size - 1) / 2
            # 更新高斯核值
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # 确保高斯核的所有值之和等于 1
        kernel = kernel / torch.sum(kernel)

        # 将高斯核重塑为深度可分离卷积的权重
        kernel = kernel.view(1, 1, *kernel.size())
        # 重复高斯核以适应多通道输入
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        # 将权重转移到指定设备上
        self.weight = kernel.to(device)

    # 调用函数，用于应用高斯滤波
    def __call__(self, input):
        """
        参数:
        对输入应用高斯滤波。
            input (torch.Tensor): 要应用高斯滤波的输入。
        返回:
            filtered (torch.Tensor): 滤波后的输出。
        """
        # 使用卷积函数对输入应用高斯滤波
        return F.conv2d(input, weight=self.weight.to(input.dtype))


# 从 diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion 导入的 LEDITSCrossAttnProcessor 类
class LEDITSCrossAttnProcessor:
    # 初始化函数，接受注意力存储、在 UNet 中的位置、PnP 和编辑提示
    def __init__(self, attention_store, place_in_unet, pnp, editing_prompts):
        # 设置注意力存储
        self.attnstore = attention_store
        # 设置在 UNet 中的位置
        self.place_in_unet = place_in_unet
        # 设置编辑提示
        self.editing_prompts = editing_prompts
        # 设置 PnP 参数
        self.pnp = pnp

    # 调用函数，用于处理注意力和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        temb=None,
    ):
        # 获取批次大小、序列长度和特征维度，如果没有编码器隐藏状态，则使用隐藏状态的形状
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # 准备注意力掩码，调整为适合序列长度和批次大小
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)

        # 如果没有编码器隐藏状态，使用当前隐藏状态；如果存在且需要归一化，则归一化编码器隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 将编码器隐藏状态转换为键向量
        key = attn.to_k(encoder_hidden_states)
        # 将编码器隐藏状态转换为值向量
        value = attn.to_v(encoder_hidden_states)

        # 将查询向量调整为批次维度
        query = attn.head_to_batch_dim(query)
        # 将键向量调整为批次维度
        key = attn.head_to_batch_dim(key)
        # 将值向量调整为批次维度
        value = attn.head_to_batch_dim(value)

        # 计算注意力得分
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 存储注意力得分到内部存储中，标记为跨注意力
        self.attnstore(
            attention_probs,
            is_cross=True,
            place_in_unet=self.place_in_unet,
            editing_prompts=self.editing_prompts,
            PnP=self.pnp,
        )

        # 使用注意力得分和值向量进行批次矩阵乘法以获取新的隐藏状态
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态从批次维度转换回头部维度
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 通过线性投影转换隐藏状态
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout 操作
        hidden_states = attn.to_out[1](hidden_states)

        # 将隐藏状态按输出缩放因子进行归一化
        hidden_states = hidden_states / attn.rescale_output_factor
        # 返回最终的隐藏状态
        return hidden_states
# 定义一个名为 LEditsPPPipelineStableDiffusionXL 的类，继承多个混入类
class LEditsPPPipelineStableDiffusionXL(
    # 继承自 DiffusionPipeline，提供扩散模型功能
    DiffusionPipeline,
    # 继承自 FromSingleFileMixin，允许从单一文件加载数据
    FromSingleFileMixin,
    # 继承自 StableDiffusionXLLoraLoaderMixin，支持加载 LoRA 权重
    StableDiffusionXLLoraLoaderMixin,
    # 继承自 TextualInversionLoaderMixin，支持文本反演功能
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin，提供图像处理适配功能
    IPAdapterMixin,
):
    """
    使用 LEDits++ 和 Stable Diffusion XL 进行文本图像编辑的管道。

    此模型继承自 [`DiffusionPipeline`] 并基于 [`StableDiffusionXLPipeline`]。查看超类文档以获取
    所有管道实现的通用方法（下载、保存、在特定设备上运行等）。

    此外，管道还继承了以下加载方法：
        - *LoRA*: [`LEditsPPPipelineStableDiffusionXL.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    以及以下保存方法：
        - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]
    ```
    # 参数说明
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器模型，用于将图像编码为潜在表示并解码。
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，Stable Diffusion XL 使用 CLIP 的文本部分。
            # 具体使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
            # 第二个冻结文本编码器，Stable Diffusion XL 使用 CLIP 的文本和池化部分。
            # 具体使用 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 变体。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # CLIPTokenizer 类的分词器。
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            # 第二个 CLIPTokenizer 类的分词器。
        unet ([`UNet2DConditionModel`]): 
            # 条件 U-Net 架构，用于对编码的图像潜在表示进行去噪。
        scheduler ([`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]):
            # 与 `unet` 结合使用的调度器，用于去噪编码的图像潜在表示。
            # 可以是 [`DPMSolverMultistepScheduler`] 或 [`DDIMScheduler`]，如传入其他调度器，将默认设置为 [`DPMSolverMultistepScheduler`]。
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            # 是否强制将负提示嵌入始终设置为 0。另见 `stabilityai/stable-diffusion-xl-base-1-0` 的配置。
        add_watermarker (`bool`, *optional*):
            # 是否使用 [invisible_watermark 库](https://github.com/ShieldMnt/invisible-watermark/) 对输出图像进行水印处理。
            # 如果未定义，且包已安装，则默认设置为 True，否则不使用水印。
    """

    # 定义模型的 CPU 离线加载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    # 定义可选组件列表
    _optional_components = [
        # 分词器、第二分词器、文本编码器等可选组件
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    # 定义回调张量输入列表
    _callback_tensor_inputs = [
        # 潜在表示、提示嵌入等输入张量
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]
    # 初始化方法，用于设置类的基本属性
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器模型
            text_encoder: CLIPTextModel,  # 文本编码器模型
            text_encoder_2: CLIPTextModelWithProjection,  # 第二文本编码器模型，带投影
            tokenizer: CLIPTokenizer,  # 文本分词器
            tokenizer_2: CLIPTokenizer,  # 第二文本分词器
            unet: UNet2DConditionModel,  # UNet2D条件模型
            scheduler: Union[DPMSolverMultistepScheduler, DDIMScheduler],  # 调度器，可以是多步DPMSolver或DDIM调度器
            image_encoder: CLIPVisionModelWithProjection = None,  # 图像编码器，带投影，可选
            feature_extractor: CLIPImageProcessor = None,  # 特征提取器，可选
            force_zeros_for_empty_prompt: bool = True,  # 是否强制将空提示处理为零
            add_watermarker: Optional[bool] = None,  # 是否添加水印，可选
        ):
            # 调用父类的初始化方法
            super().__init__()
    
            # 注册多个模块，使其可供后续使用
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
            # 将配置参数注册到类中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 计算VAE的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 初始化图像处理器，使用计算的缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 检查调度器类型并初始化为DPMSolverMultistepScheduler（如果必要）
            if not isinstance(scheduler, DDIMScheduler) and not isinstance(scheduler, DPMSolverMultistepScheduler):
                self.scheduler = DPMSolverMultistepScheduler.from_config(
                    scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2
                )
                # 记录警告，说明调度器已更改
                logger.warning(
                    "This pipeline only supports DDIMScheduler and DPMSolverMultistepScheduler. "
                    "The scheduler has been changed to DPMSolverMultistepScheduler."
                )
    
            # 设置默认样本大小
            self.default_sample_size = self.unet.config.sample_size
    
            # 如果add_watermarker为None，则根据可用性确定是否添加水印
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 根据是否添加水印初始化水印对象
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None
            # 初始化反转步骤为None
            self.inversion_steps = None
    
        # 编码提示方法，用于处理输入提示
        def encode_prompt(
            self,
            device: Optional[torch.device] = None,  # 指定计算设备，可选
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            negative_prompt: Optional[str] = None,  # 负提示，可选
            negative_prompt_2: Optional[str] = None,  # 第二个负提示，可选
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负提示的嵌入表示，可选
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 负提示的池化嵌入表示，可选
            lora_scale: Optional[float] = None,  # LoRA缩放因子，可选
            clip_skip: Optional[int] = None,  # 跳过的CLIP层数量，可选
            enable_edit_guidance: bool = True,  # 是否启用编辑指导
            editing_prompt: Optional[str] = None,  # 编辑提示，可选
            editing_prompt_embeds: Optional[torch.Tensor] = None,  # 编辑提示的嵌入表示，可选
            editing_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 编辑提示的池化嵌入表示，可选
        # 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs复制的内容
    # 为调度器步骤准备额外的关键字参数，因不同调度器的签名不同
    def prepare_extra_step_kwargs(self, eta, generator=None):
        # eta (η) 仅用于 DDIMScheduler，其他调度器将忽略该参数
        # eta 对应于 DDIM 论文中的 η，范围应在 [0, 1] 之间
    
        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤关键字参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外步骤参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外步骤参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数
        return extra_step_kwargs
    
    # 检查输入参数的有效性
    def check_inputs(
        self,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        # 如果同时提供 negative_prompt 和 negative_prompt_embeds，抛出错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        # 如果同时提供 negative_prompt_2 和 negative_prompt_embeds，抛出错误
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
    
        # 如果提供 negative_prompt_embeds 但未提供 negative_pooled_prompt_embeds，抛出错误
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 修改而来
        def prepare_latents(self, device, latents):
            # 将潜在变量移动到指定设备
            latents = latents.to(device)
    
            # 根据调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 获取添加时间标识的辅助方法
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # 将原始大小、裁剪坐标的左上角和目标大小合并为一个列表
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        # 计算通过时间嵌入维度和文本编码器投影维度得到的总维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取模型期望的附加时间嵌入的输入特征维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查实际生成的嵌入维度是否与期望的维度匹配
        if expected_add_embed_dim != passed_add_embed_dim:
            # 如果不匹配，抛出错误并提供详细信息
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将时间 ID 转换为张量，并指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 返回生成的时间 ID 张量
        return add_time_ids

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae 复制的代码
    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 数据类型
        self.vae.to(dtype=torch.float32)
        # 检查 VAE 解码器中的注意力处理器是否使用 Torch 2.0 或 XFormers
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # 如果使用 XFormers 或 Torch 2.0，则注意力块不需要是 float32，可以节省大量内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层转换为相应的数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将解码器输入卷积层转换为相应的数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将解码器中间块转换为相应的数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding 复制的代码
    def get_guidance_scale_embedding(
        # 输入张量 w 和嵌入维度，默认嵌入维度为 512，数据类型默认为 float32
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    # 函数返回嵌入向量，类型为 torch.Tensor
        ) -> torch.Tensor:
            """
            # 文档字符串，提供函数的详细说明和参数描述
            See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            Args:
                w (`torch.Tensor`):
                    # 指定引导尺度生成嵌入向量，以丰富时间步嵌入
                    Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
                embedding_dim (`int`, *optional*, defaults to 512):
                    # 要生成的嵌入维度
                    Dimension of the embeddings to generate.
                dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                    # 生成的嵌入数据类型
                    Data type of the generated embeddings.
    
            Returns:
                `torch.Tensor`: # 返回嵌入向量，形状为 (len(w), embedding_dim)
                Embedding vectors with shape `(len(w), embedding_dim)`.
            """
            # 确保输入张量 w 是一维的
            assert len(w.shape) == 1
            # 将 w 的值放大 1000 倍
            w = w * 1000.0
    
            # 计算嵌入的半维度
            half_dim = embedding_dim // 2
            # 计算嵌入的基础值
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 计算负指数以获得衰减的嵌入
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 将 w 转换为目标数据类型并计算最终的嵌入
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦嵌入合并
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度是奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保最终嵌入的形状正确
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回最终的嵌入
            return emb
    
        # 属性获取引导尺度的值
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 属性获取引导重新缩放的值
        @property
        def guidance_rescale(self):
            return self._guidance_rescale
    
        # 属性获取剪辑跳过的值
        @property
        def clip_skip(self):
            return self._clip_skip
    
        # 此属性定义了与方程 (2) 中引导权重 w 类似的引导尺度
        # 引导尺度 = 1 表示没有进行分类器自由引导。
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        @property
        def do_classifier_free_guidance(self):
            # 判断是否进行分类器自由引导
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 属性获取交叉注意力的关键字参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 属性获取去噪结束的值
        @property
        def denoising_end(self):
            return self._denoising_end
    
        # 属性获取时间步数的值
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 从指定管道复制的内容
        # Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion.prepare_unet
    # 准备 UNet 模型，设置注意力处理器
    def prepare_unet(self, attention_store, PnP: bool = False):
        # 初始化一个空字典用于存储注意力处理器
        attn_procs = {}
        # 遍历 UNet 的注意力处理器的键
        for name in self.unet.attn_processors.keys():
            # 如果名字以 "mid_block" 开头，则设置位置为 "mid"
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            # 如果名字以 "up_blocks" 开头，则设置位置为 "up"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            # 如果名字以 "down_blocks" 开头，则设置位置为 "down"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            # 如果名字不符合以上条件，则跳过当前循环
            else:
                continue

            # 如果名字包含 "attn2" 且位置不是 "mid"
            if "attn2" in name and place_in_unet != "mid":
                # 创建 LEDITSCrossAttnProcessor 实例并加入字典
                attn_procs[name] = LEDITSCrossAttnProcessor(
                    attention_store=attention_store,  # 传入注意力存储
                    place_in_unet=place_in_unet,      # 传入在 UNet 中的位置
                    pnp=PnP,                          # 传入 PnP 标志
                    editing_prompts=self.enabled_editing_prompts,  # 传入启用的编辑提示
                )
            # 否则，创建默认的 AttnProcessor 实例
            else:
                attn_procs[name] = AttnProcessor()

        # 设置 UNet 的注意力处理器
        self.unet.set_attn_processor(attn_procs)

    # 在不计算梯度的情况下调用该函数
    @torch.no_grad()
    # 用示例文档字符串替换文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 可选的去噪结束时间
        denoising_end: Optional[float] = None,
        # 可选的负提示，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 第二个可选的负提示
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 可选的负提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负池化提示嵌入
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的输入适配器图像
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 输出类型的可选参数，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典的可选参数，默认为 True
        return_dict: bool = True,
        # 可选的交叉注意力参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 引导缩放因子的可选参数，默认为 0.0
        guidance_rescale: float = 0.0,
        # 左上角裁剪坐标的可选参数，默认为 (0, 0)
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 可选的目标尺寸
        target_size: Optional[Tuple[int, int]] = None,
        # 可选的编辑提示
        editing_prompt: Optional[Union[str, List[str]]] = None,
        # 可选的编辑提示嵌入
        editing_prompt_embeddings: Optional[torch.Tensor] = None,
        # 可选的编辑池化提示嵌入
        editing_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的反向编辑方向，默认为 False
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        # 编辑引导缩放因子的可选参数，默认为 5
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        # 编辑热身步骤的可选参数，默认为 0
        edit_warmup_steps: Optional[Union[int, List[int]]] = 0,
        # 编辑冷却步骤的可选参数
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        # 编辑阈值的可选参数，默认为 0.9
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        # 可选的语义引导张量列表
        sem_guidance: Optional[List[torch.Tensor]] = None,
        # 使用交叉注意力掩膜的可选参数，默认为 False
        use_cross_attn_mask: bool = False,
        # 使用交集掩膜的可选参数，默认为 False
        use_intersect_mask: bool = False,
        # 用户掩膜的可选参数
        user_mask: Optional[torch.Tensor] = None,
        # 注意力存储步骤的可选列表，默认为空列表
        attn_store_steps: Optional[List[int]] = [],
        # 是否在步骤间平均存储的可选参数，默认为 True
        store_averaged_over_steps: bool = True,
        # 可选的跳过剪辑的参数
        clip_skip: Optional[int] = None,
        # 可选的步骤结束回调函数
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 步骤结束时的张量输入回调参数，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 其他可选参数
        **kwargs,
    # 在不计算梯度的情况下调用该函数
    @torch.no_grad()
    # 从 diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion.encode_image 修改而来
    # 定义一个方法用于编码图像，接受多个可选参数
    def encode_image(self, image, dtype=None, height=None, width=None, resize_mode="default", crops_coords=None):
        # 预处理图像，调整大小和裁剪，返回处理后的图像
        image = self.image_processor.preprocess(
            image=image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        # 后处理图像，转换为 PIL 格式
        resized = self.image_processor.postprocess(image=image, output_type="pil")

        # 检查图像的最大尺寸是否超过配置的采样大小的 1.5 倍
        if max(image.shape[-2:]) > self.vae.config["sample_size"] * 1.5:
            # 记录警告信息，提示用户输入图像分辨率过高
            logger.warning(
                "Your input images far exceed the default resolution of the underlying diffusion model. "
                "The output images may contain severe artifacts! "
                "Consider down-sampling the input using the `height` and `width` parameters"
            )
        # 将图像转换为指定的设备和数据类型
        image = image.to(self.device, dtype=dtype)
        # 检查是否需要将数据类型上调
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        # 如果需要上调数据类型
        if needs_upcasting:
            # 将图像转换为浮点数
            image = image.float()
            # 上调 VAE 模型的数据类型
            self.upcast_vae()

        # 使用 VAE 编码图像，获取潜在分布的模式
        x0 = self.vae.encode(image).latent_dist.mode()
        # 将模式转换为指定的数据类型
        x0 = x0.to(dtype)
        # 如果需要，转换回 fp16 数据类型
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 根据配置的缩放因子调整潜在向量
        x0 = self.vae.config.scaling_factor * x0
        # 返回潜在向量和调整后的图像
        return x0, resized

    # 装饰器，表示该方法在执行时不计算梯度
    @torch.no_grad()
    # 定义一个方法用于反转图像，接受多个可选参数
    def invert(
        self,
        image: PipelineImageInput,
        source_prompt: str = "",
        source_guidance_scale=3.5,
        negative_prompt: str = None,
        negative_prompt_2: str = None,
        num_inversion_steps: int = 50,
        skip: float = 0.15,
        generator: Optional[torch.Generator] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        num_zero_noise_steps: int = 3,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
# 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.rescale_noise_cfg 复制
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 重新缩放 `noise_cfg`。基于 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的研究结果。见第 3.4 节
    """
    # 计算 noise_pred_text 的标准差，沿着指定维度，保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算 noise_cfg 的标准差，沿着指定维度，保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 通过标准差的比值重新缩放指导结果（修正过度曝光）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 用指导结果与原始结果混合，通过 guidance_rescale 因子避免“普通”图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg


# 从 diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.compute_noise_ddim 复制
def compute_noise_ddim(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. 获取前一步值（t-1）
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. 计算 alphas 和 betas
    # 当前时间步的累积 alpha 值
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    # 前一步的累积 alpha 值，若为负则使用最终的 alpha 值
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )
    # 当前时间步的 beta 值
    beta_prod_t = 1 - alpha_prod_t

    # 3. 从预测噪声计算预测的原始样本，称为 "predicted x_0"
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. 限制 "predicted x_0" 的范围
    if scheduler.config.clip_sample:
        # 将预测样本限制在 -1 和 1 之间
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. 计算方差：σ_t(η)，见公式 (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    # 计算标准差
    std_dev_t = eta * variance ** (0.5)

    # 6. 计算指向 x_t 的方向，见公式 (12)
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

    # 修改以返回更新后的 xtm1，以避免误差累积
    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 如果方差大于 0，计算噪声
    if variance > 0.0:
        noise = (prev_latents - mu_xt) / (variance ** (0.5) * eta)
    else:
        # 否则噪声设置为零
        noise = torch.tensor([0.0]).to(latents.device)

    # 返回计算出的噪声和更新后的 mu_xt
    return noise, mu_xt + (eta * variance**0.5) * noise


# 从 diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.compute_noise_sde_dpm_pp_2nd 复制
def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 定义一阶更新函数，输入模型输出和样本
        def first_order_update(model_output, sample):  # timestep, prev_timestep, sample):
            # 获取当前和前一个步骤的 sigma 值
            sigma_t, sigma_s = scheduler.sigmas[scheduler.step_index + 1], scheduler.sigmas[scheduler.step_index]
            # 将 sigma 转换为 alpha 和 sigma_t
            alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s, sigma_s = scheduler._sigma_to_alpha_sigma_t(sigma_s)
            # 计算 lambda_t 和 lambda_s
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    
            # 计算 h 值
            h = lambda_t - lambda_s
    
            # 计算 mu_xt，结合样本和模型输出
            mu_xt = (sigma_t / sigma_s * torch.exp(-h)) * sample + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
    
            # 使用调度器的 DPM 解决方案更新 mu_xt
            mu_xt = scheduler.dpm_solver_first_order_update(
                model_output=model_output, sample=sample, noise=torch.zeros_like(sample)
            )
    
            # 计算 sigma
            sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
            # 如果 sigma 大于 0，计算噪声
            if sigma > 0.0:
                noise = (prev_latents - mu_xt) / sigma
            else:
                # 否则设定噪声为 0
                noise = torch.tensor([0.0]).to(sample.device)
    
            # 计算前一个样本
            prev_sample = mu_xt + sigma * noise
            # 返回噪声和前一个样本
            return noise, prev_sample
    
        # 定义二阶更新函数，输入模型输出列表和样本
        def second_order_update(model_output_list, sample):  # timestep_list, prev_timestep, sample):
            # 获取当前和前两个步骤的 sigma 值
            sigma_t, sigma_s0, sigma_s1 = (
                scheduler.sigmas[scheduler.step_index + 1],
                scheduler.sigmas[scheduler.step_index],
                scheduler.sigmas[scheduler.step_index - 1],
            )
    
            # 将 sigma 转换为 alpha 和 sigma_t
            alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s0, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigma_s0)
            alpha_s1, sigma_s1 = scheduler._sigma_to_alpha_sigma_t(sigma_s1)
    
            # 计算 lambda 值
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
            lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    
            # 获取最后两个模型输出
            m0, m1 = model_output_list[-1], model_output_list[-2]
    
            # 计算 h 和 h_0
            h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
            r0 = h_0 / h
            # 设定 D0 和 D1
            D0, D1 = m0, (1.0 / r0) * (m0 - m1)
    
            # 计算 mu_xt
            mu_xt = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
            )
    
            # 计算 sigma
            sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
            # 如果 sigma 大于 0，计算噪声
            if sigma > 0.0:
                noise = (prev_latents - mu_xt) / sigma
            else:
                # 否则设定噪声为 0
                noise = torch.tensor([0.0]).to(sample.device)
    
            # 计算前一个样本
            prev_sample = mu_xt + sigma * noise
    
            # 返回噪声和前一个样本
            return noise, prev_sample
    
        # 如果调度器的步骤索引为 None，初始化步骤索引
        if scheduler.step_index is None:
            scheduler._init_step_index(timestep)
    
        # 将模型输出转换为可用格式
        model_output = scheduler.convert_model_output(model_output=noise_pred, sample=latents)
        # 更新模型输出列表
        for i in range(scheduler.config.solver_order - 1):
            scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
        scheduler.model_outputs[-1] = model_output
    
        # 根据低阶数量决定使用一阶或二阶更新
        if scheduler.lower_order_nums < 1:
            noise, prev_sample = first_order_update(model_output, latents)
        else:
            noise, prev_sample = second_order_update(scheduler.model_outputs, latents)
    # 如果当前调度器的低阶数量小于配置中的求解器阶数
    if scheduler.lower_order_nums < scheduler.config.solver_order:
        # 增加低阶数量的计数
        scheduler.lower_order_nums += 1

    # 完成后，将步骤索引增加一
    scheduler._step_index += 1

    # 返回噪声和之前的样本
    return noise, prev_sample
# 从 diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion 复制的代码
def compute_noise(scheduler, *args):
    # 检查调度器是否为 DDIMScheduler 实例
    if isinstance(scheduler, DDIMScheduler):
        # 调用 DDIM 调度器的噪声计算函数并返回结果
        return compute_noise_ddim(scheduler, *args)
    # 检查调度器是否为 DPMSolverMultistepScheduler 实例，且满足特定配置
    elif (
        isinstance(scheduler, DPMSolverMultistepScheduler)
        and scheduler.config.algorithm_type == "sde-dpmsolver++"
        and scheduler.config.solver_order == 2
    ):
        # 调用 SDE DPM 的二阶噪声计算函数并返回结果
        return compute_noise_sde_dpm_pp_2nd(scheduler, *args)
    else:
        # 如果不满足以上条件，抛出未实现的错误
        raise NotImplementedError
```