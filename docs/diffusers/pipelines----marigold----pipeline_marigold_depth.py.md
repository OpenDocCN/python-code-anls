# `.\diffusers\pipelines\marigold\pipeline_marigold_depth.py`

```py
# 版权声明，说明该代码的版权归属
# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# 版权声明，说明该代码的版权归属
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证对该文件的使用进行说明
# Licensed under the Apache License, Version 2.0 (the "License");
# 说明除非遵循许可证，否则不可使用该文件
# you may not use this file except in compliance with the License.
# 提供获取许可证的链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 提供关于该软件的使用条款
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何类型的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 提供许可证的详细信息
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# 提供更多信息和引用说明的来源
# More information and citation instructions are available on the
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
# 从 dataclass 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 functools 模块导入 partial 函数
from functools import partial
# 导入用于类型提示的类型
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch
# 从 PIL 导入 Image 类
from PIL import Image
# 从 tqdm 导入进度条显示工具
from tqdm.auto import tqdm
# 从 transformers 导入 CLIP 模型和标记器
from transformers import CLIPTextModel, CLIPTokenizer

# 从图像处理模块导入 PipelineImageInput 类
from ...image_processor import PipelineImageInput
# 从模型模块导入相关模型
from ...models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
# 从调度器模块导入调度器类
from ...schedulers import (
    DDIMScheduler,
    LCMScheduler,
)
# 从工具模块导入基本输出和日志功能
from ...utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
# 导入 SciPy 可用性检查函数
from ...utils.import_utils import is_scipy_available
# 导入生成随机张量的工具
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入 DiffusionPipeline 类
from ..pipeline_utils import DiffusionPipeline
# 从图像处理模块导入 MarigoldImageProcessor 类
from .marigold_image_processing import MarigoldImageProcessor

# 创建一个日志记录器实例
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该管道
EXAMPLE_DOC_STRING = """
Examples:

>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
...     "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> depth = pipe(image)

>>> vis = pipe.image_processor.visualize_depth(depth.prediction)
>>> vis[0].save("einstein_depth.png")

>>> depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
>>> depth_16bit[0].save("einstein_depth_16bit.png")

"""

# 定义 MarigoldDepthOutput 类，表示单目深度预测的输出
@dataclass
class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.
    # 函数参数文档字符串，说明参数的类型和形状
        Args:
            prediction (`np.ndarray`, `torch.Tensor`):  # 预测的深度图，值范围在 [0, 1] 之间
                # 形状为 numimages × 1 × height × width，无论图像是作为 4D 数组还是列表传入
            uncertainty (`None`, `np.ndarray`, `torch.Tensor`):  # 从集成计算的置信度图，值范围在 [0, 1] 之间
                # 形状为 numimages × 1 × height × width
            latent (`None`, `torch.Tensor`):  # 与预测对应的潜在特征，兼容于管道的 latents 参数
                # 形状为 numimages * numensemble × 4 × latentheight × latentwidth
        """  # 结束文档字符串
    
        prediction: Union[np.ndarray, torch.Tensor]  # 声明 prediction 为 np.ndarray 或 torch.Tensor 类型
        uncertainty: Union[None, np.ndarray, torch.Tensor]  # 声明 uncertainty 为 None、np.ndarray 或 torch.Tensor 类型
        latent: Union[None, torch.Tensor]  # 声明 latent 为 None 或 torch.Tensor 类型
# 定义一个名为 MarigoldDepthPipeline 的类，继承自 DiffusionPipeline
class MarigoldDepthPipeline(DiffusionPipeline):
    """
    使用 Marigold 方法进行单目深度估计的管道： https://marigoldmonodepth.github.io。

    此模型继承自 [`DiffusionPipeline`]。请查阅父类文档，以了解库为所有管道实现的通用方法
    （例如下载或保存，在特定设备上运行等）。

    参数：
        unet (`UNet2DConditionModel`):
            条件 U-Net，用于在图像潜在空间的条件下去噪深度潜在。
        vae (`AutoencoderKL`):
            变分自编码器（VAE）模型，用于将图像和预测编码和解码为潜在表示。
        scheduler (`DDIMScheduler` 或 `LCMScheduler`):
            用于与 `unet` 结合使用的调度器，以去噪编码的图像潜在。
        text_encoder (`CLIPTextModel`):
            文本编码器，用于空文本嵌入。
        tokenizer (`CLIPTokenizer`):
            CLIP 分词器。
        prediction_type (`str`, *可选*):
            模型所做预测的类型。
        scale_invariant (`bool`, *可选*):
            指定预测深度图是否具有尺度不变性的模型属性。此值必须在模型配置中设置。
            与 `shift_invariant=True` 标志一起使用时，模型也被称为“仿射不变”。注意：不支持覆盖此值。
        shift_invariant (`bool`, *可选*):
            指定预测深度图是否具有平移不变性的模型属性。此值必须在模型配置中设置。
            与 `scale_invariant=True` 标志一起使用时，模型也被称为“仿射不变”。注意：不支持覆盖此值。
        default_denoising_steps (`int`, *可选*):
            生成合理质量预测所需的最小去噪扩散步骤数。此值必须在模型配置中设置。
            当调用管道而未显式设置 `num_inference_steps` 时，使用默认值。这是为了确保与各种模型
            变体兼容的合理结果，例如依赖于非常短的去噪调度的模型（`LCMScheduler`）和具有完整扩散
            调度的模型（`DDIMScheduler`）。
        default_processing_resolution (`int`, *可选*):
            管道 `processing_resolution` 参数的推荐值。此值必须在模型配置中设置。
            当调用管道而未显式设置 `processing_resolution` 时，使用默认值。这是为了确保与训练
            使用不同最佳处理分辨率值的各种模型变体兼容的合理结果。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义支持的预测类型，包括深度和视差
    supported_prediction_types = ("depth", "disparity")

    # 初始化方法，设置模型的基本参数
    def __init__(
        # UNet2DConditionModel 对象，进行图像生成
        self,
        unet: UNet2DConditionModel,
        # 自动编码器，用于处理图像
        vae: AutoencoderKL,
        # 调度器，控制生成过程中的步伐
        scheduler: Union[DDIMScheduler, LCMScheduler],
        # 文本编码器，处理输入文本信息
        text_encoder: CLIPTextModel,
        # 分词器，用于将文本转换为模型可处理的格式
        tokenizer: CLIPTokenizer,
        # 可选的预测类型，默认为 None
        prediction_type: Optional[str] = None,
        # 可选，指示是否使用尺度不变性，默认为 True
        scale_invariant: Optional[bool] = True,
        # 可选，指示是否使用平移不变性，默认为 True
        shift_invariant: Optional[bool] = True,
        # 可选，默认去噪步骤数，默认为 None
        default_denoising_steps: Optional[int] = None,
        # 可选，默认处理分辨率，默认为 None
        default_processing_resolution: Optional[int] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查给定的预测类型是否在支持的范围内
        if prediction_type not in self.supported_prediction_types:
            # 记录警告，提示可能使用了不支持的预测类型
            logger.warning(
                f"Potentially unsupported `prediction_type='{prediction_type}'`; values supported by the pipeline: "
                f"{self.supported_prediction_types}."
            )

        # 注册模型组件，包括 UNet、VAE、调度器、文本编码器和分词器
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        # 将配置参数注册到当前实例
        self.register_to_config(
            prediction_type=prediction_type,
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        # 计算 VAE 的缩放因子，基于其配置
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 设置尺度不变性
        self.scale_invariant = scale_invariant
        # 设置平移不变性
        self.shift_invariant = shift_invariant
        # 设置默认去噪步骤数
        self.default_denoising_steps = default_denoising_steps
        # 设置默认处理分辨率
        self.default_processing_resolution = default_processing_resolution

        # 初始化空文本嵌入，初始值为 None
        self.empty_text_embedding = None

        # 创建图像处理器，基于 VAE 的缩放因子
        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # 检查输入的有效性的方法
    def check_inputs(
        # 输入的图像，类型为管道图像输入
        self,
        image: PipelineImageInput,
        # 推理步骤的数量
        num_inference_steps: int,
        # 集成的大小
        ensemble_size: int,
        # 处理分辨率
        processing_resolution: int,
        # 输入的重采样方法
        resample_method_input: str,
        # 输出的重采样方法
        resample_method_output: str,
        # 批量大小
        batch_size: int,
        # 可选的集成参数
        ensembling_kwargs: Optional[Dict[str, Any]],
        # 可选的潜在变量
        latents: Optional[torch.Tensor],
        # 可选的随机数生成器，支持单个或列表形式
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        # 输出类型的字符串
        output_type: str,
        # 是否输出不确定性，布尔值
        output_uncertainty: bool,
    # 定义一个进度条的方法，接受可选参数用于控制进度条的显示
        def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
            # 检查实例是否已有进度条配置属性
            if not hasattr(self, "_progress_bar_config"):
                # 如果没有，初始化一个空字典作为配置
                self._progress_bar_config = {}
            # 如果已有配置，检查其是否为字典类型
            elif not isinstance(self._progress_bar_config, dict):
                # 如果不是，抛出类型错误
                raise ValueError(
                    f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
                )
    
            # 创建一个进度条配置字典，复制已有配置
            progress_bar_config = dict(**self._progress_bar_config)
            # 从配置中获取描述，如果没有则使用传入的描述
            progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
            # 从配置中获取是否保留进度条，默认值为传入的参数
            progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
            # 如果提供了可迭代对象，返回带进度条的迭代器
            if iterable is not None:
                return tqdm(iterable, **progress_bar_config)
            # 如果提供了总数，返回总数进度条
            elif total is not None:
                return tqdm(total=total, **progress_bar_config)
            # 如果两个参数都未提供，抛出错误
            else:
                raise ValueError("Either `total` or `iterable` has to be defined.")
    
        # 禁用梯度计算以节省内存
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义调用方法，接受多个参数以处理图像
        def __call__(
            self,
            image: PipelineImageInput,  # 输入图像
            num_inference_steps: Optional[int] = None,  # 推理步骤的数量
            ensemble_size: int = 1,  # 集成模型的数量
            processing_resolution: Optional[int] = None,  # 处理图像的分辨率
            match_input_resolution: bool = True,  # 是否匹配输入分辨率
            resample_method_input: str = "bilinear",  # 输入图像的重采样方法
            resample_method_output: str = "bilinear",  # 输出图像的重采样方法
            batch_size: int = 1,  # 批处理的大小
            ensembling_kwargs: Optional[Dict[str, Any]] = None,  # 集成模型的额外参数
            latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,  # 潜在变量
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            output_type: str = "np",  # 输出类型，默认是 NumPy
            output_uncertainty: bool = False,  # 是否输出不确定性
            output_latent: bool = False,  # 是否输出潜在变量
            return_dict: bool = True,  # 是否以字典形式返回结果
        # 定义准备潜在变量的方法
        def prepare_latents(
            self,
            image: torch.Tensor,  # 输入图像的张量表示
            latents: Optional[torch.Tensor],  # 潜在变量的张量
            generator: Optional[torch.Generator],  # 随机数生成器
            ensemble_size: int,  # 集成模型的数量
            batch_size: int,  # 批处理的大小
    # 返回两个张量的元组
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 定义一个函数来提取潜在向量
        def retrieve_latents(encoder_output):
            # 检查 encoder_output 是否具有 latent_dist 属性
            if hasattr(encoder_output, "latent_dist"):
                # 返回潜在分布的众数
                return encoder_output.latent_dist.mode()
            # 检查 encoder_output 是否具有 latents 属性
            elif hasattr(encoder_output, "latents"):
                # 返回潜在向量
                return encoder_output.latents
            # 如果没有找到潜在向量，则抛出异常
            else:
                raise AttributeError("Could not access latents of provided encoder_output")
    
        # 将编码后的图像潜在向量按批次拼接在一起
        image_latent = torch.cat(
            [
                # 对每个批次的图像调用 retrieve_latents 函数
                retrieve_latents(self.vae.encode(image[i : i + batch_size]))
                for i in range(0, image.shape[0], batch_size)
            ],
            dim=0,
        )  # 结果形状为 [N,4,h,w]
        # 将图像潜在向量乘以缩放因子
        image_latent = image_latent * self.vae.config.scaling_factor
        # 在第0维重复潜在向量以适应集成大小
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # 结果形状为 [N*E,4,h,w]
    
        # 将潜在预测初始化为 latents
        pred_latent = latents
        # 如果预测潜在向量为空，生成随机张量
        if pred_latent is None:
            pred_latent = randn_tensor(
                # 生成与图像潜在向量相同形状的随机张量
                image_latent.shape,
                generator=generator,
                device=image_latent.device,
                dtype=image_latent.dtype,
            )  # 结果形状为 [N*E,4,h,w]
    
        # 返回图像潜在向量和预测潜在向量
        return image_latent, pred_latent
    
    # 解码潜在预测，返回张量
    def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:
        # 检查预测潜在向量的维度和形状是否符合预期
        if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:
            # 抛出值错误，如果形状不匹配
            raise ValueError(
                f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
            )
    
        # 解码预测潜在向量，返回字典中的第一个元素
        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # 结果形状为 [B,3,H,W]
    
        # 计算预测的均值，保持维度
        prediction = prediction.mean(dim=1, keepdim=True)  # 结果形状为 [B,1,H,W]
        # 将预测限制在 [-1.0, 1.0] 的范围内
        prediction = torch.clip(prediction, -1.0, 1.0)  # 结果形状为 [B,1,H,W]
        # 将预测从 [-1, 1] 转换到 [0, 1]
        prediction = (prediction + 1.0) / 2.0
    
        # 返回最终预测结果
        return prediction  # 结果形状为 [B,1,H,W]
    
    # 定义一个静态方法，用于处理深度信息
    @staticmethod
    def ensemble_depth(
        # 输入深度张量
        depth: torch.Tensor,
        # 是否使用尺度不变性
        scale_invariant: bool = True,
        # 是否使用位移不变性
        shift_invariant: bool = True,
        # 是否输出不确定性
        output_uncertainty: bool = False,
        # 指定聚合方式，默认为中位数
        reduction: str = "median",
        # 正则化强度
        regularizer_strength: float = 0.02,
        # 最大迭代次数
        max_iter: int = 2,
        # 收敛容忍度
        tol: float = 1e-3,
        # 最大分辨率
        max_res: int = 1024,
```