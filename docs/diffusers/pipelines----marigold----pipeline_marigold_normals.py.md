# `.\diffusers\pipelines\marigold\pipeline_marigold_normals.py`

```py
# 版权所有声明，说明作者和版权信息
# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证进行授权
# Licensed under the Apache License, Version 2.0 (the "License");
# 您只能在遵守许可证的情况下使用此文件
# you may not use this file except in compliance with the License.
# 您可以在此处获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按“原样”分发，没有任何明示或暗示的担保或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# 额外信息和引用说明可在 Marigold 项目网站上找到
# More information and citation instructions are available on the
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
# 导入数据类装饰器
from dataclasses import dataclass
# 导入类型提示相关的类型
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入图像处理库 PIL
from PIL import Image
# 导入进度条库
from tqdm.auto import tqdm
# 导入 CLIP 模型和分词器
from transformers import CLIPTextModel, CLIPTokenizer

# 导入图像处理的管道输入
from ...image_processor import PipelineImageInput
# 导入自动编码器和 UNet 模型
from ...models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
# 导入调度器
from ...schedulers import (
    DDIMScheduler,
    LCMScheduler,
)
# 导入工具函数
from ...utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
# 导入随机张量生成工具
from ...utils.torch_utils import randn_tensor
# 导入扩散管道工具
from ..pipeline_utils import DiffusionPipeline
# 导入 Marigold 图像处理工具
from .marigold_image_processing import MarigoldImageProcessor

# 创建日志记录器，用于记录日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，提供用法示例
EXAMPLE_DOC_STRING = """
Examples:

>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
...     "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> normals = pipe(image)

>>> vis = pipe.image_processor.visualize_normals(normals.prediction)
>>> vis[0].save("einstein_normals.png")

"""

# 定义 Marigold 单目法线预测管道的输出类
@dataclass
class MarigoldNormalsOutput(BaseOutput):
    """
    Marigold 单目法线预测管道的输出类
    # 定义函数参数的文档字符串
    Args:
        # 预测法线的参数，类型可以是 numpy 数组或 PyTorch 张量
        prediction (`np.ndarray`, `torch.Tensor`):
            # 预测的法线值范围在 [-1, 1] 之间，形状为 $numimages \times 3 \times height
            \times width$，无论图像是作为 4D 数组还是列表传递。
        # 不确定性地图的参数，类型可以是 None、numpy 数组或 PyTorch 张量
        uncertainty (`None`, `np.ndarray`, `torch.Tensor`):
            # 从集合中计算得到的不确定性地图，值范围在 [0, 1] 之间，形状为 $numimages
            \times 1 \times height \times width$。
        # 潜在特征的参数，类型可以是 None 或 PyTorch 张量
        latent (`None`, `torch.Tensor`):
            # 与预测相对应的潜在特征，兼容于管道的 `latents` 参数。
            # 形状为 $numimages * numensemble \times 4 \times latentheight \times latentwidth$。
    """

    # 声明预测参数的类型，支持 numpy 数组或 PyTorch 张量
    prediction: Union[np.ndarray, torch.Tensor]
    # 声明不确定性参数的类型，支持 None、numpy 数组或 PyTorch 张量
    uncertainty: Union[None, np.ndarray, torch.Tensor]
    # 声明潜在特征参数的类型，支持 None 或 PyTorch 张量
    latent: Union[None, torch.Tensor]
# 定义一个名为 MarigoldNormalsPipeline 的类，继承自 DiffusionPipeline
class MarigoldNormalsPipeline(DiffusionPipeline):
    """
    使用 Marigold 方法进行单目法线估计的管道： https://marigoldmonodepth.github.io.

    此模型继承自 [`DiffusionPipeline`]。请查看父类文档，以了解库为所有管道实现的通用方法
    （例如下载或保存、在特定设备上运行等）。

    参数：
        unet (`UNet2DConditionModel`):
            条件 U-Net，用于在图像潜在空间条件下对法线潜在进行去噪。
        vae (`AutoencoderKL`):
            变分自编码器 (VAE) 模型，用于将图像和预测编码和解码为潜在表示。
        scheduler (`DDIMScheduler` 或 `LCMScheduler`):
            与 `unet` 结合使用的调度器，用于对编码的图像潜在进行去噪。
        text_encoder (`CLIPTextModel`):
            文本编码器，用于生成空的文本嵌入。
        tokenizer (`CLIPTokenizer`):
            CLIP 令牌化器。
        prediction_type (`str`, *可选*):
            模型生成的预测类型。
        use_full_z_range (`bool`, *可选*):
            该模型预测的法线是否使用 Z 维度的完整范围，还是仅使用其正半。
        default_denoising_steps (`int`, *可选*):
            生成合理质量预测所需的最小去噪扩散步骤数。此值必须在模型配置中设置。当
            管道被调用而未明确设置 `num_inference_steps` 时，使用默认值。这样可以确保
            与与管道兼容的各种模型版本产生合理结果，例如依赖非常短去噪调度的模型
            (`LCMScheduler`) 和具有完整扩散调度的模型 (`DDIMScheduler`)。
        default_processing_resolution (`int`, *可选*):
            管道的 `processing_resolution` 参数的推荐值。此值必须在模型配置中设置。当
            管道被调用而未明确设置 `processing_resolution` 时，使用默认值。这样可以确保
            与不同最佳处理分辨率值训练的各种模型版本产生合理结果。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 支持的预测类型，当前仅支持 "normals"
    supported_prediction_types = ("normals",)

    # 初始化方法，接受多个参数以配置模型
    def __init__(
        # 条件 U-Net 模型
        self,
        unet: UNet2DConditionModel,
        # 变分自编码器模型
        vae: AutoencoderKL,
        # 调度器，可以是 DDIMScheduler 或 LCMScheduler
        scheduler: Union[DDIMScheduler, LCMScheduler],
        # 文本编码器
        text_encoder: CLIPTextModel,
        # CLIP 令牌化器
        tokenizer: CLIPTokenizer,
        # 可选的预测类型
        prediction_type: Optional[str] = None,
        # 是否使用 Z 维度的完整范围
        use_full_z_range: Optional[bool] = True,
        # 默认去噪步骤数
        default_denoising_steps: Optional[int] = None,
        # 默认处理分辨率
        default_processing_resolution: Optional[int] = None,
    ):
        # 调用父类的构造函数
        super().__init__()

        # 检查预测类型是否在支持的类型中
        if prediction_type not in self.supported_prediction_types:
            # 如果不支持，记录警告信息
            logger.warning(
                f"Potentially unsupported `prediction_type='{prediction_type}'`; values supported by the pipeline: "
                f"{self.supported_prediction_types}."
            )

        # 注册各个模块到当前实例中
        self.register_modules(
            unet=unet,  # 注册 UNet 模块
            vae=vae,    # 注册变分自编码器模块
            scheduler=scheduler,  # 注册调度器模块
            text_encoder=text_encoder,  # 注册文本编码器模块
            tokenizer=tokenizer,  # 注册分词器模块
        )
        # 将配置参数注册到当前实例中
        self.register_to_config(
            use_full_z_range=use_full_z_range,  # 注册使用完整的 z 范围
            default_denoising_steps=default_denoising_steps,  # 注册默认去噪步骤
            default_processing_resolution=default_processing_resolution,  # 注册默认处理分辨率
        )

        # 计算 VAE 的缩放因子，基于块输出通道的数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 将使用完整 z 范围的设置保存到实例中
        self.use_full_z_range = use_full_z_range
        # 将默认去噪步骤保存到实例中
        self.default_denoising_steps = default_denoising_steps
        # 将默认处理分辨率保存到实例中
        self.default_processing_resolution = default_processing_resolution

        # 初始化空的文本嵌入
        self.empty_text_embedding = None

        # 创建图像处理器实例，并传入 VAE 缩放因子
        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def check_inputs(
        # 定义检查输入参数的方法，接收多个参数
        image: PipelineImageInput,  # 输入图像
        num_inference_steps: int,  # 推理步骤数
        ensemble_size: int,  # 集成大小
        processing_resolution: int,  # 处理分辨率
        resample_method_input: str,  # 输入重采样方法
        resample_method_output: str,  # 输出重采样方法
        batch_size: int,  # 批处理大小
        ensembling_kwargs: Optional[Dict[str, Any]],  # 集成相关的关键字参数
        latents: Optional[torch.Tensor],  # 潜在变量
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],  # 随机数生成器
        output_type: str,  # 输出类型
        output_uncertainty: bool,  # 是否输出不确定性
    def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
        # 检查是否已经初始化了进度条配置
        if not hasattr(self, "_progress_bar_config"):
            # 如果没有，初始化为空字典
            self._progress_bar_config = {}
        # 如果存在配置，但不是字典类型，抛出错误
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        # 复制当前的进度条配置
        progress_bar_config = dict(**self._progress_bar_config)
        # 设置描述，如果未提供则使用现有值
        progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
        # 设置是否在完成后保留进度条
        progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
        # 如果提供了可迭代对象，返回带进度条的可迭代对象
        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        # 如果提供了总数，返回带进度条的总数
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        # 如果两者都未提供，抛出错误
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    # 使用 torch.no_grad() 装饰器，禁用梯度计算
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的类方法，处理图像并生成推理结果
        def __call__(
            self,
            image: PipelineImageInput,  # 输入的图像数据
            num_inference_steps: Optional[int] = None,  # 推理步骤数，默认为 None
            ensemble_size: int = 1,  # 集成模型的大小，默认为 1
            processing_resolution: Optional[int] = None,  # 处理分辨率，默认为 None
            match_input_resolution: bool = True,  # 是否匹配输入分辨率
            resample_method_input: str = "bilinear",  # 输入重采样方法，默认为双线性
            resample_method_output: str = "bilinear",  # 输出重采样方法，默认为双线性
            batch_size: int = 1,  # 批处理大小，默认为 1
            ensembling_kwargs: Optional[Dict[str, Any]] = None,  # 集成参数，默认为 None
            latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,  # 潜在变量，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器，默认为 None
            output_type: str = "np",  # 输出类型，默认为 NumPy
            output_uncertainty: bool = False,  # 是否输出不确定性，默认为 False
            output_latent: bool = False,  # 是否输出潜在变量，默认为 False
            return_dict: bool = True,  # 是否返回字典格式，默认为 True
        # 从 diffusers.pipelines.marigold.pipeline_marigold_depth.MarigoldDepthPipeline.prepare_latents 复制
        def prepare_latents(
            self,
            image: torch.Tensor,  # 输入图像的张量格式
            latents: Optional[torch.Tensor],  # 潜在变量的张量格式
            generator: Optional[torch.Generator],  # 随机数生成器，默认为 None
            ensemble_size: int,  # 集成大小
            batch_size: int,  # 批处理大小
        ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回两个张量的元组
            def retrieve_latents(encoder_output):  # 定义内部函数，用于获取潜在变量
                if hasattr(encoder_output, "latent_dist"):  # 检查输出是否包含潜在分布
                    return encoder_output.latent_dist.mode()  # 返回潜在分布的众数
                elif hasattr(encoder_output, "latents"):  # 检查输出是否包含潜在变量
                    return encoder_output.latents  # 返回潜在变量
                else:  # 如果都没有，抛出异常
                    raise AttributeError("Could not access latents of provided encoder_output")
    
            image_latent = torch.cat(  # 将处理后的潜在变量进行拼接
                [
                    retrieve_latents(self.vae.encode(image[i : i + batch_size]))  # 对每个批次的图像编码并获取潜在变量
                    for i in range(0, image.shape[0], batch_size)  # 按批处理大小遍历图像
                ],
                dim=0,  # 在第0维进行拼接
            )  # [N,4,h,w]  # 得到的潜在变量张量的形状
            image_latent = image_latent * self.vae.config.scaling_factor  # 应用缩放因子调整潜在变量
            image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]  # 重复以匹配集成大小
    
            pred_latent = latents  # 初始化预测潜在变量
            if pred_latent is None:  # 如果未提供潜在变量
                pred_latent = randn_tensor(  # 生成随机潜在变量
                    image_latent.shape,  # 形状与 image_latent 相同
                    generator=generator,  # 使用提供的随机数生成器
                    device=image_latent.device,  # 使用 image_latent 的设备
                    dtype=image_latent.dtype,  # 使用 image_latent 的数据类型
                )  # [N*E,4,h,w]  # 生成的潜在变量形状
    
            return image_latent, pred_latent  # 返回图像潜在变量和预测潜在变量
    
        def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:  # 解码预测潜在变量的方法
            if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:  # 检查预测潜在变量的维度和通道数
                raise ValueError(  # 如果不符合要求，抛出异常
                    f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
                )
    
            prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]  # 解码潜在变量，得到预测图像
    
            prediction = torch.clip(prediction, -1.0, 1.0)  # 限制预测值在 -1.0 到 1.0 之间
    
            if not self.use_full_z_range:  # 如果不使用完整的潜在范围
                prediction[:, 2, :, :] *= 0.5  # 对第三个通道进行缩放
                prediction[:, 2, :, :] += 0.5  # 对第三个通道进行偏移
    
            prediction = self.normalize_normals(prediction)  # [B,3,H,W]  # 正常化预测结果
    
            return prediction  # [B,3,H,W]  # 返回最终的预测图像
    
        @staticmethod  # 静态方法的标记
    # 规范化法线向量，使其单位长度，避免数值不稳定
    def normalize_normals(normals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # 检查输入的法线张量是否为4维且第二维的大小为3
        if normals.dim() != 4 or normals.shape[1] != 3:
            # 如果不满足条件，抛出错误
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")

        # 计算法线张量在第二维的范数，并保持维度
        norm = torch.norm(normals, dim=1, keepdim=True)
        # 将法线张量除以范数，使用clamp限制最小值以避免除以零
        normals /= norm.clamp(min=eps)

        # 返回规范化后的法线张量
        return normals

    @staticmethod
    # 对法线张量进行集成处理，返回集成后的法线和可选的不确定性
    def ensemble_normals(
        normals: torch.Tensor, output_uncertainty: bool, reduction: str = "closest"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        对法线图进行集成，期望输入形状为 `(B, 3, H, W)`，其中 B 是
        每个预测的集成成员数量，大小为 `(H x W)`。

        Args:
            normals (`torch.Tensor`):
                输入的集成法线图。
            output_uncertainty (`bool`, *可选*, 默认值为 `False`):
                是否输出不确定性图。
            reduction (`str`, *可选*, 默认值为 `"closest"`):
                用于集成对齐预测的归约方法。接受的值为：`"closest"` 和
                `"mean"`。

        Returns:
            返回形状为 `(1, 3, H, W)` 的对齐和集成法线图，以及可选的不确定性张量，形状为 `(1, 1, H, W)`。
        """
        # 检查输入的法线张量是否为4维且第二维的大小为3
        if normals.dim() != 4 or normals.shape[1] != 3:
            # 如果不满足条件，抛出错误
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")
        # 检查归约方法是否有效
        if reduction not in ("closest", "mean"):
            # 如果不合法，抛出错误
            raise ValueError(f"Unrecognized reduction method: {reduction}.")

        # 计算法线的均值，保持维度
        mean_normals = normals.mean(dim=0, keepdim=True)  # [1,3,H,W]
        # 规范化均值法线
        mean_normals = MarigoldNormalsPipeline.normalize_normals(mean_normals)  # [1,3,H,W]

        # 计算均值法线与所有法线的点积，得到相似度
        sim_cos = (mean_normals * normals).sum(dim=1, keepdim=True)  # [E,1,H,W]
        # 限制相似度值在 -1 到 1 之间，以避免在 fp16 中出现 NaN
        sim_cos = sim_cos.clamp(-1, 1)  # required to avoid NaN in uncertainty with fp16

        uncertainty = None
        # 如果需要输出不确定性
        if output_uncertainty:
            # 计算相似度的反余弦，得到不确定性
            uncertainty = sim_cos.arccos()  # [E,1,H,W]
            # 计算均值并归一化
            uncertainty = uncertainty.mean(dim=0, keepdim=True) / np.pi  # [1,1,H,W]

        # 如果选择平均归约方法
        if reduction == "mean":
            # 返回均值法线和不确定性
            return mean_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

        # 找到相似度最大的索引
        closest_indices = sim_cos.argmax(dim=0, keepdim=True)  # [1,1,H,W]
        # 将索引扩展到法线的通道数
        closest_indices = closest_indices.repeat(1, 3, 1, 1)  # [1,3,H,W]
        # 根据索引从法线中提取相应的法线
        closest_normals = torch.gather(normals, 0, closest_indices)  # [1,3,H,W]

        # 返回最近法线和不确定性
        return closest_normals, uncertainty  # [1,3,H,W], [1,1,H,W]
```