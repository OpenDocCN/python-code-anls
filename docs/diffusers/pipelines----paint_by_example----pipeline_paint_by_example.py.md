# `.\diffusers\pipelines\paint_by_example\pipeline_paint_by_example.py`

```py
# 版权声明，表明该文件的所有权和使用许可
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证，使用该文件的条件
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 可以在此获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则软件按“原样”分发，不提供任何形式的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect  # 导入inspect模块，用于获取对象的信息
from typing import Callable, List, Optional, Union  # 导入类型提示功能

import numpy as np  # 导入numpy库，用于数值计算
import PIL.Image  # 导入PIL库，用于图像处理
import torch  # 导入PyTorch库，用于深度学习
from transformers import CLIPImageProcessor  # 导入CLIP图像处理器

from ...image_processor import VaeImageProcessor  # 从上级模块导入VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入模型相关类
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler  # 导入调度器
from ...utils import deprecate, logging  # 导入工具函数和日志记录
from ...utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from ..stable_diffusion import StableDiffusionPipelineOutput  # 导入稳定扩散管道的输出类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入安全检查器
from .image_encoder import PaintByExampleImageEncoder  # 从当前模块导入图像编码器

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于调试
# pylint: disable=invalid-name  # 禁用pylint关于名称的无效警告

# 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 检查encoder_output是否具有latent_dist属性且采样模式为'sample'
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 返回latent分布的样本
        return encoder_output.latent_dist.sample(generator)
    # 检查encoder_output是否具有latent_dist属性且采样模式为'argmax'
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回latent分布的众数
        return encoder_output.latent_dist.mode()
    # 检查encoder_output是否具有latents属性
    elif hasattr(encoder_output, "latents"):
        # 返回latents属性
        return encoder_output.latents
    # 如果都不满足，抛出异常
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# 准备图像和掩码以供“按示例绘制”管道使用
def prepare_mask_and_masked_image(image, mask):
    """
    准备一对 (image, mask)，使其可以被 Paint by Example 管道使用。
    这意味着这些输入将转换为``torch.Tensor``，形状为``batch x channels x height x width``，
    其中``channels``为``3``（对于``image``）和``1``（对于``mask``）。

    ``image`` 将转换为 ``torch.float32`` 并归一化为 ``[-1, 1]``。
    ``mask`` 将被二值化（``mask > 0.5``）并同样转换为 ``torch.float32``。
    ```
    # 函数参数说明
    Args:
        # 输入图像，类型可以是 np.array、PIL.Image 或 torch.Tensor
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            # 描述图像的不同可能格式，包括 PIL.Image、np.array 或 torch.Tensor
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        # 掩码，用于指定需要修复的区域
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            # 描述掩码的不同可能格式，类似于图像
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.

    # 异常说明
    Raises:
        # 触发条件为 torch.Tensor 格式图像或掩码的数值范围不正确
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        # 类型错误，当图像和掩码类型不匹配时抛出
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    # 返回值说明
    Returns:
        # 返回一个包含掩码和修复图像的元组，均为 torch.Tensor 格式，具有 4 个维度
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    # 检查输入图像是否为 torch.Tensor 类型
    if isinstance(image, torch.Tensor):
        # 如果掩码不是 torch.Tensor，抛出类型错误
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # 如果图像为单个图像，将其转换为批处理格式
        # Batch single image
        if image.ndim == 3:
            # 确保单个图像的形状为 (3, H, W)
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            # 在第一个维度添加批处理维度
            image = image.unsqueeze(0)

        # 如果掩码为二维，添加批处理和通道维度
        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            # 在前面添加两个维度
            mask = mask.unsqueeze(0).unsqueeze(0)

        # 如果掩码为三维，检查其与图像的批次匹配
        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Batched mask
            if mask.shape[0] == image.shape[0]:
                # 如果掩码的批次与图像相同，添加通道维度
                mask = mask.unsqueeze(1)
            else:
                # 否则，在前面添加批处理维度
                mask = mask.unsqueeze(0)

        # 确保图像和掩码都是四维
        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        # 确保图像和掩码的空间维度相同
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        # 确保图像和掩码的批处理大小相同
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"
        # 确保掩码只有一个通道
        assert mask.shape[1] == 1, "Mask image must have a single channel"

        # 检查图像的数值范围是否在 [-1, 1] 之间
        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # 检查掩码的数值范围是否在 [0, 1] 之间
        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # 对掩码进行反转，以便于修复
        # paint-by-example inverses the mask
        mask = 1 - mask

        # 二值化掩码
        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # 将图像转换为 float32 类型
        # Image as float32
        image = image.to(dtype=torch.float32)
    # 如果掩码是 torch.Tensor 类型，但图像不是，则抛出类型错误
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # 如果输入的 image 是 PIL 图像对象，则将其转换为列表
        if isinstance(image, PIL.Image.Image):
            image = [image]

        # 将每个图像转换为 RGB 格式，并拼接成一个数组，增加维度以适应后续处理
        image = np.concatenate([np.array(i.convert("RGB"))[None, :] for i in image], axis=0)
        # 将图像数组的维度顺序调整为 (批量, 通道, 高, 宽)
        image = image.transpose(0, 3, 1, 2)
        # 将 NumPy 数组转换为 PyTorch 张量并归一化到 [-1, 1] 范围
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # 处理 mask
        # 如果输入的 mask 是 PIL 图像对象，则将其转换为列表
        if isinstance(mask, PIL.Image.Image):
            mask = [mask]

        # 将每个掩膜图像转换为灰度格式，并拼接成一个数组，增加维度以适应后续处理
        mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
        # 将掩膜数组转换为 float32 类型并归一化到 [0, 1] 范围
        mask = mask.astype(np.float32) / 255.0

        # paint-by-example 方法反转掩膜
        mask = 1 - mask

        # 将掩膜中低于 0.5 的值设置为 0，高于或等于 0.5 的值设置为 1
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        # 将 NumPy 数组转换为 PyTorch 张量
        mask = torch.from_numpy(mask)

    # 将图像与掩膜相乘，得到被掩膜处理的图像
    masked_image = image * mask

    # 返回掩膜和被掩膜处理的图像
    return mask, masked_image
# 定义一个名为 PaintByExamplePipeline 的类，继承自 DiffusionPipeline 和 StableDiffusionMixin
class PaintByExamplePipeline(DiffusionPipeline, StableDiffusionMixin):
    r""" 
    # 警告提示，表示这是一个实验性特性
    <Tip warning={true}>
    🧪 This is an experimental feature!
    </Tip>

    # 使用 Stable Diffusion 进行图像引导的图像修补的管道。

    # 该模型从 [`DiffusionPipeline`] 继承。检查超类文档以获取所有管道的通用方法
    # （下载、保存、在特定设备上运行等）。

    # 参数说明：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        image_encoder ([`PaintByExampleImageEncoder`]):
            编码示例输入图像。`unet` 是基于示例图像而非文本提示进行条件处理。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于文本分词的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码图像潜在的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用以去噪编码图像潜在的调度器，可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            估计生成图像是否可能被视为冒犯或有害的分类模块。
            请参考 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 以获取有关模型潜在危害的更多细节。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成图像中提取特征的 `CLIPImageProcessor`；用作 `safety_checker` 的输入。
    """

    # TODO: 如果管道没有 feature_extractor，则需要在初始图像（如果为 PIL 格式）编码时给出描述性消息。

    # 定义模型在 CPU 上卸载的顺序，指定 'unet' 在前，'vae' 在后
    model_cpu_offload_seq = "unet->vae"
    # 定义在 CPU 卸载时排除的组件，指定 'image_encoder' 不参与卸载
    _exclude_from_cpu_offload = ["image_encoder"]
    # 定义可选组件，指定 'safety_checker' 为可选
    _optional_components = ["safety_checker"]

    # 初始化方法，设置管道的主要组件
    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: PaintByExampleImageEncoder,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = False,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模块，设置管道的组成部分
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # 计算 VAE 的缩放因子，基于 VAE 配置中的块输出通道数
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化 VaeImageProcessor，使用计算出的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 将是否需要安全检查器的信息注册到配置中
        self.register_to_config(requires_safety_checker=requires_safety_checker)
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制而来
        def run_safety_checker(self, image, device, dtype):
            # 如果安全检查器不存在，将有害概念标记为 None
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                # 如果输入图像是张量，使用图像处理器后处理为 PIL 格式
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果输入图像不是张量，将其转换为 PIL 格式
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 使用特征提取器处理图像并转移到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器，返回处理后的图像和有害概念的存在情况
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和有害概念
            return image, has_nsfw_concept
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制而来
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的签名
            # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
            # 应在 [0, 1] 之间
    
            # 检查调度器是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                # 如果接受 eta，将其添加到额外参数中
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            if accepts_generator:
                # 如果接受 generator，将其添加到额外参数中
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制而来
        def decode_latents(self, latents):
            # 警告信息，提示 decode_latents 方法已弃用，将在 1.0.0 中移除
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 记录弃用警告
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 根据 VAE 的缩放因子调整潜在向量
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在向量，返回图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像缩放到 [0, 1] 范围内
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为 float32 格式以兼容 bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回处理后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_image_variation.StableDiffusionImageVariationPipeline.check_inputs 复制而来
    # 检查输入参数的有效性
    def check_inputs(self, image, height, width, callback_steps):
        # 检查 `image` 是否为有效类型，必须是 `torch.Tensor`、`PIL.Image.Image` 或列表
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            # 如果 `image` 类型不符合，抛出错误
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        # 检查 `height` 和 `width` 是否能被 8 整除
        if height % 8 != 0 or width % 8 != 0:
            # 如果不能整除，抛出错误
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查 `callback_steps` 是否有效，必须是正整数或 None
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            # 如果无效，抛出错误
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # 从 StableDiffusionPipeline 复制的准备潜在变量的方法
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器列表的长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果不匹配，抛出错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在变量，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，将其转移到指定设备
            latents = latents.to(device)

        # 将初始噪声缩放到调度器所需的标准差
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在变量
        return latents

    # 从 StableDiffusionInpaintPipeline 复制的准备掩膜潜在变量的方法
    def prepare_mask_latents(
        # 定义方法的输入参数，包括掩膜和其他信息
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # 将掩码调整为与潜变量形状相同，以便将掩码与潜变量拼接
        # 这样做可以避免在使用 cpu_offload 和半精度时出现问题
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )  # 通过插值调整掩码的大小
        mask = mask.to(device=device, dtype=dtype)  # 将掩码移动到指定设备并转换数据类型

        masked_image = masked_image.to(device=device, dtype=dtype)  # 将掩码图像移动到指定设备并转换数据类型

        if masked_image.shape[1] == 4:  # 检查掩码图像是否为四通道
            masked_image_latents = masked_image  # 如果是，直接将其赋值给潜变量
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)  # 否则，使用 VAE 编码图像

        # 针对每个提示重复掩码和掩码图像潜变量，使用适合 MPS 的方法
        if mask.shape[0] < batch_size:  # 检查掩码的数量是否少于批处理大小
            if not batch_size % mask.shape[0] == 0:  # 检查掩码数量是否可整除批处理大小
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )  # 如果不匹配，抛出值错误
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)  # 重复掩码以匹配批处理大小
        if masked_image_latents.shape[0] < batch_size:  # 检查潜变量数量是否少于批处理大小
            if not batch_size % masked_image_latents.shape[0] == 0:  # 检查潜变量数量是否可整除批处理大小
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )  # 如果不匹配，抛出值错误
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)  # 重复潜变量以匹配批处理大小

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask  # 根据是否使用无分类器引导选择掩码
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )  # 根据是否使用无分类器引导选择潜变量

        # 调整设备以防止与潜变量模型输入拼接时出现设备错误
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)  # 将潜变量移动到指定设备并转换数据类型
        return mask, masked_image_latents  # 返回处理后的掩码和潜变量

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline._encode_vae_image 复制
    # 定义一个编码变分自编码器图像的私有方法
        def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
            # 检查生成器是否为列表类型
            if isinstance(generator, list):
                # 对每个图像编码并获取潜在表示，使用对应的生成器
                image_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(image.shape[0])
                ]
                # 将所有潜在表示在第0维上拼接成一个张量
                image_latents = torch.cat(image_latents, dim=0)
            else:
                # 如果生成器不是列表，直接编码图像并获取潜在表示
                image_latents = retrieve_latents(self.vae.encode(image), generator=generator)
    
            # 将潜在表示乘以缩放因子
            image_latents = self.vae.config.scaling_factor * image_latents
    
            # 返回编码后的潜在表示
            return image_latents
    
        # 定义一个编码图像的私有方法
        def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入图像是否为张量，如果不是则提取特征
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备，并转换为正确的数据类型
            image = image.to(device=device, dtype=dtype)
            # 对图像进行编码，获取图像嵌入和负提示嵌入
            image_embeddings, negative_prompt_embeds = self.image_encoder(image, return_uncond_vector=True)
    
            # 复制图像嵌入以适应每个提示的生成数量
            bs_embed, seq_len, _ = image_embeddings.shape
            image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
            # 重塑嵌入张量的形状
            image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
            # 检查是否使用无分类器引导
            if do_classifier_free_guidance:
                # 复制负提示嵌入以匹配图像嵌入的数量
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, image_embeddings.shape[0], 1)
                negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, 1, -1)
    
                # 为无分类器引导执行两个前向传播，通过拼接无条件和文本嵌入来避免两个前向传播
                image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    
            # 返回编码后的图像嵌入
            return image_embeddings
    
        # 定义一个调用方法，禁用梯度计算以提高效率
        @torch.no_grad()
        def __call__(
            # 接收示例图像和图像的参数，允许不同类型的输入
            example_image: Union[torch.Tensor, PIL.Image.Image],
            image: Union[torch.Tensor, PIL.Image.Image],
            mask_image: Union[torch.Tensor, PIL.Image.Image],
            # 可选参数定义图像的高度和宽度
            height: Optional[int] = None,
            width: Optional[int] = None,
            # 定义推理步骤的数量和引导缩放比例
            num_inference_steps: int = 50,
            guidance_scale: float = 5.0,
            # 负提示的可选输入
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: Optional[int] = 1,
            # 控制采样多样性的参数
            eta: float = 0.0,
            # 生成器的可选输入
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量输入
            latents: Optional[torch.Tensor] = None,
            # 输出类型的可选参数
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果
            return_dict: bool = True,
            # 可选的回调函数用于处理中间结果
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 每隔多少步调用一次回调
            callback_steps: int = 1,
```