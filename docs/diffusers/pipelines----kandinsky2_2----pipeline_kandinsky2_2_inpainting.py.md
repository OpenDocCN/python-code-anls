# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2_inpainting.py`

```py
# 版权信息，表明该文件属于 HuggingFace 团队，所有权利保留
# 
# 根据 Apache 许可证 2.0 版（"许可证"）授权；
# 您只能在符合许可证的情况下使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面协议另有约定，软件
# 按"原样"分发，不提供任何形式的保证或条件，无论是明示或暗示的。
# 有关许可证下权限和限制的具体信息，请参见许可证。

# 从 copy 模块导入 deepcopy 函数，用于深拷贝对象
from copy import deepcopy
# 导入类型提示相关的类型
from typing import Callable, Dict, List, Optional, Union

# 导入 numpy 库并赋予别名 np，常用于数值计算
import numpy as np
# 导入 PIL 库中的 Image 模块，用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块，用于实现功能性操作
import torch.nn.functional as F
# 导入 packaging 库中的 version 模块，用于版本管理
from packaging import version
# 从 PIL 导入 Image 类，用于处理图像对象
from PIL import Image

# 从当前包中导入版本信息
from ... import __version__
# 从模型模块导入 UNet2DConditionModel 和 VQModel 类
from ...models import UNet2DConditionModel, VQModel
# 从调度器模块导入 DDPMScheduler 类
from ...schedulers import DDPMScheduler
# 从 utils 模块导入 deprecate 和 logging 功能
from ...utils import deprecate, logging
# 从 utils.torch_utils 导入 randn_tensor 函数，用于生成随机张量
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 导入 DiffusionPipeline 和 ImagePipelineOutput 类
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 创建一个日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该模块的功能
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
        >>> from diffusers.utils import load_image
        >>> import torch
        >>> import numpy as np

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")

        >>> prompt = "a hat"
        >>> image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

        >>> pipe = KandinskyV22InpaintPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )

        >>> mask = np.zeros((768, 768), dtype=np.float32)
        >>> mask[:250, 250:-250] = 1

        >>> out = pipe(
        ...     image=init_image,
        ...     mask_image=mask,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... )

        >>> image = out.images[0]
        >>> image.save("cat_with_hat.png")
        ```py
"""

# 从 diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2 模块复制的函数
def downscale_height_and_width(height, width, scale_factor=8):
    # 计算新的高度，将原高度除以缩放因子的平方
    new_height = height // scale_factor**2
    # 如果原高度不能被缩放因子的平方整除，新的高度加 1
    if height % scale_factor**2 != 0:
        new_height += 1
    # 计算新的宽度，将原宽度除以缩放因子的平方
    new_width = width // scale_factor**2
    # 如果原宽度不能被缩放因子的平方整除，新的宽度加 1
    if width % scale_factor**2 != 0:
        new_width += 1
    # 返回调整后的高度和宽度，均乘以缩放因子
        return new_height * scale_factor, new_width * scale_factor
# 从 diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint.prepare_mask 复制
def prepare_mask(masks):
    # 初始化一个空列表，用于存储处理后的掩码
    prepared_masks = []
    # 遍历输入的每个掩码
    for mask in masks:
        # 深拷贝当前掩码，确保不修改原始数据
        old_mask = deepcopy(mask)
        # 遍历掩码的每一行
        for i in range(mask.shape[1]):
            # 遍历掩码的每一列
            for j in range(mask.shape[2]):
                # 如果当前像素值为1，则跳过
                if old_mask[0][i][j] == 1:
                    continue
                # 如果不是第一行，设置上方像素为0
                if i != 0:
                    mask[:, i - 1, j] = 0
                # 如果不是第一列，设置左侧像素为0
                if j != 0:
                    mask[:, i, j - 1] = 0
                # 如果不是第一行和第一列，设置左上角像素为0
                if i != 0 and j != 0:
                    mask[:, i - 1, j - 1] = 0
                # 如果不是最后一行，设置下方像素为0
                if i != mask.shape[1] - 1:
                    mask[:, i + 1, j] = 0
                # 如果不是最后一列，设置右侧像素为0
                if j != mask.shape[2] - 1:
                    mask[:, i, j + 1] = 0
                # 如果不是最后一行和最后一列，设置右下角像素为0
                if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                    mask[:, i + 1, j + 1] = 0
        # 将处理后的掩码添加到列表中
        prepared_masks.append(mask)
    # 将所有处理后的掩码堆叠成一个张量并返回
    return torch.stack(prepared_masks, dim=0)


# 从 diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint.prepare_mask_and_masked_image 复制
def prepare_mask_and_masked_image(image, mask, height, width):
    r"""
    准备一对（掩码，图像），以便由 Kandinsky 修复管道使用。这意味着这些输入将
    被转换为 ``torch.Tensor``，形状为 ``batch x channels x height x width``，其中 ``channels`` 为 ``3``，
    对于 ``image`` 和 ``1``，对于 ``mask``。

    ``image`` 将被转换为 ``torch.float32`` 并归一化到 ``[-1, 1]``。``mask`` 将被
    二值化（``mask > 0.5``）并同样转换为 ``torch.float32``。

    参数：
        image (Union[np.array, PIL.Image, torch.Tensor]): 要修复的图像。
            可以是 ``PIL.Image``，或 ``height x width x 3`` 的 ``np.array``，或 ``channels x height x width`` 的
            ``torch.Tensor``，或者是 ``batch x channels x height x width`` 的 ``torch.Tensor``。
        mask (_type_): 要应用于图像的掩码，即需要修复的区域。
            可以是 ``PIL.Image``，或 ``height x width`` 的 ``np.array``，或 ``1 x height x width`` 的
            ``torch.Tensor``，或是 ``batch x 1 x height x width`` 的 ``torch.Tensor``。
        height (`int`, *可选*, 默认值为 512):
            生成图像的高度（以像素为单位）。
        width (`int`, *可选*, 默认值为 512):
            生成图像的宽度（以像素为单位）。

    引发：
        ValueError: ``torch.Tensor`` 图像应在 ``[-1, 1]`` 范围内。ValueError: ``torch.Tensor`` 掩码
        应在 ``[0, 1]`` 范围内。ValueError: ``mask`` 和 ``image`` 应具有相同的空间维度。
        TypeError: ``mask`` 是 ``torch.Tensor`` 但 ``image`` 不是
            （反之亦然）。

    返回：
        tuple[torch.Tensor]: 将对（掩码，图像）作为 ``torch.Tensor``，具有 4
            维度：``batch x channels x height x width``。
    """

    # 如果输入图像为 None，抛出错误
    if image is None:
        raise ValueError("`image` input cannot be undefined.")
    # 检查 mask 是否为 None，如果是则抛出错误
    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    # 检查 image 是否为 torch.Tensor 类型
    if isinstance(image, torch.Tensor):
        # 检查 mask 是否为 torch.Tensor 类型，如果不是则抛出错误
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # 如果 image 是单张图像（3 个维度）
        if image.ndim == 3:
            # 确保图像的第一个维度为 3，表示 RGB 通道
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            # 在第一个维度增加一个批量维度
            image = image.unsqueeze(0)

        # 如果 mask 是单张图像（2 个维度）
        if mask.ndim == 2:
            # 在第一个和第二个维度增加批量和通道维度
            mask = mask.unsqueeze(0).unsqueeze(0)

        # 如果 mask 是 3 个维度
        if mask.ndim == 3:
            # 如果 mask 是单张图像（第一个维度为 1）
            if mask.shape[0] == 1:
                # 增加一个批量维度
                mask = mask.unsqueeze(0)

            # 如果 mask 是批量图像（没有通道维度）
            else:
                # 在第二个维度增加通道维度
                mask = mask.unsqueeze(1)

        # 确保 image 和 mask 都是 4 维张量
        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        # 确保 image 和 mask 的空间维度相同
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        # 确保 image 和 mask 的批量大小相同
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # 检查 image 的值是否在 [-1, 1] 范围内
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # 检查 mask 的值是否在 [0, 1] 范围内
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # 将 mask 二值化，低于 0.5 的值设为 0，其余设为 1
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # 将 image 转换为 float32 类型
        image = image.to(dtype=torch.float32)
    # 如果 mask 是张量但 image 不是，则抛出错误
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # 预处理图像
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            # 如果输入是单张图像或数组，则将其封装成列表
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # 如果列表中的元素是图像，按照传入的高度和宽度调整所有图像的大小
            image = [i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1) for i in image]
            # 将调整大小后的图像转换为 RGB 格式的 NumPy 数组，并增加一个维度
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            # 沿着第一个维度拼接所有图像数组
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            # 如果列表中的元素是 NumPy 数组，沿着第一个维度拼接这些数组
            image = np.concatenate([i[None, :] for i in image], axis=0)

        # 将图像数组的维度顺序从 (N, H, W, C) 转换为 (N, C, H, W)
        image = image.transpose(0, 3, 1, 2)
        # 将 NumPy 数组转换为 PyTorch 张量，并归一化到 [-1, 1] 的范围
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # 预处理掩膜
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            # 如果输入是单张掩膜或数组，则将其封装成列表
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            # 如果列表中的元素是掩膜图像，调整所有掩膜的大小
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            # 将调整大小后的掩膜转换为灰度格式的 NumPy 数组，并增加两个维度
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            # 将掩膜的值归一化到 [0, 1] 的范围
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            # 如果列表中的元素是 NumPy 数组，沿着第一个维度拼接这些数组
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        # 将掩膜中小于 0.5 的值设置为 0，大于等于 0.5 的值设置为 1
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        # 将掩膜转换为 PyTorch 张量
        mask = torch.from_numpy(mask)

    # 将掩膜的值反转，得到 1 - mask
    mask = 1 - mask

    # 返回处理后的掩膜和图像
    return mask, image
# Kandinsky2.1的文本引导图像修复管道类，继承自DiffusionPipeline
class KandinskyV22InpaintPipeline(DiffusionPipeline):
    """
    使用Kandinsky2.1进行文本引导图像修复的管道

    该模型继承自[`DiffusionPipeline`]。请查看超类文档以获取库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等）

    参数：
        scheduler ([`DDIMScheduler`]):
            与`unet`结合使用的调度器，用于生成图像潜变量。
        unet ([`UNet2DConditionModel`]):
            条件U-Net架构，用于去噪图像嵌入。
        movq ([`VQModel`]):
            MoVQ解码器，用于从潜变量生成图像。
    """

    # 定义模型的CPU卸载顺序
    model_cpu_offload_seq = "unet->movq"
    # 定义回调张量输入的列表
    _callback_tensor_inputs = ["latents", "image_embeds", "negative_image_embeds", "masked_image", "mask_image"]

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        # 初始化父类DiffusionPipeline
        super().__init__()

        # 注册模型的各个模块
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        # 计算MoVQ缩放因子
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)
        # 初始化警告标志
        self._warn_has_been_called = False

    # 从diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline复制的prepare_latents方法
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果潜变量为None，生成随机潜变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜变量的形状是否与期望的形状匹配
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜变量转移到指定设备
            latents = latents.to(device)

        # 将潜变量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜变量
        return latents

    # 获取引导缩放因子的属性
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 获取是否使用分类器自由引导的属性
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # 获取时间步数的属性
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 使用torch.no_grad()装饰器，避免梯度计算
    @torch.no_grad()
    def __call__(
        self,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        image: Union[torch.Tensor, PIL.Image.Image],
        mask_image: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
```