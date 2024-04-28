# `.\transformers\models\pix2struct\image_processing_pix2struct.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 版权所有 2023 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，不附带任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制。
"""Pix2Struct 的图像处理器类。"""
import io
import math
from typing import Dict, Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb, normalize, to_channel_dimension_format, to_pil_image
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_vision_available, logging
from ...utils.import_utils import requires_backends

# 如果有视觉库可用
if is_vision_available():
    import textwrap

    from PIL import Image, ImageDraw, ImageFont

# 如果有 Torch 库可用
if is_torch_available():
    import torch

logger = logging.get_logger(__name__)
# 默认字体路径
DEFAULT_FONT_PATH = "ybelkada/fonts"

# 从 https://discuss.pytorch.org/t/tf-image-extract-patches-in-pytorch/171409/2 改编
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    从给定的图像张量中提取补丁的实用函数。返回形状为 (1, `patch_height`, `patch_width`, `num_channels`x `patch_height` x `patch_width`) 的张量

    Args:
        image_tensor (torch.Tensor):
            要从中提取补丁的图像张量。
        patch_height (int):
            要提取的补丁的高度。
        patch_width (int):
            要提取的补丁的宽度。
    """
    requires_backends(torch_extract_patches, ["torch"])

    # 在第 0 维度上增加一个维度
    image_tensor = image_tensor.unsqueeze(0)
    # 使用指定的步幅展开图像张量
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    # 重塑张量形状
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    # 转置张量
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)

# 从 https://github.com/google-research/pix2struct/blob/0e1779af0f4db4b652c1d92b3bbd2550a7399123/pix2struct/preprocessing/preprocessing_utils.py#L106 改编
def render_text(
    text: str,
    text_size: int = 36,
    text_color: str = "black",
    background_color: str = "white",
    # 定义左边距，默认为5
    left_padding: int = 5,
    # 定义右边距，默认为5
    right_padding: int = 5,
    # 定义顶部边距，默认为5
    top_padding: int = 5,
    # 定义底部边距，默认为5
    bottom_padding: int = 5,
    # 定义字体的字节流，默认为None
    font_bytes: Optional[bytes] = None,
    # 定义字体文件的路径，默认为None
    font_path: Optional[str] = None,
# 定义一个函数，用于渲染文本成图像
def render_text(
    text: str = "",
    text_size: int = 36,
    text_color: str = "black",
    background_color: str = "white",
    left_padding: int = 5,
    right_padding: int = 5,
    top_padding: int = 5,
    bottom_padding: int = 5,
    font_bytes: bytes = None,
    font_path: str = None
) -> Image.Image:
    # 检查是否需要导入 vision 模块
    requires_backends(render_text, "vision")

    # 创建一个文本包装器，限制每行文本长度为80个字符
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)
    wrapped_text = "\n".join(lines)

    # 根据字体字节或字体路径创建字体对象
    if font_bytes is not None and font_path is None:
        font = io.BytesIO(font_bytes)
    elif font_path is not None:
        font = font_path
    else:
        font = hf_hub_download(DEFAULT_FONT_PATH, "Arial.TTF")
    font = ImageFont.truetype(font, encoding="UTF-8", size=text_size)

    # 使用临时画布确定渲染文本时的宽度和高度
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

    # 创建带有一定填充的实际图像
    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)
    return image

# 从给定链接中适配的函数，用于在输入图像上渲染输入文本作为标题
def render_header(
    image: np.ndarray, header: str, input_data_format: Optional[Union[str, ChildProcessError]] = None, **kwargs
):
    """
    Renders the input text as a header on the input image.
    """
    Args:
        image (`np.ndarray`):
            The image to render the header on. 用于在其上渲染标题的图像
        header (`str`):
            The header text. 标题文本
        data_format (`Union[ChannelDimension, str]`, *optional*):
            The data format of the image. Can be either "ChannelDimension.channels_first" or
            "ChannelDimension.channels_last". 图像的数据格式，可以是"ChannelDimension.channels_first"或"ChannelDimension.channels_last"

    Returns:
        `np.ndarray`: The image with the header rendered. 渲染了标题的图像
    """
    requires_backends(render_header, "vision")

    # Convert to PIL image if necessary 如果需要，将图像转换为PIL图像
    image = to_pil_image(image, input_data_format=input_data_format)

    # Render the header text as an image 将标题文本渲染为图像
    header_image = render_text(header, **kwargs)
    new_width = max(header_image.width, image.width)

    new_height = int(image.height * (new_width / image.width))
    new_header_height = int(header_image.height * (new_width / header_image.width))

    # Create a new image with the header and original image 创建一个包含标题和原始图像的新图像
    new_image = Image.new("RGB", (new_width, new_height + new_header_height), "white")
    new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

    # Convert back to the original framework if necessary 如果需要，将图像转换回原始框架
    new_image = to_numpy_array(new_image)

    # Check and adjust the channel dimension format if necessary 如果需要，检查并调整通道维度格式
    if infer_channel_dimension_format(new_image) == ChannelDimension.LAST:
        new_image = to_channel_dimension_format(new_image, ChannelDimension.LAST)

    return new_image
class Pix2StructImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Pix2Struct image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard
            deviation.
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
        max_patches (`int`, *optional*, defaults to 2048):
            The maximum number of patches to extract from the image as per the [Pix2Struct
            paper](https://arxiv.org/pdf/2210.03347.pdf).
        is_vqa (`bool`, *optional*, defaults to `False`):
            Whether or not the image processor is for the VQA task. If `True` and `header_text` is passed in, text is
            rendered onto the input images.
    """

    model_input_names = ["flattened_patches"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_normalize: bool = True,
        patch_size: Dict[str, int] = None,
        max_patches: int = 2048,
        is_vqa: bool = False,
        **kwargs,
    ) -> None:
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 设置默认的 patch 大小为 16x16
        self.patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}
        # 设置是否进行归一化
        self.do_normalize = do_normalize
        # 设置是否转换为 RGB
        self.do_convert_rgb = do_convert_rgb
        # 设置最大 patch 数量
        self.max_patches = max_patches
        # 设置是否用于 VQA 任务
        self.is_vqa = is_vqa

    def extract_flattened_patches(
        self,
        image: np.ndarray,
        max_patches: int,
        patch_size: dict,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        从图像中提取扁平化的补丁。

        Args:
            image (`np.ndarray`):
                要从中提取扁平化补丁的图像。
            max_patches (`int`):
                要提取的最大补丁数。
            patch_size (`dict`):
                包含补丁高度和宽度的字典。

        Returns:
            result (`np.ndarray`):
                一个包含 `max_patches` 个扁平化补丁的序列。
        """
        requires_backends(self.extract_flattened_patches, "torch")

        # 转换为 torch 格式
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)
        image = torch.from_numpy(image)

        patch_height, patch_width = patch_size["height"], patch_size["width"]
        image_height, image_width = get_image_size(image, ChannelDimension.FIRST)

        # 最大化比例，使得
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        # [1, rows, columns, patch_height * patch_width * image_channels]
        patches = torch_extract_patches(image, patch_height, patch_width)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # [rows * columns, 1]
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # 偏移 1，以避免包含表示填充的零值。
        row_ids += 1
        col_ids += 1

        # 准备额外的补丁特征。
        # [rows * columns, 1]
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)

        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # [max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        result = to_numpy_array(result)

        return result
    # 定义一个方法用于对图像进行归一化处理
    def normalize(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        The image std is to mimic the tensorflow implementation of the `per_image_standardization`:
        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

        Args:
            image (`np.ndarray`):
                Image to normalize.
            data_format (`str` or `ChannelDimension`, *optional`):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional`):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 如果图像的数据类型是 uint8，则转换为 float32
        if image.dtype == np.uint8:
            image = image.astype(np.float32)

        # 计算图像的均值和标准差
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

        # 返回归一化后的图像
        return normalize(
            image,
            mean=mean,
            std=adjusted_stddev,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 定义一个方法用于预处理图像数据
    def preprocess(
        self,
        images: ImageInput,
        header_text: Optional[str] = None,
        do_convert_rgb: bool = None,
        do_normalize: Optional[bool] = None,
        max_patches: Optional[int] = None,
        patch_size: Optional[Dict[str, int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```