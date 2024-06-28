# `.\models\pix2struct\image_processing_pix2struct.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本使用此文件；除非符合许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不提供任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证
"""Pix2Struct 的图像处理类"""
import io  # 导入 io 模块
import math  # 导入 math 模块
from typing import Dict, Optional, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库
from huggingface_hub import hf_hub_download  # 从 HuggingFace Hub 导入模型下载函数

from ...image_processing_utils import BaseImageProcessor, BatchFeature  # 导入图像处理相关工具
from ...image_transforms import convert_to_rgb, normalize, to_channel_dimension_format, to_pil_image  # 导入图像转换函数
from ...image_utils import (  # 导入图像工具函数
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_vision_available, logging  # 导入工具函数和判断函数
from ...utils.import_utils import requires_backends  # 导入后端依赖检查函数

if is_vision_available():  # 如果可用视觉模块
    import textwrap  # 导入文本包装模块

    from PIL import Image, ImageDraw, ImageFont  # 从 PIL 库导入图像处理相关函数

if is_torch_available():  # 如果可用 PyTorch 模块
    import torch  # 导入 PyTorch 库

logger = logging.get_logger(__name__)  # 获取日志记录器
DEFAULT_FONT_PATH = "ybelkada/fonts"  # 设置默认字体路径


# 从 https://discuss.pytorch.org/t/tf-image-extract-patches-in-pytorch/171409/2 调整
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
    requires_backends(torch_extract_patches, ["torch"])  # 检查所需的后端是否可用

    image_tensor = image_tensor.unsqueeze(0)  # 在第一维度上添加一个维度，扩展为四维张量
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))  # 使用 PyTorch 的 unfold 函数提取补丁
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)  # 重塑张量形状
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )  # 调整张量顺序和形状
    return patches.unsqueeze(0)  # 返回四维张量


# 从 https://github.com/google-research/pix2struct/blob/0e1779af0f4db4b652c1d92b3bbd2550a7399123/pix2struct/preprocessing/preprocessing_utils.py#L106 调整
def render_text(
    text: str,
    text_size: int = 36,
    text_color: str = "black",
    background_color: str = "white",
    font_path: str = DEFAULT_FONT_PATH,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
):
    """
    渲染文本为图像的实用函数。

    Args:
        text (str):
            要渲染的文本内容。
        text_size (int, optional):
            文本的字体大小，默认为 36。
        text_color (str, optional):
            文本的颜色，默认为黑色。
        background_color (str, optional):
            背景的颜色，默认为白色。
        font_path (str, optional):
            字体文件的路径，默认为 DEFAULT_FONT_PATH。
        max_width (int, optional):
            最大宽度限制，默认为 None。
        max_height (int, optional):
            最大高度限制，默认为 None。
    """
    # 定义左边界填充量，默认为5个单位
    left_padding: int = 5,
    # 定义右边界填充量，默认为5个单位
    right_padding: int = 5,
    # 定义顶部填充量，默认为5个单位
    top_padding: int = 5,
    # 定义底部填充量，默认为5个单位
    bottom_padding: int = 5,
    # 字体的字节码数据，可选参数，默认为None
    font_bytes: Optional[bytes] = None,
    # 字体文件的路径，可选参数，默认为None
    font_path: Optional[str] = None,
def render_header(
    image: np.ndarray, header: str, input_data_format: Optional[Union[str, ChildProcessError]] = None, **kwargs
):
    """
    Renders the input text as a header on the input image.

    Args:
        image (`np.ndarray`):
            Input image represented as a NumPy array.
        header (`str`):
            Text to render as the header.
        input_data_format (`Optional[Union[str, ChildProcessError]]`, *optional*):
            Format of the input data. Defaults to `None`.
        **kwargs:
            Additional keyword arguments for customization.

    Returns:
        `Image.Image`:
            An image with the rendered header text.

    Note:
        This function renders the header text onto the given image using specified or default parameters.
        It adapts the text rendering from an external source.

    Adapted from:
    https://github.com/google-research/pix2struct/blob/0e1779af0f4db4b652c1d92b3bbd2550a7399123/pix2struct/preprocessing/preprocessing_utils.py#L87
    """

    # Ensure the necessary backend is available for rendering text.
    requires_backends(render_text, "vision")

    # Wrap the input text to fit within lines of 80 characters width.
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=header)
    wrapped_text = "\n".join(lines)

    # Determine the font source based on provided bytes or path, or use default.
    if font_bytes is not None and font_path is None:
        font = io.BytesIO(font_bytes)
    elif font_path is not None:
        font = font_path
    else:
        font = hf_hub_download(DEFAULT_FONT_PATH, "Arial.TTF")
    # Load the font using PIL's `ImageFont.truetype` method.
    font = ImageFont.truetype(font, encoding="UTF-8", size=text_size)

    # Create a temporary canvas to calculate text dimensions.
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

    # Determine the dimensions of the final image including padding.
    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding

    # Create a new image with specified dimensions and background color.
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Render the wrapped text onto the image at specified padding positions.
    draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)

    # Return the final rendered image with the header text.
    return image
    Args:
        image (`np.ndarray`):
            The image to render the header on.
        header (`str`):
            The header text.
        data_format (`Union[ChannelDimension, str]`, *optional*):
            The data format of the image. Can be either "ChannelDimension.channels_first" or
            "ChannelDimension.channels_last".

    Returns:
        `np.ndarray`: The image with the header rendered.
    """
    # 检查渲染头部所需的视觉后端是否存在
    requires_backends(render_header, "vision")

    # 如果需要，将输入的图像转换为PIL图像格式
    image = to_pil_image(image, input_data_format=input_data_format)

    # 使用渲染文本函数生成头部文本对应的图像
    header_image = render_text(header, **kwargs)

    # 计算新图像的宽度为头部图像和原始图像宽度的最大值
    new_width = max(header_image.width, image.width)

    # 计算新图像的高度，保持原始图像的宽高比
    new_height = int(image.height * (new_width / image.width))
    new_header_height = int(header_image.height * (new_width / header_image.width))

    # 创建新的RGB模式的白色背景图像，大小为新宽度和高度之和
    new_image = Image.new("RGB", (new_width, new_height + new_header_height), "white")

    # 将调整大小后的头部图像粘贴到新图像的顶部
    new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))

    # 将调整大小后的原始图像粘贴到新图像的下部
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

    # 如果需要，将新图像转换回原始数据格式
    new_image = to_numpy_array(new_image)

    # 如果推断出新图像的通道维度格式为最后一个维度
    if infer_channel_dimension_format(new_image) == ChannelDimension.LAST:
        # 将新图像转换为最后一个通道维度格式
        new_image = to_channel_dimension_format(new_image, ChannelDimension.LAST)

    # 返回渲染了头部的新图像
    return new_image
    r"""
    Constructs a Pix2Struct image processor.
    构造一个 Pix2Struct 图像处理器。

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
            是否将图像转换为 RGB。

        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard
            deviation.
            是否对图像进行归一化。可以通过 `preprocess` 方法中的 `do_normalize` 参数进行覆盖。
            根据 Pix2Struct 论文和代码，图像使用其自身的均值和标准差进行归一化。

        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
            图像使用的补丁大小。根据 Pix2Struct 论文和代码，补丁大小为 16x16。

        max_patches (`int`, *optional*, defaults to 2048):
            The maximum number of patches to extract from the image as per the [Pix2Struct
            paper](https://arxiv.org/pdf/2210.03347.pdf).
            从图像中提取的最大补丁数，根据 Pix2Struct 论文。

        is_vqa (`bool`, *optional*, defaults to `False`):
            Whether or not the image processor is for the VQA task. If `True` and `header_text` is passed in, text is
            rendered onto the input images.
            图像处理器是否用于 VQA 任务。如果为 `True` 并且传入了 `header_text`，则将文本渲染到输入图像上。
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
        super().__init__(**kwargs)  # 调用父类的初始化方法，传递所有未明确指定的关键字参数
        self.patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}  # 设置补丁大小，默认为 {"height": 16, "width": 16}
        self.do_normalize = do_normalize  # 是否进行归一化
        self.do_convert_rgb = do_convert_rgb  # 是否进行 RGB 转换
        self.max_patches = max_patches  # 最大提取补丁数
        self.is_vqa = is_vqa  # 是否用于 VQA 任务

    def extract_flattened_patches(
        self,
        image: np.ndarray,
        max_patches: int,
        patch_size: dict,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract flattened patches from an image.

        Args:
            image (`np.ndarray`):
                Image to extract flattened patches from.
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict`):
                Dictionary containing the patch height and width.

        Returns:
            result (`np.ndarray`):
                A sequence of `max_patches` flattened patches.
        """
        # 检查是否需要使用 torch 后端函数
        requires_backends(self.extract_flattened_patches, "torch")

        # 将图像转换为 torch 张量格式
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)
        image = torch.from_numpy(image)

        # 获取补丁的高度和宽度
        patch_height, patch_width = patch_size["height"], patch_size["width"]
        # 获取图像的高度和宽度
        image_height, image_width = get_image_size(image, ChannelDimension.FIRST)

        # 最大化比例以便适应给定的最大补丁数和图像尺寸
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        # 对图像进行插值调整大小
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        # 提取图像的补丁
        # [1, rows, columns, patch_height * patch_width * image_channels]
        patches = torch_extract_patches(image, patch_height, patch_width)

        # 获取补丁的形状信息
        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # 重新整形补丁张量以便进一步处理
        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # 创建行和列的索引张量
        # [rows * columns, 1]
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # 将索引张量的值加一，以避免包含代表填充的零
        row_ids += 1
        col_ids += 1

        # 准备额外的补丁特征信息
        # [rows * columns, 1]
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)

        # 拼接行号、列号和补丁数据，形成最终的输出结果
        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # 对结果进行填充，以保证输出的补丁数量不超过 max_patches
        # [max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        # 将结果转换为 NumPy 数组格式
        result = to_numpy_array(result)

        return result
    # 对图像进行标准化处理，使得图像数据的均值为0，标准差为1
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
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 如果图像的数据类型是uint8，则转换为float32类型
        if image.dtype == np.uint8:
            image = image.astype(np.float32)

        # 计算图像的均值和标准差
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

        # 调用标准化函数进行图像标准化处理，返回标准化后的图像数据
        return normalize(
            image,
            mean=mean,
            std=adjusted_stddev,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 图像预处理函数，可以进行RGB转换、标准化、裁剪等操作
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
    ):
```