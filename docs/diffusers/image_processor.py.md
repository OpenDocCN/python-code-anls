# `.\diffusers\image_processor.py`

```py
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 版权声明，说明该代码的版权归 HuggingFace 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 在遵守许可证的前提下，可以使用此文件
# you may not use this file except in compliance with the License.
# 使用本文件必须遵循许可证的条款
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 可以在指定的网址获取许可证副本
#
# Unless required by applicable law or agreed to in writing, software
# 在法律要求或书面协议下，软件的分发是按“现状”基础进行的
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# 查看许可证以获取适用的权限和限制
# limitations under the License.

import math
# 导入数学库，用于数学运算
import warnings
# 导入警告库，用于发出警告信息
from typing import List, Optional, Tuple, Union
# 从 typing 模块导入类型注解，便于类型检查

import numpy as np
# 导入 numpy 库，用于数组操作
import PIL.Image
# 导入 PIL.Image，用于图像处理
import torch
# 导入 PyTorch 库，用于深度学习操作
import torch.nn.functional as F
# 导入 PyTorch 的函数式API，用于神经网络操作
from PIL import Image, ImageFilter, ImageOps
# 从 PIL 导入图像处理相关类

from .configuration_utils import ConfigMixin, register_to_config
# 从配置工具模块导入 ConfigMixin 和 register_to_config，用于配置管理
from .utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate
# 从工具模块导入常量和函数

PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]
# 定义图像输入的类型，可以是单个图像或图像列表

PipelineDepthInput = PipelineImageInput
# 深度输入类型与图像输入类型相同

def is_valid_image(image):
    # 检查输入是否为有效图像
    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)
    # 如果是 PIL 图像，或 2D/3D 的 numpy 数组或 PyTorch 张量，则返回 True

def is_valid_image_imagelist(images):
    # 检查图像输入是否为支持的格式，支持以下三种格式：
    # (1) 4D 的 PyTorch 张量或 numpy 数组
    # (2) 有效图像：PIL.Image.Image，2D np.ndarray 或 torch.Tensor（灰度图像），3D np.ndarray 或 torch.Tensor
    # (3) 有效图像列表
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
        return True
        # 如果是 4D 的 numpy 数组或 PyTorch 张量，返回 True
    elif is_valid_image(images):
        return True
        # 如果是有效的单个图像，返回 True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
        # 如果是列表，检查列表中每个图像是否有效，全部有效则返回 True
    return False
    # 如果不满足以上条件，返回 False

class VaeImageProcessor(ConfigMixin):
    # 定义 VAE 图像处理器类，继承自 ConfigMixin
    """
    Image processor for VAE.
    # VAE 的图像处理器
    # 参数列表，定义该类或函数的输入参数
        Args:
            do_resize (`bool`, *optional*, defaults to `True`):  # 是否将图像的高度和宽度缩放到 `vae_scale_factor` 的倍数
                Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
                `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
            vae_scale_factor (`int`, *optional*, defaults to `8`):  # VAE缩放因子，影响图像的缩放行为
                VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
            resample (`str`, *optional*, defaults to `lanczos`):  # 指定图像缩放时使用的重采样滤波器
                Resampling filter to use when resizing the image.
            do_normalize (`bool`, *optional*, defaults to `True`):  # 是否将图像归一化到[-1,1]的范围
                Whether to normalize the image to [-1,1].
            do_binarize (`bool`, *optional*, defaults to `False`):  # 是否将图像二值化为0或1
                Whether to binarize the image to 0/1.
            do_convert_rgb (`bool`, *optional*, defaults to be `False`):  # 是否将图像转换为RGB格式
                Whether to convert the images to RGB format.
            do_convert_grayscale (`bool`, *optional*, defaults to be `False`):  # 是否将图像转换为灰度格式
                Whether to convert the images to grayscale format.
        """
    
        config_name = CONFIG_NAME  # 将配置名称赋值给config_name变量
    
        @register_to_config  # 装饰器，将该函数注册为配置项
        def __init__((
            self,
            do_resize: bool = True,  # 初始化时的参数，是否缩放图像
            vae_scale_factor: int = 8,  # VAE缩放因子，默认值为8
            vae_latent_channels: int = 4,  # VAE潜在通道数，默认值为4
            resample: str = "lanczos",  # 重采样滤波器的默认值为lanczos
            do_normalize: bool = True,  # 初始化时的参数，是否归一化图像
            do_binarize: bool = False,  # 初始化时的参数，是否二值化图像
            do_convert_rgb: bool = False,  # 初始化时的参数，是否转换为RGB格式
            do_convert_grayscale: bool = False,  # 初始化时的参数，是否转换为灰度格式
        ):
            super().__init__()  # 调用父类的初始化方法
            if do_convert_rgb and do_convert_grayscale:  # 检查同时设置RGB和灰度格式的情况
                raise ValueError(  # 抛出值错误，提示不允许同时设置为True
                    "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                    " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                    " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
                )
    
        @staticmethod  # 静态方法，不依赖于类的实例
        def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:  # 将numpy数组转换为PIL图像列表
            """
            Convert a numpy image or a batch of images to a PIL image.
            """
            if images.ndim == 3:  # 检查图像是否为三维数组（单个图像）
                images = images[None, ...]  # 将其扩展为四维数组
            images = (images * 255).round().astype("uint8")  # 将图像值从[0, 1]转换为[0, 255]并转为无符号8位整数
            if images.shape[-1] == 1:  # 如果是单通道（灰度）图像
                # special case for grayscale (single channel) images
                pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]  # 转换为PIL灰度图像
            else:  # 处理多通道图像
                pil_images = [Image.fromarray(image) for image in images]  # 转换为PIL图像
    
            return pil_images  # 返回PIL图像列表
    
        @staticmethod  # 静态方法，不依赖于类的实例
        def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:  # 将PIL图像转换为numpy数组
            """
            Convert a PIL image or a list of PIL images to NumPy arrays.
            """
            if not isinstance(images, list):  # 如果输入不是列表
                images = [images]  # 将其转换为列表
            images = [np.array(image).astype(np.float32) / 255.0 for image in images]  # 转换为numpy数组并归一化
            images = np.stack(images, axis=0)  # 在新的轴上堆叠数组，形成四维数组
    
            return images  # 返回numpy数组
    # 将 NumPy 图像转换为 PyTorch 张量
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        # 文档字符串：将 NumPy 图像转换为 PyTorch 张量
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        # 检查图像的维度是否为 3（即 H x W x C）
        if images.ndim == 3:
            # 如果是 3 维，添加一个新的维度以适应模型输入
            images = images[..., None]
    
        # 将 NumPy 数组转置并转换为 PyTorch 张量
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        # 返回转换后的张量
        return images
    
    # 将 PyTorch 张量转换为 NumPy 图像
    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        # 文档字符串：将 PyTorch 张量转换为 NumPy 图像
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        # 将张量移至 CPU，并调整维度顺序为 H x W x C，转换为浮点型并转为 NumPy 数组
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回转换后的 NumPy 数组
        return images
    
    # 规范化图像数组到 [-1,1] 范围
    @staticmethod
    def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # 文档字符串：将图像数组规范化到 [-1,1] 范围
        """
        Normalize an image array to [-1,1].
        """
        # 将图像数组的值范围缩放到 [-1, 1]
        return 2.0 * images - 1.0
    
    # 将图像数组反规范化到 [0,1] 范围
    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # 文档字符串：将图像数组反规范化到 [0,1] 范围
        """
        Denormalize an image array to [0,1].
        """
        # 将图像数组的值范围调整到 [0, 1] 并限制在该范围内
        return (images / 2 + 0.5).clamp(0, 1)
    
    # 将 PIL 图像转换为 RGB 格式
    @staticmethod
    def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        # 文档字符串：将 PIL 图像转换为 RGB 格式
        """
        Converts a PIL image to RGB format.
        """
        # 使用 PIL 库将图像转换为 RGB 格式
        image = image.convert("RGB")
    
        # 返回转换后的图像
        return image
    
    # 将 PIL 图像转换为灰度格式
    @staticmethod
    def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:
        # 文档字符串：将 PIL 图像转换为灰度格式
        """
        Converts a PIL image to grayscale format.
        """
        # 使用 PIL 库将图像转换为灰度格式
        image = image.convert("L")
    
        # 返回转换后的图像
        return image
    
    # 对图像应用高斯模糊
    @staticmethod
    def blur(image: PIL.Image.Image, blur_factor: int = 4) -> PIL.Image.Image:
        # 文档字符串：对图像应用高斯模糊
        """
        Applies Gaussian blur to an image.
        """
        # 使用 PIL 库对图像应用高斯模糊
        image = image.filter(ImageFilter.GaussianBlur(blur_factor))
    
        # 返回模糊后的图像
        return image
    
    # 调整图像大小并填充
    @staticmethod
    def _resize_and_fill(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:  # 返回处理后的图像对象
        """  # 文档字符串，描述函数的作用
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, filling empty with data from image.  # 说明功能：调整图像大小并居中

        Args:  # 参数说明
            image: The image to resize.  # 待调整大小的图像
            width: The width to resize the image to.  # 目标宽度
            height: The height to resize the image to.  # 目标高度
        """  # 文档字符串结束

        ratio = width / height  # 计算目标宽高比
        src_ratio = image.width / image.height  # 计算源图像的宽高比

        src_w = width if ratio < src_ratio else image.width * height // image.height  # 计算源宽度
        src_h = height if ratio >= src_ratio else image.height * width // image.width  # 计算源高度

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION["lanczos"])  # 调整图像大小
        res = Image.new("RGB", (width, height))  # 创建新的 RGB 图像，尺寸为目标宽高
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))  # 将调整后的图像居中粘贴

        if ratio < src_ratio:  # 如果目标宽高比小于源宽高比
            fill_height = height // 2 - src_h // 2  # 计算需要填充的高度
            if fill_height > 0:  # 如果需要填充高度大于零
                res.paste(resized.resize((width, fill_height), box=(0, 0)), box=(0, 0))  # 填充上方空白
                res.paste(  # 填充下方空白
                    resized.resize((width, fill_height), box=(0, resized.height)), 
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:  # 如果目标宽高比大于源宽高比
            fill_width = width // 2 - src_w // 2  # 计算需要填充的宽度
            if fill_width > 0:  # 如果需要填充宽度大于零
                res.paste(resized.resize((fill_width, height), box=(0, 0)), box=(0, 0))  # 填充左侧空白
                res.paste(  # 填充右侧空白
                    resized.resize((fill_width, height), box=(resized.width, 0)), 
                    box=(fill_width + src_w, 0),
                )

        return res  # 返回最终调整后的图像

    def _resize_and_crop(  # 定义一个私有方法，用于调整大小并裁剪
        self,
        image: PIL.Image.Image,  # 输入的图像
        width: int,  # 目标宽度
        height: int,  # 目标高度
    ) -> PIL.Image.Image:  # 返回处理后的图像对象
        """  # 文档字符串，描述函数的作用
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, cropping the excess.  # 说明功能：调整大小并裁剪

        Args:  # 参数说明
            image: The image to resize.  # 待调整大小的图像
            width: The width to resize the image to.  # 目标宽度
            height: The height to resize the image to.  # 目标高度
        """  # 文档字符串结束

        ratio = width / height  # 计算目标宽高比
        src_ratio = image.width / image.height  # 计算源图像的宽高比

        src_w = width if ratio > src_ratio else image.width * height // image.height  # 计算源宽度
        src_h = height if ratio <= src_ratio else image.height * width // image.width  # 计算源高度

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION["lanczos"])  # 调整图像大小
        res = Image.new("RGB", (width, height))  # 创建新的 RGB 图像，尺寸为目标宽高
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))  # 将调整后的图像居中粘贴
        return res  # 返回最终调整后的图像

    def resize(  # 定义调整大小的公共方法
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],  # 输入的图像类型
        height: int,  # 目标高度
        width: int,  # 目标宽度
        resize_mode: str = "default",  # 指定调整大小模式，默认为 "default"
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        调整图像大小。

        参数:
            image (`PIL.Image.Image`, `np.ndarray` 或 `torch.Tensor`):
                输入图像，可以是 PIL 图像、numpy 数组或 pytorch 张量。
            height (`int`):
                要调整的高度。
            width (`int`):
                要调整的宽度。
            resize_mode (`str`, *可选*, 默认为 `default`):
                使用的调整模式，可以是 `default` 或 `fill`。如果是 `default`，将调整图像以适应
                指定的宽度和高度，可能不保持原始纵横比。如果是 `fill`，将调整图像以适应
                指定的宽度和高度，保持纵横比，然后将图像居中填充空白。如果是 `crop`，将调整
                图像以适应指定的宽度和高度，保持纵横比，然后将图像居中，裁剪多余部分。请注意，
                `fill` 和 `crop` 只支持 PIL 图像输入。

        返回:
            `PIL.Image.Image`, `np.ndarray` 或 `torch.Tensor`:
                调整后的图像。
        """
        # 检查调整模式是否有效，并确保图像为 PIL 图像
        if resize_mode != "default" and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Only PIL image input is supported for resize_mode {resize_mode}")
        # 如果输入是 PIL 图像
        if isinstance(image, PIL.Image.Image):
            # 如果调整模式是默认模式，调整图像大小
            if resize_mode == "default":
                image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
            # 如果调整模式是填充，调用填充函数
            elif resize_mode == "fill":
                image = self._resize_and_fill(image, width, height)
            # 如果调整模式是裁剪，调用裁剪函数
            elif resize_mode == "crop":
                image = self._resize_and_crop(image, width, height)
            # 如果调整模式不支持，抛出错误
            else:
                raise ValueError(f"resize_mode {resize_mode} is not supported")

        # 如果输入是 PyTorch 张量
        elif isinstance(image, torch.Tensor):
            # 使用插值调整张量大小
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
        # 如果输入是 numpy 数组
        elif isinstance(image, np.ndarray):
            # 将 numpy 数组转换为 PyTorch 张量
            image = self.numpy_to_pt(image)
            # 使用插值调整张量大小
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
            # 将张量转换回 numpy 数组
            image = self.pt_to_numpy(image)
        # 返回调整后的图像
        return image

    def binarize(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        创建掩膜。

        参数:
            image (`PIL.Image.Image`):
                输入图像，应该是 PIL 图像。

        返回:
            `PIL.Image.Image`:
                二值化图像。值小于 0.5 的设置为 0，值大于等于 0.5 的设置为 1。
        """
        # 将小于 0.5 的像素值设置为 0
        image[image < 0.5] = 0
        # 将大于等于 0.5 的像素值设置为 1
        image[image >= 0.5] = 1

        # 返回二值化后的图像
        return image
    # 定义一个方法，获取图像的默认高度和宽度
    def get_default_height_width(
            self,
            image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
            height: Optional[int] = None,
            width: Optional[int] = None,
        ) -> Tuple[int, int]:
            """
            该函数返回按 `vae_scale_factor` 下调到下一个整数倍的高度和宽度。
    
            参数：
                image(`PIL.Image.Image`, `np.ndarray` 或 `torch.Tensor`):
                    输入图像，可以是 PIL 图像、numpy 数组或 pytorch 张量。若为 numpy 数组，应该具有
                    形状 `[batch, height, width]` 或 `[batch, height, width, channel]`；若为 pytorch 张量，应该
                    具有形状 `[batch, channel, height, width]`。
                height (`int`, *可选*, 默认为 `None`):
                    预处理图像的高度。如果为 `None`，将使用输入图像的高度。
                width (`int`, *可选*, 默认为 `None`):
                    预处理的宽度。如果为 `None`，将使用输入图像的宽度。
            """
    
            # 如果高度为 None，尝试从图像中获取高度
            if height is None:
                # 如果图像是 PIL 图像，使用其高度
                if isinstance(image, PIL.Image.Image):
                    height = image.height
                # 如果图像是 pytorch 张量，使用其形状中的高度
                elif isinstance(image, torch.Tensor):
                    height = image.shape[2]
                # 否则，假设是 numpy 数组，使用其形状中的高度
                else:
                    height = image.shape[1]
    
            # 如果宽度为 None，尝试从图像中获取宽度
            if width is None:
                # 如果图像是 PIL 图像，使用其宽度
                if isinstance(image, PIL.Image.Image):
                    width = image.width
                # 如果图像是 pytorch 张量，使用其形状中的宽度
                elif isinstance(image, torch.Tensor):
                    width = image.shape[3]
                # 否则，假设是 numpy 数组，使用其形状中的宽度
                else:
                    width = image.shape[2]
    
            # 将宽度和高度调整为 vae_scale_factor 的整数倍
            width, height = (
                x - x % self.config.vae_scale_factor for x in (width, height)
            )  # 调整为 vae_scale_factor 的整数倍
    
            # 返回调整后的高度和宽度
            return height, width
    
    # 定义一个方法，预处理图像
    def preprocess(
            self,
            image: PipelineImageInput,
            height: Optional[int] = None,
            width: Optional[int] = None,
            resize_mode: str = "default",  # "default", "fill", "crop"
            crops_coords: Optional[Tuple[int, int, int, int]] = None,
    # 定义一个方法，后处理图像
    def postprocess(
            self,
            image: torch.Tensor,
            output_type: str = "pil",
            do_denormalize: Optional[List[bool]] = None,
    # 返回处理后的图像，类型为 PIL.Image.Image、np.ndarray 或 torch.Tensor
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        处理从张量输出的图像，转换为 `output_type`。
    
        参数:
            image (`torch.Tensor`):
                输入的图像，应该是形状为 `B x C x H x W` 的 pytorch 张量。
            output_type (`str`, *可选*, 默认为 `pil`):
                图像的输出类型，可以是 `pil`、`np`、`pt`、`latent` 之一。
            do_denormalize (`List[bool]`, *可选*, 默认为 `None`):
                是否将图像反归一化到 [0,1]。如果为 `None`，将使用 `VaeImageProcessor` 配置中的 `do_normalize` 值。
    
        返回:
            `PIL.Image.Image`、`np.ndarray` 或 `torch.Tensor`:
                处理后的图像。
        """
        # 检查输入的图像是否为 pytorch 张量
        if not isinstance(image, torch.Tensor):
            # 抛出值错误，如果输入格式不正确
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        # 检查输出类型是否在支持的列表中
        if output_type not in ["latent", "pt", "np", "pil"]:
            # 创建弃用信息，说明当前输出类型已过时
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            # 调用弃用函数，记录警告信息
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            # 将输出类型设置为默认值
            output_type = "np"
    
        # 如果输出类型为 "latent"，直接返回输入图像
        if output_type == "latent":
            return image
    
        # 如果 do_denormalize 为 None，则根据配置设置其值
        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]
    
        # 通过 denormalize 方法处理图像，生成新张量
        image = torch.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )
    
        # 如果输出类型为 "pt"，返回处理后的图像张量
        if output_type == "pt":
            return image
    
        # 将处理后的图像从 pytorch 张量转换为 numpy 数组
        image = self.pt_to_numpy(image)
    
        # 如果输出类型为 "np"，返回 numpy 数组
        if output_type == "np":
            return image
    
        # 如果输出类型为 "pil"，将 numpy 数组转换为 PIL 图像并返回
        if output_type == "pil":
            return self.numpy_to_pil(image)
    
    # 定义应用遮罩的函数，接受多个参数
    def apply_overlay(
        self,
        mask: PIL.Image.Image,
        init_image: PIL.Image.Image,
        image: PIL.Image.Image,
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> PIL.Image.Image:
        """
        将修复输出叠加到原始图像上
        """

        # 获取原始图像的宽度和高度
        width, height = image.width, image.height

        # 调整初始图像和掩膜图像到与原始图像相同的大小
        init_image = self.resize(init_image, width=width, height=height)
        mask = self.resize(mask, width=width, height=height)

        # 创建一个新的 RGBA 图像，用于存放初始图像的掩膜效果
        init_image_masked = PIL.Image.new("RGBa", (width, height))
        # 将初始图像按掩膜方式粘贴到新的图像上，掩膜为掩码的反转图像
        init_image_masked.paste(init_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L")))
        # 将初始图像掩膜转换为 RGBA 格式
        init_image_masked = init_image_masked.convert("RGBA")

        # 如果给定了裁剪坐标
        if crop_coords is not None:
            # 解包裁剪坐标
            x, y, x2, y2 = crop_coords
            # 计算裁剪区域的宽度和高度
            w = x2 - x
            h = y2 - y
            # 创建一个新的 RGBA 图像作为基础图像
            base_image = PIL.Image.new("RGBA", (width, height))
            # 将原始图像调整到裁剪区域的大小
            image = self.resize(image, height=h, width=w, resize_mode="crop")
            # 将调整后的图像粘贴到基础图像的指定位置
            base_image.paste(image, (x, y))
            # 将基础图像转换为 RGB 格式
            image = base_image.convert("RGB")

        # 将图像转换为 RGBA 格式
        image = image.convert("RGBA")
        # 将初始图像的掩膜叠加到当前图像上
        image.alpha_composite(init_image_masked)
        # 将结果图像转换为 RGB 格式
        image = image.convert("RGB")

        # 返回最终的图像
        return image
# 定义 VAE LDM3D 图像处理器类，继承自 VaeImageProcessor
class VaeImageProcessorLDM3D(VaeImageProcessor):
    """
    VAE LDM3D 的图像处理器。

    参数:
        do_resize (`bool`, *可选*, 默认值为 `True`):
            是否将图像的（高度，宽度）尺寸缩小到 `vae_scale_factor` 的倍数。
        vae_scale_factor (`int`, *可选*, 默认值为 `8`):
            VAE 缩放因子。如果 `do_resize` 为 `True`，图像会自动调整为该因子的倍数。
        resample (`str`, *可选*, 默认值为 `lanczos`):
            在调整图像大小时使用的重采样滤波器。
        do_normalize (`bool`, *可选*, 默认值为 `True`):
            是否将图像归一化到 [-1,1] 范围内。
    """

    # 配置名称常量
    config_name = CONFIG_NAME

    # 注册到配置中的初始化方法
    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,        # 是否调整大小，默认为 True
        vae_scale_factor: int = 8,     # VAE 缩放因子，默认为 8
        resample: str = "lanczos",     # 重采样方法，默认为 lanczos
        do_normalize: bool = True,     # 是否归一化，默认为 True
    ):
        # 调用父类的初始化方法
        super().__init__()

    # 静态方法：将 NumPy 图像或图像批次转换为 PIL 图像
    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
        """
        将 NumPy 图像或图像批次转换为 PIL 图像。
        """
        # 如果输入是 3 维数组，添加一个新的维度
        if images.ndim == 3:
            images = images[None, ...]
        # 将图像数据放大到 255，四舍五入并转换为无符号 8 位整数
        images = (images * 255).round().astype("uint8")
        # 检查最后一个维度是否为 1（灰度图像）
        if images.shape[-1] == 1:
            # 特殊情况处理灰度（单通道）图像
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            # 处理 RGB 图像（提取前三个通道）
            pil_images = [Image.fromarray(image[:, :, :3]) for image in images]

        # 返回 PIL 图像列表
        return pil_images

    # 静态方法：将 PIL 图像或图像列表转换为 NumPy 数组
    @staticmethod
    def depth_pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        """
        将 PIL 图像或图像列表转换为 NumPy 数组。
        """
        # 如果输入不是列表，将其转换为单元素列表
        if not isinstance(images, list):
            images = [images]

        # 将每个 PIL 图像转换为 NumPy 数组，并归一化到 [0, 1] 范围
        images = [np.array(image).astype(np.float32) / (2**16 - 1) for image in images]
        # 将图像堆叠成一个 4D 数组
        images = np.stack(images, axis=0)
        # 返回 NumPy 数组
        return images

    # 静态方法：将 RGB 深度图像转换为深度图
    @staticmethod
    def rgblike_to_depthmap(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        参数:
            image: RGB 类似深度图像

        返回: 深度图
        """
        # 提取深度图，使用红色通道和蓝色通道计算深度值
        return image[:, :, 1] * 2**8 + image[:, :, 2]
    # 将 NumPy 深度图像或图像批处理转换为 PIL 图像
    def numpy_to_depth(self, images: np.ndarray) -> List[PIL.Image.Image]:
        # 文档字符串，说明函数的作用
        """
        Convert a NumPy depth image or a batch of images to a PIL image.
        """
        # 检查输入图像的维度是否为 3，如果是，则在前面添加一个维度
        if images.ndim == 3:
            images = images[None, ...]
        # 从输入图像中提取深度信息，假设深度数据在最后的几维
        images_depth = images[:, :, :, 3:]
        # 检查最后一个维度是否为 6，表示有额外的信息
        if images.shape[-1] == 6:
            # 将深度值范围缩放到 0-255，并转换为无符号 8 位整数
            images_depth = (images_depth * 255).round().astype("uint8")
            # 将每个深度图像转换为 PIL 图像，使用特定模式
            pil_images = [
                Image.fromarray(self.rgblike_to_depthmap(image_depth), mode="I;16") for image_depth in images_depth
            ]
        # 检查最后一个维度是否为 4，表示仅有深度数据
        elif images.shape[-1] == 4:
            # 将深度值范围缩放到 0-65535，并转换为无符号 16 位整数
            images_depth = (images_depth * 65535.0).astype(np.uint16)
            # 将每个深度图像转换为 PIL 图像，使用特定模式
            pil_images = [Image.fromarray(image_depth, mode="I;16") for image_depth in images_depth]
        # 如果输入的形状不符合要求，抛出异常
        else:
            raise Exception("Not supported")
    
        # 返回生成的 PIL 图像列表
        return pil_images
    
    # 处理图像的后处理函数，接受图像和输出类型等参数
    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        处理图像输出，将张量转换为 `output_type` 格式。

        参数：
            image (`torch.Tensor`):
                输入的图像，应该是形状为 `B x C x H x W` 的 PyTorch 张量。
            output_type (`str`, *可选*, 默认为 `pil`):
                图像的输出类型，可以是 `pil`、`np`、`pt` 或 `latent` 之一。
            do_denormalize (`List[bool]`, *可选*, 默认为 `None`):
                是否将图像反归一化到 [0,1]。如果为 `None`，将使用 `VaeImageProcessor` 配置中的 `do_normalize` 值。

        返回：
            `PIL.Image.Image`、`np.ndarray` 或 `torch.Tensor`:
                处理后的图像。
        """
        # 检查输入图像是否为 PyTorch 张量，如果不是，则抛出错误
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        # 检查输出类型是否在支持的选项中，如果不在，发送弃用警告并设置为默认值 `np`
        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"  # 设置输出类型为默认的 `np`

        # 如果反归一化标志为 None，则根据配置初始化为与图像批大小相同的列表
        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        # 对每个图像进行反归一化处理，构建处理后的图像堆叠
        image = torch.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )

        # 将处理后的图像从 PyTorch 张量转换为 NumPy 数组
        image = self.pt_to_numpy(image)

        # 根据输出类型返回相应的处理结果
        if output_type == "np":
            # 如果图像的最后一个维度为 6，则提取深度图
            if image.shape[-1] == 6:
                image_depth = np.stack([self.rgblike_to_depthmap(im[:, :, 3:]) for im in image], axis=0)
            else:
                # 否则直接提取最后三个通道作为深度图
                image_depth = image[:, :, :, 3:]
            return image[:, :, :, :3], image_depth  # 返回 RGB 图像和深度图

        if output_type == "pil":
            # 将 NumPy 数组转换为 PIL 图像并返回
            return self.numpy_to_pil(image), self.numpy_to_depth(image)
        else:
            # 如果输出类型不被支持，抛出异常
            raise Exception(f"This type {output_type} is not supported")

    def preprocess(
        self,
        rgb: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
        depth: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
        target_res: Optional[int] = None,
# 定义一个处理 IP 适配器图像掩码的图像处理器类
class IPAdapterMaskProcessor(VaeImageProcessor):
    """
    IP适配器图像掩码的图像处理器。

    参数：
        do_resize (`bool`, *可选*, 默认为 `True`):
            是否将图像的高度和宽度缩小为 `vae_scale_factor` 的倍数。
        vae_scale_factor (`int`, *可选*, 默认为 `8`):
            VAE缩放因子。如果 `do_resize` 为 `True`，图像将自动调整为该因子的倍数。
        resample (`str`, *可选*, 默认为 `lanczos`):
            调整图像大小时使用的重采样滤波器。
        do_normalize (`bool`, *可选*, 默认为 `False`):
            是否将图像标准化到 [-1,1]。
        do_binarize (`bool`, *可选*, 默认为 `True`):
            是否将图像二值化为 0/1。
        do_convert_grayscale (`bool`, *可选*, 默认为 `True`):
            是否将图像转换为灰度格式。

    """

    # 配置名称常量
    config_name = CONFIG_NAME

    @register_to_config
    # 初始化函数，设置处理器的参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否缩放图像
        vae_scale_factor: int = 8,  # VAE缩放因子
        resample: str = "lanczos",  # 重采样滤波器
        do_normalize: bool = False,  # 是否标准化图像
        do_binarize: bool = True,  # 是否二值化图像
        do_convert_grayscale: bool = True,  # 是否转换为灰度图像
    ):
        # 调用父类的初始化方法，传递参数
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

    @staticmethod
    # 定义 downsample 函数，输入为掩码张量和其他参数，输出为下采样后的掩码张量
        def downsample(mask: torch.Tensor, batch_size: int, num_queries: int, value_embed_dim: int):
            """
            将提供的掩码张量下采样到与缩放点积注意力预期的维度匹配。如果掩码的长宽比与输出图像的长宽比不匹配，则发出警告。
    
            参数:
                mask (`torch.Tensor`):
                    由 `IPAdapterMaskProcessor.preprocess()` 生成的输入掩码张量。
                batch_size (`int`):
                    批处理大小。
                num_queries (`int`):
                    查询的数量。
                value_embed_dim (`int`):
                    值嵌入的维度。
    
            返回:
                `torch.Tensor`:
                    下采样后的掩码张量。
    
            """
            # 获取掩码的高度和宽度
            o_h = mask.shape[1]
            o_w = mask.shape[2]
            # 计算掩码的长宽比
            ratio = o_w / o_h
            # 计算下采样后掩码的高度
            mask_h = int(math.sqrt(num_queries / ratio))
            # 根据掩码高度调整，确保可以容纳所有查询
            mask_h = int(mask_h) + int((num_queries % int(mask_h)) != 0)
            # 计算下采样后掩码的宽度
            mask_w = num_queries // mask_h
    
            # 对掩码进行插值下采样
            mask_downsample = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w), mode="bicubic").squeeze(0)
    
            # 重复掩码以匹配批处理大小
            if mask_downsample.shape[0] < batch_size:
                mask_downsample = mask_downsample.repeat(batch_size, 1, 1)
    
            # 调整掩码形状为 (batch_size, -1)
            mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1)
    
            # 计算下采样后的区域大小
            downsampled_area = mask_h * mask_w
            # 如果输出图像和掩码的长宽比不相同，发出警告并填充张量
            if downsampled_area < num_queries:
                warnings.warn(
                    "掩码的长宽比与输出图像的长宽比不匹配。"
                    "请更新掩码或调整输出大小以获得最佳性能。",
                    UserWarning,
                )
                mask_downsample = F.pad(mask_downsample, (0, num_queries - mask_downsample.shape[1]), value=0.0)
            # 如果下采样后的掩码形状大于查询数量，则截断最后的嵌入
            if downsampled_area > num_queries:
                warnings.warn(
                    "掩码的长宽比与输出图像的长宽比不匹配。"
                    "请更新掩码或调整输出大小以获得最佳性能。",
                    UserWarning,
                )
                mask_downsample = mask_downsample[:, :num_queries]
    
            # 重复最后一个维度以匹配 SDPA 输出形状
            mask_downsample = mask_downsample.view(mask_downsample.shape[0], mask_downsample.shape[1], 1).repeat(
                1, 1, value_embed_dim
            )
    
            # 返回下采样后的掩码
            return mask_downsample
# PixArt 图像处理器类，继承自 VaeImageProcessor
class PixArtImageProcessor(VaeImageProcessor):
    """
    PixArt 图像的调整大小和裁剪处理器。

    参数：
        do_resize (`bool`, *可选*, 默认为 `True`):
            是否将图像的（高度，宽度）尺寸缩小为 `vae_scale_factor` 的倍数。可以接受
            来自 [`image_processor.VaeImageProcessor.preprocess`] 方法的 `height` 和 `width` 参数。
        vae_scale_factor (`int`, *可选*, 默认为 `8`):
            VAE 缩放因子。如果 `do_resize` 为 `True`，图像会自动调整为该因子的倍数。
        resample (`str`, *可选*, 默认为 `lanczos`):
            调整图像大小时使用的重采样滤镜。
        do_normalize (`bool`, *可选*, 默认为 `True`):
            是否将图像标准化到 [-1,1]。
        do_binarize (`bool`, *可选*, 默认为 `False`):
            是否将图像二值化为 0/1。
        do_convert_rgb (`bool`, *可选*, 默认为 `False`):
            是否将图像转换为 RGB 格式。
        do_convert_grayscale (`bool`, *可选*, 默认为 `False`):
            是否将图像转换为灰度格式。
    """

    # 注册到配置中的初始化方法
    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,  # 是否调整大小
        vae_scale_factor: int = 8,  # VAE 缩放因子
        resample: str = "lanczos",  # 重采样滤镜
        do_normalize: bool = True,  # 是否标准化
        do_binarize: bool = False,  # 是否二值化
        do_convert_grayscale: bool = False,  # 是否转换为灰度
    ):
        # 调用父类初始化方法，传递参数
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

    # 静态方法，分类高度和宽度到最近的比例
    @staticmethod
    def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
        """返回分箱的高度和宽度。"""
        ar = float(height / width)  # 计算高度与宽度的比率
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))  # 找到最接近的比率
        default_hw = ratios[closest_ratio]  # 获取该比率对应的默认高度和宽度
        return int(default_hw[0]), int(default_hw[1])  # 返回整数形式的高度和宽度

    @staticmethod
    # 定义一个函数，调整张量的大小并裁剪到指定的宽度和高度
    def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int) -> torch.Tensor:
        # 获取原始张量的高度和宽度
        orig_height, orig_width = samples.shape[2], samples.shape[3]
    
        # 检查是否需要调整大小
        if orig_height != new_height or orig_width != new_width:
            # 计算调整大小的比例
            ratio = max(new_height / orig_height, new_width / orig_width)
            # 计算调整后的宽度和高度
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)
    
            # 调整大小
            samples = F.interpolate(
                samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )
    
            # 计算中心裁剪的起始和结束坐标
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            # 裁剪样本到目标大小
            samples = samples[:, :, start_y:end_y, start_x:end_x]
    
        # 返回调整和裁剪后的张量
        return samples
```