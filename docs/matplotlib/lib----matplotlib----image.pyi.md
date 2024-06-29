# `D:\src\scipysrc\matplotlib\lib\matplotlib\image.pyi`

```
from collections.abc import Callable, Sequence  # 导入 Callable 和 Sequence 类型定义
import os  # 导入操作系统相关功能
import pathlib  # 导入处理路径的模块
from typing import Any, BinaryIO, Literal  # 导入类型定义

import numpy as np  # 导入 NumPy 库
from numpy.typing import ArrayLike, NDArray  # 导入 NumPy 相关类型定义
import PIL.Image  # 导入 PIL 库中的 Image 模块

import matplotlib.artist as martist  # 导入 Matplotlib 中的 artist 模块
from matplotlib.axes import Axes  # 导入 Matplotlib 中的 Axes 类
from matplotlib import cm  # 导入 Matplotlib 中的 colormap 模块
from matplotlib.backend_bases import RendererBase, MouseEvent  # 导入 Matplotlib 中的 RendererBase 和 MouseEvent 类
from matplotlib.colors import Colormap, Normalize  # 导入 Matplotlib 中的 Colormap 和 Normalize 类
from matplotlib.figure import Figure  # 导入 Matplotlib 中的 Figure 类
from matplotlib.transforms import Affine2D, BboxBase, Bbox, Transform  # 导入 Matplotlib 中的变换类

#
# These names are re-exported from matplotlib._image.
#

BESSEL: int  # 定义 BESSEL 变量，类型为整数
BICUBIC: int  # 定义 BICUBIC 变量，类型为整数
BILINEAR: int  # 定义 BILINEAR 变量，类型为整数
BLACKMAN: int  # 定义 BLACKMAN 变量，类型为整数
CATROM: int  # 定义 CATROM 变量，类型为整数
GAUSSIAN: int  # 定义 GAUSSIAN 变量，类型为整数
HAMMING: int  # 定义 HAMMING 变量，类型为整数
HANNING: int  # 定义 HANNING 变量，类型为整数
HERMITE: int  # 定义 HERMITE 变量，类型为整数
KAISER: int  # 定义 KAISER 变量，类型为整数
LANCZOS: int  # 定义 LANCZOS 变量，类型为整数
MITCHELL: int  # 定义 MITCHELL 变量，类型为整数
NEAREST: int  # 定义 NEAREST 变量，类型为整数
QUADRIC: int  # 定义 QUADRIC 变量，类型为整数
SINC: int  # 定义 SINC 变量，类型为整数
SPLINE16: int  # 定义 SPLINE16 变量，类型为整数
SPLINE36: int  # 定义 SPLINE36 变量，类型为整数

def resample(
    input_array: NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int8],  # 输入数组，可以是 float32、float64 或 int8 类型的 NumPy 数组
    output_array: NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int8],  # 输出数组，可以是 float32、float64 或 int8 类型的 NumPy 数组
    transform: Transform,  # 变换对象，类型为 Transform
    interpolation: int = ...,  # 插值方法，整数类型，默认为省略值
    resample: bool = ...,  # 是否重新采样，布尔类型，默认为省略值
    alpha: float = ...,  # 透明度，浮点数类型，默认为省略值
    norm: bool = ...,  # 是否归一化，布尔类型，默认为省略值
    radius: float = ...,  # 半径，浮点数类型，默认为省略值
) -> None: ...  # 函数返回 None，用省略号表示函数体未完整定义

#
# END names re-exported from matplotlib._image.
#

interpolations_names: set[str]  # 插值方法名称集合，包含字符串类型的元素

def composite_images(
    images: Sequence[_ImageBase],  # 图像序列，包含 _ImageBase 类型的元素
    renderer: RendererBase,  # 渲染器，类型为 RendererBase
    magnification: float = ...  # 放大倍数，浮点数类型，默认为省略值
) -> tuple[np.ndarray, float, float]: ...  # 函数返回一个元组，包含 NumPy 数组、两个浮点数作为元素

class _ImageBase(martist.Artist, cm.ScalarMappable):  # _ImageBase 类继承自 martist.Artist 和 cm.ScalarMappable
    zorder: float  # zorder 属性，浮点数类型
    origin: Literal["upper", "lower"]  # origin 属性，值为 "upper" 或 "lower"
    axes: Axes  # axes 属性，类型为 Axes

    def __init__(
        self,
        ax: Axes,  # 所属 Axes 对象，类型为 Axes
        cmap: str | Colormap | None = ...,  # 颜色映射名称、Colormap 对象或 None，默认为省略值
        norm: str | Normalize | None = ...,  # 归一化名称、Normalize 对象或 None，默认为省略值
        interpolation: str | None = ...,  # 插值方法名称或 None，默认为省略值
        origin: Literal["upper", "lower"] | None = ...,  # 坐标原点位置，值为 "upper" 或 "lower" 或 None，默认为省略值
        filternorm: bool = ...,  # 是否过滤归一化，布尔类型，默认为省略值
        filterrad: float = ...,  # 过滤器半径，浮点数类型，默认为省略值
        resample: bool | None = ...,  # 是否重新采样，布尔类型或 None，默认为省略值
        *,
        interpolation_stage: Literal["data", "rgba"] | None = ...,  # 插值阶段，值为 "data" 或 "rgba" 或 None，默认为省略值
        **kwargs
    ) -> None: ...  # 初始化方法返回 None，函数体未完整定义

    def get_size(self) -> tuple[int, int]: ...  # 获取图像大小的方法，返回整数元组
    def set_alpha(self, alpha: float | ArrayLike | None) -> None: ...  # 设置图像透明度的方法，参数可以是浮点数、数组或 None，返回 None
    def changed(self) -> None: ...  # 表示图像变化的方法，返回 None
    def make_image(
        self, renderer: RendererBase, magnification: float = ..., unsampled: bool = ...
    ) -> tuple[np.ndarray, float, float, Affine2D]: ...  # 生成图像的方法，返回元组，包含 NumPy 数组、两个浮点数和 Affine2D 对象
    def draw(self, renderer: RendererBase) -> None: ...  # 绘制图像的方法，参数为 RendererBase 类型，返回 None
    def write_png(self, fname: str | pathlib.Path | BinaryIO) -> None: ...  # 将图像保存为 PNG 格式的方法，参数可以是文件名字符串、路径对象或二进制流，返回 None
    def set_data(self, A: ArrayLike | None) -> None: ...  # 设置图像数据的方法，参数可以是数组或 None，返回 None
    def set_array(self, A: ArrayLike | None) -> None: ...  # 设置图像数组的方法，参数可以是数组或 None，返回 None
    def get_shape(self) -> tuple[int, int, int]: ...  # 获取图像形状的方法，返回整数元组
    def get_interpolation(self) -> str: ...  # 获取插值方法的方法，返回字符串
    def set_interpolation(self, s: str | None) -> None: ...  # 设置插值方法的方法，参数可以是字符串或 None，返回 None
    def get_interpolation_stage(self) -> Literal["data", "rgba"]: ...  # 获取插值阶段的方法，返回 "data" 或 "rgba"
    def set_interpolation_stage(self, s: Literal["data", "rgba"]) -> None: ...  # 设置插值阶段的方法，参数可以是 "data" 或 "rgba"，返回 None
    def can_composite(self) -> bool: ...  # 判断是否可以复合的方法，返回布尔值
    def set_resample(self, v: bool | None) -> None: ...  # 设置重新采样的方法，参数
    # 设置滤波器归一化选项
    def set_filternorm(self, filternorm: bool) -> None:
        ...
    
    # 获取当前滤波器归一化选项
    def get_filternorm(self) -> bool:
        ...
    
    # 设置滤波器半径
    def set_filterrad(self, filterrad: float) -> None:
        ...
    
    # 获取当前滤波器半径
    def get_filterrad(self) -> float:
        ...
class AxesImage(_ImageBase):
    # AxesImage 类继承自 _ImageBase 类

    def __init__(
        self,
        ax: Axes,
        *,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        interpolation: str | None = ...,
        origin: Literal["upper", "lower"] | None = ...,
        extent: tuple[float, float, float, float] | None = ...,
        filternorm: bool = ...,
        filterrad: float = ...,
        resample: bool = ...,
        interpolation_stage: Literal["data", "rgba"] | None = ...,
        **kwargs
    ) -> None:
        ...

    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox:
        ...

    def make_image(
        self, renderer: RendererBase, magnification: float = ..., unsampled: bool = ...
    ) -> tuple[np.ndarray, float, float, Affine2D]:
        ...

    def set_extent(
        self, extent: tuple[float, float, float, float], **kwargs
    ) -> None:
        ...

    def get_extent(self) -> tuple[float, float, float, float]:
        ...

    def get_cursor_data(self, event: MouseEvent) -> None | float:
        ...

class NonUniformImage(AxesImage):
    # NonUniformImage 类继承自 AxesImage 类
    mouseover: bool

    def __init__(
        self, ax: Axes, *, interpolation: Literal["nearest", "bilinear"] = ..., **kwargs
    ) -> None:
        ...

    def set_data(self, x: ArrayLike, y: ArrayLike, A: ArrayLike) -> None:
        ...  # type: ignore[override]
        # 设置图像的数据，覆盖父类的设置数据方法，忽略类型检查

    def set_interpolation(self, s: Literal["nearest", "bilinear"]) -> None:
        ...  # type: ignore[override]
        # 设置插值方法，覆盖父类的设置插值方法，忽略类型检查

class PcolorImage(AxesImage):
    # PcolorImage 类继承自 AxesImage 类

    def __init__(
        self,
        ax: Axes,
        x: ArrayLike | None = ...,
        y: ArrayLike | None = ...,
        A: ArrayLike | None = ...,
        *,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        **kwargs
    ) -> None:
        ...

    def set_data(self, x: ArrayLike, y: ArrayLike, A: ArrayLike) -> None:
        ...  # type: ignore[override]
        # 设置图像的数据，覆盖父类的设置数据方法，忽略类型检查

class FigureImage(_ImageBase):
    # FigureImage 类继承自 _ImageBase 类
    zorder: float
    figure: Figure
    ox: float
    oy: float
    magnification: float

    def __init__(
        self,
        fig: Figure,
        *,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        offsetx: int = ...,
        offsety: int = ...,
        origin: Literal["upper", "lower"] | None = ...,
        **kwargs
    ) -> None:
        ...

    def get_extent(self) -> tuple[float, float, float, float]:
        ...

class BboxImage(_ImageBase):
    # BboxImage 类继承自 _ImageBase 类
    bbox: BboxBase

    def __init__(
        self,
        bbox: BboxBase | Callable[[RendererBase | None], Bbox],
        *,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        interpolation: str | None = ...,
        origin: Literal["upper", "lower"] | None = ...,
        filternorm: bool = ...,
        filterrad: float = ...,
        resample: bool = ...,
        **kwargs
    ) -> None:
        ...

    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox:
        ...
    fname: str | pathlib.Path | BinaryIO, format: str | None = ...


# fname 参数可以是 str（字符串）、pathlib.Path（路径对象）、BinaryIO（二进制输入输出对象）类型的值
# format 参数可以是字符串类型或者 None，默认为 None
# 保存数组为图像文件
def imsave(
    fname: str | os.PathLike | BinaryIO,  # 文件名或文件对象，支持路径或二进制IO对象
    arr: ArrayLike,  # 待保存的图像数据数组
    vmin: float | None = ...,  # 数据值的最小值，可选，默认为None
    vmax: float | None = ...,  # 数据值的最大值，可选，默认为None
    cmap: str | Colormap | None = ...,  # 颜色映射名称、Colormap对象或None
    format: str | None = ...,  # 图像文件格式，可选，默认为None
    origin: Literal["upper", "lower"] | None = ...,  # 坐标原点位置，可选，"upper"或"lower"或None
    dpi: float = ...,  # 图像分辨率，每英寸点数，默认为...
    *,
    metadata: dict[str, str] | None = ...,  # 图像的元数据，字典格式或None
    pil_kwargs: dict[str, Any] | None = ...  # 传递给PIL保存的额外关键字参数，字典格式或None
) -> None: ...
# 将PIL.Image.Image对象转换为NumPy数组
def pil_to_array(pilImage: PIL.Image.Image) -> np.ndarray: ...
# 创建缩略图并返回Figure对象
def thumbnail(
    infile: str | BinaryIO,  # 输入文件名或文件对象，支持路径或二进制IO对象
    thumbfile: str | BinaryIO,  # 缩略图文件名或文件对象，支持路径或二进制IO对象
    scale: float = ...,  # 缩放比例，默认为...
    interpolation: str = ...,  # 插值方法，用于缩放图像，默认为...
    preview: bool = ...  # 是否生成预览，布尔值，默认为...
) -> Figure: ...
```