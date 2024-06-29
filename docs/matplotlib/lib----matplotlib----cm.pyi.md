# `D:\src\scipysrc\matplotlib\lib\matplotlib\cm.pyi`

```
from collections.abc import Iterator, Mapping
from matplotlib import cbook, colors
from matplotlib.colorbar import Colorbar

import numpy as np
from numpy.typing import ArrayLike

# 定义一个 ColormapRegistry 类，用于管理颜色映射对象的注册和获取
class ColormapRegistry(Mapping[str, colors.Colormap]):
    def __init__(self, cmaps: Mapping[str, colors.Colormap]) -> None: ...
    def __getitem__(self, item: str) -> colors.Colormap: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __call__(self) -> list[str]: ...
    def register(
        self, cmap: colors.Colormap, *, name: str | None = ..., force: bool = ...
    ) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get_cmap(self, cmap: str | colors.Colormap) -> colors.Colormap: ...

# 创建一个 ColormapRegistry 实例 _colormaps
_colormaps: ColormapRegistry = ...

# 定义一个函数 get_cmap，用于获取指定名称或对象的颜色映射
def get_cmap(name: str | colors.Colormap | None = ..., lut: int | None = ...) -> colors.Colormap: ...

# 定义一个 ScalarMappable 类，用于将标量映射到颜色映射对象
class ScalarMappable:
    cmap: colors.Colormap | None  # 当前对象的颜色映射，可能为 None
    colorbar: Colorbar | None  # 当前对象的颜色条，可能为 None
    callbacks: cbook.CallbackRegistry  # 回调函数注册表

    def __init__(
        self,
        norm: colors.Normalize | None = ...,
        cmap: str | colors.Colormap | None = ...,
    ) -> None: ...
    
    # 将数据数组 x 转换为 RGBA 数组
    def to_rgba(
        self,
        x: np.ndarray,
        alpha: float | ArrayLike | None = ...,
        bytes: bool = ...,
        norm: bool = ...,
    ) -> np.ndarray: ...

    # 设置当前对象的数据数组
    def set_array(self, A: ArrayLike | None) -> None: ...

    # 获取当前对象的数据数组
    def get_array(self) -> np.ndarray | None: ...

    # 获取当前对象的颜色映射
    def get_cmap(self) -> colors.Colormap: ...

    # 获取当前对象的数据范围
    def get_clim(self) -> tuple[float, float]: ...

    # 设置当前对象的数据范围
    def set_clim(self, vmin: float | tuple[float, float] | None = ..., vmax: float | None = ...) -> None: ...

    # 获取当前对象的透明度
    def get_alpha(self) -> float | None: ...

    # 设置当前对象的颜色映射
    def set_cmap(self, cmap: str | colors.Colormap) -> None: ...

    @property
    # 获取当前对象的归一化对象
    def norm(self) -> colors.Normalize: ...

    @norm.setter
    # 设置当前对象的归一化对象
    def norm(self, norm: colors.Normalize | str | None) -> None: ...

    # 设置当前对象的归一化对象
    def set_norm(self, norm: colors.Normalize | str | None) -> None: ...

    # 自动调整当前对象的数据范围
    def autoscale(self) -> None: ...

    # 自动调整当前对象的数据范围（如果为 None）
    def autoscale_None(self) -> None: ...

    # 标记当前对象已更改
    def changed(self) -> None: ...
```