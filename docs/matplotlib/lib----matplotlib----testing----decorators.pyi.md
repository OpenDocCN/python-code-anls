# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\decorators.pyi`

```
# 从 collections.abc 模块导入 Callable 和 Sequence 类型
from collections.abc import Callable, Sequence
# 从 pathlib 模块导入 Path 类型
from pathlib import Path
# 从 typing 模块导入 Any 和 TypeVar 类型
from typing import Any, TypeVar
# 从 typing_extensions 模块导入 ParamSpec 类型
from typing_extensions import ParamSpec

# 从 matplotlib.figure 模块导入 Figure 类型
from matplotlib.figure import Figure
# 从 matplotlib.typing 模块导入 RcStyleType 类型
from matplotlib.typing import RcStyleType

# 定义一个不进行任何操作的函数 remove_ticks_and_titles，接受 Figure 对象作为参数，返回 None
def remove_ticks_and_titles(figure: Figure) -> None: ...

# 定义一个装饰器函数 image_comparison，用于比较图像
def image_comparison(
    # 基准图像的文件名列表或者 None
    baseline_images: list[str] | None,
    # 图像文件的扩展名列表或者 None
    extensions: list[str] | None = ...,
    # 允许的像素值容差
    tol: float = ...,
    # freetype 库的版本号或者 None
    freetype_version: tuple[str, str] | str | None = ...,
    # 是否移除图像中的文本
    remove_text: bool = ...,
    # 保存图像时的额外参数字典或者 None
    savefig_kwarg: dict[str, Any] | None = ...,
    # 图像样式配置
    style: RcStyleType = ...,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # 返回一个装饰器函数，接受一个类型为 Callable[_P, _R] 的函数作为参数，并返回类型为 Callable[_P, _R] 的函数
    ...

# 定义一个装饰器函数 check_figures_equal，用于检查两个图像是否相等
def check_figures_equal(
    # 图像文件的扩展名序列
    *, extensions: Sequence[str] = ...,
    # 允许的像素值容差
    tol: float = ...
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # 返回一个装饰器函数，接受一个类型为 Callable[_P, _R] 的函数作为参数，并返回类型为 Callable[_P, _R] 的函数
    ...

# 定义一个内部函数 _image_directories，接受一个 Callable 类型的参数 func，并返回一个由 Path 对象组成的元组
def _image_directories(func: Callable) -> tuple[Path, Path]:
    ...
```