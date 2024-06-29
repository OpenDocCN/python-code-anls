# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\compare.pyi`

```
# 导入需要的模块和类型
from collections.abc import Callable
from typing import Literal, overload
from numpy.typing import NDArray

# 定义对外暴露的接口
__all__ = ["calculate_rms", "comparable_formats", "compare_images"]

# 声明未实现的函数签名，返回文件名字符串
def make_test_filename(fname: str, purpose: str) -> str: ...

# 返回缓存目录路径字符串
def get_cache_dir() -> str: ...

# 计算文件哈希值，返回哈希字符串
def get_file_hash(path: str, block_size: int = ...) -> str: ...

# 声明一个空字典，键为字符串，值为接受两个字符串参数并返回空的可调用对象
converter: dict[str, Callable[[str, str], None]] = {}

# 返回可比较的格式列表
def comparable_formats() -> list[str]: ...

# 将指定文件名的图像转换为字符串，返回转换后的文件名字符串
def convert(filename: str, cache: bool) -> str: ...

# 裁剪实际和预期图像至相同大小，返回裁剪后的实际图像和预期图像元组
def crop_to_same(
    actual_path: str, actual_image: NDArray, expected_path: str, expected_image: NDArray
) -> tuple[NDArray, NDArray]: ...

# 计算均方根误差，返回浮点数
def calculate_rms(expected_image: NDArray, actual_image: NDArray) -> float: ...

# 比较预期和实际图像，如果在装饰器内则返回字典或空值，否则返回字符串或空值
@overload
def compare_images(
    expected: str, actual: str, tol: float, in_decorator: Literal[True]
) -> None | dict[str, float | str]: ...

@overload
def compare_images(
    expected: str, actual: str, tol: float, in_decorator: Literal[False]
) -> None | str: ...

@overload
def compare_images(
    expected: str, actual: str, tol: float, in_decorator: bool = ...
) -> None | str | dict[str, float | str]: ...

# 保存差异图像到指定输出路径，无返回值
def save_diff_image(expected: str, actual: str, output: str) -> None: ...
```