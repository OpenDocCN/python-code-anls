# `D:\src\scipysrc\matplotlib\lib\matplotlib\dviread.pyi`

```
# 从 pathlib 模块导入 Path 类，用于处理文件路径
from pathlib import Path
# 导入 io 模块，用于提供核心的 IO 功能
import io
# 导入 os 模块，用于提供操作系统相关的功能
import os
# 从 enum 模块导入 Enum 类，用于创建枚举类型
from enum import Enum
# 从 collections.abc 模块导入 Generator 类型，用于定义生成器类型
from collections.abc import Generator

# 从 typing 模块导入 NamedTuple 类型，用于定义命名元组
from typing import NamedTuple

# 定义一个枚举类型 _dvistate，包含几个状态常量
class _dvistate(Enum):
    pre: int
    outer: int
    inpage: int
    post_post: int
    finale: int

# 定义一个命名元组 Page，包含文本、框等字段
class Page(NamedTuple):
    text: list[Text]
    boxes: list[Box]
    height: int
    width: int
    descent: int

# 定义一个命名元组 Box，包含位置和尺寸字段
class Box(NamedTuple):
    x: int
    y: int
    height: int
    width: int

# 定义一个命名元组 Text，包含文本位置、字体信息等字段
class Text(NamedTuple):
    x: int
    y: int
    font: DviFont
    glyph: int
    width: int
    @property
    # 返回文本所使用的字体的路径
    def font_path(self) -> Path: ...
    @property
    # 返回文本所使用的字体的大小
    def font_size(self) -> float: ...
    @property
    # 返回文本的字体效果
    def font_effects(self) -> dict[str, float]: ...
    @property
    # 返回文本使用的字形名称或索引
    def glyph_name_or_index(self) -> int | str: ...

# 定义一个类 Dvi，表示 DVI 文件
class Dvi:
    file: io.BufferedReader
    dpi: float | None
    fonts: dict[int, DviFont]
    state: _dvistate
    def __init__(self, filename: str | os.PathLike, dpi: float | None) -> None:
        # 初始化 Dvi 对象，接受文件名或路径以及 DPI 参数
        ...
    # 在 Python 3.9 之前的版本，替换返回类型为 Self
    def __enter__(self) -> Dvi: ...
    # 定义退出方法，用于资源清理
    def __exit__(self, etype, evalue, etrace) -> None: ...
    # 实现迭代器接口，返回生成器以生成页面对象
    def __iter__(self) -> Generator[Page, None, None]: ...
    # 关闭 Dvi 对象的方法
    def close(self) -> None: ...

# 定义一个字体类 DviFont，用于表示 DVI 文件中的字体
class DviFont:
    texname: bytes
    size: float
    widths: list[int]
    def __init__(
        self, scale: float, tfm: Tfm, texname: bytes, vf: Vf | None
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

# 定义一个特殊的 Dvi 类 Vf，表示 VF 文件，继承自 Dvi 类
class Vf(Dvi):
    def __init__(self, filename: str | os.PathLike) -> None: ...
    # 根据编码返回页面对象
    def __getitem__(self, code: int) -> Page: ...

# 定义一个 Tfm 类，表示 TFN 文件
class Tfm:
    checksum: int
    design_size: int
    width: dict[int, int]
    height: dict[int, int]
    depth: dict[int, int]
    def __init__(self, filename: str | os.PathLike) -> None: ...

# 定义一个 PsFont 命名元组，表示 PS 字体
class PsFont(NamedTuple):
    texname: bytes
    psname: bytes
    effects: dict[str, float]
    encoding: None | bytes
    filename: str

# 定义一个 PsfontsMap 类，用于管理 PS 字体映射
class PsfontsMap:
    # 在 Python 3.9 之前的版本，替换返回类型为 Self
    def __new__(cls, filename: str | os.PathLike) -> PsfontsMap: ...
    # 根据字体名称返回 PsFont 对象
    def __getitem__(self, texname: bytes) -> PsFont: ...

# 定义一个函数 find_tex_file，根据文件名查找 TeX 文件并返回路径
def find_tex_file(filename: str | os.PathLike) -> str: ...
```