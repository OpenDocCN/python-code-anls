# `D:\src\scipysrc\matplotlib\lib\matplotlib\texmanager.pyi`

```
from .backend_bases import RendererBase  # 导入 RendererBase 类，来自于 .backend_bases 模块

from matplotlib.typing import ColorType  # 导入 ColorType 类型，来自于 matplotlib.typing 模块

import numpy as np  # 导入 numpy 库，并使用 np 作为别名

class TexManager:
    texcache: str  # 定义类变量 texcache，类型为 str

    @classmethod
    def get_basefile(
        cls, tex: str, fontsize: float, dpi: float | None = ...
    ) -> str: ...
    # 类方法 get_basefile，接受 tex（TeX 表达式）、fontsize（字体大小）、dpi（分辨率，默认可选）参数，
    # 返回类型为 str，具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def get_font_preamble(cls) -> str: ...
    # 类方法 get_font_preamble，返回类型为 str，具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def get_custom_preamble(cls) -> str: ...
    # 类方法 get_custom_preamble，返回类型为 str，具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def make_tex(cls, tex: str, fontsize: float) -> str: ...
    # 类方法 make_tex，接受 tex（TeX 表达式）和 fontsize（字体大小）参数，返回类型为 str，
    # 具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def make_dvi(cls, tex: str, fontsize: float) -> str: ...
    # 类方法 make_dvi，接受 tex（TeX 表达式）和 fontsize（字体大小）参数，返回类型为 str，
    # 具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def make_png(cls, tex: str, fontsize: float, dpi: float) -> str: ...
    # 类方法 make_png，接受 tex（TeX 表达式）、fontsize（字体大小）、dpi（分辨率）参数，
    # 返回类型为 str，具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def get_grey(
        cls, tex: str, fontsize: float | None = ..., dpi: float | None = ...
    ) -> np.ndarray: ...
    # 类方法 get_grey，接受 tex（TeX 表达式）、fontsize（字体大小，可选）、dpi（分辨率，可选）参数，
    # 返回类型为 numpy.ndarray，具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def get_rgba(
        cls,
        tex: str,
        fontsize: float | None = ...,
        dpi: float | None = ...,
        rgb: ColorType = ...,
    ) -> np.ndarray: ...
    # 类方法 get_rgba，接受 tex（TeX 表达式）、fontsize（字体大小，可选）、dpi（分辨率，可选）、
    # rgb（颜色类型参数）参数，返回类型为 numpy.ndarray，具体实现未提供，因此使用省略号表示未实现内容

    @classmethod
    def get_text_width_height_descent(
        cls, tex: str, fontsize, renderer: RendererBase | None = ...
    ) -> tuple[int, int, int]: ...
    # 类方法 get_text_width_height_descent，接受 tex（TeX 表达式）、fontsize（字体大小）、
    # renderer（渲染器，可选，类型为 RendererBase 或 None）参数，返回类型为元组[int, int, int]，
    # 具体实现未提供，因此使用省略号表示未实现内容
```