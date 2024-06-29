# `D:\src\scipysrc\matplotlib\lib\matplotlib\hatch.pyi`

```
# `
from matplotlib.path import Path  # 导入 matplotlib.path 中的 Path 类，用于创建路径对象

import numpy as np  # 导入 numpy 库，别名为 np
from numpy.typing import ArrayLike  # 从 numpy.typing 导入 ArrayLike，表示数组或类似数组的对象

class HatchPatternBase: 
    # 定义一个空类作为所有哈希图案的基类
    ...

class HorizontalHatch(HatchPatternBase):
    num_lines: int  # 定义 num_lines 属性，表示水平线的数量
    num_vertices: int  # 定义 num_vertices 属性，表示顶点数量
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化水平填充模式，接受 hatch（填充模式字符串）和 density（密度）参数
        ...
    def set_vertices_and_codes(self, vertices: ArrayLike, codes: ArrayLike) -> None: 
        # 设置顶点和代码的方法，接受顶点和代码数组作为参数
        ...

class VerticalHatch(HatchPatternBase):
    num_lines: int  # 定义 num_lines 属性，表示垂直线的数量
    num_vertices: int  # 定义 num_vertices 属性，表示顶点数量
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化垂直填充模式，接受 hatch（填充模式字符串）和 density（密度）参数
        ...
    def set_vertices_and_codes(self, vertices: ArrayLike, codes: ArrayLike) -> None: 
        # 设置顶点和代码的方法，接受顶点和代码数组作为参数
        ...

class NorthEastHatch(HatchPatternBase):
    num_lines: int  # 定义 num_lines 属性，表示东北方向线的数量
    num_vertices: int  # 定义 num_vertices 属性，表示顶点数量
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化东北填充模式，接受 hatch（填充模式字符串）和 density（密度）参数
        ...
    def set_vertices_and_codes(self, vertices: ArrayLike, codes: ArrayLike) -> None: 
        # 设置顶点和代码的方法，接受顶点和代码数组作为参数
        ...

class SouthEastHatch(HatchPatternBase):
    num_lines: int  # 定义 num_lines 属性，表示东南方向线的数量
    num_vertices: int  # 定义 num_vertices 属性，表示顶点数量
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化东南填充模式，接受 hatch（填充模式字符串）和 density（密度）参数
        ...
    def set_vertices_and_codes(self, vertices: ArrayLike, codes: ArrayLike) -> None: 
        # 设置顶点和代码的方法，接受顶点和代码数组作为参数
        ...

class Shapes(HatchPatternBase):
    filled: bool  # 定义 filled 属性，表示是否填充
    num_shapes: int  # 定义 num_shapes 属性，表示形状的数量
    num_vertices: int  # 定义 num_vertices 属性，表示顶点数量
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化形状，接受 hatch（填充模式字符串）和 density（密度）参数
        ...
    def set_vertices_and_codes(self, vertices: ArrayLike, codes: ArrayLike) -> None: 
        # 设置顶点和代码的方法，接受顶点和代码数组作为参数
        ...

class Circles(Shapes):
    shape_vertices: np.ndarray  # 定义 shape_vertices 属性，表示圆形顶点数组
    shape_codes: np.ndarray  # 定义 shape_codes 属性，表示圆形代码数组
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化圆形，接受 hatch（填充模式字符串）和 density（密度）参数
        ...

class SmallCircles(Circles):
    size: float  # 定义 size 属性，表示小圆的大小
    num_rows: int  # 定义 num_rows 属性，表示行数
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化小圆，接受 hatch（填充模式字符串）和 density（密度）参数
        ...

class LargeCircles(Circles):
    size: float  # 定义 size 属性，表示大圆的大小
    num_rows: int  # 定义 num_rows 属性，表示行数
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化大圆，接受 hatch（填充模式字符串）和 density（密度）参数
        ...

class SmallFilledCircles(Circles):
    size: float  # 定义 size 属性，表示小填充圆的大小
    filled: bool  # 定义 filled 属性，表示是否填充
    num_rows: int  # 定义 num_rows 属性，表示行数
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化小填充圆，接受 hatch（填充模式字符串）和 density（密度）参数
        ...

class Stars(Shapes):
    size: float  # 定义 size 属性，表示星星的大小
    filled: bool  # 定义 filled 属性，表示是否填充
    num_rows: int  # 定义 num_rows 属性，表示行数
    shape_vertices: np.ndarray  # 定义 shape_vertices 属性，表示星形顶点数组
    shape_codes: np.ndarray  # 定义 shape_codes 属性，表示星形代码数组
    def __init__(self, hatch: str, density: int) -> None: 
        # 构造函数，初始化星星，接受 hatch（填充模式字符串）和 density（密度）参数
        ...

def get_path(hatchpattern: str, density: int = ...) -> Path: 
    # 定义一个函数，接受 hatchpattern（填充模式字符串）和 density（密度），返回一个 Path 对象
    ...
```