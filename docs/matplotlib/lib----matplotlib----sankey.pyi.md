# `D:\src\scipysrc\matplotlib\lib\matplotlib\sankey.pyi`

```py
# 从 matplotlib.axes 模块中导入 Axes 类
from matplotlib.axes import Axes

# 从 collections.abc 模块中导入 Callable 和 Iterable 类型
from collections.abc import Callable, Iterable

# 从 typing 模块中导入 Any 类型
from typing import Any

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 定义以下变量，但未赋值
__license__: str
__credits__: list[str]
__author__: str
__version__: str

# 定义常量 RIGHT、UP、DOWN，分别表示整数值
RIGHT: int
UP: int
DOWN: int

# TODO typing units
# Sankey 类的定义，用于绘制桑基图
class Sankey:
    # diagrams 属性，用于存储多个图表
    diagrams: list[Any]
    # ax 属性，表示图表的坐标轴，类型为 Axes
    ax: Axes
    # unit 属性，表示桑基图的单位
    unit: Any
    # format 属性，用于格式化数据显示的格式，可以是字符串或格式化函数
    format: str | Callable[[float], str]
    # scale 属性，表示桑基图的比例尺
    scale: float
    # gap 属性，表示桑基图中的间隙大小
    gap: float
    # radius 属性，表示桑基图中环的半径大小
    radius: float
    # shoulder 属性，表示桑基图中肩部的大小
    shoulder: float
    # offset 属性，表示桑基图的偏移量
    offset: float
    # margin 属性，表示桑基图的边距大小
    margin: float
    # pitch 属性，表示桑基图的间距
    pitch: float
    # tolerance 属性，表示桑基图的公差
    tolerance: float
    # extent 属性，表示桑基图的范围，类型为 numpy 的数组
    extent: np.ndarray

    # 构造方法，初始化 Sankey 类的实例
    def __init__(
        self,
        ax: Axes | None = ...,
        scale: float = ...,
        unit: Any = ...,
        format: str | Callable[[float], str] = ...,
        gap: float = ...,
        radius: float = ...,
        shoulder: float = ...,
        offset: float = ...,
        head_angle: float = ...,
        margin: float = ...,
        tolerance: float = ...,
        **kwargs
    ) -> None:
        ...

    # 添加方法，用于向桑基图中添加流、标签等信息
    def add(
        self,
        patchlabel: str = ...,
        flows: Iterable[float] | None = ...,
        orientations: Iterable[int] | None = ...,
        labels: str | Iterable[str | None] = ...,
        trunklength: float = ...,
        pathlengths: float | Iterable[float] = ...,
        prior: int | None = ...,
        connect: tuple[int, int] = ...,
        rotation: float = ...,
        **kwargs
    ) -> Sankey:
        # 将添加的元素包含进桑基图中
        # Replace return with Self when py3.9 is dropped
        ...

    # 结束方法，完成桑基图的绘制并返回图表列表
    def finish(self) -> list[Any]:
        ...
```