# `D:\src\scipysrc\matplotlib\lib\matplotlib\table.pyi`

```
# 导入所需的类和模块
from .artist import Artist             # 导入Artist类
from .axes import Axes                 # 导入Axes类
from .backend_bases import RendererBase # 导入RendererBase类
from .patches import Rectangle         # 导入Rectangle类
from .path import Path                 # 导入Path类
from .text import Text                 # 导入Text类
from .transforms import Bbox           # 导入Bbox类
from .typing import ColorType          # 导入ColorType类型

from collections.abc import Sequence   # 导入Sequence抽象基类
from typing import Any, Literal        # 导入Any和Literal类型

# 定义一个名为Cell的类，继承自Rectangle类
class Cell(Rectangle):
    PAD: float                          # 类常量PAD，类型为float

    # 构造函数，接受一系列位置和样式参数，初始化一个Cell对象
    def __init__(
        self,
        xy: tuple[float, float],         # 位置坐标的元组，包含两个float类型的值
        width: float,                   # 宽度，float类型
        height: float,                  # 高度，float类型
        *,
        edgecolor: ColorType = ...,     # 边框颜色，ColorType类型
        facecolor: ColorType = ...,     # 填充颜色，ColorType类型
        fill: bool = ...,               # 是否填充，bool类型
        text: str = ...,                # 文本内容，str类型
        loc: Literal["left", "center", "right"] = ...,  # 位置，Literal类型，取值为"left", "center", "right"
        fontproperties: dict[str, Any] | None = ...,     # 字体属性，字典或None类型
        visible_edges: str | None = ...                  # 可见边缘，str或None类型
    ) -> None: ...                       # 返回值为None，构造函数无具体实现

    # 返回一个Text对象，表示单元格的文本
    def get_text(self) -> Text: ...

    # 设置文本的字体大小
    def set_fontsize(self, size: float) -> None: ...

    # 返回当前文本的字体大小
    def get_fontsize(self) -> float: ...

    # 根据渲染器自动设置文本的字体大小，并返回字体大小
    def auto_set_font_size(self, renderer: RendererBase) -> float: ...

    # 返回文本边界的坐标信息，包括左下角和右上角的坐标
    def get_text_bounds(
        self, renderer: RendererBase  # 渲染器，RendererBase类型
    ) -> tuple[float, float, float, float]: ...

    # 返回单元格所需的宽度，根据给定的渲染器
    def get_required_width(self, renderer: RendererBase) -> float: ...

    # 设置文本的属性，接受任意关键字参数
    def set_text_props(self, **kwargs) -> None: ...

    # 可读写属性，返回可见的边缘
    @property
    def visible_edges(self) -> str: ...

    # 可见边缘的设置器，接受一个str或None类型的值
    @visible_edges.setter
    def visible_edges(self, value: str | None) -> None: ...

    # 返回单元格的Path对象，用于绘制路径
    def get_path(self) -> Path: ...

# 将Cell类赋值给CustomCell变量
CustomCell = Cell

# 定义一个名为Table的类，继承自Artist类
class Table(Artist):
    codes: dict[str, int]                # 属性，类型为字典，键为str，值为int

    # 类常量FONTSIZE，表示字体大小，类型为float
    FONTSIZE: float

    # 类常量AXESPAD，表示轴的填充大小，类型为float
    AXESPAD: float

    # 构造函数，接受一个Axes对象ax，位置loc（str类型或None），边界框bbox（Bbox类型或None），以及其他关键字参数
    def __init__(
        self, ax: Axes,                  # Axes对象，用于表的绘制
        loc: str | None = ...,           # 位置，str类型或None
        bbox: Bbox | None = ...,         # 边界框，Bbox类型或None
        **kwargs
    ) -> None: ...

    # 添加一个单元格到表中，接受行row和列col的位置索引，以及其他位置和样式参数
    def add_cell(self, row: int, col: int, *args, **kwargs) -> Cell: ...

    # 设置表中指定位置的单元格，接受位置和Cell对象作为参数
    def __setitem__(self, position: tuple[int, int], cell: Cell) -> None: ...

    # 返回表中指定位置的单元格，接受位置索引作为参数
    def __getitem__(self, position: tuple[int, int]) -> Cell: ...

    # 可读写属性，返回表的边缘信息
    @property
    def edges(self) -> str | None: ...

    # 边缘信息的设置器，接受一个str或None类型的值
    @edges.setter
    def edges(self, value: str | None) -> None: ...

    # 绘制表格，接受渲染器作为参数
    def draw(self, renderer) -> None: ...

    # 返回表格中的所有子元素，类型为Artist的列表
    def get_children(self) -> list[Artist]: ...

    # 返回表格在给定渲染器下的窗口范围，类型为Bbox
    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox: ...

    # 自动设置列的宽度，接受一个列索引或列索引列表作为参数
    def auto_set_column_width(self, col: int | Sequence[int]) -> None: ...

    # 自动设置表格中所有文本的字体大小，接受一个bool类型的参数，默认为...
    def auto_set_font_size(self, value: bool = ...) -> None: ...

    # 缩放表格，接受x轴和y轴的缩放比例作为参数
    def scale(self, xscale: float, yscale: float) -> None: ...

    # 设置表格的字体大小，接受一个float类型的参数
    def set_fontsize(self, size: float) -> None: ...

    # 返回一个字典，包含表格中所有单元格的位置索引到Cell对象的映射关系
    def get_celld(self) -> dict[tuple[int, int], Cell]: ...

# 定义一个函数table，用于创建和配置表格
def table(
    ax: Axes,                            # Axes对象，用于表格的绘制
    cellText: Sequence[Sequence[str]] | None = ...,   # 单元格文本内容，二维字符串序列或None
    cellColours: Sequence[Sequence[ColorType]] | None = ...,  # 单元格颜色，二维ColorType序列或None
    cellLoc: Literal["left", "center", "right"] = ...,       # 单元格文本对齐方式，Literal类型，取值为"left", "center", "right"
    colWidths: Sequence[float] | None = ...,                 # 列宽度，浮点数序列或None
    rowLabels: Sequence[str] | None = ...,                   # 行标签，字符串序列或None
    rowColours: Sequence[ColorType] | None = ...,            # 行颜色，ColorType序列或None
    rowLoc: Literal["left", "center", "right"] = ...,        # 行文本对齐方式，Literal类型，取值为"left", "center", "right"
    colLabels: Sequence[str] | None = ...,                   # 列标签，字符串序列或None
    colColours: Sequence[ColorType] | None = ...,            # 列颜色，ColorType序列或None
    colLoc: Literal["left", "center", "right"] = ...,        # 列文本对齐方式，Literal类型，取值为"left", "center", "right"
    loc: str = ...,                     # 表格位置，str类型
    bbox: Bbox | None = ...,            # 边界框，Bbox类型或None
    # 定义一个变量 edges，类型为字符串，使用"..."作为默认值
    edges: str = ...,
    # 允许接受任意额外的关键字参数，这些参数将会保存在 kwargs 中
    **kwargs
# 定义一个函数，函数名为 `)`，它接受一个参数，并且返回一个类型为 `Table` 的对象
def function_name(parameter_name) -> Table:
    ...
```