# `D:\src\scipysrc\matplotlib\lib\matplotlib\path.pyi`

```py
# 导入曲线生成相关的类
from .bezier import BezierSegment
# 导入几何变换相关的类和边界框类
from .transforms import Affine2D, Transform, Bbox
# 导入用于生成生成器、可迭代对象和序列的集合抽象基类
from collections.abc import Generator, Iterable, Sequence

# 导入 NumPy 库并引入 ArrayLike 类型提示
import numpy as np
from numpy.typing import ArrayLike

# 导入 overload 类型提示，用于函数重载
from typing import Any, overload

# 定义路径类 Path
class Path:
    # 定义代码类型为 np.uint8
    code_type: type[np.uint8]
    # 定义常量 STOP 为 np.uint8 类型
    STOP: np.uint8
    # 定义常量 MOVETO 为 np.uint8 类型
    MOVETO: np.uint8
    # 定义常量 LINETO 为 np.uint8 类型
    LINETO: np.uint8
    # 定义常量 CURVE3 为 np.uint8 类型
    CURVE3: np.uint8
    # 定义常量 CURVE4 为 np.uint8 类型
    CURVE4: np.uint8
    # 定义常量 CLOSEPOLY 为 np.uint8 类型
    CLOSEPOLY: np.uint8
    # 定义字典类型常量 NUM_VERTICES_FOR_CODE，键为 np.uint8，值为 int
    NUM_VERTICES_FOR_CODE: dict[np.uint8, int]

    # 初始化方法，接受顶点数组 vertices、代码数组 codes、插值步数 _interpolation_steps、是否闭合 closed、是否只读 readonly
    def __init__(
        self,
        vertices: ArrayLike,
        codes: ArrayLike | None = ...,
        _interpolation_steps: int = ...,
        closed: bool = ...,
        readonly: bool = ...,
    ) -> None: ...

    # 返回顶点数组 vertices 的属性
    @property
    def vertices(self) -> ArrayLike: ...

    # 设置顶点数组 vertices 的属性
    @vertices.setter
    def vertices(self, vertices: ArrayLike) -> None: ...

    # 返回代码数组 codes 的属性
    @property
    def codes(self) -> ArrayLike | None: ...

    # 设置代码数组 codes 的属性
    @codes.setter
    def codes(self, codes: ArrayLike) -> None: ...

    # 返回简化阈值 simplify_threshold 的属性
    @property
    def simplify_threshold(self) -> float: ...

    # 设置简化阈值 simplify_threshold 的属性
    @simplify_threshold.setter
    def simplify_threshold(self, threshold: float) -> None: ...

    # 返回是否应该简化 should_simplify 的属性
    @property
    def should_simplify(self) -> bool: ...

    # 设置是否应该简化 should_simplify 的属性
    @should_simplify.setter
    def should_simplify(self, should_simplify: bool) -> None: ...

    # 返回是否只读 readonly 的属性
    @property
    def readonly(self) -> bool: ...

    # 复制路径对象的方法
    def copy(self) -> Path: ...

    # 深拷贝路径对象的方法
    def __deepcopy__(self, memo: dict[int, Any] | None = ...) -> Path: ...
    deepcopy = __deepcopy__

    # 从多边形数组 XY 中生成复合路径的类方法
    @classmethod
    def make_compound_path_from_polys(cls, XY: ArrayLike) -> Path: ...

    # 从多个路径对象 args 中生成复合路径的类方法
    @classmethod
    def make_compound_path(cls, *args: Path) -> Path: ...

    # 返回路径对象的长度
    def __len__(self) -> int: ...

    # 返回迭代段的生成器方法，可选应用变换 transform、删除 NaN 值 remove_nans、剪裁范围 clip、是否捕捉 snap、线段宽度 stroke_width、简化路径 simplify、是否包含曲线 curves、素描效果 sketch
    def iter_segments(
        self,
        transform: Transform | None = ...,
        remove_nans: bool = ...,
        clip: tuple[float, float, float, float] | None = ...,
        snap: bool | None = ...,
        stroke_width: float = ...,
        simplify: bool | None = ...,
        curves: bool = ...,
        sketch: tuple[float, float, float] | None = ...,
    ) -> Generator[tuple[np.ndarray, np.uint8], None, None]: ...

    # 返回贝塞尔曲线段的生成器方法，接受任意关键字参数 kwargs
    def iter_bezier(self, **kwargs) -> Generator[BezierSegment, None, None]: ...

    # 返回清理后的路径对象的方法，可选应用变换 transform、删除 NaN 值 remove_nans、剪裁范围 clip、是否简化 simplify、是否包含曲线 curves、线段宽度 stroke_width、是否捕捉 snap、素描效果 sketch
    def cleaned(
        self,
        transform: Transform | None = ...,
        remove_nans: bool = ...,
        clip: tuple[float, float, float, float] | None = ...,
        *,
        simplify: bool | None = ...,
        curves: bool = ...,
        stroke_width: float = ...,
        snap: bool | None = ...,
        sketch: tuple[float, float, float] | None = ...
    ) -> Path: ...

    # 返回应用变换 transform 后的路径对象的方法
    def transformed(self, transform: Transform) -> Path: ...

    # 检查是否包含点 point 的方法，可选应用变换 transform、半径 radius
    def contains_point(
        self,
        point: tuple[float, float],
        transform: Transform | None = ...,
        radius: float = ...,
    ) -> bool: ...

    # 检查是否包含点数组 points 的方法，可选应用变换 transform、半径 radius
    def contains_points(
        self, points: ArrayLike, transform: Transform | None = ..., radius: float = ...
    ) -> np.ndarray: ...

    # 检查是否包含路径 path 的方法，可选应用变换 transform
    def contains_path(self, path: Path, transform: Transform | None = ...) -> bool: ...
    # 返回该路径的边界框 (bounding box)，如果提供了变换则应用于该路径
    def get_extents(self, transform: Transform | None = ..., **kwargs) -> Bbox: ...
    
    # 检查该路径是否与另一条路径相交，可以选择考虑路径是否填充
    def intersects_path(self, other: Path, filled: bool = ...) -> bool: ...
    
    # 检查该路径是否与给定边界框相交，可以选择考虑路径是否填充
    def intersects_bbox(self, bbox: Bbox, filled: bool = ...) -> bool: ...
    
    # 在当前路径上插值给定步数的路径点，返回插值后的路径
    def interpolated(self, steps: int) -> Path: ...
    
    # 将路径转换为多边形列表，可以选择应用变换、指定宽度和高度，以及是否只包括闭合路径
    def to_polygons(
        self,
        transform: Transform | None = ...,
        width: float = ...,
        height: float = ...,
        closed_only: bool = ...,
    ) -> list[ArrayLike]: ...
    
    # 返回单位矩形的路径对象
    @classmethod
    def unit_rectangle(cls) -> Path: ...
    
    # 返回具有给定顶点数的单位正多边形的路径对象
    @classmethod
    def unit_regular_polygon(cls, numVertices: int) -> Path: ...
    
    # 返回具有给定顶点数和内圆半径的单位正星形路径对象
    @classmethod
    def unit_regular_star(cls, numVertices: int, innerCircle: float = ...) -> Path: ...
    
    # 返回具有给定顶点数的单位星形路径对象
    @classmethod
    def unit_regular_asterisk(cls, numVertices: int) -> Path: ...
    
    # 返回单位圆的路径对象
    @classmethod
    def unit_circle(cls) -> Path: ...
    
    # 返回具有给定中心和半径的圆形路径对象，可以选择是否只读
    @classmethod
    def circle(
        cls,
        center: tuple[float, float] = ...,
        radius: float = ...,
        readonly: bool = ...,
    ) -> Path: ...
    
    # 返回单位圆的右半部分路径对象
    @classmethod
    def unit_circle_righthalf(cls) -> Path: ...
    
    # 返回从给定起始角度到结束角度的圆弧路径对象，可以选择是否为楔形
    @classmethod
    def arc(
        cls, theta1: float, theta2: float, n: int | None = ..., is_wedge: bool = ...
    ) -> Path: ...
    
    # 返回从给定起始角度到结束角度的圆环路径对象
    @classmethod
    def wedge(cls, theta1: float, theta2: float, n: int | None = ...) -> Path: ...
    
    # 静态方法重载：根据指定的填充图案和密度生成路径对象
    @overload
    @staticmethod
    def hatch(hatchpattern: str, density: float = ...) -> Path: ...
    
    # 静态方法重载：根据提供的 None 值和密度生成路径对象
    @overload
    @staticmethod
    def hatch(hatchpattern: None, density: float = ...) -> None: ...
    
    # 将路径对象裁剪到给定的边界框，可以选择裁剪到内部
# 定义一个函数，用于计算路径集合的边界框
def get_path_collection_extents(
    master_transform: Transform,
    # 主变换，应用于所有路径
    paths: Sequence[Path],
    # 要计算边界框的路径序列
    transforms: Iterable[Affine2D],
    # 应用于每个路径的变换列表
    offsets: ArrayLike,
    # 路径的偏移量
    offset_transform: Affine2D,
    # 应用于整体偏移的变换
) -> Bbox:
    # 函数声明，尚未实现，返回类型为边界框对象
    ...
```