# `D:\src\scipysrc\matplotlib\lib\matplotlib\transforms.pyi`

```
# 从路径模块导入 Path 类
# 注意这里的 . 表示当前模块的路径
from .path import Path
# 从补丁模块导入 Patch 类
from .patches import Patch
# 从图形模块导入 Figure 类
from .figure import Figure
# 导入 numpy 库，并指定为 np 别名
import numpy as np
# 导入 numpy.typing 中的 ArrayLike 类型
from numpy.typing import ArrayLike
# 导入 collections.abc 中的 Iterable 和 Sequence 抽象基类
from collections.abc import Iterable, Sequence
# 导入 typing 中的 Literal 类型提示
from typing import Literal

# 定义 DEBUG 常量，表示调试模式是否开启
DEBUG: bool

# 定义 TransformNode 类，表示变换节点
class TransformNode:
    # 定义常量 INVALID_NON_AFFINE，表示非仿射无效性
    INVALID_NON_AFFINE: int
    # 定义常量 INVALID_AFFINE，表示仿射无效性
    INVALID_AFFINE: int
    # 定义常量 INVALID，表示无效性
    INVALID: int
    # 定义属性 is_bbox，表示是否为边界框
    is_bbox: bool
    # 定义属性 is_affine，表示是否为仿射变换
    # 注意：此处的属性实现为标准属性，但在基类中功能上是只读的，某些子类可能会重新实现
    @property
    def is_affine(self) -> bool: ...
    # 定义属性 pass_through，表示是否透传
    pass_through: bool
    # 定义初始化方法 __init__，接受参数 shorthand_name 作为缩写名，默认为 None
    def __init__(self, shorthand_name: str | None = ...) -> None: ...
    # 定义复制方法 __copy__，返回类型为 TransformNode 的复制对象
    def __copy__(self) -> TransformNode: ...
    # 定义失效方法 invalidate，不返回任何值
    def invalidate(self) -> None: ...
    # 定义设置子节点方法 set_children，接受任意数量的 TransformNode 类型参数
    def set_children(self, *children: TransformNode) -> None: ...
    # 定义冻结方法 frozen，返回类型为 TransformNode 的冻结对象
    def frozen(self) -> TransformNode: ...

# 定义 BboxBase 类，表示边界框基类，继承自 TransformNode 类
class BboxBase(TransformNode):
    # 定义属性 is_bbox，表示是否为边界框
    is_bbox: bool
    # 定义属性 is_affine，表示是否为仿射变换
    is_affine: bool
    # 定义冻结方法 frozen，返回类型为 Bbox 的冻结对象
    def frozen(self) -> Bbox: ...
    # 定义 __array__ 方法，可以转换为 numpy 数组
    def __array__(self, *args, **kwargs): ...
    # 定义属性 x0，返回类型为 float 的 x 起始坐标
    @property
    def x0(self) -> float: ...
    # 定义属性 y0，返回类型为 float 的 y 起始坐标
    @property
    def y0(self) -> float: ...
    # 定义属性 x1，返回类型为 float 的 x 结束坐标
    @property
    def x1(self) -> float: ...
    # 定义属性 y1，返回类型为 float 的 y 结束坐标
    @property
    def y1(self) -> float: ...
    # 定义属性 p0，返回类型为元组 (float, float) 的起始点坐标
    @property
    def p0(self) -> tuple[float, float]: ...
    # 定义属性 p1，返回类型为元组 (float, float) 的结束点坐标
    @property
    def p1(self) -> tuple[float, float]: ...
    # 定义属性 xmin，返回类型为 float 的 x 最小值
    @property
    def xmin(self) -> float: ...
    # 定义属性 ymin，返回类型为 float 的 y 最小值
    @property
    def ymin(self) -> float: ...
    # 定义属性 xmax，返回类型为 float 的 x 最大值
    @property
    def xmax(self) -> float: ...
    # 定义属性 ymax，返回类型为 float 的 y 最大值
    @property
    def ymax(self) -> float: ...
    # 定义属性 min，返回类型为元组 (float, float) 的最小值点坐标
    @property
    def min(self) -> tuple[float, float]: ...
    # 定义属性 max，返回类型为元组 (float, float) 的最大值点坐标
    @property
    def max(self) -> tuple[float, float]: ...
    # 定义属性 intervalx，返回类型为元组 (float, float) 的 x 轴区间
    @property
    def intervalx(self) -> tuple[float, float]: ...
    # 定义属性 intervaly，返回类型为元组 (float, float) 的 y 轴区间
    @property
    def intervaly(self) -> tuple[float, float]: ...
    # 定义属性 width，返回类型为 float 的宽度
    @property
    def width(self) -> float: ...
    # 定义属性 height，返回类型为 float 的高度
    @property
    def height(self) -> float: ...
    # 定义属性 size，返回类型为元组 (float, float) 的尺寸
    @property
    def size(self) -> tuple[float, float]: ...
    # 定义属性 bounds，返回类型为元组 (float, float, float, float) 的边界
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...
    # 定义属性 extents，返回类型为元组 (float, float, float, float) 的范围
    @property
    def extents(self) -> tuple[float, float, float, float]: ...
    # 定义获取点方法 get_points，返回类型为 numpy 数组的点坐标
    def get_points(self) -> np.ndarray: ...
    # 定义检查 x 坐标是否包含方法 containsx，接受参数 x，返回类型为 bool
    def containsx(self, x: float) -> bool: ...
    # 定义检查 y 坐标是否包含方法 containsy，接受参数 y，返回类型为 bool
    def containsy(self, y: float) -> bool: ...
    # 定义检查点坐标是否包含方法 contains，接受参数 x, y，返回类型为 bool
    def contains(self, x: float, y: float) -> bool: ...
    # 定义检查是否与另一个 BboxBase 对象重叠方法 overlaps，接受参数 other，返回类型为 bool
    def overlaps(self, other: BboxBase) -> bool: ...
    # 定义检查是否完全包含 x 坐标方法 fully_containsx，接受参数 x，返回类型为 bool
    def fully_containsx(self, x: float) -> bool: ...
    # 定义检查是否完全包含 y 坐标方法 fully_containsy，接受参数 y，返回类型为 bool
    def fully_containsy(self, y: float) -> bool: ...
    # 定义检查是否完全包含点坐标方法 fully_contains，接受参数 x, y，返回类型为 bool
    def fully_contains(self, x: float, y: float) -> bool: ...
    # 定义检查是否完全重叠另一个 BboxBase 对象方法 fully_overlaps，接受参数 other，返回类型为 bool
    def fully_overlaps(self, other: BboxBase) -> bool: ...
    # 定义变换方法 transformed，接受参数 transform，返回类型为 Bbox 的变换对象
    def transformed(self, transform: Transform) -> Bbox: ...
    # 定义属性 coefs，表示系数字典，键为 str，值为元组 (float, float)
    coefs: dict[str, tuple[float, float]]
    # 定义 anchored 方法，接受参数 c 和 container，返回类型为 Bbox
    # 参数 c 可以是元组 (float, float) 或字符串字面量 "C", "SW", "S", "SE", "E", "NE", "N", "NW", "W"
    def anchored(
        self, c: tuple[float, float] | str, container: BboxBase | None
    # 定义一个方法splitx，接受任意数量的float类型参数，并返回Bbox对象列表
    def splitx(self, *args: float) -> list[Bbox]: ...

    # 定义一个方法splity，接受任意数量的float类型参数，并返回Bbox对象列表
    def splity(self, *args: float) -> list[Bbox]: ...

    # 定义一个方法count_contains，接受ArrayLike类型的vertices参数，返回int类型结果
    def count_contains(self, vertices: ArrayLike) -> int: ...

    # 定义一个方法count_overlaps，接受Iterable类型的BboxBase对象序列bboxes参数，返回int类型结果
    def count_overlaps(self, bboxes: Iterable[BboxBase]) -> int: ...

    # 定义一个方法expanded，接受两个float类型参数sw和sh，返回Bbox对象
    def expanded(self, sw: float, sh: float) -> Bbox: ...

    # 定义一个方法padded，接受两个float类型参数w_pad和h_pad，h_pad参数可选，默认为...
    def padded(self, w_pad: float, h_pad: float | None = ...) -> Bbox: ...

    # 定义一个方法translated，接受两个float类型参数tx和ty，返回Bbox对象
    def translated(self, tx: float, ty: float) -> Bbox: ...

    # 定义一个方法corners，返回一个numpy数组(ndarray)
    def corners(self) -> np.ndarray: ...

    # 定义一个方法rotated，接受一个float类型参数radians，返回Bbox对象
    def rotated(self, radians: float) -> Bbox: ...

    # 定义一个静态方法union，接受一个Sequence类型的BboxBase对象序列bboxes参数，返回一个Bbox对象
    @staticmethod
    def union(bboxes: Sequence[BboxBase]) -> Bbox: ...

    # 定义一个静态方法intersection，接受两个BboxBase对象bbox1和bbox2参数，返回一个可空的Bbox对象或None
    @staticmethod
    def intersection(bbox1: BboxBase, bbox2: BboxBase) -> Bbox | None: ...
class Bbox(BboxBase):
    # Bbox 类继承自 BboxBase 类

    def __init__(self, points: ArrayLike, **kwargs) -> None:
        # Bbox 类的初始化方法，接收 points 参数作为数组输入

    @staticmethod
    def unit() -> Bbox:
        # 返回一个表示单位框的 Bbox 对象的静态方法

    @staticmethod
    def null() -> Bbox:
        # 返回一个表示空框的 Bbox 对象的静态方法

    @staticmethod
    def from_bounds(x0: float, y0: float, width: float, height: float) -> Bbox:
        # 根据给定的边界参数创建一个 Bbox 对象的静态方法

    @staticmethod
    def from_extents(*args: float, minpos: float | None = ...) -> Bbox:
        # 根据给定的参数范围创建一个 Bbox 对象的静态方法

    def __format__(self, fmt: str) -> str:
        # 控制 Bbox 对象格式化输出的方法

    def ignore(self, value: bool) -> None:
        # 设置是否忽略 Bbox 的方法，接收一个布尔值参数

    def update_from_path(
        self,
        path: Path,
        ignore: bool | None = ...,
        updatex: bool = ...,
        updatey: bool = ...
    ) -> None:
        # 根据路径对象更新 Bbox 的方法，可以选择性地忽略、更新 x 或 y 轴

    def update_from_data_x(self, x: ArrayLike, ignore: bool | None = ...) -> None:
        # 根据 x 轴数据更新 Bbox 的方法，可以选择性地忽略更新

    def update_from_data_y(self, y: ArrayLike, ignore: bool | None = ...) -> None:
        # 根据 y 轴数据更新 Bbox 的方法，可以选择性地忽略更新

    def update_from_data_xy(
        self,
        xy: ArrayLike,
        ignore: bool | None = ...,
        updatex: bool = ...,
        updatey: bool = ...
    ) -> None:
        # 根据 x 和 y 轴数据更新 Bbox 的方法，可以选择性地忽略、更新 x 或 y 轴

    @property
    def minpos(self) -> float:
        # 返回 Bbox 的最小位置值的属性

    @property
    def minposx(self) -> float:
        # 返回 Bbox 在 x 轴上的最小位置值的属性

    @property
    def minposy(self) -> float:
        # 返回 Bbox 在 y 轴上的最小位置值的属性

    def get_points(self) -> np.ndarray:
        # 获取 Bbox 的角点数组的方法，返回一个 NumPy 数组

    def set_points(self, points: ArrayLike) -> None:
        # 设置 Bbox 的角点数组的方法，接收一个数组作为参数

    def set(self, other: Bbox) -> None:
        # 根据另一个 Bbox 对象设置当前 Bbox 的方法

    def mutated(self) -> bool:
        # 检查 Bbox 是否已变异的方法，返回布尔值

    def mutatedx(self) -> bool:
        # 检查 Bbox 在 x 轴上是否已变异的方法，返回布尔值

    def mutatedy(self) -> bool:
        # 检查 Bbox 在 y 轴上是否已变异的方法，返回布尔值


class TransformedBbox(BboxBase):
    def __init__(self, bbox: Bbox, transform: Transform, **kwargs) -> None:
        # 初始化 TransformedBbox 对象，接收一个 Bbox 和一个 Transform 对象作为参数

    def get_points(self) -> np.ndarray:
        # 获取 TransformedBbox 的角点数组的方法，返回一个 NumPy 数组


class LockableBbox(BboxBase):
    def __init__(
        self,
        bbox: BboxBase,
        x0: float | None = ...,
        y0: float | None = ...,
        x1: float | None = ...,
        y1: float | None = ...,
        **kwargs
    ) -> None:
        # 初始化 LockableBbox 对象，接收一个 BboxBase 对象和可选的边界参数作为参数

    @property
    def locked_x0(self) -> float | None:
        # 返回 LockableBbox 的 x0 边界值或 None 的属性

    @locked_x0.setter
    def locked_x0(self, x0: float | None) -> None:
        # 设置 LockableBbox 的 x0 边界值的方法

    @property
    def locked_y0(self) -> float | None:
        # 返回 LockableBbox 的 y0 边界值或 None 的属性

    @locked_y0.setter
    def locked_y0(self, y0: float | None) -> None:
        # 设置 LockableBbox 的 y0 边界值的方法

    @property
    def locked_x1(self) -> float | None:
        # 返回 LockableBbox 的 x1 边界值或 None 的属性

    @locked_x1.setter
    def locked_x1(self, x1: float | None) -> None:
        # 设置 LockableBbox 的 x1 边界值的方法

    @property
    def locked_y1(self) -> float | None:
        # 返回 LockableBbox 的 y1 边界值或 None 的属性

    @locked_y1.setter
    def locked_y1(self, y1: float | None) -> None:
        # 设置 LockableBbox 的 y1 边界值的方法


class Transform(TransformNode):

    @property
    def input_dims(self) -> int | None:
        # 返回输入维度的属性，可能为整数或 None

    @property
    def output_dims(self) -> int | None:
        # 返回输出维度的属性，可能为整数或 None

    @property
    def is_separable(self) -> bool:
        # 返回是否可分离的属性，布尔值

    @property
    def has_inverse(self) -> bool:
        # 返回是否有逆变换的属性，布尔值

    def __add__(self, other: Transform) -> Transform:
        # 实现两个 Transform 对象相加的方法，返回一个新的 Transform 对象

    @property
    def depth(self) -> int:
        # 返回 Transform 对象的深度属性，整数值

    def contains_branch(self, other: Transform) -> bool:
        # 检查当前 Transform 对象是否包含另一个 Transform 对象的方法，返回布尔值

    def contains_branch_separately(
        self, other_transform: Transform
    ) -> Sequence[bool]:
        # 分别检查当前 Transform 对象是否分别包含另一个 Transform 对象的方法，返回布尔值序列
    # 定义魔术方法 `__sub__`，用于实现减法操作，参数 `other` 是一个 `Transform` 类型的对象
    def __sub__(self, other: Transform) -> Transform: ...
    
    # 定义魔术方法 `__array__`，返回一个 `np.ndarray` 数组，支持数组转换操作，参数 `args` 和 `kwargs` 是可变参数
    def __array__(self, *args, **kwargs) -> np.ndarray: ...
    
    # 将输入的 `values` 转换为 `np.ndarray` 类型的数组，并返回结果
    def transform(self, values: ArrayLike) -> np.ndarray: ...
    
    # 对于仿射变换，将输入的 `values` 转换为 `np.ndarray` 类型的数组，并返回结果
    def transform_affine(self, values: ArrayLike) -> np.ndarray: ...
    
    # 对于非仿射变换，将输入的 `values` 转换为 `ArrayLike` 类型，并返回结果
    def transform_non_affine(self, values: ArrayLike) -> ArrayLike: ...
    
    # 对输入的边界框 `bbox` 进行变换，并返回类型为 `Bbox` 的对象
    def transform_bbox(self, bbox: BboxBase) -> Bbox: ...
    
    # 返回当前对象的仿射变换对象 `Transform`
    def get_affine(self) -> Transform: ...
    
    # 返回描述当前对象变换矩阵的 `np.ndarray` 数组
    def get_matrix(self) -> np.ndarray: ...
    
    # 对输入的点 `point` 进行变换，并返回一个 `np.ndarray` 数组
    def transform_point(self, point: ArrayLike) -> np.ndarray: ...
    
    # 对输入的路径 `path` 进行变换，并返回一个 `Path` 对象
    def transform_path(self, path: Path) -> Path: ...
    
    # 对输入的路径 `path` 进行仿射变换，并返回一个 `Path` 对象
    def transform_path_affine(self, path: Path) -> Path: ...
    
    # 对输入的路径 `path` 进行非仿射变换，并返回一个 `Path` 对象
    def transform_path_non_affine(self, path: Path) -> Path: ...
    
    # 对输入的角度 `angles` 和点 `pts` 进行变换，返回一个 `np.ndarray` 数组
    # 参数 `radians` 表示角度是否以弧度制，`pushoff` 表示偏移量
    def transform_angles(
        self,
        angles: ArrayLike,
        pts: ArrayLike,
        radians: bool = ...,
        pushoff: float = ...,
    ) -> np.ndarray: ...
    
    # 返回当前变换的逆变换 `Transform` 对象
    def inverted(self) -> Transform: ...
class TransformWrapper(Transform):
    # TransformWrapper 类继承自 Transform 类，用于包装变换操作
    pass_through: bool
    # pass_through 属性，表示是否直接通过变换
    def __init__(self, child: Transform) -> None:
        # TransformWrapper 类的构造函数，接受一个 Transform 类型的参数 child
        ...

    def __eq__(self, other: object) -> bool:
        # 判断当前 TransformWrapper 对象是否等于另一个对象的方法
        ...

    def frozen(self) -> Transform:
        # 返回当前变换的冻结版本的方法
        ...

    def set(self, child: Transform) -> None:
        # 设置当前变换的子变换的方法
        ...


class AffineBase(Transform):
    # AffineBase 类继承自 Transform 类，表示仿射变换的基类
    is_affine: Literal[True]
    # is_affine 属性，用于标识该变换是否是仿射变换（字面值为 True）
    def __init__(self, *args, **kwargs) -> None:
        # AffineBase 类的构造函数，接受任意数量的位置参数和关键字参数
        ...

    def __eq__(self, other: object) -> bool:
        # 判断当前 AffineBase 对象是否等于另一个对象的方法
        ...


class Affine2DBase(AffineBase):
    # Affine2DBase 类继承自 AffineBase 类，表示二维仿射变换的基类
    input_dims: Literal[2]
    output_dims: Literal[2]
    # input_dims 和 output_dims 属性，分别表示输入和输出的维度（均为 2）
    def frozen(self) -> Affine2D:
        # 返回当前二维仿射变换的冻结版本的方法
        ...

    def to_values(self) -> tuple[float, float, float, float, float, float]:
        # 将当前二维仿射变换表示为其数值元组的方法
        ...


class Affine2D(Affine2DBase):
    # Affine2D 类继承自 Affine2DBase 类，表示二维仿射变换
    def __init__(self, matrix: ArrayLike | None = ..., **kwargs) -> None:
        # Affine2D 类的构造函数，接受一个数组或 None 的 matrix 参数以及任意数量的关键字参数
        ...

    @staticmethod
    def from_values(
        a: float, b: float, c: float, d: float, e: float, f: float
    ) -> Affine2D:
        # 根据给定的数值创建 Affine2D 对象的静态方法
        ...

    def set_matrix(self, mtx: ArrayLike) -> None:
        # 设置 Affine2D 对象的变换矩阵的方法
        ...

    def clear(self) -> Affine2D:
        # 清除 Affine2D 对象的变换操作的方法
        ...

    def rotate(self, theta: float) -> Affine2D:
        # 对 Affine2D 对象进行旋转变换的方法（角度为弧度制）
        ...

    def rotate_deg(self, degrees: float) -> Affine2D:
        # 对 Affine2D 对象进行旋转变换的方法（角度为度数制）
        ...

    def rotate_around(self, x: float, y: float, theta: float) -> Affine2D:
        # 绕指定点进行旋转变换的方法（角度为弧度制）
        ...

    def rotate_deg_around(self, x: float, y: float, degrees: float) -> Affine2D:
        # 绕指定点进行旋转变换的方法（角度为度数制）
        ...

    def translate(self, tx: float, ty: float) -> Affine2D:
        # 对 Affine2D 对象进行平移变换的方法
        ...

    def scale(self, sx: float, sy: float | None = ...) -> Affine2D:
        # 对 Affine2D 对象进行缩放变换的方法
        ...

    def skew(self, xShear: float, yShear: float) -> Affine2D:
        # 对 Affine2D 对象进行错切变换的方法（角度为弧度制）
        ...

    def skew_deg(self, xShear: float, yShear: float) -> Affine2D:
        # 对 Affine2D 对象进行错切变换的方法（角度为度数制）
        ...


class IdentityTransform(Affine2DBase):
    # IdentityTransform 类继承自 Affine2DBase 类，表示恒等变换
    ...


class _BlendedMixin:
    # _BlendedMixin 类，混合类，提供了一些混合变换的方法
    def __eq__(self, other: object) -> bool:
        # 判断当前混合变换对象是否等于另一个对象的方法
        ...

    def contains_branch_seperately(self, transform: Transform) -> Sequence[bool]:
        # 判断当前混合变换对象是否分别包含给定变换的方法
        ...


class BlendedGenericTransform(_BlendedMixin, Transform):
    # BlendedGenericTransform 类继承自 _BlendedMixin 和 Transform 类，表示通用混合变换
    input_dims: Literal[2]
    output_dims: Literal[2]
    pass_through: bool
    # input_dims、output_dims 和 pass_through 属性，分别表示输入输出维度和是否直接通过变换
    def __init__(
        self, x_transform: Transform, y_transform: Transform, **kwargs
    ) -> None:
        # BlendedGenericTransform 类的构造函数，接受两个 Transform 类型的参数以及任意数量的关键字参数
        ...

    @property
    def depth(self) -> int:
        # 返回混合变换的深度的属性
        ...

    def contains_branch(self, other: Transform) -> Literal[False]:
        # 判断当前混合变换是否包含给定变换的方法（总是返回 False）
        ...

    @property
    def is_affine(self) -> bool:
        # 返回当前混合变换是否是仿射变换的属性
        ...


class BlendedAffine2D(_BlendedMixin, Affine2DBase):
    # BlendedAffine2D 类继承自 _BlendedMixin 和 Affine2DBase 类，表示混合二维仿射变换
    def __init__(
        self, x_transform: Transform, y_transform: Transform, **kwargs
    ) -> None:
        # BlendedAffine2D 类的构造函数，接受两个 Transform 类型的参数以及任意数量的关键字参数
        ...


def blended_transform_factory(
    x_transform: Transform, y_transform: Transform
) -> BlendedGenericTransform | BlendedAffine2D:
    # 工厂函数，根据给定的两个变换类型创建对应的混合变换对象
    ...


class CompositeGenericTransform(Transform):
    # CompositeGenericTransform 类继承自 Transform 类，表示通用复合变换
    pass_through: bool
    # pass_through 属性，表示是否直接通过变换
    def __init__(self, a: Transform, b: Transform, **kwargs) -> None:
        # CompositeGenericTransform 类的构造函数，接受两个 Transform 类型的参数以及任意数量的关键字参数
        ...


class CompositeAffine2D(Affine2DBase):
    # CompositeAffine2D 类继承自 Affine2DBase 类，表示复合二维仿射变换
    def __init__(self, a: Affine2DBase, b: Affine2DBase, **kwargs) -> None:
        # CompositeAffine2D 类的构造函数，接受两个 Affine2DBase 类型的参数以及任意数量的关键字参数
        ...

    @property
    def depth(self) -> int:
        # 返回复合变换的深度的属性
        ...


def composite_transform_factory(a: Transform, b: Transform) -> Transform:
    # 工厂函数，根据给定的两个变换类型创建对应的复合变换对象
    ...


class BboxTransform(Affine2DBase):
    # BboxTransform 类继承自 Affine2DBase 类，表示边界框到边界框的仿射变换
    def __init__(self, boxin:
    # 定义一个构造函数，初始化一个对象实例
    def __init__(self, boxout: BboxBase, **kwargs) -> None:
# 定义一个类 BboxTransformToMaxOnly，继承自 BboxTransformTo 类
class BboxTransformToMaxOnly(BboxTransformTo):
    ...

# 定义一个类 BboxTransformFrom，继承自 Affine2DBase 类
class BboxTransformFrom(Affine2DBase):
    # 初始化方法，接受一个 BboxBase 对象作为输入和其他关键字参数
    def __init__(self, boxin: BboxBase, **kwargs) -> None:
        ...

# 定义一个类 ScaledTranslation，继承自 Affine2DBase 类
class ScaledTranslation(Affine2DBase):
    # 初始化方法，接受 xt 和 yt 作为平移的 x 和 y 偏移量，以及一个 scale_trans 的 Affine2DBase 对象和其他关键字参数
    def __init__(
        self, xt: float, yt: float, scale_trans: Affine2DBase, **kwargs
    ) -> None:
        ...

# 定义一个类 AffineDeltaTransform，继承自 Affine2DBase 类
class AffineDeltaTransform(Affine2DBase):
    # 初始化方法，接受一个 transform 的 Affine2DBase 对象和其他关键字参数
    def __init__(self, transform: Affine2DBase, **kwargs) -> None:
        ...

# 定义一个类 TransformedPath，继承自 TransformNode 类
class TransformedPath(TransformNode):
    # 初始化方法，接受一个 path 的 Path 对象和一个 transform 的 Transform 对象
    def __init__(self, path: Path, transform: Transform) -> None:
        ...

    # 返回转换后的路径和仿射变换的元组
    def get_transformed_points_and_affine(self) -> tuple[Path, Transform]:
        ...

    # 返回转换后的路径和仿射变换的元组
    def get_transformed_path_and_affine(self) -> tuple[Path, Transform]:
        ...

    # 返回完全转换后的路径
    def get_fully_transformed_path(self) -> Path:
        ...

    # 返回仿射变换对象
    def get_affine(self) -> Transform:
        ...

# 定义一个类 TransformedPatchPath，继承自 TransformedPath 类
class TransformedPatchPath(TransformedPath):
    # 初始化方法，接受一个 patch 的 Patch 对象
    def __init__(self, patch: Patch) -> None:
        ...

# 定义一个函数 nonsingular，接受 vmin、vmax 作为浮点数参数，还有 expander、tiny 和 increasing 作为可选参数
# 返回一个包含两个浮点数的元组
def nonsingular(
    vmin: float,
    vmax: float,
    expander: float = ...,
    tiny: float = ...,
    increasing: bool = ...,
) -> tuple[float, float]:
    ...

# 定义一个函数 interval_contains，接受一个包含两个浮点数的元组 interval 和一个浮点数 val
# 返回一个布尔值表示 val 是否在 interval 范围内
def interval_contains(interval: tuple[float, float], val: float) -> bool:
    ...

# 定义一个函数 interval_contains_open，接受一个包含两个浮点数的元组 interval 和一个浮点数 val
# 返回一个布尔值表示 val 是否在 interval 范围内（不包括端点）
def interval_contains_open(interval: tuple[float, float], val: float) -> bool:
    ...

# 定义一个函数 offset_copy，接受一个 trans 的 Transform 对象，一个可选的 Figure 对象 fig，
# 以及 x 和 y 作为浮点数的偏移量，units 表示偏移量的单位，可以是 "inches"、"points" 或 "dots"
# 返回一个新的 Transform 对象
def offset_copy(
    trans: Transform,
    fig: Figure | None = ...,
    x: float = ...,
    y: float = ...,
    units: Literal["inches", "points", "dots"] = ...,
) -> Transform:
    ...
```