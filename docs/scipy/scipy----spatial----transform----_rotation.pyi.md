# `D:\src\scipysrc\scipy\scipy\spatial\transform\_rotation.pyi`

```
# 引入未来的注解语法支持，用于类型提示中的类型检查
from __future__ import annotations

# 引入类型检查时可能用到的模块
from typing import TYPE_CHECKING

# 引入用于表示序列抽象基类的模块
from collections.abc import Sequence

# 引入 NumPy 库，通常用于数值计算
import numpy as np

# 如果是类型检查环境，引入额外的 NumPy 类型提示模块
if TYPE_CHECKING:
    import numpy.typing as npt

# 定义一个整数类型，可以是 int 或者 np.integer 类型
_IntegerType = int | np.integer

# 定义一个表示旋转的类 Rotation
class Rotation:
    # 初始化方法，接受一个类似数组的四元数、是否标准化、是否复制的参数
    def __init__(self, quat: npt.ArrayLike, normalize: bool = ..., copy: bool = ...) -> None: ...

    # 属性方法，判断是否是单一旋转
    @property
    def single(self) -> bool: ...

    # 返回旋转对象的长度（通常指旋转的维度）
    def __len__(self) -> int: ...

    # 从四元数创建旋转对象的类方法，可选择标量优先
    @classmethod
    def from_quat(cls, quat: npt.ArrayLike, *, scalar_first: bool = ...) -> Rotation: ...

    # 从旋转矩阵创建旋转对象的类方法
    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike) -> Rotation: ...

    # 从旋转向量创建旋转对象的类方法，可选择是否角度为度数
    @classmethod
    def from_rotvec(cls, rotvec: npt.ArrayLike, degrees: bool = ...) -> Rotation: ...

    # 从欧拉角序列创建旋转对象的类方法，可以指定是否角度为度数
    @classmethod
    def from_euler(cls, seq: str, angles: float | npt.ArrayLike, degrees: bool = ...) -> Rotation: ...

    # 从达文波特方法创建旋转对象的类方法，指定轴、顺序、角度是否为度数
    @classmethod
    def from_davenport(cls, axes: npt.ArrayLike, order: str, angles: float | npt.ArrayLike, degrees: bool = ...) -> Rotation: ...

    # 从修正罗德里格斯参数（MRP）创建旋转对象的类方法
    @classmethod
    def from_mrp(cls, mrp: npt.ArrayLike) -> Rotation: ...

    # 将旋转对象转换为四元数的方法，可以选择是否标准化，是否标量优先
    def as_quat(self, canonical: bool = ..., *, scalar_first: bool = ...) -> np.ndarray: ...

    # 将旋转对象转换为旋转矩阵的方法
    def as_matrix(self) -> np.ndarray: ...

    # 将旋转对象转换为旋转向量的方法，可以选择是否角度为度数
    def as_rotvec(self, degrees: bool = ...) -> np.ndarray: ...

    # 将旋转对象转换为欧拉角的方法，指定序列和是否角度为度数
    def as_euler(self, seq: str, degrees: bool = ...) -> np.ndarray: ...

    # 将旋转对象转换为达文波特方法的方法，指定轴、顺序和是否角度为度数
    def as_davenport(self, axes: npt.ArrayLike, order: str, degrees: bool = ...) -> np.ndarray: ...

    # 将旋转对象转换为修正罗德里格斯参数（MRP）的方法
    def as_mrp(self) -> np.ndarray: ...

    # 类方法，连接一系列旋转对象，返回合并后的旋转对象
    @classmethod
    def concatenate(cls, rotations: Sequence[Rotation]) -> Rotation: ...

    # 应用旋转对象到向量的方法，可以选择是否反向
    def apply(self, vectors: npt.ArrayLike, inverse: bool = ...) -> np.ndarray: ...

    # 重载乘法运算符 *，用于将两个旋转对象相乘
    def __mul__(self, other: Rotation) -> Rotation: ...

    # 重载幂运算符 **，将旋转对象的幂应用到一个数值和模数上
    def __pow__(self, n: float, modulus: int | None) -> Rotation: ...

    # 返回旋转对象的逆
    def inv(self) -> Rotation: ...

    # 返回旋转对象的大小，通常指向量或标量
    def magnitude(self) -> np.ndarray | float: ...

    # 检查两个旋转对象是否近似相等，可以指定容差和角度是否为度数
    def approx_equal(self, other: Rotation, atol: float | None, degrees: bool = ...) -> np.ndarray | bool: ...

    # 计算一系列旋转对象的平均值，可以指定权重
    def mean(self, weights: npt.ArrayLike | None = ...) -> Rotation: ...

    # 将一系列旋转对象进行减少，可以指定左右对象、返回索引是否是索引三元组
    def reduce(self, left: Rotation | None = ..., right: Rotation | None = ...,
               return_indices: bool = ...) -> Rotation | tuple[Rotation, np.ndarray, np.ndarray]: ...

    # 创建一个特定组合的旋转对象，可以指定组合和轴
    @classmethod
    def create_group(cls, group: str, axis: str = ...) -> Rotation: ...

    # 获取旋转对象的索引，可以是整数、切片或类似数组
    def __getitem__(self, indexer: int | slice | npt.ArrayLike) -> Rotation: ...

    # 类方法，返回一个表示单位旋转的旋转对象，可以指定数量
    @classmethod
    def identity(cls, num: int | None = ...) -> Rotation: ...

    # 类方法，返回随机生成的旋转对象，可以指定数量和随机种子
    @classmethod
    def random(cls, num: int | None = ...,
               random_state: _IntegerType | np.random.Generator | np.random.RandomState | None = ...) -> Rotation: ...

    # 类方法，对齐向量生成旋转对象，可以指定权重和是否返回敏感性
    @classmethod
    def align_vectors(cls, a: npt.ArrayLike, b: npt.ArrayLike,
                      weights: npt.ArrayLike | None = ...,
                      return_sensitivity: bool = ...) -> tuple[Rotation, float] | tuple[Rotation, float, np.ndarray]: ...

# 定义 Slerp 类
class Slerp:
    # 成员变量，表示时间数组
    times: np.ndarray
    # 成员变量，表示时间增量数组
    timedelta: np.ndarray
    # 成员变量，表示旋转对象
    rotations: Rotation
    # 定义一个类成员变量 rotvecs，类型为 numpy 的 ndarray，用于存储旋转向量数据
    rotvecs: np.ndarray

    # 构造函数初始化方法，接受两个参数：times 作为时间数组，rotations 作为旋转对象
    def __init__(self, times: npt.ArrayLike, rotations: Rotation) -> None:
        ...

    # 调用方法，接受一个参数 times 作为时间数组，返回一个 Rotation 对象
    def __call__(self, times: npt.ArrayLike) -> Rotation:
        ...
```