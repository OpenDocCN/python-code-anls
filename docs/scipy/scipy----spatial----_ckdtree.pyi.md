# `D:\src\scipysrc\scipy\scipy\spatial\_ckdtree.pyi`

```
# 从未来模块导入注释
from __future__ import annotations
# 导入必要的类型
from typing import (
    Any,
    Generic,
    overload,
    TypeVar,
)

# 导入 numpy 库及其类型注释
import numpy as np
import numpy.typing as npt
# 导入稀疏矩阵相关模块
from scipy.sparse import coo_matrix, dok_matrix

# 导入 Literal 类型提示
from typing import Literal

# TODO: 当可能时，用 1D float64 数组替换 `ndarray`
_BoxType = TypeVar("_BoxType", None, npt.NDArray[np.float64])

# 从 `numpy.typing._scalar_like._ScalarLike` 复制的类型别名
# TODO: 一旦支持形状，请扩展为包含 0D 数组
_ArrayLike0D = bool | int | float | complex | str | bytes | np.generic

# _WeightType 类型注释，可以是数组或数组元组
_WeightType = npt.ArrayLike | tuple[npt.ArrayLike | None, npt.ArrayLike | None]

# cKDTreeNode 类的定义，描述 KD 树节点的属性
class cKDTreeNode:
    @property
    def data_points(self) -> npt.NDArray[np.float64]: ...
    @property
    def indices(self) -> npt.NDArray[np.intp]: ...

    # 在 Cython 中这些是只读属性，行为类似于属性
    @property
    def level(self) -> int: ...
    @property
    def split_dim(self) -> int: ...
    @property
    def children(self) -> int: ...
    @property
    def start_idx(self) -> int: ...
    @property
    def end_idx(self) -> int: ...
    @property
    def split(self) -> float: ...
    @property
    def lesser(self) -> cKDTreeNode | None: ...
    @property
    def greater(self) -> cKDTreeNode | None: ...

# cKDTree 泛型类的定义，描述 KD 树的属性和方法
class cKDTree(Generic[_BoxType]):
    @property
    def n(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def leafsize(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def tree(self) -> cKDTreeNode: ...

    # 在 Cython 中这些是只读属性，行为类似于属性
    @property
    def data(self) -> npt.NDArray[np.float64]: ...
    @property
    def maxes(self) -> npt.NDArray[np.float64]: ...
    @property
    def mins(self) -> npt.NDArray[np.float64]: ...
    @property
    def indices(self) -> npt.NDArray[np.float64]: ...
    @property
    def boxsize(self) -> _BoxType: ...

    # 注意：实际上 `__init__` 作为构造函数使用，而不是 `__new__`
    # 后者在设置泛型参数时更加灵活。
    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        data: npt.ArrayLike,
        leafsize: int = ...,
        compact_nodes: bool = ...,
        copy_data: bool = ...,
        balanced_tree: bool = ...,
        boxsize: None = ...,
    ) -> cKDTree[None]: ...
    @overload
    def __new__(
        cls,
        data: npt.ArrayLike,
        leafsize: int = ...,
        compact_nodes: bool = ...,
        copy_data: bool = ...,
        balanced_tree: bool = ...,
        boxsize: npt.ArrayLike = ...,
    ) -> cKDTree[npt.NDArray[np.float64]]: ...

    # TODO: 如果 `x.ndim == 1` 且 `k == 1`，返回一个包含两个标量的二元组，
    # 否则返回一个包含两个数组的二元组
    # 定义一个函数签名，用于查询操作
    def query(
        self,
        x: npt.ArrayLike,
        k: npt.ArrayLike = ...,
        eps: float = ...,
        p: float = ...,
        distance_upper_bound: float = ...,
        workers: int | None = ...,
    ) -> tuple[Any, Any]: ...

    # TODO: 如果 `x.ndim <= 1`，返回一个标量列表；否则返回一个对象数组的列表
    def query_ball_point(
        self,
        x: npt.ArrayLike,
        r: npt.ArrayLike,
        p: float = ...,
        eps: float = ...,
        workers: int | None = ...,
        return_sorted: bool | None = ...,
        return_length: bool = ...
    ) -> Any: ...

    # 定义一个函数签名，用于在给定距离范围内查询球形邻域
    def query_ball_tree(
        self,
        other: cKDTree,
        r: float,
        p: float,
        eps: float = ...,
    ) -> list[list[int]]: ...

    @overload
    def query_pairs(  # type: ignore[overload-overlap]
        self,
        r: float,
        p: float = ...,
        eps: float = ...,
        output_type: Literal["set"] = ...,
    ) -> set[tuple[int, int]]: ...
    @overload
    def query_pairs(
        self,
        r: float,
        p: float = ...,
        eps: float = ...,
        output_type: Literal["ndarray"] = ...,
    ) -> npt.NDArray[np.intp]: ...

    @overload
    def count_neighbors(  # type: ignore[overload-overlap]
        self,
        other: cKDTree,
        r: _ArrayLike0D,
        p: float = ...,
        weights: None | tuple[None, None] = ...,
        cumulative: bool = ...,
    ) -> int: ...
    @overload
    def count_neighbors(  # type: ignore[overload-overlap]
        self,
        other: cKDTree,
        r: _ArrayLike0D,
        p: float = ...,
        weights: _WeightType = ...,
        cumulative: bool = ...,
    ) -> np.float64: ...
    @overload
    def count_neighbors(  # type: ignore[overload-overlap]
        self,
        other: cKDTree,
        r: npt.ArrayLike,
        p: float = ...,
        weights: None | tuple[None, None] = ...,
        cumulative: bool = ...,
    ) -> npt.NDArray[np.intp]: ...
    @overload
    def count_neighbors(
        self,
        other: cKDTree,
        r: npt.ArrayLike,
        p: float = ...,
        weights: _WeightType = ...,
        cumulative: bool = ...,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def sparse_distance_matrix(  # type: ignore[overload-overlap]
        self,
        other: cKDTree,
        max_distance: float,
        p: float = ...,
        output_type: Literal["dok_matrix"] = ...,
    ) -> dok_matrix: ...
    @overload
    def sparse_distance_matrix(  # type: ignore[overload-overlap]
        self,
        other: cKDTree,
        max_distance: float,
        p: float = ...,
        output_type: Literal["coo_matrix"] = ...,
    ) -> coo_matrix: ...
    @overload
    def sparse_distance_matrix(  # type: ignore[overload-overlap]
        self,
        other: cKDTree,
        max_distance: float,
        p: float = ...,
        output_type: Literal["dict"] = ...,
    ) -> dict[tuple[int, int], float]: ...
    @overload
    # 定义一个方法 sparse_distance_matrix，用于计算稀疏距离矩阵
    # self: 当前对象的引用，通常是 cKDTree 的一个实例
    # other: 另一个 cKDTree 对象，表示计算距离的目标
    # max_distance: 允许的最大距离阈值，超过这个距离的点对将不会被考虑
    # p: 距离度量的参数，默认为省略号，应为一个浮点数，控制距离的计算方式
    # output_type: 返回值的类型，仅支持 "ndarray"，表示返回一个 NumPy 数组
    ) -> npt.NDArray[np.void]:
        # 这是一个占位符函数声明，实际实现并未提供在这里
        pass
```