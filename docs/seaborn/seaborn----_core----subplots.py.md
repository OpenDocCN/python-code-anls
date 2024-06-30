# `D:\src\scipysrc\seaborn\seaborn\_core\subplots.py`

```
from __future__ import annotations
from collections.abc import Generator

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 如果类型检查为真，则导入以下类型
    from seaborn._core.plot import FacetSpec, PairSpec
    from matplotlib.figure import SubFigure

class Subplots:
    """
    Interface for creating and using matplotlib subplots based on seaborn parameters.

    Parameters
    ----------
    subplot_spec : dict
        Keyword args for :meth:`matplotlib.figure.Figure.subplots`.
    facet_spec : dict
        Parameters that control subplot faceting.
    pair_spec : dict
        Parameters that control subplot pairing.
    data : PlotData
        Data used to define figure setup.

    """
    def __init__(
        self,
        subplot_spec: dict,  # 定义子图规格为字典类型
        facet_spec: FacetSpec,  # 控制面板切面的参数
        pair_spec: PairSpec,    # 控制子图配对的参数
    ):

        self.subplot_spec = subplot_spec  # 设置实例变量为传入的 subplot_spec

        self._check_dimension_uniqueness(facet_spec, pair_spec)  # 检查维度的唯一性
        self._determine_grid_dimensions(facet_spec, pair_spec)  # 确定网格的维度
        self._handle_wrapping(facet_spec, pair_spec)  # 处理包装操作
        self._determine_axis_sharing(pair_spec)  # 确定轴的共享策略

    def _check_dimension_uniqueness(
        self, facet_spec: FacetSpec, pair_spec: PairSpec
    ) -> None:
        """Reject specs that pair and facet on (or wrap to) same figure dimension."""
        err = None  # 错误信息初始化为 None

        facet_vars = facet_spec.get("variables", {})  # 获取 facet_spec 中的变量

        # 检查是否同时指定了 `col` 和 `row` 时，无法进行包装操作
        if facet_spec.get("wrap") and {"col", "row"} <= set(facet_vars):
            err = "Cannot wrap facets when specifying both `col` and `row`."
        # 检查是否同时在 `x` 和 `y` 上进行配对操作时，无法进行子图包装操作
        elif (
            pair_spec.get("wrap")
            and pair_spec.get("cross", True)
            and len(pair_spec.get("structure", {}).get("x", [])) > 1
            and len(pair_spec.get("structure", {}).get("y", [])) > 1
        ):
            err = "Cannot wrap subplots when pairing on both `x` and `y`."

        # 定义碰撞情况，检查在配对和切面操作中是否出现冲突
        collisions = {"x": ["columns", "rows"], "y": ["rows", "columns"]}
        for pair_axis, (multi_dim, wrap_dim) in collisions.items():
            if pair_axis not in pair_spec.get("structure", {}):
                continue
            elif multi_dim[:3] in facet_vars:
                err = f"Cannot facet the {multi_dim} while pairing on `{pair_axis}``."
            elif wrap_dim[:3] in facet_vars and facet_spec.get("wrap"):
                err = f"Cannot wrap the {wrap_dim} while pairing on `{pair_axis}``."
            elif wrap_dim[:3] in facet_vars and pair_spec.get("wrap"):
                err = f"Cannot wrap the {multi_dim} while faceting the {wrap_dim}."

        if err is not None:
            raise RuntimeError(err)  # 如果有错误信息，则抛出运行时错误
    ) -> None:
        """Parse faceting and pairing information to define figure structure."""
        # 初始化一个空字典，用于存储网格维度信息，列和行
        self.grid_dimensions: dict[str, list] = {}

        # 遍历列和行的维度
        for dim, axis in zip(["col", "row"], ["x", "y"]):

            # 获取facet_spec中的变量信息
            facet_vars = facet_spec.get("variables", {})

            # 如果当前维度在facet_vars中存在，则将其结构信息存入grid_dimensions中
            if dim in facet_vars:
                self.grid_dimensions[dim] = facet_spec["structure"][dim]
            # 如果当前维度不在facet_vars中，并且在pair_spec的结构信息中存在对应的轴，则创建一个与其长度相同的None列表
            elif axis in pair_spec.get("structure", {}):
                self.grid_dimensions[dim] = [
                    None for _ in pair_spec.get("structure", {})[axis]
                ]
            # 否则，创建一个单元素为None的列表
            else:
                self.grid_dimensions[dim] = [None]

            # 设置subplot_spec中的n{dim}s，即当前维度的长度
            self.subplot_spec[f"n{dim}s"] = len(self.grid_dimensions[dim])

        # 如果pair_spec中的cross为False，则将subplot_spec中的nrows设置为1
        if not pair_spec.get("cross", True):
            self.subplot_spec["nrows"] = 1

        # 计算子图的总数
        self.n_subplots = self.subplot_spec["ncols"] * self.subplot_spec["nrows"]

    def _handle_wrapping(
        self, facet_spec: FacetSpec, pair_spec: PairSpec
    ) -> None:
        """Update figure structure parameters based on facet/pair wrapping."""
        # 获取facet_spec或pair_spec中的wrap参数
        self.wrap = wrap = facet_spec.get("wrap") or pair_spec.get("wrap")
        
        # 如果wrap为False，则直接返回，不进行后续操作
        if not wrap:
            return

        # 确定wrap的维度和flow的维度
        wrap_dim = "row" if self.subplot_spec["nrows"] > 1 else "col"
        flow_dim = {"row": "col", "col": "row"}[wrap_dim]
        
        # 计算flow的值，即总子图数除以wrap的值向上取整
        n_subplots = self.subplot_spec[f"n{wrap_dim}s"]
        flow = int(np.ceil(n_subplots / wrap))

        # 如果wrap小于当前维度的长度，则将当前维度的长度设置为wrap
        if wrap < self.subplot_spec[f"n{wrap_dim}s"]:
            self.subplot_spec[f"n{wrap_dim}s"] = wrap
        
        # 设置flow维度的长度为flow
        self.subplot_spec[f"n{flow_dim}s"] = flow
        
        # 更新总子图数和wrap的维度信息
        self.n_subplots = n_subplots
        self.wrap_dim = wrap_dim

    def _determine_axis_sharing(self, pair_spec: PairSpec) -> None:
        """Update subplot spec with default or specified axis sharing parameters."""
        # 定义轴到维度的映射关系
        axis_to_dim = {"x": "col", "y": "row"}
        key: str
        val: str | bool
        
        # 遍历xy轴
        for axis in "xy":
            key = f"share{axis}"
            
            # 如果subplot_spec中不存在键为key的项，则根据pair_spec和wrap的情况确定默认的轴共享参数
            if key not in self.subplot_spec:
                if axis in pair_spec.get("structure", {}):
                    # 如果pair_spec中的cross为True且wrap为None，则默认共享一个维度
                    if self.wrap is None and pair_spec.get("cross", True):
                        val = axis_to_dim[axis]
                    else:
                        val = False
                else:
                    # 对于facet维度或者单个子图的情况，值设为True
                    val = True
                self.subplot_spec[key] = val

    def init_figure(
        self,
        pair_spec: PairSpec,
        pyplot: bool = False,
        figure_kws: dict | None = None,
        target: Axes | Figure | SubFigure | None = None,
    ):
        """Initialize the figure with optional plotting backend and parameters."""
        # 初始化图形，这里省略具体初始化的细节

    def __iter__(self) -> Generator[dict, None, None]:  # TODO TypedDict?
        """Yield each subplot dictionary with Axes object and metadata."""
        # 返回迭代器，用于迭代子图列表中的每个子图字典
        yield from self._subplot_list
    # 定义特殊方法 `__len__()`，用于返回该图形对象中子图的数量
    def __len__(self) -> int:
        """Return the number of subplots in this figure."""
        # 返回存储子图列表 `_subplot_list` 的长度，即子图的数量
        return len(self._subplot_list)
```