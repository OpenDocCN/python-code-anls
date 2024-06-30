# `D:\src\scipysrc\seaborn\seaborn\_core\moves.py`

```
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, cast

import numpy as np
from pandas import DataFrame

from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.typing import Default

default = Default()

@dataclass
class Move:
    """Base class for objects that apply simple positional transforms."""

    group_by_orient: ClassVar[bool] = True

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        """Apply transformation to data based on specific orientation."""
        # This method is intended to be overridden by subclasses
        raise NotImplementedError


@dataclass
class Jitter(Move):
    """
    Random displacement along one or both axes to reduce overplotting.

    Parameters
    ----------
    width : float
        Magnitude of jitter, relative to mark width, along the orientation axis.
        If not provided, the default value will be 0 when `x` or `y` are set, otherwise
        there will be a small amount of jitter applied by default.
    x : float
        Magnitude of jitter, in data units, along the x axis.
    y : float
        Magnitude of jitter, in data units, along the y axis.

    Examples
    --------
    .. include:: ../docstrings/objects.Jitter.rst

    """
    width: float | Default = default
    x: float = 0
    y: float = 0
    seed: int | None = None

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        """Apply jitter transformation to data."""
        # Make a copy of the input data to avoid modifying the original
        data = data.copy()
        # Initialize random number generator with optional seed
        rng = np.random.default_rng(self.seed)

        def jitter(data, col, scale):
            """Apply jitter noise to a specific column."""
            noise = rng.uniform(-.5, +.5, len(data))
            offsets = noise * scale
            return data[col] + offsets

        # Determine the width of jitter based on provided or default values
        if self.width is default:
            width = 0.0 if self.x or self.y else 0.2
        else:
            width = cast(float, self.width)

        # Apply jitter transformation based on provided parameters
        if self.width:
            data[orient] = jitter(data, orient, width * data["width"])
        if self.x:
            data["x"] = jitter(data, "x", self.x)
        if self.y:
            data["y"] = jitter(data, "y", self.y)

        return data


@dataclass
class Dodge(Move):
    """
    Displacement and narrowing of overlapping marks along orientation axis.

    Parameters
    ----------
    empty : {'keep', 'drop', 'fill'}
        Action to take with dodged marks.
    gap : float
        Size of gap between dodged marks.
    by : list of variable names, optional
        Variables to apply the movement to, otherwise use all.

    Examples
    --------
    .. include:: ../docstrings/objects.Dodge.rst

    """
    empty: str = "keep"  # Options: keep, drop, fill
    gap: float = 0
    by: Optional[list[str]] = None

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        """Apply dodge transformation to data."""
        # This method is intended to be overridden by subclasses
        raise NotImplementedError
        # 定义一个函数，返回值为 DataFrame 类型
        ) -> DataFrame:

            # 根据 groupby 对象中指定的顺序，筛选出数据中存在的分组变量
            grouping_vars = [v for v in groupby.order if v in data]
            # 对数据进行聚合，计算各分组的最大宽度
            groups = groupby.agg(data, {"width": "max"})
            # 如果设定为空值时填充，则删除包含空值的分组
            if self.empty == "fill":
                groups = groups.dropna()

            # 定义一个内部函数，用于按照特定标准分组数据
            def groupby_pos(s):
                # 根据指定的方向、列和行分组数据
                grouper = [groups[v] for v in [orient, "col", "row"] if v in data]
                return s.groupby(grouper, sort=False, observed=True)

            # 定义一个内部函数，用于缩放宽度值
            def scale_widths(w):
                # TODO 如何处理缺失宽度值？这是一个复杂的问题...
                # 如果选择填充空值，则使用平均值填充，否则填充为 0
                empty = 0 if self.empty == "fill" else w.mean()
                filled = w.fillna(empty)
                scale = filled.max()
                norm = filled.sum()
                if self.empty == "keep":
                    w = filled
                return w / norm * scale

            # 定义一个内部函数，用于将宽度转换为偏移量
            def widths_to_offsets(w):
                # 计算每个宽度的偏移量，并确保起始偏移量为 0
                return w.shift(1).fillna(0).cumsum() + (w - w.sum()) / 2

            # 对分组后的宽度应用缩放函数
            new_widths = groupby_pos(groups["width"]).transform(scale_widths)
            # 对缩放后的宽度值计算偏移量
            offsets = groupby_pos(new_widths).transform(widths_to_offsets)

            # 如果设置了间隔参数，则进一步调整新宽度值
            if self.gap:
                new_widths *= 1 - self.gap

            # 计算分组位置偏移量并更新宽度值
            groups["_dodged"] = groups[orient] + offsets
            groups["width"] = new_widths

            # 将处理后的数据与原始数据合并，根据指定的分组变量进行左连接
            out = (
                data
                .drop("width", axis=1)
                .merge(groups, on=grouping_vars, how="left")
                .drop(orient, axis=1)
                .rename(columns={"_dodged": orient})
            )

            # 返回合并后的结果
            return out
@dataclass
class Stack(Move):
    """
    Displacement of overlapping bar or area marks along the value axis.

    Examples
    --------
    .. include:: ../docstrings/objects.Stack.rst

    """

    # TODO center? (or should this be a different move, eg. Stream())

    def _stack(self, df, orient):
        """
        Perform stacking operation on DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input data to be stacked.
        orient : str
            Orientation of stacking ('x' or 'y').

        Returns
        -------
        DataFrame
            Modified DataFrame after stacking operation.
        """

        # TODO should stack do something with ymin/ymax style marks?
        # Should there be an upstream conversion to baseline/height parameterization?

        if df["baseline"].nunique() > 1:
            err = "Stack move cannot be used when baselines are already heterogeneous"
            raise RuntimeError(err)

        other = {"x": "y", "y": "x"}[orient]
        stacked_lengths = (df[other] - df["baseline"]).dropna().cumsum()
        offsets = stacked_lengths.shift(1).fillna(0)

        df[other] = stacked_lengths
        df["baseline"] = df["baseline"] + offsets

        return df

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        """
        Apply stacking operation on grouped data.

        Parameters
        ----------
        data : DataFrame
            Input data to be stacked.
        groupby : GroupBy
            Grouping object defining how to group the data.
        orient : str
            Orientation of stacking ('x' or 'y').
        scales : dict[str, Scale]
            Scales used for the stacking operation.

        Returns
        -------
        DataFrame
            Modified DataFrame after stacking operation.
        """

        # TODO where to ensure that other semantic variables are sorted properly?
        # TODO why are we not using the passed in groupby here?
        groupers = ["col", "row", orient]
        return GroupBy(groupers).apply(data, self._stack, orient)


@dataclass
class Shift(Move):
    """
    Displacement of all marks with the same magnitude / direction.

    Parameters
    ----------
    x, y : float
        Magnitude of shift, in data units, along each axis.

    Examples
    --------
    .. include:: ../docstrings/objects.Shift.rst

    """

    x: float = 0
    y: float = 0

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        """
        Apply shifting operation on DataFrame.

        Parameters
        ----------
        data : DataFrame
            Input data to be shifted.
        groupby : GroupBy
            Grouping object defining how to group the data.
        orient : str
            Orientation of shifting ('x' or 'y').
        scales : dict[str, Scale]
            Scales used for the shifting operation.

        Returns
        -------
        DataFrame
            Modified DataFrame after shifting operation.
        """

        data = data.copy(deep=False)
        data["x"] = data["x"] + self.x
        data["y"] = data["y"] + self.y
        return data


@dataclass
class Norm(Move):
    """
    Divisive scaling on the value axis after aggregating within groups.

    Parameters
    ----------
    func : str or callable
        Function called on each group to define the comparison value.
    where : str
        Query string defining the subset used to define the comparison values.
    by : list of variables
        Variables used to define aggregation groups.
    percent : bool
        If True, multiply the result by 100.

    Examples
    --------
    .. include:: ../docstrings/objects.Norm.rst

    """

    func: Union[Callable, str] = "max"
    where: Optional[str] = None
    by: Optional[list[str]] = None
    percent: bool = False

    group_by_orient: ClassVar[bool] = False

    def _norm(self, df, var):
        """
        Apply normalization operation on DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input data to be normalized.
        var : str
            Variable on which normalization is applied.

        Returns
        -------
        DataFrame
            Modified DataFrame after normalization operation.
        """

        if self.where is None:
            denom_data = df[var]
        else:
            denom_data = df.query(self.where)[var]
        df[var] = df[var] / denom_data.agg(self.func)

        if self.percent:
            df[var] = df[var] * 100

        return df
    # 定义一个特殊方法 __call__，使得对象可以像函数一样被调用
    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        # 根据给定的 orient 参数选择另一个方向的值
        other = {"x": "y", "y": "x"}[orient]
        # 调用 groupby 对象的 apply 方法，应用 self._norm 函数到数据上，并传入另一个方向的值
        return groupby.apply(data, self._norm, other)
# TODO
# @dataclass
# class Ridge(Move):
#     ...
```