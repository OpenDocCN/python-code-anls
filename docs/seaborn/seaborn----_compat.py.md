# `D:\src\scipysrc\seaborn\seaborn\_compat.py`

```
# 引入未来版本的注释支持，用于类型提示的字面值类型
from __future__ import annotations
# 引入字面值类型
from typing import Literal

# 引入常用库
import numpy as np
import pandas as pd
import matplotlib as mpl
# 从 matplotlib 中引入 Figure 类
from matplotlib.figure import Figure
# 从 seaborn.utils 中引入 _version_predates 函数
from seaborn.utils import _version_predates


def norm_from_scale(scale, norm):
    """根据给定的 Scale 和规范化的最小/最大值范围创建 Normalize 对象。"""
    # 如果已经是 Normalize 对象，则直接返回
    if isinstance(norm, mpl.colors.Normalize):
        return norm

    # 如果 Scale 为 None，则返回 None
    if scale is None:
        return None

    # 如果 norm 为 None，则 vmin 和 vmax 均为 None
    if norm is None:
        vmin = vmax = None
    else:
        vmin, vmax = norm  # 如果此处失败，考虑输出更有帮助的错误信息？

    # 定义一个继承自 mpl.colors.Normalize 的子类 ScaledNorm
    class ScaledNorm(mpl.colors.Normalize):

        def __call__(self, value, clip=None):
            # 从 matplotlib 的源码中处理值的转换
            value, is_scalar = self.process_value(value)
            self.autoscale_None(value)
            if self.vmin > self.vmax:
                raise ValueError("vmin must be less or equal to vmax")
            if self.vmin == self.vmax:
                return np.full_like(value, 0)
            if clip is None:
                clip = self.clip
            if clip:
                value = np.clip(value, self.vmin, self.vmax)
            # ***** Seaborn changes start ****
            # 转换数据并处理 vmin 和 vmax
            t_value = self.transform(value).reshape(np.shape(value))
            t_vmin, t_vmax = self.transform([self.vmin, self.vmax])
            # ***** Seaborn changes end *****
            # 如果 vmin 或 vmax 无效，则抛出 ValueError
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError("Invalid vmin or vmax")
            t_value -= t_vmin
            t_value /= (t_vmax - t_vmin)
            t_value = np.ma.masked_invalid(t_value, copy=False)
            return t_value[0] if is_scalar else t_value

    # 创建 ScaledNorm 对象
    new_norm = ScaledNorm(vmin, vmax)
    # 设置 transform 属性
    new_norm.transform = scale.get_transform().transform

    return new_norm


def get_colormap(name):
    """处理 matplotlib 3.6 中对 colormap 接口的更改。"""
    try:
        return mpl.colormaps[name]
    except AttributeError:
        return mpl.cm.get_cmap(name)


def register_colormap(name, cmap):
    """处理 matplotlib 3.6 中对 colormap 接口的更改。"""
    try:
        # 如果 colormap 名称不在注册的 colormap 中，则注册新的 colormap
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)
    except AttributeError:
        # 否则，注册指定名称的 colormap
        mpl.cm.register_cmap(name, cmap)


def set_layout_engine(
    fig: Figure,
    engine: Literal["constrained", "compressed", "tight", "none"],
) -> None:
    """处理 matplotlib 3.6 中自动布局引擎接口的更改。"""
    # 如果 fig 对象具有 set_layout_engine 方法，则设置布局引擎
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(engine)
    else:
        # 如果 matplotlib 的版本早于 3.6，则执行以下操作
        if engine == "tight":
            # 设置紧凑布局为 True，忽略类型检查，因为版本较早
            fig.set_tight_layout(True)  # type: ignore  # predates typing
        elif engine == "constrained":
            # 设置约束布局为 True，忽略类型检查
            fig.set_constrained_layout(True)  # type: ignore
        elif engine == "none":
            # 设置紧凑布局和约束布局均为 False，忽略类型检查
            fig.set_tight_layout(False)  # type: ignore
            fig.set_constrained_layout(False)  # type: ignore
# 获取图形对象的布局引擎，如果不存在则返回 None
def get_layout_engine(fig: Figure) -> mpl.layout_engine.LayoutEngine | None:
    """Handle changes to auto layout engine interface in 3.6"""
    # 检查图形对象是否具有 get_layout_engine 方法
    if hasattr(fig, "get_layout_engine"):
        # 调用图形对象的 get_layout_engine 方法获取布局引擎
        return fig.get_layout_engine()
    else:
        # 如果版本早于 3.6，则返回 None
        # _version_predates(mpl, 3.6)
        return None


# 处理动态创建共享坐标轴的变化
def share_axis(ax0, ax1, which):
    """Handle changes to post-hoc axis sharing."""
    # 如果版本早于 3.5
    if _version_predates(mpl, "3.5"):
        # 获取 ax0 对象的共享坐标轴组
        group = getattr(ax0, f"get_shared_{which}_axes")()
        # 将 ax1 加入到 ax0 的共享坐标轴组中
        group.join(ax1, ax0)
    else:
        # 否则，调用 ax1 的 share{which} 方法共享 ax0 的坐标轴
        getattr(ax1, f"share{which}")(ax0)


# 处理图例的 legendHandles 属性重命名
def get_legend_handles(legend):
    """Handle legendHandles attribute rename."""
    # 如果版本早于 3.7
    if _version_predates(mpl, "3.7"):
        # 返回图例对象的 legendHandles 属性
        return legend.legendHandles
    else:
        # 否则，返回图例对象的 legend_handles 属性
        return legend.legend_handles


# 根据版本处理包含组的分组应用
def groupby_apply_include_groups(val):
    # 如果版本早于 Pandas 2.2.0
    if _version_predates(pd, "2.2.0"):
        # 返回空字典
        return {}
    else:
        # 否则，返回包含 "include_groups" 键的字典，值为 val
        return {"include_groups": val}
```