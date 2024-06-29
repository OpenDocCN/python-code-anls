# `D:\src\scipysrc\matplotlib\lib\matplotlib\_pylab_helpers.pyi`

```py
# 导入必要的模块和类
from collections import OrderedDict  # 导入有序字典类
from matplotlib.backend_bases import FigureManagerBase  # 导入图形管理基类
from matplotlib.figure import Figure  # 导入图形类 Figure

# 定义 Gcf 类，用于管理图形
class Gcf:
    figs: OrderedDict[int, FigureManagerBase]  # figs 属性，存储图形编号到图形管理器的有序字典

    @classmethod
    def get_fig_manager(cls, num: int) -> FigureManagerBase | None:
        # 类方法：获取指定编号的图形管理器或者返回 None
        ...

    @classmethod
    def destroy(cls, num: int | FigureManagerBase) -> None:
        # 类方法：销毁指定编号或图形管理器对象的图形
        ...

    @classmethod
    def destroy_fig(cls, fig: Figure) -> None:
        # 类方法：销毁指定图形对象的图形
        ...

    @classmethod
    def destroy_all(cls) -> None:
        # 类方法：销毁所有图形
        ...

    @classmethod
    def has_fignum(cls, num: int) -> bool:
        # 类方法：检查是否存在指定编号的图形
        ...

    @classmethod
    def get_all_fig_managers(cls) -> list[FigureManagerBase]:
        # 类方法：获取所有图形管理器对象的列表
        ...

    @classmethod
    def get_num_fig_managers(cls) -> int:
        # 类方法：获取当前图形管理器对象的数量
        ...

    @classmethod
    def get_active(cls) -> FigureManagerBase | None:
        # 类方法：获取当前活动的图形管理器对象，或者返回 None
        ...

    @classmethod
    def _set_new_active_manager(cls, manager: FigureManagerBase) -> None:
        # 类方法：设置新的活动图形管理器对象（私有方法）
        ...

    @classmethod
    def set_active(cls, manager: FigureManagerBase) -> None:
        # 类方法：设置指定的图形管理器对象为活动对象
        ...

    @classmethod
    def draw_all(cls, force: bool = ...) -> None:
        # 类方法：绘制所有图形（可选择强制刷新）
        ...
```