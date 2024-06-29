# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\parasite_axes.py`

```py
from matplotlib import _api, cbook  # 导入需要的模块
import matplotlib.artist as martist  # 导入matplotlib的artist模块，用于图形元素的管理
import matplotlib.transforms as mtransforms  # 导入matplotlib的transforms模块，用于坐标变换
from matplotlib.transforms import Bbox  # 导入matplotlib的Bbox类
from .mpl_axes import Axes  # 导入自定义的Axes类，前面的点表示相对路径

class ParasiteAxesBase:
    # 寄生坐标系基类

    def __init__(self, parent_axes, aux_transform=None,
                 *, viewlim_mode=None, **kwargs):
        # 初始化方法，接受父坐标轴、辅助变换、视图限制模式等参数
        self._parent_axes = parent_axes  # 设置父坐标轴
        self.transAux = aux_transform  # 设置辅助变换
        self.set_viewlim_mode(viewlim_mode)  # 设置视图限制模式
        kwargs["frameon"] = False  # 设置关键字参数frameon为False，不显示边框
        super().__init__(parent_axes.figure, parent_axes._position, **kwargs)

    def clear(self):
        # 清空方法
        super().clear()  # 调用父类的clear方法
        martist.setp(self.get_children(), visible=False)  # 设置所有子元素不可见
        self._get_lines = self._parent_axes._get_lines  # 获取父坐标轴的获取线条方法
        self._parent_axes.callbacks._connect_picklable(
            "xlim_changed", self._sync_lims)  # 连接父坐标轴的xlim_changed事件到_sync_lims方法
        self._parent_axes.callbacks._connect_picklable(
            "ylim_changed", self._sync_lims)  # 连接父坐标轴的ylim_changed事件到_sync_lims方法

    def pick(self, mouseevent):
        # 拾取方法，处理鼠标事件
        super().pick(mouseevent)  # 调用父类的pick方法处理鼠标事件
        # 处理寄生坐标系在其宿主坐标系上的拾取事件
        for a in self.get_children():
            if (hasattr(mouseevent.inaxes, "parasites")
                    and self in mouseevent.inaxes.parasites):
                a.pick(mouseevent)

    # aux_transform support

    def _set_lim_and_transforms(self):
        # 设置限制和变换方法
        if self.transAux is not None:
            self.transAxes = self._parent_axes.transAxes  # 设置坐标轴变换为父坐标轴的坐标轴变换
            self.transData = self.transAux + self._parent_axes.transData  # 设置数据变换
            self._xaxis_transform = mtransforms.blended_transform_factory(
                self.transData, self.transAxes)  # 设置X轴变换
            self._yaxis_transform = mtransforms.blended_transform_factory(
                self.transAxes, self.transData)  # 设置Y轴变换
        else:
            super()._set_lim_and_transforms()  # 调用父类的设置限制和变换方法

    def set_viewlim_mode(self, mode):
        # 设置视图限制模式方法
        _api.check_in_list([None, "equal", "transform"], mode=mode)  # 检查模式是否在指定列表中
        self._viewlim_mode = mode  # 设置视图限制模式

    def get_viewlim_mode(self):
        # 获取视图限制模式方法
        return self._viewlim_mode  # 返回视图限制模式

    def _sync_lims(self, parent):
        # 同步限制方法
        viewlim = parent.viewLim.frozen()  # 获取父坐标轴的视图限制并冻结
        mode = self.get_viewlim_mode()  # 获取视图限制模式
        if mode is None:
            pass  # 如果模式为None，则不做操作
        elif mode == "equal":
            self.viewLim.set(viewlim)  # 如果模式为"equal"，设置寄生坐标系的视图限制
        elif mode == "transform":
            self.viewLim.set(viewlim.transformed(self.transAux.inverted()))  # 如果模式为"transform"，根据辅助变换设置视图限制
        else:
            _api.check_in_list([None, "equal", "transform"], mode=mode)  # 否则检查模式是否在指定列表中

    # end of aux_transform support

parasite_axes_class_factory = cbook._make_class_factory(
    ParasiteAxesBase, "{}Parasite")  # 创建寄生坐标系类工厂
ParasiteAxes = parasite_axes_class_factory(Axes)  # 创建寄生坐标系类

class HostAxesBase:
    # 宿主坐标系基类

    def __init__(self, *args, **kwargs):
        # 初始化方法，接受任意位置和关键字参数
        self.parasites = []  # 初始化寄生坐标系列表为空
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
    def get_aux_axes(
            self, tr=None, viewlim_mode="equal", axes_class=None, **kwargs):
        """
        Add a parasite axes to this host.

        Despite this method's name, this should actually be thought of as an
        ``add_parasite_axes`` method.

        .. versionchanged:: 3.7
           Defaults to same base axes class as host axes.

        Parameters
        ----------
        tr : `~matplotlib.transforms.Transform` or None, default: None
            If a `.Transform`, the following relation will hold:
            ``parasite.transData = tr + host.transData``.
            If None, the parasite's and the host's ``transData`` are unrelated.
        viewlim_mode : {"equal", "transform", None}, default: "equal"
            How the parasite's view limits are set: directly equal to the
            parent axes ("equal"), equal after application of *tr*
            ("transform"), or independently (None).
        axes_class : subclass type of `~matplotlib.axes.Axes`, optional
            The `~.axes.Axes` subclass that is instantiated.  If None, the base
            class of the host axes is used.
        **kwargs
            Other parameters are forwarded to the parasite axes constructor.
        """
        # 如果未提供 axes_class，则默认为宿主轴的基类
        if axes_class is None:
            axes_class = self._base_axes_class
        # 使用 parasite_axes_class_factory 函数创建对应宿主轴的寄生轴类
        parasite_axes_class = parasite_axes_class_factory(axes_class)
        # 使用 parasite_axes_class 创建一个新的寄生轴对象 ax2
        ax2 = parasite_axes_class(
            self, tr, viewlim_mode=viewlim_mode, **kwargs)
        # 注意到 ax2.transData == tr + ax1.transData
        # 在 ax2 中绘制的任何内容都会与 ax1 的刻度和网格匹配
        # 将 ax2 添加到 parasites 列表中
        self.parasites.append(ax2)
        # 设置 ax2._remove_method 用于从 parasites 列表中移除 ax2
        ax2._remove_method = self.parasites.remove
        # 返回新创建的寄生轴对象 ax2
        return ax2

    def draw(self, renderer):
        # 保存原始子元素数量
        orig_children_len = len(self._children)

        # 获取轴定位器
        locator = self.get_axes_locator()
        # 如果有定位器，则计算轴的位置并应用于 "active" 位置
        if locator:
            pos = locator(self, renderer)
            self.set_position(pos, which="active")
            self.apply_aspect(pos)
        else:
            # 否则仅应用纵横比
            self.apply_aspect()

        # 获取轴的位置矩形
        rect = self.get_position()
        # 对于每个寄生轴 ax，在其上应用相同的纵横比，并扩展子元素列表
        for ax in self.parasites:
            ax.apply_aspect(rect)
            self._children.extend(ax.get_children())

        # 调用父类的 draw 方法绘制轴
        super().draw(renderer)
        # 删除添加的子元素以恢复原始状态
        del self._children[orig_children_len:]

    def clear(self):
        # 调用父类的 clear 方法清除轴本身的内容
        super().clear()
        # 清除所有寄生轴的内容
        for ax in self.parasites:
            ax.clear()

    def pick(self, mouseevent):
        # 调用父类的 pick 方法处理拾取事件
        super().pick(mouseevent)
        # 将拾取事件也传递给所有寄生轴及其子元素
        for a in self.parasites:
            a.pick(mouseevent)
    def twinx(self, axes_class=None):
        """
        Create a twin of Axes with a shared x-axis but independent y-axis.

        The y-axis of self will have ticks on the left and the returned axes
        will have ticks on the right.
        """
        # 创建一个新的与当前 Axes 共享 x 轴但独立 y 轴的双胞胎 Axes
        ax = self._add_twin_axes(axes_class, sharex=self)
        # 将当前 Axes 的右侧坐标轴隐藏
        self.axis["right"].set_visible(False)
        # 显示新创建的 Axes 的右侧坐标轴
        ax.axis["right"].set_visible(True)
        # 隐藏新创建的 Axes 的左、顶、底部坐标轴
        ax.axis["left", "top", "bottom"].set_visible(False)
        return ax

    def twiny(self, axes_class=None):
        """
        Create a twin of Axes with a shared y-axis but independent x-axis.

        The x-axis of self will have ticks on the bottom and the returned axes
        will have ticks on the top.
        """
        # 创建一个新的与当前 Axes 共享 y 轴但独立 x 轴的双胞胎 Axes
        ax = self._add_twin_axes(axes_class, sharey=self)
        # 将当前 Axes 的顶部坐标轴隐藏
        self.axis["top"].set_visible(False)
        # 显示新创建的 Axes 的顶部坐标轴
        ax.axis["top"].set_visible(True)
        # 隐藏新创建的 Axes 的左、右、底部坐标轴
        ax.axis["left", "right", "bottom"].set_visible(False)
        return ax

    def twin(self, aux_trans=None, axes_class=None):
        """
        Create a twin of Axes with no shared axis.

        While self will have ticks on the left and bottom axis, the returned
        axes will have ticks on the top and right axis.
        """
        # 创建一个与当前 Axes 没有共享坐标轴的双胞胎 Axes
        if aux_trans is None:
            aux_trans = mtransforms.IdentityTransform()
        ax = self._add_twin_axes(
            axes_class, aux_transform=aux_trans, viewlim_mode="transform")
        # 将当前 Axes 的顶部和右侧坐标轴隐藏
        self.axis["top", "right"].set_visible(False)
        # 显示新创建的 Axes 的顶部和右侧坐标轴
        ax.axis["top", "right"].set_visible(True)
        # 隐藏新创建的 Axes 的左侧和底部坐标轴
        ax.axis["left", "bottom"].set_visible(False)
        return ax

    def _add_twin_axes(self, axes_class, **kwargs):
        """
        Helper for `.twinx`/`.twiny`/`.twin`.

        *kwargs* are forwarded to the parasite axes constructor.
        """
        # 辅助函数，用于创建 `.twinx`/`.twiny`/`.twin` 方法的双胞胎 Axes
        if axes_class is None:
            axes_class = self._base_axes_class
        ax = parasite_axes_class_factory(axes_class)(self, **kwargs)
        self.parasites.append(ax)
        ax._remove_method = self._remove_any_twin
        return ax

    def _remove_any_twin(self, ax):
        """
        Remove a twin Axes.

        Adjust visibility of axes depending on shared axis settings.
        """
        # 移除一个双胞胎 Axes

        self.parasites.remove(ax)
        restore = ["top", "right"]
        if ax._sharex:
            restore.remove("top")
        if ax._sharey:
            restore.remove("right")
        # 设置恢复显示的 Axes 的顶部和右侧坐标轴
        self.axis[tuple(restore)].set_visible(True)
        self.axis[tuple(restore)].toggle(ticklabels=False, label=False)

    @_api.make_keyword_only("3.8", "call_axes_locator")
    def get_tightbbox(self, renderer=None, call_axes_locator=True,
                      bbox_extra_artists=None):
        """
        Return the tight bounding box of the combined contents.

        Bounding box includes all parasites and the main Axes.
        """
        # 返回组合内容的紧凑边界框

        bbs = [
            *[ax.get_tightbbox(renderer, call_axes_locator=call_axes_locator)
              for ax in self.parasites],
            super().get_tightbbox(renderer,
                                  call_axes_locator=call_axes_locator,
                                  bbox_extra_artists=bbox_extra_artists)]
        return Bbox.union([b for b in bbs if b.width != 0 or b.height != 0])
# 定义一个工厂函数，用于创建宿主坐标轴的类
host_axes_class_factory = host_subplot_class_factory = \
    cbook._make_class_factory(HostAxesBase, "{}HostAxes", "_base_axes_class")

# 创建一个名为 HostAxes 的类，作为子图宿主
HostAxes = SubplotHost = host_axes_class_factory(Axes)


def host_axes(*args, axes_class=Axes, figure=None, **kwargs):
    """
    创建可以作为寄生坐标轴宿主的坐标轴。

    Parameters
    ----------
    figure : `~matplotlib.figure.Figure`
        将要添加坐标轴的图形。默认为当前图形 `.pyplot.gcf()`。

    *args, **kwargs
        将传递给底层 `~.axes.Axes` 对象创建的参数。
    """
    import matplotlib.pyplot as plt
    # 使用指定的 axes_class 创建宿主坐标轴类
    host_axes_class = host_axes_class_factory(axes_class)
    # 如果未指定 figure，则使用当前的图形
    if figure is None:
        figure = plt.gcf()
    # 创建一个新的宿主坐标轴实例
    ax = host_axes_class(figure, *args, **kwargs)
    # 将宿主坐标轴添加到图形中
    figure.add_axes(ax)
    return ax


# 将 host_axes 函数赋值给 host_subplot，以便在后续使用中可以使用相同的功能
host_subplot = host_axes
```