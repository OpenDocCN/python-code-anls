# `D:\src\scipysrc\matplotlib\lib\matplotlib\projections\__init__.py`

```py
"""
Non-separable transforms that map from data space to screen space.

Projections are defined as `~.axes.Axes` subclasses.  They include the
following elements:

- A transformation from data coordinates into display coordinates.

- An inverse of that transformation.  This is used, for example, to convert
  mouse positions from screen space back into data space.

- Transformations for the gridlines, ticks and ticklabels.  Custom projections
  will often need to place these elements in special locations, and Matplotlib
  has a facility to help with doing so.

- Setting up default values (overriding `~.axes.Axes.cla`), since the defaults
  for a rectilinear Axes may not be appropriate.

- Defining the shape of the Axes, for example, an elliptical Axes, that will be
  used to draw the background of the plot and for clipping any data elements.

- Defining custom locators and formatters for the projection.  For example, in
  a geographic projection, it may be more convenient to display the grid in
  degrees, even if the data is in radians.

- Set up interactive panning and zooming.  This is left as an "advanced"
  feature left to the reader, but there is an example of this for polar plots
  in `matplotlib.projections.polar`.

- Any additional methods for additional convenience or features.

Once the projection Axes is defined, it can be used in one of two ways:

- By defining the class attribute ``name``, the projection Axes can be
  registered with `matplotlib.projections.register_projection` and subsequently
  simply invoked by name::

      fig.add_subplot(projection="my_proj_name")

- For more complex, parameterisable projections, a generic "projection" object
  may be defined which includes the method ``_as_mpl_axes``. ``_as_mpl_axes``
  should take no arguments and return the projection's Axes subclass and a
  dictionary of additional arguments to pass to the subclass' ``__init__``
  method.  Subsequently a parameterised projection can be initialised with::

      fig.add_subplot(projection=MyProjection(param1=param1_value))

  where MyProjection is an object which implements a ``_as_mpl_axes`` method.

A full-fledged and heavily annotated example is in
:doc:`/gallery/misc/custom_projection`.  The polar plot functionality in
`matplotlib.projections.polar` may also be of interest.
"""

from .. import axes, _docstring  # 导入当前模块上级目录的 axes 和 _docstring 模块

# 导入地理投影相关的 Axes 类
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
# 导入极坐标投影的 PolarAxes 类
from .polar import PolarAxes

try:
    # 尝试导入 mpl_toolkits.mplot3d 中的 Axes3D 类
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    # 如果导入失败，发出警告
    import warnings
    warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "
                  "Matplotlib being installed (e.g. as a system package and as a pip "
                  "package). As a result, the 3D projection is not available.")
    Axes3D = None  # 设置 Axes3D 为 None，表示未导入成功


class ProjectionRegistry:
    """A mapping of registered projection names to projection classes."""

    def __init__(self):
        self._all_projection_types = {}  # 初始化一个空字典用于存储所有的投影类型
    # 注册新的投影集合
    def register(self, *projections):
        """Register a new set of projections."""
        # 遍历传入的投影对象列表
        for projection in projections:
            # 获取投影对象的名称
            name = projection.name
            # 将投影对象添加到字典中，以名称作为键
            self._all_projection_types[name] = projection

    # 根据投影类的名称获取对应的投影类
    def get_projection_class(self, name):
        """Get a projection class from its *name*."""
        # 返回给定名称在字典中对应的投影类
        return self._all_projection_types[name]

    # 返回当前所有已注册投影的名称列表，按字母顺序排序
    def get_projection_names(self):
        """Return the names of all projections currently registered."""
        # 返回已注册投影类字典的键列表，并进行排序
        return sorted(self._all_projection_types)
# 创建投影注册表实例
projection_registry = ProjectionRegistry()

# 向投影注册表中注册多个投影类，包括 Axes、PolarAxes 等等
projection_registry.register(
    axes.Axes,
    PolarAxes,
    AitoffAxes,
    HammerAxes,
    LambertAxes,
    MollweideAxes,
)

# 如果 Axes3D 可用，则向注册表中注册 Axes3D 类
if Axes3D is not None:
    projection_registry.register(Axes3D)
else:
    # 如果 Axes3D 不可用，则从命名空间中删除它
    del Axes3D

# 定义一个函数，用于向投影注册表中注册特定的投影类
def register_projection(cls):
    projection_registry.register(cls)

# 定义一个函数，根据投影名称获取相应的投影类
def get_projection_class(projection=None):
    """
    根据投影名称获取投影类。

    如果 *projection* 为 None，则返回标准的直角投影。
    """
    if projection is None:
        projection = 'rectilinear'

    try:
        return projection_registry.get_projection_class(projection)
    except KeyError as err:
        # 如果投影名称未知，则抛出 ValueError 异常
        raise ValueError("Unknown projection %r" % projection) from err

# 获取投影名称列表，并更新文档字符串的插值字典
get_projection_names = projection_registry.get_projection_names
_docstring.interpd.update(projection_names=get_projection_names())
```