# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\qt_editor\figureoptions.py`

```
# 从 itertools 模块导入 chain 函数，用于将多个可迭代对象连接成一个迭代器
from itertools import chain
# 从 matplotlib 库中导入 cbook、cm、mcolors、markers、mimage 模块
from matplotlib import cbook, cm, colors as mcolors, markers, image as mimage
# 从 matplotlib.backends.qt_compat 模块导入 QtGui 类
from matplotlib.backends.qt_compat import QtGui
# 从 matplotlib.backends.qt_editor 模块导入 _formlayout 模块
from matplotlib.backends.qt_editor import _formlayout
# 从 matplotlib.dates 模块导入 DateConverter、num2date 函数
from matplotlib.dates import DateConverter, num2date

# 定义线条样式的字典
LINESTYLES = {'-': 'Solid',
              '--': 'Dashed',
              '-.': 'DashDot',
              ':': 'Dotted',
              'None': 'None',
              }

# 定义绘制风格的字典
DRAWSTYLES = {
    'default': 'Default',
    'steps-pre': 'Steps (Pre)', 'steps': 'Steps (Pre)',
    'steps-mid': 'Steps (Mid)',
    'steps-post': 'Steps (Post)'}

# 获取所有标记的集合
MARKERS = markers.MarkerStyle.markers


def figure_edit(axes, parent=None):
    """Edit matplotlib figure options"""
    # 定义分隔符
    sep = (None, None)  # separator

    # 定义函数，用于根据轴的转换器转换轴限
    def convert_limits(lim, converter):
        """Convert axis limits for correct input editors."""
        # 如果转换器是 DateConverter 类型，则将限制转换为日期
        if isinstance(converter, DateConverter):
            return map(num2date, lim)
        # 否则将限制转换为浮点数
        return map(float, lim)

    # 获取轴映射
    axis_map = axes._axis_map
    # 构建轴限字典，包括轴标题、最小值、最大值、标签、刻度类型等信息
    axis_limits = {
        name: tuple(convert_limits(
            getattr(axes, f'get_{name}lim')(), axis.converter
        ))
        for name, axis in axis_map.items()
    }
    # 构建一般选项列表，包括标题、轴限、标签、刻度类型等信息
    general = [
        ('Title', axes.get_title()),
        sep,
        *chain.from_iterable([
            (
                (None, f"<b>{name.title()}-Axis</b>"),
                ('Min', axis_limits[name][0]),
                ('Max', axis_limits[name][1]),
                ('Label', axis.get_label().get_text()),
                ('Scale', [axis.get_scale(),
                           'linear', 'log', 'symlog', 'logit']),
                sep,
            )
            for name, axis in axis_map.items()
        ]),
        ('(Re-)Generate automatic legend', False),
    ]

    # 保存轴转换器和单位数据的字典
    axis_converter = {
        name: axis.converter
        for name, axis in axis_map.items()
    }
    axis_units = {
        name: axis.get_units()
        for name, axis in axis_map.items()
    }

    # 获取曲线列表，并对每条曲线获取标签，构建带标签的线条列表
    labeled_lines = []
    for line in axes.get_lines():
        label = line.get_label()
        # 跳过没有图例的线条
        if label == '_nolegend_':
            continue
        labeled_lines.append((label, line))
    # 空的曲线列表
    curves = []
    def prepare_data(d, init):
        """
        Prepare entry for FormLayout.

        *d* is a mapping of shorthands to style names (a single style may
        have multiple shorthands, in particular the shorthands `None`,
        `"None"`, `"none"` and `""` are synonyms); *init* is one shorthand
        of the initial style.

        This function returns a list suitable for initializing a
        FormLayout combobox,
    # 对于每个带有标签和可映射对象的元组进行迭代处理
    for label, mappable in labeled_mappables:
        # 获取可映射对象的颜色映射
        cmap = mappable.get_cmap()
        # 如果该颜色映射不在已知颜色映射列表中，则添加到列表中
        if cmap not in cm._colormaps.values():
            cmaps = [(cmap, cmap.name), *cmaps]
        # 获取可映射对象的数据范围（最小值和最大值）
        low, high = mappable.get_clim()
        # 构建包含标签、颜色映射、数据范围的数据列表
        mappabledata = [
            ('Label', label),
            ('Colormap', [cmap.name] + cmaps),
            ('Min. value', low),
            ('Max. value', high),
        ]
        # 如果可映射对象具有"get_interpolation"属性，表明是图像
        if hasattr(mappable, "get_interpolation"):  # Images.
            # 获取所有可用的插值方法
            interpolations = [
                (name, name) for name in sorted(mimage.interpolations_names)]
            # 添加插值方法到数据列表中
            mappabledata.append((
                'Interpolation',
                [mappable.get_interpolation(), *interpolations]))

            # 添加插值阶段（数据或RGBA）到数据列表中
            interpolation_stages = ['data', 'rgba']
            mappabledata.append((
                'Interpolation stage',
                [mappable.get_interpolation_stage(), *interpolation_stages]))

        # 将当前映射对象的数据列表、标签、空字符串添加到映射对象列表中
        mappables.append([mappabledata, label, ""])

    # 检查是否有标量可映射对象被显示
    has_sm = bool(mappables)

    # 创建数据列表，包含通用数据和标签"Axes"
    datalist = [(general, "Axes", "")]
    # 如果存在曲线数据，将其添加到数据列表中，并添加标签"Curves"
    if curves:
        datalist.append((curves, "Curves", ""))
    # 如果存在可映射对象数据，将其添加到数据列表中，并添加标签"Images, etc."
    if mappables:
        datalist.append((mappables, "Images, etc.", ""))

    # 使用_formlayout模块的fedit函数显示数据编辑表单
    _formlayout.fedit(
        datalist, title="Figure options", parent=parent,
        icon=QtGui.QIcon(
            str(cbook._get_data_path('images', 'qt4_editor_options.svg'))),
        apply=apply_callback)
```