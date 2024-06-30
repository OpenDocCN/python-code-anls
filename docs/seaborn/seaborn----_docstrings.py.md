# `D:\src\scipysrc\seaborn\seaborn\_docstrings.py`

```
import re                          # 导入正则表达式模块
import pydoc                       # 导入pydoc模块
from .external.docscrape import NumpyDocString  # 导入外部库中的NumpyDocString类


class DocstringComponents:
    
    # 编译正则表达式，用于匹配文本中的换行和空白行
    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(self, comp_dict, strip_whitespace=True):
        """Read entries from a dict, optionally stripping outer whitespace."""
        # 如果strip_whitespace为True，去除字典中每个值的外部空白
        if strip_whitespace:
            entries = {}
            for key, val in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()

        self.entries = entries  # 将处理后的字典赋值给实例变量self.entries

    def __getattr__(self, attr):
        """Provide dot access to entries for clean raw docstrings."""
        # 如果属性名attr在self.entries中存在，则返回对应的值，否则尝试抛出AttributeError
        if attr in self.entries:
            return self.entries[attr]
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                # 如果Python以-O选项运行，会删除文档字符串，这里检查__debug__来处理异常
                if __debug__:
                    raise err
                else:
                    pass

    @classmethod
    def from_nested_components(cls, **kwargs):
        """Add multiple sub-sets of components."""
        # 类方法，用给定的关键字参数创建实例，不去除外部空白
        return cls(kwargs, strip_whitespace=False)

    @classmethod
    def from_function_params(cls, func):
        """Use the numpydoc parser to extract components from existing func."""
        # 类方法，使用numpydoc解析器从现有函数func的文档中提取组件信息
        params = NumpyDocString(pydoc.getdoc(func))["Parameters"]
        comp_dict = {}
        for p in params:
            name = p.name
            type = p.type
            desc = "\n    ".join(p.desc)
            comp_dict[name] = f"{name} : {type}\n    {desc}"

        return cls(comp_dict)


# TODO is "vector" the best term here? We mean to imply 1D data with a variety
# of types?

# TODO now that we can parse numpydoc style strings, do we need to define dicts
# of docstring components, or just write out a docstring?


_core_params = dict(
    data="""
data : :class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence
    Input data structure. Either a long-form collection of vectors that can be
    assigned to named variables or a wide-form dataset that will be internally
    reshaped.
    """,  # TODO add link to user guide narrative when exists
    xy="""
x, y : vectors or keys in ``data``
    Variables that specify positions on the x and y axes.
    """,
    hue="""
hue : vector or key in ``data``
    Semantic variable that is mapped to determine the color of plot elements.
    """,
    palette="""
palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
    """  # 对palette参数的描述，用于绘图的颜色映射
)
    Method for choosing the colors to use when mapping the ``hue`` semantic.
    # 选择用于映射“hue”语义时的颜色方法。

    String values are passed to :func:`color_palette`. List or dict values
    # 字符串值传递给 :func:`color_palette` 函数。

    imply categorical mapping, while a colormap object implies numeric mapping.
    # 字符串值意味着分类映射，而颜色映射对象则意味着数值映射。
    """,  # noqa: E501
    hue_order="""
    # hue_order 参数
hue_order : vector of strings
    指定在 ``hue`` 语义下处理和绘制的分类级别顺序。

hue_norm : tuple or :class:`matplotlib.colors.Normalize`
    要么是一对值，设置数据单位的规范化范围，要么是一个对象，将数据单位映射到 [0, 1] 区间。
    使用时表明是数值映射。

color : :mod:`matplotlib color <matplotlib.colors>`
    当不使用 hue 映射时，指定单一的颜色规范。否则，绘图将尝试连接到 matplotlib 的属性循环。

ax : :class:`matplotlib.axes.Axes`
    用于绘图的现有坐标轴。否则，内部调用 :func:`matplotlib.pyplot.gca`。


_core_returns = dict(
    ax="""
:class:`matplotlib.axes.Axes`
    包含绘图的 matplotlib 坐标轴对象。
    """,
    facetgrid="""
:class:`FacetGrid`
    管理一个或多个子图的对象，这些子图对应于条件数据子集，并提供批量设置坐标轴属性的便捷方法。
    """,
    jointgrid="""
:class:`JointGrid`
    管理多个子图的对象，这些子图对应于联合和边缘轴，用于绘制双变量关系或分布。
    """,
    pairgrid="""
:class:`PairGrid`
    管理多个子图的对象，这些子图对应于数据集中多个变量的成对组合的联合和边缘轴。
    """
)

_seealso_blurbs = dict(

    # 关系图
    scatterplot="""
scatterplot : 使用点来绘制数据。
    """,
    lineplot="""
lineplot : 使用线条来绘制数据。
    """,

    # 分布图
    displot="""
displot : 分布绘图函数的图形级接口。
    """,
    histplot="""
histplot : 使用可选的归一化或平滑绘制分箱计数的直方图。
    """,
    kdeplot="""
kdeplot : 使用核密度估计绘制单变量或双变量分布。
    """,
    ecdfplot="""
ecdfplot : 绘制经验累积分布函数。
    """,
    rugplot="""
rugplot : 在 x 和/或 y 轴上的每个观测值处绘制一个刻度。
    """,

    # 分类图
    stripplot="""
stripplot : 使用抖动绘制分类散点图。
    """,
    swarmplot="""
swarmplot : 使用不重叠的点绘制分类散点图。
    """,
    violinplot="""
violinplot : 使用核密度估计绘制增强的箱线图。
    """,
    pointplot="""
pointplot : 使用标记和线条绘制点估计和置信区间。
    """,

    # 多图
    jointplot="""
jointplot : 绘制带有边缘分布的双变量图。
    """,
    pairplot="""
jointplot : 绘制数据集中多个变量的成对双变量图。
    """,
    jointgrid="""
JointGrid : 在双变量数据上设置具有联合和边缘视图的图形。
    """,
    pairgrid="""
# 创建一个 PairGrid 对象，用于在多个变量上设置联合和边缘视图的图表
PairGrid = Set up a figure with joint and marginal views on multiple variables.
    """,
)


# 将核心参数、返回值和相关说明组织成一个字典
_core_docs = dict(
    params=DocstringComponents(_core_params),    # 包含核心参数的文档字符串组件
    returns=DocstringComponents(_core_returns),  # 包含核心返回值的文档字符串组件
    seealso=DocstringComponents(_seealso_blurbs),  # 包含相关参考的文档字符串组件
)
```