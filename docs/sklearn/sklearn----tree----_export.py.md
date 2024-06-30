# `D:\src\scipysrc\scikit-learn\sklearn\tree\_export.py`

```
# 导入必要的模块和类
from collections.abc import Iterable              # 导入Iterable类，用于检查对象是否可迭代
from io import StringIO                          # 导入StringIO类，用于在内存中创建文本缓冲区
from numbers import Integral                     # 导入Integral类，用于整数类型的验证

import numpy as np                                # 导入NumPy库，用于科学计算

from ..base import is_classifier                 # 导入is_classifier函数，用于判断是否为分类器
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params  # 导入参数验证相关函数和类
from ..utils.validation import check_array, check_is_fitted  # 导入数据验证函数
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree  # 导入决策树相关类和函数
from ._reingold_tilford import Tree, buchheim    # 导入树结构相关类和算法


def _color_brew(n):
    """生成n个颜色，色调均匀分布。

    Parameters
    ----------
    n : int
        所需颜色的数量。

    Returns
    -------
    color_list : list, length n
        包含n个元组形式(R, G, B)的颜色列表，表示每种颜色的组成部分。
    """
    color_list = []

    # 初始化饱和度和亮度；计算色度和亮度偏移
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # 计算一些中间值
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # 使用相同色调和色度初始化RGB值
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # 将初始RGB值转换为实际颜色值，并存储
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class Sentinel:
    def __repr__(self):
        return '"tree.dot"'


SENTINEL = Sentinel()


@validate_params(
    {
        "decision_tree": [DecisionTreeClassifier, DecisionTreeRegressor],  # 决策树模型，可以是分类器或回归器
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],  # 最大深度，必须是整数或None
        "feature_names": ["array-like", None],      # 特征名称，可以是数组形式或None
        "class_names": ["array-like", "boolean", None],  # 类别名称，可以是数组形式、布尔值或None
        "label": [StrOptions({"all", "root", "none"})],  # 标签选项，必须是"all"、"root"或"none"
        "filled": ["boolean"],                      # 是否填充颜色，必须是布尔值
        "impurity": ["boolean"],                    # 是否显示不纯度，必须是布尔值
        "node_ids": ["boolean"],                    # 是否显示节点ID，必须是布尔值
        "proportion": ["boolean"],                  # 是否显示节点样本比例，必须是布尔值
        "rounded": ["boolean"],                     # 是否使用圆角框，必须是布尔值
        "precision": [Interval(Integral, 0, None, closed="left"), None],  # 显示数值的精度，必须是非负整数或None
        "ax": "no_validation",                      # matplotlib对象，不进行额外验证
        "fontsize": [Interval(Integral, 0, None, closed="left"), None],  # 字体大小，必须是非负整数或None
    },
    prefer_skip_nested_validation=True,
)
def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
):
    """绘制决策树。

    显示的样本计数会根据可能存在的样本权重进行加权。

    可视化自动适应轴的大小。
    """
    # 函数内部实现部分，未提供具体注释要求，因此不再进行额外的注释。
    pass
    # 调用 ``plt.figure`` 的 ``figsize`` 或 ``dpi`` 参数控制渲染图形的尺寸。
    # 在用户指南的 :ref:`Tree User Guide <tree>` 中可以进一步了解。
    # 0.21 版本添加的功能。

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        要绘制的决策树分类器或回归器。

    max_depth : int, default=None
        表示决策树的最大深度。如果为 None，则生成完整的树。

    feature_names : array-like of str, default=None
        特征的名称数组。
        如果为 None，则使用通用名称 ("x[0]", "x[1]", ...)。

    class_names : array-like of str or True, default=None
        目标类别的名称数组，按升序排列。
        仅适用于分类问题，不支持多输出。如果为 ``True``，显示类别名称的符号表示。

    label : {'all', 'root', 'none'}, default='all'
        是否显示每个节点的信息标签，例如不纯度等。
        选项包括 'all' 表示在每个节点显示，'root' 表示仅在顶部根节点显示，
        'none' 表示不在任何节点显示。

    filled : bool, default=False
        当设置为 ``True`` 时，绘制节点以指示分类的主要类别、回归值的极端值或多输出的节点纯度。

    impurity : bool, default=True
        当设置为 ``True`` 时，在每个节点显示不纯度。

    node_ids : bool, default=False
        当设置为 ``True`` 时，在每个节点显示 ID 号码。

    proportion : bool, default=False
        当设置为 ``True`` 时，将 'values' 和/或 'samples' 的显示改为比例和百分比。

    rounded : bool, default=False
        当设置为 ``True`` 时，绘制具有圆角的节点框，并使用 Helvetica 字体代替 Times-Roman。

    precision : int, default=3
        浮点数精度，用于节点的不纯度、阈值和值属性的显示。

    ax : matplotlib axis, default=None
        要绘制到的 matplotlib 坐标轴。如果为 None，则使用当前坐标轴。任何先前的内容将被清除。

    fontsize : int, default=None
        文本字体大小。如果为 None，则自动确定以适合图形。

    Returns
    -------
    annotations : list of artists
        包含构成树的注解框的艺术家列表。

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.plot_tree(clf)
    [...]

    """

    # 检查决策树是否已经拟合
    check_is_fitted(decision_tree)
    # 创建一个 _MPLTreeExporter 对象，用于导出决策树的可视化
    exporter = _MPLTreeExporter(
        max_depth=max_depth,          # 决策树的最大深度
        feature_names=feature_names,  # 特征名称列表
        class_names=class_names,      # 类别名称列表
        label=label,                  # 标签
        filled=filled,                # 是否填充节点的颜色
        impurity=impurity,            # 不纯度的衡量指标
        node_ids=node_ids,            # 是否显示节点 ID
        proportion=proportion,        # 是否显示节点样本的比例
        rounded=rounded,              # 节点框是否圆角化
        precision=precision,          # 数字显示精度
        fontsize=fontsize,            # 字体大小
    )
    # 调用 exporter 对象的 export 方法，将决策树可视化导出到指定的坐标轴 ax 上
    return exporter.export(decision_tree, ax=ax)
# 定义一个基础树导出器的类，用于生成树形结构的可视化
class _BaseTreeExporter:
    def __init__(
        self,
        max_depth=None,               # 树的最大深度限制
        feature_names=None,           # 特征名称列表
        class_names=None,             # 类别名称列表
        label="all",                  # 标签类型，默认为所有
        filled=False,                 # 是否填充节点的颜色
        impurity=True,                # 是否显示节点的不纯度
        node_ids=False,               # 是否显示节点的ID
        proportion=False,             # 是否显示类别占比
        rounded=False,                # 是否使用四舍五入显示数字
        precision=3,                  # 数字精度控制
        fontsize=None,                # 字体大小设置
    ):
        # 初始化各个参数
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        # 根据节点值获取相应的颜色代码
        if self.colors["bounds"] is None:
            # 分类树的情况
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0.0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            # 回归树或多输出情况
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        # 计算颜色的 alpha 值与白色的混合值
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # 返回 HTML 格式的颜色代码，格式为 #RRGGBB
        return "#%2x%2x%2x" % tuple(color)
    # 获取节点的填充颜色
    def get_fill_color(self, tree, node_id):
        # 检查是否已经存在 "rgb" 颜色设置
        if "rgb" not in self.colors:
            # 如果需要，初始化颜色和边界
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            # 对于多输出情况，查找最大和最小的不纯度
            # 下一行故意使用 -max(impurity) 而不是 min(-impurity)
            # 以及 -min(impurity) 而不是 max(-impurity)，以避免在32位操作系统上非对齐数组上的SIMD问题。
            # 更多细节请参考 https://github.com/scikit-learn/scikit-learn/issues/27506
            if tree.n_outputs != 1:
                self.colors["bounds"] = (-np.max(tree.impurity), -np.min(tree.impurity))
            # 对于回归树中的单一类别并且叶节点值不唯一的情况，查找叶节点中的最大和最小值
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        
        # 如果只有一个输出的情况下，获取节点值并进行相应处理
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :]
            # 对于回归树的单一类别情况，并且节点值是可迭代的，且存在颜色边界设置时，解包 float 值
            if (
                tree.n_classes[0] == 1
                and isinstance(node_val, Iterable)
                and self.colors["bounds"] is not None
            ):
                node_val = node_val.item()
        else:
            # 对于多输出情况，节点值由其不纯度决定
            node_val = -tree.impurity[node_id]
        
        # 返回节点值对应的颜色
        return self.get_color(node_val)
# 定义一个名为 _DOTTreeExporter 的类，它继承自 _BaseTreeExporter 类
class _DOTTreeExporter(_BaseTreeExporter):
    # 初始化方法，设置导出参数和选项
    def __init__(
        self,
        out_file=SENTINEL,  # 输出文件，默认为特殊标记 SENTINEL
        max_depth=None,  # 树的最大深度限制
        feature_names=None,  # 特征名称列表
        class_names=None,  # 类别名称列表
        label="all",  # 标签，默认为 "all"
        filled=False,  # 是否填充节点颜色，默认为 False
        leaves_parallel=False,  # 是否绘制叶子节点平行，默认为 False
        impurity=True,  # 是否显示不纯度，默认为 True
        node_ids=False,  # 是否显示节点 ID，默认为 False
        proportion=False,  # 是否显示节点样本比例，默认为 False
        rotate=False,  # 是否旋转树，默认为 False
        rounded=False,  # 是否为节点绘制圆角，默认为 False
        special_characters=False,  # 是否需要特殊字符兼容性，默认为 False
        precision=3,  # 浮点数精度，默认为 3
        fontname="helvetica",  # 字体名称，默认为 "helvetica"
    ):
        # 调用父类的初始化方法，设置树的导出参数
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        # 设置是否绘制叶子节点平行的选项
        self.leaves_parallel = leaves_parallel
        # 设置输出文件选项
        self.out_file = out_file
        # 设置是否需要特殊字符兼容性的选项
        self.special_characters = special_characters
        # 设置字体名称选项
        self.fontname = fontname
        # 设置是否旋转树的选项
        self.rotate = rotate

        # 根据是否需要特殊字符兼容性设置特殊字符列表
        if special_characters:
            self.characters = ["&#35;", "<SUB>", "</SUB>", "&le;", "<br/>", ">", "<"]
        else:
            self.characters = ["#", "[", "]", "<=", "\\n", '"', '"']

        # 用于绘制每个节点深度的字典
        self.ranks = {"leaves": []}
        # 用于渲染每个节点颜色的字典
        self.colors = {"bounds": None}

    # 导出方法，接收决策树对象并导出为特定格式
    def export(self, decision_tree):
        # 如果设置了特征名称列表，检查其长度是否与决策树中的特征数匹配，若不匹配则引发 ValueError
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_in_:
                raise ValueError(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), decision_tree.n_features_in_)
                )
        # 写入导出格式的头部信息到输出文件中
        self.head()
        # 递归处理决策树的每个节点，并添加节点和边的属性
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, criterion="impurity")
        else:
            self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        # 写入导出格式的尾部信息到输出文件中
        self.tail()

    # 写入导出格式的尾部信息到输出文件中
    def tail(self):
        # 如果需要，绘制所有叶子节点在同一深度
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write(
                    "{rank=same ; " + "; ".join(r for r in self.ranks[rank]) + "} ;\n"
                )
        # 写入格式尾部到输出文件中
        self.out_file.write("}")
    def head(self):
        # 写入开始生成图形的声明
        self.out_file.write("digraph Tree {\n")

        # 指定节点的美学特征
        self.out_file.write("node [shape=box")
        
        rounded_filled = []
        # 如果需要填充节点，则添加"filled"属性
        if self.filled:
            rounded_filled.append("filled")
        # 如果需要圆角节点，则添加"rounded"属性
        if self.rounded:
            rounded_filled.append("rounded")
        # 如果有任何美学特征要添加，将它们作为样式属性写入
        if len(rounded_filled) > 0:
            self.out_file.write(', style="%s", color="black"' % ", ".join(rounded_filled))
        
        # 设置节点的字体名称
        self.out_file.write(', fontname="%s"' % self.fontname)
        self.out_file.write("] ;\n")

        # 指定图形和边的美学特征
        if self.leaves_parallel:
            # 如果节点并列，则设置等距离和折线样式
            self.out_file.write("graph [ranksep=equally, splines=polyline] ;\n")

        # 设置边的字体名称
        self.out_file.write('edge [fontname="%s"] ;\n' % self.fontname)

        # 如果需要旋转图形方向，则设置从左到右
        if self.rotate:
            self.out_file.write("rankdir=LR ;\n")
    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        # 如果节点 ID 是叶子节点标识，抛出值错误异常
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        # 获取左右子节点的 ID
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # 添加带有描述的节点
        if self.max_depth is None or depth <= self.max_depth:
            # 如果左子节点是叶子节点，则将节点 ID 添加到 'leaves' 排序列表中
            if left_child == _tree.TREE_LEAF:
                self.ranks["leaves"].append(str(node_id))
            # 否则，将节点 ID 添加到对应深度的排名列表中
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))

            # 将节点及其描述写入输出文件
            self.out_file.write(
                "%d [label=%s" % (node_id, self.node_to_str(tree, node_id, criterion))
            )

            # 如果需要填充节点，添加填充颜色属性
            if self.filled:
                self.out_file.write(
                    ', fillcolor="%s"' % self.get_fill_color(tree, node_id)
                )
            self.out_file.write("] ;\n")

            # 如果存在父节点，则添加到父节点的边
            if parent is not None:
                self.out_file.write("%d -> %d" % (parent, node_id))
                # 如果父节点是根节点，绘制 True/False 标签
                if parent == 0:
                    angles = np.array([45, -45]) * ((self.rotate - 0.5) * -2)
                    self.out_file.write(" [labeldistance=2.5, labelangle=")
                    if node_id == 1:
                        self.out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' % angles[1])
                self.out_file.write(" ;\n")

            # 如果左子节点不是叶子节点，则递归调用自身处理左右子树
            if left_child != _tree.TREE_LEAF:
                self.recurse(
                    tree,
                    left_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1,
                )
                self.recurse(
                    tree,
                    right_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1,
                )

        else:
            # 将叶子节点 ID 添加到 'leaves' 排序列表中
            self.ranks["leaves"].append(str(node_id))

            # 写入带省略号描述的叶子节点到输出文件，并根据填充选项添加灰色填充颜色
            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write("] ;\n" % node_id)

            # 如果存在父节点，则添加到父节点的边
            if parent is not None:
                self.out_file.write("%d -> %d ;\n" % (parent, node_id))
# 定义一个继承自_BaseTreeExporter的_MPLTreeExporter类，用于Matplotlib可视化树结构
class _MPLTreeExporter(_BaseTreeExporter):
    # 初始化函数，设置各种参数和属性
    def __init__(
        self,
        max_depth=None,  # 树的最大深度限制
        feature_names=None,  # 特征名称列表
        class_names=None,  # 类别名称列表
        label="all",  # 标签显示选项，默认为'all'
        filled=False,  # 是否填充节点的颜色，默认为False
        impurity=True,  # 是否显示节点的杂质（impurity），默认为True
        node_ids=False,  # 是否显示节点的ID，默认为False
        proportion=False,  # 是否显示每个类别的比例，默认为False
        rounded=False,  # 节点框是否为圆角，默认为False
        precision=3,  # 数值显示的小数精度，默认为3
        fontsize=None,  # 字体大小，用于节点文本显示
    ):
        # 调用父类_BaseTreeExporter的初始化函数，传入各个参数
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        # 设置字体大小属性
        self.fontsize = fontsize

        # 用于存储各节点深度信息的字典，初始化只包含一个'leaves'键
        self.ranks = {"leaves": []}
        # 存储节点颜色信息的字典，初始值为None
        self.colors = {"bounds": None}

        # 定义节点名称中特殊字符的列表
        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        # 节点框样式参数的字典
        self.bbox_args = dict()
        # 如果节点框需要圆角，则设置boxstyle为'round'
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        # 箭头样式参数的字典，设置箭头朝向为'<-'
        self.arrow_args = dict(arrowstyle="<-")

    # 递归构建树结构的函数
    def _make_tree(self, node_id, et, criterion, depth=0):
        # 获取当前节点的名称字符串
        name = self.node_to_str(et, node_id, criterion=criterion)
        # 如果当前节点不是叶子节点且未达到最大深度限制，则继续递归构建
        if et.children_left[node_id] != _tree.TREE_LEAF and (
            self.max_depth is None or depth <= self.max_depth
        ):
            # 递归构建左右子树
            children = [
                self._make_tree(
                    et.children_left[node_id], et, criterion, depth=depth + 1
                ),
                self._make_tree(
                    et.children_right[node_id], et, criterion, depth=depth + 1
                ),
            ]
        else:
            # 如果当前节点是叶子节点或达到最大深度限制，则创建叶子节点
            return Tree(name, node_id)
        # 返回当前节点及其子节点构成的Tree对象
        return Tree(name, node_id, *children)
    # 定义一个方法用于导出决策树的可视化表示
    def export(self, decision_tree, ax=None):
        # 导入必要的绘图库和注释模块
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        # 如果未提供轴对象，则获取当前的轴
        if ax is None:
            ax = plt.gca()

        # 清空轴并设置为无坐标轴显示
        ax.clear()
        ax.set_axis_off()

        # 使用决策树生成内部表示，并通过布赫海姆算法绘制树形结构
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        draw_tree = buchheim(my_tree)

        # 计算树的最大边界，并计算绘图区域的宽度和高度
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        # 计算绘图比例因子
        scale_x = ax_width / max_x
        scale_y = ax_height / max_y

        # 递归绘制决策树的节点和连接线
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y)

        # 获取所有的注释对象，这些对象是图中的文本标签
        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

        # 更新所有文本框的位置和大小
        renderer = ax.figure.canvas.get_renderer()
        for ann in anns:
            ann.update_bbox_position_size(renderer)

        # 如果未指定字体大小，则根据绘图区域和文本框大小调整字体大小，避免重叠
        if self.fontsize is None:
            # 获取所有文本框的尺寸
            extents = [
                bbox_patch.get_window_extent()
                for ann in anns
                if (bbox_patch := ann.get_bbox_patch()) is not None
            ]
            # 计算最大的文本框宽度和高度
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # 计算适合的字体大小，以保证文本在图中显示合理
            size = anns[0].get_fontsize() * min(
                scale_x / max_width, scale_y / max_height
            )
            # 设置所有注释的字体大小
            for ann in anns:
                ann.set_fontsize(size)

        # 返回所有的注释对象，这些对象包含了图中所有的文本标签
        return anns
    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        # 导入 matplotlib.pyplot 库，用于绘图
        import matplotlib.pyplot as plt

        # 不带边界框的注释的参数
        common_kwargs = dict(
            zorder=100 - 10 * depth,  # 控制绘图顺序的参数
            xycoords="axes fraction",  # 坐标系设置为相对于图像的比例坐标系
        )
        if self.fontsize is not None:
            common_kwargs["fontsize"] = self.fontsize  # 如果设置了字体大小，则传递给注释参数

        # 带边界框的注释的参数
        kwargs = dict(
            ha="center",  # 文本水平居中对齐
            va="center",  # 文本垂直居中对齐
            bbox=self.bbox_args.copy(),  # 复制边界框参数
            arrowprops=self.arrow_args.copy(),  # 复制箭头参数
            **common_kwargs,  # 包含通用的注释参数
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]  # 设置箭头边缘颜色为文本颜色

        # 将坐标偏移0.5以在图中居中显示
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)  # 获取填充颜色
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()  # 获取坐标轴背景色作为填充颜色

            if node.parent is None:
                # 根节点的注释
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)  # 标注父节点到当前节点的箭头

                # 如果父节点是根节点，则绘制True/False标签
                if node.parent.parent is None:
                    # 调整文本位置以稍微在箭头上方显示
                    text_pos = (
                        (xy_parent[0] + xy[0]) / 2,
                        (xy_parent[1] + xy[1]) / 2,
                    )
                    # 使用标签文本注释箭头，指示子节点
                    if node.parent.left() == node:
                        label_text, label_ha = ("True  ", "right")
                    else:
                        label_text, label_ha = ("  False", "left")
                    ax.annotate(label_text, text_pos, ha=label_ha, **common_kwargs)
            for child in node.children:
                self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)  # 递归调用自身，绘制子节点

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"  # 如果深度超过设定值，使用灰色填充边界框
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)  # 在父节点和当前节点之间绘制省略符号的注释
@validate_params(
    # 验证参数装饰器，确保函数参数满足以下规范
    {
        "decision_tree": "no_validation",  # 决策树对象，无需特定验证
        "out_file": [str, None, HasMethods("write")],  # 输出文件名或对象，可以是字符串或支持写入方法的对象，或者为None
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],  # 最大深度，整数类型，闭区间[0, None)
        "feature_names": ["array-like", None],  # 特征名称，类似数组，或为None
        "class_names": ["array-like", "boolean", None],  # 类别名称，类似数组，布尔值，或为None
        "label": [StrOptions({"all", "root", "none"})],  # 标签显示选项，可选值为{'all', 'root', 'none'}
        "filled": ["boolean"],  # 是否填充颜色，布尔值
        "leaves_parallel": ["boolean"],  # 叶子节点并行显示，布尔值
        "impurity": ["boolean"],  # 是否显示杂质，布尔值
        "node_ids": ["boolean"],  # 是否显示节点ID，布尔值
        "proportion": ["boolean"],  # 是否显示样本比例，布尔值
        "rotate": ["boolean"],  # 是否旋转树，布尔值
        "rounded": ["boolean"],  # 节点是否圆角显示，布尔值
        "special_characters": ["boolean"],  # 是否允许特殊字符，布尔值
        "precision": [Interval(Integral, 0, None, closed="left"), None],  # 输出精度，整数类型，闭区间[0, None)
        "fontname": [str],  # 字体名称，字符串类型
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def export_graphviz(
    decision_tree,
    out_file=None,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    leaves_parallel=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rotate=False,
    rounded=False,
    special_characters=False,
    precision=3,
    fontname="helvetica",
):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported to GraphViz.

    out_file : object or str, default=None
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

        .. versionchanged:: 0.20
            Default of out_file changed from "tree.dot" to None.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of shape (n_features,), default=None
        An array containing the feature names.
        If None, generic names will be used ("x[0]", "x[1]", ...).

    class_names : array-like of shape (n_classes,) or bool, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.
    # filled : bool, default=False
    # 如果设置为True，为了分类问题，绘制节点以指示主要类别；对于回归问题，显示值的极端程度；对于多输出问题，显示节点的纯度。

    # leaves_parallel : bool, default=False
    # 如果设置为True，将所有叶节点绘制在树的底部。

    # impurity : bool, default=True
    # 如果设置为True，显示每个节点的不纯度。

    # node_ids : bool, default=False
    # 如果设置为True，显示每个节点的ID号码。

    # proportion : bool, default=False
    # 如果设置为True，将 'values' 和/或 'samples' 的显示更改为比例和百分比，分别。

    # rotate : bool, default=False
    # 如果设置为True，将树的方向从上到下改为从左到右。

    # rounded : bool, default=False
    # 如果设置为True，使用圆角绘制节点框。

    # special_characters : bool, default=False
    # 如果设置为False，忽略PostScript兼容性的特殊字符。

    # precision : int, default=3
    # 浮点数精度，用于每个节点的不纯度、阈值和值属性的小数位数。

    # fontname : str, default='helvetica'
    # 用于渲染文本的字体名称。

    # Returns
    # -------
    # dot_data : str
    # 输入树的图形表示形式，以GraphViz dot格式的字符串表示。
    # 仅在``out_file``为None时返回。

    # Examples
    # --------
    # >>> from sklearn.datasets import load_iris
    # >>> from sklearn import tree

    # >>> clf = tree.DecisionTreeClassifier()
    # >>> iris = load_iris()

    # >>> clf = clf.fit(iris.data, iris.target)
    # >>> tree.export_graphviz(clf)
    # 'digraph Tree {...

    if feature_names is not None:
        # 如果feature_names不为None，则确保其为2维数组，不强制最小样本数为0
        feature_names = check_array(
            feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )
    if class_names is not None and not isinstance(class_names, bool):
        # 如果class_names不为None且不是布尔值，则确保其为2维数组，不强制最小样本数为0
        class_names = check_array(
            class_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )

    # 检查决策树是否已经拟合
    check_is_fitted(decision_tree)
    own_file = False
    return_string = False
    # 尝试执行以下代码块，处理文件输出和字符串返回逻辑
    try:
        # 如果 out_file 是字符串类型，则打开文件以写入 utf-8 编码
        if isinstance(out_file, str):
            out_file = open(out_file, "w", encoding="utf-8")
            own_file = True

        # 如果 out_file 为 None，则将返回值设置为字符串模式，并创建 StringIO 对象
        if out_file is None:
            return_string = True
            out_file = StringIO()

        # 创建 _DOTTreeExporter 实例，用于导出决策树到指定的输出文件或字符串
        exporter = _DOTTreeExporter(
            out_file=out_file,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            leaves_parallel=leaves_parallel,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rotate=rotate,
            rounded=rounded,
            special_characters=special_characters,
            precision=precision,
            fontname=fontname,
        )

        # 调用 exporter 的 export 方法导出决策树
        exporter.export(decision_tree)

        # 如果设置了 return_string 标志，则返回导出的字符串内容
        if return_string:
            return exporter.out_file.getvalue()

    finally:
        # 最终执行的清理工作：如果 own_file 为 True，关闭文件对象 out_file
        if own_file:
            out_file.close()
# 定义一个函数来计算决策树中从指定节点开始的子树的深度
def _compute_depth(tree, node):
    """
    Returns the depth of the subtree rooted in node.
    """

    # 定义一个递归函数来计算节点当前深度以及其子节点的深度
    def compute_depth_(
        current_node, current_depth, children_left, children_right, depths
    ):
        # 将当前节点的深度加入深度列表
        depths += [current_depth]
        left = children_left[current_node]
        right = children_right[current_node]
        # 如果左右子节点均存在，则递归计算左右子节点的深度
        if left != -1 and right != -1:
            compute_depth_(
                left, current_depth + 1, children_left, children_right, depths
            )
            compute_depth_(
                right, current_depth + 1, children_left, children_right, depths
            )

    # 初始化深度列表
    depths = []
    # 调用递归函数来计算从给定节点开始的子树的深度
    compute_depth_(node, 1, tree.children_left, tree.children_right, depths)
    # 返回计算出的最大深度
    return max(depths)


# 使用装饰器 validate_params 对 export_text 函数进行参数验证和类型检查
@validate_params(
    {
        "decision_tree": [DecisionTreeClassifier, DecisionTreeRegressor],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", None],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "spacing": [Interval(Integral, 1, None, closed="left"), None],
        "decimals": [Interval(Integral, 0, None, closed="left"), None],
        "show_weights": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
# 定义导出决策树文本报告的函数 export_text
def export_text(
    decision_tree,
    *,
    feature_names=None,
    class_names=None,
    max_depth=10,
    spacing=3,
    decimals=2,
    show_weights=False,
):
    """Build a text report showing the rules of a decision tree.

    Note that backwards compatibility may not be supported.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported.
        It can be an instance of
        DecisionTreeClassifier or DecisionTreeRegressor.

    feature_names : array-like of shape (n_features,), default=None
        An array containing the feature names.
        If None generic names will be used ("feature_0", "feature_1", ...).

    class_names : array-like of shape (n_classes,), default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.

        - if `None`, the class names are delegated to `decision_tree.classes_`;
        - otherwise, `class_names` will be used as class names instead of
          `decision_tree.classes_`. The length of `class_names` must match
          the length of `decision_tree.classes_`.

        .. versionadded:: 1.3

    max_depth : int, default=10
        Only the first max_depth levels of the tree are exported.
        Truncated branches will be marked with "...".

    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.

    decimals : int, default=2
        Number of decimal digits to display.

    show_weights : bool, default=False
        If true the classification weights will be exported on each leaf.
        The classification weights are the number of samples each class.

    Returns
    -------
    # 定义一个字符串变量，用于存储决策树所有规则的文本总结
    report : str
        Text summary of all the rules in the decision tree.
        
    Examples
    --------
    
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.tree import export_text
    >>> iris = load_iris()
    >>> X = iris['data']
    >>> y = iris['target']
    >>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    >>> decision_tree = decision_tree.fit(X, y)
    >>> r = export_text(decision_tree, feature_names=iris['feature_names'])
    >>> print(r)
    |--- petal width (cm) <= 0.80
    |   |--- class: 0
    |--- petal width (cm) >  0.80
    |   |--- petal width (cm) <= 1.75
    |   |   |--- class: 1
    |   |--- petal width (cm) >  1.75
    |   |   |--- class: 2
    """
    # 如果 feature_names 参数不为空，则进行类型和数组长度检查
    if feature_names is not None:
        feature_names = check_array(
            feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )
    # 如果 class_names 参数不为空，则进行类型和数组长度检查
    if class_names is not None:
        class_names = check_array(
            class_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )

    # 检查决策树是否已经拟合
    check_is_fitted(decision_tree)
    # 获取决策树的内部树结构
    tree_ = decision_tree.tree_
    # 如果是分类器，则处理类名相关的逻辑
    if is_classifier(decision_tree):
        if class_names is None:
            class_names = decision_tree.classes_
        elif len(class_names) != len(decision_tree.classes_):
            # 如果 class_names 的长度与决策树类别数量不匹配，则抛出 ValueError
            raise ValueError(
                "When `class_names` is an array, it should contain as"
                " many items as `decision_tree.classes_`. Got"
                f" {len(class_names)} while the tree was fitted with"
                f" {len(decision_tree.classes_)} classes."
            )
    # 右子节点格式化字符串
    right_child_fmt = "{} {} <= {}\n"
    # 左子节点格式化字符串
    left_child_fmt = "{} {} >  {}\n"
    # 截断节点格式化字符串
    truncation_fmt = "{} {}\n"

    # 如果 feature_names 不为空且其长度与决策树的特征数量不匹配，则抛出 ValueError
    if feature_names is not None and len(feature_names) != tree_.n_features:
        raise ValueError(
            "feature_names must contain %d elements, got %d"
            % (tree_.n_features, len(feature_names))
        )

    # 如果决策树是分类器类型
    if isinstance(decision_tree, DecisionTreeClassifier):
        # 设置值的格式化字符串，包括权重信息
        value_fmt = "{}{} weights: {}\n"
        # 如果不显示权重，则调整值的格式化字符串
        if not show_weights:
            value_fmt = "{}{}{}\n"
    else:
        # 如果不是分类器类型，则设置值的格式化字符串，包括数值信息
        value_fmt = "{}{} value: {}\n"

    # 如果 feature_names 不为空，则根据 tree_ 结构中的特征索引构建特征名称列表
    if feature_names is not None:
        feature_names_ = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else None
            for i in tree_.feature
        ]
    else:
        # 否则，创建默认的特征名称列表
        feature_names_ = ["feature_{}".format(i) for i in tree_.feature]

    # 清空报告文本
    export_text.report = ""
    def _add_leaf(value, weighted_n_node_samples, class_name, indent):
        val = ""
        if isinstance(decision_tree, DecisionTreeClassifier):
            # 如果决策树是分类器且需要展示权重
            if show_weights:
                # 根据格式化要求生成加权值的列表
                val = [
                    "{1:.{0}f}, ".format(decimals, v * weighted_n_node_samples)
                    for v in value
                ]
                # 将列表转换为字符串格式
                val = "[" + "".join(val)[:-2] + "]"
                weighted_n_node_samples  # 权重样本数
            # 添加类别信息到值的字符串表示
            val += " class: " + str(class_name)
        else:
            # 如果不是分类器，直接生成值的列表
            val = ["{1:.{0}f}, ".format(decimals, v) for v in value]
            # 将列表转换为字符串格式
            val = "[" + "".join(val)[:-2] + "]"
        # 将格式化后的值添加到导出文本报告中
        export_text.report += value_fmt.format(indent, "", val)

    def print_tree_recurse(node, depth):
        # 根据深度生成相应的缩进字符串
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing

        # 获取当前节点的值和类别
        value = None
        if tree_.n_outputs == 1:
            value = tree_.value[node][0]
        else:
            value = tree_.value[node].T[0]
        class_name = np.argmax(value)

        # 如果类别数大于1且输出数为1，将类别索引映射为类别名称
        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
            class_name = class_names[class_name]

        # 获取当前节点的加权样本数
        weighted_n_node_samples = tree_.weighted_n_node_samples[node]

        # 如果当前深度小于等于最大深度+1
        if depth <= max_depth + 1:
            info_fmt = ""
            info_fmt_left = info_fmt
            info_fmt_right = info_fmt

            # 如果当前节点有特征
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # 获取特征名称和阈值
                name = feature_names_[node]
                threshold = tree_.threshold[node]
                threshold = "{1:.{0}f}".format(decimals, threshold)
                # 将右子节点信息添加到导出文本报告中
                export_text.report += right_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_left
                # 递归打印左子树
                print_tree_recurse(tree_.children_left[node], depth + 1)

                # 将左子节点信息添加到导出文本报告中
                export_text.report += left_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_right
                # 递归打印右子树
                print_tree_recurse(tree_.children_right[node], depth + 1)
            else:  # 如果是叶子节点
                # 调用函数添加叶子节点的信息到导出文本报告中
                _add_leaf(value, weighted_n_node_samples, class_name, indent)
        else:
            # 计算当前子树的深度
            subtree_depth = _compute_depth(tree_, node)
            # 如果子树深度为1，调用函数添加叶子节点信息到导出文本报告中
            if subtree_depth == 1:
                _add_leaf(value, weighted_n_node_samples, class_name, indent)
            else:
                # 添加截断信息到导出文本报告中
                trunc_report = "truncated branch of depth %d" % subtree_depth
                export_text.report += truncation_fmt.format(indent, trunc_report)

    # 从根节点开始递归打印决策树
    print_tree_recurse(0, 1)
    # 返回完整的导出文本报告
    return export_text.report
```