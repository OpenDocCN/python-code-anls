# `D:\src\scipysrc\scikit-learn\sklearn\tree\tests\test_export.py`

```
"""
Testing for export functions of decision trees (sklearn.tree.export).
"""

# 导入所需模块
from io import StringIO  # 导入内存字符串操作模块StringIO
from re import finditer, search  # 导入正则表达式模块中的finditer和search函数
from textwrap import dedent  # 导入文本格式化模块中的dedent函数

import numpy as np  # 导入NumPy数学计算库，重命名为np
import pytest  # 导入pytest测试框架

from numpy.random import RandomState  # 从NumPy的随机数模块中导入RandomState类

from sklearn.base import is_classifier  # 导入sklearn基础模块中的is_classifier函数
from sklearn.ensemble import GradientBoostingClassifier  # 导入sklearn集成模块中的GradientBoostingClassifier类
from sklearn.exceptions import NotFittedError  # 导入sklearn异常模块中的NotFittedError异常
from sklearn.tree import (  # 从sklearn决策树模块中导入以下类和函数
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_graphviz,
    export_text,
    plot_tree,
)

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]  # 输入特征矩阵X的示例数据
y = [-1, -1, -1, 1, 1, 1]  # 目标变量y的示例数据
y2 = [[-1, 1], [-1, 1], [-1, 1], [1, 2], [1, 2], [1, 3]]  # 另一种形式的目标变量y2的示例数据
w = [1, 1, 1, 0.5, 0.5, 0.5]  # 样本权重w的示例数据
y_degraded = [1, 1, 1, 1, 1, 1]  # 退化的目标变量y_degraded的示例数据


def test_graphviz_toy():
    # Check correctness of export_graphviz
    # 创建决策树分类器对象clf，设置最大深度为3，最小分割样本数为2，选择基尼系数作为划分标准，随机种子为2
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    clf.fit(X, y)  # 使用示例数据X和y拟合分类器clf

    # Test export code
    contents1 = export_graphviz(clf, out_file=None)  # 将clf的决策树结构导出为Graphviz格式的字符串contents1
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2  # 断言导出的内容contents1与预期的contents2相等

    # Test plot_options
    contents1 = export_graphviz(
        clf,
        filled=True,
        impurity=False,
        proportion=True,
        special_characters=True,
        rounded=True,
        out_file=None,
        fontname="sans",
    )  # 将clf的决策树结构导出为Graphviz格式的字符串contents1，设置多种绘图选项
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled, rounded", color="black", '
        'fontname="sans"] ;\n'
        'edge [fontname="sans"] ;\n'
        "0 [label=<x<SUB>0</SUB> &le; 0.0<br/>samples = 100.0%<br/>"
        'value = [0.5, 0.5]>, fillcolor="#ffffff"] ;\n'
        "1 [label=<samples = 50.0%<br/>value = [1.0, 0.0]>, "
        'fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        "2 [label=<samples = 50.0%<br/>value = [0.0, 1.0]>, "
        'fillcolor="#399de5"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2  # 断言导出的内容contents1与预期的contents2相等

    # Test max_depth
    contents1 = export_graphviz(clf, max_depth=0, class_names=True, out_file=None)
    # 将clf的决策树结构导出为Graphviz格式的字符串contents1，设定最大深度为0，并显示类别名称，不写入文件
    # 定义一个字符串，表示一个决策树的图形结构
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = y[0]"] ;\n'
        '1 [label="(...)"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # 使用 export_graphviz 函数测试 max_depth 与 plot_options
    contents1 = export_graphviz(
        clf, max_depth=0, filled=True, out_file=None, node_ids=True
    )
    # 定义一个字符串，表示设置了特定样式和填充颜色的决策树图形结构
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="node #0\\nx[0] <= 0.0\\ngini = 0.5\\n'
        'samples = 6\\nvalue = [3, 3]", fillcolor="#ffffff"] ;\n'
        '1 [label="(...)"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # 使用 DecisionTreeClassifier 创建并训练一个决策树分类器，测试多输出与加权样本
    clf = DecisionTreeClassifier(
        max_depth=2, min_samples_split=2, criterion="gini", random_state=2
    )
    clf = clf.fit(X, y2, sample_weight=w)

    # 使用 export_graphviz 函数测试多输出与加权样本的决策树图形结构
    contents1 = export_graphviz(clf, filled=True, impurity=False, out_file=None)
    # 定义一个字符串，表示设置了特定样式和填充颜色的决策树图形结构，包含节点样本值与分支条件
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\nsamples = 6\\n'
        "value = [[3.0, 1.5, 0.0]\\n"
        '[3.0, 1.0, 0.5]]", fillcolor="#ffffff"] ;\n'
        '1 [label="samples = 3\\nvalue = [[3, 0, 0]\\n'
        '[3, 0, 0]]", fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="samples = 3\\nvalue = [[0.0, 1.5, 0.0]\\n'
        '[0.0, 1.0, 0.5]]", fillcolor="#f1bd97"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        '3 [label="samples = 2\\nvalue = [[0, 1, 0]\\n'
        '[0, 1, 0]]", fillcolor="#e58139"] ;\n'
        "2 -> 3 ;\n"
        '4 [label="samples = 1\\nvalue = [[0.0, 0.5, 0.0]\\n'
        '[0.0, 0.0, 0.5]]", fillcolor="#e58139"] ;\n'
        "2 -> 4 ;\n"
        "}"
    )

    assert contents1 == contents2

    # 使用 DecisionTreeRegressor 创建并训练一个决策树回归器，测试回归输出与 plot_options
    clf = DecisionTreeRegressor(
        max_depth=3, min_samples_split=2, criterion="squared_error", random_state=2
    )
    clf.fit(X, y)

    # 使用 export_graphviz 函数测试回归输出与 plot_options 的决策树图形结构
    contents1 = export_graphviz(
        clf,
        filled=True,
        leaves_parallel=True,
        out_file=None,
        rotate=True,
        rounded=True,
        fontname="sans",
    )
    contents2 = (
        "digraph Tree {\n"  # 开始定义一个树形结构的图表
        'node [shape=box, style="filled, rounded", color="black", '  # 设置节点的形状、样式、颜色和字体
        'fontname="sans"] ;\n'  # 设置节点文本的字体
        "graph [ranksep=equally, splines=polyline] ;\n"  # 设置图表的布局和边的样式
        'edge [fontname="sans"] ;\n'  # 设置边的文本字体
        "rankdir=LR ;\n"  # 设置节点排列方向为从左到右
        '0 [label="x[0] <= 0.0\\nsquared_error = 1.0\\nsamples = 6\\n'  # 定义节点0的标签和填充颜色
        'value = 0.0", fillcolor="#f2c09c"] ;\n'  # 设置节点0的填充颜色
        '1 [label="squared_error = 0.0\\nsamples = 3\\'
        'nvalue = -1.0", '  # 定义节点1的标签和填充颜色
        'fillcolor="#ffffff"] ;\n'  # 设置节点1的填充颜色
        "0 -> 1 [labeldistance=2.5, labelangle=-45, "  # 定义从节点0到节点1的边及其样式
        'headlabel="True"] ;\n'  # 设置边的标签
        '2 [label="squared_error = 0.0\\nsamples = 3\\nvalue = 1.0", '  # 定义节点2的标签和填充颜色
        'fillcolor="#e58139"] ;\n'  # 设置节点2的填充颜色
        "0 -> 2 [labeldistance=2.5, labelangle=45, "  # 定义从节点0到节点2的边及其样式
        'headlabel="False"] ;\n'  # 设置边的标签
        "{rank=same ; 0} ;\n"  # 将节点0放置在同一水平线上
        "{rank=same ; 1; 2} ;\n"  # 将节点1和节点2放置在同一水平线上
        "}"  # 结束定义树形结构的图表
    )

    assert contents1 == contents2  # 断言contents1与contents2相等

    # 使用最大深度为3的决策树分类器进行训练
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y_degraded)  # 使用降级的学习集进行训练

    contents1 = export_graphviz(clf, filled=True, out_file=None)  # 导出训练后的决策树图表内容
    contents2 = (
        "digraph Tree {\n"  # 开始定义一个树形结构的图表
        'node [shape=box, style="filled", color="black", '  # 设置节点的形状、样式、颜色和字体
        'fontname="helvetica"] ;\n'  # 设置节点文本的字体
        'edge [fontname="helvetica"] ;\n'  # 设置边的文本字体
        '0 [label="gini = 0.0\\nsamples = 6\\nvalue = 6.0", '  # 定义节点0的标签和填充颜色
        'fillcolor="#ffffff"] ;\n'  # 设置节点0的填充颜色
        "}"  # 结束定义树形结构的图表
    )
@pytest.mark.parametrize("constructor", [list, np.array])
# 使用 pytest 的参数化装饰器，构建测试用例 constructor 可以是 list 或 np.array
def test_graphviz_feature_class_names_array_support(constructor):
    # 检查 export_graphviz 是否正确处理特征名和类名，并支持数组
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    # 使用决策树分类器 clf 对象，设置最大深度、最小样本分割、基尼系数作为判据和随机种子，并进行拟合
    clf.fit(X, y)

    # 测试带有 feature_names 的情况
    contents1 = export_graphviz(
        clf, feature_names=constructor(["feature0", "feature1"]), out_file=None
    )
    # 生成图形描述语言内容 contents1，指定特征名为 constructor 创建的数组 ["feature0", "feature1"]，不输出到文件
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature0 <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # 测试带有 class_names 的情况
    contents1 = export_graphviz(
        clf, class_names=constructor(["yes", "no"]), out_file=None
    )
    # 生成图形描述语言内容 contents1，指定类名为 constructor 创建的数组 ["yes", "no"]，不输出到文件
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = yes"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = yes"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = no"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2


def test_graphviz_errors():
    # 检查 export_graphviz 的错误情况
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)

    # 检查未拟合的决策树错误
    out = StringIO()
    with pytest.raises(NotFittedError):
        export_graphviz(clf, out)

    clf.fit(X, y)

    # 检查特征名长度与特征数量不匹配的错误
    message = "Length of feature_names, 1 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a"])

    message = "Length of feature_names, 3 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a", "b", "c"])

    # 检查参数不是估算器实例的错误
    message = "is not an estimator instance"
    with pytest.raises(TypeError, match=message):
        export_graphviz(clf.fit(X, y).tree_)

    # 检查 class_names 错误
    out = StringIO()
    # 使用 pytest 模块中的 pytest.raises 上下文管理器，断言抛出 IndexError 异常
    with pytest.raises(IndexError):
        # 调用 export_graphviz 函数，导出分类器 clf 的图形化描述到输出流 out，
        # 使用空列表作为 class_names 参数
        export_graphviz(clf, out, class_names=[])
# 测试使用 Friedman MSE 准则的决策树回归器
def test_friedman_mse_in_graphviz():
    # 创建一个使用 Friedman MSE 准则的决策树回归器
    clf = DecisionTreeRegressor(criterion="friedman_mse", random_state=0)
    # 使用给定的数据集 X 和 y 进行拟合
    clf.fit(X, y)
    # 创建一个 StringIO 对象，用于存储图形化表示的数据
    dot_data = StringIO()
    # 导出决策树的可视化表示到 dot_data
    export_graphviz(clf, out_file=dot_data)

    # 创建一个使用 Gradient Boosting 的分类器
    clf = GradientBoostingClassifier(n_estimators=2, random_state=0)
    # 使用给定的数据集 X 和 y 进行拟合
    clf.fit(X, y)
    # 遍历每个基础估计器，并导出其可视化表示到 dot_data
    for estimator in clf.estimators_:
        export_graphviz(estimator[0], out_file=dot_data)

    # 检查在生成的 dot_data 中是否存在 Friedman MSE
    for finding in finditer(r"\[.*?samples.*?\]", dot_data.getvalue()):
        assert "friedman_mse" in finding.group()


# 测试精度
def test_precision():
    # 创建用于随机数生成的随机状态对象
    rng_reg = RandomState(2)
    rng_clf = RandomState(8)
    # 遍历包含不同数据集和分类器的元组
    for X, y, clf in zip(
        (rng_reg.random_sample((5, 2)), rng_clf.random_sample((1000, 4))),
        (rng_reg.random_sample((5,)), rng_clf.randint(2, size=(1000,))),
        (
            DecisionTreeRegressor(
                criterion="friedman_mse", random_state=0, max_depth=1
            ),
            DecisionTreeClassifier(max_depth=1, random_state=0),
        ),
    ):
        # 使用给定的数据集 X 和 y 进行拟合
        clf.fit(X, y)
        # 遍历不同的精度值
        for precision in (4, 3):
            # 将分类器的可视化表示导出到 dot_data，指定精度和比例
            dot_data = export_graphviz(
                clf, out_file=None, precision=precision, proportion=True
            )

            # 检查导出的数据中值的精度
            # 检查值
            for finding in finditer(r"value = \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) <= precision + 1
            # 检查不纯度
            if is_classifier(clf):
                pattern = r"gini = \d+\.\d+"
            else:
                pattern = r"friedman_mse = \d+\.\d+"
            # 检查不纯度
            for finding in finditer(pattern, dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1
            # 检查阈值
            for finding in finditer(r"<= \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1


# 测试 export_text 方法中的错误情况
def test_export_text_errors():
    # 创建一个最大深度为2的决策树分类器
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    # 使用给定的数据集 X 和 y 进行拟合
    clf.fit(X, y)
    # 设置错误消息字符串
    err_msg = "feature_names must contain 2 elements, got 1"
    # 使用 pytest 检查导出文本时的 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, feature_names=["a"])
    # 设置错误消息字符串
    err_msg = (
        "When `class_names` is an array, it should contain as"
        " many items as `decision_tree.classes_`. Got 1 while"
        " the tree was fitted with 2 classes."
    )
    # 使用 pytest 检查导出文本时的 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, class_names=["a"])


# 测试 export_text 方法
def test_export_text():
    # 创建一个最大深度为2的决策树分类器
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    # 使用给定的数据集 X 和 y 进行拟合
    clf.fit(X, y)

    # 期望的分类报告字符串
    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    """
    ).strip()
    # 测试确保树的导出文本正确
    
    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- weights: [3.00, 0.00] class: -1
    |--- feature_1 >  0.00
    |   |--- weights: [0.00, 3.00] class: 1
    """
    ).lstrip()
    # 断言导出的决策树文本与预期的报告相符
    assert export_text(clf, show_weights=True) == expected_report
    
    expected_report = dedent(
        """
    |- feature_1 <= 0.00
    | |- class: -1
    |- feature_1 >  0.00
    | |- class: 1
    """
    ).lstrip()
    # 断言导出的决策树文本（带有指定的缩进）与预期的报告相符
    assert export_text(clf, spacing=1) == expected_report
    
    X_l = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-1, 1]]
    y_l = [-1, -1, -1, 1, 1, 1, 2]
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X_l, y_l)
    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    |--- feature_1 >  0.00
    |   |--- truncated branch of depth 2
    """
    ).lstrip()
    # 断言导出的决策树文本（最大深度为0，即不显示截断的分支）与预期的报告相符
    assert export_text(clf, max_depth=0) == expected_report
    
    X_mo = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y_mo = [[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1], [1, 1]]
    
    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_mo, y_mo)
    
    expected_report = dedent(
        """
    |--- feature_1 <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- feature_1 >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    # 断言导出的回归树文本（显示小数点后一位）与预期的报告相符
    assert export_text(reg, decimals=1) == expected_report
    assert export_text(reg, decimals=1, show_weights=True) == expected_report
    
    X_single = [[-2], [-1], [-1], [1], [1], [2]]
    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_single, y_mo)
    
    expected_report = dedent(
        """
    |--- first <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- first >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    # 断言导出的回归树文本（显示小数点后一位，带有特征名称）与预期的报告相符
    assert export_text(reg, decimals=1, feature_names=["first"]) == expected_report
    assert (
        export_text(reg, decimals=1, show_weights=True, feature_names=["first"])
        == expected_report
    )
# 使用 pytest.mark.parametrize 装饰器，为 test_export_text_feature_class_names_array_support 函数添加参数化测试
@pytest.mark.parametrize("constructor", [list, np.array])
def test_export_text_feature_class_names_array_support(constructor):
    # 检查 export_graphviz 是否正确处理特征名和类名，并支持数组
    # 初始化决策树分类器对象
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    # 使用训练数据拟合分类器
    clf.fit(X, y)

    # 准备预期输出报告，去除首部空白字符
    expected_report = dedent(
        """
    |--- b <= 0.00
    |   |--- class: -1
    |--- b >  0.00
    |   |--- class: 1
    """
    ).lstrip()
    # 断言导出文本报告是否符合预期
    assert export_text(clf, feature_names=constructor(["a", "b"])) == expected_report

    # 准备预期输出报告，去除首部空白字符
    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: cat
    |--- feature_1 >  0.00
    |   |--- class: dog
    """
    ).lstrip()
    # 断言导出文本报告是否符合预期
    assert export_text(clf, class_names=constructor(["cat", "dog"])) == expected_report


def test_plot_tree_entropy(pyplot):
    # 主要是冒烟测试
    # 检查 criterion = entropy 时 export_graphviz 的正确性
    # 初始化决策树分类器对象
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="entropy", random_state=2
    )
    # 使用训练数据拟合分类器
    clf.fit(X, y)

    # 测试导出代码
    feature_names = ["first feat", "sepal_width"]
    # 调用 plot_tree 函数生成决策树节点
    nodes = plot_tree(clf, feature_names=feature_names)
    # 断言生成的节点数是否为 5
    assert len(nodes) == 5
    # 断言各节点文本内容是否符合预期
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\nentropy = 1.0\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "True  "
    assert nodes[3].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert nodes[4].get_text() == "  False"


@pytest.mark.parametrize("fontsize", [None, 10, 20])
def test_plot_tree_gini(pyplot, fontsize):
    # 主要是冒烟测试
    # 检查 criterion = gini 时 export_graphviz 的正确性
    # 初始化决策树分类器对象
    clf = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=2,
        criterion="gini",
        random_state=2,
    )
    # 使用训练数据拟合分类器
    clf.fit(X, y)

    # 测试导出代码
    feature_names = ["first feat", "sepal_width"]
    # 调用 plot_tree 函数生成决策树节点，传入字体大小参数
    nodes = plot_tree(clf, feature_names=feature_names, fontsize=fontsize)
    # 断言生成的节点数是否为 5
    assert len(nodes) == 5
    # 如果指定了 fontsize 参数，则断言所有节点的字体大小是否符合预期
    if fontsize is not None:
        assert all(node.get_fontsize() == fontsize for node in nodes)
    # 断言各节点文本内容是否符合预期
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\ngini = 0.5\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "gini = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "True  "
    assert nodes[3].get_text() == "gini = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert nodes[4].get_text() == "  False"


def test_not_fitted_tree(pyplot):
    # 测试未拟合的决策树是否会抛出正确的异常
    # 初始化决策树回归器对象
    clf = DecisionTreeRegressor()
    # 使用 pytest 检查是否抛出 NotFittedError 异常
    with pytest.raises(NotFittedError):
        plot_tree(clf)
```