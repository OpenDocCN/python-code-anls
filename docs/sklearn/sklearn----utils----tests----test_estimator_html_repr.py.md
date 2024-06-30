# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_estimator_html_repr.py`

```
# 导入必要的模块和库

import html  # 导入html模块，用于HTML转义处理
import locale  # 导入locale模块，用于本地化设置
import re  # 导入re模块，用于正则表达式操作
from contextlib import closing  # 从contextlib模块中导入closing，用于上下文管理
from io import StringIO  # 从io模块中导入StringIO，用于内存中的文件操作
from unittest.mock import patch  # 从unittest.mock模块中导入patch，用于模拟对象

import pytest  # 导入pytest，用于单元测试

# 导入sklearn相关模块和类
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._estimator_html_repr import (
    _get_css_style,
    _get_visual_block,
    _HTMLDocumentationLinkMixin,
    _write_label_html,
    estimator_html_repr,
)
from sklearn.utils.fixes import parse_version

# 定义单元测试函数，参数为checked，测试_write_label_html函数
@pytest.mark.parametrize("checked", [True, False])
def test_write_label_html(checked):
    # 测试检查逻辑和标签生成
    name = "LogisticRegression"
    tool_tip = "hello-world"

    # 使用StringIO对象out进行HTML标签生成测试
    with closing(StringIO()) as out:
        _write_label_html(out, name, tool_tip, checked=checked)
        html_label = out.getvalue()

        # 使用正则表达式验证生成的HTML标签是否符合预期
        p = (
            r'<label for="sk-estimator-id-[0-9]*"'
            r' class="sk-toggleable__label (fitted)? sk-toggleable__label-arrow ">'
            r"LogisticRegression"
        )
        re_compiled = re.compile(p)
        assert re_compiled.search(html_label)

        # 验证生成的HTML标签是否以特定开头
        assert html_label.startswith('<div class="sk-label-container">')
        # 验证生成的HTML标签中是否包含指定的预处理文本
        assert "<pre>hello-world</pre>" in html_label
        if checked:
            assert "checked>" in html_label

# 定义单元测试函数，参数为est，测试_get_visual_block函数
@pytest.mark.parametrize("est", ["passthrough", "drop", None])
def test_get_visual_block_single_str_none(est):
    # 测试用字符串表示的估算器情况
    est_html_info = _get_visual_block(est)
    assert est_html_info.kind == "single"
    assert est_html_info.estimators == est
    assert est_html_info.names == str(est)
    assert est_html_info.name_details == str(est)

# 定义单元测试函数，测试_get_visual_block函数
def test_get_visual_block_single_estimator():
    est = LogisticRegression(C=10.0)
    est_html_info = _get_visual_block(est)
    assert est_html_info.kind == "single"
    assert est_html_info.estimators == est
    assert est_html_info.names == est.__class__.__name__
    assert est_html_info.name_details == str(est)

# 定义单元测试函数，测试_get_visual_block函数
def test_get_visual_block_pipeline():
    # 创建一个管道对象，用于数据处理和模型训练
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),  # 第一步: 缺失值填充器
            ("do_nothing", "passthrough"),  # 第二步: 无操作，直接传递数据
            ("do_nothing_more", None),  # 第三步: 无操作，直接传递数据
            ("classifier", LogisticRegression()),  # 第四步: 逻辑回归分类器
        ]
    )
    # 获取管道对象的可视化信息，通常用于显示或验证管道结构
    est_html_info = _get_visual_block(pipe)
    # 断言管道信息的类型为序列化（即连续执行的步骤）
    assert est_html_info.kind == "serial"
    # 断言管道中每个步骤的估计器名称与顺序相符
    assert est_html_info.estimators == tuple(step[1] for step in pipe.steps)
    # 断言管道中每个步骤的名称与预期列表中的名称相符
    assert est_html_info.names == [
        "imputer: SimpleImputer",
        "do_nothing: passthrough",
        "do_nothing_more: passthrough",
        "classifier: LogisticRegression",
    ]
    # 断言管道中每个步骤的名称详细信息与字符串化的估计器名称列表相符
    assert est_html_info.name_details == [str(est) for _, est in pipe.steps]
# 定义测试函数，用于测试获取视觉块特征联合的功能
def test_get_visual_block_feature_union():
    # 创建特征联合对象，包括两个转换器："pca" 和 "svd"
    f_union = FeatureUnion([("pca", PCA()), ("svd", TruncatedSVD())])
    # 获取视觉块信息
    est_html_info = _get_visual_block(f_union)
    # 断言视觉块信息的类型为 "parallel"
    assert est_html_info.kind == "parallel"
    # 断言视觉块信息的名称为 ("pca", "svd")
    assert est_html_info.names == ("pca", "svd")
    # 断言视觉块信息的估计器与转换器列表中的估计器相匹配
    assert est_html_info.estimators == tuple(
        trans[1] for trans in f_union.transformer_list
    )
    # 断言视觉块信息的名称细节为 (None, None)
    assert est_html_info.name_details == (None, None)


# 定义测试函数，用于测试获取视觉块投票分类器的功能
def test_get_visual_block_voting():
    # 创建投票分类器对象，包括两个估计器："log_reg" 和 "mlp"
    clf = VotingClassifier(
        [("log_reg", LogisticRegression()), ("mlp", MLPClassifier())]
    )
    # 获取视觉块信息
    est_html_info = _get_visual_block(clf)
    # 断言视觉块信息的类型为 "parallel"
    assert est_html_info.kind == "parallel"
    # 断言视觉块信息的估计器与分类器中的估计器相匹配
    assert est_html_info.estimators == tuple(trans[1] for trans in clf.estimators)
    # 断言视觉块信息的名称为 ("log_reg", "mlp")
    assert est_html_info.names == ("log_reg", "mlp")
    # 断言视觉块信息的名称细节为 (None, None)
    assert est_html_info.name_details == (None, None)


# 定义测试函数，用于测试获取视觉块列变换器的功能
def test_get_visual_block_column_transformer():
    # 创建列变换器对象，包括两个转换器："pca" 和 "svd"
    ct = ColumnTransformer(
        [("pca", PCA(), ["num1", "num2"]), ("svd", TruncatedSVD, [0, 3])]
    )
    # 获取视觉块信息
    est_html_info = _get_visual_block(ct)
    # 断言视觉块信息的类型为 "parallel"
    assert est_html_info.kind == "parallel"
    # 断言视觉块信息的估计器与变换器列表中的估计器相匹配
    assert est_html_info.estimators == tuple(trans[1] for trans in ct.transformers)
    # 断言视觉块信息的名称为 ("pca", "svd")
    assert est_html_info.names == ("pca", "svd")
    # 断言视觉块信息的名称细节为 (["num1", "num2"], [0, 3])


# 定义测试函数，用于测试管道估计器的 HTML 表示
def test_estimator_html_repr_pipeline():
    # 创建数值转换管道
    num_trans = Pipeline(
        steps=[("pass", "passthrough"), ("imputer", SimpleImputer(strategy="median"))]
    )

    # 创建分类转换管道
    cat_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", missing_values="empty")),
            ("one-hot", OneHotEncoder(drop="first")),
        ]
    )

    # 创建预处理管道，包括数值和分类转换器
    preprocess = ColumnTransformer(
        [
            ("num", num_trans, ["a", "b", "c", "d", "e"]),
            ("cat", cat_trans, [0, 1, 2, 3]),
        ]
    )

    # 创建特征联合对象，包括 "pca" 和 "tsvd"
    feat_u = FeatureUnion(
        [
            ("pca", PCA(n_components=1)),
            (
                "tsvd",
                Pipeline(
                    [
                        ("first", TruncatedSVD(n_components=3)),
                        ("select", SelectPercentile()),
                    ]
                ),
            ),
        ]
    )

    # 创建投票分类器，包括 "lr" 和 "mlp"
    clf = VotingClassifier(
        [
            ("lr", LogisticRegression(solver="lbfgs", random_state=1)),
            ("mlp", MLPClassifier(alpha=0.001)),
        ]
    )

    # 创建整体管道，包括预处理、特征联合和分类器
    pipe = Pipeline(
        [("preprocessor", preprocess), ("feat_u", feat_u), ("classifier", clf)]
    )
    # 获取管道的 HTML 表示
    html_output = estimator_html_repr(pipe)

    # 断言 HTML 输出中包含管道的字符串表示
    assert html.escape(str(pipe)) in html_output
    # 断言 HTML 输出中每个步骤的估计器都不显示更改
    for _, est in pipe.steps:
        assert (
            '<div class="sk-toggleable__content "><pre>' + html.escape(str(est))
        ) in html_output

    # 低级别的估计器不显示更改
    # 使用 config_context 上下文管理器，设置 print_changed_only=True，这可能是配置环境的一部分
    with config_context(print_changed_only=True):
        # 断言 HTML 输出中转换后的 "pass" 字段的转义字符串存在
        assert html.escape(str(num_trans["pass"])) in html_output
        # 断言 HTML 输出中包含 "passthrough</label>" 字符串
        assert "passthrough</label>" in html_output
        # 断言 HTML 输出中转换后的 "imputer" 字段的转义字符串存在
        assert html.escape(str(num_trans["imputer"])) in html_output

        # 遍历预处理管道中的转换器列表，每个转换器的列都存在于 HTML 输出的 <pre> 标签中
        for _, _, cols in preprocess.transformers:
            assert f"<pre>{html.escape(str(cols))}</pre>" in html_output

        # 特征联合
        for name, _ in feat_u.transformer_list:
            # 断言 HTML 输出中存在以特定标签包装的特征联合名称的转义字符串
            assert f"<label>{html.escape(name)}</label>" in html_output

        # 获取特征联合中第一个转换器（通常是 PCA）
        pca = feat_u.transformer_list[0][1]
        # 断言 HTML 输出中存在以 <pre> 标签包装的 PCA 转换器的转义字符串表示
        assert f"<pre>{html.escape(str(pca))}</pre>" in html_output

        # 获取特征联合中第二个转换器（通常是 t-SVD）
        tsvd = feat_u.transformer_list[1][1]
        # 获取 t-SVD 转换器中的 "first" 和 "select" 属性
        first = tsvd["first"]
        select = tsvd["select"]
        # 断言 HTML 输出中存在以 <pre> 标签包装的 t-SVD 转换器的 "first" 和 "select" 属性的转义字符串表示
        assert f"<pre>{html.escape(str(first))}</pre>" in html_output
        assert f"<pre>{html.escape(str(select))}</pre>" in html_output

        # 投票分类器
        for name, est in clf.estimators:
            # 断言 HTML 输出中存在以特定标签包装的分类器名称的转义字符串
            assert f"<label>{html.escape(name)}</label>" in html_output
            # 断言 HTML 输出中存在以 <pre> 标签包装的分类器估计器的转义字符串表示
            assert f"<pre>{html.escape(str(est))}</pre>" in html_output

    # 验证 HTML 输出中实现了 "prefers-color-scheme" 特性
    assert "prefers-color-scheme" in html_output
@pytest.mark.parametrize("final_estimator", [None, LinearSVC()])
# 使用pytest的参数化装饰器，final_estimator参数分别为None和LinearSVC()
def test_stacking_classifier(final_estimator):
    # 定义基础分类器列表
    estimators = [
        ("mlp", MLPClassifier(alpha=0.001)),  # 多层感知机分类器
        ("tree", DecisionTreeClassifier()),  # 决策树分类器
    ]
    # 创建堆叠分类器对象clf
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

    # 获取clf的HTML表示
    html_output = estimator_html_repr(clf)

    # 断言：clf的字符串表示应该在html_output中转义
    assert html.escape(str(clf)) in html_output

    # 如果final_estimator的默认值从LogisticRegression改变，这里需要更新
    if final_estimator is None:
        assert "LogisticRegression(" in html_output
    else:
        assert final_estimator.__class__.__name__ in html_output


@pytest.mark.parametrize("final_estimator", [None, LinearSVR()])
# 使用pytest的参数化装饰器，final_estimator参数分别为None和LinearSVR()
def test_stacking_regressor(final_estimator):
    # 创建堆叠回归器对象reg
    reg = StackingRegressor(
        estimators=[("svr", LinearSVR())], final_estimator=final_estimator
    )
    # 获取reg的HTML表示
    html_output = estimator_html_repr(reg)

    # 断言：reg的第一个估计器的字符串表示应该在html_output中转义
    assert html.escape(str(reg.estimators[0][0])) in html_output

    # 正则表达式模式匹配
    p = (
        r'<label for="sk-estimator-id-[0-9]*"'
        r' class="sk-toggleable__label (fitted)? sk-toggleable__label-arrow ">'
        r"&nbsp;LinearSVR"
    )
    re_compiled = re.compile(p)
    # 断言：在html_output中应该能找到匹配的标签
    assert re_compiled.search(html_output)

    if final_estimator is None:
        # 正则表达式模式匹配
        p = (
            r'<label for="sk-estimator-id-[0-9]*"'
            r' class="sk-toggleable__label (fitted)? sk-toggleable__label-arrow ">'
            r"&nbsp;RidgeCV"
        )
        re_compiled = re.compile(p)
        # 断言：在html_output中应该能找到匹配的标签
        assert re_compiled.search(html_output)
    else:
        # 断言：final_estimator的类名应该在html_output中转义
        assert html.escape(final_estimator.__class__.__name__) in html_output


def test_birch_duck_typing_meta():
    # 测试Birch与鸭子类型元估计器的元信息
    birch = Birch(n_clusters=AgglomerativeClustering(n_clusters=3))
    # 获取birch的HTML表示
    html_output = estimator_html_repr(birch)

    # 使用配置上下文，只打印更改部分
    with config_context(print_changed_only=True):
        # 断言：html_output中应该包含birch.n_clusters的字符串表示
        assert f"<pre>{html.escape(str(birch.n_clusters))}" in html_output
        # 断言：html_output中应该包含"AgglomerativeClustering"标签
        assert "AgglomerativeClustering</label>" in html_output

    # 断言：html_output中应该包含birch的字符串表示
    assert f"<pre>{html.escape(str(birch))}" in html_output


def test_ovo_classifier_duck_typing_meta():
    # 测试OVO与鸭子类型元估计器的元信息
    ovo = OneVsOneClassifier(LinearSVC(penalty="l1"))
    # 获取ovo的HTML表示
    html_output = estimator_html_repr(ovo)

    # 使用配置上下文，只打印更改部分
    with config_context(print_changed_only=True):
        # 断言：html_output中应该包含ovo.estimator的字符串表示
        assert f"<pre>{html.escape(str(ovo.estimator))}" in html_output
        # 正则表达式模式匹配
        p = (
            r'<label for="sk-estimator-id-[0-9]*" '
            r'class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;LinearSVC'
        )
        re_compiled = re.compile(p)
        # 断言：在html_output中应该能找到匹配的标签
        assert re_compiled.search(html_output)

    # 断言：html_output中应该包含ovo的字符串表示
    assert f"<pre>{html.escape(str(ovo))}" in html_output


def test_duck_typing_nested_estimator():
    # 测试随机搜索与鸭子类型嵌套估计器的元信息
    # 创建一个 KernelRidge 对象，使用 ExpSineSquared 作为核函数
    kernel_ridge = KernelRidge(kernel=ExpSineSquared())
    
    # 定义一个包含参数搜索空间的字典，该字典指定了参数 alpha 的候选值
    param_distributions = {"alpha": [1, 2]}
    
    # 使用随机搜索 RandomizedSearchCV 来优化 KernelRidge 模型的参数
    kernel_ridge_tuned = RandomizedSearchCV(
        kernel_ridge,  # 使用之前定义的 KernelRidge 对象
        param_distributions=param_distributions,  # 指定参数搜索空间
    )
    
    # 生成优化后的模型的 HTML 表示形式
    html_output = estimator_html_repr(kernel_ridge_tuned)
    
    # 断言检查生成的 HTML 是否包含特定的字符串，用来验证输出是否符合预期
    assert "estimator: KernelRidge</label>" in html_output
@pytest.mark.parametrize("print_changed_only", [True, False])
# 使用 pytest 的 parametrize 装饰器，参数化测试，分别测试 print_changed_only 为 True 和 False 的情况
def test_one_estimator_print_change_only(print_changed_only):
    pca = PCA(n_components=10)
    # 创建一个 PCA 模型对象，设置主成分数量为 10

    with config_context(print_changed_only=print_changed_only):
        # 使用 config_context 上下文管理器，设置 print_changed_only 参数为当前参数化的值
        pca_repr = html.escape(str(pca))
        # 对 PCA 模型的字符串表示进行 HTML 转义处理
        html_output = estimator_html_repr(pca)
        # 获取 PCA 模型的 HTML 表示
        assert pca_repr in html_output
        # 断言：转义后的 PCA 模型字符串应包含在 HTML 输出中


def test_fallback_exists():
    """Check that repr fallback is in the HTML."""
    pca = PCA(n_components=10)
    # 创建一个 PCA 模型对象，设置主成分数量为 10
    html_output = estimator_html_repr(pca)
    # 获取 PCA 模型的 HTML 表示

    assert (
        f'<div class="sk-text-repr-fallback"><pre>{html.escape(str(pca))}'
        in html_output
    )
    # 断言：转义后的 PCA 模型字符串应包含在 HTML 输出的特定格式中


def test_show_arrow_pipeline():
    """Show arrow in pipeline for top level in pipeline"""
    pipe = Pipeline([("scale", StandardScaler()), ("log_Reg", LogisticRegression())])
    # 创建一个 Pipeline 对象，包含数据标准化和逻辑回归两个步骤

    html_output = estimator_html_repr(pipe)
    # 获取 Pipeline 对象的 HTML 表示
    assert (
        'class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;&nbsp;Pipeline'
        in html_output
    )
    # 断言：HTML 输出中应包含 Pipeline 的特定标签和样式


def test_invalid_parameters_in_stacking():
    """Invalidate stacking configuration uses default repr.

    Non-regression test for #24009.
    """
    stacker = StackingClassifier(estimators=[])
    # 创建一个 StackingClassifier 对象，没有包含任何估计器

    html_output = estimator_html_repr(stacker)
    # 获取 StackingClassifier 对象的 HTML 表示
    assert html.escape(str(stacker)) in html_output
    # 断言：StackingClassifier 对象的字符串表示应包含在 HTML 输出中，经过 HTML 转义处理


def test_estimator_get_params_return_cls():
    """Check HTML repr works where a value in get_params is a class."""

    class MyEstimator:
        def get_params(self, deep=False):
            return {"inner_cls": LogisticRegression}

    est = MyEstimator()
    # 创建一个自定义的 MyEstimator 类，其 get_params 方法返回一个类对象 LogisticRegression

    assert "MyEstimator" in estimator_html_repr(est)
    # 断言：HTML 表示中应包含 MyEstimator


def test_estimator_html_repr_unfitted_vs_fitted():
    """Check that we have the information that the estimator is fitted or not in the
    HTML representation.
    """

    class MyEstimator(BaseEstimator):
        def fit(self, X, y):
            self.fitted_ = True
            return self

    X, y = load_iris(return_X_y=True)
    estimator = MyEstimator()
    # 创建一个自定义的 MyEstimator 类，包含 fit 方法用于训练

    assert "<span>Not fitted</span>" in estimator_html_repr(estimator)
    # 断言：HTML 表示中应包含 "Not fitted" 表示未训练状态
    estimator.fit(X, y)
    assert "<span>Fitted</span>" in estimator_html_repr(estimator)
    # 断言：HTML 表示中应包含 "Fitted" 表示已训练状态


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), slice(0, 3))),
            LogisticRegression(),
        ),
    ],
)
# 使用 pytest 的 parametrize 装饰器，参数化测试不同的估计器对象
def test_estimator_html_repr_fitted_icon(estimator):
    """Check that we are showing the fitted status icon only once."""
    pattern = '<span class="sk-estimator-doc-link ">i<span>Not fitted</span></span>'
    # 定义未训练状态的 HTML 表示模式

    assert estimator_html_repr(estimator).count(pattern) == 1
    # 断言：HTML 表示中未训练状态的图标只出现一次
    X, y = load_iris(return_X_y=True)
    estimator.fit(X, y)
    pattern = '<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span>'
    # 定义已训练状态的 HTML 表示模式

    assert estimator_html_repr(estimator).count(pattern) == 1
    # 断言：HTML 表示中已训练状态的图标只出现一次
def test_html_documentation_link_mixin_sklearn(mock_version):
    """Check the behaviour of the `_HTMLDocumentationLinkMixin` class for scikit-learn
    default.
    """

    # mock the `__version__` where the mixin is located
    # 使用 mock 版本来模拟 mixin 所在的 scikit-learn 版本
    with patch("sklearn.utils._estimator_html_repr.__version__", mock_version):
        # 创建 `_HTMLDocumentationLinkMixin` 实例
        mixin = _HTMLDocumentationLinkMixin()

        # 断言 `_doc_link_module` 属性是否为 "sklearn"
        assert mixin._doc_link_module == "sklearn"
        # 解析 mock 版本号
        sklearn_version = parse_version(mock_version)
        # 如果版本号是开发版，则版本为 "dev"，否则为主版本号和次版本号
        if sklearn_version.dev is None:
            version = f"{sklearn_version.major}.{sklearn_version.minor}"
        else:
            version = "dev"
        # 断言 `_doc_link_template` 属性是否为预期的 scikit-learn 文档链接模板
        assert (
            mixin._doc_link_template
            == f"https://scikit-learn.org/{version}/modules/generated/"
            "{estimator_module}.{estimator_name}.html"
        )
        # 断言 `_get_doc_link()` 方法返回的链接是否符合预期的 scikit-learn 文档链接格式
        assert (
            mixin._get_doc_link()
            == f"https://scikit-learn.org/{version}/modules/generated/"
            "sklearn.utils._HTMLDocumentationLinkMixin.html"
        )


@pytest.mark.parametrize(
    "module_path,expected_module",
    [
        ("prefix.mymodule", "prefix.mymodule"),
        ("prefix._mymodule", "prefix"),
        ("prefix.mypackage._mymodule", "prefix.mypackage"),
        ("prefix.mypackage._mymodule.submodule", "prefix.mypackage"),
        ("prefix.mypackage.mymodule.submodule", "prefix.mypackage.mymodule.submodule"),
    ],
)
def test_html_documentation_link_mixin_get_doc_link(module_path, expected_module):
    """Check the behaviour of the `_get_doc_link` with various parameter."""

    class FooBar(_HTMLDocumentationLinkMixin):
        pass

    # 将 FooBar 类的模块路径设置为参数中的 module_path
    FooBar.__module__ = module_path
    # 创建 FooBar 实例
    est = FooBar()
    # 如果设置了 `_doc_link`，则预期根据 estimator 推断出模块和名称
    est._doc_link_module = "prefix"
    est._doc_link_template = (
        "https://website.com/{estimator_module}.{estimator_name}.html"
    )
    # 断言 `_get_doc_link()` 方法返回的链接是否符合预期的格式
    assert est._get_doc_link() == f"https://website.com/{expected_module}.FooBar.html"


def test_html_documentation_link_mixin_get_doc_link_out_of_library():
    """Check the behaviour of the `_get_doc_link` with various parameter."""
    # 创建 `_HTMLDocumentationLinkMixin` 实例
    mixin = _HTMLDocumentationLinkMixin()

    # 如果 `_doc_link_module` 不是 estimator（这里是 mixin）的根模块，则返回空字符串
    mixin._doc_link_module = "xxx"
    # 断言 `_get_doc_link()` 方法返回的链接是否为空字符串
    assert mixin._get_doc_link() == ""


def test_html_documentation_link_mixin_doc_link_url_param_generator():
    # 创建 `_HTMLDocumentationLinkMixin` 实例
    mixin = _HTMLDocumentationLinkMixin()
    # 如果提供了自定义的可调用函数，则可以绕过链接生成器
    mixin._doc_link_template = (
        "https://website.com/{my_own_variable}.{another_variable}.html"
    )

    # 定义一个 URL 参数生成器的示例函数
    def url_param_generator(estimator):
        return {
            "my_own_variable": "value_1",
            "another_variable": "value_2",
        }
    # 将 mixin 对象的 _doc_link_url_param_generator 属性设置为 url_param_generator 函数
    mixin._doc_link_url_param_generator = url_param_generator

    # 断言 mixin 对象的 _get_doc_link() 方法返回的结果为指定的 URL 字符串
    assert mixin._get_doc_link() == "https://website.com/value_1.value_2.html"
# 定义一个 pytest 的 fixture，用于在测试期间设置非 UTF-8 语言环境

@pytest.fixture
def set_non_utf8_locale():
    """Pytest fixture to set non utf-8 locale during the test.
    
    The locale is set to the original one after the test has run.
    """
    
    try:
        # 尝试将语言环境设置为 "C"
        locale.setlocale(locale.LC_CTYPE, "C")
    except locale.Error:
        # 如果系统不支持 "C" 语言环境，则跳过测试
        pytest.skip("'C' locale is not available on this OS")
    
    yield  # 执行测试之前的部分结束
    
    # 重置语言环境为原始设置。Python 在启动时会调用 setlocale(LC_TYPE, "")
    # 参考 https://docs.python.org/3/library/locale.html#background-details-hints-tips-and-caveats
    # 这假设在此期间没有进行其他的语言环境更改。在某些平台上，尝试使用类似
    # locale.setlocale(locale.LC_CTYPE, locale.getlocale()) 来恢复语言环境时，
    # 可能会引发 locale.Error: unsupported locale setting 的错误。
    locale.setlocale(locale.LC_CTYPE, "")


def test_non_utf8_locale(set_non_utf8_locale):
    """Checks that utf8 encoding is used when reading the CSS file.
    
    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/27725
    """
    
    _get_css_style()  # 调用函数来检查在读取 CSS 文件时是否使用了 UTF-8 编码
```