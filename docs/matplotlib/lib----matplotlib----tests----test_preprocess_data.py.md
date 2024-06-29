# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_preprocess_data.py`

```
# 导入正则表达式和系统模块
import re
import sys

# 导入numpy库，并使用别名np
import numpy as np

# 导入pytest库用于测试
import pytest

# 从matplotlib库中导入模块和函数
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal

# Notes on testing the plotting functions itself
# *   the individual decorated plotting functions are tested in 'test_axes.py'
# *   that pyplot functions accept a data kwarg is only tested in
#     test_axes.test_pie_linewidth_0


# this gets used in multiple tests, so define it here

# 使用_preprocess_data装饰器对plot_func进行预处理，替换x和y，标签命名为"y"
@_preprocess_data(replace_names=["x", "y"], label_namer="y")
def plot_func(ax, x, y, ls="x", label=None, w="xyz"):
    # 返回格式化的字符串，包括x，y的列表形式，ls的值，w的值，以及label的值
    return f"x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}"


# 定义函数列表和函数ID列表，用于参数化测试
all_funcs = [plot_func]
all_func_ids = ['plot_func']


# 定义测试函数test_compiletime_checks
def test_compiletime_checks():
    """Test decorator invocations -> no replacements."""

    # 定义多个函数用于测试
    def func(ax, x, y): pass
    def func_args(ax, x, y, *args): pass
    def func_kwargs(ax, x, y, **kwargs): pass
    def func_no_ax_args(*args, **kwargs): pass

    # 使用_preprocess_data装饰器对func和func_kwargs进行预处理，替换x和y
    # 这是可以的，因为已提供了足够的信息来替换所有的参数
    _preprocess_data(replace_names=["x", "y"])(func)
    _preprocess_data(replace_names=["x", "y"])(func_kwargs)

    # 对func_args使用_preprocess_data装饰器进行预处理，替换x和y
    # 这也是可以的，因为提供了足够的信息来替换所有的参数
    _preprocess_data(replace_names=["x", "y"])(func_args)

    # 没有提供足够的信息来替换所有参数，因此会引发AssertionError
    with pytest.raises(AssertionError):
        _preprocess_data(replace_names=["x", "y", "z"])(func_args)

    # 完全没有进行替换，因此一切正常
    _preprocess_data(replace_names=[], label_namer=None)(func)
    _preprocess_data(replace_names=[], label_namer=None)(func_args)
    _preprocess_data(replace_names=[], label_namer=None)(func_kwargs)
    _preprocess_data(replace_names=[], label_namer=None)(func_no_ax_args)

    # label namer为未知，因此会引发AssertionError
    with pytest.raises(AssertionError):
        _preprocess_data(label_namer="z")(func)

    with pytest.raises(AssertionError):
        _preprocess_data(label_namer="z")(func_args)


# 使用@pytest.mark.parametrize装饰器，对all_funcs中的函数进行参数化测试
@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_without_data(func):
    """Test without data -> no replacements."""

    # 对函数的返回结果进行断言，验证其正确性
    assert (func(None, "x", "y") ==
            "x: ['x'], y: ['y'], ls: x, w: xyz, label: None")
    assert (func(None, x="x", y="y") ==
            "x: ['x'], y: ['y'], ls: x, w: xyz, label: None")
    assert (func(None, "x", "y", label="") ==
            "x: ['x'], y: ['y'], ls: x, w: xyz, label: ")
    assert (func(None, "x", "y", label="text") ==
            "x: ['x'], y: ['y'], ls: x, w: xyz, label: text")
    assert (func(None, x="x", y="y", label="") ==
            "x: ['x'], y: ['y'], ls: x, w: xyz, label: ")
    assert (func(None, x="x", y="y", label="text") ==
            "x: ['x'], y: ['y'], ls: x, w: xyz, label: text")


# 使用@pytest.mark.parametrize装饰器，对all_funcs中的函数进行参数化测试
@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_dict_input(func):
    """Tests with dict input, unpacking via preprocess_pipeline"""
    # 创建一个包含键值对 'a': 1 和 'b': 2 的字典对象
    data = {'a': 1, 'b': 2}
    # 断言语句，验证调用 func 函数后返回的结果是否符合预期
    assert (func(None, data.keys(), data.values()) ==
            "x: ['a', 'b'], y: [1, 2], ls: x, w: xyz, label: None")
@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_dict_data(func):
    """Test with dict data -> label comes from the value of 'x' parameter."""
    # 定义测试数据，一个包含不同键值对的字典
    data = {"a": [1, 2], "b": [8, 9], "w": "NOT"}
    # 测试函数调用，验证返回结果是否符合预期
    assert (func(None, "a", "b", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert (func(None, x="a", y="b", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert (func(None, "a", "b", label="", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert (func(None, "a", "b", label="text", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
    assert (func(None, x="a", y="b", label="", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert (func(None, x="a", y="b", label="text", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")


@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_dict_data_not_in_data(func):
    """Test the case that one var is not in data -> half replaces, half kept"""
    # 定义测试数据，一个缺少'b'键值对的字典
    data = {"a": [1, 2], "w": "NOT"}
    # 测试函数调用，验证返回结果是否符合预期
    assert (func(None, "a", "b", data=data) ==
            "x: [1, 2], y: ['b'], ls: x, w: xyz, label: b")
    assert (func(None, x="a", y="b", data=data) ==
            "x: [1, 2], y: ['b'], ls: x, w: xyz, label: b")
    assert (func(None, "a", "b", label="", data=data) ==
            "x: [1, 2], y: ['b'], ls: x, w: xyz, label: ")
    assert (func(None, "a", "b", label="text", data=data) ==
            "x: [1, 2], y: ['b'], ls: x, w: xyz, label: text")
    assert (func(None, x="a", y="b", label="", data=data) ==
            "x: [1, 2], y: ['b'], ls: x, w: xyz, label: ")
    assert (func(None, x="a", y="b", label="text", data=data) ==
            "x: [1, 2], y: ['b'], ls: x, w: xyz, label: text")


@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_pandas_data(func, pd):
    """Test with pandas dataframe -> label comes from ``data["col"].name``."""
    # 定义测试数据，一个包含不同列的 Pandas 数据帧
    data = pd.DataFrame({"a": np.array([1, 2], dtype=np.int32),
                         "b": np.array([8, 9], dtype=np.int32),
                         "w": ["NOT", "NOT"]})

    # 测试函数调用，验证返回结果是否符合预期
    assert (func(None, "a", "b", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert (func(None, x="a", y="b", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert (func(None, "a", "b", label="", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert (func(None, "a", "b", label="text", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
    assert (func(None, x="a", y="b", label="", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert (func(None, x="a", y="b", label="text", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
# 定义一个测试函数，用于测试不带 "replace_names" 参数时，所有变量都应该被替换
def test_function_call_replace_all():
    """Test without a "replace_names" argument, all vars should be replaced."""
    # 定义一个数据字典
    data = {"a": [1, 2], "b": [8, 9], "x": "xyz"

    # 定义一个装饰器，预处理数据，指定标签名为 "y"
    @_preprocess_data(label_namer="y")
    # 定义一个内部函数，接收多个参数，包括 ax, x, y, ls（默认值为 "x"），label（默认值为 None），w（默认值为 "NOT"），并返回一个格式化字符串
    def func_replace_all(ax, x, y, ls="x", label=None, w="NOT"):
        return f"x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}"

    # 断言，调用 func_replace_all 函数，验证返回结果是否符合预期
    assert (func_replace_all(None, "a", "b", w="x", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert (func_replace_all(None, x="a", y="b", w="x", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert (func_replace_all(None, "a", "b", w="x", label="", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert (
        func_replace_all(None, "a", "b", w="x", label="text", data=data) ==
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
    assert (
        func_replace_all(None, x="a", y="b", w="x", label="", data=data) ==
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert (
        func_replace_all(None, x="a", y="b", w="x", label="text", data=data) ==
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")


# 定义一个测试函数，测试在 "label_namer=None" 时，不替换标签
def test_no_label_replacements():
    """Test with "label_namer=None" -> no label replacement at all."""

    # 定义一个装饰器，预处理数据，替换名称为 ["x", "y"]，标签名为 None
    @_preprocess_data(replace_names=["x", "y"], label_namer=None)
    # 定义一个内部函数，接收多个参数，包括 ax, x, y，ls（默认为 "x"），label（默认为 None），w（默认为 "xyz"），并返回一个格式化字符串
    def func_no_label(ax, x, y, ls="x", label=None, w="xyz"):
        return f"x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}"

    # 定义一个数据字典
    data = {"a": [1, 2], "b": [8, 9], "w": "NOT"}
    # 断言，调用 func_no_label 函数，验证返回结果是否符合预期
    assert (func_no_label(None, "a", "b", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None")
    assert (func_no_label(None, x="a", y="b", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None")
    assert (func_no_label(None, "a", "b", label="", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert (func_no_label(None, "a", "b", label="text", data=data) ==
            "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")


# 定义一个测试函数，测试传入的参数多于位置参数
def test_more_args_than_pos_parameter():
    @_preprocess_data(replace_names=["x", "y"], label_namer="y")
    # 定义一个内部函数，接收多个参数，包括 ax, x, y，z（默认为 1），但此处抛出 TypeError 异常
    def func(ax, x, y, z=1):
        pass

    # 定义一个数据字典
    data = {"a": [1, 2], "b": [8, 9], "w": "NOT"}
    # 使用 pytest 模块断言，调用 func 函数时传入多余的参数，是否抛出 TypeError 异常
    with pytest.raises(TypeError):
        func(None, "a", "b", "z", "z", data=data)


# 定义一个测试函数，测试在函数文档字符串中添加说明
def test_docstring_addition():
    @_preprocess_data()
    # 定义一个内部函数，接收多个参数，包括 ax，*args 和 **kwargs，但未提供具体实现
    def funcy(ax, *args, **kwargs):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """

    # 断言，验证 funcy 函数的文档字符串是否包含特定的文本
    assert re.search(r"all parameters also accept a string", funcy.__doc__)
    assert not re.search(r"the following parameters", funcy.__doc__)

    @_preprocess_data(replace_names=[])
    # 定义一个内部函数，接收多个参数，包括 ax, x, y, z 和 bar（默认为 None），但未提供具体实现
    def funcy(ax, x, y, z, bar=None):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """
    # 断言：确保在函数文档字符串中找不到指定的文本 "all parameters also accept a string"
    assert not re.search(r"all parameters also accept a string", funcy.__doc__)

    # 断言：确保在函数文档字符串中找不到指定的文本 "the following parameters"
    assert not re.search(r"the following parameters", funcy.__doc__)

    # 使用装饰器 @_preprocess_data，并替换参数名为 ["bar"]，定义函数 funcy
    @_preprocess_data(replace_names=["bar"])
    def funcy(ax, x, y, z, bar=None):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """

    # 断言：确保在函数文档字符串中找不到指定的文本 "all parameters also accept a string"
    assert not re.search(r"all parameters also accept a string", funcy.__doc__)

    # 断言：确保在函数文档字符串中找不到指定的文本 "the following parameters .*: \*bar\*\."
    assert not re.search(r"the following parameters .*: \*bar\*\.", funcy.__doc__)

    # 使用装饰器 @_preprocess_data，并替换参数名为 ["x", "t"]，定义函数 funcy
    @_preprocess_data(replace_names=["x", "t"])
    def funcy(ax, x, y, z, t=None):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """

    # 断言：确保在函数文档字符串中找不到指定的文本 "all parameters also accept a string"
    assert not re.search(r"all parameters also accept a string", funcy.__doc__)

    # 断言：确保在函数文档字符串中找不到指定的文本 "the following parameters .*: \*x\*, \*t\*\."
    assert not re.search(r"the following parameters .*: \*x\*, \*t\*\.", funcy.__doc__)
# 定义一个测试类，用于测试不同类型的绘图方法
class TestPlotTypes:

    # 定义一个包含三种绘图函数的列表
    plotters = [Axes.scatter, Axes.bar, Axes.plot]

    # 使用参数化装饰器，对每种绘图函数进行测试
    @pytest.mark.parametrize('plotter', plotters)
    # 使用自定义装饰器检查生成的图像是否与参考图像相等，保存为扩展名为 PNG 的文件
    @check_figures_equal(extensions=['png'])
    # 测试以字典展开作为数据输入的情况
    def test_dict_unpack(self, plotter, fig_test, fig_ref):
        # 定义 x 和 y 轴的数据
        x = [1, 2, 3]
        y = [4, 5, 6]
        # 使用 zip 函数将 x 和 y 轴数据打包成字典
        ddict = dict(zip(x, y))

        # 调用绘图函数，使用字典的键和值作为数据
        plotter(fig_test.subplots(),
                ddict.keys(), ddict.values())
        # 对参考图像使用真实的 x 和 y 数据进行绘图
        plotter(fig_ref.subplots(), x, y)

    # 使用参数化装饰器，对每种绘图函数进行测试
    @pytest.mark.parametrize('plotter', plotters)
    # 使用自定义装饰器检查生成的图像是否与参考图像相等，保存为扩展名为 PNG 的文件
    # 测试使用 data 关键字参数作为数据输入的情况
    def test_data_kwarg(self, plotter, fig_test, fig_ref):
        # 定义 x 和 y 轴的数据
        x = [1, 2, 3]
        y = [4, 5, 6]

        # 调用绘图函数，使用 data 参数传递 x 和 y 数据
        plotter(fig_test.subplots(), 'xval', 'yval',
                data={'xval': x, 'yval': y})
        # 对参考图像使用真实的 x 和 y 数据进行绘图
        plotter(fig_ref.subplots(), x, y)
```