# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_matplotlib.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 测试框架

from pandas import (  # 从 Pandas 库中导入以下模块：
    DataFrame,  # 数据帧对象
    IndexSlice,  # 多级索引切片对象
    Series,  # 序列对象
)

mpl = pytest.importorskip("matplotlib")  # 导入并检查 Matplotlib 库的可用性
pytest.importorskip("jinja2")  # 导入并检查 Jinja2 模板引擎的可用性

from pandas.io.formats.style import Styler  # 从 Pandas 样式模块导入 Styler 类

pytestmark = pytest.mark.usefixtures("mpl_cleanup")  # 使用 Pytest 标记，执行 Matplotlib 清理操作


@pytest.fixture  # Pytest 的测试夹具装饰器，定义数据帧对象的装置
def df():
    return DataFrame([[1, 2], [2, 4]], columns=["A", "B"])


@pytest.fixture  # Pytest 的测试夹具装饰器，定义数据帧样式的装置
def styler(df):
    return Styler(df, uuid_len=0)


@pytest.fixture  # Pytest 的测试夹具装饰器，定义空白数据帧对象的装置
def df_blank():
    return DataFrame([[0, 0], [0, 0]], columns=["A", "B"], index=["X", "Y"])


@pytest.fixture  # Pytest 的测试夹具装饰器，定义空白数据帧样式的装置
def styler_blank(df_blank):
    return Styler(df_blank, uuid_len=0)


@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])  # Pytest 参数化装饰器，测试渐变背景和文本颜色的函数
def test_function_gradient(styler, f):
    for c_map in [None, "YlOrRd"]:
        result = getattr(styler, f)(cmap=c_map)._compute().ctx  # 调用 Styler 对象的背景渐变或文本渐变方法，并计算上下文
        assert all("#" in x[0][1] for x in result.values())  # 断言所有结果的颜色值均为十六进制格式
        assert result[(0, 0)] == result[(0, 1)]  # 断言第一行的两列样式结果相同
        assert result[(1, 0)] == result[(1, 1)]  # 断言第二行的两列样式结果相同


@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])  # Pytest 参数化装饰器，测试背景渐变颜色函数
def test_background_gradient_color(styler, f):
    result = getattr(styler, f)(subset=IndexSlice[1, "A"])._compute().ctx  # 调用 Styler 对象的背景渐变或文本渐变方法，并计算上下文
    if f == "background_gradient":
        assert result[(1, 0)] == [("background-color", "#fff7fb"), ("color", "#000000")]  # 断言特定单元格的背景和文本颜色
    elif f == "text_gradient":
        assert result[(1, 0)] == [("color", "#fff7fb")]  # 断言特定单元格的文本颜色


@pytest.mark.parametrize(
    "axis, expected",  # Pytest 参数化装饰器，定义轴和期望结果列表
    [
        (0, ["low", "low", "high", "high"]),  # 对应轴 0 的期望结果
        (1, ["low", "high", "low", "high"]),  # 对应轴 1 的期望结果
        (None, ["low", "mid", "mid", "high"]),  # 对应无特定轴的期望结果
    ],
)
@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])  # Pytest 参数化装饰器，测试带轴的背景渐变或文本渐变函数
def test_background_gradient_axis(styler, axis, expected, f):
    if f == "background_gradient":
        colors = {  # 定义背景渐变颜色字典
            "low": [("background-color", "#f7fbff"), ("color", "#000000")],
            "mid": [("background-color", "#abd0e6"), ("color", "#000000")],
            "high": [("background-color", "#08306b"), ("color", "#f1f1f1")],
        }
    elif f == "text_gradient":
        colors = {  # 定义文本渐变颜色字典
            "low": [("color", "#f7fbff")],
            "mid": [("color", "#abd0e6")],
            "high": [("color", "#08306b")],
        }
    result = getattr(styler, f)(cmap="Blues", axis=axis)._compute().ctx  # 调用 Styler 对象的背景渐变或文本渐变方法，并计算上下文
    for i, cell in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):  # 遍历特定单元格列表
        assert result[cell] == colors[expected[i]]  # 断言特定单元格的样式结果与期望颜色相符
    [
        (
            "PuBu",  # 第一个元组的第一个元素，颜色主题名称
            {
                (4, 5): [("background-color", "#86b0d3"), ("color", "#000000")],  # PuBu 主题下 (4, 5) 单元格样式
                (4, 6): [("background-color", "#83afd3"), ("color", "#f1f1f1")],  # PuBu 主题下 (4, 6) 单元格样式
            },
        ),
        (
            "YlOrRd",  # 第二个元组的第一个元素，颜色主题名称
            {
                (4, 8): [("background-color", "#fd913e"), ("color", "#000000")],  # YlOrRd 主题下 (4, 8) 单元格样式
                (4, 9): [("background-color", "#fd8f3d"), ("color", "#f1f1f1")],  # YlOrRd 主题下 (4, 9) 单元格样式
            },
        ),
        (
            None,  # 第三个元组的第一个元素，无颜色主题
            {
                (7, 0): [("background-color", "#48c16e"), ("color", "#f1f1f1")],  # 默认主题下 (7, 0) 单元格样式
                (7, 1): [("background-color", "#4cc26c"), ("color", "#000000")],  # 默认主题下 (7, 1) 单元格样式
            },
        ),
    ],
@pytest.mark.parametrize(
    "axis, gmap, expected",
    [  # 参数化测试的参数定义开始
        (
            0,  # 第一个测试参数：axis为0
            [1, 2],  # 第二个测试参数：gmap为列表[1, 2]
            {  # 第三个测试参数：期望结果为字典，包含多个元组作为键和值
                (0, 0): [("background-color", "#fff7fb"), ("color", "#000000")],  # 键(0, 0)对应的期望样式
                (1, 0): [("background-color", "#023858"), ("color", "#f1f1f1")],  # 键(1, 0)对应的期望样式
                (0, 1): [("background-color", "#fff7fb"), ("color", "#000000")],  # 键(0, 1)对应的期望样式
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],  # 键(1, 1)对应的期望样式
            },
        ),  # 第一个测试参数定义结束
        (
            1,  # 第一个测试参数：axis为1
            [1, 2],  # 第二个测试参数：gmap为列表[1, 2]
            {  # 第三个测试参数：期望结果为字典，包含多个元组作为键和值
                (0, 0): [("background-color", "#fff7fb"), ("color", "#000000")],  # 键(0, 0)对应的期望样式
                (1, 0): [("background-color", "#fff7fb"), ("color", "#000000")],  # 键(1, 0)对应的期望样式
                (0, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],  # 键(0, 1)对应的期望样式
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],  # 键(1, 1)对应的期望样式
            },
        ),  # 第二个测试参数定义结束
        (
            None,  # 第一个测试参数：axis为None
            np.array([[2, 1], [1, 2]]),  # 第二个测试参数：gmap为NumPy数组
            {  # 第三个测试参数：期望结果为字典，包含多个元组作为键和值
                (0, 0): [("background-color", "#023858"), ("color", "#f1f1f1")],  # 键(0, 0)对应的期望样式
                (1, 0): [("background-color", "#fff7fb"), ("color", "#000000")],  # 键(1, 0)对应的期望样式
                (0, 1): [("background-color", "#fff7fb"), ("color", "#000000")],  # 键(0, 1)对应的期望样式
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],  # 键(1, 1)对应的期望样式
            },
        ),  # 第三个测试参数定义结束
    ],
)
def test_background_gradient_gmap_array(styler_blank, axis, gmap, expected):
    # 参数化测试函数，测试背景渐变功能
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute().ctx
    # 断言测试结果与期望值相等
    assert result == expected


@pytest.mark.parametrize(
    "gmap, axis", [([1, 2, 3], 0), ([1, 2], 1), (np.array([[1, 2], [1, 2]]), None)]
)
def test_background_gradient_gmap_array_raises(gmap, axis):
    # 测试当gmap作为转换后的ndarray时，形状不正确的情况
    df = DataFrame([[0, 0, 0], [0, 0, 0]])
    msg = "supplied 'gmap' is not correct shape"
    with pytest.raises(ValueError, match=msg):
        df.style.background_gradient(gmap=gmap, axis=axis)._compute()
    [
        DataFrame(  # 反转列
            [[2, 1], [1, 2]], columns=["B", "A"], index=["X", "Y"]
        ),
        DataFrame(  # 反转索引
            [[2, 1], [1, 2]], columns=["A", "B"], index=["Y", "X"]
        ),
        DataFrame(  # 反转索引和列
            [[1, 2], [2, 1]], columns=["B", "A"], index=["Y", "X"]
        ),
        DataFrame(  # 添加不必要的列
            [[1, 2, 3], [2, 1, 3]], columns=["A", "B", "C"], index=["X", "Y"]
        ),
        DataFrame(  # 添加不必要的索引
            [[1, 2], [2, 1], [3, 3]], columns=["A", "B"], index=["X", "Y", "Z"]
        ),
    ],
@pytest.mark.parametrize(
    "subset, exp_gmap",  # 定义参数化测试的参数，subset 是数据切片，exp_gmap 是期望的映射结果
    [
        (None, [[1, 2], [2, 1]]),  # 若无subset，期望的映射是整个数据的映射
        (["A"], [[1], [2]]),  # 只选择数据和映射中的"A"列进行切片
        (["B", "A"], [[2, 1], [1, 2]]),  # 反转数据中的列顺序
        (IndexSlice["X", :], [[1, 2]]),  # 只选择数据和映射中的"X"索引进行切片
        (IndexSlice[["Y", "X"], :], [[2, 1], [1, 2]]),  # 反转数据中的索引顺序
    ],
)
def test_background_gradient_gmap_dataframe_align(styler_blank, gmap, subset, exp_gmap):
    # 测试确保给定的 DataFrame gmap 能与数据对齐，包括指定的 subset
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap, subset=subset)
    result = styler_blank.background_gradient(axis=None, gmap=gmap, subset=subset)
    assert expected._compute().ctx == result._compute().ctx


@pytest.mark.parametrize(
    "gmap, axis, exp_gmap",
    [
        (Series([2, 1], index=["Y", "X"]), 0, [[1, 1], [2, 2]]),  # 反转索引顺序
        (Series([2, 1], index=["B", "A"]), 1, [[1, 2], [1, 2]]),  # 反转列顺序
        (Series([1, 2, 3], index=["X", "Y", "Z"]), 0, [[1, 1], [2, 2]]),  # 添加索引
        (Series([1, 2, 3], index=["A", "B", "C"]), 1, [[1, 2], [1, 2]]),  # 添加列
    ],
)
def test_background_gradient_gmap_series_align(styler_blank, gmap, axis, exp_gmap):
    # 测试确保给定的 Series gmap 能与数据对齐，包括指定的 axis
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap)._compute()
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute()
    assert expected.ctx == result.ctx


@pytest.mark.parametrize("axis", [1, 0])
def test_background_gradient_gmap_wrong_dataframe(styler_blank, axis):
    # 测试当给定的 gmap 是 DataFrame 但 axis 设置错误时的情况
    gmap = DataFrame([[1, 2], [2, 1]], columns=["A", "B"], index=["X", "Y"])
    msg = "'gmap' is a DataFrame but underlying data for operations is a Series"
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=axis)._compute()


def test_background_gradient_gmap_wrong_series(styler_blank):
    # 测试当给定的 gmap 是 Series 但 axis 设置错误时的情况
    msg = "'gmap' is a Series but underlying data for operations is a DataFrame"
    gmap = Series([1, 2], index=["X", "Y"])
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=None)._compute()


def test_background_gradient_nullable_dtypes():
    # GH 50712
    # 测试空值数据类型在背景渐变中的处理
    df1 = DataFrame([[1], [0], [np.nan]], dtype=float)
    df2 = DataFrame([[1], [0], [None]], dtype="Int64")

    ctx1 = df1.style.background_gradient()._compute().ctx
    ctx2 = df2.style.background_gradient()._compute().ctx
    assert ctx1 == ctx2


@pytest.mark.parametrize(
    "cmap",
    ["PuBu", mpl.colormaps["PuBu"]],
)
def test_bar_colormap(cmap):
    # 测试条形图颜色映射的不同设置
    data = DataFrame([[1, 2], [3, 4]])
    ctx = data.style.bar(cmap=cmap, axis=None)._compute().ctx
    # 定义一个字典 pubu_colors，包含四组键值对，每组键是一个元组 (0或1, 0或1)，值是对应的颜色代码
    pubu_colors = {
        (0, 0): "#d0d1e6",
        (1, 0): "#056faf",
        (0, 1): "#73a9cf",
        (1, 1): "#023858",
    }
    # 遍历 pubu_colors 字典中的每对键值对
    for k, v in pubu_colors.items():
        # 断言 pubu_colors 中每个值 v 是否存在于 ctx[k][1][1] 中，如果不存在会触发 AssertionError
        assert v in ctx[k][1][1]
# 定义一个测试函数，用于验证在DataFrame样式设置中，bar图的颜色参数是否符合预期，若不符合则抛出异常
def test_bar_color_raises(df):
    # 定义异常消息，指出`color`必须是字符串、字符串列表或元组中包含两个字符串
    msg = "`color` must be string or list or tuple of 2 strings"
    
    # 断言：当颜色参数为集合类型时，抛出值错误异常，异常消息为预定义的msg
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color={"a", "b"}).to_html()
    
    # 断言：当颜色参数为列表且包含超过两个元素时，抛出值错误异常，异常消息为预定义的msg
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=["a", "b", "c"]).to_html()

    # 重新定义异常消息，指出`color`和`cmap`不能同时给定
    msg = "`color` and `cmap` cannot both be given"
    
    # 断言：当同时给定颜色和色彩映射参数时，抛出值错误异常，异常消息为预定义的msg
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color="something", cmap="something else").to_html()

# 使用pytest.mark.parametrize装饰器，对绘图方法进行参数化测试，验证在不同的绘图方法下的色彩映射实例设置是否正常
@pytest.mark.parametrize("plot_method", ["scatter", "hexbin"])
def test_pass_colormap_instance(df, plot_method):
    # 创建一个自定义的色彩映射实例，包含两种颜色
    cmap = mpl.colors.ListedColormap([[1, 1, 1], [0, 0, 0]])
    
    # 向DataFrame添加新列"c"，其值为列"A"和"B"的和
    df["c"] = df.A + df.B
    
    # 准备绘图方法的关键字参数字典，包括x轴、y轴、颜色、色彩映射等设置
    kwargs = {"x": "A", "y": "B", "c": "c", "colormap": cmap}
    
    # 如果绘图方法是hexbin，则将原本的"c"键名改为"C"，并删除原来的"c"键
    if plot_method == "hexbin":
        kwargs["C"] = kwargs.pop("c")
    
    # 调用DataFrame的绘图方法，根据不同的plot_method绘制相应的图形，传入前面准备好的关键字参数
    getattr(df.plot, plot_method)(**kwargs)
```