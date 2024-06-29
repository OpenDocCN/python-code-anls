# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_bar.py`

```
# 导入所需的模块和库
import io  # 导入 io 模块，用于处理文件流操作

import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入以下模块和函数
    NA,  # 导入 NA，表示缺失值
    DataFrame,  # 导入 DataFrame，用于处理和操作数据表
    read_csv,  # 导入 read_csv 函数，用于读取 CSV 文件
)

pytest.importorskip("jinja2")  # 检查并导入 jinja2 模块，如果不存在则跳过

def bar_grad(a=None, b=None, c=None, d=None):
    """多个测试中用于简化预期结果格式化的函数"""
    ret = [("width", "10em")]
    if all(x is None for x in [a, b, c, d]):
        return ret
    return ret + [
        (
            "background",
            f"linear-gradient(90deg,{','.join([x for x in [a, b, c, d] if x])})",
        )
    ]

def no_bar():
    """返回 bar_grad() 的结果，无参数调用"""
    return bar_grad()

def bar_to(x, color="#d65f5f"):
    """返回 bar_grad() 的结果，指定一个参数 x"""
    return bar_grad(f" {color} {x:.1f}%", f" transparent {x:.1f}%")

def bar_from_to(x, y, color="#d65f5f"):
    """返回 bar_grad() 的结果，指定两个参数 x 和 y"""
    return bar_grad(
        f" transparent {x:.1f}%",
        f" {color} {x:.1f}%",
        f" {color} {y:.1f}%",
        f" transparent {y:.1f}%",
    )

@pytest.fixture
def df_pos():
    """返回一个包含正数的 DataFrame"""
    return DataFrame([[1], [2], [3]])

@pytest.fixture
def df_neg():
    """返回一个包含负数的 DataFrame"""
    return DataFrame([[-1], [-2], [-3]])

@pytest.fixture
def df_mix():
    """返回一个包含正负数的混合 DataFrame"""
    return DataFrame([[-3], [1], [2]])

@pytest.mark.parametrize(
    "align, exp",
    [
        ("left", [no_bar(), bar_to(50), bar_to(100)]),
        ("right", [bar_to(100), bar_from_to(50, 100), no_bar()]),
        ("mid", [bar_to(33.33), bar_to(66.66), bar_to(100)]),
        ("zero", [bar_from_to(50, 66.7), bar_from_to(50, 83.3), bar_from_to(50, 100)]),
        ("mean", [bar_to(50), no_bar(), bar_from_to(50, 100)]),
        (2.0, [bar_to(50), no_bar(), bar_from_to(50, 100)]),
        (np.median, [bar_to(50), no_bar(), bar_from_to(50, 100)]),
    ],
)
def test_align_positive_cases(df_pos, align, exp):
    """测试对所有正数情况下的不同对齐方式"""
    result = df_pos.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected

@pytest.mark.parametrize(
    "align, exp",
    [
        ("left", [bar_to(100), bar_to(50), no_bar()]),
        ("right", [no_bar(), bar_from_to(50, 100), bar_to(100)]),
        ("mid", [bar_from_to(66.66, 100), bar_from_to(33.33, 100), bar_to(100)]),
        ("zero", [bar_from_to(33.33, 50), bar_from_to(16.66, 50), bar_to(50)]),
        ("mean", [bar_from_to(50, 100), no_bar(), bar_to(50)]),
        (-2.0, [bar_from_to(50, 100), no_bar(), bar_to(50)]),
        (np.median, [bar_from_to(50, 100), no_bar(), bar_to(50)]),
    ],
)
def test_align_negative_cases(df_neg, align, exp):
    """测试对所有负数情况下的不同对齐方式"""
    result = df_neg.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected
    [
        # 定义 "left" 类型的数据条形图动作序列
        ("left", [no_bar(), bar_to(80), bar_to(100)]),
        # 定义 "right" 类型的数据条形图动作序列
        ("right", [bar_to(100), bar_from_to(80, 100), no_bar()]),
        # 定义 "mid" 类型的数据条形图动作序列
        ("mid", [bar_to(60), bar_from_to(60, 80), bar_from_to(60, 100)]),
        # 定义 "zero" 类型的数据条形图动作序列
        ("zero", [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        # 定义 "mean" 类型的数据条形图动作序列，与 "zero" 类似
        ("mean", [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        # 定义 -0.0（负零）类型的数据条形图动作序列，与 "zero" 和 "mean" 类似
        (-0.0, [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        # 定义 np.nanmedian 函数返回值类型的数据条形图动作序列
        (np.nanmedian, [bar_to(50), no_bar(), bar_from_to(50, 62.5)]),
    ],
@pytest.mark.parametrize("nans", [True, False])
# 使用 pytest 的 parametrize 装饰器，对 nans 参数进行多组测试
def test_align_mixed_cases(df_mix, align, exp, nans):
    # 测试不同的对齐情况，涉及正负混合值
    # 还测试 NaN 和 no_bar 无影响的情况
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    if nans:
        # 如果 nans 为 True，则在 df_mix 的第 3 行插入 NaN
        df_mix.loc[3, :] = np.nan
        # 更新期望结果，增加一个键值对，键为 (3, 0)，值为 no_bar() 的结果
        expected.update({(3, 0): no_bar()})
    result = df_mix.style.bar(align=align)._compute().ctx
    # 断言结果与期望相符
    assert result == expected


@pytest.mark.parametrize(
    "align, exp",
    [
        (
            "left",
            {
                "index": [[no_bar(), no_bar()], [bar_to(100), bar_to(100)]],
                "columns": [[no_bar(), bar_to(100)], [no_bar(), bar_to(100)]],
                "none": [[no_bar(), bar_to(33.33)], [bar_to(66.66), bar_to(100)]],
            },
        ),
        (
            "mid",
            {
                "index": [[bar_to(33.33), bar_to(50)], [bar_to(100), bar_to(100)]],
                "columns": [[bar_to(50), bar_to(100)], [bar_to(75), bar_to(100)]],
                "none": [[bar_to(25), bar_to(50)], [bar_to(75), bar_to(100)]],
            },
        ),
        (
            "zero",
            {
                "index": [
                    [bar_from_to(50, 66.66), bar_from_to(50, 75)],
                    [bar_from_to(50, 100), bar_from_to(50, 100)],
                ],
                "columns": [
                    [bar_from_to(50, 75), bar_from_to(50, 100)],
                    [bar_from_to(50, 87.5), bar_from_to(50, 100)],
                ],
                "none": [
                    [bar_from_to(50, 62.5), bar_from_to(50, 75)],
                    [bar_from_to(50, 87.5), bar_from_to(50, 100)],
                ],
            },
        ),
        (
            2,
            {
                "index": [
                    [bar_to(50), no_bar()],
                    [bar_from_to(50, 100), bar_from_to(50, 100)],
                ],
                "columns": [
                    [bar_to(50), no_bar()],
                    [bar_from_to(50, 75), bar_from_to(50, 100)],
                ],
                "none": [
                    [bar_from_to(25, 50), no_bar()],
                    [bar_from_to(50, 75), bar_from_to(50, 100)],
                ],
            },
        ),
    ],
)
# 使用 pytest 的 parametrize 装饰器，对 align 和 exp 参数进行多组测试
@pytest.mark.parametrize("axis", ["index", "columns", "none"])
# 使用 pytest 的 parametrize 装饰器，对 axis 参数进行多组测试
def test_align_axis(align, exp, axis):
    # 测试所有可能的轴组合，包括正值和不同的对齐方式
    data = DataFrame([[1, 2], [3, 4]])
    result = (
        data.style.bar(align=align, axis=None if axis == "none" else axis)
        ._compute()
        .ctx
    )
    expected = {
        (0, 0): exp[axis][0][0],
        (0, 1): exp[axis][0][1],
        (1, 0): exp[axis][1][0],
        (1, 1): exp[axis][1][1],
    }
    # 断言结果与期望相符
    assert result == expected


@pytest.mark.parametrize(
    "values, vmin, vmax",
    [
        ("positive", 1.5, 2.5),
        ("negative", -2.5, -1.5),
        ("mixed", -2.5, 1.5),
    ],
)
# 使用 pytest 的 parametrize 装饰器，对 values, vmin, vmax 参数进行多组测试
@pytest.mark.parametrize("nullify", [None, "vmin", "vmax"])  # 使用 pytest 的参数化装饰器，测试 vmin 和 vmax 单独的情况
@pytest.mark.parametrize("align", ["left", "right", "zero", "mid"])  # 使用 pytest 的参数化装饰器，测试不同的对齐方式

def test_vmin_vmax_clipping(df_pos, df_neg, df_mix, values, vmin, vmax, nullify, align):
    # 测试当 vmin > 数据值或者 vmax < 数据值时，是否发生截断
    if align == "mid":  # 如果对齐方式为 "mid"，在每种情况下将其作为左侧或右侧处理
        if values == "positive":
            align = "left"
        elif values == "negative":
            align = "right"
    df = {"positive": df_pos, "negative": df_neg, "mixed": df_mix}[values]
    vmin = None if nullify == "vmin" else vmin  # 如果 nullify 是 "vmin"，则将 vmin 设为 None，否则保持原值
    vmax = None if nullify == "vmax" else vmax  # 如果 nullify 是 "vmax"，则将 vmax 设为 None，否则保持原值

    # 根据 vmax 进行数据帧的截断操作，将大于 vmax 的值设为 vmax
    clip_df = df.where(df <= (vmax if vmax else 999), other=vmax)
    # 根据 vmin 进行数据帧的截断操作，将小于 vmin 的值设为 vmin
    clip_df = clip_df.where(clip_df >= (vmin if vmin else -999), other=vmin)

    # 对截断后的数据帧应用样式，包括对齐方式、vmin 和 vmax 的颜色处理
    result = (
        df.style.bar(align=align, vmin=vmin, vmax=vmax, color=["red", "green"])
        ._compute()
        .ctx
    )
    # 期望结果是经过截断处理后的数据帧样式
    expected = clip_df.style.bar(align=align, color=["red", "green"])._compute().ctx
    # 断言结果与期望结果相同
    assert result == expected


@pytest.mark.parametrize(
    "values, vmin, vmax",
    [
        ("positive", 0.5, 4.5),
        ("negative", -4.5, -0.5),
        ("mixed", -4.5, 4.5),
    ],
)
@pytest.mark.parametrize("nullify", [None, "vmin", "vmax"])  # 使用 pytest 的参数化装饰器，测试 vmin 和 vmax 单独的情况
@pytest.mark.parametrize("align", ["left", "right", "zero", "mid"])  # 使用 pytest 的参数化装饰器，测试不同的对齐方式

def test_vmin_vmax_widening(df_pos, df_neg, df_mix, values, vmin, vmax, nullify, align):
    # 测试当 vmax > 数据值或者 vmin < 数据值时，是否发生扩展
    if align == "mid":  # 如果对齐方式为 "mid"，在每种情况下将其作为左侧或右侧处理
        if values == "positive":
            align = "left"
        elif values == "negative":
            align = "right"
    df = {"positive": df_pos, "negative": df_neg, "mixed": df_mix}[values]
    vmin = None if nullify == "vmin" else vmin  # 如果 nullify 是 "vmin"，则将 vmin 设为 None，否则保持原值
    vmax = None if nullify == "vmax" else vmax  # 如果 nullify 是 "vmax"，则将 vmax 设为 None，否则保持原值

    # 复制数据帧以进行扩展
    expand_df = df.copy()
    expand_df.loc[3, :], expand_df.loc[4, :] = vmin, vmax

    # 对扩展后的数据帧应用样式，包括对齐方式、vmin 和 vmax 的颜色处理
    result = (
        df.style.bar(align=align, vmin=vmin, vmax=vmax, color=["red", "green"])
        ._compute()
        .ctx
    )
    # 期望结果是经过扩展处理后的数据帧样式
    expected = expand_df.style.bar(align=align, color=["red", "green"])._compute().ctx
    # 断言结果包含在期望结果中
    assert result.items() <= expected.items()


def test_numerics():
    # 测试数据预先选择了数值
    data = DataFrame([[1, "a"], [2, "b"]])
    # 对数据应用柱状图样式并计算上下文
    result = data.style.bar()._compute().ctx
    # 断言结果中不包含特定的位置值
    assert (0, 1) not in result
    assert (1, 1) not in result


@pytest.mark.parametrize(
    "align, exp",
    [
        ("left", [no_bar(), bar_to(100, "green")]),
        ("right", [bar_to(100, "red"), no_bar()]),
        ("mid", [bar_to(25, "red"), bar_from_to(25, 100, "green")]),
        ("zero", [bar_from_to(33.33, 50, "red"), bar_from_to(50, 100, "green")]),
    ],
)
def test_colors_mixed(align, exp):
    data = DataFrame([[-1], [3]])
    # 对数据应用柱状图样式并计算上下文
    result = data.style.bar(align=align, color=["red", "green"])._compute().ctx
    # 使用断言来验证变量 result 是否等于预期的字典 exp，字典中包含了元组 (0, 0) 和 (1, 0) 分别作为键，并分别对应 exp 的第一个和第二个元素的值。
    assert result == {(0, 0): exp[0], (1, 0): exp[1]}
def test_bar_align_height():
    # 测试当使用关键字 height='no-repeat center' 且存在 'background-size' 时的情况
    # 创建包含两行的 DataFrame
    data = DataFrame([[1], [2]])
    # 对数据进行样式化，设置 align='left' 和 height=50，然后计算样式并获取上下文
    result = data.style.bar(align="left", height=50)._compute().ctx
    # 设置预期的背景样式字符串
    bg_s = "linear-gradient(90deg, #d65f5f 100.0%, transparent 100.0%) no-repeat center"
    # 设置预期的上下文字典
    expected = {
        (0, 0): [("width", "10em")],  # 第一行的样式期望包含宽度设置为 '10em'
        (1, 0): [
            ("width", "10em"),  # 第二行的样式期望包含宽度设置为 '10em'
            ("background", bg_s),  # 设置背景为预期的渐变和居中的样式
            ("background-size", "100% 50.0%"),  # 设置背景大小为 '100% 50.0%'
        ],
    }
    # 断言结果与预期相同
    assert result == expected


def test_bar_value_error_raises():
    # 创建包含一列负值的 DataFrame
    df = DataFrame({"A": [-100, -60, -30, -20]})
    # 设置错误消息
    msg = "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or"
    # 断言使用不支持的 align 参数会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.style.bar(align="poorly", color=["#d65f5f", "#5fba7d"]).to_html()

    # 设置错误消息
    msg = r"`width` must be a value in \[0, 100\]"
    # 断言使用超出范围的 width 参数会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.style.bar(width=200).to_html()

    # 设置错误消息
    msg = r"`height` must be a value in \[0, 100\]"
    # 断言使用超出范围的 height 参数会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.style.bar(height=200).to_html()


def test_bar_color_and_cmap_error_raises():
    # 创建包含一列整数的 DataFrame
    df = DataFrame({"A": [1, 2, 3, 4]})
    # 设置错误消息
    msg = "`color` and `cmap` cannot both be given"
    # 断言同时提供 color 和 cmap 参数会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color="#d65f5f", cmap="viridis").to_html()


def test_bar_invalid_color_type_error_raises():
    # 创建包含一列整数的 DataFrame
    df = DataFrame({"A": [1, 2, 3, 4]})
    # 设置错误消息，说明了 color 参数的有效类型
    msg = (
        r"`color` must be string or list or tuple of 2 strings,"
        r"\(eg: color=\['#d65f5f', '#5fba7d'\]\)"
    )
    # 断言提供无效类型的 color 参数会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=123).to_html()

    # 再次设置错误消息，说明了 color 参数的有效类型
    # 断言提供超过两个元素的颜色列表会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=["#d65f5f", "#5fba7d", "#abcdef"]).to_html()


def test_styler_bar_with_NA_values():
    # 创建包含 NaN 值的 DataFrame
    df1 = DataFrame({"A": [1, 2, NA, 4]})
    # 创建一个全是 NaN 值的 DataFrame
    df2 = DataFrame([[NA, NA], [NA, NA]])
    # 设置预期输出中的关键子串
    expected_substring = "style type="
    # 测试在包含 NaN 值的情况下，对 subset='A' 的样式化
    html_output1 = df1.style.bar(subset="A").to_html()
    # 测试在不指定轴的情况下，对 align='left' 的样式化
    html_output2 = df2.style.bar(align="left", axis=None).to_html()
    # 断言预期子串在输出中存在
    assert expected_substring in html_output1
    assert expected_substring in html_output2


def test_style_bar_with_pyarrow_NA_values():
    # 创建包含 NaN 值的 DataFrame
    data = """name,age,test1,test2,teacher
        Adam,15,95.0,80,Ashby
        Bob,16,81.0,82,Ashby
        Dave,16,89.0,84,Jones
        Fred,15,,88,Jones"""
    # 使用 pyarrow 读取 CSV 数据
    df = read_csv(io.StringIO(data), dtype_backend="pyarrow")
    # 设置预期输出中的关键子串
    expected_substring = "style type="
    # 测试在 subset='test1' 的情况下进行样式化
    html_output = df.style.bar(subset="test1").to_html()
    # 断言预期子串在输出中存在
    assert expected_substring in html_output
```