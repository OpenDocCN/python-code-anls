# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_to_excel.py`

```
# 导入所需的模块和库
import string  # 导入 string 模块
import pytest  # 导入 pytest 模块
from pandas.errors import CSSWarning  # 从 pandas.errors 导入 CSSWarning 异常类
import pandas._testing as tm  # 导入 pandas._testing 模块，并用 tm 别名引用

# 导入 Excel 格式化相关类和函数
from pandas.io.formats.excel import (
    CssExcelCell,  # 导入 CssExcelCell 类
    CSSToExcelConverter,  # 导入 CSSToExcelConverter 类
)


@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "css,expected",  # 参数化测试的参数，分别为 css 和 expected
    ],  # 参数列表未完成，应该在下方的测试用例中提供
)
def test_css_to_excel(css, expected):
    # 创建 CSSToExcelConverter 实例
    convert = CSSToExcelConverter()
    # 断言期望的转换结果与实际转换结果相等
    assert expected == convert(css)


def test_css_to_excel_multiple():
    # 创建 CSSToExcelConverter 实例
    convert = CSSToExcelConverter()
    # 定义多行 CSS 字符串，测试 CSS 转换到 Excel 的多项设置
    actual = convert(
        """
        font-weight: bold;
        text-decoration: underline;
        color: red;
        border-width: thin;
        text-align: center;
        vertical-align: top;
        unused: something;
    """
    )
    # 断言期望的字典格式与实际转换结果相等
    assert {
        "font": {"bold": True, "underline": "single", "color": "FF0000"},
        "border": {
            "top": {"style": "thin"},
            "right": {"style": "thin"},
            "bottom": {"style": "thin"},
            "left": {"style": "thin"},
        },
        "alignment": {"horizontal": "center", "vertical": "top"},
    } == actual


@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "css,inherited,expected",  # 参数化测试的参数，分别为 css, inherited 和 expected
    [  # 参数化测试的参数列表开始
        ("font-weight: bold", "", {"font": {"bold": True}}),  # 测试单个 CSS 属性
        ("", "font-weight: bold", {"font": {"bold": True}}),  # 测试继承的 CSS 属性
        (
            "font-weight: bold",
            "font-style: italic",
            {"font": {"bold": True, "italic": True}},
        ),  # 测试多个 CSS 属性的组合
        ("font-style: normal", "font-style: italic", {"font": {"italic": False}}),  # 测试覆盖继承的 CSS 属性
        ("font-style: inherit", "", {}),  # 测试继承样式的 inherit 属性
        (
            "font-style: normal; font-style: inherit",
            "font-style: italic",
            {"font": {"italic": True}},
        ),  # 测试混合使用 inherit 属性的 CSS 属性
    ],  # 参数化测试的参数列表结束
)
def test_css_to_excel_inherited(css, inherited, expected):
    # 创建 CSSToExcelConverter 实例，并传入继承的 CSS 属性
    convert = CSSToExcelConverter(inherited)
    # 断言期望的字典格式与实际转换结果相等
    assert expected == convert(css)


@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "input_color,output_color",  # 参数化测试的参数，分别为 input_color 和 output_color
    (  # 参数化测试的参数列表开始
        list(CSSToExcelConverter.NAMED_COLORS.items())  # 使用命名颜色字典的颜色
        + [("#" + rgb, rgb) for rgb in CSSToExcelConverter.NAMED_COLORS.values()]  # 使用 RGB 值表示的颜色
        + [("#F0F", "FF00FF"), ("#ABC", "AABBCC")]  # 额外的自定义颜色测试
    ),  # 参数化测试的参数列表结束
)
def test_css_to_excel_good_colors(input_color, output_color):
    # 创建 CSS 字符串，测试各种颜色属性的转换
    css = (
        f"border-top-color: {input_color}; "
        f"border-right-color: {input_color}; "
        f"border-bottom-color: {input_color}; "
        f"border-left-color: {input_color}; "
        f"background-color: {input_color}; "
        f"color: {input_color}"
    )

    expected = {}  # 初始化预期的转换结果字典

    expected["fill"] = {"patternType": "solid", "fgColor": output_color}  # 颜色填充

    expected["font"] = {"color": output_color}  # 字体颜色

    expected["border"] = {  # 边框颜色
        k: {"color": output_color, "style": "none"}  # 设置边框颜色和样式
        for k in ("top", "right", "bottom", "left")  # 遍历边框的方向
    }

    with tm.assert_produces_warning(None):  # 测试中不产生警告
        convert = CSSToExcelConverter()  # 创建 CSSToExcelConverter 实例
        assert expected == convert(css)  # 断言期望的字典格式与实际转换结果相等


@pytest.mark.parametrize("input_color", [None, "not-a-color"])
# 定义测试函数，用于验证 CSS 转换为 Excel 时对于不支持的颜色格式抛出警告
def test_css_to_excel_bad_colors(input_color):
    # 构建包含多个 CSS 样式的字符串，使用给定的输入颜色值
    css = (
        f"border-top-color: {input_color}; "
        f"border-right-color: {input_color}; "
        f"border-bottom-color: {input_color}; "
        f"border-left-color: {input_color}; "
        f"background-color: {input_color}; "
        f"color: {input_color}"
    )

    # 初始化预期结果为空字典
    expected = {}

    # 如果输入颜色不为 None，则设置预期结果的填充样式为实心填充
    if input_color is not None:
        expected["fill"] = {"patternType": "solid"}

    # 断言在 CSS 转换为 Excel 过程中会产生 CSSWarning 警告，警告信息应包含"Unhandled color format"
    with tm.assert_produces_warning(CSSWarning, match="Unhandled color format"):
        # 创建 CSSToExcelConverter 实例
        convert = CSSToExcelConverter()
        # 断言转换后的结果与预期结果相等
        assert expected == convert(css)


# 验证命名颜色在 CSSToExcelConverter 中的有效性
def tests_css_named_colors_valid():
    # 创建大写十六进制字符集合
    upper_hexs = set(map(str.upper, string.hexdigits))
    # 遍历 CSSToExcelConverter 中的命名颜色字典
    for color in CSSToExcelConverter.NAMED_COLORS.values():
        # 断言颜色字符串长度为 6，且所有字符均在大写十六进制字符集合中
        assert len(color) == 6 and all(c in upper_hexs for c in color)


# 验证从 matplotlib.colors 导入的 CSS4 颜色是否存在于 CSSToExcelConverter 中的命名颜色字典中
def test_css_named_colors_from_mpl_present():
    # 导入 matplotlib.colors 模块，如果导入失败则跳过测试
    mpl_colors = pytest.importorskip("matplotlib.colors")

    # 获取 CSSToExcelConverter 中的命名颜色字典
    pd_colors = CSSToExcelConverter.NAMED_COLORS
    # 遍历 matplotlib 中的 CSS4_COLORS 字典
    for name, color in mpl_colors.CSS4_COLORS.items():
        # 断言颜色名称存在于 pd_colors 中，并且其颜色值（去掉前导字符'#'）与 matplotlib 中一致
        assert name in pd_colors and pd_colors[name] == color[1:]


# 参数化测试函数，验证 CSS 样式在转换为 Excel 单元格样式时的优先级行为
@pytest.mark.parametrize(
    "styles,expected",
    [
        ([("color", "green"), ("color", "red")], "color: red;"),
        ([("font-weight", "bold"), ("font-weight", "normal")], "font-weight: normal;"),
        ([("text-align", "center"), ("TEXT-ALIGN", "right")], "text-align: right;"),
    ],
)
def test_css_excel_cell_precedence(styles, expected):
    """It applies favors latter declarations over former declarations"""
    # 查看 GH 47371
    # 创建 CSSToExcelConverter 实例
    converter = CSSToExcelConverter()
    # 清除 _call_cached 缓存
    converter._call_cached.cache_clear()
    # 构建 CSS 样式字典，包含给定的样式
    css_styles = {(0, 0): styles}
    # 创建 CssExcelCell 实例，用于表示 Excel 单元格
    cell = CssExcelCell(
        row=0,
        col=0,
        val="",
        style=None,
        css_styles=css_styles,
        css_row=0,
        css_col=0,
        css_converter=converter,
    )
    # 清除 _call_cached 缓存
    converter._call_cached.cache_clear()

    # 断言单元格样式经过转换后与预期的样式字符串相等
    assert cell.style == converter(expected)


# 参数化测试函数，验证 CSS 样式在转换为 Excel 单元格样式时的缓存机制
@pytest.mark.parametrize(
    "styles,cache_hits,cache_misses",
    [
        ([[("color", "green"), ("color", "red"), ("color", "green")]], 0, 1),
        (
            [
                [("font-weight", "bold")],
                [("font-weight", "normal"), ("font-weight", "bold")],
            ],
            1,
            1,
        ),
        ([[("text-align", "center")], [("TEXT-ALIGN", "center")]], 1, 1),
        (
            [
                [("font-weight", "bold"), ("text-align", "center")],
                [("font-weight", "bold"), ("text-align", "left")],
            ],
            0,
            2,
        ),
        (
            [
                [("font-weight", "bold"), ("text-align", "center")],
                [("font-weight", "bold"), ("text-align", "left")],
                [("font-weight", "bold"), ("text-align", "center")],
            ],
            1,
            2,
        ),
    ],
)
def test_css_excel_cell_cache(styles, cache_hits, cache_misses):
    """It caches unique cell styles"""
    # 根据 GitHub issue 47371 进行的调整
    # 创建一个 CSSToExcelConverter 对象
    converter = CSSToExcelConverter()
    # 清除 _call_cached 方法的缓存，确保最新数据
    converter._call_cached.cache_clear()
    
    # 将 styles 列表中的样式转换为字典，以 (0, i) 为键
    css_styles = {(0, i): _style for i, _style in enumerate(styles)}
    
    # 遍历 css_styles 字典的键，为每个键创建 CssExcelCell 对象并进行初始化
    for css_row, css_col in css_styles:
        CssExcelCell(
            row=0,
            col=0,
            val="",
            style=None,
            css_styles=css_styles,
            css_row=css_row,
            css_col=css_col,
            css_converter=converter,
        )
    
    # 获取 converter._call_cached 缓存的信息
    cache_info = converter._call_cached.cache_info()
    # 再次清除 _call_cached 方法的缓存
    converter._call_cached.cache_clear()
    
    # 断言缓存命中次数与期望的 cache_hits 相等
    assert cache_info.hits == cache_hits
    # 断言缓存未命中次数与期望的 cache_misses 相等
    assert cache_info.misses == cache_misses
```