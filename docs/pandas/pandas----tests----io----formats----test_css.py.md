# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_css.py`

```
import pytest  # 导入 pytest 模块

from pandas.errors import CSSWarning  # 从 pandas.errors 模块导入 CSSWarning 错误类

import pandas._testing as tm  # 导入 pandas._testing 模块并使用别名 tm

from pandas.io.formats.css import CSSResolver  # 从 pandas.io.formats.css 模块导入 CSSResolver 类


def assert_resolves(css, props, inherited=None):
    resolve = CSSResolver()  # 创建 CSS 解析器对象
    actual = resolve(css, inherited=inherited)  # 使用解析器解析给定的 CSS，返回解析后的结果
    assert props == actual  # 断言解析后的结果与预期属性相同


def assert_same_resolution(css1, css2, inherited=None):
    resolve = CSSResolver()  # 创建 CSS 解析器对象
    resolved1 = resolve(css1, inherited=inherited)  # 使用解析器解析第一个 CSS
    resolved2 = resolve(css2, inherited=inherited)  # 使用解析器解析第二个 CSS
    assert resolved1 == resolved2  # 断言两个 CSS 解析结果相同


@pytest.mark.parametrize(
    "name,norm,abnorm",
    [
        (
            "whitespace",
            "hello: world; foo: bar",
            " \t hello \t :\n  world \n  ;  \n foo: \tbar\n\n",
        ),
        ("case", "hello: world; foo: bar", "Hello: WORLD; foO: bar"),
        ("empty-decl", "hello: world; foo: bar", "; hello: world;; foo: bar;\n; ;"),
        ("empty-list", "", ";"),
    ],
)
def test_css_parse_normalisation(name, norm, abnorm):
    assert_same_resolution(norm, abnorm)  # 测试 CSS 解析是否能正确规范化


@pytest.mark.parametrize(
    "invalid_css,remainder,msg",
    [
        # No colon
        ("hello-world", "", "expected a colon"),
        ("border-style: solid; hello-world", "border-style: solid", "expected a colon"),
        (
            "border-style: solid; hello-world; font-weight: bold",
            "border-style: solid; font-weight: bold",
            "expected a colon",
        ),
        # Unclosed string fail
        # Invalid size
        ("font-size: blah", "font-size: 1em", "Unhandled size"),
        ("font-size: 1a2b", "font-size: 1em", "Unhandled size"),
        ("font-size: 1e5pt", "font-size: 1em", "Unhandled size"),
        ("font-size: 1+6pt", "font-size: 1em", "Unhandled size"),
        ("font-size: 1unknownunit", "font-size: 1em", "Unhandled size"),
        ("font-size: 10", "font-size: 1em", "Unhandled size"),
        ("font-size: 10 pt", "font-size: 1em", "Unhandled size"),
        # Too many args
        ("border-top: 1pt solid red green", "border-top: 1pt solid green", "Too many"),
    ],
)
def test_css_parse_invalid(invalid_css, remainder, msg):
    with tm.assert_produces_warning(CSSWarning, match=msg):  # 断言解析无效的 CSS 是否会产生特定警告
        assert_same_resolution(invalid_css, remainder)


@pytest.mark.parametrize(
    "shorthand,expansions",
    # 定义一个包含 CSS 盒模型属性及其对应属性名列表的列表
    [
        # "margin" 属性及其四个方向的属性名列表
        ("margin", ["margin-top", "margin-right", "margin-bottom", "margin-left"]),
        # "padding" 属性及其四个方向的属性名列表
        ("padding", ["padding-top", "padding-right", "padding-bottom", "padding-left"]),
        # "border-width" 属性及其四个方向的属性名列表
        (
            "border-width",
            [
                "border-top-width",
                "border-right-width",
                "border-bottom-width",
                "border-left-width",
            ],
        ),
        # "border-color" 属性及其四个方向的属性名列表
        (
            "border-color",
            [
                "border-top-color",
                "border-right-color",
                "border-bottom-color",
                "border-left-color",
            ],
        ),
        # "border-style" 属性及其四个方向的属性名列表
        (
            "border-style",
            [
                "border-top-style",
                "border-right-style",
                "border-bottom-style",
                "border-left-style",
            ],
        ),
    ],
)

# 测试 CSS 边框简写属性的展开
def test_css_side_shorthands(shorthand, expansions):
    # 将展开后的值解构到对应的变量中
    top, right, bottom, left = expansions

    # 断言展开的结果与预期相符合
    assert_resolves(
        f"{shorthand}: 1pt", {top: "1pt", right: "1pt", bottom: "1pt", left: "1pt"}
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt", {top: "1pt", right: "4pt", bottom: "1pt", left: "4pt"}
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt 2pt",
        {top: "1pt", right: "4pt", bottom: "2pt", left: "4pt"},
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt 2pt 0pt",
        {top: "1pt", right: "4pt", bottom: "2pt", left: "0pt"},
    )

    # 断言对于无法展开的情况，会产生警告
    with tm.assert_produces_warning(CSSWarning, match="Could not expand"):
        assert_resolves(f"{shorthand}: 1pt 1pt 1pt 1pt 1pt", {})


# 使用参数化测试来测试 CSS 边框简写属性的各个边
@pytest.mark.parametrize(
    "shorthand,sides",
    [
        ("border-top", ["top"]),
        ("border-right", ["right"]),
        ("border-bottom", ["bottom"]),
        ("border-left", ["left"]),
        ("border", ["top", "right", "bottom", "left"]),
    ],
)
def test_css_border_shorthand_sides(shorthand, sides):
    # 创建边框属性的字典
    def create_border_dict(sides, color=None, style=None, width=None):
        resolved = {}
        for side in sides:
            if color:
                resolved[f"border-{side}-color"] = color
            if style:
                resolved[f"border-{side}-style"] = style
            if width:
                resolved[f"border-{side}-width"] = width
        return resolved

    # 断言展开结果与预期相符合
    assert_resolves(
        f"{shorthand}: 1pt red solid", create_border_dict(sides, "red", "solid", "1pt")
    )


# 使用参数化测试来测试 CSS 边框简写属性的不同属性组合
@pytest.mark.parametrize(
    "prop, expected",
    [
        ("1pt red solid", ("red", "solid", "1pt")),
        ("red 1pt solid", ("red", "solid", "1pt")),
        ("red solid 1pt", ("red", "solid", "1pt")),
        ("solid 1pt red", ("red", "solid", "1pt")),
        ("red solid", ("red", "solid", "1.500000pt")),
        # 注意：color=black 不符合 CSS 规范
        # (参见 https://drafts.csswg.org/css-backgrounds/#border-shorthands)
        ("1pt solid", ("black", "solid", "1pt")),
        ("1pt red", ("red", "none", "1pt")),
        ("red", ("red", "none", "1.500000pt")),
        ("1pt", ("black", "none", "1pt")),
        ("solid", ("black", "solid", "1.500000pt")),
        # 尺寸
        ("1em", ("black", "none", "12pt")),
    ],
)
def test_css_border_shorthands(prop, expected):
    color, style, width = expected

    # 断言展开结果与预期相符合
    assert_resolves(
        f"border-left: {prop}",
        {
            "border-left-color": color,
            "border-left-style": style,
            "border-left-width": width,
        },
    )
    [
        # 第一个元组：输入 "margin: 1px; margin: 2px" 和空字符串，预期输出 "margin: 2px"
        ("margin: 1px; margin: 2px", "", "margin: 2px"),
        # 第二个元组：输入 "margin: 1px" 和 "margin: 2px"，预期输出 "margin: 1px"
        ("margin: 1px", "margin: 2px", "margin: 1px"),
        # 第三个元组：输入 "margin: 1px; margin: inherit" 和 "margin: 2px"，预期输出 "margin: 2px"
        ("margin: 1px; margin: inherit", "margin: 2px", "margin: 2px"),
        # 第四个元组：输入 "margin: 1px; margin-top: 2px" 和空字符串，
        # 预期输出 "margin-left: 1px; margin-right: 1px; margin-bottom: 1px; margin-top: 2px"
        (
            "margin: 1px; margin-top: 2px",
            "",
            "margin-left: 1px; margin-right: 1px; "
            "margin-bottom: 1px; margin-top: 2px",
        ),
        # 第五个元组：输入 "margin-top: 2px" 和 "margin: 1px"，预期输出 "margin: 1px; margin-top: 2px"
        ("margin-top: 2px", "margin: 1px", "margin: 1px; margin-top: 2px"),
        # 第六个元组：输入 "margin: 1px" 和 "margin-top: 2px"，预期输出 "margin: 1px"
        ("margin: 1px", "margin-top: 2px", "margin: 1px"),
        # 第七个元组：输入 "margin: 1px; margin-top: inherit" 和 "margin: 2px"，
        # 预期输出 "margin: 1px; margin-top: 2px"
        ("margin: 1px; margin-top: inherit", "margin: 2px", "margin: 1px; margin-top: 2px"),
    ],
# 定义测试函数，测试 CSS 属性的优先级解析
def test_css_precedence(style, inherited, equiv):
    # 创建 CSS 解析器实例
    resolve = CSSResolver()
    # 解析继承的 CSS 属性
    inherited_props = resolve(inherited)
    # 解析样式的 CSS 属性，将继承的属性传递给解析器
    style_props = resolve(style, inherited=inherited_props)
    # 解析等效 CSS 属性
    equiv_props = resolve(equiv)
    # 断言样式解析结果等于等效解析结果
    assert style_props == equiv_props


# 使用参数化测试装饰器，测试当样式或等效为空时的 CSS 解析结果
@pytest.mark.parametrize(
    "style,equiv",
    [
        (
            "margin: 1px; margin-top: inherit",
            "margin-bottom: 1px; margin-right: 1px; margin-left: 1px",
        ),
        ("margin-top: inherit", ""),
        ("margin-top: initial", ""),
    ],
)
def test_css_none_absent(style, equiv):
    # 断言相同的解析结果
    assert_same_resolution(style, equiv)


# 使用参数化测试装饰器，测试绝对字体大小解析结果
@pytest.mark.parametrize(
    "size,resolved",
    [
        ("xx-small", "6pt"),
        ("x-small", f"{7.5:f}pt"),
        ("small", f"{9.6:f}pt"),
        ("medium", "12pt"),
        ("large", f"{13.5:f}pt"),
        ("x-large", "18pt"),
        ("xx-large", "24pt"),
        ("8px", "6pt"),
        ("1.25pc", "15pt"),
        (".25in", "18pt"),
        ("02.54cm", "72pt"),
        ("25.4mm", "72pt"),
        ("101.6q", "72pt"),
    ],
)
# 对于绝对字体大小，再次参数化相对值
@pytest.mark.parametrize("relative_to", [None, "16pt"])  # invariant to inherited size
def test_css_absolute_font_size(size, relative_to, resolved):
    # 如果相对值为 None，则继承值也为 None
    if relative_to is None:
        inherited = None
    else:
        # 否则，使用指定的相对值作为继承值
        inherited = {"font-size": relative_to}
    # 断言解析结果与预期结果相等
    assert_resolves(f"font-size: {size}", {"font-size": resolved}, inherited=inherited)


# 使用参数化测试装饰器，测试相对字体大小解析结果
@pytest.mark.parametrize(
    "size,relative_to,resolved",
    [
        ("1em", None, "12pt"),
        ("1.0em", None, "12pt"),
        ("1.25em", None, "15pt"),
        ("1em", "16pt", "16pt"),
        ("1.0em", "16pt", "16pt"),
        ("1.25em", "16pt", "20pt"),
        ("1rem", "16pt", "12pt"),
        ("1.0rem", "16pt", "12pt"),
        ("1.25rem", "16pt", "15pt"),
        ("100%", None, "12pt"),
        ("125%", None, "15pt"),
        ("100%", "16pt", "16pt"),
        ("125%", "16pt", "20pt"),
        ("2ex", None, "12pt"),
        ("2.0ex", None, "12pt"),
        ("2.50ex", None, "15pt"),
        ("inherit", "16pt", "16pt"),
        ("smaller", None, "10pt"),
        ("smaller", "18pt", "15pt"),
        ("larger", None, f"{14.4:f}pt"),
        ("larger", "15pt", "18pt"),
    ],
)
def test_css_relative_font_size(size, relative_to, resolved):
    # 如果相对值为 None，则继承值也为 None
    if relative_to is None:
        inherited = None
    else:
        # 否则，使用指定的相对值作为继承值
        inherited = {"font-size": relative_to}
    # 断言解析结果与预期结果相等
    assert_resolves(f"font-size: {size}", {"font-size": resolved}, inherited=inherited)
```