# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_style.py`

```
# 导入必要的模块和函数
import contextlib  # 提供上下文管理工具的模块
import copy  # 提供对象复制功能的模块
import re  # 提供正则表达式操作的模块
from textwrap import dedent  # 提供文本缩进控制的函数

import numpy as np  # 导入NumPy库，用于科学计算
import pytest  # 导入pytest库，用于编写和运行测试

from pandas import (  # 从pandas库导入多个模块和类
    DataFrame,  # 表示二维数据的数据结构
    IndexSlice,  # 多级索引切片对象
    MultiIndex,  # 多级索引对象
    Series,  # 表示一维数据的数据结构
    option_context,  # 用于设置上下文选项的类
)
import pandas._testing as tm  # pandas的测试工具模块，导入为tm

jinja2 = pytest.importorskip("jinja2")  # 导入jinja2模块，如果不存在则跳过
from pandas.io.formats.style import (  # 导入pandas的样式格式化模块中的Styler类
    Styler,
)
from pandas.io.formats.style_render import (  # 导入pandas的样式渲染模块中的多个函数
    _get_level_lengths,  # 获取级别长度的函数
    _get_trimming_maximums,  # 获取修剪最大值的函数
    maybe_convert_css_to_tuples,  # 可能将CSS转换为元组的函数
    non_reducing_slice,  # 非缩减切片函数
)
    # 列表包含两个元组，每个元组包含一个布尔值和一个列表
    [
        (
            # 第一个元组的布尔值为 True
            True,
            # 第一个元组的列表包含两个字典
            [
                # 第一个字典表示可见，具有合并列属性，值为 "c0"
                {"is_visible": True, "attributes": 'colspan="2"', "value": "c0"},
                # 第二个字典表示不可见，没有属性，值为 "c0"
                {"is_visible": False, "attributes": "", "value": "c0"},
            ],
        ),
        (
            # 第二个元组的布尔值为 False
            False,
            # 第二个元组的列表包含两个字典
            [
                # 第一个字典表示可见，没有属性，值为 "c0"
                {"is_visible": True, "attributes": "", "value": "c0"},
                # 第二个字典表示可见，没有属性，值为 "c0"
                {"is_visible": True, "attributes": "", "value": "c0"},
            ],
        ),
    ],
# 使用 pytest 的装饰器标记这个函数作为一个测试函数
@pytest.mark.parametrize(
    # 参数化测试，指定参数 sparse_index 和 exp_rows
    "sparse_index, exp_rows",
    [
        (
            True,
            [
                # 第一个参数组合，sparse_index 为 True，期望的行数据为两行
                {"is_visible": True, "attributes": 'rowspan="2"', "value": "i0"},
                {"is_visible": False, "attributes": "", "value": "i0"},
            ],
        ),
        (
            False,
            [
                # 第二个参数组合，sparse_index 为 False，期望的行数据为两行，都可见
                {"is_visible": True, "attributes": "", "value": "i0"},
                {"is_visible": True, "attributes": "", "value": "i0"},
            ],
        ),
    ],
)
# 定义名为 test_mi_styler_sparsify_index 的测试函数，参数包括 mi_styler, sparse_index, exp_rows
def test_mi_styler_sparsify_index(mi_styler, sparse_index, exp_rows):
    # 定义期望的第一行、第二行的数据字典
    exp_l1_r0 = {"is_visible": True, "attributes": "", "display_value": "i1_a"}
    exp_l1_r1 = {"is_visible": True, "attributes": "", "display_value": "i1_b"}

    # 调用 mi_styler 的 _translate 方法，传入 sparse_index 和 True，获取上下文 ctx
    ctx = mi_styler._translate(sparse_index, True)

    # 断言检查第一行和第二行的数据是否符合期望
    assert exp_rows[0].items() <= ctx["body"][0][0].items()
    assert exp_rows[1].items() <= ctx["body"][1][0].items()
    # 断言检查第一行和第二行的数据是否符合期望
    assert exp_l1_r0.items() <= ctx["body"][0][1].items()
    assert exp_l1_r1.items() <= ctx["body"][1][1].items()
    ],



# 这是一个列表的结尾，用于结束列表定义。
def test_render_trimming_rows(option, val):
    # 测试自动和特定的行修剪功能
    # 创建一个 60x2 的数据框
    df = DataFrame(np.arange(120).reshape(60, 2))
    # 使用指定的上下文选项进行渲染
    with option_context(option, val):
        # 将数据框样式翻译为上下文对象
        ctx = df.style._translate(True, True)
    # 断言头部包含索引和两列数据
    assert len(ctx["head"][0]) == 3  # index + 2 data cols
    # 断言主体包含四行：三行数据 + 修剪行
    assert len(ctx["body"]) == 4  # 3 data rows + trimming row
    # 断言第一行主体包含索引和两列数据
    assert len(ctx["body"][0]) == 3  # index + 2 data cols


@pytest.mark.parametrize(
    "option, val",
    [
        ("styler.render.max_elements", 6),
        ("styler.render.max_columns", 2),
    ],
)
def test_render_trimming_cols(option, val):
    # 测试自动和特定的列修剪功能
    # 创建一个 3x10 的数据框
    df = DataFrame(np.arange(30).reshape(3, 10))
    # 使用指定的上下文选项进行渲染
    with option_context(option, val):
        # 将数据框样式翻译为上下文对象
        ctx = df.style._translate(True, True)
    # 断言头部包含索引和两列数据以及修剪的列
    assert len(ctx["head"][0]) == 4  # index + 2 data cols + trimming col
    # 断言主体包含三行数据
    assert len(ctx["body"]) == 3  # 3 data rows
    # 断言第一行主体包含索引和两列数据以及修剪的列
    assert len(ctx["body"][0]) == 4  # index + 2 data cols + trimming col


def test_render_trimming_mi():
    # 测试多重索引的修剪功能
    # 创建一个包含多重索引的数据框
    midx = MultiIndex.from_product([[1, 2], [1, 2, 3]])
    df = DataFrame(np.arange(36).reshape(6, 6), columns=midx, index=midx)
    # 使用指定的上下文选项进行渲染
    with option_context("styler.render.max_elements", 4):
        # 将数据框样式翻译为上下文对象
        ctx = df.style._translate(True, True)

    # 断言第一行主体包含索引和两列数据以及修剪的行
    assert len(ctx["body"][0]) == 5  # 2 indexes + 2 data cols + trimming row
    # 断言第一行第一列的属性包含跨行设定为两行
    assert {"attributes": 'rowspan="2"'}.items() <= ctx["body"][0][0].items()
    # 断言第一行第五列的类为数据，行为零，列为修剪
    assert {"class": "data row0 col_trim"}.items() <= ctx["body"][0][4].items()
    # 断言主体包含三行：两行数据 + 修剪的行
    assert len(ctx["body"]) == 3  # 2 data rows + trimming row


def test_render_empty_mi():
    # 测试空的多重索引情况
    # 创建一个带有空多重索引的数据框
    df = DataFrame(index=MultiIndex.from_product([["A"], [0, 1]], names=[None, "one"]))
    # 预期的 HTML 结果
    expected = dedent(
        """\
    >
      <thead>
        <tr>
          <th class="index_name level0" >&nbsp;</th>
          <th class="index_name level1" >one</th>
        </tr>
      </thead>
    """
    )
    # 断言预期的 HTML 结果在数据框样式转换为 HTML 后存在
    assert expected in df.style.to_html()


@pytest.mark.parametrize("comprehensive", [True, False])
@pytest.mark.parametrize("render", [True, False])
@pytest.mark.parametrize("deepcopy", [True, False])
def test_copy(comprehensive, render, deepcopy, mi_styler, mi_styler_comp):
    # 测试复制功能
    # 根据 comprehensive、render 和 deepcopy 参数进行测试
    styler = mi_styler_comp if comprehensive else mi_styler
    # 设置 uuid_len 为 5
    styler.uuid_len = 5

    # 根据 deepcopy 参数选择复制或深复制 styler 对象
    s2 = copy.deepcopy(styler) if deepcopy else copy.copy(styler)  # make copy and check
    # 断言 s2 不等于 styler
    assert s2 is not styler

    # 如果 render 为 True，则进行 HTML 渲染
    if render:
        styler.to_html()

    # 排除一些不需要比较的属性
    excl = [
        "cellstyle_map",  # render time vars..
        "cellstyle_map_columns",
        "cellstyle_map_index",
        "template_latex",  # render templates are class level
        "template_html",
        "template_html_style",
        "template_html_table",
    ]
    # 如果不进行深拷贝（shallow copy），则检查所有包含的属性在内存中的位置是否相同
    if not deepcopy:
        # 遍历 styler 对象的所有属性，排除掉可调用的方法和在排除列表 excl 中的属性
        for attr in [a for a in styler.__dict__ if (not callable(a) and a not in excl)]:
            # 使用 assert 断言，检查 s2 对象和 styler 对象的同名属性在内存中的 id 是否相同
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))
    else:
        # 如果进行深拷贝，则检查特定的属性，这些属性应该具有不同的内存位置（id）
        shallow = [
            "data",
            "columns",
            "index",
            "uuid_len",
            "uuid",
            "caption",
            "cell_ids",
            "hide_index_",
            "hide_columns_",
            "hide_index_names",
            "hide_column_names",
            "table_attributes",
        ]
        # 遍历 shallow 列表中的属性名
        for attr in shallow:
            # 使用 assert 断言，检查 s2 对象和 styler 对象的同名属性在内存中的 id 是否相同
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))

        # 对于不属于 shallow 列表和排除列表 excl 的其它属性，继续检查其内存位置
        for attr in [
            a
            for a in styler.__dict__
            if (not callable(a) and a not in excl and a not in shallow)
        ]:
            # 如果 s2 对象的属性值为 None，则检查其在内存中的 id 是否与 styler 对象相同
            if getattr(s2, attr) is None:
                assert id(getattr(s2, attr)) == id(getattr(styler, attr))
            else:
                # 否则，检查 s2 对象和 styler 对象的同名属性在内存中的 id 是否不同
                assert id(getattr(s2, attr)) != id(getattr(styler, attr)))
@pytest.mark.parametrize("deepcopy", [True, False])
# 使用 pytest 的参数化装饰器，测试函数 test_inherited_copy 分别使用深拷贝和浅拷贝两种方式
def test_inherited_copy(mi_styler, deepcopy):
    # 确保在复制 Styler 对象时，继承类 CustomStyler 被正确保留
    # GH 52728
    class CustomStyler(Styler):
        pass

    custom_styler = CustomStyler(mi_styler.data)
    # 根据 deepcopy 参数选择深拷贝或浅拷贝 CustomStyler 对象
    custom_styler_copy = (
        copy.deepcopy(custom_styler) if deepcopy else copy.copy(custom_styler)
    )
    # 断言复制的对象是 CustomStyler 类的实例
    assert isinstance(custom_styler_copy, CustomStyler)


def test_clear(mi_styler_comp):
    # 注意：如果这个测试因新功能而失败，应更新 'mi_styler_comp'，确保对 'copy'、'clear'、'export' 方法进行正确测试
    # GH 40675
    styler = mi_styler_comp
    styler._compute()  # 执行应用的方法

    # 创建一个与 styler 相同的清理副本
    clean_copy = Styler(styler.data, uuid=styler.uuid)

    # 排除的属性列表，这些属性在比较时不应考虑
    excl = [
        "data",
        "index",
        "columns",
        "uuid",
        "uuid_len",  # uuid 在 styler 和 clean_copy 上设置为相同的值
        "cell_ids",
        "cellstyle_map",  # 仅执行时相关
        "cellstyle_map_columns",  # 仅执行时相关
        "cellstyle_map_index",  # 仅执行时相关
        "template_latex",  # 渲染模板是类级别的
        "template_html",
        "template_html_style",
        "template_html_table",
    ]
    # 在清除之前，检查对象和清理副本中非排除属性的值是否相同
    for attr in [a for a in styler.__dict__ if not (callable(a) or a in excl)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        if hasattr(res, "__iter__") and len(res) > 0:
            assert not all(res)  # 某些元素不同
        elif hasattr(res, "__iter__") and len(res) == 0:
            pass  # 空数组
        else:
            assert not res  # 显式变量不同

    # 在清除之后，检查对象和清理副本中所有属性的值是否相同
    styler.clear()
    for attr in [a for a in styler.__dict__ if not callable(a)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        assert all(res) if hasattr(res, "__iter__") else res


def test_export(mi_styler_comp, mi_styler):
    # 需要导出的属性列表
    exp_attrs = [
        "_todo",
        "hide_index_",
        "hide_index_names",
        "hide_columns_",
        "hide_column_names",
        "table_attributes",
        "table_styles",
        "css",
    ]
    # 检查 mi_styler 和 mi_styler_comp 的属性是否相同
    for attr in exp_attrs:
        check = getattr(mi_styler, attr) == getattr(mi_styler_comp, attr)
        assert not (
            all(check) if (hasattr(check, "__iter__") and len(check) > 0) else check
        )

    # 导出 mi_styler_comp 的结果
    export = mi_styler_comp.export()
    # 使用 export 结果来更新 mi_styler，并检查更新后属性是否相同
    used = mi_styler.use(export)
    for attr in exp_attrs:
        check = getattr(used, attr) == getattr(mi_styler_comp, attr)
        assert all(check) if (hasattr(check, "__iter__") and len(check) > 0) else check

    # 将 used 对象转换为 HTML


def test_hide_raises(mi_styler):
    # 提示信息：'subset' 和 'level' 不能同时传递
    msg = "`subset` and `level` cannot be passed simultaneously"
    # 使用 pytest 模块来测试 mi_styler.hide 方法是否会引发 ValueError 异常，并检查异常消息是否与给定的 msg 变量匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 mi_styler 对象的 hide 方法，预期会抛出 ValueError 异常，
        # 并且异常消息必须与 msg 变量中定义的字符串匹配
        mi_styler.hide(axis="index", subset="something", level="something else")
    
    # 在执行下一段代码之前，定义错误消息字符串，指示了期望的 level 参数类型
    msg = "`level` must be of type `int`, `str` or list of such"
    
    # 使用 pytest 模块来测试 mi_styler.hide 方法是否会引发 ValueError 异常，并检查异常消息是否与给定的 msg 变量匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 mi_styler 对象的 hide 方法，预期会抛出 ValueError 异常，
        # 并且异常消息必须与 msg 变量中定义的字符串匹配
        mi_styler.hide(axis="index", level={"bad": 1, "type": 2})
@pytest.mark.parametrize("level", [1, "one", [1], ["one"]])
# 参数化测试，level 可以是整数、字符串、整数列表或字符串列表
def test_hide_index_level(mi_styler, level):
    # 设置多级索引和列名
    mi_styler.index.names, mi_styler.columns.names = ["zero", "one"], ["zero", "one"]
    # 隐藏指定层级的索引，返回上下文对象 ctx
    ctx = mi_styler.hide(axis="index", level=level)._translate(False, True)
    # 断言头部部分的长度
    assert len(ctx["head"][0]) == 3
    assert len(ctx["head"][1]) == 3
    assert len(ctx["head"][2]) == 4
    # 断言特定单元格是否可见
    assert ctx["head"][2][0]["is_visible"]
    assert not ctx["head"][2][1]["is_visible"]
    # 断言主体部分的特定单元格是否可见
    assert ctx["body"][0][0]["is_visible"]
    assert not ctx["body"][0][1]["is_visible"]
    assert ctx["body"][1][0]["is_visible"]
    assert not ctx["body"][1][1]["is_visible"]


@pytest.mark.parametrize("level", [1, "one", [1], ["one"]])
@pytest.mark.parametrize("names", [True, False])
# 参数化测试，level 和 names 可以是整数、字符串、整数列表或字符串列表
def test_hide_columns_level(mi_styler, level, names):
    # 设置列名
    mi_styler.columns.names = ["zero", "one"]
    if names:
        # 如果 names 为 True，设置索引名
        mi_styler.index.names = ["zero", "one"]
    # 隐藏指定层级的列，返回上下文对象 ctx
    ctx = mi_styler.hide(axis="columns", level=level)._translate(True, False)
    # 断言头部部分的长度
    assert len(ctx["head"]) == (2 if names else 1)


@pytest.mark.parametrize("method", ["map", "apply"])
@pytest.mark.parametrize("axis", ["index", "columns"])
# 参数化测试，method 可以是 "map" 或 "apply"，axis 可以是 "index" 或 "columns"
def test_apply_map_header(method, axis):
    # 创建一个 DataFrame 对象 df
    df = DataFrame({"A": [0, 0], "B": [1, 1]}, index=["C", "D"])
    # 定义一个字典 func，包含 "apply" 和 "map" 方法对应的处理函数
    func = {
        "apply": lambda s: ["attr: val" if ("A" in v or "C" in v) else "" for v in s],
        "map": lambda v: "attr: val" if ("A" in v or "C" in v) else "",
    }

    # 调用指定方法处理索引或列，并返回处理后的结果对象 result
    result = getattr(df.style, f"{method}_index")(func[method], axis=axis)
    # 断言待处理项列表的长度为1
    assert len(result._todo) == 1
    # 断言上下文对象 ctx_index 或 ctx_columns 的长度为0
    assert len(getattr(result, f"ctx_{axis}")) == 0

    # 执行计算
    result._compute()
    # 期望的结果
    expected = {
        (0, 0): [("attr", "val")],
    }
    # 断言上下文对象 ctx_index 或 ctx_columns 的值符合预期
    assert getattr(result, f"ctx_{axis}") == expected


@pytest.mark.parametrize("method", ["apply", "map"])
@pytest.mark.parametrize("axis", ["index", "columns"])
# 参数化测试，method 可以是 "apply" 或 "map"，axis 可以是 "index" 或 "columns"
def test_apply_map_header_mi(mi_styler, method, axis):
    # 定义一个字典 func，包含 "apply" 和 "map" 方法对应的处理函数
    func = {
        "apply": lambda s: ["attr: val;" if "b" in v else "" for v in s],
        "map": lambda v: "attr: val" if "b" in v else "",
    }
    # 调用指定方法处理多级索引或多级列，并执行计算
    result = getattr(mi_styler, f"{method}_index")(func[method], axis=axis)._compute()
    # 期望的结果
    expected = {(1, 1): [("attr", "val")]}
    # 断言上下文对象 ctx_index 或 ctx_columns 的值符合预期
    assert getattr(result, f"ctx_{axis}") == expected


def test_apply_map_header_raises(mi_styler):
    # 测试异常情况，期望抛出 ValueError，并包含特定消息
    with pytest.raises(ValueError, match="No axis named bad for object type DataFrame"):
        # 执行指定的映射操作，并执行计算
        mi_styler.map_index(lambda v: "attr: val;", axis="bad")._compute()


class TestStyler:
    def test_init_non_pandas(self):
        # 测试非 Pandas 对象作为参数时，期望抛出 TypeError 异常，并包含特定消息
        msg = "``data`` must be a Series or DataFrame"
        with pytest.raises(TypeError, match=msg):
            Styler([1, 2, 3])

    def test_init_series(self):
        # 测试初始化为 Series 对象的情况，验证结果对象的维度为2
        result = Styler(Series([1, 2]))
        assert result.data.ndim == 2

    def test_repr_html_ok(self, styler):
        # 测试 Styler 对象的 _repr_html_ 方法，确认没有异常抛出
        styler._repr_html_()
    def test_repr_html_mathjax(self, styler):
        # 检查在生成的 HTML 中是否存在 "tex2jax_ignore"，以验证 MathJax 是否被忽略
        assert "tex2jax_ignore" not in styler._repr_html_()

        # 使用上下文管理器设置 "styler.html.mathjax" 为 False，检查 "tex2jax_ignore" 是否被添加
        with option_context("styler.html.mathjax", False):
            assert "tex2jax_ignore" in styler._repr_html_()

    def test_update_ctx(self, styler):
        # 更新 Styler 的上下文内容，验证预期的格式是否正确
        styler._update_ctx(DataFrame({"A": ["color: red", "color: blue"]}))
        expected = {(0, 0): [("color", "red")], (1, 0): [("color", "blue")]}
        assert styler.ctx == expected

    def test_update_ctx_flatten_multi_and_trailing_semi(self, styler):
        # 更新 Styler 的上下文内容，处理多个样式属性及尾部分号，验证预期的格式是否正确
        attrs = DataFrame({"A": ["color: red; foo: bar", "color:blue ; foo: baz;"]})
        styler._update_ctx(attrs)
        expected = {
            (0, 0): [("color", "red"), ("foo", "bar")],
            (1, 0): [("color", "blue"), ("foo", "baz")],
        }
        assert styler.ctx == expected

    def test_render(self):
        # 创建一个简单的 DataFrame 和样式函数，验证渲染到 HTML 是否成功
        df = DataFrame({"A": [0, 1]})
        style = lambda x: Series(["color: red", "color: blue"], name=x.name)
        s = Styler(df, uuid="AB").apply(style)
        s.to_html()
        # 验证渲染是否成功？

    def test_multiple_render(self, df):
        # GH 39396
        # 对特定列应用样式并渲染两次，以确保 CSS 样式未重复
        s = Styler(df, uuid_len=0).map(lambda x: "color: red;", subset=["A"])
        s.to_html()  # 进行两次渲染以确保 CSS 样式未重复
        assert (
            '<style type="text/css">\n#T__row0_col0, #T__row1_col0 {\n'
            "  color: red;\n}\n</style>" in s.to_html()
        )

    def test_render_empty_dfs(self):
        # 渲染空 DataFrame，验证不同情况下的渲染结果
        empty_df = DataFrame()
        es = Styler(empty_df)
        es.to_html()
        # 只有索引没有列
        DataFrame(columns=["a"]).style.to_html()
        # 只有列没有索引
        DataFrame(index=["a"]).style.to_html()
        # 没有引发 IndexError？

    def test_render_double(self):
        # 创建一个 DataFrame 和样式函数，验证复杂样式的渲染是否成功
        df = DataFrame({"A": [0, 1]})
        style = lambda x: Series(
            ["color: red; border: 1px", "color: blue; border: 2px"], name=x.name
        )
        s = Styler(df, uuid="AB").apply(style)
        s.to_html()
        # 验证渲染是否成功？

    def test_set_properties(self):
        # 设置 DataFrame 的样式属性，并验证计算后的上下文内容是否正确
        df = DataFrame({"A": [0, 1]})
        result = df.style.set_properties(color="white", size="10px")._compute().ctx
        # 顺序是确定的
        v = [("color", "white"), ("size", "10px")]
        expected = {(0, 0): v, (1, 0): v}
        assert result.keys() == expected.keys()
        for v1, v2 in zip(result.values(), expected.values()):
            assert sorted(v1) == sorted(v2)

    def test_set_properties_subset(self):
        # 设置 DataFrame 的部分样式属性，并验证计算后的上下文内容是否正确
        df = DataFrame({"A": [0, 1]})
        result = (
            df.style.set_properties(subset=IndexSlice[0, "A"], color="white")
            ._compute()
            .ctx
        )
        expected = {(0, 0): [("color", "white")]}
        assert result == expected
    def test_empty_index_name_doesnt_display(self, blank_value):
        # 创建一个包含三列数据的DataFrame对象
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # 对DataFrame对象应用样式转换，并获取结果
        result = df.style._translate(True, True)
        # 断言头部信息的长度为1
        assert len(result["head"]) == 1
        # 准备期望的字典，用于断言样式头部信息的内容
        expected = {
            "class": "blank level0",
            "type": "th",
            "value": blank_value,
            "is_visible": True,
            "display_value": blank_value,
        }
        # 断言期望字典的条目是否在样式头部的第一个单元格中
        assert expected.items() <= result["head"][0][0].items()

    def test_index_name(self):
        # 创建一个包含三列数据的DataFrame对象
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # 将"A"列设置为索引，并对DataFrame对象应用样式转换，并获取结果
        result = df.set_index("A").style._translate(True, True)
        # 准备期望的字典，用于断言样式头部信息的内容
        expected = {
            "class": "index_name level0",
            "type": "th",
            "value": "A",
            "is_visible": True,
            "display_value": "A",
        }
        # 断言期望字典的条目是否在样式头部的第二个单元格中
        assert expected.items() <= result["head"][1][0].items()

    def test_numeric_columns(self):
        # 创建一个包含一列数据的DataFrame对象
        df = DataFrame({0: [1, 2, 3]})
        # 对DataFrame对象应用样式转换
        df.style._translate(True, True)

    def test_apply_axis(self):
        # 创建一个包含两列数据的DataFrame对象
        df = DataFrame({"A": [0, 0], "B": [1, 1]})
        # 定义一个lambda函数，用于应用到DataFrame的每行或每列
        f = lambda x: [f"val: {x.max()}" for v in x]
        # 对DataFrame对象应用样式转换，并获取结果
        result = df.style.apply(f, axis=1)
        # 断言待处理项的数量为1
        assert len(result._todo) == 1
        # 断言上下文项的数量为0
        assert len(result.ctx) == 0
        # 计算结果的内容
        result._compute()
        # 准备期望的字典，用于断言计算后的上下文内容
        expected = {
            (0, 0): [("val", "1")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "1")],
            (1, 1): [("val", "1")],
        }
        # 断言计算后的上下文内容是否与期望一致
        assert result.ctx == expected

        # 对DataFrame对象应用样式转换，并获取结果
        result = df.style.apply(f, axis=0)
        # 准备期望的字典，用于断言计算后的上下文内容
        expected = {
            (0, 0): [("val", "0")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "0")],
            (1, 1): [("val", "1")],
        }
        # 计算结果的内容
        result._compute()
        # 断言计算后的上下文内容是否与期望一致
        assert result.ctx == expected
        # 对DataFrame对象应用样式转换（默认axis为0），并获取结果
        result = df.style.apply(f)
        # 计算结果的内容
        result._compute()
        # 断言计算后的上下文内容是否与期望一致
        assert result.ctx == expected

    @pytest.mark.parametrize("axis", [0, 1])
    # 定义一个测试函数，用于测试对 apply 方法返回 Series 的应用
    def test_apply_series_return(self, axis):
        # GH 42014
        # 创建一个 DataFrame 对象，包含两行两列的数据，指定行和列的索引名称
        df = DataFrame([[1, 2], [3, 4]], index=["X", "Y"], columns=["X", "Y"])

        # 测试 apply 方法在返回 Series 时，Series 的长度小于 df 的行或列数，但标签匹配的情况
        func = lambda s: Series(["color: red;"], index=["Y"])
        # 应用样式函数到 DataFrame，并计算样式，获取上下文对象
        result = df.style.apply(func, axis=axis)._compute().ctx
        # 断言结果上下文中特定位置的样式符合预期
        assert result[(1, 1)] == [("color", "red")]
        assert result[(1 - axis, axis)] == [("color", "red")]

        # 测试 apply 方法在返回 Series 时，标签顺序不同但能对应上的情况
        func = lambda s: Series(["color: red;", "color: blue;"], index=["Y", "X"])
        # 应用样式函数到 DataFrame，并计算样式，获取上下文对象
        result = df.style.apply(func, axis=axis)._compute().ctx
        # 断言结果上下文中特定位置的样式符合预期
        assert result[(0, 0)] == [("color", "blue")]
        assert result[(1, 1)] == [("color", "red")]
        assert result[(1 - axis, axis)] == [("color", "red")]
        assert result[(axis, 1 - axis)] == [("color", "blue")]

    # 使用参数化测试装饰器，对 apply 方法返回 DataFrame 的情况进行测试
    @pytest.mark.parametrize("index", [False, True])
    @pytest.mark.parametrize("columns", [False, True])
    def test_apply_dataframe_return(self, index, columns):
        # GH 42014
        # 创建一个 DataFrame 对象，包含两行两列的数据，指定行和列的索引名称
        df = DataFrame([[1, 2], [3, 4]], index=["X", "Y"], columns=["X", "Y"])
        # 根据 index 和 columns 参数设置索引和列名
        idxs = ["X", "Y"] if index else ["Y"]
        cols = ["X", "Y"] if columns else ["Y"]
        # 创建一个包含样式字符串的 DataFrame 对象，指定行和列的索引名称
        df_styles = DataFrame("color: red;", index=idxs, columns=cols)
        # 应用样式函数到 DataFrame，并计算样式，获取上下文对象
        result = df.style.apply(lambda x: df_styles, axis=None)._compute().ctx

        # 断言结果上下文中特定位置的样式符合预期
        assert result[(1, 1)] == [("color", "red")]  # (Y,Y) styles always present
        assert (result[(0, 1)] == [("color", "red")]) is index  # (X,Y) only if index
        assert (result[(1, 0)] == [("color", "red")]) is columns  # (Y,X) only if cols
        assert (result[(0, 0)] == [("color", "red")]) is (index and columns)  # (X,X)

    # 使用参数化测试装饰器，对 apply 方法在子集上的应用进行测试
    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:],
            IndexSlice[:, ["A"]],
            IndexSlice[[1], :],
            IndexSlice[[1], ["A"]],
            IndexSlice[:2, ["A", "B"]],
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_apply_subset(self, slice_, axis, df):
        # 定义一个样式生成函数，根据指定的颜色生成 Series 对象
        def h(x, color="bar"):
            return Series(f"color: {color}", index=x.index, name=x.name)

        # 应用样式函数到 DataFrame 的子集，根据指定的轴和子集进行计算样式，获取上下文对象
        result = df.style.apply(h, axis=axis, subset=slice_, color="baz")._compute().ctx
        # 期望的结果是一个字典，表示预期的样式结果
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        # 断言计算得到的结果与期望的结果相同
        assert result == expected

    # 使用参数化测试装饰器，对 apply 方法在不同切片上的应用进行测试
    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:],
            IndexSlice[:, ["A"]],
            IndexSlice[[1], :],
            IndexSlice[[1], ["A"]],
            IndexSlice[:2, ["A", "B"]],
        ],
    )
    def test_map_subset(self, slice_, df):
        # 调用 Pandas DataFrame 样式对象的 map 方法，应用指定的样式函数并计算结果
        result = df.style.map(lambda x: "color:baz;", subset=slice_)._compute().ctx
        # 生成预期的样式上下文字典，包含应用样式的位置和样式定义
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        # 断言计算结果与预期结果相同
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:, IndexSlice["x", "A"]],
            IndexSlice[:, IndexSlice[:, "A"]],
            IndexSlice[:, IndexSlice[:, ["A", "C"]]],  # 缺失列元素
            IndexSlice[IndexSlice["a", 1], :],
            IndexSlice[IndexSlice[:, 1], :],
            IndexSlice[IndexSlice[:, [1, 3]], :],  # 缺失行元素
            IndexSlice[:, ("x", "A")],
            IndexSlice[("a", 1), :],
        ],
    )
    def test_map_subset_multiindex(self, slice_):
        # GH 19861
        # GH 33562 的修改
        # 根据 slice_ 的不同情况，设置不同的上下文对象 ctx
        if (
            isinstance(slice_[-1], tuple)
            and isinstance(slice_[-1][-1], list)
            and "C" in slice_[-1][-1]
        ):
            ctx = pytest.raises(KeyError, match="C")
        elif (
            isinstance(slice_[0], tuple)
            and isinstance(slice_[0][1], list)
            and 3 in slice_[0][1]
        ):
            ctx = pytest.raises(KeyError, match="3")
        else:
            ctx = contextlib.nullcontext()

        # 创建一个多级索引的 DataFrame df
        idx = MultiIndex.from_product([["a", "b"], [1, 2]])
        col = MultiIndex.from_product([["x", "y"], ["A", "B"]])
        df = DataFrame(np.random.default_rng(2).random((4, 4)), columns=col, index=idx)

        # 使用上下文 ctx 执行下面的代码块
        with ctx:
            # 对 DataFrame 样式应用 map 方法并转换为 HTML
            df.style.map(lambda x: "color: red;", subset=slice_).to_html()

    def test_map_subset_multiindex_code(self):
        # https://github.com/pandas-dev/pandas/issues/25858
        # 检查当提供代码时，styler.map 在多级索引情况下是否正常工作
        # 创建一个具有代码的 MultiIndex
        codes = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        columns = MultiIndex(
            levels=[["a", "b"], ["%", "#"]], codes=codes, names=["", ""]
        )
        df = DataFrame(
            [[1, -1, 1, 1], [-1, 1, 1, 1]], index=["hello", "world"], columns=columns
        )
        pct_subset = IndexSlice[:, IndexSlice[:, "%":"%"]]

        def color_negative_red(val):
            # 根据值的正负返回不同的颜色样式
            color = "red" if val < 0 else "black"
            return f"color: {color}"

        # 选择特定子集并应用颜色样式
        df.loc[pct_subset]
        df.style.map(color_negative_red, subset=pct_subset)

    @pytest.mark.parametrize(
        "stylefunc", ["background_gradient", "bar", "text_gradient"]
    )
    def test_subset_for_boolean_cols(self, stylefunc):
        # GH47838
        # 创建一个包含布尔列的 DataFrame
        df = DataFrame(
            [
                [1, 2],
                [3, 4],
            ],
            columns=[False, True],
        )
        # 根据给定的样式函数名，对 DataFrame 样式对象应用相应的方法
        styled = getattr(df.style, stylefunc)()
        styled._compute()
        # 断言样式对象的上下文中包含预期的位置元组
        assert set(styled.ctx) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    def test_empty(self):
        # 创建包含两列的 DataFrame，一列值为 1 和 0
        df = DataFrame({"A": [1, 0]})
        # 从 DataFrame 创建样式对象
        s = df.style
        # 设置样式上下文，指定第一行第一列的文本颜色为红色，第二行第一列的样式为空
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("", "")]}

        # 调用 _translate 方法，获取单元格样式并取出其中的 "cellstyle" 部分
        result = s._translate(True, True)["cellstyle"]
        # 预期的结果，包含两个字典，每个字典指定单元格的样式和选择器
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0"]},
            {"props": [("", "")], "selectors": ["row1_col0"]},
        ]
        # 断言结果与预期相符
        assert result == expected

    def test_duplicate(self):
        # 创建包含两列的 DataFrame，一列值为 1 和 0
        df = DataFrame({"A": [1, 0]})
        # 从 DataFrame 创建样式对象
        s = df.style
        # 设置样式上下文，指定第一行第一列和第二行第一列的文本颜色均为红色
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("color", "red")]}

        # 调用 _translate 方法，获取单元格样式并取出其中的 "cellstyle" 部分
        result = s._translate(True, True)["cellstyle"]
        # 预期的结果，包含一个字典，指定两个单元格的样式和选择器
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0", "row1_col0"]}
        ]
        # 断言结果与预期相符
        assert result == expected

    def test_init_with_na_rep(self):
        # 创建包含两行两列的 DataFrame，其中包含 NaN 值
        df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

        # 创建 Styler 对象，并调用 _translate 方法获取样式
        ctx = Styler(df, na_rep="NA")._translate(True, True)
        # 断言第一行第一列和第一行第二列的显示值为 "NA"
        assert ctx["body"][0][1]["display_value"] == "NA"
        assert ctx["body"][0][2]["display_value"] == "NA"

    def test_caption(self, df):
        # 创建带有标题的 Styler 对象
        styler = Styler(df, caption="foo")
        # 将 DataFrame 转换为 HTML，并检查结果中是否包含标题 "foo"
        result = styler.to_html()
        assert all(["caption" in result, "foo" in result])

        # 从 DataFrame 创建样式对象
        styler = df.style
        # 设置 Styler 对象的标题为 "baz"
        result = styler.set_caption("baz")
        # 断言 styler 和 result 是同一个对象，并且 styler 的标题为 "baz"
        assert styler is result
        assert styler.caption == "baz"

    def test_uuid(self, df):
        # 创建带有 UUID 的 Styler 对象
        styler = Styler(df, uuid="abc123")
        # 将 DataFrame 转换为 HTML，并检查结果中是否包含 UUID "abc123"
        result = styler.to_html()
        assert "abc123" in result

        # 从 DataFrame 创建样式对象
        styler = df.style
        # 设置 Styler 对象的 UUID 为 "aaa"
        result = styler.set_uuid("aaa")
        # 断言 result 和 styler 是同一个对象，并且 styler 的 UUID 为 "aaa"
        assert result is styler
        assert result.uuid == "aaa"

    def test_unique_id(self):
        # 创建包含两列的 DataFrame
        df = DataFrame({"a": [1, 3, 5, 6], "b": [2, 4, 12, 21]})
        # 生成带有特定 UUID 的 HTML 表格
        result = df.style.to_html(uuid="test")
        # 断言结果中包含 UUID "test"
        assert "test" in result
        # 从结果中提取所有的 id 属性值
        ids = re.findall('id="(.*?)"', result)
        # 断言所有 id 属性值均唯一
        assert np.unique(ids).size == len(ids)

    def test_table_styles(self, df):
        # 定义表格样式，设置 th 元素的 foo 属性为 bar
        style = [{"selector": "th", "props": [("foo", "bar")]}]  # default format
        # 创建带有特定样式的 Styler 对象
        styler = Styler(df, table_styles=style)
        # 将 Styler 对象转换为 HTML，并检查结果中是否包含样式定义
        result = " ".join(styler.to_html().split())
        assert "th { foo: bar; }" in result

        # 从 DataFrame 创建样式对象
        styler = df.style
        # 设置 Styler 对象的表格样式为定义的 style
        result = styler.set_table_styles(style)
        # 断言 styler 和 result 是同一个对象，并且 styler 的表格样式为 style
        assert styler is result
        assert styler.table_styles == style

        # GH 39563
        # 使用字符串格式定义表格样式
        style = [{"selector": "th", "props": "foo:bar;"}]  # css string format
        # 设置 Styler 对象的表格样式为定义的 style
        styler = df.style.set_table_styles(style)
        # 将 Styler 对象转换为 HTML，并检查结果中是否包含样式定义
        result = " ".join(styler.to_html().split())
        assert "th { foo: bar; }" in result
    def test_table_styles_multiple(self, df):
        # 设置表格样式，包括选择器为 "th,td" 的元素设置颜色为红色，选择器为 "tr" 的元素设置颜色为绿色
        ctx = df.style.set_table_styles(
            [
                {"selector": "th,td", "props": "color:red;"},
                {"selector": "tr", "props": "color:green;"},
            ]
        )._translate(True, True)["table_styles"]
        # 断言设置的样式是否与期望一致
        assert ctx == [
            {"selector": "th", "props": [("color", "red")]},
            {"selector": "td", "props": [("color", "red")]},
            {"selector": "tr", "props": [("color", "green")]},
        ]

    def test_table_styles_dict_multiple_selectors(self, df):
        # GH 44011
        # 设置表格样式，选择器为 "th,td" 的元素设置边框左侧为 2px 宽的黑色实线
        result = df.style.set_table_styles(
            {
                "B": [
                    {"selector": "th,td", "props": [("border-left", "2px solid black")]}
                ]
            }
        )._translate(True, True)["table_styles"]

        expected = [
            {"selector": "th.col1", "props": [("border-left", "2px solid black")]},
            {"selector": "td.col1", "props": [("border-left", "2px solid black")]},
        ]

        # 断言设置的样式是否与期望一致
        assert result == expected

    def test_maybe_convert_css_to_tuples(self):
        # 将 CSS 样式字符串转换为元组列表
        expected = [("a", "b"), ("c", "d e")]
        assert maybe_convert_css_to_tuples("a:b;c:d e;") == expected
        assert maybe_convert_css_to_tuples("a: b ;c:  d e  ") == expected
        expected = []
        assert maybe_convert_css_to_tuples("") == expected

    def test_maybe_convert_css_to_tuples_err(self):
        # 测试当传入不合规的 CSS 样式字符串时是否抛出 ValueError 异常
        msg = "Styles supplied as string must follow CSS rule formats"
        with pytest.raises(ValueError, match=msg):
            maybe_convert_css_to_tuples("err")

    def test_table_attributes(self, df):
        # 测试设置表格的 HTML 属性
        attributes = 'class="foo" data-bar'
        styler = Styler(df, table_attributes=attributes)
        result = styler.to_html()
        # 断言生成的 HTML 是否包含指定的属性
        assert 'class="foo" data-bar' in result

        result = df.style.set_table_attributes(attributes).to_html()
        # 断言生成的 HTML 是否包含指定的属性
        assert 'class="foo" data-bar' in result

    def test_apply_none(self):
        # 测试在不指定轴向的情况下应用样式函数
        def f(x):
            return DataFrame(
                np.where(x == x.max(), "color: red", ""),
                index=x.index,
                columns=x.columns,
            )

        result = DataFrame([[1, 2], [3, 4]]).style.apply(f, axis=None)._compute().ctx
        # 断言应用样式函数后的结果是否符合预期
        assert result[(1, 1)] == [("color", "red")]

    def test_trim(self, df):
        # 测试表格样式的 trim 参数
        result = df.style.to_html()  # trim=True
        # 断言生成的 HTML 是否不包含多余的空格（trim=True 的效果）
        assert result.count("#") == 0

        result = df.style.highlight_max().to_html()
        # 断言生成的 HTML 中是否包含与列数相等的高亮标记
        assert result.count("#") == len(df.columns)

    def test_export(self, df, styler):
        # 测试导出样式
        f = lambda x: "color: red" if x > 0 else "color: blue"
        g = lambda x, z: f"color: {z}" if x > 0 else f"color: {z}"
        style1 = styler
        style1.map(f).map(g, z="b").highlight_max()._compute()  # = render
        result = style1.export()
        style2 = df.style
        style2.use(result)
        # 断言两个样式对象的待处理任务列表是否一致
        assert style1._todo == style2._todo
        style2.to_html()
    def test_bad_apply_shape(self):
        # 创建一个 DataFrame，包含两行两列的数据，指定行和列的标签
        df = DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["X", "Y"])

        # 设置错误消息，用于捕获异常，检查 apply 方法是否折叠为 Series
        msg = "resulted in the apply method collapsing to a Series."
        # 断言应该抛出 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 对 DataFrame 应用样式，并触发错误条件
            df.style._apply(lambda x: "x")

        # 设置错误消息模板，检查创建了无效的索引标签
        msg = "created invalid {} labels"
        # 断言应该抛出 ValueError 异常，并匹配预期的错误消息，格式化为 "index"
        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: [""])

        # 同样的错误消息模板，检查创建了无效的索引标签，这次是四个空字符串
        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: ["", "", "", ""])

        # 检查创建了无效的索引标签，尝试在列上应用 Series 对象
        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: Series(["a:v;", ""], index=["A", "C"]), axis=0)

        # 检查创建了无效的列标签，尝试在行上应用列表
        with pytest.raises(ValueError, match=msg.format("columns")):
            df.style._apply(lambda x: ["", "", ""], axis=1)

        # 检查创建了无效的列标签，尝试在行上应用 Series 对象
        with pytest.raises(ValueError, match=msg.format("columns")):
            df.style._apply(lambda x: Series(["a:v;", ""], index=["X", "Z"]), axis=1)

        # 设置错误消息，检查返回的 ndarray 形状不正确
        msg = "returned ndarray with wrong shape"
        # 断言应该抛出 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 对 DataFrame 应用样式，并尝试在未指定轴的情况下应用二维数组
            df.style._apply(lambda x: np.array([[""], [""]]), axis=None)

    def test_apply_bad_return(self):
        # 定义一个简单的函数 f，返回空字符串
        def f(x):
            return ""

        # 创建一个包含两行两列数据的 DataFrame
        df = DataFrame([[1, 2], [3, 4]])
        # 设置错误消息，当传递给 `Styler.apply` 的函数在未指定轴的情况下返回非预期类型时使用
        msg = (
            "must return a DataFrame or ndarray when passed to `Styler.apply` "
            "with axis=None"
        )
        # 断言应该抛出 TypeError 异常，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            # 对 DataFrame 应用样式，并尝试在未指定轴的情况下应用函数 f
            df.style._apply(f, axis=None)

    @pytest.mark.parametrize("axis", ["index", "columns"])
    def test_apply_bad_labels(self, axis):
        # 定义一个函数 f，返回一个带有无效标签的 DataFrame
        def f(x):
            return DataFrame(**{axis: ["bad", "labels"]})

        # 创建一个包含两行两列数据的 DataFrame
        df = DataFrame([[1, 2], [3, 4]])
        # 设置错误消息，用于捕获异常，检查创建了无效的索引或列标签
        msg = f"created invalid {axis} labels."
        # 断言应该抛出 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 对 DataFrame 应用样式，并尝试在未指定轴的情况下应用函数 f
            df.style._apply(f, axis=None)

    def test_get_level_lengths(self):
        # 创建一个 MultiIndex，包含两个级别，每个级别包含两个标签
        index = MultiIndex.from_product([["a", "b"], [0, 1, 2]])
        # 设置预期结果，包含元组键和其对应的长度值
        expected = {
            (0, 0): 3,
            (0, 3): 3,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        # 调用 _get_level_lengths 函数，使用 sparsify=True 和 max_index=100 参数
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        # 断言结果字典等于预期字典
        tm.assert_dict_equal(result, expected)

        # 更新预期结果，使用 sparsify=False 参数
        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 1,
            (0, 5): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        # 再次调用 _get_level_lengths 函数，使用 sparsify=False 和 max_index=100 参数
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        # 断言结果字典等于更新后的预期字典
        tm.assert_dict_equal(result, expected)
    def test_get_level_lengths_un_sorted(self):
        # 创建一个多级索引对象
        index = MultiIndex.from_arrays([[1, 1, 2, 1], ["a", "b", "b", "d"]])
        # 预期的结果字典，表示每个索引位置对应的长度
        expected = {
            (0, 0): 2,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        # 调用 _get_level_lengths 函数，期望结果与预期字典相等
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)

        # 另一组预期结果，不使用稀疏化，仍然与预期字典相等
        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        # 再次调用 _get_level_lengths 函数，预期结果与新的预期字典相等
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_mi_sparse_index_names(self, blank_value):
        # 测试多级索引的名称在渲染时是否正确显示类名和显示值
        df = DataFrame(
            {"A": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        # 获得 DataFrame 样式对象的渲染结果
        result = df.style._translate(True, True)
        # 获取样式结果中的头部信息
        head = result["head"][1]
        # 预期的头部信息列表，包含类名、显示值和可见性信息
        expected = [
            {
                "class": "index_name level0",
                "display_value": "idx_level_0",
                "is_visible": True,
            },
            {
                "class": "index_name level1",
                "display_value": "idx_level_1",
                "is_visible": True,
            },
            {
                "class": "blank col0",
                "display_value": blank_value,
                "is_visible": True,
            },
        ]
        # 验证每个预期字典与结果头部信息中对应项的匹配
        for i, expected_dict in enumerate(expected):
            assert expected_dict.items() <= head[i].items()

    def test_mi_sparse_column_names(self, blank_value):
        # 创建一个包含多级索引和多级列名的 DataFrame 对象
        df = DataFrame(
            np.arange(16).reshape(4, 4),
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "a"], [0, 1, 1, 2]],
                names=["idx_level_0", "idx_level_1"],
            ),
            columns=MultiIndex.from_arrays(
                [["C1", "C1", "C2", "C2"], [1, 0, 1, 0]], names=["colnam_0", "colnam_1"]
            ),
        )
        # 获取 DataFrame 的样式渲染结果
        result = Styler(df, cell_ids=False)._translate(True, True)

        # 对每个层级进行验证
        for level in [0, 1]:
            head = result["head"][level]
            # 预期的头部信息列表，包含类名、显示值和可见性信息
            expected = [
                {
                    "class": "blank",
                    "display_value": blank_value,
                    "is_visible": True,
                },
                {
                    "class": f"index_name level{level}",
                    "display_value": f"colnam_{level}",
                    "is_visible": True,
                },
            ]
            # 验证每个预期字典与结果头部信息中对应项的匹配
            for i, expected_dict in enumerate(expected):
                assert expected_dict.items() <= head[i].items()
    def test_hide_column_headers(self, df, styler):
        # 获取隐藏列头后的样式上下文
        ctx = styler.hide(axis="columns")._translate(True, True)
        # 断言头部条目数量为零，即无未命名索引的列头
        assert len(ctx["head"]) == 0  # no header entries with an unnamed index

        # 设置索引名称为 "some_name"
        df.index.name = "some_name"
        # 获取隐藏列头后的样式上下文
        ctx = df.style.hide(axis="columns")._translate(True, True)
        # 断言头部条目数量为一，即索引名称依然可见，这是 #42101 中的更改，43404 中的回退
        assert len(ctx["head"]) == 1
        # 索引名称依然可见，这是 #42101 中的更改，43404 中的回退

    def test_hide_single_index(self, df):
        # GH 14194
        # 单个未命名索引
        ctx = df.style._translate(True, True)
        # 断言身体部分的第一个条目的可见性为真
        assert ctx["body"][0][0]["is_visible"]
        # 断言头部部分的第一个条目的可见性为真
        assert ctx["head"][0][0]["is_visible"]
        # 获取隐藏索引后的样式上下文
        ctx2 = df.style.hide(axis="index")._translate(True, True)
        # 断言身体部分的第一个条目的可见性为假
        assert not ctx2["body"][0][0]["is_visible"]
        # 断言头部部分的第一个条目的可见性为假
        assert not ctx2["head"][0][0]["is_visible"]

        # 单个命名索引
        ctx3 = df.set_index("A").style._translate(True, True)
        # 断言身体部分的第一个条目的可见性为真
        assert ctx3["body"][0][0]["is_visible"]
        # 断言头部部分有两个级别，即两个头部条目
        assert len(ctx3["head"]) == 2  # 2 header levels
        # 断言头部部分的第一个条目的可见性为真
        assert ctx3["head"][0][0]["is_visible"]

        # 获取隐藏索引后的样式上下文
        ctx4 = df.set_index("A").style.hide(axis="index")._translate(True, True)
        # 断言身体部分的第一个条目的可见性为假
        assert not ctx4["body"][0][0]["is_visible"]
        # 断言头部部分有一个级别，即一个头部条目
        assert len(ctx4["head"]) == 1  # only 1 header level
        # 断言头部部分的第一个条目的可见性为假

    def test_hide_multiindex(self):
        # GH 14194
        df = DataFrame(
            {"A": [1, 2], "B": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        ctx1 = df.style._translate(True, True)
        # 测试 'a' 和 '0'
        assert ctx1["body"][0][0]["is_visible"]
        assert ctx1["body"][0][1]["is_visible"]
        # 检查空白的头部行
        assert len(ctx1["head"][0]) == 4  # two visible indexes and two data columns

        # 获取隐藏索引后的样式上下文
        ctx2 = df.style.hide(axis="index")._translate(True, True)
        # 测试 'a' 和 '0'
        assert not ctx2["body"][0][0]["is_visible"]
        assert not ctx2["body"][0][1]["is_visible"]
        # 检查空白的头部行
        assert len(ctx2["head"][0]) == 3  # one hidden (col name) and two data columns
        # 断言头部部分的第一个条目的可见性为假
        assert not ctx2["head"][0][0]["is_visible"]
    # 定义一个测试方法，用于测试隐藏单个或多个列的功能
    def test_hide_columns_single_level(self, df):
        # GH 14194: GitHub issue number related to this test case
        # test hiding single column: 测试隐藏单列功能
        # 生成样式上下文对象，用于DataFrame样式操作，设置为转换后的格式
        ctx = df.style._translate(True, True)
        # 断言第一行第二列的列是否可见，并且显示值为"A"
        assert ctx["head"][0][1]["is_visible"]
        assert ctx["head"][0][1]["display_value"] == "A"
        # 断言第一行第三列的列是否可见，并且显示值为"B"
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][0][2]["display_value"] == "B"
        # 断言第一行第二列的单元格是否可见（列A，第1行）
        assert ctx["body"][0][1]["is_visible"]
        # 断言第二行第三列的单元格是否可见（列B，第1行）
        assert ctx["body"][1][2]["is_visible"]

        # 隐藏"A"列后生成新的样式上下文对象
        ctx = df.style.hide("A", axis="columns")._translate(True, True)
        # 断言第一行第二列的列不可见（列A）
        assert not ctx["head"][0][1]["is_visible"]
        # 断言第一行第二列的单元格不可见（列A，第1行）
        assert not ctx["body"][0][1]["is_visible"]
        # 断言第二行第三列的单元格仍然可见（列B，第1行）
        assert ctx["body"][1][2]["is_visible"]

        # 隐藏"A"和"B"列后生成新的样式上下文对象
        ctx = df.style.hide(["A", "B"], axis="columns")._translate(True, True)
        # 断言第一行第二列的列不可见（列A）
        assert not ctx["head"][0][1]["is_visible"]
        # 断言第一行第三列的列不可见（列B）
        assert not ctx["head"][0][2]["is_visible"]
        # 断言第一行第二列的单元格不可见（列A，第1行）
        assert not ctx["body"][0][1]["is_visible"]
        # 断言第二行第三列的单元格不可见（列B，第1行）
        assert not ctx["body"][1][2]["is_visible"]
    # 定义测试函数，测试隐藏多级列和索引的功能
    def test_hide_columns_index_mult_levels(self):
        # GH 14194: GitHub issue编号，引用相关问题
        # 设置具有多个列级别和索引的数据框
        i1 = MultiIndex.from_arrays(
            [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
        )
        i2 = MultiIndex.from_arrays(
            [["b", "b"], [0, 1]], names=["col_level_0", "col_level_1"]
        )
        df = DataFrame([[1, 2], [3, 4]], index=i1, columns=i2)
        # 转换数据框样式，并获取上下文
        ctx = df.style._translate(True, True)

        # 检查列标题是否可见
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][1][2]["is_visible"]
        assert ctx["head"][1][3]["display_value"] == "1"

        # 检查索引是否可见
        assert ctx["body"][0][0]["is_visible"]

        # 检查数据是否可见，并验证显示的值
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == "3"
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == "4"

        # 隐藏顶层列级别，即隐藏所有相关列
        ctx = df.style.hide("b", axis="columns")._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][0][0]["is_visible"]  # index

        # 只隐藏第一列
        ctx = df.style.hide([("b", 0)], axis="columns")._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert ctx["head"][0][3]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == "4"

        # 隐藏第二列和索引
        ctx = df.style.hide([("b", 1)], axis=1).hide(axis=0)._translate(True, True)
        assert not ctx["body"][0][0]["is_visible"]  # index
        assert len(ctx["head"][0]) == 3
        assert ctx["head"][0][1]["is_visible"]  # b
        assert ctx["head"][1][1]["is_visible"]  # 0
        assert not ctx["head"][1][2]["is_visible"]  # 1
        assert not ctx["body"][1][3]["is_visible"]  # 4
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == "3"

        # 隐藏顶层行级别，即隐藏所有相关行，因此body为空
        ctx = df.style.hide("a", axis="index")._translate(True, True)
        assert ctx["body"] == []

        # 只隐藏第一行
        ctx = df.style.hide(("a", 0), axis="index")._translate(True, True)
        for i in [0, 1, 2, 3]:
            assert "row1" in ctx["body"][0][i]["class"]  # row0不包含在body中
            assert ctx["body"][0][i]["is_visible"]
    def test_pipe(self, df):
        # 定义一个内部函数，用于设置样式的标题，返回设置好标题的 Styler 对象
        def set_caption_from_template(styler, a, b):
            return styler.set_caption(f"Dataframe with a = {a} and b = {b}")

        # 使用 pipe 方法应用 set_caption_from_template 函数，设置标题为 "Dataframe with a = A and b = B"
        styler = df.style.pipe(set_caption_from_template, "A", b="B")
        
        # 断言确保生成的 HTML 中包含指定的标题内容
        assert "Dataframe with a = A and b = B" in styler.to_html()

        # 测试另一种用法，使用 (callable, keyword_name) 对作为参数的管道调用
        def f(a, b, styler):
            return (a, b, styler)

        # 再次使用 pipe 方法，应用 f 函数，断言返回的结果与预期一致
        styler = df.style
        result = styler.pipe((f, "styler"), a=1, b=2)
        assert result == (1, 2, styler)

    def test_no_cell_ids(self):
        # GH 35588
        # GH 35663
        # 创建一个包含单元格 ID 的 Styler 对象，设置 cell_ids=False
        df = DataFrame(data=[[0]])
        styler = Styler(df, uuid="_", cell_ids=False)
        styler.to_html()  # 渲染两次以确保上下文未更新
        s = styler.to_html()
        # 断言确保生成的 HTML 中存在特定的单元格类名
        assert s.find('<td class="data row0 col0" >') != -1

    @pytest.mark.parametrize(
        "classes",
        [
            # 多个 DataFrame 对象作为参数，分别测试不同的情况
            DataFrame(
                data=[["", "test-class"], [np.nan, None]],
                columns=["A", "B"],
                index=["a", "b"],
            ),
            DataFrame(data=[["test-class"]], columns=["B"], index=["a"]),
            DataFrame(data=[["test-class", "unused"]], columns=["B", "C"], index=["a"]),
        ],
    )
    def test_set_data_classes(self, classes):
        # GH 36159
        # 创建一个包含数据的 DataFrame 对象，用于设置单元格类名
        df = DataFrame(data=[[0, 1], [2, 3]], columns=["A", "B"], index=["a", "b"])
        # 创建 Styler 对象，设置 uuid_len=0 和 cell_ids=False，并应用 set_td_classes 方法设置单元格类名
        s = Styler(df, uuid_len=0, cell_ids=False).set_td_classes(classes).to_html()
        # 断言确保生成的 HTML 中包含指定的单元格内容和类名
        assert '<td class="data row0 col0" >0</td>' in s
        assert '<td class="data row0 col1 test-class" >1</td>' in s
        assert '<td class="data row1 col0" >2</td>' in s
        assert '<td class="data row1 col1" >3</td>' in s
        # GH 39317
        # 使用 cell_ids=True 创建 Styler 对象，并再次应用 set_td_classes 方法设置单元格类名
        s = Styler(df, uuid_len=0, cell_ids=True).set_td_classes(classes).to_html()
        # 断言确保生成的 HTML 中包含指定的单元格内容、ID 和类名
        assert '<td id="T__row0_col0" class="data row0 col0" >0</td>' in s
        assert '<td id="T__row0_col1" class="data row0 col1 test-class" >1</td>' in s
        assert '<td id="T__row1_col0" class="data row1 col0" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >3</td>' in s

    def test_set_data_classes_reindex(self):
        # GH 39317
        # 创建一个包含数据的 DataFrame 对象，用于设置单元格类名和重新索引
        df = DataFrame(
            data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=[0, 1, 2], index=[0, 1, 2]
        )
        # 创建包含类名的 DataFrame 对象，用于设置单元格类名
        classes = DataFrame(
            data=[["mi", "ma"], ["mu", "mo"]],
            columns=[0, 2],
            index=[0, 2],
        )
        # 创建 Styler 对象，设置 uuid_len=0，并应用 set_td_classes 方法设置单元格类名
        s = Styler(df, uuid_len=0).set_td_classes(classes).to_html()
        # 断言确保生成的 HTML 中包含指定的单元格内容、ID 和类名
        assert '<td id="T__row0_col0" class="data row0 col0 mi" >0</td>' in s
        assert '<td id="T__row0_col2" class="data row0 col2 ma" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >4</td>' in s
        assert '<td id="T__row2_col0" class="data row2 col0 mu" >6</td>' in s
        assert '<td id="T__row2_col2" class="data row2 col2 mo" >8</td>' in s
    # 测试表格样式链接（GH 35607）
    def test_chaining_table_styles(self):
        # 创建一个包含数据的DataFrame对象
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        # 创建一个Styler对象，并设置表格样式
        styler = df.style.set_table_styles(
            [{"selector": "", "props": [("background-color", "yellow")]}]
        ).set_table_styles(
            [{"selector": ".col0", "props": [("background-color", "blue")]}],
            overwrite=False,
        )
        # 断言表格样式列表的长度为2
        assert len(styler.table_styles) == 2

    # 测试列和行样式设置（GH 35607）
    def test_column_and_row_styling(self):
        # 创建一个包含数据的DataFrame对象
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        # 创建一个Styler对象，设置表格样式针对列“A”
        s = Styler(df, uuid_len=0)
        s = s.set_table_styles({"A": [{"selector": "", "props": [("color", "blue")]}]})
        # 断言生成的HTML中包含特定的样式定义
        assert "#T_ .col0 {\n  color: blue;\n}" in s.to_html()
        # 进一步设置表格样式针对第一列
        s = s.set_table_styles(
            {0: [{"selector": "", "props": [("color", "blue")]}]}, axis=1
        )
        # 断言生成的HTML中包含特定的样式定义
        assert "#T_ .row0 {\n  color: blue;\n}" in s.to_html()

    # 测试UUID长度（GH 36345）
    @pytest.mark.parametrize("len_", [1, 5, 32, 33, 100])
    def test_uuid_len(self, len_):
        # 创建一个包含单元格数据的DataFrame对象
        df = DataFrame(data=[["A"]])
        # 创建一个Styler对象，设置UUID长度和禁用单元格ID
        s = Styler(df, uuid_len=len_, cell_ids=False).to_html()
        # 查找HTML中UUID的位置并验证其长度
        strt = s.find('id="T_')
        end = s[strt + 6 :].find('"')
        if len_ > 32:
            assert end == 32
        else:
            assert end == len_

    # 测试非降维切片（GH 36345）
    @pytest.mark.parametrize("len_", [-2, "bad", None])
    def test_uuid_len_raises(self, len_):
        # 创建一个包含单元格数据的DataFrame对象
        df = DataFrame(data=[["A"]])
        # 预期的错误消息
        msg = "``uuid_len`` must be an integer in range \\[0, 32\\]."
        # 使用pytest断言捕获类型错误，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            Styler(df, uuid_len=len_, cell_ids=False).to_html()

    # 测试非降维切片（用于DataFrame切片操作）
    @pytest.mark.parametrize(
        "slc",
        [
            IndexSlice[:, :],
            IndexSlice[:, 1],
            IndexSlice[1, :],
            IndexSlice[[1], [1]],
            IndexSlice[1, [1]],
            IndexSlice[[1], 1],
            IndexSlice[1],
            IndexSlice[1, 1],
            slice(None, None, None),
            [0, 1],
            np.array([0, 1]),
            Series([0, 1]),
        ],
    )
    def test_non_reducing_slice(self, slc):
        # 创建一个包含数据的DataFrame对象
        df = DataFrame([[0, 1], [2, 3]])
        # 调用非降维切片函数，并断言返回的结果是DataFrame对象
        tslice_ = non_reducing_slice(slc)
        assert isinstance(df.loc[tslice_], DataFrame)

    # 测试列表切片（类似于DataFrame的索引操作）
    @pytest.mark.parametrize("box", [list, Series, np.array])
    def test_list_slice(self, box):
        # 创建一个包含数据和索引的DataFrame对象
        subset = box(["A"])
        df = DataFrame({"A": [1, 2], "B": [3, 4]}, index=["A", "B"])
        expected = IndexSlice[:, ["A"]]
        # 调用非降维切片函数，并使用tm.assert_frame_equal进行DataFrame比较
        result = non_reducing_slice(subset)
        tm.assert_frame_equal(df.loc[result], df.loc[expected])
    def test_non_reducing_slice_on_multiindex(self):
        # GH 19861
        # 创建一个包含多层索引的字典
        dic = {
            ("a", "d"): [1, 4],
            ("a", "c"): [2, 3],
            ("b", "c"): [3, 2],
            ("b", "d"): [4, 1],
        }
        # 用字典创建一个 DataFrame，并指定索引为 [0, 1]
        df = DataFrame(dic, index=[0, 1])
        # 创建 IndexSlice 对象的别名 idx
        idx = IndexSlice
        # 创建一个多层切片对象 slice_
        slice_ = idx[:, idx["b", "d"]]
        # 调用 non_reducing_slice 函数处理切片对象 slice_
        tslice_ = non_reducing_slice(slice_)

        # 使用处理后的切片 tslice_ 进行 DataFrame 的定位操作
        result = df.loc[tslice_]
        # 创建预期的 DataFrame 结果
        expected = DataFrame({("b", "d"): [4, 1]})
        # 使用 pytest 提供的 assert 方法比较结果是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:, :],
            # 检查列
            IndexSlice[:, IndexSlice[["a"]]],  # 推断更深的需要列表
            IndexSlice[:, IndexSlice[["a"], ["c"]]],  # 推断更深的需要列表
            IndexSlice[:, IndexSlice["a", "c", :]],
            IndexSlice[:, IndexSlice["a", :, "e"]],
            IndexSlice[:, IndexSlice[:, "c", "e"]],
            IndexSlice[:, IndexSlice["a", ["c", "d"], :]],  # 检查列表
            IndexSlice[:, IndexSlice["a", ["c", "d", "-"], :]],  # 不允许缺失
            IndexSlice[:, IndexSlice["a", ["c", "d", "-"], "e"]],  # 没有切片
            # 检查行
            IndexSlice[IndexSlice[["U"]], :],  # 推断更深的需要列表
            IndexSlice[IndexSlice[["U"], ["W"]], :],  # 推断更深的需要列表
            IndexSlice[IndexSlice["U", "W", :], :],
            IndexSlice[IndexSlice["U", :, "Y"], :],
            IndexSlice[IndexSlice[:, "W", "Y"], :],
            IndexSlice[IndexSlice[:, "W", ["Y", "Z"]], :],  # 检查列表
            IndexSlice[IndexSlice[:, "W", ["Y", "Z", "-"]], :],  # 不允许缺失
            IndexSlice[IndexSlice["U", "W", ["Y", "Z", "-"]], :],  # 没有切片
            # 同时检查
            IndexSlice[IndexSlice[:, "W", "Y"], IndexSlice["a", "c", :]],
        ],
    )
    def test_non_reducing_multi_slice_on_multiindex(self, slice_):
        # GH 33562
        # 创建包含多层列索引和行索引的 DataFrame
        cols = MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]])
        idxs = MultiIndex.from_product([["U", "V"], ["W", "X"], ["Y", "Z"]])
        df = DataFrame(np.arange(64).reshape(8, 8), columns=cols, index=idxs)

        # 遍历切片中的每一级索引
        for lvl in [0, 1]:
            key = slice_[lvl]
            if isinstance(key, tuple):
                # 如果是元组，则进一步处理其中的子键
                for subkey in key:
                    if isinstance(subkey, list) and "-" in subkey:
                        # 如果子键是列表且包含 "-"，则引发 KeyError（自 2.0 版本后）
                        with pytest.raises(KeyError, match="-"):
                            df.loc[slice_]
                        return

        # 创建预期的 DataFrame 结果
        expected = df.loc[slice_]
        # 使用 non_reducing_slice 处理切片对象 slice_，并与预期结果比较
        result = df.loc[non_reducing_slice(slice_)]
        # 使用 pytest 提供的 assert 方法比较结果是否相等
        tm.assert_frame_equal(result, expected)
def test_hidden_index_names(mi_df):
    # 设置多重索引的名称为 ["Lev0", "Lev1"]
    mi_df.index.names = ["Lev0", "Lev1"]
    # 获取 DataFrame 样式对象
    mi_styler = mi_df.style
    # 将样式对象转换为上下文对象
    ctx = mi_styler._translate(True, True)
    # 断言头部信息中的长度，验证包括索引名行的数量
    assert len(ctx["head"]) == 3  # 2 column index levels + 1 index names row

    # 隐藏索引名称行
    mi_styler.hide(axis="index", names=True)
    # 再次转换样式对象为上下文对象
    ctx = mi_styler._translate(True, True)
    # 断言头部信息中的长度，验证隐藏了索引名称行后的数量
    assert len(ctx["head"]) == 2  # index names row is unparsed
    # 循环验证主体部分每个元素是否可见
    for i in range(4):
        assert ctx["body"][0][i]["is_visible"]  # 2 index levels + 2 data values visible

    # 再次隐藏第二层级的索引
    mi_styler.hide(axis="index", level=1)
    # 再次转换样式对象为上下文对象
    ctx = mi_styler._translate(True, True)
    # 断言头部信息中的长度，验证第二层级的索引名称行仍然隐藏
    assert len(ctx["head"]) == 2  # index names row is still hidden
    # 断言主体部分的可见性
    assert ctx["body"][0][0]["is_visible"] is True
    assert ctx["body"][0][1]["is_visible"] is False


def test_hidden_column_names(mi_df):
    # 设置多重列索引的名称为 ["Lev0", "Lev1"]
    mi_df.columns.names = ["Lev0", "Lev1"]
    # 获取 DataFrame 样式对象
    mi_styler = mi_df.style
    # 将样式对象转换为上下文对象
    ctx = mi_styler._translate(True, True)
    # 断言头部信息中指定位置的显示值是否正确
    assert ctx["head"][0][1]["display_value"] == "Lev0"
    assert ctx["head"][1][1]["display_value"] == "Lev1"

    # 隐藏列索引名称
    mi_styler.hide(names=True, axis="columns")
    # 再次转换样式对象为上下文对象
    ctx = mi_styler._translate(True, True)
    # 断言头部信息中指定位置的显示值是否变为空格符
    assert ctx["head"][0][1]["display_value"] == "&nbsp;"
    assert ctx["head"][1][1]["display_value"] == "&nbsp;"

    # 再次隐藏第一层级的列索引
    mi_styler.hide(level=0, axis="columns")
    # 再次转换样式对象为上下文对象
    ctx = mi_styler._translate(True, True)
    # 断言头部信息中的长度，验证只有一个可见的列标题
    assert len(ctx["head"]) == 1  # no index names and only one visible column headers
    assert ctx["head"][0][1]["display_value"] == "&nbsp;"


@pytest.mark.parametrize("caption", [1, ("a", "b", "c"), (1, "s")])
def test_caption_raises(mi_styler, caption):
    # 测试设置标题不符合预期时是否抛出 ValueError 异常
    msg = "`caption` must be either a string or 2-tuple of strings."
    with pytest.raises(ValueError, match=msg):
        mi_styler.set_caption(caption)


def test_hiding_headers_over_index_no_sparsify():
    # GH 43464
    # 创建一个包含多重索引的 DataFrame
    midx = MultiIndex.from_product([[1, 2], ["a", "a", "b"]])
    df = DataFrame(9, index=midx, columns=[0])
    # 将 DataFrame 样式对象转换为上下文对象
    ctx = df.style._translate(False, False)
    # 断言主体部分的长度，验证是否包含预期的行数
    assert len(ctx["body"]) == 6
    # 隐藏指定的多重索引值对应的行
    ctx = df.style.hide((1, "a"), axis=0)._translate(False, False)
    # 断言主体部分的长度，验证隐藏后的行数
    assert len(ctx["body"]) == 4
    # 断言特定类名是否在第一行第一列的单元格中
    assert "row2" in ctx["body"][0][0]["class"]


def test_hiding_headers_over_columns_no_sparsify():
    # GH 43464
    # 创建一个包含多重列索引的 DataFrame
    midx = MultiIndex.from_product([[1, 2], ["a", "a", "b"]])
    df = DataFrame(9, columns=midx, index=[0])
    # 将 DataFrame 样式对象转换为上下文对象
    ctx = df.style._translate(False, False)
    # 遍历验证指定位置的单元格是否可见
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx["head"][ix[0]][ix[1]]["is_visible"] is True
    # 隐藏指定的多重列索引值对应的列
    ctx = df.style.hide((1, "a"), axis="columns")._translate(False, False)
    # 遍历验证隐藏后指定位置的单元格是否不可见
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx["head"][ix[0]][ix[1]]["is_visible"] is False


def test_get_level_lengths_mi_hidden():
    # GH 43464
    # 创建一个包含多重索引的 DataFrame
    index = MultiIndex.from_arrays([[1, 1, 1, 2, 2, 2], ["a", "a", "b", "a", "a", "b"]])
    # 预期的索引长度字典
    expected = {
        (0, 2): 1,
        (0, 3): 1,
        (0, 4): 1,
        (0, 5): 1,
        (1, 2): 1,
        (1, 3): 1,
        (1, 4): 1,
        (1, 5): 1,
    }
    # 调用 _get_level_lengths 函数，并传入以下参数：
    # - index: 被传递给 _get_level_lengths 函数的第一个参数
    # - sparsify: 布尔值参数，控制是否稀疏化结果
    # - max_index: 整数参数，限制结果中的最大索引值为100
    # - hidden_elements: 列表参数，包含用于处理重复索引的隐藏元素
    result = _get_level_lengths(
        index,
        sparsify=False,
        max_index=100,
        hidden_elements=[0, 1, 0, 1],  # 如果索引重复，可以重复使用隐藏元素
    )
    
    # 使用测试工具断言函数 _get_level_lengths 返回的结果与期望的结果 expected 相等
    tm.assert_dict_equal(result, expected)
# 定义一个测试函数，用于测试在隐藏行索引时对行进行修剪
def test_row_trimming_hide_index():
    # GitHub issue 43703
    # 创建一个包含5行1列数据框
    df = DataFrame([[1], [2], [3], [4], [5]])
    # 在上下文中设置样式选项，最大渲染行数为2
    with option_context("styler.render.max_rows", 2):
        # 获取修剪行索引后的样式上下文，并将其翻译为可用于断言的格式
        ctx = df.style.hide([0, 1], axis="index")._translate(True, True)
    # 断言修剪后的行数为3
    assert len(ctx["body"]) == 3
    # 断言修剪后的前三行的显示值符合预期
    for r, val in enumerate(["3", "4", "..."]):
        assert ctx["body"][r][1]["display_value"] == val


# 定义一个测试函数，用于测试在多重索引情况下隐藏行索引时的行修剪
def test_row_trimming_hide_index_mi():
    # GitHub issue 44247
    # 创建一个包含5行1列数据框
    df = DataFrame([[1], [2], [3], [4], [5]])
    # 将索引设置为多重索引，以便测试多重索引的行修剪
    df.index = MultiIndex.from_product([[0], [0, 1, 2, 3, 4]])
    # 在上下文中设置样式选项，最大渲染行数为2
    with option_context("styler.render.max_rows", 2):
        # 获取修剪行索引后的样式上下文，并将其翻译为可用于断言的格式
        ctx = df.style.hide([(0, 0), (0, 1)], axis="index")._translate(True, True)
    # 断言修剪后的行数为3
    assert len(ctx["body"]) == 3

    # 断言第一级别索引头部（稀疏化）
    assert {"value": 0, "attributes": 'rowspan="2"', "is_visible": True}.items() <= ctx[
        "body"
    ][0][0].items()
    # 断言第一级别索引的其他行头部被隐藏
    assert {"value": 0, "attributes": "", "is_visible": False}.items() <= ctx["body"][
        1
    ][0].items()
    # 断言第一级别索引修剪后的显示值符合预期
    assert {"value": "...", "is_visible": True}.items() <= ctx["body"][2][0].items()

    # 断言第二级别索引头部修剪后的显示值符合预期
    for r, val in enumerate(["2", "3", "..."]):
        assert ctx["body"][r][1]["display_value"] == val
    # 断言数据值修剪后的显示值符合预期
    for r, val in enumerate(["3", "4", "..."]):
        assert ctx["body"][r][2]["display_value"] == val


# 定义一个测试函数，用于测试在隐藏列时对列进行修剪
def test_col_trimming_hide_columns():
    # GitHub issue 44272
    # 创建一个包含1行5列的数据框
    df = DataFrame([[1, 2, 3, 4, 5]])
    # 在上下文中设置样式选项，最大渲染列数为2
    with option_context("styler.render.max_columns", 2):
        # 获取修剪列后的样式上下文，并将其翻译为可用于断言的格式
        ctx = df.style.hide([0, 1], axis="columns")._translate(True, True)

    # 断言头部的长度为6：空白，[0, 1（隐藏）]，[2, 3（可见）]，+ 修剪列
    assert len(ctx["head"][0]) == 6
    # 断言头部修剪后的显示值和可见性符合预期
    for c, vals in enumerate([(1, False), (2, True), (3, True), ("...", True)]):
        assert ctx["head"][0][c + 2]["value"] == vals[0]
        assert ctx["head"][0][c + 2]["is_visible"] == vals[1]

    # 断言数据主体的长度为6：索引 + 2 隐藏列 + 2 可见列 + 修剪列


# 定义一个测试函数，用于测试应用函数时不产生空结果
def test_no_empty_apply(mi_styler):
    # GitHub issue 45313
    # 应用lambda函数到多重索引样式化器，subset为[False, False]
    mi_styler.apply(lambda s: ["a:v;"] * 2, subset=[False, False])
    # 计算多重索引样式化器
    mi_styler._compute()


# 使用参数化测试来测试输出缓冲区功能
@pytest.mark.parametrize("format", ["html", "latex", "string"])
def test_output_buffer(mi_styler, format):
    # GitHub issue 47053
    # 使用tm.ensure_clean上下文管理器删除格式化后的文件
    with tm.ensure_clean(f"delete_me.{format}") as f:
        # 调用对应格式的to_{format}方法，并将结果写入文件f
        getattr(mi_styler, f"to_{format}")(f)
```