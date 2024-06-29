# `D:\src\scipysrc\pandas\pandas\tests\io\excel\test_style.py`

```
import contextlib
import time
import uuid

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    read_excel,
)
import pandas._testing as tm

from pandas.io.excel import ExcelWriter
from pandas.io.formats.excel import ExcelFormatter

pytest.importorskip("jinja2")
# 当前代码需要确保 jinja2 模块可用，因为 Styler.__init__() 方法依赖于 jinja2。
# Styler.to_excel 方法理论上可以在没有 jinja2 的情况下计算样式并渲染到 Excel 中，因为没有模板文件，
# 但是为了避免在渲染时出现导入错误，因此在此处确认 jinja2 模块的可用性。

@pytest.fixture
def tmp_excel(tmp_path):
    # 创建临时的 Excel 文件路径，并返回其字符串表示形式作为 fixture 的值
    tmp = tmp_path / f"{uuid.uuid4()}.xlsx"
    tmp.touch()
    return str(tmp)


def assert_equal_cell_styles(cell1, cell2):
    # 检查两个单元格样式是否相等的断言方法
    # TODO: 应该寻找更好的方法来检查样式的相等性

def test_styler_default_values(tmp_excel):
    # 测试 Styler 默认值的情况，主要关注 GH 54154
    openpyxl = pytest.importorskip("openpyxl")
    df = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 1, "B": 2, "C": 3}])

    with ExcelWriter(tmp_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="custom")

    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        # 检查字体、间距、缩进
        assert wb["custom"].cell(1, 1).font.bold is False
        assert wb["custom"].cell(1, 1).alignment.horizontal is None
        assert wb["custom"].cell(1, 1).alignment.vertical is None

        # 检查边框
        assert wb["custom"].cell(1, 1).border.bottom.color is None
        assert wb["custom"].cell(1, 1).border.top.color is None
        assert wb["custom"].cell(1, 1).border.left.color is None
        assert wb["custom"].cell(1, 1).border.right.color is None

@pytest.mark.parametrize("engine", ["xlsxwriter", "openpyxl"])
def test_styler_to_excel_unstyled(engine, tmp_excel):
    # 比较 DataFrame.to_excel 和 Styler.to_excel 在未应用样式时的行为
    pytest.importorskip(engine)
    df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
    with ExcelWriter(tmp_excel, engine=engine) as writer:
        df.to_excel(writer, sheet_name="dataframe")
        df.style.to_excel(writer, sheet_name="unstyled")

    openpyxl = pytest.importorskip("openpyxl")  # 仅使用 openpyxl 进行加载的测试
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        for col1, col2 in zip(wb["dataframe"].columns, wb["unstyled"].columns):
            assert len(col1) == len(col2)
            for cell1, cell2 in zip(col1, col2):
                assert cell1.value == cell2.value
                assert_equal_cell_styles(cell1, cell2)

shared_style_params = [
    (
        "background-color: #111222",  # 设置背景颜色为 #111222
        ["fill", "fgColor", "rgb"],   # 在Excel中，设置填充颜色的RGB值
        {"xlsxwriter": "FF111222", "openpyxl": "00111222"},  # 不同库的颜色表示方式对应的RGB值
    ),
    (
        "color: #111222",  # 设置文本颜色为 #111222
        ["font", "color", "value"],   # 在Excel中，设置字体颜色的值
        {"xlsxwriter": "FF111222", "openpyxl": "00111222"},  # 不同库的颜色表示方式对应的RGB值
    ),
    ("font-family: Arial;", ["font", "name"], "arial"),  # 设置字体为 Arial
    ("font-weight: bold;", ["font", "b"], True),  # 设置字体为粗体
    ("font-style: italic;", ["font", "i"], True),  # 设置字体为斜体
    ("text-decoration: underline;", ["font", "u"], "single"),  # 设置文本下划线样式为单线
    ("number-format: $??,???.00;", ["number_format"], "$??,???.00"),  # 设置数字格式为货币格式
    ("text-align: left;", ["alignment", "horizontal"], "left"),  # 设置文本水平对齐方式为左对齐
    (
        "vertical-align: bottom;",  # 设置文本垂直对齐方式为底部
        ["alignment", "vertical"],  # 在Excel中，设置垂直对齐方式
        {"xlsxwriter": None, "openpyxl": "bottom"},  # 不同库的垂直对齐方式的映射
    ),
    ("vertical-align: middle;", ["alignment", "vertical"], "center"),  # 设置文本垂直对齐方式为中间
    # Border widths（边框宽度）
    ("border-left: 2pt solid red", ["border", "left", "style"], "medium"),  # 设置左边框样式为中等粗细
    ("border-left: 1pt dotted red", ["border", "left", "style"], "dotted"),  # 设置左边框样式为点线
    ("border-left: 2pt dotted red", ["border", "left", "style"], "mediumDashDotDot"),  # 设置左边框样式为中点划点
    ("border-left: 1pt dashed red", ["border", "left", "style"], "dashed"),  # 设置左边框样式为虚线
    ("border-left: 2pt dashed red", ["border", "left", "style"], "mediumDashed"),  # 设置左边框样式为中虚线
    ("border-left: 1pt solid red", ["border", "left", "style"], "thin"),  # 设置左边框样式为细线
    ("border-left: 3pt solid red", ["border", "left", "style"], "thick"),  # 设置左边框样式为粗线
    # Border expansion（边框颜色）
    (
        "border-left: 2pt solid #111222",  # 设置左边框颜色为 #111222
        ["border", "left", "color", "rgb"],  # 在Excel中，设置边框颜色的RGB值
        {"xlsxwriter": "FF111222", "openpyxl": "00111222"},  # 不同库的颜色表示方式对应的RGB值
    ),
    ("border: 1pt solid red", ["border", "top", "style"], "thin"),  # 设置顶部边框样式为细线
    (
        "border: 1pt solid #111222",  # 设置顶部边框颜色为 #111222
        ["border", "top", "color", "rgb"],  # 在Excel中，设置边框颜色的RGB值
        {"xlsxwriter": "FF111222", "openpyxl": "00111222"},  # 不同库的颜色表示方式对应的RGB值
    ),
    ("border: 1pt solid red", ["border", "right", "style"], "thin"),  # 设置右边框样式为细线
    (
        "border: 1pt solid #111222",  # 设置右边框颜色为 #111222
        ["border", "right", "color", "rgb"],  # 在Excel中，设置边框颜色的RGB值
        {"xlsxwriter": "FF111222", "openpyxl": "00111222"},  # 不同库的颜色表示方式对应的RGB值
    ),
    ("border: 1pt solid red", ["border", "bottom", "style"], "thin"),  # 设置底部边框样式为细线
    (
        "border: 1pt solid #111222",  # 设置底部边框颜色为 #111222
        ["border", "bottom", "color", "rgb"],  # 在Excel中，设置边框颜色的RGB值
        {"xlsxwriter": "FF111222", "openpyxl": "00111222"},  # 不同库的颜色表示方式对应的RGB值
    ),
    ("border: 1pt solid red", ["border", "left", "style"], "thin"),  # 设置左边框样式为细线
    (
        "border: 1pt solid #111222",  # 设置左边框颜色为 #111222
        ["border", "left", "color", "rgb"],  # 在Excel中，设置边框颜色的RGB值
        {"xlsxwriter": "FF111222", "openpyxl": "00111222"},  # 不同库的颜色表示方式对应的RGB值
    ),
    # Border styles（边框样式）
    (
        "border-left-style: hair; border-left-color: black",  # 设置左边框样式为发丝线，颜色为黑色
        ["border", "left", "style"],  # 在Excel中，设置边框样式
        "hair",  # 设置边框样式为发丝线
    ),
# 定义一个测试函数，用于测试自定义样式在 Excel 中的输出
def test_styler_custom_style(tmp_excel):
    # 定义 CSS 样式
    css_style = "background-color: #111222"
    # 导入 openpyxl 库，如果库不存在则跳过该测试
    openpyxl = pytest.importorskip("openpyxl")
    # 创建一个包含两行数据的 DataFrame 对象
    df = DataFrame([{"A": 1, "B": 2}, {"A": 1, "B": 2}])

    # 使用 ExcelWriter 写入 Excel 文件，引擎为 openpyxl
    with ExcelWriter(tmp_excel, engine="openpyxl") as writer:
        # 应用样式到 DataFrame
        styler = df.style.map(lambda x: css_style)
        # 将带样式的 DataFrame 写入 Excel 的自定义工作表
        styler.to_excel(writer, sheet_name="custom", index=False)

    # 使用 contextlib.closing 确保在退出代码块时关闭 workbook
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        # 检查字体、行间距和缩进
        assert wb["custom"].cell(1, 1).font.bold is False
        assert wb["custom"].cell(1, 1).alignment.horizontal is None
        assert wb["custom"].cell(1, 1).alignment.vertical is None

        # 检查边框
        assert wb["custom"].cell(1, 1).border.bottom.color is None
        assert wb["custom"].cell(1, 1).border.top.color is None
        assert wb["custom"].cell(1, 1).border.left.color is None
        assert wb["custom"].cell(1, 1).border.right.color is None

        # 检查背景颜色
        assert wb["custom"].cell(2, 1).fill.fgColor.index == "00111222"
        assert wb["custom"].cell(3, 1).fill.fgColor.index == "00111222"
        assert wb["custom"].cell(2, 2).fill.fgColor.index == "00111222"
        assert wb["custom"].cell(3, 2).fill.fgColor.index == "00111222"


# 使用 xlsxwriter 和 openpyxl 引擎参数化测试
@pytest.mark.parametrize("engine", ["xlsxwriter", "openpyxl"])
# 使用 shared_style_params 参数化测试的 CSS 样式、属性和期望结果
@pytest.mark.parametrize("css, attrs, expected", shared_style_params)
# 测试基本的样式转换到 Excel
def test_styler_to_excel_basic(engine, css, attrs, expected, tmp_excel):
    # 如果引擎不可用，则跳过该测试
    pytest.importorskip(engine)
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 1)))
    # 创建样式对象并应用到 DataFrame
    styler = df.style.map(lambda x: css)

    # 使用 ExcelWriter 写入 Excel 文件，指定引擎
    with ExcelWriter(tmp_excel, engine=engine) as writer:
        # 将 DataFrame 写入未经过样式处理的数据表单
        df.to_excel(writer, sheet_name="dataframe")
        # 将带样式的 DataFrame 写入样式处理后的工作表单
        styler.to_excel(writer, sheet_name="styled")

    # 导入 openpyxl 库用于加载 workbook
    openpyxl = pytest.importorskip("openpyxl")
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        # 检查未经样式处理的数据单元格是否不含预期样式
        # 检查经过样式处理的单元格是否含有预期样式
        u_cell, s_cell = wb["dataframe"].cell(2, 2), wb["styled"].cell(2, 2)
        for attr in attrs:
            u_cell, s_cell = getattr(u_cell, attr, None), getattr(s_cell, attr)

        # 如果期望结果是字典类型，则验证结果是否符合期望
        if isinstance(expected, dict):
            assert u_cell is None or u_cell != expected[engine]
            assert s_cell == expected[engine]
        else:
            assert u_cell is None or u_cell != expected
            assert s_cell == expected


# 使用 xlsxwriter 和 openpyxl 引擎参数化测试
@pytest.mark.parametrize("engine", ["xlsxwriter", "openpyxl"])
# 使用 shared_style_params 参数化测试的 CSS 样式、属性和期望结果
@pytest.mark.parametrize("css, attrs, expected", shared_style_params)
# 测试样式应用到索引后转换到 Excel
def test_styler_to_excel_basic_indexes(engine, css, attrs, expected, tmp_excel):
    # 如果引擎不可用，则跳过该测试
    pytest.importorskip(engine)
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 1)))

    # 创建样式对象并应用到 DataFrame 的索引
    styler = df.style
    styler.map_index(lambda x: css, axis=0)
    styler.map_index(lambda x: css, axis=1)

    # 创建未应用样式的样式对象
    null_styler = df.style
    null_styler.map(lambda x: "null: css;")
    # 对 null_styler 对象应用 map_index 方法，为每一列添加样式 "null: css;"
    null_styler.map_index(lambda x: "null: css;", axis=0)
    # 对 null_styler 对象应用 map_index 方法，为每一行添加样式 "null: css;"
    null_styler.map_index(lambda x: "null: css;", axis=1)

    # 使用 ExcelWriter 创建一个 Excel 写入对象，将数据写入临时 Excel 文件
    with ExcelWriter(tmp_excel, engine=engine) as writer:
        # 将 null_styler 中的数据写入名为 "null_styled" 的工作表
        null_styler.to_excel(writer, sheet_name="null_styled")
        # 将 styler 中的数据写入名为 "styled" 的工作表
        styler.to_excel(writer, sheet_name="styled")

    # 导入 openpyxl 模块，如果导入失败则跳过测试
    openpyxl = pytest.importorskip("openpyxl")  # test loading only with openpyxl
    # 使用 openpyxl 加载临时 Excel 文件，并使用上下文管理确保关闭工作簿
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        # 获取 "null_styled" 工作表中第二行第一列和 "styled" 工作表中第二行第一列的单元格对象
        ui_cell, si_cell = wb["null_styled"].cell(2, 1), wb["styled"].cell(2, 1)
        # 获取 "null_styled" 工作表中第一行第二列和 "styled" 工作表中第一行第二列的单元格对象
        uc_cell, sc_cell = wb["null_styled"].cell(1, 2), wb["styled"].cell(1, 2)

    # 遍历 attrs 列表中的属性，并分别获取单元格对象的属性值进行比较
    for attr in attrs:
        ui_cell, si_cell = getattr(ui_cell, attr, None), getattr(si_cell, attr)
        uc_cell, sc_cell = getattr(uc_cell, attr, None), getattr(sc_cell, attr)

    # 检查 expected 是否为字典类型，然后分别断言各个单元格的属性值是否符合预期
    if isinstance(expected, dict):
        assert ui_cell is None or ui_cell != expected[engine]
        assert si_cell == expected[engine]
        assert uc_cell is None or uc_cell != expected[engine]
        assert sc_cell == expected[engine]
    else:
        assert ui_cell is None or ui_cell != expected
        assert si_cell == expected
        assert uc_cell is None or uc_cell != expected
        assert sc_cell == expected
# 定义 Excel 边框样式列表，参考 https://openpyxl.readthedocs.io/en/stable/api/openpyxl.styles.borders.html
# 注意："width" 类型的样式行为未定义；用户应该使用 border-width 替代
excel_border_styles = [
    # 可用的边框样式
    "dashed",             # 虚线
    "mediumDashDot",      # 中等点划线
    "dashDotDot",         # 点-点-点线
    "hair",               # 非常细的线
    "dotted",             # 点线
    "mediumDashDotDot",   # 中等点-点-点线
    "double",             # 双线
    "dashDot",            # 点划线
    "slantDashDot",       # 斜线-点划线
    "mediumDashed",       # 中等虚线
]

# 测试函数，参数化引擎和边框样式
@pytest.mark.parametrize("engine", ["xlsxwriter", "openpyxl"])
@pytest.mark.parametrize("border_style", excel_border_styles)
def test_styler_to_excel_border_style(engine, border_style, tmp_excel):
    # 设置 CSS 样式字符串
    css = f"border-left: {border_style} black thin"
    # 预期的样式值
    expected = border_style
    # 导入指定的 Excel 引擎库，若引发错误则跳过
    pytest.importorskip(engine)
    # 创建一个随机数据帧
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 1)))
    # 创建样式对象，并应用 CSS 样式
    styler = df.style.map(lambda x: css)

    # 使用 ExcelWriter 写入数据帧和样式到临时 Excel 文件
    with ExcelWriter(tmp_excel, engine=engine) as writer:
        df.to_excel(writer, sheet_name="dataframe")
        styler.to_excel(writer, sheet_name="styled")

    # 导入 openpyxl 库，仅测试加载 openpyxl 引擎
    openpyxl = pytest.importorskip("openpyxl")
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        # 获取数据帧和样式写入后的单元格对象
        u_cell, s_cell = wb["dataframe"].cell(2, 2), wb["styled"].cell(2, 2)
    
    # 检查每个属性是否符合预期样式
    for attr in ["border", "left", "style"]:
        u_cell, s_cell = getattr(u_cell, attr, None), getattr(s_cell, attr)

    # 如果期望是字典，则检查特定引擎的预期样式值
    if isinstance(expected, dict):
        assert u_cell is None or u_cell != expected[engine]
        assert s_cell == expected[engine]
    else:
        assert u_cell is None or u_cell != expected
        assert s_cell == expected


# 测试自定义转换器功能
def test_styler_custom_converter(tmp_excel):
    openpyxl = pytest.importorskip("openpyxl")

    # 定义一个自定义的 CSS 转换器函数
    def custom_converter(css):
        return {"font": {"color": {"rgb": "111222"}}}

    # 创建一个随机数据帧
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 1)))
    # 创建样式对象，并应用 CSS 样式
    styler = df.style.map(lambda x: "color: #888999")
    
    # 使用 ExcelWriter 写入数据帧和样式到临时 Excel 文件，使用 openpyxl 引擎
    with ExcelWriter(tmp_excel, engine="openpyxl") as writer:
        # 使用自定义转换器进行写入
        ExcelFormatter(styler, style_converter=custom_converter).write(
            writer, sheet_name="custom"
        )

    # 使用 openpyxl 加载临时 Excel 文件
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        # 检查自定义样式转换器是否正确应用
        assert wb["custom"].cell(2, 2).font.color.value == "00111222"


# 测试将样式写入到 S3 存储桶
@pytest.mark.single_cpu
@td.skip_if_not_us_locale
def test_styler_to_s3(s3_public_bucket, s3so):
    # GH#46381

    # 模拟 S3 存储桶的名称和目标文件名
    mock_bucket_name, target_file = s3_public_bucket.name, "test.xlsx"
    # 创建一个带有样式的数据帧
    df = DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    styler = df.style.set_sticky(axis="index")
    # 将带样式的数据帧写入到指定的 S3 存储路径
    styler.to_excel(f"s3://{mock_bucket_name}/{target_file}", storage_options=s3so)
    # 设置超时时间
    timeout = 5
    # 循环，持续检查目标文件是否存在于 S3 公共存储桶中的对象列表中
    while True:
        # 检查目标文件是否存在于 S3 公共存储桶的对象键中
        if target_file in (obj.key for obj in s3_public_bucket.objects.all()):
            break  # 如果存在，则跳出循环
        # 如果文件未找到，等待0.1秒再重新检查
        time.sleep(0.1)
        # 减少超时计数器
        timeout -= 0.1
        # 断言超时计数器仍大于0，否则抛出超时异常
        assert timeout > 0, "Timed out waiting for file to appear on moto"
        # 使用指定的函数读取 Excel 文件，从模拟的 S3 路径中读取，配置了索引列为0和存储选项
        result = read_excel(
            f"s3://{mock_bucket_name}/{target_file}", index_col=0, storage_options=s3so
        )
        # 使用测试工具断言读取的 Excel 数据框架与预期的数据框架相等
        tm.assert_frame_equal(result, df)
```