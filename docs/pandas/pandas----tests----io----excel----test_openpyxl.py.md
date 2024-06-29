# `D:\src\scipysrc\pandas\pandas\tests\io\excel\test_openpyxl.py`

```
# 导入必要的模块和库
import contextlib  # 上下文管理模块，用于创建和管理上下文对象的工具
from pathlib import Path  # 处理路径的模块
import re  # 正则表达式模块，用于字符串匹配和操作
import uuid  # 生成唯一标识符的模块

import numpy as np  # 数值计算库
import pytest  # Python 的单元测试框架

import pandas as pd  # 数据分析库
from pandas import DataFrame  # Pandas 的 DataFrame 类
import pandas._testing as tm  # Pandas 测试工具

from pandas.io.excel import (  # Pandas 中处理 Excel 文件的模块
    ExcelWriter,  # Excel 写入器
    _OpenpyxlWriter,  # Pandas 内部使用的 Openpyxl 写入器
)
from pandas.io.excel._openpyxl import OpenpyxlReader  # Pandas 使用的 Openpyxl 读取器

# 确保 openpyxl 库存在，否则跳过测试
openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture
def ext():
    return ".xlsx"


@pytest.fixture
def tmp_excel(ext, tmp_path):
    # 创建临时 Excel 文件路径
    tmp = tmp_path / f"{uuid.uuid4()}{ext}"
    tmp.touch()  # 创建文件
    return str(tmp)


def test_to_excel_styleconverter():
    from openpyxl import styles  # 导入 openpyxl 中的样式模块

    # 定义一个样式字典 hstyle
    hstyle = {
        "font": {"color": "00FF0000", "bold": True},  # 字体设置为红色，加粗
        "borders": {"top": "thin", "right": "thin", "bottom": "thin", "left": "thin"},  # 设置边框
        "alignment": {"horizontal": "center", "vertical": "top"},  # 设置对齐方式
        "fill": {"patternType": "solid", "fgColor": {"rgb": "006666FF", "tint": 0.3}},  # 填充颜色
        "number_format": {"format_code": "0.00"},  # 数字格式
        "protection": {"locked": True, "hidden": False},  # 保护设置
    }

    # 创建样式对象的各个组成部分
    font_color = styles.Color("00FF0000")
    font = styles.Font(bold=True, color=font_color)
    side = styles.Side(style=styles.borders.BORDER_THIN)
    border = styles.Border(top=side, right=side, bottom=side, left=side)
    alignment = styles.Alignment(horizontal="center", vertical="top")
    fill_color = styles.Color(rgb="006666FF", tint=0.3)
    fill = styles.PatternFill(patternType="solid", fgColor=fill_color)
    number_format = "0.00"
    protection = styles.Protection(locked=True, hidden=False)

    # 将 hstyle 转换为 openpyxl 样式的关键字参数
    kw = _OpenpyxlWriter._convert_to_style_kwargs(hstyle)
    # 断言各个样式组成部分是否与预期一致
    assert kw["font"] == font
    assert kw["border"] == border
    assert kw["alignment"] == alignment
    assert kw["fill"] == fill
    assert kw["number_format"] == number_format
    assert kw["protection"] == protection


def test_write_cells_merge_styled(tmp_excel):
    from pandas.io.formats.excel import ExcelCell  # 导入 Pandas 中处理 Excel 单元格的模块

    sheet_name = "merge_styled"

    # 初始单元格列表，包含两个 ExcelCell 对象
    sty_b1 = {"font": {"color": "00FF0000"}}
    sty_a2 = {"font": {"color": "0000FF00"}}
    initial_cells = [
        ExcelCell(col=1, row=0, val=42, style=sty_b1),
        ExcelCell(col=0, row=1, val=99, style=sty_a2),
    ]

    # 合并后的样式字典
    sty_merged = {"font": {"color": "000000FF", "bold": True}}
    # 将合并后的样式字典转换为 openpyxl 样式的关键字参数
    sty_kwargs = _OpenpyxlWriter._convert_to_style_kwargs(sty_merged)
    openpyxl_sty_merged = sty_kwargs["font"]

    # 合并单元格的列表，包含一个 ExcelCell 对象
    merge_cells = [
        ExcelCell(
            col=0, row=0, val="pandas", mergestart=1, mergeend=1, style=sty_merged
        )
    ]

    # 使用 _OpenpyxlWriter 写入 Excel 单元格
    with _OpenpyxlWriter(tmp_excel) as writer:
        writer._write_cells(initial_cells, sheet_name=sheet_name)
        writer._write_cells(merge_cells, sheet_name=sheet_name)

        wks = writer.sheets[sheet_name]
    # 检查写入的单元格样式是否与预期一致
    xcell_b1 = wks["B1"]
    xcell_a2 = wks["A2"]
    assert xcell_b1.font == openpyxl_sty_merged
    assert xcell_a2.font == openpyxl_sty_merged


@pytest.mark.parametrize("iso_dates", [True, False])
def test_engine_kwargs_write(tmp_excel, iso_dates):
    # GH 42286 GH 43445
    # 创建一个字典，用于传递给 ExcelWriter 的引擎参数，包含 iso_dates 参数
    engine_kwargs = {"iso_dates": iso_dates}
    
    # 使用 ExcelWriter 打开临时 Excel 文件 tmp_excel，指定使用 openpyxl 引擎，并传入引擎参数 engine_kwargs
    with ExcelWriter(
        tmp_excel, engine="openpyxl", engine_kwargs=engine_kwargs
    ) as writer:
        # 断言确认 ExcelWriter 对象中的 iso_dates 属性与传入的 iso_dates 参数相同
        assert writer.book.iso_dates == iso_dates
        
        # ExcelWriter 要求至少写入一些内容后才能关闭
        # 在 writer 上写入一个空的 DataFrame 到 Excel 文件中
        DataFrame().to_excel(writer)
# 定义测试函数，用于测试传递无效的引擎参数是否会触发异常
def test_engine_kwargs_append_invalid(tmp_excel):
    # GH 43445
    # 测试无效引擎参数是否会触发异常
    # 创建一个包含两个字符串的 DataFrame，并将其写入 Excel 文件
    DataFrame(["hello", "world"]).to_excel(tmp_excel)
    # 使用 pytest 检查是否抛出 TypeError 异常，并验证异常信息是否包含指定的内容
    with pytest.raises(
        TypeError,
        match=re.escape(
            "load_workbook() got an unexpected keyword argument 'apple_banana'"
        ),
    ):
        # 使用 ExcelWriter 打开 Excel 文件，使用 openpyxl 引擎，追加模式，并传递一个无效的引擎参数
        with ExcelWriter(
            tmp_excel,
            engine="openpyxl",
            mode="a",
            engine_kwargs={"apple_banana": "fruit"},
        ) as writer:
            # ExcelWriter 需要我们写入一些内容以正常关闭
            # 将一个字符串 DataFrame 写入到 Sheet2
            DataFrame(["good"]).to_excel(writer, sheet_name="Sheet2")


# 参数化测试函数，测试 engine_kwargs 中的 data_only 参数是否正常工作
@pytest.mark.parametrize("data_only, expected", [(True, 0), (False, "=1+1")])
def test_engine_kwargs_append_data_only(tmp_excel, data_only, expected):
    # GH 43445
    # 测试 data_only 引擎参数是否能正常工作，用于 openpyxl 的 load_workbook
    # 创建一个包含 "=1+1" 字符串的 DataFrame，并将其写入 Excel 文件
    DataFrame(["=1+1"]).to_excel(tmp_excel)
    # 使用 ExcelWriter 打开 Excel 文件，使用 openpyxl 引擎，追加模式，并传递 data_only 参数
    with ExcelWriter(
        tmp_excel, engine="openpyxl", mode="a", engine_kwargs={"data_only": data_only}
    ) as writer:
        # 断言确保写入器的 Sheet1 的 B2 单元格的值符合预期
        assert writer.sheets["Sheet1"]["B2"].value == expected
        # ExcelWriter 需要我们写入一些内容以正常关闭？
        # 在 Sheet2 中写入一个空的 DataFrame
        DataFrame().to_excel(writer, sheet_name="Sheet2")

    # 确保 data_only 也能在读取时正常工作，并且公式/值能正常往返
    assert (
        pd.read_excel(
            tmp_excel,
            sheet_name="Sheet1",
            engine="openpyxl",
            engine_kwargs={"data_only": data_only},
        ).iloc[0, 1]
        == expected
    )


# 参数化测试函数，测试是否可以通过 engine_kwargs 将 read_only 和 data_only 传递给 openpyxl.reader.excel.load_workbook
@pytest.mark.parametrize("kwarg_name", ["read_only", "data_only"])
@pytest.mark.parametrize("kwarg_value", [True, False])
def test_engine_kwargs_append_reader(datapath, ext, kwarg_name, kwarg_value):
    # GH 55027
    # 测试是否可以通过 engine_kwargs 将 read_only 和 data_only 传递给 openpyxl.reader.excel.load_workbook
    # 构建文件路径和名称
    filename = datapath("io", "data", "excel", "test1" + ext)
    # 使用 OpenpyxlReader 打开文件，传递 read_only 或 data_only 参数
    with contextlib.closing(
        OpenpyxlReader(filename, engine_kwargs={kwarg_name: kwarg_value})
    ) as reader:
        # 断言确认 reader 对象的相应属性与传递的参数值一致
        assert getattr(reader.book, kwarg_name) == kwarg_value


# 参数化测试函数，测试写入追加模式时的行为
@pytest.mark.parametrize(
    "mode,expected", [("w", ["baz"]), ("a", ["foo", "bar", "baz"])]
)
def test_write_append_mode(tmp_excel, mode, expected):
    # 创建一个包含一个列 "baz" 的 DataFrame
    df = DataFrame([1], columns=["baz"])

    # 创建一个新的 Workbook，写入几个工作表和内容，并保存到 tmp_excel 文件中
    wb = openpyxl.Workbook()
    wb.worksheets[0].title = "foo"
    wb.worksheets[0]["A1"].value = "foo"
    wb.create_sheet("bar")
    wb.worksheets[1]["A1"].value = "bar"
    wb.save(tmp_excel)

    # 使用 ExcelWriter 打开 tmp_excel 文件，使用 openpyxl 引擎，指定模式
    with ExcelWriter(tmp_excel, engine="openpyxl", mode=mode) as writer:
        # 将 DataFrame df 写入到 Sheet 名为 "baz" 的工作表中，不包括索引列
        df.to_excel(writer, sheet_name="baz", index=False)
    # 使用 contextlib.closing 上下文管理打开 Excel 工作簿文件，并赋值给 wb2 对象
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb2:
        # 从 wb2 对象中获取所有工作表的名称，并存储在 result 列表中
        result = [sheet.title for sheet in wb2.worksheets]
        # 断言获取的工作表名称列表与预期的名称列表 expected 相等
        assert result == expected
    
        # 遍历预期值列表 expected 的索引和值
        for index, cell_value in enumerate(expected):
            # 断言第 index 个工作表的单元格 "A1" 的值等于对应的预期值 cell_value
            assert wb2.worksheets[index]["A1"].value == cell_value
@pytest.mark.parametrize(
    "if_sheet_exists,num_sheets,expected",
    [
        ("new", 2, ["apple", "banana"]),  # 第一个测试用例：创建新表格，包含两行数据
        ("replace", 1, ["pear"]),  # 第二个测试用例：替换现有表格，包含一行数据
        ("overlay", 1, ["pear", "banana"]),  # 第三个测试用例：叠加数据到现有表格，包含两行数据
    ],
)
def test_if_sheet_exists_append_modes(tmp_excel, if_sheet_exists, num_sheets, expected):
    # GH 40230
    df1 = DataFrame({"fruit": ["apple", "banana"]})  # 创建包含两种水果的 DataFrame
    df2 = DataFrame({"fruit": ["pear"]})  # 创建包含一种水果的 DataFrame

    df1.to_excel(tmp_excel, engine="openpyxl", sheet_name="foo", index=False)  # 将 df1 写入 Excel 文件，工作表名为 "foo"
    with ExcelWriter(
        tmp_excel, engine="openpyxl", mode="a", if_sheet_exists=if_sheet_exists
    ) as writer:
        df2.to_excel(writer, sheet_name="foo", index=False)  # 将 df2 根据指定模式写入 Excel 文件的 "foo" 表

    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        assert len(wb.sheetnames) == num_sheets  # 断言工作簿中的工作表数量符合预期
        assert wb.sheetnames[0] == "foo"  # 断言第一个工作表的名称为 "foo"
        result = pd.read_excel(wb, "foo", engine="openpyxl")  # 读取名为 "foo" 的工作表数据到 DataFrame 中
        assert list(result["fruit"]) == expected  # 断言读取的数据与预期结果匹配
        if len(wb.sheetnames) == 2:
            result = pd.read_excel(wb, wb.sheetnames[1], engine="openpyxl")  # 如果有第二个工作表，则读取其数据到 DataFrame 中
            tm.assert_frame_equal(result, df2)  # 断言第二个工作表的数据与 df2 相等


@pytest.mark.parametrize(
    "startrow, startcol, greeting, goodbye",
    [
        (0, 0, ["poop", "world"], ["goodbye", "people"]),  # 第一个测试用例：从第一行第一列开始插入数据
        (0, 1, ["hello", "world"], ["poop", "people"]),  # 第二个测试用例：从第一行第二列开始插入数据
        (1, 0, ["hello", "poop"], ["goodbye", "people"]),  # 第三个测试用例：从第二行第一列开始插入数据
        (1, 1, ["hello", "world"], ["goodbye", "poop"]),  # 第四个测试用例：从第二行第二列开始插入数据
    ],
)
def test_append_overlay_startrow_startcol(
    tmp_excel, startrow, startcol, greeting, goodbye
):
    df1 = DataFrame({"greeting": ["hello", "world"], "goodbye": ["goodbye", "people"]})  # 创建包含问候和告别列的 DataFrame
    df2 = DataFrame(["poop"])  # 创建包含一行数据的 DataFrame

    df1.to_excel(tmp_excel, engine="openpyxl", sheet_name="poo", index=False)  # 将 df1 写入 Excel 文件，工作表名为 "poo"
    with ExcelWriter(
        tmp_excel, engine="openpyxl", mode="a", if_sheet_exists="overlay"
    ) as writer:
        # 使用指定的起始行和列，将 df2 写入 Excel 文件的 "poo" 表，无需表头
        df2.to_excel(
            writer,
            index=False,
            header=False,
            startrow=startrow + 1,
            startcol=startcol,
            sheet_name="poo",
        )

    result = pd.read_excel(tmp_excel, sheet_name="poo", engine="openpyxl")  # 从 Excel 文件中读取 "poo" 表的数据到 DataFrame 中
    expected = DataFrame({"greeting": greeting, "goodbye": goodbye})  # 创建预期的 DataFrame
    tm.assert_frame_equal(result, expected)  # 断言读取的数据与预期结果匹配


@pytest.mark.parametrize(
    "if_sheet_exists,msg",
    [
        (
            "invalid",
            "'invalid' is not valid for if_sheet_exists. Valid options "
            "are 'error', 'new', 'replace' and 'overlay'.",
        ),  # 第一个测试用例：使用无效的 if_sheet_exists 参数，期望抛出相应的错误信息
        (
            "error",
            "Sheet 'foo' already exists and if_sheet_exists is set to 'error'.",
        ),  # 第二个测试用例：工作表 "foo" 已存在，且 if_sheet_exists 设置为 'error'，期望抛出相应的错误信息
        (
            None,
            "Sheet 'foo' already exists and if_sheet_exists is set to 'error'.",
        ),  # 第三个测试用例：工作表 "foo" 已存在，且 if_sheet_exists 未设置，期望抛出相应的错误信息
    ],
)
def test_if_sheet_exists_raises(tmp_excel, if_sheet_exists, msg):
    # GH 40230
    df = DataFrame({"fruit": ["pear"]})  # 创建包含一种水果的 DataFrame
    df.to_excel(tmp_excel, sheet_name="foo", engine="openpyxl")  # 将 df 写入 Excel 文件，工作表名为 "foo"
    # 使用 pytest 检查是否会抛出 ValueError 异常，并检查异常消息是否与给定的 msg 变量匹配
    with pytest.raises(ValueError, match=re.escape(msg)):
        # 使用 ExcelWriter 打开临时 Excel 文件 tmp_excel，使用 openpyxl 引擎，以追加模式打开
        # 如果工作表已存在，根据 if_sheet_exists 参数进行处理
        with ExcelWriter(
            tmp_excel, engine="openpyxl", mode="a", if_sheet_exists=if_sheet_exists
        ) as writer:
            # 将 DataFrame df 写入 ExcelWriter 对象的工作表名为 "foo" 的工作表中
            df.to_excel(writer, sheet_name="foo")
# 测试使用 openpyxl 引擎将 DataFrame 写入 Excel
def test_to_excel_with_openpyxl_engine(tmp_excel):
    # GH 29854: GitHub issue标识号
    # 创建两个包含不同数据的 DataFrame
    df1 = DataFrame({"A": np.linspace(1, 10, 10)})
    df2 = DataFrame({"B": np.linspace(1, 20, 10)})
    # 沿着列方向连接两个 DataFrame
    df = pd.concat([df1, df2], axis=1)
    # 对 DataFrame 应用样式，根据条件设置颜色，并突出显示最大值
    styled = df.style.map(
        lambda val: f"color: {'red' if val < 0 else 'black'}"
    ).highlight_max()
    # 将样式应用后的 DataFrame 写入 Excel 文件
    styled.to_excel(tmp_excel, engine="openpyxl")


# 使用参数化测试来验证读取 Excel 工作簿的行为
@pytest.mark.parametrize("read_only", [True, False])
def test_read_workbook(datapath, ext, read_only):
    # GH 39528: GitHub issue标识号
    # 构建文件路径
    filename = datapath("io", "data", "excel", "test1" + ext)
    # 打开 Excel 工作簿文件，使用指定的读取模式
    with contextlib.closing(
        openpyxl.load_workbook(filename, read_only=read_only)
    ) as wb:
        # 从工作簿中读取数据到 DataFrame 中，使用 openpyxl 引擎
        result = pd.read_excel(wb, engine="openpyxl")
    # 从原始文件中直接读取数据到 DataFrame 中，作为预期结果
    expected = pd.read_excel(filename)
    # 断言实际读取的 DataFrame 与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 使用参数化测试验证读取包含不正确维度信息的 Excel 文件的行为
@pytest.mark.parametrize(
    "header, expected_data",
    [
        (
            0,
            {
                "Title": [np.nan, "A", 1, 2, 3],
                "Unnamed: 1": [np.nan, "B", 4, 5, 6],
                "Unnamed: 2": [np.nan, "C", 7, 8, 9],
            },
        ),
        (2, {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
    ],
)
@pytest.mark.parametrize(
    "filename", ["dimension_missing", "dimension_small", "dimension_large"]
)
@pytest.mark.parametrize("read_only", [True, False, None])
def test_read_with_bad_dimension(
    datapath, ext, header, expected_data, filename, read_only
):
    # GH 38956, 39001: GitHub issue标识号 - 没有/不正确的维度信息
    # 构建文件路径
    path = datapath("io", "data", "excel", f"{filename}{ext}")
    # 根据不同的读取模式选择读取 Excel 文件到 DataFrame 中
    if read_only is None:
        result = pd.read_excel(path, header=header)
    else:
        with contextlib.closing(
            openpyxl.load_workbook(path, read_only=read_only)
        ) as wb:
            result = pd.read_excel(wb, engine="openpyxl", header=header)
    # 构建预期的 DataFrame
    expected = DataFrame(expected_data)
    # 断言实际读取的 DataFrame 与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 测试在 Excel 中使用追加模式写入文件
def test_append_mode_file(tmp_excel):
    # GH 39576: GitHub issue标识号
    # 创建一个空的 DataFrame
    df = DataFrame()
    # 将空 DataFrame 写入 Excel 文件，使用 openpyxl 引擎
    df.to_excel(tmp_excel, engine="openpyxl")
    # 打开 ExcelWriter，使用追加模式将 DataFrame 写入同一文件中
    with ExcelWriter(
        tmp_excel, mode="a", engine="openpyxl", if_sheet_exists="new"
    ) as writer:
        df.to_excel(writer)
    # 验证确保 zip 文件未被连接，即 "docProps/app.xml" 在文件中仅出现两次
    data = Path(tmp_excel).read_bytes()
    first = data.find(b"docProps/app.xml")
    second = data.find(b"docProps/app.xml", first + 1)
    third = data.find(b"docProps/app.xml", second + 1)
    assert second != -1 and third == -1


# 使用参数化测试验证读取包含空尾行的 Excel 文件的行为
@pytest.mark.parametrize("read_only", [True, False, None])
def test_read_with_empty_trailing_rows(datapath, ext, read_only):
    # GH 39181: GitHub issue标识号
    # 构建文件路径
    path = datapath("io", "data", "excel", f"empty_trailing_rows{ext}")
    # 根据不同的读取模式选择读取 Excel 文件到 DataFrame 中
    if read_only is None:
        result = pd.read_excel(path)
    else:
        with contextlib.closing(
            openpyxl.load_workbook(path, read_only=read_only)
        ) as wb:
            result = pd.read_excel(wb, engine="openpyxl")
    # 没有预期数据，只需确保能正常读取
    # 否则，如果条件不满足，则执行以下代码块
    else:
        # 使用 contextlib.closing 来确保在完成后关闭打开的文件
        with contextlib.closing(
            openpyxl.load_workbook(path, read_only=read_only)
        ) as wb:
            # 从已加载的 Excel 工作簿中读取数据，并使用 openpyxl 引擎
            result = pd.read_excel(wb, engine="openpyxl")
    # 创建一个期望的 DataFrame，包含预期的数据内容
    expected = DataFrame(
        {
            "Title": [np.nan, "A", 1, 2, 3],
            "Unnamed: 1": [np.nan, "B", 4, 5, 6],
            "Unnamed: 2": [np.nan, "C", 7, 8, 9],
        }
    )
    # 使用 pytest 的 assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 当 read_only 参数为 None 时，使用 read_excel 而不是 workbook
@pytest.mark.parametrize("read_only", [True, False, None])
def test_read_empty_with_blank_row(datapath, ext, read_only):
    # GH 39547 - 空的 Excel 文件，带有一个没有数据的行
    path = datapath("io", "data", "excel", f"empty_with_blank_row{ext}")
    # 如果 read_only 是 None，则使用 pd.read_excel 读取文件内容
    if read_only is None:
        result = pd.read_excel(path)
    else:
        # 否则，使用 openpyxl.load_workbook 打开 Excel 文件
        with contextlib.closing(
            openpyxl.load_workbook(path, read_only=read_only)
        ) as wb:
            # 使用 pd.read_excel 从 workbook 中读取数据
            result = pd.read_excel(wb, engine="openpyxl")
    # 期望结果是一个空的 DataFrame
    expected = DataFrame()
    # 使用 pytest 的 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_book_and_sheets_consistent(tmp_excel):
    # GH#45687 - 确保如果用户修改了 workbook，sheets 会相应更新
    with ExcelWriter(tmp_excel, engine="openpyxl") as writer:
        # 初始时，writer 的 sheets 应为空字典
        assert writer.sheets == {}
        # 创建一个名为 "test_name" 的 sheet，并保存到 sheets 字典中
        sheet = writer.book.create_sheet("test_name", 0)
        # 确保 writer 的 sheets 字典中现在包含刚刚创建的 sheet
        assert writer.sheets == {"test_name": sheet}


def test_ints_spelled_with_decimals(datapath, ext):
    # GH 46988 - openpyxl 返回此表格中的整数为浮点数
    path = datapath("io", "data", "excel", f"ints_spelled_with_decimals{ext}")
    # 使用 pd.read_excel 读取 Excel 文件
    result = pd.read_excel(path)
    # 期望结果是一个包含从 2 到 11 的整数的 DataFrame，列名为 1
    expected = DataFrame(range(2, 12), columns=[1])
    # 使用 pytest 的 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_read_multiindex_header_no_index_names(datapath, ext):
    # GH#47487
    path = datapath("io", "data", "excel", f"multiindex_no_index_names{ext}")
    # 使用 pd.read_excel 读取 Excel 文件，设置多级索引的列和头部
    result = pd.read_excel(path, index_col=[0, 1, 2], header=[0, 1, 2])
    # 期望结果是一个包含 NaN 和字符串 "x" 的 DataFrame，具有多级列索引和多级行索引
    expected = DataFrame(
        [[np.nan, "x", "x", "x"], ["x", np.nan, np.nan, np.nan]],
        columns=pd.MultiIndex.from_tuples(
            [("X", "Y", "A1"), ("X", "Y", "A2"), ("XX", "YY", "B1"), ("XX", "YY", "B2")]
        ),
        index=pd.MultiIndex.from_tuples([("A", "AA", "AAA"), ("A", "BB", "BBB")]),
    )
    # 使用 pytest 的 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)
```