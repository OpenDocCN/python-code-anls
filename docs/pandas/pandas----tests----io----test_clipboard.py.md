# `D:\src\scipysrc\pandas\pandas\tests\io\test_clipboard.py`

```
# 导入所需模块和函数
from textwrap import dedent  # 从textwrap模块导入dedent函数，用于缩进文本处理

import numpy as np  # 导入numpy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas.errors import (  # 从pandas.errors模块导入特定异常类
    PyperclipException,  # 异常类：剪贴板异常
    PyperclipWindowsException,  # 异常类：Windows剪贴板异常
)

import pandas as pd  # 导入pandas库，用于数据处理和分析
from pandas import (  # 从pandas库中导入多个函数和对象
    NA,  # 常量：表示缺失值
    DataFrame,  # 类：表示二维表格数据
    Series,  # 类：表示一维标记数组
    get_option,  # 函数：获取指定选项的当前值
    read_clipboard,  # 函数：从剪贴板读取数据并返回DataFrame
)
import pandas._testing as tm  # 导入pandas内部测试模块，用于测试辅助函数

from pandas.core.arrays import (  # 从pandas.core.arrays模块导入数组类型
    ArrowStringArray,  # 类：Arrow库支持的字符串数组
    StringArray,  # 类：字符串数组
)

from pandas.io.clipboard import (  # 从pandas.io.clipboard模块导入剪贴板操作相关函数
    CheckedCall,  # 类：封装函数调用及异常处理
    _stringifyText,  # 函数：将文本转换为字符串
    init_qt_clipboard,  # 函数：初始化Qt剪贴板
)


def build_kwargs(sep, excel):
    """
    根据参数sep和excel构建关键字参数字典。
    如果excel不为"default"，则将其加入kwargs字典。
    如果sep不为"default"，则将其加入kwargs字典。
    返回构建好的kwargs字典。
    """
    kwargs = {}  # 初始化空字典kwargs

    if excel != "default":
        kwargs["excel"] = excel  # 如果excel不为"default"，将其加入kwargs字典

    if sep != "default":
        kwargs["sep"] = sep  # 如果sep不为"default"，将其加入kwargs字典

    return kwargs  # 返回构建好的关键字参数字典


@pytest.fixture(
    params=[
        "delims",  # 参数化测试：分隔符测试数据类型
        "utf8",  # 参数化测试：UTF-8编码测试数据类型
        "utf16",  # 参数化测试：UTF-16编码测试数据类型
        "string",  # 参数化测试：字符串数据类型
        "long",  # 参数化测试：长数据类型
        "nonascii",  # 参数化测试：非ASCII字符数据类型
        "colwidth",  # 参数化测试：列宽数据类型
        "mixed",  # 参数化测试：混合数据类型
        "float",  # 参数化测试：浮点数数据类型
        "int",  # 参数化测试：整数数据类型
    ]
)
def df(request):
    """
    根据参数request.param的不同值，返回不同类型的DataFrame。
    参数化测试数据包括多种数据类型的DataFrame。
    """
    data_type = request.param  # 获取参数化测试的当前参数值

    if data_type == "delims":
        return DataFrame({"a": ['"a,\t"b|c', "d\tef`"], "b": ["hi'j", "k''lm"]})
    elif data_type == "utf8":
        return DataFrame({"a": ["µasd", "Ωœ∑`"], "b": ["øπ∆˚¬", "œ∑`®"]})
    elif data_type == "utf16":
        return DataFrame(
            {"a": ["\U0001f44d\U0001f44d", "\U0001f44d\U0001f44d"], "b": ["abc", "def"]}
        )
    elif data_type == "string":
        return DataFrame(
            np.array([f"i-{i}" for i in range(15)]).reshape(5, 3), columns=list("abc")
        )
    elif data_type == "long":
        max_rows = get_option("display.max_rows")
        return DataFrame(
            np.random.default_rng(2).integers(0, 10, size=(max_rows + 1, 3)),
            columns=list("abc"),
        )
    elif data_type == "nonascii":
        return DataFrame({"en": "in English".split(), "es": "en español".split()})
    elif data_type == "colwidth":
        _cw = get_option("display.max_colwidth") + 1
        return DataFrame(
            np.array(["x" * _cw for _ in range(15)]).reshape(5, 3), columns=list("abc")
        )
    elif data_type == "mixed":
        return DataFrame(
            {
                "a": np.arange(1.0, 6.0) + 0.01,
                "b": np.arange(1, 6).astype(np.int64),
                "c": list("abcde"),
            }
        )
    elif data_type == "float":
        return DataFrame(np.random.default_rng(2).random((5, 3)), columns=list("abc"))
    elif data_type == "int":
        return DataFrame(
            np.random.default_rng(2).integers(0, 10, (5, 3)), columns=list("abc")
        )
    else:
        raise ValueError


@pytest.fixture
def mock_ctypes(monkeypatch):
    """
    使用monkeypatch模拟Windows平台下的WinError异常。
    """

    def _mock_win_error():
        return "Window Error"  # 返回模拟的Windows错误信息

    # 在非Windows平台下设置raising=False，避免因WinError异常导致测试失败
    with monkeypatch.context() as m:
        m.setattr("ctypes.WinError", _mock_win_error, raising=False)
        yield  # 返回mock_ctypes的生成器


@pytest.mark.usefixtures("mock_ctypes")
def test_checked_call_with_bad_call(monkeypatch):
    """
    使用pytest.mark.usefixtures装饰器指定mock_ctypes作为测试函数test_checked_call_with_bad_call的前置装置。
    """
    Give CheckCall a function that returns a falsey value and
    mock get_errno so it returns false so an exception is raised.
    """

    # 定义一个返回假值的函数
    def _return_false():
        return False

    # 使用 monkeypatch 来替换 pandas.io.clipboard.get_errno 的实现，使其返回 True
    monkeypatch.setattr("pandas.io.clipboard.get_errno", lambda: True)
    
    # 构建异常匹配的消息字符串
    msg = f"Error calling {_return_false.__name__} \\(Window Error\\)"
    
    # 使用 pytest 的 pytest.raises 上下文管理器来断言抛出 PyperclipWindowsException 异常，并匹配指定的消息
    with pytest.raises(PyperclipWindowsException, match=msg):
        # 调用 CheckedCall，并传入 _return_false 函数作为参数，期待抛出异常
        CheckedCall(_return_false)()
@pytest.mark.usefixtures("mock_ctypes")
# 使用 pytest.mark.usefixtures 装饰器，指定在运行测试时使用 mock_ctypes fixture

def test_checked_call_with_valid_call(monkeypatch):
    """
    Give CheckCall a function that returns a truthy value and
    mock get_errno so it returns true so an exception is not raised.
    The function should return the results from _return_true.
    """
    # 定义一个总是返回 True 的函数 _return_true
    def _return_true():
        return True
    
    # 使用 monkeypatch.setattr 来模拟 pandas.io.clipboard.get_errno 函数始终返回 False
    monkeypatch.setattr("pandas.io.clipboard.get_errno", lambda: False)

    # 创建 CheckedCall 实例，传入 _return_true 函数作为参数
    checked_call = CheckedCall(_return_true)
    
    # 断言 checked_call() 调用结果为 True
    assert checked_call() is True


@pytest.mark.parametrize(
    "text",
    [
        "String_test",
        True,
        1,
        1.0,
        1j,
    ],
)
# 使用 pytest.mark.parametrize 装饰器，为 test_stringify_text 函数指定多组参数进行测试
def test_stringify_text(text):
    # 定义有效的类型元组
    valid_types = (str, int, float, bool)

    # 检查 text 是否属于 valid_types 中的类型之一
    if isinstance(text, valid_types):
        # 如果是有效类型，则调用 _stringifyText 函数，并断言其结果与 str(text) 相等
        result = _stringifyText(text)
        assert result == str(text)
    else:
        # 如果不是有效类型，则生成异常消息
        msg = (
            "only str, int, float, and bool values "
            f"can be copied to the clipboard, not {type(text).__name__}"
        )
        # 使用 pytest.raises 断言捕获 PyperclipException 异常，并匹配特定的异常消息
        with pytest.raises(PyperclipException, match=msg):
            _stringifyText(text)


@pytest.fixture
# 定义 pytest fixture：set_pyqt_clipboard，用于设置 PyQt 的剪贴板操作

def set_pyqt_clipboard(monkeypatch):
    # 初始化 PyQt 剪贴板函数，并在 monkeypatch context 中使用
    qt_cut, qt_paste = init_qt_clipboard()
    with monkeypatch.context() as m:
        # 设置 pd.io.clipboard.clipboard_set 和 pd.io.clipboard.clipboard_get 的属性
        m.setattr(pd.io.clipboard, "clipboard_set", qt_cut)
        m.setattr(pd.io.clipboard, "clipboard_get", qt_paste)
        # yield 用于返回 fixture 对象
        yield


@pytest.fixture
# 定义 pytest fixture：clipboard，用于处理 Qt 应用程序的剪贴板

def clipboard(qapp):
    # 获取 Qt 应用程序的剪贴板
    clip = qapp.clipboard()
    # yield 用于返回 fixture 对象
    yield clip
    # 清空剪贴板内容
    clip.clear()


@pytest.mark.single_cpu
@pytest.mark.clipboard
@pytest.mark.usefixtures("set_pyqt_clipboard")
@pytest.mark.usefixtures("clipboard")
# 使用多个 pytest.mark 装饰器标记测试类 TestClipboard，指定单 CPU 执行和剪贴板相关的标记

class TestClipboard:
    # 测试默认参数复制为制表符分隔符的功能
    # 测试明确指定分隔符时是否得到尊重
    @pytest.mark.parametrize("sep", [None, "\t", ",", "|"])
    @pytest.mark.parametrize("encoding", [None, "UTF-8", "utf-8", "utf8"])
    # 使用 pytest.mark.parametrize 装饰器，为 test_round_trip_frame_sep 函数指定多组参数进行测试
    def test_round_trip_frame_sep(self, df, sep, encoding):
        # 将 DataFrame 内容复制到剪贴板，指定 excel=None，sep 和 encoding 参数
        df.to_clipboard(excel=None, sep=sep, encoding=encoding)
        # 读取剪贴板内容为 DataFrame 对象 result
        result = read_clipboard(sep=sep or "\t", index_col=0, encoding=encoding)
        # 使用 assert_frame_equal 比较 df 和 result，断言二者相等
        tm.assert_frame_equal(df, result)

    # 测试使用空格分隔符的功能
    def test_round_trip_frame_string(self, df):
        # 将 DataFrame 内容复制到剪贴板，指定 excel=False，sep=None 参数
        df.to_clipboard(excel=False, sep=None)
        # 读取剪贴板内容为 DataFrame 对象 result
        result = read_clipboard()
        # 使用 assert 断言 df 的字符串形式与 result 的字符串形式相等
        assert df.to_string() == result.to_string()
        # 使用 assert 断言 df 的形状与 result 的形状相等
        assert df.shape == result.shape

    # 不支持两字符分隔符的功能
    # 测试当 excel=True 时，多字符分隔符不能静默传递的功能
    def test_excel_sep_warning(self, df):
        # 使用 assert_produces_warning 检查是否生成 UserWarning 异常，匹配特定的异常消息
        with tm.assert_produces_warning(
            UserWarning,
            match="to_clipboard in excel mode requires a single character separator.",
            check_stacklevel=False,
        ):
            # 将 DataFrame 内容复制到剪贴板，指定 excel=True，sep=r"\t" 参数
            df.to_clipboard(excel=True, sep=r"\t")

    # 当 excel=False 时分隔符被忽略并产生警告
    # 测试 excel=False 时分隔符被忽略并产生警告的功能
    # 测试函数：测试在指定条件下是否会产生 UserWarning 警告
    def test_copy_delim_warning(self, df):
        # 使用上下文管理器检查是否会产生 UserWarning，并且匹配警告信息包含 "ignores the sep argument"
        with tm.assert_produces_warning(UserWarning, match="ignores the sep argument"):
            # 将 DataFrame 内容复制到剪贴板，禁用 Excel 格式，设置分隔符为制表符 "\t"
            df.to_clipboard(excel=False, sep="\t")

    # 测试函数：验证 to_clipboard 方法的默认行为是使用制表符分隔并启用 Excel 模式
    @pytest.mark.parametrize("sep", ["\t", None, "default"])
    @pytest.mark.parametrize("excel", [True, None, "default"])
    def test_clipboard_copy_tabs_default(self, sep, excel, df, clipboard):
        # 构建函数参数字典
        kwargs = build_kwargs(sep, excel)
        # 将 DataFrame 内容复制到剪贴板，使用给定的参数字典
        df.to_clipboard(**kwargs)
        # 断言剪贴板的文本内容与 DataFrame 转换为 CSV 格式后的内容相同（使用制表符分隔）
        assert clipboard.text() == df.to_csv(sep="\t")

    # 测试函数：验证读取空格分隔表格的功能
    @pytest.mark.parametrize("sep", [None, "default"])
    def test_clipboard_copy_strings(self, sep, df):
        # 构建函数参数字典，禁用 Excel 模式
        kwargs = build_kwargs(sep, False)
        # 将 DataFrame 内容复制到剪贴板，使用给定的参数字典
        df.to_clipboard(**kwargs)
        # 读取剪贴板内容，分隔符使用正则表达式 \s+ 来匹配空白字符
        result = read_clipboard(sep=r"\s+")
        # 断言读取的 DataFrame 内容与原始 DataFrame 内容一致
        assert result.to_string() == df.to_string()
        # 断言 DataFrame 的形状与读取结果的形状相同
        assert df.shape == result.shape

    # 测试函数：验证通过剪贴板推断 Excel 格式的功能
    def test_read_clipboard_infer_excel(self, clipboard):
        # 避免警告：设置剪贴板读取参数为 "python" 引擎
        clip_kwargs = {"engine": "python"}

        # 设置剪贴板的文本内容
        text = dedent(
            """
            John James\tCharlie Mingus
            1\t2
            4\tHarry Carney
            """.strip()
        )
        clipboard.setText(text)
        # 从剪贴板读取 DataFrame
        df = read_clipboard(**clip_kwargs)

        # 断言 Excel 数据正确解析
        assert df.iloc[1, 1] == "Harry Carney"

        # 不同的制表符数量不会触发警告
        text = dedent(
            """
            a\t b
            1  2
            3  4
            """.strip()
        )
        clipboard.setText(text)
        res = read_clipboard(**clip_kwargs)

        text = dedent(
            """
            a  b
            1  2
            3  4
            """.strip()
        )
        clipboard.setText(text)
        exp = read_clipboard(**clip_kwargs)

        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(res, exp)

    # 测试函数：验证包含空值的 Excel 数据解析功能
    def test_infer_excel_with_nulls(self, clipboard):
        # GH41108
        # 设置剪贴板的文本内容，包含空值的 Excel 数据
        text = "col1\tcol2\n1\tred\n\tblue\n2\tgreen"

        clipboard.setText(text)
        # 从剪贴板读取 DataFrame
        df = read_clipboard()
        # 期望的 DataFrame 结果，包含空值
        df_expected = DataFrame(
            data={"col1": [1, None, 2], "col2": ["red", "blue", "green"]}
        )

        # 断言 Excel 数据正确解析
        tm.assert_frame_equal(df, df_expected)
    @pytest.mark.parametrize(
        "multiindex",
        [
            (  # Can't use `dedent` here as it will remove the leading `\t`
                "\n".join(
                    [
                        "\t\t\tcol1\tcol2",  # 第一个数据集的列标题
                        "A\t0\tTrue\t1\tred",  # 第一个数据集的第一行数据
                        "A\t1\tTrue\t\tblue",  # 第一个数据集的第二行数据
                        "B\t0\tFalse\t2\tgreen",  # 第一个数据集的第三行数据
                    ]
                ),
                [["A", "A", "B"], [0, 1, 0], [True, True, False]],  # 期望的多重索引值
            ),
            (
                "\n".join(
                    ["\t\tcol1\tcol2",  # 第二个数据集的列标题
                     "A\t0\t1\tred",  # 第二个数据集的第一行数据
                     "A\t1\t\tblue",  # 第二个数据集的第二行数据
                     "B\t0\t2\tgreen"]  # 第二个数据集的第三行数据
                ),
                [["A", "A", "B"], [0, 1, 0]],  # 期望的多重索引值
            ),
        ],
    )
    def test_infer_excel_with_multiindex(self, clipboard, multiindex):
        # GH41108
        clipboard.setText(multiindex[0])  # 将剪贴板内容设置为第一个数据集
        df = read_clipboard()  # 从剪贴板读取数据作为DataFrame
        df_expected = DataFrame(
            data={"col1": [1, None, 2], "col2": ["red", "blue", "green"]},  # 期望的DataFrame数据
            index=multiindex[1],  # 指定的多重索引
        )

        # excel data is parsed correctly
        tm.assert_frame_equal(df, df_expected)  # 断言实际DataFrame与期望DataFrame相等

    def test_invalid_encoding(self, df):
        msg = "clipboard only supports utf-8 encoding"  # 错误消息
        # test case for testing invalid encoding
        with pytest.raises(ValueError, match=msg):  # 断言抛出值错误且错误消息匹配
            df.to_clipboard(encoding="ascii")  # 尝试使用ASCII编码复制到剪贴板
        with pytest.raises(NotImplementedError, match=msg):  # 断言抛出未实现错误且错误消息匹配
            read_clipboard(encoding="ascii")  # 尝试使用ASCII编码从剪贴板读取

    @pytest.mark.parametrize("data", ["\U0001f44d...", "Ωœ∑`...", "abcd..."])
    def test_raw_roundtrip(self, data):
        # PR #25040 wide unicode wasn't copied correctly on PY3 on windows
        df = DataFrame({"data": [data]})  # 创建包含给定数据的DataFrame
        df.to_clipboard()  # 将DataFrame复制到剪贴板
        result = read_clipboard()  # 从剪贴板读取数据
        tm.assert_frame_equal(df, result)  # 断言实际DataFrame与从剪贴板读取的DataFrame相等

    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_read_clipboard_dtype_backend(
        self, clipboard, string_storage, dtype_backend, engine
    ):
        # GH#50502
        if string_storage == "pyarrow" or dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")  # 导入pyarrow，如果未安装则跳过测试

        if string_storage == "python":
            string_array = StringArray(np.array(["x", "y"], dtype=np.object_))  # 创建Python字符串数组
            string_array_na = StringArray(np.array(["x", NA], dtype=np.object_))  # 创建带有NA值的Python字符串数组

        elif dtype_backend == "pyarrow" and engine != "c":
            pa = pytest.importorskip("pyarrow")  # 导入pyarrow，如果未安装则跳过测试
            from pandas.arrays import ArrowExtensionArray

            string_array = ArrowExtensionArray(pa.array(["x", "y"]))  # 创建Arrow扩展数组
            string_array_na = ArrowExtensionArray(pa.array(["x", None]))  # 创建带有None值的Arrow扩展数组

        else:
            string_array = ArrowStringArray(pa.array(["x", "y"]))  # 创建Arrow字符串数组
            string_array_na = ArrowStringArray(pa.array(["x", None]))  # 创建带有None值的Arrow字符串数组

        text = """a,b,c,d,e,f,g,h,i
        clipboard.setText(text)
        # 将文本设置到剪贴板中

        with pd.option_context("mode.string_storage", string_storage):
            # 设置 pandas 上下文，指定字符串存储方式
            result = read_clipboard(sep=",", dtype_backend=dtype_backend, engine=engine)
            # 调用 read_clipboard 函数从剪贴板读取数据，指定分隔符和数据类型后端

        expected = DataFrame(
            {
                "a": string_array,
                "b": Series([1, 2], dtype="Int64"),
                "c": Series([4.0, 5.0], dtype="Float64"),
                "d": string_array_na,
                "e": Series([2, NA], dtype="Int64"),
                "f": Series([4.0, NA], dtype="Float64"),
                "g": Series([NA, NA], dtype="Int64"),
                "h": Series([True, False], dtype="boolean"),
                "i": Series([False, NA], dtype="boolean"),
            }
        )
        # 定义预期的 DataFrame 结构和数据

        if dtype_backend == "pyarrow":
            # 如果 dtype_backend 是 "pyarrow"
            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )
            expected["g"] = ArrowExtensionArray(pa.array([None, None]))
            # 将预期的 DataFrame 转换为 ArrowExtensionArray 格式，并调整列 'g'

        tm.assert_frame_equal(result, expected)
        # 使用 pandas 测试模块比较实际结果和预期结果的 DataFrame

    def test_invalid_dtype_backend(self):
        # 定义测试函数，测试无效的 dtype_backend 参数
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            # 使用 pytest 断言异常，检查错误消息是否匹配
            read_clipboard(dtype_backend="numpy")
            # 调用 read_clipboard 函数，传入无效的 dtype_backend 参数
```