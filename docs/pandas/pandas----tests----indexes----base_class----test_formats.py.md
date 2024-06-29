# `D:\src\scipysrc\pandas\pandas\tests\indexes\base_class\test_formats.py`

```
import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 的配置模块中的特定函数
import pandas._config.config as cf  # 导入 pandas 的配置模块中的 config 对象

from pandas import Index  # 从 pandas 库中导入 Index 类
import pandas._testing as tm  # 导入 pandas 测试模块中的 tm 对象

class TestIndexRendering:
    def test_repr_is_valid_construction_code(self):
        # 对于 Index 类，其 repr 方法返回的字符串是传统的而非样式化的
        idx = Index(["a", "b"])
        res = eval(repr(idx))  # 使用 eval 函数将 repr 返回的字符串重新解析成对象
        tm.assert_index_equal(res, idx)  # 断言重新解析后的对象与原始对象相等

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="repr different")
    @pytest.mark.parametrize(
        "index,expected",
        [
            # ASCII
            # short
            (
                Index(["a", "bb", "ccc"]),
                """Index(['a', 'bb', 'ccc'], dtype='object')""",
            ),
            # multiple lines
            (
                Index(["a", "bb", "ccc"] * 10),
                "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', "
                "'bb', 'ccc', 'a', 'bb', 'ccc',\n"
                "       'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', "
                "'bb', 'ccc', 'a', 'bb', 'ccc',\n"
                "       'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n"
                "      dtype='object')",
            ),
            # truncated
            (
                Index(["a", "bb", "ccc"] * 100),
                "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',\n"
                "       ...\n"
                "       'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n"
                "      dtype='object', length=300)",
            ),
            # Non-ASCII
            # short
            (
                Index(["あ", "いい", "ううう"]),
                """Index(['あ', 'いい', 'ううう'], dtype='object')""",
            ),
            # multiple lines
            (
                Index(["あ", "いい", "ううう"] * 10),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう'],\n"
                    "      dtype='object')"
                ),
            ),
            # truncated
            (
                Index(["あ", "いい", "ううう"] * 100),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ',\n"
                    "       ...\n"
                    "       'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう'],\n"
                    "      dtype='object', length=300)"
                ),
            ),
        ],
    )
    def test_string_index_repr(self, index, expected):
        result = repr(index)  # 调用 Index 对象的 repr 方法
        assert result == expected  # 断言 repr 方法返回的字符串与预期字符串相等

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="repr different")
    @pytest.mark.parametrize(
        "index,expected",
        [
            # short
            (
                Index(["あ", "いい", "ううう"]),
                ("Index(['あ', 'いい', 'ううう'], dtype='object')"),
            ),
            # multiple lines
            (
                Index(["あ", "いい", "ううう"] * 10),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう'],\n"
                    "      dtype='object')"
                    ""
                ),
            ),
            # truncated
            (
                Index(["あ", "いい", "ううう"] * 100),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ',\n"
                    "       ...\n"
                    "       'ううう', 'あ', 'いい', 'ううう', 'あ', "
                    "'いい', 'ううう', 'あ', 'いい',\n"
                    "       'ううう'],\n"
                    "      dtype='object', length=300)"
                ),
            ),
        ],
    )
    def test_string_index_repr_with_unicode_option(self, index, expected):
        """
        使用 pytest 的 parametrize 装饰器，参数化测试方法，对字符串索引的 Unicode 表示进行测试。

        在 Unicode 选项下，使用 cf.option_context 设置，验证索引的字符串表示是否与预期一致。
        """
        with cf.option_context("display.unicode.east_asian_width", True):
            result = repr(index)
            assert result == expected

    def test_repr_summary(self):
        """
        测试索引对象的字符串表示是否满足摘要长度要求。

        使用 cf.option_context 设置最大序列项数量，验证索引对象的字符串表示长度小于200且包含省略号。
        """
        with cf.option_context("display.max_seq_items", 10):
            result = repr(Index(np.arange(1000)))
            assert len(result) < 200
            assert "..." in result

    def test_summary_bug(self):
        """
        测试索引对象的摘要方法 `_summary()` 是否正确处理特定情况。

        验证索引对象的摘要方法不会意外格式化特定的字符串。
        """
        # GH#3869
        ind = Index(["{other}%s", "~:{range}:0"], name="A")
        result = ind._summary()
        assert "~:{range}:0" in result
        assert "{other}%s" in result

    def test_index_repr_bool_nan(self):
        """
        测试包含布尔值和 NaN 的索引对象的字符串表示是否正确。

        创建包含 True、False 和 NaN 的索引对象，并验证其字符串表示是否符合预期。
        """
        # GH32146
        arr = Index([True, False, np.nan], dtype=object)
        exp2 = repr(arr)
        out2 = "Index([True, False, nan], dtype='object')"
        assert out2 == exp2
```