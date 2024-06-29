# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_console.py`

```
import locale  # 导入locale模块，用于处理地区相关的编码设置

import pytest  # 导入pytest模块，用于编写和运行测试

from pandas._config import detect_console_encoding  # 导入detect_console_encoding函数，用于检测控制台编码


class MockEncoding:
    """
    Used to add a side effect when accessing the 'encoding' property. If the
    side effect is a str in nature, the value will be returned. Otherwise, the
    side effect should be an exception that will be raised.
    """

    def __init__(self, encoding) -> None:
        super().__init__()
        self.val = encoding

    @property
    def encoding(self):
        return self.raise_or_return(self.val)  # 返回self.val所指定的编码或者引发相关异常

    @staticmethod
    def raise_or_return(val):
        if isinstance(val, str):
            return val  # 如果val是字符串，则直接返回该字符串作为编码
        else:
            raise val  # 如果val是异常类，则引发该异常


@pytest.mark.parametrize("empty,filled", [["stdin", "stdout"], ["stdout", "stdin"]])
def test_detect_console_encoding_from_stdout_stdin(monkeypatch, empty, filled):
    # Ensures that when sys.stdout.encoding or sys.stdin.encoding is used when
    # they have values filled.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr(f"sys.{empty}", MockEncoding(""))  # 设置sys.stdin或sys.stdout的编码为""
        context.setattr(f"sys.{filled}", MockEncoding(filled))  # 设置sys.stdin或sys.stdout的编码为filled
        assert detect_console_encoding() == filled  # 断言detect_console_encoding函数返回filled作为编码


@pytest.mark.parametrize("encoding", [AttributeError, OSError, "ascii"])
def test_detect_console_encoding_fallback_to_locale(monkeypatch, encoding):
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr("locale.getpreferredencoding", lambda: "foo")  # 设置locale.getpreferredencoding返回"foo"
        context.setattr("sys.stdout", MockEncoding(encoding))  # 设置sys.stdout的编码为encoding
        assert detect_console_encoding() == "foo"  # 断言detect_console_encoding函数返回"foo"


@pytest.mark.parametrize(
    "std,locale",
    [
        ["ascii", "ascii"],
        ["ascii", locale.Error],
        [AttributeError, "ascii"],
        [AttributeError, locale.Error],
        [OSError, "ascii"],
        [OSError, locale.Error],
    ],
)
def test_detect_console_encoding_fallback_to_default(monkeypatch, std, locale):
    # When both the stdout/stdin encoding and locale preferred encoding checks
    # fail (or return 'ascii', we should default to the sys default encoding.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr(
            "locale.getpreferredencoding", lambda: MockEncoding.raise_or_return(locale)
        )  # 设置locale.getpreferredencoding返回MockEncoding.raise_or_return(locale)的结果
        context.setattr("sys.stdout", MockEncoding(std))  # 设置sys.stdout的编码为std
        context.setattr("sys.getdefaultencoding", lambda: "sysDefaultEncoding")  # 设置sys.getdefaultencoding返回"sysDefaultEncoding"
        assert detect_console_encoding() == "sysDefaultEncoding"  # 断言detect_console_encoding函数返回"sysDefaultEncoding"
```