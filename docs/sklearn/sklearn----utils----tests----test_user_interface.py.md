# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_user_interface.py`

```
# 导入必要的模块和函数
import string  # 导入字符串模块，用于生成 ASCII 字母表和其他字符串操作
import timeit  # 导入计时模块，用于测量代码执行时间

import pytest  # 导入 pytest 测试框架

from sklearn.utils._user_interface import _message_with_time, _print_elapsed_time  # 导入 sklearn 的私有函数

# 使用 pytest.mark.parametrize 装饰器定义参数化测试
@pytest.mark.parametrize(
    ["source", "message", "is_long"],
    [  # 参数化的测试用例列表
        ("ABC", string.ascii_lowercase, False),  # 短 source 和 message，is_long 为 False
        ("ABCDEF", string.ascii_lowercase, False),  # 短 source 和 message，is_long 为 False
        ("ABC", string.ascii_lowercase * 3, True),  # 短 source 和长 message，is_long 为 True
        ("ABC" * 10, string.ascii_lowercase, True),  # 长 source 和短 message，is_long 为 True
        ("ABC", string.ascii_lowercase + "\u1048", False),  # 短 source 和包含特殊字符的 message，is_long 为 False
    ],
)
@pytest.mark.parametrize(
    ["time", "time_str"],
    [  # 参数化的时间和时间字符串
        (0.2, "   0.2s"),  # 0.2 秒的时间和其格式化字符串
        (20, "  20.0s"),  # 20 秒的时间和其格式化字符串
        (2000, "33.3min"),  # 2000 秒的时间和其格式化字符串
        (20000, "333.3min"),  # 20000 秒的时间和其格式化字符串
    ],
)
def test_message_with_time(source, message, is_long, time, time_str):
    # 调用 _message_with_time 函数获取输出结果
    out = _message_with_time(source, message, time)
    
    # 根据 is_long 进行断言
    if is_long:
        assert len(out) > 70  # 如果 is_long 为 True，确保输出长度大于 70
    else:
        assert len(out) == 70  # 如果 is_long 为 False，确保输出长度等于 70

    # 断言输出以 "[" + source + "] " 开头
    assert out.startswith("[" + source + "] ")
    out = out[len(source) + 3 :]  # 去除开头的 source 字符串及其后的空格

    # 断言输出以 time_str 结尾
    assert out.endswith(time_str)
    out = out[: -len(time_str)]  # 去除末尾的 time_str 字符串

    # 继续断言输出以 ", total=" 结尾
    assert out.endswith(", total=")
    out = out[: -len(", total=")]  # 去除末尾的 ", total="

    # 断言输出以 message 结尾
    assert out.endswith(message)
    out = out[: -len(message)]  # 去除末尾的 message

    # 最后断言输出以单个空格结尾，除非是 is_long 为 True，此时应该为空字符串
    assert out.endswith(" ")
    out = out[:-1]  # 去除末尾的空格

    if is_long:
        assert not out  # 如果 is_long 为 True，则最终输出应为空字符串
    else:
        assert list(set(out)) == ["."]  # 如果 is_long 为 False，则最终输出应为单个"."字符的集合

# 参数化测试函数，测试 _print_elapsed_time 函数的行为
@pytest.mark.parametrize(
    ["message", "expected"],
    [
        ("hello", _message_with_time("ABC", "hello", 0.1) + "\n"),  # 测试带有 message 的情况
        ("", _message_with_time("ABC", "", 0.1) + "\n"),  # 测试空 message 的情况
        (None, ""),  # 测试 None message 的情况
    ],
)
def test_print_elapsed_time(message, expected, capsys, monkeypatch):
    # 使用 monkeypatch 设置计时器为固定值
    monkeypatch.setattr(timeit, "default_timer", lambda: 0)
    
    # 执行 _print_elapsed_time 函数，捕获输出
    with _print_elapsed_time("ABC", message):
        monkeypatch.setattr(timeit, "default_timer", lambda: 0.1)
    
    # 断言捕获的输出与期望的输出相等
    assert capsys.readouterr().out == expected
```