# `.\numpy\numpy\distutils\tests\test_log.py`

```py
import io  # 导入用于处理输入输出的模块
import re  # 导入正则表达式模块
from contextlib import redirect_stdout  # 导入上下文管理器，用于重定向标准输出流

import pytest  # 导入 pytest 测试框架

from numpy.distutils import log  # 导入 numpy 的日志模块


def setup_module():
    f = io.StringIO()  # 创建一个字符串IO对象，用于捕获日志输出
    with redirect_stdout(f):
        log.set_verbosity(2, force=True)  # 设置日志的详细级别为DEBUG（2）


def teardown_module():
    log.set_verbosity(0, force=True)  # 恢复日志的默认详细级别（0）


r_ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
# 编译正则表达式，用于匹配 ANSI 控制字符，例如颜色控制字符


@pytest.mark.parametrize("func_name", ["error", "warn", "info", "debug"])
def test_log_prefix(func_name):
    func = getattr(log, func_name)  # 获取日志函数，如 log.error、log.warn 等
    msg = f"{func_name} message"  # 设置日志消息内容
    f = io.StringIO()  # 创建一个字符串IO对象，用于捕获日志输出
    with redirect_stdout(f):
        func(msg)  # 调用日志函数，将消息写入日志
    out = f.getvalue()  # 获取捕获的日志输出内容
    assert out  # 确保捕获的输出不为空，作为健全性检查
    clean_out = r_ansi.sub("", out)  # 使用正则表达式去除输出中的 ANSI 控制字符
    line = next(line for line in clean_out.splitlines())  # 获取处理后输出的第一行
    assert line == f"{func_name.upper()}: {msg}"  # 断言输出的第一行格式正确
```