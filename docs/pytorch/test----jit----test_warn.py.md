# `.\pytorch\test\jit\test_warn.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import io
import os
import sys
import warnings
from contextlib import redirect_stderr

import torch
from torch.testing import FileCheck

# 将 test/ 目录下的辅助文件设为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入 JitTestCase 类用于测试
from torch.testing._internal.jit_utils import JitTestCase

# 如果当前脚本被直接运行，抛出运行时错误提示
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义 TestWarn 类，继承 JitTestCase 用于测试警告相关功能
class TestWarn(JitTestCase):

    # 测试在脚本函数中生成警告
    def test_warn(self):
        @torch.jit.script
        def fn():
            warnings.warn("I am warning you")

        # 重定向标准错误到 StringIO 对象 f
        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        # 使用 FileCheck 检查生成的警告信息，确保只出现一次
        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    # 测试在循环中只生成一次警告
    def test_warn_only_once(self):
        @torch.jit.script
        def fn():
            for _ in range(10):
                warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    # 测试在循环中通过函数调用生成警告，确保只生成一次
    def test_warn_only_once_in_loop_func(self):
        def w():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            for _ in range(10):
                w()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    # 测试在函数中调用多个生成警告的函数，确保每个函数都生成一次警告
    def test_warn_once_per_func(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            w1()
            w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    # 测试在循环中多次调用生成警告的函数，确保每个函数总共只生成一次警告
    def test_warn_once_per_func_in_loop(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            for _ in range(10):
                w1()
                w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())
    # 定义一个测试方法，用于测试在多次调用同一函数时产生多个警告的情况
    def test_warn_multiple_calls_multiple_warnings(self):
        # 定义一个 Torch 脚本函数，用于产生警告信息
        @torch.jit.script
        def fn():
            # 发出警告信息
            warnings.warn("I am warning you")
    
        # 创建一个字符串IO对象，用于捕获标准错误流的输出
        f = io.StringIO()
        # 重定向标准错误流到字符串IO对象
        with redirect_stderr(f):
            # 第一次调用 Torch 脚本函数 fn()
            fn()
            # 第二次调用 Torch 脚本函数 fn()
            fn()
    
        # 使用 FileCheck 检查捕获到的输出是否包含指定的警告信息，并确保警告信息出现了两次
        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())
    
    # 定义一个测试方法，用于测试在不同函数调用栈中产生相同警告信息的情况
    def test_warn_multiple_calls_same_func_diff_stack(self):
        # 定义一个普通函数 warn，用于产生带调用者信息的警告
        def warn(caller: str):
            # 发出带有调用者信息的警告
            warnings.warn("I am warning you from " + caller)
    
        # 定义一个 Torch 脚本函数 foo()
        @torch.jit.script
        def foo():
            # 调用 warn 函数，传递字符串 "foo"
            warn("foo")
    
        # 定义一个 Torch 脚本函数 bar()
        @torch.jit.script
        def bar():
            # 调用 warn 函数，传递字符串 "bar"
            warn("bar")
    
        # 创建一个字符串IO对象，用于捕获标准错误流的输出
        f = io.StringIO()
        # 重定向标准错误流到字符串IO对象
        with redirect_stderr(f):
            # 分别调用 Torch 脚本函数 foo() 和 bar()
            foo()
            bar()
    
        # 使用 FileCheck 检查捕获到的输出是否包含指定的警告信息，并确保每个调用的警告信息各出现一次
        FileCheck().check_count(
            str="UserWarning: I am warning you from foo", count=1, exactly=True
        ).check_count(
            str="UserWarning: I am warning you from bar", count=1, exactly=True
        ).run(
            f.getvalue()
        )
```