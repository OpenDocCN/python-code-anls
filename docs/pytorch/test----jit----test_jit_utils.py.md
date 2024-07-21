# `.\pytorch\test\jit\test_jit_utils.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库
import os
import sys
from textwrap import dedent

# 导入 PyTorch 相关模块
import torch

# 导入测试相关的 JIT 工具函数
from torch.testing._internal import jit_utils

# 将测试目录下的 helper 文件夹加入 sys.path，使得其可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入 JIT 测试用例基类
from torch.testing._internal.jit_utils import JitTestCase

# 如果直接运行此文件，则抛出 RuntimeError 提示用户不要直接运行该文件，而是通过 test/test_jit.py 来运行特定测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


# 测试各种 JIT 相关的实用函数。
class TestJitUtils(JitTestCase):
    
    # 测试捕获 POSITIONAL_OR_KEYWORD 参数
    def test_get_callable_argument_names_positional_or_keyword(self):
        def fn_positional_or_keyword_args_only(x, y):
            return x + y

        self.assertEqual(
            ["x", "y"],
            torch._jit_internal.get_callable_argument_names(
                fn_positional_or_keyword_args_only
            ),
        )

    # 测试忽略 POSITIONAL_ONLY 参数
    def test_get_callable_argument_names_positional_only(self):
        code = dedent(
            """
            def fn_positional_only_arg(x, /, y):
                return x + y
        """
        )

        # 从代码片段中获取函数对象
        fn_positional_only_arg = jit_utils._get_py3_code(code, "fn_positional_only_arg")
        self.assertEqual(
            ["y"],
            torch._jit_internal.get_callable_argument_names(fn_positional_only_arg),
        )

    # 测试忽略 VAR_POSITIONAL 参数
    def test_get_callable_argument_names_var_positional(self):
        def fn_var_positional_arg(x, *arg):
            return x + arg[0]

        self.assertEqual(
            ["x"],
            torch._jit_internal.get_callable_argument_names(fn_var_positional_arg),
        )

    # 测试忽略 KEYWORD_ONLY 参数
    def test_get_callable_argument_names_keyword_only(self):
        def fn_keyword_only_arg(x, *, y):
            return x + y

        self.assertEqual(
            ["x"], torch._jit_internal.get_callable_argument_names(fn_keyword_only_arg)
        )

    # 测试忽略 VAR_KEYWORD 参数
    def test_get_callable_argument_names_var_keyword(self):
        def fn_var_keyword_arg(**args):
            return args["x"] + args["y"]

        self.assertEqual(
            [], torch._jit_internal.get_callable_argument_names(fn_var_keyword_arg)
        )

    # 测试忽略包含各种不同类型参数的函数签名
    # 测试获取混合参数函数的参数名列表
    def test_get_callable_argument_names_hybrid(self):
        # 定义一个混合参数函数，包括位置参数 x 和 y，可变位置参数 args，可变关键字参数 kwargs
        code = dedent(
            """
            def fn_hybrid_args(x, /, y, *args, **kwargs):
                return x + y + args[0] + kwargs['z']
            """
        )
        # 获取函数 fn_hybrid_args 的 Python 3 兼容代码对象
        fn_hybrid_args = jit_utils._get_py3_code(code, "fn_hybrid_args")
        # 使用 PyTorch 内部方法获取函数的可调用参数名列表，预期返回 ["y"]
        self.assertEqual(
            ["y"], torch._jit_internal.get_callable_argument_names(fn_hybrid_args)
        )

    # 测试检查脚本抛出正则表达式匹配的异常
    def test_checkscriptassertraisesregex(self):
        # 定义一个函数 fn，试图访问元组的超出索引的元素
        def fn():
            tup = (1, 2)
            return tup[2]

        # 使用测试框架方法检查函数 fn 在不带参数的情况下是否抛出异常 Exception，且异常消息中包含字符串 "range"
        self.checkScriptRaisesRegex(fn, (), Exception, "range", name="fn")

        # 定义一个包含错误的脚本字符串 s
        s = dedent(
            """
            def fn():
                tup = (1, 2)
                return tup[2]
            """
        )

        # 使用测试框架方法检查脚本字符串 s 在不带参数的情况下是否抛出异常 Exception，且异常消息中包含字符串 "range"
        self.checkScriptRaisesRegex(s, (), Exception, "range", name="fn")

    # 测试禁用追踪器警告的上下文管理器
    def test_no_tracer_warn_context_manager(self):
        # 设置 PyTorch 追踪器状态为警告模式
        torch._C._jit_set_tracer_state_warn(True)
        # 使用禁用追踪器警告的上下文管理器，验证警告状态是否为 False
        with jit_utils.NoTracerWarnContextManager() as no_warn:
            self.assertEqual(False, torch._C._jit_get_tracer_state_warn())
        # 离开上下文管理器后，验证追踪器状态是否恢复为 True
        self.assertEqual(True, torch._C._jit_get_tracer_state_warn())
```