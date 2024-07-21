# `.\pytorch\test\dynamo\test_exc.py`

```py
# Owner(s): ["module: dynamo"]

# 引入日志记录模块
import logging
# 引入单元测试模块
import unittest

# 引入 PyTorch 相关模块
import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case

# 引入自定义模块和函数
from torch._dynamo.comptime import comptime
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_device_type import skipIf
from torch.testing._internal.common_utils import IS_FBCODE, munge_exc, TEST_Z3
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test

# 定义一个测试类，继承自 LoggingTestCase
class ExcTests(LoggingTestCase):
    maxDiff = None

    # 测试函数：测试 Unsupported 异常的实际堆栈信息
    def test_unsupported_real_stack(self):
        # 定义内部函数 fn002，调用 torch._dynamo.graph_break()
        def fn002(x):
            torch._dynamo.graph_break()

        # 定义内部函数 fn001，在其中调用 fn002
        def fn001(x):
            x = x + 1
            fn002(x)

        # 断言实际输出与期望输出一致
        self.assertExpectedInlineMunged(
            Unsupported,
            # 调用 torch.compile 函数编译 fn001，使用 eager 后端和完整图形分析
            lambda: torch.compile(fn001, backend="eager", fullgraph=True)(
                torch.randn(1)
            ),
            """\
'skip function graph_break in file _dynamo/decorators.py'

from user code:
   File "test_exc.py", line N, in fn001
    fn002(x)
  File "test_exc.py", line N, in fn002
    torch._dynamo.graph_break()""",
        )

    # 测试函数：测试内部错误并且在错误发生时抑制错误信息
    @torch._dynamo.config.patch(verbose=True, suppress_errors=True)
    @make_logging_test()
    @unittest.skipIf(IS_FBCODE, "stack trace slightly different in fbcode")
    def test_internal_error_suppress_errors(self, records):
        # 定义函数 fn001，其中抛出 AssertionError
        def fn001(x):
            def f(ctx):
                raise AssertionError

            comptime(f)

        # 编译 fn001 使用 eager 后端
        torch.compile(fn001, backend="eager")(torch.randn(1))

        # 获取日志记录中的特定记录
        record = self.getRecord(records, "WON'T CONVERT")

        # 断言实际输出与期望输出一致
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
========== TorchDynamo Stack Trace ==========
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise AssertionError
AssertionError:

from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)


========== The above exception occurred while processing the following code ==========

  File "test_exc.py", line N, in test_internal_error_suppress_errors
    torch.compile(fn001, backend="eager")(torch.randn(1))
  File "test_exc.py", line N, in fn001
    comptime(f)

==========""",
        )

    # 测试函数：测试 NotImplementedError 的处理
    @make_logging_test()
    def test_not_implemented_error(self, records):
        # 定义函数 fn001，其中抛出 NotImplementedError
        def fn001(x):
            def f(ctx):
                raise NotImplementedError

            # 确保无法发生图断裂
            for i in range(3):
                comptime(f)

        # 编译 fn001 使用 eager 后端
        torch.compile(fn001, backend="eager")(torch.randn(1))

        # 获取日志记录中的特定记录
        record = self.getRecord(records, "WON'T CONVERT")

        # 断言实际输出与期望输出一致
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
due to:
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise NotImplementedError
torch._dynamo.exc.InternalTorchDynamoError:
"""
        )
# 引入用户代码
from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
        )

# 定义测试类方法，用于测试不支持的错误情况
@torch._dynamo.config.patch(inject_BUILD_SET_unimplemented_TESTING_ONLY=True)
@make_logging_test(dynamo=logging.DEBUG)
def test_unsupported_error(self, records):
    # 定义函数 fn001，返回一个集合 {1, 2}
    def fn001(x):
        return {1, 2}

    # 使用 torch.compile 编译 fn001 函数，使用 eager 后端
    torch.compile(fn001, backend="eager")(torch.randn(1))

    # TODO: 没有图断点日志！因为图断点日志不在集中位置；不支持指令绕过它
    self.getRecord(records, "Graph break:")

# 定义测试类方法，测试内部错误不抑制的情况
@torch._dynamo.config.patch(suppress_errors=False)
def test_internal_error_no_suppress(self):
    # 定义函数 fn001，注意：在此情况下，避免使用装饰器，因为3.11改变了所归属的行号
    def fn001(x):
        # 定义函数 f，抛出 AssertionError
        def f(ctx):
            raise AssertionError

        # 调用 comptime 函数处理 f 函数
        comptime(f)

    # NB: 这里截断用户代码是可以的，因为常规异常回溯包含剩余的信息片段
    self.assertExpectedInlineMunged(
        AssertionError,
        lambda: torch.compile(fn001, backend="eager")(torch.randn(1)),
        """\
from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
    )

# 定义测试类方法，测试图断点日志记录的情况
@make_logging_test(graph_breaks=True)
def test_graph_break_log(self, records):
    # 定义函数 fn002，对输入 x 执行加法，然后调用 torch._dynamo.graph_break()
    def fn002(x):
        x = x + 1
        torch._dynamo.graph_break()
        x = x + 1
        return x

    # 定义函数 fn001，调用 fn002 处理输入 x
    def fn001(x):
        return fn002(x)

    # 使用 torch.compile 编译 fn001 函数，使用 eager 后端
    torch.compile(fn001, backend="eager")(torch.randn(1))

    # 获取记录中包含 "Graph break:" 的日志记录
    record = self.getRecord(records, "Graph break:")

    # TODO: 这也应报告封闭帧；需要将帧对象传递给它
    self.assertExpectedInline(
        munge_exc(record.getMessage()),
        """\
Graph break: from user code at:
  File "test_exc.py", line N, in fn001
    return fn002(x)
  File "test_exc.py", line N, in fn002
    torch._dynamo.graph_break()
""",  # noqa: B950
    )

# 定义测试类方法，测试后端编译器失败的情况
@torch._dynamo.config.patch(suppress_errors=False)
def test_backend_suppress_line(self):
    # 定义函数 fn001，对输入 x 执行 torch.relu 操作，然后加 1
    def fn001(x):
        x = torch.relu(x)
        return x + 1

    # 测试不让此操作归因于 x + 1
    self.assertExpectedInlineMunged(
        torch._dynamo.exc.BackendCompilerFailed,
        lambda: torch.compile(fn001, backend="relu_compile_error_TESTING_ONLY")(
            torch.randn(1)
        ),
        """\
backend='relu_compile_error_TESTING_ONLY' raised:
ReluCompileError:""",
    )

# 跳过测试，如果未安装 z3
@skipIf(not TEST_Z3, "z3 not installed")
@torch._dynamo.config.patch(
    assume_static_by_default=False,
    suppress_errors=False,
)
@torch.fx.experimental._config.patch(
    inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
    translation_validation=True,
    translation_validation_no_bisect=True,
)
    # 定义一个测试方法，用于测试错误触发情况
    def test_trigger_on_error(self):
        # 导入需要的异常类 ValidationException
        from torch.fx.experimental.validator import ValidationException
        
        # 定义一个使用 torch.compile 装饰器的函数 fn，该函数接受 x 和 shape 两个参数，返回 x 按 shape 切分后的结果
        @torch.compile
        def fn(x, shape):
            return x.split(shape)
        
        # 断言期望的内联修改抛出 ValidationException 异常
        self.assertExpectedInlineMunged(
            ValidationException,
            # 使用 lambda 函数调用 fn，传入 torch.randn(20) 和 (5, 10, 5) 作为参数
            lambda: fn(torch.randn(20), (5, 10, 5)),
            """\
# 定义一个测试类，用于测试特定功能的正确性
class TestTranslationValidation:

    # 装饰器，标记测试函数，当z3库可用时运行，否则跳过
    @skipIf(not TEST_Z3, "z3 not installed")
    # 配置torch._dynamo.config，设置默认假设为非静态，允许错误抛出
    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        suppress_errors=False,
    )
    # 配置torch.fx.experimental._config，设置特定的测试选项
    @torch.fx.experimental._config.patch(
        inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
        translation_validation=True,
    )
    # 定义测试函数test_trigger_bisect_on_error
    def test_trigger_bisect_on_error(self):
        # 导入BisectValidationException异常类
        from torch.fx.experimental.validator import BisectValidationException
        
        # 编译函数fn，实现将x按照给定shape进行分割
        @torch.compile
        def fn(x, shape):
            return x.split(shape)

        # 使用self.assertExpectedInlineMunged断言，验证是否触发BisectValidationException异常
        self.assertExpectedInlineMunged(
            BisectValidationException,
            # 调用fn函数，传入torch.randn(20)和(5, 10, 5)作为参数
            lambda: fn(torch.randn(20), (5, 10, 5)),
            """\
translation validation failed when evaluating: Eq(s1 + s2 + s3, s0)

Failure occurred while running node:
    %split : [num_users=3] = call_method[target=split](args = (%l_x_, (%l_shape_0_, %l_shape_1_, %l_shape_2_)), kwargs = {})

Model:
  ==> L['shape'][0]: 1
  ==> L['shape'][1]: 1
  ==> L['shape'][2]: 0
  ==> L['x'].size()[0]: 3
  ==> L['x'].storage_offset(): 0
  ==> L['x'].stride()[0]: 1
  ==> s0: 3
  ==> s1: 1
  ==> s2: 1
  ==> s3: 0

Assertions:
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s1)
  ==> (== L['shape'][1] s2)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s0)
  ==> (> s0 1)

Target Expressions:
  ==> (!= (+ s1 s2 s3) s0)
  ==> (<= 0 s1)
  ==> (<= 0 s2)
  ==> (<= 0 s3)
  ==> (<= 2 s0)
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s1)
  ==> (== L['shape'][1] s2)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s0)
  ==> (> s0 0)

Failed Source Expressions:
  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])""",
        )

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
```