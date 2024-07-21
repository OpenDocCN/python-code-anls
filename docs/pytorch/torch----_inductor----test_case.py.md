# `.\pytorch\torch\_inductor\test_case.py`

```
# mypy: allow-untyped-defs
# 导入上下文管理模块
import contextlib
# 导入操作系统相关模块
import os

# 从特定的 Torch 模块导入测试相关函数和类
from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,  # 导入测试运行函数并重命名为 dynamo_run_tests
    TestCase as DynamoTestCase,      # 导入测试用例类并重命名为 DynamoTestCase
)

# 从 Torch 的 _inductor 子模块导入配置相关模块
from torch._inductor import config
# 从 Torch 的 _inductor.utils 导入 fresh_inductor_cache 函数
from torch._inductor.utils import fresh_inductor_cache


def run_tests(needs=()):
    # 调用 Dynamo 模块中的 run_tests 函数并传递参数
    dynamo_run_tests(needs)


class TestCase(DynamoTestCase):
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        # 创建一个上下文管理堆栈对象
        self._inductor_test_stack = contextlib.ExitStack()
        # 将 FX 图缓存设置为启用，并将其作为上下文管理对象的一部分
        self._inductor_test_stack.enter_context(config.patch({"fx_graph_cache": True}))
        # 检查环境变量，如果未禁用新缓存且未启用 Torch 编译调试，则清空新的感应器缓存
        if (
            os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1"
            and os.environ.get("TORCH_COMPILE_DEBUG") != "1"
        ):
            self._inductor_test_stack.enter_context(fresh_inductor_cache())

    def tearDown(self):
        super().tearDown()  # 调用父类的 tearDown 方法
        # 关闭上下文管理堆栈对象
        self._inductor_test_stack.close()
```