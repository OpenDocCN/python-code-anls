# `.\pytorch\torch\_dynamo\test_case.py`

```py
# mypy: allow-untyped-defs
# 引入上下文管理、模块导入、日志记录等必要模块
import contextlib
import importlib
import logging

# 导入 PyTorch 相关模块
import torch
import torch.testing
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    IS_WINDOWS,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase as TorchTestCase,
)

# 导入本地模块中的 config、reset、utils
from . import config, reset, utils

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)


def run_tests(needs=()):
    # 导入内部的 run_tests 函数
    from torch.testing._internal.common_utils import run_tests

    # 如果符合跳过测试的条件，则直接返回
    if TEST_WITH_TORCHDYNAMO or IS_WINDOWS or TEST_WITH_CROSSREF:
        return  # skip testing

    # 如果需要的模块是字符串，则转换为元组形式
    if isinstance(needs, str):
        needs = (needs,)

    # 遍历需要的模块
    for need in needs:
        # 如果需要 CUDA 但当前环境不支持 CUDA，则返回
        if need == "cuda" and not torch.cuda.is_available():
            return
        else:
            try:
                # 尝试导入需要的模块，如果导入失败则返回
                importlib.import_module(need)
            except ImportError:
                return

    # 运行测试
    run_tests()


class TestCase(TorchTestCase):
    _exit_stack: contextlib.ExitStack  # 定义上下文管理器 ExitStack 对象

    @classmethod
    def tearDownClass(cls):
        # 关闭 ExitStack 对象，执行父类的 tearDownClass 方法
        cls._exit_stack.close()
        super().tearDownClass()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # 初始化 ExitStack 对象，并进入上下文管理器
        cls._exit_stack = contextlib.ExitStack()  # type: ignore[attr-defined]
        cls._exit_stack.enter_context(  # type: ignore[attr-defined]
            config.patch(
                raise_on_ctx_manager_usage=True,
                suppress_errors=False,
                log_compilation_metrics=False,
            ),
        )

    def setUp(self):
        # 设置测试的前置条件，包括是否启用梯度计算、执行 reset 和清空计数器
        self._prior_is_grad_enabled = torch.is_grad_enabled()
        super().setUp()
        reset()
        utils.counters.clear()

    def tearDown(self):
        # 在测试结束后打印计数器内容，并进行 reset 和清空计数器
        for k, v in utils.counters.items():
            print(k, v.most_common())
        reset()
        utils.counters.clear()
        super().tearDown()
        # 如果测试过程中改变了梯度模式，则记录警告并恢复之前的梯度模式设置
        if self._prior_is_grad_enabled is not torch.is_grad_enabled():
            log.warning("Running test changed grad mode")
            torch.set_grad_enabled(self._prior_is_grad_enabled)
```