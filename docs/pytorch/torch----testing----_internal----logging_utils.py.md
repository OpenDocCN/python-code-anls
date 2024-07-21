# `.\pytorch\torch\testing\_internal\logging_utils.py`

```py
# mypy: ignore-errors

# 导入需要的模块
import torch._dynamo.test_case  # 导入测试框架相关模块
import unittest.mock  # 导入 unittest 模块的 mock 功能
import os  # 导入操作系统相关功能模块
import contextlib  # 导入上下文管理模块
import torch._logging  # 导入 PyTorch 内部日志模块
import torch._logging._internal  # 导入 PyTorch 内部日志的内部模块
from torch._dynamo.utils import LazyString  # 从 PyTorch 的动态模块导入 LazyString 工具类
from torch._inductor import config as inductor_config  # 从 PyTorch 的感应器模块导入配置
import logging  # 导入标准日志模块
import io  # 导入 io 模块

@contextlib.contextmanager
def preserve_log_state():
    # 保存当前日志状态
    prev_state = torch._logging._internal._get_log_state()
    # 设置空的日志状态
    torch._logging._internal._set_log_state(torch._logging._internal.LogState())
    try:
        yield
    finally:
        # 恢复之前保存的日志状态
        torch._logging._internal._set_log_state(prev_state)
        # 重新初始化日志
        torch._logging._internal._init_logs()

def log_settings(settings):
    exit_stack = contextlib.ExitStack()
    # 在上下文中设置环境变量 TORCH_LOGS，使用给定的 settings 参数
    settings_patch = unittest.mock.patch.dict(os.environ, {"TORCH_LOGS": settings})
    exit_stack.enter_context(preserve_log_state())
    exit_stack.enter_context(settings_patch)
    # 初始化日志设置
    torch._logging._internal._init_logs()
    return exit_stack

def log_api(**kwargs):
    exit_stack = contextlib.ExitStack()
    exit_stack.enter_context(preserve_log_state())
    # 设置日志
    torch._logging.set_logs(**kwargs)
    return exit_stack

def kwargs_to_settings(**kwargs):
    INT_TO_VERBOSITY = {10: "+", 20: "", 40: "-"}

    settings = []

    def append_setting(name, level):
        # 根据参数的类型和值，追加设置到 settings 列表中
        if isinstance(name, str) and isinstance(level, int) and level in INT_TO_VERBOSITY:
            settings.append(INT_TO_VERBOSITY[level] + name)
            return
        else:
            raise ValueError("Invalid value for setting")

    for name, val in kwargs.items():
        if isinstance(val, bool):
            settings.append(name)
        elif isinstance(val, int):
            append_setting(name, val)
        elif isinstance(val, dict) and name == "modules":
            for module_qname, level in val.items():
                append_setting(module_qname, level)
        else:
            raise ValueError("Invalid value for setting")

    # 将 settings 列表转换为逗号分隔的字符串
    return ",".join(settings)


# Note on testing strategy:
# This class does two things:
# 1. Runs two versions of a test:
#    1a. patches the env var log settings to some specific value
#    1b. calls torch._logging.set_logs(..)
# 2. patches the emit method of each setup handler to gather records
# that are emitted to each console stream
# 3. passes a ref to the gathered records to each test case for checking
#
# The goal of this testing in general is to ensure that given some settings env var
# that the logs are setup correctly and capturing the correct records.
def make_logging_test(**kwargs):
    # 这个函数的目的是构建日志记录的测试用例，通过设置环境变量和调用函数来测试日志记录的正确性
    def wrapper(fn):
        # 定义装饰器函数 `wrapper`，接受一个函数 `fn` 作为参数

        @inductor_config.patch({"fx_graph_cache": False})
        # 使用 `inductor_config.patch` 装饰 `test_fn` 函数，设置 `fx_graph_cache` 为 False
        def test_fn(self):
            # 定义测试函数 `test_fn`，接受 `self` 参数

            torch._dynamo.reset()
            # 重置 Torch 的动态计算图

            records = []
            # 初始化空列表 `records`

            # run with env var
            # 如果没有传入关键字参数 `kwargs`
            if len(kwargs) == 0:
                # 使用环境变量运行
                with self._handler_watcher(records):
                    # 使用 `self._handler_watcher` 监控执行并记录日志到 `records`
                    fn(self, records)
                    # 执行传入的函数 `fn`，并传入 `self` 和 `records` 作为参数
            else:
                # 否则，如果传入了关键字参数 `kwargs`
                with log_settings(kwargs_to_settings(**kwargs)), self._handler_watcher(records):
                    # 使用传入的关键字参数设置日志配置，同时监控执行并记录日志到 `records`
                    fn(self, records)
                    # 执行传入的函数 `fn`，并传入 `self` 和 `records` 作为参数

            # run with API
            # 使用 API 运行
            torch._dynamo.reset()
            # 重置 Torch 的动态计算图

            records.clear()
            # 清空 `records` 列表

            with log_api(**kwargs), self._handler_watcher(records):
                # 使用 API 记录日志，并监控执行并记录日志到 `records`
                fn(self, records)
                # 执行传入的函数 `fn`，并传入 `self` 和 `records` 作为参数

        return test_fn
        # 返回定义好的 `test_fn` 函数作为装饰后的函数

    return wrapper
    # 返回定义好的 `wrapper` 装饰器函数
def make_settings_test(settings):
    # 创建一个装饰器函数，用于设置测试环境
    def wrapper(fn):
        # 定义一个测试函数，用于运行测试并记录日志
        def test_fn(self):
            # 重置 Torch 的内部状态
            torch._dynamo.reset()
            # 初始化记录日志的列表
            records = []
            # 运行测试函数，并设置环境变量
            with log_settings(settings), self._handler_watcher(records):
                fn(self, records)

        return test_fn

    return wrapper

class LoggingTestCase(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # 设置环境变量，用于测试日志记录
        cls._exit_stack.enter_context(
            unittest.mock.patch.dict(os.environ, {"___LOG_TESTING": ""})
        )
        # 修改配置，禁止输出错误信息
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.suppress_errors", True)
        )
        # 修改配置，设置日志输出为静默模式
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.verbose", False)
        )

    @classmethod
    def tearDownClass(cls):
        # 清理测试环境，在测试类所有测试方法执行完毕后调用
        cls._exit_stack.close()
        # 清空 Torch 的内部日志状态
        torch._logging._internal.log_state.clear()
        # 初始化 Torch 的日志配置
        torch._logging._init_logs()

    def hasRecord(self, records, m):
        # 判断记录列表中是否包含指定消息
        return any(m in r.getMessage() for r in records)

    def getRecord(self, records, m):
        record = None
        for r in records:
            # 查找包含指定消息的记录
            if m in r.getMessage():
                # 断言只有一个匹配的记录，否则抛出异常
                self.assertIsNone(
                    record,
                    msg=LazyString(
                        lambda: f"multiple matching records: {record} and {r} among {records}"
                    ),
                )
                record = r
        # 如果未找到匹配记录，则抛出失败异常
        if record is None:
            self.fail(f"did not find record with {m} among {records}")
        return record

    # 该方法通过修改每个处理器的 emit 方法来收集日志记录
    def _handler_watcher(self, record_list):
        exit_stack = contextlib.ExitStack()

        def emit_post_hook(record):
            nonlocal record_list
            record_list.append(record)

        # 遍历所有注册了处理器的日志记录器
        for log_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(log_qname)
            num_handlers = len(logger.handlers)
            # 断言每个日志记录器最多有两个处理器（用于调试和高于调试级别的消息）
            self.assertLessEqual(
                num_handlers,
                2,
                "All pt2 loggers should only have at most two handlers (debug artifacts and messages above debug level).",
            )
            # 断言每个日志记录器至少有一个处理器
            self.assertGreater(num_handlers, 0, "All pt2 loggers should have more than zero handlers")

            # 为每个处理器的 emit 方法应用新的 hook 函数
            for handler in logger.handlers:
                old_emit = handler.emit

                def new_emit(record):
                    old_emit(record)
                    emit_post_hook(record)

                exit_stack.enter_context(
                    unittest.mock.patch.object(handler, "emit", new_emit)
                )

        return exit_stack
    def logs_to_string(module, log_option):
        """
        logs_to_string函数用于捕获指定模块中特定日志选项的输出，并返回相应的日志流和上下文管理器。
    
        log_stream = io.StringIO()
        创建一个StringIO对象，用于存储捕获到的日志内容。
    
        handler = logging.StreamHandler(stream=log_stream)
        创建一个StreamHandler，将日志输出流定向到log_stream中。
    
        @contextlib.contextmanager
        def tmp_redirect_logs():
            定义一个上下文管理器tmp_redirect_logs，用于临时重定向日志输出。
            try:
                logger = torch._logging.getArtifactLogger(module, log_option)
                获取指定模块和日志选项的日志记录器。
                logger.addHandler(handler)
                将handler添加到logger中，实现日志输出的捕获。
                yield
                在yield之后恢复正常日志输出。
            finally:
                logger.removeHandler(handler)
                在finally块中移除handler，确保不再捕获日志输出。
    
        def ctx_manager():
            定义一个函数ctx_manager，返回一个上下文管理器对象。
            exit_stack = log_settings(log_option)
            调用log_settings函数，获取与日志选项相关的退出堆栈对象。
            exit_stack.enter_context(tmp_redirect_logs())
            在退出堆栈中进入tmp_redirect_logs()上下文管理器。
            return exit_stack
            返回退出堆栈对象。
    
        return log_stream, ctx_manager
        logs_to_string函数返回log_stream对象和ctx_manager函数，用于捕获和管理日志输出。
        ```
```