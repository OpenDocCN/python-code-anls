# `.\pytorch\test\dynamo\test_logging.py`

```
# 引入上下文管理模块
import contextlib
# 提供对函数进行装饰的工具模块
import functools
# 日志记录模块
import logging
# 系统操作模块
import os
# 单元测试的模拟对象
import unittest.mock

# 引入 PyTorch 库
import torch
# 引入 PyTorch Dynamo 的测试框架
import torch._dynamo.test_case
# 引入 PyTorch Dynamo 的测试工具
import torch._dynamo.testing
# 引入 PyTorch 分布式模块
import torch.distributed as dist
# 引入 PyTorch 中的分布式数据并行模块
from torch.nn.parallel import DistributedDataParallel as DDP

# 引入 PyTorch 内部的测试工具函数
from torch.testing._internal.common_utils import (
    find_free_port,
    munge_exc,
    skipIfTorchDynamo,
)
# 引入 PyTorch 内部的 CUDA 状态检查
from torch.testing._internal.inductor_utils import HAS_CUDA
# 引入 PyTorch 内部的日志记录测试工具
from torch.testing._internal.logging_utils import (
    LoggingTestCase,
    make_logging_test,
    make_settings_test,
)

# 当 CUDA 可用时，标记测试依赖 CUDA
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
# 当分布式环境可用时，标记测试依赖分布式环境
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)

# 示例函数，对输入张量进行逐元素乘法和加法操作
def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output

# 模拟 Dynamo 错误函数，对输入张量进行尺寸不匹配的加法操作
def dynamo_error_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output

# 模拟 Inductor 错误函数，对输入张量进行四舍五入操作
def inductor_error_fn(a):
    output = torch.round(a)
    return output

# 模拟 Inductor 调度函数，对输入张量在 CUDA 设备上进行加法操作
def inductor_schedule_fn(a):
    output = a.add(torch.ones(1000, 1000, device="cuda"))
    return output

# 定义示例参数
ARGS = (torch.ones(1000, 1000, requires_grad=True),)

# 多记录测试函数生成器，根据记录数生成对应的日志记录测试函数
def multi_record_test(num_records, **kwargs):
    @make_logging_test(**kwargs)
    def fn(self, records):
        # 使用 Inductor 优化器对 example_fn 进行优化
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        # 断言记录数与期望记录数相等
        self.assertEqual(len(records), num_records)

    return fn

# 记录数在指定范围内的测试函数生成器
def within_range_record_test(num_records_lower, num_records_higher, **kwargs):
    @make_logging_test(**kwargs)
    def fn(self, records):
        # 使用 Inductor 优化器对 example_fn 进行优化
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        # 断言记录数在指定范围内
        self.assertGreaterEqual(len(records), num_records_lower)
        self.assertLessEqual(len(records), num_records_higher)

    return fn

# 单记录测试函数生成器，记录数为 1
def single_record_test(**kwargs):
    return multi_record_test(1, **kwargs)

# 日志记录测试类，继承自 LoggingTestCase
class LoggingTests(LoggingTestCase):
    # 多记录测试示例：字节码记录数为 2
    test_bytecode = multi_record_test(2, bytecode=True)
    # 多记录测试示例：输出代码记录数为 2
    test_output_code = multi_record_test(2, output_code=True)
    # 多记录测试示例：AOT 图记录数为 3
    test_aot_graphs = multi_record_test(3, aot_graphs=True)

    # CUDA 环境下的调度测试
    @requires_cuda
    @make_logging_test(schedule=True)
    def test_schedule(self, records):
        # 使用 Inductor 优化器对 inductor_schedule_fn 进行优化
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        # 断言记录数大于 0 且小于 5
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 5)

    # CUDA 环境下的融合测试
    @requires_cuda
    @make_logging_test(fusion=True)
    def test_fusion(self, records):
        # 使用 Inductor 优化器对 inductor_schedule_fn 进行优化
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        # 断言记录数大于 0 且小于 8
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 8)

    # CUDA 环境下的 CUDA 图测试
    @requires_cuda
    @make_logging_test(cudagraphs=True)
    # 定义测试方法，验证 cudagraphs 功能
    def test_cudagraphs(self, records):
        # 调用 torch.compile 方法，使用 reduce-overhead 模式编译 inductor_schedule_fn 函数
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)
        # 执行编译后的函数，传入 GPU 上的全 1 张量
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        # 断言 records 的长度大于 0
        self.assertGreater(len(records), 0)
        # 断言 records 的长度小于 8
        self.assertLess(len(records), 8)

    # 使用装饰器 make_logging_test 创建记录日志的测试方法，允许重新编译
    @make_logging_test(recompiles=True)
    def test_recompiles(self, records):
        # 定义一个函数 fn，使用 torch.add 方法对输入的两个张量进行相加操作
        def fn(x, y):
            return torch.add(x, y)

        # 使用 torch._dynamo.optimize 方法，优化 fn 函数的 inductor 实现
        fn_opt = torch._dynamo.optimize("inductor")(fn)
        # 对两个全 1 张量调用优化后的函数
        fn_opt(torch.ones(1000, 1000), torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000), 1)
        # 断言 records 的长度大于 0
        self.assertGreater(len(records), 0)

    # 定义记录日志的测试方法，测试 dynamo 调试日志级别为 DEBUG
    test_dynamo_debug = within_range_record_test(30, 90, dynamo=logging.DEBUG)
    # 定义记录日志的测试方法，测试 dynamo 日志级别为 INFO
    test_dynamo_info = within_range_record_test(2, 10, dynamo=logging.INFO)

    # 使用 skipIfTorchDynamo 装饰器，如果 torch dynamo 太慢，则跳过执行
    @skipIfTorchDynamo("too slow")
    # 使用 make_logging_test 创建记录日志的测试方法，测试 dynamo 调试日志级别为 DEBUG
    @make_logging_test(dynamo=logging.DEBUG)
    def test_dynamo_debug_default_off_artifacts(self, records):
        # 使用 torch._dynamo.optimize 方法，优化 example_fn 函数的 inductor 实现
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        # 调用优化后的函数，传入全 1 张量
        fn_opt(torch.ones(1000, 1000))
        # 断言 records 中不包含名称为 ".__bytecode" 的记录
        self.assertEqual(len([r for r in records if ".__bytecode" in r.name]), 0)
        # 断言 records 中不包含名称为 ".__output_code" 的记录
        self.assertEqual(len([r for r in records if ".__output_code" in r.name]), 0)

    # 使用 make_logging_test 创建记录日志的测试方法
    @make_logging_test()
    def test_dynamo_error(self, records):
        try:
            # 使用 torch._dynamo.optimize 方法，优化 dynamo_error_fn 函数的 inductor 实现
            fn_opt = torch._dynamo.optimize("inductor")(dynamo_error_fn)
            # 调用优化后的函数，传入 ARGS 参数
            fn_opt(*ARGS)
        except Exception:
            pass
        # 从 records 中获取名称包含 "WON'T CONVERT" 的记录
        record = self.getRecord(records, "WON'T CONVERT")
        # 断言 record 的消息与预期相符，使用 munge_exc 函数处理异常消息
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
# 定义一个测试函数，使用装饰器 make_logging_test 包装，将测试结果记录到 records 中
def test_inductor_error(self, records):
    # 为异常处理创建一个上下文管理器
    exitstack = contextlib.ExitStack()
    # 导入 torch._inductor.lowering 模块，用于底层操作
    import torch._inductor.lowering

    # 定义一个函数 throw，抛出 AssertionError
    def throw(x):
        raise AssertionError

    # 在 lowerings 字典中注入一个错误，用于测试目的
    dict_entries = {}
    for x in list(torch._inductor.lowering.lowerings.keys()):
        if "round" in x.__name__:
            dict_entries[x] = throw

    # 使用 unittest.mock.patch.dict 上下文，临时替换 lowerings 字典
    exitstack.enter_context(
        unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries)
    )

    try:
        # 使用 torch._dynamo.optimize("inductor") 优化器，处理 inductor_error_fn 函数
        fn_opt = torch._dynamo.optimize("inductor")(inductor_error_fn)
        fn_opt(*ARGS)
    except Exception:
        pass

    # 获取名为 "WON'T CONVERT" 的记录，并验证其内容与预期是否一致
    record = self.getRecord(records, "WON'T CONVERT")
    self.assertExpectedInline(
        munge_exc(record.getMessage()),
        """\
WON'T CONVERT inductor_error_fn test_logging.py line N
due to:
Traceback (most recent call last):
  File "test_logging.py", line N, in throw
    raise AssertionError
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
LoweringException: AssertionError:
  target: aten.round.default
  args[0]: TensorBox(StorageBox(
    InputBuffer(name='primals_1', layout=FixedLayout('cpu', torch.float32, size=[1000, 1000], stride=[1000, 1]))
  ))""",
    )

    # 关闭上下文管理器
    exitstack.close()
    # 定义一个测试方法，用于测试分布式数据并行图形功能
    def test_ddp_graphs(self, records):
        # 定义一个简单的神经网络模型类
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                return self.layers(x)

        # 设置环境变量，指定主机地址为本地localhost
        os.environ["MASTER_ADDR"] = "localhost"
        # 找到一个可用的空闲端口并将其设置为主机端口
        os.environ["MASTER_PORT"] = str(find_free_port())
        # 初始化进程组，使用gloo后端，rank为0，总进程数为1
        dist.init_process_group("gloo", rank=0, world_size=1)

        # 对ToyModel应用分布式数据并行优化，指定GPU设备为cuda:0，使用单个设备ID为0，每个桶容量为4MB
        ddp_model = torch._dynamo.optimize("inductor")(
            DDP(ToyModel().to("cuda:0"), device_ids=[0], bucket_cap_mb=4)
        )

        # 对ddp_model进行前向传播，传入一个随机生成的张量数据，设备为cuda:0
        ddp_model(torch.randn(1024, 1024, device="cuda:0"))

        # 销毁进程组，释放资源
        dist.destroy_process_group()
        # 断言记录中包含特定字符串"__ddp_graphs"的数量为4
        self.assertEqual(len([r for r in records if "__ddp_graphs" in r.name]), 4)

    # 检查向已注册父记录器的子日志记录时，不会重复注册导致重复记录
    @make_settings_test("torch._dynamo.output_graph")
    def test_open_registration_with_registered_parent(self, records):
        # 获取名为"torch._dynamo.output_graph"的记录器实例
        logger = logging.getLogger("torch._dynamo.output_graph")
        # 记录一条信息日志"hi"
        logger.info("hi")
        # 断言记录数为1
        self.assertEqual(len(records), 1)

    # 检查向非注册父记录器的随机日志记录时，会注册并正确设置处理程序
    @make_settings_test("torch.utils")
    def test_open_registration(self, records):
        # 获取名为"torch.utils"的记录器实例
        logger = logging.getLogger("torch.utils")
        # 记录一条信息日志"hi"
        logger.info("hi")
        # 断言记录数为1
        self.assertEqual(len(records), 1)

    # 检查通过Python API向非注册父记录器的随机日志记录时，会注册并设置正确的处理程序
    @make_logging_test(modules={"torch.utils": logging.INFO})
    def test_open_registration_python_api(self, records):
        # 获取名为"torch.utils"的记录器实例
        logger = logging.getLogger("torch.utils")
        # 记录一条信息日志"hi"
        logger.info("hi")
        # 断言记录数为1
        self.assertEqual(len(records), 1)

    # 使用所有日志级别为DEBUG，dynamo模块日志级别为INFO的设置进行日志记录测试
    # 定义一个测试方法，用于测试所有的日志记录器是否符合预期设置
    def test_all(self, _):
        # 获取 Torch 内部的日志记录注册表
        registry = torch._logging._internal.log_registry

        # 获取与 "dynamo" 相关的日志别名到日志全名的映射列表
        dynamo_qnames = registry.log_alias_to_log_qnames["dynamo"]

        # 遍历所有日志记录器的全名
        for logger_qname in torch._logging._internal.log_registry.get_log_qnames():
            # 根据日志全名获取对应的日志记录器
            logger = logging.getLogger(logger_qname)

            # 如果日志全名以 dynamo_qnames 中的任何一项开头
            if any(logger_qname.find(d) == 0 for d in dynamo_qnames):
                # 断言该日志记录器的有效日志级别为 INFO
                self.assertEqual(
                    logger.getEffectiveLevel(),
                    logging.INFO,
                    msg=f"expected {logger_qname} is INFO, got {logging.getLevelName(logger.getEffectiveLevel())}",
                )
            else:
                # 断言该日志记录器的有效日志级别为 DEBUG
                self.assertEqual(
                    logger.getEffectiveLevel(),
                    logging.DEBUG,
                    msg=f"expected {logger_qname} is DEBUG, got {logging.getLevelName(logger.getEffectiveLevel())}",
                )

    # 标记为日志测试，并且设置图断点为 True
    @make_logging_test(graph_breaks=True)
    def test_graph_breaks(self, records):
        # 定义一个简单的函数 fn，并标记为 Torch Dynamo 的优化函数
        @torch._dynamo.optimize("inductor")
        def fn(x):
            # 在函数中触发图断点操作
            torch._dynamo.graph_break()
            return x + 1

        # 调用函数 fn
        fn(torch.ones(1))

        # 断言记录的数量为 1
        self.assertEqual(len(records), 1)

    # 标记为设置测试，并设置测试的目标为 "torch._dynamo.utils"
    @make_settings_test("torch._dynamo.utils")
    def test_dump_compile_times(self, records):
        # 使用 Torch Dynamo 优化装饰器标记函数 example_fn，并获取优化后的版本
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(torch.ones(1000, 1000))

        # 在退出时通过 atexit.register 运行该函数
        # 这里不会真正运行 atexit._run_exit_funcs()，以避免破坏其他测试的状态
        torch._dynamo.utils.dump_compile_times()

        # 断言记录中包含一条包含 "TorchDynamo compilation metrics" 的消息
        self.assertEqual(
            len(
                [r for r in records if "TorchDynamo compilation metrics" in str(r.msg)]
            ),
            1,
        )

    # 标记为日志测试，并设置 Dynamo 的日志级别为 INFO
    @make_logging_test(dynamo=logging.INFO)
    def test_custom_format_exc(self, records):
        # 获取名为 torch._dynamo 的日志记录器
        dynamo_log = logging.getLogger(torch._dynamo.__name__)

        try:
            # 抛出一个 RuntimeError 异常
            raise RuntimeError("foo")
        except RuntimeError:
            # 在捕获到异常时，记录异常信息到 dynamo_log 中
            dynamo_log.exception("test dynamo")
            dynamo_log.info("with exc", exc_info=True)

        # 记录当前调用栈信息到 dynamo_log 中
        dynamo_log.info("with stack", stack_info=True)

        # 断言记录的数量为 3
        self.assertEqual(len(records), 3)

        # 遍历 dynamo_log 的所有处理器，查找 Torch 的内部处理器
        for handler in dynamo_log.handlers:
            if torch._logging._internal._is_torch_handler(handler):
                break

        # 断言处理器存在
        self.assertIsNotNone(handler)

        # 断言记录的格式中包含 "Traceback"
        self.assertIn("Traceback", handler.format(records[0]))
        self.assertIn("Traceback", handler.format(records[1]))
        self.assertIn("Stack", handler.format(records[2]))

    # 标记为日志测试，并设置 Dynamo 的日志级别为 INFO
    @make_logging_test(dynamo=logging.INFO)
    # 定义一个测试方法，用于测试自定义日志格式的记录
    def test_custom_format(self, records):
        # 获取名为 torch._dynamo 的日志记录器对象
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        # 获取名为 "custom_format_test_artifact" 的自定义格式日志记录器对象
        test_log = torch._logging.getArtifactLogger(
            torch._dynamo.__name__, "custom_format_test_artifact"
        )
        # 记录一条信息到 dynamo_log
        dynamo_log.info("test dynamo")
        # 记录一条信息到 test_log
        test_log.info("custom format")
        # 断言记录数量为2
        self.assertEqual(len(records), 2)
        # 遍历 dynamo_log 的所有处理器
        for handler in dynamo_log.handlers:
            # 如果处理器是 Torch 内部处理器，则停止
            if torch._logging._internal._is_torch_handler(handler):
                break
        # 断言处理器不为 None
        self.assertIsNotNone(handler)
        # 断言记录的第一个元素格式化后包含"I"
        self.assertIn("I", handler.format(records[0]))
        # 断言记录的第二个元素格式化后等于"custom format"
        self.assertEqual("custom format", handler.format(records[1]))

    # 使用装饰器定义一个测试方法，测试多行日志格式记录
    @make_logging_test(dynamo=logging.INFO)
    def test_multiline_format(self, records):
        # 获取名为 torch._dynamo 的日志记录器对象
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        # 记录一条多行信息到 dynamo_log
        dynamo_log.info("test\ndynamo")
        # 使用格式字符串记录一条多行信息到 dynamo_log
        dynamo_log.info("%s", "test\ndynamo")
        # 使用格式字符串记录一条多行信息到 dynamo_log
        dynamo_log.info("test\n%s", "test\ndynamo")
        # 断言记录数量为3
        self.assertEqual(len(records), 3)
        # 遍历 dynamo_log 的所有处理器
        for handler in dynamo_log.handlers:
            # 如果处理器是 Torch 内部处理器，则停止
            if torch._logging._internal._is_torch_handler(handler):
                break
        # 断言处理器不为 None
        self.assertIsNotNone(handler)
        # 遍历每条记录，逐行格式化并断言包含"I"
        for record in records:
            r = handler.format(record)
            for l in r.splitlines():
                self.assertIn("I", l)

    # 使用装饰器定义一个测试方法，测试带有跟踪源信息的条件语句日志记录
    test_trace_source_simple = within_range_record_test(1, 100, trace_source=True)

    # 使用装饰器定义一个测试方法，测试带有跟踪源信息的条件语句日志记录
    @make_logging_test(trace_source=True)
    def test_trace_source_if_stmt(self, records):
        # 定义一个简单的函数 fn
        def fn(x):
            # 如果 x 的和大于0，返回 x 的两倍
            if x.sum() > 0:
                return x * 2
            # 否则返回 x 的三倍
            return x * 3

        # 优化函数 fn，并执行优化后的函数
        fn_opt = torch._dynamo.optimize("eager")(fn)
        fn_opt(torch.ones(3, 3))

        # 初始化找到特定信息的标记变量
        found_x2 = False
        found_x3 = False
        # 遍历记录集中的每条记录
        for record in records:
            # 获取记录消息
            msg = record.getMessage()
            # 如果消息中包含 "return x * 2"，将 found_x2 置为 True
            if "return x * 2" in msg:
                found_x2 = True
            # 如果消息中包含 "return x * 3"，将 found_x3 置为 True
            if "return x * 3" in msg:
                found_x3 = True

        # 断言找到 "return x * 2" 的记录
        self.assertTrue(found_x2)
        # 断言未找到 "return x * 3" 的记录
        self.assertFalse(found_x3)
    # 定义测试函数，用于验证源代码跟踪功能嵌套情况下的行为
    def test_trace_source_nested(self, records):
        # 定义内部函数 fn1，执行 fn2 后返回 x 的两倍
        def fn1(x):
            x = fn2(x)
            return x * 2

        # 定义内部函数 fn2，执行 fn3 后返回 x 的三倍
        def fn2(x):
            x = fn3(x)
            return x * 3

        # 定义内部函数 fn3，返回 x 的四倍
        def fn3(x):
            return x * 4

        # 对 fn1 进行优化，使其支持“急切”执行，并调用 fn1
        fn_opt = torch._dynamo.optimize("eager")(fn1)
        fn_opt(torch.ones(3, 3))

        # 初始化变量，用于检测不同返回值的记录是否存在
        found_x2 = False
        found_x3 = False
        found_x4 = False

        # 遍历记录列表中的每条记录
        for record in records:
            # 获取记录消息内容
            msg = record.getMessage()
            # 检查消息中是否包含 "return x * 2"
            if "return x * 2" in msg:
                found_x2 = True
                # 断言消息中不包含 "inline depth"
                self.assertNotIn("inline depth", msg)
            # 检查消息中是否包含 "return x * 3"
            elif "return x * 3" in msg:
                found_x3 = True
                # 断言消息中包含 "inline depth: 1"
                self.assertIn("inline depth: 1", msg)
            # 检查消息中是否包含 "return x * 4"
            elif "return x * 4" in msg:
                found_x4 = True
                # 断言消息中包含 "inline depth: 2"
                self.assertIn("inline depth: 2", msg)

        # 断言找到了每个预期的返回值记录
        self.assertTrue(found_x2)
        self.assertTrue(found_x3)
        self.assertTrue(found_x4)

    # 使用装饰器进行日志记录测试，验证条件分支函数中的源代码跟踪功能
    @make_logging_test(trace_source=True)
    def test_trace_source_cond(self, records):
        # 导入条件分支模块
        from functorch.experimental.control_flow import cond

        # 定义返回 x 的两倍的函数
        def true_fn(x):
            return x * 2

        # 定义返回 x 的三倍的函数
        def false_fn(x):
            return x * 3

        # 定义内部函数 inner，根据预测值选择 true_fn 或 false_fn
        def inner(pred, x):
            return cond(pred, true_fn, false_fn, [x])

        # 定义外部函数 outer，调用 inner 函数
        def outer(pred, x):
            return inner(pred, x)

        # 对 outer 函数进行优化，使其支持“急切”执行，并调用 outer
        fn_opt = torch._dynamo.optimize("eager")(outer)
        fn_opt(torch.tensor(True), torch.ones(3, 3))

        # 初始化变量，用于检测不同返回值的记录是否存在
        found_x2 = False
        found_x3 = False

        # 遍历记录列表中的每条记录
        for record in records:
            # 获取记录消息内容
            msg = record.getMessage()
            # 检查消息中是否包含 "return x * 2" 和 "inline depth: 3"
            if "return x * 2" in msg:
                found_x2 = True
                self.assertIn("inline depth: 3", msg)
            # 检查消息中是否包含 "return x * 3" 和 "inline depth: 3"
            if "return x * 3" in msg:
                found_x3 = True
                self.assertIn("inline depth: 3", msg)

        # 断言找到了每个预期的返回值记录
        self.assertTrue(found_x2)
        self.assertTrue(found_x3)

    # 使用装饰器进行日志记录测试，验证函数名称的源代码跟踪功能
    @make_logging_test(trace_source=True)
    def test_trace_source_funcname(self, records):
        # 注意：在版本 3.12 中，列表推导已经被内联，所以使用元组来测试
        # 定义返回生成器表达式结果的函数 fn1
        def fn1():
            # 定义返回 torch.ones(3, 3) 生成器表达式结果的函数 fn2
            def fn2():
                # 如果条件为真，返回包含 5 个 torch.ones(3, 3) 的元组
                if True:
                    return tuple(torch.ones(3, 3) for _ in range(5))
                return None

            # 调用并返回 fn2 函数的结果
            return fn2()

        # 对 fn1 函数进行优化，使其支持“急切”执行，并调用 fn1
        fn_opt = torch._dynamo.optimize("eager")(fn1)
        fn_opt()

        # 初始化变量，用于检测是否找到了预期的函数名称记录
        found_funcname = False

        # 遍历记录列表中的每条记录
        for record in records:
            # 获取记录消息内容
            msg = record.getMessage()
            # 检查消息中是否包含 "<genexpr>" 和 "fn1.fn2"
            if "<genexpr>" in msg and "fn1.fn2" in msg:
                found_funcname = True

        # 断言找到了预期的函数名称记录
        self.assertTrue(found_funcname)

    # 测试无效的日志记录标志
    def test_invalid_artifact_flag(self):
        # 使用断言，验证设置 aot_graphs=5 时是否引发 ValueError 异常
        with self.assertRaises(ValueError):
            torch._logging.set_logs(aot_graphs=5)

    # 使用分布式装饰器，验证分布式日志记录功能
    @requires_distributed()
    def test_distributed_rank_logging(self):
        # 复制当前环境变量，并设置 TORCH_LOGS="dynamo"
        env = dict(os.environ)
        env["TORCH_LOGS"] = "dynamo"
        # 运行进程并捕获标准输出和错误输出，验证日志记录是否正常运行
        stdout, stderr = self.run_process_no_exception(
            """
import torch.distributed as dist  # 导入 torch 分布式包
import logging  # 导入日志模块
from torch.testing._internal.distributed.fake_pg import FakeStore  # 从内部测试包中导入虚假存储类 FakeStore

# 创建一个 FakeStore 实例，用于模拟分布式环境中的存储
store = FakeStore()

# 初始化分布式进程组，使用假的通信后端 "fake"，当前进程的排名是 0，总进程数是 2，使用上面创建的 FakeStore 作为存储
dist.init_process_group("fake", rank=0, world_size=2, store=store)

# 获取名为 "torch._dynamo" 的日志记录器
dynamo_log = logging.getLogger("torch._dynamo")

# 在 dynamo_log 中记录一条信息级别为 info 的日志，内容为 "woof"
dynamo_log.info("woof")

# 打印字符串 "arf" 到标准输出
print("arf")
    # 定义测试函数，验证调用图断点功能的效果，同时记录日志
    def test_trace_call_graph_break(self, records):
        # 定义内部函数fn，对输入的x执行乘以2的操作，然后调用torch._dynamo.graph_break()断点函数，最后返回乘以3后的结果
        def fn(x):
            x = x * 2
            torch._dynamo.graph_break()
            return x * 3

        # 对函数fn应用torch._dynamo.optimize("eager")优化，并调用优化后的函数fn_opt，传入torch.randn(3, 3)的随机数作为参数
        fn_opt = torch._dynamo.optimize("eager")(fn)
        fn_opt(torch.randn(3, 3))

        # 断言记录的日志长度为3
        self.assertEqual(len(records), 3)
        
        # 从记录中提取消息内容的最后两行，并将其存入messages列表
        messages = [
            "\n".join(record.getMessage().split("\n")[-2:]) for record in records
        ]
        
        # 使用self.assertExpectedInline断言消息列表中的第一个消息，检查是否包含特定格式的日志消息
        self.assertExpectedInline(
            messages[0],
            """\
            x = x * 2
                ~~^~~""",
        )
        
        # 使用self.assertExpectedInline断言消息列表中的最后一个消息，检查是否包含特定格式的日志消息
        self.assertExpectedInline(
            messages[-1],
            """\
            return x * 3
                   ~~^~~""",
        )

    # 使用make_logging_test装饰器定义测试函数，验证在开启守护和重新编译的情况下的功能
    @make_logging_test(guards=True, recompiles=True)
    def test_guards_recompiles(self, records):
        # 定义函数fn，接收x、ys和zs作为参数，调用内部函数inner并返回其结果
        def fn(x, ys, zs):
            return inner(x, ys, zs)

        # 定义内部函数inner，接收x、ys和zs作为参数，遍历ys和zs列表，计算每个元素的乘积并加到x上，最后返回x的值
        def inner(x, ys, zs):
            for y, z in zip(ys, zs):
                x += y * z
            return x

        # 定义ys和zs列表
        ys = [1.0, 2.0]
        zs = [3.0]
        x = torch.tensor([1.0])

        # 对函数fn应用torch._dynamo.optimize("eager")优化，并调用优化后的函数fn_opt，传入相应的参数
        fn_opt = torch._dynamo.optimize("eager")(fn)
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        # 将记录中的消息内容连接为一个字符串，存入record_str变量
        record_str = "\n".join(r.getMessage() for r in records)

        # 使用self.assertIn断言record_str中包含特定的子字符串
        self.assertIn(
            """\
# 检查列表 L 中 'zs' 子列表的第一个元素是否等于 3.0
L['zs'][0] == 3.0                                             # for y, z in zip(ys, zs):""",
            record_str,
        )
        self.assertIn(
            """\
    triggered by the following guard failure(s):\n\
    - len(L['ys']) == 2                                             # for y, z in zip(ys, zs):""",
            record_str,
        )

# 跳过测试条件：如果 Torch Dynamo 的性能太慢，则跳过测试
@skipIfTorchDynamo("too slow")
# 标记为日志测试，并使用 Torch 默认的日志设置
@make_logging_test(**torch._logging.DEFAULT_LOGGING)
def test_default_logging(self, records):
    def fn(a):
        if a.sum() < 0:
            a = torch.sin(a)
        else:
            a = torch.cos(a)
        print("hello")
        return a + 1

    # 优化函数 fn，并使用 eager 模式进行编译
    fn_opt = torch._dynamo.optimize("eager")(fn)
    # 调用优化后的函数 fn_opt，传入参数 torch.ones(10, 10)
    fn_opt(torch.ones(10, 10))
    # 再次调用优化后的函数 fn_opt，传入参数 -torch.ones(10, 5)
    fn_opt(-torch.ones(10, 5))

    # 断言记录中包含至少一个带有 ".__graph_breaks" 的条目
    self.assertGreater(len([r for r in records if ".__graph_breaks" in r.name]), 0)
    # 断言记录中包含至少一个带有 ".__recompiles" 的条目
    self.assertGreater(len([r for r in records if ".__recompiles" in r.name]), 0)
    # 断言记录中包含至少一个带有 ".symbolic_shapes" 的条目
    self.assertGreater(len([r for r in records if ".symbolic_shapes" in r.name]), 0)
    # 断言记录中包含至少一个带有 ".__guards" 的条目
    self.assertGreater(len([r for r in records if ".__guards" in r.name]), 0)
    # 断言记录中包含至少一个消息为 "return a + 1" 的条目
    self.assertGreater(
        len([r for r in records if "return a + 1" in r.getMessage()]), 0
    )

def test_logs_out(self):
    import tempfile

    # 使用临时文件进行测试
    with tempfile.NamedTemporaryFile() as tmp:
        env = dict(os.environ)
        env["TORCH_LOGS"] = "dynamo"
        env["TORCH_LOGS_OUT"] = tmp.name
        # 运行子进程，将代码块作为字符串导入，并设置环境变量
        stdout, stderr = self.run_process_no_exception(
            """\
import torch
@torch.compile(backend="eager")
def fn(a):
    return a.sum()

fn(torch.randn(5))
                """,
            env=env,
        )
        # 打开临时文件，并读取其内容
        with open(tmp.name) as fd:
            lines = fd.read()
            # 断言临时文件内容与 stderr 的 UTF-8 解码结果相等
            self.assertEqual(lines, stderr.decode("utf-8"))

# 单记录测试
# 排除不需要测试的日志记录类型
exclusions = {
    "bytecode",
    "cudagraphs",
    "output_code",
    "schedule",
    "fusion",
    "overlap",
    "aot_graphs",
    "post_grad_graphs",
    "compiled_autograd",
    "compiled_autograd_verbose",
    "recompiles",
    "recompiles_verbose",
    "graph_breaks",
    "graph",
    "graph_sizes",
    "ddp_graphs",
    "perf_hints",
    "not_implemented",
    "trace_source",
    "trace_call",
    "trace_bytecode",
    "custom_format_test_artifact",
    "onnx",
    "onnx_diagnostics",
    "guards",
    "verbose_guards",
    "sym_node",
    "export",
}
# 对于所有 Torch 内部日志注册的名称，如果不在排除列表中，则添加为测试方法
for name in torch._logging._internal.log_registry.artifact_names:
    if name not in exclusions:
        setattr(LoggingTests, f"test_{name}", single_record_test(**{name: True}))

# 如果是主程序入口
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # 运行测试
    run_tests()
```