# `.\pytorch\test\jit\test_logging.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库
import os
import sys

import torch

# 将测试目录下的 helper 文件导入路径中，使其可被引用
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

# 如果当前脚本作为主程序执行，则抛出运行时错误提示
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestLogging，继承自 JitTestCase
class TestLogging(JitTestCase):

    # 定义测试方法 test_bump_numeric_counter
    def test_bump_numeric_counter(self):
        
        # 定义一个继承自 torch.jit.ScriptModule 的日志记录模块 ModuleThatLogs
        class ModuleThatLogs(torch.jit.ScriptModule):
            
            # 定义 forward 方法，使用 torch.jit.script_method 装饰，表示这是一个脚本方法
            @torch.jit.script_method
            def forward(self, x):
                
                # 对输入的张量 x 进行迭代，增加其每个元素的值
                for i in range(x.size(0)):
                    x += 1.0
                    # 记录名为 "foo" 的统计值，增加 1
                    torch.jit._logging.add_stat_value("foo", 1)

                # 如果 x 的所有元素之和大于 0，则记录名为 "positive" 的统计值增加 1，否则记录名为 "negative" 的统计值增加 1
                if bool(x.sum() > 0.0):
                    torch.jit._logging.add_stat_value("positive", 1)
                else:
                    torch.jit._logging.add_stat_value("negative", 1)
                    
                # 返回处理后的张量 x
                return x

        # 创建一个 LockingLogger 实例 logger，用于记录日志
        logger = torch.jit._logging.LockingLogger()
        # 保存当前的日志记录器 old_logger
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            # 创建 ModuleThatLogs 的实例 mtl
            mtl = ModuleThatLogs()
            # 对 mtl 进行 5 次迭代，每次输入一个随机张量
            for i in range(5):
                mtl(torch.rand(3, 4, 5))

            # 断言 "foo" 统计值的计数为 15
            self.assertEqual(logger.get_counter_val("foo"), 15)
            # 断言 "positive" 统计值的计数为 5
            self.assertEqual(logger.get_counter_val("positive"), 5)
        finally:
            # 恢复原来的日志记录器
            torch.jit._logging.set_logger(old_logger)

    # 定义测试方法 test_trace_numeric_counter
    def test_trace_numeric_counter(self):
        
        # 定义一个函数 foo，参数为 x
        def foo(x):
            # 记录名为 "foo" 的统计值，增加 1
            torch.jit._logging.add_stat_value("foo", 1)
            # 返回 x + 1.0
            return x + 1.0

        # 对函数 foo 进行追踪，得到 traced 对象
        traced = torch.jit.trace(foo, torch.rand(3, 4))
        # 创建一个 LockingLogger 实例 logger，用于记录日志
        logger = torch.jit._logging.LockingLogger()
        # 保存当前的日志记录器 old_logger
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            # 对 traced 对象进行调用，输入一个随机张量
            traced(torch.rand(3, 4))

            # 断言 "foo" 统计值的计数为 1
            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            # 恢复原来的日志记录器
            torch.jit._logging.set_logger(old_logger)

    # 定义测试方法 test_time_measurement_counter
    def test_time_measurement_counter(self):
        
        # 定义一个继承自 torch.jit.ScriptModule 的计时模块 ModuleThatTimes
        class ModuleThatTimes(torch.jit.ScriptModule):
            
            # 定义 forward 方法，接收输入张量 x
            def forward(self, x):
                # 记录开始时间点 tp_start
                tp_start = torch.jit._logging.time_point()
                # 对输入张量 x 进行 30 次加法操作
                for i in range(30):
                    x += 1.0
                # 记录结束时间点 tp_end
                tp_end = torch.jit._logging.time_point()
                # 记录名为 "mytimer" 的统计值，记录运行时间差
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                # 返回处理后的张量 x
                return x

        # 创建 ModuleThatTimes 的实例 mtm
        mtm = ModuleThatTimes()
        # 创建一个 LockingLogger 实例 logger，用于记录日志
        logger = torch.jit._logging.LockingLogger()
        # 保存当前的日志记录器 old_logger
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            # 对 mtm 进行调用，输入一个随机张量
            mtm(torch.rand(3, 4))
            # 断言 "mytimer" 统计值的计数大于 0
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            # 恢复原来的日志记录器
            torch.jit._logging.set_logger(old_logger)
    # 定义一个测试方法，用于测试时间测量计数脚本
    def test_time_measurement_counter_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的模块，用于计时
        class ModuleThatTimes(torch.jit.ScriptModule):
            # 定义前向传播方法，使用 Torch 脚本装饰器
            @torch.jit.script_method
            def forward(self, x):
                # 获取开始时间点
                tp_start = torch.jit._logging.time_point()
                # 循环迭代 30 次
                for i in range(30):
                    x += 1.0
                # 获取结束时间点
                tp_end = torch.jit._logging.time_point()
                # 记录时间差并存入名为 "mytimer" 的统计数据中
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                # 返回计算后的张量 x
                return x

        # 创建 ModuleThatTimes 类的实例
        mtm = ModuleThatTimes()
        # 创建一个锁定的日志记录器
        logger = torch.jit._logging.LockingLogger()
        # 获取当前的日志记录器并保存为旧日志记录器
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            # 调用模块 mtm 的 forward 方法，传入一个形状为 (3, 4) 的随机张量
            mtm(torch.rand(3, 4))
            # 断言 "mytimer" 统计计数值大于 0
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            # 恢复旧的日志记录器
            torch.jit._logging.set_logger(old_logger)

    # 定义一个测试方法，用于测试统计计数的聚合
    def test_counter_aggregation(self):
        # 定义一个简单的函数 foo，对输入张量进行操作
        def foo(x):
            # 循环迭代 3 次，每次将值 1 添加到名为 "foo" 的统计计数中
            for i in range(3):
                torch.jit._logging.add_stat_value("foo", 1)
            # 返回 x 加 1.0 后的结果
            return x + 1.0

        # 对函数 foo 进行跟踪，创建一个 Torch 脚本
        traced = torch.jit.trace(foo, torch.rand(3, 4))
        # 创建一个锁定的日志记录器
        logger = torch.jit._logging.LockingLogger()
        # 设置名为 "foo" 的统计计数聚合类型为平均值
        logger.set_aggregation_type("foo", torch.jit._logging.AggregationType.AVG)
        # 获取当前的日志记录器并保存为旧日志记录器
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            # 调用跟踪后的函数 traced，传入一个形状为 (3, 4) 的随机张量
            traced(torch.rand(3, 4))
            # 断言 "foo" 统计计数的值为 1
            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            # 恢复旧的日志记录器
            torch.jit._logging.set_logger(old_logger)

    # 定义一个测试方法，用于测试日志记录级别的设置
    def test_logging_levels_set(self):
        # 设置 Torch 的 JIT 日志记录选项为 "foo"
        torch._C._jit_set_logging_option("foo")
        # 断言当前的 JIT 日志记录选项为 "foo"
        self.assertEqual("foo", torch._C._jit_get_logging_option())
```