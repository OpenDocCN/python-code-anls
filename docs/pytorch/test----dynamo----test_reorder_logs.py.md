# `.\pytorch\test\dynamo\test_reorder_logs.py`

```py
# Owner(s): ["module: dynamo"]
# 导入所需的模块和库
import io
import warnings
from unittest.mock import patch

# 导入 torch 库及其私有模块
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
from torch._dynamo.utils import counters

# 定义一个测试类 ReorderLogsTests，继承于 torch._dynamo.test_case.TestCase
class ReorderLogsTests(torch._dynamo.test_case.TestCase):
    
    # 定义测试函数 test_dont_reorder_print
    def test_dont_reorder_print(self):
        # 定义一个内部函数 f，接受参数 x
        def f(x):
            # 对参数 x 进行操作：加法和乘法
            x = x + x
            # 打印字符串 "moo"
            print("moo")
            x = x * x
            return x

        # 清空计数器
        counters.clear()
        # 生成一个形状为 (3, 3) 的随机张量 x
        x = torch.randn(3, 3)
        # 使用 eager 模式编译函数 f
        opt_f = torch.compile(backend="eager")(f)
        
        # 使用 patch 临时替换 sys.stdout，以捕获打印输出
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            # 调用优化后的函数 opt_f，传入参数 x
            opt_out = opt_f(x)
            # 获取捕获的标准输出内容并去除首尾空白
            printed_output = mock_stdout.getvalue().strip()
            # 调用原始函数 f，传入参数 x，并保存结果
            orig_out = f(x)

        # 断言优化后的输出与原始输出相同
        self.assertTrue(same(orig_out, opt_out))
        # 断言打印输出内容为 "moo"
        self.assertEqual(printed_output, "moo")
        # 断言计数器中 "graph_break" 键对应的值为 1
        self.assertEqual(len(counters["graph_break"]), 1)

    # 使用 torch._dynamo.config.patch 装饰器，配置可重排序的打印函数为 print
    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    # 定义测试函数 test_reorder_print
    def test_reorder_print(self):
        # 定义一个内部函数 f，接受参数 x
        def f(x):
            # 打印字符串 "moo"
            print("moo")
            # 对参数 x 进行加法操作，并赋值给 x1
            x1 = x + x
            # 打印变量 x1 的值
            print(x1)
            # 对 x1 进行乘法操作，并赋值给 x2
            x2 = x1 * x1
            # 打印整数 1、2、3
            print(1, 2, 3)
            # 对 x2 进行加法操作，并赋值给 x3
            x3 = x2 + x2
            return (x1, x3)

        # 生成一个全为 1 的形状为 (3, 3) 的张量 x
        x = torch.ones(3, 3)
        # 使用 eager 模式和完整图形模式编译函数 f
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        
        # 使用 patch 临时替换 sys.stdout，以捕获打印输出
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            # 调用优化后的函数 opt_f，传入参数 x
            opt_out = opt_f(x)
            # 获取捕获的标准输出内容并去除首尾空白
            printed_output = mock_stdout.getvalue().strip()
            # 调用原始函数 f，传入参数 x，并保存结果
            orig_out = f(x)

        # 断言捕获的打印输出内容与预期值相同
        self.assertEqual(printed_output, f"moo\n{torch.ones(3, 3) * 2}\n1 2 3")
        # 断言优化后的输出与原始输出相同
        self.assertTrue(same(orig_out, opt_out))

    # 使用 torch._dynamo.config.patch 装饰器，配置可重排序的警告函数为 warnings.warn
    @torch._dynamo.config.patch(reorderable_logging_functions={warnings.warn})
    # 定义测试函数 test_reorder_warnings
    def test_reorder_warnings(self):
        # 导入 warnings 模块
        import warnings

        # 定义一个内部函数 f，接受参数 x
        def f(x):
            # 对参数 x 进行加法操作，并赋值给 x1
            x1 = x + x
            # 发出警告消息 "moo"
            warnings.warn("moo")
            # 对 x1 进行乘法操作，并赋值给 x2
            x2 = x1 * x1
            # 发出警告消息，消息内容为 x2 的字符串表示
            warnings.warn(f"{x2}")
            # 对 x2 进行加法操作，并赋值给 x3
            x3 = x2 + x2
            return x3

        # 生成一个全为 1 的形状为 (3, 3) 的张量 x
        x = torch.ones(3, 3)
        # 使用 eager 模式和完整图形模式编译函数 f
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        
        # 使用 warnings.catch_warnings 上下文管理器捕获警告
        with warnings.catch_warnings(record=True) as w:
            # 调用优化后的函数 opt_f，传入参数 x
            opt_out = opt_f(x)
            # 获取捕获的警告消息内容并转化为字符串列表
            warning_messages = [str(i.message) for i in w]
            # 调用原始函数 f，传入参数 x，并保存结果
            orig_out = f(x)

        # 断言优化后的输出与原始输出相同
        self.assertTrue(same(orig_out, opt_out))
        # 断言捕获的警告消息中包含字符串 "moo"
        self.assertIn("moo", warning_messages)

    # 继续上一个测试函数 test_reorder_warnings 的测试定义
    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_reorder_print_graph_break(self):
        def f(x):
            x1 = x + x
            # 打印计算结果
            print(f"res: {x1}")
            x2 = x1 * x1
            # 触发图断点操作
            torch._dynamo.graph_break()
            x3 = x2 + x2
            # 打印数字序列
            print(1, 2, 3)
            return x3

        x = torch.ones(3, 3)
        # 编译函数 f 以便优化
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        # 断言打印输出与预期相符
        self.assertEqual(printed_output, f"res: {torch.ones(3, 3) * 2}\n1 2 3")
        # 断言原始输出与优化后输出相同
        self.assertTrue(same(orig_out, opt_out))

    def test_reorder_custom_log_fn(self):
        custom_logs = []

        def custom_log(s: str):
            # 触发图断点操作
            torch._dynamo.graph_break()
            # 记录自定义日志
            custom_logs.append(s)

        def f(x):
            custom_log("moo")
            x1 = x + x
            custom_log(f"{x1}")
            return x + x

        x = torch.ones(3, 3)
        counters.clear()
        # 使用动态配置管理器重新排序日志函数
        with torch._dynamo.config.patch(reorderable_logging_functions={custom_log}):
            opt_f = torch.compile(backend="eager")(f)
            opt_out = opt_f(x)

        # 断言图断点计数为1
        self.assertEqual(sum(counters["graph_break"].values()), 1)
        # 断言自定义日志的内容正确记录
        self.assertEqual(custom_logs[0], "moo")
        self.assertEqual(custom_logs[1], f"{torch.ones(3, 3) * 2}")

    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_constant_mutation(self):
        def f(x):
            alist = [x]
            alist.append(x + 1)
            # 打印列表中的最后一个元素
            print(alist[-1])
            # 引起图断点操作
            alist[0].sum().item()  # graph break
            res = alist.pop()
            # 再次打印列表中的最后一个元素
            print(alist[-1])
            # 引起图断点操作
            res.sum().item()  # graph break
            return res

        inputs = (torch.tensor([1]),)
        counters.clear()
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(*inputs)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(*inputs)

        # 断言打印输出与预期相符
        self.assertEqual(printed_output, "tensor([2])\ntensor([1])")
        # 断言原始输出与优化后输出相同
        self.assertTrue(same(orig_out, opt_out))

        # 断言图断点计数为1，且键为 "Tensor.item"
        graph_break_key = counters["graph_break"].keys()
        self.assertEqual(len(graph_break_key), 1)
        self.assertEqual(next(iter(graph_break_key)), "Tensor.item")
# 如果这个脚本是作为主程序运行
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块中导入run_tests函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的run_tests函数，用于执行测试用例
    run_tests()
```