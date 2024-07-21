# `.\pytorch\benchmarks\operator_benchmark\pt_extension\cpp_extension_test.py`

```
import unittest  # 导入unittest模块，用于编写和运行单元测试

import benchmark_cpp_extension  # noqa: F401 导入名为benchmark_cpp_extension的模块，但不使用其对象

import torch  # 导入PyTorch库


class TestConsumeOp(unittest.TestCase):  # 定义测试类TestConsumeOp，继承自unittest.TestCase
    def test_jit_consume_op(self):  # 定义测试方法test_jit_consume_op
        iters = 6  # 设置迭代次数为6

        def foo(x):  # 定义内部函数foo，接受参数x
            for i in range(iters):  # 循环iters次
                result = torch.ops.operator_benchmark._consume(torch.sum(x))  # 调用torch.ops.operator_benchmark._consume函数对torch.sum(x)求和
            return result  # 返回最后一次迭代的结果

        r = torch.jit.trace(foo, (torch.rand(2, 2)))  # 使用torch.jit.trace对函数foo进行跟踪编译，输入参数为torch.rand(2, 2)，结果保存在r中

        graph = str(r.graph)  # 将r的计算图转换为字符串形式
        occurrence = graph.count("aten::sum")  # 统计计算图中"aten::sum"出现的次数，保存在occurrence中

        x = torch.rand(2, 2)  # 生成一个2x2的随机张量x
        value = r(x)  # 对输入x调用r，得到计算结果，保存在value中
        self.assertEqual(value, torch.sum(x))  # 断言value与torch.sum(x)的结果相等
        self.assertEqual(occurrence, iters)  # 断言"aten::sum"在计算图中出现的次数为iters

    def test_jit_consume_op_for_list_input(self):  # 定义测试方法test_jit_consume_op_for_list_input
        iters = 6  # 设置迭代次数为6

        def foo(x):  # 定义内部函数foo，接受参数x
            for i in range(iters):  # 循环iters次
                result = torch.ops.operator_benchmark._consume(torch.chunk(x, 2))  # 调用torch.ops.operator_benchmark._consume函数对torch.chunk(x, 2)进行操作
            return result  # 返回最后一次迭代的结果

        r = torch.jit.trace(foo, torch.rand(2, 2))  # 使用torch.jit.trace对函数foo进行跟踪编译，输入参数为torch.rand(2, 2)，结果保存在r中

        graph = str(r.graph)  # 将r的计算图转换为字符串形式
        occurrence = graph.count("aten::chunk")  # 统计计算图中"aten::chunk"出现的次数，保存在occurrence中

        x = torch.rand(2, 2)  # 生成一个2x2的随机张量x
        value = r(x)  # 对输入x调用r，得到计算结果，保存在value中

        self.assertTrue(  # 断言以下条件为真
            all(torch.allclose(t1, t2) for t1, t2 in zip(value, torch.chunk(x, 2)))  # 对value和torch.chunk(x, 2)中的张量逐一比较是否接近
        )
        self.assertEqual(occurrence, iters)  # 断言"aten::chunk"在计算图中出现的次数为iters
```