# `.\pytorch\test\dynamo\test_sdpa.py`

```py
# Owner(s): ["module: dynamo"]

import contextlib  # 导入上下文管理模块

import torch._dynamo.test_case  # 导入测试用例模块
import torch._dynamo.testing  # 导入测试相关模块
from torch._dynamo.testing import CompileCounter  # 从测试模块导入编译计数器
from torch.backends.cuda import SDPAParams  # 导入 SDPAParams 类


@contextlib.contextmanager
def allow_in_graph_sdpa_params():
    global SDPAParams
    try:
        old = SDPAParams
        SDPAParams = torch._dynamo.allow_in_graph(SDPAParams)  # 允许在图中使用 SDPAParams
        yield
    finally:
        SDPAParams = old  # 恢复原始的 SDPAParams


class TestSDPA(torch._dynamo.test_case.TestCase):
    def assert_ref_equals_params(self, actual, expected):
        self.assertIs(actual.query, expected.query)  # 断言实际查询等于预期查询
        self.assertIs(actual.key, expected.key)  # 断言实际键等于预期键
        self.assertIs(actual.value, expected.value)  # 断言实际值等于预期值
        self.assertIs(actual.attn_mask, expected.attn_mask)  # 断言实际注意力掩码等于预期注意力掩码

    def test_returns_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()  # 创建编译计数器对象

            @torch.compile(fullgraph=True, backend=counter)
            def fn(q, k, v, m):
                return SDPAParams(q, k, v, m, 0.1, True)  # 返回一个 SDPAParams 对象

            q = torch.randn(10)  # 生成大小为 10 的随机张量 q
            k = torch.randn(10)  # 生成大小为 10 的随机张量 k
            v = torch.randn(10)  # 生成大小为 10 的随机张量 v
            m = torch.randn(10)  # 生成大小为 10 的随机张量 m
            o = fn(q, k, v, m)  # 调用 fn 函数
            self.assertTrue(isinstance(o, SDPAParams))  # 断言 o 是 SDPAParams 类型的对象
            self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True))  # 断言 o 的属性与预期的 SDPAParams 对象属性相等
            self.assertEqual(counter.frame_count, 1)  # 断言编译计数器的帧计数为 1

    def test_graph_break_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()  # 创建编译计数器对象

            @torch.compile(backend=counter)
            def fn(q, k, v, m):
                z = SDPAParams(q, k, v, m, 0.1, True)  # 创建 SDPAParams 对象
                torch._dynamo.graph_break()  # 调用图中断点函数
                return z, q + 1  # 返回 z 和 q+1

            q = torch.randn(10)  # 生成大小为 10 的随机张量 q
            k = torch.randn(10)  # 生成大小为 10 的随机张量 k
            v = torch.randn(10)  # 生成大小为 10 的随机张量 v
            m = torch.randn(10)  # 生成大小为 10 的随机张量 m
            o, _ = fn(q, k, v, m)  # 调用 fn 函数并接收返回值
            self.assertTrue(isinstance(o, SDPAParams))  # 断言 o 是 SDPAParams 类型的对象
            self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True))  # 断言 o 的属性与预期的 SDPAParams 对象属性相等
            self.assertEqual(counter.frame_count, 2)  # 断言编译计数器的帧计数为 2

    def test_input_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()  # 创建编译计数器对象

            @torch.compile(backend=counter)
            def fn(sdpap, q):
                torch._dynamo.graph_break()  # 调用图中断点函数
                return sdpap, sdpap.query + q  # 返回 sdpap 和 sdpap.query + q

            q = torch.randn(10)  # 生成大小为 10 的随机张量 q
            k = torch.randn(10)  # 生成大小为 10 的随机张量 k
            v = torch.randn(10)  # 生成大小为 10 的随机张量 v
            m = torch.randn(10)  # 生成大小为 10 的随机张量 m
            s = SDPAParams(q, k, v, m, 0.1, True)  # 创建 SDPAParams 对象 s
            o, _ = fn(s, q)  # 调用 fn 函数并接收返回值
            self.assertIs(o, s)  # 断言 o 与 s 是同一个对象
            self.assertEqual(counter.frame_count, 1)  # 断言编译计数器的帧计数为 1
    # 定义一个测试方法，用于测试中间属性访问 SDPAParams 的行为
    def test_intermediate_attr_access_SDPAParams(self):
        # 进入允许图中 SDPA 参数的上下文
        with allow_in_graph_sdpa_params():
            # 创建一个编译计数器对象
            counter = CompileCounter()

            # 使用装饰器声明 fn 函数为编译函数，启用完整图模式，并使用 counter 作为后端
            @torch.compile(fullgraph=True, backend=counter)
            def fn(q, k, v, m):
                # 对输入的 q 加 1
                q += 1
                # 创建 SDPAParams 对象 z，传入 q, k, v, m, 0.1, True
                z = SDPAParams(q, k, v, m, 0.1, True)
                # 获取 SDPAParams 对象 z 的 query 属性
                a = z.query
                # 返回 a + 1, z 对象本身和 q
                return a + 1, z, q

            # 生成随机张量 q, k, v, m 各 10 维
            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            # 调用 fn 函数，并接收返回值，其中第二个返回值 o 为 SDPAParams 对象
            _, o, _ = fn(q, k, v, m)
            # 创建期望的 SDPAParams 对象 expected
            expected = SDPAParams(q, k, v, m, 0.1, True)
            # 使用断言方法检查 o 是否与 expected 相等
            self.assert_ref_equals_params(o, expected)
            # 使用断言方法检查 counter 的 frame_count 属性是否为 1
            self.assertEqual(counter.frame_count, 1)
# 如果当前脚本作为主程序运行（而不是被导入到其他程序中执行），则执行以下代码
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的 run_tests 函数，用于执行测试用例
    run_tests()
```