# `.\pytorch\test\fx\test_cse_pass.py`

```py
# Owner(s): ["oncall: fx"]

import random  # 导入 random 模块

import torch  # 导入 torch 模块
from torch.fx import symbolic_trace  # 从 torch.fx 模块导入 symbolic_trace 函数
from torch.fx.experimental.proxy_tensor import make_fx  # 从 torch.fx.experimental.proxy_tensor 模块导入 make_fx 函数
from torch.fx.passes.dialect.common.cse_pass import CSEPass, get_CSE_banned_ops  # 从 torch.fx.passes.dialect.common.cse_pass 模块导入 CSEPass 类和 get_CSE_banned_ops 函数

from torch.testing._internal.common_utils import run_tests, TestCase  # 从 torch.testing._internal.common_utils 导入 run_tests 函数和 TestCase 类


banned_ops = get_CSE_banned_ops()  # 获取 CSE 优化时禁止操作的集合
P_default = CSEPass(banned_ops=banned_ops)  # 创建一个 CSEPass 对象，使用默认的禁止操作集合


def check(self, f, t, delta, check_val=True, graph_input=False, P=None):
    """
    检查 CSE 修改后的图形 ``f``
    1) 节点数是否减少了 delta，以及
    2) 第二次通过时节点数是否进一步减少，且
    3) 修改返回结果仅在节点数减少时为 True。

    Args:
        f: 要检查的函数
        t: 传递给 f 的张量
        delta: 一个大于等于 -1 的整数。
               如果 delta = -1，则仅检查新图的节点数是否少于或等于旧图
        check_val: 如果为 True，则检查 f 的输出是否正确
        graph_input: 如果 f 是 GraphModule 类型，则为 True
        P: 要使用的传递。如果为 None，则使用 P_default
    """
    if graph_input:
        fx_g = f  # 如果是图形输入，则 fx_g 等于 f
    else:
        fx_g = make_fx(f)(t)  # 否则，通过 make_fx 函数生成 fx_g

    if P is None:
        P = P_default  # 如果 P 为 None，则使用默认的 P_default

    res = P(fx_g)  # 对 fx_g 应用优化传递 P
    new_g = res.graph_module  # 获取优化后的图形模块
    new_graph = new_g.graph  # 获取优化后的图形

    # 节点数是否减少或保持不变
    old_num_nodes = len(fx_g.graph.nodes)  # 原始图的节点数
    new_num_nodes = len(new_graph.nodes)  # 优化后图的节点数

    assert (
        new_num_nodes < old_num_nodes
    ) == modified, "如果节点数减少，则修改应为 True"

    if delta == -1:
        self.assertTrue(
            old_num_nodes >= new_num_nodes,
            (f"节点数增加 {old_num_nodes}, {new_num_nodes}"),
        )
    else:
        self.assertTrue(
            old_num_nodes == new_num_nodes + delta,
            (
                f"节点数不同 {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}"
            ),
        )

    # 第二次传递不应减少更多节点
    res = P(new_g)
    pass_2_graph = res.graph_module.graph
    pass_2_num_nodes = len(pass_2_graph.nodes)
    self.assertTrue(
        pass_2_num_nodes == new_num_nodes,
        (
            f"第二次传递的图形节点数较少 {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}"
        ),
    )

    # 检查正确性
    if check_val:
        true_result = fx_g(t)  # 真实结果
        our_result = new_g(t)  # 优化后结果
        if true_result is None:  # 如果都返回 None
            self.assertTrue(
                our_result is None, f"真实结果为 None，CSE 结果为 {our_result}"
            )
        else:  # 返回的结果相同
            self.assertTrue(
                torch.all(true_result == our_result),
                (f"结果不同 {true_result}, {our_result}"),
            )  # 检查结果相同


class TestCSEPass(TestCase):
    # TestCSEPass 类继承自 TestCase 类，用于测试 CSEPass 的功能
    def test_nochange(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x + 1
            a = x + 1
            # 计算 b = x + a
            b = x + a
            # 将 a 的值设为 x，重新赋值
            a = x
            # 计算 d = x + a
            d = x + a
            # 返回 b + d 的结果
            return b + d

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期
        check(self, f, t, 0)

    def test_empty(self):
        # 定义一个空函数 f(x)
        def f(x):
            pass

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期
        check(self, f, t, 0)

    def test_immutable_list_type(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x 按行求和
            a = x.sum(dim=1)
            # 计算 b = x 按行求和
            b = x.sum(dim=1)
            # 计算 c = x 全部元素求和
            c = x.sum()
            # 计算 d = x 全部元素求和
            d = x.sum()
            # 返回 a + b + c + d 的结果
            return a + b + c + d

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果为 2
        check(self, f, t, 2)

    def test_immutable_list_multiple_entries(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x 按行列求和
            a = x.sum(dim=[0, 1])
            # 计算 b = x 按行列求和
            b = x.sum(dim=[0, 1])
            # 计算 c = x 按行求和
            c = x.sum(dim=1)
            # 计算 d = x 按行求和
            d = x.sum(dim=1)
            # 返回 a + b + c + d 的结果
            return a + b + c + d

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果为 2
        check(self, f, t, 2)

    def test_simple(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x 的余弦值
            a = x.cos()
            # 计算 b = x 的余弦值
            b = x.cos()
            # 计算 c = a + a
            c = a + a
            # 计算 d = b + b
            d = b + b
            # 返回 c + d 的结果
            return c + d

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果为 2
        check(self, f, t, 2)

    def test_simple_2(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x 的余弦值，再计算结果的正弦值
            a = x.cos().sin()
            # 计算 b = x 的余弦值，再计算结果的正弦值
            b = x.cos().sin()
            # 计算 c = a + a
            c = a + a
            # 计算 d = b + b
            d = b + b
            # 返回 c + d 的结果
            return c + d

        # 生成一个长度为 1 的随机张量 t
        t = torch.randn(1)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果为 3
        check(self, f, t, 3)

    def test_two_args_default(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x 按行求和
            a = x.sum(dim=1)
            # 计算 b = x 按行求和，不保持维度
            b = x.sum(dim=1, keepdim=False)
            # 计算 c = x 按行求和，不保持维度
            c = x.sum(dim=1, keepdim=False)
            # 计算 d = x 按行求和
            d = x.sum(dim=1)
            # 返回 a + b + c + d 的结果
            return a + b + c + d

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果为 3
        check(self, f, t, 3)

    def test_two_args(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x 按行求和
            a = x.sum(dim=1)
            # 计算 b = x 按行求和，保持维度
            b = x.sum(dim=1, keepdim=True)
            # 计算 c = x 按行求和，保持维度
            c = x.sum(dim=1, keepdim=True)
            # 计算 d = x 按行求和
            d = x.sum(dim=1)
            # 返回 a + b + c + d 的结果
            return a + b + c + d

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果为 2
        check(self, f, t, 2)

    def test_simple_multiple_same_ops(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 计算 a = x 全部元素求和
            a = x.sum()
            # 计算 b = x 全部元素求和
            b = x.sum()
            # 计算 c = x 全部元素求和
            c = x.sum()
            # 计算 d = x 全部元素求和
            d = x.sum()
            # 返回 a + b + c + d 的结果
            return a + b + c + d

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果为 3
        check(self, f, t, 3)

    def test_nested_immutable_list_type(self):
        # 定义一个函数 f(x)，对输入 x 进行一系列操作并返回结果
        def f(x):
            # 将输入 x 在行方向上进行拼接
            a = torch.cat((x, x))
            # 将输入 x 在行方向上进行拼接
            b = torch.cat((x, x))
            # 返回 a + b 的结果
            return a + b

        # 生成一个 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，验证函数 f 对于输入 t 的输出是否符合预期，期望结果
    """
    Test that banned list ban ops as expected.
    """

    # 定义一个测试函数，接受参数 x
    def test_banned_list(self):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 计算 a = x + 1
            a = x + 1
            # 计算 b = x + 1
            b = x + 1
            # 返回 a + b 的结果
            return a + b

        # 生成一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)
        # 创建一个常量传播优化对象 P_ban_add，禁止使用 torch.ops.aten.add
        P_ban_add = P = CSEPass(banned_ops=[torch.ops.aten.add])
        # 调用 check 函数，检查函数 f 的输出，验证是否禁止了 add 操作
        check(self, f, t, 0, P=P_ban_add)  # check that add is banned
        # 再次调用 check 函数，检查函数 f 的输出，验证默认情况下 add 操作未被禁止
        check(self, f, t, 1)  # check that add is not banned by default

    # 定义一个测试函数 test_rand_like
    def test_rand_like(self):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 生成一个与 x 形状相同的随机张量 a
            a = torch.rand_like(x)
            # 生成一个与 x 形状相同的随机张量 b
            b = torch.rand_like(x)
            # 返回 a + b 的结果
            return a + b

        # 生成一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，检查函数 f 的输出，不检查返回值的数值正确性
        check(self, f, t, 0, check_val=False)

    # 定义一个测试函数 test_rand_n
    def test_rand_n(self):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 生成一个形状为 (4,) 的随机张量 a
            a = torch.randn(4)
            # 生成一个形状为 (4,) 的随机张量 b
            b = torch.randn(4)
            # 返回 a + b 的结果
            return a + b

        # 生成一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，检查函数 f 的输出，不检查返回值的数值正确性
        check(self, f, t, 0, check_val=False)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests() 的函数，用于执行测试代码或测试套件
    run_tests()
```