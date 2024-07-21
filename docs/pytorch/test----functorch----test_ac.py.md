# `.\pytorch\test\functorch\test_ac.py`

```
# Owner(s): ["oncall: pt2"]
# 导入必要的库
import random

import torch
import torch._functorch.config as config
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils.flop_counter import FlopCounterMode

# 编译函数，使用预先分区的AOT（Ahead-Of-Time）执行
def compile_with_ac(f, memory_budget):
    return torch.compile(f, backend="aot_eager_decomp_partition")

# 计算激活时的内存使用量
def get_act_mem(f):
    # 调用函数并执行反向传播
    out = f()
    out.backward()
    # 获取当前 CUDA 请求的内存量
    start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
    # 再次调用函数并获取当前 CUDA 请求的内存量
    out = f()
    cur_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
    # 计算实际使用的内存量（单位：MB）
    act_mem = (cur_mem - start_mem) / (1024 * 1024)
    # 再次执行反向传播
    out.backward()
    return act_mem

# 计算带宽和浮点运算量（FLOPs）
def get_bw_flops(f):
    # 对函数进行调用并执行反向传播
    f().backward()
    out = f()
    # 使用 FlopCounterMode 来计算总的 FLOPs，以 512x512 矩阵乘法为单位
    with FlopCounterMode(display=False) as mode:
        out.backward()
    return mode.get_total_flops() / (512**3 * 2)

# 创建输入输出对
def create_pair(B_I, O):
    # 创建指定大小的随机张量，需要的内存量为 B_I * 512 * B_I * 512 * 4 字节，即 B_I * B_I MiB
    # 需要执行的浮点运算量为 B_I * B_I * O * 512 * 512 个乘法和加法操作
    x = torch.randn(B_I * 512, B_I * 512, requires_grad=True)
    w = torch.randn(B_I * 512, O * 512, requires_grad=True)
    return x, w

# 获取函数的内存使用量和浮点运算量（FLOPs）
def get_mem_and_flops(f, memory_budget=None):
    # 返回内存使用量（四舍五入到小数点后一位）和 FLOPs
    torch._dynamo.reset()
    with config.patch(activation_memory_budget=memory_budget):
        if memory_budget is not None:
            # 如果指定了内存预算，则使用 AOT 编译函数
            f = torch.compile(f, backend="aot_eager_decomp_partition")

        # 获取激活时的内存使用量和带宽 FLOPs
        return round(get_act_mem(f), 1), get_bw_flops(f)

# 内存预算测试类
class MemoryBudgetTest(TestCase):
    # 设置测试环境，在 CUDA 设备上运行测试
    def setUp(self):
        super().setUp()
        torch.set_default_device("cuda")

    # 测试函数，验证在不同内存预算下的内存和 FLOPs 的变化
    def test_rematerializes_cheap(self):
        # 定义计算图函数 f(x, w)，其中 x 是输入张量，w 是权重张量
        def f(x, w):
            x = x.cos()  # 对输入 x 执行余弦函数
            x = torch.mm(x, w)  # 执行矩阵乘法 x * w
            return x.sum()  # 返回张量的所有元素的和

        # 创建测试数据 x 和 w
        x = torch.randn(512, 512, requires_grad=True)
        w = torch.randn(512, 512, requires_grad=True)

        # 定义调用函数的包装函数
        def call():
            return f(x, w)

        # 获取在默认条件下的内存使用量和 FLOPs
        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 1.0)  # 断言默认内存使用量为 1.0 MB
        # 获取在指定 1.0 MB 内存预算下的内存使用量和 FLOPs
        mem_10, flops_10 = get_mem_and_flops(call, memory_budget=1.0)
        # 断言内存使用量为 1.0 MB
        self.assertEqual(mem_10, 1.0)
        self.assertEqual(eager_flops, flops_10)  # 断言 FLOPs 相同
        # 获取在指定 0.5 MB 内存预算下的内存使用量和 FLOPs
        mem_5, flops_5 = get_mem_and_flops(call, memory_budget=0.5)
        # 断言内存使用量为 0.0 MB
        self.assertEqual(mem_5, 0.0)
        self.assertEqual(flops_5, eager_flops)  # 断言 FLOPs 相同
    # 定义一个测试函数，测试矩阵乘法中的链式计算，确保内存和计算资源的分配符合预期
    def test_matmul_even_chain(self):
        # 定义一个函数 f，接受一个张量 x 和权重列表 ws，执行一系列的矩阵乘法和余弦函数操作，并返回求和结果
        def f(x, ws):
            # 对输入张量 x 执行余弦函数操作
            x = x.cos()
            # 遍历权重列表 ws，对 x 执行多次矩阵乘法和余弦函数操作
            for w in ws:
                x = torch.mm(x, w).cos()
            # 返回经过操作后的张量 x 的求和结果
            return x.sum()

        # 创建一个大小为 512x512 的随机张量 x，要求梯度
        x = torch.randn(512, 512, requires_grad=True)
        # 创建一个包含 5 个大小为 512x512 的随机张量的列表 ws，每个张量要求梯度
        ws = [torch.randn(512, 512, requires_grad=True) for _ in range(5)]

        # 定义一个内部函数 call，用于调用函数 f(x, ws)
        def call():
            return f(x, ws)

        # 调用 get_mem_and_flops 函数，获取不同内存预算下的内存消耗和计算量
        eager_mem, eager_flops = get_mem_and_flops(call)

        # 循环遍历预算范围从 0 到 10
        for budget in range(0, 11):
            # 调用 get_mem_and_flops 函数，传入内存预算为 budget / 10，获取内存消耗和计算量
            mem, flops = get_mem_and_flops(call, memory_budget=budget / 10)
            if budget <= 5:
                # 当预算小于等于 5 时，验证内存消耗和计算量是否符合预期
                # 开始保存矩阵乘法结果
                self.assertEqual(mem, budget)
                self.assertEqual(flops, eager_flops + (5 - budget))
            elif budget < 10:
                # 当预算大于 5 小于 10 时，验证内存消耗和计算量是否符合预期
                # 只重新计算余弦函数操作
                self.assertEqual(mem, 5.0)
                self.assertEqual(flops, eager_flops)
            elif budget == 10:
                # 当预算为 10 时，验证内存消耗和计算量是否符合预期
                self.assertEqual(mem, 10.0)
                self.assertEqual(flops, eager_flops)
    def test_matmul_uneven_chain(self):
        # 定义一个测试函数，用于测试不均匀链中的矩阵乘法

        # 定义一个内部函数f，接受输入x和权重列表ws，计算每个权重矩阵与x的乘积的余弦值，并返回所有结果的总和
        def f(x, ws):
            xs = [torch.mm(x, w).cos() for w in ws]  # 计算每个权重矩阵与x的乘积的余弦值
            return sum([x.sum() for x in xs])  # 返回所有结果的总和

        x = torch.randn(512, 512, requires_grad=True)  # 创建一个大小为[512, 512]的随机张量x，需要计算梯度

        # 定义一个函数make_weights，根据给定的权重形状列表w_shapes，创建对应的权重张量列表ws
        def make_weights(w_shapes):
            ws = []
            for idx, dim in enumerate(w_shapes):
                ws.append(torch.randn(512, dim * 512, requires_grad=True))
            return ws

        # 定义一个函数make_weights_chain，根据给定的权重形状列表w_shapes，创建对应的权重张量列表ws，
        # 每个张量的形状与前一个张量相连接，形成链式结构
        def make_weights_chain(w_shapes):
            ws = []
            for idx, _ in enumerate(w_shapes):
                old_dim = 512 if idx == 0 else w_shapes[idx - 1] * 512
                new_dim = w_shapes[idx] * 512
                ws.append(torch.randn(old_dim, new_dim, requires_grad=True))
            return ws

        # 定义多组权重配置，每组配置包含权重形状列表和预期解决方案列表
        weight_configs = [
            (
                [11, 3, 4, 2],  # 第一组权重形状列表
                [
                    18,  # 对应解决方案：11 + 4 + 3
                    17,  # 对应解决方案：11 + 4 + 2
                    16,  # 对应解决方案：11 + 3 + 2
                    15,  # 对应解决方案：11 + 4
                    14,  # 对应解决方案：11 + 3
                    13,  # 对应解决方案：11 + 2
                    11,  # 对应解决方案：11 + 2
                    7,   # 对应解决方案：4 + 3
                    6,   # 对应解决方案：4 + 2
                    5,   # 对应解决方案：3 + 2
                ],
            ),
            (
                [3, 5, 11, 17, 14],  # 第二组权重形状列表
                [
                    42,  # 对应解决方案：17 + 14 + 11
                    30,  # 对应解决方案：11 + 17 + 2
                    19,  # 对应解决方案：11 + 5 + 3
                    8,   # 对应解决方案：5 + 3
                    3,   # 对应解决方案：3
                ],
            ),
        ]

        random.seed(0)
        random_arr = [random.randint(0, 50) for _ in range(10)]  # 生成一个包含10个随机整数的列表
        exact_sums = []
        for i in range(10):
            random.shuffle(random_arr)  # 打乱随机整数列表的顺序
            exact_sums.append(sum(random_arr[:i]))  # 计算部分随机整数列表的累积和并添加到exact_sums中
        weight_configs.append((random_arr, exact_sums))  # 将随机整数列表及其累积和添加到权重配置列表中

        # 遍历权重配置列表，每次迭代都执行以下操作
        for weight_shapes, exact_solves in weight_configs:
            ws = make_weights(weight_shapes)  # 根据当前权重形状列表创建权重张量列表ws

            # 定义一个内部函数call，用于调用函数f并返回其结果
            def call():
                return f(x, ws)

            eager_mem, eager_flops = get_mem_and_flops(call)  # 调用get_mem_and_flops函数获取内存和计算量
            total_mem = sum(weight_shapes)  # 计算当前权重形状列表的总和作为总内存量
            self.assertEqual(eager_mem, sum(weight_shapes))  # 断言eager_mem等于当前权重形状列表的总和
            # 遍历预期解决方案列表，每次迭代都执行以下操作
            for mem_achieved in exact_solves:
                mem, _ = get_mem_and_flops(call, memory_budget=mem_achieved / total_mem)  # 根据预算内存调用get_mem_and_flops函数
                self.assertEqual(mem, mem_achieved)  # 断言获取的内存等于预期解决方案中的值
    # 测试优先选择较便宜的矩阵乘法函数
    def test_prioritize_cheaper_matmul(self):
        # 定义函数 f，接受两个列表 xs 和 ws，分别进行矩阵乘法和余弦运算，返回总和
        def f(xs, ws):
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum([x.sum() for x in xs])

        # 创建第一组输入 x1 和 w1
        x1, w1 = create_pair(1, 4)
        # 创建第二组输入 x2 和 w2
        x2, w2 = create_pair(2, 2)

        # 定义函数 call，调用函数 f 并传入两组输入
        def call():
            return f([x1, x2], [w1, w2])

        # 计算调用函数 call 的内存和计算量
        eager_mem, eager_flops = get_mem_and_flops(call)
        # 断言预期内存占用为 8
        self.assertEqual(eager_mem, 8)
        # 断言预期计算量为 24
        self.assertEqual(eager_flops, 24)

        # 计算在内存预算为 0.5 时的内存和计算量
        comp_mem, comp_flops = get_mem_and_flops(call, memory_budget=0.5)
        # 断言预期内存占用为 4
        self.assertEqual(comp_mem, 4)
        # 断言预期计算量为原计算量加上额外的 4
        self.assertEqual(comp_flops, eager_flops + 4)

    @config.patch(activation_memory_budget_runtime_estimator="profile")
    def test_profile(self):
        # 定义函数 f，接受一个输入 x 和一个权重列表 ws，对 x 进行余弦运算，然后进行矩阵乘法和余弦运算，返回总和
        def f(x, ws):
            x = x.cos()
            for w in ws:
                x = torch.mm(x, w).cos()
            return x.sum()

        # 创建一个大小为 512x512 的随机张量 x，需要梯度计算
        x = torch.randn(512, 512, requires_grad=True)
        # 创建包含 5 个大小为 512x512 的随机张量的列表 ws，需要梯度计算
        ws = [torch.randn(512, 512, requires_grad=True) for _ in range(5)]

        # 定义函数 call，调用函数 f 并传入随机张量 x 和权重列表 ws
        def call():
            return f(x, ws)

        # 计算调用函数 call 的内存和计算量
        eager_mem, eager_flops = get_mem_and_flops(call)
        # 计算在内存预算为 0.2 时的内存和计算量
        mem, flops = get_mem_and_flops(call, memory_budget=0.2)
        # 断言预期内存占用为 2
        self.assertEqual(mem, 2)
        # 断言预期计算量为原计算量加上额外的 3
        self.assertEqual(flops, eager_flops + 3)

    def test_prioritize_cheaper_matmul2(self):
        # 定义函数 f，接受两个列表 xs 和 ws，分别进行矩阵乘法和余弦运算，返回总和
        def f(xs, ws):
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum([x.sum() for x in xs])

        # 创建数据列表 data，包含三个元组 (4, 4), (6, 2), (2, 6)
        data = [(4, 4), (6, 2), (2, 6)]
        # 使用列表推导式，创建两个列表 xs 和 ws，分别存储每对数据的结果
        xs, ws = zip(*[create_pair(a, b) for a, b in data])

        # 定义函数 call，调用函数 f 并传入列表 xs 和 ws
        def call():
            return f(xs, ws)

        # 计算调用函数 call 的内存和计算量
        eager_mem, eager_flops = get_mem_and_flops(call)
        # 断言预期内存占用为 40
        self.assertEqual(eager_mem, 40)
        # 断言预期计算量为 320
        self.assertEqual(eager_flops, 320)

        # 计算在内存预算为 28/eager_mem 时的内存和计算量
        mem, flops = get_mem_and_flops(call, memory_budget=28 / eager_mem)
        # 断言预期内存占用为 28
        self.assertEqual(mem, 28)
        # 断言预期计算量为原计算量加上额外的矩阵乘法计算
        self.assertEqual(flops - eager_flops, 2 * 2 * 6)

        # 计算在内存预算为 16/eager_mem 时的内存和计算量
        mem, flops = get_mem_and_flops(call, memory_budget=16 / eager_mem)
        # 断言预期内存占用为 12
        self.assertEqual(mem, 12)
        # 断言预期计算量为原计算量加上额外的矩阵乘法计算
        self.assertEqual(flops - eager_flops, 2 * 2 * 6 + 4 * 4 * 4)
    def test_attention_vs_linear(self):
        # 定义内部函数 f，接受输入 x 和权重 w，计算加权乘积后的余弦相似度之和
        def f(x, w):
            # 记录原始形状
            orig_shape = x.shape
            # 将输入 x 重塑为形状 (1, 1, x.shape[0], x.shape[1])
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
            # 使用 scaled_dot_product_attention 函数进行自注意力计算，非因果关系
            x = torch.nn.functional.scaled_dot_product_attention(
                x, x, x, is_causal=False
            ).reshape(*orig_shape)
            # 将结果与权重矩阵 w 进行矩阵乘法
            x = torch.mm(x, w)
            # 对结果应用余弦函数
            x = x.cos()
            # 返回结果的和
            return x.sum()

        # 定义尝试不同序列长度 S 和维度 D 的函数
        def try_seq_length(S, D, expected_recompute):
            # 随机生成形状为 S*512 和 D*512 的张量 x 和权重矩阵 w，需要计算梯度
            x = torch.randn(S * 512, D * 512, requires_grad=True)
            w = torch.randn(D * 512, D * 512, requires_grad=True)

            # 定义一个调用函数，返回 f(x, w) 的结果
            def call():
                return f(x, w)

            # 进入 FLOP 计数模式
            with FlopCounterMode(display=False) as mode:
                call()
            # 获取矩阵乘法的 FLOP 数量
            mm_flops = mode.get_flop_counts()["Global"][torch.ops.aten.mm]
            # 获取总的 FLOP 数量并减去矩阵乘法得到自注意力的 FLOP 数量
            attn_flops = mode.get_total_flops() - mm_flops
            # 根据输入形状调整 FLOP 数量
            mm_flops /= 512**3 * 2
            attn_flops /= 512**3 * 2

            # 获取在不同内存限制下的内存使用量和 FLOP 数量
            eager_mem, eager_flops = get_mem_and_flops(call)
            # 断言预期内存使用量
            self.assertEqual(eager_mem, S * D * 2)

            # 强制在内存预算为 0.6 时重新计算 mm 或 attn
            mem, flops = get_mem_and_flops(
                call, memory_budget=0.6
            )  # Force it to recompute one of mm or attn
            # 断言预期内存使用量
            self.assertEqual(mem, S * D)
            # 根据预期重新计算的部分选择对应的 FLOP 数量
            if expected_recompute == "attn":
                expected_flops = attn_flops
            else:
                expected_flops = mm_flops
            # 断言重新计算的 FLOP 数量与预期相符
            self.assertEqual(flops - eager_flops, expected_flops)

        # 测试的主要目的是验证当序列长度乘以 2 大于 D 时，注意力机制的计算比线性计算更昂贵
        try_seq_length(1, 1, "mm")
        try_seq_length(1, 3, "attn")
        try_seq_length(2, 2, "mm")
        try_seq_length(2, 1, "mm")
        try_seq_length(2, 5, "attn")
        try_seq_length(4, 7, "mm")
        try_seq_length(4, 9, "attn")
if __name__ == "__main__":
    # 检查当前脚本是否作为主程序运行
    # 如果当前系统支持 CUDA 并且不是使用 ROCm 进行测试
    if HAS_CUDA and not TEST_WITH_ROCM:
        # 运行测试函数
        run_tests()
```