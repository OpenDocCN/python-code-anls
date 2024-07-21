# `.\pytorch\test\test_static_runtime.py`

```
# Owner(s): ["module: unknown"]

# 导入所需的库和模块
import unittest
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.static_module import StaticModule
from typing import List


# 定义一个函数，模拟 torch.nn.functional.linear 的行为
def linear_shim(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # 计算线性层的输出，input 与 weight 的转置相乘
    output = input.matmul(weight.t())
    # 如果有偏置项，将其加到输出上
    if bias is not None:
        output += bias
    # 返回输出
    ret = output
    return ret


# 替换 torch.nn.functional.linear 函数为 linear_shim
torch.nn.functional.linear = linear_shim


# 定义一个多头注意力层的类
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        # 定义线性变换层，用于计算查询、键和值
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        # self.dropout = nn.Dropout(dropout)
        # 缩放因子，用于缩放注意力分数
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        # 分别对查询、键和值进行线性变换
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # 将变换后的张量重塑为多头张量，并对维度进行置换
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # 计算注意力分数
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = energy.masked_fill(mask == 0, -1e10)
        # 计算注意力权重
        attention = torch.softmax(energy, dim=-1)
        # x = torch.matmul(self.dropout(attention), V)
        # 计算加权后的值
        x = torch.matmul(attention, V)
        # 还原张量形状，并进行连续化
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        # 对输出应用线性变换层
        x = self.fc_o(x)
        return x, attention


# 从 https://github.com/facebookresearch/dlrm/blob/master/dlrm_s_pytorch.py 获取的函数
def create_mlp(ln, sigmoid_layer):
    layers = nn.ModuleList()
    for i in range(0, len(ln) - 1):
        n = ln[i]
        m = ln[i + 1]

        # 创建线性层，并初始化权重和偏置
        LL = nn.Linear(int(n), int(m), bias=True)

        mean = 0.0  # std_dev = np.sqrt(variance)
        std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
        std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
        bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
        LL.weight.data = torch.tensor(W, requires_grad=True)
        LL.bias.data = torch.tensor(bt, requires_grad=True)
        layers.append(LL)

        # 根据指定的层索引添加激活函数
        if i == sigmoid_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU())

    with torch.no_grad():
        # 使用 torch.jit.script 进行模型脚本化
        s = torch.jit.script(torch.nn.Sequential(*layers))
    # 将模型设置为评估模式
    s.eval()
    return s
def trivial_graph(a, b, c):
    # 创建一个大小为2x2的张量，所有元素值为3
    s = torch.tensor([[3, 3], [3, 3]])
    # 返回 a + b * c + s 的结果
    return a + b * c + s

def elementwise_square_addition(input1, input2):
    # 返回 input1 和 input2 的平方和
    return input1 * input1 + input2 * input2

def fork_wait_graph1(input1, input2):
    # 在 input1 和 input2 上执行 elementwise_square_addition 的 fork 操作
    fut = torch.jit.fork(elementwise_square_addition, input1, input2)
    # 等待并返回 fork 操作的结果
    return torch.jit.wait(fut)

def fork_wait_graph2(input1, input2):
    # 在 loop_graph 函数上执行 fork 操作，使用 input1, input2 和 5 作为参数
    fut = torch.jit.fork(loop_graph, input1, input2, 5)
    # 等待并返回 fork 操作的结果
    return torch.jit.wait(fut)

"""
   多次 fork/wait 操作的图表
   :param input: 要传递给 fork 子图的 torch.tensor 输入
   :param iters: 创建的 future/wait 对数
"""
def fork_wait_graph3(input, iters: int):
    futures : List[torch.jit.Future[torch.Tensor]] = []
    # 执行 iters 次 torch.neg 的 fork 操作，将每个 future 存储在 futures 列表中
    for _ in range(iters):
        futures.append(torch.jit.fork(torch.neg, input))
    results = []
    # 等待每个 future 并将结果存储在 results 列表中
    for future in futures:
        results.append(torch.jit.wait(future))
    # 返回所有结果的总和
    return torch.sum(torch.stack(results))

"""
   多级 fork/wait 操作的图表
   :param input: 要传递给 fork 子图的 torch.tensor 输入
   :param num_forks: 顶级 fork 的数量
   :param num_child_forks: 每个父级 fork 的子 fork 数量
"""
def fork_wait_graph4(input, num_forks: int, num_child_forks: int):
    futures : List[torch.jit.Future[torch.Tensor]] = []
    # 执行 num_forks 次 fork_wait_graph3 的 fork 操作，每次使用 input 和 num_child_forks 作为参数
    for _ in range(num_forks):
        futures.append(torch.jit.fork(fork_wait_graph3, input, num_child_forks))
    results = []
    # 等待每个 future 并将结果存储在 results 列表中
    for future in futures:
        results.append(torch.jit.wait(future))
    # 返回所有结果的总和
    return torch.sum(torch.stack(results))

def add_tensor(input1, input2):
    # 返回 input1 和 input2 的和
    return input1 + input2

def fork_wait_graph_exception(input1, input2):
    # 在 add_tensor 函数上执行 fork 操作，使用 input1 和 input2 作为参数
    fut = torch.jit.fork(add_tensor, input1, input2)
    # 等待并返回 fork 操作的结果
    return torch.jit.wait(fut)

def loop_graph(a, b, iters: int):
    # 计算初始值 c，然后对其进行 iters 次迭代操作
    c = a + b * 2
    for i in range(iters):
        c = c + b
        c *= 2
        c -= a
    # 返回最终结果 c
    return c


def output_graph(a, b, c, iters: int):
    # 创建一个大小为2x2的张量，所有元素值为3
    s = torch.tensor([[3, 3], [3, 3]])
    # 计算 k 的值
    k = a + b * c + s
    d: Dict[int, torch.Tensor] = {}
    # 迭代 iters 次，将 k + i 存储在字典 d 中
    for i in range(iters):
        d[i] = k + i
    # 返回结果字典 d
    return d


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 11
        self.b = 2

    def forward(self, x):
        # 返回 self.a + self.b + x 的结果
        return self.a + self.b + x


class SubModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 12
        self.b = 2

    def forward(self, x):
        # 将 self.b 设置为 30，然后返回 self.a + self.b + x 的结果
        self.b = 30
        return self.a + self.b + x


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub1 = SubModule()
        self.sub2 = SubModule2()
        self.a = 3
        self.b = 4

    def forward(self, x):
        # 将 self.b 设置为 20，然后返回 self.sub1(x) + self.a + self.b + self.sub2(x) 的结果
        self.b = 20
        return self.sub1(x) + self.a + self.b + self.sub2(x)


class TestStaticModule(TestCase):

    """
    Test Case: To test simple fork/wait operation in a graph
    fork is called on simple addition operation on input tensors
    """
    # 定义一个测试方法，用于测试简单的fork/wait操作
    def test_fork_wait_1(self):
        # 创建一个5x5的张量，所有元素为1
        inp1 = torch.ones(5, 5)
        # 创建一个5x5的张量，元素为随机数
        inp2 = torch.randn(5, 5)
        # 使用torch.jit.script将函数fork_wait_graph1转换为Torch脚本
        torch_graph = torch.jit.script(fork_wait_graph1)
        # 调用转换后的Torch脚本计算输出结果
        output_ref = torch_graph(inp1, inp2)
        # 创建StaticModule对象，传入Torch脚本
        static_runtime_module = StaticModule(torch_graph)
        # 使用StaticModule对象计算输出结果
        output_test = static_runtime_module(inp1, inp2)
        # 断言输出结果接近预期的参考结果
        torch.testing.assert_close(output_test, output_ref)

    """
    # Test Case: 用于测试在StaticRuntime中使用runAsync API返回Future的简单fork/wait操作
    def test_fork_wait_1_async(self):
        # 创建一个5x5的张量，所有元素为1
        inp1 = torch.ones(5, 5)
        # 创建一个5x5的张量，元素为随机数
        inp2 = torch.randn(5, 5)
        # 使用torch.jit.script将函数fork_wait_graph1转换为Torch脚本
        torch_graph = torch.jit.script(fork_wait_graph1)
        # 调用转换后的Torch脚本计算输出结果
        output_ref = torch_graph(inp1, inp2)
        # 创建StaticModule对象，传入Torch脚本
        static_runtime_module = StaticModule(torch_graph)
        # 使用StaticModule对象的runAsync方法计算输出结果的Future
        output_test = static_runtime_module.runAsync((inp1, inp2), {})
        # 等待异步操作完成
        output_test.wait()
        # 断言异步操作返回值接近预期的参考结果
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    # Test Case: 用于测试在图形中执行具有循环子图的fork/wait操作，执行混合操作
    def test_fork_wait_2(self):
        # 创建一个5x5的张量，元素为随机数
        inp1 = torch.randn(5, 5)
        # 创建一个5x5的张量，元素为随机数
        inp2 = torch.randn(5, 5)
        # 使用torch.jit.script将函数fork_wait_graph2转换为Torch脚本
        torch_graph = torch.jit.script(fork_wait_graph2)
        # 调用转换后的Torch脚本计算输出结果
        output_ref = torch_graph(inp1, inp2)
        # 创建StaticModule对象，传入Torch脚本
        static_runtime_module = StaticModule(torch_graph)
        # 使用StaticModule对象计算输出结果
        output_test = static_runtime_module(inp1, inp2)
        # 断言输出结果接近预期的参考结果
        torch.testing.assert_close(output_test, output_ref)

    """
    # Test Case: 用于测试在循环子图中执行fork/wait操作的fork/wait操作，执行混合操作
    def test_fork_wait_2_async(self):
        # 创建一个5x5的张量，元素为随机数
        inp1 = torch.randn(5, 5)
        # 创建一个5x5的张量，元素为随机数
        inp2 = torch.randn(5, 5)
        # 使用torch.jit.script将函数fork_wait_graph2转换为Torch脚本
        torch_graph = torch.jit.script(fork_wait_graph2)
        # 调用转换后的Torch脚本计算输出结果
        output_ref = torch_graph(inp1, inp2)
        # 创建StaticModule对象，传入Torch脚本
        static_runtime_module = StaticModule(torch_graph)
        # 使用StaticModule对象的runAsync方法计算输出结果的Future
        output_test = static_runtime_module.runAsync((inp1, inp2), {})
        # 等待异步操作完成
        output_test.wait()
        # 断言异步操作返回值接近预期的参考结果
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    # Test Case: 用于测试在图形中执行具有多个fork/wait操作的fork/wait操作
    def test_fork_wait_3(self):
        # 创建一个3x3的张量，所有元素为1
        input = torch.ones(3, 3)
        # 定义fork的次数为10
        num_forks = 10
        # 使用torch.jit.script将函数fork_wait_graph3转换为Torch脚本
        torch_graph = torch.jit.script(fork_wait_graph3)
        # 调用转换后的Torch脚本计算输出结果
        output_ref = torch_graph(input, num_forks)
        # 创建StaticModule对象，传入Torch脚本
        static_runtime_module = StaticModule(torch_graph)
        # 使用StaticModule对象计算输出结果
        output_test = static_runtime_module(input, num_forks)
        # 断言输出结果接近预期的参考结果
        torch.testing.assert_close(output_test, output_ref)

    """
    # Test Case: 用于测试在图形中具有多个fork/wait操作的fork/wait操作，在runAsync API返回Future中执行
    """
    # 定义一个测试方法，测试在图形中的 fork/wait 操作
    def test_fork_wait_3_async(self):
        # 创建一个大小为 3x3 的全为1的张量作为输入
        input = torch.ones(3, 3)
        # 设定 fork 的数量为 10
        num_forks = 10
        # 使用 torch.jit.script 将 fork_wait_graph3 脚本化为图形
        torch_graph = torch.jit.script(fork_wait_graph3)
        # 在脚本化的图形上执行输入和 fork 数量，获得参考输出
        output_ref = torch_graph(input, num_forks)
        # 创建一个 StaticModule 实例，封装脚本化的图形
        static_runtime_module = StaticModule(torch_graph)
        # 在静态运行模块上异步运行输入和 fork 数量，获得测试输出
        output_test = static_runtime_module.runAsync((input, num_forks), {})
        # 等待异步操作完成
        output_test.wait()
        # 使用 torch.testing.assert_close 断言测试输出与参考输出的近似程度
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    Test Case: To test fork/wait operation in a graph on
    multiple nested fork/wait operations
    """
    # 跳过这个测试，因为存在已知的问题 https://github.com/pytorch/pytorch/issues/109782
    @unittest.skip("Broken test: https://github.com/pytorch/pytorch/issues/109782")
    def test_fork_wait_4(self):
        # 创建一个大小为 3x3 的全为1的张量作为输入
        input = torch.ones(3, 3)
        # 设定第一层 fork 的数量为 10
        num_forks = 10
        # 设定第二层 fork 的数量为 10
        num_child_forks = 10
        # 使用 torch.jit.script 将 fork_wait_graph4 脚本化为图形
        torch_graph = torch.jit.script(fork_wait_graph4)
        # 创建一个 StaticModule 实例，封装脚本化的图形
        static_runtime_module = StaticModule(torch_graph)
        # 在脚本化的图形上执行输入、第一层 fork 数量和第二层 fork 数量，获得参考输出
        output_ref = torch_graph(input, num_forks, num_child_forks)
        # 在静态运行模块上执行输入、第一层 fork 数量和第二层 fork 数量，获得测试输出
        output_test = static_runtime_module(input, num_forks, num_child_forks)
        # 使用 torch.testing.assert_close 断言测试输出与参考输出的近似程度
        torch.testing.assert_close(output_test, output_ref)

    """
    Test Case: To test fork/wait operation in a graph with multiple
    nested fork/wait operations on runAsync API returning future
    """
    # 跳过这个测试，因为存在已知的问题 https://github.com/pytorch/pytorch/issues/109782
    def test_fork_wait_4_async(self):
        # 创建一个大小为 3x3 的全为1的张量作为输入
        input = torch.ones(3, 3)
        # 设定第一层 fork 的数量为 10
        num_forks = 10
        # 设定第二层 fork 的数量为 10
        num_child_forks = 10
        # 使用 torch.jit.script 将 fork_wait_graph4 脚本化为图形
        torch_graph = torch.jit.script(fork_wait_graph4)
        # 创建一个 StaticModule 实例，封装脚本化的图形
        static_runtime_module = StaticModule(torch_graph)
        # 在脚本化的图形上执行输入、第一层 fork 数量和第二层 fork 数量，获得参考输出
        output_ref = torch_graph(input, num_forks, num_child_forks)
        # 在静态运行模块上异步运行输入、第一层 fork 数量和第二层 fork 数量，获得测试输出
        output_test = static_runtime_module.runAsync(
            (input, num_forks, num_child_forks), {})
        # 等待异步操作完成
        output_test.wait()
        # 使用 torch.testing.assert_close 断言测试输出与参考输出的近似程度
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    Test Case: To test exception handling in fork/wait
    operation. Add.Tensor op is called for tensors with
    non-matching dims on the forked subgraph and the
    exception raised by subgraph is set on future returned
    by prim::fork to parent graph. Returned exception is
    checked for substring expected_error_msg as declared below
    """
    """
    Test Case: To test exception handling in fork/wait
    operation with runAsync API. Add.Tensor op is called for
    tensors with non-matching dims on the forked subgraph
    and the exception raised by subgraph is set on future returned
    by prim::fork to parent graph. Returned exception is
    checked for substring expected_error_msg as declared below
    """
    # 定义一个测试函数，用于测试在fork/wait操作中的异常处理情况
    # 使用runAsync API。在forked子图上调用Add.Tensor操作，
    # 对具有不匹配维度的张量进行操作，子图引发的异常被设置在由prim::fork返回给父图的future中。
    # 返回的异常被检查是否包含下面声明的expected_error_msg子字符串。
    def test_fork_wait_exception_async(self):
        # 创建两个随机张量，维度不匹配
        input1 = torch.randn(4, 7)
        input2 = torch.randn(4, 5)
        # 使用torch.jit.script将fork_wait_graph_exception转换为Torch脚本图
        torch_graph = torch.jit.script(fork_wait_graph_exception)
        try:
            # 创建StaticModule对象，传入Torch脚本图
            static_runtime_module = StaticModule(torch_graph)
            # 调用runAsync方法执行模块，传入input1和input2作为参数，空字典作为属性
            output_test = static_runtime_module.runAsync(
                (input1, input2), {})
        except Exception as error:
            # 定义预期的错误消息，描述张量维度不匹配的情况
            expected_error_msg = (
                "The size of tensor a (7) must match the size "
                "of tensor b (5) at non-singleton dimension 1"
            )
            # 如果捕获的异常错误消息中不包含预期的子字符串，抛出运行时异常
            if str(error).find(expected_error_msg) == -1:
                raise RuntimeError(
                    "Tried execution of add.Tensors with incompatible shape. "
                    "Exception raised by forked runtime execution does "
                    f'not contain expected substring: "{expected_error_msg}"'
                ) from error
    # 定义一个测试函数，用于测试多头注意力层的功能
    def test_multihead_attention_layer(self):
        # 定义隐藏单元维度
        HID_DIM = 256
        # 定义查询长度
        QUERY_LEN = 8
        # 定义批大小
        BATCH_SIZE = 128
        # 定义层数
        LAYERS = 3
        # 定义注意力头数
        HEADS = 8
        # 定义dropout率
        DROPOUT = 0.1
        # 设置设备为CPU
        device = torch.device("cpu")
        
        # 创建多头注意力层对象，并移动到指定设备上
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        
        # 使用torch.no_grad()上下文，生成随机输入张量src，并移动到指定设备上
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        
        # 生成用于掩码的张量，仅保留大于0的元素，将结果移动到指定设备上
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        # 将attention设置为评估模式
        attention.eval()
        # 使用torch.jit.script将attention转换为脚本模式
        attention = torch.jit.script(attention)
        # 再次将attention设置为评估模式
        attention.eval()
        
        # 使用StaticModule封装attention，生成attention_a对象
        attention_a = StaticModule(attention)
        
        # 使用封装后的attention_a对象进行前向计算，计算结果o_test与o_test_kw
        o_test = attention_a(src, src, src, src_mask)
        o_test_kw = attention_a(src, src, value=src, mask=src_mask)
        
        # 遍历参考结果o_ref和测试结果o_test，逐一进行数值比较
        for a, b in zip(o_ref, o_test):
            torch.testing.assert_close(a, b)

        # 遍历参考结果o_ref和测试结果o_test_kw，逐一进行数值比较
        for a, b in zip(o_ref, o_test_kw):
            torch.testing.assert_close(a, b)

    # 定义一个性能基准测试函数，用于测试多头注意力层的运行性能
    def test_multihead_attention_layer_benchmark(self):
        # 定义隐藏单元维度
        HID_DIM = 256
        # 定义查询长度
        QUERY_LEN = 8
        # 定义批大小
        BATCH_SIZE = 128
        # 定义层数
        LAYERS = 3
        # 定义注意力头数
        HEADS = 8
        # 定义dropout率
        DROPOUT = 0.1
        # 设置设备为CPU
        device = torch.device("cpu")
        
        # 创建多头注意力层对象，并移动到指定设备上
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        
        # 使用torch.no_grad()上下文，生成随机输入张量src，并移动到指定设备上
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        
        # 生成用于掩码的张量，仅保留大于0的元素，将结果移动到指定设备上
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        # 将attention设置为评估模式
        attention.eval()
        # 使用torch.jit.script将attention转换为脚本模式
        attention = torch.jit.script(attention)
        
        # 使用StaticModule封装attention，生成attention_a对象
        attention_a = StaticModule(attention)
        
        # 使用attention_a对象进行多头注意力层的基准测试，记录性能指标
        attention_a.benchmark([src, src, src, src_mask], {}, 2, 2)
        
        # 使用attention_a对象分别基准测试多头注意力层各个操作的性能，记录各操作的性能指标
        metrics = attention_a.benchmark_individual_ops([src, src, src, src_mask], {}, 2, 2)
    # 测试多层感知器（MLP）模型
    def test_mlp(self):
        # 从基准脚本 ./bench/dlrm_s_benchmark.sh 中获取的参数
        ln_bot = [512, 512, 64]  # 底层 MLP 模型的各层神经元数目列表
        sigmoid_bot = -1  # 底层 MLP 模型的激活函数类型
        ln_top = [100, 1024, 1024, 1024, 1]  # 顶层 MLP 模型的各层神经元数目列表
        sigmoid_top = 3  # 顶层 MLP 模型的激活函数类型
        bot_l = create_mlp(ln_bot, sigmoid_bot)  # 创建底层 MLP 模型
        bot_l_acc = StaticModule(bot_l)  # 将底层 MLP 模型封装为静态模块
        top_l = create_mlp(ln_top, sigmoid_top)  # 创建顶层 MLP 模型
        top_l_acc = StaticModule(top_l)  # 将顶层 MLP 模型封装为静态模块
        with torch.no_grad():
            bot_inp = torch.randn(2048, 512)  # 创建随机输入数据，形状为 [2048, 512]
            top_inp = torch.randn(2048, 100)  # 创建随机输入数据，形状为 [2048, 100]
        ref_bot = bot_l(bot_inp)  # 应用底层 MLP 模型到输入数据，计算参考输出
        acc_bot = bot_l_acc(bot_inp)  # 应用封装后的底层 MLP 模型到输入数据，计算加速器输出
        torch.testing.assert_close(acc_bot, ref_bot)  # 断言加速器输出与参考输出的近似程度
        ref_top = top_l(top_inp)  # 应用顶层 MLP 模型到输入数据，计算参考输出
        acc_top = top_l_acc(top_inp)  # 应用封装后的顶层 MLP 模型到输入数据，计算加速器输出
        torch.testing.assert_close(acc_top, ref_top)  # 断言加速器输出与参考输出的近似程度
        for _ in range(5):
            with torch.no_grad():
                bot_inp = torch.randn(2048, 512)  # 创建随机输入数据，形状为 [2048, 512]
                top_inp = torch.randn(2048, 100)  # 创建随机输入数据，形状为 [2048, 100]
            ref_bot = bot_l(bot_inp)  # 应用底层 MLP 模型到输入数据，计算参考输出
            acc_bot = bot_l_acc(bot_inp)  # 应用封装后的底层 MLP 模型到输入数据，计算加速器输出
            torch.testing.assert_close(acc_bot, ref_bot)  # 断言加速器输出与参考输出的近似程度
            ref_top = top_l(top_inp)  # 应用顶层 MLP 模型到输入数据，计算参考输出
            acc_top = top_l_acc(top_inp)  # 应用封装后的顶层 MLP 模型到输入数据，计算加速器输出
            torch.testing.assert_close(acc_top, ref_top)  # 断言加速器输出与参考输出的近似程度

    # 测试简单图形
    def test_trivial_graph(self):
        s = torch.full((2, 2), 2)  # 创建全为 2 的张量，形状为 [2, 2]
        tg = torch.jit.script(trivial_graph)  # 使用 Torch JIT 脚本化简单图形函数
        o_ref = tg(s, s, s)  # 应用脚本化图形函数到输入张量 s, s, s，计算参考输出
        tg_a = StaticModule(tg)  # 将脚本化图形函数封装为静态模块
        o_test = tg_a(s, s, s)  # 应用封装后的脚本化图形函数到输入张量 s, s, s，计算测试输出
        torch.testing.assert_close(o_ref, o_test)  # 断言测试输出与参考输出的近似程度

    # 测试泄漏整流线性单元（Leaky ReLU）
    def test_leaky_relu(self):
        s = torch.randn(5, 5)  # 创建形状为 [5, 5] 的随机输入张量
        tg = torch.jit.script(nn.LeakyReLU(0.1))  # 使用 Torch JIT 脚本化泄漏整流线性单元
        o_ref = tg(s)  # 应用脚本化泄漏整流线性单元到输入张量 s，计算参考输出
        tg_a = StaticModule(tg)  # 将脚本化泄漏整流线性单元封装为静态模块
        o_test = tg_a(s)  # 应用封装后的脚本化泄漏整流线性单元到输入张量 s，计算测试输出
        torch.testing.assert_close(o_ref, o_test)  # 断言测试输出与参考输出的近似程度
    def test_attr(self):
        """
        TorchScript IR of TestModule() after freezing:
        graph(%self : __torch__.test_static_runtime.___torch_mangle_0.TestModule,
              %x.1 : Tensor):
            %18 : int = prim::Constant[value=30]()
            %30 : int = prim::Constant[value=13]()
            %3 : int = prim::Constant[value=20]()
            %2 : int = prim::Constant[value=1]()
            %self.sub2.a : int = prim::Constant[value=12]()
            %self.a : int = prim::Constant[value=3]()
            = prim::SetAttr[name="b"](%self, %3)  # 设置 self 对象的属性 b 为常量 20
            %17 : Tensor = aten::add(%x.1, %30, %2)  # 将输入张量 %x.1 和常量 13 相加
            %7 : Tensor = aten::add(%17, %self.a, %2)  # 将结果 %17 和 self 对象的属性 a 相加
            %b.1 : int = prim::GetAttr[name="b"](%self)  # 获取 self 对象的属性 b 的值
            %9 : Tensor = aten::add(%7, %b.1, %2)  # 将结果 %7 和属性 b 的值相加
            %sub2 : __torch__.test_static_runtime.___torch_mangle_2.SubModule2 = prim::GetAttr[name="sub2"](%self)
            = prim::SetAttr[name="b"](%sub2, %18)  # 设置 sub2 对象的属性 b 为常量 30
            %b : int = prim::GetAttr[name="b"](%sub2)  # 获取 sub2 对象的属性 b 的值
            %22 : int = aten::add(%self.sub2.a, %b)  # 将 self.sub2.a 和属性 b 的值相加
            %23 : Tensor = aten::add(%x.1, %22, %2)  # 将输入张量 %x.1 和结果 %22 相加
            %12 : Tensor = aten::add(%9, %23, %2)  # 将结果 %9 和 %23 相加
            return (%12)  # 返回最终结果 %12
        """
        # test prim::SetAttr and prim::GetAttr impl in Static Runtime
        m = TestModule()  # 创建 TestModule 实例 m

        m.eval()  # 将 m 设置为评估模式
        input = torch.randn(2, 2)  # 创建一个形状为 (2, 2) 的随机张量 input
        output_s = m.forward(input)  # 使用 m 对象进行前向传播，得到输出 output_s

        ms = torch.jit.script(m)  # 对 m 进行 TorchScript 脚本化，得到 ms
        sm = StaticModule(ms)  # 使用 TorchScript 脚本化后的模块创建 StaticModule 实例 sm
        output_sm = sm(input)  # 使用 sm 进行前向传播，得到输出 output_sm
        torch.testing.assert_close(output_s, output_sm)  # 断言 output_s 和 output_sm 的值接近

        sm.benchmark([input], {}, 2, 2)  # 对 sm 进行基准测试
        sm.benchmark_individual_ops([input], {}, 2, 2)  # 对 sm 进行各操作单独基准测试
        sm.benchmark([], {"x": input}, 2, 2)  # 对 sm 进行基准测试，传入额外的输入参数 x
        sm.benchmark_individual_ops([], {"x": input}, 2, 2)  # 对 sm 进行各操作单独基准测试，传入额外的输入参数 x

    @unittest.skip("Temporarily disabled")
    def test_fusion_trivial_graph(self):
        s = torch.full((2, 2), 2)  # 创建一个形状为 (2, 2) 的张量，所有元素均为 2
        tg = torch.jit.script(trivial_graph)  # 对 trivial_graph 进行 TorchScript 脚本化，得到 tg
        o_ref = tg(s, s, s)  # 使用 tg 对 s 进行计算得到输出 o_ref
        torch._C._fuse_to_static_module(tg.graph)  # 将 tg 的图融合为静态模块

        assert "StaticSubgraph" in str(tg.graph)  # 断言 tg 的图中包含 "StaticSubgraph"

        o_test = tg(s, s, s)  # 再次使用 tg 进行计算得到输出 o_test
        torch.testing.assert_close(o_ref, o_test)  # 断言 o_ref 和 o_test 的值接近

    @unittest.skip("Temporarily disabled")
    def test_fusion_multihead_attention_layer(self):
        HID_DIM = 256  # 隐藏维度为 256
        QUERY_LEN = 8  # 查询长度为 8
        BATCH_SIZE = 128  # 批量大小为 128
        LAYERS = 3  # 层数为 3
        HEADS = 8  # 头数为 8
        DROPOUT = 0.1  # 丢弃率为 0.1
        device = torch.device("cpu")  # 使用 CPU 设备
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)  # 创建多头注意力层对象 attention，并移动到指定设备

        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)  # 创建形状为 (BATCH_SIZE, QUERY_LEN, HID_DIM) 的随机张量 src，并移动到指定设备
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)  # 根据 src 创建掩码张量 src_mask

        attention.eval()  # 将 attention 设置为评估模式
        attention = torch.jit.script(attention)  # 对 attention 进行 TorchScript 脚本化
        attention.eval()  # 将 attention 设置为评估模式
        o_ref = attention(src, src, src, src_mask)  # 使用 attention 进行前向传播，得到输出 o_ref

        torch._C._fuse_to_static_module(attention._c)  # 将 attention 的图融合为静态模块
        o_test = attention(src, src, src, src_mask)  # 再次使用 attention 进行前向传播，得到输出 o_test

        for a, b in zip(o_ref, o_test):
            torch.testing.assert_close(a, b)  # 断言 o_ref 和 o_test 中的每个元素的值接近
    @unittest.skip("Temporarily disabled")
    # 标记为暂时禁用，不运行此单元测试
    def test_fusion_loop(self):
        # 创建随机张量 a 和 b
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        # 设置常数 c
        c = 4
        # 使用 torch.jit.script 将函数 loop_graph 转换为 Torch 脚本
        lg = torch.jit.script(loop_graph)
        # 调用转换后的 Torch 脚本函数 lg，传入参数 a, b, c
        o_ref = lg(a, b, c)
        # 将 lg.graph 融合为静态模块
        torch._C._fuse_to_static_module(lg.graph)
        # 断言静态子图 "StaticSubgraph" 存在于 lg.graph 中
        assert "StaticSubgraph" in str(lg.graph)
        # 再次调用转换后的 Torch 脚本函数 lg，传入参数 a, b, c
        o_test = lg(a, b, c)
        # 断言 o_ref 与 o_test 的值在允许误差范围内相等
        torch.testing.assert_close(o_ref, o_test)

    @unittest.skip("Temporarily disabled")
    # 标记为暂时禁用，不运行此单元测试
    def test_fusion_outputs(self):
        # 创建随机张量 a 和 b
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        # 设置常数 c
        c = 4
        # 使用 torch.jit.script 将函数 output_graph 转换为 Torch 脚本
        og = torch.jit.script(output_graph)
        # 调用转换后的 Torch 脚本函数 og，传入参数 a, b, b, c
        o_ref = og(a, b, b, c)
        # 将 og.graph 融合为静态模块
        torch._C._fuse_to_static_module(og.graph)
        # 断言静态子图 "StaticSubgraph" 存在于 og.graph 中
        assert "StaticSubgraph" in str(og.graph)
        # 再次调用转换后的 Torch 脚本函数 og，传入参数 a, b, b, c
        o_test = og(a, b, b, c)
        # 对 o_ref 和 o_test 中的每个键进行比较，确保值在允许误差范围内相等
        for i in o_ref.keys():
            torch.testing.assert_close(o_ref[i], o_test[i])

    def test_create_object(self):
        # 定义一个 Foo 类，用于包装张量 x
        class Foo:  # noqa: B903
            def __init__(self, x: torch.Tensor) -> None:
                self.x = x

        # 定义一个 Mod 类，继承自 torch.nn.Module
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                # 创建一个 Foo 对象 foo，传入张量 y
                foo = Foo(y)
                # 返回 y 乘以 foo.x 的结果
                return y * foo.x

        # 使用 torch.jit.script 将 Mod 实例转换为 Torch 脚本并设为评估模式
        mod = torch.jit.script(Mod()).eval()
        # 创建一个随机张量 y
        y = torch.randn((1, ))
        # 计算预期输出
        expected = mod(y)

        # 使用 torch.jit.freeze 冻结 mod，然后传递给 StaticModule 类构造函数
        static_mod = StaticModule(torch.jit.freeze(mod))
        # 计算 static_mod 对输入 y 的输出
        actual = static_mod(y)

        # 断言预期输出与实际输出相等
        self.assertEqual(expected, actual)
# 如果当前脚本被直接运行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```