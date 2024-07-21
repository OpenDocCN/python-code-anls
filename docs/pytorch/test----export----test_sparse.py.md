# `.\pytorch\test\export\test_sparse.py`

```py
# Owner(s): ["module: sparse"]
#
# Test to ensure sparsity information propagates properly into traced graph.
#

import sys  # 导入 sys 模块，用于访问系统相关功能
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 深度学习库

from torch._subclasses.fake_tensor import FakeTensor  # 导入 FakeTensor 类，用于处理伪张量
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
    subtest,  # 导入子测试装饰器
    TestCase,  # 导入测试用例基类
)

# Various data types (preserved over operations).
DTYPES = [
    torch.int64,  # 64 位整数数据类型
    torch.float16,  # 16 位浮点数数据类型
    torch.bfloat16,  # Bfloat16 数据类型
    torch.float32,  # 32 位浮点数数据类型
    torch.float64,  # 64 位浮点数数据类型
]

# Various index types.
ITYPES = [torch.int32, torch.int64]  # 不同的索引类型，32 位整数和 64 位整数


# Constructs a subtest for every sparse layout currently supported in torch.sparse.
def all_sparse_layouts(test_name="layout"):
    """
    根据当前 torch.sparse 支持的每种稀疏布局构造一个子测试。
    
    Args:
    - test_name (str): 测试名称，默认为 "layout"
    
    Returns:
    - callable: 参数化装饰器，用于构造子测试
    """
    return parametrize(
        test_name,
        [
            subtest(torch.sparse_coo, name="SparseCOO"),  # 使用 subtest 创建 SparseCOO 子测试
            subtest(torch.sparse_csr, name="SparseCSR"),  # 使用 subtest 创建 SparseCSR 子测试
            subtest(torch.sparse_csc, name="SparseCSC"),  # 使用 subtest 创建 SparseCSC 子测试
            subtest(torch.sparse_bsr, name="SparseBSR"),  # 使用 subtest 创建 SparseBSR 子测试
            subtest(torch.sparse_bsc, name="SparseBSC"),  # 使用 subtest 创建 SparseBSC 子测试
        ],
    )


#
# Various network examples.
#


class IdNet(torch.nn.Module):
    """
    简单的身份网络示例，直接返回输入。
    """
    def forward(self, x):
        return x


class SumNet(torch.nn.Module):
    """
    求和网络示例，计算输入张量的元素和。
    """
    def forward(self, x):
        return x.sum()


class EltwiseNet(torch.nn.Module):
    """
    元素级操作网络示例，应用 ReLU 激活函数。
    """
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()  # 实例化 ReLU 激活函数

    def forward(self, x):
        return self.relu(2 * torch.abs(-x))


class SparseActivationCOO(torch.nn.Module):
    """
    稀疏激活 COO 网络示例，将输入张量转换为稀疏张量。
    """
    def forward(self, x):
        return [xi.to_sparse() for xi in x]


# TODO: ensure this case work too
class SparseActivationCSR(torch.nn.Module):
    """
    稀疏激活 CSR 网络示例，将输入张量转换为 CSR 格式的稀疏张量。
    """
    def forward(self, x):
        return [xi.to_sparse_csr() for xi in x]


#
# The test driver.
#


class TestSparseProp(TestCase):
    """
    稀疏传播测试用例类，继承自 unittest 的 TestCase 类。
    """
    def setUp(self):
        TestCase.setUp(self)  # 执行父类 TestCase 的 setUp 方法
    # 断言两个对象的类型，确保 x 是 FakeTensor 类型，y 是 torch.Tensor 类型
    self.assertIsInstance(x, FakeTensor)
    self.assertIsInstance(y, torch.Tensor)

    # 将 y 转换为 "meta" 设备上的张量，用于后续比较
    y = y.to("meta")

    # 使用 assertEqual 方法比较 x 和 y 的值，确保精确匹配布局和是否合并
    self.assertEqual(x, y, exact_layout=True, exact_is_coalesced=True)

    # 当 x 或 y 是 meta 张量时（例如 `x.device == "meta"`），assertEqual(x, y)
    # 只比较 x 和 y 的属性，而不比较它们的值。对于稀疏张量，这意味着跳过比较索引和值属性，
    # 这就是为什么我们在下面明确执行这些操作的原因。
    if x.layout is torch.strided:
        pass
    elif x.layout is torch.sparse_coo:
        # 对于 COO 格式的稀疏张量，比较索引和值属性的布局
        self.assertEqual(x._indices(), y._indices(), exact_layout=True)
        self.assertEqual(x._values(), y._values(), exact_layout=True)
    else:
        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            # 对于 CSR 或 BSR 格式的稀疏张量，比较行索引和列索引
            x_meta1, y_meta1 = (x.crow_indices(), y.crow_indices())
            x_meta2, y_meta2 = (x.col_indices(), y.col_indices())
        elif x.layout in {torch.sparse_csc, torch.sparse_bsc}:
            # 对于 CSC 或 BSC 格式的稀疏张量，比较列索引和行索引
            x_meta1, y_meta1 = (x.ccol_indices(), y.ccol_indices())
            x_meta2, y_meta2 = (x.row_indices(), y.row_indices())
        else:
            # 如果上述条件都不满足，输出错误信息，应该不会执行到这里
            assert 0  # unreachable

        # 使用 assertEqual 方法比较稀疏张量的属性，确保精确匹配布局
        self.assertEqual(x_meta1, y_meta1, exact_layout=True)
        self.assertEqual(x_meta2, y_meta2, exact_layout=True)
        self.assertEqual(x.values(), y.values(), exact_layout=True)

@unittest.skipIf(
    sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
)
@parametrize("dtype", DTYPES)
@parametrize("itype", ITYPES)
@all_sparse_layouts("layout")
def test_idnet(self, dtype, itype, layout):
    # 如果布局不是 COO 格式的稀疏张量，则跳过测试并输出相应信息
    if layout is not torch.sparse_coo:
        self.skipTest("TODO: support non-coo sparsity!")

    # 创建 IdNet 网络对象
    net = IdNet()

    # 使用 generate_simple_inputs 生成简单输入数据集
    for sparse_input in self.generate_simple_inputs(
        layout,
        device="cpu",
        dtype=dtype,
        index_dtype=itype,
    ):
        # 导出网络的追踪图
        prog = torch.export.export(net, (sparse_input,))

        # 测试参数和输出
        for i, node in enumerate(prog.graph.nodes):
            # 获取节点的元数据中的值
            meta = node.meta.get("val", None)

            # 如果是第一个节点，使用 assertEqualMeta 方法比较 meta 和 sparse_input
            if i == 0:
                self.assertEqualMeta(meta, sparse_input)
            else:
                # 否则，简单比较 meta 是否为 None
                self.assertEqual(meta, None)

@unittest.skipIf(
    sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
)
@parametrize("dtype", DTYPES)
@parametrize("itype", ITYPES)
@all_sparse_layouts("layout")
    # 测试 SumNet 类的功能，使用指定的数据类型、索引类型和布局
    def test_sumnet(self, dtype, itype, layout):
        # 如果布局不是稀疏的 COO 格式，则跳过测试并输出信息
        if layout is not torch.sparse_coo:
            self.skipTest("TODO: support non-coo sparsity!")

        # 创建 SumNet 类的实例
        net = SumNet()
        # 遍历生成简单输入的迭代器，每次迭代都使用指定的布局、设备、数据类型和索引类型
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            # 对网络进行前向传播，得到结果
            result = net(sparse_input)
            # 导出网络的追踪图
            prog = torch.export.export(net, (sparse_input,))
            # 测试参数/求和/输出
            for i, node in enumerate(prog.graph.nodes):
                # 获取节点的元数据中的值（如果有的话）
                meta = node.meta.get("val", None)
                if i == 0:
                    # 第一个节点应该与输入数据匹配
                    self.assertEqualMeta(meta, sparse_input)
                elif i == 1:
                    # 第二个节点应该与网络输出匹配
                    self.assertEqualMeta(meta, result)
                else:
                    # 其他节点的元数据应该为空
                    self.assertEqual(meta, None)

    # 在 Python 版本 >= 3.12 上跳过测试，因为 torch.compile 不支持这些版本
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    # 对每种数据类型进行参数化测试
    @parametrize("dtype", DTYPES)
    # 对每种索引类型进行参数化测试
    @parametrize("itype", ITYPES)
    # 对所有稀疏布局进行测试
    @all_sparse_layouts("layout")
    # 测试 EltwiseNet 类的功能，使用指定的数据类型、索引类型和布局
    def test_eltwisenet(self, dtype, itype, layout):
        # 如果布局不是稀疏的 COO 格式，则跳过测试并输出信息
        if layout is not torch.sparse_coo:
            self.skipTest("TODO: support non-coo sparsity!")

        # 创建 EltwiseNet 类的实例
        net = EltwiseNet()
        # 遍历生成简单输入的迭代器，每次迭代都使用指定的布局、设备、数据类型和索引类型
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            # 对网络进行前向传播，得到结果
            result = net(sparse_input)
            # 导出网络的追踪图
            prog = torch.export.export(net, (sparse_input,))
            # 测试参数/负数/绝对值/乘法/ReLU/输出
            for i, node in enumerate(prog.graph.nodes):
                # 获取节点的元数据中的值（如果有的话）
                meta = node.meta.get("val", None)
                if i <= 4:
                    # 前五个节点的元数据应该与网络输出匹配
                    self.assertEqualMeta(meta, result)
                else:
                    # 其他节点的元数据应该为空
                    self.assertEqual(meta, None)

    # 在 Python 版本 >= 3.12 上跳过测试，因为 torch.compile 不支持这些版本
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    # 测试 SparseActivationCOO 类的功能
    def test_activation_coo(self):
        # 创建 SparseActivationCOO 类的实例
        net = SparseActivationCOO()
        # 生成三个 3x3 的随机张量作为输入
        x = [torch.randn(3, 3) for _ in range(3)]
        # 对网络进行前向传播，得到结果
        result = net(x)
        # 导出网络的追踪图
        prog = torch.export.export(net, args=(x,))
        # 测试参数/转为稀疏张量/输出
        for i, node in enumerate(prog.graph.nodes):
            # 获取节点的元数据中的值（如果有的话）
            meta = node.meta.get("val", None)
            if i <= 2:
                # 前三个节点的元数据应该与对应的输入张量匹配
                self.assertEqualMeta(meta, x[i])
            elif i <= 5:
                # 接下来三个节点的元数据应该与网络输出匹配
                self.assertEqualMeta(meta, result[i - 3])
            else:
                # 其他节点的元数据应该为空
                self.assertEqual(meta, None)
# 调用函数 instantiate_parametrized_tests，传入 TestSparseProp 作为参数，用于实例化参数化测试
instantiate_parametrized_tests(TestSparseProp)

# 检查当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 如果是，则调用 run_tests() 函数执行测试
    run_tests()
```