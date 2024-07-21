# `.\pytorch\test\jit\test_sparse.py`

```py
# Owner(s): ["oncall: jit"]

import io  # 导入 io 模块，用于处理字节流
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 库
from torch.testing._internal.common_utils import IS_WINDOWS, TEST_MKL  # 导入测试工具和标志
from torch.testing._internal.jit_utils import JitTestCase  # 导入用于 JIT 编译的测试工具


class TestSparse(JitTestCase):
    def test_freeze_sparse_coo(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()  # 创建稀疏张量 self.a
                self.b = torch.rand(3, 4).to_sparse()  # 创建稀疏张量 self.b

            def forward(self, x):
                return x + self.a + self.b  # 返回输入张量 x 与稀疏张量 self.a、self.b 的和

        x = torch.rand(3, 4).to_sparse()  # 创建稀疏张量 x

        m = SparseTensorModule()  # 实例化 SparseTensorModule
        unfrozen_result = m.forward(x)  # 对未冻结模型进行前向传播计算

        m.eval()  # 设置模型为评估模式
        frozen = torch.jit.freeze(torch.jit.script(m))  # 对模型进行 JIT 编译并冻结

        frozen_result = frozen.forward(x)  # 对冻结模型进行前向传播计算

        self.assertEqual(unfrozen_result, frozen_result)  # 断言未冻结模型与冻结模型的结果一致

        buffer = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(frozen, buffer)  # 将冻结模型保存到字节流缓冲区
        buffer.seek(0)  # 将缓冲区指针设置到起始位置
        loaded_model = torch.jit.load(buffer)  # 从缓冲区加载模型

        loaded_result = loaded_model.forward(x)  # 对加载的模型进行前向传播计算

        self.assertEqual(unfrozen_result, loaded_result)  # 断言未冻结模型与加载模型的结果一致

    def test_serialize_sparse_coo(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()  # 创建稀疏张量 self.a
                self.b = torch.rand(3, 4).to_sparse()  # 创建稀疏张量 self.b

            def forward(self, x):
                return x + self.a + self.b  # 返回输入张量 x 与稀疏张量 self.a、self.b 的和

        x = torch.rand(3, 4).to_sparse()  # 创建稀疏张量 x
        m = SparseTensorModule()  # 实例化 SparseTensorModule
        expected_result = m.forward(x)  # 计算期望的前向传播结果

        buffer = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(torch.jit.script(m), buffer)  # 将 JIT 编译的模型保存到字节流缓冲区
        buffer.seek(0)  # 将缓冲区指针设置到起始位置
        loaded_model = torch.jit.load(buffer)  # 从缓冲区加载模型

        loaded_result = loaded_model.forward(x)  # 对加载的模型进行前向传播计算

        self.assertEqual(expected_result, loaded_result)  # 断言期望的结果与加载模型的结果一致

    @unittest.skipIf(IS_WINDOWS or not TEST_MKL, "Need MKL to run CSR matmul")
    def test_freeze_sparse_csr(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.rand(4, 4).to_sparse_csr()  # 创建 CSR 格式的稀疏张量 self.a
                self.b = torch.rand(4, 4).to_sparse_csr()  # 创建 CSR 格式的稀疏张量 self.b

            def forward(self, x):
                return x.matmul(self.a).matmul(self.b)  # 使用 CSR 稀疏张量进行矩阵乘法计算

        x = torch.rand(4, 4).to_sparse_csr()  # 创建 CSR 格式的稀疏张量 x

        m = SparseTensorModule()  # 实例化 SparseTensorModule
        unfrozen_result = m.forward(x)  # 对未冻结模型进行前向传播计算

        m.eval()  # 设置模型为评估模式
        frozen = torch.jit.freeze(torch.jit.script(m))  # 对模型进行 JIT 编译并冻结

        frozen_result = frozen.forward(x)  # 对冻结模型进行前向传播计算

        self.assertEqual(unfrozen_result.to_dense(), frozen_result.to_dense())  # 断言未冻结模型与冻结模型的结果密集化后一致

        buffer = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(frozen, buffer)  # 将冻结模型保存到字节流缓冲区
        buffer.seek(0)  # 将缓冲区指针设置到起始位置
        loaded_model = torch.jit.load(buffer)  # 从缓冲区加载模型

        loaded_result = loaded_model.forward(x)  # 对加载的模型进行前向传播计算

        self.assertEqual(unfrozen_result.to_dense(), loaded_result.to_dense())  # 断言未冻结模型与加载模型的结果密集化后一致
    def test_serialize_sparse_csr(self):
        # 定义一个测试方法，用于测试稀疏 CSR 格式的序列化和反序列化
        class SparseTensorModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的子类 SparseTensorModule
            def __init__(self):
                # 初始化方法，构造一个对象
                super().__init__()
                # 使用 torch.rand 方法生成一个 4x4 的随机稀疏 CSR 格式的张量 self.a
                self.a = torch.rand(4, 4).to_sparse_csr()
                # 使用 torch.rand 方法生成一个 4x4 的随机稀疏 CSR 格式的张量 self.b
                self.b = torch.rand(4, 4).to_sparse_csr()

            def forward(self, x):
                # 定义前向传播方法，对输入张量 x 进行操作
                # 执行 x 与 self.a 的矩阵乘法，再与 self.b 的矩阵乘法
                return x.matmul(self.a).matmul(self.b)

        # 生成一个 4x4 的随机稀疏 CSR 格式的张量 x
        x = torch.rand(4, 4).to_sparse_csr()
        # 创建 SparseTensorModule 类的一个实例 m
        m = SparseTensorModule()
        # 计算预期结果，调用 m 的 forward 方法传入 x
        expected_result = m.forward(x)

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 使用 torch.jit.script 方法将 SparseTensorModule 对象 m 脚本化并保存到 buffer 中
        torch.jit.save(torch.jit.script(m), buffer)
        # 将流的读写位置设为起始位置
        buffer.seek(0)
        # 从 buffer 中加载模型，得到 loaded_model
        loaded_model = torch.jit.load(buffer)

        # 使用加载后的模型执行前向传播，传入 x
        loaded_result = loaded_model.forward(x)

        # 使用单元测试框架中的断言方法，比较预期结果的稠密表示和加载后结果的稠密表示是否相等
        self.assertEqual(expected_result.to_dense(), loaded_result.to_dense())
```