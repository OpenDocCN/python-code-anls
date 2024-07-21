# `.\pytorch\test\test_content_store.py`

```py
# Owner(s): ["oncall: pt2"]

# 导入必要的模块和库
import tempfile
import unittest

import torch
from torch._prims.debug_prims import load_tensor_reader
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfRocm,
    TestCase,
)
from torch.utils._content_store import (
    ContentStoreReader,
    ContentStoreWriter,
    hash_storage,
)

# 定义一个测试类，用于测试ContentStore相关功能
@unittest.skipIf(IS_WINDOWS, "Test case not supported on Windows")
class TestContentStore(TestCase):
    # 测试基本功能
    def test_basic(self, device):
        # 设置测试数据
        x = torch.randn(4, device=device)
        y = torch.randn(6, device=device)
        z = x.view(2, 2)
        
        # 开始写入数据
        with tempfile.TemporaryDirectory() as loc:
            # 创建ContentStoreWriter对象，并写入tensor数据
            writer = ContentStoreWriter(loc)
            writer.write_tensor("x", x)
            writer.write_tensor("y", y)
            writer.write_tensor("z", z)
            
            # 进行一些VC UNTRACKED的变异操作
            x.data.add_(1)
            writer.write_tensor("x2", x)
            writer.write_tensor("y2", y)
            writer.write_tensor("z2", z)
            
            # 关闭writer
            del writer

            # 创建ContentStoreReader对象，并读取之前写入的tensor数据
            reader = ContentStoreReader(loc)
            n_x = reader.read_tensor("x")
            n_y = reader.read_tensor("y")
            n_z = reader.read_tensor("z")
            
            # 断言读取的数据与原数据是否一致
            self.assertEqual(n_x + 1, x)
            self.assertEqual(n_y, y)
            self.assertEqual(n_z + 1, z)
            
            # 断言存储的弱引用是否相同
            self.assertEqual(
                StorageWeakRef(n_x.untyped_storage()),
                StorageWeakRef(n_z.untyped_storage()),
            )
            
            # 继续读取其他写入的tensor数据，并进行断言
            n_x2 = reader.read_tensor("x2")
            n_y2 = reader.read_tensor("y2")
            n_z2 = reader.read_tensor("z2")
            
            self.assertEqual(n_x2, x)
            self.assertEqual(n_y2, y)
            self.assertEqual(n_z2, z)
            
            # 断言另一个存储的弱引用是否相同
            self.assertEqual(
                StorageWeakRef(n_y2.untyped_storage()),
                StorageWeakRef(n_y.untyped_storage()),
            )

    # 测试标量数据
    def test_scalar(self, device):
        # 不应该引发错误
        hash_storage(torch.tensor(2, device=device).untyped_storage())

    # 测试重复哈希
    @torch._dynamo.config.patch(cache_size_limit=1)
    def test_repeated_hash(self, device):
        # 测试重复哈希不会触发动态编译
        # 如果触发了，会在eager模式下执行prims.xor_sum，会导致失败
        for _ in range(4):
            hash_storage(torch.tensor(2, device=device).untyped_storage())

    # 跳过在Rocm上的测试
    @skipIfRocm
    # 定义一个测试函数，用于测试加载张量的功能，接受一个设备参数
    def test_load_tensor(self, device):
        # 使用临时目录作为数据存储的位置
        with tempfile.TemporaryDirectory() as loc:
            # 创建一个ContentStoreWriter对象，用于写入数据到指定位置loc
            writer = ContentStoreWriter(loc)
            # 生成一个随机张量x，指定设备为参数device
            x = torch.randn(4, device=device)

            # 定义一个内部函数same_meta_as_x，用于比较张量t与张量x的元数据是否相同
            def same_meta_as_x(t):
                self.assertEqual(t.size(), x.size())  # 检查张量尺寸是否相同
                self.assertEqual(t.stride(), x.stride())  # 检查张量步长是否相同
                self.assertEqual(t.dtype, x.dtype)  # 检查张量数据类型是否相同
                self.assertEqual(t.device, x.device)  # 检查张量设备是否相同

            # 将张量x写入到临时目录中的内容存储
            writer.write_tensor("x", x)

            # 使用load_tensor_reader上下文管理器加载存储中的张量数据
            with load_tensor_reader(loc):
                # 从存储中加载名为"x"的张量x2，并与原始张量x进行比较
                x2 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float32, device=device
                )
                self.assertEqual(x, x2)  # 检查加载的张量与原始张量是否相等
                # 再次加载"x"张量，与原始张量x进行比较
                x3 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float32, device=device
                )
                self.assertEqual(x, x3)  # 检查加载的第二个张量与原始张量是否相等
                # 必须确保不会发生别名现象！
                self.assertNotEqual(
                    StorageWeakRef(x.untyped_storage()),
                    StorageWeakRef(x2.untyped_storage()),
                )  # 检查两个张量的存储是否不同
                self.assertNotEqual(
                    StorageWeakRef(x2.untyped_storage()),
                    StorageWeakRef(x3.untyped_storage()),
                )  # 检查第二个加载的张量与第一个加载的张量的存储是否不同

                # 检查虚拟张量模式也能正常工作
                with FakeTensorMode():
                    # 从存储中加载名为"x"的张量x4，并检查其类型是否为FakeTensor
                    x4 = torch.ops.debugprims.load_tensor.default(
                        "x", (4,), (1,), dtype=torch.float32, device=device
                    )
                    self.assertIsInstance(x4, FakeTensor)  # 检查加载的张量是否为FakeTensor类型
                    same_meta_as_x(x4)  # 检查加载的FakeTensor的元数据是否与原始张量x相同

                # 检查加载双精度浮点数张量是否工作
                x5 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float64, device=device
                )
                self.assertEqual(x5.float(), x)  # 检查加载的双精度张量与原始张量x的数值是否相等
                self.assertEqual(x5.dtype, torch.float64)  # 检查加载的双精度张量的数据类型是否为torch.float64

        # 在离开临时目录上下文后，再次加载名为"x"的张量x6，并与原始张量x进行比较
        x6 = torch.ops.debugprims.load_tensor.default(
            "x", (4,), (1,), dtype=torch.float32, device=device
        )
        same_meta_as_x(x6)  # 检查加载的张量x6与原始张量x的元数据是否相同
# 调用函数 `instantiate_device_type_tests`，用于实例化设备类型测试，传入 `TestContentStore` 类和全局作用域
instantiate_device_type_tests(TestContentStore, globals())

# 如果当前脚本作为主程序运行，则执行 `run_tests()` 函数
if __name__ == "__main__":
    run_tests()
```