# `.\pytorch\test\test_cpp_extensions_open_device_registration.py`

```
# Owner(s): ["module: cpp-extensions"]

# 导入必要的标准库和第三方库
import os
import shutil
import sys
import tempfile
import types
import unittest
from typing import Union
from unittest.mock import patch

# 导入 PyTorch 相关库和模块
import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import (
    IS_ARM64,
    skipIfTorchDynamo,
    TemporaryFileName,
    TEST_CUDA,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

# 更新 TEST_CUDA 和 TEST_ROCM 标志
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None

# 删除构建路径下的文件
def remove_build_path():
    if sys.platform == "win32":
        # Windows 平台下不清理扩展构建文件夹
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)

# 生成一个虚拟的模块对象
def generate_faked_module():
    # 定义模块中的一些虚拟函数
    def device_count() -> int:
        return 1

    def get_rng_state(device: Union[int, str, torch.device] = "foo") -> torch.Tensor:
        # 使用自定义设备对象创建一个张量
        return torch.empty(4, 4, device="foo")

    def set_rng_state(
        new_state: torch.Tensor, device: Union[int, str, torch.device] = "foo"
    ) -> None:
        pass

    def is_available():
        return True

    def current_device():
        return 0

    # 创建一个名为 "foo" 的新模块对象
    foo = types.ModuleType("foo")

    # 将虚拟函数绑定到模块对象中
    foo.device_count = device_count
    foo.get_rng_state = get_rng_state
    foo.set_rng_state = set_rng_state
    foo.is_available = is_available
    foo.current_device = current_device
    foo._lazy_init = lambda: None
    foo.is_initialized = lambda: True

    return foo

# 测试类，用于测试 C++ 扩展的开放设备注册
@unittest.skipIf(IS_ARM64, "Does not work on arm")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionOpenRgistration(common.TestCase):
    """Tests Open Device Registration with C++ extensions."""

    module = None

    def setUp(self):
        super().setUp()

        # C++ 扩展使用相对路径，这些路径相对于当前文件，所以暂时改变工作目录
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # 断言模块对象不为空
        assert self.module is not None

    def tearDown(self):
        super().tearDown()

        # 恢复原来的工作目录（参考 setUp 方法）
        os.chdir(self.old_working_dir)

    @classmethod
    # 在测试类运行前的准备工作，删除构建路径
    def setUpClass(cls):
        remove_build_path()

        # 加载自定义设备扩展模块
        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",  # 扩展模块的名称
            sources=[
                "cpp_extensions/open_registration_extension.cpp",  # 源文件列表
            ],
            extra_include_paths=["cpp_extensions"],  # 额外的包含路径
            extra_cflags=["-g"],  # 额外的编译标志
            verbose=True,  # 显示详细输出
        )

        # 注册 torch.foo 模块和 foo 设备到 torch
        torch.utils.rename_privateuse1_backend("foo")
        torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
        torch._register_device_module("foo", generate_faked_module())

    # 测试自定义设备注册的基本功能
    def test_base_device_registration(self):
        self.assertFalse(self.module.custom_add_called())
        # 使用我们的自定义设备对象创建张量
        device = self.module.custom_device()
        x = torch.empty(4, 4, device=device)
        y = torch.empty(4, 4, device=device)
        # 检查我们的设备是否正确
        self.assertTrue(x.device == device)
        self.assertFalse(x.is_cpu)
        self.assertFalse(self.module.custom_add_called())
        # 调用自定义的加法核函数，注册到分发器中
        z = x + y
        # 检查是否调用了自定义的加法核函数
        self.assertTrue(self.module.custom_add_called())
        z_cpu = z.to(device="cpu")
        # 检查跨设备复制是否正确将数据复制到 CPU
        self.assertTrue(z_cpu.is_cpu)
        self.assertFalse(z.is_cpu)
        self.assertTrue(z.device == device)
        self.assertEqual(z, z_cpu)
    # 定义一个测试方法，用于测试通用注册功能
    def test_common_registration(self):
        # 检查不支持的设备和重复注册情况，期待抛出 RuntimeError 异常并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("foo", generate_faked_module())

        # 可以多次将后端名称 "privateuse1" 重命名为 "foo"
        torch.utils.rename_privateuse1_backend("foo")

        # 不允许将后端名称 "privateuse1" 多次重命名为不同的名称，期待抛出 RuntimeError 异常并包含特定错误信息
        with self.assertRaisesRegex(
            RuntimeError, "torch.register_privateuse1_backend()"
        ):
            torch.utils.rename_privateuse1_backend("dev")

        # 生成器张量和模块只能注册一次，期待抛出 RuntimeError 异常并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            torch.utils.generate_methods_for_privateuse1_backend()

        # 检查是否正确注册了 torch.foo
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 1
        )
        # 尝试调用未注册的函数名 "torch.func_name_"，期待抛出 RuntimeError 异常并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.foo"):
            torch.utils.backend_registration._get_custom_mod_func("func_name_")

        # 检查注册后的属性是否存在
        self.assertTrue(hasattr(torch.Tensor, "is_foo"))
        self.assertTrue(hasattr(torch.Tensor, "foo"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_foo"))
        self.assertTrue(hasattr(torch.TypedStorage, "foo"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_foo"))
        self.assertTrue(hasattr(torch.UntypedStorage, "foo"))
        self.assertTrue(hasattr(torch.nn.Module, "foo"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_foo"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "foo"))
    # 测试自定义设备生成器注册和钩子
    def test_open_device_generator_registration_and_hooks(self):
        # 创建自定义设备对象
        device = self.module.custom_device()
        # 确保没有调用自定义加法函数
        self.assertFalse(self.module.custom_add_called())

        # 在使用之前检查是否已注册生成器
        with self.assertRaisesRegex(
            RuntimeError,
            "Please register a generator to the PrivateUse1 dispatch key",
        ):
            # 使用指定设备创建 Torch 生成器对象
            torch.Generator(device=device)

        # 注册第一个生成器到模块中
        self.module.register_generator_first()
        # 使用指定设备创建 Torch 生成器对象
        gen = torch.Generator(device=device)
        # 确保生成器的设备与预期一致
        self.assertTrue(gen.device == device)

        # 生成器只能注册一次
        with self.assertRaisesRegex(
            RuntimeError,
            "Only can register a generator to the PrivateUse1 dispatch key once",
        ):
            # 尝试第二次注册生成器，预期会抛出异常
            self.module.register_generator_second()

        # 注册钩子函数到模块中
        self.module.register_hook()
        # 使用默认生成器生成一个 Tensor
        default_gen = self.module.default_generator(0)
        # 确保默认生成器的设备类型与私有后端名称一致
        self.assertTrue(
            default_gen.device.type == torch._C._get_privateuse1_backend_name()
        )

    # 测试使用 dispatchstub 支持私有后端的核心功能
    def test_open_device_dispatchstub(self):
        # 创建一个 CPU 上的随机输入数据
        input_data = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        # 将输入数据转换到自定义设备 "foo"
        foo_input_data = input_data.to("foo")
        # 计算输入数据的绝对值
        output_data = torch.abs(input_data)
        # 计算转换到 "foo" 设备后的输入数据的绝对值
        foo_output_data = torch.abs(foo_input_data)
        # 检查两个输出数据是否一致（在 CPU 上比较）
        self.assertEqual(output_data, foo_output_data.cpu())

        # 创建一个新的随机输出数据
        output_data = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        # 将输入数据转换到自定义设备 "foo"
        foo_input_data = input_data.to("foo")
        # 将输出数据转换到自定义设备 "foo"
        foo_output_data = output_data.to("foo")
        # 在指定位置计算输入数据的绝对值，指定输出位置
        torch.abs(input_data, out=output_data[:, :, 0:6:2])
        torch.abs(foo_input_data, out=foo_output_data[:, :, 0:6:2])
        # 检查两个输出数据是否一致（在 CPU 上比较）
        self.assertEqual(output_data, foo_output_data.cpu())

        # 创建一个新的随机输出数据
        output_data = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        # 将输入数据转换到自定义设备 "foo"
        foo_input_data = input_data.to("foo")
        # 将输出数据转换到自定义设备 "foo"
        foo_output_data = output_data.to("foo")
        # 在指定位置计算输入数据的绝对值，指定输出位置
        torch.abs(input_data, out=output_data[:, :, 0:6:3])
        torch.abs(foo_input_data, out=foo_output_data[:, :, 0:6:3])
        # 检查两个输出数据是否一致（在 CPU 上比较）
        self.assertEqual(output_data, foo_output_data.cpu())

    # 测试量化操作是否正确在自定义设备上执行
    def test_open_device_quantized(self):
        # 创建一个在 CPU 上的随机输入数据，并转换到自定义设备 "foo"
        input_data = torch.randn(3, 4, 5, dtype=torch.float32, device="cpu").to("foo")
        # 对输入数据执行量化操作
        quantized_tensor = torch.quantize_per_tensor(input_data, 0.1, 10, torch.qint8)
        # 检查量化后的张量设备是否与预期一致
        self.assertEqual(quantized_tensor.device, torch.device("foo:0"))
        # 检查量化后的张量数据类型是否为 torch.qint8
        self.assertEqual(quantized_tensor.dtype, torch.qint8)
    # 测试在指定设备类型上使用随机数生成器的分叉
    def test_open_device_random(self):
        # 检查是否在 torch.foo 上实现了 get_rng_state 方法
        with torch.random.fork_rng(device_type="foo"):
            pass

    # 测试在自定义设备上使用张量操作
    def test_open_device_tensor(self):
        # 获取自定义设备
        device = self.module.custom_device()

        # 检查打印的张量类型是否符合期望
        dtypes = {
            torch.bool: "torch.foo.BoolTensor",
            torch.double: "torch.foo.DoubleTensor",
            torch.float32: "torch.foo.FloatTensor",
            torch.half: "torch.foo.HalfTensor",
            torch.int32: "torch.foo.IntTensor",
            torch.int64: "torch.foo.LongTensor",
            torch.int8: "torch.foo.CharTensor",
            torch.short: "torch.foo.ShortTensor",
            torch.uint8: "torch.foo.ByteTensor",
        }
        for tt, dt in dtypes.items():
            # 创建指定类型和设备的空张量
            test_tensor = torch.empty(4, 4, dtype=tt, device=device)
            self.assertTrue(test_tensor.type() == dt)

        # 检查是否正确生成了对应自定义后端的属性和方法
        x = torch.empty(4, 4)
        self.assertFalse(x.is_foo)

        # 调用自定义后端方法 foo，并验证方法调用
        x = x.foo(torch.device("foo"))
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(x.is_foo)

        # 测试不同设备类型的输入
        y = torch.empty(4, 4)
        self.assertFalse(y.is_foo)

        # 使用不同设备类型调用 foo 方法，并验证方法调用
        y = y.foo(torch.device("foo:0"))
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(y.is_foo)

        # 测试不同设备类型的输入
        z = torch.empty(4, 4)
        self.assertFalse(z.is_foo)

        # 使用整数设备类型调用 foo 方法，并验证方法调用
        z = z.foo(0)
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z.is_foo)

    # 测试在自定义设备上使用打包序列
    def test_open_device_packed_sequence(self):
        # 获取自定义设备
        device = self.module.custom_device()

        # 创建张量和长度，构建打包序列
        a = torch.rand(5, 3)
        b = torch.tensor([1, 1, 1, 1, 1])
        input = torch.nn.utils.rnn.PackedSequence(a, b)

        # 验证输入序列不是自定义后端生成的
        self.assertFalse(input.is_foo)

        # 调用自定义后端方法 foo，并验证方法调用
        input_foo = input.foo()
        self.assertTrue(input_foo.is_foo)
    # 定义一个测试方法，用于测试设备存储的相关属性和方法是否正确生成
    def test_open_device_storage(self):
        # 创建一个 4x4 的空张量 x
        x = torch.empty(4, 4)
        # 获取张量 x 的存储
        z1 = x.storage()
        # 断言 z1 的 is_foo 属性为 False
        self.assertFalse(z1.is_foo)

        # 调用 z1 的 foo 方法，返回新的存储 z1
        z1 = z1.foo()
        # 断言 self.module.custom_add_called() 返回 False
        self.assertFalse(self.module.custom_add_called())
        # 断言 z1 的 is_foo 属性为 True
        self.assertTrue(z1.is_foo)

        # 使用 with 语句断言调用 z1 的 foo 方法时会抛出 RuntimeError 异常，异常信息为 "Invalid device"
        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            z1.foo(torch.device("cpu"))

        # 将 z1 移到 CPU 设备上
        z1 = z1.cpu()
        # 断言 self.module.custom_add_called() 返回 False
        self.assertFalse(self.module.custom_add_called())
        # 断言 z1 的 is_foo 属性为 False
        self.assertFalse(z1.is_foo)

        # 调用 z1 的 foo 方法，指定 device="foo:0"，non_blocking=False，返回新的存储 z1
        z1 = z1.foo(device="foo:0", non_blocking=False)
        # 断言 self.module.custom_add_called() 返回 False
        self.assertFalse(self.module.custom_add_called())
        # 断言 z1 的 is_foo 属性为 True
        self.assertTrue(z1.is_foo)

        # 使用 with 语句断言调用 z1 的 foo 方法时会抛出 RuntimeError 异常，异常信息为 "Invalid device"
        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            z1.foo(device="cuda:0", non_blocking=False)

        # 创建一个 4x4 的空张量 y
        y = torch.empty(4, 4)
        # 获取张量 y 的未类型化存储
        z2 = y.untyped_storage()
        # 断言 z2 的 is_foo 属性为 False
        self.assertFalse(z2.is_foo)

        # 调用 z2 的 foo 方法，返回新的存储 z2
        z2 = z2.foo()
        # 断言 self.module.custom_add_called() 返回 False
        self.assertFalse(self.module.custom_add_called())
        # 断言 z2 的 is_foo 属性为 True
        self.assertTrue(z2.is_foo)

        # 调用 self.module.custom_storage_registry() 方法，检查自定义存储注册
        self.module.custom_storage_registry()

        # 再次获取张量 y 的未类型化存储
        z3 = y.untyped_storage()
        # 断言 self.module.custom_storageImpl_called() 返回 False
        self.assertFalse(self.module.custom_storageImpl_called())

        # 调用 z3 的 foo 方法，返回新的存储 z3
        z3 = z3.foo()
        # 断言 self.module.custom_storageImpl_called() 返回 True
        self.assertTrue(self.module.custom_storageImpl_called())
        # 断言 self.module.custom_storageImpl_called() 返回 False
        self.assertFalse(self.module.custom_storageImpl_called())

        # 对 z3 进行切片操作 [0:3]
        z3 = z3[0:3]
        # 断言 self.module.custom_storageImpl_called() 返回 True
        self.assertTrue(self.module.custom_storageImpl_called())
    # 定义测试函数，用于测试设备序列化相关功能
    def test_open_device_serialization(self):
        # 设置模块的自定义设备索引为 -1
        self.module.set_custom_device_index(-1)
        # 创建一个未命名存储对象，指定设备为 "foo"
        storage = torch.UntypedStorage(4, device=torch.device("foo"))
        # 验证存储对象的序列化位置标签为 "foo"
        self.assertEqual(torch.serialization.location_tag(storage), "foo")

        # 设置模块的自定义设备索引为 0
        self.module.set_custom_device_index(0)
        # 创建一个未命名存储对象，指定设备为 "foo"
        storage = torch.UntypedStorage(4, device=torch.device("foo"))
        # 验证存储对象的序列化位置标签为 "foo:0"
        self.assertEqual(torch.serialization.location_tag(storage), "foo:0")

        # 获取一个 CPU 上的张量的存储对象
        cpu_storage = torch.empty(4, 4).storage()
        # 将 CPU 上的存储对象恢复到 "foo:0" 设备
        foo_storage = torch.serialization.default_restore_location(cpu_storage, "foo:0")
        # 验证恢复后的存储对象确实位于 "foo" 设备上
        self.assertTrue(foo_storage.is_foo)

        # 测试张量元数据的序列化
        x = torch.empty(4, 4).long()
        # 获得张量的 "foo" 后端版本
        y = x.foo()
        # 验证模块未检查到张量的后端元数据
        self.assertFalse(self.module.check_backend_meta(y))
        # 将张量的后端元数据设置到模块中
        self.module.custom_set_backend_meta(y)
        # 验证模块成功检查到张量的后端元数据
        self.assertTrue(self.module.check_backend_meta(y))

        # 自定义序列化注册表
        self.module.custom_serialization_registry()
        
        # 使用临时目录存储数据文件
        with tempfile.TemporaryDirectory() as tmpdir:
            # 拼接临时目录和文件名
            path = os.path.join(tmpdir, "data.pt")
            # 将张量 y 保存到文件中
            torch.save(y, path)
            # 从文件中加载张量 z1
            z1 = torch.load(path)
            # 验证 z1 被正确加载到 "foo" 后端设备上
            self.assertTrue(z1.is_foo)
            # 验证模块成功检查到 z1 的后端元数据
            self.assertTrue(self.module.check_backend_meta(z1))

            # 跨后端加载
            # 从文件中加载张量 z2 到 "cpu" 设备上
            z2 = torch.load(path, map_location="cpu")
            # 验证 z2 被正确加载到 "cpu" 后端设备上
            self.assertFalse(z2.is_foo)
            # 验证模块成功检查到 z2 的后端元数据
            self.assertFalse(self.module.check_backend_meta(z2))

    # 定义测试函数，用于测试设备存储调整大小功能
    def test_open_device_storage_resize(self):
        # 创建一个在 CPU 上随机初始化的张量
        cpu_tensor = torch.randn([8])
        # 获得张量的 "foo" 后端版本
        foo_tensor = cpu_tensor.foo()
        # 获得 "foo" 后端版本的存储对象
        foo_storage = foo_tensor.storage()
        # 验证 "foo" 后端版本的存储对象大小为 8
        self.assertTrue(foo_storage.size() == 8)

        # 只注册张量的 resize_ 函数
        # 调整 "foo" 后端版本的张量大小为 8
        foo_tensor.resize_(8)
        # 验证 "foo" 后端版本的存储对象大小仍为 8
        self.assertTrue(foo_storage.size() == 8)

        # 预期抛出 TypeError 异常，因为尝试调整超出范围的大小
        with self.assertRaisesRegex(TypeError, "Overflow"):
            foo_tensor.resize_(8**29)
    def test_open_device_storage_type(self):
        # 测试 CPU 浮点型存储
        cpu_tensor = torch.randn([8]).float()
        cpu_storage = cpu_tensor.storage()
        self.assertEqual(cpu_storage.type(), "torch.FloatStorage")

        # 在定义 FloatStorage 之前测试自定义浮点型存储
        foo_tensor = cpu_tensor.foo()
        foo_storage = foo_tensor.storage()
        self.assertEqual(foo_storage.type(), "torch.storage.TypedStorage")

        # 定义 CustomFloatStorage 类
        class CustomFloatStorage:
            @property
            def __module__(self):
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        # 在定义 FloatStorage 之后测试自定义浮点型存储
        try:
            torch.foo.FloatStorage = CustomFloatStorage()
            self.assertEqual(foo_storage.type(), "torch.foo.FloatStorage")

            # 在定义 FloatStorage 之后测试自定义整型存储
            foo_tensor2 = torch.randn([8]).int().foo()
            foo_storage2 = foo_tensor2.storage()
            self.assertEqual(foo_storage2.type(), "torch.storage.TypedStorage")
        finally:
            torch.foo.FloatStorage = None

    def test_open_device_faketensor(self):
        # 使用 FakeTensorMode 模式进行测试
        with torch._subclasses.fake_tensor.FakeTensorMode.push():
            a = torch.empty(1, device="foo")
            b = torch.empty(1, device="foo:0")
            result = a + b

    def test_open_device_named_tensor(self):
        # 使用 foo 设备和指定的命名维度 ["N", "C", "H", "W"] 创建一个空张量
        torch.empty([2, 3, 4, 5], device="foo", names=["N", "C", "H", "W"])

    # 不是开放注册测试 - 此文件仅用于方便测试自定义 C++ 操作符的 torch.compile
    def test_compile_autograd_function_returns_self(self):
        # 创建随机张量并应用自定义自动求导函数 custom_autograd_fn_returns_self
        x_ref = torch.randn(4, requires_grad=True)
        out_ref = self.module.custom_autograd_fn_returns_self(x_ref)
        out_ref.sum().backward()

        # 克隆张量并测试编译后的自动求导函数
        x_test = x_ref.clone().detach().requires_grad_(True)
        f_compiled = torch.compile(self.module.custom_autograd_fn_returns_self)
        out_test = f_compiled(x_test)
        out_test.sum().backward()

        # 断言编译前后的结果相等，并且梯度计算正确
        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)

    # 不是开放注册测试 - 此文件仅用于方便测试自定义 C++ 操作符的 torch.compile
    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        # 创建随机张量并应用自定义自动求导函数 custom_autograd_fn_aliasing
        x_ref = torch.randn(4, requires_grad=True)
        out_ref = torch.ops._test_funcs.custom_autograd_fn_aliasing(x_ref)
        out_ref.sum().backward()

        # 克隆张量并测试编译后的自动求导函数
        x_test = x_ref.clone().detach().requires_grad_(True)
        f_compiled = torch.compile(torch.ops._test_funcs.custom_autograd_fn_aliasing)
        out_test = f_compiled(x_test)
        out_test.sum().backward()

        # 断言编译前后的结果相等，并且梯度计算正确
        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)
    def test_open_device_scalar_type_fallback(self):
        # 创建一个二维张量在 CPU 上
        z_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        # 调用 torch.triu_indices 函数，指定设备为 "foo"
        z = torch.triu_indices(3, 3, device="foo")
        # 断言 z_cpu 和 z 相等
        self.assertEqual(z_cpu, z)

    def test_open_device_tensor_type_fallback(self):
        # 创建位于自定义设备 "foo" 上的张量
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("foo")
        y = torch.Tensor([1, 0, 2]).to("foo")
        # 创建位于 CPU 上的结果张量
        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        # 检查我们的设备是否正确
        device = self.module.custom_device()
        self.assertTrue(x.device == device)
        self.assertFalse(x.is_cpu)

        # 调用子操作 torch.sub，它将回退到 CPU
        z = torch.sub(x, y)
        # 断言 z_cpu 和 z 相等
        self.assertEqual(z_cpu, z)

        # 调用索引操作，它将回退到 CPU
        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("foo")
        z = x[y, y]
        # 断言 z_cpu 和 z 相等
        self.assertEqual(z_cpu, z)

    def test_open_device_tensorlist_type_fallback(self):
        # 创建位于自定义设备 "foo" 上的张量
        v_foo = torch.Tensor([1, 2, 3]).to("foo")
        # 创建位于 CPU 上的结果张量
        z_cpu = torch.Tensor([2, 4, 6])
        # 创建用于 foreach_add 操作的张量列表
        x = (v_foo, v_foo)
        y = (v_foo, v_foo)
        # 检查我们的设备是否正确
        device = self.module.custom_device()
        self.assertTrue(v_foo.device == device)
        self.assertFalse(v_foo.is_cpu)

        # 调用 _foreach_add 操作，它将回退到 CPU
        z = torch._foreach_add(x, y)
        # 断言 z_cpu 和 z[0]、z[1] 相等
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])

    def test_open_device_numpy_serialization_map_location(self):
        # 重命名私有后端为 "foo"
        torch.utils.rename_privateuse1_backend("foo")
        # 获取自定义设备
        device = self.module.custom_device()
        default_protocol = torch.serialization.DEFAULT_PROTOCOL
        # 这是一个测试通过 numpy 进行序列化的 hack
        with patch.object(torch._C, "_has_storage", return_value=False):
            x = torch.randn(2, 3)
            x_foo = x.to(device)
            sd = {"x": x_foo}
            rebuild_func = x_foo._reduce_ex_internal(default_protocol)[0]
            self.assertTrue(
                rebuild_func is torch._utils._rebuild_device_tensor_from_numpy
            )
            with TemporaryFileName() as f:
                torch.save(sd, f)
                sd_loaded = torch.load(f, map_location="cpu")
                self.assertTrue(sd_loaded["x"].is_cpu)
if __name__ == "__main__":
    # 如果当前脚本被直接执行（而不是被导入到其他脚本中），则执行以下代码块
    common.run_tests()
```