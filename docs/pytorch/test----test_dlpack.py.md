# `.\pytorch\test\test_dlpack.py`

```py
# Owner(s): ["module: tests"]

# 导入需要的模块和函数
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIfRocm,
    skipMeta,
)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import IS_JETSON, run_tests, TestCase
from torch.utils.dlpack import from_dlpack, to_dlpack

# 定义测试类 TestTorchDlPack，继承自 TestCase
class TestTorchDlPack(TestCase):
    exact_dtype = True

    # 装饰器，跳过元信息检查
    @skipMeta
    # 装饰器，仅在本地设备类型上运行测试
    @onlyNativeDeviceTypes
    # 装饰器，指定测试使用的数据类型
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    # 定义测试函数，测试 dlpack 胶囊转换
    def test_dlpack_capsule_conversion(self, device, dtype):
        # 创建指定设备和数据类型的张量 x
        x = make_tensor((5,), dtype=dtype, device=device)
        # 使用 to_dlpack 将张量 x 转换为 dlpack 格式，再用 from_dlpack 转回张量 z
        z = from_dlpack(to_dlpack(x))
        # 断言 z 和 x 相等
        self.assertEqual(z, x)

    # 装饰器，跳过元信息检查
    @skipMeta
    # 装饰器，仅在本地设备类型上运行测试
    @onlyNativeDeviceTypes
    # 装饰器，指定测试使用的数据类型
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    # 定义测试函数，测试 dlpack 协议转换
    def test_dlpack_protocol_conversion(self, device, dtype):
        # 创建指定设备和数据类型的张量 x
        x = make_tensor((5,), dtype=dtype, device=device)
        # 使用 from_dlpack 将张量 x 转换为 dlpack 格式，再赋给 z
        z = from_dlpack(x)
        # 断言 z 和 x 相等
        self.assertEqual(z, x)

    # 装饰器，跳过元信息检查
    @skipMeta
    # 装饰器，仅在本地设备类型上运行测试
    @onlyNativeDeviceTypes
    # 定义测试函数，测试 dlpack 共享存储
    def test_dlpack_shared_storage(self, device):
        # 创建指定设备和数据类型的张量 x
        x = make_tensor((5,), dtype=torch.float64, device=device)
        # 使用 to_dlpack 将张量 x 转换为 dlpack 格式，再用 from_dlpack 转回张量 z
        z = from_dlpack(to_dlpack(x))
        # 修改 z 的第一个元素值
        z[0] = z[0] + 20.0
        # 断言 z 和 x 相等
        self.assertEqual(z, x)

    # 装饰器，跳过元信息检查
    @skipMeta
    # 装饰器，仅在 CUDA 设备上运行测试
    @onlyCUDA
    # 装饰器，指定测试使用的数据类型
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    # 定义测试函数，测试带流的 dlpack 转换
    def test_dlpack_conversion_with_streams(self, device, dtype):
        # 创建一个 CUDA 流
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # 在指定流中进行操作，创建指定设备和数据类型的张量 x
            x = make_tensor((5,), dtype=dtype, device=device) + 1
        # 使用 DLPack 协议确保正确的流顺序和数据依赖
        # DLPack 管理这种同步，无需显式等待 x 填充完毕
        if IS_JETSON:
            # 在 Jetson 上，DLPack 协议的流顺序不符合预期行为
            stream.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # 使用 from_dlpack 将张量 x 转换为 dlpack 格式，再赋给 z
            z = from_dlpack(x)
        stream.synchronize()
        # 断言 z 和 x 相等
        self.assertEqual(z, x)

    # 装饰器，跳过元信息检查
    @skipMeta
    # 装饰器，仅在本地设备类型上运行测试
    @onlyNativeDeviceTypes
    # 装饰器，指定测试使用的数据类型
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    # 使用 dlpack 将张量从设备和数据类型无关的内存格式转换为 PyTorch 张量
    def test_from_dlpack(self, device, dtype):
        # 创建一个指定设备和数据类型的张量 x
        x = make_tensor((5,), dtype=dtype, device=device)
        # 使用 from_dlpack 将 dlpack 张量转换为 PyTorch 张量 y
        y = torch.from_dlpack(x)
        # 断言 x 和 y 相等
        self.assertEqual(x, y)

    # 跳过元数据测试，仅适用于本地设备类型
    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    # 使用 dlpack 将非连续的 PyTorch 张量转换为张量
    def test_from_dlpack_noncontinguous(self, device, dtype):
        # 创建一个指定设备和数据类型的张量 x，并且将其重塑为 5x5 的形状
        x = make_tensor((25,), dtype=dtype, device=device).reshape(5, 5)

        # 获取张量 x 的第一行 y1
        y1 = x[0]
        # 使用 from_dlpack 将 y1 转换为 PyTorch 张量 y1_dl
        y1_dl = torch.from_dlpack(y1)
        # 断言 y1 和 y1_dl 相等
        self.assertEqual(y1, y1_dl)

        # 获取张量 x 的第一列 y2
        y2 = x[:, 0]
        # 使用 from_dlpack 将 y2 转换为 PyTorch 张量 y2_dl
        y2_dl = torch.from_dlpack(y2)
        # 断言 y2 和 y2_dl 相等
        self.assertEqual(y2, y2_dl)

        # 获取张量 x 的第二行 y3
        y3 = x[1, :]
        # 使用 from_dlpack 将 y3 转换为 PyTorch 张量 y3_dl
        y3_dl = torch.from_dlpack(y3)
        # 断言 y3 和 y3_dl 相等
        self.assertEqual(y3, y3_dl)

        # 获取张量 x 的第二行 y4
        y4 = x[1]
        # 使用 from_dlpack 将 y4 转换为 PyTorch 张量 y4_dl
        y4_dl = torch.from_dlpack(y4)
        # 断言 y4 和 y4_dl 相等
        self.assertEqual(y4, y4_dl)

        # 获取张量 x 的转置 y5
        y5 = x.t()
        # 使用 from_dlpack 将 y5 转换为 PyTorch 张量 y5_dl
        y5_dl = torch.from_dlpack(y5)
        # 断言 y5 和 y5_dl 相等
        self.assertEqual(y5, y5_dl)

    # 跳过元数据测试，仅适用于 CUDA 设备
    @skipMeta
    @onlyCUDA
    # 使用 dlpack 在不同流中进行张量转换
    def test_dlpack_conversion_with_diff_streams(self, device, dtype):
        # 创建两个 CUDA 流 stream_a 和 stream_b
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        # 使用 DLPack 协议在交换边界处确保正确的流顺序（因此数据依赖性）
        # `tensor.__dlpack__` 方法将在当前流中插入同步事件，以确保正确地填充数据
        with torch.cuda.stream(stream_a):
            # 创建一个指定设备和数据类型的张量 x，并加上 1
            x = make_tensor((5,), dtype=dtype, device=device) + 1
            # 使用 x.__dlpack__(stream_b.cuda_stream) 将 x 转换为 PyTorch 张量 z
            z = torch.from_dlpack(x.__dlpack__(stream_b.cuda_stream))
            # 同步 stream_a 流
            stream_a.synchronize()
        # 同步 stream_b 流
        stream_b.synchronize()
        # 断言 z 和 x 相等
        self.assertEqual(z, x)

    # 跳过元数据测试，仅适用于本地设备类型
    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    # 使用 dlpack 将 PyTorch 张量转换为相同数据类型的张量
    def test_from_dlpack_dtype(self, device, dtype):
        # 创建一个指定设备和数据类型的张量 x
        x = make_tensor((5,), dtype=dtype, device=device)
        # 使用 from_dlpack 将 dlpack 张量转换为 PyTorch 张量 y
        y = torch.from_dlpack(x)
        # 使用断言确保 x 和 y 的数据类型相同
        assert x.dtype == y.dtype

    # 跳过元数据测试，仅适用于 CUDA 设备
    @skipMeta
    @onlyCUDA
    @skipMeta
    def test_dlpack_default_stream(self, device):
        class DLPackTensor:
            def __init__(self, tensor):
                self.tensor = tensor

            def __dlpack_device__(self):
                return self.tensor.__dlpack_device__()

            def __dlpack__(self, stream=None):
                # 如果 torch 版本的 HIP 模块不存在
                if torch.version.hip is None:
                    # 断言 stream 应为 1
                    assert stream == 1
                else:
                    # 否则断言 stream 应为 0
                    assert stream == 0
                # 调用 tensor 的 __dlpack__ 方法，返回 dlpack capsule
                capsule = self.tensor.__dlpack__(stream)
                return capsule

        # 在 CUDA 测试中运行在非默认流上
        with torch.cuda.stream(torch.cuda.default_stream()):
            # 创建 DLPackTensor 对象 x，使用指定设备生成大小为 (5,) 的浮点张量
            x = DLPackTensor(make_tensor((5,), dtype=torch.float32, device=device))
            # 调用 from_dlpack 函数
            from_dlpack(x)

    @skipMeta
    @onlyCUDA
    @skipCUDAIfRocm
    def test_dlpack_convert_default_stream(self, device):
        # 测试在非默认流上运行，因此下面的 _sleep 调用将在非默认流上运行，
        # 导致由于插入的同步而使默认流等待
        torch.cuda.default_stream().synchronize()
        # 在非默认流上运行 _sleep 调用，导致由于插入的同步而使默认流等待
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            # 使用指定设备生成大小为 (1,) 的零张量
            x = torch.zeros(1, device=device)
            # 在非默认流上调用 _sleep，导致默认流等待插入的同步
            torch.cuda._sleep(2**20)
            # 断言默认流是否有任务
            self.assertTrue(torch.cuda.default_stream().query())
            # 调用张量的 __dlpack__ 方法，传入 stream=1
            d = x.__dlpack__(1)
        # 检查默认流是否有任务（等待插入的 cudaStreamWaitEvent）
        self.assertFalse(torch.cuda.default_stream().query())

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_tensor_invalid_stream(self, device, dtype):
        # 使用 assertRaises 检查是否抛出 TypeError 异常
        with self.assertRaises(TypeError):
            # 使用指定设备生成指定类型和大小 (5,) 的张量
            x = make_tensor((5,), dtype=dtype, device=device)
            # 调用张量的 __dlpack__ 方法，传入一个对象作为 stream 参数
            x.__dlpack__(stream=object())

    # TODO: add interchange tests once NumPy 1.22 (dlpack support) is required
    @skipMeta
    def test_dlpack_export_requires_grad(self):
        # 创建一个需要梯度的大小为 10 的零张量
        x = torch.zeros(10, dtype=torch.float32, requires_grad=True)
        # 使用 assertRaisesRegex 检查是否抛出带有 "require gradient" 的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"require gradient"):
            # 调用张量的 __dlpack__ 方法
            x.__dlpack__()

    @skipMeta
    def test_dlpack_export_is_conj(self):
        # 创建一个复数张量
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        # 对张量进行共轭操作
        y = torch.conj(x)
        # 使用 assertRaisesRegex 检查是否抛出带有 "conjugate bit" 的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"conjugate bit"):
            # 调用张量的 __dlpack__ 方法
            y.__dlpack__()

    @skipMeta
    def test_dlpack_export_non_strided(self):
        # 创建一个稀疏的 COO 张量
        x = torch.sparse_coo_tensor([[0]], [1], size=(1,))
        # 对张量进行共轭操作
        y = torch.conj(x)
        # 使用 assertRaisesRegex 检查是否抛出带有 "strided" 的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"strided"):
            # 调用张量的 __dlpack__ 方法
            y.__dlpack__()
    # 定义一个测试方法，用于测试 dlpack_normalize_strides 函数
    def test_dlpack_normalize_strides(self):
        # 创建一个包含16个随机数的张量 x
        x = torch.rand(16)
        # 通过切片操作，获取 x 张量中间隔为3的元素的切片，并取切片结果的第一个元素
        y = x[::3][:1]
        # 断言切片结果 y 的形状为 (1,)
        self.assertEqual(y.shape, (1,))
        # 断言切片结果 y 的步长为 (3,)
        self.assertEqual(y.stride(), (3,))
        # 将 y 张量转换为 dlpack 格式的张量 z
        z = from_dlpack(y)
        # 断言转换后张量 z 的形状为 (1,)
        self.assertEqual(z.shape, (1,))
        # 断言转换后张量 z 的步长为 (1,)，以确保 __dlpack__ 方法已经归一化了步长
        # gh-83069，确保 __dlpack__ 方法已经归一化了步长
        self.assertEqual(z.stride(), (1,))
# 调用函数 instantiate_device_type_tests，传入 TestTorchDlPack 和 globals() 作为参数，用于实例化设备类型测试。
instantiate_device_type_tests(TestTorchDlPack, globals())

# 如果当前脚本作为主程序运行，则执行 run_tests() 函数。
if __name__ == "__main__":
    run_tests()
```