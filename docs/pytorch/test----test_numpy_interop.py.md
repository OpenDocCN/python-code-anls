# `.\pytorch\test\test_numpy_interop.py`

```
# mypy: ignore-errors
# 忽略 mypy 类型检查时的错误

# Owner(s): ["module: numpy"]
# 所有者信息，指明此代码模块属于 numpy

import sys
# 导入 sys 模块

from itertools import product
# 从 itertools 模块导入 product 函数，用于生成迭代器的笛卡尔积

import numpy as np
# 导入 numpy 库，并使用 np 别名

import torch
# 导入 torch 库

from torch.testing import make_tensor
# 从 torch.testing 模块导入 make_tensor 函数

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    skipMeta,
)
# 从 torch.testing._internal.common_device_type 模块导入多个符号，包括 dtypes, instantiate_device_type_tests, onlyCPU, skipMeta

from torch.testing._internal.common_dtype import all_types_and_complex_and
# 从 torch.testing._internal.common_dtype 模块导入 all_types_and_complex_and 符号

from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
# 从 torch.testing._internal.common_utils 模块导入 run_tests, skipIfTorchDynamo, TestCase 等符号

# For testing handling NumPy objects and sending tensors to / accepting
#   arrays from NumPy.
# 用于测试处理 NumPy 对象，并将张量发送到 / 从 NumPy 接受数组。

class TestNumPyInterop(TestCase):
    # Note: the warning this tests for only appears once per program, so
    # other instances of this warning should be addressed to avoid
    # the tests depending on the order in which they're run.
    # 注意：此测试的警告每个程序只出现一次，因此应解决其他引发此警告的情况，
    # 以避免测试依赖于其运行顺序。

    @onlyCPU
    def test_numpy_non_writeable(self, device):
        # 在 CPU 上运行此测试函数

        arr = np.zeros(5)
        # 创建一个包含 5 个零的 NumPy 数组 arr

        arr.flags["WRITEABLE"] = False
        # 将 arr 数组设置为不可写

        self.assertWarns(UserWarning, lambda: torch.from_numpy(arr))
        # 断言会发出 UserWarning 警告，当尝试将 arr 转换为 Torch 张量时

    @onlyCPU
    def test_numpy_unresizable(self, device) -> None:
        # 在 CPU 上运行此测试函数，且不返回值

        x = np.zeros((2, 2))
        # 创建一个 2x2 的全零 NumPy 数组 x

        y = torch.from_numpy(x)
        # 将 NumPy 数组 x 转换为 Torch 张量 y

        with self.assertRaises(ValueError):
            x.resize((5, 5))
        # 使用 x.resize 尝试重新调整 x 的形状为 (5, 5)，预期会引发 ValueError 异常

        z = torch.randn(5, 5)
        # 创建一个形状为 (5, 5) 的随机数张量 z

        w = z.numpy()
        # 将 Torch 张量 z 转换为 NumPy 数组 w

        with self.assertRaises(RuntimeError):
            z.resize_(10, 10)
        # 使用 z.resize_ 尝试重新调整 z 的形状为 (10, 10)，预期会引发 RuntimeError 异常

        with self.assertRaises(ValueError):
            w.resize((10, 10))
        # 使用 w.resize 尝试重新调整 w 的形状为 (10, 10)，预期会引发 ValueError 异常

    @onlyCPU
    def test_to_numpy_bool(self, device) -> None:
        # 在 CPU 上运行此测试函数，且不返回值

        x = torch.tensor([True, False], dtype=torch.bool)
        # 创建一个包含 True 和 False 的 Torch 布尔型张量 x

        self.assertEqual(x.dtype, torch.bool)
        # 断言 x 的数据类型为 torch.bool

        y = x.numpy()
        # 将 Torch 张量 x 转换为 NumPy 数组 y

        self.assertEqual(y.dtype, np.bool_)
        # 断言 y 的数据类型为 np.bool_

        for i in range(len(x)):
            self.assertEqual(x[i], y[i])
        # 遍历 x 和 y，逐一断言它们的元素相等

        x = torch.tensor([True], dtype=torch.bool)
        # 创建一个包含 True 的 Torch 布尔型张量 x

        self.assertEqual(x.dtype, torch.bool)
        # 断言 x 的数据类型为 torch.bool

        y = x.numpy()
        # 将 Torch 张量 x 转换为 NumPy 数组 y

        self.assertEqual(y.dtype, np.bool_)
        # 断言 y 的数据类型为 np.bool_

        self.assertEqual(x[0], y[0])
        # 断言 x 和 y 的第一个元素相等

    @skipIfTorchDynamo("conj bit not implemented in TensorVariable yet")
    # 如果在 TorchDynamo 下跳过此测试，输出指定的消息
    # 定义一个测试方法，将 tensor 转换为 NumPy 数组，测试是否按预期工作
    def test_to_numpy_force_argument(self, device) -> None:
        # 遍历是否强制转换参数
        for force in [False, True]:
            # 遍历是否需要梯度参数
            for requires_grad in [False, True]:
                # 遍历是否稀疏张量参数
                for sparse in [False, True]:
                    # 遍历是否共轭参数
                    for conj in [False, True]:
                        # 定义一个复杂数据结构
                        data = [[1 + 2j, -2 + 3j], [-1 - 2j, 3 - 2j]]
                        # 创建一个张量 tensor，并根据参数设置是否需要梯度和设备位置
                        x = torch.tensor(
                            data, requires_grad=requires_grad, device=device
                        )
                        # 将 y 设置为 x 的引用
                        y = x
                        # 如果稀疏参数为真
                        if sparse:
                            # 如果需要梯度参数为真，则跳过此次循环
                            if requires_grad:
                                continue
                            # 将 x 转换为稀疏张量
                            x = x.to_sparse()
                        # 如果共轭参数为真
                        if conj:
                            # 对 x 进行共轭操作
                            x = x.conj()
                            # 解析共轭
                            y = x.resolve_conj()
                        # 预期会出现错误的条件
                        expect_error = (
                            requires_grad or sparse or conj or not device == "cpu"
                        )
                        # 定义错误消息的正则表达式
                        error_msg = r"Use (t|T)ensor\..*(\.numpy\(\))?"
                        # 如果不是强制转换且预期会出现错误
                        if not force and expect_error:
                            # 断言会抛出指定类型的异常，并检查错误消息是否符合正则表达式
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError), error_msg, lambda: x.numpy()
                            )
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError),
                                error_msg,
                                lambda: x.numpy(force=False),
                            )
                        # 如果是强制转换且是稀疏张量
                        elif force and sparse:
                            # 断言会抛出类型错误异常，并检查错误消息是否符合正则表达式
                            self.assertRaisesRegex(
                                TypeError, error_msg, lambda: x.numpy(force=True)
                            )
                        # 否则
                        else:
                            # 断言 x 转换为 NumPy 数组是否等于 y
                            self.assertEqual(x.numpy(force=force), y)
    # 定义一个测试方法，测试从 NumPy 数组创建 Torch 张量的功能
    def test_from_numpy(self, device) -> None:
        # 定义一组 NumPy 数据类型，用于测试
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.complex64,
            np.complex128,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            np.longlong,
            np.bool_,
        ]
        # 定义复数数据类型的子集，用于特殊处理
        complex_dtypes = [
            np.complex64,
            np.complex128,
        ]

        # 遍历每种数据类型进行测试
        for dtype in dtypes:
            # 创建一个 NumPy 数组
            array = np.array([1, 2, 3, 4], dtype=dtype)
            # 从 NumPy 数组创建 Torch 张量
            tensor_from_array = torch.from_numpy(array)
            # 检查每个元素是否相等
            for i in range(len(array)):
                self.assertEqual(tensor_from_array[i], array[i])
            # 对于非复数数据类型进行特殊处理
            if dtype not in complex_dtypes:
                # 创建一个取模后的数组，用于特殊情况的测试
                array2 = array % 2
                tensor_from_array2 = torch.from_numpy(array2)
                # 检查每个元素是否相等
                for i in range(len(array2)):
                    self.assertEqual(tensor_from_array2[i], array2[i])

        # 测试不支持的数据类型
        array = np.array(["foo", "bar"], dtype=np.dtype(np.str_))
        # 检查是否抛出预期的 TypeError 异常
        with self.assertRaises(TypeError):
            tensor_from_array = torch.from_numpy(array)

        # 检查存储偏移量
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[1]
        expected = torch.arange(1, 126, dtype=torch.float64).view(5, 5, 5)[1]
        # 检查从 NumPy 数组创建的 Torch 张量是否与预期值相等
        self.assertEqual(torch.from_numpy(x), expected)

        # 检查非连续存储情况
        x = np.linspace(1, 25, 25)
        x.shape = (5, 5)
        expected = torch.arange(1, 26, dtype=torch.float64).view(5, 5).t()
        # 检查从转置后的 NumPy 数组创建的 Torch 张量是否与预期值相等
        self.assertEqual(torch.from_numpy(x.T), expected)

        # 检查带有空位的非连续存储情况
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[:, 1]
        expected = torch.arange(1, 126, dtype=torch.float64).view(5, 5, 5)[:, 1]
        # 检查从 NumPy 数组创建的 Torch 张量是否与预期值相等
        self.assertEqual(torch.from_numpy(x), expected)

        # 检查零维数组的情况
        x = np.zeros((0, 2))
        # 检查从零维 NumPy 数组创建的 Torch 张量的形状是否与预期相等
        self.assertEqual(torch.from_numpy(x).shape, (0, 2))
        x = np.zeros((2, 0))
        # 检查从零维 NumPy 数组创建的 Torch 张量的形状是否与预期相等
        self.assertEqual(torch.from_numpy(x).shape, (2, 0))

        # 检查不合适的步幅是否引发异常
        x = np.array([3.0, 5.0, 8.0])
        x.strides = (3,)
        # 检查是否抛出预期的 ValueError 异常
        self.assertRaises(ValueError, lambda: torch.from_numpy(x))

    # 根据 TorchDynamo 的条件跳过测试
    @skipIfTorchDynamo("No need to test invalid dtypes that should fail by design.")
    def test_from_numpy_no_leak_on_invalid_dtype(self):
        # 检查从 NumPy 数组创建张量时，处理无效数据类型不会导致内存泄漏。
        # 这段代码测试了在抛出异常时，临时对象是否正确地进行了减引用。
        x = np.array("value".encode("ascii"))
        # 重复执行 1000 次以下代码块
        for _ in range(1000):
            try:
                # 尝试从 NumPy 数组创建张量
                torch.from_numpy(x)
            except TypeError:
                # 如果抛出类型错误异常，则跳过
                pass
        # 断言 NumPy 数组的引用计数为 2
        self.assertTrue(sys.getrefcount(x) == 2)

    @skipMeta
    def test_from_list_of_ndarray_warning(self, device):
        # 测试从 NumPy 数组列表创建张量时是否发出警告
        warning_msg = (
            r"Creating a tensor from a list of numpy.ndarrays is extremely slow"
        )
        # 使用 assertWarnsOnceRegex 确保只捕获一次 UserWarning，并检查警告消息
        with self.assertWarnsOnceRegex(UserWarning, warning_msg):
            # 尝试从 NumPy 数组列表创建张量，并传递设备参数
            torch.tensor([np.array([0]), np.array([1])], device=device)

    def test_ctor_with_invalid_numpy_array_sequence(self, device):
        # 测试使用无效的 NumPy 数组序列创建张量时是否引发 ValueError
        # Invalid list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            # 尝试传递一个包含无效长度的 NumPy 数组列表到 torch.tensor 中
            torch.tensor(
                [np.random.random(size=(3, 3)), np.random.random(size=(3, 0))],
                device=device,
            )

        # Invalid list of list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            # 尝试传递一个包含无效长度的 NumPy 数组列表的列表到 torch.tensor 中
            torch.tensor(
                [[np.random.random(size=(3, 3)), np.random.random(size=(3, 2))]],
                device=device,
            )

        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            # 尝试传递一个包含无效长度的 NumPy 数组列表的列表到 torch.tensor 中
            torch.tensor(
                [
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 3))],
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 2))],
                ],
                device=device,
            )

        # expected shape is `[1, 2, 3]`, hence we try to iterate over 0-D array
        # leading to type error : not a sequence.
        with self.assertRaisesRegex(TypeError, "not a sequence"):
            # 尝试传递一个包含不是序列的 0-D 数组到 torch.tensor 中
            torch.tensor(
                [[np.random.random(size=(3)), np.random.random()]], device=device
            )

        # list of list or numpy array.
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            # 尝试传递一个包含无效长度的列表或 NumPy 数组到 torch.tensor 中
            torch.tensor(
                [
                    [1, 2, 3],
                    np.random.random(size=(2,)),
                ],
                device=device,
            )

    @onlyCPU
    def test_ctor_with_numpy_scalar_ctor(self, device) -> None:
        # 测试使用 NumPy 标量构造函数创建张量时的正确性
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.uint8,
            np.bool_,
        ]
        # 遍历不同的 NumPy 数据类型
        for dtype in dtypes:
            # 断言 NumPy 标量转换为张量后，其值不变
            self.assertEqual(dtype(42), torch.tensor(dtype(42)).item())

    @onlyCPU
    # 测试使用 NumPy 数组作为索引的情况
    def test_numpy_index(self, device):
        # 创建一个 NumPy 数组作为索引，数据类型为 np.int32
        i = np.array([0, 1, 2], dtype=np.int32)
        # 创建一个 5x5 的随机张量
        x = torch.randn(5, 5)
        # 遍历索引数组 i
        for idx in i:
            # 断言索引 idx 不是 int 类型
            self.assertFalse(isinstance(idx, int))
            # 断言 x[idx] 等于 x[int(idx)]
            self.assertEqual(x[idx], x[int(idx)])

    # 标记为只在 CPU 上运行的测试
    @onlyCPU
    # 测试使用 NumPy 多维数组作为索引的情况
    def test_numpy_index_multi(self, device):
        # 遍历不同的维度大小
        for dim_sz in [2, 8, 16, 32]:
            # 创建一个全零的三维 NumPy 数组，数据类型为 np.int32
            i = np.zeros((dim_sz, dim_sz, dim_sz), dtype=np.int32)
            # 将部分区域置为1
            i[: dim_sz // 2, :, :] = 1
            # 创建一个随机张量，大小与 i 相同
            x = torch.randn(dim_sz, dim_sz, dim_sz)
            # 断言 x[i == 1] 的元素个数等于 np.sum(i)
            self.assertTrue(x[i == 1].numel() == np.sum(i))

    # 标记为只在 CPU 上运行的测试，并且有两个相同标记
    @onlyCPU
    @onlyCPU
    # 测试 NumPy 数字标量与 PyTorch 张量相乘的情况
    def test_multiplication_numpy_scalar(self, device) -> None:
        # 遍历不同的 NumPy 数据类型
        for np_dtype in [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.int16,
            np.uint8,
        ]:
            # 遍历不同的 PyTorch 数据类型
            for t_dtype in [torch.float, torch.double]:
                # 使用 np.floatXY(2.0) 创建一个 NumPy 数字标量 np_sc
                # 这里的类型提示告诉 mypy 在特定情况下忽略类型检查错误
                np_sc = np_dtype(2.0)  # type: ignore[abstract, arg-type]
                # 创建一个全为1的张量，需要梯度，数据类型为 t_dtype
                t = torch.ones(2, requires_grad=True, dtype=t_dtype)
                # 计算张量 t 乘以标量 np_sc
                r1 = t * np_sc
                # 断言 r1 是一个张量
                self.assertIsInstance(r1, torch.Tensor)
                # 断言 r1 的数据类型为 t_dtype
                self.assertTrue(r1.dtype == t_dtype)
                # 断言 r1 需要梯度
                self.assertTrue(r1.requires_grad)
                # 计算标量 np_sc 乘以张量 t
                r2 = np_sc * t
                # 断言 r2 是一个张量
                self.assertIsInstance(r2, torch.Tensor)
                # 断言 r2 的数据类型为 t_dtype
                self.assertTrue(r2.dtype == t_dtype)
                # 断言 r2 需要梯度
                self.assertTrue(r2.requires_grad)

    # 标记为只在 CPU 上运行的测试，并且在 TorchDynamo 环境下跳过
    @onlyCPU
    @skipIfTorchDynamo()
    # 测试在处理 NumPy 整数溢出时的情况
    def test_parse_numpy_int_overflow(self, device):
        # 使用 lambda 表达式来测试是否抛出特定的 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            "(Overflow|an integer is required)",
            lambda: torch.mean(torch.randn(1, 1), np.uint64(-1)),
        )  # type: ignore[call-overload]
    # 测试解析 numpy 的整数类型
    def test_parse_numpy_int(self, device):
        # GitHub 上的问题链接
        # 遍历不同的 numpy 整数类型
        for nptype in [np.int16, np.int8, np.uint8, np.int32, np.int64]:
            # 定义一个标量
            scalar = 3
            # 创建一个 numpy 数组，指定数据类型为 nptype
            np_arr = np.array([scalar], dtype=nptype)
            # 获取 numpy 数组中的值
            np_val = np_arr[0]

            # 对于具有整数参数的原生函数，numpy 整数类型可以被视为 Python int：
            self.assertEqual(torch.ones(5).diag(scalar), torch.ones(5).diag(np_val))
            self.assertEqual(
                torch.ones([2, 2, 2, 2]).mean(scalar),
                torch.ones([2, 2, 2, 2]).mean(np_val),
            )

            # numpy 整数类型在自定义 Python 绑定中解析为 Python int：
            self.assertEqual(torch.Storage(np_val).size(), scalar)  # type: ignore[attr-defined]

            # 创建一个 torch 张量，数据类型为 torch.int，将 np_val 赋值给其第一个元素
            tensor = torch.tensor([2], dtype=torch.int)
            tensor[0] = np_val
            self.assertEqual(tensor[0], np_val)

            # 原始报告的问题，当作为算术运算中的 `Scalar` 参数传递时，np 整数类型会解析为正确的 PyTorch 整数类型：
            t = torch.from_numpy(np_arr)
            self.assertEqual((t + np_val).dtype, t.dtype)
            self.assertEqual((np_val + t).dtype, t.dtype)

    # 测试是否具有与 numpy 对应的存储
    def test_has_storage_numpy(self, device):
        # 遍历不同的 numpy 数据类型
        for dtype in [np.float32, np.float64, np.int64, np.int32, np.int16, np.uint8]:
            # 创建一个 numpy 数组 arr
            arr = np.array([1], dtype=dtype)
            # 使用指定的设备和数据类型 torch.float32 创建 torch 张量，并获取其存储
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.float32).storage()
            )
            # 使用指定的设备和数据类型 torch.double 创建 torch 张量，并获取其存储
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.double).storage()
            )
            # 使用指定的设备和数据类型 torch.int 创建 torch 张量，并获取其存储
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.int).storage()
            )
            # 使用指定的设备和数据类型 torch.long 创建 torch 张量，并获取其存储
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.long).storage()
            )
            # 使用指定的设备和数据类型 torch.uint8 创建 torch 张量，并获取其存储
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.uint8).storage()
            )

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    # 定义一个测试方法，用于比较 numpy 标量的操作
    def test_numpy_scalar_cmp(self, device, dtype):
        # 根据数据类型是否为复数，选择不同的测试张量
        if dtype.is_complex:
            tensors = (
                torch.tensor(complex(1, 3), dtype=dtype, device=device),  # 创建复数张量
                torch.tensor([complex(1, 3), 0, 2j], dtype=dtype, device=device),  # 创建复数张量列表
                torch.tensor(
                    [[complex(3, 1), 0], [-1j, 5]], dtype=dtype, device=device  # 创建复数张量二维数组
                ),
            )
        else:
            tensors = (
                torch.tensor(3, dtype=dtype, device=device),  # 创建标量张量
                torch.tensor([1, 0, -3], dtype=dtype, device=device),  # 创建整数张量列表
                torch.tensor([[3, 0, -1], [3, 5, 4]], dtype=dtype, device=device),  # 创建整数张量二维数组
            )

        # 遍历所有测试张量
        for tensor in tensors:
            # 如果数据类型是 bfloat16，则断言会引发 TypeError 异常，并继续下一个张量的测试
            if dtype == torch.bfloat16:
                with self.assertRaises(TypeError):
                    np_array = tensor.cpu().numpy()
                continue

            # 将张量转换为 numpy 数组
            np_array = tensor.cpu().numpy()

            # 遍历张量和对应的 numpy 数组的第一个扁平化元素和单个元素
            for t, a in product(
                (tensor.flatten()[0], tensor.flatten()[0].item()),  # 获取张量的扁平化元素和单个元素
                (np_array.flatten()[0], np_array.flatten()[0].item()),  # 获取 numpy 数组的扁平化元素和单个元素
            ):
                # 断言张量元素和 numpy 数组元素相等
                self.assertEqual(t, a)

                # 如果数据类型是 torch.complex64，并且张量 t 是张量且 numpy 数组 a 是 np.complex64 类型
                if (
                    dtype == torch.complex64
                    and torch.is_tensor(t)
                    and type(a) == np.complex64
                ):
                    # 提示信息：在这种情况下，虚部被丢弃了，需要修复
                    # https://github.com/pytorch/pytorch/issues/43579
                    self.assertFalse(t == a)  # 断言张量 t 不等于 numpy 数组 a
                else:
                    self.assertTrue(t == a)  # 断言张量 t 等于 numpy 数组 a

    # 仅在 CPU 上执行的测试装饰器，测试数据类型为半精度浮点数和布尔型的等值操作
    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    def test___eq__(self, device, dtype):
        # 创建指定形状和数据类型的张量 a
        a = make_tensor((5, 7), dtype=dtype, device=device, low=-9, high=9)
        # 创建张量 b 作为 a 的克隆并断开依赖
        b = a.clone().detach()
        # 将张量 b 转换为 numpy 数组
        b_np = b.numpy()

        # 断言所有元素都相等
        res_check = torch.ones_like(a, dtype=torch.bool)
        self.assertEqual(a == b_np, res_check)  # 断言 a 是否等于 b_np
        self.assertEqual(b_np == a, res_check)  # 断言 b_np 是否等于 a

        # 检查一个元素不相等的情况
        if dtype == torch.bool:
            b[1][3] = not b[1][3]
        else:
            b[1][3] += 1
        res_check[1][3] = False
        self.assertEqual(a == b_np, res_check)  # 断言 a 是否等于 b_np
        self.assertEqual(b_np == a, res_check)  # 断言 b_np 是否等于 a

        # 检查随机元素不相等的情况
        rand = torch.randint(0, 2, a.shape, dtype=torch.bool)
        res_check = rand.logical_not()
        b.copy_(a)

        if dtype == torch.bool:
            b[rand] = b[rand].logical_not()
        else:
            b[rand] += 1

        self.assertEqual(a == b_np, res_check)  # 断言 a 是否等于 b_np
        self.assertEqual(b_np == a, res_check)  # 断言 b_np 是否等于 a

        # 检查所有元素不相等的情况
        if dtype == torch.bool:
            b.copy_(a.logical_not())
        else:
            b.copy_(a + 1)
        res_check.fill_(False)
        self.assertEqual(a == b_np, res_check)  # 断言 a 是否等于 b_np
        self.assertEqual(b_np == a, res_check)  # 断言 b_np 是否等于 a
    # 定义一个测试方法，用于测试空张量的互操作性，并指定设备
    def test_empty_tensors_interop(self, device):
        # 创建一个随机数值的标量张量 x，数据类型为 torch.float16
        x = torch.rand((), dtype=torch.float16)
        # 使用 numpy 随机数生成器创建一个空张量 y，数据类型为 torch.float16
        y = torch.tensor(np.random.rand(0), dtype=torch.float16)
        # 也可以通过以下方式创建相同的张量 y
        # y = torch.empty_strided((0,), (0,), dtype=torch.float16)

        # 对 https://github.com/pytorch/pytorch/issues/115068 的回归测试
        # 验证 torch.true_divide(x, y) 的结果形状与张量 y 的形状相同
        self.assertEqual(torch.true_divide(x, y).shape, y.shape)
        
        # 对 https://github.com/pytorch/pytorch/issues/115066 的回归测试
        # 验证 torch.mul(x, y) 的结果形状与张量 y 的形状相同
        self.assertEqual(torch.mul(x, y).shape, y.shape)
        
        # 对 https://github.com/pytorch/pytorch/issues/113037 的回归测试
        # 使用 floor 舍入模式，验证 torch.div(x, y) 的结果形状与张量 y 的形状相同
        self.assertEqual(torch.div(x, y, rounding_mode="floor").shape, y.shape)
# 调用函数 instantiate_device_type_tests，用于实例化设备类型测试，针对 TestNumPyInterop 类，在全局命名空间中执行
instantiate_device_type_tests(TestNumPyInterop, globals())

# 检查当前脚本是否作为主程序执行
if __name__ == "__main__":
    # 如果作为主程序执行，则运行测试函数
    run_tests()
```