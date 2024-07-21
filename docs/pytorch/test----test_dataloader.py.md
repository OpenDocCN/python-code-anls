# `.\pytorch\test\test_dataloader.py`

```py
# Owner(s): ["module: dataloader"]

import ctypes  # 提供对 C 语言数据类型的支持
import errno  # 提供对错误码的支持
import faulthandler  # 提供用于调试的异常处理器
import functools  # 提供函数式编程的工具
import gc  # 提供对 Python 垃圾回收机制的接口
import itertools  # 提供迭代器生成函数
import math  # 提供数学运算函数
import operator  # 提供内置运算符函数
import os  # 提供对操作系统功能的访问
import signal  # 提供对信号处理的支持
import sys  # 提供对 Python 解释器的访问
import tempfile  # 提供创建临时文件和目录的功能
import time  # 提供时间相关的函数
import unittest  # 提供单元测试框架
import warnings  # 提供警告管理工具

import torch  # 导入 PyTorch 深度学习库
import torch.utils.data.datapipes as dp  # 导入 PyTorch 数据处理模块
from torch import multiprocessing as mp  # 导入 PyTorch 多进程模块
from torch._utils import ExceptionWrapper  # 导入 PyTorch 异常处理工具
from torch.testing._internal.common_device_type import instantiate_device_type_tests  # 导入设备类型测试函数
from torch.testing._internal.common_utils import (  # 导入常用的测试工具函数和变量
    IS_CI,
    IS_JETSON,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
    NO_MULTIPROCESSING_SPAWN,
    parametrize,
    run_tests,
    skipIfNoDill,
    skipIfRocm,
    slowTest,
    TEST_CUDA,
    TEST_NUMPY,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    TestCase,
)
from torch.utils.data import (  # 导入 PyTorch 数据集和数据加载相关模块
    _utils,
    ChainDataset,
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    IterDataPipe,
    StackDataset,
    Subset,
    TensorDataset,
)
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL  # 导入多进程状态检查间隔常量
from torch.utils.data.datapipes.iter import IterableWrapper  # 导入迭代器包装类
from torch.utils.data.dataset import random_split  # 导入数据集随机划分函数

try:
    import psutil  # 尝试导入 psutil 库来获取系统进程和系统利用率信息

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None
    err_msg = (
        "psutil not found. Some critical data loader tests relying on it "
        "(e.g., TestDataLoader.test_proper_exit) will not run."
    )
    if IS_CI:
        raise ModuleNotFoundError(err_msg) from None
    else:
        warnings.warn(err_msg)

try:
    import numpy as np  # 尝试导入 NumPy 库

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None
skipIfNoNumpy = unittest.skipIf(not HAS_NUMPY, "no NumPy")

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

TEST_CUDA_IPC = (
    torch.cuda.is_available()  # 检查是否支持 CUDA
    and sys.platform != "darwin"  # 排除 macOS 系统
    and sys.platform != "win32"  # 排除 Windows 系统
    and not IS_JETSON  # 排除 Jetson 系统
    and not TEST_WITH_ROCM  # 排除 ROCm 环境
)  # https://github.com/pytorch/pytorch/issues/90940

TEST_MULTIGPU = TEST_CUDA_IPC and torch.cuda.device_count() > 1  # 检查是否支持多 GPU

if TEST_CUDA_IPC:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")  # 设置 CUDA 分配器设置为禁用可扩展段

if not NO_MULTIPROCESSING_SPAWN:
    # We want to use `spawn` if able because some of our tests check that the
    # data loader terminates gracefully. To prevent hanging in the testing
    # process, such data loaders are run in a separate subprocess.
    #
    # We also want to test the `pin_memory=True` configuration, thus `spawn` is
    # required to launch such processes and they initialize the CUDA context.
    #
    # Mixing different start method is a recipe for disaster (e.g., using a fork
    # `mp.Event` with a spawn `mp.Process` segfaults). So we set this globally
    # to avoid bugs.
    #
    # 获取一个多进程上下文，因为某些测试或第三方库会在导入时设置 start_method，
    # 再次设置会触发 `RuntimeError` 异常。
    mp = mp.get_context(method="spawn")
# 设置进程等待超时时间为60秒
# 在一些共享物理 CPU 资源的环境中，例如 CI，进程间通信的时间可能会有很大变化。
# 在这里将超时时间设置为60秒，参考了 CPython 的 multiprocessing 设置，以避免一些 CI 构建中的不稳定性（参见 pytorch/pytorch#14501, pytorch/pytorch#16608）。
JOIN_TIMEOUT = 60.0  # seconds


# 支持的多进程上下文列表，包括 None 和所有 torch.multiprocessing.get_all_start_methods() 返回的方法
supported_multiprocessing_contexts = [None] + list(
    torch.multiprocessing.get_all_start_methods()
)


# 全局定义的 collate_fn 函数，为了支持 pickle，返回克隆后的批次数据
def _clone_collate(b):
    return [x.clone() for x in b]


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestDatasetRandomSplit(TestCase):
    def test_lengths_must_equal_dataset_size(self):
        # 确保长度参数与数据集大小相等时会抛出 ValueError 异常
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [1, 2])

    def test_splits_have_correct_size(self):
        # 测试 random_split 函数生成的分割是否具有正确的大小
        splits = random_split([1, 2, 3, 4, 5, 6], [2, 4])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(len(splits[1]), 4)

        splits = random_split([1, 2, 3, 4, 5, 6], [0.5, 0.5])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 3)
        self.assertEqual(len(splits[1]), 3)

        # 测试奇数大小的 round-robin 分割
        self.assertEqual(
            len(
                random_split(
                    range(3), [0.5, 0.5], generator=torch.Generator().manual_seed(1)
                )
            ),
            2,
        )

        # 测试奇数大小的 round-robin 分割
        splits = random_split(
            range(106), [0.1, 0.2, 0.3, 0.4], generator=torch.Generator().manual_seed(1)
        )
        self.assertEqual(len(splits[0]), 11)
        self.assertEqual(len(splits[1]), 22)
        self.assertEqual(len(splits[2]), 31)
        self.assertEqual(len(splits[3]), 42)
    def test_splits_are_mutually_exclusive(self):
        # 测试函数：验证随机划分是否互斥
        data = [5, 2, 3, 4, 1, 6]
        # 使用 random_split 函数进行数据划分，按指定位置划分
        splits = random_split(data, [2, 4])
        all_values = []
        # 将划分后的数据转为列表并添加到 all_values 中
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()  # 对原始数据排序
        all_values.sort()  # 对合并后的数据排序
        # 断言：划分后的数据是否与原始数据排序后一致
        self.assertListEqual(data, all_values)

        splits = random_split(data, [0.33, 0.67])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

        data = [1, 2, 3, 4]
        splits = random_split(data, [0.25, 0.75])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

    def test_splits_indexing_type(self):
        r"""Indices generated by random_split
        should be of integer type
        """
        # 测试函数：验证 random_split 生成的索引应为整数类型

        class CustomDataset:
            def __init__(self, test_object, custom_list):
                self.data = custom_list
                self.test_object = test_object

            def __getitem__(self, key):
                # 断言：索引类型应为整数
                self.test_object.assertEqual(type(key), int)
                return self.data[key]

            def __len__(self):
                return len(self.data)

        x = [1, 2, 3, 4, 5]
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [5])[0]
        data_loader = DataLoader(dataset)
        for batch in data_loader:
            pass

        # fractional splitting
        # 分数划分
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [1.0])[0]
        data_loader = DataLoader(dataset)
        for batch in data_loader:
            pass
    def test_splits_reproducibility(self):
        # 测试随机分割的可重复性

        # 第一个断言：使用指定种子生成器进行随机分割，验证结果
        self.assertEqual(
            [
                list(x)
                for x in random_split(
                    range(10), [3, 7], generator=torch.Generator().manual_seed(1)
                )
            ],
            [[5, 6, 1], [2, 0, 8, 9, 3, 7, 4]],
        )

        # 第二个断言：使用相同种子生成器，验证两次随机分割结果是否相同
        self.assertEqual(
            random_split(
                range(100), [60, 40], generator=torch.Generator().manual_seed(42)
            ),
            random_split(
                range(100), [60, 40], generator=torch.Generator().manual_seed(42)
            ),
        )

        # 第三个断言：使用相同种子生成器，验证分割比例为 [0.5, 0.5] 的随机分割结果是否相同
        self.assertEqual(
            random_split(
                range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)
            ),
            random_split(
                range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)
            ),
        )

        # 第四个断言：使用相同种子生成器，验证分割比例为 [0.33, 0.33, 0.34] 的随机分割结果是否相同
        self.assertEqual(
            random_split(
                range(100),
                [0.33, 0.33, 0.34],
                generator=torch.Generator().manual_seed(42),
            ),
            random_split(
                range(100),
                [0.33, 0.33, 0.34],
                generator=torch.Generator().manual_seed(42),
            ),
        )

    def test_incomplete_fractional_splits(self):
        # 测试不完整的分数分割

        # 第一个断言：应当抛出 ValueError，因为分数列表总和不为 1
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [0.1])

        # 第二个断言：应当抛出 ValueError，因为分数大于 1
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [1.1])

    def test_splits_generator(self):
        # 测试随机分割的生成器

        # 第一个断言：没有指定生成器时，随机分割会影响默认生成器
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5])
        b = torch.rand(10)
        self.assertNotEqual(a, b)

        # 第二个断言：指定了生成器时，随机分割不会影响默认生成器
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5], generator=torch.Generator().manual_seed(42))
        b = torch.rand(10)
        self.assertEqual(a, b)

    def test_slicing_of_subset_of_dataset(self):
        # 测试数据集子集的切片操作

        # 初始化数据集和其子集
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])

        # 第一个断言：验证整个子集与整个数据集的切片是否相同
        self.assertEqual(subset_of_dataset[:], dataset[:])

        # 第二个断言：验证子集的部分切片操作
        self.assertEqual(subset_of_dataset[1:2], dataset[1:2])

        # 第三个断言：验证子集的间隔切片操作
        self.assertEqual(subset_of_dataset[0:-1:2], dataset[0:-1:2])

        # 测试从随机分割中获取的子集的切片操作
        subset1, subset2 = random_split(dataset, [3, 2])

        # 第四个断言：验证子集1与其在数据集中对应切片的一致性
        self.assertEqual(subset1[:], dataset[subset1.indices[:]])

        # 第五个断言：验证子集1的部分切片操作一致性
        self.assertEqual(subset1[0:2], dataset[subset1.indices[0:2]])

        # 第六个断言：验证子集1的间隔切片操作一致性
        self.assertEqual(subset1[0:-1:2], dataset[subset1.indices[0:-1:2]])
    # 定义测试方法，用于测试 Subset 对象的切片操作
    def test_slicing_of_subset_of_subset(self):
        # 创建一个包含五个元素的 TensorDataset 对象
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        # 创建 dataset 的一个子集，包含所有元素
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        # 在 subset_of_dataset 上再创建一个子集，也包含所有元素
        subset_of_subset = Subset(subset_of_dataset, [0, 1, 2, 3, 4])
        # 验证 subset_of_subset 切片后的结果与 dataset 切片后的结果相同
        self.assertEqual(subset_of_subset[:], dataset[:])
        # 验证 subset_of_subset 的前两个元素切片后的结果与 dataset 的前两个元素切片后的结果相同
        self.assertEqual(subset_of_subset[0:2], dataset[0:2])
        # 验证 subset_of_subset 从第一个元素开始，每隔一个元素切片后的结果与 dataset 从第一个元素开始，每隔一个元素切片后的结果相同
        self.assertEqual(subset_of_subset[0:-1:2], dataset[0:-1:2])
        # 对随机划分的 subset_of_subset1 进行切片操作的测试
        subset1, subset2 = random_split(dataset, [4, 1])
        # 将 subset1 划分为两个子集，其中 subset_of_subset1 包含前三个元素
        subset_of_subset1, subset_of_subset2 = random_split(subset1, [3, 1])
        # 通过索引获取 subset_of_subset1 对应的 dataset 元素
        idx = [subset1.indices[i] for i in subset_of_subset1.indices]
        # 验证 subset_of_subset1 切片后的结果与 dataset 中对应 idx 的元素切片后的结果相同
        self.assertEqual(subset_of_subset1[:], dataset[idx.copy()])
        # 验证 subset_of_subset1 的前两个元素切片后的结果与 dataset 中对应 idx 的前两个元素切片后的结果相同
        self.assertEqual(subset_of_subset1[0:2], dataset[idx[0:2]])
        # 验证 subset_of_subset1 从第一个元素开始，每隔一个元素切片后的结果与 dataset 中对应 idx 的从第一个元素开始，每隔一个元素切片后的结果相同
        self.assertEqual(subset_of_subset1[0:-1:2], dataset[idx[0:-1:2]])
# 定义一个继承自Dataset的类，用于生成包含CUDA Tensor的数据集
class CUDACountingDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    # 实现__getitem__方法，返回一个CUDA Tensor，内容为索引值i
    def __getitem__(self, i):
        return torch.as_tensor(i, device="cuda")

    # 实现__len__方法，返回数据集的长度n
    def __len__(self):
        return self.n


# 定义一个继承自Dataset的类，用于生成普通Tensor的数据集
class CountingDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    # 实现__getitem__方法，返回索引值i
    def __getitem__(self, i):
        return i

    # 实现__len__方法，返回数据集的长度n
    def __len__(self):
        return self.n


# 定义一个继承自IterableDataset的类，用于生成可迭代的数据集
class CountingIterableDataset(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    # 实现__iter__方法，返回一个迭代器，迭代范围为0到n-1
    def __iter__(self):
        return iter(range(self.n))

    # 实现__len__方法，返回数据集的长度n
    def __len__(self):
        return self.n


# 用于测试TensorDataset的单元测试类
@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestTensorDataset(TestCase):
    # 测试TensorDataset的__len__方法
    def test_len(self):
        # 创建一个包含两个Tensor的TensorDataset对象
        source = TensorDataset(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        # 断言TensorDataset对象的长度为15
        self.assertEqual(len(source), 15)

    # 测试TensorDataset的__getitem__方法
    def test_getitem(self):
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        # 创建一个包含两个Tensor的TensorDataset对象
        source = TensorDataset(t, l)
        # 遍历数据集中的每一个元素
        for i in range(15):
            # 断言第i个元素的第一个Tensor与t[i]相等
            self.assertEqual(t[i], source[i][0])
            # 断言第i个元素的第二个Tensor与l[i]相等
            self.assertEqual(l[i], source[i][1])

    # 测试TensorDataset的__getitem__方法，当其中一个Tensor为一维时
    def test_getitem_1d(self):
        t = torch.randn(15)
        l = torch.randn(15)
        # 创建一个包含两个Tensor的TensorDataset对象
        source = TensorDataset(t, l)
        # 遍历数据集中的每一个元素
        for i in range(15):
            # 断言第i个元素的第一个Tensor与t[i]相等
            self.assertEqual(t[i], source[i][0])
            # 断言第i个元素的第二个Tensor与l[i]相等
            self.assertEqual(l[i], source[i][1])

    # 测试TensorDataset的__getitem__方法，当只有一个Tensor时
    def test_single_tensor(self):
        t = torch.randn(5, 10)
        # 创建一个只包含一个Tensor的TensorDataset对象
        source = TensorDataset(t)
        # 断言TensorDataset对象的长度为5
        self.assertEqual(len(source), 5)
        # 遍历数据集中的每一个元素
        for i in range(5):
            # 断言第i个元素与t[i]相等
            self.assertEqual(t[i], source[i][0])

    # 测试TensorDataset的__getitem__方法，当包含多个Tensor时
    def test_many_tensors(self):
        t0 = torch.randn(5, 10, 2, 3, 4, 5)
        t1 = torch.randn(5, 10)
        t2 = torch.randn(5, 10, 2, 5)
        t3 = torch.randn(5, 10, 3, 7)
        # 创建一个包含四个Tensor的TensorDataset对象
        source = TensorDataset(t0, t1, t2, t3)
        # 断言TensorDataset对象的长度为5
        self.assertEqual(len(source), 5)
        # 遍历数据集中的每一个元素
        for i in range(5):
            # 断言第i个元素的第一个Tensor与t0[i]相等
            self.assertEqual(t0[i], source[i][0])
            # 断言第i个元素的第二个Tensor与t1[i]相等
            self.assertEqual(t1[i], source[i][1])
            # 断言第i个元素的第三个Tensor与t2[i]相等
            self.assertEqual(t2[i], source[i][2])
            # 断言第i个元素的第四个Tensor与t3[i]相等
            self.assertEqual(t3[i], source[i][3])


# 用于测试StackDataset的单元测试类
@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestStackDataset(TestCase):
    # 测试传入空参数时是否抛出异常
    def test_empty(self):
        with self.assertRaisesRegex(
            ValueError, "At least one dataset should be passed"
        ):
            StackDataset()

    # 测试传入混合参数时是否抛出异常
    def test_mixed(self):
        with self.assertRaisesRegex(ValueError, "Supported either"):
            StackDataset(
                TensorDataset(torch.randn(15, 10)), a=TensorDataset(torch.randn(10, 15))
            )
    def test_size_mismatch(self):
        # 测试当数据集大小不匹配时是否引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            # 创建 StackDataset 对象，传入两个不同大小的 TensorDataset 对象
            StackDataset(
                TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(10, 15))
            )
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            # 创建 StackDataset 对象，使用命名参数传入两个不同大小的 TensorDataset 对象
            StackDataset(
                a=TensorDataset(torch.randn(15, 10)),
                b=TensorDataset(torch.randn(10, 15)),
            )

    def test_len(self):
        # 测试 StackDataset 对象的长度
        # 创建 StackDataset 对象，包含一个大小为 (15, 10) 的 TensorDataset 对象和一个大小为 (15,) 的 TensorDataset 对象
        source = StackDataset(
            TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(15))
        )
        # 断言 StackDataset 对象的长度为 15
        self.assertEqual(len(source), 15)
        
        # 创建 StackDataset 对象，只包含一个大小为 (15, 10) 的 TensorDataset 对象
        source = StackDataset(TensorDataset(torch.randn(15, 10)))
        # 断言 StackDataset 对象的长度为 15
        self.assertEqual(len(source), 15)
        
        # 创建 StackDataset 对象，使用命名参数传入一个大小为 (15, 10) 的 TensorDataset 对象
        source = StackDataset(
            a=TensorDataset(torch.randn(15, 10)), b=TensorDataset(torch.randn(15))
        )
        # 断言 StackDataset 对象的长度为 15
        self.assertEqual(len(source), 15)
        
        # 创建 StackDataset 对象，只包含一个大小为 (15, 10) 的 TensorDataset 对象
        source = StackDataset(a=TensorDataset(torch.randn(15, 10)))
        # 断言 StackDataset 对象的长度为 15
        self.assertEqual(len(source), 15)

    def test_single(self):
        # 测试 StackDataset 对象包含单个数据集时的行为
        t = TensorDataset(torch.randn(15, 10))
        # 创建 StackDataset 对象，只包含一个 TensorDataset 对象
        source = StackDataset(t)
        # 遍历数据集中的元素
        for i in range(15):
            # 断言 StackDataset 中的元素与原始 TensorDataset 中的元素相等
            self.assertEqual(t[i], source[i][0])
        
        # 创建 StackDataset 对象，使用命名参数传入一个 TensorDataset 对象
        source = StackDataset(a=t)
        # 遍历数据集中的元素
        for i in range(15):
            # 断言 StackDataset 中的元素与原始 TensorDataset 中的元素相等
            self.assertEqual(t[i], source[i]["a"])

    def test_getitem(self):
        # 测试 StackDataset 对象的 __getitem__ 方法
        t = TensorDataset(torch.randn(15, 10))
        l = TensorDataset(torch.randn(15, 5, 4))
        # 创建 StackDataset 对象，传入两个 TensorDataset 对象
        source = StackDataset(t, l)
        # 遍历数据集中的元素
        for i in range(15):
            # 断言 StackDataset 中的第一个元素与 t 中的元素相等
            self.assertEqual(t[i], source[i][0])
            # 断言 StackDataset 中的第二个元素与 l 中的元素相等
            self.assertEqual(l[i], source[i][1])
        
        # 创建 StackDataset 对象，使用命名参数传入两个 TensorDataset 对象
        source = StackDataset(a=t, b=l)
        # 遍历数据集中的元素
        for i in range(15):
            # 断言 StackDataset 中的元素与原始 TensorDataset 中的元素相等
            self.assertEqual(t[i], source[i]["a"])
            self.assertEqual(l[i], source[i]["b"])

    def test_getitems(self):
        class GetItemsDataset(Dataset):
            def __init__(self):
                self.data = torch.randn(4)

            def __getitem__(self, item):
                return self.data[item]

            def __getitems__(self, items):
                return self.data[items]

            def __len__(self):
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        # 创建 StackDataset 对象，传入一个自定义的 Dataset 对象和一个列表
        source = StackDataset(t, l)
        # 调用 StackDataset 对象的 __getitems__ 方法，获取指定索引的数据
        batch = source.__getitems__([0, 1, 2, 3])
        # 遍历数据批次中的元素
        for i in range(4):
            # 断言 StackDataset 中的第一个元素与 t 中的元素相等
            self.assertEqual(t[i], batch[i][0])
            # 断言 StackDataset 中的第二个元素与 l 中的元素相等
            self.assertEqual(l[i], batch[i][1])

        # 创建 StackDataset 对象，使用命名参数传入一个自定义的 Dataset 对象和一个列表
        source = StackDataset(t=t, l=l)
        # 调用 StackDataset 对象的 __getitems__ 方法，获取指定索引的数据
        batch = source.__getitems__([0, 1, 2, 3])
        # 遍历数据批次中的元素
        for i in range(4):
            # 断言 StackDataset 中的元素与原始 Dataset 中的元素相等
            self.assertEqual(t[i], batch[i]["t"])
            self.assertEqual(l[i], batch[i]["l"])
    # 定义一个测试方法，用于测试在索引错误情况下是否会引发异常
    def test_getitems_raises_index_error(self):
        # 定义一个继承自 Dataset 的子类 GetItemsDataset
        class GetItemsDataset(Dataset):
            def __init__(self):
                # 初始化方法，生成一个包含四个随机数的张量
                self.data = torch.randn(4)

            # 实现 __getitem__ 方法，用于按索引获取数据
            def __getitem__(self, item):
                return self.data[item]

            # 定义一个名为 __getitems__ 的方法，意图可能是获取多个项的数据，但拼写错误
            def __getitems__(self, items):
                return self.data[items]

            # 实现 __len__ 方法，返回数据集的长度
            def __len__(self):
                return 4

        # 创建 GetItemsDataset 类的实例 t
        t = GetItemsDataset()
        # 创建列表 l 包含 [1, 2, 3, 4]
        l = [1, 2, 3, 4]

        # 创建 StackDataset 类的实例 source，将 t 和 l 作为参数传入
        source = StackDataset(t, l)

        # 使用 assertRaises 断言，在执行以下代码块时会抛出 IndexError 异常
        with self.assertRaises(IndexError):
            # 调用 source 对象的 __getitems__ 方法，传入参数 [0, 4]
            source.__getitems__([0, 4])

    # 定义一个测试方法，用于测试在值错误情况下是否会引发异常
    def test_getitems_value_error(self):
        # 定义一个继承自 Dataset 的子类 GetItemsDataset
        class GetItemsDataset(Dataset):
            def __init__(self):
                # 初始化方法，生成一个包含四个随机数的张量
                self.data = torch.randn(4)

            # 实现 __getitem__ 方法，用于按索引获取数据
            def __getitem__(self, item):
                return self.data[item]

            # 实现 __getitems__ 方法，但返回数据的部分，同时存在错误，应返回全部数据
            def __getitems__(self, items):
                return self.data[items][:-1]  # 返回少了的部分

            # 实现 __len__ 方法，返回数据集的长度
            def __len__(self):
                return 4

        # 创建 GetItemsDataset 类的实例 t
        t = GetItemsDataset()
        # 创建列表 l 包含 [1, 2, 3, 4]
        l = [1, 2, 3, 4]

        # 创建 StackDataset 类的实例 source，将 t 和 l 作为参数传入
        source = StackDataset(t, l)

        # 使用 assertRaisesRegex 断言，在执行以下代码块时会抛出 ValueError 异常，
        # 并且异常信息为 "Nested dataset's output size mismatch. Expected 4, got 3"
        with self.assertRaisesRegex(
            ValueError, "Nested dataset's output size mismatch. Expected 4, got 3"
        ):
            # 调用 source 对象的 __getitems__ 方法，传入参数 [0, 1, 2, 3]
            source.__getitems__([0, 1, 2, 3])
# 如果测试条件 TEST_WITH_TSAN 为真，则跳过此测试类，因为在 TSAN 下会出现特定错误
@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestConcatDataset(TestCase):
    # 测试拼接两个单例数据集
    def test_concat_two_singletons(self):
        result = ConcatDataset([[0], [1]])
        self.assertEqual(2, len(result))  # 断言结果长度为2
        self.assertEqual(0, result[0])  # 断言第一个元素为0
        self.assertEqual(1, result[1])  # 断言第二个元素为1

    # 测试拼接两个非单例数据集
    def test_concat_two_non_singletons(self):
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))  # 断言结果长度为10
        self.assertEqual(0, result[0])  # 断言第一个元素为0
        self.assertEqual(5, result[5])  # 断言第六个元素为5

    # 测试拼接包含空数据集的情况
    def test_concat_two_non_singletons_with_empty(self):
        # 添加一个空数据集，检查是否正确处理
        result = ConcatDataset([[0, 1, 2, 3, 4], [], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))  # 断言结果长度为10
        self.assertEqual(0, result[0])  # 断言第一个元素为0
        self.assertEqual(5, result[5])  # 断言第六个元素为5

    # 测试访问超出索引范围的情况是否会抛出 IndexError
    def test_concat_raises_index_error(self):
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with self.assertRaises(IndexError):
            # 这里访问索引 11，应该抛出 IndexError
            result[11]

    # 测试合并数据集的结果
    def test_add_dataset(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d2 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d3 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        result = d1 + d2 + d3
        self.assertEqual(21, len(result))  # 断言结果长度为21
        # 断言第一个数据集的第一个元素与结果的第一个元素在误差范围内相等
        self.assertEqual(0, (d1[0][0] - result[0][0]).abs().sum())
        # 断言第二个数据集的第一个元素与结果的第八个元素在误差范围内相等
        self.assertEqual(0, (d2[0][0] - result[7][0]).abs().sum())
        # 断言第三个数据集的第一个元素与结果的第十五个元素在误差范围内相等
        self.assertEqual(0, (d3[0][0] - result[14][0]).abs().sum())

    # 测试当传入 IterableDataset 时是否会抛出 AssertionError
    def test_iterable_dataset_err(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        it1 = CountingIterableDataset(5)
        it2 = CountingIterableDataset(10)

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            # 合并包含 IterableDataset 的数据集，应该抛出 AssertionError
            ConcatDataset([d1, it2, it1])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            # 单独使用 IterableDataset，应该抛出 AssertionError
            ConcatDataset([it2])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            # 合并包含 IterableDataset 的数据集，应该抛出 AssertionError
            ConcatDataset([it1, d1])


# 接受一个虚拟变量，以便作为 `worker_init_fn` 使用
def set_faulthander_if_available(_=None):
    faulthandler.enable(sys.__stderr__)  # 启用 faulthandler 输出到 stderr
    if not IS_WINDOWS:
        # 如果不是在 Windows 平台，注册 SIGUSR1 信号，用于处理错误
        # chain=False 防止默认行为杀死进程
        faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, chain=False)


set_faulthander_if_available()


# 必须要求进程 `pid` 已经调用了 `set_faulthander_if_available`
def print_traces_of_all_threads(pid):
    if not IS_WINDOWS:
        # 如果可用，使用自定义信号输出所有线程的跟踪信息
        os.kill(pid, signal.SIGUSR1)
    else:
        # 如果没有使用faulthandler.enable()设置信号处理程序，
        # 则使用os.kill(pid, signal.SIGSEGV)发送SIGSEGV信号来终止子进程。
        os.kill(pid, signal.SIGSEGV)

    # 在父进程中等待一段时间，以便子进程有足够的时间打印输出信息。
    time.sleep(5)
# 定义一个继承自 `mp.Process` 的错误追踪进程类 `ErrorTrackingProcess`，用于捕获并存储首次遇到的异常。
# 参考自 https://stackoverflow.com/a/33599967
class ErrorTrackingProcess(mp.Process):

    # 初始化方法，设置管道连接和异常变量，并可选择是否禁用标准错误输出。
    # 在 Python 2 中，不支持 `def fn(x, *args, key=val, **kwargs)` 的语法。
    def __init__(self, disable_stderr=True, **kwargs):
        super().__init__(**kwargs)
        self._pconn, self._cconn = mp.Pipe()  # 创建管道用于进程间通信
        self._exception = None  # 初始化异常变量为 None
        self.disable_stderr = disable_stderr  # 是否禁用标准错误输出的标志

    # 运行方法，设置可能的故障处理器，如果禁用标准错误输出，则将 stderr 重定向到 /dev/null。
    # 执行 `super().run()`，并在结束后向管道发送信号。
    def run(self):
        set_faulthander_if_available()
        if self.disable_stderr:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            super().run()  # 调用父类的运行方法
            self._cconn.send(None)  # 向子进程发送空消息
        except Exception:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))  # 发送捕获到的异常信息
            raise

    # 打印所有线程的堆栈跟踪，前提是进程仍然存活且未禁用 stderr 输出。
    # 如果没有 `SIGUSR1` 信号，`set_faulthander_if_available` 会启用 `faulthandler.enable()`，
    # 而 `print_traces_of_all_threads` 可能会终止进程。因此，在调用之前先检查异常。
    def print_traces_of_all_threads(self):
        assert self.is_alive(), "can only use print_traces_of_all_threads if the process is alive"
        assert not self.disable_stderr, "do not disable stderr if you use print_traces_of_all_threads"
        _ = self.exception  # 获取异常信息
        print_traces_of_all_threads(self.pid)  # 打印所有线程的堆栈跟踪

    # 属性方法，用于获取进程的异常信息。
    @property
    def exception(self):
        if self._pconn.poll():  # 检查管道是否有数据可接收
            self._exception = self._pconn.recv()  # 接收异常信息
        if self._exception is None:
            return None
        else:
            return self._exception.exc_type(self._exception.exc_msg)  # 返回异常类型和消息

    # 发送信号方法，用于向进程发送信号 `signum`，可选择是否忽略 `ESRCH` 错误。
    # 如果进程不存在且未忽略 `ESRCH` 错误，则抛出 `OSError` 异常。
    def send_signal(self, signum, ignore_ESRCH=False):
        try:
            os.kill(self.pid, signum)  # 发送信号给进程
        except OSError as e:
            if not ignore_ESRCH or e.errno != errno.ESRCH:
                raise


# 表示一个数据集的基类 `ErrorDataset`，继承自 `Dataset`。
class ErrorDataset(Dataset):

    # 初始化方法，设置数据集的大小 `size`。
    def __init__(self, size):
        self.size = size

    # 返回数据集的长度。
    def __len__(self):
        return self.size


# 表示一个会导致段错误的数据集 `SegfaultDataset`，继承自 `Dataset`。
class SegfaultDataset(Dataset):

    # 初始化方法，设置数据集的大小 `size`。
    def __init__(self, size):
        self.size = size

    # 获取指定索引处的数据，返回 ctypes 的空字符串。
    def __getitem__(self, idx):
        return ctypes.string_at(0)

    # 返回数据集的长度。
    def __len__(self):
        return self.size


# 表示一个会休眠指定秒数后返回数据的数据集 `SleepDataset`，继承自 `Dataset`。
class SleepDataset(Dataset):

    # 初始化方法，设置数据集的大小 `size` 和休眠时间 `sleep_sec`。
    def __init__(self, size, sleep_sec):
        self.size = size
        self.sleep_sec = sleep_sec
        self.sleeped = False

    # 获取指定索引处的数据，如果未曾休眠，则休眠指定时间后返回索引。
    def __getitem__(self, idx):
        if not self.sleeped:
            time.sleep(self.sleep_sec)
            self.sleeped = True
        return idx

    # 返回数据集的长度。
    def __len__(self):
        return self.size


# 表示一个数据集，用于设置随机种子 `SeedDataset`，继承自 `Dataset`。
class SeedDataset(Dataset):

    # 初始化方法，设置数据集的大小 `size`。
    def __init__(self, size):
        self.size = size
    # 定义特殊方法 __getitem__，用于获取对象中指定索引处的元素
    def __getitem__(self, idx):
        # 返回使用 torch 模块生成的初始种子值
        return torch.initial_seed()

    # 定义特殊方法 __len__，返回对象的长度
    def __len__(self):
        # 返回对象的 size 属性作为长度
        return self.size
class WorkerSpecificIterableDataset(IterableDataset):
    # 定义一个特定于工作器的可迭代数据集类，继承自IterableDataset
    def __init__(self, sizes_for_all_workers):
        # 初始化方法，接收所有工作器的数据大小列表作为参数
        self.sizes_for_all_workers = sizes_for_all_workers

    def __iter__(self):
        # 迭代器方法，获取当前工作器的信息
        worker_info = torch.utils.data.get_worker_info()
        # 断言确保工作器信息不为None
        assert worker_info is not None
        # 返回一个迭代器，迭代范围是当前工作器对应的数据大小范围
        return iter(range(self.sizes_for_all_workers[worker_info.id]))

    def __len__(self):
        # 返回所有工作器数据大小之和作为数据集的长度
        return sum(self.sizes_for_all_workers)


# Inspired by https://stackoverflow.com/a/26703365
# If all workers will call `sync_once`, they will be blocked until all workers
# reach the call (i.e., acting like a barrier).
# This can be used to ensure that each worker at least processes one data.
class SynchronizedDataset(Dataset):
    # 同步数据集类，用于确保所有工作器至少处理一条数据
    def __init__(self, size, batch_size, num_workers):
        # 初始化方法，确保数据大小大于等于工作器数量乘以批次大小
        assert size >= num_workers * batch_size
        # 使用进程共享的整型变量来记录计数，初始值为0
        self.count = mp.Value("i", 0, lock=True)
        # 使用信号量来实现同步屏障
        self.barrier = mp.Semaphore(0)
        self.num_workers = num_workers
        self.size = size

    def sync_once(self):
        # 一次同步操作，使用计数变量和信号量实现同步屏障
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value == self.num_workers:
                self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()

    def __getitem__(self, idx):
        # 获取数据项的方法，但在这个类中未实现具体逻辑
        raise NotImplementedError

    def __len__(self):
        # 返回数据集的大小
        return self.size


class EmptyTensorDataset(torch.utils.data.Dataset):
    # 空张量数据集类，继承自PyTorch的Dataset类
    def __init__(self, len):
        # 初始化方法，接收数据集长度作为参数
        self.len = len

    def __len__(self):
        # 返回数据集的长度
        return self.len

    def __getitem__(self, any):
        # 获取数据项的方法，但在这个类中未实现具体逻辑，返回一个空张量
        return torch.empty(0)


class SynchronizedSeedDataset(SynchronizedDataset):
    # 同步种子数据集类，继承自SynchronizedDataset类
    def __getitem__(self, idx):
        # 获取数据项的方法，调用父类的同步方法后返回当前进程的随机种子值
        self.sync_once()
        return torch.initial_seed()


def _test_timeout(persistent_workers):
    # 测试超时方法，用于测试数据加载器在超时条件下的行为
    dataset = SleepDataset(10, 3)  # 使用SleepDataset创建数据集
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        timeout=1,
        persistent_workers=persistent_workers,
    )
    _ = next(iter(dataloader))  # 获取数据加载器的下一个批次数据


def _test_timeout_pin_memory(persistent_workers):
    # 测试超时方法（内存固定），用于测试带有内存固定的数据加载器在超时条件下的行为
    dataset = SleepDataset(10, 3)  # 使用SleepDataset创建数据集
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        timeout=1,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    _ = next(iter(dataloader))  # 获取数据加载器的下一个批次数据


def _test_large_sampler_indices(persistent_workers):
    # 测试大型采样器索引方法，用于测试大型索引在数据加载器中的行为
    # 参考 https://github.com/pytorch/pytorch/issues/48666
    dataloader = torch.utils.data.DataLoader(
        EmptyTensorDataset(10000000),  # 使用空张量数据集，长度为10000000
        batch_size=40960,
        persistent_workers=persistent_workers,
        num_workers=1,
    )

    it = iter(dataloader)  # 获取数据加载器的迭代器

    for x in it:
        assert x.numel() == 0  # 断言每个批次数据的元素数量为0
        raise RuntimeError("My Error")  # 抛出运行时错误


def disable_stderr(worker_id):
    r"""
    Avoids printing "ERROR: Unexpected segmentation fault encountered in worker."
    from workers. Since worker signal handler prints with low-level write(),
    this has to be done on OS level via dup.
    """
    # 禁用标准错误输出的方法，避免工作器打印由于段错误而引起的信息
    # 由于工作器信号处理程序使用低级别的写操作进行打印，因此必须通过OS级别的dup来完成此操作
    # 用作 test_segfault 的 worker_init_fn。
    """
    sys.stderr.flush()  # 刷新标准错误流，清空库的缓冲区，dup2 无法清空这些缓冲区
    
    # 不能使用 with 块，否则在函数结束时文件描述符将被关闭。
    with open(os.devnull, "w") as devnull:
        # 使用 dup2 将 /dev/null 的文件描述符复制到标准错误流的文件描述符上，
        # 从而将标准错误流重定向到 /dev/null，实现屏蔽错误输出。
        os.dup2(devnull.fileno(), sys.stderr.fileno())
def _test_segfault():
    # 创建 SegfaultDataset 实例，大小为 10
    dataset = SegfaultDataset(10)
    # 创建 DataLoader 对象，使用 dataset 作为数据源，批大小为 2，使用 2 个工作线程，禁用标准错误流
    dataloader = DataLoader(
        dataset, batch_size=2, num_workers=2, worker_init_fn=disable_stderr
    )
    # 获取下一个批次的数据以触发潜在的段错误
    _ = next(iter(dataloader))


def _test_no_segfault():
    # 创建简单的列表数据集
    dataset = [1, 2, 3]
    # 获取当前线程数
    num_threads = torch.get_num_threads()
    # 如果当前线程数小于 4，设置线程数为 4
    if num_threads < 4:
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(num_threads)
    # 获取多进程上下文，使用 fork 方法
    mp_ctx = torch.multiprocessing.get_context(method="fork")
    # 创建 DataLoader 对象，使用 dataset 作为数据源，使用 1 个工作线程，禁用标准错误流，指定多进程上下文为 mp_ctx
    dataloader = DataLoader(
        dataset,
        num_workers=1,
        worker_init_fn=disable_stderr,
        multiprocessing_context=mp_ctx,
    )
    # 获取下一个批次的数据
    _ = next(iter(dataloader))


class TestProperExitDataset(Dataset):
    def __init__(self, size, error_event):
        self.size = size
        self.error_event = error_event

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 获取工作进程信息
        worker_info = torch.utils.data.get_worker_info()
        # 如果设置了 error_event，并且在最后一个工作进程中发生错误，抛出 RuntimeError
        if (
            self.error_event is not None
            and self.error_event.is_set()
            and worker_info.id == worker_info.num_workers - 1
        ):
            raise RuntimeError("Worker error")
        # 返回包含 idx 的张量
        return torch.tensor([idx])


class TestProperExitIterableDataset(IterableDataset):
    def __init__(self, size, error_event):
        self.error_event = error_event
        self.size = size
        self.remaining = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        # 获取工作进程信息
        worker_info = torch.utils.data.get_worker_info()
        # 如果设置了 error_event，并且在最后一个工作进程中发生错误，抛出 RuntimeError
        if (
            self.error_event is not None
            and self.error_event.is_set()
            and worker_info.id == worker_info.num_workers - 1
        ):
            raise RuntimeError("Worker error")
        # 减少剩余迭代次数，如果为负数则抛出 StopIteration
        self.remaining -= 1
        if self.remaining < 0:
            raise StopIteration
        # 返回一个包含 -1000 的张量
        return torch.tensor(-1000)


# See TestDataLoader.test_proper_exit for usage
def _test_proper_exit(
    is_iterable_dataset,
    use_workers,
    pin_memory,
    exit_method,
    hold_iter_reference,
    loader_setup_event,
    tester_setup_event,
    persistent_workers,
):
    # 根据 use_workers 确定使用的工作进程数
    num_workers = 2 if use_workers else 0

    # 如果 exit_method 是 "worker_error" 或 "worker_kill"，确保 use_workers 是 True
    if exit_method == "worker_error" or exit_method == "worker_kill":
        assert use_workers is True

    # 根据 exit_method 设置 worker_error_event
    if exit_method == "worker_error":
        worker_error_event = mp.Event()
    else:
        worker_error_event = None

    # 根据 is_iterable_dataset 选择合适的数据集类
    if is_iterable_dataset:
        ds = TestProperExitIterableDataset(7, worker_error_event)
    else:
        ds = TestProperExitDataset(12, worker_error_event)

    # 创建 DataLoader 对象，使用 ds 作为数据源，批大小为 1，不打乱数据顺序，使用指定的工作进程数，
    # 使用 pin_memory，初始化工作进程的函数为 set_faulthander_if_available，持久化工作进程根据 persistent_workers
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=set_faulthander_if_available,
        persistent_workers=persistent_workers,
    )

    # 设置错误迭代次数为 2
    error_it = 2
    if use_workers:
        # 如果使用多线程或多进程处理数据集，这里是一个魔法数，每个工作进程的预取数...
        # FIXME: 在这个数变成可配置之后进行更改。
        if is_iterable_dataset:
            # 如果数据集是可迭代的，确保数据集长度乘以工作进程数大于错误索引、预取数和一个基本的偏移量
            assert len(ds) * num_workers > (error_it + 2 + 1)
        else:
            # 如果数据集不是可迭代的，确保数据加载器长度大于（错误索引、预取数和一个基本的偏移量）乘以工作进程数
            assert len(loader) > (error_it + 2 + 1) * num_workers
    else:
        if is_iterable_dataset:
            # 如果数据集是可迭代的，确保数据集长度大于错误索引和一个基本的偏移量
            assert len(ds) > error_it + 1
        else:
            # 如果数据集不是可迭代的，确保数据加载器长度大于错误索引和一个基本的偏移量
            assert len(loader) > error_it + 1

    # 从数据加载器获取迭代器
    it = iter(loader)
    if use_workers:
        # 如果使用多线程或多进程处理数据集，获取工作进程列表
        workers = it._workers

    # 定义一个函数，用于结束指定进程ID的进程
    def kill_pid(pid):
        # 使用 psutil 根据进程ID获取进程对象
        psutil_p = psutil.Process(pid)
        # 终止进程
        psutil_p.kill()
        # 等待进程结束，超时时间为 JOIN_TIMEOUT
        psutil_p.wait(JOIN_TIMEOUT)
        # 断言进程已经不再运行
        assert not psutil_p.is_running()

    # 遍历数据加载器的迭代器
    for i, _ in enumerate(it):
        if i == 0:
            if not hold_iter_reference:
                # 如果不需要保留迭代器引用，删除迭代器和加载器对象
                del it
                del loader
            # 设置加载器已设置事件
            loader_setup_event.set()
            # 等待测试设置事件
            tester_setup_event.wait()
            # 确保工作进程仍然存活
            if use_workers:
                for w in workers:
                    assert w.is_alive()
            # 如果存在工作进程错误事件，设置该事件
            if worker_error_event is not None:
                worker_error_event.set()

        if i == error_it:
            # 如果达到指定错误索引
            if exit_method == "loader_error":
                # 抛出加载器错误异常
                raise RuntimeError("Loader error")
            elif exit_method == "loader_kill":
                # 终止当前进程
                kill_pid(os.getpid())
            elif exit_method == "worker_kill":
                # 终止最后一个工作进程
                kill_pid(workers[-1].pid)  # kill last worker

    if not hold_iter_reference:
        # 尝试触发 __del__ 清理，而不是守护子进程的自动退出。
        # 从技术上讲，应该自动触发，但我不想依赖 Python gc 的具体实现细节。
        gc.collect()
class TestWorkerInfoDataset(SynchronizedDataset):
    # 定义一个测试用的数据集类，继承自SynchronizedDataset

    def __getitem__(self, idx):
        # 重载getitem方法，获取数据集中索引为idx的元素
        self.sync_once()
        # 调用sync_once方法，确保数据同步一次
        return torch.tensor(self.value)
        # 返回一个包含self.value数据的PyTorch张量


# Should be used as worker_init_fn with TestWorkerInfoDataset.
# See _test_get_worker_info below for usage.
def _test_worker_info_init_fn(worker_id):
    # 定义一个用于初始化worker的函数，应与TestWorkerInfoDataset一起使用
    # 查看下面的_test_get_worker_info函数了解如何使用

    worker_info = torch.utils.data.get_worker_info()
    # 获取worker的信息，返回WorkerInfo对象

    assert (
        worker_id == worker_info.id
    ), "worker_init_fn and worker_info should have consistent id"
    # 断言确保worker_init_fn和worker_info的id一致

    assert (
        worker_id < worker_info.num_workers
    ), "worker_init_fn and worker_info should have valid id"
    # 断言确保worker_init_fn和worker_info的id有效

    assert (
        worker_info.seed == torch.initial_seed()
    ), "worker_init_fn and worker_info should have consistent seed"
    # 断言确保worker_init_fn和worker_info的种子值一致

    dataset = worker_info.dataset
    # 获取worker_info中的数据集对象

    assert isinstance(
        dataset, TestWorkerInfoDataset
    ), "worker_info should have correct dataset copy"
    # 断言确保worker_info中包含正确的数据集副本

    assert not hasattr(dataset, "value"), "worker_info should have correct dataset copy"
    # 断言确保worker_info中没有value属性

    # test that WorkerInfo attributes are read-only
    # 测试WorkerInfo对象的属性是否只读
    try:
        worker_info.id = 3999
    except RuntimeError as e:
        assert str(e) == "Cannot assign attributes to WorkerInfo objects"
    try:
        worker_info.a = 3
    except RuntimeError as e:
        assert str(e) == "Cannot assign attributes to WorkerInfo objects"

    for k in ["id", "num_workers", "seed", "dataset"]:
        assert f"{k}=" in repr(worker_info)
    # 断言确保WorkerInfo对象的属性在其repr中包含在预期的列表中

    dataset.value = [worker_id, os.getpid()]
    # 设置数据集的value属性为[worker_id, 当前进程的PID]


def _test_get_worker_info():
    # get_worker_info在主进程中返回None
    assert torch.utils.data.get_worker_info() is None

    num_workers = 2
    batch_size = 2
    dataset = TestWorkerInfoDataset(6, batch_size, num_workers)
    # 创建一个TestWorkerInfoDataset对象，设置参数为6个元素的数据集、batch大小为2、worker数量为2

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=_test_worker_info_init_fn,
    )
    # 创建一个DataLoader对象，用于加载数据集，设置batch大小、worker数量和worker初始化函数

    it = iter(dataloader)
    # 创建一个迭代器，用于迭代DataLoader对象

    data = []
    for d in it:
        data.append(d)  # noqa: PERF402
    # 遍历迭代器并将数据添加到data列表中

    worker_pids = [w.pid for w in it._workers]
    # 获取迭代器中所有worker的进程ID，并存储在worker_pids列表中

    data = torch.cat(data, 0)
    # 将data列表中的数据拼接成一个张量

    for d in data:
        # 对于每一个d，它是一个[worker_id, worker_pid]对，这是在_test_worker_info_init_fn中设置的
        assert d[1] == worker_pids[d[0]]
    # 断言确保每个数据项的第二个元素（PID）与对应worker的PID相匹配

    # get_worker_info在数据加载完成后的主进程中再次返回None
    assert torch.utils.data.get_worker_info() is None

    # 主进程的数据集从未被分配此属性
    assert not hasattr(dataset, "value")

    try:
        _ = dataset[0]
    except AttributeError:
        return
    # 预期捕获AttributeError异常

    raise RuntimeError("Expected AttributeError")


# test custom init function
# 测试自定义的初始化函数
def init_fn(worker_id):
    torch.manual_seed(12345)


# used with test_error_in_init
# 与test_error_in_init一起使用
class ErrorIterableDataset(IterableDataset):
    def __iter__(self):
        raise RuntimeError("Error in __iter__")


# used with test_error_in_init
# 与test_error_in_init一起使用
def error_worker_init_fn(_):
    raise RuntimeError("Error in worker_init_fn")


class BulkLoadingDataset(Dataset):
    def __init__(self, length):
        self.length = length
    # 定义特殊方法 __getitem__，用于通过索引获取对象中的元素
    def __getitem__(self, indices):
        # 断言 indices 是列表或元组类型，确保输入参数的类型正确
        assert isinstance(indices, (list, tuple))
        # 将 indices 转换为 Torch 张量（tensor）
        return torch.as_tensor(indices)

    # 定义特殊方法 __len__，返回对象的长度
    def __len__(self):
        # 返回对象的长度属性 self.length
        return self.length
class BulkLoadingSampler(torch.utils.data.Sampler):
    # 定义一个批量加载的采样器，继承自 PyTorch 的数据采样器类
    def __init__(self, dataset, batch_size):
        # 初始化函数，接收数据集和批量大小作为参数
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # 迭代器方法，用于生成数据集的索引列表的迭代器
        for x in torch.randperm(len(self.dataset)).split(self.batch_size):
            yield x.tolist()

    def __len__(self):
        # 返回数据集的批量数量，使用向上取整确保覆盖所有数据
        return int(math.ceil(len(self.dataset) / float(self.batch_size)))


class TestMultiEpochDataset(IterableDataset):
    # 定义一个多周期数据集类，继承自可迭代数据集基类
    def __init__(self, length):
        # 初始化函数，接收数据集的长度作为参数
        self.length = length

    def __iter__(self):
        # 迭代器方法，生成多周期数据集的迭代器
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None  # 确保工作信息存在
        worker_id = worker_info.id  # 获取工作进程的 ID
        for idx in range(self.length // worker_info.num_workers):
            yield worker_id  # 生成当前工作进程的 ID

    def __len__(self):
        # 返回数据集的长度
        return self.length


class CustomList(list):
    # 自定义列表类，继承自 Python 内置的列表类
    pass


class CustomDict(dict):
    # 自定义字典类，继承自 Python 内置的字典类
    pass


def row_processor(row):
    # 处理行数据的函数，将每个元素加一
    return np.add(row, 1)


def filter_len(row):
    # 过滤函数，判断行的长度是否为 4
    return len(row) == 4


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
@unittest.skipIf(
    TEST_WITH_ASAN,
    "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223",
)
class TestDataLoader(TestCase):
    # 数据加载器的单元测试类，继承自 unittest 的 TestCase 类
    def setUp(self):
        # 初始化测试用例的设置
        super().setUp()
        self.data = torch.randn(100, 2, 3, 5)  # 创建一个随机张量数据
        self.labels = torch.randperm(50).repeat(2)  # 创建随机排列的标签数据
        self.dataset = TensorDataset(self.data, self.labels)  # 创建张量数据集对象
        self.persistent_workers = False  # 设置是否使用持久化工作进程为 False

    def _get_data_loader(self, dataset, **kwargs):
        # 辅助函数：获取数据加载器对象
        persistent_workers = kwargs.get("persistent_workers", self.persistent_workers)
        if persistent_workers and kwargs.get("num_workers", 0) == 0:
            persistent_workers = False
        kwargs["persistent_workers"] = persistent_workers
        return DataLoader(dataset, **kwargs)  # 返回创建的数据加载器对象

    def _test_sequential(self, loader):
        # 辅助函数：测试顺序加载器
        batch_size = loader.batch_size
        if batch_size is None:
            # 如果批量大小为 None，则遍历加载器并进行断言
            for idx, (sample, target) in enumerate(loader):
                self.assertEqual(sample, self.data[idx])
                self.assertEqual(target, self.labels[idx])
            self.assertEqual(idx, len(self.dataset) - 1)  # 断言最后一个索引
        else:
            # 如果有指定批量大小，则以批量方式遍历加载器并进行断言
            for i, (sample, target) in enumerate(loader):
                idx = i * batch_size
                self.assertEqual(sample, self.data[idx : idx + batch_size])
                self.assertEqual(target, self.labels[idx : idx + batch_size])
            self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))
    # 测试数据加载器的洗牌功能
    def _test_shuffle(self, loader):
        # 创建一个字典，键为数据集大小的范围，值为0，用于跟踪数据点是否被找到
        found_data = dict.fromkeys(range(self.data.size(0)), 0)
        # 创建一个字典，键为标签集大小的范围，值为0，用于跟踪标签是否被找到
        found_labels = dict.fromkeys(range(self.labels.size(0)), 0)
        # 获取数据加载器的批大小
        batch_size = loader.batch_size
        # 如果批大小为None，逐批次处理数据加载器中的数据
        if batch_size is None:
            for i, (batch_samples, batch_targets) in enumerate(loader):
                # 提取当前批次的样本和目标
                sample, target = (batch_samples, batch_targets)
                # 遍历数据集中的数据点，查找匹配的样本
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        # 断言找到的数据点尚未被找到过
                        self.assertFalse(found_data[data_point_idx])
                        # 标记找到的数据点
                        found_data[data_point_idx] += 1
                        break
                # 断言目标与标签集中对应位置的标签相等
                self.assertEqual(target, self.labels[data_point_idx])
                # 标记找到的标签
                found_labels[data_point_idx] += 1
                # 断言找到的数据点和标签数目等于当前迭代数加一
                self.assertEqual(sum(found_data.values()), (i + 1))
                self.assertEqual(sum(found_labels.values()), (i + 1))
            # 断言迭代器达到数据集长度减一
            self.assertEqual(i, (len(self.dataset) - 1))
        else:
            # 如果有指定的批大小，逐批次处理数据加载器中的数据
            for i, (batch_samples, batch_targets) in enumerate(loader):
                # 逐一处理当前批次的样本和目标
                for sample, target in zip(batch_samples, batch_targets):
                    # 遍历数据集中的数据点，查找匹配的样本
                    for data_point_idx, data_point in enumerate(self.data):
                        if data_point.eq(sample).all():
                            # 断言找到的数据点尚未被找到过
                            self.assertFalse(found_data[data_point_idx])
                            # 标记找到的数据点
                            found_data[data_point_idx] += 1
                            break
                    # 断言目标与标签集中对应位置的标签相等
                    self.assertEqual(target, self.labels[data_point_idx])
                    # 标记找到的标签
                    found_labels[data_point_idx] += 1
                # 断言找到的数据点和标签数目等于当前迭代数乘以批大小
                self.assertEqual(sum(found_data.values()), (i + 1) * batch_size)
                self.assertEqual(sum(found_labels.values()), (i + 1) * batch_size)
            # 断言迭代器达到数据集长度减一整除批大小后的向下取整结果
            self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    # 测试在初始化过程中的错误处理
    def _test_error(self, loader):
        # 创建数据加载器的迭代器
        it = iter(loader)
        # 记录捕获的错误次数
        errors = 0
        # 循环处理迭代器中的每一项
        while True:
            try:
                next(it)
            except NotImplementedError:
                # 捕获到未实现的错误，增加错误计数
                errors += 1
            except StopIteration:
                # 断言捕获的错误次数等于数据集长度除以批大小后向上取整的结果
                self.assertEqual(
                    errors, math.ceil(float(len(loader.dataset)) / loader.batch_size)
                )
                return

    # 测试初始化过程中出现的错误
    def test_error_in_init(self):
        # 对于不同的工作线程数进行测试
        for num_workers in [0, 2]:
            # 获取使用ErrorIterableDataset的数据加载器
            loader = self._get_data_loader(
                ErrorIterableDataset(), num_workers=num_workers
            )
            # 断言在初始化期间捕获到RuntimeError，并且错误消息中包含"Error in __iter__"
            with self.assertRaisesRegex(RuntimeError, "Error in __iter__"):
                list(iter(loader))

        # 获取使用self.dataset的数据加载器，同时设置工作线程数和worker_init_fn
        loader = self._get_data_loader(
            self.dataset, num_workers=2, worker_init_fn=error_worker_init_fn
        )
        # 断言在初始化期间捕获到RuntimeError，并且错误消息中包含"Error in worker_init_fn"
        with self.assertRaisesRegex(RuntimeError, "Error in worker_init_fn"):
            list(iter(loader))
    def test_typing(self):
        from typing import List

        # 确保没有 TypeError

        # 定义一个继承自 Dataset 的类，其元素为 List[torch.Tensor] 类型
        class SomeDatasetClass(Dataset[List[torch.Tensor]]):
            pass

        # 定义一个函数 _create_dataloader，接受一个布尔型参数 is_train，返回一个 DataLoader，其元素为 List[torch.Tensor]
        def _create_dataloader(is_train: bool) -> DataLoader[List[torch.Tensor]]:
            pass

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "No 'resource' module on Windows")
    def test_fd_limit_exceeded(self):
        # 查看注释 [ DataLoader on Linux and open files limit ]
        import subprocess

        # 使用 subprocess 模块的 check_output 函数执行系统命令
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
# 导入 torch 库，用于数据处理和计算
import torch
# 导入 resource 库，用于设置系统资源限制
import resource
# 从 torch.utils.data 中导入 DataLoader 和 IterableDataset 类
from torch.utils.data import DataLoader, IterableDataset

# 定义一个继承自 IterableDataset 的随机数据集类
class RandomDataset(IterableDataset):
    # 初始化方法，设置数据集长度和数据大小
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len  # 数据集长度
        self.size = size  # 数据大小

    # 迭代器方法，返回迭代器自身
    def __iter__(self):
        return self

    # 迭代方法，生成随机数据，直到数据集长度为 0
    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1  # 减少数据集长度
        return torch.randn(self.size)  # 返回指定大小的随机张量数据

try:
    keep_fds_alive = []  # 用于保持文件描述符活动的空列表
    # 设置系统文件描述符的资源限制
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    # 使用 DataLoader 加载 RandomDataset 类的实例
    # 设置使用 fork 进程模式，一个 worker 进程
    for random_t in DataLoader(RandomDataset(200, (2,2)), multiprocessing_context="fork",
                               num_workers=1):
        random_t.max(dim=0)  # 对随机张量数据进行维度为 0 的最大值计算
        keep_fds_alive.append(random_t)  # 将随机张量数据保存到列表中，保持其活动状态
except RuntimeError as e:
    # 捕获运行时错误，检查是否包含指定的错误信息
    assert "ulimit -n" in str(e)
    assert "set_sharing_strategy" in str(e)
    # 测试多个数据加载器的功能
    def test_multiple_dataloaders(self):
        # 对于支持的多进程上下文，分别测试以下代码块
        for multiprocessing_context in supported_multiprocessing_contexts:
            # 获取单线程数据加载器的迭代器
            loader1_it = iter(self._get_data_loader(self.dataset, num_workers=1))
            # 获取多线程数据加载器的迭代器，使用指定的多进程上下文
            loader2_it = iter(
                self._get_data_loader(
                    self.dataset,
                    num_workers=2,
                    multiprocessing_context=multiprocessing_context,
                )
            )
            # 从第一个单线程数据加载器中获取两个数据批次
            next(loader1_it)
            next(loader1_it)
            # 从第二个多线程数据加载器中获取一个数据批次
            next(loader2_it)
            next(loader2_it)
            # 再次从第一个单线程数据加载器中获取一个数据批次
            next(loader1_it)
            # 再次从第二个多线程数据加载器中获取一个数据批次
            next(loader2_it)
            # 删除迭代器，释放资源
            del loader1_it
            del loader2_it

    # 测试是否会发生段错误
    def test_segfault(self):
        # 创建一个跟踪错误的进程，目标函数为 _test_segfault
        p = ErrorTrackingProcess(target=_test_segfault)
        # 启动进程
        p.start()
        # 等待进程结束，设置超时时间
        p.join(JOIN_TIMEOUT)
        try:
            # 断言进程已经结束
            self.assertFalse(p.is_alive())
            # 断言进程退出码不为0
            self.assertNotEqual(p.exitcode, 0)
            # 如果运行在 Windows 系统上，断言异常类型为 OSError
            if IS_WINDOWS:
                self.assertIsInstance(p.exception, OSError)
                # 断言异常信息包含 "access violation reading "
                self.assertRegex(str(p.exception), r"access violation reading ")
            else:
                # 在非 Windows 系统上，断言异常类型为 RuntimeError
                self.assertIsInstance(p.exception, RuntimeError)
                # 断言异常信息符合指定的正则表达式
                self.assertRegex(
                    str(p.exception),
                    r"DataLoader worker \(pid \d+\) is killed by signal: ",
                )
        finally:
            # 终止进程，释放资源
            p.terminate()

    # 测试在父进程中调用 set_num_threads 后，DataLoader 的子进程是否会因超过 3 个线程而导致段错误
    # 在子进程中调用 set_num_threads(1) 可能会导致处理父进程的 Caffe2 线程池的继承数据结构，最终导致段错误
    # 参考链接：https://github.com/pytorch/pytorch/issues/54752
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    def test_no_segfault(self):
        # 创建一个跟踪错误的进程，目标函数为 _test_no_segfault
        p = ErrorTrackingProcess(target=_test_no_segfault)
        # 启动进程
        p.start()
        # 等待进程结束，设置超时时间
        p.join(JOIN_TIMEOUT)
        try:
            # 断言进程已经结束
            self.assertFalse(p.is_alive())
            # 如果有异常发生，断言异常类型为 RuntimeError
            if p.exception:
                self.assertIsInstance(p.exception, RuntimeError)
                # 断言异常信息符合指定的正则表达式
                self.assertRegex(
                    str(p.exception),
                    r"DataLoader worker \(pid \d+\) is killed by signal: ",
                )
                # 如果异常发生，则测试失败，显示段错误在 fork 后的工作进程中发生
                self.fail("Segfault occurred in worker process after fork")
        finally:
            # 终止进程，释放资源
            p.terminate()
    # 定义一个名为 test_timeout 的测试方法
    def test_timeout(self):
        # 如果 TEST_CUDA 为真且不是 NO_MULTIPROCESSING_SPAWN 模式
        if TEST_CUDA and not NO_MULTIPROCESSING_SPAWN:
            # 在子进程中运行该测试，只能使用 spawn 方式初始化 CUDA。
            # _test_timeout_pin_memory 在 pin_memory=True 的情况下，在迭代器构造时初始化 CUDA。
            targets = (_test_timeout, _test_timeout_pin_memory)
        else:
            # 否则只使用 _test_timeout 作为目标
            targets = (_test_timeout,)
        # 遍历 targets 中的每个目标函数
        for target in targets:
            # 创建一个 ErrorTrackingProcess 进程对象，目标为 target 函数，参数为 self.persistent_workers
            p = ErrorTrackingProcess(target=target, args=(self.persistent_workers,))
            # 启动进程
            p.start()
            # 等待进程终止，超时时间为 JOIN_TIMEOUT 秒
            p.join(JOIN_TIMEOUT)
            try:
                # 断言进程已经终止
                self.assertFalse(p.is_alive())
                # 断言进程的退出码不为 0
                self.assertNotEqual(p.exitcode, 0)
                # 断言进程抛出 RuntimeError 异常
                self.assertIsInstance(p.exception, RuntimeError)
                # 断言异常信息中包含特定字符串 "DataLoader timed out after \d+ seconds"
                self.assertRegex(
                    str(p.exception), r"DataLoader timed out after \d+ seconds"
                )
            finally:
                # 终止进程
                p.terminate()

    # 定义一个名为 test_large_sampler_indices 的测试方法
    def test_large_sampler_indices(self):
        # 测试当进程出现错误时，数据加载器能够干净地退出
        #   1. 持有迭代器的引用
        #   2. 使用产生大元素的采样器，使得 _index_queues 阻塞
        #
        # 更多上下文信息请查看：https://github.com/pytorch/pytorch/issues/48666

        # 创建一个 ErrorTrackingProcess 进程对象，目标为 _test_large_sampler_indices 函数，参数为 self.persistent_workers
        p = ErrorTrackingProcess(
            target=_test_large_sampler_indices, args=(self.persistent_workers,)
        )
        # 启动进程
        p.start()
        # 等待进程终止，超时时间为 JOIN_TIMEOUT 秒
        p.join(JOIN_TIMEOUT)
        try:
            # 断言进程已经终止
            self.assertFalse(p.is_alive())
            # 断言进程的退出码不为 0
            self.assertNotEqual(p.exitcode, 0)
            # 断言进程抛出 RuntimeError 异常
            self.assertIsInstance(p.exception, RuntimeError)
            # 断言异常信息中包含特定字符串 "My Error"
            self.assertRegex(str(p.exception), r"My Error")
        finally:
            # 终止进程
            p.terminate()
    def test_builtin_collection_conversion(self):
        # 遍历集合类型和工作线程数的组合
        for coll_ty in (list, tuple):
            for num_workers in (0, 1):
                # 创建一个 CountingDataset 对象，总数为 20
                dataset = CountingDataset(20)
                # 获取数据加载器，禁用自动批处理
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=None, num_workers=num_workers
                    )
                )
                # 断言已获取的数据等于一个从 0 到 19 的列表
                self.assertEqual(fetched, coll_ty(range(20)))
                # 获取数据加载器，启用自动批处理，每批大小为 2
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=2, num_workers=num_workers
                    )
                )
                # 断言已获取的数据等于一个从 0 到 18 的列表，步长为 2
                self.assertEqual(
                    fetched, coll_ty(torch.tensor([i, i + 1]) for i in range(0, 20, 2))
                )

                # 创建一个 CountingIterableDataset 对象，总数为 20
                dataset = CountingIterableDataset(20)
                # 获取数据加载器，禁用自动批处理
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=None, num_workers=num_workers
                    )
                )
                # 断言已获取的数据等于一个从 0 到 19 的列表
                self.assertEqual(fetched, coll_ty(range(20)))
                # 获取数据加载器，启用自动批处理，每批大小为 2
                # 由于这个 IterableDataset 没有配置每个工作线程，因此为了下面的相等测试有效，不能超过 1 个工作线程
                assert num_workers in [0, 1], "invalid test"
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=2, num_workers=num_workers
                    )
                )
                # 断言已获取的数据等于一个从 0 到 18 的列表，步长为 2
                self.assertEqual(
                    fetched, coll_ty(torch.tensor([i, i + 1]) for i in range(0, 20, 2))
                )

    def test_chain_iterable_style_dataset(self):
        # 连接（串联）两个 IterableDataset
        dataset1 = CountingIterableDataset(20)
        dataset2 = CountingIterableDataset(15)
        expected = list(range(20)) + list(range(15))
        # 遍历工作线程数为 0 或 1 的组合
        for num_workers in [0, 1]:
            # 遍历串联数据集的两种方式：加法和 ChainDataset
            for chained_dataset in [
                dataset1 + dataset2,
                ChainDataset([dataset1, dataset2]),
            ]:
                # 获取数据加载器，传入工作线程数
                fetched = list(
                    self._get_data_loader(chained_dataset, num_workers=num_workers)
                )
                # 断言已获取数据的长度等于预期的长度
                self.assertEqual(len(fetched), len(expected))
                # 逐一比较预期的值和获取的值
                for e, d in zip(expected, fetched):
                    # 断言获取的每个元素是 torch.Tensor 类型
                    self.assertIsInstance(d, torch.Tensor)
                    # 断言获取的每个元素与预期的值相等
                    self.assertEqual(e, d)

        # 使用断言验证不能使用 ChainDataset 与非 IterableDataset 进行迭代的情况
        with self.assertRaisesRegex(
            AssertionError, "ChainDataset only supports IterableDataset"
        ):
            list(iter(dataset1 + self.dataset))

        with self.assertRaisesRegex(
            AssertionError, "ChainDataset only supports IterableDataset"
        ):
            list(iter(ChainDataset([dataset1, self.dataset])))
    @unittest.skipIf(IS_MACOS, "Not working on macos")
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @skipIfRocm  # https://github.com/pytorch/pytorch/issues/90940
    跳过测试：如果在 macOS 平台上运行或者 CUDA IPC 不可用，以及 ROCm 环境存在问题

    def test_multiprocessing_contexts(self):
        reference = [
            torch.arange(3),
            torch.arange(3, 6),
            torch.arange(6, 9),
            torch.arange(9, 11),
        ]
        counting_ds_n = 11
        dl_common_args = dict(num_workers=3, batch_size=3, pin_memory=(not TEST_CUDA))
        for ctx in supported_multiprocessing_contexts:
            # 检查当前上下文是否在支持的多进程上下文列表中
            if (
                ctx in ["spawn", "forkserver"]
                and TEST_CUDA
                and not IS_WINDOWS
                and not IS_JETSON
            ):
                # 如果是在支持 CUDA 的环境中，选择 CUDACountingDataset
                ds_cls = CUDACountingDataset
            else:
                # 否则选择普通的 CountingDataset
                ds_cls = CountingDataset
            self.assertEqual(
                reference,
                list(
                    self._get_data_loader(
                        ds_cls(counting_ds_n),
                        multiprocessing_context=ctx,
                        **dl_common_args,
                    )
                ),
            )
            if ctx is not None:
                # 如果上下文对象不为空，测试上下文对象
                ctx = mp.get_context(ctx)
                self.assertEqual(
                    reference,
                    list(
                        self._get_data_loader(
                            ds_cls(counting_ds_n),
                            multiprocessing_context=ctx,
                            **dl_common_args,
                        )
                    ),
                )
    def _test_multiprocessing_iterdatapipe(self, with_dill):
        # 测试确保来自全局作用域的函数（例如从库中导入的函数）可以序列化并在多进程DataLoader中使用

        reference = [
            torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64),  # 参考数据，包含一个张量，dtype为torch.int64
            torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64),  # 另一个参考数据，同样包含一个张量，dtype为torch.int64
        ]
        datapipe: IterDataPipe = IterableWrapper([[1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])  # 创建一个迭代数据管道对象，使用IterableWrapper包装列表
        datapipe = datapipe.map(row_processor)  # 映射行处理函数到数据管道中的每一行
        datapipe = (
            datapipe.filter(lambda row: len(row) == 4)  # 如果使用dill，则过滤长度为4的行，否则使用filter_len函数过滤
            if with_dill
            else datapipe.filter(filter_len)
        )

        dl_common_args = dict(
            num_workers=2, batch_size=2, shuffle=True, pin_memory=(not TEST_CUDA)
        )
        for ctx in supported_multiprocessing_contexts:  # 遍历支持的多进程上下文列表
            self.assertEqual(
                reference,
                [
                    t.type(torch.int64)  # 张量t的数据类型转换为torch.int64
                    for t in self._get_data_loader(
                        datapipe, multiprocessing_context=ctx, **dl_common_args
                    )
                ],
            )
            if ctx is not None:
                # 测试ctx对象
                ctx = mp.get_context(ctx)  # 获取指定名称的多进程上下文对象
                self.assertEqual(
                    reference,
                    [
                        t.type(torch.int64)  # 张量t的数据类型转换为torch.int64
                        for t in self._get_data_loader(
                            datapipe, multiprocessing_context=ctx, **dl_common_args
                        )
                    ],
                )

    @skipIfNoNumpy  # 如果没有numpy，则跳过测试
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")  # 如果CUDA IPC不可用，则跳过测试
    def test_multiprocessing_iterdatapipe(self):
        self._test_multiprocessing_iterdatapipe(with_dill=False)  # 调用_test_multiprocessing_iterdatapipe函数，禁用dill

    @unittest.expectedFailure  # 预期的测试失败情况
    @skipIfNoNumpy  # 如果没有numpy，则跳过测试
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")  # 如果CUDA IPC不可用，则跳过测试
    @skipIfNoDill  # 如果没有dill，则跳过测试
    def test_multiprocessing_iterdatapipe_with_dill(self):
        self._test_multiprocessing_iterdatapipe(with_dill=True)  # 调用_test_multiprocessing_iterdatapipe函数，启用dill

    def test_worker_seed(self):
        num_workers = 6  # 定义工作进程数量
        batch_size = 1  # 定义批处理大小
        dataset = SynchronizedSeedDataset(num_workers, batch_size, num_workers)  # 创建一个同步种子数据集对象
        dataloader = self._get_data_loader(
            dataset, batch_size=batch_size, num_workers=num_workers  # 使用_get_data_loader函数创建数据加载器
        )
        seeds = set()
        seeds.update(batch[0] for batch in dataloader)  # 将数据加载器中的批次种子更新到集合中
        self.assertEqual(len(seeds), num_workers)  # 断言集合中种子的数量与工作进程数量相等
    def test_worker_seed_reproducibility(self):
        # 定义获取数据加载器的内部函数
        def get_dataloader():
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(42),
            )

        # 设定数据加载器的并行工作者数量
        num_workers = 6
        # 设定每个批次的大小
        batch_size = 1
        # 创建一个同步种子数据集实例
        dataset = SynchronizedSeedDataset(num_workers, batch_size, num_workers)
        # 断言两次获取数据加载器的结果相等，验证种子的复现性
        self.assertEqual(
            {int(batch) for batch in get_dataloader()},
            {int(batch) for batch in get_dataloader()},
        )

    def test_multi_epochs_reproducibility(self):
        # 设定并行工作者数量
        num_workers = 2
        # 设定每个批次的大小
        batch_size = 10
        # 设定总共的训练周期数
        num_epochs = 3

        # 创建多周期数据集
        dataset = TestMultiEpochDataset(batch_size * num_workers)
        # 获取数据加载器实例
        dataloader = self._get_data_loader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 遍历每个训练周期
        for ind in range(num_epochs):
            # 遍历每个批次的数据
            for batch_idx, sample in enumerate(dataloader):
                # 断言当前批次数据的值与预期一致，验证复现性
                self.assertEqual(
                    sample.tolist(), [batch_idx % num_workers] * batch_size
                )

    def test_worker_init_fn(self):
        # 创建种子数据集实例
        dataset = SeedDataset(4)
        # 获取数据加载器实例，使用自定义的工作者初始化函数
        dataloader = self._get_data_loader(
            dataset, batch_size=2, num_workers=2, worker_init_fn=init_fn
        )
        # 遍历数据加载器的每个批次
        for batch in dataloader:
            # 断言每个批次的第一个和第二个元素的值为12345
            self.assertEqual(12345, batch[0])
            self.assertEqual(12345, batch[1])

    def test_get_worker_info(self):
        # 创建一个错误跟踪进程，目标函数为 _test_get_worker_info
        p = ErrorTrackingProcess(target=_test_get_worker_info)
        # 启动进程
        p.start()
        # 等待进程结束，设置超时时间为 JOIN_TIMEOUT
        p.join(JOIN_TIMEOUT)
        try:
            # 断言进程不再存活
            self.assertFalse(p.is_alive())
            # 断言进程的退出码为0
            self.assertEqual(p.exitcode, 0)
        finally:
            # 终止进程
            p.terminate()

    def test_shuffle(self):
        # 测试数据集使用随机洗牌后的结果
        self._test_shuffle(self._get_data_loader(self.dataset, shuffle=True))

    def test_shuffle_batch_none(self):
        # 测试不分批次的数据集使用随机洗牌后的结果
        self._test_shuffle(DataLoader(self.dataset, batch_size=None, shuffle=True))

    def test_shuffle_batch(self):
        # 测试分批次的数据集使用随机洗牌后的结果
        self._test_shuffle(
            self._get_data_loader(self.dataset, batch_size=2, shuffle=True)
        )

    def test_shuffle_reproducibility(self):
        # 针对不同工作者数量测试数据加载器使用相同种子生成器和随机洗牌后的结果
        for fn in (
            lambda: DataLoader(
                self.dataset,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: DataLoader(
                self.dataset,
                shuffle=True,
                num_workers=2,
                generator=torch.Generator().manual_seed(42),
            ),
        ):
            # 断言两次获取数据加载器的结果相同，验证复现性
            self.assertEqual(list(fn()), list(fn()))

    def test_sequential_workers(self):
        # 测试数据加载器使用连续工作者的结果
        self._test_sequential(self._get_data_loader(self.dataset, num_workers=4))

    def test_seqential_batch_workers(self):
        # 测试分批次的数据加载器使用连续工作者的结果
        self._test_sequential(
            self._get_data_loader(self.dataset, batch_size=2, num_workers=4)
        )
    # 测试使用带有预取功能的多线程数据加载器的顺序读取
    def test_seqential_batch_workers_prefetch(self):
        self._test_sequential(
            DataLoader(self.dataset, batch_size=2, num_workers=4, prefetch_factor=3)
        )

    # 测试带有随机顺序加载器的多线程数据加载器
    def test_shuffle_workers(self):
        self._test_shuffle(
            self._get_data_loader(self.dataset, shuffle=True, num_workers=4)
        )

    # 测试带有随机顺序和批处理功能的多线程数据加载器
    def test_shuffle_batch_workers(self):
        self._test_shuffle(
            self._get_data_loader(
                self.dataset, batch_size=2, shuffle=True, num_workers=4
            )
        )

    # 测试带有随机顺序、批处理功能和预取功能的多线程数据加载器
    def test_shuffle_batch_workers_prefetch(self):
        self._test_shuffle(
            DataLoader(
                self.dataset,
                batch_size=2,
                shuffle=True,
                num_workers=4,
                prefetch_factor=3,
            )
        )

    # 测试带有替换和指定额外样本数的随机采样器的长度方法
    def test_random_sampler_len_with_replacement(self):
        from torch.utils.data import RandomSampler

        # 添加额外的5个样本
        num_samples = len(self.dataset) + 5
        sampler = RandomSampler(self.dataset, replacement=True, num_samples=num_samples)
        
        # 测试长度方法
        self.assertEqual(num_samples, len(sampler))

        # 测试迭代
        count_num_samples = sum(1 for _ in sampler)
        self.assertEqual(num_samples, count_num_samples)

        # 测试使用数据加载器和批处理大小为1
        batch_size = 1
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        self.assertEqual(num_samples, count_num_samples_in_data_loader)

        # 测试使用数据加载器和批处理大小为6
        batch_size = 6
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        # 确保数据加载器中样本的数量符合预期
        self.assertEqual(
            int(math.ceil(float(num_samples) / batch_size)),
            count_num_samples_in_data_loader,
        )
    def test_random_sampler_len_without_replacement(self):
        from torch.utils.data import RandomSampler

        # 添加额外的5个样本
        num_samples = len(self.dataset) + 5
        # 创建一个非替换采样器对象，包含指定数量的样本
        sampler = RandomSampler(
            self.dataset, replacement=False, num_samples=num_samples
        )
        # 测试采样器对象的长度方法
        self.assertEqual(num_samples, len(sampler))

        # 测试迭代采样器对象的行为
        count_num_samples = sum(1 for _ in sampler)
        self.assertEqual(num_samples, count_num_samples)

        # 测试使用数据加载器，batch_size = 1
        batch_size = 1
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        self.assertEqual(num_samples, count_num_samples_in_data_loader)

        # 测试使用数据加载器，batch_size = 6
        batch_size = 6
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        self.assertEqual(
            num_samples // batch_size + (num_samples % batch_size > 0),
            count_num_samples_in_data_loader,
        )

    def test_distributed_sampler_invalid_rank(self):
        from torch.utils.data.distributed import DistributedSampler

        dataset = torch.IntTensor(range(10))
        # 测试分布式采样器在指定无效的排名时是否引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            sampler = DistributedSampler(dataset, 3, 3)

        # 测试分布式采样器在指定负数排名时是否引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            sampler = DistributedSampler(dataset, 3, -1)

    def test_duplicating_data_with_drop_last(self):
        from torch.utils.data.distributed import DistributedSampler

        num_processes = 4
        num_batches = 9
        data_set = torch.IntTensor(range(num_batches))
        scanned_data = torch.IntTensor([])
        # 针对每个进程，测试数据集重复加载时是否正确丢弃最后一个批次
        for i in range(num_processes):
            s = DistributedSampler(data_set, num_processes, i)
            d_loader = self._get_data_loader(
                data_set,
                batch_size=int(num_batches / num_processes),
                drop_last=True,
                sampler=s,
            )
            # 遍历数据加载器中的每个数据批次，并将其拼接到扫描数据中
            for data in d_loader:
                scanned_data = torch.cat((scanned_data, data), 0)

        # 验证扫描数据的大小是否与其唯一值的大小相同
        self.assertEqual(scanned_data.size(), scanned_data.unique().size())
    # 测试采样器的可重复性
    def test_sampler_reproducibility(self):
        # 导入所需的数据集工具类
        from torch.utils.data import (
            RandomSampler,
            SubsetRandomSampler,
            WeightedRandomSampler,
        )

        # 定义权重列表
        weights = [0.1, 0.9, 0.4, 0.7, 3.0, 0.6]
        
        # 对每个采样器函数进行测试
        for fn in (
            lambda: RandomSampler(
                self.dataset,
                num_samples=5,
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: RandomSampler(
                self.dataset,
                replacement=False,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: WeightedRandomSampler(
                weights,
                num_samples=5,
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: WeightedRandomSampler(
                weights,
                num_samples=5,
                replacement=False,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: SubsetRandomSampler(
                range(10), generator=torch.Generator().manual_seed(42)
            ),
        ):
            # 断言两次调用采样器函数返回相同结果
            self.assertEqual(list(fn()), list(fn()))

        # 对每个采样器对象进行测试
        for sampler in (
            RandomSampler(self.dataset, num_samples=5, replacement=True),
            RandomSampler(self.dataset, replacement=False),
            WeightedRandomSampler(weights, num_samples=5, replacement=True),
            WeightedRandomSampler(weights, num_samples=5, replacement=False),
            SubsetRandomSampler(range(10)),
        ):
            # 设置随机种子
            torch.manual_seed(0)
            # 获取两次采样结果
            l1 = list(sampler) + list(sampler)

            torch.manual_seed(0)
            l2 = list(sampler) + list(sampler)
            # 断言两次采样结果相同
            self.assertEqual(l1, l2)

            # 创建两个采样器迭代器，并比较它们的输出
            its = (iter(sampler), iter(sampler))
            ls = ([], [])
            for idx in range(len(sampler)):
                for i in range(2):
                    if idx == 0:
                        torch.manual_seed(0)
                    ls[i].append(next(its[i]))
            # 断言两个迭代器的输出相同
            self.assertEqual(ls[0], ls[1])

    # 测试采样功能的辅助函数
    def _test_sampler(self, **kwargs):
        # 定义索引范围作为采样器的输入
        indices = range(2, 12)  # using a regular iterable
        # 获取数据加载器并设定采样器、批处理大小等参数
        dl = self._get_data_loader(
            self.dataset, sampler=indices, batch_size=2, **kwargs
        )
        # 断言数据加载器的长度为5
        self.assertEqual(len(dl), 5)
        # 遍历数据加载器中的每个批次，检查输入的长度和数据是否正确
        for i, (input, _target) in enumerate(dl):
            self.assertEqual(len(input), 2)
            self.assertEqual(input, self.data[i * 2 + 2 : i * 2 + 4])

    # 测试采样功能的主函数
    def test_sampler(self):
        # 调用 _test_sampler 函数进行测试
        self._test_sampler()
        # 使用多线程时再次调用 _test_sampler 函数进行测试
        self._test_sampler(num_workers=4)
        # 如果不是使用 spawn 多进程上下文，则调用 _test_batch_sampler 函数进行测试
        if not NO_MULTIPROCESSING_SPAWN:
            self._test_batch_sampler(num_workers=4, multiprocessing_context="spawn")
    def _test_batch_sampler(self, **kwargs):
        # 定义一个列表存放批次信息，每个批次是由几个连续数字组成的元组
        batches = []  # using a regular iterable
        # 生成批次信息，每个批次包含两个子批次
        for i in range(0, 20, 5):
            batches.append(tuple(range(i, i + 2)))
            batches.append(tuple(range(i + 2, i + 5)))

        # 调用内部方法获取数据加载器，并传入批次信息
        dl = self._get_data_loader(self.dataset, batch_sampler=batches, **kwargs)
        # 断言数据加载器的长度为 8
        self.assertEqual(len(dl), 8)
        # 遍历数据加载器中的数据
        for i, (input, _target) in enumerate(dl):
            if i % 2 == 0:
                offset = i * 5 // 2
                # 断言每个输入的长度为 2
                self.assertEqual(len(input), 2)
                # 断言输入数据与预期数据段相符
                self.assertEqual(input, self.data[offset : offset + 2])
            else:
                offset = i * 5 // 2
                # 断言每个输入的长度为 3
                self.assertEqual(len(input), 3)
                # 断言输入数据与预期数据段相符
                self.assertEqual(input, self.data[offset : offset + 3])

    def test_batch_sampler(self):
        # 测试默认参数情况下的批次采样方法
        self._test_batch_sampler()
        # 测试设置了多线程处理器数量为 4 的情况
        self._test_batch_sampler(num_workers=4)
        # 如果不是在 spawn 多进程上下文中，则测试使用 spawn 的多进程方式
        if not NO_MULTIPROCESSING_SPAWN:
            self._test_batch_sampler(num_workers=4, multiprocessing_context="spawn")

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        # 测试开启数据加载器的 shuffle 和 pin_memory 功能
        loader = self._get_data_loader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
        )
        # 遍历加载器中的输入和目标数据
        for input, target in loader:
            # 断言输入数据被固定在内存中
            self.assertTrue(input.is_pinned())
            # 断言目标数据被固定在内存中
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy(self):
        import numpy as np

        # 定义一个测试数据集类，每个数据项为一个全为 1 的 3 维数组，形状为 (2, 3, 4)
        class TestDataset(torch.utils.data.Dataset):
            def __getitem__(self, i):
                return np.ones((2, 3, 4)) * i

            def __len__(self):
                return 1000

        # 使用测试数据集创建数据加载器，批大小为 12
        loader = self._get_data_loader(TestDataset(), batch_size=12)
        # 获取加载器的一个批次数据
        batch = next(iter(loader))
        # 断言批次数据的类型为双精度张量
        self.assertIsInstance(batch, torch.DoubleTensor)
        # 断言批次数据的形状为 [12, 2, 3, 4]
        self.assertEqual(batch.size(), torch.Size([12, 2, 3, 4]))
    def test_numpy_gen_state(self):
        from torch.utils.data._utils.worker import _generate_state

        # 使用 NumPy 生成的状态作为参考来测试 `_generate_state` 函数是否能产生相同的结果。
        # 测试案例：((worker_id, base_seed), expected_state)
        test_cases = [
            (
                (4, 13434589827475259383),
                (2884386318, 1088094898, 3523808998, 3860348662),
            ),
            ((1, 15014285634777110771), (1934848465, 763213760, 2959016433, 179751970)),
            (
                (10, 978296274032934101),
                (1759791917, 3550927336, 1225977135, 1036538043),
            ),
            (
                (12, 11868770762134256968),
                (3974661794, 3331131333, 3630387033, 2885815368),
            ),
            (
                (9, 15378787925219019706),
                (3815056996, 3162224466, 2735102421, 3190253477),
            ),
            ((5, 9055612723125076328), (3522565701, 3368424109, 959377806, 621878693)),
            (
                (15, 14617792358407278405),
                (3402479508, 1588702753, 1169536393, 3675067356),
            ),
            (
                (9, 17363320784006640087),
                (957989458, 2518334477, 1421725660, 3086155459),
            ),
            (
                (12, 480002904169484764),
                (2732851467, 1762620729, 4055801988, 1277640511),
            ),
            (
                (15, 16803975943592702950),
                (3479415043, 4022359553, 295994005, 3358606349),
            ),
            (
                (9, 11704776406047813044),
                (1968928009, 710113752, 2442656196, 1587420279),
            ),
            (
                (10, 16357891985431864516),
                (1271733898, 4197047399, 3727213786, 2338547348),
            ),
            (
                (2, 17423369006318065007),
                (544294336, 1911284083, 3299147734, 3231058347),
            ),
            ((2, 2889492011444113593), (3721591783, 2595811276, 2212881745, 977682627)),
            ((0, 8979703111668486195), (4276723937, 2556068849, 2962827292, 233130238)),
            (
                (6, 6269787272229682235),
                (2548857855, 1216457374, 1012973562, 2999759647),
            ),
        ]

        for (worker_id, base_seed), exp in test_cases:
            # 断言 `_generate_state` 函数的输出与预期结果相同
            self.assertEqual(exp, _generate_state(base_seed, worker_id))

    def test_error(self):
        # 测试错误处理：使用 `_test_error` 函数来检查由 ErrorDataset(100) 产生的数据加载器
        self._test_error(
            self._get_data_loader(ErrorDataset(100), batch_size=2, shuffle=True)
        )

    def test_error_workers(self):
        # 测试错误处理并发情况：使用 `_test_error` 函数检查由 ErrorDataset(41) 产生的数据加载器，
        # 设置批量大小为 2，开启洗牌功能，并使用 4 个工作进程
        self._test_error(
            self._get_data_loader(
                ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4
            )
        )

    @unittest.skipIf(IS_WINDOWS, "FIXME: stuck test")
    def test_partial_workers(self):
        r"""Check that workers exit even if the iterator is not exhausted."""
        # 如果测试 CUDA，则配置 pin_memory_configs 为 (True, False)，否则为 (False,)
        if TEST_CUDA:
            pin_memory_configs = (True, False)
        else:
            pin_memory_configs = (False,)

        # 遍历 pin_memory_configs 中的配置
        for pin_memory in pin_memory_configs:
            # 调用 _get_data_loader 方法获取 DataLoader 迭代器，设置 batch_size=2, num_workers=4, pin_memory 根据当前循环的 pin_memory 配置
            loader = iter(
                self._get_data_loader(
                    self.dataset, batch_size=2, num_workers=4, pin_memory=pin_memory
                )
            )
            # 获取 DataLoader 的 _workers 属性
            workers = loader._workers
            # 如果 pin_memory=True，则获取 DataLoader 的 _pin_memory_thread 属性
            if pin_memory:
                pin_memory_thread = loader._pin_memory_thread
            # 遍历 DataLoader 迭代器
            for i, _ in enumerate(loader):
                # 当 i 达到 10 时跳出循环
                if i == 10:
                    break
            # 断言 i 的值为 10
            assert i == 10
            # 删除 loader 对象
            del loader
            # 等待并断言每个 worker 线程在 JOIN_TIMEOUT 时间内终止
            for w in workers:
                w.join(JOIN_TIMEOUT)
                self.assertFalse(w.is_alive(), "subprocess not terminated")
            # 如果 pin_memory=True，则等待 pin_memory_thread 线程在 JOIN_TIMEOUT 时间内终止
            if pin_memory:
                pin_memory_thread.join(JOIN_TIMEOUT)
                self.assertFalse(pin_memory_thread.is_alive())

    # 完成测试需要 2.5 分钟，参见 https://github.com/pytorch/pytorch/issues/46065
    @skipIfRocm
    @unittest.skipIf(not HAS_PSUTIL, "psutil not found")
    @slowTest
    def test_len(self):
        # 定义函数 check_len，验证 DataLoader dl 的长度是否符合预期 expected
        def check_len(dl, expected):
            self.assertEqual(len(dl), expected)
            n = 0
            # 遍历 DataLoader dl，计算其元素数量 n
            for _ in dl:
                n += 1
            # 断言遍历后的元素数量 n 与预期 expected 相等
            self.assertEqual(n, expected)

        # 对 self.dataset 执行 check_len，预期长度为 100
        check_len(self.dataset, 100)
        # 对 self._get_data_loader(self.dataset, batch_size=2) 执行 check_len，预期长度为 50
        check_len(self._get_data_loader(self.dataset, batch_size=2), 50)
        # 对 self._get_data_loader(self.dataset, batch_size=3) 执行 check_len，预期长度为 34
        check_len(self._get_data_loader(self.dataset, batch_size=3), 34)

    def test_iterabledataset_len(self):
        # 定义 IterableDataset 类，继承自 torch.utils.data.IterableDataset
        class IterableDataset(torch.utils.data.IterableDataset):
            # 实现 __len__ 方法，返回固定长度 10
            def __len__(self):
                return 10

            # 实现 __iter__ 方法，返回一个迭代器，迭代范围为 0 到 9
            def __iter__(self):
                return iter(range(10))

        # 创建 batch_size=1 的 DataLoader 对象 iterable_loader
        iterable_loader = DataLoader(IterableDataset(), batch_size=1)
        # 断言 iterable_loader 的长度为 10
        self.assertEqual(len(iterable_loader), 10)
        # 创建 batch_size=1, drop_last=True 的 DataLoader 对象 iterable_loader
        iterable_loader = DataLoader(IterableDataset(), batch_size=1, drop_last=True)
        # 断言 iterable_loader 的长度为 10

        iterable_loader = DataLoader(IterableDataset(), batch_size=2)
        # 断言 iterable_loader 的长度为 5
        self.assertEqual(len(iterable_loader), 5)
        # 创建 batch_size=2, drop_last=True 的 DataLoader 对象 iterable_loader
        iterable_loader = DataLoader(IterableDataset(), batch_size=2, drop_last=True)
        # 断言 iterable_loader 的长度为 5

        iterable_loader = DataLoader(IterableDataset(), batch_size=3)
        # 断言 iterable_loader 的长度为 4
        self.assertEqual(len(iterable_loader), 4)
        # 创建 batch_size=3, drop_last=True 的 DataLoader 对象 iterable_loader
        iterable_loader = DataLoader(IterableDataset(), batch_size=3, drop_last=True)
        # 断言 iterable_loader 的长度为 3

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy_scalars(self):
        # 导入 numpy 库
        import numpy as np

        # 定义一个自定义数据集类 ScalarDataset，继承自 torch.utils.data.Dataset
        class ScalarDataset(torch.utils.data.Dataset):
            # 初始化方法，接受一个数据类型参数 dtype
            def __init__(self, dtype):
                self.dtype = dtype

            # 实现数据集的获取方法，返回一个指定类型的数据项
            def __getitem__(self, i):
                return self.dtype()

            # 实现数据集长度的方法，返回固定长度 4
            def __len__(self):
                return 4

        # 定义一个字典，将 numpy 数据类型映射到对应的 torch 张量类型
        dtypes = {
            np.float64: torch.DoubleTensor,
            np.float32: torch.FloatTensor,
            np.float16: torch.HalfTensor,
            np.int64: torch.LongTensor,
            np.int32: torch.IntTensor,
            np.int16: torch.ShortTensor,
            np.int8: torch.CharTensor,
            np.uint8: torch.ByteTensor,
        }

        # 遍历每个数据类型和对应的 torch 张量类型
        for dt, tt in dtypes.items():
            # 创建一个 ScalarDataset 实例，传入当前的数据类型 dt
            dset = ScalarDataset(dt)
            # 使用 self._get_data_loader 方法获取数据加载器，批量大小为 2
            loader = self._get_data_loader(dset, batch_size=2)
            # 获取加载器的下一个批次数据
            batch = next(iter(loader))
            # 断言批次数据的类型为预期的 torch 张量类型 tt
            self.assertIsInstance(batch, tt)

    def test_default_convert_mapping_keep_type(self):
        # 创建一个自定义字典对象，包含键值对 {"a": 1, "b": 2}
        data = CustomDict({"a": 1, "b": 2})
        # 调用 _utils.collate.default_convert 方法对数据进行转换
        converted = _utils.collate.default_convert(data)

        # 断言转换后的结果与原始数据相等
        self.assertEqual(converted, data)

    def test_default_convert_sequence_keep_type(self):
        # 创建一个自定义列表对象，包含元素 [1, 2, 3]
        data = CustomList([1, 2, 3])
        # 调用 _utils.collate.default_convert 方法对数据进行转换
        converted = _utils.collate.default_convert(data)

        # 断言转换后的结果与原始数据相等
        self.assertEqual(converted, data)

    def test_default_convert_sequence_dont_keep_type(self):
        # 创建一个标准 Python 列表对象，包含元素 [0, 1]
        data = range(2)
        # 调用 _utils.collate.default_convert 方法对数据进行转换
        converted = _utils.collate.default_convert(data)

        # 断言转换后的结果与预期的标准列表 [0, 1] 相等
        self.assertEqual(converted, [0, 1])

    def test_default_collate_dtype(self):
        # 创建一个整数列表 arr
        arr = [1, 2, -1]
        # 调用 _utils.collate.default_collate 方法对列表进行整理
        collated = _utils.collate.default_collate(arr)
        # 断言整理后的结果与预期的 torch 张量相等
        self.assertEqual(collated, torch.tensor(arr))
        # 断言整理后的结果数据类型为 torch.int64
        self.assertEqual(collated.dtype, torch.int64)

        # 创建一个浮点数列表 arr
        arr = [1.1, 2.3, -0.9]
        # 调用 _utils.collate.default_collate 方法对列表进行整理
        collated = _utils.collate.default_collate(arr)
        # 断言整理后的结果与预期的 torch 张量相等，数据类型为 torch.float64
        self.assertEqual(collated, torch.tensor(arr, dtype=torch.float64))

        # 创建一个布尔值列表 arr
        arr = [True, False]
        # 调用 _utils.collate.default_collate 方法对列表进行整理
        collated = _utils.collate.default_collate(arr)
        # 断言整理后的结果与预期的 torch 张量相等
        self.assertEqual(collated, torch.tensor(arr))
        # 断言整理后的结果数据类型为 torch.bool

        # 对于不支持的类型，_utils.collate.default_collate 方法应该不做任何操作
        arr = ["a", "b", "c"]
        self.assertEqual(arr, _utils.collate.default_collate(arr))

    def test_default_collate_mapping_keep_type(self):
        # 创建一个自定义字典列表 batch
        batch = [CustomDict({"a": 1, "b": 2}), CustomDict({"a": 3, "b": 4})]
        # 调用 _utils.collate.default_collate 方法对列表进行整理
        collated = _utils.collate.default_collate(batch)

        # 创建预期的自定义字典对象，包含键 "a" 对应的 torch 张量 [1, 3]，键 "b" 对应的 torch 张量 [2, 4]
        expected = CustomDict({"a": torch.tensor([1, 3]), "b": torch.tensor([2, 4])})
        # 断言整理后的结果与预期的自定义字典对象相等
        self.assertEqual(collated, expected)

    def test_default_collate_sequence_keep_type(self):
        # 创建一个自定义列表列表 batch
        batch = [CustomList([1, 2, 3]), CustomList([4, 5, 6])]
        # 调用 _utils.collate.default_collate 方法对列表进行整理
        collated = _utils.collate.default_collate(batch)

        # 创建预期的自定义列表对象，包含 torch 张量 [1, 4]，[2, 5]，[3, 6]
        expected = CustomList(
            [
                torch.tensor([1, 4]),
                torch.tensor([2, 5]),
                torch.tensor([3, 6]),
            ]
        )
        # 断言整理后的结果与预期的自定义列表对象相等
        self.assertEqual(collated, expected)
    def test_default_collate_sequence_dont_keep_type(self):
        # 创建一个包含两个 range 对象的列表作为测试数据
        batch = [range(2), range(2)]
        # 调用 default_collate 函数进行数据整合
        collated = _utils.collate.default_collate(batch)

        # 断言整合后的结果与预期的 Torch 张量列表相等
        self.assertEqual(collated, [torch.tensor([0, 0]), torch.tensor([1, 1])])

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_bad_numpy_types(self):
        import numpy as np

        # 应当不做任何操作
        arr = np.array(["a", "b", "c"])
        self.assertEqual(arr, _utils.collate.default_collate(arr))

        # 期望抛出 TypeError 异常
        arr = np.array([[["a", "b", "c"]]])
        self.assertRaises(TypeError, lambda: _utils.collate.default_collate(arr))

        # 期望抛出 TypeError 异常
        arr = np.array([object(), object(), object()])
        self.assertRaises(TypeError, lambda: _utils.collate.default_collate(arr))

        # 期望抛出 TypeError 异常
        arr = np.array([[[object(), object(), object()]]])
        self.assertRaises(TypeError, lambda: _utils.collate.default_collate(arr))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_numpy_memmap(self):
        import numpy as np

        with tempfile.TemporaryFile() as f:
            # 创建一个 numpy 数组和相应的 memmap
            arr = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
            arr_memmap = np.memmap(f, dtype=arr.dtype, mode="w+", shape=arr.shape)
            arr_memmap[:] = arr[:]
            arr_new = np.memmap(f, dtype=arr.dtype, mode="r", shape=arr.shape)
            # 使用 default_collate 对 memmap 数组进行整合
            tensor = _utils.collate.default_collate(list(arr_new))

        # 断言整合后的结果与预期的 Torch 张量相等
        self.assertTrue(
            (tensor == tensor.new_tensor([[0, 1], [2, 3], [4, 5], [6, 7]])).all().item()
        )

    def test_default_collate_bad_sequence_type(self):
        # 创建一个包含非法元素的列表作为测试数据
        batch = [["X"], ["X", "X"]]
        # 期望抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: _utils.collate.default_collate(batch))
        # 期望抛出 RuntimeError 异常
        self.assertRaises(
            RuntimeError, lambda: _utils.collate.default_collate(batch[::-1])
        )

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_shared_tensor(self):
        import numpy as np

        t_in = torch.zeros(1)
        n_in = np.zeros(1)

        # 断言 Torch 张量不是共享的
        self.assertEqual(t_in.is_shared(), False)

        # 断言整合后的 Torch 张量不是共享的
        self.assertEqual(_utils.collate.default_collate([t_in]).is_shared(), False)
        # 断言整合后的 Torch 张量不是共享的
        self.assertEqual(_utils.collate.default_collate([n_in]).is_shared(), False)

        # FIXME: 修复下面的 hack，使得 default_collate 认为它处于工作进程中
        #        (因为它测试 `get_worker_info() != None`)，尽管实际并非如此。
        old = _utils.worker._worker_info
        try:
            _utils.worker._worker_info = "x"
            # 断言整合后的 Torch 张量是共享的
            self.assertEqual(_utils.collate.default_collate([t_in]).is_shared(), True)
            # 断言整合后的 Torch 张量是共享的
            self.assertEqual(_utils.collate.default_collate([n_in]).is_shared(), True)
        finally:
            _utils.worker._worker_info = old
    # 定义一个测试方法，用于测试过多线程创建警告
    def test_excessive_thread_creation_warning(self):
        # 使用 assertWarnsRegex 断言捕获 UserWarning，并检查警告消息中是否包含特定的文本
        with self.assertWarnsRegex(
            UserWarning,
            r"excessive worker creation might get DataLoader running slow or even freeze",
        ):
            # 创建一个 DataLoader 对象，加载 self.dataset 数据集，指定批量大小为2，同时使用1000个工作线程
            dataloader = DataLoader(self.dataset, batch_size=2, num_workers=1000)
# 测试类 `TestDataLoaderDeviceType`，继承自 `TestCase`
class TestDataLoaderDeviceType(TestCase):
    
    # 使用参数化装饰器，参数为支持的多进程上下文中不为 None 的上下文
    @parametrize(
        "context",
        [ctx for ctx in supported_multiprocessing_contexts if ctx is not None],
    )
    
    # 如果 CUDA IPC 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_nested_tensor_multiprocessing(self, device, context):
        # 对于 CUDA 设备并且上下文为 'fork'，跳过测试
        if "cuda" in device and context == "fork":
            # TODO: 当测试框架允许时，以更好的方式跳过此测试
            return

        # 创建包含 10 个元素的数据集，每个元素为包含一个随机张量的嵌套张量
        dataset = [
            torch.nested.nested_tensor([torch.randn(5)], device=device)
            for _ in range(10)
        ]

        # 根据设备是否为 CPU 且 CUDA 可用性，设置 pin_memory_settings
        pin_memory_settings = [False]
        if device == "cpu" and torch.cuda.is_available():
            pin_memory_settings.append(True)

        # 针对每个 pin_memory 设置，创建 DataLoader 对象并进行测试
        for pin_memory in pin_memory_settings:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                collate_fn=_clone_collate,  # 使用自定义的 collate_fn 函数
                pin_memory=pin_memory,
                multiprocessing_context=context,
            )

            # 遍历 DataLoader 中的每个批次，并断言其与原始数据集中的相应元素相等
            for i, batch in enumerate(loader):
                self.assertEqual(batch[0], dataset[i])

        # 错误情况：默认的 collate_fn 函数当前不支持嵌套张量的批次处理
        # 按照当前语义，需要将它们堆叠，但目前无法实现
        with self.assertRaisesRegex(
            RuntimeError, "not currently supported by the default collate_fn"
        ):
            # 创建新的 DataLoader 对象以测试错误情况
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                multiprocessing_context=context,
            )

            # 获取 DataLoader 的下一个迭代器
            next(iter(loader))


# 集成测试类 `IntegrationTestDataLoaderDataPipe`，用于验证特定的 `DataPipes` 在 `DataLoader` 中的行为
class IntegrationTestDataLoaderDataPipe(TestCase):
    r"""
    Verify the behavior of a certain ``DataPipes`` with ``DataLoader``
    """
    def test_shuffler_iterdatapipe(self):
        r"""
        Verify ``IterDataPipe.shuffle`` is controlled by ``DataLoader``
        to generate different seeds deterministically per epoch.
        """
        # 准备一个预期的列表，包含从0到99的整数
        exp = list(range(100))

        def _create_dp(buffer_size):
            # 创建一个 IterableWrapper 对象，包装预期的列表 exp
            input_ds = dp.iter.IterableWrapper(exp)
            # 对 input_ds 进行乱序操作，并添加分片过滤器
            return input_ds.shuffle(buffer_size=buffer_size).sharding_filter()

        # 针对不同的 buffer_size 进行测试
        for bs in (5, 20, 33):
            # 测试确定性
            # 针对不同的 num_workers 和 persistent_workers 进行组合测试
            for num_workers, pw in itertools.product((0, 1, 2), (True, False)):
                # 当 num_workers 为 0 且 pw 为 True 时跳过当前循环
                if num_workers == 0 and pw:
                    continue

                # 创建一个经过 shuffle_dp 处理的数据管道对象
                shuffle_dp = _create_dp(bs)

                # 根据 num_workers 是否大于 0，选择不同的 multiprocessing 上下文
                mp_ctx = "spawn" if num_workers > 0 else None

                # 创建一个 DataLoader 对象，用于处理 shuffle_dp 数据
                dl = DataLoader(
                    shuffle_dp,
                    num_workers=num_workers,
                    shuffle=True,
                    multiprocessing_context=mp_ctx,
                    persistent_workers=pw,
                )

                # 测试无种子时的结果
                dl_res_ns = list(dl)
                # 断言 dl_res_ns 排序后与预期 exp 相同
                self.assertEqual(sorted(dl_res_ns), exp)

                # 测试相同种子时的结果
                dl_res = []
                for epoch in range(2):
                    torch.manual_seed(123)
                    dl_res.append(list(dl))
                # 断言两个 epoch 的结果相同
                self.assertEqual(dl_res[0], dl_res[1])
                # 断言排序后的第一个 epoch 结果与预期 exp 相同
                self.assertEqual(sorted(dl_res[0]), exp)

                # 测试不同种子时的结果
                torch.manual_seed(321)
                dl_res.append(list(dl))

                # 断言两次结果的长度相同
                self.assertEqual(len(dl_res[0]), len(dl_res[2]))
                # 断言两次结果不相同
                self.assertNotEqual(dl_res[0], dl_res[2])
                # 断言排序后的第一个 epoch 结果与排序后的第三个 epoch 结果相同
                self.assertEqual(sorted(dl_res[0]), sorted(dl_res[2]))

                # 如果 dl._iterator 不为 None，则关闭其工作线程
                if dl._iterator is not None:
                    dl._iterator._shutdown_workers()
                    dl._iterator = None
                # 删除 dl 对象
                del dl
class StringDataset(Dataset):
    def __init__(self):
        # 初始化方法，设置字符串数据
        self.s = "12345"

    def __len__(self):
        # 返回数据集的长度，即字符串长度
        return len(self.s)

    def __getitem__(self, ndx):
        # 返回索引为 ndx 的元素和其索引
        return (self.s[ndx], ndx)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestStringDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        # 设置测试数据集为 StringDataset 类的实例
        self.dataset = StringDataset()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        # 创建 DataLoader 对象，用于测试数据集的批量加载和内存固定
        loader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
        )
        # 遍历 DataLoader 返回的数据批次
        for s, n in loader:
            # 断言第一个元素是字符串类型
            self.assertIsInstance(s[0], str)
            # 断言数据是否已固定在内存中
            self.assertTrue(n.is_pinned())


class DictDataset(Dataset):
    def __len__(self):
        # 返回数据集的长度，这里固定为 4
        return 4

    def __getitem__(self, ndx):
        # 返回包含两个键的字典，其中一个是包含索引数据的张量，另一个是嵌套字典
        return {
            "a_tensor": torch.empty(4, 2).fill_(ndx),
            "another_dict": {
                "a_number": ndx,
            },
        }


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestDictDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        # 设置测试数据集为 DictDataset 类的实例
        self.dataset = DictDataset()

    def test_sequential_batch(self):
        # 测试顺序批处理的数据加载
        for persistent_workers in (False, True):
            if persistent_workers:
                # 创建 DataLoader 对象，设置批大小、是否洗牌、是否持久工作线程等参数
                loader = DataLoader(
                    self.dataset,
                    batch_size=2,
                    shuffle=False,
                    persistent_workers=persistent_workers,
                    num_workers=1,
                )
            else:
                loader = DataLoader(
                    self.dataset,
                    batch_size=2,
                    shuffle=False,
                    persistent_workers=persistent_workers,
                )
            batch_size = loader.batch_size
            # 遍历 DataLoader 返回的数据批次
            for i, sample in enumerate(loader):
                idx = i * batch_size
                # 断言样本中包含指定的键
                self.assertEqual(set(sample.keys()), {"a_tensor", "another_dict"})
                self.assertEqual(set(sample["another_dict"].keys()), {"a_number"})

                t = sample["a_tensor"]
                # 断言张量的形状符合预期
                self.assertEqual(t.size(), torch.Size([batch_size, 4, 2]))
                self.assertTrue((t[0] == idx).all())
                self.assertTrue((t[1] == idx + 1).all())

                n = sample["another_dict"]["a_number"]
                # 断言数值的形状符合预期
                self.assertEqual(n.size(), torch.Size([batch_size]))
                self.assertEqual(n[0], idx)
                self.assertEqual(n[1], idx + 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 测试 DataLoader 类的 pin_memory 功能
    def test_pin_memory(self):
        # 创建 DataLoader 对象，设置批量大小为 2，并开启 pin_memory
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        # 遍历 DataLoader 返回的每个样本
        for sample in loader:
            # 断言样本中的 "a_tensor" 张量已经被 pin 到内存
            self.assertTrue(sample["a_tensor"].is_pinned())
            # 断言样本中的 "another_dict" 字典中的 "a_number" 数值已经被 pin 到内存
            self.assertTrue(sample["another_dict"]["a_number"].is_pinned())

    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 测试 DataLoader 类的 pin_memory 在指定 CUDA 设备上的功能
    def test_pin_memory_device(self):
        # 创建 DataLoader 对象，设置批量大小为 2，开启 pin_memory，并指定 pin_memory_device 为 "cuda"
        loader = DataLoader(
            self.dataset, batch_size=2, pin_memory=True, pin_memory_device="cuda"
        )
        # 遍历 DataLoader 返回的每个样本
        for sample in loader:
            # 断言样本中的 "a_tensor" 张量已经被 pin 到 CUDA 设备的内存上
            self.assertTrue(sample["a_tensor"].is_pinned(device="cuda"))
            # 断言样本中的 "another_dict" 字典中的 "a_number" 数值已经被 pin 到 CUDA 设备的内存上
            self.assertTrue(sample["another_dict"]["a_number"].is_pinned(device="cuda"))

    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 测试 DataLoader 类的仅在指定 CUDA 设备上开启 pin_memory 的功能
    def test_pin_memory_with_only_device(self):
        # 创建 DataLoader 对象，设置批量大小为 2，并仅在指定 CUDA 设备上开启 pin_memory
        loader = DataLoader(self.dataset, batch_size=2, pin_memory_device="cuda")
        # 遍历 DataLoader 返回的每个样本
        for sample in loader:
            # 断言样本中的 "a_tensor" 张量未被 pin 到 CUDA 设备的内存上
            self.assertFalse(sample["a_tensor"].is_pinned(device="cuda"))
            # 断言样本中的 "another_dict" 字典中的 "a_number" 数值未被 pin 到 CUDA 设备的内存上
            self.assertFalse(
                sample["another_dict"]["a_number"].is_pinned(device="cuda")
            )
class DummyDataset(torch.utils.data.Dataset):
    # 定义一个虚拟数据集类，继承自 PyTorch 的 Dataset 类
    def __init__(self):
        # 初始化方法，创建一个包含 0 到 9 的列表作为数据集
        self.data = list(range(10))

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据集中索引为 idx 的数据项
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 断言起始索引为 0，用于验证数据集的起始状态
        assert self.start == 0
        return self.data[idx]


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
@unittest.skipIf(
    TEST_WITH_ASAN,
    "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223",
)
class TestDataLoaderPersistentWorkers(TestDataLoader):
    # 定义一个测试数据加载器的子类，用于测试持久化工作进程的行为
    def setUp(self):
        # 设置测试环境，在每个测试方法执行前调用
        super().setUp()
        self.persistent_workers = True  # 启用持久化工作进程标志

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "No 'resource' module on Windows")
    def test_fd_limit_exceeded(self):
        # 测试文件描述符限制是否超过，见 NOTE [ DataLoader on Linux and open files limit ]
        import subprocess

        # 执行一个子进程来测试文件描述符限制
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import torch
import resource
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

try:
    keep_fds_alive = []
    # 设置文件描述符资源限制
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    # 使用 DataLoader 加载随机数据集，验证是否超出文件描述符限制
    for random_t in DataLoader(RandomDataset(200, (2,2)), multiprocessing_context="fork",
                               num_workers=1, persistent_workers=True):
      random_t.max(dim=0)
      keep_fds_alive.append(random_t)
except RuntimeError as e:
    # 捕获 RuntimeError 异常，验证异常信息中是否包含特定的错误信息
    assert "ulimit -n" in str(e)
    assert "set_sharing_strategy" in str(e)
""",
            ]
        )
    def test_dataset_not_reset(self):
        # 创建一个虚拟数据集对象
        dataset = DummyDataset()
        # 设置是否使用 CUDA 的内存固定选项，初始为 False
        pin_memory_configs = [False]
        # 如果测试环境支持 CUDA，则添加 True 选项
        if TEST_CUDA:
            pin_memory_configs.append(True)
        # 遍历所有内存固定选项
        for pin_memory in pin_memory_configs:
            # 获取数据加载器对象，使用自定义方法 _get_data_loader
            dataloader = self._get_data_loader(
                dataset, num_workers=2, pin_memory=pin_memory
            )
            # 重置数据集的起始值为 0
            dataset.start = 0
            # 执行 10 次迭代
            for i in range(10):
                # 遍历数据加载器中的数据
                for x in dataloader:
                    pass
                # 在每次迭代中改变数据集的起始值，但不影响由工作进程缓存的数据集，
                # 因为它们在 epochs 之间不会重新创建，可以安全地缓存值
                dataset.start = i

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    def test_early_exit(self):
        import subprocess

        # 启动一个子进程来执行 Python 代码，获取其输出结果
        proc = subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
# 导入 PyTorch 库
import torch
# 从 PyTorch 库中导入 DataLoader 和 IterableDataset 类
from torch.utils.data import DataLoader, IterableDataset

# 定义一个继承自 IterableDataset 的随机数据集类
class RandomDataset(IterableDataset):
    # 初始化方法，接受数据集长度 len 和数据维度 size 作为参数
    def __init__(self, len, size):
        # 调用父类 IterableDataset 的初始化方法
        super(RandomDataset).__init__()
        # 设置数据集的长度和数据维度属性
        self.len = len
        self.size = size

    # 定义迭代器方法，返回自身对象
    def __iter__(self):
        return self

    # 定义迭代器的下一个方法
    def __next__(self):
        # 如果数据集长度为 0，则抛出 StopIteration 异常
        if self.len <= 0:
            raise StopIteration
        # 将数据集长度减 1
        self.len -= 1
        # 返回一个随机生成的指定维度的张量
        return torch.randn(self.size)

# 程序入口点，当作为脚本直接运行时执行以下代码
if __name__ == '__main__':
    # 创建一个 DataLoader 对象 dl，加载 RandomDataset 数据集
    dl = DataLoader(
        RandomDataset(64, (28, 28)),  # 使用 RandomDataset 类创建数据集，长度为 64，数据维度为 (28, 28)
        batch_size=16,  # 指定批处理大小为 16
        num_workers=2,  # 使用 2 个工作进程处理数据
        pin_memory=True,  # 将数据加载到 CUDA 固定内存中（如果可用）
        persistent_workers=True,  # 持续使用工作进程
        multiprocessing_context="fork",  # 使用 fork 多进程上下文
    )

    # 遍历 DataLoader 对象 dl 中的第一个批次数据并终止循环
    for _ in dl:
        break
# 从 `__main__` 模块导入的类无法在派生模块中正确反序列化
# 参考：https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
self_module = __import__(os.path.splitext(os.path.basename(__file__))[0])

# 包装器函数，用于调用 `self_module` 中的 `SimpleCustomBatch` 函数
def collate_wrapper(batch):
    return self_module.SimpleCustomBatch(batch)

# 将批量数据转换为填充序列（packed sequence）
def collate_into_packed_sequence(batch):
    # 将批量中每个样本的第一个元素堆叠起来，形成一个新的张量
    data = torch.stack([sample[0] for sample in batch], 1)
    t, b = data.size()
    # 随机生成长度向量，长度为批量中的样本数 b，每个元素取值范围为 [1, t)
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    # 返回一个填充序列对象，其中数据为 data，长度为 lengths
    return torch.nn.utils.rnn.pack_padded_sequence(data, lengths, enforce_sorted=False)

# 将批量数据转换为按批次优先（batch first）的填充序列
def collate_into_packed_sequence_batch_first(batch):
    # 将批量中每个样本的第一个元素堆叠起来，形成一个新的张量
    data = torch.stack([sample[0] for sample in batch], 0)
    b, t = data.size()
    # 随机生成长度向量，长度为批量中的样本数 b，每个元素取值范围为 [1, t)
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    # 返回一个按批次优先的填充序列对象，其中数据为 data，长度为 lengths
    return torch.nn.utils.rnn.pack_padded_sequence(
        data, lengths, batch_first=True, enforce_sorted=False
    )

# 用于跳过测试条件的装饰器，当 TEST_WITH_TSAN 为真时跳过测试
@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestCustomPinFn(TestCase):
    def setUp(self):
        super().setUp()
        # 创建包含输入和目标张量的数据集
        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        self.dataset = TensorDataset(inps, tgts)

    # 用于测试自定义批处理中的内存固定功能
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_custom_batch_pin(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (
                collate_into_packed_sequence_batch_first,
                torch.nn.utils.rnn.PackedSequence,
            ),
        ]
        for collate_fn, elem_cls in test_cases:
            # 创建一个数据加载器，使用自定义的批处理函数 collate_fn，并启用内存固定
            loader = DataLoader(
                self.dataset, batch_size=2, collate_fn=collate_fn, pin_memory=True
            )
            for sample in loader:
                # 断言加载的样本类型为 elem_cls
                self.assertIsInstance(sample, elem_cls)
                # 断言样本已固定在内存中
                self.assertTrue(sample.is_pinned())

    # 用于测试工作进程中自定义批处理的内存固定功能
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_custom_batch_pin_worker(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (
                collate_into_packed_sequence_batch_first,
                torch.nn.utils.rnn.PackedSequence,
            ),
        ]
        for collate_fn, elem_cls in test_cases:
            # 创建一个数据加载器，使用自定义的批处理函数 collate_fn，并启用内存固定和一个工作进程
            loader = DataLoader(
                self.dataset,
                batch_size=2,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=1,
            )
            for sample in loader:
                # 断言加载的样本类型为 elem_cls
                self.assertIsInstance(sample, elem_cls)
                # 断言样本已固定在内存中
                self.assertTrue(sample.is_pinned())
    # 初始化方法，接受一个数据对象并将其保存在实例变量中
    def __init__(self, data):
        self.data = data
        # 初始化工作器 ID 为 None
        self.worker_id = None

    # 设置工作器 ID 的方法，用于设置当前工作器的 ID
    def worker_init_fn(self, worker_id):
        self.worker_id = worker_id

    # 获取元素的方法，根据索引 item 返回工作器 ID 和数据对象中对应位置的数据
    def __getitem__(self, item):
        return self.worker_id, self.data[item]

    # 返回数据对象的长度
    def __len__(self):
        return len(self.data)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
@unittest.skipIf(
    TEST_WITH_ASAN,
    "Flaky with ASAN, see https://github.com/pytorch/pytorch/issues/65727",
)
# 定义一个测试类 TestIndividualWorkerQueue，继承自 TestCase 类
class TestIndividualWorkerQueue(TestCase):

    # 在每个测试方法运行之前调用的设置方法
    def setUp(self):
        super().setUp()
        # 创建一个 TestWorkerQueueDataset 类型的数据集对象 self.dataset，数据集包含从 0 到 127 的整数列表
        self.dataset = TestWorkerQueueDataset(list(range(128)))

    # 定义一个私有方法 _run_ind_worker_queue_test，用于测试独立工作队列
    def _run_ind_worker_queue_test(self, batch_size, num_workers):
        # 创建一个 DataLoader 对象 loader，用于加载数据集 self.dataset
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            timeout=5,
            worker_init_fn=self.dataset.worker_init_fn,
        )
        # 当前工作线程索引初始化为 0
        current_worker_idx = 0
        # 遍历 loader 中的每个 batch
        for i, (worker_ids, sample) in enumerate(loader):
            # 断言 worker_ids 的值转换为列表与当前工作线程索引列表相等
            self.assertEqual(worker_ids.tolist(), [current_worker_idx] * batch_size)
            # 断言 sample 的值转换为列表与当前 batch 的数据范围相等
            self.assertEqual(
                sample.tolist(), list(range(i * batch_size, (i + 1) * batch_size))
            )
            # 更新当前工作线程索引
            current_worker_idx += 1
            # 如果当前工作线程索引达到 num_workers，则重置为 0
            if current_worker_idx == num_workers:
                current_worker_idx = 0

    # 定义一个测试方法 test_ind_worker_queue，测试独立工作队列功能
    def test_ind_worker_queue(self):
        max_num_workers = None
        # 如果操作系统支持获取 CPU 亲和性信息，则尝试获取最大可用工作线程数
        if hasattr(os, "sched_getaffinity"):
            try:
                max_num_workers = len(os.sched_getaffinity(0))
            except Exception:
                pass
        # 如果无法获取最大可用工作线程数，则尝试使用 CPU 核心数的一半
        if max_num_workers is None:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                # 使用 CPU 核心数的一半作为最大可用工作线程数
                max_num_workers = cpu_count // 2

        # 如果依然无法确定最大可用工作线程数，则默认为 1
        if max_num_workers is None:
            max_num_workers = 1

        # 针对不同的 batch_size 和 num_workers 运行 _run_ind_worker_queue_test 方法进行测试
        for batch_size in (8, 16, 32, 64):
            for num_workers in range(0, min(6, max_num_workers)):
                self._run_ind_worker_queue_test(
                    batch_size=batch_size, num_workers=num_workers + 1
                )


# 定义一个数据集类 SetAffinityDataset，实现 IterableDataset 接口
class SetAffinityDataset(IterableDataset):

    # 实现迭代器方法 __iter__，返回一个迭代器，用于生成随机排列的 torch 张量并查询当前亲和性掩码
    def __iter__(self):
        torch.randperm(1)
        after = os.sched_getaffinity(0)
        return iter(after)


@unittest.skipIf(
    not hasattr(os, "sched_setaffinity"), "os.sched_setaffinity is not available"
)
# 定义一个测试类 TestSetAffinity，继承自 TestCase 类
class TestSetAffinity(TestCase):

    # 定义测试方法 test_set_affinity_in_worker_init，测试在工作线程初始化中设置 CPU 亲和性功能
    def test_set_affinity_in_worker_init(self):
        # 查询当前进程的亲和性掩码，以避免设置不允许的掩码
        old_affinity = os.sched_getaffinity(0)
        if not old_affinity:
            self.skipTest("No affinity information")
        # 选择任意一个 CPU 作为期望的亲和性掩码
        expected_affinity = list(old_affinity)[-1]

        # 定义一个 worker_set_affinity 函数，用于设置工作线程的亲和性掩码
        def worker_set_affinity(_):
            os.sched_setaffinity(0, [expected_affinity])

        # 创建一个 SetAffinityDataset 类型的数据集对象 dataset
        dataset = SetAffinityDataset()

        # 创建一个 DataLoader 对象 dataloader，用于加载数据集 dataset，并在工作线程初始化时调用 worker_set_affinity 函数
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=2, worker_init_fn=worker_set_affinity
        )
        # 遍历 dataloader 中的每个样本
        for sample in dataloader:
            # 断言每个样本的值与期望的亲和性掩码相等
            self.assertEqual(sample, [expected_affinity])


# 定义一个数据集类 ConvDataset，继承自 Dataset 类
class ConvDataset(Dataset):
    # 初始化方法，用于创建对象实例时初始化对象的状态
    def __init__(self):
        # 创建一个大小为 (1, 1, 24000) 的全1张量，作为对象的一个属性
        self.x = torch.ones(1, 1, 24000)
        # 调用父进程上的卷积操作（假设是这个类的父类定义了这个方法），这里使用了不正确的语法，可能是作者的错误
        self[0]

    # 返回对象的长度，这里始终返回1，表明对象只包含一个元素
    def __len__(self):
        return 1

    # 根据索引获取对象中的元素
    def __getitem__(self, index):
        # 对对象中的 self.x 应用一维卷积操作，使用全1的卷积核 (1, 1, 2)
        return torch.nn.functional.conv1d(self.x, torch.ones(1, 1, 2))
@unittest.skipIf(IS_WINDOWS, "Needs fork")
@unittest.skipIf(
    TEST_WITH_ASAN,
    "This test hangs when running with ASAN, see https://github.com/pytorch/pytorch/issues/75492",
)
class TestConvAfterFork(TestCase):
    # Tests crash reported in https://github.com/pytorch/pytorch/issues/53565
    def test_conv_after_fork(self):
        # 创建 DataLoader 对象，使用 ConvDataset() 数据集，设置一个工作进程
        loader = DataLoader(ConvDataset(), num_workers=1)
        # 遍历 DataLoader 返回的数据
        for x in loader:
            # 断言每个数据项的形状应为 (1, 1, 1, 23999)
            self.assertEqual(x.shape, (1, 1, 1, 23999))


# 实例化 TestDataLoaderDeviceType 中的设备类型测试，并将其加入到全局变量中
instantiate_device_type_tests(TestDataLoaderDeviceType, globals())


# 如果运行的是主程序，执行测试
if __name__ == "__main__":
    run_tests()
```