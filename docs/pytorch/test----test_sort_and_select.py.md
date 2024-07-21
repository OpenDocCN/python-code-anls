# `.\pytorch\test\test_sort_and_select.py`

```
# Owner(s): ["module: tests"]

import random  # 导入random模块，用于生成随机数
from itertools import permutations, product  # 导入permutations和product函数，用于生成排列和笛卡尔积

import numpy as np  # 导入numpy库，用于数值计算

import torch  # 导入PyTorch深度学习库
from torch import nan  # 导入torch模块中的nan符号

from torch.testing import make_tensor  # 导入make_tensor函数，用于创建测试用张量
from torch.testing._internal.common_device_type import (  # 导入设备类型相关函数和变量
    dtypes,
    dtypesIfCPU,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
)
from torch.testing._internal.common_dtype import (  # 导入数据类型相关函数和变量
    all_types,
    all_types_and,
    floating_types_and,
    integral_types,
)
from torch.testing._internal.common_utils import (  # 导入通用测试工具函数
    run_tests,
    skipIfTorchDynamo,
    slowTest,
    TestCase,
)


class TestSortAndSelect(TestCase):  # 定义测试类TestSortAndSelect，继承自TestCase

    def assertIsOrdered(self, order, x, mxx, ixx, task):
        SIZE = x.size(1)  # 获取张量x的第二维大小
        if order == "descending":  # 如果order为"descending"

            def check_order(a, b):
                # `a != a` because we put NaNs
                # at the end of ascending sorted lists,
                # and the beginning of descending ones.
                return ((a != a) | (a >= b)).all().item()  # 检查是否降序排列

        elif order == "ascending":  # 如果order为"ascending"

            def check_order(a, b):
                # see above
                return ((b != b) | (a <= b)).all().item()  # 检查是否升序排列

        else:
            error(  # 如果order既不是"ascending"也不是"descending"，抛出错误
                f'unknown order "{order}", must be "ascending" or "descending"'
            )

        are_ordered = True  # 初始化排序标志为True
        for k in range(1, SIZE):  # 遍历从1到SIZE-1的索引
            self.assertTrue(
                check_order(mxx[:, k - 1], mxx[:, k]),
                f"torch.sort ({order}) values unordered for {task}",
            )  # 使用check_order函数检查排序顺序

        seen = set()  # 创建一个空集合用于记录已见过的索引
        indicesCorrect = True  # 初始化索引正确标志为True
        size0 = x.size(0)  # 获取张量x的第一维大小
        size = x.size(x.dim() - 1)  # 获取张量x的最后一维大小
        x = x.tolist()  # 将张量x转换为Python列表
        mxx = mxx.tolist()  # 将张量mxx转换为Python列表
        ixx = ixx.tolist()  # 将张量ixx转换为Python列表
        for k in range(size0):  # 遍历第一维索引
            seen.clear()  # 清空已见集合
            for j in range(size):  # 遍历最后一维索引
                self.assertEqual(
                    x[k][ixx[k][j]],
                    mxx[k][j],
                    msg=f"torch.sort ({order}) indices wrong for {task}",
                )  # 检查排序后的索引是否正确
                seen.add(ixx[k][j])  # 将当前索引加入已见集合
            self.assertEqual(len(seen), size)  # 检查已见集合的长度是否与size相等

    def test_sort_stable_none(self):
        # Called sort with stable=None used to trigger an assertion
        # See https://github.com/pytorch/pytorch/issues/117255
        x = torch.ones(10)  # 创建一个包含10个1的张量x
        y = x.sort(stable=None).values  # 对x进行排序，并获取排序后的值
        self.assertTrue(torch.all(y == torch.ones(10)).item())  # 断言排序后的张量y与全1张量相等

    @onlyCUDA  # 限制以下测试仅在CUDA可用时运行
    # 测试在设备上对大切片进行排序
    def test_sort_large_slice(self, device):
        # 测试直接使用 cub 路径
        x = torch.randn(4, 1024000, device=device)  # 创建一个指定设备上的随机张量
        res1val, res1ind = torch.sort(x, stable=True)  # 对张量进行稳定排序，返回值和索引
        torch.cuda.synchronize()  # 同步 CUDA 设备上的操作
        # assertIsOrdered 太慢了，所以直接与 CPU 上的结果比较
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), stable=True)  # 在 CPU 上对张量进行稳定排序
        self.assertEqual(res1val, res1val_cpu.cuda())  # 检查排序后的值是否与 CPU 上的一致
        self.assertEqual(res1ind, res1ind_cpu.cuda())  # 检查排序后的索引是否与 CPU 上的一致
        res1val, res1ind = torch.sort(x, descending=True, stable=True)  # 对张量进行降序稳定排序
        torch.cuda.synchronize()  # 同步 CUDA 设备上的操作
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), descending=True, stable=True)  # 在 CPU 上对张量进行降序稳定排序
        self.assertEqual(res1val, res1val_cpu.cuda())  # 检查降序排序后的值是否与 CPU 上的一致
        self.assertEqual(res1ind, res1ind_cpu.cuda())  # 检查降序排序后的索引是否与 CPU 上的一致

    # FIXME: 一旦为 cub 排序添加支持，移除不支持的类型 torch.bool
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_stable_sort(self, device, dtype):
        sizes = (100, 1000, 10000)
        for ncopies in sizes:
            x = torch.tensor([0, 1] * ncopies, dtype=dtype, device=device)  # 创建指定设备上的张量
            _, idx = x.sort(stable=True)  # 对张量进行稳定排序，获取索引
            self.assertEqual(
                idx[:ncopies],
                torch.arange(start=0, end=2 * ncopies, step=2, device=device),
            )  # 检查排序后的前半部分索引是否正确
            self.assertEqual(
                idx[ncopies:],
                torch.arange(start=1, end=2 * ncopies, step=2, device=device),
            )  # 检查排序后的后半部分索引是否正确

    @onlyCUDA
    @dtypes(torch.uint8)
    @largeTensorTest("200GB")  # 不幸的是 80GB A100 不够大
    def test_sort_large(self, device, dtype):
        t0 = torch.randperm(8192, device=device).to(dtype)  # 在设备上生成一个随机排列张量，并转换为指定数据类型
        t = t0.view(1, 8192).expand(2**18 + 1, -1).contiguous()  # 对张量进行视图变换和扩展
        v, i = t.sort()  # 对张量进行排序，获取值和索引
        del t  # 删除中间张量，释放内存
        iv, im = i.var_mean(dim=0)  # 计算索引张量的方差和均值
        del i  # 删除索引张量，释放内存
        vv, vm = v.var_mean(dim=0)  # 计算值张量的方差和均值
        del v  # 删除值张量，释放内存
        self.assertEqual(vv, torch.zeros_like(vv))  # 检查值张量的方差是否为零
        self.assertEqual(iv, torch.zeros_like(iv))  # 检查索引张量的方差是否为零
        self.assertEqual(vm, torch.arange(255, dtype=dtype, device=device))  # 检查索引张量的均值是否正确
        self.assertEqual(im, t0.sort().indices)  # 检查排序后的索引是否与预期一致

    @dtypes(torch.float32)
    def test_sort_restride(self, device, dtype):
        # 输入：非连续（步长为 5）的 3 元素数组
        tensor = torch.randn((3, 5), dtype=dtype, device=device)[:, 0]  # 创建非连续张量的子视图
        # 输出：零维张量
        # 它们将需要重新调整大小，这意味着它们还将使用输入张量的步长作为基础进行重新调整。
        values = torch.tensor(0, dtype=dtype, device=device)  # 创建值张量
        indices = torch.tensor(0, dtype=torch.long, device=device)  # 创建索引张量
        torch.sort(tensor, out=(values, indices))  # 对张量进行排序，指定输出为值和索引张量
        # 检查：输出是否已被重新调整为密集步长
        self.assertEqual(values.stride(), (1,))  # 检查值张量的步长是否正确
        self.assertEqual(indices.stride(), (1,))  # 检查索引张量的步长是否正确
        # 检查：'tensor' 按 'indices' 索引是否等于 'values'
        self.assertEqual(tensor[indices], values)  # 检查排序后的值张量与原张量按索引取值是否一致
    # 定义一个测试函数，用于测试在不同设备和数据类型下的排序操作
    def _test_sort_discontiguous(self, device, dtype):
        # 定义不同尺寸的张量大小，包括一个大于2048的尺寸，以测试不同的代码路径
        sizes = (5, 7, 2049)
        # 对sizes中的尺寸进行排列组合
        for shape in permutations(sizes):
            # 对每个shape的排列进行排列组合
            for perm in permutations((0, 1, 2)):
                # 遍历每个维度
                for dim in range(3):
                    # 生成一个随机张量，并按照perm排列维度
                    t = torch.randn(shape, device=device, dtype=dtype).permute(perm)
                    # 对张量按照指定维度dim进行排序，并分别存储结果
                    r1 = t.sort(dim=dim)
                    # 对连续存储的张量进行排序，并分别存储结果
                    r2 = t.contiguous().sort(dim=dim)
                    # 断言两种排序结果相等
                    self.assertEqual(r1, r2)
                    # 获取维度dim上的尺寸大小
                    n = t.size(dim)

                    # 断言排序后的顺序是正确的
                    self.assertTrue(
                        (
                            r1.values.narrow(dim, 1, n - 1)
                            >= r1.values.narrow(dim, 0, n - 1)
                        ).all()
                    )

                    # 断言不同段不会混合，这在处理步长不正确时很容易发生
                    self.assertTrue(
                        (t.unsqueeze(-1).transpose(dim, -1) == r1.values.unsqueeze(-1))
                        .any(dim=dim)
                        .any(dim=-1)
                        .all()
                    )

                    # 断言步长被保留
                    if self.device_type == "cuda":
                        # FIXME: 这种行为应该对所有情况都成立，而不仅仅是if条件中指定的情况
                        self.assertEqual(r1.values.stride(), t.stride())
                        self.assertEqual(r1.indices.stride(), t.stride())

    @onlyCUDA
    @dtypes(torch.float32)
    # 用CUDA进行测试排序操作
    def test_sort_discontiguous(self, device, dtype):
        self._test_sort_discontiguous(device, dtype)

    @slowTest  # 这个测试在CPU上运行较慢，但在CUDA上不慢
    @onlyCPU
    @dtypes(torch.float32)
    # 在CPU上测试排序操作，这个测试较慢
    def test_sort_discontiguous_slow(self, device, dtype):
        self._test_sort_discontiguous(device, dtype)

    @dtypes(torch.float32)
    # 测试一维输出的不连续排序操作
    def test_sort_1d_output_discontiguous(self, device, dtype):
        # 生成一个随机张量，并截取前6个元素
        tensor = torch.randn(12, device=device, dtype=dtype)[:6]
        # 创建一个空的张量，与tensor的形状相同，并每隔一个元素复制
        values = torch.empty_like(tensor)[::2]
        # 创建一个长为18的空长整型张量，并每隔两个元素复制
        indices = torch.empty(18, device=device, dtype=torch.long)[::3]
        # 对tensor进行排序，将结果存储在values和indices中
        torch.sort(tensor, out=(values, indices))
        # 对连续存储的tensor进行排序，并分别存储结果
        values_cont, indices_cont = tensor.sort()
        # 断言两种排序结果在indices和values上相等
        self.assertEqual(indices, indices_cont)
        self.assertEqual(values, values_cont)

    @slowTest
    @onlyCPU
    @dtypes(*integral_types())
    # 在CPU上测试一维并行排序操作
    def test_sort_1d_parallel(self, device, dtype):
        # 根据dtype选择合适的范围
        low = 0 if dtype == torch.uint8 else -128
        # 生成一个包含大量数据的随机整型张量
        tensor = torch.randint(
            low=low, high=127, size=(100000,), device=device, dtype=dtype
        )
        # 对tensor进行排序，并确保稳定排序
        vals, _ = torch.sort(tensor, stable=True)
        # 断言排序后的顺序是正确的
        self.assertEqual(True, torch.all(vals[:-1] <= vals[1:]))

    @dtypes(torch.float32)
    # 定义一个测试方法，用于测试一维张量的 topk 操作，输出结果不连续
    def test_topk_1d_output_discontiguous(self, device, dtype):
        # 创建一个在指定设备上、指定类型的随机张量
        tensor = torch.randn(12, device=device, dtype=dtype)
        # 根据 tensor 创建一个同样大小但空的张量，每隔一个元素取一个值
        values = torch.empty_like(tensor)[::2]
        # 根据设备和指定类型创建一个长为 18 的空张量，每隔三个元素取一个值
        indices = torch.empty(18, device=device, dtype=torch.long)[::3]
        # 遍历两种排序方式：True（排序）、False（不排序）
        for sorted in (True, False):
            # 使用 torch.topk 对 tensor 进行 topk 操作，输出结果存储在 values 和 indices 中
            # 使用指定的排序方式 sorted
            torch.topk(tensor, 6, sorted=sorted, out=(values, indices))
            # 使用张量对象的 topk 方法进行相同的操作，获取 values_cont 和 indices_cont
            values_cont, indices_cont = tensor.topk(6, sorted=sorted)
            # 断言 indices 和 indices_cont 是相等的
            self.assertEqual(indices, indices_cont)
            # 断言 values 和 values_cont 是相等的
            self.assertEqual(values, values_cont)

    # FIXME: 当 cub sort 支持 torch.half 和 torch.bfloat16 类型时，移除对 torch.bool 类型的不支持
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    # 定义一个测试函数，用于测试稳定排序是否与 NumPy 一致
    def test_stable_sort_against_numpy(self, device, dtype):
        # 根据数据类型设置特定的无穷大、负无穷大和 NaN 值
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            inf = float("inf")
            neg_inf = -float("inf")
            nan = float("nan")
        else:
            if dtype != torch.bool:
                # 对于非布尔类型，使用 torch.iinfo 获取极值
                inf = torch.iinfo(dtype).max
                neg_inf = torch.iinfo(dtype).min
            else:
                # 对于布尔类型，设定特定的 inf 和 neg_inf 值
                inf = True
                neg_inf = ~inf  # 对于布尔类型取反
            # 对于整数类型，由于没有 NaN 值，将 NaN 值设定为 inf 以简化处理
            nan = inf

        # 定义生成样本数据的函数
        def generate_samples():
            from itertools import chain, combinations

            for sizes in [(1025,), (10000,)]:
                size = sizes[0]
                # 生成二进制字符串样本数据
                yield (torch.tensor([0, 1] * size, dtype=dtype, device=device), 0)

            # 如果设备类型为 "cuda"，直接返回，不生成后续样本数据
            if self.device_type == "cuda":
                return

            # 生成另一类样本数据，包括随机填充的操作
            yield (torch.tensor([0, 1] * 100, dtype=dtype, device=device), 0)

            # 定义一个在指定维度上重复填充索引的函数
            def repeated_index_fill(t, dim, idxs, vals):
                res = t
                for idx, val in zip(idxs, vals):
                    res = res.index_fill(dim, idx, val)
                return res

            # 对不同尺寸的数据生成填充样本
            for sizes in [(1, 10), (10, 1), (10, 10), (10, 10, 10)]:
                size = min(*sizes)
                # 生成正态分布的随机数，并转换为指定数据类型
                x = (torch.randn(*sizes, device=device) * size).to(dtype)
                yield (x, 0)

                # 在每个维度上生成填充操作的样本数据，使用非空子集的值 (inf, neg_inf, nan)
                n_fill_vals = 3  # (inf, neg_inf, nan) 的基数
                for dim in range(len(sizes)):
                    # 随机生成填充的索引
                    idxs = (
                        torch.randint(high=size, size=(size // 10,))
                        for i in range(n_fill_vals)
                    )
                    vals = (inf, neg_inf, nan)
                    # 生成不同长度的填充值子集组合
                    subsets = chain.from_iterable(
                        combinations(list(zip(idxs, vals)), r)
                        for r in range(1, n_fill_vals + 1)
                    )
                    for subset in subsets:
                        idxs_subset, vals_subset = zip(*subset)
                        yield (
                            repeated_index_fill(x, dim, idxs_subset, vals_subset),
                            dim,
                        )

        # 对生成的样本数据进行稳定排序测试
        for sample, dim in generate_samples():
            # 使用 Torch 进行稳定排序，并获取排序后的索引
            _, idx_torch = sample.sort(dim=dim, stable=True)
            # 根据数据类型转换为 NumPy 数组，并在 CPU 上进行排序
            if dtype is torch.bfloat16:
                sample_numpy = sample.float().cpu().numpy()
            else:
                sample_numpy = sample.cpu().numpy()
            idx_numpy = np.argsort(sample_numpy, axis=dim, kind="stable")
            # 断言 Torch 排序的索引与 NumPy 排序的索引一致
            self.assertEqual(idx_torch, idx_numpy)

    # 标记使用的数据类型为半精度和 BF16 类型
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    # 定义测试函数 test_msort，用于测试 torch.msort 函数在不同设备和数据类型下的行为
    def test_msort(self, device, dtype):
        # 定义内部测试函数 test，用于测试给定形状的张量排序结果是否符合预期
        def test(shape):
            # 创建指定形状的张量，数据类型为 dtype，设备为 device，数值范围在 [-9, 9] 之间
            tensor = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            # 如果张量的尺寸不是空的
            if tensor.size() != torch.Size([]):
                # 如果数据类型是 torch.bfloat16
                if dtype is torch.bfloat16:
                    # 使用 numpy.msort 将张量转换为 float 类型后排序，再转换为 bfloat16 类型
                    expected = torch.from_numpy(
                        np.msort(tensor.float().cpu().numpy())
                    ).bfloat16()
                else:
                    # 使用 numpy.msort 对张量的 numpy 数组进行排序
                    expected = torch.from_numpy(np.msort(tensor.cpu().numpy()))
            else:
                # 对于空形状的张量，直接将其作为预期结果，因为 numpy.msort 不支持空形状的张量排序
                expected = tensor

            # 调用 torch.msort 函数对张量进行排序，并断言结果与预期一致
            result = torch.msort(tensor)
            self.assertEqual(result, expected)

            # 创建一个与结果张量相同类型和形状的空张量 out
            out = torch.empty_like(result)
            # 使用 torch.msort 将张量排序结果存储到 out 张量中，并断言 out 与预期结果一致
            torch.msort(tensor, out=out)
            self.assertEqual(out, expected)

        # 定义多个测试用例的形状 shapes
        shapes = (
            [],
            [
                0,
            ],
            [
                20,
            ],
            [1, 20],
            [30, 30],
            [10, 20, 30],
        )
        # 对每个形状进行测试
        for shape in shapes:
            test(shape)

    # 使用装饰器 @skipIfTorchDynamo("Fails on python 3.11") 跳过指定条件下的测试
    @skipIfTorchDynamo("Fails on python 3.11")
    # 使用装饰器 @dtypes(torch.float) 指定测试数据类型为 torch.float
    def test_sort_expanded_tensor(self, device, dtype):
        # 为了解决 https://github.com/pytorch/pytorch/issues/91420 的问题，创建测试用例
        # 创建一个布尔类型的标量张量 data，设备为 device，数据类型为 dtype
        data = torch.scalar_tensor(True, device=device, dtype=dtype)
        # 将标量张量 data 扩展为形状 [1, 1, 1]
        data = data.expand([1, 1, 1])
        # 创建参考结果张量 ref，形状为 [[[True]]]
        ref = torch.Tensor([[[True]]])
        # 使用 torch.sort 函数对 data 进行排序，保持稳定性，按照 dim=1 和 descending=True 排序
        out = torch.sort(data, stable=True, dim=1, descending=True)
        # 使用 torch.sort 函数对 ref 进行相同的排序
        expected = torch.sort(ref, stable=True, dim=1, descending=True)
        # 断言排序后的结果 out 与预期结果 expected 一致
        self.assertEqual(out, expected)

        # 创建一个正态分布的张量 data，形状为 [4, 1, 10]，设备为 device，数据类型为 dtype
        data = torch.randn(4, 1, 10, device=device, dtype=dtype)
        # 将张量 data 扩展为形状 [4, 8, 10]
        data = data.expand([4, 8, 10])
        # 创建一个连续的参考结果张量 ref，与扩展后的 data 形状相同
        ref = data.contiguous()
        # 使用 torch.sort 函数对 data 进行排序，保持稳定性，按照 dim=1 和 descending=True 排序
        out = torch.sort(data, stable=True, dim=1, descending=True)
        # 使用 torch.sort 函数对 ref 进行相同的排序
        expected = torch.sort(ref, stable=True, dim=1, descending=True)
        # 断言排序后的结果 out 与预期结果 expected 一致
        self.assertEqual(out, expected)
    # 定义一个测试函数 test_topk，接受一个设备参数 device
    def test_topk(self, device):
        # 定义一个内部函数 topKViaSort，通过排序实现 topk 操作
        def topKViaSort(t, k, dim, dir):
            # 对张量 t 按维度 dim 进行排序，dir 控制排序方向
            sorted, indices = t.sort(dim, dir)
            # 返回排序后的前 k 个元素值和对应的索引
            return sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k)

        # 定义一个内部函数 compareTensors，用于比较两个张量的值和索引
        def compareTensors(t, res1, ind1, res2, ind2, dim):
            # 断言两个张量的值完全相等
            self.assertEqual(res1, res2, atol=0, rtol=0)

            # 由于索引可能由于实现方式不同而不同，这里验证使用 topk 索引在原始张量上进行 gather 的结果
            if not ind1.eq(ind2).all():
                # 从输入张量 t 中使用 topk 索引进行 gather 操作，比较得到的值和排序索引的值
                vals = t.gather(dim, ind2)
                self.assertEqual(res1, vals, atol=0, rtol=0)

        # 定义一个内部函数 compare，比较 topk 和通过排序实现的结果
        def compare(t, k, dim, dir):
            # 使用 PyTorch 的 topk 方法获取张量 t 的前 k 个元素值和索引
            topKVal, topKInd = t.topk(k, dim, dir, True)
            # 使用自定义的 topKViaSort 方法获取张量 t 的前 k 个元素值和索引
            sortKVal, sortKInd = topKViaSort(t, k, dim, dir)
            # 调用 compareTensors 比较两种方法得到的结果
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)

        # 设置一个大小常量 SIZE
        SIZE = 100
        # 创建一个随机大小的张量 t，使用随机数填充，放置在指定设备上
        t = torch.rand(
            random.randint(1, SIZE),
            random.randint(1, SIZE),
            random.randint(1, SIZE),
            device=device,
        )

        # 循环进行多次测试
        for _kTries in range(3):
            for _dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        # 根据 transpose 参数决定是否对张量 t 进行转置
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())

                            testTensor = t.transpose(dim1, dim2)

                        # 随机选择一个维度 dim 和一个 k 值，进行 compare 操作
                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)

        # 在 CUDA 设备上测试 topk 方法，这里使用了大尺寸的张量 t
        t = torch.randn((2, 100000), device=device)
        compare(t, 2000, 1, True)
        compare(t, 2000, 1, False)

        # 在 CUDA 设备上测试 topk 方法，这里使用了中尺寸的张量 t
        t = torch.randn((2, 10000), device=device)
        compare(t, 2000, 1, True)
        compare(t, 2000, 1, False)

    # 定义一个测试函数 test_topk_quantized_scalar_input
    def test_topk_quantized_scalar_input(self):
        # 创建一个标量输入张量 x，并对其进行量化处理
        x = torch.quantize_per_tensor(torch.randn(()), 0.1, 10, torch.qint8)
        # 对量化的标量输入张量 x 进行 topk 操作，验证是否有问题
        x.topk(1)

    # 定义一个测试函数 test_topk_arguments，接受一个设备参数 device
    def test_topk_arguments(self, device):
        # 创建一个随机张量 q，放置在指定设备上
        q = torch.randn(10, 2, 10, device=device)
        # 测试 topk 方法是否正确处理参数异常情况，期望引发 TypeError 异常
        self.assertRaises(TypeError, lambda: q.topk(4, True))

    # 标记只在 CUDA 设备上运行的测试用例
    @onlyCUDA
    # 在 GPU 上测试不连续的 topk 路径
    def test_topk_noncontiguous_gpu(self, device):
        # 创建一个长度为20的张量，设备为GPU，选择所有偶数位置的元素
        single_block_t = torch.randn(20, device=device)[::2]
        # 创建一个长度为20000的张量，设备为GPU，选择所有偶数位置的元素
        multi_block_t = torch.randn(20000, device=device)[::2]
        # 创建一个长度为200000的张量，设备为GPU，选择所有偶数位置的元素
        sort_t = torch.randn(200000, device=device)[::2]
        # 遍历不同的张量和不同的k值
        for t in (single_block_t, multi_block_t, sort_t):
            for k in (5, 2000, 10000):
                # 如果k大于等于当前张量的长度，跳过此次循环
                if k >= t.shape[0]:
                    continue
                # 对当前张量t进行topk操作，获取前k个最大值及其索引
                top1, idx1 = t.topk(k)
                # 对当前张量t进行连续化后的topk操作，获取前k个最大值及其索引
                top2, idx2 = t.contiguous().topk(k)
                # 断言两种topk操作得到的最大值结果应该一致
                self.assertEqual(top1, top2)
                # 断言两种topk操作得到的索引结果应该一致
                self.assertEqual(idx1, idx2)

    # 测试不同数据类型的topk操作
    def _test_topk_dtype(self, device, dtype, integral, size):
        # 根据integral参数选择生成随机整数或者随机浮点数张量a
        if integral:
            a = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                size=(size,),
                dtype=dtype,
                device=device,
            )
        else:
            a = torch.randn(size=(size,), dtype=dtype, device=device)

        # 对张量a进行排序并取出后半部分的topk结果，然后翻转顺序
        sort_topk = a.sort()[0][-(size // 2) :].flip(0)
        # 对张量a进行topk操作，获取前一半部分的topk结果
        topk = a.topk(size // 2)
        # 断言排序后部分的topk结果应该与直接取前一半部分的topk结果一致
        self.assertEqual(sort_topk, topk[0])  # 检查数值
        # 断言排序后部分的topk结果应该与张量a中对应索引的值一致
        self.assertEqual(sort_topk, a[topk[1]])  # 检查索引

    # 测试整数类型数据的topk操作
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    def test_topk_integral(self, device, dtype):
        small = 10
        large = 4096
        verylarge = 8192  # 在GPU上进行大型topk操作
        # 对不同大小的张量进行测试
        for curr_size in (small, large, verylarge):
            self._test_topk_dtype(device, dtype, True, curr_size)

    # 测试低精度浮点数类型的topk操作
    @dtypes(torch.bfloat16, torch.half)
    def test_topk_lower_precision(self, device, dtype):
        small = 10
        large = 4096
        verylarge = 8192  # 在GPU上进行大型topk操作
        # 对不同大小的张量进行测试
        for curr_size in (small, large, verylarge):
            self._test_topk_dtype(device, dtype, False, curr_size)

    # 测试非有限数值类型的topk操作
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float, torch.double, torch.bfloat16, torch.half)
    def test_topk_nonfinite(self, device, dtype):
        # 创建包含nan、inf、正负10000和0的张量x
        x = torch.tensor(
            [float("nan"), float("inf"), 1e4, 0, -1e4, -float("inf")],
            device=device,
            dtype=dtype,
        )
        # 获取张量x的前4个最大值及其索引
        val, idx = x.topk(4)
        # 创建期望结果张量，包含nan、inf、10000和0
        expect = torch.tensor(
            [float("nan"), float("inf"), 1e4, 0], device=device, dtype=dtype
        )
        # 断言获取的最大值结果与期望结果一致
        self.assertEqual(val, expect)
        # 断言获取的索引结果应该是[0, 1, 2, 3]
        self.assertEqual(idx, [0, 1, 2, 3])

        # 获取张量x的前4个最小值及其索引
        val, idx = x.topk(4, largest=False)
        # 创建期望结果张量，包含-inf、-10000、0和10000
        expect = torch.tensor([-float("inf"), -1e4, 0, 1e4], device=device, dtype=dtype)
        # 断言获取的最小值结果与期望结果一致
        self.assertEqual(val, expect)
        # 断言获取的索引结果应该是[5, 4, 3, 2]
        self.assertEqual(idx, [5, 4, 3, 2])
    # 测试在给定设备上进行 topk 操作，包括两种不同大小的输入
    def test_topk_4d(self, device):
        # 定义两种不同大小的输入
        small = 128
        large = 8192
        # 遍历两种大小
        for size in (small, large):
            # 创建一个尺寸为 (2, size, 2, 2) 的张量，所有元素初始化为 1，指定设备
            x = torch.ones(2, size, 2, 2, device=device)
            # 将第一个维度的第 1 行元素乘以 2
            x[:, 1, :, :] *= 2.0
            # 将第一个维度的第 10 行元素乘以 1.5
            x[:, 10, :, :] *= 1.5
            # 对张量 x 沿第 1 维度进行 topk 操作，保留最大的 2 个元素
            val, ind = torch.topk(x, k=2, dim=1)
            # 创建一个预期的索引张量，大小为 (2, 2, 2, 2)，数据类型为 long，指定设备
            expected_ind = torch.ones(2, 2, 2, 2, dtype=torch.long, device=device)
            # 将第一个维度的第 1 行元素索引设为 10
            expected_ind[:, 1, :, :] = 10
            # 创建一个预期的值张量，大小为 (2, 2, 2, 2)，指定设备
            expected_val = torch.ones(2, 2, 2, 2, device=device)
            # 将第一个维度的第 0 行元素值乘以 2
            expected_val[:, 0, :, :] *= 2.0
            # 将第一个维度的第 1 行元素值乘以 1.5
            expected_val[:, 1, :, :] *= 1.5
            # 断言计算得到的 val 与预期的值张量 expected_val 相等，容差为 0
            self.assertEqual(val, expected_val, atol=0, rtol=0)
            # 断言计算得到的 ind 与预期的索引张量 expected_ind 相等，容差为 0
            self.assertEqual(ind, expected_ind, atol=0, rtol=0)

    # 仅在本机设备类型上执行此测试
    @onlyNativeDeviceTypes
    # 如果使用 CUDA，则指定特定数据类型，包括 torch.bfloat16
    @dtypesIfCUDA(*all_types_and(torch.bfloat16))
    # 否则，对所有数据类型和 torch.bfloat16、torch.half 进行测试
    @dtypes(*all_types_and(torch.bfloat16, torch.half))
    # 测试 topk 操作的特殊情况：k=0
    def test_topk_zero(self, device, dtype):
        # 创建一个尺寸为 (2, 2) 的随机张量，指定设备和数据类型
        t = torch.rand(2, 2, device=device).to(dtype=dtype)
        # 对张量 t 进行 topk 操作，保留最小的 0 个元素（实际上不会保留任何元素）
        val, idx = torch.topk(t, k=0, largest=False)
        # 断言计算得到的 val 的尺寸为 [2, 0]
        self.assertEqual(val.size(), torch.Size([2, 0]))
        # 断言计算得到的 idx 的尺寸为 [2, 0]
        self.assertEqual(idx.size(), torch.Size([2, 0]))

    # 测试在给定设备和数据类型下执行唯一值计算的特殊情况：空标量输入
    def _test_unique_scalar_empty(self, dtype, device, f):
        # 测试标量输入
        x = torch.tensor(0, dtype=dtype, device=device)
        # 调用给定的函数 f 计算唯一值、逆序索引和计数
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        # 预期的唯一值张量，包含单个元素 0
        expected_unique = torch.tensor([0], dtype=dtype, device=device)
        # 预期的逆序索引张量，包含单个元素 0
        expected_inverse = torch.tensor(0, device=device)
        # 预期的计数张量，包含单个元素 1
        expected_counts = torch.tensor([1], device=device)
        # 断言计算得到的 unique 与预期的唯一值张量 expected_unique 相等
        self.assertEqual(unique, expected_unique)
        # 断言计算得到的 inverse 与预期的逆序索引张量 expected_inverse 相等
        self.assertEqual(inverse, expected_inverse)
        # 断言计算得到的 counts 与预期的计数张量 expected_counts 相等
        self.assertEqual(counts, expected_counts)

        # 测试零大小的张量输入
        x = torch.zeros((0, 0, 3), dtype=dtype, device=device)
        # 调用给定的函数 f 计算唯一值、逆序索引和计数
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        # 预期的唯一值张量为空张量
        expected_unique = torch.tensor([], dtype=dtype, device=device)
        # 预期的逆序索引张量，大小为 (0, 0, 3)，数据类型为 long，指定设备
        expected_inverse = torch.empty((0, 0, 3), dtype=torch.long, device=device)
        # 预期的计数张量为空张量
        expected_counts = torch.tensor([], dtype=torch.long, device=device)
        # 断言计算得到的 unique 与预期的唯一值张量 expected_unique 相等
        self.assertEqual(unique, expected_unique)
        # 断言计算得到的 inverse 与预期的逆序索引张量 expected_inverse 相等
        self.assertEqual(inverse, expected_inverse)
        # 断言计算得到的 counts 与预期的计数张量 expected_counts 相等
        self.assertEqual(counts, expected_counts)
        ):
        # 定义一个辅助函数，确保输入是元组形式，如果是 torch.Tensor 类型，则封装为单元素元组
        def ensure_tuple(x):
            if isinstance(x, torch.Tensor):
                return (x,)
            return x

        # 循环两次，测试 return_inverse 和 return_counts 参数的不同组合
        for return_inverse in [True, False]:
            for return_counts in [True, False]:
                # 调用 ensure_tuple 函数确保返回值为元组形式
                ret = ensure_tuple(
                    f(x, return_inverse=return_inverse, return_counts=return_counts)
                )
                # 断言返回值元组的长度符合预期（1 + 是否返回逆序索引 + 是否返回计数）
                self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                # 断言第一个元素与预期的唯一值列表相等
                self.assertEqual(expected_unique, ret[0])
                # 如果返回了逆序索引，则断言逆序索引与预期相等
                if return_inverse:
                    self.assertEqual(expected_inverse, ret[1])
                # 如果返回了计数，则根据索引位置断言计数与预期相等
                if return_counts:
                    count_index = 1 + int(return_inverse)
                    self.assertEqual(expected_counts, ret[count_index])

                # 对于高维张量，测试每个元素的唯一性
                y = x.view(additional_shape)
                # 调用函数 f，强制返回逆序索引和计数，并断言结果与预期相等
                y_unique, y_inverse, y_counts = f(
                    y, return_inverse=True, return_counts=True
                )
                self.assertEqual(expected_unique, y_unique)
                # 根据额外的形状视图断言逆序索引与预期相等
                self.assertEqual(expected_inverse.view(additional_shape), y_inverse)
                # 断言计数与预期相等
                self.assertEqual(expected_counts, y_counts)

    # 指定特定的数据类型进行测试，如果 CPU 设备支持，则测试布尔、半精度浮点数和 BFLOAT16
    @dtypesIfCPU(*all_types_and(torch.bool, torch.float16, torch.bfloat16))
    # 指定数据类型为半精度浮点数和布尔类型
    @dtypes(*all_types_and(torch.half, torch.bool))
    # 如果 CPU 设备支持，则再次测试布尔、半精度浮点数和 BFLOAT16
    @dtypesIfCPU(*all_types_and(torch.bool, torch.float16, torch.bfloat16))
    # 指定数据类型为半精度浮点数和布尔类型
    @dtypes(*all_types_and(torch.half, torch.bool))
    # 测试 unique_consecutive 方法
    def test_unique_consecutive(self, device, dtype):
        # 如果数据类型为布尔类型
        if dtype is torch.bool:
            # 创建布尔类型的张量 x，并指定预期的唯一值、逆序索引和计数
            x = torch.tensor(
                [True, False, False, False, True, True, False, False, False],
                dtype=torch.bool,
                device=device,
            )
            expected_unique = torch.tensor(
                [True, False, True, False], dtype=torch.bool, device=device
            )
            expected_inverse = torch.tensor(
                [0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=torch.long, device=device
            )
            expected_counts = torch.tensor(
                [1, 3, 2, 3], dtype=torch.long, device=device
            )
        else:
            # 创建其他类型的张量 x，并指定预期的唯一值、逆序索引和计数
            x = torch.tensor([1, 2, 2, 2, 5, 5, 2, 2, 3], dtype=dtype, device=device)
            expected_unique = torch.tensor([1, 2, 5, 2, 3], dtype=dtype, device=device)
            expected_inverse = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4], device=device)
            expected_counts = torch.tensor([1, 3, 2, 2, 1], device=device)

        # 循环遍历两种 unique_consecutive 方法的测试
        for f in [
            torch.unique_consecutive,
            lambda x, **kwargs: x.unique_consecutive(**kwargs),
        ]:
            # 调用辅助方法 _test_unique_with_expects 进行测试，并传入相关参数
            self._test_unique_with_expects(
                device,
                dtype,
                f,
                x,
                expected_unique,
                expected_inverse,
                expected_counts,
                (3, 3),
            )
            # 调用 _test_unique_scalar_empty 方法测试标量和空输入情况
            self._test_unique_scalar_empty(dtype, device, f)
    # 将函数标记为仅接受 torch.double 类型的参数
    @dtypes(torch.double)
    # 将函数标记为仅接受 torch.float 类型的参数
    @dtypes(torch.float)
    # 仅在本地设备类型上运行测试，不支持 XLA
    @onlyNativeDeviceTypes  # Fails on XLA
    def test_kthvalue_scalar(self, device, dtype):
        # 测试标量输入（来自 https://github.com/pytorch/pytorch/issues/30818 的测试用例）
        # 测试传递标量张量或具有一个元素的1D张量两种情况是否正常工作
        res = torch.tensor(2, device=device, dtype=dtype).kthvalue(1)
        ref = torch.tensor([2], device=device, dtype=dtype).kthvalue(1)
        self.assertEqual(res[0], ref[0].squeeze())
        self.assertEqual(res[1], ref[1].squeeze())

    # 针对不同数据类型进行测试
    @dtypes(*all_types())
    # 针对 CUDA 设备和 torch.half 类型的所有数据类型进行测试
    @dtypesIfCUDA(*all_types_and(torch.half))
    def test_isin_different_dtypes(self, device):
        supported_types = all_types() if device == "cpu" else all_types_and(torch.half)
        for mult in [1, 10]:
            for assume_unique in [False, True]:
                for dtype1, dtype2 in product(supported_types, supported_types):
                    a = torch.tensor([1, 2, 3], device=device, dtype=dtype1)
                    b = torch.tensor([3, 4, 5] * mult, device=device, dtype=dtype2)
                    ec = torch.tensor([False, False, True], device=device)
                    c = torch.isin(a, b, assume_unique=assume_unique)
                    self.assertEqual(c, ec)

    # 仅在 CUDA 设备上运行测试
    @onlyCUDA
    # 针对所有数据类型进行测试
    @dtypes(*all_types())
    def test_isin_different_devices(self, device, dtype):
        a = torch.arange(6, device=device, dtype=dtype).reshape([2, 3])
        b = torch.arange(3, 30, device="cpu", dtype=dtype)
        with self.assertRaises(RuntimeError):
            torch.isin(a, b)

        c = torch.arange(6, device="cpu", dtype=dtype).reshape([2, 3])
        d = torch.arange(3, 30, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            torch.isin(c, d)

    # 针对所有整数类型进行排序溢出测试
    @dtypes(*integral_types())
    def test_sort_overflow(self, device, dtype):
        "Regression test for https://github.com/pytorch/pytorch/issues/111189"
        prev_num_threads = torch.get_num_threads()
        try:
            low = 0 if dtype == torch.uint8 else -1
            x = torch.full((32768,), low, dtype=dtype, device=device)
            x[:100] = torch.iinfo(x.dtype).max
            torch.set_num_threads(1)
            uv = x.sort().values.unique()
            self.assertEqual(uv.size(0), 2)
        finally:
            torch.set_num_threads(prev_num_threads)
# 在全局范围内实例化设备类型测试，并将其应用于 TestSortAndSelect 类
instantiate_device_type_tests(TestSortAndSelect, globals())

# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```