# `.\pytorch\test\xpu\test_gemm.py`

```
# Owner(s): ["module: intel"]

import itertools  # 导入 itertools 库，用于生成迭代器的函数
import math  # 导入 math 库，用于数学运算
import random  # 导入 random 库，用于生成随机数
from functools import partial  # 从 functools 模块导入 partial 函数，用于创建偏函数
from itertools import product  # 从 itertools 模块导入 product 函数，用于生成笛卡尔积

import numpy as np  # 导入 numpy 库，用于数值计算

import torch  # 导入 PyTorch 深度学习框架
from torch.testing import make_tensor  # 从 torch.testing 模块导入 make_tensor 函数，用于生成测试用的张量
from torch.testing._internal.common_device_type import (
    dtypes,  # 从 common_device_type 模块导入 dtypes，用于测试不同数据类型
    instantiate_device_type_tests,  # 导入 instantiate_device_type_tests 函数，用于实例化设备类型测试
    precisionOverride,  # 导入 precisionOverride 函数，用于覆盖精度设置
)
from torch.testing._internal.common_utils import iter_indices, run_tests, TestCase  # 导入常用的测试工具和测试用例基类


class TestBasicGEMM(TestCase):
    def _test_addmm_addmv(
        self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False, activation=None
    ):
        dtype = t.dtype  # 获取张量 t 的数据类型
        numpy_dtype = dtype  # 初始化 numpy_dtype 为张量的数据类型
        if dtype in {torch.bfloat16, torch.half}:  # 如果数据类型是 bfloat16 或 half
            numpy_dtype = torch.float  # 使用 float 类型的 numpy 数据类型
        if dtype.is_complex:  # 如果数据类型是复数类型
            alpha = 0.9 + 0.3j if alpha is None else alpha  # 设置 alpha 默认值为复数
            beta = 0.5 + 0.6j if beta is None else beta  # 设置 beta 默认值为复数
        else:  # 如果数据类型不是复数
            alpha = 1.2 if alpha is None else alpha  # 设置 alpha 默认值为 1.2
            beta = 0.8 if beta is None else beta  # 设置 beta 默认值为 0.8
        if activation == "gelu":  # 如果激活函数为 gelu
            res1 = f(t, m, v, alpha=alpha, beta=beta, use_gelu=True)  # 调用函数 f 进行计算，使用 gelu 激活函数
        else:
            res1 = f(t, m, v, alpha=alpha, beta=beta)  # 否则调用函数 f 进行计算
        res2 = torch.full_like(res1, math.nan)  # 创建一个与 res1 相同形状的张量，用 NaN 填充
        if transpose_out:  # 如果需要转置输出
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()  # 对结果 res2 进行转置处理
        if activation == "gelu":  # 如果激活函数为 gelu
            f(t, m, v, alpha=alpha, beta=beta, out=res2, use_gelu=True)  # 调用函数 f 进行计算，使用 gelu 激活函数，结果存储到 res2
        else:
            f(t, m, v, alpha=alpha, beta=beta, out=res2)  # 否则调用函数 f 进行计算，结果存储到 res2
        m.to(numpy_dtype).cpu().numpy()  # 将张量 m 转换为 numpy 数据类型并移动到 CPU
        v.to(numpy_dtype).cpu().numpy()  # 将张量 v 转换为 numpy 数据类型并移动到 CPU
        res3 = alpha * (
            m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy()  # 计算 m 和 v 的乘积，并乘以 alpha
        )
        if beta != 0:  # 如果 beta 不为零
            res3 += (beta * t).to(numpy_dtype).cpu().numpy()  # 将 beta 乘以 t 并加到 res3 上
        if activation == "relu":  # 如果激活函数为 relu
            res3 = res3 * (res3 > 0)  # 对 res3 应用 relu 激活函数
        elif activation == "gelu":  # 如果激活函数为 gelu
            res3_t = torch.from_numpy(res3).to(dtype)  # 将 numpy 数组转换为 PyTorch 张量
            approximate = "tanh" if t.is_cuda else "none"  # 根据张量 t 是否在 GPU 上选择逼近函数
            res3_t = torch.nn.functional.gelu(res3_t, approximate=approximate)  # 应用 gelu 激活函数
            res3 = res3_t.to(numpy_dtype).cpu().numpy()  # 将结果转换为 numpy 数据类型并移动到 CPU
        else:
            assert activation is None, f"unsupported activation {activation}"  # 如果激活函数不支持，则抛出异常
        res3 = torch.from_numpy(res3).to(dtype)  # 将 numpy 数组转换为 PyTorch 张量，并指定数据类型
        self.assertEqual(res1, res2)  # 断言 res1 等于 res2
        self.assertEqual(res1, res3)  # 断言 res1 等于 res3
    # 定义一个私有方法用于测试 torch.addmm 函数的实现，接受函数、激活函数、设备和数据类型作为参数
    def _test_addmm_impl(self, func, activation, device, dtype):
        # 创建一个大小为 10x25 的随机张量 M，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备和数据类型
        M = torch.randn(10, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        # 创建一个大小为 10x50 的随机张量 m1，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备和数据类型
        m1 = torch.randn(10, 50, device="cpu", dtype=torch.float32).to(dtype).to(device)
        # 创建一个大小为 50x25 的随机张量 m2，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备和数据类型
        m2 = torch.randn(50, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        # 调用内部方法 _test_addmm_addmv 测试 torch.addmm 函数
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # 创建一个大小为 25 的向量 V，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备和数据类型
        V = torch.randn(25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        # 调用内部方法 _test_addmm_addmv 测试 torch.addmm 函数，传入向量 V，并设置 beta=1，触发 CUDA 中的 epilogue 融合
        self._test_addmm_addmv(func, V, m1, m2, beta=1, activation=activation)

        # 测试 0-步幅情况下的张量
        # 创建一个大小为 10x1 的随机张量 M，初始设备为 CPU，数据类型为 torch.float32，扩展为大小为 10x25，然后转移到指定设备
        M = (
            torch.randn(10, 1, device="cpu", dtype=torch.float32)
            .to(dtype)
            .expand(10, 25)
            .to(device)
        )
        # 创建一个大小为 10x1 的随机张量 m1，初始设备为 CPU，数据类型为 torch.float32，扩展为大小为 10x50，然后转移到指定设备
        m1 = (
            torch.randn(10, 1, device="cpu", dtype=torch.float32)
            .to(dtype)
            .expand(10, 50)
            .to(device)
        )
        # 创建一个大小为 50x25 的随机张量 m2，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备和数据类型
        m2 = torch.randn(50, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        # 调用内部方法 _test_addmm_addmv 测试 torch.addmm 函数
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # 测试 beta=0，M 包含 nan 的情况
        # 创建一个大小为 10x25，元素全为 nan 的张量 M，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备
        M = (
            torch.full((10, 25), math.nan, device="cpu", dtype=torch.float32)
            .to(dtype)
            .to(device)
        )
        # 创建一个大小为 10x50 的随机张量 m1，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备和数据类型
        m1 = torch.randn(10, 50, device="cpu", dtype=torch.float32).to(dtype).to(device)
        # 创建一个大小为 50x25 的随机张量 m2，初始设备为 CPU，数据类型为 torch.float32，然后转移到指定设备和数据类型
        m2 = torch.randn(50, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        # 调用内部方法 _test_addmm_addmv 测试 torch.addmm 函数，设置 beta=0
        self._test_addmm_addmv(func, M, m1, m2, beta=0, activation=activation)

        # 测试转置情况
        # 使用 itertools.product 遍历所有 t1, t2, t3, t4 的组合
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):

            # 定义一个函数 maybe_transpose，根据条件 cond 对张量 m 进行可能的转置操作
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            # 根据 t1 条件，可能对大小为 10x25 的随机张量 M 进行转置，并转移到指定设备和数据类型
            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            # 根据 t2 条件，可能对大小为 10x50 的随机张量 m1 进行转置，并转移到指定设备和数据类型
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            # 根据 t3 条件，可能对大小为 50x25 的随机张量 m2 进行转置，并转移到指定设备和数据类型
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            # 调用内部方法 _test_addmm_addmv 测试 torch.addmm 函数，传入转置标志 t4
            self._test_addmm_addmv(
                func, M, m1, m2, transpose_out=t4, activation=activation
            )

            # 如果 t1 条件为真，则使用向量 V 而非矩阵 M 进行 CUDA 中的 epilogue 融合测试（不依赖于 t1）
            if t1:
                self._test_addmm_addmv(
                    func,
                    V,
                    m1,
                    m2,
                    beta=1,
                    transpose_out=t4,
                    activation=activation,
                )

    # 装饰器 precisionOverride 用于设置 torch.float 和 torch.half 数据类型的精度修正值
    @precisionOverride(
        {
            torch.float: 1e-4,
            torch.half: 1e-1,
        }
    )
    # 装饰器 dtypes 用于指定测试函数 test_addmm 的数据类型参数范围
    @dtypes(torch.float32, torch.half)
    # 定义测试函数 test_addmm，接受设备和数据类型作为参数
    def test_addmm(self, device, dtype):
        # 调用 _test_addmm_impl 方法，传入 torch.addmm 函数、None 激活函数、设备和数据类型
        self._test_addmm_impl(torch.addmm, None, device, dtype)

    # 装饰器 precisionOverride 用于设置 torch.bfloat16、torch.half 和 torch.float 数据类型的精度修正值
    @precisionOverride({torch.bfloat16: 1e-0, torch.half: 1e-3, torch.float: 1e-4})
    # 使用装饰器设置该函数的输入数据类型为 torch.bfloat16, torch.half, torch.float
    @dtypes(torch.bfloat16, torch.half, torch.float)
    # 定义测试函数 test_addmv，接受设备和数据类型作为参数
    def test_addmv(self, device, dtype):
        # 必须使用 torch.randn(...).to(bfloat16) 替代 torch.randn(..., dtype=bfloat16)，
        # 因为 randn 函数目前不支持 bfloat16 类型。
        # "*0.2" 是为了降低低精度下的误差
        ts = [
            # 生成一个形状为 (50,) 的张量，并转换到指定的数据类型 dtype
            0.2 * torch.randn(50, device=device).to(dtype),
            # 生成一个形状为 (1,) 的张量，并扩展到形状 (50,)
            0.2 * torch.randn(1, device=device).to(dtype).expand(50),
        ]
        vs = [
            # 生成一个形状为 (100,) 的张量，并转换到指定的数据类型 dtype
            0.2 * torch.randn(100, device=device).to(dtype),
            # 生成一个形状为 (1,) 的张量，并扩展到形状 (100,)
            0.2
            * torch.ones(1, device=device)
            .to(dtype)
            .expand(100),  # 降低低精度下的误差
        ]
        ms = [
            # 0维张量，形状为 ()，并扩展到形状 (50, 100)，降低低精度下的误差
            0.2
            * torch.ones((), device=device)
            .to(dtype)
            .expand(50, 100),
            # 1维张量，形状为 (1, 100)，并扩展到形状 (50, 100)，初始化以降低低精度下的误差
            0.2 * torch.randn((1, 100), device=device).to(dtype).expand(50, 100),
            # 通过确保中间和结果值在低精度类型中可以精确表示，初始化减少广播矩阵低精度下的误差
            0.2
            * torch.randint(3, (50, 1), dtype=torch.float, device=device)
            .to(dtype)
            .expand(50, 100),
            # 2维张量，形状为 (50, 100)，并转换到指定的数据类型 dtype
            0.2 * torch.randn((50, 100), device=device).to(dtype),
            # 形状为 (100, 50) 的张量的转置，并转换到指定的数据类型 dtype
            0.2 * torch.randn((100, 50), device=device).to(dtype).t(),
        ]
        # 使用 itertools.product 对 ms, vs, ts 中的张量进行排列组合，传入测试函数 _test_addmm_addmv 进行测试
        for m, v, t in itertools.product(ms, vs, ts):
            self._test_addmm_addmv(torch.addmv, t, m, v)
        # 测试 beta=0，t 全为 NaN 的情况
        t = torch.full((50,), math.nan, device=device).to(dtype)
        for m, v in itertools.product(ms, vs):
            self._test_addmm_addmv(torch.addmv, t, m, v, beta=0)

    # 使用装饰器设置该函数的输入数据类型为 torch.half 和 torch.float32
    @dtypes(
        torch.half,
        torch.float32,
    )
    # 设置 torch.half 和 torch.bfloat16 的精度覆盖率为 0.05
    @precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
    # 使用装饰器设置该函数的输入数据类型为 torch.float32, torch.bfloat16, torch.half
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    # 定义一个测试方法，用于测试张量操作函数的多个变体
    def _test_addbmm_baddbmm(self, func, b1, b2, ref, out_tensor):
        # 调用指定张量对象的指定函数，并传入参数 b1 和 b2
        getattr(out_tensor, func + "_")(b1, b2)
        # 断言操作后的张量与预期结果 ref 相等
        self.assertEqual(out_tensor, ref)
        # 克隆当前的 out_tensor 到 res3
        res3 = out_tensor.clone()

        # 在使用过程中，捕获并断言会发出 UserWarning，指明该函数的某些用法已过时
        with self.assertWarnsOnceRegex(
            UserWarning, f"This overload of {func}_ is deprecated"
        ):
            # 使用过时的函数重载形式进行操作
            getattr(out_tensor, func + "_")(1, b1, b2)
        # 断言操作后的张量与预期结果 ref*2 相等
        self.assertEqual(out_tensor, ref * 2),

        # 使用 beta=1 参数调用指定张量对象的指定函数
        getattr(res3, func + "_")(b1, b2, beta=1)
        # 断言操作后的张量与 res3 相等
        self.assertEqual(out_tensor, res3)

        # 再次捕获并断言会发出 UserWarning，指明该函数的另一种重载形式已过时
        with self.assertWarnsOnceRegex(
            UserWarning, f"This overload of {func}_ is deprecated"
        ):
            # 使用过时的函数重载形式进行操作，带有浮点数参数
            getattr(out_tensor, func + "_")(1.0, 0.5, b1, b2)
        # 断言操作后的张量与预期结果 ref*2.5 相等
        self.assertEqual(out_tensor, ref * 2.5)
        
        # 使用 beta=1.0 和 alpha=0.5 参数调用指定张量对象的指定函数
        getattr(res3, func + "_")(b1, b2, beta=1.0, alpha=0.5)
        # 断言操作后的张量与 res3 相等
        self.assertEqual(out_tensor, res3)

        # 捕获并断言会发出 UserWarning，指明该函数的另一种重载形式已过时
        with self.assertWarnsOnceRegex(
            UserWarning, f"This overload of {func} is deprecated"
        ):
            # 使用 torch 模块中指定的函数进行操作，其中包含多个张量参数
            self.assertEqual(out_tensor, getattr(torch, func)(1, out_tensor, 0, b1, b2))

        # 使用指定的函数进行操作，传入 out_tensor、b1 和 b2，并指定 beta=1 和 alpha=0.5
        res4 = getattr(torch, func)(out_tensor, b1, b2, beta=1, alpha=0.5)
        # 断言操作后的结果与预期结果 ref*3 相等
        self.assertEqual(res4, ref * 3),

        # 创建一个与 out_tensor 形状和类型相同的张量，其中所有元素均为 NaN
        nan = torch.full_like(out_tensor, math.nan)
        # 使用指定的函数进行操作，传入 nan、b1 和 b2，并指定 beta=0 和 alpha=1
        res5 = getattr(torch, func)(nan, b1, b2, beta=0, alpha=1)
        # 断言操作后的结果与预期结果 ref 相等
        self.assertEqual(res5, ref)

        # 如果 b1 是复数张量
        if b1.is_complex():
            # 使用指定的函数进行操作，传入 out_tensor、b1 和 b2，并指定复数参数 beta=0.1j 和 alpha=0.5j
            res6 = getattr(torch, func)(out_tensor, b1, b2, beta=0.1j, alpha=0.5j)
            # 断言操作后的结果与预期结果相等，使用复数计算
            self.assertEqual(res6, out_tensor * 0.1j + 0.5j * ref)
        else:
            # 使用指定的函数进行操作，传入 out_tensor、b1 和 b2，并指定浮点数参数 beta=0.1 和 alpha=0.5
            res6 = getattr(torch, func)(out_tensor, b1, b2, beta=0.1, alpha=0.5)
            # 断言操作后的结果与预期结果相等，使用浮点数计算
            self.assertEqual(res6, out_tensor * 0.1 + 0.5 * ref)

        # 创建一个与 out_tensor 形状和类型相同的张量，其中所有元素均为 NaN
        res7 = torch.full_like(out_tensor, math.nan)
        # 使用指定的函数进行操作，传入 nan、b1 和 b2，并指定 beta=0，将结果存入 res7
        getattr(torch, func)(nan, b1, b2, beta=0, out=res7)
        # 断言操作后的结果与预期结果 ref 相等
        self.assertEqual(res7, ref)

    # 设置函数的精度覆盖装饰器，对指定的数据类型进行精度设置
    @precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
    # 设置数据类型装饰器，指定函数的数据类型为 float32、bfloat16 和 half
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    # 设置函数的精度覆盖装饰器，对指定的数据类型进行精度设置
    @precisionOverride({torch.half: 0.1, torch.bfloat16: 0.5})
    # 设置数据类型装饰器，指定函数的数据类型为 float32、bfloat16 和 half
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    # 定义一个测试方法，用于测试 torch.tensordot 函数的功能
    def test_tensordot(self, device):
        # 创建张量 a，包含数字 0 到 59，形状为 (3, 4, 5)，放在指定的设备上
        a = torch.arange(60.0, device=device).reshape(3, 4, 5)
        # 创建张量 b，包含数字 0 到 23，形状为 (4, 3, 2)，放在指定的设备上
        b = torch.arange(24.0, device=device).reshape(4, 3, 2)
        # 使用 torch.tensordot 计算张量 a 和 b 的张量点积，指定维度 ([1, 0], [0, 1])，并将结果移到 CPU 上
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        # 使用 numpy 的 tensordot 计算张量 a 和 b 的张量点积，指定维度 ([1, 0], [0, 1])
        cn = torch.from_numpy(
            np.tensordot(a.cpu().numpy(), b.cpu().numpy(), axes=([1, 0], [0, 1]))
        )
        # 断言两个张量 c 和 cn 是否相等
        self.assertEqual(c, cn)

        # 创建全零张量 cout，形状为 (5, 2)，放在指定的设备上
        cout = torch.zeros((5, 2), device=device)
        # 使用 torch.tensordot 计算张量 a 和 b 的张量点积，指定维度 ([1, 0], [0, 1])，将结果存入 cout，并移到 CPU 上
        torch.tensordot(a, b, dims=([1, 0], [0, 1]), out=cout).cpu()
        # 断言张量 c 和 cout 是否相等
        self.assertEqual(c, cout)

        # 创建随机张量 a，形状为 (2, 3, 4, 5)，放在指定的设备上
        a = torch.randn(2, 3, 4, 5, device=device)
        # 创建随机张量 b，形状为 (4, 5, 6, 7)，放在指定的设备上
        b = torch.randn(4, 5, 6, 7, device=device)
        # 使用 torch.tensordot 计算张量 a 和 b 的张量点积，指定维度 2，并移到 CPU 上
        c = torch.tensordot(a, b, dims=2).cpu()
        # 使用 numpy 的 tensordot 计算张量 a 和 b 的张量点积，指定维度 2
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(), axes=2))

        # 使用断言捕获 RuntimeError 异常，检查是否包含指定的错误信息 "expects dims >= 0"
        with self.assertRaisesRegex(RuntimeError, "expects dims >= 0"):
            torch.tensordot(a, b, dims=-1)

        # 断言张量 c 和 cn 是否相等
        self.assertEqual(c, cn)

        # 计算张量 a 和 b 的张量点积，没有指定维度，将结果移到 CPU 上
        c = torch.tensordot(a, b).cpu()
        # 使用 numpy 的 tensordot 计算张量 a 和 b 的张量点积，没有指定维度
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy()))
        # 断言张量 c 和 cn 是否相等
        self.assertEqual(c, cn)

        # 创建标量张量 a 和 b，值为 0.0，使用 torch.tensordot 计算它们的张量点积
        a = torch.tensordot(torch.tensor(0.0), torch.tensor(0.0), 0)
        # 使用 numpy 的 tensordot 计算标量张量 a 和 b 的张量点积
        an = torch.from_numpy(
            np.tensordot(
                np.zeros((), dtype=np.float32), np.zeros((), dtype=np.float32), 0
            )
        )
        # 断言标量张量 a 和 an 是否相等
        self.assertEqual(a, an)

    # 装饰器函数，指定测试方法的张量类型为 torch.float
    @dtypes(torch.float)
    # 装饰器函数，覆盖精度为 torch.float32 的测试方法，指定精度为 1e-4
    @precisionOverride({torch.float32: 1e-4})
    # 定义测试方法，用于测试特定大小和步长的张量乘法
    def test_1_sized_with_0_strided(self, device, dtype):
        # 创建大小为 (8, 1, 64) 的 dtype 类型张量 a，放在指定的设备上
        a = make_tensor((8, 1, 64), dtype=dtype, device=device)
        # 使用 torch.as_strided 创建步长为 [64, 0, 1] 的张量 a_strided
        a_strided = torch.as_strided(a, size=[8, 1, 64], stride=[64, 0, 1])
        # 创建大小为 (8, 64, 512) 的 dtype 类型张量 b，放在指定的设备上
        b = make_tensor((8, 64, 512), dtype=dtype, device=device)
        # 使用 torch.as_strided 创建步长为 [64, 1, 512] 的张量 b_strided
        b_strided = torch.as_strided(b, size=[8, 64, 512], stride=[64, 1, 512])
        # 使用 torch.bmm 计算张量 a_strided 和 b_strided 的批次矩阵乘积
        res = torch.bmm(a_strided, b_strided)
        # 使用 numpy 的批次矩阵乘积计算结果，转换为指定设备和类型的张量 expect
        expect = torch.from_numpy(a_strided.cpu().numpy() @ b_strided.cpu().numpy()).to(
            device=device, dtype=dtype
        )
        # 断言张量 res 和 expect 是否相等
        self.assertEqual(expect, res)
    # 选择可以广播的维度
    def _select_broadcastable_dims(self, dims_full=None):
        # 如果未提供完整维度列表，则初始化为空列表
        if dims_full is None:
            # 随机生成维度数量，范围为1到4
            ndims = random.randint(1, 4)
            # 随机生成每个维度的大小，范围为1到8
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            # 如果提供了完整维度列表，则获取维度数量
            ndims = len(dims_full)

        # 选择用于操作的实际维度：
        # 较小的情况：可能减少维度数量，以及每个维度的大小可能减小
        smaller_ndims = random.randint(1, ndims)
        dims_small = []  # 存储较小操作的维度列表
        dims_large = []  # 存储较大操作的维度列表
        # 逆序遍历完整维度列表
        for i in range(ndims - 1, -1, -1):
            # 随机选择1到3的整数
            j = random.randint(1, 3)
            if j == 1:  # 没有减少的单例维度
                ds = dims_full[i]  # 较小操作的当前维度大小与完整维度相同
                dl = dims_full[i]  # 较大操作的当前维度大小与完整维度相同
            elif j == 2:  # 较大操作可能有减少的单例维度
                ds = dims_full[i]  # 较小操作的当前维度大小与完整维度相同
                # 如果较小操作的维度数量小于较小维度数量，则将较大操作的当前维度大小设为1，否则与完整维度相同
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # 较小操作可能有减少的单例维度
                ds = 1  # 较小操作的当前维度大小设为1
                dl = dims_full[i]  # 较大操作的当前维度大小与完整维度相同
            dims_large = [dl] + dims_large  # 将当前较大操作的维度大小添加到较大操作的维度列表中
            # 如果较小操作的维度数量小于较小维度数量，则将当前较小操作的维度大小添加到较小操作的维度列表中
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        # 返回三个维度列表，分别表示较小操作的维度，较大操作的维度以及完整的维度列表
        return (dims_small, dims_large, dims_full)
    # 定义测试广播融合矩阵乘法的方法，接受设备参数
    def test_broadcast_fused_matmul(self, device):
        # 定义测试函数列表，包含各种矩阵乘法的操作名
        fns = ["baddbmm", "addbmm", "addmm", "addmv", "addr"]

        # 遍历测试函数列表中的每个函数名
        for fn in fns:
            # 随机生成各个维度的大小
            batch_dim = random.randint(1, 8)
            n_dim = random.randint(1, 8)
            m_dim = random.randint(1, 8)
            p_dim = random.randint(1, 8)

            # 定义返回完整维度的函数
            def dims_full_for_fn():
                if fn == "baddbmm":
                    return (
                        [batch_dim, n_dim, p_dim],
                        [batch_dim, n_dim, m_dim],
                        [batch_dim, m_dim, p_dim],
                    )
                elif fn == "addbmm":
                    return (
                        [n_dim, p_dim],
                        [batch_dim, n_dim, m_dim],
                        [batch_dim, m_dim, p_dim],
                    )
                elif fn == "addmm":
                    return ([n_dim, p_dim], [n_dim, m_dim], [m_dim, p_dim])
                elif fn == "addmv":
                    return ([n_dim], [n_dim, m_dim], [m_dim])
                elif fn == "addr":
                    return ([n_dim, m_dim], [n_dim], [m_dim])
                else:
                    raise AssertionError("unknown function")

            # 获取当前函数所需的完整维度
            (t0_dims_full, t1_dims, t2_dims) = dims_full_for_fn()

            # 生成随机张量，设备为指定设备，数据类型为 float32
            t0_small = torch.randn(*t0_dims_small, device=device).float()
            t1 = torch.randn(*t1_dims, device=device).float()
            t2 = torch.randn(*t2_dims, device=device).float()

            # 将小张量 t0_small 扩展为完整维度，并放到指定设备上
            t0_full = t0_small.expand(*t0_dims_full).to(device)

            # 根据函数名获取对应的 torch 函数对象
            fntorch = getattr(torch, fn)
            # 分别对小张量和完整张量进行运算
            r0 = fntorch(t0_small, t1, t2)
            r1 = fntorch(t0_full, t1, t2)
            # 断言运算结果相等
            self.assertEqual(r0, r1)

    # 测试带有数据类型的矩阵乘法和批矩阵乘法，接受设备和数据类型参数
    @dtypes(torch.float32)
    def test_strided_mm_bmm(self, device, dtype):
        # 创建输入张量 x，指定设备和数据类型
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype, device=device)
        new_shape = [2, 2, 2]
        new_stride = [3, 1, 1]
        # 使用 as_strided 创建具有指定形状和步长的张量 sx
        sx = torch.as_strided(x, size=new_shape, stride=new_stride)

        # 定义 torch 的矩阵乘法 lambda 函数和对应的 numpy 函数 lambda 函数
        torch_fn = lambda x: torch.bmm(x, x)  # noqa: E731
        np_fn = lambda x: np.matmul(x, x)  # noqa: E731
        # 使用自定义函数比较 torch_fn 和 np_fn 在 sx 上的结果
        self.compare_with_numpy(torch_fn, np_fn, sx)

        # 重新定义 torch 的矩阵乘法 lambda 函数
        torch_fn = lambda x: torch.mm(x, x)  # noqa: E731
        # 使用自定义函数比较 torch_fn 和 np_fn 在 sx[0] 上的结果
        self.compare_with_numpy(torch_fn, np_fn, sx[0])

    # 测试矩阵乘法中空输入和混合数据类型错误，接受设备参数
    def test_mm_empty_inputs_mixed_dtype_errors(self, device):
        # 创建整数类型的随机张量 a 和浮点数类型的随机张量 b
        a = torch.randint(0, 10, [1, 10], dtype=torch.int16, device=device)
        b = torch.randn(10, 20, dtype=torch.float32, device=device)
        # 使用断言捕获 RuntimeError 异常，验证数据类型不匹配的情况
        with self.assertRaisesRegex(
            RuntimeError, "expected .* and .* to have the same dtype, but got:"
        ):
            # 执行 torch.mm 操作，验证混合数据类型错误的情况
            torch.mm(a, b)
    # 定义一个测试函数，用于验证 GitHub 上的 PyTorch 问题编号 45724
    def test_matmul_45724(self, device):
        # 创建随机张量 a 和 b，形状为 (65537, 22, 64)，数据类型为半精度浮点数，位于指定设备上
        a = torch.rand(65537, 22, 64, device=device, dtype=torch.half)
        b = torch.rand(65537, 64, 22, device=device, dtype=torch.half)
        # 创建全为 NaN 的张量 c，形状为 (65537, 22, 22)，数据类型为半精度浮点数，位于指定设备上
        c = torch.full((65537, 22, 22), math.nan, dtype=torch.half, device=device)
        # 在 CPU 上计算 a 和 b 的矩阵乘积，并将结果转换为半精度浮点数
        cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).half()
        # 将 a 和 b 的矩阵乘积存储到张量 c 中，使用指定设备
        torch.matmul(a, b, out=c)
        # 断言张量 c 与 cpu_result 相等
        self.assertEqual(c, cpu_result)

    @dtypes(
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    )
    # 定义一个测试函数，用于验证 torch.baddbmm 函数的输入数据类型兼容性
    def test_baddbmm_input_dtypes_compatibility(self, device, dtype):
        # 创建形状为 (1, 2, 2) 的随机张量 batch1 和 batch2，数据类型为单精度浮点数，位于指定设备上
        batch1 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        batch2 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        # 创建形状为 (1, 2, 2) 的随机张量 input_tensor，位于指定设备上，并将其转换为指定数据类型 dtype
        input_tensor = torch.rand((1, 2, 2), device=device).to(dtype)
        # 如果数据类型 dtype 不是 torch.float32，则期望抛出 RuntimeError 异常，异常信息为 "Input dtypes must be the same"
        if dtype != torch.float32:
            with self.assertRaisesRegex(RuntimeError, "Input dtypes must be the same"):
                y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0)
        else:
            # 创建形状为 (1, 2, 2) 的随机张量 out，并填充 NaN，数据类型为指定 dtype，位于指定设备上
            out = torch.randn((1, 2, 2), dtype=dtype, device=device).fill_(torch.nan)
            # 计算 batch1 和 batch2 的矩阵乘积，并将结果加到 input_tensor 上，使用 beta=0.0，将结果存储到 out 中
            y_ref = torch.bmm(batch1, batch2)
            y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0, out=out)
            # 断言 out 与 y_ref 相等
            self.assertEqual(out, y_ref)

    @dtypes(torch.float)
    # 定义一个测试函数，用于验证 torch.baddbmm 函数对输入为 NaN 且 beta=0.0 的情况兼容性
    def test_baddbmm_nan_input_with_zero_beta(self, device, dtype):
        # 遍历不同形状的张量 mat1 和 mat2
        for shape in [[3, 2, 2], [2, 20, 20]]:
            # 创建指定形状的随机张量 mat1 和 mat2，数据类型为指定 dtype，位于指定设备上
            mat1, mat2 = (
                torch.randn(shape, dtype=dtype, device=device) for _ in range(2)
            )
            # 创建包含两个张量的列表 inputs，其中一个填充 NaN
            inputs = [
                torch.randn(shape, dtype=dtype, device=device),
                torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan),
            ]
            # 创建包含三个张量的列表 outs，其中一个为 None，一个为随机张量，一个填充 NaN
            outs = [
                None,
                torch.randn(shape, dtype=dtype, device=device),
                torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan),
            ]
            # 遍历 inputs 和 outs 的所有组合
            options = itertools.product(inputs, outs)
            for input, out in options:
                # 计算 mat1 和 mat2 的矩阵乘积并存储到 y_ref 中
                y_ref = torch.bmm(mat1, mat2)
                # 计算 input 和 mat1、mat2 的加权矩阵乘积，使用 beta=0.0，并将结果存储到 out 中
                y = torch.baddbmm(input, mat1, mat2, beta=0.0, out=out)
                # 断言 y 和 y_ref 相等
                self.assertEqual(y_ref, y)

    @dtypes(torch.float)
    # 定义测试方法，用于测试 torch.addmm 方法的不同参数组合
    def test_addmm_sizes(self, device, dtype):
        # 遍历不同的维度值 m
        for m in [0, 1, 25]:
            # 遍历不同的维度值 n
            for n in [0, 1, 10]:
                # 遍历不同的维度值 k
                for k in [0, 1, 8]:
                    # 生成随机张量 M，形状为 (n, m)，设备为指定设备，数据类型为指定数据类型
                    M = torch.randn(n, m, device=device).to(dtype)
                    # 生成随机张量 m1，形状为 (n, k)，设备为指定设备，数据类型为指定数据类型
                    m1 = torch.randn(n, k, device=device).to(dtype)
                    # 生成随机张量 m2，形状为 (k, m)，设备为指定设备，数据类型为指定数据类型
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    # 调用自定义方法 _test_addmm_addmv，测试 torch.addmm 方法
                    self._test_addmm_addmv(torch.addmm, M, m1, m2)

                    # 生成随机张量 m1，形状为 (n, k+1)，设备为指定设备，数据类型为指定数据类型
                    m1 = torch.randn(n, k + 1, device=device).to(dtype)
                    # 生成随机张量 m2，形状为 (k, m)，设备为指定设备，数据类型为指定数据类型
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    # 断言捕获 RuntimeError 异常，异常消息包含指定的维度信息
                    self.assertRaisesRegex(
                        RuntimeError,
                        f"{n}x{k + 1}.*{k}x{m}",
                        lambda: torch.addmm(M, m1, m2),
                    )
                    # 断言捕获 RuntimeError 异常，异常消息包含指定的维度信息
                    self.assertRaisesRegex(
                        RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.mm(m1, m2)
                    )

    # 使用装饰器 precisionOverride 定义测试方法，用于测试 torch._addmm_activation 方法
    # 结合指定激活函数的情况下的 addmm 操作
    @precisionOverride(
        {
            torch.double: 1e-8,
            torch.float: 1e-4,
            torch.bfloat16: 5e-2,
            torch.half: 5e-2,
            torch.cfloat: 1e-4,
            torch.cdouble: 1e-8,
        }
    )
    # 使用装饰器 dtypes 定义测试方法的数据类型范围
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    # 定义测试方法，用于测试带有 Gelu 激活函数的 torch.addmm 操作
    def test_addmm_gelu(self, device, dtype):
        # 调用自定义方法 _test_addmm_impl，测试 torch._addmm_activation 方法的 "gelu" 操作
        self._test_addmm_impl(torch._addmm_activation, "gelu", device, dtype)

    # 使用装饰器 precisionOverride 定义测试方法，用于测试 torch._addmm_activation 方法
    # 结合指定激活函数的情况下的 addmm 操作
    @precisionOverride(
        {
            torch.double: 1e-8,
            torch.float: 1e-4,
            torch.bfloat16: 5e-2,
            torch.half: 5e-2,
            torch.cfloat: 1e-4,
            torch.cdouble: 1e-8,
        }
    )
    # 使用装饰器 dtypes 定义测试方法的数据类型范围
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    # 定义测试方法，用于测试带有 ReLU 激活函数的 torch.addmm 操作
    def test_addmm_relu(self, device, dtype):
        # 调用自定义方法 _test_addmm_impl，测试 torch._addmm_activation 方法的 "relu" 操作
        self._test_addmm_impl(torch._addmm_activation, "relu", device, dtype)

    # 使用装饰器 dtypes 定义测试方法的数据类型范围
    @dtypes(torch.float, torch.bfloat16, torch.half)
    # 定义测试方法，用于测试 addmv 函数在不同情况下的行为
    def test_addmv_rowmajor_colmajor_incx_incy_lda(self, device, dtype):
        # 设置输出大小 o 和求和大小 s
        o = 5
        s = 3
        # 创建输入矩阵 a_data，其元素从 1 到 o*s，设备为指定的 device，数据类型为 dtype
        a_data = torch.arange(1, o * s + 1, device=device, dtype=dtype).view(o, s)
        # 创建输入向量 x_data，其元素从 1 到 s，步长为 1，设备和数据类型与 a_data 相同
        x_data = torch.arange(1, s + 1, 1, device=device, dtype=dtype)
        # 创建输入向量 y_data，其元素全为 1，设备和数据类型与 a_data 相同
        y_data = torch.ones(o, device=device, dtype=dtype)
        # 创建期望输出 control，其元素为预先设定的值，设备和数据类型与 a_data 相同
        control = torch.tensor(
            [15.0, 33.0, 51.0, 69.0, 87.0], device=device, dtype=dtype
        )

        # 定义内部测试函数 _test，用于具体测试不同参数组合下的功能
        def _test(row_major, incx, incy, lda_tail):
            # 根据 row_major 参数选择性地创建 o*s 或 s*o 大小的全 NaN 张量 a_storage
            if row_major:
                a_storage = torch.full(
                    (o, s + lda_tail), float("nan"), device=device, dtype=dtype
                )
            else:
                a_storage = torch.full(
                    (s, o + lda_tail), float("nan"), device=device, dtype=dtype
                ).permute(1, 0)
            # 从 a_data 复制数据到 a_storage，并根据 row_major 调整形状得到张量 a
            a = a_storage[:o, :s].copy_(a_data)

            # 创建大小为 s*incx 的全 NaN 张量 x_storage，并从 x_data 复制数据到第一列得到向量 x
            x_storage = torch.full((s, incx), float("nan"), device=device, dtype=dtype)
            x = x_storage[:, 0].copy_(x_data)

            # 创建大小为 o*incy 的全 NaN 张量 y_storage，并从 y_data 复制数据到第一列得到向量 y
            y_storage = torch.full((o, incy), float("nan"), device=device, dtype=dtype)
            y = y_storage[:, 0].copy_(y_data)

            # 调用 self._test_addmm_addmv 方法，测试 torch.addmv 函数的行为
            self._test_addmm_addmv(torch.addmv, y, a, x)

        # 使用 itertools.product 生成所有 row_major、incx、incy、lda_tail 组合的测试
        for row_major, incx, incy, lda_tail in itertools.product(
            (False, True), (1, 2), (1, 2), (0, 1)
        ):
            # 调用 _test 函数进行测试
            _test(row_major, incx, incy, lda_tail)

    # 设置 torch.float16、torch.half、torch.float32 三种数据类型的精度覆盖
    @precisionOverride(
        {
            torch.double: 1e-8,
            torch.float: 1e-4,
            torch.bfloat16: 0.6,
            torch.half: 1e-1,
            torch.cfloat: 1e-4,
            torch.cdouble: 1e-8,
        }
    )
    # 指定测试的数据类型为 torch.bfloat16、torch.half、torch.float32 三种
    @dtypes(torch.bfloat16, torch.half, torch.float32)
    # 定义测试 cublasLtMatmul 的极端情况
    def test_corner_cases_of_cublasltmatmul(self, device, dtype):
        # 测试常见情况
        M = torch.randn(128, device=device).to(dtype)
        m1 = torch.randn(2048, 2400, device=device).to(dtype)
        m2 = torch.randn(128, 2400, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)
        # 测试 Ntrans_B 情况下的 ld >> 行数
        m1 = torch.rand([128, 2400]).to(dtype).to(device).t()
        m2 = torch.rand([2048, 25272]).to(dtype).to(device).t()[21940:24340]
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2.t(), m1)
        # 测试 trans_A 情况下的 ld >> 行数
        m1 = torch.rand([128, 25272]).to(dtype).to(device)[:, 21940:24340].t()
        m2 = torch.randn(2048, 2400, device=device).to(dtype)
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2, m1)
        # 测试大尺寸张量 dim > 65535
        M = torch.randn(16, device=device).to(dtype)
        m1 = torch.randn(32, 131071, device=device).to(dtype)
        m2 = torch.randn(16, 131071, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)
    # 测试大规模矩阵乘法的反向传播，验证是否会因为内存不足而导致 Out Of Memory (OOM) 错误
    def test_large_bmm_backward(self, device):
        # 创建大小为 [1024, 2, 1024] 的随机张量 A，并确保其连续性和转置性
        A = torch.randn([1024, 2, 1024], device=device).mT.contiguous().mT
        # 创建大小为 [1, 1024, 65536] 的随机张量 B，并设置需要计算梯度
        B = torch.randn([1, 1024, 65536], device=device, requires_grad=True)
        # 创建大小为 [1024, 2, 65536] 的随机张量 G
        G = torch.randn([1024, 2, 65536], device=device)

        # 执行矩阵乘法 A @ B，并对结果执行反向传播，使用梯度 G
        # 需要注意不要创建大小为 [1024, 1024, 65536] 的中间张量（需要256GB内存），以避免内存溢出
        (A @ B).backward(G)

    # 测试大规模矩阵乘法及矩阵-矩阵乘法的反向传播，验证是否会因为内存不足而导致 Out Of Memory (OOM) 错误
    def test_large_bmm_mm_backward(self, device):
        # 创建大小为 [1024, 2, 1024] 的随机张量 A，并确保其连续性和转置性
        A = torch.randn([1024, 2, 1024], device=device).mT.contiguous().mT
        # 创建大小为 [1024, 65536] 的随机张量 B，并设置需要计算梯度
        B = torch.randn([1024, 65536], device=device, requires_grad=True)
        # 创建大小为 [1024, 2, 65536] 的随机张量 G
        G = torch.randn([1024, 2, 65536], device=device)

        # 执行矩阵乘法 A @ B，并对结果执行反向传播，使用梯度 G
        # 需要注意不要创建大小为 [1024, 1024, 65536] 的中间张量（需要256GB内存），以避免内存溢出
        (A @ B).backward(G)

    # 检查单个矩阵乘法的结果是否与预期一致，并检查是否符合数值精度要求
    def check_single_matmul(self, x, y):
        def assertEqual(answer, expected):
            if x.dtype.is_floating_point or x.dtype.is_complex:
                k = max(x.shape[-1], 1)  # 根据矩阵的大小调整公差的尺度
                self.assertEqual(
                    answer,
                    expected,
                    msg=f"{x.shape} x {y.shape} = {answer.shape}",
                    atol=k * 5e-5,  # 绝对公差根据矩阵大小进行调整
                    rtol=1e-4,      # 相对公差
                )
            else:
                self.assertEqual(
                    answer, expected, msg=f"{x.shape} x {y.shape} = {answer.shape}"
                )

        # 测试 x @ y 的结果是否与 numpy 中的 np.matmul(x, y) 一致
        expected = np.matmul(x.cpu(), y.cpu())
        ans = torch.matmul(x, y)
        self.assertTrue(ans.is_contiguous())  # 验证结果张量是否是连续的
        assertEqual(ans, expected)

        # 测试指定输出张量 out 是否正确，并验证其连续性
        out = torch.empty_like(ans)
        ans = torch.matmul(x, y, out=out)
        self.assertIs(ans, out)  # 验证输出张量是否与预期的相同
        self.assertTrue(ans.is_contiguous())  # 验证结果张量是否是连续的
        assertEqual(ans, expected)

    # 生成用于矩阵乘法的输入尺寸，确保生成的输入对在矩阵乘法操作中是兼容的
    def gen_sizes_matmul(self, x_dim, y_dim=4, matrix_size=4, batch_size=3):
        """
        Generates sequences of tuples (x, y) of with size(x) = x_dim and
        size(y) <= y_dim that are compatible wrt. matmul
        """
        assert x_dim >= 1
        assert y_dim >= 2
        x = x_dim
        for y in range(1, y_dim + 1):
            for batch, mn in product(
                product(range(batch_size), repeat=max(x - 2, y - 2, 0)),
                product(range(matrix_size), repeat=min(y, 2)),
            ):
                if x == 1:
                    size_x = mn[:1]
                    size_y = batch + mn
                    yield size_x, size_y
                else:
                    for k in range(matrix_size):
                        size_x = (k,) + mn[:1]
                        if x > 2:
                            size_x = batch[-(x - 2) :] + size_x
                        size_y = mn
                        if y > 2:
                            size_y = batch[-(y - 2) :] + size_y
                        yield size_x, size_y

    # 将测试标记为 torch.float 类型
    @dtypes(torch.float)
    # 使用部分函数 make_tensor 来生成张量，并指定设备和数据类型
    make_arg = partial(make_tensor, device=device, dtype=dtype)

    # 遍历所有大小为 1 的矩阵乘积大小组合，以及是否连续和非连续存储的标志
    for (size_x, size_y), nctg_x, nctg_y in product(
        self.gen_sizes_matmul(1), (True, False), (True, False)
    ):
        # 生成大小为 size_x 的张量 x，并根据 nctg_x 决定是否非连续存储
        x = make_arg(size_x, noncontiguous=nctg_x)
        # 生成大小为 size_y 的张量 y，并根据 nctg_y 决定是否非连续存储
        y = make_arg(size_y, noncontiguous=nctg_y)
        # 调用检查单次矩阵乘积的函数，检查张量 x 和 y 的乘积
        self.check_single_matmul(x, y)

@dtypes(torch.float)
def test_matmul_small_brute_force_2d_Nd(self, device, dtype):
    # 使用部分函数 make_tensor 来生成张量，并指定设备和数据类型
    make_arg = partial(make_tensor, device=device, dtype=dtype)

    # 遍历所有大小为 2 的矩阵乘积大小组合，以及是否连续和非连续存储的标志
    for (size_x, size_y), nctg_x, nctg_y in product(
        self.gen_sizes_matmul(2), (True, False), (True, False)
    ):
        # 生成大小为 size_x 的张量 x，并根据 nctg_x 决定是否非连续存储
        x = make_arg(size_x, noncontiguous=nctg_x)
        # 生成大小为 size_y 的张量 y，并根据 nctg_y 决定是否非连续存储
        y = make_arg(size_y, noncontiguous=nctg_y)
        # 调用检查单次矩阵乘积的函数，检查张量 x 和 y 的乘积
        self.check_single_matmul(x, y)

@dtypes(torch.float)
def test_matmul_small_brute_force_3d_Nd(self, device, dtype):
    # 使用部分函数 make_tensor 来生成张量，并指定设备和数据类型
    make_arg = partial(make_tensor, device=device, dtype=dtype)

    # 遍历所有大小为 3 的矩阵乘积大小组合，以及是否连续和非连续存储的标志
    for (size_x, size_y), nctg_x, nctg_y in product(
        self.gen_sizes_matmul(3), (True, False), (True, False)
    ):
        # 生成大小为 size_x 的张量 x，并根据 nctg_x 决定是否非连续存储
        x = make_arg(size_x, noncontiguous=nctg_x)
        # 生成大小为 size_y 的张量 y，并根据 nctg_y 决定是否非连续存储
        y = make_arg(size_y, noncontiguous=nctg_y)
        # 调用检查单次矩阵乘积的函数，检查张量 x 和 y 的乘积
        self.check_single_matmul(x, y)

@dtypes(torch.float)
def test_matmul_out_kernel_errors_with_autograd(self, device, dtype):
    # 创建一个大小为 (256, 512) 的张量 a，设置在指定设备上，支持梯度计算
    a = torch.empty(
        (256, 512), device=device, dtype=dtype, requires_grad=True
    ).unsqueeze(0)
    # 创建一个大小为 (4, 128, 512) 的张量 b，设置在指定设备上，支持梯度计算，并对维度进行转置
    b = torch.empty(
        (4, 128, 512), device=device, dtype=dtype, requires_grad=True
    ).transpose(-1, -2)
    # 创建一个大小为 (256, 4, 128) 的张量 c，设置在指定设备上，不支持梯度计算，并进行维度转换
    c = torch.empty((256, 4, 128), device=device, dtype=dtype).movedim(1, 0)

    # 使用 a 和 b 的副本计算矩阵乘积，并将结果存储到 c 中
    torch.matmul(a.detach(), b.detach(), out=c)

    # 使用带有 out 参数的矩阵乘积函数，捕获 RuntimeError，并验证不支持自动微分的错误信息
    with self.assertRaisesRegex(
        RuntimeError,
        "functions with out=... arguments don't support automatic differentiation",
    ):
        torch.matmul(a, b, out=c)

    # 在 torch.no_grad() 环境下使用带有 out 参数的矩阵乘积函数，确保没有梯度计算发生
    with torch.no_grad():
        torch.matmul(a, b, out=c)
# 使用给定的测试类 TestBasicGEMM 实例化设备类型测试，并将其添加到全局作用域中
instantiate_device_type_tests(TestBasicGEMM, globals(), only_for="xpu")

# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则运行测试
if __name__ == "__main__":
    run_tests()
```