# `.\pytorch\test\inductor\test_mmdecomp.py`

```py
# Owner(s): ["module: nn"]

# 导入必要的库
import math  # 导入数学库
import unittest  # 导入单元测试库
from typing import List, Tuple, Union  # 导入类型提示需要的模块

import torch  # 导入PyTorch库
from torch._inductor import config  # 导入配置模块
from torch.testing._internal.common_cuda import SM80OrLater  # 导入CUDA相关模块
from torch.testing._internal.common_device_type import instantiate_device_type_tests  # 导入设备类型测试相关模块
from torch.testing._internal.common_nn import NNTestCase  # 导入神经网络测试相关模块
from torch.testing._internal.common_utils import IS_WINDOWS, parametrize, run_tests  # 导入测试工具相关模块
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU  # 导入深度学习加速器相关模块

# 默认的绝对误差容限
default_atol = {
    torch.float16: 1e-3,
    torch.bfloat16: float("infinity"),
    torch.float32: 1e-5,
}
# 默认的相对误差容限
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: float("infinity"),
    torch.float32: 1.3e-6,
}


def rand_math_tensor(
    shape: Tuple[Union[int, List[int]]],
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    packed: bool = False,
) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    # 使用torch.randn创建指定形状、设备和数据类型的随机张量
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def init_tensor(tensor_list, **kwargs) -> torch.Tensor:
    # 使用给定的tensor_list创建张量，并将其移动到指定的设备
    return torch.Tensor(tensor_list).to(**kwargs)


def run_comp_nocomp(function, *inputs, **kwargs):
    # 编译函数为C函数
    c_function = torch.compile(function)

    # 调用原始函数和编译后的函数
    f_res = function(*inputs)
    cf_res = c_function(*inputs)

    # 检查函数计算结果的近似程度，根据给定的atol和rtol容忍度
    if not (math.isinf(kwargs.get("atol", 0.0)) or math.isinf(kwargs.get("rtol", 0.0))):
        torch.testing.assert_close(f_res, cf_res, **kwargs)


# 以下函数用于多个测试中使用
def torch_mm(a, b):
    # torch.mm函数：矩阵乘法
    return torch.mm(a, b)


def torch_addmm(add, b, c):
    # torch.addmm函数：矩阵加权相加
    return torch.addmm(add, b, c)


def torch_bmm(a, b):
    # torch.bmm函数：批量矩阵乘法
    return torch.bmm(a, b)


def torch_baddbmm(add, b, c, alpha, beta):
    # torch.baddbmm函数：批量矩阵乘加
    return torch.baddbmm(add, b, c, alpha=alpha, beta=beta)


# 测试使用的张量形状列表
ts_list = [
    (1, 32, 32, 1),
    (1, 10, 10, 1),
    (1, 3, 3, 1),
    (32, 1, 1, 32),
    (3, 1, 1, 3),
    (4, 1, 1, 9),
    (9, 1, 1, 4),
]


class TestDecomp(NNTestCase):
    _do_cuda_memory_leak_check = GPU_TYPE == "cuda"
    _do_cuda_non_default_stream = GPU_TYPE == "cuda"

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize("dtype", [torch.float, torch.bfloat16])
    # 测试简单的矩阵乘法函数
    def test_simple_mm(self, device, dtype):
        # 设置一个误差放大因子
        fudge = 10
        # 设置相对误差阈值
        rtol = default_rtol[dtype] * fudge
        # 设置绝对误差阈值
        atol = default_atol[dtype] * fudge

        # 遍历测试大小列表
        for t_size in ts_list:
            # 解包测试大小元组
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            # 创建随机数填充的张量 t1, t2, tadd
            t1 = rand_math_tensor((a1_0, a1_1), dtype=dtype, device=device)
            t2 = rand_math_tensor((a2_0, a2_1), dtype=dtype, device=device)
            tadd = rand_math_tensor((a1_0, a2_1), dtype=dtype, device=device)

            # 运行比较函数，对 torch_mm 函数进行测试
            run_comp_nocomp(torch_mm, t1, t2, rtol=rtol, atol=atol)
            # 运行比较函数，对 torch_addmm 函数进行测试
            run_comp_nocomp(torch_addmm, tadd, t1, t2, rtol=rtol, atol=atol)

    # 当没有 GPU 时跳过测试，需要 Triton 支持
    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize(
        "dtype", [torch.float, torch.bfloat16] if SM80OrLater else [torch.float]
    )
    @parametrize("bs", [1, 2, 4, 10])
    # 测试批量矩阵乘法
    def test_batched_mm(self, device, dtype, bs):
        # 设置一个误差放大因子
        fudge = 3
        # 设置相对误差阈值
        rtol = default_rtol[dtype] * fudge
        # 设置绝对误差阈值
        atol = default_atol[dtype] * fudge

        # 遍历测试大小列表
        for t_size in ts_list:
            # 解包测试大小元组
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            # 创建随机数填充的批量张量 t1, t2, tadd
            t1 = rand_math_tensor((bs, a1_0, a1_1), dtype=dtype, device=device)
            t2 = rand_math_tensor((bs, a2_0, a2_1), dtype=dtype, device=device)
            tadd = rand_math_tensor((bs, a1_0, a2_1), dtype=dtype, device=device)

            # 运行比较函数，对 torch_bmm 函数进行测试
            run_comp_nocomp(torch_bmm, t1, t2, rtol=rtol, atol=atol)

            # 遍历 alpha 和 beta 的组合
            for alpha in (0, 1, -1, 0.5, -0.5):
                for beta in (0, 1, -1, 0.5, -0.5):
                    # 运行比较函数，对 torch_baddbmm 函数进行测试
                    run_comp_nocomp(
                        torch_baddbmm, tadd, t1, t2, alpha, beta, rtol=rtol, atol=atol
                    )

    # 当没有 GPU 时跳过测试，需要 Triton 支持
    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @config.patch(coordinate_descent_tuning=True)
    # 测试最后一维大小为一的批量矩阵乘法
    def test_bmm_batch2_last_dim_size_is_one(self, device):
        # 设置一个误差放大因子
        fudge = 3
        # 设置相对误差阈值
        rtol = default_rtol[torch.float32] * fudge
        # 设置绝对误差阈值
        atol = default_atol[torch.float32] * fudge

        # 创建随机数填充的张量 t1, t2
        t1 = torch.randn(1, 32, 2, device=device)
        t2 = torch.randn(1, 2, 1, device=device)

        # 运行比较函数，对 torch_bmm 函数进行测试
        run_comp_nocomp(torch_bmm, t1, t2, rtol=rtol, atol=atol)

    # 当没有 GPU 时跳过测试，需要 Triton 支持
    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    # 测试特定情况
    def test_some(self, device, dtype):
        # 如果是在 GPU 上，并且数据类型是 torch.int，则跳过测试
        if device.startswith(GPU_TYPE) and dtype == torch.int:
            return

        # 运行比较函数，对 torch_mm 函数进行测试
        run_comp_nocomp(
            torch_mm,
            init_tensor([[1], [2], [3], [4]], dtype=dtype, device=device),
            init_tensor([[1, 2, 3, 4]], dtype=dtype, device=device),
        )
        # 运行比较函数，对 torch_mm 函数进行测试
        run_comp_nocomp(
            torch_mm,
            init_tensor([[1, 2, 3, 4]], dtype=dtype, device=device),
            init_tensor([[1], [2], [3], [4]], dtype=dtype, device=device),
        )

    # 当没有 GPU 时跳过测试，需要 Triton 支持
    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    # 使用参数化装饰器，将该测试方法参数化为多组测试用例
    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    @parametrize("bs", [1, 2, 4, 10])
    # 定义批处理测试方法，接受设备类型、数据类型和批处理大小作为参数
    def test_some_batched(self, device, dtype, bs):
        # 如果设备类型以 GPU_TYPE 开头且数据类型为 torch.int，则跳过当前测试用例
        if device.startswith(GPU_TYPE) and dtype == torch.int:
            return

        # 调用 run_comp_nocomp 函数进行批处理计算，传入 torch_bmm 函数以及初始化的张量
        # 第一个张量是形状为 [[[1], [2], [3], [4]]] * bs 的张量，指定数据类型和设备
        # 第二个张量是形状为 [[[1, 2, 3, 4]]] * bs 的张量，指定数据类型和设备
        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
        )
        # 再次调用 run_comp_nocomp 函数进行批处理计算，传入相同的函数和参数
        # 但这次将第一个张量和第二个张量的维度顺序调换
        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
        )
# 定义设备类型元组，包括 "cpu" 和 GPU_TYPE
device_types = ("cpu", GPU_TYPE)

# 根据指定的设备类型实例化测试用例 TestDecomp，并将其注册到全局命名空间中
instantiate_device_type_tests(TestDecomp, globals(), only_for=device_types)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 在 Windows 系统上不支持 torch.compile() 方法
    if not IS_WINDOWS:
        # 运行测试函数
        run_tests()
```