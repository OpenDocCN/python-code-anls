# `.\pytorch\test\inductor\test_cpu_select_algorithm.py`

```
# Owner(s): ["oncall: cpu inductor"]

# 引入必要的库和模块
import contextlib  # 上下文管理器，用于管理上下文的资源
import functools  # 函数装饰器，用于包装函数以添加额外功能

import sys  # 系统相关功能
import unittest  # 单元测试框架
from typing import Optional  # 类型提示，声明可选的类型
from unittest.mock import patch  # 单元测试模拟库，用于模拟对象

import torch  # PyTorch 深度学习框架
import torch._dynamo.config  # PyTorch 内部配置模块
import torch._dynamo.config as dynamo_config  # PyTorch Dynamo 配置
import torch._inductor.config as inductor_config  # PyTorch Inductor 配置
import torch._inductor.select_algorithm as select_algorithm  # PyTorch Inductor 算法选择
from torch._dynamo.utils import counters  # PyTorch Dynamo 实用工具：计数器
from torch._inductor.cpu_vec_isa import VecAMX  # PyTorch Inductor CPU 向量指令集
from torch._inductor.test_case import run_tests, TestCase  # PyTorch Inductor 测试相关
from torch.testing._internal.common_device_type import (  # PyTorch 内部测试设备类型
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_utils import IS_MACOS, parametrize, TEST_MKL  # PyTorch 内部通用工具

# 尝试导入测试相关模块，处理可能的导入错误和跳过测试异常
try:
    try:
        from . import test_torchinductor  # 尝试相对导入测试模块
    except ImportError:
        import test_torchinductor  # 相对导入失败，尝试绝对导入
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)  # 在测试跳过情况下，退出测试
    raise  # 抛出未捕获的跳过测试异常

check_model = test_torchinductor.check_model  # 设置检查模型的函数别名

aten = torch.ops.aten  # 设置 PyTorch ATen 操作的别名


# 定义装饰器函数，用于给函数添加补丁操作
def patches(fn):
    def skip_cache(self, choices, name, key, benchmark):
        if benchmark is None:
            return {}
        timings = benchmark(choices)
        for choice, timing in timings.items():
            if isinstance(choice, select_algorithm.ExternKernelCaller):
                # 故意使 ATEN 内核变慢，以覆盖模板内核总是选择融合应用和运行时正确性检查的情况
                timings[choice] = timing * 1000
        return timings

    # 应用一系列补丁操作到函数 fn 上
    for patcher in [
        dynamo_config.patch(verbose=True),  # 应用 Dynamo 配置的补丁，启用详细模式
        inductor_config.patch(  # 应用 Inductor 配置的补丁，启用调试模式、最大自动调整、后处理融合及最大自动调整 GEMM 后端
            debug=True,
            max_autotune=True,
            epilogue_fusion=True,
            max_autotune_gemm_backends="CPP,ATEN",
        ),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),  # 对算法选择模块的 VERIFY 对象进行补丁，设置容差
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),  # 对算法选择缓存进行补丁，跳过缓存查找
    ]:
        fn = patcher(fn)  # 应用每个补丁到函数 fn

    # 对函数 fn 进行装饰，清除计数器，设置随机种子，并返回装饰后的函数
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()  # 清除计数器
        torch.manual_seed(12345)  # 设置随机种子
        return fn(*args, **kwargs)

    return wrapped  # 返回装饰后的函数


# 上下文管理器，用于验证函数，根据数据类型设置容差
@contextlib.contextmanager
def verify(dtype):
    # 对于 bfloat16 和 half 类型，由于不同的内核实现顺序关联，需要放宽容差
    atol, rtol = 1e-4, 1e-4
    if dtype == torch.half or dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    with patch.object(select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)):
        yield atol, rtol  # 返回容差设置


# 根据给定的后处理名称和可选的其他张量，返回相应的后处理函数或张量
def _get_epilogue(epilogue: str, other: Optional[torch.Tensor] = None):
    if epilogue == "none":
        return lambda x: x
    elif epilogue == "relu":
        return torch.nn.ReLU()  # 返回 ReLU 激活函数
    elif epilogue == "gelu":
        return torch.nn.GELU()  # 返回 GELU 激活函数
    elif epilogue == "silu":
        return torch.nn.SiLU()  # 返回 SiLU 激活函数
    elif epilogue == "sigmoid":
        return torch.nn.Sigmoid()  # 返回 Sigmoid 激活函数
    elif epilogue == "tanh":
        return torch.nn.Tanh()  # 返回 Tanh 激活函数
    elif epilogue == "hardswish":
        # 如果 epilogue 等于 "hardswish"，返回一个 Hardswish 激活函数对象
        return torch.nn.Hardswish()
    elif epilogue == "hardsigmoid":
        # 如果 epilogue 等于 "hardsigmoid"，返回一个 Hardsigmoid 激活函数对象
        return torch.nn.Hardsigmoid()
    elif epilogue == "leaky_relu":
        # 如果 epilogue 等于 "leaky_relu"，返回一个 LeakyReLU 激活函数对象
        return torch.nn.LeakyReLU()
    elif epilogue == "hardtanh":
        # 如果 epilogue 等于 "hardtanh"，返回一个 Hardtanh 激活函数对象
        return torch.nn.Hardtanh()
    elif epilogue == "add":
        # 如果 epilogue 等于 "add"，返回一个接受参数 x 并返回 x + other 的匿名函数
        return lambda x: x + other
    elif epilogue == "sub":
        # 如果 epilogue 等于 "sub"，返回一个接受参数 x 并返回 x - other 的匿名函数
        return lambda x: x - other
    elif epilogue == "mul":
        # 如果 epilogue 等于 "mul"，返回一个接受参数 x 并返回 x * other 的匿名函数
        return lambda x: x * other
    elif epilogue == "div":
        # 如果 epilogue 等于 "div"，返回一个接受参数 x 并返回 x / other 的匿名函数
        return lambda x: x / other
class TestSelectAlgorithm(TestCase):
    common = check_model  # 定义共享的测试模型对象

    @inductor_config.patch({"freezing": True})  # 设置配置补丁，冻结特定参数
    @patches  # 应用测试补丁
    @torch.no_grad  # 禁用梯度计算
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")  # 如果条件不满足则跳过测试
    @parametrize("batch_size", (1, 2, 1000))  # 参数化测试，批量大小取值
    @parametrize("in_features", (1, 1000))  # 参数化测试，输入特征取值
    @parametrize("out_features", (1, 1024))  # 参数化测试，输出特征取值
    @parametrize("bias", (True, False))  # 参数化测试，是否使用偏置
    @parametrize("input_3d", (True, False))  # 参数化测试，输入是否为三维
    @dtypes(torch.float, torch.bfloat16, torch.half)  # 参数化测试，数据类型取值
    def test_linear_static_shapes(
        self, batch_size, in_features, out_features, bias, input_3d, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()  # 清空计数器
        mod = M(bias=bias).to(dtype=dtype).eval()  # 创建模型实例并设置为评估模式
        B = (2, batch_size) if input_3d else (batch_size,)  # 根据输入维度确定批量大小
        v = torch.randn(*B, in_features).to(dtype=dtype)  # 生成指定维度和数据类型的随机张量
        with verify(dtype) as (atol, rtol):  # 使用指定的数据类型进行验证
            self.common(mod, (v,), atol=atol, rtol=rtol)  # 调用共享的模型测试函数，验证结果
        if (
            counters["inductor"]["decompose_mm"] > 0  # 如果某些计数器大于0
            or counters["inductor"]["decompose_addmm"] > 0
        ):
            # 特殊情况，使用向量化代码生成
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)
        else:
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)  # 否则进行自动调优选择算法

    @inductor_config.patch({"freezing": True})  # 设置配置补丁，冻结特定参数
    @patches  # 应用测试补丁
    @torch.no_grad  # 禁用梯度计算
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")  # 如果条件不满足则跳过测试
    @parametrize("bias", (True, False))  # 参数化测试，是否使用偏置
    @dtypes(torch.float)  # 参数化测试，数据类型取值
    def test_linear_input_transpose(self, bias, dtype):
        batch_size = 384  # 批量大小设置
        in_features = 196  # 输入特征数设置
        out_features = 384  # 输出特征数设置

        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            @torch.compile  # 编译 Torch 脚本以优化性能
            def forward(self, x):
                return self.linear(x)

        counters.clear()  # 清空计数器
        mod = M(bias=bias).to(dtype=dtype).eval()  # 创建模型实例并设置为评估模式
        v = torch.randn(in_features, batch_size).to(dtype=dtype)  # 生成指定维度和数据类型的随机张量
        self.common(mod, (v.transpose(0, 1),))  # 调用共享的模型测试函数，验证结果（输入转置）
        # TODO(jgong5): 支持转置输入
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)  # 检查自动调优选择算法计数为0

    @inductor_config.patch({"freezing": True})  # 设置配置补丁，冻结特定参数
    @patches  # 应用测试补丁
    @torch.no_grad  # 禁用梯度计算
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")  # 如果条件不满足则跳过测试
    @parametrize("bias", (True, False))  # 参数化测试，是否使用偏置
    @parametrize(
        "epilogue",
        (
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh",
            "hardswish",
            "hardsigmoid",
            "leaky_relu",
            "hardtanh",
            "add",
            "sub",
            "mul",
            "div",
        ),
    )  # 参数化测试，激活函数取值
    # 将装饰器应用于测试方法，指定参数类型为 torch.float, torch.bfloat16, torch.half
    @dtypes(torch.float, torch.bfloat16, torch.half)
    # 定义一个测试方法，测试带有逐点运算的线性层
    def test_linear_with_pointwise(self, bias, epilogue, dtype):
        # 设定批量大小为 384
        batch_size = 384
        # 输入特征数为 196
        in_features = 196
        # 输出特征数为 384
        out_features = 384

        # 定义一个内嵌的 PyTorch 模块类 M
        class M(torch.nn.Module):
            # 初始化方法，接受 bias, epilogue, other 参数
            def __init__(self, bias, epilogue, other):
                super().__init__()
                # 创建一个线性层，输入特征数为 in_features，输出特征数为 out_features，带有偏置参数
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                # 获取指定的 epilogue 函数并存储在 self.epilogue 中
                self.epilogue = _get_epilogue(epilogue, other)

            # 前向传播方法，接受输入张量 x，返回经过线性层和 epilogue 函数处理后的结果
            def forward(self, x):
                return self.epilogue(self.linear(x))

        # 清空计数器
        counters.clear()
        # 创建随机张量 v，形状为 (batch_size, in_features)，并转换为指定的 dtype 类型
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        # 创建随机张量 u，形状为 (batch_size, out_features)，并转换为指定的 dtype 类型
        u = torch.randn(batch_size, out_features).to(dtype=dtype)
        # 实例化 M 类，传入 bias, epilogue, other 参数，并转换为指定的 dtype 类型，并设定为评估模式
        mod = M(bias=bias, epilogue=epilogue, other=u).to(dtype=dtype).eval()
        # 使用 verify 上下文管理器验证 dtype 的数值误差阈值 (atol, rtol)
        with verify(dtype) as (atol, rtol):
            # 调用 common 方法对模型进行测试，传入输入张量 v 和误差阈值 atol, rtol
            self.common(mod, (v,), atol=atol, rtol=rtol)
        # 断言计数器中的选择算法自动调整次数为 1
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        
        # 检查多个条件，其中满足以下情况之一时不计算 epilogue 融合计数器：
        if (
            (
                dtype == torch.bfloat16
                or (
                    dtype == torch.float16
                    and torch.ops.mkldnn._is_mkldnn_fp16_supported()
                )
            )
            and epilogue != "mul"
            and epilogue != "div"
            or (dtype == torch.half and epilogue == "add" and not bias)
        ):
            # 如果满足上述条件，则期望 epilogue 融合计数器为 0
            # 多种情况下不计入 epilogue 融合：
            # 1. 对于 bfloat16，epilogue 融合是模板的一部分，不通过调度器融合。
            #    这对于硬件支持 float16 指令时也适用。例外情况是不支持一DNN线性操作的 mul 或 div 融合。
            # 2. 对于 float16，由于不适用一DNN线性操作，无偏置的线性加 epilogue add 被视为有偏置的线性。
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 0)
        else:
            # 否则，期望 epilogue 融合计数器为 1
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)
    def test_linear_with_transpose(self, bias, epilogue, dtype):
        # 设置批量大小
        batch_size = 384
        # 输入特征数
        in_features = 196
        # 输出特征数
        out_features = 128

        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            def __init__(self, bias, epilogue, other):
                super().__init__()
                # 初始化模块的后处理函数
                self.epilogue = _get_epilogue(epilogue, other)
                # 创建线性层，输入特征数为in_features，输出特征数为out_features，是否带偏置由bias决定
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x, y):
                # 模型的前向传播，先线性变换再进行后处理，然后转置结果并加上y
                return self.epilogue(self.linear(x)).transpose(0, 1) + y

        # 清空计数器
        counters.clear()
        # 生成一个随机张量v，形状为(batch_size, in_features)，数据类型为dtype
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        # 生成一个随机张量u，形状为(out_features, batch_size)，数据类型为dtype
        u = torch.randn(out_features, batch_size).to(dtype=dtype)
        # 生成一个随机张量other，形状为(batch_size, out_features)，数据类型为dtype
        other = torch.randn(batch_size, out_features).to(dtype=dtype)
        # 创建M类的实例mod，传入bias、epilogue和other作为参数，并转换为dtype类型，然后设为评估模式
        mod = M(bias=bias, epilogue=epilogue, other=other).to(dtype=dtype).eval()
        # 使用verify上下文管理器，验证模型在指定精度下的表现
        with verify(dtype) as (atol, rtol):
            # 调用self.common方法，传入mod和(v, u)，并设定绝对误差和相对误差
            self.common(mod, (v, u), atol=atol, rtol=rtol)
        # 断言计数器中"inductor"子项中"select_algorithm_autotune"的值为1
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        # 断言计数器中"inductor"子项中"cpp_epilogue_fusion_counter"的值为1
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)
    # 定义一个测试方法，测试线性层与AMX（Advanced Matrix Extensions）相关功能
    def test_linear_amx(self, bias):
        # 定义批处理大小
        batch_size = 1024
        # 定义输入特征维度
        in_features = 1024
        # 定义输出特征维度
        out_features = 1024
        # 定义数据类型为bfloat16
        dtype = torch.bfloat16

        # 定义一个内部模型类M，继承自torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                # 定义一个线性层，输入特征数、输出特征数和是否使用偏置
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            # 前向传播方法
            def forward(self, x):
                return self.linear(x)

        # 清空计数器
        counters.clear()
        # 生成随机输入数据v，并转换为指定数据类型dtype
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        # 创建一个M类的实例mod，传入偏置参数并转换为指定数据类型dtype，设置为评估模式
        mod = M(bias=bias).to(dtype=dtype).eval()
        # 使用verify上下文管理器，获取绝对误差atol和相对误差rtol
        with verify(dtype) as (atol, rtol):
            # 调用common方法进行通用测试
            self.common(mod, (v,), atol=atol, rtol=rtol)
        # 断言autotune自动调优选择算法的计数为1
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        
        # 实例化VecAMX类
        vec_amx = VecAMX()
        # 如果vec_amx为真值
        if vec_amx:
            # 断言cpp_micro_gemm_amx_counter计数大于0
            self.assertTrue(counters["inductor"]["cpp_micro_gemm_amx_counter"] > 0)
        else:
            # 否则断言cpp_micro_gemm_amx_counter计数为0
            self.assertEqual(counters["inductor"]["cpp_micro_gemm_amx_counter"], 0)

    # 使用inductor_config.patch修饰器设置冻结参数为True
    @inductor_config.patch({"freezing": True})
    # 使用patches修饰器
    @patches
    # 使用torch.no_grad修饰器
    @torch.no_grad
    # 参数化测试方法，测试线性层与嵌入层的结合
    @parametrize("bias", (True, False))
    def test_linear_with_embedding(self, bias):
        # 定义批处理大小
        batch_size = 384
        # 定义输入特征维度
        in_features = 196
        # 定义输出特征维度
        out_features = 384
        # 定义数据类型为bfloat16
        dtype = torch.bfloat16

        # 定义一个内部模型类M，继承自torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                # 定义一个线性层，输入特征数、输出特征数和是否使用偏置，并转换为指定数据类型dtype
                self.linear = torch.nn.Linear(in_features, out_features, bias).to(
                    dtype=dtype
                )
                # 定义一个嵌入层，嵌入大小为64，输出特征数为out_features
                self.emb = torch.nn.Embedding(64, out_features)

            # 前向传播方法，接受索引idx和输入数据x
            def forward(self, idx, x):
                # 返回嵌入层和线性层输出的加和结果
                return self.emb(idx) + self.linear(x)

        # 生成随机索引数据idx，并转换为指定数据类型dtype
        idx = torch.randint(0, 64, (batch_size,))
        # 生成随机输入数据x，并转换为指定数据类型dtype
        x = torch.randn(batch_size, in_features).to(dtype=dtype)
        # 创建一个M类的实例mod，传入偏置参数并设置为评估模式
        mod = M(bias=bias).eval()
        # 使用verify上下文管理器，获取绝对误差atol和相对误差rtol
        with verify(dtype) as (atol, rtol):
            # 调用common方法进行通用测试
            self.common(mod, (idx, x), atol=atol, rtol=rtol)
        # 断言autotune自动调优选择算法的计数为1
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        # 断言cpp_epilogue_fusion_counter计数为1
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)
# 应用装饰器，配置动态形状为 True，默认不假设为静态形状
@dynamo_config.patch({"dynamic_shapes": True, "assume_static_by_default": False})
# 定义一个测试基类，继承自 TestCase
class _DynamicShapesTestBase(TestCase):
    pass

# 定义一个测试类，继承自 _DynamicShapesTestBase
class TestSelectAlgorithmDynamicShapes(_DynamicShapesTestBase):
    # 将 check_model 赋值给 common
    common = check_model
    # 将 TestSelectAlgorithm 类中的 test_linear_static_shapes 方法赋值给 test_linear_dynamic_shapes
    test_linear_dynamic_shapes = TestSelectAlgorithm.test_linear_static_shapes
    # 将 TestSelectAlgorithm 类中的 test_linear_with_pointwise 方法赋值给 test_linear_with_pointwise_dynamic_shapes
    test_linear_with_pointwise_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_pointwise
    )
    # 将 TestSelectAlgorithm 类中的 test_linear_with_transpose 方法赋值给 test_linear_with_transpose_dynamic_shapes
    test_linear_with_transpose_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_transpose
    )
    # 将 TestSelectAlgorithm 类中的 test_linear_with_unary_binary 方法赋值给 test_linear_with_unary_binary_dynamic_shapes
    test_linear_with_unary_binary_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_unary_binary
    )
    # 将 TestSelectAlgorithm 类中的 test_linear_amx 方法赋值给 test_linear_amx_dynamic_shapes
    test_linear_amx_dynamic_shapes = TestSelectAlgorithm.test_linear_amx
    # 将 TestSelectAlgorithm 类中的 test_linear_with_embedding 方法赋值给 test_linear_with_embedding_dynamic_shapes
    test_linear_with_embedding_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_embedding
    )

# 在全局范围内实例化设备类型测试，针对 TestSelectAlgorithm 类，限定只在 CPU 上运行
instantiate_device_type_tests(TestSelectAlgorithm, globals(), only_for="cpu")
# 在全局范围内实例化设备类型测试，针对 TestSelectAlgorithmDynamicShapes 类，限定只在 CPU 上运行
instantiate_device_type_tests(
    TestSelectAlgorithmDynamicShapes, globals(), only_for="cpu"
)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch.testing._internal.inductor_utils 导入 HAS_CPU
    from torch.testing._internal.inductor_utils import HAS_CPU

    # 如果 HAS_CPU 为 True 并且不是 macOS 系统
    if HAS_CPU and not IS_MACOS:
        # 运行测试函数 run_tests()
        run_tests()
```