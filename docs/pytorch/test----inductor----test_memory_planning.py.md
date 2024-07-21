# `.\pytorch\test\inductor\test_memory_planning.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的模块和函数
import sys
import unittest

# 导入测试相关的模块和函数
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CUDA

# 如果在 Windows 上且是在 CI 环境中，则输出错误信息并退出
if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_memory_planning yet\n"
    )
    # 如果作为主模块运行，则退出程序
    if __name__ == "__main__":
        sys.exit(0)
    # 否则抛出跳过测试的异常
    raise unittest.SkipTest("requires sympy/functorch/filelock")  # noqa: F821

# 导入必要的 Torch 模块
import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.export import Dim
from torch.utils._triton import has_triton

# 如果没有 Triton 支持，则跳过测试
@unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
# 启用内存规划配置
@config.patch(memory_planning=True)
class TestMemoryPlanning(TestCase):
    # 生成测试用例的方法
    def _generate(self, *, device):
        """
        Generate a simple test case that has multiple simultaneously-live intermediate tensors.
        """
        # 定义一个简单的模型
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                # 计算中间结果
                t0 = x.matmul(y)
                t1 = x.matmul(z)
                t0 = x.transpose(0, 1).matmul(t1)
                t1 = x.matmul(t0)
                return t0.sum() + t1.sum()

        # 生成随机数据张量并返回
        x = torch.randn((3, 2), device=device)
        y = torch.randn((2, 4), device=device)
        z = torch.randn((2, 3), device=device)
        return (Foo(), (x, y, z))

    # 测试 Python 封装的方法
    def test_python_wrapper(self):
        # 生成测试用例
        f, args = self._generate(device="cuda")
        # 编译模型
        compiled = torch.compile(f, dynamic=True)
        # 运行并获取 C++ 代码及结果
        result, code = run_and_get_cpp_code(compiled, *args)

        # 使用 FileCheck 验证生成的 C++ 代码
        FileCheck().check(
            "pool1 = empty_strided_cuda(((4*s0*s1) + (align(4*(s0*s0))), ), (1, )"
        ).check_next(
            "buf0 = alloc_from_pool(pool1, 0, torch.float32, (s0, s0), (s0, 1))"
        ).check(
            "buf1 = alloc_from_pool(pool1, align(4*(s0*s0)),"
        ).run(
            code
        )
        # 验证 Python 和 C++ 封装后的结果是否一致
        self.assertTrue(same(f(*args), result))

    # 测试 C++ 封装的方法
    def test_cpp_wrapper(self):
        # 生成测试用例
        f, args = self._generate(device="cuda")
        # 编译模型
        compiled = torch.compile(f, dynamic=True)
        # 使用 C++ 封装运行并获取结果及代码
        with config.patch("cpp_wrapper", True):
            result, code = run_and_get_cpp_code(compiled, *args)

        # 使用 FileCheck 验证生成的 C++ 代码
        FileCheck().check(
            "pool1 = at::detail::empty_strided_cuda({(4L*s0*s1) + (align(4L*(static_cast<long>(s0*s0)))), }, {1L, }"
        ).check_next(
            "auto buf0 = alloc_from_pool(pool1, 0, at::kFloat, {s0, s0}, {s0, 1L});"
        ).check(
            "auto buf1 = alloc_from_pool(pool1, align(4L*(static_cast<long>(s0*s0))),"
        ).run(
            code
        )
        # 验证 Python 和 C++ 封装后的结果是否一致
        self.assertTrue(same(f(*args), result))

    # 在 ROCm 平台上跳过该测试
    @skipIfRocm(msg="test_aot_inductor doesn't work on ROCm")
    # 定义测试函数，用于测试ABI兼容性
    def test_abi_compatible(self):
        # 从test_aot_inductor模块中导入AOTIRunnerUtil
        from test_aot_inductor import AOTIRunnerUtil

        # 调用_generate方法生成测试函数和参数
        f, args = self._generate(device="cuda")
        
        # 创建一个维度对象Dim，命名为dim0_x，指定其范围为1到2048
        dim0_x = Dim("dim0_x", min=1, max=2048)
        
        # 定义动态形状参数dynamic_shapes
        dynamic_shapes = ({0: dim0_x}, None, None)
        
        # 使用config.patch设置"abi_compatible"为True
        with config.patch("abi_compatible", True):
            # 运行并获取Cpp代码及其结果
            result, code = run_and_get_cpp_code(
                # 使用AOTIRunnerUtil的run方法在cuda设备上运行函数f，传入参数args和动态形状dynamic_shapes
                lambda: AOTIRunnerUtil.run(
                    "cuda", f, args, dynamic_shapes=dynamic_shapes
                )
            )

        # 创建FileCheck对象，并逐一检查Cpp代码中的字符串
        FileCheck().check(
            "int64_t int_array_2[] = {24L + (align(12L*s0)), };"
        ).check_next("int64_t int_array_3[] = {1L, };").check_next(
            "AtenTensorHandle pool1_handle;"
        ).check_next(
            "aoti_torch_empty_strided(1, int_array_2, int_array_3,"
        ).check_next(
            "RAIIAtenTensorHandle pool1(pool1_handle);"
        ).check_next(
            "int64_t int_array_4[] = {s0, 3L};"
        ).check_next(
            "int64_t int_array_5[] = {3L, 1L};"
        ).check_next(
            "AtenTensorHandle tmp_tensor_handle_1;"
        ).check_next(
            "aoti_torch__alloc_from_pool(pool1, 0"
        ).run(
            code  # 在Cpp代码中执行检查
        )
        
        # 使用self.assertTrue验证函数f在给定参数args下的运行结果与预期的result相同
        self.assertTrue(same(f(*args), result))
# 如果脚本作为主程序执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 检查是否有 CUDA 支持的环境变量或配置
    if HAS_CUDA:
        # 如果有 CUDA 支持，则运行测试函数
        run_tests()
```