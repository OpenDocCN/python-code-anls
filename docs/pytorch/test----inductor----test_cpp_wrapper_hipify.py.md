# `.\pytorch\test\inductor\test_cpp_wrapper_hipify.py`

```
# Owner(s): ["module: inductor"]
# 引入 PyTorch 库
import torch

# 从 torch._inductor.codegen.aoti_hipify_utils 模块导入 maybe_hipify_code_wrapper 函数
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
# 从 torch._inductor.codegen.codegen_device_driver 模块导入 cuda_kernel_driver 函数
from torch._inductor.codegen.codegen_device_driver import cuda_kernel_driver
# 从 torch._inductor.test_case 模块导入 run_tests 和 TestCase 类
from torch._inductor.test_case import run_tests, TestCase

# 定义测试代码列表，用于 CUDA 平台
TEST_CODES = [
    "CUresult code = EXPR;",
    "CUfunction kernel = nullptr;",
    "static CUfunction kernel = nullptr;",
    "CUdeviceptr var = reinterpret_cast<CUdeviceptr>(arg.data_ptr());",
    "at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal());",
    # Hipification should be idempotent, hipifying should be a no-op for already hipified files
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
]

# 定义测试代码列表，用于 HIP 平台
HIP_CODES = [
    "hipError_t code = EXPR;",
    "hipFunction_t kernel = nullptr;",
    "static hipFunction_t kernel = nullptr;",
    "hipDeviceptr_t var = reinterpret_cast<hipDeviceptr_t>(arg.data_ptr());",
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
]

# 定义测试类 TestCppWrapperHipify，继承自 TestCase 类
class TestCppWrapperHipify(TestCase):
    
    # 测试函数：测试基本声明的 HIP 化
    def test_hipify_basic_declaration(self) -> None:
        # 确保测试代码列表和 HIP 代码列表长度相同
        assert len(TEST_CODES) == len(HIP_CODES)
        # 遍历测试代码列表
        for i in range(len(TEST_CODES)):
            # 对当前测试代码进行 HIP 化处理
            result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            # 获取期望的 HIP 代码
            expected = HIP_CODES[i]
            # 断言处理后的结果与期望的 HIP 代码一致
            self.assertEqual(result, expected)

    # 测试函数：测试跨平台 HIP 化
    def test_hipify_cross_platform(self) -> None:
        # 确保测试代码列表和 HIP 代码列表长度相同
        assert len(TEST_CODES) == len(HIP_CODES)
        # 遍历测试代码列表
        for i in range(len(TEST_CODES)):
            # 对当前测试代码进行 HIP 化处理，要求为跨平台模式
            hip_result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            # 对当前测试代码进行 HIP 化处理，不要求为跨平台模式
            result = maybe_hipify_code_wrapper(TEST_CODES[i])
            # 如果当前 PyTorch 版本支持 HIP
            if torch.version.hip is not None:
                # 断言处理后的结果与 HIP 化处理后的结果一致
                self.assertEqual(result, hip_result)
            else:
                # 断言处理后的结果与原始测试代码一致（不进行 HIP 化）
                self.assertEqual(result, TEST_CODES[i])

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```