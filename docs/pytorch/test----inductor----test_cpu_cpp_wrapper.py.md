# `.\pytorch\test\inductor\test_cpu_cpp_wrapper.py`

```
# Owner(s): ["oncall: cpu inductor"]

# 导入系统相关模块
import sys
# 导入单元测试框架
import unittest
# 导入命名元组支持
from typing import NamedTuple

# 导入PyTorch主要模块
import torch
# 导入PyTorch内部配置模块
from torch._inductor import config
# 导入PyTorch自定义的测试用例基类
from torch._inductor.test_case import TestCase as InductorTestCase
# 导入设备类型测试相关函数
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
# 导入通用测试工具函数和变量
from torch.testing._internal.common_utils import IS_MACOS, slowTest, TEST_WITH_ROCM
# 导入PyTorch自定义的感应器测试工具函数
from torch.testing._internal.inductor_utils import HAS_CPU

# 尝试导入多个测试模块，可能会出现ImportError
try:
    try:
        from . import (
            test_cpu_repro,
            test_mkldnn_pattern_matcher,
            test_torchinductor,
            test_torchinductor_dynamic_shapes,
        )
    except ImportError:
        import test_cpu_repro
        import test_mkldnn_pattern_matcher
        import test_torchinductor
        import test_torchinductor_dynamic_shapes
except unittest.SkipTest:
    # 如果unittest.SkipTest异常被抛出，程序将退出
    if __name__ == "__main__":
        sys.exit(0)
    raise

# 获取所需的设备类型测试基类
_desired_test_bases = get_desired_device_type_test_bases()
# 根据条件判断是否运行CPU相关测试
RUN_CPU = (
    HAS_CPU
    and any(getattr(x, "device_type", "") == "cpu" for x in _desired_test_bases)
    and not IS_MACOS
)

# 定义一个空的类作为C++包装器模板
class CppWrapperTemplate:
    pass

# 定义一个继承自InductorTestCase的测试类，用于测试C++包装器
class TestCppWrapper(InductorTestCase):
    device = "cpu"

# 定义一个继承自InductorTestCase的测试类，用于测试动态形状下的C++包装器（CPU）
class DynamicShapesCppWrapperCpuTests(InductorTestCase):
    device = "cpu"

# 定义一个包含C++包装器测试失败信息的字典
test_failures_cpp_wrapper = {
    # conv2d在动态形状下将回退；回退路径尚不支持
    "test_conv2d_unary_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
    "test_conv2d_binary_inplace_fusion_failed_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
    "test_conv2d_binary_inplace_fusion_pass_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
    # aten._native_multi_head_attention.default在动态形状下尚不支持
    "test_multihead_attention_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
}

# 如果在ROCm环境下，更新测试失败字典
if TEST_WITH_ROCM:
    test_failures_cpp_wrapper.update(
        {
            "test_linear_packed": test_torchinductor.TestFailure(
                ("cpp_wrapper"), is_skip=True
            ),
            "test_linear_packed_dynamic_shapes": test_torchinductor.TestFailure(
                ("cpp_wrapper"), is_skip=True
            ),
        }
    )

# 如果ABI兼容，则执行以下代码
if config.abi_compatible:
    # 定义一个列表，包含所有需要标记为预期失败的测试名称
    xfail_list = [
        "test_conv2d_binary_inplace_fusion_failed_cpu",
        "test_conv2d_binary_inplace_fusion_pass_cpu",
        "test_dynamic_qlinear_cpu",
        "test_dynamic_qlinear_qat_cpu",
        "test_lstm_packed_change_input_sizes_cpu",
        "test_profiler_mark_wrapper_call_cpu",
        "test_qconv2d_add_cpu",
        "test_qconv2d_add_relu_cpu",
        "test_qconv2d_cpu",
        "test_qconv2d_dequant_promotion_cpu",
        "test_qconv2d_maxpool2d_linear_dynamic_cpu",
        "test_qconv2d_relu_cpu",
        "test_qlinear_cpu",
        "test_qlinear_add_cpu",
        "test_qlinear_add_relu_cpu",
        "test_qlinear_dequant_promotion_cpu",
        "test_qlinear_gelu_cpu",
        "test_qlinear_relu_cpu",
    ]
    
    # 遍历预期失败的测试列表，将每个测试标记为测试失败，但不跳过
    for test_name in xfail_list:
        test_failures_cpp_wrapper[test_name] = test_torchinductor.TestFailure(
            ("cpp_wrapper",), is_skip=False
        )
        # 同时为每个测试添加一个动态形状的标记，同样标记为测试失败
        test_failures_cpp_wrapper[
            f"{test_name}_dynamic_shapes"
        ] = test_torchinductor.TestFailure(("cpp_wrapper",), is_skip=False)
    
    # 定义一个列表，包含所有需要跳过的测试名称
    skip_list = [
        "test_multihead_attention_cpu",
    ]
    
    # 遍历需要跳过的测试列表，将每个测试标记为跳过
    for test_name in skip_list:
        test_failures_cpp_wrapper[test_name] = test_torchinductor.TestFailure(
            ("cpp_wrapper",), is_skip=True
        )
        # 同时为每个测试添加一个动态形状的标记，同样标记为跳过
        test_failures_cpp_wrapper[
            f"{test_name}_dynamic_shapes"
        ] = test_torchinductor.TestFailure(("cpp_wrapper",), is_skip=True)
# 定义一个函数，用于创建测试用例
def make_test_case(
    name,
    device,
    tests,
    condition=True,
    slow=False,
    func_inputs=None,
    code_string_count=None,
):
    # 根据设备是否存在，决定测试用例的名称
    test_name = f"{name}_{device}" if device else name
    # 如果未提供代码字符串计数器，初始化为空字典
    if code_string_count is None:
        code_string_count = {}

    # 获取测试对象中的测试函数
    func = getattr(tests, test_name)
    # 确保获取到的是一个可调用的函数
    assert callable(func), "not a callable"
    # 如果需要，将函数标记为慢速测试
    func = slowTest(func) if slow else func

    # 使用装饰器配置，启用 C++ 封装，禁用自动调谐缓存
    @config.patch(cpp_wrapper=True, search_autotune_cache=False)
    def fn(self):
        # 设置测试类的环境
        tests.setUpClass()
        tests.setUp()
        try:
            # 使用保留调度键的上下文管理器，设置密集调度键
            with torch._C._PreserveDispatchKeyGuard():
                torch._C._dispatch_tls_set_dispatch_key_included(
                    torch._C.DispatchKey.Dense, True
                )

                # 运行测试函数并获取其生成的 C++ 代码
                _, code = test_torchinductor.run_and_get_cpp_code(
                    func, *func_inputs if func_inputs else []
                )
                # 断言生成的代码中包含 "CppWrapperCodeCache"
                self.assertEqual("CppWrapperCodeCache" in code, True)
                # 断言生成的代码中各字符串出现次数与期望相符
                self.assertTrue(
                    all(
                        code.count(string) == code_string_count[string]
                        for string in code_string_count
                    )
                )
        finally:
            # 清理测试类的环境
            tests.tearDown()
            tests.tearDownClass()

    # 设置生成的函数的名称为测试用例的名称
    fn.__name__ = test_name
    import copy

    # 深度复制原始测试函数的字典属性到生成的函数
    fn.__dict__ = copy.deepcopy(func.__dict__)
    # 如果满足条件，则将生成的函数设置为类的属性
    if condition:
        setattr(
            CppWrapperTemplate,
            test_name,
            fn,
        )


# 如果需要在 CPU 上运行测试
if RUN_CPU:

    # 定义一个命名元组作为基础测试类
    class BaseTest(NamedTuple):
        name: str
        device: str = "cpu"
        tests: InductorTestCase = test_torchinductor.CpuTests()
        condition: bool = True
        slow: bool = False
        func_inputs: list = None
        code_string_count: dict = {}

    # 遍历每个测试项目，并创建相应的测试用例
    ]:
        make_test_case(
            item.name,
            item.device,
            item.tests,
            item.condition,
            item.slow,
            item.func_inputs,
            item.code_string_count,
        )

    # 复制 CppWrapperTemplate 类中的测试到 TestCppWrapper 类中
    test_torchinductor.copy_tests(
        CppWrapperTemplate,
        TestCppWrapper,
        "cpp_wrapper",
        test_failures_cpp_wrapper,
    )

    # 创建动态形状的 C++ 封装模板
    DynamicShapesCppWrapperTemplate = (
        test_torchinductor_dynamic_shapes.make_dynamic_cls(CppWrapperTemplate)
    )

    # 复制动态形状的 C++ 封装模板中的测试到相应的测试类中
    test_torchinductor.copy_tests(
        DynamicShapesCppWrapperTemplate,
        DynamicShapesCppWrapperCpuTests,
        "cpp_wrapper",
        test_failures_cpp_wrapper,
        xfail_prop="_expected_failure_dynamic_wrapper",
    )


# 如果脚本被直接执行
if __name__ == "__main__":
    # 导入并运行 torch._inductor.test_case 模块中的测试
    from torch._inductor.test_case import run_tests

    # 如果需要在 CPU 上运行测试，则运行需要 "filelock" 的测试
    if RUN_CPU:
        run_tests(needs="filelock")
```