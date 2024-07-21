# `.\pytorch\test\inductor\test_cuda_cpp_wrapper.py`

```py
# 导入系统、单元测试和类型提示的命名元组
import sys
import unittest
from typing import NamedTuple

# 导入PyTorch相关模块
import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
from torch.testing._internal.common_utils import slowTest, TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import HAS_CUDA

# 尝试导入本地的测试模块，如果导入失败，则导入全局的测试模块
try:
    try:
        from . import (
            test_foreach,
            test_pattern_matcher,
            test_select_algorithm,
            test_torchinductor,
            test_torchinductor_dynamic_shapes,
        )
    except ImportError:
        import test_foreach
        import test_pattern_matcher
        import test_select_algorithm
        import test_torchinductor
        import test_torchinductor_dynamic_shapes
except unittest.SkipTest:
    # 如果是单元测试跳过，退出进程（仅在主程序中执行时）
    if __name__ == "__main__":
        sys.exit(0)
    raise

# 获取所需测试基类的列表
_desired_test_bases = get_desired_device_type_test_bases()

# 确定是否需要运行 CUDA 测试
RUN_CUDA = (
    HAS_CUDA
    and any(getattr(x, "device_type", "") == "cuda" for x in _desired_test_bases)
    and not TEST_WITH_ASAN
)

# 定义一个空的 CUDA 包装模板类
class CudaWrapperTemplate:
    pass

# 定义一个继承自 InductorTestCase 的 CUDA 包装测试类
class TestCudaWrapper(InductorTestCase):
    device = "cuda"

# 定义一个继承自 InductorTestCase 的动态形状 CUDA 包装测试类
class DynamicShapesCudaWrapperCudaTests(InductorTestCase):
    device = "cuda"

# 定义 CUDA 包装测试失败的字典，包含需要跳过的测试名称和相应的测试失败对象
test_failures_cuda_wrapper = {
    "test_mm_plus_mm2_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",), is_skip=True
    ),
}

# 如果配置为 ABI 兼容，则设置需要标记为失败的测试列表
if config.abi_compatible:
    xfail_list = [
        "test_profiler_mark_wrapper_call_cuda",
    ]
    # 将标记为失败的测试添加到测试失败字典中
    for test_name in xfail_list:
        test_failures_cuda_wrapper[test_name] = test_torchinductor.TestFailure(
            ("cuda_wrapper",), is_skip=False
        )
        test_failures_cuda_wrapper[
            f"{test_name}_dynamic_shapes"
        ] = test_torchinductor.TestFailure(("cuda_wrapper",), is_skip=False)
    
    # 设置需要跳过的测试列表
    skip_list = []
    # 将需要跳过的测试添加到测试失败字典中
    for test_name in skip_list:
        test_failures_cuda_wrapper[test_name] = test_torchinductor.TestFailure(
            ("cuda_wrapper",), is_skip=True
        )
        test_failures_cuda_wrapper[
            f"{test_name}_dynamic_shapes"
        ] = test_torchinductor.TestFailure(("cuda_wrapper",), is_skip=True)

# 定义一个生成测试用例的函数，返回一个测试函数对象
def make_test_case(
    name,
    device,
    tests,
    condition=True,
    slow=False,
    func_inputs=None,
    code_string_count=None,
):
    # 根据设备和名称生成测试函数的名称
    test_name = f"{name}_{device}" if device else name
    if code_string_count is None:
        code_string_count = {}

    # 获取测试函数对象
    func = getattr(tests, test_name)
    assert callable(func), "not a callable"
    # 如果需要，将测试函数标记为慢速测试
    func = slowTest(func) if slow else func

    @config.patch(cpp_wrapper=True, search_autotune_cache=False)
    # 定义一个函数 fn，该函数可能是一个测试函数
    def fn(self):
        # 调用 tests 类的 setUpClass 方法进行测试类的全局初始化设置
        tests.setUpClass()
        # 调用 tests 类的 setUp 方法进行测试初始化设置
        tests.setUp()
        try:
            # 使用 torch._C._PreserveDispatchKeyGuard() 进入保留调度键的上下文
            with torch._C._PreserveDispatchKeyGuard():
                # 将 DispatchKey 设置为 Dense，表示密集型操作
                torch._C._dispatch_tls_set_dispatch_key_included(
                    torch._C.DispatchKey.Dense, True
                )

                # 运行 func 函数，并获取其生成的 C++ 代码
                _, code = test_torchinductor.run_and_get_cpp_code(
                    func, *func_inputs if func_inputs else []
                )
                # 断言 CppWrapperCodeCache 是否在生成的代码中
                self.assertEqual("CppWrapperCodeCache" in code, True)
                # 断言生成的代码中各预设字符串出现的次数符合预期
                self.assertTrue(
                    all(
                        code.count(string) == code_string_count[string]
                        for string in code_string_count
                    )
                )
        finally:
            # 无论是否发生异常，调用 tests 类的 tearDown 方法进行测试清理
            tests.tearDown()
            # 调用 tests 类的 tearDownClass 方法进行测试类的全局清理

            tests.tearDownClass()

    # 将函数 fn 的名称设置为 test_name，可能是为了动态生成测试函数
    fn.__name__ = test_name
    import copy

    # 深拷贝 func 对象的字典属性到 fn 对象的字典中
    fn.__dict__ = copy.deepcopy(func.__dict__)
    
    # 如果满足某个条件（condition），则将函数 fn 设置为 CudaWrapperTemplate 的属性，可能是动态添加测试函数到类中
    if condition:
        setattr(
            CudaWrapperTemplate,
            test_name,
            fn,
        )
if RUN_CUDA:
    # 如果要求运行在 CUDA 上

    class BaseTest(NamedTuple):
        name: str
        device: str = "cuda"
        tests: InductorTestCase = test_torchinductor.GPUTests()
        # 定义一个测试基类，包含名称、设备（默认为 CUDA）、测试用例

    # 维护两个分离的测试列表，一个用于 CUDA，一个用于 C++（目前）
    ]:
        make_test_case(item.name, item.device, item.tests)
        # 对每个测试项目创建测试案例，包括名称、设备和测试用例

    from torch._inductor.utils import is_big_gpu

    if is_big_gpu(0):
        # 如果 GPU 设备是大型的

        for item in [
            BaseTest(
                "test_addmm",
                tests=test_select_algorithm.TestSelectAlgorithm(),
            ),
            BaseTest(
                "test_linear_relu",
                tests=test_select_algorithm.TestSelectAlgorithm(),
            ),
        ]:
            make_test_case(item.name, item.device, item.tests)
            # 对每个测试项目创建测试案例，包括名称、设备和测试用例

    test_torchinductor.copy_tests(
        CudaWrapperTemplate, TestCudaWrapper, "cuda_wrapper", test_failures_cuda_wrapper
    )
    # 复制 CUDA 包装器的测试案例到新的测试类中

    DynamicShapesCudaWrapperTemplate = (
        test_torchinductor_dynamic_shapes.make_dynamic_cls(CudaWrapperTemplate)
    )
    # 创建一个动态形状的 CUDA 包装器模板

    test_torchinductor.copy_tests(
        DynamicShapesCudaWrapperTemplate,
        DynamicShapesCudaWrapperCudaTests,
        "cuda_wrapper",
        test_failures_cuda_wrapper,
    )
    # 复制动态形状的 CUDA 包装器测试案例到新的测试类中

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CUDA:
        run_tests(needs="filelock")
        # 如果在主程序中且要求在 CUDA 上运行，则执行测试运行函数，需要文件锁
```