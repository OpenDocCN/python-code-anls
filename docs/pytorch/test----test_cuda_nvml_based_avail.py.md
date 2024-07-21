# `.\pytorch\test\test_cuda_nvml_based_avail.py`

```py
# Owner(s): ["module: cuda"]

# 引入必要的模块和库
import multiprocessing
import os
import sys
import unittest
from unittest.mock import patch

import torch

# NOTE: Each of the tests in this module need to be run in a brand new process to ensure CUDA is uninitialized
# prior to test initiation.
# 通过修改环境变量，在运行测试之前确保CUDA未初始化
with patch.dict(os.environ, {"PYTORCH_NVML_BASED_CUDA_CHECK": "1"}):
    # Before executing the desired tests, we need to disable CUDA initialization and fork_handler additions that would
    # otherwise be triggered by the `torch.testing._internal.common_utils` module import
    # 导入 `torch.testing._internal.common_utils` 模块之前，需要禁用CUDA初始化和fork_handler添加
    from torch.testing._internal.common_utils import (
        instantiate_parametrized_tests,
        IS_JETSON,
        IS_WINDOWS,
        NoTest,
        parametrize,
        run_tests,
        TestCase,
    )

    # NOTE: Because `remove_device_and_dtype_suffixes` initializes CUDA context (triggered via the import of
    # `torch.testing._internal.common_device_type` which imports `torch.testing._internal.common_cuda`) we need
    # to bypass that method here which should be irrelevant to the parameterized tests in this module.
    # 由于 `remove_device_and_dtype_suffixes` 方法初始化CUDA上下文（通过导入 `torch.testing._internal.common_device_type`，
    # 它导入 `torch.testing._internal.common_cuda`），我们在这里需要绕过该方法，因为它与本模块中的参数化测试无关。
    torch.testing._internal.common_utils.remove_device_and_dtype_suffixes = lambda x: x

    # Check if CUDA is available
    TEST_CUDA = torch.cuda.is_available()
    if not TEST_CUDA:
        # If CUDA is not available, skip the tests and print a message
        print("CUDA not available, skipping tests", file=sys.stderr)
        # Redefine TestCase to NoTest in case CUDA is not available
        TestCase = NoTest  # type: ignore[misc, assignment] # noqa: F811


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestExtendedCUDAIsAvail(TestCase):
    # Reminder message for subprocess test execution
    SUBPROCESS_REMINDER_MSG = (
        "\n REMINDER: Tests defined in test_cuda_nvml_based_avail.py must be run in a process "
        "where there CUDA Driver API has not been initialized. Before further debugging, ensure you are either using "
        "run_test.py or have added --subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        super().setUp()
        # Clear the lru_cache on `_cached_device_count` before the test
        torch.cuda._cached_device_count = (
            None  # clear the lru_cache on this method before our test
        )

    @staticmethod
    def in_bad_fork_test() -> bool:
        # Check if CUDA is available and if it's in a bad fork state
        _ = torch.cuda.is_available()
        return torch.cuda._is_in_bad_fork()

    # These tests validate the behavior and activation of the weaker, NVML-based, user-requested
    # `torch.cuda.is_available()` assessment. The NVML-based assessment should be attempted when
    # `PYTORCH_NVML_BASED_CUDA_CHECK` is set to 1, reverting to the default CUDA Runtime API check otherwise.
    # If the NVML-based assessment is attempted but fails, the CUDA Runtime API check should be executed
    # Skip tests on Windows due to fork requirement
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    @parametrize("nvml_avail", [True, False])
    @parametrize("avoid_init", ["1", "0", None])
    # 测试CUDA是否可用的方法，接受三个参数：self, avoid_init, nvml_avail
    def test_cuda_is_available(self, avoid_init, nvml_avail):
        # 如果是Jetson平台并且nvml可用并且避免初始化参数为"1"，则跳过测试
        if IS_JETSON and nvml_avail and avoid_init == "1":
            self.skipTest("Not working for Jetson")
        
        # 根据避免初始化参数设置环境变量字典，用于修改环境变量
        patch_env = {"PYTORCH_NVML_BASED_CUDA_CHECK": avoid_init} if avoid_init else {}
        
        # 使用patch来修改os.environ，将patch_env中的环境变量设置进去
        with patch.dict(os.environ, **patch_env):
            # 如果nvml可用，则调用torch.cuda.is_available()函数
            if nvml_avail:
                _ = torch.cuda.is_available()
            else:
                # 否则使用patch.object模拟torch.cuda._device_count_nvml的返回值为-1
                with patch.object(torch.cuda, "_device_count_nvml", return_value=-1):
                    _ = torch.cuda.is_available()
            
            # 使用"fork"上下文获取一个进程池，设置最大进程数为1
            with multiprocessing.get_context("fork").Pool(1) as pool:
                # 在进程池中应用TestExtendedCUDAIsAvail.in_bad_fork_test方法
                in_bad_fork = pool.apply(TestExtendedCUDAIsAvail.in_bad_fork_test)
            
            # 如果环境变量PYTORCH_NVML_BASED_CUDA_CHECK为"1"并且nvml可用，则断言in_bad_fork应为False
            if os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1" and nvml_avail:
                self.assertFalse(
                    in_bad_fork, TestExtendedCUDAIsAvail.SUBPROCESS_REMINDER_MSG
                )
            else:
                # 否则，断言in_bad_fork应为True
                assert in_bad_fork
# 标记测试为 DynamoStrictTest 的测试类
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestVisibleDeviceParses(TestCase):
    # 测试环境变量解析函数
    def test_env_var_parsing(self):
        # 定义内部函数 _parse_visible_devices，导入 torch.cuda 中的 _parse_visible_devices
        def _parse_visible_devices(val):
            from torch.cuda import _parse_visible_devices as _pvd

            # 使用 patch.dict 修改 os.environ 中的 "CUDA_VISIBLE_DEVICES" 变量，并清空之前的设置
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": val}, clear=True):
                # 调用 torch.cuda._parse_visible_devices 函数解析 CUDA 可见设备
                return _pvd()

        # 检查 _parse_visible_devices 函数对 "1gpu2,2ampere" 的解析结果是否为 [1, 2]
        self.assertEqual(_parse_visible_devices("1gpu2,2ampere"), [1, 2])
        # 检查 _parse_visible_devices 函数对 "0, 1, 2, -1, 3" 的解析结果是否为 [0, 1, 2]
        self.assertEqual(_parse_visible_devices("0, 1, 2, -1, 3"), [0, 1, 2])
        # 检查 _parse_visible_devices 函数对 "0, 1, 2, 1" 的解析结果是否为空列表 []
        self.assertEqual(_parse_visible_devices("0, 1, 2, 1"), [])
        # 检查 _parse_visible_devices 函数对 "2, +3, -0, 5" 的解析结果是否为 [2, 3, 0, 5]
        self.assertEqual(_parse_visible_devices("2, +3, -0, 5"), [2, 3, 0, 5])
        # 检查 _parse_visible_devices 函数对 "one,two,3,4" 的解析结果是否为空列表 []
        self.assertEqual(_parse_visible_devices("one,two,3,4"), [])
        # 检查 _parse_visible_devices 函数对 "4,3,two,one" 的解析结果是否为 [4, 3]
        self.assertEqual(_parse_visible_devices("4,3,two,one"), [4, 3])
        # 检查 _parse_visible_devices 函数对 "GPU-9e8d35e3" 的解析结果是否为 ["GPU-9e8d35e3"]
        self.assertEqual(_parse_visible_devices("GPU-9e8d35e3"), ["GPU-9e8d35e3"])
        # 检查 _parse_visible_devices 函数对 "GPU-123, 2" 的解析结果是否为 ["GPU-123"]
        self.assertEqual(_parse_visible_devices("GPU-123, 2"), ["GPU-123"])
        # 检查 _parse_visible_devices 函数对 "MIG-89c850dc" 的解析结果是否为 ["MIG-89c850dc"]
        self.assertEqual(_parse_visible_devices("MIG-89c850dc"), ["MIG-89c850dc"])

    # 测试部分 UUID 解析函数
    def test_partial_uuid_resolver(self):
        # 导入 _transform_uuid_to_ordinals 函数
        from torch.cuda import _transform_uuid_to_ordinals

        # 定义一组 UUID 列表
        uuids = [
            "GPU-9942190a-aa31-4ff1-4aa9-c388d80f85f1",
            "GPU-9e8d35e3-a134-0fdd-0e01-23811fdbd293",
            "GPU-e429a63e-c61c-4795-b757-5132caeb8e70",
            "GPU-eee1dfbc-0a0f-6ad8-5ff6-dc942a8b9d98",
            "GPU-bbcd6503-5150-4e92-c266-97cc4390d04e",
            "GPU-472ea263-58d7-410d-cc82-f7fdece5bd28",
            "GPU-e56257c4-947f-6a5b-7ec9-0f45567ccf4e",
            "GPU-1c20e77d-1c1a-d9ed-fe37-18b8466a78ad",
        ]
        
        # 检查 _transform_uuid_to_ordinals 函数对 ["GPU-9e8d35e3"] 的解析结果是否为 [1]
        self.assertEqual(_transform_uuid_to_ordinals(["GPU-9e8d35e3"], uuids), [1])
        # 检查 _transform_uuid_to_ordinals 函数对 ["GPU-e4", "GPU-9e8d35e3"] 的解析结果是否为 [2, 1]
        self.assertEqual(_transform_uuid_to_ordinals(["GPU-e4", "GPU-9e8d35e3"], uuids), [2, 1])
        # 检查 _transform_uuid_to_ordinals 函数对 ["GPU-9e8d35e3", "GPU-1", "GPU-47"] 的解析结果是否为 [1, 7, 5]
        self.assertEqual(_transform_uuid_to_ordinals("GPU-9e8d35e3,GPU-1,GPU-47".split(","), uuids), [1, 7, 5])
        # 检查 _transform_uuid_to_ordinals 函数对 ["GPU-123", "GPU-9e8d35e3"] 的解析结果是否为空列表 []
        self.assertEqual(_transform_uuid_to_ordinals(["GPU-123", "GPU-9e8d35e3"], uuids), [])
        # 检查 _transform_uuid_to_ordinals 函数对 ["GPU-9e8d35e3", "GPU-123", "GPU-47"] 的解析结果是否为 [1]
        self.assertEqual(_transform_uuid_to_ordinals(["GPU-9e8d35e3", "GPU-123", "GPU-47"], uuids), [1])
        # 检查 _transform_uuid_to_ordinals 函数对 ["GPU-9e8d35e3", "GPU-e", "GPU-47"] 的解析结果是否为 [1]
        self.assertEqual(_transform_uuid_to_ordinals(["GPU-9e8d35e3", "GPU-e", "GPU-47"], uuids), [1])
        # 检查 _transform_uuid_to_ordinals 函数对 ["GPU-9e8d35e3", "GPU-47", "GPU-9e8"] 的解析结果是否为空列表 []
        self.assertEqual(_transform_uuid_to_ordinals(["GPU-9e8d35e3", "GPU-47", "GPU-9e8"], uuids), [])
    # 定义一个测试方法，用于测试 CUDA 可见设备的序数解析功能
    def test_ordinal_parse_visible_devices(self):
        
        # 定义一个内部函数，用于模拟通过 NVML 获取设备数量
        def _device_count_nvml(val):
            # 导入 torch.cuda 模块中的 _device_count_nvml 函数，并命名为 _dc
            from torch.cuda import _device_count_nvml as _dc
            
            # 使用 patch.dict 临时修改环境变量 CUDA_VISIBLE_DEVICES 的值为 val，并清除之前的设置
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": val}, clear=True):
                # 调用 _dc 函数获取设备数量并返回
                return _dc()
        
        # 使用 patch.object 临时替换 torch.cuda 模块中的 _raw_device_count_nvml 函数，固定返回值为 2
        with patch.object(torch.cuda, "_raw_device_count_nvml", return_value=2):
            # 断言调用 _device_count_nvml 函数并传入参数 "1, 0" 后返回值为 2
            self.assertEqual(_device_count_nvml("1, 0"), 2)
            # 断言调用 _device_count_nvml 函数并传入参数 "1, 5, 0" 后返回值为 1，表明超出范围的序数会中止解析
            self.assertEqual(_device_count_nvml("1, 5, 0"), 1)
# 使用给定的参数化测试类实例化参数化测试
instantiate_parametrized_tests(TestExtendedCUDAIsAvail)

# 检查当前脚本是否作为主程序运行，如果是则执行测试运行
if __name__ == "__main__":
    run_tests()
```