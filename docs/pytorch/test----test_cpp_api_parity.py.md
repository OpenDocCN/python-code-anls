# `.\pytorch\test\test_cpp_api_parity.py`

```
# Owner(s): ["module: cpp"]

# 导入标准库模块 os
import os

# 从 cpp_api_parity 模块中导入需要的函数和类
from cpp_api_parity import (
    functional_impl_check,
    module_impl_check,
    sample_functional,
    sample_module,
)

# 从 cpp_api_parity.parity_table_parser 模块中导入 parse_parity_tracker_table 函数
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table

# 从 cpp_api_parity.utils 模块中导入 is_torch_nn_functional_test 函数
from cpp_api_parity.utils import is_torch_nn_functional_test

# 导入 PyTorch 库
import torch
import torch.testing._internal.common_nn as common_nn
import torch.testing._internal.common_utils as common

# NOTE: 如果需要打印所有 C++ 测试的源代码，可以将 PRINT_CPP_SOURCE 设为 True（用于调试目的）
PRINT_CPP_SOURCE = False

# 支持的设备列表
devices = ["cpu", "cuda"]

# 定义 PARITY_TABLE_PATH 为 "cpp_api_parity/parity-tracker.md" 的完整路径
PARITY_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "cpp_api_parity", "parity-tracker.md"
)

# 解析 PARITY_TABLE_PATH 对应的表格文件，返回 parity_table
parity_table = parse_parity_tracker_table(PARITY_TABLE_PATH)


# 使用 torch.testing._internal.common_utils.markDynamoStrictTest 装饰器标记测试类
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppApiParity(common.TestCase):
    # 初始化空字典，用于存储模块测试的参数映射
    module_test_params_map = {}
    # 初始化空字典，用于存储函数测试的参数映射
    functional_test_params_map = {}

# 预期的测试参数字典列表
expected_test_params_dicts = []

# 如果不是 ARM64 架构
if not common.IS_ARM64:
    # 遍历多个测试参数字典和测试实例类的元组列表
    for test_params_dicts, test_instance_class in [
        (sample_module.module_tests, common_nn.NewModuleTest),
        (sample_functional.functional_tests, common_nn.NewModuleTest),
        (common_nn.module_tests, common_nn.NewModuleTest),
        (common_nn.new_module_tests, common_nn.NewModuleTest),
        (common_nn.criterion_tests, common_nn.CriterionTest),
    ]:
        # 遍历每个测试参数字典
        for test_params_dict in test_params_dicts:
            # 如果测试参数字典中的 "test_cpp_api_parity" 键值为 True
            if test_params_dict.get("test_cpp_api_parity", True):
                # 如果测试参数字典属于函数测试，则将测试写入函数测试类中
                if is_torch_nn_functional_test(test_params_dict):
                    functional_impl_check.write_test_to_test_class(
                        TestCppApiParity,
                        test_params_dict,
                        test_instance_class,
                        parity_table,
                        devices,
                    )
                # 否则将测试写入模块测试类中
                else:
                    module_impl_check.write_test_to_test_class(
                        TestCppApiParity,
                        test_params_dict,
                        test_instance_class,
                        parity_table,
                        devices,
                    )
                # 将当前测试参数字典添加到期望的测试参数字典列表中
                expected_test_params_dicts.append(test_params_dict)

    # 断言所有 NN 模块/函数测试字典是否都出现在 Parity 测试中
    assert len(
        [name for name in TestCppApiParity.__dict__ if "test_torch_nn_" in name]
    ) == len(expected_test_params_dicts) * len(devices)

    # 断言是否存在自动生成的测试用例，检查 "SampleModule" 和 "sample_functional"
    assert (
        len([name for name in TestCppApiParity.__dict__ if "SampleModule" in name]) == 4
    )
    assert (
        len([name for name in TestCppApiParity.__dict__ if "sample_functional" in name])
        == 4
    )
    # 使用 module_impl_check 对象的 build_cpp_tests 方法构建 C++ 测试
    # 测试类为 TestCppApiParity，是否打印 C++ 源代码取决于 PRINT_CPP_SOURCE 变量
    module_impl_check.build_cpp_tests(
        TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE
    )
    # 使用 functional_impl_check 对象的 build_cpp_tests 方法构建 C++ 测试
    # 测试类为 TestCppApiParity，是否打印 C++ 源代码取决于 PRINT_CPP_SOURCE 变量
    functional_impl_check.build_cpp_tests(
        TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE
    )
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 设置 TestCase 类的默认数据类型检查启用标志为 True
    common.TestCase._default_dtype_check_enabled = True
    # 运行测试套件中的所有测试用例
    common.run_tests()
```