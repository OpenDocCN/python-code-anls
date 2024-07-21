# `.\pytorch\test\torch_np\test_dtype.py`

```
# Owner(s): ["module: dynamo"]

# 从 unittest 模块导入 expectedFailure 别名为 xfail
from unittest import expectedFailure as xfail

# 导入 numpy 库
import numpy

# 导入 torch._numpy 模块，别名为 tnp
import torch._numpy as tnp

# 从 torch.testing._internal.common_utils 导入多个函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

# 定义包含各种数据类型名称的列表
dtype_names = [
    "bool_",
    *[f"int{w}" for w in [8, 16, 32, 64]],
    *[f"uint{w}" for w in [8, 16, 32, 64]],
    *[f"float{w}" for w in [16, 32, 64]],
    *[f"complex{w}" for w in [64, 128]],
]

# 定义空的 numpy 数据类型参数列表
np_dtype_params = []

# 添加特定的子测试到 np_dtype_params 中
np_dtype_params = [
    subtest(("bool", "bool"), name="bool"),
    subtest(
        ("bool", numpy.dtype("bool")),
        name="numpy.dtype('bool')",
        decorators=[xfail],  # 原因："XXX: np.dtype() objects not supported"
    ),
]

# 遍历数据类型名称列表
for name in dtype_names:
    # 添加每个数据类型的子测试到 np_dtype_params 中
    np_dtype_params.append(subtest((name, name), name=repr(name)))

    # 添加 numpy 命名空间中的数据类型作为子测试到 np_dtype_params 中，并标记为预期失败
    np_dtype_params.append(
        subtest((name, getattr(numpy, name)), name=f"numpy.{name}", decorators=[xfail])
    )  # numpy 命名空间中的数据类型不支持

    # 添加 numpy 数据类型对象作为子测试到 np_dtype_params 中，并标记为预期失败
    np_dtype_params.append(
        subtest((name, numpy.dtype(name)), name=f"numpy.{name!r}", decorators=[xfail])
    )

# 定义 TestConvertDType 类，并使用 instantiate_parametrized_tests 装饰器实例化参数化测试
@instantiate_parametrized_tests
class TestConvertDType(TestCase):
    # 使用 parametrize 装饰器，参数化测试方法 test_convert_np_dtypes
    @parametrize("name, np_dtype", np_dtype_params)
    def test_convert_np_dtypes(self, name, np_dtype):
        # 使用 torch._numpy 模块的 dtype 方法创建 tnp_dtype
        tnp_dtype = tnp.dtype(np_dtype)
        
        # 如果数据类型名称是 "bool_"，断言 tnp_dtype 等于 tnp.bool_
        if name == "bool_":
            assert tnp_dtype == tnp.bool_
        # 否则，如果 tnp_dtype 的名称是 "bool_"，断言 name 以 "bool" 开头
        elif tnp_dtype.name == "bool_":
            assert name.startswith("bool")
        # 否则，断言 tnp_dtype 的名称等于 name
        else:
            assert tnp_dtype.name == name

# 如果当前脚本被直接执行，则运行测试
if __name__ == "__main__":
    run_tests()
```