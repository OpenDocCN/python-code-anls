# `.\pytorch\test\dynamo\test_inline_inbuilt_nn_modules.py`

```py
# Owner(s): ["module: dynamo"]
# 导入单元测试模块
import unittest

# 导入 torch._dynamo.config 模块
from torch._dynamo import config
# 导入 torch._dynamo.testing 模块中的 make_test_cls_with_patches 函数
from torch._dynamo.testing import make_test_cls_with_patches

try:
    # 尝试从当前目录导入以下测试模块
    from . import (
        test_aot_autograd,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        # test_repros,
    )
except ImportError:
    # 若导入失败，则分别导入这些测试模块
    import test_aot_autograd
    import test_functions
    import test_higher_order_ops
    import test_misc
    import test_modules

# 用于存储生成的测试类字典
test_classes = {}

# 创建内联内置 NN 模块测试类
def make_inline_inbuilt_nn_modules_cls(cls):
    suffix = "_inline_inbuilt_nn_modules"
    cls_prefix = "InlineInbuiltNNModules"
    
    # 调用 make_test_cls_with_patches 函数生成测试类
    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "inline_inbuilt_nn_modules", True),
        xfail_prop="_expected_failure_inline_inbuilt_nn_modules",
    )

    # 将生成的测试类添加到 test_classes 字典中
    test_classes[test_class.__name__] = test_class
    # 将测试类添加到全局命名空间中
    globals()[test_class.__name__] = test_class
    # 设置测试类的模块名为当前模块
    test_class.__module__ = __name__
    return test_class

# 待生成测试类的列表
tests = [
    test_misc.MiscTests,
    test_functions.FunctionTests,
    test_modules.NNModuleTests,
    test_higher_order_ops.HigherOrderOpTests,
    test_higher_order_ops.FuncTorchHigherOrderOpTests,
    test_aot_autograd.AotAutogradFallbackTests,
    # test_repros.ReproTests,
]
# 逐个生成测试类并调用 make_inline_inbuilt_nn_modules_cls 函数
for test in tests:
    make_inline_inbuilt_nn_modules_cls(test)
del test  # 清理循环变量 test

# 跳过特定测试用例的运行
unittest.skip(
    InlineInbuiltNNModulesMiscTests.test_cpp_extension_recommends_custom_ops_inline_inbuilt_nn_modules  # noqa: F821
)

if __name__ == "__main__":
    # 如果当前文件作为主程序运行，则从 torch._dynamo.test_case 中导入 run_tests 函数并运行
    from torch._dynamo.test_case import run_tests
    run_tests()
```