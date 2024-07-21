# `.\pytorch\test\dynamo\test_dynamic_shapes.py`

```
# 导入单元测试和警告模块
import unittest
import warnings

# 从torch._dynamo模块导入config
from torch._dynamo import config
# 从torch._dynamo.testing模块导入make_test_cls_with_patches函数
from torch._dynamo.testing import make_test_cls_with_patches
# 从torch.fx.experimental._config模块导入fx_config
from torch.fx.experimental import _config as fx_config
# 从torch.testing._internal.common_utils模块导入slowTest和TEST_Z3常量
from torch.testing._internal.common_utils import slowTest, TEST_Z3

# 尝试从当前目录相对导入一系列测试模块，若失败则从全局导入
try:
    from . import (
        test_aot_autograd,
        test_ctx_manager,
        test_export,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        test_repros,
        test_sdpa,
        test_subgraphs,
    )
except ImportError:
    import test_aot_autograd
    import test_ctx_manager
    import test_export
    import test_functions
    import test_higher_order_ops
    import test_misc
    import test_modules
    import test_repros
    import test_sdpa
    import test_subgraphs

# 定义一个空字典，用于存储测试类
test_classes = {}

# 创建一个函数，用于动态生成测试类，并将其加入全局命名空间
def make_dynamic_cls(cls):
    # 定义后缀和类名前缀
    suffix = "_dynamic_shapes"
    cls_prefix = "DynamicShapes"

    # 调用make_test_cls_with_patches函数创建测试类，配置一系列参数
    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "assume_static_by_default", False),
        (config, "specialize_int", False),
        (fx_config, "translation_validation", TEST_Z3),
        (fx_config, "check_shape_env_recorded_events", True),
        (fx_config, "validate_shape_env_version_key", True),
        xfail_prop="_expected_failure_dynamic",
    )

    # 将生成的测试类添加到test_classes字典中
    test_classes[test_class.__name__] = test_class
    # 将测试类添加到全局命名空间，使其可以被直接访问
    globals()[test_class.__name__] = test_class
    # 设置测试类的模块属性为当前模块的名称
    test_class.__module__ = __name__
    return test_class

# 定义一个包含所有待生成测试类的列表
tests = [
    test_ctx_manager.CtxManagerTests,
    test_functions.FunctionTests,
    test_misc.MiscTests,
    test_repros.ReproTests,
    test_modules.NNModuleTests,
    test_export.ExportTests,
    test_subgraphs.SubGraphTests,
    test_higher_order_ops.HigherOrderOpTests,
    test_higher_order_ops.FuncTorchHigherOrderOpTests,
    test_aot_autograd.AotAutogradFallbackTests,
    test_sdpa.TestSDPA,
]

# 逐一调用make_dynamic_cls函数生成测试类并进行处理
for test in tests:
    make_dynamic_cls(test)
del test

# 若TEST_Z3为真，则执行以下条件判断
if TEST_Z3:
    # 若config.inline_inbuilt_nn_modules为假，则进行如下操作
    if not config.inline_inbuilt_nn_modules:
        # 标记以下测试为预期失败
        unittest.expectedFailure(
            DynamicShapesMiscTests.test_parameter_free_dynamic_shapes  # noqa: F821
        )

# 标记以下测试为预期失败
unittest.expectedFailure(
    # 只有在没有动态形状的情况下才有效的测试
    DynamicShapesReproTests.test_many_views_with_mutation_dynamic_shapes  # noqa: F821
)

# 将下列测试标记为耗时测试，并执行
DynamicShapesExportTests.test_retracibility_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_dynamic_shapes  # noqa: F821
)

# 将下列测试标记为耗时测试，并执行
DynamicShapesExportTests.test_retracibility_dict_container_inp_out_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_dict_container_inp_out_dynamic_shapes  # noqa: F821
)
# 将测试函数标记为慢速测试，并将其赋值给测试用例的特定属性
DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes  # noqa: F821
)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 如果未设置 TEST_Z3，发出警告并禁用翻译验证
    if not TEST_Z3:
        warnings.warn(
            "translation validation is off. "
            "Testing with translation validation requires Z3."
        )

    # 运行测试套件
    run_tests()
```