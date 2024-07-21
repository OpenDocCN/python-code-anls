# `.\pytorch\test\export\test_export_training_ir_to_run_decomp.py`

```py
# Owner(s): ["oncall: export"]

# 尝试导入本地模块，如果失败则导入全局模块
try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

# 从 torch.export._trace 中导入 _export_for_training 函数
from torch.export._trace import _export_for_training

# 创建一个空字典用于存储测试类
test_classes = {}

# 定义一个模拟的导出函数，用于训练中的 IR 到运行分解导出
def mocked_training_ir_to_run_decomp_export(*args, **kwargs):
    # 调用 _export_for_training 函数生成导出对象
    ep = _export_for_training(*args, **kwargs)
    # 运行分解操作并返回结果
    return ep.run_decompositions(
        {}, _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY
    )

# 创建动态测试类的函数
def make_dynamic_cls(cls):
    # 定义类名前缀
    cls_prefix = "TrainingIRToRunDecompExport"
    
    # 使用 testing.make_test_cls_with_mocked_export 创建带有模拟导出功能的测试类
    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.TRAINING_IR_DECOMP_SUFFIX,
        mocked_training_ir_to_run_decomp_export,
        xfail_prop="_expected_failure_training_ir_to_run_decomp",
    )
    
    # 将创建的测试类添加到 test_classes 字典中
    test_classes[test_class.__name__] = test_class
    # 下面这行用于确保测试能够运行，不要删除
    globals()[test_class.__name__] = test_class
    # 设置测试类的模块名为当前模块名
    test_class.__module__ = __name__
    return test_class

# 定义需要动态创建测试类的测试列表
tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]

# 遍历 tests 列表，并为每个测试类创建动态测试类
for test in tests:
    make_dynamic_cls(test)

# 删除 tests 变量，清理命名空间
del test

# 如果当前模块是主模块，则执行测试
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入并运行测试
    from torch._dynamo.test_case import run_tests
    run_tests()
```