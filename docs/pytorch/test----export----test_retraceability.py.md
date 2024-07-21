# `.\pytorch\test\export\test_retraceability.py`

```
# Owner(s): ["oncall: export"]

# 尝试导入本地模块test_export和testing，如果失败则导入全局模块test_export和testing
try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

# 从torch.export模块中导入export函数
from torch.export import export

# 定义空字典test_classes，用于存储测试类
test_classes = {}


# 定义一个模拟的重追溯导出函数，接受任意位置和关键字参数
def mocked_retraceability_export(*args, **kwargs):
    # 调用原始的export函数，生成ep对象
    ep = export(*args, **kwargs)
    
    # 如果kwargs中有动态形状(dynamic_shapes)参数，并且其类型为字典，则将其转换为元组
    if "dynamic_shapes" in kwargs:
        if isinstance(kwargs["dynamic_shapes"], dict):
            kwargs["dynamic_shapes"] = tuple(kwargs["dynamic_shapes"].values())

    # 对ep对象的模块应用export，生成新的ep对象
    ep = export(ep.module(), *(args[1:]), **kwargs)
    return ep


# 定义一个生成动态类的函数make_dynamic_cls，接受一个类cls作为参数
def make_dynamic_cls(cls):
    # 定义类名前缀
    cls_prefix = "RetraceExport"
    
    # 使用testing模块的make_test_cls_with_mocked_export函数创建测试类test_class，
    # 并将其加入test_classes字典
    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.RETRACEABILITY_SUFFIX,
        mocked_retraceability_export,
        xfail_prop="_expected_failure_retrace",
    )

    test_classes[test_class.__name__] = test_class
    # 下一行如果删除将停止测试的运行
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


# 定义一个包含需要动态生成测试类的列表tests
tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]

# 遍历tests列表中的每个测试类，对每个类调用make_dynamic_cls函数
for test in tests:
    make_dynamic_cls(test)

# 删除变量test，释放内存
del test

# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块导入并运行测试函数run_tests
    from torch._dynamo.test_case import run_tests
    run_tests()
```