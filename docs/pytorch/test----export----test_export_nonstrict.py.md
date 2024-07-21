# `.\pytorch\test\export\test_export_nonstrict.py`

```
# Owner(s): ["oncall: export"]

# 尝试导入本地的 test_export 和 testing 模块，如果导入失败则从全局导入
try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

# 从 torch.export 中导入 export 函数
from torch.export import export

# 定义一个空字典，用于存储测试类
test_classes = {}

# 定义一个名为 mocked_non_strict_export 的函数，用于处理非严格模式的导出
def mocked_non_strict_export(*args, **kwargs):
    # 如果用户已经指定了 strict 参数，则保持不变
    if "strict" in kwargs:
        return export(*args, **kwargs)
    # 否则以非严格模式调用 export 函数
    return export(*args, **kwargs, strict=False)

# 定义一个函数 make_dynamic_cls，用于创建动态类
def make_dynamic_cls(cls):
    # 设置类名前缀
    cls_prefix = "NonStrictExport"

    # 使用 testing.make_test_cls_with_mocked_export 函数创建测试类
    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.NON_STRICT_SUFFIX,
        mocked_non_strict_export,
        xfail_prop="_expected_failure_non_strict",
    )

    # 将创建的测试类添加到 test_classes 字典中，键为类名
    test_classes[test_class.__name__] = test_class

    # 将测试类的名称添加到全局命名空间中，以便在其他地方使用
    globals()[test_class.__name__] = test_class

    # 设置测试类的模块为当前模块的名称
    test_class.__module__ = __name__

    # 返回创建的测试类
    return test_class

# 定义一个测试类列表
tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]

# 对每个测试类调用 make_dynamic_cls 函数进行处理
for test in tests:
    make_dynamic_cls(test)

# 清理掉不再需要的 test 变量，以释放资源
del test

# 如果当前脚本作为主程序运行，则执行 torch._dynamo.test_case 中的 run_tests 函数
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
```