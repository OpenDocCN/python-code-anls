# `.\pytorch\test\export\test_serdes.py`

```py
# Owner(s): ["oncall: export"]

# 导入必要的模块
import io

# 尝试从当前目录导入 test_export 和 testing 模块，如果失败则从全局导入
try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

# 从 torch.export 模块中导入 export、load 和 save 函数
from torch.export import export, load, save

# 定义一个空字典，用于存储测试类
test_classes = {}


# 定义一个模拟的序列化导出函数
def mocked_serder_export(*args, **kwargs):
    # 调用 export 函数进行序列化导出
    ep = export(*args, **kwargs)
    # 创建一个字节流缓冲区
    buffer = io.BytesIO()
    # 将导出对象保存到字节流缓冲区
    save(ep, buffer)
    # 将字节流缓冲区的指针位置设置为开头
    buffer.seek(0)
    # 从字节流缓冲区加载导出对象
    loaded_ep = load(buffer)
    return loaded_ep


# 定义一个函数，用于创建动态测试类
def make_dynamic_cls(cls):
    # 设置测试类的名称前缀
    cls_prefix = "SerDesExport"
    # 使用 testing 模块的函数创建带有模拟导出的测试类
    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.SERDES_SUFFIX,
        mocked_serder_export,
        xfail_prop="_expected_failure_serdes",
    )
    # 将创建的测试类添加到 test_classes 字典中，以类名为键
    test_classes[test_class.__name__] = test_class
    # 将测试类以其名称添加到全局变量中，使得测试可以运行
    globals()[test_class.__name__] = test_class
    # 设置测试类的模块信息为当前模块的名称
    test_class.__module__ = __name__


# 定义一个测试类列表
tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]

# 遍历测试类列表，为每个测试类调用 make_dynamic_cls 函数进行处理
for test in tests:
    make_dynamic_cls(test)

# 删除循环变量 test，以清理命名空间
del test

# 如果当前模块作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入并运行测试函数
    from torch._dynamo.test_case import run_tests

    run_tests()
```