# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_set_name.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 从 pandas 库中导入 Series 类
from pandas import Series

# 定义一个名为 TestSetName 的测试类
class TestSetName:
    # 定义测试方法 test_set_name，用于测试 Series 对象的 _set_name 方法
    def test_set_name(self):
        # 创建一个包含 [1, 2, 3] 的 Series 对象
        ser = Series([1, 2, 3])
        # 调用 _set_name 方法，将 Series 对象的名字设置为 "foo"，返回新的 Series 对象 ser2
        ser2 = ser._set_name("foo")
        # 断言新的 Series 对象 ser2 的名字为 "foo"
        assert ser2.name == "foo"
        # 断言原始 Series 对象 ser 的名字为 None
        assert ser.name is None
        # 断言两个 Series 对象 ser 和 ser2 是不同的对象
        assert ser is not ser2

    # 定义测试方法 test_set_name_attribute，测试直接设置 Series 对象的名字属性
    def test_set_name_attribute(self):
        # 创建一个包含 [1, 2, 3] 的 Series 对象 ser
        ser = Series([1, 2, 3])
        # 创建一个带有名字 "bar" 的 Series 对象 ser2
        ser2 = Series([1, 2, 3], name="bar")
        # 遍历不同类型的值，尝试设置 Series 对象 ser 的名字属性
        for name in [7, 7.0, "name", datetime(2001, 1, 1), (1,), "\u05d0"]:
            # 设置 ser 的名字属性为当前遍历的 name 值
            ser.name = name
            # 断言 ser 的名字属性确实为当前设置的 name 值
            assert ser.name == name
            # 设置 ser2 的名字属性为当前遍历的 name 值
            ser2.name = name
            # 断言 ser2 的名字属性确实为当前设置的 name 值
            assert ser2.name == name
```