# `ZeroNet\src\Test\TestFlag.py`

```py
# 导入 os 模块
import os
# 导入 pytest 模块
import pytest
# 从 util.Flag 模块中导入 Flag 类
from util.Flag import Flag

# 定义测试类 TestFlag
class TestFlag:
    # 定义测试方法 testFlagging
    def testFlagging(self):
        # 创建 Flag 对象
        flag = Flag()
        # 使用装饰器标记 testFn 方法为 admin 和 no_multiuser
        @flag.admin
        @flag.no_multiuser
        def testFn(anything):
            return anything

        # 断言 testFn 方法被标记为 admin 和 no_multiuser
        assert "admin" in flag.db["testFn"]
        assert "no_multiuser" in flag.db["testFn"]

    # 定义测试方法 testSubclassedFlagging
    def testSubclassedFlagging(self):
        # 创建 Flag 对象
        flag = Flag()
        # 定义 Test 类
        class Test:
            # 使用装饰器标记 testFn 方法为 admin 和 no_multiuser
            @flag.admin
            @flag.no_multiuser
            def testFn(anything):
                return anything

        # 定义 SubTest 类继承自 Test 类
        class SubTest(Test):
            pass

        # 断言 testFn 方法被标记为 admin 和 no_multiuser
        assert "admin" in flag.db["testFn"]
        assert "no_multiuser" in flag.db["testFn"]

    # 定义测试方法 testInvalidFlag
    def testInvalidFlag(self):
        # 创建 Flag 对象
        flag = Flag()
        # 使用装饰器标记 testFn 方法为 no_multiuser 和 unknown_flag，预期会抛出异常
        with pytest.raises(Exception) as err:
            @flag.no_multiuser
            @flag.unknown_flag
            def testFn(anything):
                return anything
        # 断言异常信息包含 "Invalid flag"
        assert "Invalid flag" in str(err.value)
```