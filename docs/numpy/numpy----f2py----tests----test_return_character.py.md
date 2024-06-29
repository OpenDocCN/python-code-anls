# `.\numpy\numpy\f2py\tests\test_return_character.py`

```
# 导入 pytest 测试框架
import pytest

# 导入 numpy 数组模块，并导入 array 函数
from numpy import array

# 从当前目录下的 . 包中导入 util 模块
from . import util

# 导入 platform 模块
import platform

# 检查当前机器是否为 s390x 架构，返回布尔值
IS_S390X = platform.machine() == "s390x"

# 用 pytest.mark.slow 标记的测试类，继承自 util.F2PyTest 类
@pytest.mark.slow
class TestReturnCharacter(util.F2PyTest):
    
    # 定义一个检查函数，接受参数 t 和 tname
    def check_function(self, t, tname):
        
        # 如果 tname 在以下列表中
        if tname in ["t0", "t1", "s0", "s1"]:
            # 断言 t("23") 返回 b"2"
            assert t("23") == b"2"
            # 将 t("ab") 的返回结果赋给 r，断言 r 等于 b"a"
            r = t("ab")
            assert r == b"a"
            # 将 t(array("ab")) 的返回结果赋给 r，断言 r 等于 b"a"
            r = t(array("ab"))
            assert r == b"a"
            # 将 t(array(77, "u1")) 的返回结果赋给 r，断言 r 等于 b"M"
            r = t(array(77, "u1"))
            assert r == b"M"
        
        # 如果 tname 在以下列表中
        elif tname in ["ts", "ss"]:
            # 断言 t(23) 返回 b"23"
            assert t(23) == b"23"
            # 断言 t("123456789abcdef") 返回 b"123456789a"
            assert t("123456789abcdef") == b"123456789a"
        
        # 如果 tname 在以下列表中
        elif tname in ["t5", "s5"]:
            # 断言 t(23) 返回 b"23"
            assert t(23) == b"23"
            # 断言 t("ab") 返回 b"ab"
            assert t("ab") == b"ab"
            # 断言 t("123456789abcdef") 返回 b"12345"
            assert t("123456789abcdef") == b"12345"
        
        # 如果不在以上任何一个列表中，抛出 NotImplementedError 异常
        else:
            raise NotImplementedError

# 测试类 TestFReturnCharacter 继承自 TestReturnCharacter 类
class TestFReturnCharacter(TestReturnCharacter):
    
    # sources 列表包含两个路径，指定测试源文件位置
    sources = [
        util.getpath("tests", "src", "return_character", "foo77.f"),
        util.getpath("tests", "src", "return_character", "foo90.f90"),
    ]
    
    # 使用 pytest.mark.xfail 标记测试，如果 IS_S390X 为 True，则原因为 "callback returns ' '" 的测试失败
    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    # 参数化测试，name 参数取自字符串 "t0,t1,t5,s0,s1,s5,ss"，分割后成为列表
    @pytest.mark.parametrize("name", "t0,t1,t5,s0,s1,s5,ss".split(","))
    def test_all_f77(self, name):
        # 调用父类的 check_function 方法，传入相应的函数和名称进行测试
        self.check_function(getattr(self.module, name), name)
    
    # 使用 pytest.mark.xfail 标记测试，如果 IS_S390X 为 True，则原因为 "callback returns ' '" 的测试失败
    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    # 参数化测试，name 参数取自字符串 "t0,t1,t5,ts,s0,s1,s5,ss"，分割后成为列表
    @pytest.mark.parametrize("name", "t0,t1,t5,ts,s0,s1,s5,ss".split(","))
    def test_all_f90(self, name):
        # 调用父类的 check_function 方法，传入相应的函数和名称进行测试
        self.check_function(getattr(self.module.f90_return_char, name), name)
```