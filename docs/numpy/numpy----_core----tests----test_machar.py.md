# `.\numpy\numpy\_core\tests\test_machar.py`

```
"""
Test machar. Given recent changes to hardcode type data, we might want to get
rid of both MachAr and this test at some point.

"""
# 从numpy._core._machar模块中导入MachAr类
from numpy._core._machar import MachAr
# 导入numpy._core.numerictypes模块，命名为ntypes
import numpy._core.numerictypes as ntypes
# 从numpy模块中导入errstate和array函数
from numpy import errstate, array

# 定义一个测试类TestMachAr
class TestMachAr:
    # 定义一个私有方法_run_machar_highprec
    def _run_machar_highprec(self):
        # 尝试使用足够高精度的数据类型（ntypes.float96）实例化MachAr对象，
        # 可能会引起下溢（underflow）
        try:
            # 设置hiprec为ntypes.float96
            hiprec = ntypes.float96
            # 实例化MachAr对象，使用lambda函数和array函数处理数据
            MachAr(lambda v: array(v, hiprec))
        except AttributeError:
            # 如果没有找到ntypes.float96属性，则输出相应的跳过测试信息
            "Skipping test: no ntypes.float96 available on this platform."

    # 定义一个测试方法test_underlow
    def test_underlow(self):
        # 回归测试＃759：
        # 对于dtype = np.float96，实例化MachAr会引发虚假警告。
        # 使用errstate上下文管理器，设置所有错误都抛出异常
        with errstate(all='raise'):
            try:
                # 调用私有方法_run_machar_highprec
                self._run_machar_highprec()
            except FloatingPointError as e:
                # 捕获FloatingPointError异常，如果被抛出则输出错误信息
                msg = "Caught %s exception, should not have been raised." % e
                # 抛出断言异常，包含错误信息msg
                raise AssertionError(msg)
```