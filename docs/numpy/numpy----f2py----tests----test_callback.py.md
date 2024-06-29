# `.\numpy\numpy\f2py\tests\test_callback.py`

```py
import math  # 导入 math 模块，提供数学函数支持
import textwrap  # 导入 textwrap 模块，提供文本包装和填充功能
import sys  # 导入 sys 模块，提供对解释器相关的操作访问
import pytest  # 导入 pytest 模块，用于编写和运行测试用例
import threading  # 导入 threading 模块，提供多线程支持
import traceback  # 导入 traceback 模块，用于提取和格式化异常的回溯信息
import time  # 导入 time 模块，提供时间相关的功能

import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import IS_PYPY  # 导入 IS_PYPY 变量，用于检测是否在 PyPy 下运行
from . import util  # 从当前包中导入 util 模块

class TestF77Callback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "foo.f")]

    @pytest.mark.parametrize("name", "t,t2".split(","))
    @pytest.mark.slow
    def test_all(self, name):
        self.check_function(name)

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = t(fun,[fun_extra_args])

        Wrapper for ``t``.

        Parameters
        ----------
        fun : call-back function

        Other Parameters
        ----------------
        fun_extra_args : input tuple, optional
            Default: ()

        Returns
        -------
        a : int

        Notes
        -----
        Call-back functions::

            def fun(): return a
            Return objects:
                a : int
        """)
        assert self.module.t.__doc__ == expected

    def check_function(self, name):
        t = getattr(self.module, name)  # 获取 self.module 中名称为 name 的属性，并赋值给 t
        r = t(lambda: 4)  # 调用 t，传入匿名函数 lambda: 4 作为参数，并将结果赋值给 r
        assert r == 4  # 断言 r 的值为 4
        r = t(lambda a: 5, fun_extra_args=(6, ))  # 调用 t，传入带有 fun_extra_args 参数的 lambda 函数，并将结果赋值给 r
        assert r == 5  # 断言 r 的值为 5
        r = t(lambda a: a, fun_extra_args=(6, ))  # 调用 t，传入带有 fun_extra_args 参数的 lambda 函数，并将结果赋值给 r
        assert r == 6  # 断言 r 的值为 6
        r = t(lambda a: 5 + a, fun_extra_args=(7, ))  # 调用 t，传入带有 fun_extra_args 参数的 lambda 函数，并将结果赋值给 r
        assert r == 12  # 断言 r 的值为 12
        r = t(lambda a: math.degrees(a), fun_extra_args=(math.pi, ))  # 调用 t，传入带有 fun_extra_args 参数的 math.degrees 函数，并将结果赋值给 r
        assert r == 180  # 断言 r 的值为 180
        r = t(math.degrees, fun_extra_args=(math.pi, ))  # 调用 t，传入带有 fun_extra_args 参数的 math.degrees 函数，并将结果赋值给 r
        assert r == 180  # 断言 r 的值为 180

        r = t(self.module.func, fun_extra_args=(6, ))  # 调用 t，传入带有 fun_extra_args 参数的 self.module.func 函数，并将结果赋值给 r
        assert r == 17  # 断言 r 的值为 17
        r = t(self.module.func0)  # 调用 t，传入 self.module.func0 函数，并将结果赋值给 r
        assert r == 11  # 断言 r 的值为 11
        r = t(self.module.func0._cpointer)  # 调用 t，传入 self.module.func0._cpointer 函数，并将结果赋值给 r
        assert r == 11  # 断言 r 的值为 11

        class A:
            def __call__(self):
                return 7

            def mth(self):
                return 9

        a = A()
        r = t(a)  # 调用 t，传入实例 a，并将结果赋值给 r
        assert r == 7  # 断言 r 的值为 7
        r = t(a.mth)  # 调用 t，传入 a.mth 方法，并将结果赋值给 r
        assert r == 9  # 断言 r 的值为 9

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback(self):
        def callback(code):
            if code == "r":
                return 0
            else:
                return 1

        f = getattr(self.module, "string_callback")  # 获取 self.module 中名为 "string_callback" 的属性，并赋值给 f
        r = f(callback)  # 调用 f，传入 callback 函数作为参数，并将结果赋值给 r
        assert r == 0  # 断言 r 的值为 0

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback_array(self):
        # See gh-10027
        # 创建一个长度为1的零数组，元素类型为字节串，每个元素长度为8字节
        cu1 = np.zeros((1, ), "S8")
        # 创建一个长度为1的零数组，每个元素为单个字符，每个字符为1字节
        cu2 = np.zeros((1, 8), "c")
        # 创建一个包含一个空字符串的字节数组，每个元素长度为8字节
        cu3 = np.array([""], "S8")

        def callback(cu, lencu):
            # 检查数组形状是否为(lencu,)，如果不是则返回1
            if cu.shape != (lencu,):
                return 1
            # 检查数组元素类型是否为字节串"S8"，如果不是则返回2
            if cu.dtype != "S8":
                return 2
            # 检查数组所有元素是否都是空字节串，如果不是则返回3
            if not np.all(cu == b""):
                return 3
            # 满足所有条件则返回0
            return 0

        # 获取对象self.module中名为"string_callback_array"的函数对象
        f = getattr(self.module, "string_callback_array")
        # 对cu1, cu2, cu3中的每个数组执行回调函数f，并断言返回值为0
        for cu in [cu1, cu2, cu3]:
            res = f(callback, cu, cu.size)
            assert res == 0

    def test_threadsafety(self):
        # 如果回调处理不是线程安全的，则可能导致段错误

        errors = []

        def cb():
            # 在这里睡眠以增加另一个线程在同一时间调用它们回调函数的可能性
            time.sleep(1e-3)

            # 检查重入性
            r = self.module.t(lambda: 123)
            assert r == 123

            return 42

        def runner(name):
            try:
                for j in range(50):
                    # 调用self.module中名为"t"的函数，并传递cb作为回调函数
                    r = self.module.t(cb)
                    # 断言返回值为42
                    assert r == 42
                    # 检查函数name的有效性
                    self.check_function(name)
            except Exception:
                errors.append(traceback.format_exc())

        # 创建20个线程，每个线程调用runner函数，并传递不同的参数("t"或"t2")
        threads = [
            threading.Thread(target=runner, args=(arg, ))
            for arg in ("t", "t2") for n in range(20)
        ]

        # 启动所有线程
        for t in threads:
            t.start()

        # 等待所有线程结束
        for t in threads:
            t.join()

        # 如果有错误发生，将所有错误信息合并为一个字符串并抛出AssertionError
        errors = "\n\n".join(errors)
        if errors:
            raise AssertionError(errors)

    def test_hidden_callback(self):
        try:
            # 尝试调用self.module中的"hidden_callback"函数，期望抛出异常并检查异常消息
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        try:
            # 尝试调用self.module中的"hidden_callback2"函数，期望抛出异常并检查异常消息
            self.module.hidden_callback2(2)
        except Exception as msg:
            assert str(msg).startswith("cb: Callback global_f not defined")

        # 设置self.module中的全局变量"global_f"为一个lambda函数，返回输入参数加1
        self.module.global_f = lambda x: x + 1
        # 调用self.module中的"hidden_callback"函数，预期返回3
        r = self.module.hidden_callback(2)
        assert r == 3

        # 更新self.module中的全局变量"global_f"为一个lambda函数，返回输入参数加2
        self.module.global_f = lambda x: x + 2
        # 再次调用self.module中的"hidden_callback"函数，预期返回4
        r = self.module.hidden_callback(2)
        assert r == 4

        # 删除self.module中的全局变量"global_f"
        del self.module.global_f
        # 尝试调用self.module中的"hidden_callback"函数，期望抛出异常并检查异常消息
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        # 设置self.module中的全局变量"global_f"为一个lambda函数，默认参数为0，返回输入参数加3
        self.module.global_f = lambda x=0: x + 3
        # 再次调用self.module中的"hidden_callback"函数，预期返回5
        r = self.module.hidden_callback(2)
        assert r == 5

        # 重现gh18341的问题
        # 调用self.module中的"hidden_callback2"函数，预期返回3
        r = self.module.hidden_callback2(2)
        assert r == 3
# 定义一个名为 TestF77CallbackPythonTLS 的测试类，继承自 TestF77Callback 类
class TestF77CallbackPythonTLS(TestF77Callback):
    """
    Callback tests using Python thread-local storage instead of
    compiler-provided
    """
    
    # 设置类的选项属性为包含字符串 "-DF2PY_USE_PYTHON_TLS" 的列表
    options = ["-DF2PY_USE_PYTHON_TLS"]


# 定义一个名为 TestF90Callback 的测试类，继承自 util.F2PyTest 类
class TestF90Callback(util.F2PyTest):
    # 设置类的 sources 属性为包含指定源文件路径的列表
    sources = [util.getpath("tests", "src", "callback", "gh17797.f90")]

    # 标记为慢速测试
    @pytest.mark.slow
    # 定义名为 test_gh17797 的测试方法
    def test_gh17797(self):
        # 定义一个函数 incr，参数 x，返回值为 x + 123
        def incr(x):
            return x + 123
        
        # 创建一个包含 [1, 2, 3] 的 numpy 数组 y，数据类型为 np.int64
        y = np.array([1, 2, 3], dtype=np.int64)
        # 调用 self.module 的 gh17797 方法，参数为 incr 函数和数组 y，返回结果赋给 r
        r = self.module.gh17797(incr, y)
        # 断言 r 的值等于 123 + 1 + 2 + 3
        assert r == 123 + 1 + 2 + 3


# 定义一个名为 TestGH18335 的测试类，继承自 util.F2PyTest 类
class TestGH18335(util.F2PyTest):
    """
    The reproduction of the reported issue requires specific input that
    extensions may break the issue conditions, so the reproducer is
    implemented as a separate test class. Do not extend this test with
    other tests!
    """
    # 设置类的 sources 属性为包含指定源文件路径的列表
    sources = [util.getpath("tests", "src", "callback", "gh18335.f90")]

    # 标记为慢速测试
    @pytest.mark.slow
    # 定义名为 test_gh18335 的测试方法
    def test_gh18335(self):
        # 定义一个函数 foo，参数 x，将 x[0] 的值加一
        def foo(x):
            x[0] += 1
        
        # 调用 self.module 的 gh18335 方法，参数为 foo 函数，返回结果赋给 r
        r = self.module.gh18335(foo)
        # 断言 r 的值等于 123 + 1
        assert r == 123 + 1


# 定义一个名为 TestGH25211 的测试类，继承自 util.F2PyTest 类
class TestGH25211(util.F2PyTest):
    # 设置类的 sources 属性为包含指定源文件路径的列表
    sources = [util.getpath("tests", "src", "callback", "gh25211.f"),
               util.getpath("tests", "src", "callback", "gh25211.pyf")]
    # 设置模块名为 "callback2"
    module_name = "callback2"

    # 定义名为 test_gh25211 的测试方法
    def test_gh25211(self):
        # 定义一个函数 bar，参数 x，返回值为 x*x
        def bar(x):
            return x*x
        
        # 调用 self.module 的 foo 方法，参数为 bar 函数，返回结果赋给 res
        res = self.module.foo(bar)
        # 断言 res 的值等于 110
        assert res == 110
```