# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_ccallback.py`

```
# 导入需要的测试断言和异常处理工具
from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises

# 导入时间、pytest、ctypes、线程相关的模块
import time
import pytest
import ctypes
import threading

# 导入与回调相关的模块和函数
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable

try:
    # 尝试导入cffi库，标记是否成功导入
    import cffi
    HAVE_CFFI = True
except ImportError:
    HAVE_CFFI = False

# 定义一个常量，表示错误值为2.0
ERROR_VALUE = 2.0

# Python 回调函数的定义，根据输入的值进行不同的计算并返回
def callback_python(a, user_data=None):
    if a == ERROR_VALUE:
        raise ValueError("bad value")

    if user_data is None:
        return a + 1
    else:
        return a + user_data

# 根据基地址和签名获取cffi函数的实例
def _get_cffi_func(base, signature):
    if not HAVE_CFFI:
        pytest.skip("cffi not installed")

    # 将基地址转换为void指针
    voidp = ctypes.cast(base, ctypes.c_void_p)
    address = voidp.value

    # 使用cffi创建对应的函数句柄
    ffi = cffi.FFI()
    func = ffi.cast(signature, address)
    return func

# 获取ctypes数据的函数，返回一个指向双精度浮点数的指针
def _get_ctypes_data():
    value = ctypes.c_double(2.0)
    return ctypes.cast(ctypes.pointer(value), ctypes.c_voidp)

# 获取cffi数据的函数，返回一个新的双精度浮点数的指针
def _get_cffi_data():
    if not HAVE_CFFI:
        pytest.skip("cffi not installed")
    ffi = cffi.FFI()
    return ffi.new('double *', 2.0)

# 不同调用类型对应的测试函数
CALLERS = {
    'simple': _test_ccallback.test_call_simple,
    'nodata': _test_ccallback.test_call_nodata,
    'nonlocal': _test_ccallback.test_call_nonlocal,
    'cython': _test_ccallback_cython.test_call_cython,
}

# 已知调用者所知道签名的函数字典
FUNCS = {
    'python': lambda: callback_python,
    'capsule': lambda: _test_ccallback.test_get_plus1_capsule(),
    'cython': lambda: LowLevelCallable.from_cython(_test_ccallback_cython,
                                                   "plus1_cython"),
    'ctypes': lambda: _test_ccallback_cython.plus1_ctypes,
    'cffi': lambda: _get_cffi_func(_test_ccallback_cython.plus1_ctypes,
                                   'double (*)(double, int *, void *)'),
    'capsule_b': lambda: _test_ccallback.test_get_plus1b_capsule(),
    'cython_b': lambda: LowLevelCallable.from_cython(_test_ccallback_cython,
                                                     "plus1b_cython"),
    'ctypes_b': lambda: _test_ccallback_cython.plus1b_ctypes,
    'cffi_b': lambda: _get_cffi_func(_test_ccallback_cython.plus1b_ctypes,
                                     'double (*)(double, double, int *, void *)'),
}

# 调用者不知道签名的函数字典
BAD_FUNCS = {
    'capsule_bc': lambda: _test_ccallback.test_get_plus1bc_capsule(),
    'cython_bc': lambda: LowLevelCallable.from_cython(_test_ccallback_cython,
                                                      "plus1bc_cython"),
    'ctypes_bc': lambda: _test_ccallback_cython.plus1bc_ctypes,
    'cffi_bc': lambda: _get_cffi_func(
        _test_ccallback_cython.plus1bc_ctypes,
        'double (*)(double, double, double, int *, void *)'
    ),
}

# 不同用户数据类型对应的获取函数
USER_DATAS = {
    'ctypes': _get_ctypes_data,
    'cffi': _get_cffi_data,
    'capsule': _test_ccallback.test_get_data_capsule,
}
}

# 测试回调函数的功能
def test_callbacks():
    # 定义回调函数检查函数
    def check(caller, func, user_data):
        # 从全局字典中获取调用者、函数和用户数据
        caller = CALLERS[caller]
        func = FUNCS[func]()
        user_data = USER_DATAS[user_data]()

        # 根据函数类型选择适当的操作
        if func is callback_python:
            # 如果函数是Python回调函数，定义一个新的函数func2
            def func2(x):
                return func(x, 2.0)
        else:
            # 否则，创建LowLevelCallable对象
            func2 = LowLevelCallable(func, user_data)
            func = LowLevelCallable(func)

        # 测试基本调用，确保结果为2.0
        assert_equal(caller(func, 1.0), 2.0)

        # 测试传入错误值是否引发ValueError
        assert_raises(ValueError, caller, func, ERROR_VALUE)

        # 测试传入用户数据后的调用
        assert_equal(caller(func2, 1.0), 3.0)

    # 嵌套循环测试不同的调用者、函数和用户数据组合
    for caller in sorted(CALLERS.keys()):
        for func in sorted(FUNCS.keys()):
            for user_data in sorted(USER_DATAS.keys()):
                check(caller, func, user_data)


# 测试错误的回调函数行为
def test_bad_callbacks():
    # 定义检查错误回调函数的函数
    def check(caller, func, user_data):
        # 从全局字典中获取调用者和用户数据，以及错误的函数
        caller = CALLERS[caller]
        user_data = USER_DATAS[user_data]()
        func = BAD_FUNCS[func]()

        # 根据函数类型选择适当的操作
        if func is callback_python:
            # 如果函数是Python回调函数，定义一个新的函数func2
            def func2(x):
                return func(x, 2.0)
        else:
            # 否则，创建LowLevelCallable对象
            func2 = LowLevelCallable(func, user_data)
            func = LowLevelCallable(func)

        # 测试基本调用是否引发ValueError
        assert_raises(ValueError, caller, LowLevelCallable(func), 1.0)

        # 测试传入用户数据后的调用是否引发ValueError
        assert_raises(ValueError, caller, func2, 1.0)

        # 测试错误消息是否包含预期的签名信息
        llfunc = LowLevelCallable(func)
        try:
            caller(llfunc, 1.0)
        except ValueError as err:
            msg = str(err)
            assert_(llfunc.signature in msg, msg)
            assert_('double (double, double, int *, void *)' in msg, msg)

    # 嵌套循环测试不同的调用者、错误函数和用户数据组合
    for caller in sorted(CALLERS.keys()):
        for func in sorted(BAD_FUNCS.keys()):
            for user_data in sorted(USER_DATAS.keys()):
                check(caller, func, user_data)


# 测试签名覆盖功能
def test_signature_override():
    # 获取测试用的调用者和函数
    caller = _test_ccallback.test_call_simple
    func = _test_ccallback.test_get_plus1_capsule()

    # 创建带有错误签名的LowLevelCallable对象，并验证是否引发ValueError
    llcallable = LowLevelCallable(func, signature="bad signature")
    assert_equal(llcallable.signature, "bad signature")
    assert_raises(ValueError, caller, llcallable, 3)

    # 创建带有正确签名的LowLevelCallable对象，并验证调用结果
    llcallable = LowLevelCallable(func, signature="double (double, int *, void *)")
    assert_equal(llcallable.signature, "double (double, int *, void *)")
    assert_equal(caller(llcallable, 3), 4)


# 测试线程安全性
def test_threadsafety():
    # 定义递归回调函数callback，确保在多线程中的安全性
    def callback(a, caller):
        if a <= 0:
            return 1
        else:
            res = caller(lambda x: callback(x, caller), a - 1)
            return 2 * res
    # 定义一个名为 check 的函数，接收一个参数 caller
    def check(caller):
        # 从全局字典 CALLERS 中获取对应 caller 的值，并赋给局部变量 caller
        caller = CALLERS[caller]

        # 初始化一个空列表 results，用于存储每次调用的结果
        results = []

        # 设置一个计数器 count，初始值为 10
        count = 10

        # 定义一个内部函数 run，用于在多线程中执行任务
        def run():
            # 线程休眠 0.01 秒
            time.sleep(0.01)
            # 调用传入的 caller 函数，使用 lambda 表达式作为参数，传递给 callback 函数和 caller
            r = caller(lambda x: callback(x, caller), count)
            # 将结果 r 添加到 results 列表中
            results.append(r)

        # 创建包含 20 个线程的列表 threads，每个线程的目标函数都是 run 函数
        threads = [threading.Thread(target=run) for j in range(20)]
        # 启动所有线程
        for thread in threads:
            thread.start()
        # 等待所有线程执行完毕
        for thread in threads:
            thread.join()

        # 断言结果列表 results 的每个元素都等于 2 的 count 次方，即 [2.0**count]*len(threads)
        assert_equal(results, [2.0**count]*len(threads))

    # 遍历全局字典 CALLERS 的键（即 caller 的可能取值），对每个键调用 check 函数
    for caller in CALLERS.keys():
        check(caller)
```