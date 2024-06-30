# `D:\src\scipysrc\scipy\scipy\_lib\tests\test__gcutils.py`

```
""" Test for assert_deallocated context manager and gc utilities
"""
# 导入必要的模块和函数
import gc  # 导入垃圾回收模块

# 导入所需的函数和异常类
from scipy._lib._gcutils import (set_gc_state, gc_state, assert_deallocated,
                                 ReferenceError, IS_PYPY)

# 导入断言函数
from numpy.testing import assert_equal

# 导入 pytest 模块
import pytest


def test_set_gc_state():
    # 检查当前的垃圾回收状态
    gc_status = gc.isenabled()
    try:
        # 遍历两种状态的测试用例：True 和 False
        for state in (True, False):
            gc.enable()
            # 设置垃圾回收状态
            set_gc_state(state)
            # 断言当前垃圾回收状态是否与设置的状态一致
            assert_equal(gc.isenabled(), state)
            gc.disable()
            set_gc_state(state)
            assert_equal(gc.isenabled(), state)
    finally:
        # 恢复最初的垃圾回收状态
        if gc_status:
            gc.enable()


def test_gc_state():
    # 测试 gc_state 上下文管理器
    gc_status = gc.isenabled()
    try:
        # 遍历两种初始状态：True 和 False
        for pre_state in (True, False):
            set_gc_state(pre_state)
            # 遍历两种 with 块内的状态：True 和 False
            for with_state in (True, False):
                # 在 with 块内检查 gc 状态是否为 with_state
                with gc_state(with_state):
                    assert_equal(gc.isenabled(), with_state)
                # 在 with 块外，检查是否返回到之前的状态
                assert_equal(gc.isenabled(), pre_state)
                # 即使在 with 块内部显式设置了 gc 状态，也应该返回到之前的状态
                with gc_state(with_state):
                    assert_equal(gc.isenabled(), with_state)
                    set_gc_state(not with_state)
                assert_equal(gc.isenabled(), pre_state)
    finally:
        if gc_status:
            gc.enable()


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_assert_deallocated():
    # 普通用法
    class C:
        def __init__(self, arg0, arg1, name='myname'):
            self.name = name
    # 遍历 gc 当前状态：True 和 False
    for gc_current in (True, False):
        with gc_state(gc_current):
            # 在 with 块中删除对象，这是允许的
            with assert_deallocated(C, 0, 2, 'another name') as c:
                assert_equal(c.name, 'another name')
                del c
            # 或者在 with 块中不使用对象，同样是允许的
            with assert_deallocated(C, 0, 2, name='third name'):
                pass
            assert_equal(gc.isenabled(), gc_current)


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_assert_deallocated_nodel():
    class C:
        pass
    with pytest.raises(ReferenceError):
        # 在 with 块中使用 assert_deallocated(C)，需要在使用后删除对象
        # 注意：为了使测试功能正常，需要将 assert_deallocated(C) 赋值给 _
        # _ 被用于引用计数，但在 with 块的主体中没有被引用，它只是为了引用计数存在
        with assert_deallocated(C) as _:
            pass


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_assert_deallocated_circular():
    class C:
        def __init__(self):
            self._circular = self
    # 使用 pytest 的 assertRaises 方法来检查是否引发了 ReferenceError 异常
    with pytest.raises(ReferenceError):
        # 在这个上下文中，期望出现 Circular reference（循环引用）导致没有自动垃圾回收的情况
        with assert_deallocated(C) as c:
            # 删除 c 变量，模拟对循环引用的处理
            del c
@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
# 定义一个测试函数，用于测试在某些条件下跳过测试（例如在 PyPy 上）
def test_assert_deallocated_circular2():
    # 定义一个类 C，其构造函数会创建一个自己的属性 _circular 并指向自身
    class C:
        def __init__(self):
            self._circular = self
    
    # 断言在代码块中引发 ReferenceError 异常
    with pytest.raises(ReferenceError):
        # 在此代码块中，使用 assert_deallocated 上下文管理器检查对象 C 的解引用
        with assert_deallocated(C):
            pass
        # 此处不会自动执行垃圾回收，仍然存在循环引用
```