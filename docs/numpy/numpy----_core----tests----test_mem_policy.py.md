# `.\numpy\numpy\_core\tests\test_mem_policy.py`

```
# 引入必要的模块和库
import asyncio
import gc  # 垃圾回收模块
import os  # 系统操作模块
import sys  # 系统相关的参数和函数
import threading  # 线程模块，用于多线程编程
import warnings  # 警告模块，用于处理警告信息

import pytest  # 测试框架

import numpy as np  # 科学计算库 NumPy
from numpy.testing import extbuild, assert_warns, IS_WASM, IS_EDITABLE  # NumPy 测试相关模块
from numpy._core.multiarray import get_handler_name  # NumPy 多维数组核心模块

# FIXME: numpy.testing.extbuild 使用了 `numpy.distutils`，因此在 Python 3.12 及更高版本上不起作用。
# 这是一个内部测试工具，所以暂时跳过这些测试。
@pytest.fixture
def get_module(tmp_path):
    """添加一个内存策略，将一个虚假指针插入到实际分配的内存中间 64 字节处，并在前缀中填充一些文本。
    然后在每次内存操作时检查前缀是否存在，以确保所有的分配/重新分配/释放/清零通过这里的函数进行。
    """
    if sys.platform.startswith('cygwin'):
        pytest.skip('link fails on cygwin')  # 在 cygwin 上链接失败，跳过测试
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")  # 无法在 Wasm 中构建模块，跳过测试
    if IS_EDITABLE:
        pytest.skip("Can't build module for editable install")  # 无法为可编辑安装构建模块，跳过测试

    more_init = "import_array();"
    try:
        import mem_policy
        return mem_policy  # 尝试导入 mem_policy 模块，如果存在则返回该模块
    except ImportError:
        pass

    # 如果 mem_policy 模块不存在，则构建并加载它
    return extbuild.build_and_import_extension('mem_policy',
                                               functions,
                                               prologue=prologue,
                                               include_dirs=[np.get_include()],
                                               build_dir=tmp_path,
                                               more_init=more_init)


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="no numpy.distutils")
def test_set_policy(get_module):
    get_handler_name = np._core.multiarray.get_handler_name
    get_handler_version = np._core.multiarray.get_handler_version
    orig_policy_name = get_handler_name()

    a = np.arange(10).reshape((2, 5))  # a 不拥有自己的数据
    assert get_handler_name(a) is None
    assert get_handler_version(a) is None
    assert get_handler_name(a.base) == orig_policy_name
    assert get_handler_version(a.base) == 1

    orig_policy = get_module.set_secret_data_policy()

    b = np.arange(10).reshape((2, 5))  # b 不拥有自己的数据
    assert get_handler_name(b) is None
    assert get_handler_version(b) is None
    assert get_handler_name(b.base) == 'secret_data_allocator'
    assert get_handler_version(b.base) == 1

    if orig_policy_name == 'default_allocator':
        get_module.set_old_policy(None)  # 测试 PyDataMem_SetHandler(NULL)
        assert get_handler_name() == 'default_allocator'
    else:
        get_module.set_old_policy(orig_policy)
        assert get_handler_name() == orig_policy_name

    with pytest.raises(ValueError,
                       match="Capsule must be named 'mem_handler'"):
        get_module.set_wrong_capsule_name_data_policy()


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="no numpy.distutils")
def test_default_policy_singleton(get_module):
    # 略，未提供完整代码
    # 导入NumPy核心多维数组模块中的get_handler_name函数
    get_handler_name = np._core.multiarray.get_handler_name

    # 将策略设置为默认策略，并保存原始策略
    orig_policy = get_module.set_old_policy(None)

    # 断言当前的处理器名称是否为'default_allocator'
    assert get_handler_name() == 'default_allocator'

    # 再次将策略设置为默认策略，并记录返回值作为def_policy_1
    def_policy_1 = get_module.set_old_policy(None)

    # 断言当前的处理器名称是否为'default_allocator'
    assert get_handler_name() == 'default_allocator'

    # 将策略设置回原始策略，并记录返回值作为def_policy_2
    def_policy_2 = get_module.set_old_policy(orig_policy)

    # 由于默认策略是单例，这些对象应当是同一个对象
    assert def_policy_1 is def_policy_2 is get_module.get_default_policy()
@pytest.mark.skipif(sys.version_info >= (3, 12), reason="no numpy.distutils")
# 定义测试函数，用于测试内存策略的传播
def test_policy_propagation(get_module):
    # 定义自定义的 ndarray 子类
    class MyArr(np.ndarray):
        pass

    # 获取当前的处理器名称
    get_handler_name = np._core.multiarray.get_handler_name
    orig_policy_name = get_handler_name()

    # 创建 ndarray，并设置其为 MyArr 类型，再reshape成 (2, 5)
    a = np.arange(10).view(MyArr).reshape((2, 5))

    # 断言：a 的处理器名称应为 None，且 owndata 属性为 False
    assert get_handler_name(a) is None
    assert a.flags.owndata is False

    # 断言：a.base 的处理器名称应为 None，且 owndata 属性为 False
    assert get_handler_name(a.base) is None
    assert a.base.flags.owndata is False

    # 断言：a.base.base 的处理器名称应为 orig_policy_name，且 owndata 属性为 True
    assert get_handler_name(a.base.base) == orig_policy_name
    assert a.base.base.flags.owndata is True


async def concurrent_context1(get_module, orig_policy_name, event):
    # 若原始处理器名称为 'default_allocator'，设置新的数据策略
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        # 断言：当前处理器名称应为 'secret_data_allocator'
        assert get_handler_name() == 'secret_data_allocator'
    else:
        # 否则恢复为旧的数据策略
        get_module.set_old_policy(None)
        # 断言：当前处理器名称应为 'default_allocator'
        assert get_handler_name() == 'default_allocator'
    # 设置事件为完成状态
    event.set()


async def concurrent_context2(get_module, orig_policy_name, event):
    # 等待事件完成
    await event.wait()
    # 断言：当前处理器名称不受并行上下文中的更改影响，应为 orig_policy_name
    assert get_handler_name() == orig_policy_name
    # 在子上下文中改变数据策略
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        # 断言：当前处理器名称应为 'secret_data_allocator'
        assert get_handler_name() == 'secret_data_allocator'
    else:
        get_module.set_old_policy(None)
        # 断言：当前处理器名称应为 'default_allocator'
        assert get_handler_name() == 'default_allocator'


async def async_test_context_locality(get_module):
    # 获取原始的处理器名称
    orig_policy_name = np._core.multiarray.get_handler_name()

    # 创建一个异步事件
    event = asyncio.Event()

    # 子上下文继承父上下文的数据策略
    concurrent_task1 = asyncio.create_task(
        concurrent_context1(get_module, orig_policy_name, event))
    concurrent_task2 = asyncio.create_task(
        concurrent_context2(get_module, orig_policy_name, event))
    # 等待所有并发任务完成
    await concurrent_task1
    await concurrent_task2

    # 父上下文不受子上下文数据策略更改的影响
    assert np._core.multiarray.get_handler_name() == orig_policy_name


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="no numpy.distutils")
# 测试函数：测试上下文的局部性
def test_context_locality(get_module):
    # 若运行环境为 PyPy < 7.3.6，则跳过测试
    if (sys.implementation.name == 'pypy'
            and sys.pypy_version_info[:3] < (7, 3, 6)):
        pytest.skip('no context-locality support in PyPy < 7.3.6')
    # 运行异步测试：测试上下文的局部性
    asyncio.run(async_test_context_locality(get_module))


def concurrent_thread1(get_module, event):
    # 设置秘密数据策略
    get_module.set_secret_data_policy()
    # 断言：当前处理器名称应为 'secret_data_allocator'
    assert np._core.multiarray.get_handler_name() == 'secret_data_allocator'
    # 设置事件为完成状态
    event.set()


def concurrent_thread2(get_module, event):
    # 等待事件完成
    event.wait()
    # 断言：当前处理器名称不受并行线程中的更改影响，应为 'default_allocator'
    assert np._core.multiarray.get_handler_name() == 'default_allocator'
    # 在子线程中改变数据策略
    get_module.set_secret_data_policy()


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="no numpy.distutils")
# 定义一个测试函数，测试线程的本地性
def test_thread_locality(get_module):
    # 获取当前 NumPy 数组的内存分配策略名称
    orig_policy_name = np._core.multiarray.get_handler_name()

    # 创建一个线程事件对象
    event = threading.Event()
    # 创建第一个并发任务线程，目标函数为 concurrent_thread1，传入参数为 get_module 和 event
    concurrent_task1 = threading.Thread(target=concurrent_thread1,
                                        args=(get_module, event))
    # 创建第二个并发任务线程，目标函数为 concurrent_thread2，传入参数为 get_module 和 event
    concurrent_task2 = threading.Thread(target=concurrent_thread2,
                                        args=(get_module, event))
    # 启动第一个并发任务线程
    concurrent_task1.start()
    finally:
        # 最终执行块，无论异常是否发生都会执行
        if oldval is not None:
            # 检查旧的警告值是否存在
            np._core._multiarray_umath._set_numpy_warn_if_no_mem_policy(oldval)
# 使用 pytest.mark.skipif 装饰器，当 Python 版本大于等于 3.12 时跳过测试，原因是不支持 numpy.distutils
@pytest.mark.skipif(sys.version_info >= (3, 12), reason="no numpy.distutils")
# 定义测试函数 test_owner_is_base，接受名为 get_module 的测试夹具
def test_owner_is_base(get_module):
    # 调用 get_module.get_array_with_base() 方法获取数组 a
    a = get_module.get_array_with_base()
    # 使用 pytest.warns 检查是否发出 UserWarning 并匹配 'warn_on_free' 字符串
    with pytest.warns(UserWarning, match='warn_on_free'):
        # 删除数组 a
        del a
        # 执行垃圾回收
        gc.collect()
        # 再次执行垃圾回收
        gc.collect()
```