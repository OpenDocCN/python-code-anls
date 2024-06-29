# `.\numpy\numpy\distutils\tests\test_fcompiler.py`

```py
# 从 numpy.testing 模块导入 assert_ 函数
from numpy.testing import assert_
# 从 numpy.distutils.fcompiler 模块导入
import numpy.distutils.fcompiler

# 定义可定制的编译器标志列表，每个元素为元组，包含编译器选项和对应的环境变量名
customizable_flags = [
    ('f77', 'F77FLAGS'),
    ('f90', 'F90FLAGS'),
    ('free', 'FREEFLAGS'),
    ('arch', 'FARCH'),
    ('debug', 'FDEBUG'),
    ('flags', 'FFLAGS'),
    ('linker_so', 'LDFLAGS'),
]

# 定义测试函数 test_fcompiler_flags，使用 monkeypatch 参数
def test_fcompiler_flags(monkeypatch):
    # 设置环境变量 NPY_DISTUTILS_APPEND_FLAGS 为 '0'
    monkeypatch.setenv('NPY_DISTUTILS_APPEND_FLAGS', '0')
    # 调用 numpy.distutils.fcompiler.new_fcompiler 创建一个新的编译器对象 fc
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='none')
    # 使用 flag_vars 对象的 clone 方法创建一个副本，lambda 函数将所有参数传递给 None
    flag_vars = fc.flag_vars.clone(lambda *args, **kwargs: None)

    # 遍历可定制的编译器标志列表
    for opt, envvar in customizable_flags:
        # 创建一个新的标志 '-dummy-<opt>-flag'
        new_flag = '-dummy-{}-flag'.format(opt)
        # 获取 flag_vars 对象中当前选项 opt 的值
        prev_flags = getattr(flag_vars, opt)

        # 设置环境变量 envvar 为 new_flag
        monkeypatch.setenv(envvar, new_flag)
        # 获取更新后的 flag_vars 对象中选项 opt 的值
        new_flags = getattr(flag_vars, opt)

        # 删除环境变量 envvar
        monkeypatch.delenv(envvar)
        # 断言更新后的选项值 new_flags 与预期值 [new_flag] 相等
        assert_(new_flags == [new_flag])

    # 设置环境变量 NPY_DISTUTILS_APPEND_FLAGS 为 '1'
    monkeypatch.setenv('NPY_DISTUTILS_APPEND_FLAGS', '1')

    # 再次遍历可定制的编译器标志列表
    for opt, envvar in customizable_flags:
        # 创建一个新的标志 '-dummy-<opt>-flag'
        new_flag = '-dummy-{}-flag'.format(opt)
        # 获取 flag_vars 对象中当前选项 opt 的值
        prev_flags = getattr(flag_vars, opt)
        # 设置环境变量 envvar 为 new_flag
        monkeypatch.setenv(envvar, new_flag)
        # 获取更新后的 flag_vars 对象中选项 opt 的值
        new_flags = getattr(flag_vars, opt)

        # 删除环境变量 envvar
        monkeypatch.delenv(envvar)
        # 如果原始选项值 prev_flags 为 None，则断言更新后的选项值 new_flags 等于 [new_flag]
        if prev_flags is None:
            assert_(new_flags == [new_flag])
        else:
            # 否则，断言更新后的选项值 new_flags 等于原始选项值 prev_flags 加上 [new_flag]
            assert_(new_flags == prev_flags + [new_flag])
```