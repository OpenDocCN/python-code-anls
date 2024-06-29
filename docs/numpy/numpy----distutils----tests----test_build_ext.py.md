# `.\numpy\numpy\distutils\tests\test_build_ext.py`

```
'''Tests for numpy.distutils.build_ext.'''

import os  # 导入操作系统接口模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关模块
from textwrap import indent, dedent  # 导入文本包装模块中的缩进和去除缩进函数
import pytest  # 导入 pytest 测试框架
from numpy.testing import IS_WASM  # 导入 IS_WASM 布尔标志

@pytest.mark.skipif(IS_WASM, reason="cannot start subprocess in wasm")  # 标记：如果在 WASM 环境下，则跳过测试
@pytest.mark.slow  # 标记：标记为慢速测试
def test_multi_fortran_libs_link(tmp_path):
    '''
    Ensures multiple "fake" static libraries are correctly linked.
    see gh-18295
    '''

    # We need to make sure we actually have an f77 compiler.
    # This is nontrivial, so we'll borrow the utilities
    # from f2py tests:
    from numpy.distutils.tests.utilities import has_f77_compiler  # 导入 f2py 测试中的 Fortran 77 编译器检查函数
    if not has_f77_compiler():
        pytest.skip('No F77 compiler found')  # 如果没有找到 F77 编译器，则跳过测试

    # make some dummy sources
    with open(tmp_path / '_dummy1.f', 'w') as fid:
        fid.write(indent(dedent('''\
            FUNCTION dummy_one()
            RETURN
            END FUNCTION'''), prefix=' '*6))  # 写入第一个 Fortran 源文件 _dummy1.f

    with open(tmp_path / '_dummy2.f', 'w') as fid:
        fid.write(indent(dedent('''\
            FUNCTION dummy_two()
            RETURN
            END FUNCTION'''), prefix=' '*6))  # 写入第二个 Fortran 源文件 _dummy2.f

    with open(tmp_path / '_dummy.c', 'w') as fid:
        fid.write('int PyInit_dummyext;')  # 写入一个简单的 C 源文件 _dummy.c

    # make a setup file
    with open(tmp_path / 'setup.py', 'w') as fid:
        srctree = os.path.join(os.path.dirname(__file__), '..', '..', '..')  # 获取源树路径
        fid.write(dedent(f'''\
            def configuration(parent_package="", top_path=None):
                from numpy.distutils.misc_util import Configuration
                config = Configuration("", parent_package, top_path)
                config.add_library("dummy1", sources=["_dummy1.f"])
                config.add_library("dummy2", sources=["_dummy2.f"])
                config.add_extension("dummyext", sources=["_dummy.c"], libraries=["dummy1", "dummy2"])
                return config


            if __name__ == "__main__":
                import sys
                sys.path.insert(0, r"{srctree}")
                from numpy.distutils.core import setup
                setup(**configuration(top_path="").todict())'''))  # 写入配置 setup.py

    # build the test extension and "install" into a temporary directory
    build_dir = tmp_path
    subprocess.check_call([sys.executable, 'setup.py', 'build', 'install',
                           '--prefix', str(tmp_path / 'installdir'),
                           '--record', str(tmp_path / 'tmp_install_log.txt'),
                          ],  # 调用子进程执行构建和安装命令
                          cwd=str(build_dir),
                      )

    # get the path to the so
    so = None
    with open(tmp_path / 'tmp_install_log.txt') as fid:
        for line in fid:
            if 'dummyext' in line:
                so = line.strip()  # 获取安装日志中的共享对象文件路径
                break
    assert so is not None  # 断言：确保找到共享对象文件路径
```