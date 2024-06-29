# `.\numpy\numpy\distutils\tests\utilities.py`

```py
# 从numpy.f2py.tests.util导入IS_WASM，用于测试build_ext
from numpy.testing import IS_WASM
# 导入textwrap模块，用于处理文本格式
import textwrap
# 导入shutil模块，用于文件和目录操作
import shutil
# 导入tempfile模块，用于创建临时文件和目录
import tempfile
# 导入os模块，提供了与操作系统交互的功能
import os
# 导入re模块，提供正则表达式操作
import re
# 导入subprocess模块，用于启动新进程并与其交互
import subprocess
# 导入sys模块，提供了对Python解释器的访问
import sys

#
# 检查编译器是否可用...
#

# 全局变量，用于缓存编译器状态
_compiler_status = None

# 获取编译器状态的函数
def _get_compiler_status():
    global _compiler_status
    # 如果已经获取过编译器状态，则直接返回缓存的状态
    if _compiler_status is not None:
        return _compiler_status

    # 默认编译器状态设置为(False, False, False)
    _compiler_status = (False, False, False)
    # 如果运行在WASM环境下，则无法运行编译器
    if IS_WASM:
        # 返回当前编译器状态
        return _compiler_status

    # 准备需要执行的Python代码，用于测试编译器
    code = textwrap.dedent(
        f"""\
        import os
        import sys
        sys.path = {repr(sys.path)}

        def configuration(parent_name='',top_path=None):
            global config
            from numpy.distutils.misc_util import Configuration
            config = Configuration('', parent_name, top_path)
            return config

        from numpy.distutils.core import setup
        setup(configuration=configuration)

        config_cmd = config.get_config_cmd()
        have_c = config_cmd.try_compile('void foo() {{}}')
        print('COMPILERS:%%d,%%d,%%d' %% (have_c,
                                          config.have_f77c(),
                                          config.have_f90c()))
        sys.exit(99)
        """
    )
    code = code % dict(syspath=repr(sys.path))

    # 创建临时目录
    tmpdir = tempfile.mkdtemp()
    try:
        script = os.path.join(tmpdir, "setup.py")

        # 将测试代码写入临时文件
        with open(script, "w") as f:
            f.write(code)

        # 准备执行命令：运行setup.py脚本以测试编译器
        cmd = [sys.executable, "setup.py", "config"]
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=tmpdir
        )
        # 获取命令执行结果
        out, err = p.communicate()
    finally:
        # 删除临时目录及其内容
        shutil.rmtree(tmpdir)

    # 从命令输出中匹配编译器测试结果
    m = re.search(rb"COMPILERS:(\d+),(\d+),(\d+)", out)
    if m:
        _compiler_status = (
            bool(int(m.group(1))),  # 是否有C编译器
            bool(int(m.group(2))),  # 是否有Fortran 77编译器
            bool(int(m.group(3))),  # 是否有Fortran 90编译器
        )
    # 返回最终的编译器状态
    return _compiler_status


# 检查是否有C编译器可用
def has_c_compiler():
    return _get_compiler_status()[0]


# 检查是否有Fortran 77编译器可用
def has_f77_compiler():
    return _get_compiler_status()[1]


# 检查是否有Fortran 90编译器可用
def has_f90_compiler():
    return _get_compiler_status()[2]
```