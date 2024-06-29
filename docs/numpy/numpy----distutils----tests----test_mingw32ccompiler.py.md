# `.\numpy\numpy\distutils\tests\test_mingw32ccompiler.py`

```
import shutil`
import shutil  # 导入 shutil 模块，用于文件和目录操作
import subprocess  # 导入 subprocess 模块，用于执行外部命令和获取输出
import sys  # 导入 sys 模块，用于访问系统相关的参数和功能
import pytest  # 导入 pytest 模块，用于编写和运行测试用例

from numpy.distutils import mingw32ccompiler  # 从 numpy.distutils 中导入 mingw32ccompiler 模块

@pytest.mark.skipif(sys.platform != 'win32', reason='win32 only test')
def test_build_import():
    '''Test the mingw32ccompiler.build_import_library, which builds a
    `python.a` from the MSVC `python.lib`
    '''
    
    # 确保 `nm.exe` 存在并且支持当前的 Python 版本。当 PATH 中混杂了 64 位的 nm，而 Python 是 32 位时，可能会出错
    try:
        out = subprocess.check_output(['nm.exe', '--help'])
    except FileNotFoundError:
        pytest.skip("'nm.exe' not on path, is mingw installed?")
    
    # 提取出 `nm.exe` 输出中的支持的目标格式信息
    supported = out[out.find(b'supported targets:'):]
    
    # 根据 Python 的位数判断 `nm.exe` 是否支持相应的格式
    if sys.maxsize < 2**32 when using 32-bit python. Supported "
                             "formats: '%s'" % supported)
    elif b'pe-x86-64' not in supported:
        raise ValueError("'nm.exe' found but it does not support 64-bit "
                         "dlls when using 64-bit python. Supported "
                         "formats: '%s'" % supported)
    
    # 隐藏导入库以强制重新构建
    has_import_lib, fullpath = mingw32ccompiler._check_for_import_lib()
    if has_import_lib:
        shutil.move(fullpath, fullpath + '.bak')

    try:
        # 真正执行函数测试
        mingw32ccompiler.build_import_library()

    finally:
        # 恢复隐藏的导入库
        if has_import_lib:
            shutil.move(fullpath + '.bak', fullpath)
```