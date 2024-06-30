# `D:\src\scipysrc\sympy\sympy\utilities\_compilation\tests\test_compilation.py`

```
# 导入 shutil 模块，用于高级文件操作
import shutil
# 导入 os 模块，提供了与操作系统交互的功能
import os
# 导入 subprocess 模块，允许创建子进程，执行外部命令
import subprocess
# 导入 tempfile 模块，用于创建临时文件和目录
import tempfile
# 从 sympy.external 模块中导入 import_module 函数
from sympy.external import import_module
# 从 sympy.testing.pytest 模块中导入 skip 函数，用于跳过测试
from sympy.testing.pytest import skip

# 从 sympy.utilities._compilation.compilation 模块中导入相关函数和类
from sympy.utilities._compilation.compilation import compile_link_import_py_ext, compile_link_import_strings, compile_sources, get_abspath

# 使用 import_module 导入 'numpy' 库
numpy = import_module('numpy')
# 使用 import_module 导入 'cython' 库
cython = import_module('cython')

# 定义 _sources1 列表，包含 C 语言源码和 Cython 源码的元组
_sources1 = [
    ('sigmoid.c', r"""
#include <math.h>

void sigmoid(int n, const double * const restrict in,
             double * const restrict out, double lim){
    for (int i=0; i<n; ++i){
        const double x = in[i];
        out[i] = x*pow(pow(x/lim, 8)+1, -1./8.);
    }
}
"""),
    ('_sigmoid.pyx', r"""
import numpy as np
cimport numpy as cnp

cdef extern void c_sigmoid "sigmoid" (int, const double * const,
                                      double * const, double)

def sigmoid(double [:] inp, double lim=350.0):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(
        inp.size, dtype=np.float64)
    c_sigmoid(inp.size, &inp[0], &out[0], lim)
    return out
""")
]

# 定义 npy 函数，实现给定公式的数学运算
def npy(data, lim=350.0):
    return data/((data/lim)**8+1)**(1/8.)

# 定义测试函数 test_compile_link_import_strings
def test_compile_link_import_strings():
    # 如果 numpy 模块未安装，则跳过测试
    if not numpy:
        skip("numpy not installed.")
    # 如果 cython 模块未安装，则跳过测试
    if not cython:
        skip("cython not installed.")

    # 从 sympy.utilities._compilation 模块中导入 has_c 函数
    from sympy.utilities._compilation import has_c
    # 如果系统中没有 C 编译器，则跳过测试
    if not has_c():
        skip("No C compiler found.")

    # 定义编译选项 compile_kw 字典
    compile_kw = {"std": 'c99', "include_dirs": [numpy.get_include()]}
    info = None
    try:
        # 调用 compile_link_import_strings 函数进行编译、链接和导入
        mod, info = compile_link_import_strings(_sources1, compile_kwargs=compile_kw)
        # 创建一个随机数组，模拟 64 MB 的数据
        data = numpy.random.random(1024*1024*8)  # 64 MB of RAM needed..
        # 使用编译得到的模块计算 sigmoid 函数的结果
        res_mod = mod.sigmoid(data)
        # 使用 npy 函数计算预期的 sigmoid 结果
        res_npy = npy(data)
        # 断言编译得到的结果与预期结果非常接近
        assert numpy.allclose(res_mod, res_npy)
    finally:
        # 如果 info 不为空且包含 'build_dir' 字段，则删除临时建立的目录
        if info and info['build_dir']:
            shutil.rmtree(info['build_dir'])

# 定义测试函数 test_compile_sources，接收一个 tmpdir 参数
def test_compile_sources(tmpdir):
    # 从 sympy.utilities._compilation 模块中导入 has_c 函数
    from sympy.utilities._compilation import has_c
    # 如果系统中没有 C 编译器，则跳过测试
    if not has_c():
        skip("No C compiler found.")

    # 将 tmpdir 转换为字符串类型，并作为 build_dir 的路径
    build_dir = str(tmpdir)
    # 使用 tempfile.mkstemp 在 build_dir 目录下创建一个 C 源文件
    _handle, file_path = tempfile.mkstemp('.c', dir=build_dir)
    with open(file_path, 'wt') as ofh:
        # 将简单的 C 函数写入文件中
        ofh.write("""
        int foo(int bar) {
            return 2*bar;
        }
        """)
    # 编译指定的 C 源文件，返回编译后的对象文件列表
    obj, = compile_sources([file_path], cwd=build_dir)
    # 获取编译得到的对象文件的绝对路径
    obj_path = get_abspath(obj, cwd=build_dir)
    # 断言编译得到的对象文件确实存在
    assert os.path.exists(obj_path)
    try:
        # 使用 subprocess 调用 'nm' 命令，检查对象文件的符号表信息
        _ = subprocess.check_output(["nm", "--help"])
    except subprocess.CalledProcessError:
        pass  # 无法测试对象文件的内容
    else:
        # 如果 'nm' 命令执行成功，读取并检查对象文件的符号表输出
        nm_out = subprocess.check_output(["nm", obj_path])
        assert 'foo' in nm_out.decode('utf-8')

    # 如果 cython 模块未安装，则直接返回，不继续执行下面的测试
    if not cython:
        return  # the final (optional) part of the test below requires Cython.

    # 使用 tempfile.mkstemp 在 build_dir 目录下创建一个 Cython 源文件
    _handle, pyx_path = tempfile.mkstemp('.pyx', dir=build_dir)
    with open(pyx_path, 'wt') as ofh:
        # 将 Cython 源码写入文件中，调用外部定义的 C 函数
        ofh.write(("cdef extern int foo(int)\n"
                   "def _foo(arg):\n"
                   "    return foo(arg)"))
    # 使用自定义函数 compile_link_import_py_ext 编译、链接和导入给定的 Pyrex 文件（pyx_path），并指定额外的目标文件（obj_path）和构建目录（build_dir）
    mod = compile_link_import_py_ext([pyx_path], extra_objs=[obj_path], build_dir=build_dir)
    # 断言检查 mod 对象的 _foo 方法对参数 21 的调用结果是否为 42
    assert mod._foo(21) == 42
```