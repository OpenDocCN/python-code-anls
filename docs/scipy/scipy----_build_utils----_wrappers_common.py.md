# `D:\src\scipysrc\scipy\scipy\_build_utils\_wrappers_common.py`

```
# 用于生成 BLAS/LAPACK 包装器的辅助函数和变量。

import os  # 导入操作系统模块
from stat import ST_MTIME  # 导入 stat 模块中的 ST_MTIME 常量

# 用于将签名文件中的类型转换为 C 类型的映射字典
C_TYPES = {'int': 'int',
           'c': 'npy_complex64',
           'd': 'double',
           's': 'float',
           'z': 'npy_complex128',
           'char': 'char',
           'bint': 'int',
           'void': 'void',
           'cselect1': '_cselect1',
           'cselect2': '_cselect2',
           'dselect2': '_dselect2',
           'dselect3': '_dselect3',
           'sselect2': '_sselect2',
           'sselect3': '_sselect3',
           'zselect1': '_zselect1',
           'zselect2': '_zselect2'}

# 用于将签名文件中的复杂类型转换为 Numpy 复杂类型的映射字典
NPY_TYPES = {'c': 'npy_complex64', 'z': 'npy_complex128',
             'cselect1': '_cselect1', 'cselect2': '_cselect2',
             'dselect2': '_dselect2', 'dselect3': '_dselect3',
             'sselect2': '_sselect2', 'sselect3': '_sselect3',
             'zselect1': '_zselect1', 'zselect2': '_zselect2'}

# 具有复杂返回值的 BLAS/LAPACK 函数列表，使用 G77 ABI 封装器中的 'wrp' 后缀包装器
WRAPPED_FUNCS = ['cdotc', 'cdotu', 'zdotc', 'zdotu', 'cladiv', 'zladiv']

# 在新的 Accelerate 库中缺失的函数列表，因此使用旧版本 Accelerate 库中的标准符号
USE_OLD_ACCELERATE = ['lsame', 'dcabs1']

# C 预处理器的前言内容，包括一些头文件的引入
C_PREAMBLE = """
#include "npy_cblas.h"
#include "fortran_defs.h"
"""

# LAPACK 声明部分，定义了一些特定的函数指针类型
LAPACK_DECLS = """
typedef int (*_cselect1)(npy_complex64*);
typedef int (*_cselect2)(npy_complex64*, npy_complex64*);
typedef int (*_dselect2)(double*, double*);
typedef int (*_dselect3)(double*, double*, double*);
typedef int (*_sselect2)(float*, float*);
typedef int (*_sselect3)(float*, float*, float*);
typedef int (*_zselect1)(npy_complex128*);
typedef int (*_zselect2)(npy_complex128*, npy_complex128*);
"""

# C++ 的头文件保护开始部分，用于在 C++ 编译时防止头文件被多次包含
CPP_GUARD_BEGIN = """
#ifdef __cplusplus
extern "C" {
#endif
"""

# C++ 的头文件保护结束部分
CPP_GUARD_END = """
#ifdef __cplusplus
}
#endif
"""

def read_signatures(lines):
    """
    读取 BLAS/LAPACK 签名文件，并将其拆分为函数名、返回类型、参数名和参数类型。
    """
    sigs = []
    for line in lines:
        line = line.strip()  # 去除字符串首尾的空白字符
        if not line or line.startswith('#'):  # 如果是空行或注释行则跳过
            continue
        line = line[:-1].split('(')  # 去掉末尾的括号，然后按括号分割
        args = line[1]  # 获取参数部分
        name_and_type = line[0].split(' ')  # 分割得到函数名和返回类型
        ret_type = name_and_type[0]  # 返回类型
        name = name_and_type[1]  # 函数名
        # 按逗号和空格分割参数列表，并获取参数类型和参数名
        argtypes, argnames = zip(*[arg.split(' *') for arg in args.split(', ')])
        # 如果参数名与返回类型相同，则修改参数名以避免冲突
        if ret_type in argnames:
            argnames = [n if n != ret_type else n + '_' for n in argnames]
        # 如果参数名是 Python 关键字，则修改参数名
        argnames = [n if n not in ['lambda', 'in'] else n + '_' for n in argnames]
        sigs.append({
            'name': name,
            'return_type': ret_type,
            'argnames': argnames,
            'argtypes': list(argtypes)
        })
    return sigs
# 如果目标文件 'dst' 不存在，则抛出 ValueError 异常，指出文件不存在
def newer(dst, src):
    # 如果目标文件 'dst' 不存在
    if not os.path.exists(dst):
        # 抛出异常，指出文件不存在，并显示文件的绝对路径
        raise ValueError("file '%s' does not exist" % os.path.abspath(dst))
    
    # 如果源文件 'src' 不存在，则返回 1，表示 'dst' 比 'src' 更新
    if not os.path.exists(src):
        return 1
    
    # 获取目标文件 'dst' 和源文件 'src' 的最后修改时间
    mtime1 = os.stat(dst)[ST_MTIME]
    mtime2 = os.stat(src)[ST_MTIME]
    
    # 返回比较结果，True 表示 'dst' 比 'src' 更新，False 表示 'dst' 和 'src' 同时存在且 'dst' 不如 'src' 更新
    return mtime1 > mtime2


# 检查所有目标文件是否都存在且都比对应的源文件更新
def all_newer(dst_files, src_files):
    """True only if all dst_files exist and are newer than all src_files."""
    # 使用生成器表达式检查所有目标文件和对应源文件的更新情况
    return all(os.path.exists(dst) and newer(dst, src)
               for dst in dst_files for src in src_files)


# 获取 BLAS 宏和函数名
def get_blas_macro_and_name(name, accelerate):
    """Complex-valued and some Accelerate functions have special symbols."""
    # 如果使用 Accelerate
    if accelerate:
        # 如果函数名在 USE_OLD_ACCELERATE 中
        if name in USE_OLD_ACCELERATE:
            # 返回空字符串和函数名加上下划线后缀
            return '', f'{name}_'
        # 如果函数名是 'xerbla_array'
        elif name == 'xerbla_array':
            # 返回空字符串和函数名加上双下划线后缀
            return '', name + '__'
    
    # 如果函数名在 WRAPPED_FUNCS 中
    if name in WRAPPED_FUNCS:
        # 将函数名加上 'wrp' 后缀
        name = name + 'wrp'
        # 返回宏 'F_FUNC' 和大写后的函数名以及小写后的函数名
        return 'F_FUNC', f'{name},{name.upper()}'
    
    # 默认情况返回宏 'BLAS_FUNC' 和函数名
    return 'BLAS_FUNC', name


# 将文件内容写入指定路径的文件
def write_files(file_dict):
    """
    Takes a mapping of full filepath to file contents to write at that path.
    """
    # 遍历文件路径和内容的映射字典
    for file_path, content in file_dict.items():
        # 使用 'w' 模式打开文件
        with open(file_path, 'w') as f:
            # 写入文件内容
            f.write(content)
```