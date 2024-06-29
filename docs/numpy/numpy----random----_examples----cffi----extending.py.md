# `.\numpy\numpy\random\_examples\cffi\extending.py`

```
"""
Use cffi to access any of the underlying C functions from distributions.h
"""
# 导入必要的库
import os              # 用于操作系统相关功能的标准库
import numpy as np     # 导入NumPy库，用于科学计算
import cffi            # 导入cffi库，用于调用C语言函数接口
from .parse import parse_distributions_h  # 导入自定义函数parse_distributions_h，用于解析distributions.h文件

ffi = cffi.FFI()       # 创建一个FFI对象

# 获取NumPy的头文件所在目录
inc_dir = os.path.join(np.get_include(), 'numpy')

# 定义基本的NumPy类型
ffi.cdef('''
    typedef intptr_t npy_intp;
    typedef unsigned char npy_bool;
''')

# 调用自定义函数，解析distributions.h文件并注册C函数到FFI对象
parse_distributions_h(ffi, inc_dir)

# 使用FFI对象加载NumPy随机模块中的C函数库
lib = ffi.dlopen(np.random._generator.__file__)

# 比较distributions.h中的random_standard_normal_fill函数与Generator.standard_random函数
bit_gen = np.random.PCG64()  # 创建PCG64位生成器实例
rng = np.random.Generator(bit_gen)  # 创建随机数生成器实例
state = bit_gen.state  # 获取生成器当前状态

interface = rng.bit_generator.cffi  # 获取随机数生成器的CFFI接口
n = 100  # 设定生成随机数的数量

# 创建一个C类型的double数组，用于存储从C函数中获取的随机数
vals_cffi = ffi.new('double[%d]' % n)

# 调用C函数库中的random_standard_normal_fill函数，生成随机数填充到vals_cffi数组中
lib.random_standard_normal_fill(interface.bit_generator, n, vals_cffi)

# 重置生成器状态
bit_gen.state = state

# 使用Python接口生成相同数量的标准正态分布随机数
vals = rng.standard_normal(n)

# 对比两种方法生成的随机数是否相等
for i in range(n):
    assert vals[i] == vals_cffi[i]
```