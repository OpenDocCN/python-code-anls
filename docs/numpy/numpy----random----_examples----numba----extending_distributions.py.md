# `.\numpy\numpy\random\_examples\numba\extending_distributions.py`

```py
"""
在这个示例中，构建所需的库需要NumPy的源代码分发或NumPy的git存储库的克隆，因为distributions.c没有包含在二进制分发中。

在*nix系统上，在numpy/random/src/distributions目录中执行以下操作：

export ${PYTHON_VERSION}=3.8 # Python版本
export PYTHON_INCLUDE=#Python包含文件夹的路径，通常是 \
    ${PYTHON_HOME}/include/python${PYTHON_VERSION}m
export NUMPY_INCLUDE=#NumPy包含文件夹的路径，通常是 \
    ${PYTHON_HOME}/lib/python${PYTHON_VERSION}/site-packages/numpy/_core/include
gcc -shared -o libdistributions.so -fPIC distributions.c \
    -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE}
mv libdistributions.so ../../_examples/numba/

在Windows系统上：

rem PYTHON_HOME和PYTHON_VERSION需要根据具体设置调整，这里是一个示例
set PYTHON_HOME=c:\Anaconda
set PYTHON_VERSION=38
cl.exe /LD .\distributions.c -DDLL_EXPORT \
    -I%PYTHON_HOME%\lib\site-packages\numpy\_core\include \
    -I%PYTHON_HOME%\include %PYTHON_HOME%\libs\python%PYTHON_VERSION%.lib
move distributions.dll ../../_examples/numba/
"""

import os  # 导入操作系统接口模块

import numba as nb  # 导入Numba模块
import numpy as np  # 导入NumPy模块
from cffi import FFI  # 从cffi模块导入FFI类

from numpy.random import PCG64  # 从NumPy的随机模块中导入PCG64随机数生成器

ffi = FFI()  # 创建一个FFI对象
if os.path.exists('./distributions.dll'):  # 检查当前路径下是否存在'distributions.dll'
    lib = ffi.dlopen('./distributions.dll')  # 使用FFI加载'distributions.dll'库
elif os.path.exists('./libdistributions.so'):  # 如果不存在'distributions.dll'，则检查是否存在'libdistributions.so'
    lib = ffi.dlopen('./libdistributions.so')  # 使用FFI加载'libdistributions.so'库
else:
    raise RuntimeError('Required DLL/so file was not found.')  # 如果都不存在，则抛出运行时错误

ffi.cdef("""
double random_standard_normal(void *bitgen_state);
""")
# 定义一个C函数原型，声明'read_random_standard_normal'函数，接收一个void*参数并返回double类型

x = PCG64()  # 创建一个PCG64随机数生成器对象
xffi = x.cffi  # 获取PCG64对象的CFFI接口
bit_generator = xffi.bit_generator  # 获取PCG64对象的bit_generator属性

random_standard_normal = lib.random_standard_normal  # 将从库中加载的random_standard_normal函数赋值给变量

def normals(n, bit_generator):
    # 定义一个函数normals，接收n和bit_generator参数
    out = np.empty(n)  # 创建一个大小为n的空NumPy数组out
    for i in range(n):
        out[i] = random_standard_normal(bit_generator)  # 使用库中的random_standard_normal函数填充数组out
    return out  # 返回填充好的数组out

normalsj = nb.jit(normals, nopython=True)  # 使用Numba对normals函数进行即时编译，要求nopython模式

# Numba需要一个void*指针的内存地址
# 也可以从x.ctypes.bit_generator.value获取地址
bit_generator_address = int(ffi.cast('uintptr_t', bit_generator))  # 将bit_generator转换为uintptr_t类型的整数地址

norm = normalsj(1000, bit_generator_address)  # 调用编译后的normalsj函数，传入参数1000和bit_generator_address
print(norm[:12])  # 打印数组norm的前12个元素
```