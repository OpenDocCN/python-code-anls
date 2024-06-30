# `D:\src\scipysrc\scikit-learn\sklearn\utils\_vector_sentinel.pxd`

```
# 导入Cython生成的NumPy接口模块
cimport numpy as cnp

# 导入C++ STL中的vector模板类
from libcpp.vector cimport vector

# 从上级目录中导入自定义类型定义
from ..utils._typedefs cimport intp_t, float64_t, int32_t, int64_t

# 定义一个融合类型，可以是float64_t、intp_t、int32_t或int64_t类型的vector
ctypedef fused vector_typed:
    vector[float64_t]
    vector[intp_t]
    vector[int32_t]
    vector[int64_t]

# 声明一个Cython函数，将C++ STL的vector转换为NumPy的ndarray
cdef cnp.ndarray vector_to_nd_array(vector_typed * vect_ptr)
```