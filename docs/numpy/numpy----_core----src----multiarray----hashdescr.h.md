# `.\numpy\numpy\_core\src\multiarray\hashdescr.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_HASHDESCR_H_
#define NUMPY_CORE_SRC_MULTIARRAY_HASHDESCR_H_

# 定义条件编译指令，防止重复包含该头文件
NPY_NO_EXPORT npy_hash_t
# 函数声明：计算给定对象的描述符的哈希值，并返回
PyArray_DescrHash(PyObject* odescr);

# 结束条件编译指令
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_HASHDESCR_H_ */
```