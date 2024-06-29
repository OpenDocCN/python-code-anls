# `.\numpy\numpy\_core\src\common\npy_hashtable.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_

#include <Python.h>

// 定义宏，指定使用当前版本的 NumPy API，禁止使用过时的 API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 包含 NumPy 的 ndarraytypes.h 文件
#include "numpy/ndarraytypes.h"

// 定义一个结构体 PyArrayIdentityHash，用于实现身份哈希表
typedef struct {
    int key_len;  /* number of identities used */  // 关键字的长度，即使用的标识数目
    /* Buckets stores: val1, key1[0], key1[1], ..., val2, key2[0], ... */
    PyObject **buckets;  // 存储桶，按顺序存储值和对应的键
    npy_intp size;  /* current size */  // 当前大小
    npy_intp nelem;  /* number of elements */  // 元素的数量
#ifdef Py_GIL_DISABLED
    PyThread_type_lock *mutex;  // 线程锁，如果 GIL 被禁用
#endif
} PyArrayIdentityHash;  // PyArrayIdentityHash 结构体声明结束

// 以下为函数声明

// 向身份哈希表中设置元素，如果已存在则替换
NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *value, int replace);

// 从身份哈希表中获取元素
NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash const *tb, PyObject *const *key);

// 创建并返回一个新的身份哈希表
NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len);

// 释放身份哈希表及其资源
NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb);

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_ */
```