# `.\numpy\numpy\_core\src\umath\extobj.h`

```py
#ifndef _NPY_PRIVATE__EXTOBJ_H_
#define _NPY_PRIVATE__EXTOBJ_H_

#include <numpy/ndarraytypes.h>  /* for NPY_NO_EXPORT */

/*
 * Represent the current ufunc error (and buffer) state.  we are using a
 * capsule for now to store this, but it could make sense to refactor it into
 * a proper (immutable) object.
 * NOTE: Part of this information should be integrated into the public API
 *       probably.  We expect extending it e.g. with a "fast" flag.
 *       (although the public only needs to know *if* errors are checked, not
 *       what we do with them, like warn, raise, ...).
 */
typedef struct {
    int errmask;                // 错误掩码，用于表示错误状态
    npy_intp bufsize;           // 缓冲区大小，用于表示缓冲区状态
    PyObject *pyfunc;           // Python 对象，用于存储相关函数对象
} npy_extobj;

/* Clearing is only `pyfunc` XDECREF, but could grow in principle */
static inline void
npy_extobj_clear(npy_extobj *extobj)
{
    Py_XDECREF(extobj->pyfunc); // 释放 extobj 结构体中的 pyfunc 对象
}

NPY_NO_EXPORT int
_check_ufunc_fperr(int errmask, const char *ufunc_name);

NPY_NO_EXPORT int
_get_bufsize_errmask(int *buffersize, int *errormask);

NPY_NO_EXPORT int
init_extobj(void);   // 初始化 extobj 结构体

/*
 * Private Python exposure of the extobj.
 */
NPY_NO_EXPORT PyObject *
extobj_make_extobj(PyObject *NPY_UNUSED(mod),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames);

NPY_NO_EXPORT PyObject *
extobj_get_extobj_dict(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(noarg));

#endif


这段代码是一个 C 语言头文件，主要定义了一个结构体 `npy_extobj` 和若干函数。注释解释了结构体字段的含义，以及每个函数的作用和用途。
```