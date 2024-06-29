# `.\numpy\numpy\f2py\src\fortranobject.h`

```
#ifndef Py_FORTRANOBJECT_H
#define Py_FORTRANOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#endif

#ifdef FORTRANOBJECT_C
#define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL _npy_f2py_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"

#ifdef F2PY_REPORT_ATEXIT
#include <sys/timeb.h>
// clang-format off
extern void f2py_start_clock(void);
extern void f2py_stop_clock(void);
extern void f2py_start_call_clock(void);
extern void f2py_stop_call_clock(void);
extern void f2py_cb_start_clock(void);
extern void f2py_cb_stop_clock(void);
extern void f2py_cb_start_call_clock(void);
extern void f2py_cb_stop_call_clock(void);
extern void f2py_report_on_exit(int, void *);
// clang-format on
#endif

#ifdef DMALLOC
#include "dmalloc.h"
#endif

/* Fortran object interface */

/*
123456789-123456789-123456789-123456789-123456789-123456789-123456789-12

PyFortranObject represents various Fortran objects:
Fortran (module) routines, COMMON blocks, module data.

Author: Pearu Peterson <pearu@cens.ioc.ee>
*/

#define F2PY_MAX_DIMS 40
#define F2PY_MESSAGE_BUFFER_SIZE 300  // Increase on "stack smashing detected"

typedef void (*f2py_set_data_func)(char *, npy_intp *);
typedef void (*f2py_void_func)(void);
typedef void (*f2py_init_func)(int *, npy_intp *, f2py_set_data_func, int *);

/*typedef void* (*f2py_c_func)(void*,...);*/

typedef void *(*f2pycfunc)(void);

typedef struct {
    char *name; /* attribute (array||routine) name */
    int rank;   /* array rank, 0 for scalar, max is F2PY_MAX_DIMS,
                   || rank=-1 for Fortran routine */
    struct {
        npy_intp d[F2PY_MAX_DIMS];
    } dims;              /* dimensions of the array, || not used */
    int type;            /* PyArray_<type> || not used */
    int elsize;                /* Element size || not used */
    char *data;          /* pointer to array || Fortran routine */
    f2py_init_func func; /* initialization function for
                            allocatable arrays:
                            func(&rank,dims,set_ptr_func,name,len(name))
                            || C/API wrapper for Fortran routine */
    char *doc;           /* documentation string; only recommended
                            for routines. */
} FortranDataDef;

typedef struct {
    PyObject_HEAD
    int len;              /* Number of attributes */
    FortranDataDef *defs; /* An array of FortranDataDef's */
    PyObject *dict;       /* Fortran object attribute dictionary */
} PyFortranObject;

#define PyFortran_Check(op) (Py_TYPE(op) == &PyFortran_Type)
#define PyFortran_Check1(op) (0 == strcmp(Py_TYPE(op)->tp_name, "fortran"))

extern PyTypeObject PyFortran_Type;
extern int
F2PyDict_SetItemString(PyObject *dict, char *name, PyObject *obj);
extern PyObject *
PyFortranObject_New(FortranDataDef *defs, f2py_void_func init);
extern PyObject *

#endif  // Py_FORTRANOBJECT_H
/* 定义一个函数原型，用于创建一个 PyFortranObject 作为属性 */
PyFortranObject_NewAsAttr(FortranDataDef *defs);

/* 
   创建一个 Python Capsule 对象，将一个 void 指针封装起来，并指定一个析构函数。
   这个 Capsule 对象可以用来在 Python 和 C 之间传递指针。
*/
PyObject *
F2PyCapsule_FromVoidPtr(void *ptr, void (*dtor)(PyObject *));

/* 
   从 Python Capsule 对象中获取 void 指针。
   这个函数用于从 Python 中获取在 Capsule 中封装的原始指针。
*/
void *
F2PyCapsule_AsVoidPtr(PyObject *obj);

/* 
   检查一个对象是否是 F2Py Capsule 类型。
   这个宏用于确定一个对象是否是有效的 Capsule 对象。
*/
int
F2PyCapsule_Check(PyObject *ptr);

/* 
   在多线程环境中，用于设置线程本地的回调指针。
*/
extern void *
F2PySwapThreadLocalCallbackPtr(char *key, void *ptr);

/* 
   在多线程环境中，用于获取线程本地的回调指针。
*/
extern void *
F2PyGetThreadLocalCallbackPtr(char *key);

/* 定义一个宏，用于检查一个数组是否是 C 连续存储的 */
#define ISCONTIGUOUS(m) (PyArray_FLAGS(m) & NPY_ARRAY_C_CONTIGUOUS)

/* 定义一系列常量，用于描述参数的意图 */
#define F2PY_INTENT_IN 1
#define F2PY_INTENT_INOUT 2
#define F2PY_INTENT_OUT 4
#define F2PY_INTENT_HIDE 8
#define F2PY_INTENT_CACHE 16
#define F2PY_INTENT_COPY 32
#define F2PY_INTENT_C 64
#define F2PY_OPTIONAL 128
#define F2PY_INTENT_INPLACE 256
#define F2PY_INTENT_ALIGNED4 512
#define F2PY_INTENT_ALIGNED8 1024
#define F2PY_INTENT_ALIGNED16 2048

/* 
   定义一系列宏，用于检查数组的对齐要求
*/
#define F2PY_ALIGN4(intent) (intent & F2PY_INTENT_ALIGNED4)
#define F2PY_ALIGN8(intent) (intent & F2PY_INTENT_ALIGNED8)
#define F2PY_ALIGN16(intent) (intent & F2PY_INTENT_ALIGNED16)

/* 
   获取指定意图下的数组对齐大小
*/
#define F2PY_GET_ALIGNMENT(intent) \
    (F2PY_ALIGN4(intent)           \
             ? 4                   \
             : (F2PY_ALIGN8(intent) ? 8 : (F2PY_ALIGN16(intent) ? 16 : 1)))

/* 
   检查数组是否满足指定的对齐要求
*/
#define F2PY_CHECK_ALIGNMENT(arr, intent) \
    ARRAY_ISALIGNED(arr, F2PY_GET_ALIGNMENT(intent))

/* 
   检查数组是否兼容字符类型
*/
#define F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr) ((PyArray_DESCR(arr)->type_num == NPY_STRING && PyArray_ITEMSIZE(arr) >= 1) \
                                                 || PyArray_DESCR(arr)->type_num == NPY_UINT8)

/* 
   检查数组是否是 Unicode 类型
*/
#define F2PY_IS_UNICODE_ARRAY(arr) (PyArray_DESCR(arr)->type_num == NPY_UNICODE)

/* 
   定义一个函数原型，用于从 Python 对象创建一个多维数组
*/
extern PyArrayObject *
ndarray_from_pyobj(const int type_num, const int elsize_, npy_intp *dims,
                   const int rank, const int intent, PyObject *obj,
                   const char *errmess);

/* 
   定义一个函数原型，用于从 Python 对象创建一个一维数组
*/
extern PyArrayObject *
array_from_pyobj(const int type_num, npy_intp *dims, const int rank,
                 const int intent, PyObject *obj);

/* 
   定义一个函数，用于复制一个多维数组
*/
extern int
copy_ND_array(const PyArrayObject *in, PyArrayObject *out);

#ifdef DEBUG_COPY_ND_ARRAY
/* 
   如果定义了 DEBUG_COPY_ND_ARRAY，定义一个函数原型，用于打印数组的属性
*/
extern void
dump_attrs(const PyArrayObject *arr);
#endif

/* 
   定义一个函数原型，用于描述一个 Python 对象
*/
extern int f2py_describe(PyObject *obj, char *buf);

/* 
   以下是一系列宏和函数，用于在签名文件表达式中使用的实用工具。
   可以参考 signature-file.rst 获取更多文档信息。
*/

/* 
   宏，用于获取数组元素的大小
*/
#define f2py_itemsize(var) (PyArray_ITEMSIZE(capi_ ## var ## _as_array))

/* 
   宏，用于获取数组的大小
*/
#define f2py_size(var, ...) f2py_size_impl((PyArrayObject *)(capi_ ## var ## _as_array), ## __VA_ARGS__, -1)

/* 
   宏，用于获取数组的秩（维度数量）
*/
#define f2py_rank(var) var ## _Rank

/* 
   宏，用于获取数组的指定维度的大小
*/
#define f2py_shape(var,dim) var ## _Dims[dim]

/* 
   宏，用于获取数组的第一维度大小
*/
#define f2py_len(var) f2py_shape(var,0)

/* 
   宏，用于获取数组的逆序排列的维度大小
*/
#define f2py_fshape(var,dim) f2py_shape(var,rank(var)-dim-1)

/* 
   宏，用于获取适用于字符串长度的数组大小
*/
#define f2py_flen(var) f2py_fshape(var,0)

/* 
   宏，用于获取对象的字符串长度
*/
#define f2py_slen(var) capi_ ## var ## _len

/* 
   定义一个函数原型，用于获取数组的大小
*/
extern npy_intp f2py_size_impl(PyArrayObject* var, ...);
```