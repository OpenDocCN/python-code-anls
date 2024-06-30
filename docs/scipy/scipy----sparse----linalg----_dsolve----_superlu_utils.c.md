# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\_superlu_utils.c`

```
/* Should be imported before Python.h */

#include <Python.h>  // 包含 Python C API 的头文件

#define NO_IMPORT_ARRAY  // 禁止导入数组
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_superlu_ARRAY_API  // 定义数组 API 的唯一符号

#include "_superluobject.h"  // 导入 SuperLU 对象的头文件
#include <setjmp.h>  // 导入 setjmp.h 头文件，支持非本地跳转

/* Abort to be used inside the superlu module so that memory allocation
   errors don't exit Python and memory allocated internal to SuperLU is freed.
   Calling program should deallocate (using SUPERLU_FREE) all memory that could have
   been allocated.  (It's ok to FREE unallocated memory)---will be ignored.
*/

#ifndef WITH_THREAD
static SuperLUGlobalObject superlu_py_global = {0};  // 非多线程环境下的 SuperLU 全局对象
#endif

static SuperLUGlobalObject *get_tls_global(void)
{
#ifndef WITH_THREAD
    if (superlu_py_global.memory_dict == NULL) {
        superlu_py_global.memory_dict = PyDict_New();  // 如果内存字典为空，创建一个新的 Python 字典对象
    }
    return &superlu_py_global;  // 返回全局对象的指针
#else
    PyObject *thread_dict;
    SuperLUGlobalObject *obj;
    const char *key = "scipy.sparse.linalg._dsolve._superlu.__global_object";  // 线程私有字典中的键名

    thread_dict = PyThreadState_GetDict();  // 获取当前线程状态的字典对象
    if (thread_dict == NULL) {
        /* Should never happen */
        PyErr_SetString(PyExc_SystemError, "no thread state obtained");  // 报告系统错误，线程状态未获取
        return NULL;
    }

    obj = (SuperLUGlobalObject*)PyDict_GetItemString(thread_dict, key);  // 在线程字典中查找全局对象
    if (obj && Py_TYPE(obj) == &SuperLUGlobalType) {
        return obj;  // 如果找到且对象类型匹配，则返回该对象
    }

    obj = (SuperLUGlobalObject*)PyObject_New(SuperLUGlobalObject, &SuperLUGlobalType);  // 新建一个 SuperLU 全局对象
    if (obj == NULL) {
        return (SuperLUGlobalObject*)PyErr_NoMemory();  // 如果内存分配失败，返回内存错误异常
    }
    obj->memory_dict = PyDict_New();  // 创建一个新的 Python 字典对象用于存储内存信息
    obj->jmpbuf_valid = 0;  // 设置跳转缓冲区状态为无效

    PyDict_SetItemString(thread_dict, key, (PyObject *)obj);  // 将全局对象存储到线程字典中

    return obj;  // 返回线程私有的 SuperLU 全局对象
#endif
}

jmp_buf *superlu_python_jmpbuf(void)
{
    SuperLUGlobalObject *g;

    g = get_tls_global();  // 获取线程私有的 SuperLU 全局对象
    if (g == NULL) {
        abort();  // 如果获取失败，中止程序
    }
    g->jmpbuf_valid = 1;  // 设置跳转缓冲区状态为有效
    return &g->jmpbuf;  // 返回跳转缓冲区的地址
}

void superlu_python_module_abort(char *msg)
{
    SuperLUGlobalObject *g;
    NPY_ALLOW_C_API_DEF;

    NPY_ALLOW_C_API;
    g = get_tls_global();  // 获取线程私有的 SuperLU 全局对象
    if (g == NULL) {
        /* We have to longjmp (or SEGV results), but the
           destination is not known --- no choice but abort.
           However, this should never happen.
        */
        abort();  // 如果获取失败，中止程序
    }
    PyErr_SetString(PyExc_RuntimeError, msg);  // 设置运行时错误并传递错误消息

    if (!g->jmpbuf_valid) {
        abort();  // 如果跳转缓冲区无效，中止程序
    }

    g->jmpbuf_valid = 0;  // 设置跳转缓冲区状态为无效
    NPY_DISABLE_C_API;  // 禁用 C API
    longjmp(g->jmpbuf, -1);  // 跳转到设定的跳转缓冲区，返回错误码 -1
}

void *superlu_python_module_malloc(size_t size)
{
    SuperLUGlobalObject *g;
    PyObject *key = NULL;
    void *mem_ptr;
    NPY_ALLOW_C_API_DEF;

    NPY_ALLOW_C_API;
    g = get_tls_global();  // 获取线程私有的 SuperLU 全局对象
    if (g == NULL) {
        return NULL;  // 如果获取失败，返回空指针
    }
    mem_ptr = malloc(size);  // 分配指定大小的内存
    if (mem_ptr == NULL) {
        NPY_DISABLE_C_API;  // 如果内存分配失败，禁用 C API
        return NULL;
    }
    key = PyLong_FromVoidPtr(mem_ptr);  // 将内存指针转换为 Python 长整型对象
    if (key == NULL)
        goto fail;
    if (PyDict_SetItem(g->memory_dict, key, Py_None))
        goto fail;
    Py_DECREF(key);  // 释放 Python 对象的引用计数
    NPY_DISABLE_C_API;  // 禁用 C API

    return mem_ptr;  // 返回分配的内存指针

  fail:
    Py_XDECREF(key);  // 失败时释放 Python 对象的引用计数
    NPY_DISABLE_C_API;  // 禁用 C API
    free(mem_ptr);  // 失败时释放分配的内存
    return NULL;  // 返回空指针
}
    superlu_python_module_abort
    ("superlu_malloc: Cannot set dictionary key value in malloc.");
    # 调用名为 superlu_python_module_abort 的函数，并传入字符串参数
    return NULL
    # 返回 NULL
`
}

void superlu_python_module_free(void *ptr)
{
    SuperLUGlobalObject *g;
    PyObject *key;
    PyObject *ptype, *pvalue, *ptraceback;
    NPY_ALLOW_C_API_DEF;

    if (ptr == NULL)
        return;

    NPY_ALLOW_C_API;
    // 获取全局对象
    g = get_tls_global();
    // 如果获取不到全局对象，则终止程序
    if (g == NULL) {
        abort();
    }
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    // 将指针转换为长整型对象作为字典的键
    key = PyLong_FromVoidPtr(ptr);
    /* This will only free the pointer if it could find it in the dictionary
     * of already allocated pointers --- thus after abort, the module can free all
     * the memory that "might" have been allocated to avoid memory leaks on abort
     * calls.
     */
    // 如果在内存字典中找到了指定键，则从字典中删除并释放指针
    if (!PyDict_DelItem(g->memory_dict, key)) {
        free(ptr);
    }
    // 减少键对象的引用计数
    Py_DECREF(key);
    // 恢复之前保存的异常状态
    PyErr_Restore(ptype, pvalue, ptraceback);
    NPY_DISABLE_C_API;
    return;
}

static void SuperLUGlobal_dealloc(SuperLUGlobalObject *self)
{
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    // 遍历内存字典中的所有键值对
    while (PyDict_Next(self->memory_dict, &pos, &key, &value)) {
        void *ptr;
        // 将值对象转换为指针
        ptr = PyLong_AsVoidPtr(value);
        // 释放指针指向的内存
        free(ptr);
    }

    // 释放内存字典对象
    Py_XDECREF(self->memory_dict);
    // 删除对象自身
    PyObject_Del(self);
}

// 定义类型对象 SuperLUGlobalType
PyTypeObject SuperLUGlobalType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_SuperLUGlobal",             /* tp_name */
    sizeof(SuperLUGlobalObject),  /* tp_basicsize */
    0,                            /* tp_itemsize */
    (destructor)SuperLUGlobal_dealloc, /* tp_dealloc */
    0,                /* tp_print */
    0,                            /* tp_getattr */
    0,                /* tp_setattr */
    0,                /* tp_reserved */
    0,                /* tp_repr */
    0,                /* tp_as_number */
    0,                /* tp_as_sequence */
    0,                /* tp_as_mapping */
    0,                /* tp_hash */
    0,                /* tp_call */
    0,                /* tp_str */
    0,                /* tp_getattro */
    0,                /* tp_setattro */
    0,                /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    NULL,                       /* tp_doc */
    0,                /* tp_traverse */
    0,                /* tp_clear */
    0,                /* tp_richcompare */
    0,                /* tp_weaklistoffset */
    0,                /* tp_iter */
    0,                /* tp_iternext */
    0,                          /* tp_methods */
    0,                /* tp_members */
    0,                       /* tp_getset */
    0,                /* tp_base */
    0,                /* tp_dict */
    0,                /* tp_descr_get */
    0,                /* tp_descr_set */
    0,                /* tp_dictoffset */
    0,                /* tp_init */
    0,                /* tp_alloc */
    0,                /* tp_new */
    0,                /* tp_free */
    0,                /* tp_is_gc */
    0,                /* tp_bases */
    0,                /* tp_mro */
    0,                /* tp_cache */
    0,                /* tp_subclasses */
    0,                /* tp_weaklist */
    0,                /* tp_del */


注释：
    0,                /* tp_version_tag */
/*
 * Stub for error handling; does nothing, as we don't want to spew debug output.
 */
int input_error(char *srname, int *info)
{
    // 此函数用作错误处理的占位符，不执行任何操作，避免输出调试信息。
    return 0;
}

/*
 * Stubs for Harwell Subroutine Library functions that SuperLU tries to call.
 */
void mc64id_(int *a)
{
    // 当前功能不可用时，调用此函数中止程序并显示错误信息。
    superlu_python_module_abort("chosen functionality not available");
}

void mc64ad_(int *a, int *b, int *c, int d[], int e[], double f[],
         int *g, int h[], int *i, int j[], int *k, double l[],
         int m[], int n[])
{
    // 当前功能不可用时，调用此函数中止程序并显示错误信息。
    superlu_python_module_abort("chosen functionality not available");
}
```