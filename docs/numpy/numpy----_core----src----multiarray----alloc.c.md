# `.\numpy\numpy\_core\src\multiarray\alloc.c`

```
/*
 * 定义 NPY_NO_DEPRECATED_API 以及 _MULTIARRAYMODULE，这些是预处理器宏
 * NPY_SSIZE_T_CLEAN 用于确保 Py_ssize_t 被定义为 size_t，这是一种约定
 * 包含 Python.h，structmember.h，pymem.h 以及一系列 NumPy 头文件
 * assert.h 被包含用于运行时断言
 * 对于 Linux 系统，包含 sys/mman.h 以及在内核版本低于 2.6.38 时定义 MADV_HUGEPAGE
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <pymem.h>
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"
#include "npy_config.h"
#include "alloc.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"
#include <assert.h>
#ifdef NPY_OS_LINUX
#include <sys/mman.h>
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif
#endif

/*
 * 定义 NBUCKETS、NBUCKETS_DIM 和 NCACHE 分别作为数据、维度/步长的缓存桶数目以及每个缓存桶中的缓存条目数
 * 定义 cache_bucket 结构，用于缓存指针
 * datacache 和 dimcache 是全局静态变量，用于存储数据和维度的缓存
 */
#define NBUCKETS 1024 /* number of buckets for data*/
#define NBUCKETS_DIM 16 /* number of buckets for dimensions/strides */
#define NCACHE 7 /* number of cache entries per bucket */
typedef struct {
    npy_uintp available; /* number of cached pointers */
    void * ptrs[NCACHE];
} cache_bucket;
static cache_bucket datacache[NBUCKETS];
static cache_bucket dimcache[NBUCKETS_DIM];

/*
 * 下面的函数定义了两个功能：
 * - _get_madvise_hugepage 检查是否启用了 MADV_HUGEPAGE，返回一个 Python 布尔值
 * - _set_madvise_hugepage 启用或禁用 MADV_HUGEPAGE，并返回先前的设置
 * 这两个函数都是 NPY_NO_EXPORT，意味着它们在库外不可见
 */
NPY_NO_EXPORT PyObject *
_get_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
#ifdef NPY_OS_LINUX
    if (npy_thread_unsafe_state.madvise_hugepage) {
        Py_RETURN_TRUE;
    }
#endif
    Py_RETURN_FALSE;
}

NPY_NO_EXPORT PyObject *
_set_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *enabled_obj)
{
    int was_enabled = npy_thread_unsafe_state.madvise_hugepage;
    int enabled = PyObject_IsTrue(enabled_obj);
    if (enabled < 0) {
        return NULL;
    }
    npy_thread_unsafe_state.madvise_hugepage = enabled;
    if (was_enabled) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/*
 * _npy_alloc_cache 是一个内联函数，用于管理小内存块缓存，避免使用更昂贵的 libc 分配
 * 根据 nelem、esz 和 msz 的值，决定从 cache 中获取缓存指针还是进行新分配
 * 要求在调用此函数时必须持有全局解释器锁 (GIL)
 */
static inline void *
_npy_alloc_cache(npy_uintp nelem, npy_uintp esz, npy_uint msz,
                 cache_bucket * cache, void * (*alloc)(size_t))
{
    void * p;
    assert((esz == 1 && cache == datacache) ||
           (esz == sizeof(npy_intp) && cache == dimcache));
    assert(PyGILState_Check());
#ifndef Py_GIL_DISABLED
    if (nelem < msz) {
        if (cache[nelem].available > 0) {
            return cache[nelem].ptrs[--(cache[nelem].available)];
        }
    }
#endif
    # 分配内存空间，分配的大小为 nelem 乘以 esz
    p = alloc(nelem * esz);
    # 如果成功分配了内存空间（即 p 非空）
    if (p) {
#ifdef _PyPyGC_AddMemoryPressure
        _PyPyPyGC_AddMemoryPressure(nelem * esz);
#endif
#ifdef NPY_OS_LINUX
        /* 允许内核为大数组分配巨大页面 */
        if (NPY_UNLIKELY(nelem * esz >= ((1u<<22u))) &&
            npy_thread_unsafe_state.madvise_hugepage) {
            npy_uintp offset = 4096u - (npy_uintp)p % (4096u);
            npy_uintp length = nelem * esz - offset;
            /**
             * 故意不检查可能由旧内核版本返回的错误；乐观地尝试启用巨大页面。
             */
            madvise((void*)((npy_uintp)p + offset), length, MADV_HUGEPAGE);
        }
#endif
    }
    return p;
}

/*
 * 返回指针 p 到缓存，nelem 是缓存桶大小的元素数（1 或 sizeof(npy_intp)）
 */
static inline void
_npy_free_cache(void * p, npy_uintp nelem, npy_uint msz,
                cache_bucket * cache, void (*dealloc)(void *))
{
    assert(PyGILState_Check());
#ifndef Py_GIL_DISABLED
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
#endif
    dealloc(p);
}


/*
 * 数组数据缓存，sz 是要分配的字节数
 */
NPY_NO_EXPORT void *
npy_alloc_cache(npy_uintp sz)
{
    return _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
}

/* 零初始化数据，sz 是要分配的字节数 */
NPY_NO_EXPORT void *
npy_alloc_cache_zero(size_t nmemb, size_t size)
{
    void * p;
    size_t sz = nmemb * size;
    NPY_BEGIN_THREADS_DEF;
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
        if (p) {
            memset(p, 0, sz);
        }
        return p;
    }
    NPY_BEGIN_THREADS;
    p = PyDataMem_NEW_ZEROED(nmemb, size);
    NPY_END_THREADS;
    return p;
}

NPY_NO_EXPORT void
npy_free_cache(void * p, npy_uintp sz)
{
    _npy_free_cache(p, sz, NBUCKETS, datacache, &PyDataMem_FREE);
}

/*
 * 维度/步幅缓存，使用不同的分配器，并始终是 npy_intp 的倍数
 */
NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz)
{
    /*
     * 确保任何临时分配可以用于数组元数据，该元数据使用一个内存块存储维度和步幅
     */
    if (sz < 2) {
        sz = 2;
    }
    return _npy_alloc_cache(sz, sizeof(npy_intp), NBUCKETS_DIM, dimcache,
                            &PyArray_malloc);
}

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sz)
{
    /* 见 npy_alloc_cache_dim */
    if (sz < 2) {
        sz = 2;
    }
    _npy_free_cache(p, sz, NBUCKETS_DIM, dimcache,
                    &PyArray_free);
}

/* 类似于 arrayobject.c 中的 array_dealloc */
static inline void
WARN_NO_RETURN(PyObject* warning, const char * msg) {
    # 如果发出警告，并且返回值小于0表示出错
    if (PyErr_WarnEx(warning, msg, 1) < 0) {
        # 声明一个指针变量s，指向PyObject类型
        PyObject * s;
        
        # 从字符串"PyDataMem_UserFREE"创建一个PyUnicode对象，并将其赋值给s
        s = PyUnicode_FromString("PyDataMem_UserFREE");
        # 如果s非空
        if (s) {
            # 输出一个不可恢复的异常，异常消息为s
            PyErr_WriteUnraisable(s);
            # 减少s的引用计数
            Py_DECREF(s);
        }
        else {
            # 输出一个不可恢复的异常，异常消息为Py_None
            PyErr_WriteUnraisable(Py_None);
        }
    }
/*NUMPY_API
 * Allocates memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW(size_t size)
{
    void *result;

    assert(size != 0);  // 确保分配的大小不为零
    result = malloc(size);  // 调用标准库函数分配内存
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);  // 调用跟踪内存分配的函数
    return result;  // 返回分配的内存地址
}

/*NUMPY_API
 * Allocates zeroed memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW_ZEROED(size_t nmemb, size_t size)
{
    void *result;

    result = calloc(nmemb, size);  // 调用标准库函数分配并清零内存
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, nmemb * size);  // 调用跟踪内存分配的函数
    return result;  // 返回分配的内存地址
}

/*NUMPY_API
 * Free memory for array data.
 */
NPY_NO_EXPORT void
PyDataMem_FREE(void *ptr)
{
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);  // 调用取消跟踪内存分配的函数
    free(ptr);  // 释放内存
}

/*NUMPY_API
 * Reallocate/resize memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_RENEW(void *ptr, size_t size)
{
    void *result;

    assert(size != 0);  // 确保重新分配的大小不为零
    result = realloc(ptr, size);  // 调用标准库函数重新分配内存
    if (result != ptr) {
        PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);  // 如果地址有变化，则取消旧地址的内存跟踪
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);  // 调用跟踪内存分配的函数
    return result;  // 返回重新分配的内存地址
}

// The default data mem allocator malloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserNEW
// since itself does not handle eventhook and tracemalloc logic.
static inline void *
default_malloc(void *NPY_UNUSED(ctx), size_t size)
{
    return _npy_alloc_cache(size, 1, NBUCKETS, datacache, &malloc);  // 调用特定分配缓存的函数
}

// The default data mem allocator calloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserNEW_ZEROED
// since itself does not handle eventhook and tracemalloc logic.
static inline void *
default_calloc(void *NPY_UNUSED(ctx), size_t nelem, size_t elsize)
{
    void * p;
    size_t sz = nelem * elsize;
    NPY_BEGIN_THREADS_DEF;  // 开始线程安全区域定义
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &malloc);  // 调用特定分配缓存的函数
        if (p) {
            memset(p, 0, sz);  // 如果分配成功，清零分配的内存
        }
        return p;  // 返回分配的内存地址
    }
    NPY_BEGIN_THREADS;  // 开始线程安全区域
    p = calloc(nelem, elsize);  // 调用标准库函数分配并清零内存
    NPY_END_THREADS;  // 结束线程安全区域
    return p;  // 返回分配的内存地址
}

// The default data mem allocator realloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserRENEW
// since itself does not handle eventhook and tracemalloc logic.
static inline void *
default_realloc(void *NPY_UNUSED(ctx), void *ptr, size_t new_size)
{
    return realloc(ptr, new_size);  // 调用标准库函数重新分配内存
}

// The default data mem allocator free routine does not make use of a ctx.
// It should be called only through PyDataMem_UserFREE
// since itself does not handle eventhook and tracemalloc logic.
static inline void
default_free(void *NPY_UNUSED(ctx), void *ptr, size_t size)
{
    _npy_free_cache(ptr, size, NBUCKETS, datacache, &free);  // 调用特定释放缓存的函数
}

/* Memory handler global default */
PyDataMem_Handler default_handler = {
    "default_allocator",  // 默认内存分配器的名称
    1,  // 是否线程安全的标志
    {
        NULL,            /* 上下文为空 */
        default_malloc,  /* 分配内存的默认函数指针 */
        default_calloc,  /* 分配并清零内存的默认函数指针 */
        default_realloc, /* 重新分配内存的默认函数指针 */
        default_free     /* 释放内存的默认函数指针 */
    }
};
/* singleton capsule of the default handler */
PyObject *PyDataMem_DefaultHandler;
PyObject *current_handler;

int uo_index=0;   /* user_override index */

/* Wrappers for the default or any user-assigned PyDataMem_Handler */

/* Allocate memory using the user-defined memory handler */
NPY_NO_EXPORT void *
PyDataMem_UserNEW(size_t size, PyObject *mem_handler)
{
    void *result;
    // Cast the PyObject mem_handler to PyDataMem_Handler*
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        return NULL;  // Return NULL if mem_handler is invalid
    }
    assert(size != 0);  // Ensure size is non-zero
    // Call malloc function of the allocator stored in handler
    result = handler->allocator.malloc(handler->allocator.ctx, size);
    // Track memory allocation using PyTraceMalloc_Track
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    return result;  // Return allocated memory pointer
}

/* Allocate zeroed memory using the user-defined memory handler */
NPY_NO_EXPORT void *
PyDataMem_UserNEW_ZEROED(size_t nmemb, size_t size, PyObject *mem_handler)
{
    void *result;
    // Cast the PyObject mem_handler to PyDataMem_Handler*
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        return NULL;  // Return NULL if mem_handler is invalid
    }
    // Call calloc function of the allocator stored in handler
    result = handler->allocator.calloc(handler->allocator.ctx, nmemb, size);
    // Track memory allocation using PyTraceMalloc_Track
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, nmemb * size);
    return result;  // Return allocated memory pointer
}

/* Free memory using the user-defined memory handler */
NPY_NO_EXPORT void
PyDataMem_UserFREE(void *ptr, size_t size, PyObject *mem_handler)
{
    // Cast the PyObject mem_handler to PyDataMem_Handler*
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        // Issue a warning and return if mem_handler is invalid
        WARN_NO_RETURN(PyExc_RuntimeWarning,
                     "Could not get pointer to 'mem_handler' from PyCapsule");
        return;
    }
    // Untrack memory allocation using PyTraceMalloc_Untrack
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    // Call free function of the allocator stored in handler
    handler->allocator.free(handler->allocator.ctx, ptr, size);
}

/* Reallocate memory using the user-defined memory handler */
NPY_NO_EXPORT void *
PyDataMem_UserRENEW(void *ptr, size_t size, PyObject *mem_handler)
{
    void *result;
    // Cast the PyObject mem_handler to PyDataMem_Handler*
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        return NULL;  // Return NULL if mem_handler is invalid
    }

    assert(size != 0);  // Ensure size is non-zero
    // Call realloc function of the allocator stored in handler
    result = handler->allocator.realloc(handler->allocator.ctx, ptr, size);
    // If reallocation changes the pointer, untrack old and track new
    if (result != ptr) {
        PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    return result;  // Return reallocated memory pointer
}

/*NUMPY_API
 * Set a new allocation policy. If the input value is NULL, will reset
 * the policy to the default. Return the previous policy, or
 * return NULL if an error has occurred. We wrap the user-provided
 * functions so they will still call the python and numpy
 * memory management callback hooks.
 */
NPY_NO_EXPORT PyObject *
PyDataMem_SetHandler(PyObject *handler)
{
    PyObject *old_handler;
    PyObject *token;
    // Get the current handler from context, return NULL on failure
    if (PyContextVar_Get(current_handler, NULL, &old_handler)) {
        return NULL;
    }
    // If handler is NULL, set it to the default handler
    if (handler == NULL) {
        handler = PyDataMem_DefaultHandler;
    }
    # 检查传入的 PyCapsule 对象是否有效，并且其名称必须是 'mem_handler'
    if (!PyCapsule_IsValid(handler, MEM_HANDLER_CAPSULE_NAME)) {
        # 如果不是有效的 Capsule 或者名称不匹配，设置 ValueError 异常，并返回空指针
        PyErr_SetString(PyExc_ValueError, "Capsule must be named 'mem_handler'");
        return NULL;
    }
    # 将当前的内存处理器设置为传入的 handler
    token = PyContextVar_Set(current_handler, handler);
    if (token == NULL) {
        # 如果设置失败，需要恢复旧的 handler，并返回空指针
        Py_DECREF(old_handler);
        return NULL;
    }
    # 释放 token 对象的引用，因为已经设置成功
    Py_DECREF(token);
    # 返回之前的内存处理器对象，现在已经使用新的 handler
    return old_handler;
/*NUMPY_API
 * 返回下一个 PyArrayObject 分配数据的策略。如果失败，则返回 NULL。
 */
NPY_NO_EXPORT PyObject *
PyDataMem_GetHandler()
{
    PyObject *handler;
    // 获取当前的内存处理器对象
    if (PyContextVar_Get(current_handler, NULL, &handler)) {
        return NULL;
    }
    return handler;  // 返回内存处理器对象
}

NPY_NO_EXPORT PyObject *
get_handler_name(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *arr=NULL;
    // 解析参数，获取可能传入的 ndarray 对象
    if (!PyArg_ParseTuple(args, "|O:get_handler_name", &arr)) {
        return NULL;
    }
    // 如果传入的不是 ndarray 对象，则抛出异常
    if (arr != NULL && !PyArray_Check(arr)) {
         PyErr_SetString(PyExc_ValueError, "if supplied, argument must be an ndarray");
         return NULL;
    }
    PyObject *mem_handler;
    PyDataMem_Handler *handler;
    PyObject *name;
    // 如果传入了 ndarray 对象
    if (arr != NULL) {
        // 获取 ndarray 的内存处理器对象
        mem_handler = PyArray_HANDLER((PyArrayObject *) arr);
        // 如果获取失败，返回 None
        if (mem_handler == NULL) {
            Py_RETURN_NONE;
        }
        Py_INCREF(mem_handler);
    }
    else {
        // 否则，获取默认的内存处理器对象
        mem_handler = PyDataMem_GetHandler();
        // 如果获取失败，返回 NULL
        if (mem_handler == NULL) {
            return NULL;
        }
    }
    // 从内存处理器对象中获取处理器结构体指针
    handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    // 如果获取失败，释放内存处理器对象并返回 NULL
    if (handler == NULL) {
        Py_DECREF(mem_handler);
        return NULL;
    }
    // 从处理器结构体中获取处理器名称，转换成 Python 字符串对象
    name = PyUnicode_FromString(handler->name);
    Py_DECREF(mem_handler);
    return name;  // 返回处理器名称对象
}

NPY_NO_EXPORT PyObject *
get_handler_version(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *arr=NULL;
    // 解析参数，获取可能传入的 ndarray 对象
    if (!PyArg_ParseTuple(args, "|O:get_handler_version", &arr)) {
        return NULL;
    }
    // 如果传入的不是 ndarray 对象，则抛出异常
    if (arr != NULL && !PyArray_Check(arr)) {
         PyErr_SetString(PyExc_ValueError, "if supplied, argument must be an ndarray");
         return NULL;
    }
    PyObject *mem_handler;
    PyDataMem_Handler *handler;
    PyObject *version;
    // 如果传入了 ndarray 对象
    if (arr != NULL) {
        // 获取 ndarray 的内存处理器对象
        mem_handler = PyArray_HANDLER((PyArrayObject *) arr);
        // 如果获取失败，返回 None
        if (mem_handler == NULL) {
            Py_RETURN_NONE;
        }
        Py_INCREF(mem_handler);
    }
    else {
        // 否则，获取默认的内存处理器对象
        mem_handler = PyDataMem_GetHandler();
        // 如果获取失败，返回 NULL
        if (mem_handler == NULL) {
            return NULL;
        }
    }
    // 从内存处理器对象中获取处理器结构体指针
    handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    // 如果获取失败，释放内存处理器对象并返回 NULL
    if (handler == NULL) {
        Py_DECREF(mem_handler);
        return NULL;
    }
    // 从处理器结构体中获取处理器版本号，转换成 Python 整数对象
    version = PyLong_FromLong(handler->version);
    Py_DECREF(mem_handler);
    return version;  // 返回处理器版本号对象
}
```