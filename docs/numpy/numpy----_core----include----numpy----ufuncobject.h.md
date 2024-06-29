# `.\numpy\numpy\_core\include\numpy\ufuncobject.h`

```py
/*
 * 定义一个条件编译宏，用于防止重复包含 numpy 的 ufunc 相关头文件
 */
#ifndef NUMPY_CORE_INCLUDE_NUMPY_UFUNCOBJECT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_UFUNCOBJECT_H_

/*
 * 包含 numpy 库的数学相关头文件
 */
#include <numpy/npy_math.h>
/*
 * 包含 numpy 库的通用头文件
 */
#include <numpy/npy_common.h>

/*
 * 如果是 C++ 代码，则使用 extern "C" 语法将代码声明为 C 函数接口
 */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * 定义了一个指向泛型函数指针类型 PyUFuncGenericFunction，
 * 用于表示标准的逐元素或广义的 ufunc 内部循环。
 * 这个函数指针接受参数列表和数据维度以及步长信息。
 */
typedef void (*PyUFuncGenericFunction)(
            char **args,
            npy_intp const *dimensions,
            npy_intp const *strides,
            void *innerloopdata);

/*
 * 定义了一个指向带掩码的标准逐元素 ufunc 的最通用的一维内部循环函数指针类型
 * 这里的 "masked" 意味着它会跳过掩码数组 maskptr 中对应位置为真值的数据项的计算。
 * 接受数据指针数组、步长数组、掩码指针、掩码步长、数据项计数和内部循环数据。
 */
typedef void (PyUFunc_MaskedStridedInnerLoopFunc)(
                char **dataptrs, npy_intp *strides,
                char *maskptr, npy_intp mask_stride,
                npy_intp count,
                NpyAuxData *innerloopdata);

/* 声明 _tagPyUFuncObject 结构体类型，用于解析类型和选择内部循环的函数指针 */
struct _tagPyUFuncObject;

/*
 * 给定 ufunc 调用的操作数，应确定计算输入和输出数据类型并返回一个内部循环函数。
 * 此函数应验证是否遵循了类型转换规则，如果没有则应该失败。
 *
 * 对于向后兼容性，普通的类型解析函数不支持具有对象语义的辅助数据。
 * 返回掩码通用函数的类型解析调用返回一个标准的 NpyAuxData 对象，
 * 其中 NPY_AUXDATA_FREE 和 NPY_AUXDATA_CLONE 宏适用。
 *
 * ufunc:             ufunc 对象。
 * casting:           提供给 ufunc 的 'casting' 参数。
 * operands:          一个长度为 (ufunc->nin + ufunc->nout) 的数组，
 *                    输出参数可能为 NULL。
 * type_tup:          可能为 NULL，也可能为 ufunc 传递的 type_tup。
 * out_dtypes:        应填充一个数组，其中包含 (ufunc->nin + ufunc->nout) 个新的
 *                    dtypes 的新引用，每个输入和输出一个。
 *                    这些 dtypes 应该都是本机字节序格式。
 *
 * 成功返回 0，失败返回 -1（并设置异常），如果应返回 Py_NotImplemented 返回 -2。
 */
typedef int (PyUFunc_TypeResolutionFunc)(
                                struct _tagPyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);
    #if NPY_FEATURE_VERSION >= NPY_1_16_API_VERSION
        /*
         * for each core_num_dim_ix distinct dimension names,
         * the possible "frozen" size (-1 if not frozen).
         */
        npy_intp *core_dim_sizes;

        /*
         * for each distinct core dimension, a set of UFUNC_CORE_DIM* flags
         */
        npy_uint32 *core_dim_flags;

        /* Identity for reduction, when identity == PyUFunc_IdentityValue */
        PyObject *identity_value;
    #endif  /* NPY_FEATURE_VERSION >= NPY_1_16_API_VERSION */

        /* New in NPY_API_VERSION 0x0000000F and above */
    #if NPY_FEATURE_VERSION >= NPY_1_22_API_VERSION
        /* New private fields related to dispatching */
        void *_dispatch_cache;
        /* A PyListObject of `(tuple of DTypes, ArrayMethod/Promoter)` */
        PyObject *_loops;
    #endif



#if NPY_FEATURE_VERSION >= NPY_1_16_API_VERSION
    /*
     * 对于每个 core_num_dim_ix 不同的维度名称，
     * 可能的“冻结”大小（如果未冻结则为-1）。
     */
    npy_intp *core_dim_sizes;

    /*
     * 对于每个不同的核心维度，一组 UFUNC_CORE_DIM* 标志
     */
    npy_uint32 *core_dim_flags;

    /* 当 identity == PyUFunc_IdentityValue 时，用于约简的标识 */
    PyObject *identity_value;
#endif  /* NPY_FEATURE_VERSION >= NPY_1_16_API_VERSION */

/* 在 NPY_API_VERSION 0x0000000F 及以上版本中新增 */
#if NPY_FEATURE_VERSION >= NPY_1_22_API_VERSION
    /* 与调度相关的新私有字段 */
    void *_dispatch_cache;
    /* 一个 PyListObject 包含“(DTypes 元组, ArrayMethod/Promoter)” */
    PyObject *_loops;
#endif
/* 结构体定义，表示 Python 中的通用函数对象 */
} PyUFuncObject;

/* 包含 Python 数组对象的头文件 */
#include "arrayobject.h"

/* UFunc 的常量定义 */

/* UFunc 的核心维度大小由操作数确定 */
#define UFUNC_CORE_DIM_SIZE_INFERRED 0x0002
/* UFunc 的核心维度可以忽略 */
#define UFUNC_CORE_DIM_CAN_IGNORE 0x0004
/* 在执行期间推断出的标志 */
#define UFUNC_CORE_DIM_MISSING 0x00040000

/* UFunc 对象的属性标志 */
#define UFUNC_OBJ_ISOBJECT      1
#define UFUNC_OBJ_NEEDS_API     2

/* 多线程相关宏定义 */

#if NPY_ALLOW_THREADS
/* 在线程开始时保存线程状态 */
#define NPY_LOOP_BEGIN_THREADS do {if (!(loop->obj & UFUNC_OBJ_NEEDS_API)) _save = PyEval_SaveThread();} while (0);
/* 在线程结束时恢复线程状态 */
#define NPY_LOOP_END_THREADS   do {if (!(loop->obj & UFUNC_OBJ_NEEDS_API)) PyEval_RestoreThread(_save);} while (0);
#else
/* 不允许多线程时的空宏定义 */
#define NPY_LOOP_BEGIN_THREADS
#define NPY_LOOP_END_THREADS
#endif

/* UFunc 单位的定义 */

/* UFunc 单位为 0，操作顺序可以重新排序 */
#define PyUFunc_Zero 0
/* UFunc 单位为 1，操作顺序可以重新排序 */
#define PyUFunc_One 1
/* UFunc 单位为 -1，操作顺序可以重新排序，用于位与操作的缩减 */
#define PyUFunc_MinusOne 2
/* UFunc 没有单位，操作顺序不能重新排序，不允许多轴同时缩减 */
#define PyUFunc_None -1
/* UFunc 没有单位，操作顺序可以重新排序，允许多轴同时缩减 */
#define PyUFunc_ReorderableNone -2
/* UFunc 单位是标识值，操作顺序可以重新排序，允许多轴同时缩减 */
#define PyUFunc_IdentityValue -3

/* UFunc 的运算类型 */

#define UFUNC_REDUCE 0
#define UFUNC_ACCUMULATE 1
#define UFUNC_REDUCEAT 2
#define UFUNC_OUTER 3

/* PyUFunc_PyFuncData 结构体，用于保存 Python 函数数据 */

typedef struct {
        int nin;            /* 输入参数个数 */
        int nout;           /* 输出参数个数 */
        PyObject *callable; /* Python 可调用对象 */
} PyUFunc_PyFuncData;

/* 用户定义的一维循环函数信息的链表结构 */

typedef struct _loop1d_info {
        PyUFuncGenericFunction func;    /* 通用函数指针 */
        void *data;                     /* 数据指针 */
        int *arg_types;                 /* 参数类型数组 */
        struct _loop1d_info *next;      /* 下一个链表节点 */
        int nargs;                      /* 参数个数 */
        PyArray_Descr **arg_dtypes;     /* 参数数据类型数组 */
} PyUFunc_Loop1d;

/* UFUNC_PYVALS_NAME 宏定义 */

#define UFUNC_PYVALS_NAME "UFUNC_PYVALS"

/* 下面的宏定义已经废弃，请使用 npy_set_floatstatus_* 在 npymath 库中 */
#define UFUNC_FPE_DIVIDEBYZERO  NPY_FPE_DIVIDEBYZERO
#define UFUNC_FPE_OVERFLOW      NPY_FPE_OVERFLOW
#define UFUNC_FPE_UNDERFLOW     NPY_FPE_UNDERFLOW
#define UFUNC_FPE_INVALID       NPY_FPE_INVALID

/* 生成浮点异常错误的宏 */
#define generate_divbyzero_error() npy_set_floatstatus_divbyzero()
#define generate_overflow_error() npy_set_floatstatus_overflow()

/* 如果未定义 UFUNC_NOFPE，则定义其默认行为 */

#ifndef UFUNC_NOFPE
/* 清除 Borland C++ 的默认浮点异常处理 */
#if defined(__BORLANDC__)
#define UFUNC_NOFPE _control87(MCW_EM, MCW_EM);
#else
#define UFUNC_NOFPE
#endif
#endif

#include "__ufunc_api.h"

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_UFUNCOBJECT_H_ */




#else
// 如果没有进入上一个 #ifdef 条件，则定义 UFUNC_NOFPE 宏
#define UFUNC_NOFPE
#endif
#endif

// 包含私有的 ufunc API 头文件 "__ufunc_api.h"
#include "__ufunc_api.h"

#ifdef __cplusplus
// 如果是 C++ 编译环境，则结束 extern "C" 块
}
#endif

// 结束 NUMPY_UFUNCOBJECT_H_ 宏的定义
#endif  /* NUMPY_CORE_INCLUDE_NUMPY_UFUNCOBJECT_H_ */
```