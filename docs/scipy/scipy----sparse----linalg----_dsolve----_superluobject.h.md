# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\_superluobject.h`

```
/*
 * _superlu object
 *
 * Python object representing SuperLU factorization + some utility functions.
 */

#ifndef __SUPERLU_OBJECT
#define __SUPERLU_OBJECT

#include <Python.h>
#include <setjmp.h>

/* Undef a macro from Python which conflicts with superlu */
#ifdef c_abs
#undef c_abs
#endif

#include "SuperLU/SRC/slu_zdefs.h"
#include "numpy/ndarrayobject.h"
#include "SuperLU/SRC/slu_util.h"
#include "SuperLU/SRC/slu_dcomplex.h"
#include "SuperLU/SRC/slu_scomplex.h"


#define _CHECK_INTEGER(x) (PyArray_ISINTEGER((PyArrayObject*)x) && PyArray_ITEMSIZE((PyArrayObject*)x) == sizeof(int))

/*
 * SuperLUObject definition
 */
typedef struct {
    PyObject_HEAD
    npy_intp m, n;                               // 表示矩阵维度的整数类型
    SuperMatrix L;                               // 存储下三角矩阵的结构体
    SuperMatrix U;                               // 存储上三角矩阵的结构体
    int *perm_r;                                 // 行置换数组
    int *perm_c;                                 // 列置换数组
    PyObject *cached_U;                          // 缓存的上三角矩阵对象
    PyObject *cached_L;                          // 缓存的下三角矩阵对象
    PyObject *py_csc_construct_func;              // CSC格式矩阵构造函数对象
    int type;                                    // 类型标志
} SuperLUObject;

typedef struct {
    PyObject_HEAD
    int jmpbuf_valid;                            // jmp_buf是否有效的标志
    jmp_buf jmpbuf;                              // 用于异常处理的跳转缓冲区
    PyObject *memory_dict;                       // 内存字典对象
} SuperLUGlobalObject;

extern PyTypeObject SuperLUType;                 // SuperLUObject类型对象声明
extern PyTypeObject SuperLUGlobalType;           // SuperLUGlobalObject类型对象声明

int DenseSuper_from_Numeric(SuperMatrix *, PyObject *);  // 从数值对象创建稠密超级矩阵
int NRFormat_from_spMatrix(SuperMatrix *, int, int, int, PyArrayObject *,
               PyArrayObject *, PyArrayObject *, int);  // 从稀疏矩阵创建非规则格式矩阵
int NCFormat_from_spMatrix(SuperMatrix *, int, int, int, PyArrayObject *,
               PyArrayObject *, PyArrayObject *, int);  // 从稀疏矩阵创建列压缩格式矩阵
int SparseFormat_from_spMatrix(SuperMatrix * A, int m, int n, int nnz, int csr,
               PyArrayObject * nzvals,
               PyArrayObject * indices,
               PyArrayObject * pointers,
               int typenum, Stype_t stype, Mtype_t mtype,
               int* identity_col_to_sup, int* identity_sup_to_col);  // 从稀疏矩阵创建稀疏格式矩阵
int LU_to_csc_matrix(SuperMatrix *L, SuperMatrix *U,
                     PyObject **L_csc, PyObject **U_csc,
                     PyObject *py_csc_construct_func);  // 将LU分解转换为CSC格式矩阵
colperm_t superlu_module_getpermc(int);          // 获取列置换对象的函数声明
PyObject *newSuperLUObject(SuperMatrix *, PyObject *, int, int, PyObject *);  // 创建新的SuperLU对象
int set_superlu_options_from_dict(superlu_options_t * options,
                  int ilu, PyObject * option_dict,
                  int *panel_size, int *relax);  // 从字典设置SuperLU选项
void XDestroy_SuperMatrix_Store(SuperMatrix *);  // 销毁稠密格式超级矩阵对象
void XDestroy_SuperNode_Matrix(SuperMatrix *);   // 销毁超级节点格式矩阵对象
void XDestroy_CompCol_Matrix(SuperMatrix *);     // 销毁压缩列格式矩阵对象
void XDestroy_CompCol_Permuted(SuperMatrix *);   // 销毁经置换的压缩列格式矩阵对象
void XStatFree(SuperLUStat_t *);                 // 释放SuperLU统计信息对象
jmp_buf *superlu_python_jmpbuf(void);            // 获取SuperLU异常处理的跳转缓冲区

/* Custom thread begin/end statements: Numpy versions < 1.9 are not safe
 * vs. calling END_THREADS multiple times. Moreover, the _save variable needs to
 * be volatile, due to the use of setjmp.
 */
#define SLU_BEGIN_THREADS_DEF volatile PyThreadState *_save = NULL  // 定义线程开始时保存线程状态的变量
#define SLU_BEGIN_THREADS do { if (_save == NULL) _save = PyEval_SaveThread(); } while (0)  // 开始线程保护代码段
#define SLU_END_THREADS   do { if (_save) { PyEval_RestoreThread((PyThreadState*)_save); _save = NULL;} } while (0)  // 结束线程保护代码段

#endif /* __SUPERLU_OBJECT */
/*
 * Definitions for other SuperLU data types than Z,
 * and type-generic definitions.
 */

// 定义宏CHECK_SLU_TYPE，用于检查给定的数据类型是否为支持的数值类型（单精度浮点、双精度浮点、复数单精度浮点、复数双精度浮点）
#define CHECK_SLU_TYPE(type) \
    (type == NPY_FLOAT || type == NPY_DOUBLE || type == NPY_CFLOAT || type == NPY_CDOUBLE)

// 定义宏TYPE_GENERIC_FUNC，用于生成根据数据类型调用不同函数的通用函数
#define TYPE_GENERIC_FUNC(name, returntype)                \
    returntype s##name(name##_ARGS);                       \
    returntype d##name(name##_ARGS);                       \
    returntype c##name(name##_ARGS);                       \
    static returntype name(int type, name##_ARGS)          \
    {                                                      \
        switch(type) {                                     \
        case NPY_FLOAT:   s##name(name##_ARGS_REF); break; \
        case NPY_DOUBLE:  d##name(name##_ARGS_REF); break; \
        case NPY_CFLOAT:  c##name(name##_ARGS_REF); break; \
        case NPY_CDOUBLE: z##name(name##_ARGS_REF); break; \
        default: return;                                   \
        }                                                  \
    }

// 定义宏SLU_TYPECODE_TO_NPY，用于将SuperLU库中的数据类型转换为NumPy中对应的数据类型
#define SLU_TYPECODE_TO_NPY(s)                    \
    ( ((s) == SLU_S) ? NPY_FLOAT :                \
      ((s) == SLU_D) ? NPY_DOUBLE :               \
      ((s) == SLU_C) ? NPY_CFLOAT :               \
      ((s) == SLU_Z) ? NPY_CDOUBLE : -1)

// 定义宏NPY_TYPECODE_TO_SLU，用于将NumPy中的数据类型转换为SuperLU库中对应的数据类型
#define NPY_TYPECODE_TO_SLU(s)                    \
    ( ((s) == NPY_FLOAT) ? SLU_S :                \
      ((s) == NPY_DOUBLE) ? SLU_D :               \
      ((s) == NPY_CFLOAT) ? SLU_C :               \
      ((s) == NPY_CDOUBLE) ? SLU_Z : -1)

// 定义宏gstrf_ARGS和gstrf_ARGS_REF，用于声明和引用gstrf函数的参数列表
#define gstrf_ARGS                                                  \
    superlu_options_t *a, SuperMatrix *b,                           \
    int c, int d, int *e, void *f, int g,                           \
    int *h, int *i, SuperMatrix *j, SuperMatrix *k,                 \
    GlobalLU_t *l, SuperLUStat_t *m, int *n
#define gstrf_ARGS_REF a,b,c,d,e,f,g,h,i,j,k,l,m,n

// 定义宏gsitrf_ARGS和gsitrf_ARGS_REF，用于声明和引用gsitrf函数的参数列表，与gstrf相同
#define gsitrf_ARGS gstrf_ARGS
#define gsitrf_ARGS_REF gstrf_ARGS_REF

// 定义宏gstrs_ARGS和gstrs_ARGS_REF，用于声明和引用gstrs函数的参数列表
#define gstrs_ARGS                              \
    trans_t a, SuperMatrix *b, SuperMatrix *c,  \
    int *d, int *e, SuperMatrix *f,             \
    SuperLUStat_t *g, int *h
#define gstrs_ARGS_REF a,b,c,d,e,f,g,h

// 定义宏gssv_ARGS和gssv_ARGS_REF，用于声明和引用gssv函数的参数列表
#define gssv_ARGS                                               \
    superlu_options_t *a, SuperMatrix *b, int *c, int *d,       \
    SuperMatrix *e, SuperMatrix *f, SuperMatrix *g,             \
    SuperLUStat_t *h, int *i
#define gssv_ARGS_REF a,b,c,d,e,f,g,h,i

// 定义宏Create_Dense_Matrix_ARGS和Create_Dense_Matrix_ARGS_REF，用于声明和引用Create_Dense_Matrix函数的参数列表
#define Create_Dense_Matrix_ARGS                               \
    SuperMatrix *a, int b, int c, void *d, int e,              \
    Stype_t f, Dtype_t g, Mtype_t h
#define Create_Dense_Matrix_ARGS_REF a,b,c,d,e,f,g,h

// 定义宏Create_CompRow_Matrix_ARGS和Create_CompRow_Matrix_ARGS_REF，用于声明和引用Create_CompRow_Matrix函数的参数列表
#define Create_CompRow_Matrix_ARGS              \
    SuperMatrix *a, int b, int c, int d,        \
    void *e, int *f, int *g,                    \
    Stype_t h, Dtype_t i, Mtype_t j
#define Create_CompRow_Matrix_ARGS_REF a,b,c,d,e,f,g,h,i,j

// 定义宏Create_CompCol_Matrix_ARGS，与Create_CompRow_Matrix_ARGS相同
#define Create_CompCol_Matrix_ARGS Create_CompRow_Matrix_ARGS
// 将 Create_CompCol_Matrix_ARGS_REF 定义为 Create_CompRow_Matrix_ARGS_REF 的别名

#define Create_CompCol_Matrix_ARGS_REF Create_CompRow_Matrix_ARGS_REF

// 定义 Create_SuperNode_Matrix_ARGS 宏，用于声明接受多个参数的函数签名
#define Create_SuperNode_Matrix_ARGS \
    SuperMatrix *a, int b, int c, int d, \
    void* e, int* f, int* g, int* h, int* i, int* j, \
    Stype_t k, Dtype_t l, Mtype_t m

// 定义 Create_SuperNode_Matrix_ARGS_REF 宏，用于展开 Create_SuperNode_Matrix 函数的参数列表
#define Create_SuperNode_Matrix_ARGS_REF a,b,c,d,e,f,g,h,i,j,k,l,m

// 声明一个通用类型的函数 gstrf，无返回值
TYPE_GENERIC_FUNC(gstrf, void);

// 声明一个通用类型的函数 gsitrf，无返回值
TYPE_GENERIC_FUNC(gsitrf, void);

// 声明一个通用类型的函数 gstrs，无返回值
TYPE_GENERIC_FUNC(gstrs, void);

// 声明一个通用类型的函数 gssv，无返回值
TYPE_GENERIC_FUNC(gssv, void);

// 声明一个通用类型的函数 Create_Dense_Matrix，无返回值
TYPE_GENERIC_FUNC(Create_Dense_Matrix, void);

// 声明一个通用类型的函数 Create_CompRow_Matrix，无返回值
TYPE_GENERIC_FUNC(Create_CompRow_Matrix, void);

// 声明一个通用类型的函数 Create_CompCol_Matrix，无返回值
TYPE_GENERIC_FUNC(Create_CompCol_Matrix, void);

// 声明一个通用类型的函数 Create_SuperNode_Matrix，无返回值
TYPE_GENERIC_FUNC(Create_SuperNode_Matrix, void);

// 结束条件，关闭 __SUPERLU_OBJECT 宏定义区域
#endif                /* __SUPERLU_OBJECT */
```