# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\_superlumodule.c`

```
/*
 * _superlu module
 *
 * Python interface to SuperLU decompositions.
 */

/* Copyright 1999 Travis Oliphant
 *
 * Permission to copy and modified this file is granted under
 * the revised BSD license. No warranty is expressed or IMPLIED
 */

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_superlu_ARRAY_API
#include <numpy/ndarrayobject.h>

#include "_superluobject.h"
#include "SuperLU/SRC/superlu_enum_consts.h"


/*
 * NULL-safe deconstruction functions
 */
void XDestroy_SuperMatrix_Store(SuperMatrix * A)
{
    Destroy_SuperMatrix_Store(A);    /* 销毁以 SuperMatrix 存储形式的矩阵 A */
    A->Store = NULL;    /* 将 A 的存储指针设为 NULL */
}

void XDestroy_SuperNode_Matrix(SuperMatrix * A)
{
    if (A->Store) {
        Destroy_SuperNode_Matrix(A);    /* 如果 A 的存储指针不为空，则销毁超节点存储形式的矩阵 A */
    }
    A->Store = NULL;    /* 将 A 的存储指针设为 NULL */
}

void XDestroy_CompCol_Matrix(SuperMatrix * A)
{
    if (A->Store) {
        Destroy_CompCol_Matrix(A);    /* 如果 A 的存储指针不为空，则销毁压缩列存储形式的矩阵 A */
    }
    A->Store = NULL;    /* 将 A 的存储指针设为 NULL */
}

void XDestroy_CompCol_Permuted(SuperMatrix * A)
{
    if (A->Store) {
        Destroy_CompCol_Permuted(A);    /* 如果 A 的存储指针不为空，则销毁经过置换的压缩列存储形式的矩阵 A */
    }
    A->Store = NULL;    /* 将 A 的存储指针设为 NULL */
}

void XStatFree(SuperLUStat_t * stat)
{
    if (stat->ops) {
        StatFree(stat);    /* 如果 stat 的操作不为空，则释放 stat 的统计信息 */
    }
    stat->ops = NULL;    /* 将 stat 的操作指针设为 NULL */
}


/*
 * Data-type dependent implementations for Xgssv and Xgstrf;
 *
 * These have to included from separate files because of SuperLU include
 * structure.
 */

static PyObject *Py_gssv(PyObject * self, PyObject * args,
             PyObject * kwdict)
{
    volatile PyObject *Py_B = NULL;
    volatile PyArrayObject *Py_X = NULL;
    volatile PyArrayObject *nzvals = NULL;
    volatile PyArrayObject *colind = NULL, *rowptr = NULL;
    volatile int N, nnz;
    volatile int info;
    volatile int csc = 0;
    volatile int *perm_r = NULL, *perm_c = NULL;
    volatile SuperMatrix A = { 0 }, B = { 0 }, L = { 0 }, U = { 0 };
    volatile superlu_options_t options = { 0 };
    volatile SuperLUStat_t stat = { 0 };
    volatile PyObject *option_dict = NULL;
    volatile int type;
    volatile jmp_buf *jmpbuf_ptr;
    SLU_BEGIN_THREADS_DEF;

    static char *kwlist[] = {
        "N", "nnz", "nzvals", "colind", "rowptr", "B", "csc",
        "options", NULL
    };

    /* Get input arguments */
    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "iiO!O!O!O|iO", kwlist,
                     &N, &nnz, &PyArray_Type, &nzvals,
                     &PyArray_Type, &colind, &PyArray_Type,
                     &rowptr, &Py_B, &csc, &option_dict)) {
        return NULL;    /* 解析输入参数失败，返回 NULL */
    }

    if (!_CHECK_INTEGER(colind) || !_CHECK_INTEGER(rowptr)) {
        PyErr_SetString(PyExc_TypeError,
            "colind and rowptr must be of type cint");
        return NULL;    /* colind 和 rowptr 不是 cint 类型，返回 NULL */
    }

    type = PyArray_TYPE((PyArrayObject*)nzvals);
    if (!CHECK_SLU_TYPE(type)) {
        PyErr_SetString(PyExc_TypeError,
            "nzvals is not of a type supported by SuperLU");
        return NULL;    /* nzvals 不是 SuperLU 支持的类型，返回 NULL */
    }

    if (!set_superlu_options_from_dict((superlu_options_t*)&options, 0,
                                       (PyObject*)option_dict, NULL, NULL)) {
        return NULL;    /* 从字典设置 SuperLU 选项失败，返回 NULL */
    }

    /* Create Space for output */
    Py_X = (PyArrayObject*)PyArray_FROMANY(
        (PyObject*)Py_B, type, 1, 2,
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    # 将 Py_B 转换为 PyArrayObject 类型的对象 Py_X，确保其是按照 Fortran 风格连续存储并进行必要的拷贝
    if (Py_X == NULL)
        return NULL;

    if (PyArray_DIM((PyArrayObject*)Py_X, 0) != N) {
        PyErr_SetString(PyExc_ValueError,
                        "b array has invalid shape");
        Py_DECREF(Py_X);
        return NULL;
    }

    if (csc) {
        if (NCFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz,
                                   (PyArrayObject *)nzvals, (PyArrayObject *)colind,
                                   (PyArrayObject *)rowptr, type)) {
            Py_DECREF(Py_X);
            return NULL;
        }
    }
    else {
        if (NRFormat_from_spMatrix((SuperMatrix*)&A, N, N, nnz, (PyArrayObject *)nzvals,
                                   (PyArrayObject *)colind, (PyArrayObject *)rowptr,
                                   type)) {
            Py_DECREF(Py_X);
            return NULL;
        }
    }

    if (DenseSuper_from_Numeric((SuperMatrix*)&B, (PyObject*)Py_X)) {
        Destroy_SuperMatrix_Store((SuperMatrix*)&A);
        Py_DECREF(Py_X);
        return NULL;
    }

    /* B and Py_X share same data now but Py_X "owns" it */

    /* 设置选项 */

    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    SLU_BEGIN_THREADS;
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        SLU_END_THREADS;
        goto fail;
    }
    else {
        perm_c = intMalloc(N);
        perm_r = intMalloc(N);
        StatInit((SuperLUStat_t*)&stat);

        /* 计算稀疏矩阵的直接逆 */
        gssv(type, (superlu_options_t*)&options, (SuperMatrix*)&A, (int*)perm_c, (int*)perm_r,
             (SuperMatrix*)&L, (SuperMatrix*)&U, (SuperMatrix*)&B, (SuperLUStat_t*)&stat,
             (int*)&info);
        SLU_END_THREADS;
    }

    SUPERLU_FREE((void*)perm_r);
    SUPERLU_FREE((void*)perm_c);
    Destroy_SuperMatrix_Store((SuperMatrix*)&A);    /* 仅持有数据的指针 */
    Destroy_SuperMatrix_Store((SuperMatrix*)&B);
    Destroy_SuperNode_Matrix((SuperMatrix*)&L);
    Destroy_CompCol_Matrix((SuperMatrix*)&U);
    StatFree((SuperLUStat_t*)&stat);

    return Py_BuildValue("Ni", Py_X, info);

  fail:
    SUPERLU_FREE((void*)perm_r);
    SUPERLU_FREE((void*)perm_c);
    XDestroy_SuperMatrix_Store((SuperMatrix*)&A);    /* 仅持有数据的指针 */
    XDestroy_SuperMatrix_Store((SuperMatrix*)&B);
    XDestroy_SuperNode_Matrix((SuperMatrix*)&L);
    XDestroy_CompCol_Matrix((SuperMatrix*)&U);
    XStatFree((SuperLUStat_t*)&stat);
    Py_XDECREF(Py_X);
    return NULL;
static PyObject *Py_gstrf(PyObject * self, PyObject * args,
              PyObject * keywds)
{
    /* default value for SuperLU parameters */
    // 定义 SuperLU 参数的默认值
    int N, nnz;
    PyArrayObject *rowind, *colptr, *nzvals;
    // 定义存储行指针、列指针、非零元素数组的 NumPy 数组对象
    SuperMatrix A = { 0 };
    // 定义 SuperMatrix 结构 A，并初始化为零
    PyObject *result;
    PyObject *py_csc_construct_func = NULL;
    PyObject *option_dict = NULL;
    int type;
    int ilu = 0;

    static char *kwlist[] = { "N", "nnz", "nzvals", "colind", "rowptr",
        "csc_construct_func", "options", "ilu",
    NULL
    };
    // 定义关键字参数列表

    int res =
    PyArg_ParseTupleAndKeywords(args, keywds, "iiO!O!O!O|Oi", kwlist,
                    &N, &nnz,
                    &PyArray_Type, &nzvals,
                    &PyArray_Type, &rowind,
                    &PyArray_Type, &colptr,
                    &py_csc_construct_func,
                    &option_dict,
                    &ilu);
    // 解析 Python 函数的输入参数，并赋值给对应的变量

    if (!res)
    return NULL;
    // 如果参数解析失败，返回空指针

    if (!_CHECK_INTEGER(colptr) || !_CHECK_INTEGER(rowind)) {
    PyErr_SetString(PyExc_TypeError,
            "rowind and colptr must be of type cint");
    return NULL;
    }
    // 检查 rowind 和 colptr 是否为 cint 类型，如果不是则抛出类型错误异常并返回空指针

    type = PyArray_TYPE((PyArrayObject*)nzvals);
    if (!CHECK_SLU_TYPE(type)) {
    PyErr_SetString(PyExc_TypeError,
            "nzvals is not of a type supported by SuperLU");
    return NULL;
    }
    // 获取 nzvals 的 NumPy 数组对象的数据类型，检查其是否为 SuperLU 支持的类型，
    // 如果不是则抛出类型错误异常并返回空指针

    if (NCFormat_from_spMatrix(&A, N, N, nnz, nzvals, rowind, colptr,
                   type)) {
    goto fail;
    }
    // 调用 NCFormat_from_spMatrix 函数，将输入参数转换为 SuperLU 中的稀疏矩阵格式
    // 如果转换失败，则跳转至 fail 标签处

    result = newSuperLUObject(&A, option_dict, type, ilu, py_csc_construct_func);
    if (result == NULL) {
    goto fail;
    }
    // 调用 newSuperLUObject 函数，创建一个新的 SuperLU 对象
    // 如果创建失败，则跳转至 fail 标签处

    /* arrays of input matrix will not be freed */
    // 输入矩阵的数组不会被释放
    Destroy_SuperMatrix_Store(&A);
    // 销毁 SuperMatrix 对象 A 的存储空间
    return result;

  fail:
    /* arrays of input matrix will not be freed */
    // 输入矩阵的数组不会被释放
    XDestroy_SuperMatrix_Store(&A);
    // 销毁 SuperMatrix 对象 A 的存储空间并返回空指针
    return NULL;
}
    // 如果 res 为假（NULL），返回空指针
    if (!res)
        return NULL;

    // 根据 itrans 的不同取值设置 trans 变量
    if (itrans == 'n' || itrans == 'N') {
        trans = NOTRANS;
    } else if (itrans == 't' || itrans == 'T') {
        trans = TRANS;
    } else if (itrans == 'h' || itrans == 'H') {
        trans = CONJ;
    } else {
        // 如果 itrans 不是 N, T, 或 H，设置异常并返回空指针
        PyErr_SetString(PyExc_ValueError, "trans must be N, T, or H");
        return NULL;
    }

    // 如果 L_N 和 U_N 不相等，设置异常并返回空指针
    if (L_N != U_N) {
        PyErr_SetString(PyExc_ValueError, "L and U must have the same dimension");
        return NULL;
    }

    // 检查 L_rowind, L_colptr, U_rowind, U_colptr 是否为 cint 类型，如果不是，设置异常并返回空指针
    if (!_CHECK_INTEGER(L_rowind) || !_CHECK_INTEGER(L_colptr) ||
        !_CHECK_INTEGER(U_rowind) || !_CHECK_INTEGER(U_colptr)) {
        PyErr_SetString(PyExc_TypeError, "row indices and column pointers must be of type cint");
        return NULL;
    }

    // 获取 L_nzvals 和 U_nzvals 的类型，并比较它们是否相同，如果不同，设置异常并返回空指针
    int L_type = PyArray_TYPE((PyArrayObject*)L_nzvals);
    int U_type = PyArray_TYPE((PyArrayObject*)U_nzvals);
    if (L_type != U_type) {
        PyErr_SetString(PyExc_TypeError,
                        "nzvals types of L and U differ");
        return NULL;
    }

    // 检查 L_nzvals 的类型是否受 SuperLU 支持，如果不支持，设置异常并返回空指针
    if (!CHECK_SLU_TYPE(L_type)) {
        PyErr_SetString(PyExc_TypeError,
                        "nzvals is not of a type supported by SuperLU");
        return NULL;
    }

    /* 创建 SuperLU 的 L 和 U 矩阵 */
    int* L_col_to_sup = intMalloc(L_N + 1);   // 分配 L_col_to_sup 数组的内存空间
    int* L_sup_to_col = intMalloc(L_N + 1);   // 分配 L_sup_to_col 数组的内存空间
    for (int i = 0; i <= L_N; i++) {
        L_col_to_sup[i] = i;   // 初始化 L_col_to_sup 和 L_sup_to_col 数组
        L_sup_to_col[i] = i;
    }
    L_col_to_sup[L_N] = L_N - 1;   // 最后一个元素特殊处理
    SuperMatrix L_super = {0};   // 初始化 L_super 和 U_super 为零
    SuperMatrix U_super = {0};
    // 将稀疏格式的 L_nzvals, L_rowind, L_colptr 转换为 SuperMatrix 的格式
    int L_conv_err = SparseFormat_from_spMatrix(
            &L_super, L_N, L_N, L_nnz, -1,
            (PyArrayObject*)L_nzvals, (PyArrayObject*)L_rowind, (PyArrayObject*)L_colptr,
            L_type, SLU_SC, SLU_TRLU, L_col_to_sup, L_sup_to_col);
    if (L_conv_err) {
        return NULL;   // 转换出错时返回空指针
    }
    // 将稀疏格式的 U_nzvals, U_rowind, U_colptr 转换为 SuperMatrix 的格式
    int U_conv_err = SparseFormat_from_spMatrix(
            &U_super, U_N, U_N, U_nnz, 0,
            (PyArrayObject*)U_nzvals, (PyArrayObject*)U_rowind, (PyArrayObject*)U_colptr,
            U_type, SLU_NC, SLU_TRU, NULL, NULL);
    if (U_conv_err) {
        Destroy_SuperMatrix_Store((SuperMatrix*)&L_super);   // 转换出错时销毁已创建的 L_super
        return NULL;   // 返回空指针
    }

    /* 读取右侧向量（即解向量） */
    PyArrayObject* X_arr = (PyArrayObject*)PyArray_FROMANY(
        (PyObject*)X_py, L_type, 1, 2,
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (X_arr == NULL) {
        SUPERLU_FREE((void*)L_col_to_sup);   // 释放 L_col_to_sup 和 L_sup_to_col 的内存
        SUPERLU_FREE((void*)L_sup_to_col);
        Destroy_SuperMatrix_Store((SuperMatrix*)&L_super);   // 销毁已创建的 L_super 和 U_super
        Destroy_SuperMatrix_Store((SuperMatrix*)&U_super);
        return NULL;   // 返回空指针
    }
    # 检查输入数组 X_arr 的第一维度是否等于 L_N
    if (PyArray_DIM((PyArrayObject*)X_arr, 0) != L_N) {
        # 如果不相等，设置异常字符串并释放相关资源后返回空指针
        PyErr_SetString(PyExc_ValueError,
                        "right hand side array has invalid shape");
        SUPERLU_FREE((void*)L_col_to_sup);
        SUPERLU_FREE((void*)L_sup_to_col);
        Destroy_SuperMatrix_Store((SuperMatrix*)&L_super);
        Destroy_SuperMatrix_Store((SuperMatrix*)&U_super);
        Py_DECREF(X_arr);
        return NULL;
    }

    # 创建一个名为 X 的 SuperMatrix 对象，从 Python 数组 X_arr 密集格式的数值构造出来
    SuperMatrix X;
    if (DenseSuper_from_Numeric((SuperMatrix*)&X, (PyObject*)X_arr)) {
        # 如果构造失败，释放资源后返回空指针
        SUPERLU_FREE((void*)L_col_to_sup);
        SUPERLU_FREE((void*)L_sup_to_col);
        Destroy_SuperMatrix_Store((SuperMatrix*)&L_super);
        Destroy_SuperMatrix_Store((SuperMatrix*)&U_super);
        Py_DECREF(X_arr);
        return NULL;
    } /* X and X_arr share the same data but X_arr "owns" it. */

    /* 调用 SuperLU 函数 */
    int info=0;
    SuperLUStat_t stat = { 0 };
    StatInit((SuperLUStat_t *)&stat);

    # 为列置换 perm_c 分配内存并初始化为自然顺序
    int* perm_c = intMalloc(L_N);
    for (int i=0; i<L_N; i++) {
        perm_c[i] = i;
    }
    # 行置换 perm_r 和列置换 perm_c 相同
    int* perm_r = perm_c;

    # 设置跳转点指向 superlu_python_jmpbuf 返回的 jmp_buf
    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    SLU_BEGIN_THREADS;
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        SLU_END_THREADS;
        goto fail;
    }
    # 调用 gstrs 函数进行超级 LU 分解求解
    gstrs(L_type, trans, &L_super, &U_super, perm_c, perm_r,
          (SuperMatrix *)&X, (SuperLUStat_t *)&stat, (int *)&info);
    SLU_END_THREADS;

    # 如果 gstrs 返回非零信息，设置异常字符串并跳转到 fail 标签处理错误
    if (info) {
        PyErr_SetString(PyExc_SystemError,
                        "gstrs was called with invalid arguments");
        goto fail;
    }

    /* 释放资源并返回结果 */
    SUPERLU_FREE((void*)L_col_to_sup);
    SUPERLU_FREE((void*)L_sup_to_col);
    SUPERLU_FREE((void*)perm_c);
    Destroy_SuperMatrix_Store(&L_super);
    Destroy_SuperMatrix_Store(&U_super);
    XStatFree((SuperLUStat_t *)&stat);

    # 返回 Python 对象 X_arr 和整数 info 的元组
    return Py_BuildValue("Ni", X_arr, info);

  fail:
    # 处理失败情况，释放资源并返回空指针
    SUPERLU_FREE((void*)L_col_to_sup);
    SUPERLU_FREE((void*)L_sup_to_col);
    SUPERLU_FREE((void*)perm_c);
    Destroy_SuperMatrix_Store(&L_super);
    Destroy_SuperMatrix_Store(&U_super);
    XStatFree((SuperLUStat_t *)&stat);
    Py_DECREF(X_arr);
    return NULL;
/*
 * Main SuperLU module
 */

// 定义 SuperLU 方法数组，包含了三个函数 gssv, gstrf, gstrs，每个函数对应一个 PyCFunction
static PyMethodDef SuperLU_Methods[] = {
    {"gssv", (PyCFunction) Py_gssv, METH_VARARGS | METH_KEYWORDS, // 定义 gssv 函数
     gssv_doc}, // gssv 函数的文档字符串
    {"gstrf", (PyCFunction) Py_gstrf, METH_VARARGS | METH_KEYWORDS, // 定义 gstrf 函数
     gstrf_doc}, // gstrf 函数的文档字符串
    {"gstrs", (PyCFunction) Py_gstrs, METH_VARARGS | METH_KEYWORDS, // 定义 gstrs 函数
     gstrs_doc}, // gstrs 函数的文档字符串
    {NULL, NULL} // 结束符号
};

// 定义模块结构体 moduledef
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, // 模块定义的初始化
    "_superlu", // 模块名称
    NULL, // 模块文档字符串
    -1, // 模块状态
    SuperLU_Methods, // 模块中的方法数组
    NULL, // 模块的槽
    NULL, // 模块的全局状态
    NULL, // 模块的导出函数
    NULL // 模块的销毁函数
};

// 初始化 Python 模块 _superlu
PyMODINIT_FUNC
PyInit__superlu(void)
{
    PyObject *module, *mdict;

    import_array(); // 导入数组模块

    // 准备 SuperLUType 类型对象
    if (PyType_Ready(&SuperLUType) < 0) {
        return NULL; // 初始化失败则返回空指针
    }
    // 准备 SuperLUGlobalType 类型对象
    if (PyType_Ready(&SuperLUGlobalType) < 0) {
        return NULL; // 初始化失败则返回空指针
    }

    // 创建 Python 模块对象
    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL; // 创建失败则返回空指针
    }
    // 获取模块字典对象
    mdict = PyModule_GetDict(module);
    if (mdict == NULL) {
        return NULL; // 获取失败则返回空指针
    }

    // 将 SuperLUType 类型对象添加到模块字典中
    if (PyDict_SetItemString(mdict, "SuperLU", (PyObject *) &SuperLUType)) {
        return NULL; // 设置失败则返回空指针
    }

    return module; // 返回创建的模块对象
}
```