# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\_superluobject.c`

```
/* -*-c-*-  */
/*
 * _superlu object
 *
 * Python object representing SuperLU factorization + some utility functions.
 */

#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_superlu_ARRAY_API

#include "_superluobject.h"
#include <ctype.h>


/***********************************************************************
 * SuperLUObject methods
 */

static PyObject *SuperLU_solve(SuperLUObject * self, PyObject * args,
                               PyObject * kwds)
{
    volatile PyArrayObject *b, *x = NULL;
    volatile SuperMatrix B = { 0 };
    volatile int itrans = 'N';
    volatile int info;
    volatile trans_t trans;
    volatile SuperLUStat_t stat = { 0 };
    static char *kwlist[] = { "rhs", "trans", NULL };
    volatile jmp_buf *jmpbuf_ptr;
    SLU_BEGIN_THREADS_DEF;

    if (!CHECK_SLU_TYPE(self->type)) {
        PyErr_SetString(PyExc_ValueError, "unsupported data type");
        return NULL;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|C", kwlist,
                                     &PyArray_Type, &b, &itrans))
        return NULL;

    /* solve transposed system: matrix was passed row-wise instead of
     * column-wise */
    if (itrans == 'n' || itrans == 'N')
        trans = NOTRANS;
    else if (itrans == 't' || itrans == 'T')
        trans = TRANS;
    else if (itrans == 'h' || itrans == 'H')
        trans = CONJ;
    else {
        PyErr_SetString(PyExc_ValueError, "trans must be N, T, or H");
        return NULL;
    }

    x = (PyArrayObject*)PyArray_FROMANY(
        (PyObject*)b, self->type, 1, 2,
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (x == NULL) {
        goto fail;
    }

    if (PyArray_DIM((PyArrayObject*)x, 0) != self->n) {
        PyErr_SetString(PyExc_ValueError, "b is of incompatible size");
        goto fail;
    }

    if (DenseSuper_from_Numeric((SuperMatrix*)&B, (PyObject *)x))
        goto fail;

    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        goto fail;
    }

    StatInit((SuperLUStat_t *)&stat);

    /* Solve the system, overwriting vector x. */
    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    SLU_BEGIN_THREADS;
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        SLU_END_THREADS;
        goto fail;
    }
    gstrs(self->type,
          trans, &self->L, &self->U, self->perm_c, self->perm_r,
          (SuperMatrix *)&B, (SuperLUStat_t *)&stat, (int *)&info);
    SLU_END_THREADS;

    if (info) {
        PyErr_SetString(PyExc_SystemError,
                        "gstrs was called with invalid arguments");
        goto fail;
    }

    /* free memory */
    Destroy_SuperMatrix_Store((SuperMatrix *)&B);
    StatFree((SuperLUStat_t *)&stat);
    return (PyObject *) x;

  fail:
    XDestroy_SuperMatrix_Store((SuperMatrix *)&B);
    XStatFree((SuperLUStat_t *)&stat);
    Py_XDECREF(x);
    return NULL;
}

/** table of object methods
 */
PyMethodDef SuperLU_methods[] = {
    {"solve", (PyCFunction) SuperLU_solve, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL}                /* sentinel */



    // 创建一个Python C扩展模块中的方法和名称对应表的条目，名称为"solve"
    {"solve", (PyCFunction) SuperLU_solve, METH_VARARGS | METH_KEYWORDS, NULL},
    // 创建一个表明对名称和函数指针的结束的"哨兵"条目
    {NULL, NULL}                /* sentinel */
};

/***********************************************************************
 * SuperLUType methods
 */

// 释放 SuperLU 对象的内存
static void SuperLU_dealloc(SuperLUObject * self)
{
    // 释放缓存的 U 矩阵对象
    Py_XDECREF(self->cached_U);
    // 释放缓存的 L 矩阵对象
    Py_XDECREF(self->cached_L);
    // 释放 Python 回调函数对象
    Py_XDECREF(self->py_csc_construct_func);
    // 将缓存的 U 置为 NULL
    self->cached_U = NULL;
    // 将缓存的 L 置为 NULL
    self->cached_L = NULL;
    // 将 Python 回调函数置为 NULL
    self->py_csc_construct_func = NULL;
    // 释放行置换数组
    SUPERLU_FREE(self->perm_r);
    // 释放列置换数组
    SUPERLU_FREE(self->perm_c);
    // 将行置换数组置为 NULL
    self->perm_r = NULL;
    // 将列置换数组置为 NULL
    self->perm_c = NULL;
    // 销毁超节点矩阵 L
    XDestroy_SuperNode_Matrix(&self->L);
    // 销毁压缩列矩阵 U
    XDestroy_CompCol_Matrix(&self->U);
    // 释放 SuperLU 对象本身
    PyObject_Del(self);
}

// SuperLU 对象属性获取器
static PyObject *SuperLU_getter(PyObject *selfp, void *data)
{
    SuperLUObject *self = (SuperLUObject *)selfp;
    char *name = (char*)data;

    // 如果请求获取 shape 属性
    if (strcmp(name, "shape") == 0) {
        // 返回包含 m 和 n 的元组
        return Py_BuildValue("(i,i)", self->m, self->n);
    }
    // 如果请求获取 nnz 属性
    else if (strcmp(name, "nnz") == 0)
        // 返回非零元素个数的整数值
        return Py_BuildValue("i",
                             ((SCformat *) self->L.Store)->nnz +
                             ((SCformat *) self->U.Store)->nnz);
    // 如果请求获取 perm_r 属性
    else if (strcmp(name, "perm_r") == 0) {
        PyObject *perm_r;
        // 创建一个 NumPy 数组来保存行置换数组
        perm_r = PyArray_SimpleNewFromData(
            1, (npy_intp *) (&self->n), NPY_INT,
            (void *) self->perm_r);
        // 检查数组创建是否成功
        if (perm_r == NULL) {
            return NULL;
        }

        /* For ref counting of the memory */
        // 设置数组的基础对象为 SuperLU 对象，增加 SuperLU 对象的引用计数
        PyArray_SetBaseObject((PyArrayObject*)perm_r, (PyObject*)self);
        Py_INCREF(self);
        // 返回创建的行置换数组
        return perm_r;
    }
    // 如果请求获取 perm_c 属性
    else if (strcmp(name, "perm_c") == 0) {
        PyObject *perm_c;

        // 创建一个 NumPy 数组来保存列置换数组
        perm_c = PyArray_SimpleNewFromData(
            1, (npy_intp *) (&self->n), NPY_INT,
            (void *) self->perm_c);
        // 检查数组创建是否成功
        if (perm_c == NULL) {
            return NULL;
        }

        /* For ref counting of the memory */
        // 设置数组的基础对象为 SuperLU 对象，增加 SuperLU 对象的引用计数
        PyArray_SetBaseObject((PyArrayObject*)perm_c, (PyObject*)self);
        Py_INCREF(self);
        // 返回创建的列置换数组
        return perm_c;
    }
    // 如果请求获取 U 或者 L 属性
    else if (strcmp(name, "U") == 0 || strcmp(name, "L") == 0) {
        int ok;
        // 如果缓存的 U 矩阵对象为空
        if (self->cached_U == NULL) {
            // 调用 LU_to_csc_matrix 函数生成 U 和 L 的 CSC 格式矩阵
            ok = LU_to_csc_matrix(&self->L, &self->U,
                                  &self->cached_L, &self->cached_U,
                                  self->py_csc_construct_func);
            // 如果转换过程出错，则返回 NULL
            if (ok != 0) {
                return NULL;
            }
        }
        // 如果请求获取 U 属性
        if (strcmp(name, "U") == 0) {
            // 增加缓存的 U 矩阵对象的引用计数并返回
            Py_INCREF(self->cached_U);
            return self->cached_U;
        }
        // 如果请求获取 L 属性
        else {
            // 增加缓存的 L 矩阵对象的引用计数并返回
            Py_INCREF(self->cached_L);
            return self->cached_L;
        }
    }
    // 如果请求的属性名不是 shape、nnz、perm_r、perm_c、U 或 L
    else {
        // 抛出运行时错误，指示这是一个 bug
        PyErr_SetString(PyExc_RuntimeError,
                        "internal error (this is a bug)");
        return NULL;
    }
}


/***********************************************************************
 * SuperLUType structure
 */

// SuperLU 对象的属性定义
PyGetSetDef SuperLU_getset[] = {
    {"shape", SuperLU_getter, (setter)NULL, (char*)NULL, (void*)"shape"},
    {"nnz", SuperLU_getter, (setter)NULL, (char*)NULL, (void*)"nnz"},
    # 创建一个包含多个条目的静态键值对数组
    {"perm_r", SuperLU_getter, (setter)NULL, (char*)NULL, (void*)"perm_r"},
    {"perm_c", SuperLU_getter, (setter)NULL, (char*)NULL, (void*)"perm_c"},
    {"U", SuperLU_getter, (setter)NULL, (char*)NULL, (void*)"U"},
    {"L", SuperLU_getter, (setter)NULL, (char*)NULL, (void*)"L"},
    # 数组的结束标志，用于指示键值对数组的结束
    {NULL}
};

// 定义 SuperLUType 结构体，表示 Python 中的 SuperLU 类型
PyTypeObject SuperLUType = {
    PyVarObject_HEAD_INIT(NULL, 0)    // 初始化对象头
    "SuperLU",                        // 类型名称为 "SuperLU"
    sizeof(SuperLUObject),            // 结构体大小为 SuperLUObject 的大小
    0,                                // 初始引用计数
    (destructor) SuperLU_dealloc,     /* tp_dealloc */ // 析构函数指针
    0,                                /* tp_print */   // 打印函数指针
    0,                                /* tp_getattr */ // 获取属性函数指针
    0,                                /* tp_setattr */ // 设置属性函数指针
    0,                                /* tp_compare / tp_reserved */ // 比较函数指针
    0,                                /* tp_repr */    // 表示函数指针
    0,                                /* tp_as_number */ // 数值协议函数指针
    0,                                /* tp_as_sequence */ // 序列协议函数指针
    0,                                /* tp_as_mapping */ // 映射协议函数指针
    0,                                /* tp_hash */    // 哈希函数指针
    0,                                /* tp_call */    // 调用对象函数指针
    0,                                /* tp_str */     // 字符串表示函数指针
    0,                                /* tp_getattro */ // 获取属性函数指针
    0,                                /* tp_setattro */ // 设置属性函数指针
    0,                                /* tp_as_buffer */ // 缓冲协议函数指针
    Py_TPFLAGS_DEFAULT,               /* tp_flags */   // 默认标志位
    NULL,                             /* tp_doc */     // 文档字符串
    0,                                /* tp_traverse */ // 遍历对象函数指针
    0,                                /* tp_clear */   // 清除对象函数指针
    0,                                /* tp_richcompare */ // 富比较函数指针
    0,                                /* tp_weaklistoffset */ // 弱引用偏移量
    0,                                /* tp_iter */    // 迭代器函数指针
    0,                                /* tp_iternext */ // 迭代器下一个函数指针
    SuperLU_methods,                  /* tp_methods */ // 方法列表
    0,                                /* tp_members */ // 成员变量列表
    SuperLU_getset,                   /* tp_getset */  // 获取/设置方法列表
    0,                                /* tp_base */    // 基类指针
    0,                                /* tp_dict */    // 字典描述
    0,                                /* tp_descr_get */ // 获取描述符函数指针
    0,                                /* tp_descr_set */ // 设置描述符函数指针
    0,                                /* tp_dictoffset */ // 字典偏移量
    0,                                /* tp_init */    // 初始化函数指针
    0,                                /* tp_alloc */   // 分配函数指针
    0,                                /* tp_new */     // 新建对象函数指针
    0,                                /* tp_free */    // 释放函数指针
    0,                                /* tp_is_gc */   // 垃圾回收函数指针
    0,                                /* tp_bases */   // 基类元组
    0,                                /* tp_mro */     // 方法解析顺序元组
    0,                                /* tp_cache */   // 缓存字典
    0,                                /* tp_subclasses */ // 子类链表
    0,                                /* tp_weaklist */ // 弱引用列表
    0,                                /* tp_del */     // 删除函数指针
    0,                                /* tp_version_tag */ // 版本标签
};

// 从 SuperMatrix 结构体转换为 Python 对象的函数
int DenseSuper_from_Numeric(SuperMatrix *X, PyObject *PyX)
{
    volatile PyArrayObject *aX;        // Python 中的数组对象
    volatile int m, n, ldx, nd;        // 变量定义，包括维度信息和跳转缓冲区

    // 检查 PyX 是否为数组对象
    if (!PyArray_Check(PyX)) {
        PyErr_SetString(PyExc_TypeError,
                        "argument is not an array.");
        return -1;
    }

    // 将 PyX 转换为 PyArrayObject 类型的 aX
    aX = (PyArrayObject*)PyX;

    // 检查数组元素类型是否为支持的 SuperLU 类型
    if (!CHECK_SLU_TYPE(PyArray_TYPE((PyArrayObject*)aX))) {
        PyErr_SetString(PyExc_ValueError, "unsupported array data type");
        return -1;
    }

    // 检查数组是否为 Fortran 连续存储
    if (!(PyArray_FLAGS((PyArrayObject*)aX) & NPY_ARRAY_F_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "array is not fortran contiguous");
        return -1;
    }

    // 获取数组的维度
    nd = PyArray_NDIM((PyArrayObject*)aX);
    # 如果数组的维度是1
    if (nd == 1) {
        # 获取数组 aX 的第一个维度大小作为 m
        m = PyArray_DIM((PyArrayObject*)aX, 0);
        # 设置 n 为 1
        n = 1;
        # 设置 ldx 为 m
        ldx = m;
    }
    # 如果数组的维度是2
    else if (nd == 2) {
        # 获取数组 aX 的第一个维度大小作为 m
        m = PyArray_DIM((PyArrayObject*)aX, 0);
        # 获取数组 aX 的第二个维度大小作为 n
        n = PyArray_DIM((PyArrayObject*)aX, 1);
        # 设置 ldx 为 m
        ldx = m;
    }
    else {
        # 如果维度不是1或2，则设置错误信息并返回 -1
        PyErr_SetString(PyExc_ValueError, "wrong number of dimensions in array");
        return -1;
    }

    # 获取当前线程的跳转缓存指针
    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    # 如果发生跳转（错误情况）
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        # 返回 -1 表示错误
        return -1;
    }
    else {
        # 否则，创建一个稠密矩阵 X，使用 aX 的数据填充
        Create_Dense_Matrix(PyArray_TYPE((PyArrayObject*)aX), X, m, n,
                            PyArray_DATA((PyArrayObject*)aX), ldx, SLU_DN,
                            NPY_TYPECODE_TO_SLU(PyArray_TYPE((PyArrayObject*)aX)),
                            SLU_GE);
    }
    # 成功执行，返回 0
    return 0;
/* Natively handles Compressed Sparse Row and CSC */

/* 
   Convert a sparse matrix in SuperMatrix format to Compressed Sparse Row (CSR) format.
   Parameters:
   - A: SuperMatrix representing the sparse matrix
   - m: Number of rows
   - n: Number of columns
   - nnz: Number of non-zeros in the matrix
   - nzvals: NumPy array of non-zero values in the matrix
   - colind: NumPy array of column indices for each non-zero element
   - rowptr: NumPy array of row pointers for the CSR format
   - typenum: Type number indicating the data type of matrix elements
   Returns:
   - 0 on success, -1 on failure
*/
int NRFormat_from_spMatrix(SuperMatrix * A, int m, int n, int nnz,
                           PyArrayObject * nzvals, PyArrayObject * colind,
                           PyArrayObject * rowptr, int typenum)
{
    /* Invoke SparseFormat_from_spMatrix to convert to CSR format */
    return SparseFormat_from_spMatrix(A, m, n, nnz, 1, nzvals, colind, rowptr,
                                      typenum, SLU_NR, SLU_GE, NULL, NULL);
}

/* 
   Convert a sparse matrix in SuperMatrix format to Compressed Sparse Column (CSC) format.
   Parameters:
   - A: SuperMatrix representing the sparse matrix
   - m: Number of rows
   - n: Number of columns
   - nnz: Number of non-zeros in the matrix
   - nzvals: NumPy array of non-zero values in the matrix
   - rowind: NumPy array of row indices for each non-zero element
   - colptr: NumPy array of column pointers for the CSC format
   - typenum: Type number indicating the data type of matrix elements
   Returns:
   - 0 on success, -1 on failure
*/
int NCFormat_from_spMatrix(SuperMatrix * A, int m, int n, int nnz,
                           PyArrayObject * nzvals, PyArrayObject * rowind,
                           PyArrayObject * colptr, int typenum)
{
    /* Invoke SparseFormat_from_spMatrix to convert to CSC format */
    return SparseFormat_from_spMatrix(A, m, n, nnz, 0, nzvals, rowind, colptr,
                                      typenum, SLU_NC, SLU_GE, NULL, NULL);
}

/* 
   Create a sparse matrix in CSR, CSC, or supernodal CSC format from a SuperMatrix.
   Parameters:
   - A: SuperMatrix representing the sparse matrix
   - m: Number of rows
   - n: Number of columns
   - nnz: Number of non-zeros in the matrix
   - csr: Flag indicating matrix format (1=CSR, 0=CSC, -1=supernodal CSC)
   - nzvals: NumPy array of non-zero values in the matrix
   - indices: NumPy array of column or row indices for each non-zero element
   - pointers: NumPy array of row or column pointers for the matrix format
   - typenum: Type number indicating the data type of matrix elements
   - stype: SuperMatrix type (SLU_NR for CSR, SLU_NC for CSC)
   - mtype: Matrix type (SLU_GE for general)
   - identity_col_to_sup: Array mapping columns to supernodes (for supernodal CSC)
   - identity_sup_to_col: Array mapping supernodes to columns (for supernodal CSC)
   Returns:
   - 0 on success, -1 on failure
*/
int SparseFormat_from_spMatrix(SuperMatrix * A, int m, int n, int nnz, int csr,
                               PyArrayObject * nzvals,
                               PyArrayObject * indices,
                               PyArrayObject * pointers,
                               int typenum,
                               Stype_t stype, Mtype_t mtype,
                               int* identity_col_to_sup,
                               int* identity_sup_to_col)
{
    volatile int ok = 0;
    volatile jmp_buf *jmpbuf_ptr;

    /* Check if the input arrays meet the required conditions */
    ok = (PyArray_EquivTypenums(PyArray_TYPE(nzvals), typenum) &&
          PyArray_EquivTypenums(PyArray_TYPE(indices), NPY_INT) &&
          PyArray_EquivTypenums(PyArray_TYPE(pointers), NPY_INT) &&
          PyArray_NDIM(nzvals) == 1 &&
          PyArray_NDIM(indices) == 1 &&
          PyArray_NDIM(pointers) == 1 &&
          PyArray_IS_C_CONTIGUOUS(nzvals) &&
          PyArray_IS_C_CONTIGUOUS(indices) &&
          PyArray_IS_C_CONTIGUOUS(pointers) &&
          nnz <= PyArray_DIM(nzvals, 0) &&
          nnz <= PyArray_DIM(indices, 0) &&
          (csr ? (m+1) : (n+1)) <= PyArray_DIM(pointers, 0));
    if (!ok) {
        /* Raise a ValueError if the conditions are not met */
        PyErr_SetString(PyExc_ValueError,
                        "sparse matrix arrays must be 1-D C-contiguous and of proper "
                        "sizes and types");
        return -1;
    }

    /* Set up a jump point for error handling */
    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        return -1;
    }

    /* Continue with the conversion process */
    // The rest of the function body continues after this point
}
    # 如果进入此分支，表示 csr 不为 1、0 或 -1，这是一个错误情况
    else:
        # 检查 nzvals 的类型是否符合预期的稠密数组类型
        if (!CHECK_SLU_TYPE(PyArray_TYPE(nzvals))):
            # 设置类型错误异常，并返回 -1 表示错误
            PyErr_SetString(PyExc_TypeError, "Invalid type for array.");
            return -1;
        
        # 根据 csr 的值选择不同的稀疏矩阵类型进行创建
        if (csr == 1):
            # 创建 Compressed Row 格式的稀疏矩阵
            Create_CompRow_Matrix(PyArray_TYPE(nzvals),
                                  A, m, n, nnz, PyArray_DATA(nzvals),
                                  (int *) PyArray_DATA(indices),
                                  (int *) PyArray_DATA(pointers),
                                  stype,
                                  NPY_TYPECODE_TO_SLU(PyArray_TYPE(nzvals)),
                                  mtype);
        elif (csr == 0):
            # 创建 Compressed Column 格式的稀疏矩阵
            Create_CompCol_Matrix(PyArray_TYPE(nzvals),
                                  A, m, n, nnz, PyArray_DATA(nzvals),
                                  (int *) PyArray_DATA(indices),
                                  (int *) PyArray_DATA(pointers),
                                  stype,
                                  NPY_TYPECODE_TO_SLU(PyArray_TYPE(nzvals)),
                                  mtype);
        elif (csr == -1):
            # 创建超节点格式的稀疏矩阵，使用 pointers 参数初始化 nzval_colptr 和 rowind_colptr
            Create_SuperNode_Matrix(PyArray_TYPE(nzvals),
                                    A, m, n, nnz, PyArray_DATA(nzvals),
                                    (int *) PyArray_DATA(pointers),
                                    (int *) PyArray_DATA(indices),
                                    (int *) PyArray_DATA(pointers),
                                    identity_col_to_sup, identity_sup_to_col,
                                    stype,
                                    NPY_TYPECODE_TO_SLU(PyArray_TYPE(nzvals)),
                                    mtype);
    
    # 函数执行完毕，返回 0 表示成功
    return 0;
/*
 * Create Scipy sparse matrices out from Superlu LU decomposition.
 */

static int LU_to_csc(SuperMatrix *L, SuperMatrix *U,
                     int *U_indices, int *U_indptr, char *U_data,
                     int *L_indices, int *L_indptr, char *L_data,
                     Dtype_t dtype);
/*
 * 将Superlu的LU分解转换为Scipy稀疏矩阵格式的CSC（压缩稀疏列）格式。
 */

int LU_to_csc_matrix(SuperMatrix *L, SuperMatrix *U,
                     PyObject **L_csc, PyObject **U_csc,
                     PyObject *py_csc_construct_func)
{
    SCformat *Lstore;
    NCformat *Ustore;
    PyObject *U_indices = NULL, *U_indptr = NULL, *U_data = NULL;
    PyObject *L_indices = NULL, *L_indptr = NULL, *L_data = NULL;
    PyObject *datatuple = NULL, *shape = NULL;
    int result = -1, ok;
    int type;
    npy_intp dims[1];

    *L_csc = NULL;
    *U_csc = NULL;

    if (U->Stype != SLU_NC || L->Stype != SLU_SC ||
        U->Mtype != SLU_TRU || L->Mtype != SLU_TRLU ||
        L->nrow != U->nrow || L->ncol != L->nrow ||
        U->ncol != U->nrow || L->Dtype != U->Dtype)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "internal error: invalid Superlu matrix data");
        return -1;
    }

    Ustore = (NCformat*)U->Store;
    Lstore = (SCformat*)L->Store;

    type = SLU_TYPECODE_TO_NPY(L->Dtype);

    /* Allocate output */
#define CREATE_1D_ARRAY(name, type, size)               \
        do {                                            \
            dims[0] = size;                             \
            name = PyArray_EMPTY(1, dims, type, 0);     \
            if (name == NULL) goto fail;                \
        } while (0)

    CREATE_1D_ARRAY(L_indices, NPY_INT, Lstore->nnz);
    CREATE_1D_ARRAY(L_indptr, NPY_INT, L->ncol + 1);
    CREATE_1D_ARRAY(L_data, type, Lstore->nnz);

    CREATE_1D_ARRAY(U_indices, NPY_INT, Ustore->nnz);
    CREATE_1D_ARRAY(U_indptr, NPY_INT, U->ncol + 1);
    CREATE_1D_ARRAY(U_data, type, Ustore->nnz);

#undef CREATE_1D_ARRAY

    /* Copy data over */
    ok = LU_to_csc(
        L, U,
        (int*)PyArray_DATA((PyArrayObject*)L_indices),
        (int*)PyArray_DATA((PyArrayObject*)L_indptr),
        (void*)PyArray_DATA((PyArrayObject*)L_data),
        (int*)PyArray_DATA((PyArrayObject*)U_indices),
        (int*)PyArray_DATA((PyArrayObject*)U_indptr),
        (void*)PyArray_DATA((PyArrayObject*)U_data),
        L->Dtype
        );

    if (ok != 0) {
        goto fail;
    }

    /* Create sparse matrices */
    shape = Py_BuildValue("ii", L->nrow, L->ncol);
    if (shape == NULL) {
        goto fail;
    }

    datatuple = Py_BuildValue("OOO", L_data, L_indices, L_indptr);
    if (datatuple == NULL) {
        goto fail;
    }
    *L_csc = PyObject_CallFunction(py_csc_construct_func,
                                   "OO", datatuple, shape);
    if (*L_csc == NULL) {
        goto fail;
    }

    Py_DECREF(datatuple);
    datatuple = Py_BuildValue("OOO", U_data, U_indices, U_indptr);
    # 如果 datatuple 是 NULL，则执行以下操作
    if (datatuple == NULL) {
        # 释放 *L_csc 指向的 Python 对象，并将其置为 NULL
        Py_DECREF(*L_csc);
        *L_csc = NULL;
        # 跳转到 fail 标签处，表示执行失败
        goto fail;
    }
    # 使用 datatuple 和 shape 作为参数调用 py_csc_construct_func 函数来构造一个 PyObject 对象，并将其赋值给 *U_csc
    *U_csc = PyObject_CallFunction(py_csc_construct_func,
                                   "OO", datatuple, shape);
    # 如果 *U_csc 为 NULL，表示函数调用失败
    if (*U_csc == NULL) {
        # 释放 *L_csc 指向的 Python 对象，并将其置为 NULL
        Py_DECREF(*L_csc);
        *L_csc = NULL;
        # 跳转到 fail 标签处，表示执行失败
        goto fail;
    }

    # 将 result 的值设为 0
    result = 0;
fail:
    // 释放 U_indices 引用的对象
    Py_XDECREF(U_indices);
    // 释放 U_indptr 引用的对象
    Py_XDECREF(U_indptr);
    // 释放 U_data 引用的对象
    Py_XDECREF(U_data);
    // 释放 L_indices 引用的对象
    Py_XDECREF(L_indices);
    // 释放 L_indptr 引用的对象
    Py_XDECREF(L_indptr);
    // 释放 L_data 引用的对象
    Py_XDECREF(L_data);
    // 释放 shape 引用的对象
    Py_XDECREF(shape);
    // 释放 datatuple 引用的对象
    Py_XDECREF(datatuple);

    // 返回 result 变量
    return result;
}


/*
 * Convert SuperLU L and U matrices to CSC format.
 *
 * The LU decomposition U factor is partly stored in U and partly in the upper
 * diagonal of L.  The L matrix is stored in column-addressable rectangular
 * superblock format.
 *
 * This routine is partly adapted from SuperLU MATLAB wrappers and the
 * SuperLU Print_SuperNode_Matrix routine.
 */
static int
LU_to_csc(SuperMatrix *L, SuperMatrix *U,
          int *L_rowind, int *L_colptr, char *L_data,
          int *U_rowind, int *U_colptr, char *U_data,
          Dtype_t dtype)
{
    SCformat *Lstore;
    NCformat *Ustore;
    npy_intp elsize;
    int isup, icol, icolstart, icolend, iptr, istart, iend;
    char *src, *dst;
    int U_nnz, L_nnz;

    // 获取 U 和 L 矩阵的存储格式
    Ustore = (NCformat*)U->Store;
    Lstore = (SCformat*)L->Store;

    // 根据 dtype 确定元素大小 elsize
    switch (dtype) {
    case SLU_S: elsize = 4; break;
    case SLU_D: elsize = 8; break;
    case SLU_C: elsize = 8; break;
    case SLU_Z: elsize = 16; break;
    default:
        // 若出现未知的 dtype，抛出 ValueError 异常
        PyErr_SetString(PyExc_ValueError, "unknown dtype");
        return -1;
    }

    // 定义宏 IS_ZERO 判断元素是否为零
#define IS_ZERO(p)                                                      \
    ((dtype == SLU_S) ? (*(float*)(p) == 0) :                           \
     ((dtype == SLU_D) ? (*(double*)(p) == 0) :                         \
      ((dtype == SLU_C) ? (*(float*)(p) == 0 && *((float*)(p)+1) == 0) : \
       (*(double*)(p) == 0 && *((double*)(p)+1) == 0))))

    // 初始化 U 和 L 矩阵的列指针
    U_colptr[0] = 0;
    L_colptr[0] = 0;
    // 初始化 U 和 L 矩阵的非零元素数目
    U_nnz = 0;
    L_nnz = 0;

    // 循环处理每个超节点
    /* For each supernode */
    for (isup = 0; isup <= Lstore->nsuper; ++isup) {
        icolstart = Lstore->sup_to_col[isup];
        icolend = Lstore->sup_to_col[isup+1];
        istart = Lstore->rowind_colptr[icolstart];
        iend = Lstore->rowind_colptr[icolstart+1];

        /* 对每个超级节点中的每一列进行处理 */
        for (icol = icolstart; icol < icolend; ++icol) {

            /* 处理 Ustore 中的数据 */
            for (iptr = Ustore->colptr[icol]; iptr < Ustore->colptr[icol+1]; ++iptr) {
                src = (char*)Ustore->nzval + elsize * iptr;
                if (!IS_ZERO(src)) {
                    if (U_nnz >= Ustore->nnz)
                        goto size_error;
                    U_rowind[U_nnz] = Ustore->rowind[iptr];
                    /* 将 Ustore 中的非零值复制到 U_data */
                    dst = U_data + elsize * U_nnz;
                    memcpy(dst, src, elsize);
                    ++U_nnz;
                }
            }

            /* 处理 Lstore 中的数据 */
            src = (char*)Lstore->nzval + elsize * Lstore->nzval_colptr[icol];
            iptr = istart;

            /* 处理 L 的上三角部分 */
            for (; iptr < iend; ++iptr) {
                if (Lstore->rowind[iptr] > icol) {
                    break;
                }
                if (!IS_ZERO(src)) {
                    if (U_nnz >= Ustore->nnz)
                        goto size_error;
                    U_rowind[U_nnz] = Lstore->rowind[iptr];
                    dst = U_data + elsize * U_nnz;
                    memcpy(dst, src, elsize);
                    ++U_nnz;
                }
                src += elsize;
            }

            /* 在 L 的对角线上添加单位值 */
            if (L_nnz >= Lstore->nnz) return -1;
            dst = L_data + elsize * L_nnz;
            switch (dtype) {
            case SLU_S: *(float*)dst = 1.0; break;
            case SLU_D: *(double*)dst = 1.0; break;
            case SLU_C: *(float*)dst = 1.0; *((float*)dst+1) = 0.0; break;
            case SLU_Z: *(double*)dst = 1.0; *((double*)dst+1) = 0.0; break;
            }
            L_rowind[L_nnz] = icol;
            ++L_nnz;

            /* 处理 L 的下三角部分 */
            for (; iptr < iend; ++iptr) {
                if (!IS_ZERO(src)) {
                    if (L_nnz >= Lstore->nnz)
                         goto size_error;
                    L_rowind[L_nnz] = Lstore->rowind[iptr];
                    dst = L_data + elsize * L_nnz;
                    memcpy(dst, src, elsize);
                    ++L_nnz;
                }
                src += elsize;
            }

            /* 记录列指针 */
            U_colptr[icol+1] = U_nnz;
            L_colptr[icol+1] = L_nnz;
        }
    }

    return 0;
size_error:
    PyErr_SetString(PyExc_RuntimeError,
                    "internal error: superlu matrixes have wrong nnz");
    return -1;
}




PyObject *newSuperLUObject(SuperMatrix * A, PyObject * option_dict,
                           int intype, int ilu, PyObject * py_csc_construct_func)
{
    // A must be in SLU_NC format used by the factorization routine.
    volatile SuperLUObject *self;
    volatile SuperMatrix AC = { 0 };    /* Matrix postmultiplied by Pc */
    volatile int lwork = 0;
    volatile int *etree = NULL;
    volatile int info;
    volatile int n;
    volatile superlu_options_t options;
    volatile SuperLUStat_t stat = { 0 };
    volatile int panel_size, relax;
    volatile GlobalLU_t Glu;
    static volatile GlobalLU_t static_Glu;
    volatile GlobalLU_t *Glu_ptr;
    volatile jmp_buf *jmpbuf_ptr;
    SLU_BEGIN_THREADS_DEF;

    n = A->ncol;

    if (!set_superlu_options_from_dict((superlu_options_t*)&options, ilu, option_dict,
                                       (int*)&panel_size, (int*)&relax)) {
        return NULL;
    }

    // Create SLUObject
    self = PyObject_New(SuperLUObject, &SuperLUType);
    if (self == NULL)
        return PyErr_NoMemory();
    self->m = A->nrow;
    self->n = n;
    self->perm_r = NULL;
    self->perm_c = NULL;
    self->L.Store = NULL;
    self->U.Store = NULL;
    self->cached_U = NULL;
    self->cached_L = NULL;
    self->py_csc_construct_func = NULL;
    self->type = intype;

    jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
    if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
        goto fail;
    }

    // Calculate and apply minimum degree ordering
    etree = intMalloc(n);
    self->perm_r = intMalloc(n);
    self->perm_c = intMalloc(n);
    StatInit((SuperLUStat_t *)&stat);

    // Calculate column permutation
    get_perm_c(options.ColPerm, A, self->perm_c);

    // Apply column permutation
    sp_preorder((superlu_options_t*)&options, A, self->perm_c, (int*)etree,
                (SuperMatrix*)&AC);

    // Perform factorization
    if (!CHECK_SLU_TYPE(SLU_TYPECODE_TO_NPY(A->Dtype))) {
        PyErr_SetString(PyExc_ValueError, "Invalid type in SuperMatrix.");
        goto fail;
    }

    if (options.Fact == SamePattern || options.Fact == SamePattern_SameRowPerm) {
        // Use static Glu for same pattern or pattern with same row permutation
        Glu_ptr = &static_Glu;
    }
    else {
        // Use dynamic Glu and handle exceptions with threads
        Glu_ptr = &Glu;
        jmpbuf_ptr = (volatile jmp_buf *)superlu_python_jmpbuf();
        SLU_BEGIN_THREADS;
        if (setjmp(*(jmp_buf*)jmpbuf_ptr)) {
            SLU_END_THREADS;
            goto fail;
        }
    }

    // Factorize using iterative refinement (if ilu is true)
    gsitrf(SLU_TYPECODE_TO_NPY(A->Dtype),
           (superlu_options_t*)&options, (SuperMatrix*)&AC, relax, panel_size,
           (int*)etree, NULL, lwork, self->perm_c, self->perm_r,
           (SuperMatrix*)&self->L, (SuperMatrix*)&self->U, (GlobalLU_t*)Glu_ptr,
           (SuperLUStat_t*)&stat, (int*)&info);
    }
    else {
        # 调用超级 LU 库中的 gstrf 函数进行因式分解
        gstrf(SLU_TYPECODE_TO_NPY(A->Dtype),
              (superlu_options_t*)&options, (SuperMatrix*)&AC, relax, panel_size,
              (int*)etree, NULL, lwork, self->perm_c, self->perm_r,
              (SuperMatrix*)&self->L, (SuperMatrix*)&self->U, (GlobalLU_t*)Glu_ptr,
              (SuperLUStat_t*)&stat, (int*)&info);
    }

    # 结束线程安全操作
    SLU_END_THREADS;

    # 检查 gstrf 执行后的信息码
    if (info) {
        if (info < 0)
            PyErr_SetString(PyExc_SystemError,
                            "gstrf was called with invalid arguments");
        else {
            # 若信息码表明因式分解结果是奇异的情况
            if (info <= n)
                PyErr_SetString(PyExc_RuntimeError,
                                "Factor is exactly singular");
            else
                PyErr_NoMemory();
        }
        # 转到异常处理标签
        goto fail;
    }

    # 增加 Python 对象的引用计数，关联 CSC 构造函数
    Py_INCREF(py_csc_construct_func);
    self->py_csc_construct_func = py_csc_construct_func;

    /* free memory */
    # 释放内存
    SUPERLU_FREE((void*)etree);
    # 销毁 CompCol_Permuted 结构
    Destroy_CompCol_Permuted((SuperMatrix*)&AC);
    # 释放 SuperLUStat_t 结构占用的内存
    StatFree((SuperLUStat_t*)&stat);

    # 返回 Python 对象
    return (PyObject *) self;

  fail:
    # 释放内存并处理异常情况
    SUPERLU_FREE((void*)etree);
    # 彻底销毁 CompCol_Permuted 结构
    XDestroy_CompCol_Permuted((SuperMatrix*)&AC);
    # 释放 SuperLUStat_t 结构占用的内存
    XStatFree((SuperLUStat_t*)&stat);
    # 减少 Python 对象的引用计数并返回空指针
    Py_DECREF(self);
    return NULL;
/***********************************************************************
 * Preparing superlu_options_t
 */

/*
 * ENUM_CHECK_INIT macro: Initializes variables based on input type.
 * Sets 'i' to -1, 's' to an empty string, and 'tmpobj' to NULL.
 * Returns 1 immediately if 'input' is None.
 * Converts 'input' to string if it's bytes or Unicode, and converts to long if it's an integer.
 */
#define ENUM_CHECK_INIT                         \
    long i = -1;                                \
    char *s = "";                               \
    PyObject *tmpobj = NULL;                    \
    if (input == Py_None) return 1;             \
    if (PyBytes_Check(input)) {                \
        s = PyBytes_AS_STRING(input);          \
    }                                           \
    else if (PyUnicode_Check(input)) {          \
        tmpobj = PyUnicode_AsASCIIString(input);\
        if (tmpobj == NULL) return 0;           \
        s = PyBytes_AS_STRING(tmpobj);         \
    }                                           \
    else if (PyLong_Check(input)) {              \
        i = PyLong_AsLong(input);                \
    }

/*
 * ENUM_CHECK_FINISH macro: Cleans up and returns error message if needed.
 * Deallocates 'tmpobj', sets a ValueError with 'message', and returns 0.
 */
#define ENUM_CHECK_FINISH(message)              \
    Py_XDECREF(tmpobj);                         \
    PyErr_SetString(PyExc_ValueError, message); \
    return 0;

/*
 * ENUM_CHECK_NAME macro: Checks if input matches 'sname' or 'name' integer value.
 * If match is found, sets 'value' to 'name', deallocates 'tmpobj', and returns 1.
 */
#define ENUM_CHECK_NAME(name, sname)                    \
    if (my_strxcmp(s, sname) == 0 || i == (long)name) { \
        *value = name;                                  \
        Py_XDECREF(tmpobj);                             \
        return 1;                                       \
    }

/*
 * ENUM_CHECK macro: Wrapper around ENUM_CHECK_NAME for direct name matching.
 * Uses stringified 'name' for comparison.
 */
#define ENUM_CHECK(name) ENUM_CHECK_NAME(name, #name)

/*
 * my_strxcmp function: Compares two strings while ignoring case, underscores, and whitespace.
 * Returns 0 if strings match after ignoring specified characters, otherwise returns the comparison result.
 */
static int my_strxcmp(const char *a, const char *b)
{
    int c;

    while (*a != '\0' && *b != '\0') {
        while (*a == '_' || isspace(*a))
            ++a;
        while (*b == '_' || isspace(*b))
            ++b;
        c = (int) tolower(*a) - (int) tolower(*b);
        if (c != 0) {
            return c;
        }
        ++a;
        ++b;
    }
    return (int) tolower(*a) - (int) tolower(*b);
}

/*
 * yes_no_cvt function: Converts Python boolean input to corresponding yes_no_t value.
 * Sets 'value' to YES if input is True, NO if input is False, and raises ValueError for other types.
 * Returns 1 on successful conversion, 0 on error.
 */
static int yes_no_cvt(PyObject * input, yes_no_t * value)
{
    if (input == Py_None) {
        return 1;
    }
    else if (input == Py_True) {
        *value = YES;
    }
    else if (input == Py_False) {
        *value = NO;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "value not a boolean");
        return 0;
    }
    return 1;
}

/*
 * fact_cvt function: Converts Python input to corresponding fact_t enumeration value.
 * Uses ENUM_CHECK_INIT to initialize variables, and ENUM_CHECK/ENUM_CHECK_FINISH for comparison and error handling.
 * Returns 1 on successful conversion, 0 on error.
 */
static int fact_cvt(PyObject * input, fact_t * value)
{
    ENUM_CHECK_INIT;
    ENUM_CHECK(DOFACT);
    ENUM_CHECK(SamePattern);
    ENUM_CHECK(SamePattern_SameRowPerm);
    ENUM_CHECK(FACTORED);
    ENUM_CHECK_FINISH("invalid value for 'Fact' parameter");
}

/*
 * rowperm_cvt function: Converts Python input to corresponding rowperm_t enumeration value.
 * Uses ENUM_CHECK_INIT to initialize variables, and ENUM_CHECK/ENUM_CHECK_FINISH for comparison and error handling.
 * Returns 1 on successful conversion, 0 on error.
 */
static int rowperm_cvt(PyObject * input, rowperm_t * value)
{
    ENUM_CHECK_INIT;
    ENUM_CHECK(NOROWPERM);
    ENUM_CHECK(MY_PERMR);
    ENUM_CHECK_FINISH("invalid value for 'RowPerm' parameter");
}

/*
 * colperm_cvt function: Converts Python input to corresponding colperm_t enumeration value.
 * Uses ENUM_CHECK_INIT to initialize variables, and ENUM_CHECK/ENUM_CHECK_FINISH for comparison and error handling.
 * Returns 1 on successful conversion, 0 on error.
 */
static int colperm_cvt(PyObject * input, colperm_t * value)
{
    ENUM_CHECK_INIT;
    ENUM_CHECK(NATURAL);
    ENUM_CHECK(MMD_ATA);
    ENUM_CHECK(MMD_AT_PLUS_A);
    ENUM_CHECK(COLAMD);
    ENUM_CHECK(MY_PERMC);
    ENUM_CHECK_FINISH("invalid value for 'ColPerm' parameter");
}
static int trans_cvt(PyObject * input, trans_t * value)
{
    ENUM_CHECK_INIT;  // 初始化枚举检查
    ENUM_CHECK(NOTRANS);  // 检查枚举值是否为 NOTRANS
    ENUM_CHECK(TRANS);  // 检查枚举值是否为 TRANS
    ENUM_CHECK(CONJ);  // 检查枚举值是否为 CONJ
    if (my_strxcmp(s, "N") == 0) {  // 如果输入字符串与 "N" 相同
        *value = NOTRANS;  // 将枚举值设置为 NOTRANS
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "T") == 0) {  // 如果输入字符串与 "T" 相同
        *value = TRANS;  // 将枚举值设置为 TRANS
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "H") == 0) {  // 如果输入字符串与 "H" 相同
        *value = CONJ;  // 将枚举值设置为 CONJ
        return 1;  // 返回成功标志
    }
    ENUM_CHECK_FINISH("invalid value for 'Trans' parameter");  // 结束枚举检查，提示无效的 'Trans' 参数值
}

static int iterrefine_cvt(PyObject * input, IterRefine_t * value)
{
    ENUM_CHECK_INIT;  // 初始化枚举检查
    ENUM_CHECK(NOREFINE);  // 检查枚举值是否为 NOREFINE
    ENUM_CHECK(SLU_SINGLE);  // 检查枚举值是否为 SLU_SINGLE
    ENUM_CHECK_NAME(SLU_SINGLE, "SINGLE");  // 检查枚举值是否为 SLU_SINGLE，并验证名称为 "SINGLE"
    ENUM_CHECK(SLU_DOUBLE);  // 检查枚举值是否为 SLU_DOUBLE
    ENUM_CHECK_NAME(SLU_DOUBLE, "DOUBLE");  // 检查枚举值是否为 SLU_DOUBLE，并验证名称为 "DOUBLE"
    ENUM_CHECK(SLU_EXTRA);  // 检查枚举值是否为 SLU_EXTRA
    ENUM_CHECK_NAME(SLU_EXTRA, "EXTRA");  // 检查枚举值是否为 SLU_EXTRA，并验证名称为 "EXTRA"
    ENUM_CHECK_FINISH("invalid value for 'IterRefine' parameter");  // 结束枚举检查，提示无效的 'IterRefine' 参数值
}

static int norm_cvt(PyObject * input, norm_t * value)
{
    ENUM_CHECK_INIT;  // 初始化枚举检查
    ENUM_CHECK(ONE_NORM);  // 检查枚举值是否为 ONE_NORM
    ENUM_CHECK(TWO_NORM);  // 检查枚举值是否为 TWO_NORM
    ENUM_CHECK(INF_NORM);  // 检查枚举值是否为 INF_NORM
    ENUM_CHECK_FINISH("invalid value for 'ILU_Norm' parameter");  // 结束枚举检查，提示无效的 'ILU_Norm' 参数值
}

static int milu_cvt(PyObject * input, milu_t * value)
{
    ENUM_CHECK_INIT;  // 初始化枚举检查
    ENUM_CHECK(SILU);  // 检查枚举值是否为 SILU
    ENUM_CHECK(SMILU_1);  // 检查枚举值是否为 SMILU_1
    ENUM_CHECK(SMILU_2);  // 检查枚举值是否为 SMILU_2
    ENUM_CHECK(SMILU_3);  // 检查枚举值是否为 SMILU_3
    ENUM_CHECK_FINISH("invalid value for 'ILU_MILU' parameter");  // 结束枚举检查，提示无效的 'ILU_MILU' 参数值
}

static int droprule_one_cvt(PyObject * input, int *value)
{
    ENUM_CHECK_INIT;  // 初始化枚举检查
    if (my_strxcmp(s, "BASIC") == 0) {  // 如果输入字符串与 "BASIC" 相同
        *value = DROP_BASIC;  // 将值设置为 DROP_BASIC
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "PROWS") == 0) {  // 如果输入字符串与 "PROWS" 相同
        *value = DROP_PROWS;  // 将值设置为 DROP_PROWS
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "COLUMN") == 0) {  // 如果输入字符串与 "COLUMN" 相同
        *value = DROP_COLUMN;  // 将值设置为 DROP_COLUMN
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "AREA") == 0) {  // 如果输入字符串与 "AREA" 相同
        *value = DROP_AREA;  // 将值设置为 DROP_AREA
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "SECONDARY") == 0) {  // 如果输入字符串与 "SECONDARY" 相同
        *value = DROP_SECONDARY;  // 将值设置为 DROP_SECONDARY
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "DYNAMIC") == 0) {  // 如果输入字符串与 "DYNAMIC" 相同
        *value = DROP_DYNAMIC;  // 将值设置为 DROP_DYNAMIC
        return 1;  // 返回成功标志
    }
    if (my_strxcmp(s, "INTERP") == 0) {  // 如果输入字符串与 "INTERP" 相同
        *value = DROP_INTERP;  // 将值设置为 DROP_INTERP
        return 1;  // 返回成功标志
    }
    ENUM_CHECK_FINISH("invalid value for 'ILU_DropRule' parameter");  // 结束枚举检查，提示无效的 'ILU_DropRule' 参数值
}

static int droprule_cvt(PyObject * input, int *value)
{
    PyObject *seq = NULL;  // 初始化序列对象为 NULL
    int i;  // 声明整型变量 i
    int rule = 0;  // 声明整型变量 rule 并初始化为 0

    if (input == Py_None) {  // 如果输入对象是 Python 的 None
        /* Leave as default */  // 留作默认处理
        return 1;  // 返回成功标志
    }
    else if (PyLong_Check(input)) {  // 如果输入对象是 Python 的整数对象
        *value = PyLong_AsLong(input);  // 将 Python 的长整型转换为 C 的长整型，并赋值给 value
        return 1;  // 返回成功标志
    }
    else if (PyBytes_Check(input) || PyUnicode_Check(input)) {  // 如果输入对象是 Python 的字节对象或 Unicode 对象
        /* Comma-separated string */  // 逗号分隔的字符串
        char *fmt = "s";  // 声明字符串格式符 fmt 为 "s"
        if (PyBytes_Check(input)) {  // 如果输入对象是 Python 的字节对象
            fmt = "y";  // 将字符串格式符 fmt 设置为 "y"
        }
        seq = PyObject_CallMethod(input, "split", fmt, ",");  // 调用输入对象的 split 方法，按逗号分隔字符串
        if (seq == NULL || !PySequence_Check(seq))  // 如果 seq 为 NULL 或者不是序列对象
            goto fail;  // 跳转到失败处理
    }
    else if (PySequence_Check(input)) {  // 如果输入对象是 Python 的序列对象
        /* Sequence of strings or integers */  // 字符串或整数序列
        seq = input;  // 将输入对象赋值给 seq
        Py_INCREF(seq);  // 增加序列对象的引用计数
    }
    /* 如果条件不成立，设置异常并跳转到失败处理标签 */
    else {
        PyErr_SetString(PyExc_ValueError, "invalid value for drop rule");
        goto fail;
    }

    /* 对序列中的多个值进行按位或操作 */
    for (i = 0; i < PySequence_Size(seq); ++i) {
        PyObject *item;
        int one_value = 0;

        /* 获取序列中的第 i 个元素 */
        item = PySequence_ITEM(seq, i);
        if (item == NULL) {
            /* 如果获取失败，跳转到失败处理标签 */
            goto fail;
        }
        /* 调用函数 droprule_one_cvt 处理 item，并将结果存入 one_value */
        if (!droprule_one_cvt(item, &one_value)) {
            Py_DECREF(item);
            /* 如果处理失败，跳转到失败处理标签 */
            goto fail;
        }
        Py_DECREF(item);
        /* 将处理后的值按位或到 rule 中 */
        rule |= one_value;
    }
    /* 释放序列对象的引用 */
    Py_DECREF(seq);

    /* 将最终计算得到的 rule 值赋给 value 指针所指向的变量 */
    *value = rule;
    return 1;

  fail:
    /* 失败处理标签：释放序列对象的引用并返回失败标志 */
    Py_XDECREF(seq);
    return 0;
}

// 将 Python 对象转换为 double 类型，并存储到 value 指针指向的位置
static int double_cvt(PyObject * input, double *value)
{
    // 如果输入对象是 None，则直接返回成功
    if (input == Py_None)
        return 1;
    // 将输入对象转换为 double 类型，并存储到 value 中
    *value = PyFloat_AsDouble(input);
    // 检查转换过程中是否发生错误
    if (PyErr_Occurred())
        return 0;
    return 1;
}

// 将 Python 对象转换为 int 类型，并存储到 value 指针指向的位置
static int int_cvt(PyObject * input, int *value)
{
    // 如果输入对象是 None，则直接返回成功
    if (input == Py_None)
        return 1;
    // 将输入对象转换为 long 类型（Python 的 int 类型），并存储到 value 中
    *value = PyLong_AsLong(input);
    // 检查转换过程中是否发生错误
    if (PyErr_Occurred())
        return 0;
    return 1;
}

// 从给定的 Python 字典中设置 SuperLU 库的选项
int set_superlu_options_from_dict(superlu_options_t * options,
                                  int ilu, PyObject * option_dict,
                                  int *panel_size, int *relax)
{
    PyObject *args;
    int ret;
    int _relax, _panel_size;

    // 定义 Python 字典中可能包含的选项关键字
    static char *kwlist[] = {
        "Fact", "Equil", "ColPerm", "Trans", "IterRefine",
        "DiagPivotThresh", "PivotGrowth", "ConditionNumber",
        "RowPerm", "SymmetricMode", "PrintStat", "ReplaceTinyPivot",
        "SolveInitialized", "RefineInitialized", "ILU_Norm",
        "ILU_MILU", "ILU_DropTol", "ILU_FillTol", "ILU_FillFactor",
        "ILU_DropRule", "PanelSize", "Relax", NULL
    };

    // 如果需要 ILU 预处理，则设置 ILU 的默认选项
    if (ilu) {
        ilu_set_default_options(options);
    }
    else {
        // 否则，设置默认选项
        set_default_options(options);
    }

    // 获取当前环境中的面板大小和松弛因子
    _panel_size = sp_ienv(1);
    _relax = sp_ienv(2);

    // 如果 option_dict 是空，则使用默认选项
    if (option_dict == NULL) {
        /* Proceed with default options */
        ret = 1;
    }
    else {
        // 创建一个空元组
        args = PyTuple_New(0);
        // 解析传入的关键字参数，并根据指定格式转换成对应的C变量，返回值为解析成功与否的标志
        ret = PyArg_ParseTupleAndKeywords(args, option_dict,
                                          "|O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&O&",
                                          kwlist, fact_cvt, &options->Fact,
                                          yes_no_cvt, &options->Equil,
                                          colperm_cvt, &options->ColPerm,
                                          trans_cvt, &options->Trans,
                                          iterrefine_cvt, &options->IterRefine,
                                          double_cvt,
                                          &options->DiagPivotThresh,
                                          yes_no_cvt, &options->PivotGrowth,
                                          yes_no_cvt,
                                          &options->ConditionNumber,
                                          rowperm_cvt, &options->RowPerm,
                                          yes_no_cvt, &options->SymmetricMode,
                                          yes_no_cvt, &options->PrintStat,
                                          yes_no_cvt,
                                          &options->ReplaceTinyPivot,
                                          yes_no_cvt,
                                          &options->SolveInitialized,
                                          yes_no_cvt,
                                          &options->RefineInitialized,
                                          norm_cvt, &options->ILU_Norm,
                                          milu_cvt, &options->ILU_MILU,
                                          double_cvt, &options->ILU_DropTol,
                                          double_cvt, &options->ILU_FillTol,
                                          double_cvt, &options->ILU_FillFactor,
                                          droprule_cvt, &options->ILU_DropRule,
                                          int_cvt, &_panel_size, int_cvt,
                                          &_relax);
        // 释放创建的空元组
        Py_DECREF(args);
    }

    // 如果 panel_size 参数不为空指针，则将 _panel_size 的值赋给 panel_size
    if (panel_size != NULL) {
        *panel_size = _panel_size;
    }

    // 如果 relax 参数不为空指针，则将 _relax 的值赋给 relax
    if (relax != NULL) {
        *relax = _relax;
    }

    // 返回解析的结果标志
    return ret;
}



# 这行代码表示一个单独的右花括号 '}'，用于结束一个代码块或者数据结构的定义。
```