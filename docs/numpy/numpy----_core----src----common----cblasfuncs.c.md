# `.\numpy\numpy\_core\src\common\cblasfuncs.c`

```
/*
 * This module provides a BLAS optimized matrix multiply,
 * inner product and dot for numpy arrays
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "npy_cblas.h"
#include "arraytypes.h"
#include "common.h"
#include "dtypemeta.h"

#include <assert.h>


static const double oneD[2] = {1.0, 0.0}, zeroD[2] = {0.0, 0.0};
static const float oneF[2] = {1.0, 0.0}, zeroF[2] = {0.0, 0.0};


/*
 * Helper: dispatch to appropriate cblas_?gemm for typenum.
 */
static void
gemm(int typenum, enum CBLAS_ORDER order,
     enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
     npy_intp m, npy_intp n, npy_intp k,
     PyArrayObject *A, npy_intp lda, PyArrayObject *B, npy_intp ldb, PyArrayObject *R)
{
    const void *Adata = PyArray_DATA(A), *Bdata = PyArray_DATA(B);
    void *Rdata = PyArray_DATA(R);
    npy_intp ldc = PyArray_DIM(R, 1) > 1 ? PyArray_DIM(R, 1) : 1;

    switch (typenum) {
        case NPY_DOUBLE:
            // Perform double precision matrix multiplication using cblas_dgemm
            CBLAS_FUNC(cblas_dgemm)(order, transA, transB, m, n, k, 1.,
                        Adata, lda, Bdata, ldb, 0., Rdata, ldc);
            break;
        case NPY_FLOAT:
            // Perform single precision matrix multiplication using cblas_sgemm
            CBLAS_FUNC(cblas_sgemm)(order, transA, transB, m, n, k, 1.f,
                        Adata, lda, Bdata, ldb, 0.f, Rdata, ldc);
            break;
        case NPY_CDOUBLE:
            // Perform complex double precision matrix multiplication using cblas_zgemm
            CBLAS_FUNC(cblas_zgemm)(order, transA, transB, m, n, k, oneD,
                        Adata, lda, Bdata, ldb, zeroD, Rdata, ldc);
            break;
        case NPY_CFLOAT:
            // Perform complex single precision matrix multiplication using cblas_cgemm
            CBLAS_FUNC(cblas_cgemm)(order, transA, transB, m, n, k, oneF,
                        Adata, lda, Bdata, ldb, zeroF, Rdata, ldc);
            break;
    }
}


/*
 * Helper: dispatch to appropriate cblas_?gemv for typenum.
 */
static void
gemv(int typenum, enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
     PyArrayObject *A, npy_intp lda, PyArrayObject *X, npy_intp incX,
     PyArrayObject *R)
{
    const void *Adata = PyArray_DATA(A), *Xdata = PyArray_DATA(X);
    void *Rdata = PyArray_DATA(R);

    npy_intp m = PyArray_DIM(A, 0), n = PyArray_DIM(A, 1);

    switch (typenum) {
        case NPY_DOUBLE:
            // Perform double precision matrix-vector multiplication using cblas_dgemv
            CBLAS_FUNC(cblas_dgemv)(order, trans, m, n, 1., Adata, lda, Xdata, incX,
                        0., Rdata, 1);
            break;
        case NPY_FLOAT:
            // Perform single precision matrix-vector multiplication using cblas_sgemv
            CBLAS_FUNC(cblas_sgemv)(order, trans, m, n, 1.f, Adata, lda, Xdata, incX,
                        0.f, Rdata, 1);
            break;
        case NPY_CDOUBLE:
            // Perform complex double precision matrix-vector multiplication using cblas_zgemv
            CBLAS_FUNC(cblas_zgemv)(order, trans, m, n, oneD, Adata, lda, Xdata, incX,
                        zeroD, Rdata, 1);
            break;
        case NPY_CFLOAT:
            // Perform complex single precision matrix-vector multiplication using cblas_cgemv
            CBLAS_FUNC(cblas_cgemv)(order, trans, m, n, oneF, Adata, lda, Xdata, incX,
                        zeroF, Rdata, 1);
            break;
    }
}
/*
 * Perform a symmetric rank-k update operation using BLAS routines based on the
 * type of elements in the arrays A and R.
 */
syrk(int typenum, enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
     npy_intp n, npy_intp k,
     PyArrayObject *A, npy_intp lda, PyArrayObject *R)
{
    // 获取数组 A 的数据指针
    const void *Adata = PyArray_DATA(A);
    // 获取数组 R 的数据指针
    void *Rdata = PyArray_DATA(R);
    // 计算 R 的第二维度，如果大于 1 则使用该值，否则设为 1
    npy_intp ldc = PyArray_DIM(R, 1) > 1 ? PyArray_DIM(R, 1) : 1;

    npy_intp i;
    npy_intp j;

    // 根据 typenum 的值选择相应的 BLAS 函数进行操作
    switch (typenum) {
        case NPY_DOUBLE:
            // 双精度情况下的 BLAS 函数调用
            CBLAS_FUNC(cblas_dsyrk)(order, CblasUpper, trans, n, k, 1.,
                        Adata, lda, 0., Rdata, ldc);

            // 对称阵处理，调整非对角元素的值
            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_double*)PyArray_GETPTR2(R, j, i)) =
                            *((npy_double*)PyArray_GETPTR2(R, i, j));
                }
            }
            break;
        case NPY_FLOAT:
            // 单精度情况下的 BLAS 函数调用
            CBLAS_FUNC(cblas_ssyrk)(order, CblasUpper, trans, n, k, 1.f,
                        Adata, lda, 0.f, Rdata, ldc);

            // 对称阵处理，调整非对角元素的值
            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_float*)PyArray_GETPTR2(R, j, i)) =
                            *((npy_float*)PyArray_GETPTR2(R, i, j));
                }
            }
            break;
        case NPY_CDOUBLE:
            // 复双精度情况下的 BLAS 函数调用
            CBLAS_FUNC(cblas_zsyrk)(order, CblasUpper, trans, n, k, oneD,
                        Adata, lda, zeroD, Rdata, ldc);

            // 对称阵处理，调整非对角元素的值
            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_cdouble*)PyArray_GETPTR2(R, j, i)) =
                            *((npy_cdouble*)PyArray_GETPTR2(R, i, j));
                }
            }
            break;
        case NPY_CFLOAT:
            // 复单精度情况下的 BLAS 函数调用
            CBLAS_FUNC(cblas_csyrk)(order, CblasUpper, trans, n, k, oneF,
                        Adata, lda, zeroF, Rdata, ldc);

            // 对称阵处理，调整非对角元素的值
            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_cfloat*)PyArray_GETPTR2(R, j, i)) =
                            *((npy_cfloat*)PyArray_GETPTR2(R, i, j));
                }
            }
            break;
    }
}


/*
 * Determine the shape of the matrix represented by the PyArrayObject.
 * Returns one of _scalar, _column, _row, or _matrix.
 */
static MatrixShape
_select_matrix_shape(PyArrayObject *array)
{
    // 根据数组的维度确定其形状
    switch (PyArray_NDIM(array)) {
        case 0:
            return _scalar;
        case 1:
            // 一维数组，根据第一维大小判断是列向量还是标量
            if (PyArray_DIM(array, 0) > 1)
                return _column;
            return _scalar;
        case 2:
            // 二维数组，根据维度大小判断是列向量、行向量还是矩阵
            if (PyArray_DIM(array, 0) > 1) {
                if (PyArray_DIM(array, 1) == 1)
                    return _column;
                else
                    return _matrix;
            }
            if (PyArray_DIM(array, 1) == 1)
                return _scalar;
            return _row;
    }
    return _matrix;  // 默认返回矩阵形状
}


/*
 * Check if the array strides are misaligned and return 1 if true, 0 otherwise.
 * Ensures that the data segment is aligned with an itemsize address.
 */
NPY_NO_EXPORT int
_bad_strides(PyArrayObject *ap)
{
    int itemsize = PyArray_ITEMSIZE(ap);
    int i, N=PyArray_NDIM(ap);

    // 检查数组的步幅是否对齐，返回值为 1 表示不对齐，为 0 表示对齐
    # 获取数组的步长信息
    npy_intp *strides = PyArray_STRIDES(ap);
    # 获取数组的维度信息
    npy_intp *dims = PyArray_DIMS(ap);
    
    # 检查数组的数据指针是否按照元素大小对齐
    if (((npy_intp)(PyArray_DATA(ap)) % itemsize) != 0) {
        # 如果不是对齐的，返回错误码 1
        return 1;
    }
    
    # 遍历数组的每一个维度
    for (i = 0; i < N; i++) {
        # 检查数组的步长是否为负数或者不是元素大小的整数倍
        if ((strides[i] < 0) || (strides[i] % itemsize) != 0) {
            # 如果步长不合法，返回错误码 1
            return 1;
        }
        # 检查数组步长为 0 但维度大于 1 的情况
        if ((strides[i] == 0 && dims[i] > 1)) {
            # 如果条件不满足，返回错误码 1
            return 1;
        }
    }
    
    # 若所有检查通过，返回正常码 0
    return 0;
/*
 * dot(a,b)
 * 返回浮点类型数组 a 和 b 的点积。
 * 与通用的 numpy 等效函数类似，点积的求和是在 a 的最后一个维度和 b 的倒数第二个维度上进行。
 * 注意：第一个参数不是共轭的。
 *
 * 这个函数是供 PyArray_MatrixProduct2 使用的。假定在进入时，数组 ap1 和 ap2 具有相同的数据类型 typenum，
 * 可以是 float、double、cfloat 或 cdouble，并且维度 <= 2。假定 __array_ufunc__ 的处理已经完成。
 */
NPY_NO_EXPORT PyObject *
cblas_matrixproduct(int typenum, PyArrayObject *ap1, PyArrayObject *ap2,
                    PyArrayObject *out)
{
    PyArrayObject *result = NULL, *out_buf = NULL;
    npy_intp j, lda, ldb;
    npy_intp l;
    int nd;
    npy_intp ap1stride = 0;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp numbytes;
    MatrixShape ap1shape, ap2shape;

    // 检查 ap1 是否有不良步长
    if (_bad_strides(ap1)) {
        // 创建 ap1 的副本，按照任意顺序
        PyObject *op1 = PyArray_NewCopy(ap1, NPY_ANYORDER);

        // 释放原 ap1 对象
        Py_DECREF(ap1);
        // 将 op1 转换为 PyArrayObject 类型并赋值给 ap1
        ap1 = (PyArrayObject *)op1;
        if (ap1 == NULL) {
            goto fail;
        }
    }
    // 检查 ap2 是否有不良步长
    if (_bad_strides(ap2)) {
        // 创建 ap2 的副本，按照任意顺序
        PyObject *op2 = PyArray_NewCopy(ap2, NPY_ANYORDER);

        // 释放原 ap2 对象
        Py_DECREF(ap2);
        // 将 op2 转换为 PyArrayObject 类型并赋值给 ap2
        ap2 = (PyArrayObject *)op2;
        if (ap2 == NULL) {
            goto fail;
        }
    }
    // 选择 ap1 和 ap2 的矩阵形状
    ap1shape = _select_matrix_shape(ap1);
    ap2shape = _select_matrix_shape(ap2);

    // 在此处继续执行矩阵乘法的计算
    // 检查是否有一个操作数是标量（scalar）
    if (ap1shape == _scalar || ap2shape == _scalar) {
        PyArrayObject *oap1, *oap2;
        oap1 = ap1; oap2 = ap2;
        /* One of ap1 or ap2 is a scalar */
        // 如果 ap1 是标量，则将 ap2 视为标量
        if (ap1shape == _scalar) {
            /* Make ap2 the scalar */
            PyArrayObject *t = ap1;
            ap1 = ap2;
            ap2 = t;
            ap1shape = ap2shape;
            ap2shape = _scalar;
        }

        // 如果 ap1 的形状为行向量（_row）
        if (ap1shape == _row) {
            // 获取 ap1 的步长（stride）
            ap1stride = PyArray_STRIDE(ap1, 1);
        }
        // 否则，如果 ap1 的维度大于 0
        else if (PyArray_NDIM(ap1) > 0) {
            // 获取 ap1 的步长（stride）
            ap1stride = PyArray_STRIDE(ap1, 0);
        }

        // 如果 ap1 或 ap2 中有任意一个是标量（维度为 0）
        if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
            npy_intp *thisdims;
            // 如果 ap1 是标量
            if (PyArray_NDIM(ap1) == 0) {
                // 使用 ap2 的维度和数量设置 dimensions 数组和 l
                nd = PyArray_NDIM(ap2);
                thisdims = PyArray_DIMS(ap2);
            }
            // 否则，ap2 是标量
            else {
                // 使用 ap1 的维度和数量设置 dimensions 数组和 l
                nd = PyArray_NDIM(ap1);
                thisdims = PyArray_DIMS(ap1);
            }
            // 计算总长度 l 并设置 dimensions 数组
            l = 1;
            for (j = 0; j < nd; j++) {
                dimensions[j] = thisdims[j];
                l *= dimensions[j];
            }
        }
        // 否则，ap1 和 ap2 都不是标量（维度大于 0）
        else {
            // 获取 ap1 和 ap2 的最后一个维度的长度
            l = PyArray_DIM(oap1, PyArray_NDIM(oap1) - 1);

            // 检查 ap1 和 ap2 是否对齐
            if (PyArray_DIM(oap2, 0) != l) {
                // 如果不对齐，调用错误处理函数并跳转到失败位置
                dot_alignment_error(oap1, PyArray_NDIM(oap1) - 1, oap2, 0);
                goto fail;
            }

            // 计算 nd 作为 ap1 和 ap2 的总维度数减去 2
            nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;

            /*
             * nd = 0 or 1 or 2. If nd == 0 do nothing ...
             */

            // 如果 nd 等于 1
            if (nd == 1) {
                /*
                 * Either PyArray_NDIM(ap1) is 1 dim or PyArray_NDIM(ap2) is
                 * 1 dim and the other is 2 dim
                 */
                // 设置 dimensions[0] 为 ap1 或 ap2 的合适维度长度
                dimensions[0] = (PyArray_NDIM(oap1) == 2) ?
                                PyArray_DIM(oap1, 0) : PyArray_DIM(oap2, 1);
                // 设置 l 为 dimensions[0]
                l = dimensions[0];
                /*
                 * Fix it so that dot(shape=(N,1), shape=(1,))
                 * and dot(shape=(1,), shape=(1,N)) both return
                 * an (N,) array (but use the fast scalar code)
                 */
            }
            // 否则，如果 nd 等于 2
            else if (nd == 2) {
                // 设置 dimensions[0] 和 dimensions[1] 为 ap1 和 ap2 的合适维度长度
                dimensions[0] = PyArray_DIM(oap1, 0);
                dimensions[1] = PyArray_DIM(oap2, 1);
                /*
                 * We need to make sure that dot(shape=(1,1), shape=(1,N))
                 * and dot(shape=(N,1),shape=(1,1)) uses
                 * scalar multiplication appropriately
                 */
                // 如果 ap1 的形状为行向量，则 l 设置为 dimensions[1]，否则为 dimensions[0]
                if (ap1shape == _row) {
                    l = dimensions[1];
                }
                else {
                    l = dimensions[0];
                }
            }

            // 检查求和维度是否为 0 大小
            if (PyArray_DIM(oap1, PyArray_NDIM(oap1) - 1) == 0) {
                // 如果是，则 l 设置为 0
                l = 0;
            }
        }
    }
    else {
        /*
         * (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2)
         * Both ap1 and ap2 are vectors or matrices
         */
        // 获取 ap1 的最后一个维度的大小
        l = PyArray_DIM(ap1, PyArray_NDIM(ap1) - 1);

        // 检查 ap2 的第一个维度是否与 ap1 的最后一个维度大小相等
        if (PyArray_DIM(ap2, 0) != l) {
            // 如果不相等，触发 dot_alignment_error，并跳转到 fail 标签
            dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, 0);
            goto fail;
        }
        // 计算输出数组的维度数
        nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;

        // 根据 nd 的值设置输出数组的维度
        if (nd == 1) {
            dimensions[0] = (PyArray_NDIM(ap1) == 2) ?
                            PyArray_DIM(ap1, 0) : PyArray_DIM(ap2, 1);
        }
        else if (nd == 2) {
            dimensions[0] = PyArray_DIM(ap1, 0);
            dimensions[1] = PyArray_DIM(ap2, 1);
        }
    }

    // 为求和创建新的数组，返回结果给 out_buf
    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }

    // 获取输出数组的字节数
    numbytes = PyArray_NBYTES(out_buf);
    // 将输出数组的数据区域清零
    memset(PyArray_DATA(out_buf), 0, numbytes);
    // 如果 numbytes 为零或者 l 为零，则释放对象并返回结果
    if (numbytes == 0 || l == 0) {
            Py_DECREF(ap1);
            Py_DECREF(ap2);
            Py_DECREF(out_buf);
            return PyArray_Return(result);
    }

    }
    else if ((ap2shape == _column) && (ap1shape != _matrix)) {
        NPY_BEGIN_ALLOW_THREADS;

        /* Dot product between two vectors -- Level 1 BLAS */
        // 使用 Level 1 BLAS 计算两个向量的点积
        PyDataType_GetArrFuncs(PyArray_DESCR(out_buf))->dotfunc(
                 PyArray_DATA(ap1), PyArray_STRIDE(ap1, (ap1shape == _row)),
                 PyArray_DATA(ap2), PyArray_STRIDE(ap2, 0),
                 PyArray_DATA(out_buf), l, NULL);
        NPY_END_ALLOW_THREADS;
    }
    else if (ap1shape == _matrix && ap2shape != _matrix) {
        /* Matrix vector multiplication -- Level 2 BLAS */
        /* lda must be MAX(M,1) */
        enum CBLAS_ORDER Order;
        npy_intp ap2s;

        // 如果 ap1 不是单一段，进行复制
        if (!PyArray_ISONESEGMENT(ap1)) {
            PyObject *new;
            new = PyArray_Copy(ap1);
            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        NPY_BEGIN_ALLOW_THREADS
        // 根据 ap1 是否连续设置 Order 和 lda
        if (PyArray_ISCONTIGUOUS(ap1)) {
            Order = CblasRowMajor;
            lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        }
        else {
            Order = CblasColMajor;
            lda = (PyArray_DIM(ap1, 0) > 1 ? PyArray_DIM(ap1, 0) : 1);
        }
        // 计算 ap2 的步长
        ap2s = PyArray_STRIDE(ap2, 0) / PyArray_ITEMSIZE(ap2);
        // 使用 Level 2 BLAS 计算矩阵向量乘法
        gemv(typenum, Order, CblasNoTrans, ap1, lda, ap2, ap2s, out_buf);
        NPY_END_ALLOW_THREADS;
    }
    # 如果第一个数组是向量而第二个数组是矩阵
    else if (ap1shape != _matrix && ap2shape == _matrix) {
        /* Vector matrix multiplication -- Level 2 BLAS */
        # 定义 CBLAS_ORDER 枚举变量 Order
        enum CBLAS_ORDER Order;
        # 定义 ap1s 为整数指针
        npy_intp ap1s;

        # 如果 ap2 不是连续存储的，则复制 ap2 并使其连续
        if (!PyArray_ISONESEGMENT(ap2)) {
            PyObject *new;
            new = PyArray_Copy(ap2);
            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        # 进入多线程执行区域
        NPY_BEGIN_ALLOW_THREADS
        # 如果 ap2 是连续存储的，则按行主序处理
        if (PyArray_ISCONTIGUOUS(ap2)) {
            Order = CblasRowMajor;
            lda = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);
        }
        # 否则按列主序处理
        else {
            Order = CblasColMajor;
            lda = (PyArray_DIM(ap2, 0) > 1 ? PyArray_DIM(ap2, 0) : 1);
        }
        # 根据 ap1shape 类型确定 ap1s 的值
        if (ap1shape == _row) {
            ap1s = PyArray_STRIDE(ap1, 1) / PyArray_ITEMSIZE(ap1);
        }
        else {
            ap1s = PyArray_STRIDE(ap1, 0) / PyArray_ITEMSIZE(ap1);
        }
        # 执行向量矩阵乘法运算，结果存储在 out_buf 中
        gemv(typenum, Order, CblasTrans, ap2, lda, ap1, ap1s, out_buf);
        # 离开多线程执行区域
        NPY_END_ALLOW_THREADS;
    }
    else {
        /*
         * (PyArray_NDIM(ap1) == 2 && PyArray_NDIM(ap2) == 2)
         * 矩阵乘法 -- Level 3 BLAS
         *  L x M  乘以 M x N
         */
        enum CBLAS_ORDER Order;
        enum CBLAS_TRANSPOSE Trans1, Trans2;
        npy_intp M, N, L;

        /* 优化可能性： */
        /*
         * 如果适用，可以处理单段数组
         * 使用适当的 Order、Trans1 和 Trans2 的值。
         */
        if (!PyArray_IS_C_CONTIGUOUS(ap2) && !PyArray_IS_F_CONTIGUOUS(ap2)) {
            PyObject *new = PyArray_Copy(ap2);

            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        if (!PyArray_IS_C_CONTIGUOUS(ap1) && !PyArray_IS_F_CONTIGUOUS(ap1)) {
            PyObject *new = PyArray_Copy(ap1);

            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }

        NPY_BEGIN_ALLOW_THREADS;

        Order = CblasRowMajor;
        Trans1 = CblasNoTrans;
        Trans2 = CblasNoTrans;
        L = PyArray_DIM(ap1, 0);
        N = PyArray_DIM(ap2, 1);
        M = PyArray_DIM(ap2, 0);
        lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        ldb = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);

        /*
         * 避免对 Fortran 排序的数组进行临时拷贝
         */
        if (PyArray_IS_F_CONTIGUOUS(ap1)) {
            Trans1 = CblasTrans;
            lda = (PyArray_DIM(ap1, 0) > 1 ? PyArray_DIM(ap1, 0) : 1);
        }
        if (PyArray_IS_F_CONTIGUOUS(ap2)) {
            Trans2 = CblasTrans;
            ldb = (PyArray_DIM(ap2, 0) > 1 ? PyArray_DIM(ap2, 0) : 1);
        }

        /*
         * 如果是矩阵乘以其转置的情况，则使用 syrk 函数。
         * 否则，对所有其他情况使用 gemm 函数。
         */
        if (
            (PyArray_BYTES(ap1) == PyArray_BYTES(ap2)) &&
            (PyArray_DIM(ap1, 0) == PyArray_DIM(ap2, 1)) &&
            (PyArray_DIM(ap1, 1) == PyArray_DIM(ap2, 0)) &&
            (PyArray_STRIDE(ap1, 0) == PyArray_STRIDE(ap2, 1)) &&
            (PyArray_STRIDE(ap1, 1) == PyArray_STRIDE(ap2, 0)) &&
            ((Trans1 == CblasTrans) ^ (Trans2 == CblasTrans)) &&
            ((Trans1 == CblasNoTrans) ^ (Trans2 == CblasNoTrans))
        ) {
            if (Trans1 == CblasNoTrans) {
                syrk(typenum, Order, Trans1, N, M, ap1, lda, out_buf);
            }
            else {
                syrk(typenum, Order, Trans1, N, M, ap2, ldb, out_buf);
            }
        }
        else {
            gemm(typenum, Order, Trans1, Trans2, L, N, M, ap1, lda, ap2, ldb,
                 out_buf);
        }
        NPY_END_ALLOW_THREADS;
    }


    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* 触发可能的回写到 `result` */
    PyArray_ResolveWritebackIfCopy(out_buf);
    Py_DECREF(out_buf);
    # 将 result 转换为 Python 对象并返回
    return PyArray_Return(result);
    // 递减并清理 Python 对象 ap1 的引用计数
    Py_XDECREF(ap1);
    // 递减并清理 Python 对象 ap2 的引用计数
    Py_XDECREF(ap2);
    // 递减并清理 Python 对象 out_buf 的引用计数
    Py_XDECREF(out_buf);
    // 递减并清理 Python 对象 result 的引用计数，并返回 NULL 指针表示函数执行失败
    return NULL;
}
```