# `D:\src\scipysrc\scipy\scipy\interpolate\src\_fitpackmodule.c`

```
#include <Python.h>
#include "numpy/arrayobject.h"

#define PyInt_AsLong PyLong_AsLong  // 定义宏 PyInt_AsLong 为 PyLong_AsLong，用于将 Python 的 int 对象转换为 C 的 long 型

static PyObject *fitpack_error;  // 定义一个静态的 PyObject 指针 fitpack_error，用于在 C 代码中表示 FITPACK 的错误状态

#include "__fitpack.h"  // 引入 FITPACK 的头文件 __fitpack.h

#ifdef HAVE_ILP64

#define F_INT npy_int64  // 如果定义了 HAVE_ILP64，将 F_INT 定义为 npy_int64，即 64 位整型
#define F_INT_NPY NPY_INT64  // 将 F_INT_NPY 定义为 NPY_INT64，表示 NumPy 中的 64 位整型
#define F_INT_MAX NPY_MAX_INT64  // 将 F_INT_MAX 定义为 NPY_MAX_INT64，表示 NumPy 中的最大 64 位整数

#if NPY_BITSOF_SHORT == 64
#define F_INT_PYFMT   "h"  // 如果 NumPy 中的 short 整型位数为 64，则将 F_INT_PYFMT 定义为 "h"
#elif NPY_BITSOF_INT == 64
#define F_INT_PYFMT   "i"  // 如果 NumPy 中的 int 整型位数为 64，则将 F_INT_PYFMT 定义为 "i"
#elif NPY_BITSOF_LONG == 64
#define F_INT_PYFMT   "l"  // 如果 NumPy 中的 long 整型位数为 64，则将 F_INT_PYFMT 定义为 "l"
#elif NPY_BITSOF_LONGLONG == 64
#define F_INT_PYFMT   "L"  // 如果 NumPy 中的 long long 整型位数为 64，则将 F_INT_PYFMT 定义为 "L"
#else
#error No compatible 64-bit integer size. \  // 如果没有兼容的 64 位整型，抛出错误
       Please contact NumPy maintainers and give detailed information about your \
       compiler and platform, or set NPY_USE_BLAS64_=0
#endif

#else

#define F_INT int  // 如果未定义 HAVE_ILP64，将 F_INT 定义为 int，即默认为 32 位整型
#define F_INT_NPY NPY_INT  // 将 F_INT_NPY 定义为 NPY_INT，表示 NumPy 中的 int 整型
#define F_INT_MAX NPY_MAX_INT32  // 将 F_INT_MAX 定义为 NPY_MAX_INT32，表示 NumPy 中的最大 32 位整数
#if NPY_BITSOF_SHORT == 32
#define F_INT_PYFMT   "h"  // 如果 NumPy 中的 short 整型位数为 32，则将 F_INT_PYFMT 定义为 "h"
#elif NPY_BITSOF_INT == 32
#define F_INT_PYFMT   "i"  // 如果 NumPy 中的 int 整型位数为 32，则将 F_INT_PYFMT 定义为 "i"
#elif NPY_BITSOF_LONG == 32
#define F_INT_PYFMT   "l"  // 如果 NumPy 中的 long 整型位数为 32，则将 F_INT_PYFMT 定义为 "l"
#else
#error No compatible 32-bit integer size. \  // 如果没有兼容的 32 位整型，抛出错误
       Please contact NumPy maintainers and give detailed information about your \
       compiler and platform
#endif

#endif


/*
 * Multiply mx*my, check for integer overflow.
 * Inputs are Fortran ints, and the output is npy_intp, for use
 * in PyArray_SimpleNew et al.
 * Return -1 on overflow.
 */
npy_intp
_mul_overflow_intp(F_INT mx, F_INT my) {
    npy_intp int_max, mxy;

    /* Limit is min of (largest array size, max of Fortran int) */
    int_max = (F_INT_MAX < NPY_MAX_INTP) ? F_INT_MAX : NPY_MAX_INTP;
    /* v = int_max/my is largest integer multiple of `my` such that
       v * my <= int_max
    */
    if (my != 0 && int_max/my < mx) {
        /* Integer overflow */
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot produce output of size %dx%d (size too large)",
                     mx, my);
        return -1;
    }
    mxy = (npy_intp)mx * (npy_intp)my;  // 计算 mx 和 my 的乘积，转换为 npy_intp 类型
    return mxy;  // 返回计算结果
}


/*
 * Multiply mx*my, check for integer overflow, where both inputs and
 * the output are Fortran ints.
 * Return -1 on overflow.
 */
F_INT
_mul_overflow_f_int(F_INT mx, F_INT my) {
    F_INT mxy;

    /* v = int_max/my is largest integer multiple of `my` such that
       v * my <= F_INT_MAX
    */
    if (my != 0 && F_INT_MAX/my < mx) {
        /* Integer overflow */
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot produce output of size %dx%d (size too large)",
                     mx, my);
        return -1;
    }
    mxy = mx * my;  // 计算 mx 和 my 的乘积
    return mxy;  // 返回计算结果
}


/*
 * Functions moved verbatim from __fitpack.h
 */


/*
 * Python-C wrapper of FITPACK (by P. Dierckx) (in netlib known as dierckx)
 * Author: Pearu Peterson <pearu@ioc.ee>
 * June 1.-4., 1999
 * June 7. 1999
 * $Revision$
 * $Date$
 */

/*  module_methods:
 * {"_parcur", fitpack_parcur, METH_VARARGS, doc_parcur},
 * {"_surfit", fitpack_surfit, METH_VARARGS, doc_surfit},
 * {"_insert", fitpack_insert, METH_VARARGS, doc_insert},
 */

/* link libraries: (one item per line)
   ddierckx
 */
/* python files: (to be imported to Multipack.py)
   fitpack.py
 */
/* 定义符号常量和宏，根据不同的编译条件指定不同的函数名 */

#if defined(UPPERCASE_FORTRAN)
    #if defined(NO_APPEND_FORTRAN)
    /* 无需修改 */
    #else
        /* 定义大写且带下划线的函数名 */
        #define PARCUR PARCUR_
        #define CLOCUR CLOCUR_
        #define SURFIT SURFIT_
        #define INSERT INSERT_
    #endif
#else
    #if defined(NO_APPEND_FORTRAN)
        /* 定义不带下划线的函数名 */
        #define PARCUR parcur
        #define CLOCUR clocur
        #define SURFIT surfit
        #define INSERT insert
    #else
        /* 定义带下划线的函数名 */
        #define PARCUR parcur_
        #define CLOCUR clocur_
        #define SURFIT surfit_
        #define INSERT insert_
    #endif
#endif

/* 声明四个 Fortran 函数 */
void PARCUR(F_INT*,F_INT*,F_INT*,F_INT*,double*,F_INT*,double*,
        double*,double*,double*,F_INT*,double*,F_INT*,F_INT*,
        double*,F_INT*,double*,double*,double*,F_INT*,F_INT*,F_INT*);
void CLOCUR(F_INT*,F_INT*,F_INT*,F_INT*,double*,F_INT*,double*,
        double*,F_INT*,double*,F_INT*,F_INT*,double*,F_INT*,
        double*,double*,double*,F_INT*,F_INT*,F_INT*);
void SURFIT(F_INT*,F_INT*,double*,double*,double*,double*,
        double*,double*,double*,double*,F_INT*,F_INT*,double*,
        F_INT*,F_INT*,F_INT*,double*,F_INT*,double*,F_INT*,double*,
        double*,double*,double*,F_INT*,double*,F_INT*,F_INT*,F_INT*,F_INT*);
void INSERT(F_INT*,double*,F_INT*,double*,F_INT*,double*,double*,
        F_INT*,double*,F_INT*,F_INT*);

/* 注意 curev, cualde 函数不需要接口 */

/* 定义静态文档字符串 */
static char doc_surfit[] = " [tx,ty,c,o] = _surfit(x, y, z, w, xb, xe, yb, ye,"\
      " kx,ky,iopt,s,eps,tx,ty,nxest,nyest,wrk,lwrk1,lwrk2)";
/* Python 封装函数 */
static PyObject *
fitpack_surfit(PyObject *dummy, PyObject *args)
{
    /* 声明一系列变量 */
    F_INT iopt, m, kx, ky, nxest, nyest, lwrk1, lwrk2, *iwrk, kwrk, ier;
    F_INT lwa, nxo, nyo, i, lcest, nmax, nx, ny, lc;
    npy_intp dims[1], lc_intp;
    double *x, *y, *z, *w, xb, xe, yb, ye, s, *tx, *ty, *c, fp;
    double *wrk1, *wrk2, *wa = NULL, eps;
    PyArrayObject *ap_x = NULL, *ap_y = NULL, *ap_z, *ap_w = NULL;
    PyArrayObject *ap_tx = NULL,*ap_ty = NULL,*ap_c = NULL, *ap_wrk = NULL;
    PyObject *x_py = NULL, *y_py = NULL, *z_py = NULL, *w_py = NULL;
    PyObject *tx_py = NULL, *ty_py = NULL, *wrk_py = NULL;

    /* 初始化部分变量 */
    nx = ny = ier = nxo = nyo = 0;
    /* 解析 Python 函数参数 */
    if (!PyArg_ParseTuple(args, ("OOOOdddd" F_INT_PYFMT F_INT_PYFMT F_INT_PYFMT
                                 "ddOO" F_INT_PYFMT F_INT_PYFMT "O"
                                 F_INT_PYFMT F_INT_PYFMT),
                &x_py, &y_py, &z_py, &w_py, &xb, &xe, &yb, &ye,
                &kx, &ky, &iopt, &s, &eps, &tx_py, &ty_py, &nxest,
                &nyest, &wrk_py, &lwrk1, &lwrk2)) {
        return NULL;
    }
    /* 将 Python 数组转换为连续的 PyArrayObject */
    ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x_py, NPY_DOUBLE, 0, 1);
    ap_y = (PyArrayObject *)PyArray_ContiguousFromObject(y_py, NPY_DOUBLE, 0, 1);
    ap_z = (PyArrayObject *)PyArray_ContiguousFromObject(z_py, NPY_DOUBLE, 0, 1);
    ap_w = (PyArrayObject *)PyArray_ContiguousFromObject(w_py, NPY_DOUBLE, 0, 1);
    ap_wrk=(PyArrayObject *)PyArray_ContiguousFromObject(wrk_py, NPY_DOUBLE, 0, 1);
    /* 将 Python 对象 wrk_py 转换为一个连续的 NumPy 数组对象，数据类型为 NPY_DOUBLE */
    /*ap_iwrk=(PyArrayObject *)PyArray_ContiguousFromObject(iwrk_py, F_INT_NPY, 0, 1);*/
    /* 以下是尝试将 Python 对象 iwrk_py 转换为一个连续的 NumPy 数组对象，数据类型为 F_INT_NPY，已被注释掉 */
    if (ap_x == NULL
            || ap_y == NULL
            || ap_z == NULL
            || ap_w == NULL
            || ap_wrk == NULL) {
        /* 检查各个 NumPy 数组对象是否成功创建，如果有任何一个为 NULL，则跳转到 fail 标签 */
        goto fail;
    }
    x = (double *) PyArray_DATA(ap_x);
    /* 获取 ap_x 的数据指针，并将其转换为 double 类型的指针 x */
    y = (double *) PyArray_DATA(ap_y);
    /* 获取 ap_y 的数据指针，并将其转换为 double 类型的指针 y */
    z = (double *) PyArray_DATA(ap_z);
    /* 获取 ap_z 的数据指针，并将其转换为 double 类型的指针 z */
    w = (double *) PyArray_DATA(ap_w);
    /* 获取 ap_w 的数据指针，并将其转换为 double 类型的指针 w */
    m = PyArray_DIMS(ap_x)[0];
    /* 获取 ap_x 数组的第一个维度的大小，并赋给 m */
    nmax = nxest;
    /* 将 nxest 的值赋给 nmax */
    if (nmax < nyest) {
        /* 如果 nmax 小于 nyest，则将 nyest 的值赋给 nmax */
        nmax = nyest;
    }
    /* lcest = (nxest - kx - 1)*(nyest - ky - 1); */
    /* 计算 lcest 的值，代表一个数学表达式，但在实际代码中被注释掉 */
    lcest = _mul_overflow_f_int(nxest - kx - 1, nyest - ky - 1);
    /* 使用 _mul_overflow_f_int 函数计算上述数学表达式，防止溢出 */
    if (lcest < 0) {
        /* 如果 lcest 小于 0，则抛出内存错误并跳转到 fail 标签 */
        PyErr_NoMemory();
        goto fail;
    }
    /* kwrk computation is unlikely to overflow if lcest above did not.*/
    /* 计算 kwrk 的值，如果上面的 lcest 没有溢出，那么 kwrk 的计算不太可能溢出 */
    kwrk = m + (nxest - 2*kx - 1)*(nyest - 2*ky - 1);
    /* 根据数学表达式计算 kwrk 的值 */
    lwa = 2*nmax + lcest + lwrk1 + lwrk2 + kwrk;
    /* 计算 lwa 的值，包括多个变量的和 */
    if ((wa = malloc(lwa*sizeof(double))) == NULL) {
        /* 分配大小为 lwa*sizeof(double) 的内存给 wa，如果分配失败，则抛出内存错误并跳转到 fail 标签 */
        PyErr_NoMemory();
        goto fail;
    }
    /*
     * NOTE: the work arrays MUST be aligned on double boundaries, as Fortran
     *       compilers (e.g. gfortran) may assume that.  Malloc gives suitable
     *       alignment, so we are here careful not to mess it up.
     */
    /* 注意：工作数组必须在 double 边界上对齐，因为 Fortran 编译器（如 gfortran）可能会假设如此。使用 malloc 可以提供适当的对齐，因此我们在这里小心地不要搞砸。 */
    tx = wa;
    /* 将 wa 的地址赋给 tx */
    ty = tx + nmax;
    /* 将 tx 向后移动 nmax 个 double 类型的元素，将结果赋给 ty */
    c = ty + nmax;
    /* 将 ty 向后移动 nmax 个 double 类型的元素，将结果赋给 c */
    wrk1 = c + lcest;
    /* 将 c 向后移动 lcest 个 double 类型的元素，将结果赋给 wrk1 */
    iwrk = (F_INT *)(wrk1 + lwrk1);
    /* 将 wrk1 向后移动 lwrk1 个 double 类型的元素，然后将结果强制转换为 F_INT 指针类型赋给 iwrk */
    wrk2 = ((double *)iwrk) + kwrk;
    /* 将 iwrk 转换为 double 类型指针后向后移动 kwrk 个 double 类型的元素，将结果赋给 wrk2 */
    if (iopt) {
        /* 如果 iopt 不为 0 */
        ap_tx = (PyArrayObject *)PyArray_ContiguousFromObject(tx_py, NPY_DOUBLE, 0, 1);
        /* 将 Python 对象 tx_py 转换为一个连续的 NumPy 数组对象，数据类型为 NPY_DOUBLE */
        ap_ty = (PyArrayObject *)PyArray_ContiguousFromObject(ty_py, NPY_DOUBLE, 0, 1);
        /* 将 Python 对象 ty_py 转换为一个连续的 NumPy 数组对象，数据类型为 NPY_DOUBLE */
        if (ap_tx == NULL || ap_ty == NULL) {
            /* 如果 ap_tx 或 ap_ty 为 NULL，则跳转到 fail 标签 */
            goto fail;
        }
        nx = nxo = PyArray_DIMS(ap_tx)[0];
        /* 获取 ap_tx 数组的第一个维度大小，并赋给 nx 和 nxo */
        ny = nyo = PyArray_DIMS(ap_ty)[0];
        /* 获取 ap_ty 数组的第一个维度大小，并赋给 ny 和 nyo */
        memcpy(tx, PyArray_DATA(ap_tx), nx*sizeof(double));
        /* 将 ap_tx 数组的数据拷贝到 tx 指向的内存位置，拷贝长度为 nx*sizeof(double) 字节 */
        memcpy(ty, PyArray_DATA(ap_ty), ny*sizeof(double));
        /* 将 ap_ty 数组的数据拷贝到 ty 指向的内存位置，拷贝长度为 ny*sizeof(double) 字节 */
    }
    if (iopt==1) {
        /* 如果 iopt 等于 1 */
        /* lc = (nx - kx - 1)*(ny - ky - 1); */
        /* 计算 lc 的值，代表一个数学表达式，但在实际代码中被注释掉 */
        lc = _mul_overflow_f_int(nx - kx - 1, ny - ky -1);
        /* 使用 _mul_overflow_f_int 函数计算上述数学表达式，防止溢出 */
        if (lc < 0) {
            /* 如果 lc 小于 0，则跳转到 fail 标签 */
            goto fail;
        }

        memcpy(wrk1, PyArray_DATA(ap_wrk), lc*sizeof(double));
        /* 将 ap_wrk 数组的数据拷贝到 wrk1 指向的内存位置，拷贝长度为 lc*sizeof(double) 字节 */
        /*memcpy(iwrk,PyArray_DATA(ap_iwrk),n*sizeof(int));*/
        /* 将 ap_iwrk 数组的数据拷贝到 iwrk 指向的内存位置，拷贝长度为 n*sizeof(int) 字节，但在实际代码中被注释掉 */
    }
    SURFIT(&iopt, &m, x, y, z, w, &xb, &xe, &yb, &ye, &kx, &ky,
            &s, &nxest, &nyest, &nmax, &eps, &nx, tx, &ny, ty,
            c, &fp, wrk1, &lwrk1, wrk2, &lwrk2, iwrk, &kwrk, &ier);
    /* 调用 SURFIT 函数，传递多个参数指针 */
    i = 0;
    while ((ier > 10) && (i++ < 5)) {
    lc_intp = _mul_overflow_intp(nx - kx - 1, ny - ky -1);
    # 计算 lc_intp，表示数组 ap_c 的长度，检查是否会溢出

    if (lc_intp < 0) {
        goto fail;
    }
    # 如果 lc_intp 小于 0，则跳转到 fail 标签，表示失败

    Py_XDECREF(ap_tx);
    Py_XDECREF(ap_ty);
    # 释放之前可能存在的 ap_tx 和 ap_ty 对象的引用

    dims[0] = nx;
    ap_tx = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    dims[0] = ny;
    ap_ty = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    # 创建新的一维数组对象 ap_tx 和 ap_ty，分别存储长度为 nx 和 ny 的 NPY_DOUBLE 类型的数组

    dims[0] = lc_intp;
    ap_c = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    # 创建新的一维数组对象 ap_c，其长度为 lc_intp，类型为 NPY_DOUBLE

    if (ap_tx == NULL
            || ap_ty == NULL
            || ap_c == NULL) {
        goto fail;
    }
    # 检查 ap_tx、ap_ty、ap_c 是否成功创建，如果有任何一个为 NULL，则跳转到 fail 标签，表示失败

    if ((iopt == 0)||(nx > nxo)||(ny > nyo)) {
        Py_XDECREF(ap_wrk);
        dims[0] = lc_intp;
        ap_wrk = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (ap_wrk == NULL) {
            goto fail;
        }
        /*ap_iwrk = (PyArrayObject *)PyArray_SimpleNew(1,&n,F_INT_NPY);*/
    }
    # 如果 iopt 为 0，或者 nx 大于 nxo，或者 ny 大于 nyo，则重新创建 ap_wrk 数组对象，其长度为 lc_intp

    if (PyArray_DIMS(ap_wrk)[0] < lc_intp) {
        Py_XDECREF(ap_wrk);
        dims[0] = lc_intp;
        ap_wrk = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (ap_wrk == NULL) {
            goto fail;
        }
    }
    # 检查 ap_wrk 的长度是否小于 lc_intp，如果是，则重新创建 ap_wrk 数组对象，长度为 lc_intp

    memcpy(PyArray_DATA(ap_tx), tx, nx*sizeof(double));
    memcpy(PyArray_DATA(ap_ty), ty, ny*sizeof(double));
    memcpy(PyArray_DATA(ap_c), c, lc_intp*sizeof(double));
    memcpy(PyArray_DATA(ap_wrk), wrk1, lc_intp*sizeof(double));
    # 将 tx 数组的前 nx 个 double 类型数据复制到 ap_tx 中
    # 将 ty 数组的前 ny 个 double 类型数据复制到 ap_ty 中
    # 将 c 数组的前 lc_intp 个 double 类型数据复制到 ap_c 中
    # 将 wrk1 数组的前 lc_intp 个 double 类型数据复制到 ap_wrk 中

    /*memcpy(PyArray_DATA(ap_iwrk),iwrk,n*sizeof(int));*/
    # 未解除注释的代码，可能表示将 iwrk 数组的前 n 个 int 类型数据复制到 ap_iwrk 中

    free(wa);
    # 释放之前分配的 wa 内存空间

    Py_DECREF(ap_x);
    Py_DECREF(ap_y);
    Py_DECREF(ap_z);
    Py_DECREF(ap_w);
    # 逐个减少 ap_x、ap_y、ap_z、ap_w 的引用计数，释放相应的内存空间

    return Py_BuildValue("NNN{s:N,s:i,s:d}",PyArray_Return(ap_tx),
            PyArray_Return(ap_ty),PyArray_Return(ap_c),
            "wrk",PyArray_Return(ap_wrk),
            "ier",ier,"fp",fp);
    # 构建一个 Python 对象，返回包含 ap_tx、ap_ty、ap_c 三个数组对象的元组
    # 以及一个包含 ap_wrk、ier 和 fp 的字典，以指定的格式返回 Python 对象
fail:
    // 释放动态分配的内存
    free(wa);
    // 减少对象的引用计数，避免内存泄漏
    Py_XDECREF(ap_x);
    Py_XDECREF(ap_y);
    Py_XDECREF(ap_z);
    Py_XDECREF(ap_w);
    Py_XDECREF(ap_tx);
    Py_XDECREF(ap_ty);
    Py_XDECREF(ap_wrk);
    /*Py_XDECREF(ap_iwrk);*/
    // 如果没有已设置的异常，设置一个值错误的异常
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError,
                "An error occurred.");
    }
    // 返回空指针，表示函数调用失败
    return NULL;
}


static char doc_parcur[] = " [t,c,o] = _parcur(x,w,u,ub,ue,k,iopt,ipar,s,t,nest,wrk,iwrk,per)";
static PyObject *
fitpack_parcur(PyObject *dummy, PyObject *args)
{
    F_INT k, iopt, ipar, nest, *iwrk, idim, m, mx, no=0, nc, ier, lwa, lwrk, i, per;
    F_INT n=0,  lc;
    npy_intp dims[1];
    double *x, *w, *u, *c, *t, *wrk, *wa=NULL, ub, ue, fp, s;
    PyObject *x_py = NULL, *u_py = NULL, *w_py = NULL, *t_py = NULL;
    PyObject *wrk_py=NULL, *iwrk_py=NULL;
    PyArrayObject *ap_x = NULL, *ap_u = NULL, *ap_w = NULL, *ap_t = NULL, *ap_c = NULL;
    PyArrayObject *ap_wrk = NULL, *ap_iwrk = NULL;

    // 解析 Python 函数参数并检查是否成功
    if (!PyArg_ParseTuple(args, ("OOOdd" F_INT_PYFMT F_INT_PYFMT F_INT_PYFMT
                                 "dO" F_INT_PYFMT "OO" F_INT_PYFMT),
                          &x_py, &w_py, &u_py, &ub, &ue, &k, &iopt, &ipar,
                          &s, &t_py, &nest, &wrk_py, &iwrk_py, &per)) {
        return NULL;
    }
    // 将 Python 对象转换为连续的 NumPy 数组对象
    ap_x = (PyArrayObject *)PyArray_ContiguousFromObject(x_py, NPY_DOUBLE, 0, 1);
    ap_u = (PyArrayObject *)PyArray_ContiguousFromObject(u_py, NPY_DOUBLE, 0, 1);
    ap_w = (PyArrayObject *)PyArray_ContiguousFromObject(w_py, NPY_DOUBLE, 0, 1);
    ap_wrk=(PyArrayObject *)PyArray_ContiguousFromObject(wrk_py, NPY_DOUBLE, 0, 1);
    ap_iwrk=(PyArrayObject *)PyArray_ContiguousFromObject(iwrk_py, F_INT_NPY, 0, 1);
    // 检查数组对象是否成功创建
    if (ap_x == NULL
            || ap_u == NULL
            || ap_w == NULL
            || ap_wrk == NULL
            || ap_iwrk == NULL) {
        // 转到错误处理标签
        goto fail;
    }
    // 获取数组对象中的数据指针
    x = (double *) PyArray_DATA(ap_x);
    u = (double *) PyArray_DATA(ap_u);
    w = (double *) PyArray_DATA(ap_w);
    // 获取数组对象的维度信息
    m = PyArray_DIMS(ap_w)[0];
    mx = PyArray_DIMS(ap_x)[0];
    idim = mx/m;
    // 根据选项计算工作区的长度
    if (per) {
        lwrk = m*(k + 1) + nest*(7 + idim + 5*k);
    }
    else {
        lwrk = m*(k + 1) + nest*(6 + idim + 3*k);
    }
    nc = idim*nest;
    lwa = nc + 2*nest + lwrk;
    // 分配内存以存储工作区数组
    if ((wa = malloc(lwa*sizeof(double))) == NULL) {
        PyErr_NoMemory();
        // 转到错误处理标签
        goto fail;
    }
    // 设置工作数组的指针
    t = wa;
    c = t + nest;
    wrk = c + nc;
    iwrk = (F_INT *)(wrk + lwrk);
    // 如果选项为真，从 NumPy 数组对象中复制数据到工作数组
    if (iopt) {
        ap_t=(PyArrayObject *)PyArray_ContiguousFromObject(t_py, NPY_DOUBLE, 0, 1);
        if (ap_t == NULL) {
            // 转到错误处理标签
            goto fail;
        }
        n = no = PyArray_DIMS(ap_t)[0];
        memcpy(t, PyArray_DATA(ap_t), n*sizeof(double));
        Py_DECREF(ap_t);
        ap_t = NULL;
    }
    // 如果选项为 1，从 NumPy 数组对象中复制数据到工作数组和工作整型数组
    if (iopt == 1) {
        memcpy(wrk, PyArray_DATA(ap_wrk), n*sizeof(double));
        memcpy(iwrk, PyArray_DATA(ap_iwrk), n*sizeof(F_INT));
    }
    // 如果 per 为真，则调用 CLOCUR 函数进行曲线拟合
    if (per) {
        CLOCUR(&iopt, &ipar, &idim, &m, u, &mx, x, w, &k, &s, &nest,
                &n, t, &nc, c, &fp, wrk, &lwrk, iwrk, &ier);
    }
    // 如果 per 不为真，则调用 PARCUR 函数进行参数曲线拟合
    else {
        PARCUR(&iopt, &ipar, &idim, &m, u, &mx, x, w, &ub, &ue, &k,
                &s, &nest, &n, t, &nc, c, &fp, wrk, &lwrk, iwrk, &ier);
    }
    // 如果返回错误码为 10，设置一个 ValueError 异常，并跳转到 fail 标签处
    if (ier == 10) {
        PyErr_SetString(PyExc_ValueError, "Invalid inputs.");
        goto fail;
    }
    // 如果返回的错误码大于 0 且 n 等于 0，则将 n 设置为 1
    if (ier > 0 && n == 0) {
        n = 1;
    }
    // 计算 lc 的值
    lc = (n - k - 1) * idim;
    // 创建一个维度为 n 的 PyArrayObject 对象 ap_t，并分配 NPY_DOUBLE 类型的内存
    dims[0] = n;
    ap_t = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    // 创建一个维度为 lc 的 PyArrayObject 对象 ap_c，并分配 NPY_DOUBLE 类型的内存
    dims[0] = lc;
    ap_c = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    // 如果 ap_t 或 ap_c 为空，则跳转到 fail 标签处
    if (ap_t == NULL || ap_c == NULL) {
        goto fail;
    }
    // 如果 iopt 不等于 1 或者 n 大于 no，则释放 ap_wrk 和 ap_iwrk，并将它们置为 NULL
    if (iopt != 1 || n > no) {
        Py_XDECREF(ap_wrk);
        ap_wrk = NULL;
        Py_XDECREF(ap_iwrk);
        ap_iwrk = NULL;

        // 创建一个维度为 n 的 PyArrayObject 对象 ap_wrk，并分配 NPY_DOUBLE 类型的内存
        dims[0] = n;
        ap_wrk = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        // 如果 ap_wrk 为空，则跳转到 fail 标签处
        if (ap_wrk == NULL) {
            goto fail;
        }
        // 创建一个维度为 n 的 PyArrayObject 对象 ap_iwrk，并分配 F_INT_NPY 类型的内存
        ap_iwrk = (PyArrayObject *)PyArray_SimpleNew(1, dims, F_INT_NPY);
        // 如果 ap_iwrk 为空，则跳转到 fail 标签处
        if (ap_iwrk == NULL) {
            goto fail;
        }
    }
    // 将数组 t 的数据复制到 PyArrayObject 对象 ap_t 中
    memcpy(PyArray_DATA(ap_t), t, n * sizeof(double));
    // 将数组 c 的部分数据复制到 PyArrayObject 对象 ap_c 中
    for (i = 0; i < idim; i++)
        memcpy((double *)PyArray_DATA(ap_c) + i * (n - k - 1), c + i * n, (n - k - 1) * sizeof(double));
    // 将数组 wrk 的数据复制到 PyArrayObject 对象 ap_wrk 中
    memcpy(PyArray_DATA(ap_wrk), wrk, n * sizeof(double));
    // 将数组 iwrk 的数据复制到 PyArrayObject 对象 ap_iwrk 中
    memcpy(PyArray_DATA(ap_iwrk), iwrk, n * sizeof(F_INT));
    // 释放数组 wa 指向的内存空间
    free(wa);
    // 释放 ap_x 和 ap_w 所指向的 PyArrayObject 对象的引用
    Py_DECREF(ap_x);
    Py_DECREF(ap_w);
    // 返回一个 PyTupleObject 对象，包含多个元素，每个元素是一个 PyArrayObject 对象
    return Py_BuildValue(("NN{s:N,s:d,s:d,s:N,s:N,s:" F_INT_PYFMT ",s:d}"), PyArray_Return(ap_t),
            PyArray_Return(ap_c), "u", PyArray_Return(ap_u), "ub", ub, "ue", ue,
            "wrk", PyArray_Return(ap_wrk), "iwrk", PyArray_Return(ap_iwrk),
            "ier", ier, "fp", fp);
fail:
    // 释放动态分配的内存
    free(wa);
    // 释放 Python 对象的引用计数
    Py_XDECREF(ap_x);
    Py_XDECREF(ap_u);
    Py_XDECREF(ap_w);
    Py_XDECREF(ap_t);
    Py_XDECREF(ap_wrk);
    Py_XDECREF(ap_iwrk);
    // 返回空指针，表示失败
    return NULL;
}

static char doc_insert[] = " [tt,cc,ier] = _insert(iopt,t,c,k,x,m)";
static PyObject *
fitpack_insert(PyObject *dummy, PyObject* args)
{
    F_INT iopt, n, nn, k, ier, m, nest;
    npy_intp dims[1];
    double x;
    double *t_in, *c_in, *t_out, *c_out, *t_buf = NULL, *c_buf = NULL, *p;
    double *t1, *t2, *c1, *c2;
    PyArrayObject *ap_t_in = NULL, *ap_c_in = NULL, *ap_t_out = NULL, *ap_c_out = NULL;
    PyObject *t_py = NULL, *c_py = NULL;
    PyObject *ret = NULL;

    // 解析 Python 函数参数
    if (!PyArg_ParseTuple(args, (F_INT_PYFMT "OO" F_INT_PYFMT "d" F_INT_PYFMT),
                          &iopt, &t_py, &c_py, &k, &x, &m)) {
        return NULL;
    }
    // 将 Python 对象转换为连续的 NumPy 数组
    ap_t_in = (PyArrayObject *)PyArray_ContiguousFromObject(t_py, NPY_DOUBLE, 0, 1);
    ap_c_in = (PyArrayObject *)PyArray_ContiguousFromObject(c_py, NPY_DOUBLE, 0, 1);
    // 检查转换是否成功
    if (ap_t_in == NULL || ap_c_in == NULL) {
        goto fail;
    }
    // 获取 NumPy 数组的数据指针和维度信息
    t_in = (double *)PyArray_DATA(ap_t_in);
    c_in = (double *)PyArray_DATA(ap_c_in);
    n = PyArray_DIMS(ap_t_in)[0];
    nest = n + m;
    dims[0] = nest;
    // 创建输出的 NumPy 数组对象
    ap_t_out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    ap_c_out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    // 检查创建输出数组是否成功
    if (ap_t_out == NULL || ap_c_out == NULL) {
        goto fail;
    }
    // 获取输出数组的数据指针
    t_out = (double *)PyArray_DATA(ap_t_out);
    c_out = (double *)PyArray_DATA(ap_c_out);

    /*
     * 调用 INSERT 程序插入 m 次节点，即：
     *
     *     for _ in range(n, nest):
     *         t, c = INSERT(t, c)
     *     return t, c
     *
     * 需要确保传递给 INSERT 程序的输入和输出缓冲区不指向同一内存，
     * 这在 Fortran 中是不允许的。因此我们使用临时存储空间，并在
     * 输入和输出缓冲区之间进行切换。
     */
    t2 = t_in;
    c2 = c_in;
    t1 = t_out;
    c1 = c_out;

    for (; n < nest; n++) {
        /* 交换缓冲区 */
        p = t2; t2 = t1; t1 = p;
        p = c2; c2 = c1; c1 = p;

        /* 分配临时缓冲区（当 m > 1 时需要） */
        if (t2 == t_in) {
            if (t_buf == NULL) {
                t_buf = calloc(nest, sizeof(double));
                c_buf = calloc(nest, sizeof(double));
                if (t_buf == NULL || c_buf == NULL) {
                    PyErr_NoMemory();
                    goto fail;
                }
            }
            t2 = t_buf;
            c2 = c_buf;
        }

        // 调用 INSERT 函数
        INSERT(&iopt, t1, &n, c1, &k, &x, t2, &nn, c2, &nest, &ier);

        // 检查插入过程中是否出错
        if (ier) {
            break;
        }
    }

    /* 确保输出最终存储在输出缓冲区中 */
    if (t2 != t_out) {
        memcpy(t_out, t2, nest * sizeof(double));
        memcpy(c_out, c2, nest * sizeof(double));
    }

    // 释放 Python 对象的引用
    Py_DECREF(ap_c_in);
    Py_DECREF(ap_t_in);
    // 释放临时缓冲区
    free(t_buf);
    free(c_buf);
    # 使用 Py_BuildValue 函数创建一个 Python 对象，包含三个部分：两个 PyArray 对象和一个整数 ier
    ret = Py_BuildValue(("NN" F_INT_PYFMT), PyArray_Return(ap_t_out), PyArray_Return(ap_c_out), ier)
    # 返回由 Py_BuildValue 创建的 Python 对象 ret
    return ret
fail:
    // 释放 Python 对象的引用计数，防止内存泄漏
    Py_XDECREF(ap_c_out);
    Py_XDECREF(ap_t_out);
    Py_XDECREF(ap_c_in);
    Py_XDECREF(ap_t_in);
    // 释放临时缓冲区的内存
    free(t_buf);
    free(c_buf);
    // 返回空指针表示函数执行失败
    return NULL;
}

static struct PyMethodDef fitpack_module_methods[] = {
{"_parcur",
    fitpack_parcur,
    METH_VARARGS, doc_parcur},
{"_surfit",
    fitpack_surfit,
    METH_VARARGS, doc_surfit},
{"_insert",
    fitpack_insert,
    METH_VARARGS, doc_insert},
{NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    // 定义 Python 模块的头部信息
    PyModuleDef_HEAD_INIT,
    // 模块名称为 "_fitpack"
    "_fitpack",
    // 模块的文档字符串为空
    NULL,
    // 指定模块状态为不可子模块化
    -1,
    // 指定模块包含的方法集合
    fitpack_module_methods,
    // 模块的全局状态对象，这里为空
    NULL,
    // 在模块创建时的槽函数，这里为空
    NULL,
    // 在模块销毁时的槽函数，这里为空
    NULL,
    // 在模块执行某些特定操作时的槽函数，这里为空
    NULL
};

PyMODINIT_FUNC
// 初始化 "_fitpack" 模块的入口函数
PyInit__fitpack(void)
{
    PyObject *module, *mdict;

    // 导入 NumPy C API
    import_array();

    // 创建 Python 模块对象
    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        // 如果创建失败，则返回空指针
        return NULL;
    }

    // 获取模块的字典对象
    mdict = PyModule_GetDict(module);

    // 创建自定义的异常对象 "_fitpack.error"
    fitpack_error = PyErr_NewException ("_fitpack.error", NULL, NULL);
    if (fitpack_error == NULL) {
        // 如果创建异常对象失败，则返回空指针
        return NULL;
    }
    // 将异常对象添加到模块的字典中
    if (PyDict_SetItemString(mdict, "error", fitpack_error)) {
        // 如果添加失败，则返回空指针
        return NULL;
    }

    // 返回创建的 Python 模块对象
    return module;
}
```