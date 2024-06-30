# `D:\src\scipysrc\scipy\scipy\integrate\_odepackmodule.c`

```
/*
这个函数是调用用户定义的 Python 函数的 C 包装器，这些函数定义了微分方程（以及雅可比函数）。
Fortran 代码调用 ode_function() 和 ode_jacobian_function()，而这些函数又调用这个函数来调用用户通过 odeint 提供的 Python 函数。

如果发生错误，返回 NULL；否则返回一个 NumPy 数组。
*/

PyArrayObject *sequence = NULL;
PyObject *tfloat = NULL;
PyObject *firstargs = NULL;
PyObject *arglist = NULL;
PyObject *result = NULL;
PyArrayObject *result_array = NULL;

/* 从输入构建序列参数 */
sequence = (PyArrayObject *) PyArray_SimpleNewFromData(1, &n, NPY_DOUBLE,
                                                       (char *) x);
if (sequence == NULL) {
    // 如果构建序列参数失败，跳转到失败标签
    goto fail;
}

tfloat = PyFloat_FromDouble(t);
    // 如果 tfloat 为空指针，跳转到失败处理标签
    if (tfloat == NULL) {
        goto fail;
    }

    // 创建一个包含两个元素的元组 firstargs
    /* firstargs 是一个元组，用来保存前两个参数。 */
    firstargs = PyTuple_New(2);
    if (firstargs == NULL) {
        goto fail;
    }

    // 如果 tfirst 等于 0，设置元组的第一个元素为 sequence，第二个元素为 tfloat；否则相反
    if (tfirst == 0) {
        PyTuple_SET_ITEM(firstargs, 0, (PyObject *) sequence);
        PyTuple_SET_ITEM(firstargs, 1, tfloat);
    } else {
        PyTuple_SET_ITEM(firstargs, 0, tfloat);
        PyTuple_SET_ITEM(firstargs, 1, (PyObject *) sequence);
    }
    /* firstargs 现在拥有 sequence 和 tfloat 的引用。 */
    sequence = NULL;
    tfloat = NULL;

    // 将 firstargs 和 args 拼接成一个新的序列 arglist
    arglist = PySequence_Concat(firstargs, args);
    if (arglist == NULL) {
        goto fail;
    }

    // 调用 Python 函数
    /* 调用 Python 函数。 */
    result = PyObject_CallObject(func, arglist);
    if (result == NULL) {
        goto fail;
    }

    // 将 result 转换为 PyArrayObject 类型的 result_array，数据类型为 NPY_DOUBLE，不使用任何标志
    result_array = (PyArrayObject *)
                   PyArray_ContiguousFromObject(result, NPY_DOUBLE, 0, 0);
fail:
    // 释放 Python 对象 sequence 的引用计数
    Py_XDECREF(sequence);
    // 释放 Python 对象 tfloat 的引用计数
    Py_XDECREF(tfloat);
    // 释放 Python 对象 firstargs 的引用计数
    Py_XDECREF(firstargs);
    // 释放 Python 对象 arglist 的引用计数
    Py_XDECREF(arglist);
    // 释放 Python 对象 result 的引用计数
    Py_XDECREF(result);
    // 返回 result_array 对象的 PyObject 指针
    return (PyObject *) result_array;
}


static PyObject *odepack_error;

#if defined(UPPERCASE_FORTRAN)
    #if defined(NO_APPEND_FORTRAN)
        /* 如果定义了 UPPERCASE_FORTRAN 和 NO_APPEND_FORTRAN 则不做任何操作 */
    #else
        // 将 LSODA 定义为 LSODA_
        #define LSODA  LSODA_
    #endif
#else
    #if defined(NO_APPEND_FORTRAN)
        // 将 LSODA 定义为 lsoda
        #define LSODA  lsoda
    #else
        // 将 LSODA 定义为 lsoda_
        #define LSODA  lsoda_
    #endif
#endif

typedef void lsoda_f_t(F_INT *n, double *t, double *y, double *ydot);
typedef int lsoda_jac_t(F_INT *n, double *t, double *y, F_INT *ml, F_INT *mu,
                        double *pd, F_INT *nrowpd);

void LSODA(lsoda_f_t *f, F_INT *neq, double *y, double *t, double *tout, F_INT *itol,
           double *rtol, double *atol, F_INT *itask, F_INT *istate, F_INT *iopt,
           double *rwork, F_INT *lrw, F_INT *iwork, F_INT *liw, lsoda_jac_t *jac,
           F_INT *jt);

/*
void ode_function(int *n, double *t, double *y, double *ydot)
{
  ydot[0] = -0.04*y[0] + 1e4*y[1]*y[2];
  ydot[2] = 3e7*y[1]*y[1];
  ydot[1] = -ydot[0] - ydot[2];
  return;
}
*/

void
ode_function(F_INT *n, double *t, double *y, double *ydot)
{
    /*
    这是从 Fortran 代码调用的函数。它应该
        -- 使用 call_odeint_user_function 获取一个 multiarrayobject 结果
        -- 检查错误并在有任何错误时将 *n 设置为 -1
        -- 否则将计算结果放入 ydot 中
    */

    PyArrayObject *result_array = NULL;

    // 调用 call_odeint_user_function 获取 Python 数组对象 result_array
    result_array = (PyArrayObject *)
                   call_odeint_user_function(global_params.python_function,
                                             *n, y, *t, global_params.tfirst,
                                             global_params.extra_arguments,
                                             odepack_error);
    // 如果未能获取到 result_array，则设置 *n 为 -1 并返回
    if (result_array == NULL) {
        *n = -1;
        return;
    }

    // 如果返回的数组 result_array 的维度大于 1，则设置 *n 为 -1 并抛出异常
    if (PyArray_NDIM(result_array) > 1) {
        *n = -1;
        PyErr_Format(PyExc_RuntimeError,
                "The array return by func must be one-dimensional, but got ndim=%d.",
                PyArray_NDIM(result_array));
        Py_DECREF(result_array);
        return;
    }

    // 如果返回的数组 result_array 的大小与 *n 不匹配，则设置 *n 为 -1 并抛出异常
    if (PyArray_Size((PyObject *)result_array) != *n) {
        PyErr_Format(PyExc_RuntimeError,
            "The size of the array returned by func (%ld) does not match "
            "the size of y0 (%d).",
            PyArray_Size((PyObject *)result_array), *n);
        *n = -1;
        Py_DECREF(result_array);
        return;
    }

    // 将 result_array 中的数据复制到 ydot 中，并释放 result_array
    memcpy(ydot, PyArray_DATA(result_array), (*n)*sizeof(double));
    Py_DECREF(result_array);
    return;
}
/*
 *  将一个连续存储的矩阵 `c` 复制到一个 Fortran 排序的矩阵 `f` 中。
 *  `ldf` 是 `f` 中 Fortran 数组的主维度。
 *  `nrows` 和 `ncols` 分别是矩阵的行数和列数。
 *  如果 `transposed` 是 0，则 c[i, j] 对应 *(c + ncols*i + j)。
 *  如果 `transposed` 非零，则 c[i, j] 对应 *(c + i + nrows*j)（即 `c` 是按 F-contiguous 排序存储的）。
 */

static void
copy_array_to_fortran(double *f, F_INT ldf, F_INT nrows, F_INT ncols,
                      double *c, F_INT transposed)
{
    F_INT i, j;
    F_INT row_stride, col_stride;

    /* 步长是 sizeof(double) 的倍数，而不是字节数。*/
    if (transposed) {
        row_stride = 1;
        col_stride = nrows;
    }
    else {
        row_stride = ncols;
        col_stride = 1;
    }
    for (i = 0; i < nrows; ++i) {
        for (j = 0; j < ncols; ++j) {
            double value;
            /* value = c[i,j] */
            value = *(c + row_stride*i + col_stride*j);
            /* f[i,j] = value */
            *(f + ldf*j + i) = value;
        }
    }
}


int
ode_jacobian_function(F_INT *n, double *t, double *y, F_INT *ml, F_INT *mu,
                      double *pd, F_INT *nrowpd)
{
    /*
        这是从 Fortran 代码调用的函数。它应该：
            -- 使用 call_odeint_user_function 获取一个 multiarrayobject 结果
            -- 检查错误并在有任何错误时返回 -1（尽管调用程序会忽略这一点）。
            -- 否则将计算结果放入 pd 中
    */

    PyArrayObject *result_array;
    npy_intp ndim, nrows, ncols, dim_error;
    npy_intp *dims;

    result_array = (PyArrayObject *)
                   call_odeint_user_function(global_params.python_jacobian,
                                             *n, y, *t, global_params.tfirst,
                                             global_params.extra_arguments,
                                             odepack_error);
    if (result_array == NULL) {
        *n = -1;
        return -1;
    }

    ncols = *n;
    if (global_params.jac_type == 4) {
        nrows = *ml + *mu + 1;
    }
    else {
        nrows = *n;
    }

    if (!global_params.jac_transpose) {
        npy_intp tmp;
        tmp = nrows;
        nrows = ncols;
        ncols = tmp;
    }

    ndim = PyArray_NDIM(result_array);
    if (ndim > 2) {
        PyErr_Format(PyExc_RuntimeError,
            "Jacobian 数组必须是二维的，但是得到了 ndim=%d。",
            ndim);
        *n = -1;
        Py_DECREF(result_array);
        return -1;
    }

    dims = PyArray_DIMS(result_array);
    dim_error = 0;
    if (ndim == 0) {
        if ((nrows != 1) || (ncols != 1)) {
            dim_error = 1;
        }
    }
    if (ndim == 1) {
        if ((nrows != 1) || (dims[0] != ncols)) {
            dim_error = 1;
        }
    }
    if (ndim == 2) {
        if ((dims[0] != nrows) || (dims[1] != ncols)) {
            dim_error = 1;
        }
    }
    # 如果存在维度错误，执行以下逻辑
    if (dim_error) {
        # 初始化空字符串指针 b
        char *b = "";
        # 如果全局参数中的 jac_type 等于 4
        if (global_params.jac_type == 4) {
            # 将 b 设置为 "banded " 字符串
            b = "banded ";
        }
        # 抛出运行时错误异常，格式化错误信息，包含期望的 Jacobian 数组形状
        PyErr_Format(PyExc_RuntimeError,
            "Expected a %sJacobian array with shape (%d, %d)",
            b, nrows, ncols);
        # 设置 n 为 -1，释放结果数组的 Python 对象引用
        *n = -1;
        Py_DECREF(result_array);
        # 返回 -1 表示执行错误
        return -1;
    }

    /*
     *  global_params.jac_type 可能是 1（完整 Jacobian）或 4（带状 Jacobian）。
     *  global_params.jac_transpose 是 col_deriv 的非值，因此如果 global_params.jac_transpose
     *  是 0，则用户创建的数组已经是 Fortran 顺序，当复制到 pd 时不需要转置。
     */

    # 如果 global_params.jac_type 为 1 并且 global_params.jac_transpose 为 0
    if ((global_params.jac_type == 1) && !global_params.jac_transpose) {
        /* 完整 Jacobian，无需转置，因此可以使用 memcpy 进行快速复制 */
        memcpy(pd, PyArray_DATA(result_array), (*n)*(*nrowpd)*sizeof(double));
    }
    else {
        /*
         *  global_params.jac_type == 4（带状 Jacobian），或者
         *  global_params.jac_type == 1 并且 global_params.jac_transpose == 1。
         *
         *  当 global_params.jac_type 为 4 时无法使用 memcpy，因为 pd 的前导维度不一定等于矩阵的行数。
         */
        # 定义 m 为（完整或压缩带状）Jacobian 的行数
        npy_intp m;  /* Number of rows in the (full or packed banded) Jacobian. */
        # 如果 global_params.jac_type 等于 4
        if (global_params.jac_type == 4) {
            # 计算 m 为 *ml + *mu + 1
            m = *ml + *mu + 1;
        }
        else {
            # 否则 m 等于 *n
            m = *n;
        }
        # 调用函数将数组复制到 Fortran 顺序中
        copy_array_to_fortran(pd, *nrowpd, m, *n,
            (double *) PyArray_DATA(result_array),
            !global_params.jac_transpose);
    }

    # 释放结果数组的 Python 对象引用
    Py_DECREF(result_array);
    # 返回 0 表示执行成功
    return 0;
int
setup_extra_inputs(PyArrayObject **ap_rtol, PyObject *o_rtol,
                   PyArrayObject **ap_atol, PyObject *o_atol,
                   PyArrayObject **ap_tcrit, PyObject *o_tcrit,
                   long *numcrit, int neq)
{
    int itol = 0;   /* 初始化容错标志，默认为0 */
    double tol = 1.49012e-8;   /* 容错的默认值 */
    npy_intp one = 1;   /* 一个整数变量，值为1，用于创建长度为1的数组 */

    /* 设置容错 */
    if (o_rtol == NULL) {
        *ap_rtol = (PyArrayObject *) PyArray_SimpleNew(1, &one, NPY_DOUBLE);
        if (*ap_rtol == NULL) {
            PYERR2(odepack_error,"Error constructing relative tolerance.");
        }
        *(double *) PyArray_DATA(*ap_rtol) = tol;    /* 默认值 */
    }
    else {
        *ap_rtol = (PyArrayObject *) PyArray_ContiguousFromObject(o_rtol,
                                                            NPY_DOUBLE, 0, 1);
        if (*ap_rtol == NULL) {
            PYERR2(odepack_error,"Error converting relative tolerance.");
        }
        /* XXX Fix the following. */
        if (PyArray_NDIM(*ap_rtol) == 0); /* rtol is scalar */  /* 如果 rtol 是标量 */
        else if (PyArray_DIMS(*ap_rtol)[0] == neq) {
            itol |= 2;      /* 设置 rtol 数组标志位 */
        }
        else {
            PYERR(odepack_error, "Tolerances must be an array of the same length as the\n     number of equations or a scalar.");
        }
    }

    /* 设置容错 */
    if (o_atol == NULL) {
        *ap_atol = (PyArrayObject *) PyArray_SimpleNew(1, &one, NPY_DOUBLE);
        if (*ap_atol == NULL) {
            PYERR2(odepack_error,"Error constructing absolute tolerance");
        }
        *(double *)PyArray_DATA(*ap_atol) = tol;
    }
    else {
        *ap_atol = (PyArrayObject *) PyArray_ContiguousFromObject(o_atol, NPY_DOUBLE, 0, 1);
        if (*ap_atol == NULL) {
            PYERR2(odepack_error,"Error converting absolute tolerance.");
        }
        /* XXX Fix the following. */
        if (PyArray_NDIM(*ap_atol) == 0); /* atol is scalar */  /* 如果 atol 是标量 */
        else if (PyArray_DIMS(*ap_atol)[0] == neq) {
            itol |= 1;        /* 设置 atol 数组标志位 */
        }
        else {
            PYERR(odepack_error,"Tolerances must be an array of the same length as the\n     number of equations or a scalar.");
        }
    }
    itol++;             /* 增加以获得正确的值 */

    /* 设置临界时间 */
    if (o_tcrit != NULL) {
        *ap_tcrit = (PyArrayObject *) PyArray_ContiguousFromObject(o_tcrit, NPY_DOUBLE, 0, 1);
        if (*ap_tcrit == NULL) {
            PYERR2(odepack_error,"Error constructing critical times.");
        }
        *numcrit = PyArray_Size((PyObject *) (*ap_tcrit));
    }
    return itol;

fail:       /* 为了使用 PYERR 而需要 */
    return -1;
}
    // 如果条件为真，抛出 odepack_error 异常，指出 jt 值不正确
    else {
        PYERR(odepack_error,"Incorrect value for jt.");
    }

    // 如果 mxordn 小于 0，抛出 odepack_error 异常，指出 mxordn 值不正确
    if (mxordn < 0) {
        PYERR(odepack_error,"Incorrect value for mxordn.");
    }
    
    // 如果 mxords 小于 0，抛出 odepack_error 异常，指出 mxords 值不正确
    if (mxords < 0) {
        PYERR(odepack_error,"Incorrect value for mxords.");
    }
    
    // 将 nyh 设置为 neq 的值
    nyh = neq;

    // 计算 lrn 的值，用于分配内存空间，包括状态数据
    lrn = 20 + nyh*(mxordn+1) + 3*neq;
    
    // 计算 lrs 的值，用于分配内存空间，包括结果数据和工作空间
    lrs = 20 + nyh*(mxords+1) + 3*neq + lmat;

    // 设置 lrw 指针指向的值为 lrn 和 lrs 中的较大者
    *lrw = PyArray_MAX(lrn,lrs);
    
    // 设置 liw 指针指向的值为 20 + neq
    *liw = 20 + neq;
    
    // 返回 0 表示函数执行成功
    return 0;
    // 函数开始
    static PyObject *
    odepack_odeint(PyObject *dummy, PyObject *args, PyObject *kwdict)
    {
        PyObject *fcn, *y0, *p_tout, *o_rtol = NULL, *o_atol = NULL;
        PyArrayObject *ap_y = NULL, *ap_yout = NULL;
        PyArrayObject *ap_rtol = NULL, *ap_atol = NULL;
        PyArrayObject *ap_tout = NULL;
        PyObject *extra_args = NULL;
        PyObject *Dfun = Py_None;
        F_INT neq, itol = 1, itask = 1, istate = 1, iopt = 0, lrw, *iwork, liw, jt = 4;
        double *y, t, *tout, *rtol, *atol, *rwork;
        double h0 = 0.0, hmax = 0.0, hmin = 0.0;
        long ixpr = 0, mxstep = 0, mxhnil = 0, mxordn = 12, mxords = 5, ml = -1, mu = -1;
        long tfirst;
        PyObject *o_tcrit = NULL;
        PyArrayObject *ap_tcrit = NULL;
        PyArrayObject *ap_hu = NULL, *ap_tcur = NULL, *ap_tolsf = NULL, *ap_tsw = NULL;
        PyArrayObject *ap_nst = NULL, *ap_nfe = NULL, *ap_nje = NULL, *ap_nqu = NULL;
        PyArrayObject *ap_mused = NULL;
        long imxer = 0, lenrw = 0, leniw = 0, col_deriv = 0;
        npy_intp out_sz = 0, dims[2];
        long k, ntimes, crit_ind = 0;
        long allocated = 0, full_output = 0, numcrit = 0;
        long t0count;
        double *yout, *yout_ptr, *tout_ptr, *tcrit = NULL;
        double *wa;
        static char *kwlist[] = {"fun", "y0", "t", "args", "Dfun", "col_deriv",
                                 "ml", "mu", "full_output", "rtol", "atol", "tcrit",
                                 "h0", "hmax", "hmin", "ixpr", "mxstep", "mxhnil",
                                 "mxordn", "mxords", "tfirst", NULL};
        odepack_params save_params;

        // 解析传入的参数并检查
        if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OOllllOOOdddllllll", kwlist,
                                         &fcn, &y0, &p_tout, &extra_args, &Dfun,
                                         &col_deriv, &ml, &mu, &full_output, &o_rtol, &o_atol,
                                         &o_tcrit, &h0, &hmax, &hmin, &ixpr, &mxstep, &mxhnil,
                                         &mxordn, &mxords, &tfirst)) {
            return NULL;
        }

        // 处理可选参数的特殊情况
        if (o_tcrit == Py_None) {
            o_tcrit = NULL;
        }
        if (o_rtol == Py_None) {
            o_rtol = NULL;
        }
        if (o_atol == Py_None) {
            o_atol = NULL;
        }

        /* 设置 jt, ml, 和 mu */
        if (Dfun == Py_None) {
            /* 如果没有传入 Dfun，则设置 jt 为内部生成 */
            jt++;
        }
        if (ml < 0 && mu < 0) {
            /* 如果 ml 和 mu 都未给出，则标记 jt 为完整雅可比 */
            jt -= 3;
        }
        if (ml < 0) {
            /* 如果只给出一个而未给出另一个，则 ml 设置为 0 */
            ml = 0;
        }
        if (mu < 0) {
            /* 如果只给出一个而未给出另一个，则 mu 设置为 0 */
            mu = 0;
        }

        /* 将当前的 global_params 存储到 save_params 中 */
        memcpy(&save_params, &global_params, sizeof(save_params));
    /* 如果 extra_args 是 NULL，则创建一个空元组 */
    if (extra_args == NULL) {
        /* 如果创建空元组失败，则跳转到 fail 标签 */
        if ((extra_args = PyTuple_New(0)) == NULL) {
            goto fail;
        }
    }
    else {
        /* 增加 extra_args 的引用计数，退出时会减少引用计数 */
        Py_INCREF(extra_args);   /* We decrement on exit. */
    }

    /* 检查 extra_args 是否为元组类型 */
    if (!PyTuple_Check(extra_args)) {
        PYERR(odepack_error, "Extra arguments must be in a tuple.");
    }

    /* 检查 fcn 和 Dfun 是否为可调用对象 */
    if (!PyCallable_Check(fcn) || (Dfun != Py_None && !PyCallable_Check(Dfun))) {
        PYERR(odepack_error, "The function and its Jacobian must be callable functions.");
    }

    /* 设置全局参数 global_params */
    global_params.python_function = fcn;
    global_params.extra_arguments = extra_args;
    global_params.python_jacobian = Dfun;
    global_params.jac_transpose = !(col_deriv);
    global_params.jac_type = jt;
    global_params.tfirst = tfirst;

    /* 初始化输入向量 y0 */
    ap_y = (PyArrayObject *) PyArray_ContiguousFromObject(y0, NPY_DOUBLE, 0, 0);
    if (ap_y == NULL) {
        goto fail;
    }
    /* 检查 y0 是否为一维数组 */
    if (PyArray_NDIM(ap_y) > 1) {
        PyErr_SetString(PyExc_ValueError, "Initial condition y0 must be one-dimensional.");
        goto fail;
    }
    y = (double *) PyArray_DATA(ap_y);
    neq = PyArray_Size((PyObject *) ap_y);
    dims[1] = neq;

    /* 设置积分的输出时间点 */
    ap_tout = (PyArrayObject *) PyArray_ContiguousFromObject(p_tout, NPY_DOUBLE, 0, 0);
    if (ap_tout == NULL) {
        goto fail;
    }
    /* 检查输出时间点数组是否为一维数组 */
    if (PyArray_NDIM(ap_tout) > 1) {
        PyErr_SetString(PyExc_ValueError, "Output times t must be one-dimensional.");
        goto fail;
    }
    tout = (double *) PyArray_DATA(ap_tout);
    ntimes = PyArray_Size((PyObject *)ap_tout);
    dims[0] = ntimes;

    t0count = 0;
    if (ntimes > 0) {
        /* 复制 tout[0] 给 t，并统计它出现的次数 */
        t = tout[0];
        t0count = 1;
        while ((t0count < ntimes) && (tout[t0count] == t)) {
            ++t0count;
        }
    }

    /* 设置用于存储输出结果的数组 ap_yout */
    ap_yout= (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    if (ap_yout== NULL) {
        goto fail;
    }
    yout = (double *) PyArray_DATA(ap_yout);

    /* 将初始向量复制到输出数组的第一行（或多行） */
    yout_ptr = yout;
    for (k = 0; k < t0count; ++k) {
        memcpy(yout_ptr, y, neq*sizeof(double));
        yout_ptr += neq;
    }

    /* 设置额外的输入参数 */
    itol = setup_extra_inputs(&ap_rtol, o_rtol, &ap_atol, o_atol, &ap_tcrit,
                              o_tcrit, &numcrit, neq);
    if (itol < 0 ) {
        goto fail;  /* 如果设置失败，则跳转到 fail 标签 */
    }
    rtol = (double *) PyArray_DATA(ap_rtol);
    atol = (double *) PyArray_DATA(ap_atol);

    /* 计算工作数组的大小 */
    if (compute_lrw_liw(&lrw, &liw, neq, jt, ml, mu, mxordn, mxords) < 0) {
        goto fail;
    }

    /* 分配工作数组的内存 */
    if ((wa = (double *)malloc(lrw*sizeof(double) + liw*sizeof(F_INT)))==NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    allocated = 1;
    rwork = wa;
    iwork = (F_INT *)(wa + lrw);

    /* 设置工作数组中的初始值 */
    iwork[0] = ml;
    iwork[1] = mu;

设置 `iwork` 数组的第二个元素为 `mu`。


    if (h0 != 0.0 || hmax != 0.0 || hmin != 0.0 || ixpr != 0 || mxstep != 0 ||
            mxhnil != 0 || mxordn != 0 || mxords != 0) {

如果 `h0`, `hmax`, `hmin`, `ixpr`, `mxstep`, `mxhnil`, `mxordn`, `mxords` 中的任何一个不等于0，则执行下面的操作。


        rwork[4] = h0;
        rwork[5] = hmax;
        rwork[6] = hmin;
        iwork[4] = ixpr;
        iwork[5] = mxstep;
        iwork[6] = mxhnil;
        iwork[7] = mxordn;
        iwork[8] = mxords;
        iopt = 1;

将 `h0`, `hmax`, `hmin` 分别存储到 `rwork` 数组的第5、6、7个元素，将 `ixpr`, `mxstep`, `mxhnil`, `mxordn`, `mxords` 分别存储到 `iwork` 数组的第5、6、7、8、9个元素，并将 `iopt` 设置为1。


    istate = 1;

将 `istate` 设置为1。


    k = t0count;

将 `k` 设置为 `t0count` 的值。


    /* If full output make some useful output arrays */
    if (full_output) {

如果 `full_output` 为真，则执行以下操作。


        out_sz = ntimes-1;

计算 `out_sz` 的值为 `ntimes - 1`。


        ap_hu = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, NPY_DOUBLE);
        ap_tcur = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, NPY_DOUBLE);
        ap_tolsf = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, NPY_DOUBLE);
        ap_tsw = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, NPY_DOUBLE);
        ap_nst = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, F_INT_NPY);
        ap_nfe = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, F_INT_NPY);
        ap_nje = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, F_INT_NPY);
        ap_nqu = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, F_INT_NPY);
        ap_mused = (PyArrayObject *) PyArray_SimpleNew(1, &out_sz, F_INT_NPY);

为 `ap_hu`, `ap_tcur`, `ap_tolsf`, `ap_tsw`, `ap_nst`, `ap_nfe`, `ap_nje`, `ap_nqu`, `ap_mused` 分别创建新的 `PyArrayObject` 数组对象。


        if (ap_hu == NULL || ap_tcur == NULL || ap_tolsf == NULL ||
                ap_tsw == NULL || ap_nst == NULL || ap_nfe == NULL ||
                ap_nje == NULL || ap_nqu == NULL || ap_mused == NULL) {
            goto fail;
        }
    }

如果任何一个数组对象的创建失败，则跳转到 `fail` 标签处。


    if (o_tcrit != NULL) {

如果 `o_tcrit` 不为空，则执行以下操作。


        /* There are critical points */
        itask = 4;

设置 `itask` 的值为4，表示存在临界点任务。


        tcrit = (double *)(PyArray_DATA(ap_tcrit));
        rwork[0] = *tcrit;
    }

将 `ap_tcrit` 数组的数据指针转换为 `double*` 类型，并将其第一个元素赋值给 `rwork` 数组的第一个元素。
    while (k < ntimes && istate > 0) {    /* 循环处理所需的时间点 */

        tout_ptr = tout + k;
        /* 如果使用 itask == 4，则检查是否需要使用 tcrit */
        if (itask == 4) {
            if (!tcrit) {
                PYERR(odepack_error, "Internal error - tcrit must be defined!");
            }
            /* 检查当前时间是否超过了临界时间 */
            if (*tout_ptr > *(tcrit + crit_ind)) {
                crit_ind++;
                rwork[0] = *(tcrit+crit_ind);
            }
        }
        /* 如果已经处理完所有的临界时间点 */
        if (crit_ind >= numcrit) {
            itask = 1;  /* 没有更多临界时间点 */
        }

        /* 调用 LSODA 进行积分计算 */
        LSODA(ode_function, &neq, y, &t, tout_ptr, &itol, rtol, atol, &itask,
              &istate, &iopt, rwork, &lrw, iwork, &liw,
              ode_jacobian_function, &jt);
        
        /* 如果需要完整输出 */
        if (full_output) {
            /* 将 LSODA 计算得到的结果保存到相应的数组中 */
            *((double *)PyArray_DATA(ap_hu) + (k-1)) = rwork[10];
            *((double *)PyArray_DATA(ap_tcur) + (k-1)) = rwork[12];
            *((double *)PyArray_DATA(ap_tolsf) + (k-1)) = rwork[13];
            *((double *)PyArray_DATA(ap_tsw) + (k-1)) = rwork[14];
            *((F_INT *)PyArray_DATA(ap_nst) + (k-1)) = iwork[10];
            *((F_INT *)PyArray_DATA(ap_nfe) + (k-1)) = iwork[11];
            *((F_INT *)PyArray_DATA(ap_nje) + (k-1)) = iwork[12];
            *((F_INT *)PyArray_DATA(ap_nqu) + (k-1)) = iwork[13];
            /* 如果出现了错误状态 */
            if (istate == -5 || istate == -4) {
                imxer = iwork[15];
            }
            else {
                imxer = -1;
            }
            lenrw = iwork[16];
            leniw = iwork[17];
            *((F_INT *)PyArray_DATA(ap_mused) + (k-1)) = iwork[18];
        }
        /* 检查是否有 Python 异常发生 */
        if (PyErr_Occurred()) {
            goto fail;
        }
        /* 将积分计算结果拷贝到输出数组中 */
        memcpy(yout_ptr, y, neq*sizeof(double));  /* 将积分结果拷贝到输出中 */
        yout_ptr += neq;
        k++;
    }

    /* 恢复 global_params 到之前保存的 save_params */
    memcpy(&global_params, &save_params, sizeof(save_params));

    /* 释放 Python 对象的引用 */
    Py_DECREF(extra_args);
    Py_DECREF(ap_atol);
    Py_DECREF(ap_rtol);
    Py_XDECREF(ap_tcrit);
    Py_DECREF(ap_y);
    Py_DECREF(ap_tout);
    free(wa);

    /* 如果需要完整输出，则构建返回值 */
    if (full_output) {
        return Py_BuildValue("N{s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:l,s:l,s:l,s:N}l",
                    PyArray_Return(ap_yout),
                    "hu", PyArray_Return(ap_hu),
                    "tcur", PyArray_Return(ap_tcur),
                    "tolsf", PyArray_Return(ap_tolsf),
                    "tsw", PyArray_Return(ap_tsw),
                    "nst", PyArray_Return(ap_nst),
                    "nfe", PyArray_Return(ap_nfe),
                    "nje", PyArray_Return(ap_nje),
                    "nqu", PyArray_Return(ap_nqu),
                    "imxer", imxer,
                    "lenrw", lenrw,
                    "leniw", leniw,
                    "mused", PyArray_Return(ap_mused),
                    (long)istate);
    }
    else {
        /* 否则，返回简化的结果 */
        return Py_BuildValue("Nl", PyArray_Return(ap_yout), (long)istate);
    }
fail:
    /* 将全局参数从之前存储的 save_params 中恢复 */
    memcpy(&global_params, &save_params, sizeof(save_params));

    // 释放引用计数为NULL的对象
    Py_XDECREF(extra_args);
    Py_XDECREF(ap_y);
    Py_XDECREF(ap_rtol);
    Py_XDECREF(ap_atol);
    Py_XDECREF(ap_tcrit);
    Py_XDECREF(ap_tout);
    Py_XDECREF(ap_yout);
    
    // 如果已分配内存，释放wa指向的内存空间
    if (allocated) {
        free(wa);
    }
    
    // 如果需要完整输出，释放引用计数为NULL的对象
    if (full_output) {
        Py_XDECREF(ap_hu);
        Py_XDECREF(ap_tcur);
        Py_XDECREF(ap_tolsf);
        Py_XDECREF(ap_tsw);
        Py_XDECREF(ap_nst);
        Py_XDECREF(ap_nfe);
        Py_XDECREF(ap_nje);
        Py_XDECREF(ap_nqu);
        Py_XDECREF(ap_mused);
    }
    
    // 函数返回空指针
    return NULL;
}


static struct PyMethodDef odepack_module_methods[] = {
    // 定义Python模块中的方法odeint，其实现为odepack_odeint，接受位置参数和关键字参数，带有文档字符串doc_odeint
    {"odeint", (PyCFunction) odepack_odeint, METH_VARARGS|METH_KEYWORDS, doc_odeint},
    // 结束方法定义列表，标记结尾为NULL
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    // Python模块定义的头部初始化
    PyModuleDef_HEAD_INIT,
    // 模块名称为"_odepack"
    "_odepack",
    // 模块文档字符串为空
    NULL,
    // 模块状态为-1
    -1,
    // 模块方法定义列表为odepack_module_methods
    odepack_module_methods,
    // 模块在初始化和清理时的内存管理方法为空
    NULL,
    // 模块状态机制为空
    NULL,
    // 模块的全局状态为空
    NULL,
    // 模块的内存分配函数为空
    NULL
};

// Python模块的初始化函数，名称为PyInit__odepack
PyMODINIT_FUNC
PyInit__odepack(void)
{
    PyObject *module, *mdict;

    // 导入NumPy的数组接口
    import_array();

    // 创建Python模块对象
    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        // 如果模块创建失败，返回空指针
        return NULL;
    }

    // 获取Python模块的字典对象
    mdict = PyModule_GetDict(module);
    if (mdict == NULL) {
        // 如果获取字典对象失败，返回空指针
        return NULL;
    }

    // 创建一个名为"_odepack.error"的新异常对象
    odepack_error = PyErr_NewException("_odepack.error", NULL, NULL);
    if (odepack_error == NULL) {
        // 如果创建异常对象失败，返回空指针
        return NULL;
    }

    // 将异常对象添加到模块的字典中，键为"error"
    if (PyDict_SetItemString(mdict, "error", odepack_error)) {
        // 如果添加失败，返回空指针
        return NULL;
    }

    // 返回创建的Python模块对象
    return module;
}
```