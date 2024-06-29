# `.\numpy\numpy\fft\_pocketfft_umath.cpp`

```py
/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 */

/*
 *  Main implementation file.
 *
 *  Copyright (C) 2004-2018 Max-Planck-Society
 *  \author Martin Reinecke
 */

// 定义宏，防止使用废弃的 NumPy API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

// 清理 PY_SSIZE_T 类型
#define PY_SSIZE_T_CLEAN
#include <assert.h>
#include <Python.h>

// 引入 NumPy 头文件
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

// 引入 NumPy 的配置文件
#include "npy_config.h"

// 定义不使用多线程选项
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft/pocketfft_hdronly.h"

/*
 * In order to ensure that C++ exceptions are converted to Python
 * ones before crossing over to the C machinery, we must catch them.
 * This template can be used to wrap a C++ written ufunc to do this via:
 *      wrap_legacy_cpp_ufunc<cpp_ufunc>
 */
template<PyUFuncGenericFunction cpp_ufunc>
static void
wrap_legacy_cpp_ufunc(char **args, npy_intp const *dimensions,
                      ptrdiff_t const *steps, void *func)
{
    NPY_ALLOW_C_API_DEF
    try {
        cpp_ufunc(args, dimensions, steps, func);
    }
    catch (std::bad_alloc& e) {
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    catch (const std::exception& e) {
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        NPY_DISABLE_C_API;
    }
}

/*
 * Transfer to and from a contiguous buffer.
 * copy_input: copy min(nin, n) elements from input to buffer and zero rest.
 * copy_output: copy n elements from buffer to output.
 */
template <typename T>
static inline void
copy_input(char *in, npy_intp step_in, size_t nin,
           T buff[], size_t n)
{
    // 复制输入数据到缓冲区，如果输入数据量小于缓冲区大小，则剩余部分填充为零
    size_t ncopy = nin <= n ? nin : n;
    char *ip = in;
    size_t i;
    for (i = 0; i < ncopy; i++, ip += step_in) {
      buff[i] = *(T *)ip;
    }
    for (; i < n; i++) {
      buff[i] = 0;
    }
}

template <typename T>
static inline void
copy_output(T buff[], char *out, npy_intp step_out, size_t n)
{
    // 从缓冲区复制数据到输出
    char *op = out;
    for (size_t i = 0; i < n; i++, op += step_out) {
        *(T *)op = buff[i];
    }
}

/*
 * Gufunc loops calling the pocketfft code.
 */
template <typename T>
static void
fft_loop(char **args, npy_intp const *dimensions, ptrdiff_t const *steps,
         void *func)
{
    // 获取输入、中间、输出位置
    char *ip = args[0], *fp = args[1], *op = args[2];
    // 获取循环次数和步长
    size_t n_outer = (size_t)dimensions[0];
    ptrdiff_t si = steps[0], sf = steps[1], so = steps[2];
    // 获取输入和输出的大小
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    // 获取输入和输出的步长
    ptrdiff_t step_in = steps[3], step_out = steps[4];
    // 获取 FFT 的方向，前向还是后向
    bool direction = *((bool *)func); /* pocketfft::FORWARD or BACKWARD */

    // 断言输出大小大于零
    assert (nout > 0);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * For the common case of nin >= nout, fixed factor, and suitably sized
     * outer loop, we call pocketfft directly to benefit from its vectorization.
     * (For nin>nout, this just removes the extra input points, as required;
     * the vlen constraint avoids compiling extra code for longdouble, which
     * cannot be vectorized so does not benefit.)
     */
    # 获取模板参数 T 对应的 SIMD 长度
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    # 检查是否满足使用 SIMD 加速的条件：SIMD 长度大于 1，外层循环次数大于等于 SIMD 长度，输入输出数据长度符合要求，且没有指定数据分裂
    if (vlen > 1 && n_outer >= vlen && nin >= nout && sf == 0) {
        # 定义用于计算的形状参数，此处是二维情况，形状为 [n_outer, nout]
        std::vector<size_t> shape = { n_outer, nout };
        # 定义输入数据的步长数组
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        # 定义输出数据的步长数组
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        # 定义变换操作的轴数组，这里指定在第二个轴上进行变换
        std::vector<size_t> axes = { 1 };
        # 调用 pocketfft 库的复数到复数变换函数，进行二维傅里叶变换
        pocketfft::c2c(shape, strides_in, strides_out, axes, direction,
                       (std::complex<T> *)ip, (std::complex<T> *)op, *(T *)fp);
        # 函数执行完毕，直接返回
        return;
    }
    #endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    // 获取 FFT 执行计划，使用 pocketfft 库中的具体实现 pocketfft_c<T>
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_c<T>>(nout);
    // 检查输出是否不是连续的，如果不是则需要缓冲区
    auto buffered = (step_out != sizeof(std::complex<T>));
    // 创建一个 std::complex<T> 类型的数组，用作缓冲区，如果需要的话
    pocketfft::detail::arr<std::complex<T>> buff(buffered ? nout : 0);
    // 外部循环，处理每个输入输出的数据块
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        // 确定要操作的输出数据位置
        std::complex<T> *op_or_buff = buffered ? buff.data() : (std::complex<T> *)op;
        // 如果输入数据和输出数据不在同一位置，则进行复制操作
        if (ip != (char*)op_or_buff) {
            copy_input(ip, step_in, nin, op_or_buff, nout);
        }
        // 执行 FFT 变换
        plan->exec((pocketfft::detail::cmplx<T> *)op_or_buff, *(T *)fp, direction);
        // 如果使用了缓冲区，则需要将结果从缓冲区复制回输出位置
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    // 函数返回，处理结束
    return;
}

template <typename T>
static void
rfft_impl(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *func, size_t npts)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = (size_t)dimensions[0];
    ptrdiff_t si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    ptrdiff_t step_in = steps[3], step_out = steps[4];

    // 断言确保输出数据点数大于 0，且符合实际情况
    assert (nout > 0 && nout == npts / 2 + 1);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * Call pocketfft directly if vectorization is possible.
     */
    // 如果支持向量化，并且条件允许，直接调用 pocketfft 库中的向量化函数
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= npts && sf == 0) {
        // 定义输入和输出数据的形状和步长
        std::vector<size_t> shape_in = { n_outer, npts };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        std::vector<size_t> axes = { 1 };
        // 调用向量化的 r2c FFT 变换
        pocketfft::r2c(shape_in, strides_in, strides_out, axes, pocketfft::FORWARD,
                       (T *)ip, (std::complex<T> *)op, *(T *)fp);
        // 函数返回，处理结束
        return;
    }
#endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    // 获取 FFT 执行计划，使用 pocketfft 库中的具体实现 pocketfft_r<T>
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(npts);
    // 检查输出是否不是连续的，如果不是则需要缓冲区
    auto buffered = (step_out != sizeof(std::complex<T>));
    // 创建一个 std::complex<T> 类型的数组，用作缓冲区，如果需要的话
    pocketfft::detail::arr<std::complex<T>> buff(buffered ? nout : 0);
    // 确定实际使用的输入数据点数，取最小值
    auto nin_used = nin <= npts ? nin : npts;
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        // 使用条件运算符确定 op_or_buff 是直接使用缓冲区还是指向 op 的指针
        std::complex<T> *op_or_buff = buffered ? buff.data() : (std::complex<T> *)op;
        
        /*
         * 内部的 pocketfft 程序在原地工作，对于实数变换，频率数据因此需要压缩，
         * 利用这一点，在零频率项（即所有输入的总和，因此必须是实数）和对于偶数点数的奈奎斯特频率项没有虚部。
         * Pocketfft 使用 FFTpack 的顺序，R0,R1,I1,...Rn-1,In-1,Rn（仅当点数 npts 为奇数时最后的 In）。为了使解包易于进行，
         * 我们在缓冲区中将实数数据偏移了一个位置，因此我们只需要移动 R0 并创建 I0=0。注意，copy_input 将会将偶数点数的情况下的 In 分量置零。
         */
        // 调用 copy_input 函数，将输入数据复制到 op_or_buff 中
        copy_input(ip, step_in, nin_used, &((T *)op_or_buff)[1], nout*2 - 1);
        
        // 执行 FFT 变换，使用 pocketfft 库的前向变换
        plan->exec(&((T *)op_or_buff)[1], *(T *)fp, pocketfft::FORWARD);
        
        // 将 op_or_buff 的第一个元素设为其虚部，实现 I0->R0, I0=0 的转换
        op_or_buff[0] = op_or_buff[0].imag();
        
        // 如果使用了缓冲区，将处理完的输出数据复制回 op
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    // 函数返回
    return;
/*
 * For the forward real, we cannot know what the requested number of points is
 * just based on the number of points in the complex output array (e.g., 10
 * and 11 real input points both lead to 6 complex output points), so we
 * define versions for both even and odd number of points.
 */
template <typename T>
static void
rfft_n_even_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    // 获取输出数组中的复数点数
    size_t nout = (size_t)dimensions[2];
    assert (nout > 0); // 断言：输出点数应大于零
    // 计算输入点数
    size_t npts = 2 * nout - 2;
    // 调用实际的 FFT 实现函数
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

/*
 * For the forward real, we cannot know what the requested number of points is
 * just based on the number of points in the complex output array (e.g., 10
 * and 11 real input points both lead to 6 complex output points), so we
 * define versions for both even and odd number of points.
 */
template <typename T>
static void
rfft_n_odd_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    // 获取输出数组中的复数点数
    size_t nout = (size_t)dimensions[2];
    assert (nout > 0); // 断言：输出点数应大于零
    // 计算输入点数
    size_t npts = 2 * nout - 1;
    // 调用实际的 FFT 实现函数
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

/*
 * This function handles the inverse real FFT operation.
 */
template <typename T>
static void
irfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    // 获取输入、滤波器和输出数组的指针
    char *ip = args[0], *fp = args[1], *op = args[2];
    // 获取外部循环的大小
    size_t n_outer = (size_t)dimensions[0];
    // 获取输入和输出数组的步长
    ptrdiff_t si = steps[0], sf = steps[1], so = steps[2];
    // 获取输入数组的大小和输出数组的复数点数
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    // 获取输入和输出数组的步长
    ptrdiff_t step_in = steps[3], step_out = steps[4];

    // 计算输入数组中的点数
    size_t npts_in = nout / 2 + 1;

    assert(nout > 0); // 断言：输出点数应大于零

#ifndef POCKETFFT_NO_VECTORS
    /*
     * Call pocketfft directly if vectorization is possible.
     */
    // 如果支持向量化，并且满足调用条件，则直接调用 pocketfft 函数进行计算
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= npts_in && sf == 0) {
        // 设置要进行计算的维度和步长
        std::vector<size_t> axes = { 1 };
        std::vector<size_t> shape_out = { n_outer, nout };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out };
        // 调用 pocketfft 的逆变换函数
        pocketfft::c2r(shape_out, strides_in, strides_out, axes, pocketfft::BACKWARD,
                       (std::complex<T> *)ip, (T *)op, *(T *)fp);
        return;
    }
#endif

    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    // 否则，使用非向量化的循环进行计算，尽量减少拷贝操作
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(nout);
    auto buffered = (step_out != sizeof(T));
    // 如果输出不是连续的，则分配缓冲区
    pocketfft::detail::arr<T> buff(buffered ? nout : 0);
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        // 确定输出数组的位置，可以是缓冲区或者直接操作数组
        T *op_or_buff = buffered ? buff.data() : (T *)op;
        
        /*
         * Pocket_fft 在原地操作，对于反向实数变换，频率数据需要压缩，
         * 移除零频率项的虚部（这是所有输入的总和，因此必须是实数），
         * 以及偶数点数时的奈奎斯特频率的虚部。因此，我们按以下顺序将数据复制到缓冲区
         * （也被 FFTpack 使用）：R0,R1,I1,...Rn-1,In-1,Rn[,In]（对于奇数点数才有In）。
         */
        
        // 复制 R0 到输出数组或缓冲区的第一个位置
        op_or_buff[0] = ((T *)ip)[0];  /* copy R0 */
        
        // 如果输出点数大于1
        if (nout > 1) {
            /*
             * 复制 R1,I1... 直到 Rn-1,In-1（如果可能），如果不需要所有输入点数或者输入较短，
             * 则提前停止并在其后补零。
             */
            copy_input(ip + step_in, step_in, nin - 1,
                       (std::complex<T> *)&op_or_buff[1], (nout - 1) / 2);
            
            // 对于偶数的 nout，仍然需要设置 Rn
            if (nout % 2 == 0) {
                op_or_buff[nout - 1] = (nout / 2 >= nin) ? (T)0 :
                    ((T *)(ip + (nout / 2) * step_in))[0];
            }
        }
        
        // 执行逆向变换操作
        plan->exec(op_or_buff, *(T *)fp, pocketfft::BACKWARD);
        
        // 如果使用了缓冲区，则将结果复制回输出数组
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    // 函数结束，无返回值
    return;
}

// 定义用于 FFT 的通用函数指针数组，包含双精度、单精度和长双精度的前向 FFT
static PyUFuncGenericFunction fft_functions[] = {
    wrap_legacy_cpp_ufunc<fft_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<fft_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<fft_loop<npy_longdouble>>
};

// 定义 FFT 的数据类型数组，包括复数双精度、双精度、复数单精度、单精度、复数长双精度、长双精度
static const char fft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};

// 定义用于 FFT 的数据指针数组，全部指向前向 FFT
static void *const fft_data[] = {
    (void*)&pocketfft::FORWARD,
    (void*)&pocketfft::FORWARD,
    (void*)&pocketfft::FORWARD
};

// 定义用于 IFFT 的数据指针数组，全部指向后向 FFT
static void *const ifft_data[] = {
    (void*)&pocketfft::BACKWARD,
    (void*)&pocketfft::BACKWARD,
    (void*)&pocketfft::BACKWARD
};

// 定义用于偶数长度实数 FFT 的通用函数指针数组，包含双精度、单精度和长双精度
static PyUFuncGenericFunction rfft_n_even_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_longdouble>>
};

// 定义用于奇数长度实数 FFT 的通用函数指针数组，包含双精度、单精度和长双精度
static PyUFuncGenericFunction rfft_n_odd_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_longdouble>>
};

// 定义实数 FFT 的数据类型数组，包括双精度、复数双精度、单精度、复数单精度、长双精度、复数长双精度
static const char rfft_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_FLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};

// 定义用于逆 FFT 的通用函数指针数组，包含双精度、单精度和长双精度
static PyUFuncGenericFunction irfft_functions[] = {
    wrap_legacy_cpp_ufunc<irfft_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<irfft_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<irfft_loop<npy_longdouble>>
};

// 定义逆 FFT 的数据类型数组，包括复数双精度、双精度、单精度、复数单精度、长双精度、双精度
static const char irfft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE
};

// 添加通用函数到给定的 Python 字典
static int
add_gufuncs(PyObject *dictionary) {
    PyObject *f;

    // 创建 fft 函数对象并添加到字典中
    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, fft_data, fft_types, 3, 2, 1, PyUFunc_None,
        "fft", "complex forward FFT\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "fft", f);
    Py_DECREF(f);

    // 创建 ifft 函数对象并添加到字典中
    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, ifft_data, fft_types, 3, 2, 1, PyUFunc_None,
        "ifft", "complex backward FFT\n", 0, "(m),()->(n)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "ifft", f);
    Py_DECREF(f);

    // 创建 rfft_n_even 函数对象并添加到字典中
    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_even_functions, NULL, rfft_types, 3, 2, 1, PyUFunc_None,
        "rfft_n_even", "real forward FFT for even n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_even", f);
    Py_DECREF(f);

    // 创建 rfft_n_odd 函数对象并添加到字典中
    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_odd_functions, NULL, rfft_types, 3, 2, 1, PyUFunc_None,
        "rfft_n_odd", "real forward FFT for odd n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_odd", f);
    Py_DECREF(f);
    # 调用 PyUFunc_FromFuncAndDataAndSignature 函数创建一个 PyUFunc 对象，并指定相关参数
    f = PyUFunc_FromFuncAndDataAndSignature(
        irfft_functions, NULL, irfft_types, 3, 2, 1, PyUFunc_None,
        "irfft", "real backward FFT\n", 0, "(m),()->(n)");
    
    # 如果创建失败（即 f 为 NULL），则返回 -1
    if (f == NULL) {
        return -1;
    }
    
    # 将创建的 PyUFunc 对象 f 添加到 dictionary 字典中，键为 "irfft"
    PyDict_SetItemString(dictionary, "irfft", f);
    
    # 减少 PyUFunc 对象的引用计数，避免内存泄漏
    Py_DECREF(f);
    
    # 返回成功状态码 0
    return 0;
}

static struct PyModuleDef moduledef = {
    // 定义 Python 模块的基本信息，使用默认的头部初始化
    PyModuleDef_HEAD_INIT,
    "_multiarray_umath",  // 模块名为 "_multiarray_umath"
    NULL,  // 模块的文档字符串为 NULL
    -1,    // 模块状态为 -1（表示模块不可重入）
    NULL,  // 模块方法结构体为 NULL
    NULL,  // 模块全局变量的结构体为 NULL
    NULL,  // 模块的初始化函数为 NULL
    NULL,  // 模块的清理函数为 NULL
    NULL   // 模块的销毁函数为 NULL
};

/* Initialization function for the module */
// 模块的初始化函数，命名为 PyInit__pocketfft_umath
PyMODINIT_FUNC PyInit__pocketfft_umath(void)
{
    // 创建一个 Python 模块对象
    PyObject *m = PyModule_Create(&moduledef);
    // 如果创建失败，返回 NULL
    if (m == NULL) {
        return NULL;
    }

    /* Import the array and ufunc objects */
    // 导入数组对象和通用函数对象
    import_array();
    import_ufunc();

    // 获取模块的字典对象
    PyObject *d = PyModule_GetDict(m);
    // 如果添加通用函数失败，清理内存并返回 NULL
    if (add_gufuncs(d) < 0) {
        Py_DECREF(d);
        Py_DECREF(m);
        return NULL;
    }

    // 返回创建的模块对象
    return m;
}
```