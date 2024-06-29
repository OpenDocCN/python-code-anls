# `.\numpy\numpy\_core\src\multiarray\vdot.c`

```py
/*
 * 定义宏，禁用已弃用的 NumPy API，并指定使用当前版本的 API
 * 定义宏，指示要包含 multiarray 模块
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 定义宏，确保在包含 Python.h 之前清理 PY_SSIZE_T 的定义
 * 包含 Python.h 头文件，提供 Python C API 的支持
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
 * 包含常用函数的头文件
 * 包含向量点积相关函数的头文件
 * 包含与 numpy 的 CBLAS 接口相关的头文件
 */
#include "common.h"
#include "vdot.h"
#include "npy_cblas.h"


/*
 * 所有数据假定是对齐的。
 * 计算复数浮点数向量点积的函数
 */
NPY_NO_EXPORT void
CFLOAT_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
            char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
#if defined(HAVE_CBLAS)
    // 计算每个输入向量的步长，确保对齐
    CBLAS_INT is1b = blas_stride(is1, sizeof(npy_cfloat));
    CBLAS_INT is2b = blas_stride(is2, sizeof(npy_cfloat));

    // 如果步长均合法，则使用 CBLAS 函数计算点积
    if (is1b && is2b) {
        // 使用双精度浮点数数组保持计算稳定性
        double sum[2] = {0., 0.};

        // 循环计算向量点积，使用块处理以提高效率
        while (n > 0) {
            // 确定当前处理的块大小
            CBLAS_INT chunk = n < NPY_CBLAS_CHUNK ? n : NPY_CBLAS_CHUNK;
            float tmp[2];

            // 调用 CBLAS 函数计算向量点积的实部和虚部
            CBLAS_FUNC(cblas_cdotc_sub)((CBLAS_INT)n, ip1, is1b, ip2, is2b, tmp);
            sum[0] += (double)tmp[0];
            sum[1] += (double)tmp[1];

            // 根据步长更新指针位置，准备处理下一个块
            ip1 += chunk * is1;
            ip2 += chunk * is2;
            n -= chunk;
        }

        // 将计算结果存入输出数组中
        ((float *)op)[0] = (float)sum[0];
        ((float *)op)[1] = (float)sum[1];
    }
    // 如果步长不合法，则使用简单的循环计算向量点积
    else
#endif
    {
        // 初始化实部和虚部的和
        float sumr = (float)0.0;
        float sumi = (float)0.0;
        npy_intp i;

        // 循环计算向量点积的实部和虚部
        for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
            // 提取每个向量元素的实部和虚部
            const float ip1r = ((float *)ip1)[0];
            const float ip1i = ((float *)ip1)[1];
            const float ip2r = ((float *)ip2)[0];
            const float ip2i = ((float *)ip2)[1];

            // 计算向量点积的实部和虚部
            sumr += ip1r * ip2r + ip1i * ip2i;
            sumi += ip1r * ip2i - ip1i * ip2r;
        }

        // 将计算结果存入输出数组中
        ((float *)op)[0] = sumr;
        ((float *)op)[1] = sumi;
    }
}


/*
 * 所有数据假定是对齐的。
 * 计算双精度复数向量点积的函数
 */
NPY_NO_EXPORT void
CDOUBLE_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
             char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
#if defined(HAVE_CBLAS)
    // 计算每个输入向量的步长，确保对齐
    CBLAS_INT is1b = blas_stride(is1, sizeof(npy_cdouble));
    CBLAS_INT is2b = blas_stride(is2, sizeof(npy_cdouble));

    // 如果步长均合法，则使用 CBLAS 函数计算点积
    if (is1b && is2b) {
        // 使用双精度浮点数数组保持计算稳定性
        double sum[2] = {0., 0.};

        // 循环计算向量点积，使用块处理以提高效率
        while (n > 0) {
            // 确定当前处理的块大小
            CBLAS_INT chunk = n < NPY_CBLAS_CHUNK ? n : NPY_CBLAS_CHUNK;
            double tmp[2];

            // 调用 CBLAS 函数计算向量点积的实部和虚部
            CBLAS_FUNC(cblas_zdotc_sub)((CBLAS_INT)n, ip1, is1b, ip2, is2b, tmp);
            sum[0] += (double)tmp[0];
            sum[1] += (double)tmp[1];

            // 根据步长更新指针位置，准备处理下一个块
            ip1 += chunk * is1;
            ip2 += chunk * is2;
            n -= chunk;
        }

        // 将计算结果存入输出数组中
        ((double *)op)[0] = (double)sum[0];
        ((double *)op)[1] = (double)sum[1];
    }
    // 如果步长不合法，则使用简单的循环计算向量点积
    else
#endif
    {
        // 初始化实部和虚部的总和为0
        double sumr = (double)0.0;
        double sumi = (double)0.0;
        // 循环迭代器
        npy_intp i;
    
        // 遍历从0到n的范围
        for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
            // 提取ip1和ip2的实部和虚部
            const double ip1r = ((double *)ip1)[0];
            const double ip1i = ((double *)ip1)[1];
            const double ip2r = ((double *)ip2)[0];
            const double ip2i = ((double *)ip2)[1];
    
            // 计算实部和虚部的加权和
            sumr += ip1r * ip2r + ip1i * ip2i;
            // 计算实部和虚部的差值
            sumi += ip1r * ip2i - ip1i * ip2r;
        }
        // 将累积的实部和虚部写入op的内存位置
        ((double *)op)[0] = sumr;
        ((double *)op)[1] = sumi;
    }
/*
 * 该函数计算复数向量的内积，并将结果存储在输出数组中。
 * 输入参数 ip1 和 ip2 是输入数组的指针，is1 和 is2 是它们的步长。
 * 输出参数 op 是输出数组的指针，n 是数组的长度。
 * ignore 参数未使用。
 */
NPY_NO_EXPORT void
CLONGDOUBLE_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
                 char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    // 初始化实部和虚部的临时变量为0
    npy_longdouble tmpr = 0.0L;
    npy_longdouble tmpi = 0.0L;
    npy_intp i;

    // 循环计算复数向量的内积
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        // 从输入数组中读取复数的实部和虚部
        const npy_longdouble ip1r = ((npy_longdouble *)ip1)[0];
        const npy_longdouble ip1i = ((npy_longdouble *)ip1)[1];
        const npy_longdouble ip2r = ((npy_longdouble *)ip2)[0];
        const npy_longdouble ip2i = ((npy_longdouble *)ip2)[1];

        // 计算复数向量的实部和虚部的内积
        tmpr += ip1r * ip2r + ip1i * ip2i;
        tmpi += ip1r * ip2i - ip1i * ip2r;
    }
    // 将计算结果存储到输出数组中
    ((npy_longdouble *)op)[0] = tmpr;
    ((npy_longdouble *)op)[1] = tmpi;
}

/*
 * 该函数计算对象数组的点积（dot product），并将结果存储在输出对象中。
 * 输入参数 ip1 和 ip2 是输入对象数组的指针，is1 和 is2 是它们的步长。
 * 输出参数 op 是输出对象的指针，n 是对象数组的长度。
 * ignore 参数未使用。
 */
NPY_NO_EXPORT void
OBJECT_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2, char *op, npy_intp n,
            void *NPY_UNUSED(ignore))
{
    npy_intp i;
    PyObject *tmp0, *tmp1, *tmp2, *tmp = NULL;
    PyObject **tmp3;

    // 循环计算对象数组的点积
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        // 检查输入对象是否为空
        if ((*((PyObject **)ip1) == NULL) || (*((PyObject **)ip2) == NULL)) {
            tmp1 = Py_False;
            Py_INCREF(Py_False);  // 增加 Py_False 的引用计数
        }
        else {
            // 调用第一个对象的 conjugate 方法
            tmp0 = PyObject_CallMethod(*((PyObject **)ip1), "conjugate", NULL);
            if (tmp0 == NULL) {
                Py_XDECREF(tmp);  // 减少临时对象的引用计数
                return;
            }
            // 计算两个对象的乘积
            tmp1 = PyNumber_Multiply(tmp0, *((PyObject **)ip2));
            Py_DECREF(tmp0);  // 减少临时对象的引用计数
            if (tmp1 == NULL) {
                Py_XDECREF(tmp);  // 减少临时对象的引用计数
                return;
            }
        }
        // 更新临时对象的引用
        if (i == 0) {
            tmp = tmp1;
        }
        else {
            tmp2 = PyNumber_Add(tmp, tmp1);
            Py_XDECREF(tmp);  // 减少临时对象的引用计数
            Py_XDECREF(tmp1);  // 减少临时对象的引用计数
            if (tmp2 == NULL) {
                return;
            }
            tmp = tmp2;
        }
    }
    // 将最终的临时对象赋值给输出对象
    tmp3 = (PyObject**) op;
    tmp2 = *tmp3;
    *((PyObject **)op) = tmp;
    Py_XDECREF(tmp2);  // 减少临时对象的引用计数
}
```