# `D:\src\scipysrc\scipy\scipy\optimize\_lsap.c`

```
/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "rectangular_lsap/rectangular_lsap.h"

// 定义静态函数 linear_sum_assignment，用于解决线性分配问题
static PyObject*
linear_sum_assignment(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* a = NULL;  // 用于存放分配结果的数组 a
    PyObject* b = NULL;  // 用于存放分配结果的数组 b
    PyObject* result = NULL;  // 结果对象（暂时未使用）
    PyObject* obj_cost = NULL;  // 成本矩阵的 Python 对象
    int maximize = 0;  // 是否最大化分配成本标志，默认为 0，即最小化成本
    static const char *kwlist[] = {    (const char*)"cost_matrix",
                                    (const char*)"maximize",
                                    NULL};
    
    // 解析函数参数，获取成本矩阵和最大化标志
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", (char**)kwlist,
                                     &obj_cost, &maximize))
        return NULL;

    // 将 obj_cost 转换为连续的双精度浮点型 NumPy 数组
    PyArrayObject* obj_cont =
      (PyArrayObject*)PyArray_ContiguousFromAny(obj_cost, NPY_DOUBLE, 0, 0);
    if (!obj_cont) {
        return NULL;
    }

    // 检查数组维度是否为 2，否则抛出错误
    if (PyArray_NDIM(obj_cont) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "expected a matrix (2-D array), got a %d array",
                     PyArray_NDIM(obj_cont));
        goto cleanup;
    }

    // 获取指向成本矩阵数据的指针
    double* cost_matrix = (double*)PyArray_DATA(obj_cont);
    if (cost_matrix == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid cost matrix object");
        goto cleanup;
    }

    // 获取成本矩阵的行数和列数
    npy_intp num_rows = PyArray_DIM(obj_cont, 0);
    npy_intp num_cols = PyArray_DIM(obj_cont, 1);
    npy_intp dim[1] = { num_rows < num_cols ? num_rows : num_cols };

    // 创建用于存放分配结果的整型 NumPy 数组 a
    a = PyArray_SimpleNew(1, dim, NPY_INT64);
    if (!a)
        goto cleanup;

    // 创建用于存放分配结果的整型 NumPy 数组 b
    b = PyArray_SimpleNew(1, dim, NPY_INT64);
    // 如果 b 为假值，则跳转到清理代码的标签 cleanup
    if (!b)
        goto cleanup;

    // 获取数组对象 a 的数据指针，并将其强制转换为 int64_t 类型的指针
    int64_t* a_ptr = PyArray_DATA((PyArrayObject*)a);
    // 获取数组对象 b 的数据指针，并将其强制转换为 int64_t 类型的指针
    int64_t* b_ptr = PyArray_DATA((PyArrayObject*)b);
    // 定义整型变量 ret，用于存储函数 solve_rectangular_linear_sum_assignment 的返回值
    int ret;

    // 开始 Python GIL 线程安全区域
    Py_BEGIN_ALLOW_THREADS

    // 调用 solve_rectangular_linear_sum_assignment 函数解决矩形线性和分配问题，返回结果存储在 ret 中
    ret = solve_rectangular_linear_sum_assignment(
      num_rows, num_cols, cost_matrix, maximize, a_ptr, b_ptr);

    // 结束 Python GIL 线程安全区域
    Py_END_ALLOW_THREADS

    // 如果 ret 表示矩形线性和分配问题无解
    if (ret == RECTANGULAR_LSAP_INFEASIBLE) {
        // 设置 Python 异常，表示成本矩阵无解
        PyErr_SetString(PyExc_ValueError, "cost matrix is infeasible");
        // 跳转到清理代码的标签 cleanup
        goto cleanup;
    }
    // 如果 ret 表示矩形线性和分配问题包含无效数值
    else if (ret == RECTANGULAR_LSAP_INVALID) {
        // 设置 Python 异常，表示矩阵包含无效数值
        PyErr_SetString(PyExc_ValueError,
                        "matrix contains invalid numeric entries");
        // 跳转到清理代码的标签 cleanup
        goto cleanup;
    }

    // 构建一个 Python 元组对象，包含数组对象 a 和 b
    result = Py_BuildValue("OO", a, b);
cleanup:
    // 释放对象引用，减少对象的引用计数
    Py_XDECREF((PyObject*)obj_cont);
    // 释放对象引用，减少对象的引用计数
    Py_XDECREF(a);
    // 释放对象引用，减少对象的引用计数
    Py_XDECREF(b);
    // 返回计算结果
    return result;
}

static PyMethodDef lsap_methods[] = {
    { "linear_sum_assignment",
      // 函数名 "linear_sum_assignment"，对应的C函数为linear_sum_assignment
      (PyCFunction)linear_sum_assignment,
      // 函数接受位置参数和关键字参数
      METH_VARARGS | METH_KEYWORDS,
      // 函数的文档字符串，解释了线性求和分配问题的解法和参数说明
      "Solve the linear sum assignment problem.\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "cost_matrix : array\n"
      "    The cost matrix of the bipartite graph.\n"
      "\n"
      "maximize : bool (default: False)\n"
      "    Calculates a maximum weight matching if true.\n"
      "\n"
      "Returns\n"
      "-------\n"
      "row_ind, col_ind : array\n"
      "    An array of row indices and one of corresponding column indices giving\n"
      "    the optimal assignment. The cost of the assignment can be computed\n"
      "    as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be\n"
      "    sorted; in the case of a square cost matrix they will be equal to\n"
      "    ``numpy.arange(cost_matrix.shape[0])``.\n"
      "\n"
      "See Also\n"
      "--------\n"
      "scipy.sparse.csgraph.min_weight_full_bipartite_matching : for sparse inputs\n"
      "\n"
      "Notes\n"
      "-----\n"
      "\n"
      "The linear sum assignment problem [1]_ is also known as minimum weight\n"
      "matching in bipartite graphs. A problem instance is described by a matrix\n"
      "C, where each C[i,j] is the cost of matching vertex i of the first partite\n"
      "set (a 'worker') and vertex j of the second set (a 'job'). The goal is to\n"
      "find a complete assignment of workers to jobs of minimal cost.\n"
      "\n"
      "Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is\n"
      "assigned to column j. Then the optimal assignment has cost\n"
      "\n"
      ".. math::\n"
      "    \\min \\sum_i \\sum_j C_{i,j} X_{i,j}\n"
      "\n"
      "where, in the case where the matrix X is square, each row is assigned to\n"
      "exactly one column, and each column to exactly one row.\n"
      "\n"
      "This function can also solve a generalization of the classic assignment\n"
      "problem where the cost matrix is rectangular. If it has more rows than\n"
      "columns, then not every row needs to be assigned to a column, and vice\n"
      "versa.\n"
      "\n"
      "This implementation is a modified Jonker-Volgenant algorithm with no\n"
      "initialization, described in ref. [2]_.\n"
      "\n"
      ".. versionadded:: 0.17.0\n"
      "\n"
      "References\n"
      "----------\n"
      "\n"
      ".. [1] https://en.wikipedia.org/wiki/Assignment_problem\n"
      "\n"
      ".. [2] DF Crouse. On implementing 2D rectangular assignment algorithms.\n"
      "       *IEEE Transactions on Aerospace and Electronic Systems*,\n"
      "       52(4):1679-1696, August 2016, :doi:`10.1109/TAES.2016.140952`\n"
      "\n"
      "Examples\n"
      "--------\n"
      ">>> import numpy as np\n"
      ">>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])\n"
      ">>> from scipy.optimize import linear_sum_assignment\n"
      ">>> row_ind, col_ind = linear_sum_assignment(cost)\n"
      ">>> col_ind\n"
      "array([1, 0, 2])\n"
      ">>> cost[row_ind, col_ind].sum()\n"
      "5\n"},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    // Python 模块定义头初始化
    PyModuleDef_HEAD_INIT,
    // 模块名 "_lsap"
    "_lsap",
    // 模块文档字符串，描述解决矩形线性求和分配问题
    "Solves the rectangular linear sum assignment.",
    // 模块状态 -1 表示全局模块
    -1,
    lsap_methods,  # 定义一个变量 lsap_methods，可能是用于存储某种方法或算法的信息
    NULL,  # 定义一个变量 NULL，通常表示空值或未定义的值
    NULL,  # 定义一个变量 NULL，通常表示空值或未定义的值
    NULL,  # 定义一个变量 NULL，通常表示空值或未定义的值
    NULL,  # 定义一个变量 NULL，通常表示空值或未定义的值
};

PyMODINIT_FUNC
PyInit__lsap(void)
{
    // 调用NumPy C API中的import_array函数，初始化NumPy数组支持
    import_array();
    // 使用PyModule_Create函数创建Python模块对象，使用预定义的moduledef结构体
    return PyModule_Create(&moduledef);
}
```