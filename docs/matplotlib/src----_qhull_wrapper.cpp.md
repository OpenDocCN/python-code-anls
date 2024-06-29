# `D:\src\scipysrc\matplotlib\src\_qhull_wrapper.cpp`

```py
/*
 * Wrapper module for libqhull, providing Delaunay triangulation.
 *
 * This module's methods should not be accessed directly.  To obtain a Delaunay
 * triangulation, construct an instance of the matplotlib.tri.Triangulation
 * class without specifying a triangles array.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef _MSC_VER
/* The Qhull header does not declare this as extern "C", but only MSVC seems to
 * do name mangling on global variables. We thus need to declare this before
 * the header so that it treats it correctly, and doesn't mangle the name. */
extern "C" {
extern const char qh_version[];
}
#endif

#include "libqhull_r/qhull_ra.h"
#include <cstdio>
#include <vector>

#ifndef MPL_DEVNULL
#error "MPL_DEVNULL must be defined as the OS-equivalent of /dev/null"
#endif

#define STRINGIFY(x) STR(x)
#define STR(x) #x

namespace py = pybind11;
using namespace pybind11::literals;

// Input numpy array class.
typedef py::array_t<double, py::array::c_style | py::array::forcecast> CoordArray;

// Output numpy array class.
typedef py::array_t<int> IndexArray;

static const char* qhull_error_msg[6] = {
    "",                     /* 0 = qh_ERRnone */
    "input inconsistency",  /* 1 = qh_ERRinput */
    "singular input data",  /* 2 = qh_ERRsingular */
    "precision error",      /* 3 = qh_ERRprec */
    "insufficient memory",  /* 4 = qh_ERRmem */
    "internal error"};      /* 5 = qh_ERRqhull */

/* Return the indices of the 3 vertices that comprise the specified facet (i.e.
 * triangle). */
static void
get_facet_vertices(qhT* qh, const facetT* facet, int indices[3])
{
    vertexT *vertex, **vertexp;
    FOREACHvertex_(facet->vertices) {
        *indices++ = qh_pointid(qh, vertex->point);
    }
}

/* Return the indices of the 3 triangles that are neighbors of the specified
 * facet (triangle). */
static void
get_facet_neighbours(const facetT* facet, std::vector<int>& tri_indices,
                     int indices[3])
{
    facetT *neighbor, **neighborp;
    FOREACHneighbor_(facet) {
        *indices++ = (neighbor->upperdelaunay ? -1 : tri_indices[neighbor->id]);
    }
}

/* Return true if the specified points arrays contain at least 3 unique points,
 * or false otherwise. */
static bool
at_least_3_unique_points(py::ssize_t npoints, const double* x, const double* y)
{
    const py::ssize_t unique1 = 0;  /* First unique point has index 0. */
    py::ssize_t unique2 = 0;        /* Second unique point index is 0 until set. */

    if (npoints < 3) {
        return false;
    }
    // 循环遍历点集，从第二个点开始
    for (py::ssize_t i = 1; i < npoints; ++i) {
        // 如果还未找到第二个唯一点
        if (unique2 == 0) {
            /* 正在寻找第二个唯一点 */
            // 如果当前点与第一个唯一点不同
            if (x[i] != x[unique1] || y[i] != y[unique1]) {
                // 将当前点标记为第二个唯一点
                unique2 = i;
            }
        }
        else {
            /* 正在寻找第三个唯一点 */
            // 如果当前点既不是第一个唯一点，也不是第二个唯一点
            if ( (x[i] != x[unique1] || y[i] != y[unique1]) &&
                 (x[i] != x[unique2] || y[i] != y[unique2]) ) {
                /* 找到三个唯一点，它们的索引为 0、unique2 和 i */
                // 返回 true，表示找到了三个唯一点
                return true;
            }
        }
    }

    /* 在点集中未找到三个唯一点 */
    // 返回 false，表示未找到三个唯一点
    return false;
}

/* Holds on to info from Qhull so that it can be destructed automatically. */
class QhullInfo {
public:
    QhullInfo(FILE *error_file, qhT* qh) {
        this->error_file = error_file;
        this->qh = qh;
    }

    ~QhullInfo() {
        // 释放 Qhull 实例所占用的内存
        qh_freeqhull(this->qh, !qh_ALL);
        int curlong, totlong;  /* Memory remaining. */
        // 释放 Qhull 实例的内存池
        qh_memfreeshort(this->qh, &curlong, &totlong);
        // 如果有剩余内存，则发出运行时警告
        if (curlong || totlong) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Qhull could not free all allocated memory", 1);
        }

        // 如果错误文件不是 stderr，则关闭它
        if (this->error_file != stderr) {
            fclose(error_file);
        }
    }

private:
    FILE* error_file;  // 错误信息输出文件
    qhT* qh;           // Qhull 对象
};

/* Delaunay implementation method.
 * If hide_qhull_errors is true then qhull error messages are discarded;
 * if it is false then they are written to stderr. */
static py::tuple
delaunay_impl(py::ssize_t npoints, const double* x, const double* y,
              bool hide_qhull_errors)
{
    qhT qh_qh;                  /* qh variable type and name must be like */
    qhT* qh = &qh_qh;           /* this for Qhull macros to work correctly. */
    facetT* facet;
    int i, ntri, max_facet_id;
    int exitcode;               /* Value returned from qh_new_qhull(). */
    const int ndim = 2;
    double x_mean = 0.0;
    double y_mean = 0.0;

    QHULL_LIB_CHECK

    /* Allocate points. */
    std::vector<coordT> points(npoints * ndim);

    /* Determine mean x, y coordinates. */
    for (i = 0; i < npoints; ++i) {
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= npoints;
    y_mean /= npoints;

    /* Prepare points array to pass to qhull. */
    for (i = 0; i < npoints; ++i) {
        points[2*i  ] = x[i] - x_mean;
        points[2*i+1] = y[i] - y_mean;
    }

    /* qhull expects a FILE* to write errors to. */
    FILE* error_file = NULL;
    if (hide_qhull_errors) {
        /* qhull errors are ignored by writing to OS-equivalent of /dev/null.
         * Rather than have OS-specific code here, instead it is determined by
         * meson.build and passed in via the macro MPL_DEVNULL. */
        // 打开用于忽略错误的文件流（类似 /dev/null）
        error_file = fopen(STRINGIFY(MPL_DEVNULL), "w");
        if (error_file == NULL) {
            throw std::runtime_error("Could not open devnull");
        }
    }
    else {
        /* qhull errors written to stderr. */
        // 将错误信息写入 stderr
        error_file = stderr;
    }

    /* Perform Delaunay triangulation. */
    QhullInfo info(error_file, qh);
    // 初始化 Qhull 实例，准备进行 Delaunay 三角剖分
    qh_zero(qh, error_file);
    // 调用 Qhull 进行三角剖分计算
    exitcode = qh_new_qhull(qh, ndim, (int)npoints, points.data(), False,
                            (char*)"qhull d Qt Qbb Qc Qz", NULL, error_file);
    // 如果 exitcode 不为 qh_ERRnone，则抛出运行时错误，显示 qhull Delaunay 三角剖分计算错误信息
    if (exitcode != qh_ERRnone) {
        // 构造错误消息字符串
        std::string msg =
            py::str("Error in qhull Delaunay triangulation calculation: {} (exitcode={})")
            .format(qhull_error_msg[exitcode], exitcode).cast<std::string>();
        // 如果隐藏了 qhull 错误，追加提示信息
        if (hide_qhull_errors) {
            msg += "; use python verbose option (-v) to see original qhull error.";
        }
        // 抛出运行时异常，显示详细错误消息
        throw std::runtime_error(msg);
    }

    /* Split facets so that they only have 3 points each. */
    // 调用 qh_triangulate 对象 qh 进行面分割，使每个面只包含三个点

    /* Determine ntri and max_facet_id.
       Note that libqhull uses macros to iterate through collections. */
    // 初始化 ntri 为 0，遍历所有面并计数非上层 Delaunay 面，确定最大面 id

    /* Create array to map facet id to triangle index. */
    // 创建用于将面 id 映射到三角形索引的数组

    /* Allocate Python arrays to return. */
    // 分配用于返回的 Python 数组

    /* Determine triangles array and set tri_indices array. */
    // 遍历所有面并确定三角形数组及设置 tri_indices 数组

    /* Determine neighbors array. */
    // 遍历所有面并确定邻居数组

    // 返回由 triangles 和 neighbors 组成的 Python 元组
    return py::make_tuple(triangles, neighbors);
/* Process Python arguments and call Delaunay implementation method. */
static py::tuple
delaunay(const CoordArray& x, const CoordArray& y, int verbose)
{
    // 检查输入数组 x 和 y 是否为一维数组
    if (x.ndim() != 1 || y.ndim() != 1) {
        throw std::invalid_argument("x and y must be 1D arrays");
    }

    // 获取数组 x 和 y 的长度
    auto npoints = x.shape(0);
    // 检查 x 和 y 的长度是否相等
    if (npoints != y.shape(0)) {
        throw std::invalid_argument("x and y must be 1D arrays of the same length");
    }

    // 检查点的数量是否至少为 3
    if (npoints < 3) {
        throw std::invalid_argument("x and y arrays must have a length of at least 3");
    }

    // 检查数组 x 和 y 中是否至少包含三个唯一的点
    if (!at_least_3_unique_points(npoints, x.data(), y.data())) {
        throw std::invalid_argument("x and y arrays must consist of at least 3 unique points");
    }

    // 调用 Delaunay 三角化的实现函数并返回结果
    return delaunay_impl(npoints, x.data(), y.data(), verbose == 0);
}

/* Define Python bindings for the _qhull module. */
PYBIND11_MODULE(_qhull, m) {
    // 设置模块的文档字符串
    m.doc() = "Computing Delaunay triangulations.\n";

    // 定义 Python 函数 delaunay 的绑定
    m.def("delaunay", &delaunay, "x"_a, "y"_a, "verbose"_a,
        "--\n\n"
        "Compute a Delaunay triangulation.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "x, y : 1d arrays\n"
        "    The coordinates of the point set, which must consist of at least\n"
        "    three unique points.\n"
        "verbose : int\n"
        "    Python's verbosity level.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "triangles, neighbors : int arrays, shape (ntri, 3)\n"
        "    Indices of triangle vertices and indices of triangle neighbors.\n");

    // 定义 Python 函数 version 的绑定
    m.def("version", []() { return qh_version; },
        "version()\n--\n\n"
        "Return the qhull version string.");
}
```