# `D:\src\scipysrc\matplotlib\src\py_adaptors.h`

```py
/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_ADAPTORS_H
#define MPL_PY_ADAPTORS_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This module contains a number of C++ classes that adapt Python data
 * structures to C++ and Agg-friendly interfaces.
 */

#include <Python.h>

#include "numpy/arrayobject.h"

#include "agg_basics.h"
#include "py_exceptions.h"

extern "C" {
int convert_path(PyObject *obj, void *pathp);
}

namespace mpl {

/************************************************************
 * mpl::PathIterator acts as a bridge between NumPy and Agg.  Given a
 * pair of NumPy arrays, vertices and codes, it iterates over
 * those vertices and codes, using the standard Agg vertex source
 * interface:
 *
 *     unsigned vertex(double* x, double* y)
 */
class PathIterator
{
    /* We hold references to the Python objects, not just the
       underlying data arrays, so that Python reference counting
       can work.
    */
    PyArrayObject *m_vertices;  // NumPy array object holding vertex data
    PyArrayObject *m_codes;     // NumPy array object holding code data

    unsigned m_iterator;        // Current iterator position
    unsigned m_total_vertices;  // Total number of vertices

    /* This class doesn't actually do any simplification, but we
       store the value here, since it is obtained from the Python
       object.
    */
    bool m_should_simplify;     // Flag indicating if simplification should be applied
    double m_simplify_threshold; // Threshold for simplification

  public:
    inline PathIterator()
        : m_vertices(NULL),
          m_codes(NULL),
          m_iterator(0),
          m_total_vertices(0),
          m_should_simplify(false),
          m_simplify_threshold(1.0 / 9.0)
    {
    }

    // Constructor initializing with vertices, codes, and simplification parameters
    inline PathIterator(PyObject *vertices,
                        PyObject *codes,
                        bool should_simplify,
                        double simplify_threshold)
        : m_vertices(NULL), m_codes(NULL), m_iterator(0)
    {
        // Attempt to initialize with given parameters; throw exception on failure
        if (!set(vertices, codes, should_simplify, simplify_threshold))
            throw mpl::exception();
    }

    // Constructor initializing with vertices and codes (defaults for simplification)
    inline PathIterator(PyObject *vertices, PyObject *codes)
        : m_vertices(NULL), m_codes(NULL), m_iterator(0)
    {
        // Attempt to initialize with defaults; throw exception on failure
        if (!set(vertices, codes))
            throw mpl::exception();
    }

    // Copy constructor
    inline PathIterator(const PathIterator &other)
    {
        // Increase reference counts of the Python objects to manage ownership
        Py_XINCREF(other.m_vertices);
        m_vertices = other.m_vertices;

        Py_XINCREF(other.m_codes);
        m_codes = other.m_codes;

        // Copy other member variables
        m_iterator = 0;
        m_total_vertices = other.m_total_vertices;
        m_should_simplify = other.m_should_simplify;
        m_simplify_threshold = other.m_simplify_threshold;
    }

    // Destructor
    ~PathIterator()
    {
        // Decrease reference counts to release Python objects properly
        Py_XDECREF(m_vertices);
        Py_XDECREF(m_codes);
    }

    // Setter method for initializing with vertices, codes, and simplification parameters
    inline int
    set(PyObject *vertices, PyObject *codes, bool should_simplify, double simplify_threshold)
    {
        // 设置是否应简化和简化阈值
        m_should_simplify = should_simplify;
        m_simplify_threshold = simplify_threshold;
    
        // 释放旧的顶点数据并根据新的Python对象创建新的顶点数组
        Py_XDECREF(m_vertices);
        m_vertices = (PyArrayObject *)PyArray_FromObject(vertices, NPY_DOUBLE, 2, 2);
    
        // 检查顶点数组是否有效
        if (!m_vertices || PyArray_DIM(m_vertices, 1) != 2) {
            PyErr_SetString(PyExc_ValueError, "Invalid vertices array");
            return 0;
        }
    
        // 释放旧的代码数据（如果有），并根据新的Python对象创建新的代码数组
        Py_XDECREF(m_codes);
        m_codes = NULL;
    
        if (codes != NULL && codes != Py_None) {
            m_codes = (PyArrayObject *)PyArray_FromObject(codes, NPY_UINT8, 1, 1);
    
            // 检查代码数组是否有效
            if (!m_codes || PyArray_DIM(m_codes, 0) != PyArray_DIM(m_vertices, 0)) {
                PyErr_SetString(PyExc_ValueError, "Invalid codes array");
                return 0;
            }
        }
    
        // 设置总顶点数和迭代器初始值
        m_total_vertices = (unsigned)PyArray_DIM(m_vertices, 0);
        m_iterator = 0;
    
        // 返回成功标志
        return 1;
    }
    
    inline int set(PyObject *vertices, PyObject *codes)
    {
        // 调用带有默认参数的set函数
        return set(vertices, codes, false, 0.0);
    }
    
    inline unsigned vertex(double *x, double *y)
    {
        // 如果迭代器超过总顶点数，返回停止命令并设置默认坐标
        if (m_iterator >= m_total_vertices) {
            *x = 0.0;
            *y = 0.0;
            return agg::path_cmd_stop;
        }
    
        // 获取当前顶点对的指针，并设置x和y坐标
        const size_t idx = m_iterator++;
        char *pair = (char *)PyArray_GETPTR2(m_vertices, idx, 0);
        *x = *(double *)pair;
        *y = *(double *)(pair + PyArray_STRIDE(m_vertices, 1));
    
        // 如果有代码数组，返回当前索引处的代码；否则根据索引返回移动到或线段到命令
        if (m_codes != NULL) {
            return (unsigned)(*(char *)PyArray_GETPTR1(m_codes, idx));
        } else {
            return idx == 0 ? agg::path_cmd_move_to : agg::path_cmd_line_to;
        }
    }
    
    inline void rewind(unsigned path_id)
    {
        // 重置迭代器到指定的路径ID
        m_iterator = path_id;
    }
    
    inline unsigned total_vertices() const
    {
        // 返回总顶点数
        return m_total_vertices;
    }
    
    inline bool should_simplify() const
    {
        // 返回是否应简化的标志
        return m_should_simplify;
    }
    
    inline double simplify_threshold() const
    {
        // 返回简化阈值
        return m_simplify_threshold;
    }
    
    inline bool has_codes() const
    {
        // 返回是否存在代码数组的标志
        return m_codes != NULL;
    }
    
    inline void *get_id()
    {
        // 返回顶点数组的指针作为ID
        return (void *)m_vertices;
    }
};

// PathGenerator 类的定义
class PathGenerator
{
    // 成员变量声明
    PyObject *m_paths;     // 保存路径对象的 Python 对象指针
    Py_ssize_t m_npaths;   // 路径对象的数量

  public:
    // 定义路径迭代器类型
    typedef PathIterator path_iterator;

    // 构造函数，初始化成员变量
    PathGenerator() : m_paths(NULL), m_npaths(0) {}

    // 析构函数，释放成员变量 m_paths 的 Python 对象引用
    ~PathGenerator()
    {
        Py_XDECREF(m_paths);
    }

    // 设置路径对象的方法
    int set(PyObject *obj)
    {
        // 检查传入的对象是否为序列类型
        if (!PySequence_Check(obj)) {
            return 0; // 如果不是序列类型，返回 0 表示设置失败
        }

        // 释放原有 m_paths 的 Python 对象引用
        Py_XDECREF(m_paths);
        // 将传入的对象赋值给 m_paths
        m_paths = obj;
        // 增加 m_paths 的 Python 对象引用计数
        Py_INCREF(m_paths);

        // 获取 m_paths 中对象的数量并赋值给 m_npaths
        m_npaths = PySequence_Size(m_paths);

        return 1; // 设置成功，返回 1
    }

    // 返回路径对象数量的方法
    Py_ssize_t num_paths() const
    {
        return m_npaths; // 返回 m_npaths 的值
    }

    // 返回路径对象数量的方法（重载）
    Py_ssize_t size() const
    {
        return m_npaths; // 返回 m_npaths 的值
    }

    // () 运算符重载，返回指定索引位置的路径迭代器
    path_iterator operator()(size_t i)
    {
        path_iterator path; // 声明路径迭代器变量
        PyObject *item;

        // 获取 m_paths 中第 i % m_npaths 位置的对象
        item = PySequence_GetItem(m_paths, i % m_npaths);
        if (item == NULL) {
            throw mpl::exception(); // 获取失败，抛出异常
        }
        // 将获取的 Python 对象转换为路径迭代器
        if (!convert_path(item, &path)) {
            Py_DECREF(item);
            throw mpl::exception(); // 转换失败，抛出异常
        }
        Py_DECREF(item); // 释放 Python 对象的引用
        return path; // 返回路径迭代器
    }
};
}

#endif
```