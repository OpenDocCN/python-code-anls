# `.\numpy\numpy\_core\src\common\binop_override.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_
#define NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_

#include <string.h>  // 引入字符串操作相关的头文件
#include <Python.h>  // 引入 Python C API 的头文件
#include "numpy/arrayobject.h"  // 引入 NumPy 数组对象的头文件

#include "get_attr_string.h"  // 引入获取属性字符串的自定义头文件
#include "npy_static_data.h"  // 引入 NumPy 静态数据的头文件

static int
binop_should_defer(PyObject *self, PyObject *other, int inplace)
{
    /*
     * This function assumes that self.__binop__(other) is underway and
     * implements the rules described above. Python's C API is funny, and
     * makes it tricky to tell whether a given slot is called for __binop__
     * ("forward") or __rbinop__ ("reversed"). You are responsible for
     * determining this before calling this function; it only provides the
     * logic for forward binop implementations.
     */
    
    /*
     * NB: there's another copy of this code in
     *    numpy.ma.core.MaskedArray._delegate_binop
     * which should possibly be updated when this is.
     */
    
    PyObject *attr;  // 定义 PyObject 类型的变量 attr
    double self_prio, other_prio;  // 定义双精度浮点数变量 self_prio 和 other_prio
    int defer;  // 定义整型变量 defer

    /*
     * attribute check is expensive for scalar operations, avoid if possible
     */
    if (other == NULL ||  // 如果 other 为 NULL，直接返回 0
        self == NULL ||  // 如果 self 为 NULL，直接返回 0
        Py_TYPE(self) == Py_TYPE(other) ||  // 如果 self 和 other 类型相同，直接返回 0
        PyArray_CheckExact(other) ||  // 如果 other 是精确匹配的 NumPy 数组，直接返回 0
        PyArray_CheckAnyScalarExact(other)) {  // 如果 other 是任意标量的精确匹配，直接返回 0
        return 0;
    }

    /*
     * Classes with __array_ufunc__ are living in the future, and only need to
     * check whether __array_ufunc__ equals None.
     */
    attr = PyArray_LookupSpecial(other, npy_interned_str.array_ufunc);  // 查找 other 对象中的 __array_ufunc__ 属性
    if (attr != NULL) {  // 如果找到了 __array_ufunc__ 属性
        defer = !inplace && (attr == Py_None);  // 如果不是原地操作且 __array_ufunc__ 为 None，则推迟执行
        Py_DECREF(attr);  // 减少引用计数
        return defer;  // 返回 defer 值
    }
    else if (PyErr_Occurred()) {  // 如果出现了错误
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
        // 清除错误信息，暂时保留这个 TODO 注释
    }

    /*
     * Otherwise, we need to check for the legacy __array_priority__. But if
     * other.__class__ is a subtype of self.__class__, then it's already had
     * a chance to run, so no need to defer to it.
     */
    if (PyType_IsSubtype(Py_TYPE(other), Py_TYPE(self))) {  // 如果 other 是 self 的子类
        return 0;  // 直接返回 0，不推迟执行
    }

    self_prio = PyArray_GetPriority((PyObject *)self, NPY_SCALAR_PRIORITY);  // 获取 self 的优先级
    other_prio = PyArray_GetPriority((PyObject *)other, NPY_SCALAR_PRIORITY);  // 获取 other 的优先级
    return self_prio < other_prio;  // 返回比较结果，判断是否推迟执行
}

#endif  // NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_
/*
 * 定义宏BINOP_IS_FORWARD，用于检查是否是正向二元操作。
 * 如果 m2 的类型有定义 tp_as_number 并且其 SLOT_NAME 的函数指针不等于 test_func，
 * 则返回真，表示这是正向操作。
 */
#define BINOP_IS_FORWARD(m1, m2, SLOT_NAME, test_func)  \
    (Py_TYPE(m2)->tp_as_number != NULL &&                               \
     (void*)(Py_TYPE(m2)->tp_as_number->SLOT_NAME) != (void*)(test_func))

/*
 * 定义宏BINOP_GIVE_UP_IF_NEEDED，用于在需要时放弃二元操作的执行。
 * 如果是正向操作并且 binop_should_defer((PyObject*)m1, (PyObject*)m2, 0) 返回真，
 * 则增加 Py_NotImplemented 的引用计数并返回 Py_NotImplemented。
 */
#define BINOP_GIVE_UP_IF_NEEDED(m1, m2, slot_expr, test_func)           \
    do {                                                                \
        if (BINOP_IS_FORWARD(m1, m2, slot_expr, test_func) &&           \
            binop_should_defer((PyObject*)m1, (PyObject*)m2, 0)) {      \
            Py_INCREF(Py_NotImplemented);                               \
            return Py_NotImplemented;                                   \
        }                                                               \
    } while (0)

/*
 * 定义宏INPLACE_GIVE_UP_IF_NEEDED，用于在需要时放弃增强赋值操作的执行。
 * 如果是正向操作并且 binop_should_defer((PyObject*)m1, (PyObject*)m2, 1) 返回真，
 * 则增加 Py_NotImplemented 的引用计数并返回 Py_NotImplemented。
 */
#define INPLACE_GIVE_UP_IF_NEEDED(m1, m2, slot_expr, test_func)         \
    do {                                                                \
        if (BINOP_IS_FORWARD(m1, m2, slot_expr, test_func) &&           \
            binop_should_defer((PyObject*)m1, (PyObject*)m2, 1)) {      \
            Py_INCREF(Py_NotImplemented);                               \
            return Py_NotImplemented;                                   \
        }                                                               \
    } while (0)

/*
 * 定义宏RICHCMP_GIVE_UP_IF_NEEDED，用于在需要时放弃富比较操作的执行。
 * 如果 binop_should_defer((PyObject*)m1, (PyObject*)m2, 0) 返回真，
 * 则增加 Py_NotImplemented 的引用计数并返回 Py_NotImplemented。
 */
#define RICHCMP_GIVE_UP_IF_NEEDED(m1, m2)                               \
    do {                                                                \
        if (binop_should_defer((PyObject*)m1, (PyObject*)m2, 0)) {      \
            Py_INCREF(Py_NotImplemented);                               \
            return Py_NotImplemented;                                   \
        }                                                               \
    } while (0)



// 使用 do-while 循环语句，这种循环至少会执行一次，因为条件判断放在循环体末尾
#endif  /* NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_ */



#endif  /* NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_ */



# endif  /* NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_ */



# endif  /* NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_ */



# endif  /* NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_ */



# endif  /* NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_ */
```