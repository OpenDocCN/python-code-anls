# `.\numpy\numpy\_core\src\multiarray\arrayobject.h`

```py
#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_

#ifdef __cplusplus
extern "C" {
#endif

// 声明一个函数 _strings_richcompare，它接受两个 PyArrayObject 对象和两个整数参数，返回一个 PyObject 对象
NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip);

// 声明一个函数 array_richcompare，它接受一个 PyArrayObject 对象和一个 PyObject 对象以及一个整数参数，返回一个 PyObject 对象
NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op);

// 声明一个函数 array_might_be_written，它接受一个 PyArrayObject 对象，返回一个整数值
NPY_NO_EXPORT int
array_might_be_written(PyArrayObject *obj);

/*
 * 定义一个常量 NPY_ARRAY_WARN_ON_WRITE，表示一个标志位，用于标记我们希望在将来转换为视图的数组。
 * 当第一次尝试写入数组时会发出警告（但允许写入成功）。
 * 这个标志仅供内部使用，可能会在将来的版本中移除，因此不会暴露给用户代码。
 */
static const int NPY_ARRAY_WARN_ON_WRITE = (1 << 31);

/*
 * 下面三个标志用于内部表示曾经是 Python 标量（int, float, complex）的数组。
 * 这些标志只能在本地上下文中使用，当数组不被返回时。
 * 使用三个标志是为了避免在使用标志时需要双重检查实际的 dtype。
 */
static const int NPY_ARRAY_WAS_PYTHON_INT = (1 << 30);
static const int NPY_ARRAY_WAS_PYTHON_FLOAT = (1 << 29);
static const int NPY_ARRAY_WAS_PYTHON_COMPLEX = (1 << 28);

/*
 * 标记曾经是一个巨大整数，后来转换为对象数组（或无符号/非默认整数数组），但随后由临时数组替换。
 * 这个标志仅在 ufunc 机制中使用，其中正确覆盖所有类型解析路径是棘手的。
 */
static const int NPY_ARRAY_WAS_INT_AND_REPLACED = (1 << 27);
static const int NPY_ARRAY_WAS_PYTHON_LITERAL = (1 << 30 | 1 << 29 | 1 << 28);

/*
 * 此标志允许相同类型的强制转换，类似于 NPY_ARRAY_FORCECAST。
 * 数组从不设置此标志；它们仅用作各种 FromAny 函数的参数标志。
 */
static const int NPY_ARRAY_SAME_KIND_CASTING = (1 << 26);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_ */
```