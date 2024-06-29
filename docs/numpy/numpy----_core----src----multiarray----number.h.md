# `.\numpy\numpy\_core\src\multiarray\number.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_
// 如果未定义 NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_ 宏，则开始条件编译防护
#define NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_

// 定义 NumericOps 结构体，用于存储各种数值操作的函数指针
typedef struct {
    PyObject *add;               // 加法
    PyObject *subtract;          // 减法
    PyObject *multiply;          // 乘法
    PyObject *divide;            // 除法
    PyObject *remainder;         // 求余
    PyObject *divmod;            // 返回除法结果和余数
    PyObject *power;             // 指数运算
    PyObject *square;            // 平方
    PyObject *reciprocal;        // 倒数
    PyObject *_ones_like;        // 类似 ones 的操作
    PyObject *sqrt;              // 平方根
    PyObject *cbrt;              // 立方根
    PyObject *negative;          // 取负
    PyObject *positive;          // 取正
    PyObject *absolute;          // 绝对值
    PyObject *invert;            // 按位取反
    PyObject *left_shift;        // 左移
    PyObject *right_shift;       // 右移
    PyObject *bitwise_and;       // 按位与
    PyObject *bitwise_xor;       // 按位异或
    PyObject *bitwise_or;        // 按位或
    PyObject *less;              // 小于比较
    PyObject *less_equal;        // 小于等于比较
    PyObject *equal;             // 等于比较
    PyObject *not_equal;         // 不等于比较
    PyObject *greater;           // 大于比较
    PyObject *greater_equal;     // 大于等于比较
    PyObject *floor_divide;      // 地板除
    PyObject *true_divide;       // 真除
    PyObject *logical_or;        // 逻辑或
    PyObject *logical_and;       // 逻辑与
    PyObject *floor;             // 向下取整
    PyObject *ceil;              // 向上取整
    PyObject *maximum;           // 最大值
    PyObject *minimum;           // 最小值
    PyObject *rint;              // 四舍五入到最接近的整数
    PyObject *conjugate;         // 复数的共轭
    PyObject *matmul;            // 矩阵乘法
    PyObject *clip;              // 裁剪
} NumericOps;

// 声明外部的 NumericOps 结构体变量 n_ops
extern NPY_NO_EXPORT NumericOps n_ops;
// 声明外部的 PyNumberMethods 结构体变量 array_as_number
extern NPY_NO_EXPORT PyNumberMethods array_as_number;

// 声明 PyArrayObject 到 PyObject 的类型转换函数
NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v);

// 声明设置 NumericOps 的函数，参数为一个 PyObject 字典
NPY_NO_EXPORT int
_PyArray_SetNumericOps(PyObject *dict);

// 声明通用的二元操作函数，参数为两个 PyObject 和一个操作符对象
NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyObject *m1, PyObject *m2, PyObject *op);

// 声明通用的一元操作函数，参数为一个 PyArrayObject 和一个操作符对象
NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

// 声明通用的约简操作函数，参数包括一个 PyArrayObject，一个操作符对象，一个轴，一个返回类型和一个输出数组对象
NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

// 声明通用的累积操作函数，参数包括一个 PyArrayObject，一个操作符对象，一个轴，一个返回类型和一个输出数组对象
NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_ */
// 结束条件编译防护，确保只有在未定义时才会包含该文件
```