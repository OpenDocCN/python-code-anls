# `D:\src\scipysrc\scipy\scipy\spatial\src\distance_pybind.cpp`

```
// 引入 pybind11 库，用于 Python 和 C++ 的互操作
#include <pybind11/pybind11.h>
// 引入 pybind11 的 numpy 支持
#include <pybind11/numpy.h>
// 引入 numpy 的数组对象头文件
#include <numpy/arrayobject.h>
// 引入数学计算库
#include <cmath>
// 引入断言库，用于运行时检查
#include <cassert>

// 引入自定义头文件
#include "function_ref.h"
#include "views.h"
#include "distance_metrics.h"

// 引入字符串流库
#include <sstream>
// 引入字符串库
#include <string>

// 命名空间别名
namespace py = pybind11;

// 匿名命名空间，内部实现细节
namespace {

// 定义模板，用于表示距离函数的引用
template <typename T>
using DistanceFunc = FunctionRef<
    void(StridedView2D<T>, StridedView2D<const T>, StridedView2D<const T>)>;

// 定义模板，用于表示加权距离函数的引用
template <typename T>
using WeightedDistanceFunc = FunctionRef<
    void(StridedView2D<T>, StridedView2D<const T>,
         StridedView2D<const T>, StridedView2D<const T>)>;

// 验证权重数组的值是否大于等于零
template <typename T>
void validate_weights(const ArrayDescriptor& w, const T* w_data) {
    // 定义索引数组
    intptr_t idx[NPY_MAXDIMS] = {0};
    // 如果权重数组维度超过最大允许维度，抛出异常
    if (w.ndim > NPY_MAXDIMS) {
        throw std::invalid_argument("Too many dimensions");
    }

    // 计算迭代次数
    intptr_t numiter = 1;
    for (intptr_t ax = 0; ax < w.ndim - 1; ++ax) {
        numiter *= w.shape[ax];
    }

    // 是否有效标志位
    bool is_valid = true;
    const T* row_ptr = w_data;
    const auto inner_size = w.shape[w.ndim - 1];
    const auto stride = w.strides[w.ndim - 1];

    // 迭代验证权重数组中的每个值
    while (is_valid && numiter > 0) {
        for (intptr_t i = 0; i < inner_size; ++i) {
            if (row_ptr[i * stride] < 0) {
                is_valid = false;
            }
        }

        // 更新索引并移动指针
        for (intptr_t ax = w.ndim - 2; ax >= 0; --ax) {
            if (idx[ax] + 1 < w.shape[ax]) {
                ++idx[ax];
                row_ptr += w.strides[ax];
                break;
            } else {
                row_ptr -= idx[ax] * w.strides[ax];
                idx[ax] = 0;
            }
        }
        --numiter;
    }

    // 如果发现无效值，抛出异常
    if (!is_valid) {
        throw std::invalid_argument("Input weights should be all non-negative");
    }
}

// 实现距离计算的函数模板
template <typename T>
void pdist_impl(ArrayDescriptor out, T* out_data,
                ArrayDescriptor x, const T* in_data,
                DistanceFunc<T> f) {
    // 获取输入数组的行列数
    const intptr_t num_rows = x.shape[0], num_cols = x.shape[1];

    // 定义输出视图
    StridedView2D<T> out_view;
    out_view.strides = {out.strides[0], 0};
    out_view.shape = {x.shape[0] - 1, x.shape[1]};
    out_view.data = out_data;

    // 定义输入视图
    StridedView2D<const T> x_view;
    x_view.strides = {x.strides[0], x.strides[1]};
    x_view.shape = {out_view.shape[0], num_cols};
    x_view.data = in_data + x.strides[0];

    // 第二个输入视图
    StridedView2D<const T> y_view;
    y_view.strides = {0, x.strides[1]};
    y_view.shape = {out_view.shape[0], num_cols};
    y_view.data = in_data;

    // 循环计算距离
    for (intptr_t i = 0; i < num_rows - 1; ++i) {
        f(out_view, x_view, y_view);

        // 更新输出视图和输入视图
        out_view.data += out_view.shape[0] * out_view.strides[0];
        out_view.shape[0] -= 1;
        x_view.shape[0] = y_view.shape[0] = out_view.shape[0];
        x_view.data += x.strides[0];
        y_view.data += x.strides[0];
    }
}

// 模板声明结束
template <typename T>
    // 初始化 x_view，设置其步长和形状，指向 x_data 的第二行起始位置
    x_view.strides = {0, x.strides[1]};
    x_view.shape = {num_rowsY, num_cols};

    // 初始化 y_view，设置其步长和形状，指向 y_data 的第一行起始位置
    y_view.strides = {y.strides[0], y.strides[1]};
    y_view.shape = {out_view.shape[0], num_cols};
    y_view.data = y_data;
    # 将 x 数据赋值给 x 视图的数据
    x_view.data = x_data;
    
    # 创建一个二维步进视图 y_view，并设置其步进和形状
    StridedView2D<const T> y_view;
    y_view.strides = {y.strides[0], y.strides[1]};  # 设置 y_view 的行和列步进
    y_view.shape = {num_rowsY, num_cols};           # 设置 y_view 的形状
    y_view.data = y_data;                          # 将 y 数据赋值给 y_view 的数据
    
    # 创建一个二维步进视图 w_view，并设置其步进和形状
    StridedView2D<const T> w_view;
    w_view.strides = {0, w.strides[0]};             # 设置 w_view 的行步进为 0，列步进为 w 的列步进
    w_view.shape = {num_rowsY, num_cols};           # 设置 w_view 的形状
    w_view.data = w_data;                          # 将 w 数据赋值给 w_view 的数据
    
    # 遍历 num_rowsX 行的数据
    for (intptr_t i = 0; i < num_rowsX; ++i) {
        # 调用函数 f，传入 out_view、x_view、y_view、w_view
        f(out_view, x_view, y_view, w_view);
    
        # 更新 out_view 的数据指针，使其指向下一行的数据
        out_view.data += out.strides[0];
    
        # 更新 x_view 的数据指针，使其指向下一行的数据
        x_view.data += x.strides[0];
    }
}

// 从 NumPy 数组中提取形状和步幅信息。将字节步幅转换为元素步幅，并在访问时避免额外的指针间接性。
ArrayDescriptor get_descriptor(const py::array& arr) {
    // 获取数组的维度
    const auto ndim = arr.ndim();
    // 创建 ArrayDescriptor 对象，指定维度数
    ArrayDescriptor desc(ndim);

    // 获取数组的形状并赋给 desc 对象
    const auto arr_shape = arr.shape();
    desc.shape.assign(arr_shape, arr_shape + ndim);

    // TODO: 用 `arr.itemsize()` 替换以下代码，这是一个临时的解决方案：
    // 获取数组元素的大小
    desc.element_size = PyArray_ITEMSIZE(reinterpret_cast<PyArrayObject *>(arr.ptr()));
    // 获取数组的步幅并赋给 desc 对象
    const auto arr_strides = arr.strides();
    desc.strides.assign(arr_strides, arr_strides + ndim);
    // 遍历每个维度
    for (intptr_t i = 0; i < ndim; ++i) {
        // 如果某维度的长度小于等于1，则将其步幅设为0，以符合 NumPy 的放松步幅检查规则
        if (arr_shape[i] <= 1) {
            desc.strides[i] = 0;
            continue;
        }

        // 如果当前维度的步幅不能整除元素大小，则抛出异常
        if (desc.strides[i] % desc.element_size != 0) {
            std::stringstream msg;
            msg << "Arrays must be aligned to element size, but found stride of ";
            msg << desc.strides[i] << " bytes for elements of size " << desc.element_size;
            throw std::runtime_error(msg.str());
        }
        // 将步幅转换为元素步幅
        desc.strides[i] /= desc.element_size;
    }
    // 返回 ArrayDescriptor 对象
    return desc;
}

// 将 Python 对象转换为指定数据类型 T 的 NumPy 数组。
// flags 可以是任何 NumPy 数组构造器的标志。
template <typename T>
py::array_t<T> npy_asarray(const py::handle& obj, int flags = 0) {
    // 获取数据类型描述符
    auto descr = reinterpret_cast<PyArray_Descr*>(
        py::dtype::of<T>().release().ptr());
    // 将 Python 对象转换为 NumPy 数组
    auto* arr = PyArray_FromAny(obj.ptr(), descr, 0, 0, flags, nullptr);
    // 如果转换失败，则抛出异常
    if (arr == nullptr) {
        throw py::error_already_set();
    }
    // 将 C 数组转换为 py::array_t<T> 类型并返回
    return py::reinterpret_steal<py::array_t<T>>(arr);
}

// 将 Python 对象转换为未指定数据类型的 NumPy 数组。
// flags 可以是任何 NumPy 数组构造器的标志。
py::array npy_asarray(const py::handle& obj, int flags = 0) {
    // 将 Python 对象转换为 NumPy 数组
    auto* arr = PyArray_FromAny(obj.ptr(), nullptr, 0, 0, flags, nullptr);
    // 如果转换失败，则抛出异常
    if (arr == nullptr) {
        throw py::error_already_set();
    }
    // 将 C 数组转换为 py::array 类型并返回
    return py::reinterpret_steal<py::array>(arr);
}

template <typename scalar_t>
py::array pdist_unweighted(const py::array& out_obj, const py::array& x_obj,
                           DistanceFunc<scalar_t> f) {
    // 将输入对象 x_obj 转换为 scalar_t 类型的 NumPy 数组 x
    auto x = npy_asarray<scalar_t>(x_obj,
                                   NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    // 将输出对象 out_obj 转换为 scalar_t 类型的 py::array_t 数组 out
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);
    // 获取输出数组的描述符
    auto out_desc = get_descriptor(out);
    // 获取输出数组的数据指针
    auto out_data = out.mutable_data();
    // 获取输入数组 x 的描述符
    auto x_desc = get_descriptor(x);
    // 获取输入数组 x 的数据指针
    auto x_data = x.data();
    {
        // 释放全局解释器锁，调用底层函数 pdist_impl 进行计算
        py::gil_scoped_release guard;
        pdist_impl(out_desc, out_data, x_desc, x_data, f);
    }
    // 返回移动语义的输出数组 out
    return std::move(out);
}

template <typename scalar_t>
py::array pdist_weighted(
        const py::array& out_obj, const py::array& x_obj,
        const py::array& w_obj, WeightedDistanceFunc<scalar_t> f) {
    # 将 Python 对象 x_obj 转换为 NumPy 数组，元素类型为 scalar_t，保证内存对齐且未交换字节序
    auto x = npy_asarray<scalar_t>(x_obj,
                                   NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    # 将 Python 对象 w_obj 转换为 NumPy 数组，元素类型为 scalar_t，保证内存对齐且未交换字节序
    auto w = npy_asarray<scalar_t>(w_obj,
                                   NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    # 将 Python 对象 out_obj 转换为 py::array_t<scalar_t> 类型的 NumPy 数组
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);
    # 获取输出数组 out 的描述符
    auto out_desc = get_descriptor(out);
    # 获取输出数组 out 的可变数据指针
    auto out_data = out.mutable_data();
    # 获取输入数组 x 的描述符
    auto x_desc = get_descriptor(x);
    # 获取输入数组 x 的数据指针
    auto x_data = x.data();
    # 获取权重数组 w 的描述符
    auto w_desc = get_descriptor(w);
    # 获取权重数组 w 的数据指针
    auto w_data = w.data();
    {
        # 释放全局解释器锁 GIL，允许多线程或异步操作
        py::gil_scoped_release guard;
        # 验证权重数组的描述符和数据是否有效
        validate_weights(w_desc, w_data);
        # 调用加权距离计算的实现函数，传入输出、输入和权重数组的描述符和数据，以及函数 f
        pdist_weighted_impl(
            out_desc, out_data, x_desc, x_data, w_desc, w_data, f);
    }
    # 返回移动语义转移后的输出数组 out
    return std::move(out);
}

template <typename scalar_t>
// 定义一个模板函数 cdist_unweighted，计算两组数据 x 和 y 之间的无权距离，并将结果写入输出数组 out
py::array cdist_unweighted(const py::array& out_obj, const py::array& x_obj,
                           const py::array& y_obj, DistanceFunc<scalar_t> f) {
    // 将输入数组 x 转换为 numpy 数组，要求内存对齐且不交换字节顺序
    auto x = npy_asarray<scalar_t>(x_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    // 将输入数组 y 转换为 numpy 数组，要求内存对齐且不交换字节顺序
    auto y = npy_asarray<scalar_t>(y_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    // 将输出数组 out 转换为 py::array_t 类型
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);

    // 获取输出数组的描述符和可变数据指针
    auto out_desc = get_descriptor(out);
    auto out_data = out.mutable_data();
    // 获取输入数组 x 的描述符和数据指针
    auto x_desc = get_descriptor(x);
    auto x_data = x.data();
    // 获取输入数组 y 的描述符和数据指针
    auto y_desc = get_descriptor(y);
    auto y_data = y.data();
    {
        // 释放 GIL，允许多线程执行，确保不会影响 Python 解释器的其他线程
        py::gil_scoped_release guard;
        // 调用 cdist_impl 函数计算距离，将结果写入 out 数组
        cdist_impl(out_desc, out_data, x_desc, x_data, y_desc, y_data, f);
    }
    // 返回结果数组 out
    return std::move(out);
}

template <typename scalar_t>
// 定义一个模板函数 cdist_weighted，计算两组数据 x 和 y 之间的加权距离，并将结果写入输出数组 out
py::array cdist_weighted(
        const py::array& out_obj, const py::array& x_obj,
        const py::array& y_obj, const py::array& w_obj,
        WeightedDistanceFunc<scalar_t> f) {
    // 将输入数组 x 转换为 numpy 数组，要求内存对齐且不交换字节顺序
    auto x = npy_asarray<scalar_t>(x_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    // 将输入数组 y 转换为 numpy 数组，要求内存对齐且不交换字节顺序
    auto y = npy_asarray<scalar_t>(y_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    // 将输入数组 w 转换为 numpy 数组，要求内存对齐且不交换字节顺序
    auto w = npy_asarray<scalar_t>(w_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    // 将输出数组 out 转换为 py::array_t 类型
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);

    // 获取输出数组的描述符和可变数据指针
    auto out_desc = get_descriptor(out);
    auto out_data = out.mutable_data();
    // 获取输入数组 x 的描述符和数据指针
    auto x_desc = get_descriptor(x);
    auto x_data = x.data();
    // 获取输入数组 y 的描述符和数据指针
    auto y_desc = get_descriptor(y);
    auto y_data = y.data();
    // 获取输入数组 w 的描述符和数据指针
    auto w_desc = get_descriptor(w);
    auto w_data = w.data();
    {
        // 释放 GIL，允许多线程执行，确保不会影响 Python 解释器的其他线程
        py::gil_scoped_release guard;
        // 验证权重数组 w，并调用 cdist_weighted_impl 函数计算加权距离，将结果写入 out 数组
        validate_weights(w_desc, w_data);
        cdist_weighted_impl(
            out_desc, out_data, x_desc, x_data, y_desc, y_data, w_desc, w_data, f);
    }
    // 返回结果数组 out
    return std::move(out);
}

// 返回两种数据类型的最小公共类型
py::dtype npy_promote_types(const py::dtype& type1, const py::dtype& type2) {
    // 调用 NumPy C API 函数，获取类型 type1 和 type2 的最小公共类型描述符
    PyArray_Descr* descr = PyArray_PromoteTypes(
        reinterpret_cast<PyArray_Descr*>(type1.ptr()),
        reinterpret_cast<PyArray_Descr*>(type2.ptr()));
    // 若无法获取描述符，抛出 Python 异常
    if (descr == nullptr) {
        throw py::error_already_set();
    }
    // 将 C API 返回的描述符转换为 py::dtype 类型，并返回
    return py::reinterpret_steal<py::dtype>(reinterpret_cast<PyObject*>(descr));
}

template <typename Container>
// 准备输出参数的辅助函数，根据输入的 obj、dtype 和 out_shape 创建输出数组
py::array prepare_out_argument(const py::object& obj, const py::dtype& dtype,
                               const Container& out_shape) {
    // 若 obj 为空，则创建一个新的 dtype 和形状为 out_shape 的数组并返回
    if (obj.is_none()) {
        return py::array(dtype, out_shape);
    }

    // 若 obj 不是 ndarray 类型，则抛出类型错误异常
    if (!py::isinstance<py::array>(obj)) {
        throw py::type_error("out argument must be an ndarray");
    }

    // 将 obj 转换为 py::array 类型
    py::array out = py::cast<py::array>(obj);
    // 获取数组的维度和形状
    const auto ndim = out.ndim();
    const auto shape = out.shape();
    // 将 out 转换为 PyArrayObject 指针
    auto pao = reinterpret_cast<PyArrayObject*>(out.ptr());
    # 检查输出数组的维度是否与期望的维度匹配，以及形状是否一致
    if (ndim != static_cast<intptr_t>(out_shape.size()) ||
        !std::equal(shape, shape + ndim, out_shape.begin())) {
        throw std::invalid_argument("Output array has incorrect shape.");
    }
    # 检查输出数组是否是 C 连续的
    if (!PyArray_ISCONTIGUOUS(pao)) {
        throw std::invalid_argument("Output array must be C-contiguous");
    }
    # 检查输出数组的数据类型是否与期望的数据类型相等
    if (out.dtype().not_equal(dtype)) {
        # 获取数据类型的字符串表示形式
        const py::handle& handle = dtype;
        throw std::invalid_argument("wrong out dtype, expected " +
                                    std::string(py::str(handle)));
    }
    # 检查输出数组是否符合预期的行为，即是否对齐、可写且本机字节顺序
    if (!PyArray_ISBEHAVED(pao)) {
        throw std::invalid_argument(
            "out array must be aligned, writable and native byte order");
    }
    # 返回处理后的输出数组
    return out;
// 定义函数 prepare_single_weight，准备单个权重数组
py::array prepare_single_weight(const py::object& obj, intptr_t len) {
    // 将 Python 对象转换为 NumPy 数组
    py::array weight = npy_asarray(obj);
    // 检查数组维度是否为 1
    if (weight.ndim() != 1) {
        // 若不是，则抛出异常
        throw std::invalid_argument("Weights must be a vector (ndim = 1)");
    } else if (weight.shape(0) != len) {
        // 检查数组长度是否与给定长度相同
        std::stringstream msg;
        msg << "Weights must have same size as input vector. ";
        msg << weight.shape(0) << " vs. " << len << ".";
        // 若长度不同，则抛出异常
        throw std::invalid_argument(msg.str());
    }
    // 返回准备好的权重数组
    return weight;
}

// 定义函数 common_type，返回输入 dtype
py::dtype common_type(py::dtype type) {
    return type;
}

// 使用模板定义函数 common_type，接受多个 dtype，返回它们的公共类型
template <typename... Args>
py::dtype common_type(const py::dtype& type1, const py::dtype& type2,
                      const Args&... tail) {
    // 递归调用 common_type，推导出所有 dtype 的公共类型
    return common_type(npy_promote_types(type1, type2), tail...);
}

// 定义函数 dtype_num，返回 NumPy dtype 对应的类型编号
int dtype_num(const py::dtype& dtype) {
    // 从 dtype 中获取其对应的类型编号
    return reinterpret_cast<const PyArray_Descr*>(
        dtype.ptr())->type_num;
}

// 定义函数 promote_type_real，根据输入 dtype 提升为实数类型
py::dtype promote_type_real(const py::dtype& dtype) {
    switch (dtype.kind()) {
    case 'b':
    case 'i':
    case 'u': {
        // 对于整数和布尔类型，提升为双精度浮点数
        return py::dtype::template of<double>();
    }
    case 'f': {
        if (dtype_num(dtype) == NPY_LONGDOUBLE) {
            // 如果是长双精度浮点数，保持原样
            return dtype;
        } else {
            // 否则，默认将浮点数类型提升为双精度浮点数
            // TODO: 允许输出 float32 类型
            return py::dtype::template of<double>();
        }
    }
    default: {
        // 其他类型保持原样返回
        return dtype;
    }
    }
}

// 定义宏 DISPATCH_DTYPE，用于从 NumPy dtype 分派表达式
#define DISPATCH_DTYPE(dtype, expression)                               \
    do {                                                                \
        const py::dtype& type_obj = dtype;                              \  # 声明一个常量引用type_obj，初始化为传入的dtype
        switch (dtype_num(type_obj)) {                                  \  # 调用dtype_num函数，根据type_obj返回一个枚举值
        case NPY_HALF:                                                  \  # 如果返回值为NPY_HALF，则执行以下代码块
        case NPY_FLOAT: /* TODO: Enable scalar_t=float dispatch */      \  # 如果返回值为NPY_FLOAT，暂时未启用，可能将来支持scalar_t=float的分发
        case NPY_DOUBLE: {                                              \  # 如果返回值为NPY_DOUBLE，则执行以下代码块
            using scalar_t = double;                                    \  # 定义一个类型别名scalar_t为double
            expression();                                               \  # 调用expression函数，处理当前数据类型为double的情况
            break;                                                      \  # 跳出switch语句
        }                                                               \
        case NPY_LONGDOUBLE: {                                          \  # 如果返回值为NPY_LONGDOUBLE，则执行以下代码块
            using scalar_t = long double;                               \  # 定义一个类型别名scalar_t为long double
            expression();                                               \  # 调用expression函数，处理当前数据类型为long double的情况
            break;                                                      \  # 跳出switch语句
        }                                                               \
        default: {                                                      \  # 如果返回值为其他枚举值，则执行以下代码块
            const py::handle& handle = type_obj;                        \  # 声明一个常量引用handle，初始化为type_obj
            throw std::invalid_argument(                                \  # 抛出一个invalid_argument异常
                "Unsupported dtype " + std::string(py::str(handle)));   \  # 异常信息包含不支持的dtype类型信息
        }                                                               \
        }                                                               \
    } while (0)                                                          \  # 执行完毕后退出do-while循环，条件始终为假，即执行一次
template <typename Func>
// 定义模板函数 pdist，用于计算距离矩阵，支持不同的距离函数
py::array pdist(const py::object& out_obj, const py::object& x_obj,
                const py::object& w_obj, Func&& f) {
    auto x = npy_asarray(x_obj);
    // 将 Python 对象 x 转换为 NumPy 数组 x
    if (x.ndim() != 2) {
        // 如果 x 不是二维数组，则抛出异常
        throw std::invalid_argument("x must be 2-dimensional");
    }

    const intptr_t m = x.shape(1);
    const intptr_t n = x.shape(0);
    std::array<intptr_t, 1> out_shape{{(n * (n - 1)) / 2}};
    // 计算输出数组的形状，以存储距离矩阵
    if (w_obj.is_none()) {
        // 如果权重对象 w_obj 为空
        auto dtype = promote_type_real(x.dtype());
        // 推断出适当的数据类型以存储结果
        auto out = prepare_out_argument(out_obj, dtype, out_shape);
        // 准备输出参数数组
        DISPATCH_DTYPE(dtype, [&]{
            // 根据数据类型分发到具体的距离计算函数
            pdist_unweighted<scalar_t>(out, x, f);
        });
        return out;
    }

    auto w = prepare_single_weight(w_obj, m);
    // 准备单个权重数组 w
    auto dtype = promote_type_real(common_type(x.dtype(), w.dtype()));
    // 推断出适当的数据类型以存储结果，考虑 x 和 w 的数据类型
    auto out = prepare_out_argument(out_obj, dtype, out_shape);
    // 准备输出参数数组
    DISPATCH_DTYPE(dtype, [&]{
        // 根据数据类型分发到具体的加权距离计算函数
        pdist_weighted<scalar_t>(out, x, w, f);
    });
    return out;
}

template <typename Func>
// 定义模板函数 cdist，用于计算两个集合之间的距离矩阵，支持不同的距离函数
py::array cdist(const py::object& out_obj, const py::object& x_obj,
                const py::object& y_obj, const py::object& w_obj, Func&& f) {
    auto x = npy_asarray(x_obj);
    auto y = npy_asarray(y_obj);
    // 将 Python 对象 x 和 y 转换为 NumPy 数组 x 和 y
    if (x.ndim() != 2) {
        // 如果 x 不是二维数组，则抛出异常
        throw std::invalid_argument("XA must be a 2-dimensional array.");
    }
    if (y.ndim() != 2) {
        // 如果 y 不是二维数组，则抛出异常
        throw std::invalid_argument("XB must be a 2-dimensional array.");
    }
    const intptr_t m = x.shape(1);
    if (m != y.shape(1)) {
        // 如果 x 和 y 的列数不相等，则抛出异常
        throw std::invalid_argument(
            "XA and XB must have the same number of columns "
            "(i.e. feature dimension).");
    }

    std::array<intptr_t, 2> out_shape{{x.shape(0), y.shape(0)}};
    // 计算输出数组的形状，以存储距离矩阵
    if (w_obj.is_none()) {
        // 如果权重对象 w_obj 为空
        auto dtype = promote_type_real(common_type(x.dtype(), y.dtype()));
        // 推断出适当的数据类型以存储结果，考虑 x 和 y 的数据类型
        auto out = prepare_out_argument(out_obj, dtype, out_shape);
        // 准备输出参数数组
        DISPATCH_DTYPE(dtype, [&]{
            // 根据数据类型分发到具体的距离计算函数
            cdist_unweighted<scalar_t>(out, x, y, f);
        });
        return out;
    }

    auto w = prepare_single_weight(w_obj, m);
    // 准备单个权重数组 w
    auto dtype = promote_type_real(
        common_type(x.dtype(), y.dtype(), w.dtype()));
    // 推断出适当的数据类型以存储结果，考虑 x、y 和 w 的数据类型
    auto out = prepare_out_argument(out_obj, dtype, out_shape);
    // 准备输出参数数组
    DISPATCH_DTYPE(dtype, [&]{
        // 根据数据类型分发到具体的加权距离计算函数
        cdist_weighted<scalar_t>(out, x, y, w, f);
    });
    return out;
}

PYBIND11_MODULE(_distance_pybind, m) {
    // Python 扩展模块初始化函数
    if (_import_array() != 0) {
        // 导入 NumPy 失败时抛出异常
        throw py::error_already_set();
    }
    using namespace pybind11::literals;
    // 使用 pybind11 的 _literals 命名空间
    m.def("pdist_canberra",
          // 定义 pdist_canberra 函数，计算 Canberra 距离
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, CanberraDistance{});
          },
          // 函数参数说明：x、w 和 out
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_hamming",
          // 定义 pdist_hamming 函数，计算 Hamming 距离
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, HammingDistance{});
          },
          // 函数参数说明：x、w 和 out
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
}
    // 定义 pdist_dice 函数，使用 DiceDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_dice",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 DiceDistance 策略计算距离，并返回结果
              return pdist(out, x, w, DiceDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_jaccard 函数，使用 JaccardDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_jaccard",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 JaccardDistance 策略计算距离，并返回结果
              return pdist(out, x, w, JaccardDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_kulczynski1 函数，使用 Kulczynski1Distance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_kulczynski1",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 Kulczynski1Distance 策略计算距离，并返回结果
              return pdist(out, x, w, Kulczynski1Distance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_rogerstanimoto 函数，使用 RogerstanimotoDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_rogerstanimoto",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 RogerstanimotoDistance 策略计算距离，并返回结果
              return pdist(out, x, w, RogerstanimotoDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_russellrao 函数，使用 RussellRaoDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_russellrao",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 RussellRaoDistance 策略计算距离，并返回结果
              return pdist(out, x, w, RussellRaoDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_sokalmichener 函数，使用 SokalmichenerDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_sokalmichener",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 SokalmichenerDistance 策略计算距离，并返回结果
              return pdist(out, x, w, SokalmichenerDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_sokalsneath 函数，使用 SokalsneathDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_sokalsneath",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 SokalsneathDistance 策略计算距离，并返回结果
              return pdist(out, x, w, SokalsneathDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_yule 函数，使用 YuleDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_yule",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 YuleDistance 策略计算距离，并返回结果
              return pdist(out, x, w, YuleDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_chebyshev 函数，使用 ChebyshevDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_chebyshev",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 ChebyshevDistance 策略计算距离，并返回结果
              return pdist(out, x, w, ChebyshevDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_cityblock 函数，使用 CityBlockDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_cityblock",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 CityBlockDistance 策略计算距离，并返回结果
              return pdist(out, x, w, CityBlockDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_euclidean 函数，使用 EuclideanDistance 计算距离度量，支持可选参数 x、w、out
    m.def("pdist_euclidean",
          [](py::object x, py::object w, py::object out) {
              // 调用 pdist 函数，使用 EuclideanDistance 策略计算距离，并返回结果
              return pdist(out, x, w, EuclideanDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 pdist_minkowski 函数，使用 MinkowskiDistance 计算距离度量，支持可选参数 x、w、out 和额外参数 p
    m.def("pdist_minkowski",
          [](py::object x, py::object w, py::object out, double p) {
              // 根据参数 p 的值选择不同的距离度量策略
              if (p == 1.0) {
                  // 当 p 为 1 时，使用 CityBlockDistance 计算距离，并返回结果
                  return pdist(out, x, w, CityBlockDistance{});
              } else if (p == 2.0) {
                  // 当 p 为 2 时，使用 EuclideanDistance 计算距离，并返回结果
                  return pdist(out, x, w, EuclideanDistance{});
              } else if (std::isinf(p)) {
                  // 当 p 为无穷大时，使用 ChebyshevDistance 计算距离，并返回结果
                  return pdist(out, x, w, ChebyshevDistance{});
              } else {
                  // 其他情况下，使用 MinkowskiDistance 根据参数 p 计算距离，并返回结果
                  return pdist(out, x, w, MinkowskiDistance{p});
              }
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none(), "p"_a=2.0);
    // 定义 Python 模块中的 pdist_sqeuclidean 函数，使用 lambda 表达式调用 pdist 函数计算平方欧氏距离
    // x: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("pdist_sqeuclidean",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, SquareEuclideanDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 pdist_braycurtis 函数，使用 lambda 表达式调用 pdist 函数计算 Bray-Curtis 距离
    // x: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("pdist_braycurtis",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, BraycurtisDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_canberra 函数，使用 lambda 表达式调用 cdist 函数计算 Canberra 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_canberra",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, CanberraDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_dice 函数，使用 lambda 表达式调用 cdist 函数计算 Dice 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_dice",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, DiceDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_jaccard 函数，使用 lambda 表达式调用 cdist 函数计算 Jaccard 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_jaccard",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, JaccardDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_kulczynski1 函数，使用 lambda 表达式调用 cdist 函数计算 Kulczynski1 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_kulczynski1",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, Kulczynski1Distance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_hamming 函数，使用 lambda 表达式调用 cdist 函数计算 Hamming 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_hamming",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, HammingDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_rogerstanimoto 函数，使用 lambda 表达式调用 cdist 函数计算 Rogerstanimoto 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_rogerstanimoto",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, RogerstanimotoDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_russellrao 函数，使用 lambda 表达式调用 cdist 函数计算 RussellRao 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_russellrao",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, RussellRaoDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_sokalmichener 函数，使用 lambda 表达式调用 cdist 函数计算 Sokalmichener 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_sokalmichener",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, SokalmichenerDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_sokalsneath 函数，使用 lambda 表达式调用 cdist 函数计算 Sokalsneath 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_sokalsneath",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, SokalsneathDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    // 定义 Python 模块中的 cdist_yule 函数，使用 lambda 表达式调用 cdist 函数计算 Yule 距离
    // x: 必选参数，输入数据
    // y: 必选参数，输入数据
    // w: 可选参数，默认为 None，权重
    // out: 可选参数，默认为 None，输出结果
    m.def("cdist_yule",
    # 定义名为 cdist_chebyshev 的函数，接受参数 x, y, w, out，返回调用 cdist 函数的结果，使用 Chebyshev 距离度量
    m.def("cdist_chebyshev",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, ChebyshevDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    # 定义名为 cdist_cityblock 的函数，接受参数 x, y, w, out，返回调用 cdist 函数的结果，使用 City Block (曼哈顿) 距离度量
    m.def("cdist_cityblock",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, CityBlockDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    # 定义名为 cdist_euclidean 的函数，接受参数 x, y, w, out，返回调用 cdist 函数的结果，使用 Euclidean (欧氏) 距离度量
    m.def("cdist_euclidean",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, EuclideanDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    # 定义名为 cdist_minkowski 的函数，接受参数 x, y, w, out, p，根据 p 的值选择 Minkowski, City Block, Euclidean 或 Chebyshev 距离度量
    m.def("cdist_minkowski",
          [](py::object x, py::object y, py::object w, py::object out,
             double p) {
              if (p == 1.0) {
                  return cdist(out, x, y, w, CityBlockDistance{});
              } else if (p == 2.0) {
                  return cdist(out, x, y, w, EuclideanDistance{});
              } else if (std::isinf(p)) {
                  return cdist(out, x, y, w, ChebyshevDistance{});
              } else {
                  return cdist(out, x, y, w, MinkowskiDistance{p});
              }
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none(), "p"_a=2.0);

    # 定义名为 cdist_sqeuclidean 的函数，接受参数 x, y, w, out，返回调用 cdist 函数的结果，使用 Square Euclidean (欧氏平方) 距离度量
    m.def("cdist_sqeuclidean",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, SquareEuclideanDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());

    # 定义名为 cdist_braycurtis 的函数，接受参数 x, y, w, out，返回调用 cdist 函数的结果，使用 Bray-Curtis 距离度量
    m.def("cdist_braycurtis",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, BraycurtisDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
}

// 结束匿名命名空间
}  // namespace (anonymous)
```