# `D:\src\scipysrc\matplotlib\src\_image_wrapper.cpp`

```py
/* 将 C++ 代码包含在 Pybind11 的头文件中，这是一个 C++ 和 Python 之间的桥梁 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

/* 引入自定义的 C++ 头文件 */
#include "_image_resample.h"
#include "py_converters_11.h"

/* 命名空间的简化，使用 pybind11 的命名空间 */
namespace py = pybind11;
using namespace pybind11::literals;

/**********************************************************************
 * 自由函数
 * */

/* 函数文档字符串，描述了 image_resample 函数的参数和功能 */
const char* image_resample__doc__ =
R"""(Resample input_array, blending it in-place into output_array, using an affine transform.

Parameters
----------
input_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`
    If 2-d, the image is grayscale.  If 3-d, the image must be of size 4 in the last
    dimension and represents RGBA data.

output_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`
    The dtype and number of dimensions must match `input_array`.

transform : matplotlib.transforms.Transform instance
    The transformation from the input array to the output array.

interpolation : int, default: NEAREST
    The interpolation method.  Must be one of the following constants defined in this
    module:

      NEAREST, BILINEAR, BICUBIC, SPLINE16, SPLINE36, HANNING, HAMMING, HERMITE, KAISER,
      QUADRIC, CATROM, GAUSSIAN, BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN

resample : bool, optional
    When `True`, use a full resampling method.  When `False`, only resample when the
    output image is larger than the input image.

alpha : float, default: 1
    The transparency level, from 0 (transparent) to 1 (opaque).

norm : bool, default: False
    Whether to norm the interpolation function.

radius: float, default: 1
    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.
)""";

/* 获取变换的网格点 */
static py::array_t<double>
_get_transform_mesh(const py::object& transform, const py::ssize_t *dims)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* 给定一个非仿射变换对象，创建一个网格，将输出图像中的每个像素映射到输入图像。
    这在实际重采样过程中用作查找表。 */

    // 如果属性不存在，会引发 Python 的 AttributeError
    auto inverse = transform.attr("inverted")();

    // 定义输入网格的维度
    py::ssize_t mesh_dims[2] = {dims[0]*dims[1], 2};
    py::array_t<double> input_mesh(mesh_dims);
    auto p = input_mesh.mutable_data();

    // 构建输入网格
    for (auto y = 0; y < dims[0]; ++y) {
        for (auto x = 0; x < dims[1]; ++x) {
            *p++ = (double)x;
            *p++ = (double)y;
        }
    }

    // 应用反转的变换到输入网格
    auto output_mesh = inverse.attr("transform")(input_mesh);

    // 将输出网格转换为数组
    auto output_mesh_array =
        py::array_t<double, py::array::c_style | py::array::forcecast>(output_mesh);

    // 检查输出网格的维度是否为2
    if (output_mesh_array.ndim() != 2) {
        throw std::runtime_error(
            "Inverse transformed mesh array should be 2D not {}D"_s.format(
                output_mesh_array.ndim()));
    }

    return output_mesh_array;
}

// 使用通用的 py::array 作为输入和输出数组，而不是更常见的 py::array_t<type>，
// 因为这个函数支持多种数组数据类型。
static void
// 对输入数组进行重采样操作，并将结果写入输出数组
void image_resample(py::array input_array,
                   py::array& output_array,
                   const py::object& transform,
                   interpolation_e interpolation,
                   bool resample_,  // 避免与 resample() 函数名称冲突
                   float alpha,
                   bool norm,
                   float radius)
{
    // 验证输入数组的数据类型
    auto dtype = input_array.dtype();  // 在确定重采样器时验证数据类型
    // 获取输入数组的维度
    auto ndim = input_array.ndim();

    // 检查输入数组的维度是否为2D或3D
    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument("Input array must be a 2D or 3D array");
    }

    // 如果输入数组是3D的，确保其最后一个维度为4（RGBA格式）
    if (ndim == 3 && input_array.shape(2) != 4) {
        throw std::invalid_argument(
            "3D input array must be RGBA with shape (M, N, 4), has trailing dimension of {}"_s.format(
                input_array.shape(2)));
    }

    // 确保输入数组是连续的，不考虑数据类型
    input_array = py::array::ensure(input_array, py::array::c_style);

    // 验证输出数组的维度
    auto out_ndim = output_array.ndim();

    // 检查输出数组的维度与输入数组是否一致
    if (out_ndim != ndim) {
        throw std::invalid_argument(
            "Input ({}D) and output ({}D) arrays have different dimensionalities"_s.format(
                ndim, out_ndim));
    }

    // 如果输出数组是3D的，确保其最后一个维度为4（RGBA格式）
    if (out_ndim == 3 && output_array.shape(2) != 4) {
        throw std::invalid_argument(
            "3D output array must be RGBA with shape (M, N, 4), has trailing dimension of {}"_s.format(
                output_array.shape(2)));
    }

    // 确保输入和输出数组的数据类型一致
    if (!output_array.dtype().is(dtype)) {
        throw std::invalid_argument("Input and output arrays have mismatched types");
    }

    // 确保输出数组是C连续的
    if ((output_array.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument("Output array must be C-contiguous");
    }

    // 确保输出数组可写
    if (!output_array.writeable()) {
        throw std::invalid_argument("Output array must be writeable");
    }

    // 设置重采样参数
    resample_params_t params;
    params.interpolation = interpolation;
    params.transform_mesh = nullptr;
    params.resample = resample_;
    params.norm = norm;
    params.radius = radius;
    params.alpha = alpha;

    // 如果变换参数为空，则默认为仿射变换
    // 否则，根据变换对象的属性确定是否为仿射变换
    py::array_t<double> transform_mesh; // 只有在变换不是仿射变换时才使用，需要在函数生命周期内保持其有效性

    if (transform.is_none()) {
        params.is_affine = true;
    } else {
        // 如果属性不存在或类型转换失败，则抛出Python的AttributeError或TypeError异常
        bool is_affine = py::cast<bool>(transform.attr("is_affine"));

        if (is_affine) {
            // 将变换对象转换为仿射变换参数
            convert_trans_affine(transform, params.affine);
            params.is_affine = true;
        } else {
            // 获取非仿射变换的网格
            transform_mesh = _get_transform_mesh(transform, output_array.shape());
            params.transform_mesh = transform_mesh.data();
            params.is_affine = false;
        }
    }
    // 如果条件满足，根据输入数组的维度和数据类型选择合适的重采样器函数
    if (auto resampler =
            (ndim == 2) ? (
                // 如果数据类型为 uint8_t，则选择对应的灰度图像重采样函数
                (dtype.equal(py::dtype::of<std::uint8_t>())) ? resample<agg::gray8> :
                // 如果数据类型为 int8_t，则选择对应的灰度图像重采样函数
                (dtype.equal(py::dtype::of<std::int8_t>())) ? resample<agg::gray8> :
                // 如果数据类型为 uint16_t，则选择对应的灰度图像重采样函数
                (dtype.equal(py::dtype::of<std::uint16_t>())) ? resample<agg::gray16> :
                // 如果数据类型为 int16_t，则选择对应的灰度图像重采样函数
                (dtype.equal(py::dtype::of<std::int16_t>())) ? resample<agg::gray16> :
                // 如果数据类型为 float，则选择对应的灰度图像重采样函数
                (dtype.equal(py::dtype::of<float>())) ? resample<agg::gray32> :
                // 如果数据类型为 double，则选择对应的灰度图像重采样函数
                (dtype.equal(py::dtype::of<double>())) ? resample<agg::gray64> :
                nullptr) : (
            // ndim == 3，处理三维数组情况
                // 如果数据类型为 uint8_t，则选择对应的 RGBA 彩色图像重采样函数
                (dtype.equal(py::dtype::of<std::uint8_t>())) ? resample<agg::rgba8> :
                // 如果数据类型为 int8_t，则选择对应的 RGBA 彩色图像重采样函数
                (dtype.equal(py::dtype::of<std::int8_t>())) ? resample<agg::rgba8> :
                // 如果数据类型为 uint16_t，则选择对应的 RGBA 彩色图像重采样函数
                (dtype.equal(py::dtype::of<std::uint16_t>())) ? resample<agg::rgba16> :
                // 如果数据类型为 int16_t，则选择对应的 RGBA 彩色图像重采样函数
                (dtype.equal(py::dtype::of<std::int16_t>())) ? resample<agg::rgba16> :
                // 如果数据类型为 float，则选择对应的 RGBA 彩色图像重采样函数
                (dtype.equal(py::dtype::of<float>())) ? resample<agg::rgba32> :
                // 如果数据类型为 double，则选择对应的 RGBA 彩色图像重采样函数
                (dtype.equal(py::dtype::of<double>())) ? resample<agg::rgba64> :
                nullptr)) {
        // 进入 Python 线程允许区域
        Py_BEGIN_ALLOW_THREADS
        // 调用选择的重采样器函数，处理输入数组并输出到输出数组
        resampler(
            input_array.data(), input_array.shape(1), input_array.shape(0),
            output_array.mutable_data(), output_array.shape(1), output_array.shape(0),
            params);
        // 结束 Python 线程允许区域
        Py_END_ALLOW_THREADS
    } else {
        // 如果未找到合适的重采样器函数，则抛出无效参数异常
        throw std::invalid_argument("arrays must be of dtype byte, short, float32 or float64");
    }
PYBIND11_MODULE(_image, m) {
    // 在 Python 中创建名为 _image 的模块，并将其绑定到变量 m 上

    py::enum_<interpolation_e>(m, "_InterpolationType")
        // 定义一个 Python 枚举类型 _InterpolationType，并将其绑定到 m 上
        .value("NEAREST", NEAREST)    // 添加枚举值 NEAREST
        .value("BILINEAR", BILINEAR)  // 添加枚举值 BILINEAR
        .value("BICUBIC", BICUBIC)    // 添加枚举值 BICUBIC
        .value("SPLINE16", SPLINE16)  // 添加枚举值 SPLINE16
        .value("SPLINE36", SPLINE36)  // 添加枚举值 SPLINE36
        .value("HANNING", HANNING)    // 添加枚举值 HANNING
        .value("HAMMING", HAMMING)    // 添加枚举值 HAMMING
        .value("HERMITE", HERMITE)    // 添加枚举值 HERMITE
        .value("KAISER", KAISER)      // 添加枚举值 KAISER
        .value("QUADRIC", QUADRIC)    // 添加枚举值 QUADRIC
        .value("CATROM", CATROM)      // 添加枚举值 CATROM
        .value("GAUSSIAN", GAUSSIAN)  // 添加枚举值 GAUSSIAN
        .value("BESSEL", BESSEL)      // 添加枚举值 BESSEL
        .value("MITCHELL", MITCHELL)  // 添加枚举值 MITCHELL
        .value("SINC", SINC)          // 添加枚举值 SINC
        .value("LANCZOS", LANCZOS)    // 添加枚举值 LANCZOS
        .value("BLACKMAN", BLACKMAN)  // 添加枚举值 BLACKMAN
        .export_values();             // 导出枚举值到 Python

    m.def("resample", &image_resample,
        "input_array"_a,               // 输入数组参数的名称
        "output_array"_a,              // 输出数组参数的名称
        "transform"_a,                 // 变换参数的名称
        "interpolation"_a = interpolation_e::NEAREST,  // 插值方法参数，默认为 NEAREST
        "resample"_a = false,          // 是否重采样的布尔参数，默认为 false
        "alpha"_a = 1,                 // alpha 参数，默认为 1
        "norm"_a = false,              // 是否归一化的布尔参数，默认为 false
        "radius"_a = 1,                // 半径参数，默认为 1
        image_resample__doc__);        // resample 函数的文档字符串
}
```