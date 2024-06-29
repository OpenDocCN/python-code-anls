# `D:\src\scipysrc\matplotlib\src\py_converters_11.h`

```
#ifndef MPL_PY_CONVERTERS_11_H
#define MPL_PY_CONVERTERS_11_H

// pybind11 equivalent of py_converters.h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "agg_basics.h"
#include "agg_trans_affine.h"
#include "path_converters.h"

// 声明转换函数，将 Python 对象转换为 agg::trans_affine 类型
void convert_trans_affine(const py::object& transform, agg::trans_affine& affine);

namespace PYBIND11_NAMESPACE { namespace detail {

    // 定义 type_caster 结构体模板，用于将 agg::rect_d 类型转换为 Python 对象
    template <> struct type_caster<agg::rect_d> {
    public:
        PYBIND11_TYPE_CASTER(agg::rect_d, const_name("rect_d"));

        // 实现 load 方法，将 Python 对象转换为 agg::rect_d 类型
        bool load(handle src, bool) {
            // 如果传入对象为 None，则设定矩形的默认值
            if (src.is_none()) {
                value.x1 = 0.0;
                value.y1 = 0.0;
                value.x2 = 0.0;
                value.y2 = 0.0;
                return true;
            }

            // 尝试将输入对象转换为双精度浮点数的 NumPy 数组
            auto rect_arr = py::array_t<double>::ensure(src);

            // 检查数组的维度和形状，根据不同维度进行不同处理
            if (rect_arr.ndim() == 2) {
                if (rect_arr.shape(0) != 2 || rect_arr.shape(1) != 2) {
                    throw py::value_error("Invalid bounding box");
                }

                // 从数组中读取矩形的坐标值
                value.x1 = *rect_arr.data(0, 0);
                value.y1 = *rect_arr.data(0, 1);
                value.x2 = *rect_arr.data(1, 0);
                value.y2 = *rect_arr.data(1, 1);

            } else if (rect_arr.ndim() == 1) {
                if (rect_arr.shape(0) != 4) {
                    throw py::value_error("Invalid bounding box");
                }

                // 从数组中读取矩形的坐标值
                value.x1 = *rect_arr.data(0);
                value.y1 = *rect_arr.data(1);
                value.x2 = *rect_arr.data(2);
                value.y2 = *rect_arr.data(3);

            } else {
                throw py::value_error("Invalid bounding box");
            }

            return true;
        }
    };

    // 定义 type_caster 结构体模板，用于将 agg::trans_affine 类型转换为 Python 对象
    template <> struct type_caster<agg::trans_affine> {
    public:
        PYBIND11_TYPE_CASTER(agg::trans_affine, const_name("trans_affine"));

        // 实现 load 方法，将 Python 对象转换为 agg::trans_affine 类型
        bool load(handle src, bool) {
            // 如果传入对象为 None，则认为是单位变换，不做改变
            if (src.is_none()) {
                return true;
            }

            // 尝试将输入对象转换为双精度浮点数的二维 NumPy 数组
            auto array = py::array_t<double, py::array::c_style>::ensure(src);

            // 检查数组是否成功转换，并验证其维度和形状
            if (!array || array.ndim() != 2 ||
                    array.shape(0) != 3 || array.shape(1) != 3) {
                throw std::invalid_argument("Invalid affine transformation matrix");
            }

            // 从数组中读取变换矩阵的值，存入 agg::trans_affine 对象
            auto buffer = array.data();
            value.sx = buffer[0];
            value.shx = buffer[1];
            value.tx = buffer[2];
            value.shy = buffer[3];
            value.sy = buffer[4];
            value.ty = buffer[5];

            return true;
        }
    };

    // 声明 type_caster 结构体模板，用于将枚举类型 e_snap_mode 转换为 Python 对象
    public:
        # 定义一个公共部分，该部分包含下面的函数和变量

        PYBIND11_TYPE_CASTER(e_snap_mode, const_name("e_snap_mode"));
        # 使用 PYBIND11 宏定义了一个类型转换器 e_snap_mode，使用常量名 "e_snap_mode"

        bool load(handle src, bool) {
            # 定义一个名为 load 的函数，接受两个参数：src 和一个布尔值

            if (src.is_none()) {
                # 如果 src 是 None 类型
                value = SNAP_AUTO;
                # 将 value 设置为 SNAP_AUTO
                return true;
                # 返回 true，表示加载成功
            }

            value = src.cast<bool>() ? SNAP_TRUE : SNAP_FALSE;
            # 将 src 转换为布尔值，并根据其值设置 value 为 SNAP_TRUE 或 SNAP_FALSE

            return true;
            # 返回 true，表示加载成功
        }
    };
#ifdef MPL_PY_ADAPTORS_H
    // 定义模板特化，用于将 mpl::PathIterator 转换为 Python 对象
    template <> struct type_caster<mpl::PathIterator> {
    public:
        // 使用宏定义 PYBIND11_TYPE_CASTER，指定类型名称为 "PathIterator"
        PYBIND11_TYPE_CASTER(mpl::PathIterator, const_name("PathIterator"));

        // 载入函数，从 Python 对象加载数据到 mpl::PathIterator 对象
        bool load(handle src, bool) {
            // 如果 Python 对象为 None，则返回 true
            if (src.is_none()) {
                return true;
            }

            // 获取 vertices 属性
            auto vertices = src.attr("vertices");
            // 获取 codes 属性
            auto codes = src.attr("codes");
            // 获取 should_simplify 属性并转换为 bool 类型
            auto should_simplify = src.attr("should_simplify").cast<bool>();
            // 获取 simplify_threshold 属性并转换为 double 类型
            auto simplify_threshold = src.attr("simplify_threshold").cast<double>();

            // 将获取到的数据设置到 mpl::PathIterator 对象中
            if (!value.set(vertices.ptr(), codes.ptr(),
                           should_simplify, simplify_threshold)) {
                return false;
            }

            return true;
        }
    };
#endif

#ifdef MPL_BACKEND_AGG_BASIC_TYPES_H
    // 定义模板特化，用于将 SketchParams 转换为 Python 对象
    template <> struct type_caster<SketchParams> {
    public:
        // 使用宏定义 PYBIND11_TYPE_CASTER，指定类型名称为 "SketchParams"
        PYBIND11_TYPE_CASTER(SketchParams, const_name("SketchParams"));

        // 载入函数，从 Python 对象加载数据到 SketchParams 对象
        bool load(handle src, bool) {
            // 如果 Python 对象为 None，则将 scale 设置为 0.0，并返回 true
            if (src.is_none()) {
                value.scale = 0.0;
                return true;
            }

            // 从 Python 对象获取一个包含三个 double 类型数据的元组
            auto params = src.cast<std::tuple<double, double, double>>();
            // 将元组的值解包并设置到 SketchParams 对象的成员变量中
            std::tie(value.scale, value.length, value.randomness) = params;

            return true;
        }
    };
#endif
```