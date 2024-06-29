# `D:\src\scipysrc\matplotlib\src\_path_wrapper.cpp`

```py
py::list
convert_polygon_vector(std::vector<Polygon> &polygons)
{
    // 创建一个空的 Python 列表，大小为传入多边形向量的大小
    auto result = py::list(polygons.size());

    // 遍历每个多边形
    for (size_t i = 0; i < polygons.size(); ++i) {
        const auto& poly = polygons[i];
        // 创建一个 Numpy 数组，包含多边形顶点的坐标
        py::ssize_t dims[] = { static_cast<py::ssize_t>(poly.size()), 2 };
        result[i] = py::array(dims, reinterpret_cast<const double *>(poly.data()));
    }

    // 返回填充好的 Python 列表
    return result;
}

static bool
Py_point_in_path(double x, double y, double r, mpl::PathIterator path,
                 agg::trans_affine trans)
{
    // 调用底层函数判断点 (x, y) 是否在路径中
    return point_in_path(x, y, r, path, trans);
}

static py::array_t<double>
Py_points_in_path(py::array_t<double> points_obj, double r, mpl::PathIterator path,
                  agg::trans_affine trans)
{
    // 将传入的 Python 数组转换为 C++ 数组视图
    numpy::array_view<double, 2> points;

    if (!convert_points(points_obj.ptr(), &points)) {
        throw py::error_already_set();
    }

    // 检查点数组的形状
    if (!check_trailing_shape(points, "points", 2)) {
        throw py::error_already_set();
    }

    // 创建一个用于存储结果的 Python 数组
    py::ssize_t dims[] = { static_cast<py::ssize_t>(points.size()) };
    py::array_t<uint8_t> results(dims);
    auto results_mutable = results.mutable_unchecked<1>();

    // 调用底层函数计算点集中每个点是否在路径内，结果存储在 results 中
    points_in_path(points, r, path, trans, results_mutable);

    // 返回结果数组
    return results;
}

static py::tuple
Py_update_path_extents(mpl::PathIterator path, agg::trans_affine trans,
                       agg::rect_d rect, py::array_t<double> minpos, bool ignore)
{
    bool changed;

    // 检查 minpos 数组的维度和长度是否符合要求
    if (minpos.ndim() != 1) {
        throw py::value_error(
            "minpos must be 1D, got " + std::to_string(minpos.ndim()));
    }
    if (minpos.shape(0) != 2) {
        throw py::value_error(
            "minpos must be of length 2, got " + std::to_string(minpos.shape(0)));
    }

    // 创建一个结构体用于存储路径的边界信息
    extent_limits e;

    // 根据 ignore 参数设置边界信息
    if (ignore) {
        reset_limits(e);
    } else {
        if (rect.x1 > rect.x2) {
            e.x0 = std::numeric_limits<double>::infinity();
            e.x1 = -std::numeric_limits<double>::infinity();
        } else {
            e.x0 = rect.x1;
            e.x1 = rect.x2;
        }
        if (rect.y1 > rect.y2) {
            e.y0 = std::numeric_limits<double>::infinity();
            e.y1 = -std::numeric_limits<double>::infinity();
        } else {
            e.y0 = rect.y1;
            e.y1 = rect.y2;
        }
        e.xm = *minpos.data(0);
        e.ym = *minpos.data(1);
    }

    // 调用底层函数更新路径的边界信息
    update_path_extents(path, trans, e);

    // 检查边界信息是否改变
    changed = (e.x0 != rect.x1 || e.y0 != rect.y1 || e.x1 != rect.x2 || e.y1 != rect.y2 ||
               e.xm != *minpos.data(0) || e.ym != *minpos.data(1));

    // 创建一个 2x2 的数组用于存储返回的边界信息
    py::ssize_t extentsdims[] = { 2, 2 };
    py::array_t<double> outextents(extentsdims);
    // 将变量 e 的 x0 赋值给 outextents 数组的第一行第一列
    *outextents.mutable_data(0, 0) = e.x0;
    // 将变量 e 的 y0 赋值给 outextents 数组的第一行第二列
    *outextents.mutable_data(0, 1) = e.y0;
    // 将变量 e 的 x1 赋值给 outextents 数组的第二行第一列
    *outextents.mutable_data(1, 0) = e.x1;
    // 将变量 e 的 y1 赋值给 outextents 数组的第二行第二列
    *outextents.mutable_data(1, 1) = e.y1;
    
    // 创建一个包含两个元素的数组，用于存储最小位置维度信息
    py::ssize_t minposdims[] = { 2 };
    // 创建一个双精度浮点数数组 outminpos
    py::array_t<double> outminpos(minposdims);
    // 将变量 e 的 xm 赋值给 outminpos 数组的第一个元素
    *outminpos.mutable_data(0) = e.xm;
    // 将变量 e 的 ym 赋值给 outminpos 数组的第二个元素
    *outminpos.mutable_data(1) = e.ym;
    
    // 返回一个 Python 元组，包含 outextents、outminpos 和 changed 变量
    return py::make_tuple(outextents, outminpos, changed);
static py::tuple
Py_get_path_collection_extents(agg::trans_affine master_transform,
                               py::object paths_obj, py::object transforms_obj,
                               py::object offsets_obj, agg::trans_affine offset_trans)
{
    // 定义路径生成器和转换数组视图以及偏移数组视图，以及极限值对象
    mpl::PathGenerator paths;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    extent_limits e;

    // 尝试将 Python 对象转换为 C++ 类型，如果失败则抛出异常
    if (!convert_pathgen(paths_obj.ptr(), &paths)) {
        throw py::error_already_set();
    }
    if (!convert_transforms(transforms_obj.ptr(), &transforms)) {
        throw py::error_already_set();
    }
    if (!convert_points(offsets_obj.ptr(), &offsets)) {
        throw py::error_already_set();
    }

    // 调用 C++ 函数计算路径集合的边界
    get_path_collection_extents(
        master_transform, paths, transforms, offsets, offset_trans, e);

    // 创建用于存储边界值的 Python 数组对象
    py::ssize_t dims[] = { 2, 2 };
    py::array_t<double> extents(dims);
    *extents.mutable_data(0, 0) = e.x0;
    *extents.mutable_data(0, 1) = e.y0;
    *extents.mutable_data(1, 0) = e.x1;
    *extents.mutable_data(1, 1) = e.y1;

    // 创建用于存储最小位置的 Python 数组对象
    py::ssize_t minposdims[] = { 2 };
    py::array_t<double> minpos(minposdims);
    *minpos.mutable_data(0) = e.xm;
    *minpos.mutable_data(1) = e.ym;

    // 返回边界值和最小位置的 Python 元组对象
    return py::make_tuple(extents, minpos);
}

static py::object
Py_point_in_path_collection(double x, double y, double radius,
                            agg::trans_affine master_transform, py::object paths_obj,
                            py::object transforms_obj, py::object offsets_obj,
                            agg::trans_affine offset_trans, bool filled)
{
    // 定义路径生成器和转换数组视图以及偏移数组视图，以及结果向量
    mpl::PathGenerator paths;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    std::vector<int> result;

    // 尝试将 Python 对象转换为 C++ 类型，如果失败则抛出异常
    if (!convert_pathgen(paths_obj.ptr(), &paths)) {
        throw py::error_already_set();
    }
    if (!convert_transforms(transforms_obj.ptr(), &transforms)) {
        throw py::error_already_set();
    }
    if (!convert_points(offsets_obj.ptr(), &offsets)) {
        throw py::error_already_set();
    }

    // 调用 C++ 函数判断点是否在路径集合中
    point_in_path_collection(x, y, radius, master_transform, paths, transforms, offsets,
                             offset_trans, filled, result);

    // 创建用于存储结果的 Python 数组对象
    py::ssize_t dims[] = { static_cast<py::ssize_t>(result.size()) };
    return py::array(dims, result.data());
}

static bool
Py_path_in_path(mpl::PathIterator a, agg::trans_affine atrans,
                mpl::PathIterator b, agg::trans_affine btrans)
{
    // 调用 C++ 函数判断一个路径是否在另一个路径内部
    return path_in_path(a, atrans, b, btrans);
}

static py::list
Py_clip_path_to_rect(mpl::PathIterator path, agg::rect_d rect, bool inside)
{
    // 定义用于存储多边形结果的向量
    std::vector<Polygon> result;

    // 调用 C++ 函数将路径裁剪到矩形内
    clip_path_to_rect(path, rect, inside, result);

    // 将 C++ 结果转换为 Python 的列表对象并返回
    return convert_polygon_vector(result);
}

static py::object
Py_affine_transform(py::array_t<double, py::array::c_style | py::array::forcecast> vertices_arr,
                    agg::trans_affine trans)
{
    // 检查 vertices_arr 的维度是否为2
    if (vertices_arr.ndim() == 2) {
        // 获取二维数组 vertices_arr 的非检查视图
        auto vertices = vertices_arr.unchecked<2>();

        // 检查 vertices 数组的最后一个维度是否为2
        if (!check_trailing_shape(vertices, "vertices", 2)) {
            // 如果检查失败，抛出 Python 异常并返回
            throw py::error_already_set();
        }

        // 创建一个双精度的二维数组 result，维度为 vertices.shape(0) x 2
        py::ssize_t dims[] = { vertices.shape(0), 2 };
        py::array_t<double> result(dims);
        // 获取 result 的可变视图，以便进行修改
        auto result_mutable = result.mutable_unchecked<2>();

        // 对 vertices 应用二维仿射变换 trans，结果存储在 result_mutable 中
        affine_transform_2d(vertices, trans, result_mutable);
        // 返回结果数组 result
        return result;
    } else if (vertices_arr.ndim() == 1) {
        // 获取一维数组 vertices_arr 的非检查视图
        auto vertices = vertices_arr.unchecked<1>();

        // 创建一个双精度的一维数组 result，维度为 vertices.shape(0)
        py::ssize_t dims[] = { vertices.shape(0) };
        py::array_t<double> result(dims);
        // 获取 result 的可变视图，以便进行修改
        auto result_mutable = result.mutable_unchecked<1>();

        // 对 vertices 应用一维仿射变换 trans，结果存储在 result_mutable 中
        affine_transform_1d(vertices, trans, result_mutable);
        // 返回结果数组 result
        return result;
    } else {
        // 如果 vertices_arr 的维度既不是1也不是2，抛出值错误异常
        throw py::value_error(
            "vertices must be 1D or 2D, not" + std::to_string(vertices_arr.ndim()) + "D");
    }
static int
Py_count_bboxes_overlapping_bbox(agg::rect_d bbox, py::object bboxes_obj)
{
    // 定义一个 numpy 的三维双精度浮点数组的视图，用于存储边界框数据
    numpy::array_view<const double, 3> bboxes;

    // 尝试将 Python 对象转换为 C++ 的边界框数组，如果失败则抛出异常
    if (!convert_bboxes(bboxes_obj.ptr(), &bboxes)) {
        throw py::error_already_set();
    }

    // 调用计算重叠边界框数量的函数，并返回结果
    return count_bboxes_overlapping_bbox(bbox, bboxes);
}

static bool
Py_path_intersects_path(mpl::PathIterator p1, mpl::PathIterator p2, bool filled)
{
    // 定义两个仿射变换对象
    agg::trans_affine t1;
    agg::trans_affine t2;
    bool result;

    // 检查路径 p1 和 p2 是否相交，并存储结果
    result = path_intersects_path(p1, p2);

    // 如果需要考虑填充情况
    if (filled) {
        // 如果没有相交，则检查 p1 是否在 p2 中
        if (!result) {
            result = path_in_path(p1, t1, p2, t2);
        }
        // 如果仍然没有相交，则检查 p2 是否在 p1 中
        if (!result) {
            result = path_in_path(p2, t1, p1, t2);
        }
    }

    // 返回最终的相交结果
    return result;
}

static bool
Py_path_intersects_rectangle(mpl::PathIterator path, double rect_x1, double rect_y1,
                             double rect_x2, double rect_y2, bool filled)
{
    // 调用函数检查路径是否与矩形相交，并返回结果
    return path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2, filled);
}

static py::list
Py_convert_path_to_polygons(mpl::PathIterator path, agg::trans_affine trans,
                            double width, double height, bool closed_only)
{
    // 定义一个多边形的向量，用于存储转换后的多边形
    std::vector<Polygon> result;

    // 将路径转换为多边形，并存储到 result 中
    convert_path_to_polygons(path, trans, width, height, closed_only, result);

    // 将多边形向量转换为 Python 的列表并返回
    return convert_polygon_vector(result);
}

static py::tuple
Py_cleanup_path(mpl::PathIterator path, agg::trans_affine trans, bool remove_nans,
                agg::rect_d clip_rect, e_snap_mode snap_mode, double stroke_width,
                std::optional<bool> simplify, bool return_curves, SketchParams sketch)
{
    // 如果未指定简化选项，则根据路径的默认设置进行简化
    if (!simplify.has_value()) {
        simplify = path.should_simplify();
    }

    // 确定是否需要进行剪裁操作
    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2);

    // 定义用于存储顶点和编码的向量
    std::vector<double> vertices;
    std::vector<npy_uint8> codes;

    // 清理路径并将结果存储在 vertices 和 codes 中
    cleanup_path(path, trans, remove_nans, do_clip, clip_rect, snap_mode, stroke_width,
                 *simplify, return_curves, sketch, vertices, codes);

    // 计算编码的长度
    auto length = static_cast<py::ssize_t>(codes.size());

    // 创建用于存储顶点的 NumPy 数组
    py::ssize_t vertices_dims[] = { length, 2 };
    py::array pyvertices(vertices_dims, vertices.data());

    // 创建用于存储编码的 NumPy 数组
    py::ssize_t codes_dims[] = { length };
    py::array pycodes(codes_dims, codes.data());

    // 返回顶点数组和编码数组组成的 Python 元组
    return py::make_tuple(pyvertices, pycodes);
}



const char *Py_convert_to_string__doc__ =
R"""(--

Convert *path* to a bytestring.

The first five parameters (up to *sketch*) are interpreted as in `.cleanup_path`. The
following ones are detailed below.

Parameters
----------
path : Path
trans : Transform or None
clip_rect : sequence of 4 floats, or None
simplify : bool
sketch : tuple of 3 floats, or None
precision : int
    The precision used to "%.*f"-format the values. Trailing zeros and decimal points
    are always removed. (precision=-1 is a special case used to implement
    ttconv-back-compatible conversion.)
codes : sequence of 5 bytestrings
    The bytes representation of each opcode (MOVETO, LINETO, CURVE3, CURVE4, CLOSEPOLY),
    in that order. If the bytes for CURVE3 is empty, quad segments are automatically
    converted to cubic ones (this is used by backends such as pdf and ps, which do not
    support quads).


# 如果CURVE3的字节为空，则自动将二次曲线段转换为三次曲线段。
# 这是为了支持不支持二次曲线的后端（如pdf和ps）而设计的。
# 注意：注释内容不要超出代码块。
postfix : bool
    Whether the opcode comes after the values (True) or before (False).
)""";

static py::object
Py_convert_to_string(mpl::PathIterator path, agg::trans_affine trans,
                     agg::rect_d cliprect, std::optional<bool> simplify,
                     SketchParams sketch, int precision,
                     std::array<std::string, 5> codes_obj, bool postfix)
{
    char *codes[5];
    std::string buffer;
    bool status;

    // 将std::array中的每个std::string转换为C风格的char*并存储在codes数组中
    for (auto i = 0; i < 5; ++i) {
        codes[i] = const_cast<char *>(codes_obj[i].c_str());
    }

    // 如果simplify参数未指定值，则根据path对象的should_simplify方法来确定
    if (!simplify.has_value()) {
        simplify = path.should_simplify();
    }

    // 调用convert_to_string函数，将path转换为字符串表示，并存储在buffer中
    status = convert_to_string(path, trans, cliprect, *simplify, sketch, precision,
                               codes, postfix, buffer);

    // 如果转换失败，抛出异常
    if (!status) {
        throw py::value_error("Malformed path codes");
    }

    // 返回转换后的字符串表示，转换为py::bytes对象返回
    return py::bytes(buffer);
}

const char *Py_is_sorted_and_has_non_nan__doc__ =
R"""(--

Return whether the 1D *array* is monotonically increasing, ignoring NaNs, and has at
least one non-nan value.)""";

static bool
Py_is_sorted_and_has_non_nan(py::object obj)
{
    bool result;

    // 尝试将输入的obj对象转换为PyArrayObject*类型的数组对象
    PyArrayObject *array = (PyArrayObject *)PyArray_CheckFromAny(
        obj.ptr(), NULL, 1, 1, NPY_ARRAY_NOTSWAPPED, NULL);

    // 如果转换失败，抛出异常
    if (array == NULL) {
        throw py::error_already_set();
    }

    // 根据数组的数据类型选择适当的is_sorted_and_has_non_nan函数进行调用
    switch (PyArray_TYPE(array)) {
    case NPY_INT:
        result = is_sorted_and_has_non_nan<npy_int>(array);
        break;
    case NPY_LONG:
        result = is_sorted_and_has_non_nan<npy_long>(array);
        break;
    case NPY_LONGLONG:
        result = is_sorted_and_has_non_nan<npy_longlong>(array);
        break;
    case NPY_FLOAT:
        result = is_sorted_and_has_non_nan<npy_float>(array);
        break;
    case NPY_DOUBLE:
        result = is_sorted_and_has_non_nan<npy_double>(array);
        break;
    default:
        // 对于其他类型的数组，强制转换为NPY_DOUBLE类型，并调用对应的函数
        Py_DECREF(array);
        array = (PyArrayObject *)PyArray_FromObject(obj.ptr(), NPY_DOUBLE, 1, 1);
        if (array == NULL) {
            throw py::error_already_set();
        }
        result = is_sorted_and_has_non_nan<npy_double>(array);
    }

    // 释放数组对象并返回结果
    Py_DECREF(array);

    return result;
}

// Python绑定模块，注册一系列函数和它们的文档字符串
PYBIND11_MODULE(_path, m)
{
    // 导入Python的数组支持库
    auto ia = [m]() -> const void* {
        import_array();
        return &m;
    };
    // 如果导入失败，抛出异常
    if (ia() == NULL) {
        throw py::error_already_set();
    }

    // 向Python中注册几个函数及其文档字符串
    m.def("point_in_path", &Py_point_in_path,
          "x"_a, "y"_a, "radius"_a, "path"_a, "trans"_a);
    m.def("points_in_path", &Py_points_in_path,
          "points"_a, "radius"_a, "path"_a, "trans"_a);
    m.def("update_path_extents", &Py_update_path_extents,
          "path"_a, "trans"_a, "rect"_a, "minpos"_a, "ignore"_a);
    m.def("get_path_collection_extents", &Py_get_path_collection_extents,
          "master_transform"_a, "paths"_a, "transforms"_a, "offsets"_a,
          "offset_transform"_a);
}
    // 将 C++ 函数 Py_point_in_path_collection 注册为 Python 模块方法
    m.def("point_in_path_collection", &Py_point_in_path_collection,
          "x"_a, "y"_a, "radius"_a, "master_transform"_a, "paths"_a, "transforms"_a,
          "offsets"_a, "offset_trans"_a, "filled"_a);
    
    // 将 C++ 函数 Py_path_in_path 注册为 Python 模块方法
    m.def("path_in_path", &Py_path_in_path,
          "path_a"_a, "trans_a"_a, "path_b"_a, "trans_b"_a);
    
    // 将 C++ 函数 Py_clip_path_to_rect 注册为 Python 模块方法
    m.def("clip_path_to_rect", &Py_clip_path_to_rect,
          "path"_a, "rect"_a, "inside"_a);
    
    // 将 C++ 函数 Py_affine_transform 注册为 Python 模块方法
    m.def("affine_transform", &Py_affine_transform,
          "points"_a, "trans"_a);
    
    // 将 C++ 函数 Py_count_bboxes_overlapping_bbox 注册为 Python 模块方法
    m.def("count_bboxes_overlapping_bbox", &Py_count_bboxes_overlapping_bbox,
          "bbox"_a, "bboxes"_a);
    
    // 将 C++ 函数 Py_path_intersects_path 注册为 Python 模块方法
    m.def("path_intersects_path", &Py_path_intersects_path,
          "path1"_a, "path2"_a, "filled"_a = false);
    
    // 将 C++ 函数 Py_path_intersects_rectangle 注册为 Python 模块方法
    m.def("path_intersects_rectangle", &Py_path_intersects_rectangle,
          "path"_a, "rect_x1"_a, "rect_y1"_a, "rect_x2"_a, "rect_y2"_a,
          "filled"_a = false);
    
    // 将 C++ 函数 Py_convert_path_to_polygons 注册为 Python 模块方法
    m.def("convert_path_to_polygons", &Py_convert_path_to_polygons,
          "path"_a, "trans"_a, "width"_a = 0.0, "height"_a = 0.0,
          "closed_only"_a = false);
    
    // 将 C++ 函数 Py_cleanup_path 注册为 Python 模块方法
    m.def("cleanup_path", &Py_cleanup_path,
          "path"_a, "trans"_a, "remove_nans"_a, "clip_rect"_a, "snap_mode"_a,
          "stroke_width"_a, "simplify"_a, "return_curves"_a, "sketch"_a);
    
    // 将 C++ 函数 Py_convert_to_string 注册为 Python 模块方法
    m.def("convert_to_string", &Py_convert_to_string,
          "path"_a, "trans"_a, "clip_rect"_a, "simplify"_a, "sketch"_a, "precision"_a,
          "codes"_a, "postfix"_a,
          Py_convert_to_string__doc__);
    
    // 将 C++ 函数 Py_is_sorted_and_has_non_nan 注册为 Python 模块方法
    m.def("is_sorted_and_has_non_nan", &Py_is_sorted_and_has_non_nan,
          "array"_a,
          Py_is_sorted_and_has_non_nan__doc__);
}



# 这行代码关闭了一个代码块，可能是一个函数定义、循环、条件语句或其他代码块的结束位置。
```