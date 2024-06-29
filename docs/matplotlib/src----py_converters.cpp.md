# `D:\src\scipysrc\matplotlib\src\py_converters.cpp`

```
#define NO_IMPORT_ARRAY
#define PY_SSIZE_T_CLEAN
#include "py_converters.h"
#include "numpy_cpp.h"

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_math_stroke.h"

extern "C" {

// 定义静态函数 convert_string_enum，用于将 Python 对象转换为枚举值
static int convert_string_enum(PyObject *obj, const char *name, const char **names, int *values, int *result)
{
    PyObject *bytesobj;
    char *str;

    // 如果传入的对象为空或者为 Py_None，则返回1表示成功
    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    // 如果对象是 Unicode 字符串，将其转换为 ASCII 字符串
    if (PyUnicode_Check(obj)) {
        bytesobj = PyUnicode_AsASCIIString(obj);
        if (bytesobj == NULL) {
            return 0;
        }
    } else if (PyBytes_Check(obj)) {
        // 如果对象是字节串，增加其引用计数
        Py_INCREF(obj);
        bytesobj = obj;
    } else {
        // 如果对象不是字符串类型，抛出类型错误异常
        PyErr_Format(PyExc_TypeError, "%s must be str or bytes", name);
        return 0;
    }

    // 获取字节串的 C 字符串表示
    str = PyBytes_AsString(bytesobj);
    if (str == NULL) {
        Py_DECREF(bytesobj);
        return 0;
    }

    // 遍历枚举名称列表，查找匹配的值
    for ( ; *names != NULL; names++, values++) {
        if (strncmp(str, *names, 64) == 0) {
            *result = *values;
            Py_DECREF(bytesobj);
            return 1;
        }
    }

    // 如果没有找到匹配的枚举值，抛出值错误异常
    PyErr_Format(PyExc_ValueError, "invalid %s value", name);
    Py_DECREF(bytesobj);
    return 0;
}

// 定义函数 convert_from_method，从对象中调用方法并进行转换
int convert_from_method(PyObject *obj, const char *name, converter func, void *p)
{
    PyObject *value;

    // 调用对象的方法获取值
    value = PyObject_CallMethod(obj, name, NULL);
    if (value == NULL) {
        // 如果方法调用失败且对象没有此方法，则清除错误并返回1表示成功（假定默认值）
        if (!PyObject_HasAttrString(obj, name)) {
            PyErr_Clear();
            return 1;
        }
        return 0;
    }

    // 使用指定的转换函数转换值
    if (!func(value, p)) {
        Py_DECREF(value);
        return 0;
    }

    // 释放 Python 对象的引用并返回成功
    Py_DECREF(value);
    return 1;
}

// 定义函数 convert_from_attr，从对象的属性中获取值并进行转换
int convert_from_attr(PyObject *obj, const char *name, converter func, void *p)
{
    PyObject *value;

    // 获取对象的属性值
    value = PyObject_GetAttrString(obj, name);
    if (value == NULL) {
        // 如果属性获取失败且对象没有此属性，则清除错误并返回1表示成功（假定默认值）
        if (!PyObject_HasAttrString(obj, name)) {
            PyErr_Clear();
            return 1;
        }
        return 0;
    }

    // 使用指定的转换函数转换属性值
    if (!func(value, p)) {
        Py_DECREF(value);
        return 0;
    }

    // 释放 Python 对象的引用并返回成功
    Py_DECREF(value);
    return 1;
}

// 定义函数 convert_double，将 Python 对象转换为双精度浮点数
int convert_double(PyObject *obj, void *p)
{
    double *val = (double *)p;

    // 将 Python 对象转换为双精度浮点数
    *val = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        return 0;
    }

    return 1;
}

// 定义函数 convert_bool，将 Python 对象转换为布尔值
int convert_bool(PyObject *obj, void *p)
{
    bool *val = (bool *)p;
    switch (PyObject_IsTrue(obj)) {
        // 根据 Python 对象的真值转换为布尔值
        case 0: *val = false; break;
        case 1: *val = true; break;
        default: return 0;  // 如果出现错误，返回0表示失败
    }
    return 1;
}

// 定义函数 convert_cap，将 Python 对象转换为线端形状枚举值
int convert_cap(PyObject *capobj, void *capp)
{
    // 定义线端形状的名称列表和对应的值列表
    const char *names[] = {"butt", "round", "projecting", NULL};
    int values[] = {agg::butt_cap, agg::round_cap, agg::square_cap};
    int result = agg::butt_cap;

    // 使用 convert_string_enum 函数将 Python 对象转换为枚举值
    if (!convert_string_enum(capobj, "capstyle", names, values, &result)) {
        return 0;
    }

    // 将转换后的枚举值存储到 capp 指向的变量中
    *(agg::line_cap_e *)capp = (agg::line_cap_e)result;
    return 1;
}

// 定义函数 convert_join，将 Python 对象转换为连接方式枚举值
int convert_join(PyObject *joinobj, void *joinp)
{
    // 定义连接方式的名称列表和对应的值列表
    const char *names[] = {"miter", "round", "bevel", NULL};
    // 定义一个整型数组 `values`，包含三个元素，分别为枚举类型 `agg::miter_join_revert`、`agg::round_join`、`agg::bevel_join`
    int values[] = {agg::miter_join_revert, agg::round_join, agg::bevel_join};
    // 定义整型变量 `result` 并初始化为 `agg::miter_join_revert`
    int result = agg::miter_join_revert;
    
    // 检查是否成功将字符串类型的 `joinobj` 转换为枚举类型 `agg::line_join_e`，并将结果存储在 `result` 中
    if (!convert_string_enum(joinobj, "joinstyle", names, values, &result)) {
        // 如果转换失败，则返回 0
        return 0;
    }
    
    // 将 `result` 强制类型转换为 `agg::line_join_e` 并存储在 `joinp` 指向的位置
    *(agg::line_join_e *)joinp = (agg::line_join_e)result;
    
    // 返回 1 表示成功完成转换和赋值操作
    return 1;
}

// 将 Python 对象转换为矩形结构体的函数
int convert_rect(PyObject *rectobj, void *rectp)
{
    // 将 void 指针转换为 agg::rect_d 结构体指针
    agg::rect_d *rect = (agg::rect_d *)rectp;

    // 如果传入的 rectobj 是空或者 None，将矩形坐标设为 0
    if (rectobj == NULL || rectobj == Py_None) {
        rect->x1 = 0.0;
        rect->y1 = 0.0;
        rect->x2 = 0.0;
        rect->y2 = 0.0;
    } else {
        // 将传入的 rectobj 转换为双精度浮点型数组对象
        PyArrayObject *rect_arr = (PyArrayObject *)PyArray_ContiguousFromAny(
                rectobj, NPY_DOUBLE, 1, 2);
        if (rect_arr == NULL) {
            return 0;
        }

        // 如果数组是二维的，检查维度是否为 2x2
        if (PyArray_NDIM(rect_arr) == 2) {
            if (PyArray_DIM(rect_arr, 0) != 2 ||
                PyArray_DIM(rect_arr, 1) != 2) {
                PyErr_SetString(PyExc_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }

        } else {  // 如果数组是一维的，检查长度是否为 4
            if (PyArray_DIM(rect_arr, 0) != 4) {
                PyErr_SetString(PyExc_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }
        }

        // 获取数组数据的指针并填充矩形结构体
        double *buff = (double *)PyArray_DATA(rect_arr);
        rect->x1 = buff[0];
        rect->y1 = buff[1];
        rect->x2 = buff[2];
        rect->y2 = buff[3];

        Py_DECREF(rect_arr);
    }
    return 1;
}

// 将 Python 对象转换为 RGBA 颜色结构体的函数
int convert_rgba(PyObject *rgbaobj, void *rgbap)
{
    // 将 void 指针转换为 agg::rgba 结构体指针
    agg::rgba *rgba = (agg::rgba *)rgbap;
    PyObject *rgbatuple = NULL;
    int success = 1;

    // 如果传入的 rgbaobj 是空或者 None，将颜色分量设为 0
    if (rgbaobj == NULL || rgbaobj == Py_None) {
        rgba->r = 0.0;
        rgba->g = 0.0;
        rgba->b = 0.0;
        rgba->a = 0.0;
    } else {
        // 尝试将传入的 rgbaobj 转换为元组对象
        if (!(rgbatuple = PySequence_Tuple(rgbaobj))) {
            success = 0;
            goto exit;
        }
        rgba->a = 1.0;
        // 解析元组对象，将颜色分量赋值给 rgba 结构体
        if (!PyArg_ParseTuple(
                 rgbatuple, "ddd|d:rgba", &(rgba->r), &(rgba->g), &(rgba->b), &(rgba->a))) {
            success = 0;
            goto exit;
        }
    }
exit:
    Py_XDECREF(rgbatuple);
    return success;
}

// 将 Python 对象转换为虚线样式结构体的函数
int convert_dashes(PyObject *dashobj, void *dashesp)
{
    // 将 void 指针转换为 Dashes 结构体指针
    Dashes *dashes = (Dashes *)dashesp;

    double dash_offset = 0.0;
    PyObject *dashes_seq = NULL;

    // 尝试解析传入的 dashobj，获取虚线样式的偏移量和序列
    if (!PyArg_ParseTuple(dashobj, "dO:dashes", &dash_offset, &dashes_seq)) {
        return 0;
    }

    // 如果虚线样式序列是 None，直接返回成功
    if (dashes_seq == Py_None) {
        return 1;
    }

    // 检查虚线样式序列是否是一个合法的 Python 序列
    if (!PySequence_Check(dashes_seq)) {
        PyErr_SetString(PyExc_TypeError, "Invalid dashes sequence");
        return 0;
    }

    // 获取虚线样式序列的长度
    Py_ssize_t nentries = PySequence_Size(dashes_seq);
    // 如果虚线样式长度为奇数，按照 pdf/ps/svg 规范迭代两次
    Py_ssize_t dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;
    // 循环遍历虚线模式数组，设置每对长度和跳过值
    for (Py_ssize_t i = 0; i < dash_pattern_length; ++i) {
        PyObject *item;  // 声明一个 Python 对象指针变量 item
        double length;   // 声明一个 double 类型变量 length，用于存储虚线长度
        double skip;     // 声明一个 double 类型变量 skip，用于存储虚线跳过值

        // 获取虚线模式序列中第 i % nentries 个元素
        item = PySequence_GetItem(dashes_seq, i % nentries);
        if (item == NULL) {  // 检查获取操作是否成功，如果失败则返回 0
            return 0;
        }
        // 将获取的 Python 对象转换为 double 类型的长度值
        length = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {  // 检查转换过程中是否发生异常
            Py_DECREF(item);    // 出现异常时释放 Python 对象，并返回 0
            return 0;
        }
        Py_DECREF(item);  // 正常情况下释放 Python 对象

        ++i;  // i 自增，准备获取下一个虚线模式元素

        // 获取虚线模式序列中第 i % nentries 个元素
        item = PySequence_GetItem(dashes_seq, i % nentries);
        if (item == NULL) {  // 检查获取操作是否成功，如果失败则返回 0
            return 0;
        }
        // 将获取的 Python 对象转换为 double 类型的跳过值
        skip = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {  // 检查转换过程中是否发生异常
            Py_DECREF(item);    // 出现异常时释放 Python 对象，并返回 0
            return 0;
        }
        Py_DECREF(item);  // 正常情况下释放 Python 对象

        // 将获取的长度和跳过值添加到虚线模式对象中
        dashes->add_dash_pair(length, skip);
    }

    // 设置虚线模式对象的起始偏移量
    dashes->set_dash_offset(dash_offset);

    // 返回操作成功的标志
    return 1;
}

// 将 Python 对象转换为 DashesVector 结构
int convert_dashes_vector(PyObject *obj, void *dashesp)
{
    // 将 void 指针转换为 DashesVector 指针
    DashesVector *dashes = (DashesVector *)dashesp;

    // 检查对象是否为序列
    if (!PySequence_Check(obj)) {
        return 0;
    }

    // 获取序列的长度
    Py_ssize_t n = PySequence_Size(obj);

    // 遍历序列中的每个元素
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item;
        Dashes subdashes;

        // 获取序列中的第 i 个元素
        item = PySequence_GetItem(obj, i);
        if (item == NULL) {
            return 0;
        }

        // 将 Python 对象转换为 Dashes 结构
        if (!convert_dashes(item, &subdashes)) {
            Py_DECREF(item);
            return 0;
        }
        Py_DECREF(item);

        // 将转换后的 Dashes 结构加入到 DashesVector 中
        dashes->push_back(subdashes);
    }

    return 1;
}

// 将 Python 对象转换为 agg::trans_affine 结构的仿射变换
int convert_trans_affine(PyObject *obj, void *transp)
{
    // 将 void 指针转换为 agg::trans_affine 指针
    agg::trans_affine *trans = (agg::trans_affine *)transp;

    /** 如果 obj 为 None，则假定为恒等变换 */
    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    // 从任意类型的 Python 对象中获取 NPY_DOUBLE 类型的二维数组
    PyArrayObject *array = (PyArrayObject *)PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 2, 2);
    if (array == NULL) {
        return 0;
    }

    // 检查数组的维度是否为 3x3
    if (PyArray_DIM(array, 0) == 3 && PyArray_DIM(array, 1) == 3) {
        // 获取数组数据的指针
        double *buffer = (double *)PyArray_DATA(array);
        // 将数组中的值赋给 trans_affine 结构的成员变量
        trans->sx = buffer[0];
        trans->shx = buffer[1];
        trans->tx = buffer[2];

        trans->shy = buffer[3];
        trans->sy = buffer[4];
        trans->ty = buffer[5];

        Py_DECREF(array);
        return 1;
    }

    Py_DECREF(array);
    // 设置错误消息，指出仿射变换矩阵无效
    PyErr_SetString(PyExc_ValueError, "Invalid affine transformation matrix");
    return 0;
}

// 将 Python 对象转换为 mpl::PathIterator 结构的路径
int convert_path(PyObject *obj, void *pathp)
{
    // 将 void 指针转换为 mpl::PathIterator 指针
    mpl::PathIterator *path = (mpl::PathIterator *)pathp;

    PyObject *vertices_obj = NULL;
    PyObject *codes_obj = NULL;
    PyObject *should_simplify_obj = NULL;
    PyObject *simplify_threshold_obj = NULL;
    bool should_simplify;
    double simplify_threshold;

    int status = 0;

    // 如果 obj 为 None，则返回成功
    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    // 从对象中获取属性 "vertices"
    vertices_obj = PyObject_GetAttrString(obj, "vertices");
    if (vertices_obj == NULL) {
        goto exit;
    }

    // 从对象中获取属性 "codes"
    codes_obj = PyObject_GetAttrString(obj, "codes");
    if (codes_obj == NULL) {
        goto exit;
    }

    // 从对象中获取属性 "should_simplify"
    should_simplify_obj = PyObject_GetAttrString(obj, "should_simplify");
    if (should_simplify_obj == NULL) {
        goto exit;
    }
    // 根据属性值设置 should_simplify 变量
    switch (PyObject_IsTrue(should_simplify_obj)) {
        case 0: should_simplify = false; break;
        case 1: should_simplify = true; break;
        default: goto exit;  // 发生错误
    }

    // 从对象中获取属性 "simplify_threshold"
    simplify_threshold_obj = PyObject_GetAttrString(obj, "simplify_threshold");
    if (simplify_threshold_obj == NULL) {
        goto exit;
    }
    // 获取属性值并将其转换为 double 类型
    simplify_threshold = PyFloat_AsDouble(simplify_threshold_obj);
    if (PyErr_Occurred()) {
        goto exit;
    }

    // 将属性值应用于 PathIterator 结构
    if (!path->set(vertices_obj, codes_obj, should_simplify, simplify_threshold)) {
        goto exit;
    }

    status = 1;

exit:
    // 清理获取的 Python 对象
    Py_XDECREF(vertices_obj);
    Py_XDECREF(codes_obj);
    Py_XDECREF(should_simplify_obj);
    Py_XDECREF(simplify_threshold_obj);

    return status;
}
// 将 Python 对象转换为 PathGenerator 类型的 C++ 对象
int convert_pathgen(PyObject *obj, void *pathgenp)
{
    // 将 void 指针强制转换为 PathGenerator 指针
    mpl::PathGenerator *paths = (mpl::PathGenerator *)pathgenp;
    // 使用 PathGenerator 对象的 set 方法设置路径，如果失败则设置类型错误异常并返回 0
    if (!paths->set(obj)) {
        PyErr_SetString(PyExc_TypeError, "Not an iterable of paths");
        return 0;
    }
    // 转换成功，返回 1
    return 1;
}

// 将 Python 对象转换为 ClipPath 类型的 C++ 对象
int convert_clippath(PyObject *clippath_tuple, void *clippathp)
{
    // 将 void 指针强制转换为 ClipPath 指针
    ClipPath *clippath = (ClipPath *)clippathp;
    // 创建 PathIterator 和 trans_affine 对象
    mpl::PathIterator path;
    agg::trans_affine trans;

    // 如果 clippath_tuple 不为空且不为 Py_None
    if (clippath_tuple != NULL && clippath_tuple != Py_None) {
        // 使用 PyArg_ParseTuple 解析 clippath_tuple，设置 clippath->path 和 clippath->trans
        if (!PyArg_ParseTuple(clippath_tuple,
                              "O&O&:clippath",
                              &convert_path,
                              &clippath->path,
                              &convert_trans_affine,
                              &clippath->trans)) {
            return 0;  // 解析失败，返回 0
        }
    }

    // 转换成功，返回 1
    return 1;
}

// 将 Python 对象转换为 e_snap_mode 类型的 C++ 对象
int convert_snap(PyObject *obj, void *snapp)
{
    // 将 void 指针强制转换为 e_snap_mode 指针
    e_snap_mode *snap = (e_snap_mode *)snapp;
    // 如果 obj 为空或者为 Py_None，则设置 *snap 为 SNAP_AUTO
    if (obj == NULL || obj == Py_None) {
        *snap = SNAP_AUTO;
    } else {
        // 否则根据 PyObject_IsTrue(obj) 的返回值设置 *snap 的值
        switch (PyObject_IsTrue(obj)) {
            case 0: *snap = SNAP_FALSE; break;
            case 1: *snap = SNAP_TRUE; break;
            default: return 0;  // 出错，返回 0
        }
    }
    // 转换成功，返回 1
    return 1;
}

// 将 Python 对象转换为 SketchParams 类型的 C++ 对象
int convert_sketch_params(PyObject *obj, void *sketchp)
{
    // 将 void 指针强制转换为 SketchParams 指针
    SketchParams *sketch = (SketchParams *)sketchp;

    // 如果 obj 为空或者为 Py_None，则设置 sketch->scale 为 0.0
    if (obj == NULL || obj == Py_None) {
        sketch->scale = 0.0;
    } else if (!PyArg_ParseTuple(obj,
                                 "ddd:sketch_params",
                                 &sketch->scale,
                                 &sketch->length,
                                 &sketch->randomness)) {
        return 0;  // 解析失败，返回 0
    }

    // 转换成功，返回 1
    return 1;
}

// 将 Python 对象转换为 GCAgg 类型的 C++ 对象
int convert_gcagg(PyObject *pygc, void *gcp)
{
    // 将 void 指针强制转换为 GCAgg 指针
    GCAgg *gc = (GCAgg *)gcp;
    // 检查是否成功从Python对象pygc中获取以下属性，并将它们转换为相应的C数据结构，存储到对应的gc结构体字段中：
    // - "_linewidth" 转换为 double 类型，存储到 gc->linewidth
    // - "_alpha" 转换为 double 类型，存储到 gc->alpha
    // - "_forced_alpha" 转换为 bool 类型，存储到 gc->forced_alpha
    // - "_rgb" 转换为 RGBA 类型，存储到 gc->color
    // - "_antialiased" 转换为 bool 类型，存储到 gc->isaa
    // - "_capstyle" 转换为 capstyle 类型，存储到 gc->cap
    // - "_joinstyle" 转换为 joinstyle 类型，存储到 gc->join
    // - "get_dashes" 调用获取虚线样式，转换为 dashes 类型，存储到 gc->dashes
    // - "_cliprect" 转换为 rect 类型，存储到 gc->cliprect
    // - "get_clip_path" 调用获取剪裁路径，转换为 clippath 类型，存储到 gc->clippath
    // - "get_snap" 调用获取捕捉模式，转换为 snap_mode 类型，存储到 gc->snap_mode
    // - "get_hatch_path" 调用获取填充图案路径，转换为 path 类型，存储到 gc->hatchpath
    // - "get_hatch_color" 调用获取填充图案颜色，转换为 RGBA 类型，存储到 gc->hatch_color
    // - "get_hatch_linewidth" 调用获取填充图案线宽，转换为 double 类型，存储到 gc->hatch_linewidth
    // - "get_sketch_params" 调用获取草图参数，转换为 sketch_params 类型，存储到 gc->sketch
    // 如果任何一个转换失败，则返回0表示失败；否则返回1表示成功。
    if (!(convert_from_attr(pygc, "_linewidth", &convert_double, &gc->linewidth) &&
          convert_from_attr(pygc, "_alpha", &convert_double, &gc->alpha) &&
          convert_from_attr(pygc, "_forced_alpha", &convert_bool, &gc->forced_alpha) &&
          convert_from_attr(pygc, "_rgb", &convert_rgba, &gc->color) &&
          convert_from_attr(pygc, "_antialiased", &convert_bool, &gc->isaa) &&
          convert_from_attr(pygc, "_capstyle", &convert_cap, &gc->cap) &&
          convert_from_attr(pygc, "_joinstyle", &convert_join, &gc->join) &&
          convert_from_method(pygc, "get_dashes", &convert_dashes, &gc->dashes) &&
          convert_from_attr(pygc, "_cliprect", &convert_rect, &gc->cliprect) &&
          convert_from_method(pygc, "get_clip_path", &convert_clippath, &gc->clippath) &&
          convert_from_method(pygc, "get_snap", &convert_snap, &gc->snap_mode) &&
          convert_from_method(pygc, "get_hatch_path", &convert_path, &gc->hatchpath) &&
          convert_from_method(pygc, "get_hatch_color", &convert_rgba, &gc->hatch_color) &&
          convert_from_method(pygc, "get_hatch_linewidth", &convert_double, &gc->hatch_linewidth) &&
          convert_from_method(pygc, "get_sketch_params", &convert_sketch_params, &gc->sketch))) {
        // 如果转换失败，返回0表示失败
        return 0;
    }
    
    // 转换成功，返回1表示成功
    return 1;
# 结束了前一个函数的定义，即 convert_face 函数
}

# 将 Python 对象 color 转换为 RGBA 颜色值，存储到 rgba 中
int convert_face(PyObject *color, GCAgg &gc, agg::rgba *rgba)
{
    # 如果无法将 color 转换为 RGBA 颜色值，则返回 0 表示失败
    if (!convert_rgba(color, rgba)) {
        return 0;
    }

    # 如果 color 不为空且不是 None
    if (color != NULL && color != Py_None) {
        # 如果强制使用 alpha 通道或者 color 是长度为 3 的序列
        if (gc.forced_alpha || PySequence_Size(color) == 3) {
            # 将 gc.alpha 的值赋给 rgba 的 alpha 通道
            rgba->a = gc.alpha;
        }
    }

    # 返回 1 表示成功转换
    return 1;
}

# 将 Python 对象 obj 转换为点集，存储到 pointsp 中
int convert_points(PyObject *obj, void *pointsp)
{
    # 将 pointsp 强制转换为 numpy::array_view<double, 2> 指针
    numpy::array_view<double, 2> *points = (numpy::array_view<double, 2> *)pointsp;
    # 如果 obj 是 NULL 或者 None，则返回 1 表示成功
    if (obj == NULL || obj == Py_None) {
        return 1;
    }
    # 尝试将 obj 设置为 points 的值，如果失败或者校验尺寸不通过则返回 0
    if (!points->set(obj)
        || (points->size() && !check_trailing_shape(*points, "points", 2))) {
        return 0;
    }
    # 返回 1 表示成功转换
    return 1;
}

# 将 Python 对象 obj 转换为变换矩阵，存储到 transp 中
int convert_transforms(PyObject *obj, void *transp)
{
    # 将 transp 强制转换为 numpy::array_view<double, 3> 指针
    numpy::array_view<double, 3> *trans = (numpy::array_view<double, 3> *)transp;
    # 如果 obj 是 NULL 或者 None，则返回 1 表示成功
    if (obj == NULL || obj == Py_None) {
        return 1;
    }
    # 尝试将 obj 设置为 trans 的值，如果失败或者校验尺寸不通过则返回 0
    if (!trans->set(obj)
        || (trans->size() && !check_trailing_shape(*trans, "transforms", 3, 3))) {
        return 0;
    }
    # 返回 1 表示成功转换
    return 1;
}

# 将 Python 对象 obj 转换为边界框数组，存储到 bboxp 中
int convert_bboxes(PyObject *obj, void *bboxp)
{
    # 将 bboxp 强制转换为 numpy::array_view<double, 3> 指针
    numpy::array_view<double, 3> *bbox = (numpy::array_view<double, 3> *)bboxp;
    # 如果 obj 是 NULL 或者 None，则返回 1 表示成功
    if (obj == NULL || obj == Py_None) {
        return 1;
    }
    # 尝试将 obj 设置为 bbox 的值，如果失败或者校验尺寸不通过则返回 0
    if (!bbox->set(obj)
        || (bbox->size() && !check_trailing_shape(*bbox, "bbox array", 2, 2))) {
        return 0;
    }
    # 返回 1 表示成功转换
    return 1;
}

# 将 Python 对象 obj 转换为颜色数组，存储到 colorsp 中
int convert_colors(PyObject *obj, void *colorsp)
{
    # 将 colorsp 强制转换为 numpy::array_view<double, 2> 指针
    numpy::array_view<double, 2> *colors = (numpy::array_view<double, 2> *)colorsp;
    # 如果 obj 是 NULL 或者 None，则返回 1 表示成功
    if (obj == NULL || obj == Py_None) {
        return 1;
    }
    # 尝试将 obj 设置为 colors 的值，如果失败或者校验尺寸不通过则返回 0
    if (!colors->set(obj)
        || (colors->size() && !check_trailing_shape(*colors, "colors", 4))) {
        return 0;
    }
    # 返回 1 表示成功转换
    return 1;
}
```