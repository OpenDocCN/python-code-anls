# `D:\src\scipysrc\matplotlib\src\_path.h`

```py
// C++ 头文件保护开始，防止多次包含该头文件
#ifndef MPL_PATH_H
#define MPL_PATH_H

// 包含必要的 C++ 头文件
#include <limits>
#include <math.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// 包含相关的自定义头文件
#include "agg_conv_contour.h"
#include "agg_conv_curve.h"
#include "agg_conv_stroke.h"
#include "agg_conv_transform.h"
#include "agg_trans_affine.h"

#include "path_converters.h"
#include "_backend_agg_basic_types.h"
#include "numpy_cpp.h"

// 定义一个常量数组，存储不同多边形的顶点数
const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3 };

// 定义一个结构体 XY，表示二维坐标点
struct XY
{
    double x;
    double y;

    // XY 结构体的构造函数，初始化 x 和 y 值
    XY(double x_, double y_) : x(x_), y(y_)
    {
    }

    // 重载 == 运算符，判断两个 XY 结构体是否相等
    bool operator==(const XY& o)
    {
        return (x == o.x && y == o.y);
    }

    // 重载 != 运算符，判断两个 XY 结构体是否不相等
    bool operator!=(const XY& o)
    {
        return (x != o.x || y != o.y);
    }
};

// 定义一个类型别名 Polygon，表示 XY 结构体的向量，用于存储多边形的顶点
typedef std::vector<XY> Polygon;

// 定义一个函数 _finalize_polygon，用于清理多边形向量的最后一个多边形
void _finalize_polygon(std::vector<Polygon> &result, int closed_only)
{
    // 如果结果向量为空，直接返回
    if (result.size() == 0) {
        return;
    }

    // 获取结果向量中的最后一个多边形
    Polygon &polygon = result.back();

    // 清理结果向量中最后一个多边形的函数
    if (polygon.size() == 0) {
        result.pop_back();
    } else if (closed_only) {
        if (polygon.size() < 3) {
            result.pop_back();
        } else if (polygon.front() != polygon.back()) {
            polygon.push_back(polygon.front());
        }
    }
}

//
// 下面的函数是从 Agg 2.3 示例 (interactive_polygon.cpp) 中找到的。
// 它已被泛化以处理（可能是曲线的）折线，而不仅仅是多边形。
// 原始注释已被保留。
// -- Michael Droettboom 2007-10-02
//

//======= Crossings Multiply algorithm of InsideTest ========================
//
// 由 Eric Haines, 3D/Eye Inc, erich@eye.com 提供
//
// 这个版本通常比《Graphics Gems IV》中原始版本更快；
// 通过将用于测试 X 轴交叉的除法转换为一个巧妙的乘法测试，这部分测试变得更快，
// 这也使得测试“都在左侧或都在右侧”的部分对于三角形而言比每次计算交点更慢。
// 主要的提速在于三角形测试，速度大约快了 15%；
// 所有其他多边形复杂性几乎与以前相同。
// 在除法非常昂贵的机器上（我测试的 HP 9000 系列并非如此），这个测试应该比旧代码总体上快得多。
// 您的表现可能会有所不同，取决于机器和测试数据，但总体上我认为这段代码既更短又更快。
// 这个测试的灵感来自于 Joseph Samosky 和 Mark Haigh-Hutchinson 提交的未发表的《Graphics Gems》。
// Samosky 的相关工作见：
//
// Samosky, Joseph，“SectionView: A system for interactively specifying and
// visualizing sections through three-dimensional medical image data”,
// M.S. Thesis, Department of Electrical Engineering and Computer Science,
// Massachusetts Institute of Technology, 1993.
//
// Shoot a test ray along +X axis. The strategy is to compare vertex Y values
// to the testing point's Y and quickly discard edges which are entirely to one
// side of the test ray. Note that CONVEX and WINDING code can be added as
// for the CrossingsTest() code; it is left out here for clarity.
//
// Input 2D polygon _pgon_ with _numverts_ number of vertices and test point
// _point_, returns 1 if inside, 0 if outside.
template <class PathIterator, class PointArray, class ResultArray>
void point_in_path_impl(PointArray &points, PathIterator &path, ResultArray &inside_flag)
{
    uint8_t yflag1;  // Variable to store y-flag for the current vertex
    double vtx0, vty0, vtx1, vty1;  // Variables to store vertex coordinates
    double tx, ty;  // Variables to store test point coordinates
    double sx, sy;  // Variables to store segment endpoint coordinates
    double x, y;  // General purpose variables for coordinates
    size_t i;  // Loop index variable
    bool all_done;  // Flag to indicate if all segments are processed

    // Determine the number of shapes in the polygon and initialize
    // y-flag and subpath flag arrays
    size_t n = safe_first_shape(points);
    std::vector<uint8_t> yflag0(n);  // Array to store y-flags for vertices
    std::vector<uint8_t> subpath_flag(n);  // Array to store subpath flags

    // Initialize the path iterator to start at the beginning
    path.rewind(0);

    // Initialize inside_flag array to 0 for all vertices
    for (i = 0; i < n; ++i) {
        inside_flag[i] = 0;
    }

    unsigned code = 0;  // Initialize the code variable for path commands

    // Loop through the path commands until reaching the end
    } while (code != agg::path_cmd_stop);
}

template <class PathIterator, class PointArray, class ResultArray>
inline void points_in_path(PointArray &points,
                           const double r,
                           PathIterator &path,
                           agg::trans_affine &trans,
                           ResultArray &result)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> no_nans_t;
    typedef agg::conv_curve<no_nans_t> curve_t;
    typedef agg::conv_contour<curve_t> contour_t;

    size_t i;  // Loop index variable

    // Initialize result array to false for all vertices
    for (i = 0; i < safe_first_shape(points); ++i) {
        result[i] = false;
    }

    // If the total number of vertices in the path is less than 3, return early
    if (path.total_vertices() < 3) {
        return;
    }

    // Transform the path using the provided affine transformation
    transformed_path_t trans_path(path, trans);
    no_nans_t no_nans_path(trans_path, true, path.has_codes());
    curve_t curved_path(no_nans_path);

    // If a non-zero radius is provided, apply contouring to the curved path
    if (r != 0.0) {
        contour_t contoured_path(curved_path);
        contoured_path.width(r);
        point_in_path_impl(points, contoured_path, result);
    } else {
        point_in_path_impl(points, curved_path, result);
    }
}

template <class PathIterator>
inline bool point_in_path(
    double x, double y, const double r, PathIterator &path, agg::trans_affine &trans)
{
    npy_intp shape[] = {1, 2};  // Shape array for numpy array view
    numpy::array_view<double, 2> points(shape);  // Numpy array view for point coordinates
    points(0, 0) = x;  // Set x-coordinate of the point
    points(0, 1) = y;  // Set y-coordinate of the point

    int result[1];  // Array to store result of point in path test
    result[0] = 0;  // Initialize result to 0

    // Check if the point lies inside the path and store result in 'result'
    points_in_path(points, r, path, trans, result);

    // Return true if the point is inside the path, false otherwise
    return result[0] != 0;
}

template <class PathIterator>
inline bool point_on_path(
    double x, double y, const double r, PathIterator &path, agg::trans_affine &trans)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> no_nans_t;
    typedef agg::conv_curve<no_nans_t> curve_t;
    typedef agg::conv_stroke<curve_t> stroke_t;

    npy_intp shape[] = {1, 2};  // Shape array for numpy array view
    numpy::array_view<double, 2> points(shape);  // Numpy array view for point coordinates
    points(0, 0) = x;  // Set x-coordinate of the point
    points(0, 1) = y;  // Set y-coordinate of the point

    int result[1];  // Array to store result of point on path test
    # 将 result 数组的第一个元素设为 0
    result[0] = 0;

    # 使用 transformed_path_t 类型对象 trans_path 对 path 应用变换 trans
    transformed_path_t trans_path(path, trans);
    
    # 创建 no_nans_t 类型对象 nan_removed_path，移除其中的 NaN 值，并根据 path 的编码情况决定是否保留编码
    no_nans_t nan_removed_path(trans_path, true, path.has_codes());
    
    # 使用 curved_path 类型对象 curved_path 对 nan_removed_path 进行曲线处理
    curve_t curved_path(nan_removed_path);
    
    # 使用 stroke_t 类型对象 stroked_path 对 curved_path 进行描边处理
    stroke_t stroked_path(curved_path);
    
    # 设置 stroked_path 的宽度为 r 的两倍
    stroked_path.width(r * 2.0);
    
    # 使用 point_in_path_impl 函数计算 points 是否在 stroked_path 路径内，并将结果写入 result 数组的第一个元素
    point_in_path_impl(points, stroked_path, result);
    
    # 返回判断结果，即 result 数组的第一个元素是否不为 0
    return result[0] != 0;
}

// 定义了一个结构体 extent_limits，用于存储数据的边界限制
struct extent_limits
{
    double x0;    // x 轴的最小值
    double y0;    // y 轴的最小值
    double x1;    // x 轴的最大值
    double y1;    // y 轴的最大值
    double xm;    // 数据中的最小正值，用于对数缩放
    double ym;    // 数据中的最小正值，用于对数缩放
};

// 重置边界限制对象 e 的各项值
void reset_limits(extent_limits &e)
{
    e.x0 = std::numeric_limits<double>::infinity();    // 初始化 x0 为正无穷大
    e.y0 = std::numeric_limits<double>::infinity();    // 初始化 y0 为正无穷大
    e.x1 = -std::numeric_limits<double>::infinity();   // 初始化 x1 为负无穷大
    e.y1 = -std::numeric_limits<double>::infinity();   // 初始化 y1 为负无穷大
    /* xm 和 ym 是数据中的最小正值，用于对数缩放 */
    e.xm = std::numeric_limits<double>::infinity();    // 初始化 xm 为正无穷大
    e.ym = std::numeric_limits<double>::infinity();    // 初始化 ym 为正无穷大
}

// 更新边界限制对象 e 的值，根据给定的 x 和 y 值
inline void update_limits(double x, double y, extent_limits &e)
{
    if (x < e.x0)    // 更新 x0，记录最小的 x 值
        e.x0 = x;
    if (y < e.y0)    // 更新 y0，记录最小的 y 值
        e.y0 = y;
    if (x > e.x1)    // 更新 x1，记录最大的 x 值
        e.x1 = x;
    if (y > e.y1)    // 更新 y1，记录最大的 y 值
        e.y1 = y;
    /* xm 和 ym 是数据中的最小正值，用于对数缩放 */
    if (x > 0.0 && x < e.xm)    // 更新 xm，记录最小的正 x 值
        e.xm = x;
    if (y > 0.0 && y < e.ym)    // 更新 ym，记录最小的正 y 值
        e.ym = y;
}

// 更新路径的边界限制，通过遍历路径中的顶点来更新
template <class PathIterator>
void update_path_extents(PathIterator &path, agg::trans_affine &trans, extent_limits &extents)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;    // 转换后的路径类型
    typedef PathNanRemover<transformed_path_t> nan_removed_t;         // 去除 NaN 值的路径类型
    double x, y;
    unsigned code;

    transformed_path_t tpath(path, trans);    // 创建转换后的路径对象
    nan_removed_t nan_removed(tpath, true, path.has_codes());    // 创建去除 NaN 值的路径对象

    nan_removed.rewind(0);    // 从路径起始处开始遍历

    while ((code = nan_removed.vertex(&x, &y)) != agg::path_cmd_stop) {
        if ((code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
            continue;    // 忽略结束多边形的顶点
        }
        update_limits(x, y, extents);    // 更新边界限制对象 extents
    }
}

// 获取路径集合的整体边界限制，考虑路径的变换和偏移
template <class PathGenerator, class TransformArray, class OffsetArray>
void get_path_collection_extents(agg::trans_affine &master_transform,
                                 PathGenerator &paths,
                                 TransformArray &transforms,
                                 OffsetArray &offsets,
                                 agg::trans_affine &offset_trans,
                                 extent_limits &extent)
{
    if (offsets.size() != 0 && offsets.shape(1) != 2) {
        throw std::runtime_error("Offsets array must have shape (N, 2)");    // 检查偏移数组的形状是否正确
    }

    size_t Npaths = paths.size();    // 获取路径的数量
    size_t Noffsets = safe_first_shape(offsets);    // 获取偏移数组的大小
    size_t N = std::max(Npaths, Noffsets);    // 计算最大的路径数和偏移数
    size_t Ntransforms = std::min(safe_first_shape(transforms), N);    // 计算有效的变换数量
    size_t i;

    agg::trans_affine trans;    // 变换对象

    reset_limits(extent);    // 重置边界限制对象 extent 的值
    // 循环遍历 N 次，执行以下操作
    for (i = 0; i < N; ++i) {
        // 使用 PathGenerator 的 path_iterator 创建路径迭代器 path，从 paths 数组中选择第 i % Npaths 个路径
        typename PathGenerator::path_iterator path(paths(i % Npaths));

        // 如果 Ntransforms 大于 0，则根据 transforms 数组的第 i % Ntransforms 行创建仿射变换 trans
        if (Ntransforms) {
            size_t ti = i % Ntransforms;
            // 从 transforms 数组中获取仿射变换的参数，并创建对应的仿射变换对象 trans
            trans = agg::trans_affine(transforms(ti, 0, 0),
                                      transforms(ti, 1, 0),
                                      transforms(ti, 0, 1),
                                      transforms(ti, 1, 1),
                                      transforms(ti, 0, 2),
                                      transforms(ti, 1, 2));
        } else {
            // 如果 Ntransforms 为 0，则使用 master_transform 作为仿射变换 trans
            trans = master_transform;
        }

        // 如果 Noffsets 大于 0，则获取 offsets 数组的第 i % Noffsets 行的偏移量，并应用到 trans 上
        if (Noffsets) {
            double xo = offsets(i % Noffsets, 0);
            double yo = offsets(i % Noffsets, 1);
            // 将偏移量 xo, yo 应用到 offset_trans 上
            offset_trans.transform(&xo, &yo);
            // 在当前的仿射变换 trans 上叠加一个平移变换，以偏移量 xo, yo
            trans *= agg::trans_affine_translation(xo, yo);
        }

        // 更新路径 path 的边界范围，应用仿射变换 trans，并更新 extent
        update_path_extents(path, trans, extent);
    }
// 将点 (x, y) 以半径 radius 放置在路径集合中，根据条件判断路径是否包含点
template <class PathGenerator, class TransformArray, class OffsetArray>
void point_in_path_collection(double x,
                              double y,
                              double radius,
                              agg::trans_affine &master_transform,
                              PathGenerator &paths,
                              TransformArray &transforms,
                              OffsetArray &offsets,
                              agg::trans_affine &offset_trans,
                              bool filled,
                              std::vector<int> &result)
{
    // 获取路径集合的大小
    size_t Npaths = paths.size();

    // 如果路径集合为空，直接返回
    if (Npaths == 0) {
        return;
    }

    // 获取偏移量集合的大小
    size_t Noffsets = safe_first_shape(offsets);
    // 计算最大的迭代次数，取路径集合和偏移量集合大小的最大值
    size_t N = std::max(Npaths, Noffsets);
    // 获取变换集合的大小，并且不能超过 N 的大小
    size_t Ntransforms = std::min(safe_first_shape(transforms), N);
    // 迭代变量 i
    size_t i;

    // 变换对象
    agg::trans_affine trans;

    // 遍历所有路径
    for (i = 0; i < N; ++i) {
        // 获取当前路径的迭代器
        typename PathGenerator::path_iterator path = paths(i % Npaths);

        // 如果有变换操作
        if (Ntransforms) {
            // 获取当前变换的索引
            size_t ti = i % Ntransforms;
            // 构造变换矩阵
            trans = agg::trans_affine(transforms(ti, 0, 0),
                                      transforms(ti, 1, 0),
                                      transforms(ti, 0, 1),
                                      transforms(ti, 1, 1),
                                      transforms(ti, 0, 2),
                                      transforms(ti, 1, 2));
            // 将当前变换与主变换组合
            trans *= master_transform;
        } else {
            // 否则直接使用主变换
            trans = master_transform;
        }

        // 如果有偏移量
        if (Noffsets) {
            // 获取当前偏移量的索引
            double xo = offsets(i % Noffsets, 0);
            double yo = offsets(i % Noffsets, 1);
            // 应用偏移变换
            offset_trans.transform(&xo, &yo);
            trans *= agg::trans_affine_translation(xo, yo);
        }

        // 如果需要填充操作
        if (filled) {
            // 判断点 (x, y) 是否在路径上，并将结果存入 result
            if (point_in_path(x, y, radius, path, trans)) {
                result.push_back(i);
            }
        } else {
            // 判断点 (x, y) 是否在路径上的边界，并将结果存入 result
            if (point_on_path(x, y, radius, path, trans)) {
                result.push_back(i);
            }
        }
    }
}

// 判断路径 a 是否完全包含在路径 b 中
template <class PathIterator1, class PathIterator2>
bool path_in_path(PathIterator1 &a,
                  agg::trans_affine &atrans,
                  PathIterator2 &b,
                  agg::trans_affine &btrans)
{
    // 定义转换后的路径类型
    typedef agg::conv_transform<PathIterator2> transformed_path_t;
    // 定义去除 NaN 点后的路径类型
    typedef PathNanRemover<transformed_path_t> no_nans_t;
    // 定义曲线类型
    typedef agg::conv_curve<no_nans_t> curve_t;

    // 如果路径 a 的顶点数小于 3，直接返回 false
    if (a.total_vertices() < 3) {
        return false;
    }

    // 转换路径 b
    transformed_path_t b_path_trans(b, btrans);
    // 去除 NaN 点
    no_nans_t b_no_nans(b_path_trans, true, b.has_codes());
    // 创建曲线对象
    curve_t b_curved(b_no_nans);

    // 定义变量 x, y
    double x, y;
    // 重置曲线对象
    b_curved.rewind(0);
    // 遍历路径 b 的所有顶点
    while (b_curved.vertex(&x, &y) != agg::path_cmd_stop) {
        // 如果路径 a 中的某个点 (x, y) 不在路径 b 中，返回 false
        if (!point_in_path(x, y, 0.0, a, atrans)) {
            return false;
        }
    }

    // 如果路径 a 完全包含在路径 b 中，返回 true
    return true;
}
    # 使用 Sutherland-Hodgman 裁剪算法进行裁剪操作
/*
namespace clip_to_rect_filters
{
/* There are four different passes needed to create/remove
   vertices (one for each side of the rectangle).  The differences
   between those passes are encapsulated in these functor classes.
*/

// 定义用于矩形裁剪的四个不同通行（每个矩形边一个），这些差异封装在以下几个函数类中。

// 用于在 x 轴上进行裁剪的函数对象类
struct bisectx
{
    double m_x;

    bisectx(double x) : m_x(x)
    {
    }

    // 计算与 x 轴相交的点的坐标
    inline void bisect(double sx, double sy, double px, double py, double *bx, double *by) const
    {
        *bx = m_x;
        double dx = px - sx;
        double dy = py - sy;
        *by = sy + dy * ((m_x - sx) / dx);
    }
};

// 在 x 小于等于给定值时返回 true 的函数对象类
struct xlt : public bisectx
{
    xlt(double x) : bisectx(x)
    {
    }

    // 检查给定点是否在 x 小于等于 m_x 的范围内
    inline bool is_inside(double x, double y) const
    {
        return x <= m_x;
    }
};

// 在 x 大于等于给定值时返回 true 的函数对象类
struct xgt : public bisectx
{
    xgt(double x) : bisectx(x)
    {
    }

    // 检查给定点是否在 x 大于等于 m_x 的范围内
    inline bool is_inside(double x, double y) const
    {
        return x >= m_x;
    }
};

// 用于在 y 轴上进行裁剪的函数对象类
struct bisecty
{
    double m_y;

    bisecty(double y) : m_y(y)
    {
    }

    // 计算与 y 轴相交的点的坐标
    inline void bisect(double sx, double sy, double px, double py, double *bx, double *by) const
    {
        *by = m_y;
        double dx = px - sx;
        double dy = py - sy;
        *bx = sx + dx * ((m_y - sy) / dy);
    }
};

// 在 y 小于等于给定值时返回 true 的函数对象类
struct ylt : public bisecty
{
    ylt(double y) : bisecty(y)
    {
    }

    // 检查给定点是否在 y 小于等于 m_y 的范围内
    inline bool is_inside(double x, double y) const
    {
        return y <= m_y;
    }
};

// 在 y 大于等于给定值时返回 true 的函数对象类
struct ygt : public bisecty
{
    ygt(double y) : bisecty(y)
    {
    }

    // 检查给定点是否在 y 大于等于 m_y 的范围内
    inline bool is_inside(double x, double y) const
    {
        return y >= m_y;
    }
};
}
*/

// 使用指定的过滤器对多边形进行一次矩形裁剪的模板函数
template <class Filter>
inline void clip_to_rect_one_step(const Polygon &polygon, Polygon &result, const Filter &filter)
{
    double sx, sy, px, py, bx, by;
    bool sinside, pinside;
    result.clear();

    if (polygon.size() == 0) {
        return;
    }

    sx = polygon.back().x;  // 获取多边形最后一个顶点的 x 坐标
    sy = polygon.back().y;  // 获取多边形最后一个顶点的 y 坐标
    for (Polygon::const_iterator i = polygon.begin(); i != polygon.end(); ++i) {
        px = i->x;  // 获取当前顶点的 x 坐标
        py = i->y;  // 获取当前顶点的 y 坐标

        sinside = filter.is_inside(sx, sy);  // 检查起始点是否在裁剪区域内
        pinside = filter.is_inside(px, py);  // 检查当前点是否在裁剪区域内

        if (sinside ^ pinside) {
            filter.bisect(sx, sy, px, py, &bx, &by);  // 计算裁剪边界与多边形边的交点坐标
            result.push_back(XY(bx, by));  // 将交点坐标添加到裁剪结果中
        }

        if (pinside) {
            result.push_back(XY(px, py));  // 将当前顶点添加到裁剪结果中
        }

        sx = px;  // 更新起始点的 x 坐标为当前顶点的 x 坐标
        sy = py;  // 更新起始点的 y 坐标为当前顶点的 y 坐标
    }
}

// 将路径对象裁剪到指定矩形区域内的模板函数
template <class PathIterator>
void clip_path_to_rect(PathIterator &path, agg::rect_d &rect, bool inside, std::vector<Polygon> &results)
{
    double xmin, ymin, xmax, ymax;
    if (rect.x1 < rect.x2) {
        xmin = rect.x1;  // 确定矩形左边界的 x 坐标
        xmax = rect.x2;  // 确定矩形右边界的 x 坐标
    } else {
        xmin = rect.x2;  // 确定矩形左边界的 x 坐标
        xmax = rect.x1;  // 确定矩形右边界的 x 坐标
    }

    if (rect.y1 < rect.y2) {
        ymin = rect.y1;  // 确定矩形上边界的 y 坐标
        ymax = rect.y2;  // 确定矩形下边界的 y 坐标
    } else {
        ymin = rect.y2;  // 确定矩形上边界的 y 坐标
        ymax = rect.y1;  // 确定矩形下边界的 y 坐标
    }

    if (!inside) {
        std::swap(xmin, xmax);  // 如果不在矩形内部，交换左右边界的坐标值
        std::swap(ymin, ymax);  // 如果不在矩形内部，交换上下边界的坐标值
    }

    typedef agg::conv_curve<PathIterator> curve_t;
    curve_t curve(path);  // 创建曲线转换器对象

    Polygon polygon1, polygon2;  // 定义两个多边形对象
    double x = 0, y = 0;
}
    # 初始化一个无符号整数 code，用于存储路径命令
    unsigned code = 0;
    # 重置曲线到起始点
    curve.rewind(0);

    # 循环处理路径中的每个子路径
    do {
        # 清空 polygon1，准备存储下一个子路径的顶点
        polygon1.clear();
        
        # 读取曲线的下一个顶点，并存储到 polygon1 中
        do {
            # 如果是移动到新起点的命令，将起点添加到 polygon1 中
            if (code == agg::path_cmd_move_to) {
                polygon1.push_back(XY(x, y));
            }

            # 获取下一个顶点的命令代码和坐标
            code = curve.vertex(&x, &y);

            # 如果是结束命令，结束当前子路径的读取
            if (code == agg::path_cmd_stop) {
                break;
            }

            # 如果不是移动命令，将顶点添加到 polygon1 中
            if (code != agg::path_cmd_move_to) {
                polygon1.push_back(XY(x, y));
            }
        } while ((code & agg::path_cmd_end_poly) != agg::path_cmd_end_poly);

        # 对 polygon1 执行一系列裁剪操作，将结果存储在 results 中
        clip_to_rect_one_step(polygon1, polygon2, clip_to_rect_filters::xlt(xmax));
        clip_to_rect_one_step(polygon2, polygon1, clip_to_rect_filters::xgt(xmin));
        clip_to_rect_one_step(polygon1, polygon2, clip_to_rect_filters::ylt(ymax));
        clip_to_rect_one_step(polygon2, polygon1, clip_to_rect_filters::ygt(ymin));

        # 如果 polygon1 非空，将其最终结果存储到 results 中
        if (polygon1.size()) {
            _finalize_polygon(results, 1);
            results.push_back(polygon1);
        }
    } while (code != agg::path_cmd_stop);

    # 完成最后一个多边形的处理，并将其存储到 results 中
    _finalize_polygon(results, 1);
// 计算两条线段是否相交，根据给定的四个端点坐标
inline bool segments_intersect(const double &x1,
                               const double &y1,
                               const double &x2,
                               const double &y2,
                               const double &x3,
                               const double &y3,
                               const double &x4,
                               const double &y4)
{
    // 计算行列式的值，判断是否存在相交
    double den = ((y4 - y3) * (x2 - x1)) - ((x4 - x3) * (y2 - y1));

    // 如果 den == 0，有两种可能性：
    if (den == 0) {
        // 情况1：线段平行或重合
        // 检查端点是否共线，以及端点是否在另一条线段上
        if (((y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1)) &&
            ((y2 - y1) * (x4 - x1) == (y4 - y1) * (x2 - x1))) {
            // 线段重合
            return true;
        } else {
            // 线段平行但不重合
            return false;
        }
    }

    // 计算 t 和 u 的值，判断交点是否在两条线段内
    double t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den;
    double u = -((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den;

    // 如果 t 和 u 在 [0, 1] 范围内，则表示两条线段相交
    return (t >= 0 && t <= 1 && u >= 0 && u <= 1);
}
    // 1 - 如果通过前三个点组成的三角形的面积为零，表示它们共线
    if (isclose(den, 0.0)) {
        // 计算前三个点组成的三角形的面积
        double t_area = (x2*y3 - x3*y2) - x1*(y3 - y2) + y1*(x3 - x2);
        
        // 如果三角形的面积为零，则判断线段是否共线
        if (isclose(t_area, 0.0)) {
            // 如果 x1、x2、x3 相等，说明线段具有无限的斜率（垂直线），且在同一条直线上
            return (fmin(y1, y2) <= fmin(y3, y4) && fmin(y3, y4) <= fmax(y1, y2)) ||
                    (fmin(y3, y4) <= fmin(y1, y2) && fmin(y1, y2) <= fmax(y3, y4));
        }
        // 2 - 如果 t_area 不为零，则说明线段平行但不共线
        else {
            return false;
        }
    }

    // 计算两个分子 n1 和 n2
    const double n1 = ((x4 - x3) * (y1 - y3)) - ((y4 - y3) * (x1 - x3));
    const double n2 = ((x2 - x1) * (y1 - y3)) - ((y2 - y1) * (x1 - x3));

    // 计算参数 u1 和 u2
    const double u1 = n1 / den;
    const double u2 = n2 / den;

    // 检查参数 u1 和 u2 是否在 [0, 1] 范围内
    return ((u1 > 0.0 || isclose(u1, 0.0)) &&
            (u1 < 1.0 || isclose(u1, 1.0)) &&
            (u2 > 0.0 || isclose(u2, 0.0)) &&
            (u2 < 1.0 || isclose(u2, 1.0)));
// 定义一个模板函数，判断两条路径是否相交
template <class PathIterator1, class PathIterator2>
bool path_intersects_path(PathIterator1 &p1, PathIterator2 &p2)
{
    // 定义去除 NaN 点的类型
    typedef PathNanRemover<mpl::PathIterator> no_nans_t;
    // 定义转换成曲线的类型
    typedef agg::conv_curve<no_nans_t> curve_t;

    // 如果其中一条路径的顶点少于两个，直接返回不相交
    if (p1.total_vertices() < 2 || p2.total_vertices() < 2) {
        return false;
    }

    // 创建去除 NaN 点后的路径对象
    no_nans_t n1(p1, true, p1.has_codes());
    no_nans_t n2(p2, true, p2.has_codes());

    // 用去除 NaN 点后的路径对象创建曲线对象
    curve_t c1(n1);
    curve_t c2(n2);

    // 定义曲线段的端点坐标
    double x11, y11, x12, y12;
    double x21, y21, x22, y22;

    // 获取曲线 c1 的起始点坐标
    c1.vertex(&x11, &y11);
    // 遍历曲线 c1 的所有段
    while (c1.vertex(&x12, &y12) != agg::path_cmd_stop) {
        // 如果路径 c1 的当前线段几乎为零长度，则跳过到下一个顶点
        if ((isclose((x11 - x12) * (x11 - x12) + (y11 - y12) * (y11 - y12), 0))){
            continue;
        }
        // 重置曲线 c2 到起始位置
        c2.rewind(0);
        // 获取曲线 c2 的起始点坐标
        c2.vertex(&x21, &y21);

        // 遍历曲线 c2 的所有段
        while (c2.vertex(&x22, &y22) != agg::path_cmd_stop) {
            // 如果路径 c2 的当前线段几乎为零长度，则跳过到下一个顶点
            if ((isclose((x21 - x22) * (x21 - x22) + (y21 - y22) * (y21 - y22), 0))){
                continue;
            }

            // 检查两条线段是否相交
            if (segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22)) {
                return true;  // 如果相交，返回 true
            }
            // 更新 c2 的当前端点坐标
            x21 = x22;
            y21 = y22;
        }
        // 更新 c1 的当前端点坐标
        x11 = x12;
        y11 = y12;
    }

    return false;  // 如果所有线段都不相交，返回 false
}

// 判断线段 (x1,y1)-(x2,y2) 是否与以 (cx,cy) 为中心、大小为 (w,h) 的矩形相交
// 更详细的解释见 doc/segment_intersects_rectangle.svg
inline bool segment_intersects_rectangle(double x1, double y1,
                                         double x2, double y2,
                                         double cx, double cy,
                                         double w, double h)
{
    return fabs(x1 + x2 - 2.0 * cx) < fabs(x1 - x2) + w &&
           fabs(y1 + y2 - 2.0 * cy) < fabs(y1 - y2) + h &&
           2.0 * fabs((x1 - cx) * (y1 - y2) - (y1 - cy) * (x1 - x2)) <
               w * fabs(y1 - y2) + h * fabs(x1 - x2);
}

// 判断路径是否与以 (rect_x1, rect_y1) 和 (rect_x2, rect_y2) 为对角线的矩形相交
template <class PathIterator>
bool path_intersects_rectangle(PathIterator &path,
                               double rect_x1, double rect_y1,
                               double rect_x2, double rect_y2,
                               bool filled)
{
    // 定义去除 NaN 点的类型
    typedef PathNanRemover<mpl::PathIterator> no_nans_t;
    // 定义转换成曲线的类型
    typedef agg::conv_curve<no_nans_t> curve_t;

    // 如果路径没有顶点，直接返回不相交
    if (path.total_vertices() == 0) {
        return false;
    }

    // 创建去除 NaN 点后的路径对象
    no_nans_t no_nans(path, true, path.has_codes());
    // 用去除 NaN 点后的路径对象创建曲线对象
    curve_t curve(no_nans);

    // 计算矩形的中心点坐标和宽高
    double cx = (rect_x1 + rect_x2) * 0.5, cy = (rect_y1 + rect_y2) * 0.5;
    double w = fabs(rect_x1 - rect_x2), h = fabs(rect_y1 - rect_y2);

    double x1, y1, x2, y2;

    // 获取曲线的起始点坐标
    curve.vertex(&x1, &y1);
    // 如果路径在矩形内部，直接返回相交
    if (2.0 * fabs(x1 - cx) <= w && 2.0 * fabs(y1 - cy) <= h) {
        return true;
    }
    # 当还有顶点可以读取时继续循环
    while (curve.vertex(&x2, &y2) != agg::path_cmd_stop) {
        # 检查线段是否与指定矩形相交
        if (segment_intersects_rectangle(x1, y1, x2, y2, cx, cy, w, h)) {
            # 如果相交，返回true表示相交
            return true;
        }
        # 更新起始点为当前顶点，准备下一个线段的检查
        x1 = x2;
        y1 = y2;
    }

    # 如果需要填充图形
    if (filled) {
        # 创建仿射变换对象
        agg::trans_affine trans;
        # 检查点(cx, cy)是否在路径path内，使用仿射变换trans
        if (point_in_path(cx, cy, 0.0, path, trans)) {
            # 如果在路径内，返回true表示包含该点
            return true;
        }
    }

    # 默认情况下，返回false表示未找到任何相交或包含关系
    return false;
template <class PathIterator>
void convert_path_to_polygons(PathIterator &path,
                              agg::trans_affine &trans,
                              double width,
                              double height,
                              int closed_only,
                              std::vector<Polygon> &result)
{
    // 定义路径转换器类型，转换路径坐标为仿射变换后的坐标
    typedef agg::conv_transform<mpl::PathIterator> transformed_path_t;
    // 定义去除 NaN 点的路径处理器类型
    typedef PathNanRemover<transformed_path_t> nan_removal_t;
    // 定义裁剪路径处理器类型
    typedef PathClipper<nan_removal_t> clipped_t;
    // 定义简化路径处理器类型
    typedef PathSimplifier<clipped_t> simplify_t;
    // 定义曲线转换器类型，将简化后的路径转换为曲线
    typedef agg::conv_curve<simplify_t> curve_t;

    // 根据给定的宽度和高度确定是否执行裁剪操作
    bool do_clip = width != 0.0 && height != 0.0;
    // 确定是否需要简化路径
    bool simplify = path.should_simplify();

    // 创建路径转换器对象，传入原始路径和仿射变换
    transformed_path_t tpath(path, trans);
    // 创建去除 NaN 点的路径处理器对象
    nan_removal_t nan_removed(tpath, true, path.has_codes());
    // 创建裁剪路径处理器对象，传入去除 NaN 后的路径及裁剪参数
    clipped_t clipped(nan_removed, do_clip, width, height);
    // 创建简化路径处理器对象，传入裁剪后的路径及简化参数
    simplify_t simplified(clipped, simplify, path.simplify_threshold());
    // 创建曲线转换器对象，传入简化后的路径
    curve_t curve(simplified);

    // 将新创建的多边形对象添加到结果向量中
    result.push_back(Polygon());
    // 指向当前操作的多边形对象
    Polygon *polygon = &result.back();
    double x, y;
    unsigned code;

    // 遍历曲线对象的顶点，直到结束标志
    while ((code = curve.vertex(&x, &y)) != agg::path_cmd_stop) {
        // 如果当前命令是结束多边形的命令
        if ((code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
            // 完成当前多边形的最终化处理
            _finalize_polygon(result, 1);
            // 在结果向量中添加一个新的多边形对象
            result.push_back(Polygon());
            // 更新当前多边形指针
            polygon = &result.back();
        } else {
            // 如果当前命令是移动到新起点的命令
            if (code == agg::path_cmd_move_to) {
                // 完成当前多边形的最终化处理
                _finalize_polygon(result, closed_only);
                // 在结果向量中添加一个新的多边形对象
                result.push_back(Polygon());
                // 更新当前多边形指针
                polygon = &result.back();
            }
            // 将顶点坐标添加到当前多边形对象中
            polygon->push_back(XY(x, y));
        }
    }

    // 完成最后一个多边形的最终化处理
    _finalize_polygon(result, closed_only);
}

template <class VertexSource>
void
__cleanup_path(VertexSource &source, std::vector<double> &vertices, std::vector<npy_uint8> &codes)
{
    unsigned code;
    double x, y;
    // 循环处理顶点直到结束标志
    do {
        // 获取当前顶点的命令和坐标
        code = source.vertex(&x, &y);
        // 将顶点坐标添加到顶点向量
        vertices.push_back(x);
        vertices.push_back(y);
        // 将顶点命令添加到命令向量
        codes.push_back((npy_uint8)code);
    } while (code != agg::path_cmd_stop);
}

template <class PathIterator>
void cleanup_path(PathIterator &path,
                  agg::trans_affine &trans,
                  bool remove_nans,
                  bool do_clip,
                  const agg::rect_base<double> &rect,
                  e_snap_mode snap_mode,
                  double stroke_width,
                  bool do_simplify,
                  bool return_curves,
                  SketchParams sketch_params,
                  std::vector<double> &vertices,
                  std::vector<unsigned char> &codes)
{
    // 定义路径转换器类型，转换路径坐标为仿射变换后的坐标
    typedef agg::conv_transform<mpl::PathIterator> transformed_path_t;
    // 定义去除 NaN 点的路径处理器类型
    typedef PathNanRemover<transformed_path_t> nan_removal_t;
    // 定义裁剪路径处理器类型
    typedef PathClipper<nan_removal_t> clipped_t;
    // 定义路径捕捉处理器类型
    typedef PathSnapper<clipped_t> snapped_t;
    // 定义简化路径处理器类型
    typedef PathSimplifier<snapped_t> simplify_t;
    // 定义曲线转换器类型，将简化后的路径转换为曲线
    typedef agg::conv_curve<simplify_t> curve_t;
    // 定义草图处理器类型，将曲线转换为草图
    typedef Sketch<curve_t> sketch_t;

    // 创建路径转换器对象，传入原始路径和仿射变换
    transformed_path_t tpath(path, trans);
    // 创建去除 NaN 点的路径处理器对象
    nan_removal_t nan_removed(tpath, remove_nans, path.has_codes());
    // 创建裁剪路径处理器对象，传入去除 NaN 后的路径及裁剪参数
    clipped_t clipped(nan_removed, do_clip, rect.x1, rect.y1, rect.x2, rect.y2);
    // 创建路径捕捉处理器对象，传入裁剪后的路径及捕捉模式
    snapped_t snapped(clipped, snap_mode, stroke_width);
    // 创建简化路径处理器对象，传入捕捉后的路径及简化参数
    simplify_t simplified(snapped, do_simplify, path.simplify_threshold());
    // 创建曲线转换器对象，传入简化后的路径
    curve_t curve(simplified);

    // 循环处理路径中的顶点，并将其添加到顶点向量和命令向量中
    __cleanup_path(curve, vertices, codes);
}
    // 创建一个 nan_removal_t 对象，用于处理可能存在的 NaN 值
    nan_removal_t nan_removed(tpath, remove_nans, path.has_codes());
    
    // 创建一个 clipped_t 对象，用于处理边界裁剪
    clipped_t clipped(nan_removed, do_clip, rect);
    
    // 创建一个 snapped_t 对象，用于路径顶点的对齐处理
    snapped_t snapped(clipped, snap_mode, path.total_vertices(), stroke_width);
    
    // 创建一个 simplify_t 对象，用于简化路径
    simplify_t simplified(snapped, do_simplify, path.simplify_threshold());

    // 预留足够空间以存储路径顶点的两倍空间
    vertices.reserve(path.total_vertices() * 2);
    
    // 预留足够空间以存储路径代码
    codes.reserve(path.total_vertices());

    // 如果需要返回曲线并且绘制参数的比例为0时
    if (return_curves && sketch_params.scale == 0.0) {
        // 使用 __cleanup_path 函数清理简化后的路径，并存储顶点和代码
        __cleanup_path(simplified, vertices, codes);
    } else {
        // 创建一个 curve_t 对象，用于处理曲线
        curve_t curve(simplified);
        
        // 创建一个 sketch_t 对象，用于绘制草图
        sketch_t sketch(curve, sketch_params.scale, sketch_params.length, sketch_params.randomness);
        
        // 使用 __cleanup_path 函数清理草图路径，并存储顶点和代码
        __cleanup_path(sketch, vertices, codes);
    }
}

void quad2cubic(double x0, double y0,
                double x1, double y1,
                double x2, double y2,
                double *outx, double *outy)
{
    // 计算第一个控制点的坐标，转换二次贝塞尔曲线为三次贝塞尔曲线
    outx[0] = x0 + 2./3. * (x1 - x0);
    outy[0] = y0 + 2./3. * (y1 - y0);
    // 计算第二个控制点的坐标，转换二次贝塞尔曲线为三次贝塞尔曲线
    outx[1] = outx[0] + 1./3. * (x2 - x0);
    outy[1] = outy[0] + 1./3. * (y2 - y0);
    // 最后一个点的坐标直接等于原始的第三个控制点
    outx[2] = x2;
    outy[2] = y2;
}

void __add_number(double val, char format_code, int precision,
                  std::string& buffer)
{
    if (precision == -1) {
        // 特殊情况，用于兼容旧的ttconv代码，该代码通过将值强制转换为整数来截断，而不是像printf函数那样四舍五入。
        // 唯一会出现非整数值的情况是通过quad2cubic转换引入的浮点误差，为了补偿这一点，先四舍五入到最接近的1/3，然后再截断。
        char str[255];
        PyOS_snprintf(str, 255, "%d", (int)(round(val * 3)) / 3);
        buffer += str;
    } else {
        // 使用PyOS_double_to_string函数将浮点数转换为字符串，控制精度和格式
        char *str = PyOS_double_to_string(
          val, format_code, precision, Py_DTSF_ADD_DOT_0, NULL);
        // 删除尾部的零和小数点
        char *c = str + strlen(str) - 1;  // 从最后一个字符开始
        // 倒退直到找到非零的字符，如果有的话，还包括尾部的小数点
        while (*c == '0') {
            --c;
        }
        if (*c == '.') {
            --c;
        }
        try {
            // 将处理后的字符串追加到buffer中
            buffer.append(str, c + 1);
        } catch (std::bad_alloc& e) {
            PyMem_Free(str);
            throw e;
        }
        PyMem_Free(str);
    }
}

template <class PathIterator>
bool __convert_to_string(PathIterator &path,
                         int precision,
                         char **codes,
                         bool postfix,
                         std::string& buffer)
{
    const char format_code = 'f';

    // 初始化贝塞尔曲线控制点和最后的坐标
    double x[3];
    double y[3];
    double last_x = 0.0;
    double last_y = 0.0;

    unsigned code;
    // 循环直到遍历完路径上的所有顶点和命令
    while ((code = path.vertex(&x[0], &y[0])) != agg::path_cmd_stop) {
        // 如果是闭合多边形命令
        if (code == CLOSEPOLY) {
            // 将对应的代码添加到缓冲区中
            buffer += codes[4];
        } else if (code < 5) {
            // 获取当前命令对应的顶点数量
            size_t size = NUM_VERTICES[code];

            // 遍历当前命令中的每个顶点
            for (size_t i = 1; i < size; ++i) {
                // 获取顶点，并检查其命令是否与当前命令相同
                unsigned subcode = path.vertex(&x[i], &y[i]);
                if (subcode != code) {
                    // 如果命令不一致，则返回失败
                    return false;
                }
            }

            /* 对于不支持二次曲线的格式，将其转换为三次曲线 */
            if (code == CURVE3 && codes[code - 1][0] == '\0') {
                // 调用函数将二次曲线转换为三次曲线
                quad2cubic(last_x, last_y, x[0], y[0], x[1], y[1], x, y);
                code++;  // 命令更新为三次曲线
                size = 3; // 更新顶点数量为三个
            }

            // 如果不是后缀模式，则将命令代码添加到缓冲区中
            if (!postfix) {
                buffer += codes[code - 1];
                buffer += ' ';
            }

            // 遍历顶点数组，并将每个顶点的坐标格式化后添加到缓冲区中
            for (size_t i = 0; i < size; ++i) {
                __add_number(x[i], format_code, precision, buffer);
                buffer += ' ';
                __add_number(y[i], format_code, precision, buffer);
                buffer += ' ';
            }

            // 如果是后缀模式，则将命令代码添加到缓冲区中
            if (postfix) {
                buffer += codes[code - 1];
            }

            // 更新最后一个顶点的坐标
            last_x = x[size - 1];
            last_y = y[size - 1];
        } else {
            // 如果命令值未知，则返回失败
            // Unknown code value
            return false;
        }

        // 在每个命令处理后添加换行符到缓冲区中
        buffer += '\n';
    }

    // 所有命令处理完成，返回成功
    return true;
}

// 结束 C++ 头文件的条件编译指令，确保只有在头文件被包含一次时才会被编译

template <class PathIterator>
bool convert_to_string(PathIterator &path,
                       agg::trans_affine &trans,
                       agg::rect_d &clip_rect,
                       bool simplify,
                       SketchParams sketch_params,
                       int precision,
                       char **codes,
                       bool postfix,
                       std::string& buffer)
{
    size_t buffersize;
    typedef agg::conv_transform<mpl::PathIterator> transformed_path_t; // 定义类型转换器，将路径转换为特定类型
    typedef PathNanRemover<transformed_path_t> nan_removal_t; // 定义移除 NaN 值的路径处理器
    typedef PathClipper<nan_removal_t> clipped_t; // 定义路径剪裁器
    typedef PathSimplifier<clipped_t> simplify_t; // 定义路径简化器
    typedef agg::conv_curve<simplify_t> curve_t; // 定义曲线转换器
    typedef Sketch<curve_t> sketch_t; // 定义草图生成器

    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2); // 检查是否需要剪裁路径

    transformed_path_t tpath(path, trans); // 创建转换后的路径对象
    nan_removal_t nan_removed(tpath, true, path.has_codes()); // 创建移除 NaN 值后的路径处理对象
    clipped_t clipped(nan_removed, do_clip, clip_rect); // 创建剪裁后的路径对象
    simplify_t simplified(clipped, simplify, path.simplify_threshold()); // 创建简化后的路径对象

    buffersize = (size_t) path.total_vertices() * (precision + 5) * 4; // 计算缓冲区大小
    if (buffersize == 0) {
        return true; // 如果缓冲区大小为0，直接返回true
    }

    if (sketch_params.scale != 0.0) {
        buffersize *= 10; // 根据比例因子调整缓冲区大小
    }

    buffer.reserve(buffersize); // 预留缓冲区空间

    if (sketch_params.scale == 0.0) {
        return __convert_to_string(simplified, precision, codes, postfix, buffer); // 如果比例因子为0，则直接将简化后的路径转换为字符串
    } else {
        curve_t curve(simplified); // 创建曲线对象
        sketch_t sketch(curve, sketch_params.scale, sketch_params.length, sketch_params.randomness); // 创建草图对象
        return __convert_to_string(sketch, precision, codes, postfix, buffer); // 将草图转换为字符串
    }

}

template<class T>
bool is_sorted_and_has_non_nan(PyArrayObject *array)
{
    char* ptr = PyArray_BYTES(array); // 获取数组的起始指针
    npy_intp size = PyArray_DIM(array, 0), // 获取数组的维度大小
             stride = PyArray_STRIDE(array, 0); // 获取数组的步长
    using limits = std::numeric_limits<T>; // 使用模板类型的数值极限
    T last = limits::has_infinity ? -limits::infinity() : limits::min(); // 初始化上一个值为极限的负无穷或最小值
    bool found_non_nan = false; // 初始化找到非 NaN 值为false

    for (npy_intp i = 0; i < size; ++i, ptr += stride) {
        T current = *(T*)ptr; // 获取当前元素的值
        // 以下检查 !isnan(current)，但对整数类型也有效。在 MSVC 上不存在 isnan(IntegralType) 的重载。
        if (current == current) { // 如果当前值不是 NaN
            found_non_nan = true; // 标记找到了非 NaN 值
            if (current < last) {
                return false; // 如果当前值小于上一个值，返回false，表示数组未排序
            }
            last = current; // 更新上一个值为当前值
        }
    }
    return found_non_nan; // 返回是否找到非 NaN 值
};

#endif
```