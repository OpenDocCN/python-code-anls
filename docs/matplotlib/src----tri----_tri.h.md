# `D:\src\scipysrc\matplotlib\src\tri\_tri.h`

```
#ifndef MPL_TRI_H
#define MPL_TRI_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <list>
#include <map>
#include <set>
#include <vector>

namespace py = pybind11;

/* An edge of a triangle consisting of a triangle index in the range 0 to
 * ntri-1 and an edge index in the range 0 to 2.  Edge i goes from the
 * triangle's point i to point (i+1)%3. */
struct TriEdge final
{
    TriEdge();  // 默认构造函数
    TriEdge(int tri_, int edge_);  // 带参数的构造函数，初始化三角形索引和边索引
    bool operator<(const TriEdge& other) const;  // 重载小于运算符，用于比较 TriEdge
    bool operator==(const TriEdge& other) const;  // 重载等于运算符，用于比较 TriEdge
    bool operator!=(const TriEdge& other) const;  // 重载不等于运算符，用于比较 TriEdge
    friend std::ostream& operator<<(std::ostream& os, const TriEdge& tri_edge);  // 重载流输出运算符，用于输出 TriEdge

    int tri, edge;  // 三角形索引和边索引
};

// 2D point with x,y coordinates.
struct XY
{
    XY();  // 默认构造函数
    XY(const double& x_, const double& y_);  // 带参数的构造函数，初始化 x 和 y 坐标
    double angle() const;  // 返回点与 x 轴的夹角（弧度制）
    double cross_z(const XY& other) const;  // 返回点与另一点的 z 方向上的叉积分量
    bool is_right_of(const XY& other) const;  // 按 x 然后 y 的顺序比较点的位置关系
    bool operator==(const XY& other) const;  // 重载等于运算符，用于比较两个 XY 对象是否相等
    bool operator!=(const XY& other) const;  // 重载不等于运算符，用于比较两个 XY 对象是否不等
    XY operator*(const double& multiplier) const;  // 重载乘法运算符，返回乘以标量后的新 XY 对象
    const XY& operator+=(const XY& other);  // 重载加法赋值运算符，将另一点加到当前点上
    const XY& operator-=(const XY& other);  // 重载减法赋值运算符，将另一点从当前点减去
    XY operator+(const XY& other) const;  // 重载加法运算符，返回两个点相加后的新 XY 对象
    XY operator-(const XY& other) const;  // 重载减法运算符，返回两个点相减后的新 XY 对象
    friend std::ostream& operator<<(std::ostream& os, const XY& xy);  // 重载流输出运算符，用于输出 XY 对象

    double x, y;  // 点的 x 和 y 坐标
};

// 3D point with x,y,z coordinates.
struct XYZ final
{
    XYZ(const double& x_, const double& y_, const double& z_);  // 带参数的构造函数，初始化 x、y、z 坐标
    XYZ cross(const XYZ& other) const;  // 返回当前点与另一点的叉积
    double dot(const XYZ& other) const;  // 返回当前点与另一点的点积
    XYZ operator-(const XYZ& other) const;  // 重载减法运算符，返回两个点相减后的新 XYZ 对象
    friend std::ostream& operator<<(std::ostream& os, const XYZ& xyz);  // 重载流输出运算符，用于输出 XYZ 对象

    double x, y, z;  // 点的 x、y、z 坐标
};

// 2D bounding box, which may be empty.
class BoundingBox final
{
public:
    BoundingBox();  // 默认构造函数
    void add(const XY& point);  // 向边界框添加一个点
    void expand(const XY& delta);  // 根据增量扩展边界框

    // Consider these member variables read-only.
    bool empty;  // 表示边界框是否为空
    XY lower, upper;  // 边界框的下界和上界
};

/* A single line of a contour, which may be a closed line loop or an open line
 * strip.  Identical adjacent points are avoided using push_back(), and a closed
 * line loop should also not have identical first and last points. */
class ContourLine final : public std::vector<XY>
{
public:
    ContourLine();  // 默认构造函数
    void push_back(const XY& point);  // 将一个点添加到轮廓线末尾
    void write() const;  // 写入轮廓线信息到输出流
};

// A Contour is a collection of zero or more ContourLines.
typedef std::vector<ContourLine> Contour;  // Contour 是多个 ContourLine 的集合

// Debug contour writing function.
void write_contour(const Contour& contour);  // 将轮廓数据写入输出流

/* Triangulation with npoints points and ntri triangles.  Derived fields are
 * calculated when they are first needed. */
class Triangulation final
{
public:
    typedef py::array_t<double, py::array::c_style | py::array::forcecast> CoordinateArray;  // 定义坐标数组类型
    typedef py::array_t<double, py::array::c_style | py::array::forcecast> TwoCoordinateArray;  // 定义双坐标数组类型
    /* 定义三角形、掩码、边界和邻居的 Pybind11 数组类型 */
    typedef py::array_t<int,    py::array::c_style | py::array::forcecast> TriangleArray;
    typedef py::array_t<bool,   py::array::c_style | py::array::forcecast> MaskArray;
    typedef py::array_t<int,    py::array::c_style | py::array::forcecast> EdgeArray;
    typedef py::array_t<int,    py::array::c_style | py::array::forcecast> NeighborArray;

    /* 单个边界是由组成该边界的 TriEdge 向量组成的，
     * 跟随未被掩盖的三角形在左侧。 */
    typedef std::vector<TriEdge> Boundary;
    typedef std::vector<Boundary> Boundaries;

    /* 构造函数，可选参数包括掩码、边界和邻居。
     * 后两者在首次需要时计算。
     *   x: 形状为 (npoints) 的点的 x 坐标的双精度数组。
     *   y: 形状为 (npoints) 的点的 y 坐标的双精度数组。
     *   triangles: 形状为 (ntri,3) 的三角形点索引的整数数组。
     *              顺时针排列的三角形被更改为逆时针。
     *   mask: 可选的形状为 (ntri) 的布尔数组，指示哪些三角形被掩盖。
     *   edges: 可选的形状为 (?,2) 的整数数组，包含起始和结束点索引，
     *          每个边 (start,end 和 end,start) 只出现一次。
     *   neighbors: 可选的形状为 (ntri,3) 的整数数组，指示哪些三角形是
     *              TriEdge 的邻居，如果没有这样的邻居则为 -1。
     *   correct_triangle_orientations: 是否应该纠正三角形方向，
     *                                  以使顶点逆时针排序。 */
    Triangulation(const CoordinateArray& x,
                  const CoordinateArray& y,
                  const TriangleArray& triangles,
                  const MaskArray& mask,
                  const EdgeArray& edges,
                  const NeighborArray& neighbors,
                  bool correct_triangle_orientations);

    /* 从传入的参数 (x,y) 坐标和形状为 (npoints) 的点 z 数组中，
     * 计算所有未被掩盖的三角形的平面方程系数。
     * 返回的数组形状为 (npoints,3)，允许计算三角形 tri 中 (x,y) 坐标处的 z 值，
     *      z = array[tri,0]*x + array[tri,1]*y + array[tri,2]。 */
    TwoCoordinateArray calculate_plane_coefficients(const CoordinateArray& z);

    // 返回边界集合，如有必要则创建。
    const Boundaries& get_boundaries() const;

    // 返回指定 TriEdge 的边界及边界边缘。
    void get_boundary_edge(const TriEdge& triEdge,
                           int& boundary,
                           int& edge) const;

    /* 返回边缘数组，如有必要则创建。 */
    EdgeArray& get_edges();

    /* 返回指定三角形边缘的邻居三角形的索引。 */
    // 返回指定三角形边的邻居三角形索引，如果没有邻居则返回TriEdge(-1,-1)
    int get_neighbor(int tri, int edge) const;

    /* 返回指定三角形边的邻居三角形边，如果没有邻居则返回TriEdge(-1,-1) */
    TriEdge get_neighbor_edge(int tri, int edge) const;

    /* 返回邻居数组，如有必要则创建之 */
    NeighborArray& get_neighbors();

    // 返回此三角剖分中的点数
    int get_npoints() const;

    // 返回此三角剖分中的三角形数
    int get_ntri() const;

    /* 返回指定三角形边起始点的索引 */
    int get_triangle_point(int tri, int edge) const;
    int get_triangle_point(const TriEdge& tri_edge) const;

    // 返回指定点索引的坐标
    XY get_point_coords(int point) const;

    // 指示指定三角形是否被屏蔽（masked）
    bool is_masked(int tri) const;

    /* 设置或清除屏蔽数组。清除各种派生字段，以便在下次需要时重新计算。
     *   mask: 形状为(ntri)的布尔数组，指示哪些三角形被屏蔽，或空数组以清除屏蔽。 */
    void set_mask(const MaskArray& mask);

    // 调试函数，用于写入边界信息
    void write_boundaries() const;
private:
    // 三角剖分中的边，由起点和终点索引组成。
    struct Edge final
    {
        Edge() : start(-1), end(-1) {}
        Edge(int start_, int end_) : start(start_), end(end_) {}
        // 比较运算符，按照起点索引排序，如果起点相同则按终点索引排序。
        bool operator<(const Edge& other) const {
            return start != other.start ? start < other.start : end < other.end;
        }
        int start, end;
    };

    /* 三角剖分边界的边，由边界索引和边界内部的边索引组成。
     * 用于在边界集合中索引对应的TriEdge。 */
    struct BoundaryEdge final
    {
        BoundaryEdge() : boundary(-1), edge(-1) {}
        BoundaryEdge(int boundary_, int edge_)
            : boundary(boundary_), edge(edge_) {}
        int boundary, edge;
    };

    /* 计算边界集合。通常通过get_boundaries()访问，如果必要则调用此函数。 */
    void calculate_boundaries();

    /* 计算边缘数组。通常通过get_edges()访问，如果必要则调用此函数。 */
    void calculate_edges();

    /* 计算邻居数组。通常通过get_neighbors()访问，如果必要则调用此函数。 */
    void calculate_neighbors();

    /* 修正每个三角形，使顶点按逆时针方式排序。 */
    void correct_triangles();

    /* 确定指定点索引在指定三角形中的边索引（0、1或2），如果点不在三角形中则返回-1。 */
    int get_edge_in_triangle(int tri, int point) const;

    // 检查是否存在边缘数据。
    bool has_edges() const;

    // 检查是否存在掩码数据。
    bool has_mask() const;

    // 检查是否存在邻居数据。
    bool has_neighbors() const;


    // 与Python共享的变量，始终设置。
    CoordinateArray _x, _y;    // 双精度数组（npoints）。
    TriangleArray _triangles;  // 整数数组（ntri,3），三角形顶点索引，顺时针排序。

    // 与Python共享的变量，可能未设置（大小为0）。
    MaskArray _mask;           // 布尔数组（ntri）。

    // 从Python派生的变量，可能未设置（大小为0）。
    // 如果未设置，将在需要时重新计算。
    EdgeArray _edges;          // 整数数组（?,2），起点和终点索引。
    NeighborArray _neighbors;  // 整数数组（ntri,3），邻居三角形索引或-1（表示没有邻居）。

    // 仅限于C++内部的变量。
    Boundaries _boundaries;

    // 用于从TriEdges查找BoundaryEdges的映射。通常通过get_boundary_edge()访问。
    typedef std::map<TriEdge, BoundaryEdge> TriEdgeToBoundaryMap;
    TriEdgeToBoundaryMap _tri_edge_to_boundary_map;
};
    /* Define a type alias for a NumPy array of unsigned chars. */
    typedef py::array_t<unsigned char> CodeArray;

    /* Constructor for TriContourGenerator class.
     *   triangulation: Triangulation object used for contour generation.
     *   z: NumPy array of shape (npoints) containing z-values at triangulation points. */
    TriContourGenerator(Triangulation& triangulation,
                        const CoordinateArray& z);

    /* Method to create and return a non-filled contour.
     *   level: Contour level at which to generate the contour line.
     * Returns a new Python tuple [segs0, segs1, ...] where
     *   segs0: NumPy array of shape (?, 2) containing point coordinates of the first contour line,
     *   segs1: NumPy array of shape (?, 2) containing point coordinates of the second contour line, etc. */
    py::tuple create_contour(const double& level);

    /* Method to create and return a filled contour.
     *   lower_level: Lower contour level.
     *   upper_level: Upper contour level.
     * Returns a new Python tuple (segs, kinds) where
     *   segs: NumPy array of shape (n_points, 2) containing coordinates of all points in the filled contour,
     *   kinds: NumPy array of shape (n_points) containing point code types as unsigned bytes. */
    py::tuple create_filled_contour(const double& lower_level,
                                    const double& upper_level);
/* 清除访问标志。
 *   include_boundaries: 是否清除边界标志，仅用于填充轮廓。 */
void clear_visited_flags(bool include_boundaries);

/* 将非填充轮廓从 C++ 转换到 Python。
 * 返回新的 Python 元组 ([segs0, segs1, ...], [kinds0, kinds1...])
 * 其中
 *   segs0: 形状为 (n_points,2) 的双精度数组，表示第一个轮廓线的点坐标，依此类推。
 *   kinds0: 形状为 (n_points) 的无符号字节数组，表示第一个轮廓线的点种类代码，依此类推。 */
py::tuple contour_line_to_segs_and_kinds(const Contour& contour);

/* 将填充轮廓从 C++ 转换到 Python。
 * 返回新的 Python 元组 ([segs], [kinds])
 * 其中
 *   segs: 形状为 (n_points,2) 的双精度数组，表示所有点的坐标。
 *   kinds: 形状为 (n_points) 的无符号字节数组，表示所有点的种类代码。 */
py::tuple contour_to_segs_and_kinds(const Contour& contour);

/* 返回指定 TriEdge 上与指定水平线相交的点。 */
XY edge_interp(int tri, int edge, const double& level);

/* 查找并跟踪非填充轮廓线，这些线从 Triangulation 的边界开始并结束。
 *   contour: 要添加新线条的轮廓。
 *   level: 轮廓线水平。 */
void find_boundary_lines(Contour& contour,
                         const double& level);

/* 查找并跟踪填充轮廓线，这些线在 Triangulation 的边界上开始并结束，并且处于指定的轮廓水平范围内。
 *   contour: 要添加新线条的轮廓。
 *   lower_level: 较低的轮廓水平。
 *   upper_level: 较高的轮廓水平。 */
void find_boundary_lines_filled(Contour& contour,
                                const double& lower_level,
                                const double& upper_level);

/* 查找并跟踪位于 Triangulation 内部且因此不与任何边界相交的指定轮廓水平的线条。
 *   contour: 要添加新线条的轮廓。
 *   level: 轮廓水平。
 *   on_upper: 是否在上层或下层轮廓水平。
 *   filled: 轮廓是否填充。 */
void find_interior_lines(Contour& contour,
                         const double& level,
                         bool on_upper,
                         bool filled);
    /* Follow contour line around boundary of the Triangulation from the
     * specified TriEdge to its end which can be on either the lower or upper
     * levels.  Only used for filled contours.
     *   contour_line: Contour line to append new points to.
     *   tri_edge: On entry, TriEdge to start from.  On exit, TriEdge that is
     *             finished on.
     *   lower_level: Lower contour level.
     *   upper_level: Upper contour level.
     *   on_upper: Whether starts on upper level or not.
     * Return true if finishes on upper level, false if lower. */
    bool follow_boundary(ContourLine& contour_line,
                         TriEdge& tri_edge,
                         const double& lower_level,
                         const double& upper_level,
                         bool on_upper);



    /* Follow contour line across interior of Triangulation.
     *   contour_line: Contour line to append new points to.
     *   tri_edge: On entry, TriEdge to start from.  On exit, TriEdge that is
     *             finished on.
     *   end_on_boundary: Whether this line ends on a boundary, or loops back
     *                    upon itself.
     *   level: Contour level to follow.
     *   on_upper: Whether following upper or lower contour level. */
    void follow_interior(ContourLine& contour_line,
                         TriEdge& tri_edge,
                         bool end_on_boundary,
                         const double& level,
                         bool on_upper);



    // Return the Triangulation boundaries.
    const Boundaries& get_boundaries() const;



    /* Return the edge by which the a level leaves a particular triangle,
     * which is 0, 1 or 2 if the contour passes through the triangle or -1
     * otherwise.
     *   tri: Triangle index.
     *   level: Contour level to follow.
     *   on_upper: Whether following upper or lower contour level. */
    int get_exit_edge(int tri, const double& level, bool on_upper) const;



    // Return the z-value at the specified point index.
    const double& get_z(int point) const;



    /* Return the point at which the a level intersects the line connecting the
     * two specified point indices. */
    XY interp(int point1, int point2, const double& level) const;



    // Variables shared with python, always set.
    Triangulation _triangulation;
    CoordinateArray _z;        // double array (npoints).



    // Variables internal to C++ only.
    typedef std::vector<bool> InteriorVisited;    // Size 2*ntri
    typedef std::vector<bool> BoundaryVisited;
    typedef std::vector<BoundaryVisited> BoundariesVisited;
    typedef std::vector<bool> BoundariesUsed;

    InteriorVisited _interior_visited;
    BoundariesVisited _boundaries_visited;  // Only used for filled contours.
    BoundariesUsed _boundaries_used;        // Only used for filled contours.
};

/* TriFinder 类使用书籍《Computational Geometry, Algorithms and Applications》
 * 中的梯形映射算法实现。
 *
 * 兴趣域由垂直边梯形组成，左右边界为三角剖分的点，上下边界为三角剖分的边。
 * 每个三角形由一个或多个这些梯形表示。边以随机顺序逐个插入。
 *
 * 创建梯形映射的同时，还创建了一个搜索树，允许快速查找 O(log N) 包含兴趣点的梯形。
 * 搜索树有三种节点类型：所有叶节点表示梯形，所有分支节点有两个子节点，
 * 分别是 x-节点和 y-节点。X-节点表示三角剖分中的点，它的两个子节点分别是点左右两侧的搜索树部分。
 * Y-节点表示三角剖分中的边，它的两个子节点分别是边下方和上方的搜索树部分。
 *
 * 节点可以在搜索树中重复，并通过其多个父节点进行引用计数。
 *
 * 该算法仅适用于有效的三角剖分，即不应包含重复点、由共线点形成的三角形或重叠三角形。
 * 它对由共线点形成的三角形具有一定的容忍度，但只限于最简单的情况。
 * 该算法不显式测试三角剖分的有效性，因为这比三角查找本身计算复杂得多。
 */
class TrapezoidMapTriFinder final
{
public:
    typedef Triangulation::CoordinateArray CoordinateArray;
    typedef py::array_t<int, py::array::c_style | py::array::forcecast> TriIndexArray;

    /* 构造函数。必须在使用对象之前调用 initialize() 来初始化对象。
     *   triangulation: 要在其中查找三角形的三角剖分。 */
    TrapezoidMapTriFinder(Triangulation& triangulation);

    ~TrapezoidMapTriFinder();

    /* 返回三角形索引数组。接受点坐标的一维数组 x 和 y，并返回一个相同大小的数组，
     * 其中包含这些点处三角形的索引。 */
    TriIndexArray find_many(const CoordinateArray& x, const CoordinateArray& y);
    /* 返回一个新的 Python 列表引用，包含关于树的以下统计信息：
     *   0: 节点数（树的大小）
     *   1: 唯一节点数（树中唯一的 Node 对象数）
     *   2: 梯形数（树的叶节点数）
     *   3: 唯一梯形数
     *   4: 最大父节点计数（树中某节点重复出现的最大次数）
     *   5: 树的最大深度（搜索树需要的最大比较次数加一）
     *   6: 所有梯形深度的平均值（搜索树需要的平均比较次数加一） */
    py::list get_tree_stats();
    
    /* 在使用之前初始化对象。如果例如通过设置掩码更改三角剖分，可以多次调用此函数。 */
    void initialize();
    
    // 将搜索树以文本形式打印到标准输出；用于调试目的。
    void print_tree();
/* A Point consists of x,y coordinates as well as the index of a triangle
 * associated with the point, so that a search at this point's coordinates
 * can return a valid triangle index. */
struct Point final : XY
{
    Point() : XY(), tri(-1) {}  // Default constructor initializing coordinates and triangle index.
    Point(const double& x, const double& y) : XY(x,y), tri(-1) {}  // Constructor initializing coordinates and triangle index.
    explicit Point(const XY& xy) : XY(xy), tri(-1) {}  // Constructor initializing coordinates and triangle index.

    int tri;  // Index of the triangle associated with this point.
};

/* An Edge connects two Points, left and right.  It is always true that
 * right->is_right_of(*left).  Stores indices of triangles below and above
 * the Edge which are used to map from trapezoid to triangle index.  Also
 * stores pointers to the 3rd points of the below and above triangles,
 * which are only used to disambiguate triangles with colinear points. */
struct Edge final
{
    Edge(const Point* left_,
         const Point* right_,
         int triangle_below_,
         int triangle_above_,
         const Point* point_below_,
         const Point* point_above_);

    // Return -1 if point to left of edge, 0 if on edge, +1 if to right.
    int get_point_orientation(const XY& xy) const;

    // Return slope of edge, even if vertical (divide by zero is OK here).
    double get_slope() const;

    /* Return y-coordinate of point on edge with specified x-coordinate.
     * x must be within the x-limits of this edge. */
    double get_y_at_x(const double& x) const;

    // Return true if the specified point is either of the edge end points.
    bool has_point(const Point* point) const;

    bool operator==(const Edge& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Edge& edge)
    {
        return os << *edge.left << "->" << *edge.right;  // Print the edge in the format "left->right".
    }

    void print_debug() const;

    const Point* left;        // Not owned. Pointer to the left endpoint of the edge.
    const Point* right;       // Not owned. Pointer to the right endpoint of the edge.
    int triangle_below;       // Index of triangle below (to right of) Edge.
    int triangle_above;       // Index of triangle above (to left of) Edge.
    const Point* point_below; // Used only for resolving ambiguous cases;
    const Point* point_above; //     is 0 if corresponding triangle is -1
};

class Node;  // Forward declaration.

// Helper structure used by TrapezoidMapTriFinder::get_tree_stats.
struct NodeStats final
{
    NodeStats()
        : node_count(0), trapezoid_count(0), max_parent_count(0),
          max_depth(0), sum_trapezoid_depth(0.0)
    {}

    long node_count;                    // Count of nodes.
    long trapezoid_count;               // Count of trapezoids.
    long max_parent_count;              // Maximum parent count.
    long max_depth;                     // Maximum depth.
    double sum_trapezoid_depth;         // Sum of trapezoid depths.
    std::set<const Node*> unique_nodes;         // Set of unique nodes.
    std::set<const Node*> unique_trapezoid_nodes; // Set of unique trapezoid nodes.
};

struct Trapezoid;  // Forward declaration.
    /* Node of the trapezoid map search tree.  There are 3 possible types:
     * Type_XNode, Type_YNode and Type_TrapezoidNode.  Data members are
     * represented using a union: an XNode has a Point and 2 child nodes
     * (left and right of the point), a YNode has an Edge and 2 child nodes
     * (below and above the edge), and a TrapezoidNode has a Trapezoid.
     * Each Node has multiple parents so it can appear in the search tree
     * multiple times without having to create duplicate identical Nodes.
     * The parent collection acts as a reference count to the number of times
     * a Node occurs in the search tree.  When the parent count is reduced to
     * zero a Node can be safely deleted. */
    class Node final
    {
    public:
        // Constructor for Type_XNode, takes a Point and 2 child Nodes.
        Node(const Point* point, Node* left, Node* right);

        // Constructor for Type_YNode, takes an Edge and 2 child Nodes.
        Node(const Edge* edge, Node* below, Node* above);

        // Constructor for Type_TrapezoidNode, takes a Trapezoid.
        Node(Trapezoid* trapezoid);

        // Destructor to clean up resources associated with the Node.
        ~Node();

        // Add a parent Node to this Node's collection of parents.
        void add_parent(Node* parent);

        /* Recursively validate the structure of the tree. Reduces to a no-op
         * if NDEBUG is defined. */
        void assert_valid(bool tree_complete) const;

        // Recursively gather statistics about the tree.
        void get_stats(int depth, NodeStats& stats) const;

        // Return the index of the triangle corresponding to this node.
        int get_tri() const;

        // Check if this Node has a particular child Node.
        bool has_child(const Node* child) const;

        // Check if this Node has no parents.
        bool has_no_parents() const;

        // Check if this Node has a specific parent Node.
        bool has_parent(const Node* parent) const;

        /* Recursively print a textual representation of the tree to stdout.
         * Argument depth is used for indentation. */
        void print(int depth = 0) const;

        // Remove a parent Node from this Node's collection.
        // Returns true if no parents remain after removal.
        bool remove_parent(Node* parent);

        // Replace an old child Node with a new child Node.
        void replace_child(Node* old_child, Node* new_child);

        // Replace this Node with a specified new_node in all parents.
        void replace_with(Node* new_node);

        /* Recursive search through the tree to find the Node containing the
         * specified XY point. */
        const Node* search(const XY& xy);

        /* Recursive search through the tree to find the Trapezoid containing
         * the left endpoint of the specified Edge. Returns nullptr if fails,
         * which can only happen if the triangulation is invalid. */
        Trapezoid* search(const Edge& edge);

        // Copy constructor and assignment operator are declared but not implemented.
        // This prevents accidental copying of Node objects.
        Node(const Node& other);
        Node& operator=(const Node& other);
    };
    private:
        // 定义枚举类型 Type，用于标识节点的类型
        typedef enum {
            Type_XNode,           // X 节点
            Type_YNode,           // Y 节点
            Type_TrapezoidNode    // 梯形节点
        } Type;
        Type _type;               // 节点的具体类型

        // 联合体，根据节点类型存储不同的数据结构
        union {
            struct {
                const Point* point;  // 不拥有指针，指向点的常量指针
                Node* left;          // 拥有指针，指向左子节点
                Node* right;         // 拥有指针，指向右子节点
            } xnode;                 // X 节点的数据结构
            struct {
                const Edge* edge;    // 不拥有指针，指向边的常量指针
                Node* below;         // 拥有指针，指向下方节点
                Node* above;         // 拥有指针，指向上方节点
            } ynode;                 // Y 节点的数据结构
            Trapezoid* trapezoid;    // 拥有指针，指向梯形节点
        } _union;                    // 联合体实例

        typedef std::list<Node*> Parents;  // 父节点列表类型定义
        Parents _parents;                  // 不拥有指针，指向父节点列表
    };

    /* 一个梯形由左右两个点和下上两条边界限定。有最多四个相邻梯形，分别在左下、右下、左上、右上。
     * 左下邻居是左侧具有相同下边界的梯形，或者如果没有这样的梯形则为 0（其他相邻梯形类似处理）。
     * 要根据特定梯形获取对应的三角形的索引，使用下边界成员变量 below.triangle_above 或 above.triangle_below。 */
    struct Trapezoid final
    {
        /* 定义梯形结构体，包括左右顶点、上下边 */
        Trapezoid(const Point* left_,
                  const Point* right_,
                  const Edge& below_,
                  const Edge& above_);
    
        /* 断言此梯形是否有效，如果定义了 NDEBUG 则为空操作 */
        void assert_valid(bool tree_complete) const;
    
        /* 返回此梯形的四个角点之一，仅用于调试目的 */
        XY get_lower_left_point() const;
        XY get_lower_right_point() const;
        XY get_upper_left_point() const;
        XY get_upper_right_point() const;
    
        /* 打印调试信息 */
        void print_debug() const;
    
        /* 设置四个相邻梯形及其对应的反向梯形（如果非空），以保持一致性 */
        void set_lower_left(Trapezoid* lower_left_);
        void set_lower_right(Trapezoid* lower_right_);
        void set_upper_left(Trapezoid* upper_left_);
        void set_upper_right(Trapezoid* upper_right_);
    
        /* 复制构造函数和赋值运算符被定义但未实现，以防止对象被复制 */
        Trapezoid(const Trapezoid& other);
        Trapezoid& operator=(const Trapezoid& other);
    
        const Point* left;     // 不拥有
        const Point* right;    // 不拥有
        const Edge& below;
        const Edge& above;
    
        // 四个相邻梯形，可以为0，不拥有
        Trapezoid* lower_left;   // 左侧共享下边的梯形
        Trapezoid* lower_right;  // 右侧共享下边的梯形
        Trapezoid* upper_left;   // 左侧共享上边的梯形
        Trapezoid* upper_right;  // 右侧共享上边的梯形
    
        Node* trapezoid_node;    // 拥有此梯形的节点
    };
    
    /* 将指定的边添加到搜索树中，如果成功则返回true */
    bool add_edge_to_tree(const Edge& edge);
    
    /* 清除此对象分配的所有内存 */
    void clear();
    
    /* 在指定点处查找三角形索引，如果没有则返回-1 */
    int find_one(const XY& xy);
    
    /* 确定指定边相交的梯形，如果成功则返回true */
    bool find_trapezoids_intersecting_edge(const Edge& edge,
                                           std::vector<Trapezoid*>& trapezoids);
    
    
    // 与Python共享的变量，始终设置
    Triangulation& _triangulation;
    
    // 仅限C++内部使用的变量
    Point* _points;    // 包括所有三角剖分点和边界矩形的角点。拥有此数组
    
    typedef std::vector<Edge> Edges;
    Edges _edges;   // 包括所有三角剖分边以及边界矩形的底部和顶部边。拥有此向量
    
    Node* _tree;    // 梯形映射搜索树的根节点。拥有此节点
    }
};

#endif



// 结束了一个类定义或者条件编译的预处理指令块，对应于一个 #if 或 #ifdef 开始的地方
```