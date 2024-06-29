# `D:\src\scipysrc\matplotlib\src\tri\_tri.cpp`

```
/* 这个文件广泛使用断言来辅助代码开发和调试。
 * 标准的 Matplotlib 构建会禁用断言，因此不会影响性能。
 * 要启用断言，需要取消定义 NDEBUG 宏，可以通过向 Meson 配置传递 ``b_ndebug=false`` 来实现。
 */
#include "../mplutils.h"  // 引入自定义的 Matplotlib 实用工具库头文件
#include "_tri.h"          // 引入三角形相关功能的私有头文件

#include <algorithm>       // 引入算法标准库
#include <random>          // 引入随机数生成器标准库
#include <set>             // 引入集合容器标准库

TriEdge::TriEdge()
    : tri(-1), edge(-1)   // TriEdge 类默认构造函数，初始化 tri 和 edge 为 -1
{}

TriEdge::TriEdge(int tri_, int edge_)
    : tri(tri_), edge(edge_)  // TriEdge 类带参构造函数，初始化 tri 和 edge
{}

bool TriEdge::operator<(const TriEdge& other) const
{
    // 按照 tri 的值进行比较，如果 tri 值不同，则比较 tri 的大小
    if (tri != other.tri)
        return tri < other.tri;
    else
        return edge < other.edge;  // 如果 tri 值相同，则比较 edge 的大小
}

bool TriEdge::operator==(const TriEdge& other) const
{
    return tri == other.tri && edge == other.edge;  // 比较两个 TriEdge 对象是否相等
}

bool TriEdge::operator!=(const TriEdge& other) const
{
    return !operator==(other);  // 判断两个 TriEdge 对象是否不相等
}

std::ostream& operator<<(std::ostream& os, const TriEdge& tri_edge)
{
    return os << tri_edge.tri << ' ' << tri_edge.edge;  // 输出 TriEdge 对象的 tri 和 edge
}



XY::XY()
{}  // XY 类默认构造函数

XY::XY(const double& x_, const double& y_)
    : x(x_), y(y_)  // XY 类带参构造函数，初始化 x 和 y
{}

double XY::angle() const
{
    return atan2(y, x);  // 返回向量 (x, y) 的极角
}

double XY::cross_z(const XY& other) const
{
    return x * other.y - y * other.x;  // 计算向量 (x, y) 和另一个向量 other 的叉积在 z 方向上的分量
}

bool XY::is_right_of(const XY& other) const
{
    // 判断向量 (x, y) 是否位于向量 other 的右侧（考虑共线情况）
    if (x == other.x)
        return y > other.y;
    else
        return x > other.x;
}

bool XY::operator==(const XY& other) const
{
    return x == other.x && y == other.y;  // 判断两个 XY 对象是否相等
}

bool XY::operator!=(const XY& other) const
{
    return !operator==(other);  // 判断两个 XY 对象是否不相等
}

XY XY::operator*(const double& multiplier) const
{
    return XY(x * multiplier, y * multiplier);  // 返回当前向量乘以标量 multiplier 后的结果
}

const XY& XY::operator+=(const XY& other)
{
    x += other.x;
    y += other.y;
    return *this;  // 返回更新后的当前向量
}

const XY& XY::operator-=(const XY& other)
{
    x -= other.x;
    y -= other.y;
    return *this;  // 返回更新后的当前向量
}

XY XY::operator+(const XY& other) const
{
    return XY(x + other.x, y + other.y);  // 返回两个向量相加的结果
}

XY XY::operator-(const XY& other) const
{
    return XY(x - other.x, y - other.y);  // 返回两个向量相减的结果
}

std::ostream& operator<<(std::ostream& os, const XY& xy)
{
    return os << '(' << xy.x << ' ' << xy.y << ')';  // 输出 XY 对象的坐标
}



XYZ::XYZ(const double& x_, const double& y_, const double& z_)
    : x(x_), y(y_), z(z_)  // XYZ 类带参构造函数，初始化 x、y 和 z
{}

XYZ XYZ::cross(const XYZ& other) const
{
    // 计算当前向量和另一个向量 other 的叉乘结果
    return XYZ(y * other.z - z * other.y,
               z * other.x - x * other.z,
               x * other.y - y * other.x);
}

double XYZ::dot(const XYZ& other) const
{
    return x * other.x + y * other.y + z * other.z;  // 计算当前向量和另一个向量 other 的点积
}

XYZ XYZ::operator-(const XYZ& other) const
{
    return XYZ(x - other.x, y - other.y, z - other.z);  // 返回两个向量相减的结果
}

std::ostream& operator<<(std::ostream& os, const XYZ& xyz)
{
    return os << '(' << xyz.x << ' ' << xyz.y << ' ' << xyz.z << ')';  // 输出 XYZ 对象的坐标
}



BoundingBox::BoundingBox()
    : empty(true), lower(0.0, 0.0), upper(0.0, 0.0)
{}

void BoundingBox::add(const XY& point)
{
    if (empty) {
        empty = false;
        lower = upper = point;  // 第一次添加点时，设置 lower 和 upper 为该点
    } else {
        // 如果点的 x 坐标小于 lower.x，则更新 lower.x
        if      (point.x < lower.x) lower.x = point.x;
        // 否则，如果点的 x 坐标大于 upper.x，则更新 upper.x
        else if (point.x > upper.x) upper.x = point.x;

        // 如果点的 y 坐标小于 lower.y，则更新 lower.y
        if      (point.y < lower.y) lower.y = point.y;
        // 否则，如果点的 y 坐标大于 upper.y，则更新 upper.y
        else if (point.y > upper.y) upper.y = point.y;
    }
}

void BoundingBox::expand(const XY& delta)
{
    // 如果边界框不为空，扩展边界框的上下界
    if (!empty) {
        lower -= delta;  // 减去扩展量
        upper += delta;  // 加上扩展量
    }
}



ContourLine::ContourLine()
    : std::vector<XY>()
{}

void ContourLine::push_back(const XY& point)
{
    // 如果线条为空或者新点不等于最后一个点，则将新点加入线条
    if (empty() || point != back())
        std::vector<XY>::push_back(point);
}

void ContourLine::write() const
{
    // 输出包含点数量的轮廓线信息
    std::cout << "ContourLine of " << size() << " points:";
    for (const_iterator it = begin(); it != end(); ++it)
        std::cout << ' ' << *it;  // 输出每个点的坐标
    std::cout << std::endl;
}



void write_contour(const Contour& contour)
{
    // 输出包含线条数量的轮廓信息
    std::cout << "Contour of " << contour.size() << " lines." << std::endl;
    for (Contour::const_iterator it = contour.begin(); it != contour.end(); ++it)
        it->write();  // 输出每条轮廓线的信息
}



Triangulation::Triangulation(const CoordinateArray& x,
                             const CoordinateArray& y,
                             const TriangleArray& triangles,
                             const MaskArray& mask,
                             const EdgeArray& edges,
                             const NeighborArray& neighbors,
                             bool correct_triangle_orientations)
    : _x(x),
      _y(y),
      _triangles(triangles),
      _mask(mask),
      _edges(edges),
      _neighbors(neighbors)
{
    // 检查坐标数组是否为1维且长度相同
    if (_x.ndim() != 1 || _y.ndim() != 1 || _x.shape(0) != _y.shape(0))
        throw std::invalid_argument("x and y must be 1D arrays of the same length");

    // 检查三角形数组是否为2维且每行有3列
    if (_triangles.ndim() != 2 || _triangles.shape(1) != 3)
        throw std::invalid_argument("triangles must be a 2D array of shape (?,3)");

    // 可选的掩码数组检查
    if (_mask.size() > 0 &&
        (_mask.ndim() != 1 || _mask.shape(0) != _triangles.shape(0)))
        throw std::invalid_argument(
            "mask must be a 1D array with the same length as the triangles array");

    // 可选的边数组检查
    if (_edges.size() > 0 &&
        (_edges.ndim() != 2 || _edges.shape(1) != 2))
        throw std::invalid_argument("edges must be a 2D array with shape (?,2)");

    // 可选的邻居数组检查
    if (_neighbors.size() > 0 &&
        (_neighbors.ndim() != 2 || _neighbors.shape() != _triangles.shape()))
        throw std::invalid_argument(
            "neighbors must be a 2D array with the same shape as the triangles array");

    // 如果需要修正三角形方向，则调用修正函数
    if (correct_triangle_orientations)
        correct_triangles();
}

void Triangulation::calculate_boundaries()
{
    get_neighbors();  // 确保已经创建了邻居数组 _neighbors

    // 创建包含所有边界三角形边的集合，即没有邻居三角形的边
    typedef std::set<TriEdge> BoundaryEdges;
    BoundaryEdges boundary_edges;
    for (int tri = 0; tri < get_ntri(); ++tri) {
        if (!is_masked(tri)) {
            for (int edge = 0; edge < 3; ++edge) {
                if (get_neighbor(tri, edge) == -1) {
                    boundary_edges.insert(TriEdge(tri, edge));
                }
            }
        }
    }
    // 当boundary_edges非空时，取任意边界边并沿边界前进直到返回起始点，从boundary_edges中移除已使用的边。
    // 同时初始化_tri_edge_to_boundary_map。
    while (!boundary_edges.empty()) {
        // 新边界的起始点。
        BoundaryEdges::iterator it = boundary_edges.begin();
        int tri = it->tri;
        int edge = it->edge;
        _boundaries.push_back(Boundary());
        Boundary& boundary = _boundaries.back();

        // 循环处理当前边界直到完成整个边界的处理。
        while (true) {
            // 将当前三角形边界加入边界列表，并从boundary_edges中移除当前边。
            boundary.push_back(TriEdge(tri, edge));
            boundary_edges.erase(it);
            // 将当前三角形边界映射到边界地图_tri_edge_to_boundary_map中。
            _tri_edge_to_boundary_map[TriEdge(tri, edge)] =
                BoundaryEdge(_boundaries.size()-1, boundary.size()-1);

            // 移动到当前三角形的下一条边。
            edge = (edge+1) % 3;

            // 找到边界边的起始点索引。
            int point = get_triangle_point(tri, edge);

            // 通过遍历邻居三角形找到下一个TriEdge，直到找到没有邻居的三角形边界。
            while (get_neighbor(tri, edge) != -1) {
                tri = get_neighbor(tri, edge);
                edge = get_edge_in_triangle(tri, point);
            }

            // 如果到达当前边界的起始点，则完成当前边界的处理。
            if (TriEdge(tri,edge) == boundary.front())
                break;
            else
                it = boundary_edges.find(TriEdge(tri, edge));
        }
    }
// 计算三角网格的边
void Triangulation::calculate_edges()
{
    // 断言确保边数组为空，即尚未计算过边
    assert(!has_edges() && "Expected empty edges array");

    // 创建存储所有边的集合，以起点索引小于终点索引的方式存储
    typedef std::set<Edge> EdgeSet;
    EdgeSet edge_set;
    // 遍历每个三角形
    for (int tri = 0; tri < get_ntri(); ++tri) {
        if (!is_masked(tri)) {
            // 遍历当前三角形的每条边
            for (int edge = 0; edge < 3; edge++) {
                int start = get_triangle_point(tri, edge);  // 起点索引
                int end   = get_triangle_point(tri, (edge+1)%3);  // 终点索引
                // 将边插入集合，确保起点索引小于终点索引
                edge_set.insert(start > end ? Edge(start,end) : Edge(end,start));
            }
        }
    }

    // 将集合转换为 Python 的 _edges 数组
    py::ssize_t dims[2] = {static_cast<py::ssize_t>(edge_set.size()), 2};
    _edges = EdgeArray(dims);  // 初始化 _edges 数组
    auto edges = _edges.mutable_data();  // 获取可变数据指针

    int i = 0;
    // 将边集合中的数据复制到 _edges 数组中
    for (EdgeSet::const_iterator it = edge_set.begin(); it != edge_set.end(); ++it) {
        edges[i++] = it->start;  // 边的起点索引
        edges[i++] = it->end;    // 边的终点索引
    }
}

// 计算三角网格的邻居关系
void Triangulation::calculate_neighbors()
{
    // 断言确保邻居数组为空，即尚未计算过邻居关系
    assert(!has_neighbors() && "Expected empty neighbors array");

    // 创建形状为 (ntri, 3) 的 _neighbors 数组，并初始化所有元素为 -1
    py::ssize_t dims[2] = {get_ntri(), 3};
    _neighbors = NeighborArray(dims);  // 初始化 _neighbors 数组
    auto* neighbors = _neighbors.mutable_data();  // 获取可变数据指针

    int tri, edge;
    std::fill(neighbors, neighbors+3*get_ntri(), -1);  // 将 _neighbors 数组初始化为 -1

    // 使用映射从边到三角边的方式，为每条边找到相应的邻居边
    typedef std::map<Edge, TriEdge> EdgeToTriEdgeMap;
    EdgeToTriEdgeMap edge_to_tri_edge_map;
    // 遍历每个三角形
    for (tri = 0; tri < get_ntri(); ++tri) {
        if (!is_masked(tri)) {
            // 遍历当前三角形的每条边
            for (edge = 0; edge < 3; ++edge) {
                int start = get_triangle_point(tri, edge);  // 起点索引
                int end   = get_triangle_point(tri, (edge+1)%3);  // 终点索引
                EdgeToTriEdgeMap::iterator it =
                    edge_to_tri_edge_map.find(Edge(end,start));
                if (it == edge_to_tri_edge_map.end()) {
                    // 如果边映射中不存在相邻边，则将当前边添加到映射中
                    edge_to_tri_edge_map[Edge(start,end)] = TriEdge(tri,edge);
                } else {
                    // 找到相邻边，设置 _neighbors 中的两个元素，并从映射中移除边
                    neighbors[3*tri + edge] = it->second.tri;
                    neighbors[3*it->second.tri + it->second.edge] = tri;
                    edge_to_tri_edge_map.erase(it);
                }
            }
        }
    }

    // 剩余的边在边映射中对应边界边，但边界边的计算在其他地方单独处理
}
# 计算平面系数的函数，返回一个二维坐标数组，表示每个三角形平面的系数
Triangulation::TwoCoordinateArray Triangulation::calculate_plane_coefficients(
    const CoordinateArray& z)
{
    # 如果 z 不是一维数组，或者长度与三角剖分的 x 和 y 数组长度不同，则抛出异常
    if (z.ndim() != 1 || z.shape(0) != _x.shape(0))
        throw std::invalid_argument(
            "z must be a 1D array with the same length as the triangulation x and y arrays");

    # 初始化一个数组，用于存储平面系数
    int dims[2] = {get_ntri(), 3};
    Triangulation::TwoCoordinateArray planes_array(dims);
    auto planes = planes_array.mutable_unchecked<2>();  // 可变的二维数组引用
    auto triangles = _triangles.unchecked<2>();         // 不可变的二维数组引用
    auto x = _x.unchecked<1>();                         // x 坐标数组的不可变引用
    auto y = _y.unchecked<1>();                         // y 坐标数组的不可变引用
    auto z_ptr = z.unchecked<1>();                      // z 坐标数组的不可变引用

    int point;
    # 遍历每个三角形
    for (int tri = 0; tri < get_ntri(); ++tri) {
        # 如果三角形被掩蔽（masked），将平面系数置为0
        if (is_masked(tri)) {
            planes(tri, 0) = 0.0;
            planes(tri, 1) = 0.0;
            planes(tri, 2) = 0.0;
        }
        else {
            # 计算三角形的法向量
            point = triangles(tri, 0);
            XYZ point0(x(point), y(point), z_ptr(point));
            point = triangles(tri, 1);
            XYZ side01 = XYZ(x(point), y(point), z_ptr(point)) - point0;
            point = triangles(tri, 2);
            XYZ side02 = XYZ(x(point), y(point), z_ptr(point)) - point0;

            XYZ normal = side01.cross(side02);

            # 如果法向量在 z 方向上为零，说明三角形的点共线，使用伪逆避免除零错误
            if (normal.z == 0.0) {
                double sum2 = (side01.x*side01.x + side01.y*side01.y +
                               side02.x*side02.x + side02.y*side02.y);
                double a = (side01.x*side01.z + side02.x*side02.z) / sum2;
                double b = (side01.y*side01.z + side02.y*side02.z) / sum2;
                planes(tri, 0) = a;
                planes(tri, 1) = b;
                planes(tri, 2) = point0.z - a*point0.x - b*point0.y;
            }
            else {
                # 计算平面系数：-normal_x/normal_z, -normal_y/normal_z, normal.dot(point0)/normal_z
                planes(tri, 0) = -normal.x / normal.z;           // x 系数
                planes(tri, 1) = -normal.y / normal.z;           // y 系数
                planes(tri, 2) = normal.dot(point0) / normal.z;  // 常数项
            }
        }
    }

    # 返回计算得到的平面系数数组
    return planes_array;
}

# 修正三角形数据的函数
void Triangulation::correct_triangles()
{
    auto triangles = _triangles.mutable_data();    // 可变的三角形数据引用
    auto neighbors = _neighbors.mutable_data();    // 可变的邻居数据引用
    // 循环遍历三角形数组中的每一个三角形
    for (int tri = 0; tri < get_ntri(); ++tri) {
        // 获取当前三角形的第一个顶点坐标
        XY point0 = get_point_coords(triangles[3*tri]);
        // 获取当前三角形的第二个顶点坐标
        XY point1 = get_point_coords(triangles[3*tri+1]);
        // 获取当前三角形的第三个顶点坐标
        XY point2 = get_point_coords(triangles[3*tri+2]);
        
        // 检查三角形的顶点顺序是否为顺时针，如果是，则交换顶点顺序为逆时针
        if ( (point1 - point0).cross_z(point2 - point0) < 0.0) {
            // 交换第二个和第三个顶点，将顶点顺序改为逆时针
            std::swap(triangles[3*tri+1], triangles[3*tri+2]);
            // 如果存在邻居信息，也交换邻居顶点的顺序
            if (has_neighbors())
                std::swap(neighbors[3*tri+1], neighbors[3*tri+2]);
        }
    }
const Triangulation::Boundaries& Triangulation::get_boundaries() const
{
    // 如果边界信息为空，则计算边界信息
    if (_boundaries.empty())
        const_cast<Triangulation*>(this)->calculate_boundaries();
    // 返回边界信息的引用
    return _boundaries;
}

void Triangulation::get_boundary_edge(const TriEdge& triEdge,
                                      int& boundary,
                                      int& edge) const
{
    // 确保 _tri_edge_to_boundary_map 已经被创建
    get_boundaries();
    // 查找 triEdge 在 _tri_edge_to_boundary_map 中的映射
    TriEdgeToBoundaryMap::const_iterator it =
        _tri_edge_to_boundary_map.find(triEdge);
    // 断言 triEdge 存在于 _tri_edge_to_boundary_map 中，否则输出错误信息
    assert(it != _tri_edge_to_boundary_map.end() &&
           "TriEdge is not on a boundary");
    // 获取边界和边的信息
    boundary = it->second.boundary;
    edge = it->second.edge;
}

int Triangulation::get_edge_in_triangle(int tri, int point) const
{
    // 断言三角形索引在合法范围内
    assert(tri >= 0 && tri < get_ntri() && "Triangle index out of bounds");
    // 断言点的索引在合法范围内
    assert(point >= 0 && point < get_npoints() && "Point index out of bounds.");

    // 获取三角形数据的指针
    auto triangles = _triangles.data();

    // 遍历三角形的边
    for (int edge = 0; edge < 3; ++edge) {
        // 如果找到点在三角形中的边，返回边的索引
        if (triangles[3*tri + edge] == point)
            return edge;
    }
    // 如果点不在三角形中，返回 -1
    return -1;  // point is not in triangle.
}

Triangulation::EdgeArray& Triangulation::get_edges()
{
    // 如果还没有计算边信息，则进行计算
    if (!has_edges())
        calculate_edges();
    // 返回边的数组引用
    return _edges;
}

int Triangulation::get_neighbor(int tri, int edge) const
{
    // 断言三角形索引在合法范围内
    assert(tri >= 0 && tri < get_ntri() && "Triangle index out of bounds");
    // 断言边的索引在合法范围内
    assert(edge >= 0 && edge < 3 && "Edge index out of bounds");
    // 如果还没有计算邻居信息，则进行计算
    if (!has_neighbors())
        const_cast<Triangulation&>(*this).calculate_neighbors();
    // 返回邻居信息数组中的对应位置
    return _neighbors.data()[3*tri + edge];
}

TriEdge Triangulation::get_neighbor_edge(int tri, int edge) const
{
    // 获取邻居三角形的索引
    int neighbor_tri = get_neighbor(tri, edge);
    // 如果没有邻居，返回无效的 TriEdge
    if (neighbor_tri == -1)
        return TriEdge(-1,-1);
    else
        // 返回邻居三角形和共享边的 TriEdge
        return TriEdge(neighbor_tri,
                       // 获取共享边的第三个点在邻居三角形中的索引
                       get_edge_in_triangle(neighbor_tri,
                                            get_triangle_point(tri,
                                                               (edge+1)%3)));
}

Triangulation::NeighborArray& Triangulation::get_neighbors()
{
    // 如果还没有计算邻居信息，则进行计算
    if (!has_neighbors())
        calculate_neighbors();
    // 返回邻居信息数组的引用
    return _neighbors;
}

int Triangulation::get_npoints() const
{
    // 返回点的数量
    return _x.shape(0);
}

int Triangulation::get_ntri() const
{
    // 返回三角形的数量
    return _triangles.shape(0);
}

XY Triangulation::get_point_coords(int point) const
{
    // 断言点的索引在合法范围内
    assert(point >= 0 && point < get_npoints() && "Point index out of bounds.");
    // 返回点的坐标
    return XY(_x.data()[point], _y.data()[point]);
}

int Triangulation::get_triangle_point(int tri, int edge) const
{
    // 断言三角形索引在合法范围内
    assert(tri >= 0 && tri < get_ntri() && "Triangle index out of bounds");
    // 断言边的索引在合法范围内
    assert(edge >= 0 && edge < 3 && "Edge index out of bounds");
    // 返回三角形中边的顶点索引
    return _triangles.data()[3*tri + edge];
}

int Triangulation::get_triangle_point(const TriEdge& tri_edge) const
{
    // 调用另一个重载函数获取三角形中边的顶点索引
    return get_triangle_point(tri_edge.tri, tri_edge.edge);
}

bool Triangulation::has_edges() const
{
    // 返回是否已经计算了边的信息
    return _edges.size() > 0;
}
// 检查三角剖分对象是否有掩码数据
bool Triangulation::has_mask() const
{
    return _mask.size() > 0; // 返回掩码数据的大小是否大于0
}

// 检查三角剖分对象是否有邻居数据
bool Triangulation::has_neighbors() const
{
    return _neighbors.size() > 0; // 返回邻居数据的大小是否大于0
}

// 检查指定三角形是否被掩盖
bool Triangulation::is_masked(int tri) const
{
    assert(tri >= 0 && tri < get_ntri() && "Triangle index out of bounds."); // 断言：三角形索引在有效范围内
    return has_mask() && _mask.data()[tri]; // 返回该三角形是否被掩盖
}

// 设置三角剖分对象的掩码数据
void Triangulation::set_mask(const MaskArray& mask)
{
    if (mask.size() > 0 &&
        (mask.ndim() != 1 || mask.shape(0) != _triangles.shape(0)))
        throw std::invalid_argument(
            "mask must be a 1D array with the same length as the triangles array"); // 如果掩码数组不符合要求，抛出异常

    _mask = mask; // 设置掩码数据

    // 清除衍生字段，以便在需要时重新计算
    _edges = EdgeArray(); // 清空边缘数据
    _neighbors = NeighborArray(); // 清空邻居数据
    _boundaries.clear(); // 清空边界数据
}

// 打印三角剖分对象的边界信息
void Triangulation::write_boundaries() const
{
    const Boundaries& bs = get_boundaries(); // 获取边界信息
    std::cout << "Number of boundaries: " << bs.size() << std::endl; // 打印边界数量

    for (Boundaries::const_iterator it = bs.begin(); it != bs.end(); ++it) {
        const Boundary& b = *it;
        std::cout << "  Boundary of " << b.size() << " points: "; // 打印边界点数量

        for (Boundary::const_iterator itb = b.begin(); itb != b.end(); ++itb) {
            std::cout << *itb << ", "; // 打印边界点的索引
        }
        std::cout << std::endl;
    }
}

// 构造函数：三角轮廓生成器
TriContourGenerator::TriContourGenerator(Triangulation& triangulation,
                                         const CoordinateArray& z)
    : _triangulation(triangulation),
      _z(z),
      _interior_visited(2*_triangulation.get_ntri()),
      _boundaries_visited(0),
      _boundaries_used(0)
{
    if (_z.ndim() != 1 || _z.shape(0) != _triangulation.get_npoints())
        throw std::invalid_argument(
            "z must be a 1D array with the same length as the x and y arrays"); // 如果z数组不符合要求，抛出异常
}

// 清除访问标记：包括内部和可能的边界
void TriContourGenerator::clear_visited_flags(bool include_boundaries)
{
    // 清除内部访问标记
    std::fill(_interior_visited.begin(), _interior_visited.end(), false);

    if (include_boundaries) {
        if (_boundaries_visited.empty()) {
            const Boundaries& boundaries = get_boundaries(); // 获取边界信息

            // 初始化边界访问标记
            _boundaries_visited.reserve(boundaries.size());
            for (Boundaries::const_iterator it = boundaries.begin();
                    it != boundaries.end(); ++it)
                _boundaries_visited.push_back(BoundaryVisited(it->size()));

            // 初始化边界使用标记
            _boundaries_used = BoundariesUsed(boundaries.size());
        }

        // 清除边界访问标记
        for (BoundariesVisited::iterator it = _boundaries_visited.begin();
                it != _boundaries_visited.end(); ++it)
            std::fill(it->begin(), it->end(), false);

        // 清除边界使用标记
        std::fill(_boundaries_used.begin(), _boundaries_used.end(), false);
    }
}
    // 创建两个空的 Python 列表，用于存储顶点和代码
    py::list vertices_list(contour.size());
    py::list codes_list(contour.size());

    // 遍历每一个轮廓线
    for (Contour::size_type i = 0; i < contour.size(); ++i) {
        // 获取当前轮廓线的引用
        const ContourLine& contour_line = contour[i];
        // 获取当前轮廓线上点的数量
        py::ssize_t npoints = static_cast<py::ssize_t>(contour_line.size());

        // 创建一个二维数组，存储每个点的 (x, y) 坐标
        py::ssize_t segs_dims[2] = {npoints, 2};
        CoordinateArray segs(segs_dims);
        double* segs_ptr = segs.mutable_data();

        // 创建一个一维数组，存储每个点的代码
        py::ssize_t codes_dims[1] = {npoints};
        CodeArray codes(codes_dims);
        unsigned char* codes_ptr = codes.mutable_data();

        // 遍历轮廓线上的每一个点
        for (ContourLine::const_iterator it = contour_line.begin();
             it != contour_line.end(); ++it) {
            // 将点的坐标 (x, y) 存入数组 segs
            *segs_ptr++ = it->x;
            *segs_ptr++ = it->y;
            // 设置每个点的代码，第一个点为 MOVETO，其余点为 LINETO
            *codes_ptr++ = (it == contour_line.begin() ? MOVETO : LINETO);
        }

        // 如果轮廓线是封闭的，最后一个点的代码设为 CLOSEPOLY
        if (contour_line.size() > 1 &&
            contour_line.front() == contour_line.back())
            *(codes_ptr-1) = CLOSEPOLY;

        // 将顶点数组和代码数组存入 Python 列表中
        vertices_list[i] = segs;
        codes_list[i] = codes;
    }

    // 返回存储顶点和代码的 Python 元组
    return py::make_tuple(vertices_list, codes_list);
py::tuple TriContourGenerator::contour_to_segs_and_kinds(const Contour& contour)
{
    // 将通过调用 create_filled_contour() 生成的所有多边形转换为它们的 Python 等效形式，
    // 以便返回给调用函数。所有多边形的点和种类代码被组合成单个 NumPy 数组，
    // 每个数组包含以下内容，避免确定哪些多边形是孔，因为这将由渲染器决定。
    // 如果所有多边形中总共有 ntotal 个点，那么创建的两个 NumPy 数组为：
    //   vertices 是一个形状为 (ntotal, 2) 的双精度数组，包含多边形中点的 (x, y) 坐标
    //   codes 是一个形状为 (ntotal,) 的 uint8 数组，包含路径类中定义的“种类代码”
    // 然后它们分别在 Python 列表 vertices_list 和 codes_list 中返回。

    Contour::const_iterator line;  // 声明轮廓的线条迭代器
    ContourLine::const_iterator point;  // 声明轮廓线条中点的迭代器

    // 计算所有轮廓线中点的总数
    py::ssize_t n_points = 0;
    for (line = contour.begin(); line != contour.end(); ++line)
        n_points += static_cast<py::ssize_t>(line->size());

    // 创建用于点坐标的 segs 数组
    py::ssize_t segs_dims[2] = {n_points, 2};
    TwoCoordinateArray segs(segs_dims);
    double* segs_ptr = segs.mutable_data();

    // 创建用于代码类型的 codes 数组
    py::ssize_t codes_dims[1] = {n_points};
    CodeArray codes(codes_dims);
    unsigned char* codes_ptr = codes.mutable_data();

    // 遍历每条轮廓线及其点，填充 segs 和 codes 数组
    for (line = contour.begin(); line != contour.end(); ++line) {
        for (point = line->begin(); point != line->end(); point++) {
            *segs_ptr++ = point->x;
            *segs_ptr++ = point->y;
            *codes_ptr++ = (point == line->begin() ? MOVETO : LINETO);
        }

        // 如果线条的长度大于1，则将最后一个点标记为 CLOSEPOLY
        if (line->size() > 1)
            *(codes_ptr - 1) = CLOSEPOLY;
    }

    // 创建 Python 列表 vertices_list 和 codes_list，将 segs 和 codes 数组作为其元素
    py::list vertices_list(1);
    vertices_list[0] = segs;

    py::list codes_list(1);
    codes_list[0] = codes;

    // 返回包含 vertices_list 和 codes_list 的 Python 元组
    return py::make_tuple(vertices_list, codes_list);
}

py::tuple TriContourGenerator::create_contour(const double& level)
{
    // 清除访问标志，准备新的轮廓
    clear_visited_flags(false);
    Contour contour;

    // 找到轮廓的边界线
    find_boundary_lines(contour, level);

    // 找到轮廓的内部线
    find_interior_lines(contour, level, false, false);

    // 将轮廓转换为点和种类代码数组，并返回
    return contour_to_segs_and_kinds(contour);
}

py::tuple TriContourGenerator::create_filled_contour(const double& lower_level,
                                                     const double& upper_level)
{
    // 检查填充轮廓的级别是否递增，否则抛出异常
    if (lower_level >= upper_level)
        throw std::invalid_argument("filled contour levels must be increasing");

    // 清除访问标志，准备新的轮廓
    clear_visited_flags(true);
    Contour contour;

    // 找到填充轮廓的边界线
    find_boundary_lines_filled(contour, lower_level, upper_level);

    // 找到填充轮廓的内部线
    find_interior_lines(contour, lower_level, false, true);
    find_interior_lines(contour, upper_level, true, true);

    // 将填充轮廓转换为点和种类代码数组，并返回
    return contour_to_segs_and_kinds(contour);
}
// 根据给定的三角形、边和水平值，进行边界线内插计算，并返回插值结果
XY TriContourGenerator::edge_interp(int tri, int edge, const double& level)
{
    // 调用interp函数对三角形边界进行内插计算，返回内插结果
    return interp(_triangulation.get_triangle_point(tri, edge),
                  _triangulation.get_triangle_point(tri, (edge+1)%3),
                  level);
}

// 查找边界线的起始点，生成轮廓线并添加到contour中
void TriContourGenerator::find_boundary_lines(Contour& contour,
                                              const double& level)
{
    // 获取三角剖分和边界
    const Triangulation& triang = _triangulation;
    const Boundaries& boundaries = get_boundaries();
    
    // 遍历所有边界
    for (Boundaries::const_iterator it = boundaries.begin();
            it != boundaries.end(); ++it) {
        // 获取当前边界
        const Boundary& boundary = *it;
        
        // 初始化起始点和结束点的水平位置判断
        bool startAbove, endAbove = false;
        
        // 遍历边界的每条边
        for (Boundary::const_iterator itb = boundary.begin();
                itb != boundary.end(); ++itb) {
            // 对于每条边的第一个点，判断其水平位置是否大于等于给定的水平值
            if (itb == boundary.begin())
                startAbove = get_z(triang.get_triangle_point(*itb)) >= level;
            else
                startAbove = endAbove;
            
            // 对于每条边的下一个点，判断其水平位置是否大于等于给定的水平值
            endAbove = get_z(triang.get_triangle_point(itb->tri,
                                                       (itb->edge+1)%3)) >= level;
            
            // 如果起点在水平值上方且终点在水平值下方，开始一个新的轮廓线
            if (startAbove && !endAbove) {
                // 在contour中添加一个新的轮廓线
                contour.push_back(ContourLine());
                ContourLine& contour_line = contour.back();
                
                // 获取当前边的TriEdge
                TriEdge tri_edge = *itb;
                
                // 跟踪轮廓线的内部路径
                follow_interior(contour_line, tri_edge, true, level, false);
            }
        }
    }
}

// 查找填充边界线的起始点，生成填充轮廓线并添加到contour中
void TriContourGenerator::find_boundary_lines_filled(Contour& contour,
                                                     const double& lower_level,
                                                     const double& upper_level)
{
    // 获取三角剖分和边界
    const Triangulation& triang = _triangulation;
    const Boundaries& boundaries = get_boundaries();
    
    // 遍历所有边界
    for (Boundaries::const_iterator it = boundaries.begin();
            it != boundaries.end(); ++it) {
        // 获取当前边界
        const Boundary& boundary = *it;
        
        // 初始化起始点和结束点的水平位置判断
        bool startAbove, endAbove = false;
        
        // 遍历边界的每条边
        for (Boundary::const_iterator itb = boundary.begin();
                itb != boundary.end(); ++itb) {
            // 对于每条边的第一个点，判断其水平位置是否在上限值以上
            if (itb == boundary.begin())
                startAbove = get_z(triang.get_triangle_point(*itb)) >= upper_level;
            else
                startAbove = endAbove;
            
            // 对于每条边的下一个点，判断其水平位置是否在下限值以下
            endAbove = get_z(triang.get_triangle_point(itb->tri,
                                                       (itb->edge+1)%3)) >= lower_level;
            
            // 如果起点在上限值以上且终点在下限值以下，开始一个新的填充轮廓线
            if (startAbove && !endAbove) {
                // 在contour中添加一个新的轮廓线
                contour.push_back(ContourLine());
                ContourLine& contour_line = contour.back();
                
                // 获取当前边的TriEdge
                TriEdge tri_edge = *itb;
                
                // 跟踪轮廓线的内部路径
                follow_interior(contour_line, tri_edge, true, upper_level, true);
            }
        }
    }
}
    // 遍历所有边界
    for (Boundaries::size_type i = 0; i < boundaries.size(); ++i) {
        // 获取当前边界对象
        const Boundary& boundary = boundaries[i];
        // 遍历当前边界的所有边
        for (Boundary::size_type j = 0; j < boundary.size(); ++j) {
            // 检查当前边界是否已被访问过
            if (!_boundaries_visited[i][j]) {
                // 获取当前边界边的起点和终点的 z 值
                double z_start = get_z(triang.get_triangle_point(boundary[j]));
                double z_end = get_z(triang.get_triangle_point(
                                   boundary[j].tri, (boundary[j].edge+1)%3));

                // 判断当前边界边的 z 值是否在上限和下限之间
                bool incr_upper = (z_start < upper_level && z_end >= upper_level);
                bool decr_lower = (z_start >= lower_level && z_end < lower_level);

                // 如果边界边的 z 值在上限或下限之间
                if (decr_lower || incr_upper) {
                    // 添加新的轮廓线
                    contour.push_back(ContourLine());
                    ContourLine& contour_line = contour.back();
                    TriEdge start_tri_edge = boundary[j];
                    TriEdge tri_edge = start_tri_edge;

                    // 跟随内部轮廓线直到返回起点
                    bool on_upper = incr_upper;
                    do {
                        follow_interior(contour_line, tri_edge, true,
                            on_upper ? upper_level : lower_level, on_upper);
                        on_upper = follow_boundary(contour_line, tri_edge,
                                       lower_level, upper_level, on_upper);
                    } while (tri_edge != start_tri_edge);

                    // 封闭多边形
                    contour_line.push_back(contour_line.front());
                }
            }
        }
    }

    // 添加那些处于下限和上限之间的完整边界。这些边界未被内部轮廓线触及，其状态存储在 _boundaries_used 中。
    for (Boundaries::size_type i = 0; i < boundaries.size(); ++i) {
        // 如果当前边界未被使用
        if (!_boundaries_used[i]) {
            // 获取当前边界对象
            const Boundary& boundary = boundaries[i];
            // 获取边界起始点的 z 值
            double z = get_z(triang.get_triangle_point(boundary[0]));
            // 如果 z 值在下限和上限之间
            if (z >= lower_level && z < upper_level) {
                // 添加新的轮廓线
                contour.push_back(ContourLine());
                ContourLine& contour_line = contour.back();
                // 将边界的点坐标添加到轮廓线中
                for (Boundary::size_type j = 0; j < boundary.size(); ++j)
                    contour_line.push_back(triang.get_point_coords(
                                      triang.get_triangle_point(boundary[j])));

                // 封闭多边形
                contour_line.push_back(contour_line.front());
            }
        }
    }
// 结束 TriContourGenerator 类的成员函数 find_interior_lines 的定义
void TriContourGenerator::find_interior_lines(Contour& contour,
                                              const double& level,
                                              bool on_upper,
                                              bool filled)
{
    // 获取三角剖分的引用
    const Triangulation& triang = _triangulation;
    // 获取三角形数量
    int ntri = triang.get_ntri();
    // 循环遍历每个三角形
    for (int tri = 0; tri < ntri; ++tri) {
        // 根据是否在上部（on_upper）决定 visited_index
        int visited_index = (on_upper ? tri+ntri : tri);

        // 如果该三角形已经被访问或者被遮蔽，则跳过
        if (_interior_visited[visited_index] || triang.is_masked(tri))
            continue;  // Triangle has already been visited or is masked.

        // 标记该三角形已被访问
        _interior_visited[visited_index] = true;

        // 确定离开该三角形的边
        int edge = get_exit_edge(tri, level, on_upper);
        // 断言确保边的索引在有效范围内
        assert(edge >= -1 && edge < 3 && "Invalid exit edge");
        // 如果 edge 为 -1，则表示等值线不通过该三角形，继续下一个三角形
        if (edge == -1)
            continue;  // Contour does not pass through this triangle.

        // 发现新的等值线环的起始点
        contour.push_back(ContourLine());
        ContourLine& contour_line = contour.back();
        // 获取与给定边相邻的三角形边
        TriEdge tri_edge = triang.get_neighbor_edge(tri, edge);
        // 根据起始边跟踪内部等值线
        follow_interior(contour_line, tri_edge, false, level, on_upper);

        // 将线环闭合
        contour_line.push_back(contour_line.front());
    }
}

// TriContourGenerator 类的成员函数 follow_boundary 的定义
bool TriContourGenerator::follow_boundary(ContourLine& contour_line,
                                          TriEdge& tri_edge,
                                          const double& lower_level,
                                          const double& upper_level,
                                          bool on_upper)
{
    // 获取三角剖分的引用
    const Triangulation& triang = _triangulation;
    // 获取边界集合的引用
    const Boundaries& boundaries = get_boundaries();

    // 根据三角边获取等效的边界边
    int boundary, edge;
    triang.get_boundary_edge(tri_edge, boundary, edge);
    // 标记该边界已被使用
    _boundaries_used[boundary] = true;

    // 初始化停止标志和起始边标志
    bool stop = false;
    bool first_edge = true;
    double z_start, z_end = 0;
    // 循环直到停止条件满足
    while (!stop)
    {
        // 确保当前边界边未被访问过，如果已经访问过，则抛出断言错误信息
        assert(!_boundaries_visited[boundary][edge] && "Boundary already visited");
        // 标记当前边界边为已访问
        _boundaries_visited[boundary][edge] = true;
    
        // 计算边界边起点和终点的 z 值。
        // 如果是第一条边界边，则起点的 z 值由三角形边获取。
        // 否则，起点的 z 值为上一条边界边的终点 z 值。
        if (first_edge)
            z_start = get_z(triang.get_triangle_point(tri_edge));
        else
            z_start = z_end;
    
        // 终点的 z 值由当前三角形边的下一个顶点决定。
        z_end = get_z(triang.get_triangle_point(tri_edge.tri, (tri_edge.edge + 1) % 3));
    
        // 根据 z 值的变化情况进行处理
        if (z_end > z_start) {  // z 值增加
            // 如果不处于上界且是第一条边，则终点 z 值大于等于下界且起点 z 值小于下界，则停止
            if (!(!on_upper && first_edge) &&
                z_end >= lower_level && z_start < lower_level) {
                stop = true;
                on_upper = false;
            } else if (z_end >= upper_level && z_start < upper_level) {
                // 终点 z 值大于等于上界且起点 z 值小于上界，则停止，并切换至上界状态
                stop = true;
                on_upper = true;
            }
        } else {  // z 值减少
            // 如果处于上界且不是第一条边，则起点 z 值大于等于上界且终点 z 值小于上界，则停止
            if (!(on_upper && first_edge) &&
                z_start >= upper_level && z_end < upper_level) {
                stop = true;
                on_upper = true;
            } else if (z_start >= lower_level && z_end < lower_level) {
                // 起点 z 值大于等于下界且终点 z 值小于下界，则停止，并切换至下界状态
                stop = true;
                on_upper = false;
            }
        }
    
        // 标记已处理完第一条边
        first_edge = false;
    
        // 如果未停止，则移动到下一条边界边，将相应的点添加到轮廓线中
        if (!stop) {
            // 计算下一条边界边的索引，添加其对应点的坐标到轮廓线中
            edge = (edge + 1) % (int)boundaries[boundary].size();
            tri_edge = boundaries[boundary][edge];
            contour_line.push_back(triang.get_point_coords(
                                       triang.get_triangle_point(tri_edge)));
        }
    }
    
    // 返回是否处于上界状态
    return on_upper;
// 将给定边上的插值点添加到轮廓线中
void TriContourGenerator::follow_interior(ContourLine& contour_line,
                                          TriEdge& tri_edge,
                                          bool end_on_boundary,
                                          const double& level,
                                          bool on_upper)
{
    int& tri = tri_edge.tri;  // 当前三角形的索引
    int& edge = tri_edge.edge;  // 当前三角形的边的索引

    // 初始点
    contour_line.push_back(edge_interp(tri, edge, level));

    while (true) {
        int visited_index = tri;
        if (on_upper)
            visited_index += _triangulation.get_ntri();  // 如果在上层三角形，索引偏移为总三角形数

        // 如果不是以边界结束且已访问过当前三角形，则跳出循环
        if (!end_on_boundary && _interior_visited[visited_index])
            break;

        // 确定离开当前三角形的边
        edge = get_exit_edge(tri, level, on_upper);
        assert(edge >= 0 && edge < 3 && "Invalid exit edge");  // 断言：边的索引合法性检查

        _interior_visited[visited_index] = true;  // 标记当前三角形为已访问

        // 添加新点到轮廓线点集合
        assert(edge >= 0 && edge < 3 && "Invalid triangle edge");  // 断言：三角形边的索引合法性检查
        contour_line.push_back(edge_interp(tri, edge, level));

        // 移动到下一个三角形
        TriEdge next_tri_edge = _triangulation.get_neighbor_edge(tri, edge);

        // 如果以边界结束，且下一个三角形为无效（-1），则跳出循环
        if (end_on_boundary && next_tri_edge.tri == -1)
            break;

        tri_edge = next_tri_edge;  // 更新当前三角形和边的索引
        assert(tri_edge.tri != -1 && "Invalid triangle for internal loop");  // 断言：内部循环中的三角形索引合法性检查
    }
}

// 返回三角剖分的边界
const TriContourGenerator::Boundaries& TriContourGenerator::get_boundaries() const
{
    return _triangulation.get_boundaries();
}

// 根据给定三角形和阈值级别获取退出边的索引
int TriContourGenerator::get_exit_edge(int tri,
                                       const double& level,
                                       bool on_upper) const
{
    assert(tri >= 0 && tri < _triangulation.get_ntri() &&
           "Triangle index out of bounds.");  // 断言：三角形索引合法性检查

    // 根据顶点的 Z 值和阈值级别计算配置值
    unsigned int config =
        (get_z(_triangulation.get_triangle_point(tri, 0)) >= level) |
        (get_z(_triangulation.get_triangle_point(tri, 1)) >= level) << 1 |
        (get_z(_triangulation.get_triangle_point(tri, 2)) >= level) << 2;

    if (on_upper) config = 7 - config;  // 如果在上层三角形，调整配置值

    // 根据配置值返回相应的退出边索引
    switch (config) {
        case 0: return -1;
        case 1: return 2;
        case 2: return 0;
        case 3: return 2;
        case 4: return 1;
        case 5: return 1;
        case 6: return 0;
        case 7: return -1;
        default: assert(0 && "Invalid config value"); return -1;  // 断言：配置值合法性检查
    }
}

// 返回指定点的 Z 值
const double& TriContourGenerator::get_z(int point) const
{
    assert(point >= 0 && point < _triangulation.get_npoints() &&
           "Point index out of bounds.");  // 断言：点索引合法性检查
    return _z.data()[point];
}

// 对两个点之间进行插值
XY TriContourGenerator::interp(int point1,
                               int point2,
                               const double& level) const
{
    assert(point1 >= 0 && point1 < _triangulation.get_npoints() &&
           "Point index 1 out of bounds.");  // 断言：第一个点索引合法性检查
    // ...
}
    # 断言：验证 point2 处于有效范围内，即大于等于 0 并且小于 _triangulation 对象的点数。
    # 如果不满足条件，输出错误信息 "Point index 2 out of bounds."
    assert(point2 >= 0 && point2 < _triangulation.get_npoints() &&
           "Point index 2 out of bounds.");
    
    # 断言：验证 point1 和 point2 不相同。
    # 如果相同，输出错误信息 "Identical points"
    assert(point1 != point2 && "Identical points");
    
    # 计算插值因子 fraction，用于根据 point1 和 point2 之间的高度差和水平面 level 的比例确定插值权重。
    double fraction = (get_z(point2) - level) / (get_z(point2) - get_z(point1));
    
    # 返回根据插值因子计算出的两个点之间的插值坐标。
    return _triangulation.get_point_coords(point1)*fraction +
           _triangulation.get_point_coords(point2)*(1.0 - fraction);
}

// TrapezoidMapTriFinder 类的构造函数，初始化成员变量 _triangulation, _points 和 _tree
TrapezoidMapTriFinder::TrapezoidMapTriFinder(Triangulation& triangulation)
    : _triangulation(triangulation),
      _points(0),
      _tree(0)
{}

// TrapezoidMapTriFinder 类的析构函数，清理资源
TrapezoidMapTriFinder::~TrapezoidMapTriFinder()
{
    clear();
}

// 将边添加到搜索树中
bool
TrapezoidMapTriFinder::add_edge_to_tree(const Edge& edge)
{
    // 查找与边相交的梯形
    std::vector<Trapezoid*> trapezoids;
    if (!find_trapezoids_intersecting_edge(edge, trapezoids))
        return false;
    assert(!trapezoids.empty() && "No trapezoids intersect edge");

    // 左端点和右端点
    const Point* p = edge.left;
    const Point* q = edge.right;
    Trapezoid* left_old = 0;    // 左边的旧梯形
    Trapezoid* left_below = 0;  // 左下方的梯形
    Trapezoid* left_above = 0;  // 左上方的梯形

    // 遍历与边相交的梯形，从左到右
    // 用2个或更多新梯形替换每个旧梯形，并用新节点替换搜索树中的对应节点
    size_t ntraps = trapezoids.size();
    }

    return true;
}

// 清理资源，释放内存
void
TrapezoidMapTriFinder::clear()
{
    delete [] _points;  // 释放 _points 数组的内存
    _points = 0;

    _edges.clear();     // 清空 _edges 容器

    delete _tree;       // 删除 _tree 搜索树对象
    _tree = 0;
}

// 查找多个点在三角网格中的索引
TrapezoidMapTriFinder::TriIndexArray
TrapezoidMapTriFinder::find_many(const CoordinateArray& x,
                                 const CoordinateArray& y)
{
    if (x.ndim() != 1 || x.shape(0) != y.shape(0))
        throw std::invalid_argument(
            "x and y must be array-like with same shape");

    // 创建用于返回的整数数组
    auto n = x.shape(0);
    TriIndexArray tri_indices_array(n);
    auto tri_indices = tri_indices_array.mutable_unchecked<1>();
    auto x_data = x.data();
    auto y_data = y.data();

    // 填充返回的数组
    for (py::ssize_t i = 0; i < n; ++i)
        tri_indices(i) = find_one(XY(x_data[i], y_data[i]));

    return tri_indices_array;
}

// 查找单个点在三角网格中的索引
int
TrapezoidMapTriFinder::find_one(const XY& xy)
{
    const Node* node = _tree->search(xy);
    assert(node != 0 && "Search tree for point returned null node");
    return node->get_tri();
}

// 查找与给定边相交的梯形
bool
TrapezoidMapTriFinder::find_trapezoids_intersecting_edge(
    const Edge& edge,
    std::vector<Trapezoid*>& trapezoids)
{
    // 这是 de Berg 等人的 FollowSegment 算法，加入了一些额外的检查以处理简单共线（无效）的三角形
    trapezoids.clear();  // 清空存储相交梯形的向量
    Trapezoid* trapezoid = _tree->search(edge);  // 在搜索树中查找与边相交的梯形
    if (trapezoid == 0) {
        assert(trapezoid != 0 && "search(edge) returns null trapezoid");
        return false;
    }

    trapezoids.push_back(trapezoid);  // 将找到的梯形加入到 trapezoids 中
    // 当边的右端点在当前梯形的右边时，执行循环
    while (edge.right->is_right_of(*trapezoid->right)) {
        // 获取边和当前梯形右端点的相对方向
        int orient = edge.get_point_orientation(*trapezoid->right);
        // 如果方向为0，表示边经过当前梯形的右端点
        if (orient == 0) {
            // 根据边的上下点确定方向
            if (edge.point_below == trapezoid->right)
                orient = +1;
            else if (edge.point_above == trapezoid->right)
                orient = -1;
            else {
                // 如果边的点在边上，抛出断言错误并返回false
                assert(0 && "Unable to deal with point on edge");
                return false;
            }
        }

        // 根据方向移动到相邻的梯形
        if (orient == -1)
            trapezoid = trapezoid->lower_right;
        else if (orient == +1)
            trapezoid = trapezoid->upper_right;

        // 如果移动后的梯形为空，抛出断言错误并返回false
        if (trapezoid == 0) {
            assert(0 && "Expected trapezoid neighbor");
            return false;
        }
        
        // 将当前梯形添加到梯形列表中
        trapezoids.push_back(trapezoid);
    }

    // 如果成功完成循环，返回true
    return true;
}

py::list
TrapezoidMapTriFinder::get_tree_stats()
{
    NodeStats stats;  // 创建一个 NodeStats 结构体实例 stats

    // 调用 _tree 对象的 get_stats 方法，计算树的统计信息并存储在 stats 中
    _tree->get_stats(0, stats);

    // 创建一个长度为 7 的 Python 列表 ret，用于存储树的统计信息
    py::list ret(7);

    // 将不同的统计信息分别存入 ret 列表中的不同位置
    ret[0] = stats.node_count;  // 存储节点总数
    ret[1] = stats.unique_nodes.size();  // 存储唯一节点的数量
    ret[2] = stats.trapezoid_count;  // 存储梯形的总数
    ret[3] = stats.unique_trapezoid_nodes.size();  // 存储唯一梯形节点的数量
    ret[4] = stats.max_parent_count;  // 存储最大父节点数量
    ret[5] = stats.max_depth;  // 存储树的最大深度
    ret[6] = stats.sum_trapezoid_depth / stats.trapezoid_count;  // 存储平均梯形深度

    // 返回包含树统计信息的 Python 列表 ret
    return ret;
}

void
TrapezoidMapTriFinder::initialize()
{
    clear();  // 调用 clear 方法，初始化对象状态

    const Triangulation& triang = _triangulation;  // 获取引用的三角剖分对象

    // 设置点数组 _points，包含所有三角剖分的点以及包围矩形的四个角点
    int npoints = triang.get_npoints();  // 获取三角剖分中点的数量
    _points = new Point[npoints + 4];  // 动态分配点数组空间，大小为 npoints + 4
    BoundingBox bbox;  // 创建边界框对象 bbox

    // 遍历三角剖分中的每个点，初始化 _points 数组，并更新边界框 bbox
    for (int i = 0; i < npoints; ++i) {
        XY xy = triang.get_point_coords(i);  // 获取第 i 个点的坐标
        // 处理避免 -0.0 值与 0.0 不同的问题
        if (xy.x == -0.0)
            xy.x = 0.0;
        if (xy.y == -0.0)
            xy.y = 0.0;
        _points[i] = Point(xy);  // 使用坐标创建 Point 对象并存入 _points 数组
        bbox.add(xy);  // 将点的坐标添加到边界框中
    }

    // 最后的四个点是包围矩形的角点。为防止角点已在三角剖分中，稍微扩展边界矩形。
    if (bbox.empty) {
        bbox.add(XY(0.0, 0.0));
        bbox.add(XY(1.0, 1.0));
    }
    else {
        const double small = 0.1;  // 任意大于 0.0 的值
        bbox.expand( (bbox.upper - bbox.lower) * small );  // 根据边界框的大小扩展边界
    }

    // 设置四个角点到 _points 数组中的最后四个位置
    _points[npoints  ] = Point(bbox.lower);  // SW 点
    _points[npoints+1] = Point(bbox.upper.x, bbox.lower.y);  // SE 点
    _points[npoints+2] = Point(bbox.lower.x, bbox.upper.y);  // NW 点
    _points[npoints+3] = Point(bbox.upper);  // NE 点

    // 设置边数组 _edges
    // 首先是包围矩形的底部和顶部边
    _edges.push_back(Edge(&_points[npoints],  &_points[npoints+1], -1, -1, 0, 0));  // 底部边
    _edges.push_back(Edge(&_points[npoints+2], &_points[npoints+3], -1, -1, 0, 0));  // 顶部边

    // 添加所有指向右边的三角剖分中的边。不显式包括指向左边的边，因为相邻的三角形会提供，除非没有这样的邻居。
    int ntri = triang.get_ntri();  // 获取三角剖分中的三角形数量
    // 遍历每一个三角形
    for (int tri = 0; tri < ntri; ++tri) {
        // 检查当前三角形是否被标记，如果未被标记则执行以下操作
        if (!triang.is_masked(tri)) {
            // 遍历当前三角形的三条边
            for (int edge = 0; edge < 3; ++edge) {
                // 获取当前边的起始点、结束点和第三个顶点
                Point* start = _points + triang.get_triangle_point(tri, edge);
                Point* end   = _points + triang.get_triangle_point(tri, (edge + 1) % 3);
                Point* other = _points + triang.get_triangle_point(tri, (edge + 2) % 3);

                // 获取当前边的邻居边信息
                TriEdge neighbor = triang.get_neighbor_edge(tri, edge);

                // 如果结束点在起始点的右侧
                if (end->is_right_of(*start)) {
                    // 如果邻居三角形不存在（即边界边），将新边加入边集合
                    const Point* neighbor_point_below = (neighbor.tri == -1) ?
                        0 : _points + triang.get_triangle_point(neighbor.tri, (neighbor.edge + 2) % 3);
                    _edges.push_back(Edge(start, end, neighbor.tri, tri,
                                          neighbor_point_below, other));
                }
                // 如果邻居三角形不存在且结束点不在起始点右侧，将新边加入边集合
                else if (neighbor.tri == -1)
                    _edges.push_back(Edge(end, start, tri, -1, other, 0));

                // 如果起始点关联的三角形未设置，则将当前三角形标记为其关联的三角形
                if (start->tri == -1)
                    start->tri = tri;
            }
        }
    }

    // 初始化第一个梯形为包围矩形
    _tree = new Node(new Trapezoid(&_points[npoints], &_points[npoints+1],
                                   _edges[0], _edges[1]));
    // 断言树的有效性（用于调试目的）
    _tree->assert_valid(false);

    // 使用随机数生成器对除了前两条边以外的所有边进行随机打乱
    std::mt19937 rng(1234);
    std::shuffle(_edges.begin() + 2, _edges.end(), rng);

    // 逐个将边加入到树中
    size_t nedges = _edges.size();
    for (size_t index = 2; index < nedges; ++index) {
        // 将边添加到树中，如果添加失败则抛出异常
        if (!add_edge_to_tree(_edges[index]))
            throw std::runtime_error("Triangulation is invalid");
        // 断言树的有效性（用于调试目的），最后一次循环时检查整棵树的有效性
        _tree->assert_valid(index == nedges - 1);
    }
}

void
TrapezoidMapTriFinder::print_tree()
{
    // 确保树节点不为空，如果为空则断言失败
    assert(_tree != 0 && "Null Node tree");
    // 调用树节点的打印方法打印整棵树
    _tree->print();
}

TrapezoidMapTriFinder::Edge::Edge(const Point* left_,
                                  const Point* right_,
                                  int triangle_below_,
                                  int triangle_above_,
                                  const Point* point_below_,
                                  const Point* point_above_)
    : left(left_),
      right(right_),
      triangle_below(triangle_below_),
      triangle_above(triangle_above_),
      point_below(point_below_),
      point_above(point_above_)
{
    // 确保左点不为空，如果为空则断言失败
    assert(left != 0 && "Null left point");
    // 确保右点不为空，如果为空则断言失败
    assert(right != 0 && "Null right point");
    // 确保右点在左点的右侧，如果不是则断言失败
    assert(right->is_right_of(*left) && "Incorrect point order");
    // 确保三角形下方索引大于等于-1，如果小于则断言失败
    assert(triangle_below >= -1 && "Invalid triangle below index");
    // 确保三角形上方索引大于等于-1，如果小于则断言失败
    assert(triangle_above >= -1 && "Invalid triangle above index");
}

int
TrapezoidMapTriFinder::Edge::get_point_orientation(const XY& xy) const
{
    // 计算点相对边的方向，根据叉乘结果判断点在边的左侧(+1)、右侧(-1)还是共线(0)
    double cross_z = (xy - *left).cross_z(*right - *left);
    return (cross_z > 0.0) ? +1 : ((cross_z < 0.0) ? -1 : 0);
}

double
TrapezoidMapTriFinder::Edge::get_slope() const
{
    // 计算边的斜率，这里允许除以零
    XY diff = *right - *left;
    return diff.y / diff.x;
}

double
TrapezoidMapTriFinder::Edge::get_y_at_x(const double& x) const
{
    if (left->x == right->x) {
        // 如果边是垂直的，返回左端点的较低的y值
        assert(x == left->x && "x outside of edge");
        return left->y;
    }
    else {
        // 计算在给定x处边上的y值，基于线性插值
        double lambda = (x - left->x) / (right->x - left->x);
        assert(lambda >= 0 && lambda <= 1.0 && "Lambda out of bounds");
        return left->y + lambda*(right->y - left->y);
    }
}

bool
TrapezoidMapTriFinder::Edge::has_point(const Point* point) const
{
    // 确保点不为空，如果为空则断言失败
    assert(point != 0 && "Null point");
    // 检查边是否包含给定点
    return (left == point || right == point);
}

bool
TrapezoidMapTriFinder::Edge::operator==(const Edge& other) const
{
    // 检查两条边是否是同一个对象
    return this == &other;
}

void
TrapezoidMapTriFinder::Edge::print_debug() const
{
    // 打印调试信息，显示边的详细信息和相关三角形的索引
    std::cout << "Edge " << *this << " tri_below=" << triangle_below
        << " tri_above=" << triangle_above << std::endl;
}

TrapezoidMapTriFinder::Node::Node(const Point* point, Node* left, Node* right)
    : _type(Type_XNode)
{
    // 确保点不为空，如果为空则断言失败
    assert(point != 0 && "Invalid point");
    // 确保左节点不为空，如果为空则断言失败
    assert(left != 0 && "Invalid left node");
    // 确保右节点不为空，如果为空则断言失败
    assert(right != 0 && "Invalid right node");
    // 初始化X节点的属性
    _union.xnode.point = point;
    _union.xnode.left = left;
    _union.xnode.right = right;
    // 将当前节点设置为左右子节点的父节点
    left->add_parent(this);
    right->add_parent(this);
}

TrapezoidMapTriFinder::Node::Node(const Edge* edge, Node* below, Node* above)
    : _type(Type_YNode)
{
    // 确保边不为空，如果为空则断言失败
    assert(edge != 0 && "Invalid edge");
    // 确保下方节点不为空，如果为空则断言失败
    assert(below != 0 && "Invalid below node");
    // 确保上方节点不为空，如果为空则断言失败
    assert(above != 0 && "Invalid above node");
    // 将 edge 赋给 _union 结构体中的 ynode 的 edge 成员变量
    _union.ynode.edge = edge;
    // 将 below 赋给 _union 结构体中的 ynode 的 below 成员变量
    _union.ynode.below = below;
    // 将 above 赋给 _union 结构体中的 ynode 的 above 成员变量
    _union.ynode.above = above;
    // 将当前对象 (this) 添加为 below 节点的父节点
    below->add_parent(this);
    // 将当前对象 (this) 添加为 above 节点的父节点
    above->add_parent(this);
}

// TrapezoidMapTriFinder::Node 类的构造函数，接受一个指向 Trapezoid 类对象的指针作为参数
TrapezoidMapTriFinder::Node::Node(Trapezoid* trapezoid)
    : _type(Type_TrapezoidNode)  // 初始化节点类型为 Type_TrapezoidNode
{
    assert(trapezoid != 0 && "Null Trapezoid");  // 断言 trapezoid 不为 nullptr
    _union.trapezoid = trapezoid;  // 将传入的 trapezoid 对象赋值给联合体的 trapezoid 成员
    trapezoid->trapezoid_node = this;  // 将当前节点指针赋给 trapezoid 对象的 trapezoid_node 成员
}

// TrapezoidMapTriFinder::Node 类的析构函数
TrapezoidMapTriFinder::Node::~Node()
{
    switch (_type) {
        case Type_XNode:
            // 如果节点类型为 Type_XNode
            // 尝试从左右子节点中移除当前节点作为父节点，成功则删除左右子节点
            if (_union.xnode.left->remove_parent(this))
                delete _union.xnode.left;
            if (_union.xnode.right->remove_parent(this))
                delete _union.xnode.right;
            break;
        case Type_YNode:
            // 如果节点类型为 Type_YNode
            // 尝试从下方和上方子节点中移除当前节点作为父节点，成功则删除下方和上方子节点
            if (_union.ynode.below->remove_parent(this))
                delete _union.ynode.below;
            if (_union.ynode.above->remove_parent(this))
                delete _union.ynode.above;
            break;
        case Type_TrapezoidNode:
            // 如果节点类型为 Type_TrapezoidNode
            // 删除联合体中的 trapezoid 对象
            delete _union.trapezoid;
            break;
    }
}

// 向节点添加一个父节点
void
TrapezoidMapTriFinder::Node::add_parent(Node* parent)
{
    assert(parent != 0 && "Null parent");  // 断言 parent 不为 nullptr
    assert(parent != this && "Cannot be parent of self");  // 断言 parent 不是自身
    assert(!has_parent(parent) && "Parent already in collection");  // 断言 parent 不已经存在于父节点集合中
    _parents.push_back(parent);  // 将 parent 添加到父节点集合中
}

// 在调试模式下验证节点的有效性
void
TrapezoidMapTriFinder::Node::assert_valid(bool tree_complete) const
{
#ifndef NDEBUG
    // 检查父节点
    for (Parents::const_iterator it = _parents.begin();
         it != _parents.end(); ++it) {
        Node* parent = *it;
        assert(parent != this && "Cannot be parent of self");  // 断言 parent 不是自身
        assert(parent->has_child(this) && "Parent missing child");  // 断言 parent 包含当前节点作为子节点
    }

    // 检查子节点，并递归验证
    switch (_type) {
        case Type_XNode:
            assert(_union.xnode.left != 0 && "Null left child");  // 断言左子节点不为 nullptr
            assert(_union.xnode.left->has_parent(this) && "Incorrect parent");  // 断言左子节点的父节点为当前节点
            assert(_union.xnode.right != 0 && "Null right child");  // 断言右子节点不为 nullptr
            assert(_union.xnode.right->has_parent(this) && "Incorrect parent");  // 断言右子节点的父节点为当前节点
            _union.xnode.left->assert_valid(tree_complete);  // 递归验证左子节点
            _union.xnode.right->assert_valid(tree_complete);  // 递归验证右子节点
            break;
        case Type_YNode:
            assert(_union.ynode.below != 0 && "Null below child");  // 断言下方子节点不为 nullptr
            assert(_union.ynode.below->has_parent(this) && "Incorrect parent");  // 断言下方子节点的父节点为当前节点
            assert(_union.ynode.above != 0 && "Null above child");  // 断言上方子节点不为 nullptr
            assert(_union.ynode.above->has_parent(this) && "Incorrect parent");  // 断言上方子节点的父节点为当前节点
            _union.ynode.below->assert_valid(tree_complete);  // 递归验证下方子节点
            _union.ynode.above->assert_valid(tree_complete);  // 递归验证上方子节点
            break;
        case Type_TrapezoidNode:
            assert(_union.trapezoid != 0 && "Null trapezoid");  // 断言 trapezoid 不为 nullptr
            assert(_union.trapezoid->trapezoid_node == this &&
                   "Incorrect trapezoid node");  // 断言 trapezoid 对象中的 trapezoid_node 成员为当前节点
            _union.trapezoid->assert_valid(tree_complete);  // 调用 trapezoid 对象的 assert_valid 方法验证其有效性
            break;
    }
#endif
}

// 获取节点统计信息，包括节点数量和最大深度
void
TrapezoidMapTriFinder::Node::get_stats(int depth,
                                       NodeStats& stats) const
{
    stats.node_count++;  // 节点计数加一
    if (depth > stats.max_depth)
        stats.max_depth = depth;  // 更新最大深度
    // 尝试将当前节点插入到唯一节点集合中，并检查是否插入成功（即节点是否是新节点）
    bool new_node = stats.unique_nodes.insert(this).second;
    
    // 如果是新节点，则更新统计信息中的最大父节点数
    if (new_node)
        stats.max_parent_count = std::max(stats.max_parent_count,
                                          static_cast<long>(_parents.size()));

    // 根据节点类型进行不同的统计处理
    switch (_type) {
        case Type_XNode:
            // 如果节点类型为 Type_XNode，则递归获取左右子节点的统计信息
            _union.xnode.left->get_stats(depth+1, stats);
            _union.xnode.right->get_stats(depth+1, stats);
            break;
        case Type_YNode:
            // 如果节点类型为 Type_YNode，则递归获取上下子节点的统计信息
            _union.ynode.below->get_stats(depth+1, stats);
            _union.ynode.above->get_stats(depth+1, stats);
            break;
        default:  // Type_TrapezoidNode:
            // 如果节点类型为 Type_TrapezoidNode（默认类型），则更新梯形节点相关的统计信息
            stats.unique_trapezoid_nodes.insert(this);  // 插入当前节点到唯一梯形节点集合中
            stats.trapezoid_count++;  // 增加梯形节点计数
            stats.sum_trapezoid_depth += depth;  // 累加梯形节点的深度
            break;
    }
// 返回当前节点所关联的三角形索引
int
TrapezoidMapTriFinder::Node::get_tri() const
{
    switch (_type) {
        case Type_XNode:
            // 如果节点类型为 XNode，则返回其关联点的三角形索引
            return _union.xnode.point->tri;
        case Type_YNode:
            // 如果节点类型为 YNode，则根据边的上下关系返回相应的三角形索引
            if (_union.ynode.edge->triangle_above != -1)
                return _union.ynode.edge->triangle_above;
            else
                return _union.ynode.edge->triangle_below;
        default:  // 默认情况下为 Type_TrapezoidNode:
            // 对于梯形节点，确保上下边的三角形索引一致，然后返回下边的上方三角形索引
            assert(_union.trapezoid->below.triangle_above ==
                   _union.trapezoid->above.triangle_below &&
                   "Inconsistent triangle indices from trapezoid edges");
            return _union.trapezoid->below.triangle_above;
    }
}

// 检查当前节点是否具有指定的子节点
bool
TrapezoidMapTriFinder::Node::has_child(const Node* child) const
{
    assert(child != 0 && "Null child node");  // 确保子节点不为空
    switch (_type) {
        case Type_XNode:
            // 如果节点类型为 XNode，则检查左右子节点是否与指定子节点相同
            return (_union.xnode.left == child || _union.xnode.right == child);
        case Type_YNode:
            // 如果节点类型为 YNode，则检查上下子节点是否与指定子节点相同
            return (_union.ynode.below == child ||
                    _union.ynode.above == child);
        default:  // 默认情况下为 Type_TrapezoidNode:
            // 对于梯形节点，不具有子节点，返回 false
            return false;
    }
}

// 检查当前节点是否没有父节点
bool
TrapezoidMapTriFinder::Node::has_no_parents() const
{
    return _parents.empty();  // 返回父节点集合是否为空
}

// 检查当前节点是否具有指定的父节点
bool
TrapezoidMapTriFinder::Node::has_parent(const Node* parent) const
{
    // 使用 STL 中的 find 函数查找指定的父节点是否在父节点集合中
    return (std::find(_parents.begin(), _parents.end(), parent) !=
            _parents.end());
}

// 打印当前节点的信息，包括节点类型及其关联内容
void
TrapezoidMapTriFinder::Node::print(int depth /* = 0 */) const
{
    // 打印缩进，以区分节点的层次
    for (int i = 0; i < depth; ++i) std::cout << "  ";
    switch (_type) {
        case Type_XNode:
            // 如果节点类型为 XNode，则打印其关联点的信息，并递归打印左右子节点
            std::cout << "XNode " << *_union.xnode.point << std::endl;
            _union.xnode.left->print(depth + 1);
            _union.xnode.right->print(depth + 1);
            break;
        case Type_YNode:
            // 如果节点类型为 YNode，则打印其关联边的信息，并递归打印上下子节点
            std::cout << "YNode " << *_union.ynode.edge << std::endl;
            _union.ynode.below->print(depth + 1);
            _union.ynode.above->print(depth + 1);
            break;
        case Type_TrapezoidNode:
            // 如果节点类型为梯形节点，则打印其四个角点的信息
            std::cout << "Trapezoid ll="
                << _union.trapezoid->get_lower_left_point()  << " lr="
                << _union.trapezoid->get_lower_right_point() << " ul="
                << _union.trapezoid->get_upper_left_point()  << " ur="
                << _union.trapezoid->get_upper_right_point() << std::endl;
            break;
    }
}

// 移除当前节点的指定父节点，并返回是否当前节点已经没有父节点
bool
TrapezoidMapTriFinder::Node::remove_parent(Node* parent)
{
    assert(parent != 0 && "Null parent");  // 确保父节点不为空
    assert(parent != this && "Cannot be parent of self");  // 确保不能移除自身作为父节点
    // 使用 STL 中的 find 函数查找指定的父节点是否在父节点集合中
    Parents::iterator it = std::find(_parents.begin(), _parents.end(), parent);
    assert(it != _parents.end() && "Parent not in collection");  // 确保找到了要移除的父节点
    _parents.erase(it);  // 移除父节点
    return _parents.empty();  // 返回当前节点是否没有父节点了
}

// 替换当前节点的旧子节点为新子节点
void
TrapezoidMapTriFinder::Node::replace_child(Node* old_child, Node* new_child)
{
    switch (_type) {
        // 根据节点类型进行不同操作
        case Type_XNode:
            // 确保 old_child 是当前节点的子节点
            assert((_union.xnode.left == old_child ||
                    _union.xnode.right == old_child) && "Not a child Node");
            // 确保 new_child 不为空
            assert(new_child != 0 && "Null child node");
            // 更新左子节点或右子节点为 new_child
            if (_union.xnode.left == old_child)
                _union.xnode.left = new_child;
            else
                _union.xnode.right = new_child;
            break;
        case Type_YNode:
            // 确保 old_child 是当前节点的子节点
            assert((_union.ynode.below == old_child ||
                    _union.ynode.above == old_child) && "Not a child node");
            // 确保 new_child 不为空
            assert(new_child != 0 && "Null child node");
            // 更新下方子节点或上方子节点为 new_child
            if (_union.ynode.below == old_child)
                _union.ynode.below = new_child;
            else
                _union.ynode.above = new_child;
            break;
        case Type_TrapezoidNode:
            // 对于梯形节点类型，不支持该操作，输出错误信息
            assert(0 && "Invalid type for this operation");
            break;
    }
    // 将旧子节点从当前节点中移除
    old_child->remove_parent(this);
    // 将新子节点添加为当前节点的父节点
    new_child->add_parent(this);
// 用新节点替换当前节点
void
TrapezoidMapTriFinder::Node::replace_with(Node* new_node)
{
    // 断言新节点不为空
    assert(new_node != 0 && "Null replacement node");

    // 循环直到_parents集合为空
    while (!_parents.empty())
        // 调用父节点的replace_child方法，用新节点替换当前节点，并从_parents集合中移除每个父节点
        _parents.front()->replace_child(this, new_node);
}

// 根据坐标搜索节点
const TrapezoidMapTriFinder::Node*
TrapezoidMapTriFinder::Node::search(const XY& xy)
{
    // 根据节点类型进行不同处理
    switch (_type) {
        case Type_XNode:
            // 如果xy坐标与_union.xnode.point相等，则返回当前节点
            if (xy == *_union.xnode.point)
                return this;
            // 如果xy坐标在_union.xnode.point右侧，则递归搜索右子节点
            else if (xy.is_right_of(*_union.xnode.point))
                return _union.xnode.right->search(xy);
            // 否则递归搜索左子节点
            else
                return _union.xnode.left->search(xy);
        case Type_YNode: {
            // 计算xy坐标相对于_union.ynode.edge的方向
            int orient = _union.ynode.edge->get_point_orientation(xy);
            // 如果方向为0，则返回当前节点
            if (orient == 0)
                return this;
            // 如果方向小于0，则递归搜索上方节点
            else if (orient < 0)
                return _union.ynode.above->search(xy);
            // 否则递归搜索下方节点
            else
                return _union.ynode.below->search(xy);
        }
        default:  // Type_TrapezoidNode:
            // 默认情况下返回当前节点
            return this;
    }
}

// 根据边搜索梯形
TrapezoidMapTriFinder::Trapezoid*
TrapezoidMapTriFinder::Node::search(const Edge& edge)
{
    // 该方法未实现，应该根据边搜索相关梯形并返回
}
    # 如果存在上方的梯形（upper_right 不为 0）
    if (upper_right != 0) {
        # 断言上方梯形的关系正确：上方梯形的上方梯形应为当前梯形，上方梯形的左上角应为当前梯形，用于验证上方梯形的正确性
        assert(upper_right->above == above &&
               upper_right->upper_left == this &&
               "Incorrect upper_right trapezoid");
        # 断言上方梯形的右上角点与当前梯形的右上角点相同，用于验证右上角点的正确性
        assert(get_upper_right_point() == upper_right->get_upper_left_point() &&
               "Incorrect upper right point");
    }

    # 断言梯形节点不为 null
    assert(trapezoid_node != 0 && "Null trapezoid_node");

    # 如果树已完整构建
    if (tree_complete) {
        # 断言下方三角形的上方梯形应为上方三角形的下方梯形，用于验证从梯形边获取的三角形索引的一致性
        assert(below.triangle_above == above.triangle_below &&
               "Inconsistent triangle indices from trapezoid edges");
    }
// 结束条件，用于条件编译
#endif
}

// 获取梯形的左下角点坐标
XY
TrapezoidMapTriFinder::Trapezoid::get_lower_left_point() const
{
    // 获取左边界点的 x 坐标
    double x = left->x;
    // 返回 XY 对象，其 y 坐标为 below 对象在 x 处的 y 值
    return XY(x, below.get_y_at_x(x));
}

// 获取梯形的右下角点坐标
XY
TrapezoidMapTriFinder::Trapezoid::get_lower_right_point() const
{
    // 获取右边界点的 x 坐标
    double x = right->x;
    // 返回 XY 对象，其 y 坐标为 below 对象在 x 处的 y 值
    return XY(x, below.get_y_at_x(x));
}

// 获取梯形的左上角点坐标
XY
TrapezoidMapTriFinder::Trapezoid::get_upper_left_point() const
{
    // 获取左边界点的 x 坐标
    double x = left->x;
    // 返回 XY 对象，其 y 坐标为 above 对象在 x 处的 y 值
    return XY(x, above.get_y_at_x(x));
}

// 获取梯形的右上角点坐标
XY
TrapezoidMapTriFinder::Trapezoid::get_upper_right_point() const
{
    // 获取右边界点的 x 坐标
    double x = right->x;
    // 返回 XY 对象，其 y 坐标为 above 对象在 x 处的 y 值
    return XY(x, above.get_y_at_x(x));
}

// 打印梯形对象的调试信息
void
TrapezoidMapTriFinder::Trapezoid::print_debug() const
{
    // 输出梯形对象的各种属性信息到标准输出流
    std::cout << "Trapezoid " << this
        << " left=" << *left
        << " right=" << *right
        << " below=" << below
        << " above=" << above
        << " ll=" << lower_left
        << " lr=" << lower_right
        << " ul=" << upper_left
        << " ur=" << upper_right
        << " node=" << trapezoid_node
        << " llp=" << get_lower_left_point()
        << " lrp=" << get_lower_right_point()
        << " ulp=" << get_upper_left_point()
        << " urp=" << get_upper_right_point() << std::endl;
}

// 设置梯形对象的左下角指针
void
TrapezoidMapTriFinder::Trapezoid::set_lower_left(Trapezoid* lower_left_)
{
    // 设置左下角指针
    lower_left = lower_left_;
    // 如果左下角指针非空，则将其左下角指针指向当前梯形对象
    if (lower_left != 0)
        lower_left->lower_right = this;
}

// 设置梯形对象的右下角指针
void
TrapezoidMapTriFinder::Trapezoid::set_lower_right(Trapezoid* lower_right_)
{
    // 设置右下角指针
    lower_right = lower_right_;
    // 如果右下角指针非空，则将其右下角指针指向当前梯形对象
    if (lower_right != 0)
        lower_right->lower_left = this;
}

// 设置梯形对象的左上角指针
void
TrapezoidMapTriFinder::Trapezoid::set_upper_left(Trapezoid* upper_left_)
{
    // 设置左上角指针
    upper_left = upper_left_;
    // 如果左上角指针非空，则将其左上角指针指向当前梯形对象
    if (upper_left != 0)
        upper_left->upper_right = this;
}

// 设置梯形对象的右上角指针
void
TrapezoidMapTriFinder::Trapezoid::set_upper_right(Trapezoid* upper_right_)
{
    // 设置右上角指针
    upper_right = upper_right_;
    // 如果右上角指针非空，则将其右上角指针指向当前梯形对象
    if (upper_right != 0)
        upper_right->upper_left = this;
}
```