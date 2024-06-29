# `D:\src\scipysrc\matplotlib\src\tri\_tri_wrapper.cpp`

```
#include "_tri.h"  // 引入自定义的 _tri.h 头文件

using namespace pybind11::literals;  // 使用 pybind11 命名空间下的 literals

// 定义 Python 模块 _tri
PYBIND11_MODULE(_tri, m) {
    // 定义 Triangulation 类
    py::class_<Triangulation>(m, "Triangulation", py::is_final())
        // 构造函数定义，接受多个参数描述三角剖分的各种数组
        .def(py::init<const Triangulation::CoordinateArray&,
                      const Triangulation::CoordinateArray&,
                      const Triangulation::TriangleArray&,
                      const Triangulation::MaskArray&,
                      const Triangulation::EdgeArray&,
                      const Triangulation::NeighborArray&,
                      bool>(),
            "x"_a,  // x 参数说明
            "y"_a,  // y 参数说明
            "triangles"_a,  // triangles 参数说明
            "mask"_a,  // mask 参数说明
            "edges"_a,  // edges 参数说明
            "neighbors"_a,  // neighbors 参数说明
            "correct_triangle_orientations"_a,  // correct_triangle_orientations 参数说明
            "Create a new C++ Triangulation object.\n"  // 构造函数说明
            "This should not be called directly, use the python class\n"
            "matplotlib.tri.Triangulation instead.\n")
        
        // 计算平面方程系数的方法
        .def("calculate_plane_coefficients", &Triangulation::calculate_plane_coefficients,
            "Calculate plane equation coefficients for all unmasked triangles.")
        
        // 获取 edges 数组的方法
        .def("get_edges", &Triangulation::get_edges,
            "Return edges array.")
        
        // 获取 neighbors 数组的方法
        .def("get_neighbors", &Triangulation::get_neighbors,
            "Return neighbors array.")
        
        // 设置或清除 mask 数组的方法
        .def("set_mask", &Triangulation::set_mask,
            "Set or clear the mask array.");

    // 定义 TriContourGenerator 类
    py::class_<TriContourGenerator>(m, "TriContourGenerator", py::is_final())
        // 构造函数定义，接受 Triangulation 对象和 z 数组作为参数
        .def(py::init<Triangulation&,
                      const TriContourGenerator::CoordinateArray&>(),
            "triangulation"_a,  // triangulation 参数说明
            "z"_a,  // z 参数说明
            "Create a new C++ TriContourGenerator object.\n"  // 构造函数说明
            "This should not be called directly, use the functions\n"
            "matplotlib.axes.tricontour and tricontourf instead.\n")
        
        // 创建非填充轮廓的方法
        .def("create_contour", &TriContourGenerator::create_contour,
            "Create and return a non-filled contour.")
        
        // 创建填充轮廓的方法
        .def("create_filled_contour", &TriContourGenerator::create_filled_contour,
            "Create and return a filled contour.");

    // 定义 TrapezoidMapTriFinder 类
    py::class_<TrapezoidMapTriFinder>(m, "TrapezoidMapTriFinder", py::is_final())
        // 构造函数定义，接受 Triangulation 对象作为参数
        .def(py::init<Triangulation&>(),
            "triangulation"_a,  // triangulation 参数说明
            "Create a new C++ TrapezoidMapTriFinder object.\n"  // 构造函数说明
            "This should not be called directly, use the python class\n"
            "matplotlib.tri.TrapezoidMapTriFinder instead.\n")
        
        // 查找包含给定点坐标 (x, y) 的多个三角形索引的方法
        .def("find_many", &TrapezoidMapTriFinder::find_many,
            "Find indices of triangles containing the point coordinates (x, y).")
        
        // 返回梯形地图使用的树的统计信息的方法
        .def("get_tree_stats", &TrapezoidMapTriFinder::get_tree_stats,
            "Return statistics about the tree used by the trapezoid map.")
        
        // 初始化对象，从三角剖分创建梯形地图的方法
        .def("initialize", &TrapezoidMapTriFinder::initialize,
            "Initialize this object, creating the trapezoid map from the triangulation.")
        
        // 将搜索树作为文本打印到标准输出的方法，用于调试目的
        .def("print_tree", &TrapezoidMapTriFinder::print_tree,
            "Print the search tree as text to stdout; useful for debug purposes.");
}
```