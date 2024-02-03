# `.\PaddleOCR\deploy\cpp_infer\include\clipper.h`

```
/*******************************************************************************
*                                                                              *
* Author    :  Angus Johnson                                                   *
* Version   :  6.4.2                                                           *
* Date      :  27 February 2017                                                *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
*                                                                              *
* License:                                                                     *
* Use, modification & distribution is subject to Boost Software License Ver 1. *
* http://www.boost.org/LICENSE_1_0.txt                                         *
*                                                                              *
* Attributions:                                                                *
* The code in this library is an extension of Bala Vatti's clipping algorithm: *
* "A generic solution to polygon clipping"                                     *
* Communications of the ACM, Vol 35, Issue 7 (July 1992) pp 56-63.             *
* http://portal.acm.org/citation.cfm?id=129906                                 *
*                                                                              *
* Computer graphics and geometric modeling: implementation and algorithms      *
* By Max K. Agoston                                                            *
* Springer; 1 edition (January 4, 2005)                                        *
* http://books.google.com/books?q=vatti+clipping+agoston                       *
*                                                                              *
* See also:                                                                    *
*******************************************************************************/
// 定义了一些关于 Clipper 库的信息和版本号
* "Polygon Offsetting by Computing Winding Numbers"                            *
* Paper no. DETC2005-85513 pp. 565-575                                         *
* ASME 2005 International Design Engineering Technical Conferences             *
* and Computers and Information in Engineering Conference (IDETC/CIE2005)      *
* September 24-28, 2005 , Long Beach, California, USA                          *
* http://www.me.berkeley.edu/~mcmains/pubs/DAC05OffsetPolygon.pdf              *
*                                                                              *
*******************************************************************************/

// 防止头文件重复包含
#pragma once

#ifndef clipper_hpp
#define clipper_hpp

// 定义 Clipper 库的版本号
#define CLIPPER_VERSION "6.4.2"

// 当启用时，使用 32 位整数代替 64 位整数，提高性能，但坐标值范围限制在+/- 46340
//#define use_int32

// 当启用时，为 IntPoint 添加 Z 成员，对性能有轻微影响
//#define use_xyz

// 当启用时，启用线段裁剪，对性能有非常轻微影响
#define use_lines

// 当启用时，启用对废弃函数的临时支持
//#define use_deprecated

#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <ostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>

// Clipper 库的命名空间
namespace ClipperLib {

// 定义裁剪类型枚举
enum ClipType { ctIntersection, ctUnion, ctDifference, ctXor };
// 定义多边形类型枚举
enum PolyType { ptSubject, ptClip };
// 定义多边形填充类型枚举
enum PolyFillType { pftEvenOdd, pftNonZero, pftPositive, pftNegative };

#ifdef use_int32
// 当启用 use_int32 时，使用 int 类型作为 cInt
typedef int cInt;
// 定义 cInt 的范围
static cInt const loRange = 0x7FFF;
static cInt const hiRange = 0x7FFF;
#else
// 否则，使用 signed long long 类型作为 cInt
typedef signed long long cInt;
// 定义常量 loRange，表示低范围
static cInt const loRange = 0x3FFFFFFF;
// 定义常量 hiRange，表示高范围
static cInt const hiRange = 0x3FFFFFFFFFFFFFFFLL;
// 定义 signed long long 类型 long64，用于 Int128 类
typedef signed long long long64;
// 定义 unsigned long long 类型 ulong64
typedef unsigned long long ulong64;

#endif

// 定义结构体 IntPoint，包含 X 和 Y 两个成员变量
struct IntPoint {
  cInt X; // X 坐标
  cInt Y; // Y 坐标
#ifdef use_xyz
  cInt Z; // Z 坐标
  // 构造函数，根据传入的 x、y、z 值初始化 IntPoint 对象
  IntPoint(cInt x = 0, cInt y = 0, cInt z = 0) : X(x), Y(y), Z(z){};
#else
  // 构造函数，根据传入的 x、y 值初始化 IntPoint 对象
  IntPoint(cInt x = 0, cInt y = 0) : X(x), Y(y){};
#endif

  // 重载 == 运算符，判断两个 IntPoint 对象是否相等
  friend inline bool operator==(const IntPoint &a, const IntPoint &b) {
    return a.X == b.X && a.Y == b.Y;
  }
  // 重载 != 运算符，判断两个 IntPoint 对象是否不相等
  friend inline bool operator!=(const IntPoint &a, const IntPoint &b) {
    return a.X != b.X || a.Y != b.Y;
  }
};
//------------------------------------------------------------------------------

// 定义类型别名 Path，表示一条路径
typedef std::vector<IntPoint> Path;
// 定义类型别名 Paths，表示多条路径
typedef std::vector<Path> Paths;

// 重载 << 运算符，将 IntPoint 对象添加到 Path 中
inline Path &operator<<(Path &poly, const IntPoint &p) {
  poly.push_back(p);
  return poly;
}
// 重载 << 运算符，将 Path 对象添加到 Paths 中
inline Paths &operator<<(Paths &polys, const Path &p) {
  polys.push_back(p);
  return polys;
}

// 重载 << 运算符，用于输出 IntPoint 对象到流
std::ostream &operator<<(std::ostream &s, const IntPoint &p);
// 重载 << 运算符，用于输出 Path 对象到流
std::ostream &operator<<(std::ostream &s, const Path &p);
// 重载 << 运算符，用于输出 Paths 对象到流
std::ostream &operator<<(std::ostream &s, const Paths &p);

// 定义结构体 DoublePoint，包含 X 和 Y 两个 double 类型成员变量
struct DoublePoint {
  double X; // X 坐标
  double Y; // Y 坐标
  // 构造函数，根据传入的 x、y 值初始化 DoublePoint 对象
  DoublePoint(double x = 0, double y = 0) : X(x), Y(y) {}
  // 构造函数，根据传入的 IntPoint 对象初始化 DoublePoint 对象
  DoublePoint(IntPoint ip) : X((double)ip.X), Y((double)ip.Y) {}
};
//------------------------------------------------------------------------------

#ifdef use_xyz
// 定义回调函数指针类型 ZFillCallback，用于填充 Z 坐标
typedef void (*ZFillCallback)(IntPoint &e1bot, IntPoint &e1top, IntPoint &e2bot,
                              IntPoint &e2top, IntPoint &pt);
#endif

// 枚举类型 InitOptions，表示初始化选项
enum InitOptions {
  ioReverseSolution = 1, // 反转解决方案
  ioStrictlySimple = 2, // 严格简单
  ioPreserveCollinear = 4 // 保留共线
};
// 枚举类型 JoinType，表示连接类型
enum JoinType { jtSquare, jtRound, jtMiter };
// 枚举类型 EndType，表示结束类型
enum EndType {
  etClosedPolygon, // 闭合多边形
  etClosedLine, // 闭合线段
  etOpenButt, // 开放端点为平头
  etOpenSquare, // 开放端点为方头
  etOpenRound // 开放端点为圆头
};

// 定义类 PolyNode
class PolyNode;
// 定义类型别名 PolyNodes，表示多个 PolyNode 指针
typedef std::vector<PolyNode *> PolyNodes;

// 定义类 PolyNode
class PolyNode {
// 定义 PolyNode 类，表示多边形节点
public:
  // 默认构造函数
  PolyNode();
  // 虚析构函数
  virtual ~PolyNode(){};
  // 多边形轮廓
  Path Contour;
  // 子节点列表
  PolyNodes Childs;
  // 父节点指针
  PolyNode *Parent;
  // 获取下一个节点
  PolyNode *GetNext() const;
  // 判断是否为孔
  bool IsHole() const;
  // 判断是否为开放多边形
  bool IsOpen() const;
  // 获取子节点数量
  int ChildCount() const;

private:
  // 节点在父节点子节点列表中的索引
  unsigned Index;
  // 是否为开放多边形
  bool m_IsOpen;
  // 连接类型
  JoinType m_jointype;
  // 结束类型
  EndType m_endtype;
  // 获取下一个同级节点
  PolyNode *GetNextSiblingUp() const;
  // 添加子节点
  void AddChild(PolyNode &child);
  // 声明 Clipper 类和 ClipperOffset 类为友元类，以便访问 Index
  friend class Clipper;
  friend class ClipperOffset;
};

// 定义 PolyTree 类，继承自 PolyNode 类
class PolyTree : public PolyNode {
public:
  // 析构函数，清空所有节点
  ~PolyTree() { Clear(); };
  // 获取第一个节点
  PolyNode *GetFirst() const;
  // 清空所有节点
  void Clear();
  // 获取节点总数
  int Total() const;

private:
  // 所有节点列表
  PolyNodes AllNodes;
  // 声明 Clipper 类为友元类，以便访问 AllNodes
  friend class Clipper;
};

// 判断多边形的方向
bool Orientation(const Path &poly);
// 计算多边形的面积
double Area(const Path &poly);
// 判断点是否在多边形内部
int PointInPolygon(const IntPoint &pt, const Path &path);

// 简化单个多边形
void SimplifyPolygon(const Path &in_poly, Paths &out_polys, PolyFillType fillType = pftEvenOdd);
// 简化多个多边形
void SimplifyPolygons(const Paths &in_polys, Paths &out_polys, PolyFillType fillType = pftEvenOdd);
// 简化多个多边形
void SimplifyPolygons(Paths &polys, PolyFillType fillType = pftEvenOdd);

// 清理单个多边形
void CleanPolygon(const Path &in_poly, Path &out_poly, double distance = 1.415);
// 清理单个多边形
void CleanPolygon(Path &poly, double distance = 1.415);
// 清理多个多边形
void CleanPolygons(const Paths &in_polys, Paths &out_polys, double distance = 1.415);
// 清理多个多边形
void CleanPolygons(Paths &polys, double distance = 1.415);

// 计算 Minkowski 和
void MinkowskiSum(const Path &pattern, const Path &path, Paths &solution, bool pathIsClosed);
// 计算 Minkowski 和
void MinkowskiSum(const Path &pattern, const Paths &paths, Paths &solution, bool pathIsClosed);
// 计算 Minkowski 差
void MinkowskiDiff(const Path &poly1, const Path &poly2, Paths &solution);

// 将 PolyTree 转换为 Paths
void PolyTreeToPaths(const PolyTree &polytree, Paths &paths);
// 从 PolyTree 中获取封闭路径
void ClosedPathsFromPolyTree(const PolyTree &polytree, Paths &paths);
// 从 PolyTree 中打开路径并存储到 Paths 中
void OpenPathsFromPolyTree(PolyTree &polytree, Paths &paths);

// 反转给定路径
void ReversePath(Path &p);
// 反转给定路径集合
void ReversePaths(Paths &p);

// 表示整数矩形的结构体
struct IntRect {
  cInt left;
  cInt top;
  cInt right;
  cInt bottom;
};

// 内部使用的枚举类型
enum EdgeSide { esLeft = 1, esRight = 2 };

// 内部使用的结构体的前向声明
struct TEdge;
struct IntersectNode;
struct LocalMinimum;
struct OutPt;
struct OutRec;
struct Join;

// 定义存储 OutRec 指针的列表类型
typedef std::vector<OutRec *> PolyOutList;
// 定义存储 TEdge 指针的列表类型
typedef std::vector<TEdge *> EdgeList;
// 定义存储 Join 指针的列表类型
typedef std::vector<Join *> JoinList;
// 定义存储 IntersectNode 指针的列表类型
typedef std::vector<IntersectNode *> IntersectList;

//------------------------------------------------------------------------------

// ClipperBase 是 Clipper 类的祖先类。不应直接实例化。该类简单地将多边形坐标集转换为存储在 LocalMinima 列表中的边对象。
class ClipperBase {
public:
  // 构造函数
  ClipperBase();
  // 虚析构函数
  virtual ~ClipperBase();
  // 添加路径到 ClipperBase 对象中
  virtual bool AddPath(const Path &pg, PolyType PolyTyp, bool Closed);
  // 添加多个路径到 ClipperBase 对象中
  bool AddPaths(const Paths &ppg, PolyType PolyTyp, bool Closed);
  // 清空 ClipperBase 对象
  virtual void Clear();
  // 获取边界矩形
  IntRect GetBounds();
  // 获取是否保留共线点的标志
  bool PreserveCollinear() { return m_PreserveCollinear; };
  // 设置是否保留共线点的标志
  void PreserveCollinear(bool value) { m_PreserveCollinear = value; };
# 定义一个类，继承自 ClipperBase 类
class Clipper : public virtual ClipperBase {
public:
  # 构造函数，可以设置初始选项
  Clipper(int initOptions = 0);
  # 执行裁剪操作，返回裁剪结果路径
  bool Execute(ClipType clipType, Paths &solution, PolyFillType fillType = pftEvenOdd);
  # 执行裁剪操作，返回裁剪结果路径，可以指定主题和裁剪填充类型
  bool Execute(ClipType clipType, Paths &solution, PolyFillType subjFillType, PolyFillType clipFillType);
  # 执行裁剪操作，返回裁剪结果多边形树
  bool Execute(ClipType clipType, PolyTree &polytree, PolyFillType fillType = pftEvenOdd);
  # 执行裁剪操作，返回裁剪结果多边形树，可以指定主题和裁剪填充类型
  bool Execute(ClipType clipType, PolyTree &polytree, PolyFillType subjFillType, PolyFillType clipFillType);
  # 返回是否反转解决方案的标志
  bool ReverseSolution() { return m_ReverseOutput; };
  # 设置是否反转解决方案的标志
  void ReverseSolution(bool value) { m_ReverseOutput = value; };
  # 返回是否严格简单的标志
  bool StrictlySimple() { return m_StrictSimple; };
  # 设置是否严格简单的标志
  void StrictlySimple(bool value) { m_StrictSimple = value; };
  # 设置交点的 Z 值填充回调函数（如果未定义，则 Z 为 0）
#ifdef use_xyz
  void ZFillFunction(ZFillCallback zFillFunc);
#endif
protected:
  # 内部执行函数
  virtual bool ExecuteInternal();
// Clipper 类的私有成员变量
private:
  JoinList m_Joins; // 连接列表
  JoinList m_GhostJoins; // 鬼连接列表
  IntersectList m_IntersectList; // 相交列表
  ClipType m_ClipType; // 剪切类型
  typedef std::list<cInt> MaximaList; // 最大值列表
  MaximaList m_Maxima; // 最大值列表
  TEdge *m_SortedEdges; // 排序后的边缘
  bool m_ExecuteLocked; // 执行锁定状态
  PolyFillType m_ClipFillType; // 剪切填充类型
  PolyFillType m_SubjFillType; // 主题填充类型
  bool m_ReverseOutput; // 反向输出
  bool m_UsingPolyTree; // 使用 PolyTree
  bool m_StrictSimple; // 严格简单
#ifdef use_xyz
  ZFillCallback m_ZFill; // 自定义回调
#ifdef use_xyz
  void SetZ(IntPoint &pt, TEdge &e1, TEdge &e2); // 设置 Z 值
#endif
};

//------------------------------------------------------------------------------

// ClipperOffset 类
class ClipperOffset {
public:
  ClipperOffset(double miterLimit = 2.0, double roundPrecision = 0.25); // 构造函数
  ~ClipperOffset(); // 析构函数
  void AddPath(const Path &path, JoinType joinType, EndType endType); // 添加路径
  void AddPaths(const Paths &paths, JoinType joinType, EndType endType); // 添加多个路径
  void Execute(Paths &solution, double delta); // 执行操作
  void Execute(PolyTree &solution, double delta); // 执行操作
  void Clear(); // 清空
  double MiterLimit; // 斜接限制
  double ArcTolerance; // 弧度公差

private:
  Paths m_destPolys; // 目标多边形
  Path m_srcPoly; // 源多边形
  Path m_destPoly; // 目标多边形
  std::vector<DoublePoint> m_normals; // 法线向量
  double m_delta, m_sinA, m_sin, m_cos; // 偏移值、正弦值、余弦值
  double m_miterLim, m_StepsPerRad; // 斜接限制、每弧度步数
  IntPoint m_lowest; // 最低点
  PolyNode m_polyNodes; // 多边形节点

  void FixOrientations(); // 修正方向
  void DoOffset(double delta); // 执行偏移
  void OffsetPoint(int j, int &k, JoinType jointype); // 偏移点
  void DoSquare(int j, int k); // 执行方形
  void DoMiter(int j, int k, double r); // 执行斜接
  void DoRound(int j, int k); // 执行圆角
};

//------------------------------------------------------------------------------

// clipperException 类
class clipperException : public std::exception {
public:
  clipperException(const char *description) : m_descr(description) {} // 构造函数
  virtual ~clipperException() throw() {} // 析构函数
  virtual const char *what() const throw() { return m_descr.c_str(); } // 返回描述信息

private:
  std::string m_descr; // 描述信息
};

//------------------------------------------------------------------------------

} // ClipperLib 命名空间
#endif // clipper_hpp
```