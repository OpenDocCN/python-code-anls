# `.\PaddleOCR\deploy\cpp_infer\src\clipper.cpp`

```py
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
/*
* "Polygon Offsetting by Computing Winding Numbers"
* Paper no. DETC2005-85513 pp. 565-575
* ASME 2005 International Design Engineering Technical Conferences
* and Computers and Information in Engineering Conference (IDETC/CIE2005)
* September 24-28, 2005 , Long Beach, California, USA
* http://www.me.berkeley.edu/~mcmains/pubs/DAC05OffsetPolygon.pdf
*/

/*
* This is a translation of the Delphi Clipper library and the naming style
* used has retained a Delphi flavour.
*/
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "include/clipper.h"

namespace ClipperLib {

// Define constants for mathematical calculations
static double const pi = 3.141592653589793238;
static double const two_pi = pi * 2;
static double const def_arc_tolerance = 0.25;

// Define enum for direction
enum Direction { dRightToLeft, dLeftToRight };

// Define constants for edge status
static int const Unassigned = -1; // edge not currently 'owning' a solution
static int const Skip = -2;       // edge that would otherwise close a path

// Define macros for horizontal and tolerance values
#define HORIZONTAL (-1.0E+40)
#define TOLERANCE (1.0e-20)
#define NEAR_ZERO(val) (((val) > -TOLERANCE) && ((val) < TOLERANCE)
// 定义表示边的结构体，包含底部点、当前点、顶部点、水平距离、多边形类型、边的侧面、风向变化、风向计数、相反多边形类型的风向计数、输出索引、下一个边、上一个边、左边的下一个边、活动边表的下一个边、活动边表的上一个边
struct TEdge {
  IntPoint Bot;
  IntPoint Curr; // 每次新的扫描线更新的当前点
  IntPoint Top;
  double Dx;
  PolyType PolyTyp;
  EdgeSide Side; // 侧面只指当前解决方案多边形的一侧
  int WindDelta; // 根据绕组方向为1或-1
  int WindCnt;
  int WindCnt2; // 相反多边形类型的绕组计数
  int OutIdx;
  TEdge *Next;
  TEdge *Prev;
  TEdge *NextInLML;
  TEdge *NextInAEL;
  TEdge *PrevInAEL;
  TEdge *NextInSEL;
  TEdge *PrevInSEL;
};

// 定义表示交点的结构体，包含边1、边2、交点
struct IntersectNode {
  TEdge *Edge1;
  TEdge *Edge2;
  IntPoint Pt;
};

// 定义表示局部最小值的结构体，包含Y坐标、左边界、右边界
struct LocalMinimum {
  cInt Y;
  TEdge *LeftBound;
  TEdge *RightBound;
};

// 定义输出点结构体
struct OutPt;

// OutRec: 包含剪切解决方案中的路径。当边在剪切解决方案中时，活动边表中的边将携带指向OutRec的指针
struct OutRec {
  int Idx;
  bool IsHole;
  bool IsOpen;
  OutRec *FirstLeft; // 参见clipper.pas中的注释
  PolyNode *PolyNd;
  OutPt *Pts;
  OutPt *BottomPt;
};

// 定义输出点结构体，包含索引、点、下一个点、上一个点
struct OutPt {
  int Idx;
  IntPoint Pt;
  OutPt *Next;
  OutPt *Prev;
};

// 定义连接结构体，包含输出点1、输出点2、偏移点
struct Join {
  OutPt *OutPt1;
  OutPt *OutPt2;
  IntPoint OffPt;
};

// 定义局部最小值排序器结构体，用于排序局部最小值
struct LocMinSorter {
  inline bool operator()(const LocalMinimum &locMin1,
                         const LocalMinimum &locMin2) {
    return locMin2.Y < locMin1.Y;
  }
};

// 定义将double类型四舍五入为cInt类型的函数
inline cInt Round(double val) {
  if ((val < 0))
    return static_cast<cInt>(val - 0.5);
  else
    return static_cast<cInt>(val + 0.5);
}

// 定义取绝对值的函数
inline cInt Abs(cInt val) { return val < 0 ? -val : val; }
// 清空 PolyTree 对象，释放所有节点内存
void PolyTree::Clear() {
  // 遍历所有节点，释放内存
  for (PolyNodes::size_type i = 0; i < AllNodes.size(); ++i)
    delete AllNodes[i];
  // 重置 AllNodes 和 Childs 容器大小为 0
  AllNodes.resize(0);
  Childs.resize(0);
}
//------------------------------------------------------------------------------

// 获取 PolyTree 对象的第一个子节点
PolyNode *PolyTree::GetFirst() const {
  // 如果 Childs 容器不为空，返回第一个子节点
  if (!Childs.empty())
    return Childs[0];
  else
    return 0;
}
//------------------------------------------------------------------------------

// 返回 PolyTree 对象中节点的总数
int PolyTree::Total() const {
  // 初始化结果为所有节点的数量
  int result = (int)AllNodes.size();
  // 如果存在负偏移，忽略隐藏的外部多边形
  if (result > 0 && Childs[0] != AllNodes[0])
    result--;
  return result;
}

//------------------------------------------------------------------------------
// PolyNode methods ...
//------------------------------------------------------------------------------

// PolyNode 构造函数，初始化成员变量
PolyNode::PolyNode() : Parent(0), Index(0), m_IsOpen(false) {}
//------------------------------------------------------------------------------

// 返回 PolyNode 对象的子节点数量
int PolyNode::ChildCount() const { return (int)Childs.size(); }
//------------------------------------------------------------------------------

// 添加子节点到 PolyNode 对象
void PolyNode::AddChild(PolyNode &child) {
  // 获取当前子节点数量
  unsigned cnt = (unsigned)Childs.size();
  // 将子节点添加到 Childs 容器中
  Childs.push_back(&child);
  // 设置子节点的父节点和索引
  child.Parent = this;
  child.Index = cnt;
}
//------------------------------------------------------------------------------

// 获取 PolyNode 对象的下一个子节点
PolyNode *PolyNode::GetNext() const {
  // 如果 Childs 容器不为空，返回第一个子节点
  if (!Childs.empty())
    return Childs[0];
  else
    return GetNextSiblingUp();
}
//------------------------------------------------------------------------------

// 获取 PolyNode 对象的下一个同级节点
PolyNode *PolyNode::GetNextSiblingUp() const {
  // 如果没有父节点，返回空指针
  if (!Parent) // protects against PolyTree.GetNextSiblingUp()
    return 0;
  // 如果当前节点是父节点的最后一个子节点，返回父节点的下一个同级节点
  else if (Index == Parent->Childs.size() - 1)
    return Parent->GetNextSiblingUp();
  // 否则返回当前节点的下一个同级节点
  else
    return Parent->Childs[Index + 1];
}
//------------------------------------------------------------------------------

// 判断 PolyNode 对象是否为孔
bool PolyNode::IsHole() const {
  bool result = true;
  PolyNode *node = Parent;
  while (node) {
    # 将 result 取反
    result = !result;
    # 将 node 指向其父节点
    node = node->Parent;
  }
  # 返回最终结果
  return result;
// 结束 PolyNode 类的定义
}
//------------------------------------------------------------------------------

// 返回 m_IsOpen 的值，表示 PolyNode 是否是开放的
bool PolyNode::IsOpen() const { return m_IsOpen; }
//------------------------------------------------------------------------------

#ifndef use_int32

//------------------------------------------------------------------------------
// Int128 类（允许在有符号 64 位整数上进行安全的数学运算）
// 例如：Int128 val1((long64)9223372036854775807); // 即 2^63 -1
//      Int128 val2((long64)9223372036854775807);
//      Int128 val3 = val1 * val2;
//      val3.AsString => "85070591730234615847396907784232501249"（8.5e+37）
//------------------------------------------------------------------------------

// 定义 Int128 类
class Int128 {
public:
  ulong64 lo; // 低 64 位
  long64 hi; // 高 64 位

  // 构造函数，初始化为 0
  Int128(long64 _lo = 0) {
    lo = (ulong64)_lo;
    if (_lo < 0)
      hi = -1;
    else
      hi = 0;
  }

  // 拷贝构造函数
  Int128(const Int128 &val) : lo(val.lo), hi(val.hi) {}

  // 构造函数，接受高位和低位的值
  Int128(const long64 &_hi, const ulong64 &_lo) : lo(_lo), hi(_hi) {}

  // 赋值运算符重载
  Int128 &operator=(const long64 &val) {
    lo = (ulong64)val;
    if (val < 0)
      hi = -1;
    else
      hi = 0;
    return *this;
  }

  // 相等运算符重载
  bool operator==(const Int128 &val) const {
    return (hi == val.hi && lo == val.lo);
  }

  // 不等运算符重载
  bool operator!=(const Int128 &val) const { return !(*this == val); }

  // 大于运算符重载
  bool operator>(const Int128 &val) const {
    if (hi != val.hi)
      return hi > val.hi;
    else
      return lo > val.lo;
  }

  // 小于运算符重载
  bool operator<(const Int128 &val) const {
    if (hi != val.hi)
      return hi < val.hi;
    else
      return lo < val.lo;
  }

  // 大于等于运算符重载
  bool operator>=(const Int128 &val) const { return !(*this < val); }

  // 小于等于运算符重载
  bool operator<=(const Int128 &val) const { return !(*this > val); }

  // 加法赋值运算符重载
  Int128 &operator+=(const Int128 &rhs) {
    hi += rhs.hi;
    lo += rhs.lo;
    if (lo < rhs.lo)
      hi++;
    return *this;
  }

  // 加法运算符重载
  Int128 operator+(const Int128 &rhs) const {
    Int128 result(*this);
    result += rhs;
    return result;
  }

  // 减法赋值运算符重载
  Int128 &operator-=(const Int128 &rhs) {
    *this += -rhs;
  // 返回当前对象的引用
  return *this;
}

// 重载减法运算符，计算两个Int128对象的差
Int128 operator-(const Int128 &rhs) const {
  // 创建一个新的Int128对象，值为当前对象的值
  Int128 result(*this);
  // 用rhs对象的值减去result对象的值
  result -= rhs;
  // 返回计算结果
  return result;
}

// 重载负号运算符，实现一元负运算
Int128 operator-() const // unary negation
{
  // 如果lo为0，则返回一个值为-hi的Int128对象
  if (lo == 0)
    return Int128(-hi, 0);
  // 否则返回一个值为~hi和~lo+1的Int128对象
  else
    return Int128(~hi, ~lo + 1);
}

// 转换为double类型
operator double() const {
  // 定义一个64位移位常量
  const double shift64 = 18446744073709551616.0; // 2^64
  // 如果hi小于0
  if (hi < 0) {
    // 如果lo为0，则返回hi乘以64位移位常量
    if (lo == 0)
      return (double)hi * shift64;
    // 否则返回-(~lo + ~hi * shift64)
    else
      return -(double)(~lo + ~hi * shift64);
  } else
    // 返回lo加上hi乘以64位移位常量
    return (double)(lo + hi * shift64);
}
// 定义一个函数，用于计算两个 long64 类型数的乘积，并返回 Int128 类型结果
Int128 Int128Mul(long64 lhs, long64 rhs) {
  // 判断是否需要对结果取反
  bool negate = (lhs < 0) != (rhs < 0);

  // 如果 lhs 小于 0，则取其相反数
  if (lhs < 0)
    lhs = -lhs;
  // 将 lhs 拆分为高 32 位和低 32 位
  ulong64 int1Hi = ulong64(lhs) >> 32;
  ulong64 int1Lo = ulong64(lhs & 0xFFFFFFFF);

  // 如果 rhs 小于 0，则取其相反数
  if (rhs < 0)
    rhs = -rhs;
  // 将 rhs 拆分为高 32 位和低 32 位
  ulong64 int2Hi = ulong64(rhs) >> 32;
  ulong64 int2Lo = ulong64(rhs & 0xFFFFFFFF);

  // 计算 a、b、c 三个中间结果
  ulong64 a = int1Hi * int2Hi;
  ulong64 b = int1Lo * int2Lo;
  ulong64 c = int1Hi * int2Lo + int1Lo * int2Hi;

  // 计算最终结果 tmp
  Int128 tmp;
  tmp.hi = long64(a + (c >> 32));
  tmp.lo = long64(c << 32);
  tmp.lo += long64(b);
  // 处理溢出情况
  if (tmp.lo < b)
    tmp.hi++;
  // 如果需要取反，则对结果取反
  if (negate)
    tmp = -tmp;
  return tmp;
};
#endif

// 计算多边形的方向，根据多边形的面积是否大于等于 0 来判断
bool Orientation(const Path &poly) { return Area(poly) >= 0; }
//------------------------------------------------------------------------------

// 计算多边形的面积
double Area(const Path &poly) {
  int size = (int)poly.size();
  if (size < 3)
    return 0;

  double a = 0;
  // 遍历多边形的顶点，计算面积
  for (int i = 0, j = size - 1; i < size; ++i) {
    a += ((double)poly[j].X + poly[i].X) * ((double)poly[j].Y - poly[i].Y);
    j = i;
  }
  return -a * 0.5;
}
//------------------------------------------------------------------------------

// 计算 OutPt 类型的多边形的面积
double Area(const OutPt *op) {
  const OutPt *startOp = op;
  if (!op)
    return 0;
  double a = 0;
  // 遍历多边形的顶点，计算面积
  do {
    a += (double)(op->Prev->Pt.X + op->Pt.X) *
         (double)(op->Prev->Pt.Y - op->Pt.Y);
    op = op->Next;
  } while (op != startOp);
  return a * 0.5;
}
//------------------------------------------------------------------------------

// 计算 OutRec 类型的多边形的面积
double Area(const OutRec &outRec) { return Area(outRec.Pts); }
//------------------------------------------------------------------------------
// 检查给定点是否是多边形的顶点
bool PointIsVertex(const IntPoint &Pt, OutPt *pp) {
  // 初始化指针 pp2 指向给定的 OutPt 指针 pp
  OutPt *pp2 = pp;
  // 循环遍历 OutPt 链表，直到回到起始点 pp
  do {
    // 如果当前 OutPt 的点坐标等于给定点 Pt，则返回 true
    if (pp2->Pt == Pt)
      return true;
    // 移动 pp2 指针到下一个 OutPt
    pp2 = pp2->Next;
  } while (pp2 != pp);
  // 如果循环结束仍未找到相等的点，则返回 false
  return false;
}
//------------------------------------------------------------------------------

// 使用 Hormann & Agathos 的算法判断点是否在多边形内部
// 参考链接：http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.88.5498&rep=rep1&type=pdf
int PointInPolygon(const IntPoint &pt, const Path &path) {
  // 返回值：0 表示不在内部，+1 表示在内部，-1 表示在多边形边界上
  int result = 0;
  // 获取多边形的顶点数量
  size_t cnt = path.size();
  // 如果顶点数量小于 3，直接返回不在内部
  if (cnt < 3)
    return 0;
  // 获取多边形的第一个顶点
  IntPoint ip = path[0];
  // 遍历多边形的每个顶点
  for (size_t i = 1; i <= cnt; ++i) {
    // 获取下一个顶点
    IntPoint ipNext = (i == cnt ? path[0] : path[i]);
    // 如果下一个顶点的 Y 坐标与给定点的 Y 坐标相等
    if (ipNext.Y == pt.Y) {
      // 如果下一个顶点的 X 坐标与给定点的 X 坐标相等，或者在同一水平线上
      if ((ipNext.X == pt.X) ||
          (ip.Y == pt.Y && ((ipNext.X > pt.X) == (ip.X < pt.X))))
        return -1;
    }
    // 如果给定点的 Y 坐标在当前顶点和下一个顶点的 Y 坐标之间
    if ((ip.Y < pt.Y) != (ipNext.Y < pt.Y)) {
      // 如果当前顶点的 X 坐标大于等于给定点的 X 坐标
      if (ip.X >= pt.X) {
        // 根据 X 坐标关系更新 result
        if (ipNext.X > pt.X)
          result = 1 - result;
        else {
          // 计算叉积，判断给定点是否在多边形内部
          double d = (double)(ip.X - pt.X) * (ipNext.Y - pt.Y) -
                     (double)(ipNext.X - pt.X) * (ip.Y - pt.Y);
          if (!d)
            return -1;
          if ((d > 0) == (ipNext.Y > ip.Y))
            result = 1 - result;
        }
      } else {
        // 根据 X 坐标关系更新 result
        if (ipNext.X > pt.X) {
          // 计算叉积，判断给定点是否在多边形内部
          double d = (double)(ip.X - pt.X) * (ipNext.Y - pt.Y) -
                     (double)(ipNext.X - pt.X) * (ip.Y - pt.Y);
          if (!d)
            return -1;
          if ((d > 0) == (ipNext.Y > ip.Y))
            result = 1 - result;
        }
      }
    }
    // 更新当前顶点为下一个顶点
    ip = ipNext;
  }
  // 返回最终结果
  return result;
}
//------------------------------------------------------------------------------

// 判断给定点是否在多边形内部
int PointInPolygon(const IntPoint &pt, OutPt *op) {
  // 返回值：0 表示不在内部，+1 表示在内部，-1 表示在多边形边界上
  int result = 0;
  // 记录起始的 OutPt 指针
  OutPt *startOp = op;
  // 无限循环，直到找到结果
  for (;;) {
    // 如果下一个点的 Y 坐标等于给定点的 Y 坐标
    if (op->Next->Pt.Y == pt.Y) {
      // 如果下一个点的 X 坐标等于给定点的 X 坐标，或者当前点的 Y 坐标等于给定点的 Y 坐标且下一个点的 X 坐标相对给定点的 X 坐标位置关系相反
      if ((op->Next->Pt.X == pt.X) ||
          (op->Pt.Y == pt.Y && ((op->Next->Pt.X > pt.X) == (op->Pt.X < pt.X))))
        return -1;
    }
    // 如果当前点和下一个点的 Y 坐标与给定点的 Y 坐标位置关系不同
    if ((op->Pt.Y < pt.Y) != (op->Next->Pt.Y < pt.Y)) {
      // 如果当前点的 X 坐标大于等于给定点的 X 坐标
      if (op->Pt.X >= pt.X) {
        // 如果下一个点的 X 坐标大于给定点的 X 坐标
        if (op->Next->Pt.X > pt.X)
          result = 1 - result;
        else {
          // 计算叉积
          double d = (double)(op->Pt.X - pt.X) * (op->Next->Pt.Y - pt.Y) -
                     (double)(op->Next->Pt.X - pt.X) * (op->Pt.Y - pt.Y);
          // 如果叉积为 0，则返回 -1
          if (!d)
            return -1;
          // 如果叉积的正负与下一个点的 Y 坐标相对当前点的 Y 坐标位置关系相同，则更新结果
          if ((d > 0) == (op->Next->Pt.Y > op->Pt.Y))
            result = 1 - result;
        }
      } else {
        // 如果下一个点的 X 坐标大于给定点的 X 坐标
        if (op->Next->Pt.X > pt.X) {
          // 计算叉积
          double d = (double)(op->Pt.X - pt.X) * (op->Next->Pt.Y - pt.Y) -
                     (double)(op->Next->Pt.X - pt.X) * (op->Pt.Y - pt.Y);
          // 如果叉积为 0，则返回 -1
          if (!d)
            return -1;
          // 如果叉积的正负与下一个点的 Y 坐标相对当前点的 Y 坐标位置关系相同，则更新结果
          if ((d > 0) == (op->Next->Pt.Y > op->Pt.Y))
            result = 1 - result;
        }
      }
    }
    // 移动到下一个点
    op = op->Next;
    // 如果回到起始点，则跳出循环
    if (startOp == op)
      break;
  }
  // 返回最终结果
  return result;
// 检查一个多边形是否完全包含另一个多边形
bool Poly2ContainsPoly1(OutPt *OutPt1, OutPt *OutPt2) {
  // 从第一个多边形的顶点开始遍历
  OutPt *op = OutPt1;
  do {
    // 调用 PointInPolygon 函数判断点是否在多边形内部
    // 返回值：0 表示不在内部，+1 表示在内部，-1 表示在多边形上
    int res = PointInPolygon(op->Pt, OutPt2);
    if (res >= 0)
      return res > 0;
    op = op->Next;
  } while (op != OutPt1);
  return true;
}

// 比较两条边的斜率是否相等
bool SlopesEqual(const TEdge &e1, const TEdge &e2, bool UseFullInt64Range) {
#ifndef use_int32
  // 如果使用完整的 Int64 范围
  if (UseFullInt64Range)
    return Int128Mul(e1.Top.Y - e1.Bot.Y, e2.Top.X - e2.Bot.X) ==
           Int128Mul(e1.Top.X - e1.Bot.X, e2.Top.Y - e2.Bot.Y);
  else
#endif
    return (e1.Top.Y - e1.Bot.Y) * (e2.Top.X - e2.Bot.X) ==
           (e1.Top.X - e1.Bot.X) * (e2.Top.Y - e2.Bot.Y);
}

// 比较三个点的斜率是否相等
bool SlopesEqual(const IntPoint pt1, const IntPoint pt2, const IntPoint pt3,
                 bool UseFullInt64Range) {
#ifndef use_int32
  // 如果使用完整的 Int64 范围
  if (UseFullInt64Range)
    return Int128Mul(pt1.Y - pt2.Y, pt2.X - pt3.X) ==
           Int128Mul(pt1.X - pt2.X, pt2.Y - pt3.Y);
  else
#endif
    return (pt1.Y - pt2.Y) * (pt2.X - pt3.X) ==
           (pt1.X - pt2.X) * (pt2.Y - pt3.Y);
}

// 比较四个点的斜率是否相等
bool SlopesEqual(const IntPoint pt1, const IntPoint pt2, const IntPoint pt3,
                 const IntPoint pt4, bool UseFullInt64Range) {
#ifndef use_int32
  // 如果使用完整的 Int64 范围
  if (UseFullInt64Range)
    return Int128Mul(pt1.Y - pt2.Y, pt3.X - pt4.X) ==
           Int128Mul(pt1.X - pt2.X, pt3.Y - pt4.Y);
  else
#endif
    return (pt1.Y - pt2.Y) * (pt3.X - pt4.X) ==
           (pt1.X - pt2.X) * (pt3.Y - pt4.Y);
}

// 判断一条边是否水平
inline bool IsHorizontal(TEdge &e) { return e.Dx == HORIZONTAL; }
// 计算两个点之间的斜率
inline double GetDx(const IntPoint pt1, const IntPoint pt2) {
  return (pt1.Y == pt2.Y) ? HORIZONTAL
                          : (double)(pt2.X - pt1.X) / (pt2.Y - pt1.Y);
}
//---------------------------------------------------------------------------

// 设置边的斜率
inline void SetDx(TEdge &e) {
  cInt dy = (e.Top.Y - e.Bot.Y);
  if (dy == 0)
    e.Dx = HORIZONTAL;
  else
    e.Dx = (double)(e.Top.X - e.Bot.X) / dy;
}
//---------------------------------------------------------------------------

// 交换两条边的边侧
inline void SwapSides(TEdge &Edge1, TEdge &Edge2) {
  EdgeSide Side = Edge1.Side;
  Edge1.Side = Edge2.Side;
  Edge2.Side = Side;
}
//------------------------------------------------------------------------------

// 交换两条边的多边形索引
inline void SwapPolyIndexes(TEdge &Edge1, TEdge &Edge2) {
  int OutIdx = Edge1.OutIdx;
  Edge1.OutIdx = Edge2.OutIdx;
  Edge2.OutIdx = OutIdx;
}
//------------------------------------------------------------------------------

// 计算边在指定 Y 坐标处的 X 坐标
inline cInt TopX(TEdge &edge, const cInt currentY) {
  return (currentY == edge.Top.Y)
             ? edge.Top.X
             : edge.Bot.X + Round(edge.Dx * (currentY - edge.Bot.Y));
}
//------------------------------------------------------------------------------

// 计算两条边的交点
void IntersectPoint(TEdge &Edge1, TEdge &Edge2, IntPoint &ip) {
#ifdef use_xyz
  ip.Z = 0;
#endif

  double b1, b2;
  if (Edge1.Dx == Edge2.Dx) {
    ip.Y = Edge1.Curr.Y;
    ip.X = TopX(Edge1, ip.Y);
    return;
  } else if (Edge1.Dx == 0) {
    ip.X = Edge1.Bot.X;
    if (IsHorizontal(Edge2))
      ip.Y = Edge2.Bot.Y;
    else {
      b2 = Edge2.Bot.Y - (Edge2.Bot.X / Edge2.Dx);
      ip.Y = Round(ip.X / Edge2.Dx + b2);
    }
  } else if (Edge2.Dx == 0) {
    ip.X = Edge2.Bot.X;
    if (IsHorizontal(Edge1))
      ip.Y = Edge1.Bot.Y;
    else {
      b1 = Edge1.Bot.Y - (Edge1.Bot.X / Edge1.Dx);
      ip.Y = Round(ip.X / Edge1.Dx + b1);
    }
  } else {
    b1 = Edge1.Bot.X - Edge1.Bot.Y * Edge1.Dx;
    # 计算两条边的交点的 Y 坐标
    b2 = Edge2.Bot.X - Edge2.Bot.Y * Edge2.Dx;
    # 计算交点的 X 坐标
    double q = (b2 - b1) / (Edge1.Dx - Edge2.Dx);
    # 对交点的 Y 坐标四舍五入
    ip.Y = Round(q);
    # 根据两条边的斜率大小确定交点的 X 坐标
    if (std::fabs(Edge1.Dx) < std::fabs(Edge2.Dx))
      ip.X = Round(Edge1.Dx * q + b1);
    else
      ip.X = Round(Edge2.Dx * q + b2);
  }

  # 如果交点的 Y 坐标小于两条边的顶部 Y 坐标，则调整交点的位置
  if (ip.Y < Edge1.Top.Y || ip.Y < Edge2.Top.Y) {
    # 根据两条边的顶部 Y 坐标确定交点的 Y 坐标
    if (Edge1.Top.Y > Edge2.Top.Y)
      ip.Y = Edge1.Top.Y;
    else
      ip.Y = Edge2.Top.Y;
    # 根据两条边的斜率大小确定交点的 X 坐标
    if (std::fabs(Edge1.Dx) < std::fabs(Edge2.Dx))
      ip.X = TopX(Edge1, ip.Y);
    else
      ip.X = TopX(Edge2, ip.Y);
  }
  // 最后，确保交点的 Y 坐标不低于当前扫描线的 Y 坐标
  if (ip.Y > Edge1.Curr.Y) {
    ip.Y = Edge1.Curr.Y;
    # 根据两条边的斜率大小确定交点的 X 坐标
    if (std::fabs(Edge1.Dx) > std::fabs(Edge2.Dx))
      ip.X = TopX(Edge2, ip.Y);
    else
      ip.X = TopX(Edge1, ip.Y);
  }
// 反转多边形点之间的链接关系
void ReversePolyPtLinks(OutPt *pp) {
  // 如果输入为空指针，则直接返回
  if (!pp)
    return;
  OutPt *pp1, *pp2;
  pp1 = pp;
  // 循环遍历多边形点，反转其前后链接关系
  do {
    pp2 = pp1->Next;
    pp1->Next = pp1->Prev;
    pp1->Prev = pp2;
    pp1 = pp2;
  } while (pp1 != pp);
}
//------------------------------------------------------------------------------

// 释放多边形点内存
void DisposeOutPts(OutPt *&pp) {
  // 如果输入为空指针，则直接返回
  if (pp == 0)
    return;
  // 断开多边形点之间的链接关系
  pp->Prev->Next = 0;
  // 循环释放多边形点内存
  while (pp) {
    OutPt *tmpPp = pp;
    pp = pp->Next;
    delete tmpPp;
  }
}
//------------------------------------------------------------------------------

// 初始化边界信息
inline void InitEdge(TEdge *e, TEdge *eNext, TEdge *ePrev, const IntPoint &Pt) {
  // 使用 0 填充边界信息
  std::memset(e, int(0), sizeof(TEdge));
  // 设置边界的前后链接关系和当前点
  e->Next = eNext;
  e->Prev = ePrev;
  e->Curr = Pt;
  e->OutIdx = Unassigned;
}
//------------------------------------------------------------------------------

// 初始化边界信息
void InitEdge2(TEdge &e, PolyType Pt) {
  // 根据当前点和下一个点的 Y 坐标关系，设置边界的顶点和底点
  if (e.Curr.Y >= e.Next->Curr.Y) {
    e.Bot = e.Curr;
    e.Top = e.Next->Curr;
  } else {
    e.Top = e.Curr;
    e.Bot = e.Next->Curr;
  }
  // 计算边界的斜率
  SetDx(e);
  e.PolyTyp = Pt;
}
//------------------------------------------------------------------------------

// 移除边界
TEdge *RemoveEdge(TEdge *e) {
  // 从双向链表中移除边界
  e->Prev->Next = e->Next;
  e->Next->Prev = e->Prev;
  TEdge *result = e->Next;
  e->Prev = 0; // 标记为已移除
  return result;
}
//------------------------------------------------------------------------------

// 反转水平边界的顶点和底点
inline void ReverseHorizontal(TEdge &e) {
  // 交换水平边界的顶点和底点的 X 坐标，以便与相邻的下边界对齐
  std::swap(e.Top.X, e.Bot.X);
#ifdef use_xyz
  std::swap(e.Top.Z, e.Bot.Z);
#endif
}
//------------------------------------------------------------------------------
// 交换两个 IntPoint 类型的变量的数值
void SwapPoints(IntPoint &pt1, IntPoint &pt2) {
  IntPoint tmp = pt1; // 临时保存 pt1 的值
  pt1 = pt2; // 将 pt2 的值赋给 pt1
  pt2 = tmp; // 将临时保存的值赋给 pt2
}
//------------------------------------------------------------------------------

// 获取两条线段的重叠部分的端点
bool GetOverlapSegment(IntPoint pt1a, IntPoint pt1b, IntPoint pt2a,
                       IntPoint pt2b, IntPoint &pt1, IntPoint &pt2) {
  // 前提条件：线段共线
  if (Abs(pt1a.X - pt1b.X) > Abs(pt1a.Y - pt1b.Y)) {
    if (pt1a.X > pt1b.X)
      SwapPoints(pt1a, pt1b); // 确保 pt1a 的 X 坐标小于等于 pt1b 的 X 坐标
    if (pt2a.X > pt2b.X)
      SwapPoints(pt2a, pt2b); // 确保 pt2a 的 X 坐标小于等于 pt2b 的 X 坐标
    if (pt1a.X > pt2a.X)
      pt1 = pt1a;
    else
      pt1 = pt2a;
    if (pt1b.X < pt2b.X)
      pt2 = pt1b;
    else
      pt2 = pt2b;
    return pt1.X < pt2.X; // 返回是否 pt1 的 X 坐标小于 pt2 的 X 坐标
  } else {
    if (pt1a.Y < pt1b.Y)
      SwapPoints(pt1a, pt1b); // 确保 pt1a 的 Y 坐标大于等于 pt1b 的 Y 坐标
    if (pt2a.Y < pt2b.Y)
      SwapPoints(pt2a, pt2b); // 确保 pt2a 的 Y 坐标大于等于 pt2b 的 Y 坐标
    if (pt1a.Y < pt2a.Y)
      pt1 = pt1a;
    else
      pt1 = pt2a;
    if (pt1b.Y > pt2b.Y)
      pt2 = pt1b;
    else
      pt2 = pt2b;
    return pt1.Y > pt2.Y; // 返回是否 pt1 的 Y 坐标大于 pt2 的 Y 坐标
  }
}
//------------------------------------------------------------------------------

// 判断两个 OutPt 类型的对象中哪一个在底部
bool FirstIsBottomPt(const OutPt *btmPt1, const OutPt *btmPt2) {
  OutPt *p = btmPt1->Prev;
  while ((p->Pt == btmPt1->Pt) && (p != btmPt1))
    p = p->Prev;
  double dx1p = std::fabs(GetDx(btmPt1->Pt, p->Pt));
  p = btmPt1->Next;
  while ((p->Pt == btmPt1->Pt) && (p != btmPt1))
    p = p->Next;
  double dx1n = std::fabs(GetDx(btmPt1->Pt, p->Pt));

  p = btmPt2->Prev;
  while ((p->Pt == btmPt2->Pt) && (p != btmPt2))
    p = p->Prev;
  double dx2p = std::fabs(GetDx(btmPt2->Pt, p->Pt));
  p = btmPt2->Next;
  while ((p->Pt == btmPt2->Pt) && (p != btmPt2))
    p = p->Next;
  double dx2n = std::fabs(GetDx(btmPt2->Pt, p->Pt));

  if (std::max(dx1p, dx1n) == std::max(dx2p, dx2n) &&
      std::min(dx1p, dx1n) == std::min(dx2p, dx2n))
    return Area(btmPt1) > 0; // 如果相同则使用方向
  else
    return (dx1p >= dx2p && dx1p >= dx2n) || (dx1n >= dx2p && dx1n >= dx2n);
}
//------------------------------------------------------------------------------

// 获取最底部的点
OutPt *GetBottomPt(OutPt *pp) {
  OutPt *dups = 0;
  OutPt *p = pp->Next;
  while (p != pp) {
    if (p->Pt.Y > pp->Pt.Y) {
      pp = p;
      dups = 0;
    } else if (p->Pt.Y == pp->Pt.Y && p->Pt.X <= pp->Pt.X) {
      if (p->Pt.X < pp->Pt.X) {
        dups = 0;
        pp = p;
      } else {
        if (p->Next != pp && p->Prev != pp)
          dups = p;
      }
    }
    p = p->Next;
  }
  if (dups) {
    // 如果 BottomPt 看起来至少有 2 个顶点，那么...
    while (dups != p) {
      if (!FirstIsBottomPt(p, dups))
        pp = dups;
      dups = dups->Next;
      while (dups->Pt != pp->Pt)
        dups = dups->Next;
    }
  }
  return pp;
}
//------------------------------------------------------------------------------

// 判断 pt2 是否在 pt1 和 pt3 之间
bool Pt2IsBetweenPt1AndPt3(const IntPoint pt1, const IntPoint pt2,
                           const IntPoint pt3) {
  if ((pt1 == pt3) || (pt1 == pt2) || (pt3 == pt2))
    return false;
  else if (pt1.X != pt3.X)
    return (pt2.X > pt1.X) == (pt2.X < pt3.X);
  else
    return (pt2.Y > pt1.Y) == (pt2.Y < pt3.Y);
}
//------------------------------------------------------------------------------

// 判断水平线段是否重叠
bool HorzSegmentsOverlap(cInt seg1a, cInt seg1b, cInt seg2a, cInt seg2b) {
  if (seg1a > seg1b)
    std::swap(seg1a, seg1b);
  if (seg2a > seg2b)
    std::swap(seg2a, seg2b);
  return (seg1a < seg2b) && (seg2a < seg1b);
}

//------------------------------------------------------------------------------
// ClipperBase 类方法...
//------------------------------------------------------------------------------

ClipperBase::ClipperBase() // 构造函数
{
  m_CurrentLM = m_MinimaList.begin(); // 在这里 begin() == end()
  m_UseFullRange = false;
}
//------------------------------------------------------------------------------

ClipperBase::~ClipperBase() // 析构函数
{
  Clear();
}
// 检查给定点是否在指定范围内，根据 useFullRange 参数决定使用全范围还是局部范围
void RangeTest(const IntPoint &Pt, bool &useFullRange) {
  // 如果使用全范围
  if (useFullRange) {
    // 检查点的坐标是否超出允许范围
    if (Pt.X > hiRange || Pt.Y > hiRange || -Pt.X > hiRange || -Pt.Y > hiRange)
      throw clipperException("Coordinate outside allowed range");
  } 
  // 如果使用局部范围
  else if (Pt.X > loRange || Pt.Y > loRange || -Pt.X > loRange || -Pt.Y > loRange) {
    // 切换为使用全范围，并重新进行范围检查
    useFullRange = true;
    RangeTest(Pt, useFullRange);
  }
}
//------------------------------------------------------------------------------

// 查找下一个局部最小值
TEdge *FindNextLocMin(TEdge *E) {
  // 无限循环，直到找到下一个局部最小值
  for (;;) {
    // 移动到下一个不是同一水平线的边
    while (E->Bot != E->Prev->Bot || E->Curr == E->Top)
      E = E->Next;
    // 如果当前边和前一边都不是水平线，则找到下一个局部最小值
    if (!IsHorizontal(*E) && !IsHorizontal(*E->Prev))
      break;
    // 移动到前一边不是水平线的位置
    while (IsHorizontal(*E->Prev))
      E = E->Prev;
    TEdge *E2 = E;
    // 移动到下一边不是水平线的位置
    while (IsHorizontal(*E))
      E = E->Next;
    // 如果是水平线，则继续查找下一个
    if (E->Top.Y == E->Prev->Bot.Y)
      continue;
    // 如果前一边的底部 X 坐标小于当前边的底部 X 坐标，则将当前边设置为前一边
    if (E2->Prev->Bot.X < E->Bot.X)
      E = E2;
    break;
  }
  return E;
}
//------------------------------------------------------------------------------

// 处理边界
TEdge *ClipperBase::ProcessBound(TEdge *E, bool NextIsForward) {
  TEdge *Result = E;
  TEdge *Horz = 0;

  if (E->OutIdx == Skip) {
    // 如果当前边的 OutIdx 为 Skip
    // 如果下一个方向是正向
    if (NextIsForward) {
      // 移动到下一个不是同一水平线的边
      while (E->Top.Y == E->Next->Bot.Y)
        E = E->Next;
      // 在第二次解析边界时不包括顶部水平线
      while (E != Result && IsHorizontal(*E))
        E = E->Prev;
    } else {
      // 移动到前一个不是同一水平线的边
      while (E->Top.Y == E->Prev->Bot.Y)
        E = E->Prev;
      // 在第二次解析边界时不包括顶部水平线
      while (E != Result && IsHorizontal(*E))
        E = E->Next;
    }

    if (E == Result) {
      // 如果当前边等于结果边
      if (NextIsForward)
        Result = E->Next;
      else
        Result = E->Prev;
  } else {
    // 如果边界中还有以 E 开头的结果之外的更多边
    if (NextIsForward)
      // 如果下一个是正向的，则 E 等于结果的下一个边
      E = Result->Next;
    else
      // 如果下一个是反向的，则 E 等于结果的前一个边
      E = Result->Prev;
    // 创建一个局部最小值对象 locMin
    MinimaList::value_type locMin;
    // 设置 locMin 的 Y 值为 E 的底部 Y 值
    locMin.Y = E->Bot.Y;
    // 设置 locMin 的左边界为 0
    locMin.LeftBound = 0;
    // 设置 locMin 的右边界为 E
    locMin.RightBound = E;
    // 将 E 的 WindDelta 设置为 0
    E->WindDelta = 0;
    // 处理边界，返回结果
    Result = ProcessBound(E, NextIsForward);
    // 将 locMin 添加到 m_MinimaList 中
    m_MinimaList.push_back(locMin);
  }
  // 返回结果
  return Result;
}

// 初始化 EStart
TEdge *EStart;

// 如果边是水平的
if (IsHorizontal(*E)) {
  // 对于开放路径需要小心，因为这可能不是真正的局部最小值（即 E 可能跟随一个跳过边）
  // 此外，连续的水平边可能在向右前先向左
  if (NextIsForward)
    // 如果下一个是正向的，则 EStart 等于 E 的前一个边
    EStart = E->Prev;
  else
    // 如果下一个是反向的，则 EStart 等于 E 的下一个边
    EStart = E->Next;
  // 如果 EStart 是水平的（即相邻的水平跳过边）
  if (IsHorizontal(*EStart)) {
    // 如果 EStart 的底部 X 值不等于 E 的底部 X 值且 EStart 的顶部 X 值不等于 E 的底部 X 值
    if (EStart->Bot.X != E->Bot.X && EStart->Top.X != E->Bot.X)
      // 反转水平边 E
      ReverseHorizontal(*E);
  } else if (EStart->Bot.X != E->Bot.X)
    // 如果 EStart 的底部 X 值不等于 E 的底部 X 值，则反转水平边 E
    ReverseHorizontal(*E);
}

// 将 EStart 设置为 E
EStart = E;
// 如果下一个是正向的
if (NextIsForward) {
  // 当结果的顶部 Y 值等于结果的下一个边的底部 Y 值且下一个边的 OutIdx 不等于 Skip 时
  while (Result->Top.Y == Result->Next->Bot.Y && Result->Next->OutIdx != Skip)
    Result = Result->Next;
  // 如果是水平的且下一个边的 OutIdx 不等于 Skip
  if (IsHorizontal(*Result) && Result->Next->OutIdx != Skip) {
    // 在边界的顶部，只有当前一个边连接到水平边的左顶点时，才将水平边添加到边界
    // 除非遇到 Skip 边，那么水平边就成为顶部分界
    Horz = Result;
    while (IsHorizontal(*Horz->Prev))
      Horz = Horz->Prev;
    if (Horz->Prev->Top.X > Result->Next->Top.X)
      Result = Horz->Prev;
  }
  // 当 E 不等于 Result 时
  while (E != Result) {
    // 将 E 的 NextInLML 设置为 E 的下一个边
    E->NextInLML = E->Next;
    // 如果是水平的且不是 EStart 且 E 的底部 X 值不等于 E 的前一个边的顶部 X 值
    if (IsHorizontal(*E) && E != EStart && E->Bot.X != E->Prev->Top.X)
      // 反转水平边 E
      ReverseHorizontal(*E);
    // 将 E 设置为下一个边
    E = E->Next;
  }
  // 如果是水平的且不是 EStart 且 E 的底部 X 值不等于 E 的前一个边的顶部 X 值
  if (IsHorizontal(*E) && E != EStart && E->Bot.X != E->Prev->Top.X)
    // 反转水平边 E
    ReverseHorizontal(*E);
    Result = Result->Next; // 将 Result 移动到当前边界之外的下一个边界
  } else {
    while (Result->Top.Y == Result->Prev->Bot.Y && Result->Prev->OutIdx != Skip)
      Result = Result->Prev;
    // 如果当前边界是水平边界且前一个边界不是要跳过的边界
    if (IsHorizontal(*Result) && Result->Prev->OutIdx != Skip) {
      Horz = Result;
      // 找到连续的水平边界
      while (IsHorizontal(*Horz->Next))
        Horz = Horz->Next;
      // 如果下一个边界的顶点 X 坐标等于前一个边界的顶点 X 坐标，或者下一个边界的顶点 X 坐标大于前一个边界的顶点 X 坐标
      if (Horz->Next->Top.X == Result->Prev->Top.X ||
          Horz->Next->Top.X > Result->Prev->Top.X)
        Result = Horz->Next;
    }

    // 将当前边界之外的边界的 NextInLML 指向前一个边界
    while (E != Result) {
      E->NextInLML = E->Prev;
      // 如果当前边界是水平边界且不是起始边界且底部 X 坐标不等于下一个边界的顶部 X 坐标
      if (IsHorizontal(*E) && E != EStart && E->Bot.X != E->Next->Top.X)
        ReverseHorizontal(*E);
      E = E->Prev;
    }
    // 如果当前边界是水平边界且不是起始边界且底部 X 坐标不等于下一个边界的顶部 X 坐标
    if (IsHorizontal(*E) && E != EStart && E->Bot.X != E->Next->Top.X)
      ReverseHorizontal(*E);
    Result = Result->Prev; // 将 Result 移动到当前边界之外的下一个边界
  }

  return Result;
// ClipperBase 类的 AddPath 方法，用于向路径集合中添加路径
bool ClipperBase::AddPath(const Path &pg, PolyType PolyTyp, bool Closed) {
#ifdef use_lines
  // 如果路径未闭合且为剪切路径，则抛出异常
  if (!Closed && PolyTyp == ptClip)
    throw clipperException("AddPath: Open paths must be subject.");
#else
  // 如果路径未闭合，则抛出异常
  if (!Closed)
    throw clipperException("AddPath: Open paths have been disabled.");
#endif

  // 计算路径中的顶点数量
  int highI = (int)pg.size() - 1;
  // 如果路径闭合，则去除重复的起始点
  if (Closed)
    while (highI > 0 && (pg[highI] == pg[0]))
      --highI;
  // 去除重复的顶点
  while (highI > 0 && (pg[highI] == pg[highI - 1]))
    --highI;
  // 如果路径点数量不符合要求，则返回 false
  if ((Closed && highI < 2) || (!Closed && highI < 1))
    return false;

  // 创建一个新的边缘数组
  TEdge *edges = new TEdge[highI + 1];

  bool IsFlat = true;
  // 1. 初始化基本边缘
  try {
    edges[1].Curr = pg[1];
    RangeTest(pg[0], m_UseFullRange);
    RangeTest(pg[highI], m_UseFullRange);
    InitEdge(&edges[0], &edges[1], &edges[highI], pg[0]);
    InitEdge(&edges[highI], &edges[0], &edges[highI - 1], pg[highI]);
    for (int i = highI - 1; i >= 1; --i) {
      RangeTest(pg[i], m_UseFullRange);
      InitEdge(&edges[i], &edges[i + 1], &edges[i - 1], pg[i]);
    }
  } catch (...) {
    delete[] edges;
    throw; // 范围测试失败
  }
  TEdge *eStart = &edges[0];

  // 2. 去除重复顶点和（闭合时）共线边
  TEdge *E = eStart, *eLoopStop = eStart;
  for (;;) {
    // 允许匹配起始点和结束点，即使未闭合
    if (E->Curr == E->Next->Curr && (Closed || E->Next != eStart)) {
      if (E == E->Next)
        break;
      if (E == eStart)
        eStart = E->Next;
      E = RemoveEdge(E);
      eLoopStop = E;
      continue;
    }
    if (E->Prev == E->Next)
      break; // 只有两个顶点
    // 如果边界已经关闭并且当前边与前一边和后一边的斜率相等，并且不保留共线点或者当前点不在前一点和后一点之间
    else if (Closed && SlopesEqual(E->Prev->Curr, E->Curr, E->Next->Curr,
                                   m_UseFullRange) &&
             (!m_PreserveCollinear ||
              !Pt2IsBetweenPt1AndPt3(E->Prev->Curr, E->Curr, E->Next->Curr))) {
      // 允许在开放路径中存在共线边，但在闭合路径中，默认情况下将相邻的共线边合并为一条边
      // 但是，如果启用了PreserveCollinear属性，则仅从闭合路径中删除重叠的共线边（即尖峰）
      if (E == eStart)
        eStart = E->Next;
      // 移除当前边
      E = RemoveEdge(E);
      E = E->Prev;
      eLoopStop = E;
      continue;
    }
    // 移动到下一条边
    E = E->Next;
    // 如果到达循环结束点或者（未闭合并且下一条边是起始边）
    if ((E == eLoopStop) || (!Closed && E->Next == eStart))
      break;
  }

  // 如果路径是完全平坦的或者闭合路径中只有一个边
  if ((!Closed && (E == E->Next)) || (Closed && (E->Prev == E->Next))) {
    // 释放内存并返回false
    delete[] edges;
    return false;
  }

  // 如果路径是开放的
  if (!Closed) {
    // 设置标记为存在开放路径
    m_HasOpenPaths = true;
    // 设置起始边的前一条边的OutIdx为Skip
    eStart->Prev->OutIdx = Skip;
  }

  // 3. 进行边的第二阶段初始化...
  E = eStart;
  do {
    // 初始化边的第二阶段
    InitEdge2(*E, PolyTyp);
    E = E->Next;
    // 如果是平坦的并且当前点的Y坐标不等于起始点的Y坐标
    if (IsFlat && E->Curr.Y != eStart->Curr.Y)
      IsFlat = false;
  } while (E != eStart);

  // 4. 最后，将边界添加到LocalMinima列表中...

  // 当将完全平坦的路径添加到LocalMinima列表时，必须以不同的方式处理，以避免无限循环等问题...
  if (IsFlat) {
    // 如果是闭合路径
    if (Closed) {
      // 释放内存并返回false
      delete[] edges;
      return false;
    }
    // 设置前一条边的OutIdx为Skip
    E->Prev->OutIdx = Skip;
    // 创建一个LocalMinima对象
    MinimaList::value_type locMin;
    locMin.Y = E->Bot.Y;
    locMin.LeftBound = 0;
    locMin.RightBound = E;
    locMin.RightBound->Side = esRight;
    locMin.RightBound->WindDelta = 0;
    for (;;) {
      // 如果底部点的X坐标不等于前一条边的顶部点的X坐标
      if (E->Bot.X != E->Prev->Top.X)
        ReverseHorizontal(*E);
      // 如果下一条边的OutIdx为Skip，则跳出循环
      if (E->Next->OutIdx == Skip)
        break;
      // 设置下一条边的NextInLML为下一条边
      E->NextInLML = E->Next;
      E = E->Next;
    }
    // 将locMin添加到MinimaList中
    m_MinimaList.push_back(locMin);
    // 将edges添加到m_edges中
    m_edges.push_back(edges);
    // 返回 true
    return true;
  }

  // 将边界添加到边缘列表
  m_edges.push_back(edges);
  // 声明变量 leftBoundIsForward 和 EMin
  bool leftBoundIsForward;
  TEdge *EMin = 0;

  // 解决当开放路径具有匹配的起始点和终点时，在下面的 while 循环中避免无限循环的问题
  if (E->Prev->Bot == E->Prev->Top)
    E = E->Next;

  // 无限循环，直到找到下一个局部最小值
  for (;;) {
    E = FindNextLocMin(E);
    // 如果找到的局部最小值等于 EMin，则退出循环
    if (E == EMin)
      break;
    else if (!EMin)
      EMin = E;

    // E 和 E.Prev 现在共享一个局部最小值（如果是水平的，则左对齐）
    // 比较它们的斜率以找到哪个开始哪个边界...
    MinimaList::value_type locMin;
    locMin.Y = E->Bot.Y;
    if (E->Dx < E->Prev->Dx) {
      locMin.LeftBound = E->Prev;
      locMin.RightBound = E;
      leftBoundIsForward = false; // Q.nextInLML = Q.prev
    } else {
      locMin.LeftBound = E;
      locMin.RightBound = E->Prev;
      leftBoundIsForward = true; // Q.nextInLML = Q.next
    }

    // 根据情况设置 WindDelta
    if (!Closed)
      locMin.LeftBound->WindDelta = 0;
    else if (locMin.LeftBound->Next == locMin.RightBound)
      locMin.LeftBound->WindDelta = -1;
    else
      locMin.LeftBound->WindDelta = 1;
    locMin.RightBound->WindDelta = -locMin.LeftBound->WindDelta;

    // 处理左边界
    E = ProcessBound(locMin.LeftBound, leftBoundIsForward);
    if (E->OutIdx == Skip)
      E = ProcessBound(E, leftBoundIsForward);

    // 处理右边界
    TEdge *E2 = ProcessBound(locMin.RightBound, !leftBoundIsForward);
    if (E2->OutIdx == Skip)
      E2 = ProcessBound(E2, !leftBoundIsForward);

    // 如果左边界或右边界的 OutIdx 为 Skip，则将其置为 0
    if (locMin.LeftBound->OutIdx == Skip)
      locMin.LeftBound = 0;
    else if (locMin.RightBound->OutIdx == Skip)
      locMin.RightBound = 0;
    // 将 locMin 添加到 MinimaList
    m_MinimaList.push_back(locMin);
    // 如果 leftBoundIsForward 为 false，则将 E 设置为 E2
    if (!leftBoundIsForward)
      E = E2;
  }
  // 返回 true
  return true;
// ClipperBase 类的 AddPaths 方法，用于向 Clipper 中添加路径
bool ClipperBase::AddPaths(const Paths &ppg, PolyType PolyTyp, bool Closed) {
  // 初始化结果为 false
  bool result = false;
  // 遍历传入的路径集合
  for (Paths::size_type i = 0; i < ppg.size(); ++i)
    // 如果成功添加路径，则将结果设置为 true
    if (AddPath(ppg[i], PolyTyp, Closed))
      result = true;
  // 返回结果
  return result;
}
//------------------------------------------------------------------------------

// ClipperBase 类的 Clear 方法，用于清空 Clipper 中的数据
void ClipperBase::Clear() {
  // 清空本地最小值列表
  DisposeLocalMinimaList();
  // 遍历边缘列表，释放内存
  for (EdgeList::size_type i = 0; i < m_edges.size(); ++i) {
    TEdge *edges = m_edges[i];
    delete[] edges;
  }
  // 清空边缘列表
  m_edges.clear();
  // 重置标志位
  m_UseFullRange = false;
  m_HasOpenPaths = false;
}
//------------------------------------------------------------------------------

// ClipperBase 类的 Reset 方法，用于重置 Clipper 中的数据
void ClipperBase::Reset() {
  // 将当前最小值指针指向最小值列表的开头
  m_CurrentLM = m_MinimaList.begin();
  // 如果最小值列表为空，则直接返回
  if (m_CurrentLM == m_MinimaList.end())
    return;
  // 对最小值列表进行排序
  std::sort(m_MinimaList.begin(), m_MinimaList.end(), LocMinSorter());

  // 重置扫描线队列
  m_Scanbeam = ScanbeamList();
  // 重置所有边缘
  for (MinimaList::iterator lm = m_MinimaList.begin(); lm != m_MinimaList.end();
       ++lm) {
    InsertScanbeam(lm->Y);
    TEdge *e = lm->LeftBound;
    if (e) {
      e->Curr = e->Bot;
      e->Side = esLeft;
      e->OutIdx = Unassigned;
    }

    e = lm->RightBound;
    if (e) {
      e->Curr = e->Bot;
      e->Side = esRight;
      e->OutIdx = Unassigned;
    }
  }
  // 重置活动边缘数
  m_ActiveEdges = 0;
  // 将当前最小值指针指向最小值列表的开头
  m_CurrentLM = m_MinimaList.begin();
}
//------------------------------------------------------------------------------

// ClipperBase 类的 DisposeLocalMinimaList 方法，用于清空最小值列表
void ClipperBase::DisposeLocalMinimaList() {
  // 清空最小值列表
  m_MinimaList.clear();
  // 将当前最小值指针指向最小值列表的开头
  m_CurrentLM = m_MinimaList.begin();
}
//------------------------------------------------------------------------------

// ClipperBase 类的 PopLocalMinima 方法，用于弹出最小值
bool ClipperBase::PopLocalMinima(cInt Y, const LocalMinimum *&locMin) {
  // 如果当前最小值指针指向最小值列表的末尾，或者当前最小值的 Y 坐标不等于传入的 Y 坐标，则返回 false
  if (m_CurrentLM == m_MinimaList.end() || (*m_CurrentLM).Y != Y)
    return false;
  // 将 locMin 指向当前最小值
  locMin = &(*m_CurrentLM);
  // 将当前最小值指针向后移动一位
  ++m_CurrentLM;
  // 返回 true
  return true;
}
//------------------------------------------------------------------------------

// 获取剪切区域的边界矩形
IntRect ClipperBase::GetBounds() {
  IntRect result;
  MinimaList::iterator lm = m_MinimaList.begin();
  // 如果最小值列表为空，则返回空矩形
  if (lm == m_MinimaList.end()) {
    result.left = result.top = result.right = result.bottom = 0;
    return result;
  }
  // 设置初始边界值
  result.left = lm->LeftBound->Bot.X;
  result.top = lm->LeftBound->Bot.Y;
  result.right = lm->LeftBound->Bot.X;
  result.bottom = lm->LeftBound->Bot.Y;
  // 遍历最小值列表，更新边界值
  while (lm != m_MinimaList.end()) {
    // todo - 针对开放路径进行修复
    result.bottom = std::max(result.bottom, lm->LeftBound->Bot.Y);
    TEdge *e = lm->LeftBound;
    for (;;) {
      TEdge *bottomE = e;
      while (e->NextInLML) {
        if (e->Bot.X < result.left)
          result.left = e->Bot.X;
        if (e->Bot.X > result.right)
          result.right = e->Bot.X;
        e = e->NextInLML;
      }
      result.left = std::min(result.left, e->Bot.X);
      result.right = std::max(result.right, e->Bot.X);
      result.left = std::min(result.left, e->Top.X);
      result.right = std::max(result.right, e->Top.X);
      result.top = std::min(result.top, e->Top.Y);
      if (bottomE == lm->LeftBound)
        e = lm->RightBound;
      else
        break;
    }
    ++lm;
  }
  return result;
}
//------------------------------------------------------------------------------

// 插入扫描线
void ClipperBase::InsertScanbeam(const cInt Y) { m_Scanbeam.push(Y); }
//------------------------------------------------------------------------------

// 弹出扫描线
bool ClipperBase::PopScanbeam(cInt &Y) {
  if (m_Scanbeam.empty())
    return false;
  Y = m_Scanbeam.top();
  m_Scanbeam.pop();
  while (!m_Scanbeam.empty() && Y == m_Scanbeam.top()) {
    m_Scanbeam.pop();
  } // 弹出重复的扫描线
  return true;
}
//------------------------------------------------------------------------------

// 释放所有输出记录
void ClipperBase::DisposeAllOutRecs() {
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
    DisposeOutRec(i);
  m_PolyOuts.clear();
}
// 释放指定索引处的 OutRec 对象
void ClipperBase::DisposeOutRec(PolyOutList::size_type index) {
  // 获取指定索引处的 OutRec 对象
  OutRec *outRec = m_PolyOuts[index];
  // 如果 OutRec 对象的点集存在，则释放点集
  if (outRec->Pts)
    DisposeOutPts(outRec->Pts);
  // 释放 OutRec 对象
  delete outRec;
  // 将指定索引处的 OutRec 对象置为 0
  m_PolyOuts[index] = 0;
}
//------------------------------------------------------------------------------

// 从 AEL 中删除指定的边
void ClipperBase::DeleteFromAEL(TEdge *e) {
  // 获取指定边的前一个和后一个边
  TEdge *AelPrev = e->PrevInAEL;
  TEdge *AelNext = e->NextInAEL;
  // 如果前一个和后一个边都不存在，并且当前边不是活动边，则直接返回，表示已经删除
  if (!AelPrev && !AelNext && (e != m_ActiveEdges))
    return; // already deleted
  // 更新前一个边的下一个边为后一个边，或者更新活动边为后一个边
  if (AelPrev)
    AelPrev->NextInAEL = AelNext;
  else
    m_ActiveEdges = AelNext;
  // 更新后一个边的前一个边为前一个边
  if (AelNext)
    AelNext->PrevInAEL = AelPrev;
  // 将当前边的前一个和后一个边置为 0
  e->NextInAEL = 0;
  e->PrevInAEL = 0;
}
//------------------------------------------------------------------------------

// 创建一个新的 OutRec 对象并返回
OutRec *ClipperBase::CreateOutRec() {
  // 创建一个新的 OutRec 对象
  OutRec *result = new OutRec;
  // 初始化 OutRec 对象的属性
  result->IsHole = false;
  result->IsOpen = false;
  result->FirstLeft = 0;
  result->Pts = 0;
  result->BottomPt = 0;
  result->PolyNd = 0;
  // 将新创建的 OutRec 对象添加到 PolyOuts 列表中
  m_PolyOuts.push_back(result);
  // 设置新创建的 OutRec 对象的索引
  result->Idx = (int)m_PolyOuts.size() - 1;
  // 返回新创建的 OutRec 对象
  return result;
}
//------------------------------------------------------------------------------

// 交换 AEL 中两个边的位置
void ClipperBase::SwapPositionsInAEL(TEdge *Edge1, TEdge *Edge2) {
  // 检查两个边是否已经从 AEL 中移除
  if (Edge1->NextInAEL == Edge1->PrevInAEL ||
      Edge2->NextInAEL == Edge2->PrevInAEL)
    return;

  // 如果 Edge1 的下一个边是 Edge2
  if (Edge1->NextInAEL == Edge2) {
    // 更新 Edge2 的下一个边的前一个边为 Edge1
    TEdge *Next = Edge2->NextInAEL;
    if (Next)
      Next->PrevInAEL = Edge1;
    // 更新 Edge1 的前一个边的下一个边为 Edge2
    TEdge *Prev = Edge1->PrevInAEL;
    if (Prev)
      Prev->NextInAEL = Edge2;
    // 更新 Edge2 的前一个边为 Edge1，后一个边为 Next
    Edge2->PrevInAEL = Prev;
    Edge2->NextInAEL = Edge1;
    // 更新 Edge1 的前一个边为 Edge2，后一个边为 Next
    Edge1->PrevInAEL = Edge2;
    Edge1->NextInAEL = Next;
  } 
  // 如果 Edge2 的下一个边是 Edge1
  else if (Edge2->NextInAEL == Edge1) {
    // 更新 Edge1 的下一个边的前一个边为 Edge2
    TEdge *Next = Edge1->NextInAEL;
    if (Next)
      Next->PrevInAEL = Edge2;
    // 更新 Edge2 的前一个边的下一个边为 Edge1
    TEdge *Prev = Edge2->PrevInAEL;
    if (Prev)
      Prev->NextInAEL = Edge1;
    // 更新 Edge1 的前一个边为 Prev
    Edge1->PrevInAEL = Prev;
    # 将 Edge2 设置为 Edge1 的下一个边
    Edge1->NextInAEL = Edge2;
    # 将 Edge1 设置为 Edge2 的前一个边
    Edge2->PrevInAEL = Edge1;
    # 将 Edge2 的下一个边设置为 Next
    Edge2->NextInAEL = Next;
  } else {
    # 保存 Edge1 的下一个边
    TEdge *Next = Edge1->NextInAEL;
    # 保存 Edge1 的前一个边
    TEdge *Prev = Edge1->PrevInAEL;
    # 将 Edge1 的下一个边设置为 Edge2 的下一个边
    Edge1->NextInAEL = Edge2->NextInAEL;
    # 如果 Edge1 的下一个边存在，则将其前一个边设置为 Edge1
    if (Edge1->NextInAEL)
      Edge1->NextInAEL->PrevInAEL = Edge1;
    # 将 Edge1 的前一个边设置为 Edge2 的前一个边
    Edge1->PrevInAEL = Edge2->PrevInAEL;
    # 如果 Edge1 的前一个边存在，则将其下一个边设置为 Edge1
    if (Edge1->PrevInAEL)
      Edge1->PrevInAEL->NextInAEL = Edge1;
    # 将 Edge2 的下一个边设置为 Next
    Edge2->NextInAEL = Next;
    # 如果 Edge2 的下一个边存在，则将其前一个边设置为 Edge2
    if (Edge2->NextInAEL)
      Edge2->NextInAEL->PrevInAEL = Edge2;
    # 将 Edge2 的前一个边设置为 Prev
    Edge2->PrevInAEL = Prev;
    # 如果 Edge2 的前一个边存在，则将其下一个边设置为 Edge2
    if (Edge2->PrevInAEL)
      Edge2->PrevInAEL->NextInAEL = Edge2;
  }

  # 如果 Edge1 的前一个边不存在，则将 Edge1 设置为活动边
  if (!Edge1->PrevInAEL)
    m_ActiveEdges = Edge1;
  # 如果 Edge2 的前一个边不存在，则将 Edge2 设置为活动边
  else if (!Edge2->PrevInAEL)
    m_ActiveEdges = Edge2;
// 更新边到活动边列表中
void ClipperBase::UpdateEdgeIntoAEL(TEdge *&e) {
  // 检查边是否有效
  if (!e->NextInLML)
    throw clipperException("UpdateEdgeIntoAEL: invalid call");

  // 更新下一个边在局部最小值列表中的索引
  e->NextInLML->OutIdx = e->OutIdx;
  TEdge *AelPrev = e->PrevInAEL;
  TEdge *AelNext = e->NextInAEL;
  // 更新前一个边的下一个边
  if (AelPrev)
    AelPrev->NextInAEL = e->NextInLML;
  else
    m_ActiveEdges = e->NextInLML;
  // 更新后一个边的前一个边
  if (AelNext)
    AelNext->PrevInAEL = e->NextInLML;
  // 更新下一个边的属性
  e->NextInLML->Side = e->Side;
  e->NextInLML->WindDelta = e->WindDelta;
  e->NextInLML->WindCnt = e->WindCnt;
  e->NextInLML->WindCnt2 = e->WindCnt2;
  e = e->NextInLML;
  e->Curr = e->Bot;
  e->PrevInAEL = AelPrev;
  e->NextInAEL = AelNext;
  // 如果边不是水平的，则插入扫描线
  if (!IsHorizontal(*e))
    InsertScanbeam(e->Top.Y);
}
//------------------------------------------------------------------------------

// 检查是否有局部最小值待处理
bool ClipperBase::LocalMinimaPending() {
  return (m_CurrentLM != m_MinimaList.end());
}

//------------------------------------------------------------------------------
// TClipper methods ...
//------------------------------------------------------------------------------

// Clipper 类的构造函数
Clipper::Clipper(int initOptions)
    : ClipperBase() // 调用基类构造函数
{
  m_ExecuteLocked = false;
  m_UseFullRange = false;
  m_ReverseOutput = ((initOptions & ioReverseSolution) != 0);
  m_StrictSimple = ((initOptions & ioStrictlySimple) != 0);
  m_PreserveCollinear = ((initOptions & ioPreserveCollinear) != 0);
  m_HasOpenPaths = false;
#ifdef use_xyz
  m_ZFill = 0;
#endif
}
//------------------------------------------------------------------------------

#ifdef use_xyz
// 设置 Z 填充函数
void Clipper::ZFillFunction(ZFillCallback zFillFunc) { m_ZFill = zFillFunc; }
//------------------------------------------------------------------------------
#endif

// 执行剪裁操作
bool Clipper::Execute(ClipType clipType, Paths &solution,
                      PolyFillType fillType) {
  return Execute(clipType, solution, fillType, fillType);
}
// Clipper 类的 Execute 方法重载，用于执行裁剪操作并返回结果
bool Clipper::Execute(ClipType clipType, PolyTree &polytree,
                      PolyFillType fillType) {
  // 调用另一个 Execute 方法，传入相同的参数
  return Execute(clipType, polytree, fillType, fillType);
}
//------------------------------------------------------------------------------

// Clipper 类的 Execute 方法重载，用于执行裁剪操作并返回结果
bool Clipper::Execute(ClipType clipType, Paths &solution,
                      PolyFillType subjFillType, PolyFillType clipFillType) {
  // 如果执行被锁定，则返回 false
  if (m_ExecuteLocked)
    return false;
  // 如果存在开放路径，则抛出异常
  if (m_HasOpenPaths)
    throw clipperException(
        "Error: PolyTree struct is needed for open path clipping.");
  // 锁定执行
  m_ExecuteLocked = true;
  // 清空解决方案路径
  solution.resize(0);
  // 设置主题填充类型和裁剪填充类型
  m_SubjFillType = subjFillType;
  m_ClipFillType = clipFillType;
  m_ClipType = clipType;
  m_UsingPolyTree = false;
  // 执行内部裁剪操作
  bool succeeded = ExecuteInternal();
  // 如果执行成功，则构建结果
  if (succeeded)
    BuildResult(solution);
  // 释放所有 OutRec 对象
  DisposeAllOutRecs();
  // 解锁执行
  m_ExecuteLocked = false;
  return succeeded;
}
//------------------------------------------------------------------------------

// Clipper 类的 Execute 方法重载，用于执行裁剪操作并返回结果
bool Clipper::Execute(ClipType clipType, PolyTree &polytree,
                      PolyFillType subjFillType, PolyFillType clipFillType) {
  // 如果执行被锁定，则返回 false
  if (m_ExecuteLocked)
    return false;
  // 锁定执行
  m_ExecuteLocked = true;
  // 设置主题填充类型和裁剪填充类型
  m_SubjFillType = subjFillType;
  m_ClipFillType = clipFillType;
  m_ClipType = clipType;
  m_UsingPolyTree = true;
  // 执行内部裁剪操作
  bool succeeded = ExecuteInternal();
  // 如果执行成功，则构建结果
  if (succeeded)
    BuildResult2(polytree);
  // 释放所有 OutRec 对象
  DisposeAllOutRecs();
  // 解锁执行
  m_ExecuteLocked = false;
  return succeeded;
}
//------------------------------------------------------------------------------

// 修复 OutRec 对象之间的链接关系
void Clipper::FixHoleLinkage(OutRec &outrec) {
  // 跳过包含最外层多边形或已具有正确所有者/子链接的 OutRec 对象
  if (!outrec.FirstLeft ||
      (outrec.IsHole != outrec.FirstLeft->IsHole && outrec.FirstLeft->Pts))
    return;

  OutRec *orfl = outrec.FirstLeft;
  while (orfl && ((orfl->IsHole == outrec.IsHole) || !orfl->Pts))
    # 将指针 orfl 指向的对象的 FirstLeft 属性赋值给 orfl，相当于将 orfl 指向的对象的 FirstLeft 属性值赋给 orfl
    orfl = orfl->FirstLeft;
    # 将 orfl 指向的对象赋值给 outrec 的 FirstLeft 属性
    outrec.FirstLeft = orfl;
//------------------------------------------------------------------------------

// Clipper 类的内部执行函数，返回布尔值表示执行是否成功
bool Clipper::ExecuteInternal() {
  // 初始化执行状态
  bool succeeded = true;
  try {
    // 重置 Clipper 对象
    Reset();
    // 初始化最大值列表和排序边
    m_Maxima = MaximaList();
    m_SortedEdges = 0;

    // 设置执行状态为真
    succeeded = true;
    // 定义底部和顶部扫描线的 Y 坐标
    cInt botY, topY;
    // 如果无法弹出底部扫描线，则返回执行失败
    if (!PopScanbeam(botY))
      return false;
    // 将局部最小值插入到活动边表中
    InsertLocalMinimaIntoAEL(botY);
    // 当仍有顶部扫描线或者还有待处理的局部最小值时循环执行
    while (PopScanbeam(topY) || LocalMinimaPending()) {
      // 处理水平边
      ProcessHorizontals();
      // 清除虚拟连接
      ClearGhostJoins();
      // 处理交点
      if (!ProcessIntersections(topY)) {
        succeeded = false;
        break;
      }
      // 处理扫描线顶部的边
      ProcessEdgesAtTopOfScanbeam(topY);
      botY = topY;
      // 将局部最小值插入到活动边表中
      InsertLocalMinimaIntoAEL(botY);
    }
  } catch (...) {
    // 捕获异常，设置执行状态为失败
    succeeded = false;
  }

  // 如果执行成功
  if (succeeded) {
    // 修正方向...
    for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i) {
      OutRec *outRec = m_PolyOuts[i];
      // 如果点为空或者多边形是开放的，则继续下一次循环
      if (!outRec->Pts || outRec->IsOpen)
        continue;
      // 如果多边形是孔或者需要反转输出，并且面积大于 0，则反转多边形点链接
      if ((outRec->IsHole ^ m_ReverseOutput) == (Area(*outRec) > 0))
        ReversePolyPtLinks(outRec->Pts);
    }

    // 如果连接列表不为空，则执行连接共同边
    if (!m_Joins.empty())
      JoinCommonEdges();

    // 不幸的是，FixupOutPolygon() 必须在 JoinCommonEdges() 之后执行
    for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i) {
      OutRec *outRec = m_PolyOuts[i];
      // 如果点为空，则继续下一次循环
      if (!outRec->Pts)
        continue;
      // 如果多边形是开放的，则修正多边形线
      if (outRec->IsOpen)
        FixupOutPolyline(*outRec);
      else
        FixupOutPolygon(*outRec);
    }

    // 如果设置了严格简单多边形标志，则执行简单多边形处理
    if (m_StrictSimple)
      DoSimplePolygons();
  }

  // 清除连接和虚拟连接
  ClearJoins();
  ClearGhostJoins();
  // 返回执行状态
  return succeeded;
}
//------------------------------------------------------------------------------

// 设置边的环绕计数
void Clipper::SetWindingCount(TEdge &edge) {
  // 获取 AEL 中与 'edge' 相同多边形类型且紧邻 'edge' 的边
  TEdge *e = edge.PrevInAEL;
  while (e && ((e->PolyTyp != edge.PolyTyp) || (e->WindDelta == 0)))
    e = e->PrevInAEL;
  // 如果找不到符合条件的边
  if (!e) {
    // 如果边的 WindDelta 为 0
    if (edge.WindDelta == 0) {
      // 确定填充类型为主多边形还是剪切多边形
      PolyFillType pft =
          (edge.PolyTyp == ptSubject ? m_SubjFillType : m_ClipFillType);
      // 根据填充类型确定 WindCnt 的值
      edge.WindCnt = (pft == pftNegative ? -1 : 1);
    } else
      // 否则直接将 WindDelta 赋值给 WindCnt
      edge.WindCnt = edge.WindDelta;
    // 初始化 WindCnt2 为 0
    edge.WindCnt2 = 0;
    // 将当前边赋值给 e，准备计算 WindCnt2
    e = m_ActiveEdges; // ie get ready to calc WindCnt2
  } else if (edge.WindDelta == 0 && m_ClipType != ctUnion) {
    // 如果边的 WindDelta 为 0 且剪切类型不是并集
    edge.WindCnt = 1;
    // 将前一个边的 WindCnt2 赋值给当前边的 WindCnt2
    edge.WindCnt2 = e->WindCnt2;
    // 将下一个边赋值给 e，准备计算 WindCnt2
    e = e->NextInAEL; // ie get ready to calc WindCnt2
  } else if (IsEvenOddFillType(edge)) {
    // 如果是 EvenOdd 填充类型
    // EvenOdd 填充 ...
    if (edge.WindDelta == 0) {
      // 是否在主多边形内部
      bool Inside = true;
      TEdge *e2 = e->PrevInAEL;
      // 遍历前一个边
      while (e2) {
        if (e2->PolyTyp == e->PolyTyp && e2->WindDelta != 0)
          Inside = !Inside;
        e2 = e2->PrevInAEL;
      }
      // 根据 Inside 确定 WindCnt 的值
      edge.WindCnt = (Inside ? 0 : 1);
    } else {
      // 否则直接将 WindDelta 赋值给 WindCnt
      edge.WindCnt = edge.WindDelta;
    }
    // 将前一个边的 WindCnt2 赋值给当前边的 WindCnt2
    edge.WindCnt2 = e->WindCnt2;
    // 将下一个边赋值给 e，准备计算 WindCnt2
    e = e->NextInAEL; // ie get ready to calc WindCnt2
  } else {
    // 非零、正数或负数填充 ...
    if (e->WindCnt * e->WindDelta < 0) {
      // 前一个边的 WindCount (WC) 朝零减少
      // 所以我们在前一个多边形外部 ...
      if (Abs(e->WindCnt) > 1) {
        // 在前一个多边形外部但仍在另一个多边形内部
        // 当反转前一个多边形的方向时使用相同的 WC
        if (e->WindDelta * edge.WindDelta < 0)
          edge.WindCnt = e->WindCnt;
        // 否则继续减少 WC ...
        else
          edge.WindCnt = e->WindCnt + edge.WindDelta;
      } else
        // 现在在所有相同填充类型的多边形外部，设置自己的 WC ...
        edge.WindCnt = (edge.WindDelta == 0 ? 1 : edge.WindDelta);
    } else {
      // 如果前一个边是距离零点'increasing' WindCount (WC)的距离
      // 那么我们在前一个多边形内部...
      if (edge.WindDelta == 0)
        edge.WindCnt = (e->WindCnt < 0 ? e->WindCnt - 1 : e->WindCnt + 1);
      // 如果风向与前一个相反，则使用相同的WC
      else if (e->WindDelta * edge.WindDelta < 0)
        edge.WindCnt = e->WindCnt;
      // 否则添加到WC...
      else
        edge.WindCnt = e->WindCnt + edge.WindDelta;
    }
    edge.WindCnt2 = e->WindCnt2;
    e = e->NextInAEL; // 即准备计算WindCnt2

  }

  // 更新WindCnt2...
  if (IsEvenOddAltFillType(edge)) {
    // EvenOdd填充...
    while (e != &edge) {
      if (e->WindDelta != 0)
        edge.WindCnt2 = (edge.WindCnt2 == 0 ? 1 : 0);
      e = e->NextInAEL;
    }
  } else {
    // 非零、正数或负数填充...
    while (e != &edge) {
      edge.WindCnt2 += e->WindDelta;
      e = e->NextInAEL;
    }
  }
// 检查边的填充类型是否为 EvenOdd，根据边的 PolyTyp 返回对应的填充类型
bool Clipper::IsEvenOddFillType(const TEdge &edge) const {
  if (edge.PolyTyp == ptSubject)
    return m_SubjFillType == pftEvenOdd;
  else
    return m_ClipFillType == pftEvenOdd;
}
//------------------------------------------------------------------------------

// 检查边的填充类型是否为 EvenOdd，根据边的 PolyTyp 返回对应的填充类型
bool Clipper::IsEvenOddAltFillType(const TEdge &edge) const {
  if (edge.PolyTyp == ptSubject)
    return m_ClipFillType == pftEvenOdd;
  else
    return m_SubjFillType == pftEvenOdd;
}
//------------------------------------------------------------------------------

// 检查边是否为贡献边，根据边的 PolyTyp 和填充类型进行判断
bool Clipper::IsContributing(const TEdge &edge) const {
  PolyFillType pft, pft2;
  // 根据边的 PolyTyp 设置填充类型 pft 和 pft2
  if (edge.PolyTyp == ptSubject) {
    pft = m_SubjFillType;
    pft2 = m_ClipFillType;
  } else {
    pft = m_ClipFillType;
    pft2 = m_SubjFillType;
  }

  switch (pft) {
  case pftEvenOdd:
    // 如果 subj 线段被标记为在 subj 多边形内部，则返回 false
    if (edge.WindDelta == 0 && edge.WindCnt != 1)
      return false;
    break;
  case pftNonZero:
    if (Abs(edge.WindCnt) != 1)
      return false;
    break;
  case pftPositive:
    if (edge.WindCnt != 1)
      return false;
    break;
  default: // pftNegative
    if (edge.WindCnt != -1)
      return false;
  }

  switch (m_ClipType) {
  case ctIntersection:
    switch (pft2) {
    case pftEvenOdd:
    case pftNonZero:
      return (edge.WindCnt2 != 0);
    case pftPositive:
      return (edge.WindCnt2 > 0);
    default:
      return (edge.WindCnt2 < 0);
    }
    break;
  case ctUnion:
    switch (pft2) {
    case pftEvenOdd:
    case pftNonZero:
      return (edge.WindCnt2 == 0);
    case pftPositive:
      return (edge.WindCnt2 <= 0);
    default:
      return (edge.WindCnt2 >= 0);
    }
    break;
  case ctDifference:
    # 如果边的 PolyTyp 为 ptSubject
    if (edge.PolyTyp == ptSubject)
      # 根据 pft2 的不同取值进行判断
      switch (pft2) {
      # 如果 pft2 为 pftEvenOdd 或 pftNonZero
      case pftEvenOdd:
      case pftNonZero:
        # 返回边的 WindCnt2 是否为 0
        return (edge.WindCnt2 == 0);
      # 如果 pft2 为 pftPositive
      case pftPositive:
        # 返回边的 WindCnt2 是否小于等于 0
        return (edge.WindCnt2 <= 0);
      # 其他情况
      default:
        # 返回边的 WindCnt2 是否大于等于 0
        return (edge.WindCnt2 >= 0);
      }
    # 如果边的 PolyTyp 不为 ptSubject
    else
      # 根据 pft2 的不同取值进行判断
      switch (pft2) {
      # 如果 pft2 为 pftEvenOdd 或 pftNonZero
      case pftEvenOdd:
      case pftNonZero:
        # 返回边的 WindCnt2 是否不为 0
        return (edge.WindCnt2 != 0);
      # 如果 pft2 为 pftPositive
      case pftPositive:
        # 返回边的 WindCnt2 是否大于 0
        return (edge.WindCnt2 > 0);
      # 其他情况
      default:
        # 返回边的 WindCnt2 是否小于 0
        return (edge.WindCnt2 < 0);
      }
    # 结束当前 case
    break;
  # 如果 ct2 为 ctXor
  case ctXor:
    # 如果边的 WindDelta 为 0，则 XOr 总是有贡献，除非是开放的
    if (edge.WindDelta == 0) // XOr always contributing unless open
      # 根据 pft2 的不同取值进行判断
      switch (pft2) {
      # 如果 pft2 为 pftEvenOdd 或 pftNonZero
      case pftEvenOdd:
      case pftNonZero:
        # 返回边的 WindCnt2 是否为 0
        return (edge.WindCnt2 == 0);
      # 如果 pft2 为 pftPositive
      case pftPositive:
        # 返回边的 WindCnt2 是否小于等于 0
        return (edge.WindCnt2 <= 0);
      # 其他情况
      default:
        # 返回边的 WindCnt2 是否大于等于 0
        return (edge.WindCnt2 >= 0);
      }
    else
      # 返回 true
      return true;
    # 结束当前 case
    break;
  # 其他情况
  default:
    # 返回 true
    return true;
  }
// 添加局部最小多边形，根据给定的两个边和点
OutPt *Clipper::AddLocalMinPoly(TEdge *e1, TEdge *e2, const IntPoint &Pt) {
  OutPt *result; // 结果指针
  TEdge *e, *prevE; // 边和前一个边指针
  // 如果 e2 是水平的或者 e1 的 Dx 大于 e2 的 Dx
  if (IsHorizontal(*e2) || (e1->Dx > e2->Dx)) {
    result = AddOutPt(e1, Pt); // 添加 e1 的 OutPt
    e2->OutIdx = e1->OutIdx;
    e1->Side = esLeft;
    e2->Side = esRight;
    e = e1;
    // 如果 e 的前一个边在 AEL 中等于 e2，则 prevE 为 e2 的前一个边，否则为 e 的前一个边
    if (e->PrevInAEL == e2)
      prevE = e2->PrevInAEL;
    else
      prevE = e->PrevInAEL;
  } else {
    result = AddOutPt(e2, Pt); // 添加 e2 的 OutPt
    e1->OutIdx = e2->OutIdx;
    e1->Side = esRight;
    e2->Side = esLeft;
    e = e2;
    // 如果 e 的前一个边在 AEL 中等于 e1，则 prevE 为 e1 的前一个边，否则为 e 的前一个边
    if (e->PrevInAEL == e1)
      prevE = e1->PrevInAEL;
    else
      prevE = e->PrevInAEL;
  }

  // 如果 prevE 存在且 prevE 的 OutIdx 大于等于 0 且 prevE 的顶点 Y 坐标小于 Pt 的 Y 坐标 且 e 的顶点 Y 坐标小于 Pt 的 Y 坐标
  if (prevE && prevE->OutIdx >= 0 && prevE->Top.Y < Pt.Y && e->Top.Y < Pt.Y) {
    cInt xPrev = TopX(*prevE, Pt.Y); // 计算 prevE 在 Y 坐标为 Pt.Y 时的 X 坐标
    cInt xE = TopX(*e, Pt.Y); // 计算 e 在 Y 坐标为 Pt.Y 时的 X 坐标
    // 如果 xPrev 等于 xE 且 e 的 WindDelta 不为 0 且 prevE 的 WindDelta 不为 0 且两条线段斜率相等
    if (xPrev == xE && (e->WindDelta != 0) && (prevE->WindDelta != 0) &&
        SlopesEqual(IntPoint(xPrev, Pt.Y), prevE->Top, IntPoint(xE, Pt.Y),
                    e->Top, m_UseFullRange)) {
      OutPt *outPt = AddOutPt(prevE, Pt); // 添加 prevE 的 OutPt
      AddJoin(result, outPt, e->Top); // 添加连接点
    }
  }
  return result; // 返回结果指针
}
//------------------------------------------------------------------------------

// 添加局部最大多边形，根据给定的两个边和点
void Clipper::AddLocalMaxPoly(TEdge *e1, TEdge *e2, const IntPoint &Pt) {
  AddOutPt(e1, Pt); // 添加 e1 的 OutPt
  if (e2->WindDelta == 0)
    AddOutPt(e2, Pt); // 如果 e2 的 WindDelta 为 0，则添加 e2 的 OutPt
  // 如果 e1 的 OutIdx 等于 e2 的 OutIdx
  if (e1->OutIdx == e2->OutIdx) {
    e1->OutIdx = Unassigned; // 将 e1 的 OutIdx 设为未分配
    e2->OutIdx = Unassigned; // 将 e2 的 OutIdx 设为未分配
  } else if (e1->OutIdx < e2->OutIdx)
    AppendPolygon(e1, e2); // 将 e1 和 e2 连接成多边形
  else
    AppendPolygon(e2, e1); // 将 e2 和 e1 连接成多边形
}
//------------------------------------------------------------------------------

// 添加边到 SEL 中
void Clipper::AddEdgeToSEL(TEdge *edge) {
  // PEdge 中的 SEL 指针被重用来构建水平边的列表
  // 但是，对于水平边处理，我们不需要担心顺序
  if (!m_SortedEdges) {
    m_SortedEdges = edge; // 如果没有排序的边，则将当前边设为排序的边
    edge->PrevInSEL = 0; // 前一个边指针设为 0
    edge->NextInSEL = 0; // 后一个边指针设为 0
  } else {
    edge->NextInSEL = m_SortedEdges; // 否则，将当前边设为排序的边的下一个边
    # 将边的前一个边设置为0
    edge->PrevInSEL = 0;
    # 将排序后的边的前一个边设置为当前边
    m_SortedEdges->PrevInSEL = edge;
    # 将当前边设置为排序后的边
    m_SortedEdges = edge;
// 从排序边缘列表中弹出一条边缘
bool Clipper::PopEdgeFromSEL(TEdge *&edge) {
  // 如果排序边缘列表为空，则返回false
  if (!m_SortedEdges)
    return false;
  // 将排序边缘列表的第一条边缘赋值给edge
  edge = m_SortedEdges;
  // 从排序边缘列表中删除这条边缘
  DeleteFromSEL(m_SortedEdges);
  // 返回true
  return true;
}
//------------------------------------------------------------------------------

// 将活动边缘列表复制到排序边缘列表
void Clipper::CopyAELToSEL() {
  // 从活动边缘列表中获取第一条边缘
  TEdge *e = m_ActiveEdges;
  // 将第一条边缘赋值给排序边缘列表
  m_SortedEdges = e;
  // 遍历活动边缘列表，将每条边缘的前一个和后一个边缘赋值给排序边缘列表
  while (e) {
    e->PrevInSEL = e->PrevInAEL;
    e->NextInSEL = e->NextInAEL;
    e = e->NextInAEL;
  }
}
//------------------------------------------------------------------------------

// 添加连接点
void Clipper::AddJoin(OutPt *op1, OutPt *op2, const IntPoint OffPt) {
  // 创建一个连接对象
  Join *j = new Join;
  // 设置连接对象的属性
  j->OutPt1 = op1;
  j->OutPt2 = op2;
  j->OffPt = OffPt;
  // 将连接对象添加到连接列表中
  m_Joins.push_back(j);
}
//------------------------------------------------------------------------------

// 清空连接列表
void Clipper::ClearJoins() {
  // 遍历连接列表，删除每个连接对象
  for (JoinList::size_type i = 0; i < m_Joins.size(); i++)
    delete m_Joins[i];
  // 将连接列表大小重置为0
  m_Joins.resize(0);
}
//------------------------------------------------------------------------------

// 清空虚拟连接列表
void Clipper::ClearGhostJoins() {
  // 遍历虚拟连接列表，删除每个虚拟连接对象
  for (JoinList::size_type i = 0; i < m_GhostJoins.size(); i++)
    delete m_GhostJoins[i];
  // 将虚拟连接列表大小重置为0
  m_GhostJoins.resize(0);
}
//------------------------------------------------------------------------------

// 添加虚拟连接点
void Clipper::AddGhostJoin(OutPt *op, const IntPoint OffPt) {
  // 创建一个虚拟连接对象
  Join *j = new Join;
  // 设置虚拟连接对象的属性
  j->OutPt1 = op;
  j->OutPt2 = 0;
  j->OffPt = OffPt;
  // 将虚拟连接对象添加到虚拟连接列表中
  m_GhostJoins.push_back(j);
}
//------------------------------------------------------------------------------

// 将局部最小值插入到活动边缘列表
void Clipper::InsertLocalMinimaIntoAEL(const cInt botY) {
  // 定义局部最小值
  const LocalMinimum *lm;
  // 当存在局部最小值时循环
  while (PopLocalMinima(botY, lm)) {
    // 获取局部最小值的左边界和右边界
    TEdge *lb = lm->LeftBound;
    TEdge *rb = lm->RightBound;

    // 初始化OutPt指针为0
    OutPt *Op1 = 0;
    // 如果 lb 或者 rb 为空，则执行以下操作
    if (!lb || !rb) {
      // nb: 不将 LB 插入 AEL 或 SEL
      InsertEdgeIntoAEL(rb, 0);
      SetWindingCount(*rb);
      if (IsContributing(*rb))
        Op1 = AddOutPt(rb, rb->Bot);
      //} else if (!rb) {
      //  InsertEdgeIntoAEL(lb, 0);
      //  SetWindingCount(*lb);
      //  if (IsContributing(*lb))
      //    Op1 = AddOutPt(lb, lb->Bot);
      // 将 lb 的顶点 Y 坐标插入扫描线表
      InsertScanbeam(lb->Top.Y);
    } else {
      // 将 lb 插入 AEL
      InsertEdgeIntoAEL(lb, 0);
      // 将 rb 插入 AEL，并将 lb 作为前一个边
      InsertEdgeIntoAEL(rb, lb);
      SetWindingCount(*lb);
      // 设置 rb 的 WindCnt 为 lb 的 WindCnt
      rb->WindCnt = lb->WindCnt;
      rb->WindCnt2 = lb->WindCnt2;
      if (IsContributing(*lb))
        // 添加局部最小多边形
        Op1 = AddLocalMinPoly(lb, rb, lb->Bot);
      // 将 lb 的顶点 Y 坐标插入扫描线表
      InsertScanbeam(lb->Top.Y);
    }

    // 如果 rb 存在
    if (rb) {
      // 如果 rb 是水平边，则将其添加到 SEL
      if (IsHorizontal(*rb)) {
        AddEdgeToSEL(rb);
        // 如果 rb 的下一个边存在，则将其顶点 Y 坐标插入扫描线表
        if (rb->NextInLML)
          InsertScanbeam(rb->NextInLML->Top.Y);
      } else
        // 将 rb 的顶点 Y 坐标插入扫描线表
        InsertScanbeam(rb->Top.Y);
    }

    // 如果 lb 或者 rb 为空，则继续下一次循环
    if (!lb || !rb)
      continue;

    // 如果任何输出多边形共享一条边，则它们需要稍后连接...
    if (Op1 && IsHorizontal(*rb) && m_GhostJoins.size() > 0 &&
        (rb->WindDelta != 0)) {
      for (JoinList::size_type i = 0; i < m_GhostJoins.size(); ++i) {
        Join *jr = m_GhostJoins[i];
        // 如果水平的 Rb 和一个 'ghost' 水平边重叠，则将 'ghost' 连接转换为准备好的真实连接...
        if (HorzSegmentsOverlap(jr->OutPt1->Pt.X, jr->OffPt.X, rb->Bot.X,
                                rb->Top.X))
          AddJoin(jr->OutPt1, Op1, jr->OffPt);
      }
    }

    // 如果 lb 的 OutIdx 大于等于 0 并且 lb 的前一个边存在于 AEL 中
    // 并且 lb 的前一个边的当前 X 坐标等于 lb 的底部 X 坐标
    // 并且 lb 的前一个边的 OutIdx 大于等于 0
    // 并且 lb 的前一个边和 lb 的斜率相等
    // 并且 lb 和 lb 的前一个边的 WindDelta 不为 0
    if (lb->OutIdx >= 0 && lb->PrevInAEL &&
        lb->PrevInAEL->Curr.X == lb->Bot.X && lb->PrevInAEL->OutIdx >= 0 &&
        SlopesEqual(lb->PrevInAEL->Bot, lb->PrevInAEL->Top, lb->Curr, lb->Top,
                    m_UseFullRange) &&
        (lb->WindDelta != 0) && (lb->PrevInAEL->WindDelta != 0)) {
      // 添加 lb 的前一个边的底部点到输出点
      OutPt *Op2 = AddOutPt(lb->PrevInAEL, lb->Bot);
      // 添加连接点
      AddJoin(Op1, Op2, lb->Top);
    }
    # 如果左边边界 lb 的下一个边界不是右边边界 rb
    if (lb->NextInAEL != rb) {

      # 如果右边边界 rb 的 OutIdx 大于等于 0，且右边边界 rb 的前一个边界的 OutIdx 大于等于 0，
      # 且斜率相等，且 WindDelta 不为 0，且前一个边界的 WindDelta 不为 0
      if (rb->OutIdx >= 0 && rb->PrevInAEL->OutIdx >= 0 &&
          SlopesEqual(rb->PrevInAEL->Curr, rb->PrevInAEL->Top, rb->Curr,
                      rb->Top, m_UseFullRange) &&
          (rb->WindDelta != 0) && (rb->PrevInAEL->WindDelta != 0)) {
        # 在 rb 的前一个边界的 AEL 中添加一个新的 OutPt，连接 Op1 和 Op2，连接点为 rb 的 Top
        OutPt *Op2 = AddOutPt(rb->PrevInAEL, rb->Bot);
        AddJoin(Op1, Op2, rb->Top);
      }

      # 将 lb 的下一个边界赋值给 e
      TEdge *e = lb->NextInAEL;
      # 如果 e 存在
      if (e) {
        # 当 e 不等于 rb 时
        while (e != rb) {
          # 注意：为了计算绕组数等，IntersectEdges() 假设 param1 在交点上方且在 param2 右侧
          # ...
          # 在 rb 和 e 之间的交点处相交，这里的顺序很重要
          IntersectEdges(rb, e, lb->Curr);
          # 将 e 更新为下一个边界
          e = e->NextInAEL;
        }
      }
    }
  }
// 从排序边表中删除指定边
void Clipper::DeleteFromSEL(TEdge *e) {
  // 获取待删除边的前一条边和后一条边
  TEdge *SelPrev = e->PrevInSEL;
  TEdge *SelNext = e->NextInSEL;
  // 如果前一条边和后一条边都不存在，并且待删除边不是排序边表的第一条边，则直接返回，表示已经删除过了
  if (!SelPrev && !SelNext && (e != m_SortedEdges))
    return;
  // 更新前一条边的后一条边为待删除边的后一条边
  if (SelPrev)
    SelPrev->NextInSEL = SelNext;
  else
    m_SortedEdges = SelNext;
  // 更新后一条边的前一条边为待删除边的前一条边
  if (SelNext)
    SelNext->PrevInSEL = SelPrev;
  // 清空待删除边的前一条边和后一条边
  e->NextInSEL = 0;
  e->PrevInSEL = 0;
}
//------------------------------------------------------------------------------

#ifdef use_xyz
// 设置点的 Z 值
void Clipper::SetZ(IntPoint &pt, TEdge &e1, TEdge &e2) {
  // 如果点的 Z 值不为 0 或者不需要填充 Z 值，则直接返回
  if (pt.Z != 0 || !m_ZFill)
    return;
  // 根据点的位置设置 Z 值
  else if (pt == e1.Bot)
    pt.Z = e1.Bot.Z;
  else if (pt == e1.Top)
    pt.Z = e1.Top.Z;
  else if (pt == e2.Bot)
    pt.Z = e2.Bot.Z;
  else if (pt == e2.Top)
    pt.Z = e2.Top.Z;
  else
    (*m_ZFill)(e1.Bot, e1.Top, e2.Bot, e2.Top, pt);
}
//------------------------------------------------------------------------------
#endif

// 计算两条边的交点
void Clipper::IntersectEdges(TEdge *e1, TEdge *e2, IntPoint &Pt) {
  // 判断两条边是否为贡献边
  bool e1Contributing = (e1->OutIdx >= 0);
  bool e2Contributing = (e2->OutIdx >= 0);

#ifdef use_xyz
  // 设置交点的 Z 值
  SetZ(Pt, *e1, *e2);
#endif

#ifdef use_lines
  // 如果其中一条边在开放路径上 ...
  if (e1->WindDelta == 0 || e2->WindDelta == 0) {
    // 忽略主-主开放路径的交点，除非它们都是开放路径，并且都是 '贡献极大值' ...
    if (e1->WindDelta == 0 && e2->WindDelta == 0)
      return;

    // 如果交点是主线与主多边形的交点 ...
    else if (e1->PolyTyp == e2->PolyTyp && e1->WindDelta != e2->WindDelta &&
             m_ClipType == ctUnion) {
      if (e1->WindDelta == 0) {
        if (e2Contributing) {
          AddOutPt(e1, Pt);
          if (e1Contributing)
            e1->OutIdx = Unassigned;
        }
      } else {
        if (e1Contributing) {
          AddOutPt(e2, Pt);
          if (e2Contributing)
            e2->OutIdx = Unassigned;
        }
      }
    }
  }
    } else if (e1->PolyTyp != e2->PolyTyp) {
      // 如果两个边的多边形类型不同，则执行以下操作
      // 当 clip.WndCnt 的绝对值为 1 时，切换 subj 开放路径 OutIdx 的开关...
      if ((e1->WindDelta == 0) && abs(e2->WindCnt) == 1 &&
          (m_ClipType != ctUnion || e2->WindCnt2 == 0)) {
        // 如果 e1 的 WindDelta 为 0，e2 的 WindCnt 的绝对值为 1，且满足条件，则执行以下操作
        // 向 e1 添加点 Pt
        AddOutPt(e1, Pt);
        // 如果 e1Contributing 为真，则将 e1 的 OutIdx 设置为 Unassigned
        if (e1Contributing)
          e1->OutIdx = Unassigned;
      } else if ((e2->WindDelta == 0) && (abs(e1->WindCnt) == 1) &&
                 (m_ClipType != ctUnion || e1->WindCnt2 == 0)) {
        // 如果 e2 的 WindDelta 为 0，e1 的 WindCnt 的绝对值为 1，且满足条件，则执行以下操作
        // 向 e2 添加点 Pt
        AddOutPt(e2, Pt);
        // 如果 e2Contributing 为真，则将 e2 的 OutIdx 设置为 Unassigned
        if (e2Contributing)
          e2->OutIdx = Unassigned;
      }
    }
    // 返回
    return;
  }
#endif

  // 更新边的绕数
  // 假设 e1 在交点上方且在 e2 的右侧
  if (e1->PolyTyp == e2->PolyTyp) {
    // 如果填充类型为 EvenOdd
    if (IsEvenOddFillType(*e1)) {
      // 交换 e1 和 e2 的绕数
      int oldE1WindCnt = e1->WindCnt;
      e1->WindCnt = e2->WindCnt;
      e2->WindCnt = oldE1WindCnt;
    } else {
      // 更新绕数
      if (e1->WindCnt + e2->WindDelta == 0)
        e1->WindCnt = -e1->WindCnt;
      else
        e1->WindCnt += e2->WindDelta;
      if (e2->WindCnt - e1->WindDelta == 0)
        e2->WindCnt = -e2->WindCnt;
      else
        e2->WindCnt -= e1->WindDelta;
    }
  } else {
    // 更新第二个绕数
    if (!IsEvenOddFillType(*e2))
      e1->WindCnt2 += e2->WindDelta;
    else
      e1->WindCnt2 = (e1->WindCnt2 == 0) ? 1 : 0;
    // 更新第一个绕数
    if (!IsEvenOddFillType(*e1))
      e2->WindCnt2 -= e1->WindDelta;
    else
      e2->WindCnt2 = (e2->WindCnt2 == 0) ? 1 : 0;
  }

  // 获取填充类型
  PolyFillType e1FillType, e2FillType, e1FillType2, e2FillType2;
  if (e1->PolyTyp == ptSubject) {
    e1FillType = m_SubjFillType;
    e1FillType2 = m_ClipFillType;
  } else {
    e1FillType = m_ClipFillType;
    e1FillType2 = m_SubjFillType;
  }
  if (e2->PolyTyp == ptSubject) {
    e2FillType = m_SubjFillType;
    e2FillType2 = m_ClipFillType;
  } else {
    e2FillType = m_ClipFillType;
    e2FillType2 = m_SubjFillType;
  }

  // 获取绕数
  cInt e1Wc, e2Wc;
  switch (e1FillType) {
  case pftPositive:
    e1Wc = e1->WindCnt;
    break;
  case pftNegative:
    e1Wc = -e1->WindCnt;
    break;
  default:
    e1Wc = Abs(e1->WindCnt);
  }
  switch (e2FillType) {
  case pftPositive:
    e2Wc = e2->WindCnt;
    break;
  case pftNegative:
    e2Wc = -e2->WindCnt;
    break;
  default:
    e2Wc = Abs(e2->WindCnt);
  }

  // 如果两个边都是有效的
  if (e1Contributing && e2Contributing) {
    // 如果满足条件，则添加局部最大多边形
    if ((e1Wc != 0 && e1Wc != 1) || (e2Wc != 0 && e2Wc != 1) ||
        (e1->PolyTyp != e2->PolyTyp && m_ClipType != ctXor)) {
      AddLocalMaxPoly(e1, e2, Pt);
    } else {
      // 否则添加输出点
      AddOutPt(e1, Pt);
      AddOutPt(e2, Pt);
      SwapSides(*e1, *e2);
      SwapPolyIndexes(*e1, *e2);
    // 如果 e1 正在贡献
    } else if (e1Contributing) {
        // 如果 e2 的 winding count 为 0 或 1
        if (e2Wc == 0 || e2Wc == 1) {
            // 添加输出点
            AddOutPt(e1, Pt);
            // 交换边的两侧
            SwapSides(*e1, *e2);
            // 交换多边形索引
            SwapPolyIndexes(*e1, *e2);
        }
    } else if (e2Contributing) {
        // 如果 e2 正在贡献
        if (e1Wc == 0 || e1Wc == 1) {
            // 添加输出点
            AddOutPt(e2, Pt);
            // 交换边的两侧
            SwapSides(*e1, *e2);
            // 交换多边形索引
            SwapPolyIndexes(*e1, *e2);
        }
    } else if ((e1Wc == 0 || e1Wc == 1) && (e2Wc == 0 || e2Wc == 1)) {
        // 如果两条边都没有贡献...

        cInt e1Wc2, e2Wc2;
        switch (e1FillType2) {
            case pftPositive:
                e1Wc2 = e1->WindCnt2;
                break;
            case pftNegative:
                e1Wc2 = -e1->WindCnt2;
                break;
            default:
                e1Wc2 = Abs(e1->WindCnt2);
        }
        switch (e2FillType2) {
            case pftPositive:
                e2Wc2 = e2->WindCnt2;
                break;
            case pftNegative:
                e2Wc2 = -e2->WindCnt2;
                break;
            default:
                e2Wc2 = Abs(e2->WindCnt2);
        }

        if (e1->PolyTyp != e2->PolyTyp) {
            // 添加局部最小多边形
            AddLocalMinPoly(e1, e2, Pt);
        } else if (e1Wc == 1 && e2Wc == 1)
            switch (m_ClipType) {
                case ctIntersection:
                    if (e1Wc2 > 0 && e2Wc2 > 0)
                        // 添加局部最小多边形
                        AddLocalMinPoly(e1, e2, Pt);
                    break;
                case ctUnion:
                    if (e1Wc2 <= 0 && e2Wc2 <= 0)
                        // 添加局部最小多边形
                        AddLocalMinPoly(e1, e2, Pt);
                    break;
                case ctDifference:
                    if (((e1->PolyTyp == ptClip) && (e1Wc2 > 0) && (e2Wc2 > 0)) ||
                        ((e1->PolyTyp == ptSubject) && (e1Wc2 <= 0) && (e2Wc2 <= 0)))
                        // 添加局部最小多边形
                        AddLocalMinPoly(e1, e2, Pt);
                    break;
                case ctXor:
                    // 添加局部最小多边形
                    AddLocalMinPoly(e1, e2, Pt);
            }
        else
            // 交换边的两侧
            SwapSides(*e1, *e2);
    }
// 设置边的孔状态
void Clipper::SetHoleState(TEdge *e, OutRec *outrec) {
  // 获取前一个边
  TEdge *e2 = e->PrevInAEL;
  // 临时边
  TEdge *eTmp = 0;
  // 遍历前一个边
  while (e2) {
    // 如果前一个边的输出索引大于等于0且风向增量不为0
    if (e2->OutIdx >= 0 && e2->WindDelta != 0) {
      // 如果临时边为空
      if (!eTmp)
        eTmp = e2;
      // 如果临时边的输出索引等于前一个边的输出索引
      else if (eTmp->OutIdx == e2->OutIdx)
        eTmp = 0;
    }
    // 获取前一个边的前一个边
    e2 = e2->PrevInAEL;
  }
  // 如果临时边为空
  if (!eTmp) {
    // 设置输出记录的第一个左边为空
    outrec->FirstLeft = 0;
    // 设置输出记录为非孔
    outrec->IsHole = false;
  } else {
    // 设置输出记录的第一个左边为临时边对应的多边形输出记录
    outrec->FirstLeft = m_PolyOuts[eTmp->OutIdx];
    // 设置输出记录为第一个左边的非孔状态
    outrec->IsHole = !outrec->FirstLeft->IsHole;
  }
}
//------------------------------------------------------------------------------

// 获取最下面的输出记录
OutRec *GetLowermostRec(OutRec *outRec1, OutRec *outRec2) {
  // 确定哪个多边形片段具有正确的孔状态
  if (!outRec1->BottomPt)
    outRec1->BottomPt = GetBottomPt(outRec1->Pts);
  if (!outRec2->BottomPt)
    outRec2->BottomPt = GetBottomPt(outRec2->Pts);
  OutPt *OutPt1 = outRec1->BottomPt;
  OutPt *OutPt2 = outRec2->BottomPt;
  if (OutPt1->Pt.Y > OutPt2->Pt.Y)
    return outRec1;
  else if (OutPt1->Pt.Y < OutPt2->Pt.Y)
    return outRec2;
  else if (OutPt1->Pt.X < OutPt2->Pt.X)
    return outRec1;
  else if (OutPt1->Pt.X > OutPt2->Pt.X)
    return outRec2;
  else if (OutPt1->Next == OutPt1)
    return outRec2;
  else if (OutPt2->Next == OutPt2)
    return outRec1;
  else if (FirstIsBottomPt(OutPt1, OutPt2))
    return outRec1;
  else
    return outRec2;
}
//------------------------------------------------------------------------------

// 判断输出记录1是否在输出记录2的右边
bool OutRec1RightOfOutRec2(OutRec *outRec1, OutRec *outRec2) {
  do {
    outRec1 = outRec1->FirstLeft;
    if (outRec1 == outRec2)
      return true;
  } while (outRec1);
  return false;
}
//------------------------------------------------------------------------------

// 获取输出记录
OutRec *Clipper::GetOutRec(int Idx) {
  OutRec *outrec = m_PolyOuts[Idx];
  // 循环直到输出记录等于其索引对应的输出记录
  while (outrec != m_PolyOuts[outrec->Idx])
    outrec = m_PolyOuts[outrec->Idx];
  return outrec;
}
// 将两个边界相交的多边形连接起来
void Clipper::AppendPolygon(TEdge *e1, TEdge *e2) {
  // 获取两个输出多边形的起始和结束点
  OutRec *outRec1 = m_PolyOuts[e1->OutIdx];
  OutRec *outRec2 = m_PolyOuts[e2->OutIdx];

  OutRec *holeStateRec;
  // 确定哪个多边形在另一个多边形的右侧
  if (OutRec1RightOfOutRec2(outRec1, outRec2))
    holeStateRec = outRec2;
  else if (OutRec1RightOfOutRec2(outRec2, outRec1))
    holeStateRec = outRec1;
  else
    holeStateRec = GetLowermostRec(outRec1, outRec2);

  // 获取两个输出多边形的起始和结束点，并将e2多边形连接到e1多边形上，并删除对e2的指针
  OutPt *p1_lft = outRec1->Pts;
  OutPt *p1_rt = p1_lft->Prev;
  OutPt *p2_lft = outRec2->Pts;
  OutPt *p2_rt = p2_lft->Prev;

  // 将e2多边形连接到e1多边形上，并删除对e2的指针
  if (e1->Side == esLeft) {
    if (e2->Side == esLeft) {
      // z y x a b c
      ReversePolyPtLinks(p2_lft);
      p2_lft->Next = p1_lft;
      p1_lft->Prev = p2_lft;
      p1_rt->Next = p2_rt;
      p2_rt->Prev = p1_rt;
      outRec1->Pts = p2_rt;
    } else {
      // x y z a b c
      p2_rt->Next = p1_lft;
      p1_lft->Prev = p2_rt;
      p2_lft->Prev = p1_rt;
      p1_rt->Next = p2_lft;
      outRec1->Pts = p2_lft;
    }
  } else {
    if (e2->Side == esRight) {
      // a b c z y x
      ReversePolyPtLinks(p2_lft);
      p1_rt->Next = p2_rt;
      p2_rt->Prev = p1_rt;
      p2_lft->Next = p1_lft;
      p1_lft->Prev = p2_lft;
    } else {
      // a b c x y z
      p1_rt->Next = p2_lft;
      p2_lft->Prev = p1_rt;
      p1_lft->Prev = p2_rt;
      p2_rt->Next = p1_lft;
    }
  }

  outRec1->BottomPt = 0;
  // 如果holeStateRec等于outRec2，则更新outRec1的FirstLeft
  if (holeStateRec == outRec2) {
    if (outRec2->FirstLeft != outRec1)
      outRec1->FirstLeft = outRec2->FirstLeft;
    // 将 outRec2 的 IsHole 属性设置为 outRec1 的 IsHole 属性
    outRec1->IsHole = outRec2->IsHole;
  }
  // 将 outRec2 的 Pts 属性设置为 0
  outRec2->Pts = 0;
  // 将 outRec2 的 BottomPt 属性设置为 0
  outRec2->BottomPt = 0;
  // 将 outRec2 的 FirstLeft 属性设置为 outRec1
  outRec2->FirstLeft = outRec1;

  // 获取 e1 和 e2 的 OutIdx 属性值
  int OKIdx = e1->OutIdx;
  int ObsoleteIdx = e2->OutIdx;

  // 将 e1 和 e2 的 OutIdx 属性设置为 Unassigned
  e1->OutIdx = Unassigned; // nb: safe because we only get here via AddLocalMaxPoly
  e2->OutIdx = Unassigned;

  // 遍历活动边缘列表中的边
  TEdge *e = m_ActiveEdges;
  while (e) {
    // 如果当前边的 OutIdx 等于 ObsoleteIdx
    if (e->OutIdx == ObsoleteIdx) {
      // 将当前边的 OutIdx 设置为 OKIdx
      e->OutIdx = OKIdx;
      // 将当前边的 Side 属性设置为 e1 的 Side 属性
      e->Side = e1->Side;
      // 退出循环
      break;
    }
    // 移动到下一个边
    e = e->NextInAEL;
  }

  // 将 outRec2 的 Idx 属性设置为 outRec1 的 Idx 属性
  outRec2->Idx = outRec1->Idx;
// 添加一个新的 OutPt 到指定的 TEdge 上
OutPt *Clipper::AddOutPt(TEdge *e, const IntPoint &pt) {
  // 如果 TEdge 的 OutIdx 小于 0，表示还没有关联 OutRec
  if (e->OutIdx < 0) {
    // 创建一个新的 OutRec
    OutRec *outRec = CreateOutRec();
    // 设置 OutRec 的 IsOpen 属性
    outRec->IsOpen = (e->WindDelta == 0);
    // 创建一个新的 OutPt
    OutPt *newOp = new OutPt;
    // 设置 OutPt 的 Idx 和 Pt 属性
    newOp->Idx = outRec->Idx;
    newOp->Pt = pt;
    newOp->Next = newOp;
    newOp->Prev = newOp;
    // 如果 OutRec 不是开放的，则设置其为孔洞状态
    if (!outRec->IsOpen)
      SetHoleState(e, outRec);
    // 将 TEdge 的 OutIdx 设置为 OutRec 的 Idx
    e->OutIdx = outRec->Idx;
    return newOp;
  } else {
    // 获取与 TEdge 关联的 OutRec
    OutRec *outRec = m_PolyOuts[e->OutIdx];
    // OutRec.Pts 是最左边的点，OutRec.Pts.Prev 是最右边的点
    OutPt *op = outRec->Pts;

    // 判断是否需要将新的 OutPt 添加到最前面
    bool ToFront = (e->Side == esLeft);
    if (ToFront && (pt == op->Pt))
      return op;
    else if (!ToFront && (pt == op->Prev->Pt))
      return op->Prev;

    // 创建一个新的 OutPt，并设置其属性
    OutPt *newOp = new OutPt;
    newOp->Idx = outRec->Idx;
    newOp->Pt = pt;
    newOp->Next = op;
    newOp->Prev = op->Prev;
    newOp->Prev->Next = newOp;
    op->Prev = newOp;
    // 如果需要将新的 OutPt 添加到最前面，则更新 OutRec.Pts
    if (ToFront)
      outRec->Pts = newOp;
    return newOp;
  }
}
//------------------------------------------------------------------------------

// 获取指定 TEdge 的最后一个 OutPt
OutPt *Clipper::GetLastOutPt(TEdge *e) {
  // 获取与 TEdge 关联的 OutRec
  OutRec *outRec = m_PolyOuts[e->OutIdx];
  // 如果 TEdge 的 Side 是 esLeft，则返回 OutRec.Pts，否则返回 OutRec.Pts.Prev
  if (e->Side == esLeft)
    return outRec->Pts;
  else
    return outRec->Pts->Prev;
}
//------------------------------------------------------------------------------

// 处理水平边
void Clipper::ProcessHorizontals() {
  TEdge *horzEdge;
  // 从 SEL 中弹出水平边，然后处理水平边
  while (PopEdgeFromSEL(horzEdge))
    ProcessHorizontal(horzEdge);
}
//------------------------------------------------------------------------------

// 判断是否为极小值点
inline bool IsMinima(TEdge *e) {
  // 判断条件：TEdge 存在，且前一个边的下一个点不是当前边，后一个边的下一个点也不是当前边
  return e && (e->Prev->NextInLML != e) && (e->Next->NextInLML != e);
}
//------------------------------------------------------------------------------

// 判断是否为极大值点
inline bool IsMaxima(TEdge *e, const cInt Y) {
  // 判断条件：TEdge 存在，且顶部的 Y 坐标等于指定的 Y，且没有下一个边
  return e && e->Top.Y == Y && !e->NextInLML;
}
//------------------------------------------------------------------------------
// 检查边 e 的顶点是否为 Y，并且下一个边在左边最大链表中
inline bool IsIntermediate(TEdge *e, const cInt Y) {
  return e->Top.Y == Y && e->NextInLML;
}
//------------------------------------------------------------------------------

// 获取与边 e 的最大值对应的边
TEdge *GetMaximaPair(TEdge *e) {
  if ((e->Next->Top == e->Top) && !e->Next->NextInLML)
    return e->Next;
  else if ((e->Prev->Top == e->Top) && !e->Prev->NextInLML)
    return e->Prev;
  else
    return 0;
}
//------------------------------------------------------------------------------

// 类似于 GetMaximaPair()，但如果 MaxPair 不在 AEL 中则返回 0（除非是水平的）
TEdge *GetMaximaPairEx(TEdge *e) {
  TEdge *result = GetMaximaPair(e);
  if (result &&
      (result->OutIdx == Skip ||
       (result->NextInAEL == result->PrevInAEL && !IsHorizontal(*result))))
    return 0;
  return result;
}
//------------------------------------------------------------------------------

// 交换 SEL 中两个边的位置
void Clipper::SwapPositionsInSEL(TEdge *Edge1, TEdge *Edge2) {
  if (!(Edge1->NextInSEL) && !(Edge1->PrevInSEL))
    return;
  if (!(Edge2->NextInSEL) && !(Edge2->PrevInSEL))
    return;

  if (Edge1->NextInSEL == Edge2) {
    TEdge *Next = Edge2->NextInSEL;
    if (Next)
      Next->PrevInSEL = Edge1;
    TEdge *Prev = Edge1->PrevInSEL;
    if (Prev)
      Prev->NextInSEL = Edge2;
    Edge2->PrevInSEL = Prev;
    Edge2->NextInSEL = Edge1;
    Edge1->PrevInSEL = Edge2;
    Edge1->NextInSEL = Next;
  } else if (Edge2->NextInSEL == Edge1) {
    TEdge *Next = Edge1->NextInSEL;
    if (Next)
      Next->PrevInSEL = Edge2;
    TEdge *Prev = Edge2->PrevInSEL;
    if (Prev)
      Prev->NextInSEL = Edge1;
    Edge1->PrevInSEL = Prev;
    Edge1->NextInSEL = Edge2;
    Edge2->PrevInSEL = Edge1;
    Edge2->NextInSEL = Next;
  } else {
    TEdge *Next = Edge1->NextInSEL;
    TEdge *Prev = Edge1->PrevInSEL;
    Edge1->NextInSEL = Edge2->NextInSEL;
    if (Edge1->NextInSEL)
      Edge1->NextInSEL->PrevInSEL = Edge1;
    Edge1->PrevInSEL = Edge2->PrevInSEL;
    // 如果 Edge1 的前一个边存在，则将 Edge1 设置为前一个边的下一个边
    if (Edge1->PrevInSEL)
      Edge1->PrevInSEL->NextInSEL = Edge1;
    // 将 Edge2 设置为 Next 边
    Edge2->NextInSEL = Next;
    // 如果 Edge2 的下一个边存在，则将 Edge2 设置为下一个边的前一个边
    if (Edge2->NextInSEL)
      Edge2->NextInSEL->PrevInSEL = Edge2;
    // 将 Edge2 设置为 Prev 边
    Edge2->PrevInSEL = Prev;
    // 如果 Edge2 的前一个边存在，则将 Edge2 设置为前一个边的下一个边
    if (Edge2->PrevInSEL)
      Edge2->PrevInSEL->NextInSEL = Edge2;
  }

  // 如果 Edge1 的前一个边不存在，则将 Edge1 设置为 m_SortedEdges
  if (!Edge1->PrevInSEL)
    m_SortedEdges = Edge1;
  // 如果 Edge2 的前一个边不存在，则将 Edge2 设置为 m_SortedEdges
  else if (!Edge2->PrevInSEL)
    m_SortedEdges = Edge2;
// 获取当前边在 AEL 中的下一个边
TEdge *GetNextInAEL(TEdge *e, Direction dir) {
  return dir == dLeftToRight ? e->NextInAEL : e->PrevInAEL;
}
//------------------------------------------------------------------------------

// 获取水平边的方向、左右端点坐标
void GetHorzDirection(TEdge &HorzEdge, Direction &Dir, cInt &Left,
                      cInt &Right) {
  if (HorzEdge.Bot.X < HorzEdge.Top.X) {
    Left = HorzEdge.Bot.X;
    Right = HorzEdge.Top.X;
    Dir = dLeftToRight;
  } else {
    Left = HorzEdge.Top.X;
    Right = HorzEdge.Bot.X;
    Dir = dRightToLeft;
  }
}
//------------------------------------------------------------------------

/*******************************************************************************
* Notes: Horizontal edges (HEs) at scanline intersections (ie at the Top or    *
* Bottom of a scanbeam) are processed as if layered. The order in which HEs    *
* are processed doesn't matter. HEs intersect with other HE Bot.Xs only [#]    *
* (or they could intersect with Top.Xs only, ie EITHER Bot.Xs OR Top.Xs),      *
* and with other non-horizontal edges [*]. Once these intersections are        *
* processed, intermediate HEs then 'promote' the Edge above (NextInLML) into   *
* the AEL. These 'promoted' edges may in turn intersect [%] with other HEs.    *
*******************************************************************************/

// 处理水平边
void Clipper::ProcessHorizontal(TEdge *horzEdge) {
  Direction dir;
  cInt horzLeft, horzRight;
  bool IsOpen = (horzEdge->WindDelta == 0);

  // 获取水平边的方向、左右端点坐标
  GetHorzDirection(*horzEdge, dir, horzLeft, horzRight);

  TEdge *eLastHorz = horzEdge, *eMaxPair = 0;
  while (eLastHorz->NextInLML && IsHorizontal(*eLastHorz->NextInLML))
    eLastHorz = eLastHorz->NextInLML;
  if (!eLastHorz->NextInLML)
    eMaxPair = GetMaximaPair(eLastHorz);

  MaximaList::const_iterator maxIt;
  MaximaList::const_reverse_iterator maxRit;
  if (m_Maxima.size() > 0) {
    // 获取范围内的第一个极大值点 (X) ...
    // 如果方向为从左到右
    if (dir == dLeftToRight) {
      // 初始化最大值迭代器为最大值列表的起始位置
      maxIt = m_Maxima.begin();
      // 遍历最大值列表，直到找到大于当前水平边底部 X 坐标的最大值
      while (maxIt != m_Maxima.end() && *maxIt <= horzEdge->Bot.X)
        ++maxIt;
      // 如果找到了大于当前水平边顶部 X 坐标的最大值，则将最大值迭代器指向最大值列表的末尾
      if (maxIt != m_Maxima.end() && *maxIt >= eLastHorz->Top.X)
        maxIt = m_Maxima.end();
    } else {
      // 初始化最大值逆向迭代器为最大值列表的末尾位置
      maxRit = m_Maxima.rbegin();
      // 遍历最大值列表，直到找到小于当前水平边底部 X 坐标的最大值
      while (maxRit != m_Maxima.rend() && *maxRit > horzEdge->Bot.X)
        ++maxRit;
      // 如果找到了小于当前水平边顶部 X 坐标的最大值，则将最大值逆向迭代器指向最大值列表的末尾
      if (maxRit != m_Maxima.rend() && *maxRit <= eLastHorz->Top.X)
        maxRit = m_Maxima.rend();
    }
  }

  // 初始化输出点指针 op1
  OutPt *op1 = 0;

  // 无限循环，遍历连续的水平边
  for (;;) // loop through consec. horizontal edges
  {

    // 判断当前水平边是否为最后一个水平边
    bool IsLastHorz = (horzEdge == eLastHorz);
    // 获取下一个在活动边表中的边
    TEdge *e = GetNextInAEL(horzEdge, dir);
    while (e) {

      // 如果最大值列表中有值
      if (m_Maxima.size() > 0) {
        // 如果方向为从左到右
        if (dir == dLeftToRight) {
          // 遍历最大值列表，直到找到小于当前边的 X 坐标的最大值
          while (maxIt != m_Maxima.end() && *maxIt < e->Curr.X) {
            // 如果当前水平边有输出索引且不是开放的
            if (horzEdge->OutIdx >= 0 && !IsOpen)
              // 在当前水平边上添加输出点
              AddOutPt(horzEdge, IntPoint(*maxIt, horzEdge->Bot.Y));
            ++maxIt;
          }
        } else {
          // 如果方向为从右到左
          while (maxRit != m_Maxima.rend() && *maxRit > e->Curr.X) {
            // 如果当前水平边有输出索引且不是开放的
            if (horzEdge->OutIdx >= 0 && !IsOpen)
              // 在当前水平边上添加输出点
              AddOutPt(horzEdge, IntPoint(*maxRit, horzEdge->Bot.Y));
            ++maxRit;
          }
        }
      };

      // 如果当前边的 X 坐标超出了水平边的右边界（从左到右）或左边界（从右到左），则跳出循环
      if ((dir == dLeftToRight && e->Curr.X > horzRight) ||
          (dir == dRightToLeft && e->Curr.X < horzLeft))
        break;

      // 如果当前边的 X 坐标等于水平边的顶部 X 坐标，并且水平边的下一个边存在且当前边的 Dx 小于下一个边的 Dx，则跳出循环
      if (e->Curr.X == horzEdge->Top.X && horzEdge->NextInLML &&
          e->Dx < horzEdge->NextInLML->Dx)
        break;

      // 如果当前水平边有输出索引且不是开放的
      if (horzEdge->OutIdx >= 0 && !IsOpen) // note: may be done multiple times
      {
#ifdef use_xyz
        // 如果定义了 use_xyz 宏
        if (dir == dLeftToRight)
          // 如果方向是从左到右，则设置当前边的 Z 值
          SetZ(e->Curr, *horzEdge, *e);
        else
          // 否则设置当前边的 Z 值
          SetZ(e->Curr, *e, *horzEdge);
#endif
        // 添加水平边的当前点到输出点
        op1 = AddOutPt(horzEdge, e->Curr);
        // 获取下一个水平边
        TEdge *eNextHorz = m_SortedEdges;
        // 遍历下一个水平边
        while (eNextHorz) {
          // 如果下一个水平边的输出索引大于等于 0 并且水平段重叠
          if (eNextHorz->OutIdx >= 0 &&
              HorzSegmentsOverlap(horzEdge->Bot.X, horzEdge->Top.X,
                                  eNextHorz->Bot.X, eNextHorz->Top.X)) {
            // 获取下一个水平边的最后一个输出点
            OutPt *op2 = GetLastOutPt(eNextHorz);
            // 添加连接点
            AddJoin(op2, op1, eNextHorz->Top);
          }
          // 获取下一个水平边
          eNextHorz = eNextHorz->NextInSEL;
        }
        // 添加虚拟连接点
        AddGhostJoin(op1, horzEdge->Bot);
      }

      // 确保我们仍然在水平边的范围内，同时匹配 eMaxPair 时，确保我们在连续水平边的最后
      if (e == eMaxPair && IsLastHorz) {
        // 如果当前边是 eMaxPair 并且是最后一个水平边
        if (horzEdge->OutIdx >= 0)
          // 添加局部最大多边形
          AddLocalMaxPoly(horzEdge, eMaxPair, horzEdge->Top);
        // 从活动边表中删除水平边和 eMaxPair
        DeleteFromAEL(horzEdge);
        DeleteFromAEL(eMaxPair);
        return;
      }

      if (dir == dLeftToRight) {
        // 如果方向是从左到右
        IntPoint Pt = IntPoint(e->Curr.X, horzEdge->Curr.Y);
        // 求交点
        IntersectEdges(horzEdge, e, Pt);
      } else {
        // 否则
        IntPoint Pt = IntPoint(e->Curr.X, horzEdge->Curr.Y);
        // 求交点
        IntersectEdges(e, horzEdge, Pt);
      }
      // 获取下一个活动边
      TEdge *eNext = GetNextInAEL(e, dir);
      // 交换活动边的位置
      SwapPositionsInAEL(horzEdge, e);
      // 更新当前边为下一个边
      e = eNext;
    } // 结束 while(e)

    // 如果水平边的下一个不是水平边，跳出循环
    if (!horzEdge->NextInLML || !IsHorizontal(*horzEdge->NextInLML))
      break;

    // 更新水平边到活动边表
    UpdateEdgeIntoAEL(horzEdge);
    // 如果水平边的输出索引大于等于 0，则添加输出点
    if (horzEdge->OutIdx >= 0)
      AddOutPt(horzEdge, horzEdge->Bot);
    // 获取水平边的方向
    GetHorzDirection(*horzEdge, dir, horzLeft, horzRight);

  } // 结束 for (;;)

  // 如果水平边的输出索引大于等于 0 并且 op1 不存在
  if (horzEdge->OutIdx >= 0 && !op1) {
    // 获取水平边的最后一个输出点
    op1 = GetLastOutPt(horzEdge);
    // 获取下一个水平边
    TEdge *eNextHorz = m_SortedEdges;
    // 当前水平边的下一个水平边存在时执行循环
    while (eNextHorz) {
      // 如果下一个水平边的 OutIdx 大于等于 0 并且水平边与下一个水平边有重叠
      if (eNextHorz->OutIdx >= 0 &&
          HorzSegmentsOverlap(horzEdge->Bot.X, horzEdge->Top.X,
                              eNextHorz->Bot.X, eNextHorz->Top.X)) {
        // 获取下一个水平边的最后一个输出点
        OutPt *op2 = GetLastOutPt(eNextHorz);
        // 添加连接点
        AddJoin(op2, op1, eNextHorz->Top);
      }
      // 移动到下一个水平边
      eNextHorz = eNextHorz->NextInSEL;
    }
    // 添加虚拟连接点
    AddGhostJoin(op1, horzEdge->Top);
  }

  // 如果水平边有下一个边
  if (horzEdge->NextInLML) {
    // 如果水平边的 OutIdx 大于等于 0
    if (horzEdge->OutIdx >= 0) {
      // 添加输出点
      op1 = AddOutPt(horzEdge, horzEdge->Top);
      // 更新边到活动边表
      UpdateEdgeIntoAEL(horzEdge);
      // 如果水平边的 WindDelta 为 0，则返回
      if (horzEdge->WindDelta == 0)
        return;
      // 注意：此处水平边不再是水平的
      TEdge *ePrev = horzEdge->PrevInAEL;
      TEdge *eNext = horzEdge->NextInAEL;
      // 如果前一个边存在且满足一定条件
      if (ePrev && ePrev->Curr.X == horzEdge->Bot.X &&
          ePrev->Curr.Y == horzEdge->Bot.Y && ePrev->WindDelta != 0 &&
          (ePrev->OutIdx >= 0 && ePrev->Curr.Y > ePrev->Top.Y &&
           SlopesEqual(*horzEdge, *ePrev, m_UseFullRange))) {
        // 添加输出点
        OutPt *op2 = AddOutPt(ePrev, horzEdge->Bot);
        // 添加连接点
        AddJoin(op1, op2, horzEdge->Top);
      } else if (eNext && eNext->Curr.X == horzEdge->Bot.X &&
                 eNext->Curr.Y == horzEdge->Bot.Y && eNext->WindDelta != 0 &&
                 eNext->OutIdx >= 0 && eNext->Curr.Y > eNext->Top.Y &&
                 SlopesEqual(*horzEdge, *eNext, m_UseFullRange)) {
        // 添加输出点
        OutPt *op2 = AddOutPt(eNext, horzEdge->Bot);
        // 添加连接点
        AddJoin(op1, op2, horzEdge->Top);
      }
    } else
      // 更新边到活动边表
      UpdateEdgeIntoAEL(horzEdge);
  } else {
    // 如果水平边的 OutIdx 大于等于 0，则添加输出点
    if (horzEdge->OutIdx >= 0)
      AddOutPt(horzEdge, horzEdge->Top);
    // 从活动边表中删除水平边
    DeleteFromAEL(horzEdge);
  }
// 处理交点，传入顶部 Y 坐标
bool Clipper::ProcessIntersections(const cInt topY) {
  // 如果没有活动边，直接返回 true
  if (!m_ActiveEdges)
    return true;
  try {
    // 构建交点列表
    BuildIntersectList(topY);
    // 获取交点列表的大小
    size_t IlSize = m_IntersectList.size();
    // 如果交点列表为空，直接返回 true
    if (IlSize == 0)
      return true;
    // 如果交点列表只有一个元素或者修复交点顺序成功，则处理交点列表
    if (IlSize == 1 || FixupIntersectionOrder())
      ProcessIntersectList();
    else
      return false;
  } catch (...) {
    // 出现异常时，重置排序边为 0，释放交点节点，抛出异常
    m_SortedEdges = 0;
    DisposeIntersectNodes();
    throw clipperException("ProcessIntersections error");
  }
  // 重置排序边为 0，返回 true
  m_SortedEdges = 0;
  return true;
}
//------------------------------------------------------------------------------

// 释放交点节点
void Clipper::DisposeIntersectNodes() {
  // 遍历交点列表，释放交点节点
  for (size_t i = 0; i < m_IntersectList.size(); ++i)
    delete m_IntersectList[i];
  // 清空交点列表
  m_IntersectList.clear();
}
//------------------------------------------------------------------------------

// 构建交点列表，传入顶部 Y 坐标
void Clipper::BuildIntersectList(const cInt topY) {
  // 如果没有活动边，直接返回
  if (!m_ActiveEdges)
    return;

  // 准备排序...
  TEdge *e = m_ActiveEdges;
  m_SortedEdges = e;
  // 遍历活动边，设置前一个和后一个边，计算当前 X 坐标
  while (e) {
    e->PrevInSEL = e->PrevInAEL;
    e->NextInSEL = e->NextInAEL;
    e->Curr.X = TopX(*e, topY);
    e = e->NextInAEL;
  }

  // 冒泡排序...
  bool isModified;
  do {
    isModified = false;
    e = m_SortedEdges;
    while (e->NextInSEL) {
      TEdge *eNext = e->NextInSEL;
      IntPoint Pt;
      // 如果当前边的 X 坐标大于下一个边的 X 坐标
      if (e->Curr.X > eNext->Curr.X) {
        // 计算交点，如果交点的 Y 坐标小于顶部 Y 坐标，将交点设置为当前边的顶部坐标
        IntersectPoint(*e, *eNext, Pt);
        if (Pt.Y < topY)
          Pt = IntPoint(TopX(*e, topY), topY);
        // 创建交点节点，设置边和交点，添加到交点列表中
        IntersectNode *newNode = new IntersectNode;
        newNode->Edge1 = e;
        newNode->Edge2 = eNext;
        newNode->Pt = Pt;
        m_IntersectList.push_back(newNode);

        // 交换在 SEL 中的位置，标记已修改
        SwapPositionsInSEL(e, eNext);
        isModified = true;
      } else
        e = eNext;
    }
    // 如果前一个边在 SEL 中存在，将其下一个边置为 0，否则退出循环
    if (e->PrevInSEL)
      e->PrevInSEL->NextInSEL = 0;
    else
      break;
  } while (isModified);
  // 重置排序边为 0，重要操作
  m_SortedEdges = 0;
}
// 处理交点列表中的交点
void Clipper::ProcessIntersectList() {
  // 遍历交点列表
  for (size_t i = 0; i < m_IntersectList.size(); ++i) {
    // 获取当前交点
    IntersectNode *iNode = m_IntersectList[i];
    {
      // 处理交点对应的边，交点和位置交换
      IntersectEdges(iNode->Edge1, iNode->Edge2, iNode->Pt);
      SwapPositionsInAEL(iNode->Edge1, iNode->Edge2);
    }
    // 释放交点内存
    delete iNode;
  }
  // 清空交点列表
  m_IntersectList.clear();
}
//------------------------------------------------------------------------------

// 比较函数，用于对交点列表进行排序，按照 Y 坐标从大到小排序
bool IntersectListSort(IntersectNode *node1, IntersectNode *node2) {
  return node2->Pt.Y < node1->Pt.Y;
}
//------------------------------------------------------------------------------

// 判断交点对应的边是否相邻
inline bool EdgesAdjacent(const IntersectNode &inode) {
  return (inode.Edge1->NextInSEL == inode.Edge2) ||
         (inode.Edge1->PrevInSEL == inode.Edge2);
}
//------------------------------------------------------------------------------

// 调整交点顺序，确保交点只在相邻边之间
bool Clipper::FixupIntersectionOrder() {
  // 前提条件：交点按照从底部到顶部排序
  // 确保交点只在相邻边之间，需要调整交点顺序
  CopyAELToSEL();
  // 对交点列表进行排序
  std::sort(m_IntersectList.begin(), m_IntersectList.end(), IntersectListSort);
  size_t cnt = m_IntersectList.size();
  for (size_t i = 0; i < cnt; ++i) {
    // 如果交点对应的边不相邻
    if (!EdgesAdjacent(*m_IntersectList[i])) {
      size_t j = i + 1;
      // 找到下一个相邻的交点
      while (j < cnt && !EdgesAdjacent(*m_IntersectList[j]))
        j++;
      // 如果没有找到相邻的交点，返回 false
      if (j == cnt)
        return false;
      // 交换交点位置
      std::swap(m_IntersectList[i], m_IntersectList[j]);
    }
    // 交换交点对应的边在 SEL 中的位置
    SwapPositionsInSEL(m_IntersectList[i]->Edge1, m_IntersectList[i]->Edge2);
  }
  return true;
}
//------------------------------------------------------------------------------

// 处理最大值点
void Clipper::DoMaxima(TEdge *e) {
  // 获取与当前边相关的最大值点对
  TEdge *eMaxPair = GetMaximaPairEx(e);
  // 如果没有最大值点对
  if (!eMaxPair) {
    // 如果当前边有输出索引，添加输出点
    if (e->OutIdx >= 0)
      AddOutPt(e, e->Top);
    // 从 AEL 中删除当前边
    DeleteFromAEL(e);
  }
    // 如果当前边为空，则直接返回
    return;
  }

  // 获取当前边的下一个边
  TEdge *eNext = e->NextInAEL;
  // 遍历当前边的下一个边，直到遇到最大对边或者为空
  while (eNext && eNext != eMaxPair) {
    // 计算当前边和下一个边的交点
    IntersectEdges(e, eNext, e->Top);
    // 交换当前边和下一个边在活动边表中的位置
    SwapPositionsInAEL(e, eNext);
    // 更新下一个边为当前边的下一个边
    eNext = e->NextInAEL;
  }

  // 如果当前边和最大对边都没有关联的多边形索引
  if (e->OutIdx == Unassigned && eMaxPair->OutIdx == Unassigned) {
    // 从活动边表中删除当前边和最大对边
    DeleteFromAEL(e);
    DeleteFromAEL(eMaxPair);
  } 
  // 如果当前边和最大对边都有关联的多边形索引
  else if (e->OutIdx >= 0 && eMaxPair->OutIdx >= 0) {
    // 如果当前边有关联的多边形索引，则添加局部最大多边形
    if (e->OutIdx >= 0)
      AddLocalMaxPoly(e, eMaxPair, e->Top);
    // 从活动边表中删除当前边和最大对边
    DeleteFromAEL(e);
    DeleteFromAEL(eMaxPair);
  }
#ifdef use_lines
  // 如果当前边是垂直边
  else if (e->WindDelta == 0) {
    // 如果当前边在 AEL 中
    if (e->OutIdx >= 0) {
      // 将当前边的顶点添加到输出点
      AddOutPt(e, e->Top);
      // 重置当前边的 OutIdx
      e->OutIdx = Unassigned;
    }
    // 从 AEL 中删除当前边
    DeleteFromAEL(e);

    // 如果当前边的最大值对应的边在 AEL 中
    if (eMaxPair->OutIdx >= 0) {
      // 将最大值对应的边的顶点添加到输出点
      AddOutPt(eMaxPair, e->Top);
      // 重置最大值对应的边的 OutIdx
      eMaxPair->OutIdx = Unassigned;
    }
    // 从 AEL 中删除最大值对应的边
    DeleteFromAEL(eMaxPair);
  }
#endif
  // 如果不是垂直边，抛出异常
  else
    throw clipperException("DoMaxima error");
}
//------------------------------------------------------------------------------

// 处理扫描线顶部的边
void Clipper::ProcessEdgesAtTopOfScanbeam(const cInt topY) {
  // 获取当前活动边
  TEdge *e = m_ActiveEdges;
  // 遍历当前活动边
  while (e) {
    // 1. 处理极大值，将其视为“弯曲”的水平边，但排除具有水平边的极大值。注意：e 不能是水平边。
    bool IsMaximaEdge = IsMaxima(e, topY);

    if (IsMaximaEdge) {
      // 获取与当前边相关的最大值对应的边
      TEdge *eMaxPair = GetMaximaPairEx(e);
      // 检查最大值是否具有水平边
      IsMaximaEdge = (!eMaxPair || !IsHorizontal(*eMaxPair));
    }

    if (IsMaximaEdge) {
      // 如果是极大值边
      if (m_StrictSimple)
        // 将当前边的顶点的 X 坐标添加到最大值列表
        m_Maxima.push_back(e->Top.X);
      // 获取当前边的前一个边
      TEdge *ePrev = e->PrevInAEL;
      // 处理极大值
      DoMaxima(e);
      // 更新当前边
      if (!ePrev)
        e = m_ActiveEdges;
      else
        e = ePrev->NextInAEL;
    } else {
      // 2. 提升水平边，否则更新当前边的 X 和 Y 坐标
      if (IsIntermediate(e, topY) && IsHorizontal(*e->NextInLML)) {
        // 更新当前边到 AEL 中
        UpdateEdgeIntoAEL(e);
        // 如果当前边的 OutIdx 大于等于 0
        if (e->OutIdx >= 0)
          // 将当前边的底部顶点添加到输出点
          AddOutPt(e, e->Bot);
        // 将当前边添加到 SEL 中
        AddEdgeToSEL(e);
      } else {
        // 更新当前边的 X 和 Y 坐标
        e->Curr.X = TopX(*e, topY);
        e->Curr.Y = topY;
#ifdef use_xyz
        // 如果当前边的顶点 Y 坐标等于 topY，则将 Z 坐标设置为顶点的 Z 坐标，否则设置为 0
        e->Curr.Z =
            topY == e->Top.Y ? e->Top.Z : (topY == e->Bot.Y ? e->Bot.Z : 0);
#endif
      }

      // 当 StrictlySimple 且 'e' 被另一条边触及时，确保两条边在此处都有一个顶点
      if (m_StrictSimple) {
        // 获取当前边的前一个边
        TEdge *ePrev = e->PrevInAEL;
        // 如果当前边的 OutIdx 大于等于 0 且 WindDelta 不为 0 且前一个边存在且前一个边的 OutIdx 大于等于 0 且前一个边的当前 X 坐标等于当前边的当前 X 坐标 且前一个边的 WindDelta 不为 0
        if ((e->OutIdx >= 0) && (e->WindDelta != 0) && ePrev &&
            (ePrev->OutIdx >= 0) && (ePrev->Curr.X == e->Curr.X) &&
            (ePrev->WindDelta != 0)) {
          // 创建一个顶点
          IntPoint pt = e->Curr;
#ifdef use_xyz
          // 如果定义了 use_xyz 宏，则设置当前边的终点为 pt
          SetZ(pt, *ePrev, *e);
#endif
          // 在当前边 ePrev 后添加一个 OutPt，并设置其坐标为 pt
          OutPt *op = AddOutPt(ePrev, pt);
          // 在当前边 e 后添加一个 OutPt，并设置其坐标为 pt
          OutPt *op2 = AddOutPt(e, pt);
          // 添加一个 StrictlySimple (type-3) join，将 op 和 op2 连接起来
          AddJoin(op, op2, pt);
        }
      }

      // 遍历活动边表中的边
      e = e->NextInAEL;
    }
  }

  // 3. 处理扫描线顶部的水平边 ...
  // 对最大值点进行排序
  m_Maxima.sort();
  // 处理水平边
  ProcessHorizontals();
  // 清空最大值点列表
  m_Maxima.clear();

  // 4. 提升中间顶点 ...
  // 重新设置 e 为活动边表的头部
  e = m_ActiveEdges;
  while (e) {
    // 如果当前边是中间顶点
    if (IsIntermediate(e, topY)) {
      OutPt *op = 0;
      // 如果当前边的 OutIdx 大于等于 0，则添加一个 OutPt
      if (e->OutIdx >= 0)
        op = AddOutPt(e, e->Top);
      // 更新当前边到活动边表
      UpdateEdgeIntoAEL(e);

      // 如果输出多边形共享一条边，稍后需要连接 ...
      TEdge *ePrev = e->PrevInAEL;
      TEdge *eNext = e->NextInAEL;
      if (ePrev && ePrev->Curr.X == e->Bot.X && ePrev->Curr.Y == e->Bot.Y &&
          op && ePrev->OutIdx >= 0 && ePrev->Curr.Y > ePrev->Top.Y &&
          SlopesEqual(e->Curr, e->Top, ePrev->Curr, ePrev->Top,
                      m_UseFullRange) &&
          (e->WindDelta != 0) && (ePrev->WindDelta != 0)) {
        OutPt *op2 = AddOutPt(ePrev, e->Bot);
        AddJoin(op, op2, e->Top);
      } else if (eNext && eNext->Curr.X == e->Bot.X &&
                 eNext->Curr.Y == e->Bot.Y && op && eNext->OutIdx >= 0 &&
                 eNext->Curr.Y > eNext->Top.Y &&
                 SlopesEqual(e->Curr, e->Top, eNext->Curr, eNext->Top,
                             m_UseFullRange) &&
                 (e->WindDelta != 0) && (eNext->WindDelta != 0)) {
        OutPt *op2 = AddOutPt(eNext, e->Bot);
        AddJoin(op, op2, e->Top);
      }
    }
    // 移动到下一个活动边
    e = e->NextInAEL;
  }
}
//------------------------------------------------------------------------------

// 修正输出多边形的折线
void Clipper::FixupOutPolyline(OutRec &outrec) {
  OutPt *pp = outrec.Pts;
  OutPt *lastPP = pp->Prev;
  // 遍历输出多边形的点
  while (pp != lastPP) {
    pp = pp->Next;
    # 如果当前点的前一个点与当前点相同
    if (pp->Pt == pp->Prev->Pt) {
      # 如果当前点是最后一个点
      if (pp == lastPP)
        lastPP = pp->Prev;
      # 临时保存当前点的前一个点
      OutPt *tmpPP = pp->Prev;
      # 调整指针，删除当前点
      tmpPP->Next = pp->Next;
      pp->Next->Prev = tmpPP;
      delete pp;
      pp = tmpPP;
    }
  }

  # 如果当前点等于前一个点
  if (pp == pp->Prev) {
    # 释放当前点及其后续点
    DisposeOutPts(pp);
    # 将输出记录的点数置为0
    outrec.Pts = 0;
    return;
  }
// 修复输出多边形，去除重复点并简化连续平行边，通过移除中间顶点
void Clipper::FixupOutPolygon(OutRec &outrec) {
  // 初始化变量
  OutPt *lastOK = 0;
  outrec.BottomPt = 0;
  OutPt *pp = outrec.Pts;
  bool preserveCol = m_PreserveCollinear || m_StrictSimple;

  // 循环处理多边形顶点
  for (;;) {
    // 如果当前顶点的前一个或后一个顶点与当前顶点相同，说明存在问题，需要处理
    if (pp->Prev == pp || pp->Prev == pp->Next) {
      // 释放顶点内存，重置多边形顶点
      DisposeOutPts(pp);
      outrec.Pts = 0;
      return;
    }

    // 检查是否存在重复点和共线边
    if ((pp->Pt == pp->Next->Pt) || (pp->Pt == pp->Prev->Pt) ||
        (SlopesEqual(pp->Prev->Pt, pp->Pt, pp->Next->Pt, m_UseFullRange) &&
         (!preserveCol ||
          !Pt2IsBetweenPt1AndPt3(pp->Prev->Pt, pp->Pt, pp->Next->Pt)))) {
      // 移除中间顶点
      lastOK = 0;
      OutPt *tmp = pp;
      pp->Prev->Next = pp->Next;
      pp->Next->Prev = pp->Prev;
      pp = pp->Prev;
      delete tmp;
    } else if (pp == lastOK)
      break;
    else {
      if (!lastOK)
        lastOK = pp;
      pp = pp->Next;
    }
  }
  outrec.Pts = pp;
}
//------------------------------------------------------------------------------

// 计算多边形顶点数量
int PointCount(OutPt *Pts) {
  if (!Pts)
    return 0;
  int result = 0;
  OutPt *p = Pts;
  do {
    result++;
    p = p->Next;
  } while (p != Pts);
  return result;
}
//------------------------------------------------------------------------------

// 构建结果多边形集合
void Clipper::BuildResult(Paths &polys) {
  // 预留空间
  polys.reserve(m_PolyOuts.size());
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i) {
    // 如果多边形没有顶点，继续下一个
    if (!m_PolyOuts[i]->Pts)
      continue;
    Path pg;
    OutPt *p = m_PolyOuts[i]->Pts->Prev;
    int cnt = PointCount(p);
    // 如果顶点数量小于2，继续下一个
    if (cnt < 2)
      continue;
    pg.reserve(cnt);
    // 将顶点添加到多边形中
    for (int i = 0; i < cnt; ++i) {
      pg.push_back(p->Pt);
      p = p->Prev;
    }
    polys.push_back(pg);
  }
}
//------------------------------------------------------------------------------
// 构建结果 PolyTree，清空 polytree
void Clipper::BuildResult2(PolyTree &polytree) {
  polytree.Clear();
  // 预留足够的空间以容纳所有节点
  polytree.AllNodes.reserve(m_PolyOuts.size());
  // 遍历每个输出多边形/轮廓并添加到 polytree 中
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); i++) {
    OutRec *outRec = m_PolyOuts[i];
    int cnt = PointCount(outRec->Pts);
    // 如果是开放的并且点数小于2，或者不是开放的并且点数小于3，则跳过
    if ((outRec->IsOpen && cnt < 2) || (!outRec->IsOpen && cnt < 3))
      continue;
    // 修复孔洞链接
    FixHoleLinkage(*outRec);
    PolyNode *pn = new PolyNode();
    // 注意：polytree 接管所有 PolyNodes 的所有权
    polytree.AllNodes.push_back(pn);
    outRec->PolyNd = pn;
    pn->Parent = 0;
    pn->Index = 0;
    pn->Contour.reserve(cnt);
    OutPt *op = outRec->Pts->Prev;
    for (int j = 0; j < cnt; j++) {
      pn->Contour.push_back(op->Pt);
      op = op->Prev;
    }
  }

  // 修复 PolyNode 链接等
  polytree.Childs.reserve(m_PolyOuts.size());
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); i++) {
    OutRec *outRec = m_PolyOuts[i];
    if (!outRec->PolyNd)
      continue;
    if (outRec->IsOpen) {
      outRec->PolyNd->m_IsOpen = true;
      polytree.AddChild(*outRec->PolyNd);
    } else if (outRec->FirstLeft && outRec->FirstLeft->PolyNd)
      outRec->FirstLeft->PolyNd->AddChild(*outRec->PolyNd);
    else
      polytree.AddChild(*outRec->PolyNd);
  }
}
//------------------------------------------------------------------------------

// 交换两个交点节点的内容
void SwapIntersectNodes(IntersectNode &int1, IntersectNode &int2) {
  // 仅交换内容（因为 fIntersectNodes 是单链表）
  IntersectNode inode = int1; // 获取 Int1 的副本
  int1.Edge1 = int2.Edge1;
  int1.Edge2 = int2.Edge2;
  int1.Pt = int2.Pt;
  int2.Edge1 = inode.Edge1;
  int2.Edge2 = inode.Edge2;
  int2.Pt = inode.Pt;
}
//------------------------------------------------------------------------------

// 判断 e2 是否在 e1 之前插入
inline bool E2InsertsBeforeE1(TEdge &e1, TEdge &e2) {
  if (e2.Curr.X == e1.Curr.X) {
    if (e2.Top.Y > e1.Top.Y)
      return e2.Top.X < TopX(e1, e2.Top.Y);
    else
      # 如果 e1 的顶点 X 坐标大于 e2 在 e1 的顶点 Y 坐标处的 X 坐标，则返回 True
      return e1.Top.X > TopX(e2, e1.Top.Y);
  } else
    # 如果 e2 的当前点 X 坐标小于 e1 的当前点 X 坐标，则返回 True
    return e2.Curr.X < e1.Curr.X;
// 结束当前的代码块
}

// 检查两个区间[a1, a2]和[b1, b2]是否有重叠部分，并返回重叠部分的左右边界
bool GetOverlap(const cInt a1, const cInt a2, const cInt b1, const cInt b2,
                cInt &Left, cInt &Right) {
  if (a1 < a2) {
    if (b1 < b2) {
      Left = std::max(a1, b1);
      Right = std::min(a2, b2);
    } else {
      Left = std::max(a1, b2);
      Right = std::min(a2, b1);
    }
  } else {
    if (b1 < b2) {
      Left = std::max(a2, b1);
      Right = std::min(a1, b2);
    } else {
      Left = std::max(a2, b2);
      Right = std::min(a1, b1);
    }
  }
  return Left < Right;
}
//------------------------------------------------------------------------------

// 更新 OutRec 中的所有 OutPt 的 Idx 字段为 OutRec 的 Idx
inline void UpdateOutPtIdxs(OutRec &outrec) {
  OutPt *op = outrec.Pts;
  do {
    op->Idx = outrec.Idx;
    op = op->Prev;
  } while (op != outrec.Pts);
}
//------------------------------------------------------------------------------

// 将边插入到活动边列表中
void Clipper::InsertEdgeIntoAEL(TEdge *edge, TEdge *startEdge) {
  if (!m_ActiveEdges) {
    edge->PrevInAEL = 0;
    edge->NextInAEL = 0;
    m_ActiveEdges = edge;
  } else if (!startEdge && E2InsertsBeforeE1(*m_ActiveEdges, *edge)) {
    edge->PrevInAEL = 0;
    edge->NextInAEL = m_ActiveEdges;
    m_ActiveEdges->PrevInAEL = edge;
    m_ActiveEdges = edge;
  } else {
    if (!startEdge)
      startEdge = m_ActiveEdges;
    while (startEdge->NextInAEL &&
           !E2InsertsBeforeE1(*startEdge->NextInAEL, *edge))
      startEdge = startEdge->NextInAEL;
    edge->NextInAEL = startEdge->NextInAEL;
    if (startEdge->NextInAEL)
      startEdge->NextInAEL->PrevInAEL = edge;
    edge->PrevInAEL = startEdge;
    startEdge->NextInAEL = edge;
  }
}
//----------------------------------------------------------------------

// 复制一个 OutPt 节点，并根据 InsertAfter 决定插入到原节点之前还是之后
OutPt *DupOutPt(OutPt *outPt, bool InsertAfter) {
  OutPt *result = new OutPt;
  result->Pt = outPt->Pt;
  result->Idx = outPt->Idx;
  if (InsertAfter) {
    result->Next = outPt->Next;
    result->Prev = outPt;
    outPt->Next->Prev = result;
    # 如果outPt的Next指针不为空，则将result赋值给outPt的Next指针
    outPt->Next = result;
  } else {
    # 如果outPt的Next指针为空，则将result赋值给outPt的Prev指针，并调整相邻节点的指针
    result->Prev = outPt->Prev;
    result->Next = outPt;
    outPt->Prev->Next = result;
    outPt->Prev = result;
  }
  # 返回result节点
  return result;
// 结束 JoinHorz 函数定义
}
//------------------------------------------------------------------------------

// 水平连接两个 OutPt 对象，根据给定的点 Pt 和 DiscardLeft 参数确定方向
bool JoinHorz(OutPt *op1, OutPt *op1b, OutPt *op2, OutPt *op2b,
              const IntPoint Pt, bool DiscardLeft) {
  // 确定 op1 和 op1b 的方向
  Direction Dir1 = (op1->Pt.X > op1b->Pt.X ? dRightToLeft : dLeftToRight);
  // 确定 op2 和 op2b 的方向
  Direction Dir2 = (op2->Pt.X > op2b->Pt.X ? dRightToLeft : dLeftToRight);
  // 如果 op1 和 op2 的方向相同，则返回 false
  if (Dir1 == Dir2)
    return false;

  // 当 DiscardLeft 为真时，确保 Op1b 在 Op1 的左侧，否则在右侧
  // 为了在插入 Op1b 和 Op2b 时方便...
  // 当 DiscardLeft 为真时，在添加 Op1b 之前确保我们在 Pt 的右侧或正好在 Pt 上，
  // 否则确保我们在 Pt 的左侧或正好在 Pt 上（Op2b 同理）
  if (Dir1 == dLeftToRight) {
    while (op1->Next->Pt.X <= Pt.X && op1->Next->Pt.X >= op1->Pt.X &&
           op1->Next->Pt.Y == Pt.Y)
      op1 = op1->Next;
    if (DiscardLeft && (op1->Pt.X != Pt.X))
      op1 = op1->Next;
    op1b = DupOutPt(op1, !DiscardLeft);
    if (op1b->Pt != Pt) {
      op1 = op1b;
      op1->Pt = Pt;
      op1b = DupOutPt(op1, !DiscardLeft);
    }
  } else {
    while (op1->Next->Pt.X >= Pt.X && op1->Next->Pt.X <= op1->Pt.X &&
           op1->Next->Pt.Y == Pt.Y)
      op1 = op1->Next;
    if (!DiscardLeft && (op1->Pt.X != Pt.X))
      op1 = op1->Next;
    op1b = DupOutPt(op1, DiscardLeft);
    if (op1b->Pt != Pt) {
      op1 = op1b;
      op1->Pt = Pt;
      op1b = DupOutPt(op1, DiscardLeft);
    }
  }

  // 确定 op2 和 op2b 的位置
  if (Dir2 == dLeftToRight) {
    while (op2->Next->Pt.X <= Pt.X && op2->Next->Pt.X >= op2->Pt.X &&
           op2->Next->Pt.Y == Pt.Y)
      op2 = op2->Next;
    if (DiscardLeft && (op2->Pt.X != Pt.X))
      op2 = op2->Next;
    op2b = DupOutPt(op2, !DiscardLeft);
    if (op2b->Pt != Pt) {
      op2 = op2b;
      op2->Pt = Pt;
      op2b = DupOutPt(op2, !DiscardLeft);
    };
  } else {
    // 当 op2 的下一个点的 X 坐标在 Pt 的 X 坐标范围内，并且 Y 坐标等于 Pt 的 Y 坐标时，移动 op2 指针
    while (op2->Next->Pt.X >= Pt.X && op2->Next->Pt.X <= op2->Pt.X &&
           op2->Next->Pt.Y == Pt.Y)
      op2 = op2->Next;
    // 如果不丢弃左侧点，并且 op2 的 X 坐标不等于 Pt 的 X 坐标，则移动 op2 指针
    if (!DiscardLeft && (op2->Pt.X != Pt.X))
      op2 = op2->Next;
    // 复制 op2 点，根据是否丢弃左侧点
    op2b = DupOutPt(op2, DiscardLeft);
    // 如果 op2b 点不等于 Pt，则更新 op2 指针为 op2b，并将 op2b 点设置为 Pt
    if (op2b->Pt != Pt) {
      op2 = op2b;
      op2->Pt = Pt;
      op2b = DupOutPt(op2, DiscardLeft);
    };
  };

  // 根据 Dir1 和 DiscardLeft 的值判断操作
  if ((Dir1 == dLeftToRight) == DiscardLeft) {
    // 更新 op1 和 op2 之间的连接关系
    op1->Prev = op2;
    op2->Next = op1;
    op1b->Next = op2b;
    op2b->Prev = op1b;
  } else {
    // 更新 op1 和 op2 之间的连接关系
    op1->Next = op2;
    op2->Prev = op1;
    op1b->Prev = op2b;
    op2b->Next = op1b;
  }
  // 返回 true
  return true;
// 检查两个输出多边形的连接点，根据连接类型进行处理
bool Clipper::JoinPoints(Join *j, OutRec *outRec1, OutRec *outRec2) {
  // 获取连接点1和连接点2
  OutPt *op1 = j->OutPt1, *op1b;
  OutPt *op2 = j->OutPt2, *op2b;

  // 判断连接类型：水平连接或非水平连接
  bool isHorizontal = (j->OutPt1->Pt.Y == j->OffPt.Y);

  // 处理严格简单连接
  if (isHorizontal && (j->OffPt == j->OutPt1->Pt) &&
      (j->OffPt == j->OutPt2->Pt)) {
    // 如果输出多边形不同，则返回false
    if (outRec1 != outRec2)
      return false;
    // 复制连接点1和连接点2
    op1b = j->OutPt1->Next;
    while (op1b != op1 && (op1b->Pt == j->OffPt))
      op1b = op1b->Next;
    bool reverse1 = (op1b->Pt.Y > j->OffPt.Y);
    op2b = j->OutPt2->Next;
    while (op2b != op2 && (op2b->Pt == j->OffPt))
      op2b = op2b->Next;
    bool reverse2 = (op2b->Pt.Y > j->OffPt.Y);
    // 如果两个连接方向相同，则返回false
    if (reverse1 == reverse2)
      return false;
    // 根据连接方向进行处理
    if (reverse1) {
      op1b = DupOutPt(op1, false);
      op2b = DupOutPt(op2, true);
      op1->Prev = op2;
      op2->Next = op1;
      op1b->Next = op2b;
      op2b->Prev = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    } else {
      op1b = DupOutPt(op1, true);
      op2b = DupOutPt(op2, false);
      op1->Next = op2;
      op2->Prev = op1;
      op1b->Prev = op2b;
      op2b->Next = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    }
  } else if (isHorizontal) {
    // 处理水平连接和非水平连接
    // 将 op1b 初始化为 op1，用于记录 op1 的起始位置
    op1b = op1;
    // 在 op1 的前一个节点的 Y 坐标与 op1 的 Y 坐标相等，且前一个节点不是 op1b 且不是 op2 时，向前移动 op1
    while (op1->Prev->Pt.Y == op1->Pt.Y && op1->Prev != op1b &&
           op1->Prev != op2)
      op1 = op1->Prev;
    // 在 op1b 的下一个节点的 Y 坐标与 op1b 的 Y 坐标相等，且下一个节点不是 op1 且不是 op2 时，向后移动 op1b
    while (op1b->Next->Pt.Y == op1b->Pt.Y && op1b->Next != op1 &&
           op1b->Next != op2)
      op1b = op1b->Next;
    // 如果 op1b 的下一个节点是 op1 或 op2，则返回 false，表示一个平坦的多边形
    if (op1b->Next == op1 || op1b->Next == op2)
      return false; // a flat 'polygon'

    // 类似上述操作，处理 op2
    op2b = op2;
    while (op2->Prev->Pt.Y == op2->Pt.Y && op2->Prev != op2b &&
           op2->Prev != op1b)
      op2 = op2->Prev;
    while (op2b->Next->Pt.Y == op2b->Pt.Y && op2b->Next != op2 &&
           op2b->Next != op1)
      op2b = op2b->Next;
    if (op2b->Next == op2 || op2b->Next == op1)
      return false; // a flat 'polygon'

    // 计算水平边的左右端点
    cInt Left, Right;
    if (!GetOverlap(op1->Pt.X, op1b->Pt.X, op2->Pt.X, op2b->Pt.X, Left, Right))
      return false;

    // 处理水平边的连接，确定是否需要丢弃左侧
    IntPoint Pt;
    bool DiscardLeftSide;
    if (op1->Pt.X >= Left && op1->Pt.X <= Right) {
      Pt = op1->Pt;
      DiscardLeftSide = (op1->Pt.X > op1b->Pt.X);
    } else if (op2->Pt.X >= Left && op2->Pt.X <= Right) {
      Pt = op2->Pt;
      DiscardLeftSide = (op2->Pt.X > op2b->Pt.X);
    } else if (op1b->Pt.X >= Left && op1b->Pt.X <= Right) {
      Pt = op1b->Pt;
      DiscardLeftSide = op1b->Pt.X > op1->Pt.X;
    } else {
      Pt = op2b->Pt;
      DiscardLeftSide = (op2b->Pt.X > op2->Pt.X);
    }
    // 设置 OutPt1 和 OutPt2，然后调用 JoinHorz 进行水平连接
    j->OutPt1 = op1;
    j->OutPt2 = op2;
    return JoinHorz(op1, op1b, op2, op2b, Pt, DiscardLeftSide);
  } else {
    // 对于非水平连接的情况
    //    1. Jr.OutPt1.Pt.Y == Jr.OutPt2.Pt.Y
    //    2. Jr.OutPt1.Pt > Jr.OffPt.Y

    // 确保多边形的方向正确...
    op1b = op1->Next;
    // 找到 op1 的下一个点，直到找到不等于 op1 的点
    while ((op1b->Pt == op1->Pt) && (op1b != op1))
      op1b = op1b->Next;
    // 检查是否需要反转 op1
    bool Reverse1 = ((op1b->Pt.Y > op1->Pt.Y) ||
                     !SlopesEqual(op1->Pt, op1b->Pt, j->OffPt, m_UseFullRange));
    if (Reverse1) {
      op1b = op1->Prev;
      // 找到 op1 的前一个点，直到找到不等于 op1 的点
      while ((op1b->Pt == op1->Pt) && (op1b != op1))
        op1b = op1b->Prev;
      // 如果需要反转 op1，则返回 false
      if ((op1b->Pt.Y > op1->Pt.Y) ||
          !SlopesEqual(op1->Pt, op1b->Pt, j->OffPt, m_UseFullRange))
        return false;
    };
    op2b = op2->Next;
    // 找到 op2 的下一个点，直到找到不等于 op2 的点
    while ((op2b->Pt == op2->Pt) && (op2b != op2))
      op2b = op2b->Next;
    // 检查是否需要反转 op2
    bool Reverse2 = ((op2b->Pt.Y > op2->Pt.Y) ||
                     !SlopesEqual(op2->Pt, op2b->Pt, j->OffPt, m_UseFullRange));
    if (Reverse2) {
      op2b = op2->Prev;
      // 找到 op2 的前一个点，直到找到不等于 op2 的点
      while ((op2b->Pt == op2->Pt) && (op2b != op2))
        op2b = op2b->Prev;
      // 如果需要反转 op2，则返回 false
      if ((op2b->Pt.Y > op2->Pt.Y) ||
          !SlopesEqual(op2->Pt, op2b->Pt, j->OffPt, m_UseFullRange))
        return false;
    }

    // 检查是否存在特殊情况，如果存在则返回 false
    if ((op1b == op1) || (op2b == op2) || (op1b == op2b) ||
        ((outRec1 == outRec2) && (Reverse1 == Reverse2)))
      return false;

    // 根据需要反转的情况进行操作
    if (Reverse1) {
      op1b = DupOutPt(op1, false);
      op2b = DupOutPt(op2, true);
      op1->Prev = op2;
      op2->Next = op1;
      op1b->Next = op2b;
      op2b->Prev = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    } else {
      op1b = DupOutPt(op1, true);
      op2b = DupOutPt(op2, false);
      op1->Next = op2;
      op2->Prev = op1;
      op1b->Prev = op2b;
      op2b->Next = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    }
  }
// 解析并返回第一个具有点集的 OutRec 对象，如果没有则返回其 FirstLeft 对象
static OutRec *ParseFirstLeft(OutRec *FirstLeft) {
  while (FirstLeft && !FirstLeft->Pts)
    FirstLeft = FirstLeft->FirstLeft;
  return FirstLeft;
}
//------------------------------------------------------------------------------

// 修正 FirstLeft 指针，确保 NewOutRec 包含在 OldOutRec 之前
void Clipper::FixupFirstLefts1(OutRec *OldOutRec, OutRec *NewOutRec) {
  // 遍历所有的 PolyOut 对象
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i) {
    OutRec *outRec = m_PolyOuts[i];
    // 解析并获取当前 OutRec 对象的第一个具有点集的 FirstLeft 对象
    OutRec *firstLeft = ParseFirstLeft(outRec->FirstLeft);
    // 如果当前 OutRec 对象有点集，并且其 FirstLeft 对象为 OldOutRec
    if (outRec->Pts && firstLeft == OldOutRec) {
      // 如果 NewOutRec 包含在当前 OutRec 对象中，则将其设置为 FirstLeft
      if (Poly2ContainsPoly1(outRec->Pts, NewOutRec->Pts))
        outRec->FirstLeft = NewOutRec;
    }
  }
}
//----------------------------------------------------------------------

// 修正 FirstLeft 指针，处理一个多边形被分割成两个的情况
void Clipper::FixupFirstLefts2(OutRec *InnerOutRec, OutRec *OuterOutRec) {
  // 获取 OuterOutRec 的 FirstLeft 对象
  OutRec *orfl = OuterOutRec->FirstLeft;
  // 遍历所有的 PolyOut 对象
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i) {
    OutRec *outRec = m_PolyOuts[i];

    // 如果当前 OutRec 对象没有点集，或者等于 InnerOutRec 或者 OuterOutRec，则继续下一次循环
    if (!outRec->Pts || outRec == OuterOutRec || outRec == InnerOutRec)
      continue;
    // 解析并获取当前 OutRec 对象的第一个具有点集的 FirstLeft 对象
    OutRec *firstLeft = ParseFirstLeft(outRec->FirstLeft);
    // 如果当前 OutRec 对象的 FirstLeft 对象不等于 orfl、InnerOutRec 或 OuterOutRec，则继续下一次循环
    if (firstLeft != orfl && firstLeft != InnerOutRec &&
        firstLeft != OuterOutRec)
      continue;
    // 如果当前 OutRec 对象包含在 InnerOutRec 中，则将其 FirstLeft 设置为 InnerOutRec
    if (Poly2ContainsPoly1(outRec->Pts, InnerOutRec->Pts))
      outRec->FirstLeft = InnerOutRec;
    // 如果当前 OutRec 对象包含在 OuterOutRec 中，则将其 FirstLeft 设置为 OuterOutRec
    else if (Poly2ContainsPoly1(outRec->Pts, OuterOutRec->Pts))
      outRec->FirstLeft = OuterOutRec;
    // 如果当前 OutRec 对象的 FirstLeft 对象等于 InnerOutRec 或 OuterOutRec，则将其设置为 orfl
    else if (outRec->FirstLeft == InnerOutRec ||
             outRec->FirstLeft == OuterOutRec)
      outRec->FirstLeft = orfl;
  }
}
//----------------------------------------------------------------------
// 修正 FirstLeft，不检查 NewOutRec 是否包含多边形
void Clipper::FixupFirstLefts3(OutRec *OldOutRec, OutRec *NewOutRec) {
  // 遍历所有多边形输出记录
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i) {
    OutRec *outRec = m_PolyOuts[i];
    // 解析 FirstLeft 指向的输出记录
    OutRec *firstLeft = ParseFirstLeft(outRec->FirstLeft);
    // 如果当前输出记录有点并且 FirstLeft 指向 OldOutRec，则重新赋值 FirstLeft 为 NewOutRec
    if (outRec->Pts && firstLeft == OldOutRec)
      outRec->FirstLeft = NewOutRec;
  }
}
//----------------------------------------------------------------------

// 合并共同边
void Clipper::JoinCommonEdges() {
  // 遍历所有连接列表
  for (JoinList::size_type i = 0; i < m_Joins.size(); i++) {
    Join *join = m_Joins[i];

    // 获取连接点所在的输出记录
    OutRec *outRec1 = GetOutRec(join->OutPt1->Idx);
    OutRec *outRec2 = GetOutRec(join->OutPt2->Idx);

    // 如果其中一个输出记录没有点或者是开放的，则跳过
    if (!outRec1->Pts || !outRec2->Pts)
      continue;
    if (outRec1->IsOpen || outRec2->IsOpen)
      continue;

    // 获取具有正确孔状态（FirstLeft）的多边形片段，然后调用 JoinPoints() ...
    OutRec *holeStateRec;
    if (outRec1 == outRec2)
      holeStateRec = outRec1;
    else if (OutRec1RightOfOutRec2(outRec1, outRec2))
      holeStateRec = outRec2;
    else if (OutRec1RightOfOutRec2(outRec2, outRec1))
      holeStateRec = outRec1;
    else
      holeStateRec = GetLowermostRec(outRec1, outRec2);

    // 如果 JoinPoints() 失败，则跳过
    if (!JoinPoints(join, outRec1, outRec2))
      continue;
    // 如果两个输出记录相同
    if (outRec1 == outRec2) {
      // 代替连接两个多边形，我们通过将一个多边形分割成两个来创建一个新的多边形。
      outRec1->Pts = join->OutPt1;
      outRec1->BottomPt = 0;
      outRec2 = CreateOutRec();
      outRec2->Pts = join->OutPt2;

      // 更新所有 OutRec2.Pts Idx 的值
      UpdateOutPtIdxs(*outRec2);

      // 如果 Poly2 包含 Poly1
      if (Poly2ContainsPoly1(outRec2->Pts, outRec1->Pts)) {
        // outRec1 包含 outRec2
        outRec2->IsHole = !outRec1->IsHole;
        outRec2->FirstLeft = outRec1;

        // 修正 FirstLeft 指针
        if (m_UsingPolyTree)
          FixupFirstLefts2(outRec2, outRec1);

        // 如果 outRec2 是孔或者输出翻转后的面积大于0，则反转 PolyPt 链接
        if ((outRec2->IsHole ^ m_ReverseOutput) == (Area(*outRec2) > 0))
          ReversePolyPtLinks(outRec2->Pts);

      } else if (Poly2ContainsPoly1(outRec1->Pts, outRec2->Pts)) {
        // outRec2 包含 outRec1
        outRec2->IsHole = outRec1->IsHole;
        outRec1->IsHole = !outRec2->IsHole;
        outRec2->FirstLeft = outRec1->FirstLeft;
        outRec1->FirstLeft = outRec2;

        // 修正 FirstLeft 指针
        if (m_UsingPolyTree)
          FixupFirstLefts2(outRec1, outRec2);

        // 如果 outRec1 是孔或者输出翻转后的面积大于0，则反转 PolyPt 链接
        if ((outRec1->IsHole ^ m_ReverseOutput) == (Area(*outRec1) > 0))
          ReversePolyPtLinks(outRec1->Pts);
      } else {
        // 两个多边形完全分离
        outRec2->IsHole = outRec1->IsHole;
        outRec2->FirstLeft = outRec1->FirstLeft;

        // 修正可能需要重新分配给 OutRec2 的 FirstLeft 指针
        if (m_UsingPolyTree)
          FixupFirstLefts1(outRec1, outRec2);
      }

    } else {
      // 连接了两个多边形

      outRec2->Pts = 0;
      outRec2->BottomPt = 0;
      outRec2->Idx = outRec1->Idx;

      outRec1->IsHole = holeStateRec->IsHole;
      if (holeStateRec == outRec2)
        outRec1->FirstLeft = outRec2->FirstLeft;
      outRec2->FirstLeft = outRec1;

      // 修正 FirstLeft 指针
      if (m_UsingPolyTree)
        FixupFirstLefts3(outRec2, outRec1);
    }
  }
// ClipperOffset 支持函数...
//------------------------------------------------------------------------------

// 计算两点之间的单位法线向量
DoublePoint GetUnitNormal(const IntPoint &pt1, const IntPoint &pt2) {
  // 如果两点重合，则返回零向量
  if (pt2.X == pt1.X && pt2.Y == pt1.Y)
    return DoublePoint(0, 0);

  // 计算两点之间的距离
  double Dx = (double)(pt2.X - pt1.X);
  double dy = (double)(pt2.Y - pt1.Y);
  double f = 1 * 1.0 / std::sqrt(Dx * Dx + dy * dy);
  // 根据距离计算单位法线向量
  Dx *= f;
  dy *= f;
  return DoublePoint(dy, -Dx);
}

//------------------------------------------------------------------------------
// ClipperOffset 类
//------------------------------------------------------------------------------

// 构造函数，初始化 MiterLimit 和 ArcTolerance
ClipperOffset::ClipperOffset(double miterLimit, double arcTolerance) {
  this->MiterLimit = miterLimit;
  this->ArcTolerance = arcTolerance;
  m_lowest.X = -1;
}
//------------------------------------------------------------------------------

// 析构函数，清空数据
ClipperOffset::~ClipperOffset() { Clear(); }
//------------------------------------------------------------------------------

// 清空数据
void ClipperOffset::Clear() {
  // 释放内存
  for (int i = 0; i < m_polyNodes.ChildCount(); ++i)
    delete m_polyNodes.Childs[i];
  m_polyNodes.Childs.clear();
  m_lowest.X = -1;
}
//------------------------------------------------------------------------------

// 添加路径到多边形
void ClipperOffset::AddPath(const Path &path, JoinType joinType,
                            EndType endType) {
  int highI = (int)path.size() - 1;
  if (highI < 0)
    return;
  PolyNode *newNode = new PolyNode();
  newNode->m_jointype = joinType;
  newNode->m_endtype = endType;

  // 从路径中去除重复点，并找到最低点的索引
  if (endType == etClosedLine || endType == etClosedPolygon)
    while (highI > 0 && path[0] == path[highI])
      highI--;
  newNode->Contour.reserve(highI + 1);
  newNode->Contour.push_back(path[0]);
  int j = 0, k = 0;
  for (int i = 1; i <= highI; i++)
    // 如果新节点的轮廓点不等于路径中的点
    if (newNode->Contour[j] != path[i]) {
      // j 自增
      j++;
      // 将路径中的点添加到新节点的轮廓中
      newNode->Contour.push_back(path[i]);
      // 如果路径中的点的 Y 坐标大于新节点轮廓中第 k 个点的 Y 坐标
      // 或者 Y 坐标相等但 X 坐标小于新节点轮廓中第 k 个点的 X 坐标
      if (path[i].Y > newNode->Contour[k].Y ||
          (path[i].Y == newNode->Contour[k].Y &&
           path[i].X < newNode->Contour[k].X))
        // 更新 k 为 j
        k = j;
    }
  // 如果路径类型为封闭多边形且 j 小于 2
  if (endType == etClosedPolygon && j < 2) {
    // 删除新节点并返回
    delete newNode;
    return;
  }
  // 将新节点添加为子节点
  m_polyNodes.AddChild(*newNode);

  // 如果该路径的最低点低于所有其他点，则更新 m_lowest
  if (endType != etClosedPolygon)
    return;
  // 如果 m_lowest 的 X 坐标小于 0
  if (m_lowest.X < 0)
    // 更新 m_lowest 为当前子节点的索引和 k
    m_lowest = IntPoint(m_polyNodes.ChildCount() - 1, k);
  else {
    // 获取 m_lowest 对应的点
    IntPoint ip = m_polyNodes.Childs[(int)m_lowest.X]->Contour[(int)m_lowest.Y];
    // 如果新节点的第 k 个点的 Y 坐标大于 ip 的 Y 坐标
    // 或者 Y 坐标相等但 X 坐标小于 ip 的 X 坐标
    if (newNode->Contour[k].Y > ip.Y ||
        (newNode->Contour[k].Y == ip.Y && newNode->Contour[k].X < ip.X))
      // 更新 m_lowest 为当前子节点的索引和 k
      m_lowest = IntPoint(m_polyNodes.ChildCount() - 1, k);
  }
// 添加一组路径到偏移对象中，指定连接类型和结束类型
void ClipperOffset::AddPaths(const Paths &paths, JoinType joinType,
                             EndType endType) {
  // 遍历所有路径
  for (Paths::size_type i = 0; i < paths.size(); ++i)
    // 添加单个路径到偏移对象中
    AddPath(paths[i], joinType, endType);
}
//------------------------------------------------------------------------------

// 修正路径的方向
void ClipperOffset::FixOrientations() {
  // 如果最低顶点的 X 坐标大于等于 0 并且最低顶点所在的封闭路径方向错误
  if (m_lowest.X >= 0 &&
      !Orientation(m_polyNodes.Childs[(int)m_lowest.X]->Contour)) {
    // 遍历所有子节点
    for (int i = 0; i < m_polyNodes.ChildCount(); ++i) {
      PolyNode &node = *m_polyNodes.Childs[i];
      // 如果节点是封闭多边形或者封闭线段并且方向错误，则反转路径
      if (node.m_endtype == etClosedPolygon ||
          (node.m_endtype == etClosedLine && Orientation(node.Contour)))
        ReversePath(node.Contour);
    }
  } else {
    // 遍历所有子节点
    for (int i = 0; i < m_polyNodes.ChildCount(); ++i) {
      PolyNode &node = *m_polyNodes.Childs[i];
      // 如果节点是封闭线段并且方向错误，则反转路径
      if (node.m_endtype == etClosedLine && !Orientation(node.Contour))
        ReversePath(node.Contour);
    }
  }
}
//------------------------------------------------------------------------------

// 执行路径偏移操作
void ClipperOffset::Execute(Paths &solution, double delta) {
  // 清空解决方案路径
  solution.clear();
  // 修正路径方向
  FixOrientations();
  // 执行路径偏移
  DoOffset(delta);

  // 清理 'corners' ...
  Clipper clpr;
  // 将目标多边形添加到 Clipper 对象中
  clpr.AddPaths(m_destPolys, ptSubject, true);
  // 如果偏移距离大于 0
  if (delta > 0) {
    // 执行联合操作
    clpr.Execute(ctUnion, solution, pftPositive, pftPositive);
  } else {
    // 获取 Clipper 对象的边界矩形
    IntRect r = clpr.GetBounds();
    // 创建外部路径
    Path outer(4);
    outer[0] = IntPoint(r.left - 10, r.bottom + 10);
    outer[1] = IntPoint(r.right + 10, r.bottom + 10);
    outer[2] = IntPoint(r.right + 10, r.top - 10);
    outer[3] = IntPoint(r.left - 10, r.top - 10);

    // 将外部路径添加到 Clipper 对象中
    clpr.AddPath(outer, ptSubject, true);
    // 反转解决方案路径
    clpr.ReverseSolution(true);
    // 执行联合操作
    clpr.Execute(ctUnion, solution, pftNegative, pftNegative);
    # 如果解决方案的大小大于0，则删除第一个元素
    if (solution.size() > 0)
      solution.erase(solution.begin());
  }
// ClipperOffset 类的 Execute 方法，用于执行偏移操作并生成解决方案
void ClipperOffset::Execute(PolyTree &solution, double delta) {
  // 清空解决方案
  solution.Clear();
  // 修正多边形的方向
  FixOrientations();
  // 执行偏移操作
  DoOffset(delta);

  // 现在清理 'corners' ...
  Clipper clpr;
  // 将目标多边形添加到 Clipper 对象中作为主要路径
  clpr.AddPaths(m_destPolys, ptSubject, true);
  // 如果偏移距离大于 0
  if (delta > 0) {
    // 执行联合操作并将结果存储在解决方案中
    clpr.Execute(ctUnion, solution, pftPositive, pftPositive);
  } else {
    // 获取 Clipper 对象的边界矩形
    IntRect r = clpr.GetBounds();
    // 创建外部路径
    Path outer(4);
    outer[0] = IntPoint(r.left - 10, r.bottom + 10);
    outer[1] = IntPoint(r.right + 10, r.bottom + 10);
    outer[2] = IntPoint(r.right + 10, r.top - 10);
    outer[3] = IntPoint(r.left - 10, r.top - 10);

    // 将外部路径添加到 Clipper 对象中作为主要路径
    clpr.AddPath(outer, ptSubject, true);
    // 反转解决方案
    clpr.ReverseSolution(true);
    // 执行联合操作并将结果存储在解决方案中
    clpr.Execute(ctUnion, solution, pftNegative, pftNegative);
    // 移除外部 PolyNode 矩形
    if (solution.ChildCount() == 1 && solution.Childs[0]->ChildCount() > 0) {
      PolyNode *outerNode = solution.Childs[0];
      solution.Childs.reserve(outerNode->ChildCount());
      solution.Childs[0] = outerNode->Childs[0];
      solution.Childs[0]->Parent = outerNode->Parent;
      for (int i = 1; i < outerNode->ChildCount(); ++i)
        solution.AddChild(*outerNode->Childs[i]);
    } else
      solution.Clear();
  }
}
//------------------------------------------------------------------------------

// ClipperOffset 类的 DoOffset 方法，用于执行偏移操作
void ClipperOffset::DoOffset(double delta) {
  // 清空目标多边形
  m_destPolys.clear();
  // 设置偏移距离
  m_delta = delta;

  // 如果偏移距离接近于零，将任何封闭的多边形复制到 m_destPolys 并返回
  if (NEAR_ZERO(delta)) {
    m_destPolys.reserve(m_polyNodes.ChildCount());
    for (int i = 0; i < m_polyNodes.ChildCount(); i++) {
      PolyNode &node = *m_polyNodes.Childs[i];
      if (node.m_endtype == etClosedPolygon)
        m_destPolys.push_back(node.Contour);
    }
    return;
  }

  // 查看文档文件夹中的 offset_triginometry3.svg ...
  if (MiterLimit > 2)
    m_miterLim = 2 / (MiterLimit * MiterLimit);
  else
    // 设置最大迭代限制为0.5
    m_miterLim = 0.5;

    // 计算y的值
    double y;
    if (ArcTolerance <= 0.0)
        y = def_arc_tolerance;
    else if (ArcTolerance > std::fabs(delta) * def_arc_tolerance)
        y = std::fabs(delta) * def_arc_tolerance;
    else
        y = ArcTolerance;

    // 根据y的值计算步数
    double steps = pi / std::acos(1 - y / std::fabs(delta));
    if (steps > std::fabs(delta) * pi)
        steps = std::fabs(delta) * pi; // 检查是否过度精确
    m_sin = std::sin(two_pi / steps);
    m_cos = std::cos(two_pi / steps);
    m_StepsPerRad = steps / two_pi;
    if (delta < 0.0)
        m_sin = -m_sin;

    // 预留目标多边形的空间
    m_destPolys.reserve(m_polyNodes.ChildCount() * 2);
    for (int i = 0; i < m_polyNodes.ChildCount(); i++) {
        PolyNode &node = *m_polyNodes.Childs[i];
        m_srcPoly = node.Contour;

        int len = (int)m_srcPoly.size();
        // 如果源多边形为空或者delta小于等于0且多边形点数小于3或者结束类型不是封闭多边形，则跳过
        if (len == 0 ||
            (delta <= 0 && (len < 3 || node.m_endtype != etClosedPolygon)))
            continue;

        m_destPoly.clear();
        if (len == 1) {
            // 处理只有一个点的情况
            if (node.m_jointype == jtRound) {
                double X = 1.0, Y = 0.0;
                for (cInt j = 1; j <= steps; j++) {
                    m_destPoly.push_back(IntPoint(Round(m_srcPoly[0].X + X * delta),
                                                  Round(m_srcPoly[0].Y + Y * delta)));
                    double X2 = X;
                    X = X * m_cos - m_sin * Y;
                    Y = X2 * m_sin + Y * m_cos;
                }
            } else {
                double X = -1.0, Y = -1.0;
                for (int j = 0; j < 4; ++j) {
                    m_destPoly.push_back(IntPoint(Round(m_srcPoly[0].X + X * delta),
                                                  Round(m_srcPoly[0].Y + Y * delta)));
                    if (X < 0)
                        X = 1;
                    else if (Y < 0)
                        Y = 1;
                    else
                        X = -1;
                }
            }
            m_destPolys.push_back(m_destPoly);
            continue;
        }
        // 构建法线
        m_normals.clear();
        m_normals.reserve(len);
        for (int j = 0; j < len - 1; ++j)
            m_normals.push_back(GetUnitNormal(m_srcPoly[j], m_srcPoly[j + 1]));
    }
    // 如果节点的结束类型是封闭线段或封闭多边形，则计算并添加单位法向量到 m_normals 中
    if (node.m_endtype == etClosedLine || node.m_endtype == etClosedPolygon)
      m_normals.push_back(GetUnitNormal(m_srcPoly[len - 1], m_srcPoly[0]));
    // 否则，将上一个法向量复制并添加到 m_normals 中
    else
      m_normals.push_back(DoublePoint(m_normals[len - 2]));

    // 如果节点的结束类型是封闭多边形
    if (node.m_endtype == etClosedPolygon) {
      // 计算偏移点并添加到目标多边形中
      int k = len - 1;
      for (int j = 0; j < len; ++j)
        OffsetPoint(j, k, node.m_jointype);
      // 将目标多边形添加到目标多边形列表中
      m_destPolys.push_back(m_destPoly);
    } 
    // 如果节点的结束类型是封闭线段
    else if (node.m_endtype == etClosedLine) {
      // 计算偏移点并添加到目标多边形中
      int k = len - 1;
      for (int j = 0; j < len; ++j)
        OffsetPoint(j, k, node.m_jointype);
      // 将目标多边形添加到目标多边形列表中
      m_destPolys.push_back(m_destPoly);
      // 清空目标多边形
      m_destPoly.clear();
      // 重新构建 m_normals
      DoublePoint n = m_normals[len - 1];
      for (int j = len - 1; j > 0; j--)
        m_normals[j] = DoublePoint(-m_normals[j - 1].X, -m_normals[j - 1].Y);
      m_normals[0] = DoublePoint(-n.X, -n.Y);
      k = 0;
      for (int j = len - 1; j >= 0; j--)
        OffsetPoint(j, k, node.m_jointype);
      // 将目标多边形添加到目标多边形列表中
      m_destPolys.push_back(m_destPoly);
    } else {
      // 初始化变量 k 为 0
      int k = 0;
      // 遍历节点，计算偏移点
      for (int j = 1; j < len - 1; ++j)
        OffsetPoint(j, k, node.m_jointype);

      // 初始化变量 pt1
      IntPoint pt1;
      // 如果节点的结束类型为 etOpenButt
      if (node.m_endtype == etOpenButt) {
        // 设置 j 为 len - 1
        int j = len - 1;
        // 计算新的点 pt1
        pt1 = IntPoint((cInt)Round(m_srcPoly[j].X + m_normals[j].X * delta),
                       (cInt)Round(m_srcPoly[j].Y + m_normals[j].Y * delta));
        // 将新点添加到目标多边形
        m_destPoly.push_back(pt1);
        // 计算另一个新点 pt1
        pt1 = IntPoint((cInt)Round(m_srcPoly[j].X - m_normals[j].X * delta),
                       (cInt)Round(m_srcPoly[j].Y - m_normals[j].Y * delta));
        // 将另一个新点添加到目标多边形
        m_destPoly.push_back(pt1);
      } else {
        // 设置 j 为 len - 1
        int j = len - 1;
        // 设置 k 为 len - 2
        k = len - 2;
        // 重置 m_sinA 为 0
        m_sinA = 0;
        // 反转法线向量
        m_normals[j] = DoublePoint(-m_normals[j].X, -m_normals[j].Y);
        // 根据节点的结束类型执行不同的操作
        if (node.m_endtype == etOpenSquare)
          DoSquare(j, k);
        else
          DoRound(j, k);
      }

      // 重新构建法线向量
      for (int j = len - 1; j > 0; j--)
        m_normals[j] = DoublePoint(-m_normals[j - 1].X, -m_normals[j - 1].Y);
      m_normals[0] = DoublePoint(-m_normals[1].X, -m_normals[1].Y);

      // 设置 k 为 len - 1
      k = len - 1;
      // 遍历节点，计算偏移点
      for (int j = k - 1; j > 0; --j)
        OffsetPoint(j, k, node.m_jointype);

      // 如果节点的结束类型为 etOpenButt
      if (node.m_endtype == etOpenButt) {
        // 计算新的点 pt1
        pt1 = IntPoint((cInt)Round(m_srcPoly[0].X - m_normals[0].X * delta),
                       (cInt)Round(m_srcPoly[0].Y - m_normals[0].Y * delta));
        // 将新点添加到目标多边形
        m_destPoly.push_back(pt1);
        // 计算另一个新点 pt1
        pt1 = IntPoint((cInt)Round(m_srcPoly[0].X + m_normals[0].X * delta),
                       (cInt)Round(m_srcPoly[0].Y + m_normals[0].Y * delta));
        // 将另一个新点添加到目标多边形
        m_destPoly.push_back(pt1);
      } else {
        // 设置 k 为 1
        k = 1;
        // 重置 m_sinA 为 0
        m_sinA = 0;
        // 根据节点的结束类型执行不同的操作
        if (node.m_endtype == etOpenSquare)
          DoSquare(0, 1);
        else
          DoRound(0, 1);
      }
      // 将目标多边形添加到目标多边形集合
      m_destPolys.push_back(m_destPoly);
    }
  }
void ClipperOffset::OffsetPoint(int j, int &k, JoinType jointype) {
  // 计算两个向量的叉乘，得到正弦值
  m_sinA = (m_normals[k].X * m_normals[j].Y - m_normals[j].X * m_normals[k].Y);
  // 如果正弦值乘以偏移量的绝对值小于1.0
  if (std::fabs(m_sinA * m_delta) < 1.0) {
    // 计算两个向量的点积，得到余弦值
    double cosA =
        (m_normals[k].X * m_normals[j].X + m_normals[j].Y * m_normals[k].Y);
    // 如果余弦值大于0，表示夹角为0度
    if (cosA > 0) // angle => 0 degrees
    {
      // 将偏移后的点添加到目标多边形中
      m_destPoly.push_back(
          IntPoint(Round(m_srcPoly[j].X + m_normals[k].X * m_delta),
                   Round(m_srcPoly[j].Y + m_normals[k].Y * m_delta)));
      return;
    }
    // 否则夹角为180度
  } else if (m_sinA > 1.0)
    m_sinA = 1.0;
  else if (m_sinA < -1.0)
    m_sinA = -1.0;

  // 如果正弦值乘以偏移量小于0
  if (m_sinA * m_delta < 0) {
    // 将偏移后的点添加到目标多边形中
    m_destPoly.push_back(
        IntPoint(Round(m_srcPoly[j].X + m_normals[k].X * m_delta),
                 Round(m_srcPoly[j].Y + m_normals[k].Y * m_delta)));
    // 将当前点添加到目标多边形中
    m_destPoly.push_back(m_srcPoly[j]);
    // 将另一个偏移后的点添加到目标多边形中
    m_destPoly.push_back(
        IntPoint(Round(m_srcPoly[j].X + m_normals[j].X * m_delta),
                 Round(m_srcPoly[j].Y + m_normals[j].Y * m_delta)));
  } else
    // 根据连接类型进行处理
    switch (jointype) {
    case jtMiter: {
      // 计算尖角的比例因子
      double r = 1 + (m_normals[j].X * m_normals[k].X +
                      m_normals[j].Y * m_normals[k].Y);
      // 如果比例因子大于等于尖角限制值，执行尖角处理
      if (r >= m_miterLim)
        DoMiter(j, k, r);
      else
        // 否则执行方形处理
        DoSquare(j, k);
      break;
    }
    case jtSquare:
      // 执行方形处理
      DoSquare(j, k);
      break;
    case jtRound:
      // 执行圆角处理
      DoRound(j, k);
      break;
    }
  // 更新 k 的值为 j
  k = j;
}
// 在偏移路径中创建一个正方形拐角
void ClipperOffset::DoSquare(int j, int k) {
  // 计算斜率
  double dx = std::tan(std::atan2(m_sinA, m_normals[k].X * m_normals[j].X +
                                              m_normals[k].Y * m_normals[j].Y) /
                       4);
  // 添加第一个点到目标多边形
  m_destPoly.push_back(IntPoint(
      Round(m_srcPoly[j].X + m_delta * (m_normals[k].X - m_normals[k].Y * dx)),
      Round(m_srcPoly[j].Y +
            m_delta * (m_normals[k].Y + m_normals[k].X * dx))));
  // 添加第二个点到目标多边形
  m_destPoly.push_back(IntPoint(
      Round(m_srcPoly[j].X + m_delta * (m_normals[j].X + m_normals[j].Y * dx)),
      Round(m_srcPoly[j].Y +
            m_delta * (m_normals[j].Y - m_normals[j].X * dx))));
}
//------------------------------------------------------------------------------

// 在偏移路径中创建一个尖角
void ClipperOffset::DoMiter(int j, int k, double r) {
  // 计算比例
  double q = m_delta / r;
  // 添加点到目标多边形
  m_destPoly.push_back(
      IntPoint(Round(m_srcPoly[j].X + (m_normals[k].X + m_normals[j].X) * q),
               Round(m_srcPoly[j].Y + (m_normals[k].Y + m_normals[j].Y) * q)));
}
//------------------------------------------------------------------------------

// 在偏移路径中创建一个圆角
void ClipperOffset::DoRound(int j, int k) {
  // 计算角度
  double a = std::atan2(m_sinA, m_normals[k].X * m_normals[j].X +
                                    m_normals[k].Y * m_normals[j].Y);
  // 计算步数
  int steps = std::max((int)Round(m_StepsPerRad * std::fabs(a)), 1);

  double X = m_normals[k].X, Y = m_normals[k].Y, X2;
  // 循环添加点到目标多边形
  for (int i = 0; i < steps; ++i) {
    m_destPoly.push_back(IntPoint(Round(m_srcPoly[j].X + X * m_delta),
                                  Round(m_srcPoly[j].Y + Y * m_delta)));
    X2 = X;
    X = X * m_cos - m_sin * Y;
    Y = X2 * m_sin + Y * m_cos;
  }
  // 添加最后一个点到目标多边形
  m_destPoly.push_back(
      IntPoint(Round(m_srcPoly[j].X + m_normals[j].X * m_delta),
               Round(m_srcPoly[j].Y + m_normals[j].Y * m_delta)));
}

//------------------------------------------------------------------------------
// 其他公共函数
//------------------------------------------------------------------------------
// 执行简单多边形操作的函数
void Clipper::DoSimplePolygons() {
  // 初始化索引 i 为 0
  PolyOutList::size_type i = 0;
  // 遍历多边形输出列表
  while (i < m_PolyOuts.size()) {
    // 获取当前多边形输出对象
    OutRec *outrec = m_PolyOuts[i++];
    // 获取多边形的第一个点
    OutPt *op = outrec->Pts;
    // 如果点为空或者多边形未闭合，则继续下一次循环
    if (!op || outrec->IsOpen)
      continue;
    // 遍历多边形的每个点，直到找到重复点
    do {
      // 获取下一个点
      OutPt *op2 = op->Next;
      // 当下一个点不是起始点时
      while (op2 != outrec->Pts) {
        // 如果当前点和下一个点相同，并且下一个点的前后点不是当前点
        if ((op->Pt == op2->Pt) && op2->Next != op && op2->Prev != op) {
          // 将多边形分割成两部分
          OutPt *op3 = op->Prev;
          OutPt *op4 = op2->Prev;
          op->Prev = op4;
          op4->Next = op;
          op2->Prev = op3;
          op3->Next = op2;

          outrec->Pts = op;
          // 创建新的多边形输出对象
          OutRec *outrec2 = CreateOutRec();
          outrec2->Pts = op2;
          UpdateOutPtIdxs(*outrec2);
          // 判断哪个多边形包含另一个
          if (Poly2ContainsPoly1(outrec2->Pts, outrec->Pts)) {
            // OutRec2 被 OutRec1 包含
            outrec2->IsHole = !outrec->IsHole;
            outrec2->FirstLeft = outrec;
            if (m_UsingPolyTree)
              FixupFirstLefts2(outrec2, outrec);
          } else if (Poly2ContainsPoly1(outrec->Pts, outrec2->Pts)) {
            // OutRec1 被 OutRec2 包含
            outrec2->IsHole = outrec->IsHole;
            outrec->IsHole = !outrec2->IsHole;
            outrec2->FirstLeft = outrec->FirstLeft;
            outrec->FirstLeft = outrec2;
            if (m_UsingPolyTree)
              FixupFirstLefts2(outrec, outrec2);
          } else {
            // 两个多边形是独立的
            outrec2->IsHole = outrec->IsHole;
            outrec2->FirstLeft = outrec->FirstLeft;
            if (m_UsingPolyTree)
              FixupFirstLefts1(outrec, outrec2);
          }
          op2 = op; // 准备下一次迭代
        }
        op2 = op2->Next;
      }
      op = op->Next;
    } while (op != outrec->Pts);
  }
}
//------------------------------------------------------------------------------
# 反转给定路径 p 中的元素顺序
void ReversePath(Path &p) { std::reverse(p.begin(), p.end()); }
//------------------------------------------------------------------------------

# 反转给定路径集合 p 中每个路径的元素顺序
void ReversePaths(Paths &p) {
  for (Paths::size_type i = 0; i < p.size(); ++i)
    ReversePath(p[i]);
}
//------------------------------------------------------------------------------

# 简化多边形，将输入的多边形 in_poly 简化后存储在输出多边形集合 out_polys 中，使用指定的填充类型 fillType
void SimplifyPolygon(const Path &in_poly, Paths &out_polys,
                     PolyFillType fillType) {
  创建 Clipper 对象 c
  c.StrictlySimple(true); # 设置 Clipper 对象为严格简单模式
  将输入多边形 in_poly 添加到 Clipper 对象中作为主题路径
  c.Execute(ctUnion, out_polys, fillType, fillType); # 执行联合操作，将简化后的多边形存储在 out_polys 中
}
//------------------------------------------------------------------------------

# 简化多边形集合，将输入的多边形集合 in_polys 简化后存储在输出多边形集合 out_polys 中，使用指定的填充类型 fillType
void SimplifyPolygons(const Paths &in_polys, Paths &out_polys,
                      PolyFillType fillType) {
  创建 Clipper 对象 c
  c.StrictlySimple(true); # 设置 Clipper 对象为严格简单模式
  将输入多边形集合 in_polys 添加到 Clipper 对象中作为主题路径集合
  c.Execute(ctUnion, out_polys, fillType, fillType); # 执行联合操作，将简化后的多边形集合存储在 out_polys 中
}
//------------------------------------------------------------------------------

# 简化多边形集合，将输入的多边形集合 polys 简化后存储在输出多边形集合 polys 中，使用指定的填充类型 fillType
void SimplifyPolygons(Paths &polys, PolyFillType fillType) {
  调用 SimplifyPolygons 函数，输入输出多边形集合均为 polys
  SimplifyPolygons(polys, polys, fillType);
}
//------------------------------------------------------------------------------

# 计算两个整数点 pt1 和 pt2 之间的距离的平方
inline double DistanceSqrd(const IntPoint &pt1, const IntPoint &pt2) {
  计算两点在 X 轴上的距离
  double Dx = ((double)pt1.X - pt2.X);
  计算两点在 Y 轴上的距离
  double dy = ((double)pt1.Y - pt2.Y);
  返回两点距离的平方
  return (Dx * Dx + dy * dy);
}
//------------------------------------------------------------------------------
// 计算点到直线的距离的平方
double DistanceFromLineSqrd(const IntPoint &pt, const IntPoint &ln1,
                            const IntPoint &ln2) {
  // 一般形式的直线方程 (Ax + By + C = 0)
  // 给定两点 (x1, y1) & (x2, y2) 的方程是 ...
  // (y1 - y2)x + (x2 - x1)y + (y2 - y1)x1 - (x2 - x1)y1 = 0
  // A = (y1 - y2); B = (x2 - x1); C = (y2 - y1)x1 - (x2 - x1)y1
  // 点 (x, y) 到直线的垂直距离 = (Ax + By + C) / Sqrt(A^2 + B^2)
  // 参考 http://en.wikipedia.org/wiki/Perpendicular_distance
  double A = double(ln1.Y - ln2.Y);
  double B = double(ln2.X - ln1.X);
  double C = A * ln1.X + B * ln1.Y;
  C = A * pt.X + B * pt.Y - C;
  return (C * C) / (A * A + B * B);
}
//---------------------------------------------------------------------------

// 判断三个点是否共线
bool SlopesNearCollinear(const IntPoint &pt1, const IntPoint &pt2,
                         const IntPoint &pt3, double distSqrd) {
  // 当几何上处于另外两点之间的点被测试距离时，此函数更准确。
  // 即更有可能检测到“尖峰”...
  if (Abs(pt1.X - pt2.X) > Abs(pt1.Y - pt2.Y)) {
    if ((pt1.X > pt2.X) == (pt1.X < pt3.X))
      return DistanceFromLineSqrd(pt1, pt2, pt3) < distSqrd;
    else if ((pt2.X > pt1.X) == (pt2.X < pt3.X))
      return DistanceFromLineSqrd(pt2, pt1, pt3) < distSqrd;
    else
      return DistanceFromLineSqrd(pt3, pt1, pt2) < distSqrd;
  } else {
    if ((pt1.Y > pt2.Y) == (pt1.Y < pt3.Y))
      return DistanceFromLineSqrd(pt1, pt2, pt3) < distSqrd;
    else if ((pt2.Y > pt1.Y) == (pt2.Y < pt3.Y))
      return DistanceFromLineSqrd(pt2, pt1, pt3) < distSqrd;
    else
      return DistanceFromLineSqrd(pt3, pt1, pt2) < distSqrd;
  }
}
//------------------------------------------------------------------------------

// 判断两点是否距离很近
bool PointsAreClose(IntPoint pt1, IntPoint pt2, double distSqrd) {
  double Dx = (double)pt1.X - pt2.X;
  double dy = (double)pt1.Y - pt2.Y;
  return ((Dx * Dx) + (dy * dy) <= distSqrd);
}
// 从链表中排除指定节点
OutPt *ExcludeOp(OutPt *op) {
  // 保存要排除节点的前一个节点
  OutPt *result = op->Prev;
  // 调整前后节点的指针，排除当前节点
  result->Next = op->Next;
  op->Next->Prev = result;
  // 重置节点的索引值
  result->Idx = 0;
  // 返回排除节点后的链表
  return result;
}
//------------------------------------------------------------------------------

void CleanPolygon(const Path &in_poly, Path &out_poly, double distance) {
  // distance = proximity in units/pixels below which vertices
  // will be stripped. Default ~= sqrt(2).
  
  // 获取输入多边形的大小
  size_t size = in_poly.size();

  // 如果输入多边形为空，则清空输出多边形并返回
  if (size == 0) {
    out_poly.clear();
    return;
  }

  // 创建一个包含相同大小的 OutPt 数组
  OutPt *outPts = new OutPt[size];
  // 初始化 OutPt 数组
  for (size_t i = 0; i < size; ++i) {
    outPts[i].Pt = in_poly[i];
    outPts[i].Next = &outPts[(i + 1) % size];
    outPts[i].Next->Prev = &outPts[i];
    outPts[i].Idx = 0;
  }

  // 计算距离的平方
  double distSqrd = distance * distance;
  // 从第一个节点开始遍历链表
  OutPt *op = &outPts[0];
  while (op->Idx == 0 && op->Next != op->Prev) {
    // 检查当前节点与前一个节点的距离是否小于阈值，如果是则排除当前节点
    if (PointsAreClose(op->Pt, op->Prev->Pt, distSqrd)) {
      op = ExcludeOp(op);
      size--;
    } else if (PointsAreClose(op->Prev->Pt, op->Next->Pt, distSqrd)) {
      // 检查前一个节点与后一个节点的距离是否小于阈值，如果是则排除前一个和当前节点
      ExcludeOp(op->Next);
      op = ExcludeOp(op);
      size -= 2;
    } else if (SlopesNearCollinear(op->Prev->Pt, op->Pt, op->Next->Pt, distSqrd)) {
      // 检查三点是否共线，如果是则排除当前节点
      op = ExcludeOp(op);
      size--;
    } else {
      // 标记当前节点，继续遍历
      op->Idx = 1;
      op = op->Next;
    }
  }

  // 如果剩余节点数小于3，则将大小设为0
  if (size < 3)
    size = 0;
  // 调整输出多边形的大小
  out_poly.resize(size);
  // 将剩余节点的坐标复制到输出多边形中
  for (size_t i = 0; i < size; ++i) {
    out_poly[i] = op->Pt;
    op = op->Next;
  }
  // 释放 OutPt 数组的内存
  delete[] outPts;
}
//------------------------------------------------------------------------------

void CleanPolygon(Path &poly, double distance) {
  // 调用 CleanPolygon 函数，输入输出多边形相同
  CleanPolygon(poly, poly, distance);
}
//------------------------------------------------------------------------------

void CleanPolygons(const Paths &in_polys, Paths &out_polys, double distance) {
  // 调整输出多边形数组的大小
  out_polys.resize(in_polys.size());
  // 遍历输入多边形数组
  for (Paths::size_type i = 0; i < in_polys.size(); ++i)
    # 调用 CleanPolygon 函数，传入输入多边形列表中第 i 个多边形、输出多边形列表中第 i 个多边形以及距离参数
    CleanPolygon(in_polys[i], out_polys[i], distance);
// 清理多边形集合，根据给定距离进行清理
void CleanPolygons(Paths &polys, double distance) {
  // 调用重载函数 CleanPolygons，传入相同的多边形集合和距离参数
  CleanPolygons(polys, polys, distance);
}
//------------------------------------------------------------------------------

// 计算 Minkowski 和运算结果
void Minkowski(const Path &poly, const Path &path, Paths &solution, bool isSum,
               bool isClosed) {
  // 根据是否封闭多边形设置增量值
  int delta = (isClosed ? 1 : 0);
  // 获取多边形和路径的点数
  size_t polyCnt = poly.size();
  size_t pathCnt = path.size();
  // 创建临时路径集合
  Paths pp;
  pp.reserve(pathCnt);
  // 根据是否求和进行不同的计算
  if (isSum)
    for (size_t i = 0; i < pathCnt; ++i) {
      Path p;
      p.reserve(polyCnt);
      for (size_t j = 0; j < poly.size(); ++j)
        p.push_back(IntPoint(path[i].X + poly[j].X, path[i].Y + poly[j].Y));
      pp.push_back(p);
    }
  else
    for (size_t i = 0; i < pathCnt; ++i) {
      Path p;
      p.reserve(polyCnt);
      for (size_t j = 0; j < poly.size(); ++j)
        p.push_back(IntPoint(path[i].X - poly[j].X, path[i].Y - poly[j].Y));
      pp.push_back(p);
    }

  // 清空结果集合
  solution.clear();
  solution.reserve((pathCnt + delta) * (polyCnt + 1));
  // 构建 Minkowski 和运算结果
  for (size_t i = 0; i < pathCnt - 1 + delta; ++i)
    for (size_t j = 0; j < polyCnt; ++j) {
      Path quad;
      quad.reserve(4);
      quad.push_back(pp[i % pathCnt][j % polyCnt]);
      quad.push_back(pp[(i + 1) % pathCnt][j % polyCnt]);
      quad.push_back(pp[(i + 1) % pathCnt][(j + 1) % polyCnt]);
      quad.push_back(pp[i % pathCnt][(j + 1) % polyCnt]);
      if (!Orientation(quad))
        ReversePath(quad);
      solution.push_back(quad);
    }
}
//------------------------------------------------------------------------------

// 计算 Minkowski 和运算结果的和
void MinkowskiSum(const Path &pattern, const Path &path, Paths &solution,
                  bool pathIsClosed) {
  // 调用 Minkowski 函数，传入模式和路径，设置为求和模式
  Minkowski(pattern, path, solution, true, pathIsClosed);
  // 创建 Clipper 对象
  Clipper c;
  // 将结果集合添加到 Clipper 中作为主要对象
  c.AddPaths(solution, ptSubject, true);
  // 执行联合操作，将结果保存到 solution 中
  c.Execute(ctUnion, solution, pftNonZero, pftNonZero);
}
//------------------------------------------------------------------------------
// 将输入路径按照给定的偏移量进行平移，结果保存在输出路径中
void TranslatePath(const Path &input, Path &output, const IntPoint delta) {
  // 前提条件：输入路径不等于输出路径
  output.resize(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    output[i] = IntPoint(input[i].X + delta.X, input[i].Y + delta.Y);
}
//------------------------------------------------------------------------------

// 计算 Minkowski 和，将结果保存在 solution 中
void MinkowskiSum(const Path &pattern, const Paths &paths, Paths &solution,
                  bool pathIsClosed) {
  Clipper c;
  for (size_t i = 0; i < paths.size(); ++i) {
    Paths tmp;
    // 计算 Minkowski 和
    Minkowski(pattern, paths[i], tmp, true, pathIsClosed);
    c.AddPaths(tmp, ptSubject, true);
    if (pathIsClosed) {
      Path tmp2;
      // 将路径进行平移
      TranslatePath(paths[i], tmp2, pattern[0]);
      c.AddPath(tmp2, ptClip, true);
    }
  }
  // 执行合并操作
  c.Execute(ctUnion, solution, pftNonZero, pftNonZero);
}
//------------------------------------------------------------------------------

// 计算 Minkowski 差，将结果保存在 solution 中
void MinkowskiDiff(const Path &poly1, const Path &poly2, Paths &solution) {
  Minkowski(poly1, poly2, solution, false, true);
  Clipper c;
  c.AddPaths(solution, ptSubject, true);
  c.Execute(ctUnion, solution, pftNonZero, pftNonZero);
}
//------------------------------------------------------------------------------

// 节点类型枚举
enum NodeType { ntAny, ntOpen, ntClosed };

// 将 PolyNode 添加到 Paths 中
void AddPolyNodeToPaths(const PolyNode &polynode, NodeType nodetype,
                        Paths &paths) {
  bool match = true;
  if (nodetype == ntClosed)
    match = !polynode.IsOpen();
  else if (nodetype == ntOpen)
    return;

  if (!polynode.Contour.empty() && match)
    paths.push_back(polynode.Contour);
  for (int i = 0; i < polynode.ChildCount(); ++i)
    AddPolyNodeToPaths(*polynode.Childs[i], nodetype, paths);
}
//------------------------------------------------------------------------------

// 将 PolyTree 转换为 Paths
void PolyTreeToPaths(const PolyTree &polytree, Paths &paths) {
  paths.resize(0);
  paths.reserve(polytree.Total());
  AddPolyNodeToPaths(polytree, ntAny, paths);
}
// 从 PolyTree 中提取所有闭合路径，并存储在 paths 中
void ClosedPathsFromPolyTree(const PolyTree &polytree, Paths &paths) {
  // 清空 paths 中的内容
  paths.resize(0);
  // 预留足够的空间以容纳 polytree 中的路径
  paths.reserve(polytree.Total());
  // 将 PolyNode 中的闭合路径添加到 paths 中
  AddPolyNodeToPaths(polytree, ntClosed, paths);
}
//------------------------------------------------------------------------------

// 从 PolyTree 中提取所有开放路径，并存储在 paths 中
void OpenPathsFromPolyTree(PolyTree &polytree, Paths &paths) {
  // 清空 paths 中的内容
  paths.resize(0);
  // 预留足够的空间以容纳 polytree 中的路径
  paths.reserve(polytree.Total());
  // 开放路径只存在于顶层，因此...
  for (int i = 0; i < polytree.ChildCount(); ++i)
    // 如果子节点是开放路径，则将其添加到 paths 中
    if (polytree.Childs[i]->IsOpen())
      paths.push_back(polytree.Childs[i]->Contour);
}
//------------------------------------------------------------------------------

// 重载运算符 <<，用于打印 IntPoint 对象
std::ostream &operator<<(std::ostream &s, const IntPoint &p) {
  // 打印 IntPoint 对象的坐标
  s << "(" << p.X << "," << p.Y << ")";
  return s;
}
//------------------------------------------------------------------------------

// 重载运算符 <<，用于打印 Path 对象
std::ostream &operator<<(std::ostream &s, const Path &p) {
  // 如果路径为空，则直接返回
  if (p.empty())
    return s;
  // 打印路径中每个点的坐标
  Path::size_type last = p.size() - 1;
  for (Path::size_type i = 0; i < last; i++)
    s << "(" << p[i].X << "," << p[i].Y << "), ";
  s << "(" << p[last].X << "," << p[last].Y << ")\n";
  return s;
}
//------------------------------------------------------------------------------

// 重载运算符 <<，用于打印 Paths 对象
std::ostream &operator<<(std::ostream &s, const Paths &p) {
  // 打印每个路径
  for (Paths::size_type i = 0; i < p.size(); i++)
    s << p[i];
  s << "\n";
  return s;
}
//------------------------------------------------------------------------------

} // ClipperLib namespace
```