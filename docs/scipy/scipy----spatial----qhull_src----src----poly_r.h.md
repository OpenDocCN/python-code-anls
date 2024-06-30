# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\poly_r.h`

```
/*<html><pre>  -<a                             href="qh-poly_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly_r.h
   header file for poly_r.c and poly2_r.c

   see qh-poly_r.htm, libqhull_r.h and poly_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/poly_r.h#3 $$Change: 2701 $
   $DateTime: 2019/06/25 15:24:47 $$Author: bbarber $
*/

#ifndef qhDEFpoly
#define qhDEFpoly 1

#include "libqhull_r.h"

/*===============   constants ========================== */

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="ALGORITHMfault">-</a>

  qh_ALGORITHMfault
    use as argument to checkconvex() to report errors during buildhull
*/
#define qh_ALGORITHMfault 0

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="DATAfault">-</a>

  qh_DATAfault
    use as argument to checkconvex() to report errors during initialhull
*/
#define qh_DATAfault 1

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="DUPLICATEridge">-</a>

  qh_DUPLICATEridge
    special value for facet->neighbor to indicate a duplicate ridge

  notes:
    set by qh_matchneighbor for qh_matchdupridge
*/
#define qh_DUPLICATEridge (facetT *)1L

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="MERGEridge">-</a>

  qh_MERGEridge       flag in facet
    special value for facet->neighbor to indicate a duplicate ridge that needs merging

  notes:
    set by qh_matchnewfacets..qh_matchdupridge from qh_DUPLICATEridge
    used by qh_mark_dupridges to set facet->mergeridge, facet->mergeridge2 from facet->dupridge
*/
#define qh_MERGEridge (facetT *)2L


/*============ -structures- ====================*/

/*=========== -macros- =========================*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLfacet_">-</a>

  FORALLfacet_( facetlist ) { ... }
    assign 'facet' to each facet in facetlist

  notes:
    uses 'facetT *facet;'
    assumes last facet is a sentinel

  see:
    FORALLfacets
*/
#define FORALLfacet_( facetlist ) if (facetlist) for ( facet=(facetlist); facet && facet->next; facet= facet->next )

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLnew_facets">-</a>

  FORALLnew_facets { ... }
    assign 'newfacet' to each facet in qh.newfacet_list

  notes:
    uses 'facetT *newfacet;'
    at exit, newfacet==NULL
*/
#define FORALLnew_facets for ( newfacet=qh->newfacet_list; newfacet && newfacet->next; newfacet=newfacet->next )

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLvertex_">-</a>

  FORALLvertex_( vertexlist ) { ... }
    assign 'vertex' to each vertex in vertexlist

  notes:
    uses 'vertexT *vertex;'
*/


注释：
    at exit, vertex==NULL


// 在退出时，检查顶点是否为 NULL
/*
#define FORALLvertex_( vertexlist ) for (vertex=( vertexlist );vertex && vertex->next;vertex= vertex->next )

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLvisible_facets">-</a>

  FORALLvisible_facets { ... }
    assign 'visible' to each visible facet in qh.visible_list

  notes:
    uses 'facetT *visible;'
    at exit, visible==NULL
*/
#define FORALLvisible_facets for (visible=qh->visible_list; visible && visible->visible; visible= visible->next)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLsame_">-</a>

  FORALLsame_( newfacet ) { ... }
    assign 'same' to each facet in newfacet->f.samecycle

  notes:
    uses 'facetT *same;'
    stops when it returns to newfacet
*/
#define FORALLsame_(newfacet) for (same= newfacet->f.samecycle; same != newfacet; same= same->f.samecycle)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLsame_cycle_">-</a>

  FORALLsame_cycle_( newfacet ) { ... }
    assign 'same' to each facet in newfacet->f.samecycle

  notes:
    uses 'facetT *same;'
    at exit, same == NULL
*/
#define FORALLsame_cycle_(newfacet) \
     for (same= newfacet->f.samecycle; \
         same; same= (same == newfacet ?  NULL : same->f.samecycle))

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighborA_">-</a>

  FOREACHneighborA_( facet ) { ... }
    assign 'neighborA' to each neighbor in facet->neighbors

  FOREACHneighborA_( vertex ) { ... }
    assign 'neighborA' to each neighbor in vertex->neighbors

  declare:
    facetT *neighborA, **neighborAp;

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHneighborA_(facet)  FOREACHsetelement_(facetT, facet->neighbors, neighborA)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvisible_">-</a>

  FOREACHvisible_( facets ) { ... }
    assign 'visible' to each facet in facets

  notes:
    uses 'facetT *facet, *facetp;'
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvisible_(facets) FOREACHsetelement_(facetT, facets, visible)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHnewfacet_">-</a>

  FOREACHnewfacet_( facets ) { ... }
    assign 'newfacet' to each facet in facets

  notes:
    uses 'facetT *newfacet, *newfacetp;'
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHnewfacet_(facets) FOREACHsetelement_(facetT, facets, newfacet)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertexA_">-</a>

  FOREACHvertexA_( vertices ) { ... }
    assign 'vertexA' to each vertex in vertices

  notes:
*/
    # 声明了两个指针变量 vertexA 和 vertexAp，它们都指向类型为 vertexT 的对象
    uses 'vertexT *vertexA, *vertexAp;'
    # 参考 qset_r.h 文件中的 FOREACHsetelement_ 宏定义，具体内容可以查看该文件中对应部分
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
/*
#define FOREACHvertexA_(vertices) FOREACHsetelement_(vertexT, vertices, vertexA)
*/

/*
  -<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertexreverse12_">-</a>

  FOREACHvertexreverse12_( vertices ) { ... }
    assign 'vertex' to each vertex in vertices
    reverse order of first two vertices

  notes:
    uses 'vertexT *vertex, *vertexp;'
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertexreverse12_(vertices) FOREACHsetelementreverse12_(vertexT, vertices, vertex)

/*
=============== prototypes poly_r.c in alphabetical order ================
*/

#ifdef __cplusplus
extern "C" {
#endif

/*
  void qh_appendfacet(qhT *qh, facetT *facet);
    Append a facet to the Qhull data structure.

  void qh_appendvertex(qhT *qh, vertexT *vertex);
    Append a vertex to the Qhull data structure.

  void qh_attachnewfacets(qhT *qh /* qh.visible_list, qh.newfacet_list */);
    Attach newly created facets to the Qhull data structure.

  boolT qh_checkflipped(qhT *qh, facetT *facet, realT *dist, boolT allerror);
    Check if a facet is flipped based on the given distance.

  void qh_delfacet(qhT *qh, facetT *facet);
    Delete a facet from the Qhull data structure.

  void qh_deletevisible(qhT *qh /* qh.visible_list, qh.horizon_list */);
    Delete visible facets from the Qhull data structure.

  setT *qh_facetintersect(qhT *qh, facetT *facetA, facetT *facetB, int *skipAp,int *skipBp, int extra);
    Compute the intersection of two facets.

  int qh_gethash(qhT *qh, int hashsize, setT *set, int size, int firstindex, void *skipelem);
    Compute a hash value for a set in Qhull data structure.

  facetT *qh_getreplacement(qhT *qh, facetT *visible);
    Get a replacement facet from the Qhull data structure.

  facetT *qh_makenewfacet(qhT *qh, setT *vertices, boolT toporient, facetT *facet);
    Create a new facet with given vertices and orientation.

  void qh_makenewplanes(qhT *qh /* qh.newfacet_list */);
    Create new hyperplanes for newly created facets.

  facetT *qh_makenew_nonsimplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew);
    Create a new nonsimplicial facet.

  facetT *qh_makenew_simplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew);
    Create a new simplicial facet.

  void qh_matchneighbor(qhT *qh, facetT *newfacet, int newskip, int hashsize,
                          int *hashcount);
    Match neighbors for a new facet.

  coordT qh_matchnewfacets(qhT *qh);
    Match new facets in the Qhull data structure.

  boolT qh_matchvertices(qhT *qh, int firstindex, setT *verticesA, int skipA,
                          setT *verticesB, int *skipB, boolT *same);
    Match vertices in the Qhull data structure.

  facetT *qh_newfacet(qhT *qh);
    Allocate memory for a new facet.

  ridgeT *qh_newridge(qhT *qh);
    Allocate memory for a new ridge.

  int qh_pointid(qhT *qh, pointT *point);
    Get the ID of a point in Qhull data structure.

  void qh_removefacet(qhT *qh, facetT *facet);
    Remove a facet from the Qhull data structure.

  void qh_removevertex(qhT *qh, vertexT *vertex);
    Remove a vertex from the Qhull data structure.

  void qh_update_vertexneighbors(qhT *qh);
    Update vertex neighbors in the Qhull data structure.

  void qh_update_vertexneighbors_cone(qhT *qh);
    Update vertex neighbors using a cone in the Qhull data structure.
*/

/*
========== -prototypes poly2_r.c in alphabetical order ===========
*/

boolT qh_addfacetvertex(qhT *qh, facetT *facet, vertexT *newvertex);
void qh_addhash(void *newelem, setT *hashtable, int hashsize, int hash);
void qh_check_bestdist(qhT *qh);
void qh_check_maxout(qhT *qh);
void qh_check_output(qhT *qh);
void qh_check_point(qhT *qh, pointT *point, facetT *facet, realT *maxoutside, realT *maxdist, facetT **errfacet1, facetT **errfacet2, int *errcount);
void qh_check_points(qhT *qh);
void qh_checkconvex(qhT *qh, facetT *facetlist, int fault);
void qh_checkfacet(qhT *qh, facetT *facet, boolT newmerge, boolT *waserrorp);
void qh_checkflipped_all(qhT *qh, facetT *facetlist);
boolT qh_checklists(qhT *qh, facetT *facetlist);
*/
// 检查多边形 facetlist 中的几何正确性
void qh_checkpolygon(qhT *qh, facetT *facetlist);

// 检查顶点 vertex 的几何正确性，可以选择进行所有检查，并通过 waserrorp 返回是否有错误
void qh_checkvertex(qhT *qh, vertexT *vertex, boolT allchecks, boolT *waserrorp);

// 清除 qh 中心点相关数据结构的内容，类型由 type 指定
void qh_clearcenters(qhT *qh, qh_CENTER type);

// 根据 vertices 创建一个简单形的凸包
void qh_createsimplex(qhT *qh, setT *vertices);

// 删除 ridge 对象，并清理相关数据结构
void qh_delridge(qhT *qh, ridgeT *ridge);

// 删除 vertex 对象，并清理相关数据结构
void qh_delvertex(qhT *qh, vertexT *vertex);

// 根据 facet 返回其关联的 vertex 集合
setT *qh_facet3vertex(qhT *qh, facetT *facet);

// 查找包含给定 point 的最佳 facet，并返回距离和是否在外面
facetT *qh_findbestfacet(qhT *qh, pointT *point, boolT bestoutside, realT *bestdist, boolT *isoutside);

// 在给定的 upperfacet 下查找最佳的 facet，返回距离和分区数量
facetT *qh_findbestlower(qhT *qh, facetT *upperfacet, pointT *point, realT *bestdistp, int *numpart);

// 查找包含给定 point 的 facet，可以选择不包含 upper 界限，并返回距离和是否在外面
facetT *qh_findfacet_all(qhT *qh, pointT *point, boolT noupper, realT *bestdist, boolT *isoutside, int *numpart);

// 在 facetlist 中查找好的 facet，返回数量
int qh_findgood(qhT *qh, facetT *facetlist, int goodhorizon);

// 在 facetlist 中查找所有好的 facet
void qh_findgood_all(qhT *qh, facetT *facetlist);

// 找到 qh.facet_list 中距离最远的 facet，并进行标记
void qh_furthestnext(qhT *qh /* qh.facet_list */);

// 将 facet 标记为外部 facet
void qh_furthestout(qhT *qh, facetT *facet);

// 检测 facet 中的无限循环，并进行修正
void qh_infiniteloop(qhT *qh, facetT *facet);

// 初始化创建凸包数据结构
void qh_initbuild(qhT *qh);

// 初始化创建凸包，使用给定的顶点集合
void qh_initialhull(qhT *qh, setT *vertices);

// 初始化创建顶点集合，返回集合对象
setT *qh_initialvertices(qhT *qh, int dim, setT *maxpoints, pointT *points, int numpoints);

// 检查 point 是否是 vertices 集合中的一个顶点，并返回该顶点对象
vertexT *qh_isvertex(pointT *point, setT *vertices);

// 根据 point 创建新的 facet，并返回新的顶点对象
vertexT *qh_makenewfacets(qhT *qh, pointT *point /* qh.horizon_list, visible_list */);

// 匹配重复的 ridge，并返回 hash 值
coordT qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount);

// 检查 qh.facet_list 中近似共面的 facet
void qh_nearcoplanar(qhT *qh /* qh.facet_list */);

// 在 facet 中查找距离 point 最近的顶点，并返回最近距离
vertexT *qh_nearvertex(qhT *qh, facetT *facet, pointT *point, realT *bestdistp);

// 创建一个新的哈希表，并返回新的哈希表大小
int qh_newhashtable(qhT *qh, int newsize);

// 根据 point 创建新的顶点，并返回新的顶点对象
vertexT *qh_newvertex(qhT *qh, pointT *point);

// 在 3D 空间中查找下一个 ridge，并返回顶点对象
ridgeT *qh_nextridge3d(ridgeT *atridge, facetT *facet, vertexT **vertexp);

// 返回 facetA 的相邻 neighbor 中与 facetA 不同的顶点
vertexT *qh_opposite_vertex(qhT *qh, facetT *facetA, facetT *neighbor);

// 检查 qh.facet_list 中外部共面的 facet
void qh_outcoplanar(qhT *qh /* qh.facet_list */);

// 根据 id 返回 qh 点集合中的点对象
pointT *qh_point(qhT *qh, int id);

// 将 point 添加到集合 set 中，并关联 elem
void qh_point_add(qhT *qh, setT *set, pointT *point, void *elem);

// 返回 qh.facet_list 中与点相关联的 facet 集合
setT *qh_pointfacet(qhT *qh /* qh.facet_list */);

// 返回 qh.facet_list 中与点相关联的 vertex 集合
setT *qh_pointvertex(qhT *qh /* qh.facet_list */);

// 将 facet 添加到 facetlist 的开头
void qh_prependfacet(qhT *qh, facetT *facet, facetT **facetlist);

// 打印哈希表中的内容到文件 fp
void qh_printhashtable(qhT *qh, FILE *fp);

// 打印 qh 中的数据结构列表
void qh_printlists(qhT *qh);

// 替换 facet 中的 oldvertex 为 newvertex
void qh_replacefacetvertex(qhT *qh, facetT *facet, vertexT *oldvertex, vertexT *newvertex);

// 重置 qh 中的列表，可以选择是否重置统计和可见性
void qh_resetlists(qhT *qh, boolT stats, boolT resetVisible /* qh.newvertex_list qh.newfacet_list qh.visible_list */);

// 设置 qh 中所有 Voronoi 图相关的信息
void qh_setvoronoi_all(qhT *qh);

// 对 qh.facet_list 中的 facet 进行三角化
void qh_triangulate(qhT *qh /* qh.facet_list */);

// 对 facetA 进行三角化，并返回第一个顶点
void qh_triangulate_facet(qhT *qh, facetT *facetA, vertexT **first_vertex);

// 对两个相邻的 facet 进行连接的三角化处理
void qh_triangulate_link(qhT *qh, facetT *oldfacetA, facetT *facetA, facetT *oldfacetB, facetT *facetB);

// 对 facetA 和 facetB 之间进行镜像的三角化处理
void qh_triangulate_mirror(qhT *qh, facetT *facetA, facetT *facetB);

// 对 facetA 进行空间中的三角化处理
void qh_triangulate_null(qhT *qh, facetT *facetA);

// 返回 vertexsetA 和 vertexsetB 的交集，并返回交集集合对象
void qh_vertexintersect(qhT *qh, setT **vertexsetA, setT *vertexsetB);

// 返回 vertexsetA 和 vertexsetB 的交集，并返回新的交集集合对象
setT *qh_vertexintersect_new(qhT *qh, setT *vertexsetA, setT *vertexsetB);
// 声明一个名为 qh_vertexneighbors 的函数，该函数接受一个名为 qh 的参数，参数类型为 qhT*，用于处理顶点邻居关系
void qh_vertexneighbors(qhT *qh /* qh.facet_list */);

// 声明一个名为 qh_vertexsubset 的函数，该函数接受两个参数，类型为 setT*，用于判断一个顶点集合是否是另一个顶点集合的子集
boolT qh_vertexsubset(setT *vertexsetA, setT *vertexsetB);

// 如果是 C++ 环境，则结束 extern "C" 块
#ifdef __cplusplus
} /* extern "C" */
#endif

// 结束 qhDEFpoly 头文件的条件编译
#endif /* qhDEFpoly */
```