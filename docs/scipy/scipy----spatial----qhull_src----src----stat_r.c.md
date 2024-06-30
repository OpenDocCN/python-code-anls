# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\stat_r.c`

```
/*<html><pre>  -<a                             href="qh-stat_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   stat_r.c
   contains all statistics that are collected for qhull

   see qh-stat_r.htm and stat_r.h

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/stat_r.c#7 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*========== functions in alphabetic order ================*/

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="allstatA">-</a>

  qh_allstatA()
    define statistics in groups of 20

  notes:
    (otherwise, 'gcc -O2' uses too much memory)
    uses qhstat.next
*/
void qh_allstatA(qhT *qh) {

   /* zdef_(type,name,doc,average) */
   zzdef_(zdoc, Zdoc2, "precision statistics", -1);
   zdef_(zinc, Znewvertex, NULL, -1);
   zdef_(wadd, Wnewvertex, "ave. distance of a new vertex to a facet", Znewvertex);
   zzdef_(wmax, Wnewvertexmax, "max. distance of a new vertex to a facet", -1);
   zdef_(wmax, Wvertexmax, "max. distance of an output vertex to a facet", -1);
   zdef_(wmin, Wvertexmin, "min. distance of an output vertex to a facet", -1);
   zdef_(wmin, Wmindenom, "min. denominator in hyperplane computation", -1);

   qh->qhstat.precision= qh->qhstat.next;  /* usually call qh_joggle_restart, printed if Q0 or QJn */
   zzdef_(zdoc, Zdoc3, "precision problems (corrected unless 'Q0' or an error)", -1);
   zzdef_(zinc, Zcoplanarridges, "coplanar half ridges in output", -1);
   zzdef_(zinc, Zconcaveridges, "concave half ridges in output", -1);
   zzdef_(zinc, Zflippedfacets, "flipped facets", -1);
   zzdef_(zinc, Zcoplanarhorizon, "coplanar horizon facets for new vertices", -1);
   zzdef_(zinc, Zcoplanarpart, "coplanar points during partitioning", -1);
   zzdef_(zinc, Zminnorm, "degenerate hyperplanes recomputed with gaussian elimination", -1);
   zzdef_(zinc, Znearlysingular, "nearly singular or axis-parallel hyperplanes", -1);
   zzdef_(zinc, Zback0, "zero divisors during back substitute", -1);
   zzdef_(zinc, Zgauss0, "zero divisors during gaussian elimination", -1);
   zzdef_(zinc, Zmultiridge, "dupridges with multiple neighbors", -1);
   zzdef_(zinc, Zflipridge, "dupridges with flip facet into good neighbor", -1);
   zzdef_(zinc, Zflipridge2, "dupridges with flip facet into good flip neighbor", -1);
}
void qh_allstatB(qhT *qh) {
  // 定义并初始化一个用于存储摘要信息的全局变量 zdoc
  zzdef_(zdoc, Zdoc1, "summary information", -1);
  // 定义并初始化一个用于存储输出顶点数量的全局变量 zinc
  zdef_(zinc, Zvertices, "number of vertices in output", -1);
  // 定义并初始化一个用于存储输出中面片数量的全局变量 zinc
  zdef_(zinc, Znumfacets, "number of facets in output", -1);
  // 定义并初始化一个用于存储输出中非单纯形面片数量的全局变量 zinc
  zdef_(zinc, Znonsimplicial, "number of non-simplicial facets in output", -1);
  // 定义并初始化一个用于存储曾经非单纯形但现在是单纯形的面片数量的全局变量 zinc
  zdef_(zinc, Znowsimplicial, "simplicial facets that were non-simplicial", -1);
  // 定义并初始化一个用于存储输出中脊的数量的全局变量 zinc
  zdef_(zinc, Znumridges, "number of ridges in output", -1);
  // 定义并初始化一个用于存储平均每个面片的脊数量的全局变量 zadd
  zdef_(zadd, Znumridges, "average number of ridges per facet", Znumfacets);
  // 定义并初始化一个用于存储最大脊数量的全局变量 zmax
  zdef_(zmax, Zmaxridges, "maximum number of ridges", -1);
  // 定义并初始化一个用于存储平均每个面片的邻居数量的全局变量 zadd
  zdef_(zadd, Znumneighbors, "average number of neighbors per facet", Znumfacets);
  // 定义并初始化一个用于存储最大邻居数量的全局变量 zmax
  zdef_(zmax, Zmaxneighbors, "maximum number of neighbors", -1);
  // 定义并初始化一个用于存储平均每个面片的顶点数量的全局变量 zadd
  zdef_(zadd, Znumvertices, "average number of vertices per facet", Znumfacets);
  // 定义并初始化一个用于存储最大顶点数量的全局变量 zmax
  zdef_(zmax, Zmaxvertices, "maximum number of vertices", -1);
  // 定义并初始化一个用于存储平均每个顶点的邻居数量的全局变量 zadd
  zdef_(zadd, Znumvneighbors, "average number of neighbors per vertex", Zvertices);
  // 定义并初始化一个用于存储最大顶点邻居数量的全局变量 zmax
  zdef_(zmax, Zmaxvneighbors, "maximum number of neighbors", -1);
  // 定义并初始化一个用于存储输入后 Qhull 的 CPU 秒数的全局变量 wadd
  zdef_(wadd, Wcpu, "cpu seconds for qhull after input", -1);
  // 定义并初始化一个用于存储总共创建顶点数量的全局变量 zinc
  zdef_(zinc, Ztotvertices, "vertices created altogether", -1);
  // 定义并初始化一个用于存储总共创建面片数量的全局变量 zzdef_
  zzdef_(zinc, Zsetplane, "facets created altogether", -1);
  // 定义并初始化一个用于存储总共创建脊数量的全局变量 zinc
  zdef_(zinc, Ztotridges, "ridges created altogether", -1);
  // 定义并初始化一个用于存储后处理合并前面片数量的全局变量 zinc
  zdef_(zinc, Zpostfacets, "facets before post merge", -1);
  // 定义并初始化一个用于存储平均每个面片的合并次数的全局变量 zadd
  zdef_(zadd, Znummergetot, "average merges per facet (at most 511)", Znumfacets);
  // 定义并初始化一个用于存储单个面片最大合并次数的全局变量 zmax
  zdef_(zmax, Znummergemax, "  maximum merges for a facet (at most 511)", -1);
  // 定义并初始化一个用于存储角度的全局变量 zinc
  zdef_(zinc, Zangle, NULL, -1);
  // 定义并初始化一个用于存储所有脊的法向量角度的平均余弦值的全局变量 wadd
  zdef_(wadd, Wangle, "average cosine (angle) of facet normals for all ridges", Zangle);
  // 定义并初始化一个用于存储所有脊中最大法向量角度的余弦值的全局变量 wmax
  zdef_(wmax, Wanglemax, "  maximum cosine of facet normals (flatest) across a ridge", -1);
  // 定义并初始化一个用于存储所有脊中最小法向量角度的余弦值的全局变量 wmin
  zdef_(wmin, Wanglemin, "  minimum cosine of facet normals (sharpest) across a ridge", -1);
  // 定义并初始化一个用于存储所有面片的总面积的全局变量 wadd
  zdef_(wadd, Wareatot, "total area of facets", -1);
  // 定义并初始化一个用于存储最大面片面积的全局变量 wmax
  zdef_(wmax, Wareamax, "  maximum facet area", -1);
  // 定义并初始化一个用于存储最小面片面积的全局变量 wmin
  zdef_(wmin, Wareamin, "  minimum facet area", -1);
}
void qh_allstatC(qhT *qh) {
    // 定义并初始化统计信息文本，标识为 "build hull statistics"，未指定索引 (-1)
    zdef_(zdoc, Zdoc9, "build hull statistics", -1);
    // 定义并初始化递增统计信息，标识为 "points processed"，未指定索引 (-1)
    zzdef_(zinc, Zprocessed, "points processed", -1);
    // 定义并初始化递增统计信息，标识为 "retries due to precision problems"，未指定索引 (-1)
    zzdef_(zinc, Zretry, "retries due to precision problems", -1);
    // 定义并初始化统计信息文本，标识为 "  max. random joggle"，未指定索引 (-1)
    zdef_(wmax, Wretrymax, "  max. random joggle", -1);
    // 定义并初始化统计信息文本，标识为 "max. vertices at any one time"，未指定索引 (-1)
    zdef_(zmax, Zmaxvertex, "max. vertices at any one time", -1);
    // 定义并初始化递增统计信息，标识为 "ave. visible facets per iteration"，依赖于 Zprocessed
    zdef_(zinc, Ztotvisible, "ave. visible facets per iteration", Zprocessed);
    // 定义并初始化递增统计信息，标识为 "  ave. visible facets without an horizon neighbor"，依赖于 Zprocessed
    zdef_(zinc, Zinsidevisible, "  ave. visible facets without an horizon neighbor", Zprocessed);
    // 定义并初始化增加统计信息，标识为 "  ave. facets deleted per iteration"，依赖于 Zprocessed
    zdef_(zadd, Zvisfacettot,  "  ave. facets deleted per iteration", Zprocessed);
    // 定义并初始化统计信息文本，标识为 "    maximum"，未指定索引 (-1)
    zdef_(zmax, Zvisfacetmax,  "    maximum", -1);
    // 定义并初始化增加统计信息，标识为 "ave. visible vertices per iteration"，依赖于 Zprocessed
    zdef_(zadd, Zvisvertextot, "ave. visible vertices per iteration", Zprocessed);
    // 定义并初始化统计信息文本，标识为 "    maximum"，未指定索引 (-1)
    zdef_(zmax, Zvisvertexmax, "    maximum", -1);
    // 定义并初始化递增统计信息，标识为 "ave. horizon facets per iteration"，依赖于 Zprocessed
    zdef_(zinc, Ztothorizon, "ave. horizon facets per iteration", Zprocessed);
    // 定义并初始化增加统计信息，标识为 "ave. new or merged facets per iteration"，依赖于 Zprocessed
    zdef_(zadd, Znewfacettot,  "ave. new or merged facets per iteration", Zprocessed);
    // 定义并初始化统计信息文本，标识为 "    maximum (includes initial simplex)"，未指定索引 (-1)
    zdef_(zmax, Znewfacetmax,  "    maximum (includes initial simplex)", -1);
    // 定义并初始化增加统计信息，标识为 "average new facet balance"，依赖于 Zprocessed
    zdef_(wadd, Wnewbalance, "average new facet balance", Zprocessed);
    // 定义并初始化统计信息文本，标识为 "  standard deviation"，未指定索引 (-1)
    zdef_(wadd, Wnewbalance2, "  standard deviation", -1);
    // 定义并初始化增加统计信息，标识为 "average partition balance"，依赖于 Zpbalance
    zdef_(wadd, Wpbalance, "average partition balance", Zpbalance);
    // 定义并初始化统计信息文本，标识为 "  standard deviation"，未指定索引 (-1)
    zdef_(wadd, Wpbalance2, "  standard deviation", -1);
    // 定义并初始化递增统计信息，标识为 "  count"，未指定索引 (-1)
    zdef_(zinc, Zpbalance,  "  count", -1);
    // 定义并初始化递增统计信息，标识为 "searches of all points for initial simplex"，未指定索引 (-1)
    zdef_(zinc, Zsearchpoints, "searches of all points for initial simplex", -1);
    // 定义并初始化递增统计信息，标识为 "determinants for facet area"，未指定索引 (-1)
    zdef_(zinc, Zdetfacetarea, "determinants for facet area", -1);
    // 定义并初始化递增统计信息，标识为 "  determinants not computed because vertex too low"，未指定索引 (-1)
    zdef_(zinc, Znoarea, "  determinants not computed because vertex too low", -1);
    // 定义并初始化递增统计信息，标识为 "determinants for initial hull or voronoi vertices"，未指定索引 (-1)
    zdef_(zinc, Zdetsimplex, "determinants for initial hull or voronoi vertices", -1);
    // 定义并初始化递增统计信息，标识为 "points ignored (!above max_outside)"，未指定索引 (-1)
    zdef_(zinc, Znotmax, "points ignored (!above max_outside)", -1);
    // 定义并初始化递增统计信息，标识为 "points ignored (pinched apex)"，未指定索引 (-1)
    zdef_(zinc, Zpinchedapex, "points ignored (pinched apex)", -1);
    // 定义并初始化递增统计信息，标识为 "points ignored (!above a good facet)"，未指定索引 (-1)
    zdef_(zinc, Znotgood, "points ignored (!above a good facet)", -1);
    // 定义并初始化递增统计信息，标识为 "points ignored (didn't create a good new facet)"，未指定索引 (-1)
    zdef_(zinc, Znotgoodnew, "points ignored (didn't create a good new facet)", -1);
    // 定义并初始化递增统计信息，标识为 "good facets found"，未指定索引 (-1)
    zdef_(zinc, Zgoodfacet, "good facets found", -1);
    // 定义并初始化递增统计信息，标识为 "distance tests for facet visibility"，未指定索引 (-1)
    zzdef_(zinc, Znumvisibility, "distance tests for facet visibility", -1);
    // 定义并初始化递增统计信息，标识为 "distance tests to report minimum vertex"，未指定索引 (-1)
    zdef_(zinc, Zdistvertex, "distance tests to report minimum vertex", -1);
    // 定义并初始化递增统计信息，标识为 "points checked for facets' outer planes"，未指定索引 (-1)
    zzdef_(zinc, Ztotcheck, "points checked for facets' outer planes", -1);
    // 定义并初始化递增统计信息，标识为 "  ave. distance tests per check"，依赖于 Ztotcheck
    zzdef_(zinc, Zcheckpart, "  ave. distance tests per check", Ztotcheck);
}
void qh_allstatD(qhT *qh) {
  // 定义增量统计器，名称为 zinc，用于重置 visit_id，初始值为 -1
  zdef_(zinc, Zvisit, "resets of visit_id", -1);
  // 定义增量统计器，名称为 zinc，用于重置 vertex_visit，初始值为 -1
  zdef_(zinc, Zvvisit, "  resets of vertex_visit", -1);
  // 定义最大值统计器，名称为 zmax，用于记录 visit_id 的最大值除以 2，初始值为 -1
  zdef_(zmax, Zvisit2max, "  max visit_id/2", -1);
  // 定义最大值统计器，名称为 zmax，用于记录 vertex_visit 的最大值除以 2，初始值为 -1
  zdef_(zmax, Zvvisit2max, "  max vertex_visit/2", -1);

  // 定义文档类型统计器，名称为 zdoc，用于分区统计，初始值为 -1
  zdef_(zdoc, Zdoc4, "partitioning statistics (see previous for outer planes)", -1);
  // 定义增量统计器，名称为 zzadd，用于记录总删除的顶点数，初始值为 -1
  zzdef_(zadd, Zdelvertextot, "total vertices deleted", -1);
  // 定义最大值统计器，名称为 zmax，用于记录每次迭代删除的最大顶点数，初始值为 -1
  zdef_(zmax, Zdelvertexmax, "    maximum vertices deleted per iteration", -1);
  // 定义增量统计器，名称为 zinc，用于记录调用 findbest 函数的次数，初始值为 -1
  zdef_(zinc, Zfindbest, "calls to findbest", -1);
  // 定义增量统计器，名称为 zadd，用于记录平均测试的面片数，初始值为 Zfindbest
  zdef_(zadd, Zfindbesttot, " ave. facets tested", Zfindbest);
  // 定义最大值统计器，名称为 zmax，用于记录最大测试的面片数，初始值为 -1
  zdef_(zmax, Zfindbestmax, " max. facets tested", -1);
  // 定义增量统计器，名称为 zinc，用于记录调用 findbestnew 函数的次数，初始值为 -1
  zdef_(zinc, Zfindnew, "calls to findbestnew", -1);
  // 定义增量统计器，名称为 zadd，用于记录平均测试的面片数，初始值为 Zfindnew
  zdef_(zadd, Zfindnewtot, " ave. facets tested", Zfindnew);
  // 定义最大值统计器，名称为 zmax，用于记录最大测试的面片数，初始值为 -1
  zdef_(zmax, Zfindnewmax, " max. facets tested", -1);
  // 定义增量统计器，名称为 zinc，用于记录调用 findhorizon 函数的次数，初始值为 -1
  zdef_(zinc, Zfindhorizon, "calls to findhorizon", -1);
  // 定义增量统计器，名称为 zadd，用于记录平均测试的面片数，初始值为 Zfindhorizon
  zdef_(zadd, Zfindhorizontot, " ave. facets tested", Zfindhorizon);
  // 定义最大值统计器，名称为 zmax，用于记录最大测试的面片数，初始值为 -1
  zdef_(zmax, Zfindhorizonmax, " max. facets tested", -1);
  // 定义增量统计器，名称为 zinc，用于记录在 findhorizon 中平均测试到明显更好的面片数，初始值为 Zfindhorizon
  zdef_(zinc, Zfindjump, " ave. clearly better", Zfindhorizon);
  // 定义增量统计器，名称为 zinc，用于记录在 qh_findbesthorizon 中新的最佳面片数，初始值为 -1
  zdef_(zinc, Znewbesthorizon, " new bestfacets during qh_findbesthorizon", -1);
  // 定义增量统计器，名称为 zinc，用于角度测试的重分区的共面点数，初始值为 -1
  zdef_(zinc, Zpartangle, "angle tests for repartitioned coplanar points", -1);
  // 定义增量统计器，名称为 zinc，用于在一个角面上重分区的共面点数，初始值为 -1
  zdef_(zinc, Zpartcorner, "  repartitioned coplanar points above a corner facet", -1);
  // 定义增量统计器，名称为 zinc，用于在一个隐藏面上重分区的共面点数，初始值为 -1
  zdef_(zinc, Zparthidden, "  repartitioned coplanar points above a hidden facet", -1);
  // 定义增量统计器，名称为 zinc，用于在一个扭曲面上重分区的共面点数，初始值为 -1
  zdef_(zinc, Zparttwisted, "  repartitioned coplanar points above a twisted facet", -1);
}
void qh_allstatE(qhT *qh) {
  // 定义并初始化计数器 zinc，用于记录内部点的数量
  zdef_(zinc, Zpartinside, "inside points", -1);
  // 定义并初始化计数器 zinc，记录保留在面片内附近的内部点的数量
  zdef_(zinc, Zpartnear, "  near inside points kept with a facet", -1);
  // 定义并初始化计数器 zinc，记录与面片共面的内部点的数量
  zdef_(zinc, Zcoplanarinside, "  inside points that were coplanar with a facet", -1);
  // 定义并初始化计数器 zinc，记录调用 findbestlower 函数的次数
  zdef_(zinc, Zbestlower, "calls to findbestlower", -1);
  // 定义并初始化计数器 zinc，记录带有搜索顶点邻居的 findbestlower 函数调用次数
  zdef_(zinc, Zbestlowerv, "  with search of vertex neighbors", -1);
  // 定义并初始化计数器 zinc，记录带有罕见搜索所有面片的 findbestlower 函数调用次数
  zdef_(zinc, Zbestlowerall, "  with rare search of all facets", -1);
  // 定义并初始化计数器 zmax，记录单次搜索所有面片的最大面片数量
  zdef_(zmax, Zbestloweralln, "  facets per search of all facets", -1);
  // 定义并初始化计数器 wadd，记录在最终检查时 max_outside 的差异
  zdef_(wadd, Wmaxout, "difference in max_outside at final check", -1);
  // 定义并初始化计数器 zinc，记录初始分区的距离测试数量
  zzdef_(zinc, Zpartitionall, "distance tests for initial partition", -1);
  // 定义并初始化计数器 zinc，记录每个点的分区数量
  zdef_(zinc, Ztotpartition, "partitions of a point", -1);
  // 定义并初始化计数器 zinc，记录分区操作的距离测试数量
  zzdef_(zinc, Zpartition, "distance tests for partitioning", -1);
  // 定义并初始化计数器 zinc，记录翻转面片检查的距离测试数量
  zzdef_(zinc, Zdistcheck, "distance tests for checking flipped facets", -1);
  // 定义并初始化计数器 zinc，记录检查凸性的距离测试数量
  zzdef_(zinc, Zdistconvex, "distance tests for checking convexity", -1);
  // 定义并初始化计数器 zinc，记录检查良好点的距离测试数量
  zdef_(zinc, Zdistgood, "distance tests for checking good point", -1);
  // 定义并初始化计数器 zinc，记录输出时的距离测试数量
  zdef_(zinc, Zdistio, "distance tests for output", -1);
  // 定义并初始化计数器 zinc，记录统计时的距离测试数量
  zdef_(zinc, Zdiststat, "distance tests for statistics", -1);
  // 定义并初始化计数器 zinc，记录总的距离测试数量
  zzdef_(zinc, Zdistplane, "total number of distance tests", -1);
  // 定义并初始化计数器 zinc，记录共面点或已删除顶点的分区数量
  zdef_(zinc, Ztotpartcoplanar, "partitions of coplanar points or deleted vertices", -1);
  // 定义并初始化计数器 zzdef，记录为这些分区执行的距离测试数量
  zzdef_(zinc, Zpartcoplanar, "   distance tests for these partitions", -1);
  // 定义并初始化计数器 zinc，记录计算最远点时的距离测试数量
  zdef_(zinc, Zcomputefurthest, "distance tests for computing furthest", -1);
}

void qh_allstatE2(qhT *qh) {
  // 定义并初始化计数器 zdoc，记录匹配脊柱的统计信息
  zdef_(zdoc, Zdoc5, "statistics for matching ridges", -1);
  // 定义并初始化计数器 zinc，记录新面片匹配脊柱的总查找次数
  zdef_(zinc, Zhashlookup, "total lookups for matching ridges of new facets", -1);
  // 定义并初始化计数器 zinc，记录匹配脊柱的平均测试次数
  zdef_(zinc, Zhashtests, "average number of tests to match a ridge", Zhashlookup);
  // 定义并初始化计数器 zinc，记录子脊柱（重复和边界）的总查找次数
  zdef_(zinc, Zhashridge, "total lookups of subridges (duplicates and boundary)", -1);
  // 定义并初始化计数器 zinc，记录每个子脊柱的平均测试次数
  zdef_(zinc, Zhashridgetest, "average number of tests per subridge", Zhashridge);
  // 定义并初始化计数器 zinc，记录同一合并周期内的重复脊柱数量
  zdef_(zinc, Zdupsame, "duplicated ridges in same merge cycle", -1);
  // 定义并初始化计数器 zinc，记录翻转面片的重复脊柱数量
  zdef_(zinc, Zdupflip, "duplicated ridges with flipped facets", -1);

  // 定义并初始化计数器 zdoc，记录确定合并操作的统计信息
  zdef_(zdoc, Zdoc6, "statistics for determining merges", -1);
  // 定义并初始化计数器 zinc，记录用于脊柱凸性计算的角度数量
  zdef_(zinc, Zangletests, "angles computed for ridge convexity", -1);
  // 定义并初始化计数器 zinc，记录在最佳合并中使用 centrum 而非顶点的次数
  zdef_(zinc, Zbestcentrum, "best merges used centrum instead of vertices",-1);
  // 定义并初始化计数器 zzdef，记录最佳合并的距离测试数量
  zzdef_(zinc, Zbestdist, "distance tests for best merge", -1);
  // 定义并初始化计数器 zzdef，记录用于 centrum 凸性检查的距离测试数量
  zzdef_(zinc, Zcentrumtests, "distance tests for centrum convexity", -1);
  // 定义并初始化计数器 zzdef，记录用于顶点凸性检查的距离测试数量
  zzdef_(zinc, Zvertextests, "distance tests for vertex convexity", -1);
  // 定义并初始化计数器 zzdef，记录用于检查单纯凸性的距离测试数量
  zzdef_(zinc, Zdistzero, "distance tests for checking simplicial convexity", -1);
  // 定义并初始化计数器 zinc，记录在 getmergeset 中的共面角度数量
  zdef_(zinc, Zcoplanarangle, "coplanar angles in getmergeset", -1);
  // 定义并初始化计数器 zinc，记录在 getmergeset 中的共面 centrum 或顶点数量
  zdef_(zinc, Zcoplanarcentrum, "coplanar centrums or vertices in getmergeset", -1);
  // 定义并初始化计数器 zinc，记录在 getmergeset 中的凹陷脊柱数量
  zdef_(zinc, Zconcaveridge, "concave ridges in getmergeset", -1);
  // 定义并初始化计数器 zinc，记录在 getmergeset 中的凹陷共面脊柱数量
  zdef_(zinc, Zconcavecoplanarridge, "concave-coplanar ridges in getmergeset", -1);
  // 定义并初始化计数器 zinc，记录在 getmergeset 中的扭曲脊柱数量
  zdef_(zinc, Ztwistedridge, "
void qh_allstatF(qhT *qh) {
  // 定义并初始化统计数据项，用于合并过程的统计信息
  zdef_(zdoc, Zdoc7, "statistics for merging", -1);
  // 定义并初始化统计数据项，记录合并迭代次数
  zdef_(zinc, Zpremergetot, "merge iterations", -1);
  // 定义并初始化统计数据项，记录每次迭代初始非凸边缘的平均数量
  zdef_(zadd, Zmergeinittot, "ave. initial non-convex ridges per iteration", Zpremergetot);
  // 定义并初始化统计数据项，记录每次迭代初始非凸边缘的最大数量
  zdef_(zadd, Zmergeinitmax, "  maximum", -1);
  // 定义并初始化统计数据项，记录每次迭代额外添加的非凸边缘的平均数量
  zdef_(zadd, Zmergesettot, "  ave. additional non-convex ridges per iteration", Zpremergetot);
  // 定义并初始化统计数据项，记录每次迭代额外添加的非凸边缘的最大数量
  zdef_(zadd, Zmergesetmax, "  maximum additional in one pass", -1);
  // 定义并初始化统计数据项，记录后合并过程中初始非凸边缘的数量
  zdef_(zadd, Zmergeinittot2, "initial non-convex ridges for post merging", -1);
  // 定义并初始化统计数据项，记录后合并过程中额外添加的非凸边缘的数量
  zdef_(zadd, Zmergesettot2, "  additional non-convex ridges", -1);
  // 定义并初始化统计数据项，记录顶点或共面点相对于面的最大距离（考虑舍入误差）
  zdef_(wmax, Wmaxoutside, "max distance of vertex or coplanar point above facet (w/roundoff)", -1);
  // 定义并初始化统计数据项，记录顶点相对于面的最大距离（或舍入误差）
  zdef_(wmin, Wminvertex, "max distance of vertex below facet (or roundoff)", -1);
  // 定义并初始化统计数据项，记录因合并宽面而冻结的中心点数量
  zdef_(zinc, Zwidefacet, "centrums frozen due to a wide merge", -1);
  // 定义并初始化统计数据项，记录因额外顶点而冻结的中心点数量
  zdef_(zinc, Zwidevertices, "centrums frozen due to extra vertices", -1);
  // 定义并初始化统计数据项，记录合并的总面数或面循环数
  zzdef_(zinc, Ztotmerge, "total number of facets or cycles of facets merged", -1);
  // 定义并初始化统计数据项，记录合并的简单形
  zdef_(zinc, Zmergesimplex, "merged a simplex", -1);
  // 定义并初始化统计数据项，记录合并到共面视图的单纯形
  zdef_(zinc, Zonehorizon, "simplices merged into coplanar horizon", -1);
  // 定义并初始化统计数据项，记录合并到共面视图的面循环数
  zzdef_(zinc, Zcyclehorizon, "cycles of facets merged into coplanar horizon", -1);
  // 定义并初始化统计数据项，记录合并到共面视图的面循环中的平均面数
  zzdef_(zadd, Zcyclefacettot, "  ave. facets per cycle", Zcyclehorizon);
  // 定义并初始化统计数据项，记录合并到共面视图的面循环中的最大面数
  zdef_(zmax, Zcyclefacetmax, "  max. facets", -1);
  // 定义并初始化统计数据项，记录合并到共面视图的新面数
  zdef_(zinc, Zmergeintocoplanar, "new facets merged into coplanar horizon", -1);
  // 定义并初始化统计数据项，记录合并到视图的新面数
  zdef_(zinc, Zmergeintohorizon, "new facets merged into horizon", -1);
  // 定义并初始化统计数据项，记录新合并的面数
  zdef_(zinc, Zmergenew, "new facets merged", -1);
  // 定义并初始化统计数据项，记录合并到新面的视图的视图面数
  zdef_(zinc, Zmergehorizon, "horizon facets merged into new facets", -1);
  // 定义并初始化统计数据项，记录合并时被删除的顶点数
  zdef_(zinc, Zmergevertex, "vertices deleted by merging", -1);
  // 定义并初始化统计数据项，记录合并到共面视图的顶点数
  zdef_(zinc, Zcyclevertex, "vertices deleted by merging into coplanar horizon", -1);
  // 定义并初始化统计数据项，记录因退化面而删除的顶点数
  zdef_(zinc, Zdegenvertex, "vertices deleted by degenerate facet", -1);
  // 定义并初始化统计数据项，记录因重复边缘中的翻转面而进行的合并次数
  zdef_(zinc, Zmergeflipdup, "merges due to flipped facets in duplicated ridge", -1);
  // 定义并初始化统计数据项，记录因冗余邻居而进行的合并次数
  zdef_(zinc, Zredundant, "merges due to redundant neighbors", -1);
  // 定义并初始化统计数据项，记录由于非简单合并检查而进行的冗余合并次数
  zdef_(zinc, Zredundantmerge, "  detected by qh_test_nonsimplicial_merge instead of qh_test_redundant_neighbors", -1);
  // 定义并初始化统计数据项，记录非凸顶点邻居的测试数
  zdef_(zadd, Ztestvneighbor, "non-convex vertex neighbors", -1);
}
// 定义宏函数 zdef_，用于设置统计信息，并创建相关变量
void qh_allstatG(qhT *qh) {
  // 定义 zinc 变量，用于记录各种类型的合并次数，初始化为 0
  zdef_(zinc, Zacoplanar, "merges due to angle coplanar facets", -1);
  // 定义 wadd 变量，用于记录平均合并距离，初始化为 Zacoplanar 类型
  zdef_(wadd, Wacoplanartot, "  average merge distance", Zacoplanar);
  // 定义 wmax 变量，用于记录最大合并距离，初始化为 -1
  zdef_(wmax, Wacoplanarmax, "  maximum merge distance", -1);
  
  // 类似地定义下列变量，用于记录不同类型的合并次数、平均合并距离和最大合并距离
  zdef_(zinc, Zcoplanar, "merges due to coplanar facets", -1);
  zdef_(wadd, Wcoplanartot, "  average merge distance", Zcoplanar);
  zdef_(wmax, Wcoplanarmax, "  maximum merge distance", -1);
  
  zdef_(zinc, Zconcave, "merges due to concave facets", -1);
  zdef_(wadd, Wconcavetot, "  average merge distance", Zconcave);
  zdef_(wmax, Wconcavemax, "  maximum merge distance", -1);
  
  zdef_(zinc, Zconcavecoplanar, "merges due to concave-coplanar facets", -1);
  zdef_(wadd, Wconcavecoplanartot, "  average merge distance", Zconcavecoplanar);
  zdef_(wmax, Wconcavecoplanarmax, "  maximum merge distance", -1);
  
  zdef_(zinc, Zavoidold, "coplanar/concave merges due to avoiding old merge", -1);
  zdef_(wadd, Wavoidoldtot, "  average merge distance", Zavoidold);
  zdef_(wmax, Wavoidoldmax, "  maximum merge distance", -1);
  
  zdef_(zinc, Zdegen, "merges due to degenerate facets", -1);
  zdef_(wadd, Wdegentot, "  average merge distance", Zdegen);
  zdef_(wmax, Wdegenmax, "  maximum merge distance", -1);
  
  zdef_(zinc, Zflipped, "merges due to removing flipped facets", -1);
  zdef_(wadd, Wflippedtot, "  average merge distance", Zflipped);
  zdef_(wmax, Wflippedmax, "  maximum merge distance", -1);
  
  zdef_(zinc, Zduplicate, "merges due to dupridges", -1);
  zdef_(wadd, Wduplicatetot, "  average merge distance", Zduplicate);
  zdef_(wmax, Wduplicatemax, "  maximum merge distance", -1);
  
  zdef_(zinc, Ztwisted, "merges due to twisted facets", -1);
  zdef_(wadd, Wtwistedtot, "  average merge distance", Ztwisted);
  zdef_(wmax, Wtwistedmax, "  maximum merge distance", -1);
}
void qh_allstatH(qhT *qh) {
  // 定义并初始化统计数据项，用于记录顶点合并的统计信息
  zdef_(zdoc, Zdoc8, "statistics for vertex merges", -1);
  // 定义并初始化统计数据项，用于记录因为重复的尖角脊而合并的顶点
  zzdef_(zinc, Zpinchduplicate, "merge pinched vertices for a duplicate ridge", -1);
  // 定义并初始化统计数据项，用于记录因为尖角脊而合并的顶点
  zzdef_(zinc, Zpinchedvertex, "merge pinched vertices for a dupridge", -1);
  // 定义并初始化统计数据项，用于记录因为两个面共享顶点而重命名的顶点
  zdef_(zinc, Zrenameshare, "renamed vertices shared by two facets", -1);
  // 定义并初始化统计数据项，用于记录因为一个尖锐面而重命名的顶点
  zdef_(zinc, Zrenamepinch, "renamed vertices in a pinched facet", -1);
  // 定义并初始化统计数据项，用于记录因为多个面共享顶点而重命名的顶点
  zdef_(zinc, Zrenameall, "renamed vertices shared by multiple facets", -1);
  // 定义并初始化统计数据项，用于记录因为重复的脊导致的重命名失败
  zdef_(zinc, Zfindfail, "rename failures due to duplicated ridges", -1);
  // 定义并初始化统计数据项，用于记录发现的新顶点在脊中的数量
  zdef_(zinc, Znewvertexridge, "  found new vertex in ridge", -1);
  // 定义并初始化统计数据项，用于记录因为重命名的顶点而删除的脊
  zdef_(zinc, Zdelridge, "deleted ridges due to renamed vertices", -1);
  // 定义并初始化统计数据项，用于记录因为重命名的顶点而丢弃的邻居面
  zdef_(zinc, Zdropneighbor, "dropped neighbors due to renamed vertices", -1);
  // 定义并初始化统计数据项，用于记录因为重命名的顶点而合并的退化面
  zdef_(zinc, Zdropdegen, "merge degenerate facets due to dropped neighbors", -1);
  // 定义并初始化统计数据项，用于记录因为没有邻居而删除的面
  zdef_(zinc, Zdelfacetdup, "  facets deleted because of no neighbors", -1);
  // 定义并初始化统计数据项，用于记录因为没有脊而从面中移除的顶点
  zdef_(zinc, Zremvertex, "vertices removed from facets due to no ridges", -1);
  // 定义并初始化统计数据项，用于记录因为顶点删除而删除的顶点数目
  zdef_(zinc, Zremvertexdel, "  deleted", -1);
  // 定义并初始化统计数据项，用于记录在合并尖角脊后重新尝试添加点的次数
  zdef_(zinc, Zretryadd, "retry qh_addpoint after merge pinched vertex", -1);
  // 定义并初始化统计数据项，用于记录因为重复脊而合并的顶点的总数
  zdef_(zadd, Zretryaddtot, "  tot. merge pinched vertex due to dupridge", -1);
  // 定义并初始化统计数据项，用于记录在一个 qh_addpoint 操作中最多合并的顶点数目
  zdef_(zmax, Zretryaddmax, "  max. merge pinched vertex for a qh_addpoint", -1);
  // 定义并初始化统计数据项，用于记录顶点交叉以定位冗余顶点的次数
  zdef_(zinc, Zintersectnum, "vertex intersections for locating redundant vertices", -1);
  // 定义并初始化统计数据项，用于记录无法找到冗余顶点的交叉点数
  zdef_(zinc, Zintersectfail, "intersections failed to find a redundant vertex", -1);
  // 定义并初始化统计数据项，用于记录发现的冗余顶点的交叉点数
  zdef_(zinc, Zintersect, "intersections found redundant vertices", -1);
  // 定义并初始化统计数据项，用于记录每个顶点平均找到的冗余顶点数目
  zdef_(zadd, Zintersecttot, "   ave. number found per vertex", Zintersect);
  // 定义并初始化统计数据项，用于记录每个顶点最大找到的冗余顶点数目
  zdef_(zmax, Zintersectmax, "   max. found for a vertex", -1);
  // 定义并初始化统计数据项，用于记录顶点脊的数量
  zdef_(zinc, Zvertexridge, NULL, -1);
  // 定义并初始化统计数据项，用于记录每个测试顶点平均拥有的脊的数量
  zdef_(zadd, Zvertexridgetot, "  ave. number of ridges per tested vertex", Zvertexridge);
  // 定义并初始化统计数据项，用于记录每个测试顶点最大拥有的脊的数量
  zdef_(zmax, Zvertexridgemax, "  max. number of ridges per tested vertex", -1);

  // 定义并初始化统计数据项，用于记录内存使用统计信息（单位：字节）
  zdef_(zdoc, Zdoc10, "memory usage statistics (in bytes)", -1);
  // 定义并初始化统计数据项，用于记录面和它们的法线、邻居和顶点集合的内存使用
  zdef_(zadd, Zmemfacets, "for facets and their normals, neighbor and vertex sets", -1);
  // 定义并初始化统计数据项，用于记录顶点及其邻居集合的内存使用
  zdef_(zadd, Zmemvertices, "for vertices and their neighbor sets", -1);
  // 定义并初始化统计数据项，用于记录输入点、外部点和共面点集合及 qhT 结构的内存使用
  zdef_(zadd, Zmempoints, "for input points, outside and coplanar sets, and qhT", -1);
  // 定义并初始化统计数据项，用于记录脊及其顶点集合的内存使用
  zdef_(zadd, Zmemridges, "for ridges and their vertex sets", -1);
} /* allstat */
/* 设置 qhstat 结构体中 vridges 字段的值为 qhstat 结构体中 next 字段的值 */
qh->qhstat.vridges= qh->qhstat.next; /* printed in qh_produce_output2 if non-zero Zridge or Zridgemid */

/* 定义和初始化名称为 zdoc 的统计项，描述为“Voronoi ridge statistics”，无特定的顺序 */
zzdef_(zdoc, Zdoc11, "Voronoi ridge statistics", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“non-simplicial Voronoi vertices for all ridges”，依赖于 Zridge */
zzdef_(zinc, Zridge, "non-simplicial Voronoi vertices for all ridges", -1);

/* 定义和初始化名称为 wadd 的统计项，描述为“ave. distance to ridge”，依赖于 Zridge */
zzdef_(wadd, Wridge, "  ave. distance to ridge", Zridge);

/* 定义和初始化名称为 wmax 的统计项，描述为“max. distance to ridge”，无特定的顺序 */
zzdef_(wmax, Wridgemax, "  max. distance to ridge", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“bounded ridges”，依赖于 Zridgemid */
zzdef_(zinc, Zridgemid, "bounded ridges", -1);

/* 定义和初始化名称为 wadd 的统计项，描述为“ave. distance of midpoint to ridge”，依赖于 Zridgemid */
zzdef_(wadd, Wridgemid, "  ave. distance of midpoint to ridge", Zridgemid);

/* 定义和初始化名称为 wmax 的统计项，描述为“max. distance of midpoint to ridge”，无特定的顺序 */
zzdef_(wmax, Wridgemidmax, "  max. distance of midpoint to ridge", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“bounded ridges with ok normal”，无特定的顺序 */
zzdef_(zinc, Zridgeok, "bounded ridges with ok normal", -1);

/* 定义和初始化名称为 wadd 的统计项，描述为“ave. angle to ridge”，依赖于 Zridgeok */
zzdef_(wadd, Wridgeok, "  ave. angle to ridge", Zridgeok);

/* 定义和初始化名称为 wmax 的统计项，描述为“max. angle to ridge”，无特定的顺序 */
zzdef_(wmax, Wridgeokmax, "  max. angle to ridge", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“bounded ridges with near-zero normal”，无特定的顺序 */
zzdef_(zinc, Zridge0, "bounded ridges with near-zero normal", -1);

/* 定义和初始化名称为 wadd 的统计项，描述为“ave. angle to ridge”，依赖于 Zridge0 */
zzdef_(wadd, Wridge0, "  ave. angle to ridge", Zridge0);

/* 定义和初始化名称为 wmax 的统计项，描述为“max. angle to ridge”，无特定的顺序 */
zzdef_(wmax, Wridge0max, "  max. angle to ridge", -1);

/* 定义和初始化名称为 zdoc 的统计项，描述为“Triangulation statistics ('Qt')”，无特定的顺序 */
zdef_(zdoc, Zdoc12, "Triangulation statistics ('Qt')", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“non-simplicial facets triangulated”，无特定的顺序 */
zdef_(zinc, Ztricoplanar, "non-simplicial facets triangulated", -1);

/* 定义和初始化名称为 zadd 的统计项，描述为“ave. new facets created (may be deleted)”，依赖于 Ztricoplanar */
zdef_(zadd, Ztricoplanartot, "  ave. new facets created (may be deleted)", Ztricoplanar);

/* 定义和初始化名称为 zmax 的统计项，描述为“max. new facets created”，无特定的顺序 */
zdef_(zmax, Ztricoplanarmax, "  max. new facets created", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“null new facets deleted (duplicated vertex)”，无特定的顺序 */
zdef_(zinc, Ztrinull, "null new facets deleted (duplicated vertex)", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“mirrored pairs of new facets deleted (same vertices)”，无特定的顺序 */
zdef_(zinc, Ztrimirror, "mirrored pairs of new facets deleted (same vertices)", -1);

/* 定义和初始化名称为 zinc 的统计项，描述为“degenerate new facets in output (same ridge)”，无特定的顺序 */
zdef_(zinc, Ztridegen, "degenerate new facets in output (same ridge)", -1);
    qh_fprintf(qh, qh->ferr, 6373, "qhull internal error: qh_checklists failed on qh_collectstatistics\n");
    # 使用 qh_fprintf 函数向 qh->ferr 文件流写入错误信息，报告 Qhull 内部错误

    if (!qh->ERREXITcalled)
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    # 如果 ERREXITcalled 标志位未设置，则调用 qh_errexit 函数，以错误码 qh_ERRqhull 退出程序

  }
  FORALLfacets
    facet->seen= False;
    # 对所有的 facet 对象，将其 seen 标志位设置为 False，表示未被访问过

  if (qh->DELAUNAY) {
    FORALLfacets {
      if (facet->upperdelaunay != qh->UPPERdelaunay)
        facet->seen= True; /* remove from angle statistics */
      # 如果进行 Delaunay 三角化，检查 facet 的 upperdelaunay 属性是否等于 UPPERdelaunay
      # 如果不等，则将 facet 的 seen 标志位设置为 True，表示从角度统计中移除
    }
  }

  FORALLfacets {
    if (facet->visible && qh->NEWfacets)
      continue;
    # 对于每个 facet，如果其 visible 属性为真并且 NEWfacets 标志位为真，则跳过当前循环

    sizvertices= qh_setsize(qh, facet->vertices);
    sizneighbors= qh_setsize(qh, facet->neighbors);
    sizridges= qh_setsize(qh, facet->ridges);
    # 计算当前 facet 的顶点数、邻居数和棱数

    zinc_(Znumfacets);
    zadd_(Znumvertices, sizvertices);
    zmax_(Zmaxvertices, sizvertices);
    zadd_(Znumneighbors, sizneighbors);
    zmax_(Zmaxneighbors, sizneighbors);
    zadd_(Znummergetot, facet->nummerge);
    i= facet->nummerge; /* avoid warnings */
    zmax_(Znummergemax, i);
    # 更新统计信息：增加 facet 总数，顶点数总和，最大顶点数，邻居数总和，最大邻居数，
    # 合并数量总和和最大合并数量。i 用于避免警告，因为 facet->nummerge 是整数类型

    if (!facet->simplicial) {
      if (sizvertices == qh->hull_dim) {
        zinc_(Znowsimplicial);
      } else {
        zinc_(Znonsimplicial);
      }
      # 如果 facet 不是单纯形（simplicial），则根据其顶点数是否等于 hull_dim 更新 Znowsimplicial 或 Znonsimplicial 统计信息
    }

    if (sizridges) {
      zadd_(Znumridges, sizridges);
      zmax_(Zmaxridges, sizridges);
      # 如果 facet 有棱，更新棱数总和和最大棱数统计信息
    }

    zadd_(Zmemfacets, (int)sizeof(facetT) + qh->normal_size + 2*(int)sizeof(setT)
       + SETelemsize * (sizneighbors + sizvertices));
    # 更新 facet 内存使用统计信息，包括 facetT 结构体大小、法向量大小、邻居集合和顶点集合的内存占用

    if (facet->ridges) {
      zadd_(Zmemridges,
        (int)sizeof(setT) + SETelemsize * sizridges + sizridges *
         ((int)sizeof(ridgeT) + (int)sizeof(setT) + SETelemsize * (qh->hull_dim-1))/2);
      # 如果 facet 有棱，则更新棱集合的内存使用统计信息
    }

    if (facet->outsideset)
      zadd_(Zmempoints, (int)sizeof(setT) + SETelemsize * qh_setsize(qh, facet->outsideset));
    # 如果 facet 有外部点集合，则更新外部点集合的内存使用统计信息

    if (facet->coplanarset)
      zadd_(Zmempoints, (int)sizeof(setT) + SETelemsize * qh_setsize(qh, facet->coplanarset));
    # 如果 facet 有共面点集合，则更新共面点集合的内存使用统计信息

    if (facet->seen) /* Delaunay upper envelope */
      continue;
    # 如果 facet 的 seen 标志位为真，则跳过当前循环

    facet->seen= True;
    # 将 facet 的 seen 标志位设置为 True，表示已访问过

    FOREACHneighbor_(facet) {
      if (neighbor == qh_DUPLICATEridge || neighbor == qh_MERGEridge
          || neighbor->seen || !facet->normal || !neighbor->normal)
        continue;
      # 遍历当前 facet 的每个邻居，如果邻居是重复棱、合并棱，或者邻居已访问过，或者 facet 或邻居没有法向量，则跳过当前循环

      dotproduct= qh_getangle(qh, facet->normal, neighbor->normal);
      zinc_(Zangle);
      wadd_(Wangle, dotproduct);
      wmax_(Wanglemax, dotproduct)
      wmin_(Wanglemin, dotproduct)
      # 计算 facet 和邻居法向量的夹角，并更新夹角统计信息
    }

    if (facet->normal) {
      FOREACHvertex_(facet->vertices) {
        zinc_(Zdiststat);
        qh_distplane(qh, vertex->point, facet, &dist);
        wmax_(Wvertexmax, dist);
        wmin_(Wvertexmin, dist);
        # 如果 facet 有法向量，则遍历 facet 的每个顶点，计算顶点到平面的距离，并更新距离统计信息
      }
    }
  }

  FORALLvertices {
    if (vertex->deleted)
      continue;
    # 对于每个顶点，如果其 deleted 标志位为真，则跳过当前循环

    zadd_(Zmemvertices, (int)sizeof(vertexT));
    # 更新顶点内存使用统计信息，包括 vertexT 结构体大小

    if (vertex->neighbors) {
      sizneighbors= qh_setsize(qh, vertex->neighbors);
      zadd_(Znumvneighbors, sizneighbors);
      zmax_(Zmaxvneighbors, sizneighbors);
      zadd_(Zmemvertices, (int)sizeof(vertexT) + SETelemsize * sizneighbors);
      # 如果顶点有邻居集合，则更新邻居数总和、最大邻居数和邻居集合的内存使用统计信息
    }
  }

  qh->RANDOMdist= qh->old_randomdist;
  # 将 qh->RANDOMdist 设置为 qh->old_randomdist，恢复随机数分布设定
/* collectstatistics */
#endif /* qh_KEEPstatistics */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="initstatistics">-</a>

  qh_initstatistics(qh)
    initialize statistics

  notes:
    NOerrors -- qh_initstatistics can not use qh_errexit(), qh_fprintf, or qh.ferr
    On first call, only qhmem.ferr is defined.  qh_memalloc is not setup.
    Also invoked by QhullQh().
*/
void qh_initstatistics(qhT *qh) {
  int i;
  realT realx;
  int intx;

  // 统计全部类型的统计信息
  qh_allstatistics(qh);
  // 初始化统计信息的下一个索引为 0
  qh->qhstat.next= 0;
  // 调用各个特定类型的初始化统计信息函数
  qh_allstatA(qh);
  qh_allstatB(qh);
  qh_allstatC(qh);
  qh_allstatD(qh);
  qh_allstatE(qh);
  qh_allstatE2(qh);
  qh_allstatF(qh);
  qh_allstatG(qh);
  qh_allstatH(qh);
  qh_allstatI(qh);
  // 检查统计信息下一个索引是否超出预设的缓冲区大小
  if (qh->qhstat.next > (int)sizeof(qh->qhstat.id)) {
    // 输出错误信息并退出程序
    qh_fprintf_stderr(6184, "qhull internal error (qh_initstatistics): increase size of qhstat.id[].  qhstat.next %d should be <= sizeof(qh->qhstat.id) %d\n", 
          qh->qhstat.next, (int)sizeof(qh->qhstat.id));
#if 0 /* for locating error, Znumridges should be duplicated */
    for(i=0; i < ZEND; i++) {
      int j;
      for(j=i+1; j < ZEND; j++) {
        if (qh->qhstat.id[i] == qh->qhstat.id[j]) {
          // 输出错误信息，指出重复的统计信息编号
          qh_fprintf_stderr(6185, "qhull error (qh_initstatistics): duplicated statistic %d at indices %d and %d\n",
              qh->qhstat.id[i], i, j);
        }
      }
    }
#endif
    // 异常退出程序
    qh_exit(qh_ERRqhull);  /* can not use qh_errexit() */
  }
  // 初始化各种统计信息的起始值
  qh->qhstat.init[zinc].i= 0;
  qh->qhstat.init[zadd].i= 0;
  qh->qhstat.init[zmin].i= INT_MAX;
  qh->qhstat.init[zmax].i= INT_MIN;
  qh->qhstat.init[wadd].r= 0;
  qh->qhstat.init[wmin].r= REALmax;
  qh->qhstat.init[wmax].r= -REALmax;
  // 根据统计信息的类型进行初始化
  for(i=0; i < ZEND; i++) {
    if (qh->qhstat.type[i] > ZTYPEreal) {
      // 如果是实数类型，用默认的实数值进行初始化
      realx= qh->qhstat.init[(unsigned char)(qh->qhstat.type[i])].r;
      qh->qhstat.stats[i].r= realx;
    }else if (qh->qhstat.type[i] != zdoc) {
      // 如果是整数类型（但不是文档类型），用默认的整数值进行初始化
      intx= qh->qhstat.init[(unsigned char)(qh->qhstat.type[i])].i;
      qh->qhstat.stats[i].i= intx;
    }
  }
} /* initstatistics */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="newstats">-</a>

  qh_newstats(qh )
    returns True if statistics for zdoc

  returns:
    next zdoc
*/
boolT qh_newstats(qhT *qh, int idx, int *nextindex) {
  boolT isnew= False;
  int start, i;

  // 如果当前统计信息类型为文档类型，则从下一个索引开始
  if (qh->qhstat.type[qh->qhstat.id[idx]] == zdoc)
    start= idx+1;
  else
    start= idx;
  // 检查是否有新的统计信息需要输出
  for(i= start; i < qh->qhstat.next && qh->qhstat.type[qh->qhstat.id[i]] != zdoc; i++) {
    if (!qh_nostatistic(qh, qh->qhstat.id[i]) && !qh->qhstat.printed[qh->qhstat.id[i]])
        isnew= True;
  }
  // 返回下一个统计信息的索引位置
  *nextindex= i;
  return isnew;
} /* newstats */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="nostatistic">-</a>

  qh_nostatistic(qh, index )
    true if no statistic to print
*/
#if qh_KEEPstatistics
/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printallstatistics">-</a>

  qh_printallstatistics(qh, fp, string )
    print all statistics with header 'string'
*/
void qh_printallstatistics(qhT *qh, FILE *fp, const char *string) {
  // 收集所有统计数据
  qh_allstatistics(qh);
  // 收集统计数据
  qh_collectstatistics(qh);
  // 打印统计数据到文件
  qh_printstatistics(qh, fp, string);
  // 打印内存统计信息到文件
  qh_memstatistics(qh, fp);
}


/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printstatistics">-</a>

  qh_printstatistics(qh, fp, string )
    print statistics to a file with header 'string'
    skips statistics with qhstat.printed[] (reset with qh_allstatistics)

  see:
    qh_printallstatistics()
*/
void qh_printstatistics(qhT *qh, FILE *fp, const char *string) {
  int i, k;
  realT ave; /* ignored */

  // 如果点的数量不等于顶点的数量或者 Zpbalance 为 0，则设置 Wpbalance 和 Wpbalance2 为 0
  if (qh->num_points != qh->num_vertices || zval_(Zpbalance) == 0) {
    wval_(Wpbalance)= 0.0;
    wval_(Wpbalance2)= 0.0;
  }else
    // 否则，计算 Wpbalance2 的标准差
    wval_(Wpbalance2)= qh_stddev(qh, zval_(Zpbalance), wval_(Wpbalance),
                                 wval_(Wpbalance2), &ave);
  // 如果 Zprocessed 为 0，则设置 Wnewbalance2 为 0
  if (zval_(Zprocessed) == 0)
    wval_(Wnewbalance2)= 0.0;
  else
    // 否则，计算 Wnewbalance2 的标准差
    wval_(Wnewbalance2)= qh_stddev(qh, zval_(Zprocessed), wval_(Wnewbalance),
                                 wval_(Wnewbalance2), &ave);
  // 打印统计信息到文件
  qh_fprintf(qh, fp, 9350, "\n\
%s\n\
qhull invoked by: %s | %s\n  %s with options:\n%s\n", 
    string, qh->rbox_command, qh->qhull_command, qh_version, qh->qhull_options);

  // 打印精度常数的统计信息到文件
  qh_fprintf(qh, fp, 9351, "\nprecision constants:\n\
 %6.2g max. abs. coordinate in the (transformed) input ('Qbd:n')\n\
 %6.2g max. roundoff error for distance computation ('En')\n\
 %6.2g max. roundoff error for angle computations\n\
 %6.2g min. distance for outside points ('Wn')\n\
 %6.2g min. distance for visible facets ('Vn')\n\
 %6.2g max. distance for coplanar facets ('Un')\n\
 %6.2g max. facet width for recomputing centrum and area\n\
",
  qh->MAXabs_coord, qh->DISTround, qh->ANGLEround, qh->MINoutside,
        qh->MINvisible, qh->MAXcoplanar, qh->WIDEfacet);
  // 如果 KEEPnearinside 被设置
  if (qh->KEEPnearinside)
    // 输出关于 NEARinside 的信息，包括最大近内点距离
    qh_fprintf(qh, fp, 9352, "\
 %6.2g max. distance for near-inside points\n", qh->NEARinside);
  // 如果 premerge_cos 小于 REALmax 的一半，则输出有关 pre-merge 角度的信息
  if (qh->premerge_cos < REALmax/2) qh_fprintf(qh, fp, 9353, "\
 %6.2g max. cosine for pre-merge angle\n", qh->premerge_cos);
  // 如果 PREmerge 标志为真，则输出有关 pre-merge 中心的半径信息
  if (qh->PREmerge) qh_fprintf(qh, fp, 9354, "\
 %6.2g radius of pre-merge centrum\n", qh->premerge_centrum);
  // 如果 postmerge_cos 小于 REALmax 的一半，则输出有关 post-merge 角度的信息
  if (qh->postmerge_cos < REALmax/2) qh_fprintf(qh, fp, 9355, "\
 %6.2g max. cosine for post-merge angle\n", qh->postmerge_cos);
  // 如果 POSTmerge 标志为真，则输出有关 post-merge 中心的半径信息
  if (qh->POSTmerge) qh_fprintf(qh, fp, 9356, "\
 %6.2g radius of post-merge centrum\n", qh->postmerge_centrum);
  // 输出有关合并两个简单面的最大距离，算术操作的最大舍入误差和分母的最小值信息
  qh_fprintf(qh, fp, 9357, "\
 %6.2g max. distance for merging two simplicial facets\n\
 %6.2g max. roundoff error for arithmetic operations\n\
 %6.2g min. denominator for division\n\
  zero diagonal for Gauss: ", qh->ONEmerge, REALepsilon, qh->MINdenom);
  // 输出每个维度上的 NEARzero 数组的值
  for(k=0; k < qh->hull_dim; k++)
    qh_fprintf(qh, fp, 9358, "%6.2e ", qh->NEARzero[k]);
  // 输出空行
  qh_fprintf(qh, fp, 9359, "\n\n");
  // 输出统计信息，逐行打印统计数据
  for(i=0 ; i < qh->qhstat.next; )
    qh_printstats(qh, fp, i, &i);
/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printstatlevel">-</a>

  qh_printstatlevel(qhT *qh, FILE *fp, int id)
    打印指定统计信息的级别信息

  notes:
    如果 id >= ZEND 或者已经打印过，则不进行任何操作
    如果统计信息类型为 zdoc，则打印相应的文档信息

*/
void qh_printstatlevel(qhT *qh, FILE *fp, int id) {

  if (id >= ZEND || qh->qhstat.printed[id])
    return;
  if (qh->qhstat.type[id] == zdoc) {
    qh_fprintf(qh, fp, 9360, "%s\n", qh->qhstat.doc[id]);
    return;
  }
  if (qh_nostatistic(qh, id) || !qh->qhstat.doc[id])
    return;
  qh->qhstat.printed[id]= True;
  if (qh->qhstat.count[id] != -1
      && qh->qhstat.stats[(unsigned char)(qh->qhstat.count[id])].i == 0)
    qh_fprintf(qh, fp, 9361, " *0 cnt*");
  else if (qh->qhstat.type[id] >= ZTYPEreal && qh->qhstat.count[id] == -1)
    qh_fprintf(qh, fp, 9362, "%7.2g", qh->qhstat.stats[id].r);
  else if (qh->qhstat.type[id] >= ZTYPEreal && qh->qhstat.count[id] != -1)
    qh_fprintf(qh, fp, 9363, "%7.2g", qh->qhstat.stats[id].r/ qh->qhstat.stats[(unsigned char)(qh->qhstat.count[id])].i);
  else if (qh->qhstat.type[id] < ZTYPEreal && qh->qhstat.count[id] == -1)
    qh_fprintf(qh, fp, 9364, "%7d", qh->qhstat.stats[id].i);
  else if (qh->qhstat.type[id] < ZTYPEreal && qh->qhstat.count[id] != -1)
    qh_fprintf(qh, fp, 9365, "%7.3g", (realT) qh->qhstat.stats[id].i / qh->qhstat.stats[(unsigned char)(qh->qhstat.count[id])].i);
  qh_fprintf(qh, fp, 9366, " %s\n", qh->qhstat.doc[id]);
} /* printstatlevel */


/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printstats">-</a>

  qh_printstats(qhT *qh, FILE *fp, int idx, int *nextindex )
    打印一个 zdoc 组的统计信息

  returns:
    如果下一个 zdoc 不为空，则返回其值
*/
void qh_printstats(qhT *qh, FILE *fp, int idx, int *nextindex) {
  int j, nexti;

  if (qh_newstats(qh, idx, &nexti)) {
    qh_fprintf(qh, fp, 9367, "\n");
    for (j=idx; j<nexti; j++)
      qh_printstatlevel(qh, fp, qh->qhstat.id[j]);
  }
  if (nextindex)
    *nextindex= nexti;
} /* printstats */

#if qh_KEEPstatistics

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="stddev">-</a>

  qh_stddev(qhT *qh, int num, realT tot, realT tot2, realT *ave )
    从统计信息计算标准差和平均值

    tot2 是平方和的总和
  notes:
    计算均方根：
      (x-ave)^2
      == x^2 - 2x tot/num +   (tot/num)^2
      == tot2 - 2 tot tot/num + tot tot/num
      == tot2 - tot ave
*/
realT qh_stddev(qhT *qh, int num, realT tot, realT tot2, realT *ave) {
  realT stddev;

  if (num <= 0) {
    qh_fprintf(qh, qh->ferr, 7101, "qhull warning (qh_stddev): expecting num > 0.  Got num %d, tot %4.4g, tot2 %4.4g.  Returning 0.0\n",
      num, tot, tot2);
    return 0.0;
  }
  *ave= tot/num;
  stddev= sqrt(fabs(tot2/num - *ave * *ave));
  return stddev;
} /* stddev */
#else
/* 
   计算标准偏差的函数，用于 qhull_r-exports.def 导出

   参数说明：
   - qh: Qhull 对象，未使用，保留参数
   - num: 整数值，未使用，保留参数
   - tot: 实数值，未使用，保留参数
   - tot2: 实数值，未使用，保留参数
   - ave: 实数指针，未使用，保留参数

   返回值：
   - 返回值为 0.0，因为当前函数体为空

   注意事项：
   - 该函数仅在未开启统计数据保留(qh_KEEPstatistics 未定义)时才存在
*/

realT qh_stddev(qhT *qh, int num, realT tot, realT tot2, realT *ave) { /* for qhull_r-exports.def */
  QHULL_UNUSED(qh)
  QHULL_UNUSED(num)
  QHULL_UNUSED(tot)
  QHULL_UNUSED(tot2)
  QHULL_UNUSED(ave)
    
  return 0.0;
}
#endif /* qh_KEEPstatistics */

#if !qh_KEEPstatistics
/*
   未开启统计数据保留时的空函数实现
*/

void    qh_collectstatistics(qhT *qh) {}

/*
   未开启统计数据保留时的空函数实现
   参数说明：
   - qh: Qhull 对象，未使用，保留参数
   - fp: 文件指针，未使用，保留参数
   - string: 字符串指针，未使用，保留参数
*/

void    qh_printallstatistics(qhT *qh, FILE *fp, const char *string) {}

/*
   未开启统计数据保留时的空函数实现
   参数说明：
   - qh: Qhull 对象，未使用，保留参数
   - fp: 文件指针，未使用，保留参数
   - string: 字符串指针，未使用，保留参数
*/

void    qh_printstatistics(qhT *qh, FILE *fp, const char *string) {}
#endif
```