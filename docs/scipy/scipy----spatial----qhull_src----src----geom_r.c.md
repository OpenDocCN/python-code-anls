# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\geom_r.c`

```
/*<html><pre>  -<a                             href="qh-geom_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   geom_r.c
   geometric routines of qhull

   see qh-geom_r.htm and geom_r.h

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/geom_r.c#4 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $

   infrequent code goes into geom2_r.c
*/

#include "qhull_ra.h"

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="distplane">-</a>

  qh_distplane(qh, point, facet, dist )
    return distance from point to facet

  returns:
    dist
    if qh.RANDOMdist, joggles result

  notes:
    dist > 0 if point is above facet (i.e., outside)
    does not error (for qh_sortfacets, qh_outerinner)
    for nearly coplanar points, the returned values may be duplicates
      for example pairs of nearly incident points, rbox 175 C1,2e-13 t1538759579 | qhull d T4
      622 qh_distplane: e-014  # count of two or more duplicate values for unique calls
      258 qh_distplane: e-015
      38 qh_distplane: e-016
      40 qh_distplane: e-017
      6 qh_distplane: e-018
      5 qh_distplane: -e-018
      33 qh_distplane: -e-017
         3153 qh_distplane: -2.775557561562891e-017  # duplicated value for 3153 unique calls
      42 qh_distplane: -e-016
      307 qh_distplane: -e-015
      1271 qh_distplane: -e-014
      13 qh_distplane: -e-013

  see:
    qh_distnorm in geom2_r.c
    qh_distplane [geom_r.c], QhullFacet::distance, and QhullHyperplane::distance are copies
*/
void qh_distplane(qhT *qh, pointT *point, facetT *facet, realT *dist) {
  coordT *normal= facet->normal, *coordp, randr;
  int k;

  switch (qh->hull_dim){
  case 2:
    *dist= facet->offset + point[0] * normal[0] + point[1] * normal[1];
    break;
  case 3:
    *dist= facet->offset + point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
    break;
  case 4:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3];
    break;
  case 5:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4];
    break;
  case 6:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4]+point[5]*normal[5];
    break;
  case 7:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4]+point[5]*normal[5]+point[6]*normal[6];
    break;
  case 8:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4]+point[5]*normal[5]+point[6]*normal[6]+point[7]*normal[7];
    break;
  default:
    *dist= facet->offset;
    coordp= point;
    for (k=qh->hull_dim; k--; )
      *dist += *coordp++ * *normal++;
    break;
  }
  zzinc_(Zdistplane);  /* increment the Zdistplane counter */
  if (!qh->RANDOMdist && qh->IStracing < 4)
    return;

返回语句，用于在函数中立即结束并返回结果。


  if (qh->RANDOMdist) {

条件语句开始，检查变量 `qh->RANDOMdist` 是否为真（非零）。


    randr= qh_RANDOMint;

将 `qh_RANDOMint` 的值赋给变量 `randr`。


    *dist += (2.0 * randr / qh_RANDOMmax - 1.0) *
      qh->RANDOMfactor * qh->MAXabs_coord;

计算一个表达式并将结果加到 `*dist` 指向的变量中：
- `(2.0 * randr / qh_RANDOMmax - 1.0)`：计算一个范围在 `-1.0` 到 `1.0` 之间的随机数。
- `qh->RANDOMfactor` 和 `qh->MAXabs_coord`：分别是乘法因子，用于调整最终的结果。


  }

条件语句块结束标志，表明条件语句的范围。
/* 如果未定义 qh_NOtrace，则进入条件判断 */
#ifndef qh_NOtrace
  /* 如果跟踪级别大于等于4 */
  if (qh->IStracing >= 4) {
    /* 输出跟踪信息到 qh->ferr */
    qh_fprintf(qh, qh->ferr, 8001, "qh_distplane: ");
    /* 输出距离值到 qh->ferr */
    qh_fprintf(qh, qh->ferr, 8002, qh_REAL_1, *dist);
    /* 输出点和面的标识到 qh->ferr */
    qh_fprintf(qh, qh->ferr, 8003, "from p%d to f%d\n", qh_pointid(qh, point), facet->id);
  }
#endif
  /* 函数返回，结束 distplane 函数的执行 */
  return;
} /* distplane */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="findbest">-</a>

  qh_findbest(qh, point, startfacet, bestoutside, qh_ISnewfacets, qh_NOupper, dist, isoutside, numpart )
    find facet that is furthest below a point
    for upperDelaunay facets
      returns facet only if !qh_NOupper and clearly above

  input:
    starts search at 'startfacet' (can not be flipped)
    if !bestoutside(qh_ALL), stops at qh.MINoutside

  returns:
    best facet (reports error if NULL)
    early out if isoutside defined and bestdist > qh.MINoutside
    dist is distance to facet
    isoutside is true if point is outside of facet
    numpart counts the number of distance tests

  see also:
    qh_findbestnew()

  notes:
    If merging (testhorizon), searches horizon facets of coplanar best facets because
    after qh_distplane, this and qh_partitionpoint are the most expensive in 3-d
      avoid calls to distplane, function calls, and real number operations.
    caller traces result
    Optimized for outside points.   Tried recording a search set for qh_findhorizon.
    Made code more complicated.

  when called by qh_partitionvisible():
    indicated by qh_ISnewfacets
    qh.newfacet_list is list of simplicial, new facets
    qh_findbestnew set if qh_sharpnewfacets returns True (to use qh_findbestnew)
    qh.bestfacet_notsharp set if qh_sharpnewfacets returns False

  when called by qh_findfacet(), qh_partitionpoint(), qh_partitioncoplanar(),
                 qh_check_bestdist(), qh_addpoint()
    indicated by !qh_ISnewfacets
    returns best facet in neighborhood of given facet
      this is best facet overall if dist >= -qh.MAXcoplanar
        or hull has at least a "spherical" curvature

  design:
    initialize and test for early exit
    repeat while there are better facets
      for each neighbor of facet
        exit if outside facet found
        test for better facet
    if point is inside and partitioning
      test for new facets with a "sharp" intersection
      if so, future calls go to qh_findbestnew()
    test horizon facets
*/
facetT *qh_findbest(qhT *qh, pointT *point, facetT *startfacet,
                     boolT bestoutside, boolT isnewfacets, boolT noupper,
                     realT *dist, boolT *isoutside, int *numpart) {
  realT bestdist= -REALmax/2 /* avoid underflow */; /* 初始化最佳距离为负的最大实数，避免下溢 */
  facetT *facet, *neighbor, **neighborp; /* 定义面、邻面、邻面指针变量 */
  facetT *bestfacet= NULL, *lastfacet= NULL; /* 最佳面和最后面的初始化为 NULL */
  int oldtrace= qh->IStracing; /* 保存旧的跟踪级别 */
  unsigned int visitid= ++qh->visit_id; /* 增加访问标识 */
  int numpartnew=0; /* 初始化新的距离测试计数器为0 */
  boolT testhorizon= True; /* 如果精确需要，则为 True，例如，rbox c D6 | qhull Q0 Tv */

  /* 增加 Zfindbest 统计计数 */
  zinc_(Zfindbest);
#ifndef qh_NOtrace
  // 如果未定义 qh_NOtrace，则执行以下代码块
  if (qh->IStracing >= 4 || (qh->TRACElevel && qh->TRACEpoint >= 0 && qh->TRACEpoint == qh_pointid(qh, point))) {
    // 如果跟踪级别大于等于4，或者跟踪级别非零且跟踪点等于当前点的ID
    if (qh->TRACElevel > qh->IStracing)
      qh->IStracing= qh->TRACElevel;
    // 如果跟踪级别大于当前设置的跟踪级别，则更新当前跟踪级别为设定的跟踪级别
    qh_fprintf(qh, qh->ferr, 8004, "qh_findbest: point p%d starting at f%d isnewfacets? %d, unless %d exit if > %2.2g,",
             qh_pointid(qh, point), startfacet->id, isnewfacets, bestoutside, qh->MINoutside);
    // 输出调试信息到指定文件中
    qh_fprintf(qh, qh->ferr, 8005, " testhorizon? %d, noupper? %d,", testhorizon, noupper);
    // 输出调试信息到指定文件中
    qh_fprintf(qh, qh->ferr, 8006, " Last qh_addpoint p%d,", qh->furthest_id);
    // 输出调试信息到指定文件中
    qh_fprintf(qh, qh->ferr, 8007, " Last merge #%d, max_outside %2.2g\n", zzval_(Ztotmerge), qh->max_outside);
    // 输出调试信息到指定文件中
  }
#endif
// 如果 isoutside 不为空指针，则将其设置为 True
if (isoutside)
  *isoutside= True;
// 如果 startfacet 的 flipped 属性为假，则执行以下代码块（在测试其邻居之前测试 startfacet）
if (!startfacet->flipped) {  /* test startfacet before testing its neighbors */
  // 将 numpart 设置为 1
  *numpart= 1;
  // 计算点到 startfacet 的距离，并将结果存储在 dist 中
  qh_distplane(qh, point, startfacet, dist);  /* this code is duplicated below */
  // 如果 bestoutside 为假，并且距离大于等于 MINoutside，并且 startfacet 不是 upperdelaunay 或者 noupper 为真
  if (!bestoutside && *dist >= qh->MINoutside
  && (!startfacet->upperdelaunay || !noupper)) {
    // 将 bestfacet 设置为 startfacet，并跳转到 LABELreturn_best 标签处
    bestfacet= startfacet;
    goto LABELreturn_best;
  }
  // 将 bestdist 设置为 dist 的值
  bestdist= *dist;
  // 如果 startfacet 不是 upperdelaunay，则将 bestfacet 设置为 startfacet
  if (!startfacet->upperdelaunay) {
    bestfacet= startfacet;
  }
}else
  // 否则将 numpart 设置为 0
  *numpart= 0;
// 将 startfacet 的 visitid 设置为 visitid
startfacet->visitid= visitid;
// 将 facet 设置为 startfacet
facet= startfacet;
// 循环直到 facet 不为空
while (facet) {
  // 如果启用跟踪级别为4，则输出邻居信息到指定文件中
  trace4((qh, qh->ferr, 4001, "qh_findbest: neighbors of f%d, bestdist %2.2g f%d\n",
              facet->id, bestdist, getid_(bestfacet)));
  // 将 lastfacet 设置为 facet
  lastfacet= facet;
  // 遍历 facet 的每个邻居
  FOREACHneighbor_(facet) {
    // 如果邻居不是新的面，并且 isnewfacets 为真，则跳过本次循环
    if (!neighbor->newfacet && isnewfacets)
      continue;
    // 如果邻居的 visitid 等于 visitid，则跳过本次循环
    if (neighbor->visitid == visitid)
      continue;
    // 将邻居的 visitid 设置为 visitid
    neighbor->visitid= visitid;
    // 如果邻居的 flipped 属性为假
    if (!neighbor->flipped) {  /* code duplicated above */
      // 增加 numpart 的计数
      (*numpart)++;
      // 计算点到邻居的距离，并将结果存储在 dist 中
      qh_distplane(qh, point, neighbor, dist);
      // 如果距离大于 bestdist
      if (*dist > bestdist) {
        // 如果 bestoutside 为假，并且距离大于等于 MINoutside，并且邻居不是 upperdelaunay 或者 noupper 为真
        if (!bestoutside && *dist >= qh->MINoutside
        && (!neighbor->upperdelaunay || !noupper)) {
          // 将 bestfacet 设置为邻居，并跳转到 LABELreturn_best 标签处
          bestfacet= neighbor;
          goto LABELreturn_best;
        }
        // 如果邻居不是 upperdelaunay，则将 bestfacet 设置为邻居，并更新 bestdist 的值
        if (!neighbor->upperdelaunay) {
          bestfacet= neighbor;
          bestdist= *dist;
          break; /* switch to neighbor */
        // 否则如果 bestfacet 为 NULL，则更新 bestdist 的值
        }else if (!bestfacet) {
          bestdist= *dist;
          break; /* switch to neighbor */
        }
      } /* end of *dist>bestdist */
    } /* end of !flipped */
  } /* end of FOREACHneighbor */
  // 将 facet 设置为 neighbor（仅在 *dist>bestdist 时非空）
  facet= neighbor;  /* non-NULL only if *dist>bestdist */
} /* end of while facet (directed search) */
// 如果 isnewfacets 为真
if (isnewfacets) {
  // 如果 bestfacet 为假，则设置 bestdist 为 -REALmax/2，并调用 qh_findbestnew 查找最佳新面
  if (!bestfacet) { /* startfacet is upperdelaunay (or flipped) w/o !flipped newfacet neighbors */
    bestdist= -REALmax/2;
    bestfacet= qh_findbestnew(qh, point, qh->newfacet_list, &bestdist, bestoutside, isoutside, &numpartnew);
    // 设置 testhorizon 为假，因为 qh_findbestnew 调用 qh_findbesthorizon
    testhorizon= False;
  }
    }else if (!qh->findbest_notsharp && bestdist < -qh->DISTround) {
      // 如果不是在查找非尖角且最佳距离小于负的 DISTround
      if (qh_sharpnewfacets(qh)) {
        /* 很少使用，qh_findbestnew 将重新测试所有的 facets */
        zinc_(Zfindnewsharp);
        // 增加 Zfindnewsharp 的计数
        bestfacet= qh_findbestnew(qh, point, bestfacet, &bestdist, bestoutside, isoutside, &numpartnew);
        // 设置 testhorizon 为 False，因为 qh_findbestnew 会调用 qh_findbesthorizon
        testhorizon= False;
        // 设置 qh->findbestnew 为 True
        qh->findbestnew= True;
      }else
        // 设置 qh->findbest_notsharp 为 True
        qh->findbest_notsharp= True;
    }
  }
  // 如果没有找到 bestfacet，则调用 qh_findbestlower 函数
  if (!bestfacet)
    bestfacet= qh_findbestlower(qh, lastfacet, point, &bestdist, numpart); /* lastfacet is non-NULL because startfacet is non-NULL */
  // 如果 testhorizon 为真，则调用 qh_findbesthorizon 函数（说明 qh_findbestnew 没有被调用）
  if (testhorizon)
    bestfacet= qh_findbesthorizon(qh, !qh_IScheckmax, point, bestfacet, noupper, &bestdist, &numpartnew);
  // 设置 dist 指针的值为 bestdist
  *dist= bestdist;
  // 如果 isoutside 为真且 bestdist 小于 qh->MINoutside，则设置 isoutside 为 False
  if (isoutside && bestdist < qh->MINoutside)
    *isoutside= False;
LABELreturn_best:
  zadd_(Zfindbesttot, *numpart);
  zmax_(Zfindbestmax, *numpart);
  (*numpart) += numpartnew;
  qh->IStracing= oldtrace;
  return bestfacet;
}  /* findbest */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="findbesthorizon">-</a>

  qh_findbesthorizon(qh, qh_IScheckmax, point, startfacet, qh_NOupper, &bestdist, &numpart )
    search coplanar and better horizon facets from startfacet/bestdist
    ischeckmax turns off statistics and minsearch update
    all arguments must be initialized, including *bestdist and *numpart
    qh.coplanarfacetset used to maintain current search set, reset whenever best facet is substantially better
  returns(ischeckmax):
    best facet
    updates f.maxoutside for neighbors of searched facets (if qh_MAXoutside)
  returns(!ischeckmax):
    best facet that is not upperdelaunay or newfacet (qh.first_newfacet)
    allows upperdelaunay that is clearly outside
  returns:
    bestdist is distance to bestfacet
    numpart -- updates number of distance tests

  notes:
    called by qh_findbest if point is not outside a facet (directed search)
    called by qh_findbestnew if point is not outside a new facet
    called by qh_check_maxout for each point in hull
    called by qh_check_bestdist for each point in hull (rarely used)

    no early out -- use qh_findbest() or qh_findbestnew()
    Searches coplanar or better horizon facets

  when called by qh_check_maxout() (qh_IScheckmax)
    startfacet must be closest to the point
      Otherwise, if point is beyond and below startfacet, startfacet may be a local minimum
      even though other facets are below the point.
    updates facet->maxoutside for good, visited facets
    may return NULL

    searchdist is qh.max_outside + 2 * DISTround
      + max( MINvisible('Vn'), MAXcoplanar('Un'));
    This setting is a guess.  It must be at least max_outside + 2*DISTround
    because a facet may have a geometric neighbor across a vertex

  design:
    for each horizon facet of coplanar best facets
      continue if clearly inside
      unless upperdelaunay or clearly outside
         update best facet
*/
facetT *qh_findbesthorizon(qhT *qh, boolT ischeckmax, pointT* point, facetT *startfacet, boolT noupper, realT *bestdist, int *numpart) {
  facetT *bestfacet= startfacet;  // 初始化最佳的 facet 为起始 facet
  realT dist;  // 用于存储距离的变量
  facetT *neighbor, **neighborp, *facet;  // 定义相邻 facet 和当前处理的 facet 的指针
  facetT *nextfacet= NULL; /* optimize last facet of coplanarfacetset */
  int numpartinit= *numpart, coplanarfacetset_size, numcoplanar= 0, numfacet= 0;  // 初始化一些计数变量
  unsigned int visitid= ++qh->visit_id;  // 更新 visitid，标记访问的顺序
  boolT newbest= False; /* for tracing */  // 用于追踪的标志变量
  realT minsearch, searchdist;  /* skip facets that are too far from point */  // 跳过距离点太远的 facet 的最小搜索距离和搜索距离
  boolT is_5x_minsearch;  // 标志变量，是否为 5 倍的 minsearch

  if (!ischeckmax) {
    zinc_(Zfindhorizon);  // 如果不是检查最大值，增加 Zfindhorizon 统计计数
  }else {
#if qh_MAXoutside
    if ((!qh->ONLYgood || startfacet->good) && *bestdist > startfacet->maxoutside)
      startfacet->maxoutside= *bestdist;
#endif
  }
  // 计算搜索距离，基于 qh_SEARCHdist 的表达式，是 qh.max_outside 和精度常数的倍数
  searchdist= qh_SEARCHdist; /* an expression, a multiple of qh.max_outside and precision constants */
  // 计算最小搜索距离
  minsearch= *bestdist - searchdist;
  // 如果需要检查最大值
  if (ischeckmax) {
    /* Always check coplanar facets.  Needed for RBOX 1000 s Z1 G1e-13 t996564279 | QHULL Tv */
    // 最小化 minsearch 和 -searchdist 的值
    minimize_(minsearch, -searchdist);
  }
  // 设置 coplanarfacetset 的大小为 0
  coplanarfacetset_size= 0;
  // 将 startfacet 的 visitid 设置为当前的 visitid
  startfacet->visitid= visitid;
  // 初始化 facet 为 startfacet
  facet= startfacet;
  // 开始循环，直到结束条件为真
  while (True) {
    // 每次循环 numfacet 自增
    numfacet++;
    // 检查是否是 5 倍于 minsearch，并且 facet->nummerge 大于 10 且 facet->neighbors 的大小大于 100
    is_5x_minsearch= (ischeckmax && facet->nummerge > 10 && qh_setsize(qh, facet->neighbors) > 100);  /* QH11033 FIX: qh_findbesthorizon: many tests for facets with many merges and neighbors.  Can hide coplanar facets, e.g., 'rbox 1000 s Z1 G1e-13' with 4400+ neighbors */
    // 输出调试信息，显示 facet 相关的信息，如 bestdist、bestfacet 的 id 等
    trace4((qh, qh->ferr, 4002, "qh_findbesthorizon: test neighbors of f%d bestdist %2.2g f%d ischeckmax? %d noupper? %d minsearch %2.2g is_5x? %d searchdist %2.2g\n",
                facet->id, *bestdist, getid_(bestfacet), ischeckmax, noupper,
                minsearch, is_5x_minsearch, searchdist));
    // 遍历 facet 的每个邻居
    FOREACHneighbor_(facet) {
      // 如果邻居的 visitid 等于当前的 visitid，则继续下一个循环
      if (neighbor->visitid == visitid)
        continue;
      // 将邻居的 visitid 设置为当前的 visitid
      neighbor->visitid= visitid;
      // 如果邻居没有翻转
      if (!neighbor->flipped) {  /* neighbors of flipped facets always searched via nextfacet */
        // 计算点到邻居平面的距离，并存储在 dist 中
        qh_distplane(qh, point, neighbor, &dist); /* duplicate qh_distpane for new facets, they may be coplanar */
        // numpart 自增
        (*numpart)++;
        // 如果 dist 大于 *bestdist
        if (dist > *bestdist) {
          // 如果邻居不是上凸 Delaunay 或者是检查最大值，或者不需要上凸（!noupper）并且 dist 大于 qh->MINoutside
          if (!neighbor->upperdelaunay || ischeckmax || (!noupper && dist >= qh->MINoutside)) {
            // 如果不是检查最大值
            if (!ischeckmax) {
              // 更新 minsearch
              minsearch= dist - searchdist;
              // 如果 dist 大于 *bestdist 加上 searchdist
              if (dist > *bestdist + searchdist) {
                // 计数器 zinc_ 自增，用于统计所有在 qh.coplanarfacetset 中至少低于 searchdist 的内容
                zinc_(Zfindjump);  /* everything in qh.coplanarfacetset at least searchdist below */
                // 将 coplanarfacetset_size 设为 0
                coplanarfacetset_size= 0;
              }
            }
            // 更新 bestfacet 和 *bestdist
            bestfacet= neighbor;
            *bestdist= dist;
            // 设置 newbest 为真
            newbest= True;
          }
        }else if (is_5x_minsearch) {
          // 如果 dist 小于 5 倍的 minsearch，则跳过这个邻居，不设置 nextfacet。dist 是负数
          if (dist < 5 * minsearch)
            continue;
        }else if (dist < minsearch)
          continue;  /* skip this neighbor, do not set nextfacet.  If ischeckmax, dist can't be positive */
#if qh_MAXoutside
        // 如果是检查最大值且 dist 大于邻居的 maxoutside，则更新邻居的 maxoutside
        if (ischeckmax && dist > neighbor->maxoutside)
          neighbor->maxoutside= dist;
#endif
      } /* end of !flipped, need to search neighbor */
      // 如果有 nextfacet
      if (nextfacet) {
        // numcoplanar 自增
        numcoplanar++;
        // 如果 coplanarfacetset_size 为 0
        if (!coplanarfacetset_size++) {
          // 设置 qh->coplanarfacetset 的第一个元素为 nextfacet，并截断 set
          SETfirst_(qh->coplanarfacetset)= nextfacet;
          SETtruncate_(qh->coplanarfacetset, 1);
        }else
          // 追加 nextfacet 到 qh->coplanarfacetset 中
          qh_setappend(qh, &qh->coplanarfacetset, nextfacet); /* Was needed for RBOX 1000 s W1e-13 P0 t996547055 | QHULL d Qbb Qc Tv
                                                 and RBOX 1000 s Z1 G1e-13 t996564279 | qhull Tv  */
      }
      // 设置 nextfacet 为当前的邻居
      nextfacet= neighbor;
    } /* end of EACHneighbor */
    // 将 facet 设置为 nextfacet
    facet= nextfacet;
    // 如果 facet 存在，则将 nextfacet 设为 NULL
    if (facet)
      nextfacet= NULL;
    // 如果 coplanarfacetset_size 为 0，则跳出循环
    else if (!coplanarfacetset_size)
      break;
    else if (!--coplanarfacetset_size) {
      # 如果 coplanarfacetset_size 减到零，则执行以下操作：
      #   获取 coplanarfacetset 集合的第一个元素作为 facet
      facet= SETfirstt_(qh->coplanarfacetset, facetT);
      #   清空 coplanarfacetset 集合
      SETtruncate_(qh->coplanarfacetset, 0);
    }else
      # 否则，从 coplanarfacetset 集合中删除最后一个元素，并将其作为 facet
      facet= (facetT *)qh_setdellast(qh->coplanarfacetset);
  } /* while True, i.e., "for each facet in qh.coplanarfacetset" */
  
  # 如果不是在检查最大值阶段，则执行以下操作：
  if (!ischeckmax) {
    #   将 *numpart - numpartinit 的值加入 Zfindhorizontot 统计
    zadd_(Zfindhorizontot, *numpart - numpartinit);
    #   更新 Zfindhorizonmax 统计的最大值为 *numpart - numpartinit
    zmax_(Zfindhorizonmax, *numpart - numpartinit);
    #   如果发现了新的最佳结果，则增加 Znewbesthorizon 统计
    if (newbest)
      zinc_(Znewbesthorizon);
  }
  
  # 记录调试信息，输出当前函数的关键参数和变量
  trace4((qh, qh->ferr, 4003, "qh_findbesthorizon: p%d, newbest? %d, bestfacet f%d, bestdist %2.2g, numfacet %d, coplanarfacets %d, numdist %d\n",
    qh_pointid(qh, point), newbest, getid_(bestfacet), *bestdist, numfacet, numcoplanar, *numpart - numpartinit));
  
  # 返回最佳的 facet
  return bestfacet;
}  /* findbesthorizon */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="findbestnew">-</a>

  qh_findbestnew(qh, point, startfacet, dist, isoutside, numpart )
    find best newfacet for point
    searches all of qh.newfacet_list starting at startfacet
    searches horizon facets of coplanar best newfacets
    searches all facets if startfacet == qh.facet_list
  returns:
    best new or horizon facet that is not upperdelaunay
    early out if isoutside and not 'Qf'
    dist is distance to facet
    isoutside is true if point is outside of facet
    numpart is number of distance tests

  notes:
    Always used for merged new facets (see qh_USEfindbestnew)
    Avoids upperdelaunay facet unless (isoutside and outside)

    Uses qh.visit_id, qh.coplanarfacetset.
    If share visit_id with qh_findbest, coplanarfacetset is incorrect.

    If merging (testhorizon), searches horizon facets of coplanar best facets because
    a point maybe coplanar to the bestfacet, below its horizon facet,
    and above a horizon facet of a coplanar newfacet.  For example,
      rbox 1000 s Z1 G1e-13 | qhull
      rbox 1000 s W1e-13 P0 t992110337 | QHULL d Qbb Qc

    qh_findbestnew() used if
       qh_sharpnewfacets -- newfacets contains a sharp angle
       if many merges, qh_premerge found a merge, or 'Qf' (qh.findbestnew)

  see also:
    qh_partitionall() and qh_findbest()

  design:
    for each new facet starting from startfacet
      test distance from point to facet
      return facet if clearly outside
      unless upperdelaunay and a lowerdelaunay exists
         update best facet
    test horizon facets
*/
facetT *qh_findbestnew(qhT *qh, pointT *point, facetT *startfacet,
           realT *dist, boolT bestoutside, boolT *isoutside, int *numpart) {
  realT bestdist= -REALmax/2;  /* Initialize the best distance to a very small value */
  facetT *bestfacet= NULL, *facet;  /* Initialize the best facet and current facet pointers */
  int oldtrace= qh->IStracing, i;  /* Store the current tracing setting */
  unsigned int visitid= ++qh->visit_id;  /* Increment and store the current visit ID */
  realT distoutside= 0.0;  /* Initialize the distance outside */
  boolT isdistoutside; /* True if distoutside is defined */
  boolT testhorizon= True; /* Flag indicating if testing horizon facets is necessary */

  /* Handle topology errors if startfacet is invalid */
  if (!startfacet || !startfacet->next) {
    if (qh->MERGING) {
      qh_fprintf(qh, qh->ferr, 6001, "qhull topology error (qh_findbestnew): merging has formed and deleted a cone of new facets.  Can not continue.\n");
      qh_errexit(qh, qh_ERRtopology, NULL, NULL);
    } else {
      qh_fprintf(qh, qh->ferr, 6002, "qhull internal error (qh_findbestnew): no new facets for point p%d\n",
              qh->furthest_id);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
  }
  
  zinc_(Zfindnew);  /* Increment the Zfindnew counter */

  /* Determine if point is outside the convex hull */
  if (qh->BESToutside || bestoutside)
    isdistoutside= False;
  else {
    isdistoutside= True;
    distoutside= qh_DISToutside; /* multiple of qh.MINoutside & qh.max_outside, see user_r.h */
  }
  
  /* Set the isoutside flag if it is provided */
  if (isoutside)
    *isoutside= True;
  
  *numpart= 0;  /* Initialize the number of distance tests to zero */
#ifndef qh_NOtrace
  // 检查是否需要跟踪信息输出，如果启用了跟踪级别 4 或者 TRACEpoint 指定了且当前点与跟踪点匹配，则输出相关信息
  if (qh->IStracing >= 4 || (qh->TRACElevel && qh->TRACEpoint >= 0 && qh->TRACEpoint == qh_pointid(qh, point))) {
    // 如果跟踪级别大于当前 IS 跟踪级别，则提升 IS 跟踪级别到 TRACElevel
    if (qh->TRACElevel > qh->IStracing)
      qh->IStracing= qh->TRACElevel;
    // 输出点 p%d 和当前最优面 f%d 的信息，包括停止条件和距离信息
    qh_fprintf(qh, qh->ferr, 8008, "qh_findbestnew: point p%d facet f%d. Stop? %d if dist > %2.2g,",
             qh_pointid(qh, point), startfacet->id, isdistoutside, distoutside);
    // 输出最近的添加点信息，包括点 p%d、访问 ID 和顶点访问 ID
    qh_fprintf(qh, qh->ferr, 8009, " Last qh_addpoint p%d, qh.visit_id %d, vertex_visit %d,",  qh->furthest_id, visitid, qh->vertex_visit);
    // 输出最近的合并次数信息
    qh_fprintf(qh, qh->ferr, 8010, " Last merge #%d\n", zzval_(Ztotmerge));
  }
#endif

  /* visit all new facets starting with startfacet, maybe qh->facet_list */
  // 访问所有以 startfacet 开始的新面，可能包括 qh->newfacet_list
  for (i=0, facet=startfacet; i < 2; i++, facet= qh->newfacet_list) {
    FORALLfacet_(facet) {
      // 如果当前面是 startfacet 且已经遍历过一次，则跳出循环
      if (facet == startfacet && i)
        break;
      // 设置当前面的访问 ID 为当前访问 ID
      facet->visitid= visitid;
      // 如果当前面未翻转，则计算点到面的距离，并增加 *numpart 计数
      if (!facet->flipped) {
        qh_distplane(qh, point, facet, dist);
        (*numpart)++;
        // 如果当前距离大于最佳距离，则更新最佳面和距离
        if (*dist > bestdist) {
          // 如果不是上升 Delaunay 或者距离大于等于 MINoutside，则更新最佳面和距离
          if (!facet->upperdelaunay || *dist >= qh->MINoutside) {
            bestfacet= facet;
            // 如果是外部距离且当前距离大于等于 distoutside，则跳转到 LABELreturn_bestnew
            if (isdistoutside && *dist >= distoutside)
              goto LABELreturn_bestnew;
            bestdist= *dist;
          }
        }
      } /* end of !flipped */
    } /* FORALLfacet from startfacet or qh->newfacet_list */
  }

  // 如果 testhorizon 为真或者 bestfacet 未设置，则调用 qh_findbesthorizon 查找最佳的水平面
  if (testhorizon || !bestfacet)
    bestfacet= qh_findbesthorizon(qh, !qh_IScheckmax, point, bestfacet ? bestfacet : startfacet,
                                        !qh_NOupper, &bestdist, numpart);

  // 更新 *dist 并根据条件更新 isoutside
  *dist= bestdist;
  if (isoutside && *dist < qh->MINoutside)
    *isoutside= False;

LABELreturn_bestnew:
  // 更新跟踪统计信息，并输出跟踪信息到日志
  zadd_(Zfindnewtot, *numpart);
  zmax_(Zfindnewmax, *numpart);
  trace4((qh, qh->ferr, 4004, "qh_findbestnew: bestfacet f%d bestdist %2.2g for p%d f%d bestoutside? %d \n",
    getid_(bestfacet), *dist, qh_pointid(qh, point), startfacet->id, bestoutside));
  // 恢复旧的跟踪级别
  qh->IStracing= oldtrace;
  // 返回最佳面
  return bestfacet;
}  /* findbestnew */

/* ============ hyperplane functions -- keep code together [?] ============ */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="backnormal">-</a>

  qh_backnormal(qh, rows, numrow, numcol, sign, normal, nearzero )
    given an upper-triangular rows array and a sign,
    # 解决正规方程式 x = U 的后向代换问题

  # 返回:
  #    normal= x，即解的向量

  # 如果无法使用 divzero() 进行归一化(qh.MINdenom_2 和 qh.MINdenom_1_2)，则会出现以下情况：
  #   - 如果在最后一行失败
  #     这意味着超平面与 [0,..,1] 交点
  #     将正规向量的最后一个坐标设置为 sign
  #   - 否则
  #     将正规向量的尾部设置为 [...,sign,0,...]，即解 b= [0...0]
  #     设置 nearzero

  # 注意:
  #    假设 numrow == numcol-1
  #    参见 Golub & van Loan, 1983 年，Eq. 4.4-9 "高斯消元与完全枢轴选取"

  # 解决 Ux=b，其中 Ax=b 且 PA=LU
  # b= [0,...,0,sign 或 0]  (sign 是 -1 或 +1)
  # A 的最后一行= [0,...,0,1]

  # 设计:
  #   对于每一行从末尾开始
  #     执行后向代换
  #     如果接近零
  #       使用 qh_divzero 进行除法
  #       如果除法为零且不是最后一行
  #         将正规向量的尾部设置为 0
/*
void qh_backnormal(qhT *qh, realT **rows, int numrow, int numcol, boolT sign,
        coordT *normal, boolT *nearzero) {
  int i, j;
  coordT *normalp, *normal_tail, *ai, *ak;
  realT diagonal;
  boolT waszero;
  int zerocol= -1;

  normalp= normal + numcol - 1;
  *normalp--= (sign ? -1.0 : 1.0);
  // 逆向遍历行
  for (i=numrow; i--; ) {
    *normalp= 0.0;
    // 指向当前行对角线元素之后的元素
    ai= rows[i] + i + 1;
    ak= normalp+1;
    // 计算正规化向量的每个分量
    for (j=i+1; j < numcol; j++)
      *normalp -= *ai++ * *ak++;
    diagonal= (rows[i])[i];
    // 若对角线元素绝对值大于最小允许分母的一半，进行正规化
    if (fabs_(diagonal) > qh->MINdenom_2)
      *(normalp--) /= diagonal;
    else {
      waszero= False;
      // 处理除零情况，进行正规化
      *normalp= qh_divzero(*normalp, diagonal, qh->MINdenom_1_2, &waszero);
      if (waszero) {
        // 如果遇到除零情况，记录下该列
        zerocol= i;
        *(normalp--)= (sign ? -1.0 : 1.0);
        // 将正规化向量其余元素置零
        for (normal_tail= normalp+2; normal_tail < normal + numcol; normal_tail++)
          *normal_tail= 0.0;
      }else
        normalp--;
    }
  }
  // 如果存在除零列，设置 nearzero 标志并进行相关记录
  if (zerocol != -1) {
    *nearzero= True;
    trace4((qh, qh->ferr, 4005, "qh_backnormal: zero diagonal at column %d.\n", i));
    zzinc_(Zback0);
    qh_joggle_restart(qh, "zero diagonal on back substitution");
  }
} /* backnormal */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="gausselim">-</a>

  qh_gausselim(qh, rows, numrow, numcol, sign )
    Gaussian elimination with partial pivoting

  returns:
    rows is upper triangular (includes row exchanges)
    flips sign for each row exchange
    sets nearzero if pivot[k] < qh.NEARzero[k], else clears it

  notes:
    if nearzero, the determinant's sign may be incorrect.
    assumes numrow <= numcol

  design:
    for each row
      determine pivot and exchange rows if necessary
      test for near zero
      perform gaussian elimination step
*/
void qh_gausselim(qhT *qh, realT **rows, int numrow, int numcol, boolT *sign, boolT *nearzero) {
  realT *ai, *ak, *rowp, *pivotrow;
  realT n, pivot, pivot_abs= 0.0, temp;
  int i, j, k, pivoti, flip=0;

  *nearzero= False;
  // 高斯消元法过程
  for (k=0; k < numrow; k++) {
    // 确定主元并根据需要交换行
    pivot_abs= fabs_((rows[k])[k]);
    pivoti= k;
    for (i=k+1; i < numrow; i++) {
      if ((temp= fabs_((rows[i])[k])) > pivot_abs) {
        pivot_abs= temp;
        pivoti= i;
      }
    }
    if (pivoti != k) {
      rowp= rows[pivoti];
      rows[pivoti]= rows[k];
      rows[k]= rowp;
      *sign ^= 1;
      flip ^= 1;
    }
    // 如果主元绝对值小于阈值，设置 nearzero 标志
    if (pivot_abs <= qh->NEARzero[k]) {
      *nearzero= True;
      // 如果主元为零，输出跟踪信息并做相关处理
#ifndef qh_NOtrace
      if (qh->IStracing >= 4) {
        qh_fprintf(qh, qh->ferr, 8011, "qh_gausselim: 0 pivot at column %d. (%2.2g < %2.2g)\n", k, pivot_abs, qh->DISTround);
        qh_printmatrix(qh, qh->ferr, "Matrix:", rows, numrow, numcol);
      }
#endif
      zzinc_(Zgauss0);
      qh_joggle_restart(qh, "zero pivot for Gaussian elimination");
      // 跳转到下一列处理
      goto LABELnextcol;
    }
    pivotrow= rows[k] + k;
    pivot= *pivotrow++;  /* 获取pivot的有符号值，并移动到下一行 */
    for (i=k+1; i < numrow; i++) {
      ai= rows[i] + k;
      ak= pivotrow;
      n= (*ai++)/pivot;   /* 计算除以pivot后的商，这里不需要处理除零情况，因为|pivot| >= |*ai| */
      for (j= numcol - (k+1); j--; )
        *ai++ -= n * *ak++;  /* 更新当前行的元素 */
    }
  LABELnextcol:
    ;  /* 空语句标签，用于跳转到下一列的处理 */
  }
  wmin_(Wmindenom, pivot_abs);  /* 更新最后一个pivot元素 */
  if (qh->IStracing >= 5)
    qh_printmatrix(qh, qh->ferr, "qh_gausselem: result", rows, numrow, numcol);
/*-------------------------------------
  qh_getangle(qh, vect1, vect2 )
    返回两个向量的点积
    如果 qh.RANDOMdist 为真，则对结果进行微调

  注解：
    由于舍入误差，角度可能大于1.0或小于-1.0
---------------------------------------*/
realT qh_getangle(qhT *qh, pointT *vect1, pointT *vect2) {
  realT angle= 0, randr;
  int k;

  for (k=qh->hull_dim; k--; )
    angle += *vect1++ * *vect2++;
  if (qh->RANDOMdist) {
    randr= qh_RANDOMint;
    angle += (2.0 * randr / qh_RANDOMmax - 1.0) *
      qh->RANDOMfactor;
  }
  trace4((qh, qh->ferr, 4006, "qh_getangle: %4.4g\n", angle));
  return(angle);
} /* getangle */


/*-------------------------------------
  qh_getcenter(qh, vertices )
    返回一组顶点的算术中心作为一个新的点

  注解：
    为中心分配点数组
---------------------------------------*/
pointT *qh_getcenter(qhT *qh, setT *vertices) {
  int k;
  pointT *center, *coord;
  vertexT *vertex, **vertexp;
  int count= qh_setsize(qh, vertices);

  if (count < 2) {
    qh_fprintf(qh, qh->ferr, 6003, "qhull internal error (qh_getcenter): not defined for %d points\n", count);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  center= (pointT *)qh_memalloc(qh, qh->normal_size);
  for (k=0; k < qh->hull_dim; k++) {
    coord= center+k;
    *coord= 0.0;
    FOREACHvertex_(vertices)
      *coord += vertex->point[k];
    *coord /= count;  /* count>=2 by QH6003 */
  }
  return(center);
} /* getcenter */


/*-------------------------------------
  qh_getcentrum(qh, facet )
    返回面的中心作为一个新的点

  注解：
    分配中心
---------------------------------------*/
pointT *qh_getcentrum(qhT *qh, facetT *facet) {
  realT dist;
  pointT *centrum, *point;

  point= qh_getcenter(qh, facet->vertices);
  zzinc_(Zcentrumtests);
  qh_distplane(qh, point, facet, &dist);
  centrum= qh_projectpoint(qh, point, facet, dist);
  qh_memfree(qh, point, qh->normal_size);
  trace4((qh, qh->ferr, 4007, "qh_getcentrum: for f%d, %d vertices dist= %2.2g\n",
          facet->id, qh_setsize(qh, facet->vertices), dist));
  return centrum;
} /* getcentrum */


/*-------------------------------------
  qh_getdistance(qh, facet, neighbor, mindist, maxdist )
    返回面内非邻近顶点到邻居的最小和最大距离

  返回：
    最大绝对值

  设计：
    对于面的每个不在邻居中的顶点
      测试顶点到邻居的距离
---------------------------------------*/
coordT qh_getdistance(qhT *qh, facetT *facet, facetT *neighbor, coordT *mindist, coordT *maxdist) {
  vertexT *vertex, **vertexp;
  coordT dist, maxd, mind;

  FOREACHvertex_(facet->vertices)
    vertex->seen= False;
  FOREACHvertex_(neighbor->vertices)
    vertex->seen= True;
    // 将顶点的seen标记设置为True，表示顶点已被处理过
    vertex->seen= True;
    // 初始化最小距离mind和最大距离maxd
    mind= 0.0;
    maxd= 0.0;
    // 遍历当前facet的所有顶点
    FOREACHvertex_(facet->vertices) {
        // 如果顶点未被处理过
        if (!vertex->seen) {
            // 增加Zbestdist计数
            zzinc_(Zbestdist);
            // 计算顶点到邻居平面的距离
            qh_distplane(qh, vertex->point, neighbor, &dist);
            // 更新最小距离mind和最大距离maxd
            if (dist < mind)
                mind= dist;
            else if (dist > maxd)
                maxd= dist;
        }
    }
    // 将最小距离和最大距离存储到输出参数mindist和maxdist中
    *mindist= mind;
    *maxdist= maxd;
    // 将mind取反
    mind= -mind;
    // 如果最大距离大于取反后的最小距离，则返回最大距离，否则返回取反后的最小距离
    if (maxd > mind)
        return maxd;
    else
        return mind;
} /* getdistance */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="normalize">-</a>

  qh_normalize(qh, normal, dim, toporient )
    normalize a vector and report if too small
    does not use min norm

  see:
    qh_normalize2
*/
void qh_normalize(qhT *qh, coordT *normal, int dim, boolT toporient) {
  // 调用 qh_normalize2 函数，对法向量进行标准化处理，不考虑最小规范化
  qh_normalize2(qh, normal, dim, toporient, NULL, NULL);
} /* normalize */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="normalize2">-</a>

  qh_normalize2(qh, normal, dim, toporient, minnorm, ismin )
    normalize a vector and report if too small
    qh.MINdenom/MINdenom1 are the upper limits for divide overflow

  returns:
    normalized vector
    flips sign if !toporient
    if minnorm non-NULL,
      sets ismin if normal < minnorm

  notes:
    if zero norm
       sets all elements to sqrt(1.0/dim)
    if divide by zero (divzero())
       sets largest element to   +/-1
       bumps Znearlysingular

  design:
    computes norm
    test for minnorm
    if not near zero
      normalizes normal
    else if zero norm
      sets normal to standard value
    else
      uses qh_divzero to normalize
      if nearzero
        sets norm to direction of maximum value
*/
void qh_normalize2(qhT *qh, coordT *normal, int dim, boolT toporient,
            realT *minnorm, boolT *ismin) {
  int k;
  realT *colp, *maxp, norm= 0, temp, *norm1, *norm2, *norm3;
  boolT zerodiv;

  // 定义法向量的分量指针
  norm1= normal+1;
  norm2= normal+2;
  norm3= normal+3;

  // 根据维度不同计算法向量的模
  if (dim == 2)
    norm= sqrt((*normal)*(*normal) + (*norm1)*(*norm1));
  else if (dim == 3)
    norm= sqrt((*normal)*(*normal) + (*norm1)*(*norm1) + (*norm2)*(*norm2));
  else if (dim == 4) {
    norm= sqrt((*normal)*(*normal) + (*norm1)*(*norm1) + (*norm2)*(*norm2)
               + (*norm3)*(*norm3));
  }else if (dim > 4) {
    norm= (*normal)*(*normal) + (*norm1)*(*norm1) + (*norm2)*(*norm2)
               + (*norm3)*(*norm3);
    for (k=dim-4, colp=normal+4; k--; colp++)
      norm += (*colp) * (*colp);
    norm= sqrt(norm);
  }

  // 如果有最小规范化要求，则进行判断和处理
  if (minnorm) {
    if (norm < *minnorm)
      *ismin= True;
    else
      *ismin= False;
  }

  // 更新最小分母限制
  wmin_(Wmindenom, norm);

  // 如果法向量模大于最小分母限制，则进行标准化处理
  if (norm > qh->MINdenom) {
    if (!toporient)
      norm= -norm;
    *normal /= norm;
    *norm1 /= norm;
    if (dim == 2)
      ; /* all done */
    else if (dim == 3)
      *norm2 /= norm;
    else if (dim == 4) {
      *norm2 /= norm;
      *norm3 /= norm;
    }else if (dim >4) {
      *norm2 /= norm;
      *norm3 /= norm;
      for (k=dim-4, colp=normal+4; k--; )
        *colp++ /= norm;
    }
  }else if (norm == 0.0) {
    // 如果法向量模为零，则设置法向量各分量为标准值
    temp= sqrt(1.0/dim);
    for (k=dim, colp=normal; k--; )
      *colp++= temp;
  }else {
    // 如果法向量模不大于最小分母限制，则进行特殊处理
    if (!toporient)
      norm= -norm;
    for (k=dim, colp=normal; k--; colp++) { /* k used below */
      // 对于每个维度 k，遍历法向量 normal 中的元素 colp
      temp= qh_divzero(*colp, norm, qh->MINdenom_1, &zerodiv);
      // 计算 *colp / norm，避免零除错误，将结果存入 temp，并检查是否发生零除
      if (!zerodiv)
        // 如果没有发生零除，更新 *colp 为 temp 的值
        *colp= temp;
      else {
        // 如果发生零除
        maxp= qh_maxabsval(normal, dim);
        // 找到法向量 normal 中绝对值最大的元素的指针，存入 maxp
        temp= ((*maxp * norm >= 0.0) ? 1.0 : -1.0);
        // 根据 maxp 所指元素的符号和 norm 的符号设置 temp 的值为 1.0 或 -1.0
        for (k=dim, colp=normal; k--; colp++)
          // 将法向量 normal 中的所有元素置为 0.0
          *colp= 0.0;
        *maxp= temp;
        // 更新 maxp 所指元素的值为 temp
        zzinc_(Znearlysingular);
        // 增加 Znearlysingular 的计数器
        /* qh_joggle_restart ignored for Znearlysingular, normal part of qh_sethyperplane_gauss */
        // 输出消息到跟踪日志，指示法向量太小导致近似奇异
        trace0((qh, qh->ferr, 1, "qh_normalize: norm=%2.2g too small during p%d\n",
               norm, qh->furthest_id));
        // 返回函数，标志处理中止
        return;
      }
    }
} /* normalize */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="projectpoint">-</a>

  qh_projectpoint(qh, point, facet, dist )
    project point onto a facet by dist

  returns:
    returns a new point

  notes:
    if dist= distplane(point,facet)
      this projects point to hyperplane
    assumes qh_memfree_() is valid for normal_size
*/
pointT *qh_projectpoint(qhT *qh, pointT *point, facetT *facet, realT dist) {
  pointT *newpoint, *np, *normal;
  int normsize= qh->normal_size;
  int k;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */

  // 分配内存以存储新的点
  qh_memalloc_(qh, normsize, freelistp, newpoint, pointT);
  np= newpoint;
  normal= facet->normal;
  // 对每个维度进行投影计算
  for (k=qh->hull_dim; k--; )
    *(np++)= *point++ - dist * *normal++;
  return(newpoint);
} /* projectpoint */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="setfacetplane">-</a>

  qh_setfacetplane(qh, facet )
    sets the hyperplane for a facet
    if qh.RANDOMdist, joggles hyperplane

  notes:
    uses global buffers qh.gm_matrix and qh.gm_row
    overwrites facet->normal if already defined
    updates Wnewvertex if PRINTstatistics
    sets facet->upperdelaunay if upper envelope of Delaunay triangulation

  design:
    copy vertex coordinates to qh.gm_matrix/gm_row
    compute determinate
    if nearzero
      recompute determinate with gaussian elimination
      if nearzero
        force outside orientation by testing interior point
*/
void qh_setfacetplane(qhT *qh, facetT *facet) {
  pointT *point;
  vertexT *vertex, **vertexp;
  int normsize= qh->normal_size;
  int k,i, oldtrace= 0;
  realT dist;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */
  coordT *coord, *gmcoord;
  pointT *point0= SETfirstt_(facet->vertices, vertexT)->point;
  boolT nearzero= False;

  // 增加平面设置计数
  zzinc_(Zsetplane);
  // 如果还未定义法向量，分配内存
  if (!facet->normal)
    qh_memalloc_(qh, normsize, freelistp, facet->normal, coordT);
#ifndef qh_NOtrace
  // 跟踪调试信息的处理
  if (facet == qh->tracefacet) {
    oldtrace= qh->IStracing;
    qh->IStracing= 5;
    qh_fprintf(qh, qh->ferr, 8012, "qh_setfacetplane: facet f%d created.\n", facet->id);
    qh_fprintf(qh, qh->ferr, 8013, "  Last point added to hull was p%d.", qh->furthest_id);
    if (zzval_(Ztotmerge))
      qh_fprintf(qh, qh->ferr, 8014, "  Last merge was #%d.", zzval_(Ztotmerge));
    qh_fprintf(qh, qh->ferr, 8015, "\n\nCurrent summary is:\n");
    qh_printsummary(qh, qh->ferr);
  }
#endif
  // 如果维度小于等于4，进行以下操作
  if (qh->hull_dim <= 4) {
    i= 0;
    // 如果启用随机扰动，对顶点坐标进行随机扰动
    if (qh->RANDOMdist) {
      gmcoord= qh->gm_matrix;
      FOREACHvertex_(facet->vertices) {
        qh->gm_row[i++]= gmcoord;
        coord= vertex->point;
        for (k=qh->hull_dim; k--; )
          *(gmcoord++)= *coord++ * qh_randomfactor(qh, qh->RANDOMa, qh->RANDOMb);
      }
    } else {
      // 否则直接复制顶点坐标到 gm_row
      FOREACHvertex_(facet->vertices)
        qh->gm_row[i++]= vertex->point;
    }
    qh_sethyperplane_det(qh, qh->hull_dim, qh->gm_row, point0, facet->toporient,
                facet->normal, &facet->offset, &nearzero);
  }
  // 如果凸包维度大于4或者 nearzero 为真，则执行以下操作
  if (qh->hull_dim > 4 || nearzero) {
    // 初始化变量 i 为 0
    i= 0;
    // 获取 gm_matrix 的起始地址
    gmcoord= qh->gm_matrix;
    // 遍历 facet 的顶点列表
    FOREACHvertex_(facet->vertices) {
      // 如果当前顶点不是 point0
      if (vertex->point != point0) {
        // 将 gm_matrix 的当前行指针设为 gmcoord
        qh->gm_row[i++]= gmcoord;
        // 获取当前顶点的坐标赋值给 coord
        coord= vertex->point;
        // 将 point 指向的坐标赋值给 point
        point= point0;
        // 针对凸包维度进行循环
        for (k=qh->hull_dim; k--; )
          // 计算并存储当前顶点相对于 point0 的坐标差值到 gm_matrix
          *(gmcoord++)= *coord++ - *point++;
      }
    }
    // 为了 areasimplex，将当前行指针设为 gmcoord
    qh->gm_row[i]= gmcoord;  /* for areasimplex */
    // 如果设置了 RANDOMdist
    if (qh->RANDOMdist) {
      // 重新初始化 gmcoord 为 gm_matrix 的起始地址
      gmcoord= qh->gm_matrix;
      // 针对凸包维度进行两次循环
      for (i=qh->hull_dim-1; i--; ) {
        for (k=qh->hull_dim; k--; )
          // 将 gm_matrix 中的元素乘以随机因子
          *(gmcoord++) *= qh_randomfactor(qh, qh->RANDOMa, qh->RANDOMb);
      }
    }
    // 根据 Gaussian 消元法设置超平面
    qh_sethyperplane_gauss(qh, qh->hull_dim, qh->gm_row, point0, facet->toporient,
                facet->normal, &facet->offset, &nearzero);
    // 如果 nearzero 为真
    if (nearzero) {
      // 如果确定外部定向被翻转
      if (qh_orientoutside(qh, facet)) {
        // 输出消息表明由于 nearzero 高斯和 interior_point 测试而翻转了定向
        trace0((qh, qh->ferr, 2, "qh_setfacetplane: flipped orientation due to nearzero gauss and interior_point test.  During p%d\n", qh->furthest_id));
      /* this is part of using Gaussian Elimination.  For example in 5-d
           1 1 1 1 0
           1 1 1 1 1
           0 0 0 1 0
           0 1 0 0 0
           1 0 0 0 0
           norm= 0.38 0.38 -0.76 0.38 0
         has a determinate of 1, but g.e. after subtracting pt. 0 has
         0's in the diagonal, even with full pivoting.  It does work
         if you subtract pt. 4 instead. */
      }
    }
  }
  // 将 facet 的 upperdelaunay 设为 False
  facet->upperdelaunay= False;
  // 如果设置了 DELAUNAY
  if (qh->DELAUNAY) {
    // 如果设置了 UPPERdelaunay
    if (qh->UPPERdelaunay) {     /* matches qh_triangulate_facet and qh.lower_threshold in qh_initbuild */
      // 如果 facet 的法向量的最后一个分量大于等于 ANGLEround 乘以 ZEROdelaunay
      if (facet->normal[qh->hull_dim -1] >= qh->ANGLEround * qh_ZEROdelaunay)
        // 将 facet 的 upperdelaunay 设为 True
        facet->upperdelaunay= True;
    }else {
      // 如果 facet 的法向量的最后一个分量大于 -ANGLEround 乘以 ZEROdelaunay
      if (facet->normal[qh->hull_dim -1] > -qh->ANGLEround * qh_ZEROdelaunay)
        // 将 facet 的 upperdelaunay 设为 True
        facet->upperdelaunay= True;
    }
  }
  // 如果设置了 PRINTstatistics、IStracing、TRACElevel 或 JOGGLEmax 小于 REALmax
  if (qh->PRINTstatistics || qh->IStracing || qh->TRACElevel || qh->JOGGLEmax < REALmax) {
    // 将 old_randomdist 设为 RANDOMdist 的当前值
    qh->old_randomdist= qh->RANDOMdist;
    // 将 RANDOMdist 设为 False
    qh->RANDOMdist= False;
    FOREACHvertex_(facet->vertices) {
        // 遍历当前面（facet）的顶点列表中的每个顶点（vertex）
        if (vertex->point != point0) {
            // 如果顶点的坐标与 point0 不相等
            boolT istrace= False;
            // 初始化一个布尔变量 istrace 为 False
            zinc_(Zdiststat);
            // 增加 Zdiststat 计数器的值
            qh_distplane(qh, vertex->point, facet, &dist);
            // 计算顶点 vertex 到面 facet 的距离，并将结果存储在 dist 中
            dist= fabs_(dist);
            // 取 dist 的绝对值
            zinc_(Znewvertex);
            // 增加 Znewvertex 计数器的值
            wadd_(Wnewvertex, dist);
            // 将 dist 加到 Wnewvertex 统计中
            if (dist > wwval_(Wnewvertexmax)) {
                // 如果 dist 大于当前 Wnewvertexmax 的值
                wwval_(Wnewvertexmax)= dist;
                // 更新 Wnewvertexmax 的值为 dist
                if (dist > qh->max_outside) {
                    // 如果 dist 大于当前 qh->max_outside 的值
                    qh->max_outside= dist;  /* used by qh_maxouter(qh) */
                    // 更新 qh->max_outside，并标记为用于 qh_maxouter(qh) 函数
                    if (dist > qh->TRACEdist)
                        istrace= True;
                    // 如果 dist 大于 qh->TRACEdist，则设置 istrace 为 True
                }
            } else if (-dist > qh->TRACEdist)
                // 否则如果 -dist 大于 qh->TRACEdist
                istrace= True;
                // 设置 istrace 为 True
            if (istrace) {
                // 如果 istrace 为 True
                qh_fprintf(qh, qh->ferr, 3060, "qh_setfacetplane: ====== vertex p%d(v%d) increases max_outside to %2.2g for new facet f%d last p%d\n",
                    qh_pointid(qh, vertex->point), vertex->id, dist, facet->id, qh->furthest_id);
                // 打印调试信息到 qh->ferr 流中
                qh_errprint(qh, "DISTANT", facet, NULL, NULL, NULL);
                // 打印 "DISTANT" 类型的错误信息到 qh_errprint 函数中
            }
        }
    }
    qh->RANDOMdist= qh->old_randomdist;
    // 将 qh->RANDOMdist 设置为 qh->old_randomdist 的值
}


这段代码是一个循环，遍历一个面（facet）的顶点列表（vertices），对每个顶点进行处理。具体操作包括计算顶点到面的距离，更新距离统计数据，并根据条件调整一些数据结构和打印调试信息。
#ifndef qh_NOtrace
  // 如果跟踪未关闭且跟踪级别大于等于4
  if (qh->IStracing >= 4) {
    // 输出面的信息，包括面的 ID、偏移量和法向量
    qh_fprintf(qh, qh->ferr, 8017, "qh_setfacetplane: f%d offset %2.2g normal: ",
             facet->id, facet->offset);
    // 输出法向量的每个分量
    for (k=0; k < qh->hull_dim; k++)
      qh_fprintf(qh, qh->ferr, 8018, "%2.2g ", facet->normal[k]);
    // 换行结束输出
    qh_fprintf(qh, qh->ferr, 8019, "\n");
  }
#endif
// 检查面的方向是否被翻转，不检查共面性
  qh_checkflipped(qh, facet, NULL, qh_ALL);
// 如果当前面是追踪的面
  if (facet == qh->tracefacet) {
    // 恢复原始跟踪级别
    qh->IStracing= oldtrace;
    // 输出当前面的信息
    qh_printfacet(qh, qh->ferr, facet);
  }
} /* setfacetplane */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sethyperplane_det">-</a>

  qh_sethyperplane_det(qh, dim, rows, point0, toporient, normal, offset, nearzero )
    given dim X dim array indexed by rows[], one row per point,
        toporient(flips all signs),
        and point0 (any row)
    set normalized hyperplane equation from oriented simplex

  returns:
    normal (normalized)
    offset (places point0 on the hyperplane)
    sets nearzero if hyperplane not through points

  notes:
    only defined for dim == 2..4
    rows[] is not modified
    solves det(P-V_0, V_n-V_0, ..., V_1-V_0)=0, i.e. every point is on hyperplane
    see Bower & Woodworth, A programmer's geometry, Butterworths 1983.

  derivation of 3-d minnorm
    Goal: all vertices V_i within qh.one_merge of hyperplane
    Plan: exactly translate the facet so that V_0 is the origin
          exactly rotate the facet so that V_1 is on the x-axis and y_2=0.
          exactly rotate the effective perturbation to only effect n_0
             this introduces a factor of sqrt(3)
    n_0 = ((y_2-y_0)*(z_1-z_0) - (z_2-z_0)*(y_1-y_0)) / norm
    Let M_d be the max coordinate difference
    Let M_a be the greater of M_d and the max abs. coordinate
    Let u be machine roundoff and distround be max error for distance computation
    The max error for n_0 is sqrt(3) u M_a M_d / norm.  n_1 is approx. 1 and n_2 is approx. 0
    The max error for distance of V_1 is sqrt(3) u M_a M_d M_d / norm.  Offset=0 at origin
    Then minnorm = 1.8 u M_a M_d M_d / qh.ONEmerge
    Note that qh.one_merge is approx. 45.5 u M_a and norm is usually about M_d M_d

  derivation of 4-d minnorm
    same as above except rotate the facet so that V_1 on x-axis and w_2, y_3, w_3=0
     [if two vertices fixed on x-axis, can rotate the other two in yzw.]
    n_0 = det3_(...) = y_2 det2_(z_1, w_1, z_3, w_3) = - y_2 w_1 z_3
     [all other terms contain at least two factors nearly zero.]
    The max error for n_0 is sqrt(4) u M_a M_d M_d / norm
    Then minnorm = 2 u M_a M_d M_d M_d / qh.ONEmerge
    Note that qh.one_merge is approx. 82 u M_a and norm is usually about M_d M_d M_d
*/
void qh_sethyperplane_det(qhT *qh, int dim, coordT **rows, coordT *point0,
          boolT toporient, coordT *normal, realT *offset, boolT *nearzero) {
  realT maxround, dist;
  int i;
  pointT *point;


  if (dim == 2) {
    // 设置二维情况下的法向量
    normal[0]= dY(1,0);
    normal[1]= dX(0,1);
    qh_normalize2(qh, normal, dim, toporient, NULL, NULL);
    # 调用函数 qh_normalize2 对法向量 normal 进行归一化处理
    *offset= -(point0[0]*normal[0]+point0[1]*normal[1]);
    # 计算偏移量 offset，即点 point0 关于法向量 normal 的投影
    *nearzero= False;  # 因为法向量近似为零 => 存在相交点
  }else if (dim == 3) {
    normal[0]= det2_(dY(2,0), dZ(2,0),
                     dY(1,0), dZ(1,0));
    # 计算三维空间中的法向量 normal[0]
    normal[1]= det2_(dX(1,0), dZ(1,0),
                     dX(2,0), dZ(2,0));
    # 计算三维空间中的法向量 normal[1]
    normal[2]= det2_(dX(2,0), dY(2,0),
                     dX(1,0), dY(1,0));
    # 计算三维空间中的法向量 normal[2]
    qh_normalize2(qh, normal, dim, toporient, NULL, NULL);
    # 调用函数 qh_normalize2 对法向量 normal 进行归一化处理
    *offset= -(point0[0]*normal[0] + point0[1]*normal[1]
               + point0[2]*normal[2]);
    # 计算偏移量 offset，即点 point0 关于法向量 normal 的投影
    maxround= qh->DISTround;
    # 设置最大容差值为 qh->DISTround
    for (i=dim; i--; ) {
      point= rows[i];
      # 遍历每个点 rows[i]
      if (point != point0) {
        # 如果当前点不是 point0
        dist= *offset + (point[0]*normal[0] + point[1]*normal[1]
               + point[2]*normal[2]);
        # 计算当前点到超平面的距离 dist
        if (dist > maxround || dist < -maxround) {
          *nearzero= True;
          # 如果距离超过容差范围，则认为存在相交点
          break;
        }
      }
    }
  }else if (dim == 4) {
    normal[0]= - det3_(dY(2,0), dZ(2,0), dW(2,0),
                        dY(1,0), dZ(1,0), dW(1,0),
                        dY(3,0), dZ(3,0), dW(3,0));
    # 计算四维空间中的法向量 normal[0]
    normal[1]=   det3_(dX(2,0), dZ(2,0), dW(2,0),
                        dX(1,0), dZ(1,0), dW(1,0),
                        dX(3,0), dZ(3,0), dW(3,0));
    # 计算四维空间中的法向量 normal[1]
    normal[2]= - det3_(dX(2,0), dY(2,0), dW(2,0),
                        dX(1,0), dY(1,0), dW(1,0),
                        dX(3,0), dY(3,0), dW(3,0));
    # 计算四维空间中的法向量 normal[2]
    normal[3]=   det3_(dX(2,0), dY(2,0), dZ(2,0),
                        dX(1,0), dY(1,0), dZ(1,0),
                        dX(3,0), dY(3,0), dZ(3,0));
    # 计算四维空间中的法向量 normal[3]
    qh_normalize2(qh, normal, dim, toporient, NULL, NULL);
    # 调用函数 qh_normalize2 对法向量 normal 进行归一化处理
    *offset= -(point0[0]*normal[0] + point0[1]*normal[1]
               + point0[2]*normal[2] + point0[3]*normal[3]);
    # 计算偏移量 offset，即点 point0 关于法向量 normal 的投影
    maxround= qh->DISTround;
    # 设置最大容差值为 qh->DISTround
    for (i=dim; i--; ) {
      point= rows[i];
      # 遍历每个点 rows[i]
      if (point != point0) {
        # 如果当前点不是 point0
        dist= *offset + (point[0]*normal[0] + point[1]*normal[1]
               + point[2]*normal[2] + point[3]*normal[3]);
        # 计算当前点到超平面的距离 dist
        if (dist > maxround || dist < -maxround) {
          *nearzero= True;
          # 如果距离超过容差范围，则认为存在相交点
          break;
        }
      }
    }
  }
  if (*nearzero) {
    zzinc_(Zminnorm);
    # 增加 Zminnorm 的值
    /* qh_joggle_restart not needed, will call qh_sethyperplane_gauss instead */
    trace0((qh, qh->ferr, 3, "qh_sethyperplane_det: degenerate norm during p%d, use qh_sethyperplane_gauss instead.\n", qh->furthest_id));
    # 记录日志，指示使用 qh_sethyperplane_gauss 替代 qh_joggle_restart 处理异常法向量
  }
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sethyperplane_gauss">-</a>

  qh_sethyperplane_gauss(qh, dim, rows, point0, toporient, normal, offset, nearzero )
    given(dim-1) X dim array of rows[i]= V_{i+1} - V_0 (point0)
    set normalized hyperplane equation from oriented simplex

  returns:
    normal (normalized)
    offset (places point0 on the hyperplane)

  notes:
    if nearzero
      orientation may be incorrect because of incorrect sign flips in gausselim
    solves [V_n-V_0,...,V_1-V_0, 0 .. 0 1] * N == [0 .. 0 1]
        or [V_n-V_0,...,V_1-V_0, 0 .. 0 1] * N == [0]
    i.e., N is normal to the hyperplane, and the unnormalized
        distance to [0 .. 1] is either 1 or   0

  design:
    perform gaussian elimination
    flip sign for negative values
    perform back substitution
    normalize result
    compute offset
*/
void qh_sethyperplane_gauss(qhT *qh, int dim, coordT **rows, pointT *point0,
                boolT toporient, coordT *normal, coordT *offset, boolT *nearzero) {
  coordT *pointcoord, *normalcoef;
  int k;
  boolT sign= toporient, nearzero2= False;

  // 调用高斯消元函数计算简化的超平面方程
  qh_gausselim(qh, rows, dim-1, dim, &sign, nearzero);
  
  // 检查对角线元素的符号，根据需要翻转符号
  for (k=dim-1; k--; ) {
    if ((rows[k])[k] < 0)
      sign ^= 1;
  }
  
  // 如果 nearzero 标志位被设置，表明由于高斯消元的符号错误，可能导致方向错误
  if (*nearzero) {
    zzinc_(Znearlysingular);
    /* qh_joggle_restart ignored for Znearlysingular, normal part of qh_sethyperplane_gauss */
    // 输出警告信息，标记近似奇异超平面或轴平行超平面
    trace0((qh, qh->ferr, 4, "qh_sethyperplane_gauss: nearly singular or axis parallel hyperplane during p%d.\n", qh->furthest_id));
    // 根据当前状态重新计算法向量
    qh_backnormal(qh, rows, dim-1, dim, sign, normal, &nearzero2);
  } else {
    // 否则，正常计算法向量
    qh_backnormal(qh, rows, dim-1, dim, sign, normal, &nearzero2);
    // 如果标志位 nearzero2 被设置，表明在归一化过程中发现了奇异或轴平行的超平面
    if (nearzero2) {
      zzinc_(Znearlysingular);
      // 输出警告信息，标记归一化过程中发现的奇异或轴平行超平面
      trace0((qh, qh->ferr, 5, "qh_sethyperplane_gauss: singular or axis parallel hyperplane at normalization during p%d.\n", qh->furthest_id));
    }
  }
  
  // 如果 nearzero2 被设置，更新 *nearzero 以便外部函数知晓
  if (nearzero2)
    *nearzero= True;
  
  // 对法向量进行归一化处理
  qh_normalize2(qh, normal, dim, True, NULL, NULL);
  
  // 计算超平面的偏移量
  pointcoord= point0;
  normalcoef= normal;
  *offset= -(*pointcoord++ * *normalcoef++);
  for (k=dim-1; k--; )
    *offset -= *pointcoord++ * *normalcoef++;
} /* sethyperplane_gauss */
```