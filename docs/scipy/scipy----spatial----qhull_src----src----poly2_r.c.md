# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\poly2_r.c`

```
/*<html><pre>  -<a                             href="qh-poly_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly2_r.c
   implements polygons and simplicies

   see qh-poly_r.htm, poly_r.h and libqhull_r.h

   frequently used code is in poly_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/poly2_r.c#18 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*======== functions in alphabetical order ==========*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="addfacetvertex">-</a>

  qh_addfacetvertex(qh, facet, newvertex )
    add newvertex to facet.vertices if not already there
    vertices are inverse sorted by vertex->id

  returns:
    True if new vertex for facet

  notes:
    see qh_replacefacetvertex
*/
boolT qh_addfacetvertex(qhT *qh, facetT *facet, vertexT *newvertex) {
  vertexT *vertex;
  int vertex_i= 0, vertex_n;
  boolT isnew= True;

  FOREACHvertex_i_(qh, facet->vertices) {
    if (vertex->id < newvertex->id) {
      break;
    }else if (vertex->id == newvertex->id) {
      isnew= False;
      break;
    }
  }
  if (isnew)
    qh_setaddnth(qh, &facet->vertices, vertex_i, newvertex);
  return isnew;
} /* addfacetvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="addhash">-</a>

  qh_addhash( newelem, hashtable, hashsize, hash )
    add newelem to linear hash table at hash if not already there
*/
void qh_addhash(void *newelem, setT *hashtable, int hashsize, int hash) {
  int scan;
  void *elem;

  for (scan= (int)hash; (elem= SETelem_(hashtable, scan));
       scan= (++scan >= hashsize ? 0 : scan)) {
    if (elem == newelem)
      break;
  }
  /* loop terminates because qh_HASHfactor >= 1.1 by qh_initbuffers */
  if (!elem)
    SETelem_(hashtable, scan)= newelem;
} /* addhash */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_bestdist">-</a>

  qh_check_bestdist(qh)
    check that all points are within max_outside of the nearest facet
    if qh.ONLYgood,
      ignores !good facets

  see:
    qh_check_maxout(), qh_outerinner()

  notes:
    only called from qh_check_points()
      seldom used since qh.MERGING is almost always set
    if notverified>0 at end of routine
      some points were well inside the hull.  If the hull contains
      a lens-shaped component, these points were not verified.  Use
      options 'Qi Tv' to verify all points.  (Exhaustive check also verifies)

  design:
    determine facet for each point (if any)
    for each point
      start with the assigned facet or with the first facet
      find the best facet for the point and check all coplanar facets
      error if point is outside of facet
*/
/* 检查最佳距离函数，用于验证所有点是否在凸壳内部 */

void qh_check_bestdist(qhT *qh) {
  boolT waserror= False, unassigned;
  facetT *facet, *bestfacet, *errfacet1= NULL, *errfacet2= NULL;
  facetT *facetlist;
  realT dist, maxoutside, maxdist= -REALmax;
  pointT *point;
  int numpart= 0, facet_i, facet_n, notgood= 0, notverified= 0;
  setT *facets;

  /* 输出跟踪信息到文件流 qh->ferr */
  trace1((qh, qh->ferr, 1020, "qh_check_bestdist: check points below nearest facet.  Facet_list f%d\n",
      qh->facet_list->id));

  // 计算最大外部距离
  maxoutside= qh_maxouter(qh);
  maxoutside += qh->DISTround;  /* 增加一个 qh.DISTround 用于检查计算 */

  /* 输出跟踪信息到文件流 qh->ferr */
  trace1((qh, qh->ferr, 1021, "qh_check_bestdist: check that all points are within %2.2g of best facet\n", maxoutside));

  // 获取所有点的相关面集合
  facets= qh_pointfacet(qh /* qh.facet_list */);

  // 如果不是快速帮助模式并且设置了打印精度信息
  if (!qh_QUICKhelp && qh->PRINTprecision)
    // 打印验证完成的 qhull 输出信息
    qh_fprintf(qh, qh->ferr, 8091, "\n\
qhull output completed.  Verifying that %d points are\n\
below %2.2g of the nearest %sfacet.\n",
               qh_setsize(qh, facets), maxoutside, (qh->ONLYgood ?  "good " : ""));

  // 遍历每个点及其分配的面
  FOREACHfacet_i_(qh, facets) {  /* for each point with facet assignment */
    // 如果面未分配则标记为未分配状态
    if (facet)
      unassigned= False;
    else {
      unassigned= True;
      facet= qh->facet_list;
    }

    // 获取当前点的坐标
    point= qh_point(qh, facet_i);

    // 如果当前点是已知的好的点则跳过
    if (point == qh->GOODpointp)
      continue;

    // 计算点到面的距离
    qh_distplane(qh, point, facet, &dist);
    numpart++;

    // 找到最佳视平面
    bestfacet= qh_findbesthorizon(qh, !qh_IScheckmax, point, facet, qh_NOupper, &dist, &numpart);
    /* 统计报告后发生 */

    // 更新最大距离值
    maximize_(maxdist, dist);

    // 如果距离超出最大外部距离
    if (dist > maxoutside) {
      // 如果只验证好的面并且最佳面不是好的面
      if (qh->ONLYgood && !bestfacet->good
      && !((bestfacet= qh_findgooddist(qh, point, bestfacet, &dist, &facetlist))
      && dist > maxoutside))
        // 增加不好的面的计数
        notgood++;
      else {
        // 发生错误，记录错误信息并标记错误
        waserror= True;
        qh_fprintf(qh, qh->ferr, 6109, "qhull precision error (qh_check_bestdist): point p%d is outside facet f%d, distance= %6.8g maxoutside= %6.8g\n",
                facet_i, bestfacet->id, dist, maxoutside);
        if (errfacet1 != bestfacet) {
          errfacet2= errfacet1;
          errfacet1= bestfacet;
        }
      }
    } else if (unassigned && dist < -qh->MAXcoplanar) {
      // 如果面未分配并且距离小于负的最大共面距离
      notverified++;
    }
  }

  // 释放临时分配的内存
  qh_settempfree(qh, &facets);

  // 如果有未验证的点且不是德劳内和快速帮助模式，打印精度信息
  if (notverified && !qh->DELAUNAY && !qh_QUICKhelp && qh->PRINTprecision)
    qh_fprintf(qh, qh->ferr, 8092, "\n%d points were well inside the hull.  If the hull contains\n\
a lens-shaped component, these points were not verified.  Use\n\
options 'Qci Tv' to verify all points.\n", notverified);

  // 如果最大距离超出外部错误限制
  if (maxdist > qh->outside_err) {
    qh_fprintf(qh, qh->ferr, 6110, "qhull precision error (qh_check_bestdist): a coplanar point is %6.2g from convex hull.  The maximum value is qh.outside_err (%6.2g)\n",
              maxdist, qh->outside_err);
    // 退出程序并报告错误
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
  } else if (waserror && qh->outside_err > REALmax/2) {
    # 如果 qh_errexit2 函数调用返回，表示发生错误，根据错误的精度和两个错误方面参数输出错误信息并退出程序
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
    
    # 否则，如果发生错误，错误信息被记录到 qh.ferr 中，但不会影响输出结果
    trace0((qh, qh->ferr, 20, "qh_check_bestdist: max distance outside %2.2g\n", maxdist));
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_maxout">-</a>

  qh_check_maxout(qh)
    updates qh.max_outside by checking all points against bestfacet
    if qh.ONLYgood, ignores !good facets

  returns:
    updates facet->maxoutside via qh_findbesthorizon()
    sets qh.maxoutdone
    if printing qh.min_vertex (qh_outerinner),
      it is updated to the current vertices
    removes inside/coplanar points from coplanarset as needed

  notes:
    defines coplanar as qh.min_vertex instead of qh.MAXcoplanar
    may not need to check near-inside points because of qh.MAXcoplanar
      and qh.KEEPnearinside (before it was -qh.DISTround)

  see also:
    qh_check_bestdist()

  design:
    if qh.min_vertex is needed
      for all neighbors of all vertices
        test distance from vertex to neighbor
    determine facet for each point (if any)
    for each point with an assigned facet
      find the best facet for the point and check all coplanar facets
        (updates outer planes)
    remove near-inside points from coplanar sets
*/
void qh_check_maxout(qhT *qh) {
  facetT *facet, *bestfacet, *neighbor, **neighborp, *facetlist, *maxbestfacet= NULL, *minfacet, *maxfacet, *maxpointfacet;
  realT dist, maxoutside, mindist, nearest;
  realT maxoutside_base, minvertex_base;
  pointT *point, *maxpoint= NULL;
  int numpart= 0, facet_i, facet_n, notgood= 0;
  setT *facets, *vertices;
  vertexT *vertex, *minvertex;

  trace1((qh, qh->ferr, 1022, "qh_check_maxout: check and update qh.min_vertex %2.2g and qh.max_outside %2.2g\n", qh->min_vertex, qh->max_outside));
  // 计算最小顶点的基础值，考虑合并距离和距离舍入误差
  minvertex_base= fmin_(qh->min_vertex, -(qh->ONEmerge+qh->DISTround));
  // 初始化最大外接球半径和最小距离为0
  maxoutside= mindist= 0.0;
  // 设置最小顶点为顶点列表的第一个顶点
  minvertex= qh->vertex_list;
  // 初始化最大面和最小面为第一个面
  maxfacet= minfacet= maxpointfacet= qh->facet_list;
  
  // 如果需要顶点邻居信息，并且满足某些条件输出或跟踪
  if (qh->VERTEXneighbors
  && (qh->PRINTsummary || qh->KEEPinside || qh->KEEPcoplanar
        || qh->TRACElevel || qh->PRINTstatistics || qh->VERIFYoutput || qh->CHECKfrequently
        || qh->PRINTout[0] == qh_PRINTsummary || qh->PRINTout[0] == qh_PRINTnone)) {
    // 跟踪消息：确定实际的最小顶点
    trace1((qh, qh->ferr, 1023, "qh_check_maxout: determine actual minvertex\n"));
    // 获取所有顶点的邻居信息
    vertices= qh_pointvertex(qh /* qh.facet_list */);
    FORALLvertices {
      /* 对于每个顶点进行循环 */

      FOREACHneighbor_(vertex) {
        /* 对于顶点的每个相邻顶点进行循环 */

        zinc_(Zdistvertex);  /* 增加 Zdistvertex 计数，主循环也会计算距离 */
        
        /* 计算顶点到相邻顶点的距离 */
        qh_distplane(qh, vertex->point, neighbor, &dist);
        
        /* 如果距离小于最小距离 */
        if (dist < mindist) {
          /* 检查是否超出了最大外围限制，如果是，并且允许精确打印或不允许宽松条件 */
          if (qh->min_vertex/minvertex_base > qh_WIDEmaxoutside && (qh->PRINTprecision || !qh->ALLOWwide)) {
            /* 在后处理中，应该在 qh_mergefacet 中捕获 */
            nearest= qh_vertex_bestdist(qh, neighbor->vertices);
            /* 输出 Qhull 精度警告：在后处理（qh_check_maxout）中，p%d(v%d) 比 f%d 的最近顶点 %2.2g 还要小 */
            qh_fprintf(qh, qh->ferr, 7083, "Qhull precision warning: in post-processing (qh_check_maxout) p%d(v%d) is %2.2g below f%d nearest vertices %2.2g\n",
              qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, nearest);
          }
          /* 更新最小距离、最小顶点和最小面 */
          mindist= dist;
          minvertex= vertex;
          minfacet= neighbor;
        }
      }
#ifndef qh_NOtrace
        // 如果未定义 qh_NOtrace，则执行以下代码块
        if (-dist > qh->TRACEdist || dist > qh->TRACEdist
        || neighbor == qh->tracefacet || vertex == qh->tracevertex) {
          // 如果距离 dist 的负值大于 qh.TRACEdist 或者 dist 大于 qh.TRACEdist
          // 或者邻近面是 tracefacet 或者顶点是 tracevertex
          nearest= qh_vertex_bestdist(qh, neighbor->vertices);
          // 计算 neighbor 的顶点中与 qh 中最近的顶点的距离
          qh_fprintf(qh, qh->ferr, 8093, "qh_check_maxout: p%d(v%d) is %.2g from f%d nearest vertices %2.2g\n",
                    qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, nearest);
          // 输出警告信息，显示距离和相关信息
        }
#endif
      }
    }
    if (qh->MERGING) {
      wmin_(Wminvertex, qh->min_vertex);
      // 如果 MERGING 是真值，更新 Wminvertex 和 qh.min_vertex 的值
    }
    qh->min_vertex= mindist;
    // 设置 qh.min_vertex 为 mindist 的值
    qh_settempfree(qh, &vertices);
    // 释放 vertices 中的临时内存空间
  }
  trace1((qh, qh->ferr, 1055, "qh_check_maxout: determine actual maxoutside\n"));
  // 输出追踪消息，标识计算实际的 maxoutside
  maxoutside_base= fmax_(qh->max_outside, qh->ONEmerge+qh->DISTround);
  // 计算 maxoutside_base，它是 qh.max_outside 和 qh.ONEmerge+qh.DISTround 的最大值
  /* maxoutside_base is same as qh.MAXoutside without qh.MINoutside (qh_detmaxoutside) */
  // maxoutside_base 等同于 qh.MAXoutside，但不包括 qh.MINoutside
  facets= qh_pointfacet(qh /* qh.facet_list */);
  // 获取 qh.facet_list 中的点相关联的面列表
  FOREACHfacet_i_(qh, facets) {     // 对于每个具有面分配的点
    if (facet) {
      // 如果 facet 存在
      point= qh_point(qh, facet_i);
      // 获取 facet_i 对应的点
      if (point == qh->GOODpointp)
        continue;
      // 如果点是 GOODpointp，则跳过当前循环
      zzinc_(Ztotcheck);
      // 增加 Ztotcheck 的计数
      qh_distplane(qh, point, facet, &dist);
      // 计算点到面的距离，并将结果存储在 dist 中
      numpart++;
      // 增加 numpart 的计数
      bestfacet= qh_findbesthorizon(qh, qh_IScheckmax, point, facet, !qh_NOupper, &dist, &numpart);
      // 查找最佳的水平面（horizon），并返回最佳面和距离
      if (bestfacet && dist >= maxoutside) { 
        // 如果找到最佳面并且距离大于等于 maxoutside
        if (qh->ONLYgood && !bestfacet->good
        && !((bestfacet= qh_findgooddist(qh, point, bestfacet, &dist, &facetlist))
        && dist > maxoutside)) {       
          // 如果 ONLYgood 为真且 bestfacet 不是 good
          // 或者找到一个比 maxoutside 更好的距离
          notgood++;
          // 增加 notgood 的计数
        }else if (dist/maxoutside_base > qh_WIDEmaxoutside && (qh->PRINTprecision || !qh->ALLOWwide)) {
          // 否则，如果距离与 maxoutside_base 的比率大于 qh_WIDEmaxoutside
          // 并且 PRINTprecision 为真或者 ALLOWwide 为假
          nearest= qh_vertex_bestdist(qh, bestfacet->vertices);
          // 计算 bestfacet 的顶点中与 qh 中最近的顶点的距离
          if (nearest < fmax_(qh->ONEmerge, qh->max_outside) * qh_RATIOcoplanaroutside * 2) {
            // 如果最近距离小于特定阈值
            qh_fprintf(qh, qh->ferr, 7087, "Qhull precision warning: in post-processing (qh_check_maxout) p%d for f%d is %2.2g above twisted facet f%d nearest vertices %2.2g\n",
              qh_pointid(qh, point), facet->id, dist, bestfacet->id, nearest);
          }else {
            // 否则，输出警告信息，显示距离和相关信息
            qh_fprintf(qh, qh->ferr, 7088, "Qhull precision warning: in post-processing (qh_check_maxout) p%d for f%d is %2.2g above hidden facet f%d nearest vertices %2.2g\n",
              qh_pointid(qh, point), facet->id, dist, bestfacet->id, nearest);
          }
          maxbestfacet= bestfacet;
          // 设置 maxbestfacet 为 bestfacet
        }
        maxoutside= dist;
        // 设置 maxoutside 为 dist
        maxfacet= bestfacet;
        // 设置 maxfacet 为 bestfacet
        maxpoint= point;
        // 设置 maxpoint 为 point
        maxpointfacet= facet;
        // 设置 maxpointfacet 为 facet
      }
      if (dist > qh->TRACEdist || (bestfacet && bestfacet == qh->tracefacet))
        // 如果距离大于 qh.TRACEdist 或者 bestfacet 等于 qh.tracefacet
        qh_fprintf(qh, qh->ferr, 8094, "qh_check_maxout: p%d is %.2g above f%d\n",
              qh_pointid(qh, point), dist, (bestfacet ? bestfacet->id : UINT_MAX));
        // 输出警告信息，显示距离和相关信息
    }
  }
  zzadd_(Zcheckpart, numpart);
  // 将 numpart 的值添加到 Zcheckpart 中
  qh_settempfree(qh, &facets);
  // 释放 facets 中的临时内存空间
  wval_(Wmaxout)= maxoutside - qh->max_outside;
  // 设置 Wmaxout 的值为 maxoutside 减去 qh.max_outside
  wmax_(Wmaxoutside, qh->max_outside);
  // 设置 Wmaxoutside 的值为 qh.max_outside
  if (!qh->APPROXhull && maxoutside > qh->DISTround) { /* initial value for f.maxoutside */
  // 如果 qh.APPROXhull 是假值并且 maxoutside 大于 qh.DISTround
    FORALLfacets {
      # 遍历所有的凸包面 facet
      if (maxoutside < facet->maxoutside) {
        # 检查当前 facet 的最大外部距离是否大于已记录的 maxoutside
        if (!qh->KEEPcoplanar) {
          # 如果不保留共面点，更新 maxoutside
          maxoutside= facet->maxoutside;
        }else if (maxoutside + qh->DISTround < facet->maxoutside) { /* maxoutside is computed distance, e.g., rbox 100 s D3 t1547136913 | qhull R1e-3 Tcv Qc */
          # 否则，如果 maxoutside 加上 DISTround 小于 facet->maxoutside，发出精度警告
          qh_fprintf(qh, qh->ferr, 7082, "Qhull precision warning (qh_check_maxout): f%d.maxoutside (%4.4g) is greater than computed qh.max_outside (%2.2g) + qh.DISTround (%2.2g).  It should be less than or equal\n",
            facet->id, facet->maxoutside, maxoutside, qh->DISTround); 
        }
      }
    }
  }
  # 更新全局变量 qh->max_outside
  qh->max_outside= maxoutside; 
  # 调用函数处理近共面点
  qh_nearcoplanar(qh /* qh.facet_list */);
  # 设置标志表示 max_outside 已经更新完成
  qh->maxoutdone= True;
  # 输出跟踪信息，描述最大外部距离的相关情况
  trace1((qh, qh->ferr, 1024, "qh_check_maxout:  p%d(v%d) is qh.min_vertex %2.2g below facet f%d.  Point p%d for f%d is qh.max_outside %2.2g above f%d.  %d points are outside of not-good facets\n", 
    qh_pointid(qh, minvertex->point), minvertex->id, qh->min_vertex, minfacet->id, qh_pointid(qh, maxpoint), maxpointfacet->id, qh->max_outside, maxfacet->id, notgood));
  # 如果不允许 wide facets，进行相应检查
  if(!qh->ALLOWwide) {
    # 如果 maxoutside 与基础值的比值超过设定的阈值 qh_WIDEmaxoutside
    if (maxoutside/maxoutside_base > qh_WIDEmaxoutside) {
      # 输出精度错误信息，指出 qh.max_outside 在后处理过程中的大幅增加
      qh_fprintf(qh, qh->ferr, 6297, "Qhull precision error (qh_check_maxout): large increase in qh.max_outside during post-processing dist %2.2g (%.1fx).  See warning QH0032/QH0033.  Allow with 'Q12' (allow-wide) and 'Pp'\n",
        maxoutside, maxoutside/maxoutside_base);
      # 通过 qh_errexit 函数退出程序，指定错误类型 qh_ERRwide
      qh_errexit(qh, qh_ERRwide, maxbestfacet, NULL);
    }else if (!qh->APPROXhull && maxoutside_base > (qh->ONEmerge * qh_WIDEmaxoutside2)) {
      # 如果不是近似凸包，并且 maxoutside_base 超过阈值 qh_ONEmerge * qh_WIDEmaxoutside2
      if (maxoutside > (qh->ONEmerge * qh_WIDEmaxoutside2)) {  /* wide facets may have been deleted */
        # 输出精度错误信息，指出合并操作导致了宽面的产生
        qh_fprintf(qh, qh->ferr, 6298, "Qhull precision error (qh_check_maxout): a facet merge, vertex merge, vertex, or coplanar point produced a wide facet %2.2g (%.1fx). Trace with option 'TWn' to identify the merge.   Allow with 'Q12' (allow-wide)\n",
          maxoutside_base, maxoutside_base/(qh->ONEmerge + qh->DISTround));
        # 通过 qh_errexit 函数退出程序，指定错误类型 qh_ERRwide
        qh_errexit(qh, qh_ERRwide, maxbestfacet, NULL);
      }
    }else if (qh->min_vertex/minvertex_base > qh_WIDEmaxoutside) {
      # 如果 min_vertex 与基础值的比值超过设定的阈值 qh_WIDEmaxoutside
      qh_fprintf(qh, qh->ferr, 6354, "Qhull precision error (qh_check_maxout): large increase in qh.min_vertex during post-processing dist %2.2g (%.1fx).  See warning QH7083.  Allow with 'Q12' (allow-wide) and 'Pp'\n",
        qh->min_vertex, qh->min_vertex/minvertex_base);
      # 通过 qh_errexit 函数退出程序，指定错误类型 qh_ERRwide
      qh_errexit(qh, qh_ERRwide, minfacet, NULL);
    }else if (minvertex_base < -(qh->ONEmerge * qh_WIDEmaxoutside2)) {
      // 检查最小顶点基准是否小于负宽度最大外部乘积的阈值，可能意味着宽度面可能已被删除
      if (qh->min_vertex < -(qh->ONEmerge * qh_WIDEmaxoutside2)) {  /* wide facets may have been deleted */
        // 如果最小顶点小于负宽度最大外部乘积的阈值，表明宽度面可能已被删除
        qh_fprintf(qh, qh->ferr, 6380, "Qhull precision error (qh_check_maxout): a facet or vertex merge produced a wide facet: v%d below f%d distance %2.2g (%.1fx). Trace with option 'TWn' to identify the merge.  Allow with 'Q12' (allow-wide)\n",
          minvertex->id, minfacet->id, mindist, -qh->min_vertex/(qh->ONEmerge + qh->DISTround));
        // 打印精度错误消息，指出面或顶点合并导致产生一个宽度面
        qh_errexit(qh, qh_ERRwide, minfacet, NULL);
        // 报错并退出程序，指示发现宽度面的错误
      }
    }
  }
#else /* qh_NOmerge */
void qh_check_maxout(qhT *qh) {
  QHULL_UNUSED(qh)
}
#endif


#else /* qh_NOmerge */
// 如果定义了qh_NOmerge，定义一个名为qh_check_maxout的空函数，接受一个qhT类型的指针参数qh，并标记为未使用
void qh_check_maxout(qhT *qh) {
  QHULL_UNUSED(qh)
}
#endif



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_output">-</a>

  qh_check_output(qh)
    performs the checks at the end of qhull algorithm
    Maybe called after Voronoi output.  If so, it recomputes centrums since they are Voronoi centers instead.
*/
void qh_check_output(qhT *qh) {
  int i;

  if (qh->STOPcone)
    return;
  if (qh->VERIFYoutput || qh->IStracing || qh->CHECKfrequently) {
    qh_checkpolygon(qh, qh->facet_list);
    qh_checkflipped_all(qh, qh->facet_list);
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }else if (!qh->MERGING && qh_newstats(qh, qh->qhstat.precision, &i)) {
    qh_checkflipped_all(qh, qh->facet_list);
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }
} /* check_output */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_output">-</a>

  qh_check_output(qh)
    在qhull算法结束时执行检查
    可能在Voronoi输出后调用。如果是这样，则重新计算中心，因为它们是Voronoi中心而不是凸包中心。
*/
void qh_check_output(qhT *qh) {
  int i;

  // 如果STOPcone标志被设置，则直接返回
  if (qh->STOPcone)
    return;
  
  // 如果需要验证输出，或者正在追踪，或者需要频繁检查
  if (qh->VERIFYoutput || qh->IStracing || qh->CHECKfrequently) {
    // 对凸包中的每个面进行多边形检查
    qh_checkpolygon(qh, qh->facet_list);
    // 检查所有翻转的面
    qh_checkflipped_all(qh, qh->facet_list);
    // 检查凸包是否保持凸性，使用qh_ALGORITHMfault作为算法类型
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  } else if (!qh->MERGING && qh_newstats(qh, qh->qhstat.precision, &i)) {
    // 如果不处于MERGING状态，并且成功获取了新的统计数据
    qh_checkflipped_all(qh, qh->facet_list);
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }
} /* check_output */



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_point">-</a>

  qh_check_point(qh, point, facet, maxoutside, maxdist, errfacet1, errfacet2, errcount )
    check that point is less than maxoutside from facet

  notes:
    only called from qh_checkpoints
    reports up to qh_MAXcheckpoint-1 errors per facet
*/
void qh_check_point(qhT *qh, pointT *point, facetT *facet, realT *maxoutside, realT *maxdist, facetT **errfacet1, facetT **errfacet2, int *errcount) {
  realT dist, nearest;

  /* occurs after statistics reported */
  qh_distplane(qh, point, facet, &dist);
  maximize_(*maxdist, dist);
  if (dist > *maxoutside) {
    (*errcount)++;
    if (*errfacet1 != facet) {
      *errfacet2= *errfacet1;
      *errfacet1= facet;
    }
    if (*errcount < qh_MAXcheckpoint) {
      nearest= qh_vertex_bestdist(qh, facet->vertices);
      qh_fprintf(qh, qh->ferr, 6111, "qhull precision error: point p%d is outside facet f%d, distance= %6.8g maxoutside= %6.8g nearest vertices %2.2g\n",
                qh_pointid(qh, point), facet->id, dist, *maxoutside, nearest);
    }
  }
} /* qh_check_point */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_point">-</a>

  qh_check_point(qh, point, facet, maxoutside, maxdist, errfacet1, errfacet2, errcount )
    检查点是否比facet的maxoutside距离更近

  notes:
    只从qh_checkpoints调用
    每个facet最多报告qh_MAXcheckpoint-1个错误

*/
void qh_check_point(qhT *qh, pointT *point, facetT *facet, realT *maxoutside, realT *maxdist, facetT **errfacet1, facetT **errfacet2, int *errcount) {
  realT dist, nearest;

  /* 在报告统计数据后发生 */
  qh_distplane(qh, point, facet, &dist);
  maximize_(*maxdist, dist);
  if (dist > *maxoutside) {
    (*errcount)++;
    if (*errfacet1 != facet) {
      *errfacet2= *errfacet1;
      *errfacet1= facet;
    }
    if (*errcount < qh_MAXcheckpoint) {
      nearest= qh_vertex_bestdist(qh, facet->vertices);
      qh_fprintf(qh, qh->ferr, 6111, "qhull precision error: point p%d is outside facet f%d, distance= %6.8g maxoutside= %6.8g nearest vertices %2.2g\n",
                qh_pointid(qh, point), facet->id, dist, *maxoutside, nearest);
    }
  }
} /* qh_check_point */



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_points">-</a>

  qh_check_points(qh)
    checks that all points are inside all facets

  notes:
    if many points and qh_check_maxout not called (i.e., !qh.MERGING),
       calls qh_findbesthorizon via qh_check_bestdist (seldom done).
    ignores flipped facets
    maxoutside includes 2 qh.DISTrounds
      one qh.DISTround for the computed distances in qh_check_points
    qh_printafacet and qh_printsummary needs only one qh.DISTround
    the computation for qh.VERIFYdirect does not account for qh.other_points

  design:
    if many points
      use qh_check_bestdist()
    else
      for all facets
        for all points
          check that point is inside facet
*/


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_points">-</a>

  qh_check_points(qh)
    检查所有点是否都在所有的面内

  notes:
    如果点很多且未调用qh_check_maxout（即!qh.MERGING），则通过qh_check_bestdist调用qh_findbesthorizon（很少这样做）。
    忽略翻转的面
    maxoutside包括2个qh.DISTrounds
      qh_check_points中计算距离的一个qh.DISTround
    qh_printafacet和qh_printsummary只需要一个qh.DISTround
    qh.VERIFYdirect的计算不考虑qh.other_points

  design:
    如果点很多
      使用qh_check_bestdist()
    否则
      对于所有面
        对于所有点
          检查点是否在面内
*/
void qh_check_points(qhT *qh) {
  // 定义变量：facetT 类型的指针变量 facet，用于迭代访问凸包的每一个面元
  facetT *facet, *errfacet1 = NULL, *errfacet2 = NULL;
  // 定义变量：total（实数类型）、maxoutside（实数类型，初始值设为负无穷大）、maxdist（实数类型，初始值设为负无穷大）
  realT total, maxoutside, maxdist = -REALmax;
  // 定义变量：pointT 类型的指针变量 point，用于迭代访问凸包的每一个点
  pointT *point, **pointp, *pointtemp;
  // 定义变量：errcount（整数类型），用于计算错误数量
  int errcount;
  // 定义变量：testouter（布尔类型），用于指示是否进行外部平面检查
  boolT testouter;

  // 计算最大外部距离（maxoutside），考虑到额外的 DISTround
  maxoutside = qh_maxouter(qh);
  maxoutside += qh->DISTround;
  /* one more qh.DISTround for check computation */
  // 输出追踪消息，指示进行点的检查，确保它们距离所有面元的平面都小于 maxoutside
  trace1((qh, qh->ferr, 1025, "qh_check_points: check all points below %2.2g of all facet planes\n",
          maxoutside));

  // 根据凸包的状态选择计算总数
  if (qh->num_good)   /* miss counts other_points and !good facets */
     total = (float) qh->num_good * (float) qh->num_points;
  else
     total = (float) qh->num_facets * (float) qh->num_points;

  // 如果总数大于等于验证的直接阈值，并且还未进行最大外部距离的检查，则执行以下操作
  if (total >= qh_VERIFYdirect && !qh->maxoutdone) {
    // 如果不是快速帮助模式且跳过外部平面检查并且合并模式开启，则发出警告信息
    if (!qh_QUICKhelp && qh->SKIPcheckmax && qh->MERGING)
      qh_fprintf(qh, qh->ferr, 7075, "qhull input warning: merging without checking outer planes('Q5' or 'Po').  Verify may report that a point is outside of a facet.\n");
    // 检查最佳距离
    qh_check_bestdist(qh);
  } else {
    // 如果设置了最大外部距离并且已完成，则设置 testouter 为真
    if (qh_MAXoutside && qh->maxoutdone)
      testouter = True;
    else
      testouter = False;

    // 如果不是快速帮助模式，则根据凸包的状态发出相应的警告信息
    if (!qh_QUICKhelp) {
      // 如果是精确合并模式，则发出精确合并警告信息
      if (qh->MERGEexact)
        qh_fprintf(qh, qh->ferr, 7076, "qhull input warning: exact merge ('Qx').  Verify may report that a point is outside of a facet.  See qh-optq.htm#Qx\n");
      // 如果跳过外部平面检查或者不处理靠近内部点，则发出相应的警告信息
      else if (qh->SKIPcheckmax || qh->NOnearinside)
        qh_fprintf(qh, qh->ferr, 7077, "qhull input warning: no outer plane check ('Q5') or no processing of near-inside points ('Q8').  Verify may report that a point is outside of a facet.\n");
    }

    // 如果设置了输出精度信息，则根据 testouter 输出相应的信息
    if (qh->PRINTprecision) {
      if (testouter)
        qh_fprintf(qh, qh->ferr, 8098, "\n\
Output completed.  Verifying that all points are below outer planes of\n\
all %sfacets.  Will make %2.0f distance computations.\n",
                  (qh->ONLYgood ?  "good " : ""), total);
      else
        qh_fprintf(qh, qh->ferr, 8099, "\n\
Output completed.  Verifying that all points are below %2.2g of\n\
all %sfacets.  Will make %2.0f distance computations.\n",
                  maxoutside, (qh->ONLYgood ?  "good " : ""), total);
    }

    // 遍历所有面元进行检查
    FORALLfacets {
      // 如果面元不是好的或者仅考虑好的面元，则跳过当前面元
      if (!facet->good && qh->ONLYgood)
        continue;
      // 如果面元已经翻转，则跳过当前面元
      if (facet->flipped)
        continue;
      // 如果面元的法向量不存在，则发出警告信息，并记录第一个错误的面元
      if (!facet->normal) {
        qh_fprintf(qh, qh->ferr, 7061, "qhull warning (qh_check_points): missing normal for facet f%d\n", facet->id);
        if (!errfacet1)
          errfacet1 = facet;
        continue;
      }

      // 如果设置了 testouter，则重新计算 maxoutside
      if (testouter) {
#if qh_MAXoutside
        maxoutside = facet->maxoutside + 2 * qh->DISTround;
        /* one DISTround to actual point and another to computed point */
/*-------------------------------------
  #endif
  -------------------------------------*/
      }
      // 重置错误计数器
      errcount= 0;
      // 遍历所有点
      FORALLpoints {
        // 如果点不是好的点（GOODpointp），则检查点
        if (point != qh->GOODpointp)
          qh_check_point(qh, point, facet, &maxoutside, &maxdist, &errfacet1, &errfacet2, &errcount);
      }
      // 遍历其他点列表
      FOREACHpoint_(qh->other_points) {
        // 如果点不是好的点（GOODpointp），则检查点
        if (point != qh->GOODpointp)
          qh_check_point(qh, point, facet, &maxoutside, &maxdist, &errfacet1, &errfacet2, &errcount);
      }
      // 如果错误计数超过最大允许的检查点数
      if (errcount >= qh_MAXcheckpoint) {
        // 输出精度错误信息到qh->ferr，指明有多少个额外的点在facet f%d之外，最大距离为maxdist
        qh_fprintf(qh, qh->ferr, 6422, "qhull precision error (qh_check_points): %d additional points outside facet f%d, maxdist= %6.8g\n",
             errcount-qh_MAXcheckpoint+1, facet->id, maxdist);
      }
    }
    // 如果最大距离大于qh->outside_err
    if (maxdist > qh->outside_err) {
      // 输出精度错误信息到qh->ferr，指明一个共面点距离凸壳的最大值，以及qh->outside_err的最大值
      qh_fprintf(qh, qh->ferr, 6112, "qhull precision error (qh_check_points): a coplanar point is %6.2g from convex hull.  The maximum value(qh.outside_err) is %6.2g\n",
                maxdist, qh->outside_err );
      // 终止程序，输出错误信息
      qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2 );
    }else if (errfacet1 && qh->outside_err > REALmax/2)
        // 如果存在errfacet1，并且qh->outside_err超过REALmax的一半，则终止程序，输出错误信息
        qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2 );
    /* else if errfacet1, the error was logged to qh.ferr but does not effect the output */
    // 否则，如果存在errfacet1，错误已经记录在qh->ferr中，但不会影响输出
    trace0((qh, qh->ferr, 21, "qh_check_points: max distance outside %2.2g\n", maxdist));
  }
} /* check_points */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkconvex">-</a>

  qh_checkconvex(qh, facetlist, fault )
    检查facetlist中的每个ridge是否是凸的
    如果fault = qh_DATAfault，则报告来自于带有qh.ZEROcentrum的qh_initialhull的错误
    否则，fault = qh_ALGORITHMfault

  返回：
    计数Zconcaveridges和Zcoplanarridges
    如果!qh.FORCEoutput（'Fo'）且concaveridge存在，或者合并共面ridge，则报告错误
    如果由qh_setvoronoi_all/qh_ASvoronoi设置，则覆盖Voronoi中心

  注意：
    被qh_initial_hull、qh_check_output、qh_all_merges（'Tc'）、qh_build_withrestart（'QJ'）调用
    不测试f.tricoplanar facets（qh_triangulate）
    必须不强于qh_test_appendmerge
    如果不合并，
      对相邻的simplicial facets的顶点进行测试 < -qh.DISTround
    否则如果ZEROcentrum且simplicial facet，
      对相邻的simplicial facets的顶点进行测试 < 0.0
      对相邻的非simplicial facets的中心进行测试 < 0.0
    否则如果ZEROcentrum 
      对相邻facets的中心进行测试 < 0.0
    否则 
      对相邻facets的中心进行测试 < -qh.DISTround（'En' 'Rn'）
    不测试与-qh.centrum_radius相对的情况，因为重复计算可能存在不同的舍入误差（例如 'Rn'）

  设计：
    对所有facets进行循环
      报告翻转的facets
      如果ZEROcentrum且有simplicial neighbors
        对邻居进行顶点测试
      否则
        对邻居的中心进行测试
*/
  # 检查凸壳性质的函数，用于检查给定的凸包数据结构中的各个面是否满足特定的凸性质要求
  void qh_checkconvex(qhT *qh, facetT *facetlist, int fault) {
    facetT *facet, *neighbor, **neighborp, *errfacet1=NULL, *errfacet2=NULL;  // 定义凸包面、邻居面及错误面的变量
    vertexT *vertex;  // 定义顶点变量
    realT dist;  // 定义距离变量
    pointT *centrum;  // 定义凸包中心点变量
    boolT waserror= False, centrum_warning= False, tempcentrum= False, first_nonsimplicial= False, tested_simplicial, allsimplicial;  // 定义布尔型变量，用于标记各种错误状态和几何特性
    int neighbor_i, neighbor_n;  // 定义邻居面索引和数量变量

    if (qh->ZEROcentrum) {
      // 如果启用了ZEROcentrum选项，记录跟踪信息并标记首次非简单化
      trace1((qh, qh->ferr, 1064, "qh_checkconvex: check that facets are not-flipped and for qh.ZEROcentrum that simplicial vertices are below their neighbor (dist<0.0)\n"));
      first_nonsimplicial= True;
    }else if (!qh->MERGING) {
      // 如果未启用MERGING选项，记录跟踪信息并标记首次非简单化
      trace1((qh, qh->ferr, 1026, "qh_checkconvex: check that facets are not-flipped and that simplicial vertices are convex by qh.DISTround ('En', 'Rn')\n"));
      first_nonsimplicial= True;
    }else
      // 否则，记录跟踪信息并标记首次非简单化
      trace1((qh, qh->ferr, 1062, "qh_checkconvex: check that facets are not-flipped and that their centrums are convex by qh.DISTround ('En', 'Rn') \n"));

    if (!qh->RERUN) {
      // 如果非重新运行模式，重置凸包数据中的边缘和共面边数统计
      zzval_(Zconcaveridges)= 0;
      zzval_(Zcoplanarridges)= 0;
    }

    // 遍历所有凸包面
    FORALLfacet_(facetlist) {
      if (facet->flipped) {
        // 如果当前面被标记为flipped，触发重启函数并记录错误信息
        qh_joggle_restart(qh, "flipped facet"); /* also tested by qh_checkflipped */
        qh_fprintf(qh, qh->ferr, 6113, "qhull precision error: f%d is flipped (interior point is outside)\n",
                 facet->id);
        errfacet1= facet;
        waserror= True;
        continue;
      }
      if (facet->tricoplanar)
        // 如果当前面标记为tricoplanar，则跳过
        continue;
      if (qh->MERGING && (!qh->ZEROcentrum || !facet->simplicial)) {
        // 如果启用MERGING选项且非ZEROcentrum或非simplicial面，则标记非全部简单化
        allsimplicial= False;
        tested_simplicial= False;
        // 下面可能会有更多的条件和处理逻辑，但在注释范围内只记录当前的逻辑判断
    }else {
      // 设置所有的 simplicial 变量为 True，并标记已测试 simplicial 为 True
      allsimplicial= True;
      tested_simplicial= True;
      // 对 facet 的每一个邻居进行遍历
      FOREACHneighbor_i_(qh, facet) {
        // 如果邻居是 tricoplanar，则跳过
        if (neighbor->tricoplanar)
          continue;
        // 如果邻居不是 simplicial，则将 allsimplicial 设置为 False，并继续下一个邻居
        if (!neighbor->simplicial) {
          allsimplicial= False;
          continue;
        }
        // 获取当前邻居的顶点
        vertex= SETelemt_(facet->vertices, neighbor_i, vertexT);
        // 计算当前顶点到邻居的平面距离
        qh_distplane(qh, vertex->point, neighbor, &dist);
        // 如果距离大于等于负的 DISTround 值
        if (dist >= -qh->DISTround) {
          // 如果是数据错误，进行错误处理
          if (fault == qh_DATAfault) {
            // 重新启动 Qhull，并打印错误消息
            qh_joggle_restart(qh, "non-convex initial simplex");
            if (dist > qh->DISTround)
              qh_fprintf(qh, qh->ferr, 6114, "qhull precision error: initial simplex is not convex, since p%d(v%d) is %6.4g above opposite f%d\n", 
                  qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id);
            else
              qh_fprintf(qh, qh->ferr, 6379, "qhull precision error: initial simplex is not convex, since p%d(v%d) is within roundoff of opposite facet f%d (dist %6.4g)\n",
                  qh_pointid(qh, vertex->point), vertex->id, neighbor->id, dist);
            // 设置 Qhull 错误状态并退出
            qh_errexit(qh, qh_ERRsingular, neighbor, NULL);
          }
          // 如果距离大于 DISTround 值
          if (dist > qh->DISTround) {
            // 增加 concave 边缘计数，并重新启动 Qhull
            zzinc_(Zconcaveridges);
            qh_joggle_restart(qh, "concave ridge");
            // 打印错误消息
            qh_fprintf(qh, qh->ferr, 6115, "qhull precision error: f%d is concave to f%d, since p%d(v%d) is %6.4g above f%d\n",
              facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id);
            // 记录错误的两个 facet
            errfacet1= facet;
            errfacet2= neighbor;
            waserror= True;
          }else if (qh->ZEROcentrum) {
            // 如果 ZEROcentrum 开启且距离大于 0.0
            if (dist > 0.0) {     /* qh_checkzero checked convex (dist < (- 2*qh->DISTround)), computation may differ e.g. 'Rn' */
              // 增加 coplanar 边缘计数，并重新启动 Qhull
              zzinc_(Zcoplanarridges);
              qh_joggle_restart(qh, "coplanar ridge");
              // 打印错误消息
              qh_fprintf(qh, qh->ferr, 6116, "qhull precision error: f%d is clearly not convex to f%d, since p%d(v%d) is %6.4g above or coplanar with f%d with qh.ZEROcentrum\n",
                facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id);
              // 记录错误的两个 facet
              errfacet1= facet;
              errfacet2= neighbor;
              waserror= True;
            }
          }else {
            // 否则，增加 coplanar 边缘计数，并打印追踪消息
            zzinc_(Zcoplanarridges);
            qh_joggle_restart(qh, "coplanar ridge");
            trace0((qh, qh->ferr, 22, "qhull precision error: f%d is coplanar to f%d, since p%d(v%d) is within %6.4g of f%d, during p%d\n",
              facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, qh->furthest_id));
          }
        }
      }
    }
    // 如果不是所有的面都是单纯形（非简单形），则执行以下代码块
    if (!allsimplicial) {
      // 如果是第一个非简单形面，则输出跟踪信息并设置标记为 False
      if (first_nonsimplicial) {
        trace1((qh, qh->ferr, 1063, "qh_checkconvex: starting with f%d, also check that centrums of non-simplicial ridges are below their neighbors (dist<0.0)\n",
             facet->id));
        first_nonsimplicial= False;
      }
      
      // 根据 qh->CENTERtype 的设置，获取面的中心点
      if (qh->CENTERtype == qh_AScentrum) {
        // 如果面的中心点不存在，则调用 qh_getcentrum 函数获取
        if (!facet->center)
          facet->center= qh_getcentrum(qh, facet);
        centrum= facet->center;
      } else {
        // 如果尚未警告过并且面不是简单形，则输出警告信息
        if (!centrum_warning && !facet->simplicial) {  /* recomputed centrum correct for simplicial facets */
           centrum_warning= True;
           qh_fprintf(qh, qh->ferr, 7062, "qhull warning: recomputing centrums for convexity test.  This may lead to false, precision errors.\n");
        }
        // 获取面的中心点
        centrum= qh_getcentrum(qh, facet);
        tempcentrum= True;
      }
      
      // 遍历面的邻居面
      FOREACHneighbor_(facet) {
        // 如果邻居面是简单形且已经进行过测试，则跳过
        if (neighbor->simplicial && tested_simplicial) /* tested above since f.simplicial */
          continue;
        // 如果邻居面是三共面的，则跳过
        if (neighbor->tricoplanar)
          continue;
        
        // 增加 Zdistconvex 计数器，计算面到邻居面的距离
        zzinc_(Zdistconvex);
        qh_distplane(qh, centrum, neighbor, &dist);
        
        // 如果距离大于 qh->DISTround，则认为面是凹面
        if (dist > qh->DISTround) {
          // 增加 Zconcaveridges 计数器，输出凹面错误信息
          zzinc_(Zconcaveridges);
          qh_joggle_restart(qh, "concave ridge");
          qh_fprintf(qh, qh->ferr, 6117, "qhull precision error: f%d is concave to f%d.  Centrum of f%d is %6.4g above f%d\n",
            facet->id, neighbor->id, facet->id, dist, neighbor->id);
          errfacet1= facet;
          errfacet2= neighbor;
          waserror= True;
        } else if (dist >= 0.0) {   /* if arithmetic always rounds the same,
                                     can test against centrum radius instead */
          // 增加 Zcoplanarridges 计数器，输出共面或凹面错误信息
          zzinc_(Zcoplanarridges);
          qh_joggle_restart(qh, "coplanar ridge");
          qh_fprintf(qh, qh->ferr, 6118, "qhull precision error: f%d is coplanar or concave to f%d.  Centrum of f%d is %6.4g above f%d\n",
            facet->id, neighbor->id, facet->id, dist, neighbor->id);
          errfacet1= facet;
          errfacet2= neighbor;
          waserror= True;
        }
      }
      
      // 如果临时中心点被创建，则释放内存
      if (tempcentrum)
        qh_memfree(qh, centrum, qh->normal_size);
    }
  }
  
  // 如果发生错误且未强制输出，则通过 qh_errexit2 函数退出程序
  if (waserror && !qh->FORCEoutput)
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkfacet">-</a>

  qh_checkfacet(qh, facet, newmerge, waserror )
    checks for consistency errors in facet
    newmerge set if from merge_r.c

  returns:
    sets waserror if any error occurs

  checks:
    vertex ids are inverse sorted
    unless newmerge, at least hull_dim neighbors and vertices (exactly if simplicial)
    if non-simplicial, at least as many ridges as neighbors
    neighbors are not duplicated
    ridges are not duplicated
    in 3-d, ridges=verticies
    (qh.hull_dim-1) ridge vertices
    neighbors are reciprocated
    ridge neighbors are facet neighbors and a ridge for every neighbor
    simplicial neighbors match facetintersect
    vertex intersection matches vertices of common ridges
    vertex neighbors and facet vertices agree
    all ridges have distinct vertex sets

  notes:
    called by qh_tracemerge and qh_checkpolygon
    uses neighbor->seen

  design:
    check sets
    check vertices
    check sizes of neighbors and vertices
    check for qh_MERGEridge and qh_DUPLICATEridge flags
    check neighbor set
    check ridge set
    check ridges, neighbors, and vertices
*/
void qh_checkfacet(qhT *qh, facetT *facet, boolT newmerge, boolT *waserrorp) {
  facetT *neighbor, **neighborp, *errother=NULL;
  ridgeT *ridge, **ridgep, *errridge= NULL, *ridge2;
  vertexT *vertex, **vertexp;
  unsigned int previousid= INT_MAX;
  int numneighbors, numvertices, numridges=0, numRvertices=0;
  boolT waserror= False;
  int skipA, skipB, ridge_i, ridge_n, i, last_v= qh->hull_dim-2;
  setT *intersection;

  trace4((qh, qh->ferr, 4088, "qh_checkfacet: check f%d newmerge? %d\n", facet->id, newmerge));
  // 检查 facet 的 id 是否超过了当前的 facet_id，若超过则报错
  if (facet->id >= qh->facet_id) {
    qh_fprintf(qh, qh->ferr, 6414, "qhull internal error (qh_checkfacet): unknown facet id f%d >= qh.facet_id (%d)\n", facet->id, qh->facet_id);
    waserror= True;
  }
  // 检查 facet 的 visitid 是否大于当前的 visit_id，若大于则报错
  if (facet->visitid > qh->visit_id) {
    qh_fprintf(qh, qh->ferr, 6415, "qhull internal error (qh_checkfacet): expecting f%d.visitid <= qh.visit_id (%d).  Got visitid %d\n", facet->id, qh->visit_id, facet->visitid);
    waserror= True;
  }
  // 如果 facet 是可见的并且不是处于 NEWtentative 状态，则报错
  if (facet->visible && !qh->NEWtentative) {
    qh_fprintf(qh, qh->ferr, 6119, "qhull internal error (qh_checkfacet): facet f%d is on qh.visible_list\n",
      facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  // 如果 facet 是冗余的且不可见，并且 degen_mergeset 为空，则报错
  if (facet->redundant && !facet->visible && qh_setsize(qh, qh->degen_mergeset)==0) {
    qh_fprintf(qh, qh->ferr, 6399, "qhull internal error (qh_checkfacet): redundant facet f%d not on qh.visible_list\n",
      facet->id);
    waserror= True;
  }
  // 如果 facet 是退化的且不可见，并且 degen_mergeset 为空，则报错
  if (facet->degenerate && !facet->visible && qh_setsize(qh, qh->degen_mergeset)==0) { 
    qh_fprintf(qh, qh->ferr, 6400, "qhull internal error (qh_checkfacet): degenerate facet f%d is not on qh.visible_list and qh.degen_mergeset is empty\n",
      facet->id);
    waserror= True;
  }
  // 如果 facet 的法向量为空，则报错
  if (!facet->normal) {
    ```
    qh_fprintf(qh, qh->ferr, 6120, "qhull internal error (qh_checkfacet): facet f%d does not have a normal\n",
      facet->id);
    // 打印错误消息到qh->ferr流，指示facet f%d没有法线
    waserror= True;
  }
  if (!facet->newfacet) {
    // 如果facet不是新创建的
    if (facet->dupridge) {
      // 如果facet有dupridge标记
      qh_fprintf(qh, qh->ferr, 6349, "qhull internal error (qh_checkfacet): f%d is 'dupridge' but it is not a newfacet on qh.newfacet_list f%d\n",
        facet->id, getid_(qh->newfacet_list));
      waserror= True;
    }
    if (facet->newmerge) {
      // 如果facet有newmerge标记
      qh_fprintf(qh, qh->ferr, 6383, "qhull internal error (qh_checkfacet): f%d is 'newmerge' but it is not a newfacet on qh.newfacet_list f%d.  Missing call to qh_reducevertices\n",  
        facet->id, getid_(qh->newfacet_list));
      waserror= True;
    }
  }
  // 检查facet的vertices、ridges、outsideset、coplanarset和neighbors是否有效
  qh_setcheck(qh, facet->vertices, "vertices for f", facet->id);
  qh_setcheck(qh, facet->ridges, "ridges for f", facet->id);
  qh_setcheck(qh, facet->outsideset, "outsideset for f", facet->id);
  qh_setcheck(qh, facet->coplanarset, "coplanarset for f", facet->id);
  qh_setcheck(qh, facet->neighbors, "neighbors for f", facet->id);
  // 遍历facet的vertices，检查是否有被删除的顶点或者顶点ID顺序错误
  FOREACHvertex_(facet->vertices) {
    if (vertex->deleted) {
      // 如果顶点被删除
      qh_fprintf(qh, qh->ferr, 6121, "qhull internal error (qh_checkfacet): deleted vertex v%d in f%d\n", vertex->id, facet->id);
      qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);
      waserror= True;
    }
    if (vertex->id >= previousid) {
      // 如果顶点ID不是降序排列
      qh_fprintf(qh, qh->ferr, 6122, "qhull internal error (qh_checkfacet): vertices of f%d are not in descending id order at v%d\n", facet->id, vertex->id);
      waserror= True;
      break;
    }
    previousid= vertex->id;
  }
  // 计算facet的neighbors、vertices和ridges的数量
  numneighbors= qh_setsize(qh, facet->neighbors);
  numvertices= qh_setsize(qh, facet->vertices);
  numridges= qh_setsize(qh, facet->ridges);
  if (facet->simplicial) {
    // 如果facet是简单面
    if (numvertices+numneighbors != 2*qh->hull_dim
    && !facet->degenerate && !facet->redundant) {
      // 检查简单面的顶点数和邻居数是否符合条件
      qh_fprintf(qh, qh->ferr, 6123, "qhull internal error (qh_checkfacet): for simplicial facet f%d, #vertices %d + #neighbors %d != 2*qh->hull_dim\n",
                facet->id, numvertices, numneighbors);
      qh_setprint(qh, qh->ferr, "", facet->neighbors);
      waserror= True;
    }
  }else { /* non-simplicial */
    // 如果facet不是简单面
    if (!newmerge
    &&(numvertices < qh->hull_dim || numneighbors < qh->hull_dim)
    && !facet->degenerate && !facet->redundant) {
      // 检查非简单面的顶点数和邻居数是否符合条件
      qh_fprintf(qh, qh->ferr, 6124, "qhull internal error (qh_checkfacet): for facet f%d, #vertices %d or #neighbors %d < qh->hull_dim\n",
         facet->id, numvertices, numneighbors);
       waserror= True;
    }
    /* in 3-d, can get a vertex twice in an edge list, e.g., RBOX 1000 s W1e-13 t995849315 D2 | QHULL d Tc Tv TP624 TW1e-13 T4 */
    // 在三维中，可能会在边缘列表中两次获取同一个顶点
    if (numridges < numneighbors
    ||(qh->hull_dim == 3 && numvertices > numridges && !qh->NEWfacets)
    ||(qh->hull_dim == 2 && numridges + numvertices + numneighbors != 6)) {
      // 检查条件：如果当前面 facet 不是退化的也不是冗余的
      if (!facet->degenerate && !facet->redundant) {
        // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出 facet f%d 的情况
        qh_fprintf(qh, qh->ferr, 6125, "qhull internal error (qh_checkfacet): for facet f%d, #ridges %d < #neighbors %d or(3-d) > #vertices %d or(2-d) not all 2\n",
            facet->id, numridges, numneighbors, numvertices);
        waserror= True;  // 设置错误标志
      }
    }
  }
  // 遍历 facet 的每一个邻居 neighbor
  FOREACHneighbor_(facet) {
    // 如果邻居是 MERGEridge 或 DUPLICATEridge
    if (neighbor == qh_MERGEridge || neighbor == qh_DUPLICATEridge) {
      // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出 facet f%d 仍然有 MERGEridge 或 DUPLICATEridge 邻居
      qh_fprintf(qh, qh->ferr, 6126, "qhull internal error (qh_checkfacet): facet f%d still has a MERGEridge or DUPLICATEridge neighbor\n", facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);  // 退出 qhull
    }
    // 如果邻居标记为可见
    if (neighbor->visible) {
      // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出 facet f%d 的已删除邻居 f%d（qh.visible_list）
      qh_fprintf(qh, qh->ferr, 6401, "qhull internal error (qh_checkfacet): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
        facet->id, neighbor->id);
      errother= neighbor;  // 记录错误的邻居
      waserror= True;  // 设置错误标志
    }
    neighbor->seen= True;  // 标记邻居为已见过
  }
  // 再次遍历 facet 的每一个邻居 neighbor
  FOREACHneighbor_(facet) {
    // 如果邻居没有在其邻居集合中找到当前面 facet
    if (!qh_setin(neighbor->neighbors, facet)) {
      // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出 facet f%d 有邻居 f%d，但 f%d 没有邻居 f%d
      qh_fprintf(qh, qh->ferr, 6127, "qhull internal error (qh_checkfacet): facet f%d has neighbor f%d, but f%d does not have neighbor f%d\n",
              facet->id, neighbor->id, neighbor->id, facet->id);
      errother= neighbor;  // 记录错误的邻居
      waserror= True;  // 设置错误标志
    }
    // 如果邻居未被标记为已见过
    if (!neighbor->seen) {
      // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出 facet f%d 有重复的邻居 f%d
      qh_fprintf(qh, qh->ferr, 6128, "qhull internal error (qh_checkfacet): facet f%d has a duplicate neighbor f%d\n",
              facet->id, neighbor->id);
      errother= neighbor;  // 记录错误的邻居
      waserror= True;  // 设置错误标志
    }
    neighbor->seen= False;  // 标记邻居为未见过，准备下一次循环
  }
  // 再次遍历 facet 的每一个 ridge（边）
  FOREACHridge_(facet->ridges) {
    // 检查 ridge 的顶点集合
    qh_setcheck(qh, ridge->vertices, "vertices for r", ridge->id);
    ridge->seen= False;  // 标记 ridge 为未见过，准备下一次循环
  }
  // 再次遍历 facet 的每一个 ridge（边）
  FOREACHridge_(facet->ridges) {
    // 如果 ridge 已经被标记为已见过
    if (ridge->seen) {
      // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出 facet f%d 有重复的 ridge r%d
      qh_fprintf(qh, qh->ferr, 6129, "qhull internal error (qh_checkfacet): facet f%d has a duplicate ridge r%d\n",
              facet->id, ridge->id);
      errridge= ridge;  // 记录错误的 ridge
      waserror= True;  // 设置错误标志
    }
    ridge->seen= True;  // 标记 ridge 为已见过
    numRvertices= qh_setsize(qh, ridge->vertices);  // 计算 ridge 的顶点数目
    // 如果 ridge 的顶点数目不等于 hull_dim - 1
    if (numRvertices != qh->hull_dim - 1) {
      // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出连接 facet f%d 和 f%d 的 ridge 有 %d 个顶点
      qh_fprintf(qh, qh->ferr, 6130, "qhull internal error (qh_checkfacet): ridge between f%d and f%d has %d vertices\n",
                ridge->top->id, ridge->bottom->id, numRvertices);
      errridge= ridge;  // 记录错误的 ridge
      waserror= True;  // 设置错误标志
    }
    neighbor= otherfacet_(ridge, facet);  // 获取 ridge 的另一个相邻面
    neighbor->seen= True;  // 标记邻居面为已见过
    // 如果当前面 facet 的邻居集合中没有包含邻居面 neighbor
    if (!qh_setin(facet->neighbors, neighbor)) {
      // 打印错误信息到 qh->ferr 流，报告 qhull 内部错误，指出对于 facet f%d，ridge r%d 的邻居面 f%d 不在 facet 的邻居集合中
      qh_fprintf(qh, qh->ferr, 6131, "qhull internal error (qh_checkfacet): for facet f%d, neighbor f%d of ridge r%d not in facet\n",
           facet->id, neighbor->id, ridge->id);
      errridge= ridge;  // 记录错误的 ridge
      waserror= True;  // 设置错误标志
    }
    // 检查如果 ridge 和 neighbor 都不是新创建的 facet
    if (!facet->newfacet && !neighbor->newfacet) {
      // 检查 ridge 是否未被测试过，或者是非凸的，或者需要合并顶点
      if ((!ridge->tested) | ridge->nonconvex | ridge->mergevertex) {
        // 输出错误信息到错误流，指示 ridge 的问题及其相关的 facet 和 neighbor 的信息
        qh_fprintf(qh, qh->ferr, 6384, "qhull internal error (qh_checkfacet): ridge r%d is nonconvex (%d), mergevertex (%d) or not tested (%d) for facet f%d, neighbor f%d\n",
          ridge->id, ridge->nonconvex, ridge->mergevertex, ridge->tested, facet->id, neighbor->id);
        // 记录错误的 ridge，并设置错误标志
        errridge= ridge;
        waserror= True;
      }
    }
  }
  // 如果 facet 不是单纯形结构
  if (!facet->simplicial) {
    // 遍历 facet 的每个 neighbor
    FOREACHneighbor_(facet) {
      // 如果 neighbor 还没有被看到过，输出错误信息，指示缺少 ridge 的情况
      if (!neighbor->seen) {
        qh_fprintf(qh, qh->ferr, 6132, "qhull internal error (qh_checkfacet): facet f%d does not have a ridge for neighbor f%d\n",
              facet->id, neighbor->id);
        // 记录错误的 neighbor
        errother= neighbor;
        waserror= True;
      }
      // 计算 facet 和 neighbor 之间的顶点交集
      intersection= qh_vertexintersect_new(qh, facet->vertices, neighbor->vertices);
      qh_settemppush(qh, intersection);
      // 将 facet 的所有顶点标记为未见过
      FOREACHvertex_(facet->vertices) {
        vertex->seen= False;
        vertex->seen2= False;
      }
      // 标记交集中的顶点为已见过
      FOREACHvertex_(intersection)
        vertex->seen= True;
      // 遍历 facet 的所有 ridge
      FOREACHridge_(facet->ridges) {
        // 如果 ridge 不是 facet 和 neighbor 之间的 ridge，跳过
        if (neighbor != otherfacet_(ridge, facet))
            continue;
        // 遍历 ridge 的所有顶点
        FOREACHvertex_(ridge->vertices) {
          // 如果顶点不在交集中，输出错误信息，指示顶点不在正确的位置
          if (!vertex->seen) {
            qh_fprintf(qh, qh->ferr, 6133, "qhull internal error (qh_checkfacet): vertex v%d in r%d not in f%d intersect f%d\n",
                  vertex->id, ridge->id, facet->id, neighbor->id);
            // 立即退出 Qhull 运行
            qh_errexit(qh, qh_ERRqhull, facet, ridge);
          }
          // 标记顶点为已见过
          vertex->seen2= True;
        }
      }
      // 如果不是新合并，检查交集中的顶点是否都在 ridge 中
      if (!newmerge) {
        FOREACHvertex_(intersection) {
          // 如果顶点不在任何 ridge 中，输出拓扑错误信息
          if (!vertex->seen2) {
            if (!qh->MERGING) {
              qh_fprintf(qh, qh->ferr, 6420, "qhull topology error (qh_checkfacet): vertex v%d in f%d intersect f%d but not in a ridge.  Last point was p%d\n",
                     vertex->id, facet->id, neighbor->id, qh->furthest_id);
              // 如果不强制输出，打印错误信息并退出 Qhull 运行
              if (!qh->FORCEoutput) {
                qh_errprint(qh, "ERRONEOUS", facet, neighbor, NULL, vertex);
                qh_errexit(qh, qh_ERRtopology, NULL, NULL);
              }
            } else {
              // 否则，输出追踪信息并继续执行
              trace4((qh, qh->ferr, 4025, "qh_checkfacet: vertex v%d in f%d intersect f%d but not in a ridge.  Repaired by qh_remove_extravertices in qh_reducevertices\n",
                vertex->id, facet->id, neighbor->id));
            }
          }
        }
      }
      // 释放交集的临时存储空间
      qh_settempfree(qh, &intersection);
    }
  } else { /* simplicial */
    FOREACHneighbor_(facet) {
      // 遍历当前facet的每个邻居neighbor
      if (neighbor->simplicial && !facet->degenerate && !neighbor->degenerate) {
        // 如果邻居是单纯的(simplicial)，并且当前facet和邻居都不是退化的(degenerate)
        skipA= SETindex_(facet->neighbors, neighbor);
        // 获取当前facet中与邻居的索引
        skipB= qh_setindex(neighbor->neighbors, facet);
        // 获取邻居中与当前facet的索引
        if (skipA<0 || skipB<0 || !qh_setequal_skip(facet->vertices, skipA, neighbor->vertices, skipB)) {
          // 如果索引不匹配或者顶点集合不相等
          qh_fprintf(qh, qh->ferr, 6135, "qhull internal error (qh_checkfacet): facet f%d skip %d and neighbor f%d skip %d do not match \n",
                     facet->id, skipA, neighbor->id, skipB);
          // 输出错误信息到qh->ferr
          errother= neighbor;
          // 将邻居设为错误的邻居
          waserror= True;
          // 设置错误标志为True
        }
      }
    }
  }
  if (!newmerge && qh->CHECKduplicates && qh->hull_dim < 5 && (qh->IStracing > 2 || qh->CHECKfrequently)) {
    // 如果不是新的合并(newmerge为假)，并且允许检查重复(qh->CHECKduplicates为真)，并且凸壳维度小于5，并且处于追踪级别大于2或者频繁检查
    FOREACHridge_i_(qh, facet->ridges) {           /* expensive, if was merge and qh_maybe_duplicateridges hasn't been called yet */
      // 遍历当前facet的每条ridge
      if (!ridge->mergevertex) {
        // 如果ridge没有合并的顶点
        for (i=ridge_i+1; i < ridge_n; i++) {
          // 遍历当前ridge后面的每条ridge
          ridge2= SETelemt_(facet->ridges, i, ridgeT);
          // 获取第二个ridge
          if (SETelem_(ridge->vertices, last_v) == SETelem_(ridge2->vertices, last_v)) { /* SETfirst is likely to be the same */
            // 如果第一个ridge的最后一个顶点与第二个ridge的最后一个顶点相等
            if (SETfirst_(ridge->vertices) == SETfirst_(ridge2->vertices)) {
              // 如果第一个ridge的第一个顶点与第二个ridge的第一个顶点相等
              if (qh_setequal(ridge->vertices, ridge2->vertices)) {
                // 如果两个ridge的顶点集合完全相等
                qh_fprintf(qh, qh->ferr, 6294, "qhull internal error (qh_checkfacet): ridges r%d and r%d (f%d) have the same vertices\n", /* same as duplicate ridge */
                    ridge->id, ridge2->id, facet->id);
                // 输出错误信息到qh->ferr，指出ridges有相同的顶点
                errridge= ridge;
                // 将第一个ridge标记为错误的ridge
                waserror= True;
                // 设置错误标志为True
              }
            }
          }
        }
      }
    }
  }
  if (waserror) {
    // 如果发生了错误
    qh_errprint(qh, "ERRONEOUS", facet, errother, errridge, NULL);
    // 打印错误信息
    *waserrorp= True;
    // 设置外部错误指针为True
  }
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkflipped_all">-</a>

  qh_checkflipped_all(qh, facetlist )
    checks orientation of facets in list against interior point

  notes:
    called by qh_checkoutput
*/
void qh_checkflipped_all(qhT *qh, facetT *facetlist) {
  facetT *facet;
  boolT waserror= False;
  realT dist;

  // 如果 facetlist 是 qh->facet_list，重置 Zflippedfacets 计数器
  if (facetlist == qh->facet_list)
    zzval_(Zflippedfacets)= 0;
  
  // 遍历 facetlist 中的每个 facet
  FORALLfacet_(facetlist) {
    // 如果 facet 有法向量并且检查法向量是否翻转，计算翻转距离
    if (facet->normal && !qh_checkflipped(qh, facet, &dist, !qh_ALL)) {
      // 打印错误信息，标记错误状态
      qh_fprintf(qh, qh->ferr, 6136, "qhull precision error: facet f%d is flipped, distance= %6.12g\n",
              facet->id, dist);
      // 如果非强制输出，打印错误信息，并设置错误标记
      if (!qh->FORCEoutput) {
        qh_errprint(qh, "ERRONEOUS", facet, NULL, NULL, NULL);
        waserror= True;
      }
    }
  }
  // 如果存在错误，打印详细信息并退出
  if (waserror) {
    qh_fprintf(qh, qh->ferr, 8101, "\n\
A flipped facet occurs when its distance to the interior point is\n\
greater than or equal to %2.2g, the maximum roundoff error.\n", -qh->DISTround);
    qh_errexit(qh, qh_ERRprec, NULL, NULL);
  }
} /* checkflipped_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checklists">-</a>

  qh_checklists(qh, facetlist )
    Check and repair facetlist and qh.vertex_list for infinite loops or overwritten facets
    Checks that qh.newvertex_list is on qh.vertex_list
    if facetlist is qh.facet_list
      Checks that qh.visible_list and qh.newfacet_list are on qh.facet_list
    Updates qh.facetvisit and qh.vertexvisit

  returns:
    True if no errors found
    If false, repairs erroneous lists to prevent infinite loops by FORALL macros

  notes:
    called by qh_buildtracing, qh_checkpolygon, qh_collectstatistics, qh_printfacetlist, qh_printsummary
    not called by qh_printlists

  design:
    if facetlist
      check qh.facet_tail
      for each facet
        check for infinite loop or overwritten facet
        check previous facet
      if facetlist is qh.facet_list
        check qh.next_facet, qh.visible_list and qh.newfacet_list
    if vertexlist
      check qh.vertex_tail
      for each vertex
        check for infinite loop or overwritten vertex
        check previous vertex
      check qh.newvertex_list
*/
boolT qh_checklists(qhT *qh, facetT *facetlist) {
  facetT *facet, *errorfacet= NULL, *errorfacet2= NULL, *previousfacet;
  vertexT *vertex, *vertexlist, *previousvertex, *errorvertex= NULL;
  boolT waserror= False, newseen= False, nextseen= False, newvertexseen= False, visibleseen= False;

  // 如果 facetlist 是 qh->newfacet_list 或者 qh->visible_list
  if (facetlist == qh->newfacet_list || facetlist == qh->visible_list) {
    vertexlist= qh->vertex_list;  // 设置 vertexlist 为 qh->vertex_list
    previousvertex= NULL;  // 初始化 previousvertex 为 NULL
    // 输出跟踪信息到标准错误流
    trace2((qh, qh->ferr, 2110, "qh_checklists: check qh.%s_list f%d and qh.vertex_list v%d\n", 
        (facetlist == qh->newfacet_list ? "newfacet" : "visible"), facetlist->id, getid_(vertexlist)));
  } else {
    vertexlist= qh->vertex_list;  // 设置 vertexlist 为 qh->vertex_list
    previousvertex= NULL;  // 初始化 previousvertex 为 NULL
    trace2((qh, qh->ferr, 2111, "qh_checklists: check %slist f%d and qh.vertex_list v%d\n", 
        (facetlist == qh->facet_list ? "qh.facet_" : "facet"), getid_(facetlist), getid_(vertexlist)));

// 跟踪调试信息，记录检查的列表信息和顶点列表信息

  }
  if (facetlist) {

// 如果facetlist不为NULL，执行以下代码块
    if (qh->facet_tail == NULL || qh->facet_tail->id != 0 || qh->facet_tail->next != NULL) {

// 检查qh.facet_tail是否为NULL，或者其id不为0，或者其next不为NULL，若满足条件则输出内部错误信息并退出
      qh_fprintf(qh, qh->ferr, 6397, "qhull internal error (qh_checklists): either qh.facet_tail f%d is NULL, or its id is not 0, or its next is not NULL\n", 
          getid_(qh->facet_tail));
      qh_errexit(qh, qh_ERRqhull, qh->facet_tail, NULL);
    }
    previousfacet= (facetlist == qh->facet_list ? NULL : facetlist->previous);

// 设置previousfacet为facetlist的前一个facet，如果facetlist等于qh.facet_list则设为NULL

    qh->visit_id++;

// 增加qh的visit_id计数器，用于跟踪访问标识

    FORALLfacet_(facetlist) {

// 遍历facetlist中的每个facet
      if (facet->visitid >= qh->visit_id || facet->id >= qh->facet_id) {

// 如果facet的visitid大于等于qh的visit_id或者facet的id大于等于qh的facet_id，则标记为错误
        waserror= True;
        errorfacet= facet;
        errorfacet2= previousfacet;
        if (facet->visitid == qh->visit_id)
          qh_fprintf(qh, qh->ferr, 6039, "qhull internal error (qh_checklists): f%d already in facetlist causing an infinite loop ... f%d > f%d ... > f%d > f%d.  Truncate facetlist at f%d\n", 
            facet->id, facet->id, facet->next->id, getid_(previousfacet), facet->id, getid_(previousfacet));
        else
          qh_fprintf(qh, qh->ferr, 6350, "qhull internal error (qh_checklists): unknown or overwritten facet f%d, either id >= qh.facet_id (%d) or f.visitid %u > qh.visit_id %u.  Facetlist terminated at previous facet f%d\n", 
              facet->id, qh->facet_id, facet->visitid, qh->visit_id, getid_(previousfacet));
        if (previousfacet)
          previousfacet->next= qh->facet_tail;
        else
          facetlist= qh->facet_tail;
        break;
      }
      facet->visitid= qh->visit_id;

// 设置facet的visitid为当前的visit_id，标记为已访问

      if (facet->previous != previousfacet) {

// 如果facet的前一个不等于previousfacet，则输出错误信息并标记为错误
        qh_fprintf(qh, qh->ferr, 6416, "qhull internal error (qh_checklists): expecting f%d.previous == f%d.  Got f%d\n",
          facet->id, getid_(previousfacet), getid_(facet->previous));
        waserror= True;
        errorfacet= facet;
        errorfacet2= facet->previous;
      }
      previousfacet= facet;

// 将previousfacet更新为当前的facet

      if (facetlist == qh->facet_list) {

// 如果facetlist等于qh.facet_list，执行以下代码块
        if (facet == qh->visible_list) {

// 如果facet等于qh.visible_list，则执行以下代码块
          if(newseen){
            qh_fprintf(qh, qh->ferr, 6285, "qhull internal error (qh_checklists): qh.visible_list f%d is after qh.newfacet_list f%d.  It should be at, before, or NULL\n",
              facet->id, getid_(qh->newfacet_list));
            waserror= True;
            errorfacet= facet;
            errorfacet2= qh->newfacet_list;
          }
          visibleseen= True;
        }
        if (facet == qh->newfacet_list)
          newseen= True;
        if (facet == qh->facet_next)
          nextseen= True;
      }
    }
    # 如果 facetlist 等于 qh->facet_list
    if (facetlist == qh->facet_list) {
      # 如果 nextseen 为假且 qh->facet_next 存在且其下一个也存在
      if (!nextseen && qh->facet_next && qh->facet_next->next) {
        # 输出 qhull 内部错误信息，指示 qh.facet_next 不在 qh.facet_list 上
        qh_fprintf(qh, qh->ferr, 6369, "qhull internal error (qh_checklists): qh.facet_next f%d for qh_addpoint is not on qh.facet_list f%d\n", 
          qh->facet_next->id, facetlist->id);
        waserror= True;
        errorfacet= qh->facet_next;
        errorfacet2= facetlist;
      }
      # 如果 newseen 为假且 qh->newfacet_list 存在且其下一个也存在
      if (!newseen && qh->newfacet_list && qh->newfacet_list->next) {
        # 输出 qhull 内部错误信息，指示 qh.newfacet_list 不在 qh.facet_list 上
        qh_fprintf(qh, qh->ferr, 6286, "qhull internal error (qh_checklists): qh.newfacet_list f%d is not on qh.facet_list f%d\n", 
          qh->newfacet_list->id, facetlist->id);
        waserror= True;
        errorfacet= qh->newfacet_list;
        errorfacet2= facetlist;
      }
      # 如果 visibleseen 为假且 qh->visible_list 存在且其下一个也存在
      if (!visibleseen && qh->visible_list && qh->visible_list->next) {
        # 输出 qhull 内部错误信息，指示 qh.visible_list 不在 qh.facet_list 上
        qh_fprintf(qh, qh->ferr, 6138, "qhull internal error (qh_checklists): qh.visible_list f%d is not on qh.facet_list f%d\n", 
          qh->visible_list->id, facetlist->id);
        waserror= True;
        errorfacet= qh->visible_list;
        errorfacet2= facetlist;
      }
    }
  }
  # 如果 vertexlist 存在
  if (vertexlist) {
    # 如果 qh->vertex_tail 为空或者其 id 不为 0 或者其下一个不为空
    if (qh->vertex_tail == NULL || qh->vertex_tail->id != 0 || qh->vertex_tail->next != NULL) {
      # 输出 qhull 内部错误信息，指示 qh.vertex_tail 的状态异常
      qh_fprintf(qh, qh->ferr, 6366, "qhull internal error (qh_checklists): either qh.vertex_tail v%d is NULL, or its id is not 0, or its next is not NULL\n", 
           getid_(qh->vertex_tail));
      # 打印错误信息的详细内容
      qh_errprint(qh, "ERRONEOUS", errorfacet, errorfacet2, NULL, qh->vertex_tail);
      # 终止 qhull 运行，抛出错误
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    # 增加访问过的顶点计数
    qh->vertex_visit++;
    FORALLvertex_(vertexlist) {
        // 遍历顶点列表中的每一个顶点

        if (vertex->visitid >= qh->vertex_visit || vertex->id >= qh->vertex_id) {
            // 如果顶点的访问标识大于或等于当前qhull的顶点访问标识，
            // 或者顶点的ID大于或等于当前qhull的顶点ID

            waserror= True;
            // 设置错误标志为True
            errorvertex= vertex;
            // 记录错误的顶点为当前顶点

            if (vertex->visitid == qh->visit_id)
                // 如果顶点的访问标识等于当前qhull的访问ID
                qh_fprintf(qh, qh->ferr, 6367, "qhull internal error (qh_checklists): v%d already in vertexlist causing an infinite loop ... v%d > v%d ... > v%d > v%d.  Truncate vertexlist at v%d\n", 
                    vertex->id, vertex->id, vertex->next->id, getid_(previousvertex), vertex->id, getid_(previousvertex));
            else
                // 否则，输出未知或覆盖的顶点错误信息
                qh_fprintf(qh, qh->ferr, 6368, "qhull internal error (qh_checklists): unknown or overwritten vertex v%d, either id >= qh.vertex_id (%d) or v.visitid %u > qh.visit_id %u.  vertexlist terminated at previous vertex v%d\n", 
                    vertex->id, qh->vertex_id, vertex->visitid, qh->visit_id, getid_(previousvertex));

            if (previousvertex)
                // 如果存在前一个顶点，则将前一个顶点的下一个设置为qhull的顶点尾部
                previousvertex->next= qh->vertex_tail;
            else
                // 否则，将顶点列表的头部设置为qhull的顶点尾部
                vertexlist= qh->vertex_tail;

            break;
            // 中断循环
        }

        vertex->visitid= qh->vertex_visit;
        // 设置顶点的访问标识为当前qhull的顶点访问标识

        if (vertex->previous != previousvertex) {
            // 如果顶点的前一个顶点不等于预期的前一个顶点
            qh_fprintf(qh, qh->ferr, 6427, "qhull internal error (qh_checklists): expecting v%d.previous == v%d.  Got v%d\n",
                  vertex->id, previousvertex, getid_(vertex->previous));
            waserror= True;
            // 设置错误标志为True
            errorvertex= vertex;
            // 记录错误的顶点为当前顶点
        }

        previousvertex= vertex;
        // 更新前一个顶点为当前顶点

        if(vertex == qh->newvertex_list)
            // 如果当前顶点等于qhull的新顶点列表
            newvertexseen= True;
            // 设置新顶点已见标志为True
    }

    if(!newvertexseen && qh->newvertex_list && qh->newvertex_list->next) {
        // 如果新顶点未被看到并且qhull的新顶点列表存在且有下一个顶点
        qh_fprintf(qh, qh->ferr, 6287, "qhull internal error (qh_checklists): new vertex list v%d is not on vertex list\n", qh->newvertex_list->id);
        waserror= True;
        // 设置错误标志为True
        errorvertex= qh->newvertex_list;
        // 记录错误的顶点为qhull的新顶点列表
    }
  }
  // 结束顶点列表的遍历

  if (waserror) {
      // 如果发生了错误
    qh_errprint(qh, "ERRONEOUS", errorfacet, errorfacet2, NULL, errorvertex);
    // 打印错误信息
    return False;
    // 返回False
  }
  return True;
  // 没有发生错误，返回True
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkpolygon">-</a>

  qh_checkpolygon(qh, facetlist )
    checks the correctness of the structure

  notes:
    called by qh_addpoint, qh_all_vertexmerge, qh_check_output, qh_initialhull, qh_prepare_output, qh_triangulate
    call with qh.facet_list or qh.newfacet_list or another list
    checks num_facets and num_vertices if qh.facet_list

  design:
    check and repair lists for infinite loop
    for each facet
      check f.newfacet and f.visible
      check facet and outside set if qh.NEWtentative and not f.newfacet, or not f.visible
    initializes vertexlist for qh.facet_list or qh.newfacet_list
    for each vertex
      check vertex
      check v.newfacet
    for each facet
      count f.ridges
      check and count f.vertices
    if checking qh.facet_list
      check facet count
      if qh.VERTEXneighbors
        check and count v.neighbors for all vertices
        check v.neighbors count and report possible causes of mismatch
        check that facets are in their v.neighbors
      check vertex count
*/

void qh_checkpolygon(qhT *qh, facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  facetT *errorfacet= NULL, *errorfacet2= NULL;
  vertexT *vertex, **vertexp, *vertexlist;
  int numfacets= 0, numvertices= 0, numridges= 0;
  int totvneighbors= 0, totfacetvertices= 0;
  boolT waserror= False, newseen= False, newvertexseen= False, nextseen= False, visibleseen= False;
  boolT checkfacet;

  // 输出跟踪信息，检查所有从 facetlist 开始的所有面片，同时检查是否使用了 qh.NEWtentative
  trace1((qh, qh->ferr, 1027, "qh_checkpolygon: check all facets from f%d, qh.NEWtentative? %d\n", facetlist->id, qh->NEWtentative));

  // 如果 qh_checklists 函数返回 false，记录错误并输出相关信息，若面片数量小于 4000，则打印列表
  if (!qh_checklists(qh, facetlist)) {
    waserror= True;
    qh_fprintf(qh, qh->ferr, 6374, "qhull internal error: qh_checklists failed in qh_checkpolygon\n");
    if (qh->num_facets < 4000)
      qh_printlists(qh);
  }

  // 如果 facetlist 不等于 qh.facet_list 或 qh.ONLYgood 为真，则设置 nextseen 为 True，允许 f.outsideset
  if (facetlist != qh->facet_list || qh->ONLYgood)
    nextseen= True; /* allow f.outsideset */

  // 遍历所有面片
  FORALLfacet_(facetlist) {
    // 检查是否当前面片为 qh.visible_list
    if (facet == qh->visible_list)
      visibleseen= True;
    // 检查是否当前面片为 qh.newfacet_list
    if (facet == qh->newfacet_list)
      newseen= True;

    // 如果当前面片标记为 newfacet 但未出现在 qh.newfacet_list 或 qh.visible_list 中，则输出错误信息并终止程序
    if (facet->newfacet && !newseen && !visibleseen) {
        qh_fprintf(qh, qh->ferr, 6289, "qhull internal error (qh_checkpolygon): f%d is 'newfacet' but it is not on qh.newfacet_list f%d or visible_list f%d\n",  facet->id, getid_(qh->newfacet_list), getid_(qh->visible_list));
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }

    // 如果当前面片未标记为 newfacet 但出现在 qh.newfacet_list 中，则输出错误信息并终止程序
    if (!facet->newfacet && newseen) {
        qh_fprintf(qh, qh->ferr, 6292, "qhull internal error (qh_checkpolygon): f%d is on qh.newfacet_list f%d but it is not 'newfacet'\n",  facet->id, getid_(qh->newfacet_list));
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }

    // More checks and operations follow for each facet...
    // 检查面的可见性是否符合预期
    if (facet->visible != (visibleseen & !newseen)) {
      // 如果面标记为可见但未出现在visible_list中，则输出错误信息
      if(facet->visible)
        qh_fprintf(qh, qh->ferr, 6290, "qhull internal error (qh_checkpolygon): f%d is 'visible' but it is not on qh.visible_list f%d\n", facet->id, getid_(qh->visible_list));
      // 如果面未标记为可见但出现在visible_list中，则输出错误信息
      else
        qh_fprintf(qh, qh->ferr, 6291, "qhull internal error (qh_checkpolygon): f%d is on qh.visible_list f%d but it is not 'visible'\n", facet->id, qh->newfacet_list->id);
      // 发现错误后退出程序
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    // 根据 NEWtentative 标志决定是否检查面的新建状态
    if (qh->NEWtentative) {
      // 如果 NEWtentative 为真，则检查面是否非新建状态
      checkfacet= !facet->newfacet;
    }else {
      // 如果 NEWtentative 为假，则检查面是否不可见
      checkfacet= !facet->visible;
    }
    // 如果需要检查面的状态
    if(checkfacet) {
      // 如果下一个面未被查看过
      if (!nextseen) {
        // 如果当前面是 qh->facet_next，则设置 nextseen 为真
        if (facet == qh->facet_next)  /* previous facets do not have outsideset */
          nextseen= True;
        // 否则，如果当前面的 outsideset 非空，则设置 nextseen 为真
        else if (qh_setsize(qh, facet->outsideset)) {
          if (!qh->NARROWhull
#if !qh_COMPUTEfurthest
          || facet->furthestdist >= qh->MINoutside
#endif
          ) {
            // 输出错误信息到 qh->ferr 文件流，指示面 f%d 在 qh.facet_next f%d 之前有外部点
            qh_fprintf(qh, qh->ferr, 6137, "qhull internal error (qh_checkpolygon): f%d has outside points before qh.facet_next f%d\n",
                     facet->id, getid_(qh->facet_next));
            // 终止程序执行，显示错误信息，标记错误类型为 qh_ERRqhull
            qh_errexit2(qh, qh_ERRqhull, facet, qh->facet_next);
          }
        }
      }
      // 增加 numfacets 计数
      numfacets++;
      // 检查面的有效性，设置 waserror 标志
      qh_checkfacet(qh, facet, False, &waserror);
    }else if (facet->visible && qh->NEWfacets) {
      // 对于可见面并且 qh.NEWfacets 为真的情况下
      if (!SETempty_(facet->neighbors) || !SETempty_(facet->ridges)) {
        // 输出错误信息到 qh->ferr 文件流，指示对于可见面 f%d，期望其邻居和棱为空，但得到了实际的邻居数和棱数
        qh_fprintf(qh, qh->ferr, 6376, "qhull internal error (qh_checkpolygon): expecting empty f.neighbors and f.ridges for visible facet f%d.  Got %d neighbors and %d ridges\n", 
          facet->id, qh_setsize(qh, facet->neighbors), qh_setsize(qh, facet->ridges));
        // 终止程序执行，显示错误信息，标记错误类型为 qh_ERRqhull
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
    }
  }
  // 根据 facetlist 的不同赋值 vertexlist
  if (facetlist == qh->facet_list) {
    vertexlist= qh->vertex_list;
  }else if (facetlist == qh->newfacet_list) {
    vertexlist= qh->newvertex_list;
  }else {
    vertexlist= NULL;
  }
  // 遍历 vertexlist 中的每个顶点 vertex
  FORALLvertex_(vertexlist) {
    // 检查顶点的有效性，不包括所有的顶点，设置 waserror 标志
    qh_checkvertex(qh, vertex, !qh_ALL, &waserror);
    // 如果当前顶点是 qh.newvertex_list，设置 newvertexseen 为真
    if(vertex == qh->newvertex_list)
      newvertexseen= True;
    // 重置顶点的 seen、visitid 标志
    vertex->seen= False;
    vertex->visitid= 0;
    // 如果顶点标记为 newfacet 且 newvertexseen 为假且顶点未被删除，则输出错误信息
    if(vertex->newfacet && !newvertexseen && !vertex->deleted) {
      // 输出错误信息到 qh->ferr 文件流，指示顶点 v%d 被标记为 'newfacet'，但未出现在 new vertex list v%d 中
      qh_fprintf(qh, qh->ferr, 6288, "qhull internal error (qh_checkpolygon): v%d is 'newfacet' but it is not on new vertex list v%d\n", vertex->id, getid_(qh->newvertex_list));
      // 终止程序执行，显示错误信息，标记错误类型为 qh_ERRqhull
      qh_errexit(qh, qh_ERRqhull, qh->visible_list, NULL);
    }
  }
  // 遍历 facetlist 中的每个面 facet
  FORALLfacet_(facetlist) {
    // 对于可见面跳过后续处理
    if (facet->visible)
      continue;
    // 如果面是单纯形，则增加 numridges 计数
    if (facet->simplicial)
      numridges += qh->hull_dim;
    else
      // 否则增加面的棱数到 numridges 计数
      numridges += qh_setsize(qh, facet->ridges);
    // 遍历面的每个顶点
    FOREACHvertex_(facet->vertices) {
      // 增加顶点的 visitid 标志
      vertex->visitid++;
      // 如果顶点未被标记为 seen，则标记为 seen，并增加 numvertices 计数
      if (!vertex->seen) {
        vertex->seen= True;
        numvertices++;
        // 如果顶点对应的点为未知点，则输出错误信息
        if (qh_pointid(qh, vertex->point) == qh_IDunknown) {
          // 输出错误信息到 qh->ferr 文件流，指示顶点 v%d 的点 %p 未知
          qh_fprintf(qh, qh->ferr, 6139, "qhull internal error (qh_checkpolygon): unknown point %p for vertex v%d first_point %p\n",
                   vertex->point, vertex->id, qh->first_point);
          // 设置 waserror 标志
          waserror= True;
        }
      }
    }
  }
  // 更新 vertex_visit 计数
  qh->vertex_visit += (unsigned int)numfacets;
  // 如果 facetlist 为 qh->facet_list，则检查 numfacets 是否等于实际的面数和隐藏面数的差
  if (facetlist == qh->facet_list) {
    if (numfacets != qh->num_facets - qh->num_visible) {
      // 输出错误信息到 qh->ferr 文件流，指示实际面数与累计面数和隐藏面数的差异
      qh_fprintf(qh, qh->ferr, 6140, "qhull internal error (qh_checkpolygon): actual number of facets is %d, cumulative facet count is %d - %d visible facets\n",
              numfacets, qh->num_facets, qh->num_visible);
      // 设置 waserror 标志
      waserror= True;
    }
    // 增加 vertex_visit 计数
    qh->vertex_visit++;
    # 检查顶点是否有邻接顶点，若有则执行以下操作
    if (qh->VERTEXneighbors) {
      # 遍历所有顶点
      FORALLvertices {
        # 如果顶点没有邻接顶点，则输出错误信息并标记错误
        if (!vertex->neighbors) {
          qh_fprintf(qh, qh->ferr, 6407, "qhull internal error (qh_checkpolygon): missing vertex neighbors for v%d\n", vertex->id);
          waserror= True;
        }
        # 检查顶点的邻接顶点集合是否正确，记录错误信息
        qh_setcheck(qh, vertex->neighbors, "neighbors for v", vertex->id);
        # 如果顶点被删除，则跳过该顶点
        if (vertex->deleted)
          continue;
        # 统计顶点的总邻接顶点数目
        totvneighbors += qh_setsize(qh, vertex->neighbors);
      }
      # 遍历所有凸包面
      FORALLfacet_(facetlist) {
        # 如果面不可见，则统计其顶点数目，用于后续错误检查
        if (!facet->visible)
          totfacetvertices += qh_setsize(qh, facet->vertices);
      }
      # 如果顶点邻接总数与凸包面顶点总数不一致，则输出错误信息并标记错误
      if (totvneighbors != totfacetvertices) {
        qh_fprintf(qh, qh->ferr, 6141, "qhull internal error (qh_checkpolygon): vertex neighbors inconsistent (tot_vneighbors %d != tot_facetvertices %d).  Maybe duplicate or missing vertex\n",
                totvneighbors, totfacetvertices);
        waserror= True;
        # 再次遍历所有顶点，检查重复面和不存在的顶点的错误情况
        FORALLvertices {
          if (vertex->deleted)
            continue;
          # 增加访问标记，检查邻接面是否有重复
          qh->visit_id++;
          FOREACHneighbor_(vertex) {
            if (neighbor->visitid==qh->visit_id) {
              # 如果邻接面已被访问，则输出错误信息
              qh_fprintf(qh, qh->ferr, 6275, "qhull internal error (qh_checkpolygon): facet f%d occurs twice in neighbors of vertex v%d\n",
                  neighbor->id, vertex->id);
              errorfacet2= errorfacet;
              errorfacet= neighbor;
            }
            # 更新邻接面的访问标记，并检查顶点是否为邻接面的顶点
            neighbor->visitid= qh->visit_id;
            if (!qh_setin(neighbor->vertices, vertex)) {
              # 如果顶点不是邻接面的顶点，则输出错误信息
              qh_fprintf(qh, qh->ferr, 6276, "qhull internal error (qh_checkpolygon): facet f%d is a neighbor of vertex v%d but v%d is not a vertex of f%d\n",
                  neighbor->id, vertex->id, vertex->id, neighbor->id);
              errorfacet2= errorfacet;
              errorfacet= neighbor;
            }
          }
        }
        # 再次遍历所有面，检查顶点是否是面的顶点，但面不是顶点的邻接面的错误情况
        FORALLfacet_(facetlist){
          if (!facet->visible) {
            # 面的顶点按照反向排序，通常不会重复，检查顶点是否为面的顶点但面不是顶点的邻接面的错误情况
            FOREACHvertex_(facet->vertices){
              if (!qh_setin(vertex->neighbors, facet)) {
                qh_fprintf(qh, qh->ferr, 6277, "qhull internal error (qh_checkpolygon): v%d is a vertex of facet f%d but f%d is not a neighbor of v%d\n",
                  vertex->id, facet->id, facet->id, vertex->id);
                errorfacet2= errorfacet;
                errorfacet= facet;
              }
            }
          }
        }
      }
    }
    # 检查实际顶点数是否与预期不同的错误情况
    if (numvertices != qh->num_vertices - qh_setsize(qh, qh->del_vertices)) {
      qh_fprintf(qh, qh->ferr, 6142, "qhull internal error (qh_checkpolygon): actual number of vertices is %d, cumulative vertex count is %d\n",
              numvertices, qh->num_vertices - qh_setsize(qh, qh->del_vertices));
      waserror= True;
    }
    # 如果是二维情况下，检查顶点数与面数是否相等的错误情况
    if (qh->hull_dim == 2 && numvertices != numfacets) {
      qh_fprintf(qh, qh->ferr, 6143, "qhull internal error (qh_checkpolygon): #vertices %d != #facets %d\n",
        numvertices, numfacets);
      waserror= True;
    }
    # 如果凸壳维度为3，并且顶点数加上面数减去边数的一半不等于2时，发出警告
    if (qh->hull_dim == 3 && numvertices + numfacets - numridges/2 != 2) {
      # 打印警告信息到错误输出流，说明顶点数、面数、边数的关系异常
      qh_fprintf(qh, qh->ferr, 7063, "qhull warning: #vertices %d + #facets %d - #edges %d != 2.  A vertex appears twice in a edge list.  May occur during merging.\n",
          numvertices, numfacets, numridges/2);
      /* 如果大量合并导致一个顶点出现两次在边列表中，例如，使用命令 RBOX 1000 s W1e-13 t995849315 D2 | QHULL d Tc Tv */
    }
  }
  # 如果发生过错误，通过 qh_errexit2 函数退出程序，参数说明错误类型和相关的面
  if (waserror)
    qh_errexit2(qh, qh_ERRqhull, errorfacet, errorfacet2);
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkvertex">-</a>

  qh_checkvertex(qh, vertex, allchecks, &waserrorp )
    check vertex for consistency
    if allchecks, checks vertex->neighbors

  returns:
    sets waserrorp if any error occurs

  notes:
    called by qh_tracemerge and qh_checkpolygon
    neighbors checked efficiently in qh_checkpolygon
*/
void qh_checkvertex(qhT *qh, vertexT *vertex, boolT allchecks, boolT *waserrorp) {
  boolT waserror= False;  /* 初始化错误标志为 False */
  facetT *neighbor, **neighborp, *errfacet=NULL;  /* 定义邻近面、邻近面指针和错误面 */

  if (qh_pointid(qh, vertex->point) == qh_IDunknown) {
    qh_fprintf(qh, qh->ferr, 6144, "qhull internal error (qh_checkvertex): unknown point id %p\n", vertex->point);
    waserror= True;  /* 如果顶点的点标识未知，设置错误标志为 True */
  }
  if (vertex->id >= qh->vertex_id) {
    qh_fprintf(qh, qh->ferr, 6145, "qhull internal error (qh_checkvertex): unknown vertex id v%d >= qh.vertex_id (%d)\n", vertex->id, qh->vertex_id);
    waserror= True;  /* 如果顶点的 ID 大于或等于 qh.vertex_id，设置错误标志为 True */
  }
  if (vertex->visitid > qh->vertex_visit) {
    qh_fprintf(qh, qh->ferr, 6413, "qhull internal error (qh_checkvertex): expecting v%d.visitid <= qh.vertex_visit (%d).  Got visitid %d\n", vertex->id, qh->vertex_visit, vertex->visitid);
    waserror= True;  /* 如果顶点的访问标识大于 qh.vertex_visit，设置错误标志为 True */
  }
  if (allchecks && !waserror && !vertex->deleted) {
    if (qh_setsize(qh, vertex->neighbors)) {
      FOREACHneighbor_(vertex) {
        if (!qh_setin(neighbor->vertices, vertex)) {
          qh_fprintf(qh, qh->ferr, 6146, "qhull internal error (qh_checkvertex): neighbor f%d does not contain v%d\n", neighbor->id, vertex->id);
          errfacet= neighbor;
          waserror= True;  /* 如果顶点未被其邻近面包含，设置错误标志为 True */
        }
      }
    }
  }
  if (waserror) {
    qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);  /* 打印错误信息 */
    if (errfacet)
      qh_errexit(qh, qh_ERRqhull, errfacet, NULL);  /* 如果存在错误面，退出程序 */
    *waserrorp= True;  /* 设置外部错误指针 */
  }
} /* checkvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="clearcenters">-</a>

  qh_clearcenters(qh, type )
    clear old data from facet->center

  notes:
    sets new centertype
    nop if CENTERtype is the same
*/
void qh_clearcenters(qhT *qh, qh_CENTER type) {
  facetT *facet;  /* 定义面对象 */

  if (qh->CENTERtype != type) {
    FORALLfacets {  /* 遍历所有面 */
      if (facet->tricoplanar && !facet->keepcentrum)
          facet->center= NULL;  /* 如果面为三共面且不保留中心点，则将中心点设为 NULL */
      else if (qh->CENTERtype == qh_ASvoronoi){
        if (facet->center) {
          qh_memfree(qh, facet->center, qh->center_size);
          facet->center= NULL;  /* 如果中心类型为 Voronoi，则释放并设为空 */
        }
      }else /* qh.CENTERtype == qh_AScentrum */ {
        if (facet->center) {
          qh_memfree(qh, facet->center, qh->normal_size);
          facet->center= NULL;  /* 如果中心类型为 centrum，则释放并设为空 */
        }
      }
    }
    qh->CENTERtype= type;  /* 设置新的中心类型 */
  }
  trace2((qh, qh->ferr, 2043, "qh_clearcenters: switched to center type %d\n", type));  /* 跟踪信息，记录中心类型切换 */
} /* clearcenters */
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="createsimplex">-</a>

  qh_createsimplex(qh, vertices )
    creates a simplex from a set of vertices

  returns:
    initializes qh.facet_list to the simplex

  notes: 
    only called by qh_initialhull

  design:
    for each vertex
      create a new facet
    for each new facet
      create its neighbor set
*/
void qh_createsimplex(qhT *qh, setT *vertices /* qh.facet_list */) {
  facetT *facet= NULL, *newfacet;
  boolT toporient= True;
  int vertex_i, vertex_n, nth;
  setT *newfacets= qh_settemp(qh, qh->hull_dim+1);
  vertexT *vertex;

  // 遍历输入的顶点集合
  FOREACHvertex_i_(qh, vertices) {
    // 创建一个新的 facet
    newfacet= qh_newfacet(qh);
    // 将 vertices 中的顶点复制到新 facet 的 vertices 属性中
    newfacet->vertices= qh_setnew_delnthsorted(qh, vertices, vertex_n, vertex_i, 0);
    // 根据 toporient 的值设置新 facet 的 toporient 属性
    if (toporient)
      newfacet->toporient= True;
    // 将新 facet 添加到 qh.facet_list 中
    qh_appendfacet(qh, newfacet);
    // 标记新 facet 为 newfacet
    newfacet->newfacet= True;
    // 将顶点添加到 qh 中
    qh_appendvertex(qh, vertex);
    // 将新创建的 facet 添加到 newfacets 集合中
    qh_setappend(qh, &newfacets, newfacet);
    // 切换 toporient 的值
    toporient ^= True;
  }
  // 遍历所有新创建的 facets
  FORALLnew_facets {
    nth= 0;
    // 遍历所有的 facet
    FORALLfacet_(qh->newfacet_list) {
      // 将当前 facet 添加为新 facet 的邻居
      if (facet != newfacet)
        SETelem_(newfacet->neighbors, nth++)= facet;
    }
    // 截断新 facet 的邻居集合
    qh_settruncate(qh, newfacet->neighbors, qh->hull_dim);
  }
  // 释放临时集合 newfacets
  qh_settempfree(qh, &newfacets);
  // 记录追踪信息
  trace1((qh, qh->ferr, 1028, "qh_createsimplex: created simplex\n"));
} /* createsimplex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="delridge">-</a>

  qh_delridge(qh, ridge )
    delete a ridge's vertices and frees its memory

  notes:
    assumes r.top->ridges and r.bottom->ridges have been updated
*/
void qh_delridge(qhT *qh, ridgeT *ridge) {

  // 如果当前 ridge 是追踪的 ridge，则将 qh->traceridge 置为 NULL
  if (ridge == qh->traceridge)
    qh->traceridge= NULL;
  // 释放 ridge 的顶点集合
  qh_setfree(qh, &(ridge->vertices));
  // 释放 ridge 的内存
  qh_memfree(qh, ridge, (int)sizeof(ridgeT));
} /* delridge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="delvertex">-</a>

  qh_delvertex(qh, vertex )
    deletes a vertex and frees its memory

  notes:
    assumes vertex->adjacencies have been updated if needed
    unlinks from vertex_list
*/
void qh_delvertex(qhT *qh, vertexT *vertex) {

  // 如果 vertex 被删除且未分区且不允许 NOerrexit，则记录错误信息并退出
  if (vertex->deleted && !vertex->partitioned && !qh->NOerrexit) {
    qh_fprintf(qh, qh->ferr, 6395, "qhull internal error (qh_delvertex): vertex v%d was deleted but it was not partitioned as a coplanar point\n",
      vertex->id);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 如果 vertex 是追踪的 vertex，则将 qh->tracevertex 置为 NULL
  if (vertex == qh->tracevertex)
    qh->tracevertex= NULL;
  // 从 vertex_list 中移除 vertex
  qh_removevertex(qh, vertex);
  // 释放 vertex 的邻居集合
  qh_setfree(qh, &vertex->neighbors);
  // 释放 vertex 的内存
  qh_memfree(qh, vertex, (int)sizeof(vertexT));
} /* delvertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="facet3vertex">-</a>

  qh_facet3vertex(qh )
    return temporary set of 3-d vertices in qh_ORIENTclock order

  design:
    if simplicial facet
      build set from facet->vertices with facet->toporient
*/
    else:
        # 对于排序后的每条边界线（ridge），执行以下操作
        for each ridge in order:
            # 根据当前边界线（ridge）的顶点构建一个集合
            build set from ridge's vertices
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findbestfacet">-</a>

  qh_findbestfacet(qh, point, bestoutside, bestdist, isoutside )
    find facet that is furthest below a point

    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    if bestoutside is set (e.g., qh_ALL)
      returns best facet that is not upperdelaunay
      if Delaunay and inside, point is outside circumsphere of bestfacet
    else
      returns first facet below point
      if point is inside, returns nearest, !upperdelaunay facet
    distance to facet
    isoutside set if outside of facet

  notes:
    Distance is measured by distance to the facet's hyperplane.  For
    Delaunay facets, this is not the same as the containing facet.  It may
    be an adjacent facet or a different tricoplanar facet.  See 
    <a href="../html/qh-code.htm#findfacet">locate a facet with qh_findbestfacet()</a>

    For tricoplanar facets, this finds one of the tricoplanar facets closest
    to the point.  

    If inside, qh_findbestfacet performs an exhaustive search
       this may be too conservative.  Sometimes it is clearly required.

    qh_findbestfacet is not used by qhull.
    uses qh.visit_id and qh.coplanarset

  see:
    <a href="geom_r.c#findbest">qh_findbest</a>
*/


注释：
这段代码定义了 `qh_findbestfacet` 函数，用于寻找离给定点最远的下方的 facet（多面体）。该函数针对 Delaunay 三角剖分进行了特别处理，并提供了详细的返回值说明和注意事项。
/* 查找最优面函数，用于找到最适合给定点的最佳面
   qh: Qhull程序的全局状态结构体
   point: 给定的点坐标
   bestoutside: 是否考虑面的外部
   bestdist: 存储最佳距离的指针
   isoutside: 是否在面的外部的指针
*/
facetT *qh_findbestfacet(qhT *qh, pointT *point, boolT bestoutside,
                         realT *bestdist, boolT *isoutside) {
  facetT *bestfacet= NULL;  // 最优面的指针，初始化为NULL
  int numpart, totpart= 0;  // 用于计数的变量初始化

  // 调用qh_findbest函数寻找最优面
  bestfacet= qh_findbest(qh, point, qh->facet_list,
                        bestoutside, !qh_ISnewfacets, bestoutside /* qh_NOupper */,
                        bestdist, isoutside, &totpart);
  // 如果最佳距离小于负的DISTround值，进行额外处理
  if (*bestdist < -qh->DISTround) {
    // 调用qh_findfacet_all函数查找所有满足条件的面
    bestfacet= qh_findfacet_all(qh, point, !qh_NOupper, bestdist, isoutside, &numpart);
    totpart += numpart;  // 更新总计数

    // 根据条件再次判断最优面的选择
    if ((isoutside && *isoutside && bestoutside)
        || (isoutside && !*isoutside && bestfacet->upperdelaunay)) {
      // 重新调用qh_findbest函数以确定最优面
      bestfacet= qh_findbest(qh, point, bestfacet,
                            bestoutside, False, bestoutside,
                            bestdist, isoutside, &totpart);
      totpart += numpart;  // 更新总计数
    }
  }

  // 输出跟踪信息并返回最优面的指针
  trace3((qh, qh->ferr, 3014, "qh_findbestfacet: f%d dist %2.2g isoutside %d totpart %d\n",
          bestfacet->id, *bestdist, (isoutside ? *isoutside : UINT_MAX), totpart));
  return bestfacet;
} /* findbestfacet */

/* 查找最佳非上层非翻转邻面函数，用于给定点和面查找最适合的邻面
   upperfacet: 给定面的指针
   point: 给定点坐标
   bestdistp: 存储最佳距离的指针
   numpart: 用于计数的指针
*/
facetT *qh_findbestlower(qhT *qh, facetT *upperfacet, pointT *point, realT *bestdistp, int *numpart) {
  facetT *neighbor, **neighborp, *bestfacet= NULL;  // 定义变量和指针
  realT bestdist= -REALmax/2 /* avoid underflow */;  // 初始化最佳距离，避免下溢
  realT dist;  // 存储距离的变量
  vertexT *vertex;  // 顶点指针
  boolT isoutside= False;  // 是否在外部的标志，这里未使用

  zinc_(Zbestlower);  // 增加Zbestlower计数

  // 遍历给定面的邻面
  FOREACHneighbor_(upperfacet) {
    // 如果邻面是上层Delaunay面或者已经翻转，跳过
    if (neighbor->upperdelaunay || neighbor->flipped)
      continue;

    (*numpart)++;  // 增加计数
    // 计算点到邻面的距离
    qh_distplane(qh, point, neighbor, &dist);
    // 如果距离比当前最佳距离大，则更新最优面和最佳距离
    if (dist > bestdist) {
      bestfacet= neighbor;
      bestdist= dist;
    }
  }

  // 如果没有找到最优面，进行额外的处理
  if (!bestfacet) {
    zinc_(Zbestlowerv);  // 增加Zbestlowerv计数
    // 很少情况下调用，numpart不计算近点的计算
    // 查找最近点
    vertex= qh_nearvertex(qh, upperfacet, point, &dist);
    qh_vertexneighbors(qh);  // 更新顶点的邻面列表
    // 遍历最近点的邻面
    FOREACHneighbor_(vertex) {
      // 如果邻面是上层Delaunay面或者已经翻转，跳过
      if (neighbor->upperdelaunay || neighbor->flipped)
        continue;

      (*numpart)++;  // 增加计数
      // 计算点到邻面的距离
      qh_distplane(qh, point, neighbor, &dist);
      // 如果距离比当前最佳距离大，则更新最优面和最佳距离
      if (dist > bestdist) {
        bestfacet= neighbor;
        bestdist= dist;
      }
    }
  }

  // 如果仍未找到最优面，进行最后的处理
  if (!bestfacet) {
    zinc_(Zbestlowerall);  // 增加Zbestlowerall计数
    zmax_(Zbestloweralln, qh->num_facets);  // 更新Zbestloweralln的最大值
    // 输出跟踪信息
    trace3((qh, qh->ferr, 3025, "qh_findbestlower: all neighbors of facet %d are flipped or upper Delaunay.  Search all facets\n",
            upperfacet->id));
    // 很少调用的情况
  }

  // 返回找到的最优面的指针
  return bestfacet;
}
    bestfacet= qh_findfacet_all(qh, point, qh_NOupper, &bestdist, &isoutside, numpart);

获取通过 `qh_findfacet_all` 函数找到的最佳面对象，该函数通过给定的参数在几何处理库 `qh` 中查找符合条件的面。


  *bestdistp= bestdist;

将指针 `bestdistp` 指向的内存位置设置为 `bestdist` 的值，`bestdist` 是一个双精度浮点数，表示找到的最佳距离。


  trace3((qh, qh->ferr, 3015, "qh_findbestlower: f%d dist %2.2g for f%d p%d\n",
          bestfacet->id, bestdist, upperfacet->id, qh_pointid(qh, point)));

记录跟踪日志到 `qh->ferr` 中，输出格式化的消息字符串，包括最佳面对象 `bestfacet` 的标识符 `id`、距离 `bestdist`、上层面对象 `upperfacet` 的标识符 `id` 和点 `point` 的标识符。


  return bestfacet;

返回找到的最佳面对象 `bestfacet`。
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findbestlower">-</a>

  qh_findbestlower(qh, facet, point, dist, mindist, bestfacet, isoutside, numpart )
    search for the facet below a point, ignoring flipped and visible facets
    returns nearest facet and distance

  returns:
    returns first facet below point
    distance to facet
    isoutside if point is outside of the hull
    number of distance tests

  notes:
    called by qh_findbestfacet (QH3015)
    searches all neighbors
*/
facetT *qh_findbestlower(qhT *qh, facetT *facet, pointT *point, realT *dist, realT *mindist,
                          facetT **bestfacet, boolT *isoutside, int *numpart) {
  facetT *neighbor, **neighborp;
  realT angle, bestangle= REALmax;

  *dist= -REALmax;
  FOREACHneighbor_(facet) {
    if (neighbor->flipped || !neighbor->normal || neighbor->visible)
      continue;
    qh_distplane(qh, point, neighbor, dist);
    if (*dist > *mindist) {
      *mindist= *dist;
      *bestfacet= neighbor;
      if (*dist > qh->MINoutside) {
        *isoutside= True;
        break;
      }
    }
  }
  *numpart= qh->num_facets;
  trace3((qh, qh->ferr, 3016, "qh_findbestlower: f%d, p%d, dist %2.2g, mindist %2.2g, bestfacet %d, isoutside %d, numpart %d\n",
          getid_(facet), qh_pointid(qh, point), *dist, *mindist, getid_(*bestfacet), *isoutside, *numpart));
  return *bestfacet;
} /* findbestlower */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findfacet_all">-</a>

  qh_findfacet_all(qh, point, noupper, bestdist, isoutside, numpart )
    exhaustive search for facet below a point
    ignore flipped and visible facets, f.normal==NULL, and if noupper, f.upperdelaunay facets

    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    returns first facet below point
    if point is inside,
      returns nearest facet
    distance to facet
    isoutside if point is outside of the hull
    number of distance tests

  notes:
    called by qh_findbestlower if all neighbors are flipped or upper Delaunay (QH3025)
    primarily for library users (qh_findbestfacet), rarely used by Qhull
*/
facetT *qh_findfacet_all(qhT *qh, pointT *point, boolT noupper, realT *bestdist, boolT *isoutside,
                          int *numpart) {
  facetT *bestfacet= NULL, *facet;
  realT dist;
  int totpart= 0;

  *bestdist= -REALmax;
  *isoutside= False;
  FORALLfacets {
    if (facet->flipped || !facet->normal || facet->visible)
      continue;
    if (noupper && facet->upperdelaunay)
      continue;
    totpart++;
    qh_distplane(qh, point, facet, &dist);
    if (dist > *bestdist) {
      *bestdist= dist;
      bestfacet= facet;
      if (dist > qh->MINoutside) {
        *isoutside= True;
        break;
      }
    }
  }
  *numpart= totpart;
  trace3((qh, qh->ferr, 3016, "qh_findfacet_all: p%d, noupper? %d, f%d, dist %2.2g, isoutside %d, totpart %d\n",
      qh_pointid(qh, point), noupper, getid_(bestfacet), *bestdist, *isoutside, totpart));
  return bestfacet;
} /* findfacet_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findgood">-</a>

  qh_findgood(qh, facetlist, goodhorizon )
    identify good facets for qh.PRINTgood and qh_buildcone_onlygood
    goodhorizon is count of good, horizon facets from qh_find_horizon, otherwise 0 from qh_findgood_all
    if not qh.MERGING and qh.GOODvertex>0
      facet includes point as vertex
      if !match, returns goodhorizon
    if qh.GOODpoint
      facet is visible or coplanar (>0) or not visible (<0)
    if qh.GOODthreshold
      facet->normal matches threshold
    if !goodhorizon and !match,
      selects facet with closest angle to thresholds
      sets GOODclosest

  returns:
    number of new, good facets found
    determines facet->good
    may update qh.GOODclosest

  notes:
    called from qh_initbuild, qh_buildcone_onlygood, and qh_findgood_all
    qh_findgood_all (called from qh_prepare_output) further reduces the good region

  design:
    count good facets
    if not merging, clear good facets that fail qh.GOODvertex ('QVn', but not 'QV-n')
    clear good facets that fail qh.GOODpoint ('QGn' or 'QG-n')
*/
    # 清除未达到 qh.GOODthreshold 的好的面（facets）
    if not goodhorizon and not find f.good:
        # 如果没有好的水平线（goodhorizon）并且没有找到好的面（f.good），
        # 将 GOODclosest 设置为与阈值角度最接近的面（facet）
        sets GOODclosest to facet with closest angle to thresholds
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findgood_all">-</a>

  qh_findgood_all(qh, facetlist )
    apply other constraints for good facets (used by qh.PRINTgood)
    if qh.GOODvertex
      facet includes (>0) or doesn't include (<0) point as vertex
      if last good facet and ONLYgood, prints warning and continues
    if qh.SPLITthresholds (e.g., qh.DELAUNAY)
      facet->normal matches threshold, or if none, the closest one
    calls qh_findgood
    nop if good not used

  returns:
    clears facet->good if not good
    sets qh.num_good

  notes:
    called by qh_prepare_output and qh_printneighborhood
*/

void qh_findgood_all(qhT *qh, facetT *facetlist) {
  facetT *facet;
  int numgood;

  /* 初始化 numgood 为 0 */
  numgood = 0;

  /* 遍历所有的 facet */
  FORALLfacet_(facetlist) {
    /* 如果 facet 标记为 good，则增加 numgood 计数 */
    if (facet->good)
      numgood++;
  }

  /* 如果 qh.GOODvertex 大于 0 并且不处于 MERGING 状态 */
  if (qh->GOODvertex > 0 && !qh->MERGING) {
    /* 再次遍历所有的 facet */
    FORALLfacet_(facetlist) {
      /* 如果 facet 标记为 good 并且 qh.GOODvertexp 不是 facet 的顶点 */
      if (facet->good && !qh_isvertex(qh->GOODvertexp, facet->vertices)) {
        /* 将 facet 标记为非 good，减少 numgood 计数 */
        facet->good = False;
        numgood--;
      }
    }
  }

  /* 如果 qh.GOODpoint 存在并且 numgood 大于 0 */
  if (qh->GOODpoint && numgood) {
    /* 再次遍历所有的 facet */
    FORALLfacet_(facetlist) {
      /* 如果 facet 标记为 good 并且 facet 的法向量存在 */
      if (facet->good && facet->normal) {
        /* 增加 Zdistgood 计数 */
        zinc_(Zdistgood);
        /* 计算 qh.GOODpointp 到 facet 的距离 */
        qh_distplane(qh, qh->GOODpointp, facet, &dist);
        /* 根据距离判断是否保持 facet 的 good 标记 */
        if ((qh->GOODpoint > 0) ^ (dist > 0.0)) {
          facet->good = False;
          numgood--;
        }
      }
    }
  }

  /* 如果 qh.GOODthreshold 存在并且（numgood 大于 0 或者 goodhorizon 大于 0 或者 qh.GOODclosest 存在） */
  if (qh->GOODthreshold && (numgood || goodhorizon || qh->GOODclosest)) {
    /* 再次遍历所有的 facet */
    FORALLfacet_(facetlist) {
      /* 如果 facet 标记为 good 并且 facet 的法向量存在 */
      if (facet->good && facet->normal) {
        /* 如果 facet 的法向量不在阈值内 */
        if (!qh_inthresholds(qh, facet->normal, &angle)) {
          facet->good = False;
          numgood--;
          /* 如果角度比最佳角度小，则更新最佳 facet */
          if (angle < bestangle) {
            bestangle = angle;
            bestfacet = facet;
          }
        }
      }
    }

    /* 如果 numgood 为 0 并且（goodhorizon 为 0 或者 qh.GOODclosest 存在） */
    if (numgood == 0 && (goodhorizon == 0 || qh->GOODclosest)) {
      /* 如果 qh.GOODclosest 存在 */
      if (qh->GOODclosest) {
        /* 如果 qh.GOODclosest 是可见的，则置为 NULL */
        if (qh->GOODclosest->visible)
          qh->GOODclosest = NULL;
        else {
          /* 否则计算 qh.GOODclosest 到阈值的角度 */
          qh_inthresholds(qh, qh->GOODclosest->normal, &angle);
          if (angle < bestangle)
            bestfacet = qh->GOODclosest;
        }
      }
      /* 如果 bestfacet 存在且不等于 qh.GOODclosest */
      if (bestfacet && bestfacet != qh->GOODclosest) {   /* numgood == 0 */
        /* 如果 qh.GOODclosest 存在，则将其标记为非 good */
        if (qh->GOODclosest)
          qh->GOODclosest->good = False;
        /* 更新 qh.GOODclosest 为 bestfacet */
        qh->GOODclosest = bestfacet;
        bestfacet->good = True;
        numgood++;
        /* 输出跟踪信息 */
        trace2((qh, qh->ferr, 2044, "qh_findgood: f%d is closest(%2.2g) to thresholds\n",
               bestfacet->id, bestangle));
        return numgood;
      }
    } else if (qh->GOODclosest) { /* numgood > 0 */
      /* 如果 numgood 大于 0，则将 qh.GOODclosest 标记为非 good */
      qh->GOODclosest->good = False;
      qh->GOODclosest = NULL;
    }
  }

  /* 增加 Zgoodfacet 计数 */
  zadd_(Zgoodfacet, numgood);
  /* 输出跟踪信息 */
  trace2((qh, qh->ferr, 2045, "qh_findgood: found %d good facets with %d good horizon and qh.GOODclosest f%d\n",
               numgood, goodhorizon, getid_(qh->GOODclosest)));

  /* 如果没有找到 good facet，并且 qh.GOODvertex 大于 0 并且不处于 MERGING 状态，则返回 goodhorizon */
  if (!numgood && qh->GOODvertex > 0 && !qh->MERGING)
    return goodhorizon;
  /* 否则返回 numgood */
  return numgood;
} /* findgood */
    # 如果qh.ONLYgood为假，则首先调用qh_findgood函数

  design:
    # 使用qh_findgood函数标记好的面（facets）
    # 对于失败的qh.GOODvertex，清除面的f.good标记
    # 对于失败的qh.SPLITthreholds，同样清除面的f.good标记
       # 如果没有更多的好的面（facets），从qh.SPLITthresholds中选择最佳的一个
/* 
   找到所有好的面片并进行处理
   qh：Qhull数据结构
   facetlist：需要处理的面片列表
*/
void qh_findgood_all(qhT *qh, facetT *facetlist) {
  facetT *facet, *bestfacet=NULL;  // 定义面片和最佳面片变量
  realT angle, bestangle= REALmax;  // 角度和最佳角度初始化为最大实数值
  int  numgood=0, startgood;  // 好面片数量和起始好面片数量

  // 如果没有设置好顶点、阈值、点或分裂阈值，直接返回
  if (!qh->GOODvertex && !qh->GOODthreshold && !qh->GOODpoint
  && !qh->SPLITthresholds)
    return;
  
  // 如果不仅查找好的面片，则调用qh_findgood函数
  if (!qh->ONLYgood)
    qh_findgood(qh, qh->facet_list, 0);
  
  // 遍历facetlist中的所有面片，统计good标记的面片数量
  FORALLfacet_(facetlist) {
    if (facet->good)
      numgood++;
  }
  
  // 如果GOODvertex小于0或者（GOODvertex大于0且MERGING为真）
  FORALLfacet_(facetlist) {
    if (facet->good && ((qh->GOODvertex > 0) ^ !!qh_isvertex(qh->GOODvertexp, facet->vertices))) { /* convert to bool */
      if (!--numgood) {
        // 如果只查找好的面片，输出警告信息并返回
        if (qh->ONLYgood) {
          qh_fprintf(qh, qh->ferr, 7064, "qhull warning: good vertex p%d does not match last good facet f%d.  Ignored.\n",
             qh_pointid(qh, qh->GOODvertexp), facet->id);
          return;
        } else if (qh->GOODvertex > 0)
          qh_fprintf(qh, qh->ferr, 7065, "qhull warning: point p%d is not a vertex('QV%d').\n",
              qh->GOODvertex-1, qh->GOODvertex-1);
        else
          qh_fprintf(qh, qh->ferr, 7066, "qhull warning: point p%d is a vertex for every facet('QV-%d').\n",
              -qh->GOODvertex - 1, -qh->GOODvertex - 1);
      }
      facet->good= False;  // 将该面片标记为非好面片
    }
  }
  
  startgood= numgood;  // 记录初始好面片数量
  
  // 如果存在分裂阈值，则进一步处理
  if (qh->SPLITthresholds) {
    FORALLfacet_(facetlist) {
      if (facet->good) {
        // 如果面片不在阈值范围内，则标记为非好面片，并记录最佳面片及其角度
        if (!qh_inthresholds(qh, facet->normal, &angle)) {
          facet->good= False;
          numgood--;
          if (angle < bestangle) {
            bestangle= angle;
            bestfacet= facet;
          }
        }
      }
    }
    // 如果没有好的面片且存在最佳面片，则将其标记为好的面片并输出信息
    if (!numgood && bestfacet) {
      bestfacet->good= True;
      numgood++;
      trace0((qh, qh->ferr, 23, "qh_findgood_all: f%d is closest(%2.2g) to split thresholds\n",
           bestfacet->id, bestangle));
      return;
    }
  }
  
  // 如果只有一个好的面片且不打印好的面片信息且GOODclosest也是好的，则取消GOODclosest的选择
  if (numgood == 1 && !qh->PRINTgood && qh->GOODclosest && qh->GOODclosest->good) {
    trace2((qh, qh->ferr, 2109, "qh_findgood_all: undo selection of qh.GOODclosest f%d since it would fail qh_inthresholds in qh_skipfacet\n",
      qh->GOODclosest->id));
    qh->GOODclosest->good= False;
    numgood= 0;
  }
  
  qh->num_good= numgood;  // 更新Qhull数据结构中好面片的数量
  trace0((qh, qh->ferr, 24, "qh_findgood_all: %d good facets remain out of %d facets\n",
        numgood, startgood));  // 输出剩余的好面片数量
} /* findgood_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="furthestnext">-</a>

  qh_furthestnext()
    set qh.facet_next to facet with furthest of all furthest points
    searches all facets on qh.facet_list

  notes:
    this may help avoid precision problems
*/
void qh_furthestnext(qhT *qh /* qh.facet_list */) {
  facetT *facet, *bestfacet= NULL;  // 定义面片和最佳面片变量
  realT dist, bestdist= -REALmax;  // 距离和最佳距离初始化为最小实数值
  
  // 遍历所有面片
  FORALLfacets {
    if (facet->outsideset) {
#if qh_COMPUTEfurthest
      // 如果定义了 qh_COMPUTEfurthest 宏，则计算最远点
      pointT *furthest;
      // 声明最远点指针变量 furthest
      furthest= (pointT *)qh_setlast(facet->outsideset);
      // 将 facet->outsideset 中的最后一个点作为最远点 furthest
      zinc_(Zcomputefurthest);
      // 增加 Zcomputefurthest 计数器
      qh_distplane(qh, furthest, facet, &dist);
      // 计算最远点 furthest 到平面 facet 的距离，并存储在 dist 中
#else
      // 否则使用预先计算的最远距离 facet->furthestdist
      dist= facet->furthestdist;
      // 将预先计算的最远距离存储在 dist 中
#endif
      if (dist > bestdist) {
        // 如果计算得到的距离 dist 大于当前最佳距离 bestdist
        bestfacet= facet;
        // 更新最佳 facet
        bestdist= dist;
        // 更新最佳距离 bestdist
      }
    }
  }
  if (bestfacet) {
    // 如果找到了最佳 facet
    qh_removefacet(qh, bestfacet);
    // 从 qh 数据结构中移除最佳 facet
    qh_prependfacet(qh, bestfacet, &qh->facet_next);
    // 将最佳 facet 插入到 qh->facet_next 之前
    trace1((qh, qh->ferr, 1029, "qh_furthestnext: made f%d next facet(dist %.2g)\n",
            bestfacet->id, bestdist));
    // 打印跟踪消息，说明哪个 facet 被设置为下一个 facet，以及其距离
  }
} /* furthestnext */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="furthestout">-</a>

  qh_furthestout(qh, facet )
    make furthest outside point the last point of outsideset

  returns:
    updates facet->outsideset
    clears facet->notfurthest
    sets facet->furthestdist

  design:
    determine best point of outsideset
    make it the last point of outsideset
*/
void qh_furthestout(qhT *qh, facetT *facet) {
  // 声明变量
  pointT *point, **pointp, *bestpoint= NULL;
  realT dist, bestdist= -REALmax;

  // 遍历 facet->outsideset 中的每个点
  FOREACHpoint_(facet->outsideset) {
    // 计算当前点到 facet 的距离，并存储在 dist 中
    qh_distplane(qh, point, facet, &dist);
    // 增加 Zcomputefurthest 计数器
    zinc_(Zcomputefurthest);
    // 如果当前点到 facet 的距离大于最佳距离 bestdist
    if (dist > bestdist) {
      // 更新最佳点为当前点
      bestpoint= point;
      // 更新最佳距离为当前距离 dist
      bestdist= dist;
    }
  }
  // 如果找到了最佳点 bestpoint
  if (bestpoint) {
    // 从 outsideset 中删除最佳点
    qh_setdel(facet->outsideset, point);
    // 将最佳点添加到 outsideset 的末尾
    qh_setappend(qh, &facet->outsideset, point);
#if !qh_COMPUTEfurthest
    // 如果没有定义 qh_COMPUTEfurthest 宏，则更新 facet 的最远距离
    facet->furthestdist= bestdist;
#endif
  }
  // 标记 facet 的 notfurthest 属性为 False，表示不是最远点
  facet->notfurthest= False;
  // 打印跟踪消息，说明哪个点被标记为 facet 的最远点
  trace3((qh, qh->ferr, 3017, "qh_furthestout: p%d is furthest outside point of f%d\n",
          qh_pointid(qh, point), facet->id));
} /* furthestout */


/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="infiniteloop">-</a>

  qh_infiniteloop(qh, facet )
    report infinite loop error due to facet
*/
void qh_infiniteloop(qhT *qh, facetT *facet) {
  // 输出错误消息，提示潜在的无限循环
  qh_fprintf(qh, qh->ferr, 6149, "qhull internal error (qh_infiniteloop): potential infinite loop detected.  If visible, f.replace. If newfacet, f.samecycle\n");
  // 引发 qhull 错误并退出
  qh_errexit(qh, qh_ERRqhull, facet, NULL);
} /* qh_infiniteloop */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initbuild">-</a>

  qh_initbuild()
    initialize hull and outside sets with point array
    qh.FIRSTpoint/qh.NUMpoints is point array
    if qh.GOODpoint
      adds qh.GOODpoint to initial hull

  returns:
    qh_facetlist with initial hull
    points partioned into outside sets, coplanar sets, or inside
    initializes qh.GOODpointp, qh.GOODvertexp,

  design:
    initialize global variables used during qh_buildhull
    determine precision constants and points with max/min coordinate values
      if qh.SCALElast, scale last coordinate(for 'd')
    initialize qh.newfacet_list, qh.facet_tail
    initialize qh.vertex_list, qh.newvertex_list, qh.vertex_tail
    determine initial vertices
    build initial simplex
    # 将输入点分区成初始简单形体的面（facets）
    partition input points into facets of initial simplex
    
    # 初始化列表
    set up lists
    
    # 如果仅处理良好的情况
    if qh.ONLYgood:
        # 检查一致性
        check consistency
        # 如果定义了 qh.GOODvertex，则添加它
        add qh.GOODvertex if defined
/*
void qh_initbuild(qhT *qh) {
    // 定义局部变量 maxpoints 和 vertices
    setT *maxpoints, *vertices;
    // 定义指向 facetT 结构的指针 facet
    facetT *facet;
    // 定义整型变量 i 和 numpart
    int i, numpart;
    // 定义实数变量 dist
    realT dist;
    // 定义布尔变量 isoutside
    boolT isoutside;

    // 如果设置了 PRINTstatistics 标志位，则输出 Qhull 统计信息到 ferr 流
    if (qh->PRINTstatistics) {
        qh_fprintf(qh, qh->ferr, 9350, "qhull %s Statistics: %s | %s\n",
                   qh_version, qh->rbox_command, qh->qhull_command);
        fflush(NULL);
    }
    // 初始化 qhull 中的各种 ID 和计数器
    qh->furthest_id= qh_IDunknown;
    qh->lastreport= 0;
    qh->lastfacets= 0;
    qh->lastmerges= 0;
    qh->lastplanes= 0;
    qh->lastdist= 0;
    qh->facet_id= qh->vertex_id= qh->ridge_id= 0;
    qh->visit_id= qh->vertex_visit= 0;
    qh->maxoutdone= False;

    // 根据 GOODpoint 的值初始化 GOODpointp
    if (qh->GOODpoint > 0)
        qh->GOODpointp= qh_point(qh, qh->GOODpoint-1);
    else if (qh->GOODpoint < 0)
        qh->GOODpointp= qh_point(qh, -qh->GOODpoint-1);
    
    // 根据 GOODvertex 的值初始化 GOODvertexp
    if (qh->GOODvertex > 0)
        qh->GOODvertexp= qh_point(qh, qh->GOODvertex-1);
    else if (qh->GOODvertex < 0)
        qh->GOODvertexp= qh_point(qh, -qh->GOODvertex-1);
    
    // 检查 GOODpointp 和 GOODvertexp 的有效性，如果不在有效范围内则报错退出
    if ((qh->GOODpoint
         && (qh->GOODpointp < qh->first_point  /* also catches !GOODpointp */
             || qh->GOODpointp > qh_point(qh, qh->num_points-1)))
    || (qh->GOODvertex
         && (qh->GOODvertexp < qh->first_point  /* also catches !GOODvertexp */
             || qh->GOODvertexp > qh_point(qh, qh->num_points-1)))) {
        qh_fprintf(qh, qh->ferr, 6150, "qhull input error: either QGn or QVn point is > p%d\n",
                   qh->num_points-1);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    
    // 计算从 first_point 开始的 num_points 个点中的最大点集合
    maxpoints= qh_maxmin(qh, qh->first_point, qh->num_points, qh->hull_dim);
    
    // 如果设置了 SCALElast 标志位，则对最后一个坐标进行缩放
    if (qh->SCALElast)
        qh_scalelast(qh, qh->first_point, qh->num_points, qh->hull_dim, qh->MINlastcoord, qh->MAXlastcoord, qh->MAXabs_coord);
    
    // 设置舍入误差
    qh_detroundoff(qh);
    
    // 如果设置了 DELAUNAY 标志位，并且上界和下界的阈值满足条件，则进行特定处理
    if (qh->DELAUNAY && qh->upper_threshold[qh->hull_dim-1] > REALmax/2
                    && qh->lower_threshold[qh->hull_dim-1] < -REALmax/2) {
        // 遍历 PRINTout 数组
        for (i=qh_PRINTEND; i--; ) {
            // 检查是否应该设置 upper_threshold，如果条件满足则退出循环
            if (qh->PRINTout[i] == qh_PRINTgeom && qh->DROPdim < 0
                && !qh->GOODthreshold && !qh->SPLITthresholds)
                break;  /* in this case, don't set upper_threshold */
        }
        // 如果未找到满足条件的 i，则根据 UPPERdelaunay 设置 lower_threshold 或者 upper_threshold
        if (i < 0) {
            if (qh->UPPERdelaunay) { /* matches qh.upperdelaunay in qh_setfacetplane */
                qh->lower_threshold[qh->hull_dim-1]= qh->ANGLEround * qh_ZEROdelaunay;
                qh->GOODthreshold= True;
            } else {
                qh->upper_threshold[qh->hull_dim-1]= -qh->ANGLEround * qh_ZEROdelaunay;
                if (!qh->GOODthreshold)
                    qh->SPLITthresholds= True; /* build upper-convex hull even if Qg */
                    /* qh_initqhull_globals errors if Qg without Pdk/etc. */
            }
        }
  }
  }
  trace4((qh, qh->ferr, 4091, "qh_initbuild: create sentinels for qh.facet_tail and qh.vertex_tail\n"));
  # 调试输出函数，打印初始化建立过程的信息，包括创建 qh.facet_tail 和 qh.vertex_tail 的哨兵
  qh->facet_list= qh->newfacet_list= qh->facet_tail= qh_newfacet(qh);
  # 初始化 qh.facet_list、qh.newfacet_list、qh.facet_tail，并创建新的 facet 对象并返回
  qh->num_facets= qh->num_vertices= qh->num_visible= 0;
  # 初始化顶点、面和可见面的数量
  qh->vertex_list= qh->newvertex_list= qh->vertex_tail= qh_newvertex(qh, NULL);
  # 初始化 qh.vertex_list、qh.newvertex_list、qh.vertex_tail，并创建新的 vertex 对象并返回
  vertices= qh_initialvertices(qh, qh->hull_dim, maxpoints, qh->first_point, qh->num_points);
  # 根据输入的参数初始化顶点集合
  qh_initialhull(qh, vertices);  /* initial qh->facet_list */
  # 初始化凸壳，使用 vertices 初始化 qh->facet_list
  qh_partitionall(qh, vertices, qh->first_point, qh->num_points);
  # 对所有顶点进行分区处理

  if (qh->PRINToptions1st || qh->TRACElevel || qh->IStracing) {
    if (qh->TRACElevel || qh->IStracing)
      qh_fprintf(qh, qh->ferr, 8103, "\nTrace level T%d, IStracing %d, point TP%d, merge TM%d, dist TW%2.2g, qh.tracefacet_id %d, traceridge_id %d, tracevertex_id %d, last qh.RERUN %d, %s | %s\n",
         qh->TRACElevel, qh->IStracing, qh->TRACEpoint, qh->TRACEmerge, qh->TRACEdist, qh->tracefacet_id, qh->traceridge_id, qh->tracevertex_id, qh->TRACElastrun, qh->rbox_command, qh->qhull_command);
    # 如果启用了追踪等级或者正在进行追踪，则输出追踪信息
    qh_fprintf(qh, qh->ferr, 8104, "Options selected for Qhull %s:\n%s\n", qh_version, qh->qhull_options);
    # 输出 Qhull 选项信息
  }

  qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
  # 重置列表，包括 qh.visible_list、newvertex_list 和 qh.newfacet_list
  qh->facet_next= qh->facet_list;
  # 设置下一个操作的面为 qh.facet_list 中的第一个面
  qh_furthestnext(qh /* qh.facet_list */);
  # 设置下一个最远面
  if (qh->PREmerge) {
    qh->cos_max= qh->premerge_cos;
    qh->centrum_radius= qh->premerge_centrum; /* overwritten by qh_premerge */
    # 设置 cos_max 和 centrum_radius 为预合并值
  }
  if (qh->ONLYgood) {
    if (qh->GOODvertex > 0 && qh->MERGING) {
      qh_fprintf(qh, qh->ferr, 6151, "qhull input error: 'Qg QVn' (only good vertex) does not work with merging.\nUse 'QJ' to joggle the input or 'Q0' to turn off merging.\n");
      # 如果仅使用好的顶点，并且启用了合并，则输出错误信息
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
      # 退出程序，报告输入错误
    }
    if (!(qh->GOODthreshold || qh->GOODpoint
         || (!qh->MERGEexact && !qh->PREmerge && qh->GOODvertexp))) {
      qh_fprintf(qh, qh->ferr, 6152, "qhull input error: 'Qg' (ONLYgood) needs a good threshold('Pd0D0'), a good point(QGn or QG-n), or a good vertex with 'QJ' or 'Q0' (QVn).\n");
      # 如果仅使用好的顶点，但是没有指定好的阈值、点或者顶点，则输出错误信息
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
      # 退出程序，报告输入错误
    }
    if (qh->GOODvertex > 0  && !qh->MERGING  /* matches qh_partitionall */
    && !qh_isvertex(qh->GOODvertexp, vertices)) {
      # 如果指定了好的顶点数目，且没有启用合并，并且 GOODvertexp 不是顶点集合中的一个顶点
      facet= qh_findbestnew(qh, qh->GOODvertexp, qh->facet_list,
                          &dist, !qh_ALL, &isoutside, &numpart);
      # 在 qh.facet_list 中查找最佳新面
      zadd_(Zdistgood, numpart);
      # 记录 zadd_ 信息
      if (!isoutside) {
        qh_fprintf(qh, qh->ferr, 6153, "qhull input error: point for QV%d is inside initial simplex.  It can not be made a vertex.\n",
               qh_pointid(qh, qh->GOODvertexp));
        # 如果 GOODvertexp 在初始单纯形内部，则输出错误信息
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
        # 退出程序，报告输入错误
      }
      if (!qh_addpoint(qh, qh->GOODvertexp, facet, False)) {
        # 如果无法添加 GOODvertexp 作为顶点
        qh_settempfree(qh, &vertices);
        # 释放 vertices
        qh_settempfree(qh, &maxpoints);
        # 释放 maxpoints
        return;
        # 返回
      }
    }
  }
    qh_findgood(qh, qh->facet_list, 0);

# 调用qh_findgood函数，用于初始化几何算法的进一步处理，传入qh结构体指针和facet_list指针，参数0可能指示特定的处理方式或选项。


  }

# 结束if语句块，没有else分支或其他条件执行。


  qh_settempfree(qh, &vertices);

# 调用qh_settempfree函数，释放vertices临时分配的内存，qh是qh结构体指针，&vertices是vertices变量的地址。


  qh_settempfree(qh, &maxpoints);

# 调用qh_settempfree函数，释放maxpoints临时分配的内存，qh是qh结构体指针，&maxpoints是maxpoints变量的地址。


  trace1((qh, qh->ferr, 1030, "qh_initbuild: initial hull created and points partitioned\n"));

# 使用trace1函数输出跟踪信息，记录qh_initbuild函数的操作，包括创建初始凸壳并分区点集。qh是qh结构体指针，qh->ferr是错误输出流，1030是消息代码，"qh_initbuild: initial hull created and points partitioned\n"是描述性消息字符串。
} /* initbuild */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initialhull">-</a>

  qh_initialhull(qh, vertices )
    constructs the initial hull as a DIM3 simplex of vertices

  notes:
    only called by qh_initbuild

  design:
    creates a simplex (initializes lists)
    determines orientation of simplex
    sets hyperplanes for facets
    doubles checks orientation (in case of axis-parallel facets with Gaussian elimination)
    checks for flipped facets and qh.NARROWhull
    checks the result
*/
void qh_initialhull(qhT *qh, setT *vertices) {
  facetT *facet, *firstfacet, *neighbor, **neighborp;
  realT angle, minangle= REALmax, dist;

  qh_createsimplex(qh, vertices /* qh.facet_list */);  // 创建一个初始的凸包（简单形式），使用给定的顶点集合
  qh_resetlists(qh, False, qh_RESETvisible);           // 重置凸包中的列表
  qh->facet_next= qh->facet_list;      /* advance facet when processed */  // 设置下一个要处理的面为列表中的第一个面
  qh->interior_point= qh_getcenter(qh, vertices);       // 计算凸包的内部点，用于后续的定向检查
  if (qh->IStracing) {
    qh_fprintf(qh, qh->ferr, 8105, "qh_initialhull: ");
    qh_printpoint(qh, qh->ferr, "qh.interior_point", qh->interior_point);  // 如果启用追踪，打印内部点的信息
  }
  firstfacet= qh->facet_list;
  qh_setfacetplane(qh, firstfacet);   /* qh_joggle_restart if flipped */   // 设置第一个面的超平面，并检查是否翻转
  if (firstfacet->flipped) {
    trace1((qh, qh->ferr, 1065, "qh_initialhull: ignore f%d flipped.  Test qh.interior_point (p-2) for clearly flipped\n", firstfacet->id));
    firstfacet->flipped= False;       // 如果第一个面被标记为翻转，取消标记以避免错误处理
  }
  zzinc_(Zdistcheck);
  qh_distplane(qh, qh->interior_point, firstfacet, &dist);  // 计算内部点到第一个面的距离，用于确定初始方向是否正确
  if (dist > qh->DISTround) {  /* clearly flipped */
    trace1((qh, qh->ferr, 1060, "qh_initialhull: initial orientation incorrect, qh.interior_point is %2.2g from f%d.  Reversing orientation of all facets\n",
          dist, firstfacet->id));
    FORALLfacets
      facet->toporient ^= (unsigned char)True;   // 如果初始方向明显错误，翻转所有面的方向
    qh_setfacetplane(qh, firstfacet);            // 重新设置第一个面的超平面
  }
  FORALLfacets {
    if (facet != firstfacet)
      qh_setfacetplane(qh, facet);    /* qh_joggle_restart if flipped */  // 设置其余每个面的超平面，并检查是否翻转
  }
  FORALLfacets {
    if (facet->flipped) {
      trace1((qh, qh->ferr, 1066, "qh_initialhull: ignore f%d flipped.  Test qh.interior_point (p-2) for clearly flipped\n", facet->id));
      facet->flipped= False;      // 忽略任何被标记为翻转的面，以避免错误处理
    }
    zzinc_(Zdistcheck);
    qh_distplane(qh, qh->interior_point, facet, &dist);  // 再次计算内部点到每个面的距离，用于确认初始方向是否正确
    if (dist > qh->DISTround) {  /* clearly flipped, due to axis-parallel facet or coplanar firstfacet */
      trace1((qh, qh->ferr, 1031, "qh_initialhull: initial orientation incorrect, qh.interior_point is %2.2g from f%d.  Either axis-parallel facet or coplanar firstfacet f%d.  Force outside orientation of all facets\n"));
      FORALLfacets { /* reuse facet, then 'break' */
        facet->flipped= False;
        facet->toporient ^= (unsigned char)True;   // 如果初始方向明显错误，强制翻转所有面的方向
        qh_orientoutside(qh, facet);  /* force outside orientation for f.normal */  // 强制设置面的外部定向
      }
      break;
    }
  }
  FORALLfacets {
    // 检查是否需要翻转当前面（facet），根据情况处理初步的 Delaunay 三角剖分或 Voronoi 图问题
    if (!qh_checkflipped(qh, facet, NULL, qh_ALL)) {
      // 如果处于 Delaunay 模式且非无限点状态，尝试重新启动 joggle 过程，处理初始的 Delaunay 三角剖分的共圆或共球问题
      if (qh->DELAUNAY && !qh->ATinfinity) {
        qh_joggle_restart(qh, "initial Delaunay cocircular or cospherical");
        // 如果开启了上限 Delaunay 选项，报告精度错误，说明初始的 Delaunay 输入点是共圆或共球的
        if (qh->UPPERdelaunay)
          qh_fprintf(qh, qh->ferr, 6240, "Qhull precision error: initial Delaunay input sites are cocircular or cospherical.  Option 'Qs' searches all points.  Use option 'QJ' to joggle the input, otherwise cannot compute the upper Delaunay triangulation or upper Voronoi diagram of cocircular/cospherical points.\n");
        else
          qh_fprintf(qh, qh->ferr, 6239, "Qhull precision error: initial Delaunay input sites are cocircular or cospherical.  Use option 'Qz' for the Delaunay triangulation or Voronoi diagram of cocircular/cospherical points; it adds a point \"at infinity\".  Alternatively use option 'QJ' to joggle the input.  Use option 'Qs' to search all points for the initial simplex.\n");
        // 打印带有投影到抛物面上的最后坐标的输入点列表
        qh_printvertexlist(qh, qh->ferr, "\ninput sites with last coordinate projected to a paraboloid\n", qh->facet_list, NULL, qh_ALL);
        // 退出程序，因输入错误导致
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }else {
        // 否则，重新启动 joggle 过程，处理初始的简单形式为平坦的情况
        qh_joggle_restart(qh, "initial simplex is flat");
        // 报告精度错误，说明初始的简单形式为平坦的，即面与内部点共面
        qh_fprintf(qh, qh->ferr, 6154, "Qhull precision error: Initial simplex is flat (facet %d is coplanar with the interior point)\n",
                   facet->id);
        // 退出程序，因输入错误导致，调用 qh_printhelp_singular 函数
        qh_errexit(qh, qh_ERRsingular, NULL, NULL);  /* calls qh_printhelp_singular */
      }
    }
    // 对于每个面（facet），计算其与相邻面之间的最小角度
    FOREACHneighbor_(facet) {
      angle= qh_getangle(qh, facet->normal, neighbor->normal);
      // 最小化计算得到的角度值
      minimize_( minangle, angle);
    }
  }
  // 如果最小角度小于设定的最大窄角度且非禁用窄角度处理，则处理窄角度问题
  if (minangle < qh_MAXnarrow && !qh->NOnarrow) {
    realT diff= 1.0 + minangle;

    // 标记当前计算为窄角度的凸壳问题
    qh->NARROWhull= True;
    // 设定窄角度选项，调整处理精度
    qh_option(qh, "_narrow-hull", NULL, &diff);
    // 如果最小角度低于警告级别的窄角度且未设置重新运行标志且允许打印精度信息，则打印窄角度问题的帮助信息
    if (minangle < qh_WARNnarrow && !qh->RERUN && qh->PRINTprecision)
      qh_printhelp_narrowhull(qh, qh->ferr, minangle);
  }
  // 设定处理过的顶点数目为当前凸壳维度加一
  zzval_(Zprocessed)= qh->hull_dim+1;
  // 检查并修正当前凸壳的多边形
  qh_checkpolygon(qh, qh->facet_list);
  // 检查并修正当前凸壳的凸性
  qh_checkconvex(qh, qh->facet_list, qh_DATAfault);
  // 如果跟踪级别大于等于 1，输出初始凸壳已构建完成的信息
  if (qh->IStracing >= 1) {
    qh_fprintf(qh, qh->ferr, 8105, "qh_initialhull: simplex constructed\n");
  }
} /* initialhull */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initialvertices">-</a>

  qh_initialvertices(qh, dim, maxpoints, points, numpoints )
    determines a non-singular set of initial vertices
    maxpoints may include duplicate points

  returns:
    temporary set of dim+1 vertices in descending order by vertex id
    if qh.RANDOMoutside && !qh.ALLpoints
      picks random points
    if dim >= qh_INITIALmax,
      uses min/max x and max points with non-zero determinants

  notes:
    unless qh.ALLpoints,
      uses maxpoints as long as determinate is non-zero
*/
setT *qh_initialvertices(qhT *qh, int dim, setT *maxpoints, pointT *points, numpoints) {
  pointT *point, **pointp;  /* 定义指向点的指针和点数组指针 */

  setT *vertices, *simplex, *tested;  /* 定义顶点集合、简单形式集合和测试过的点集合 */

  realT randr;  /* 实数类型的随机数 */

  int idx, point_i, point_n, k;  /* 索引、点下标、点个数、计数变量 */

  boolT nearzero= False;  /* 布尔值，表示是否接近零 */

  vertices= qh_settemp(qh, dim + 1);  /* 创建包含 dim+1 个顶点的临时集合 */
  simplex= qh_settemp(qh, dim + 1);   /* 创建包含 dim+1 个顶点的简单形式集合 */

  if (qh->ALLpoints)  /* 如果使用所有点作为初始顶点 */
    qh_maxsimplex(qh, dim, NULL, points, numpoints, &simplex);
  else if (qh->RANDOMoutside) {  /* 如果使用随机点作为初始顶点 */
    while (qh_setsize(qh, simplex) != dim+1) {  /* 确保简单形式集合包含 dim+1 个点 */
      randr= qh_RANDOMint;  /* 获取随机整数 */
      randr= randr/(qh_RANDOMmax+1);  /* 将随机整数归一化为 [0, 1) */
      randr= floor(qh->num_points * randr);  /* 缩放到点的数量范围内 */
      idx= (int)randr;  /* 转换为整数索引 */
      while (qh_setin(simplex, qh_point(qh, idx))) {
        idx++; /* 如果随机数总是返回相同的值，增加索引以找到未包含的点 */
        idx= idx < qh->num_points ? idx : 0;
      }
      qh_setappend(qh, &simplex, qh_point(qh, idx));  /* 添加未包含的随机点 */
    }
  } else if (qh->hull_dim >= qh_INITIALmax) {  /* 如果维度大于等于初始最大维度 */
    tested= qh_settemp(qh, dim+1);  /* 创建测试过的点集合 */
    qh_setappend(qh, &simplex, SETfirst_(maxpoints));   /* 添加最大和最小 X 坐标点 */
    qh_setappend(qh, &simplex, SETsecond_(maxpoints));
    qh_maxsimplex(qh, fmin_(qh_INITIALsearch, dim), maxpoints, points, numpoints, &simplex);  /* 添加搜索到的最大简单形式 */
    k= qh_setsize(qh, simplex);  /* 获取简单形式集合的大小 */
    FOREACHpoint_i_(qh, maxpoints) {  /* 遍历所有最大点集合的点 */
      if (k >= dim)  /* 如果简单形式的大小已经满足维度要求 */
        break;
      if (point_i & 0x1) {     /* 首先尝试最大坐标点 */
        if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
          qh_detsimplex(qh, point, simplex, k, &nearzero);  /* 计算简单形式的行列式 */
          if (nearzero)
            qh_setappend(qh, &tested, point);  /* 如果接近零，将点添加到测试集合 */
          else {
            qh_setappend(qh, &simplex, point);  /* 否则将点添加到简单形式集合 */
            k++;
          }
        }
      }
    }
    FOREACHpoint_i_(qh, maxpoints) {  /* 继续遍历最大点集合的点 */
      if (k >= dim)  /* 如果简单形式的大小已经满足维度要求 */
        break;
      if ((point_i & 0x1) == 0) {  /* 然后尝试最小坐标点 */
        if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
          qh_detsimplex(qh, point, simplex, k, &nearzero);  /* 计算简单形式的行列式 */
          if (nearzero)
            qh_setappend(qh, &tested, point);  /* 如果接近零，将点添加到测试集合 */
          else {
            qh_setappend(qh, &simplex, point);  /* 否则将点添加到简单形式集合 */
            k++;
          }
        }
      }
    }
    /* 从最大点集合中移除测试过的点 */
    FOREACHpoint_i_(qh, maxpoints) {
      if (qh_setin(simplex, point) || qh_setin(tested, point))
        SETelem_(maxpoints, point_i)= NULL;
    }
  }
  # 调用 qh_setcompact 函数，使几何处理库中的几何对象更紧凑
  qh_setcompact(qh, maxpoints);
  # 将索引 idx 设为 0
  idx= 0;
  # 当 k 小于维度且 qh_point 返回非空时，执行循环
  while (k < dim && (point= qh_point(qh, idx++))) {
    # 如果点不在简单形成中并且不在已测试集合中
    if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
      # 判断是否为接近零点
      qh_detsimplex(qh, point, simplex, k, &nearzero);
      # 如果不接近零
      if (!nearzero){
        # 将点添加到简单形成中
        qh_setappend(qh, &simplex, point);
        # 增加 k 的计数
        k++;
      }
    }
  }
  # 释放 tested 集合的临时内存
  qh_settempfree(qh, &tested);
  # 调用 qh_maxsimplex 函数，计算几何处理库中的最大简单形成
  qh_maxsimplex(qh, dim, maxpoints, points, numpoints, &simplex);
}else /* qh.hull_dim < qh_INITIALmax */
  # 调用 qh_maxsimplex 函数，计算几何处理库中的最大简单形成
  qh_maxsimplex(qh, dim, maxpoints, points, numpoints, &simplex);
# 对于简单形成中的每个点
FOREACHpoint_(simplex)
  # 将新创建的顶点添加到顶点集合中，按降序排列
  qh_setaddnth(qh, &vertices, 0, qh_newvertex(qh, point)); /* descending order */
# 释放 simplex 集合的临时内存
qh_settempfree(qh, &simplex);
# 返回顶点集合
return vertices;
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="isvertex">-</a>

  qh_isvertex( point, vertices )
    returns vertex if point is in vertex set, else returns NULL

  notes:
    for qh.GOODvertex
*/
vertexT *qh_isvertex(pointT *point, setT *vertices) {
  vertexT *vertex, **vertexp;

  FOREACHvertex_(vertices) {  // 遍历顶点集合中的每个顶点
    if (vertex->point == point)  // 如果顶点的坐标与给定点坐标相同
      return vertex;  // 返回该顶点
  }
  return NULL;  // 如果未找到匹配的顶点，则返回空指针
} /* isvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenewfacets">-</a>

  qh_makenewfacets(qh, point )
    make new facets from point and qh.visible_list

  returns:
    apex (point) of the new facets
    qh.newfacet_list= list of new facets with hyperplanes and ->newfacet
    qh.newvertex_list= list of vertices in new facets with ->newfacet set

    if (qh.NEWtentative)
      newfacets reference horizon facets, but not vice versa
      ridges reference non-simplicial horizon ridges, but not vice versa
      does not change existing facets
    else
      sets qh.NEWfacets
      new facets attached to horizon facets and ridges
      for visible facets,
        visible->r.replace is corresponding new facet

  see also:
    qh_makenewplanes() -- make hyperplanes for facets
    qh_attachnewfacets() -- attachnewfacets if not done here qh->NEWtentative
    qh_matchnewfacets() -- match up neighbors
    qh_update_vertexneighbors() -- update vertex neighbors and delvertices
    qh_deletevisible() -- delete visible facets
    qh_checkpolygon() --check the result
    qh_triangulate() -- triangulate a non-simplicial facet

  design:
    for each visible facet
      make new facets to its horizon facets
      update its f.replace
      clear its neighbor set
*/
vertexT *qh_makenewfacets(qhT *qh, pointT *point /* qh.visible_list */) {
  facetT *visible, *newfacet= NULL, *newfacet2= NULL, *neighbor, **neighborp;
  vertexT *apex;
  int numnew=0;

  if (qh->CHECKfrequently) {
    qh_checkdelridge(qh);  // 如果频繁检查，则检查是否需要删除边缘
  }
  qh->newfacet_list= qh->facet_tail;  // 设置新面列表的起始位置
  qh->newvertex_list= qh->vertex_tail;  // 设置新顶点列表的起始位置
  apex= qh_newvertex(qh, point);  // 创建一个新顶点并返回
  qh_appendvertex(qh, apex);  // 将新顶点添加到顶点列表中
  qh->visit_id++;  // 增加访问标识号

  // 对于每个可见面
  FORALLvisible_facets {
    FOREACHneighbor_(visible)
      neighbor->seen= False;  // 标记邻居面为未见过
    if (visible->ridges) {
      visible->visitid= qh->visit_id;  // 设置访问标识号
      newfacet2= qh_makenew_nonsimplicial(qh, visible, apex, &numnew);  // 创建非简单形式的新面
    }
    if (visible->simplicial)
      newfacet= qh_makenew_simplicial(qh, visible, apex, &numnew);  // 创建简单形式的新面
    if (!qh->NEWtentative) {
      if (newfacet2)  // 如果存在 newfacet2（如果所有边缘都已定义，则 newfacet 为空）
        newfacet= newfacet2;
      if (newfacet)
        visible->f.replace= newfacet;  // 设置 visible 面的替代面
      else
        zinc_(Zinsidevisible);  // 增加内部可见面计数
      if (visible->ridges)      // 对于可见面，边缘和邻居不再有效
        SETfirst_(visible->ridges)= NULL;
      SETfirst_(visible->neighbors)= NULL;
    }
  }
  if (!qh->NEWtentative)
    # 设置 qh 结构体中的 NEWfacets 标志为 True，表示创建了新的 facets
    qh->NEWfacets= True;
  # 如果启用了追踪（tracing），则输出日志，描述创建了多少个新的 facets，它们的编号范围以及与指定点相关的信息
  trace1((qh, qh->ferr, 1032, "qh_makenewfacets: created %d new facets f%d..f%d from point p%d to horizon\n",
    numnew, qh->first_newfacet, qh->facet_id-1, qh_pointid(qh, point)));
  # 如果追踪级别高于等于 4，打印新创建的 facets 列表
  if (qh->IStracing >= 4)
    qh_printfacetlist(qh, qh->newfacet_list, NULL, qh_ALL);
  # 返回 apex 变量，这是一个函数的返回值
  return apex;
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchdupridge">-</a>

  qh_matchdupridge(qh, atfacet, atskip, hashsize, hashcount )
    match duplicate ridges in qh.hash_table for atfacet@atskip
    duplicates marked with ->dupridge and qh_DUPLICATEridge

  returns:
    vertex-facet distance (>0.0) for qh_MERGEridge ridge
    updates hashcount
    set newfacet, facet, matchfacet's hyperplane (removes from mergecycle of coplanarhorizon facets)

  see also:
    qh_matchneighbor

  notes:
    only called by qh_matchnewfacets for qh_buildcone and qh_triangulate_facet
    assumes atfacet is simplicial
    assumes atfacet->neighbors @ atskip == qh_DUPLICATEridge
    usually keeps ridge with the widest merge
    both MRGdupridge and MRGflipped are required merges -- rbox 100 C1,2e-13 D4 t1 | qhull d Qbb
      can merge flipped f11842 skip 3 into f11862 skip 2 and vice versa (forced by goodmatch/goodmatch2)
         blocks -- cannot merge f11862 skip 2 and f11863 skip2 (the widest merge)
         must block -- can merge f11843 skip 3 into f11842 flipped skip 3, but not vice versa
      can merge f11843 skip 3 into f11863 skip 2, but not vice versa
    working/unused.h: [jan'19] Dropped qh_matchdupridge_coplanarhorizon, it was the same or slightly worse.  Complex addition, rarely occurs

  design:
    compute hash value for atfacet and atskip
    repeat twice -- once to make best matches, once to match the rest
      for each possible facet in qh.hash_table
        if it is a matching facet with the same orientation and pass 2
          make match
          unless tricoplanar, mark match for merging (qh_MERGEridge)
          [e.g., tricoplanar RBOX s 1000 t993602376 | QHULL C-1e-3 d Qbb FA Qt]
        if it is a matching facet with the same orientation and pass 1
          test if this is a better match
      if pass 1,
        make best match (it will not be merged)
        set newfacet, facet, matchfacet's hyperplane (removes from mergecycle of coplanarhorizon facets)

*/
/* 计算重复尖脊的匹配情况 */

coordT qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
  boolT same, ismatch, isduplicate= False;
  int hash, scan;
  facetT *facet, *newfacet, *nextfacet;
  facetT *maxmatch= NULL, *maxmatch2= NULL, *goodmatch= NULL, *goodmatch2= NULL;
  int skip, newskip, nextskip= 0, makematch;
  int maxskip= 0, maxskip2= 0, goodskip= 0, goodskip2= 0;
  coordT maxdist= -REALmax, maxdist2= 0.0, dupdist, dupdist2, low, high, maxgood, gooddist= 0.0;

  // 计算最大的可接受距离
  maxgood= qh_WIDEdupridge * (qh->ONEmerge + qh->DISTround); 

  // 获取当前尖脊的哈希值
  hash= qh_gethash(qh, hashsize, atfacet->vertices, qh->hull_dim, 1,
                     SETelem_(atfacet->vertices, atskip));

  // 输出调试信息，显示当前尖脊的哈希匹配情况
  trace2((qh, qh->ferr, 2046, "qh_matchdupridge: find dupridge matches for f%d skip %d hash %d hashcount %d\n",
          atfacet->id, atskip, hash, *hashcount));

  // 遍历两次以获取所有的尖脊匹配
  for (makematch=0; makematch < 2; makematch++) { /* makematch is false on the first pass and 1 on the second */
    qh->visit_id++;
  } /* end of foreach newfacet at 'hash' */

  // 如果是第一次遍历
  if (!makematch) {
    // 如果没有找到最大匹配和优良匹配
    if (!maxmatch && !goodmatch) {
      // 输出错误信息并退出程序
      qh_fprintf(qh, qh->ferr, 6157, "qhull internal error (qh_matchdupridge): no maximum or good match for dupridge new f%d skip %d at hash %d..%d\n",
          atfacet->id, atskip, hash, scan);
      qh_errexit(qh, qh_ERRqhull, atfacet, NULL);
    }
    
    // 如果有优良匹配
    if (goodmatch) {
      // 更新优良匹配的邻居信息
      SETelem_(goodmatch->neighbors, goodskip)= goodmatch2;
      SETelem_(goodmatch2->neighbors, goodskip2)= goodmatch;
      *hashcount -= 2; /* 移除两个未匹配的尖脊 */
      
      // 如果一个尖脊已翻转而另一个未翻转，增加跟踪计数
      if (goodmatch->flipped) {
        if (!goodmatch2->flipped) {
          zzinc_(Zflipridge);
        }else {
          zzinc_(Zflipridge2);
          /* 如果 qh_DUPLICATEridge，则由 qh_matchneighbor 调用 qh_joggle_restart */
        }
      }
      /* 之前已经追踪过 */
    } else {
      // 更新最大匹配的邻居信息
      SETelem_(maxmatch->neighbors, maxskip)= maxmatch2; /* maxmatch!=NULL by QH6157 */
      SETelem_(maxmatch2->neighbors, maxskip2)= maxmatch;
      *hashcount -= 2; /* 移除两个未匹配的尖脊 */
      zzinc_(Zmultiridge);
      
      // 如果 qh_DUPLICATEridge，则由 qh_matchneighbor 调用 qh_joggle_restart
      trace0((qh, qh->ferr, 25, "qh_matchdupridge: keep dupridge f%d skip %d and f%d skip %d, dist %4.4g\n",
          maxmatch2->id, maxskip2, maxmatch->id, maxskip, maxdist));
    }
  }
  
  // 如果有优良匹配，则返回优良匹配的距离
  if (goodmatch)
    return gooddist;
  
  // 否则返回第二最大距离的距离
  return maxdist2;
} /* matchdupridge */

#else /* qh_NOmerge */

// 如果定义了 qh_NOmerge，则返回默认值 0.0
coordT qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(atfacet)
  QHULL_UNUSED(atskip)
  QHULL_UNUSED(hashsize)
  QHULL_UNUSED(hashcount)

  return 0.0;
}
#endif /* qh_NOmerge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nearcoplanar">-</a>

  qh_nearcoplanar()
    for all facets, remove near-inside points from facet->coplanarset</li>
    # 定义由 qh_outerinner() 返回的内部平面上的共面点集合

  # 返回：
  # 如果 qh->KEEPcoplanar 为真且 qh->KEEPinside 为假，则
  # facet->coplanarset 只包含共面点
  # 如果 qh.JOGGLEmax 为真，则
  # 由于顶点可能移出，而共面点可能移入，再次减少内部平面 qh.JOGGLEmax 对角线距离

  # 注：
  # 用于 qh.PREmerge 和 qh.JOGGLEmax
  # 必须与 qh_detroundoff 中对 qh.NEARcoplanar 计算一致

  # 设计：
  # 如果不保留共面点或内部点
  #   释放所有共面点集合
  # 否则，如果不同时保留共面点和内部点
  #   从共面点集合中移除非共面或非内部点
/*
void qh_nearcoplanar(qhT *qh /* qh.facet_list */) {
  facetT *facet;  // 声明一个变量facet，用于遍历facet列表中的每个面
  pointT *point, **pointp;  // 声明变量point和pointp，用于处理点和点的指针数组
  int numpart;  // 声明整型变量numpart，用于计数
  realT dist, innerplane;  // 声明实型变量dist和innerplane，用于存储距离和平面内部距离

  // 如果KEEPcoplanar和KEEPinside都为假，则执行以下操作
  if (!qh->KEEPcoplanar && !qh->KEEPinside) {
    FORALLfacets {  // 遍历所有的facet
      if (facet->coplanarset)  // 如果facet的coplanarset不为空
        qh_setfree(qh, &facet->coplanarset);  // 释放facet的coplanarset
    }
  }else if (!qh->KEEPcoplanar || !qh->KEEPinside) {  // 如果KEEPcoplanar或KEEPinside有一个为假，则执行以下操作
    qh_outerinner(qh, NULL, NULL, &innerplane);  // 调用qh_outerinner函数计算内外平面距离，存储在innerplane中
    if (qh->JOGGLEmax < REALmax/2)  // 如果JOGGLEmax小于REALmax的一半
      innerplane -= qh->JOGGLEmax * sqrt((realT)qh->hull_dim);  // 减去JOGGLEmax乘以hull_dim的平方根
    numpart= 0;  // 将numpart初始化为0
    FORALLfacets {  // 再次遍历所有的facet
      if (facet->coplanarset) {  // 如果facet的coplanarset不为空
        FOREACHpoint_(facet->coplanarset) {  // 遍历coplanarset中的每个点
          numpart++;  // 计数增加
          qh_distplane(qh, point, facet, &dist);  // 计算点到面的距离，存储在dist中
          if (dist < innerplane) {  // 如果距离小于innerplane
            if (!qh->KEEPinside)  // 如果KEEPinside为假
              SETref_(point)= NULL;  // 将point的引用设置为NULL
          }else if (!qh->KEEPcoplanar)  // 如果距离不小于innerplane但KEEPcoplanar为假
            SETref_(point)= NULL;  // 将point的引用设置为NULL
        }
        qh_setcompact(qh, facet->coplanarset);  // 紧缩coplanarset
      }
    }
    zzadd_(Zcheckpart, numpart);  // 将numpart添加到Zcheckpart中
  }
} /* nearcoplanar */
*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nearvertex">-</a>

  qh_nearvertex(qh, facet, point, bestdist )
    return nearest vertex in facet to point

  returns:
    vertex and its distance

  notes:
    if qh.DELAUNAY
      distance is measured in the input set
    searches neighboring tricoplanar facets (requires vertexneighbors)
      Slow implementation.  Recomputes vertex set for each point.
    The vertex set could be stored in the qh.keepcentrum facet.
*/
vertexT *qh_nearvertex(qhT *qh, facetT *facet, pointT *point, realT *bestdistp) {
  realT bestdist= REALmax, dist;  // 声明实型变量bestdist和dist，初始化为REALmax
  vertexT *bestvertex= NULL, *vertex, **vertexp, *apex;  // 声明vertexT类型的指针变量，包括bestvertex, vertex, vertexp, apex
  coordT *center;  // 声明coordT类型的指针变量center，用于存储坐标
  facetT *neighbor, **neighborp;  // 声明facetT类型的指针变量neighbor和neighborp
  setT *vertices;  // 声明setT类型的指针变量vertices，用于存储顶点集合
  int dim= qh->hull_dim;  // 获取qh结构体中的hull_dim，赋值给整型变量dim

  if (qh->DELAUNAY)  // 如果qh的DELAUNAY为真
    dim--;  // 将dim减1
  if (facet->tricoplanar) {  // 如果facet是三共面的
    if (!qh->VERTEXneighbors || !facet->center) {  // 如果VERTEXneighbors或facet的center为空
      qh_fprintf(qh, qh->ferr, 6158, "qhull internal error (qh_nearvertex): qh.VERTEXneighbors and facet->center required for tricoplanar facets\n");  // 输出错误信息到qh->ferr
      qh_errexit(qh, qh_ERRqhull, facet, NULL);  // 调用qh_errexit退出程序
    }
    vertices= qh_settemp(qh, qh->TEMPsize);  // 为vertices分配临时空间
    apex= SETfirstt_(facet->vertices, vertexT);  // 获取facet的第一个顶点，赋值给apex
    center= facet->center;  // 获取facet的center，赋值给center
    FOREACHneighbor_(apex) {  // 遍历apex的每个邻居
      if (neighbor->center == center) {  // 如果邻居的center和facet的center相同
        FOREACHvertex_(neighbor->vertices)  // 遍历邻居的每个顶点
          qh_setappend(qh, &vertices, vertex);  // 将顶点添加到vertices中
      }
    }
  }else
    vertices= facet->vertices;  // 否则，将facet的顶点集合赋给vertices
  FOREACHvertex_(vertices) {  // 遍历vertices中的每个顶点
    dist= qh_pointdist(vertex->point, point, -dim);  // 计算顶点到点的距离，存储在dist中
    if (dist < bestdist) {  // 如果距离小于bestdist
      bestdist= dist;  // 更新bestdist为当前距离
      bestvertex= vertex;  // 更新bestvertex为当前顶点
    }
  }
  if (facet->tricoplanar)
*/
    # 释放指定顶点集合的临时内存，qh 是 Qhull 对象，vertices 是顶点集合的指针
    qh_settempfree(qh, &vertices);

  # 计算 bestdistp 指向的值的平方根，并将结果保存在 bestdistp 指向的位置
  *bestdistp= sqrt(bestdist);

  # 如果 bestvertex 为空指针，则记录错误信息到 qh->ferr，然后退出程序
  if (!bestvertex) {
      qh_fprintf(qh, qh->ferr, 6261, "qhull internal error (qh_nearvertex): did not find bestvertex for f%d p%d\n", facet->id, qh_pointid(qh, point));
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }

  # 输出跟踪信息到 qh->ferr，显示最近顶点的 ID、距离、面的 ID 以及点的 ID
  trace3((qh, qh->ferr, 3019, "qh_nearvertex: v%d dist %2.2g for f%d p%d\n",
        bestvertex->id, *bestdistp, facet->id, qh_pointid(qh, point))); /* bestvertex!=0 by QH2161 */

  # 返回 bestvertex，表示找到的最近顶点
  return bestvertex;
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newhashtable">-</a>

  qh_newhashtable(qh, newsize )
    returns size of qh.hash_table of at least newsize slots

  notes:
    assumes qh.hash_table is NULL
    qh_HASHfactor determines the number of extra slots
    size is not divisible by 2, 3, or 5
*/
int qh_newhashtable(qhT *qh, int newsize) {
  int size;

  size= ((newsize+1)*qh_HASHfactor) | 0x1;  /* odd number */
  while (True) {
    if (newsize<0 || size<0) {
        qh_fprintf(qh, qh->qhmem.ferr, 6236, "qhull error (qh_newhashtable): negative request (%d) or size (%d).  Did int overflow due to high-D?\n", newsize, size); /* WARN64 */
        qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
    }
    if ((size%3) && (size%5))
      break;
    size += 2;
    /* loop terminates because there is an infinite number of primes */
  }
  qh->hash_table= qh_setnew(qh, size);
  qh_setzero(qh, qh->hash_table, 0, size);
  return size;
} /* newhashtable */



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newvertex">-</a>

  qh_newvertex(qh, point )
    returns a new vertex for point
*/
vertexT *qh_newvertex(qhT *qh, pointT *point) {
  vertexT *vertex;

  zinc_(Ztotvertices);
  vertex= (vertexT *)qh_memalloc(qh, (int)sizeof(vertexT));
  memset((char *) vertex, (size_t)0, sizeof(vertexT));
  if (qh->vertex_id == UINT_MAX) {
    qh_memfree(qh, vertex, (int)sizeof(vertexT));
    qh_fprintf(qh, qh->ferr, 6159, "qhull error: 2^32 or more vertices.  vertexT.id field overflows.  Vertices would not be sorted correctly.\n");
    qh_errexit(qh, qh_ERRother, NULL, NULL);
  }
  if (qh->vertex_id == qh->tracevertex_id)
    qh->tracevertex= vertex;
  vertex->id= qh->vertex_id++;
  vertex->point= point;
  trace4((qh, qh->ferr, 4060, "qh_newvertex: vertex p%d(v%d) created\n", qh_pointid(qh, vertex->point),
          vertex->id));
  return(vertex);
} /* newvertex */



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nextridge3d">-</a>

  qh_nextridge3d( atridge, facet, vertex )
    return next ridge and vertex for a 3d facet
    returns NULL on error
    [for QhullFacet::nextRidge3d] Does not call qh_errexit nor access qhT.

  notes:
    in qh_ORIENTclock order
    this is a O(n^2) implementation to trace all ridges
    be sure to stop on any 2nd visit
    same as QhullRidge::nextRidge3d
    does not use qhT or qh_errexit [QhullFacet.cpp]

  design:
    for each ridge
      exit if it is the ridge after atridge
*/
ridgeT *qh_nextridge3d(ridgeT *atridge, facetT *facet, vertexT **vertexp) {
  vertexT *atvertex, *vertex, *othervertex;
  ridgeT *ridge, **ridgep;

  if ((atridge->top == facet) ^ qh_ORIENTclock)
    atvertex= SETsecondt_(atridge->vertices, vertexT);
  else
    atvertex= SETfirstt_(atridge->vertices, vertexT);
  FOREACHridge_(facet->ridges) {
    # 如果当前处理的ridge等于输入的atridge，则跳过本次循环，进入下一次循环
    if (ridge == atridge)
      continue;
    # 如果ridge的顶点等于facet，并且qh_ORIENTclock为真，执行以下操作
    if ((ridge->top == facet) ^ qh_ORIENTclock) {
      # 将ridge的第二个顶点设置为othervertex，将第一个顶点设置为vertex
      othervertex= SETsecondt_(ridge->vertices, vertexT);
      vertex= SETfirstt_(ridge->vertices, vertexT);
    }else {
      # 否则，将ridge的第一个顶点设置为vertex，将第二个顶点设置为othervertex
      vertex= SETsecondt_(ridge->vertices, vertexT);
      othervertex= SETfirstt_(ridge->vertices, vertexT);
    }
    # 如果vertex等于atvertex
    if (vertex == atvertex) {
      # 如果vertexp非空，将其指向othervertex所指的顶点
      if (vertexp)
        *vertexp= othervertex;
      # 返回ridge
      return ridge;
    }
  }
  # 循环结束时如果未找到匹配的ridge，返回NULL
  return NULL;
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="opposite_vertex">-</a>

  qh_opposite_vertex(qh, facetA, neighbor )
    return the opposite vertex in facetA to neighbor

*/
vertexT *qh_opposite_vertex(qhT *qh, facetT *facetA, facetT *neighbor) {
    vertexT *opposite= NULL;
    facetT *facet;
    int facet_i, facet_n;

    // 如果 facetA 是简单面（simplicial），即面上只有顶点，没有边，则查找相邻面 neighbor 在 facetA 中的对应顶点
    if (facetA->simplicial) {
      // 遍历 facetA 的相邻面列表
      FOREACHfacet_i_(qh, facetA->neighbors) {
        // 如果找到了相邻面 neighbor
        if (facet == neighbor) {
          // 获取 facetA 中对应位置 facet_i 的顶点，作为 opposite
          opposite= SETelemt_(facetA->vertices, facet_i, vertexT);
          break;
        }
      }
    }
    // 如果未找到对应的 opposite 顶点，则输出错误信息并退出程序
    if (!opposite) {
      qh_fprintf(qh, qh->ferr, 6396, "qhull internal error (qh_opposite_vertex): opposite vertex in facet f%d to neighbor f%d is not defined.  Either is facet is not simplicial or neighbor not found\n",
        facetA->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facetA, neighbor);
    }
    return opposite;
} /* opposite_vertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="outcoplanar">-</a>

  qh_outcoplanar()
    move points from all facets' outsidesets to their coplanarsets

  notes:
    for post-processing under qh.NARROWhull

  design:
    遍历每个面
      对于每个面的外部点
        将该点移动到面的共面点集合中
*/
void qh_outcoplanar(qhT *qh /* facet_list */) {
  pointT *point, **pointp;
  facetT *facet;
  realT dist;

  trace1((qh, qh->ferr, 1033, "qh_outcoplanar: move outsideset to coplanarset for qh->NARROWhull\n"));
  // 遍历所有面
  FORALLfacets {
    // 遍历当前面的外部点集合
    FOREACHpoint_(facet->outsideset) {
      qh->num_outside--;
      // 如果保留共面点或靠近内部的点
      if (qh->KEEPcoplanar || qh->KEEPnearinside) {
        // 计算点到面的距离
        qh_distplane(qh, point, facet, &dist);
        // 将点按照距离分配到共面点集合中
        zinc_(Zpartition);
        qh_partitioncoplanar(qh, point, facet, &dist, qh->findbestnew);
      }
    }
    // 释放外部点集合的内存
    qh_setfree(qh, &facet->outsideset);
  }
} /* outcoplanar */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="point">-</a>

  qh_point(qh, id )
    return point for a point id, or NULL if unknown

  alternative code:
    return((pointT *)((unsigned long)qh.first_point
           + (unsigned long)((id)*qh.normal_size)));
*/
pointT *qh_point(qhT *qh, int id) {

  // 如果点的 id 小于 0，返回 NULL
  if (id < 0)
    return NULL;
  // 如果点的 id 在第一个点数组中
  if (id < qh->num_points)
    return qh->first_point + id * qh->hull_dim;
  // 如果点的 id 在其他点集合中
  id -= qh->num_points;
  if (id < qh_setsize(qh, qh->other_points))
    return SETelemt_(qh->other_points, id, pointT);
  // 如果点的 id 超出范围，返回 NULL
  return NULL;
} /* point */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="point_add">-</a>

  qh_point_add(qh, set, point, elem )
    stores elem at set[point.id]

  returns:
    access function for qh_pointfacet and qh_pointvertex

  notes:
    检查点的 id，然后将 elem 存储在 set[point.id] 处
*/
void qh_point_add(qhT *qh, setT *set, pointT *point, void *elem) {
  int id, size;

  // 获取集合 set 的大小，并返回 size
  SETreturnsize_(set, size);
  // 获取点的 id
  if ((id= qh_pointid(qh, point)) < 0)
    // 如果 id 小于 0，则输出警告信息至 qh 的错误流 qh->ferr，警告信息包含未知点的地址和标识符
    qh_fprintf(qh, qh->ferr, 7067, "qhull internal warning (point_add): unknown point %p id %d\n",
      point, id);
  // 否则，如果 id 大于等于 size
  else if (id >= size) {
    // 输出错误信息至 qh 的错误流 qh->ferr，错误信息包含点的标识符和点的总数 size
    qh_fprintf(qh, qh->ferr, 6160, "qhull internal error (point_add): point p%d is out of bounds(%d)\n",
             id, size);
    // 通过 qh_errexit 函数，以 Qhull 错误码 qh_ERRqhull 结束程序，并输出相关信息
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  // 否则
  }else
    // 将 elem 赋值给集合 set 中的第 id 个元素
    SETelem_(set, id)= elem;
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointfacet">-</a>

  qh_pointfacet()
    return temporary set of facet for each point
    the set is indexed by point id
    at most one facet per point, arbitrary selection

  notes:
    each point is assigned to at most one of vertices, coplanarset, or outsideset
    unassigned points are interior points or 
    vertices assigned to one of its facets
    coplanarset assigned to the facet
    outside set assigned to the facet
    NULL if no facet for point (inside)
      includes qh.GOODpointp

  access:
    FOREACHfacet_i_(qh, facets) { ... }
    SETelem_(facets, i)

  design:
    for each facet
      add each vertex
      add each coplanar point
      add each outside point
*/
setT *qh_pointfacet(qhT *qh /* qh.facet_list */) {
  int numpoints= qh->num_points + qh_setsize(qh, qh->other_points);  // 计算点的总数，包括额外的点
  setT *facets;  // 用于存放临时的面集合
  facetT *facet;  // 面结构体指针
  vertexT *vertex, **vertexp;  // 顶点结构体指针和顶点结构体指针的指针
  pointT *point, **pointp;  // 点结构体指针和点结构体指针的指针

  facets= qh_settemp(qh, numpoints);  // 分配大小为numpoints的临时集合
  qh_setzero(qh, facets, 0, numpoints);  // 将集合facets清零
  qh->vertex_visit++;  // 递增顶点访问标识

  // 遍历所有的面
  FORALLfacets {
    // 遍历当前面的所有顶点
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit) {  // 如果顶点未被访问过
        vertex->visitid= qh->vertex_visit;  // 设置访问标识为当前标识
        qh_point_add(qh, facets, vertex->point, facet);  // 将顶点添加到面集合中
      }
    }
    // 遍历当前面的共面点集合
    FOREACHpoint_(facet->coplanarset)
      qh_point_add(qh, facets, point, facet);  // 将共面点添加到面集合中
    // 遍历当前面的外部点集合
    FOREACHpoint_(facet->outsideset)
      qh_point_add(qh, facets, point, facet);  // 将外部点添加到面集合中
  }
  return facets;  // 返回存放临时面集合的指针
} /* pointfacet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointvertex">-</a>

  qh_pointvertex(qh )
    return temporary set of vertices indexed by point id
    entry is NULL if no vertex for a point
      this will include qh.GOODpointp

  access:
    FOREACHvertex_i_(qh, vertices) { ... }
    SETelem_(vertices, i)
*/
setT *qh_pointvertex(qhT *qh /* qh.facet_list */) {
  int numpoints= qh->num_points + qh_setsize(qh, qh->other_points);  // 计算点的总数，包括额外的点
  setT *vertices;  // 用于存放临时的顶点集合
  vertexT *vertex;  // 顶点结构体指针

  vertices= qh_settemp(qh, numpoints);  // 分配大小为numpoints的临时集合
  qh_setzero(qh, vertices, 0, numpoints);  // 将集合vertices清零
  FORALLvertices
    qh_point_add(qh, vertices, vertex->point, vertex);  // 将顶点添加到顶点集合中
  return vertices;  // 返回存放临时顶点集合的指针
} /* pointvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="prependfacet">-</a>

  qh_prependfacet(qh, facet, facetlist )
    prepend facet to the start of a facetlist

  returns:
    increments qh.numfacets
    updates facetlist, qh.facet_list, facet_next

  notes:
    be careful of prepending since it can lose a pointer.
      e.g., can lose _next by deleting and then prepending before _next
*/
void qh_prependfacet(qhT *qh, facetT *facet, facetT **facetlist) {
  facetT *prevfacet, *list;

  trace4((qh, qh->ferr, 4061, "qh_prependfacet: prepend f%d before f%d\n",
          facet->id, getid_(*facetlist)));
  if (!*facetlist)
    // 如果facetlist为空，将facet设为facetlist的头部
    *facetlist= facet;
  else {
    // 否则，在facetlist头部插入facet
    list= *facetlist;
    while (list && list != facet) {  // 遍历直到找到facet或者到达列表末尾
      prevfacet= list;
      list= prevfacet->next;
    }
    if (!list) {  // 如果没有找到facet，将facet插入到facetlist的头部
      facet->next= *facetlist;
      *facetlist= facet;
    }
  }
  qh->num_facets++;  // 面数量加一
} /* prependfacet */
    (*facetlist)= qh->facet_tail;

将 qh 指针所指向的 facet_tail 赋值给 facetlist 指针所指向的内容。


  list= *facetlist;

将 facetlist 指针所指向的内容赋值给 list 变量。


  prevfacet= list->previous;

将 list 变量的 previous 成员赋值给 prevfacet 变量。


  facet->previous= prevfacet;

将 prevfacet 变量的值赋给 facet 结构体的 previous 成员。


  if (prevfacet)
    prevfacet->next= facet;

如果 prevfacet 不为空（即前一个面的存在），将 facet 指针赋值给 prevfacet 的 next 成员。


  list->previous= facet;

将 facet 指针赋值给 list 变量的 previous 成员。


  facet->next= *facetlist;

将 facetlist 指针所指向的内容赋值给 facet 结构体的 next 成员。


  if (qh->facet_list == list)  /* this may change *facetlist */
    qh->facet_list= facet;

如果 qh 指针的 facet_list 成员等于 list 变量，将 facet 指针赋值给 qh 指针的 facet_list 成员。这可能会改变 facetlist 指针所指向的内容。


  if (qh->facet_next == list)
    qh->facet_next= facet;

如果 qh 指针的 facet_next 成员等于 list 变量，将 facet 指针赋值给 qh 指针的 facet_next 成员。


  *facetlist= facet;

将 facet 指针赋值给 facetlist 指针所指向的内容。


  qh->num_facets++;

将 qh 指针的 num_facets 成员递增。增加面的计数。
} /* prependfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="printhashtable">-</a>

  qh_printhashtable(qh, fp )
    print hash table to fp

  notes:
    not in I/O to avoid bringing io_r.c in

  design:
    for each hash entry
      if defined
        if unmatched or will merge (NULL, qh_MERGEridge, qh_DUPLICATEridge)
          print entry and neighbors
*/
void qh_printhashtable(qhT *qh, FILE *fp) {
  facetT *facet, *neighbor;  /* 声明指向 facet 和 neighbor 的指针 */
  int id, facet_i, facet_n, neighbor_i= 0, neighbor_n= 0;  /* 声明整型变量 */
  vertexT *vertex, **vertexp;  /* 声明指向 vertex 的指针 */

  FOREACHfacet_i_(qh, qh->hash_table) {  /* 遍历 hash_table 中的每个 facet */
    if (facet) {  /* 如果 facet 不为空 */
      FOREACHneighbor_i_(qh, facet) {  /* 遍历 facet 的每个 neighbor */
        if (!neighbor || neighbor == qh_MERGEridge || neighbor == qh_DUPLICATEridge)  /* 如果 neighbor 为空或者是 qh_MERGEridge 或 qh_DUPLICATEridge */
          break;  /* 跳出循环 */
      }
      if (neighbor_i == neighbor_n)  /* 如果 neighbor_i 等于 neighbor_n */
        continue;  /* 继续下一次循环 */
      qh_fprintf(qh, fp, 9283, "hash %d f%d ", facet_i, facet->id);  /* 打印 hash 和 facet 的 id */
      FOREACHvertex_(facet->vertices)  /* 遍历 facet 的 vertices */
        qh_fprintf(qh, fp, 9284, "v%d ", vertex->id);  /* 打印 vertex 的 id */
      qh_fprintf(qh, fp, 9285, "\n neighbors:");  /* 打印换行和 neighbors 标签 */
      FOREACHneighbor_i_(qh, facet) {  /* 再次遍历 facet 的每个 neighbor */
        if (neighbor == qh_MERGEridge)  /* 如果 neighbor 是 qh_MERGEridge */
          id= -3;  /* id 设为 -3 */
        else if (neighbor == qh_DUPLICATEridge)  /* 如果 neighbor 是 qh_DUPLICATEridge */
          id= -2;  /* id 设为 -2 */
        else
          id= getid_(neighbor);  /* 否则获取 neighbor 的 id */
        qh_fprintf(qh, fp, 9286, " %d", id);  /* 打印 id */
      }
      qh_fprintf(qh, fp, 9287, "\n");  /* 打印换行 */
    }
  }
} /* printhashtable */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="printlists">-</a>

  qh_printlists(qh)
    print out facet and vertex lists for debugging (without 'f/v' tags)

  notes:
    not in I/O to avoid bringing io_r.c in
*/
void qh_printlists(qhT *qh) {
  facetT *facet;  /* 声明指向 facet 的指针 */
  vertexT *vertex;  /* 声明指向 vertex 的指针 */
  int count= 0;  /* 声明计数器变量并初始化为 0 */

  qh_fprintf(qh, qh->ferr, 3062, "qh_printlists: max_outside %2.2g all facets:", qh->max_outside);  /* 打印信息包括 max_outside 和所有 facets */
  FORALLfacets{  /* 遍历所有 facets */
    if (++count % 100 == 0)  /* 每 100 个 facets 打印换行 */
      qh_fprintf(qh, qh->ferr, 8109, "\n     ");
    qh_fprintf(qh, qh->ferr, 8110, " %d", facet->id);  /* 打印 facet 的 id */
  }
    qh_fprintf(qh, qh->ferr, 8111, "\n  qh.visible_list f%d, newfacet_list f%d, facet_next f%d for qh_addpoint\n  qh.newvertex_list v%d all vertices:",
      getid_(qh->visible_list), getid_(qh->newfacet_list), getid_(qh->facet_next), getid_(qh->newvertex_list));  /* 打印 qh 的各个列表的 id */
  count= 0;  /* 重置计数器为 0 */
  FORALLvertices{  /* 遍历所有 vertices */
    if (++count % 100 == 0)  /* 每 100 个 vertices 打印换行 */
      qh_fprintf(qh, qh->ferr, 8112, "\n     ");
    qh_fprintf(qh, qh->ferr, 8113, " %d", vertex->id);  /* 打印 vertex 的 id */
  }
  qh_fprintf(qh, qh->ferr, 8114, "\n");  /* 打印换行 */
} /* printlists */

/*-<a                             href="qh-poly.htm#TOC"
  >-------------------------------</a><a name="addfacetvertex">-</a>

  qh_replacefacetvertex(qh, facet, oldvertex, newvertex )
    replace oldvertex with newvertex in f.vertices
    vertices are inverse sorted by vertex->id

  returns:
    toporient is flipped if an odd parity, position change

  notes:
    for simplicial facets in qh_rename_adjacentvertex
    see qh_addfacetvertex
*/
void qh_replacefacetvertex(qhT *qh, facetT *facet, vertexT *oldvertex, vertexT *newvertex) {
  vertexT *vertex;  // 定义顶点变量
  facetT *neighbor;  // 定义相邻面变量
  int vertex_i, vertex_n= 0;  // 定义顶点索引和顶点数目变量
  int old_i= -1, new_i= -1;  // 初始化旧顶点索引和新顶点索引为 -1

  trace3((qh, qh->ferr, 3038, "qh_replacefacetvertex: replace v%d with v%d in f%d\n", oldvertex->id, newvertex->id, facet->id));
  // 输出跟踪信息，记录替换旧顶点和新顶点在面中的操作

  if (!facet->simplicial) {
    qh_fprintf(qh, qh->ferr, 6283, "qhull internal error (qh_replacefacetvertex): f%d is not simplicial\n", facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
    // 如果面不是单纯性的，输出错误信息并退出程序
  }

  FOREACHvertex_i_(qh, facet->vertices) {
    // 遍历面的顶点集合
    if (new_i == -1 && vertex->id < newvertex->id) {
      new_i= vertex_i;  // 如果找到合适位置插入新顶点，记录新顶点索引
    } else if (vertex->id == newvertex->id) {
      qh_fprintf(qh, qh->ferr, 6281, "qhull internal error (qh_replacefacetvertex): f%d already contains new v%d\n", facet->id, newvertex->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
      // 如果面已经包含新顶点，输出错误信息并退出程序
    }
    if (vertex->id == oldvertex->id) {
      old_i= vertex_i;  // 记录旧顶点索引
    }
  }

  if (old_i == -1) {
    qh_fprintf(qh, qh->ferr, 6282, "qhull internal error (qh_replacefacetvertex): f%d does not contain old v%d\n", facet->id, oldvertex->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
    // 如果面不包含旧顶点，输出错误信息并退出程序
  }

  if (new_i == -1) {
    new_i= vertex_n;  // 如果没有找到合适位置插入新顶点，设置新顶点索引为顶点数目
  }

  if (old_i < new_i)
    new_i--;  // 调整新顶点索引位置

  if ((old_i & 0x1) != (new_i & 0x1))
    facet->toporient ^= 1;  // 调整面的方向标志位

  qh_setdelnthsorted(qh, facet->vertices, old_i);  // 删除旧顶点在面的顶点集合中的位置
  qh_setaddnth(qh, &facet->vertices, new_i, newvertex);  // 在新位置插入新顶点到面的顶点集合

  neighbor= SETelemt_(facet->neighbors, old_i, facetT);  // 获取旧顶点对应的相邻面
  qh_setdelnthsorted(qh, facet->neighbors, old_i);  // 删除旧顶点在面的相邻面集合中的位置
  qh_setaddnth(qh, &facet->neighbors, new_i, neighbor);  // 在新位置插入相邻面到面的相邻面集合
} /* replacefacetvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="resetlists">-</a>

  qh_resetlists(qh, stats, resetVisible /* qh.newvertex_list newfacet_list visible_list */) {
    vertexT *vertex;  // 定义顶点变量
    facetT *newfacet, *visible;  // 定义新面和可见面变量
    int totnew=0, totver=0;  // 初始化新面数目和顶点数目为 0

    trace2((qh, qh->ferr, 2066, "qh_resetlists: reset newvertex_list v%d, newfacet_list f%d, visible_list f%d, facet_list f%d next f%d vertex_list v%d -- NEWfacets? %d, NEWtentative? %d, stats? %d\n",
      getid_(qh->newvertex_list), getid_(qh->newfacet_list), getid_(qh->visible_list), getid_(qh->facet_list), getid_(qh->facet_next), getid_(qh->vertex_list), qh->NEWfacets, qh->NEWtentative, stats));
    // 输出跟踪信息，记录重置各个列表和相关统计信息

    if (stats) {
      FORALLvertex_(qh->newvertex_list)
        totver++;  // 统计新顶点列表中的顶点数目
      FORALLnew_facets
        totnew++;  // 统计新面列表中的面数目
      zadd_(Zvisvertextot, totver);  // 更新可见顶点的总数目统计
      zmax_(Zvisvertexmax, totver);  // 更新可见顶点的最大数目统计
      zadd_(Znewfacettot, totnew);  // 更新新面的总数目统计
    }

    if (resetVisible) {
      // 如果需要重置可见面列表
      visible_list is restored to facet_list
      visible_list 被恢复为 facet_list
    } else {
      // 否则保持可见面的可见性和替换属性
      f.visible/f.replace is retained
      f.visible/f.replace 被保留
    }

    // 重置各个列表为空
    newvertex_list, newfacet_list, visible_list are NULL
  }
    zmax_(Znewfacetmax, totnew);


    // 调用 zmax_ 函数，传入 Znewfacetmax 和 totnew 作为参数
    zmax_(Znewfacetmax, totnew);



  }
  FORALLvertex_(qh->newvertex_list)
    vertex->newfacet= False;
  qh->newvertex_list= NULL;
  qh->first_newfacet= 0;


  // 结束上一部分的代码块

  // 遍历 qh 结构体中 newvertex_list 所有的顶点
  FORALLvertex_(qh->newvertex_list)
    // 将顶点的 newfacet 标志设为 False
    vertex->newfacet= False;

  // 将 qh 结构体中 newvertex_list 置为 NULL
  qh->newvertex_list= NULL;

  // 将 qh 结构体中的 first_newfacet 置为 0
  qh->first_newfacet= 0;



  FORALLnew_facets {
    newfacet->newfacet= False;
    newfacet->dupridge= False;
  }
  qh->newfacet_list= NULL;


  // 遍历所有的 new_facets
  FORALLnew_facets {
    // 将 newfacet 的 newfacet 和 dupridge 标志设为 False
    newfacet->newfacet= False;
    newfacet->dupridge= False;
  }

  // 将 qh 结构体中的 newfacet_list 置为 NULL
  qh->newfacet_list= NULL;



  if (resetVisible) {
    FORALLvisible_facets {
      visible->f.replace= NULL;
      visible->visible= False;
    }
    qh->num_visible= 0;
  }
  qh->visible_list= NULL; 


  // 如果 resetVisible 为真，则执行以下代码块
  if (resetVisible) {
    // 遍历所有可见的 facets
    FORALLvisible_facets {
      // 将 visible 结构体中的 f.replace 置为 NULL
      visible->f.replace= NULL;
      // 将 visible 结构体中的 visible 标志设为 False
      visible->visible= False;
    }
    // 将 qh 结构体中的 num_visible 置为 0
    qh->num_visible= 0;
  }

  // 将 qh 结构体中的 visible_list 置为 NULL
  qh->visible_list= NULL;



  qh->NEWfacets= False;
  qh->NEWtentative= False;


  // 将 qh 结构体中的 NEWfacets 标志设为 False
  qh->NEWfacets= False;
  // 将 qh 结构体中的 NEWtentative 标志设为 False
  qh->NEWtentative= False;
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="setvoronoi_all">-</a>

  qh_setvoronoi_all(qh)
    compute Voronoi centers for all facets
    includes upperDelaunay facets if qh.UPPERdelaunay ('Qu')

  returns:
    facet->center is the Voronoi center

  notes:
    unused/untested code: please email bradb@shore.net if this works ok for you

  use:
    FORALLvertices {...} to locate the vertex for a point.
    FOREACHneighbor_(vertex) {...} to visit the Voronoi centers for a Voronoi cell.
*/
void qh_setvoronoi_all(qhT *qh) {
  facetT *facet;

  qh_clearcenters(qh, qh_ASvoronoi);  // 清除所有 Voronoi 中心
  qh_vertexneighbors(qh);             // 计算每个顶点的邻居列表

  FORALLfacets {                      // 遍历所有的面
    if (!facet->normal || !facet->upperdelaunay || qh->UPPERdelaunay) {  // 如果面没有法向量或不是上半 Delaunay 或者 qh.UPPERdelaunay 被设置
      if (!facet->center)            // 如果面没有 Voronoi 中心
        facet->center= qh_facetcenter(qh, facet->vertices);  // 计算面的 Voronoi 中心并设置
    }
  }
} /* setvoronoi_all */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate">-</a>

  qh_triangulate()
    triangulate non-simplicial facets on qh.facet_list,
    if qh->VORONOI, sets Voronoi centers of non-simplicial facets
    nop if hasTriangulation

  returns:
    all facets simplicial
    each tricoplanar facet has ->f.triowner == owner of ->center,normal,etc.
    resets qh.newfacet_list and visible_list

  notes:
    called by qh_prepare_output and user_eg2_r.c
    call after qh_check_output since may switch to Voronoi centers, and qh_checkconvex skips f.tricoplanar facets
    Output may overwrite ->f.triowner with ->f.area
    while running, 'triangulated_facet_list' is a list of
       one non-simplicial facet followed by its 'f.tricoplanar' triangulated facets
    See qh_buildcone
*/
void qh_triangulate(qhT *qh /* qh.facet_list */) {
  facetT *facet, *nextfacet, *owner;
  facetT *neighbor, *visible= NULL, *facet1, *facet2, *triangulated_facet_list= NULL;
  facetT *orig_neighbor= NULL, *otherfacet;
  vertexT *triangulated_vertex_list= NULL;
  mergeT *merge;
  mergeType mergetype;
  int neighbor_i, neighbor_n;
  boolT onlygood= qh->ONLYgood;

  if (qh->hasTriangulation)
      return;   // 如果已经进行了三角化，则直接返回
  trace1((qh, qh->ferr, 1034, "qh_triangulate: triangulate non-simplicial facets\n"));  // 输出调试信息
  if (qh->hull_dim == 2)
    return;     // 如果维度为2，则直接返回，不进行三角化
  if (qh->VORONOI) {  // 如果启用了 Voronoi 设置
    qh_clearcenters(qh, qh_ASvoronoi);  // 清除所有 Voronoi 中心
    qh_vertexneighbors(qh);             // 计算每个顶点的邻居列表
  }
  qh->ONLYgood= False; /* for makenew_nonsimplicial */  // 设置为 False，用于 makenew_nonsimplicial 函数
  qh->visit_id++;     // 增加 visit_id
  qh_initmergesets(qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */);  // 初始化合并集合
  qh->newvertex_list= qh->vertex_tail;
  for (facet=qh->facet_list; facet && facet->next; facet= nextfacet) { /* non-simplicial facets moved to end */
    nextfacet= facet->next;
    if (facet->visible || facet->simplicial)
      continue;   // 如果是可见或者是 simplicial 的，则跳过

    /* triangulate all non-simplicial facets, otherwise merging does not work, e.g., RBOX c P-0.1 P+0.1 P+0.1 D3 | QHULL d Qt Tv */
    if (!triangulated_facet_list)
      triangulated_facet_list= facet;  /* 如果三角化的面列表为空，则将当前面设置为第一个三角化的面 */
    qh_triangulate_facet(qh, facet, &triangulated_vertex_list); /* 调用qh_triangulate_facet函数进行面的三角化，同时更新三角化顶点列表 */
  }
  /* 因为f.visible缺少qh.visible_list，导致qh_checkpolygon无效 */
  trace2((qh, qh->ferr, 2047, "qh_triangulate: 从面列表f%d中删除空面。空面具有相同的第一个（顶点尖端）和第二个顶点\n", getid_(triangulated_facet_list)));
  for (facet=triangulated_facet_list; facet && facet->next; facet= nextfacet) {
    nextfacet= facet->next;
    if (facet->visible)
      continue;
    if (facet->ridges) {
      if (qh_setsize(qh, facet->ridges) > 0) {
        qh_fprintf(qh, qh->ferr, 6161, "qhull内部错误（qh_triangulate）：仍然为f%d定义了脊\n", facet->id);
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
      qh_setfree(qh, &facet->ridges);  /* 释放脊集合 */
    }
    if (SETfirst_(facet->vertices) == SETsecond_(facet->vertices)) {
      zinc_(Ztrinull);
      qh_triangulate_null(qh, facet); /* 将删除空面 */
    }
  }
  trace2((qh, qh->ferr, 2048, "qh_triangulate: 删除%d个或更多镜像面。由于空面，镜像面具有相同的顶点\n", qh_setsize(qh, qh->degen_mergeset)));
  qh->visible_list= qh->facet_tail;
  while ((merge= (mergeT *)qh_setdellast(qh->degen_mergeset))) {
    facet1= merge->facet1;
    facet2= merge->facet2;
    mergetype= merge->mergetype;
    qh_memfree(qh, merge, (int)sizeof(mergeT));
    if (mergetype == MRGmirror) {
      zinc_(Ztrimirror);
      qh_triangulate_mirror(qh, facet1, facet2);  /* 将删除这两个面 */
    }
  }
  qh_freemergesets(qh);  /* 释放合并集合 */
  trace2((qh, qh->ferr, 2049, "qh_triangulate: 从v%d更新顶点的邻居列表\n", getid_(triangulated_vertex_list)));
  qh->newvertex_list= triangulated_vertex_list;  /* 所有三角化面的顶点 */
  qh->visible_list= NULL;
  qh_update_vertexneighbors(qh /* qh.newvertex_list, 空的newfacet_list和visible_list */);
  qh_resetlists(qh, False, !qh_RESETvisible /* qh.newvertex_list, 空的newfacet_list和visible_list */);

  trace2((qh, qh->ferr, 2050, "qh_triangulate: 识别从f%d中产生的退化三角面片\n", getid_(triangulated_facet_list)));
  trace2((qh, qh->ferr, 2051, "qh_triangulate: 并用拥有中心、法线等的三角面片替换facet->f.triowner\n"));
  FORALLfacet_(triangulated_facet_list) {
    if (facet->tricoplanar && !facet->visible) {
      // 检查当前面片是否为三平面面片且不可见
      FOREACHneighbor_i_(qh, facet) {
        // 遍历当前面片的邻居面片
        if (neighbor_i == 0) {  /* first iteration */
          // 如果是第一次迭代
          if (neighbor->tricoplanar)
            // 如果邻居面片也是三平面面片
            orig_neighbor= neighbor->f.triowner;
          else
            // 否则直接使用邻居面片
            orig_neighbor= neighbor;
        }else {
          // 对于非第一次迭代
          if (neighbor->tricoplanar)
            // 如果邻居面片是三平面面片
            otherfacet= neighbor->f.triowner;
          else
            // 否则直接使用邻居面片
            otherfacet= neighbor;
          // 检查是否存在与初始邻居相同的面片
          if (orig_neighbor == otherfacet) {
            // 如果存在，则标记当前面片为退化的
            zinc_(Ztridegen);
            facet->degenerate= True;
            // 中断循环
            break;
          }
        }
      }
    }
  }
  if (qh->IStracing >= 4)
    // 如果跟踪级别高于等于4，打印 qh 对象的列表信息
    qh_printlists(qh);
  trace2((qh, qh->ferr, 2052, "qh_triangulate: delete visible facets -- non-simplicial, null, and mirrored facets\n"));
  // 初始化变量
  owner= NULL;
  visible= NULL;
  for (facet=triangulated_facet_list; facet && facet->next; facet= nextfacet) { 
    /* deleting facets, triangulated_facet_list is no longer valid */
    // 遍历三角化面片列表，删除面片，triangulated_facet_list 不再有效
    nextfacet= facet->next;
    if (facet->visible) {
      // 如果当前面片可见
      if (facet->tricoplanar) { /* a null or mirrored facet */
        // 如果当前面片是 null 或者镜像面片，删除该面片
        qh_delfacet(qh, facet);
        qh->num_visible--;
      }else {  /* a non-simplicial facet followed by its tricoplanars */
        // 如果是非简单面片，后跟其三平面面片
        if (visible && !owner) {
          // 如果存在可见面片且没有所有者
          trace2((qh, qh->ferr, 2053, "qh_triangulate: delete f%d.  All tricoplanar facets degenerate for non-simplicial facet\n",
                       visible->id));
          // 删除可见面片
          qh_delfacet(qh, visible);
          qh->num_visible--;
        }
        // 更新可见面片和所有者
        visible= facet;
        owner= NULL;
      }
    }else if (facet->tricoplanar) {
      // 如果当前面片是三平面面片但不可见
      if (facet->f.triowner != visible || visible==NULL) {
        // 如果三平面面片的拥有者不是可见面片或者可见面片为NULL，打印错误信息并退出
        qh_fprintf(qh, qh->ferr, 6162, "qhull internal error (qh_triangulate): tricoplanar facet f%d not owned by its visible, non-simplicial facet f%d\n", facet->id, getid_(visible));
        qh_errexit2(qh, qh_ERRqhull, facet, visible);
      }
      // 如果存在所有者，设置面片的拥有者
      if (owner)
        facet->f.triowner= owner;
      else if (!facet->degenerate) {
        // 否则如果面片不是退化的，设置当前面片为所有者，并处理相关属性
        owner= facet;
        nextfacet= visible->next; /* rescan tricoplanar facets with owner, visible!=0 by QH6162 */
        facet->keepcentrum= True;  /* one facet owns ->normal, etc. */
        facet->coplanarset= visible->coplanarset;
        facet->outsideset= visible->outsideset;
        visible->coplanarset= NULL;
        visible->outsideset= NULL;
        if (!qh->TRInormals) { /* center and normal copied to tricoplanar facets */
          visible->center= NULL;
          visible->normal= NULL;
        }
        // 删除可见面片
        qh_delfacet(qh, visible);
        qh->num_visible--;
      }
    }
    // 重置面片的退化状态
    facet->degenerate= False; /* reset f.degenerate set by qh_triangulate*/
  }
  // 如果存在可见面片但没有所有者，删除可见面片
  if (visible && !owner) {
    trace2((qh, qh->ferr, 2054, "qh_triangulate: all tricoplanar facets degenerate for last non-simplicial facet f%d\n",
                 visible->id));
    qh_delfacet(qh, visible);
    qh->num_visible--;

减少qh结构体中num_visible字段的值，减一操作。


  }

结束if语句块。


  qh->ONLYgood= onlygood; /* restore value */

将onlygood的值恢复给qh结构体中的ONLYgood字段，注释表明这是对该字段进行恢复操作。


  if (qh->CHECKfrequently)

如果qh结构体中的CHECKfrequently字段为真（非零值），则执行以下代码块。


    qh_checkpolygon(qh, qh->facet_list);

调用qh_checkpolygon函数，传入qh结构体和qh结构体中的facet_list字段作为参数。


  qh->hasTriangulation= True;

将qh结构体中的hasTriangulation字段设置为True，表明已经进行了三角化操作。
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_facet">-</a>

  qh_triangulate_facet(qh, facetA, &firstVertex )
    triangulate a non-simplicial facet
      if qh.CENTERtype=qh_ASvoronoi, sets its Voronoi center
  returns:
    qh.newfacet_list == simplicial facets
      facet->tricoplanar set and ->keepcentrum false
      facet->degenerate set if duplicated apex
      facet->f.trivisible set to facetA
      facet->center copied from facetA (created if qh_ASvoronoi)
        qh_eachvoronoi, qh_detvridge, qh_detvridge3 assume centers copied
      facet->normal,offset,maxoutside copied from facetA

  notes:
      only called by qh_triangulate
      qh_makenew_nonsimplicial uses neighbor->seen for the same
      if qh.TRInormals, newfacet->normal will need qh_free
        if qh.TRInormals and qh_AScentrum, newfacet->center will need qh_free
        keepcentrum is also set on Zwidefacet in qh_mergefacet
        freed by qh_clearcenters

  see also:
      qh_addpoint() -- add a point
      qh_makenewfacets() -- construct a cone of facets for a new vertex

  design:
      if qh_ASvoronoi,
         compute Voronoi center (facet->center)
      select first vertex (highest ID to preserve ID ordering of ->vertices)
      triangulate from vertex to ridges
      copy facet->center, normal, offset
      update vertex neighbors
*/
void qh_triangulate_facet(qhT *qh, facetT *facetA, vertexT **first_vertex) {
  facetT *newfacet;
  facetT *neighbor, **neighborp;
  vertexT *apex;
  int numnew=0;

  trace3((qh, qh->ferr, 3020, "qh_triangulate_facet: triangulate facet f%d\n", facetA->id));

  qh->first_newfacet= qh->facet_id; // 设置第一个新面的 ID 为当前 ID 计数器的值
  if (qh->IStracing >= 4) // 如果跟踪级别大于等于 4，则输出 facetA 的详细信息
    qh_printfacet(qh, qh->ferr, facetA);

  // 将facetA的每个邻面的seen标记为False，同时将coplanarhorizon标记为False
  FOREACHneighbor_(facetA) {
    neighbor->seen= False;
    neighbor->coplanarhorizon= False;
  }

  // 如果qh.CENTERtype为qh_ASvoronoi，并且facetA的center尚未设置
  // 则根据facetA的法向量和qh.ANGLEround * qh_ZEROdelaunay的阈值设置其Voronoi中心
  if (qh->CENTERtype == qh_ASvoronoi && !facetA->center && fabs_(facetA->normal[qh->hull_dim -1]) >= qh->ANGLEround * qh_ZEROdelaunay) {
    facetA->center= qh_facetcenter(qh, facetA->vertices);
  }

  // 设置qh.visible_list和qh.newfacet_list为facetA的facet_tail
  qh->visible_list= qh->newfacet_list= qh->facet_tail;

  // 设置facetA的visitid为qh的visit_id
  facetA->visitid= qh->visit_id;

  // 从facetA的顶点集合中选择第一个顶点作为apex
  apex= SETfirstt_(facetA->vertices, vertexT);

  // 调用qh_makenew_nonsimplicial函数来创建非单纯面的新面，并传递apex作为参数
  qh_makenew_nonsimplicial(qh, facetA, apex, &numnew);

  // 将facetA标记为将要被删除，置为删除状态
  qh_willdelete(qh, facetA, NULL);

  // 对于所有新创建的面，设置一些初始属性
  FORALLnew_facets {
    newfacet->tricoplanar= True; // 设置为三共面
    newfacet->f.trivisible= facetA; // 设置f.trivisible为facetA
    newfacet->degenerate= False; // 不是退化面
    newfacet->upperdelaunay= facetA->upperdelaunay; // 设置upperdelaunay属性与facetA相同
    newfacet->good= facetA->good; // 设置good属性与facetA相同
    // 如果输入几何结构体qh的TRInormals字段为真，则处理'Q11'模式下的法向量和中心点复制
    if (qh->TRInormals) { /* 'Q11' triangulate duplicates ->normal and ->center */
      // 设置新面结构体newfacet的keepcentrum字段为真，表示保留中心点
      newfacet->keepcentrum= True;
      // 如果facetA的法向量存在，则为newfacet分配内存并复制facetA的法向量数据
      if(facetA->normal){
        newfacet->normal= (double *)qh_memalloc(qh, qh->normal_size);
        memcpy((char *)newfacet->normal, facetA->normal, (size_t)qh->normal_size);
      }
      // 根据qh的CENTERtype类型来设置newfacet的center字段
      if (qh->CENTERtype == qh_AScentrum)
        // 如果CENTERtype为qh_AScentrum，则调用qh_getcentrum函数获取中心点并赋给newfacet->center
        newfacet->center= qh_getcentrum(qh, newfacet);
      else if (qh->CENTERtype == qh_ASvoronoi && facetA->center){
        // 如果CENTERtype为qh_ASvoronoi且facetA的中心点存在，则为newfacet分配内存并复制facetA的中心点数据
        newfacet->center= (double *)qh_memalloc(qh, qh->center_size);
        memcpy((char *)newfacet->center, facetA->center, (size_t)qh->center_size);
      }
    }else {
      // 如果qh的TRInormals字段为假，则设置newfacet的keepcentrum字段为假，不保留中心点
      newfacet->keepcentrum= False;
      // 在qh_triangulate函数结束时，会保证至少一个面的keepcentrum为真
      /* one facet will have keepcentrum=True at end of qh_triangulate */
      // 直接将facetA的法向量和中心点赋给newfacet，不进行复制
      newfacet->normal= facetA->normal;
      newfacet->center= facetA->center;
    }
    // 将facetA的offset字段值赋给newfacet的offset字段
    newfacet->offset= facetA->offset;
#if qh_MAXoutside
    newfacet->maxoutside= facetA->maxoutside;
#endif


// 如果定义了 qh_MAXoutside 宏，则将 newfacet 的 maxoutside 属性设置为 facetA 的 maxoutside 属性的值



  }
  qh_matchnewfacets(qh /* qh.newfacet_list */); /* ignore returned value, maxdupdist */ 
  zinc_(Ztricoplanar);
  zadd_(Ztricoplanartot, numnew);
  zmax_(Ztricoplanarmax, numnew);


  // 调用 qh_matchnewfacets 函数，参数为 qh.newfacet_list，忽略返回值和 maxdupdist
  // 增加 Ztricoplanar 计数器
  // 将 numnew 添加到 Ztricoplanartot 计数器中
  // 更新 Ztricoplanarmax 计数器的最大值为 numnew



  if (!(*first_vertex))
    (*first_vertex)= qh->newvertex_list;


  // 如果 *first_vertex 为 NULL，则将其赋值为 qh->newvertex_list 的值



  qh->newvertex_list= NULL;
  qh->visible_list= NULL;


  // 将 qh->newvertex_list 和 qh->visible_list 设置为 NULL



  /* only update v.neighbors for qh.newfacet_list.  qh.visible_list and qh.newvertex_list are NULL */
  qh_update_vertexneighbors(qh /* qh.newfacet_list */);


  // 只更新 qh.newfacet_list 中的顶点邻居关系
  // 调用 qh_update_vertexneighbors 函数，参数为 qh.newfacet_list



  qh_resetlists(qh, False, !qh_RESETvisible /* qh.newfacet_list */);


  // 重置 qh 结构体中的列表，不重置可见列表（!qh_RESETvisible），只重置 qh.newfacet_list
  // 调用 qh_resetlists 函数，参数为 qh, False, !qh_RESETvisible



} /* triangulate_facet */


// 函数 triangulate_facet 的结束标记



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_link">-</a>

  qh_triangulate_link(qh, oldfacetA, facetA, oldfacetB, facetB)
    relink facetA to facetB via null oldfacetA or mirrored oldfacetA and oldfacetB
  returns:
    if neighbors are already linked, will merge as MRGmirror (qh.degen_mergeset, 4-d and up)
*/
void qh_triangulate_link(qhT *qh, facetT *oldfacetA, facetT *facetA, facetT *oldfacetB, facetT *facetB) {


// 文档注释：定义了 qh-poly_r.htm#TOC 锚点，描述了 qh_triangulate_link 函数的作用和返回值
// 函数 qh_triangulate_link 的声明，参数包括 qh 句柄以及四个 facet 类型的指针参数



  int errmirror= False;

  if (oldfacetA == oldfacetB) {
    trace3((qh, qh->ferr, 3052, "qh_triangulate_link: relink neighbors f%d and f%d of null facet f%d\n",
      facetA->id, facetB->id, oldfacetA->id));
  }else {
    trace3((qh, qh->ferr, 3021, "qh_triangulate_link: relink neighbors f%d and f%d of mirrored facets f%d and f%d\n",
      facetA->id, facetB->id, oldfacetA->id, oldfacetB->id));
  }


// 初始化 errmirror 为 False
// 如果 oldfacetA 等于 oldfacetB，则输出消息到 trace 日志，描述重新链接 null facet f%d 的邻居 f%d 和 f%d
// 否则，输出消息到 trace 日志，描述重新链接镜像 facet f%d 和 f%d 的邻居 f%d 和 f%d



  if (qh_setin(facetA->neighbors, facetB)) {
    if (!qh_setin(facetB->neighbors, facetA))
      errmirror= True;
    else if (!facetA->redundant || !facetB->redundant || !qh_hasmerge(qh->degen_mergeset, MRGmirror, facetA, facetB))
      qh_appendmergeset(qh, facetA, facetB, MRGmirror, 0.0, 1.0);
  }else if (qh_setin(facetB->neighbors, facetA))
    errmirror= True;


// 检查 facetA 的邻居集合中是否包含 facetB
// 如果是，则检查 facetB 的邻居集合是否包含 facetA，若不包含则设置 errmirror 为 True
// 否则，如果 facetA 或 facetB 不是冗余的，或者 qh.degen_mergeset 中不存在 MRGmirror 类型的合并操作，
// 则将 MRGmirror 类型的合并操作添加到 qh.degen_mergeset 中
// 如果 facetB 的邻居集合中包含 facetA，则设置 errmirror 为 True



  if (errmirror) {
    qh_fprintf(qh, qh->ferr, 6163, "qhull internal error (qh_triangulate_link): neighbors f%d and f%d do not match for null facet or mirrored facets f%d and f%d\n",
       facetA->id, facetB->id, oldfacetA->id, oldfacetB->id);
    qh_errexit2(qh, qh_ERRqhull, facetA, facetB);
  }


// 如果 errmirror 为 True，则输出错误消息到 qh->ferr，描述邻居 f%d 和 f%d 不匹配的情况
// 调用 qh_errexit2 函数，参数为 qh, qh_ERRqhull, facetA, facetB，退出程序



  qh_setreplace(qh, facetB->neighbors, oldfacetB, facetA);
  qh_setreplace(qh, facetA->neighbors, oldfacetA, facetB);


// 使用 qh_setreplace 函数将 facetB 的邻居集合中的 oldfacetB 替换为 facetA
// 使用 qh_setreplace 函数将 facetA 的邻居集合中的 oldfacetA 替换为 facetB



} /* triangulate_link */


// 函数 triangulate_link 的结束标记



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_mirror">-</a>

  qh_triangulate_mirror(qh, facetA, facetB)
    delete two mirrored facets identified by qh_triangulate_null() and itself
      a mirrored facet shares the same vertices of a logical ridge
  design:
    since a null facet duplicates the first two vertices, the opposing neighbors absorb the null facet
    if they are already neighbors, the opposing neighbors become MRGmirror facets
*/


// 文档注释：定义了 qh-poly_r.htm#TOC 锚点，描述了 qh_triangulate_mirror 函数的作用和设计思路
// 函数 qh_triangulate_mirror 的声明，参数包括 qh 句柄以及两个 facet 类型的指针参数
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_mirror">-</a>

  qh_triangulate_mirror(qh, facetA, facetB)
    delete mirrored facets facetA and facetB and link their neighbors
  parameters:
    qh      - pointer to the qhT structure (global data structure for Qhull)
    facetA  - pointer to the first facet to be mirrored
    facetB  - pointer to the second facet to be mirrored
  design:
    Iterates through neighbors of facetA and facetB to link them appropriately.
    Skips processing if neighbors are already merged or deleted.
*/
void qh_triangulate_mirror(qhT *qh, facetT *facetA, facetT *facetB) {
  facetT *neighbor, *neighborB;
  int neighbor_i, neighbor_n;

  trace3((qh, qh->ferr, 3022, "qh_triangulate_mirror: delete mirrored facets f%d and f%d and link their neighbors\n",
         facetA->id, facetB->id));
  FOREACHneighbor_i_(qh, facetA) {
    neighborB= SETelemt_(facetB->neighbors, neighbor_i, facetT);
    if (neighbor == facetB && neighborB == facetA)
      continue; /* occurs twice */
    else if (neighbor->redundant && neighborB->redundant) { /* also mirrored facets (D5+) */
      if (qh_hasmerge(qh->degen_mergeset, MRGmirror, neighbor, neighborB))
        continue;
    }
    if (neighbor->visible && neighborB->visible) /* previously deleted as mirrored facets */
      continue;
    qh_triangulate_link(qh, facetA, neighbor, facetB, neighborB);
  }
  qh_willdelete(qh, facetA, NULL);
  qh_willdelete(qh, facetB, NULL);
} /* triangulate_mirror */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_null">-</a>

  qh_triangulate_null(qh, facetA)
    remove null facetA from qh_triangulate_facet()
      a null facet has vertex #1 (apex) == vertex #2
  returns:
    adds facetA to ->visible for deletion after qh_update_vertexneighbors
    qh->degen_mergeset contains mirror facets (4-d and up only)
  design:
    since a null facet duplicates the first two vertices, the opposing neighbors absorb the null facet
    if they are already neighbors, the opposing neighbors will be merged (MRGmirror)
*/
void qh_triangulate_null(qhT *qh, facetT *facetA) {
  facetT *neighbor, *otherfacet;

  trace3((qh, qh->ferr, 3023, "qh_triangulate_null: delete null facet f%d\n", facetA->id));
  neighbor= SETfirstt_(facetA->neighbors, facetT);
  otherfacet= SETsecondt_(facetA->neighbors, facetT);
  qh_triangulate_link(qh, facetA, neighbor, facetA, otherfacet);
  qh_willdelete(qh, facetA, NULL);
} /* triangulate_null */

#else /* qh_NOmerge */
void qh_triangulate(qhT *qh) {
  QHULL_UNUSED(qh)
}
#endif /* qh_NOmerge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexintersect">-</a>

  qh_vertexintersect(qh, verticesA, verticesB )
    intersects two vertex sets (inverse id ordered)
    vertexsetA is a temporary set at the top of qh->qhmem.tempstack

  returns:
    replaces vertexsetA with the intersection

  notes:
    only called by qh_neighbor_intersections
    if !qh.QHULLfinished, non-simplicial facets may have f.vertices with extraneous vertices
      cleaned by qh_remove_extravertices in qh_reduce_vertices
    could optimize by overwriting vertexsetA
*/
void qh_vertexintersect(qhT *qh, setT **vertexsetA, setT *vertexsetB) {
  setT *intersection;

  intersection= qh_vertexintersect_new(qh, *vertexsetA, vertexsetB);
  qh_settempfree(qh, vertexsetA);
  *vertexsetA= intersection;
  qh_settemppush(qh, intersection);
} /* vertexintersect */
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexintersect_new">-</a>

  qh_vertexintersect_new(qh, verticesA, verticesB )
    intersects two vertex sets (inverse id ordered)

  returns:
    a new set

  notes:
    called by qh_checkfacet, qh_vertexintersect, qh_rename_sharedvertex, qh_findbest_pinchedvertex, qh_neighbor_intersections
    if !qh.QHULLfinished, non-simplicial facets may have f.vertices with extraneous vertices
       cleaned by qh_remove_extravertices in qh_reduce_vertices
*/
setT *qh_vertexintersect_new(qhT *qh, setT *vertexsetA, setT *vertexsetB) {
  // 创建一个新的空集合来存放交集
  setT *intersection = qh_setnew(qh, qh->hull_dim - 1);
  // 获取 vertexsetA 和 vertexsetB 的元素数组的地址
  vertexT **vertexA = SETaddr_(vertexsetA, vertexT);
  vertexT **vertexB = SETaddr_(vertexsetB, vertexT);

  // 遍历 vertexA 和 vertexB 数组，找到交集
  while (*vertexA && *vertexB) {
    if (*vertexA == *vertexB) {
      // 如果元素在两个集合中都存在，则加入到交集中
      qh_setappend(qh, &intersection, *vertexA);
      vertexA++; vertexB++;
    } else {
      // 比较两个元素的 ID，将较小的元素向前移动一步
      if ((*vertexA)->id > (*vertexB)->id)
        vertexA++;
      else
        vertexB++;
    }
  }
  return intersection;  // 返回计算得到的交集集合
} /* vertexintersect_new */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexneighbors">-</a>

  qh_vertexneighbors(qh)
    for each vertex in qh.facet_list,
      determine its neighboring facets

  returns:
    sets qh.VERTEXneighbors
      nop if qh.VERTEXneighbors already set
      qh_addpoint() will maintain them

  notes:
    assumes all vertex->neighbors are NULL

  design:
    for each facet
      for each vertex
        append facet to vertex->neighbors
*/
void qh_vertexneighbors(qhT *qh /* qh.facet_list */) {
  facetT *facet;
  vertexT *vertex, **vertexp;

  if (qh->VERTEXneighbors)
    return;  // 如果已经设置了顶点邻接信息，直接返回

  // 输出调试信息，标记正在计算顶点邻接面
  trace1((qh, qh->ferr, 1035, "qh_vertexneighbors: determining neighboring facets for each vertex\n"));

  qh->vertex_visit++;  // 递增访问计数
  FORALLfacets {  // 遍历所有的面
    if (facet->visible)  // 如果面被标记为可见，则跳过
      continue;
    FOREACHvertex_(facet->vertices) {  // 遍历面的所有顶点
      if (vertex->visitid != qh->vertex_visit) {
        // 如果顶点的访问标记与当前访问计数不同，则更新顶点的邻接信息
        vertex->visitid = qh->vertex_visit;
        vertex->neighbors = qh_setnew(qh, qh->hull_dim);
      }
      // 将当前面添加到顶点的邻接面集合中
      qh_setappend(qh, &vertex->neighbors, facet);
    }
  }
  qh->VERTEXneighbors = True;  // 设置顶点邻接信息已经计算完毕
} /* vertexneighbors */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexsubset">-</a>

  qh_vertexsubset( vertexsetA, vertexsetB )
    returns True if vertexsetA is a subset of vertexsetB
    assumes vertexsets are sorted

  note:
    empty set is a subset of any other set
*/
boolT qh_vertexsubset(setT *vertexsetA, setT *vertexsetB) {
  // 获取 vertexsetA 和 vertexsetB 的元素数组的地址
  vertexT **vertexA = (vertexT **) SETaddr_(vertexsetA, vertexT);
  vertexT **vertexB = (vertexT **) SETaddr_(vertexsetB, vertexT);

  while (True) {
    if (!*vertexA)
      return True;  // 如果 vertexsetA 已经遍历完毕，则说明是 vertexsetB 的子集
    if (!*vertexB)
      return False;  // 如果 vertexsetB 已经遍历完毕而 vertexsetA 还有元素，则不是子集
    if ((*vertexA)->id > (*vertexB)->id)
      return False;  // 如果 vertexA 的当前元素 ID 大于 vertexB 的当前元素 ID，则不是子集
    if (*vertexA == *vertexB)
      vertexA++;  // 如果两者相等，则继续比较下一个元素
    vertexB++;  // 否则移动 vertexB 的指针
  }
}
    vertexB++;  # 增加 vertexB 变量的值，这里可能是在某种循环或条件判断中对 vertexB 进行递增操作
  }
  return False; /* 避免出现警告 */
} /* vertexsubset */
```