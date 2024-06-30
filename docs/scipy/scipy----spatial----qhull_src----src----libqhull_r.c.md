# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\libqhull_r.c`

```
/*
   <html><pre>  -<a href="qh-qhull_r.htm">qh-qhull_r.htm</a>
   libqhull_r.c
   Quickhull algorithm for convex hulls
   qhull() and top-level routines
   see qh-qhull_r.htm, libqhull_r.h, unix_r.c
   see qhull_ra.h for internal functions
   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/libqhull_r.c#16 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*============= functions in alphabetic order after qhull() =======*/

/*-<a href="qh-qhull_r.htm#TOC">-</a><a name="qhull">-</a>

  qh_qhull(qh)
    compute DIM3 convex hull of qh.num_points starting at qh.first_point
    qh->contains all global options and variables

  returns:
    returns polyhedron
      qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices,

    returns global variables
      qh.hulltime, qh.max_outside, qh.interior_point, qh.max_vertex, qh.min_vertex

    returns precision constants
      qh.ANGLEround, centrum_radius, cos_max, DISTround, MAXabs_coord, ONEmerge

  notes:
    unless needed for output
      qh.max_vertex and qh.min_vertex are max/min due to merges

  see:
    to add individual points to either qh.num_points
      use qh_addpoint()

    if qh.GETarea
      qh_produceoutput() returns qh.totarea and qh.totvol via qh_getarea()

  design:
    record starting time
    initialize hull and partition points
    build convex hull
    unless early termination
      update facet->maxoutside for vertices, coplanar, and near-inside points
    error if temporary sets exist
    record end time
*/

void qh_qhull(qhT *qh) {
  int numoutside;

  qh->hulltime= qh_CPUclock;
  if (qh->RERUN || qh->JOGGLEmax < REALmax/2)
    qh_build_withrestart(qh);
  else {
    qh_initbuild(qh);
    qh_buildhull(qh);
  }
  if (!qh->STOPadd && !qh->STOPcone && !qh->STOPpoint) {
    if (qh->ZEROall_ok && !qh->TESTvneighbors && qh->MERGEexact)
      qh_checkzero(qh, qh_ALL);
    if (qh->ZEROall_ok && !qh->TESTvneighbors && !qh->WAScoplanar) {
      trace2((qh, qh->ferr, 2055, "qh_qhull: all facets are clearly convex and no coplanar points.  Post-merging and check of maxout not needed.\n"));
      qh->DOcheckmax= False;


注释：
    } else {
      qh_initmergesets(qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */);
      // 初始化合并集合，参数为 qh.facet_mergeset, qh.degen_mergeset, qh.vertex_mergeset
      if (qh->MERGEexact || (qh->hull_dim > qh_DIMreduceBuild && qh->PREmerge))
        // 如果设置了精确合并或者（维度大于减少构建的维度并且设置了预合并）
        qh_postmerge(qh, "First post-merge", qh->premerge_centrum, qh->premerge_cos,
             (qh->POSTmerge ? False : qh->TESTvneighbors)); /* calls qh_reducevertices */
        // 调用 qh_postmerge 函数，标记为 "First post-merge"，使用 premerge_centrum 和 premerge_cos 参数，
        // 如果 POSTmerge 为真，则不进行测试顶点邻居（calls qh_reducevertices）
      else if (!qh->POSTmerge && qh->TESTvneighbors)
        // 否则如果未设置 POSTmerge 且测试顶点邻居
        qh_postmerge(qh, "For testing vertex neighbors", qh->premerge_centrum,
             qh->premerge_cos, True);                       /* calls qh_test_vneighbors */
        // 调用 qh_postmerge 函数，标记为 "For testing vertex neighbors"，使用 premerge_centrum 和 premerge_cos 参数，
        // 并测试顶点邻居（calls qh_test_vneighbors）
      if (qh->POSTmerge)
        // 如果设置了 POSTmerge
        qh_postmerge(qh, "For post-merging", qh->postmerge_centrum,
             qh->postmerge_cos, qh->TESTvneighbors);
        // 调用 qh_postmerge 函数，标记为 "For post-merging"，使用 postmerge_centrum 和 postmerge_cos 参数，
        // 并测试顶点邻居
      if (qh->visible_list == qh->facet_list) {            /* qh_postmerge was called */
        // 如果 visible_list 等于 facet_list，说明已经调用了 qh_postmerge
        qh->findbestnew= True;
        // 设置 findbestnew 为真
        qh_partitionvisible(qh, !qh_ALL, &numoutside /* qh.visible_list */);
        // 分割可见性，参数为 !qh_ALL 和 numoutside（注释为 qh.visible_list）
        qh->findbestnew= False;
        // 设置 findbestnew 为假
        qh_deletevisible(qh /* qh.visible_list */);        /* stops at first !f.visible */
        // 删除可见列表中不可见的项，参数为 qh.visible_list，直到遇到第一个不可见的项
        qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
        // 重置列表，参数为假，重置的列表有 qh.visible_list、newvertex_list 和 qh.newfacet_list
      }
      qh_all_vertexmerges(qh, -1, NULL, NULL);
      // 执行所有顶点合并操作，参数为 -1，空指针 NULL
      qh_freemergesets(qh);
      // 释放合并集合资源，参数为 qh
    }
    if (qh->TRACEpoint == qh_IDunknown && qh->TRACElevel > qh->IStracing) {
      // 如果 TRACEpoint 等于 qh_IDunknown 并且 TRACElevel 大于 IStracing
      qh->IStracing= qh->TRACElevel;
      // 设置 IStracing 为 TRACElevel
      qh_fprintf(qh, qh->ferr, 2112, "qh_qhull: finished qh_buildhull and qh_postmerge, start tracing (TP-1)\n");
      // 输出消息到 ferr 流，标记为 2112，消息内容为 "qh_qhull: finished qh_buildhull and qh_postmerge, start tracing (TP-1)"
    }
    if (qh->DOcheckmax){
      // 如果设置了 DOcheckmax
      if (qh->REPORTfreq) {
        // 如果设置了 REPORTfreq
        qh_buildtracing(qh, NULL, NULL);
        // 构建追踪信息，参数为 qh，空指针 NULL
        qh_fprintf(qh, qh->ferr, 8115, "\nTesting all coplanar points.\n");
        // 输出消息到 ferr 流，标记为 8115，消息内容为 "\nTesting all coplanar points.\n"
      }
      qh_check_maxout(qh);
      // 检查最大输出，参数为 qh
    }
    if (qh->KEEPnearinside && !qh->maxoutdone)
      // 如果设置了 KEEPnearinside 且 maxoutdone 为假
      qh_nearcoplanar(qh);
      // 检查附近共面点，参数为 qh
  }
  if (qh_setsize(qh, qh->qhmem.tempstack) != 0) {
    // 如果临时栈不为空
    qh_fprintf(qh, qh->ferr, 6164, "qhull internal error (qh_qhull): temporary sets not empty(%d) at end of Qhull\n",
             qh_setsize(qh, qh->qhmem.tempstack));
    // 输出错误消息到 ferr 流，标记为 6164，消息内容为 "qhull internal error (qh_qhull): temporary sets not empty(%d) at end of Qhull\n"
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    // 调用错误退出函数，参数为 qh，错误类型为 qh_ERRqhull，空指针 NULL
  }
  qh->hulltime= qh_CPUclock - qh->hulltime;
  // 计算凸壳算法运行时间
  qh->QHULLfinished= True;
  // 设置 QHULLfinished 为真
  trace1((qh, qh->ferr, 1036, "Qhull: algorithm completed\n"));
  // 输出追踪消息到 ferr 流，标记为 1036，消息内容为 "Qhull: algorithm completed\n"
} /* qhull */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="addpoint">-</a>

  qh_addpoint(qh, furthest, facet, checkdist )
    add point (usually furthest point) above facet to hull
    if checkdist,
      check that point is above facet.
      if point is not outside of the hull, uses qh_partitioncoplanar()
      assumes that facet is defined by qh_findbestfacet()
    else if facet specified,
      assumes that point is above facet (major damage if below)
    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    returns False if user requested an early termination
      qh.visible_list, newfacet_list, delvertex_list, NEWfacets may be defined
    updates qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices
    clear qh.maxoutdone (will need to call qh_check_maxout() for facet->maxoutside)
    if unknown point, adds a pointer to qh.other_points
      do not deallocate the point's coordinates

  notes:
    called from qh_initbuild, qh_buildhull, and qh_addpoint
    tail recursive call if merged a pinchedvertex due to a duplicated ridge
      no more than qh.num_vertices calls (QH6296)
    assumes point is near its best facet and not at a local minimum of a lens
      distributions.  Use qh_findbestfacet to avoid this case.
    uses qh.visible_list, qh.newfacet_list, qh.delvertex_list, qh.NEWfacets
    if called from a user application after qh_qhull and 'QJ' (joggle),
      facet merging for precision problems is disabled by default

  design:
    exit if qh.STOPadd vertices 'TAn'
    add point to other_points if needed
    if checkdist
      if point not above facet
        partition coplanar point
        exit
    exit if pre STOPpoint requested
    find horizon and visible facets for point
    build cone of new facets to the horizon
    exit if build cone fails due to qh.ONLYgood
    tail recursive call if build cone fails due to pinched vertices
    exit if STOPcone requested
    merge non-convex new facets
    if merge found, many merges, or 'Qf'
       use qh_findbestnew() instead of qh_findbest()
    partition outside points from visible facets
    delete visible facets
    check polyhedron if requested
    exit if post STOPpoint requested
    reset working lists of facets and vertices
*/
boolT qh_addpoint(qhT *qh, pointT *furthest, facetT *facet, boolT checkdist) {
  realT dist, pbalance;
  facetT *replacefacet, *newfacet;
  vertexT *apex;
  boolT isoutside= False;
  int numpart, numpoints, goodvisible, goodhorizon, apexpointid;

  qh->maxoutdone= False;
  // 检查要添加的点是否已知，若未知则将其添加到qh->other_points中
  if (qh_pointid(qh, furthest) == qh_IDunknown)
    qh_setappend(qh, &qh->other_points, furthest);
  // 若facet为空，则输出错误信息并返回
  if (!facet) {
    qh_fprintf(qh, qh->ferr, 6213, "qhull internal error (qh_addpoint): NULL facet.  Need to call qh_findbestfacet first\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);


    // 调用 qh_errexit 函数，如果出错则退出程序，传递相应的错误码和参数
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }


  qh_detmaxoutside(qh);


  // 计算超出点的最大距离
  qh_detmaxoutside(qh);
  if (checkdist) {
    // 查找最远点
    facet= qh_findbest(qh, furthest, facet, !qh_ALL, !qh_ISnewfacets, !qh_NOupper,
                        &dist, &isoutside, &numpart);
    // 将分区数添加到 Zpartition 中
    zzadd_(Zpartition, numpart);
    if (!isoutside) {
      // 如果最远点不在外部集合中，则更新相应的标记，并处理共面点
      zinc_(Znotmax);  /* last point of outsideset is no longer furthest. */
      facet->notfurthest= True;
      qh_partitioncoplanar(qh, furthest, facet, &dist, qh->findbestnew);
      return True;
    }
  }
  // 构建追踪数据
  qh_buildtracing(qh, furthest, facet);
  if (qh->STOPpoint < 0 && qh->furthest_id == -qh->STOPpoint-1) {
    // 如果到达停止点，则更新标记并返回
    facet->notfurthest= True;
    return False;
  }
  // 查找视界
  qh_findhorizon(qh, furthest, facet, &goodvisible, &goodhorizon);
  if (qh->ONLYgood && !qh->GOODclosest && !(goodvisible+goodhorizon)) {
    // 如果仅查找好的点，并且没有找到好的点，则更新标记并重置列表
    zinc_(Znotgood);
    facet->notfurthest= True;
    /* last point of outsideset is no longer furthest.  This is ok
        since all points of the outside are likely to be bad */
    qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
    return True;
  }
  // 构建锥体
  apex= qh_buildcone(qh, furthest, facet, goodhorizon, &replacefacet);
  /* qh.newfacet_list, visible_list, newvertex_list */
  if (!apex) {
    // 如果无法构建锥体，则根据情况返回
    if (qh->ONLYgood)
      return True; /* ignore this furthest point, a good new facet was not found */
    if (replacefacet) {
      // 如果需要替换面，则根据情况进行重试或者报错退出
      if (qh->retry_addpoint++ >= qh->num_vertices) {
        qh_fprintf(qh, qh->ferr, 6296, "qhull internal error (qh_addpoint): infinite loop (%d retries) of merging pinched vertices due to dupridge for point p%d, facet f%d, and %d vertices\n",
          qh->retry_addpoint, qh_pointid(qh, furthest), facet->id, qh->num_vertices);
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
      /* retry qh_addpoint after resolving a dupridge via qh_merge_pinchedvertices */
      return qh_addpoint(qh, furthest, replacefacet, True /* checkdisk */);
    }
    qh->retry_addpoint= 0;
    return True; /* ignore this furthest point, resolved a dupridge by making furthest a coplanar point */
  }
  if (qh->retry_addpoint) {
    // 如果有重试点，则进行相应的处理和统计
    zinc_(Zretryadd);
    zadd_(Zretryaddtot, qh->retry_addpoint);
    zmax_(Zretryaddmax, qh->retry_addpoint);
    qh->retry_addpoint= 0;
  }
  // 记录处理过的点数
  apexpointid= qh_pointid(qh, apex->point);
  zzinc_(Zprocessed);
  if (qh->STOPcone && qh->furthest_id == qh->STOPcone-1) {
    // 如果到达停止锥体，则更新标记并返回
    facet->notfurthest= True;
    return False;  /* visible_list etc. still defined */
  }
  // 初始化合并设置并执行预合并或精确合并
  qh->findbestnew= False;
  if (qh->PREmerge || qh->MERGEexact) {
    qh_initmergesets(qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */);
    qh_premerge(qh, apexpointid, qh->premerge_centrum, qh->premerge_cos /* qh.newfacet_list */);
    if (qh_USEfindbestnew)
      qh->findbestnew= True;
    else {
      // 如果不使用 qh_findbestnew，则遍历新的面，并根据情况更新标记
      FORALLnew_facets {
        if (!newfacet->simplicial) {
          qh->findbestnew= True;  /* use qh_findbestnew instead of qh_findbest*/
          break;
        }
      }
    }
  }else if (qh->BESToutside)
    # 设置标志位，指示要找到最佳的新顶点
    qh->findbestnew= True;
  
  # 如果跟踪级别大于等于4，检查可见面列表的多边形
  if (qh->IStracing >= 4)
    qh_checkpolygon(qh, qh->visible_list);
  
  # 分割可见面，不包括所有的点，更新可见面列表
  qh_partitionvisible(qh, !qh_ALL, &numpoints /* qh.visible_list */);
  
  # 取消设置标志位，不再找到最佳新顶点
  qh->findbestnew= False;
  qh->findbest_notsharp= False;
  
  # 更新平衡计数，计算点的平衡性
  zinc_(Zpbalance);
  pbalance= numpoints - (realT) qh->hull_dim /* assumes all points extreme */
                * (qh->num_points - qh->num_vertices)/qh->num_vertices;
  wadd_(Wpbalance, pbalance);
  wadd_(Wpbalance2, pbalance * pbalance);
  
  # 删除可见面列表中的顶点
  qh_deletevisible(qh /* qh.visible_list */);
  
  # 更新顶点的最大值
  zmax_(Zmaxvertex, qh->num_vertices);
  
  # 设置标志位，指示没有新的面被创建
  qh->NEWfacets= False;
  
  # 如果跟踪级别大于等于4，打印列表和检查多边形
  if (qh->IStracing >= 4) {
    if (qh->num_facets < 200)
      qh_printlists(qh);
    qh_printfacetlist(qh, qh->newfacet_list, NULL, True);
    qh_checkpolygon(qh, qh->facet_list);
  }else if (qh->CHECKfrequently) {
    # 根据频率检查多边形
    if (qh->num_facets < 1000)
      qh_checkpolygon(qh, qh->facet_list);
    else
      qh_checkpolygon(qh, qh->newfacet_list);
  }
  
  # 如果设置了停止点，并且最远的点的ID等于停止点减1，并且顶点合并集的大小大于0，则返回假
  if (qh->STOPpoint > 0 && qh->furthest_id == qh->STOPpoint-1 && qh_setsize(qh, qh->vertex_mergeset) > 0)
    return False;
  
  # 重置列表，包括可见面列表、新顶点列表和新面列表
  qh_resetlists(qh, True, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
  
  # 如果存在面合并集，执行所有顶点合并，并释放合并集
  if (qh->facet_mergeset) {
    /* 面合并发生在顶点合并之后（qh_premerge）和重置列表之后 */
    qh_all_vertexmerges(qh, apexpointid, NULL, NULL);
    qh_freemergesets(qh);
  }
  
  # 如果设置了停止点，并且最远的点的ID等于停止点减1，则返回假
  /* qh_triangulate(qh); to test qh.TRInormals */
  if (qh->STOPpoint > 0 && qh->furthest_id == qh->STOPpoint-1)
    return False;
  
  # 跟踪信息，记录添加点到凸包的平衡信息
  trace2((qh, qh->ferr, 2056, "qh_addpoint: added p%d to convex hull with point balance %2.2g\n",
    qh_pointid(qh, furthest), pbalance));
  
  # 返回真，表示成功添加点到凸包
  return True;
/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="build_withrestart">-</a>

  qh_build_withrestart(qh)
    allow restarts due to qh.JOGGLEmax while calling qh_buildhull()
       qh_errexit always undoes qh_build_withrestart()
    qh.FIRSTpoint/qh.NUMpoints is point array
       it may be moved by qh_joggleinput
*/
void qh_build_withrestart(qhT *qh) {
  int restart;
  vertexT *vertex, **vertexp;

  // 允许在调用qh_buildhull()时由于qh.JOGGLEmax而重启
  qh->ALLOWrestart= True;
  while (True) {
    // 设置长跳转以允许在qh_joggle_restart()中重启
    restart= setjmp(qh->restartexit); /* simple statement for CRAY J916 */
    if (restart) {       /* only from qh_joggle_restart() */
      // 如果发生重启，重置错误码并增加重试次数
      qh->last_errcode= qh_ERRnone;
      zzinc_(Zretry);
      wmax_(Wretrymax, qh->JOGGLEmax);
      /* QH7078 warns about using 'TCn' with 'QJn' */
      // 如果从joggle中断，阻止正常输出
      qh->STOPcone= qh_IDunknown; /* if break from joggle, prevents normal output */
      // 对每个未分区的顶点设置分区标志，避免在qh_freebuild -> qh_delvertex中出错
      FOREACHvertex_(qh->del_vertices) {
        if (vertex->point && !vertex->partitioned)
          vertex->partitioned= True; /* avoid error in qh_freebuild -> qh_delvertex */
      }
    }
    // 如果不是重复运行，并且JOGGLEmax小于REALmax的一半
    if (!qh->RERUN && qh->JOGGLEmax < REALmax/2) {
      // 如果构建计数超过JOGGLEmaxretry，报错并退出
      if (qh->build_cnt > qh_JOGGLEmaxretry) {
        qh_fprintf(qh, qh->ferr, 6229, "qhull input error: %d attempts to construct a convex hull with joggled input.  Increase joggle above 'QJ%2.2g' or modify qh_JOGGLE... parameters in user_r.h\n",
           qh->build_cnt, qh->JOGGLEmax);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
      // 如果已经进行过构建，并且不是重启，则退出循环
      if (qh->build_cnt && !restart)
        break;
    }else if (qh->build_cnt && qh->build_cnt >= qh->RERUN)
      break;
    // 重置STOPcone标志，释放构建的数据结构
    qh->STOPcone= 0;
    qh_freebuild(qh, True);  /* first call is a nop */
    // 增加构建计数，并初始化选项长度
    qh->build_cnt++;
    if (!qh->qhull_optionsiz)
      // 计算选项字符串的长度，注意64位警告
      qh->qhull_optionsiz= (int)strlen(qh->qhull_options);   /* WARN64 */
    else {
      // 设置选项字符串结束标志，并指定选项行的长度
      qh->qhull_options[qh->qhull_optionsiz]= '\0';
      qh->qhull_optionlen= qh_OPTIONline;  /* starts a new line */
    }
    // 添加运行编号选项
    qh_option(qh, "_run", &qh->build_cnt, NULL);
    // 如果是最后一次运行，则设置追踪标志
    if (qh->build_cnt == qh->RERUN) {
      qh->IStracing= qh->TRACElastrun;  /* duplicated from qh_initqhull_globals */
      // 如果存在点追踪或距离追踪小于REALmax的一半，设置追踪级别
      if (qh->TRACEpoint != qh_IDnone || qh->TRACEdist < REALmax/2 || qh->TRACEmerge) {
        qh->TRACElevel= (qh->IStracing? qh->IStracing : 3);
        qh->IStracing= 0;
      }
      qh->qhmem.IStracing= qh->IStracing;
    }
    // 如果JOGGLEmax小于REALmax的一半，执行joggleinput过程
    if (qh->JOGGLEmax < REALmax/2)
      qh_joggleinput(qh);
    // 初始化构建相关数据结构
    qh_initbuild(qh);
    // 构建凸包
    qh_buildhull(qh);
    // 如果JOGGLEmax小于REALmax的一半，并且没有合并，则检查凸包的凸性
    if (qh->JOGGLEmax < REALmax/2 && !qh->MERGING)
      qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }
  // 禁止重启标志
  qh->ALLOWrestart= False;
} /* qh_build_withrestart */
  # 返回具有 qh.newfacet_list 和 qh.first_newfacet（f.id）的锥体的顶点
  # 如果 qh.ONLYgood 并且没有好的 facets，则返回 NULL
  # 如果合并挤压顶点可以解决 dupridge，则返回 NULL 和 retryfacet
  # 一个水平顶点几乎与另一个顶点相邻
  # 将重试 qh_addpoint
  # 如果通过使最远点成为共面点解决了 dupridge，则返回 NULL
  # 最远点几乎与现有顶点相邻
  # 如果通过合并 facets 解决了 dupridge，则更新 qh.degen_mergeset（MRGridge）
  # 更新 qh.newfacet_list、visible_list、newvertex_list
  # 更新 qh.facet_list、vertex_list、num_facets、num_vertices

  # 注释:
  # 由 qh_addpoint 调用
  # 参见 qh_triangulate，它在后处理中对非单纯 facets 进行三角化

  # 设计:
  # 为点到水平面创建新的 facets
  # 计算平衡统计数据
  # 为点创建超平面
  # 如果 qh.ONLYgood 并且不是 good 的话就退出（qh_buildcone_onlygood）
  # 匹配相邻的新 facets
  # 如果有 dupridges
  #   如果 !qh.IGNOREpinched 并且 dupridge 通过共面最远点解决则退出
  #   如果 !qh.IGNOREpinched 并且 dupridge 通过 qh_buildcone_mergepinched 解决则重试 qh_buildcone
  #   否则通过合并 facets 解决 dupridges
  # 更新顶点邻居并删除内部顶点
vertexT *qh_buildcone(qhT *qh, pointT *furthest, facetT *facet, int goodhorizon, facetT **retryfacet) {
  vertexT *apex;
  realT newbalance;
  int numnew;

  // 初始化重试面为NULL
  *retryfacet= NULL;
  // 记录第一个新面的ID，用于统计新面的数量
  qh->first_newfacet= qh->facet_id;
  // 设置标志，用于后续确定是否需要附加新面
  qh->NEWtentative= (qh->MERGEpinched || qh->ONLYgood); /* cleared by qh_attachnewfacets or qh_resetlists */
  
  // 创建新的凸包面，返回顶点
  apex= qh_makenewfacets(qh, furthest /* qh.newfacet_list visible_list, attaches new facets if !qh.NEWtentative */);
  // 计算新面的数量
  numnew= (int)(qh->facet_id - qh->first_newfacet);
  // 计算新面的平衡度量
  newbalance= numnew - (realT)(qh->num_facets - qh->num_visible) * qh->hull_dim / qh->num_vertices;
  
  /* newbalance statistics updated below if the new facets are accepted */

  // 如果只处理优化的情况
  if (qh->ONLYgood) { /* qh.MERGEpinched is false by QH6362 */
    // 如果无法成功创建优化凸包，设置当前面不是最远面，返回NULL
    if (!qh_buildcone_onlygood(qh, apex, goodhorizon /* qh.newfacet_list */)) {
      facet->notfurthest= True;
      return NULL;
    }
  } else if(qh->MERGEpinched) {
    // 如果启用了MERGEpinched选项，并且没有禁用MERGEpinched
#ifndef qh_NOmerge
    // 尝试使用MERGEpinched选项创建凸包，可能会返回需要重试的面
    if (qh_buildcone_mergepinched(qh, apex, facet, retryfacet /* qh.newfacet_list */))
      return NULL;
#else
    // 输出错误信息，因为禁用了MERGEpinched选项
    qh_fprintf(qh, qh->ferr, 6375, "qhull option error (qh_buildcone): option 'Q14' (qh.MERGEpinched) is not available due to qh_NOmerge\n");
    // 退出程序，因为MERGEpinched选项禁用了，无法继续
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
#endif
  } else {
    // 如果不是仅处理优化的情况，也不是启用了MERGEpinched选项
    // 将新的面连接到地平线
    qh_matchnewfacets(qh); /* ignore returned value.  qh_forcedmerges will merge dupridges if any */
    // 创建新的平面
    qh_makenewplanes(qh /* qh.newfacet_list */);
    // 更新顶点的相邻关系
    qh_update_vertexneighbors_cone(qh);
  }
  
  // 更新新面的平衡度量
  wadd_(Wnewbalance, newbalance);
  // 更新平衡度量的平方值
  wadd_(Wnewbalance2, newbalance * newbalance);
  // 输出跟踪信息，记录创建了多少新面，以及新面的平衡度量
  trace2((qh, qh->ferr, 2067, "qh_buildcone: created %d newfacets for p%d(v%d) new facet balance %2.2g\n",
    numnew, qh_pointid(qh, furthest), apex->id, newbalance));
  // 返回顶点，表示成功创建了凸包
  return apex;
} /* buildcone */
    # 如果检测到匹配的重复结构，并且使用了广泛的合并（qh_RATIOtrypinched）
      # 如果顶点被压紧（即几乎相邻）
        # 删除新面的锥体
        # 删除顶点并重置面列表
        # 如果共面，压紧的顶点
          # 将顶点分割为共面点
        # 否则
           # 重复地合并最近的一对压紧顶点和随后的面合并
        # 返回 True
      # 否则
        # MRGridge优于顶点合并，但可能会报告错误
    # 附加新的面
    # 为点创建超平面
    # 更新顶点邻居并删除内部顶点
/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="buildcone_onlygood">-</a>

  qh_buildcone_onlygood(qh, apex, goodhorizon )
    build cone of good, new facets from apex and its qh.newfacet_list to the horizon
    goodhorizon is count of good, horizon facets from qh_find_horizon

  returns:
    False if a f.good facet or a qh.GOODclosest facet is not found
    updates qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices

  notes:
    called from qh_buildcone
    QH11030 FIX: Review effect of qh.GOODclosest on qh_buildcone_onlygood ('Qg').  qh_findgood preserves old value if didn't find a good facet.  See qh_findgood_all for disabling

  design:
    make hyperplanes for point
    if qh_findgood fails to find a f.good facet or a qh.GOODclosest facet
      delete cone of new facets
      return NULL (ignores apex)
    else
      attach cone to horizon
      match neighboring new facets
*/
boolT qh_buildcone_onlygood(qhT *qh, vertexT *apex, int goodhorizon) {

这段代码定义了函数 `qh_buildcone_onlygood`，用于构建从顶点 apex 开始到视图边界的优良新面的锥体。


  facetT *newfacet, *nextfacet;

  qh_makenewplanes(qh /* qh.newfacet_list */);

调用 `qh_makenewplanes` 函数为顶点创建新的超平面。


  if(qh_findgood(qh, qh->newfacet_list, goodhorizon) == 0) {

检查调用 `qh_findgood` 函数是否找到了足够数量的优良新面。如果没有找到优良新面或最近的优良面，返回值为 0。


      delete cone of new facets
      return NULL (ignores apex)

如果没有找到优良新面或最近的优良面，则删除所有新面构成的锥体，并返回 NULL，表示忽略当前顶点 apex。


    else {
      attach cone to horizon
      match neighboring new facets

如果找到了优良新面，将这些新面连接到视图边界，并匹配相邻的新面。



这是函数的结尾，没有额外的代码需要注释。
    # 如果没有最接近点 GOODclosest
    if (!qh->GOODclosest) {
      # 遍历 newfacet_list 中的每个 newfacet，直到 newfacet 为 NULL 或者 newfacet->next 为 NULL
      for (newfacet=qh->newfacet_list; newfacet && newfacet->next; newfacet= nextfacet) {
        # 将 newfacet->next 赋值给 nextfacet
        nextfacet= newfacet->next;
        # 调用 qh_delfacet 删除 newfacet
        qh_delfacet(qh, newfacet);
      }
      # 调用 qh_delvertex 删除顶点 apex
      qh_delvertex(qh, apex);
      # 调用 qh_resetlists 重置列表（不生成统计信息），重置可见列表和新顶点列表，以及 newfacet_list
      qh_resetlists(qh, False /*no stats*/, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
      # 增加 Znotgoodnew 计数
      zinc_(Znotgoodnew);
      /* !good outside points dropped from hull */
      # 返回 False，表示从凸壳中删除了不好的外部点
      return False;
    }
  }
  # 将 qh.visible_list 中的新面附加到凸包
  qh_attachnewfacets(qh /* qh.visible_list */);
  # 匹配新的面，忽略返回值，qh_forcedmerges 会合并重复的边缘（如果有的话）
  qh_matchnewfacets(qh); /* ignore returned value.  qh_forcedmerges will merge dupridges if any */
  # 更新顶点邻居关系的锥形
  qh_update_vertexneighbors_cone(qh);
  # 返回 True，表示操作成功完成
  return True;
/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="buildhull">-</a>

  qh_buildhull(qh)
    construct a convex hull by adding outside points one at a time

  returns:

  notes:
    may be called multiple times
    checks facet and vertex lists for incorrect flags
    to recover from STOPcone, call qh_deletevisible and qh_resetlists

  design:
    check visible facet and newfacet flags
    check newfacet vertex flags and qh.STOPcone/STOPpoint
    for each facet with a furthest outside point
      add point to facet
      exit if qh.STOPcone or qh.STOPpoint requested
    if qh.NARROWhull for initial simplex
      partition remaining outside points to coplanar sets
*/
void qh_buildhull(qhT *qh) {
  facetT *facet;
  pointT *furthest;
  vertexT *vertex;
  int id;

  trace1((qh, qh->ferr, 1037, "qh_buildhull: start build hull\n"));
  // 遍历所有的面，检查是否有标记错误的面
  FORALLfacets {
    if (facet->visible || facet->newfacet) {
      // 如果面标记为可见或新面，输出错误信息并退出
      qh_fprintf(qh, qh->ferr, 6165, "qhull internal error (qh_buildhull): visible or new facet f%d in facet list\n",
                   facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
  }
  // 遍历所有的顶点，检查是否有标记错误的顶点或者是否需要停止添加点
  FORALLvertices {
    if (vertex->newfacet) {
      // 如果顶点标记为新顶点，输出错误信息并退出
      qh_fprintf(qh, qh->ferr, 6166, "qhull internal error (qh_buildhull): new vertex f%d in vertex list\n",
                   vertex->id);
      qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    // 获取顶点的标识号
    id= qh_pointid(qh, vertex->point);
    // 如果遇到需要停止的点或锥体，输出信息并退出
    if ((qh->STOPpoint>0 && id == qh->STOPpoint-1) ||
        (qh->STOPpoint<0 && id == -qh->STOPpoint-1) ||
        (qh->STOPcone>0 && id == qh->STOPcone-1)) {
      trace1((qh, qh->ferr, 1038,"qh_buildhull: stop point or cone P%d in initial hull\n", id));
      return;
    }
  }
  // 初始化 facet_next 为 facet_list，用于处理每个面
  qh->facet_next= qh->facet_list;      /* advance facet when processed */
  // 循环添加最远的外部点直到完成凸壳
  while ((furthest= qh_nextfurthest(qh, &facet))) {
    // 如果需要停止添加点，输出信息并退出
    qh->num_outside--;  /* if ONLYmax, furthest may not be outside */
    if (qh->STOPadd>0 && (qh->num_vertices - qh->hull_dim - 1 >= qh->STOPadd - 1)) {
      trace1((qh, qh->ferr, 1059, "qh_buildhull: stop after adding %d vertices\n", qh->STOPadd-1));
      return;
    }
    // 将最远的外部点添加到面中
    if (!qh_addpoint(qh, furthest, facet, qh->ONLYmax))
      break;
  }
  // 如果为 NARROWhull 模式，将剩余的外部点移动到共面点集合中
  if (qh->NARROWhull)
    qh_outcoplanar(qh /* facet_list */ );
  // 检查是否所有的外部点都已经处理完毕
  if (qh->num_outside && !furthest) {
    qh_fprintf(qh, qh->ferr, 6167, "qhull internal error (qh_buildhull): %d outside points were never processed.\n", qh->num_outside);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 输出信息，凸壳构建完成
  trace1((qh, qh->ferr, 1039, "qh_buildhull: completed the hull construction\n"));
} /* buildhull */
    # 如果 !furthest 条件成立，则打印进度消息
    if !furthest, prints progress message

  # 返回：
  #   qh.lastreport, lastcpu, lastfacets, lastmerges, lastplanes, lastdist 用于跟踪进度
  #   更新 qh.furthest_id （如果 furthest 为 NULL，则设置为 -3）
  #   在溢出时重置 visit_id 和 vertex_visit
  #
  # 参见：
  #   qh_tracemerging()
  #
  # 设计：
  #   如果 !furthest
  #     打印进度消息并退出
  #   如果是 'TFn' 迭代
  #     打印进度消息
  #   否则如果在跟踪状态
  #     跟踪最远点和凸面
  #   在可能发生溢出时重置 qh.visit_id 和 qh.vertex_visit
  #   设置 qh.furthest_id 以进行跟踪
  returns:
    tracks progress with qh.lastreport, lastcpu, lastfacets, lastmerges, lastplanes, lastdist
    updates qh.furthest_id (-3 if furthest is NULL)
    also resets visit_id, vertext_visit on wrap around

  see:
    qh_tracemerging()

  design:
    if !furthest
      print progress message
      exit
    if 'TFn' iteration
      print progress message
    else if tracing
      trace furthest point and facet
    reset qh.visit_id and qh.vertex_visit if overflow may occur
    set qh.furthest_id for tracing
/*
void qh_buildtracing(qhT *qh, pointT *furthest, facetT *facet) {
  realT dist= 0;  // 定义实数类型的变量 dist，并初始化为 0
  double cpu;  // 定义双精度浮点数变量 cpu
  int total, furthestid;  // 定义整数变量 total 和 furthestid
  time_t timedata;  // 定义时间类型变量 timedata
  struct tm *tp;  // 定义指向 tm 结构体的指针 tp
  vertexT *vertex;  // 定义顶点类型的指针变量 vertex

  qh->old_randomdist= qh->RANDOMdist;  // 将 qh->RANDOMdist 赋值给 qh->old_randomdist
  qh->RANDOMdist= False;  // 将 qh->RANDOMdist 设为 False
  if (!furthest) {  // 如果 furthest 为 NULL
    time(&timedata);  // 获取当前时间
    tp= localtime(&timedata);  // 将时间转换为本地时间
    cpu= (double)qh_CPUclock - (double)qh->hulltime;  // 计算 CPU 时间
    cpu /= (double)qh_SECticks;  // 将 CPU 时间转换为秒数
    total= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);  // 计算 total
    qh_fprintf(qh, qh->ferr, 8118, "\n\
At %02d:%02d:%02d & %2.5g CPU secs, qhull has created %d facets and merged %d.\n\
 The current hull contains %d facets and %d vertices.  Last point was p%d\n",
      tp->tm_hour, tp->tm_min, tp->tm_sec, cpu, qh->facet_id -1,
      total, qh->num_facets, qh->num_vertices, qh->furthest_id);
    return;  // 返回
  }
  furthestid= qh_pointid(qh, furthest);  // 获取 furthest 的标识符

#ifndef qh_NOtrace
  if (qh->TRACEpoint == furthestid) {  // 如果 TRACEpoint 等于 furthestid
    trace1((qh, qh->ferr, 1053, "qh_buildtracing: start trace T%d for point TP%d above facet f%d\n", qh->TRACElevel, furthestid, facet->id));  // 输出追踪信息
    qh->IStracing= qh->TRACElevel;  // 设置追踪级别
    qh->qhmem.IStracing= qh->TRACElevel;  // 设置追踪级别
  } else if (qh->TRACEpoint != qh_IDnone && qh->TRACEdist < REALmax/2) {  // 否则如果 TRACEpoint 不是 qh_IDnone 并且 TRACEdist 小于 REALmax 的一半
    qh->IStracing= 0;  // 取消追踪
    qh->qhmem.IStracing= 0;  // 取消追踪
  }
#endif

  if (qh->REPORTfreq && (qh->facet_id-1 > qh->lastreport + (unsigned int)qh->REPORTfreq)) {  // 如果 REPORTfreq 存在并且 facet_id-1 大于 lastreport 加上 REPORTfreq
    qh->lastreport= qh->facet_id-1;  // 更新 lastreport
    time(&timedata);  // 获取当前时间
    tp= localtime(&timedata);  // 将时间转换为本地时间
    cpu= (double)qh_CPUclock - (double)qh->hulltime;  // 计算 CPU 时间
    cpu /= (double)qh_SECticks;  // 将 CPU 时间转换为秒数
    total= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);  // 计算 total
    zinc_(Zdistio);  // 增加 Zdistio 计数
    qh_distplane(qh, furthest, facet, &dist);  // 计算距离平面的距离
    qh_fprintf(qh, qh->ferr, 8119, "\n\
At %02d:%02d:%02d & %2.5g CPU secs, qhull has created %d facets and merged %d.\n\
 The current hull contains %d facets and %d vertices.  There are %d\n\
 outside points.  Next is point p%d(v%d), %2.2g above f%d.\n",
      tp->tm_hour, tp->tm_min, tp->tm_sec, cpu, qh->facet_id -1,
      total, qh->num_facets, qh->num_vertices, qh->num_outside+1,
      furthestid, qh->vertex_id, dist, getid_(facet));
  } else if (qh->IStracing >=1) {  // 否则如果 IStracing 大于等于 1
    cpu= (double)qh_CPUclock - (double)qh->hulltime;  // 计算 CPU 时间
    cpu /= (double)qh_SECticks;  // 将 CPU 时间转换为秒数
    qh_distplane(qh, furthest, facet, &dist);  // 计算距离平面的距离
    qh_fprintf(qh, qh->ferr, 1049, "qh_addpoint: add p%d(v%d) %2.2g above f%d to hull of %d facets, %d merges, %d outside at %4.4g CPU secs.  Previous p%d(v%d) delta %4.4g CPU, %d facets, %d merges, %d hyperplanes, %d distplanes, %d retries\n",
      furthestid, qh->vertex_id, dist, getid_(facet), qh->num_facets, zzval_(Ztotmerge), qh->num_outside+1, cpu, qh->furthest_id, qh->vertex_id - 1,
      cpu - qh->lastcpu, qh->num_facets - qh->lastfacets,  zzval_(Ztotmerge) - qh->lastmerges, zzval_(Zsetplane) - qh->lastplanes, zzval_(Zdistplane) - qh->lastdist, qh->retry_addpoint);
    qh->lastcpu= cpu;  // 更新 lastcpu
    qh->lastfacets= qh->num_facets;  // 更新 lastfacets
    qh->lastmerges= zzval_(Ztotmerge);  // 更新 lastmerges
    `
    # 设置 qh 对象的 lastplanes 属性为 Zsetplane 的值
    qh->lastplanes= zzval_(Zsetplane);
    # 设置 qh 对象的 lastdist 属性为 Zdistplane 的值
    qh->lastdist= zzval_(Zdistplane);
    
    
    
    # 更新 Zvisit2max 统计，参数为 qh->visit_id 的一半
    zmax_(Zvisit2max, (int)qh->visit_id/2);
    # 检查 qh->visit_id 是否超过 31 位的无符号整数范围
    if (qh->visit_id > (unsigned int) INT_MAX) { /* 31 bits */
        # 增加 Zvisit 统计计数
        zinc_(Zvisit);
        # 检查 qh 对象的 facet_list 是否符合要求，否则输出错误信息
        if (!qh_checklists(qh, qh->facet_list)) {
          qh_fprintf(qh, qh->ferr, 6370, "qhull internal error: qh_checklists failed on reset of qh.visit_id %u\n", qh->visit_id);
          qh_errexit(qh, qh_ERRqhull, NULL, NULL);
        }
        # 重置 qh 对象的 visit_id 属性为 0
        qh->visit_id= 0;
        # 遍历所有的 facets，将它们的 visitid 属性设置为 0
        FORALLfacets
          facet->visitid= 0;
    }
    
    
    
    # 更新 Zvvisit2max 统计，参数为 qh->vertex_visit 的一半
    zmax_(Zvvisit2max, (int)qh->vertex_visit/2);
    # 检查 qh->vertex_visit 是否超过 31 位的无符号整数范围
    if (qh->vertex_visit > (unsigned int) INT_MAX) { /* 31 bits */
        # 增加 Zvvisit 统计计数
        zinc_(Zvvisit);
        # 如果 qh->visit_id 非零且 qh 对象的 facet_list 不符合要求，则输出错误信息
        if (qh->visit_id && !qh_checklists(qh, qh->facet_list)) {
          qh_fprintf(qh, qh->ferr, 6371, "qhull internal error: qh_checklists failed on reset of qh.vertex_visit %u\n", qh->vertex_visit);
          qh_errexit(qh, qh_ERRqhull, NULL, NULL);
        }
        # 重置 qh 对象的 vertex_visit 属性为 0
        qh->vertex_visit= 0;
        # 遍历所有的 vertices，将它们的 visitid 属性设置为 0
        FORALLvertices
          vertex->visitid= 0;
    }
    
    
    
    # 设置 qh 对象的 furthest_id 属性为 furthestid 的值
    qh->furthest_id= furthestid;
    # 恢复 qh 对象的 RANDOMdist 属性为 old_randomdist 的值
    qh->RANDOMdist= qh->old_randomdist;
/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="findhorizon">-</a>

  qh_findhorizon(qh, point, facet, goodvisible, goodhorizon )
    给定一个可见的 facet，找到点的视图边界和可见的 facets
    对于所有 facets，!facet->visible

  returns:
    返回 qh.visible_list/num_visible，包含所有可见的 facets
      将可见的 facets 标记为 ->visible
    更新 good visible 和 good horizon facets 的计数
    更新 qh.max_outside, qh.max_vertex, facet->maxoutside

  see:
    类似于 qh_delpoint()

  design:
    将 facet 移动到 qh.facet_list 的末尾作为 qh.visible_list
    对于所有可见的 facets
     对于每个未访问的邻居 facet
       计算点到邻居 facet 的距离
       如果点在邻居 facet 上方
         将邻居 facet 移动到 qh.visible_list 的末尾
       否则如果点与邻居 facet 共面
         更新 qh.max_outside, qh.max_vertex, neighbor->maxoutside
         标记邻居 facet 为共面（稍后将创建一个同样的循环）
         更新视图边界的统计信息
*/
void qh_findhorizon(qhT *qh, pointT *point, facetT *facet, int *goodvisible, int *goodhorizon) {
  facetT *neighbor, **neighborp, *visible;
  int numhorizon= 0, coplanar= 0;
  realT dist;

  trace1((qh, qh->ferr, 1040, "qh_findhorizon: find horizon for point p%d facet f%d\n",qh_pointid(qh, point),facet->id));
  *goodvisible= *goodhorizon= 0;
  zinc_(Ztotvisible);
  qh_removefacet(qh, facet);  /* 将 facet 从 visible_list 中移除，放在 qh->facet_list 的末尾 */
  qh_appendfacet(qh, facet);   /* 将 facet 添加到 qh.facet_list 的末尾 */
  qh->num_visible= 1;          /* 设置当前可见 facets 数为 1 */
  if (facet->good)
    (*goodvisible)++;          /* 如果 facet 是 good 的，增加 good visible facets 的计数 */
  qh->visible_list= facet;     /* 设置当前的 visible_list 为 facet */
  facet->visible= True;        /* 标记 facet 为可见 */
  facet->f.replace= NULL;      /* 清空 facet 的替换标记 */
  if (qh->IStracing >=4)
    qh_errprint(qh, "visible", facet, NULL, NULL, NULL);  /* 如果启用跟踪，打印 facet 的可见信息 */
  qh->visit_id++;              /* 增加访问 ID */

  FORALLvisible_facets {       /* 遍历所有可见的 facets */
    if (visible->tricoplanar && !qh->TRInormals) {
      qh_fprintf(qh, qh->ferr, 6230, "qhull internal error (qh_findhorizon): does not work for tricoplanar facets.  Use option 'Q11'\n");
      qh_errexit(qh, qh_ERRqhull, visible, NULL);  /* 如果是三共面的 facets，报错退出 */
    }
    if (qh_setsize(qh, visible->neighbors) == 0) {
      qh_fprintf(qh, qh->ferr, 6295, "qhull internal error (qh_findhorizon): visible facet f%d does not have neighbors\n", visible->id);
      qh_errexit(qh, qh_ERRqhull, visible, NULL);  /* 如果可见 facet 没有邻居，报错退出 */
    }
    visible->visitid= qh->visit_id;
    # 设置当前可见面的访问标识为当前全局访问标识
    FOREACHneighbor_(visible) {
      # 遍历当前可见面的邻居面
      if (neighbor->visitid == qh->visit_id)
        # 如果邻居面的访问标识与当前全局访问标识相同，则跳过
        continue;
      # 将邻居面的访问标识设为当前全局访问标识
      neighbor->visitid= qh->visit_id;
      # 增加可见面的数量统计
      zzinc_(Znumvisibility);
      # 计算当前面与邻居面的距离
      qh_distplane(qh, point, neighbor, &dist);
      # 如果距离大于当前最小可见距离
      if (dist > qh->MINvisible) {
        # 增加总可见面的数量统计
        zinc_(Ztotvisible);
        # 将邻居面从可见面列表中移除并重新添加到列表末尾
        qh_removefacet(qh, neighbor);  /* append to end of qh->visible_list */
        qh_appendfacet(qh, neighbor);
        # 标记邻居面为可见
        neighbor->visible= True;
        # 清空邻居面的替换标志
        neighbor->f.replace= NULL;
        # 增加可见面的计数器
        qh->num_visible++;
        # 如果邻居面是好的（good），增加好的可见面计数器
        if (neighbor->good)
          (*goodvisible)++;
        # 如果设置了追踪级别大于等于4，则打印相关信息
        if (qh->IStracing >=4)
          qh_errprint(qh, "visible", neighbor, NULL, NULL, NULL);
      }else {
        # 如果距离大于等于负的最大共面距离
        if (dist >= -qh->MAXcoplanar) {
          # 标记邻居面为共面地平线
          neighbor->coplanarhorizon= True;
          # 增加共面地平线的数量统计
          zzinc_(Zcoplanarhorizon);
          # 重新启动共面调整
          qh_joggle_restart(qh, "coplanar horizon");
          # 增加共面计数器
          coplanar++;
          # 如果启用了合并操作
          if (qh->MERGING) {
            # 如果距离大于0，则更新最大外部距离和顶点距离的最大值
            if (dist > 0) {
              maximize_(qh->max_outside, dist);
              maximize_(qh->max_vertex, dist);
#if qh_MAXoutside
              maximize_(neighbor->maxoutside, dist);
#endif
            }else
              minimize_(qh->min_vertex, dist);  /* due to merge later */


          }
          // 打印跟踪信息，表明点 p%d 与边界 f%d 共面，距离为 dist，小于 qh->MINvisible(%2.7g)
          trace2((qh, qh->ferr, 2057, "qh_findhorizon: point p%d is coplanar to horizon f%d, dist=%2.7g < qh->MINvisible(%2.7g)\n",
              qh_pointid(qh, point), neighbor->id, dist, qh->MINvisible));
        }else
          neighbor->coplanarhorizon= False;
        // 增加统计量 Ztothorizon
        zinc_(Ztothorizon);
        numhorizon++;
        // 如果邻居点是好的，则增加 goodhorizon 统计量
        if (neighbor->good)
          (*goodhorizon)++;
        // 如果正在跟踪级别高于等于4，打印更详细的错误信息
        if (qh->IStracing >=4)
          qh_errprint(qh, "horizon", neighbor, NULL, NULL, NULL);
      }
    }
  }
}
// 如果没有找到任何水平面，执行 qh_joggle_restart 以重启 qhull，同时输出相关错误信息
if (!numhorizon) {
  qh_joggle_restart(qh, "empty horizon");
  qh_fprintf(qh, qh->ferr, 6168, "qhull topology error (qh_findhorizon): empty horizon for p%d.  It was above all facets.\n", qh_pointid(qh, point));
  // 如果面的数量小于100，打印面列表
  if (qh->num_facets < 100) {
    qh_printfacetlist(qh, qh->facet_list, NULL, True);
  }
  // 出现拓扑错误，退出程序
  qh_errexit(qh, qh_ERRtopology, NULL, NULL);
}
// 打印跟踪信息，显示找到的水平面数量及其性质
trace1((qh, qh->ferr, 1041, "qh_findhorizon: %d horizon facets(good %d), %d visible(good %d), %d coplanar\n",
       numhorizon, *goodhorizon, qh->num_visible, *goodvisible, coplanar));
// 如果正在跟踪级别高于等于4并且面的数量小于100，打印面列表
if (qh->IStracing >= 4 && qh->num_facets < 100)
  qh_printlists(qh);
} /* findhorizon */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="joggle_restart">-</a>

  qh_joggle_restart(qh, reason )
    if joggle ('QJn') and not merging, restart on precision and topology errors
*/
void qh_joggle_restart(qhT *qh, const char *reason) {

  // 如果 JOGGLEmax 小于 REALmax 的一半
  if (qh->JOGGLEmax < REALmax/2) {
    // 如果允许重启，并且不在预合并和精确合并状态下
    if (qh->ALLOWrestart && !qh->PREmerge && !qh->MERGEexact) {
      // 打印跟踪信息，表明因为某个原因需要重新启动 qhull
      trace0((qh, qh->ferr, 26, "qh_joggle_restart: qhull restart because of %s\n", reason));
      /* May be called repeatedly if qh->ALLOWrestart */
      // 使用 longjmp 跳转到重启出口点，错误代码为 qh_ERRprec
      longjmp(qh->restartexit, qh_ERRprec);
    }
  }
} /* qh_joggle_restart */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="nextfurthest">-</a>

  qh_nextfurthest(qh, visible )
    returns next furthest point and visible facet for qh_addpoint()
    starts search at qh.facet_next

  returns:
    removes furthest point from outside set
    NULL if none available
    advances qh.facet_next over facets with empty outside sets

  design:
    for each facet from qh.facet_next
      if empty outside set
        advance qh.facet_next
      else if qh.NARROWhull
        determine furthest outside point
        if furthest point is not outside
          advance qh.facet_next(point will be coplanar)
    remove furthest point from outside set
*/
pointT *qh_nextfurthest(qhT *qh, facetT **visible) {
  facetT *facet;
  int size, idx, loopcount= 0;
  realT randr, dist;
  pointT *furthest;

  while ((facet= qh->facet_next) != qh->facet_tail) {
    // 检查 facet 是否为空或者循环次数是否超过了 qh->num_facets
    if (!facet || loopcount++ > qh->num_facets) {
      // 如果是，输出错误消息到 qh->ferr
      qh_fprintf(qh, qh->ferr, 6406, "qhull internal error (qh_nextfurthest): null facet or infinite loop detected for qh.facet_next f%d facet_tail f%d\n",
        getid_(facet), getid_(qh->facet_tail));
      // 打印当前数据结构列表的信息
      qh_printlists(qh);
      // 终止程序，显示错误信息和相关的 facet 和 qh->facet_tail
      qh_errexit2(qh, qh_ERRqhull, facet, qh->facet_tail);
    }
    // 检查 facet 是否有 outsideset
    if (!facet->outsideset) {
      // 如果没有，继续下一个 facet
      qh->facet_next= facet->next;
      continue;
    }
    // 获取 facet->outsideset 的大小，并将结果存储在 size 中
    SETreturnsize_(facet->outsideset, size);
    // 如果 size 为 0
    if (!size) {
      // 释放 facet->outsideset 的内存空间
      qh_setfree(qh, &facet->outsideset);
      // 继续下一个 facet
      qh->facet_next= facet->next;
      continue;
    }
    // 如果启用了 NARROWhull 模式
    if (qh->NARROWhull) {
      // 如果 facet->notfurthest 为真，调用 qh_furthestout 处理
      if (facet->notfurthest)
        qh_furthestout(qh, facet);
      // 获取 facet->outsideset 中最远的点，并将其转换为 pointT 类型
      furthest= (pointT *)qh_setlast(facet->outsideset);
/* 如果定义了 qh_COMPUTEfurthest 宏 */
if (qh_COMPUTEfurthest) {
  /* 计算到 facet 最远点的距离 */
  qh_distplane(qh, furthest, facet, &dist);
  /* 统计 COMPUTEfurthest 操作次数 */
  zinc_(Zcomputefurthest);
} else {
  /* 否则直接使用 facet->furthestdist 作为距离 */
  dist = facet->furthestdist;
}

/* 如果距离小于 MINoutside */
if (dist < qh->MINoutside) {
  /* 如果余下的外部点与 facet 共面，则设置下一个处理的 facet 并继续 */
  qh->facet_next = facet->next;
  continue;
}

/* 如果非 RANDOMoutside 和非 VIRTUALmemory 模式 */
if (!qh->RANDOMoutside && !qh->VIRTUALmemory) {
  /* 如果设置了 PICKfurthest 标志 */
  if (qh->PICKfurthest) {
    /* 选择下一个最远点 */
    qh_furthestnext(qh /* qh.facet_list */);
    facet = qh->facet_next;
  }
  /* 设置 visible 指针为当前 facet */
  *visible = facet;
  /* 返回 outsideset 中的最后一个点，并从集合中删除 */
  return ((pointT *)qh_setdellast(facet->outsideset));
}

/* 如果是 RANDOMoutside 模式 */
if (qh->RANDOMoutside) {
  int outcoplanar = 0;
  /* 如果是 NARROWhull 模式，计算共面的外部点数 */
  if (qh->NARROWhull) {
    FORALLfacets {
      if (facet == qh->facet_next)
        break;
      if (facet->outsideset)
        outcoplanar += qh_setsize(qh, facet->outsideset);
    }
  }
  /* 计算随机数并根据随机数选择外部点 */
  randr = qh_RANDOMint;
  randr = randr / (qh_RANDOMmax + 1);
  randr = floor((qh->num_outside - outcoplanar) * randr);
  idx = (int)randr;
  /* 遍历 facet_next 之后的所有 facet */
  FORALLfacet_(qh->facet_next) {
    if (facet->outsideset) {
      /* 返回 outsideset 中第 idx 个点 */
      SETreturnsize_(facet->outsideset, size);
      if (!size)
        qh_setfree(qh, &facet->outsideset);
      else if (size > idx) {
        *visible = facet;
        return ((pointT *)qh_setdelnth(qh, facet->outsideset, idx));
      } else
        idx -= size;
    }
  }
  /* 如果 num_outside 计数错误，输出错误信息 */
  qh_fprintf(qh, qh->ferr, 6169, "qhull internal error (qh_nextfurthest): num_outside %d is too low\nby at least %d, or a random real %g >= 1.0\n",
             qh->num_outside, idx + 1, randr);
  /* 异常退出 */
  qh_errexit(qh, qh_ERRqhull, NULL, NULL);
} else { /* VIRTUALmemory 模式 */
  /* 在 facet_tail->previous 中删除最后一个外部点 */
  facet = qh->facet_tail->previous;
  furthest = (pointT *)qh_setdellast(facet->outsideset);
  /* 如果删除失败则移除 facet */
  if (!furthest) {
    if (facet->outsideset)
      qh_setfree(qh, &facet->outsideset);
    qh_removefacet(qh, facet);
    qh_prependfacet(qh, facet, &qh->facet_list);
    continue;
  }
  /* 设置 visible 指针为当前 facet */
  *visible = facet;
  /* 返回删除的外部点 */
  return furthest;
}
/* 如果以上条件都不符合，则返回 NULL */
}

/* 结束函数 */
} /* nextfurthest */
    # 对所有的面（facets）进行遍历
    for all facets
      # 对点集中剩余的所有点进行遍历
      for all remaining points in pointset
        # 计算点到面的距离
        compute distance from point to facet
        # 如果点在面的外部
        if point is outside facet
          # 从点集中移除该点（通过不重新添加）
          remove point from pointset (by not reappending)
          # 更新最佳点（bestpoint）
          update bestpoint
          # 将点或旧的最佳点（bestpoint）添加到面的外部集合
          append point or old bestpoint to facet's outside set
      # 将最佳点（bestpoint）添加到面的外部集合中（最远的点）
      append bestpoint to facet's outside set (furthest)
    # 对点集中剩余的所有点进行遍历
    for all points remaining in pointset
      # 将点分配到面的外部集合和共面集合中
      partition point into facets' outside sets and coplanar sets
/*
void qh_partitionall(qhT *qh, setT *vertices, pointT *points, int numpoints){
  setT *pointset;
  vertexT *vertex, **vertexp;
  pointT *point, **pointp, *bestpoint;
  int size, point_i, point_n, point_end, remaining, i, id;
  facetT *facet;
  realT bestdist= -REALmax, dist, distoutside;

  trace1((qh, qh->ferr, 1042, "qh_partitionall: partition all points into outside sets\n"));

  // 创建临时点集合，用于处理所有点
  pointset= qh_settemp(qh, numpoints);

  // 初始化外部点数为0
  qh->num_outside= 0;

  // 将所有点添加到临时点集合中
  pointp= SETaddr_(pointset, pointT);
  for (i=numpoints, point= points; i--; point += qh->hull_dim)
    *(pointp++)= point;
  
  // 截断临时点集合至包含 numpoints 个点
  qh_settruncate(qh, pointset, numpoints);

  // 标记已经在凸包顶点集合中的点为NULL
  FOREACHvertex_(vertices) {
    if ((id= qh_pointid(qh, vertex->point)) >= 0)
      SETelem_(pointset, id)= NULL;
  }

  // 检查GOODpointp是否在pointset中，如果是则标记为NULL
  id= qh_pointid(qh, qh->GOODpointp);
  if (id >=0 && qh->STOPcone-1 != id && -qh->STOPpoint-1 != id)
    SETelem_(pointset, id)= NULL;

  // 如果GOODvertexp存在且ONLYgood为真且非MERGING状态，则标记为NULL
  if (qh->GOODvertexp && qh->ONLYgood && !qh->MERGING) {
    if ((id= qh_pointid(qh, qh->GOODvertexp)) >= 0)
      SETelem_(pointset, id)= NULL;
  }

  // 如果没有BESToutside条件
  if (!qh->BESToutside) {
    distoutside= qh_DISToutside; // 外部点距离阈值，是qh.MINoutside与qh.max_outside的倍数，参见user_r.h
    zval_(Ztotpartition)= qh->num_points - qh->hull_dim - 1; // 设置总分区数，不包含GOOD...
    remaining= qh->num_facets;
    point_end= numpoints;

    // 遍历所有面
    FORALLfacets {
      // 计算当前面的期望大小
      size= point_end/(remaining--) + 100;
      // 创建当前面的外部点集合
      facet->outsideset= qh_setnew(qh, size);
      bestpoint= NULL;
      point_end= 0;

      // 遍历临时点集合中的点
      FOREACHpoint_i_(qh, pointset) {
        if (point) {
          zzinc_(Zpartitionall);
          // 计算点到面的距离
          qh_distplane(qh, point, facet, &dist);
          if (dist < distoutside)
            SETelem_(pointset, point_end++)= point;
          else {
            qh->num_outside++;
            if (!bestpoint) {
              bestpoint= point;
              bestdist= dist;
            } else if (dist > bestdist) {
              qh_setappend(qh, &facet->outsideset, bestpoint);
              bestpoint= point;
              bestdist= dist;
            } else
              qh_setappend(qh, &facet->outsideset, point);
          }
        }
      }

      // 添加最佳点到外部点集合，更新最远点距离
      if (bestpoint) {
        qh_setappend(qh, &facet->outsideset, bestpoint);
#if !qh_COMPUTEfurthest
        facet->furthestdist= bestdist;
#endif
      } else
        qh_setfree(qh, &facet->outsideset);

      // 截断临时点集合至新的点结束位置
      qh_settruncate(qh, pointset, point_end);
    }
  }

  // 如果有BESToutside条件或其他特殊条件，则分区点集合中的点
  if (qh->BESToutside || qh->MERGING || qh->KEEPcoplanar || qh->KEEPinside || qh->KEEPnearinside) {
    qh->findbestnew= True;
    // 遍历临时点集合中的点，并分区到几何形式列表中
    FOREACHpoint_i_(qh, pointset) {
      if (point)
        qh_partitionpoint(qh, point, qh->facet_list);
    }
    qh->findbestnew= False;
  }

  // 更新统计信息，释放临时点集合
  zzadd_(Zpartitionall, zzval_(Zpartition));
  zzval_(Zpartition)= 0;
  qh_settempfree(qh, &pointset);

  // 如果追踪等级高于等于4，则打印凸包面列表
  if (qh->IStracing >= 4)
    qh_printfacetlist(qh, qh->facet_list, NULL, True);
} /* partitionall */
*/
/*
 * qh_partitioncoplanar(qh, point, facet, dist, allnew )
 * 分割共面点到一个面
 * dist 是点到面的距离
 * 如果 dist 为 NULL，
 *   寻找最佳面，并且如果在内部则不执行任何操作
 * 如果 allnew (qh.findbestnew)
 *   寻找新的面而不使用 qh_findbest()
 *
 * 返回:
 *   qh.max_ouside 更新
 *   如果 qh.KEEPcoplanar 或 qh.KEEPinside
 *     将点分配给最佳的共面集合
 *   qh.repart_facetid == 0 (用于检测通过 qh_partitionpoint 的无限递归)
 *
 * 注意:
 *   facet->maxoutside 在 qh_check_maxout 结束时更新
 *
 * 设计:
 *   如果 dist 未定义
 *     找到最佳面用于点
 *     如果点足够远离面（取决于 qh.NEARinside 和 qh.KEEPinside）
 *       退出
 *   如果保留共面/接近内部/内部点
 *     如果点在最远的共面点之上
 *       将点附加到共面集合（它是新的最远点）
 *       更新 qh.max_outside
 *     否则
 *       将点附加到共面集合的倒数第二个位置
 *   否则，如果点明显在 qh.max_outside 之外，并且 bestfacet->coplanarset
 *     并且 bestfacet 与 facet 的角度大于垂直
 *     使用 qh_findbest() 重新分配点 -- 它可能被放到外部集合
 *   否则
 *     更新 qh.max_outside
 */
void qh_partitioncoplanar(qhT *qh, pointT *point, facetT *facet, realT *dist, boolT allnew) {
    facetT *bestfacet;
    pointT *oldfurthest;
    realT bestdist, angle, nearest, dist2= 0.0;
    int numpart= 0;
    boolT isoutside, oldfindbest, repartition= False;

    trace4((qh, qh->ferr, 4090, "qh_partitioncoplanar: partition coplanar point p%d starting with f%d dist? %2.2g, allnew? %d, gh.repart_facetid f%d\n",
      qh_pointid(qh, point), facet->id, (dist ? *dist : 0.0), allnew, qh->repart_facetid));
    qh->WAScoplanar= True;

    // 如果 dist 为 NULL
    if (!dist) {
        // 如果 allnew 为真，使用 qh_findbestnew 寻找最佳面
        if (allnew)
            bestfacet= qh_findbestnew(qh, point, facet, &bestdist, qh_ALL, &isoutside, &numpart);
        else
            // 否则，使用 qh_findbest 寻找最佳面
            bestfacet= qh_findbest(qh, point, facet, qh_ALL, !qh_ISnewfacets, qh->DELAUNAY,
                                &bestdist, &isoutside, &numpart);
        // 统计操作
        zinc_(Ztotpartcoplanar);
        zzadd_(Zpartcoplanar, numpart);
    # 如果不是 Delaunay 三角化并且不需要保持内部点，则对于 'd'，bestdist 跳过上面的 Delaunay 三角化的 facet
    if (!qh->DELAUNAY && !qh->KEEPinside) {
        # 如果需要保持附近的内部点，并且 bestdist 小于负的 NEARinside 值
        if (qh->KEEPnearinside) {
            if (bestdist < -qh->NEARinside) {
                # 增加计数器 Zcoplanarinside
                zinc_(Zcoplanarinside);
                # 输出调试信息，指示点 p%d 比附近内部 facet f%d 的距离 %2.2g 更远，是否全部是新的？ %d
                trace4((qh, qh->ferr, 4062, "qh_partitioncoplanar: point p%d is more than near-inside facet f%d dist %2.2g allnew? %d\n",
                        qh_pointid(qh, point), bestfacet->id, bestdist, allnew));
                # 重置 repart_facetid，返回
                qh->repart_facetid= 0;
                return;
            }
        } else if (bestdist < -qh->MAXcoplanar) {
            # 输出调试信息，指示点 p%d 在内部 facet f%d 中，距离 %2.2g，是否全部是新的？ %d
            trace4((qh, qh->ferr, 4063, "qh_partitioncoplanar: point p%d is inside facet f%d dist %2.2g allnew? %d\n",
                    qh_pointid(qh, point), bestfacet->id, bestdist, allnew));
            # 增加计数器 Zcoplanarinside
            zinc_(Zcoplanarinside);
            # 重置 repart_facetid，返回
            qh->repart_facetid= 0;
            return;
        }
    } else {
        # 如果是 Delaunay 三角化或者需要保持内部点，则设置 bestfacet 和 bestdist
        bestfacet= facet;
        bestdist= *dist;
    }
    # 如果 bestfacet 是可见的，则输出错误信息并退出
    if(bestfacet->visible){
        qh_fprintf(qh, qh->ferr, 6405, "qhull internal error (qh_partitioncoplanar): cannot partition coplanar p%d of f%d into visible facet f%d\n",
            qh_pointid(qh, point), facet->id, bestfacet->id);
        qh_errexit2(qh, qh_ERRqhull, facet, bestfacet);
    }
    # 如果 bestdist 大于 qh->max_outside
    if (bestdist > qh->max_outside) {
        # 如果不存在 dist 并且 facet 不等于 bestfacet，则无法从 qh_partitionpoint 递归
        if (!dist && facet != bestfacet) {
            # 增加计数器 Zpartangle
            zinc_(Zpartangle);
            # 计算角度 angle
            angle= qh_getangle(qh, facet->normal, bestfacet->normal);
            # 如果角度小于 0
            if (angle < 0) {
                # 找到距离最近的顶点
                nearest= qh_vertex_bestdist(qh, bestfacet->vertices);
                # 输出调试信息，指示 repartition coplanar 点 p%d 从 facet f%d 作为一个外部点重新分区，位于角落 facet f%d，距离 %2.2g，角度 %2.2g
                trace2((qh, qh->ferr, 2058, "qh_partitioncoplanar: repartition coplanar point p%d from f%d as an outside point above corner facet f%d dist %2.2g with angle %2.2g\n",
                  qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, angle));
                # 设置 repartition 为 True
                repartition= True;
                # 增加计数器 Zpartcorner
                zinc_(Zpartcorner);
            }
        }
    }
    // 如果不需要重新划分（repartition为假）
    if (!repartition) {
      // 如果当前最佳面外部的距离超过阈值 qh->MAXoutside * qh_RATIOcoplanaroutside
      if (bestdist > qh->MAXoutside * qh_RATIOcoplanaroutside) {
        // 找到最近的顶点
        nearest = qh_vertex_bestdist(qh, bestfacet->vertices);
        // 如果当前面和最佳面的 id 相同
        if (facet->id == bestfacet->id) {
          // 如果当前面的 id 与 qh->repart_facetid 相同
          if (facet->id == qh->repart_facetid) {
            // 输出错误消息并退出程序，提示递归调用导致的无限循环
            qh_fprintf(qh, qh->ferr, 6404, "Qhull internal error (qh_partitioncoplanar): infinite loop due to recursive call to qh_partitionpoint.  Repartition point p%d from f%d as a outside point dist %2.2g nearest vertices %2.2g\n",
                       qh_pointid(qh, point), facet->id, bestdist, nearest);
            qh_errexit(qh, qh_ERRqhull, facet, NULL);
          }
          // 设置 qh->repart_facetid 为当前面的 id，用于在调用 qh_partitionpoint 后重置
          qh->repart_facetid = facet->id; /* reset after call to qh_partitionpoint */
        }
        // 如果 point 等于 qh->coplanar_apex
        if (point == qh->coplanar_apex) {
          // 输出错误消息并退出程序，提示无法重新划分共面点导致的拓扑错误
          qh_fprintf(qh, qh->ferr, 6425, "Qhull topology error (qh_partitioncoplanar): can not repartition coplanar point p%d from f%d as outside point above f%d.  It previously failed to form a cone of facets, dist %2.2g, nearest vertices %2.2g\n",
                     qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, nearest);
          qh_errexit(qh, qh_ERRtopology, facet, NULL);
        }
        // 如果最近顶点距离小于阈值 2 * qh->MAXoutside * qh_RATIOcoplanaroutside
        if (nearest < 2 * qh->MAXoutside * qh_RATIOcoplanaroutside) {
          // 增加 Zparttwisted 计数，输出精度警告信息，提示重新划分扭曲面上的共面点
          zinc_(Zparttwisted);
          qh_fprintf(qh, qh->ferr, 7085, "Qhull precision warning: repartition coplanar point p%d from f%d as an outside point above twisted facet f%d dist %2.2g nearest vertices %2.2g\n",
                     qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, nearest);
        } else {
          // 增加 Zparthidden 计数，输出精度警告信息，提示重新划分隐藏面上的共面点
          zinc_(Zparthidden);
          qh_fprintf(qh, qh->ferr, 7086, "Qhull precision warning: repartition coplanar point p%d from f%d as an outside point above hidden facet f%d dist %2.2g nearest vertices %2.2g\n",
                     qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, nearest);
        }
        // 设置 repartition 为真，表示需要重新划分
        repartition = True;
      }
    }
    // 如果需要重新划分
    if (repartition) {
      // 保存旧的 findbestnew 值，并设为假，用于暂时关闭最佳面查找
      oldfindbest = qh->findbestnew;
      qh->findbestnew = False;
      // 调用 qh_partitionpoint 进行重新划分
      qh_partitionpoint(qh, point, bestfacet);
      // 恢复原来的 findbestnew 值
      qh->findbestnew = oldfindbest;
      // 重置 qh->repart_facetid
      qh->repart_facetid = 0;
      return;
    }
    // 未进行重新划分时的处理
    qh->repart_facetid = 0;
    // 更新 qh->max_outside 的值为 bestdist
    qh->max_outside = bestdist;
    // 如果 bestdist 大于阈值 qh->TRACEdist 或者正在进行详细跟踪（qh->IStracing >= 3）
    if (bestdist > qh->TRACEdist || qh->IStracing >= 3) {
      // 输出调试信息，提示由于共面点的重新划分导致 qh->max_outside 的增加
      qh_fprintf(qh, qh->ferr, 3041, "qh_partitioncoplanar: == p%d from f%d increases qh.max_outside to %2.2g of f%d last p%d\n",
                 qh_pointid(qh, point), facet->id, bestdist, bestfacet->id, qh->furthest_id);
      // 打印距离信息
      qh_errprint(qh, "DISTANT", facet, bestfacet, NULL, NULL);
    }
  }
  // 如果需要保留共面点、内部点或近内部点
  if (qh->KEEPcoplanar + qh->KEEPinside + qh->KEEPnearinside) {
    // 获取 bestfacet->coplanarset 中的最后一个点的指针
    oldfurthest = (pointT *)qh_setlast(bestfacet->coplanarset);
    // 如果存在最后一个点
    if (oldfurthest) {
      // 计算最后一个点到当前面的距离
      zinc_(Zcomputefurthest);
      qh_distplane(qh, oldfurthest, bestfacet, &dist2);
    }
    // 如果不存在最后一个点或者距离小于当前的 bestdist
    if (!oldfurthest || dist2 < bestdist)
      // 将 point 添加到 bestfacet->coplanarset 中
      qh_setappend(qh, &bestfacet->coplanarset, point);
    else
      qh_setappend2ndlast(qh, &bestfacet->coplanarset, point);

在上下文中，如果点 `point` 与 `bestfacet` 所表示的平面几何上共面（或在平面内部），则将该点添加到 `bestfacet` 的 `coplanarset` 集合的倒数第二个位置。


  trace4((qh, qh->ferr, 4064, "qh_partitioncoplanar: point p%d is coplanar with facet f%d (or inside) dist %2.2g\n",
          qh_pointid(qh, point), bestfacet->id, bestdist));

记录跟踪消息，表示点 `point` 与面 `bestfacet` 共面（或在其内部），并打印相关信息，包括点的标识 `p%d`、面的标识 `f%d` 以及它们之间的距离 `bestdist` 的值。
} /* partitioncoplanar */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="partitionpoint">-</a>

  qh_partitionpoint(qh, point, facet )
    assigns point to an outside set, coplanar set, or inside set (i.e., dropt)
    if qh.findbestnew
      uses qh_findbestnew() to search all new facets
    else
      uses qh_findbest()

  notes:
    after qh_distplane(), this and qh_findbest() are most expensive in 3-d

  design:
    find best facet for point
      (either exhaustive search of new facets or directed search from facet)
    if qh.NARROWhull
      retain coplanar and nearinside points as outside points
    if point is outside bestfacet
      if point above furthest point for bestfacet
        append point to outside set (it becomes the new furthest)
        if outside set was empty
          move bestfacet to end of qh.facet_list (i.e., after qh.facet_next)
        update bestfacet->furthestdist
      else
        append point one before end of outside set
    else if point is coplanar to bestfacet
      if keeping coplanar points or need to update qh.max_outside
        partition coplanar point into bestfacet
    else if near-inside point
      partition as coplanar point into bestfacet
    else is an inside point
      if keeping inside points
        partition as coplanar point into bestfacet
*/
void qh_partitionpoint(qhT *qh, pointT *point, facetT *facet) {
  realT bestdist, previousdist;
  boolT isoutside, isnewoutside= False;
  facetT *bestfacet;
  int numpart;

  // 根据 qh.findbestnew 决定使用 qh_findbestnew() 还是 qh_findbest() 来查找最佳的 facet
  if (qh->findbestnew)
    bestfacet= qh_findbestnew(qh, point, facet, &bestdist, qh->BESToutside, &isoutside, &numpart);
  else
    bestfacet= qh_findbest(qh, point, facet, qh->BESToutside, qh_ISnewfacets, !qh_NOupper,
                          &bestdist, &isoutside, &numpart);

  // 统计 partition 操作的次数
  zinc_(Ztotpartition);
  zzadd_(Zpartition, numpart);

  // 如果 bestfacet 是可见的，报错并退出
  if(bestfacet->visible){
    qh_fprintf(qh, qh->ferr, 6293, "qhull internal error (qh_partitionpoint): cannot partition p%d of f%d into visible facet f%d\n",
      qh_pointid(qh, point), facet->id, bestfacet->id);
    qh_errexit2(qh, qh_ERRqhull, facet, bestfacet);
  }

  // 如果是 NARROWhull 模式下的处理
  if (qh->NARROWhull) {
    // 如果是 DELAUNAY 模式，并且 point 不是外部点，并且 bestdist 大于等于 -qh->MAXcoplanar
    // 则触发 joggle_restart，表示几乎是 incident 点（narrow hull）
    if (qh->DELAUNAY && !isoutside && bestdist >= -qh->MAXcoplanar)
      qh_joggle_restart(qh, "nearly incident point (narrow hull)");

    // 如果保留近内点
    if (qh->KEEPnearinside) {
      if (bestdist >= -qh->NEARinside)
        isoutside= True;
    } else if (bestdist >= -qh->MAXcoplanar)
      isoutside= True;
  }

  // 如果 point 是外部点
  if (isoutside) {
    // 如果 bestfacet 的 outsideset 是空的或者最后一个元素为空
    // 则将 point 添加到 bestfacet 的 outsideset 中
    qh_setappend(qh, &(bestfacet->outsideset), point);

    // 如果不是 NARROWhull 或者 bestdist 大于 qh->MINoutside，则设置 isnewoutside 为 True
    if (!qh->NARROWhull || bestdist > qh->MINoutside)
      isnewoutside= True;

    // 如果未定义 COMPUTEfurthest，则更新 bestfacet 的 furthestdist
#if !qh_COMPUTEfurthest
    bestfacet->furthestdist= bestdist;
#endif
  } else {
    // 如果 point 是与 bestfacet 共面的点
    if (point is coplanar to bestfacet) {
      // 如果保留共面点或需要更新 qh.max_outside，则将共面点分配到 bestfacet 中
      partition coplanar point into bestfacet
    } else if (near-inside point) {
      // 将近内点作为共面点分配到 bestfacet 中
      partition as coplanar point into bestfacet
    } else is an inside point {
      // 如果保留内部点，则将其作为共面点分配到 bestfacet 中
      partition as coplanar point into bestfacet
    }
  }
}
#if qh_COMPUTEfurthest
      zinc_(Zcomputefurthest);
      // 如果设置了计算最远点标记，则调用 Zcomputefurthest 函数
      qh_distplane(qh, oldfurthest, bestfacet, &previousdist);
      // 计算点到最佳平面的距离，将结果存储在 previousdist 中
      if (previousdist < bestdist)
        // 如果计算得到的距离比当前最佳距离小，则将点添加到 bestfacet 的 outsideset 中
        qh_setappend(qh, &(bestfacet->outsideset), point);
      else
        // 否则将点添加到 bestfacet 的 outsideset 的倒数第二个位置
        qh_setappend2ndlast(qh, &(bestfacet->outsideset), point);
#else
      // 否则，使用 bestfacet 的 furthestdist 值作为 previousdist
      previousdist= bestfacet->furthestdist;
      // 如果 previousdist 小于 bestdist
      if (previousdist < bestdist) {
        // 将点添加到 bestfacet 的 outsideset 中
        qh_setappend(qh, &(bestfacet->outsideset), point);
        // 更新 bestfacet 的 furthestdist 为 bestdist
        bestfacet->furthestdist= bestdist;
        // 如果设置了 NARROWhull，并且 previousdist 小于 MINoutside，且 bestdist 大于等于 MINoutside，则设置 isnewoutside 为 True
        if (qh->NARROWhull && previousdist < qh->MINoutside && bestdist >= qh->MINoutside)
          isnewoutside= True;
      }else
        // 否则将点添加到 bestfacet 的 outsideset 的倒数第二个位置
        qh_setappend2ndlast(qh, &(bestfacet->outsideset), point);
#endif
    }
    // 如果 isnewoutside 为 True 并且 qh.facet_next 不等于 bestfacet
    if (isnewoutside && qh->facet_next != bestfacet) {
      // 如果 bestfacet 是 newfacet
      if (bestfacet->newfacet) {
        // 如果 qh.facet_next 也是 newfacet，则将 qh.facet_next 设置为 qh.newfacet_list
        if (qh->facet_next->newfacet)
          qh->facet_next= qh->newfacet_list; /* 确保它在 qh.facet_next 之后 */
      }else {
        // 否则，先移除 bestfacet，再将其追加到 qh 中
        qh_removefacet(qh, bestfacet);  /* 确保它在 qh.facet_next 之后 */
        qh_appendfacet(qh, bestfacet);
        // 如果 qh.newfacet_list 存在，则将 bestfacet 的 newfacet 设置为 True
        if(qh->newfacet_list){
          bestfacet->newfacet= True;
        }
      }
    }
    // 增加 qh.num_outside 计数
    qh->num_outside++;
    // 输出跟踪信息
    trace4((qh, qh->ferr, 4065, "qh_partitionpoint: point p%d is outside facet f%d newfacet? %d, newoutside? %d (or narrowhull)\n",
          qh_pointid(qh, point), bestfacet->id, bestfacet->newfacet, isnewoutside));
  }else if (qh->DELAUNAY || bestdist >= -qh->MAXcoplanar) { /* for 'd', bestdist skips upperDelaunay facets */
    // 如果设置了 DELAUNAY 或者 bestdist 大于等于 -qh->MAXcoplanar
    if (qh->DELAUNAY)
      // 如果设置了 DELAUNAY，则重新开始 joggle
      qh_joggle_restart(qh, "nearly incident point");
    /* 允许通过 joggle 的共面点，可能是内部点 */
    // 增加 Zcoplanarpart 计数
    zzinc_(Zcoplanarpart);
    // 如果设置了 KEEPcoplanar 或者 KEEPnearinside，或者 bestdist 大于 qh.max_outside
    if ((qh->KEEPcoplanar + qh->KEEPnearinside) || bestdist > qh->max_outside)
      // 对共面点进行分区，根据 qh.findbestnew 进行操作
      qh_partitioncoplanar(qh, point, bestfacet, &bestdist, qh->findbestnew);
    else {
      // 否则输出跟踪信息，指出点是共面到 bestfacet（但被丢弃）
      trace4((qh, qh->ferr, 4066, "qh_partitionpoint: point p%d is coplanar to facet f%d (dropped)\n",
          qh_pointid(qh, point), bestfacet->id));
    }
  }else if (qh->KEEPnearinside && bestdist >= -qh->NEARinside) {
    // 如果设置了 KEEPnearinside 并且 bestdist 大于等于 -qh->NEARinside
    zinc_(Zpartnear);
    // 对共面点进行分区，根据 qh.findbestnew 进行操作
    qh_partitioncoplanar(qh, point, bestfacet, &bestdist, qh->findbestnew);
  }else {
    // 否则增加 Zpartinside 计数
    zinc_(Zpartinside);
    // 输出跟踪信息，指出点在所有的面内部，距离最近的是 bestfacet，距离为 bestdist
    trace4((qh, qh->ferr, 4067, "qh_partitionpoint: point p%d is inside all facets, closest to f%d dist %2.2g\n",
          qh_pointid(qh, point), bestfacet->id, bestdist));
    // 如果设置了 KEEPinside，则对共面点进行分区，根据 qh.findbestnew 进行操作
    if (qh->KEEPinside)
      qh_partitioncoplanar(qh, point, bestfacet, &bestdist, qh->findbestnew);
  }
} /* partitionpoint */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="partitionvisible">-</a>

  qh_partitionvisible(qh, allpoints, numoutside )
    partitions outside points in visible facets (qh.visible_list) to qh.newfacet_list
    if keeping coplanar/near-inside/inside points
      partitions coplanar points; repartitions if 'allpoints' (not used)
    1st neighbor (if any) of visible facets points to a horizon facet or a new facet

  returns:
    # 更新 qh.newfacet_list 中的外部集合和共面集合
    # 更新 qh.num_outside（外部点的计数）
    # 不截断 f.outsideset、f.coplanarset 或 qh.del_vertices（参见 qh_deletevisible）
    
    # 注意：
    # 被 qh_qhull、qh_addpoint 和 qh_all_vertexmerges 调用
    # qh.findbest_notsharp 应该是清楚的（如果集合，则会有额外的工作）
    
    # 设计：
    # 对于所有具有外部集合或共面集合的可见面：
    #   为可见面选择一个新的面
    #   如果是外部集合：
    #     将外部集合划分为新的面
    #   如果是共面集合并且保持共面/接近内部/内部点：
    #     如果是所有点：
    #       将共面集合划分为新的面，可能会分配到外部
    #     否则：
    #       将共面集合划分为新的面的共面集合
    # 对于每个被删除的顶点：
    #   如果是所有点：
    #     将顶点划分为新的面，可能会分配到外部
    #   否则：
    #     将顶点划分为新的面的共面集合
void qh_partitionvisible(qhT *qh, boolT allpoints, int *numoutside /* qh.visible_list */) {
  // 定义变量和指针
  facetT *visible, *newfacet;  // 定义可见面和新面
  pointT *point, **pointp;     // 定义点和点指针
  int delsize, coplanar=0, size;  // 定义删除大小、共面计数、大小

  // 输出追踪信息
  trace3((qh, qh->ferr, 3042, "qh_partitionvisible: partition outside and coplanar points of visible and merged facets f%d into new facets f%d\n",
    qh->visible_list->id, qh->newfacet_list->id));
  
  // 如果仅仅是最大化，则调整最小外部数目
  if (qh->ONLYmax)
    maximize_(qh->MINoutside, qh->max_vertex);
  
  // 初始化外部点数目为零
  *numoutside= 0;
  
  // 遍历所有可见面
  FORALLvisible_facets {
    // 如果可见面既没有外部集合也没有共面集合，则继续下一个可见面
    if (!visible->outsideset && !visible->coplanarset)
      continue;
    
    // 获取替代面，如果没有则使用新面列表中的第一个新面
    newfacet= qh_getreplacement(qh, visible);
    if (!newfacet)
      newfacet= qh->newfacet_list;
    
    // 如果没有下一个新面，则输出拓扑错误信息并退出
    if (!newfacet->next) {
      qh_fprintf(qh, qh->ferr, 6170, "qhull topology error (qh_partitionvisible): all new facets deleted as\n       degenerate facets. Can not continue.\n");
      qh_errexit(qh, qh_ERRtopology, NULL, NULL);
    }
    
    // 如果有外部集合
    if (visible->outsideset) {
      // 计算外部集合的大小，并将其加到外部点数目中
      size= qh_setsize(qh, visible->outsideset);
      *numoutside += size;
      qh->num_outside -= size;
      // 遍历外部集合中的每一个点，并将其分割到新面中
      FOREACHpoint_(visible->outsideset)
        qh_partitionpoint(qh, point, newfacet);
    }
    
    // 如果有共面集合，并且保留共面、内部或靠近内部的点
    if (visible->coplanarset && (qh->KEEPcoplanar + qh->KEEPinside + qh->KEEPnearinside)) {
      // 计算共面集合的大小，并将其加到共面计数中
      size= qh_setsize(qh, visible->coplanarset);
      coplanar += size;
      // 遍历共面集合中的每一个点
      FOREACHpoint_(visible->coplanarset) {
        // 如果需要处理所有点，则将点分割到新面中
        if (allpoints) /* not used */
          qh_partitionpoint(qh, point, newfacet);
        // 否则，根据最佳新面分割共面点
        else
          qh_partitioncoplanar(qh, point, newfacet, NULL, qh->findbestnew);
      }
    }
  }
  
  // 计算待删除顶点集合的大小
  delsize= qh_setsize(qh, qh->del_vertices);
  
  // 如果存在待删除的顶点
  if (delsize > 0) {
    // 输出追踪信息
    trace3((qh, qh->ferr, 3049, "qh_partitionvisible: partition %d deleted vertices as coplanar? %d points into new facets f%d\n",
      delsize, !allpoints, qh->newfacet_list->id));
    
    // 遍历所有待删除的顶点
    FOREACHvertex_(qh->del_vertices) {
      // 如果顶点存在且未被分割
      if (vertex->point && !vertex->partitioned) {
        // 如果没有定义新面列表或新面列表等于尾部，则输出错误信息并退出
        if (!qh->newfacet_list || qh->newfacet_list == qh->facet_tail) {
          qh_fprintf(qh, qh->ferr, 6284, "qhull internal error (qh_partitionvisible): all new facets deleted or none defined.  Can not partition deleted v%d.\n", vertex->id);
          qh_errexit(qh, qh_ERRqhull, NULL, NULL);
        }
        // 如果需要处理所有点，则将顶点分割到新面中
        if (allpoints) /* not used */
          /* [apr'2019] infinite loop if vertex recreates the same facets from the same horizon
             e.g., qh_partitionpoint if qh.DELAUNAY with qh.MERGEindependent for all mergetype, ../eg/qtest.sh t427764 '1000 s W1e-13 D3' 'd' */
          qh_partitionpoint(qh, vertex->point, qh->newfacet_list);
        // 否则，根据所有新面搜索并分割共面点
        else
          qh_partitioncoplanar(qh, vertex->point, qh->newfacet_list, NULL, qh_ALL);
        // 标记顶点已分割
        vertex->partitioned= True;
      }
    }
  }
  
  // 输出追踪信息
  trace1((qh, qh->ferr, 1043,"qh_partitionvisible: partitioned %d points from outsidesets, %d points from coplanarsets, and %d deleted vertices\n", *numoutside, coplanar, delsize));
} /* partitionvisible */
/*
   qh_printsummary(qh, fp )
     将摘要信息打印到文件流 fp 中

   notes:
     不在 io_r.c 中实现，以便 user_eg.c 可以防止加载 io_r.c
     qh_printsummary 和 qh_countfacets 必须匹配计数
     更新 qh.facet_visit 以检测无限循环

   design:
     确定点的数量、顶点数量和共面点数量
     打印摘要信息
*/
void qh_printsummary(qhT *qh, FILE *fp) {
  realT ratio, outerplane, innerplane;
  double cpu;
  int size, id, nummerged, numpinched, numvertices, numcoplanars= 0, nonsimplicial=0, numdelaunay= 0;
  facetT *facet;
  const char *s;
  int numdel= zzval_(Zdelvertextot);
  int numtricoplanars= 0;
  boolT goodused;

  // 计算点的总数（包括其他点集合中的点）
  size= qh->num_points + qh_setsize(qh, qh->other_points);
  // 计算顶点数量（减去被删除的顶点集合）
  numvertices= qh->num_vertices - qh_setsize(qh, qh->del_vertices);
  // 获取 GOODpointp 的 ID
  id= qh_pointid(qh, qh->GOODpointp);

  // 检查 facet_list 是否符合列表结构，若不符合且未调用 ERREXIT，则输出错误信息并退出
  if (!qh_checklists(qh, qh->facet_list) && !qh->ERREXITcalled) {
    qh_fprintf(qh, fp, 6372, "qhull internal error: qh_checklists failed at qh_printsummary\n");
    if (qh->num_facets < 4000)
      qh_printlists(qh);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }

  // 如果是 DELAUNAY 并且已调用 ERREXIT，则更新 facet 的状态
  if (qh->DELAUNAY && qh->ERREXITcalled) {
    /* 更新 f.good 并计算 qh.num_good，与 qh_findgood_all 类似 */
    FORALLfacets {
      if (facet->visible)
        facet->good= False; /* 将被删除 */
      else if (facet->good) {
        if (facet->normal && !qh_inthresholds(qh, facet->normal, NULL))
          facet->good= False;
        else
          numdelaunay++;
      }
    }
    qh->num_good= numdelaunay;
  }

  // 统计共面集合中点的数量以及非简单形式的面的数量
  FORALLfacets {
    if (facet->coplanarset)
      numcoplanars += qh_setsize(qh, facet->coplanarset);
    if (facet->good) {
      if (facet->simplicial) {
        if (facet->keepcentrum && facet->tricoplanar)
          numtricoplanars++;
      } else if (qh_setsize(qh, facet->vertices) != qh->hull_dim)
        nonsimplicial++;
    }
  }

  // 如果 ID 是有效的且不等于 STOPcone-1 或 -STOPpoint-1，则减少 size
  if (id >=0 && qh->STOPcone-1 != id && -qh->STOPpoint-1 != id)
    size--;

  // 如果 STOPadd、STOPcone 或 STOPpoint 有值，则输出相应的信息
  if (qh->STOPadd || qh->STOPcone || qh->STOPpoint)
    qh_fprintf(qh, fp, 9288, "\nEarly exit due to 'TAn', 'TVn', 'TCn', 'TRn', or precision error with 'QJn'.");

  // 检查是否使用了 GOOD 相关的参数
  goodused= False;
  if (qh->ERREXITcalled)
    ; /* qh_findgood_all 未调用 */
  else if (qh->UPPERdelaunay) {
    if (qh->GOODvertex || qh->GOODpoint || qh->SPLITthresholds)
      goodused= True;
  } else if (qh->DELAUNAY) {
    if (qh->GOODvertex || qh->GOODpoint || qh->GOODthreshold)
      goodused= True;
  } else if (qh->num_good > 0 || qh->GOODthreshold)
    goodused= True;

  // 计算合并的面的数量
  nummerged= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);

  // 如果是 VORONOI 模式，则输出相应的 Voronoi 图信息
  if (qh->VORONOI) {
    if (qh->UPPERdelaunay)
      qh_fprintf(qh, fp, 9289, "\n\
Furthest-site Voronoi vertices by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    else
      qh_fprintf(qh, fp, 9290, "\n\
Voronoi diagram by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    qh_fprintf(qh, fp, 9291, "  Number of Voronoi regions%s: %d\n",
              qh->ATinfinity ? " and at-infinity" : "", numvertices);
    # 输出 Voronoi 区域的数量以及是否存在无限远点信息
    if (numdel)
      # 如果存在已删除点的数量，则输出已合并删除点的总数
      qh_fprintf(qh, fp, 9292, "  Total number of deleted points due to merging: %d\n", numdel);
    if (numcoplanars - numdel > 0)
      # 如果存在几乎共面的点数量大于已删除点数量，则输出几乎共面点的数量
      qh_fprintf(qh, fp, 9293, "  Number of nearly incident points: %d\n", numcoplanars - numdel);
    else if (size - numvertices - numdel > 0)
      # 否则，输出几乎共面点的总数
      qh_fprintf(qh, fp, 9294, "  Total number of nearly incident points: %d\n", size - numvertices - numdel);
    # 输出 Voronoi 顶点的数量，可选是否为 'good' 类型
    qh_fprintf(qh, fp, 9295, "  Number of%s Voronoi vertices: %d\n",
              goodused ? " 'good'" : "", qh->num_good);
    if (nonsimplicial)
      # 如果存在非简单形式的 Voronoi 顶点，则输出其数量
      qh_fprintf(qh, fp, 9296, "  Number of%s non-simplicial Voronoi vertices: %d\n",
              goodused ? " 'good'" : "", nonsimplicial);
  }else if (qh->DELAUNAY) {
    # 如果是 Delaunay 三角剖分模式，则根据设置输出上限信息
    if (qh->UPPERdelaunay)
      qh_fprintf(qh, fp, 9297, "\n\
# 输出描述性信息，关于使用凸包进行最远点位的 Delaunay 三角剖分，指定点数和维度
Furthest-site Delaunay triangulation by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
else
  # 输出描述性信息，关于使用凸包进行 Delaunay 三角剖分，指定点数和维度
  qh_fprintf(qh, fp, 9298, "\n\
Delaunay triangulation by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);

# 输出输入点的数量信息，如果包含无穷远点则附加描述
qh_fprintf(qh, fp, 9299, "  Number of input sites%s: %d\n",
          qh->ATinfinity ? " and at-infinity" : "", numvertices);

# 如果存在已删除的点，输出已删除点的数量
if (numdel)
  qh_fprintf(qh, fp, 9300, "  Total number of deleted points due to merging: %d\n", numdel);

# 如果存在几乎共面的点，输出其数量
if (numcoplanars - numdel > 0)
  qh_fprintf(qh, fp, 9301, "  Number of nearly incident points: %d\n", numcoplanars - numdel);
else if (size - numvertices - numdel > 0)
  qh_fprintf(qh, fp, 9302, "  Total number of nearly incident points: %d\n", size - numvertices - numdel);

# 输出 Delaunay 区域的数量，如果使用了 'good' 标志则附加描述
qh_fprintf(qh, fp, 9303, "  Number of%s Delaunay regions: %d\n",
          goodused ? " 'good'" : "", qh->num_good);

# 如果存在非单纯形的区域，输出其数量及相应描述
if (nonsimplicial)
  qh_fprintf(qh, fp, 9304, "  Number of%s non-simplicial Delaunay regions: %d\n",
             goodused ? " 'good'" : "", nonsimplicial);
    # 如果有非简单形式的面
    if (nonsimplicial)
      # 使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出非简单形式的面的数量
      qh_fprintf(qh, fp, 9317, "  Number of%s non-simplicial facets: %d\n",
              goodused ? " 'good'" : "", nonsimplicial);
  }
  # 如果存在三角形平面
  if (numtricoplanars)
      # 使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出三角形平面的数量
      qh_fprintf(qh, fp, 9318, "  Number of triangulated facets: %d\n", numtricoplanars);
  # 使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出 Qhull 统计信息的标题行
  qh_fprintf(qh, fp, 9319, "\nStatistics for: %s | %s",
                      qh->rbox_command, qh->qhull_command);
  # 如果存在随机旋转值
  if (qh->ROTATErandom != INT_MIN)
    # 使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出随机旋转值
    qh_fprintf(qh, fp, 9320, " QR%d\n\n", qh->ROTATErandom);
  else
    # 使用 qh_fprintf 函数向文件指针 fp 写入空行
    qh_fprintf(qh, fp, 9321, "\n\n");
  # 使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出处理的点的数量统计
  qh_fprintf(qh, fp, 9322, "  Number of points processed: %d\n", zzval_(Zprocessed));
  # 使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出创建的超平面数量统计
  qh_fprintf(qh, fp, 9323, "  Number of hyperplanes created: %d\n", zzval_(Zsetplane));
  # 如果处于 Delaunay 模式，使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出凸壳中的面的数量统计
  if (qh->DELAUNAY)
    qh_fprintf(qh, fp, 9324, "  Number of facets in hull: %d\n", qh->num_facets - qh->num_visible);
  # 使用 qh_fprintf 函数向文件指针 fp 写入消息，格式化输出 Qhull 运行过程中的距离测试数量统计
  qh_fprintf(qh, fp, 9325, "  Number of distance tests for qhull: %d\n", zzval_(Zpartition)+
      zzval_(Zpartitionall)+zzval_(Znumvisibility)+zzval_(Zpartcoplanar));
if (nummerged) {
    // 如果有合并的面片数量，则输出合并过程中的距离测试次数统计信息
    qh_fprintf(qh, fp, 9330,"  Number of distance tests for merging: %d\n",zzval_(Zbestdist)+
          zzval_(Zcentrumtests)+zzval_(Zvertextests)+zzval_(Zdistcheck)+zzval_(Zdistzero));
    // 输出检查合并过程中的距离测试次数统计信息
    qh_fprintf(qh, fp, 9331,"  Number of distance tests for checking: %d\n",zzval_(Zcheckpart)+zzval_(Zdistconvex));
    // 输出合并的面片数量
    qh_fprintf(qh, fp, 9332,"  Number of merged facets: %d\n", nummerged);
}

// 计算被捏合的顶点数量
numpinched= zzval_(Zpinchduplicate) + zzval_(Zpinchedvertex);
if (numpinched)
    // 输出被捏合的顶点数量信息
    qh_fprintf(qh, fp, 9375,"  Number of merged pinched vertices: %d\n", numpinched);

// 如果不是在随机外部点上且 Qhull 完成计算
if (!qh->RANDOMoutside && qh->QHULLfinished) {
    // 计算 CPU 时间
    cpu= (double)qh->hulltime;
    cpu /= (double)qh_SECticks;
    wval_(Wcpu)= cpu;
    // 输出计算凸包所用的 CPU 时间信息
    qh_fprintf(qh, fp, 9333, "  CPU seconds to compute hull (after input): %2.4g\n", cpu);
}

// 如果是重新运行
if (qh->RERUN) {
    // 如果没有预合并且没有精确合并
    if (!qh->PREmerge && !qh->MERGEexact)
        // 输出具有精度错误的运行百分比信息
        qh_fprintf(qh, fp, 9334, "  Percentage of runs with precision errors: %4.1f\n",
           zzval_(Zretry)*100.0/qh->build_cnt);  /* careful of order */
}
else if (qh->JOGGLEmax < REALmax/2) {
    // 如果输入数据有抖动
    if (zzval_(Zretry))
        // 输出重新尝试次数及其抖动值
        qh_fprintf(qh, fp, 9335, "  After %d retries, input joggled by: %2.2g\n",
         zzval_(Zretry), qh->JOGGLEmax);
    else
        // 输出输入数据抖动值
        qh_fprintf(qh, fp, 9336, "  Input joggled by: %2.2g\n", qh->JOGGLEmax);
}

// 如果总面积不为零
if (qh->totarea != 0.0)
    // 输出总面积信息
    qh_fprintf(qh, fp, 9337, "  %s facet area:   %2.8g\n",
            zzval_(Ztotmerge) ? "Approximate" : "Total", qh->totarea);

// 如果总体积不为零
if (qh->totvol != 0.0)
    // 输出总体积信息
    qh_fprintf(qh, fp, 9338, "  %s volume:       %2.8g\n",
            zzval_(Ztotmerge) ? "Approximate" : "Total", qh->totvol);

// 如果正在合并
if (qh->MERGING) {
    // 计算外部点和内部点距离
    qh_outerinner(qh, NULL, &outerplane, &innerplane);
    // 如果外部点距离超过 2 倍的 DISTround
    if (outerplane > 2 * qh->DISTround) {
        // 输出点在面片上方的最大距离及其比率信息
        qh_fprintf(qh, fp, 9339, "  Maximum distance of point above facet: %2.2g", outerplane);
        ratio= outerplane/(qh->ONEmerge + qh->DISTround);
        // 如果比率大于 0.05 并且 2 倍的 ONEmerge 大于 MINoutside 且抖动值超过 REALmax 的一半
        if (ratio > 0.05 && 2* qh->ONEmerge > qh->MINoutside && qh->JOGGLEmax > REALmax/2)
            // 输出比率信息
            qh_fprintf(qh, fp, 9340, " (%.1fx)\n", ratio);
        else
            // 否则，换行输出
            qh_fprintf(qh, fp, 9341, "\n");
    }
}
    # 如果内部平面距离小于两倍 qh->DISTround 的负值时执行以下操作
    if (innerplane < -2 * qh->DISTround) {
      # 在文件流 fp 中打印顶点位于平面以下的最大距离信息
      qh_fprintf(qh, fp, 9342, "  Maximum distance of vertex below facet: %2.2g", innerplane);
      # 计算 ratio，并根据条件判断是否需要添加额外信息
      ratio= -innerplane/(qh->ONEmerge+qh->DISTround);
      if (ratio > 0.05 && qh->JOGGLEmax > REALmax/2)
        # 在文件流 fp 中打印 ratio 的信息，如果满足条件则添加比例信息
        qh_fprintf(qh, fp, 9343, " (%.1fx)\n", ratio);
      else
        # 在文件流 fp 中打印换行符
        qh_fprintf(qh, fp, 9344, "\n");
    }
  }
  # 在文件流 fp 中打印空行
  qh_fprintf(qh, fp, 9345, "\n");
} /* printsummary */


注释：


# 这是一个 C 语言风格的注释，用于注释掉 `printsummary` 函数或代码块的结束
# 在 C 语言中，这种注释通常用于表示函数或代码块的结束标记
```