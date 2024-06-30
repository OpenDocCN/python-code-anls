# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\merge_r.c`

```
/*<html><pre>  -<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="TOP">-</a>

   merge_r.c
   merges non-convex facets

   see qh-merge_r.htm and merge_r.h

   other modules call qh_premerge() and qh_postmerge()

   the user may call qh_postmerge() to perform additional merges.

   To remove deleted facets and vertices (qhull() in libqhull_r.c):
     qh_partitionvisible(qh, !qh_ALL, &numoutside);  // visible_list, newfacet_list
     qh_deletevisible();         // qh.visible_list
     qh_resetlists(qh, False, qh_RESETvisible);       // qh.visible_list newvertex_list newfacet_list

   assumes qh.CENTERtype= centrum

   merges occur in qh_mergefacet and in qh_mergecycle
   vertex->neighbors not set until the first merge occurs

   Copyright (c) 1993-2019 C.B. Barber.
   $Id: //main/2019/qhull/src/libqhull_r/merge_r.c#12 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

#ifndef qh_NOmerge

/* MRGnone, etc. */
const char *mergetypes[]= {
  "none",
  "coplanar",
  "anglecoplanar",
  "concave",
  "concavecoplanar",
  "twisted",
  "flip",
  "dupridge",
  "subridge",
  "vertices",
  "degen",
  "redundant",
  "mirror",
  "coplanarhorizon",
};

/*===== functions(alphabetical after premerge and postmerge) ======*/

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="premerge">-</a>

  qh_premerge(qh, apexpointid, maxcentrum )
    pre-merge nonconvex facets in qh.newfacet_list for apexpointid
    maxcentrum defines coplanar and concave (qh_test_appendmerge)

  returns:
    deleted facets added to qh.visible_list with facet->visible set

  notes:
    only called by qh_addpoint
    uses globals, qh.MERGEexact, qh.PREmerge

  design:
    mark dupridges in qh.newfacet_list
    merge facet cycles in qh.newfacet_list
    merge dupridges and concave facets in qh.newfacet_list
    check merged facet cycles for degenerate and redundant facets
    merge degenerate and redundant facets
    collect coplanar and concave facets
    merge concave, coplanar, degenerate, and redundant facets
*/
void qh_premerge(qhT *qh, int apexpointid, realT maxcentrum, realT maxangle /* qh.newfacet_list */) {
  boolT othermerge= False;  /* 是否发生了除了标准预合并之外的其他合并操作 */

  if (qh->ZEROcentrum && qh_checkzero(qh, !qh_ALL))
    return;  /* 如果ZEROcentrum为真并且qh_checkzero返回真，则直接返回 */

  trace2((qh, qh->ferr, 2008, "qh_premerge: premerge centrum %2.2g angle %4.4g for apex p%d newfacet_list f%d\n",
            maxcentrum, maxangle, apexpointid, getid_(qh->newfacet_list)));
  if (qh->IStracing >= 4 && qh->num_facets < 100)
    qh_printlists(qh);  /* 如果追踪级别大于等于4并且facets数量小于100，则打印列表 */

  qh->centrum_radius= maxcentrum;  /* 设置中心半径 */
  qh->cos_max= maxangle;  /* 设置最大角度余弦值 */
  
  if (qh->hull_dim >=3) {
    qh_mark_dupridges(qh, qh->newfacet_list, qh_ALL); /* 标记newfacet_list中的dupridge */
    qh_mergecycle_all(qh, qh->newfacet_list, &othermerge); /* 合并newfacet_list中的循环 */
    qh_forcedmerges(qh, &othermerge /* qh.facet_mergeset */); /* 执行强制合并操作 */
  }else /* qh.hull_dim == 2 */
    # 对新面列表中的面进行合并循环操作
    qh_mergecycle_all(qh, qh->newfacet_list, &othermerge);
  # 对新面列表中的面执行翻转合并操作
  qh_flippedmerges(qh, qh->newfacet_list, &othermerge);
  # 如果不是精确合并模式或者总合并次数不为零，则执行以下操作
  if (!qh->MERGEexact || zzval_(Ztotmerge)) {
    # 增加预合并总计数
    zinc_(Zpremergetot);
    # 禁用后续合并
    qh->POSTmerging= False;
    # 初始化合并设置，针对新面列表中的面
    qh_getmergeset_initial(qh, qh->newfacet_list);
    # 执行所有合并操作，包括非常规的合并
    qh_all_merges(qh, othermerge, False);
  }
} /* premerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="postmerge">-</a>

  qh_postmerge(qh, reason, maxcentrum, maxangle, vneighbors )
    post-merge nonconvex facets as defined by maxcentrum and maxangle
    'reason' is for reporting progress
    if vneighbors ('Qv'),
      calls qh_test_vneighbors at end of qh_all_merge from qh_postmerge

  returns:
    if first call (qh.visible_list != qh.facet_list),
      builds qh.facet_newlist, qh.newvertex_list
    deleted facets added to qh.visible_list with facet->visible
    qh.visible_list == qh.facet_list

  notes:
    called by qh_qhull after qh_buildhull
    called if a merge may be needed due to
      qh.MERGEexact ('Qx'), qh_DIMreduceBuild, POSTmerge (e.g., 'Cn'), or TESTvneighbors ('Qv')
    if firstmerge,
      calls qh_reducevertices before qh_getmergeset

  design:
    if first call
      set qh.visible_list and qh.newfacet_list to qh.facet_list
      add all facets to qh.newfacet_list
      mark non-simplicial facets, facet->newmerge
      set qh.newvertext_list to qh.vertex_list
      add all vertices to qh.newvertex_list
      if a pre-merge occurred
        set vertex->delridge {will retest the ridge}
        if qh.MERGEexact
          call qh_reducevertices()
      if no pre-merging
        merge flipped facets
    determine non-convex facets
    merge all non-convex facets
*/
void qh_postmerge(qhT *qh, const char *reason, realT maxcentrum, realT maxangle,
                      boolT vneighbors) {
  facetT *newfacet;
  boolT othermerges= False;
  vertexT *vertex;

  if (qh->REPORTfreq || qh->IStracing) {
    qh_buildtracing(qh, NULL, NULL);
    qh_printsummary(qh, qh->ferr);
    if (qh->PRINTstatistics)
      qh_printallstatistics(qh, qh->ferr, "reason");
    qh_fprintf(qh, qh->ferr, 8062, "\n%s with 'C%.2g' and 'A%.2g'\n",
        reason, maxcentrum, maxangle);
  }
  trace2((qh, qh->ferr, 2009, "qh_postmerge: postmerge.  test vneighbors? %d\n",
            vneighbors));
  qh->centrum_radius= maxcentrum;  /* 设置几何中心半径 */
  qh->cos_max= maxangle;           /* 设置角度余弦的最大值 */
  qh->POSTmerging= True;           /* 设置 POSTmerging 标志为 True */
  if (qh->visible_list != qh->facet_list) {  /* 如果是第一次调用，因为 qh_buildhull，如果 qh.POSTmerge 则可能是多次调用 */
    qh->NEWfacets= True;           /* 设置 NEWfacets 标志为 True */
    qh->visible_list= qh->newfacet_list= qh->facet_list;  /* 设置 visible_list 和 newfacet_list 为 facet_list */
    FORALLnew_facets {              /* 遍历所有新的面 */
      newfacet->newfacet= True;     /* 设置 newfacet 标志为 True */
       if (!newfacet->simplicial)
        newfacet->newmerge= True;   /* 如果不是 simplicial 面，则设置 newmerge 标志为 True */
     zinc_(Zpostfacets);            /* 增加 Zpostfacets 统计计数 */
    }
    qh->newvertex_list= qh->vertex_list;  /* 设置 newvertex_list 为 vertex_list */
    FORALLvertices
      vertex->newfacet= True;       /* 设置所有顶点的 newfacet 标志为 True */
    if (qh->VERTEXneighbors) {      /* 如果发生了合并 */
      if (qh->MERGEexact && qh->hull_dim <= qh_DIMreduceBuild)
        qh_reducevertices(qh);      /* 如果 MERGEexact 并且 hull_dim <= DIMreduceBuild，则调用 qh_reducevertices */
    }
    # 如果qh->PREmerge和qh->MERGEexact都为假，则执行flippedmerges操作
    if (!qh->PREmerge && !qh->MERGEexact)
      qh_flippedmerges(qh, qh->newfacet_list, &othermerges);
  }
  # 初始化mergeset，使用qh->newfacet_list作为输入
  qh_getmergeset_initial(qh, qh->newfacet_list);
  # 执行所有的merges操作，不进行顶点减少操作，使用vneighbors作为参数
  qh_all_merges(qh, False, vneighbors); /* calls qh_reducevertices before exiting */
  # 遍历所有新的facets
  FORALLnew_facets
    # 将newfacet->newmerge设置为False，如果f.vertices中没有'delridge'顶点则设置为True
    newfacet->newmerge= False;   /* Was True if no vertex in f.vertices was 'delridge' */
} /* post_merge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="all_merges">-</a>

  qh_all_merges(qh, othermerge, vneighbors )
    merge all non-convex facets

    set othermerge if already merged facets (calls qh_reducevertices)
    if vneighbors ('Qv' at qh.POSTmerge)
      tests vertex neighbors for convexity at end (qh_test_vneighbors)
    qh.facet_mergeset lists the non-convex ridges in qh_newfacet_list
    qh.degen_mergeset is defined
    if qh.MERGEexact && !qh.POSTmerging,
      does not merge coplanar facets

  returns:
    deleted facets added to qh.visible_list with facet->visible
    deleted vertices added qh.delvertex_list with vertex->delvertex

  notes:
    unless !qh.MERGEindependent,
      merges facets in independent sets
    uses qh.newfacet_list as implicit argument since merges call qh_removefacet()
    [apr'19] restored qh_setdellast in place of qh_next_facetmerge.  Much faster for post-merge

  design:
    while merges occur
      for each merge in qh.facet_mergeset
        unless one of the facets was already merged in this pass
          merge the facets
        test merged facets for additional merges
        add merges to qh.facet_mergeset
        if qh.POSTmerging
          periodically call qh_reducevertices to reduce extra vertices and redundant vertices
      after each pass, if qh.VERTEXneighbors
        if qh.POSTmerging or was a merge with qh.hull_dim<=5
          call qh_reducevertices
          update qh.facet_mergeset if degenredundant merges
      if 'Qv' and qh.POSTmerging
        test vertex neighbors for convexity
*/
void qh_all_merges(qhT *qh, boolT othermerge, boolT vneighbors) {
  facetT *facet1, *facet2, *newfacet;
  mergeT *merge;
  boolT wasmerge= False, isreduce;
  void **freelistp;  /* used if !qh_NOmem by qh_memfree_() */
  vertexT *vertex;
  realT angle, distance;
  mergeType mergetype;
  int numcoplanar=0, numconcave=0, numconcavecoplanar= 0, numdegenredun= 0, numnewmerges= 0, numtwisted= 0;

  trace2((qh, qh->ferr, 2010, "qh_all_merges: starting to merge %d facet and %d degenerate merges for new facets f%d, othermerge? %d\n",
            qh_setsize(qh, qh->facet_mergeset), qh_setsize(qh, qh->degen_mergeset), getid_(qh->newfacet_list), othermerge));

  while (True) {
    wasmerge= False;
    /* Start of while loop to perform facet merges */
    while (qh_setsize(qh, qh->facet_mergeset) > 0 || qh_setsize(qh, qh->degen_mergeset) > 0) {
        // 循环直到没有待合并的面或者退化面集合中的元素
        if (qh_setsize(qh, qh->degen_mergeset) > 0) {
            // 如果退化面集合中有元素，则执行退化冗余合并操作，并增加相应计数
            numdegenredun += qh_merge_degenredundant(qh);
            wasmerge = True;  // 设置合并标志为真
        }
        // 循环处理面集合中的每一个合并元素
        while ((merge = (mergeT *)qh_setdellast(qh->facet_mergeset))) {
            facet1 = merge->facet1;
            facet2 = merge->facet2;
            vertex = merge->vertex1;  // 对于 qh.facet_mergeset，此处的 vertex1 未使用
            mergetype = merge->mergetype;
            angle = merge->angle;
            distance = merge->distance;
            qh_memfree_(qh, merge, (int)sizeof(mergeT), freelistp);   // 释放 merge 对象内存，merge 对象现在无效
            // 如果其中一个面被标记为可见，则跳过当前合并操作
            if (facet1->visible || facet2->visible) {
                trace3((qh, qh->ferr, 3045, "qh_all_merges: drop merge of f%d (del? %d) into f%d (del? %d) mergetype %d, dist %4.4g, angle %4.4g.  One or both facets is deleted\n",
                        facet1->id, facet1->visible, facet2->id, facet2->visible, mergetype, distance, angle));
                continue;
            } else if (mergetype == MRGcoplanar || mergetype == MRGanglecoplanar) {
                // 如果是共面或角共面合并，并且设置了独立合并标志，检查是否在独立集合中
                if (qh->MERGEindependent) {
                    if ((!facet1->tested && facet1->newfacet)
                        || (!facet2->tested && facet2->newfacet)) {
                        trace3((qh, qh->ferr, 3064, "qh_all_merges: drop merge of f%d (tested? %d) into f%d (tested? %d) mergetype %d, dist %2.2g, angle %4.4g.  Merge independent sets of coplanar merges\n",
                                facet1->id, facet1->visible, facet2->id, facet2->visible, mergetype, distance, angle));
                        continue;
                    }
                }
            }
            // 记录合并操作的跟踪信息
            trace3((qh, qh->ferr, 3047, "qh_all_merges: merge f%d and f%d type %d dist %2.2g angle %4.4g\n",
                    facet1->id, facet2->id, mergetype, distance, angle));
            // 根据合并类型调用对应的合并函数
            if (mergetype == MRGtwisted)
                qh_merge_twisted(qh, facet1, facet2);
            else
                qh_merge_nonconvex(qh, facet1, facet2, mergetype);
            // 增加合并计数并执行退化冗余合并操作
            numnewmerges++;
            numdegenredun += qh_merge_degenredundant(qh);
            wasmerge = True;  // 设置合并标志为真
            // 根据合并类型更新不同类型合并的计数
            if (mergetype == MRGconcave)
                numconcave++;
            else if (mergetype == MRGconcavecoplanar)
                numconcavecoplanar++;
            else if (mergetype == MRGtwisted)
                numtwisted++;
            else if (mergetype == MRGcoplanar || mergetype == MRGanglecoplanar)
                numcoplanar++;
            else {
                // 若合并类型不在预期范围内，输出错误信息并退出程序
                qh_fprintf(qh, qh->ferr, 6394, "qhull internal error (qh_all_merges): expecting concave, coplanar, or twisted merge.  Got merge f%d f%d v%d mergetype %d\n",
                           getid_(facet1), getid_(facet2), getid_(vertex), mergetype);
                qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
            }
        } /* while qh_setdellast */

        // 如果启用了后合并并且当前的凸壳维度小于等于指定维度，且新增合并次数超过指定阈值，则执行顶点减少操作
        if (qh->POSTmerging && qh->hull_dim <= qh_DIMreduceBuild
            && numnewmerges > qh_MAXnewmerges) {
            numnewmerges = 0;
            wasmerge = othermerge = False;
            qh_reducevertices(qh);  // 否则，大量的后合并操作会变得太慢
        }
        // 获取新的面合并集合并设置到 qh.newfacet_list
        qh_getmergeset(qh, qh->newfacet_list); /* qh.facet_mergeset */
    } /* while facet_mergeset or degen_mergeset */
    // 如果还存在待处理的 facet_mergeset 或 degen_mergeset，则继续循环

    if (qh->VERTEXneighbors) {  /* at least one merge */
      // 如果存在至少一个顶点邻居被合并
      isreduce= False;
      // 初始化标志变量 isreduce 为 False
      if (qh->POSTmerging && qh->hull_dim >= 4) {
        // 如果启用了后合并，并且凸壳维度大于等于 4
        isreduce= True;
        // 设置 isreduce 为 True
      }else if (qh->POSTmerging || !qh->MERGEexact) {
        // 或者如果启用了后合并，或者不要求精确合并
        if ((wasmerge || othermerge) && qh->hull_dim > 2 && qh->hull_dim <= qh_DIMreduceBuild)
          // 如果曾经进行过合并或其他合并，并且凸壳维度大于 2 且小于等于 qh_DIMreduceBuild
          isreduce= True;
          // 设置 isreduce 为 True
      }
      if (isreduce) {
        // 如果需要进行减少操作
        wasmerge= othermerge= False;
        // 将 wasmerge 和 othermerge 置为 False
        if (qh_reducevertices(qh)) {
          // 如果成功减少了顶点
          qh_getmergeset(qh, qh->newfacet_list); /* facet_mergeset */
          // 获取新的合并集合 facet_mergeset
          continue;
          // 继续下一轮循环
        }
      }
    }
    if (vneighbors && qh_test_vneighbors(qh /* qh.newfacet_list */))
      // 如果存在顶点邻居，并且需要测试顶点邻居
      continue;
      // 继续下一轮循环

    break;
    // 跳出循环

  } /* while (True) */
  // 结束主循环

  if (wasmerge || othermerge) {
    // 如果曾经进行过合并或其他合并
    trace3((qh, qh->ferr, 3033, "qh_all_merges: skip qh_reducevertices due to post-merging, no qh.VERTEXneighbors (%d), or hull_dim %d ==2 or >%d\n", qh->VERTEXneighbors, qh->hull_dim, qh_DIMreduceBuild))
    // 输出跟踪信息
    FORALLnew_facets {
      newfacet->newmerge= False;
      // 将每个新的 facet 的 newmerge 属性置为 False
    }
  }

  if (qh->CHECKfrequently && !qh->MERGEexact) {
    // 如果需要频繁检查且不需要精确合并
    qh->old_randomdist= qh->RANDOMdist;
    // 保存旧的随机距离设置
    qh->RANDOMdist= False;
    // 禁用随机距离
    qh_checkconvex(qh, qh->newfacet_list, qh_ALGORITHMfault);
    // 检查凸性
    /* qh_checkconnect(qh); [this is slow and it changes the facet order] */
    // 检查连接性（此操作较慢且会改变 facet 的顺序）
    qh->RANDOMdist= qh->old_randomdist;
    // 恢复随机距离设置
  }

  trace1((qh, qh->ferr, 1009, "qh_all_merges: merged %d coplanar %d concave %d concavecoplanar %d twisted facets and %d degen or redundant facets.\n",
    numcoplanar, numconcave, numconcavecoplanar, numtwisted, numdegenredun));
  // 输出跟踪信息，显示已合并的各种类型的 facets 数量

  if (qh->IStracing >= 4 && qh->num_facets < 500)
    // 如果跟踪级别大于等于 4 且 facets 数量小于 500
    qh_printlists(qh);
    // 打印当前的 lists
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="all_vertexmerges">-</a>

  qh_all_vertexmerges(qh, apexpointid, facet, &retryfacet )
    merge vertices in qh.vertex_mergeset and subsequent merges

  returns:
    returns retryfacet for facet (if defined)
    updates qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices
    mergesets are empty
    if merges, resets facet lists

  notes:
    called from qh_qhull, qh_addpoint, and qh_buildcone_mergepinched
    vertex merges occur after facet merges and qh_resetlists

  design:
    while merges in vertex_mergeset (MRGvertices)
      merge a pair of pinched vertices
      update vertex neighbors
      merge non-convex and degenerate facets and check for ridges with duplicate vertices
      partition outside points of deleted, "visible" facets
*/
void qh_all_vertexmerges(qhT *qh, int apexpointid, facetT *facet, facetT **retryfacet) {
  int numpoints; /* ignore count of partitioned points.  Used by qh_addpoint for Zpbalance */

  if (retryfacet)
    *retryfacet= facet;
  while (qh_setsize(qh, qh->vertex_mergeset) > 0) {
    trace1((qh, qh->ferr, 1057, "qh_all_vertexmerges: starting to merge %d vertex merges for apex p%d facet f%d\n",
            qh_setsize(qh, qh->vertex_mergeset), apexpointid, getid_(facet)));
    if (qh->IStracing >= 4  && qh->num_facets < 1000)
      qh_printlists(qh);
    qh_merge_pinchedvertices(qh, apexpointid /* qh.vertex_mergeset, visible_list, newvertex_list, newfacet_list */);
    qh_update_vertexneighbors(qh); /* update neighbors of qh.newvertex_list from qh_newvertices for deleted facets on qh.visible_list */
                           /* test ridges and merge non-convex facets */
    qh_getmergeset(qh, qh->newfacet_list);
    qh_all_merges(qh, True, False); /* calls qh_reducevertices */
    if (qh->CHECKfrequently)
      qh_checkpolygon(qh, qh->facet_list);
    qh_partitionvisible(qh, !qh_ALL, &numpoints /* qh.visible_list qh.del_vertices*/);
    if (retryfacet)
      *retryfacet= qh_getreplacement(qh, *retryfacet);
    qh_deletevisible(qh /* qh.visible_list  qh.del_vertices*/);
    qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
    if (qh->IStracing >= 4  && qh->num_facets < 1000) {
      qh_printlists(qh);
      qh_checkpolygon(qh, qh->facet_list);
    }
  }
} /* all_vertexmerges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="appendmergeset">-</a>

  qh_appendmergeset(qh, facet, vertex, neighbor, mergetype, dist, angle )
    appends an entry to qh.facet_mergeset or qh.degen_mergeset
    if 'dist' is unknown, set it to 0.0
        if 'angle' is unknown, set it to 1.0 (coplanar)

  returns:
    merge appended to facet_mergeset or degen_mergeset
      sets ->degenerate or ->redundant if degen_mergeset

  notes:
    caller collects statistics and/or caller of qh_mergefacet
*/
    see: qh_test_appendmerge()

  design:
    allocate merge entry
    # 分配一个合并条目，用于后续操作

    if regular merge
      # 如果是常规的合并操作
      append to qh.facet_mergeset
      # 将合并条目追加到 qh.facet_mergeset 中

    else if degenerate merge and qh.facet_mergeset is all degenerate
      # 否则如果是退化合并，并且 qh.facet_mergeset 中全是退化面
      append to qh.degen_mergeset
      # 将合并条目追加到 qh.degen_mergeset 中

    else if degenerate merge
      # 否则如果是退化合并
      prepend to qh.degen_mergeset (merged last)
      # 将合并条目前置到 qh.degen_mergeset 中（最后合并）

    else if redundant merge
      # 否则如果是冗余合并
      append to qh.degen_mergeset
      # 将合并条目追加到 qh.degen_mergeset 中
/*
void qh_appendmergeset(qhT *qh, facetT *facet, facetT *neighbor, mergeType mergetype, coordT dist, realT angle) {
    mergeT *merge, *lastmerge;  // 定义两个指向mergeT结构体的指针变量，merge和lastmerge
    void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */  // 用于内存分配的指针，如果qh_NOmem未设置
    const char *mergename;  // 定义一个指向字符常量的指针变量mergename

    // 如果facet或neighbor已经标记为redundant，并且mergetype不是MRGmirror，则忽略该次合并操作
    if ((facet->redundant && mergetype != MRGmirror) || neighbor->redundant) {
        trace3((qh, qh->ferr, 3051, "qh_appendmergeset: f%d is already redundant (%d) or f%d is already redundant (%d).  Ignore merge f%d and f%d type %d\n",
            facet->id, facet->redundant, neighbor->id, neighbor->redundant, facet->id, neighbor->id, mergetype));
        return;  // 返回，不执行合并操作
    }
    
    // 如果facet已经标记为degenerate，并且mergetype是MRGdegen，则忽略该次合并操作
    if (facet->degenerate && mergetype == MRGdegen) {
        trace3((qh, qh->ferr, 3077, "qh_appendmergeset: f%d is already degenerate.  Ignore merge f%d type %d (MRGdegen)\n",
            facet->id, facet->id, mergetype));
        return;  // 返回，不执行合并操作
    }
    
    // 如果qh.facet_mergeset或qh.degen_mergeset为空，则报错并退出程序
    if (!qh->facet_mergeset || !qh->degen_mergeset) {
        qh_fprintf(qh, qh->ferr, 6403, "qhull internal error (qh_appendmergeset): expecting temp set defined for qh.facet_mergeset (0x%x) and qh.degen_mergeset (0x%x).  Got NULL\n",
            qh->facet_mergeset, qh->degen_mergeset);
        /* otherwise qh_setappend creates a new set that is not freed by qh_freebuild() */
        qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    
    // 如果neighbor被翻转了，而facet没有被翻转，并且mergetype不是MRGdupridge，则报错并退出程序
    if (neighbor->flipped && !facet->flipped) {
        if (mergetype != MRGdupridge) {
            qh_fprintf(qh, qh->ferr, 6355, "qhull internal error (qh_appendmergeset): except for MRGdupridge, cannot merge a non-flipped facet f%d into flipped f%d, mergetype %d, dist %4.4g\n",
                facet->id, neighbor->id, mergetype, dist);
            qh_errexit(qh, qh_ERRqhull, NULL, NULL);
        } else {
            trace2((qh, qh->ferr, 2106, "qh_appendmergeset: dupridge will merge a non-flipped facet f%d into flipped f%d, dist %4.4g\n",
                facet->id, neighbor->id, dist));
        }
    }
    
    // 分配mergeT结构体的内存空间，并将其初始化
    qh_memalloc_(qh, (int)sizeof(mergeT), freelistp, merge, mergeT);
    merge->angle= angle;  // 设置merge的角度
    merge->distance= dist;  // 设置merge的距离
    merge->facet1= facet;  // 设置merge的第一个facet
    merge->facet2= neighbor;  // 设置merge的第二个facet
    merge->vertex1= NULL;  // 初始化merge的第一个顶点
    merge->vertex2= NULL;  // 初始化merge的第二个顶点
    merge->ridge1= NULL;  // 初始化merge的第一个ridge
    merge->ridge2= NULL;  // 初始化merge的第二个ridge
    merge->mergetype= mergetype;  // 设置merge的合并类型
    
    // 根据mergetype设置mergename，用于跟踪输出
    if(mergetype > 0 && mergetype <= sizeof(mergetypes))
        mergename= mergetypes[mergetype];
    else
        mergename= mergetypes[MRGnone];
    
    // 根据不同的mergetype执行不同的操作
    if (mergetype < MRGdegen)
        qh_setappend(qh, &(qh->facet_mergeset), merge);  // 将merge追加到qh.facet_mergeset集合中
    else if (mergetype == MRGdegen) {
        facet->degenerate= True;  // 标记facet为degenerate
        if (!(lastmerge= (mergeT *)qh_setlast(qh->degen_mergeset))
        || lastmerge->mergetype == MRGdegen)
            qh_setappend(qh, &(qh->degen_mergeset), merge);  // 将merge追加到qh.degen_mergeset集合中
        else
            qh_setaddnth(qh, &(qh->degen_mergeset), 0, merge);  /* merged last */  // 将merge插入到qh.degen_mergeset集合的第一个位置
    } else if (mergetype == MRGredundant) {
        facet->redundant= True;  // 标记facet为redundant
        qh_setappend(qh, &(qh->degen_mergeset), merge);  // 将merge追加到qh.degen_mergeset集合中
    } else /* mergetype == MRGmirror */ {
        // 如果mergetype是MRGmirror，则继续后续的操作

        // 继续后续的操作
    # 检查是否存在重复或冗余的面（facet），如果存在则输出错误信息并退出程序
    if (facet->redundant || neighbor->redundant) {
      # 输出错误信息到错误流，指示 facet 或 neighbor 已经是一个镜像面（即 'redundant'）
      qh_fprintf(qh, qh->ferr, 6092, "qhull internal error (qh_appendmergeset): facet f%d or f%d is already a mirrored facet (i.e., 'redundant')\n",
           facet->id, neighbor->id);
      # 退出程序，指示 Qhull 内部错误，并提供相关 facet 和 neighbor 的信息
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    
    # 检查 facet 和 neighbor 是否具有相同的顶点集合，如果不是则输出错误信息并退出程序
    if (!qh_setequal(facet->vertices, neighbor->vertices)) {
      # 输出错误信息到错误流，指示镜像面 facet 和 neighbor 的顶点集合不相同
      qh_fprintf(qh, qh->ferr, 6093, "qhull internal error (qh_appendmergeset): mirrored facets f%d and f%d do not have the same vertices\n",
           facet->id, neighbor->id);
      # 退出程序，指示 Qhull 内部错误，并提供相关 facet 和 neighbor 的信息
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    
    # 将 facet 和 neighbor 标记为冗余（redundant）
    facet->redundant= True;
    neighbor->redundant= True;
    
    # 将当前的 merge 结构追加到 Qhull 的 degen_mergeset 集合中
    qh_setappend(qh, &(qh->degen_mergeset), merge);
  }
  
  # 根据 merge 的类型不同，输出不同类型的调试信息到跟踪日志
  if (merge->mergetype >= MRGdegen) {
    # 输出简略的 merge 信息到跟踪日志，包括 merge 的类型和集合大小
    trace3((qh, qh->ferr, 3044, "qh_appendmergeset: append merge f%d and f%d type %d (%s) to qh.degen_mergeset (size %d)\n",
      merge->facet1->id, merge->facet2->id, merge->mergetype, mergename, qh_setsize(qh, qh->degen_mergeset)));
  } else {
    # 输出详细的 merge 信息到跟踪日志，包括 merge 的类型、距离和角度等信息，以及集合大小
    trace3((qh, qh->ferr, 3027, "qh_appendmergeset: append merge f%d and f%d type %d (%s) dist %2.2g angle %4.4g to qh.facet_mergeset (size %d)\n",
      merge->facet1->id, merge->facet2->id, merge->mergetype, mergename, merge->distance, merge->angle, qh_setsize(qh, qh->facet_mergeset)));
  }
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="appendmergeset">-</a>

  qh_appendvertexmerge(qh, vertex, vertex2, mergetype, distance, ridge1, ridge2 )
    将一个顶点合并操作追加到 qh.vertex_mergeset 中
    MRGsubridge 包含两个边脊 (来自 MRGdupridge)
    MRGvertices 包含两个边脊

  notes:
    被 qh_getpinchedmerges 调用用于 MRGsubridge
    被 qh_maybe_duplicateridge 和 qh_maybe_duplicateridges 调用用于 MRGvertices
    是向 qh.vertex_mergeset 添加顶点合并的唯一方式
    被 qh_next_vertexmerge 检查
*/
void qh_appendvertexmerge(qhT *qh, vertexT *vertex, vertexT *destination, mergeType mergetype, realT distance, ridgeT *ridge1, ridgeT *ridge2) {
  mergeT *merge;
  void **freelistp; /* 如果 !qh_NOmem 时由 qh_memalloc_() 使用 */
  const char *mergename;

  if (!qh->vertex_mergeset) {
    qh_fprintf(qh, qh->ferr, 6387, "qhull internal error (qh_appendvertexmerge): expecting temp set defined for qh.vertex_mergeset (0x%x).  Got NULL\n",
      qh->vertex_mergeset);
    /* 否则 qh_setappend 会创建一个新的集合，qh_freebuild() 不会释放它 */
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh_memalloc_(qh, (int)sizeof(mergeT), freelistp, merge, mergeT);
  merge->angle= qh_ANGLEnone;
  merge->distance= distance;
  merge->facet1= NULL;
  merge->facet2= NULL;
  merge->vertex1= vertex;
  merge->vertex2= destination;
  merge->ridge1= ridge1;
  merge->ridge2= ridge2;
  merge->mergetype= mergetype;
  if(mergetype > 0 && mergetype <= sizeof(mergetypes))
    mergename= mergetypes[mergetype];
  else
    mergename= mergetypes[MRGnone];
  if (mergetype == MRGvertices) {
    if (!ridge1 || !ridge2 || ridge1 == ridge2) {
      qh_fprintf(qh, qh->ferr, 6106, "qhull internal error (qh_appendvertexmerge): expecting two distinct ridges for MRGvertices.  Got r%d r%d\n",
        getid_(ridge1), getid_(ridge2));
      qh_errexit(qh, qh_ERRqhull, NULL, ridge1);
    }
  }
  qh_setappend(qh, &(qh->vertex_mergeset), merge);
  trace3((qh, qh->ferr, 3034, "qh_appendvertexmerge: append merge v%d into v%d r%d r%d dist %2.2g type %d (%s)\n",
    vertex->id, destination->id, getid_(ridge1), getid_(ridge2), distance, merge->mergetype, mergename));
} /* appendvertexmerge */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="basevertices">-</a>

  qh_basevertices(qh, samecycle )
    返回 samecycle 的基本顶点的临时集合
    samecycle 是循环中的第一个 facet
    假设 apex 是 samecycle->vertices 的第一个

  returns:
    vertices(settemp)
    所有 ->seen 都被清除

  notes:
    使用 qh_vertex_visit;

  design:
    对于 samecycle 中的每个 facet
      对于 facet->vertices 中的每个未见过的顶点
        追加到结果中
*/
setT *qh_basevertices(qhT *qh, facetT *samecycle) {
  facetT *same;  // 定义一个 facetT 类型的指针变量 same，用于遍历同一个循环中的面
  vertexT *apex, *vertex, **vertexp;  // 定义顶点变量：apex 表示顶点，vertex 表示顶点，vertexp 表示顶点指针数组
  setT *vertices= qh_settemp(qh, qh->TEMPsize);  // 创建一个临时的顶点集合，用于存储结果顶点集合的指针

  apex= SETfirstt_(samecycle->vertices, vertexT);  // 从同一个循环中的顶点集合中获取第一个顶点赋给 apex
  apex->visitid= ++qh->vertex_visit;  // 给 apex 的 visitid 属性赋值为当前的顶点访问编号
  FORALLsame_cycle_(samecycle) {  // 循环遍历同一个循环中的每个面
    if (same->mergeridge)  // 如果 same 面有合并脊
      continue;  // 跳过当前面的处理
    FOREACHvertex_(same->vertices) {  // 遍历当前面 same 的顶点集合中的每个顶点
      if (vertex->visitid != qh->vertex_visit) {  // 如果顶点的 visitid 不等于当前的顶点访问编号
        qh_setappend(qh, &vertices, vertex);  // 将顶点添加到结果顶点集合 vertices 中
        vertex->visitid= qh->vertex_visit;  // 更新顶点的 visitid 为当前的顶点访问编号
        vertex->seen= False;  // 将顶点的 seen 属性设置为 False
      }
    }
  }
  trace4((qh, qh->ferr, 4019, "qh_basevertices: found %d vertices\n",
         qh_setsize(qh, vertices)));  // 记录找到的顶点数量的跟踪信息
  return vertices;  // 返回存储结果顶点集合的指针
} /* basevertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="check_dupridge">-</a>

  qh_check_dupridge(qh, facet1, dist1, facet2, dist2 )
    Check dupridge between facet1 and facet2 for wide merge
    dist1 is the maximum distance of facet1's vertices to facet2
    dist2 is the maximum distance of facet2's vertices to facet1

  returns
    Level 1 log of the dupridge with the minimum distance between vertices
    Throws error if the merge will increase the maximum facet width by qh_WIDEduplicate (100x)

  notes:
    only called from qh_forcedmerges
*/
void qh_check_dupridge(qhT *qh, facetT *facet1, realT dist1, facetT *facet2, realT dist2) {
  vertexT *vertex, **vertexp, *vertexA, **vertexAp;  // 定义顶点变量和顶点指针数组
  realT dist, innerplane, mergedist, outerplane, prevdist, ratio, vertexratio;  // 定义距离、平面距离、合并距离等实数变量
  realT minvertex= REALmax;  // 初始化 minvertex 为一个很大的实数值

  mergedist= fmin_(dist1, dist2);  // 计算 dist1 和 dist2 的最小值赋给 mergedist
  qh_outerinner(qh, NULL, &outerplane, &innerplane);  /* ratio from qh_printsummary */  // 调用 qh_outerinner 函数计算外部平面和内部平面的比率
  FOREACHvertex_(facet1->vertices) {     /* The dupridge is between facet1 and facet2, so either facet can be tested */  // 遍历 facet1 的顶点集合
    FOREACHvertexA_(facet1->vertices) {  // 再次遍历 facet1 的顶点集合
      if (vertex > vertexA){   /* Test each pair once */  // 如果 vertex 大于 vertexA，则测试每对顶点一次
        dist= qh_pointdist(vertex->point, vertexA->point, qh->hull_dim);  // 计算顶点 vertex 和 vertexA 之间的距离
        minimize_(minvertex, dist);  // 更新最小顶点距离 minvertex
        /* Not quite correct.  A facet may have a dupridge and another pair of nearly adjacent vertices. */
      }
    }
  }
  prevdist= fmax_(outerplane, innerplane);  // 计算外部平面和内部平面的最大值赋给 prevdist
  maximize_(prevdist, qh->ONEmerge + qh->DISTround);  // 将 ONEmerge 和 DISTround 的和与 prevdist 取最大值
  maximize_(prevdist, qh->MINoutside + qh->DISTround);  // 将 MINoutside 和 DISTround 的和与 prevdist 取最大值
  ratio= mergedist/prevdist;  // 计算 mergedist 与 prevdist 的比率
  vertexratio= minvertex/prevdist;  // 计算 minvertex 与 prevdist 的比率
  trace0((qh, qh->ferr, 16, "qh_check_dupridge: dupridge between f%d and f%d (vertex dist %2.2g), dist %2.2g, reverse dist %2.2g, ratio %2.2g while processing p%d\n",
        facet1->id, facet2->id, minvertex, dist1, dist2, ratio, qh->furthest_id));  // 记录 dupridge 的跟踪信息
  if (ratio > qh_WIDEduplicate) {  // 如果 ratio 大于设定的 WIDEduplicate 值
    qh_fprintf(qh, qh->ferr, 6271, "qhull topology error (qh_check_dupridge): wide merge (%.1fx wider) due to dupridge between f%d and f%d (vertex dist %2.2g), merge dist %2.2g, while processing p%d\n- Allow error with option 'Q12'\n",
      ratio, facet1->id, facet2->id, minvertex, mergedist, qh->furthest_id);  // 输出拓扑错误信息
    # 如果顶点比率小于设定的 qh_WIDEpinched 值，则执行以下操作
    if (vertexratio < qh_WIDEpinched)
      # 输出一条警告消息到 qh->ferr 流，提醒用户可以尝试使用实验选项 merge-pinched-vertices ('Q14') 来避免这个错误，
      # 该选项会合并几乎相邻的顶点。
      qh_fprintf(qh, qh->ferr, 8145, "- Experimental option merge-pinched-vertices ('Q14') may avoid this error.  It merges nearly adjacent vertices.\n");
    
    # 如果 qh->DELAUNAY 为真，则执行以下操作
    if (qh->DELAUNAY)
      # 输出一条建议消息到 qh->ferr 流，建议用户可以为输入的站点使用一个边界框来减少这个错误。
      qh_fprintf(qh, qh->ferr, 8145, "- A bounding box for the input sites may alleviate this error.\n");
    
    # 如果不允许宽容忽略的情况，执行以下操作
    if (!qh->ALLOWwide)
      # 引发一个错误退出，使用错误代码 qh_ERRwide，并传入 facet1 和 facet2 两个参数
      qh_errexit2(qh, qh_ERRwide, facet1, facet2);
  }
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkconnect">-</a>

  qh_checkconnect(qh)
    check that new facets are connected
    new facets are on qh.newfacet_list

  notes:
    this is slow and it changes the order of the facets
    uses qh.visit_id

  design:
    move first new facet to end of qh.facet_list
    for all newly appended facets
      append unvisited neighbors to end of qh.facet_list
    for all new facets
      report error if unvisited
*/
void qh_checkconnect(qhT *qh /* qh.newfacet_list */) {
  facetT *facet, *newfacet, *errfacet= NULL, *neighbor, **neighborp;

  facet= qh->newfacet_list;
  qh_removefacet(qh, facet);  // 从 qh.newfacet_list 中移除第一个新面片
  qh_appendfacet(qh, facet);  // 将第一个新面片添加到 qh.facet_list 的末尾
  facet->visitid= ++qh->visit_id;  // 设置第一个新面片的 visitid 为当前的 visit_id

  FORALLfacet_(facet) {  // 遍历所有面片
    FOREACHneighbor_(facet) {  // 遍历当前面片的所有邻居
      if (neighbor->visitid != qh->visit_id) {  // 如果邻居的 visitid 不等于当前的 visit_id
        qh_removefacet(qh, neighbor);  // 从 qh.facet_list 中移除邻居
        qh_appendfacet(qh, neighbor);  // 将邻居添加到 qh.facet_list 的末尾
        neighbor->visitid= qh->visit_id;  // 设置邻居的 visitid 为当前的 visit_id
      }
    }
  }

  FORALLnew_facets {  // 遍历所有新面片
    if (newfacet->visitid == qh->visit_id)  // 如果新面片的 visitid 等于当前的 visit_id
      break;
    qh_fprintf(qh, qh->ferr, 6094, "qhull internal error (qh_checkconnect): f%d is not attached to the new facets\n",
         newfacet->id);
    errfacet= newfacet;  // 记录错误的新面片
  }
  if (errfacet)
    qh_errexit(qh, qh_ERRqhull, errfacet, NULL);  // 如果有错误的新面片，触发 Qhull 的错误退出处理
} /* checkconnect */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkdelfacet">-</a>

  qh_checkdelfacet(qh, facet, mergeset )
    check that mergeset does not reference facet

*/
void qh_checkdelfacet(qhT *qh, facetT *facet, setT *mergeset) {
  mergeT *merge, **mergep;

  FOREACHmerge_(mergeset) {  // 遍历 mergeset 中的每个 merge
    if (merge->facet1 == facet || merge->facet2 == facet) {  // 如果 merge 引用了 facet
      qh_fprintf(qh, qh->ferr, 6390, "qhull internal error (qh_checkdelfacet): cannot delete f%d.  It is referenced by merge f%d f%d mergetype %d\n",
        facet->id, merge->facet1->id, getid_(merge->facet2), merge->mergetype);
      qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);  // 触发 Qhull 的双面片错误退出处理
    }
  }
} /* checkdelfacet */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkdelridge">-</a>

  qh_checkdelridge(qh)
    check that qh_delridge_merge is not needed for deleted ridges

    notes:
      called from qh_mergecycle, qh_makenewfacets, qh_attachnewfacets
      errors if qh.vertex_mergeset is non-empty
      errors if any visible or new facet has a ridge with r.nonconvex set
      assumes that vertex.delfacet is not needed
*/
void qh_checkdelridge(qhT *qh /* qh.visible_facets, vertex_mergeset */) {
  facetT *newfacet, *visible;
  ridgeT *ridge, **ridgep;

  if (!SETempty_(qh->vertex_mergeset)) {  // 如果 qh.vertex_mergeset 不为空
    qh_fprintf(qh, qh->ferr, 6382, "qhull internal error (qh_checkdelridge): expecting empty qh.vertex_mergeset in order to avoid calling qh_delridge_merge.  Got %d merges\n", qh_setsize(qh, qh->vertex_mergeset));
    # 调用 qh_errexit 函数，用于处理 Qhull 库的错误退出情况
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }

  # 遍历所有新创建的凸包面（facet）
  FORALLnew_facets {
    # 遍历当前新凸包面的所有边界（ridge）
    FOREACHridge_(newfacet->ridges) {
      # 检查边界是否标记为非凸（nonconvex）
      if (ridge->nonconvex) {
        # 输出错误消息至 qh->ferr 流，指出发现不符合预期的非凸边界标记
        qh_fprintf(qh, qh->ferr, 6313, "qhull internal error (qh_checkdelridge): unexpected 'nonconvex' flag for ridge r%d in newfacet f%d.  Otherwise need to call qh_delridge_merge\n",
           ridge->id, newfacet->id);
        # 用 qh_errexit 处理 Qhull 错误退出，传递相关的凸包面和边界对象
        qh_errexit(qh, qh_ERRqhull, newfacet, ridge);
      }
    }
  }

  # 遍历所有可见的凸包面
  FORALLvisible_facets {
    # 遍历当前可见凸包面的所有边界
    FOREACHridge_(visible->ridges) {
      # 检查边界是否标记为非凸
      if (ridge->nonconvex) {
        # 输出错误消息至 qh->ferr 流，指出发现不符合预期的非凸边界标记
        qh_fprintf(qh, qh->ferr, 6385, "qhull internal error (qh_checkdelridge): unexpected 'nonconvex' flag for ridge r%d in visible facet f%d.  Otherwise need to call qh_delridge_merge\n",
          ridge->id, visible->id);
        # 用 qh_errexit 处理 Qhull 错误退出，传递相关的凸包面和边界对象
        qh_errexit(qh, qh_ERRqhull, visible, ridge);
      }
    }
  }
/* checkdelridge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkzero">-</a>

  qh_checkzero(qh, testall )
    check that facets are clearly convex for qh.DISTround with qh.MERGEexact

    if testall,
      test all facets for qh.MERGEexact post-merging
    else
      test qh.newfacet_list

    if qh.MERGEexact,
      allows coplanar ridges
      skips convexity test while qh.ZEROall_ok

  returns:
    True if all facets !flipped, !dupridge, normal
         if all horizon facets are simplicial
         if all vertices are clearly below neighbor
         if all opposite vertices of horizon are below
    clears qh.ZEROall_ok if any problems or coplanar facets

  notes:
    called by qh_premerge (qh.CHECKzero, 'C-0') and qh_qhull ('Qx')
    uses qh.vertex_visit
    horizon facets may define multiple new facets

  design:
    for all facets in qh.newfacet_list or qh.facet_list
      check for flagged faults (flipped, etc.)
    for all facets in qh.newfacet_list or qh.facet_list
      for each neighbor of facet
        skip horizon facets for qh.newfacet_list
        test the opposite vertex
      if qh.newfacet_list
        test the other vertices in the facet's horizon facet
*/
boolT qh_checkzero(qhT *qh, boolT testall) {
  facetT *facet, *neighbor;
  facetT *horizon, *facetlist;
  int neighbor_i, neighbor_n;
  vertexT *vertex, **vertexp;
  realT dist;

  if (testall)
    facetlist= qh->facet_list;
  else {
    facetlist= qh->newfacet_list;
    /* Check each facet in qh.newfacet_list */
    FORALLfacet_(facetlist) {
      /* Get the first neighbor (horizon facet) */
      horizon= SETfirstt_(facet->neighbors, facetT);
      /* Ensure the horizon facet is simplicial */
      if (!horizon->simplicial)
        goto LABELproblem;
      /* Check for flagged faults */
      if (facet->flipped || facet->dupridge || !facet->normal)
        goto LABELproblem;
    }
    /* Skip convexity test if qh.MERGEexact and qh.ZEROall_ok */
    if (qh->MERGEexact && qh->ZEROall_ok) {
      trace2((qh, qh->ferr, 2011, "qh_checkzero: skip convexity check until first pre-merge\n"));
      return True;
    }
  }

  /* Iterate over facets in facetlist (qh.newfacet_list or qh.facet_list) */
  FORALLfacet_(facetlist) {
    qh->vertex_visit++;
    horizon= NULL;
    /* Iterate over neighbors of current facet */
    FOREACHneighbor_i_(qh, facet) {
      /* Skip the first neighbor if not testing all facets */
      if (!neighbor_i && !testall) {
        horizon= neighbor;
        continue; /* horizon facet tested in qh_findhorizon */
      }
      /* Get the vertex opposite to the current neighbor */
      vertex= SETelemt_(facet->vertices, neighbor_i, vertexT);
      /* Mark the vertex as visited */
      vertex->visitid= qh->vertex_visit;
      zzinc_(Zdistzero);
      /* Calculate distance to the plane of the neighbor */
      qh_distplane(qh, vertex->point, neighbor, &dist);
      /* Check if the vertex is clearly below the neighbor */
      if (dist >= -2 * qh->DISTround) {  /* need 2x for qh_distround and 'Rn' for qh_checkconvex, same as qh.premerge_centrum */
        qh->ZEROall_ok= False;
        /* If not MERGEexact or testing all facets or dist > qh.DISTround, mark as non-convex */
        if (!qh->MERGEexact || testall || dist > qh->DISTround)
          goto LABELnonconvex;
      }
    }
    /* Continue to next facet in facetlist */
    // 如果不是测试全部并且存在horizon
    if (!testall && horizon) {
      // 遍历horizon的所有顶点
      FOREACHvertex_(horizon->vertices) {
        // 如果顶点的访问ID与qh->vertex_visit不同
        if (vertex->visitid != qh->vertex_visit) {
          // 增加计数器Zdistzero
          zzinc_(Zdistzero);
          // 计算顶点到facet平面的距离
          qh_distplane(qh, vertex->point, facet, &dist);
          // 如果距离大于等于-2 * qh->DISTround
          if (dist >= -2 * qh->DISTround) {
            // 设置qh->ZEROall_ok为False
            qh->ZEROall_ok= False;
            // 如果不是精确合并或者距离大于qh->DISTround
            if (!qh->MERGEexact || dist > qh->DISTround)
              // 跳转到LABELnonconvexhorizon标签处
              goto LABELnonconvexhorizon;
          }
          // 跳出循环
          break;
        }
      }
    }
  }
  // 输出调试信息到qh->ferr，用于检查零点测试的情况
  trace2((qh, qh->ferr, 2012, "qh_checkzero: testall %d, facets are %s\n", testall,
        (qh->MERGEexact && !testall) ?
           "not concave, flipped, or dupridge" : "clearly convex"));
  // 返回True表示测试通过
  return True;

 LABELproblem:
  // 设置qh->ZEROall_ok为False
  qh->ZEROall_ok= False;
  // 输出调试信息到qh->ferr，指示需要qh_premerge
  trace2((qh, qh->ferr, 2013, "qh_checkzero: qh_premerge is needed.  New facet f%d or its horizon f%d is non-simplicial, flipped, dupridge, or mergehorizon\n",
       facet->id, horizon->id));
  // 返回False表示存在问题
  return False;

 LABELnonconvex:
  // 输出调试信息到qh->ferr，指示facet和neighbor不是清晰凸的
  trace2((qh, qh->ferr, 2014, "qh_checkzero: facet f%d and f%d are not clearly convex.  v%d dist %.2g\n",
         facet->id, neighbor->id, vertex->id, dist));
  // 返回False表示facet和neighbor不是清晰凸的
  return False;

 LABELnonconvexhorizon:
  // 输出调试信息到qh->ferr，指示facet和horizon不是清晰凸的
  trace2((qh, qh->ferr, 2060, "qh_checkzero: facet f%d and horizon f%d are not clearly convex.  v%d dist %.2g\n",
      facet->id, horizon->id, vertex->id, dist));
  // 返回False表示facet和horizon不是清晰凸的
  return False;
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="compare_anglemerge">-</a>

  qh_compare_anglemerge( mergeA, mergeB )
    used by qsort() to order qh.facet_mergeset by mergetype and angle (qh.ANGLEmerge, 'Q1')
    lower numbered mergetypes done first (MRGcoplanar before MRGconcave)

  notes:
    qh_all_merges processes qh.facet_mergeset by qh_setdellast
    [mar'19] evaluated various options with eg/q_benchmark and merging of pinched vertices (Q14)
*/
int qh_compare_anglemerge(const void *p1, const void *p2) {
  const mergeT *a= *((mergeT *const*)p1), *b= *((mergeT *const*)p2);

  if (a->mergetype != b->mergetype)
    return (a->mergetype < b->mergetype ? 1 : -1); /* 选择MRGcoplanar（1）优先于MRGconcave（3） */
  else
    return (a->angle > b->angle ? 1 : -1);         /* 选择共面合并（1.0）优先于尖锐合并（-0.5） */
} /* compare_anglemerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="compare_facetmerge">-</a>

  qh_compare_facetmerge( mergeA, mergeB )
    used by qsort() to order merges by mergetype, first merge, first
    lower numbered mergetypes done first (MRGcoplanar before MRGconcave)
    if same merge type, flat merges are first

  notes:
    qh_all_merges processes qh.facet_mergeset by qh_setdellast
    [mar'19] evaluated various options with eg/q_benchmark and merging of pinched vertices (Q14)
*/
int qh_compare_facetmerge(const void *p1, const void *p2) {
  const mergeT *a= *((mergeT *const*)p1), *b= *((mergeT *const*)p2);

  if (a->mergetype != b->mergetype)
    return (a->mergetype < b->mergetype ? 1 : -1); /* 选择MRGcoplanar（1）优先于MRGconcave（3） */
  else if (a->mergetype == MRGanglecoplanar)
    return (a->angle > b->angle ? 1 : -1);         /* 如果是MRGanglecoplanar，选择共面合并（1.0）优先于尖锐合并（-0.5） */
  else
    return (a->distance < b->distance ? 1 : -1);   /* 选择平面（0.0）合并优先于宽阔（1e-10）合并 */
} /* compare_facetmerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="comparevisit">-</a>

  qh_comparevisit( vertexA, vertexB )
    used by qsort() to order vertices by their visitid

  notes:
    only called by qh_find_newvertex
*/
int qh_comparevisit(const void *p1, const void *p2) {
  const vertexT *a= *((vertexT *const*)p1), *b= *((vertexT *const*)p2);

  if (a->visitid > b->visitid)
    return 1;
  return -1;
} /* comparevisit */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="copynonconvex">-</a>

  qh_copynonconvex(qh, atridge )
    set non-convex flag on other ridges (if any) between same neighbors

  notes:
    may be faster if use smaller ridge set

  design:
    for each ridge of atridge's top facet
      if ridge shares the same neighbor
        set nonconvex flag
*/
void qh_copynonconvex(qhT *qh, ridgeT *atridge) {
  facetT *facet, *otherfacet;
  ridgeT *ridge, **ridgep;

  facet= atridge->top;  // 将顶部面设为给定边的顶部面
  otherfacet= atridge->bottom;  // 将底部面设为给定边的底部面
  atridge->nonconvex= False;  // 初始化边的非凸标志为假
  FOREACHridge_(facet->ridges) {  // 遍历顶部面的所有边
    if (otherfacet == ridge->top || otherfacet == ridge->bottom) {  // 如果底部面等于当前边的顶部或底部面
      if (ridge != atridge) {  // 如果当前边不是给定边本身
        ridge->nonconvex= True;  // 设置当前边的非凸标志为真
        trace4((qh, qh->ferr, 4020, "qh_copynonconvex: moved nonconvex flag from r%d to r%d between f%d and f%d\n",
                atridge->id, ridge->id, facet->id, otherfacet->id));  // 记录日志：移动了非凸标志从边 r%d 到 r%d，位于面 f%d 和 f%d 之间
        break;  // 结束循环
      }
    }
  }
} /* copynonconvex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="degen_redundant_facet">-</a>

  qh_degen_redundant_facet(qh, facet )
    check for a degenerate (too few neighbors) or redundant (subset of vertices) facet

  notes:
    called at end of qh_mergefacet, qh_renamevertex, and qh_reducevertices
    bumps vertex_visit
    called if a facet was redundant but no longer is (qh_merge_degenredundant)
    qh_appendmergeset() only appends first reference to facet (i.e., redundant)
    see: qh_test_redundant_neighbors, qh_maydropneighbor

  design:
    test for redundant neighbor
    test for degenerate facet
*/
void qh_degen_redundant_facet(qhT *qh, facetT *facet) {
  vertexT *vertex, **vertexp;
  facetT *neighbor, **neighborp;

  trace3((qh, qh->ferr, 3028, "qh_degen_redundant_facet: test facet f%d for degen/redundant\n",
          facet->id));  // 跟踪日志：测试面 f%d 是否退化或冗余
  if (facet->flipped) {
    trace2((qh, qh->ferr, 3074, "qh_degen_redundant_facet: f%d is flipped, will merge later\n", facet->id));  // 跟踪日志：面 f%d 已翻转，稍后将合并
    return;  // 如果面已翻转，直接返回
  }
  FOREACHneighbor_(facet) {  // 遍历面的所有邻居面
    if (neighbor->flipped) /* disallow merge of non-flipped into flipped, neighbor will be merged later */
      continue;  // 如果邻居面已翻转，跳过合并
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6357, "qhull internal error (qh_degen_redundant_facet): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
        facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);  // 输出错误消息并退出
    }
    qh->vertex_visit++;  // 增加顶点访问计数
    FOREACHvertex_(neighbor->vertices)
      vertex->visitid= qh->vertex_visit;  // 设置邻居面的所有顶点的访问标识为当前顶点访问计数
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit)
        break;  // 如果找到未访问的顶点，退出循环
    }
    if (!vertex) {
      trace2((qh, qh->ferr, 2015, "qh_degen_redundant_facet: f%d is contained in f%d.  merge\n", facet->id, neighbor->id));  // 跟踪日志：面 f%d 包含在面 f%d 中，合并
      qh_appendmergeset(qh, facet, neighbor, MRGredundant, 0.0, 1.0);  // 将面标记为冗余，并添加到合并集中
      return;  // 直接返回，不再继续检查
    }
  }
  if (qh_setsize(qh, facet->neighbors) < qh->hull_dim) {
    qh_appendmergeset(qh, facet, facet, MRGdegen, 0.0, 1.0);  // 将面标记为退化，并添加到合并集中
    trace2((qh, qh->ferr, 2016, "qh_degen_redundant_facet: f%d is degenerate.\n", facet->id));  // 跟踪日志：面 f%d 退化
  }
} /* degen_redundant_facet */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="delridge_merge">-</a>

  qh_delridge_merge(qh, ridge )
    delete ridge due to a merge

  notes:
    only called by merge_r.c (qh_mergeridges, qh_renameridgevertex)
    ridges also freed in qh_freeqhull and qh_mergecycle_ridges


# 仅由 merge_r.c 中的 qh_mergeridges 和 qh_renameridgevertex 调用
# ridges 在 qh_freeqhull 和 qh_mergecycle_ridges 中也被释放

  design:
    if needed, moves ridge.nonconvex to another ridge
    sets vertex.delridge for qh_reducevertices
    deletes ridge from qh.vertex_mergeset
    deletes ridge from its neighboring facets
    frees up its memory


# 设计：
# 如果需要的话，将 ridge.nonconvex 移动到另一个 ridge
# 为了 qh_reducevertices 设置 vertex.delridge
# 从 qh.vertex_mergeset 中删除 ridge
# 从其相邻的 facets 中删除 ridge
# 释放它的内存
/*
void qh_delridge_merge(qhT *qh, ridgeT *ridge) {
  vertexT *vertex, **vertexp;
  mergeT *merge;
  int merge_i, merge_n;

  trace3((qh, qh->ferr, 3036, "qh_delridge_merge: delete ridge r%d between f%d and f%d\n",
    ridge->id, ridge->top->id, ridge->bottom->id));
  if (ridge->nonconvex)
    qh_copynonconvex(qh, ridge);
  FOREACHvertex_(ridge->vertices)
    vertex->delridge= True;
  FOREACHmerge_i_(qh, qh->vertex_mergeset) {
    if (merge->ridge1 == ridge || merge->ridge2 == ridge) {
      trace3((qh, qh->ferr, 3029, "qh_delridge_merge: drop merge of v%d into v%d (dist %2.2g r%d r%d) due to deleted, duplicated ridge r%d\n",
        merge->vertex1->id, merge->vertex2->id, merge->distance, merge->ridge1->id, merge->ridge2->id, ridge->id));
      if (merge->ridge1 == ridge)
        merge->ridge2->mergevertex= False;
      else
        merge->ridge1->mergevertex= False;
      qh_setdelnth(qh, qh->vertex_mergeset, merge_i);
      merge_i--; merge_n--; /* next merge after deleted */
    }
  }
  qh_setdel(ridge->top->ridges, ridge);
  qh_setdel(ridge->bottom->ridges, ridge);
  qh_delridge(qh, ridge);
} /* delridge_merge */
*/

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="drop_mergevertex">-</a>

  qh_drop_mergevertex(qh, merge )

  clear mergevertex flags for ridges of a vertex merge
*/
void qh_drop_mergevertex(qhT *qh, mergeT *merge)
{
  if (merge->mergetype == MRGvertices) {
    merge->ridge1->mergevertex= False;
    merge->ridge1->mergevertex2= True;
    merge->ridge2->mergevertex= False;
    merge->ridge2->mergevertex2= True;
    trace3((qh, qh->ferr, 3032, "qh_drop_mergevertex: unset mergevertex for r%d and r%d due to dropped vertex merge v%d to v%d.  Sets mergevertex2\n",
      merge->ridge1->id, merge->ridge2->id, merge->vertex1->id, merge->vertex2->id));
  }
} /* drop_mergevertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="find_newvertex">-</a>

  qh_find_newvertex(qh, oldvertex, vertices, ridges )
    locate new vertex for renaming old vertex
    vertices is a set of possible new vertices
      vertices sorted by number of deleted ridges

  returns:
    newvertex or NULL
      each ridge includes both newvertex and oldvertex
    vertices without oldvertex sorted by number of deleted ridges
    qh.vertex_visit updated
    sets v.seen

  notes:
    called by qh_redundant_vertex due to vertex->delridge and qh_rename_sharedvertex
    sets vertex->visitid to 0..setsize() for vertices
    new vertex is in one of the ridges
    renaming will not cause a duplicate ridge
    renaming will minimize the number of deleted ridges
    newvertex may not be adjacent in the dual (though unlikely)

  design:
    for each vertex in vertices
      set vertex->visitid to number of ridges
    remove unvisited vertices
    set qh.vertex_visit above all possible values

*/
    # 按照脊的数量对顶点进行排序（最小化需要重命名的脊）
    sort vertices by number of ridges (minimize ridges that need renaming)
    
    # 将每个脊添加到 qh.hash_table 中
    add each ridge to qh.hash_table
    
    # 对于顶点集合中的每个顶点
    for each vertex in vertices
      # 找到第一个顶点，重命名后不会导致重复的脊
      find the first vertex that would not cause a duplicate ridge after a rename
/*
vertexT *qh_find_newvertex(qhT *qh, vertexT *oldvertex, setT *vertices, setT *ridges) {
*/
// 定义函数qh_find_newvertex，接受四个参数：qh为qhT类型指针，oldvertex为vertexT类型指针，vertices和ridges为setT类型指针
vertexT *vertex, **vertexp;
setT *newridges;
ridgeT *ridge, **ridgep;
int size, hashsize;
int hash;
unsigned int maxvisit;

#ifndef qh_NOtrace
if (qh->IStracing >= 4) {
// 如果定义了qh_NOtrace，并且qh->IStracing大于等于4
qh_fprintf(qh, qh->ferr, 8063, "qh_find_newvertex: find new vertex for v%d from ",
         oldvertex->id);
// 输出跟踪信息，显示正在查找新顶点的信息，包括旧顶点的id
FOREACHvertex_(vertices)
  qh_fprintf(qh, qh->ferr, 8064, "v%d ", vertex->id);
// 遍历vertices集合中的每个顶点，并输出顶点的id
FOREACHridge_(ridges)
  qh_fprintf(qh, qh->ferr, 8065, "r%d ", ridge->id);
// 遍历ridges集合中的每条脊，并输出脊的id
qh_fprintf(qh, qh->ferr, 8066, "\n");
// 输出换行符
}
#endif
FOREACHridge_(ridges) {
// 遍历ridges集合中的每条脊
FOREACHvertex_(ridge->vertices)
  vertex->seen= False;
// 遍历脊中的每个顶点，将顶点的seen属性设置为False
}
FOREACHvertex_(vertices) {
// 遍历vertices集合中的每个顶点
vertex->visitid= 0;  /* v.visitid will be number of ridges */
// 将顶点的visitid属性设置为0，用于存储顶点相连的脊的数量
vertex->seen= True;
// 将顶点的seen属性设置为True
}
FOREACHridge_(ridges) {
// 再次遍历ridges集合中的每条脊
FOREACHvertex_(ridge->vertices) {
  if (vertex->seen)
    vertex->visitid++;
// 如果顶点的seen属性为True，则增加顶点的visitid属性
}
}
FOREACHvertex_(vertices) {
// 再次遍历vertices集合中的每个顶点
if (!vertex->visitid) {
  qh_setdelnth(qh, vertices, SETindex_(vertices,vertex));
  vertexp--; /* repeat since deleted this vertex */
// 如果顶点的visitid为0，则从vertices集合中删除该顶点，并减少vertexp指针以便重新遍历
}
}
maxvisit= (unsigned int)qh_setsize(qh, ridges);
maximize_(qh->vertex_visit, maxvisit);
// 将qh_setsize(qh, ridges)的值转换为unsigned int类型，用于设置vertex_visit的最大值
if (!qh_setsize(qh, vertices)) {
// 如果vertices集合为空
trace4((qh, qh->ferr, 4023, "qh_find_newvertex: vertices not in ridges for v%d\n",
        oldvertex->id));
// 输出跟踪信息，指示顶点oldvertex->id没有在ridges中找到
return NULL;
// 返回空指针
}
qsort(SETaddr_(vertices, vertexT), (size_t)qh_setsize(qh, vertices),
              sizeof(vertexT *), qh_comparevisit);
// 对vertices集合中的顶点按照visitid属性排序
/* can now use qh->vertex_visit */
// 现在可以使用qh->vertex_visit属性
if (qh->PRINTstatistics) {
// 如果定义了PRINTstatistics
size= qh_setsize(qh, vertices);
zinc_(Zintersect);
zadd_(Zintersecttot, size);
zmax_(Zintersectmax, size);
// 增加和最大化相关的统计信息
}
hashsize= qh_newhashtable(qh, qh_setsize(qh, ridges));
// 根据ridges集合的大小创建新的哈希表
FOREACHridge_(ridges)
  qh_hashridge(qh, qh->hash_table, hashsize, ridge, oldvertex);
// 遍历ridges集合中的每条脊，并将脊与oldvertex相关联到哈希表中
FOREACHvertex_(vertices) {
// 再次遍历vertices集合中的每个顶点
newridges= qh_vertexridges(qh, vertex, !qh_ALL);
// 获取与顶点相关联的脊的集合
FOREACHridge_(newridges) {
  if (qh_hashridge_find(qh, qh->hash_table, hashsize, ridge, vertex, oldvertex, &hash)) {
    zinc_(Zvertexridge);
    break;
// 如果在哈希表中找到了与顶点和oldvertex相关的脊，则增加计数并终止循环
  }
}
qh_settempfree(qh, &newridges);
// 释放临时分配的newridges集合
if (!ridge)
  break;  /* found a rename */
// 如果没有找到相关的脊，则终止循环，表示找到了重命名
}
if (vertex) {
// 如果找到了顶点
/* counted in qh_renamevertex */
trace2((qh, qh->ferr, 2020, "qh_find_newvertex: found v%d for old v%d from %d vertices and %d ridges.\n",
  vertex->id, oldvertex->id, qh_setsize(qh, vertices), qh_setsize(qh, ridges)));
// 输出跟踪信息，显示找到的顶点id及其相关的顶点和脊的数量
}else {
zinc_(Zfindfail);
trace0((qh, qh->ferr, 14, "qh_find_newvertex: no vertex for renaming v%d (all duplicated ridges) during p%d\n",
  oldvertex->id, qh->furthest_id));
// 增加计数器并输出跟踪信息，指示未找到顶点用于重命名
}
qh_setfree(qh, &qh->hash_table);
// 释放哈希表资源
return vertex;
// 返回顶点指针
} /* find_newvertex */
    # 确定最佳的凹点顶点，将其重命名为其最近的邻近顶点
    # 重命名将从 newfacet_list 中移除重复的 MRGdupridge

  # 返回值：
  # 凹点顶点（apex 或 subridge）、最近的顶点（subridge 或邻近顶点）及它们之间的距离

  # 注意：
  # 只被 qh_getpinchedmerges 调用
  # 假定 qh.VERTEXneighbors 已定义
  # 参见 qh_findbest_ridgevertex

  # 设计思路：
  # 如果面片具有相同的顶点
  #   返回最近的顶点对
  # 否则
  #   subridge 是两个新面片的交集减去 apex
  #   subridge 包括 qh.hull_dim-2 个地平线顶点
  #   subridge 也是新面片的匹配脊（它的重复）
  #   确定离 apex 最近的顶点
  #   确定 subridge 顶点对中最近的一对
  #   对于 subridge 中的每个顶点
  #     确定离其最近的邻近顶点（不在 subridge 中）
/*
vertexT *qh_findbest_pinchedvertex(qhT *qh, mergeT *merge, vertexT *apex, vertexT **nearestp, coordT *distp /* qh.newfacet_list */) {
  // 定义函数 qh_findbest_pinchedvertex，返回类型为 vertexT 指针，参数包括 qh 结构体指针 qh、merge 结构体指针 merge、顶点结构体指针 apex、顶点结构体指针指针 nearestp、坐标类型指针 distp
  vertexT *vertex, **vertexp, *vertexA, **vertexAp;
  // 声明顶点结构体指针 vertex、顶点结构体指针数组 vertexp、顶点结构体指针 vertexA、顶点结构体指针数组 vertexAp
  vertexT *bestvertex= NULL, *bestpinched= NULL;
  // 声明并初始化顶点结构体指针 bestvertex 和 bestpinched
  setT *subridge, *maybepinched;
  // 声明集合结构体指针 subridge 和 maybepinched
  coordT dist, bestdist= REALmax;
  // 声明坐标类型变量 dist 和 bestdist，并初始化 bestdist 为 REALmax
  coordT pincheddist= (qh->ONEmerge+qh->DISTround)*qh_RATIOpinchedsubridge;
  // 声明并计算 pincheddist，其值为 (qh->ONEmerge+qh->DISTround)*qh_RATIOpinchedsubridge

  if (!merge->facet1->simplicial || !merge->facet2->simplicial) {
    // 如果 merge->facet1 或 merge->facet2 不是简单形式的面，则输出错误信息
    qh_fprintf(qh, qh->ferr, 6351, "qhull internal error (qh_findbest_pinchedvertex): expecting merge of adjacent, simplicial new facets.  f%d or f%d is not simplicial\n",
      merge->facet1->id, merge->facet2->id);
    // 使用 qh_fprintf 输出错误信息到 qh->ferr 流
    qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
    // 调用 qh_errexit2 结束程序执行，传入 qh 和错误码 qh_ERRqhull，以及错误相关的 merge->facet1 和 merge->facet2
  }
  subridge= qh_vertexintersect_new(qh, merge->facet1->vertices, merge->facet2->vertices); /* new setT.  No error_exit() */
  // 计算 merge->facet1->vertices 和 merge->facet2->vertices 的交集，结果存入 subridge
  if (qh_setsize(qh, subridge) == qh->hull_dim) { /* duplicate vertices */
    // 如果 subridge 中的顶点数等于 qh->hull_dim，则执行以下操作
    bestdist= qh_vertex_bestdist2(qh, subridge, &bestvertex, &bestpinched);
    // 调用 qh_vertex_bestdist2 计算 subridge 中顶点的最佳距离，并更新 bestdist、bestvertex 和 bestpinched
    if(bestvertex == apex) {
      // 如果 bestvertex 等于 apex，则交换 bestvertex 和 bestpinched
      bestvertex= bestpinched;
      bestpinched= apex;
    }
  }else {
    // 否则，如果 subridge 中顶点数不等于 qh->hull_dim - 2，则执行以下操作
    qh_setdel(subridge, apex);
    // 从 subridge 中删除 apex
    if (qh_setsize(qh, subridge) != qh->hull_dim - 2) {
      // 如果 subridge 中顶点数不等于 qh->hull_dim - 2，则输出错误信息
      qh_fprintf(qh, qh->ferr, 6409, "qhull internal error (qh_findbest_pinchedvertex): expecting subridge of qh.hull_dim-2 vertices for the intersection of new facets f%d and f%d minus their apex.  Got %d vertices\n",
          merge->facet1->id, merge->facet2->id, qh_setsize(qh, subridge));
      // 使用 qh_fprintf 输出错误信息到 qh->ferr 流
      qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
      // 调用 qh_errexit2 结束程序执行，传入 qh 和错误码 qh_ERRqhull，以及错误相关的 merge->facet1 和 merge->facet2
    }
    FOREACHvertex_(subridge) {
      // 遍历 subridge 中的每个顶点
      dist= qh_pointdist(vertex->point, apex->point, qh->hull_dim);
      // 计算 vertex->point 到 apex->point 的距离，存入 dist
      if (dist < bestdist) {
        // 如果 dist 小于 bestdist，则更新 bestpinched 和 bestvertex
        bestpinched= apex;
        bestvertex= vertex;
        bestdist= dist;
      }
    }
    if (bestdist > pincheddist) {
      // 如果 bestdist 大于 pincheddist，则执行以下操作
      FOREACHvertex_(subridge) {
        // 再次遍历 subridge 中的每个顶点
        FOREACHvertexA_(subridge) {
          // 遍历 subridge 中的每个顶点对 vertexA_
          if (vertexA->id > vertex->id) { /* once per vertex pair, do not compare addresses */
            // 如果 vertexA 的 id 大于 vertex 的 id，则计算它们之间的距离
            dist= qh_pointdist(vertexA->point, vertex->point, qh->hull_dim);
            // 计算 vertexA->point 到 vertex->point 的距离，存入 dist
            if (dist < bestdist) {
              // 如果 dist 小于 bestdist，则更新 bestpinched 和 bestvertex
              bestpinched= vertexA;
              bestvertex= vertex;
              bestdist= dist;
            }
          }
        }
      }
    }
    if (bestdist > pincheddist) {
      // 如果 bestdist 大于 pincheddist，则执行以下操作
      FOREACHvertexA_(subridge) {
        // 再次遍历 subridge 中的每个顶点
        maybepinched= qh_neighbor_vertices(qh, vertexA, subridge); /* subridge and apex tested above */
        // 获取 vertexA 的邻居顶点集合，存入 maybepinched
        FOREACHvertex_(maybepinched) {
          // 遍历 maybepinched 中的每个顶点
          dist= qh_pointdist(vertex->point, vertexA->point, qh->hull_dim);
          // 计算 vertex->point 到 vertexA->point 的距离，存入 dist
          if (dist < bestdist) {
            // 如果 dist 小于 bestdist，则更新 bestvertex 和 bestpinched
            bestvertex= vertex;
            bestpinched= vertexA;
            bestdist= dist;
          }
        }
        qh_settempfree(qh, &maybepinched);
        // 释放 maybepinched 集合的临时空间
      }
    }
  }
  *distp= bestdist;
  // 将 bestdist 赋值给 distp 指针指向的变量
  qh_setfree(qh, &subridge); /* qh_err_exit not called since allocated */
  // 释放 subridge 集合的内存空间，由于分配，不调用 qh_err_exit
  if (!bestvertex) {  /* should never happen if qh.hull_dim > 2 */
    // 如果 bestvertex 为 NULL，则输出错误信息
    // 输出错误信息到qh结构体的ferr文件流，指定错误代码6274，格式化错误消息字符串
    qh_fprintf(qh, qh->ferr, 6274, "qhull internal error (qh_findbest_pinchedvertex): did not find best vertex for subridge of dupridge between f%d and f%d, while processing p%d\n", merge->facet1->id, merge->facet2->id, qh->furthest_id);
    // 终止qhull程序，指定错误类型为qh_ERRqhull，传递相关的facet1和facet2信息
    qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
  }
  // 将nearestp指向bestvertex指针指向的位置
  *nearestp= bestvertex;
  // 输出跟踪信息到qhull的日志文件流ferr，指定跟踪代码2061，格式化最佳pinched顶点和最近顶点的信息
  trace2((qh, qh->ferr, 2061, "qh_findbest_pinchedvertex: best pinched p%d(v%d) and vertex p%d(v%d) are closest (%2.2g) for duplicate subridge between f%d and f%d\n",
      // 获取bestpinched顶点的标识和id，获取bestvertex顶点的标识和id，输出最小距离，以及相关的facet1和facet2的id
      qh_pointid(qh, bestpinched->point), bestpinched->id, qh_pointid(qh, bestvertex->point), bestvertex->id, bestdist, merge->facet1->id, merge->facet2->id));
  // 返回bestpinched顶点的地址
  return bestpinched;
/*-<a                             href="qh-geom2_r.htm#TOC"
  >-------------------------------</a><a name="findbest_ridgevertex">-</a>

  qh_findbest_ridgevertex(qh, ridge, pinchedp, distp )
    Determine the best vertex/pinched-vertex to merge for ridges with the same vertices

  returns:
    vertex, pinched vertex, and the distance between them

  notes:
    assumes qh.hull_dim>=3
    see qh_findbest_pinchedvertex
*/
vertexT *qh_findbest_ridgevertex(qhT *qh, ridgeT *ridge, vertexT **pinchedp, coordT *distp) {
  vertexT *bestvertex;  // 声明一个顶点指针变量

  *distp= qh_vertex_bestdist2(qh, ridge->vertices, &bestvertex, pinchedp);  // 调用函数计算最佳顶点和挤压顶点的距离，并返回最佳顶点和挤压顶点
  trace4((qh, qh->ferr, 4069, "qh_findbest_ridgevertex: best pinched p%d(v%d) and vertex p%d(v%d) are closest (%2.2g) for duplicated ridge r%d (same vertices) between f%d and f%d\n",
      qh_pointid(qh, (*pinchedp)->point), (*pinchedp)->id, qh_pointid(qh, bestvertex->point), bestvertex->id, *distp, ridge->id, ridge->top->id, ridge->bottom->id));
  return bestvertex;  // 返回最佳顶点
} /* findbest_ridgevertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="findbest_test">-</a>

  qh_findbest_test(qh, testcentrum, facet, neighbor, &bestfacet, &dist, &mindist, &maxdist )
    test neighbor of facet for qh_findbestneighbor()
    if testcentrum,
      tests centrum (assumes it is defined)
    else
      tests vertices
    initially *bestfacet==NULL and *dist==REALmax

  returns:
    if a better facet (i.e., vertices/centrum of facet closer to neighbor)
      updates bestfacet, dist, mindist, and maxdist

  notes:
    called by qh_findbestneighbor
    ignores pairs of flipped facets, unless that's all there is
*/
void qh_findbest_test(qhT *qh, boolT testcentrum, facetT *facet, facetT *neighbor,
      facetT **bestfacet, realT *distp, realT *mindistp, realT *maxdistp) {
  realT dist, mindist, maxdist;  // 声明距离和最小/最大距离变量

  if (facet->flipped && neighbor->flipped && *bestfacet && !(*bestfacet)->flipped)
    return; /* do not merge flipped into flipped facets */  // 如果面和其邻居面都是翻转的，并且最佳面也是翻转的，则直接返回，不合并

  if (testcentrum) {
    zzinc_(Zbestdist);  // 增加计数器 Zbestdist
    qh_distplane(qh, facet->center, neighbor, &dist);  // 计算面的中心到邻居面的距离
    dist *= qh->hull_dim; /* estimate furthest vertex */  // 估算最远顶点的距离
    if (dist < 0) {
      maxdist= 0;
      mindist= dist;
      dist= -dist;
    }else {
      mindist= 0;
      maxdist= dist;
    }
  }else
    dist= qh_getdistance(qh, facet, neighbor, &mindist, &maxdist);  // 计算面和邻居面之间的距离，以及最小和最大距离

  if (dist < *distp) {
    *bestfacet= neighbor;  // 更新最佳面为当前邻居面
    *mindistp= mindist;  // 更新最小距离
    *maxdistp= maxdist;  // 更新最大距离
    *distp= dist;  // 更新距离
  }
} /* findbest_test */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="findbestneighbor">-</a>

  qh_findbestneighbor(qh, facet, dist, mindist, maxdist )
    finds best neighbor (least dist) of a facet for merging

  returns:
    returns min and max distances and their max absolute value

  notes:
    error if qh_ASvoronoi
    avoids merging old into new
*/
    # 假设 ridge->nonconvex 只在一对面之间的一个边缘上设置
    # 可以使用早期退出谓词，但不值得
    
    design:
      # 如果是一个大的面
      if a large facet
        # 将测试中心点
        will test centrum
      else
        # 将测试顶点
        will test vertices
      # 如果是一个大的面
      if a large facet
        # 对非凸邻居进行最佳合并测试
        test nonconvex neighbors for best merge
      else
        # 对所有邻居进行最佳合并测试
        test all neighbors for the best merge
      # 如果测试中心点
      if testing centrum
        # 获取距离信息
        get distance information
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="flippedmerges">-</a>

  qh_flippedmerges(qh, facetlist, wasmerge )
    merge flipped facets into best neighbor
    assumes qh.facet_mergeset at top of temporary stack

  returns:
    no flipped facets on facetlist
    sets wasmerge if merge occurred
    degen/redundant merges passed through

  notes:
    othermerges not needed since qh.facet_mergeset is empty before & after
      keep it in case of change

  design:
    append flipped facets to qh.facetmergeset
    for each flipped merge
      find best neighbor
      merge facet into neighbor
      merge degenerate and redundant facets
    remove flipped merges from qh.facet_mergeset
*/
void qh_flippedmerges(qhT *qh, facetT *facetlist, boolT *wasmerge) {
  facetT *facet, *neighbor, *facet1;
  realT dist, mindist, maxdist;
  mergeT *merge, **mergep;
  setT *othermerges;
  int nummerge= 0, numdegen= 0;

  // 输出追踪信息到日志，表示进入函数
  trace4((qh, qh->ferr, 4024, "qh_flippedmerges: begin\n"));
  // 遍历输入的面列表中的每一个面
  FORALLfacet_(facetlist) {
    // 如果 facet 被翻转且不可见，则将其追加到合并集中
    if (facet->flipped && !facet->visible)
      qh_appendmergeset(qh, facet, facet, MRGflip, 0.0, 1.0);
  }
  // 从临时堆栈中弹出其他的合并集合
  othermerges= qh_settemppop(qh);
  // 检查从临时堆栈中弹出的合并集是否与 qh->facet_mergeset 相等，若不相等则报错并退出
  if(othermerges != qh->facet_mergeset) {
    qh_fprintf(qh, qh->ferr, 6392, "qhull internal error (qh_flippedmerges): facet_mergeset (%d merges) not at top of tempstack (%d merges)\n",
        qh_setsize(qh, qh->facet_mergeset), qh_setsize(qh, othermerges));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 将 qh->facet_mergeset 设为临时集合，并将其他合并集推入临时堆栈
  qh->facet_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh_settemppush(qh, othermerges);
  // 遍历其他合并集中的每一个合并项
  FOREACHmerge_(othermerges) {
    facet1= merge->facet1;
    // 如果合并类型不是 MRGflip 或者 facet1 可见，则跳过
    if (merge->mergetype != MRGflip || facet1->visible)
      continue;
    // 如果跟踪总合并数等于指定值，则设置跟踪级别
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
      qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
    // 查找 facet1 的最佳相邻面
    neighbor= qh_findbestneighbor(qh, facet1, &dist, &mindist, &maxdist);
    // 输出调试信息，表明进行了翻转合并
    trace0((qh, qh->ferr, 15, "qh_flippedmerges: merge flipped f%d into f%d dist %2.2g during p%d\n",
      facet1->id, neighbor->id, dist, qh->furthest_id));
    // 执行 facet1 和 neighbor 的合并操作
    qh_mergefacet(qh, facet1, neighbor, merge->mergetype, &mindist, &maxdist, !qh_MERGEapex);
    // 记录合并次数，并在需要时更新统计信息
    nummerge++;
    if (qh->PRINTstatistics) {
      zinc_(Zflipped);
      wadd_(Wflippedtot, dist);
      wmax_(Wflippedmax, dist);
    }
  }
  // 再次遍历其他合并集，处理剩余的合并项
  FOREACHmerge_(othermerges) {
    // 如果 merge 中的 facet1 或 facet2 可见，则释放 merge 占用的内存空间
    if (merge->facet1->visible || merge->facet2->visible)
      qh_memfree(qh, merge, (int)sizeof(mergeT)); /* invalidates merge and othermerges */
    else
      // 否则将 merge 添加到 qh->facet_mergeset 中
      qh_setappend(qh, &qh->facet_mergeset, merge);
  }
  // 释放其他合并集所占用的临时内存空间
  qh_settempfree(qh, &othermerges);
  // 执行合并冗余 facet 的操作，并更新 numdegen 计数
  numdegen += qh_merge_degenredundant(qh); /* somewhat better here than after each flipped merge -- qtest.sh 10 '500 C1,2e-13 D4' 'd Qbb' */
  // 如果发生了合并操作，则将 wasmerge 设置为 True
  if (nummerge)
    *wasmerge= True;
  // 输出调试信息，表明已经合并了多少个翻转合并和冗余合并的 facet
  trace1((qh, qh->ferr, 1010, "qh_flippedmerges: merged %d flipped and %d degenredundant facets into a good neighbor\n",
    nummerge, numdegen));
/*---------------------------------
  qh_forcedmerges(qh, wasmerge )
    merge dupridges
    calls qh_check_dupridge to report an error on wide merges
    assumes qh_settemppop is qh.facet_mergeset

  returns:
    removes all dupridges on facet_mergeset
    wasmerge set if merge
    qh.facet_mergeset may include non-forced merges (none for now)
    qh.degen_mergeset includes degen/redun merges

  notes:
    called by qh_premerge
    dupridges occur when the horizon is pinched,
        i.e. a subridge occurs in more than two horizon ridges.
    could rename vertices that pinch the horizon
    assumes qh_merge_degenredundant() has not been called
    othermerges isn't needed since facet_mergeset is empty afterwards
      keep it in case of change

  design:
    for each dupridge
      find current facets by chasing f.replace links
      check for wide merge due to dupridge
      determine best direction for facet
      merge one facet into the other
      remove dupridges from qh.facet_mergeset
*/

void qh_forcedmerges(qhT *qh, boolT *wasmerge) {
  facetT *facet1, *facet2, *merging, *merged, *newfacet;
  mergeT *merge, **mergep;
  realT dist, mindist, maxdist, dist2, mindist2, maxdist2;
  setT *othermerges;
  int nummerge=0, numflip=0, numdegen= 0;
  boolT wasdupridge= False;

  if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
  
  // Trace message indicating the start of forced merges
  trace3((qh, qh->ferr, 3054, "qh_forcedmerges: merge dupridges\n"));

  // Save current facet_mergeset in othermerges and verify consistency
  othermerges= qh_settemppop(qh); /* was facet_mergeset */
  if (qh->facet_mergeset != othermerges ) {
    // Error message and exit if temporary set inconsistency is detected
    qh_fprintf(qh, qh->ferr, 6279, "qhull internal error (qh_forcedmerges): qh_settemppop (size %d) is not qh->facet_mergeset (size %d)\n",
        qh_setsize(qh, othermerges), qh_setsize(qh, qh->facet_mergeset));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }

  // Clear and prepare facet_mergeset for storing temporary merges
  qh->facet_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh_settemppush(qh, othermerges);

  // Process each merge in othermerges, specifically handling MRGdupridge merges
  FOREACHmerge_(othermerges) {
    if (merge->mergetype != MRGdupridge)
      continue; // Skip non-dupridge merges

    // Mark that a dupridge was found
    wasdupridge= True;

    // Trace message handling if specified by TRACEmerge setting
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
      qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;

    // Retrieve facets involved in the merge and check for errors
    facet1= qh_getreplacement(qh, merge->facet1);
    facet2= qh_getreplacement(qh, merge->facet2);
    if (facet1 == facet2)
      continue; // Skip if facets are already merged

    // Check and report error if facets are no longer neighbors
    if (!qh_setin(facet2->neighbors, facet1)) {
      qh_fprintf(qh, qh->ferr, 6096, "qhull internal error (qh_forcedmerges): f%d and f%d had a dupridge but as f%d and f%d they are no longer neighbors\n",
               merge->facet1->id, merge->facet2->id, facet1->id, facet2->id);
      qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
    }

    // Calculate distances between facets in both directions
    dist= qh_getdistance(qh, facet1, facet2, &mindist, &maxdist);
    dist2= qh_getdistance(qh, facet2, facet1, &mindist2, &maxdist2);
    // 检查是否存在重复的梁，如果有，则执行检查和处理
    qh_check_dupridge(qh, facet1, dist, facet2, dist2);

    // 如果 dist 小于 dist2，则执行以下操作
    if (dist < dist2) {
      // 如果 facet2 是翻转的并且 facet1 不是翻转的，并且 dist2 小于指定阈值，优先选择翻转的 facet 进行合并
      merging= facet2;
      merged= facet1;
      dist= dist2;
      mindist= mindist2;
      maxdist= maxdist2;
    }else {
      // 如果 facet1 是翻转的并且 facet2 不是翻转的，并且 dist 小于指定阈值，优先选择翻转的 facet 进行合并
      merging= facet1;
      merged= facet2;
    }

    // 执行合并操作，更新最小距离和最大距离，可能更新顶点
    qh_mergefacet(qh, merging, merged, merge->mergetype, &mindist, &maxdist, !qh_MERGEapex);

    // 在这里执行 qh_merge_degenredundant 操作，而不是在结尾处执行，这样更好 -- qtest.sh 10 '500 C1,2e-13 D4' 'd Qbb'
    numdegen += qh_merge_degenredundant(qh);

    // 如果 facet1 是翻转的，则增加翻转合并计数，否则增加正常合并计数
    if (facet1->flipped) {
      zinc_(Zmergeflipdup);
      numflip++;
    } else {
      nummerge++;
    }

    // 如果启用了打印统计信息，则更新重复合并的统计数据
    if (qh->PRINTstatistics) {
      zinc_(Zduplicate);        // 增加重复次数计数
      wadd_(Wduplicatetot, dist);  // 累加重复距离总和
      wmax_(Wduplicatemax, dist);  // 更新重复距离的最大值
    }
  }
  
  // 遍历其他合并操作，根据合并类型决定是否释放内存
  FOREACHmerge_(othermerges) {
    if (merge->mergetype == MRGdupridge)
      qh_memfree(qh, merge, (int)sizeof(mergeT)); /* 释放 merge 的内存，使 merge 和 othermerges 失效 */
    else
      qh_setappend(qh, &qh->facet_mergeset, merge); // 将 merge 追加到 facet_mergeset 集合中
  }

  // 释放临时合并集合 othermerges 的内存
  qh_settempfree(qh, &othermerges);

  // 如果存在 dupridge，则执行以下操作
  if (wasdupridge) {
    // 遍历所有新的面，如果某个面有 dupridge，则标记其 dupridge 和 mergeridge 为 False
    FORALLnew_facets {
      if (newfacet->dupridge) {
        newfacet->dupridge= False;
        newfacet->mergeridge= False;
        newfacet->mergeridge2= False;

        // 如果面的邻居数小于 hull_dim，则认为是 degenerate，执行强制合并操作
        if (qh_setsize(qh, newfacet->neighbors) < qh->hull_dim) {
          qh_appendmergeset(qh, newfacet, newfacet, MRGdegen, 0.0, 1.0);  // 添加 degenerate 合并操作到 facet_mergeset 集合
          trace2((qh, qh->ferr, 2107, "qh_forcedmerges: dupridge f%d is degenerate with fewer than %d neighbors\n",
                      newfacet->id, qh->hull_dim)); // 输出跟踪信息
        }
      }
    }

    // 执行合并 degenerate 的面，更新 numdegen
    numdegen += qh_merge_degenredundant(qh);
  }

  // 如果有正常合并或翻转合并发生，则标记 wasmerge 为 True，并输出统计信息
  if (nummerge || numflip) {
    *wasmerge= True;
    trace1((qh, qh->ferr, 1011, "qh_forcedmerges: merged %d facets, %d flipped facets, and %d degenredundant facets across dupridges\n",
                  nummerge, numflip, numdegen));
  }
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="freemergesets">-</a>

  qh_freemergesets(qh )
    free the merge sets

  notes:
    matches qh_initmergesets
*/
void qh_freemergesets(qhT *qh) {

  if (!qh->facet_mergeset || !qh->degen_mergeset || !qh->vertex_mergeset) {
    // 如果任何一个 mergeset 为空，则输出错误信息，并退出程序
    qh_fprintf(qh, qh->ferr, 6388, "qhull internal error (qh_freemergesets): expecting mergesets.  Got a NULL mergeset, qh.facet_mergeset (0x%x), qh.degen_mergeset (0x%x), qh.vertex_mergeset (0x%x)\n",
      qh->facet_mergeset, qh->degen_mergeset, qh->vertex_mergeset);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (!SETempty_(qh->facet_mergeset) || !SETempty_(qh->degen_mergeset) || !SETempty_(qh->vertex_mergeset)) {
    // 如果任何一个 mergeset 不为空，则输出错误信息，并退出程序
    qh_fprintf(qh, qh->ferr, 6389, "qhull internal error (qh_freemergesets): expecting empty mergesets.  Got qh.facet_mergeset (%d merges), qh.degen_mergeset (%d merges), qh.vertex_mergeset (%d merges)\n",
      qh_setsize(qh, qh->facet_mergeset), qh_setsize(qh, qh->degen_mergeset), qh_setsize(qh, qh->vertex_mergeset));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 释放三个 mergeset
  qh_settempfree(qh, &qh->facet_mergeset);
  qh_settempfree(qh, &qh->vertex_mergeset);
  qh_settempfree(qh, &qh->degen_mergeset);
} /* freemergesets */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="getmergeset">-</a>

  qh_getmergeset(qh, facetlist )
    determines nonconvex facets on facetlist
    tests !tested ridges and nonconvex ridges of !tested facets

  returns:
    returns sorted qh.facet_mergeset of facet-neighbor pairs to be merged
    all ridges tested

  notes:
    facetlist is qh.facet_newlist, use qh_getmergeset_initial for all facets
    assumes no nonconvex ridges with both facets tested
    uses facet->tested/ridge->tested to prevent duplicate tests
    can not limit tests to modified ridges since the centrum changed
    uses qh.visit_id

  design:
    for each facet on facetlist
      for each ridge of facet
        if untested ridge
          test ridge for convexity
          if non-convex
            append ridge to qh.facet_mergeset
    sort qh.facet_mergeset by mergetype and angle or distance
*/
void qh_getmergeset(qhT *qh, facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int nummerges;
  boolT simplicial;

  nummerges= qh_setsize(qh, qh->facet_mergeset);
  // 输出跟踪信息
  trace4((qh, qh->ferr, 4026, "qh_getmergeset: started.\n"));
  // 增加访问标识号
  qh->visit_id++;
  // 遍历所有的 facet
  FORALLfacet_(facetlist) {
    // 如果 facet 已经被测试过，则跳过
    if (facet->tested)
      continue;
    // 设置 facet 的访问标识号
    facet->visitid= qh->visit_id;
    // 对于每个 facet 的相邻面
    FOREACHneighbor_(facet)
      // 将 neighbor 的 seen 标志设为 False
      neighbor->seen= False;
    /* facet must be non-simplicial due to merge to qh.facet_newlist */
    FOREACHridge_(facet->ridges) {
      // 遍历当前面元的所有棱
      if (ridge->tested && !ridge->nonconvex)
        // 如果棱已经被测试过且不是非凸的，则跳过
        continue;
      /* if r.tested & r.nonconvex, need to retest and append merge */
      // 如果棱已经被测试过或者是非凸的，需要重新测试并追加合并
      neighbor= otherfacet_(ridge, facet);
      if (neighbor->seen) { /* another ridge for this facet-neighbor pair was already tested in this loop */
        // 如果已经在此循环中测试过这个面元-相邻面元对的另一条棱
        ridge->tested= True;
        ridge->nonconvex= False;   // 每个面元-相邻面元对只标记一条棱为非凸
      }else if (neighbor->visitid != qh->visit_id) {
        // 如果相邻面元的访问标记与当前的访问标记不同
        neighbor->seen= True;
        ridge->nonconvex= False;
        simplicial= False;
        if (ridge->simplicialbot && ridge->simplicialtop)
          simplicial= True;
        if (qh_test_appendmerge(qh, facet, neighbor, simplicial))
          ridge->nonconvex= True;
        ridge->tested= True;
      }
    }
    facet->tested= True;
  }
  nummerges= qh_setsize(qh, qh->facet_mergeset);
  // 计算面元合并集合的大小
  if (qh->ANGLEmerge)
    // 如果启用了角度合并，则按角度合并方式排序面元合并集合
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_anglemerge);
  else
    // 否则按面元合并方式排序面元合并集合
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_facetmerge);
  nummerges += qh_setsize(qh, qh->degen_mergeset);
  // 将退化合并集合的大小加到总合并数上
  if (qh->POSTmerging) {
    // 如果在合并之后执行了后处理
    zadd_(Zmergesettot2, nummerges);
  }else {
    // 否则将总合并数加到合并总计中，并更新合并集合的最大值
    zadd_(Zmergesettot, nummerges);
    zmax_(Zmergesetmax, nummerges);
  }
  trace2((qh, qh->ferr, 2021, "qh_getmergeset: %d merges found\n", nummerges));
  // 输出跟踪信息，显示找到的合并次数
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="getmergeset_initial">-</a>

  qh_getmergeset_initial(qh, facetlist )
    determine initial qh.facet_mergeset for facets
    tests all facet/neighbor pairs on facetlist

  returns:
    sorted qh.facet_mergeset with nonconvex ridges
    sets facet->tested, ridge->tested, and ridge->nonconvex

  notes:
    uses visit_id, assumes ridge->nonconvex is False
    see qh_getmergeset

  design:
    for each facet on facetlist
      for each untested neighbor of facet
        test facet and neighbor for convexity
        if non-convex
          append merge to qh.facet_mergeset
          mark one of the ridges as nonconvex
    sort qh.facet_mergeset by mergetype and angle or distance
*/
void qh_getmergeset_initial(qhT *qh, facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int nummerges;
  boolT simplicial;

  qh->visit_id++;  /* 增加访问标识符，用于标记当前遍历的阶段 */
  FORALLfacet_(facetlist) {
    facet->visitid= qh->visit_id;  /* 标记当前 facet 的访问标识符为当前的 visit_id */
    FOREACHneighbor_(facet) {  /* 遍历当前 facet 的所有邻居 */
      if (neighbor->visitid != qh->visit_id) {
        simplicial= False; /* 初始化 simplicial 标记为 False，用于检查是否为简单多边形 */
        if (facet->simplicial && neighbor->simplicial)
          simplicial= True;  /* 如果当前 facet 和邻居都是简单多边形，则设置 simplicial 为 True */
        if (qh_test_appendmerge(qh, facet, neighbor, simplicial)) {  /* 检查并添加 merge 到 qh.facet_mergeset */
          FOREACHridge_(neighbor->ridges) {
            if (facet == otherfacet_(ridge, neighbor)) {
              ridge->nonconvex= True;  /* 将找到的非凸边缘标记为非凸 */
              break;    /* 只标记一个 ridge 为非凸 */
            }
          }
        }
      }
    }
    facet->tested= True;  /* 标记当前 facet 已经被测试过 */
    FOREACHridge_(facet->ridges)
      ridge->tested= True;  /* 标记当前 facet 的所有 ridge 已经被测试过 */
  }
  nummerges= qh_setsize(qh, qh->facet_mergeset);  /* 获取当前 qh.facet_mergeset 的大小 */
  if (qh->ANGLEmerge)
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_anglemerge);
  else
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_facetmerge);
  nummerges += qh_setsize(qh, qh->degen_mergeset);  /* 将 degen_mergeset 的大小加到 nummerges 中 */
  if (qh->POSTmerging) {
    zadd_(Zmergeinittot2, nummerges);  /* 如果在后期合并阶段，将 nummerges 添加到 Zmergeinittot2 */
  }else {
    zadd_(Zmergeinittot, nummerges);  /* 否则将 nummerges 添加到 Zmergeinittot */
    zmax_(Zmergeinitmax, nummerges);  /* 更新 Zmergeinitmax 的最大值 */
  }
  trace2((qh, qh->ferr, 2022, "qh_getmergeset_initial: %d merges found\n", nummerges));  /* 输出找到的 merges 数量 */
} /* getmergeset_initial */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="getpinchedmerges">-</a>

  qh_getpinchedmerges(qh, apex, maxdist, iscoplanar )
    get pinched merges for dupridges in qh.facet_mergeset
    qh.NEWtentative==True
      qh.newfacet_list with apex
      qh.horizon_list is attached to qh.visible_list instead of qh.newfacet_list
      maxdist for vertex-facet of a dupridge
    qh.facet_mergeset is empty
    qh.vertex_mergeset is a temporary set

  returns:
    False if nearest vertex would increase facet width by more than maxdist or qh_WIDEpinched
*/
    # 如果顶点处于共面状态，并且顶点是顶点（即使顶点是共面点）
    True and iscoplanar, if the pinched vertex is the apex (i.e., make the apex a coplanar point)
    # 如果顶点不是共面状态，并且应合并双峰的顶点
    True and !iscoplanar, if should merge a pinched vertex of a dupridge
      # qh.vertex_mergeset 包含一个或多个带有顶点和附近邻近顶点的 MRGsubridge
      qh.vertex_mergeset contains one or more MRGsubridge with a pinched vertex and a nearby, neighboring vertex
    # qh.facet_mergeset 为空
    qh.facet_mergeset is empty
    
    notes:
      # 被 qh_buildcone_mergepinched 调用
      called by qh_buildcone_mergepinched
      # 维度大于等于 3
      hull_dim >= 3
      # 顶点处于双峰和水平线
      a pinched vertex is in a dupridge and the horizon
      # 选择距离邻近顶点最近的顶点
      selects the pinched vertex that is closest to its neighbor
    
    design:
      # 对每个双峰
      for each dupridge
          # 确定最佳的顶点以合并到邻近顶点
          determine the best pinched vertex to be merged into a neighboring vertex
          # 如果合并顶点会产生宽合并（qh_WIDEpinched）
             # 忽略带有警告的尖顶点，并使用 qh_merge_degenredundant 替代
             ignore pinched vertex with a warning, and use qh_merge_degenredundant instead
          else
             # 将尖顶点附加到 vertex_mergeset 以进行合并
             append the pinched vertex to vertex_mergeset for merging
/*
  查找需要合并的钝化顶点和最佳合并。若发生钝化，则设置标志位iscoplanar为False
  在找到最佳合并前，初始化相关变量和距离
*/
boolT qh_getpinchedmerges(qhT *qh, vertexT *apex, coordT maxdupdist, boolT *iscoplanar /* qh.newfacet_list, qh.vertex_mergeset */) {
  mergeT *merge, **mergep, *bestmerge= NULL;  // 声明合并对象的指针和最佳合并
  vertexT *nearest, *pinched, *bestvertex= NULL, *bestpinched= NULL;  // 声明顶点的指针和最佳钝化顶点
  boolT result;  // 声明布尔类型结果变量
  coordT dist, prevdist, bestdist= REALmax/(qh_RATIOcoplanarapex+1.0); /* allow *3.0 */  // 声明距离变量和最佳距离

  // 输出调试信息，描述正在尝试合并钝化顶点的过程
  trace2((qh, qh->ferr, 2062, "qh_getpinchedmerges: try to merge pinched vertices for dupridges in new facets with apex p%d(v%d) max dupdist %2.2g\n",
      qh_pointid(qh, apex->point), apex->id, maxdupdist));
  *iscoplanar= False;  // 将iscoplanar标志位设为False，表示未发生钝化

  // 计算prevdist，用于限制合并操作的距离
  prevdist= fmax_(qh->ONEmerge + qh->DISTround, qh->MINoutside + qh->DISTround);
  maximize_(prevdist, qh->max_outside);
  maximize_(prevdist, -qh->min_vertex);

  // 标记重复的边缘，并创建钝化边缘集合
  qh_mark_dupridges(qh, qh->newfacet_list, !qh_ALL); /* qh.facet_mergeset, creates ridges */
  /* qh_mark_dupridges is called a second time in qh_premerge */

  // 遍历钝化边缘集合，寻找MRGdupridge类型的合并操作
  FOREACHmerge_(qh->facet_mergeset) {  /* read-only */
    if (merge->mergetype != MRGdupridge) {
      // 若不是MRGdupridge类型的合并，则输出错误信息并退出程序
      qh_fprintf(qh, qh->ferr, 6393, "qhull internal error (qh_getpinchedmerges): expecting MRGdupridge from qh_mark_dupridges.  Got merge f%d f%d type %d\n",
        getid_(merge->facet1), getid_(merge->facet2), merge->mergetype);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    // 计算顶点之间的距离
    pinched= qh_findbest_pinchedvertex(qh, merge, apex, &nearest, &dist /* qh.newfacet_list */);
    // 若钝化顶点是apex并且距离小于预设的最佳距离，优先选择平面顶点
    if (pinched == apex && dist < qh_RATIOcoplanarapex*bestdist) {
      bestdist= dist/qh_RATIOcoplanarapex;
      bestmerge= merge;
      bestpinched= pinched;
      bestvertex= nearest;
    }else if (dist < bestdist) {
      // 若距离小于当前最佳距离，则更新最佳合并和相关顶点
      bestdist= dist;
      bestmerge= merge;
      bestpinched= pinched;
      bestvertex= nearest;
    }
  }

  result= False;  // 初始化结果为False，表示未找到合适的合并

  // 若存在最佳合并且距离小于最大允许的重复距离
  if (bestmerge && bestdist < maxdupdist) {
    ratio= bestdist / prevdist;  // 计算距离比率

    // 若距离比率大于设置的宽距化阈值
    if (ratio > qh_WIDEpinched) {
      // 若涉及到平面的合并，输出详细信息描述宽合并情况
      if (bestmerge->facet1->mergehorizon || bestmerge->facet2->mergehorizon) { /* e.g., rbox 175 C3,2e-13 t1539182828 | qhull d */
        trace1((qh, qh->ferr, 1051, "qh_getpinchedmerges: dupridge (MRGdupridge) of coplanar horizon would produce a wide merge (%.0fx) due to pinched vertices v%d and v%d (dist %2.2g) for f%d and f%d.  qh_mergecycle_all will merge one or both facets\n",
          ratio, bestpinched->id, bestvertex->id, bestdist, bestmerge->facet1->id, bestmerge->facet2->id));
      }else {
        // 否则输出警告信息，描述宽合并可能导致的精度问题
        qh_fprintf(qh, qh->ferr, 7081, "qhull precision warning (qh_getpinchedmerges): pinched vertices v%d and v%d (dist %2.2g, %.0fx) would produce a wide merge for f%d and f%d.  Will merge dupridge instead\n",
          bestpinched->id, bestvertex->id, bestdist, ratio, bestmerge->facet1->id, bestmerge->facet2->id);
      }
    }else {
      // 如果条件不成立，进入这个分支
      if (bestpinched == apex) {
        // 如果 bestpinched 等于 apex，则记录日志并设置 qh->coplanar_apex 为 apex 的点
        trace2((qh, qh->ferr, 2063, "qh_getpinchedmerges: will make the apex a coplanar point.  apex p%d(v%d) is the nearest vertex to v%d on dupridge.  Dist %2.2g\n",
          qh_pointid(qh, apex->point), apex->id, bestvertex->id, bestdist*qh_RATIOcoplanarapex));
        qh->coplanar_apex= apex->point;
        *iscoplanar= True;
        result= True;
      }else if (qh_setin(bestmerge->facet1->vertices, bestpinched) != qh_setin(bestmerge->facet2->vertices, bestpinched)) { /* pinched in one facet but not the other facet */
        // 如果 bestpinched 在一个面片中但不在另一个面片中，则记录日志并执行顶点合并操作
        trace2((qh, qh->ferr, 2064, "qh_getpinchedmerges: will merge new facets to resolve dupridge between f%d and f%d with pinched v%d and v%d\n",
          bestmerge->facet1->id, bestmerge->facet2->id, bestpinched->id, bestvertex->id));
        qh_appendvertexmerge(qh, bestpinched, bestvertex, MRGsubridge, bestdist, NULL, NULL);
        result= True;
      }else {
        // 否则，记录日志并执行顶点合并操作
        trace2((qh, qh->ferr, 2065, "qh_getpinchedmerges: will merge pinched v%d into v%d to resolve dupridge between f%d and f%d\n",
          bestpinched->id, bestvertex->id, bestmerge->facet1->id, bestmerge->facet2->id));
        qh_appendvertexmerge(qh, bestpinched, bestvertex, MRGsubridge, bestdist, NULL, NULL);
        result= True;
      }
    }
  }
  // 删除 MRGdupridge，qh_mark_dupridges 在 qh_premerge 中第二次调用
  while ((merge= (mergeT *)qh_setdellast(qh->facet_mergeset)))
    qh_memfree(qh, merge, (int)sizeof(mergeT));
  // 返回结果
  return result;
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="hasmerge">-</a>

  qh_hasmerge( mergeset, mergetype, facetA, facetB )
    True if mergeset has mergetype for facetA and facetB
*/
boolT qh_hasmerge(setT *mergeset, mergeType type, facetT *facetA, facetT *facetB) {
  mergeT *merge, **mergep;  // 定义指向 mergeT 结构体指针的指针

  FOREACHmerge_(mergeset) {  // 遍历 mergeset 中的每一个 mergeT 结构体
    if (merge->mergetype == type) {  // 如果 merge 的 mergetype 和给定的 type 相同
      if (merge->facet1 == facetA && merge->facet2 == facetB)  // 如果 merge 的 facet1 和 facet2 分别等于 facetA 和 facetB
        return True;  // 返回 True
      if (merge->facet1 == facetB && merge->facet2 == facetA)  // 如果 merge 的 facet1 和 facet2 分别等于 facetB 和 facetA
        return True;  // 返回 True
    }
  }
  return False;  // 没有找到符合条件的 merge，返回 False
}/* hasmerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="hashridge">-</a>

  qh_hashridge(qh, hashtable, hashsize, ridge, oldvertex )
    add ridge to hashtable without oldvertex

  notes:
    assumes hashtable is large enough

  design:
    determine hash value for ridge without oldvertex
    find next empty slot for ridge
*/
void qh_hashridge(qhT *qh, setT *hashtable, int hashsize, ridgeT *ridge, vertexT *oldvertex) {
  int hash;  // 哈希值
  ridgeT *ridgeA;  // 指向 ridgeT 结构体的指针

  hash= qh_gethash(qh, hashsize, ridge->vertices, qh->hull_dim-1, 0, oldvertex);  // 计算 ridge 在不包含 oldvertex 的情况下的哈希值
  while (True) {  // 循环直到找到合适的插入位置
    if (!(ridgeA= SETelemt_(hashtable, hash, ridgeT))) {  // 如果 hashtable 中 hash 位置为空
      SETelem_(hashtable, hash)= ridge;  // 将 ridge 插入 hashtable 中的 hash 位置
      break;  // 跳出循环
    } else if (ridgeA == ridge)  // 如果 hashtable 中 hash 位置已经存储了 ridge
      break;  // 跳出循环
    if (++hash == hashsize)  // 如果 hash 超出了 hashtable 的大小
      hash= 0;  // 将 hash 重置为 0
  }
} /* hashridge */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="hashridge_find">-</a>

  qh_hashridge_find(qh, hashtable, hashsize, ridge, vertex, oldvertex, hashslot )
    returns matching ridge without oldvertex in hashtable
      for ridge without vertex
    if oldvertex is NULL
      matches with any one skip

  returns:
    matching ridge or NULL
    if no match,
      if ridge already in   table
        hashslot= -1
      else
        hashslot= next NULL index

  notes:
    assumes hashtable is large enough
    can't match ridge to itself

  design:
    get hash value for ridge without vertex
    for each hashslot
      return match if ridge matches ridgeA without oldvertex
*/
ridgeT *qh_hashridge_find(qhT *qh, setT *hashtable, int hashsize, ridgeT *ridge,
              vertexT *vertex, vertexT *oldvertex, int *hashslot) {
  int hash;  // 哈希值
  ridgeT *ridgeA;  // 指向 ridgeT 结构体的指针

  *hashslot= 0;  // 初始化 hashslot
  zinc_(Zhashridge);  // 增加 Zhashridge 计数器
  hash= qh_gethash(qh, hashsize, ridge->vertices, qh->hull_dim-1, 0, vertex);  // 计算 ridge 在不包含 vertex 的情况下的哈希值
  while ((ridgeA= SETelemt_(hashtable, hash, ridgeT))) {  // 循环直到找到匹配的 ridge 或者空槽
    if (ridgeA == ridge)  // 如果找到了与 ridge 相同的 ridgeA
      *hashslot= -1;  // 设置 hashslot 为 -1
    else {
      zinc_(Zhashridgetest);  // 增加 Zhashridgetest 计数器
      if (qh_setequal_except(ridge->vertices, vertex, ridgeA->vertices, oldvertex))  // 检查 ridge 和 ridgeA 是否相等，除了 oldvertex 外
        return ridgeA;  // 返回匹配的 ridgeA
    }
    if (++hash == hashsize)  // 如果 hash 超出了 hashtable 的大小
      hash= 0;  // 将 hash 重置为 0
  }
  if (!*hashslot)  // 如果 hashslot 仍然为 0
    *hashslot= hash;  // 设置 hashslot 为当前的 hash 值
  return NULL;  // 没有找到匹配的 ridge，返回 NULL
} /* hashridge_find */
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="initmergesets">-</a>

  qh_initmergesets(qh )
    initialize the merge sets
    if 'all', include qh.degen_mergeset

  notes:
    matches qh_freemergesets
*/
void qh_initmergesets(qhT *qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */) {

  if (qh->facet_mergeset || qh->degen_mergeset || qh->vertex_mergeset) {
    // 如果任何一个 mergeset 已经存在，则输出错误信息并退出程序
    qh_fprintf(qh, qh->ferr, 6386, "qhull internal error (qh_initmergesets): expecting NULL mergesets.  Got qh.facet_mergeset (0x%x), qh.degen_mergeset (0x%x), qh.vertex_mergeset (0x%x)\n",
      qh->facet_mergeset, qh->degen_mergeset, qh->vertex_mergeset);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 创建临时 mergeset：degen_mergeset、vertex_mergeset、facet_mergeset
  qh->degen_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh->vertex_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh->facet_mergeset= qh_settemp(qh, qh->TEMPsize); /* last temporary set for qh_forcedmerges */
} /* initmergesets */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="makeridges">-</a>

  qh_makeridges(qh, facet )
    creates explicit ridges between simplicial facets

  returns:
    facet with ridges and without qh_MERGEridge
    ->simplicial is False
    if facet was tested, new ridges are tested

  notes:
    allows qh_MERGEridge flag
    uses existing ridges
    duplicate neighbors ok if ridges already exist (qh_mergecycle_ridges)

  see:
    qh_mergecycle_ridges()
    qh_rename_adjacentvertex for qh_merge_pinchedvertices

  design:
    look for qh_MERGEridge neighbors
    mark neighbors that already have ridges
    for each unprocessed neighbor of facet
      create a ridge for neighbor and facet
    if any qh_MERGEridge neighbors
      delete qh_MERGEridge flags (previously processed by qh_mark_dupridges)
*/
void qh_makeridges(qhT *qh, facetT *facet) {
  facetT *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int neighbor_i, neighbor_n;
  boolT toporient, mergeridge= False;

  // 如果 facet 不是简单的，直接返回
  if (!facet->simplicial)
    return;
  trace4((qh, qh->ferr, 4027, "qh_makeridges: make ridges for f%d\n", facet->id));
  // 设定 facet 的 simplicial 标志为 False，表示它不再是简单的
  facet->simplicial= False;
  // 遍历 facet 的每个邻居
  FOREACHneighbor_(facet) {
    // 如果邻居是 qh_MERGEridge，则标记 mergeridge 为 True
    if (neighbor == qh_MERGEridge)
      mergeridge= True;
    else
      // 否则，将邻居的 seen 标志设为 False
      neighbor->seen= False;
  }
  // 对于 facet 的每条 ridge
  FOREACHridge_(facet->ridges)
    // 标记 ridge 的另一侧邻居的 seen 标志为 True
    otherfacet_(ridge, facet)->seen= True;
  // 遍历 facet 的每个邻居
  FOREACHneighbor_i_(qh, facet) {
    // 如果邻居是 qh_MERGEridge，则跳过这个邻居（已由 qh_mark_dupridges 修复）
    if (neighbor == qh_MERGEridge)
      continue;
    else if (!neighbor->seen) {  /* no current ridges */
      // 如果相邻面还没有被处理过，则创建一个新的边界（ridge）
      ridge= qh_newridge(qh);
      // 将当前面（facet）的顶点集合中去掉第 neighbor_i 个顶点后的剩余顶点集合赋给新创建的边界的顶点集合
      ridge->vertices= qh_setnew_delnthsorted(qh, facet->vertices, qh->hull_dim,
                                                          neighbor_i, 0);
      // 根据 neighbor_i 的奇偶性来确定新创建的边界的方向
      toporient= (boolT)(facet->toporient ^ (neighbor_i & 0x1));
      if (toporient) {
        // 如果方向为顶部定向，则设置顶部面为当前面，底部面为相邻面，并设置简单化标志
        ridge->top= facet;
        ridge->bottom= neighbor;
        ridge->simplicialtop= True;
        ridge->simplicialbot= neighbor->simplicial;
      }else {
        // 如果方向为底部定向，则设置顶部面为相邻面，底部面为当前面，并设置简单化标志
        ridge->top= neighbor;
        ridge->bottom= facet;
        ridge->simplicialtop= neighbor->simplicial;
        ridge->simplicialbot= True;
      }
      // 如果当前面已经被测试过且不是合并边界，则设置新创建的边界也为已测试状态
      if (facet->tested && !mergeridge)
        ridge->tested= True;
      flip= (facet->toporient ^ neighbor->toporient)^(skip1 & 0x1) ^ (skip2 & 0x1);
      // 计算 flip，用于确定 ridge 的 top 和 bottom
      if (facet->toporient ^ (skip1 & 0x1) ^ flip) {
        // 如果条件成立，设置 ridge 的 top 为 neighbor，bottom 为 facet
        ridge->top= neighbor;
        ridge->bottom= facet;
        // 设置 simplicialtop 和 simplicialbot 标志
        ridge->simplicialtop= True;
        ridge->simplicialbot= neighbor->simplicial;
      }else {
        // 否则，设置 ridge 的 top 为 facet，bottom 为 neighbor
        ridge->top= facet;
        ridge->bottom= neighbor;
        // 设置 simplicialtop 和 simplicialbot 标志
        ridge->simplicialtop= neighbor->simplicial;
        ridge->simplicialbot= True;
      }
      // 将 ridge 追加到 facet 的 ridges 集合中
      qh_setappend(qh, &(facet->ridges), ridge);
      // 在跟踪输出中记录 ridge 追加操作的信息
      trace5((qh, qh->ferr, 5005, "makeridges: appended r%d to ridges for f%d.  Next is ridges for neighbor f%d\n",
            ridge->id, facet->id, neighbor->id));
      // 将 ridge 追加到 neighbor 的 ridges 集合中
      qh_setappend(qh, &(neighbor->ridges), ridge);
      // 如果 ridge 的 id 与 traceridge_id 相等，则设置 qh->traceridge 为当前的 ridge
      if (qh->ridge_id == qh->traceridge_id)
        qh->traceridge= ridge;
    # 遍历所有在facetlist上的facets
    for all facets on facetlist
      # 如果facet包含dupridge
      if facet contains a dupridge
        # 遍历facet的每个邻居
        for each neighbor of facet
          # 如果邻居标记为qh_MERGEridge（合并的一侧）
          if neighbor marked qh_MERGEridge (one side of the merge)
            # 设置facet的mergeridge标记
            set facet->mergeridge
          else
            # 如果邻居包含dupridge并且反向链接是qh_MERGEridge
            if neighbor contains a dupridge
            and the back link is qh_MERGEridge
              # 将dupridge追加到qh.facet_mergeset中
              append dupridge to qh.facet_mergeset
   # 如果!allmerges用于稍后重复执行qh_mark_dupridges，则退出
   exit if !allmerges for repeating qh_mark_dupridges later
   # 对每个dupridge执行以下操作
   for each dupridge
     # 准备合并前的ridge集合
     make ridge sets in preparation for merging
     # 从邻居集合中移除qh_MERGEridge
     remove qh_MERGEridge from neighbor set
   # 对每个dupridge执行以下操作
   for each dupridge
     # 从包含qh_MERGEridge的邻居集合中恢复缺失的邻居
     restore the missing neighbor from the neighbor set that was qh_MERGEridge
     # 为这个邻居添加缺失的ridge
     add the missing ridge for this neighbor
void qh_mark_dupridges(qhT *qh, facetT *facetlist, boolT allmerges) {
  facetT *facet, *neighbor, **neighborp;
  int nummerge=0;
  mergeT *merge, **mergep;

  // 打印调试信息，标识正在查找 facetlist 中的重复边缘，以及是否应用所有合并
  trace4((qh, qh->ferr, 4028, "qh_mark_dupridges: identify dupridges in facetlist f%d, allmerges? %d\n",
    facetlist->id, allmerges));
  
  // 遍历 facetlist 中的每个 facet，首次调用时不需要执行
  FORALLfacet_(facetlist) {  
    // 将当前 facet 的 mergeridge2 和 mergeridge 标志设为 False
    facet->mergeridge2= False;
    facet->mergeridge= False;
  }
  
  // 再次遍历 facetlist 中的每个 facet
  FORALLfacet_(facetlist) {
    // 如果当前 facet 是 dupridge
    if (facet->dupridge) {
      // 遍历当前 facet 的每个邻居
      FOREACHneighbor_(facet) {
        // 如果邻居是 qh_MERGEridge
        if (neighbor == qh_MERGEridge) {
          // 将当前 facet 的 mergeridge 标志设为 True，并继续下一个邻居
          facet->mergeridge= True;
          continue;
        }
        // 如果邻居也是 dupridge
        if (neighbor->dupridge) {
          // 如果邻居的邻居集合中不包含当前 facet（即它是 qh_MERGEridge，且邻居是不同的）
          if (!qh_setin(neighbor->neighbors, facet)) { 
            // 向合并集合中添加一个新的合并项，表示发现了重复边缘
            qh_appendmergeset(qh, facet, neighbor, MRGdupridge, 0.0, 1.0);
            // 将当前 facet 的 mergeridge2 和 mergeridge 标志设为 True
            facet->mergeridge2= True;
            facet->mergeridge= True;
            nummerge++;
          } else if (qh_setequal(facet->vertices, neighbor->vertices)) {
            // 如果当前 facet 和邻居具有相同的顶点集合（除了视界和 qh_MERGEridge，参见 QH7085）
            // 记录调试信息，指出由于子边缘的重复顶点导致 dupridge
            trace3((qh, qh->ferr, 3043, "qh_mark_dupridges): dupridge due to duplicate vertices for subridges f%d and f%d\n",
                 facet->id, neighbor->id));
            // 向合并集合中添加一个新的合并项，表示发现了重复边缘
            qh_appendmergeset(qh, facet, neighbor, MRGdupridge, 0.0, 1.0);
            // 将当前 facet 的 mergeridge2 和 mergeridge 标志设为 True
            facet->mergeridge2= True;
            facet->mergeridge= True;
            nummerge++;
            // 中断当前邻居的遍历，因为所有邻居都会有相同的情况
            break; 
          }
        }
      }
    }
  }
  
  // 如果未发现任何重复边缘，则直接返回
  if (!nummerge)
    return;
  
  // 如果不是要应用所有合并
  if (!allmerges) {
    // 记录调试信息，指出发现了多少个重复边缘，用于 qh_getpinchedmerges
    trace1((qh, qh->ferr, 1012, "qh_mark_dupridges: found %d duplicated ridges (MRGdupridge) for qh_getpinchedmerges\n", nummerge));
    return;
  }
  
  // 记录调试信息，指出发现了多少个重复边缘，用于 qh_premerge，并准备合并 facet 的准备工作
  trace1((qh, qh->ferr, 1048, "qh_mark_dupridges: found %d duplicated ridges (MRGdupridge) for qh_premerge.  Prepare facets for merging\n", nummerge));
  
  // 准备进行合并的准备工作，创建边缘
  FORALLfacet_(facetlist) {
    // 如果当前 facet 是 mergeridge，并且不是 mergeridge2
    if (facet->mergeridge && !facet->mergeridge2)
      // 创建当前 facet 的边缘
      qh_makeridges(qh, facet);
  }
  
  // 记录调试信息，指出由于 qh_MERGEridge 的存在而恢复缺失的邻居和边缘
  trace3((qh, qh->ferr, 3075, "qh_mark_dupridges: restore missing neighbors and ridges due to qh_MERGEridge\n"));
  
  // 遍历所有合并集合中的合并项，恢复缺失的邻居
  FOREACHmerge_(qh->facet_mergeset) {
    if (merge->mergetype == MRGdupridge) { /* 检查合并类型是否为重复棱 */
      if (merge->facet2->mergeridge2 && qh_setin(merge->facet2->neighbors, merge->facet1)) {
        /* 如果merge->facet2有多个相同的子棱，并且merge->facet1是merge->facet2的邻居
           由于重复或多个子棱，例如，../eg/qtest.sh t712682 '200 s W1e-13  C1,1e-13 D5' 'd'
           merge->facet1:    - 邻近的面片：f27779 f59186 f59186 f59186 MERGEridge f59186
           merge->facet2:    - 邻近的面片：f27779 f59100 f59100 f59100 f59100 f59100
           或者，../eg/qtest.sh 100 '500 s W1e-13 C1,1e-13 D4' 'd'
           合并后两个面片都将变得退化，需要考虑特殊情况处理 */
        qh_fprintf(qh, qh->ferr, 6361, "qhull topological error (qh_mark_dupridges): multiple dupridges for f%d and f%d, including reverse\n",
          merge->facet1->id, merge->facet2->id);
        // 输出拓扑错误信息，并指出重复子棱的面片编号
        qh_errexit2(qh, qh_ERRtopology, merge->facet1, merge->facet2);
        // 退出程序，报告错误的拓扑情况，指定涉及的两个面片
      } else
        // 将merge->facet1添加到merge->facet2的邻居集合中
        qh_setappend(qh, &merge->facet2->neighbors, merge->facet1);
      // 对merge->facet1进行制造棱操作，处理缺失的棱
      qh_makeridges(qh, merge->facet1);   /* and the missing ridges */
    }
  }
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="maybe_duplicateridge">-</a>

  qh_maybe_duplicateridge(qh, ridge )
    add MRGvertices if neighboring facet has another ridge with the same vertices

  returns:
    adds rename requests to qh.vertex_mergeset

  notes:
    called by qh_renamevertex
    nop if 2-D
    expensive test
    Duplicate ridges may lead to new facets with same vertex set (QH7084), will try merging vertices
    same as qh_maybe_duplicateridges

  design:
    for the two neighbors
      if non-simplicial
        for each ridge with the same first and last vertices (max id and min id)
          if the remaining vertices are the same
            get the closest pair of vertices
            add to vertex_mergeset for merging
*/
void qh_maybe_duplicateridge(qhT *qh, ridgeT *ridgeA) {
  ridgeT *ridge, **ridgep;
  vertexT *vertex, *pinched;
  facetT *neighbor;
  coordT dist;
  int i, k, last= qh->hull_dim-2;

  // 如果维度小于3，则无需处理，直接返回
  if (qh->hull_dim < 3 )
    return;

  // 遍历当前 ridgeA 的两个相邻面
  for (neighbor= ridgeA->top, i=0; i<2; neighbor= ridgeA->bottom, i++) {
    // 如果相邻面不是简单面且存在将被合并的新旧顶点，则跳过
    if (!neighbor->simplicial && neighbor->nummerge > 0) {
      // 遍历相邻面的所有 ridges
      FOREACHridge_(neighbor->ridges) {
        // 如果当前 ridge 不是 ridgeA 并且有相同的首尾顶点
        if (ridge != ridgeA && SETfirst_(ridge->vertices) == SETfirst_(ridgeA->vertices)) {
          // 如果剩余的顶点也是相同的
          if (SETelem_(ridge->vertices, last) == SETelem_(ridgeA->vertices, last)) {
            // 检查中间的顶点是否都相同
            for (k=1; k<last; k++) {
              if (SETelem_(ridge->vertices, k) != SETelem_(ridgeA->vertices, k))
                break;
            }
            // 如果中间顶点都相同，则执行以下操作
            if (k == last) {
              // 查找最佳的 ridge 顶点，并计算到 pinched 的距离
              vertex= qh_findbest_ridgevertex(qh, ridge, &pinched, &dist);
              // 记录日志，将 pinched 合并到 vertex 中，因为有重复的 ridges
              trace2((qh, qh->ferr, 2069, "qh_maybe_duplicateridge: will merge v%d into v%d (dist %2.2g) due to duplicate ridges r%d/r%d with the same vertices.  mergevertex set\n",
                pinched->id, vertex->id, dist, ridgeA->id, ridge->id, ridgeA->top->id, ridgeA->bottom->id, ridge->top->id, ridge->bottom->id));
              // 将合并请求添加到 vertex_mergeset 中
              qh_appendvertexmerge(qh, pinched, vertex, MRGvertices, dist, ridgeA, ridge);
              // 禁用 qh_checkfacet 中的重复顶点检查
              ridge->mergevertex= True;
              ridgeA->mergevertex= True;
            }
          }
        }
      }
    }
  }
} /* maybe_duplicateridge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="maybe_duplicateridges">-</a>

  qh_maybe_duplicateridges(qh, facet )
    if Q15, add MRGvertices if facet has ridges with the same vertices

  returns:
    adds rename requests to qh.vertex_mergeset

  notes:
    called at end of qh_mergefacet and qh_mergecycle_all
    only enabled if qh.CHECKduplicates ('Q15') and 3-D or more
    expensive test, not worth it
    same as qh_maybe_duplicateridge

  design:

*/
    # 遍历面的所有边对
    for all ridge pairs in facet
        # 检查是否首尾顶点相同（最大ID和最小ID相同）
        if the same first and last vertices (max id and min id)
          # 检查除首尾顶点外的其余顶点是否相同
          if the remaining vertices are the same
            # 获取最近的一对顶点
            get the closest pair of vertices
            # 将这对顶点添加到顶点合并集合以便后续合并
            add to vertex_mergeset for merging
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="maydropneighbor">-</a>

  qh_maydropneighbor(qh, facet )
    drop neighbor relationship if ridge was deleted between a non-simplicial facet and its neighbors

  returns:
    for deleted ridges
      ridges made for simplicial neighbors
      neighbor sets updated
      appends degenerate facets to qh.facet_mergeset

  notes:
    called by qh_renamevertex
    assumes neighbors do not include qh_MERGEridge (qh_makeridges)
    won't cause redundant facets since vertex inclusion is the same
    may drop vertex and neighbor if no ridge
    uses qh.visit_id

  design:
*/

void qh_maydropneighbor(qhT *qh, facetT *facet) {
  facetT *neighbor;
  ridgeT *ridge;
  int neighbor_i, neighbor_n;

  /* Check if hull dimension is less than 3 or duplicates check is disabled */
  if (qh->hull_dim < 3 || !qh->CHECKduplicates)
    return;

  /* Iterate over each ridge of the facet */
  FOREACHneighbor_i_(qh, facet->neighbors) {
    neighbor = SETelemt_(facet->neighbors, neighbor_i, facetT);
    
    /* Skip if the neighbor is marked as degenerate, redundant, has duplicate ridges, or flipped */
    if (neighbor->degenerate || neighbor->redundant || neighbor->dupridge || neighbor->flipped)
      continue;

    /* Iterate through ridges of the facet and compare with neighbors */
    FOREACHridge_(neighbor->ridges) {
      if (!ridge || ridge->simplicial)
        continue;

      /* Handle the case where ridges need to be dropped */
      /* Update neighbor sets and append degenerate facets to qh.facet_mergeset */
      /* Uses qh.visit_id to manage the operation */
    }
  }
}
    # 访问所有具有边缘的邻居
    for each unvisited neighbor of facet:
        # 对于每个未访问过的 facet 的邻居
        delete neighbor and facet from the non-simplicial neighbor sets
        # 从非简单邻居集合中删除邻居和 facet
        if neighbor becomes degenerate:
            # 如果邻居变为退化状态
            append neighbor to qh.degen_mergeset
            # 将邻居添加到 qh.degen_mergeset 中
    if facet is degenerate:
        # 如果 facet 是退化状态
        append facet to qh.degen_mergeset
        # 将 facet 添加到 qh.degen_mergeset 中
/*
  qh_maydropneighbor(qhT *qh, facetT *facet)
    根据给定的几何处理对象和面元，检查面元的邻居情况，可能删除不必要的邻居

  参数:
    qh: Qhull 库的全局状态对象
    facet: 要检查的面元对象

  返回值:
    无

  注释:
    - 增加访问标记以区分不同访问轮次
    - 在调试模式下输出关于当前面元操作的跟踪信息
    - 对于简单面元，报告错误并终止程序
    - 遍历面元的所有棱，更新棱顶点的访问标记
    - 遍历面元的所有邻居面元，检查它们的可见性和访问标记
    - 如果邻居不可见或者访问标记不匹配当前轮次，进行必要的错误报告和处理
    - 如果面元的邻居数量小于几何维度，可能会导致面元退化，触发相关处理
*/
void qh_maydropneighbor(qhT *qh, facetT *facet) {
  ridgeT *ridge, **ridgep;
  facetT *neighbor, **neighborp;

  qh->visit_id++;
  trace4((qh, qh->ferr, 4029, "qh_maydropneighbor: test f%d for no ridges to a neighbor\n",
          facet->id));
  if (facet->simplicial) {
    qh_fprintf(qh, qh->ferr, 6278, "qhull internal error (qh_maydropneighbor): not valid for simplicial f%d while adding furthest p%d\n",
      facet->id, qh->furthest_id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  FOREACHridge_(facet->ridges) {
    ridge->top->visitid= qh->visit_id;
    ridge->bottom->visitid= qh->visit_id;
  }
  FOREACHneighbor_(facet) {
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6358, "qhull internal error (qh_maydropneighbor): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
            facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    if (neighbor->visitid != qh->visit_id) {
      trace2((qh, qh->ferr, 2104, "qh_maydropneighbor: facets f%d and f%d are no longer neighbors while adding furthest p%d\n",
            facet->id, neighbor->id, qh->furthest_id));
      if (neighbor->simplicial) {
        qh_fprintf(qh, qh->ferr, 6280, "qhull internal error (qh_maydropneighbor): not valid for simplicial neighbor f%d of f%d while adding furthest p%d\n",
            neighbor->id, facet->id, qh->furthest_id);
        qh_errexit2(qh, qh_ERRqhull, neighbor, facet);
      }
      zinc_(Zdropneighbor);
      qh_setdel(neighbor->neighbors, facet);
      if (qh_setsize(qh, neighbor->neighbors) < qh->hull_dim) {
        zinc_(Zdropdegen);
        qh_appendmergeset(qh, neighbor, neighbor, MRGdegen, 0.0, qh_ANGLEnone);
        trace2((qh, qh->ferr, 2023, "qh_maydropneighbors: f%d is degenerate.\n", neighbor->id));
      }
      qh_setdel(facet->neighbors, neighbor);
      neighborp--;  /* repeat, deleted a neighbor */
    }
  }
  if (qh_setsize(qh, facet->neighbors) < qh->hull_dim) {
    zinc_(Zdropdegen);
    qh_appendmergeset(qh, facet, facet, MRGdegen, 0.0, qh_ANGLEnone);
    trace2((qh, qh->ferr, 2024, "qh_maydropneighbors: f%d is degenerate.\n", facet->id));
  }
} /* maydropneighbor */
    # 对于每一个合并操作在 qh.degen_mergeset 上进行迭代
    for each merge on qh.degen_mergeset
      # 如果这个合并是多余的
      if redundant merge
        # 如果非多余的面被合并到多余的面中
        if non-redundant facet merged into redundant facet
          # 重新检查面以确定是否多余
          recheck facet for redundancy
        else
          # 合并多余的面到其他面中
          merge redundant facet into other facet
/*
int qh_merge_degenredundant(qhT *qh) {
  // 定义变量
  int size;
  mergeT *merge;
  facetT *bestneighbor, *facet1, *facet2, *facet3;
  realT dist, mindist, maxdist;
  vertexT *vertex, **vertexp;
  int nummerges= 0;
  mergeType mergetype;
  setT *mergedfacets;

  // 打印跟踪信息到日志文件
  trace2((qh, qh->ferr, 2095, "qh_merge_degenredundant: merge %d degenerate, redundant, and mirror facets\n",
    qh_setsize(qh, qh->degen_mergeset)));
  
  // 创建临时集合以存储合并后的面
  mergedfacets= qh_settemp(qh, qh->TEMPsize);
  
  // 循环处理所有待合并的面
  while ((merge= (mergeT *)qh_setdellast(qh->degen_mergeset))) {
    facet1= merge->facet1;
    facet2= merge->facet2;
    mergetype= merge->mergetype;
    // 释放 merge 结构的内存
    qh_memfree(qh, merge, (int)sizeof(mergeT)); /* 'merge' is invalidated */
    
    // 如果 facet1 已经是可见的，则继续处理下一个 merge
    if (facet1->visible)
      continue;
    
    // 标记 facet1 不再是退化或冗余面
    facet1->degenerate= False;
    facet1->redundant= False;
    
    // 如果跟踪级别等于总合并次数减一，则设置跟踪标志
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
      qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
    
    // 如果合并类型为 MRGredundant
    if (mergetype == MRGredundant) {
      // 增加冗余面合并计数
      zinc_(Zredundant);
      
      // 获取 facet2 的替代面（如果 facet2 可见，则为 facet2 自身）
      facet3= qh_getreplacement(qh, facet2);
      
      // 如果找不到替代面，则输出错误信息并退出
      if (!facet3) {
          qh_fprintf(qh, qh->ferr, 6097, "qhull internal error (qh_merge_degenredunant): f%d is redundant but visible f%d has no replacement\n",
               facet1->id, getid_(facet2));
          qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
      }
      
      // 将 facet3 添加到合并后的面集合中
      qh_setunique(qh, &mergedfacets, facet3);
      
      // 如果 facet1 和 facet3 相同，则继续处理下一个 merge
      if (facet1 == facet3) {
        continue;
      }
      
      // 打印合并操作的跟踪信息到日志文件
      trace2((qh, qh->ferr, 2025, "qh_merge_degenredundant: merge redundant f%d into f%d (arg f%d)\n",
            facet1->id, facet3->id, facet2->id));
      
      // 执行面的合并操作，不包括顶点
      qh_mergefacet(qh, facet1, facet3, mergetype, NULL, NULL, !qh_MERGEapex);
      
      // 合并计数加一
      nummerges++;
      
      // merge 距离已经计入，不再处理
      /* merge distance is already accounted for */
    }
    }else {  /* mergetype == MRGdegen or MRGmirror, other merges may have fixed */
      // 如果 mergetype 是 MRGdegen 或 MRGmirror，其他合并可能已经修复
      if (!(size= qh_setsize(qh, facet1->neighbors))) {
        // 如果 facet1 的邻居集合为空
        zinc_(Zdelfacetdup);
        // 增加 Zdelfacetdup 统计计数
        trace2((qh, qh->ferr, 2026, "qh_merge_degenredundant: facet f%d has no neighbors.  Deleted\n", facet1->id));
        // 输出调试信息，表示 facet1 没有邻居，被删除
        qh_willdelete(qh, facet1, NULL);
        // 告知 Qhull 删除 facet1
        FOREACHvertex_(facet1->vertices) {
          // 遍历 facet1 的顶点集合
          qh_setdel(vertex->neighbors, facet1);
          // 从顶点的邻居集合中删除 facet1
          if (!SETfirst_(vertex->neighbors)) {
            // 如果顶点的邻居集合为空
            zinc_(Zdegenvertex);
            // 增加 Zdegenvertex 统计计数
            trace2((qh, qh->ferr, 2027, "qh_merge_degenredundant: deleted v%d because f%d has no neighbors\n",
                 vertex->id, facet1->id));
            // 输出调试信息，表示删除顶点 vertex 因为 facet1 没有邻居
            vertex->deleted= True;
            // 标记顶点 vertex 已删除
            qh_setappend(qh, &qh->del_vertices, vertex);
            // 将顶点 vertex 加入 Qhull 的已删除顶点列表中
          }
        }
        nummerges++;
        // 合并计数加一
      }else if (size < qh->hull_dim) {
        // 如果 facet1 的邻居数量小于 hull_dim
        bestneighbor= qh_findbestneighbor(qh, facet1, &dist, &mindist, &maxdist);
        // 找到 facet1 的最佳邻居
        trace2((qh, qh->ferr, 2028, "qh_merge_degenredundant: facet f%d has %d neighbors, merge into f%d dist %2.2g\n",
              facet1->id, size, bestneighbor->id, dist));
        // 输出调试信息，表示 facet1 有 size 个邻居，将其合并到最佳邻居 bestneighbor 中
        qh_mergefacet(qh, facet1, bestneighbor, mergetype, &mindist, &maxdist, !qh_MERGEapex);
        // 执行 facet 合并操作
        nummerges++;
        // 合并计数加一
        if (qh->PRINTstatistics) {
          // 如果需要打印统计信息
          zinc_(Zdegen);
          // 增加 Zdegen 统计计数
          wadd_(Wdegentot, dist);
          // 将 dist 添加到 Wdegentot 中
          wmax_(Wdegenmax, dist);
          // 更新 Wdegenmax 的最大值
        }
      } /* else, another merge fixed the degeneracy and redundancy tested */
      // 否则，其他合并操作已修复了退化和冗余情况
    }
  }
  qh_settempfree(qh, &mergedfacets);
  // 释放临时集合 mergedfacets
  return nummerges;
  // 返回合并操作的计数
/* merge_degenredundant */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="merge_nonconvex">-</a>

  qh_merge_nonconvex(qh, facet1, facet2, mergetype )
    remove non-convex ridge between facet1 into facet2
    mergetype gives why the facet's are non-convex

  returns:
    merges one of the facets into the best neighbor

  notes:
    mergetype is MRGcoplanar..MRGconvex

  design:
    if one of the facets is a new facet
      prefer merging new facet into old facet
    find best neighbors for both facets
    merge the nearest facet into its best neighbor
    update the statistics
*/
void qh_merge_nonconvex(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype) {
  facetT *bestfacet, *bestneighbor, *neighbor, *merging, *merged;
  realT dist, dist2, mindist, mindist2, maxdist, maxdist2;

  if (mergetype < MRGcoplanar || mergetype > MRGconcavecoplanar) {
    qh_fprintf(qh, qh->ferr, 6398, "qhull internal error (qh_merge_nonconvex): expecting mergetype MRGcoplanar..MRGconcavecoplanar.  Got merge f%d and f%d type %d\n",
      facet1->id, facet2->id, mergetype);
    qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
  }
  if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
  trace3((qh, qh->ferr, 3003, "qh_merge_nonconvex: merge #%d for f%d and f%d type %d\n",
      zzval_(Ztotmerge) + 1, facet1->id, facet2->id, mergetype));
  /* concave or coplanar */
  if (!facet1->newfacet) {
    bestfacet= facet2;   /* avoid merging old facet if new is ok */
    facet2= facet1;
    facet1= bestfacet;
  }else
    bestfacet= facet1;
  bestneighbor= qh_findbestneighbor(qh, bestfacet, &dist, &mindist, &maxdist);
  neighbor= qh_findbestneighbor(qh, facet2, &dist2, &mindist2, &maxdist2);
  if (dist < dist2) {
    merging= bestfacet;
    merged= bestneighbor;
  }else if (qh->AVOIDold && !facet2->newfacet
  && ((mindist >= -qh->MAXcoplanar && maxdist <= qh->max_outside)
       || dist * 1.5 < dist2)) {
    zinc_(Zavoidold);
    wadd_(Wavoidoldtot, dist);
    wmax_(Wavoidoldmax, dist);
    trace2((qh, qh->ferr, 2029, "qh_merge_nonconvex: avoid merging old facet f%d dist %2.2g.  Use f%d dist %2.2g instead\n",
           facet2->id, dist2, facet1->id, dist2));
    merging= bestfacet;
    merged= bestneighbor;
  }else {
    merging= facet2;
    merged= neighbor;
    dist= dist2;
    mindist= mindist2;
    maxdist= maxdist2;
  }
  qh_mergefacet(qh, merging, merged, mergetype, &mindist, &maxdist, !qh_MERGEapex);
  /* caller merges qh_degenredundant */
  if (qh->PRINTstatistics) {
    if (mergetype == MRGanglecoplanar) {
      zinc_(Zacoplanar);
      wadd_(Wacoplanartot, dist);
      wmax_(Wacoplanarmax, dist);
    }else if (mergetype == MRGconcave) {
      zinc_(Zconcave);
      wadd_(Wconcavetot, dist);
      wmax_(Wconcavemax, dist);
    }
  }
}
    }else if (mergetype == MRGconcavecoplanar) {
      zinc_(Zconcavecoplanar);
      wadd_(Wconcavecoplanartot, dist);
      wmax_(Wconcavecoplanarmax, dist);
    }else { /* MRGcoplanar */
      zinc_(Zcoplanar);
      // 将距离添加到总的同面合并权重中
      wadd_(Wcoplanartot, dist);
      // 更新同面合并中的最大距离权重
      wmax_(Wcoplanarmax, dist);
    }
} /* merge_nonconvex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="merge_pinchedvertices">-</a>

  qh_merge_pinchedvertices(qh, apex )
    merge pinched vertices in qh.vertex_mergeset to avoid qh_forcedmerges of dupridges

  notes:
    only called by qh_all_vertexmerges
    hull_dim >= 3

  design:
    make vertex neighbors if necessary
    for each pinched vertex
      determine the ridges for the pinched vertex (make ridges as needed)
      merge the pinched vertex into the horizon vertex
      merge the degenerate and redundant facets that result
    check and resolve new dupridges
*/
void qh_merge_pinchedvertices(qhT *qh, int apexpointid /* qh.newfacet_list */) {
  mergeT *merge, *mergeA, **mergeAp;
  vertexT *vertex, *vertex2;
  realT dist;
  boolT firstmerge= True;

  // 创建顶点的邻居关系（如果需要）
  qh_vertexneighbors(qh);
  
  // 检查是否存在未清空的可见列表、新面列表或新顶点列表，若存在则报错退出
  if (qh->visible_list || qh->newfacet_list || qh->newvertex_list) {
    qh_fprintf(qh, qh->ferr, 6402, "qhull internal error (qh_merge_pinchedvertices): qh.visible_list (f%d), newfacet_list (f%d), or newvertex_list (v%d) not empty\n",
      getid_(qh->visible_list), getid_(qh->newfacet_list), getid_(qh->newvertex_list));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  
  // 将 visible_list、newfacet_list 设置为 facet_tail，newvertex_list 设置为 vertex_tail
  qh->visible_list= qh->newfacet_list= qh->facet_tail;
  qh->newvertex_list= qh->vertex_tail;
  
  // 设置 isRenameVertex 为 True，禁用 qh_checkfacet 中的重复边顶点检查
  qh->isRenameVertex= True; /* disable duplicate ridge vertices check in qh_checkfacet */
  
  // 从 vertex_mergeset 中逐个处理 pinched merges
  while ((merge= qh_next_vertexmerge(qh /* qh.vertex_mergeset */))) { /* only one at a time from qh_getpinchedmerges */
    // 如果启用了跟踪模式，并且满足条件，则设置跟踪级别
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
      qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
    
    // 如果 merge 的类型是 MRGsubridge，增加 Zpinchedvertex 计数，并输出跟踪信息
    if (merge->mergetype == MRGsubridge) {
      zzinc_(Zpinchedvertex);
      trace1((qh, qh->ferr, 1050, "qh_merge_pinchedvertices: merge one of %d pinched vertices before adding apex p%d.  Try to resolve duplicate ridges in newfacets\n",
        qh_setsize(qh, qh->vertex_mergeset)+1, apexpointid));
      qh_remove_mergetype(qh, qh->vertex_mergeset, MRGsubridge);
    } else {
      // 否则，增加 Zpinchduplicate 计数，并输出首次合并信息
      zzinc_(Zpinchduplicate);
      if (firstmerge)
        trace1((qh, qh->ferr, 1056, "qh_merge_pinchedvertices: merge %d pinched vertices from dupridges in merged facets, apex p%d\n",
           qh_setsize(qh, qh->vertex_mergeset)+1, apexpointid));
      firstmerge= False;
    }
    
    // 获取 merge 的两个顶点和距离
    vertex= merge->vertex1;
    vertex2= merge->vertex2;
    dist= merge->distance;
    
    // 释放 merge 对象占用的内存，并调用 qh_rename_adjacentvertex 进行顶点重命名
    qh_memfree(qh, merge, (int)sizeof(mergeT)); /* merge is invalidated */
    qh_rename_adjacentvertex(qh, vertex, vertex2, dist);
    
#ifndef qh_NOtrace
    // 如果qh->IStracing的值大于等于2，则进入条件判断
    if (qh->IStracing >= 2) {
        // 对qh->degen_mergeset中的每一个mergeA进行循环处理
        FOREACHmergeA_(qh->degen_mergeset) {
            // 如果mergeA的mergetype为MRGdegen，则输出日志指明合并了一个退化的面f%d到相邻的面中
            if (mergeA->mergetype == MRGdegen) {
                qh_fprintf(qh, qh->ferr, 2072, "qh_merge_pinchedvertices: merge degenerate f%d into an adjacent facet\n", mergeA->facet1->id);
            } else {
                // 否则，输出日志指明将面f%d合并到面f%d中，同时指定合并类型为mergeA->mergetype
                qh_fprintf(qh, qh->ferr, 2084, "qh_merge_pinchedvertices: merge f%d into f%d mergeType %d\n", mergeA->facet1->id, mergeA->facet2->id, mergeA->mergetype);
            }
        }
    }
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="merge_twisted">-</a>

  qh_merge_twisted(qh, facet1, facet2 )
    remove twisted ridge between facet1 into facet2 or report error

  returns:
    merges one of the facets into the best neighbor

  notes:
    a twisted ridge has opposite vertices that are convex and concave

  design:
    find best neighbors for both facets
    error if wide merge
    merge the nearest facet into its best neighbor
    update statistics
*/
void qh_merge_twisted(qhT *qh, facetT *facet1, facetT *facet2) {
  facetT *neighbor2, *neighbor, *merging, *merged;
  vertexT *bestvertex, *bestpinched;
  realT dist, dist2, mindist, mindist2, maxdist, maxdist2, mintwisted, bestdist;

  // 如果在跟踪模式下，设置详细跟踪级别
  if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;

  // 记录详细的跟踪信息
  trace3((qh, qh->ferr, 3050, "qh_merge_twisted: merge #%d for twisted f%d and f%d\n",
      zzval_(Ztotmerge) + 1, facet1->id, facet2->id));

  /* twisted */
  // 查找facet1的最佳相邻面及其相关距离信息
  neighbor= qh_findbestneighbor(qh, facet1, &dist, &mindist, &maxdist);
  // 查找facet2的最佳相邻面及其相关距离信息
  neighbor2= qh_findbestneighbor(qh, facet2, &dist2, &mindist2, &maxdist2);

  // 计算twisted的阈值
  mintwisted= qh_RATIOtwisted * qh->ONEmerge;
  maximize_(mintwisted, facet1->maxoutside);
  maximize_(mintwisted, facet2->maxoutside);

  // 如果两个facet之间的距离超过twisted阈值，输出错误信息并退出
  if (dist > mintwisted && dist2 > mintwisted) {
    bestdist= qh_vertex_bestdist2(qh, facet1->vertices, &bestvertex, &bestpinched);
    if (bestdist > mintwisted) {
      // 输出twisted面不包含pinched顶点的错误信息
      qh_fprintf(qh, qh->ferr, 6417, "qhull precision error (qh_merge_twisted): twisted facet f%d does not contain pinched vertices.  Too wide to merge into neighbor.  mindist %2.2g maxdist %2.2g vertexdist %2.2g maxpinched %2.2g neighbor f%d mindist %2.2g maxdist %2.2g\n",
        facet1->id, mindist, maxdist, bestdist, mintwisted, facet2->id, mindist2, maxdist2);
    } else {
      // 输出twisted面包含pinched顶点的错误信息
      qh_fprintf(qh, qh->ferr, 6418, "qhull precision error (qh_merge_twisted): twisted facet f%d with pinched vertices.  Could merge vertices, but too wide to merge into neighbor.   mindist %2.2g maxdist %2.2g vertexdist %2.2g neighbor f%d mindist %2.2g maxdist %2.2g\n",
        facet1->id, mindist, maxdist, bestdist, facet2->id, mindist2, maxdist2);
    }
    // 退出程序，报告宽度合并错误
    qh_errexit2(qh, qh_ERRwide, facet1, facet2);
  }

  // 选择距离较短的facet进行合并
  if (dist < dist2) {
    merging= facet1;
    merged= neighbor;
  } else {
    // 忽略qh.AVOIDold（'Q4'）
    merging= facet2;
    merged= neighbor2;
    dist= dist2;
    mindist= mindist2;
    maxdist= maxdist2;
  }

  // 执行facet的合并操作
  qh_mergefacet(qh, merging, merged, MRGtwisted, &mindist, &maxdist, !qh_MERGEapex);
  
  // 更新统计数据
  zinc_(Ztwisted);
  wadd_(Wtwistedtot, dist);
  wmax_(Wtwistedmax, dist);
} /* merge_twisted */
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle">-</a>

  qh_mergecycle(qh, samecycle, newfacet )
    merge a cycle of facets starting at samecycle into a newfacet
    newfacet is a horizon facet with ->normal
    samecycle facets are simplicial from an apex

  returns:
    initializes vertex neighbors on first merge
    samecycle deleted (placed on qh.visible_list)
    newfacet at end of qh.facet_list
    deleted vertices on qh.del_vertices

  notes:
    only called by qh_mergecycle_all for multiple, same cycle facets
    see qh_mergefacet

  design:
    make vertex neighbors if necessary
    make ridges for newfacet
    merge neighbor sets of samecycle into newfacet
    merge ridges of samecycle into newfacet
    merge vertex neighbors of samecycle into newfacet
    make apex of samecycle the apex of newfacet
    if newfacet wasn't a new facet
      add its vertices to qh.newvertex_list
    delete samecycle facets a make newfacet a newfacet
*/
void qh_mergecycle(qhT *qh, facetT *samecycle, facetT *newfacet) {
  int traceonce= False, tracerestore= 0;
  vertexT *apex;
#ifndef qh_NOtrace
  facetT *same;
#endif

  zzinc_(Ztotmerge);  /* Increment total merge count for statistics */
  if (qh->REPORTfreq2 && qh->POSTmerging) {
    if (zzval_(Ztotmerge) > qh->mergereport + qh->REPORTfreq2)
      qh_tracemerging(qh);  /* Trace merging if conditions are met */
  }
#ifndef qh_NOtrace
  if (qh->TRACEmerge == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;  /* Set tracing levels */
  trace2((qh, qh->ferr, 2030, "qh_mergecycle: merge #%d for facets from cycle f%d into coplanar horizon f%d\n",
        zzval_(Ztotmerge), samecycle->id, newfacet->id));  /* Trace message for merging operation */
  if (newfacet == qh->tracefacet) {
    tracerestore= qh->IStracing;
    qh->IStracing= 4;  /* Set tracing level to 4 for detailed tracing */
    qh_fprintf(qh, qh->ferr, 8068, "qh_mergecycle: ========= trace merge %d of samecycle %d into trace f%d, furthest is p%d\n",
               zzval_(Ztotmerge), samecycle->id, newfacet->id,  qh->furthest_id);  /* Trace specific merge details */
    traceonce= True;
  }
  if (qh->IStracing >=4) {
    qh_fprintf(qh, qh->ferr, 8069, "  same cycle:");
    FORALLsame_cycle_(samecycle)
      qh_fprintf(qh, qh->ferr, 8070, " f%d", same->id);  /* Trace all facets in the same cycle */
    qh_fprintf(qh, qh->ferr, 8071, "\n");
  }
  if (qh->IStracing >=4)
    qh_errprint(qh, "MERGING CYCLE", samecycle, newfacet, NULL, NULL);  /* Trace error message related to merging cycle */
#endif /* !qh_NOtrace */
  if (newfacet->tricoplanar) {  /* Handle tricoplanar facets */
    if (!qh->TRInormals) {
      qh_fprintf(qh, qh->ferr, 6224, "qhull internal error (qh_mergecycle): does not work for tricoplanar facets.  Use option 'Q11'\n");
      qh_errexit(qh, qh_ERRqhull, newfacet, NULL);  /* Exit with error for unsupported tricoplanar facets */
    }
    newfacet->tricoplanar= False;  /* Clear tricoplanar flag */
    newfacet->keepcentrum= False;  /* Clear keepcentrum flag */
  }
  if (qh->CHECKfrequently)
    qh_checkdelridge(qh);  /* Check and delete ridges frequently */
  if (!qh->VERTEXneighbors)
    /* If vertex neighbors are not yet initialized, initialize them */
    qh_setvoronoi_all(qh);
}
    # 调用 qh_vertexneighbors 函数，传入参数 qh
    qh_vertexneighbors(qh);
  # 将 samecycle->vertices 的第一个顶点设置为 apex
  apex= SETfirstt_(samecycle->vertices, vertexT);
  # 对 newfacet 进行边缘创建操作
  qh_makeridges(qh, newfacet);
  # 合并同一循环中的邻居
  qh_mergecycle_neighbors(qh, samecycle, newfacet);
  # 合并同一循环中的边缘
  qh_mergecycle_ridges(qh, samecycle, newfacet);
  # 合并同一循环中的顶点邻居
  qh_mergecycle_vneighbors(qh, samecycle, newfacet);
  # 如果 newfacet->vertices 中第一个顶点不是 apex，则将 apex 添加到顶点列表的开头
  if (SETfirstt_(newfacet->vertices, vertexT) != apex)
    qh_setaddnth(qh, &newfacet->vertices, 0, apex);  /* apex has last id */
  # 如果 newfacet 不是新创建的，则调用 qh_newvertices 处理 newfacet->vertices
  if (!newfacet->newfacet)
    qh_newvertices(qh, newfacet->vertices);
  # 合并同一循环中的面
  qh_mergecycle_facets(qh, samecycle, newfacet);
  # 跟踪合并过程，此处处理 MRGcoplanarhorizon 类型的合并
  qh_tracemerge(qh, samecycle, newfacet, MRGcoplanarhorizon);
  /* 在 qh_forcedmerges() 之后检查 degen_redundant_neighbors */
  if (traceonce) {
    # 输出跟踪消息到 qh->ferr
    qh_fprintf(qh, qh->ferr, 8072, "qh_mergecycle: end of trace facet\n");
    # 恢复跟踪状态为 tracerestore
    qh->IStracing= tracerestore;
  }
} /* mergecycle */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_all">-</a>

  qh_mergecycle_all(qh, facetlist, wasmerge )
    merge all samecycles of coplanar facets into horizon
    don't merge facets with ->mergeridge (these already have ->normal)
    all facets are simplicial from apex
    all facet->cycledone == False

  returns:
    all newfacets merged into coplanar horizon facets
    deleted vertices on  qh.del_vertices
    sets wasmerge if any merge

  notes:
    called by qh_premerge
    calls qh_mergecycle for multiple, same cycle facets

  design:
    for each facet on facetlist
      skip facets with dupridges and normals
      check that facet is in a samecycle (->mergehorizon)
      if facet only member of samecycle
        sets vertex->delridge for all vertices except apex
        merge facet into horizon
      else
        mark all facets in samecycle
        remove facets with dupridges from samecycle
        merge samecycle into horizon (deletes facets from facetlist)
*/
void qh_mergecycle_all(qhT *qh, facetT *facetlist, boolT *wasmerge) {
  facetT *facet, *same, *prev, *horizon, *newfacet;
  facetT *samecycle= NULL, *nextfacet, *nextsame;
  vertexT *apex, *vertex, **vertexp;
  int cycles=0, total=0, facets, nummerge, numdegen= 0;

  trace2((qh, qh->ferr, 2031, "qh_mergecycle_all: merge new facets into coplanar horizon facets.  Bulk merge a cycle of facets with the same horizon facet\n"));
  // 循环处理每个输入的 facet
  for (facet=facetlist; facet && (nextfacet= facet->next); facet= nextfacet) {
    // 如果 facet 已经有法向量（normal），则跳过
    if (facet->normal)
      continue;
    // 如果 facet 没有 mergehorizon，输出错误信息并退出
    if (!facet->mergehorizon) {
      qh_fprintf(qh, qh->ferr, 6225, "qhull internal error (qh_mergecycle_all): f%d without normal\n", facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    // 获取与 facet 相邻的第一个 horizon facet
    horizon= SETfirstt_(facet->neighbors, facetT);
    // 如果 facet 是同一 cycle 的唯一成员
    if (facet->f.samecycle == facet) {
      // 如果正在跟踪 merge 操作，更新跟踪状态
      if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
        qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
      // 增加统计信息，表明正在处理 horizon facet
      zinc_(Zonehorizon);
      /* merge distance done in qh_findhorizon */
      // 获取 facet 的顶点集合的第一个顶点作为 apex
      apex= SETfirstt_(facet->vertices, vertexT);
      // 遍历 facet 的所有顶点，设置除 apex 外的顶点的 delridge 标志为 True
      FOREACHvertex_(facet->vertices) {
        if (vertex != apex)
          vertex->delridge= True;
      }
      // 将 horizon 的 newcycle 标记为 NULL
      horizon->f.newcycle= NULL;
      // 执行 merge 操作，将 facet 合并到 horizon 中，采用 MRGcoplanarhorizon 操作标志
      qh_mergefacet(qh, facet, horizon, MRGcoplanarhorizon, NULL, NULL, qh_MERGEapex);
    }else {
      // 将当前面片的同周期面片列表的头部赋值给samecycle，facets计数清零
      samecycle= facet;
      facets= 0;
      prev= facet;
      // 遍历同周期面片列表
      for (same= facet->f.samecycle; same;  /* FORALLsame_cycle_(facet) */
           same= (same == facet ? NULL :nextsame)) { /* ends at facet */
        nextsame= same->f.samecycle;
        // 如果同周期面片已标记为处理完毕或不可见，报告无限循环
        if (same->cycledone || same->visible)
          qh_infiniteloop(qh, same);
        same->cycledone= True;
        // 如果同周期面片有法向量
        if (same->normal) {
          // 解除prev和same之间的链接 ->mergeridge
          prev->f.samecycle= same->f.samecycle; /* unlink ->mergeridge */
          same->f.samecycle= NULL;
        }else {
          prev= same;
          facets++;
        }
      }
      // 跳过已处理完毕的下一个面片
      while (nextfacet && nextfacet->cycledone)  /* will delete samecycle */
        nextfacet= nextfacet->next;
      // 重置horizon的新周期标记
      horizon->f.newcycle= NULL;
      // 合并周期内的面片
      qh_mergecycle(qh, samecycle, horizon);
      // 更新nummerge并根据上限值qh_MAXnummerge进行限制
      nummerge= horizon->nummerge + facets;
      if (nummerge > qh_MAXnummerge)
        horizon->nummerge= qh_MAXnummerge;
      else
        horizon->nummerge= (short unsigned int)nummerge; /* limited to 9 bits by qh_MAXnummerge, -Wconversion */
      // 增加计数器Zcyclehorizon
      zzinc_(Zcyclehorizon);
      // 更新总面片计数
      total += facets;
      // 更新面片数统计变量Zcyclefacettot
      zzadd_(Zcyclefacettot, facets);
      // 更新最大面片数统计变量Zcyclefacetmax
      zmax_(Zcyclefacetmax, facets);
    }
    // 周期计数增加
    cycles++;
  }
  // 若存在周期
  if (cycles) {
    // 遍历所有新面片
    FORALLnew_facets {
      /* qh_maybe_duplicateridges postponed since qh_mergecycle_ridges deletes ridges without calling qh_delridge_merge */
      // 若新面片具有共面地平线
      if (newfacet->coplanarhorizon) {
        // 测试并消除冗余邻面片
        qh_test_redundant_neighbors(qh, newfacet);
        // 检查并可能复制边界
        qh_maybe_duplicateridges(qh, newfacet);
        // 标记共面地平线为False
        newfacet->coplanarhorizon= False;
      }
    }
    // 合并去冗余面片，并设置wasmerge为True
    numdegen += qh_merge_degenredundant(qh);
    *wasmerge= True;
    // 输出合并周期和去冗余面片的跟踪信息
    trace1((qh, qh->ferr, 1013, "qh_mergecycle_all: merged %d same cycles or facets into coplanar horizons and %d degenredundant facets\n",
      cycles, numdegen));
  }
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_facets">-</a>

  qh_mergecycle_facets(qh, samecycle, newfacet )
    finish merge of samecycle into newfacet

  returns:
    samecycle prepended to visible_list for later deletion and partitioning
      each facet->f.replace == newfacet

    newfacet moved to end of qh.facet_list
      makes newfacet a newfacet (get's facet1->id if it was old)
      sets newfacet->newmerge
      clears newfacet->center (unless merging into a large facet)
      clears newfacet->tested and ridge->tested for facet1

    adds neighboring facets to facet_mergeset if redundant or degenerate

  design:
    make newfacet a new facet and set its flags
    move samecycle facets to qh.visible_list for later deletion
    unless newfacet is large
      remove its centrum
*/
void qh_mergecycle_facets(qhT *qh, facetT *samecycle, facetT *newfacet) {
  facetT *same, *next;

  trace4((qh, qh->ferr, 4030, "qh_mergecycle_facets: make newfacet new and samecycle deleted\n"));
  qh_removefacet(qh, newfacet);  /* 将newfacet从当前位置移除 */
  qh_appendfacet(qh, newfacet);  /* 将newfacet添加到qh->facet_list末尾 */
  newfacet->newfacet= True;  /* 标记newfacet为新facet */
  newfacet->simplicial= False;  /* 将newfacet标记为非单纯形 */
  newfacet->newmerge= True;  /* 标记newfacet为新合并的 */

  for (same= samecycle->f.samecycle; same; same= (same == samecycle ?  NULL : next)) {
    next= same->f.samecycle;  /* 保存下一个samecycle中的facet到next变量 */
    qh_willdelete(qh, same, newfacet);  /* 处理samecycle中每个facet，标记为将删除 */
  }
  if (newfacet->center
      && qh_setsize(qh, newfacet->vertices) <= qh->hull_dim + qh_MAXnewcentrum) {
    qh_memfree(qh, newfacet->center, qh->normal_size);  /* 释放newfacet的中心 */
    newfacet->center= NULL;  /* 将newfacet的中心设为NULL */
  }
  trace3((qh, qh->ferr, 3004, "qh_mergecycle_facets: merged facets from cycle f%d into f%d\n",
             samecycle->id, newfacet->id));
} /* mergecycle_facets */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_neighbors">-</a>

  qh_mergecycle_neighbors(qh, samecycle, newfacet )
    add neighbors for samecycle facets to newfacet

  returns:
    newfacet with updated neighbors and vice-versa
    newfacet has ridges
    all neighbors of newfacet marked with qh.visit_id
    samecycle facets marked with qh.visit_id-1
    ridges updated for simplicial neighbors of samecycle with a ridge

  notes:
    assumes newfacet not in samecycle
    usually, samecycle facets are new, simplicial facets without internal ridges
      not so if horizon facet is coplanar to two different samecycles

  see:
    qh_mergeneighbors()

  design:
    check samecycle
    delete neighbors from newfacet that are also in samecycle
    for each neighbor of a facet in samecycle
      if neighbor is simplicial
        if first visit
          move the neighbor relation to newfacet
          update facet links for its ridges
        else
          make ridges for neighbor
          remove samecycle reference
      else
        update neighbor sets
*/
void qh_mergecycle_neighbors(qhT *qh, facetT *samecycle, facetT *newfacet) {
  // 定义变量
  facetT *same, *neighbor, **neighborp;
  int delneighbors= 0, newneighbors= 0;
  unsigned int samevisitid;
  ridgeT *ridge, **ridgep;

  // 标记 samecycle 中所有面的访问标识，并检查是否有重复访问或已可见的面
  samevisitid= ++qh->visit_id;
  FORALLsame_cycle_(samecycle) {
    if (same->visitid == samevisitid || same->visible)
      qh_infiniteloop(qh, samecycle);
    same->visitid= samevisitid;
  }
  
  // 标记 newfacet 的访问标识
  newfacet->visitid= ++qh->visit_id;
  
  // 跟踪日志，删除 newfacet 中与 samecycle 共享的邻面
  trace4((qh, qh->ferr, 4031, "qh_mergecycle_neighbors: delete shared neighbors from newfacet\n"));
  FOREACHneighbor_(newfacet) {
    if (neighbor->visitid == samevisitid) {
      SETref_(neighbor)= NULL;  /* samecycle neighbors deleted */
      delneighbors++;
    }else
      neighbor->visitid= qh->visit_id;
  }
  
  // 压缩 newfacet 的邻面集合
  qh_setcompact(qh, newfacet->neighbors);

  // 跟踪日志，更新邻面
  trace4((qh, qh->ferr, 4032, "qh_mergecycle_neighbors: update neighbors\n"));
  FORALLsame_cycle_(samecycle) {
    FOREACHneighbor_(same) {
      if (neighbor->visitid == samevisitid)
        continue;
      if (neighbor->simplicial) {
        // 如果邻面是单纯的，则处理
        if (neighbor->visitid != qh->visit_id) {
          // 将 neighbor 添加到 newfacet 的邻面集合中，并更新 neighbor 的邻面
          qh_setappend(qh, &newfacet->neighbors, neighbor);
          qh_setreplace(qh, neighbor->neighbors, same, newfacet);
          newneighbors++;
          neighbor->visitid= qh->visit_id;
          
          // 更新 neighbor 的边界
          FOREACHridge_(neighbor->ridges) { /* update ridge in case of qh_makeridges */
            if (ridge->top == same) {
              ridge->top= newfacet;
              break;
            } else if (ridge->bottom == same) {
              ridge->bottom= newfacet;
              break;
            }
          }
        } else {
          // 否则，调用 qh_makeridges 处理 neighbor
          qh_makeridges(qh, neighbor);
          qh_setdel(neighbor->neighbors, same);
          /* same can't be horizon facet for neighbor */
        }
      } else { /* non-simplicial neighbor */
        // 处理非单纯邻面
        qh_setdel(neighbor->neighbors, same);
        if (neighbor->visitid != qh->visit_id) {
          // 将 newfacet 添加到 neighbor 的邻面集合中，并更新 newfacet 的邻面
          qh_setappend(qh, &neighbor->neighbors, newfacet);
          qh_setappend(qh, &newfacet->neighbors, neighbor);
          neighbor->visitid= qh->visit_id;
          newneighbors++;
        }
      }
    }
  }
  
  // 记录日志，输出删除的邻面数和添加的邻面数
  trace2((qh, qh->ferr, 2032, "qh_mergecycle_neighbors: deleted %d neighbors and added %d\n",
             delneighbors, newneighbors));
} /* mergecycle_neighbors */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_ridges">-</a>

  qh_mergecycle_ridges(qh, samecycle, newfacet )
    add ridges/neighbors for facets in samecycle to newfacet
    all new/old neighbors of newfacet marked with qh.visit_id
    facets in samecycle marked with qh.visit_id-1
    newfacet marked with qh.visit_id

  returns:
    newfacet has merged ridges

  notes:
    ridge already updated for simplicial neighbors of samecycle with a ridge
    qh_checkdelridge called by qh_mergecycle

  see:
    qh_mergeridges()
    qh_makeridges()

  design:
    remove ridges between newfacet and samecycle
    # 对于每个在同一个循环中的面（facet），依次处理
    for each facet in samecycle
      # 对于面（facet）中的每条边（ridge），依次处理
      for each ridge in facet
        # 更新边（ridge）中的面（facet）指针
        update facet pointers in ridge
        # 跳过在 qh_mergecycle_neighbors 中处理过的边（ridge）
        skip ridges processed in qh_mergecycle_neighors
        # 释放 newfacet 和 samecycle 之间的边（ridge）
        free ridges between newfacet and samecycle
        # 在同一个循环中的面（facet）之间的边（ridge）第二次访问时释放
        free ridges between facets of samecycle (on 2nd visit)
        # 将剩余的边（ridge）追加到 newfacet 中
        append remaining ridges to newfacet
      # 如果面（facet）是简单的（simplicial）
      if simplicial facet
        # 对于面（facet）的每个邻居（neighbor），依次处理
        for each neighbor of facet
          # 如果邻居（neighbor）也是简单的面（facet），且不是同一个循环中的面（facet）或者 newfacet
          if simplicial facet
          and not samecycle facet or newfacet
            # 创建邻居（neighbor）和 newfacet 之间的边（ridge）
            make ridge between neighbor and newfacet
void qh_mergecycle_ridges(qhT *qh, facetT *samecycle, facetT *newfacet) {
  facetT *same, *neighbor= NULL;  /* 声明facetT类型的指针变量same和neighbor，初始置为NULL */
  int numold=0, numnew=0;  /* 声明整型变量numold和numnew，初始置为0 */
  int neighbor_i, neighbor_n;  /* 声明整型变量neighbor_i和neighbor_n */
  unsigned int samevisitid;  /* 声明无符号整型变量samevisitid */
  ridgeT *ridge, **ridgep;  /* 声明ridgeT类型的指针变量ridge和ridgep */
  boolT toporient;  /* 声明boolT类型的变量toporient */
  void **freelistp; /* used if !qh_NOmem by qh_memfree_() */  /* 声明void类型的指针变量freelistp，用于内存释放 */

  trace4((qh, qh->ferr, 4033, "qh_mergecycle_ridges: delete shared ridges from newfacet\n"));
  samevisitid= qh->visit_id -1;  /* 计算samevisitid，为qh->visit_id减1 */
  FOREACHridge_(newfacet->ridges) {  /* 遍历newfacet的所有ridges */
    neighbor= otherfacet_(ridge, newfacet);  /* 获得ridge的另一个相邻facet赋值给neighbor */
    if (neighbor->visitid == samevisitid)
      SETref_(ridge)= NULL; /* ridge被释放 */
  }
  qh_setcompact(qh, newfacet->ridges);  /* 压缩newfacet的ridges集合 */

  trace4((qh, qh->ferr, 4034, "qh_mergecycle_ridges: add ridges to newfacet\n"));
  FORALLsame_cycle_(samecycle) {  /* 对于samecycle中的每个facet */
    FOREACHridge_(same->ridges) {  /* 遍历same的所有ridges */
      if (ridge->top == same) {  /* 如果ridge的top为same */
        ridge->top= newfacet;  /* 将ridge的top设置为newfacet */
        neighbor= ridge->bottom;  /* 设置neighbor为ridge的bottom */
      }else if (ridge->bottom == same) {  /* 如果ridge的bottom为same */
        ridge->bottom= newfacet;  /* 将ridge的bottom设置为newfacet */
        neighbor= ridge->top;  /* 设置neighbor为ridge的top */
      }else if (ridge->top == newfacet || ridge->bottom == newfacet) {  /* 如果ridge的top或bottom为newfacet */
        qh_setappend(qh, &newfacet->ridges, ridge);  /* 将ridge追加到newfacet的ridges集合中 */
        numold++;  /* numold加1，已经由qh_mergecycle_neighbors设置 */
        continue;  /* 继续下一轮循环 */
      }else {  /* 否则出现错误情况 */
        qh_fprintf(qh, qh->ferr, 6098, "qhull internal error (qh_mergecycle_ridges): bad ridge r%d\n", ridge->id);
        qh_errexit(qh, qh_ERRqhull, NULL, ridge);  /* 输出错误信息并退出程序 */
      }
      if (neighbor == newfacet) {  /* 如果neighbor是newfacet */
        if (qh->traceridge == ridge)
          qh->traceridge= NULL;  /* 如果traceridge等于ridge，则将traceridge置为NULL */
        qh_setfree(qh, &(ridge->vertices));  /* 释放ridge的vertices集合 */
        qh_memfree_(qh, ridge, (int)sizeof(ridgeT), freelistp);  /* 释放ridge的内存 */
        numold++;  /* numold加1 */
      }else if (neighbor->visitid == samevisitid) {  /* 如果neighbor的visitid等于samevisitid */
        qh_setdel(neighbor->ridges, ridge);  /* 从neighbor的ridges集合中删除ridge */
        if (qh->traceridge == ridge)
          qh->traceridge= NULL;  /* 如果traceridge等于ridge，则将traceridge置为NULL */
        qh_setfree(qh, &(ridge->vertices));  /* 释放ridge的vertices集合 */
        qh_memfree_(qh, ridge, (int)sizeof(ridgeT), freelistp);  /* 释放ridge的内存 */
        numold++;  /* numold加1 */
      }else {  /* 否则 */
        qh_setappend(qh, &newfacet->ridges, ridge);  /* 将ridge追加到newfacet的ridges集合中 */
        numold++;  /* numold加1 */
      }
    }
    if (same->ridges)
      qh_settruncate(qh, same->ridges, 0);  /* 清空same的ridges集合 */
    if (!same->simplicial)
      continue;  /* 如果same不是simplicial，则继续下一轮循环 */
    FOREACHneighbor_i_(qh, same) {  /* 对于same的每个neighbor */
      if (neighbor->visitid != samevisitid && neighbor->simplicial) {  /* 如果neighbor的visitid不等于samevisitid并且neighbor是simplicial的 */
        ridge= qh_newridge(qh);  /* 创建一个新的ridge */
        ridge->vertices= qh_setnew_delnthsorted(qh, same->vertices, qh->hull_dim,
                                                          neighbor_i, 0);  /* 设置ridge的vertices集合 */
        toporient= (boolT)(same->toporient ^ (neighbor_i & 0x1));  /* 计算toporient */
        if (toporient) {
          ridge->top= newfacet;  /* 设置ridge的top为newfacet */
          ridge->bottom= neighbor;  /* 设置ridge的bottom为neighbor */
          ridge->simplicialbot= True;  /* 设置simplicialbot为True */
        }else {
          ridge->top= neighbor;  /* 设置ridge的top为neighbor */
          ridge->bottom= newfacet;  /* 设置ridge的bottom为newfacet */
          ridge->simplicialtop= True;  /* 设置simplicialtop为True */
        }
        qh_setappend(qh, &(newfacet->ridges), ridge);  /* 将ridge追加到newfacet的ridges集合中 */
        qh_setappend(qh, &(neighbor->ridges), ridge);  /* 将ridge追加到neighbor的ridges集合中 */
        if (qh->ridge_id == qh->traceridge_id)
          qh->traceridge= ridge;  /* 如果ridge_id等于traceridge_id，则设置traceridge为ridge */
        numnew++;  /* numnew加1 */
      }
      // 这里缺少结束的 '}'，应在实际运行中修复。
    }
  }
}
    }
  }



// 这段代码可能是C或C++代码，这里是一个函数或者循环的结尾
// 确保大括号的匹配和作用域正确闭合
// 如果在函数中，可能是函数的结尾
// 如果在循环或条件语句中，可能是循环或条件语句块的结尾
// 这里缺少上下文，难以准确描述具体的作用

  trace2((qh, qh->ferr, 2033, "qh_mergecycle_ridges: found %d old ridges and %d new ones\n",
             numold, numnew));



// 调用名为trace2的函数，传递了四个参数
// 参数分别是：qh, qh->ferr, 2033, 以及一个格式化字符串
// 格式化字符串中包含了两个%d占位符，分别用numold和numnew填充
// 该函数调用的具体功能和实现需要查看trace2函数的定义或者文档
// 该行代码用于打印或记录一条跟踪信息或调试信息，描述了发现旧的和新的“ridge”数量
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_vneighbors">-</a>

  qh_mergecycle_vneighbors(qh, samecycle, newfacet )
    create vertex neighbors for newfacet from vertices of facets in samecycle
    samecycle marked with visitid == qh.visit_id - 1

  returns:
    newfacet vertices with updated neighbors
    marks newfacet with qh.visit_id-1
    deletes vertices that are merged away
    sets delridge on all vertices (faster here than in mergecycle_ridges)

  see:
    qh_mergevertex_neighbors()

  design:
    for each vertex of samecycle facet
      set vertex->delridge
      delete samecycle facets from vertex neighbors
      append newfacet to vertex neighbors
      if vertex only in newfacet
        delete it from newfacet
        add it to qh.del_vertices for later deletion
*/
void qh_mergecycle_vneighbors(qhT *qh, facetT *samecycle, facetT *newfacet) {
  facetT *neighbor, **neighborp;
  unsigned int mergeid;
  vertexT *vertex, **vertexp, *apex;
  setT *vertices;

  trace4((qh, qh->ferr, 4035, "qh_mergecycle_vneighbors: update vertex neighbors for newfacet\n"));
  mergeid= qh->visit_id - 1;  // 计算当前访问 ID 的前一个值，用于标记合并状态
  newfacet->visitid= mergeid;  // 将新的 facet 标记为当前合并状态的 visitid
  vertices= qh_basevertices(qh, samecycle); /* temp */  // 获取 samecycle 所有顶点的集合
  apex= SETfirstt_(samecycle->vertices, vertexT);  // 获取 samecycle 的第一个顶点作为顶点 apex
  qh_setappend(qh, &vertices, apex);  // 将顶点 apex 添加到 vertices 集合中
  FOREACHvertex_(vertices) {
    vertex->delridge= True;  // 设置顶点的 delridge 标志为 True
    FOREACHneighbor_(vertex) {
      if (neighbor->visitid == mergeid)
        SETref_(neighbor)= NULL;  // 删除与当前顶点相关的 visitid 为 mergeid 的邻居
    }
    qh_setcompact(qh, vertex->neighbors);  // 紧缩顶点的邻居集合
    qh_setappend(qh, &vertex->neighbors, newfacet);  // 将新的 facet 添加到顶点的邻居集合中
    if (!SETsecond_(vertex->neighbors)) {  // 如果顶点仅存在于新的 facet 中
      zinc_(Zcyclevertex);  // 增加计数器 Zcyclevertex
      trace2((qh, qh->ferr, 2034, "qh_mergecycle_vneighbors: deleted v%d when merging cycle f%d into f%d\n",
        vertex->id, samecycle->id, newfacet->id));  // 记录删除顶点的日志信息
      qh_setdelsorted(newfacet->vertices, vertex);  // 从新的 facet 的顶点集合中删除顶点
      vertex->deleted= True;  // 标记顶点为已删除
      qh_setappend(qh, &qh->del_vertices, vertex);  // 将顶点添加到 qh.del_vertices 中以便稍后删除
    }
  }
  qh_settempfree(qh, &vertices);  // 释放临时顶点集合
  trace3((qh, qh->ferr, 3005, "qh_mergecycle_vneighbors: merged vertices from cycle f%d into f%d\n",
             samecycle->id, newfacet->id));  // 记录合并顶点的日志信息
} /* mergecycle_vneighbors */
    # 将 facet2 包含 facet1 的顶点、邻居和棱边
    # 将 facet2 移动到 qh.facet_list 的末尾
    # 将 facet2 设置为新的 facet
    # 设置 facet2->newmerge 集合
    # 清除 facet2->center（除非合并到一个大 facet）
    # 清除 facet2->tested 和 ridge->tested 为 facet1
    
    # 将 facet1 添加到 visible_list 中，以便稍后删除和分区
    # facet1->f.replace == facet2
    
    # 如果邻近的 facet 重复或退化，将其添加到 facet_mergeset
    
    # 注：
    # 在完成后，检查 facet1 和 facet2 是否具有退化或冗余的邻居和 dupridge
    # mindist/maxdist 可能为 NULL（仅当两者都为 NULL 时）
    # 如果 fmax_(maxdist,-mindist) > TRACEdist，则追踪合并
    
    # 参见：
    # qh_mergecycle()
    
    # 设计：
    # 追踪合并并检查是否为退化单纯形
    # 为两个 facets 创建棱边
    # 更新 qh.max_outside、qh.max_vertex、qh.min_vertex
    # 更新 facet2->maxoutside 并保持中心
    # 更新 facet2->nummerge
    # 更新 facet2 的测试标志
    # 如果 facet1 是单纯形
    #   合并 facet1 到 facet2
    # 否则
    #   合并 facet1 的邻居到 facet2
    #   合并 facet1 的棱边到 facet2
    #   合并 facet1 的顶点到 facet2
    #   合并 facet1 的顶点邻居到 facet2
    # 将 facet2 的顶点添加到 qh.new_vertexlist
    # 将 facet2 移动到 qh.newfacet_list 的末尾
    # 除非 MRGcoplanarhorizon
    #   检查 facet2 是否有冗余的邻居
    #   检查 facet1 是否有退化的邻居
    #   可能检查是否有重复的棱边（'Q15'）
    # 将 facet1 移动到 qh.visible_list 以便稍后删除
`
/*
 * 函数定义，合并两个 facet，参数包括 qh（qhull 的数据结构指针）、facet1 和 facet2（需要合并的两个 facet）、mergetype（合并类型）、mindist 和 maxdist（合并的最小和最大距离）、mergeapex（是否合并顶点）。
 */
void qh_mergefacet(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype, realT *mindist, realT *maxdist, boolT mergeapex) {
  boolT traceonce= False;  // 标记是否只追踪一次
  vertexT *vertex, **vertexp;  // 定义 vertex 类型的指针 vertex 和 vertexp
  realT mintwisted, vertexdist;  // 定义 real 类型的变量 mintwisted 和 vertexdist
  realT onemerge;  // 定义 real 类型的变量 onemerge
  int tracerestore=0, nummerge;  // 定义 int 类型的变量 tracerestore 和 nummerge
  const char *mergename;  // 定义常量字符指针 mergename

  // 根据 mergetype 确定 mergename，mergetype 在 mergetypes 数组中的位置
  if(mergetype > 0 && mergetype <= sizeof(mergetypes))
    mergename= mergetypes[mergetype];
  else
    mergename= mergetypes[MRGnone];

  // 如果任一 facet 为 tricoplanar，则检查 qh->TRInormals 是否为假
  if (facet1->tricoplanar || facet2->tricoplanar) {
    if (!qh->TRInormals) {
      // 如果 qh->TRInormals 为假，打印错误信息并退出
      qh_fprintf(qh, qh->ferr, 6226, "qhull internal error (qh_mergefacet): merge f%d into f%d for mergetype %d (%s) does not work for tricoplanar facets.  Use option 'Q11'\n",
        facet1->id, facet2->id, mergetype, mergename);
      qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
    }
    // 如果 facet2 为 tricoplanar，将其设置为假，并将 keepcentrum 设置为假
    if (facet2->tricoplanar) {
      facet2->tricoplanar= False;
      facet2->keepcentrum= False;
    }
  }

  zzinc_(Ztotmerge);  // 增加 Ztotmerge 的计数

  // 如果需要报告合并并且合并频率大于阈值，则调用 qh_tracemerging 函数
  if (qh->REPORTfreq2 && qh->POSTmerging) {
    if (zzval_(Ztotmerge) > qh->mergereport + qh->REPORTfreq2)
      qh_tracemerging(qh);
  }

#ifndef qh_NOtrace
  // 如果 qh->build_cnt 大于等于 qh->RERUN，并且 mindist 和 maxdist 超过 qh->TRACEdist，则设置 tracerestore 和 qh->IStracing 并打印追踪信息
  if (qh->build_cnt >= qh->RERUN) {
    if (mindist && (-*mindist > qh->TRACEdist || *maxdist > qh->TRACEdist)) {
      tracerestore= 0;
      qh->IStracing= qh->TRACElevel;
      traceonce= True;
      qh_fprintf(qh, qh->ferr, 8075, "qh_mergefacet: ========= trace wide merge #%d(%2.2g) for f%d into f%d for mergetype %d (%s), last point was p%d\n",
          zzval_(Ztotmerge), fmax_(-*mindist, *maxdist), facet1->id, facet2->id, mergetype, mergename, qh->furthest_id);
    }else if (facet1 == qh->tracefacet || facet2 == qh->tracefacet) {
      tracerestore= qh->IStracing;
      qh->IStracing= 4;
      traceonce= True;
      qh_fprintf(qh, qh->ferr, 8076, "qh_mergefacet: ========= trace merge #%d for f%d into f%d for mergetype %d (%s), furthest is p%d\n",
                 zzval_(Ztotmerge), facet1->id, facet2->id, mergetype, mergename, qh->furthest_id);
    }
  }
  // 如果 qh->IStracing 大于等于 2，打印合并的详细信息
  if (qh->IStracing >= 2) {
    realT mergemin= -2;
    realT mergemax= -2;

    if (mindist) {
      mergemin= *mindist;
      mergemax= *maxdist;
    }
    qh_fprintf(qh, qh->ferr, 2081, "qh_mergefacet: #%d merge f%d into f%d for merge for mergetype %d (%s), mindist= %2.2g, maxdist= %2.2g, max_outside %2.2g\n",
    zzval_(Ztotmerge), facet1->id, facet2->id, mergetype, mergename, mergemin, mergemax, qh->max_outside);
  }
#endif /* !qh_NOtrace */

  // 如果不允许宽合并并且 mindist 不为 NULL，则计算 mintwisted，并更新 facet1 和 facet2 的 maxoutside
  if(!qh->ALLOWwide && mindist) {
    mintwisted= qh_WIDEmaxoutside * qh->ONEmerge;  /* same as qh_merge_twisted and qh_check_maxout (poly2) */
    maximize_(mintwisted, facet1->maxoutside);
    maximize_(mintwisted, facet2->maxoutside);
    // 检查最大距离是否大于最小扭曲度或者最小距离的负值是否大于最小扭曲度
    if (*maxdist > mintwisted || -*mindist > mintwisted) {
      // 计算 facet1 的顶点集合的最佳距离
      vertexdist = qh_vertex_bestdist(qh, facet1->vertices);
      // 计算一个合并的阈值
      onemerge = qh->ONEmerge + qh->DISTround;
      // 如果顶点距离大于最小扭曲度
      if (vertexdist > mintwisted) {
        // 打印宽合并的错误信息，包括 facet1 和 facet2 的标识符、合并类型、最大距离、最小距离等信息
        qh_fprintf(qh, qh->ferr, 6347, "qhull precision error (qh_mergefacet): wide merge for facet f%d into f%d for mergetype %d (%s).  maxdist %2.2g (%.1fx) mindist %2.2g (%.1fx) vertexdist %2.2g  Allow with 'Q12' (allow-wide)\n",
          facet1->id, facet2->id, mergetype, mergename, *maxdist, *maxdist/onemerge, *mindist, -*mindist/onemerge, vertexdist);
      } else {
        // 打印收紧合并的错误信息，包括 facet1 和 facet2 的标识符、合并类型、最大距离、最小距离等信息
        qh_fprintf(qh, qh->ferr, 6348, "qhull precision error (qh_mergefacet): wide merge for pinched facet f%d into f%d for mergetype %d (%s).  maxdist %2.2g (%.1fx) mindist %2.2g (%.1fx) vertexdist %2.2g  Allow with 'Q12' (allow-wide)\n",
          facet1->id, facet2->id, mergetype, mergename, *maxdist, *maxdist/onemerge, *mindist, -*mindist/onemerge, vertexdist);
      }
      // 引发 qhull 的宽合并错误退出
      qh_errexit2(qh, qh_ERRwide, facet1, facet2);
    }
  }
  // 如果 facet1 和 facet2 是同一个或者任意一个是可见的，则引发 qhull 的内部错误退出
  if (facet1 == facet2 || facet1->visible || facet2->visible) {
    qh_fprintf(qh, qh->ferr, 6099, "qhull internal error (qh_mergefacet): either f%d and f%d are the same or one is a visible facet, mergetype %d (%s)\n",
             facet1->id, facet2->id, mergetype, mergename);
    qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
  }
  // 如果当前剩余的非可见面的数量不超过凸包维度加一，则引发 qhull 的拓扑错误退出
  if (qh->num_facets - qh->num_visible <= qh->hull_dim + 1) {
    // 打印拓扑错误信息，指出剩余面数过少的原因
    qh_fprintf(qh, qh->ferr, 6227, "qhull topology error: Only %d facets remain.  The input is too degenerate or the convexity constraints are too strong.\n", 
          qh->hull_dim+1);
    // 如果维度大于等于5且未使用精确合并选项，则建议使用 'Qx' 选项以避免此问题
    if (qh->hull_dim >= 5 && !qh->MERGEexact)
      qh_fprintf(qh, qh->ferr, 8079, "    Option 'Qx' may avoid this problem.\n");
    // 引发 qhull 的拓扑错误退出
    qh_errexit(qh, qh_ERRtopology, NULL, NULL);
  }
  // 如果未初始化顶点邻居列表，则初始化之
  if (!qh->VERTEXneighbors)
    qh_vertexneighbors(qh);
  // 对 facet1 和 facet2 分别调用 qh_makeridges 函数
  qh_makeridges(qh, facet1);
  qh_makeridges(qh, facet2);
  // 如果距离存在，则更新 qh->max_outside 和 qh->max_vertex
  if (mindist) {
    maximize_(qh->max_outside, *maxdist);
    maximize_(qh->max_vertex, *maxdist);
#if qh_MAXoutside
    maximize_(facet2->maxoutside, *maxdist);
#endif
    // 如果定义了 qh_MAXoutside，则更新 facet2->maxoutside 的值为 maxdist 中的较大值
    minimize_(qh->min_vertex, *mindist);
    // 更新 qh->min_vertex 的值为 mindist 中的较小值
    if (!facet2->keepcentrum
    && (*maxdist > qh->WIDEfacet || *mindist < -qh->WIDEfacet)) {
      // 如果 facet2->keepcentrum 为假且 maxdist 大于 qh->WIDEfacet 或 mindist 小于 -qh->WIDEfacet
      facet2->keepcentrum= True;
      // 设置 facet2->keepcentrum 为真
      zinc_(Zwidefacet);
      // 增加 Zwidefacet 的计数
    }
  }
  nummerge= facet1->nummerge + facet2->nummerge + 1;
  // 计算 nummerge 为 facet1->nummerge、facet2->nummerge 和 1 的和
  if (nummerge >= qh_MAXnummerge)
    facet2->nummerge= qh_MAXnummerge;
  else
    facet2->nummerge= (short unsigned int)nummerge; /* limited to 9 bits by qh_MAXnummerge, -Wconversion */
  // 如果 nummerge 大于等于 qh_MAXnummerge，则设置 facet2->nummerge 为 qh_MAXnummerge，否则设置为 nummerge（类型转换为 short unsigned int）
  facet2->newmerge= True;
  // 设置 facet2->newmerge 为真
  facet2->dupridge= False;
  // 设置 facet2->dupridge 为假
  qh_updatetested(qh, facet1, facet2);
  // 更新测试状态，关联 facet1 和 facet2
  if (qh->hull_dim > 2 && qh_setsize(qh, facet1->vertices) == qh->hull_dim)
    // 如果凸包维度大于2并且 facet1 的顶点数等于凸包维度
    qh_mergesimplex(qh, facet1, facet2, mergeapex);
    // 合并简单形状（simplex）
  else {
    qh->vertex_visit++;
    // 增加顶点访问计数
    FOREACHvertex_(facet2->vertices)
      vertex->visitid= qh->vertex_visit;
    // 遍历 facet2 的顶点集合，设置顶点的 visitid 为当前的顶点访问计数
    if (qh->hull_dim == 2)
      qh_mergefacet2d(qh, facet1, facet2);
      // 在二维情况下，合并 facet1 和 facet2 的邻居和顶点
    else {
      qh_mergeneighbors(qh, facet1, facet2);
      // 合并邻居
      qh_mergevertices(qh, facet1->vertices, &facet2->vertices);
      // 合并顶点集合
    }
    qh_mergeridges(qh, facet1, facet2);
    // 合并边缘
    qh_mergevertex_neighbors(qh, facet1, facet2);
    // 合并顶点的邻居
    if (!facet2->newfacet)
      qh_newvertices(qh, facet2->vertices);
      // 如果 facet2 不是新的面，添加新顶点
  }
  if (facet2->coplanarhorizon) {
    zinc_(Zmergeintocoplanar);
    // 增加 Zmergeintocoplanar 的计数
  }else if (!facet2->newfacet) {
    zinc_(Zmergeintohorizon);
    // 增加 Zmergeintohorizon 的计数
  }else if (!facet1->newfacet && facet2->newfacet) {
    zinc_(Zmergehorizon);
    // 增加 Zmergehorizon 的计数
  }else {
    zinc_(Zmergenew);
    // 增加 Zmergenew 的计数
  }
  qh_removefacet(qh, facet2);  /* append as a newfacet to end of qh->facet_list */
  // 从几何处理对象中移除 facet2，并将其作为新的面附加到 qh->facet_list 的末尾
  qh_appendfacet(qh, facet2);
  // 将 facet2 附加到几何处理对象的面列表中
  facet2->newfacet= True;
  // 设置 facet2->newfacet 为真
  facet2->tested= False;
  // 设置 facet2->tested 为假
  qh_tracemerge(qh, facet1, facet2, mergetype);
  // 跟踪合并过程
  if (traceonce) {
    qh_fprintf(qh, qh->ferr, 8080, "qh_mergefacet: end of wide tracing\n");
    // 输出跟踪信息到 ferr 流，指示宽跟踪的结束
    qh->IStracing= tracerestore;
    // 恢复跟踪状态
  }
  if (mergetype != MRGcoplanarhorizon) {
    trace3((qh, qh->ferr, 3076, "qh_mergefacet: check f%d and f%d for redundant and degenerate neighbors\n",
        facet1->id, facet2->id));
    // 如果合并类型不是 MRGcoplanarhorizon，输出检查冗余和退化邻居的跟踪信息
    qh_test_redundant_neighbors(qh, facet2);
    // 检查冗余邻居
    qh_test_degen_neighbors(qh, facet1);
    // 检查退化邻居，先于 qh_test_redundant_neighbors，因为 MRGdegen 比 MRGredundant 更难
    qh_degen_redundant_facet(qh, facet2);
    // 处理退化和冗余面
    qh_maybe_duplicateridges(qh, facet2);
    // 可能复制边缘
  }
  qh_willdelete(qh, facet1, facet2);
  // 标记将要删除的面
} /* mergefacet */
// 结束合并面的函数
    # qh_mergefacet() 保留非单纯结构
    # 这些在二维中不是必需的，但后续的程序可能会用到它们
    
    # 保留 qh.vertex_visit 以供 qh_mergevertex_neighbors() 使用
    
    # 设计思路：
    # 获取顶点和邻居
    # 确定新的顶点和邻居
    # 设置新的顶点和邻居，并调整方向
    # 如果需要，为新邻居创建边界
/*
void qh_mergefacet2d(qhT *qh, facetT *facet1, facetT *facet2) {
  vertexT *vertex1A, *vertex1B, *vertex2A, *vertex2B, *vertexA, *vertexB;
  facetT *neighbor1A, *neighbor1B, *neighbor2A, *neighbor2B, *neighborA, *neighborB;

  // 获取 facet1 和 facet2 的顶点和邻居信息
  vertex1A = SETfirstt_(facet1->vertices, vertexT);
  vertex1B = SETsecondt_(facet1->vertices, vertexT);
  vertex2A = SETfirstt_(facet2->vertices, vertexT);
  vertex2B = SETsecondt_(facet2->vertices, vertexT);
  neighbor1A = SETfirstt_(facet1->neighbors, facetT);
  neighbor1B = SETsecondt_(facet1->neighbors, facetT);
  neighbor2A = SETfirstt_(facet2->neighbors, facetT);
  neighbor2B = SETsecondt_(facet2->neighbors, facetT);

  // 根据顶点情况选择合适的顶点和邻居关系进行合并
  if (vertex1A == vertex2A) {
    vertexA = vertex1B;
    vertexB = vertex2B;
    neighborA = neighbor2A;
    neighborB = neighbor1A;
  } else if (vertex1A == vertex2B) {
    vertexA = vertex1B;
    vertexB = vertex2A;
    neighborA = neighbor2B;
    neighborB = neighbor1A;
  } else if (vertex1B == vertex2A) {
    vertexA = vertex1A;
    vertexB = vertex2B;
    neighborA = neighbor2A;
    neighborB = neighbor1B;
  } else { /* 1B == 2B */
    vertexA = vertex1A;
    vertexB = vertex2A;
    neighborA = neighbor2B;
    neighborB = neighbor1B;
  }

  // 根据顶点 id 大小确定顶点和邻居的顺序，并更新 facet2 的顶点和邻居关系
  if (vertexA->id > vertexB->id) {
    SETfirst_(facet2->vertices) = vertexA;
    SETsecond_(facet2->vertices) = vertexB;
    if (vertexB == vertex2A)
      facet2->toporient = !facet2->toporient;
    SETfirst_(facet2->neighbors) = neighborA;
    SETsecond_(facet2->neighbors) = neighborB;
  } else {
    SETfirst_(facet2->vertices) = vertexB;
    SETsecond_(facet2->vertices) = vertexA;
    if (vertexB == vertex2B)
      facet2->toporient = !facet2->toporient;
    SETfirst_(facet2->neighbors) = neighborB;
    SETsecond_(facet2->neighbors) = neighborA;
  }

  // 更新邻居关系，将 facet1 的邻居替换为 facet2
  qh_setreplace(qh, neighborB->neighbors, facet1, facet2);

  // 打印调试信息，描述合并的顶点和邻居关系
  trace4((qh, qh->ferr, 4036, "qh_mergefacet2d: merged v%d and neighbor f%d of f%d into f%d\n",
       vertexA->id, neighborB->id, facet1->id, facet2->id));
} /* mergefacet2d */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergeneighbors">-</a>

  qh_mergeneighbors(qh, facet1, facet2 )
    merges the neighbors of facet1 into facet2

  notes:
    only called by qh_mergefacet
    qh.hull_dim >= 3
    see qh_mergecycle_neighbors

  design:
    for each neighbor of facet1
      if neighbor is also a neighbor of facet2
        if neighbor is simplicial
          make ridges for later deletion as a degenerate facet
        update its neighbor set
      else
        move the neighbor relation to facet2
    remove the neighbor relation for facet1 and facet2
*/
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergineighbors">-</a>

  qh_mergeneighbors(qh, facet1, facet2 )
    merge the neighbor set of facet1 into facet2

  returns:
    updates neighbors and ridges of facet2
    updates neighbors of facet1's neighbors to use facet2

  see:
    qh_makeridges()

  design:
    increment visit_id to mark current visit
    mark all neighbors of facet2 with current visit_id
    for each neighbor of facet1
      if neighbor is marked with current visit_id
        if neighbor is simplicial, create ridges if needed
        adjust neighbor sets to include or replace facet2
      else if neighbor is not facet2
        add neighbor to facet2's neighbor set
        update neighbor's neighbor set to replace facet1 with facet2
    adjust neighbor sets of facet1 and facet2 to remove each other
*/
void qh_mergeneighbors(qhT *qh, facetT *facet1, facetT *facet2) {
  facetT *neighbor, **neighborp;

  trace4((qh, qh->ferr, 4037, "qh_mergeneighbors: merge neighbors of f%d and f%d\n",
          facet1->id, facet2->id));
  qh->visit_id++;
  FOREACHneighbor_(facet2) {
    neighbor->visitid= qh->visit_id;
  }
  FOREACHneighbor_(facet1) {
    if (neighbor->visitid == qh->visit_id) {
      if (neighbor->simplicial)    /* is degen, needs ridges */
        qh_makeridges(qh, neighbor);
      if (SETfirstt_(neighbor->neighbors, facetT) != facet1) /*keep newfacet->horizon*/
        qh_setdel(neighbor->neighbors, facet1);
      else {
        qh_setdel(neighbor->neighbors, facet2);
        qh_setreplace(qh, neighbor->neighbors, facet1, facet2);
      }
    }else if (neighbor != facet2) {
      qh_setappend(qh, &(facet2->neighbors), neighbor);
      qh_setreplace(qh, neighbor->neighbors, facet1, facet2);
    }
  }
  qh_setdel(facet1->neighbors, facet2);  /* here for makeridges */
  qh_setdel(facet2->neighbors, facet1);
} /* mergeneighbors */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergeridges">-</a>

  qh_mergeridges(qh, facet1, facet2 )
    merges the ridge set of facet1 into facet2

  returns:
    may delete all ridges for a vertex
    sets vertex->delridge on deleted ridges

  see:
    qh_mergecycle_ridges()

  design:
    delete ridges between facet1 and facet2
      mark (delridge) vertices on these ridges for later testing
    for each remaining ridge
      rename facet1 to facet2
*/
void qh_mergeridges(qhT *qh, facetT *facet1, facetT *facet2) {
  ridgeT *ridge, **ridgep;

  trace4((qh, qh->ferr, 4038, "qh_mergeridges: merge ridges of f%d into f%d\n",
          facet1->id, facet2->id));
  FOREACHridge_(facet2->ridges) {
    if ((ridge->top == facet1) || (ridge->bottom == facet1)) {
      /* ridge.nonconvex is irrelevant due to merge */
      qh_delridge_merge(qh, ridge);  /* expensive in high-d, could rebuild */
      ridgep--; /* deleted this ridge, repeat with next ridge*/
    }
  }
  FOREACHridge_(facet1->ridges) {
    if (ridge->top == facet1) {
      ridge->top= facet2;
      ridge->simplicialtop= False;
    }else { /* ridge.bottom is facet1 */
      ridge->bottom= facet2;
      ridge->simplicialbot= False;
    }
    qh_setappend(qh, &(facet2->ridges), ridge);
  }
} /* mergeridges */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergesimplex">-</a>

  qh_mergesimplex(qh, facet1, facet2, mergeapex )
    merge simplicial facet1 into facet2
    mergeapex==qh_MERGEapex if merging samecycle into horizon facet
      vertex id is latest (most recently created)
    facet1 may be contained in facet2
    ridges exist for both facets

  returns:
    facet2 with updated vertices, ridges, neighbors
    updated neighbors for facet1's vertices
    facet1 not deleted
    sets vertex->delridge on deleted ridges

  notes:
    handles non-simplicial neighbors
*/
    # 这段代码是处理一种特殊情况，因为这是最常见的合并操作，通常从qh_mergefacet()调用。

  design:
    # 设计思路：

    # 如果 qh_MERGEapex 为真
    if qh_MERGEapex:
      # 如果需要，将 facet2 的顶点添加到 qh.new_vertexlist 中
      add vertices of facet2 to qh.new_vertexlist if necessary
      # 将顶点 apex 添加到 facet2 中

    # 如果 qh_MERGEapex 不为真
    else:
      # 对于 facet1 和 facet2 之间的每条 ridge
      for each ridge between facet1 and facet2
        # 将顶点标记为被删除的 ridge
        set vertex->delridge
      # 确定 facet1 的顶点（即将被合并的顶点）
      # 除非 apex 已经在 facet2 中
      unless apex already in facet2
        # 将 apex 插入到 facet2 的顶点中
        insert apex into vertices for facet2
      # 如果需要，将 facet2 的顶点添加到 qh.new_vertexlist 中
      add vertices of facet2 to qh.new_vertexlist if necessary
      # 如果需要，将 apex 添加到 qh.new_vertexlist 中
      add apex to qh.new_vertexlist if necessary
      # 对于 facet1 的每个顶点
      for each vertex of facet1
        # 如果有 apex
          # 在其顶点邻居中将 facet1 重命名为 facet2
        if apex
          # 否则
        else
          # 从顶点邻居中删除 facet1
          # 如果只存在于 facet2 中
            # 将顶点添加到 qh.del_vertices 中以便稍后删除
      # 对于 facet1 的每个 ridge
      for each ridge of facet1
        # 删除 facet1 和 facet2 之间的 ridges
        # 重命名 facet 为 facet2 后，将其他的 ridge 追加到 facet2 中
/*
void qh_mergesimplex(qhT *qh, facetT *facet1, facetT *facet2, boolT mergeapex) {
  vertexT *vertex, **vertexp, *opposite;
  ridgeT *ridge, **ridgep;
  boolT isnew= False;
  facetT *neighbor, **neighborp, *otherfacet;

  if (mergeapex) {
    opposite= SETfirstt_(facet1->vertices, vertexT); /* 获取位于 facet1 的顶点集合中的第一个顶点 */
    trace4((qh, qh->ferr, 4086, "qh_mergesimplex: merge apex v%d of f%d into facet f%d\n",
      opposite->id, facet1->id, facet2->id));
    if (!facet2->newfacet)
      qh_newvertices(qh, facet2->vertices);  /* 对 facet2 中的顶点集合执行新顶点标记操作 */
    if (SETfirstt_(facet2->vertices, vertexT) != opposite) {
      qh_setaddnth(qh, &facet2->vertices, 0, opposite); /* 在 facet2 的顶点集合中添加 opposite 顶点 */
      isnew= True; /* 标记操作是否添加了新顶点 */
    }
  }else {
    zinc_(Zmergesimplex); /* 计数器递增，用于统计简单形的合并操作次数 */
    FOREACHvertex_(facet1->vertices)
      vertex->seen= False; /* 将 facet1 的所有顶点标记为未见过 */
    FOREACHridge_(facet1->ridges) {
      if (otherfacet_(ridge, facet1) == facet2) {
        FOREACHvertex_(ridge->vertices) {
          vertex->seen= True; /* 标记 ridge 相关顶点为已见过 */
          vertex->delridge= True; /* 标记顶点为删除 ridge 相关 */
        }
        break; /* 找到相应 ridge 后中断循环 */
      }
    }
    FOREACHvertex_(facet1->vertices) {
      if (!vertex->seen)
        break;  /* 必须发生的情况，确保顶点被遍历 */
    }
    opposite= vertex; /* 找到未标记的顶点作为 opposite */
    trace4((qh, qh->ferr, 4039, "qh_mergesimplex: merge opposite v%d of f%d into facet f%d\n",
          opposite->id, facet1->id, facet2->id));
    isnew= qh_addfacetvertex(qh, facet2, opposite); /* 向 facet2 添加 opposite 顶点 */
    if (!facet2->newfacet)
      qh_newvertices(qh, facet2->vertices); /* 对 facet2 中的顶点集合执行新顶点标记操作 */
    else if (!opposite->newfacet) {
      qh_removevertex(qh, opposite); /* 从凸包中删除 opposite 顶点 */
      qh_appendvertex(qh, opposite); /* 将 opposite 顶点追加到凸包中 */
    }
  }
  trace4((qh, qh->ferr, 4040, "qh_mergesimplex: update vertex neighbors of f%d\n",
          facet1->id));
  FOREACHvertex_(facet1->vertices) {
    if (vertex == opposite && isnew)
      qh_setreplace(qh, vertex->neighbors, facet1, facet2); /* 替换顶点 vertex 的邻居中的 facet1 为 facet2 */
    else {
      qh_setdel(vertex->neighbors, facet1); /* 从 vertex 的邻居集合中删除 facet1 */
      if (!SETsecond_(vertex->neighbors))
        qh_mergevertex_del(qh, vertex, facet1, facet2); /* 合并顶点 vertex 的删除操作 */
    }
  }
  trace4((qh, qh->ferr, 4041, "qh_mergesimplex: merge ridges and neighbors of f%d into f%d\n",
          facet1->id, facet2->id));
  qh->visit_id++; /* 访问标识递增 */
  FOREACHneighbor_(facet2)
    neighbor->visitid= qh->visit_id; /* 设置 facet2 的邻居的 visitid 为当前的 visit_id */
  FOREACHridge_(facet1->ridges) {
    otherfacet= otherfacet_(ridge, facet1);
    if (otherfacet == facet2) {
      /* ridge.nonconvex is irrelevant due to merge */
      qh_delridge_merge(qh, ridge);  /* 删除 ridge 并合并相关操作 */
      ridgep--; /* 删除此 ridge，重复处理下一个 ridge */
      qh_setdel(facet2->neighbors, facet1); /* simplicial 面可能有重复的邻居，需要删除每一个 */
    }else if (otherfacet->dupridge && !qh_setin(otherfacet->neighbors, facet1)) {
      qh_fprintf(qh, qh->ferr, 6356, "qhull topology error (qh_mergesimplex): f%d is a dupridge of f%d, cannot merge f%d into f%d\n",
        facet1->id, otherfacet->id, facet1->id, facet2->id);
      qh_errexit2(qh, qh_ERRqhull, facet1, otherfacet); /* 错误处理，不应合并的情况 */
    }else {
      // 输出调试信息，记录简并操作详细信息
      trace4((qh, qh->ferr, 4059, "qh_mergesimplex: move r%d with f%d to f%d, new neighbor? %d, maybe horizon? %d\n",
        ridge->id, otherfacet->id, facet2->id, (otherfacet->visitid != qh->visit_id), (SETfirstt_(otherfacet->neighbors, facetT) == facet1)));
      // 将当前的边缘添加到 facet2 的边缘列表中
      qh_setappend(qh, &facet2->ridges, ridge);
      // 如果 otherfacet 没有被访问过
      if (otherfacet->visitid != qh->visit_id) {
        // 将 otherfacet 添加到 facet2 的邻居列表中
        qh_setappend(qh, &facet2->neighbors, otherfacet);
        // 替换 otherfacet 的邻居列表中的 facet1 为 facet2
        qh_setreplace(qh, otherfacet->neighbors, facet1, facet2);
        // 将 otherfacet 的访问标记设为当前访问标记
        otherfacet->visitid= qh->visit_id;
      }else {
        // 如果 otherfacet 是简并的，需要创建边缘
        if (otherfacet->simplicial)
          qh_makeridges(qh, otherfacet);
        // 如果 otherfacet 的邻居列表中的第一个元素是 facet1
        if (SETfirstt_(otherfacet->neighbors, facetT) == facet1) {
          // 从 otherfacet 的邻居列表中删除 facet2，将 facet1 替换为 facet2
          qh_setdel(otherfacet->neighbors, facet2);
          qh_setreplace(qh, otherfacet->neighbors, facet1, facet2);
        }else {
          // facet2 已经是 otherfacet 的邻居，根据 f.visitid
          // 从 otherfacet 的邻居列表中删除 facet1
          qh_setdel(otherfacet->neighbors, facet1);
        }
      }
      // 如果 ridge 的顶点是 facet1
      if (ridge->top == facet1) { /* wait until after qh_makeridges */
        // 将 ridge 的顶点设置为 facet2，标记 ridge 的顶点不是简单形式
        ridge->top= facet2;
        ridge->simplicialtop= False;
      }else {
        // 将 ridge 的底部设置为 facet2，标记 ridge 的底部不是简单形式
        ridge->bottom= facet2;
        ridge->simplicialbot= False;
      }
    }
  }
  // 输出调试信息，记录合并简单形式的详细信息
  trace3((qh, qh->ferr, 3006, "qh_mergesimplex: merged simplex f%d v%d into facet f%d\n",
          facet1->id, opposite->id, facet2->id));
/* mergesimplex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergevertex_del">-</a>

  qh_mergevertex_del(qh, vertex, facet1, facet2 )
    delete a vertex because of merging facet1 into facet2

  returns:
    deletes vertex from facet2
    adds vertex to qh.del_vertices for later deletion
*/
void qh_mergevertex_del(qhT *qh, vertexT *vertex, facetT *facet1, facetT *facet2) {

  zinc_(Zmergevertex);  // 增加 Zmergevertex 统计计数
  trace2((qh, qh->ferr, 2035, "qh_mergevertex_del: deleted v%d when merging f%d into f%d\n",
          vertex->id, facet1->id, facet2->id));  // 输出调试信息，记录删除的顶点和合并的两个面的 ID
  qh_setdelsorted(facet2->vertices, vertex);  // 从 facet2 的顶点集合中删除顶点 vertex
  vertex->deleted= True;  // 标记顶点 vertex 为已删除状态
  qh_setappend(qh, &qh->del_vertices, vertex);  // 将顶点 vertex 添加到 qh.del_vertices 中，以便稍后删除
} /* mergevertex_del */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergevertex_neighbors">-</a>

  qh_mergevertex_neighbors(qh, facet1, facet2 )
    merge the vertex neighbors of facet1 to facet2

  returns:
    if vertex is current qh.vertex_visit
      deletes facet1 from vertex->neighbors
    else
      renames facet1 to facet2 in vertex->neighbors
    deletes vertices if only one neighbor

  notes:
    assumes vertex neighbor sets are good
*/
void qh_mergevertex_neighbors(qhT *qh, facetT *facet1, facetT *facet2) {
  vertexT *vertex, **vertexp;

  trace4((qh, qh->ferr, 4042, "qh_mergevertex_neighbors: merge vertex neighborset for f%d into f%d\n",
          facet1->id, facet2->id));  // 输出调试信息，记录正在合并顶点邻居集合的两个面的 ID
  if (qh->tracevertex) {
    qh_fprintf(qh, qh->ferr, 8081, "qh_mergevertex_neighbors: of f%d into f%d at furthest p%d f0= %p\n",
               facet1->id, facet2->id, qh->furthest_id, qh->tracevertex->neighbors->e[0].p);
    qh_errprint(qh, "TRACE", NULL, NULL, NULL, qh->tracevertex);  // 输出调试信息，显示追踪顶点的状态
  }
  FOREACHvertex_(facet1->vertices) {
    if (vertex->visitid != qh->vertex_visit)
      qh_setreplace(qh, vertex->neighbors, facet1, facet2);  // 替换顶点 vertex 的邻居中的 facet1 为 facet2
    else {
      qh_setdel(vertex->neighbors, facet1);  // 从顶点 vertex 的邻居中删除 facet1
      if (!SETsecond_(vertex->neighbors))
        qh_mergevertex_del(qh, vertex, facet1, facet2);  // 如果顶点 vertex 的邻居只剩一个，则删除顶点
    }
  }
  if (qh->tracevertex)
    qh_errprint(qh, "TRACE", NULL, NULL, NULL, qh->tracevertex);  // 输出调试信息，显示追踪顶点的状态
} /* mergevertex_neighbors */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergevertices">-</a>

  qh_mergevertices(qh, vertices1, vertices2 )
    merges the vertex set of facet1 into facet2

  returns:
    replaces vertices2 with merged set
    preserves vertex_visit for qh_mergevertex_neighbors
    updates qh.newvertex_list

  design:
    create a merged set of both vertices (in inverse id order)
*/
void qh_mergevertices(qhT *qh, setT *vertices1, setT **vertices2) {
  int newsize= qh_setsize(qh, vertices1)+qh_setsize(qh, *vertices2) - qh->hull_dim + 1;
  setT *mergedvertices;
  vertexT *vertex, **vertexp, **vertex2= SETaddr_(*vertices2, vertexT);

  mergedvertices= qh_settemp(qh, newsize);  // 创建一个临时集合来存放合并后的顶点
  FOREACHvertex_(vertices1) {
    // 如果 vertex2 指针为空，或者 vertex 的 id 大于 (*vertex2) 指向的对象的 id
    if (!*vertex2 || vertex->id > (*vertex2)->id)
      // 将 vertex 添加到 mergedvertices 集合中
      qh_setappend(qh, &mergedvertices, vertex);
    else {
      // 当 vertex2 指针不为空且 (*vertex2) 指向的对象的 id 大于 vertex 的 id 时循环
      while (*vertex2 && (*vertex2)->id > vertex->id)
        // 将 (*vertex2) 指向的对象添加到 mergedvertices 集合中，然后递增 vertex2 指针
        qh_setappend(qh, &mergedvertices, *vertex2++);
      // 如果 vertex2 指针为空或者 (*vertex2) 指向的对象的 id 小于 vertex 的 id
      if (!*vertex2 || (*vertex2)->id < vertex->id)
        // 将 vertex 添加到 mergedvertices 集合中
        qh_setappend(qh, &mergedvertices, vertex);
      else
        // 将 (*vertex2) 指向的对象添加到 mergedvertices 集合中，然后递增 vertex2 指针
        qh_setappend(qh, &mergedvertices, *vertex2++);
    }
  }
  // 当 vertex2 指针不为空时，循环将所有 (*vertex2) 指向的对象添加到 mergedvertices 集合中
  while (*vertex2)
    qh_setappend(qh, &mergedvertices, *vertex2++);
  // 如果 newsize 小于 mergedvertices 集合的大小，则输出错误信息并终止程序
  if (newsize < qh_setsize(qh, mergedvertices)) {
    qh_fprintf(qh, qh->ferr, 6100, "qhull internal error (qh_mergevertices): facets did not share a ridge\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 释放 vertices2 集合占用的内存
  qh_setfree(qh, vertices2);
  // 将 mergedvertices 集合赋值给 vertices2 指针所指向的对象
  *vertices2= mergedvertices;
  // 弹出临时集合栈中的数据
  qh_settemppop(qh);
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="neighbor_intersections">-</a>

  qh_neighbor_intersections(qh, vertex )
    return intersection of all vertices in vertex->neighbors except for vertex

  returns:
    returns temporary set of vertices
    does not include vertex
    NULL if a neighbor is simplicial
    NULL if empty set

  notes:
    only called by qh_redundant_vertex for qh_reducevertices
      so f.vertices does not contain extraneous vertices that are not in f.ridges
    used for renaming vertices

  design:
    initialize the intersection set with vertices of the first two neighbors
    delete vertex from the intersection
    for each remaining neighbor
      intersect its vertex set with the intersection set
      return NULL if empty
    return the intersection set
*/
setT *qh_neighbor_intersections(qhT *qh, vertexT *vertex) {
  facetT *neighbor, **neighborp, *neighborA, *neighborB;
  setT *intersect;
  int neighbor_i, neighbor_n;

  // 检查每个邻居是否是单纯的，如果是则返回 NULL
  FOREACHneighbor_(vertex) {
    if (neighbor->simplicial)
      return NULL;
  }

  // 获取顶点的第一个和第二个邻居
  neighborA = SETfirstt_(vertex->neighbors, facetT);
  neighborB = SETsecondt_(vertex->neighbors, facetT);
  zinc_(Zintersectnum); // 计数器增加

  // 如果第一个邻居不存在，直接复制第一个邻居的顶点集合作为交集
  if (!neighborA)
    return NULL;

  // 如果第二个邻居不存在，使用第一个邻居的顶点集合初始化交集
  if (!neighborB)
    intersect = qh_setcopy(qh, neighborA->vertices, 0);
  else
    intersect = qh_vertexintersect_new(qh, neighborA->vertices, neighborB->vertices);

  // 将交集压入临时栈中
  qh_settemppush(qh, intersect);

  // 从交集中删除当前顶点
  qh_setdelsorted(intersect, vertex);

  // 遍历其余的邻居，与交集进行交集操作
  FOREACHneighbor_i_(qh, vertex) {
    if (neighbor_i >= 2) {
      zinc_(Zintersectnum); // 计数器增加
      qh_vertexintersect(qh, &intersect, neighbor->vertices);
      if (!SETfirst_(intersect)) {
        zinc_(Zintersectfail); // 计数器增加
        qh_settempfree(qh, &intersect); // 释放临时交集
        return NULL;
      }
    }
  }

  // 记录日志，显示交集的顶点数和顶点的标识符
  trace3((qh, qh->ferr, 3007, "qh_neighbor_intersections: %d vertices in neighbor intersection of v%d\n",
          qh_setsize(qh, intersect), vertex->id));
  return intersect;
} /* neighbor_intersections */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="neighbor_vertices">-</a>

  qh_neighbor_vertices(qh, vertex )
    return neighboring vertices for a vertex (not in subridge)
    assumes vertices have full vertex->neighbors

  returns:
    temporary set of vertices

  notes:
    updates qh.visit_id and qh.vertex_visit
    similar to qh_vertexridges

*/
setT *qh_neighbor_vertices(qhT *qh, vertexT *vertexA, setT *subridge) {
  facetT *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  setT *vertices= qh_settemp(qh, qh->TEMPsize);

  // 增加访问标识
  qh->visit_id++;

  // 设置邻居的访问标识
  FOREACHneighbor_(vertexA)
    neighbor->visitid = qh->visit_id;

  // 增加顶点的访问标识
  qh->vertex_visit++;
  vertexA->visitid = qh->vertex_visit;

  // 设置子桥的顶点的访问标识
  FOREACHvertex_(subridge) {
    vertex->visitid = qh->vertex_visit;
  }

  // 遍历顶点的每个邻居，将邻居的顶点添加到临时集合中
  FOREACHneighbor_(vertexA) {
    if (*neighborp)   /* 如果 neighborp 指针指向的值为真，则表示上一个邻居没有新的边缘 */
      qh_neighbor_vertices_facet(qh, vertexA, neighbor, &vertices);
  }
  trace3((qh, qh->ferr, 3035, "qh_neighbor_vertices: %d non-subridge, vertex neighbors for v%d\n",
    qh_setsize(qh, vertices), vertexA->id));
  return vertices;
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="neighbor_vertices_facet">-</a>

  qh_neighbor_vertices_facet(qh, vertex, facet, vertices )
    add neighboring vertices on ridges for vertex in facet
    neighbor->visitid==qh.visit_id if it hasn't been visited
    v.visitid==qh.vertex_visit if it is already in vertices

  returns:
    vertices updated
    sets facet->visitid to qh.visit_id-1

  notes:
    only called by qh_neighbor_vertices
    similar to qh_vertexridges_facet

  design:
    for each ridge of facet
      if ridge of visited neighbor (i.e., unprocessed)
        if vertex in ridge
          append unprocessed vertices of ridge
    mark facet processed
*/
void qh_neighbor_vertices_facet(qhT *qh, vertexT *vertexA, facetT *facet, setT **vertices) {
  ridgeT *ridge, **ridgep;
  facetT *neighbor;
  vertexT *second, *last, *vertex, **vertexp;
  int last_i= qh->hull_dim-2, count= 0;
  boolT isridge;

  // 如果facet是简单的（即每个面的顶点数都为hull_dim），直接添加所有顶点到vertices中
  if (facet->simplicial) {
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit) {
        vertex->visitid= qh->vertex_visit;
        qh_setappend(qh, vertices, vertex);
        count++;
      }
    }
  }else {
    // 对于非简单的facet，遍历其所有ridge
    FOREACHridge_(facet->ridges) {
      neighbor= otherfacet_(ridge, facet);
      // 如果邻居面还未访问过
      if (neighbor->visitid == qh->visit_id) {
        isridge= False;
        // 判断当前ridge是否包含vertexA
        if (SETfirst_(ridge->vertices) == vertexA) {
          isridge= True;
        }else if (last_i > 2) {
          // 获取ridge的第二个和倒数第二个顶点
          second= SETsecondt_(ridge->vertices, vertexT);
          last= SETelemt_(ridge->vertices, last_i, vertexT);
          // 检查是否vertexA在second和last之间（根据顶点id倒序排列）
          if (second->id >= vertexA->id && last->id <= vertexA->id) {
            if (second == vertexA || last == vertexA)
              isridge= True;
            else if (qh_setin(ridge->vertices, vertexA))
              isridge= True;
          }
        }else if (SETelem_(ridge->vertices, last_i) == vertexA) {
          isridge= True;
        }else if (last_i > 1 && SETsecond_(ridge->vertices) == vertexA) {
          isridge= True;
        }
        // 如果vertexA在ridge中，则将未访问的顶点添加到vertices中
        if (isridge) {
          FOREACHvertex_(ridge->vertices) {
            if (vertex->visitid != qh->vertex_visit) {
              vertex->visitid= qh->vertex_visit;
              qh_setappend(qh, vertices, vertex);
              count++;
            }
          }
        }
      }
    }
  }
  // 将facet的visitid设置为qh.visit_id-1，表示facet已被处理
  facet->visitid= qh->visit_id-1;
  // 如果有顶点被添加，则记录日志
  if (count) {
    trace4((qh, qh->ferr, 4079, "qh_neighbor_vertices_facet: found %d vertex neighbors for v%d in f%d (simplicial? %d)\n",
      count, vertexA->id, facet->id, facet->simplicial));
  }
} /* neighbor_vertices_facet */
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="next_vertexmerge">-</a>

  qh_next_vertexmerge(qh )
    return next vertex merge from qh.vertex_mergeset

  returns:
    vertex merge either MRGvertices or MRGsubridge
    drops merges of deleted vertices

  notes:
    called from qh_merge_pinchedvertices
*/
mergeT *qh_next_vertexmerge(qhT *qh /* qh.vertex_mergeset */) {
  // 定义变量
  mergeT *merge;
  int merge_i, merge_n, best_i= -1;
  realT bestdist= REALmax;

  // 遍历 qh.vertex_mergeset 中的每一个 merge
  FOREACHmerge_i_(qh, qh->vertex_mergeset) {
    // 检查 merge 中的 vertex1 和 vertex2 是否为空
    if (!merge->vertex1 || !merge->vertex2) {
      // 如果为空，输出错误信息并退出程序
      qh_fprintf(qh, qh->ferr, 6299, "qhull internal error (qh_next_vertexmerge): expecting two vertices for vertex merge.  Got v%d v%d and optional f%d\n",
        getid_(merge->vertex1), getid_(merge->vertex2), getid_(merge->facet1));
      qh_errexit(qh, qh_ERRqhull, merge->facet1, NULL);
    }
    // 检查 merge 中的 vertex1 或 vertex2 是否已被删除
    if (merge->vertex1->deleted || merge->vertex2->deleted) {
      // 如果有被删除的，输出跟踪信息并删除这个 merge
      trace3((qh, qh->ferr, 3030, "qh_next_vertexmerge: drop merge of v%d (del? %d) into v%d (del? %d) due to deleted vertex of r%d and r%d\n",
        merge->vertex1->id, merge->vertex1->deleted, merge->vertex2->id, merge->vertex2->deleted, getid_(merge->ridge1), getid_(merge->ridge2)));
      qh_drop_mergevertex(qh, merge);
      // 从 qh.vertex_mergeset 中删除当前 merge
      qh_setdelnth(qh, qh->vertex_mergeset, merge_i);
      merge_i--; merge_n--; // 调整计数器以匹配删除的 merge
      qh_memfree(qh, merge, (int)sizeof(mergeT)); // 释放 merge 的内存
    } else if (merge->distance < bestdist) {
      // 如果当前 merge 的距离小于当前最佳距离，更新最佳 merge 的索引
      bestdist = merge->distance;
      best_i = merge_i;
    }
  }
  // 将 merge 置为 NULL，如果找到了最佳的 merge，将其赋给 merge
  merge = NULL;
  if (best_i >= 0) {
    merge = SETelemt_(qh->vertex_mergeset, best_i, mergeT);
    # 如果最佳距离除以 qh->ONEmerge 大于 qh_WIDEpinched，则执行以下代码块
    if (bestdist/qh->ONEmerge > qh_WIDEpinched) {
      # 如果合并类型为 MRGvertices，则执行以下代码块
      if (merge->mergetype==MRGvertices) {
        # 如果 ridge1 的顶部等于 ridge2 的底部并且 ridge1 的底部等于 ridge2 的顶部，则输出错误信息
        if (merge->ridge1->top == merge->ridge2->bottom && merge->ridge1->bottom == merge->ridge2->top)
          qh_fprintf(qh, qh->ferr, 6391, "qhull topology error (qh_next_vertexmerge): no nearly adjacent vertices to resolve opposite oriented ridges r%d and r%d in f%d and f%d.  Nearest v%d and v%d dist %2.2g (%.1fx)\n",
            merge->ridge1->id, merge->ridge2->id, merge->ridge1->top->id, merge->ridge1->bottom->id, merge->vertex1->id, merge->vertex2->id, bestdist, bestdist/qh->ONEmerge);
        # 否则输出另一种错误信息
        else
          qh_fprintf(qh, qh->ferr, 6381, "qhull topology error (qh_next_vertexmerge): no nearly adjacent vertices to resolve duplicate ridges r%d and r%d.  Nearest v%d and v%d dist %2.2g (%.1fx)\n",
            merge->ridge1->id, merge->ridge2->id, merge->vertex1->id, merge->vertex2->id, bestdist, bestdist/qh->ONEmerge);
      # 如果不是 MRGvertices 类型，则输出另一种错误信息
      }else {
        qh_fprintf(qh, qh->ferr, 6208, "qhull topology error (qh_next_vertexmerge): no nearly adjacent vertices to resolve dupridge.  Nearest v%d and v%d dist %2.2g (%.1fx)\n",
          merge->vertex1->id, merge->vertex2->id, bestdist, bestdist/qh->ONEmerge);
      }
      # 发生拓扑错误，退出程序
      qh_errexit(qh, qh_ERRtopology, NULL, merge->ridge1);
    }
    # 从 qh->vertex_mergeset 集合中删除第 best_i 个元素
    qh_setdelnth(qh, qh->vertex_mergeset, best_i);
  }
  # 返回 merge 对象
  return merge;
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="opposite_horizonfacet">-</a>
  
  qh_opposite_horizonfacet(qh, merge, opposite )
    return horizon facet for one of the merge facets, and its opposite vertex across the ridge
    assumes either facet1 or facet2 of merge is 'mergehorizon'
    assumes both facets are simplicial facets on qh.new_facetlist
  
  returns:
    horizon facet and opposite vertex
  
  notes:
    called by qh_getpinchedmerges
*/
facetT *qh_opposite_horizonfacet(qhT *qh, mergeT *merge, vertexT **opposite) {
  facetT *facet, *horizon, *otherfacet;
  int neighbor_i;
  
  // 检查 merge 的两个面是否都是简单面（simplicial）且至少有一个是 mergehorizon
  if (!merge->facet1->simplicial || !merge->facet2->simplicial || (!merge->facet1->mergehorizon && !merge->facet2->mergehorizon)) {
    // 若条件不满足，输出错误信息并退出
    qh_fprintf(qh, qh->ferr, 6273, "qhull internal error (qh_opposite_horizonfacet): expecting merge of simplicial facets, at least one of which is mergehorizon.  Either simplicial or mergehorizon is wrong\n");
    qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
  }
  
  // 根据 merge 中哪一个面是 mergehorizon，确定 facet 和 otherfacet
  if (merge->facet1->mergehorizon) {
    facet= merge->facet1;
    otherfacet= merge->facet2;
  } else {
    facet= merge->facet2;
    otherfacet= merge->facet1;
  }
  
  // 获取 facet 的邻居面列表中的第一个面作为 horizon
  horizon= SETfirstt_(facet->neighbors, facetT);
  // 查找 otherfacet 在邻居面列表中的索引
  neighbor_i= qh_setindex(otherfacet->neighbors, facet);
  if (neighbor_i==-1)
    // 如果索引未找到，将其设置为 qh_MERGEridge 的索引
    neighbor_i= qh_setindex(otherfacet->neighbors, qh_MERGEridge);
  if (neighbor_i==-1) {
    // 若索引仍未找到，输出错误信息并退出
    qh_fprintf(qh, qh->ferr, 6238, "qhull internal error (qh_opposite_horizonfacet): merge facet f%d not connected to mergehorizon f%d\n",
      otherfacet->id, facet->id);
    qh_errexit2(qh, qh_ERRqhull, otherfacet, facet);
  }
  
  // 将 otherfacet 的顶点列表中的第 neighbor_i 个顶点作为 opposite
  *opposite= SETelemt_(otherfacet->vertices, neighbor_i, vertexT);
  // 返回 horizon 面
  return horizon;
} /* opposite_horizonfacet */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="reducevertices">-</a>

  qh_reducevertices(qh)
    reduce extra vertices, shared vertices, and redundant vertices
    facet->newmerge is set if merged since last call
    vertex->delridge is set if vertex was on a deleted ridge
    if !qh.MERGEvertices, only removes extra vertices

  returns:
    True if also merged degen_redundant facets
    vertices are renamed if possible
    clears facet->newmerge and vertex->delridge

  notes:
    called by qh_all_merges and qh_postmerge
    ignored if 2-d

  design:
    merge any degenerate or redundant facets
    repeat until no more degenerate or redundant facets
      for each newly merged facet
        remove extra vertices
      if qh.MERGEvertices
        for each newly merged facet
          for each vertex
            if vertex was on a deleted ridge
              rename vertex if it is shared
        for each new, undeleted vertex
          remove delridge flag
          if vertex is redundant
            merge degenerate or redundant facets
*/
boolT qh_reducevertices(qhT *qh) {
  int numshare=0, numrename= 0;  // 初始化共享顶点计数和重命名顶点计数
  boolT degenredun= False;  // 标志变量，指示是否存在退化或冗余情况
  facetT *newfacet;  // 定义面结构体指针变量
  vertexT *vertex, **vertexp;  // 定义顶点结构体指针变量及其指针的数组

  if (qh->hull_dim == 2)  // 如果凸壳维度为2，则返回False
    return False;
  trace2((qh, qh->ferr, 2101, "qh_reducevertices: reduce extra vertices, shared vertices, and redundant vertices\n"));
  // 输出调试信息，指示减少额外顶点、共享顶点和冗余顶点
  if (qh_merge_degenredundant(qh))  // 调用函数，合并和减少冗余顶点
    degenredun= True;  // 如果函数调用返回True，则更新标志变量

LABELrestart:  // 定义重启标签
  FORALLnew_facets {  // 遍历所有新面
    if (newfacet->newmerge) {  // 如果新面标记为需要合并
      if (!qh->MERGEvertices)  // 如果不允许合并顶点
        newfacet->newmerge= False;  // 取消新面的合并标记
      if (qh_remove_extravertices(qh, newfacet)) {  // 调用函数，移除多余顶点
        qh_degen_redundant_facet(qh, newfacet);  // 调用函数，处理退化和冗余面
        if (qh_merge_degenredundant(qh)) {  // 再次调用函数，合并和减少冗余顶点
          degenredun= True;  // 更新标志变量
          goto LABELrestart;  // 跳转到重启标签，重新处理
        }
      }
    }
  }
  if (!qh->MERGEvertices)  // 如果不允许合并顶点，则返回False
    return False;
  FORALLnew_facets {  // 再次遍历所有新面
    if (newfacet->newmerge) {  // 如果新面标记为需要合并
      newfacet->newmerge= False;  // 取消新面的合并标记
      FOREACHvertex_(newfacet->vertices) {  // 遍历新面的所有顶点
        if (vertex->delridge) {  // 如果顶点标记为删除
          if (qh_rename_sharedvertex(qh, vertex, newfacet)) {  // 调用函数，重命名共享顶点
            numshare++;  // 更新共享顶点计数
            if (qh_merge_degenredundant(qh)) {  // 再次调用函数，合并和减少冗余顶点
              degenredun= True;  // 更新标志变量
              goto LABELrestart;  // 跳转到重启标签，重新处理
            }
            vertexp--; /* repeat since deleted vertex */  // 减少指针以处理删除的顶点
          }
        }
      }
    }
  }
  FORALLvertex_(qh->newvertex_list) {  // 遍历所有新顶点列表
    if (vertex->delridge && !vertex->deleted) {  // 如果顶点标记为删除且未被删除
      vertex->delridge= False;  // 取消顶点删除标记
      if (qh->hull_dim >= 4 && qh_redundant_vertex(qh, vertex)) {  // 如果凸壳维度大于等于4且顶点为冗余顶点
        numrename++;  // 更新重命名顶点计数
        if (qh_merge_degenredundant(qh)) {  // 再次调用函数，合并和减少冗余顶点
          degenredun= True;  // 更新标志变量
          goto LABELrestart;  // 跳转到重启标签，重新处理
        }
      }
    }
  }
  trace1((qh, qh->ferr, 1014, "qh_reducevertices: renamed %d shared vertices and %d redundant vertices. Degen? %d\n",
          numshare, numrename, degenredun));
  // 输出调试信息，指示重命名的共享顶点数和冗余顶点数，以及是否存在退化情况
  return degenredun;  // 返回是否存在退化或冗余情况的标志
} /* reducevertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="redundant_vertex">-</a>

  qh_redundant_vertex(qh, vertex )
    rename a redundant vertex if qh_find_newvertex succeeds
    assumes vertices have full vertex->neighbors

  returns:
    if find a replacement vertex
      returns new vertex
      qh_renamevertex sets vertex->deleted for redundant vertex

  notes:
    only called by qh_reducevertices for vertex->delridge and hull_dim >= 4
    may add degenerate facets to qh.facet_mergeset
    doesn't change vertex->neighbors or create redundant facets

  design:
    intersect vertices of all facet neighbors of vertex
    determine ridges for these vertices
    if find a new vertex for vertex among these ridges and vertices
      rename vertex to the new vertex
*/
vertexT *qh_redundant_vertex(qhT *qh, vertexT *vertex) {
  vertexT *newvertex= NULL;  // 定义新顶点指针变量，初始化为NULL
  setT *vertices, *ridges;  // 定义顶点集合和边集合指针变量

  trace3((qh, qh->ferr, 3008, "qh_redundant_vertex: check if v%d from a deleted ridge can be renamed\n", vertex->id));
  // 输出调试信息，检查从删除的边缘中的顶点是否可以重命名
  if ((vertices= qh_neighbor_intersections(qh, vertex))) {  // 调用函数，获取顶点的邻近交集
    ridges= qh_vertexridges(qh, vertex, !qh_ALL);  // 调用函数，获取顶点的边缘
    # 如果找到了新顶点，则执行以下操作
    if ((newvertex = qh_find_newvertex(qh, vertex, vertices, ridges))) {
      # 增加 Zrenameall 计数器
      zinc_(Zrenameall);
      # 重命名顶点，同时使得相关的边缘数据结构无效化
      qh_renamevertex(qh, vertex, newvertex, ridges, NULL, NULL); /* ridges invalidated */
    }
    # 释放临时分配的 ridges 结构
    qh_settempfree(qh, &ridges);
    # 释放临时分配的 vertices 结构
    qh_settempfree(qh, &vertices);
  }
  # 返回新顶点
  return newvertex;
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="remove_extravertices">-</a>

  qh_remove_extravertices(qh, facet )
    remove extra vertices from non-simplicial facets

  returns:
    returns True if it finds them
      deletes facet from vertex neighbors
      facet may be redundant (test with qh_degen_redundant)

  notes:
    called by qh_renamevertex and qh_reducevertices
    a merge (qh_reducevertices) or qh_renamevertex may drop all ridges for a vertex in a facet

  design:
    for each vertex in facet
      if vertex not in a ridge (i.e., no longer used)
        delete vertex from facet
        delete facet from vertex's neighbors
        unless vertex in another facet
          add vertex to qh.del_vertices for later deletion
*/
boolT qh_remove_extravertices(qhT *qh, facetT *facet) {
  ridgeT *ridge, **ridgep;
  vertexT *vertex, **vertexp;
  boolT foundrem= False;

  if (facet->simplicial) {
    return False;
  }
  trace4((qh, qh->ferr, 4043, "qh_remove_extravertices: test non-simplicial f%d for extra vertices\n",
          facet->id));
  // 标记当前 facet 中的所有 vertex 为未见过
  FOREACHvertex_(facet->vertices)
    vertex->seen= False;
  // 遍历当前 facet 的所有 ridges，并标记 ridge 中的 vertex 为已见过
  FOREACHridge_(facet->ridges) {
    FOREACHvertex_(ridge->vertices)
      vertex->seen= True;
  }
  // 再次遍历 facet 的所有 vertex，如果某个 vertex 没有被标记为已见过，则将其删除，并处理相关的数据结构
  FOREACHvertex_(facet->vertices) {
    if (!vertex->seen) {
      foundrem= True;
      // 增加删除 vertex 的计数
      zinc_(Zremvertex);
      // 从 facet 的 vertices 集合中删除 vertex
      qh_setdelsorted(facet->vertices, vertex);
      // 从 vertex 的 neighbors 集合中删除 facet
      qh_setdel(vertex->neighbors, facet);
      // 如果 vertex 的 neighbors 集合为空，则标记 vertex 为已删除，并加入 qh.del_vertices 等待后续删除
      if (!qh_setsize(qh, vertex->neighbors)) {
        vertex->deleted= True;
        qh_setappend(qh, &qh->del_vertices, vertex);
        // 增加删除 vertex 的计数
        zinc_(Zremvertexdel);
        trace2((qh, qh->ferr, 2036, "qh_remove_extravertices: v%d deleted because it's lost all ridges\n", vertex->id));
      } else
        trace3((qh, qh->ferr, 3009, "qh_remove_extravertices: v%d removed from f%d because it's lost all ridges\n", vertex->id, facet->id));
      vertexp--; /*repeat*/
    }
  }
  return foundrem;
} /* remove_extravertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="remove_mergetype">-</a>

  qh_remove_mergetype(qh, mergeset, mergetype )
    Remove mergetype merges from mergeset

  notes:
    Does not preserve order
*/
void qh_remove_mergetype(qhT *qh, setT *mergeset, mergeType type) {
  mergeT *merge;
  int merge_i, merge_n;

  // 遍历 mergeset 中的每个 merge，如果其类型为指定的 mergetype，则从 mergeset 中删除
  FOREACHmerge_i_(qh, mergeset) {
    if (merge->mergetype == type) {
        // 跟踪日志，记录删除的 merge 的详细信息
        trace3((qh, qh->ferr, 3037, "qh_remove_mergetype: remove merge f%d f%d v%d v%d r%d r%d dist %2.2g type %d",
            getid_(merge->facet1), getid_(merge->facet2), getid_(merge->vertex1), getid_(merge->vertex2), getid_(merge->ridge1), getid_(merge->ridge2), merge->distance, type));
        // 从 mergeset 中删除当前位置的 merge，并调整迭代器和计数
        qh_setdelnth(qh, mergeset, merge_i);
        merge_i--; merge_n--;  /* repeat with next merge */
    }
  }
} /* remove_mergetype */
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="rename_adjacentvertex">-</a>

  qh_rename_adjacentvertex(qh, oldvertex, newvertex )
    renames oldvertex as newvertex.  Must be adjacent (i.e., in the same subridge)
    no-op if either vertex is deleted

  notes:
    called from qh_merge_pinchedvertices

  design:
    for all neighbors of oldvertex
      if simplicial, rename oldvertex to newvertex and drop if degenerate
      if needed, add oldvertex neighbor to newvertex
    determine ridges for oldvertex
    rename oldvertex as newvertex in ridges (qh_renamevertex)
*/
void qh_rename_adjacentvertex(qhT *qh, vertexT *oldvertex, vertexT *newvertex, realT dist) {
  setT *ridges;
  facetT *neighbor, **neighborp, *maxfacet= NULL;
  ridgeT *ridge, **ridgep;
  boolT istrace= False;
  int oldsize= qh_setsize(qh, oldvertex->neighbors);
  int newsize= qh_setsize(qh, newvertex->neighbors);
  coordT maxdist2= -REALmax, dist2;

  // 检查是否需要跟踪此函数的执行过程
  if (qh->IStracing >= 4 || oldvertex->id == qh->tracevertex_id || newvertex->id == qh->tracevertex_id) {
    istrace= True;
  }
  // 增加计数器，记录执行次数
  zzinc_(Ztotmerge);
  // 如果在跟踪模式下，输出详细的函数调用信息
  if (istrace) {
    qh_fprintf(qh, qh->ferr, 2071, "qh_rename_adjacentvertex: merge #%d rename v%d (%d neighbors) to v%d (%d neighbors) dist %2.2g\n",
      zzval_(Ztotmerge), oldvertex->id, oldsize, newvertex->id, newsize, dist);
  }
  // 如果任一顶点已被删除，则忽略重命名操作
  if (oldvertex->deleted || newvertex->deleted) {
    if (istrace || qh->IStracing >= 2) {
      qh_fprintf(qh, qh->ferr, 2072, "qh_rename_adjacentvertex: ignore rename.  Either v%d (%d) or v%d (%d) is deleted\n",
        oldvertex->id, oldvertex->deleted, newvertex->id, newvertex->deleted);
    }
    return;
  }
  // 如果任一顶点没有邻居，则输出错误信息并退出程序
  if (oldsize == 0 || newsize == 0) {
    qh_fprintf(qh, qh->ferr, 2072, "qhull internal error (qh_rename_adjacentvertex): expecting neighbor facets for v%d and v%d.  Got %d and %d neighbors resp.\n",
      oldvertex->id, newvertex->id, oldsize, newsize);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  // 遍历每个旧顶点的邻居
  FOREACHneighbor_(oldvertex) {
    // 如果邻居是简单形式（simplicial）
    if (neighbor->simplicial) {
      // 如果新顶点已经是该邻居的顶点之一，则标记将会使此邻居变为退化状态
      if (qh_setin(neighbor->vertices, newvertex)) {
        if (istrace || qh->IStracing >= 2) {
          qh_fprintf(qh, qh->ferr, 2070, "qh_rename_adjacentvertex: simplicial f%d contains old v%d and new v%d.  Will be marked degenerate by qh_renamevertex\n",
            neighbor->id, oldvertex->id, newvertex->id);
        }
        qh_makeridges(qh, neighbor); /* no longer simplicial, nummerge==0, skipped by qh_maybe_duplicateridge */
      } else {
        // 将旧顶点替换为新顶点，并确保新顶点包含此邻居
        qh_replacefacetvertex(qh, neighbor, oldvertex, newvertex);
        qh_setunique(qh, &newvertex->neighbors, neighbor);
        // 更新新顶点的邻居信息
        qh_newvertices(qh, neighbor->vertices);  /* for qh_update_vertexneighbors of vertex neighbors */
      }
    }
  }
  // 获取旧顶点的所有棱
  ridges= qh_vertexridges(qh, oldvertex, qh_ALL);
  // 如果在跟踪模式下，输出所有关联的棱信息
  if (istrace) {
    FOREACHridge_(ridges) {
      qh_printridge(qh, qh->ferr, ridge);
    }
  }
  FOREACHneighbor_(oldvertex) {
    // 如果邻居不是简单的（simplicial），执行以下操作
    if (!neighbor->simplicial){
      // 将新顶点添加到邻居的顶点列表中
      qh_addfacetvertex(qh, neighbor, newvertex);
      // 将邻居添加到新顶点的邻居集合中，并确保唯一性
      qh_setunique(qh, &newvertex->neighbors, neighbor);
      // 对邻居的顶点集合进行更新，以便后续更新新顶点的邻居信息
      qh_newvertices(qh, neighbor->vertices);  /* for qh_update_vertexneighbors of vertex neighbors */
      // 如果当前的 newfacet_list 等于 facet_tail，则将邻居从列表中移除并重新添加，以确保 qh_partitionvisible 有一个新的 facet 可用
      if (qh->newfacet_list == qh->facet_tail) {
        qh_removefacet(qh, neighbor);  /* add a neighbor to newfacet_list so that qh_partitionvisible has a newfacet */
        qh_appendfacet(qh, neighbor);
        neighbor->newfacet= True;
      }
    }
  }
  // 对顶点进行重命名，处理旧顶点与新顶点的关系，同时使得相关的边界信息失效
  qh_renamevertex(qh, oldvertex, newvertex, ridges, NULL, NULL);  /* ridges invalidated */
  // 如果旧顶点已删除且未被分区，则执行以下操作
  if (oldvertex->deleted && !oldvertex->partitioned) {
    // 遍历新顶点的每个邻居
    FOREACHneighbor_(newvertex) {
      // 如果邻居不可见，则计算旧顶点到邻居的平面距离
      if (!neighbor->visible) {
        qh_distplane(qh, oldvertex->point, neighbor, &dist2);
        // 如果距离大于最大距离平方，则更新最大距离平方和相应的最大邻居
        if (dist2>maxdist2) {
          maxdist2= dist2;
          maxfacet= neighbor;
        }
      }
    }
    // 记录调试信息，指示旧顶点作为共面点用于最远邻居的分区
    trace2((qh, qh->ferr, 2096, "qh_rename_adjacentvertex: partition old p%d(v%d) as a coplanar point for furthest f%d dist %2.2g.  Maybe repartition later (QH0031)\n",
      qh_pointid(qh, oldvertex->point), oldvertex->id, maxfacet->id, maxdist2))
    // 使用最远邻居进行共面分区，使用最大距离平方加速，否则重复距离测试
    qh_partitioncoplanar(qh, oldvertex->point, maxfacet, NULL, !qh_ALL);  /* faster with maxdist2, otherwise duplicates distance tests from maxdist2/dist2 */
    // 标记旧顶点已经分区
    oldvertex->partitioned= True;
  }
  // 释放临时变量 ridges
  qh_settempfree(qh, &ridges);
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="rename_sharedvertex">-</a>

  qh_rename_sharedvertex(qh, vertex, facet )
    detect and rename if shared vertex in facet
    vertices have full ->neighbors

  returns:
    newvertex or NULL
    the vertex may still exist in other facets (i.e., a neighbor was pinched)
    does not change facet->neighbors
    updates vertex->neighbors

  notes:
    only called by qh_reducevertices after qh_remove_extravertices
       so f.vertices does not contain extraneous vertices
    a shared vertex for a facet is only in ridges to one neighbor
    this may undo a pinched facet

    it does not catch pinches involving multiple facets.  These appear
      to be difficult to detect, since an exhaustive search is too expensive.

  design:
    if vertex only has two neighbors
      determine the ridges that contain the vertex
      determine the vertices shared by both neighbors
      if can find a new vertex in this set
        rename the vertex to the new vertex
*/
vertexT *qh_rename_sharedvertex(qhT *qh, vertexT *vertex, facetT *facet) {
  facetT *neighbor, **neighborp, *neighborA= NULL;
  setT *vertices, *ridges;
  vertexT *newvertex= NULL;

  if (qh_setsize(qh, vertex->neighbors) == 2) {
    neighborA= SETfirstt_(vertex->neighbors, facetT);
    if (neighborA == facet)
      neighborA= SETsecondt_(vertex->neighbors, facetT);
  }else if (qh->hull_dim == 3)
    return NULL;
  else {
    qh->visit_id++;
    FOREACHneighbor_(facet)
      neighbor->visitid= qh->visit_id;
    FOREACHneighbor_(vertex) {
      if (neighbor->visitid == qh->visit_id) {
        if (neighborA)
          return NULL;
        neighborA= neighbor;
      }
    }
  }
  
  if (!neighborA) {
    qh_fprintf(qh, qh->ferr, 6101, "qhull internal error (qh_rename_sharedvertex): v%d's neighbors not in f%d\n",
        vertex->id, facet->id);
    qh_errprint(qh, "ERRONEOUS", facet, NULL, NULL, vertex);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  
  if (neighborA) { /* avoid warning */
    /* the vertex is shared by facet and neighborA */
    ridges= qh_settemp(qh, qh->TEMPsize);
    neighborA->visitid= ++qh->visit_id;
    qh_vertexridges_facet(qh, vertex, facet, &ridges);
    trace2((qh, qh->ferr, 2037, "qh_rename_sharedvertex: p%d(v%d) is shared by f%d(%d ridges) and f%d\n",
      qh_pointid(qh, vertex->point), vertex->id, facet->id, qh_setsize(qh, ridges), neighborA->id));
    zinc_(Zintersectnum);
    vertices= qh_vertexintersect_new(qh, facet->vertices, neighborA->vertices);
    qh_setdel(vertices, vertex);
    qh_settemppush(qh, vertices);
    if ((newvertex= qh_find_newvertex(qh, vertex, vertices, ridges)))
      qh_renamevertex(qh, vertex, newvertex, ridges, facet, neighborA);  /* ridges invalidated */
    qh_settempfree(qh, &vertices);
    qh_settempfree(qh, &ridges);
  }
  
  return newvertex;
} /* rename_sharedvertex */
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="renameridgevertex">-</a>

  qh_renameridgevertex(qh, ridge, oldvertex, newvertex )
    renames oldvertex as newvertex in ridge

  returns:
    True if renames oldvertex
    False if deleted the ridge

  notes:
    called by qh_renamevertex
    caller sets newvertex->delridge for deleted ridges (qh_reducevertices)

  design:
    delete oldvertex from ridge
    if newvertex already in ridge
      copy ridge->noconvex to another ridge if possible
      delete the ridge
    else
      insert newvertex into the ridge
      adjust the ridge's orientation
*/
boolT qh_renameridgevertex(qhT *qh, ridgeT *ridge, vertexT *oldvertex, vertexT *newvertex) {
  int nth= 0, oldnth;
  facetT *temp;
  vertexT *vertex, **vertexp;

  // 获取 oldvertex 在 ridge->vertices 中的索引
  oldnth= qh_setindex(ridge->vertices, oldvertex);
  if (oldnth < 0) {
    // 如果 oldvertex 不在 ridge->vertices 中，输出错误信息并退出
    qh_fprintf(qh, qh->ferr, 6424, "qhull internal error (qh_renameridgevertex): oldvertex v%d not found in r%d.  Cannot rename to v%d\n",
        oldvertex->id, ridge->id, newvertex->id);
    qh_errexit(qh, qh_ERRqhull, NULL, ridge);
  }
  // 从 ridge->vertices 中删除 oldvertex
  qh_setdelnthsorted(qh, ridge->vertices, oldnth);
  
  // 遍历 ridge->vertices
  FOREACHvertex_(ridge->vertices) {
    // 如果找到 newvertex 已经在 ridge->vertices 中，执行以下操作
    if (vertex == newvertex) {
      // 增加 Zdelridge 计数
      zinc_(Zdelridge);
      // 如果 ridge->nonconvex 已设置，复制 ridge 的非凸属性到另一个 ridge
      if (ridge->nonconvex)
        qh_copynonconvex(qh, ridge);
      // 输出消息并删除 ridge
      trace2((qh, qh->ferr, 2038, "qh_renameridgevertex: ridge r%d deleted.  It contained both v%d and v%d\n",
        ridge->id, oldvertex->id, newvertex->id));
      qh_delridge_merge(qh, ridge); /* ridge.vertices deleted */
      return False;
    }
    // 找到合适的位置 nth，以便将 newvertex 插入 ridge->vertices 中
    if (vertex->id < newvertex->id)
      break;
    nth++;
  }
  // 在第 nth 个位置插入 newvertex 到 ridge->vertices
  qh_setaddnth(qh, &ridge->vertices, nth, newvertex);
  // 设置 ridge 的简单形状标志为 False
  ridge->simplicialtop= False;
  ridge->simplicialbot= False;
  // 如果旧索引 oldnth 与新索引 nth 之间差值为奇数，交换 ridge 的顶部和底部
  if (abs(oldnth - nth)%2) {
    trace3((qh, qh->ferr, 3010, "qh_renameridgevertex: swapped the top and bottom of ridge r%d\n",
            ridge->id));
    temp= ridge->top;
    ridge->top= ridge->bottom;
    ridge->bottom= temp;
  }
  return True;
} /* renameridgevertex */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="renamevertex">-</a>

  qh_renamevertex(qh, oldvertex, newvertex, ridges, oldfacet, neighborA )
    renames oldvertex as newvertex in ridges of non-simplicial neighbors
    set oldfacet/neighborA if oldvertex is shared between two facets (qh_rename_sharedvertex)
    otherwise qh_redundant_vertex or qh_rename_adjacentvertex

  returns:
    if oldfacet and multiple neighbors, oldvertex may still exist afterwards
    otherwise sets oldvertex->deleted for later deletion
    one or more ridges maybe deleted
    ridges is invalidated
    merges may be added to degen_mergeset via qh_maydropneighbor or qh_degen_redundant_facet

  notes:
    qh_rename_sharedvertex can not change neighbors of newvertex (since it's a subset)
    qh_redundant_vertex due to vertex->delridge for qh_reducevertices
*/
    # qh_rename_adjacentvertex for complete renames
    design:
      # 遍历所有的“ridge”（即边界）
      for each ridge in ridges
        # 将 oldvertex 重命名为 newvertex，并删除退化的 ridges
        rename oldvertex to newvertex and delete degenerate ridges
      # 如果 oldfacet 未定义
      if oldfacet not defined
        # 对于 oldvertex 的每个非单纯邻居
        for each non-simplicial neighbor of oldvertex
          # 从邻居的顶点列表中删除 oldvertex
          delete oldvertex from neighbor's vertices
          # 从邻居中移除额外的顶点
          remove extra vertices from neighbor
        # 将 oldvertex 添加到 qh.del_vertices 列表中
        add oldvertex to qh.del_vertices
      # 否则如果 oldvertex 仅在 oldfacet 和 neighborA 之间
      else if oldvertex only between oldfacet and neighborA
        # 从 oldfacet 和 neighborA 中删除 oldvertex
        delete oldvertex from oldfacet and neighborA
        # 将 oldvertex 添加到 qh.del_vertices 列表中
        add oldvertex to qh.del_vertices
      # 否则，oldvertex 同时在 oldfacet、neighborA 以及其他 facet 中（即被夹住）
      else oldvertex is in oldfacet and neighborA and other facets (i.e., pinched)
        # 从 oldfacet 中删除 oldvertex
        delete oldvertex from oldfacet
        # 从 oldvertex 的邻居列表中删除 oldfacet
        delete oldfacet from old vertex's neighbors
        # 从 neighborA 中移除额外的顶点（例如 oldvertex）
        remove extra vertices (e.g., oldvertex) from neighborA
/*
void qh_renamevertex(qhT *qh, vertexT *oldvertex, vertexT *newvertex, setT *ridges, facetT *oldfacet, facetT *neighborA) {
  facetT *neighbor, **neighborp;  // 定义邻居面的指针和邻居面的临时变量
  ridgeT *ridge, **ridgep;  // 定义棱的指针和棱的临时变量
  int topsize, bottomsize;  // 定义顶部大小和底部大小
  boolT istrace= False;  // 定义是否追踪标志，默认为 False

#ifndef qh_NOtrace
  // 如果启用追踪或者旧顶点或新顶点的 id 与追踪顶点的 id 相同，则设置追踪标志为 True
  if (qh->IStracing >= 2 || oldvertex->id == qh->tracevertex_id ||
        newvertex->id == qh->tracevertex_id) {
    istrace= True;
    // 输出追踪信息，显示重命名顶点及其相关信息
    qh_fprintf(qh, qh->ferr, 2086, "qh_renamevertex: rename v%d to v%d in %d ridges with old f%d and neighbor f%d\n",
      oldvertex->id, newvertex->id, qh_setsize(qh, ridges), getid_(oldfacet), getid_(neighborA));
  }
#endif

  // 遍历每个棱
  FOREACHridge_(ridges) {
    // 如果成功重命名棱中的顶点
    if (qh_renameridgevertex(qh, ridge, oldvertex, newvertex)) { /* ridge is deleted if False, invalidating ridges */
      // 计算顶部和底部的顶点数目
      topsize= qh_setsize(qh, ridge->top->vertices);
      bottomsize= qh_setsize(qh, ridge->bottom->vertices);
      // 如果顶部顶点数小于凸包维度或者顶部不是简单形式且新顶点在顶部顶点集合中
      if (topsize < qh->hull_dim || (topsize == qh->hull_dim && !ridge->top->simplicial && qh_setin(ridge->top->vertices, newvertex))) {
        // 输出追踪信息，表示忽略重复检查，因为重命名后顶部面将变得退化
        trace4((qh, qh->ferr, 4070, "qh_renamevertex: ignore duplicate check for r%d.  top f%d (size %d) will be degenerate after rename v%d to v%d\n",
          ridge->id, ridge->top->id, topsize, oldvertex->id, newvertex->id));
      } else if (bottomsize < qh->hull_dim || (bottomsize == qh->hull_dim && !ridge->bottom->simplicial && qh_setin(ridge->bottom->vertices, newvertex))) {
        // 输出追踪信息，表示忽略重复检查，因为重命名后底部面将变得退化
        trace4((qh, qh->ferr, 4071, "qh_renamevertex: ignore duplicate check for r%d.  bottom f%d (size %d) will be degenerate after rename v%d to v%d\n",
          ridge->id, ridge->bottom->id, bottomsize, oldvertex->id, newvertex->id));
      } else
        // 否则，进行棱的可能重复检查
        qh_maybe_duplicateridge(qh, ridge);
    }
  }

  // 如果旧面为空
  if (!oldfacet) {
    // 输出追踪信息，表示在多个面中重命名顶点，用于 qh_redundant_vertex 或 MRGsubridge
    if (istrace)
      qh_fprintf(qh, qh->ferr, 2087, "qh_renamevertex: renaming v%d to v%d in several facets for qh_redundant_vertex or MRGsubridge\n",
               oldvertex->id, newvertex->id);
    // 遍历旧顶点的每个邻居面
    FOREACHneighbor_(oldvertex) {
      // 如果邻居面是简单形式
      if (neighbor->simplicial) {
        // 处理简单形式的退化面
        qh_degen_redundant_facet(qh, neighbor); /* e.g., rbox 175 C3,2e-13 D4 t1545235541 | qhull d */
      } else {
        // 如果启用了追踪，则输出重命名非简单形式邻居面中的顶点信息
        if (istrace)
          qh_fprintf(qh, qh->ferr, 4080, "qh_renamevertex: rename vertices in non-simplicial neighbor f%d of v%d\n", neighbor->id, oldvertex->id);
        // 可能删除邻居面
        qh_maydropneighbor(qh, neighbor);
        // 从邻居面的顶点集合中删除旧顶点
        qh_setdelsorted(neighbor->vertices, oldvertex); /* if degenerate, qh_degen_redundant_facet will add to mergeset */
        // 如果移除了额外顶点，则调整邻居面指针
        if (qh_remove_extravertices(qh, neighbor))
          neighborp--; /* neighbor deleted from oldvertex neighborset */
        // 处理非简单形式邻居面的退化面情况
        qh_degen_redundant_facet(qh, neighbor); /* either direction may be redundant, faster if combine? */
        // 测试退化面的冗余邻居面
        qh_test_redundant_neighbors(qh, neighbor);
        // 测试退化邻居面
        qh_test_degen_neighbors(qh, neighbor);
      }
    }
    // 如果旧顶点尚未标记为删除，则标记为删除并添加到删除顶点列表中
    if (!oldvertex->deleted) {
      oldvertex->deleted= True;
      qh_setappend(qh, &qh->del_vertices, oldvertex);
    // 如果 oldvertex 的邻居数为 2，则进行重命名处理
    } else if (qh_setsize(qh, oldvertex->neighbors) == 2) {
        // 增加重命名共享点的计数
        zinc_(Zrenameshare);
        // 如果启用了跟踪输出，打印重命名信息到错误文件
        if (istrace)
            qh_fprintf(qh, qh->ferr, 3039, "qh_renamevertex: renaming v%d to v%d in oldfacet f%d for qh_rename_sharedvertex\n",
                       oldvertex->id, newvertex->id, oldfacet->id);
        // 遍历 oldvertex 的每个邻居
        FOREACHneighbor_(oldvertex) {
            // 从邻居的顶点列表中删除 oldvertex
            qh_setdelsorted(neighbor->vertices, oldvertex);
            // 处理邻居可能存在的冗余面
            qh_degen_redundant_facet(qh, neighbor);
        }
        // 标记 oldvertex 为已删除
        oldvertex->deleted = True;
        // 将 oldvertex 添加到删除顶点列表中
        qh_setappend(qh, &qh->del_vertices, oldvertex);
    } else {
        // 增加重命名捏合点的计数
        zinc_(Zrenamepinch);
        // 如果启用了跟踪输出或者跟踪级别大于等于 1，打印捏合重命名信息到错误文件
        if (istrace || qh->IStracing >= 1)
            qh_fprintf(qh, qh->ferr, 3040, "qh_renamevertex: renaming pinched v%d to v%d between f%d and f%d\n",
                       oldvertex->id, newvertex->id, oldfacet->id, neighborA->id);
        // 从 oldfacet 的顶点列表中删除 oldvertex
        qh_setdelsorted(oldfacet->vertices, oldvertex);
        // 从 oldvertex 的邻居列表中删除 oldfacet
        qh_setdel(oldvertex->neighbors, oldfacet);
        // 如果移除了额外的顶点，则处理 neighborA 可能存在的冗余面
        if (qh_remove_extravertices(qh, neighborA))
            qh_degen_redundant_facet(qh, neighborA);
    }
    // 如果存在 oldfacet，则处理可能存在的冗余面
    if (oldfacet)
        qh_degen_redundant_facet(qh, oldfacet);
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_appendmerge">-</a>

  qh_test_appendmerge(qh, facet, neighbor, simplicial )
    test convexity and append to qh.facet_mergeset if non-convex
    if pre-merging,
      no-op if qh.SKIPconvex, or qh.MERGEexact and coplanar
    if simplicial, assumes centrum test is valid (e.g., adjacent, simplicial new facets)

  returns:
    true if appends facet/neighbor to qh.facet_mergeset
    sets facet->center as needed
    does not change facet->seen

  notes:
    called from qh_getmergeset_initial, qh_getmergeset, and qh_test_vneighbors
    must be at least as strong as qh_checkconvex (poly2_r.c)
    assumes !f.flipped

  design:
    exit if qh.SKIPconvex ('Q0') and !qh.POSTmerging
    if qh.cos_max ('An') is defined and merging coplanars
      if the angle between facet normals is too shallow
        append an angle-coplanar merge to qh.mergeset
        return True
    test convexity of facet and neighbor
*/
boolT qh_test_appendmerge(qhT *qh, facetT *facet, facetT *neighbor, boolT simplicial) {
  realT angle= -REALmax;  // 初始化角度为最小可能值
  boolT okangle= False;   // 角度是否有效

  if (qh->SKIPconvex && !qh->POSTmerging)  // 如果设置了跳过凸性检查并且不进行后处理合并，则直接退出
    return False;
  if (qh->cos_max < REALmax/2 && (!qh->MERGEexact || qh->POSTmerging)) {
    angle= qh_getangle(qh, facet->normal, neighbor->normal);  // 计算两个面法线之间的角度
    okangle= True;  // 标记角度为有效
    zinc_(Zangletests);  // 增加角度测试的计数器
    if (angle > qh->cos_max) {  // 如果角度大于设定的最大角度限制
      zinc_(Zcoplanarangle);  // 增加共面角计数器
      qh_appendmergeset(qh, facet, neighbor, MRGanglecoplanar, 0.0, angle);  // 将角度共面合并添加到mergeset中
      trace2((qh, qh->ferr, 2039, "qh_test_appendmerge: coplanar angle %4.4g between f%d and f%d\n",
         angle, facet->id, neighbor->id));  // 跟踪输出共面角度信息
      return True;  // 返回true，表示成功添加共面合并
    }
  }
  if (simplicial || qh->hull_dim <= 3)
    return qh_test_centrum_merge(qh, facet, neighbor, angle, okangle);  // 检查中心点合并的情况
  else
    return qh_test_nonsimplicial_merge(qh, facet, neighbor, angle, okangle);  // 检查非简单面合并的情况
} /* test_appendmerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_centrum_merge">-</a>

  qh_test_centrum_merge(qh, facet, neighbor, angle, okangle )
    test centrum convexity and append non-convex facets to qh.facet_mergeset
    'angle' is angle between facets if okangle is true, otherwise use 0.0

  returns:
    true if append facet/neighbor to qh.facet_mergeset
    sets facet->center as needed
    does not change facet->seen

  notes:
    called from test_appendmerge if adjacent simplicial facets or 2-d/3-d
    at least as strict as qh_checkconvex, including qh.DISTround ('En' and 'Rn')

  design:
    make facet's centrum if needed
    if facet's centrum is above the neighbor (qh.centrum_radius)
      set isconcave

    if facet's centrum is not below the neighbor (-qh.centrum_radius)
      set iscoplanar
    make neighbor's centrum if needed
    if neighbor's centrum is above the facet
      set isconcave
    else if neighbor's centrum is not below the facet
      set iscoplanar
*/
    # 如果是凹多边形或者是共面并且正在合并共面的部分
    if isconcave or iscoplanar and merging coplanars
      # 如果需要获取角度（通过 qh.ANGLEmerge 'An'）
      get angle if needed (qh.ANGLEmerge 'An')
      # 将凹共面、凹面或者共面合并追加到 qh.mergeset 中
      append concave-coplanar, concave ,or coplanar merge to qh.mergeset
/*
  qh_test_centrum_merge(qh, facet, neighbor, angle, okangle)
  在几何计算中测试是否可以合并两个面 facet 和 neighbor

  qh: Qhull 的全局数据结构
  facet: 要测试的面
  neighbor: 面 facet 的一个潜在邻居
  angle: 用于合并的角度
  okangle: 角度是否有效的标志

  返回值: 如果可以合并返回 True，否则返回 False
*/
boolT qh_test_centrum_merge(qhT *qh, facetT *facet, facetT *neighbor, realT angle, boolT okangle) {
  coordT dist, dist2, mergedist;
  boolT isconcave= False, iscoplanar= False;

  // 如果 facet 的中心点未初始化，则计算并设置中心点
  if (!facet->center)
    facet->center= qh_getcentrum(qh, facet);
  // 增加计数器 Zcentrumtests
  zzinc_(Zcentrumtests);
  // 计算邻居 neighbor 到 facet 的距离 dist
  qh_distplane(qh, facet->center, neighbor, &dist);
  // 如果 dist 大于 centrum_radius，则 facet 是凹的
  if (dist > qh->centrum_radius)
    isconcave= True;
  // 否则如果 dist 大于等于 -centrum_radius，则 facet 和 neighbor 是共面的
  else if (dist >= -qh->centrum_radius)
    iscoplanar= True;

  // 如果邻居 neighbor 的中心点未初始化，则计算并设置中心点
  if (!neighbor->center)
    neighbor->center= qh_getcentrum(qh, neighbor);
  // 再次增加计数器 Zcentrumtests
  zzinc_(Zcentrumtests);
  // 计算 facet 到邻居 neighbor 的距离 dist2
  qh_distplane(qh, neighbor->center, facet, &dist2);
  // 如果 dist2 大于 centrum_radius，则 facet 是凹的
  if (dist2 > qh->centrum_radius)
    isconcave= True;
  // 否则如果不是共面且 dist2 大于等于 -centrum_radius，则 facet 和 neighbor 是共面的
  else if (!iscoplanar && dist2 >= -qh->centrum_radius)
    iscoplanar= True;

  // 如果既不是凹的且不共面，或者 MERGEexact 且不进行 POSTmerging，则无法合并，返回 False
  if (!isconcave && (!iscoplanar || (qh->MERGEexact && !qh->POSTmerging)))
    return False;

  // 如果角度不合适且启用了 ANGLEmerge，则重新计算角度并增加角度测试计数器 Zangletests
  if (!okangle && qh->ANGLEmerge) {
    angle= qh_getangle(qh, facet->normal, neighbor->normal);
    zinc_(Zangletests);
  }

  // 根据凹性和共面性分类处理合并情况
  if (isconcave && iscoplanar) {
    // 如果 dist 大于 dist2，则将 facet 和 neighbor 标记为 MRGconcavecoplanar 合并集，并记录追踪信息
    if (dist > dist2)
      qh_appendmergeset(qh, facet, neighbor, MRGconcavecoplanar, dist, angle);
    else
      qh_appendmergeset(qh, neighbor, facet, MRGconcavecoplanar, dist2, angle);
    trace0((qh, qh->ferr, 36, "qh_test_centrum_merge: concave f%d to coplanar f%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
           facet->id, neighbor->id, dist, dist2, angle, qh->furthest_id));
  } else if (isconcave) {
    // 如果 facet 是凹的，则合并 facet 和 neighbor 标记为 MRGconcave 合并集，并记录追踪信息
    mergedist= fmax_(dist, dist2);
    zinc_(Zconcaveridge);
    qh_appendmergeset(qh, facet, neighbor, MRGconcave, mergedist, angle);
    trace0((qh, qh->ferr, 37, "qh_test_centrum_merge: concave f%d to f%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, neighbor->id, dist, dist2, angle, qh->furthest_id));
  } else /* iscoplanar */ {
    // 如果 facet 和 neighbor 是共面的，则合并 facet 和 neighbor 标记为 MRGcoplanar 合并集，并记录追踪信息
    mergedist= fmin_(fabs_(dist), fabs_(dist2));
    zinc_(Zcoplanarcentrum);
    qh_appendmergeset(qh, facet, neighbor, MRGcoplanar, mergedist, angle);
    trace2((qh, qh->ferr, 2097, "qh_test_centrum_merge: coplanar f%d to f%d dist %4.4g, reverse dist %4.4g angle %4.4g\n",
              facet->id, neighbor->id, dist, dist2, angle));
  }
  // 返回合并成功的标志 True
  return True;
} /* test_centrum_merge */

/*
  qh_test_degen_neighbors(qh, facet)
  检测面 facet 的退化邻居，并将其附加到 qh.degen_mergeset 中

  qh: Qhull 的全局数据结构
  facet: 要检测的面

  notes:
  在 qh_mergefacet() 和 qh_renamevertex() 结束后调用
  在 test_redundant_facet() 后调用，因为 MRGredundant 比 MRGdegen 更便宜
  退化的面指的是邻居少于 hull_dim 的面
  参见 qh_merge_degenredundant()
*/
void qh_test_degen_neighbors(qhT *qh, facetT *facet) {
  facetT *neighbor, **neighborp;
  int size;

  // 追踪消息，记录正在测试 facet 的退化邻居
  trace4((qh, qh->ferr, 4073, "qh_test_degen_neighbors: test for degenerate neighbors of f%d\n", facet->id));
  // 遍历 facet 的每个邻居
  FOREACHneighbor_(facet) {
    # 检查邻居是否可见，如果可见则输出错误信息并退出
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6359, "qhull internal error (qh_test_degen_neighbors): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
        facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    
    # 如果邻居被标记为退化（degenerate）、冗余（redundant）、或者有冗余边（dupridge），则跳过当前循环
    if (neighbor->degenerate || neighbor->redundant || neighbor->dupridge) /* will merge or delete */
      continue;
    
    # 计算邻居的邻居数目
    # 如果邻居的邻居数小于凸壳的维度，将当前邻居与自身标记为退化，添加到合并集合中
    if ((size= qh_setsize(qh, neighbor->neighbors)) < qh->hull_dim) {
      qh_appendmergeset(qh, neighbor, neighbor, MRGdegen, 0.0, 1.0);
      # 输出跟踪信息，表示邻居 f%d 被标记为退化，具有 size 个邻居，是 f%d 的邻居
      trace2((qh, qh->ferr, 2019, "qh_test_degen_neighbors: f%d is degenerate with %d neighbors.  Neighbor of f%d.\n", neighbor->id, size, facet->id));
    }
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_nonsimplicial_merge">-</a>
  
  qh_test_nonsimplicial_merge(qh, facet, neighbor, angle, okangle )
    test centrum and vertex convexity and append non-convex or redundant facets to qh.facet_mergeset
    'angle' is angle between facets if okangle is true, otherwise use 0.0
    skips coplanar merges if pre-merging with qh.MERGEexact ('Qx')

  returns:
    true if appends facet/neighbor to qh.facet_mergeset
    sets facet->center as needed
    does not change facet->seen

  notes:
    only called from test_appendmerge if a non-simplicial facet and at least 4-d
    at least as strict as qh_checkconvex, including qh.DISTround ('En' and 'Rn')
      centrums must be < -qh.centrum_radius
    tests vertices as well as centrums since a facet may be twisted relative to its neighbor

  design:
    set precision constants for maxoutside, clearlyconcave, minvertex, and coplanarcentrum
      use maxoutside for coplanarcentrum if premerging with 'Qx' and qh_MAXcoplanarcentrum merges
      otherwise use qh.centrum_radious for coplanarcentrum
    make facet and neighbor centrums if needed
    isconcave if a centrum is above neighbor (coplanarcentrum)
    iscoplanar if a centrum is not below neighbor (-qh.centrum_radius)
    maybeconvex if a centrum is clearly below neighbor (-clearyconvex)
    return False if both centrums clearly below neighbor (-clearyconvex)
    return MRGconcave if isconcave

    facets are neither clearly convex nor clearly concave
    test vertices as well as centrums
    if maybeconvex
      determine mindist and maxdist for vertices of the other facet
      maybe MRGredundant
    otherwise
      determine mindist and maxdist for vertices of either facet
      maybe MRGredundant
      maybeconvex if a vertex is clearly below neighbor (-clearconvex)

    vertices are concave if dist > clearlyconcave
    vertices are twisted if dist > maxoutside (isconcave and maybeconvex)
    return False if not concave and pre-merge of 'Qx' (qh.MERGEexact)
    vertices are coplanar if dist in -minvertex..maxoutside
    if !isconcave, vertices are coplanar if dist >= -qh.MAXcoplanar (n*qh.premerge_centrum)

    return False if neither concave nor coplanar
    return MRGtwisted if isconcave and maybeconvex
    return MRGconcavecoplanar if isconcave and isconvex
    return MRGconcave if isconcave
    return MRGcoplanar if iscoplanar
*/
  # 计算最大外接球半径，考虑合并阈值和距离修正因子
  maxoutside= fmax_(neighbor->maxoutside, qh->ONEmerge + qh->DISTround);
  maxoutside= fmax_(maxoutside, facet->maxoutside);

  # 计算显著凹面比例乘以最大外接球半径，用于判断是否显著凹面
  clearlyconcave= qh_RATIOconcavehorizon * maxoutside;

  # 计算最小顶点坐标的绝对值，用于判断是否共面
  minvertex= fmax_(-qh->min_vertex, qh->MAXcoplanar);

  # 计算显著凸面比例乘以最小顶点坐标绝对值，用于判断是否显著凸面
  clearlyconvex= qh_RATIOconvexmerge * minvertex;

  # 如果启用精确合并且不是后期合并，且某个面的合并次数超过阈值，则使用最大外接球半径作为共面中心的半径
  if (qh->MERGEexact && !qh->POSTmerging && (facet->nummerge > qh_MAXcoplanarcentrum || neighbor->nummerge > qh_MAXcoplanarcentrum))
    coplanarcentrum= maxoutside;
  else
    coplanarcentrum= qh->centrum_radius;

  # 如果面的中心点未计算，计算它
  if (!facet->center)
    facet->center= qh_getcentrum(qh, facet);
  zzinc_(Zcentrumtests);

  # 计算面与邻近面之间的距离
  qh_distplane(qh, facet->center, neighbor, &dist);

  # 根据距离判断是否凹面
  if (dist > coplanarcentrum)
    isconcave= True;
  elif (dist >= -qh->centrum_radius)
    iscoplanar= True;
  else
    maybeconvex= True;

  # 如果邻近面的中心点未计算，计算它
  if (!neighbor->center)
    neighbor->center= qh_getcentrum(qh, neighbor);
  zzinc_(Zcentrumtests);

  # 计算邻近面与面之间的距离
  qh_distplane(qh, neighbor->center, facet, &dist2);

  # 根据距离判断是否凹面
  if (dist2 > coplanarcentrum)
    isconcave= True;
  elif (dist2 >= -qh->centrum_radius)
    iscoplanar= True;
  else:
    # 如果两个中心点都明显凸面，则返回 False
    if (maybeconvex)
      return False;

  # 如果判定为凹面，根据条件记录合并信息
  if (isconcave):
    if (!okangle && qh->ANGLEmerge):
      angle= qh_getangle(qh, facet->normal, neighbor->normal);
      zinc_(Zangletests);
    mergedist= fmax_(dist, dist2);
    zinc_(Zconcaveridge);
    qh_appendmergeset(qh, facet, neighbor, MRGconcave, mergedist, angle);
    trace0((qh, qh->ferr, 18, "qh_test_nonsimplicial_merge: concave centrum for f%d or f%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, neighbor->id, dist, dist2, angle, qh->furthest_id));
    return True;

  # 如果既非明显凸面也非明显凹面，进一步测试顶点距离
  if (maybeconvex):
    if (dist < -clearlyconvex):
      maxdist= dist;  /* facet centrum clearly convex, no need to test its vertex distance */
      mindist= dist;

      # 寻找最远顶点，并记录距离
      maxvertex2= qh_furthestvertex(qh, neighbor, facet, &maxdist2, &mindist2);
      if (!maxvertex2):
        qh_appendmergeset(qh, neighbor, facet, MRGredundant, maxdist2, qh_ANGLEnone);
        isredundant= True;
    }else { /* dist2 < -clearlyconvex */
      maxdist2= dist2;   /* 设置最大距离为当前 dist2 */
      mindist2= dist2;    /* 设置最小距离为当前 dist2 */
      maxvertex= qh_furthestvertex(qh, facet, neighbor, &maxdist, &mindist);   /* 获取最远顶点及其距离 */
      if (!maxvertex) {    /* 如果最远顶点不存在 */
        qh_appendmergeset(qh, facet, neighbor, MRGredundant, maxdist, qh_ANGLEnone);    /* 将当前面和邻面标记为冗余 */
        isredundant= True;  /* 设置为冗余 */
      }
    }
  }else {
    maxvertex= qh_furthestvertex(qh, facet, neighbor, &maxdist, &mindist);   /* 获取最远顶点及其距离 */
    if (maxvertex) {    /* 如果最远顶点存在 */
      maxvertex2= qh_furthestvertex(qh, neighbor, facet, &maxdist2, &mindist2);   /* 获取另一侧最远顶点及其距离 */
      if (!maxvertex2) {    /* 如果另一侧最远顶点不存在 */
        qh_appendmergeset(qh, neighbor, facet, MRGredundant, maxdist2, qh_ANGLEnone);    /* 将邻面和当前面标记为冗余 */
        isredundant= True;  /* 设置为冗余 */
      }else if (mindist < -clearlyconvex || mindist2 < -clearlyconvex)
        maybeconvex= True;   /* 可能是凸的 */
    }else { /* !maxvertex */
      qh_appendmergeset(qh, facet, neighbor, MRGredundant, maxdist, qh_ANGLEnone);    /* 将当前面和邻面标记为冗余 */
      isredundant= True;  /* 设置为冗余 */
    }
  }
  if (isredundant) {
    zinc_(Zredundantmerge);    /* 增加冗余合并计数 */
    return True;    /* 返回真，表示冗余 */
  }

  if (maxdist > clearlyconcave || maxdist2 > clearlyconcave)
    isconcave= True;    /* 如果最大距离大于 clearlyconcave，则是凹的 */
  else if (maybeconvex) {
    if (maxdist > maxoutside || maxdist2 > maxoutside)
      isconcave= True;  /* 如果最大距离大于 maxoutside 或者另一侧最大距离大于 maxoutside，则是凹的 */
  }
  if (!isconcave && qh->MERGEexact && !qh->POSTmerging)
    return False;   /* 如果不是凹的，并且是精确合并且非后处理合并，则返回假 */

  if (isconcave && !iscoplanar) {
    if (maxdist < maxoutside && (-qh->MAXcoplanar || (maxdist2 < maxoutside && mindist2 >= -qh->MAXcoplanar)))
      iscoplanar= True; /* 如果是凹的且非共面，并且满足一定条件，则是共面的 */
  }else if (!iscoplanar) {
    if (mindist >= -qh->MAXcoplanar || mindist2 >= -qh->MAXcoplanar)
      iscoplanar= True;  /* 如果满足一定条件，则是共面的 */
  }
  if (!isconcave && !iscoplanar)
    return False;   /* 如果既不是凹的也不是共面的，则返回假 */

  if (!okangle && qh->ANGLEmerge) {
    angle= qh_getangle(qh, facet->normal, neighbor->normal);   /* 获取面法向量之间的夹角 */
    zinc_(Zangletests);   /* 增加角度测试计数 */
  }
  if (isconcave && maybeconvex) {
    zinc_(Ztwistedridge);   /* 增加扭曲边缘计数 */
    if (maxdist > maxdist2)
      qh_appendmergeset(qh, facet, neighbor, MRGtwisted, maxdist, angle);   /* 标记扭曲的合并 */
    else
      qh_appendmergeset(qh, neighbor, facet, MRGtwisted, maxdist2, angle);   /* 标记扭曲的合并（反向） */
    trace0((qh, qh->ferr, 27, "qh_test_nonsimplicial_merge: twisted concave f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
           facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }else if (isconcave && iscoplanar) {
    zinc_(Zconcavecoplanarridge);   /* 增加凹共面边缘计数 */
    if (maxdist > maxdist2)
      qh_appendmergeset(qh, facet, neighbor, MRGconcavecoplanar, maxdist, angle);   /* 标记凹共面的合并 */
    else
      qh_appendmergeset(qh, neighbor, facet, MRGconcavecoplanar, maxdist2, angle);   /* 标记凹共面的合并（反向） */
    trace0((qh, qh->ferr, 28, "qh_test_nonsimplicial_merge: concave coplanar f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }else if (isconcave) {
    mergedist= fmax_(maxdist, maxdist2);   /* 计算最大距离 */
    zinc_(Zconcaveridge);   /* 增加凹边缘计数 */
    # 如果不是共面的情况，则执行以下操作
    qh_appendmergeset(qh, facet, neighbor, MRGconcave, mergedist, angle);
    # 输出调试信息到跟踪日志，记录非共面合并的详细信息，包括面和顶点信息、距离、角度等
    trace0((qh, qh->ferr, 29, "qh_test_nonsimplicial_merge: concave f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }else /* iscoplanar */ {
    # 计算合并距离，取最大值，表示共面情况下的合并距离
    mergedist= fmax_(fmax_(maxdist, maxdist2), fmax_(-mindist, -mindist2));
    # 增加共面合并计数
    zinc_(Zcoplanarcentrum);
    # 执行共面合并操作
    qh_appendmergeset(qh, facet, neighbor, MRGcoplanar, mergedist, angle);
    # 输出调试信息到跟踪日志，记录共面合并的详细信息，包括面和顶点信息、距离、角度等
    trace2((qh, qh->ferr, 2099, "qh_test_nonsimplicial_merge: coplanar f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }
  # 返回 True，表示合并成功
  return True;
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_redundant_neighbors">-</a>

  qh_test_redundant_neighbors(qh, facet )
    append degenerate facet or its redundant neighbors to qh.degen_mergeset

  returns:
    bumps vertex_visit

  notes:
    called at end of qh_mergefacet(), qh_mergecycle_all(), and qh_renamevertex
    call before qh_test_degen_neighbors (MRGdegen are more likely to cause problems)
    a redundant neighbor's vertices is a subset of the facet's vertices
    with pinched and flipped facets, a redundant neighbor may have a wildly different normal

    see qh_merge_degenredundant() and qh_-_facet()

  design:
    if facet is degenerate
       appends facet to degen_mergeset
    else
       appends redundant neighbors of facet to degen_mergeset
*/
void qh_test_redundant_neighbors(qhT *qh, facetT *facet) {
  vertexT *vertex, **vertexp;
  facetT *neighbor, **neighborp;
  int size;

  // 输出跟踪信息，显示当前处理的 facet 编号和 vertex_visit 的值增加 1
  trace4((qh, qh->ferr, 4022, "qh_test_redundant_neighbors: test neighbors of f%d vertex_visit %d\n",
          facet->id, qh->vertex_visit+1));
  
  // 如果 facet 的邻居数小于凸壳的维度，将 facet 添加到 degen_mergeset 中，并输出相应的跟踪信息
  if ((size= qh_setsize(qh, facet->neighbors)) < qh->hull_dim) {
    qh_appendmergeset(qh, facet, facet, MRGdegen, 0.0, 1.0);
    trace2((qh, qh->ferr, 2017, "qh_test_redundant_neighbors: f%d is degenerate with %d neighbors.\n", facet->id, size));
  } else {
    // 增加 vertex_visit 的值
    qh->vertex_visit++;
    
    // 标记 facet 的所有顶点的 visitid 为当前的 vertex_visit
    FOREACHvertex_(facet->vertices)
      vertex->visitid= qh->vertex_visit;
    
    // 遍历 facet 的每一个邻居
    FOREACHneighbor_(facet) {
      // 如果邻居是可见的，输出错误信息并退出程序
      if (neighbor->visible) {
        qh_fprintf(qh, qh->ferr, 6360, "qhull internal error (qh_test_redundant_neighbors): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
          facet->id, neighbor->id);
        qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
      }
      
      // 如果邻居是 degenerate、redundant 或 dupridge，则继续下一个邻居
      if (neighbor->degenerate || neighbor->redundant || neighbor->dupridge)
        continue;
      
      // 如果 facet 是 flipped 的，并且邻居不是 flipped 的，则继续下一个邻居
      if (facet->flipped && !neighbor->flipped)
        continue;
      
      // 检查邻居的顶点是否都被 visitid 标记为当前 vertex_visit
      FOREACHvertex_(neighbor->vertices) {
        if (vertex->visitid != qh->vertex_visit)
          break;
      }
      
      // 如果所有邻居的顶点都被标记了 visitid，则将邻居添加到 degen_mergeset 中，并输出相应的跟踪信息
      if (!vertex) {
        qh_appendmergeset(qh, neighbor, facet, MRGredundant, 0.0, 1.0);
        trace2((qh, qh->ferr, 2018, "qh_test_redundant_neighbors: f%d is contained in f%d.  merge\n", neighbor->id, facet->id));
      }
    }
  }
} /* test_redundant_neighbors */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_vneighbors">-</a>

  qh_test_vneighbors(qh)
    test vertex neighbors for convexity
    tests all facets on qh.newfacet_list

  returns:
    true if non-convex vneighbors appended to qh.facet_mergeset
    initializes vertex neighbors if needed

  notes:
    called by qh_all_merges from qh_postmerge if qh.TESTvneighbors ('Qv')

*/
    # 假设所有的面片邻居已经被测试过
    # 这可能会消耗大量资源
    # 这并不保证所有的中心都在所有面片下方
    # 但是这种情况可能性很小
    # 使用了qh.visit_id
    
    design:
      # 如果需要的话，构建顶点的邻居关系
      build vertex neighbors if necessary
      # 对于所有的新面片
      for all new facets
        # 对于所有的顶点
        for all vertices
          # 对于顶点的每一个未访问过的面片邻居
          for each unvisited facet neighbor of the vertex
            # 测试新的面片和邻居是否是凸的
            test new facet and neighbor for convexity
/* 
   qh_test_vneighbors(qhT *qh /* qh.newfacet_list */) {
   检查顶点邻居的凸性，返回是否有非凸的邻居
*/
boolT qh_test_vneighbors(qhT *qh /* qh.newfacet_list */) {
  facetT *newfacet, *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  int nummerges= 0;

  trace1((qh, qh->ferr, 1015, "qh_test_vneighbors: testing vertex neighbors for convexity\n"));
  如果尚未计算顶点的邻居关系，则进行计算
  if (!qh->VERTEXneighbors)
    qh_vertexneighbors(qh);
  将所有新面片的 'seen' 标志设置为 False
  FORALLnew_facets
    newfacet->seen= False;
  遍历所有新面片
  FORALLnew_facets {
    将当前面片标记为已访问
    newfacet->seen= True;
    更新访问标识
    newfacet->visitid= qh->visit_id++;
    遍历当前面片的邻居
    FOREACHneighbor_(newfacet)
      将当前面片的访问标识设置给邻居
      newfacet->visitid= qh->visit_id;
    遍历当前面片的顶点列表
    FOREACHvertex_(newfacet->vertices) {
      遍历当前顶点的邻居
      FOREACHneighbor_(vertex) {
        如果邻居已经被访问或者邻居的访问标识等于当前访问标识，则跳过
        if (neighbor->seen || neighbor->visitid == qh->visit_id)
          continue;
        调用 qh_test_appendmerge 函数，检查是否可以合并当前面片和邻居面片
        如果合并成功，增加合并计数器
        if (qh_test_appendmerge(qh, newfacet, neighbor, False)) /* ignores optimization for simplicial ridges */
          nummerges++;
      }
    }
  }
  将非凸的邻居合并计数器添加到统计数据中
  zadd_(Ztestvneighbor, nummerges);
  打印跟踪信息，指示发现了多少个非凸的顶点邻居
  trace1((qh, qh->ferr, 1016, "qh_test_vneighbors: found %d non-convex, vertex neighbors\n",
           nummerges));
  返回是否有非凸的邻居
  return (nummerges > 0);
} /* test_vneighbors */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="tracemerge">-</a>

  qh_tracemerge(qh, facet1, facet2 )
    print trace message after merge
*/
void qh_tracemerge(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype) {
  boolT waserror= False;
  const char *mergename;

#ifndef qh_NOtrace
  如果合并类型在合并类型数组的有效范围内，则使用对应的合并类型名称
  if(mergetype > 0 && mergetype <= sizeof(mergetypes))
    mergename= mergetypes[mergetype];
  否则使用默认的合并类型名称
  else
    mergename= mergetypes[MRGnone];
  如果当前正在进行详细跟踪，则打印 MERGED 消息
  if (qh->IStracing >= 4)
    qh_errprint(qh, "MERGED", facet2, NULL, NULL, NULL);
  如果合并的面片是跟踪面片，或者跟踪的顶点有新的面片，则打印跟踪信息
  if (facet2 == qh->tracefacet || (qh->tracevertex && qh->tracevertex->newfacet)) {
    打印跟踪信息，指示合并后的面片和顶点
    qh_fprintf(qh, qh->ferr, 8085, "qh_tracemerge: trace facet and vertex after merge of f%d into f%d type %d (%s), furthest p%d\n",
      facet1->id, facet2->id, mergetype, mergename, qh->furthest_id);
    如果跟踪的面片不是合并后的面片，则打印跟踪信息
    if (facet2 != qh->tracefacet)
      qh_errprint(qh, "TRACE", qh->tracefacet,
        (qh->tracevertex && qh->tracevertex->neighbors) ?
           SETfirstt_(qh->tracevertex->neighbors, facetT) : NULL,
        NULL, qh->tracevertex);
  }
  如果存在跟踪的顶点
  if (qh->tracevertex) {
    如果跟踪的顶点已被删除，则打印顶点已删除的消息
    if (qh->tracevertex->deleted)
      qh_fprintf(qh, qh->ferr, 8086, "qh_tracemerge: trace vertex deleted at furthest p%d\n",
            qh->furthest_id);
    否则检查顶点的有效性
    else
      qh_checkvertex(qh, qh->tracevertex, qh_ALL, &waserror);
  }
  如果存在跟踪的面片，并且该面片是有效的但不可见的，则检查面片的有效性
  if (qh->tracefacet && qh->tracefacet->normal && !qh->tracefacet->visible)
    qh_checkfacet(qh, qh->tracefacet, True /* newmerge */, &waserror);
#endif /* !qh_NOtrace */
  如果设置了频繁检查选项，或者详细跟踪开启，并且面片数量小于 500，则打印当前面片列表
  if (qh->CHECKfrequently || qh->IStracing >= 4) { /* can't check polygon here */
    如果详细跟踪开启，并且面片数量小于 500，则打印当前面片列表
    if (qh->IStracing >= 4 && qh->num_facets < 500) {
      qh_printlists(qh);
    }
    检查合并后的面片的有效性
    qh_checkfacet(qh, facet2, True /* newmerge */, &waserror);
  }
  如果出现错误，则退出程序
  if (waserror)
    qh_errexit(qh, qh_ERRqhull, NULL, NULL); /* erroneous facet logged by qh_checkfacet */
} /* tracemerge */
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="tracemerging">-</a>

  qh_tracemerging(qh)
    print trace message during POSTmerging

  returns:
    updates qh.mergereport

  notes:
    called from qh_mergecycle() and qh_mergefacet()

  see:
    qh_buildtracing()
*/
void qh_tracemerging(qhT *qh) {
  realT cpu;  // 实数类型的变量，用于存储 CPU 时间
  int total;  // 整数类型的变量，用于存储总数值
  time_t timedata;  // 时间类型的变量，用于存储时间数据
  struct tm *tp;  // 指向 tm 结构的指针，用于存储本地时间

  qh->mergereport= zzval_(Ztotmerge);  // 更新 qh.mergereport 的值为 Ztotmerge 的当前值
  time(&timedata);  // 获取当前时间
  tp= localtime(&timedata);  // 将当前时间转换为本地时间
  cpu= qh_CPUclock;  // 获取 qh_CPUclock 的值，并赋给 cpu
  cpu /= qh_SECticks;  // 将 cpu 值除以 qh_SECticks，得到 CPU 时间
  total= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);  // 计算总数值
  qh_fprintf(qh, qh->ferr, 8087, "\n\
At %d:%d:%d & %2.5g CPU secs, qhull has merged %d facets with max_outside %2.2g, min_vertex %2.2g.\n\
  The hull contains %d facets and %d vertices.\n",
      tp->tm_hour, tp->tm_min, tp->tm_sec, cpu, total, qh->max_outside, qh->min_vertex,
      qh->num_facets - qh->num_visible,
      qh->num_vertices-qh_setsize(qh, qh->del_vertices));
} /* tracemerging */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="updatetested">-</a>

  qh_updatetested(qh, facet1, facet2 )
    clear facet2->tested and facet1->ridge->tested for merge

  returns:
    deletes facet2->center unless it's already large
      if so, clears facet2->ridge->tested

  notes:
    only called by qh_mergefacet

  design:
    clear facet2->tested
    clear ridge->tested for facet1's ridges
    if facet2 has a centrum
      if facet2 is large
        set facet2->keepcentrum
      else if facet2 has 3 vertices due to many merges, or not large and post merging
        clear facet2->keepcentrum
      unless facet2->keepcentrum
        clear facet2->center to recompute centrum later
        clear ridge->tested for facet2's ridges
*/
void qh_updatetested(qhT *qh, facetT *facet1, facetT *facet2) {
  ridgeT *ridge, **ridgep;  // ridgeT 结构的指针和指针数组
  int size;  // 整数类型的变量，用于存储大小

  facet2->tested= False;  // 将 facet2->tested 设置为 False
  FOREACHridge_(facet1->ridges)  // 遍历 facet1 的所有边缘
    ridge->tested= False;  // 将每个边缘的 tested 属性设置为 False
  if (!facet2->center)  // 如果 facet2 没有 center
    return;  // 直接返回
  size= qh_setsize(qh, facet2->vertices);  // 计算 facet2 的顶点集合大小
  if (!facet2->keepcentrum) {  // 如果 facet2 没有 keepcentrum
    if (size > qh->hull_dim + qh_MAXnewcentrum) {  // 如果 size 大于 hull_dim + qh_MAXnewcentrum
      facet2->keepcentrum= True;  // 设置 facet2->keepcentrum 为 True
      zinc_(Zwidevertices);  // 增加 Zwidevertices 的计数
    }
  }else if (size <= qh->hull_dim + qh_MAXnewcentrum) {
    /* center and keepcentrum was set */
    if (size == qh->hull_dim || qh->POSTmerging)
      facet2->keepcentrum= False;  // 如果 size 等于 hull_dim 或者 POSTmerging 为真，则设置 facet2->keepcentrum 为 False，需要重新计算 centrum
  }
  if (!facet2->keepcentrum) {  // 如果 facet2 没有 keepcentrum
    qh_memfree(qh, facet2->center, qh->normal_size);  // 释放 facet2 的 center 内存
    facet2->center= NULL;  // 将 facet2 的 center 设置为 NULL
    FOREACHridge_(facet2->ridges)  // 遍历 facet2 的所有边缘
      ridge->tested= False;  // 将每个边缘的 tested 属性设置为 False
  }
} /* updatetested */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="vertexridges">-</a>

  qh_vertexridges(qh, vertex, allneighbors )
    return temporary set of ridges adjacent to a vertex
    vertex->neighbors defined (qh_vertexneighbors)

  notes:
    uses qh.visit_id
*/
    does not include implicit ridges for simplicial facets
    # 不包括对简单面片的隐式边缘

    skips last neighbor, unless allneighbors.  For new facets, the last neighbor shares ridges with adjacent neighbors
    # 跳过最后一个邻居，除非使用了 allneighbors 参数。对于新的面片，最后一个邻居与相邻邻居共享边缘

    if the last neighbor is not simplicial, it will have ridges for its simplicial neighbors
    # 如果最后一个邻居不是简单面片，它将具有与其简单面片邻居相关的边缘

    Use allneighbors when a new cone is attached to an existing convex hull
    # 当将新的锥体附加到现有凸壳时使用 allneighbors 参数

    similar to qh_neighbor_vertices
    # 类似于 qh_neighbor_vertices 函数

  design:
    for each neighbor of vertex
      add ridges that include the vertex to ridges
    # 设计：
    # 对于顶点的每个邻居，
    # 将包含该顶点的边缘添加到边缘集合中
/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="vertexridges">-</a>

  setT *qh_vertexridges(qhT *qh, vertexT *vertex, boolT allneighbors)
    compute ridges for a vertex
    allneighbors if True, includes all neighbors

  returns:
    temporary set of ridges for the vertex

  notes:
    clears vertex->visitid for each neighbor
    uses qh->visit_id for processing neighbors

  design:
    initialize ridges as a temporary set
    update visit_id for vertex neighbors
    for each neighbor of the vertex
      if allneighbors is True or neighbor has not been visited
        compute ridges with qh_vertexridges_facet()
    if tracing or statistics printing is enabled
      update statistics and trace the number of found ridges
*/
setT *qh_vertexridges(qhT *qh, vertexT *vertex, boolT allneighbors) {
  facetT *neighbor, **neighborp;
  setT *ridges= qh_settemp(qh, qh->TEMPsize);
  int size;

  qh->visit_id += 2;  /* visit_id for vertex neighbors, visit_id-1 for facets of visited ridges */
  FOREACHneighbor_(vertex)
    neighbor->visitid= qh->visit_id;
  FOREACHneighbor_(vertex) {
    if (*neighborp || allneighbors)   /* no new ridges in last neighbor */
      qh_vertexridges_facet(qh, vertex, neighbor, &ridges);
  }
  if (qh->PRINTstatistics || qh->IStracing) {
    size= qh_setsize(qh, ridges);
    zinc_(Zvertexridge);
    zadd_(Zvertexridgetot, size);
    zmax_(Zvertexridgemax, size);
    trace3((qh, qh->ferr, 3011, "qh_vertexridges: found %d ridges for v%d\n",
             size, vertex->id));
  }
  return ridges;
} /* vertexridges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="vertexridges_facet">-</a>

  qh_vertexridges_facet(qhT *qh, vertexT *vertex, facetT *facet, setT **ridges )
    add adjacent ridges for vertex in facet
    neighbor->visitid==qh.visit_id if it hasn't been visited

  returns:
    ridges updated
    sets facet->visitid to qh.visit_id-1

  design:
    for each ridge of facet
      if ridge of visited neighbor (i.e., unprocessed)
        if vertex in ridge
          append ridge
    mark facet processed
*/
void qh_vertexridges_facet(qhT *qh, vertexT *vertex, facetT *facet, setT **ridges) {
  ridgeT *ridge, **ridgep;
  facetT *neighbor;
  int last_i= qh->hull_dim-2;
  vertexT *second, *last;

  FOREACHridge_(facet->ridges) {
    neighbor= otherfacet_(ridge, facet);
    if (neighbor->visitid == qh->visit_id) {
      if (SETfirst_(ridge->vertices) == vertex) {
        qh_setappend(qh, ridges, ridge);
      }else if (last_i > 2) {
        second= SETsecondt_(ridge->vertices, vertexT);
        last= SETelemt_(ridge->vertices, last_i, vertexT);
        if (second->id >= vertex->id && last->id <= vertex->id) { /* vertices inverse sorted by id */
          if (second == vertex || last == vertex)
            qh_setappend(qh, ridges, ridge);
          else if (qh_setin(ridge->vertices, vertex))
            qh_setappend(qh, ridges, ridge);
        }
      }else if (SETelem_(ridge->vertices, last_i) == vertex
          || (last_i > 1 && SETsecond_(ridge->vertices) == vertex)) {
        qh_setappend(qh, ridges, ridge);
      }
    }
  }
  facet->visitid= qh->visit_id-1;
} /* vertexridges_facet */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="willdelete">-</a>

  qh_willdelete(qhT *qh, facetT *facet, replace )
    moves facet to visible list for qh_deletevisible
    sets facet->f.replace to replace (may be NULL)
    clears f.ridges and f.neighbors -- no longer valid

  returns:
    bumps qh.num_visible
*/
void qh_willdelete(qhT *qh, facetT *facet, facetT *replace) {

  // 输出跟踪信息到日志，描述将要删除的面 facet 的操作，包括移动到可见列表，设置其替换项 replace，并清除邻居和边缘信息
  trace4((qh, qh->ferr, 4081, "qh_willdelete: move f%d to visible list, set its replacement as f%d, and clear f.neighbors and f.ridges\n", facet->id, getid_(replace)));

  // 检查 qh.visible_list 和 qh.newfacet_list 的状态，若 qh.visible_list 为空且 qh.newfacet_list 不为空，则报错并退出
  if (!qh->visible_list && qh->newfacet_list) {
    qh_fprintf(qh, qh->ferr, 6378, "qhull internal error (qh_willdelete): expecting qh.visible_list at before qh.newfacet_list f%d.   Got NULL\n",
        qh->newfacet_list->id);
    qh_errexit2(qh, qh_ERRqhull, NULL, NULL);
  }

  // 从 qh.facet_list 中移除 facet
  qh_removefacet(qh, facet);

  // 将 facet 添加到 qh.visible_list 的头部
  qh_prependfacet(qh, facet, &qh->visible_list);

  // 增加可见面的计数
  qh->num_visible++;

  // 设置 facet 的可见性为真
  facet->visible= True;

  // 设置 facet 的替换项为 replace
  facet->f.replace= replace;

  // 若 facet 的边缘信息存在，则将第一个边缘设置为 NULL
  if (facet->ridges)
    SETfirst_(facet->ridges)= NULL;

  // 若 facet 的邻居信息存在，则将第一个邻居设置为 NULL
  if (facet->neighbors)
    SETfirst_(facet->neighbors)= NULL;
} /* willdelete */

#else /* qh_NOmerge */

// 下面的函数都是空函数，用于在不支持面合并时（qh_NOmerge）避免编译错误

void qh_all_vertexmerges(qhT *qh, int apexpointid, facetT *facet, facetT **retryfacet) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(apexpointid)
  QHULL_UNUSED(facet)
  QHULL_UNUSED(retryfacet)
}

void qh_premerge(qhT *qh, int apexpointid, realT maxcentrum, realT maxangle) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(apexpointid)
  QHULL_UNUSED(maxcentrum)
  QHULL_UNUSED(maxangle)
}

void qh_postmerge(qhT *qh, const char *reason, realT maxcentrum, realT maxangle,
                      boolT vneighbors) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(reason)
  QHULL_UNUSED(maxcentrum)
  QHULL_UNUSED(maxangle)
  QHULL_UNUSED(vneighbors)
}

void qh_checkdelfacet(qhT *qh, facetT *facet, setT *mergeset) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(facet)
  QHULL_UNUSED(mergeset)
}

void qh_checkdelridge(qhT *qh /* qh.visible_facets, vertex_mergeset */) {
  QHULL_UNUSED(qh)
}

boolT qh_checkzero(qhT *qh, boolT testall) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(testall)
  
  return True;
}

void qh_freemergesets(qhT *qh) {
  QHULL_UNUSED(qh)
}

void qh_initmergesets(qhT *qh) {
  QHULL_UNUSED(qh)
}

void qh_merge_pinchedvertices(qhT *qh, int apexpointid /* qh.newfacet_list */) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(apexpointid)
}

#endif /* qh_NOmerge */
```