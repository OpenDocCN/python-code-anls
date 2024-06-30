# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\poly_r.c`

```
/*<html><pre>  -<a                             href="qh-poly_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly_r.c
   implements polygons and simplices

   see qh-poly_r.htm, poly_r.h and libqhull_r.h

   infrequent code is in poly2_r.c
   (all but top 50 and their callers 12/3/95)

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/poly_r.c#7 $$Change: 2705 $
   $DateTime: 2019/06/26 16:34:45 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*======== functions in alphabetical order ==========*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="appendfacet">-</a>

  qh_appendfacet(qh, facet )
    appends facet to end of qh.facet_list,

  returns:
    updates qh.newfacet_list, facet_next, facet_list
    increments qh.numfacets

  notes:
    assumes qh.facet_list/facet_tail is defined (createsimplex)

  see:
    qh_removefacet()

*/
void qh_appendfacet(qhT *qh, facetT *facet) {
  facetT *tail= qh->facet_tail;

  if (tail == qh->newfacet_list) {
    qh->newfacet_list= facet;
    if (tail == qh->visible_list) /* visible_list is at or before newfacet_list */
      qh->visible_list= facet;
  }
  if (tail == qh->facet_next)
    qh->facet_next= facet;
  facet->previous= tail->previous;
  facet->next= tail;
  if (tail->previous)
    tail->previous->next= facet;
  else
    qh->facet_list= facet;
  tail->previous= facet;
  qh->num_facets++;
  trace4((qh, qh->ferr, 4044, "qh_appendfacet: append f%d to facet_list\n", facet->id));
} /* appendfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="appendvertex">-</a>

  qh_appendvertex(qh, vertex )
    appends vertex to end of qh.vertex_list,

  returns:
    sets vertex->newfacet
    updates qh.vertex_list, newvertex_list
    increments qh.num_vertices

  notes:
    assumes qh.vertex_list/vertex_tail is defined (createsimplex)

*/
void qh_appendvertex(qhT *qh, vertexT *vertex) {
  vertexT *tail= qh->vertex_tail;

  if (tail == qh->newvertex_list)
    qh->newvertex_list= vertex;
  vertex->newfacet= True;
  vertex->previous= tail->previous;
  vertex->next= tail;
  if (tail->previous)
    tail->previous->next= vertex;
  else
    qh->vertex_list= vertex;
  tail->previous= vertex;
  qh->num_vertices++;
  trace4((qh, qh->ferr, 4045, "qh_appendvertex: append v%d to qh.newvertex_list and set v.newfacet\n", vertex->id));
} /* appendvertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="attachnewfacets">-</a>

  qh_attachnewfacets(qh)
    attach horizon facets to new facets in qh.newfacet_list
    newfacets have neighbor and ridge links to horizon but not vice versa

  returns:
    clears qh.NEWtentative
    set qh.NEWfacets
    horizon facets linked to new facets
      ridges changed from visible facets to new facets
      simplicial ridges deleted

*/
    qh.visible_list, no ridges valid
    facet->f.replace is a newfacet (if any)

notes:
    used for qh.NEWtentative, otherwise see qh_makenew_nonsimplicial and qh_makenew_simplicial
    qh_delridge_merge not needed (as tested by qh_checkdelridge)

design:
    delete interior ridges and neighbor sets by
      for each visible, non-simplicial facet
        for each ridge
          if last visit or if neighbor is simplicial
            if horizon neighbor
              delete ridge for horizon's ridge set
            delete ridge
        erase neighbor set
    attach horizon facets and new facets by
      for all new facets
        if corresponding horizon facet is simplicial
          locate corresponding visible facet {may be more than one}
          link visible facet to new facet
          replace visible facet with new facet in horizon
        else it is non-simplicial
          for all visible neighbors of the horizon facet
            link visible neighbor to new facet
            delete visible neighbor from horizon facet
          append new facet to horizon's neighbors
          the first ridge of the new facet is the horizon ridge
          link the new facet into the horizon ridge


注释：
这段代码段描述了一种算法的设计和实现细节，用于处理几何计算中的边界面问题。具体来说，它包括两个主要部分：

1. 删除内部的边和邻居集合：
   - 遍历每个可见的非单纯面（即非简单面）
   - 对于每条边，根据最后的访问或者邻居是否是单纯面来判断是否删除边
   - 如果是水平面的邻居，则删除水平面的边集合中的边
   - 最后删除边本身，并清空邻居集合

2. 连接水平面和新面：
   - 针对所有的新面进行处理
   - 如果对应的水平面是单纯面，找到对应的可见面，并将其连接到新面上，然后在水平面中用新面替换可见面
   - 如果水平面不是单纯面，遍历水平面的所有可见邻居，并将它们连接到新面上，然后将新面添加到水平面的邻居集合中，新面的第一条边是水平面的边，同时将新面连接到水平面的边上
/*
void qh_attachnewfacets(qhT *qh /* qh.visible_list, qh.newfacet_list */) {
  // 定义变量
  facetT *newfacet= NULL, *neighbor, **neighborp, *horizon, *visible;
  ridgeT *ridge, **ridgep;

  // 打印跟踪信息
  trace3((qh, qh->ferr, 3012, "qh_attachnewfacets: delete interior ridges\n"));
  
  // 如果启用了频繁检查选项，检查是否需要删除内部边
  if (qh->CHECKfrequently) {
    qh_checkdelridge(qh);
  }
  
  // 增加访问标识
  qh->visit_id++;
  
  // 对所有可见面进行迭代
  FORALLvisible_facets {
    visible->visitid= qh->visit_id;  // 设置可见面的访问标识为当前访问标识
    if (visible->ridges) {
      // 遍历可见面的所有边
      FOREACHridge_(visible->ridges) {
        neighbor= otherfacet_(ridge, visible);  // 获取相邻的面
        // 如果相邻面的访问标识与当前标识相同，或者相邻面不可见且是单纯的
        if (neighbor->visitid == qh->visit_id
            || (!neighbor->visible && neighbor->simplicial)) {
          if (!neighbor->visible)  // 如果相邻面不可见，删除单纯地平面边
            qh_setdel(neighbor->ridges, ridge);
          qh_delridge(qh, ridge); // 第二次访问时删除边
        }
      }
    }
  }
  
  // 打印跟踪信息
  trace1((qh, qh->ferr, 1017, "qh_attachnewfacets: attach horizon facets to new facets\n"));
  
  // 对所有新面进行迭代
  FORALLnew_facets {
    horizon= SETfirstt_(newfacet->neighbors, facetT);  // 获取新面的第一个相邻面作为视野面
    if (horizon->simplicial) {
      visible= NULL;
      // 遍历视野面的所有相邻面，可能有多个视野边
      FOREACHneighbor_(horizon) {
        if (neighbor->visible) {
          if (visible) {
            // 如果找到可见面，将其替换为新面
            if (qh_setequal_skip(newfacet->vertices, 0, horizon->vertices,
                                  SETindex_(horizon->neighbors, neighbor))) {
              visible= neighbor;
              break;
            }
          } else
            visible= neighbor;
        }
      }
      // 如果找到可见面，则将其替换为新面
      if (visible) {
        visible->f.replace= newfacet;
        qh_setreplace(qh, horizon->neighbors, visible, newfacet);
      } else {
        // 如果未找到可见面，输出错误信息并退出
        qh_fprintf(qh, qh->ferr, 6102, "qhull internal error (qh_attachnewfacets): could not find visible facet for horizon f%d of newfacet f%d\n",
                 horizon->id, newfacet->id);
        qh_errexit2(qh, qh_ERRqhull, horizon, newfacet);
      }
    } else { // 非单纯面，具有新面的边
      // 遍历视野面的所有相邻面，可能适用于多个新面
      FOREACHneighbor_(horizon) {
        if (neighbor->visible) {
          neighbor->f.replace= newfacet;
          qh_setdelnth(qh, horizon->neighbors, SETindex_(horizon->neighbors, neighbor));
          neighborp--; // 重复处理
        }
      }
      // 将新面添加到视野面的相邻面列表中
      qh_setappend(qh, &horizon->neighbors, newfacet);
      ridge= SETfirstt_(newfacet->ridges, ridgeT);
      // 设置新面的边界
      if (ridge->top == horizon) {
        ridge->bottom= newfacet;
        ridge->simplicialbot= True;
      } else {
        ridge->top= newfacet;
        ridge->simplicialtop= True;
      }
    }
  } /* newfacets */
  
  // 打印跟踪信息
  trace4((qh, qh->ferr, 4094, "qh_attachnewfacets: clear f.ridges and f.neighbors for visible facets, may become invalid before qh_deletevisible\n"));
  
  // 清空所有可见面的边和相邻面
  FORALLvisible_facets {
    if (visible->ridges)
      SETfirst_(visible->ridges)= NULL; 
    SETfirst_(visible->neighbors)= NULL;
  }
  
  // 重置标志
  qh->NEWtentative= False;
  qh->NEWfacets= True;
  
  // 如果打印统计信息选项被启用
  if (qh->PRINTstatistics) {
    // 对所有可见面进行迭代
    FORALLvisible_facets {
      // 如果替换标志未设置，则增加计数器
      if (!visible->f.replace)
        zinc_(Zinsidevisible);
    }
  }


注释：


    # 这部分代码看起来可能属于一个函数或控制流结构的结尾
    }
    # 外层的代码块结束
  }
  # 最外层的代码块结束


这段代码示例中，代码块的缩进结构显示可能是一个函数或其他语言控制结构（如循环或条件语句）的结尾。
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkflipped">-</a>

  qh_checkflipped(qh, facet, dist, allerror )
    checks facet orientation to interior point

    if allerror set,
      tests against -qh.DISTround
    else
      tests against 0.0 since tested against -qh.DISTround before

  returns:
    False if it flipped orientation (sets facet->flipped)
    distance if non-NULL

  notes:
    called by qh_setfacetplane, qh_initialhull, and qh_checkflipped_all
*/
boolT qh_checkflipped(qhT *qh, facetT *facet, realT *distp, boolT allerror) {
  realT dist;

  // 如果 facet 已经翻转过并且 distp 为 NULL，则返回 False
  if (facet->flipped && !distp)
    return False;
  
  // 增加距离检查的计数
  zzinc_(Zdistcheck);

  // 计算 facet 平面到内部点的距离
  qh_distplane(qh, qh->interior_point, facet, &dist);

  // 如果 distp 不为 NULL，则将计算得到的距离存入 distp
  if (distp)
    *distp = dist;

  // 根据 allerror 的设置进行距离测试
  if ((allerror && dist >= -qh->DISTround) || (!allerror && dist > 0.0)) {
    // 如果距离条件不符合，则标记 facet 已经翻转，并输出距离信息
    facet->flipped = True;
    trace0((qh, qh->ferr, 19, "qh_checkflipped: facet f%d flipped, allerror? %d, distance= %6.12g during p%d\n",
              facet->id, allerror, dist, qh->furthest_id));
    
    // 如果当前的面数大于凸壳维度加一，重新启动 joggle 进程以修复翻转的面
    if (qh->num_facets > qh->hull_dim + 1) {
      zzinc_(Zflippedfacets);
      qh_joggle_restart(qh, "flipped facet");
    }
    return False;
  }
  
  // 距离条件符合，则返回 True
  return True;
} /* checkflipped */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="delfacet">-</a>

  qh_delfacet(qh, facet )
    removes facet from facet_list and frees up its memory

  notes:
    assumes vertices and ridges already freed or referenced elsewhere
*/
void qh_delfacet(qhT *qh, facetT *facet) {
  void **freelistp; /* used if !qh_NOmem by qh_memfree_() */

  // 输出删除 facet 的跟踪信息
  trace3((qh, qh->ferr, 3057, "qh_delfacet: delete f%d\n", facet->id));
  
  // 在检查频率设置为高或者输出验证时，检查 facet 的删除操作
  if (qh->CHECKfrequently || qh->VERIFYoutput) { 
    if (!qh->NOerrexit) {
      qh_checkdelfacet(qh, facet, qh->facet_mergeset);
      qh_checkdelfacet(qh, facet, qh->degen_mergeset);
      qh_checkdelfacet(qh, facet, qh->vertex_mergeset);
    }
  }
  
  // 如果 facet 是 tracefacet，则将其置为 NULL
  if (facet == qh->tracefacet)
    qh->tracefacet = NULL;
  
  // 如果 facet 是 GOODclosest，则将其置为 NULL
  if (facet == qh->GOODclosest)
    qh->GOODclosest = NULL;
  
  // 从 facet_list 中移除 facet
  qh_removefacet(qh, facet);
  
  // 如果非三角共面或保留了中心，则释放 normal 和 center
  if (!facet->tricoplanar || facet->keepcentrum) {
    qh_memfree_(qh, facet->normal, qh->normal_size, freelistp);
    if (qh->CENTERtype == qh_ASvoronoi) {
      qh_memfree_(qh, facet->center, qh->center_size, freelistp);
    } else { // AScentrum
      qh_memfree_(qh, facet->center, qh->normal_size, freelistp);
    }
  }
  
  // 释放 neighbors 集合
  qh_setfree(qh, &(facet->neighbors));
  
  // 如果存在 ridges，则释放 ridges 集合
  if (facet->ridges)
    qh_setfree(qh, &(facet->ridges));
  
  // 释放 vertices 集合
  qh_setfree(qh, &(facet->vertices));
  
  // 如果存在 outsideset，则释放 outsideset 集合
  if (facet->outsideset)
    qh_setfree(qh, &(facet->outsideset));
  
  // 如果存在 coplanarset，则释放 coplanarset 集合
  if (facet->coplanarset)
    qh_setfree(qh, &(facet->coplanarset));
  
  // 最后释放 facet 本身的内存空间
  qh_memfree_(qh, facet, (int)sizeof(facetT), freelistp);
} /* delfacet */
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="deletevisible">-</a>

  qh_deletevisible()
    delete visible facets and vertices

  returns:
    deletes each facet and removes from facetlist
    deletes vertices on qh.del_vertices and ridges in qh.del_ridges
    at exit, qh.visible_list empty (== qh.newfacet_list)

  notes:
    called by qh_all_vertexmerges, qh_addpoint, and qh_qhull
    ridges already deleted or moved elsewhere
    deleted vertices on qh.del_vertices
    horizon facets do not reference facets on qh.visible_list
    new facets in qh.newfacet_list
    uses   qh.visit_id;
*/
void qh_deletevisible(qhT *qh /* qh.visible_list */) {
  facetT *visible, *nextfacet;
  vertexT *vertex, **vertexp;
  int numvisible= 0, numdel= qh_setsize(qh, qh->del_vertices);

  trace1((qh, qh->ferr, 1018, "qh_deletevisible: delete %d visible facets and %d vertices\n",
         qh->num_visible, numdel));
  for (visible=qh->visible_list; visible && visible->visible;
                visible= nextfacet) { /* deleting current */
    nextfacet= visible->next;
    numvisible++;
    qh_delfacet(qh, visible);  /* f.ridges deleted or moved elsewhere, deleted f.vertices on qh.del_vertices */
  }
  if (numvisible != qh->num_visible) {
    qh_fprintf(qh, qh->ferr, 6103, "qhull internal error (qh_deletevisible): qh->num_visible %d is not number of visible facets %d\n",
             qh->num_visible, numvisible);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh->num_visible= 0;
  zadd_(Zvisfacettot, numvisible);
  zmax_(Zvisfacetmax, numvisible);
  zzadd_(Zdelvertextot, numdel);
  zmax_(Zdelvertexmax, numdel);
  FOREACHvertex_(qh->del_vertices)
    qh_delvertex(qh, vertex);
  qh_settruncate(qh, qh->del_vertices, 0);
} /* deletevisible */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="facetintersect">-</a>

  qh_facetintersect(qh, facetA, facetB, skipa, skipB, prepend )
    return vertices for intersection of two simplicial facets
    may include 1 prepended entry (if more, need to settemppush)

  returns:
    returns set of qh.hull_dim-1 + prepend vertices
    returns skipped index for each test and checks for exactly one

  notes:
    does not need settemp since set in quick memory

  see also:
    qh_vertexintersect and qh_vertexintersect_new
    use qh_setnew_delnthsorted to get nth ridge (no skip information)

  design:
    locate skipped vertex by scanning facet A's neighbors
    locate skipped vertex by scanning facet B's neighbors
    intersect the vertex sets
*/
setT *qh_facetintersect(qhT *qh, facetT *facetA, facetT *facetB,
                         int *skipA,int *skipB, int prepend) {
  setT *intersect;
  int dim= qh->hull_dim, i, j;
  facetT **neighborsA, **neighborsB;

  neighborsA= SETaddr_(facetA->neighbors, facetT);
  neighborsB= SETaddr_(facetB->neighbors, facetT);
  i= j= 0;
  if (facetB == *neighborsA++)
    *skipA= 0;
  else if (facetB == *neighborsA++)
    *skipA= 1;
  else if (facetB == *neighborsA++)
    *skipA= 2;
  else {
    for (i=3; i < dim; i++) {
      if (facetB == *neighborsA++) {
        *skipA= i;
        break;
      }
    }
  }
  if (facetA == *neighborsB++)
    *skipB= 0;
  else if (facetA == *neighborsB++)
    *skipB= 1;
  else if (facetA == *neighborsB++)
    *skipB= 2;
  else {
    for (j=3; j < dim; j++) {
      if (facetA == *neighborsB++) {
        *skipB= j;
        break;
      }
    }
  }
  if (i >= dim || j >= dim) {
    qh_fprintf(qh, qh->ferr, 6104, "qhull internal error (qh_facetintersect): f%d or f%d not in other's neighbors\n",
            facetA->id, facetB->id);
    qh_errexit2(qh, qh_ERRqhull, facetA, facetB);
  }
  intersect= qh_setnew_delnthsorted(qh, facetA->vertices, qh->hull_dim, *skipA, prepend);
  trace4((qh, qh->ferr, 4047, "qh_facetintersect: f%d skip %d matches f%d skip %d\n",
          facetA->id, *skipA, facetB->id, *skipB));
  return(intersect);



*skipA= 0;  // 设置 skipA 的初始值为 0
else if (facetB == *neighborsA++)
    *skipA= 1;  // 如果 facetB 等于 neighborsA 指向的值，设置 skipA 为 1
else if (facetB == *neighborsA++)
    *skipA= 2;  // 如果 facetB 等于 neighborsA 指向的值，设置 skipA 为 2
else {
    for (i=3; i < dim; i++) {
      if (facetB == *neighborsA++) {
        *skipA= i;  // 遍历 neighborsA 直到找到 facetB，设置 skipA 为对应的 i 值
        break;
      }
    }
  }
if (facetA == *neighborsB++)
    *skipB= 0;  // 如果 facetA 等于 neighborsB 指向的值，设置 skipB 为 0
else if (facetA == *neighborsB++)
    *skipB= 1;  // 如果 facetA 等于 neighborsB 指向的值，设置 skipB 为 1
else if (facetA == *neighborsB++)
    *skipB= 2;  // 如果 facetA 等于 neighborsB 指向的值，设置 skipB 为 2
else {
    for (j=3; j < dim; j++) {
      if (facetA == *neighborsB++) {
        *skipB= j;  // 遍历 neighborsB 直到找到 facetA，设置 skipB 为对应的 j 值
        break;
      }
    }
  }
if (i >= dim || j >= dim) {
    qh_fprintf(qh, qh->ferr, 6104, "qhull internal error (qh_facetintersect): f%d or f%d not in other's neighbors\n",
            facetA->id, facetB->id);
    qh_errexit2(qh, qh_ERRqhull, facetA, facetB);  // 如果 i 或 j 超出范围，输出错误信息并退出程序
  }
intersect= qh_setnew_delnthsorted(qh, facetA->vertices, qh->hull_dim, *skipA, prepend);  // 根据 skipA 创建一个新的集合 intersect
trace4((qh, qh->ferr, 4047, "qh_facetintersect: f%d skip %d matches f%d skip %d\n",
          facetA->id, *skipA, facetB->id, *skipB));  // 输出追踪信息
return(intersect);  // 返回创建的 intersect 集合
/* facetintersect */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="gethash">-</a>

  qh_gethash(qh, hashsize, set, size, firstindex, skipelem )
    return hashvalue for a set with firstindex and skipelem

  notes:
    returned hash is in [0,hashsize)
    assumes at least firstindex+1 elements
    assumes skipelem is NULL, in set, or part of hash

    hashes memory addresses which may change over different runs of the same data
    using sum for hash does badly in high d
*/
int qh_gethash(qhT *qh, int hashsize, setT *set, int size, int firstindex, void *skipelem) {
  void **elemp= SETelemaddr_(set, firstindex, void);
  ptr_intT hash= 0, elem;
  unsigned int uresult;
  int i;
#ifdef _MSC_VER                   /* Microsoft Visual C++ -- warn about 64-bit issues */
#pragma warning( push)            /* WARN64 -- ptr_intT holds a 64-bit pointer */
#pragma warning( disable : 4311)  /* 'type cast': pointer truncation from 'void*' to 'ptr_intT' */
#endif

  switch (size-firstindex) {
  case 1:
    hash= (ptr_intT)(*elemp) - (ptr_intT) skipelem;   // 计算 hash 值，从第一个元素减去 skipelem 的指针值
    break;
  case 2:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] - (ptr_intT) skipelem;   // 计算 hash 值，第一个元素加上第二个元素的指针值，再减去 skipelem 的指针值
    break;
  case 3:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      - (ptr_intT) skipelem;   // 计算 hash 值，前三个元素的指针值之和，再减去 skipelem 的指针值
    break;
  case 4:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      + (ptr_intT)elemp[3] - (ptr_intT) skipelem;   // 计算 hash 值，前四个元素的指针值之和，再减去 skipelem 的指针值
    break;
  case 5:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      + (ptr_intT)elemp[3] + (ptr_intT)elemp[4] - (ptr_intT) skipelem;   // 计算 hash 值，前五个元素的指针值之和，再减去 skipelem 的指针值
    break;
  case 6:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      + (ptr_intT)elemp[3] + (ptr_intT)elemp[4]+ (ptr_intT)elemp[5]
      - (ptr_intT) skipelem;   // 计算 hash 值，前六个元素的指针值之和，再减去 skipelem 的指针值
    break;
  default:
    hash= 0;
    i= 3;
    do {     /* this is about 10% in 10-d */
      if ((elem= (ptr_intT)*elemp++) != (ptr_intT)skipelem) {
        hash ^= (elem << i) + (elem >> (32-i));   // 如果元素不等于 skipelem 的指针值，则将元素进行位运算后加入 hash
        i += 3;
        if (i >= 32)
          i -= 32;
      }
    }while (*elemp);   // 循环直到遍历完所有元素
    break;
  }
  if (hashsize<0) {
    qh_fprintf(qh, qh->ferr, 6202, "qhull internal error: negative hashsize %d passed to qh_gethash [poly_r.c]\n", hashsize);
    qh_errexit2(qh, qh_ERRqhull, NULL, NULL);   // 如果 hashsize 小于 0，输出错误信息并退出程序
  }
  uresult= (unsigned int)hash;
  uresult %= (unsigned int)hashsize;   // 计算最终的 hash 值
  /* result= 0; for debugging */
  return (int)uresult;   // 返回计算得到的 hash 值
#ifdef _MSC_VER
#pragma warning( pop)
#endif
} /* gethash */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="getreplacement">-</a>

  qh_getreplacement(qh, visible )
    get replacement for visible facet

  returns:
    valid facet from visible.replace (may be chained)
*/
facetT *qh_getreplacement(qhT *qh, facetT *visible) {
  unsigned int count= 0;

  facetT *result= visible;
  while (result && result->visible) {   // 循环直到找到一个不为空且可见的 facet
    result= result->f.replace;   // 获取 visible facet 的替换 facet
    count++;   // 计数器加一

  /* facetintersect */

  /*-<a                             href="qh-poly_r.htm#TOC"
    >-------------------------------</a><a name="gethash">-</a>

    qh_gethash(qh, hashsize, set, size, firstindex, skipelem )
      return hashvalue for a set with firstindex and skipelem

    notes:
      returned hash is in [0,hashsize)
      assumes at least firstindex+1 elements
      assumes skipelem is NULL, in set, or part of hash

      hashes memory addresses which may change over different runs of the same data
      using sum for hash does badly in high d
  */
  int qh_gethash(qhT *qh, int hashsize, setT *set, int size, int firstindex, void *skipelem) {
    void **elemp= SETelemaddr_(set, firstindex, void);
    ptr_intT hash= 0, elem;
    unsigned int uresult;
    int i;
  #ifdef _MSC_VER                   /* Microsoft Visual C++ -- warn about 64-bit issues */
  #pragma warning( push)            /* WARN64 -- ptr_intT holds a 64-bit pointer */
  #pragma warning( disable : 4311)  /* 'type cast': pointer truncation from 'void*' to 'ptr_intT' */
  #endif

    switch (size-firstindex) {
    case 1:
      hash= (ptr_intT)(*elemp) - (ptr_intT) skipelem;   // 计算 hash 值，从第一个元素减去 skipelem 的指针值
      break;
    case 2:
      hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] - (ptr_intT) skipelem;   // 计算 hash 值，第一个元素加上第二个元素的指针值，再减去 skipelem 的指针值
      break;
    case 3:
      hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
        - (ptr_intT) skipelem;   // 计算 hash 值，前三个元素的指针值之和，再减去 skipelem 的指针值
      break;
    case 4:
      hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
        + (ptr_intT)elemp[3] - (ptr_intT) skipelem;   // 计算 hash 值，前四个元素的指针值之和，再减去 skipelem 的指针值
      break;
    case 5:
      hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
        + (ptr_intT)elemp[3] + (ptr_intT)elemp[4] - (ptr_intT) skipelem;   // 计算 hash 值，前五个元素的指针值之和，再减去 skipelem 的指针值
      break;
    case 6:
      hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
        + (ptr_intT)elemp[3] + (ptr_intT)elemp[4]+ (ptr_intT)elemp[5]
        - (ptr_intT) skipelem;   // 计算 hash 值，前六个元素的指针值之和，再减去 skipelem 的指针值
      break;
    default:
      hash= 0;
      i= 3;
      do {     /* this is about 10% in 10-d */
        if ((elem= (ptr_intT)*elemp++) != (ptr_intT)skipelem) {
          hash ^= (elem << i) + (elem >> (32-i));   // 如果元素不等于 skipelem 的指针值，则将元素进行位运算后加入 hash
          i += 3;
          if (i >= 32)
            i -= 32;
        }
      }while (*elemp);   // 循环直到遍历完所有元素
      break;
    }
    if (hashsize<0) {
      qh_fprintf(qh, qh->ferr, 6202, "qhull internal error: negative hashsize %d passed to qh_gethash [poly_r.c]\n", hashsize);
      qh_errexit2(qh, qh_ERRqhull, NULL, NULL);   // 如果 hashsize 小于 0，输出错误信息并退出程序
    }
    uresult= (unsigned int)hash;
    uresult %= (unsigned int)hashsize;   // 计算最终的 hash 值
    /* result= 0; for debugging */
    return (int)uresult;   // 返回计算得到的 hash 值
  #ifdef _MSC_VER
  #pragma warning( pop)
  #endif
  } /* gethash */

  /*-<a                             href="qh-poly_r.htm#TOC"
    >-------------------------------</a><a name="getreplacement">-</a>

    qh_getreplacement(qh, visible )
      get replacement for visible facet

    returns:
      valid facet from visible.replace (may be chained)
  */
  facetT *qh_getreplacement(qhT *qh, facetT *visible) {
    unsigned int count= 0;

    facetT *result= visible;
    while (result && result->visible) {   // 循环直到找到一个不为空且可见的 facet
      result= result->f.replace;   // 获取 visible facet 的替换 facet
      count++;   // 计数器加一
    `
        # 如果计数器 count 增加后的值大于 qh 对象的 facet_id 属性
        if (count++ > qh->facet_id)
          # 调用 qh_infiniteloop 函数，传入 qh 对象和 visible 参数，处理无穷循环情况
          qh_infiniteloop(qh, visible);
      }
      # 返回结果
      return result;
}

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenewfacet">-</a>

  qh_makenewfacet(qh, vertices, toporient, horizon )
    creates a new facet from given vertices with optional topological orientation

  returns:
    returns the newly created facet
      adds the facet to qh.facet_list
      sets facet->vertices to vertices
      if horizon is provided, sets facet->neighbor to horizon (one-way link)
    updates qh.vertex_list with vertices
*/
facetT *qh_makenewfacet(qhT *qh, setT *vertices, boolT toporient, facetT *horizon) {
  facetT *newfacet;  // 新建的 facet 对象
  vertexT *vertex, **vertexp;

  // 更新 vertices 中的每个 vertex，确保它们的位置在列表末尾
  FOREACHvertex_(vertices) {
    if (!vertex->newfacet) {
      qh_removevertex(qh, vertex);  // 从当前位置删除 vertex
      qh_appendvertex(qh, vertex);  // 将 vertex 添加到列表末尾
    }
  }
  newfacet= qh_newfacet(qh);  // 创建一个新的 facet 对象
  newfacet->vertices= vertices;  // 设置 facet 的 vertices 属性为给定的 vertices
  if (toporient)
    newfacet->toporient= True;  // 如果指定了 toporient，则设置 facet 的 toporient 为 True
  if (horizon)
    qh_setappend(qh, &(newfacet->neighbors), horizon);  // 如果提供了 horizon，则将 horizon 添加到 facet 的 neighbors 属性中
  qh_appendfacet(qh, newfacet);  // 将新创建的 facet 添加到 qh.facet_list 中
  return(newfacet);  // 返回新创建的 facet 对象
} /* makenewfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenewplanes">-</a>

  qh_makenewplanes()
    为 qh.newfacet_list 中的 facets 创建新的超平面

  returns:
    所有的 facets 都有了超平面或者标记为合并
    如果 horizon 是共面的，则不创建超平面（将会合并）
    如果 qh.JOGGLEmax 存在，则更新 qh.min_vertex

  notes:
    对于 facet->mergehorizon 的 facets，facet->f.samecycle 已定义
*/
void qh_makenewplanes(qhT *qh /* qh.newfacet_list */) {
  facetT *newfacet;

  trace4((qh, qh->ferr, 4074, "qh_makenewplanes: make new hyperplanes for facets on qh.newfacet_list f%d\n",
    qh->newfacet_list->id));
  // 遍历所有的 new facets
  FORALLnew_facets {
    if (!newfacet->mergehorizon)
      qh_setfacetplane(qh, newfacet); /* 更新 Wnewvertexmax */
  }
  if (qh->JOGGLEmax < REALmax/2)
    minimize_(qh->min_vertex, -wwval_(Wnewvertexmax));
} /* makenewplanes */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenew_nonsimplicial">-</a>

  qh_makenew_nonsimplicial(qh, visible, apex, numnew )
    为可见 facet 的边缘创建新的 facets

  returns:
    第一个新创建的 facet，并根据需要增加 numnew
    如果 !qh->NEWtentative，则连接新的 facets
    对于简单可见的标记边缘邻居
    如果 (qh.NEWtentative)
      在 newfacet、horizon 和可见 facet 上标记边缘
    否则
      在 newfacet 和 horizon 之间标记边缘和邻居
      删除可见 facet 的边缘并清空其 f.neighbors

  notes:
    被 qh_makenewfacets 和 qh_triangulatefacet 调用
    如果 visible 已经处理过，则设置 qh.visit_id
    设置 neighbor->seen 以构建 f.samecycle
      假设所有的 'seen' 标志初始为 false
    qh_delridge_merge 不需要（由 qh_checkdelridge 在 qh_makenewfacets 中测试）

  design:
    # 对于可见面的每条边缘进行处理
    for each ridge of visible facet
      # 获取可见面的邻居面
      get neighbor of visible facet
      # 如果邻居面已经被处理过
      if neighbor was already processed
        # 删除这条边缘（稍后将删除所有可见面）
        delete the ridge (will delete all visible facets later)
      # 如果邻居面是一个地平线面
      if neighbor is a horizon facet
        # 创建一个新的面
        create a new facet
        # 如果邻居面与当前面共面
        if neighbor coplanar
          # 将新面添加到邻居面的同一循环中以便稍后合并
          adds newfacet to f.samecycle for later merging
        else
          # 更新邻居面的邻居集合
          (checks for non-simplicial facet with multiple ridges to visible facet)
        # 更新邻居面的边缘集合
        (checks for simplicial neighbor to non-simplicial visible facet)
        (deletes ridge if neighbor is simplicial)
/*
facetT *qh_makenew_nonsimplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew) {
  void **freelistp; /* used if !qh_NOmem by qh_memfree_() */
  ridgeT *ridge, **ridgep;
  facetT *neighbor, *newfacet= NULL, *samecycle;
  setT *vertices;
  boolT toporient;
  unsigned int ridgeid;

  FOREACHridge_(visible->ridges) {
    ridgeid= ridge->id;  // 获取当前ridge的id
    neighbor= otherfacet_(ridge, visible);  // 获取与当前ridge共享的相邻facet
    if (neighbor->visible) {  // 如果相邻facet可见
      if (!qh->NEWtentative) {  // 如果不是NEWtentative模式
        if (neighbor->visitid == qh->visit_id) {  // 如果相邻facet的visitid与当前visit_id相同
          if (qh->traceridge == ridge)
            qh->traceridge= NULL;  // 如果traceridge与当前ridge相同，则将其置空
          qh_setfree(qh, &(ridge->vertices));  /* 删除ridge的vertices集合 */
          qh_memfree_(qh, ridge, (int)sizeof(ridgeT), freelistp);  // 释放ridge内存
        }
      }
    }else {  // 如果相邻facet是一个horizon facet
      toporient= (ridge->top == visible);  // 判断ridge的顶部是否为当前visible facet
      vertices= qh_setnew(qh, qh->hull_dim); /* 创建一个新的setT对象，确保这是一个快速操作 */
      qh_setappend(qh, &vertices, apex);  // 将apex点添加到vertices集合中
      qh_setappend_set(qh, &vertices, ridge->vertices);  // 将ridge的vertices集合添加到vertices中
      newfacet= qh_makenewfacet(qh, vertices, toporient, neighbor);  // 创建一个新的facet
      (*numnew)++;  // 增加新facet计数
      if (neighbor->coplanarhorizon) {  // 如果相邻facet与当前facet是coplanar的
        newfacet->mergehorizon= True;  // 设置mergehorizon标志为True
        if (!neighbor->seen) {  // 如果相邻facet之前未被看见过
          newfacet->f.samecycle= newfacet;  // 将当前facet设置为同一个cycle的起点
          neighbor->f.newcycle= newfacet;  // 设置相邻facet的newcycle为当前facet
        }else {
          samecycle= neighbor->f.newcycle;
          newfacet->f.samecycle= samecycle->f.samecycle;  // 设置当前facet的samecycle
          samecycle->f.samecycle= newfacet;  // 更新相邻facet的samecycle
        }
      }
      if (qh->NEWtentative) {  // 如果是NEWtentative模式
        if (!neighbor->simplicial)
          qh_setappend(qh, &(newfacet->ridges), ridge);  // 将ridge添加到新facet的ridges集合中
      }else {  /* qh_attachnewfacets */
        if (neighbor->seen) {  // 如果相邻facet之前被看见过
          if (neighbor->simplicial) {  // 如果相邻facet是simplicial的
            qh_fprintf(qh, qh->ferr, 6105, "qhull internal error (qh_makenew_nonsimplicial): simplicial f%d sharing two ridges with f%d\n",
                   neighbor->id, visible->id);  // 输出错误信息到qh->ferr
            qh_errexit2(qh, qh_ERRqhull, neighbor, visible);  // 退出qhull程序
          }
          qh_setappend(qh, &(neighbor->neighbors), newfacet);  // 将当前新facet添加到相邻facet的neighbors集合中
        }else
          qh_setreplace(qh, neighbor->neighbors, visible, newfacet);  // 替换相邻facet的neighbors集合中的visible为新facet
        if (neighbor->simplicial) {
          qh_setdel(neighbor->ridges, ridge);  // 从相邻facet的ridges集合中删除ridge
          qh_delridge(qh, ridge);  // 删除ridge
        }else {
          qh_setappend(qh, &(newfacet->ridges), ridge);  // 将ridge添加到新facet的ridges集合中
          if (toporient) {
            ridge->top= newfacet;  // 设置ridge的顶部为新facet
            ridge->simplicialtop= True;  // 设置ridge的simplicialtop标志为True
          }else {
            ridge->bottom= newfacet;  // 设置ridge的底部为新facet
            ridge->simplicialbot= True;  // 设置ridge的simplicialbot标志为True
          }
        }
      }
      trace4((qh, qh->ferr, 4048, "qh_makenew_nonsimplicial: created facet f%d from v%d and r%d of horizon f%d\n",
          newfacet->id, apex->id, ridgeid, neighbor->id));  // 跟踪输出创建新facet的信息
    }
    neighbor->seen= True;  // 将相邻facet的seen标志设置为True
  } /* for each ridge */
  return newfacet;  // 返回新创建的facet
} /* makenew_nonsimplicial */
#else /* qh_NOmerge */
facetT *qh_makenew_nonsimplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew) {
  QHULL_UNUSED(qh)  // 使用宏QHULL_UNUSED，表示qh参数未使用
  QHULL_UNUSED(visible)  // 使用宏QHULL_UNUSED，表示visible参数未使用
  QHULL_UNUSED(apex)  // 使用宏QHULL_UNUSED，表示apex参数未使用
  QHULL_UNUSED(numnew)  // 使用宏QHULL_UNUSED，表示numnew参数未使用

  return NULL;  // 返回空指针，表示未执行任何操作
}
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenew_simplicial">-</a>

  qh_makenew_simplicial(qh, visible, apex, numnew )
    make new facets for simplicial visible facet and apex

  returns:
    attaches new facets if !qh.NEWtentative
      neighbors between newfacet and horizon

  notes:
    nop if neighbor->seen or neighbor->visible (see qh_makenew_nonsimplicial)

  design:
    locate neighboring horizon facet for visible facet
    determine vertices and orientation
    create new facet
    if coplanar,
      add new facet to f.samecycle
    update horizon facet's neighbor list
*/
facetT *qh_makenew_simplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew) {
  facetT *neighbor, **neighborp, *newfacet= NULL;
  setT *vertices;
  boolT flip, toporient;
  int horizonskip= 0, visibleskip= 0;

  // 遍历可见面的邻居面
  FOREACHneighbor_(visible) {
    // 如果邻居面没有被访问过且不可见
    if (!neighbor->seen && !neighbor->visible) {
      // 计算邻居面和可见面的公共顶点集合
      vertices= qh_facetintersect(qh, neighbor, visible, &horizonskip, &visibleskip, 1);
      // 将顶点集合的第一个顶点设置为顶点 apex
      SETfirst_(vertices)= apex;
      // 根据跳过标志位来确定是否需要翻转新面的方向
      flip= ((horizonskip & 0x1) ^ (visibleskip & 0x1));
      // 确定新面的方向
      if (neighbor->toporient)
        toporient= horizonskip & 0x1;
      else
        toporient= (horizonskip & 0x1) ^ 0x1;
      // 创建新面
      newfacet= qh_makenewfacet(qh, vertices, toporient, neighbor);
      (*numnew)++;
      // 如果邻居面是共面的且启用了精确合并选项
      if (neighbor->coplanarhorizon && (qh->PREmerge || qh->MERGEexact)) {
#ifndef qh_NOmerge
        // 将新面添加到同一循环中，并标记为合并的地平面
        newfacet->f.samecycle= newfacet;
        newfacet->mergehorizon= True;
#endif
      }
      // 如果不是在新面的预备状态下
      if (!qh->NEWtentative)
        // 更新邻居面的邻居列表
        SETelem_(neighbor->neighbors, horizonskip)= newfacet;
      // 输出调试信息
      trace4((qh, qh->ferr, 4049, "qh_makenew_simplicial: create facet f%d top %d from v%d and horizon f%d skip %d top %d and visible f%d skip %d, flip? %d\n",
            newfacet->id, toporient, apex->id, neighbor->id, horizonskip,
              neighbor->toporient, visible->id, visibleskip, flip));
    }
  }
  // 返回新创建的面
  return newfacet;
} /* makenew_simplicial */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchneighbor">-</a>

  qh_matchneighbor(qh, newfacet, newskip, hashsize, hashcount )
    either match subridge of newfacet with neighbor or add to hash_table

  returns:
    matched ridges of newfacet, except for duplicate ridges
    duplicate ridges marked by qh_DUPLICATEridge for qh_matchdupridge

  notes:
    called by qh_matchnewfacets
    assumes newfacet is simplicial
    ridge is newfacet->vertices w/o newskip vertex
    do not allocate memory (need to free hash_table cleanly)
    uses linear hash chains
    see qh_matchdupridge (poly2_r.c)

  design:
*/


注释完成。
    for each possible matching facet in qh.hash_table:
        # 遍历哈希表中的每一个可能匹配的面

      if vertices match:
          # 如果顶点匹配

        set ismatch, if facets have opposite orientation:
            # 如果面有相反的方向，则设置ismatch为真
          
        if ismatch and matching facet doesn't have a match:
            # 如果ismatch为真，并且匹配的面没有匹配项

          match the facets by updating their neighbor sets:
              # 通过更新它们的邻居集合来匹配这些面

        else:
            # 否则

          note: dupridge detected when a match 'f&d skip %d' has already been seen 
                need to mark all of the dupridges for qh_matchdupridge:
              # 注意：当已经看到一个匹配 'f&d skip %d' 时检测到重复的边缘
              # 需要标记所有的重复边缘用于 qh_matchdupridge

          indicate a duplicate ridge by qh_DUPLICATEridge and f.dupridge:
              # 通过 qh_DUPLICATEridge 和 f.dupridge 指示重复的边缘

          add facet to hashtable:
              # 将面添加到哈希表中

          unless the other facet was already a duplicate ridge:
              # 除非另一个面已经是一个重复的边缘

            mark both facets with a duplicate ridge:
                # 将两个面标记为重复的边缘

            add other facet (if defined) to hash table:
                # 将另一个面（如果已定义）添加到哈希表中

  state at "indicate a duplicate ridge":
      # 在指示重复边缘时的状态：

    newfacet@newskip= the argument
      # newfacet@newskip = 参数值

    facet= the hashed facet@skip that has the same vertices as newfacet@newskip
      # facet = 具有与 newfacet@newskip 相同顶点的哈希面@skip

    same= true if matched vertices have the same orientation
      # 如果匹配的顶点具有相同的方向，则same为真

    matchfacet= neighbor at facet@skip
      # matchfacet = facet@skip 处的邻居

    matchfacet=qh_DUPLICATEridge, matchfacet was previously detected as a dupridge of facet@skip
      # matchfacet=qh_DUPLICATEridge，matchfacet 之前被检测为 facet@skip 的重复边缘

    ismatch if 'vertex orientation (same) matches facet/newfacet orientation (toporient)
      # 如果 '顶点方向（same）与面/新面方向（toporient）匹配，则ismatch为真

    unknown facet will match later
      # 未来未知的面将会匹配

  details at "indicate a duplicate ridge":
      # 在指示重复边缘时的详细信息：

    if !ismatch and matchfacet:
        # 如果不是匹配，并且matchfacet存在

      dupridge is between hashed facet@skip/matchfacet@matchskip and arg newfacet@newskip/unknown
        # 重复边缘在哈希面@skip/matchfacet@matchskip 和 参数 newfacet@newskip/未知 之间

      set newfacet@newskip, facet@skip, and matchfacet@matchskip to qh_DUPLICATEridge
        # 将 newfacet@newskip、facet@skip 和 matchfacet@matchskip 设置为 qh_DUPLICATEridge

      add newfacet and matchfacet to hash_table
        # 将 newfacet 和 matchfacet 添加到哈希表中

    if ismatch and matchfacet:
        # 如果ismatch为真，并且matchfacet存在

      same as !ismatch and matchfacet -- it matches facet instead of matchfacet
        # 与 !ismatch 和 matchfacet 一样 -- 它匹配的是 facet 而不是 matchfacet

    if !ismatch and !matchfacet:
        # 如果不是匹配，并且matchfacet不存在

      dupridge between hashed facet@skip/unknown and arg newfacet@newskip/unknown
        # 重复边缘在哈希面@skip/未知 和 参数 newfacet@newskip/未知 之间

      set newfacet@newskip and facet@skip to qh_DUPLICATEridge
        # 将 newfacet@newskip 和 facet@skip 设置为 qh_DUPLICATEridge

      add newfacet to hash_table
        # 将 newfacet 添加到哈希表中

    if ismatch and matchfacet==qh_DUPLICATEridge:
        # 如果ismatch为真，并且matchfacet等于qh_DUPLICATEridge

      dupridge with already duplicated hashed facet@skip and arg newfacet@newskip/unknown
        # 已经重复的哈希面@skip 和 参数 newfacet@newskip/未知 之间的重复边缘

      set newfacet@newskip to qh_DUPLICATEridge
        # 将 newfacet@newskip 设置为 qh_DUPLICATEridge

      add newfacet to hash_table
        # 将 newfacet 添加到哈希表中

      facet's hyperplane already set
        # 面的超平面已经设置
/*
void qh_matchneighbor(qhT *qh, facetT *newfacet, int newskip, int hashsize, int *hashcount) {
  boolT newfound= False;   /* True, if new facet is already in hash chain */
  boolT same, ismatch;
  int hash, scan;
  facetT *facet, *matchfacet;
  int skip, matchskip;

  hash= qh_gethash(qh, hashsize, newfacet->vertices, qh->hull_dim, 1,
                     SETelem_(newfacet->vertices, newskip));
  trace4((qh, qh->ferr, 4050, "qh_matchneighbor: newfacet f%d skip %d hash %d hashcount %d\n",
          newfacet->id, newskip, hash, *hashcount));
  zinc_(Zhashlookup);
  for (scan=hash; (facet= SETelemt_(qh->hash_table, scan, facetT));
       scan= (++scan >= hashsize ? 0 : scan)) {
    if (facet == newfacet) {
      newfound= True;
      continue;
    }
    zinc_(Zhashtests);
    }
  }
  if (!newfound)
    SETelem_(qh->hash_table, scan)= newfacet;  /* same as qh_addhash */
  (*hashcount)++;
  trace4((qh, qh->ferr, 4053, "qh_matchneighbor: no match for f%d skip %d at hash %d\n",
           newfacet->id, newskip, hash));
} /* matchneighbor */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchnewfacets">-</a>

  qh_matchnewfacets(qh )
    match new facets in qh.newfacet_list to their newfacet neighbors
    all facets are simplicial

  returns:
    if dupridges and merging 
      returns maxdupdist (>=0.0) from vertex to opposite facet
      sets facet->dupridge
      missing neighbor links identify dupridges to be merged (qh_DUPLICATEridge)
    else  
      qh.newfacet_list with full neighbor sets
        vertices for the nth neighbor match all but the nth vertex
    if not merging and qh.FORCEoutput
      for facets with normals (i.e., with dupridges)
      sets facet->flippped for flipped normals, also prevents point partitioning

  notes:
    called by qh_buildcone* and qh_triangulate_facet
    neighbor[0] of new facets is the horizon facet
    if NEWtentative, new facets not attached to the horizon
    assumes qh.hash_table is NULL
    vertex->neighbors has not been updated yet
    do not allocate memory after qh.hash_table (need to free it cleanly)
    
  design:
    truncate neighbor sets to horizon facet for all new facets
    initialize a hash table
    for all new facets
      match facet with neighbors
    if unmatched facets (due to duplicate ridges)
      for each new facet with a duplicate ridge
        try to match facets with the same coplanar horizon
    if not all matched
      for each new facet with a duplicate ridge
        match it with a coplanar facet, or identify a pinched vertex
    if not merging and qh.FORCEoutput
      check for flipped facets
*/
coordT qh_matchnewfacets(qhT *qh /* qh.newfacet_list */) {
  int numnew=0, hashcount=0, newskip;
  facetT *newfacet, *neighbor;
  coordT maxdupdist= 0.0, maxdist2;
  int dim= qh->hull_dim, hashsize, neighbor_i, neighbor_n;
  setT *neighbors;
#ifndef qh_NOtrace
  int facet_i, facet_n, numunused= 0;
  facetT *facet;
#endif

trace1((qh, qh->ferr, 1019, "qh_matchnewfacets: match neighbors for new facets.\n"));
// 跟踪调试信息：匹配新面的邻居

FORALLnew_facets {
  numnew++;
  {  /* inline qh_setzero(qh, newfacet->neighbors, 1, qh->hull_dim); */
    neighbors= newfacet->neighbors;
    neighbors->e[neighbors->maxsize].i= dim+1; /*may be overwritten*/
    // 设置 newfacet 的 neighbors 初始值
    memset((char *)SETelemaddr_(neighbors, 1, void), 0, (size_t)(dim * SETelemsize));
    // 初始化 neighbors 中的元素为 0
  }
}

qh_newhashtable(qh, numnew*(qh->hull_dim-1)); /* twice what is normally needed,
                                 but every ridge could be DUPLICATEridge */
// 初始化一个新的哈希表，大小为 numnew*(qh->hull_dim-1)，通常的两倍大小，
// 因为每个 ridge 可能是 DUPLICATEridge
hashsize= qh_setsize(qh, qh->hash_table);
// 获取哈希表的大小
FORALLnew_facets {
  if (!newfacet->simplicial) {
    qh_fprintf(qh, qh->ferr, 6377, "qhull internal error (qh_matchnewfacets): expecting simplicial facets on qh.newfacet_list f%d for qh_matchneighbors, qh_matchneighbor, and qh_matchdupridge.  Got non-simplicial f%d\n",
      qh->newfacet_list->id, newfacet->id);
    qh_errexit2(qh, qh_ERRqhull, newfacet, qh->newfacet_list);
    // 错误处理：如果 newfacet 不是简单面（simplicial）
  }
  for (newskip=1; newskip<qh->hull_dim; newskip++) /* furthest/horizon already matched */
    // 遍历 newfacet 的邻居，除了最远的/地平线已经匹配
    qh_matchneighbor(qh, newfacet, newskip, hashsize, &hashcount);
#if 0   /* use the following to trap hashcount errors */
  {
    int count= 0, k;
    facetT *facet, *neighbor;

    count= 0;
    FORALLfacet_(qh->newfacet_list) {  /* newfacet already in use */
      for (k=1; k < qh->hull_dim; k++) {
        neighbor= SETelemt_(facet->neighbors, k, facetT);
        if (!neighbor || neighbor == qh_DUPLICATEridge)
          count++;
      }
      if (facet == newfacet)
        break;
    }
    if (count != hashcount) {
      qh_fprintf(qh, qh->ferr, 6266, "qhull error (qh_matchnewfacets): after adding facet %d, hashcount %d != count %d\n",
               newfacet->id, hashcount, count);
      qh_errexit(qh, qh_ERRdebug, newfacet, NULL);
    }
  }
#endif  /* end of trap code */
} /* end FORALLnew_facets */
// 遍历所有的 newfacet

if (hashcount) { /* all neighbors matched, except for qh_DUPLICATEridge neighbors */
  qh_joggle_restart(qh, "ridge with multiple neighbors");
  if (hashcount) {
    FORALLnew_facets {
      if (newfacet->dupridge) {
        FOREACHneighbor_i_(qh, newfacet) {
          if (neighbor == qh_DUPLICATEridge) {
            maxdist2= qh_matchdupridge(qh, newfacet, neighbor_i, hashsize, &hashcount);
            maximize_(maxdupdist, maxdist2);
          }
        }
      }
    }
  }
}
// 如果 hashcount 不为 0，则处理多个邻居的情况

if (hashcount) {
  qh_fprintf(qh, qh->ferr, 6108, "qhull internal error (qh_matchnewfacets): %d neighbors did not match up\n",
      hashcount);
  qh_printhashtable(qh, qh->ferr);
  qh_errexit(qh, qh_ERRqhull, NULL, NULL);
}
// 如果 hashcount 不为 0，则输出错误信息，并退出程序

#ifndef qh_NOtrace
if (qh->IStracing >= 3) {
  FOREACHfacet_i_(qh, qh->hash_table) {
    if (!facet)
      numunused++;
  }
}
#endif /* qh_NOtrace */
// 如果开启了跟踪模式，并且跟踪级别大于等于 3，则统计未使用的面数
    # 调用 qh_fprintf 函数，向 qh->ferr 文件流写入格式化后的字符串
    qh_fprintf(qh, qh->ferr, 3063, "qh_matchnewfacets: maxdupdist %2.2g, new facets %d, unused hash entries %d, hashsize %d\n",
               maxdupdist, numnew, numunused, qh_setsize(qh, qh->hash_table));
  }
#endif /* !qh_NOtrace */
/* 结束条件检查，如果未定义 qh_NOtrace，则执行以下代码块 */

  qh_setfree(qh, &qh->hash_table);
  /* 释放哈希表占用的内存空间 */

  if (qh->PREmerge || qh->MERGEexact) {
    /* 如果定义了 PREmerge 或者 MERGEexact */

    if (qh->IStracing >= 4)
      /* 如果跟踪级别 IStracing 大于等于 4 */

      qh_printfacetlist(qh, qh->newfacet_list, NULL, qh_ALL);
      /* 打印新面列表 qh->newfacet_list 的信息，参数为 qh_ALL */

  }
  /* 返回 maxdupdist 变量的值作为函数返回值 */
  return maxdupdist;
} /* matchnewfacets */
/* 函数 matchnewfacets 结束 */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchvertices">-</a>
/* 链接至 qh-poly_r.htm 的标签，用于描述 matchvertices 函数 */

  qh_matchvertices(qh, firstindex, verticesA, skipA, verticesB, skipB, same )
    tests whether vertices match with a single skip
    starts match at firstindex since all new facets have a common vertex
/* qh_matchvertices 函数定义，测试顶点是否匹配，使用单个跳过索引。
   从 firstindex 处开始匹配，因为所有新的面都有一个公共顶点 */

  returns:
/* 返回值：

    true if matched vertices
    skip index for skipB
    sets same iff vertices have the same orientation
/* 如果顶点匹配则返回 true。
   返回 skipB 的索引。
   设置 same 变量当且仅当顶点具有相同的方向 */

  notes:
/* 注意事项：

    called by qh_matchneighbor and qh_matchdupridge
/* 被 qh_matchneighbor 和 qh_matchdupridge 调用 */

    assumes skipA is in A and both sets are the same size
/* 假设 skipA 在集合 A 中，并且两个集合的大小相同 */

  design:
/* 设计：

    set up pointers
/* 设置指针 */

    scan both sets checking for a match
/* 扫描两个集合以检查是否匹配 */

    test orientation
/* 测试方向 */
*/
boolT qh_matchvertices(qhT *qh, int firstindex, setT *verticesA, int skipA,
       setT *verticesB, int *skipB, boolT *same) {
/* qh_matchvertices 函数签名 */

  vertexT **elemAp, **elemBp, **skipBp=NULL, **skipAp;
/* 声明指向顶点的指针变量 */

  elemAp= SETelemaddr_(verticesA, firstindex, vertexT);
  /* 设置 elemAp 指向 verticesA 集合中的第 firstindex 个顶点 */

  elemBp= SETelemaddr_(verticesB, firstindex, vertexT);
  /* 设置 elemBp 指向 verticesB 集合中的第 firstindex 个顶点 */

  skipAp= SETelemaddr_(verticesA, skipA, vertexT);
  /* 设置 skipAp 指向 verticesA 集合中的第 skipA 个顶点 */

  do if (elemAp != skipAp) {
    /* 如果 elemAp 不等于 skipAp */

    while (*elemAp != *elemBp++) {
      /* 当 elemAp 指向的顶点不等于 elemBp 指向的顶点时 */

      if (skipBp)
        /* 如果 skipBp 非空 */

        return False;
        /* 返回 false */

      skipBp= elemBp;  /* one extra like FOREACH */
      /* 将 skipBp 设置为 elemBp，类似 FOREACH 的一次额外操作 */
    }
  }while (*(++elemAp));
  /* 循环直到 elemAp 指向的顶点为空 */

  if (!skipBp)
    skipBp= ++elemBp;
  /* 如果 skipBp 为空，则将其设置为 elemBp 的后一个 */

  *skipB= SETindex_(verticesB, skipB); /* i.e., skipBp - verticesB
                                       verticesA and verticesB are the same size, otherwise trace4 may segfault */
  /* 计算 skipB 的值，即 skipBp 在 verticesB 中的索引
     verticesA 和 verticesB 的大小相同，否则 trace4 可能会导致段错误 */

  *same= !((skipA & 0x1) ^ (*skipB & 0x1)); /* result is 0 or 1 */
  /* 设置 same 变量，判断 skipA 和 *skipB 的奇偶性是否相同 */

  trace4((qh, qh->ferr, 4054, "qh_matchvertices: matched by skip %d(v%d) and skip %d(v%d) same? %d\n",
          skipA, (*skipAp)->id, *skipB, (*(skipBp-1))->id, *same));
  /* 跟踪调试信息，打印匹配的跳过信息和方向信息 */

  return(True);
  /* 返回 true */
} /* matchvertices */
/* matchvertices 函数结束 */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newfacet">-</a>
/* 链接至 qh-poly_r.htm 的标签，用于描述 newfacet 函数 */

  qh_newfacet(qh)
    return a new facet
/* qh_newfacet 函数定义，返回一个新的面 */

  returns:
/* 返回值：

    all fields initialized or cleared   (NULL)
/* 所有字段都被初始化或清除（NULL） */

    preallocates neighbors set
/* 预分配邻居集合 */

*/
facetT *qh_newfacet(qhT *qh) {
/* qh_newfacet 函数签名 */

  facetT *facet;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */
  /* freelistp 变量声明，如果 !qh_NOmem 则由 qh_memalloc_() 使用 */

  qh_memalloc_(qh, (int)sizeof(facetT), freelistp, facet, facetT);
  /* 使用 qh_memalloc_() 分配内存给 facet 变量 */

  memset((char *)facet, (size_t)0, sizeof(facetT));
  /* 使用 0 初始化 facet 变量所占内存 */

  if (qh->facet_id == qh->tracefacet_id)
    /* 如果 facet_id 等于 tracefacet_id */

    qh->tracefacet= facet;
    /* 设置 qh->tracefacet 指向 facet */

  facet->id= qh->facet_id++;
  /* 设置 facet 的 id，并递增 qh->facet_id */

  facet->neighbors= qh_setnew(qh, qh->hull_dim);
  /* 使用 qh_setnew() 创建 facet 的邻居集合 */

#if !qh_COMPUTEfurthest
  facet->furthestdist= 0.0;
#endif

#if qh_MAXoutside
  if (qh->FORCEoutput && qh->APPROXhull)
    facet->maxoutside= qh->MINoutside;
  else
    facet->maxoutside= qh->DISTround; /* same value as test for QH7082 */
#endif

  facet->simplicial= True;
  /* 设置 facet 的 simplicial 属性为真 */

  facet->good= True;
  /* 设置 facet 的 good 属性为真 */

  facet->newfacet= True;
  /* 设置 facet 的 newfacet 属性为真 */

  trace4((qh, qh->ferr, 4055, "qh_newfacet: created facet f%d\n", facet->id));
  /* 跟踪调试信息，打印创建 facet 的信息 */

  return(facet);
  /* 返回 facet 变量 */
} /* newfacet */
/* newfacet 函数结束 */
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newridge">-</a>

  qh_newridge()
    return a new ridge
  notes:
    caller sets qh.traceridge
*/
ridgeT *qh_newridge(qhT *qh) {
  ridgeT *ridge;
  void **freelistp;   /* 如果 !qh_NOmem，则由 qh_memalloc_() 使用 */

  // 使用 qh_memalloc_() 分配 ridgeT 结构的内存空间，并将其地址赋给 ridge
  qh_memalloc_(qh, (int)sizeof(ridgeT), freelistp, ridge, ridgeT);
  // 将分配的内存空间清零
  memset((char *)ridge, (size_t)0, sizeof(ridgeT));
  // 增加总 ridge 计数
  zinc_(Ztotridges);
  // 如果 ridge ID 达到 UINT_MAX，则发出警告，并重新从 0 开始计数
  if (qh->ridge_id == UINT_MAX) {
    qh_fprintf(qh, qh->ferr, 7074, "qhull warning: more than 2^32 ridges.  Qhull results are OK.  Since the ridge ID wraps around to 0, two ridges may have the same identifier.\n");
  }
  // 设置当前 ridge 的 ID，并增加全局 ridge ID
  ridge->id= qh->ridge_id++;
  // 记录日志信息，追踪新创建的 ridge
  trace4((qh, qh->ferr, 4056, "qh_newridge: created ridge r%d\n", ridge->id));
  // 返回新创建的 ridge
  return(ridge);
} /* newridge */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointid">-</a>

  qh_pointid(qh, point )
    return id for a point,
    returns qh_IDnone(-3) if null, qh_IDinterior(-2) if interior, or qh_IDunknown(-1) if not known

  alternative code if point is in qh.first_point...
    unsigned long id;
    id= ((unsigned long)point - (unsigned long)qh.first_point)/qh.normal_size;

  notes:
    Valid points are non-negative
    WARN64 -- id truncated to 32-bits, at most 2G points
    NOerrors returned (QhullPoint::id)
    if point not in point array
      the code does a comparison of unrelated pointers.
*/
int qh_pointid(qhT *qh, pointT *point) {
  ptr_intT offset, id;

  // 如果 point 或 qh 为 null，则返回 qh_IDnone(-3)
  if (!point || !qh)
    return qh_IDnone;
  // 如果 point 是内部点，则返回 qh_IDinterior(-2)
  else if (point == qh->interior_point)
    return qh_IDinterior;
  // 如果 point 在 qh.first_point 之间，计算其在点数组中的索引并返回其 ID
  else if (point >= qh->first_point
  && point < qh->first_point + qh->num_points * qh->hull_dim) {
    offset= (ptr_intT)(point - qh->first_point);
    id= offset / qh->hull_dim;
  }else if ((id= qh_setindex(qh->other_points, point)) != -1)
    // 如果 point 在其他点集合中，返回其在点数组中的索引加上已知点数
    id += qh->num_points;
  else
    // 如果 point 不在点数组中，返回 qh_IDunknown(-1)
    return qh_IDunknown;
  // 返回计算出的点的 ID
  return (int)id;
} /* pointid */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="removefacet">-</a>

  qh_removefacet(qh, facet )
    unlinks facet from qh.facet_list,

  returns:
    updates qh.facet_list .newfacet_list .facet_next visible_list
    decrements qh.num_facets

  see:
    qh_appendfacet
*/
void qh_removefacet(qhT *qh, facetT *facet) {
  facetT *next= facet->next, *previous= facet->previous; /* next is always defined */

  // 如果要移除的 facet 是 newfacet_list 中的第一个，则更新 newfacet_list
  if (facet == qh->newfacet_list)
    qh->newfacet_list= next;
  // 如果要移除的 facet 是 facet_next 中的第一个，则更新 facet_next
  if (facet == qh->facet_next)
    qh->facet_next= next;
  // 如果要移除的 facet 是 visible_list 中的第一个，则更新 visible_list
  if (facet == qh->visible_list)
    qh->visible_list= next;
  // 如果 facet 有前驱，则重新连接前后 facet，否则 facet 是 facet_list 的第一个 facet
  if (previous) {
    previous->next= next;
    next->previous= previous;
  }else {  /* 1st facet in qh->facet_list */
    qh->facet_list= next;
    qh->facet_list->previous= NULL;
  }
  // 减少 qh.num_facets 的计数
  qh->num_facets--;
  // 记录日志信息，追踪移除的 facet
  trace4((qh, qh->ferr, 4057, "qh_removefacet: removed f%d from facet_list, newfacet_list, and visible_list\n", facet->id));
} /* removefacet */
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="removevertex">-</a>

  qh_removevertex(qh, vertex )
    unlinks vertex from qh.vertex_list,

  returns:
    updates qh.vertex_list .newvertex_list
    decrements qh.num_vertices
*/
void qh_removevertex(qhT *qh, vertexT *vertex) {
  vertexT *next= vertex->next, *previous= vertex->previous; /* next is always defined */

  trace4((qh, qh->ferr, 4058, "qh_removevertex: remove v%d from qh.vertex_list\n", vertex->id));
  if (vertex == qh->newvertex_list)
    qh->newvertex_list= next;
  if (previous) {
    previous->next= next;
    next->previous= previous;
  }else {  /* 1st vertex in qh->vertex_list */
    qh->vertex_list= next;
    qh->vertex_list->previous= NULL;
  }
  qh->num_vertices--;
} /* removevertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="update_vertexneighbors">-</a>

  qh_update_vertexneighbors(qh )
    update vertex neighbors and delete interior vertices

  returns:
    if qh.VERTEXneighbors, 
      if qh.newvertex_list,
         removes visible neighbors from vertex neighbors
      if qh.newfacet_list
         adds new facets to vertex neighbors
      if qh.visible_list
         interior vertices added to qh.del_vertices for later partitioning as coplanar points
    if not qh.VERTEXneighbors (not merging)
      interior vertices of visible facets added to qh.del_vertices for later partitioning as coplanar points
  
  notes
    [jan'19] split off qh_update_vertexneighbors_cone.  Optimize the remaining cases in a future release
    called by qh_triangulate_facet after triangulating a non-simplicial facet, followed by reset_lists
    called by qh_triangulate after triangulating null and mirror facets
    called by qh_all_vertexmerges after calling qh_merge_pinchedvertices

  design:
    if qh.VERTEXneighbors
      for each vertex on newvertex_list (i.e., new vertices and vertices of new facets)
        delete visible facets from vertex neighbors
      for each new facet on newfacet_list
        for each vertex of facet
          append facet to vertex neighbors
      for each visible facet on qh.visible_list
        for each vertex of facet
          if the vertex is not on a new facet and not itself deleted
            if the vertex has a not-visible neighbor (due to merging)
               remove the visible facet from the vertex's neighbors
            otherwise
               add the vertex to qh.del_vertices for later deletion

    if not qh.VERTEXneighbors (not merging)
      for each vertex of a visible facet
        if the vertex is not on a new facet and not itself deleted
           add the vertex to qh.del_vertices for later deletion
*/
void qh_update_vertexneighbors(qhT *qh /* qh.newvertex_list, newfacet_list, visible_list */) {
  facetT *newfacet= NULL, *neighbor, **neighborp, *visible;
  vertexT *vertex, **vertexp;
  int neighborcount= 0;

  if (qh->VERTEXneighbors) {
    // 如果需要更新顶点的邻居信息
    trace3((qh, qh->ferr, 3013, "qh_update_vertexneighbors: update v.neighbors for qh.newvertex_list (v%d) and qh.newfacet_list (f%d)\n",
         getid_(qh->newvertex_list), getid_(qh->newfacet_list)));
    // 遍历新顶点列表
    FORALLvertex_(qh->newvertex_list) {
      neighborcount= 0;
      // 遍历当前顶点的邻居
      FOREACHneighbor_(vertex) {
        // 如果邻居是可见的
        if (neighbor->visible) {
          neighborcount++;
          // 清除邻居的引用
          SETref_(neighbor)= NULL;
        }
      }
      // 如果有需要删除的邻居
      if (neighborcount) {
        // 输出日志，指示删除顶点的邻居信息
        trace4((qh, qh->ferr, 4046, "qh_update_vertexneighbors: delete %d of %d vertex neighbors for v%d.  Removes to-be-deleted, visible facets\n",
          neighborcount, qh_setsize(qh, vertex->neighbors), vertex->id));
        // 压缩顶点的邻居集合
        qh_setcompact(qh, vertex->neighbors);
      }
    }
    // 遍历所有新的面
    FORALLnew_facets {
      // 如果是第一个新面并且面的 ID 大于等于第一个新面的 ID
      if (qh->first_newfacet && newfacet->id >= qh->first_newfacet) {
        // 将面的顶点添加到相应顶点的邻居列表中
        FOREACHvertex_(newfacet->vertices)
          qh_setappend(qh, &vertex->neighbors, newfacet);
      }else {  /* 在 qh_merge_pinchedvertices 之后调用。在 7-D 中，邻居数量可能比新面多。qh_setin 操作昂贵 */
        // 将面的顶点添加到相应顶点的邻居列表中，确保唯一性
        FOREACHvertex_(newfacet->vertices)
          qh_setunique(qh, &vertex->neighbors, newfacet); 
      }
    }
    // 输出日志，指示删除内部顶点信息
    trace3((qh, qh->ferr, 3058, "qh_update_vertexneighbors: delete interior vertices for qh.visible_list (f%d)\n",
        getid_(qh->visible_list)));
    // 遍历所有可见的面
    FORALLvisible_facets {
      // 遍历当前可见面的顶点
      FOREACHvertex_(visible->vertices) {
        // 如果顶点不是新的面的顶点，并且没有被删除
        if (!vertex->newfacet && !vertex->deleted) {
          // 检查顶点的邻居是否都是可见的（在合并过程中可能发生）
          FOREACHneighbor_(vertex) {
            if (!neighbor->visible)
              break;
          }
          // 如果有不可见的邻居，则从顶点的邻居列表中删除当前面
          if (neighbor)
            qh_setdel(vertex->neighbors, visible);
          else {
            // 否则标记顶点为已删除，并将其添加到待删除顶点列表中
            vertex->deleted= True;
            qh_setappend(qh, &qh->del_vertices, vertex);
            // 输出日志，指示删除内部顶点信息
            trace2((qh, qh->ferr, 2041, "qh_update_vertexneighbors: delete interior vertex p%d(v%d) of visible f%d\n",
                  qh_pointid(qh, vertex->point), vertex->id, visible->id));
          }
        }
      }
    }
  }else {  /* !VERTEXneighbors */
    // 如果不需要更新顶点的邻居信息
    trace3((qh, qh->ferr, 3058, "qh_update_vertexneighbors: delete old vertices for qh.visible_list (f%d)\n",
      getid_(qh->visible_list)));
    // 遍历所有可见的面
    FORALLvisible_facets {
      // 遍历当前可见面的顶点
      FOREACHvertex_(visible->vertices) {
        // 如果顶点不是新的面的顶点，并且没有被删除
        if (!vertex->newfacet && !vertex->deleted) {
          // 标记顶点为已删除，并将其添加到待删除顶点列表中
          vertex->deleted= True;
          qh_setappend(qh, &qh->del_vertices, vertex);
          // 输出日志，指示将删除内部顶点信息
          trace2((qh, qh->ferr, 2042, "qh_update_vertexneighbors: will delete interior vertex p%d(v%d) of visible f%d\n",
                  qh_pointid(qh, vertex->point), vertex->id, visible->id));
        }
      }
    }
  }
} /* update_vertexneighbors */
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="update_vertexneighbors_cone">-</a>

  qh_update_vertexneighbors_cone(qh )
    update vertex neighbors for a cone of new facets and delete interior vertices

  returns:
    if qh.VERTEXneighbors, 
      if qh.newvertex_list,
         removes visible neighbors from vertex neighbors
      if qh.newfacet_list
         adds new facets to vertex neighbors
      if qh.visible_list
         interior vertices added to qh.del_vertices for later partitioning as coplanar points
    if not qh.VERTEXneighbors (not merging)
      interior vertices of visible facets added to qh.del_vertices for later partitioning as coplanar points
  
  notes
    called by qh_addpoint after create cone and before premerge

  design:
    if qh.VERTEXneighbors
      for each vertex on newvertex_list (i.e., new vertices and vertices of new facets)
        delete visible facets from vertex neighbors
      for each new facet on newfacet_list
        for each vertex of facet
          append facet to vertex neighbors
      for each visible facet on qh.visible_list
        for each vertex of facet
          if the vertex is not on a new facet and not itself deleted
            if the vertex has a not-visible neighbor (due to merging)
               remove the visible facet from the vertex's neighbors
            otherwise
               add the vertex to qh.del_vertices for later deletion

    if not qh.VERTEXneighbors (not merging)
      for each vertex of a visible facet
        if the vertex is not on a new facet and not itself deleted
           add the vertex to qh.del_vertices for later deletion

*/

void qh_update_vertexneighbors_cone(qhT *qh /* qh.newvertex_list, newfacet_list, visible_list */) {
  facetT *newfacet= NULL, *neighbor, **neighborp, *visible;
  vertexT *vertex, **vertexp;
  int delcount= 0;

  if (qh->VERTEXneighbors) {
    trace3((qh, qh->ferr, 3059, "qh_update_vertexneighbors_cone: update v.neighbors for qh.newvertex_list (v%d) and qh.newfacet_list (f%d)\n",
         getid_(qh->newvertex_list), getid_(qh->newfacet_list)));

    /* Iterate through each vertex in qh.newvertex_list */
    FORALLvertex_(qh->newvertex_list) {
      delcount= 0;
      
      /* Iterate through neighbors of the current vertex */
      FOREACHneighbor_(vertex) {
        if (neighbor->visible) { /* Check if neighbor is visible */
          delcount++;
          qh_setdelnth(qh, vertex->neighbors, SETindex_(vertex->neighbors, neighbor));
          neighborp--; /* Adjust pointer to repeat iteration */
        }
      }

      /* Trace message about deleted visible vertex neighbors */
      if (delcount) {
        trace4((qh, qh->ferr, 4021, "qh_update_vertexneighbors_cone: deleted %d visible vertexneighbors of v%d\n",
          delcount, vertex->id));
      }
    }

    /* Iterate through each new facet in qh.newfacet_list */
    FORALLnew_facets {
      /* Iterate through vertices of the current facet and append the facet to their neighbors */
      FOREACHvertex_(newfacet->vertices)
        qh_setappend(qh, &vertex->neighbors, newfacet);
    }

    /* Trace message about deleting interior vertices from qh.visible_list */
    trace3((qh, qh->ferr, 3065, "qh_update_vertexneighbors_cone: delete interior vertices, if any, for qh.visible_list (f%d)\n",
        getid_(qh->visible_list)));
    # 对所有可见面进行循环遍历
    FORALLvisible_facets {
      # 对当前可见面的所有顶点进行循环遍历
      FOREACHvertex_(visible->vertices) {
        # 检查顶点是否不属于新面且未被标记删除
        if (!vertex->newfacet && !vertex->deleted) {
          # 对顶点的所有相邻顶点进行循环遍历
          FOREACHneighbor_(vertex) { /* this can happen under merging, qh_checkfacet QH4025 */
            # 如果存在不可见的相邻顶点，则中断循环
            if (!neighbor->visible)
              break;
          }
          # 如果存在不可见的相邻顶点，将当前顶点从可见面的邻居顶点集合中删除
          if (neighbor)
            qh_setdel(vertex->neighbors, visible);
          else {
            # 否则标记当前顶点为已删除状态
            vertex->deleted= True;
            # 将当前顶点追加到待删除顶点列表中
            qh_setappend(qh, &qh->del_vertices, vertex);
            # 记录删除操作的详细信息到日志中
            trace2((qh, qh->ferr, 2102, "qh_update_vertexneighbors_cone: will delete interior vertex p%d(v%d) of visible f%d\n",
              qh_pointid(qh, vertex->point), vertex->id, visible->id));
          }
        }
      }
    }
  }else {  /* !VERTEXneighbors */
    # 如果不需要处理顶点的相邻关系
    trace3((qh, qh->ferr, 3066, "qh_update_vertexneighbors_cone: delete interior vertices for qh.visible_list (f%d)\n",
      getid_(qh->visible_list)));
    # 对所有可见面进行循环遍历
    FORALLvisible_facets {
      # 对当前可见面的所有顶点进行循环遍历
      FOREACHvertex_(visible->vertices) {
        # 检查顶点是否不属于新面且未被标记删除
        if (!vertex->newfacet && !vertex->deleted) {
          # 标记当前顶点为已删除状态
          vertex->deleted= True;
          # 将当前顶点追加到待删除顶点列表中
          qh_setappend(qh, &qh->del_vertices, vertex);
          # 记录删除操作的详细信息到日志中
          trace2((qh, qh->ferr, 2059, "qh_update_vertexneighbors_cone: will delete interior vertex p%d(v%d) of visible f%d\n",
                  qh_pointid(qh, vertex->point), vertex->id, visible->id));
        }
      }
    }
  }
} /* update_vertexneighbors_cone */
```