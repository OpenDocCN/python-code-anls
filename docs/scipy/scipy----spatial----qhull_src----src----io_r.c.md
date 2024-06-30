# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\io_r.c`

```
/* 这部分代码是 qhull 应用程序的输入/输出相关的函数实现 */

#include "qhull_ra.h"

/*========= -functions in alphabetical order after qh_produce_output(qh)  =====*/

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="produce_output">-</a>

  qh_produce_output(qh )
  qh_produce_output2(qh )
    打印出 qhull 计算结果的所需格式
    qh_produce_output2 不会调用 qh_prepare_output
      qh_checkpolygon 对 qh_prepare_output 有效
    如果 qh.GETarea
      计算并打印面积和体积
    qh.PRINTout[] 是输出格式的数组

  notes:
    按照 qh.PRINTout 中的顺序打印输出
*/
void qh_produce_output(qhT *qh) {
    int tempsize= qh_setsize(qh, qh->qhmem.tempstack);

    // 准备输出
    qh_prepare_output(qh);
    // 生成输出
    qh_produce_output2(qh);
    // 检查临时集合是否为空
    if (qh_setsize(qh, qh->qhmem.tempstack) != tempsize) {
        qh_fprintf(qh, qh->ferr, 6206, "qhull internal error (qh_produce_output): temporary sets not empty(%d)\n",
            qh_setsize(qh, qh->qhmem.tempstack));
        qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
} /* produce_output */


void qh_produce_output2(qhT *qh) {
  int i, tempsize= qh_setsize(qh, qh->qhmem.tempstack), d_1;

  fflush(NULL); // 刷新所有流
  // 如果设置了 PRINTsummary，则在 ferr 流上打印摘要
  if (qh->PRINTsummary)
    qh_printsummary(qh, qh->ferr);
  // 否则，如果 PRINTout[0] 是 PRINTnone，则在 fout 流上打印摘要
  else if (qh->PRINTout[0] == qh_PRINTnone)
    qh_printsummary(qh, qh->fout);
  // 遍历 PRINTout 数组中的所有输出格式
  for (i=0; i < qh_PRINTEND; i++)
    // 在 fout 流上打印相应格式的几何信息
    qh_printfacets(qh, qh->fout, qh->PRINTout[i], qh->facet_list, NULL, !qh_ALL);
  fflush(NULL); // 刷新所有流

  // 打印所有统计信息
  qh_allstatistics(qh);
  // 如果 PRINTprecision 被设置，并且未合并并且 JOGGLEmax 大于 REALmax/2 或者 RERUN 已设置
  if (qh->PRINTprecision && !qh->MERGING && (qh->JOGGLEmax > REALmax/2 || qh->RERUN))
    qh_printstats(qh, qh->ferr, qh->qhstat.precision, NULL);
  // 如果 VERIFYoutput 被设置，并且 Zridge 或 Zridgemid 大于 0
  if (qh->VERIFYoutput && (zzval_(Zridge) > 0 || zzval_(Zridgemid) > 0))
    qh_printstats(qh, qh->ferr, qh->qhstat.vridges, NULL);
  // 如果 PRINTstatistics 被设置
  if (qh->PRINTstatistics) {
    // 打印统计信息到 ferr 流
    qh_printstatistics(qh, qh->ferr, "");
    // 打印内存统计信息到 ferr 流
    qh_memstatistics(qh, qh->ferr);
    // 计算并打印各数据结构大小信息
    d_1= (int)sizeof(setT) + (qh->hull_dim - 1) * SETelemsize;
    qh_fprintf(qh, qh->ferr, 8040, "\
    size in bytes: merge %d ridge %d vertex %d facet %d\n\
         normal %d ridge vertices %d facet vertices or neighbors %d\n",
            (int)sizeof(mergeT), (int)sizeof(ridgeT),
            (int)sizeof(vertexT), (int)sizeof(facetT),
            qh->normal_size, d_1, d_1 + SETelemsize);
  }
  // 再次检查临时集合是否为空
  if (qh_setsize(qh, qh->qhmem.tempstack) != tempsize) {

    // 如果临时集合不为空，则输出错误信息并退出程序
    qh_fprintf(qh, qh->ferr, 6206, "qhull internal error (qh_produce_output): temporary sets not empty(%d)\n",
            qh_setsize(qh, qh->qhmem.tempstack));
        qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
} /* produce_output2 */


这段代码是 qhull 应用程序的一部分，主要实现了一些用于输出计算结果的函数。其中 `qh_produce_output` 和 `qh_produce_output2` 函数负责生成输出，打印各种格式的几何信息和统计数据。代码中还有对临时集合的检查，以确保在生成输出前后集合是空的，否则会输出错误信息并退出程序。
    # 使用 qh_fprintf 函数向 qh 对象的 ferr 流输出错误信息
    qh_fprintf(qh, qh->ferr, 6065, "qhull internal error (qh_produce_output2): temporary sets not empty(%d)\n",
               qh_setsize(qh, qh->qhmem.tempstack));
    # 调用 qh_errexit 函数，以 qh_ERRqhull 作为错误类型，结束程序运行，不返回任何信息
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="dfacet">-</a>

  qh_dfacet(qh, id )
    print facet by id, for debugging

*/
void qh_dfacet(qhT *qh, unsigned int id) {
  facetT *facet;

  FORALLfacets {  // 遍历所有的 facetT 结构体
    if (facet->id == id) {  // 如果找到与给定 id 匹配的 facet
      qh_printfacet(qh, qh->fout, facet);  // 调用 qh_printfacet 函数打印 facet 的信息
      break;  // 找到匹配的 facet 后退出循环
    }
  }
} /* dfacet */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="dvertex">-</a>

  qh_dvertex(qh, id )
    print vertex by id, for debugging
*/
void qh_dvertex(qhT *qh, unsigned int id) {
  vertexT *vertex;

  FORALLvertices {  // 遍历所有的 vertexT 结构体
    if (vertex->id == id) {  // 如果找到与给定 id 匹配的 vertex
      qh_printvertex(qh, qh->fout, vertex);  // 调用 qh_printvertex 函数打印 vertex 的信息
      break;  // 找到匹配的 vertex 后退出循环
    }
  }
} /* dvertex */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="compare_facetarea">-</a>

  qh_compare_facetarea( p1, p2 )
    used by qsort() to order facets by area
*/
int qh_compare_facetarea(const void *p1, const void *p2) {
  const facetT *a= *((facetT *const*)p1), *b= *((facetT *const*)p2);

  if (!a->isarea)  // 如果 a 不是面积 facet，则优先排序
    return -1;
  if (!b->isarea)  // 如果 b 不是面积 facet，则排在 a 之后
    return 1;
  if (a->f.area > b->f.area)  // 按面积大小升序排序
    return 1;
  else if (a->f.area == b->f.area)  // 面积相等时不改变顺序
    return 0;
  return -1;  // 其他情况按面积大小降序排序
} /* compare_facetarea */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="compare_facetvisit">-</a>

  qh_compare_facetvisit( p1, p2 )
    used by qsort() to order facets by visit id or id
*/
int qh_compare_facetvisit(const void *p1, const void *p2) {
  const facetT *a= *((facetT *const*)p1), *b= *((facetT *const*)p2);
  int i,j;

  if (!(i= (int)a->visitid))  // 如果 a 的 visitid 为 0，则使用 id 区分
    i= (int)(0 - a->id); /* sign distinguishes id from visitid */  // 用符号区分 id 和 visitid
  if (!(j= (int)b->visitid))  // 如果 b 的 visitid 为 0，则使用 id 区分
    j= (int)(0 - b->id);  // 用符号区分 id 和 visitid
  return(i - j);  // 按照 visitid 或 id 的顺序排序
} /* compare_facetvisit */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="compare_nummerge">-</a>

  qh_compare_nummerge( p1, p2 )
    used by qsort() to order facets by number of merges

notes:
    called by qh_markkeep ('PMerge-keep')
*/
int qh_compare_nummerge(const void *p1, const void *p2) {
  const facetT *a= *((facetT *const*)p1), *b= *((facetT *const*)p2);

  return(a->nummerge - b->nummerge);  // 按照 nummerge 的大小排序
} /* compare_nummerge */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="copyfilename">-</a>

  qh_copyfilename(qh, dest, size, source, length )
    copy filename identified by qh_skipfilename()

  notes:
    see qh_skipfilename() for syntax
*/
void qh_copyfilename(qhT *qh, char *filename, int size, const char* source, int length) {
  char c= *source;  // 获取源字符串的第一个字符

  if (length > size + 1) {  // 如果字符串长度超过指定大小加1
      qh_fprintf(qh, qh->ferr, 6040, "qhull error: filename is more than %d characters, %s\n",  size-1, source);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);  // 输出错误信息并退出程序
  }
  strncpy(filename, source, (size_t)length);  // 复制源字符串到目标字符串中，长度为指定长度
  filename[length]= '\0';  // 在目标字符串末尾添加结束符

  if (c == '\'' || c == '"') {  // 如果第一个字符是单引号或双引号
    char *s= filename + 1;  // s指向目标字符串的第二个字符
    char *t= filename;  // t指向目标字符串的第一个字符
    while (*s) {  // 遍历目标字符串
      if (*s == c) {  // 如果当前字符是与第一个字符相同的引号
          if (s[-1] == '\\')  // 如果前一个字符是反斜杠
              t[-1]= c;  // 将目标字符串中的反斜杠替换为引号
      }else
          *t++= *s;  // 复制字符到目标字符串
      s++;
    }
    *t= '\0';  // 添加目标字符串结束符
  }
} /* copyfilename */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="countfacets">-</a>

  qh_countfacets(qh, facetlist, facets, printall,
          numfacets, numsimplicial, totneighbors, numridges, numcoplanar, numtricoplanars  )
    count good facets for printing and set visitid
    if allfacets, ignores qh_skipfacet()

  notes:
    qh_printsummary and qh_countfacets must match counts

  returns:
    numfacets, numsimplicial, total neighbors, numridges, coplanars
    each facet with ->visitid indicating 1-relative position
      ->visitid==0 indicates not good

  notes
    numfacets >= numsimplicial
    if qh.NEWfacets,
      does not count visible facets (matches qh_printafacet)

  design:
    for all facets on facetlist and in facets set
      unless facet is skipped or visible (i.e., will be deleted)
        mark facet->visitid
        update counts
*/
void qh_countfacets(qhT *qh, facetT *facetlist, setT *facets, boolT printall,
    int *numfacetsp, int *numsimplicialp, int *totneighborsp, int *numridgesp, int *numcoplanarsp, int *numtricoplanarsp) {
  facetT *facet, **facetp;
  int numfacets= 0, numsimplicial= 0, numridges= 0, totneighbors= 0, numcoplanars= 0, numtricoplanars= 0;

  FORALLfacet_(facetlist) {  // 遍历 facetlist 中的所有 facet
    if ((facet->visible && qh->NEWfacets)  // 如果 facet 可见且存在 NEWfacets
    || (!printall && qh_skipfacet(qh, facet)))  // 或者不打印全部且跳过该 facet
      facet->visitid= 0;  // 将 visitid 设为 0
    else {
      facet->visitid= (unsigned int)(++numfacets);  // 设定 visitid 为递增的 facet 数量
      totneighbors += qh_setsize(qh, facet->neighbors);  // 计算邻居总数
      if (facet->simplicial) {  // 如果 facet 是简单的
        numsimplicial++;  // 增加简单 facet 的计数
        if (facet->keepcentrum && facet->tricoplanar)  // 如果保留中心并且是三共面
          numtricoplanars++;  // 增加三共面 facet 的计数
      } else
        numridges += qh_setsize(qh, facet->ridges);  // 增加非简单 facet 的计数
      if (facet->coplanarset)  // 如果存在共面集合
        numcoplanars += qh_setsize(qh, facet->coplanarset);  // 计算共面集合的大小
    }
  }

  FOREACHfacet_(facets) {  // 遍历 facets 中的所有 facet
    if ((facet->visible && qh->NEWfacets)  // 如果 facet 可见且存在 NEWfacets
    || (!printall && qh_skipfacet(qh, facet)))  // 或者不打印全部且跳过该 facet
      facet->visitid= 0;  // 将 visitid 设为 0
    else {
      facet->visitid= (unsigned int)(++numfacets);  // 设定 visitid 为递增的 facet 数量
      totneighbors += qh_setsize(qh, facet->neighbors);  // 计算邻居总数
      if (facet->simplicial) {
        numsimplicial++;  // 增加简单 facet 的计数
        if (facet->keepcentrum && facet->tricoplanar)  // 如果保留中心并且是三共面
          numtricoplanars++;  // 增加三共面 facet 的计数
      } else
        numridges += qh_setsize(qh, facet->ridges);  // 增加非简单 facet 的计数
      if (facet->coplanarset)  // 如果存在共面集合
        numcoplanars += qh_setsize(qh, facet->coplanarset);  // 计算共面集合的大小
    }
  }
    }
  }
  qh->visit_id += (unsigned int)numfacets + 1;
  // 更新 visit_id 字段，加上 numfacets 的值加一，转换为无符号整数类型

  *numfacetsp= numfacets;
  // 将 numfacets 的值赋给 numfacetsp 指向的位置，用于返回给调用者

  *numsimplicialp= numsimplicial;
  // 将 numsimplicial 的值赋给 numsimplicialp 指向的位置，用于返回给调用者

  *totneighborsp= totneighbors;
  // 将 totneighbors 的值赋给 totneighborsp 指向的位置，用于返回给调用者

  *numridgesp= numridges;
  // 将 numridges 的值赋给 numridgesp 指向的位置，用于返回给调用者

  *numcoplanarsp= numcoplanars;
  // 将 numcoplanars 的值赋给 numcoplanarsp 指向的位置，用于返回给调用者

  *numtricoplanarsp= numtricoplanars;
  // 将 numtricoplanars 的值赋给 numtricoplanarsp 指向的位置，用于返回给调用者
```c`
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="detvnorm">-</a>

  qh_detvnorm(qh, vertex, vertexA, centers, offset )
    compute separating plane of the Voronoi diagram for a pair of input sites
    centers= set of facets (i.e., Voronoi vertices)
      facet->visitid= 0 iff vertex-at-infinity (i.e., unbounded)

  assumes:
    qh_ASvoronoi and qh_vertexneighbors() already set

  returns:
    norm
      a pointer into qh.gm_matrix to qh.hull_dim-1 reals
      copy the data before reusing qh.gm_matrix
    offset
      if 'QVn'
        sign adjusted so that qh.GOODvertexp is inside
      else
        sign adjusted so that vertex is inside

    qh.gm_matrix= simplex of points from centers relative to first center

  notes:
    in io_r.c so that code for 'v Tv' can be removed by removing io_r.c
    returns pointer into qh.gm_matrix to avoid tracking of temporary memory

  design:
    determine midpoint of input sites
    build points as the set of Voronoi vertices
    select a simplex from points (if necessary)
      include midpoint if the Voronoi region is unbounded
    relocate the first vertex of the simplex to the origin
    compute the normalized hyperplane through the simplex
    orient the hyperplane toward 'QVn' or 'vertex'
    if 'Tv' or 'Ts'
      if bounded
        test that hyperplane is the perpendicular bisector of the input sites
      test that Voronoi vertices not in the simplex are still on the hyperplane
    free up temporary memory
*/
pointT *qh_detvnorm(qhT *qh, vertexT *vertex, vertexT *vertexA, setT *centers, realT *offsetp) {
  facetT *facet, **facetp;
  int  i, k, pointid, pointidA, point_i, point_n;
  setT *simplex= NULL;
  pointT *point, **pointp, *point0, *midpoint, *normal, *inpoint;
  coordT *coord, *gmcoord, *normalp;
  setT *points= qh_settemp(qh, qh->TEMPsize);
  boolT nearzero= False;
  boolT unbounded= False;
  int numcenters= 0;
  int dim= qh->hull_dim - 1;
  realT dist, offset, angle, zero= 0.0;

  midpoint= qh->gm_matrix + qh->hull_dim * qh->hull_dim;  /* last row */
  for (k=0; k < dim; k++)
    midpoint[k]= (vertex->point[k] + vertexA->point[k])/2;
  FOREACHfacet_(centers) {
    numcenters++;
    if (!facet->visitid)
      unbounded= True;
    else {
      if (!facet->center)
        facet->center= qh_facetcenter(qh, facet->vertices);
      qh_setappend(qh, &points, facet->center);
    }
  }
  if (numcenters > dim) {
    simplex= qh_settemp(qh, qh->TEMPsize);
    qh_setappend(qh, &simplex, vertex->point);
    if (unbounded)
      qh_setappend(qh, &simplex, midpoint);
    qh_maxsimplex(qh, dim, points, NULL, 0, &simplex);
    qh_setdelnth(qh, simplex, 0);
  }else if (numcenters == dim) {
    if (unbounded)
      qh_setappend(qh, &points, midpoint);
    simplex= points;
  }else {
    qh_fprintf(qh, qh->ferr, 6216, "qhull internal error (qh_detvnorm): too few points(%d) to compute separating plane\n", numcenters);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  i= 0;
  gmcoord= qh->gm_matrix;
  point0= SETfirstt_(simplex, pointT);
  FOREACHpoint_(simplex) {
    if (qh->IStracing >= 4)
      qh_printmatrix(qh, qh->ferr, "qh_detvnorm: Voronoi vertex or midpoint",
                              &point, 1, dim);
    // 如果点不是简单形的第一个点，则计算其在几何矩阵中的位置
    if (point != point0) {
      qh->gm_row[i++]= gmcoord;
      coord= point0;
      // 计算当前点与第一个点之间的几何矩阵差值
      for (k=dim; k--; )
        *(gmcoord++)= *point++ - *coord++;
    }
  }
  // 将当前几何矩阵位置记录为不与中点重叠，可能稍后用于 qh_areasimplex
  qh->gm_row[i]= gmcoord;
  normal= gmcoord;
  // 计算超平面法向量及偏移量
  qh_sethyperplane_gauss(qh, dim, qh->gm_row, point0, True,
                normal, &offset, &nearzero);
  // 确定使用哪个点来计算距离
  if (qh->GOODvertexp == vertexA->point)
    inpoint= vertexA->point;
  else
    inpoint= vertex->point;
  zinc_(Zdistio);
  // 计算点到超平面的距离
  dist= qh_distnorm(dim, inpoint, normal, &offset);
  // 如果距离为正，则反转超平面方向
  if (dist > 0) {
    offset= -offset;
    normalp= normal;
    // 反转法向量方向
    for (k=dim; k--; ) {
      *normalp= -(*normalp);
      normalp++;
    }
  }
  // 如果需要验证输出或打印统计信息
  if (qh->VERIFYoutput || qh->PRINTstatistics) {
    pointid= qh_pointid(qh, vertex->point);
    pointidA= qh_pointid(qh, vertexA->point);
    // 如果不是无界的情况下
    if (!unbounded) {
      zinc_(Zdiststat);
      // 计算中点到超平面的距离
      dist= qh_distnorm(dim, midpoint, normal, &offset);
      if (dist < 0)
        dist= -dist;
      // 统计中点距离相关信息
      zzinc_(Zridgemid);
      wwmax_(Wridgemidmax, dist);
      wwadd_(Wridgemid, dist);
      trace4((qh, qh->ferr, 4014, "qh_detvnorm: points %d %d midpoint dist %2.2g\n",
                 pointid, pointidA, dist));
      // 计算中点在法向量方向上的单位向量
      for (k=0; k < dim; k++)
        midpoint[k]= vertexA->point[k] - vertex->point[k];  /* overwrites midpoint! */
      qh_normalize(qh, midpoint, dim, False);
      // 计算法向量与中点的夹角
      angle= qh_distnorm(dim, midpoint, normal, &zero); /* qh_detangle uses dim+1 */
      if (angle < 0.0)
        angle= angle + 1.0;
      else
        angle= angle - 1.0;
      if (angle < 0.0)
        angle= -angle;
      // 统计夹角相关信息
      trace4((qh, qh->ferr, 4015, "qh_detvnorm: points %d %d angle %2.2g nearzero %d\n",
                 pointid, pointidA, angle, nearzero));
      if (nearzero) {
        zzinc_(Zridge0);
        wwmax_(Wridge0max, angle);
        wwadd_(Wridge0, angle);
      }else {
        zzinc_(Zridgeok)
        wwmax_(Wridgeokmax, angle);
        wwadd_(Wridgeok, angle);
      }
    }
    // 如果简单形不等于点集，则继续处理
    if (simplex != points) {
      FOREACHpoint_i_(qh, points) {
        // 如果点不在简单形内，则进行距离计算和统计
        if (!qh_setin(simplex, point)) {
          facet= SETelemt_(centers, point_i, facetT);
          zinc_(Zdiststat);
          dist= qh_distnorm(dim, point, normal, &offset);
          if (dist < 0)
            dist= -dist;
          // 统计点到超平面的距离信息
          zzinc_(Zridge);
          wwmax_(Wridgemax, dist);
          wwadd_(Wridge, dist);
          trace4((qh, qh->ferr, 4016, "qh_detvnorm: points %d %d Voronoi vertex %d dist %2.2g\n",
                             pointid, pointidA, facet->visitid, dist));
        }
      }
    }
  }
  // 将偏移量返回给调用者
  *offsetp= offset;
  // 如果简单形不等于点集，则释放简单形
  if (simplex != points)
    qh_settempfree(qh, &simplex);
  // 释放点集
  qh_settempfree(qh, &points);
  // 返回计算出的法向量
  return normal;
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="detvridge">-</a>

  qh_detvridge(qh, vertexA )
    determine Voronoi ridge from 'seen' neighbors of vertexA
    include one vertex-at-infinite if an !neighbor->visitid

  returns:
    temporary set of centers (facets, i.e., Voronoi vertices)
    sorted by center id
*/
setT *qh_detvridge(qhT *qh, vertexT *vertex) {
  // 创建两个临时集合，用于存放中心点和三角形中心
  setT *centers= qh_settemp(qh, qh->TEMPsize);
  setT *tricenters= qh_settemp(qh, qh->TEMPsize);
  facetT *neighbor, **neighborp;
  boolT firstinf= True; // 布尔值，用于跟踪是否已经添加了无穷远点

  // 遍历顶点的每一个相邻面
  FOREACHneighbor_(vertex) {
    // 如果相邻面已经被处理过
    if (neighbor->seen) {
      // 如果该相邻面的 visitid 存在，且不是三角平面的情况下或者三角中心是唯一的，则添加到中心集合中
      if (neighbor->visitid) {
        if (!neighbor->tricoplanar || qh_setunique(qh, &tricenters, neighbor->center))
          qh_setappend(qh, &centers, neighbor);
      } else if (firstinf) { // 如果是第一个无穷远点
        firstinf= False;
        qh_setappend(qh, &centers, neighbor); // 添加到中心集合中
      }
    }
  }
  // 根据中心点的标识符对中心集合进行排序
  qsort(SETaddr_(centers, facetT), (size_t)qh_setsize(qh, centers),
             sizeof(facetT *), qh_compare_facetvisit);
  // 释放三角形中心集合的临时内存
  qh_settempfree(qh, &tricenters);
  // 返回中心集合
  return centers;
} /* detvridge */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="detvridge3">-</a>

  qh_detvridge3(qh, atvertex, vertex )
    determine 3-d Voronoi ridge from 'seen' neighbors of atvertex and vertex
    include one vertex-at-infinite for !neighbor->visitid
    assumes all facet->seen2= True

  returns:
    temporary set of centers (facets, i.e., Voronoi vertices)
    listed in adjacency order (!oriented)
    all facet->seen2= True

  design:
    mark all neighbors of atvertex
    for each adjacent neighbor of both atvertex and vertex
      if neighbor selected
        add neighbor to set of Voronoi vertices
*/
setT *qh_detvridge3(qhT *qh, vertexT *atvertex, vertexT *vertex) {
  // 创建两个临时集合，用于存放中心点和三角形中心
  setT *centers= qh_settemp(qh, qh->TEMPsize);
  setT *tricenters= qh_settemp(qh, qh->TEMPsize);
  facetT *neighbor, **neighborp, *facet= NULL;
  boolT firstinf= True; // 布尔值，用于跟踪是否已经添加了无穷远点

  // 标记所有 atvertex 的相邻面
  FOREACHneighbor_(atvertex)
    neighbor->seen2= False;

  // 遍历 vertex 的每一个相邻面，找到第一个未标记的面
  FOREACHneighbor_(vertex) {
    if (!neighbor->seen2) {
      facet= neighbor;
      break;
    }
  }

  // 进入循环，处理所有与 atvertex 和 vertex 相邻的面
  while (facet) {
    facet->seen2= True; // 标记该面为已处理

    // 如果相邻面已经被处理过
    if (neighbor->seen) {
      // 如果该相邻面的 visitid 存在，且不是三角平面的情况下或者三角中心是唯一的，则添加到中心集合中
      if (facet->visitid) {
        if (!facet->tricoplanar || qh_setunique(qh, &tricenters, facet->center))
          qh_setappend(qh, &centers, facet);
      } else if (firstinf) { // 如果是第一个无穷远点
        firstinf= False;
        qh_setappend(qh, &centers, facet); // 添加到中心集合中
      }
    }

    // 遍历当前面的每一个相邻面
    FOREACHneighbor_(facet) {
      // 如果相邻面未标记
      if (!neighbor->seen2) {
        // 如果该相邻面是 vertex 的相邻面，则终止循环
        if (qh_setin(vertex->neighbors, neighbor))
          break;
        else
          neighbor->seen2= True; // 标记相邻面为已处理
      }
    }

    // 移动到下一个相邻面
    facet= neighbor;
  }

  // 如果需要频繁检查，则执行一些额外操作（未给出）

  // 返回中心集合
  return centers;
} /* detvridge3 */
    # 遍历顶点的相邻点列表
    FOREACHneighbor_(vertex) {
        # 检查相邻点是否未被标记为已访问
        if (!neighbor->seen2) {
            # 如果未被标记为已访问，输出错误信息并终止程序
            qh_fprintf(qh, qh->ferr, 6217, "qhull internal error (qh_detvridge3): neighbors of vertex p%d are not connected at facet %d\n",
                       qh_pointid(qh, vertex->point), neighbor->id);
            qh_errexit(qh, qh_ERRqhull, neighbor, NULL);
        }
    }
    # 标记顶点相邻点为已访问
    FOREACHneighbor_(atvertex)
        neighbor->seen2= True;
    # 释放临时存储的三角形中心点集合
    qh_settempfree(qh, &tricenters);
    # 返回计算得到的中心点集合
    return centers;
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="eachvoronoi">-</a>

  qh_eachvoronoi(qh, fp, printvridge, vertex, visitall, innerouter, inorder )
    if visitall,
      visit all Voronoi ridges for vertex (i.e., an input site)
    else
      visit all unvisited Voronoi ridges for vertex
      all vertex->seen= False if unvisited
    assumes
      all facet->seen= False
      all facet->seen2= True (for qh_detvridge3)
      all facet->visitid == 0 if vertex_at_infinity
                         == index of Voronoi vertex
                         >= qh.num_facets if ignored
    innerouter:
      qh_RIDGEall--  both inner (bounded) and outer(unbounded) ridges
      qh_RIDGEinner- only inner
      qh_RIDGEouter- only outer

    if inorder
      orders vertices for 3-d Voronoi diagrams

  returns:
    number of visited ridges (does not include previously visited ridges)

    if printvridge,
      calls printvridge( fp, vertex, vertexA, centers)
        fp== any pointer (assumes FILE*)
             fp may be NULL for QhullQh::qh_fprintf which calls appendQhullMessage
        vertex,vertexA= pair of input sites that define a Voronoi ridge
        centers= set of facets (i.e., Voronoi vertices)
                 ->visitid == index or 0 if vertex_at_infinity
                 ordered for 3-d Voronoi diagram
  notes:
    uses qh.vertex_visit

  see:
    qh_eachvoronoi_all()

  design:
    mark selected neighbors of atvertex
    for each selected neighbor (either Voronoi vertex or vertex-at-infinity)
      for each unvisited vertex
        if atvertex and vertex share more than d-1 neighbors
          bump totalcount
          if printvridge defined
            build the set of shared neighbors (i.e., Voronoi vertices)
            call printvridge
*/

int qh_eachvoronoi(qhT *qh, FILE *fp, printvridgeT printvridge, vertexT *atvertex, boolT visitall, qh_RIDGE innerouter, boolT inorder) {
  boolT unbounded;
  int count;
  facetT *neighbor, **neighborp, *neighborA, **neighborAp;
  setT *centers;
  setT *tricenters= qh_settemp(qh, qh->TEMPsize);  /* 临时的三角形中心点集合 */

  vertexT *vertex, **vertexp;
  boolT firstinf;
  unsigned int numfacets= (unsigned int)qh->num_facets;
  int totridges= 0;  /* 初始化总边数为 0 */

  qh->vertex_visit++;  /* 增加顶点访问计数 */

  atvertex->seen= True;  /* 标记当前顶点为已访问 */

  if (visitall) {
    FORALLvertices
      vertex->seen= False;  /* 标记所有顶点为未访问 */
  }

  FOREACHneighbor_(atvertex) {
    if (neighbor->visitid < numfacets)
      neighbor->seen= True;  /* 标记当前顶点的相邻面为已访问 */
  }

  FOREACHneighbor_(atvertex) {
    // 检查邻居是否已被访问过
    if (neighbor->seen) {
      // 遍历邻居顶点列表中的每一个顶点
      FOREACHvertex_(neighbor->vertices) {
        // 如果顶点尚未被当前遍历标识访问过，并且未被访问过
        if (vertex->visitid != qh->vertex_visit && !vertex->seen) {
          // 标记顶点为当前遍历标识访问过
          vertex->visitid = qh->vertex_visit;
          // 初始化计数器和首次无穷标志
          count = 0;
          firstinf = True;
          // 清空三角中心点集合
          qh_settruncate(qh, tricenters, 0);
          // 遍历顶点的相邻邻居
          FOREACHneighborA_(vertex) {
            // 如果相邻邻居已被访问过
            if (neighborA->seen) {
              // 如果相邻邻居的访问标识存在
              if (neighborA->visitid) {
                // 如果相邻邻居不是三角形的平面或者三角中心点在集合中唯一，则增加计数器
                if (!neighborA->tricoplanar || qh_setunique(qh, &tricenters, neighborA->center))
                  count++;
              } else if (firstinf) {
                // 第一个无穷相邻邻居，增加计数器，并标记非首次无穷
                count++;
                firstinf = False;
              }
            }
          }
          // 如果计数大于等于凸壳维度减一（例如，3维Voronoi图中的3个顶点）
          if (count >= qh->hull_dim - 1) {
            // 如果存在首次无穷相邻邻居
            if (firstinf) {
              // 如果是外部边界，继续下一个顶点的处理
              if (innerouter == qh_RIDGEouter)
                continue;
              // 设置为有界的Voronoi边界
              unbounded = False;
            } else {
              // 如果是内部边界，继续下一个顶点的处理
              if (innerouter == qh_RIDGEinner)
                continue;
              // 设置为无界的Voronoi边界
              unbounded = True;
            }
            // 增加到总边数中
            totridges++;
            // 输出Voronoi边界信息到跟踪日志中
            trace4((qh, qh->ferr, 4017, "qh_eachvoronoi: Voronoi ridge of %d vertices between sites %d and %d\n",
                  count, qh_pointid(qh, atvertex->point), qh_pointid(qh, vertex->point)));
            // 如果需要打印Voronoi边界
            if (printvridge) {
              // 如果按顺序且凸壳维度为3+1，则为3维Voronoi图
              if (inorder && qh->hull_dim == 3 + 1)
                centers = qh_detvridge3(qh, atvertex, vertex);
              else
                centers = qh_detvridge(qh, vertex);
              // 调用打印函数打印Voronoi边界
              (*printvridge)(qh, fp, atvertex, vertex, centers, unbounded);
              // 释放临时分配的中心点集合
              qh_settempfree(qh, &centers);
            }
          }
        }
      }
    }
  }
  // 将atvertex的所有邻居的seen标志设置为False
  FOREACHneighbor_(atvertex)
    neighbor->seen = False;
  // 释放临时分配的三角中心点集合
  qh_settempfree(qh, &tricenters);
  // 返回计算得到的总边数
  return totridges;
} /* eachvoronoi */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="eachvoronoi_all">-</a>

  qh_eachvoronoi_all(qh, fp, printvridge, isUpper, innerouter, inorder )
    visit all Voronoi ridges

    innerouter:
      see qh_eachvoronoi()

    if inorder
      orders vertices for 3-d Voronoi diagrams

  returns
    total number of ridges

    if isUpper == facet->upperdelaunay  (i.e., a Vornoi vertex)
      facet->visitid= Voronoi vertex index(same as 'o' format)
    else
      facet->visitid= 0

    if printvridge,
      calls printvridge( fp, vertex, vertexA, centers)
      [see qh_eachvoronoi]

  notes:
    Not used for qhull.exe
    same effect as qh_printvdiagram but ridges not sorted by point id
*/
int qh_eachvoronoi_all(qhT *qh, FILE *fp, printvridgeT printvridge, boolT isUpper, qh_RIDGE innerouter, boolT inorder) {
  facetT *facet;
  vertexT *vertex;
  int numcenters= 1;  /* vertex 0 is vertex-at-infinity */
  int totridges= 0;

  // 清除 Voronoi 中心并设置
  qh_clearcenters(qh, qh_ASvoronoi);
  // 计算顶点的邻居
  qh_vertexneighbors(qh);
  // 将 visit_id 最大化为当前面数
  maximize_(qh->visit_id, (unsigned int)qh->num_facets);
  // 遍历所有面
  FORALLfacets {
    facet->visitid= 0;
    facet->seen= False;
    facet->seen2= True;
  }
  // 再次遍历所有面，根据 upperdelaunay 属性设置 visitid
  FORALLfacets {
    if (facet->upperdelaunay == isUpper)
      facet->visitid= (unsigned int)(numcenters++);
  }
  // 将所有顶点的 seen 属性设置为 False
  FORALLvertices
    vertex->seen= False;
  // 再次遍历所有顶点
  FORALLvertices {
    // 如果 GOODvertex 存在且当前顶点不等于 GOODvertex 对应的点 ID，则跳过
    if (qh->GOODvertex > 0 && qh_pointid(qh, vertex->point)+1 != qh->GOODvertex)
      continue;
    // 调用 qh_eachvoronoi 处理 Voronoi 图的每个顶点
    totridges += qh_eachvoronoi(qh, fp, printvridge, vertex,
                   !qh_ALL, innerouter, inorder);
  }
  // 返回总的 Voronoi 边数
  return totridges;
} /* eachvoronoi_all */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="facet2point">-</a>

  qh_facet2point(qh, facet, point0, point1, mindist )
    return two projected temporary vertices for a 2-d facet
    may be non-simplicial

  returns:
    point0 and point1 oriented and projected to the facet
    returns mindist (maximum distance below plane)
*/
void qh_facet2point(qhT *qh, facetT *facet, pointT **point0, pointT **point1, realT *mindist) {
  vertexT *vertex0, *vertex1;
  realT dist;

  // 根据 facet 的方向确定顶点顺序
  if (facet->toporient ^ qh_ORIENTclock) {
    vertex0= SETfirstt_(facet->vertices, vertexT);
    vertex1= SETsecondt_(facet->vertices, vertexT);
  }else {
    vertex1= SETfirstt_(facet->vertices, vertexT);
    vertex0= SETsecondt_(facet->vertices, vertexT);
  }
  zadd_(Zdistio, 2);
  // 计算顶点 vertex0 到 facet 的距离
  qh_distplane(qh, vertex0->point, facet, &dist);
  *mindist= dist;
  // 投影顶点 vertex0 到 facet 上
  *point0= qh_projectpoint(qh, vertex0->point, facet, dist);
  // 计算顶点 vertex1 到 facet 的距离
  qh_distplane(qh, vertex1->point, facet, &dist);
  // 更新最小距离
  minimize_(*mindist, dist);
  // 投影顶点 vertex1 到 facet 上
  *point1= qh_projectpoint(qh, vertex1->point, facet, dist);
} /* facet2point */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="facetvertices">-</a>

  qh_facetvertices(qh, facetlist, facets, allfacets )
    # 返回一个临时的顶点集合和/或面片列表，如果包括所有面片（allfacets），则忽略 qh_skipfacet()
    # 使用 qh.vertex_visit 返回顶点
    
    # 优化说明：针对 facet_list 的所有面片（allfacets）进行了优化
    
    # 设计思路：
    # 如果处理 facet_list 的所有面片（allfacets）
    #   从 vertex_list 创建顶点集合
    # 否则
    #   对于选定的每个面片（facets 或 facetlist）
    #     将未访问的顶点追加到顶点集合中
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="markkeep">-</a>

  qh_markkeep(qh, facetlist )
    restrict good facets for qh.KEEParea, qh.KEEPmerge, and qh.KEEPminArea
    ignores visible facets (!part of convex hull)

  returns:
    may clear facet->good
    recomputes qh.num_good

  notes:
    only called by qh_prepare_output after qh_findgood_all
    does not throw errors except memory/corruption of qset_r.c

  design:
    get set of good facets
    if qh.KEEParea
      sort facets by area
      clear facet->good for all but n largest facets
    if qh.KEEPmerge
      sort facets by merge count
      clear facet->good for all but n most merged facets
    if qh.KEEPminarea
      clear facet->good if area too small
    update qh.num_good
*/
void qh_markkeep(qhT *qh, facetT *facetlist) {
    facetT *facet, **facetp;  // 声明 facet 和 facetp 变量，facetp 是指向 facet 的指针数组
    setT *facets = qh_settemp(qh, qh->num_facets);  // 创建临时集合 facets，用于存储 facetT 结构的指针
    int size, count;  // 声明整型变量 size 和 count

    trace2((qh, qh->ferr, 2006, "qh_markkeep: only keep %d largest and/or %d most merged facets and/or min area %.2g\n",
            qh->KEEParea, qh->KEEPmerge, qh->KEEPminArea));
    // 记录调试信息，打印保留的最大面积、最合并和/或最小面积的要求

    FORALLfacet_(facetlist) {
        // 遍历 facetlist 中的每个 facet
        if (!facet->visible && facet->good)
            // 如果 facet 不可见且标记为 good
            qh_setappend(qh, &facets, facet);
            // 将 facet 添加到 facets 集合中
    }

    size = qh_setsize(qh, facets);  // 获取 facets 集合的大小
    if (qh->KEEParea) {
        // 如果需要保留最大面积的 facet
        qsort(SETaddr_(facets, facetT), (size_t)size,
             sizeof(facetT *), qh_compare_facetarea);
        // 对 facets 集合中的 facet 按面积排序
        if ((count = size - qh->KEEParea) > 0) {
            // 计算需要移除的 facet 数量
            FOREACHfacet_(facets) {
                // 遍历 facets 集合
                facet->good = False;  // 将 facet 的 good 标记设置为 False
                if (--count == 0)
                    break;
                // 如果达到移除的数量，退出循环
            }
        }
    }
    if (qh->KEEPmerge) {
        // 如果需要保留最多合并的 facet
        qsort(SETaddr_(facets, facetT), (size_t)size,
             sizeof(facetT *), qh_compare_nummerge);
        // 对 facets 集合中的 facet 按合并次数排序
        if ((count = size - qh->KEEPmerge) > 0) {
            // 计算需要移除的 facet 数量
            FOREACHfacet_(facets) {
                // 遍历 facets 集合
                facet->good = False;  // 将 facet 的 good 标记设置为 False
                if (--count == 0)
                    break;
                // 如果达到移除的数量，退出循环
            }
        }
    }
    if (qh->KEEPminArea < REALmax / 2) {
        // 如果需要保留最小面积大于指定值的 facet
        FOREACHfacet_(facets) {
            // 遍历 facets 集合
            if (!facet->isarea || facet->f.area < qh->KEEPminArea)
                // 如果 facet 不是面积或其面积小于指定值
                facet->good = False;  // 将 facet 的 good 标记设置为 False
        }
    }
    qh_settempfree(qh, &facets);  // 释放 facets 集合的临时内存
    count = 0;
    FORALLfacet_(facetlist) {
        // 再次遍历 facetlist 中的每个 facet
        if (facet->good)
            // 如果 facet 被标记为 good
            count++;  // 计数增加
    }
    qh->num_good = count;  // 更新 qh 中的 num_good 字段
} /* markkeep */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="markvoronoi">-</a>

  qh_markvoronoi(qh, facetlist, facets, printall, isLower, numcenters )
    mark voronoi vertices for printing by site pairs

  returns:
    temporary set of vertices indexed by pointid
    isLower set if printing lower hull (i.e., at least one facet is lower hull)
    numcenters= total number of Voronoi vertices
    bumps qh.printoutnum for vertex-at-infinity
    clears all facet->seen and sets facet->seen2

    if selected
      facet->visitid= Voronoi vertex id
    else if upper hull (or 'Qu' and lower hull)
      facet->visitid= 0
    else
      facet->visitid >= qh->num_facets

  notes:
    ignores qh.ATinfinity, if defined
*/
setT *qh_markvoronoi(qhT *qh, facetT *facetlist, setT *facets, boolT printall, boolT *isLowerp, int *numcentersp) {
    int numcenters = 0;
    facetT *facet, **facetp;  // 声明 facet 和 facetp 变量，facetp 是指向 facet 的指针数组
    setT *vertices;  // 声明 vertices 变量，用于存储点集合的指针
    boolT isLower = False;  // 声明并初始化 isLower 变量为 False

    qh->printoutnum++;  // 增加 qh 中的 printoutnum 字段值
    qh_clearcenters(qh, qh_ASvoronoi);  // 清除 qh 中的 Voronoi 中心点集合
    qh_vertexneighbors(qh);  // 计算 qh 中的顶点邻居信息
    vertices = qh_pointvertex(qh);  // 获取 qh 中的点与顶点映射关系
    if (qh->ATinfinity)
        SETelem_(vertices, qh->num_points - 1) = NULL;
    // 如果定义了 qh.ATinfinity，将 vertices 中的最后一个元素设为 NULL
    qh->visit_id++;  // 增加 qh 中的 visit_id 字段
    maximize_(qh->visit_id, (unsigned int)qh->num_facets);
    // 将 qh 中的 visit_id 字段设置为 qh.num_facets 和当前 visit_id 的最大值

    FORALLfacet_(facetlist) {
        // 遍历 facetlist 中的每个 facet
        if (printall || !qh_skipfacet(qh, facet)) {
            // 如果需要打印所有或不需要跳过当前 facet
            if (!facet->upperdelaunay) {
                // 如果当前 facet 不是上半 Delaunay 三角剖分
                isLower = True;  // 设置 isLower 变量为 True
                break;  // 退出循环
            }
        }
    }

    FOREACHfacet_(facets) {
        // 遍历 facets 集合中的每个 facet
        if (printall || !qh_skipfacet(qh, facet)) {
            // 如果需要打印所有或不需要跳过当前 facet
            if (!facet->upperdelaunay) {
                // 如果当前 facet 不是上半 Delaunay 三角剖分
                isLower = True;  // 设置 isLower 变量为 True
                break;  // 退出循环
            }
        }
    }

    FORALLfacets {
        // 遍历所有 facet
    // 如果 facet 的 normal 存在且 upperdelaunay 等于 isLower，则将 visitid 设为 0
    // 否则将 visitid 设为 qh->visit_id
    if (facet->normal && (facet->upperdelaunay == isLower))
      facet->visitid= 0;  /* facetlist or facets may overwrite */
    else
      facet->visitid= qh->visit_id;

    // 将 facet 的 seen 设为 False
    facet->seen= False;
    
    // 将 facet 的 seen2 设为 True
    facet->seen2= True;
  }

  // 增加 numcenters 计数（对应于 qh_INFINITE）
  numcenters++;  /* qh_INFINITE */

  // 遍历 facetlist 中的所有 facet
  FORALLfacet_(facetlist) {
    // 如果 printall 为真或者 qh_skipfacet 函数对当前 facet 返回假，则设置 facet 的 visitid 为 numcenters，并增加 numcenters
    if (printall || !qh_skipfacet(qh, facet))
      facet->visitid= (unsigned int)(numcenters++);
  }

  // 遍历 facets 链表中的所有 facet
  FOREACHfacet_(facets) {
    // 如果 printall 为真或者 qh_skipfacet 函数对当前 facet 返回假，则设置 facet 的 visitid 为 numcenters，并增加 numcenters
    if (printall || !qh_skipfacet(qh, facet))
      facet->visitid= (unsigned int)(numcenters++);
  }

  // 将 isLowerp 指向的值设为 isLower
  *isLowerp= isLower;

  // 将 numcentersp 指向的值设为 numcenters
  *numcentersp= numcenters;

  // 输出调试信息，显示 isLower 和 numcenters 的当前值
  trace2((qh, qh->ferr, 2007, "qh_markvoronoi: isLower %d numcenters %d\n", isLower, numcenters));

  // 返回 vertices 指针
  return vertices;
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="order_vertexneighbors">-</a>

  qh_order_vertexneighbors(qh, vertex )
    order facet neighbors of a 2-d or 3-d vertex by adjacency

  notes:
    does not orient the neighbors

  design:
    initialize a new neighbor set with the first facet in vertex->neighbors
    while vertex->neighbors non-empty
      select next neighbor in the previous facet's neighbor set
    set vertex->neighbors to the new neighbor set
*/
void qh_order_vertexneighbors(qhT *qh, vertexT *vertex) {
  setT *newset;
  facetT *facet, *neighbor, **neighborp;

  trace4((qh, qh->ferr, 4018, "qh_order_vertexneighbors: order neighbors of v%d for 3-d\n", vertex->id));
  // 创建一个临时集合用于存放重新排序后的邻居列表
  newset= qh_settemp(qh, qh_setsize(qh, vertex->neighbors));
  // 从 vertex->neighbors 中删除最后一个 facet，并将其添加到 newset 中作为初始邻居
  facet= (facetT *)qh_setdellast(vertex->neighbors);
  qh_setappend(qh, &newset, facet);
  // 当 vertex->neighbors 非空时进行循环
  while (qh_setsize(qh, vertex->neighbors)) {
    // 遍历 vertex 的每个邻居
    FOREACHneighbor_(vertex) {
      // 如果当前 facet 的邻居集合中包含 neighbor
      if (qh_setin(facet->neighbors, neighbor)) {
        // 从 vertex->neighbors 中删除 neighbor，将 neighbor 添加到 newset 中，并更新当前 facet 为 neighbor
        qh_setdel(vertex->neighbors, neighbor);
        qh_setappend(qh, &newset, neighbor);
        facet= neighbor;
        break;
      }
    }
    // 如果没有找到邻居，输出错误信息并退出
    if (!neighbor) {
      qh_fprintf(qh, qh->ferr, 6066, "qhull internal error (qh_order_vertexneighbors): no neighbor of v%d for f%d\n",
        vertex->id, facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
  }
  // 释放 vertex->neighbors 的旧内存空间，将 newset 设置为 vertex 的新邻居列表
  qh_setfree(qh, &vertex->neighbors);
  qh_settemppop(qh);
  vertex->neighbors= newset;
} /* order_vertexneighbors */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="prepare_output">-</a>

  qh_prepare_output(qh )
    prepare for qh_produce_output2(qh) according to
      qh.KEEPminArea, KEEParea, KEEPmerge, GOODvertex, GOODthreshold, GOODpoint, ONLYgood, SPLITthresholds
    does not reset facet->good

  notes
    called by qh_produce_output, qh_new_qhull, Qhull.outputQhull
    except for PRINTstatistics, no-op if previously called with same options
*/
void qh_prepare_output(qhT *qh) {
  // 如果进行 Voronoi 输出，清除中心点并重新计算邻居信息
  if (qh->VORONOI) {
    qh_clearcenters(qh, qh_ASvoronoi);
    qh_vertexneighbors(qh);
  }
  // 如果需要进行三角剖分且尚未进行过，执行三角剖分并检查多边形
  if (qh->TRIangulate && !qh->hasTriangulation) {
    qh_triangulate(qh);
    // 如果需要验证输出且不频繁检查，检查多边形
    if (qh->VERIFYoutput && !qh->CHECKfrequently)
      qh_checkpolygon(qh, qh->facet_list);
  }
  // 标记所有好的 facet
  qh_findgood_all(qh, qh->facet_list);
  // 如果需要计算面积，计算 facet_list 的面积
  if (qh->GETarea)
    qh_getarea(qh, qh->facet_list);
  // 如果需要保留面积、合并或最小面积小于 REALmax/2，标记保留的 facet
  if (qh->KEEParea || qh->KEEPmerge || qh->KEEPminArea < REALmax/2)
    qh_markkeep(qh, qh->facet_list);
  // 如果需要打印统计信息，收集统计数据
  if (qh->PRINTstatistics)
    qh_collectstatistics(qh);
}
    must match qh_countfacets



  notes
    preserves qh.visit_id
    facet->normal may be null if PREmerge/MERGEexact and STOPcone before merge



  see
    qh_printbegin() and qh_printend()



  design:
    test for printing facet
    call appropriate routine for format
    or output results directly


这段代码看起来像是注释和说明，描述了一些关于函数或者程序设计的设计考虑、实现细节或者相关的函数调用。
/*
void qh_printafacet(qhT *qh, FILE *fp, qh_PRINT format, facetT *facet, boolT printall) {
  realT color[4], offset, dist, outerplane, innerplane;
  boolT zerodiv;
  coordT *point, *normp, *coordp, **pointp, *feasiblep;
  int k;
  vertexT *vertex, **vertexp;
  facetT *neighbor, **neighborp;
*/

// 如果不需要打印所有细节并且跳过该面，直接返回
if (!printall && qh_skipfacet(qh, facet))
  return;

// 如果面是可见的且是新面，并且不是打印面格式，直接返回
if (facet->visible && qh->NEWfacets && format != qh_PRINTfacets)
  return;

// 增加打印输出计数
qh->printoutnum++;

// 根据打印格式进行处理
switch (format) {
case qh_PRINTarea:
  // 如果面是面积面，打印面积值
  if (facet->isarea) {
    qh_fprintf(qh, fp, 9009, qh_REAL_1, facet->f.area);
    qh_fprintf(qh, fp, 9010, "\n");
  } else
    // 否则打印 0
    qh_fprintf(qh, fp, 9011, "0\n");
  break;
case qh_PRINTcoplanars:
  // 打印共面点集的大小和每个点的标识
  qh_fprintf(qh, fp, 9012, "%d", qh_setsize(qh, facet->coplanarset));
  FOREACHpoint_(facet->coplanarset)
    qh_fprintf(qh, fp, 9013, " %d", qh_pointid(qh, point));
  qh_fprintf(qh, fp, 9014, "\n");
  break;
case qh_PRINTcentrums:
  // 打印中心
  qh_printcenter(qh, fp, format, NULL, facet);
  break;
case qh_PRINTfacets:
  // 打印面本身
  qh_printfacet(qh, fp, facet);
  break;
case qh_PRINTfacets_xridge:
  // 打印包含 ridge 的面的头部信息
  qh_printfacetheader(qh, fp, facet);
  break;
case qh_PRINTgeom:  /* either 2 , 3, or 4-d by qh_printbegin */
  // 如果没有法向量，跳过
  if (!facet->normal)
    break;
  
  // 计算颜色并进行归一化
  for (k = qh->hull_dim; k--; ) {
    color[k] = (facet->normal[k] + 1.0) / 2.0;
    maximize_(color[k], -1.0);
    minimize_(color[k], +1.0);
  }
  qh_projectdim3(qh, color, color);
  
  // 根据维度进行几何打印
  if (qh->PRINTdim != qh->hull_dim)
    qh_normalize2(qh, color, 3, True, NULL, NULL);
  if (qh->hull_dim <= 2)
    qh_printfacet2geom(qh, fp, facet, color);
  else if (qh->hull_dim == 3) {
    if (facet->simplicial)
      qh_printfacet3geom_simplicial(qh, fp, facet, color);
    else
      qh_printfacet3geom_nonsimplicial(qh, fp, facet, color);
  } else {
    if (facet->simplicial)
      qh_printfacet4geom_simplicial(qh, fp, facet, color);
    else
      qh_printfacet4geom_nonsimplicial(qh, fp, facet, color);
  }
  break;
case qh_PRINTids:
  // 打印面的标识号
  qh_fprintf(qh, fp, 9015, "%d\n", facet->id);
  break;
case qh_PRINTincidences:
case qh_PRINToff:
case qh_PRINTtriangles:
  // 根据维度和格式打印顶点
  if (qh->hull_dim == 3 && format != qh_PRINTtriangles)
    qh_printfacet3vertex(qh, fp, facet, format);
  else if (facet->simplicial || qh->hull_dim == 2 || format == qh_PRINToff)
    qh_printfacetNvertex_simplicial(qh, fp, facet, format);
  else
    qh_printfacetNvertex_nonsimplicial(qh, fp, facet, qh->printoutvar++, format);
  break;
case qh_PRINTinner:
  // 打印内部平面的偏移量
  qh_outerinner(qh, facet, NULL, &innerplane);
  offset = facet->offset - innerplane;
  goto LABELprintnorm;
  break; /* prevent warning */
case qh_PRINTmerges:
  // 打印融合次数
  qh_fprintf(qh, fp, 9016, "%d\n", facet->nummerge);
  break;
case qh_PRINTnormals:
  // 打印法向量
  offset = facet->offset;
  goto LABELprintnorm;
  break; /* prevent warning */
case qh_PRINTouter:
  // 打印外部平面
  qh_outerinner(qh, facet, &outerplane, NULL);
    offset= facet->offset - outerplane;

计算偏移量 `offset`，其值为 `facet->offset` 减去 `outerplane`。


  LABELprintnorm:

定义标签 `LABELprintnorm`，用于在需要时跳转到打印法向量的代码块。


    if (!facet->normal) {
      qh_fprintf(qh, fp, 9017, "no normal for facet f%d\n", facet->id);
      break;
    }

检查 `facet` 结构体是否有法向量 `facet->normal`。如果不存在法向量，则使用 `qh_fprintf` 输出错误信息到文件指针 `fp`，包含 facet ID，并终止当前代码块的执行。


    if (qh->CDDoutput) {
      qh_fprintf(qh, fp, 9018, qh_REAL_1, -offset);
      for (k=0; k < qh->hull_dim; k++)
        qh_fprintf(qh, fp, 9019, qh_REAL_1, -facet->normal[k]);
    }else {
      for (k=0; k < qh->hull_dim; k++)
        qh_fprintf(qh, fp, 9020, qh_REAL_1, facet->normal[k]);
      qh_fprintf(qh, fp, 9021, qh_REAL_1, offset);
    }

根据 `qh->CDDoutput` 的值选择打印的方式：
- 如果为真，先打印 `-offset`，然后打印法向量各分量的负值。
- 否则，依次打印法向量各分量的值，最后打印 `offset`。


    qh_fprintf(qh, fp, 9022, "\n");
    break;

打印换行符并结束当前打印法向量的代码块。


  case qh_PRINTmathematica:  /* either 2 or 3-d by qh_printbegin */
  case qh_PRINTmaple:
    if (qh->hull_dim == 2)
      qh_printfacet2math(qh, fp, facet, format, qh->printoutvar++);
    else
      qh_printfacet3math(qh, fp, facet, format, qh->printoutvar++);
    break;

根据 `qh->printformat` 的值选择打印数学软件（Mathematica 或 Maple）格式的代码块，调用对应的函数 `qh_printfacet2math` 或 `qh_printfacet3math`。


  case qh_PRINTneighbors:
    qh_fprintf(qh, fp, 9023, "%d", qh_setsize(qh, facet->neighbors));
    FOREACHneighbor_(facet)
      qh_fprintf(qh, fp, 9024, " %d",
               neighbor->visitid ? neighbor->visitid - 1: 0 - neighbor->id);
    qh_fprintf(qh, fp, 9025, "\n");
    break;

打印 facet 的邻居信息数目及其相应的 visitid（如果存在）。使用 `qh_setsize` 函数获取邻居数量，`FOREACHneighbor_` 宏遍历邻居列表，并根据 visitid 打印不同的值。


  case qh_PRINTpointintersect:
    if (!qh->feasible_point) {
      qh_fprintf(qh, qh->ferr, 6067, "qhull input error (qh_printafacet): option 'Fp' needs qh->feasible_point\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    if (facet->offset > 0)
      goto LABELprintinfinite;
    point= coordp= (coordT *)qh_memalloc(qh, qh->normal_size);
    normp= facet->normal;
    feasiblep= qh->feasible_point;
    if (facet->offset < -qh->MINdenom) {
      for (k=qh->hull_dim; k--; )
        *(coordp++)= (*(normp++) / - facet->offset) + *(feasiblep++);
    }else {
      for (k=qh->hull_dim; k--; ) {
        *(coordp++)= qh_divzero(*(normp++), facet->offset, qh->MINdenom_1,
                                 &zerodiv) + *(feasiblep++);
        if (zerodiv) {
          qh_memfree(qh, point, qh->normal_size);
          goto LABELprintinfinite;
        }
      }
    }
    qh_printpoint(qh, fp, NULL, point);
    qh_memfree(qh, point, qh->normal_size);
    break;

根据 `qh->feasible_point` 的存在与否，打印 intersect 点的信息：
- 如果 `qh->feasible_point` 不存在，打印错误信息并退出。
- 如果 `facet->offset > 0`，跳转到 `LABELprintinfinite` 标签。
- 计算 intersect 点的坐标，并使用 `qh_printpoint` 函数打印。


  LABELprintinfinite:
    for (k=qh->hull_dim; k--; )
      qh_fprintf(qh, fp, 9026, qh_REAL_1, qh_INFINITE);
    qh_fprintf(qh, fp, 9027, "\n");
    break;

打印无限点的信息，输出 `qh_INFINITE` 的值，然后换行。


  case qh_PRINTpointnearest:
    FOREACHpoint_(facet->coplanarset) {
      int id, id2;
      vertex= qh_nearvertex(qh, facet, point, &dist);
      id= qh_pointid(qh, vertex->point);
      id2= qh_pointid(qh, point);
      qh_fprintf(qh, fp, 9028, "%d %d %d " qh_REAL_1 "\n", id, id2, facet->id, dist);
    }
    break;

打印最接近点的信息，遍历 coplanarset 中的每个点，计算距离并打印相关信息。


  case qh_PRINTpoints:  /* VORONOI only by qh_printbegin */
    if (qh->CDDoutput)
      qh_fprintf(qh, fp, 9029, "1 ");
    qh_printcenter(qh, fp, format, NULL, facet);
    break;

根据 `qh->CDDoutput` 的值打印 Voronoi 点信息或者中心点信息。


  case qh_PRINTvertices:
    qh_fprintf(qh, fp, 9030, "%d", qh_setsize(qh, facet->vertices));
    FOREACHvertex_(facet->vertices)
      qh_fprintf(qh, fp, 9031, " %d", qh_pointid(qh, vertex->point));
    qh_fprintf(qh, fp, 9032, "\n");

打印 facet 的顶点信息数目及其对应的 ID 列表。
    break;
  default:
    break;
  }



    # 当前代码段是一个 switch 语句的结尾部分，用于处理所有未匹配到的情况。
    # 在 switch 语句中，break; 用于终止当前 case 分支的执行，并跳出 switch 语句。
    # default: 标签指定了在所有 case 标签都不匹配时执行的代码块。
    break;
  }


这段代码是一个 switch 语句的结尾部分，其中 `default:` 标签指定了在所有 case 标签都不匹配时执行的代码块。在每个 case 分支末尾通常都会有 `break;` 语句，用于结束当前 case 分支的执行并跳出 switch 语句。
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printbegin">-</a>

  qh_printbegin(qh )
    prints header for all output formats

  returns:
    checks for valid format

  notes:
    uses qh.visit_id for 3/4off
    changes qh.interior_point if printing centrums
    qh_countfacets clears facet->visitid for non-good facets

  see
    qh_printend() and qh_printafacet()

  design:
    count facets and related statistics
    print header for format
*/
void qh_printbegin(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall) {
  int numfacets, numsimplicial, numridges, totneighbors, numcoplanars, numtricoplanars;
  int i, num;
  facetT *facet, **facetp;
  vertexT *vertex, **vertexp;
  setT *vertices;
  pointT *point, **pointp, *pointtemp;

  qh->printoutnum= 0;  // 初始化输出计数为 0
  qh_countfacets(qh, facetlist, facets, printall, &numfacets, &numsimplicial,
      &totneighbors, &numridges, &numcoplanars, &numtricoplanars);  // 统计各种几何元素的数量

  switch (format) {
  case qh_PRINTnone:  // 如果输出格式为 none，不执行任何操作
    break;
  case qh_PRINTarea:  // 如果输出格式为 area，输出 numfacets 的值到文件流 fp
    qh_fprintf(qh, fp, 9033, "%d\n", numfacets);
    break;
  case qh_PRINTcoplanars:  // 如果输出格式为 coplanars，输出 numfacets 的值到文件流 fp
    qh_fprintf(qh, fp, 9034, "%d\n", numfacets);
    break;
  case qh_PRINTcentrums:  // 如果输出格式为 centrums
    if (qh->CENTERtype == qh_ASnone)  // 如果 qh->CENTERtype 为 none，则清除中心点
      qh_clearcenters(qh, qh_AScentrum);
    qh_fprintf(qh, fp, 9035, "%d\n%d\n", qh->hull_dim, numfacets);  // 输出维度和 numfacets 到文件流 fp
    break;
  case qh_PRINTfacets:  // 如果输出格式为 facets 或 facets_xridge
  case qh_PRINTfacets_xridge:
    if (facetlist)  // 如果 facetlist 不为空，则打印顶点列表
      qh_printvertexlist(qh, fp, "Vertices and facets:\n", facetlist, facets, printall);
    break;
  case qh_PRINTgeom:  // 如果输出格式为 geom
    if (qh->hull_dim > 4)  // 如果凸壳维度大于 4，跳转到 LABELnoformat 标签
      goto LABELnoformat;
    if (qh->VORONOI && qh->hull_dim > 3)  // 如果使用 Voronoi 且凸壳维度大于 3，跳转到 LABELnoformat 标签
      goto LABELnoformat;
    if (qh->hull_dim == 2 && (qh->PRINTridges || qh->DOintersections))  // 如果凸壳维度为 2 且需要打印 ridges 或 intersections，发出警告
      qh_fprintf(qh, qh->ferr, 7049, "qhull warning: output for ridges and intersections not implemented in 2-d\n");
    if (qh->hull_dim == 4 && (qh->PRINTinner || qh->PRINTouter ||
                             (qh->PRINTdim == 4 && qh->PRINTcentrums)))  // 如果凸壳维度为 4 且需要打印 inner/outer planes 或 centrums，发出警告
      qh_fprintf(qh, qh->ferr, 7050, "qhull warning: output for outer/inner planes and centrums not implemented in 4-d\n");
    if (qh->PRINTdim == 4 && (qh->PRINTspheres))  // 如果凸壳维度为 4 且需要打印 spheres，发出警告
      qh_fprintf(qh, qh->ferr, 7051, "qhull warning: output for vertices not implemented in 4-d\n");
    if (qh->PRINTdim == 4 && qh->DOintersections && qh->PRINTnoplanes)  // 如果凸壳维度为 4 且需要 intersections 但不需要 planes，发出警告
      qh_fprintf(qh, qh->ferr, 7052, "qhull warning: 'Gnh' generates no output in 4-d\n");
    if (qh->PRINTdim == 2) {  // 如果凸壳维度为 2，输出 rbox_command 和 qhull_command 到文件流 fp
      qh_fprintf(qh, fp, 9036, "{appearance {linewidth 3} LIST # %s | %s\n",
              qh->rbox_command, qh->qhull_command);
    }else if (qh->PRINTdim == 3) {  // 如果凸壳维度为 3，输出 rbox_command 和 qhull_command 到文件流 fp
      qh_fprintf(qh, fp, 9037, "{appearance {+edge -evert linewidth 2} LIST # %s | %s\n",
              qh->rbox_command, qh->qhull_command);
    }else if (qh->PRINTdim == 4) {
      // 如果 PRINTdim 等于 4，则执行以下操作
      qh->visit_id++;  // 增加访问标识符
      num= 0;  // 初始化 num 变量为 0
      FORALLfacet_(facetlist)    /* get number of ridges to be printed */
        qh_printend4geom(qh, NULL, facet, &num, printall);  // 调用函数统计要打印的边界数量
      FOREACHfacet_(facets)
        qh_printend4geom(qh, NULL, facet, &num, printall);  // 再次遍历所有面，并调用函数统计边界数量
      qh->ridgeoutnum= num;  // 将统计得到的边界数量赋值给 ridgeoutnum
      qh->printoutvar= 0;  /* counts number of ridges in output */
      qh_fprintf(qh, fp, 9038, "LIST # %s | %s\n", qh->rbox_command, qh->qhull_command);
      // 打印信息列表，包括 rbox_command 和 qhull_command
    }

    if (qh->PRINTdots) {
      // 如果 PRINTdots 标志位为真，则执行以下操作
      qh->printoutnum++;  // 增加打印数量计数器
      num= qh->num_points + qh_setsize(qh, qh->other_points);  // 计算点的总数
      if (qh->DELAUNAY && qh->ATinfinity)
        num--;  // 如果是 DELAUNAY 且 ATinfinity 标志位存在，则减少一个点的数量
      if (qh->PRINTdim == 4)
        qh_fprintf(qh, fp, 9039, "4VECT %d %d 1\n", num, num);
      else
        qh_fprintf(qh, fp, 9040, "VECT %d %d 1\n", num, num);
      // 根据 PRINTdim 打印不同格式的向量信息

      for (i=num; i--; ) {
        if (i % 20 == 0)
          qh_fprintf(qh, fp, 9041, "\n");  // 每20个数据换行
        qh_fprintf(qh, fp, 9042, "1 ");  // 打印数据为 "1 "
      }
      qh_fprintf(qh, fp, 9043, "# 1 point per line\n1 ");  // 打印说明和最后的 "1 "

      for (i=num-1; i--; ) { /* num at least 3 for D2 */
        if (i % 20 == 0)
          qh_fprintf(qh, fp, 9044, "\n");  // 每20个数据换行
        qh_fprintf(qh, fp, 9045, "0 ");  // 打印数据为 "0 "
      }
      qh_fprintf(qh, fp, 9046, "# 1 color for all\n");  // 打印说明

      FORALLpoints {
        // 遍历所有点
        if (!qh->DELAUNAY || !qh->ATinfinity || qh_pointid(qh, point) != qh->num_points-1) {
          // 如果不是 DELAUNAY 或者不在无穷远处，或者点的标识符不是最后一个点的标识符
          if (qh->PRINTdim == 4)
            qh_printpoint(qh, fp, NULL, point);  // 根据 PRINTdim 调用不同的函数打印点
          else
            qh_printpoint3(qh, fp, point);
        }
      }

      FOREACHpoint_(qh->other_points) {
        // 遍历其他点集合
        if (qh->PRINTdim == 4)
          qh_printpoint(qh, fp, NULL, point);  // 根据 PRINTdim 调用不同的函数打印点
        else
          qh_printpoint3(qh, fp, point);
      }
      qh_fprintf(qh, fp, 9047, "0 1 1 1  # color of points\n");  // 打印点的颜色信息
    }

    if (qh->PRINTdim == 4  && !qh->PRINTnoplanes)
      /* 4dview loads up multiple 4OFF objects slowly */
      qh_fprintf(qh, fp, 9048, "4OFF %d %d 1\n", 3*qh->ridgeoutnum, qh->ridgeoutnum);
      // 如果 PRINTdim 是 4 并且不是 PRINTnoplanes，则打印 4OFF 对象信息

    qh->PRINTcradius= 2 * qh->DISTround;  /* include test DISTround */
    // 设置 PRINTcradius 为 2 倍的 DISTround

    if (qh->PREmerge) {
      // 如果 PREmerge 标志位为真，则执行以下操作
      maximize_(qh->PRINTcradius, qh->premerge_centrum + qh->DISTround);
      // 将 PRINTcradius 设置为 PRINTcradius 和 premerge_centrum + DISTround 的较大值
    }else if (qh->POSTmerge) {
      // 如果 POSTmerge 标志位为真，则执行以下操作
      maximize_(qh->PRINTcradius, qh->postmerge_centrum + qh->DISTround);
      // 将 PRINTcradius 设置为 PRINTcradius 和 postmerge_centrum + DISTround 的较大值
    }

    qh->PRINTradius= qh->PRINTcradius;  // 将 PRINTcradius 赋值给 PRINTradius

    if (qh->PRINTspheres + qh->PRINTcoplanar)
      maximize_(qh->PRINTradius, qh->MAXabs_coord * qh_MINradius);
    // 如果 PRINTspheres 或 PRINTcoplanar 标志位为真，则将 PRINTradius 设置为 PRINTradius 和 MAXabs_coord * MINradius 的较大值

    if (qh->premerge_cos < REALmax/2) {
      // 如果 premerge_cos 小于 REALmax/2，则执行以下操作
      maximize_(qh->PRINTradius, (1- qh->premerge_cos) * qh->MAXabs_coord);
      // 将 PRINTradius 设置为 PRINTradius 和 (1- premerge_cos) * MAXabs_coord 的较大值
    }else if (!qh->PREmerge && qh->POSTmerge && qh->postmerge_cos < REALmax/2) {
      // 如果不是 PREmerge 且是 POSTmerge 且 postmerge_cos 小于 REALmax/2，则执行以下操作
      maximize_(qh->PRINTradius, (1- qh->postmerge_cos) * qh->MAXabs_coord);
      // 将 PRINTradius 设置为 PRINTradius 和 (1- postmerge_cos) * MAXabs_coord 的较大值
    }

    maximize_(qh->PRINTradius, qh->MINvisible);  // 将 PRINTradius 设置为 PRINTradius 和 MINvisible 的较大值

    if (qh->JOGGLEmax < REALmax/2)
      qh->PRINTradius += qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
      // 如果 JOGGLEmax 小于 REALmax/2，则将 PRINTradius 增加 JOGGLEmax 与 hull_dim 的平方根的乘积
    // 检查是否需要打印顶点数据，条件是PRINTdim不等于4且至少有一种打印选项被启用
    if (qh->PRINTdim != 4 &&
        (qh->PRINTcoplanar || qh->PRINTspheres || qh->PRINTcentrums)) {
      // 获取所有面的顶点列表
      vertices= qh_facetvertices(qh, facetlist, facets, printall);
      // 如果启用了PRINTspheres选项且PRINTdim小于等于3，则打印球面的信息
      if (qh->PRINTspheres && qh->PRINTdim <= 3)
        qh_printspheres(qh, fp, vertices, qh->PRINTradius);
      // 如果启用了PRINTcoplanar或PRINTcentrums选项
      if (qh->PRINTcoplanar || qh->PRINTcentrums) {
        // 设置标志表示首次打印中心点
        qh->firstcentrum= True;
        // 如果启用了PRINTcoplanar选项且未启用PRINTspheres选项，则打印每个顶点到内部点的向量
        if (qh->PRINTcoplanar && !qh->PRINTspheres) {
          // 遍历顶点列表，打印每个顶点到内部点的向量
          FOREACHvertex_(vertices)
            qh_printpointvect2(qh, fp, vertex->point, NULL, qh->interior_point, qh->PRINTradius);
        }
        // 遍历所有面
        FORALLfacet_(facetlist) {
          // 如果非打印所有面且需要跳过该面，则继续下一个面
          if (!printall && qh_skipfacet(qh, facet))
            continue;
          // 如果面的法向量不存在，则继续下一个面
          if (!facet->normal)
            continue;
          // 如果启用了PRINTcentrums选项且PRINTdim小于等于3，则打印面的中心点
          if (qh->PRINTcentrums && qh->PRINTdim <= 3)
            qh_printcentrum(qh, fp, facet, qh->PRINTcradius);
          // 如果未启用PRINTcoplanar选项，则继续下一个面
          if (!qh->PRINTcoplanar)
            continue;
          // 遍历面的共面点集合，打印每个点到面的法向量的向量
          FOREACHpoint_(facet->coplanarset)
            qh_printpointvect2(qh, fp, point, facet->normal, NULL, qh->PRINTradius);
          // 遍历面的外部点集合，打印每个点到面的法向量的向量
          FOREACHpoint_(facet->outsideset)
            qh_printpointvect2(qh, fp, point, facet->normal, NULL, qh->PRINTradius);
        }
        // 遍历所有的面
        FOREACHfacet_(facets) {
          // 如果非打印所有面且需要跳过该面，则继续下一个面
          if (!printall && qh_skipfacet(qh, facet))
            continue;
          // 如果面的法向量不存在，则继续下一个面
          if (!facet->normal)
            continue;
          // 如果启用了PRINTcentrums选项且PRINTdim小于等于3，则打印面的中心点
          if (qh->PRINTcentrums && qh->PRINTdim <= 3)
            qh_printcentrum(qh, fp, facet, qh->PRINTcradius);
          // 如果未启用PRINTcoplanar选项，则继续下一个面
          if (!qh->PRINTcoplanar)
            continue;
          // 遍历面的共面点集合，打印每个点到面的法向量的向量
          FOREACHpoint_(facet->coplanarset)
            qh_printpointvect2(qh, fp, point, facet->normal, NULL, qh->PRINTradius);
          // 遍历面的外部点集合，打印每个点到面的法向量的向量
          FOREACHpoint_(facet->outsideset)
            qh_printpointvect2(qh, fp, point, facet->normal, NULL, qh->PRINTradius);
        }
      }
      // 释放顶点列表的临时内存
      qh_settempfree(qh, &vertices);
    }
    // 增加访问标识，用于打印超平面交点信息
    qh->visit_id++; /* for printing hyperplane intersections */
    // 结束switch语句
    break;
  case qh_PRINTids:
    // 打印面的数量
    qh_fprintf(qh, fp, 9049, "%d\n", numfacets);
    // 结束switch语句
    break;
  case qh_PRINTincidences:
    // 如果是VORONOI模式且PRINTprecision选项被启用，则打印警告信息
    if (qh->VORONOI && qh->PRINTprecision)
      qh_fprintf(qh, qh->ferr, 7053, "qhull warning: input sites of Delaunay regions (option 'i').  Use option 'p' or 'o' for Voronoi centers.  Disable warning with option 'Pp'\n");
    // 将顶点id作为打印输出的变量
    qh->printoutvar= (int)qh->vertex_id;  /* centrum id for 4-d+, non-simplicial facets */
    // 如果hull_dim小于等于3，则打印面的数量
    if (qh->hull_dim <= 3)
      qh_fprintf(qh, fp, 9050, "%d\n", numfacets);
    else
      qh_fprintf(qh, fp, 9051, "%d\n", numsimplicial+numridges);
    // 结束switch语句
    break;
  case qh_PRINTinner:
  case qh_PRINTnormals:
  case qh_PRINTouter:
    // 如果是CDDoutput模式，则打印相应格式的输出
    if (qh->CDDoutput)
      qh_fprintf(qh, fp, 9052, "%s | %s\nbegin\n    %d %d real\n", qh->rbox_command,
            qh->qhull_command, numfacets, qh->hull_dim+1);
    else
      // 否则，打印hull_dim+1和面的数量
      qh_fprintf(qh, fp, 9053, "%d\n%d\n", qh->hull_dim+1, numfacets);
    // 结束switch语句
    break;
  case qh_PRINTmathematica:
  case qh_PRINTmaple:
    // 如果hull_dim大于3，则跳转到LABELnoformat标签
    if (qh->hull_dim > 3)  /* qh_initbuffers also checks */
      goto LABELnoformat;
    # 如果设置了 VORONOI 标志位，则输出警告信息指示输出为 Delaunay 三角化
    if (qh->VORONOI)
      qh_fprintf(qh, qh->ferr, 7054, "qhull warning: output is the Delaunay triangulation\n");

    # 根据输出格式选择性地输出不同格式的绘图命令
    if (format == qh_PRINTmaple) {
      if (qh->hull_dim == 2)
        qh_fprintf(qh, fp, 9054, "PLOT(CURVES(\n");
      else
        qh_fprintf(qh, fp, 9055, "PLOT3D(POLYGONS(\n");
    }else
      qh_fprintf(qh, fp, 9056, "{\n");

    # 重置打印输出变量，用于统计非首要的面（facets）数量
    qh->printoutvar= 0;

    # 结束当前 case 分支
    break;

  case qh_PRINTmerges:
    # 输出合并后的面（facets）数量
    qh_fprintf(qh, fp, 9057, "%d\n", numfacets);
    break;

  case qh_PRINTpointintersect:
    # 输出与点相交的面（facets）数量和点的维度
    qh_fprintf(qh, fp, 9058, "%d\n%d\n", qh->hull_dim, numfacets);
    break;

  case qh_PRINTneighbors:
    # 输出相邻面（facets）的数量
    qh_fprintf(qh, fp, 9059, "%d\n", numfacets);
    break;

  case qh_PRINToff:
  case qh_PRINTtriangles:
    # 如果设置了 VORONOI 标志位，则跳转到 LABELnoformat 分支
    if (qh->VORONOI)
      goto LABELnoformat;

    # 计算维度数
    num= qh->hull_dim;

    # 根据格式选择性地输出维度和其他信息
    if (format == qh_PRINToff || qh->hull_dim == 2)
      qh_fprintf(qh, fp, 9060, "%d\n%d %d %d\n", num,
        qh->num_points+qh_setsize(qh, qh->other_points), numfacets, totneighbors/2);
    else { /* qh_PRINTtriangles */
      # 设置打印输出变量为点数加上其他点的集合大小，用于第一个中心
      qh->printoutvar= qh->num_points+qh_setsize(qh, qh->other_points);

      # 如果是 DELAUNAY，则减少一个维度
      if (qh->DELAUNAY)
        num--;  /* drop last dimension */

      # 输出维度和其他信息
      qh_fprintf(qh, fp, 9061, "%d\n%d %d %d\n", num, qh->printoutvar
        + numfacets - numsimplicial, numsimplicial + numridges, totneighbors/2);
    }

    # 输出所有点的标识符
    FORALLpoints
      qh_printpointid(qh, qh->fout, NULL, num, point, qh_IDunknown);

    # 输出其他点集合中的所有点的标识符
    FOREACHpoint_(qh->other_points)
      qh_printpointid(qh, qh->fout, NULL, num, point, qh_IDunknown);

    # 如果格式为 PRINTtriangles 并且维度大于 2，则输出非单纯形（non-simplicial）面的中心点
    if (format == qh_PRINTtriangles && qh->hull_dim > 2) {
      FORALLfacets {
        if (!facet->simplicial && facet->visitid)
          qh_printcenter(qh, qh->fout, format, NULL, facet);
      }
    }
    break;

  case qh_PRINTpointnearest:
    # 输出最近点的共面点（coplanars）数量
    qh_fprintf(qh, fp, 9062, "%d\n", numcoplanars);
    break;

  case qh_PRINTpoints:
    # 如果未设置 VORONOI 标志位，则跳转到 LABELnoformat 分支
    if (!qh->VORONOI)
      goto LABELnoformat;

    # 如果是 CDD 输出，则输出相应的命令
    if (qh->CDDoutput)
      qh_fprintf(qh, fp, 9063, "%s | %s\nbegin\n%d %d real\n", qh->rbox_command,
           qh->qhull_command, numfacets, qh->hull_dim);
    else
      qh_fprintf(qh, fp, 9064, "%d\n%d\n", qh->hull_dim-1, numfacets);
    break;

  case qh_PRINTvertices:
    # 输出顶点（vertices）的数量
    qh_fprintf(qh, fp, 9065, "%d\n", numfacets);
    break;

  case qh_PRINTsummary:
  default:
  LABELnoformat:
    # 输出错误信息，指示无法使用当前维度的格式
    qh_fprintf(qh, qh->ferr, 6068, "qhull internal error (qh_printbegin): can not use this format for dimension %d\n",
         qh->hull_dim);
    # 引发 Qhull 错误并退出程序
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
} /* printbegin */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printcenter">-</a>

  qh_printcenter(qh, fp, string, facet )
    print facet->center as centrum or Voronoi center
    string may be NULL.  Don't include '%' codes.
    nop if qh->CENTERtype neither CENTERvoronoi nor CENTERcentrum
    if upper envelope of Delaunay triangulation and point at-infinity
      prints qh_INFINITE instead;

  notes:
    defines facet->center if needed
    if format=PRINTgeom, adds a 0 if would otherwise be 2-d
    Same as QhullFacet::printCenter
*/
void qh_printcenter(qhT *qh, FILE *fp, qh_PRINT format, const char *string, facetT *facet) {
  int k, num;

  // 如果不是以 Voronoi 或 centrum 为中心类型，则直接返回
  if (qh->CENTERtype != qh_ASvoronoi && qh->CENTERtype != qh_AScentrum)
    return;

  // 如果存在字符串，则将其写入文件流 fp
  if (string)
    qh_fprintf(qh, fp, 9066, string);

  // 如果是以 Voronoi 为中心类型
  if (qh->CENTERtype == qh_ASvoronoi) {
    num = qh->hull_dim - 1;

    // 如果没有 facet->normal 或者不是上包络的 Delaunay 三角化，并且不存在 qh->ATinfinity
    if (!facet->normal || !facet->upperdelaunay || !qh->ATinfinity) {
      // 如果 facet->center 不存在，则计算并定义 facet->center
      if (!facet->center)
        facet->center = qh_facetcenter(qh, facet->vertices);

      // 输出 facet 的中心坐标
      for (k = 0; k < num; k++)
        qh_fprintf(qh, fp, 9067, qh_REAL_1, facet->center[k]);
    } else {
      // 否则，输出 qh_INFINITE 作为坐标
      for (k = 0; k < num; k++)
        qh_fprintf(qh, fp, 9068, qh_REAL_1, qh_INFINITE);
    }
  } else /* qh.CENTERtype == qh_AScentrum */ {
    num = qh->hull_dim;

    // 如果是 PRINTtriangles 格式且是 Delaunay 三角化，则减少一个维度
    if (format == qh_PRINTtriangles && qh->DELAUNAY)
      num--;

    // 如果 facet->center 不存在，则计算并定义 facet->center
    if (!facet->center)
      facet->center = qh_getcentrum(qh, facet);

    // 输出 facet 的中心坐标
    for (k = 0; k < num; k++)
      qh_fprintf(qh, fp, 9069, qh_REAL_1, facet->center[k]);
  }

  // 如果是 PRINTgeom 格式且 num 为 2，则输出 " 0\n"；否则输出 "\n"
  if (format == qh_PRINTgeom && num == 2)
    qh_fprintf(qh, fp, 9070, " 0\n");
  else
    qh_fprintf(qh, fp, 9071, "\n");
} /* printcenter */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printcentrum">-</a>

  qh_printcentrum(qh, fp, facet, radius )
    print centrum for a facet in OOGL format
    radius defines size of centrum
    2-d or 3-d only

  returns:
    defines facet->center if needed
*/
void qh_printcentrum(qhT *qh, FILE *fp, facetT *facet, realT radius) {
  pointT *centrum, *projpt;
  boolT tempcentrum= False;
  realT xaxis[4], yaxis[4], normal[4], dist;
  realT green[3]={0, 1, 0};
  vertexT *apex;
  int k;

  // 如果是以 centrum 为中心类型
  if (qh->CENTERtype == qh_AScentrum) {
    // 如果 facet->center 不存在，则计算并定义 facet->center
    if (!facet->center)
      facet->center = qh_getcentrum(qh, facet);
    centrum = facet->center;
  } else {
    // 否则，计算并定义 centrum
    centrum = qh_getcentrum(qh, facet);
    tempcentrum = True;
  }

  // 输出 centrum 的几何描述信息到文件流 fp 中
  qh_fprintf(qh, fp, 9072, "{appearance {-normal -edge normscale 0} ");

  // 如果是第一个 centrum，输出特定格式的信息
  if (qh->firstcentrum) {
    qh->firstcentrum = False;
    qh_fprintf(qh, fp, 9073, "{INST geom { define centrum CQUAD  # f%d\n\
-0.3 -0.3 0.0001     0 0 1 1\n\
 0.3 -0.3 0.0001     0 0 1 1\n\
 0.3  0.3 0.0001     0 0 1 1\n\
-0.3  0.3 0.0001     0 0 1 1 } transform { \n", facet->id);
  } else


注释：这段代码主要是关于处理几何中心（centrum）和 Voronoi 中心的打印功能，依据不同的条件输出相应的坐标信息到文件流中。
    qh_fprintf(qh, fp, 9074, "{INST geom { : centrum } transform { # f%d\n", facet->id);
    # 使用 qh_fprintf 函数向文件流 fp 中写入格式化字符串，包含 facet->id 的值作为标识符
    
    apex= SETfirstt_(facet->vertices, vertexT);
    # 将 facet->vertices 中的第一个顶点赋值给 apex
    
    qh_distplane(qh, apex->point, facet, &dist);
    # 计算顶点 apex->point 到 facet 所定义的平面的距离，并将结果保存在 dist 变量中
    
    projpt= qh_projectpoint(qh, apex->point, facet, dist);
    # 使用 qh_projectpoint 函数投影顶点 apex->point 到 facet 所在平面上，得到投影点的坐标，存放在 projpt 中
    
    for (k=qh->hull_dim; k--; ) {
        xaxis[k]= projpt[k] - centrum[k];
        normal[k]= facet->normal[k];
    }
    # 遍历数组 xaxis 和 normal，根据计算得到的 projpt 和 facet->normal 以及 centrum 的值计算 x 轴和法线向量
    
    if (qh->hull_dim == 2) {
        xaxis[2]= 0;
        normal[2]= 0;
    }else if (qh->hull_dim == 4) {
        qh_projectdim3(qh, xaxis, xaxis);
        qh_projectdim3(qh, normal, normal);
        qh_normalize2(qh, normal, qh->PRINTdim, True, NULL, NULL);
    }
    # 根据 qh->hull_dim 的值进行条件判断和处理，调整 xaxis 和 normal 的第三个分量
    
    qh_crossproduct(3, xaxis, normal, yaxis);
    # 计算 xaxis 和 normal 的叉乘结果，并将结果保存在 yaxis 中
    
    qh_fprintf(qh, fp, 9075, "%8.4g %8.4g %8.4g 0\n", xaxis[0], xaxis[1], xaxis[2]);
    # 将 xaxis 的值按指定格式写入文件流 fp
    
    qh_fprintf(qh, fp, 9076, "%8.4g %8.4g %8.4g 0\n", yaxis[0], yaxis[1], yaxis[2]);
    # 将 yaxis 的值按指定格式写入文件流 fp
    
    qh_fprintf(qh, fp, 9077, "%8.4g %8.4g %8.4g 0\n", normal[0], normal[1], normal[2]);
    # 将 normal 的值按指定格式写入文件流 fp
    
    qh_printpoint3(qh, fp, centrum);
    # 使用 qh_printpoint3 函数将 centrum 的坐标按照特定格式输出到文件流 fp
    
    qh_fprintf(qh, fp, 9078, "1 }}}\n");
    # 向文件流 fp 写入固定格式的字符串 "1 }}}\n"
    
    qh_memfree(qh, projpt, qh->normal_size);
    # 释放 projpt 占用的内存空间，大小由 qh->normal_size 决定
    
    qh_printpointvect(qh, fp, centrum, facet->normal, NULL, radius, green);
    # 使用 qh_printpointvect 函数将 centrum 和 facet->normal 以及其他参数按指定格式输出到文件流 fp
    
    if (tempcentrum)
        qh_memfree(qh, centrum, qh->normal_size);
    # 如果 tempcentrum 不为 NULL，则释放 centrum 占用的内存空间，大小由 qh->normal_size 决定
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printend">-</a>

  qh_printend(qh, fp, format )
    prints trailer for all output formats

  see:
    qh_printbegin() and qh_printafacet()

*/
void qh_printend(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall) {
  int num;
  facetT *facet, **facetp;

  // 如果没有打印任何面片，输出警告信息
  if (!qh->printoutnum)
    qh_fprintf(qh, qh->ferr, 7055, "qhull warning: no facets printed\n");
  
  switch (format) {
  case qh_PRINTgeom:
    // 如果是四维凸壳且不忽略某一维度且不打印平面信息
    if (qh->hull_dim == 4 && qh->DROPdim < 0  && !qh->PRINTnoplanes) {
      qh->visit_id++;
      num= 0;
      // 遍历所有面片列表，打印四维几何信息
      FORALLfacet_(facetlist)
        qh_printend4geom(qh, fp, facet, &num, printall);
      // 遍历额外面片集合，打印四维几何信息
      FOREACHfacet_(facets)
        qh_printend4geom(qh, fp, facet, &num, printall);
      // 检查打印的边数是否与预期相符
      if (num != qh->ridgeoutnum || qh->printoutvar != qh->ridgeoutnum) {
        qh_fprintf(qh, qh->ferr, 6069, "qhull internal error (qh_printend): number of ridges %d != number printed %d and at end %d\n", qh->ridgeoutnum, qh->printoutvar, num);
        qh_errexit(qh, qh_ERRqhull, NULL, NULL);
      }
    }else
      // 否则，输出结束符号
      qh_fprintf(qh, fp, 9079, "}\n");
    break;
  case qh_PRINTinner:
  case qh_PRINTnormals:
  case qh_PRINTouter:
    // 如果是CDD输出格式，输出结束符号
    if (qh->CDDoutput)
      qh_fprintf(qh, fp, 9080, "end\n");
    break;
  case qh_PRINTmaple:
    // 如果是Maple输出格式，输出结束符号
    qh_fprintf(qh, fp, 9081, "));\n");
    break;
  case qh_PRINTmathematica:
    // 如果是Mathematica输出格式，输出结束符号
    qh_fprintf(qh, fp, 9082, "}\n");
    break;
  case qh_PRINTpoints:
    // 如果是CDD输出格式，输出结束符号
    if (qh->CDDoutput)
      qh_fprintf(qh, fp, 9083, "end\n");
    break;
  default:
    break;
  }
} /* printend */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printend4geom">-</a>

  qh_printend4geom(qh, fp, facet, numridges, printall )
    helper function for qh_printbegin/printend

  returns:
    number of printed ridges

  notes:
    just counts printed ridges if fp=NULL
    uses facet->visitid
    must agree with qh_printfacet4geom...

  design:
    computes color for facet from its normal
    prints each ridge of facet
*/
void qh_printend4geom(qhT *qh, FILE *fp, facetT *facet, int *nump, boolT printall) {
  realT color[3];
  int i, num= *nump;
  facetT *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;

  // 如果不打印所有面片且该面片可跳过，则返回
  if (!printall && qh_skipfacet(qh, facet))
    return;
  
  // 如果不打印平面信息或者面片可见且是新面片，则返回
  if (qh->PRINTnoplanes || (facet->visible && qh->NEWfacets))
    return;
  
  // 如果面片没有法向量，返回
  if (!facet->normal)
    return;
  
  // 如果输出流不为空，计算面片颜色并规范化
  if (fp) {
    for (i=0; i < 3; i++) {
      color[i]= (facet->normal[i]+1.0)/2.0;
      maximize_(color[i], -1.0);
      minimize_(color[i], +1.0);
    }
  }
  
  // 设置当前面片的访问标记
  facet->visitid= qh->visit_id;
  
  // 如果面片是简单形式
  if (facet->simplicial) {
    // 遍历面片的每个相邻面片
    FOREACHneighbor_(facet) {
      // 如果相邻面片的访问标记与当前标记不同
      if (neighbor->visitid != qh->visit_id) {
        // 如果输出流不为空，输出三角形面片信息
        if (fp)
          qh_fprintf(qh, fp, 9084, "3 %d %d %d %8.4g %8.4g %8.4g 1 # f%d f%d\n",
                 3*num, 3*num+1, 3*num+2, color[0], color[1], color[2],
                 facet->id, neighbor->id);
        num++;
      }
    }
  }
}
    }
  }else {
    // 遍历当前 facet 的所有 ridges
    FOREACHridge_(facet->ridges) {
      // 获取当前 ridge 相邻的 facet
      neighbor= otherfacet_(ridge, facet);
      // 如果邻居 facet 的 visitid 不等于当前的 visit_id，则执行以下操作
      if (neighbor->visitid != qh->visit_id) {
        // 如果 fp 非空，则向 fp 写入格式化字符串
        if (fp)
          qh_fprintf(qh, fp, 9085, "3 %d %d %d %8.4g %8.4g %8.4g 1 #r%d f%d f%d\n",
                 3*num, 3*num+1, 3*num+2, color[0], color[1], color[2],
                 ridge->id, facet->id, neighbor->id);
        // num 自增
        num++;
      }
    }
  }
  // 将 num 赋值给 nump 指针指向的变量
  *nump= num;
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printextremes">-</a>

  qh_printextremes(qh, fp, facetlist, facets, printall )
    print extreme points for convex hulls or halfspace intersections

  notes:
    #points, followed by ids, one per line

    sorted by id
    same order as qh_printpoints_out if no coplanar/interior points
*/
void qh_printextremes(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall) {
  setT *vertices, *points;
  pointT *point;
  vertexT *vertex, **vertexp;
  int id;
  int numpoints=0, point_i, point_n;
  int allpoints= qh->num_points + qh_setsize(qh, qh->other_points);

  points= qh_settemp(qh, allpoints);  /* 临时集合用于存放极点的点集 */
  qh_setzero(qh, points, 0, allpoints);  /* 初始化点集，全部置为NULL */
  vertices= qh_facetvertices(qh, facetlist, facets, printall);  /* 获取构成所有面的顶点集 */
  FOREACHvertex_(vertices) {  /* 遍历每个顶点 */
    id= qh_pointid(qh, vertex->point);  /* 获取顶点的唯一ID */
    if (id >= 0) {
      SETelem_(points, id)= vertex->point;  /* 将顶点按ID放入临时集合 */
      numpoints++;  /* 统计有效顶点数 */
    }
  }
  qh_settempfree(qh, &vertices);  /* 释放顶点集合的临时空间 */
  qh_fprintf(qh, fp, 9086, "%d\n", numpoints);  /* 在文件流上打印顶点数 */
  FOREACHpoint_i_(qh, points) {  /* 遍历临时点集中的点 */
    if (point)
      qh_fprintf(qh, fp, 9087, "%d\n", point_i);  /* 在文件流上打印点的ID */
  }
  qh_settempfree(qh, &points);  /* 释放临时点集合的空间 */
} /* printextremes */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printextremes_2d">-</a>

  qh_printextremes_2d(qh, fp, facetlist, facets, printall )
    prints point ids for facets in qh_ORIENTclock order

  notes:
    #points, followed by ids, one per line
    if facetlist/facets are disjoint than the output includes skips
    errors if facets form a loop
    does not print coplanar points
*/
void qh_printextremes_2d(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall) {
  int numfacets, numridges, totneighbors, numcoplanars, numsimplicial, numtricoplanars;
  setT *vertices;
  facetT *facet, *startfacet, *nextfacet;
  vertexT *vertexA, *vertexB;

  qh_countfacets(qh, facetlist, facets, printall, &numfacets, &numsimplicial,
      &totneighbors, &numridges, &numcoplanars, &numtricoplanars); /* marks qh->visit_id */
  vertices= qh_facetvertices(qh, facetlist, facets, printall);  /* 获取构成所有面的顶点集 */
  qh_fprintf(qh, fp, 9088, "%d\n", qh_setsize(qh, vertices));  /* 在文件流上打印顶点数 */
  qh_settempfree(qh, &vertices);  /* 释放顶点集合的临时空间 */
  if (!numfacets)
    return;  /* 如果没有面，直接返回 */
  facet= startfacet= facetlist ? facetlist : SETfirstt_(facets, facetT);  /* 获取第一个面 */
  qh->vertex_visit++;  /* 顶点访问标记加一 */
  qh->visit_id++;  /* 访问ID加一 */
  do {
    if (facet->toporient ^ qh_ORIENTclock) {  /* 检查面的方向是否为顺时针方向 */
      vertexA= SETfirstt_(facet->vertices, vertexT);  /* 获取面的第一个顶点 */
      vertexB= SETsecondt_(facet->vertices, vertexT);  /* 获取面的第二个顶点 */
      nextfacet= SETfirstt_(facet->neighbors, facetT);  /* 获取面的相邻面 */
    }else {
      vertexA= SETsecondt_(facet->vertices, vertexT);  /* 获取面的第二个顶点 */
      vertexB= SETfirstt_(facet->vertices, vertexT);  /* 获取面的第一个顶点 */
      nextfacet= SETsecondt_(facet->neighbors, facetT);  /* 获取面的另一个相邻面 */
    }
    # 检查当前 facet 是否已经被访问过，避免循环引用
    if (facet->visitid == qh->visit_id) {
      # 如果已经访问过，则输出错误信息并终止程序
      qh_fprintf(qh, qh->ferr, 6218, "qhull internal error (qh_printextremes_2d): loop in facet list.  facet %d nextfacet %d\n",
                 facet->id, nextfacet->id);
      qh_errexit2(qh, qh_ERRqhull, facet, nextfacet);
    }
    # 如果 facet 尚未被访问过
    if (facet->visitid) {
      # 如果 vertexA 的访问标记与当前顶点访问序号不同
      if (vertexA->visitid != qh->vertex_visit) {
        # 更新 vertexA 的访问标记为当前顶点访问序号，并输出顶点 ID 到文件流 fp
        vertexA->visitid= qh->vertex_visit;
        qh_fprintf(qh, fp, 9089, "%d\n", qh_pointid(qh, vertexA->point));
      }
      # 如果 vertexB 的访问标记与当前顶点访问序号不同
      if (vertexB->visitid != qh->vertex_visit) {
        # 更新 vertexB 的访问标记为当前顶点访问序号，并输出顶点 ID 到文件流 fp
        vertexB->visitid= qh->vertex_visit;
        qh_fprintf(qh, fp, 9090, "%d\n", qh_pointid(qh, vertexB->point));
      }
    }
    # 将当前 facet 的访问标记设置为当前 qh 的访问序号
    facet->visitid= qh->visit_id;
    # 将 facet 移动到下一个 facet
    facet= nextfacet;
  }while (facet && facet != startfacet);
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printextremes_d">-</a>

  qh_printextremes_d(qh, fp, facetlist, facets, printall )
    print extreme points of input sites for Delaunay triangulations

  notes:
    #points, followed by ids, one per line

    unordered
*/
void qh_printextremes_d(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall) {
  setT *vertices;                     /* 存储顶点集合的临时变量 */
  vertexT *vertex, **vertexp;         /* 存储顶点及其指针的变量 */
  boolT upperseen, lowerseen;         /* 检测顶点是否具有上下极值的标志 */
  facetT *neighbor, **neighborp;      /* 存储邻居面及其指针的变量 */
  int numpoints=0;                    /* 记录具有上下极值的顶点数量 */

  vertices= qh_facetvertices(qh, facetlist, facets, printall);   /* 获取包含顶点的集合 */
  qh_vertexneighbors(qh);            /* 计算顶点的邻居关系 */
  FOREACHvertex_(vertices) {         /* 遍历每个顶点 */
    upperseen= lowerseen= False;     /* 初始化上下极值标志为假 */
    FOREACHneighbor_(vertex) {       /* 遍历顶点的每个邻居 */
      if (neighbor->upperdelaunay)   /* 如果邻居面是上半平面的 Delaunay 三角化 */
        upperseen= True;             /* 设置上半平面极值标志为真 */
      else
        lowerseen= True;             /* 否则设置下半平面极值标志为真 */
    }
    if (upperseen && lowerseen) {     /* 如果顶点同时具有上下极值 */
      vertex->seen= True;            /* 设置顶点已看到的标志为真 */
      numpoints++;                   /* 计数器增加 */
    } else
      vertex->seen= False;           /* 否则设置顶点已看到的标志为假 */
  }
  qh_fprintf(qh, fp, 9091, "%d\n", numpoints);   /* 输出具有上下极值顶点的数量 */
  FOREACHvertex_(vertices) {         /* 再次遍历每个顶点 */
    if (vertex->seen)                /* 如果顶点已看到 */
      qh_fprintf(qh, fp, 9092, "%d\n", qh_pointid(qh, vertex->point));   /* 输出顶点的标识符 */
  }
  qh_settempfree(qh, &vertices);     /* 释放顶点集合的临时内存 */
} /* printextremes_d */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet">-</a>

  qh_printfacet(qh, fp, facet )
    prints all fields of a facet to fp

  notes:
    ridges printed in neighbor order
*/
void qh_printfacet(qhT *qh, FILE *fp, facetT *facet) {

  qh_printfacetheader(qh, fp, facet);   /* 打印面的标题信息 */
  if (facet->ridges)                    /* 如果面有脊 */
    qh_printfacetridges(qh, fp, facet); /* 打印面的脊信息 */
} /* printfacet */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet2geom">-</a>

  qh_printfacet2geom(qh, fp, facet, color )
    print facet as part of a 2-d VECT for Geomview

    notes:
      assume precise calculations in io_r.c with roundoff covered by qh_GEOMepsilon
      mindist is calculated within io_r.c.  maxoutside is calculated elsewhere
      so a DISTround error may have occurred.
*/
void qh_printfacet2geom(qhT *qh, FILE *fp, facetT *facet, realT color[3]) {
  pointT *point0, *point1;    /* 保存面的两个顶点 */
  realT mindist, innerplane, outerplane;   /* 保存最小距离及内外平面距离 */
  int k;

  qh_facet2point(qh, facet, &point0, &point1, &mindist);   /* 计算面的两个顶点及最小距离 */
  qh_geomplanes(qh, facet, &outerplane, &innerplane);      /* 计算面的外部和内部平面距离 */
  if (qh->PRINTouter || (!qh->PRINTnoplanes && !qh->PRINTinner))
    qh_printfacet2geom_points(qh, fp, point0, point1, facet, outerplane, color);  /* 打印外部平面信息 */
  if (qh->PRINTinner || (!qh->PRINTnoplanes && !qh->PRINTouter &&
                outerplane - innerplane > 2 * qh->MAXabs_coord * qh_GEOMepsilon)) {
    for (k=3; k--; )
      color[k]= 1.0 - color[k];
    qh_printfacet2geom_points(qh, fp, point0, point1, facet, innerplane, color);  /* 打印内部平面信息 */
  }
  qh_memfree(qh, point1, qh->normal_size);   /* 释放面的第二个顶点内存 */
  qh_memfree(qh, point0, qh->normal_size);   /* 释放面的第一个顶点内存 */
} /* printfacet2geom */
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet2geom_points">-</a>

  qh_printfacet2geom_points(qh, fp, point1, point2, facet, offset, color )
    prints a 2-d facet as a VECT with 2 points at some offset.
    The points are on the facet's plane.
*/
void qh_printfacet2geom_points(qhT *qh, FILE *fp, pointT *point1, pointT *point2,
                               facetT *facet, realT offset, realT color[3]) {
  pointT *p1= point1, *p2= point2;

  // 输出格式化的字符串作为 Geomview VECT 格式，表示一个平面上的二维面元
  qh_fprintf(qh, fp, 9093, "VECT 1 2 1 2 1 # f%d\n", facet->id);
  
  // 如果偏移量不为零，则对点进行偏移投影
  if (offset != 0.0) {
    p1= qh_projectpoint(qh, p1, facet, -offset);
    p2= qh_projectpoint(qh, p2, facet, -offset);
  }
  
  // 输出两个点的坐标，表示二维面元在平面上的位置
  qh_fprintf(qh, fp, 9094, "%8.4g %8.4g %8.4g\n%8.4g %8.4g %8.4g\n",
           p1[0], p1[1], 0.0, p2[0], p2[1], 0.0);
  
  // 如果偏移量不为零，则释放投影后的点
  if (offset != 0.0) {
    qh_memfree(qh, p1, qh->normal_size);
    qh_memfree(qh, p2, qh->normal_size);
  }
  
  // 输出颜色信息，用于标识面元的颜色
  qh_fprintf(qh, fp, 9095, "%8.4g %8.4g %8.4g 1.0\n", color[0], color[1], color[2]);
} /* printfacet2geom_points */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet2math">-</a>

  qh_printfacet2math(qh, fp, facet, format, notfirst )
    print 2-d Maple or Mathematica output for a facet
    may be non-simplicial

  notes:
    use %16.8f since Mathematica 2.2 does not handle exponential format
    see qh_printfacet3math
*/
void qh_printfacet2math(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format, int notfirst) {
  pointT *point0, *point1;
  realT mindist;
  const char *pointfmt;

  // 获取面元的两个端点和它们之间的最小距离
  qh_facet2point(qh, facet, &point0, &point1, &mindist);
  
  // 如果不是第一个输出的面元，则在输出前加上逗号
  if (notfirst)
    qh_fprintf(qh, fp, 9096, ",");
  
  // 根据输出格式选择合适的字符串格式，输出到文件中
  if (format == qh_PRINTmaple)
    pointfmt= "[[%16.8f, %16.8f], [%16.8f, %16.8f]]\n";
  else
    pointfmt= "Line[{{%16.8f, %16.8f}, {%16.8f, %16.8f}}]\n";
  qh_fprintf(qh, fp, 9097, pointfmt, point0[0], point0[1], point1[0], point1[1]);
  
  // 释放申请的内存空间，避免内存泄漏
  qh_memfree(qh, point1, qh->normal_size);
  qh_memfree(qh, point0, qh->normal_size);
} /* printfacet2math */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet3geom_nonsimplicial">-</a>

  qh_printfacet3geom_nonsimplicial(qh, fp, facet, color )
    print Geomview OFF for a 3-d nonsimplicial facet.
    if DOintersections, prints ridges to unvisited neighbors(qh->visit_id)

  notes
    uses facet->visitid for intersections and ridges
*/
/*
   qh_printfacet3geom_nonsimplicial(qh, fp, facet, color)
   打印非单纯形几何的面片到文件中，使用指定的颜色

   qhT *qh       : Qhull上下文
   FILE *fp      : 输出文件指针
   facetT *facet : 要打印的面片
   realT color[3]: 颜色数组，表示RGB颜色值
*/
void qh_printfacet3geom_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]) {
  ridgeT *ridge, **ridgep;          // 边缘和边缘指针
  setT *projectedpoints, *vertices; // 投影点集合和顶点集合
  vertexT *vertex, **vertexp, *vertexA, *vertexB; // 顶点和顶点指针，用于边的端点
  pointT *projpt, *point, **pointp; // 投影点、点和点指针
  facetT *neighbor;                 // 相邻面片
  realT dist, outerplane, innerplane; // 距离、外部平面和内部平面
  int cntvertices, k;               // 顶点数量和循环变量k
  realT black[3]={0, 0, 0}, green[3]={0, 1, 0}; // 黑色和绿色的颜色数组

  // 计算面片的外部和内部平面
  qh_geomplanes(qh, facet, &outerplane, &innerplane);
  
  // 获取面片的顶点集合，已经被定向
  vertices= qh_facet3vertex(qh, facet); /* oriented */
  
  // 计算顶点集合的大小
  cntvertices= qh_setsize(qh, vertices);
  
  // 临时存储投影点的集合
  projectedpoints= qh_settemp(qh, cntvertices);
  
  // 遍历顶点集合，计算投影点并添加到投影点集合中
  FOREACHvertex_(vertices) {
    zinc_(Zdistio); // 计数增加
    qh_distplane(qh, vertex->point, facet, &dist); // 计算顶点到面片的距离
    projpt= qh_projectpoint(qh, vertex->point, facet, dist); // 计算顶点在面片上的投影点
    qh_setappend(qh, &projectedpoints, projpt); // 将投影点添加到投影点集合中
  }
  
  // 如果需要打印外部平面或者不需要打印平面但是外部平面与内部平面差距大于一定阈值，则打印外部平面
  if (qh->PRINTouter || (!qh->PRINTnoplanes && !qh->PRINTinner))
    qh_printfacet3geom_points(qh, fp, projectedpoints, facet, outerplane, color);
  
  // 如果需要打印内部平面或者不需要打印平面但是外部平面与内部平面差距大于一定阈值，则打印内部平面
  if (qh->PRINTinner || (!qh->PRINTnoplanes && !qh->PRINTouter &&
                outerplane - innerplane > 2 * qh->MAXabs_coord * qh_GEOMepsilon)) {
    // 反转颜色数组来得到内部平面的颜色
    for (k=3; k--; )
      color[k]= 1.0 - color[k];
    // 打印内部平面
    qh_printfacet3geom_points(qh, fp, projectedpoints, facet, innerplane, color);
  }
  
  // 释放投影点集合中的点内存
  FOREACHpoint_(projectedpoints)
    qh_memfree(qh, point, qh->normal_size);
  
  // 释放临时投影点集合和顶点集合的内存
  qh_settempfree(qh, &projectedpoints);
  qh_settempfree(qh, &vertices);
  
  // 如果需要计算交点或者打印边缘，且面片不可见或者不是新的面片
  if ((qh->DOintersections || qh->PRINTridges)
  && (!facet->visible || !qh->NEWfacets)) {
    // 设置面片的访问ID
    facet->visitid= qh->visit_id;
    
    // 遍历面片的所有边缘
    FOREACHridge_(facet->ridges) {
      // 获取边缘的相邻面片
      neighbor= otherfacet_(ridge, facet);
      
      // 如果相邻面片的访问ID不等于当前访问ID
      if (neighbor->visitid != qh->visit_id) {
        // 如果需要计算交点，则打印超平面交点
        if (qh->DOintersections)
          qh_printhyperplaneintersection(qh, fp, facet, neighbor, ridge->vertices, black);
        
        // 如果需要打印边缘，则打印边缘的几何线段
        if (qh->PRINTridges) {
          vertexA= SETfirstt_(ridge->vertices, vertexT);
          vertexB= SETsecondt_(ridge->vertices, vertexT);
          qh_printline3geom(qh, fp, vertexA->point, vertexB->point, green);
        }
      }
    }
  }
} /* printfacet3geom_nonsimplicial */
    # 遍历点的坐标维度
    for (k=0; k < qh->hull_dim; k++) {
      # 如果当前维度等于需要丢弃的维度
      if (k == qh->DROPdim)
        # 打印 '0 ' 到文件流 fp
        qh_fprintf(qh, fp, 9099, "0 ");
      else
        # 打印点的当前维度值到文件流 fp，格式为 "%8.4g "
        qh_fprintf(qh, fp, 9100, "%8.4g ", point[k]);
    }
    # 如果要打印的点数不等于总点数
    if (printpoints != points)
      # 释放内存中的点数据
      qh_memfree(qh, point, qh->normal_size);
    # 打印换行符到文件流 fp
    qh_fprintf(qh, fp, 9101, "\n");
  }
  # 如果要打印的点数不等于总点数
  if (printpoints != points)
    # 释放临时点集合 printpoints
    qh_settempfree(qh, &printpoints);
  # 打印点的数量 n 到文件流 fp
  qh_fprintf(qh, fp, 9102, "%d ", n);
  # 循环打印点的索引号到文件流 fp
  for (i=0; i < n; i++)
    qh_fprintf(qh, fp, 9103, "%d ", i);
  # 打印颜色值和透明度信息到文件流 fp，格式为 "%8.4g %8.4g %8.4g 1.0 }\n"
  qh_fprintf(qh, fp, 9104, "%8.4g %8.4g %8.4g 1.0 }\n", color[0], color[1], color[2]);
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet3geom_simplicial">-</a>

  qh_printfacet3geom_simplicial(qh )
    print Geomview OFF for a 3-d simplicial facet.

  notes:
    may flip color
    uses facet->visitid for intersections and ridges

    assume precise calculations in io_r.c with roundoff covered by qh_GEOMepsilon
    innerplane may be off by qh->DISTround.  Maxoutside is calculated elsewhere
    so a DISTround error may have occurred.
*/
void qh_printfacet3geom_simplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]) {
  setT *points, *vertices;
  vertexT *vertex, **vertexp, *vertexA, *vertexB;
  facetT *neighbor, **neighborp;
  realT outerplane, innerplane;
  realT black[3]={0, 0, 0}, green[3]={0, 1, 0};
  int k;

  // 计算 facet 的外部和内部平面方程系数
  qh_geomplanes(qh, facet, &outerplane, &innerplane);
  
  // 获取 facet 的顶点集合
  vertices= qh_facet3vertex(qh, facet);
  
  // 临时设置点集合，用于输出
  points= qh_settemp(qh, qh->TEMPsize);
  
  // 将顶点集合中的点添加到 points 中
  FOREACHvertex_(vertices)
    qh_setappend(qh, &points, vertex->point);
  
  // 如果应该打印外部平面或者不禁止平面打印并且不打印内部平面，则打印外部平面
  if (qh->PRINTouter || (!qh->PRINTnoplanes && !qh->PRINTinner))
    qh_printfacet3geom_points(qh, fp, points, facet, outerplane, color);
  
  // 如果应该打印内部平面或者不禁止平面打印并且不打印外部平面，则打印内部平面
  if (qh->PRINTinner || (!qh->PRINTnoplanes && !qh->PRINTouter &&
              outerplane - innerplane > 2 * qh->MAXabs_coord * qh_GEOMepsilon)) {
    // 翻转颜色
    for (k=3; k--; )
      color[k]= 1.0 - color[k];
    qh_printfacet3geom_points(qh, fp, points, facet, innerplane, color);
  }
  
  // 释放临时点集合和顶点集合的内存
  qh_settempfree(qh, &points);
  qh_settempfree(qh, &vertices);
  
  // 如果需要计算相交或者打印棱，则执行以下操作
  if ((qh->DOintersections || qh->PRINTridges)
  && (!facet->visible || !qh->NEWfacets)) {
    facet->visitid= qh->visit_id;
    // 遍历 facet 的每一个相邻面
    FOREACHneighbor_(facet) {
      // 如果相邻面未被访问过
      if (neighbor->visitid != qh->visit_id) {
        // 根据相邻面和当前面的顶点集合，计算相交的超平面
        vertices= qh_setnew_delnthsorted(qh, facet->vertices, qh->hull_dim,
                          SETindex_(facet->neighbors, neighbor), 0);
        if (qh->DOintersections)
           qh_printhyperplaneintersection(qh, fp, facet, neighbor, vertices, black);
        // 如果需要打印棱，则打印棱
        if (qh->PRINTridges) {
          vertexA= SETfirstt_(vertices, vertexT);
          vertexB= SETsecondt_(vertices, vertexT);
          qh_printline3geom(qh, fp, vertexA->point, vertexB->point, green);
        }
        qh_setfree(qh, &vertices);
      }
    }
  }
} /* printfacet3geom_simplicial */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet3math">-</a>

  qh_printfacet3math(qh, fp, facet, notfirst )
    print 3-d Maple or Mathematica output for a facet

  notes:
    may be non-simplicial
    use %16.8f since Mathematica 2.2 does not handle exponential format
    see qh_printfacet2math
*/
void qh_printfacet3math(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format, int notfirst) {
  vertexT *vertex, **vertexp;
  setT *points, *vertices;
  pointT *point, **pointp;
  boolT firstpoint= True;
  realT dist;
  const char *pointfmt, *endfmt;

  // 如果不是第一个 facet，则打印输出之前应该打印的格式
  if (notfirst)
    qh_fprintf(qh, fp, 9105, ",\n");

# 调用qh_fprintf函数，向文件流fp中写入格式化的字符串",\n"，使用标识符9105。

  vertices= qh_facet3vertex(qh, facet);

# 调用qh_facet3vertex函数，获取facet的顶点列表，并将结果赋给vertices变量。
  points= qh_settemp(qh, qh_setsize(qh, vertices));

# 创建临时集合points，其大小为vertices集合的大小，使用qh_settemp函数实现。
  FOREACHvertex_(vertices) {

# 对vertices集合中的每个顶点执行以下操作：
    zinc_(Zdistio);

# 增加Zdistio计数器的值。
    qh_distplane(qh, vertex->point, facet, &dist);

# 调用qh_distplane函数，计算顶点vertex关于facet所在平面的距离，结果保存在dist中。
    point= qh_projectpoint(qh, vertex->point, facet, dist);

# 调用qh_projectpoint函数，将顶点vertex投影到facet所在平面上，结果保存在point中。
    qh_setappend(qh, &points, point);

# 将point添加到points集合的末尾，使用qh_setappend函数。
  }
  if (format == qh_PRINTmaple) {

# 如果format等于qh_PRINTmaple，执行以下操作：
    qh_fprintf(qh, fp, 9106, "[");

# 向文件流fp中写入格式化的字符串"["，使用标识符9106。
    pointfmt= "[%16.8f, %16.8f, %16.8f]";

# 设置pointfmt格式字符串，用于格式化三维点的输出。
    endfmt= "]";

# 设置endfmt格式字符串，用于输出字符串的结尾。
  }else {

# 否则，执行以下操作：
    qh_fprintf(qh, fp, 9107, "Polygon[{");

# 向文件流fp中写入格式化的字符串"Polygon[{"，使用标识符9107。
    pointfmt= "{%16.8f, %16.8f, %16.8f}";

# 设置pointfmt格式字符串，用于格式化三维点的输出。
    endfmt= "}]";

# 设置endfmt格式字符串，用于输出字符串的结尾。
  }
  FOREACHpoint_(points) {

# 对points集合中的每个点执行以下操作：
    if (firstpoint)
      firstpoint= False;
    else
      qh_fprintf(qh, fp, 9108, ",\n");

# 如果是第一个点，将firstpoint设置为False；否则，向文件流fp中写入格式化的字符串",\n"，使用标识符9108。
    qh_fprintf(qh, fp, 9109, pointfmt, point[0], point[1], point[2]);

# 向文件流fp中写入格式化的字符串，使用pointfmt格式化point的前三个元素，标识符9109。
  }
  FOREACHpoint_(points)
    qh_memfree(qh, point, qh->normal_size);

# 释放points集合中每个点的内存，使用qh_memfree函数。
  qh_settempfree(qh, &points);

# 释放临时集合points及其内部元素占用的内存，使用qh_settempfree函数。
  qh_settempfree(qh, &vertices);

# 释放临时集合vertices及其内部元素占用的内存，使用qh_settempfree函数。
  qh_fprintf(qh, fp, 9110, "%s", endfmt);

# 向文件流fp中写入格式化的字符串endfmt，使用标识符9110。
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet3math">-</a>

  qh_printfacet3math(qh, fp, facet, format )
    print mathematical representation of a 3-d facet (equation, area, and volume)

  notes:
    'format' determines the format of output (e.g., qh_PRINTgeom)
*/
void qh_printfacet3math(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format) {
  realT *eq;
  int k;

  eq= facet->eqn;
  qh_fprintf(qh, fp, 9109, " {%.2g,%.2g,%.2g,%6.2g} %6.2g %6.2g\n",
             eq[qh->hull_dim], eq[qh->hull_dim+1], eq[qh->hull_dim+2],
             eq[qh->hull_dim+3], facet->f.area, facet->f.volume);
} /* printfacet3math */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet3vertex">-</a>

  qh_printfacet3vertex(qh, fp, facet, format )
    print vertices in a 3-d facet as point ids

  notes:
    prints number of vertices first if format == qh_PRINToff
    the facet may be non-simplicial
*/
void qh_printfacet3vertex(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format) {
  vertexT *vertex, **vertexp;
  setT *vertices;

  vertices= qh_facet3vertex(qh, facet);
  if (format == qh_PRINToff)
    qh_fprintf(qh, fp, 9111, "%d ", qh_setsize(qh, vertices));
  FOREACHvertex_(vertices)
    qh_fprintf(qh, fp, 9112, "%d ", qh_pointid(qh, vertex->point));
  qh_fprintf(qh, fp, 9113, "\n");
  qh_settempfree(qh, &vertices);
} /* printfacet3vertex */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet4geom_nonsimplicial">-</a>

  qh_printfacet4geom_nonsimplicial(qh, fp, facet, color )
    print Geomview 4OFF file for a 4d nonsimplicial facet
    prints all ridges to unvisited neighbors (qh.visit_id)
    if qh.DROPdim
      prints in OFF format

  notes:
    must agree with printend4geom()
*/
void qh_printfacet4geom_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]) {
  facetT *neighbor;
  ridgeT *ridge, **ridgep;
  vertexT *vertex, **vertexp;
  pointT *point;
  int k;
  realT dist;

  facet->visitid= qh->visit_id;
  if (qh->PRINTnoplanes || (facet->visible && qh->NEWfacets))
    return;
  FOREACHridge_(facet->ridges) {
    neighbor= otherfacet_(ridge, facet);
    if (neighbor->visitid == qh->visit_id)
      continue;
    if (qh->PRINTtransparent && !neighbor->good)
      continue;
    if (qh->DOintersections)
      qh_printhyperplaneintersection(qh, fp, facet, neighbor, ridge->vertices, color);
    else {
      if (qh->DROPdim >= 0)
        qh_fprintf(qh, fp, 9114, "OFF 3 1 1 # f%d\n", facet->id);
      else {
        qh->printoutvar++;
        qh_fprintf(qh, fp, 9115, "# r%d between f%d f%d\n", ridge->id, facet->id, neighbor->id);
      }
      FOREACHvertex_(ridge->vertices) {
        zinc_(Zdistio);
        qh_distplane(qh, vertex->point,facet, &dist);
        point=qh_projectpoint(qh, vertex->point,facet, dist);
        for (k=0; k < qh->hull_dim; k++) {
          if (k != qh->DROPdim)
            qh_fprintf(qh, fp, 9116, "%8.4g ", point[k]);
        }
        qh_fprintf(qh, fp, 9117, "\n");
        qh_memfree(qh, point, qh->normal_size);
      }
      if (qh->DROPdim >= 0)
        qh_fprintf(qh, fp, 9118, "3 0 1 2 %8.4g %8.4g %8.4g\n", color[0], color[1], color[2]);
    }
  }
} /* printfacet4geom_nonsimplicial */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacet4geom_simplicial">-</a>

  qh_printfacet4geom_simplicial(qh, fp, facet, color )
    print Geomview 4OFF file for a 4d simplicial facet
    prints triangles for unvisited neighbors (qh.visit_id)

  notes:
    # 必须与 printend4geom() 函数一致
/*
  打印简化几何的四面体（facet）信息
*/
void qh_printfacet4geom_simplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]) {
  setT *vertices; // 定义一个 set 类型的指针，保存顶点集合
  facetT *neighbor, **neighborp; // 定义 facet 类型的指针 neighbor 和 neighbor 的指针数组
  vertexT *vertex, **vertexp; // 定义 vertex 类型的指针 vertex 和 vertex 的指针数组
  int k; // 定义整型变量 k

  facet->visitid= qh->visit_id; // 设置 facet 的访问 ID 为当前的访问 ID
  if (qh->PRINTnoplanes || (facet->visible && qh->NEWfacets)) // 如果不打印平面或 facet 可见且为新的 facet
    return; // 直接返回，不执行后续代码
  FOREACHneighbor_(facet) { // 遍历 facet 的所有邻接 facet
    if (neighbor->visitid == qh->visit_id) // 如果邻接 facet 已经访问过，跳过
      continue;
    if (qh->PRINTtransparent && !neighbor->good) // 如果需要打印透明 facet 且邻接 facet 不好，跳过
      continue;
    vertices= qh_setnew_delnthsorted(qh, facet->vertices, qh->hull_dim, SETindex_(facet->neighbors, neighbor), 0); // 生成新的顶点集合，按指定的顺序排好序
    if (qh->DOintersections) // 如果需要打印交点信息
      qh_printhyperplaneintersection(qh, fp, facet, neighbor, vertices, color); // 打印超平面交点信息
    else {
      if (qh->DROPdim >= 0) // 如果需要丢弃某维度
        qh_fprintf(qh, fp, 9119, "OFF 3 1 1 # ridge between f%d f%d\n", facet->id, neighbor->id); // 打印边界信息
      else {
        qh->printoutvar++; // 增加输出变量计数
        qh_fprintf(qh, fp, 9120, "# ridge between f%d f%d\n", facet->id, neighbor->id); // 打印边界信息
      }
      FOREACHvertex_(vertices) { // 遍历顶点集合中的每个顶点
        for (k=0; k < qh->hull_dim; k++) { // 遍历顶点的维度
          if (k != qh->DROPdim) // 如果维度不等于丢弃的维度
            qh_fprintf(qh, fp, 9121, "%8.4g ", vertex->point[k]); // 打印顶点的坐标
        }
        qh_fprintf(qh, fp, 9122, "\n"); // 换行
      }
      if (qh->DROPdim >= 0) // 如果需要丢弃某维度
        qh_fprintf(qh, fp, 9123, "3 0 1 2 %8.4g %8.4g %8.4g\n", color[0], color[1], color[2]); // 打印颜色信息
    }
    qh_setfree(qh, &vertices); // 释放顶点集合
  }
} /* printfacet4geom_simplicial */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacetNvertex_nonsimplicial">-</a>

  qh_printfacetNvertex_nonsimplicial(qh, fp, facet, id, format )
    打印 N-d 非简化四面体的顶点
    将每条边三角化到指定 ID
*/
void qh_printfacetNvertex_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, int id, qh_PRINT format) {
  vertexT *vertex, **vertexp; // 定义 vertex 类型的指针 vertex 和 vertex 的指针数组
  ridgeT *ridge, **ridgep; // 定义 ridge 类型的指针 ridge 和 ridge 的指针数组

  if (facet->visible && qh->NEWfacets) // 如果 facet 可见且为新的 facet
    return; // 直接返回，不执行后续代码
  FOREACHridge_(facet->ridges) { // 遍历 facet 的所有边
    if (format == qh_PRINTtriangles) // 如果格式为三角形
      qh_fprintf(qh, fp, 9124, "%d ", qh->hull_dim); // 打印维度
    qh_fprintf(qh, fp, 9125, "%d ", id); // 打印 ID
    if ((ridge->top == facet) ^ qh_ORIENTclock) { // 如果边的顶点在 facet 上，或是与方向钟相反
      FOREACHvertex_(ridge->vertices) // 遍历边的顶点
        qh_fprintf(qh, fp, 9126, "%d ", qh_pointid(qh, vertex->point)); // 打印顶点 ID
    } else {
      FOREACHvertexreverse12_(ridge->vertices) // 遍历边的顶点，按反向顺序
        qh_fprintf(qh, fp, 9127, "%d ", qh_pointid(qh, vertex->point)); // 打印顶点 ID
    }
    qh_fprintf(qh, fp, 9128, "\n"); // 换行
  }
} /* printfacetNvertex_nonsimplicial */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacetNvertex_simplicial">-</a>

  qh_printfacetNvertex_simplicial(qh, fp, facet, format )
    打印 N-d 简化四面体的顶点
    打印非简化四面体的顶点信息
      2-d 四面体（通过 qh_mergefacet2d 保持方向）
      打印 4-d 及更高维的 PRINToff（'o'）
*/
/* 打印简单形式的面的顶点信息到文件流中

   qh_printfacetNvertex_simplicial:
   qh - Qhull的全局数据结构
   fp - 要写入的文件指针
   facet - 要打印顶点信息的面

   如果输出格式是q_PRINToff或者qh_PRINTtriangles，则打印面的顶点数目
*/
void qh_printfacetNvertex_simplicial(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format) {
  vertexT *vertex, **vertexp;

  if (format == qh_PRINToff || format == qh_PRINTtriangles)
    qh_fprintf(qh, fp, 9129, "%d ", qh_setsize(qh, facet->vertices));
  
  /* 如果面的方向不是qh_ORIENTclock，或者qh->hull_dim > 2并且面不是简单形式的 */
  if ((facet->toporient ^ qh_ORIENTclock)
  || (qh->hull_dim > 2 && !facet->simplicial)) {
    /* 正向遍历面的顶点列表，打印顶点的ID */
    FOREACHvertex_(facet->vertices)
      qh_fprintf(qh, fp, 9130, "%d ", qh_pointid(qh, vertex->point));
  } else {
    /* 反向遍历面的顶点列表，打印顶点的ID */
    FOREACHvertexreverse12_(facet->vertices)
      qh_fprintf(qh, fp, 9131, "%d ", qh_pointid(qh, vertex->point));
  }
  
  /* 打印换行符 */
  qh_fprintf(qh, fp, 9132, "\n");
} /* printfacetNvertex_simplicial */


/* 打印面的头部字段到文件流中

   qh_printfacetheader:
   qh - Qhull的全局数据结构
   fp - 要写入的文件指针
   facet - 要打印头部信息的面

   打印面的标识和各种标志位，用于输出 'f' 和调试信息
*/
void qh_printfacetheader(qhT *qh, FILE *fp, facetT *facet) {
  pointT *point, **pointp, *furthest;
  facetT *neighbor, **neighborp;
  realT dist;

  /* 如果面是合并后的面 */
  if (facet == qh_MERGEridge) {
    qh_fprintf(qh, fp, 9133, " MERGEridge\n");
    return;
  } else if (facet == qh_DUPLICATEridge) {
    qh_fprintf(qh, fp, 9134, " DUPLICATEridge\n");
    return;
  } else if (!facet) {
    qh_fprintf(qh, fp, 9135, " NULLfacet\n");
    return;
  }
  
  /* 保存旧的随机距离计算状态并关闭随机距离计算 */
  qh->old_randomdist= qh->RANDOMdist;
  qh->RANDOMdist= False;
  
  /* 打印面的ID */
  qh_fprintf(qh, fp, 9136, "- f%d\n", facet->id);
  qh_fprintf(qh, fp, 9137, "    - flags:");
  
  /* 根据面的各种标志位打印相应的信息 */
  if (facet->toporient)
    qh_fprintf(qh, fp, 9138, " top");
  else
    qh_fprintf(qh, fp, 9139, " bottom");
  if (facet->simplicial)
    qh_fprintf(qh, fp, 9140, " simplicial");
  if (facet->tricoplanar)
    qh_fprintf(qh, fp, 9141, " tricoplanar");
  if (facet->upperdelaunay)
    qh_fprintf(qh, fp, 9142, " upperDelaunay");
  if (facet->visible)
    qh_fprintf(qh, fp, 9143, " visible");
  if (facet->newfacet)
    qh_fprintf(qh, fp, 9144, " newfacet");
  if (facet->tested)
    qh_fprintf(qh, fp, 9145, " tested");
  if (!facet->good)
    qh_fprintf(qh, fp, 9146, " notG");
  if (facet->seen && qh->IStracing)
    qh_fprintf(qh, fp, 9147, " seen");
  if (facet->seen2 && qh->IStracing)
    qh_fprintf(qh, fp, 9418, " seen2");
  if (facet->isarea)
    qh_fprintf(qh, fp, 9419, " isarea");
  if (facet->coplanarhorizon)
    qh_fprintf(qh, fp, 9148, " coplanarhorizon");
  if (facet->mergehorizon)
    qh_fprintf(qh, fp, 9149, " mergehorizon");
  if (facet->cycledone)
    qh_fprintf(qh, fp, 9420, " cycledone");
  if (facet->keepcentrum)
    qh_fprintf(qh, fp, 9150, " keepcentrum");
  if (facet->dupridge)
    qh_fprintf(qh, fp, 9151, " dupridge");
  if (facet->mergeridge && !facet->mergeridge2)
    qh_fprintf(qh, fp, 9152, " mergeridge1");
  if (facet->mergeridge2)
    qh_fprintf(qh, fp, 9153, " mergeridge2");
  if (facet->newmerge)
    qh_fprintf(qh, fp, 9154, " newmerge");
  if (facet->flipped)
    qh_fprintf(qh, fp, 9155, " flipped");
  
  /* 打印换行符 */
  qh_fprintf(qh, fp, 9132, "\n");
} /* printfacetheader */
    # 输出一个描述性消息到文件流 fp，指示 facet 被翻转了
    qh_fprintf(qh, fp, 9155, " flipped");
  # 如果 facet->notfurthest 为真，输出描述性消息到文件流 fp，指示 facet 不是最远点
    qh_fprintf(qh, fp, 9156, " notfurthest");
  # 如果 facet->degenerate 为真，输出描述性消息到文件流 fp，指示 facet 是退化的
    qh_fprintf(qh, fp, 9157, " degenerate");
  # 如果 facet->redundant 为真，输出描述性消息到文件流 fp，指示 facet 是冗余的
    qh_fprintf(qh, fp, 9158, " redundant");
  # 输出一个换行符到文件流 fp
    qh_fprintf(qh, fp, 9159, "\n");
  # 如果 facet->isarea 为真，输出描述性消息到文件流 fp，指示 facet 的面积值
    qh_fprintf(qh, fp, 9160, "    - area: %2.2g\n", facet->f.area);
  # 否则，如果 qh->NEWfacets 为真且 facet 是可见的且有替换对象，输出描述性消息到文件流 fp，指示 facet 的替换关系
    qh_fprintf(qh, fp, 9161, "    - replacement: f%d\n", facet->f.replace->id);
  # 否则，如果 facet->newfacet 为真
  }else if (facet->newfacet) {
    # 如果 facet->f.samecycle 也为真且不等于 facet 本身，输出描述性消息到文件流 fp，指示 facet 与同一个可见/地平线相同的 facet
      qh_fprintf(qh, fp, 9162, "    - shares same visible/horizon as f%d\n", facet->f.samecycle->id);
  # 否则，如果 facet->tricoplanar 为真（且 !isarea）
  }else if (facet->tricoplanar /* !isarea */) {
    # 如果 facet->f.triowner 为真，输出描述性消息到文件流 fp，指示该 facet 拥有法线和中心的 facet
      qh_fprintf(qh, fp, 9163, "    - owner of normal & centrum is facet f%d\n", facet->f.triowner->id);
  # 否则，如果 facet->f.newcycle 为真，输出描述性消息到文件流 fp，指示 facet 是新周期的地平线
    qh_fprintf(qh, fp, 9164, "    - was horizon to f%d\n", facet->f.newcycle->id);
  # 如果 facet->nummerge 等于 qh_MAXnummerge，输出描述性消息到文件流 fp，指示 facet 的最大合并次数
    qh_fprintf(qh, fp, 9427, "    - merges: %dmax\n", qh_MAXnummerge);
  # 否则，如果 facet->nummerge 不为零，输出描述性消息到文件流 fp，指示 facet 的合并次数
    qh_fprintf(qh, fp, 9165, "    - merges: %d\n", facet->nummerge);
  # 打印 facet 的法线坐标到文件流 fp
    qh_printpointid(qh, fp, "    - normal: ", qh->hull_dim, facet->normal, qh_IDunknown);
  # 输出 facet 的偏移值到文件流 fp
    qh_fprintf(qh, fp, 9166, "    - offset: %10.7g\n", facet->offset);
  # 如果 qh->CENTERtype 是 qh_ASvoronoi 或者 facet->center 为真，输出 facet 的中心信息到文件流 fp
    qh_printcenter(qh, fp, qh_PRINTfacets, "    - center: ", facet);
#if qh_MAXoutside
  // 如果定义了 qh_MAXoutside，检查 facet->maxoutside 是否大于 qh->DISTround
  if (facet->maxoutside > qh->DISTround) /* initial value */
    // 输出 facet 的最大外距离
    qh_fprintf(qh, fp, 9167, "    - maxoutside: %10.7g\n", facet->maxoutside);
#endif

// 如果 facet->outsideset 不为空集
if (!SETempty_(facet->outsideset)) {
  furthest= (pointT *)qh_setlast(facet->outsideset);
  // 如果外部点集合大小小于 6
  if (qh_setsize(qh, facet->outsideset) < 6) {
    // 输出外部点集合的信息，包括最远点的标识
    qh_fprintf(qh, fp, 9168, "    - outside set(furthest p%d):\n", qh_pointid(qh, furthest));
    // 遍历并输出每个点的详细信息
    FOREACHpoint_(facet->outsideset)
      qh_printpoint(qh, fp, "     ", point);
  }else if (qh_setsize(qh, facet->outsideset) < 21) {
    // 如果外部点集合大小在 6 到 20 之间，直接输出所有点的信息
    qh_printpoints(qh, fp, "    - outside set:", facet->outsideset);
  }else {
    // 如果外部点集合超过 20 个点，输出点的数量和最远点的信息
    qh_fprintf(qh, fp, 9169, "    - outside set:  %d points.", qh_setsize(qh, facet->outsideset));
    qh_printpoint(qh, fp, "  Furthest", furthest);
  }
}

#if !qh_COMPUTEfurthest
// 如果未计算最远距离，输出 facet 的最远距离
qh_fprintf(qh, fp, 9170, "    - furthest distance= %2.2g\n", facet->furthestdist);
#endif

// 如果 facet->coplanarset 不为空集
if (!SETempty_(facet->coplanarset)) {
  furthest= (pointT *)qh_setlast(facet->coplanarset);
  // 如果共面点集合大小小于 6
  if (qh_setsize(qh, facet->coplanarset) < 6) {
    // 输出共面点集合的信息，包括最远点的标识
    qh_fprintf(qh, fp, 9171, "    - coplanar set(furthest p%d):\n", qh_pointid(qh, furthest));
    // 遍历并输出每个点的详细信息
    FOREACHpoint_(facet->coplanarset)
      qh_printpoint(qh, fp, "     ", point);
  }else if (qh_setsize(qh, facet->coplanarset) < 21) {
    // 如果共面点集合大小在 6 到 20 之间，直接输出所有点的信息
    qh_printpoints(qh, fp, "    - coplanar set:", facet->coplanarset);
  }else {
    // 如果共面点集合超过 20 个点，输出点的数量和最远点的信息
    qh_fprintf(qh, fp, 9172, "    - coplanar set:  %d points.", qh_setsize(qh, facet->coplanarset));
    qh_printpoint(qh, fp, "  Furthest", furthest);
  }
  // 增加 Zdistio 计数
  zinc_(Zdistio);
  // 计算并输出最远点到 facet 的距离
  qh_distplane(qh, furthest, facet, &dist);
  qh_fprintf(qh, fp, 9173, "      furthest distance= %2.2g\n", dist);
}

// 输出 facet 的顶点信息
qh_printvertices(qh, fp, "    - vertices:", facet->vertices);

// 输出 facet 的相邻面信息
qh_fprintf(qh, fp, 9174, "    - neighboring facets:");
FOREACHneighbor_(facet) {
  // 根据 neighbor 类型输出相邻面的信息
  if (neighbor == qh_MERGEridge)
    qh_fprintf(qh, fp, 9175, " MERGEridge");
  else if (neighbor == qh_DUPLICATEridge)
    qh_fprintf(qh, fp, 9176, " DUPLICATEridge");
  else
    qh_fprintf(qh, fp, 9177, " f%d", neighbor->id);
}
// 输出换行符
qh_fprintf(qh, fp, 9178, "\n");

// 恢复 qh->RANDOMdist 的值为 qh->old_randomdist
qh->RANDOMdist= qh->old_randomdist;
} /* printfacetheader */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacetridges">-</a>

  qh_printfacetridges(qh, fp, facet )
    prints ridges of a facet to fp

  notes:
    ridges printed in neighbor order
    assumes the ridges exist
    for 'f' output
    same as QhullFacet::printRidges
*/
void qh_printfacetridges(qhT *qh, FILE *fp, facetT *facet) {
  facetT *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int numridges= 0;
  int n;

  // 如果 facet 可见并且是新创建的面
  if (facet->visible && qh->NEWfacets) {
    // 输出面的潜在标识的边
    qh_fprintf(qh, fp, 9179, "    - ridges (tentative ids):");
    // 遍历并输出面的所有边的标识
    FOREACHridge_(facet->ridges)
      qh_fprintf(qh, fp, 9180, " r%d", ridge->id);
    // 输出换行符
    qh_fprintf(qh, fp, 9181, "\n");
  }else {
    // 否则，正常输出面的边信息
    qh_fprintf(qh, fp, 9182, "    - ridges:\n");
    # 遍历 facet 结构体中的 ridges 链表，初始化每个 ridge 的 seen 标志为 False
    FOREACHridge_(facet->ridges)
      ridge->seen= False;
    # 如果凸壳的维度为 3
    if (qh->hull_dim == 3) {
      # 从 facet 的 ridges 中获取第一个 ridge
      ridge= SETfirstt_(facet->ridges, ridgeT);
      # 当 ridge 存在且未被标记为 seen 时循环执行以下操作
      while (ridge && !ridge->seen) {
        # 将 ridge 的 seen 标志设置为 True
        ridge->seen= True;
        # 打印 ridge 的详细信息到文件流 fp
        qh_printridge(qh, fp, ridge);
        # 增加 numridges 计数
        numridges++;
        # 获取下一个 3D 空间中的 ridge
        ridge= qh_nextridge3d(ridge, facet, NULL);
      }
    }else {
      # 对于非三维空间的情况，遍历 facet 的邻居列表
      FOREACHneighbor_(facet) {
        # 遍历 facet 的 ridges 列表
        FOREACHridge_(facet->ridges) {
          # 如果 ridge 的另一端相邻的 facet 与当前邻居相同且 ridge 未被标记为 seen
          if (otherfacet_(ridge, facet) == neighbor && !ridge->seen) {
            # 将 ridge 的 seen 标志设置为 True
            ridge->seen= True;
            # 打印 ridge 的详细信息到文件流 fp
            qh_printridge(qh, fp, ridge);
            # 增加 numridges 计数
            numridges++;
          }
        }
      }
    }
    # 获取 facet 中 ridges 集合的大小
    n= qh_setsize(qh, facet->ridges);
    # 如果 ridges 集合中只有一个元素且 facet 是新生成的且 qh->NEWtentative 为真
    if (n == 1 && facet->newfacet && qh->NEWtentative) {
      # 输出一条特定消息到文件流 fp，表明这是一个地平线 ridge 到可见 facet 的情况
      qh_fprintf(qh, fp, 9411, "     - horizon ridge to visible facet\n");
    }
    # 如果实际处理的 ridges 数量与 ridges 集合中元素数量不一致
    if (numridges != n) {
      # 输出一条消息到文件流 fp，列出所有的 ridges
      qh_fprintf(qh, fp, 9183, "     - all ridges:");
      # 遍历 facet 的 ridges 集合，输出每个 ridge 的 ID
      FOREACHridge_(facet->ridges)
        qh_fprintf(qh, fp, 9184, " r%d", ridge->id);
      # 输出换行符
      qh_fprintf(qh, fp, 9185, "\n");
    }
    /* non-3d ridges w/o non-simplicial neighbors */
    # 遍历 facet 的 ridges 集合，对于未被标记为 seen 的 ridge
    FOREACHridge_(facet->ridges) {
      if (!ridge->seen)
        # 打印 ridge 的详细信息到文件流 fp
        qh_printridge(qh, fp, ridge);
    }
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printfacets">-</a>

  qh_printfacets(qh, fp, format, facetlist, facets, printall )
    prints facetlist and/or facet set in output format

  notes:
    also used for specialized formats ('FO' and summary)
    turns off 'Rn' option since want actual numbers
*/
void qh_printfacets(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall) {
  int numfacets, numsimplicial, numridges, totneighbors, numcoplanars, numtricoplanars;
  facetT *facet, **facetp;
  setT *vertices;
  coordT *center;
  realT outerplane, innerplane;

  qh->old_randomdist= qh->RANDOMdist;
  qh->RANDOMdist= False; /* 设置随机距离为假 */

  if (qh->CDDoutput && (format == qh_PRINTcentrums || format == qh_PRINTpointintersect || format == qh_PRINToff))
    qh_fprintf(qh, qh->ferr, 7056, "qhull warning: CDD format is not available for centrums, halfspace\nintersections, and OFF file format.\n");

  if (format == qh_PRINTnone)
    ; /* 什么都不打印 */
  else if (format == qh_PRINTaverage) {
    vertices= qh_facetvertices(qh, facetlist, facets, printall); /* 获取顶点集合 */
    center= qh_getcenter(qh, vertices); /* 计算几何中心 */
    qh_fprintf(qh, fp, 9186, "%d 1\n", qh->hull_dim); /* 打印维度信息 */
    qh_printpointid(qh, fp, NULL, qh->hull_dim, center, qh_IDunknown); /* 打印几何中心点的标识 */
    qh_memfree(qh, center, qh->normal_size); /* 释放几何中心内存 */
    qh_settempfree(qh, &vertices); /* 释放顶点集合内存 */
  } else if (format == qh_PRINTextremes) {
    if (qh->DELAUNAY)
      qh_printextremes_d(qh, fp, facetlist, facets, printall); /* 打印极值点（Delaunay模式） */
    else if (qh->hull_dim == 2)
      qh_printextremes_2d(qh, fp, facetlist, facets, printall); /* 打印二维极值点 */
    else
      qh_printextremes(qh, fp, facetlist, facets, printall); /* 打印一般极值点 */
  } else if (format == qh_PRINToptions)
    qh_fprintf(qh, fp, 9187, "Options selected for Qhull %s:\n%s\n", qh_version, qh->qhull_options); /* 打印Qhull的选项 */
  else if (format == qh_PRINTpoints && !qh->VORONOI)
    qh_printpoints_out(qh, fp, facetlist, facets, printall); /* 打印输出点 */
  else if (format == qh_PRINTqhull)
    qh_fprintf(qh, fp, 9188, "%s | %s\n", qh->rbox_command, qh->qhull_command); /* 打印Qhull命令 */
  else if (format == qh_PRINTsize) {
    qh_fprintf(qh, fp, 9189, "0\n2 "); /* 打印0和2 */
    qh_fprintf(qh, fp, 9190, qh_REAL_1, qh->totarea); /* 打印总面积 */
    qh_fprintf(qh, fp, 9191, qh_REAL_1, qh->totvol); /* 打印总体积 */
    qh_fprintf(qh, fp, 9192, "\n");
  } else if (format == qh_PRINTsummary) {
    qh_countfacets(qh, facetlist, facets, printall, &numfacets, &numsimplicial,
      &totneighbors, &numridges, &numcoplanars, &numtricoplanars); /* 计算各种面的数量 */
    vertices= qh_facetvertices(qh, facetlist, facets, printall); /* 获取顶点集合 */
    qh_fprintf(qh, fp, 9193, "10 %d %d %d %d %d %d %d %d %d %d\n2 ", qh->hull_dim,
                qh->num_points + qh_setsize(qh, qh->other_points),
                qh->num_vertices, qh->num_facets - qh->num_visible,
                qh_setsize(qh, vertices), numfacets, numcoplanars,
                numfacets - numsimplicial, zzval_(Zdelvertextot),
                numtricoplanars); /* 打印汇总信息 */
    qh_settempfree(qh, &vertices); /* 释放顶点集合内存 */
    // 调用 qh_outerinner 函数，计算 qh 对象的外层和内层平面
    qh_outerinner(qh, NULL, &outerplane, &innerplane);
    // 调用 qh_fprintf 函数，将外层和内层平面的数据格式化输出到文件流 fp 中
    qh_fprintf(qh, fp, 9194, qh_REAL_2n, outerplane, innerplane);
  } else if (format == qh_PRINTvneighbors)
    // 如果 format 等于 qh_PRINTvneighbors，则调用 qh_printvneighbors 函数输出相邻顶点信息
    qh_printvneighbors(qh, fp, facetlist, facets, printall);
  else if (qh->VORONOI && format == qh_PRINToff)
    // 如果 qh 对象启用了 Voronoi 图，并且 format 等于 qh_PRINToff，则调用 qh_printvoronoi 函数输出 Voronoi 图信息
    qh_printvoronoi(qh, fp, format, facetlist, facets, printall);
  else if (qh->VORONOI && format == qh_PRINTgeom) {
    // 如果 qh 对象启用了 Voronoi 图，并且 format 等于 qh_PRINTgeom，则依次调用 qh_printbegin、qh_printvoronoi 和 qh_printend 函数输出几何信息
    qh_printbegin(qh, fp, format, facetlist, facets, printall);
    qh_printvoronoi(qh, fp, format, facetlist, facets, printall);
    qh_printend(qh, fp, format, facetlist, facets, printall);
  } else if (qh->VORONOI
  && (format == qh_PRINTvertices || format == qh_PRINTinner || format == qh_PRINTouter))
    // 如果 qh 对象启用了 Voronoi 图，并且 format 等于 qh_PRINTvertices、qh_PRINTinner 或 qh_PRINTouter，则调用 qh_printvdiagram 函数输出 Voronoi 图中的顶点和边信息
    qh_printvdiagram(qh, fp, format, facetlist, facets, printall);
  else {
    // 其余情况下，依次调用 qh_printbegin、qh_printafacet 和 qh_printend 函数输出所有几何图形的信息
    qh_printbegin(qh, fp, format, facetlist, facets, printall);
    FORALLfacet_(facetlist)
      qh_printafacet(qh, fp, format, facet, printall);
    FOREACHfacet_(facets)
      qh_printafacet(qh, fp, format, facet, printall);
    qh_printend(qh, fp, format, facetlist, facets, printall);
  }
  // 将 qh 对象的随机距离重置为旧的随机距离
  qh->RANDOMdist = qh->old_randomdist;
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printhyperplaneintersection">-</a>

  qh_printhyperplaneintersection(qh, fp, facet1, facet2, vertices, color )
    print Geomview OFF or 4OFF for the intersection of two hyperplanes in 3-d or 4-d
*/
void qh_printhyperplaneintersection(qhT *qh, FILE *fp, facetT *facet1, facetT *facet2,
                   setT *vertices, realT color[3]) {
  realT costheta, denominator, dist1, dist2, s, t, mindenom, p[4];
  vertexT *vertex, **vertexp;
  int i, k;
  boolT nearzero1, nearzero2;

  // 计算两个超平面法向量之间的夹角余弦值
  costheta= qh_getangle(qh, facet1->normal, facet2->normal);
  // 计算分母
  denominator= 1 - costheta * costheta;
  // 计算顶点集合的大小
  i= qh_setsize(qh, vertices);
  // 根据凸包维度不同，选择不同的输出格式
  if (qh->hull_dim == 3)
    qh_fprintf(qh, fp, 9195, "VECT 1 %d 1 %d 1 ", i, i);
  else if (qh->hull_dim == 4 && qh->DROPdim >= 0)
    qh_fprintf(qh, fp, 9196, "OFF 3 1 1 ");
  else
    qh->printoutvar++;
  // 输出注释信息
  qh_fprintf(qh, fp, 9197, "# intersect f%d f%d\n", facet1->id, facet2->id);
  // 计算一个小数，用于处理分母接近零的情况
  mindenom= 1 / (10.0 * qh->MAXabs_coord);
  // 遍历顶点集合中的每个顶点
  FOREACHvertex_(vertices) {
    zadd_(Zdistio, 2);
    // 计算顶点到两个超平面的距离
    qh_distplane(qh, vertex->point, facet1, &dist1);
    qh_distplane(qh, vertex->point, facet2, &dist2);
    // 计算交点的两个参数 s 和 t
    s= qh_divzero(-dist1 + costheta * dist2, denominator, mindenom, &nearzero1);
    t= qh_divzero(-dist2 + costheta * dist1, denominator, mindenom, &nearzero2);
    // 处理参数接近零的情况
    if (nearzero1 || nearzero2)
      s= t= 0.0;
    // 计算交点的坐标
    for (k=qh->hull_dim; k--; )
      p[k]= vertex->point[k] + facet1->normal[k] * s + facet2->normal[k] * t;
    // 根据凸包维度选择输出格式
    if (qh->PRINTdim <= 3) {
      qh_projectdim3(qh, p, p);
      qh_fprintf(qh, fp, 9198, "%8.4g %8.4g %8.4g # ", p[0], p[1], p[2]);
    }else
      qh_fprintf(qh, fp, 9199, "%8.4g %8.4g %8.4g %8.4g # ", p[0], p[1], p[2], p[3]);
    // 输出对应的顶点信息
    if (nearzero1 + nearzero2)
      qh_fprintf(qh, fp, 9200, "p%d(coplanar facets)\n", qh_pointid(qh, vertex->point));
    else
      qh_fprintf(qh, fp, 9201, "projected p%d\n", qh_pointid(qh, vertex->point));
  }
  // 输出颜色信息，用于表示交点
  if (qh->hull_dim == 3)
    qh_fprintf(qh, fp, 9202, "%8.4g %8.4g %8.4g 1.0\n", color[0], color[1], color[2]);
  else if (qh->hull_dim == 4 && qh->DROPdim >= 0)
    qh_fprintf(qh, fp, 9203, "3 0 1 2 %8.4g %8.4g %8.4g 1.0\n", color[0], color[1], color[2]);
} /* printhyperplaneintersection */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printline3geom">-</a>

  qh_printline3geom(qh, fp, pointA, pointB, color )
    prints a line as a VECT
    prints 0's for qh.DROPdim

  notes:
    if pointA == pointB,
      it's a 1 point VECT
*/
void qh_printline3geom(qhT *qh, FILE *fp, pointT *pointA, pointT *pointB, realT color[3]) {
  int k;
  realT pA[4], pB[4];

  // 将点投影到三维空间
  qh_projectdim3(qh, pointA, pA);
  qh_projectdim3(qh, pointB, pB);
  // 如果两点不重合，输出线段的起点和终点
  if ((fabs(pA[0] - pB[0]) > 1e-3) ||
      (fabs(pA[1] - pB[1]) > 1e-3) ||
      (fabs(pA[2] - pB[2]) > 1e-3)) {
    qh_fprintf(qh, fp, 9204, "VECT 1 2 1 2 1\n");
    # 循环三次，输出数组 pB 中的值，格式为 8 位精度，总共 4 位小数，每个值后跟一个空格
    for (k=0; k < 3; k++)
       qh_fprintf(qh, fp, 9205, "%8.4g ", pB[k]);
    # 输出点的标识号和注释到文件流 fp，格式为 " # p%d"，其中 %d 是点的标识号
    qh_fprintf(qh, fp, 9206, " # p%d\n", qh_pointid(qh, pointB));
  }else
    # 向文件流 fp 输出默认的 VECT 行，指定 RGBA 值均为 1
    qh_fprintf(qh, fp, 9207, "VECT 1 1 1 1 1\n");
  # 输出数组 pA 中的值，格式为 8 位精度，总共 4 位小数，每个值后跟一个空格
  for (k=0; k < 3; k++)
    qh_fprintf(qh, fp, 9208, "%8.4g ", pA[k]);
  # 输出点的标识号和注释到文件流 fp，格式为 " # p%d"，其中 %d 是点的标识号
  qh_fprintf(qh, fp, 9209, " # p%d\n", qh_pointid(qh, pointA));
  # 输出颜色数组 color 的 RGB 值和透明度值到文件流 fp，每个值格式为 8 位精度，总共 4 位小数
  qh_fprintf(qh, fp, 9210, "%8.4g %8.4g %8.4g 1\n", color[0], color[1], color[2]);
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printneighborhood">-</a>

  qh_printneighborhood(qh, fp, format, facetA, facetB, printall )
    print neighborhood of one or two facets

  notes:
    calls qh_findgood_all()
    bumps qh.visit_id
*/
void qh_printneighborhood(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetA, facetT *facetB, boolT printall) {
  facetT *neighbor, **neighborp, *facet;
  setT *facets;

  if (format == qh_PRINTnone)
    return;  // 如果打印格式是 qh_PRINTnone，则直接返回，不执行后续操作
  qh_findgood_all(qh, qh->facet_list);  // 调用 qh_findgood_all() 函数，对 facet_list 进行处理
  if (facetA == facetB)
    facetB= NULL;  // 如果 facetA 等于 facetB，则将 facetB 置为 NULL
  facets= qh_settemp(qh, 2*(qh_setsize(qh, facetA->neighbors)+1));  // 为 facets 分配临时空间
  qh->visit_id++;  // 递增 visit_id，用于标记访问过的 facet
  for (facet=facetA; facet; facet= ((facet == facetA) ? facetB : NULL)) {  // 遍历 facetA 和 facetB（如果有）
    if (facet->visitid != qh->visit_id) {  // 如果 facet 尚未被访问过
      facet->visitid= qh->visit_id;  // 设置 facet 的 visitid 为当前 visit_id
      qh_setappend(qh, &facets, facet);  // 将 facet 添加到 facets 集合中
    }
    FOREACHneighbor_(facet) {  // 遍历 facet 的相邻面
      if (neighbor->visitid == qh->visit_id)  // 如果相邻面已经被访问过，则跳过
        continue;
      neighbor->visitid= qh->visit_id;  // 设置相邻面的 visitid 为当前 visit_id
      if (printall || !qh_skipfacet(qh, neighbor))  // 如果需要打印全部或者不应跳过相邻面
        qh_setappend(qh, &facets, neighbor);  // 将相邻面添加到 facets 集合中
    }
  }
  qh_printfacets(qh, fp, format, NULL, facets, printall);  // 打印 facets 集合中的所有面到文件 fp 中
  qh_settempfree(qh, &facets);  // 释放 facets 的临时空间
} /* printneighborhood */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printpoint">-</a>

  qh_printpoint(qh, fp, string, point )
  qh_printpointid(qh, fp, string, dim, point, id )
    prints the coordinates of a point

  returns:
    if string is defined
      prints 'string p%d'.  Skips p%d if id=qh_IDunknown(-1) or qh_IDnone(-3)

  notes:
    nop if point is NULL
    Same as QhullPoint's printPoint
*/
void qh_printpoint(qhT *qh, FILE *fp, const char *string, pointT *point) {
  int id= qh_pointid(qh, point);  // 获取点的标识符

  qh_printpointid(qh, fp, string, qh->hull_dim, point, id);  // 调用 qh_printpointid() 打印点的坐标
} /* printpoint */

void qh_printpointid(qhT *qh, FILE *fp, const char *string, int dim, pointT *point, int id) {
  int k;
  realT r; /*bug fix*/  // 实数 r，用于存储点的坐标

  if (!point)
    return;  // 如果点为空，则直接返回
  if (string) {  // 如果定义了 string
    qh_fprintf(qh, fp, 9211, "%s", string);  // 打印 string
    if (id != qh_IDunknown && id != qh_IDnone)
      qh_fprintf(qh, fp, 9212, " p%d: ", id);  // 如果 id 不是 qh_IDunknown(-1) 或 qh_IDnone(-3)，则打印 id
  }
  for (k=dim; k--; ) {  // 遍历点的坐标维度
    r= *point++;  // 获取当前维度的坐标值
    if (string)
      qh_fprintf(qh, fp, 9213, " %8.4g", r);  // 如果定义了 string，则按指定格式打印坐标值
    else
      qh_fprintf(qh, fp, 9214, qh_REAL_1, r);  // 否则，按 qh_REAL_1 的格式打印坐标值
  }
  qh_fprintf(qh, fp, 9215, "\n");  // 换行
} /* printpointid */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printpoint3">-</a>

  qh_printpoint3(qh, fp, point )
    prints 2-d, 3-d, or 4-d point as Geomview 3-d coordinates
*/
void qh_printpoint3(qhT *qh, FILE *fp, pointT *point) {
  int k;
  realT p[4];  // 用于存储点的坐标

  qh_projectdim3(qh, point, p);  // 投影点到三维空间
  for (k=0; k < 3; k++)
    qh_fprintf(qh, fp, 9216, "%8.4g ", p[k]);  // 打印点的三维坐标
  qh_fprintf(qh, fp, 9217, " # p%d\n", qh_pointid(qh, point));  // 打印点的标识符
} /* printpoint3 */
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printpoints_out">-</a>

  qh_printpoints_out(qh, fp, facetlist, facets, printall )
    prints vertices, coplanar/inside points, for facets by their point coordinates
    allows qh.CDDoutput

  notes:
    same format as qhull input
    if no coplanar/interior points,
      same order as qh_printextremes
*/
void qh_printpoints_out(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall) {
  // 计算所有点的数量，包括主要点集和其他点集的点数
  int allpoints= qh->num_points + qh_setsize(qh, qh->other_points);
  // numpoints：记录实际需要输出的点的数量
  int numpoints=0, point_i, point_n;
  // 临时存储所有点的集合
  setT *vertices, *points;
  // 当前处理的面片和顶点
  facetT *facet, **facetp;
  pointT *point, **pointp;
  vertexT *vertex, **vertexp;
  int id;

  // 分配一个临时集合来存储所有的点
  points= qh_settemp(qh, allpoints);
  // 将临时点集清零
  qh_setzero(qh, points, 0, allpoints);
  // 获取所有面片的顶点集合
  vertices= qh_facetvertices(qh, facetlist, facets, printall);
  // 遍历每个顶点，将其点的ID存储到临时点集合中
  FOREACHvertex_(vertices) {
    id= qh_pointid(qh, vertex->point);
    if (id >= 0)
      SETelem_(points, id)= vertex->point;
  }

  // 如果需要输出内部或共面点
  if (qh->KEEPinside || qh->KEEPcoplanar || qh->KEEPnearinside) {
    // 遍历每个面片，处理共面点集合
    FORALLfacet_(facetlist) {
      if (!printall && qh_skipfacet(qh, facet))
        continue;
      FOREACHpoint_(facet->coplanarset) {
        id= qh_pointid(qh, point);
        if (id >= 0)
          SETelem_(points, id)= point;
      }
    }
    // 遍历每个用户定义面片，处理共面点集合
    FOREACHfacet_(facets) {
      if (!printall && qh_skipfacet(qh, facet))
        continue;
      FOREACHpoint_(facet->coplanarset) {
        id= qh_pointid(qh, point);
        if (id >= 0)
          SETelem_(points, id)= point;
      }
    }
  }

  // 释放顶点集合的临时存储空间
  qh_settempfree(qh, &vertices);

  // 计算实际输出的点的数量
  FOREACHpoint_i_(qh, points) {
    if (point)
      numpoints++;
  }

  // 根据qh.CDDoutput输出不同的格式
  if (qh->CDDoutput)
    qh_fprintf(qh, fp, 9218, "%s | %s\nbegin\n%d %d real\n", qh->rbox_command,
               qh->qhull_command, numpoints, qh->hull_dim + 1);
  else
    qh_fprintf(qh, fp, 9219, "%d\n%d\n", qh->hull_dim, numpoints);

  // 遍历并输出每个点的坐标
  FOREACHpoint_i_(qh, points) {
    if (point) {
      if (qh->CDDoutput)
        qh_fprintf(qh, fp, 9220, "1 ");
      qh_printpoint(qh, fp, NULL, point);
    }
  }

  // 如果是CDD输出模式，输出结束标记
  if (qh->CDDoutput)
    qh_fprintf(qh, fp, 9221, "end\n");

  // 释放临时点集合的存储空间
  qh_settempfree(qh, &points);
} /* printpoints_out */
    # 计算新点的坐标，新点为原始点加上方向向量与半径的乘积
    pointA[k]= point[k]+diff[k] * radius;
    # 打印从 point 到 pointA 的线段的几何信息到文件流 fp 中，使用指定颜色
    qh_printline3geom(qh, fp, point, pointA, color);
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printpointvect2">-</a>

  qh_printpointvect2(qh, fp, point, normal, center, radius )
    prints a 2-d, 3-d, or 4-d point as 2 3-d VECT's for an imprecise point
*/
void qh_printpointvect2(qhT *qh, FILE *fp, pointT *point, coordT *normal, pointT *center, realT radius) {
  // 定义红色和黄色的颜色向量
  realT red[3]={1, 0, 0}, yellow[3]={1, 1, 0};

  // 调用 qh_printpointvect 输出使用红色向量的 VECT
  qh_printpointvect(qh, fp, point, normal, center, radius, red);
  // 调用 qh_printpointvect 输出使用黄色向量的 VECT
  qh_printpointvect(qh, fp, point, normal, center, -radius, yellow);
} /* printpointvect2 */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printridge">-</a>

  qh_printridge(qh, fp, ridge )
    prints the information in a ridge

  notes:
    for qh_printfacetridges()
    same as operator<< [QhullRidge.cpp]
*/
void qh_printridge(qhT *qh, FILE *fp, ridgeT *ridge) {
  // 打印 ridge 的 ID 和相关信息到文件流 fp
  qh_fprintf(qh, fp, 9222, "     - r%d", ridge->id);
  // 如果 ridge 被测试过，则打印 tested
  if (ridge->tested)
    qh_fprintf(qh, fp, 9223, " tested");
  // 如果 ridge 是非凸的，则打印 nonconvex
  if (ridge->nonconvex)
    qh_fprintf(qh, fp, 9224, " nonconvex");
  // 如果 ridge 包含合并顶点，则打印 mergevertex
  if (ridge->mergevertex)
    qh_fprintf(qh, fp, 9421, " mergevertex");
  // 如果 ridge 包含第二个合并顶点，则打印 mergevertex2
  if (ridge->mergevertex2)
    qh_fprintf(qh, fp, 9422, " mergevertex2");
  // 如果 ridge 是上部分的单纯形，则打印 simplicialtop
  if (ridge->simplicialtop)
    qh_fprintf(qh, fp, 9425, " simplicialtop");
  // 如果 ridge 是底部分的单纯形，则打印 simplicialbot
  if (ridge->simplicialbot)
    qh_fprintf(qh, fp, 9423, " simplicialbot");
  // 打印换行符
  qh_fprintf(qh, fp, 9225, "\n");
  // 打印 ridge 的顶点列表到文件流 fp
  qh_printvertices(qh, fp, "           vertices:", ridge->vertices);
  // 如果 ridge 同时连接顶部和底部，则打印连接的面的 ID
  if (ridge->top && ridge->bottom)
    qh_fprintf(qh, fp, 9226, "           between f%d and f%d\n",
            ridge->top->id, ridge->bottom->id);
} /* printridge */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printspheres">-</a>

  qh_printspheres(qh, fp, vertices, radius )
    prints 3-d vertices as OFF spheres

  notes:
    inflated octahedron from Stuart Levy earth/mksphere2
*/
void qh_printspheres(qhT *qh, FILE *fp, setT *vertices, realT radius) {
  vertexT *vertex, **vertexp;

  // 增加打印输出计数
  qh->printoutnum++;
  // 打印 OFF 格式的球体定义到文件流 fp
  qh_fprintf(qh, fp, 9227, "{appearance {-edge -normal normscale 0} {\n\
INST geom {define vsphere OFF\n\
18 32 48\n\
\n\
0 0 1\n\
1 0 0\n\
0 1 0\n\
-1 0 0\n\
0 -1 0\n\
0 0 -1\n\
0.707107 0 0.707107\n\
0 -0.707107 0.707107\n\
0.707107 -0.707107 0\n\
-0.707107 0 0.707107\n\
-0.707107 -0.707107 0\n\
0 0.707107 0.707107\n\
-0.707107 0.707107 0\n\
0.707107 0.707107 0\n\
0.707107 0 -0.707107\n\
0 0.707107 -0.707107\n\
-0.707107 0 -0.707107\n\
0 -0.707107 -0.707107\n\
\n\
3 0 6 11\n\
3 0 7 6 \n\
3 0 9 7 \n\
3 0 11 9\n\
3 1 6 8 \n\
3 1 8 14\n\
3 1 13 6\n\
3 1 14 13\n\
3 2 11 13\n\
3 2 12 11\n\
3 2 13 15\n\
3 2 15 12\n\
3 3 9 12\n\
3 3 10 9\n\
3 3 12 16\n\
3 3 16 10\n\
3 4 7 10\n\
3 4 8 7\n\
3 4 10 17\n\
3 4 17 8\n\
3 5 14 17\n\
3 5 15 14\n\
3 5 16 15\n\
3 5 17 16\n\
3 6 13 11\n\
3 7 8 6\n\
3 9 10 7\n\
3 11 12 9\n\
3 14 8 17\n\
3 15 13 14\n\
3 16 12 15\n\
void qh_printvdiagram(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall) {
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvdiagram">-</a>

  qh_printvdiagram(qh, fp, format, facetlist, facets, printall )
    print voronoi diagram
      # of pairs of input sites
      #indices site1 site2 vertex1 ...

    sites indexed by input point id
      point 0 is the first input point
    vertices indexed by 'o' and 'p' order
      vertex 0 is the 'vertex-at-infinity'
      vertex 1 is the first Voronoi vertex

  see:
    qh_printvoronoi()
    qh_eachvoronoi_all()

  notes:
    if all facets are upperdelaunay,
      prints upper hull (furthest-site Voronoi diagram)
*/
  setT *vertices;
  int totcount, numcenters;
  boolT isLower;
  qh_RIDGE innerouter= qh_RIDGEall;
  printvridgeT printvridge= NULL;

  if (format == qh_PRINTvertices) {
    innerouter= qh_RIDGEall;
    printvridge= qh_printvridge;
  }else if (format == qh_PRINTinner) {
    innerouter= qh_RIDGEinner;
    printvridge= qh_printvnorm;
  }else if (format == qh_PRINTouter) {
    innerouter= qh_RIDGEouter;
    printvridge= qh_printvnorm;
  }else {
    qh_fprintf(qh, qh->ferr, 6219, "qhull internal error (qh_printvdiagram): unknown print format %d.\n", format);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  vertices= qh_markvoronoi(qh, facetlist, facets, printall, &isLower, &numcenters);
  totcount= qh_printvdiagram2(qh, NULL, NULL, vertices, innerouter, False);
  qh_fprintf(qh, fp, 9231, "%d\n", totcount);
  totcount= qh_printvdiagram2(qh, fp, printvridge, vertices, innerouter, True /* inorder*/);
  qh_settempfree(qh, &vertices);
#if 0  /* for testing qh_eachvoronoi_all */
  qh_fprintf(qh, fp, 9232, "\n");
  totcount= qh_eachvoronoi_all(qh, fp, printvridge, qh->UPPERdelaunay, innerouter, True /* inorder*/);
  qh_fprintf(qh, fp, 9233, "%d\n", totcount);
#endif
} /* printvdiagram */


注释：

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvdiagram">-</a>

  qh_printvdiagram(qh, fp, format, facetlist, facets, printall )
    打印Voronoi图
      # 输入点对的数量
      #索引 站点1 站点2 顶点1 ...

    站点按输入点ID索引
      点0是第一个输入点
    顶点按'o'和'p'顺序索引
      顶点0是'无限远顶点'
      顶点1是第一个Voronoi顶点

  参见:
    qh_printvoronoi()
    qh_eachvoronoi_all()

  注意:
    如果所有的facets都是upperdelaunay，
      打印上凸壳（最远点Voronoi图）
*/
    # 标记 Voronoi 顶点的 facet->visitid
    qh_markvoronoi 将 facet->visitid 用于标记 Voronoi 顶点

    # 将所有 facet->seen 设置为 False
    all facet->seen= False 将所有的 facet->seen 属性设置为 False，表示这些 facet 尚未被访问或处理过

    # 将所有 facet->seen2 设置为 True
    all facet->seen2= True 将所有的 facet->seen2 属性设置为 True，表示这些 facet 已被标记为已处理

  # 返回：
  # 返回 Voronoi 边的总数
  returns:
    total number of Voronoi ridges 返回 Voronoi 边（ridges）的总数

    # 如果 printvridge 为真，
    if printvridge,

      # 对每个边调用 printvridge(fp, vertex, vertexA, centers)
      calls printvridge(fp, vertex, vertexA, centers) 对每个边调用 printvridge 函数，传入 fp, vertex, vertexA, centers 参数

      # 参见 qh_eachvoronoi()
      [see qh_eachvoronoi()] 参见 qh_eachvoronoi() 函数的文档说明，该函数可能用于遍历 Voronoi 结构的每个元素

  # 参见：
  # 查看 qh_eachvoronoi_all() 函数的文档
  see:
    qh_eachvoronoi_all() 参见 qh_eachvoronoi_all() 函数的文档，可能提供了更详细的 Voronoi 结构的全局处理方法
/*
int qh_printvdiagram2(qhT *qh, FILE *fp, printvridgeT printvridge, setT *vertices, qh_RIDGE innerouter, boolT inorder) {
  int totcount= 0;  // 初始化总计数器为0
  int vertex_i, vertex_n;  // 声明顶点索引变量
  vertexT *vertex;  // 声明顶点指针

  FORALLvertices  // 遍历所有顶点
    vertex->seen= False;  // 将顶点的 seen 属性设为 False
  FOREACHvertex_i_(qh, vertices) {  // 遍历给定顶点集合中的每个顶点
    if (vertex) {  // 如果顶点非空
      if (qh->GOODvertex > 0 && qh_pointid(qh, vertex->point)+1 != qh->GOODvertex)  // 检查顶点的 ID 是否符合条件
        continue;  // 如果不符合条件则跳过该顶点
      totcount += qh_eachvoronoi(qh, fp, printvridge, vertex, !qh_ALL, innerouter, inorder);  // 调用函数处理该顶点的 Voronoi 图
    }
  }
  return totcount;  // 返回处理的顶点数目
} /* printvdiagram2 */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvertex">-</a>

  qh_printvertex(qh, fp, vertex )
    prints the information in a vertex
    Duplicated as operator<< [QhullVertex.cpp]
*/
void qh_printvertex(qhT *qh, FILE *fp, vertexT *vertex) {
  pointT *point;  // 定义点指针变量
  int k, count= 0;  // 声明循环计数器和计数变量
  facetT *neighbor, **neighborp;  // 声明面邻居指针和指针数组
  realT r; /*bug fix*/  // 定义实数 r，用于修复 bug

  if (!vertex) {  // 如果顶点为空
    qh_fprintf(qh, fp, 9234, "  NULLvertex\n");  // 打印 NULLvertex 信息
    return;  // 返回
  }
  qh_fprintf(qh, fp, 9235, "- p%d(v%d):", qh_pointid(qh, vertex->point), vertex->id);  // 打印顶点的 ID 和点 ID
  point= vertex->point;  // 获取顶点的点
  if (point) {  // 如果点非空
    for (k=qh->hull_dim; k--; ) {  // 遍历点的每个维度
      r= *point++;  // 获取点的坐标
      qh_fprintf(qh, fp, 9236, " %5.2g", r);  // 打印点的坐标
    }
  }
  if (vertex->deleted)  // 如果顶点已删除
    qh_fprintf(qh, fp, 9237, " deleted");  // 打印 deleted
  if (vertex->delridge)  // 如果顶点的边界已删除
    qh_fprintf(qh, fp, 9238, " delridge");  // 打印 delridge
  if (vertex->newfacet)  // 如果顶点是新的面
    qh_fprintf(qh, fp, 9415, " newfacet");  // 打印 newfacet
  if (vertex->seen && qh->IStracing)  // 如果顶点已见过且在追踪状态
    qh_fprintf(qh, fp, 9416, " seen");  // 打印 seen
  if (vertex->seen2 && qh->IStracing)  // 如果顶点已见过第二次且在追踪状态
    qh_fprintf(qh, fp, 9417, " seen2");  // 打印 seen2
  qh_fprintf(qh, fp, 9239, "\n");  // 打印换行
  if (vertex->neighbors) {  // 如果顶点有邻居
    qh_fprintf(qh, fp, 9240, "  neighbors:");  // 打印邻居信息头部
    FOREACHneighbor_(vertex) {  // 遍历每个邻居
      if (++count % 100 == 0)  // 每100个邻居换行
        qh_fprintf(qh, fp, 9241, "\n     ");
      qh_fprintf(qh, fp, 9242, " f%d", neighbor->id);  // 打印邻居的 ID
    }
    qh_fprintf(qh, fp, 9243, "\n");  // 打印换行
  }
} /* printvertex */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvertexlist">-</a>

  qh_printvertexlist(qh, fp, string, facetlist, facets, printall )
    prints vertices used by a facetlist or facet set
    tests qh_skipfacet() if !printall
*/
void qh_printvertexlist(qhT *qh, FILE *fp, const char* string, facetT *facetlist,
                         setT *facets, boolT printall) {
  vertexT *vertex, **vertexp;  // 声明顶点指针和顶点指针数组
  setT *vertices;  // 声明顶点集合

  vertices= qh_facetvertices(qh, facetlist, facets, printall);  // 获取由面列表或面集使用的顶点集合
  qh_fprintf(qh, fp, 9244, "%s", string);  // 打印给定的字符串信息
  FOREACHvertex_(vertices)  // 遍历每个顶点
    qh_printvertex(qh, fp, vertex);  // 打印顶点信息
  qh_settempfree(qh, &vertices);  // 释放临时顶点集合内存
} /* printvertexlist */


/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvertices">-</a>

  qh_printvertices(qh, fp, string, vertices )
    prints vertices in a set
    duplicated as printVertexSet [QhullVertex.cpp]
*/
/* 打印顶点信息到文件流中 */
void qh_printvertices(qhT *qh, FILE *fp, const char* string, setT *vertices) {
  vertexT *vertex, **vertexp;

  qh_fprintf(qh, fp, 9245, "%s", string);  // 使用给定的字符串格式化输出到文件流
  FOREACHvertex_(vertices)
    qh_fprintf(qh, fp, 9246, " p%d(v%d)", qh_pointid(qh, vertex->point), vertex->id);
    // 对于每个顶点，输出其在点集中的ID和顶点本身的ID到文件流
  qh_fprintf(qh, fp, 9247, "\n");  // 输出换行符到文件流
} /* printvertices */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvneighbors">-</a>

  qh_printvneighbors(qh, fp, facetlist, facets, printall )
    print vertex neighbors of vertices in facetlist and facets ('FN')

  notes:
    qh_countfacets clears facet->visitid for non-printed facets

  design:
    collect facet count and related statistics
    if necessary, build neighbor sets for each vertex
    collect vertices in facetlist and facets
    build a point array for point->vertex and point->coplanar facet
    for each point
      list vertex neighbors or coplanar facet
*/
void qh_printvneighbors(qhT *qh, FILE *fp, facetT* facetlist, setT *facets, boolT printall) {
  int numfacets, numsimplicial, numridges, totneighbors, numneighbors, numcoplanars, numtricoplanars;
  setT *vertices, *vertex_points, *coplanar_points;
  int numpoints= qh->num_points + qh_setsize(qh, qh->other_points);
  vertexT *vertex, **vertexp;
  int vertex_i, vertex_n;
  facetT *facet, **facetp, *neighbor, **neighborp;
  pointT *point, **pointp;

  qh_countfacets(qh, facetlist, facets, printall, &numfacets, &numsimplicial,
      &totneighbors, &numridges, &numcoplanars, &numtricoplanars);  /* sets facet->visitid */
  qh_fprintf(qh, fp, 9248, "%d\n", numpoints);  // 输出点的数量到文件流
  qh_vertexneighbors(qh);  // 计算顶点的邻居关系
  vertices= qh_facetvertices(qh, facetlist, facets, printall);  // 获取包含在facetlist和facets中的所有顶点集合
  vertex_points= qh_settemp(qh, numpoints);  // 分配临时点集合
  coplanar_points= qh_settemp(qh, numpoints);  // 分配临时共面点集合
  qh_setzero(qh, vertex_points, 0, numpoints);  // 清空顶点点集合
  qh_setzero(qh, coplanar_points, 0, numpoints);  // 清空共面点集合
  FOREACHvertex_(vertices)
    qh_point_add(qh, vertex_points, vertex->point, vertex);  // 将顶点和其点添加到点集合中
  FORALLfacet_(facetlist) {
    FOREACHpoint_(facet->coplanarset)
      qh_point_add(qh, coplanar_points, point, facet);  // 将共面集合中的点和facet关联添加到共面点集合中
  }
  FOREACHfacet_(facets) {
    FOREACHpoint_(facet->coplanarset)
      qh_point_add(qh, coplanar_points, point, facet);  // 将facets中共面集合中的点和facet关联添加到共面点集合中
  }
  FOREACHvertex_i_(qh, vertex_points) {
    if (vertex) {  // 如果存在顶点
      numneighbors= qh_setsize(qh, vertex->neighbors);  // 获取顶点邻居的数量
      qh_fprintf(qh, fp, 9249, "%d", numneighbors);  // 输出顶点邻居的数量到文件流
      if (qh->hull_dim == 3)
        qh_order_vertexneighbors(qh, vertex);  // 如果维度为3，对顶点的邻居进行排序
      else if (qh->hull_dim >= 4)
        qsort(SETaddr_(vertex->neighbors, facetT), (size_t)numneighbors,
             sizeof(facetT *), qh_compare_facetvisit);  // 如果维度大于等于4，对顶点邻居进行排序
      FOREACHneighbor_(vertex)
        qh_fprintf(qh, fp, 9250, " %d",
                 neighbor->visitid ? neighbor->visitid - 1 : 0 - neighbor->id);  // 输出邻居的visitid或者-id到文件流
      qh_fprintf(qh, fp, 9251, "\n");  // 输出换行符到文件流
    }else if ((facet= SETelemt_(coplanar_points, vertex_i, facetT)))
      qh_fprintf(qh, fp, 9252, "1 %d\n",
                  facet->visitid ? facet->visitid - 1 : 0 - facet->id);  // 输出共面点的visitid或者-id到文件流
    else
      qh_fprintf(qh, fp, 9253, "0\n");



    // 如果不满足前面的条件，执行以下操作：
    // 使用 qh_fprintf 函数将字符串 "0\n" 写入文件 fp 中，使用格式化标识符 9253
    qh_fprintf(qh, fp, 9253, "0\n");
  }
  // 释放 coplanar_points 集合占用的临时内存空间
  qh_settempfree(qh, &coplanar_points);
  // 释放 vertex_points 集合占用的临时内存空间
  qh_settempfree(qh, &vertex_points);
  // 释放 vertices 集合占用的临时内存空间
  qh_settempfree(qh, &vertices);


这段代码的作用是根据条件执行不同的文件写入操作，并释放多个集合所占用的临时内存空间。
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvoronoi">-</a>

  qh_printvoronoi(qh, fp, format, facetlist, facets, printall )
    print voronoi diagram in 'o' or 'G' format
    for 'o' format
      prints voronoi centers for each facet and for infinity
      for each vertex, lists ids of printed facets or infinity
      assumes facetlist and facets are disjoint
    for 'G' format
      prints an OFF object
      adds a 0 coordinate to center
      prints infinity but does not list in vertices

  see:
    qh_printvdiagram()

  notes:
    if 'o',
      prints a line for each point except "at-infinity"
    if all facets are upperdelaunay,
      reverses lower and upper hull
*/
void qh_printvoronoi(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall) {
  int k, numcenters, numvertices= 0, numneighbors, numinf, vid=1, vertex_i, vertex_n;
  facetT *facet, **facetp, *neighbor, **neighborp;
  setT *vertices;
  vertexT *vertex;
  boolT isLower;
  unsigned int numfacets= (unsigned int)qh->num_facets;

  // 标记维诺图顶点并计算相关信息
  vertices= qh_markvoronoi(qh, facetlist, facets, printall, &isLower, &numcenters);

  // 遍历标记的维诺图顶点
  FOREACHvertex_i_(qh, vertices) {
    if (vertex) {
      numvertices++;
      numneighbors= numinf= 0;

      // 遍历当前顶点的相邻面
      FOREACHneighbor_(vertex) {
        // 检查相邻面是否为无穷远的面
        if (neighbor->visitid == 0)
          numinf= 1;
        else if (neighbor->visitid < numfacets)
          numneighbors++;
      }

      // 如果当前顶点只有无穷远面相邻且没有其他相邻面，则将该顶点置空并减少顶点数目
      if (numinf && !numneighbors) {
        SETelem_(vertices, vertex_i)= NULL;
        numvertices--;
      }
    }
  }

  // 根据格式输出维诺图信息的头部
  if (format == qh_PRINTgeom)
    qh_fprintf(qh, fp, 9254, "{appearance {+edge -face} OFF %d %d 1 # Voronoi centers and cells\n",
                numcenters, numvertices);
  else
    qh_fprintf(qh, fp, 9255, "%d\n%d %d 1\n", qh->hull_dim-1, numcenters, qh_setsize(qh, vertices));

  // 根据不同的格式输出无穷远点信息
  if (format == qh_PRINTgeom) {
    for (k=qh->hull_dim-1; k--; )
      qh_fprintf(qh, fp, 9256, qh_REAL_1, 0.0);
    qh_fprintf(qh, fp, 9257, " 0 # infinity not used\n");
  } else {
    for (k=qh->hull_dim-1; k--; )
      qh_fprintf(qh, fp, 9258, qh_REAL_1, qh_INFINITE);
    qh_fprintf(qh, fp, 9259, "\n");
  }

  // 输出每个facetlist中的面的维诺图信息
  FORALLfacet_(facetlist) {
    if (facet->visitid && facet->visitid < numfacets) {
      if (format == qh_PRINTgeom)
        qh_fprintf(qh, fp, 9260, "# %d f%d\n", vid++, facet->id);
      qh_printcenter(qh, fp, format, NULL, facet);
    }
  }

  // 输出facets集合中的面的维诺图信息
  FOREACHfacet_(facets) {
    if (facet->visitid && facet->visitid < numfacets) {
      if (format == qh_PRINTgeom)
        qh_fprintf(qh, fp, 9261, "# %d f%d\n", vid++, facet->id);
      qh_printcenter(qh, fp, format, NULL, facet);
    }
  }

  // 输出每个标记的维诺图顶点的相邻面信息
  FOREACHvertex_i_(qh, vertices) {
    numneighbors= 0;
    numinf=0;
    // 统计当前顶点的相邻面数和无穷远面数

    FOREACHneighbor_(vertex) {
      if (neighbor->visitid == 0)
        numinf= 1;
      else if (neighbor->visitid < numfacets)
        numneighbors++;
    }
    // 如果顶点只有无穷远面相邻且没有其他相邻面，则将该顶点置空并减少顶点数目
    if (numinf && !numneighbors) {
      SETelem_(vertices, vertex_i)= NULL;
      numvertices--;
    }
  }
}
    # 如果顶点存在，则进行以下处理
    if (vertex) {
      # 如果凸壳维度为3，则按顶点邻居顺序排序
      if (qh->hull_dim == 3)
        qh_order_vertexneighbors(qh, vertex);
      # 如果凸壳维度大于等于4，则使用快速排序对顶点邻居进行排序
      else if (qh->hull_dim >= 4)
        qsort(SETaddr_(vertex->neighbors, facetT),
             (size_t)qh_setsize(qh, vertex->neighbors),
             sizeof(facetT *), qh_compare_facetvisit);
      
      # 遍历顶点的每个邻居
      FOREACHneighbor_(vertex) {
        # 如果邻居的访问标识为0，则设置numinf为1
        if (neighbor->visitid == 0)
          numinf= 1;
        # 否则，如果邻居的访问标识小于numfacets，则增加numneighbors计数
        else if (neighbor->visitid < numfacets)
          numneighbors++;
      }
    }
    
    # 如果输出格式为qh_PRINTgeom
    if (format == qh_PRINTgeom) {
      # 如果顶点存在，则输出顶点的邻居信息
      if (vertex) {
        qh_fprintf(qh, fp, 9262, "%d", numneighbors);
        FOREACHneighbor_(vertex) {
          # 如果邻居的访问标识存在且小于numfacets，则输出邻居的访问标识
          if (neighbor->visitid && neighbor->visitid < numfacets)
            qh_fprintf(qh, fp, 9263, " %d", neighbor->visitid);
        }
        qh_fprintf(qh, fp, 9264, " # p%d(v%d)\n", vertex_i, vertex->id);
      } else
        # 如果顶点不存在，则输出顶点p%d是共面或孤立的信息
        qh_fprintf(qh, fp, 9265, " # p%d is coplanar or isolated\n", vertex_i);
    } else {
      # 如果numinf为真，则增加numneighbors计数
      if (numinf)
        numneighbors++;
      
      # 输出顶点的邻居信息
      qh_fprintf(qh, fp, 9266, "%d", numneighbors);
      if (vertex) {
        FOREACHneighbor_(vertex) {
          # 如果邻居的访问标识为0，并且numinf为真，则输出邻居的访问标识
          if (neighbor->visitid == 0) {
            if (numinf) {
              numinf= 0;
              qh_fprintf(qh, fp, 9267, " %d", neighbor->visitid);
            }
          } else if (neighbor->visitid < numfacets)
            # 否则，如果邻居的访问标识小于numfacets，则输出邻居的访问标识
            qh_fprintf(qh, fp, 9268, " %d", neighbor->visitid);
        }
      }
      qh_fprintf(qh, fp, 9269, "\n");
    }
  }
  
  # 如果输出格式为qh_PRINTgeom，则输出结束大括号
  if (format == qh_PRINTgeom)
    qh_fprintf(qh, fp, 9270, "}\n");
  
  # 释放临时分配的vertices集合
  qh_settempfree(qh, &vertices);
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvnorm">-</a>

  qh_printvnorm(qh, fp, vertex, vertexA, centers, unbounded )
    print one separating plane of the Voronoi diagram for a pair of input sites
    unbounded==True if centers includes vertex-at-infinity

  assumes:
    qh_ASvoronoi and qh_vertexneighbors() already set

  note:
    parameter unbounded is UNUSED by this callback

  see:
    qh_printvdiagram()
    qh_eachvoronoi()
*/
void qh_printvnorm(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded) {
  pointT *normal;
  realT offset;
  int k;
  QHULL_UNUSED(unbounded);

  // 计算顶点 vertex 和 vertexA 之间的法向量及偏移量
  normal= qh_detvnorm(qh, vertex, vertexA, centers, &offset);

  // 打印格式化输出到文件流 fp
  qh_fprintf(qh, fp, 9271, "%d %d %d ",
      2+qh->hull_dim, qh_pointid(qh, vertex->point), qh_pointid(qh, vertexA->point));

  // 打印法向量的每个分量
  for (k=0; k< qh->hull_dim-1; k++)
    qh_fprintf(qh, fp, 9272, qh_REAL_1, normal[k]);

  // 打印偏移量
  qh_fprintf(qh, fp, 9273, qh_REAL_1, offset);

  // 打印换行符
  qh_fprintf(qh, fp, 9274, "\n");
} /* printvnorm */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="printvridge">-</a>

  qh_printvridge(qh, fp, vertex, vertexA, centers, unbounded )
    print one ridge of the Voronoi diagram for a pair of input sites
    unbounded==True if centers includes vertex-at-infinity

  see:
    qh_printvdiagram()

  notes:
    the user may use a different function
    parameter unbounded is UNUSED
*/
void qh_printvridge(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded) {
  facetT *facet, **facetp;
  QHULL_UNUSED(unbounded);

  // 打印格式化输出到文件流 fp
  qh_fprintf(qh, fp, 9275, "%d %d %d", qh_setsize(qh, centers)+2,
       qh_pointid(qh, vertex->point), qh_pointid(qh, vertexA->point));

  // 遍历 centers 中的每个 facet，打印其 visitid
  FOREACHfacet_(centers)
    qh_fprintf(qh, fp, 9276, " %d", facet->visitid);

  // 打印换行符
  qh_fprintf(qh, fp, 9277, "\n");
} /* printvridge */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="projectdim3">-</a>

  qh_projectdim3(qh, source, destination )
    project 2-d 3-d or 4-d point to a 3-d point
    uses qh.DROPdim and qh.hull_dim
    source and destination may be the same

  notes:
    allocate 4 elements to destination just in case
*/
void qh_projectdim3(qhT *qh, pointT *source, pointT *destination) {
  int i,k;

  // 将二维、三维或四维点投影到三维点
  for (k=0, i=0; k < qh->hull_dim; k++) {
    if (qh->hull_dim == 4) {
      // 如果点的维度为四维且不是要丢弃的维度，则复制到目标点中
      if (k != qh->DROPdim)
        destination[i++]= source[k];
    } else if (k == qh->DROPdim)
      // 否则，在要丢弃的维度上填充0
      destination[i++]= 0;
    else
      // 复制点到目标点中
      destination[i++]= source[k];
  }

  // 填充剩余的维度为0
  while (i < 3)
    destination[i++]= 0.0;
} /* projectdim3 */

/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="readfeasible">-</a>

  qh_readfeasible(qh, dim, curline )
    read feasible point from current line and qh.fin

  returns:
    number of lines read from qh.fin
*/
    # 分配内存给 qh.feasible_point 以存储坐标信息
    
    notes:
      # 检查是否存在 qh.HALFspace
      # 假设 dim > 1
    
    see:
      # 参考 qh_setfeasible
/*-<a                             href="qh-io_r.htm#TOC"
  >-------------------------------</a><a name="readpoints">-</a>

  qh_readpoints(qh, numpoints, dimension, ismalloc )
    read points from qh.fin into qh.first_point, qh.num_points
    qh.fin is lines of coordinates, one per vertex, first line number of points
    if 'rbox D4',
      gives message
    if qh.ATinfinity,
      adds point-at-infinity for Delaunay triangulations

  returns:
    number of points, array of point coordinates, dimension, ismalloc True
    if qh.DELAUNAY & !qh.PROJECTinput, projects points to paraboloid
        and clears qh.PROJECTdelaunay
    if qh.HALFspace, reads optional feasible point, reads halfspaces,
        converts to dual.

  for feasible point in "cdd format" in 3-d:
    3 1
    coordinates
    comments
    begin
    n 4 real/integer
    ...
    end

  notes:
    dimension will change in qh_initqhull_globals if qh.PROJECTinput
    uses malloc() since qh_mem not initialized
    QH11012 FIX: qh_readpoints needs rewriting, too long
*/

int qh_readpoints(qhT *qh, int numpoints, int dimension, boolT ismalloc) {
    /* 读取点集合并存储到 qh.fin 中的 qh.first_point, qh.num_points
       qh.fin 包含每个顶点的坐标，第一行是点的数量
       如果是 'rbox D4'，显示消息
       如果 qh.ATinfinity，为 Delaunay 三角化添加无穷远点 */

    /* 返回值:
       点的数量，点坐标数组，维度，ismalloc 为 True
       如果 qh.DELAUNAY & !qh.PROJECTinput，将点投影到抛物面上，并清除 qh.PROJECTdelaunay
       如果 qh.HALFspace，读取可行点（可选），读取半空间，转换为对偶形式 */

    /* 对于 "cdd 格式" 的三维可行点:
       3 1
       坐标
       注释
       开始
       n 4 实数/整数
       ...
       结束 */

    /* 注意:
       如果 qh.PROJECTinput，dimension 将在 qh_initqhull_globals 中更改
       由于 qh_mem 未初始化，使用 malloc()
       QH11012 修复: qh_readpoints 需要重写，太长了 */

    // 声明变量
    boolT isfirst = True;
    int linecount = 0, tokcount = 0;
    const char *s;
    char *t, firstline[qh_MAXfirst + 1];
    coordT *coords, value;

    // 如果没有设置 qh->HALFspace，输出错误消息并退出
    if (!qh->HALFspace) {
        qh_fprintf(qh, qh->ferr, 6070, "qhull input error: feasible point(dim 1 coords) is only valid for halfspace intersection\n");
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }

    // 如果已设置 qh->feasible_string，输出警告信息
    if (qh->feasible_string)
        qh_fprintf(qh, qh->ferr, 7057, "qhull input warning: feasible point(dim 1 coords) overrides 'Hn,n,n' feasible point for halfspace intersection\n");

    // 分配内存以存储可行点坐标
    if (!(qh->feasible_point = (coordT *)qh_malloc((size_t)dimension * sizeof(coordT)))) {
        qh_fprintf(qh, qh->ferr, 6071, "qhull error: insufficient memory for feasible point\n");
        qh_errexit(qh, qh_ERRmem, NULL, NULL);
    }

    // 将可行点坐标数组赋值给 coords
    coords = qh->feasible_point;

    // 读取坐标行，直到结束
    while ((s = (isfirst ? curline : fgets(firstline, qh_MAXfirst, qh->fin)))) {
        // 如果是第一行，标记为 False
        if (isfirst)
            isfirst = False;
        else
            linecount++;  // 增加行数计数器

        // 遍历当前行的每个字符
        while (*s) {
            // 跳过空白字符
            while (isspace(*s))
                s++;

            // 将字符串转换为浮点数
            value = qh_strtod(s, &t);

            // 如果未转换任何内容，跳出循环
            if (s == t)
                break;

            // 更新 s 为下一个位置
            s = t;

            // 将值存入 coords 数组，并增加计数器
            *(coords++) = value;
            tokcount++;

            // 如果达到指定的维度
            if (tokcount == dimension) {
                // 跳过剩余的空白字符
                while (isspace(*s))
                    s++;

                // 再次尝试转换，如果有额外的内容，输出错误消息并退出
                qh_strtod(s, &t);
                if (s != t) {
                    qh_fprintf(qh, qh->ferr, 6072, "qhull input error: coordinates for feasible point do not finish out the line: %s\n", s);
                    qh_errexit(qh, qh_ERRinput, NULL, NULL);
                }
                return linecount;  // 返回行数计数器
            }
        }
    }

    // 如果未能读取到指定的维度数量的坐标，输出错误消息并退出
    qh_fprintf(qh, qh->ferr, 6073, "qhull input error: only %d coordinates.  Could not read %d-d feasible point.\n",
               tokcount, dimension);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);

    return 0;  // 返回 0 表示出错
} /* readfeasible */
  coordT *qh_readpoints(qhT *qh, int *numpoints, int *dimension, boolT *ismalloc) {
    coordT *points, *coords, *infinity= NULL;
    realT paraboloid, maxboloid= -REALmax, value;
    realT *coordp= NULL, *offsetp= NULL, *normalp= NULL;
    char *s= 0, *t, firstline[qh_MAXfirst+1];
    int diminput=0, numinput=0, dimfeasible= 0, newnum, k, tempi;
    int firsttext=0, firstshort=0, firstlong=0, firstpoint=0;
    int tokcount= 0, linecount=0, maxcount, coordcount=0;
    boolT islong, isfirst= True, wasbegin= False;
    boolT isdelaunay= qh->DELAUNAY && !qh->PROJECTinput;

    // 如果是CDD格式的输入，则读取数据直到找到"begin"开头的行
    if (qh->CDDinput) {
      while ((s= fgets(firstline, qh_MAXfirst, qh->fin))) {
        linecount++;
        if (qh->HALFspace && linecount == 1 && isdigit(*s)) {
          // 读取可行解的维度和内容
          dimfeasible= qh_strtol(s, &s);
          while (isspace(*s))
            s++;
          if (qh_strtol(s, &s) == 1)
            linecount += qh_readfeasible(qh, dimfeasible, s);
          else
            dimfeasible= 0;
        } else if (!memcmp(firstline, "begin", (size_t)5) || !memcmp(firstline, "BEGIN", (size_t)5))
          break;
        else if (!*qh->rbox_command)
          // 如果还没有rbox_command，则将当前行内容拼接到rbox_command末尾
          strncat(qh->rbox_command, s, sizeof(qh->rbox_command)-1);
      }
      if (!s) {
        // 如果未找到"begin"行，则报错
        qh_fprintf(qh, qh->ferr, 6074, "qhull input error: missing \"begin\" for cdd-formated input\n");
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
    }

    // 继续读取输入文件，直到找到维度和点数信息
    while (!numinput && (s= fgets(firstline, qh_MAXfirst, qh->fin))) {
      linecount++;
      if (!memcmp(s, "begin", (size_t)5) || !memcmp(s, "BEGIN", (size_t)5))
        wasbegin= True;
      while (*s) {
        while (isspace(*s))
          s++;
        if (!*s)
          break;
        if (!isdigit(*s)) {
          if (!*qh->rbox_command) {
            // 如果还没有rbox_command，则将当前行内容拼接到rbox_command末尾
            strncat(qh->rbox_command, s, sizeof(qh->rbox_command)-1);
            firsttext= linecount;
          }
          break;
        }
        if (!diminput)
          // 读取维度信息
          diminput= qh_strtol(s, &s);
        else {
          // 读取点数信息
          numinput= qh_strtol(s, &s);
          if (numinput == 1 && diminput >= 2 && qh->HALFspace && !qh->CDDinput) {
            // 如果只有一个点，维度大于等于2，并且不是CDD格式，则读取可行解
            linecount += qh_readfeasible(qh, diminput, s); /* checks if ok */
            dimfeasible= diminput;
            diminput= numinput= 0;
          } else
            break;
        }
      }
    }
    if (!s) {
      // 如果未成功读取到维度和点数信息，则报错
      qh_fprintf(qh, qh->ferr, 6075, "qhull input error: short input file.  Did not find dimension and number of points\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }

    // 确保维度大于等于点数，若不是则交换它们
    if (diminput > numinput) {
      tempi= diminput;    /* exchange dim and n, e.g., for cdd input format */
      diminput= numinput;
      numinput= tempi;
    }

    // 检查维度是否大于等于2
    if (diminput < 2) {
      qh_fprintf(qh, qh->ferr, 6220, "qhull input error: dimension %d (first or smaller number) should be at least 2\n",
              diminput);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }

    // 检查点数是否在合理范围内
    if (numinput < 1 || numinput > qh_POINTSmax) {
      qh_fprintf(qh, qh->ferr, 6411, "qhull input error: expecting between 1 and %d points.  Got %d %d-d points\n",
        qh_POINTSmax, numinput, diminput);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    /* 检查是否存在半空间设置，如果存在，则不能与Delaunay或Voronoi选项一起使用 */
    if (isdelaunay && qh->HALFspace) {
        qh_fprintf(qh, qh->ferr, 6037, "qhull option error (qh_readpoints): can not use Delaunay('d') or Voronoi('v') with halfspace intersection('H')\n");
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
        /* 否则会导致内存分配错误，错误信息与 qh_initqhull_globals 中相同 */
    } else if (isdelaunay) {
        /* 设置为非Delaunay投影 */
        qh->PROJECTdelaunay = False;
        /* 根据输入类型设置维度 */
        if (qh->CDDinput)
            *dimension = diminput;
        else
            *dimension = diminput + 1;
        /* 设置点的数量 */
        *numpoints = numinput;
        /* 如果存在无穷点，则数量加一 */
        if (qh->ATinfinity)
            (*numpoints)++;
    } else if (qh->HALFspace) {
        /* 对于半空间，设置维度为输入维度减一 */
        *dimension = diminput - 1;
        /* 设置点的数量 */
        *numpoints = numinput;
        /* 如果维度小于3，则输出错误信息并退出 */
        if (diminput < 3) {
            qh_fprintf(qh, qh->ferr, 6221, "qhull input error: dimension %d (first number, includes offset) should be at least 3 for halfspaces\n",
                       diminput);
            qh_errexit(qh, qh_ERRinput, NULL, NULL);
        }
        /* 如果存在可行点维度，但与当前维度不符合，则输出错误信息并退出 */
        if (dimfeasible) {
            if (dimfeasible != *dimension) {
                qh_fprintf(qh, qh->ferr, 6222, "qhull input error: dimension %d of feasible point is not one less than dimension %d for halfspaces\n",
                           dimfeasible, diminput);
                qh_errexit(qh, qh_ERRinput, NULL, NULL);
            }
        } else
            /* 设置可行点维度 */
            qh_setfeasible(qh, *dimension);
    } else {
        /* 对于其他情况，根据输入类型设置维度 */
        if (qh->CDDinput)
            *dimension = diminput - 1;
        else
            *dimension = diminput;
        /* 设置点的数量 */
        *numpoints = numinput;
    }
    /* 设置法线大小用于跟踪 qh_printpoint */
    qh->normal_size = *dimension * (int)sizeof(coordT);
    /* 如果存在半空间设置，则分配半空间的内存 */
    if (qh->HALFspace) {
        qh->half_space = coordp = (coordT *)qh_malloc((size_t)qh->normal_size + sizeof(coordT));
        /* 如果是 CDD 输入，则设置偏移指针和法线指针 */
        if (qh->CDDinput) {
            offsetp = qh->half_space;
            normalp = offsetp + 1;
        } else {
            normalp = qh->half_space;
            offsetp = normalp + *dimension;
        }
    }
    /* 设置最大行的长度用于跟踪 */
    qh->maxline = diminput * (qh_REALdigits + 5);
    maximize_(qh->maxline, 500);
    /* 分配行的内存空间 */
    qh->line = (char *)qh_malloc((size_t)(qh->maxline + 1) * sizeof(char));
    /* 设置为使用 malloc 分配内存，因为内存尚未设置 */
    *ismalloc = True;
    /* 分配坐标点的内存空间，根据条件 numinput 和 diminput 大于等于2 */
    coords = points = qh->temp_malloc = (coordT *)qh_malloc((size_t)((*numpoints) * (*dimension)) * sizeof(coordT));
    /* 如果分配失败，则输出错误信息并退出 */
    if (!coords || !qh->line || (qh->HALFspace && !qh->half_space)) {
        qh_fprintf(qh, qh->ferr, 6076, "qhull error: insufficient memory to read %d points\n",
                   numinput);
        qh_errexit(qh, qh_ERRmem, NULL, NULL);
    }
    /* 如果是 Delaunay 且存在无穷点，则将无穷点设置为 0 */
    if (isdelaunay && qh->ATinfinity) {
        infinity = points + numinput * (*dimension);
        for (k = (*dimension) - 1; k--; )
            infinity[k] = 0.0;
    }
    /* 设置最大计数为 numinput 乘以 diminput */
    maxcount = numinput * diminput;
    /* 设置 paraboloid 为 0.0 */
    paraboloid = 0.0;
    /* 循环读取每行，根据 isfirst 判断是否为第一行 */
    while ((s = (isfirst ? s : fgets(qh->line, qh->maxline, qh->fin)))) {
    // 如果不是第一行输入
    if (!isfirst) {
      // 增加行计数
      linecount++;
      // 如果当前字符为 'e' 或 'E'
      if (*s == 'e' || *s == 'E') {
        // 如果匹配到 "end" 或 "END"，表示输入可能结束
        if (!memcmp(s, "end", (size_t)3) || !memcmp(s, "END", (size_t)3)) {
          // 如果有输入到 CDDinput，跳出循环
          if (qh->CDDinput )
            break;
          // 如果之前是 "begin"，给出警告信息
          else if (wasbegin)
            qh_fprintf(qh, qh->ferr, 7058, "qhull input warning: the input appears to be in cdd format.  If so, use 'Fd'\n");
        }
      }
    }
    // 重置长行标志为假
    islong= False;
    // 当字符串未结束时进行循环
    while (*s) {
      // 跳过空白字符
      while (isspace(*s))
        s++;
      // 将当前位置的字符串转换为 double 类型的值
      value= qh_strtod(s, &t);
      // 如果未转换任何内容
      if (s == t) {
        // 如果 rbox_command 为空，则将 s 的内容追加到 rbox_command
        if (!*qh->rbox_command)
         strncat(qh->rbox_command, s, sizeof(qh->rbox_command)-1);
        // 如果 s 不为空且未标记过第一个文本行，则标记当前行
        if (*s && !firsttext)
          firsttext= linecount;
        // 如果不是长行且未标记过第一个短行，并且坐标计数非零，则标记当前行
        if (!islong && !firstshort && coordcount)
          firstshort= linecount;
        // 跳出内层循环
        break;
      }
      // 如果尚未标记第一个点，则标记当前行
      if (!firstpoint)
        firstpoint= linecount;
      // 更新 s 的位置到下一个数字的起始位置
      s= t;
      // 如果标记的 token 数量超过最大限制，则继续下一个循环
      if (++tokcount > maxcount)
        continue;
      // 如果是半空间输入模式
      if (qh->HALFspace) {
        // 如果是 CDDinput 格式，则将 value 取反并放入 coordp
        if (qh->CDDinput)
          *(coordp++)= -value; /* both coefficients and offset */
        // 否则直接放入 coordp
        else
          *(coordp++)= value;
      } else {
        // 将 value 放入 coords
        *(coords++)= value;
        // 如果是 CDDinput 格式且坐标计数为零
        if (qh->CDDinput && !coordcount) {
          // 如果 value 不等于 1.0，输出错误信息并退出
          if (value != 1.0) {
            qh_fprintf(qh, qh->ferr, 6077, "qhull input error: for cdd format, point at line %d does not start with '1'\n",
                   linecount);
            qh_errexit(qh, qh_ERRinput, NULL, NULL);
          }
          // 回退 coords 指针
          coords--;
        } else if (isdelaunay) {
          // 更新 paraboloid 的平方和
          paraboloid += value * value;
          // 如果在无限点上，根据输入模式更新 infinity 数组
          if (qh->ATinfinity) {
            if (qh->CDDinput)
              infinity[coordcount-1] += value;
            else
              infinity[coordcount] += value;
          }
        }
      }
      // 更新坐标计数，如果达到输入维度则重置为零
      if (++coordcount == diminput) {
        coordcount= 0;
        // 如果是 Delaunay 模式
        if (isdelaunay) {
          // 将 paraboloid 值放入 coords，更新最大值 maxboloid，重置 paraboloid
          *(coords++)= paraboloid;
          maximize_(maxboloid, paraboloid);
          paraboloid= 0.0;
        } else if (qh->HALFspace) {
          // 如果设置半空间失败，输出错误信息并退出
          if (!qh_sethalfspace(qh, *dimension, coords, &coords, normalp, offsetp, qh->feasible_point)) {
            qh_fprintf(qh, qh->ferr, 8048, "The halfspace was on line %d\n", linecount);
            if (wasbegin)
              qh_fprintf(qh, qh->ferr, 8049, "The input appears to be in cdd format.  If so, you should use option 'Fd'\n");
            qh_errexit(qh, qh_ERRinput, NULL, NULL);
          }
          // 重置 coordp 指针
          coordp= qh->half_space;
        }
        // 跳过空白字符
        while (isspace(*s))
          s++;
        // 如果 s 不为空，则标记为长行，并标记第一个长行
        if (*s) {
          islong= True;
          if (!firstlong)
            firstlong= linecount;
        }
      }
    }
    // 如果不是长行且未标记第一个短行，并且坐标计数非零，则标记当前行
    if (!islong && !firstshort && coordcount)
      firstshort= linecount;
    // 如果不是第一行且 s 到 qh->line 的距离大于最大行限制，输出错误信息并退出
    if (!isfirst && s - qh->line >= qh->maxline) {
      qh_fprintf(qh, qh->ferr, 6078, "qhull input error: line %d contained more than %d characters\n",
              linecount, (int) (s - qh->line));   /* WARN64 */
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    // 标记不是第一行输入
    isfirst= False;
  }
  // 如果 rbox_command 的第一个字符不为空
  if (qh->rbox_command[0])
    qh->rbox_command[strlen(qh->rbox_command)-1]= '\0'; /* remove \n, previous qh_errexit's display command as two lines */
  // 截断字符串末尾的换行符 '\n'，用于修正前一个 qh_errexit 函数的显示命令为两行显示

  if (tokcount != maxcount) {
    // 检查 tokcount 是否不等于 maxcount

    newnum= fmin_(numinput, tokcount/diminput);
    // 计算 newnum，为 numinput 和 tokcount/diminput 中较小的值

    if (qh->ALLOWshort)
      // 如果允许短输入（ALLOWshort 为真），发出警告消息
      qh_fprintf(qh, qh->ferr, 7073, "qhull warning: instead of %d points in %d-d, input contains %d points and %d extra coordinates.\n",
          numinput, diminput, tokcount/diminput, tokcount % diminput);
    else
      // 否则，发出错误消息
      qh_fprintf(qh, qh->ferr, 6410, "qhull error: instead of %d points in %d-d, input contains %d points and %d extra coordinates.\n",
          numinput, diminput, tokcount/diminput, tokcount % diminput);

    if (firsttext)
      // 如果存在 firsttext，记录第一条注释的行号
      qh_fprintf(qh, qh->ferr, 8051, "    Line %d is the first comment.\n", firsttext);

    // 记录第一个数据点的行号
    qh_fprintf(qh, qh->ferr, 8033,   "    Line %d is the first point.\n", firstpoint);

    // 如果存在 firstshort，记录第一条短线的行号
    if (firstshort)
      qh_fprintf(qh, qh->ferr, 8052, "    Line %d is the first short line.\n", firstshort);

    // 如果存在 firstlong，记录第一条长线的行号
    if (firstlong)
      qh_fprintf(qh, qh->ferr, 8053, "    Line %d is the first long line.\n", firstlong);

    // 如果允许短输入，继续处理，发出消息记录处理点数
    if (qh->ALLOWshort)
      qh_fprintf(qh, qh->ferr, 8054, "    Continuing with %d points.\n", newnum);
    else {
      // 否则，发出覆盖选项 'Qa'（允许短输入）的消息，并终止程序
      qh_fprintf(qh, qh->ferr, 8077, "    Override with option 'Qa' (allow-short)\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }

    // 更新 numinput 为 newnum
    numinput= newnum;

    // 如果是 Delaunay 三角剖分且存在无穷点
    if (isdelaunay && qh->ATinfinity) {
      // 调整 infinity 数组
      for (k= tokcount % diminput; k--; )
        infinity[k] -= *(--coords);
      *numpoints= newnum+1;
    } else {
      // 否则，调整 coords 指针和 numpoints 的值
      coords -= tokcount % diminput;
      *numpoints= newnum;
    }
  }

  // 如果是 Delaunay 三角剖分且存在无穷点
  if (isdelaunay && qh->ATinfinity) {
    // 根据 numinput 调整 infinity 数组
    for (k= (*dimension) - 1; k--; )
      infinity[k] /= numinput;

    // 如果 coords 指向 infinity 数组，则调整为指向最后一个元素之前
    if (coords == infinity)
      coords += (*dimension) -1;
    else {
      // 否则，将 infinity 数组中的值复制到 coords 中，并将 coords 向后移动
      for (k=0; k < (*dimension) - 1; k++)
        *(coords++)= infinity[k];
    }

    // 将 coords 指向的位置设为 maxboloid 的 1.1 倍
    *(coords++)= maxboloid * 1.1;
  }

  // 如果 qh->rbox_command 是 "./rbox D4"，发出特定消息
  if (!strcmp(qh->rbox_command, "./rbox D4"))
    qh_fprintf(qh, qh->ferr, 8055, "\n\
/* This function reads and processes a set of points for Qhull.
   If any errors occur, it suggests troubleshooting steps.
   It frees allocated memory before returning. */
void readpoints(qhT *qh, int dim, int numinput, FILE *ferr, pointT *points) {
  /* Free previously allocated memory for qh->line */
  qh_free(qh->line);
  qh->line = NULL;
  
  /* Free qh->half_space if it exists */
  if (qh->half_space) {
    qh_free(qh->half_space);
    qh->half_space = NULL;
  }
  
  /* Reset temporary memory allocation */
  qh->temp_malloc = NULL;
  
  /* Print a trace message indicating the number of points and dimensions read */
  trace1((qh, qh->ferr, 1008, "qh_readpoints: read in %d %d-dimensional points\n",
          numinput, dim));
  
  /* Return the processed points */
  return points;
} /* readpoints */


/*-<a href="qh-io_r.htm#TOC">-</a><a name="setfeasible">-</a>

  qh_setfeasible(qh, dim )
    set qh.feasible_point from qh.feasible_string in "n,n,n" or "n n n" format

  notes:
    "n,n,n" already checked by qh_initflags()
    see qh_readfeasible()
    called only once from qh_new_qhull, otherwise leaks memory
*/
void qh_setfeasible(qhT *qh, int dim) {
  int tokcount = 0;
  char *s;
  coordT *coords, value;

  /* Check if qh->feasible_string exists */
  if (!(s = qh->feasible_string)) {
    qh_fprintf(qh, qh->ferr, 6223, "qhull input error: halfspace intersection needs a feasible point.  Either prepend the input with 1 point or use 'Hn,n,n'.  See manual.\n");
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  
  /* Allocate memory for qh->feasible_point */
  if (!(qh->feasible_point = (pointT *)qh_malloc((size_t)dim * sizeof(coordT)))) {
    qh_fprintf(qh, qh->ferr, 6079, "qhull error: insufficient memory for 'Hn,n,n'\n");
    qh_errexit(qh, qh_ERRmem, NULL, NULL);
  }
  
  coords = qh->feasible_point;
  
  /* Parse and convert coordinates from qh->feasible_string */
  while (*s) {
    value = qh_strtod(s, &s);
    if (++tokcount > dim) {
      qh_fprintf(qh, qh->ferr, 7059, "qhull input warning: more coordinates for 'H%s' than dimension %d\n",
          qh->feasible_string, dim);
      break;
    }
    *(coords++) = value;
    if (*s)
      s++;
  }
  
  /* Fill remaining coordinates with zeros */
  while (++tokcount <= dim)
    *(coords++) = 0.0;
} /* setfeasible */


/*-<a href="qh-io_r.htm#TOC">-</a><a name="skipfacet">-</a>

  qh_skipfacet(qh, facet )
    returns 'True' if this facet is not to be printed

  notes:
    based on the user provided slice thresholds and 'good' specifications
*/
boolT qh_skipfacet(qhT *qh, facetT *facet) {
  facetT *neighbor, **neighborp;

  /* Check printing options */
  if (qh->PRINTneighbors) {
    if (facet->good)
      return !qh->PRINTgood;
    
    /* Check neighbors for 'good' facets */
    FOREACHneighbor_(facet) {
      if (neighbor->good)
        return False;
    }
    
    /* Return True if no 'good' neighbors found */
    return True;
  } else if (qh->PRINTgood)
    /* Return True if facet is not 'good' */
    return !facet->good;
  else if (!facet->normal)
    /* Return True if facet has no normal vector */
    return True;
  
  /* Return result of threshold check for facet normal vector */
  return (!qh_inthresholds(qh, facet->normal, NULL));
} /* skipfacet */


/*-<a href="qh-io_r.htm#TOC">-</a><a name="skipfilename">-</a>

  qh_skipfilename(qh, string )
    returns pointer to character after filename

  notes:
    skips leading spaces
*/
char *qh_skipfilename(qhT *qh, char *string) {
  /* Skip leading spaces */
  while (isspace((unsigned char)*string))
    string++;
  
  /* Return pointer to first non-space character */
  return string;
} /* skipfilename */
    # 如果字符串以空格或换行符结尾
    ends with spacing or eol
    # 如果字符串以单引号（'）或双引号（"）开头，那么它必须以相同类型的引号结尾，跳过转义字符 '\' 或 '\"'
    if starts with ' or " ends with the same, skipping \' or \"
    # 对于 qhull，qh_argv_to_command() 函数只使用双引号
    For qhull, qh_argv_to_command() only uses double quotes
/*
   跳过文件名中的空白字符并返回下一个非空白字符的指针
*/
char *qh_skipfilename(qhT *qh, char *filename) {
  char *s= filename;  /* non-const due to return */  
  char c;

  while (*s && isspace(*s))  // 跳过起始处的空白字符
    s++;
  c= *s++;  // 保存当前字符并移动到下一个字符

  if (c == '\0') {  // 如果起始字符为空，则报错并退出程序
    qh_fprintf(qh, qh->ferr, 6204, "qhull input error: filename expected, none found.\n");
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }

  if (c == '\'' || c == '"') {  // 如果起始字符是单引号或双引号
    while (*s !=c || s[-1] == '\\') {  // 寻找与起始引号匹配的结束引号
      if (!*s) {  // 如果没有找到匹配的引号，则报错并退出程序
        qh_fprintf(qh, qh->ferr, 6203, "qhull input error: missing quote after filename -- %s\n", filename);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
      s++;
    }
    s++;  // 移动到引号之后的下一个字符
  }
  else while (*s && !isspace(*s))  // 如果起始字符不是引号，则继续移动直到遇到空白字符
      s++;

  return s;  // 返回处理后的字符指针，指向文件名的下一个字符
} /* skipfilename */
```