# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\geom2_r.c`

```
/*<html><pre>  -<a                             href="qh-geom_r.htm"
  >-------------------------------</a><a name="TOP">-</a>


   geom2_r.c
   infrequently used geometric routines of qhull

   see qh-geom_r.htm and geom_r.h

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/geom2_r.c#15 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $

   frequently used code goes into geom_r.c
*/

#include "qhull_ra.h"

/*================== functions in alphabetic order ============*/

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="copypoints">-</a>

  qh_copypoints(qh, points, numpoints, dimension )
    return qh_malloc'd copy of points

  notes:
    qh_free the returned points to avoid a memory leak
*/
coordT *qh_copypoints(qhT *qh, coordT *points, int numpoints, int dimension) {
  int size;
  coordT *newpoints;

  size= numpoints * dimension * (int)sizeof(coordT);
  if (!(newpoints= (coordT *)qh_malloc((size_t)size))) {
    qh_fprintf(qh, qh->ferr, 6004, "qhull error: insufficient memory to copy %d points\n",
        numpoints);
    qh_errexit(qh, qh_ERRmem, NULL, NULL);
  }
  memcpy((char *)newpoints, (char *)points, (size_t)size); /* newpoints!=0 by QH6004 */
  return newpoints;
} /* copypoints */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="crossproduct">-</a>

  qh_crossproduct( dim, vecA, vecB, vecC )
    crossproduct of 2 dim vectors
    C= A x B

  notes:
    from Glasner, Graphics Gems I, p. 639
    only defined for dim==3
*/
void qh_crossproduct(int dim, realT vecA[3], realT vecB[3], realT vecC[3]){

  if (dim == 3) {
    vecC[0]=   det2_(vecA[1], vecA[2],
                     vecB[1], vecB[2]);
    vecC[1]= - det2_(vecA[0], vecA[2],
                     vecB[0], vecB[2]);
    vecC[2]=   det2_(vecA[0], vecA[1],
                     vecB[0], vecB[1]);
  }
} /* vcross */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="determinant">-</a>

  qh_determinant(qh, rows, dim, nearzero )
    compute signed determinant of a square matrix
    uses qh.NEARzero to test for degenerate matrices

  returns:
    determinant
    overwrites rows and the matrix
    if dim == 2 or 3
      nearzero iff determinant < qh->NEARzero[dim-1]
      (!quite correct, not critical)
    if dim >= 4
      nearzero iff diagonal[k] < qh->NEARzero[k]
*/
realT qh_determinant(qhT *qh, realT **rows, int dim, boolT *nearzero) {
  realT det=0;
  int i;
  boolT sign= False;

  *nearzero= False;
  if (dim < 2) {
    qh_fprintf(qh, qh->ferr, 6005, "qhull internal error (qh_determinate): only implemented for dimension >= 2\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }else if (dim == 2) {
    det= det2_(rows[0][0], rows[0][1],
                 rows[1][0], rows[1][1]);
    # 如果绝对值小于指定的阈值乘以 NEARzero 的第二个元素，则将 *nearzero 设置为 True
    if (fabs_(det) < 10*qh->NEARzero[1])  /* QH11031 FIX: not really correct, what should this be? */
      *nearzero= True;
  }else if (dim == 3) {
    # 计算 3x3 矩阵的行列式值
    det= det3_(rows[0][0], rows[0][1], rows[0][2],
                 rows[1][0], rows[1][1], rows[1][2],
                 rows[2][0], rows[2][1], rows[2][2]);
    # 如果绝对值小于指定的阈值乘以 NEARzero 的第三个元素，则将 *nearzero 设置为 True
    if (fabs_(det) < 10*qh->NEARzero[2])  /* QH11031 FIX: what should this be?  det 5.5e-12 was flat for qh_maxsimplex of qdelaunay 0,0 27,27 -36,36 -9,63 */
      *nearzero= True;
  }else {
    # 对行列式进行高斯消元处理
    qh_gausselim(qh, rows, dim, dim, &sign, nearzero);  /* if nearzero, diagonal still ok */
    det= 1.0;
    # 计算对角线上元素的乘积，得到行列式的值
    for (i=dim; i--; )
      det *= (rows[i])[i];
    # 如果高斯消元过程中需要交换奇数行，则将行列式值取反
    if (sign)
      det= -det;
  }
  # 返回计算得到的行列式值
  return det;
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detjoggle">-</a>

  qh_detjoggle(qh, points, numpoints, dimension )
    determine default max joggle for point array
      as qh_distround * qh_JOGGLEdefault

  returns:
    initial value for JOGGLEmax from points and REALepsilon

  notes:
    computes DISTround since qh_maxmin not called yet
    if qh->SCALElast, last dimension will be scaled later to MAXwidth

    loop duplicated from qh_maxmin
*/
realT qh_detjoggle(qhT *qh, pointT *points, int numpoints, int dimension) {
  realT abscoord, distround, joggle, maxcoord, mincoord;
  pointT *point, *pointtemp;
  realT maxabs= -REALmax;
  realT sumabs= 0;
  realT maxwidth= 0;
  int k;

  // 如果设置了 roundoff，则使用 DISTround 值作为 distround
  if (qh->SETroundoff)
    distround= qh->DISTround; /* 'En' */
  else {
    // 计算各维度的绝对坐标范围和最大宽度
    for (k = 0; k < dimension; k++) {
      // 如果 SCALElast 为真，并且当前维度是最后一维，则将 abscoord 设置为 maxwidth
      if (qh->SCALElast && k == dimension - 1)
        abscoord= maxwidth;
      // 如果是 DELAUNAY 模式，并且当前维度是最后一维，则使用特定的计算方法
      else if (qh->DELAUNAY && k == dimension - 1) /* will qh_setdelaunay() */
        abscoord= 2 * maxabs * maxabs;  /* may be low by qh->hull_dim/2 */
      else {
        // 计算当前维度上所有点的最大和最小坐标
        maxcoord= -REALmax;
        mincoord= REALmax;
        FORALLpoint_(qh, points, numpoints) {
          maximize_(maxcoord, point[k]);
          minimize_(mincoord, point[k]);
        }
        // 更新最大宽度和绝对坐标
        maximize_(maxwidth, maxcoord - mincoord);
        abscoord= fmax_(maxcoord, -mincoord);
      }
      // 累加绝对坐标的和并更新最大绝对坐标
      sumabs += abscoord;
      maximize_(maxabs, abscoord);
    } /* for k */
    // 计算最终的 distround
    distround= qh_distround(qh, qh->hull_dim, maxabs, sumabs);
  }
  // 计算 joggle 作为 distround 乘以 JOGGLEdefault
  joggle= distround * qh_JOGGLEdefault;
  // 确保 joggle 至少是 REALepsilon 乘以 JOGGLEdefault
  maximize_(joggle, REALepsilon * qh_JOGGLEdefault);
  // 打印跟踪信息并返回 joggle 的值
  trace2((qh, qh->ferr, 2001, "qh_detjoggle: joggle=%2.2g maxwidth=%2.2g\n", joggle, maxwidth));
  return joggle;
} /* detjoggle */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detmaxoutside">-</a>

  qh_detmaxoutside(qh);
    determine qh.MAXoutside target for qh_RATIO... tests of distance
    updates option '_max-outside'

  notes:
    called from qh_addpoint and qh_detroundoff
    accounts for qh.ONEmerge, qh.DISTround, qh.MINoutside ('Wn'), qh.max_outside
    see qh_maxout for qh.max_outside with qh.DISTround
*/

void qh_detmaxoutside(qhT *qh) {
  realT maxoutside;

  // 计算 MAXoutside 的目标值，考虑到 ONEmerge、DISTround、MINoutside 和 max_outside
  maxoutside= fmax_(qh->max_outside, qh->ONEmerge + qh->DISTround);
  maximize_(maxoutside, qh->MINoutside);
  // 更新 qh->MAXoutside 的值，并打印跟踪信息
  trace3((qh, qh->ferr, 3056, "qh_detmaxoutside: MAXoutside %2.2g from qh.max_outside %2.2g, ONEmerge %2.2g, MINoutside %2.2g, DISTround %2.2g\n",
      qh->MAXoutside, qh->max_outside, qh->ONEmerge, qh->MINoutside, qh->DISTround));
} /* detmaxoutside */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detroundoff">-</a>

  qh_detroundoff(qh)
    determine maximum roundoff errors from
      REALepsilon, REALmax, REALmin, qh.hull_dim, qh.MAXabs_coord,
      qh.MAXsumcoord, qh.MAXwidth, qh.MINdenom_1

*/
    accounts for qh.SETroundoff, qh.RANDOMdist, qh->MERGEexact
      qh.premerge_cos, qh.postmerge_cos, qh.premerge_centrum,
      qh.postmerge_centrum, qh.MINoutside,
      qh_RATIOnearinside, qh_COPLANARratio, qh_WIDEcoplanar


    # 设置和计算几何算法中的一些参数和选项，包括舍入误差、随机分布、精确合并等
    设置 qh.SETroundoff, qh.RANDOMdist, qh->MERGEexact
    设置 qh.premerge_cos, qh.postmerge_cos, qh.premerge_centrum,
         qh.postmerge_centrum, qh.MINoutside,
         qh_RATIOnearinside, qh_COPLANARratio, qh_WIDEcoplanar



  returns:
    sets qh.DISTround, etc. (see below)
    appends precision constants to qh.qhull_options


  # 返回：
  #   设置 qh.DISTround 等变量（详见下文）
  #   将精度常量附加到 qh.qhull_options



  see:
    qh_maxmin() for qh.NEARzero


  # 参见：
  #   使用 qh_maxmin() 来处理 qh.NEARzero 的情况



  design:
    determine qh.DISTround for distance computations
    determine minimum denominators for qh_divzero
    determine qh.ANGLEround for angle computations
    adjust qh.premerge_cos,... for roundoff error
    determine qh.ONEmerge for maximum error due to a single merge
    determine qh.NEARinside, qh.MAXcoplanar, qh.MINvisible,
      qh.MINoutside, qh.WIDEfacet
    initialize qh.max_vertex and qh.minvertex


  # 设计：
  #   确定用于距离计算的 qh.DISTround
  #   确定用于 qh_divzero 的最小分母
  #   确定用于角度计算的 qh.ANGLEround
  #   调整 qh.premerge_cos,... 以处理舍入误差
  #   确定由单次合并引起的最大误差 qh.ONEmerge
  #   确定 qh.NEARinside, qh.MAXcoplanar, qh.MINvisible,
  #         qh.MINoutside, qh.WIDEfacet
  #   初始化 qh.max_vertex 和 qh.minvertex
/*
void qh_detroundoff(qhT *qh) {
*/
// 设置_max-width选项，如果未设置roundoff选项，则计算并设置DISTround
qh_option(qh, "_max-width", NULL, &qh->MAXwidth);
if (!qh->SETroundoff) {
    // 计算并设置DISTround
    qh->DISTround= qh_distround(qh, qh->hull_dim, qh->MAXabs_coord, qh->MAXsumcoord);
    // 设置Error-roundoff选项
    qh_option(qh, "Error-roundoff", NULL, &qh->DISTround);
}
// 计算并设置MINdenom和相关变量
qh->MINdenom= qh->MINdenom_1 * qh->MAXabs_coord;
qh->MINdenom_1_2= sqrt(qh->MINdenom_1 * qh->hull_dim) ;  /* if will be normalized */
qh->MINdenom_2= qh->MINdenom_1_2 * qh->MAXabs_coord;
                                              /* for inner product */
// 设置ANGLEround为1.01倍hull_dim乘以REALepsilon
qh->ANGLEround= 1.01 * qh->hull_dim * REALepsilon;
// 如果RANDOMdist为真，则增加ANGLEround
if (qh->RANDOMdist) {
    qh->ANGLEround += qh->RANDOMfactor;
    // 跟踪信息输出到日志
    trace4((qh, qh->ferr, 4096, "qh_detroundoff: increase qh.ANGLEround by option 'R%2.2g'\n", qh->RANDOMfactor));
}
// 如果premerge_cos小于REALmax的一半，则减去ANGLEround，并设置Angle-premerge-with-random选项
if (qh->premerge_cos < REALmax/2) {
    qh->premerge_cos -= qh->ANGLEround;
    if (qh->RANDOMdist)
        qh_option(qh, "Angle-premerge-with-random", NULL, &qh->premerge_cos);
}
// 如果postmerge_cos小于REALmax的一半，则减去ANGLEround，并设置Angle-postmerge-with-random选项
if (qh->postmerge_cos < REALmax/2) {
    qh->postmerge_cos -= qh->ANGLEround;
    if (qh->RANDOMdist)
        qh_option(qh, "Angle-postmerge-with-random", NULL, &qh->postmerge_cos);
}
// 根据DISTround调整premerge_centrum和postmerge_centrum
qh->premerge_centrum += 2 * qh->DISTround;    /*2 for centrum and distplane()*/
qh->postmerge_centrum += 2 * qh->DISTround;
// 如果RANDOMdist为真且MERGEexact或PREmerge为真，则设置Centrum-premerge-with-random选项
if (qh->RANDOMdist && (qh->MERGEexact || qh->PREmerge))
    qh_option(qh, "Centrum-premerge-with-random", NULL, &qh->premerge_centrum);
// 如果RANDOMdist为真且POSTmerge为真，则设置Centrum-postmerge-with-random选项
if (qh->RANDOMdist && qh->POSTmerge)
    qh_option(qh, "Centrum-postmerge-with-random", NULL, &qh->postmerge_centrum);
// 计算ONEmerge，用于简并单形面的最大顶点偏移
{
    realT maxangle= 1.0, maxrho;

    // 计算最小的premerge_cos和postmerge_cos
    minimize_(maxangle, qh->premerge_cos);
    minimize_(maxangle, qh->postmerge_cos);
    // 计算ONEmerge的值，考虑到顶点到超平面的距离和DISTround
    qh->ONEmerge= sqrt((realT)qh->hull_dim) * qh->MAXwidth *
      sqrt(1.0 - maxangle * maxangle) + qh->DISTround;
    // 计算maxrho，用于最大化ONEmerge
    maxrho= qh->hull_dim * qh->premerge_centrum + qh->DISTround;
    maximize_(qh->ONEmerge, maxrho);
    maxrho= qh->hull_dim * qh->postmerge_centrum + qh->DISTround;
    maximize_(qh->ONEmerge, maxrho);
    // 如果正在进行合并操作，则设置_one-merge选项
    if (qh->MERGING)
        qh_option(qh, "_one-merge", NULL, &qh->ONEmerge);
}
// 设置NEARinside为ONEmerge乘以RATIOnearinside，仅在KEEPnearinside为真时使用
qh->NEARinside= qh->ONEmerge * qh_RATIOnearinside; /* only used if qh->KEEPnearinside */
// 如果JOGGLEmax小于REALmax的一半，并且KEEPcoplanar或KEEPinside为真，则调整NEARinside用于joggle
if (qh->JOGGLEmax < REALmax/2 && (qh->KEEPcoplanar || qh->KEEPinside)) {
    realT maxdist;             /* adjust qh.NEARinside for joggle */
    qh->KEEPnearinside= True;
    // 计算maxdist，用于调整NEARinside以支持joggle
    maxdist= sqrt((realT)qh->hull_dim) * qh->JOGGLEmax + qh->DISTround;
    maxdist= 2*maxdist;        /* vertex and coplanar point can joggle in opposite directions */
    // 最大化NEARinside，必须与qh_nearcoplanar()保持一致
    maximize_(qh->NEARinside, maxdist);
}
// 如果KEEPnearinside为真，则设置_near-inside选项
if (qh->KEEPnearinside)
    qh_option(qh, "_near-inside", NULL, &qh->NEARinside);
// 如果JOGGLEmax小于DISTround，则...
    # 使用 qh_fprintf 函数向 qh->ferr 文件中写入错误消息，编号为 6006，格式化字符串包含 'QJn' 的 joggle 值和 DISTround 的值
    qh_fprintf(qh, qh->ferr, 6006, "qhull option error: the joggle for 'QJn', %.2g, is below roundoff for distance computations, %.2g\n",
         qh->JOGGLEmax, qh->DISTround);
    # 使用 qh_errexit 函数终止程序执行，指示输入错误
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  
  # 如果 MINvisible 大于 REALmax 的一半
  if (qh->MINvisible > REALmax/2) {
    # 如果 MERGING 选项未开启，将 MINvisible 设置为 DISTround 的值
    if (!qh->MERGING)
      qh->MINvisible= qh->DISTround;
    # 如果 hull_dim 小于等于 3，将 MINvisible 设置为 premerge_centrum 的值
    else if (qh->hull_dim <= 3)
      qh->MINvisible= qh->premerge_centrum;
    # 否则，将 MINvisible 设置为 qh_COPLANARratio 乘以 premerge_centrum 的值
    else
      qh->MINvisible= qh_COPLANARratio * qh->premerge_centrum;
    # 如果 APPROXhull 选项开启，并且 MINvisible 大于 MINoutside 的值，则将 MINvisible 设置为 MINoutside 的值
    if (qh->APPROXhull && qh->MINvisible > qh->MINoutside)
      qh->MINvisible= qh->MINoutside;
    # 使用 qh_option 函数更新 "Visible-distance" 选项值为 MINvisible
    qh_option(qh, "Visible-distance", NULL, &qh->MINvisible);
  }
  
  # 如果 MAXcoplanar 大于 REALmax 的一半
  if (qh->MAXcoplanar > REALmax/2) {
    # 将 MAXcoplanar 设置为 MINvisible 的值
    qh->MAXcoplanar= qh->MINvisible;
    # 使用 qh_option 函数更新 "U-max-coplanar" 选项值为 MAXcoplanar
    qh_option(qh, "U-max-coplanar", NULL, &qh->MAXcoplanar);
  }
  
  # 如果 APPROXhull 选项未开启
  if (!qh->APPROXhull) {             /* user may specify qh->MINoutside */
    # 将 MINoutside 设置为 2 倍的 MINvisible
    qh->MINoutside= 2 * qh->MINvisible;
    # 如果 premerge_cos 小于 REALmax 的一半，使用 maximize_ 函数将 MINoutside 更新为 (1 - qh->premerge_cos) * qh->MAXabs_coord 和当前 MINoutside 中的较大值
    if (qh->premerge_cos < REALmax/2)
      maximize_(qh->MINoutside, (1- qh->premerge_cos) * qh->MAXabs_coord);
    # 使用 qh_option 函数更新 "Width-outside" 选项值为 MINoutside
    qh_option(qh, "Width-outside", NULL, &qh->MINoutside);
  }
  
  # 将 WIDEfacet 设置为 MINoutside 的值
  qh->WIDEfacet= qh->MINoutside;
  # 使用 maximize_ 函数将 WIDEfacet 更新为 qh_WIDEcoplanar 乘以 MAXcoplanar 和 MINvisible 中的较大值
  maximize_(qh->WIDEfacet, qh_WIDEcoplanar * qh->MAXcoplanar);
  maximize_(qh->WIDEfacet, qh_WIDEcoplanar * qh->MINvisible);
  # 使用 qh_option 函数更新 "_wide-facet" 选项值为 WIDEfacet
  qh_option(qh, "_wide-facet", NULL, &qh->WIDEfacet);
  
  # 如果 MINvisible 大于 MINoutside 加上 3 倍的 REALepsilon，并且 BESToutside 和 FORCEoutput 选项均未开启
  if (qh->MINvisible > qh->MINoutside + 3 * REALepsilon
  && !qh->BESToutside && !qh->FORCEoutput)
    # 使用 qh_fprintf 函数向 qh->ferr 文件中写入警告消息，编号为 7001，格式化字符串包含 MINvisible 和 MINoutside 的值
    qh_fprintf(qh, qh->ferr, 7001, "qhull input warning: minimum visibility V%.2g is greater than \nminimum outside W%.2g.  Flipped facets are likely.\n",
             qh->MINvisible, qh->MINoutside);
  
  # 将 max_vertex 设置为 DISTround 的值
  qh->max_vertex= qh->DISTround;
  # 将 min_vertex 设置为 -DISTround 的值
  qh->min_vertex= -qh->DISTround;
  
  # 调用 qh_detmaxoutside 函数计算外部最大值
  qh_detmaxoutside(qh);
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detroundoff">-</a>

  qh_detroundoff(qh, dimension, maxabs, maxsumabs )
    compute maximum round-off error for a determinant computation

  returns:
    max roundoff error for qh.REALepsilon

  notes:
    calculate roundoff error according to Golub & van Loan, 1983, Lemma 3.2-1, "Rounding Errors"

  see:
    qh_detsimplex in geom_r.c
*/
realT qh_detroundoff(qhT *qh, int dimension, realT *maxabs, realT *maxsumabs) {
  realT epsilon= qh->REALepsilon;
  realT detround, epsilon2, one= 1.0;
  int dim2= dimension / 2;

  if (qh->RANDOMdist) {
    epsilon2= 2 * epsilon;
    detround= 4 * qh_divzero(epsilon2, one - epsilon2);  /* Joggle through sqrt and sum */
  }else {
    detround= epsilon * qh_divzero(one, sqrt(one - qh_divzero(epsilon, one)));  /* Joggle through max and maxsum */
  }
  trace2((qh, qh->ferr, 2046, "qh_detroundoff: detround=%2.2g maxabs %2.2g maxsumabs %2.2g\n",
          detround, *maxabs, *maxsumabs));
  return detround;
} /* detroundoff */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detsimplex">-</a>

  qh_detsimplex(qh, apex, points, dim, nearzero )
    compute determinant of a simplex with point apex and base points

  returns:
     signed determinant and nearzero from qh_determinant

  notes:
     called by qh_maxsimplex and qh_initialvertices
     uses qh.gm_matrix/qh.gm_row (assumes they're big enough)

  design:
    construct qm_matrix by subtracting apex from points
    compute determinate
*/
realT qh_detsimplex(qhT *qh, pointT *apex, setT *points, int dim, boolT *nearzero) {
  pointT *coorda, *coordp, *gmcoord, *point, **pointp;
  coordT **rows;
  int k,  i=0;
  realT det;

  zinc_(Zdetsimplex);
  gmcoord= qh->gm_matrix;
  rows= qh->gm_row;
  FOREACHpoint_(points) {
    if (i == dim)
      break;
    rows[i++]= gmcoord;
    coordp= point;
    coorda= apex;
    for (k=dim; k--; )
      *(gmcoord++)= *coordp++ - *coorda++;
  }
  if (i < dim) {
    qh_fprintf(qh, qh->ferr, 6007, "qhull internal error (qh_detsimplex): #points %d < dimension %d\n",
               i, dim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  det= qh_determinant(qh, rows, dim, nearzero);
  trace2((qh, qh->ferr, 2002, "qh_detsimplex: det=%2.2g for point p%d, dim %d, nearzero? %d\n",
          det, qh_pointid(qh, apex), dim, *nearzero));
  return det;
} /* detsimplex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="distnorm">-</a>

  qh_distnorm( dim, point, normal, offset )
    return distance from point to hyperplane at normal/offset

  returns:
    dist

  notes:
    dist > 0 if point is outside of hyperplane

  see:
    qh_distplane in geom_r.c
*/
realT qh_distnorm(int dim, pointT *point, pointT *normal, realT *offsetp) {
  coordT *normalp= normal, *coordp= point;
  realT dist;
  int k;

  dist= *offsetp;
  for (k=dim; k--; )
    dist += *(coordp++) * *(normalp++);
  return dist;
} /* distnorm */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="distround">-</a>

  qh_distround(qh, dimension, maxabs, maxsumabs )
    compute maximum round-off error for a distance computation
      to a normalized hyperplane
    maxabs is the maximum absolute value of a coordinate
    maxsumabs is the maximum possible sum of absolute coordinate values
    if qh.RANDOMdist ('Qr'), adjusts qh_distround

  returns:
    max dist round for qh.REALepsilon and qh.RANDOMdist

  notes:
    calculate roundoff error according to Golub & van Loan, 1983, Lemma 3.2-1, "Rounding Errors"
    use sqrt(dim) since one vector is normalized
      or use maxsumabs since one vector is < 1
*/
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="facetarea">-</a>

  qh_facetarea(qh, facet )
    return area for a facet

  notes:
    if non-simplicial,
      uses centrum to triangulate facet and sums the projected areas.
    if (qh->DELAUNAY),
      computes projected area instead for last coordinate
    assumes facet->normal exists
    projecting tricoplanar facets to the hyperplane does not appear to make a difference

  design:
    if simplicial
      compute area
    else
      for each ridge
        compute area from centrum to ridge
    negate area if upper Delaunay facet
*/
realT qh_facetarea(qhT *qh, facetT *facet) {
  vertexT *apex;
  pointT *centrum;
  realT area= 0.0;
  ridgeT *ridge, **ridgep;

  if (facet->simplicial) {
    apex= SETfirstt_(facet->vertices, vertexT);
    // 计算简单面的面积
    area= qh_facetarea_simplex(qh, qh->hull_dim, apex->point, facet->vertices,
                    apex, facet->toporient, facet->normal, &facet->offset);
  }else {
    if (qh->CENTERtype == qh_AScentrum)
      centrum= facet->center;
    else
      // 获取面的重心
      centrum= qh_getcentrum(qh, facet);
    // 遍历 facet 的每个 ridge
    FOREACHridge_(facet->ridges)
      // 计算 facet 的面积并累加到 area
      area += qh_facetarea_simplex(qh, qh->hull_dim, centrum, ridge->vertices,
                 NULL, (boolT)(ridge->top == facet),  facet->normal, &facet->offset);
    // 如果 CENTERtype 不是 qh_AScentrum，则释放 centrum 所占内存
    if (qh->CENTERtype != qh_AScentrum)
      qh_memfree(qh, centrum, qh->normal_size);
  }
  // 如果 facet 是上半 Delaunay 三角形且开启了 DELAUNAY 模式，则取反 area
  if (facet->upperdelaunay && qh->DELAUNAY)
    area= -area;  /* the normal should be [0,...,1] */
  // 打印日志记录 facet 的面积计算结果
  trace4((qh, qh->ferr, 4009, "qh_facetarea: f%d area %2.2g\n", facet->id, area));
  // 返回计算得到的 facet 面积
  return area;
/* facetarea_simplex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="facetarea_simplex">-</a>

  qh_facetarea_simplex(qh, dim, apex, vertices, notvertex, toporient, normal, offset )
    return area for a simplex defined by
      an apex, a base of vertices, an orientation, and a unit normal
    if simplicial or tricoplanar facet,
      notvertex is defined and it is skipped in vertices

  returns:
    computes area of simplex projected to plane [normal,offset]
    returns 0 if vertex too far below plane (qh->WIDEfacet)
      vertex can't be apex of tricoplanar facet

  notes:
    if (qh->DELAUNAY),
      computes projected area instead for last coordinate
    uses qh->gm_matrix/gm_row and qh->hull_dim
    helper function for qh_facetarea

  design:
    if Notvertex
      translate simplex to apex
    else
      project simplex to normal/offset
      translate simplex to apex
    if Delaunay
      set last row/column to 0 with -1 on diagonal
    else
      set last row to Normal
    compute determinate
    scale and flip sign for area
*/
realT qh_facetarea_simplex(qhT *qh, int dim, coordT *apex, setT *vertices,
        vertexT *notvertex,  boolT toporient, coordT *normal, realT *offset) {
  pointT *coorda, *coordp, *gmcoord;
  coordT **rows, *normalp;
  int k,  i=0;
  realT area, dist;
  vertexT *vertex, **vertexp;
  boolT nearzero;

  // 获取 Qhull 对象中的 gm_matrix 和 gm_row
  gmcoord= qh->gm_matrix;
  rows= qh->gm_row;

  // 遍历所有顶点
  FOREACHvertex_(vertices) {
    // 如果当前顶点是 notvertex，则跳过
    if (vertex == notvertex)
      continue;

    // 将当前行设置为 gm_matrix 的起始点
    rows[i++]= gmcoord;
    coorda= apex;
    coordp= vertex->point;
    normalp= normal;

    // 如果存在 notvertex，将简单形式转换到 apex
    if (notvertex) {
      for (k=dim; k--; )
        *(gmcoord++)= *coordp++ - *coorda++;
    }else {
      // 否则，将简单形式投影到 normal/offset，并转换到 apex
      dist= *offset;
      for (k=dim; k--; )
        dist += *coordp++ * *normalp++;
      // 如果顶点远离平面，返回 0
      if (dist < -qh->WIDEfacet) {
        zinc_(Znoarea);
        return 0.0;
      }
      coordp= vertex->point;
      normalp= normal;
      for (k=dim; k--; )
        *(gmcoord++)= (*coordp++ - dist * *normalp++) - *coorda++;
    }
  }

  // 确保行数与维度匹配
  if (i != dim-1) {
    qh_fprintf(qh, qh->ferr, 6008, "qhull internal error (qh_facetarea_simplex): #points %d != dim %d -1\n",
               i, dim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }

  // 将最后一行设置为 gm_matrix 的当前点
  rows[i]= gmcoord;

  // 如果是 Delaunay 结构，将最后一行/列设为 0，对角线设为 -1
  if (qh->DELAUNAY) {
    for (i=0; i < dim-1; i++)
      rows[i][dim-1]= 0.0;
    for (k=dim; k--; )
      *(gmcoord++)= 0.0;
    rows[dim-1][dim-1]= -1.0;
  }else {
    // 否则，将最后一行设为 Normal 向量
    normalp= normal;
    for (k=dim; k--; )
      *(gmcoord++)= *normalp++;
  }

  // 计算行列式
  zinc_(Zdetfacetarea);
  area= qh_determinant(qh, rows, dim, &nearzero);

  // 根据 toporient 确定面积的正负，并乘以 AREAfactor
  if (toporient)
    area= -area;
  area *= qh->AREAfactor;

  // 跟踪输出面积计算结果
  trace4((qh, qh->ferr, 4010, "qh_facetarea_simplex: area=%2.2g for point p%d, toporient %d, nearzero? %d\n",
          area, qh_pointid(qh, apex), toporient, nearzero));

  // 返回计算得到的面积
  return area;
} /* facetarea_simplex */
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="facetcenter">-</a>

  qh_facetcenter(qh, vertices )
    return Voronoi center (Voronoi vertex) for a facet's vertices

  returns:
    return temporary point equal to the center

  see:
    qh_voronoi_center()
*/
pointT *qh_facetcenter(qhT *qh, setT *vertices) {
  // 创建临时集合以保存顶点的点
  setT *points= qh_settemp(qh, qh_setsize(qh, vertices));
  vertexT *vertex, **vertexp;
  pointT *center;

  // 遍历顶点集合，将每个顶点的点添加到临时集合中
  FOREACHvertex_(vertices)
    qh_setappend(qh, &points, vertex->point);
  
  // 计算并返回 Voronoi 中心点，即以顶点集合为基础计算的中心点
  center= qh_voronoi_center(qh, qh->hull_dim-1, points);
  
  // 释放临时集合的内存
  qh_settempfree(qh, &points);
  
  // 返回计算得到的中心点
  return center;
} /* facetcenter */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="findgooddist">-</a>

  qh_findgooddist(qh, point, facetA, dist, facetlist )
    find best good facet visible for point from facetA
    assumes facetA is visible from point

  returns:
    best facet, i.e., good facet that is furthest from point
      distance to best facet
      NULL if none

    moves good, visible facets (and some other visible facets)
      to end of qh->facet_list

  notes:
    uses qh->visit_id

  design:
    initialize bestfacet if facetA is good
    move facetA to end of facetlist
    for each facet on facetlist
      for each unvisited neighbor of facet
        move visible neighbors to end of facetlist
        update best good neighbor
        if no good neighbors, update best facet
*/
facetT *qh_findgooddist(qhT *qh, pointT *point, facetT *facetA, realT *distp,
               facetT **facetlist) {
  realT bestdist= -REALmax, dist;
  facetT *neighbor, **neighborp, *bestfacet=NULL, *facet;
  boolT goodseen= False;

  // 如果 facetA 是好的，则计算到该面的距离，并标记为已见过的好面
  if (facetA->good) {
    zzinc_(Zcheckpart);  /* calls from check_bestdist occur after print stats */
    qh_distplane(qh, point, facetA, &bestdist);
    bestfacet= facetA;
    goodseen= True;
  }
  
  // 从 facet_list 中移除 facetA，并将其移到列表的末尾
  qh_removefacet(qh, facetA);
  qh_appendfacet(qh, facetA);
  *facetlist= facetA;
  facetA->visitid= ++qh->visit_id;
  
  // 遍历 facet_list 中的每个面
  FORALLfacet_(*facetlist) {
    // 遍历当前面的每个邻居面
    FOREACHneighbor_(facet) {
      // 如果邻居面已经被访问过，则跳过
      if (neighbor->visitid == qh->visit_id)
        continue;
      
      // 标记当前邻居面为已访问过
      neighbor->visitid= qh->visit_id;
      
      // 如果已经找到好的邻居面，并且当前邻居面不是好的，则跳过
      if (goodseen && !neighbor->good)
        continue;
      
      // 计算点到当前邻居面的距离
      zzinc_(Zcheckpart);
      qh_distplane(qh, point, neighbor, &dist);
      
      // 如果距离大于0，则将当前邻居面从 facet_list 中移除并移到末尾
      if (dist > 0) {
        qh_removefacet(qh, neighbor);
        qh_appendfacet(qh, neighbor);
        
        // 如果当前邻居面是好的，则更新最佳邻居面
        if (neighbor->good) {
          goodseen= True;
          if (dist > bestdist) {
            bestdist= dist;
            bestfacet= neighbor;
          }
        }
      }
    }
  }
  
  // 如果找到了最佳邻居面，则更新距离并返回该面
  if (bestfacet) {
    *distp= bestdist;
    trace2((qh, qh->ferr, 2003, "qh_findgooddist: p%d is %2.2g above good facet f%d\n",
      qh_pointid(qh, point), bestdist, bestfacet->id));
    return bestfacet;
  }
  
  // 如果没有找到最佳邻居面，则记录日志并返回NULL
  trace4((qh, qh->ferr, 4011, "qh_findgooddist: no good facet for p%d above f%d\n",
      qh_pointid(qh, point), facetA->id));
  return NULL;
}
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="furthestnewvertex">-</a>

  qh_furthestnewvertex(qh, unvisited, facet, &maxdist )
    return furthest unvisited, new vertex to a facet

  return:
    NULL if no vertex is above facet
    maxdist to facet
    updates v.visitid

  notes:
    Ignores vertices in facetB
    Does not change qh.vertex_visit.  Use in conjunction with qh_furthestvertex
*/
vertexT *qh_furthestnewvertex(qhT *qh, unsigned int unvisited, facetT *facet, realT *maxdistp /* qh.newvertex_list */) {
  vertexT *maxvertex= NULL, *vertex;
  coordT dist, maxdist= 0.0;

  FORALLvertex_(qh->newvertex_list) {  // 遍历 qh->newvertex_list 中的所有顶点
    if (vertex->newfacet && vertex->visitid <= unvisited) {  // 如果顶点未被访问过且在未访问的范围内
      vertex->visitid= qh->vertex_visit;  // 更新顶点的访问标识为当前访问标识
      qh_distplane(qh, vertex->point, facet, &dist);  // 计算顶点到平面 facet 的距离
      if (dist > maxdist) {  // 如果距离大于当前最大距离
        maxdist= dist;  // 更新最大距离
        maxvertex= vertex;  // 更新最远的顶点
      }
    }
  }
  trace4((qh, qh->ferr, 4085, "qh_furthestnewvertex: v%d dist %2.2g is furthest new vertex for f%d\n",
    getid_(maxvertex), maxdist, facet->id));  // 记录跟踪信息
  *maxdistp= maxdist;  // 将最大距离存入 maxdistp 指向的位置
  return maxvertex;  // 返回最远的顶点
} /* furthestnewvertex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="furthestvertex">-</a>

  qh_furthestvertex(qh, facetA, facetB, &maxdist, &mindist )
    return furthest vertex in facetA from facetB, or NULL if none

  return:
    maxdist and mindist to facetB or 0.0 if none
    updates qh.vertex_visit

  notes:
    Ignores vertices in facetB
*/
vertexT *qh_furthestvertex(qhT *qh, facetT *facetA, facetT *facetB, realT *maxdistp, realT *mindistp) {
  vertexT *maxvertex= NULL, *vertex, **vertexp;
  coordT dist, maxdist= -REALmax, mindist= REALmax;

  qh->vertex_visit++;  // 增加顶点访问计数
  FOREACHvertex_(facetB->vertices)  // 遍历 facetB 的所有顶点
    vertex->visitid= qh->vertex_visit;  // 将顶点的访问标识设置为当前访问计数
  FOREACHvertex_(facetA->vertices) {  // 遍历 facetA 的所有顶点
    if (vertex->visitid != qh->vertex_visit) {  // 如果顶点未被访问过
      vertex->visitid= qh->vertex_visit;  // 更新顶点的访问标识
      zzinc_(Zvertextests);  // 增加顶点测试的计数
      qh_distplane(qh, vertex->point, facetB, &dist);  // 计算顶点到平面 facetB 的距离
      if (!maxvertex) {  // 如果还没有最远的顶点
        maxdist= dist;  // 设置最大距离为当前距离
        mindist= dist;  // 设置最小距离为当前距离
        maxvertex= vertex;  // 更新最远的顶点
      } else if (dist > maxdist) {  // 如果当前距离大于最大距离
        maxdist= dist;  // 更新最大距离
        maxvertex= vertex;  // 更新最远的顶点
      } else if (dist < mindist)  // 如果当前距离小于最小距离
        mindist= dist;  // 更新最小距离
    }
  }
  if (!maxvertex) {  // 如果没有找到最远的顶点
    trace3((qh, qh->ferr, 3067, "qh_furthestvertex: all vertices of f%d are in f%d.  Returning 0.0 for max and mindist\n",
      facetA->id, facetB->id));  // 记录跟踪信息
    maxdist= mindist= 0.0;  // 设置最大和最小距离为 0
  } else {
    trace4((qh, qh->ferr, 4084, "qh_furthestvertex: v%d dist %2.2g is furthest (mindist %2.2g) of f%d above f%d\n",
      maxvertex->id, maxdist, mindist, facetA->id, facetB->id));  // 记录跟踪信息
  }
  *maxdistp= maxdist;  // 将最大距离存入 maxdistp 指向的位置
  *mindistp= mindist;  // 将最小距离存入 mindistp 指向的位置
  return maxvertex;  // 返回最远的顶点
} /* furthestvertex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="getarea">-</a>

  qh_getarea(qh, facetlist )
    set area of all facets in facetlist
    # 收集统计信息
    # 如果存在面积和体积则不执行任何操作

  returns:
    # 设置 qh->totarea/totvol 为凸包的总面积和体积
    # 对于 Delaunay 三角化，计算下层或上层凸壳的投影面积
    # 如果 qh->ATinfinity，则忽略上层凸壳

  notes:
    # 可以通过从内部扩展面积到外部体积来计算外部体积
    # 下面尝试的垂直投影严重低估：
    #   qh.totoutvol += (-dist + facet->maxoutside + qh->DISTround)
    #                         * area/ qh->hull_dim;

  design:
    # 对于 facetlist 上的每个面片
    #   计算面片的面积
    #   更新 qh.totarea 和 qh.totvol
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="getarea">-</a>

  qh_getarea(qh, facetlist)
    computes area and volume properties for each facet in facetlist

  notes:
    sets qh->hasAreaVolume if successfully computed
    updates qh->totarea and qh->totvol with computed values

  design:
    iterate through each facet in facetlist
      skip facet if it lacks a normal vector
      skip facet if it's upper Delaunay and qh->ATinfinity is set
      compute facet area if not already computed
      update qh->totarea with computed area
      if not DELAUNAY
        compute distance from qh->interior_point to facet
        update qh->totvol with computed volume
      if PRINTstatistics is set, update statistics on facet areas
*/
void qh_getarea(qhT *qh, facetT *facetlist) {
  realT area;
  realT dist;
  facetT *facet;

  if (qh->hasAreaVolume)
    return;
  if (qh->REPORTfreq)
    qh_fprintf(qh, qh->ferr, 8020, "computing area of each facet and volume of the convex hull\n");
  else
    trace1((qh, qh->ferr, 1001, "qh_getarea: computing area for each facet and its volume to qh.interior_point (dist*area/dim)\n"));
  qh->totarea= qh->totvol= 0.0;
  FORALLfacet_(facetlist) {
    if (!facet->normal)
      continue;
    if (facet->upperdelaunay && qh->ATinfinity)
      continue;
    if (!facet->isarea) {
      facet->f.area= qh_facetarea(qh, facet);
      facet->isarea= True;
    }
    area= facet->f.area;
    if (qh->DELAUNAY) {
      if (facet->upperdelaunay == qh->UPPERdelaunay)
        qh->totarea += area;
    }else {
      qh->totarea += area;
      qh_distplane(qh, qh->interior_point, facet, &dist);
      qh->totvol += -dist * area/ qh->hull_dim;
    }
    if (qh->PRINTstatistics) {
      wadd_(Wareatot, area);
      wmax_(Wareamax, area);
      wmin_(Wareamin, area);
    }
  }
  qh->hasAreaVolume= True;
} /* getarea */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="gram_schmidt">-</a>

  qh_gram_schmidt(qh, dim, row )
    implements Gram-Schmidt orthogonalization by rows

  returns:
    false if zero norm
    overwrites rows[dim][dim]

  notes:
    implements the modified Gram-Schmidt algorithm
    handles cases where norms are zero or very small

  design:
    iterate over each row
      compute the norm of the row
      if the norm is zero, return false
      normalize the row
      for each subsequent row
        compute the inner product with the current row
        orthogonalize the current row with respect to the inner product
*/
boolT qh_gram_schmidt(qhT *qh, int dim, realT **row) {
  realT *rowi, *rowj, norm;
  int i, j, k;

  for (i=0; i < dim; i++) {
    rowi= row[i];
    for (norm=0.0, k=dim; k--; rowi++)
      norm += *rowi * *rowi;
    norm= sqrt(norm);
    wmin_(Wmindenom, norm);
    if (norm == 0.0)  /* either 0 or overflow due to sqrt */
      return False;
    for (k=dim; k--; )
      *(--rowi) /= norm;
    for (j=i+1; j < dim; j++) {
      rowj= row[j];
      for (norm=0.0, k=dim; k--; )
        norm += *rowi++ * *rowj++;
      for (k=dim; k--; )
        *(--rowj) -= *(--rowi) * norm;
    }
  }
  return True;
} /* gram_schmidt */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="inthresholds">-</a>

  qh_inthresholds(qh, normal, angle )
    return True if normal within qh.lower_/upper_threshold

  returns:
    estimate of angle by summing of threshold diffs
      angle may be NULL
      smaller "angle" is better

  notes:
    invalid if qh.SPLITthresholds

  see:
    qh.lower_threshold in qh_initbuild()
    qh_initthresholds()

  design:
    for each dimension
      test threshold
*/
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="joggleinput">-</a>

  qh_joggleinput(qh)
    randomly joggle input to Qhull by qh.JOGGLEmax
    initial input is qh.first_point/qh.num_points of qh.hull_dim
      repeated calls use qh.input_points/qh.num_points

  returns:
    joggles points at qh.first_point/qh.num_points
    copies data to qh.input_points/qh.input_malloc if first time
    determines qh.JOGGLEmax if it was zero
    if qh.DELAUNAY
      computes the Delaunay projection of the joggled points

  notes:
    if qh.DELAUNAY, unnecessarily joggles the last coordinate
    the initial 'QJn' may be set larger than qh_JOGGLEmaxincrease

  design:
    if qh.DELAUNAY
      set qh.SCALElast for reduced precision errors
    if first call
      initialize qh.input_points to the original input points
      if qh.JOGGLEmax == 0
        determine default qh.JOGGLEmax
    else
      increase qh.JOGGLEmax according to qh.build_cnt
    joggle the input by adding a random number in [-qh.JOGGLEmax,qh.JOGGLEmax]
    if qh.DELAUNAY
      sets the Delaunay projection
*/
void qh_joggleinput(qhT *qh) {
  int i, seed, size;
  coordT *coordp, *inputp;
  realT randr, randa, randb;

  if (!qh->input_points) { /* first call */
    // 将原始输入点赋给 qh.input_points
    qh->input_points= qh->first_point;
    // 设置输入内存指示为 qh.POINTSmalloc
    qh->input_malloc= qh->POINTSmalloc;
    // 计算所需内存大小
    size= qh->num_points * qh->hull_dim * (int)sizeof(coordT);
    // 分配内存
    if (!(qh->first_point= (coordT *)qh_malloc((size_t)size))) {
      // 内存不足报错
      qh_fprintf(qh, qh->ferr, 6009, "qhull error: insufficient memory to joggle %d points\n",
          qh->num_points);
      qh_errexit(qh, qh_ERRmem, NULL, NULL);
    }
    // 设置 POINTSmalloc 标志为真
    qh->POINTSmalloc= True;
    // 若 qh.JOGGLEmax 为 0，则确定默认值
    if (qh->JOGGLEmax == 0.0) {
      qh->JOGGLEmax= qh_detjoggle(qh, qh->input_points, qh->num_points, qh->hull_dim);
      qh_option(qh, "QJoggle", NULL, &qh->JOGGLEmax);
    }
  } else {                 /* repeated call */
    // 如果非第一次调用且 build_cnt 大于 qh_JOGGLEretry
    if (!qh->RERUN && qh->build_cnt > qh_JOGGLEretry) {
      // 每隔 qh_JOGGLEagain 次增加 qh.JOGGLEmax
      if (((qh->build_cnt-qh_JOGGLEretry-1) % qh_JOGGLEagain) == 0) {
        // 计算最大允许的 joggle 值
        realT maxjoggle= qh->MAXwidth * qh_JOGGLEmaxincrease;
        // 如果当前 qh.JOGGLEmax 小于 maxjoggle，则增加 qh.JOGGLEmax
        if (qh->JOGGLEmax < maxjoggle) {
          qh->JOGGLEmax *= qh_JOGGLEincrease;
          minimize_(qh->JOGGLEmax, maxjoggle);
        }
      }
    }
  }
  qh_option(qh, "QJoggle", NULL, &qh->JOGGLEmax);

# 设置 QJoggle 选项，将其值存储到 qh->JOGGLEmax 中

  }
  if (qh->build_cnt > 1 && qh->JOGGLEmax > fmax_(qh->MAXwidth/4, 0.1)) {
      // 检查是否需要进行 QJoggle 操作，确保其值不会过大
      qh_fprintf(qh, qh->ferr, 6010, "qhull input error (qh_joggleinput): the current joggle for 'QJn', %.2g, is too large for the width\nof the input.  If possible, recompile Qhull with higher-precision reals.\n",
                qh->JOGGLEmax);
      // 如果 QJoggle 值过大，输出错误信息并终止程序运行
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  /* for some reason, using qh->ROTATErandom and qh_RANDOMseed does not repeat the run. Use 'TRn' instead */
  // 由于某些原因，使用 qh->ROTATErandom 和 qh_RANDOMseed 无法重复运行。因此使用 'TRn' 代替
  seed= qh_RANDOMint;
  qh_option(qh, "_joggle-seed", &seed, NULL);
  // 设置 _joggle-seed 选项，使用 seed 作为随机数种子

  trace0((qh, qh->ferr, 6, "qh_joggleinput: joggle input by %4.4g with seed %d\n",
    qh->JOGGLEmax, seed));
  // 输出调试信息，显示 QJoggle 操作的参数和随机种子值

  inputp= qh->input_points;
  coordp= qh->first_point;
  randa= 2.0 * qh->JOGGLEmax/qh_RANDOMmax;
  randb= -qh->JOGGLEmax;
  size= qh->num_points * qh->hull_dim;
  // 初始化 QJoggle 操作的参数

  for (i=size; i--; ) {
    randr= qh_RANDOMint;
    *(coordp++)= *(inputp++) + (randr * randa + randb);
    // 执行 QJoggle 操作，为每个坐标添加随机扰动
  }

  if (qh->DELAUNAY) {
    // 如果设置了 DELAUNAY 选项，进行相关操作
    qh->last_low= qh->last_high= qh->last_newhigh= REALmax;
    qh_setdelaunay(qh, qh->hull_dim, qh->num_points, qh->first_point);
    // 设置 Delaunay 选项，并进行 Delaunay 三角化操作
  }
/* joggleinput 的结束标记 */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxabsval">-</a>

  qh_maxabsval( normal, dim )
    返回指向 dim 维向量中最大绝对值的指针
    若 dim=0 则返回 NULL
*/
realT *qh_maxabsval(realT *normal, int dim) {
  realT maxval= -REALmax;  /* 初始化最大值为负无穷 */
  realT *maxp= NULL, *colp, absval;  /* 初始化最大值指针为空，定义列指针和绝对值变量 */
  int k;

  for (k=dim, colp= normal; k--; colp++) {  /* 遍历 dim 维向量的每一列 */
    absval= fabs_(*colp);  /* 计算当前列的绝对值 */
    if (absval > maxval) {  /* 如果当前绝对值大于当前最大值 */
      maxval= absval;  /* 更新最大值 */
      maxp= colp;  /* 更新最大值指针 */
    }
  }
  return maxp;  /* 返回最大值指针 */
} /* maxabsval */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxmin">-</a>

  qh_maxmin(qh, points, numpoints, dimension )
    返回每个维度的最大和最小点
    确定最大和最小坐标

  返回:
    返回一个临时的最大和最小点集合
      可能包含重复的点。不包括 qh.GOODpoint
    设置 qh.NEARzero, qh.MAXabs_coord, qh.MAXsumcoord, qh.MAXwidth
         qh.MAXlastcoord, qh.MINlastcoord
    初始化 qh.max_outside, qh.min_vertex, qh.WAScoplanar, qh.ZEROall_ok

  注意:
    在 qh_detjoggle() 中的循环被复制了一次

  设计:
    初始化全局精度变量
    检查 REAL 的定义...
    对于每个维度
      对于每个点
        收集最大和最小点
      收集最大值的最大值和最小值的最小值
      为高斯消元确定 qh.NEARzero
*/
setT *qh_maxmin(qhT *qh, pointT *points, int numpoints, int dimension) {
  int k;
  realT maxcoord, temp;
  pointT *minimum, *maximum, *point, *pointtemp;
  setT *set;

  qh->max_outside= 0.0;  /* 初始化最大外部值 */
  qh->MAXabs_coord= 0.0;  /* 初始化最大绝对坐标 */
  qh->MAXwidth= -REALmax;  /* 初始化最大宽度 */
  qh->MAXsumcoord= 0.0;  /* 初始化最大总坐标 */
  qh->min_vertex= 0.0;  /* 初始化最小顶点 */
  qh->WAScoplanar= False;  /* 初始化 WAScoplanar 为假 */
  if (qh->ZEROcentrum)
    qh->ZEROall_ok= True;  /* 如果 ZEROcentrum 存在，则将 ZEROall_ok 设置为真 */
  if (REALmin < REALepsilon && REALmin < REALmax && REALmin > -REALmax
  && REALmax > 0.0 && -REALmax < 0.0)
    ; /* all ok */  /* 如果浮点常数定义正确，则一切正常 */
  else {
    qh_fprintf(qh, qh->ferr, 6011, "qhull error: one or more floating point constants in user_r.h are inconsistent. REALmin %g, -REALmax %g, 0.0, REALepsilon %g, REALmax %g\n",
          REALmin, -REALmax, REALepsilon, REALmax);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);  /* 如果浮点常数定义不一致，则报错退出 */
  }
  set= qh_settemp(qh, 2*dimension);  /* 创建一个临时集合 */
  trace1((qh, qh->ferr, 8082, "qh_maxmin: dim             min             max           width    nearzero  min-point  max-point\n"));
  for (k=0; k < dimension; k++) {  /* 对于每个维度 */
    if (points == qh->GOODpointp)  /* 如果点等于 GOODpointp */
      minimum= maximum= points + dimension;  /* 最小和最大均指向 points + dimension */
    else
      minimum= maximum= points;  /* 否则最小和最大均指向 points */
    FORALLpoint_(qh, points, numpoints) {  /* 对于所有的点 */
      if (point == qh->GOODpointp)  /* 如果点等于 GOODpointp */
        continue;  /* 跳过此点 */
      if (maximum[k] < point[k])  /* 如果最大值小于当前点的第 k 维 */
        maximum= point;  /* 更新最大点 */
      else if (minimum[k] > point[k])  /* 否则如果最小值大于当前点的第 k 维 */
        minimum= point;  /* 更新最小点 */
    }
    if (k == dimension-1) {  /* 如果 k 等于维度减一 */
      qh->MINlastcoord= minimum[k];  /* 更新最小最后坐标 */
      qh->MAXlastcoord= maximum[k];  /* 更新最大最后坐标 */
    }
    if (qh->SCALElast && k == dimension-1)
      maxcoord= qh->MAXabs_coord;
    else {
      maxcoord= fmax_(maximum[k], -minimum[k]);
      // 计算最大坐标：取最大值函数 fmax_ 对 maximum[k] 和 -minimum[k] 进行比较
      if (qh->GOODpointp) {
        temp= fmax_(qh->GOODpointp[k], -qh->GOODpointp[k]);
        // 如果 GOODpointp 存在，则计算 temp 为 GOODpointp[k] 和 -qh->GOODpointp[k] 的最大值
        maximize_(maxcoord, temp);
        // 更新 maxcoord 的值为 maxcoord 和 temp 的最大值
      }
      temp= maximum[k] - minimum[k];
      // 计算两个极值之间的差
      maximize_(qh->MAXwidth, temp);
      // 更新 qh->MAXwidth 的值为 qh->MAXwidth 和 temp 的最大值
    }
    maximize_(qh->MAXabs_coord, maxcoord);
    // 更新 qh->MAXabs_coord 的值为 qh->MAXabs_coord 和 maxcoord 的最大值
    qh->MAXsumcoord += maxcoord;
    // 将 maxcoord 加到 qh->MAXsumcoord 中
    qh_setappend(qh, &set, minimum);
    // 将 minimum 添加到 qh->set 中
    qh_setappend(qh, &set, maximum);
    // 将 maximum 添加到 qh->set 中
    /* calculation of qh NEARzero is based on Golub & van Loan, 1983,
       Eq. 4.4-13 for "Gaussian elimination with complete pivoting".
       Golub & van Loan say that n^3 can be ignored and 10 be used in
       place of rho */
    // 计算 qh NEARzero 的值基于 Golub & van Loan, 1983 年的公式 Eq. 4.4-13，
    // 用于描述带有完全主元选取的高斯消元过程
    // Golub & van Loan 指出 n^3 可以忽略，并且用 10 替代 rho
    qh->NEARzero[k]= 80 * qh->MAXsumcoord * REALepsilon;
    // 根据给定公式计算 NEARzero[k] 的值
    trace1((qh, qh->ferr, 8106, "           %3d % 14.8e % 14.8e % 14.8e  %4.4e  p%-9d p%-d\n",
            k, minimum[k], maximum[k], maximum[k]-minimum[k], qh->NEARzero[k], qh_pointid(qh, minimum), qh_pointid(qh, maximum)));
    // 输出跟踪信息到 trace1 中，显示当前迭代中的各个参数和计算结果
    if (qh->SCALElast && k == dimension-1)
      trace1((qh, qh->ferr, 8107, "           last coordinate scaled to (%4.4g, %4.4g), width %4.4e for option 'Qbb'\n",
            qh->MAXabs_coord - qh->MAXwidth, qh->MAXabs_coord, qh->MAXwidth));
      // 如果满足条件，则输出跟踪信息到 trace1，显示最后一个坐标的缩放情况及相关宽度信息
  }
  if (qh->IStracing >= 1)
    qh_printpoints(qh, qh->ferr, "qh_maxmin: found the max and min points (by dim):", set);
    // 如果跟踪级别大于等于 1，则输出找到的最大和最小点的信息到 qh_printpoints 中
  return(set);
  // 返回结果集合 set
} /* maxmin */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxouter">-</a>
  
  qh_maxouter(qh)
    返回从几何结构到外部平面的最大距离
    通常为 qh.max_outside + qh.DISTround
    不包括 qh.JOGGLEmax

  see:
    qh_outerinner()

  notes:
    如果实际点进行计算，需要额外添加 qh.DISTround
    参考 qh_detmaxoutside 中的 qh_RATIO... 目标

  for joggle:
    qh_setfacetplane() 更新了 qh.max_outer，用于 Wnewvertexmax（到顶点的最大距离）
    需要使用 Wnewvertexmax，因为可能存在一个高面片的共面点被低面片替换
    如果测试输入点，需要添加 qh.JOGGLEmax
*/
realT qh_maxouter(qhT *qh) {
  realT dist;

  dist= fmax_(qh->max_outside, qh->DISTround);
  dist += qh->DISTround;
  trace4((qh, qh->ferr, 4012, "qh_maxouter: max distance from facet to outer plane is %4.4g, qh.max_outside is %4.4g\n", dist, qh->max_outside));
  return dist;
} /* maxouter */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxsimplex">-</a>

  qh_maxsimplex(qh, dim, maxpoints, points, numpoints, simplex )
    确定一组点的最大单纯形
    maxpoints 是具有最小或最大坐标的点的子集
    可以从已经在单纯形中的点开始
    跳过 qh.GOODpointp（假设它不在 maxpoints 中）

  returns:
    含有 dim+1 个点的单纯形

  notes:
    被 qh_initialvertices、qh_detvnorm 和 qh_voronoi_center 调用
    需要 qh.MAXwidth 以估计每个顶点的行列式
    假设 points 中至少有所需的点
    最大化 x、y、z、w 等的行列式
    使用 maxpoints，只要行列式显然非零

  design:
    初始化至少两个点的单纯形
      （查找具有最大或最小 x 坐标的点）
    创建 dim+1 个顶点的单纯形如下
      添加 maxpoints 中最大化点与单纯形顶点的行列式
      如果最后一个点并且 maxdet/prevdet < qh_RATIOmaxsimplex（3.0e-2）
        标记 maybe_falsenarrow
      如果没有 maxpoint 或 maxnearzero 或 maybe_falsenarrow
        对所有点搜索最大行列式
        如果 maybe_falsenarrow 并且 !maxnearzero 且 maxdet > prevdet，早期退出
*/
void qh_maxsimplex(qhT *qh, int dim, setT *maxpoints, pointT *points, numpoints, setT **simplex) {
  pointT *point, **pointp, *pointtemp, *maxpoint, *minx=NULL, *maxx=NULL;
  boolT nearzero, maxnearzero= False, maybe_falsenarrow;
  int i, sizinit;
  realT maxdet= -1.0, prevdet= -1.0, det, mincoord= REALmax, maxcoord= -REALmax, mindet, ratio, targetdet;

  if (qh->MAXwidth <= 0.0) {
    qh_fprintf(qh, qh->ferr, 6421, "qhull internal error (qh_maxsimplex): qh.MAXwidth required for qh_maxsimplex.  Used to estimate determinate\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  sizinit= qh_setsize(qh, *simplex);
  if (sizinit >= 2) {
    maxdet= pow(qh->MAXwidth, sizinit - 1);
  }else {
    // 如果不是第一次计算，根据最大点数设置大小，并找到最大和最小的 x 坐标点
    if (qh_setsize(qh, maxpoints) >= 2) {
      FOREACHpoint_(maxpoints) {
        // 找到最大的 x 坐标点及其对应的点
        if (maxcoord < point[0]) {
          maxcoord= point[0];
          maxx= point;
        }
        // 找到最小的 x 坐标点及其对应的点
        if (mincoord > point[0]) {
          mincoord= point[0];
          minx= point;
        }
      }
    }else {
      // 遍历所有点，排除 GOODpointp，并找到最大和最小的 x 坐标点
      FORALLpoint_(qh, points, numpoints) {
        if (point == qh->GOODpointp)
          continue;
        if (maxcoord < point[0]) {
          maxcoord= point[0];
          maxx= point;
        }
        if (mincoord > point[0]) {
          mincoord= point[0];
          minx= point;
        }
      }
    }
    // 计算最大和最小 x 坐标点之间的距离
    maxdet= maxcoord - mincoord;
    // 将 minx 添加到 simplex 中，并保证 simplex 中的点是唯一的
    qh_setunique(qh, simplex, minx);
    // 如果 simplex 中的点数小于 2，则将 maxx 添加到 simplex 中
    if (qh_setsize(qh, *simplex) < 2)
      qh_setunique(qh, simplex, maxx);
    // 更新 sizinit 为 simplex 中点的数量
    sizinit= qh_setsize(qh, *simplex);
    // 如果 sizinit 小于 2，则重新启动并报错
    if (sizinit < 2) {
      qh_joggle_restart(qh, "input has same x coordinate");
      // 检查 zzval_(Zsetplane) 是否大于 hull_dim+1，若是，则报错
      if (zzval_(Zsetplane) > qh->hull_dim+1) {
        qh_fprintf(qh, qh->ferr, 6012, "qhull precision error (qh_maxsimplex for voronoi_center): %d points with the same x coordinate %4.4g\n",
                 qh_setsize(qh, maxpoints)+numpoints, mincoord);
        qh_errexit(qh, qh_ERRprec, NULL, NULL);
      }else {
        // 否则报输入错误
        qh_fprintf(qh, qh->ferr, 6013, "qhull input error: input is less than %d-dimensional since all points have the same x coordinate %4.4g\n",
                 qh->hull_dim, mincoord);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
    }
  }
  // 对于每个维度，计算最大的行列式值及其对应的点
  for (i=sizinit; i < dim+1; i++) {
    prevdet= maxdet;
    maxpoint= NULL;
    maxdet= -1.0;
    FOREACHpoint_(maxpoints) {
      // 如果点不在 simplex 中并且不是 maxpoint，则计算行列式值
      if (!qh_setin(*simplex, point) && point != maxpoint) {
        // 计算 i 维度下的行列式值，并检查是否存在重复或多次迭代
        det= qh_detsimplex(qh, point, *simplex, i, &nearzero); /* retests maxpoints if duplicate or multiple iterations */
        // 计算行列式值的绝对值，并找出最大的值及其对应的点
        if ((det= fabs_(det)) > maxdet) {
          maxdet= det;
          maxpoint= point;
          maxnearzero= nearzero;
        }
      }
    }
    maybe_falsenarrow= False;
    ratio= 1.0;
    // 计算目标行列式值并确定可能的误差
    targetdet= prevdet * qh->MAXwidth;
    mindet= 10 * qh_RATIOmaxsimplex * targetdet;
    if (maxdet > 0.0) {
      ratio= maxdet / targetdet;
      if (ratio < qh_RATIOmaxsimplex)
        maybe_falsenarrow= True;
    }
    # 如果 maxpoint 为空或者 maxnearzero 为真或者 maybe_falsenarrow 为真，则进入条件判断
    if (!maxpoint || maxnearzero || maybe_falsenarrow) {
      # 增加搜索点的计数器 Zsearchpoints
      zinc_(Zsearchpoints);
      # 如果 maxpoint 为空，则记录跟踪信息，搜索所有点，找到第 i+1 个初始顶点，优于 mindet 和 targetdet
      if (!maxpoint) {
        trace0((qh, qh->ferr, 7, "qh_maxsimplex: searching all points for %d-th initial vertex, better than mindet %4.4g, targetdet %4.4g\n",
                i+1, mindet, targetdet));
      } else if (qh->ALLpoints) {
        # 如果 ALLpoints 标志为真，则记录跟踪信息，搜索所有点（'Qs'），找到第 i+1 个初始顶点，优于 p%d 的 det、targetdet、ratio
        trace0((qh, qh->ferr, 30, "qh_maxsimplex: searching all points ('Qs') for %d-th initial vertex, better than p%d det %4.4g, targetdet %4.4g, ratio %4.4g\n",
                i+1, qh_pointid(qh, maxpoint), maxdet, targetdet, ratio));
      } else if (maybe_falsenarrow) {
        # 如果 maybe_falsenarrow 为真，则记录跟踪信息，搜索所有点，找到第 i+1 个初始顶点，优于 p%d 的 det、mindet、ratio
        trace0((qh, qh->ferr, 17, "qh_maxsimplex: searching all points for %d-th initial vertex, better than p%d det %4.4g and mindet %4.4g, ratio %4.4g\n",
                i+1, qh_pointid(qh, maxpoint), maxdet, mindet, ratio));
      } else {
        # 记录跟踪信息，搜索所有点，找到第 i+1 个初始顶点，优于 p%d 的 det、mindet、targetdet
        trace0((qh, qh->ferr, 8, "qh_maxsimplex: searching all points for %d-th initial vertex, better than p%d det %2.2g and mindet %4.4g, targetdet %4.4g\n",
                i+1, qh_pointid(qh, maxpoint), maxdet, mindet, targetdet));
      }
      # 遍历所有点，寻找最优的 maxpoint
      FORALLpoint_(qh, points, numpoints) {
        # 如果 point 是 GOODpointp，则跳过
        if (point == qh->GOODpointp)
          continue;
        # 如果 point 既不在 maxpoints 集合中，也不在 simplex 集合中
        if (!qh_setin(maxpoints, point) && !qh_setin(*simplex, point)) {
          # 计算 point 加入 *simplex 后的行列式值，同时获取可能的 nearzero 值
          det= qh_detsimplex(qh, point, *simplex, i, &nearzero);
          # 取行列式的绝对值
          if ((det= fabs_(det)) > maxdet) {
            # 如果新的行列式值大于当前的 maxdet，则更新 maxdet、maxpoint 和 maxnearzero
            maxdet= det;
            maxpoint= point;
            maxnearzero= nearzero;
            # 如果 maxdet 大于 mindet 且 maxnearzero 为假且 ALLpoints 为假，则提前结束循环
            if (!maxnearzero && !qh->ALLpoints && maxdet > mindet)
              break;
          }
        }
      }
    } /* !maxpoint */
    # 如果 maxpoint 仍为空，则输出错误信息并退出程序
    if (!maxpoint) {
      qh_fprintf(qh, qh->ferr, 6014, "qhull internal error (qh_maxsimplex): not enough points available\n");
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    # 将 maxpoint 添加到 simplex 集合中
    qh_setappend(qh, simplex, maxpoint);
    # 记录跟踪信息，选择了点 p%d 作为第 i+1 个初始顶点，行列式值为 maxdet，目标行列式值为 prevdet * qh->MAXwidth，mindet 为 %4.4g
    trace1((qh, qh->ferr, 1002, "qh_maxsimplex: selected point p%d for %d`th initial vertex, det=%4.4g, targetdet=%4.4g, mindet=%4.4g\n",
            qh_pointid(qh, maxpoint), i+1, maxdet, prevdet * qh->MAXwidth, mindet));
  } /* i */
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="outerinner">-</a>

  qh_outerinner(qh, facet, outerplane, innerplane  )
    if facet and qh.maxoutdone (i.e., qh_check_maxout)
      returns outer and inner plane for facet
    else
      returns maximum outer and inner plane
    accounts for qh.JOGGLEmax

  see:
    qh_maxouter(qh), qh_check_bestdist(), qh_check_points()

  notes:
    outerplaner or innerplane may be NULL
    facet is const
    Does not error (QhullFacet)

    includes qh.DISTround for actual points
    adds another qh.DISTround if testing with floating point arithmetic
*/
void qh_outerinner(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane) {
  realT dist, mindist;
  vertexT *vertex, **vertexp;

  if (outerplane) {
    if (!qh_MAXoutside || !facet || !qh->maxoutdone) {
      *outerplane= qh_maxouter(qh);       /* includes qh.DISTround */
    }else { /* qh_MAXoutside ... */
#if qh_MAXoutside
      *outerplane= facet->maxoutside + qh->DISTround;
#endif

    }
    if (qh->JOGGLEmax < REALmax/2)
      *outerplane += qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
  }
  if (innerplane) {
    /* Calculate inner plane */
    mindist= qh->hull_dim * qh->DISTround;
    FOREACHvertex_(facet->vertices) {
      qh_distplane(qh, vertex->point, facet, &dist);
      minimize_(mindist, dist);
    }
    *innerplane= facet->offset - mindist;
    if (qh->JOGGLEmax < REALmax/2)
      *innerplane -= qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
  }
} /* outerinner */
    // 如果 facet 存在，则计算最小距离到该 facet 的平面距离
    if (facet) {
        // 初始化最小距离为实数最大值
        mindist= REALmax;
        // 遍历 facet 的所有顶点
        FOREACHvertex_(facet->vertices) {
            // 增加距离计数
            zinc_(Zdistio);
            // 计算顶点到 facet 的距离
            qh_distplane(qh, vertex->point, facet, &dist);
            // 更新最小距离
            minimize_(mindist, dist);
        }
        // 计算内部平面距离，并考虑舍入误差
        *innerplane= mindist - qh->DISTround;
    } else {
        // 如果 facet 不存在，则使用 qh->min_vertex 计算内部平面距离，并考虑舍入误差
        *innerplane= qh->min_vertex - qh->DISTround;
    }
    // 如果 JOGGLEmax 小于实数最大值的一半，则进一步调整内部平面距离
    if (qh->JOGGLEmax < REALmax/2)
        *innerplane -= qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
}
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="pointdist">-</a>

  qh_pointdist( point1, point2, dim )
    return distance between two points

  notes:
    returns distance squared if 'dim' is negative
*/
// 计算两个点之间的距离
coordT qh_pointdist(pointT *point1, pointT *point2, int dim) {
  coordT dist, diff;
  int k;

  dist= 0.0;
  // 遍历每一个维度
  for (k= (dim > 0 ? dim : -dim); k--; ) {
    // 计算每个维度上的差值
    diff= *point1++ - *point2++;
    // 累加平方差值
    dist += diff * diff;
  }
  // 如果 dim 大于 0，则返回平方根，否则返回距离的平方
  if (dim > 0)
    return(sqrt(dist));
  return dist;
} /* pointdist */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="printmatrix">-</a>

  qh_printmatrix(qh, fp, string, rows, numrow, numcol )
    print matrix to fp given by row vectors
    print string as header
    qh may be NULL if fp is defined

  notes:
    print a vector by qh_printmatrix(qh, fp, "", &vect, 1, len)
*/
// 打印由行向量组成的矩阵到文件流 fp 中
void qh_printmatrix(qhT *qh, FILE *fp, const char *string, realT **rows, int numrow, int numcol) {
  realT *rowp;
  realT r; /*bug fix*/
  int i,k;

  // 打印字符串作为表头
  qh_fprintf(qh, fp, 9001, "%s\n", string);
  // 遍历每一行
  for (i=0; i < numrow; i++) {
    rowp= rows[i];
    // 遍历每一列
    for (k=0; k < numcol; k++) {
      r= *rowp++;
      // 打印每个元素的值
      qh_fprintf(qh, fp, 9002, "%6.3g ", r);
    }
    // 换行
    qh_fprintf(qh, fp, 9003, "\n");
  }
} /* printmatrix */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="printpoints">-</a>

  qh_printpoints(qh, fp, string, points )
    print pointids to fp for a set of points
    if string, prints string and 'p' point ids
*/
// 打印点集合的点 ID 到文件流 fp 中
void qh_printpoints(qhT *qh, FILE *fp, const char *string, setT *points) {
  pointT *point, **pointp;

  // 如果有字符串，打印字符串和每个点的 ID
  if (string) {
    qh_fprintf(qh, fp, 9004, "%s", string);
    FOREACHpoint_(points)
      qh_fprintf(qh, fp, 9005, " p%d", qh_pointid(qh, point));
    qh_fprintf(qh, fp, 9006, "\n");
  }else {
    // 否则，只打印每个点的 ID
    FOREACHpoint_(points)
      qh_fprintf(qh, fp, 9007, " %d", qh_pointid(qh, point));
    qh_fprintf(qh, fp, 9008, "\n");
  }
} /* printpoints */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="projectinput">-</a>

  qh_projectinput(qh)
    project input points using qh.lower_bound/upper_bound and qh->DELAUNAY
    if qh.lower_bound[k]=qh.upper_bound[k]= 0,
      removes dimension k
    if halfspace intersection
      removes dimension k from qh.feasible_point
    input points in qh->first_point, num_points, input_dim

  returns:
    new point array in qh->first_point of qh->hull_dim coordinates
    sets qh->POINTSmalloc
    if qh->DELAUNAY
      projects points to paraboloid
      lowbound/highbound is also projected
    if qh->ATinfinity
      adds point "at-infinity"
    if qh->POINTSmalloc
      frees old point array

  notes:
    checks that qh.hull_dim agrees with qh.input_dim, PROJECTinput, and DELAUNAY


  design:
    sets project[k] to -1 (delete), 0 (keep), 1 (add for Delaunay)

*/
// 使用 qh.lower_bound/upper_bound 和 qh->DELAUNAY 投影输入点
void qh_projectinput(qhT *qh) {
  // 省略函数体的具体实现和设计思路
}
    # 计算 qh->hull_dim 和 qh->num_points 的 newdim 和 newnum
    determines newdim and newnum for qh->hull_dim and qh->num_points
    # 将 points 投影到 newpoints 上
    projects points to newpoints
    # 将 qh.lower_bound 投影到自身
    projects qh.lower_bound to itself
    # 将 qh.upper_bound 投影到自身
    projects qh.upper_bound to itself
    # 如果 qh->DELAUNAY 为真
    if qh->DELAUNAY
      # 如果 qh->ATINFINITY 为真
      if qh->ATINFINITY
        # 将 points 投影到抛物面上
        projects points to paraboloid
        # 计算 "infinity" 点作为顶点的平均值，并比所有点高 10%
        computes "infinity" point as vertex average and 10% above all points
      else
        # 使用 qh_setdelaunay 将 points 投影到抛物面上
        uses qh_setdelaunay to project points to paraboloid
/*
   函数 qh_projectinput 用于将输入点集投影到更低维度空间，并进行必要的内存分配和错误检查。

   参数说明：
   - qh: qhT 结构体指针，包含了 Qhull 的状态和数据结构

   返回值：无

   实现步骤：
*/

void qh_projectinput(qhT *qh) {
  int k,i;
  int newdim= qh->input_dim, newnum= qh->num_points;
  signed char *project;
  int projectsize= (qh->input_dim + 1) * (int)sizeof(*project);
  pointT *newpoints, *coord, *infinity;
  realT paraboloid, maxboloid= 0;

  // 分配内存用于存储投影向量
  project= (signed char *)qh_memalloc(qh, projectsize);
  // 初始化投影向量为零
  memset((char *)project, 0, (size_t)projectsize);
  
  // 遍历输入维度，跳过 Delaunay 边界
  for (k=0; k < qh->input_dim; k++) {
    if (qh->lower_bound[k] == 0.0 && qh->upper_bound[k] == 0.0) {
      project[k]= -1;
      newdim--;
    }
  }
  
  // 如果是 Delaunay 三角化，则进行特殊处理
  if (qh->DELAUNAY) {
    project[k]= 1;
    newdim++;
    if (qh->ATinfinity)
      newnum++;
  }
  
  // 检查投影后的维度是否与凸壳维度一致，如果不一致则输出错误信息并退出
  if (newdim != qh->hull_dim) {
    qh_memfree(qh, project, projectsize);
    qh_fprintf(qh, qh->ferr, 6015, "qhull internal error (qh_projectinput): dimension after projection %d != hull_dim %d\n", newdim, qh->hull_dim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  
  // 分配新的点集内存，如果内存不足则输出错误信息并退出
  if (!(newpoints= qh->temp_malloc= (coordT *)qh_malloc((size_t)(newnum * newdim) * sizeof(coordT)))) {
    qh_memfree(qh, project, projectsize);
    qh_fprintf(qh, qh->ferr, 6016, "qhull error: insufficient memory to project %d points\n",
           qh->num_points);
    qh_errexit(qh, qh_ERRmem, NULL, NULL);
  }
  
  // 将输入点集进行投影到新的维度空间
  qh_projectpoints(qh, project, qh->input_dim+1, qh->first_point,
                    qh->num_points, qh->input_dim, newpoints, newdim);
  
  // 更新下界和上界
  trace1((qh, qh->ferr, 1003, "qh_projectinput: updating lower and upper_bound\n"));
  qh_projectpoints(qh, project, qh->input_dim+1, qh->lower_bound,
                    1, qh->input_dim+1, qh->lower_bound, newdim+1);
  qh_projectpoints(qh, project, qh->input_dim+1, qh->upper_bound,
                    1, qh->input_dim+1, qh->upper_bound, newdim+1);
  
  // 如果是半空间计算，则需要额外处理可行点
  if (qh->HALFspace) {
    if (!qh->feasible_point) {
      qh_memfree(qh, project, projectsize);
      qh_fprintf(qh, qh->ferr, 6017, "qhull internal error (qh_projectinput): HALFspace defined without qh.feasible_point\n");
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    qh_projectpoints(qh, project, qh->input_dim, qh->feasible_point,
                      1, qh->input_dim, qh->feasible_point, newdim);
  }
  
  // 释放投影向量的内存
  qh_memfree(qh, project, projectsize);
  
  // 如果之前分配了 POINTSmalloc，则释放原始点集内存
  if (qh->POINTSmalloc)
    qh_free(qh->first_point);
  
  // 更新 Qhull 数据结构中的点集信息
  qh->first_point= newpoints;
  qh->POINTSmalloc= True;
  qh->temp_malloc= NULL;
  
  // 如果是 Delaunay 三角化且需要无限远点，则进行额外处理
  if (qh->DELAUNAY && qh->ATinfinity) {
    coord= qh->first_point;
    infinity= qh->first_point + qh->hull_dim * qh->num_points;
    for (k=qh->hull_dim-1; k--; )
      infinity[k]= 0.0;
    for (i=qh->num_points; i--; ) {
      paraboloid= 0.0;
      for (k=0; k < qh->hull_dim-1; k++) {
        paraboloid += *coord * *coord;
        infinity[k] += *coord;
        coord++;
      }
      *(coord++)= paraboloid;
      maximize_(maxboloid, paraboloid);
    }
    // 最后一个坐标指向无限远点
    /* coord == infinity */
    for (k=qh->hull_dim-1; k--; )
      *(coord++) /= qh->num_points;
    *(coord++)= maxboloid * 1.1;
    qh->num_points++;
  }
}
    trace0((qh, qh->ferr, 9, "qh_projectinput: projected points to paraboloid for Delaunay\n"));


// 使用 trace0 函数输出调试信息，显示消息 "qh_projectinput: projected points to paraboloid for Delaunay"



  }else if (qh->DELAUNAY)  /* !qh->ATinfinity */
    qh_setdelaunay(qh, qh->hull_dim, qh->num_points, qh->first_point);


// 如果 qh->DELAUNAY 为真且 !qh->ATinfinity 也为真，则调用 qh_setdelaunay 函数
// 该函数设置 Delaunay 三角剖分，传递参数为 qh 结构体的维度 qh->hull_dim、点数 qh->num_points 和第一个点 qh->first_point
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="projectpoints">-</a>

  qh_projectpoints(qh, project, n, points, numpoints, dim, newpoints, newdim )
    project points/numpoints/dim to newpoints/newdim
    if project[k] == -1
      delete dimension k
    if project[k] == 1
      add dimension k by duplicating previous column
    n is size of project

  notes:
    newpoints may be points if only adding dimension at end

  design:
    check that 'project' and 'newdim' agree
    for each dimension
      if project == -1
        skip dimension
      else
        determine start of column in newpoints
        determine start of column in points
          if project == +1, duplicate previous column
        copy dimension (column) from points to newpoints
*/
void qh_projectpoints(qhT *qh, signed char *project, int n, realT *points,
        int numpoints, int dim, realT *newpoints, int newdim) {
  int testdim= dim, oldk=0, newk=0, i,j=0,k;
  realT *newp, *oldp;

  for (k=0; k < n; k++)
    testdim += project[k];
  // 检查投影后的维度是否与新维度相符
  if (testdim != newdim) {
    qh_fprintf(qh, qh->ferr, 6018, "qhull internal error (qh_projectpoints): newdim %d should be %d after projection\n",
      newdim, testdim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  for (j=0; j<n; j++) {
    if (project[j] == -1)
      oldk++;
    else {
      newp= newpoints+newk++;
      if (project[j] == +1) {
        if (oldk >= dim)
          continue;
        oldp= points+oldk;
      }else
        oldp= points+oldk++;
      // 复制点的维度数据到新的点集
      for (i=numpoints; i--; ) {
        *newp= *oldp;
        newp += newdim;
        oldp += dim;
      }
    }
    if (oldk >= dim)
      break;
  }
  trace1((qh, qh->ferr, 1004, "qh_projectpoints: projected %d points from dim %d to dim %d\n",
    numpoints, dim, newdim));
} /* projectpoints */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="rotateinput">-</a>

  qh_rotateinput(qh, rows )
    rotate input using row matrix
    input points given by qh->first_point, num_points, hull_dim
    assumes rows[dim] is a scratch buffer
    if qh->POINTSmalloc, overwrites input points, else mallocs a new array

  returns:
    rotated input
    sets qh->POINTSmalloc

  design:
    see qh_rotatepoints
*/
void qh_rotateinput(qhT *qh, realT **rows) {

  if (!qh->POINTSmalloc) {
    // 如果未分配点内存，复制点集并标记为已分配
    qh->first_point= qh_copypoints(qh, qh->first_point, qh->num_points, qh->hull_dim);
    qh->POINTSmalloc= True;
  }
  // 使用行矩阵旋转输入点集
  qh_rotatepoints(qh, qh->first_point, qh->num_points, qh->hull_dim, rows);
}  /* rotateinput */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="rotatepoints">-</a>

  qh_rotatepoints(qh, points, numpoints, dim, row )
    rotate numpoints points by a d-dim row matrix
    assumes rows[dim] is a scratch buffer

  returns:
    rotated points in place

  design:
    This function rotates a set of points in the given dimensional space
    using the provided row matrix. It assumes that 'points' is the array
    of points to be rotated, 'numpoints' is the number of points,
    'dim' is the original dimension, and 'row' is the row matrix used
    for rotation.
*/
    # 对于每个点进行操作循环
    for each point
      # 对于每个坐标进行操作循环
      for each coordinate
        # 使用当前行的指定维度计算局部内积
        use row[dim] to compute partial inner product
      # 对于每个坐标进行操作循环
      for each coordinate
        # 根据局部内积旋转
        rotate by partial inner product
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="scalelast">-</a>

  qh_scalelast(qh, points, numpoints, dim, low, high, newhigh )
    将最后一个坐标缩放到[0.0, newhigh]，用于Delaunay三角剖分
    输入参数 points 指向点集，numpoints 表示点的数量，dim 表示点的维度
              low 是最小的坐标值，high 是最大的坐标值，newhigh 是新的最大坐标值

  returns:
    修改最后一个坐标的范围从[low, high]变为[0.0, newhigh]
    覆盖每个点的最后一个坐标
    保存 low/high/newhigh 到 qh->last_low, qh->last_high, qh->last_newhigh，以便 qh_setdelaunay() 使用

  notes:
    为了减少精度问题，qh_scalelast 使得最后一个坐标类似于其他坐标
    Delaunay三角剖分中的最后一个坐标是输入坐标的平方和
    注意对于具有大正坐标的窄分布（例如[995933.64, 995963.48]），范围[0.0, newwidth] 是错误的

    当被 qh_setdelaunay 调用时，low/high 可能与传递给 qh_setdelaunay 的数据不匹配

  design:
    计算缩放和移动因子
    应用于每个点的最后一个坐标
*/
void qh_scalelast(qhT *qh, coordT *points, int numpoints, int dim, coordT low,
                   coordT high, coordT newhigh) {
  realT scale, shift;
  coordT *coord, newlow;
  int i;
  boolT nearzero= False;

  // 设置新的低坐标值为 0.0
  newlow= 0.0;
  // 输出追踪信息到日志文件
  trace4((qh, qh->ferr, 4013, "qh_scalelast: scale last coordinate from [%2.2g, %2.2g] to [%2.2g, %2.2g]\n",
    low, high, newlow, newhigh));
  // 保存 low/high/newhigh 到全局变量中
  qh->last_low= low;
  qh->last_high= high;
  qh->last_newhigh= newhigh;
  // 计算缩放因子，避免除零错误
  scale= qh_divzero(newhigh - newlow, high - low,
                  qh->MINdenom_1, &nearzero);
  // 如果除法操作中出现了近似零的情况
  if (nearzero) {
    # 如果qh->DELAUNAY为真，则输出错误信息，指示无法将最后一个坐标缩放到指定范围内。
    if (qh->DELAUNAY)
      qh_fprintf(qh, qh->ferr, 6019, "qhull input error (qh_scalelast): can not scale last coordinate to [%4.4g, %4.4g].  Input is cocircular or cospherical.   Use option 'Qz' to add a point at infinity.\n",
             newlow, newhigh);
    # 否则，输出错误信息，指示新的范围与现有范围过宽，导致无法缩放最后一个坐标。
    else
      qh_fprintf(qh, qh->ferr, 6020, "qhull input error (qh_scalelast): can not scale last coordinate to [%4.4g, %4.4g].  New bounds are too wide for compared to existing bounds [%4.4g, %4.4g] (width %4.4g)\n",
             newlow, newhigh, low, high, high-low);
    # 调用qh_errexit函数，以处理输入错误并退出程序。
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  # 计算坐标的偏移量，确保最后一个坐标被缩放到新的范围内。
  shift= newlow - low * scale;
  # 设置坐标指针指向最后一个坐标
  coord= points + dim - 1;
  # 遍历所有点，对最后一个坐标进行缩放和平移操作
  for (i=numpoints; i--; coord += dim)
    *coord= *coord * scale + shift;
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="scalepoints">-</a>

  qh_scalepoints(qh, points, numpoints, dim, newlows, newhighs )
    将点缩放到新的上下界限
    当 newlow= -REALmax 或者 newhigh= +REALmax 时保留旧的界限

  returns:
    缩放后的点
    覆盖旧的点

  design:
    对于每一个坐标
      计算当前的最小和最大界限
      计算缩放和偏移因子
      缩放所有的点
      强制应用新的最小和最大界限到所有点
*/
void qh_scalepoints(qhT *qh, pointT *points, int numpoints, int dim,
        realT *newlows, realT *newhighs) {
  int i,k;
  realT shift, scale, *coord, low, high, newlow, newhigh, mincoord, maxcoord;
  boolT nearzero= False;

  for (k=0; k < dim; k++) {
    newhigh= newhighs[k];
    newlow= newlows[k];
    if (newhigh > REALmax/2 && newlow < -REALmax/2)
      continue;
    low= REALmax;
    high= -REALmax;
    // 计算当前维度的最小和最大界限
    for (i=numpoints, coord=points+k; i--; coord += dim) {
      minimize_(low, *coord);
      maximize_(high, *coord);
    }
    // 如果新的高界限过大，则使用当前的高界限
    if (newhigh > REALmax/2)
      newhigh= high;
    // 如果新的低界限过小，则使用当前的低界限
    if (newlow < -REALmax/2)
      newlow= low;
    // 如果是在求凸包并且当前维度是最后一维，并且新的高界限小于新的低界限，报错并退出
    if (qh->DELAUNAY && k == dim-1 && newhigh < newlow) {
      qh_fprintf(qh, qh->ferr, 6021, "qhull input error: 'Qb%d' or 'QB%d' inverts paraboloid since high bound %.2g < low bound %.2g\n",
               k, k, newhigh, newlow);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    // 计算缩放因子，避免零点
    scale= qh_divzero(newhigh - newlow, high - low,
                  qh->MINdenom_1, &nearzero);
    if (nearzero) {
      qh_fprintf(qh, qh->ferr, 6022, "qhull input error: %d'th dimension's new bounds [%2.2g, %2.2g] too wide for\nexisting bounds [%2.2g, %2.2g]\n",
              k, newlow, newhigh, low, high);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    // 计算偏移量
    shift= (newlow * high - low * newhigh)/(high-low);
    // 缩放点集
    coord= points+k;
    for (i=numpoints; i--; coord += dim)
      *coord= *coord * scale + shift;
    // 再次遍历点集，确保所有点在新的界限内，因为存在舍入误差
    coord= points+k;
    if (newlow < newhigh) {
      mincoord= newlow;
      maxcoord= newhigh;
    }else {
      mincoord= newhigh;
      maxcoord= newlow;
    }
    for (i=numpoints; i--; coord += dim) {
      minimize_(*coord, maxcoord);  /* 因为舍入误差 */
      maximize_(*coord, mincoord);
    }
    // 输出调试信息，记录缩放后的结果
    trace0((qh, qh->ferr, 10, "qh_scalepoints: scaled %d'th coordinate [%2.2g, %2.2g] to [%.2g, %.2g] for %d points by %2.2g and shifted %2.2g\n",
      k, low, high, newlow, newhigh, numpoints, scale, shift));
  }
} /* scalepoints */
    # points 是一个 dim*count 的实数数组。前 dim-1 个坐标是第一个输入点的坐标。
    # array[dim] 是第二个输入点的第一个坐标。
    # array[2*dim] 是第三个输入点的第一个坐标。
    points is a dim*count realT array.  The first dim-1 coordinates
    are the coordinates of the first input point.  array[dim] is
    the first coordinate of the second input point.  array[2*dim] is
    the first coordinate of the third input point.

    # 如果 qh.last_low 已定义（即通过 'Qbb' 调用了 qh_scalelast）
    # 调用 qh_scalelast 来缩放最后一个坐标，使其与其他点的坐标相同
    if qh.last_low defined (i.e., 'Qbb' called qh_scalelast)
      calls qh_scalelast to scale the last coordinate the same as the other points

  # 返回：
  # 对于每个点，
  # 将 point[dim-1] 设置为坐标的平方和
  # 如有需要，将 points 缩放到 'Qbb'
  returns:
    for each point
      sets point[dim-1] to sum of squares of coordinates
    scale points to 'Qbb' if needed

  # 注意：
  # 要投影一个点，使用以下方式：
  # qh_setdelaunay(qh, qh->hull_dim, 1, point)
  notes:
    to project one point, use
      qh_setdelaunay(qh, qh->hull_dim, 1, point)

    # 不要使用选项 'Qbk'、'QBk' 或 'QbB'，因为它们会在原始投影后再次缩放坐标。
    Do not use options 'Qbk', 'QBk', or 'QbB' since they scale
    the coordinates after the original projection.
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sethalfspace">-</a>

  qh_sethalfspace(qh, dim, coords, nextp, normal, offset, feasible )
    set point to dual of halfspace relative to feasible point
    halfspace is normal coefficients and offset.

  returns:
    false and prints error if feasible point is outside of hull
    overwrites coordinates for point at dim coords
    nextp= next point (coords)
    does not call qh_errexit

  design:
    compute distance from feasible point to halfspace
    divide each normal coefficient by -dist
*/
boolT qh_sethalfspace(qhT *qh, int dim, coordT *coords, coordT **nextp,
         coordT *normal, coordT *offset, coordT *feasible) {
  coordT *normp= normal, *feasiblep= feasible, *coordp= coords;
  realT dist;
  realT r; /*bug fix*/
  int k;
  boolT zerodiv;

  // 计算从可行点到半空间的距离
  dist= *offset;
  for (k=dim; k--; )
    dist += *(normp++) * *(feasiblep++);
  
  // 如果距离大于零，说明可行点在半空间外，报错并返回 false
  if (dist > 0)
    goto LABELerroroutside;

  // 将正常系数按照 -dist 进行归一化处理，计算出新的点坐标
  normp= normal;
  if (dist < -qh->MINdenom) {
    for (k=dim; k--; )
      *(coordp++)= *(normp++) / -dist;
  }else {
    for (k=dim; k--; ) {
      *(coordp++)= qh_divzero(*(normp++), -dist, qh->MINdenom_1, &zerodiv);
      if (zerodiv)
        goto LABELerroroutside;
    }
  }

  // 更新下一个点的指针位置
  *nextp= coordp;

  // 如果开启了跟踪功能，输出详细信息
#ifndef qh_NOtrace
  if (qh->IStracing >= 4) {
    qh_fprintf(qh, qh->ferr, 8021, "qh_sethalfspace: halfspace at offset %6.2g to point: ", *offset);
    for (k=dim, coordp=coords; k--; ) {
      r= *coordp++;
      qh_fprintf(qh, qh->ferr, 8022, " %6.2g", r);
    }
    qh_fprintf(qh, qh->ferr, 8023, "\n");
  }
#endif

  // 返回处理成功的标志
  return True;

LABELerroroutside:
  // 如果可行点在半空间外，输出错误信息
  feasiblep= feasible;
  normp= normal;
  qh_fprintf(qh, qh->ferr, 6023, "qhull input error: feasible point is not clearly inside halfspace\nfeasible point: ");
  for (k=dim; k--; )
    qh_fprintf(qh, qh->ferr, 8024, qh_REAL_1, r=*(feasiblep++));
  qh_fprintf(qh, qh->ferr, 8025, "\n     halfspace: ");
  for (k=dim; k--; )
    qh_fprintf(qh, qh->ferr, 8026, qh_REAL_1, r=*(normp++));
  qh_fprintf(qh, qh->ferr, 8027, "\n     at offset: ");
  qh_fprintf(qh, qh->ferr, 8028, qh_REAL_1, *offset);
  qh_fprintf(qh, qh->ferr, 8029, " and distance: ");
  qh_fprintf(qh, qh->ferr, 8030, qh_REAL_1, dist);
  qh_fprintf(qh, qh->ferr, 8031, "\n");

  // 返回处理失败的标志
  return False;
} /* sethalfspace */
/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sethalfspace_all">-</a>

  qh_sethalfspace_all(qh, dim, count, halfspaces, feasible )
    generate dual for halfspace intersection with feasible point
    array of count halfspaces
      each halfspace is normal coefficients followed by offset
      the origin is inside the halfspace if the offset is negative
    feasible is a point inside all halfspaces (http://www.qhull.org/html/qhalf.htm#notes)

  returns:
    malloc'd array of count X dim-1 points

  notes:
    call before qh_init_B or qh_initqhull_globals
    free memory when done
    unused/untested code: please email bradb@shore.net if this works ok for you
    if using option 'Fp', qh.feasible_point must be set (e.g., to 'feasible')
    qh->feasible_point is a malloc'd array that is freed by qh_freebuffers.

  design:
    see qh_sethalfspace
*/
coordT *qh_sethalfspace_all(qhT *qh, int dim, int count, coordT *halfspaces, pointT *feasible) {
  int i, newdim;
  pointT *newpoints;
  coordT *coordp, *normalp, *offsetp;

  // 打印跟踪信息到错误流，说明正在计算半空间交点的对偶
  trace0((qh, qh->ferr, 12, "qh_sethalfspace_all: compute dual for halfspace intersection\n"));
  
  // 计算新的维度
  newdim= dim - 1;
  
  // 分配存储半空间对偶点的内存空间
  if (!(newpoints= (coordT *)qh_malloc((size_t)(count * newdim) * sizeof(coordT)))){
    // 内存分配失败时的错误处理
    qh_fprintf(qh, qh->ferr, 6024, "qhull error: insufficient memory to compute dual of %d halfspaces\n",
          count);
    qh_errexit(qh, qh_ERRmem, NULL, NULL);
  }
  
  // 初始化指针
  coordp= newpoints;
  normalp= halfspaces;
  
  // 遍历每个半空间
  for (i=0; i < count; i++) {
    offsetp= normalp + newdim;
    
    // 调用qh_sethalfspace设置半空间，并检查是否可行点feasible在其中
    if (!qh_sethalfspace(qh, newdim, coordp, &coordp, normalp, offsetp, feasible)) {
      // 如果feasible不在半空间内，释放之前分配的内存并报告错误
      qh_free(newpoints);  /* feasible is not inside halfspace as reported by qh_sethalfspace */
      qh_fprintf(qh, qh->ferr, 8032, "The halfspace was at index %d\n", i);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    normalp= offsetp + 1;
  }
  
  // 返回计算得到的半空间对偶点数组
  return newpoints;
} /* sethalfspace_all */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sharpnewfacets">-</a>

  qh_sharpnewfacets(qh)

  returns:
    true if could be an acute angle (facets in different quadrants)

  notes:
    for qh_findbest

  design:
    for all facets on qh.newfacet_list
      if two facets are in different quadrants
        set issharp
*/
boolT qh_sharpnewfacets(qhT *qh) {
  facetT *facet;
  boolT issharp= False;
  int *quadrant, k;

  // 分配存储每个维度象限信息的内存空间
  quadrant= (int *)qh_memalloc(qh, qh->hull_dim * (int)sizeof(int));
  
  // 遍历所有在qh.newfacet_list上的facets
  FORALLfacet_(qh->newfacet_list) {
    if (facet == qh->newfacet_list) {
      // 对于第一个facet，记录其法向量的象限信息
      for (k=qh->hull_dim; k--; )
        quadrant[ k]= (facet->normal[ k] > 0);
    }else {
      // 对于其他facet，检查法向量的象限信息是否与第一个facet不同，若不同则设置为issharp
      for (k=qh->hull_dim; k--; ) {
        if (quadrant[ k] != (facet->normal[ k] > 0)) {
          issharp= True;
          break;
        }
      }
    }
    // 如果 issharp 为真，则跳出循环
    if (issharp)
      break;
  }
  // 释放 quadrant 占用的内存，大小为 qh->hull_dim * sizeof(int) 字节
  qh_memfree(qh, quadrant, qh->hull_dim * (int)sizeof(int));
  // 记录调试信息到日志，记录函数名和相关信息
  trace3((qh, qh->ferr, 3001, "qh_sharpnewfacets: %d\n", issharp));
  // 返回 issharp 变量的值作为函数的结果
  return issharp;
/* sharpnewfacets 表示函数组织或结构的结束 */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="vertex_bestdist">-</a>
  
  qh_vertex_bestdist(qh, vertices )
  qh_vertex_bestdist2(qh, vertices, vertexp, vertexp2 )
    return nearest distance between vertices
    optionally returns vertex and vertex2

  notes:
    called by qh_partitioncoplanar, qh_mergefacet, qh_check_maxout, qh_checkpoint
*/
/* vertex_bestdist 函数计算给定顶点集合中两两顶点之间的最短距离 */
coordT qh_vertex_bestdist(qhT *qh, setT *vertices) {
  vertexT *vertex, *vertex2;

  /* 调用 qh_vertex_bestdist2 函数来计算最短距离，并返回结果 */
  return qh_vertex_bestdist2(qh, vertices, &vertex, &vertex2);
} /* vertex_bestdist */

/* qh_vertex_bestdist2 函数计算给定顶点集合中两两顶点之间的最短距离，并可选返回距离最近的两个顶点 */
coordT qh_vertex_bestdist2(qhT *qh, setT *vertices, vertexT **vertexp/*= NULL*/, vertexT **vertexp2/*= NULL*/) {
  vertexT *vertex, *vertexA, *bestvertex= NULL, *bestvertex2= NULL;
  coordT dist, bestdist= REALmax;
  int k, vertex_i, vertex_n;

  /* 遍历顶点集合中的每个顶点 */
  FOREACHvertex_i_(qh, vertices) {
    /* 遍历当前顶点后面的所有顶点 */
    for (k= vertex_i+1; k < vertex_n; k++) {
      /* 获取当前顶点及其后续顶点 */
      vertexA= SETelemt_(vertices, k, vertexT);
      /* 计算两顶点间的欧氏距离 */
      dist= qh_pointdist(vertex->point, vertexA->point, -qh->hull_dim);
      /* 如果计算得到的距离比当前记录的最短距离还要小 */
      if (dist < bestdist) {
        /* 更新最短距离及对应的顶点 */
        bestdist= dist;
        bestvertex= vertex;
        bestvertex2= vertexA;
      }
    }
  }
  /* 将距离最近的两个顶点（如果指针非空）赋给传入的参数 */
  *vertexp= bestvertex;
  *vertexp2= bestvertex2;
  /* 返回最短距离的平方根作为结果 */
  return sqrt(bestdist);
} /* vertex_bestdist */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="voronoi_center">-</a>

  qh_voronoi_center(qh, dim, points )
    return Voronoi center for a set of points
    dim is the orginal dimension of the points
    gh.gm_matrix/qh.gm_row are scratch buffers

  returns:
    center as a temporary point (qh_memalloc)
    if non-simplicial,
      returns center for max simplex of points

  notes:
    only called by qh_facetcenter
    from Bowyer & Woodwark, A Programmer's Geometry, 1983, p. 65

  design:
    if non-simplicial
      determine max simplex for points
    translate point0 of simplex to origin
    compute sum of squares of diagonal
    compute determinate
    compute Voronoi center (see Bowyer & Woodwark)
*/
/* qh_voronoi_center 函数计算给定点集合的Voronoi中心 */
pointT *qh_voronoi_center(qhT *qh, int dim, setT *points) {
  pointT *point, **pointp, *point0;
  pointT *center= (pointT *)qh_memalloc(qh, qh->center_size);
  setT *simplex;
  int i, j, k, size= qh_setsize(qh, points);
  coordT *gmcoord;
  realT *diffp, sum2, *sum2row, *sum2p, det, factor;
  boolT nearzero, infinite;

  /* 如果点的数量等于维度加一，则使用当前点集合作为简单形 */
  if (size == dim+1)
    simplex= points;
  /* 如果点的数量少于维度加一，则报错 */
  else if (size < dim+1) {
    qh_memfree(qh, center, qh->center_size);
    qh_fprintf(qh, qh->ferr, 6025, "qhull internal error (qh_voronoi_center):  need at least %d points to construct a Voronoi center\n",
             dim+1);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    simplex= points;  /* 永远不会执行，避免警告 */
  }else {
    /* 否则，确定点集的最大简单形 */
    simplex= qh_settemp(qh, dim+1);
    qh_maxsimplex(qh, dim, points, NULL, 0, &simplex);
  }
  /* 将简单形的第一个点作为起点 */
  point0= SETfirstt_(simplex, pointT);
  gmcoord= qh->gm_matrix;
  /* 遍历每个维度 */
  for (k=0; k < dim; k++) {
    /* 设置当前维度的行 */
    qh->gm_row[k]= gmcoord;
    // 对每个简单形的顶点进行遍历
    FOREACHpoint_(simplex) {
      // 如果当前顶点不等于起始点
      if (point != point0)
        // 将该顶点与起始点在维度k上的差值存入gmcoord，并移动gmcoord指针
        *(gmcoord++)= point[k] - point0[k];
    }
  }
  // 将gmcoord指针的当前位置存入sum2row
  sum2row= gmcoord;
  // 对每个维度i进行遍历
  for (i=0; i < dim; i++) {
    // 初始化sum2为0.0
    sum2= 0.0;
    // 对每个维度k进行遍历
    for (k=0; k < dim; k++) {
      // 计算qh->gm_row[k] + i的值，并将其平方后累加到sum2
      diffp= qh->gm_row[k] + i;
      sum2 += *diffp * *diffp;
    }
    // 将sum2存入gmcoord，并移动gmcoord指针
    *(gmcoord++)= sum2;
  }
  // 计算qh->gm_row的行列式，并将结果存入det，nearzero用于返回近似为零的情况
  det= qh_determinant(qh, qh->gm_row, dim, &nearzero);
  // 计算0.5除以det的值，避免除以零错误，并将结果存入factor，infinite表示是否为无限大
  factor= qh_divzero(0.5, det, qh->MINdenom, &infinite);
  // 如果结果为无限大
  if (infinite) {
    // 将center的每个维度都设置为qh_INFINITE
    for (k=dim; k--; )
      center[k]= qh_INFINITE;
    // 如果启用跟踪输出，打印出现无限大情况的简单形的信息
    if (qh->IStracing)
      qh_printpoints(qh, qh->ferr, "qh_voronoi_center: at infinity for ", simplex);
  }else {
    // 对每个维度i进行遍历
    for (i=0; i < dim; i++) {
      // 重新初始化gmcoord和sum2p
      gmcoord= qh->gm_matrix;
      sum2p= sum2row;
      // 对每个维度k进行遍历
      for (k=0; k < dim; k++) {
        // 将qh->gm_row[k]指向gmcoord
        qh->gm_row[k]= gmcoord;
        // 如果k等于i
        if (k == i) {
          // 对每个维度j进行逆序遍历
          for (j=dim; j--; )
            // 将sum2p的当前值存入gmcoord，并移动gmcoord和sum2p指针
            *(gmcoord++)= *sum2p++;
        }else {
          // 对每个简单形的顶点进行遍历
          FOREACHpoint_(simplex) {
            // 如果当前顶点不等于起始点
            if (point != point0)
              // 将该顶点与起始点在维度k上的差值存入gmcoord，并移动gmcoord指针
              *(gmcoord++)= point[k] - point0[k];
          }
        }
      }
      // 计算qh->gm_row的行列式乘以factor，加上point0[i]的值，并存入center[i]
      center[i]= qh_determinant(qh, qh->gm_row, dim, &nearzero)*factor + point0[i];
    }
#ifndef qh_NOtrace
    // 如果未禁用跟踪并且跟踪级别大于等于3，则执行以下代码块
    if (qh->IStracing >= 3) {
      // 打印消息，包括det和factor的值
      qh_fprintf(qh, qh->ferr, 3061, "qh_voronoi_center: det %2.2g factor %2.2g ", det, factor);
      // 打印中心点的矩阵表示
      qh_printmatrix(qh, qh->ferr, "center:", &center, 1, dim);
      // 如果跟踪级别大于等于5，则进一步打印点的信息
      if (qh->IStracing >= 5) {
        // 打印简单形的点集
        qh_printpoints(qh, qh->ferr, "points", simplex);
        // 遍历简单形中的每个点，打印其编号和到中心点的距离
        FOREACHpoint_(simplex)
          qh_fprintf(qh, qh->ferr, 8034, "p%d dist %.2g, ", qh_pointid(qh, point),
                   qh_pointdist(point, center, dim));
        // 打印换行符
        qh_fprintf(qh, qh->ferr, 8035, "\n");
      }
    }
#endif
  }
  // 如果简单形不等于点集，则释放简单形的临时内存
  if (simplex != points)
    qh_settempfree(qh, &simplex);
  // 返回中心点
  return center;
} /* voronoi_center */
```