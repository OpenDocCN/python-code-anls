# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\geom_r.h`

```
/*
<html><pre>  -<a                             href="qh-geom_r.htm"
  >-------------------------------</a><a name="TOP">-</a>
  
  geom_r.h
    header file for geometric routines
    
   see qh-geom_r.htm and geom_r.c
   
   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/geom_r.h#1 $$Change: 2661 $
   $DateTime: 2019/05/24 20:09:58 $$Author: bbarber $
*/

#ifndef qhDEFgeom
#define qhDEFgeom 1

#include "libqhull_r.h"
*/

/* ============ -macros- ======================== */

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="fabs_">-</a>
  
  fabs_(a)
    returns the absolute value of a
*/
#define fabs_( a ) ((( a ) < 0 ) ? -( a ):( a ))

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="fmax_">-</a>
  
  fmax_(a,b)
    returns the maximum value of a and b
*/
#define fmax_( a,b )  ( ( a ) < ( b ) ? ( b ) : ( a ) )

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="fmin_">-</a>
  
  fmin_(a,b)
    returns the minimum value of a and b
*/
#define fmin_( a,b )  ( ( a ) > ( b ) ? ( b ) : ( a ) )

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="maximize_">-</a>
  
  maximize_(maxval, val)
    set maxval to val if val is greater than maxval
*/
#define maximize_( maxval, val ) { if (( maxval ) < ( val )) ( maxval )= ( val ); }

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="minimize_">-</a>
  
  minimize_(minval, val)
    set minval to val if val is less than minval
*/
#define minimize_( minval, val ) { if (( minval ) > ( val )) ( minval )= ( val ); }

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="det2_">-</a>
  
  det2_(a1, a2,
        b1, b2)
  
    compute a 2-d determinate
*/
#define det2_( a1,a2,b1,b2 ) (( a1 )*( b2 ) - ( a2 )*( b1 ))

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="det3_">-</a>
  
  det3_(a1, a2, a3,
       b1, b2, b3,
       c1, c2, c3)
  
    compute a 3-d determinate
*/
#define det3_( a1,a2,a3,b1,b2,b3,c1,c2,c3 ) ( ( a1 )*det2_( b2,b3,c2,c3 ) \
                - ( b1 )*det2_( a2,a3,c2,c3 ) + ( c1 )*det2_( a2,a3,b2,b3 ) )

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="dX">-</a>
  
  dX( p1, p2 )
  dY( p1, p2 )
  dZ( p1, p2 )
  
    given two indices into rows[],
    
    compute the difference between X, Y, or Z coordinates
*/
#define dX( p1,p2 )  ( *( rows[p1] ) - *( rows[p2] ))
#define dY( p1,p2 )  ( *( rows[p1]+1 ) - *( rows[p2]+1 ))
#define dZ( p1,p2 )  ( *( rows[p1]+2 ) - *( rows[p2]+2 ))
#define dW( p1,p2 )  ( *( rows[p1]+3 ) - *( rows[p2]+3 ))

/*============= prototypes in alphabetical order, infrequent at end ======= */

#ifdef __cplusplus
extern "C" {
#endif

void    qh_backnormal(qhT *qh, realT **rows, int numrow, int numcol, boolT sign, coordT *normal, boolT *nearzero);
// 声明函数 qh_backnormal，用于计算并返回背景法向量

void    qh_distplane(qhT *qh, pointT *point, facetT *facet, realT *dist);
// 声明函数 qh_distplane，用于计算点到平面的距离

facetT *qh_findbest(qhT *qh, pointT *point, facetT *startfacet,
                     boolT bestoutside, boolT isnewfacets, boolT noupper,
                     realT *dist, boolT *isoutside, int *numpart);
// 声明函数 qh_findbest，用于找到最佳的平面，以便在给定点处进行凸包计算

facetT *qh_findbesthorizon(qhT *qh, boolT ischeckmax, pointT *point,
                     facetT *startfacet, boolT noupper, realT *bestdist, int *numpart);
// 声明函数 qh_findbesthorizon，用于在视图界限上找到最佳的水平线

facetT *qh_findbestnew(qhT *qh, pointT *point, facetT *startfacet, realT *dist,
                     boolT bestoutside, boolT *isoutside, int *numpart);
// 声明函数 qh_findbestnew，用于找到最佳的新平面，以便在给定点处进行凸包计算

void    qh_gausselim(qhT *qh, realT **rows, int numrow, int numcol, boolT *sign, boolT *nearzero);
// 声明函数 qh_gausselim，用于高斯消元操作

realT   qh_getangle(qhT *qh, pointT *vect1, pointT *vect2);
// 声明函数 qh_getangle，用于计算两个向量之间的角度

pointT *qh_getcenter(qhT *qh, setT *vertices);
// 声明函数 qh_getcenter，用于计算凸包的中心点

pointT *qh_getcentrum(qhT *qh, facetT *facet);
// 声明函数 qh_getcentrum，用于计算凸包的中心

coordT  qh_getdistance(qhT *qh, facetT *facet, facetT *neighbor, coordT *mindist, coordT *maxdist);
// 声明函数 qh_getdistance，用于计算两个凸包之间的距离

void    qh_normalize(qhT *qh, coordT *normal, int dim, boolT toporient);
// 声明函数 qh_normalize，用于标准化法向量

void    qh_normalize2(qhT *qh, coordT *normal, int dim, boolT toporient,
            realT *minnorm, boolT *ismin);
// 声明函数 qh_normalize2，用于标准化法向量并返回最小范数

pointT *qh_projectpoint(qhT *qh, pointT *point, facetT *facet, realT dist);
// 声明函数 qh_projectpoint，用于将点投影到平面上

void    qh_setfacetplane(qhT *qh, facetT *newfacets);
// 声明函数 qh_setfacetplane，用于设置平面的法向量

void    qh_sethyperplane_det(qhT *qh, int dim, coordT **rows, coordT *point0,
              boolT toporient, coordT *normal, realT *offset, boolT *nearzero);
// 声明函数 qh_sethyperplane_det，用于通过行列式设置超平面

void    qh_sethyperplane_gauss(qhT *qh, int dim, coordT **rows, pointT *point0,
             boolT toporient, coordT *normal, coordT *offset, boolT *nearzero);
// 声明函数 qh_sethyperplane_gauss，用于通过高斯消元设置超平面

boolT   qh_sharpnewfacets(qhT *qh);
// 声明函数 qh_sharpnewfacets，用于检查新凸包是否尖锐

/*========= infrequently used code in geom2_r.c =============*/

coordT *qh_copypoints(qhT *qh, coordT *points, int numpoints, int dimension);
// 声明函数 qh_copypoints，用于复制点集

void    qh_crossproduct(int dim, realT vecA[3], realT vecB[3], realT vecC[3]);
// 声明函数 qh_crossproduct，用于计算向量的叉乘

realT   qh_determinant(qhT *qh, realT **rows, int dim, boolT *nearzero);
// 声明函数 qh_determinant，用于计算行列式

realT   qh_detjoggle(qhT *qh, pointT *points, int numpoints, int dimension);
// 声明函数 qh_detjoggle，用于扰动行列式的计算

void    qh_detmaxoutside(qhT *qh);
// 声明函数 qh_detmaxoutside，用于确定最大外部点

void    qh_detroundoff(qhT *qh);
// 声明函数 qh_detroundoff，用于处理舍入误差

realT   qh_detsimplex(qhT *qh, pointT *apex, setT *points, int dim, boolT *nearzero);
// 声明函数 qh_detsimplex，用于计算简单形状的行列式

realT   qh_distnorm(int dim, pointT *point, pointT *normal, realT *offsetp);
// 声明函数 qh_distnorm，用于计算点到法线的距离

realT   qh_distround(qhT *qh, int dimension, realT maxabs, realT maxsumabs);
// 声明函数 qh_distround，用于处理距离的舍入

realT   qh_divzero(realT numer, realT denom, realT mindenom1, boolT *zerodiv);
// 声明函数 qh_divzero，用于处理除零错误

realT   qh_facetarea(qhT *qh, facetT *facet);
// 声明函数 qh_facetarea，用于计算凸包面片的面积

realT   qh_facetarea_simplex(qhT *qh, int dim, coordT *apex, setT *vertices,
          vertexT *notvertex,  boolT toporient, coordT *normal, realT *offset);
// 声明函数 qh_facetarea_simplex，用于计算简单凸包面片的面积

pointT *qh_facetcenter(qhT *qh, setT *vertices);
// 声明函数 qh_facetcenter，用于计算凸包面片的中心点

facetT *qh_findgooddist(qhT *qh, pointT *point, facetT *facetA, realT *distp, facetT **facetlist);
// 声明函数 qh_findgooddist，用于找到距离给定点最近的凸包面片

vertexT *qh_furthestnewvertex(qhT *qh, unsigned int unvisited, facetT *facet, realT *maxdistp /* qh.newvertex_list */);
// 声明函数 qh_furthestnewvertex，用于找到最远的新顶点
// 返回距离 facetA 和 facetB 最远的顶点的指针
vertexT *qh_furthestvertex(qhT *qh, facetT *facetA, facetT *facetB, realT *maxdistp, realT *mindistp);

// 计算给定 facetlist 中所有面的面积，并存储在每个面的 area 属性中
void qh_getarea(qhT *qh, facetT *facetlist);

// 对于给定的 dim 维度和 rows 数组，进行 Gram-Schmidt 正交化处理
boolT qh_gram_schmidt(qhT *qh, int dim, realT **rows);

// 检查给定的法向量 normal 是否在指定的角度阈值内
boolT qh_inthresholds(qhT *qh, coordT *normal, realT *angle);

// 对输入的点集进行微小扰动，以改善数值稳定性
void qh_joggleinput(qhT *qh);

// 返回给定法向量中的最大绝对值，并返回结果数组
realT *qh_maxabsval(realT *normal, int dim);

// 计算点集中的最大和最小值，并返回作为集合
setT *qh_maxmin(qhT *qh, pointT *points, int numpoints, int dimension);

// 返回 Qhull 计算的最大外接圆半径
realT qh_maxouter(qhT *qh);

// 计算给定点集的最大简单形式（最远点对）
void qh_maxsimplex(qhT *qh, int dim, setT *maxpoints, pointT *points, int numpoints, setT **simplex);

// 返回给定法向量中的最小绝对值，并返回结果
realT qh_minabsval(realT *normal, int dim);

// 返回两个向量 vecA 和 vecB 之间最小的不同维度索引
int qh_mindiff(realT *vecA, realT *vecB, int dim);

// 检查给定的面 facet 是否朝向外部
boolT qh_orientoutside(qhT *qh, facetT *facet);

// 计算指定面 facet 的外部和内部平面方程
void qh_outerinner(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane);

// 计算两个点之间的欧几里得距离
coordT qh_pointdist(pointT *point1, pointT *point2, int dim);

// 打印给定的矩阵 rows 到文件流 fp 中，附带描述字符串 string
void qh_printmatrix(qhT *qh, FILE *fp, const char *string, realT **rows, int numrow, int numcol);

// 打印点集合 points 到文件流 fp 中，附带描述字符串 string
void qh_printpoints(qhT *qh, FILE *fp, const char *string, setT *points);

// 对输入点集进行投影处理
void qh_projectinput(qhT *qh);

// 对给定点集 points 进行投影，结果存储在 newpoints 中
void qh_projectpoints(qhT *qh, signed char *project, int n, realT *points,
                      int numpoints, int dim, realT *newpoints, int newdim);

// 对输入的点集进行旋转处理
void qh_rotateinput(qhT *qh, realT **rows);

// 对给定点集 points 进行旋转处理，结果存储在 rows 中
void qh_rotatepoints(qhT *qh, realT *points, int numpoints, int dim, realT **rows);

// 对输入的点集进行缩放处理
void qh_scaleinput(qhT *qh);

// 对给定点集 points 进行最后一个维度的缩放处理，结果存储在 points 中
void qh_scalelast(qhT *qh, coordT *points, int numpoints, int dim, coordT low,
                  coordT high, coordT newhigh);

// 对给定点集 points 进行整体缩放处理，新的最低和最高值由 newlows 和 newhighs 给出
void qh_scalepoints(qhT *qh, pointT *points, int numpoints, int dim,
                    realT *newlows, realT *newhighs);

// 设置半空间的法向量，计算可行点并返回是否成功
boolT qh_sethalfspace(qhT *qh, int dim, coordT *coords, coordT **nextp,
                      coordT *normal, coordT *offset, coordT *feasible);

// 设置所有半空间的法向量，返回可行点的坐标数组
coordT *qh_sethalfspace_all(qhT *qh, int dim, int count, coordT *halfspaces, pointT *feasible);

// 返回顶点集合 vertices 中的最佳距离
coordT qh_vertex_bestdist(qhT *qh, setT *vertices);

// 返回顶点集合 vertices 中的最佳两点距离，并通过 vertexp 和 vertexp2 返回对应顶点
coordT qh_vertex_bestdist2(qhT *qh, setT *vertices, vertexT **vertexp, vertexT **vertexp2);

// 计算给定点集 points 的 Voronoi 中心，并返回结果点
pointT *qh_voronoi_center(qhT *qh, int dim, setT *points);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFgeom */


这些注释为每个函数声明提供了详细的解释，说明了每个函数的目的和作用。
```