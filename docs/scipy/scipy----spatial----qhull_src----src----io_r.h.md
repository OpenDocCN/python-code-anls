# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\io_r.h`

```
/*
  <html><pre>  -<a                             href="qh-io_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

  io_r.h
  declarations of Input/Output functions

  see README, libqhull_r.h and io_r.c

  Copyright (c) 1993-2019 The Geometry Center.
  $Id: //main/2019/qhull/src/libqhull_r/io_r.h#2 $$Change: 2671 $
  $DateTime: 2019/06/06 11:24:01 $$Author: bbarber $
*/

#ifndef qhDEFio
#define qhDEFio 1

#include "libqhull_r.h"

/*============ constants and flags ==================*/

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_MAXfirst">-</a>

  qh_MAXfirst
    maximum length of first two lines of stdin
*/
#define qh_MAXfirst  200

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_MINradius">-</a>

  qh_MINradius
    min radius for Gp and Gv, fraction of maxcoord
*/
#define qh_MINradius 0.02

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_GEOMepsilon">-</a>

  qh_GEOMepsilon
    adjust outer planes for 'lines closer' and geomview roundoff.
    This prevents bleed through.
*/
#define qh_GEOMepsilon 2e-3

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_WHITESPACE">-</a>

  qh_WHITESPACE
    possible values of white space
*/
#define qh_WHITESPACE " \n\t\v\r\f"


/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="RIDGE">-</a>

  qh_RIDGE
    to select which ridges to print in qh_eachvoronoi
*/
typedef enum
{
    qh_RIDGEall= 0, qh_RIDGEinner, qh_RIDGEouter
}
qh_RIDGE;

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="printvridgeT">-</a>

  printvridgeT
    prints results of qh_printvdiagram

  see:
    <a href="io_r.c#printvridge">qh_printvridge</a> for an example
*/
typedef void (*printvridgeT)(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);

/*============== -prototypes in alphabetical order =========*/

#ifdef __cplusplus
extern "C" {
#endif

/* Function prototypes */

void    qh_dfacet(qhT *qh, unsigned int id);
void    qh_dvertex(qhT *qh, unsigned int id);
int     qh_compare_facetarea(const void *p1, const void *p2);
int     qh_compare_facetvisit(const void *p1, const void *p2);
int     qh_compare_nummerge(const void *p1, const void *p2);
void    qh_copyfilename(qhT *qh, char *filename, int size, const char* source, int length);
void    qh_countfacets(qhT *qh, facetT *facetlist, setT *facets, boolT printall,
              int *numfacetsp, int *numsimplicialp, int *totneighborsp,
              int *numridgesp, int *numcoplanarsp, int *numnumtricoplanarsp);
pointT *qh_detvnorm(qhT *qh, vertexT *vertex, vertexT *vertexA, setT *centers, realT *offsetp);
setT   *qh_detvridge(qhT *qh, vertexT *vertex);
setT   *qh_detvridge3(qhT *qh, vertexT *atvertex, vertexT *vertex);

#ifdef __cplusplus
}
#endif

#endif /* qhDEFio */
*/
# 定义函数 qh_eachvoronoi，接受一些参数并返回一个整数值
int qh_eachvoronoi(qhT *qh, FILE *fp, printvridgeT printvridge, vertexT *atvertex, boolT visitall, qh_RIDGE innerouter, boolT inorder);

# 定义函数 qh_eachvoronoi_all，接受一些参数并返回一个整数值
int qh_eachvoronoi_all(qhT *qh, FILE *fp, printvridgeT printvridge, boolT isUpper, qh_RIDGE innerouter, boolT inorder);

# 定义函数 qh_facet2point，接受一些参数并不返回值
void qh_facet2point(qhT *qh, facetT *facet, pointT **point0, pointT **point1, realT *mindist);

# 定义函数 qh_facetvertices，接受一些参数并返回一个 setT 结构体指针
setT *qh_facetvertices(qhT *qh, facetT *facetlist, setT *facets, boolT allfacets);

# 定义函数 qh_geomplanes，接受一些参数并不返回值
void qh_geomplanes(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane);

# 定义函数 qh_markkeep，接受一些参数并不返回值
void qh_markkeep(qhT *qh, facetT *facetlist);

# 定义函数 qh_markvoronoi，接受一些参数并返回一个 setT 结构体指针
setT *qh_markvoronoi(qhT *qh, facetT *facetlist, setT *facets, boolT printall, boolT *isLowerp, int *numcentersp);

# 定义函数 qh_order_vertexneighbors，接受一些参数并不返回值
void qh_order_vertexneighbors(qhT *qh, vertexT *vertex);

# 定义函数 qh_prepare_output，接受一些参数并不返回值
void qh_prepare_output(qhT *qh);

# 定义函数 qh_printafacet，接受一些参数并不返回值
void qh_printafacet(qhT *qh, FILE *fp, qh_PRINT format, facetT *facet, boolT printall);

# 定义函数 qh_printbegin，接受一些参数并不返回值
void qh_printbegin(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);

# 定义函数 qh_printcenter，接受一些参数并不返回值
void qh_printcenter(qhT *qh, FILE *fp, qh_PRINT format, const char *string, facetT *facet);

# 定义函数 qh_printcentrum，接受一些参数并不返回值
void qh_printcentrum(qhT *qh, FILE *fp, facetT *facet, realT radius);

# 定义函数 qh_printend，接受一些参数并不返回值
void qh_printend(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);

# 定义函数 qh_printend4geom，接受一些参数并不返回值
void qh_printend4geom(qhT *qh, FILE *fp, facetT *facet, int *num, boolT printall);

# 定义函数 qh_printextremes，接受一些参数并不返回值
void qh_printextremes(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);

# 定义函数 qh_printextremes_2d，接受一些参数并不返回值
void qh_printextremes_2d(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);

# 定义函数 qh_printextremes_d，接受一些参数并不返回值
void qh_printextremes_d(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);

# 定义函数 qh_printfacet，接受一些参数并不返回值
void qh_printfacet(qhT *qh, FILE *fp, facetT *facet);

# 定义函数 qh_printfacet2math，接受一些参数并不返回值
void qh_printfacet2math(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format, int notfirst);

# 定义函数 qh_printfacet2geom，接受一些参数并不返回值
void qh_printfacet2geom(qhT *qh, FILE *fp, facetT *facet, realT color[3]);

# 定义函数 qh_printfacet2geom_points，接受一些参数并不返回值
void qh_printfacet2geom_points(qhT *qh, FILE *fp, pointT *point1, pointT *point2,
                               facetT *facet, realT offset, realT color[3]);

# 定义函数 qh_printfacet3math，接受一些参数并不返回值
void qh_printfacet3math(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format, int notfirst);

# 定义函数 qh_printfacet3geom_nonsimplicial，接受一些参数并不返回值
void qh_printfacet3geom_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);

# 定义函数 qh_printfacet3geom_points，接受一些参数并不返回值
void qh_printfacet3geom_points(qhT *qh, FILE *fp, setT *points, facetT *facet, realT offset, realT color[3]);

# 定义函数 qh_printfacet3geom_simplicial，接受一些参数并不返回值
void qh_printfacet3geom_simplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);

# 定义函数 qh_printfacet3vertex，接受一些参数并不返回值
void qh_printfacet3vertex(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format);

# 定义函数 qh_printfacet4geom_nonsimplicial，接受一些参数并不返回值
void qh_printfacet4geom_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);

# 定义函数 qh_printfacet4geom_simplicial，接受一些参数并不返回值
void qh_printfacet4geom_simplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);

# 定义函数 qh_printfacetNvertex_nonsimplicial，接受一些参数并不返回值
void qh_printfacetNvertex_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, int id, qh_PRINT format);

# 定义函数 qh_printfacetNvertex_simplicial，接受一些参数并不返回值
void qh_printfacetNvertex_simplicial(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format);

# 定义函数 qh_printfacetheader，接受一些参数并不返回值
void qh_printfacetheader(qhT *qh, FILE *fp, facetT *facet);

# 定义函数 qh_printfacetridges，接受一些参数并不返回值
void qh_printfacetridges(qhT *qh, FILE *fp, facetT *facet);
// 声明函数：输出给定凸包算法状态的所有面元信息
void qh_printfacets(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);

// 声明函数：在给定的文件流中打印两个面的超平面交点信息
void qh_printhyperplaneintersection(qhT *qh, FILE *fp, facetT *facet1, facetT *facet2,
                                    setT *vertices, realT color[3]);

// 声明函数：在给定的文件流中打印连接两个点的线段的几何信息
void qh_printline3geom(qhT *qh, FILE *fp, pointT *pointA, pointT *pointB, realT color[3]);

// 声明函数：在给定的文件流中打印两个面的邻域信息
void qh_printneighborhood(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetA, facetT *facetB, boolT printall);

// 声明函数：在给定的文件流中打印点的信息（包括坐标和可选的字符串）
void qh_printpoint(qhT *qh, FILE *fp, const char *string, pointT *point);

// 声明函数：在给定的文件流中打印带有标识符的点的信息
void qh_printpointid(qhT *qh, FILE *fp, const char *string, int dim, pointT *point, int id);

// 声明函数：在给定的文件流中打印点的坐标信息
void qh_printpoint3(qhT *qh, FILE *fp, pointT *point);

// 声明函数：在给定的文件流中打印凸包算法的点集信息
void qh_printpoints_out(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);

// 声明函数：在给定的文件流中打印点和向量的信息
void qh_printpointvect(qhT *qh, FILE *fp, pointT *point, coordT *normal, pointT *center, realT radius, realT color[3]);

// 声明函数：在给定的文件流中打印点、向量和中心点的信息
void qh_printpointvect2(qhT *qh, FILE *fp, pointT *point, coordT *normal, pointT *center, realT radius);

// 声明函数：在给定的文件流中打印凸包算法的脊信息
void qh_printridge(qhT *qh, FILE *fp, ridgeT *ridge);

// 声明函数：在给定的文件流中打印以给定点为中心的球体信息
void qh_printspheres(qhT *qh, FILE *fp, setT *vertices, realT radius);

// 声明函数：在给定的文件流中打印凸包算法的 Voronoi 图信息
void qh_printvdiagram(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);

// 声明函数：在给定的文件流中打印凸包算法的第二种 Voronoi 图信息
int qh_printvdiagram2(qhT *qh, FILE *fp, printvridgeT printvridge, setT *vertices, qh_RIDGE innerouter, boolT inorder);

// 声明函数：在给定的文件流中打印凸包算法的顶点信息
void qh_printvertex(qhT *qh, FILE *fp, vertexT *vertex);

// 声明函数：在给定的文件流中打印凸包算法的面元列表信息
void qh_printvertexlist(qhT *qh, FILE *fp, const char* string, facetT *facetlist,
                        setT *facets, boolT printall);

// 声明函数：在给定的文件流中打印凸包算法的顶点信息
void qh_printvertices(qhT *qh, FILE *fp, const char* string, setT *vertices);

// 声明函数：在给定的文件流中打印凸包算法的顶点邻居信息
void qh_printvneighbors(qhT *qh, FILE *fp, facetT* facetlist, setT *facets, boolT printall);

// 声明函数：在给定的文件流中打印凸包算法的 Voronoi 图信息
void qh_printvoronoi(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);

// 声明函数：在给定的文件流中打印凸包算法的顶点法线信息
void qh_printvnorm(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);

// 声明函数：在给定的文件流中打印凸包算法的脊信息
void qh_printvridge(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);

// 声明函数：生成凸包算法的输出信息
void qh_produce_output(qhT *qh);

// 声明函数：生成凸包算法的第二种输出信息
void qh_produce_output2(qhT *qh);

// 声明函数：在三维空间中投影点到另一点的信息
void qh_projectdim3(qhT *qh, pointT *source, pointT *destination);

// 声明函数：读取可行点集的信息
int qh_readfeasible(qhT *qh, int dim, const char *curline);

// 声明函数：读取点集的信息并返回坐标数组
coordT *qh_readpoints(qhT *qh, int *numpoints, int *dimension, boolT *ismalloc);

// 声明函数：设置凸包算法的可行点集信息
void qh_setfeasible(qhT *qh, int dim);

// 声明函数：跳过不需要处理的面元
boolT qh_skipfacet(qhT *qh, facetT *facet);

// 声明函数：跳过不需要处理的文件名
char *qh_skipfilename(qhT *qh, char *filename);

// C++ 兼容性结束声明
#ifdef __cplusplus
} /* extern "C" */
#endif

// 结束 ifndef 指令
#endif /* qhDEFio */
```