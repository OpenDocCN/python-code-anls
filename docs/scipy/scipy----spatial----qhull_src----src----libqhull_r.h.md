# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\libqhull_r.h`

```
/*<html><pre>  -<a                             href="qh-qhull_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   libqhull_r.h
   user-level header file for using qhull.a library

   see qh-qhull_r.htm, qhull_ra.h

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/libqhull_r.h#13 $$Change: 2714 $
   $DateTime: 2019/06/28 16:16:13 $$Author: bbarber $

   includes function prototypes for libqhull_r.c, geom_r.c, global_r.c, io_r.c, user_r.c

   use mem_r.h for mem_r.c
   use qset_r.h for qset_r.c

   see unix_r.c for an example of using libqhull_r.h

   recompile qhull if you change this file
*/

#ifndef qhDEFlibqhull
#define qhDEFlibqhull 1

/*=========================== -included files ==============*/

/* user_r.h first for QHULL_CRTDBG */
#include "user_r.h"      /* user definable constants (e.g., realT). */

#include "mem_r.h"   /* Needed for qhT in libqhull_r.h */
#include "qset_r.h"   /* Needed for QHULL_LIB_CHECK */
/* include stat_r.h after defining boolT.  Needed for qhT in libqhull_r.h */

#include <setjmp.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <stdio.h>

#ifndef __STDC__
#ifndef __cplusplus
#if     !defined(_MSC_VER)
#error  Neither __STDC__ nor __cplusplus is defined.  Please use strict ANSI C or C++ to compile
#error  Qhull.  You may need to turn off compiler extensions in your project configuration.  If
#error  your compiler is a standard C compiler, you can delete this warning from libqhull_r.h
#endif
#endif
#endif

/*============ constants and basic types ====================*/

extern const char qh_version[]; /* defined in global_r.c */
extern const char qh_version2[]; /* defined in global_r.c */

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="coordT">-</a>

  coordT
    coordinates and coefficients are stored as realT (i.e., double)

  notes:
    Qhull works well if realT is 'float'.  If so joggle (QJ) is not effective.

    Could use 'float' for data and 'double' for calculations (realT vs. coordT)
      This requires many type casts, and adjusted error bounds.
      Also C compilers may do expressions in double anyway.
*/
#define coordT realT

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="pointT">-</a>

  pointT
    a point is an array of coordinates, usually qh.hull_dim
    qh_pointid returns
      qh_IDnone if point==0 or qh is undefined
      qh_IDinterior for qh.interior_point
      qh_IDunknown if point is neither in qh.first_point... nor qh.other_points

  notes:
    qh.STOPcone and qh.STOPpoint assume that qh_IDunknown==-1 (other negative numbers indicate points)
    qh_IDunknown is also returned by getid_() for unknown facet, ridge, or vertex
*/
#define pointT coordT
typedef enum
{
    qh_IDnone= -3, qh_IDinterior= -2, qh_IDunknown= -1
}
qh_pointT;
/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="flagT">-</a>

  flagT
    Boolean flag as a bit
*/
// 定义一个类型别名 flagT 为无符号整数，表示布尔标志位
#define flagT unsigned int

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="boolT">-</a>

  boolT
    boolean value, either True or False

  notes:
    needed for portability
    Use qh_False/qh_True as synonyms
*/
// 定义一个类型别名 boolT 为无符号整数，表示布尔值，可以取 True 或 False
// 为了便于移植性，定义了 qh_False 和 qh_True 作为 True 和 False 的同义词
#define boolT unsigned int
#ifdef False
#undef False
#endif
#ifdef True
#undef True
#endif
#define False 0
#define True 1
#define qh_False 0
#define qh_True 1

#include "stat_r.h"  /* needs boolT */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="CENTERtype">-</a>

  qh_CENTER
    to distinguish facet->center
*/
// 定义枚举类型 qh_CENTER，用于区分 facet->center
typedef enum
{
    qh_ASnone = 0,    /* If not MERGING and not VORONOI */
    qh_ASvoronoi,     /* Set by qh_clearcenters on qh_prepare_output, or if not MERGING and VORONOI */
    qh_AScentrum      /* If MERGING (assumed during merging) */
}
qh_CENTER;

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="qh_PRINT">-</a>

  qh_PRINT
    output formats for printing (qh.PRINTout).
    'Fa' 'FV' 'Fc' 'FC'


   notes:
   some of these names are similar to qhT names.  The similar names are only
   used in switch statements in qh_printbegin() etc.
*/
// 定义枚举类型 qh_PRINT，表示用于打印输出的格式
typedef enum {
    qh_PRINTnone = 0,
    qh_PRINTarea, qh_PRINTaverage,           /* 'Fa' 'FV' 'Fc' 'FC' */
    qh_PRINTcoplanars, qh_PRINTcentrums,
    qh_PRINTfacets, qh_PRINTfacets_xridge,   /* 'f' 'FF' 'G' 'FI' 'Fi' 'Fn' */
    qh_PRINTgeom, qh_PRINTids, qh_PRINTinner, qh_PRINTneighbors,
    qh_PRINTnormals, qh_PRINTouter, qh_PRINTmaple, /* 'n' 'Fo' 'i' 'm' 'Fm' 'FM', 'o' */
    qh_PRINTincidences, qh_PRINTmathematica, qh_PRINTmerges, qh_PRINToff,
    qh_PRINToptions, qh_PRINTpointintersect, /* 'FO' 'Fp' 'FP' 'p' 'FQ' 'FS' */
    qh_PRINTpointnearest, qh_PRINTpoints, qh_PRINTqhull, qh_PRINTsize,
    qh_PRINTsummary, qh_PRINTtriangles,      /* 'Fs' 'Ft' 'Fv' 'FN' 'Fx' */
    qh_PRINTvertices, qh_PRINTvneighbors, qh_PRINTextremes,
    qh_PRINTEND
} qh_PRINT;

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="qh_ALL">-</a>

  qh_ALL
    argument flag for selecting everything
*/
// 定义宏 qh_ALL 为 True，表示用于选择所有内容的标志位
#define qh_ALL True
#define qh_NOupper True      /* argument for qh_findbest */
#define qh_IScheckmax True   /* argument for qh_findbesthorizon */
#define qh_ISnewfacets True  /* argument for qh_findbest */
#define qh_RESETvisible True /* argument for qh_resetlists */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="qh_ERR">-</a>

  qh_ERR...
    Qhull exit status codes, for indicating errors
    See: MSG_ERROR (6000) and MSG_WARNING (7000) [user_r.h]
*/
// 定义 Qhull 的退出状态码，用于指示错误
#define qh_ERRnone 0    /* no error occurred during qhull */
#define qh_ERRinput 1   /* input inconsistency */
#define qh_ERRsingular 2 /* 表示奇异输入数据，调用 qh_printhelp_singular 函数 */
#define qh_ERRprec  3    /* 表示精度错误，调用 qh_printhelp_degenerate 函数 */
#define qh_ERRmem   4    /* 表示内存不足，与 mem_r.h 匹配 */
#define qh_ERRqhull 5    /* 表示检测到内部错误，与 mem_r.h 匹配，调用 qh_printhelp_internal 函数 */
#define qh_ERRother 6    /* 表示检测到其他错误 */
#define qh_ERRtopology 7 /* 表示拓扑错误，可能由于几乎相邻的顶点，调用 qh_printhelp_topology 函数 */
#define qh_ERRwide 8     /* 表示宽度面错误，可能由于几乎相邻的顶点，调用 qh_printhelp_wide 函数 */
#define qh_ERRdebug 9   /* 调试代码中的错误，由 qh_errexit 处理 */

/*-<a                             href="qh-qhull_r.htm#TOC"
>--------------------------------</a><a name="qh_FILEstderr">-</a>

qh_FILEstderr
伪造的 stderr，用于区分错误输出和正常输出
用于 C++ 接口，必须重新定义 qh_fprintf_qhull
*/
#define qh_FILEstderr ((FILE *)1)

/* ============ -structures- ====================
   每个以下结构体都由 typedef 定义
   所有 realT 和 coordT 字段都在结构的开头出现
        （否则由于对齐可能会浪费空间）
   将所有标志定义在一起并打包成 32 位数字

   DEFqhT 和 DEFsetT 在 mem_r.h、qset_r.h 和 stat_r.h 中也是如此定义
*/

typedef struct vertexT vertexT;
typedef struct ridgeT ridgeT;
typedef struct facetT facetT;

#ifndef DEFqhT
#define DEFqhT 1
typedef struct qhT qhT;          /* 下面定义 */
#endif

#ifndef DEFsetT
#define DEFsetT 1
typedef struct setT setT;        /* 在 qset_r.h 中定义 */
#endif

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="facetT">-</a>

  facetT
    定义一个面

  注：
   qhull() 函数生成的凸壳是面的列表。

  拓扑信息：
    f.previous,next     面的双向链表，next 始终被定义
    f.vertices          顶点集合
    f.ridges            边集合
    f.neighbors         邻居集合
    f.toporient         如果面具有顶部方向则为 True（否则为底部）

  几何信息：
    f.offset,normal     超平面方程
    f.maxoutside        到外平面的偏移量 -- 所有点在内部
    f.center            用于测试凸性的中心或输出的 Voronoi 中心
    f.simplicial        如果面是单纯形则为 True
    f.flipped           如果面不包括 qh.interior_point 则为 True

  用于构建凸壳：
    f.visible           如果面在可见面列表中（将被删除）则为 True
    f.newfacet          如果面在新创建的面列表中则为 True
    f.coplanarset       与此面共面的点的集合
                        （包括稍后测试的近似内部点）
    f.outsideset        不在此面外部的点的集合
    f.furthestdist      到外部集合最远点的距离
    f.visitid           在循环中标记已访问的面
    f.replace           replacement facet for to-be-deleted, visible facets
    f.samecycle,newcycle cycle of facets for merging into horizon facet
/* QhullFacet.cpp -- 如果增加或删除字段，更新静态初始化器列表以匹配 s_empty_facet */

struct facetT {
#if !qh_COMPUTEfurthest
  coordT   furthestdist;  /* 距离外部集的最远点的距离 */
#endif
#if qh_MAXoutside
  coordT   maxoutside;    /* 点到平面的最大计算距离
                            在 QHULLfinished 之前，这是一个近似值，因为 qh_mergefacet 未始终设置 maxdist
                            实际的外部平面是 +DISTround，而计算的外部平面是 +2*DISTround
                            初始的 maxoutside 是 qh.DISTround，否则距离测试需要考虑 DISTround */
#define qh_MAXnummerge 511 /* 2^9-1 */
                            /* 23 个标志位（最多 23 个，由于 nummerge），在 io_r.c 的 "flags:" 中打印 */
  flagT    tricoplanar:1; /* 如果 TRIangulate 并且是简单面并且与邻居共面，则为真 */
                          /*   所有的 tricoplanar 共享相同的顶点 */
                          /*   所有的 tricoplanar 共享相同的 ->center, ->normal, ->offset, ->maxoutside */
                          /*     ->keepcentrum 是 true 对于拥有者。它具有 ->coplanareset */
                          /*   如果 ->degenerate，不构成面（只有一个逻辑边） */
                          /*   在 qh_triangulate 过程中，f.trivisible 指向原始的面 */
  flagT    newfacet:1;    /* 如果面在 qh.newfacet_list 上（新的/ qh.first_newfacet 或合并的） */
  flagT    visible:1;     /* 如果是可见面（将被删除） */
  flagT    toporient:1;   /* 如果是以顶部方向创建的
                             合并后，使用脊的方向 */
  flagT    simplicial:1;  /* 如果是简单面，->ridges 可能是隐式的 */
  flagT    seen:1;        /* 用于只执行一次的操作，如 visitid */
  flagT    seen2:1;       /* 用于只执行一次的操作，如 visitid */
  flagT    flipped:1;     /* 如果面已翻转 */
  flagT    upperdelaunay:1; /* 如果面是 Delaunay 三角剖分的上凸包 */
  flagT    notfurthest:1; /* 如果外部集的最后一个点不是最远点 */

/*-------- 主要用于输出的标志位 ---------*/
  flagT    good:1;        /* 如果面标记为输出好的 */
  flagT    isarea:1;      /* 如果 facet->f.area 已定义 */
/*-------- flags for merging ------------------*/
/* 定义用于合并的标志位 */
flagT    dupridge:1;  /* True if facet has one or more dupridge in a new facet (qh_matchneighbor),
                         a dupridge has a subridge shared by more than one new facet */
/* 如果面在新面中有一个或多个dupridge，则为True（qh_matchneighbor），
   dupridge具有被多个新面共享的子ridge */

flagT    mergeridge:1; /* True if facet or neighbor has a qh_MERGEridge (qh_mark_dupridges)
                          ->normal defined for mergeridge and mergeridge2 */
/* 如果面或其相邻面有qh_MERGEridge，则为True（qh_mark_dupridges） */

flagT    mergeridge2:1; /* True if neighbor has a qh_MERGEridge (qh_mark_dupridges) */
/* 如果相邻面有qh_MERGEridge，则为True（qh_mark_dupridges） */

flagT    coplanarhorizon:1;  /* True if horizon facet is coplanar at last use */
/* 如果水平面在最后一次使用时共面，则为True */

flagT     mergehorizon:1; /* True if will merge into horizon (its first neighbor w/ f.coplanarhorizon). */
/* 如果将合并到水平面中（其第一个相邻面具有f.coplanarhorizon），则为True */

flagT     cycledone:1;/* True if mergecycle_all already done */
/* 如果已完成mergecycle_all，则为True */

flagT    tested:1;    /* True if facet convexity has been tested (false after merge */
/* 如果已测试面的凸性（在合并后为false），则为True */

flagT    keepcentrum:1; /* True if keep old centrum after a merge, or marks owner for ->tricoplanar
                           Set by qh_updatetested if more than qh_MAXnewcentrum extra vertices
                           Set by qh_mergefacet if |maxdist| > qh.WIDEfacet */
/* 如果在合并后保留旧的中心点或者标记所有者为->tricoplanar，则为True
   如果额外顶点数超过qh_MAXnewcentrum，则由qh_updatetested设置
   如果|maxdist| > qh.WIDEfacet，则由qh_mergefacet设置 */

flagT    newmerge:1;  /* True if facet is newly merged for reducevertices */
/* 如果面是为了reducevertices而新合并的，则为True */

flagT    degenerate:1; /* True if facet is degenerate (degen_mergeset or ->tricoplanar) */
/* 如果面是退化的（degen_mergeset或->tricoplanar），则为True */

flagT    redundant:1;  /* True if facet is redundant (degen_mergeset)
                         Maybe merge degenerate and redundant to gain another flag */
/* 如果面是冗余的（degen_mergeset），也许合并退化和冗余以获得另一个标志位 */

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="ridgeT">-</a>

  ridgeT
    defines a ridge

  notes:
  a ridge is hull_dim-1 simplex between two neighboring facets.  If the
  facets are non-simplicial, there may be more than one ridge between
  two facets.  E.G. a 4-d hypercube has two triangles between each pair
  of neighboring facets.

  topological information:
    vertices            a set of vertices
    top,bottom          neighboring facets with orientation

  geometric information:
    tested              True if ridge is clearly convex
    nonconvex           True if ridge is non-convex
*/
/* 定义一个ridge */

/* QhullRidge.cpp -- Update static initializer list for s_empty_ridge if add or remove fields */
/* 更新s_empty_ridge的静态初始化器列表，如果添加或删除字段 */
/* Define a structure 'ridgeT' representing a ridge in a geometric context */

struct ridgeT {
  setT    *vertices;    /* Set of vertices belonging to this ridge, sorted inversely by ID.
                           NULL if the ridge is degenerate (e.g., matchsame) */
  facetT  *top;         /* The upper facet associated with this ridge */
  facetT  *bottom;      /* The lower facet associated with this ridge.
                           Orientation depends on vertex order (odd/even) and top/bottom */
  unsigned int id;      /* Unique identifier for the ridge, same size as vertex_id, displayed as 'r%d' */
  flagT    seen:1;      /* Flag used to ensure operations are performed only once */
  flagT    tested:1;    /* True when the ridge has been tested for convexity by centrum or opposite vertices */
  flagT    nonconvex:1; /* True if getmergeset detected a non-convex neighbor.
                           Only one ridge between neighbors may have nonconvex flag */
  flagT    mergevertex:1; /* True if pending qh_appendvertexmerge due to
                             qh_maybe_duplicateridge or qh_maybe_duplicateridges
                             disables check for duplicate vertices in qh_checkfacet */
  flagT    mergevertex2:1; /* True if qh_drop_mergevertex of MRGvertices, printed but not used */
  flagT    simplicialtop:1; /* True if top was simplicial (original vertices) */
  flagT    simplicialbot:1; /* True if bottom was simplicial (original vertices).
                               Use qh_test_centrum_merge if top and bot, need to retest since centrum may change */
};

/* Documentation comment for vertexT structure */

/* QhullVertex.cpp -- Update static initializer list for s_empty_vertex if add or remove fields */
/* Define a structure representing a vertex in a vertex list */
struct vertexT {
  vertexT *next;        /* 指向链表中下一个顶点或者指向 vertex_list 的尾部 */
  vertexT *previous;    /* 指向链表中上一个顶点或者为NULL（用于C++接口） */
  pointT  *point;       /* 包含 hull_dim 个坐标的点的指针 */
  setT    *neighbors;   /* 顶点的相邻面集合，通过 qh_vertexneighbors() 初始化 */
  unsigned int id;      /* 唯一标识符，从 1 开始到 qh.vertex_id，哨兵为0，以 'r%d' 格式打印 */
  unsigned int visitid; /* 用于 qh.vertex_visit 的标识符，大小必须匹配 */
  flagT    seen:1;      /* 用于仅执行一次操作的标志 */
  flagT    seen2:1;     /* 另一个用于 seen 的标志 */
  flagT    deleted:1;   /* 将通过 qh.del_vertices 删除的顶点 */
  flagT    delridge:1;  /* 属于已删除的ridge的顶点，通过 qh_reducevertices 清除 */
  flagT    newfacet:1;  /* 如果顶点在新facet中为真，顶点在 qh.newvertex_list 上，并且在 qh.newfacet_list 上有一个facet */
  flagT    partitioned:1; /* 如果已删除的顶点已被分区 */
};

/* -qh global variables -qh ============================ */

/* -<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh">-</a>

  qhT
   Qhull 的所有全局变量都在 qhT 中，包括 qhmemT、qhstatT 和 rbox globals

   此版本的 Qhull 是可重入的，但不是线程安全的。

   不要在同一个 qhT 实例上运行单独的线程。

   QHULL_LIB_CHECK 用于检查程序和相应的 qhull 库是否使用相同类型的头文件构建。

   QHULL_LIB_TYPE 可能的取值为 QHULL_NON_REENTRANT、QHULL_QH_POINTER 或 QHULL_REENTRANT
*/

#define QHULL_NON_REENTRANT 0
#define QHULL_QH_POINTER 1
#define QHULL_REENTRANT 2

#define QHULL_LIB_TYPE QHULL_REENTRANT

#define QHULL_LIB_CHECK qh_lib_check(QHULL_LIB_TYPE, sizeof(qhT), sizeof(vertexT), sizeof(ridgeT), sizeof(facetT), sizeof(setT), sizeof(qhmemT));
#define QHULL_LIB_CHECK_RBOX qh_lib_check(QHULL_LIB_TYPE, sizeof(qhT), sizeof(vertexT), sizeof(ridgeT), sizeof(facetT), 0, 0);

struct qhT {

/* -<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-const">-</a>

  qh constants
    Qhull 的配置标志和常量

  notes:
    用户通过定义标志配置 Qhull。这些标志通过 qh_setflags() 复制到 qh 中。qh-quick_r.htm#options 定义了这些标志。
/* -<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-prec">-</a>

  qh precision constants
    Qhull 的精度常量

  notes:
    精度常量用于 Qhull
    # qh_detroundoff [geom2_r.c] 计算距离和其他计算的最大舍入误差。
    # 它还设置了上述 qh 常量的默认值。
    qh_detroundoff [geom2_r.c] computes the maximum roundoff error for distance
    and other computations.  It also sets default values for the
    qh constants above.
/*
  realT ANGLEround;       /* 角度的最大舍入误差 */
  realT centrum_radius;   /* 凸性的最大中心半径（'Cn' + 2*qh.DISTround） */
  realT cos_max;          /* 凸性的最大余弦值（加上舍入误差） */
  realT DISTround;        /* 距离的最大舍入误差，qh.SETroundoff ('En') 覆盖 qh_distround */
  realT MAXabs_coord;     /* 坐标的最大绝对值 */
  realT MAXlastcoord;     /* qh_scalelast 的最大最后坐标 */
  realT MAXoutside;       /* qh.max_outside/f.maxoutside 的最大目标，qh_addpoint 重新计算，与 qh_MAXoutside 无关 */
  realT MAXsumcoord;      /* 坐标和的最大值 */
  realT MAXwidth;         /* 点坐标的最大直线宽度 */
  realT MINdenom_1;       /* 1/x 的最小绝对值 */
  realT MINdenom;         /* 如果分母 < MINdenom，则使用 divzero */
  realT MINdenom_1_2;     /* 允许归一化的 1/x 的最小绝对值 */
  realT MINdenom_2;       /* 如果分母 < MINdenom_2，则使用 divzero */
  realT MINlastcoord;     /* qh_scalelast 的最小最后坐标 */
  realT *NEARzero;        /* 高斯消元中靠近零的 hull_dim 数组 */
  realT NEARinside;       /* 如果靠近 facet，则保留点以供 qh_check_maxout 使用 */
  realT ONEmerge;         /* 合并单纯面的最大距离 */
  realT outside_err;      /* 应用程序中的 epsilon，用于共面点，qh_check_bestdist() 和 qh_check_points() 如果点在外部则报告错误 */
  realT WIDEfacet;        /* 用于跳过面积计算中的宽面和锁定中心 */
  boolT NARROWhull;       /* 如果角度 < qh_MAXnarrow，则在 qh_initialhull 中设置 */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-codetern">-</a>

  qh internal constants
    Qhull 的内部常量
*/
  char qhull[sizeof("qhull")]; /* 用于调试时检查所有权的 "qhull" */
  jmp_buf errexit;        /* qh_errexit 的退出标签，由 setjmp() 和 NOerrexit 定义 */
  char    jmpXtra[40];    /* 如果 jmp_buf 被编译器错误地定义了，额外的字节 */
  jmp_buf restartexit;    /* qh_errexit 的重启标签，由 setjmp() 和 ALLOWrestart 定义 */
  char    jmpXtra2[40];   /* 如果 jmp_buf 被编译器错误地定义了，额外的字节 */
  FILE *  fin;            /* 输入文件指针，由 qh_initqhull_start2 初始化 */
  FILE *  fout;           /* 输出文件指针 */
  FILE *  ferr;           /* 错误文件指针 */
  pointT *interior_point; /* 初始单纯形的中心点 */
  int     normal_size;    /* 面法线和点坐标的字节数 */
  int     center_size;    /* Voronoi 中心的字节数 */
  int     TEMPsize;       /* 小型临时集的大小（在快速内存中） */
*/
# 标记：qh facet and vertex lists
# 连接到：qh-globa_r.htm#TOC
# qh_facet_and_vertex_lists 函数的定义和说明
def qh_facet_and_vertex_lists():
    # 定义并初始化各种列表和标识符
    # facets - 面的列表
    # new_facets - 新面的列表
    # visible_facets - 可见面的列表
    # vertices - 顶点的列表
    # new_vertices - 新顶点的列表
    # 包括计数、下一个 ID 和跟踪 ID
    # 参见 qh_resetlists() 函数
    pass
facetT *facet_list;     /* 指向第一个 facet 的指针 */
facetT *facet_tail;     /* 指向 facet_list 结尾的指针（带有 id 0 和 next==NULL 的虚拟 facet） */
facetT *facet_next;     /* 用于 buildhull() 的下一个 facet，
                           先前的 facets 没有外部集合
                           NARROWhull 中，先前的 facets 可能有用于 qh_outcoplanar 的共面外部集合 */
facetT *newfacet_list;  /* 新 facet 的列表，添加到 facet_list 的末尾
                           qh_postmerge 将 newfacet_list 设置为 facet_list */
facetT *visible_list;   /* 可见 facet 的列表，位于 newfacet_list 之前，
                           如果 !facet->visible，则与 newfacet_list 相同
                           qh_findhorizon 在 facet_list 末尾设置 visible_list
                           qh_willdelete 在 visible_list 前插入
                           qh_triangulate 在 facet_list 末尾将镜像 facets 附加到 visible_list
                           qh_postmerge 将 visible_list 设置为 facet_list
                           qh_deletevisible 删除可见 facets */
int       num_visible;  /* 当前可见 facets 的数量 */
unsigned int tracefacet_id; /* 在初始化时设置，然后随时可以打印 */
facetT  *tracefacet;    /* 在 newfacet/mergefacet 中设置，在 delfacet 和 qh_errexit 中取消设置 */
unsigned int traceridge_id; /* 在初始化时设置，然后随时可以打印 */
ridgeT  *traceridge;    /* 在 newridge 中设置，在 delridge、errexit、errexit2、makenew_nonsimplicial、mergecycle_ridges 中取消设置 */
unsigned int tracevertex_id; /* 在 buildtracing 中设置，随时可以打印 */
vertexT *tracevertex;   /* 在 newvertex 中设置，在 delvertex 和 qh_errexit 中取消设置 */
vertexT *vertex_list;   /* 所有顶点的列表，到 vertex_tail */
vertexT *vertex_tail;   /* vertex_list 的结尾（带有 ID 0 和 next NULL 的虚拟 vertex） */
vertexT *newvertex_list; /* newfacet_list 中顶点的列表，到 vertex_tail
                            所有顶点都设置了 'newfacet' */
int   num_facets;       /* facet_list 中 facets 的数量
                           包括可见 faces (num_visible) */
int   num_vertices;     /* facet_list 中顶点的数量 */
int   num_outside;      /* outsidesets 中点的数量（用于跟踪和 RANDOMoutside）
                           包括 NARROWhull/qh_outcoplanar() 中的共面 outsideset 点 */
int   num_good;         /* 好的 facets 的数量（在 qh_findgood_all 或 qh_markkeep 后） */
unsigned int facet_id;  /* 下一个新 facet 的 ID，从 newfacet() 获取 */
unsigned int ridge_id;  /* 下一个新 ridge 的 ID，从 newridge() 获取 */
unsigned int vertex_id; /* 下一个新 vertex 的 ID，从 newvertex() 获取 */
unsigned int first_newfacet; /* 对于 qh_buildcone，first_newfacet 的 ID，如果没有则为 0 */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-var">-</a>

  qh global variables
    # 定义最小距离和最大距离、下一次访问的 ID、多个标志以及其他全局变量。
    # 在 qh_initbuild 或者如果在 qh_buildhull 中使用则在 qh_maxmin 中初始化。
    defines minimum and maximum distances, next visit ids, several flags,
    and other global variables.
    initialize in qh_initbuild or qh_maxmin if used in qh_buildhull
/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-set">-</a>

  qh global sets
    defines sets for merging, initial simplex, hashing, extra input points,
    and deleted vertices
*/
// 临时的合并集，用于待处理的合并操作
setT *facet_mergeset;
// 临时的退化和冗余合并集
setT *degen_mergeset;
// 临时的顶点合并集
setT *vertex_mergeset;
// 用于匹配 qh_matchfacets 中的山脊的哈希表，大小由 setsize() 决定
setT *hash_table;
// 额外的输入点集合
setT *other_points;
// 待删除的顶点集合，在检查面片时将 v.deleted 设置为 true
setT *del_vertices;

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-buf">-</a>

  qh global buffers
    defines buffers for maxtrix operations, input, and error messages
*/
// 为 geom_r.c 定义的 (dim+1)Xdim 矩阵
coordT *gm_matrix;
// gm_matrix 行的数组
coordT **gm_row;
// malloc 分配的最大输入行 + 1 的字符数组
char* line;
// 最大行数
int maxline;
// 用于半空间的 malloc 分配的输入数组（qh.normal_size + coordT）
coordT *half_space;
// 点的 malloc 分配输入数组
coordT *temp_malloc;

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-static">-</a>

  qh static variables
    defines static variables for individual functions

  notes:
    do not use 'static' within a function.  Multiple instances of qhull
    may exist.

    do not assume zero initialization, 'QPn' may cause a restart
*/
// 在 qh_errexit 期间为 true（防止重复调用）。参见 qh.NOerrexit
boolT ERREXITcalled;
// 用于 qh_printcentrum 的标志
boolT firstcentrum;
// 在 io、跟踪或统计期间保存 RANDOMdist 标志
boolT old_randomdist;
// 用于搜索 qh_findbesthorizon() 的共面面片集合
setT *coplanarfacetset;
// 用于 qh_setdelaunay 的 qh_scalelast 参数
realT last_low;
realT last_high;
realT last_newhigh;
// 用于 qh_buildtracing 的最后 CPU 时间
realT lastcpu;
// 最后一次的面片数目
int lastfacets;
// 最后一次的合并数目 (zzval_(Ztotmerge))
int lastmerges;
// 最后一次的平面数目 (zzval_(Zsetplane))
int lastplanes;
// 最后一次的距离平面数目 (zzval_(Zdistplane))
int lastdist;
// 最后一次的面片 ID
unsigned int lastreport;
// 用于 qh_tracemerging 的合并报告
int mergereport;
// 在 save_qhull 中保存 qh->qhmem.tempstack 的旧临时堆栈
setT *old_tempstack;
// 4OFF 输出的山脊数目 (qh_printbegin 等)
int ridgeoutnum;

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-const">-</a>

  qh memory management, rbox globals, and statistics

  Replaces global data structures defined for libqhull
*/
/*
  int     last_random;    /* 上次从 qh_rand (random_r.c) 获取的随机数 */
  jmp_buf rbox_errexit;   /* 来自 rboxlib_r.c 的 errexit，仅由 qh_rboxpoints() 定义 */
  char    jmpXtra3[40];   /* 额外的字节，以防编译器错误定义 jmp_buf */
  int     rbox_isinteger; /* 表示 rbox 是否为整数 */
  double  rbox_out_offset; /* rbox 输出的偏移量 */
  void *  cpp_object;     /* C++ 指针，目前由 RboxPoints.qh_fprintf_rbox 使用 */

  /* 最后，由 qh_initqhull_start2 (global_r.c) 初始化为零 */
  qhmemT  qhmem;          /* Qhull 管理的内存 (mem_r.h) */
  /* 在 qhmem 后面，其大小取决于统计数据的数量 */
  qhstatT qhstat;         /* Qhull 统计信息 (stat_r.h) */
};

/*=========== -宏定义- =========================*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="otherfacet_">-</a>

  otherfacet_(ridge, facet)
    返回给定面（facet）的一个相邻面（neighbor）, ridge 为面的一个边
*/
#define otherfacet_(ridge, facet) \
                        (((ridge)->top == (facet)) ? (ridge)->bottom : (ridge)->top)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="getid_">-</a>

  getid_(p)
    返回面（facet）、边（ridge）或顶点（vertex）的 ID
    如果 p 为 NULL，则返回 qh_IDunknown(-1)
    如果 p 指向 facet_tail 或 vertex_tail，则返回 0
*/
#define getid_(p)       ((p) ? (int)((p)->id) : qh_IDunknown)

/*============== FORALL 宏 ===================*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLfacets">-</a>

  FORALLfacets { ... }
    将 'facet' 赋值为 qh.facet_list 中的每一个面（facet）

  注意:
    使用 'facetT *facet;'
    假定最后一个面是一个哨兵（sentinel）
    假定 qh 已定义

  参见:
    FORALLfacet_( facetlist )
*/
#define FORALLfacets for (facet=qh->facet_list;facet && facet->next;facet=facet->next)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLpoints">-</a>

  FORALLpoints { ... }
    将 'point' 赋值为 qh.first_point 和 qh.num_points 中的每一个点（point）

  注意:
    假定 qh 已定义

  声明:
    coordT *point, *pointtemp;
*/
#define FORALLpoints FORALLpoint_(qh, qh->first_point, qh->num_points)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLpoint_">-</a>

  FORALLpoint_(qh, points, num) { ... }
    将 'point' 赋值为 points 数组中的每一个点（point），共 num 个点

  声明:
    coordT *point, *pointtemp;
*/
#define FORALLpoint_(qh, points, num) for (point=(points), \
      pointtemp= (points)+qh->hull_dim*(num); point < pointtemp; point += qh->hull_dim)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLvertices">-</a>

  FORALLvertices { ... }
    将 'vertex' 赋值为 qh.vertex_list 中的每一个顶点（vertex）

  声明:
    vertexT *vertex;

  注意:
    假定 qh.vertex_list 以 NULL 或哨兵（v.next==NULL）结尾
    假定 qh 已定义
*/
/*
#define FORALLvertices for (vertex=qh->vertex_list;vertex && vertex->next;vertex= vertex->next)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHfacet_">-</a>

  FOREACHfacet_( facets ) { ... }
    assign 'facet' to each facet in facets

  declare:
    facetT *facet, **facetp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHfacet_(facets)    FOREACHsetelement_(facetT, facets, facet)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighbor_">-</a>

  FOREACHneighbor_( facet ) { ... }
    assign 'neighbor' to each neighbor in facet->neighbors

  FOREACHneighbor_( vertex ) { ... }
    assign 'neighbor' to each neighbor in vertex->neighbors

  declare:
    facetT *neighbor, **neighborp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHneighbor_(facet)  FOREACHsetelement_(facetT, facet->neighbors, neighbor)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHpoint_">-</a>

  FOREACHpoint_( points ) { ... }
    assign 'point' to each point in points set

  declare:
    pointT *point, **pointp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHpoint_(points)    FOREACHsetelement_(pointT, points, point)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHridge_">-</a>

  FOREACHridge_( ridges ) { ... }
    assign 'ridge' to each ridge in ridges set

  declare:
    ridgeT *ridge, **ridgep;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHridge_(ridges)    FOREACHsetelement_(ridgeT, ridges, ridge)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertex_">-</a>

  FOREACHvertex_( vertices ) { ... }
    assign 'vertex' to each vertex in vertices set

  declare:
    vertexT *vertex, **vertexp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertex_(vertices) FOREACHsetelement_(vertexT, vertices,vertex)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHfacet_i_">-</a>

  FOREACHfacet_i_(qh, facets ) { ... }
    assign 'facet' and 'facet_i' for each facet in facets set

  declare:
    facetT *facet;
    int     facet_n, facet_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHfacet_i_(qh, facets)    FOREACHsetelement_i_(qh, facetT, facets, facet)


注释：
/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighbor_i_">-</a>

  FOREACHneighbor_i_(qh, facet ) { ... }
    assign 'neighbor' and 'neighbor_i' for each neighbor in facet->neighbors

  declare:
    facetT *neighbor;
    int     neighbor_n, neighbor_i;

  notes:
    see <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
    for facet neighbors of vertex, need to define a new macro
*/
#define FOREACHneighbor_i_(qh, facet)  FOREACHsetelement_i_(qh, facetT, facet->neighbors, neighbor)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHpoint_i_">-</a>

  FOREACHpoint_i_(qh, points ) { ... }
    assign 'point' and 'point_i' for each point in points set

  declare:
    pointT *point;
    int     point_n, point_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHpoint_i_(qh, points)    FOREACHsetelement_i_(qh, pointT, points, point)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHridge_i_">-</a>

  FOREACHridge_i_(qh, ridges ) { ... }
    assign 'ridge' and 'ridge_i' for each ridge in ridges set

  declare:
    ridgeT *ridge;
    int     ridge_n, ridge_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHridge_i_(qh, ridges)    FOREACHsetelement_i_(qh, ridgeT, ridges, ridge)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertex_i_">-</a>

  FOREACHvertex_i_(qh, vertices ) { ... }
    assign 'vertex' and 'vertex_i' for each vertex in vertices set

  declare:
    vertexT *vertex;
    int     vertex_n, vertex_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHvertex_i_(qh, vertices) FOREACHsetelement_i_(qh, vertexT, vertices, vertex)

#ifdef __cplusplus
extern "C" {
#endif

/********* -libqhull_r.c prototypes (duplicated from qhull_ra.h) **********************/

void    qh_qhull(qhT *qh);
boolT   qh_addpoint(qhT *qh, pointT *furthest, facetT *facet, boolT checkdist);
void    qh_errexit2(qhT *qh, int exitcode, facetT *facet, facetT *otherfacet);
void    qh_printsummary(qhT *qh, FILE *fp);

/********* -user_r.c prototypes (alphabetical) **********************/

void    qh_errexit(qhT *qh, int exitcode, facetT *facet, ridgeT *ridge);
void    qh_errprint(qhT *qh, const char* string, facetT *atfacet, facetT *otherfacet, ridgeT *atridge, vertexT *atvertex);
int     qh_new_qhull(qhT *qh, int dim, int numpoints, coordT *points, boolT ismalloc,
                char *qhull_cmd, FILE *outfile, FILE *errfile);
void    qh_printfacetlist(qhT *qh, facetT *facetlist, setT *facets, boolT printall);
void    qh_printhelp_degenerate(qhT *qh, FILE *fp);
void    qh_printhelp_internal(qhT *qh, FILE *fp);


注释：
// 打印窄凸壳帮助信息，包括最小角度限制
void qh_printhelp_narrowhull(qhT *qh, FILE *fp, realT minangle);

// 打印奇异结构帮助信息
void qh_printhelp_singular(qhT *qh, FILE *fp);

// 打印拓扑结构帮助信息
void qh_printhelp_topology(qhT *qh, FILE *fp);

// 打印宽凸壳帮助信息
void qh_printhelp_wide(qhT *qh, FILE *fp);

// 用户自定义内存大小配置
void qh_user_memsizes(qhT *qh);

/********* -usermem_r.c prototypes (alphabetical) **********************/

// 退出程序，指定退出码
void qh_exit(int exitcode);

// 在标准错误输出流上打印格式化消息
void qh_fprintf_stderr(int msgcode, const char *fmt, ... );

// 释放内存
void qh_free(void *mem);

// 分配指定大小的内存
void *qh_malloc(size_t size);

/********* -userprintf_r.c and userprintf_rbox_r.c prototypes **********************/

// 在文件流上打印格式化消息
void qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );

// 在文件流上打印格式化消息（用于rbox命令的特定版本）
void qh_fprintf_rbox(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );

/***** -geom_r.c/geom2_r.c/random_r.c prototypes (duplicated from geom_r.h, random_r.h) ****************/

// 寻找最佳的凸包面，返回该面
facetT *qh_findbest(qhT *qh, pointT *point, facetT *startfacet,
                    boolT bestoutside, boolT newfacets, boolT noupper,
                    realT *dist, boolT *isoutside, int *numpart);

// 寻找新的最佳凸包面，返回该面
facetT *qh_findbestnew(qhT *qh, pointT *point, facetT *startfacet,
                       realT *dist, boolT bestoutside, boolT *isoutside, int *numpart);

// 执行Gram-Schmidt正交化过程
boolT qh_gram_schmidt(qhT *qh, int dim, realT **rows);

// 计算凸包面的外部和内部平面
void qh_outerinner(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane);

// 打印qhull的摘要信息
void qh_printsummary(qhT *qh, FILE *fp);

// 处理输入点集合，可能包括旋转、缩放等
void qh_projectinput(qhT *qh);

// 生成随机矩阵
void qh_randommatrix(qhT *qh, realT *buffer, int dim, realT **row);

// 旋转输入点集合
void qh_rotateinput(qhT *qh, realT **rows);

// 缩放输入点集合
void qh_scaleinput(qhT *qh);

// 设置Delaunay三角化，给定点集合和维度
void qh_setdelaunay(qhT *qh, int dim, int count, pointT *points);

// 设置所有半空间，返回半空间数组
coordT *qh_sethalfspace_all(qhT *qh, int dim, int count, coordT *halfspaces, pointT *feasible);

/***** -global_r.c prototypes (alphabetical) ***********************/

// 返回当前系统时间（时钟）
unsigned long qh_clock(qhT *qh);

// 检查qhull的标志，处理指定命令和隐藏标志
void qh_checkflags(qhT *qh, char *command, char *hiddenflags);

// 清空qhull的输出标志
void qh_clear_outputflags(qhT *qh);

// 释放qhull使用的所有缓冲区
void qh_freebuffers(qhT *qh);

// 释放qhull使用的所有资源
void qh_freeqhull(qhT *qh, boolT allmem);

// 初始化qhull，设置输入输出文件流、错误文件流以及其他参数
void qh_init_A(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile, int argc, char *argv[]);

// 初始化qhull，使用指定点集合和维度
void qh_init_B(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);

// 初始化qhull命令参数
void qh_init_qhull_command(qhT *qh, int argc, char *argv[]);

// 初始化qhull的缓冲区，使用指定点集合和维度
void qh_initbuffers(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);

// 初始化qhull的标志，使用指定命令
void qh_initflags(qhT *qh, char *command);

// 初始化qhull的缓冲区
void qh_initqhull_buffers(qhT *qh);

// 初始化qhull的全局参数，使用指定点集合和维度
void qh_initqhull_globals(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);

// 初始化qhull的内存
void qh_initqhull_mem(qhT *qh);

// 初始化qhull的输出标志
void qh_initqhull_outputflags(qhT *qh);

// 初始化qhull的起始状态，设置输入输出文件流、错误文件流
void qh_initqhull_start(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile);

// 初始化qhull的起始状态，设置输入输出文件流、错误文件流（备用版本）
void qh_initqhull_start2(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile);

// 初始化qhull的阈值，使用指定命令
void qh_initthresholds(qhT *qh, char *command);

// 检查qhull库的一致性，检查各种结构体和内存大小
void qh_lib_check(int qhullLibraryType, int qhTsize, int vertexTsize, int ridgeTsize, int facetTsize, int setTsize, int qhmemTsize);
// 定义函数原型，声明了一个接受 qhT 结构体指针、const char 指针和两个整型指针作为参数的函数 qh_option
void qh_option(qhT *qh, const char *option, int *i, realT *r);

// 定义函数原型，声明了一个接受 qhT 结构体指针和 FILE 结构体指针作为参数的函数 qh_zero
void qh_zero(qhT *qh, FILE *errfile);

/********* -io_r.c prototypes (duplicated from io_r.h) ***********************/

// 定义函数原型，声明了一个接受 qhT 结构体指针和 unsigned int 类型参数作为参数的函数 qh_dfacet
void qh_dfacet(qhT *qh, unsigned int id);

// 定义函数原型，声明了一个接受 qhT 结构体指针和 unsigned int 类型参数作为参数的函数 qh_dvertex
void qh_dvertex(qhT *qh, unsigned int id);

// 定义函数原型，声明了一个接受 qhT 结构体指针、FILE 结构体指针、qh_PRINT 枚举类型、两个 facetT 结构体指针和一个 boolT 类型参数作为参数的函数 qh_printneighborhood
void qh_printneighborhood(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetA, facetT *facetB, boolT printall);

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_produce_output
void qh_produce_output(qhT *qh);

// 定义函数原型，声明了一个接受 qhT 结构体指针、int 整型指针、int 整型指针、boolT 类型指针作为参数的函数 qh_readpoints
coordT *qh_readpoints(qhT *qh, int *numpoints, int *dimension, boolT *ismalloc);

/********* -mem_r.c prototypes (duplicated from mem_r.h) **********************/

// 定义函数原型，声明了一个接受 qhT 结构体指针和 FILE 结构体指针作为参数的函数 qh_meminit
void qh_meminit(qhT *qh, FILE *ferr);

// 定义函数原型，声明了一个接受 qhT 结构体指针、int 整型指针和 int 整型指针作为参数的函数 qh_memfreeshort
void qh_memfreeshort(qhT *qh, int *curlong, int *totlong);

/********* -poly_r.c/poly2_r.c prototypes (duplicated from poly_r.h) **********************/

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_check_output
void qh_check_output(qhT *qh);

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_check_points
void qh_check_points(qhT *qh);

// 定义函数原型，声明了一个接受 qhT 结构体指针、facetT 结构体指针、setT 结构体指针和 boolT 类型参数作为参数的函数 qh_facetvertices
setT *qh_facetvertices(qhT *qh, facetT *facetlist, setT *facets, boolT allfacets);

// 定义函数原型，声明了一个接受 qhT 结构体指针、pointT 结构体指针、boolT 类型和 realT 类型指针作为参数的函数 qh_findbestfacet
facetT *qh_findbestfacet(qhT *qh, pointT *point, boolT bestoutside, realT *bestdist, boolT *isoutside);

// 定义函数原型，声明了一个接受 qhT 结构体指针、facetT 结构体指针、pointT 结构体指针和 realT 类型指针作为参数的函数 qh_nearvertex
vertexT *qh_nearvertex(qhT *qh, facetT *facet, pointT *point, realT *bestdistp);

// 定义函数原型，声明了一个接受 qhT 结构体指针和 int 整型参数作为参数的函数 qh_point
pointT *qh_point(qhT *qh, int id);

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_pointfacet
setT *qh_pointfacet(qhT *qh /* qh.facet_list */);

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_pointid
int qh_pointid(qhT *qh, pointT *point);

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_pointvertex
setT *qh_pointvertex(qhT *qh /* qh.facet_list */);

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_setvoronoi_all
void qh_setvoronoi_all(qhT *qh);

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_triangulate
void qh_triangulate(qhT *qh /* qh.facet_list */);

/********* -rboxlib_r.c prototypes **********************/
// 定义函数原型，声明了一个接受 qhT 结构体指针和 char 指针作为参数的函数 qh_rboxpoints
int qh_rboxpoints(qhT *qh, char* rbox_command);

// 定义函数原型，声明了一个接受 qhT 结构体指针和 int 整型作为参数的函数 qh_errexit_rbox
void qh_errexit_rbox(qhT *qh, int exitcode);

/********* -stat_r.c prototypes (duplicated from stat_r.h) **********************/

// 定义函数原型，声明了一个接受 qhT 结构体指针作为参数的函数 qh_collectstatistics
void qh_collectstatistics(qhT *qh);

// 定义函数原型，声明了一个接受 qhT 结构体指针、FILE 结构体指针和 const char 指针作为参数的函数 qh_printallstatistics
void qh_printallstatistics(qhT *qh, FILE *fp, const char *string);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFlibqhull */
```