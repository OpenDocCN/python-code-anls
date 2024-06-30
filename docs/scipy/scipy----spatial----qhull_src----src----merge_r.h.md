# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\merge_r.h`

```
/*
  merge_r.h
  merge_r.c 的头文件

  参见 qh-merge_r.htm 和 merge_r.c

  版权所有 1993-2019 年 C.B. Barber.
  $Id: //main/2019/qhull/src/libqhull_r/merge_r.h#1 $$Change: 2661 $
  $DateTime: 2019/05/24 20:09:58 $$Author: bbarber $
*/

#ifndef qhDEFmerge
#define qhDEFmerge 1

#include "libqhull_r.h"


/*============ -constants- ==============*/

/* qh_ANGLEnone
   表示 mergeT->angle 缺失的角度 */
#define qh_ANGLEnone 2.0

/* MRG... (mergeType)
   指示合并的类型 (mergeT->type)
   MRGcoplanar...MRGtwisted 由 qh_test_centrum_merge, qh_test_nonsimplicial_merge 设置 */
typedef enum {  /* 必须与 mergetypes[] 匹配 */
  MRGnone= 0,           /* 无合并 */
  MRGcoplanar,          /* (1) 平面合并，如果中心 ('Cn') 或顶点未明确在邻居上方或下方 */
  MRGanglecoplanar,     /* (2) 角度合并，如果角度 ('An') 是共面的 */
  MRGconcave,           /* (3) 凹陷脊 */
  MRGconcavecoplanar,   /* (4) 凹陷和共面脊，一侧凹陷，另一侧共面 */
  MRGtwisted,           /* (5) 扭曲脊，既有凹也有凸，facet1 更宽 */
  MRGflip,              /* (6) 翻转的 facet，如果 qh.interior_point 在 facet 上方，且 facet1 == facet2 */
  MRGdupridge,          /* (7) 重复脊，如果邻居超过两个。由 qh_mark_dupridges 设置 */
  MRGsubridge,          /* (8) 合并收紧顶点以移除 MRGdupridge 的子脊 */
  MRGvertices,          /* (9) 合并收紧顶点以移除一个 facet 的具有相同顶点的脊 */
  MRGdegen,             /* (10) 退化 facet（邻居不足）facet1 == facet2 */
  MRGredundant,         /* (11) 冗余 facet（顶点子集） */
  MRGmirror,            /* (12) 镜像 facet：由于 qh_triangulate 中的 null facets，具有相同顶点 */
  MRGcoplanarhorizon,   /* (13) 新 facet 与水平面共面（qh_mergecycle_all） */
  ENDmrg
} mergeType;

#endif /* qhDEFmerge */
/*
 * 宏定义 qh_MERGEapex
 * 用于 qh_mergefacet() 指示一个顶点合并的标志
 */
#define qh_MERGEapex     True

/*
 * 结构体定义 mergeT
 * 用于合并几何体的结构
 */
typedef struct mergeT mergeT;
struct mergeT {         /* 在 qh_appendmergeset 中初始化 */
  realT   angle;        /* 两个面的法线之间的余弦值，null 或者直角是 0.0，共面是 1.0，锐角是 -1.0 */
  realT   distance;     /* 顶点之间的距离的绝对值，重心和面之间的距离，或者顶点和面之间的距离 */
  facetT *facet1;       /* 将要合并的第一个面 */
  facetT *facet2;       /* 将要合并的第二个面 */
  vertexT *vertex1;     /* 将要合并的第一个顶点，用于 MRGsubridge 或者 MRGvertices */
  vertexT *vertex2;     /* 将要合并的第二个顶点 */
  ridgeT  *ridge1;      /* MRGvertices 解决的重复边 */
  ridgeT  *ridge2;      /* 如果任一条边被删除，则合并被删除 (qh_delridge) */
  mergeType mergetype;  /* 合并类型 */
};

/*
 * 宏定义 FOREACHmerge_
 * 用于遍历 merges 集合中的每一个 merge
 * 注意：
 * 使用 'mergeT *merge, **mergep;'
 * 如果使用 qh_mergefacet()，因为 qh.facet_mergeset 可能会改变，所以重新开始或者使用 qh_setdellast()
 * 参见 FOREACHsetelement_
 */
#define FOREACHmerge_(merges) FOREACHsetelement_(mergeT, merges, merge)

/*
 * 宏定义 FOREACHmergeA_
 * 用于遍历 vertices 集合中的每一个 merge
 * 注意：
 * 使用 'mergeT *mergeA, *mergeAp;'
 * 参见 FOREACHsetelement_
 */
#define FOREACHmergeA_(merges) FOREACHsetelement_(mergeT, merges, mergeA)

/*
 * 宏定义 FOREACHmerge_i_
 * 用于遍历 mergeset 集合中的每一个 merge，并声明 merge 和 merge_i
 * 声明：
 * mergeT *merge;
 * int     merge_n, merge_i;
 * 参见 FOREACHsetelement_i_
 */
#define FOREACHmerge_i_(qh, mergeset) FOREACHsetelement_i_(qh, mergeT, mergeset, merge)

/*
 * 函数原型声明 qh_premerge
 * 参数：
 * qhT *qh - Qhull 的指针
 * apexpointid - 顶点 id
 * maxcentrum - 最大重心
 * maxangle - 最大角度
 * 用于合并前的操作
 */
void    qh_premerge(qhT *qh, int apexpointid, realT maxcentrum, realT maxangle);

/*
 * 函数原型声明 qh_postmerge
 * 参数：
 * qhT *qh - Qhull 的指针
 * reason - 原因字符串
 * maxcentrum - 最大重心
 * maxangle - 最大角度
 * vneighbors - 是否有相邻顶点
 * 用于合并后的操作
 */
void    qh_postmerge(qhT *qh, const char *reason, realT maxcentrum, realT maxangle,
             boolT vneighbors);

#ifdef __cplusplus
extern "C" {
#endif
void    qh_all_merges(qhT *qh, boolT othermerge, boolT vneighbors);
# 声明函数 qh_all_merges，接受一个 qh 结构体指针和两个布尔值参数

void    qh_all_vertexmerges(qhT *qh, int apexpointid, facetT *facet, facetT **retryfacet);
# 声明函数 qh_all_vertexmerges，接受一个 qh 结构体指针，一个整数参数，一个 facet 结构体指针参数，一个指向 facet 结构体指针的指针参数

void    qh_appendmergeset(qhT *qh, facetT *facet, facetT *neighbor, mergeType mergetype, coordT dist, realT angle);
# 声明函数 qh_appendmergeset，接受一个 qh 结构体指针，一个 facet 结构体指针参数，一个 neighbor facet 结构体指针参数，一个 mergeType 枚举类型参数，一个 coordT 类型参数，一个 realT 类型参数

void    qh_appendvertexmerge(qhT *qh, vertexT *vertex, vertexT *destination, mergeType mergetype, realT distance, ridgeT *ridge1, ridgeT *ridge2);
# 声明函数 qh_appendvertexmerge，接受一个 qh 结构体指针，一个 vertexT 结构体指针参数，一个 destination vertexT 结构体指针参数，一个 mergeType 枚举类型参数，一个 realT 类型参数，一个 ridgeT 结构体指针参数，一个 ridgeT 结构体指针参数

setT   *qh_basevertices(qhT *qh, facetT *samecycle);
# 声明函数 qh_basevertices，返回一个 setT 结构体指针，接受一个 qh 结构体指针参数和一个 facet 结构体指针参数

void    qh_check_dupridge(qhT *qh, facetT *facet1, realT dist1, facetT *facet2, realT dist2);
# 声明函数 qh_check_dupridge，接受一个 qh 结构体指针，两个 facet 结构体指针参数，两个 realT 类型参数

void    qh_checkconnect(qhT *qh /* qh.new_facets */);
# 声明函数 qh_checkconnect，接受一个 qh 结构体指针，包含一个注释参数

void    qh_checkdelfacet(qhT *qh, facetT *facet, setT *mergeset);
# 声明函数 qh_checkdelfacet，接受一个 qh 结构体指针，一个 facet 结构体指针参数，一个 setT 结构体指针参数

void    qh_checkdelridge(qhT *qh /* qh.visible_facets, vertex_mergeset */);
# 声明函数 qh_checkdelridge，接受一个 qh 结构体指针，包含一个注释参数

boolT   qh_checkzero(qhT *qh, boolT testall);
# 声明函数 qh_checkzero，返回一个 boolT 类型，接受一个 qh 结构体指针参数和一个布尔值参数

int     qh_compare_anglemerge(const void *p1, const void *p2);
# 声明函数 qh_compare_anglemerge，返回一个整数，接受两个常量指针参数

int     qh_compare_facetmerge(const void *p1, const void *p2);
# 声明函数 qh_compare_facetmerge，返回一个整数，接受两个常量指针参数

int     qh_comparevisit(const void *p1, const void *p2);
# 声明函数 qh_comparevisit，返回一个整数，接受两个常量指针参数

void    qh_copynonconvex(qhT *qh, ridgeT *atridge);
# 声明函数 qh_copynonconvex，接受一个 qh 结构体指针参数和一个 ridgeT 结构体指针参数

void    qh_degen_redundant_facet(qhT *qh, facetT *facet);
# 声明函数 qh_degen_redundant_facet，接受一个 qh 结构体指针参数和一个 facet 结构体指针参数

void    qh_drop_mergevertex(qhT *qh, mergeT *merge);
# 声明函数 qh_drop_mergevertex，接受一个 qh 结构体指针参数和一个 mergeT 结构体指针参数

void    qh_delridge_merge(qhT *qh, ridgeT *ridge);
# 声明函数 qh_delridge_merge，接受一个 qh 结构体指针参数和一个 ridgeT 结构体指针参数

vertexT *qh_find_newvertex(qhT *qh, vertexT *oldvertex, setT *vertices, setT *ridges);
# 声明函数 qh_find_newvertex，返回一个 vertexT 结构体指针，接受一个 qh 结构体指针参数，一个 vertexT 结构体指针参数，两个 setT 结构体指针参数

vertexT *qh_findbest_pinchedvertex(qhT *qh, mergeT *merge, vertexT *apex, vertexT **pinchedp, realT *distp /* qh.newfacet_list */);
# 声明函数 qh_findbest_pinchedvertex，返回一个 vertexT 结构体指针，接受一个 qh 结构体指针参数，一个 mergeT 结构体指针参数，一个 vertexT 结构体指针参数，一个指向 vertexT 结构体指针的指针参数，一个 realT 类型参数，包含一个注释参数

vertexT *qh_findbest_ridgevertex(qhT *qh, ridgeT *ridge, vertexT **pinchedp, coordT *distp);
# 声明函数 qh_findbest_ridgevertex，返回一个 vertexT 结构体指针，接受一个 qh 结构体指针参数，一个 ridgeT 结构体指针参数，一个指向 vertexT 结构体指针的指针参数，一个 coordT 类型参数

void    qh_findbest_test(qhT *qh, boolT testcentrum, facetT *facet, facetT *neighbor,
           facetT **bestfacet, realT *distp, realT *mindistp, realT *maxdistp);
# 声明函数 qh_findbest_test，接受一个 qh 结构体指针参数，一个布尔值参数，一个 facet 结构体指针参数，一个 facet 结构体指针参数，四个指针参数

facetT *qh_findbestneighbor(qhT *qh, facetT *facet, realT *distp, realT *mindistp, realT *maxdistp);
# 声明函数 qh_findbestneighbor，返回一个 facetT 结构体指针，接受一个 qh 结构体指针参数，一个 facet 结构体指针参数，三个 realT 类型参数的指针

void    qh_flippedmerges(qhT *qh, facetT *facetlist, boolT *wasmerge);
# 声明函数 qh_flippedmerges，接受一个 qh 结构体指针参数，一个 facet 结构体指针参数，一个指向布尔值的指针参数

void    qh_forcedmerges(qhT *qh, boolT *wasmerge);
# 声明函数 qh_forcedmerges，接受一个 qh 结构体指针参数，一个指向布尔值的指针参数

void    qh_freemergesets(qhT *qh);
# 声明函数 qh_freemergesets，接受一个 qh 结构体指针参数

void    qh_getmergeset(qhT *qh, facetT *facetlist);
# 声明函数 qh_getmergeset，接受一个 qh 结构体指针参数和一个 facet 结构体指针参数

void    qh_getmergeset_initial(qhT *qh, facetT *facetlist);
# 声明函数 qh_getmergeset_initial，接受一个 qh 结构体指针参数和一个 facet 结构体指针参数

boolT   qh_getpinchedmerges(qhT *qh, vertexT *apex, coordT maxdupdist, boolT *iscoplanar /* qh.newfacet_list, vertex_mergeset */);
# 声明函数 qh_getpinchedmerges，返回
// 合并顶点，并处理受压顶点
void qh_merge_pinchedvertices(qhT *qh, int apexpointid /* qh.newfacet_list */);

// 合并两个指定的凸包
void qh_merge_twisted(qhT *qh, facetT *facet1, facetT *facet2);

// 合并同一个环中的两个面，并更新其它结构
void qh_mergecycle(qhT *qh, facetT *samecycle, facetT *newfacet);

// 对所有面列表中的面进行循环合并，标记是否有合并操作
void qh_mergecycle_all(qhT *qh, facetT *facetlist, boolT *wasmerge);

// 合并两个面的环结构
void qh_mergecycle_facets(qhT *qh, facetT *samecycle, facetT *newfacet);

// 合并同一个环中的两个面的相邻关系
void qh_mergecycle_neighbors(qhT *qh, facetT *samecycle, facetT *newfacet);

// 合并同一个环中的两个面的脊线结构
void qh_mergecycle_ridges(qhT *qh, facetT *samecycle, facetT *newfacet);

// 合并同一个环中的两个面的顶点邻居关系
void qh_mergecycle_vneighbors(qhT *qh, facetT *samecycle, facetT *newfacet);

// 根据合并类型和距离信息合并两个面
void qh_mergefacet(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype, realT *mindist, realT *maxdist, boolT mergeapex);

// 在二维情况下合并两个面
void qh_mergefacet2d(qhT *qh, facetT *facet1, facetT *facet2);

// 合并两个面的邻居关系
void qh_mergeneighbors(qhT *qh, facetT *facet1, facetT *facet2);

// 合并两个面的脊线结构
void qh_mergeridges(qhT *qh, facetT *facet1, facetT *facet2);

// 合并两个简单形式的凸包
void qh_mergesimplex(qhT *qh, facetT *facet1, facetT *facet2, boolT mergeapex);

// 删除一个顶点的所有邻接关系，同时更新与两个面相关的顶点关系
void qh_mergevertex_del(qhT *qh, vertexT *vertex, facetT *facet1, facetT *facet2);

// 合并两个面的顶点邻居关系
void qh_mergevertex_neighbors(qhT *qh, facetT *facet1, facetT *facet2);

// 合并两个顶点集合，更新其中的顶点信息
void qh_mergevertices(qhT *qh, setT *vertices1, setT **vertices);

// 返回一个顶点的邻居交集
setT *qh_neighbor_intersections(qhT *qh, vertexT *vertex);

// 返回一个顶点的邻居顶点集合，可能包括子脊线
setT *qh_neighbor_vertices(qhT *qh, vertexT *vertex, setT *subridge);

// 返回一个顶点在面中的邻居顶点集合
void qh_neighbor_vertices_facet(qhT *qh, vertexT *vertexA, facetT *facet, setT **vertices);

// 处理新的顶点集合，更新相关信息
void qh_newvertices(qhT *qh, setT *vertices);

// 返回下一个顶点合并操作
mergeT *qh_next_vertexmerge(qhT *qh);

// 返回一个顶点的对应水平面面
facetT *qh_opposite_horizonfacet(qhT *qh, mergeT *merge, vertexT **vertex);

// 尝试减少顶点数量，返回是否成功
boolT qh_reducevertices(qhT *qh);

// 返回一个多余的顶点，如果有的话
vertexT *qh_redundant_vertex(qhT *qh, vertexT *vertex);

// 移除面中多余的顶点，返回是否成功
boolT qh_remove_extravertices(qhT *qh, facetT *facet);

// 移除合并类型对应的所有合并操作
void qh_remove_mergetype(qhT *qh, setT *mergeset, mergeType type);

// 重命名一个邻接顶点，更新距离信息
void qh_rename_adjacentvertex(qhT *qh, vertexT *oldvertex, vertexT *newvertex, realT dist);

// 返回一个面中共享的重命名顶点
vertexT *qh_rename_sharedvertex(qhT *qh, vertexT *vertex, facetT *facet);

// 尝试重命名面中的一个脊线顶点
boolT qh_renameridgevertex(qhT *qh, ridgeT *ridge, vertexT *oldvertex, vertexT *newvertex);

// 重命名一个顶点，更新相关的脊线和面信息
void qh_renamevertex(qhT *qh, vertexT *oldvertex, vertexT *newvertex, setT *ridges,
                     facetT *oldfacet, facetT *neighborA);

// 测试追加合并操作是否可行，返回是否成功
boolT qh_test_appendmerge(qhT *qh, facetT *facet, facetT *neighbor, boolT simplicial);

// 测试面的邻接关系是否可能导致退化情况
void qh_test_degen_neighbors(qhT *qh, facetT *facet);

// 测试面与邻居面是否能够中心合并，返回是否成功
boolT qh_test_centrum_merge(qhT *qh, facetT *facet, facetT *neighbor, realT angle, boolT okangle);

// 测试面与邻居面是否为非简单形式合并，返回是否成功
boolT qh_test_nonsimplicial_merge(qhT *qh, facetT *facet, facetT *neighbor, realT angle, boolT okangle);

// 测试面的冗余邻居关系
void qh_test_redundant_neighbors(qhT *qh, facetT *facet);

// 测试顶点的邻接关系，返回是否存在顶点邻居
boolT qh_test_vneighbors(qhT *qh /* qh.newfacet_list */);

// 跟踪并记录合并的两个面及其类型
void qh_tracemerge(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype);

// 跟踪并记录所有合并操作
void qh_tracemerging(qhT *qh);

// 撤销最近的新面
void qh_undo_newfacets(qhT *qh);

// 更新测试信息，涉及到的两个面
void qh_updatetested(qhT *qh, facetT *facet1, facetT *facet2);
setT *qh_vertexridges(qhT *qh, vertexT *vertex, boolT allneighbors);


# 返回顶点关联的所有棱的集合
setT *qh_vertexridges(qhT *qh, vertexT *vertex, boolT allneighbors);



void qh_vertexridges_facet(qhT *qh, vertexT *vertex, facetT *facet, setT **ridges);


# 计算给定顶点和面之间的所有棱，并将结果保存在ridges中
void qh_vertexridges_facet(qhT *qh, vertexT *vertex, facetT *facet, setT **ridges);



void qh_willdelete(qhT *qh, facetT *facet, facetT *replace);


# 指示qhT数据结构中的facet将被删除，replace是用于替换facet的面
void qh_willdelete(qhT *qh, facetT *facet, facetT *replace);



#ifdef __cplusplus
} /* extern "C" */
#endif


# 如果是C++编译环境，则使用extern "C"将函数声明放入C++代码块
#ifdef __cplusplus
} /* extern "C" */
#endif



#endif /* qhDEFmerge */


# 结束条件编译指令，确保qhDEFmerge宏定义未被重复包含
#endif /* qhDEFmerge */
```