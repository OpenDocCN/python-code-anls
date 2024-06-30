# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\stat_r.h`

```
/*<html><pre>  -<a                             href="qh-stat_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   stat_r.h
     contains all statistics that are collected for qhull

   see qh-stat_r.htm and stat_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/stat_r.h#3 $$Change: 2711 $
   $DateTime: 2019/06/27 22:34:56 $$Author: bbarber $

   recompile qhull if you change this file

   Integer statistics are Z* while real statistics are W*.

   define MAYdebugx to call a routine at every statistic event

*/

#ifndef qhDEFstat
#define qhDEFstat 1

/* Depends on realT.  Do not include "libqhull_r" to avoid circular dependency */

#ifndef DEFqhT
#define DEFqhT 1
typedef struct qhT qhT;         /* Defined by libqhull_r.h */
#endif

#ifndef DEFqhstatT
#define DEFqhstatT 1
typedef struct qhstatT qhstatT; /* Defined here */
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="KEEPstatistics">-</a>

  qh_KEEPstatistics
    0 turns off statistic reporting and gathering (except zzdef/zzinc/zzadd/zzval/wwval)

  set qh_KEEPstatistics in user_r.h to 0 to turn off statistics
*/
#ifndef qh_KEEPstatistics
#define qh_KEEPstatistics 1
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="statistics">-</a>

  Zxxx for integers, Wxxx for reals

  notes:
    be sure that all statistics are defined in stat_r.c
      otherwise initialization may core dump
    can pick up all statistics by:
      grep '[zw].*_[(][ZW]' *.c >z.x
    remove trailers with query">-</a>
    remove leaders with  query-replace-regexp [ ^I]+  (
*/
#if qh_KEEPstatistics
enum qh_statistics {     /* alphabetical after Z/W */
    Zacoplanar,                 // Count of coplanar sets that are active
    Wacoplanarmax,              // Maximum number of active coplanar sets
    Wacoplanartot,              // Total count of active coplanar sets
    Zangle,                     // Count of angle tests for facet inclusion
    Wangle,                     // Angle of the current facet
    Wanglemax,                  // Maximum angle found
    Wanglemin,                  // Minimum angle found
    Zangletests,                // Number of angle tests performed
    Wareatot,                   // Total area of all facets
    Wareamax,                   // Maximum area of any facet
    Wareamin,                   // Minimum area of any facet
    Zavoidold,                  // Count of avoiding old points during qhull
    Wavoidoldmax,               // Maximum number of old points avoided
    Wavoidoldtot,               // Total number of old points avoided
    Zback0,                     // Count of backward steps to the last vertex
    Zbestcentrum,               // Count of best centrum computations
    Zbestdist,                  // Count of best distance computations
    Zbestlower,                 // Count of best lower computations
    Zbestlowerall,              // Count of best lower facet tests
    Zbestloweralln,             // Count of best lower tests for new facets
    Zbestlowerv,                // Count of best lower vertex tests
    Zcentrumtests,              // Count of centrum tests
    Zcheckpart,                 // Count of checks of partitions
    Zcomputefurthest,           // Count of computations of furthest point
    Zconcave,                   // Count of concave facets
    Wconcavemax,                // Maximum concavity found
    Wconcavetot,                // Total concavity of all facets
    Zconcavecoplanar,           // Count of concave coplanar facets
    Wconcavecoplanarmax,        // Maximum concavity of coplanar facets
    Wconcavecoplanartot,        // Total concavity of coplanar facets
    Zconcavecoplanarridge,      // Count of ridges of concave coplanar facets
    Zconcaveridge,              // Count of ridges of concave facets
    Zconcaveridges,             // Count of all ridges of concave facets
    Zcoplanar,                  // Count of coplanar facets
    Wcoplanarmax,               // Maximum number of coplanar facets
    Wcoplanartot,               // Total number of coplanar facets
    Zcoplanarangle,             // Count of angle tests for coplanar facets
    Zcoplanarcentrum,           // Count of centrum tests for coplanar facets
    Zcoplanarhorizon,           // Count of horizon searches for coplanar facets
    Zcoplanarinside,            // Count of inside tests for coplanar facets
    Zcoplanarpart,              // Count of partitions of coplanar facets
    Zcoplanarridges,            // Count of ridges of coplanar facets
    Wcpu,                       // CPU time used
    Zcyclefacetmax,             // Maximum number of facets in a cycle
    Zcyclefacettot,             // Total number of facets in cycles
    Zcyclehorizon,              // Count of horizon facets in cycles
    Zcyclevertex,               // Count of vertices in cycles
    Zdegen,                     // Count of degenerate facets
    Wdegenmax,                  // Maximum number of degenerate facets
    Wdegentot,                  // Total number of degenerate facets
    Zdegenvertex,               // Count of vertices of degenerate facets
    Zdelfacetdup,               // Count of deleted facet duplicates
    Zdelridge,                  // Count of deleted ridges
    Zdelvertextot,              // Total count of deleted vertices
    Zdelvertexmax,              // Maximum count of deleted vertices
    Zdetfacetarea,              // Count of determinants for facet area
    Zdetsimplex,                // Count of determinants for simplices
    Zdistcheck,                 // Count of distance checks
    Zdistconvex,                // Count of distance checks with convex sets
    Zdistgood,                  // Count of distance checks with good facets
    Zdistio,                    // Count of distance checks with I/O facets
    Zdistplane,                 // Count of distance checks with plane facets
    Zdiststat,                  // Count of distance statistics
    Zdistvertex,                // Count of distance checks with vertex facets
    Zdistzero,                  // Count of zero distance points
    Zdoc1,                      // Documentation statistics 1
    Zdoc2,                      // Documentation statistics 2
    Zdoc3,                      // Documentation statistics 3
    Zdoc4,                      // Documentation statistics 4
    Zdoc5,                      // Documentation statistics 5
    Zdoc6,                      // Documentation statistics 6
    Zdoc7,                      // Documentation statistics 7


This is a continuation of the enumeration list for statistics collected in the qhull library.
    Zdoc8,  # 变量 Zdoc8，可能表示某种文档或数据结构
    Zdoc9,  # 变量 Zdoc9，可能表示某种文档或数据结构
    Zdoc10,  # 变量 Zdoc10，可能表示某种文档或数据结构
    Zdoc11,  # 变量 Zdoc11，可能表示某种文档或数据结构
    Zdoc12,  # 变量 Zdoc12，可能表示某种文档或数据结构
    Zdropdegen,  # 变量 Zdropdegen，可能与删除或丢弃有关的数据结构或标志
    Zdropneighbor,  # 变量 Zdropneighbor，可能与删除或丢弃邻居相关的数据结构或标志
    Zdupflip,  # 变量 Zdupflip，可能与复制和翻转有关的数据结构或标志
    Zduplicate,  # 变量 Zduplicate，可能与复制相关的数据结构或标志
    Wduplicatemax,  # 变量 Wduplicatemax，可能表示复制的最大数量或相关的统计数据
    Wduplicatetot,  # 变量 Wduplicatetot，可能表示复制的总数或相关的统计数据
    Zdupsame,  # 变量 Zdupsame，可能与相同或重复相关的数据结构或标志
    Zflipped,  # 变量 Zflipped，可能与翻转相关的数据结构或标志
    Wflippedmax,  # 变量 Wflippedmax，可能表示翻转的最大数量或相关的统计数据
    Wflippedtot,  # 变量 Wflippedtot，可能表示翻转的总数或相关的统计数据
    Zflippedfacets,  # 变量 Zflippedfacets，可能表示翻转的面片或相关的数据结构
    Zflipridge,  # 变量 Zflipridge，可能与翻转边缘相关的数据结构或标志
    Zflipridge2,  # 变量 Zflipridge2，可能与第二个翻转边缘相关的数据结构或标志
    Zfindbest,  # 变量 Zfindbest，可能与查找最佳结果相关的数据结构或标志
    Zfindbestmax,  # 变量 Zfindbestmax，可能表示查找最佳结果的最大数量或相关的统计数据
    Zfindbesttot,  # 变量 Zfindbesttot，可能表示查找最佳结果的总数或相关的统计数据
    Zfindcoplanar,  # 变量 Zfindcoplanar，可能与查找共面相关的数据结构或标志
    Zfindfail,  # 变量 Zfindfail，可能与查找失败相关的数据结构或标志
    Zfindhorizon,  # 变量 Zfindhorizon，可能与查找视角相关的数据结构或标志
    Zfindhorizonmax,  # 变量 Zfindhorizonmax，可能表示查找视角的最大数量或相关的统计数据
    Zfindhorizontot,  # 变量 Zfindhorizontot，可能表示查找视角的总数或相关的统计数据
    Zfindjump,  # 变量 Zfindjump，可能与查找跳跃相关的数据结构或标志
    Zfindnew,  # 变量 Zfindnew，可能与查找新项相关的数据结构或标志
    Zfindnewmax,  # 变量 Zfindnewmax，可能表示查找新项的最大数量或相关的统计数据
    Zfindnewtot,  # 变量 Zfindnewtot，可能表示查找新项的总数或相关的统计数据
    Zfindnewjump,  # 变量 Zfindnewjump，可能与查找新项跳跃相关的数据结构或标志
    Zfindnewsharp,  # 变量 Zfindnewsharp，可能与查找新项尖锐相关的数据结构或标志
    Zgauss0,  # 变量 Zgauss0，可能与高斯分布或零相关的数据结构或标志
    Zgoodfacet,  # 变量 Zgoodfacet，可能与良好面片或相关的数据结构或标志
    Zhashlookup,  # 变量 Zhashlookup，可能与哈希查找相关的数据结构或标志
    Zhashridge,  # 变量 Zhashridge，可能与哈希边缘相关的数据结构或标志
    Zhashridgetest,  # 变量 Zhashridgetest，可能与哈希边缘测试相关的数据结构或标志
    Zhashtests,  # 变量 Zhashtests，可能与哈希测试相关的数据结构或标志
    Zinsidevisible,  # 变量 Zinsidevisible，可能与可见内部相关的数据结构或标志
    Zintersect,  # 变量 Zintersect，可能与相交相关的数据结构或标志
    Zintersectfail,  # 变量 Zintersectfail，可能与相交失败相关的数据结构或标志
    Zintersectmax,  # 变量 Zintersectmax，可能表示相交的最大数量或相关的统计数据
    Zintersectnum,  # 变量 Zintersectnum，可能表示相交的数量或相关的统计数据
    Zintersecttot,  # 变量 Zintersecttot，可能表示相交的总数或相关的统计数据
    Zmaxneighbors,  # 变量 Zmaxneighbors，可能表示最大邻居数或相关的统计数据
    Wmaxout,  # 变量 Wmaxout，可能表示最大输出或相关的统计数据
    Wmaxoutside,  # 变量 Wmaxoutside，可能表示最大外部或相关的统计数据
    Zmaxridges,  # 变量 Zmaxridges，可能表示最大边缘或相关的统计数据
    Zmaxvertex,  # 变量 Zmaxvertex，可能表示最大顶点或相关的统计数据
    Zmaxvertices,  # 变量 Zmaxvertices，可能表示最大顶点数或相关的统计数据
    Zmaxvneighbors,  # 变量 Zmaxvneighbors，可能表示最大顶点邻居数或相关的统计数据
    Zmemfacets,  # 变量 Zmemfacets，可能与面片存储相关的数据结构或标志
    Zmempoints,  # 变量 Zmempoints，可能与点存储相关的数据结构或标志
    Zmemridges,  # 变量 Zmemridges，可能与边缘存储相关的数据结构或标志
    Zmemvertices,  # 变量 Zmemvertices，可能与顶点存储相关的数据结构或标志
    Zmergeflipdup,  # 变量 Zmergeflipdup，可能与合并、翻转或复制相关的数据结构或标志
    Zmergehorizon,  # 变量 Zmergehorizon，可能与合并视角相关的数据结构或标志
    Zmergeinittot,  # 变量 Zmergeinittot，可能表示合并初始化的总数或相关的统计数据
    Zmergeinitmax,  # 变量 Zmergeinitmax，可能表示合并初始化的最大数量或相关的统计数据
    Zmergeinittot2,  # 变量 Zmergeinittot2，可能表示第二个合并初始化的总数或相关的统计数据
    Zmergeintocoplanar,  # 变量 Zmergeintocoplanar，可能与合并共面相关的数据结构或标志
    Zmergeintohorizon,  # 变量 Zmergeintohorizon，可能与合并到视角相关的数据结构或标志
    Zmergenew,  # 变量 Zmergenew，可能与合并新项相关的数据结构或标志
    Zmergesettot,  # 变量 Zmergesettot，可能表示合并集的总数或相关的统计数据
    Zmergesetmax,  # 变量 Zmergesetmax，可能表示合并集的最大数量或
    // 定义一个名称为 Zwidevertices 的变量或标识符
    Zwidevertices,
    // 表示某种终止符或结束标记，用于结束某个语句块或结构
    ZEND};
/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="ZZstat">-</a>

  Zxxx/Wxxx statistics that remain defined if qh_KEEPstatistics=0

  notes:
    be sure to use zzdef, zzinc, etc. with these statistics (no double checking!)
*/
#else
enum qh_statistics {     /* for zzdef etc. macros */
  Zback0,                 // index 0: unused
  Zbestdist,              // index 1: best distance
  Zcentrumtests,          // index 2: centrum tests
  Zcheckpart,             // index 3: check part
  Zconcaveridges,         // index 4: concave ridges
  Zcoplanarhorizon,       // index 5: coplanar horizon
  Zcoplanarpart,          // index 6: coplanar part
  Zcoplanarridges,        // index 7: coplanar ridges
  Zcyclefacettot,         // index 8: cycle facet total
  Zcyclehorizon,          // index 9: cycle horizon
  Zdelvertextot,          // index 10: delete vertex total
  Zdistcheck,             // index 11: distance check
  Zdistconvex,            // index 12: distance convex
  Zdistplane,             // index 13: distance plane
  Zdistzero,              // index 14: distance zero
  Zdoc1,                  // index 15: documentation 1
  Zdoc2,                  // index 16: documentation 2
  Zdoc3,                  // index 17: documentation 3
  Zdoc11,                 // index 18: documentation 11
  Zflippedfacets,         // index 19: flipped facets
  Zflipridge,             // index 20: flip ridge
  Zflipridge2,            // index 21: flip ridge 2
  Zgauss0,                // index 22: Gauss 0
  Zminnorm,               // index 23: minimum norm
  Zmultiridge,            // index 24: multi ridge
  Znearlysingular,        // index 25: nearly singular
  Wnewvertexmax,          // index 26: new vertex max
  Znumvisibility,         // index 27: number visibility
  Zpartcoplanar,          // index 28: part coplanar
  Zpartition,             // index 29: partition
  Zpartitionall,          // index 30: partition all
  Zpinchduplicate,        // index 31: pinch duplicate
  Zpinchedvertex,         // index 32: pinched vertex
  Zprocessed,             // index 33: processed
  Zretry,                 // index 34: retry
  Zridge,                 // index 35: ridge
  Wridge,                 // index 36: ridge
  Wridgemax,              // index 37: ridge max
  Zridge0,                // index 38: ridge 0
  Wridge0,                // index 39: ridge 0
  Wridge0max,             // index 40: ridge 0 max
  Zridgemid,              // index 41: ridge mid
  Wridgemid,              // index 42: ridge mid
  Wridgemidmax,           // index 43: ridge mid max
  Zridgeok,               // index 44: ridge ok
  Wridgeok,               // index 45: ridge ok
  Wridgeokmax,            // index 46: ridge ok max
  Zsetplane,              // index 47: set plane
  Ztotcheck,              // index 48: total check
  Ztotmerge,              // index 49: total merge
  Zvertextests,           // index 50: vertex tests
  ZEND                    // index 51: end of enumeration
};
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="ztype">-</a>

  ztype
    the type of a statistic sets its initial value.

  notes:
    The type should be the same as the macro for collecting the statistic
*/
enum ztypes {zdoc,zinc,zadd,zmax,zmin,ZTYPEreal,wadd,wmax,wmin,ZTYPEend};

/*========== macros and constants =============*/

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="MAYdebugx">-</a>

  MAYdebugx
    define as maydebug() to be called frequently for error trapping
*/
#define MAYdebugx

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zdef_">-</a>

  zzdef_, zdef_( type, name, doc, -1)
    define a statistic (assumes 'qhstat.next= 0;')

  zdef_( type, name, doc, count)
    define an averaged statistic
    printed as name/count
*/
#define zzdef_(stype,name,string,cnt) qh->qhstat.id[qh->qhstat.next++]=name; \
   qh->qhstat.doc[name]= string; qh->qhstat.count[name]= cnt; qh->qhstat.type[name]= stype
#if qh_KEEPstatistics
#define zdef_(stype,name,string,cnt) qh->qhstat.id[qh->qhstat.next++]=name; \
   qh->qhstat.doc[name]= string; qh->qhstat.count[name]= cnt; qh->qhstat.type[name]= stype
#else
#define zdef_(type,name,doc,count)
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zinc_">-</a>

  zzinc_( name ), zinc_( name)
    increment an integer statistic
*/
#define zzinc_(id) {MAYdebugx; qh->qhstat.stats[id].i++;}
#if qh_KEEPstatistics
#define zinc_(id) {MAYdebugx; qh->qhstat.stats[id].i++;}
#else
#define zinc_(id) {}
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zadd_">-</a>

  zzadd_( name, value ), zadd_( name, value ), wadd_( name, value )
    # 声明一个变量 `add`，用于存储待加的值
    add value to an integer or real statistic
/*
#define zzadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].i += (val);}
#define wwadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].r += (val);}
#if qh_KEEPstatistics
#define zadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].i += (val);}
#define wadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].r += (val);}
#else
#define zadd_(id, val) {}
#define wadd_(id, val) {}
#endif
*/

/*
  -<a href="qh-stat_r.htm#TOC">--------------------------------</a><a name="zval_">-</a>

  zzval_( name ), zval_( name ), wwval_( name )
    set or return value of a statistic
*/
#define zzval_(id) ((qh->qhstat.stats[id]).i)
#define wwval_(id) ((qh->qhstat.stats[id]).r)
#if qh_KEEPstatistics
#define zval_(id) ((qh->qhstat.stats[id]).i)
#define wval_(id) ((qh->qhstat.stats[id]).r)
#else
#define zval_(id) qh->qhstat.tempi
#define wval_(id) qh->qhstat.tempr
#endif

/*
  -<a href="qh-stat_r.htm#TOC">--------------------------------</a><a name="zmax_">-</a>

  zmax_( id, val ), wmax_( id, value )
    maximize id with val
*/
#define wwmax_(id, val) {MAYdebugx; maximize_(qh->qhstat.stats[id].r,(val));}
#if qh_KEEPstatistics
#define zmax_(id, val) {MAYdebugx; maximize_(qh->qhstat.stats[id].i,(val));}
#define wmax_(id, val) {MAYdebugx; maximize_(qh->qhstat.stats[id].r,(val));}
#else
#define zmax_(id, val) {}
#define wmax_(id, val) {}
#endif

/*
  -<a href="qh-stat_r.htm#TOC">--------------------------------</a><a name="zmin_">-</a>

  zmin_( id, val ), wmin_( id, value )
    minimize id with val
*/
#if qh_KEEPstatistics
#define zmin_(id, val) {MAYdebugx; minimize_(qh->qhstat.stats[id].i,(val));}
#define wmin_(id, val) {MAYdebugx; minimize_(qh->qhstat.stats[id].r,(val));}
#else
#define zmin_(id, val) {}
#define wmin_(id, val) {}
#endif

/*================== stat_r.h types ==============*/

/*
  -<a href="qh-stat_r.htm#TOC">--------------------------------</a><a name="intrealT">-</a>

  intrealT
    union of integer and real, used for statistics
*/
typedef union intrealT intrealT;    /* union of int and realT */
union intrealT {
    int i;
    realT r;
};

/*
  -<a href="qh-stat_r.htm#TOC">--------------------------------</a><a name="qhstat">-</a>

  qhstat
    Data structure for statistics, similar to qh and qhrbox

    Allocated as part of qhT (libqhull_r.h)
*/
/* 定义结构体 qhstatT，包含各种统计信息的存储 */
struct qhstatT {
  intrealT   stats[ZEND];     /* integer and real statistics 数字和实数统计数据 */
  unsigned char id[ZEND+10];  /* id's in print order 按打印顺序的 ID */
  const char *doc[ZEND];      /* array of documentation strings 文档字符串数组 */
  short int  count[ZEND];     /* -1 if none, else index of count to use 如果没有则为 -1，否则使用计数的索引 */
  char       type[ZEND];      /* type, see ztypes above 类型，参见上述 ztypes */
  char       printed[ZEND];   /* true, if statistic has been printed 如果统计已打印则为 true */
  intrealT   init[ZTYPEend];  /* initial values by types, set initstatistics 按类型的初始值，由 initstatistics 设置 */

  int        next;            /* next index for zdef_ 下一个 zdef_ 的索引 */
  int        precision;       /* index for precision problems, printed on qh_errexit and qh_produce_output2/Q0/QJn 精度问题的索引，在 qh_errexit 和 qh_produce_output2/Q0/QJn 上打印 */
  int        vridges;         /* index for Voronoi ridges, printed on qh_produce_output2 Voronoi 生成的索引，在 qh_produce_output2 上打印 */
  int        tempi;           /* temporary integer 临时整数 */
  realT      tempr;           /* temporary real 临时实数 */
};

/*========== function prototypes ===========*/

#ifdef __cplusplus
extern "C" {
#endif

/* 函数原型声明 */
void    qh_allstatA(qhT *qh);
void    qh_allstatB(qhT *qh);
void    qh_allstatC(qhT *qh);
void    qh_allstatD(qhT *qh);
void    qh_allstatE(qhT *qh);
void    qh_allstatE2(qhT *qh);
void    qh_allstatF(qhT *qh);
void    qh_allstatG(qhT *qh);
void    qh_allstatH(qhT *qh);
void    qh_allstatI(qhT *qh);
void    qh_allstatistics(qhT *qh);
void    qh_collectstatistics(qhT *qh);
void    qh_initstatistics(qhT *qh);
boolT   qh_newstats(qhT *qh, int idx, int *nextindex);
boolT   qh_nostatistic(qhT *qh, int i);
void    qh_printallstatistics(qhT *qh, FILE *fp, const char *string);
void    qh_printstatistics(qhT *qh, FILE *fp, const char *string);
void    qh_printstatlevel(qhT *qh, FILE *fp, int id);
void    qh_printstats(qhT *qh, FILE *fp, int idx, int *nextindex);
realT   qh_stddev(qhT *qh, int num, realT tot, realT tot2, realT *ave);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif   /* qhDEFstat */
```