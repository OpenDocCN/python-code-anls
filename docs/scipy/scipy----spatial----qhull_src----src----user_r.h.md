# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\user_r.h`

```
/*
   user_r.h
   user redefinable constants

   for each source file, user_r.h is included first

   see qh-user_r.htm.  see COPYING for copyright information.

   See user_r.c for sample code.

   before reading any code, review libqhull_r.h for data structure definitions

Sections:
   ============= qhull library constants ======================
   ============= data types and configuration macros ==========
   ============= performance related constants ================
   ============= memory constants =============================
   ============= joggle constants =============================
   ============= conditional compilation ======================
   ============= merge constants ==============================
   ============= Microsoft DevStudio ==========================
*/

#include <float.h>      // 包含浮点数处理相关的常量和函数
#include <limits.h>     // 包含整数处理相关的常量
#include <time.h>       // 包含时间处理相关的常量和函数

#ifndef qhDEFuser       // 如果未定义 qhDEFuser 宏，则定义之
#define qhDEFuser 1
#endif

/* Derived from Qt's corelib/global/qglobal.h */
#if !defined(SAG_COM) && !defined(__CYGWIN__) && (defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
#   define QHULL_OS_WIN   // 定义 QHULL_OS_WIN 宏表示当前操作系统为 Windows
#elif defined(__MWERKS__) && defined(__INTEL__) /* Metrowerks discontinued before the release of Intel Macs */
#   define QHULL_OS_WIN   // 类似条件定义，适用于 Metrowerks 编译器和 Intel Macs

/*============================================================*/
/*============= qhull library constants ======================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="filenamelen">-</a>

  FILENAMElen -- max length for TI and TO filenames

*/
#define qh_FILENAMElen 500  // 定义文件名的最大长度为 500

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="msgcode">-</a>

  msgcode -- Unique message codes for qh_fprintf

  If add new messages, assign these values and increment in user.h and user_r.h
  See QhullError.h for 10000 error codes.
  Cannot use '0031' since it would be octal

  def counters =  [31/32/33/38, 1067, 2113, 3079, 4097, 5006,
     6428, 7027/7028/7035/7068/7070/7102, 8163, 9428, 10000, 11034]

  See: qh_ERR* [libqhull_r.h]
*/
#define MSG_TRACE0     0    // 定义不同级别的消息码用于 qh_fprintf 的日志输出
#define MSG_TRACE1  1000
#define MSG_TRACE2  2000
#define MSG_TRACE3  3000
#define MSG_TRACE4  4000
#define MSG_TRACE5  5000
#define MSG_ERROR   6000   // 定义错误消息码，输出到 qh.ferr
#define MSG_WARNING 7000   // 定义警告消息码
#define MSG_STDERR  8000   // 定义输出到 qh.ferr 的日志消息
#define MSG_OUTPUT  9000   // 定义输出消息码
#define MSG_QHULL_ERROR 10000  // 定义 QhullError.cpp 抛出的错误消息码
#define MSG_FIX    11000   // 定义修正性质的消息码，格式为 'QH11... FIX: ...'
/* 定义 MSG_MAXLEN 为 3000，用于描述 qh_printhelp_degenerate() 函数在 user_r.c 文件中的行为 */

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="qh_OPTIONline">-</a>

  qh_OPTIONline -- max length of an option line 'FO'
*/
#define qh_OPTIONline 80
/* 定义 qh_OPTIONline 为 80，表示选项行的最大长度 */

/*============================================================*/
/*============= data types and configuration macros ==========*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="realT">-</a>

  realT
    set the size of floating point numbers

  qh_REALdigits
    maximimum number of significant digits

  qh_REAL_1, qh_REAL_2n, qh_REAL_3n
    format strings for printf

  qh_REALmax, qh_REALmin
    maximum and minimum (near zero) values

  qh_REALepsilon
    machine roundoff.  Maximum roundoff error for addition and multiplication.

  notes:
   Select whether to store floating point numbers in single precision (float)
   or double precision (double).

   Use 'float' to save about 8% in time and 25% in space.  This is particularly
   helpful if high-d where convex hulls are space limited.  Using 'float' also
   reduces the printed size of Qhull's output since numbers have 8 digits of
   precision.

   Use 'double' when greater arithmetic precision is needed.  This is needed
   for Delaunay triangulations and Voronoi diagrams when you are not merging
   facets.

   If 'double' gives insufficient precision, your data probably includes
   degeneracies.  If so you should use facet merging (done by default)
   or exact arithmetic (see imprecision section of manual, qh-impre.htm).
   You may also use option 'Po' to force output despite precision errors.

   You may use 'long double', but many format statements need to be changed
   and you may need a 'long double' square root routine.  S. Grundmann
   (sg@eeiwzb.et.tu-dresden.de) has done this.  He reports that the code runs
   much slower with little gain in precision.

   WARNING: on some machines,    int f(){realT a= REALmax;return (a == REALmax);}
      returns False.  Use (a > REALmax/2) instead of (a == REALmax).

   REALfloat =   1      all numbers are 'float' type
             =   0      all numbers are 'double' type
*/
#define REALfloat 0
/* 定义 REALfloat 为 0，表示所有数字为 double 类型 */

#if (REALfloat == 1)
#define realT float
#define REALmax FLT_MAX
#define REALmin FLT_MIN
#define REALepsilon FLT_EPSILON
#define qh_REALdigits 8   /* maximum number of significant digits */
#define qh_REAL_1 "%6.8g "
#define qh_REAL_2n "%6.8g %6.8g\n"
#define qh_REAL_3n "%6.8g %6.8g %6.8g\n"

#elif (REALfloat == 0)
#define realT double
#define REALmax DBL_MAX
#define REALmin DBL_MIN
#define REALepsilon DBL_EPSILON
#define qh_REALdigits 16    /* maximum number of significant digits */
#define qh_REAL_1 "%6.16g "
#define qh_REAL_2n "%6.16g %6.16g\n"
#define qh_REAL_3n "%6.16g %6.16g %6.16g\n"

#else
#error unknown float option
#endif
/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="countT">-</a>

  countT
    计数和标识符的类型（例如点的数量、顶点标识符）
    目前仅用于 C++ 代码。对于 setT 没有使用它，因为大多数集合很小。

    定义为 'int' 是为了与 C 代码兼容和 QH11026

    QH11026 修正：countT 可能被定义为 'unsigned int'，但需要先解决几个代码问题。参见 Changes.txt 中的 countT
*/

#ifndef DEFcountT
#define DEFcountT 1
typedef int countT;
#endif
#define COUNTmax INT_MAX

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="qh_POINTSmax">-</a>

  qh_POINTSmax
    qh.num_points 和 qh_readpoints 中点的最大数量，用于点的分配
*/
#define qh_POINTSmax (INT_MAX-16)

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="CPUclock">-</a>

  qh_CPUclock
    定义用于报告 Qhull 总运行时间的 clock() 函数
    返回 CPU 时钟作为 'long int'
    qh_CPUclock 仅用于报告 Qhull 的总运行时间

  qh_SECticks
    每秒钟的时钟 ticks 数

  注：
    查找 CLOCKS_PER_SEC、CLOCKS_PER_SECOND 或假设微秒来定义自定义时钟，将 qh_CLOCKtype 设置为 0

    如果您的系统不使用 clock() 返回 CPU 时钟 ticks，则用相应的函数替换 qh_CPUclock。
    将其转换为 'unsigned long' 是为了防止长时间运行时的 wrap-around。默认情况下，
    <time.h> 将 clock_t 定义为 'long'

   将 qh_CLOCKtype 设置为

     1          对于 CLOCKS_PER_SEC、CLOCKS_PER_SECOND 或微秒
                注意：如果经过的时间超过1小时可能会失败

     2          使用 qh_clock() 与 POSIX times()（见 global_r.c）
*/
#define qh_CLOCKtype 1  /* 更改为所需的编号 */

#if (qh_CLOCKtype == 1)

#if defined(CLOCKS_PER_SECOND)
#define qh_CPUclock    ((unsigned long)clock())  /* 返回 CPU 时钟 */
#define qh_SECticks CLOCKS_PER_SECOND

#elif defined(CLOCKS_PER_SEC)
#define qh_CPUclock    ((unsigned long)clock())  /* 返回 CPU 时钟 */
#define qh_SECticks CLOCKS_PER_SEC

#elif defined(CLK_TCK)
#define qh_CPUclock    ((unsigned long)clock())  /* 返回 CPU 时钟 */
#define qh_SECticks CLK_TCK

#else
#define qh_CPUclock    ((unsigned long)clock())  /* 返回 CPU 时钟 */
#define qh_SECticks 1E6
#endif

#elif (qh_CLOCKtype == 2)
#define qh_CPUclock    qh_clock()  /* 返回 CPU 时钟 */
#define qh_SECticks 100

#else /* qh_CLOCKtype == ? */
#error 未知的时钟选项
#endif

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RANDOM">-</a>

  qh_RANDOMtype, qh_RANDOMmax, qh_RANDOMseed
    定义随机数生成器

    qh_RANDOMint 生成一个介于 0 和 qh_RANDOMmax 之间的随机整数。
*/
    # 设置 qh_RANDOMseed 函数用于初始化 qh_RANDOMint 的随机数种子
    
    # 设置 qh_RANDOMtype 为默认值 5，选项如下：
    #   1 - 使用带有31位的 random()（UCB）
    #   2 - 使用带有 RAND_MAX 或15位的 rand()（系统5）
    #   3 - 使用带有31位的 rand()（Sun）
    #   4 - 使用带有31位的 lrand48()（Solaris）
    #   5 - 使用带有31位的 qh_rand(qh)（随 Qhull 一起提供，需要 'qh' 库）
    
    # 注意：
    #   随机数用于 rbox 生成点集。Qhull 用于旋转输入（'QRn' 选项），
    #   模拟随机算法（'Qr' 选项），以及模拟舍入误差（'Rn' 选项）。
    #   
    #   不同系统的随机数生成器不同。大多数系统提供 rand()，但周期不同。
    #   rand() 的周期不是关键，因为 qhull 通常不使用随机数。
    #   
    #   默认生成器是 Park & Miller 的最小标准随机数生成器 [CACM 31:1195 '88]，包含在 Qhull 中。
    #   
    #   如果 qh_RANDOMmax 设置不正确，qhull 将报告警告并且 Geomview 输出可能会不可见。
/*
#define qh_RANDOMtype 5   /* *** change to the desired number *** */

#if (qh_RANDOMtype == 1)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, random()/MAX */
#define qh_RANDOMint random()
#define qh_RANDOMseed_(qh, seed) srandom(seed);

#elif (qh_RANDOMtype == 2)
#ifdef RAND_MAX
#define qh_RANDOMmax ((realT)RAND_MAX)
#else
#define qh_RANDOMmax ((realT)32767)   /* 15 bits (System 5) */
#endif
#define qh_RANDOMint  rand()
#define qh_RANDOMseed_(qh, seed) srand((unsigned int)seed);

#elif (qh_RANDOMtype == 3)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, Sun */
#define qh_RANDOMint  rand()
#define qh_RANDOMseed_(qh, seed) srand((unsigned int)seed);

#elif (qh_RANDOMtype == 4)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, lrand38()/MAX */
#define qh_RANDOMint lrand48()
#define qh_RANDOMseed_(qh, seed) srand48(seed);

#elif (qh_RANDOMtype == 5)  /* 'qh' is an implicit parameter */
#define qh_RANDOMmax ((realT)2147483646UL)  /* 31 bits, qh_rand/MAX */
#define qh_RANDOMint qh_rand(qh)
#define qh_RANDOMseed_(qh, seed) qh_srand(qh, seed);
/* unlike rand(), never returns 0 */

#else
#error: unknown random option
#endif
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="ORIENTclock">-</a>

  qh_ORIENTclock
    0 for inward pointing normals by Geomview convention
*/
#define qh_ORIENTclock 0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RANDOMdist">-</a>

  qh_RANDOMdist
    define for random perturbation of qh_distplane and qh_setfacetplane (qh.RANDOMdist, 'QRn')

  For testing qh.DISTround.  Qhull should not depend on computations always producing the same roundoff error 

  #define qh_RANDOMdist 1e-13
*/

/*============================================================*/
/*============= joggle constants =============================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEdefault">-</a>

  qh_JOGGLEdefault
    default qh.JOGGLEmax is qh.DISTround * qh_JOGGLEdefault

  notes:
    rbox s r 100 | qhull QJ1e-15 QR0 generates 90% faults at distround 7e-16
    rbox s r 100 | qhull QJ1e-14 QR0 generates 70% faults
    rbox s r 100 | qhull QJ1e-13 QR0 generates 35% faults
    rbox s r 100 | qhull QJ1e-12 QR0 generates 8% faults
    rbox s r 100 | qhull QJ1e-11 QR0 generates 1% faults
    rbox s r 100 | qhull QJ1e-10 QR0 generates 0% faults
    rbox 1000 W0 | qhull QJ1e-12 QR0 generates 86% faults
    rbox 1000 W0 | qhull QJ1e-11 QR0 generates 20% faults
    rbox 1000 W0 | qhull QJ1e-10 QR0 generates 2% faults
    the later have about 20 points per facet, each of which may interfere

    pick a value large enough to avoid retries on most inputs
*/
#define qh_JOGGLEdefault 30000.0
/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEincrease">-</a>

  qh_JOGGLEincrease
    在 qh_JOGGLEretry 或 qh_JOGGLEagain 时增加 qh.JOGGLEmax 的因子
*/
#define qh_JOGGLEincrease 10.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEretry">-</a>

  qh_JOGGLEretry
    如果 ZZretry = qh_JOGGLEretry，则增加 qh.JOGGLEmax

notes:
第一次运气不佳时，尝试两次使用原始值
*/
#define qh_JOGGLEretry 2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEagain">-</a>

  qh_JOGGLEagain
    每次 qh_JOGGLEagain 之后，增加 qh.JOGGLEmax

  notes:
    1 表示已经尝试 qh_JOGGLEretry 次数后，仍可再尝试一次
*/
#define qh_JOGGLEagain 1

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEmaxincrease">-</a>

  qh_JOGGLEmaxincrease
    相对于 qh.MAXwidth，由 qh.JOGGLEincrease 引起的 qh.JOGGLEmax 最大增加量

  notes:
    qh.joggleinput 将在此值下重试，直到达到 qh_JOGGLEmaxretry 次
*/
#define qh_JOGGLEmaxincrease 1e-2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEmaxretry">-</a>

  qh_JOGGLEmaxretry
    在 qh_JOGGLEmaxretry 次尝试后停止
*/
#define qh_JOGGLEmaxretry 50

/*============================================================*/
/*============= performance related constants ================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="HASHfactor">-</a>

  qh_HASHfactor
    总的哈希槽数量 / 使用的哈希槽数量。必须至少为 1.1。

  notes:
    qh.hash_table 的最差情况下为 50% 填充率，通常为 25% 填充率
*/
#define qh_HASHfactor 2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="VERIFYdirect">-</a>

  qh_VERIFYdirect
    使用 'Tv' 时，如果操作计数较小，则验证所有点相对所有面

  notes:
    如果操作计数大，则调用 qh_check_bestdist() 替代
*/
#define qh_VERIFYdirect 1000000

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="INITIALsearch">-</a>

  qh_INITIALsearch
    如果 qh_INITIALmax，则搜索点的维度直到这个值
*/
#define qh_INITIALsearch 6

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="INITIALmax">-</a>

  qh_INITIALmax
    如果维度 >= qh_INITIALmax，则使用初始单纯形的最小/最大坐标点

  notes:
    使用非零行列式的点
    使用选项 'Qs' 来覆盖（速度慢）
*/
#define qh_INITIALmax 8

/*============================================================*/
/*============= memory constants =============================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MEMalign">-</a>

  qh_MEMalign
    memory alignment for qh_meminitbuffers() in global_r.c

  notes:
    to avoid bus errors, memory allocation must consider alignment requirements.
    malloc() automatically takes care of alignment.   Since mem_r.c manages
    its own memory, we need to explicitly specify alignment in
    qh_meminitbuffers().

    A safe choice is sizeof(double).  sizeof(float) may be used if doubles
    do not occur in data structures and pointers are the same size.  Be careful
    of machines (e.g., DEC Alpha) with large pointers.

    If using gcc, best alignment is [fmax_() is defined in geom_r.h]
              #define qh_MEMalign fmax_(__alignof__(realT),__alignof__(void *))
*/
#define qh_MEMalign ((int)(fmax_(sizeof(realT), sizeof(void *))))

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MEMbufsize">-</a>

  qh_MEMbufsize
    size of additional memory buffers

  notes:
    used for qh_meminitbuffers() in global_r.c
*/
#define qh_MEMbufsize 0x10000       /* allocate 64K memory buffers */

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MEMinitbuf">-</a>

  qh_MEMinitbuf
    size of initial memory buffer

  notes:
    use for qh_meminitbuffers() in global_r.c
*/
#define qh_MEMinitbuf 0x20000      /* initially allocate 128K buffer */

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="INFINITE">-</a>

  qh_INFINITE
    on output, indicates Voronoi center at infinity
*/
#define qh_INFINITE  -10.101

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="DEFAULTbox">-</a>

  qh_DEFAULTbox
    default box size (Geomview expects 0.5)

  qh_DEFAULTbox
    default box size for integer coorindate (rbox only)
*/
#define qh_DEFAULTbox 0.5
#define qh_DEFAULTzbox 1e6

/*============================================================*/
/*============= conditional compilation ======================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="compiler">-</a>

  __cplusplus
    defined by C++ compilers

  __MSC_VER
    defined by Microsoft Visual C++

  __MWERKS__ && __INTEL__
    defined by Metrowerks when compiling for Windows (not Intel-based Macintosh)

  __MWERKS__ && __POWERPC__
    defined by Metrowerks when compiling for PowerPC-based Macintosh

  __STDC__
    defined for strict ANSI C
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="COMPUTEfurthest">-</a>

  qh_COMPUTEfurthest
    This macro or symbol is not defined here, but it's supposed to be defined somewhere else.
    It is related to computing the furthest point.

*/
    compute furthest distance to an outside point instead of storing it with the facet
    =1 to compute furthest

  notes:
    computing furthest saves memory but costs time
      about 40% more distance tests for partitioning
      removes facet->furthestdist
/*
   定义常量 qh_COMPUTEfurthest，值为 0
   用途：控制是否计算最远点

   qh_COMPUTEfurthest constant, value 0
   Purpose: Controls computation of furthest point
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="KEEPstatistics">-</a>

  qh_KEEPstatistics
    =0 removes most of statistic gathering and reporting

  notes:
    if 0, code size is reduced by about 4%.
*/
#define qh_KEEPstatistics 1

/*
   定义常量 qh_KEEPstatistics，值为 1
   用途：控制是否保留大部分统计信息的收集和报告

   qh_KEEPstatistics constant, value 1
   Purpose: Controls whether to keep most statistic gathering and reporting
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXoutside">-</a>

  qh_MAXoutside
    record outer plane for each facet
    =1 to record facet->maxoutside

  notes:
    this takes a realT per facet and slightly slows down qhull
    it produces better outer planes for geomview output
*/
#define qh_MAXoutside 1

/*
   定义常量 qh_MAXoutside，值为 1
   用途：控制是否记录每个 facet 的外部平面，并记录 facet->maxoutside

   qh_MAXoutside constant, value 1
   Purpose: Controls whether to record outer plane for each facet

   Notes:
   This takes a realT per facet and slightly slows down qhull.
   It produces better outer planes for geomview output.
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="NOmerge">-</a>

  qh_NOmerge
    disables facet merging if defined
    For MSVC compiles, use qhull_r-exports-nomerge.def instead of qhull_r-exports.def

  notes:
    This saves about 25% space, 30% space in combination with qh_NOtrace,
    and 36% with qh_NOtrace and qh_KEEPstatistics 0

    Unless option 'Q0' is used
      qh_NOmerge sets 'QJ' to avoid precision errors

  see:
    <a href="mem_r.h#NOmem">qh_NOmem</a> in mem_r.h

    see user_r.c/user_eg.c for removing io_r.o

  #define qh_NOmerge
*/

/*
   定义宏 qh_NOmerge，用于禁用面合并
   用途：禁用面合并功能

   qh_NOmerge macro, disables facet merging
   Purpose: Disables facet merging functionality

   Notes:
   This saves about 25% space, 30% space in combination with qh_NOtrace,
   and 36% with qh_NOtrace and qh_KEEPstatistics 0.

   Unless option 'Q0' is used, qh_NOmerge sets 'QJ' to avoid precision errors.

   See also:
   'qh_NOmem' in mem_r.h
   user_r.c/user_eg.c for removing io_r.o
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="NOtrace">-</a>

  qh_NOtrace
    no tracing if defined
    disables 'Tn', 'TMn', 'TPn' and 'TWn'
    override with 'Qw' for qh_addpoint tracing and various other items

  notes:
    This saves about 15% space.
    Removes all traceN((...)) code and substantial sections of qh.IStracing code

  #define qh_NOtrace
*/

/*
   定义宏 qh_NOtrace，用于禁用跟踪
   用途：禁用跟踪功能

   qh_NOtrace macro, disables tracing
   Purpose: Disables tracing functionality

   Notes:
   This saves about 15% space.
   Removes all traceN((...)) code and substantial sections of qh.IStracing code.
*/

#if 0  /* sample code */
    exitcode= qh_new_qhull(qhT *qh, dim, numpoints, points, ismalloc,
                      flags, outfile, errfile);
    qh_freeqhull(qhT *qh, !qh_ALL); /* frees long memory used by second call */
    qh_memfreeshort(qhT *qh, &curlong, &totlong);  /* frees short memory and memory allocator */
#endif

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="QUICKhelp">-</a>

  qh_QUICKhelp
    =1 to use abbreviated help messages, e.g., for degenerate inputs
*/
#define qh_QUICKhelp    0

/*
   定义常量 qh_QUICKhelp，值为 0
   用途：控制是否使用简短的帮助信息

   qh_QUICKhelp constant, value 0
   Purpose: Controls whether to use abbreviated help messages
*/

/*============================================================*/
/*============= merge constants ==============================*/
/*============================================================*/

/*
   These constants effect facet merging.  You probably will not need
   to modify them.  They effect the performance of facet merging.
*/

/*
   这些常量影响面的合并。通常情况下，您不需要修改它们。
   它们影响面合并的性能。
*/
/*
   -<a                             href="qh-user_r.htm#TOC"
     >--------------------------------</a><a name="BESTcentrum">-</a>

   qh_BESTcentrum
      if > 2*dim+n vertices, qh_findbestneighbor() tests centrums (faster)
      else, qh_findbestneighbor() tests all vertices (much better merges)

   qh_BESTcentrum2
      if qh_BESTcentrum2 * DIM3 + BESTcentrum < #vertices tests centrums
*/
#define qh_BESTcentrum 20          // 定义常量 qh_BESTcentrum 为 20
#define qh_BESTcentrum2 2          // 定义常量 qh_BESTcentrum2 为 2

/*
   -<a                             href="qh-user_r.htm#TOC"
     >--------------------------------</a><a name="BESTnonconvex">-</a>

   qh_BESTnonconvex
     if > dim+n neighbors, qh_findbestneighbor() tests nonconvex ridges.

   notes:
     It is needed because qh_findbestneighbor is slow for large facets
*/
#define qh_BESTnonconvex 15        // 定义常量 qh_BESTnonconvex 为 15

/*
   -<a                             href="qh-user_r.htm#TOC"
     >--------------------------------</a><a name="COPLANARratio">-</a>

   qh_COPLANARratio
     for 3-d+ merging, qh.MINvisible is n*premerge_centrum

   notes:
     for non-merging, it's DISTround
*/
#define qh_COPLANARratio 3         // 定义常量 qh_COPLANARratio 为 3

/*
   -<a                             href="qh-user_r.htm#TOC"
     >--------------------------------</a><a name="DIMmergeVertex">-</a>

   qh_DIMmergeVertex
     max dimension for vertex merging (it is not effective in high-d)
*/
#define qh_DIMmergeVertex 6        // 定义常量 qh_DIMmergeVertex 为 6

/*
   -<a                             href="qh-user_r.htm#TOC"
     >--------------------------------</a><a name="DIMreduceBuild">-</a>

   qh_DIMreduceBuild
      max dimension for vertex reduction during build (slow in high-d)
*/
#define qh_DIMreduceBuild 5        // 定义常量 qh_DIMreduceBuild 为 5

/*
   -<a                             href="qh-user_r.htm#TOC"
     >--------------------------------</a><a name="DISToutside">-</a>

   qh_DISToutside
     When is a point clearly outside of a facet?
     Stops search in qh_findbestnew or qh_partitionall
     qh_findbest uses qh.MINoutside since since it is only called if no merges.

   notes:
     'Qf' always searches for best facet
     if !qh.MERGING, same as qh.MINoutside.
     if qh_USEfindbestnew, increase value since neighboring facets may be ill-behaved
       [Note: Zdelvertextot occurs normally with interior points]
             RBOX 1000 s Z1 G1e-13 t1001188774 | QHULL Tv
     When there is a sharp edge, need to move points to a
     clearly good facet; otherwise may be lost in another partitioning.
     if too big then O(n^2) behavior for partitioning in cone
     if very small then important points not processed
     Needed in qh_partitionall for
       RBOX 1000 s Z1 G1e-13 t1001032651 | QHULL Tv
     Needed in qh_findbestnew for many instances of
       RBOX 1000 s Z1 G1e-13 t | QHULL Tv

   See:
     qh_DISToutside -- when is a point clearly outside of a facet
     qh_SEARCHdist -- when is facet coplanar with the best facet?
     qh_USEfindbestnew -- when to use qh_findbestnew for qh_partitionpoint()
*/
#define qh_DISToutside ((qh_USEfindbestnew ? 2 : 1) * \
     fmax_((qh->MERGING ? 2 : 1)*qh->MINoutside, qh->max_outside))   // 定义宏 qh_DISToutside，根据条件计算距离阈值
/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXcheckpoint">-</a>

  qh_MAXcheckpoint
    Report up to qh_MAXcheckpoint errors per facet in qh_check_point ('Tv')
*/
#define qh_MAXcheckpoint 10

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXcoplanarcentrum">-</a>

  qh_MAXcoplanarcentrum
    if pre-merging with qh.MERGEexact ('Qx') and f.nummerge > qh_MAXcoplanarcentrum
      use f.maxoutside instead of qh.centrum_radius for coplanarity testing

  notes:
    see qh_test_nonsimplicial_merges
    with qh.MERGEexact, a coplanar ridge is ignored until post-merging
    otherwise a large facet with many merges may take all the facets
*/
#define qh_MAXcoplanarcentrum 10

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXnewcentrum">-</a>

  qh_MAXnewcentrum
    if <= dim+n vertices (n approximates the number of merges),
      reset the centrum in qh_updatetested() and qh_mergecycle_facets()

  notes:
    needed to reduce cost and because centrums may move too much if
    many vertices in high-d
*/
#define qh_MAXnewcentrum 5

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXnewmerges">-</a>

  qh_MAXnewmerges
    if >n newmerges, qh_merge_nonconvex() calls qh_reducevertices_centrums.

  notes:
    It is needed because postmerge can merge many facets at once
*/
#define qh_MAXnewmerges 2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOconcavehorizon">-</a>

  qh_RATIOconcavehorizon
    ratio of horizon vertex distance to max_outside for concave, twisted new facets in qh_test_nonsimplicial_merge
    if too small, end up with vertices far below merged facets
*/
#define qh_RATIOconcavehorizon 20.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOconvexmerge">-</a>

  qh_RATIOconvexmerge
    ratio of vertex distance to qh.min_vertex for clearly convex new facets in qh_test_nonsimplicial_merge

  notes:
    must be convex for MRGtwisted
*/
#define qh_RATIOconvexmerge 10.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOcoplanarapex">-</a>

  qh_RATIOcoplanarapex
    ratio of best distance for coplanar apex vs. vertex merge in qh_getpinchedmerges

  notes:
    A coplanar apex always works, while a vertex merge may fail
*/
#define qh_RATIOcoplanarapex 3.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOcoplanaroutside">-</a>

  qh_RATIOcoplanaroutside
    qh.MAXoutside ratio to repartition a coplanar point in qh_partitioncoplanar and qh_check_maxout

  notes:
    combines several tests, see qh_partitioncoplanar

*/
#define qh_RATIOcoplanaroutside 30.0
/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOmaxsimplex">-</a>

  qh_RATIOmaxsimplex
    qh_MAXsimplex 中搜索所有点时，最大行列式与预估行列式的比率

  notes:
    每当将点添加到单纯形中时，max 行列式应该接近先前行列式 * qh.MAXwidth
    如果 maxdet 明显较小，单纯形可能不是完整维度的。
    如果是这样，所有点都将被搜索，停止在 10 倍 qh_RATIOmaxsimplex 处
*/
#define qh_RATIOmaxsimplex 1.0e-3

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOnearinside">-</a>

  qh_RATIOnearinside
    保留在 qh_check_maxout() 中内部点的比率，为 qh.NEARinside 到 qh.ONEmerge

  notes:
    这是一种过度保守的设置，因为不知道正确的值。
    它影响 'Qc' 报告所有共面点
    对于 'd' 不适用，因为非极端点是共面的，几乎相交的点
*/
#define qh_RATIOnearinside 5

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOpinchedsubridge">-</a>

  qh_RATIOpinchedsubridge
    接受在 qh_findbest_pinchedvertex 中的顶点的比率，为 qh.ONEmerge

    跳过搜索邻近顶点
    面的宽度可能增加这个比率
*/
#define qh_RATIOpinchedsubridge 10.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOtrypinched">-</a>

  qh_RATIOtrypinched
    尝试在 qh_buildcone_mergepinched 中的 qh_ONEmerge 的比率，以尝试 qh_getpinchedmerges

    否则重复的边缘将增加面的宽度这么多
*/
#define qh_RATIOtrypinched 4.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOtwisted">-</a>

  qh_RATIOtwisted
    合并在 qh_merge_twisted 中扭曲面的最大比率，为 qh.ONEmerge 的最大比率
*/
#define qh_RATIOtwisted 20.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="SEARCHdist">-</a>

  qh_SEARCHdist
    何时一个面与最佳面共面？
    qh_findbesthorizon: 需要搜索最佳面的所有共面面。
        如果 ischeckmax 并且邻居超过 100 个 (is_5x_minsearch)，增加最小搜索距离

  See:
    qh_DISToutside -- 何时点明显在面外
    qh_SEARCHdist -- 何时面与最佳面共面？
    qh_USEfindbestnew -- 何时使用 qh_findbestnew 进行 qh_partitionpoint()
*/
#define qh_SEARCHdist ((qh_USEfindbestnew ? 2 : 1) * \
      (qh->max_outside + 2 * qh->DISTround + fmax_( qh->MINvisible, qh->MAXcoplanar)));
/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="USEfindbestnew">-</a>

  qh_USEfindbestnew
     Always use qh_findbestnew for qh_partitionpoint, otherwise use
     qh_findbestnew if merged new facet or sharpnewfacets.

  See:
    qh_DISToutside -- when is a point clearly outside of a facet
    qh_SEARCHdist -- when is facet coplanar with the best facet?
    qh_USEfindbestnew -- when to use qh_findbestnew for qh_partitionpoint()
*/
#define qh_USEfindbestnew (zzval_(Ztotmerge) > 50)

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXnarrow">-</a>

  qh_MAXnarrow
    max. cosine in initial hull that sets qh.NARROWhull

  notes:
    If qh.NARROWhull, the initial partition does not make
    coplanar points.  If narrow, a coplanar point can be
    coplanar to two facets of opposite orientations and
    distant from the exact convex hull.

    Conservative estimate.  Don't actually see problems until it is -1.0
*/
#define qh_MAXnarrow -0.99999999

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="WARNnarrow">-</a>

  qh_WARNnarrow
    max. cosine in initial hull to warn about qh.NARROWhull

  notes:
    this is a conservative estimate.
    Don't actually see problems until it is -1.0.  See qh-impre.htm
*/
#define qh_WARNnarrow -0.999999999999999

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEcoplanar">-</a>

  qh_WIDEcoplanar
    n*MAXcoplanar or n*MINvisible for a WIDEfacet

    if vertex is further than qh.WIDEfacet from the hyperplane
    then its ridges are not counted in computing the area, and
    the facet's centrum is frozen.

  notes:
    qh.WIDEfacet= max(qh.MAXoutside,qh_WIDEcoplanar*qh.MAXcoplanar,
    qh_WIDEcoplanar * qh.MINvisible);
*/
#define qh_WIDEcoplanar 6

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEduplicate">-</a>

  qh_WIDEduplicate
    merge ratio for errexit from qh_forcedmerges due to duplicate ridge
    Override with option Q12-allow-wide

  Notes:
    Merging a duplicate ridge can lead to very wide facets.
*/
#define qh_WIDEduplicate 100

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEdupridge">-</a>

  qh_WIDEdupridge
    Merge ratio for selecting a forced dupridge merge

  Notes:
    Merging a dupridge can lead to very wide facets.
*/
#define qh_WIDEdupridge 50

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEmaxoutside">-</a>

  qh_WIDEmaxoutside
    Precision ratio for maximum increase for qh.max_outside in qh_check_maxout
    Precision errors while constructing the hull, may lead to very wide facets when checked in qh_check_maxout
*/
    # Nearly incident points in 4-d and higher is the most likely culprit
    Nearly incident points in 4-d 及更高维度的几乎相交点是最可能的罪魁祸首
    # Skip qh_check_maxout with 'Q5' (no-check-outer)
    使用 'Q5'（no-check-outer）跳过 qh_check_maxout
    # Do not error with option 'Q12' (allow-wide)
    使用选项 'Q12'（allow-wide）时不要报错
    # Do not warn with options 'Q12 Pp'
    使用选项 'Q12 Pp' 时不要发出警告
/*
   定义一个常量 qh_WIDEmaxoutside，表示最大外部距离的精度比例
   在 qh_check_maxout 中跳过检查 'Q5' no-check-outer
   选项 'Q12' allow-wide 不会因错误而中断
*/
#define qh_WIDEmaxoutside 100

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEmaxoutside2">-</a>

  qh_WIDEmaxoutside2
    qh_check_maxout 中 qh.max_outside 的最大值的精度比例
    使用 'Q5' no-check-outer 可跳过 qh_check_maxout
    使用 'Q12' allow-wide 可避免错误中断
*/
#define qh_WIDEmaxoutside2 (10*qh_WIDEmaxoutside)


/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEpinched">-</a>

  qh_WIDEpinched
    qh_getpinchedmerges 和 qh_next_vertexmerge 中捏合顶点之间的合并比例
    如果开启选项 Q14 merge-pinched-vertices，则报告警告并合并重复的山脊
    注意：
    合并捏合顶点应该能防止重复的山脊（见 qh_WIDEduplicate）
    合并重复的山脊可能比合并捏合顶点更好
    对于 qh_pointdist 找到了高达 45 倍的比例 -- ((i=1; i<20; i++)); do rbox 175 C1,6e-13 t | qhull d T4 2>&1 | tee x.1 | grep  -E 'QH|non-simplicial|Statis|pinched'; done
    实际到面的距离是 qh_pointdist 的三分之一到十分之一（T1）
*/
#define qh_WIDEpinched 100

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="ZEROdelaunay">-</a>

  qh_ZEROdelaunay
    输入站点与其凸包共面时出现零 Delaunay 面
    零 Delaunay 面的最后法线系数在 0 的 qh_ZEROdelaunay * qh.ANGLEround 范围内

  注意：
    qh_ZEROdelaunay 不适用于有小偏差的输入 ('QJ')

    可以通过在输入周围放置一个盒子来避免零 Delaunay 面的出现

    使用选项 'PDk:-n' 可以明确定义零 Delaunay 面
      k= 输入站点的维度（例如，三维 Delaunay 三角剖分的 k=3）
      n= 零 Delaunay 面的截止值（例如，'PD3:-1e-12'）
*/
#define qh_ZEROdelaunay 2

/*============================================================*/
/*============= Microsoft DevStudio ==========================*/
/*============================================================*/

/*
   使用 CRT 库查找内存泄漏
   https://msdn.microsoft.com/en-us/library/x98tx3cf(v=vs.100).aspx

   在调试窗口和 stderr 中启用 qh_lib_check 报告

   自 2005 年起 => msvcr80d，2010 年 => msvcr100d，2012 年 => msvcr110d

   Watch: {,,msvcr80d.dll}_crtBreakAlloc  在泄漏报告中的 {n} 值
   _CrtSetBreakAlloc(689); // qh_lib_check() [global_r.c]

   示例：
     http://free-cad.sourceforge.net/SrcDocu/d2/d7f/MemDebug_8cpp_source.html
     https://github.com/illlust/Game/blob/master/library/MemoryLeak.cpp
*/
#if 0   /* 默认关闭（0），用于 QHULL_CRTDBG */
#define QHULL_CRTDBG
#endif

#if defined(_MSC_VER) && defined(_DEBUG) && defined(QHULL_CRTDBG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#endif /* qh_DEFuser */
```