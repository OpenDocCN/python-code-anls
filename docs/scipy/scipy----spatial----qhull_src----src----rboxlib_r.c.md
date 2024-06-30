# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\rboxlib_r.c`

```
/*
   rboxlib_r.c
     Generate input points

   notes:
     For documentation, see prompt[] of rbox_r.c
     50 points generated for 'rbox D4'

   WARNING:
     incorrect range if qh_RANDOMmax is defined wrong (user_r.h)
*/

#include "libqhull_r.h"  /* First for user_r.h */
#include "random_r.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER  /* Microsoft Visual C++ */
#pragma warning( disable : 4706)  /* assignment within conditional expression. */
#pragma warning( disable : 4996)  /* this function (strncat,sprintf,strcpy) or variable may be unsafe. */
#endif

#define MAXdim 200
#define PI 3.1415926535897932384

/* ------------------------------ prototypes ----------------*/
int qh_roundi(qhT *qh, double a);  // 声明函数 qh_roundi，用于将双精度浮点数 a 四舍五入为整数
void qh_out1(qhT *qh, double a);   // 声明函数 qh_out1，用于输出一个双精度浮点数 a
void qh_out2n(qhT *qh, double a, double b);  // 声明函数 qh_out2n，用于输出两个双精度浮点数 a 和 b
void qh_out3n(qhT *qh, double a, double b, double c);  // 声明函数 qh_out3n，用于输出三个双精度浮点数 a、b 和 c
void qh_outcoord(qhT *qh, int iscdd, double *coord, int dim);  // 声明函数 qh_outcoord，用于输出坐标信息
void qh_outcoincident(qhT *qh, int coincidentpoints, double radius, int iscdd, double *coord, int dim);  // 声明函数 qh_outcoincident，用于输出重合点信息
void qh_rboxpoints2(qhT *qh, char* rbox_command, double **simplex);  // 声明函数 qh_rboxpoints2，用于生成点集

void qh_fprintf_rbox(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );  // 声明函数 qh_fprintf_rbox，用于向文件流 fp 写入格式化输出
void qh_free(void *mem);  // 声明函数 qh_free，用于释放内存
void *qh_malloc(size_t size);  // 声明函数 qh_malloc，用于分配内存
int qh_rand(qhT *qh);  // 声明函数 qh_rand，用于生成随机数
void qh_srand(qhT *qh, int seed);  // 声明函数 qh_srand，用于初始化随机数生成器

/*-<a href="qh-qhull_r.htm#TOC">-</a><a name="rboxpoints">-</a>

  qh_rboxpoints(qh, rbox_command )
    Generate points to qh.fout according to rbox options
    Report errors on qh.ferr

  returns:
    0 (qh_ERRnone) on success
    1 (qh_ERRinput) on input error
    4 (qh_ERRmem) on memory error
    5 (qh_ERRqhull) on internal error

  notes:
    To avoid using stdio, redefine qh_malloc, qh_free, and qh_fprintf_rbox (user_r.c)
    Split out qh_rboxpoints2() to avoid -Wclobbered

  design:
    Straight line code (consider defining a struct and functions):

    Parse arguments into variables
    Determine the number of points
    Generate the points
*/
int qh_rboxpoints(qhT *qh, char* rbox_command) {
  int exitcode;
  double *simplex;

  simplex = NULL;
  exitcode = setjmp(qh->rbox_errexit);  // 设置跳转点以处理错误退出
  if (exitcode) {
    /* same code for error exit and normal return.  qh.NOerrexit is set */
    if (simplex)
      qh_free(simplex);  // 如果简单形式点集不为空，则释放内存
    return exitcode;  // 返回退出代码
  }
  qh_rboxpoints2(qh, rbox_command, &simplex);  // 调用函数生成点集
  /* same code for error exit and normal return */
  if (simplex)
    qh_free(simplex);  // 如果简单形式点集不为空，则释放内存
  return qh_ERRnone;  // 返回无错误状态
} /* rboxpoints */
void qh_rboxpoints2(qhT *qh, char* rbox_command, double **simplex) {
    // 定义变量
    int i,j,k;
    int gendim;
    int coincidentcount=0, coincidenttotal=0, coincidentpoints=0;
    int cubesize, diamondsize, seed=0, count, apex;
    int dim=3, numpoints=0, totpoints, addpoints=0;
    int issphere=0, isaxis=0,  iscdd=0, islens=0, isregular=0, iswidth=0, addcube=0;
    int isgap=0, isspiral=0, NOcommand=0, adddiamond=0;
    int israndom=0, istime=0;
    int isbox=0, issimplex=0, issimplex2=0, ismesh=0;
    double width=0.0, gap=0.0, radius=0.0, coincidentradius=0.0;
    double coord[MAXdim], offset, meshm=3.0, meshn=4.0, meshr=5.0;
    double *coordp, *simplexp;
    int nthroot, mult[MAXdim];
    double norm, factor, randr, rangap, tempr, lensangle=0, lensbase=1;
    double anglediff, angle, x, y, cube=0.0, diamond=0.0;
    double box= qh_DEFAULTbox; /* scale all numbers before output */
    double randmax= qh_RANDOMmax;
    char command[250], seedbuf[50];
    char *s=command, *t, *first_point=NULL;
    time_t timedata;

    // 初始化命令字符串
    *command= '\0';
    strncat(command, rbox_command, sizeof(command)-sizeof(seedbuf)-strlen(command)-1);

    // 跳过程序名
    while (*s && !isspace(*s))
        s++;
    // 解析命令行参数
    while (*s) {
        while (*s && isspace(*s))
            s++;
        if (*s == '-')
            s++;
        if (!*s)
            break;
        if (isdigit(*s)) {
            // 解析点数
            numpoints= qh_strtol(s, &s);
            continue;
        }
        /* ============= read flags =============== */
        switch (*s++) {
        case 'c':
            // 添加立方体标志
            addcube= 1;
            t= s;
            while (isspace(*t))
                t++;
            if (*t == 'G')
                // 设置立方体尺寸
                cube= qh_strtod(++t, &s);
            break;
        case 'd':
            // 添加菱形标志
            adddiamond= 1;
            t= s;
            while (isspace(*t))
                t++;
            if (*t == 'G')
                // 设置菱形尺寸
                diamond= qh_strtod(++t, &s);
            break;
        case 'h':
            // 设置 CDD 格式标志
            iscdd= 1;
            break;
        case 'l':
            // 设置螺旋线标志
            isspiral= 1;
            break;
        case 'n':
            // 设置 NO 标志
            NOcommand= 1;
            break;
        case 'r':
            // 设置正则网格标志
            isregular= 1;
            break;
        case 's':
            // 设置球体标志
            issphere= 1;
            break;
        case 't':
            // 设置时间种子或随机种子
            istime= 1;
            if (isdigit(*s)) {
                seed= qh_strtol(s, &s);
                israndom= 0;
            }else
                israndom= 1;
            break;
        case 'x':
            // 设置简单形状1标志
            issimplex= 1;
            break;
        case 'y':
            // 设置简单形状2标志
            issimplex2= 1;
            break;
        case 'z':
            // 设置整数坐标标志
            qh->rbox_isinteger= 1;
            break;
        case 'B':
            // 设置盒子尺寸
            box= qh_strtod(s, &s);
            isbox= 1;
            break;
        }
    }
}
    case 'C':
      // 如果字符串 s 的第一个字符不为空，则解析并设置 coincidentpoints
      if (*s)
        coincidentpoints=  qh_strtol(s, &s);
      // 如果下一个字符是逗号，则解析并设置 coincidentradius
      if (*s == ',') {
        ++s;
        coincidentradius=  qh_strtod(s, &s);
      }
      // 如果下一个字符是逗号，则解析并设置 coincidenttotal
      if (*s == ',') {
        ++s;
        coincidenttotal=  qh_strtol(s, &s);
      }
      // 如果字符串 s 不为空且下一个字符不是空格，则输出错误信息并退出
      if (*s && !isspace(*s)) {
        qh_fprintf_rbox(qh, qh->ferr, 7080, "rbox error: arguments for 'Cn,r,m' are not 'int', 'float', and 'int'.  Remaining string is '%s'\n", s);
        qh_errexit_rbox(qh, qh_ERRinput);
      }
      // 如果 coincidentpoints 等于 0，则输出错误信息并退出
      if (coincidentpoints==0){
        qh_fprintf_rbox(qh, qh->ferr, 6268, "rbox error: missing arguments for 'Cn,r,m' where n is the number of coincident points, r is the radius (default 0.0), and m is the number of points\n");
        qh_errexit_rbox(qh, qh_ERRinput);
      }
      // 如果 coincidentpoints, coincidenttotal 或 coincidentradius 有任何一个小于 0，则输出错误信息并退出
      if (coincidentpoints<0 || coincidenttotal<0 || coincidentradius<0.0){
        qh_fprintf_rbox(qh, qh->ferr, 6269, "rbox error: negative arguments for 'Cn,m,r' where n (%d) is the number of coincident points, m (%d) is the number of points, and r (%.2g) is the radius (default 0.0)\n", coincidentpoints, coincidenttotal, coincidentradius);
        qh_errexit_rbox(qh, qh_ERRinput);
      }
      break;
    case 'D':
      // 解析并设置维度 dim，如果 dim 小于 1 或大于 MAXdim，则输出错误信息并退出
      dim= qh_strtol(s, &s);
      if (dim < 1
      || dim > MAXdim) {
        qh_fprintf_rbox(qh, qh->ferr, 6189, "rbox error: dimension, D%d, out of bounds (>=%d or <=0)\n", dim, MAXdim);
        qh_errexit_rbox(qh, qh_ERRinput);
      }
      break;
    case 'G':
      // 如果下一个字符是数字，则解析并设置 gap，否则设置 gap 为 0.5
      if (isdigit(*s))
        gap= qh_strtod(s, &s);
      else
        gap= 0.5;
      // 设置标志位 isgap 为 1
      isgap= 1;
      break;
    case 'L':
      // 如果下一个字符是数字，则解析并设置 radius，否则设置 radius 为 10
      if (isdigit(*s))
        radius= qh_strtod(s, &s);
      else
        radius= 10;
      // 设置标志位 islens 为 1
      islens= 1;
      break;
    case 'M':
      // 设置标志位 ismesh 为 1
      ismesh= 1;
      // 如果字符串 s 不为空，则解析并设置 meshn
      if (*s)
        meshn= qh_strtod(s, &s);
      // 如果下一个字符是逗号，则解析并设置 meshm，否则设置为 0.0
      if (*s == ',') {
        ++s;
        meshm= qh_strtod(s, &s);
      }else
        meshm= 0.0;
      // 如果下一个字符是逗号，则解析并设置 meshr，否则计算默认值
      if (*s == ',') {
        ++s;
        meshr= qh_strtod(s, &s);
      }else
        meshr= sqrt(meshn*meshn + meshm*meshm);
      // 如果字符串 s 不为空且下一个字符不是空格，则输出警告信息并使用默认值
      if (*s && !isspace(*s)) {
        qh_fprintf_rbox(qh, qh->ferr, 7069, "rbox warning: assuming 'M3,4,5' since mesh args are not integers or reals\n");
        meshn= 3.0, meshm=4.0, meshr=5.0;
      }
      break;
    case 'O':
      // 解析并设置偏移量 qh->rbox_out_offset
      qh->rbox_out_offset= qh_strtod(s, &s);
      break;
    case 'P':
      // 如果 first_point 为空，则设置其值为 s 的前一个位置
      if (!first_point)
        first_point= s - 1;
      // 增加点数计数器 addpoints，直到遇到空格或结束符为止
      addpoints++;
      while (*s && !isspace(*s))   /* read points later */
        s++;
      break;
    case 'W':
      // 解析并设置宽度 width
      width= qh_strtod(s, &s);
      // 设置标志位 iswidth 为 1
      iswidth= 1;
      break;
    case 'Z':
      // 如果下一个字符是数字，则解析并设置 radius，否则设置 radius 为 1.0
      if (isdigit(*s))
        radius= qh_strtod(s, &s);
      else
        radius= 1.0;
      // 设置标志位 isaxis 为 1
      isaxis= 1;
      break;
    default:
      // 默认情况下，输出未知标志的错误信息并退出
      qh_fprintf_rbox(qh, qh->ferr, 6352, "rbox error: unknown flag at '%s'.\nExecute 'rbox' without arguments for documentation.\n", s - 1);
      qh_errexit_rbox(qh, qh_ERRinput);
    }
    if (*s && !isspace(*s)) {
      // 检查字符串指针 s 所指向的字符不为空且不是空白字符
      qh_fprintf_rbox(qh, qh->ferr, 6353, "rbox error: missing space between flags at %s.\n", s);
      // 输出错误消息，指示在字符串 s 处缺少空格
      qh_errexit_rbox(qh, qh_ERRinput);
      // 以输入错误的错误码退出程序
    }
  }

  /* ============= defaults, constants, and sizes =============== */
  // 默认值、常量和大小设置部分
  if (qh->rbox_isinteger && !isbox)
    // 如果设置了整数类型且不是盒子模型
    box= qh_DEFAULTzbox;
    // 设置盒子大小为默认值
  if (addcube) {
    // 如果需要添加立方体
    tempr= floor(ldexp(1.0,dim)+0.5);
    // 计算临时值，用于确定立方体大小
    cubesize= (int)tempr;
    // 设置立方体大小
    if (cube == 0.0)
      // 如果未指定立方体大小
      cube= box;
      // 使用默认盒子大小
  }else
    cubesize= 0;
    // 否则立方体大小为零
  if (adddiamond) {
    // 如果需要添加钻石形
    diamondsize= 2*dim;
    // 设置钻石形的大小
    if (diamond == 0.0)
      // 如果未指定钻石形大小
      diamond= box;
      // 使用默认盒子大小
  }else
    diamondsize= 0;
    // 否则钻石形大小为零
  if (islens) {
    // 如果是透镜模型
    if (isaxis) {
        // 如果同时指定了轴对称
        qh_fprintf_rbox(qh, qh->ferr, 6190, "rbox error: can not combine 'Ln' with 'Zn'\n");
        // 输出错误消息，指示不能同时使用 'Ln' 和 'Zn'
        qh_errexit_rbox(qh, qh_ERRinput);
        // 以输入错误的错误码退出程序
    }
    if (radius <= 1.0) {
        // 如果透镜半径小于等于1.0
        qh_fprintf_rbox(qh, qh->ferr, 6191, "rbox error: lens radius %.2g should be greater than 1.0\n",
               radius);
        // 输出错误消息，指示透镜半径应大于1.0
        qh_errexit_rbox(qh, qh_ERRinput);
        // 以输入错误的错误码退出程序
    }
    lensangle= asin(1.0/radius);
    // 计算透镜的角度
    lensbase= radius * cos(lensangle);
    // 计算透镜的基座
  }

  if (!numpoints) {
    // 如果未指定点的数量
    if (issimplex2)
        ; /* ok */
        // 如果是简单形状2，什么也不做（表示合法）
    else if (isregular + issimplex + islens + issphere + isaxis + isspiral + iswidth + ismesh) {
        // 如果指定了其他类型的形状和特性
        qh_fprintf_rbox(qh, qh->ferr, 6192, "rbox error: missing count\n");
        // 输出错误消息，指示缺少点的数量信息
        qh_errexit_rbox(qh, qh_ERRinput);
        // 以输入错误的错误码退出程序
    }else if (adddiamond + addcube + addpoints)
        ; /* ok */
        // 如果是添加钻石、立方体或点，什么也不做（表示合法）
    else {
        numpoints= 50;  /* ./rbox D4 is the test case */
        // 否则，默认设置点的数量为50，用于测试情况 './rbox D4'
        issphere= 1;
        // 设置为球体形状
    }
  }
  if ((issimplex + islens + isspiral + ismesh > 1)
  || (issimplex + issphere + isspiral + ismesh > 1)) {
    // 如果同时指定了多个形状类型
    qh_fprintf_rbox(qh, qh->ferr, 6193, "rbox error: can only specify one of 'l', 's', 'x', 'Ln', or 'Mn,m,r' ('Ln s' is ok).\n");
    // 输出错误消息，指示只能指定其中一个形状类型
    qh_errexit_rbox(qh, qh_ERRinput);
    // 以输入错误的错误码退出程序
  }
  if (coincidentpoints>0 && (numpoints == 0 || coincidenttotal > numpoints)) {
    // 如果有重合点要求且点的数量为0或重合总数大于点的数量
    qh_fprintf_rbox(qh, qh->ferr, 6270, "rbox error: 'Cn,r,m' requested n coincident points for each of m points.  Either there is no points or m (%d) is greater than the number of points (%d).\n", coincidenttotal, numpoints);
    // 输出错误消息，指示请求了 n 个重合点，但点的数量不足
    qh_errexit_rbox(qh, qh_ERRinput);
    // 以输入错误的错误码退出程序
  }
  if (coincidentpoints > 0 && isregular) {
    // 如果有重合点要求且指定了规则点
    qh_fprintf_rbox(qh, qh->ferr, 6423, "rbox error: 'Cn,r,m' is not implemented for regular points ('r')\n");
    // 输出错误消息，指示规则点不支持重合点要求
    qh_errexit_rbox(qh, qh_ERRinput);
    // 以输入错误的错误码退出程序
  }

  if (coincidenttotal == 0)
    // 如果没有指定总的重合点数
    coincidenttotal= numpoints;
    // 将总的重合点数设置为点的数量

  /* ============= print header with total points =============== */
  // 打印包含总点数的头部信息
  if (issimplex || ismesh)
    // 如果是简单形状或网格
    totpoints= numpoints;
    // 总点数等于点的数量
  else if (issimplex2)
    // 如果是简单形状2
    totpoints= numpoints+dim+1;
    // 总点数等于点的数量加维度加1
  else if (isregular) {
    // 如果是规则点
    totpoints= numpoints;
    // 总点数等于点的数量
    if (dim == 2) {
        // 如果维度是2
        if (islens)
          // 如果是透镜形状
          totpoints += numpoints - 2;
          // 总点数加上点的数量减2
    }else if (dim == 3) {
        // 如果维度是3
        if (islens)
          // 如果是透镜形状
          totpoints += 2 * numpoints;
          // 总点数加上2倍的点的数量
      else if (isgap)
        // 否则如果是间隙形状
        totpoints += 1 + numpoints;
        // 总点数加上1再加点的数量
      else
        // 否则
        totpoints += 2;
        // 总点数加上2
    }
  }else
    // 计算总点数，包括 numpoints 和 isaxis
    totpoints= numpoints + isaxis;
  // 添加 cubesize、diamondsize、addpoints 的点数到总点数
  totpoints += cubesize + diamondsize + addpoints;
  // 添加 coincidentpoints 乘以 coincidenttotal 的点数到总点数
  totpoints += coincidentpoints*coincidenttotal;

  /* ============= 随机数种子 =============== */
  // 如果 istime 为 0，则基于 command 中的字符生成种子
  if (istime == 0) {
    for (s=command; *s; s++) {
      // 如果 issimplex2 为真且字符为 'y'，则将 'y' 的种子设为与 'x' 相同
      if (issimplex2 && *s == 'y') /* make 'y' same seed as 'x' */
        i= 'x';
      else
        i= *s;
      // 使用字符 i 更新种子
      seed= 11*seed + i;
    }
  // 如果是随机生成种子
  }else if (israndom) {
    // 使用当前时间作为种子
    seed= (int)time(&timedata);
    // 将种子追加到 command 字符串中，末尾添加一个额外的 't' 字符
    sprintf(seedbuf, " t%d", seed);  /* appends an extra t, not worth removing */
    // 将 seedbuf 添加到 command 的末尾，保证不超过 command 的大小
    strncat(command, seedbuf, sizeof(command) - strlen(command) - 1);
    // 查找 command 中的 " t " 并删除之
    t= strstr(command, " t ");
    if (t)
      strcpy(t+1, t+3); /* remove " t " */
  // 否则，种子被显式设置为 n
  } /* else, seed explicitly set to n */
  // 使用 qh_RANDOMseed_ 函数设置 QHull 库的随机数种子
  qh_RANDOMseed_(qh, seed);

  /* ============= 打印头部信息 =============== */

  // 如果是 CDD 格式
  if (iscdd)
      // 打印格式化的头部信息到输出文件流 qh->fout 中
      qh_fprintf_rbox(qh, qh->fout, 9391, "%s\nbegin\n        %d %d %s\n",
      NOcommand ? "" : command,
      totpoints, dim+1,
      qh->rbox_isinteger ? "integer" : "real");
  // 如果没有 command 信息
  else if (NOcommand)
      // 打印格式化的简化头部信息到输出文件流 qh->fout 中
      qh_fprintf_rbox(qh, qh->fout, 9392, "%d\n%d\n", dim, totpoints);
  else
      /* qh_fprintf_rbox special cases 9393 to append 'command' to the RboxPoints.comment() */
      // 打印格式化的头部信息，包括 command 到输出文件流 qh->fout 中
      qh_fprintf_rbox(qh, qh->fout, 9393, "%d %s\n%d\n", dim, command, totpoints);

  /* ============= 显式指定的点 =============== */
  // 如果存在 first_point
  if ((s= first_point)) {
    // 遍历 first_point 中的点
    while (s && *s) { /* 'P' */
      count= 0;
      // 如果是 CDD 格式
      if (iscdd)
        // 输出值为 1.0 到输出流 qh->fout
        qh_out1(qh, 1.0);
      // 遍历解析点的坐标值
      while (*++s) {
        qh_out1(qh, qh_strtod(s, &s));
        count++;
        // 遇到空白字符或末尾时结束当前点的处理
        if (isspace(*s) || !*s)
          break;
        // 如果不是逗号，则报错缺少逗号
        if (*s != ',') {
          qh_fprintf_rbox(qh, qh->ferr, 6194, "rbox error: missing comma after coordinate in %s\n\n", s);
          qh_errexit_rbox(qh, qh_ERRinput);
        }
      }
      // 如果坐标值少于维度 dim，则补充为 0.0
      if (count < dim) {
        for (k=dim-count; k--; )
          qh_out1(qh, 0.0);
      // 如果坐标值多于维度 dim，则报错
      }else if (count > dim) {
        qh_fprintf_rbox(qh, qh->ferr, 6195, "rbox error: %d coordinates instead of %d coordinates in %s\n\n",
                  count, dim, s);
        qh_errexit_rbox(qh, qh_ERRinput);
      }
      // 输出空行到输出流 qh->fout
      qh_fprintf_rbox(qh, qh->fout, 9394, "\n");
      // 查找下一个 'P' 字符
      while ((s= strchr(s, 'P'))) {
        if (isspace(s[-1]))
          break;
      }
    }
  }

  /* ============= 简单分布 =============== */
  // 如果 issimplex 或 issimplex2 为真
  if (issimplex+issimplex2) {
    // 分配内存给 simplex 数组，存储 dim * (dim+1) 个 double 类型数据
    if (!(*simplex= (double *)qh_malloc( (size_t)(dim * (dim+1)) * sizeof(double)))) {
      qh_fprintf_rbox(qh, qh->ferr, 6196, "rbox error: insufficient memory for simplex\n");
      qh_errexit_rbox(qh, qh_ERRmem); /* qh_ERRmem */
    }
    // simplexp 指向 simplex 数组
    simplexp= *simplex;
    // 如果是 regular 分布
    if (isregular) {
      // 设置正交单位向量组成的 simplex
      for (i=0; i<dim; i++) {
        for (k=0; k<dim; k++)
          *(simplexp++)= i==k ? 1.0 : 0.0;
      }
      // 添加额外的 -1.0 到 simplexp 中
      for (k=0; k<dim; k++)
        *(simplexp++)= -1.0;
    }else {
      // 否则，随机生成 simplex
      for (i=0; i<dim+1; i++) {
        for (k=0; k<dim; k++) {
          randr= qh_RANDOMint;
          *(simplexp++)= 2.0 * randr/randmax - 1.0;
        }
      }
    }
    /* 如果 issimplex2 为真，则进入简单形式处理 */
    if (issimplex2) {
        /* 将 simplexp 指向 simplex 的内容 */
        simplexp= *simplex;
        /* 遍历 dim+1 次 */
        for (i=0; i<dim+1; i++) {
            /* 如果 iscdd 为真，则输出 1.0 */
            if (iscdd)
                qh_out1(qh, 1.0);
            /* 遍历 dim 次 */
            for (k=0; k<dim; k++)
                /* 输出 *(simplexp++) * box 的结果 */
                qh_out1(qh, *(simplexp++) * box);
            /* 输出一个换行符到 qh->fout */
            qh_fprintf_rbox(qh, qh->fout, 9395, "\n");
        }
    }
    /* 遍历 numpoints 次 */
    for (j=0; j<numpoints; j++) {
        /* 如果 iswidth 为真，则计算 apex 为随机数模 (dim+1) */
        if (iswidth)
            apex= qh_RANDOMint % (dim+1);
        else
            apex= -1;
        /* 将 coord[k] 初始化为 0.0 */
        for (k=0; k<dim; k++)
            coord[k]= 0.0;
        /* 将 norm 初始化为 0.0 */
        norm= 0.0;
        /* 遍历 dim+1 次 */
        for (i=0; i<dim+1; i++) {
            /* 生成随机数 randr */
            randr= qh_RANDOMint;
            /* 计算 factor 为 randr/randmax */
            factor= randr/randmax;
            /* 如果 i 等于 apex，则将 factor 乘以 width */
            if (i == apex)
                factor *= width;
            /* 计算 norm += factor */
            norm += factor;
            /* 遍历 dim 次 */
            for (k=0; k<dim; k++) {
                /* 计算 simplexp 为 *simplex + i*dim + k */
                simplexp= *simplex + i*dim + k;
                /* 计算 coord[k] += factor * (*simplexp) */
                coord[k] += factor * (*simplexp);
            }
        }
        /* 遍历 dim 次 */
        for (k=0; k<dim; k++)
            /* 计算 coord[k] *= box/norm */
            coord[k] *= box/norm;
        /* 输出坐标到 qh */
        qh_outcoord(qh, iscdd, coord, dim);
        /* 如果 coincidentcount++ 小于 coincidenttotal，则输出重合点 */
        if(coincidentcount++ < coincidenttotal)
            qh_outcoincident(qh, coincidentpoints, coincidentradius, iscdd, coord, dim);
    }
    /* 将 isregular 设为 0，继续 isbox */
    isregular= 0; /* continue with isbox */
    /* 将 numpoints 设为 0 */
    numpoints= 0;
  }

  /* ============= mesh distribution =============== */
  /* 如果 ismesh 为真，则进行网格分布 */
  if (ismesh) {
      /* 计算 nthroot 为 numpoints 的 dim 次方根加 0.99999 后取整 */
      nthroot= (int)(pow((double)numpoints, 1.0/dim) + 0.99999);
      /* 将 mult[k] 初始化为 0 */
      for (k=dim; k--; )
          mult[k]= 0;
      /* 遍历 numpoints 次 */
      for (i=0; i < numpoints; i++) {
          /* 将 coordp 设为 coord */
          coordp= coord;
          /* 遍历 dim 次 */
          for (k=0; k < dim; k++) {
              /* 如果 k 等于 0，则 *(coordp++)= mult[0] * meshn + mult[1] * (-meshm) */
              if (k == 0)
                  *(coordp++)= mult[0] * meshn + mult[1] * (-meshm);
              /* 如果 k 等于 1，则 *(coordp++)= mult[0] * meshm + mult[1] * meshn */
              else if (k == 1)
                  *(coordp++)= mult[0] * meshm + mult[1] * meshn;
              /* 否则 *(coordp++)= mult[k] * meshr */
              else
                  *(coordp++)= mult[k] * meshr;
          }
          /* 输出坐标到 qh */
          qh_outcoord(qh, iscdd, coord, dim);
          /* 如果 coincidentcount++ 小于 coincidenttotal，则输出重合点 */
          if(coincidentcount++ < coincidenttotal)
              qh_outcoincident(qh, coincidentpoints, coincidentradius, iscdd, coord, dim);
          /* 遍历 dim 次 */
          for (k=0; k < dim; k++) {
              /* 如果 ++mult[k] 小于 nthroot，则跳出循环 */
              if (++mult[k] < nthroot)
                  break;
              /* 否则将 mult[k] 设为 0 */
              mult[k]= 0;
          }
      }
  }
  /* ============= regular points for 's' =============== */
  /* 否则如果 isregular 为真且 !islens 为真，则处理规则点 's' */
  else if (isregular && !islens) {
      /* 如果 dim 不等于 2 且不等于 3，则输出错误信息并退出 */
      if (dim != 2 && dim != 3) {
          qh_fprintf_rbox(qh, qh->ferr, 6197, "rbox error: regular points can be used only in 2-d and 3-d\n\n");
          qh_errexit_rbox(qh, qh_ERRinput);
      }
      /* 如果 !isaxis 或 radius 等于 0.0，则设 isaxis 为 1，radius 为 1.0 */
      if (!isaxis || radius == 0.0) {
          isaxis= 1;
          radius= 1.0;
      }
      /* 如果 dim 等于 3 且 iscdd 为真，则输出 1.0 */
      if (dim == 3) {
          if (iscdd)
              qh_out1(qh, 1.0);
          /* 输出三维坐标 (0.0, 0.0, -box) */
          qh_out3n(qh, 0.0, 0.0, -box);
          /* 如果 !isgap，则如果 iscdd 为真则输出 1.0，再输出 (0.0, 0.0, box) */
          if (!isgap) {
              if (iscdd)
                  qh_out1(qh, 1.0);
              qh_out3n(qh, 0.0, 0.0, box);
          }
      }
      /* 将 angle 设为 0.0 */
      angle= 0.0;
      /* 计算 anglediff 为 2.0 * PI/numpoints */
      anglediff= 2.0 * PI/numpoints;
  for (i=0; i < numpoints; i++) {
    // 计算当前角度
    angle += anglediff;
    // 根据极坐标计算点的 x, y 坐标
    x= radius * cos(angle);
    y= radius * sin(angle);
    // 如果是二维空间
    if (dim == 2) {
      // 如果需要计算 CDD 输出
      if (iscdd)
        qh_out1(qh, 1.0);
      // 输出二维坐标乘以箱子尺寸
      qh_out2n(qh, x*box, y*box);
    }else {
      // 计算点到原点的距离
      norm= sqrt(1.0 + x*x + y*y);
      // 如果需要计算 CDD 输出
      if (iscdd)
        qh_out1(qh, 1.0);
      // 输出三维坐标乘以箱子尺寸，并且归一化
      qh_out3n(qh, box*x/norm, box*y/norm, box/norm);
      // 如果需要创建间隙
      if (isgap) {
        // 减少坐标点的半径
        x *= 1-gap;
        y *= 1-gap;
        // 重新计算归一化后的坐标
        norm= sqrt(1.0 + x*x + y*y);
        // 如果需要计算 CDD 输出
        if (iscdd)
          qh_out1(qh, 1.0);
        // 输出三维坐标乘以箱子尺寸，并且归一化
        qh_out3n(qh, box*x/norm, box*y/norm, box/norm);
      }
    }
  }
}
/* ============= regular points for 'r Ln D2' =============== */
else if (isregular && islens && dim == 2) {
  double cos_0;

  // 设置初始角度和角度差
  angle= lensangle;
  anglediff= 2 * lensangle/(numpoints - 1);
  // 计算 cos(lensangle)
  cos_0= cos(lensangle);
  // 循环生成规则点
  for (i=0; i < numpoints; i++, angle -= anglediff) {
    // 计算点的 x, y 坐标
    x= radius * sin(angle);
    y= radius * (cos(angle) - cos_0);
    // 如果需要计算 CDD 输出
    if (iscdd)
      qh_out1(qh, 1.0);
    // 输出二维坐标乘以箱子尺寸
    qh_out2n(qh, x*box, y*box);
    // 如果不是第一个和最后一个点
    if (i != 0 && i != numpoints - 1) {
      // 如果需要计算 CDD 输出
      if (iscdd)
        qh_out1(qh, 1.0);
      // 输出二维坐标乘以箱子尺寸，y 取相反数
      qh_out2n(qh, x*box, -y*box);
    }
  }
}
/* ============= regular points for 'r Ln D3' =============== */
else if (isregular && islens && dim != 2) {
  // 如果维度不是二维或三维，则报错退出
  if (dim != 3) {
    qh_fprintf_rbox(qh, qh->ferr, 6198, "rbox error: regular points can be used only in 2-d and 3-d\n\n");
    qh_errexit_rbox(qh, qh_ERRinput);
  }
  // 设置初始角度和角度差
  angle= 0.0;
  anglediff= 2* PI/numpoints;
  // 如果不需要间隙，默认设置 gap 和 isgap
  if (!isgap) {
    isgap= 1;
    gap= 0.5;
  }
  // 计算偏移量
  offset= sqrt(radius * radius - (1-gap)*(1-gap)) - lensbase;
  // 循环生成规则点
  for (i=0; i < numpoints; i++, angle += anglediff) {
    // 计算点的 x, y 坐标
    x= cos(angle);
    y= sin(angle);
    // 如果需要计算 CDD 输出
    if (iscdd)
      qh_out1(qh, 1.0);
    // 输出三维坐标乘以箱子尺寸，并且 z 为 0
    qh_out3n(qh, box*x, box*y, 0.0);
    // 减少坐标点的半径
    x *= 1-gap;
    y *= 1-gap;
    // 如果需要计算 CDD 输出
    if (iscdd)
      qh_out1(qh, 1.0);
    // 输出三维坐标乘以箱子尺寸，并且 z 为 offset 或 -offset
    qh_out3n(qh, box*x, box*y, box * offset);
    if (iscdd)
      qh_out1(qh, 1.0);
    qh_out3n(qh, box*x, box*y, -box * offset);
  }
}
/* ============= apex of 'Zn' distribution + gendim =============== */
else {
  // 如果是轴对齐
  if (isaxis) {
    // 设置 gendim
    gendim= dim-1;
    // 如果需要计算 CDD 输出
    if (iscdd)
      qh_out1(qh, 1.0);
    // 输出 gendim 个 0，最后输出 -box
    for (j=0; j < gendim; j++)
      qh_out1(qh, 0.0);
    qh_out1(qh, -box);
    // 输出换行符
    qh_fprintf_rbox(qh, qh->fout, 9398, "\n");
  }else if (islens)
    // 设置 gendim
    gendim= dim-1;
  else
    // 设置 gendim
    gendim= dim;
  /* ============= generate random point in unit cube =============== */
}

/* ============= write cube vertices =============== */
if (addcube) {
  // 循环写入立方体顶点
  for (j=0; j<cubesize; j++) {
    // 如果需要计算 CDD 输出
    if (iscdd)
      qh_out1(qh, 1.0);
    // 生成每个维度的顶点坐标
    for (k=dim-1; k>=0; k--) {
      if (j & ( 1 << k))
        qh_out1(qh, cube);
      else
        qh_out1(qh, -cube);
    }
    // 输出换行符
    qh_fprintf_rbox(qh, qh->fout, 9400, "\n");
  }
}

/* ============= write diamond vertices =============== */
if (adddiamond) {
    // 遍历钻石尺寸范围内的索引 j
    for (j=0; j<diamondsize; j++) {
      // 如果 iscdd 为真，输出 1.0
      if (iscdd)
        qh_out1(qh, 1.0);
      // 从最高维度向最低维度遍历钻石的每个顶点
      for (k=dim-1; k>=0; k--) {
        // 如果 j 除以 2 不等于 k，则输出 0.0
        if (j/2 != k)
          qh_out1(qh, 0.0);
        // 否则，如果 j 的最低位为 1，则输出 diamond
        else if (j & 0x1)
          qh_out1(qh, diamond);
        // 否则，输出 -diamond
        else
          qh_out1(qh, -diamond);
      }
      // 输出钻石顶点数据后，输出换行符
      qh_fprintf_rbox(qh, qh->fout, 9401, "\n");
    }
  }

  // 如果 iscdd 为真，输出 "end\nhull\n"
  if (iscdd)
    qh_fprintf_rbox(qh, qh->fout, 9402, "end\nhull\n");
/*------------------------------------------------
outxxx - output functions for qh_rboxpoints
*/
/* 
   返回四舍五入后的整数值，用于将浮点数转换为整数
   如果浮点数小于零，进行向下取整，避免超出整数表示范围
   如果浮点数大于或等于零，进行向上取整，避免超出整数表示范围
*/
int qh_roundi(qhT *qh, double a) {
  if (a < 0.0) {
    if (a - 0.5 < INT_MIN) {
      qh_fprintf_rbox(qh, qh->ferr, 6200, "rbox input error: negative coordinate %2.2g is too large.  Reduce 'Bn'\n", a);
      qh_errexit_rbox(qh, qh_ERRinput);
    }
    return (int)(a - 0.5);
  }else {
    if (a + 0.5 > INT_MAX) {
      qh_fprintf_rbox(qh, qh->ferr, 6201, "rbox input error: coordinate %2.2g is too large.  Reduce 'Bn'\n", a);
      qh_errexit_rbox(qh, qh_ERRinput);
    }
    return (int)(a + 0.5);
  }
} /* qh_roundi */

/* 
   输出单个浮点数到输出流中，根据配置输出整数或实数格式
*/
void qh_out1(qhT *qh, double a) {

  if (qh->rbox_isinteger)
    qh_fprintf_rbox(qh, qh->fout, 9403, "%d ", qh_roundi(qh, a+qh->rbox_out_offset));
  else
    qh_fprintf_rbox(qh, qh->fout, 9404, qh_REAL_1, a+qh->rbox_out_offset);
} /* qh_out1 */

/* 
   输出两个浮点数到输出流中，根据配置输出整数或实数格式
*/
void qh_out2n(qhT *qh, double a, double b) {

  if (qh->rbox_isinteger)
    qh_fprintf_rbox(qh, qh->fout, 9405, "%d %d\n", qh_roundi(qh, a+qh->rbox_out_offset), qh_roundi(qh, b+qh->rbox_out_offset));
  else
    qh_fprintf_rbox(qh, qh->fout, 9406, qh_REAL_2n, a+qh->rbox_out_offset, b+qh->rbox_out_offset);
} /* qh_out2n */

/* 
   输出三个浮点数到输出流中，根据配置输出整数或实数格式
*/
void qh_out3n(qhT *qh, double a, double b, double c) {

  if (qh->rbox_isinteger)
    qh_fprintf_rbox(qh, qh->fout, 9407, "%d %d %d\n", qh_roundi(qh, a+qh->rbox_out_offset), qh_roundi(qh, b+qh->rbox_out_offset), qh_roundi(qh, c+qh->rbox_out_offset));
  else
    qh_fprintf_rbox(qh, qh->fout, 9408, qh_REAL_3n, a+qh->rbox_out_offset, b+qh->rbox_out_offset, c+qh->rbox_out_offset);
} /* qh_out3n */

/* 
   输出坐标数组到输出流中，根据配置输出整数或实数格式
*/
void qh_outcoord(qhT *qh, int iscdd, double *coord, int dim) {
    double *p= coord;
    int k;

    if (iscdd)
      qh_out1(qh, 1.0);
    for (k=0; k < dim; k++)
      qh_out1(qh, *(p++));
    qh_fprintf_rbox(qh, qh->fout, 9396, "\n");
} /* qh_outcoord */

/* 
   输出多个具有相同坐标的点到输出流中，根据配置输出整数或实数格式
   在每个坐标上加上随机偏移量以模拟重叠点
*/
void qh_outcoincident(qhT *qh, int coincidentpoints, double radius, int iscdd, double *coord, int dim) {
  double *p;
  double randr, delta;
  int i,k;
  double randmax= qh_RANDOMmax;

  for (i=0; i<coincidentpoints; i++) {
    p= coord;
    if (iscdd)
      qh_out1(qh, 1.0);
    for (k=0; k < dim; k++) {
      randr= qh_RANDOMint;
      delta= 2.0 * randr/randmax - 1.0; /* -1..+1 */
      delta *= radius;
      qh_out1(qh, *(p++) + delta);
    }
    qh_fprintf_rbox(qh, qh->fout, 9410, "\n");
  }
} /* qh_outcoincident */

/*------------------------------------------------
   只从 qh_rboxpoints2 或 qh_fprintf_rbox 调用
   qh_fprintf_rbox 仅从 qh_rboxpoints2 调用
   为与 exit() 兼容，最大的退出码是 '255'
*/
/* 
   在错误情况下，通过 longjmp 跳转到错误处理函数，结束程序执行
*/
void qh_errexit_rbox(qhT *qh, int exitcode)
{
    longjmp(qh->rbox_errexit, exitcode);
} /* qh_errexit_rbox */
```