# `D:\src\scipysrc\scipy\scipy\optimize\_direct\DIRsubrout.c`

```
/* DIRsubrout.f -- translated by f2c (version 20050501).

   f2c output hand-cleaned by SGJ (August 2007).
*/

#include "direct-internal.h"
#include <math.h>
// #include "numpy/ndarrayobject.h"

/* Table of constant values */

static integer c__1 = 1;          // 定义常数 c__1 为整数 1
static integer c__32 = 32;        // 定义常数 c__32 为整数 32
static integer c__0 = 0;          // 定义常数 c__0 为整数 0

/* +-----------------------------------------------------------------------+ */
/* | INTEGER Function DIRGetlevel                                          | */
/* | Returns the level of the hyperrectangle. Depending on the value of the| */
/* | global variable JONES. IF JONES equals 0, the level is given by       | */
/* |               kN + p, where the rectangle has p sides with a length of| */
/* |             1/3^(k+1), and N-p sides with a length of 1/3^k.          | */
/* | If JONES equals 1, the level is the power of 1/3 of the length of the | */
/* | longest side hyperrectangle.                                          | */
/* |                                                                       | */
/* | On Return :                                                           | */
/* |    the maximal length                                                 | */
/* |                                                                       | */
/* | pos     -- the position of the midpoint in the array length           | */
/* | length  -- the array with the dimensions                              | */
/* | maxfunc -- the leading dimension of length                            | */
/* | n       -- the dimension of the problem                                | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
integer direct_dirgetlevel_(integer *pos, integer *length, integer *maxfunc, integer
    *n, integer jones)
{
    /* System generated locals */
    integer length_dim1, length_offset, ret_val, i__1;

    /* Local variables */
    integer i__, k, p, help;

    (void) maxfunc;

/* JG 09/15/00 Added variable JONES (see above) */
    /* Parameter adjustments */
    length_dim1 = *n;               // 维度为 n 的数组 length 的第一维
    length_offset = 1 + length_dim1;
    length -= length_offset;        // 调整数组 length 的指针，使其从偏移量 1 + length_dim1 处开始

    /* Function Body */
    if (jones == 0) {               // 如果 jones 等于 0
        help = length[*pos * length_dim1 + 1];   // 获取 length 数组中指定位置的第一个元素
        k = help;                   // 设置 k 为 help
        p = 1;                      // 设置 p 为 1
        i__1 = *n;                  // 设置循环上限为 n
        for (i__ = 2; i__ <= i__1; ++i__) {  // 循环遍历数组 length 的每个维度
            if (length[i__ + *pos * length_dim1] < k) {  // 如果当前维度长度小于 k
                k = length[i__ + *pos * length_dim1];   // 更新 k 为当前维度长度
            }
            if (length[i__ + *pos * length_dim1] == help) {  // 如果当前维度长度等于 help
                ++p;                // 增加 p 的计数
            }
/* L100: */
        }
        if (k == help) {            // 如果 k 等于 help
            ret_val = k * *n + *n - p;  // 计算返回值为 k * n + n - p
        } else {
            ret_val = k * *n + p;   // 计算返回值为 k * n + p
        }
    } else {                        // 如果 jones 不等于 0
        help = length[*pos * length_dim1 + 1];   // 获取 length 数组中指定位置的第一个元素
        i__1 = *n;                  // 设置循环上限为 n
        for (i__ = 2; i__ <= i__1; ++i__) {  // 循环遍历数组 length 的每个维度
            if (length[i__ + *pos * length_dim1] < help) {  // 如果当前维度长度小于 help
                help = length[i__ + *pos * length_dim1];   // 更新 help 为当前维度长度
            }
/* L10: */
        }
        ret_val = help;             // 返回值为 help
    }
    return ret_val;                 // 返回最终计算结果
} /* dirgetlevel_ */
/* +-----------------------------------------------------------------------+ */
/* | Program       : Direct.f (subfile DIRsubrout.f)                       | */
/* | Last modified : 07-16-2001                                            | */
/* | Written by    : Joerg Gablonsky                                       | */
/* | Subroutines used by the algorithm DIRECT.                             | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRChoose                                               | */
/* |    Decide, which is the next sampling point.                          | */
/* |    Changed 09/25/00 JG                                                | */
/* |         Added maxdiv to call and changed S to size maxdiv.            | */
/* |    Changed 01/22/01 JG                                                | */
/* |         Added Ifeasiblef to call to keep track if a feasible point has| */
/* |         been found.                                                   | */
/* |    Changed 07/16/01 JG                                                | */
/* |         Changed if statement to prevent run-time errors.              |
                 | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirchoose_(integer *anchor, integer *s, integer *actdeep,
     doublereal *f, doublereal *minf, doublereal epsrel, doublereal epsabs, doublereal *thirds,
     integer *maxpos, integer *length, integer *maxfunc, const integer *maxdeep,
     const integer *maxdiv, integer *n, FILE *logfile,
    integer *cheat, doublereal *kmax, integer *ifeasiblef, integer jones)
{
    /* System generated locals */
    integer s_dim1, s_offset, length_dim1, length_offset, i__1;

    /* Local variables */
    integer i__, j, k;
    doublereal helplower;
    integer i___, j___;
    doublereal helpgreater;
    integer novaluedeep = 0;
    doublereal help2;
    integer novalue;

    /* Parameter adjustments */
    f -= 3;                         /* Adjusts base index of array f */
    ++anchor;                       /* Adjusts base index of array anchor */
    s_dim1 = *maxdiv;               /* Sets first dimension of s to maxdiv */
    s_offset = 1 + s_dim1;          /* Computes offset for 2D array s */
    s -= s_offset;                  /* Adjusts base index of 2D array s */
    length_dim1 = *n;               /* Sets first dimension of length to n */
    length_offset = 1 + length_dim1; /* Computes offset for 2D array length */
    length -= length_offset;        /* Adjusts base index of 2D array length */

    /* Function Body */
    helplower = HUGE_VAL;           /* Initializes helplower with a large value */
    helpgreater = 0.;               /* Initializes helpgreater to zero */
    k = 1;                          /* Initializes k to 1 */
    if (*ifeasiblef >= 1) {         /* Checks if feasible point has been found */
    i__1 = *actdeep;
    for (j = 0; j <= i__1; ++j) {   /* Loop over actdeep */
        if (anchor[j] > 0) {        /* Checks if anchor[j] is positive */
        s[k + s_dim1] = anchor[j];  /* Sets s[k][1] to anchor[j] */
        s[k + (s_dim1 << 1)] = direct_dirgetlevel_(&s[k + s_dim1], &length[
            length_offset], maxfunc, n, jones); /* Calls direct_dirgetlevel_ */
        goto L12;                   /* Jumps to label L12 */
        }
/* L1001: */
    }
L12:
    ++k;                            /* Increments k */
    *maxpos = 1;                    /* Sets maxpos to 1 */
    return;                         /* Returns from subroutine */
    } else {
    i__1 = *actdeep;
    /* (remaining code would continue here, but it's not provided in this snippet) */
    }
}
    for (j = 0; j <= i__1; ++j) {
        # 循环遍历 j 从 0 到 i__1
        if (anchor[j] > 0) {
            # 如果 anchor[j] 大于 0，则执行以下操作
            s[k + s_dim1] = anchor[j];
            # 将 anchor[j] 赋值给 s 数组中的特定位置
            s[k + (s_dim1 << 1)] = direct_dirgetlevel_(&s[k + s_dim1], &length[
                length_offset], maxfunc, n, jones);
            # 调用 direct_dirgetlevel_ 函数，将其返回值赋值给 s 数组中的特定位置
            ++k;
            # 增加 k 的值
        }
    }
    }
    }
    // 初始化 novalue 变量为 0
    novalue = 0;
    // 如果 anchor 数组的最后一个元素大于 0，则将 novalue 设置为该元素的值
    if (anchor[-1] > 0) {
    novalue = anchor[-1];
    // 使用 direct_dirgetlevel_ 函数计算 novaluedeep 的值
    novaluedeep = direct_dirgetlevel_(&novalue, &length[length_offset], maxfunc,
        n, jones);
    }
    // 将 maxpos 指针指向的位置设置为 k-1
    *maxpos = k - 1;
    // 循环，从 k-1 到 *maxdeep
    i__1 = *maxdeep;
    for (j = k - 1; j <= i__1; ++j) {
    // 将 s[k+s_dim1] 的值设为 0
    s[k + s_dim1] = 0;
/* L11: */
    }
    // 倒序循环，从 *maxpos 到 1
    for (j = *maxpos; j >= 1; --j) {
    // 初始化 helplower 为 HUGE_VAL
    helplower = HUGE_VAL;
    // 初始化 helpgreater 为 0
    helpgreater = 0.;
    // 从 s[j+s_dim1] 处获取 j___ 的值
    j___ = s[j + s_dim1];
    // 循环，从 1 到 j-1
    i__1 = j - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        // 从 s[i__+s_dim1] 处获取 i___ 的值
        i___ = s[i__ + s_dim1];
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Changed IF statement into two to prevent run-time errors  | */
/* |             which could occur if the compiler checks the second       | */
/* |             expression in an .AND. statement although the first       | */
/* |             statement is already not true.                            | */
/* +-----------------------------------------------------------------------+ */
        // 如果 i___ 大于 0 并且 i__ 不等于 j
        if (i___ > 0 && ! (i__ == j)) {
        // 如果 f[(i___ << 1) + 2] <= 1
        if (f[(i___ << 1) + 2] <= 1.) {
            // 计算 help2 的值
            help2 = thirds[s[i__ + (s_dim1 << 1)]] - thirds[s[j + (
                s_dim1 << 1)]];
            help2 = (f[(i___ << 1) + 1] - f[(j___ << 1) + 1]) / help2;
            // 如果 help2 <= 0
            if (help2 <= 0.) {
/*              if (logfile)                                                 */
/*                  fprintf(logfile, "thirds > 0, help2 <= 0\n");            */
            // 跳转到标签 L60
            goto L60;
            }
            // 如果 help2 小于 helplower
            if (help2 < helplower) {
/*              if (logfile)                                                 */
/*                  fprintf(logfile, "helplower = %g\n", help2);             */
            // 将 helplower 设置为 help2 的值
            helplower = help2;
            }
        }
        }
/* L30: */
    }
    // 循环，从 j+1 到 *maxpos
    i__1 = *maxpos;
    for (i__ = j + 1; i__ <= i__1; ++i__) {
        // 从 s[i__+s_dim1] 处获取 i___ 的值
        i___ = s[i__ + s_dim1];
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Changed IF statement into two to prevent run-time errors  | */
/* |             which could occur if the compiler checks the second       | */
/* |             expression in an .AND. statement although the first       | */
/* |             statement is already not true.                            | */
/* +-----------------------------------------------------------------------+ */
        // 如果 i___ 大于 0 并且 i__ 不等于 j
        if (i___ > 0 && ! (i__ == j)) {
        // 如果 f[(i___ << 1) + 2] <= 1
        if (f[(i___ << 1) + 2] <= 1.) {
            // 计算 help2 的值
            help2 = thirds[s[i__ + (s_dim1 << 1)]] - thirds[s[j + (
                s_dim1 << 1)]];
            help2 = (f[(i___ << 1) + 1] - f[(j___ << 1) + 1]) / help2;
            // 如果 help2 <= 0
            if (help2 <= 0.) {
            // 如果 logfile 存在，输出日志信息
            if (logfile)
                 fprintf(logfile, "thirds < 0, help2 <= 0\n");
            // 跳转到标签 L60
            goto L60;
            }
            // 如果 help2 大于 helpgreater
            if (help2 > helpgreater) {
/*            if (logfile)                                                   */

/*            if (logfile)                                                   */
/*                fprintf(logfile, "helpgreater = %g\n", help2);              */
            // 将 helpgreater 设置为 help2 的值
            helpgreater = help2;
            }
        }
/* L60: */
    }


这段代码是一个较复杂的算法实现，主要用于计算一些数值和记录日志。注释详细解释了每个变量和条件语句的作用，以及相关的跳转标签。
/*                  fprintf(logfile, "helpgreater = %g\n", help2);           */
            helpgreater = help2;  // 将 help2 的值赋给 helpgreater
            }
        }
        }
/* L31: */
    }
    if (helpgreater <= helplower) {  // 如果 helpgreater 小于等于 helplower
        if (*cheat == 1 && helplower > *kmax) {  // 如果 cheat 等于 1 且 helplower 大于 kmax 指向的值
        helplower = *kmax;  // 将 kmax 指向的值赋给 helplower
        }
        if (f[(j___ << 1) + 1] - helplower * thirds[s[j + (s_dim1 << 1)]] >
             MIN(*minf - epsrel * fabs(*minf),
             *minf - epsabs)) {  // 如果条件成立
        if (logfile)
             fprintf(logfile, "> minf - epslminfl\n");  // 输出日志信息到 logfile
        goto L60;  // 转到标号 L60 处
        }
    } else {
        if (logfile)
/*         fprintf(logfile, "helpgreater > helplower: %g  %g  %g\n",         */
/*             helpgreater, helplower, helpgreater - helplower);             */
        goto L60;  // 转到标号 L60 处
    }
    goto L40;  // 转到标号 L40 处
L60:
    s[j + s_dim1] = 0;  // 将 s 数组中指定位置的值设为 0
L40:
    ;
    }
    if (novalue > 0) {  // 如果 novalue 大于 0
    ++(*maxpos);  // maxpos 指向的值加 1
    s[*maxpos + s_dim1] = novalue;  // 将 novalue 赋给 s 数组中指定位置的值
    s[*maxpos + (s_dim1 << 1)] = novaluedeep;  // 将 novaluedeep 赋给 s 数组中指定位置的值
    }
} /* dirchoose_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRDoubleInsert                                         | */
/* |      Routine to make sure that if there are several potential optimal | */
/* |      hyperrectangles of the same level (i.e. hyperrectangles that have| */
/* |      the same level and the same function value at the center), all of| */
/* |      them are divided. This is the way as originally described in     | */
/* |      Jones et.al.                                                     | */
/* | JG 07/16/01 Added errorflag to calling sequence. We check if more     | */
/* |             we reach the capacity of the array S. If this happens, we | */
/* |             return to the main program with an error.                 | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirdoubleinsert_(integer *anchor, integer *s, integer *
    maxpos, integer *point, doublereal *f, const integer *maxdeep, integer *
    maxfunc, const integer *maxdiv, integer *ierror)
{
    /* System generated locals */
    integer s_dim1, s_offset, i__1;

    /* Local variables */
    integer i__, oldmaxpos, pos, help, iflag, actdeep;

    (void)  maxdeep; (void) maxfunc;

/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Added flag to prevent run time-errors on some systems.    | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    ++anchor;
    f -= 3;
    --point;
    s_dim1 = *maxdiv;
    s_offset = 1 + s_dim1;
    s -= s_offset;

    /* Function Body */
    oldmaxpos = *maxpos;
    i__1 = oldmaxpos;
    for (i__ = 1; i__ <= i__1; ++i__) {
    if (s[i__ + s_dim1] > 0) {  // 如果 s 数组中指定位置的值大于 0
        actdeep = s[i__ + (s_dim1 << 1)];  // 将 s 数组中指定位置的值赋给 actdeep
        help = anchor[actdeep];  // 将 anchor 数组中指定位置的值赋给 help
        pos = point[help];  // 将 point 数组中指定位置的值赋给 pos
        iflag = 0;  // 将 iflag 设为 0
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Added flag to prevent run time-errors on some systems. On | */
/* |             some systems the second conditions in an AND statement is | */
/* |             evaluated even if the first one is already not true.      | */
/* +-----------------------------------------------------------------------+ */
/* 在某些系统上，添加了标志以防止运行时错误。在某些系统上，即使第一个条件已经不成立，AND语句中的第二个条件也会被评估。 */
        while(pos > 0 && iflag == 0) {
        /* 如果指定条件成立，则执行以下代码块 */
        if (f[(pos << 1) + 1] - f[(help << 1) + 1] <= 1e-13) {
            /* 如果最大位置小于最大分割数，则执行以下代码块 */
            if (*maxpos < *maxdiv) {
            ++(*maxpos);
            /* 更新数组s中的元素 */
            s[*maxpos + s_dim1] = pos;
            s[*maxpos + (s_dim1 << 1)] = actdeep;
            /* 更新pos的值 */
            pos = point[pos];
            } else {
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Maximum number of elements possible in S has been reached!| */
/* +-----------------------------------------------------------------------+ */
/* 达到了数组S中可能的最大元素数量！ */
            *ierror = -6;
            /* 返回到调用函数处 */
            return;
            }
        } else {
            /* 如果指定条件不成立，则执行以下代码块 */
            iflag = 1;
        }
        /* 继续循环直到条件不再满足 */
        }
    }
/* L10: */
/* 结束while循环的标签 */
    }
} /* dirdoubleinsert_ */

/* +-----------------------------------------------------------------------+ */
/* | INTEGER Function GetmaxDeep                                           | */
/* | function to get the maximal length (1/length) of the n-dimensional    | */
/* | rectangle with midpoint pos.                                          | */
/* |                                                                       | */
/* | On Return :                                                           | */
/* |    the maximal length                                                 | */
/* |                                                                       | */
/* | pos     -- the position of the midpoint in the array length           | */
/* | length  -- the array with the dimensions                              | */
/* | maxfunc -- the leading dimension of length                            | */
/* | n       -- the dimension of the problem                                | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
/* 获取n维矩形以pos为中点时的最大长度的函数 */
integer direct_dirgetmaxdeep_(integer *pos, integer *length, integer *maxfunc,
    integer *n)
{
    /* System generated locals */
    integer length_dim1, length_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, help;

    /* 忽略maxfunc未使用的警告 */
    (void) maxfunc;

    /* Parameter adjustments */
    /* 调整参数 */
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Function Body */
    /* 函数体 */
    /* 初始化help变量 */
    help = length[*pos * length_dim1 + 1];
    /* 循环计算最大长度 */
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing MIN */
    /* 计算最小值 */
    i__2 = help, i__3 = length[i__ + *pos * length_dim1];
    help = MIN(i__2,i__3);
/* L10: */
    }
    /* 返回最大长度 */
    return help;
} /* dirgetmaxdeep_ */

static integer isinbox_(doublereal *x, doublereal *a, doublereal *b, integer *n,
    integer *lmaxdim)
{
    /* System generated locals */
    /* 系统生成的本地变量 */
    integer ret_val, i__1;
    /* 声明本地变量 */
    integer outofbox, i__;
    
    (void) lmaxdim;
    
    /* 函数主体 */
    outofbox = 1;  // 初始化 outofbox 为 1，表示所有点都在框内
    i__1 = *n;  // 设置循环次数为传入参数 n 的值
    for (i__ = 0; i__ < i__1; ++i__) {  // 循环遍历每个点
    if (a[i__] > x[i__] || b[i__] < x[i__]) {  // 检查点是否在指定的框之外
        outofbox = 0;  // 如果有任何点在框外，则将 outofbox 置为 0
        goto L1010;  // 跳转到标签 L1010 处继续执行
    }
/* L1000: */
    }
L1010:
    ret_val = outofbox;
    return ret_val;
} /* isinbox_ */

/* +-----------------------------------------------------------------------+ */
/* | JG Added 09/25/00                                                     | */
/* |                                                                       | */
/* |                       SUBROUTINE DIRResortlist                        | */
/* |                                                                       | */
/* | Resort the list so that the infeasible point is in the list with the  | */
/* | replaced value.                                                       | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ static void dirresortlist_(integer *replace, integer *anchor,
    doublereal *f, integer *point, integer *length, integer *n, integer *
    maxfunc, integer *maxdim, const integer *maxdeep, FILE *logfile,
                        integer jones)
{
    /* System generated locals */
    integer length_dim1, length_offset, i__1;

    /* Local variables */
    integer i__, l, pos;
    integer start;

    (void) maxdim; (void) maxdeep;

/* +-----------------------------------------------------------------------+ */
/* | Get the length of the hyper rectangle with infeasible mid point and   | */
/* | Index of the corresponding list.                                      | */
/* +-----------------------------------------------------------------------+ */
/* JG 09/25/00 Replaced with DIRgetlevel */
/*      l = DIRgetmaxDeep(replace,length,maxfunc,n) */
    /* Parameter adjustments */
    --point;
    f -= 3;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    ++anchor;

    /* Function Body */
    // 调用 DIRgetlevel 函数获取超矩形的长度及相应列表的索引
    l = direct_dirgetlevel_(replace, &length[length_offset], maxfunc, n, jones);
    // 设置起始点为锚点列表中的第 l 个位置
    start = anchor[l];
/* +-----------------------------------------------------------------------+ */
/* | If the hyper rectangle with infeasible midpoint is already the start  | */
/* | of the list, give out message, nothing to do.                         | */
/* +-----------------------------------------------------------------------+ */
    // 如果要替换的点已经是列表中的起始点，则无需重新排序
    if (*replace == start) {
/*         write(logfile,*) 'No resorting of list necessarry, since new ', */
/*     + 'point is already anchor of list .',l */
    } else {
/* +-----------------------------------------------------------------------+ */
/* | Take the hyper rectangle with infeasible midpoint out of the list.    | */
/* +-----------------------------------------------------------------------+ */
    // 将具有不可行中点的超矩形从列表中移出
    pos = start;
    i__1 = *maxfunc;
    for (i__ = 1; i__ <= i__1; ++i__) {
        // 循环，从1到i__1，逐次处理下面的逻辑
        if (point[pos] == *replace) {
            // 检查 point[pos] 是否等于指针 replace 所指向的值
            point[pos] = point[*replace];
            // 如果相等，将 point[pos] 的值设置为指针 replace 所指向的值
            goto L20;
            // 跳转到标签 L20 处继续执行
        } else {
            // 如果不相等
            pos = point[pos];
            // 将 pos 的值设置为 point[pos] 的值
        }
        if (pos == 0) {
            // 如果 pos 等于 0
            if (logfile)
                 // 如果 logfile 存在
                 fprintf(logfile, "Error in DIRREsortlist: "
                     "We went through the whole list\n"
                     "and could not find the point to replace!!\n");
            // 在日志中记录错误信息
            goto L20;
            // 跳转到标签 L20 处继续执行
        }
/* +-----------------------------------------------------------------------+ */
/* | 如果列表的锚点比附近点的值高，则将不可行点放在列表的开头。               | */
/* +-----------------------------------------------------------------------+ */
L20:
    if (f[(start << 1) + 1] > f[(*replace << 1) + 1]) {
        anchor[l] = *replace;
        point[*replace] = start;
/*            write(logfile,*) 'Point is replacing current anchor for ' */
/*     +             , 'this list ',l,replace,start */
    } else {
/* +-----------------------------------------------------------------------+ */
/* | 根据（被替换的）函数值将点插入列表中。                                  | */
/* +-----------------------------------------------------------------------+ */
        pos = start;
        i__1 = *maxfunc;
        for (i__ = 1; i__ <= i__1; ++i__) {
/* +-----------------------------------------------------------------------+ */
/* | 点必须添加到列表的末尾。                                                | */
/* +-----------------------------------------------------------------------+ */
        if (point[pos] == 0) {
            point[*replace] = point[pos];
            point[pos] = *replace;
/*                  write(logfile,*) 'Point is added at the end of the ' */
/*     +             , 'list ',l, replace */
            goto L40;
        } else {
            if (f[(point[pos] << 1) + 1] > f[(*replace << 1) + 1]) {
            point[*replace] = point[pos];
            point[pos] = *replace;
/*                     write(logfile,*) 'There are points with a higher ' */
/*     +               ,'f-value in the list ',l,replace, pos */
            goto L40;
            }
            pos = point[pos];
        }
/* L30: */
        }
L40:
        ;
    }
    }
} /* dirresortlist_ */

/* +-----------------------------------------------------------------------+ */
/* | JG Added 09/25/00                                                     | */
/* |                       SUBROUTINE DIRreplaceInf                        | */
/* |                                                                       | */
/* | Find out if there are infeasible points which are near feasible ones. | */
/* | If this is the case, replace the function value at the center of the  | */
/* | hyper rectangle by the lowest function value of a nearby function.    | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirreplaceinf_(integer *free, integer *freeold,
    doublereal *f, doublereal *c__, doublereal *thirds, integer *length,
    integer *anchor, integer *point, doublereal *c1, doublereal *c2,
    integer *maxfunc, const integer *maxdeep, integer *maxdim, integer *n,
    FILE *logfile, doublereal *fmax, integer jones)
{
    /* System generated locals */
    // 声明整型变量 c_dim1, c_offset, length_dim1, length_offset, i__1, i__2, i__3
    integer c_dim1, c_offset, length_dim1, length_offset, i__1, i__2, i__3;
    // 声明双精度浮点数组 a 和 b，各含有 32 个元素
    doublereal a[32], b[32];
    // 声明整型变量 i__, j, k, l，用于循环和索引
    integer i__, j, k, l;
    // 声明双精度浮点数组 x，含有 32 个元素，用于存储浮点数
    doublereal x[32];
    // 声明变量 sidelength，双精度浮点数，用于存储边长
    doublereal sidelength;
    // 声明整型变量 help，用于辅助目的
    integer help;

    // 调用 freeold 函数，不使用其返回值
    (void) freeold;
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --point;  // 减少指针 point 的值，移动到前一个位置
    f -= 3;  // 减少数组 f 的起始地址，使其向前偏移 3 个位置
    ++anchor;  // 增加 anchor 的值，移动到下一个位置
    length_dim1 = *maxdim;  // 将 length_dim1 设置为 maxdim 所指向的值
    length_offset = 1 + length_dim1;  // 计算 length_offset 的偏移量
    length -= length_offset;  // 减少数组 length 的起始地址，使其向前偏移 length_offset 个位置
    c_dim1 = *maxdim;  // 将 c_dim1 设置为 maxdim 所指向的值
    c_offset = 1 + c_dim1;  // 计算 c_offset 的偏移量
    c__ -= c_offset;  // 减少数组 c__ 的起始地址，使其向前偏移 c_offset 个位置
    --c2;  // 减少指针 c2 的值，移动到前一个位置
    --c1;  // 减少指针 c1 的值，移动到前一个位置

    /* Function Body */
    i__1 = *free - 1;  // 计算循环的上界
    for (i__ = 1; i__ <= i__1; ++i__) {  // 循环迭代 i__ 从 1 到 *free-1
        if (f[(i__ << 1) + 2] > 0.) {  // 如果数组 f 的特定元素大于 0
/* +-----------------------------------------------------------------------+ */
/* | Get the maximum side length of the hyper rectangle and then set the   | */
/* | new side length to this lengths times the growth factor.              | */
/* +-----------------------------------------------------------------------+ */
            help = direct_dirgetmaxdeep_(&i__, &length[length_offset], maxfunc, n);  // 调用函数 direct_dirgetmaxdeep_，获取最大超矩形的一边长度
            sidelength = thirds[help] * 2.;  // 计算新的边长，为 help 对应的 thirds 值乘以 2

/* +-----------------------------------------------------------------------+ */
/* | Set the Center and the upper and lower bounds of the rectangles.      | */
/* +-----------------------------------------------------------------------+ */
            i__2 = *n;  // 获取 n 的值
            for (j = 1; j <= i__2; ++j) {  // 循环迭代 j 从 1 到 n
                sidelength = thirds[length[i__ + j * length_dim1]];  // 重新计算 sidelength 为 length 数组的特定元素
                a[j - 1] = c__[j + i__ * c_dim1] - sidelength;  // 设置 a 数组的特定元素
                b[j - 1] = c__[j + i__ * c_dim1] + sidelength;  // 设置 b 数组的特定元素
/* L20: */
            }

/* +-----------------------------------------------------------------------+ */
/* | The function value is reset to 'Inf', since it may have been changed  | */
/* | in an earlier iteration and now the feasible point which was close    | */
/* | is not close anymore (since the hyper rectangle surrounding the       | */
/* | current point may have shrunk).                                       | */
/* +-----------------------------------------------------------------------+ */
            f[(i__ << 1) + 1] = HUGE_VAL;  // 将数组 f 的特定元素设置为 HUGE_VAL
            f[(i__ << 1) + 2] = 2.;  // 将数组 f 的特定元素设置为 2

/* +-----------------------------------------------------------------------+ */
/* | Check if any feasible point is near this infeasible point.            | */
/* +-----------------------------------------------------------------------+ */
            i__2 = *free - 1;  // 计算循环的上界
            for (k = 1; k <= i__2; ++k) {  // 循环迭代 k 从 1 到 *free-1
/* +-----------------------------------------------------------------------+ */
/* | If the point k is feasible, check if it is near.                      | */
/* +-----------------------------------------------------------------------+ */
                if (f[(k << 1) + 2] == 0.) {  // 如果数组 f 的特定元素等于 0
/* +-----------------------------------------------------------------------+ */
/* | Copy the coordinates of the point k into x.                           | */
/* +-----------------------------------------------------------------------+ */
/* | The loop iterates over each element in the array x, copying values    | */
/* | from column k of matrix c into x.                                     | */
/* +-----------------------------------------------------------------------+ */
            i__3 = *n;
            for (l = 1; l <= i__3; ++l) {
            x[l - 1] = c__[l + k * c_dim1];
/* L40: */
            }
/* +-----------------------------------------------------------------------+ */
/* | Check if the point k is within the box defined by vectors a and b,    | */
/* | using the function isinbox_. If true, update values in array f.       | */
/* +-----------------------------------------------------------------------+ */
            if (isinbox_(x, a, b, n, &c__32) == 1) {
/* Computing MIN */
             d__1 = f[(i__ << 1) + 1], d__2 = f[(k << 1) + 1];
             f[(i__ << 1) + 1] = MIN(d__1,d__2);
             f[(i__ << 1) + 2] = 1.;
            }
        }
/* L30: */
        }
        if (f[(i__ << 1) + 2] == 1.) {
/* | If the second component of f[i] is 1, adjust the first component by   | */
/* | adding a small increment, ensuring it remains positive. Also update   | */
/* | values in array x based on matrix c and vectors c1 and c2.            | */
        f[(i__ << 1) + 1] += (d__1 = f[(i__ << 1) + 1], fabs(d__1)) *
            1e-6f;
        i__2 = *n;
        for (l = 1; l <= i__2; ++l) {
            x[l - 1] = c__[l + i__ * c_dim1] * c1[l] + c__[l + i__ *
                c_dim1] * c2[l];
/* L200: */
        }
/* | Call subroutine dirresortlist_ to perform a reordering operation based | */
/* | on updated indices and values.                                        | */
        dirresortlist_(&i__, &anchor[-1], &f[3], &point[1],
                   &length[length_offset], n,
                   maxfunc, maxdim, maxdeep, logfile, jones);
        } else {
/* +-----------------------------------------------------------------------+ */
/* | Comment added by JG on 01/22/01:                                      | */
/* | Replaced fixed value for infeasible points with maximum value found,  | */
/* | increased by 1.                                                       | */
/* +-----------------------------------------------------------------------+ */
        if (! (*fmax == f[(i__ << 1) + 1])) {
/* | If fmax is not equal to the current value of f[i], update f[i] to the  | */
/* | maximum of fmax+1 or its current value.                               | */
            d__1 = *fmax + 1., d__2 = f[(i__ << 1) + 1];
            f[(i__ << 1) + 1] = MAX(d__1,d__2);
        }
        }
    }
/* L10: */
    }
/* L1000: */
} /* dirreplaceinf_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInsert                                               | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ static void dirinsert_(integer *start, integer *ins, integer *point,
    doublereal *f, integer *maxfunc)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, help;

/* | Comment: Rewrote this routine on 09/17/00.                             | */
/* | The loop evaluates conditions and swaps elements in array 'point' and  | */
/* | updates based on comparisons with array 'f'.                          | */
    i__1 = *maxfunc;
    for (i__ = 1; i__ <= i__1; ++i__) {
    # 如果指针数组中以 start 指向的位置对应的值为 0
    if (point[*start] == 0) {
        # 将 ins 指向的值赋给 point 数组中 start 指向的位置
        point[*start] = *ins;
        # 将 ins 指向的位置置为 0
        point[*ins] = 0;
        # 函数执行完毕，返回
        return;
    } else if (f[(*ins << 1) + 1] < f[(point[*start] << 1) + 1]) {
         # 将 point 数组中 start 指向的位置的值暂存到 help 中
         help = point[*start];
         # 将 ins 指向的值赋给 point 数组中 start 指向的位置
         point[*start] = *ins;
         # 将 help 中暂存的值赋给 point 数组中 ins 指向的位置
         point[*ins] = help;
         # 函数执行完毕，返回
         return;
    }
    # 将 start 指向 point 数组中 start 指向的位置
    *start = point[*start];
/* L10: */
    }
} /* dirinsert_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInsertList                                           | */
/* |    Changed 02-24-2000                                                 | */
/* |      Got rid of the distinction between feasible and infeasible points| */
/* |      I could do this since infeasible points get set to a high        | */
/* |      function value, which may be replaced by a function value of a   | */
/* |      nearby function at the end of the main loop.                     | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirinsertlist_(integer *new__, integer *anchor, integer *
    point, doublereal *f, integer *maxi, integer *length, integer *
    maxfunc, const integer *maxdeep, integer *n, integer *samp,
                        integer jones)
{
    /* System generated locals */
    integer length_dim1, length_offset, i__1;

    /* Local variables */
    integer j;
    integer pos;
    integer pos1, pos2, deep;

    (void) maxdeep;

    /* Adjust indices and dimensions for arrays */
    f -= 3;
    --point;
    ++anchor;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Loop over a range of indices */
    i__1 = *maxi;
    for (j = 1; j <= i__1; ++j) {
        /* Assign values to local variables */
        pos1 = *new__;
        pos2 = point[pos1];
        *new__ = point[pos2];

        /* Call subroutine to get the level */
        deep = direct_dirgetlevel_(&pos1, &length[length_offset], maxfunc, n, jones);

        /* Conditionally update anchor based on function values */
        if (anchor[deep] == 0) {
            if (f[(pos2 << 1) + 1] < f[(pos1 << 1) + 1]) {
                anchor[deep] = pos2;
                point[pos2] = pos1;
                point[pos1] = 0;
            } else {
                anchor[deep] = pos1;
                point[pos2] = 0;
            }
        } else {
            pos = anchor[deep];
            if (f[(pos2 << 1) + 1] < f[(pos1 << 1) + 1]) {
                if (f[(pos2 << 1) + 1] < f[(pos << 1) + 1]) {
                    anchor[deep] = pos2;
                    /* Swap positions if necessary */
                    if (f[(pos1 << 1) + 1] < f[(pos << 1) + 1]) {
                        point[pos2] = pos1;
                        point[pos1] = pos;
                    } else {
                        point[pos2] = pos;
                        /* Insert pos1 into the list */
                        dirinsert_(&pos, &pos1, &point[1], &f[3], maxfunc);
                    }
                } else {
                    /* Insert pos2 and pos1 into the list */
                    dirinsert_(&pos, &pos2, &point[1], &f[3], maxfunc);
                    dirinsert_(&pos, &pos1, &point[1], &f[3], maxfunc);
                }
            } else {
                if (f[(pos1 << 1) + 1] < f[(pos << 1) + 1]) {
                    /* Insert pos1 into the list */


注释：
/*      f(pos1,1) < f(pos2,1) < f(pos,1) */
/* 检查条件：f(pos1,1) < f(pos2,1) < f(pos,1) */
            anchor[deep] = pos1;
/* 将 anchor 数组中索引为 deep 的位置设置为 pos1 */
            if (f[(pos << 1) + 1] < f[(pos2 << 1) + 1]) {
/* 如果 f[(pos << 1) + 1] 小于 f[(pos2 << 1) + 1]，执行以下操作 */
            point[pos1] = pos;
/* 将 point 数组中索引为 pos1 的位置设置为 pos */
            dirinsert_(&pos, &pos2, &point[1], &f[3], maxfunc);
/* 调用 dirinsert_ 函数，传递参数 pos, pos2, point 数组的起始地址，f 数组的起始地址，maxfunc */
            } else {
/* 如果条件不满足，执行以下操作 */
            point[pos1] = pos2;
/* 将 point 数组中索引为 pos1 的位置设置为 pos2 */
            point[pos2] = pos;
/* 将 point 数组中索引为 pos2 的位置设置为 pos */
            }
        } else {
/* 如果条件不满足，执行以下操作 */
            dirinsert_(&pos, &pos1, &point[1], &f[3], maxfunc);
/* 调用 dirinsert_ 函数，传递参数 pos, pos1, point 数组的起始地址，f 数组的起始地址，maxfunc */
            dirinsert_(&pos, &pos2, &point[1], &f[3], maxfunc);
/* 再次调用 dirinsert_ 函数，传递参数 pos, pos2, point 数组的起始地址，f 数组的起始地址，maxfunc */
        }
        }
    }
/* 结束循环 */
/* L10: */
    }
/* 结束循环 */
/* JG 09/24/00 Changed this to Getlevel */
/*      deep = DIRGetMaxdeep(samp,length,maxfunc,n) */
    deep = direct_dirgetlevel_(samp, &length[length_offset], maxfunc, n, jones);
/* 调用 direct_dirgetlevel_ 函数，传递参数 samp, length 数组的指定位置地址，maxfunc, n, jones */
    pos = anchor[deep];
/* 将 pos 设置为 anchor 数组中索引为 deep 的值 */
    if (f[(*samp << 1) + 1] < f[(pos << 1) + 1]) {
/* 如果 f[(*samp << 1) + 1] 小于 f[(pos << 1) + 1]，执行以下操作 */
    anchor[deep] = *samp;
/* 将 anchor 数组中索引为 deep 的位置设置为 samp */
    point[*samp] = pos;
/* 将 point 数组中索引为 samp 的位置设置为 pos */
    } else {
/* 如果条件不满足，执行以下操作 */
    dirinsert_(&pos, samp, &point[1], &f[3], maxfunc);
/* 调用 dirinsert_ 函数，传递参数 pos, samp, point 数组的起始地址，f 数组的起始地址，maxfunc */
    }
} /* dirinsertlist_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInsertList2  (Old way to do it.)                     | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ static void dirinsertlist_2__(integer *start, integer *j, integer *k,
     integer *list2, doublereal *w, integer *maxi, integer *n)
{
/* 子例程 DIRInsertList2 的实现 */
    /* System generated locals */
/* 系统生成的局部变量声明 */
    integer list2_dim1, list2_offset, i__1;
/* 声明整型变量 list2_dim1, list2_offset, i__1 */

    /* Local variables */
/* 局部变量声明 */
    integer i__, pos;
/* 声明整型变量 i__, pos */

    /* Parameter adjustments */
/* 参数调整 */
    --w;
    list2_dim1 = *n;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;

    /* Function Body */
/* 函数体 */

    pos = *start;
/* 将 pos 设置为 start 的值 */
    if (*start == 0) {
/* 如果 start 等于 0，执行以下操作 */
    list2[*j + list2_dim1] = 0;
/* 将 list2 数组中索引为 *j+list2_dim1 的位置设置为 0 */
    *start = *j;
/* 将 start 设置为 j 的值 */
    goto L50;
/* 跳转到 L50 标签处 */
    }
    if (w[*start] > w[*j]) {
/* 如果 w[*start] 大于 w[*j]，执行以下操作 */
    list2[*j + list2_dim1] = *start;
/* 将 list2 数组中索引为 *j+list2_dim1 的位置设置为 start */
    *start = *j;
/* 将 start 设置为 j 的值 */
    } else {
/* 如果条件不满足，执行以下操作 */
    i__1 = *maxi;
/* 将 i__1 设置为 maxi 的值 */
    for (i__ = 1; i__ <= i__1; ++i__) {
/* 循环 i 从 1 到 i__1 */
        if (list2[pos + list2_dim1] == 0) {
/* 如果 list2[pos+list2_dim1] 等于 0，执行以下操作 */
        list2[*j + list2_dim1] = 0;
/* 将 list2 数组中索引为 *j+list2_dim1 的位置设置为 0 */
        list2[pos + list2_dim1] = *j;
/* 将 list2 数组中索引为 pos+list2_dim1 的位置设置为 j */
        goto L50;
/* 跳转到 L50 标签处 */
        } else {
/* 如果条件不满足，执行以下操作 */
        if (w[*j] < w[list2[pos + list2_dim1]]) {
/* 如果 w[*j] 小于 w[list2[pos+list2_dim1]]，执行以下操作 */
            list2[*j + list2_dim1] = list2[pos + list2_dim1];
/* 将 list2 数组中索引为 *j+list2_dim1 的位置设置为 list2[pos+list2_dim1] */
            list2[pos + list2_dim1] = *j;
/* 将 list2 数组中索引为 pos+list2_dim1 的位置设置为 j */
            goto L50;
/* 跳转到 L50 标签处 */
        }
        }
        pos = list2[pos + list2_dim1];
/* 将 pos 设置为 list2[pos+list2_dim1] 的值 */
/* L10: */
    }
    }
L50:
/* 标签 L50 处 */
    list2[*j + (list2_dim1 << 1)] = *k;
/* 将 list2 数组中索引为 *j+(list2_dim1<<1) 的位置设置为 k */
} /* dirinsertlist_2__ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRSearchmin                                            | */
/* |    Search for the minimum in the list.                                ! */
/* +-----------------------------------------------------------------------+ */
/* 子例程 DIRSearchmin 的实现，用于在列表中搜索最小值 */
/* Subroutine */ static void dirsearchmin_(integer *start, integer *list2, integer *
    pos, integer *k, integer *n)
{
    /* System generated locals */
/* 系统生成的局部变量声明 */
    integer list2_dim1, list2_offset;

    /* Parameter adjustments */
/* 参数调整 */
    list2_dim1 = *n;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;

    /* Function Body */
/* 函数体 */
    *k = *start;
    # 将指针 k 指向的内容设置为指针 start 指向的内容

    *pos = list2[*start + (list2_dim1 << 1)];
    # 将指针 pos 指向的内容设置为 list2 数组中索引为 *start + (list2_dim1 << 1) 的元素

    *start = list2[*start + list2_dim1];
    # 将指针 start 指向的内容设置为 list2 数组中索引为 *start + list2_dim1 的元素
/* } ends a preceding block of code, assumed to be a comment end marker */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRSamplepoints                                         | */
/* |    Subroutine to sample the new points.                               | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirsamplepoints_(doublereal *c__, integer *arrayi,
    doublereal *delta, integer *sample, integer *start, integer *length,
    FILE *logfile, doublereal *f, integer *free,
    integer *maxi, integer *point, doublereal *x, doublereal *l,
     doublereal *minf, integer *minpos, doublereal *u, integer *n,
    integer *maxfunc, const integer *maxdeep, integer *oops)
{
    /* System generated locals */
    integer length_dim1, length_offset, c_dim1, c_offset, i__1, i__2;

    /* Local variables */
    integer j, k, pos;

    (void) minf; (void) minpos; (void) maxfunc; (void) maxdeep; (void) oops;

    /* Parameter adjustments */
    --u;
    --l;
    --x;
    --arrayi;
    --point;
    f -= 3;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    c_dim1 = *n;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    *oops = 0;
    pos = *free;
    *start = *free;
    i__1 = *maxi + *maxi;
    for (k = 1; k <= i__1; ++k) {
    i__2 = *n;
    for (j = 1; j <= i__2; ++j) {
        length[j + *free * length_dim1] = length[j + *sample *
            length_dim1];
        c__[j + *free * c_dim1] = c__[j + *sample * c_dim1];
/* L20: */
    }
    pos = *free;
    *free = point[*free];
    if (*free == 0) {
         if (logfile)
          fprintf(logfile, "Error, no more free positions! "
              "Increase maxfunc!\n");
        *oops = 1;
        return;
    }
/* L10: */
    }
    point[pos] = 0;
    pos = *start;
    i__1 = *maxi;
    for (j = 1; j <= i__1; ++j) {
    c__[arrayi[j] + pos * c_dim1] = c__[arrayi[j] + *sample * c_dim1] + *
        delta;
    pos = point[pos];
    c__[arrayi[j] + pos * c_dim1] = c__[arrayi[j] + *sample * c_dim1] - *
        delta;
    pos = point[pos];
/* L30: */
    }
    ASRT(pos <= 0); /* Assertion to check if pos is less than or equal to 0 */
} /* dirsamplepoints_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRDivide                                               | */
/* |    Subroutine to divide the hyper rectangles according to the rules.  | */
/* |    Changed 02-24-2000                                                 | */
/* |      Replaced if statement by min (line 367)                          | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirdivide_(integer *new__, integer *currentlength,
    integer *length, integer *point, integer *arrayi, integer *sample,
    integer *list2, doublereal *w, integer *maxi, doublereal *f, integer *
    maxfunc, const integer *maxdeep, integer *n)
{
    /* System generated locals */
    // 定义整型变量：第一维长度、偏移量、第二个列表的第一维长度、偏移量、计数变量 i__1、i__2
    integer length_dim1, length_offset, list2_dim1, list2_offset, i__1, i__2;
    // 定义双精度实数变量 d__1、d__2

    // 声明局部变量：i__、j、k、pos、pos2、start
    integer i__, j, k, pos, pos2;
    integer start;

    // 忽略未使用的变量 maxfunc 和 maxdeep
    (void) maxfunc; (void) maxdeep;

    // 调整参数的偏移量
    // 减少数组 f 的基址，将 point 和 w 指针移动到正确的位置
    f -= 3;
    --point;
    --w;

    // 设置二维数组 list2 的维度和偏移量
    list2_dim1 = *n;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;

    // 减少数组 arrayi 的基址，设置 length 的第一维度和偏移量
    --arrayi;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    // 函数主体
    // 初始化 start 为 0
    start = 0;
    // 将 pos 初始化为参数 new__ 的值
    pos = *new__;

    // 循环遍历 i__ 从 1 到 maxi
    i__1 = *maxi;
    for (i__ = 1; i__ <= i__1; ++i__) {
        // 从 arrayi 数组中获取 j
        j = arrayi[i__];
        // 将 w[j] 设置为 f[(pos << 1) + 1] 的值
        w[j] = f[(pos << 1) + 1];
        // 将 k 设置为 pos
        k = pos;
        // 将 pos 设置为 point[pos] 的值
        pos = point[pos];
/* Computing MIN */
    // 从数组 f 和 w 中获取两个值，取较小的那个，并将结果存入 w[j]
    d__1 = f[(pos << 1) + 1], d__2 = w[j];
    w[j] = MIN(d__1,d__2);
    // 更新 pos 到下一个位置
    pos = point[pos];
    // 调用 dirinsertlist_2__ 函数，向列表 list2 中插入数据
    dirinsertlist_2__(&start, &j, &k, &list2[list2_offset], &w[1], maxi,
        n);
/* L10: */
    }
    // 断言，确保 pos 的值小于等于 0
    ASRT(pos <= 0);
    // 循环，处理每个 j 值
    i__1 = *maxi;
    for (j = 1; j <= i__1; ++j) {
    // 调用 dirsearchmin_ 函数，搜索最小值
    dirsearchmin_(&start, &list2[list2_offset], &pos, &k, n);
    // 初始化 pos2
    pos2 = start;
    // 将当前长度加 1，并存入 length 数组
    length[k + *sample * length_dim1] = *currentlength + 1;
    // 循环，处理每个 i 值
    i__2 = *maxi - j + 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        // 将当前长度加 1，并存入 length 数组
        length[k + pos * length_dim1] = *currentlength + 1;
        // 更新 pos 到下一个位置
        pos = point[pos];
        // 将当前长度加 1，并存入 length 数组
        length[k + pos * length_dim1] = *currentlength + 1;
/* JG 07/10/01 pos2 = 0 at the end of the 30-loop. Since we end */
/*             the loop now, we do not need to reassign pos and pos2. */
        // 如果 pos2 大于 0，则更新 pos 和 pos2
        if (pos2 > 0) {
        pos = list2[pos2 + (list2_dim1 << 1)];
        pos2 = list2[pos2 + list2_dim1];
        }
/* L30: */
    }
/* L20: */
    }
} /* dirdivide_ */

/* +-----------------------------------------------------------------------+ */
/* |                                                                       | */
/* |                       SUBROUTINE DIRINFCN                             | */
/* |                                                                       | */
/* | Subroutine DIRinfcn unscales the variable x for use in the            | */
/* | user-supplied function evaluation subroutine fcn. After fcn returns   | */
/* | to DIRinfcn, DIRinfcn then rescales x for use by DIRECT.              | */
/* |                                                                       | */
/* | On entry                                                              | */
/* |                                                                       | */
/* |        fcn -- The argument containing the name of the user-supplied   | */
/* |               subroutine that returns values for the function to be   | */
/* |               minimized.                                              | */
/* |                                                                       | */
/* |          x -- A double-precision vector of length n. The point at     | */
/* |               which the derivative is to be evaluated.                | */
/* |                                                                       | */
/* |        xs1 -- A double-precision vector of length n. Used for         | */
/* |               scaling and unscaling the vector x by DIRinfcn.         | */
/* |                                                                       | */
/* |        xs2 -- A double-precision vector of length n. Used for         | */
/* |               scaling and unscaling the vector x by DIRinfcn.         | */
/* |                                                                       | */
/* |          n -- An integer. The dimension of the problem.               | */
/* |       kret -- An Integer. If kret =  1, the point is infeasible,      | */
/* |                              kret = -1, bad problem set up,           | */
/* |                              kret =  0, feasible.                     | */
/* |                                                                       | */
/* | On return                                                             | */
/* |                                                                       | */
/* |          f -- A double-precision scalar.                              | */
/* |                                                                       | */
/* | Subroutines and Functions                                             | */
/* |                                                                       | */
/* | The subroutine whose name is passed through the argument fcn.         | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ PyObject* direct_dirinfcn_(PyObject* fcn, doublereal *x, PyObject *x_seq,
    doublereal *c1, doublereal *c2, integer *n, doublereal *f, integer *flag__,
    PyObject* args)
{
    int i;

    /* +-----------------------------------------------------------------------+ */
    /* | Variables to pass user defined data to the function to be optimized.  | */
    /* +-----------------------------------------------------------------------+ */
    /* +-----------------------------------------------------------------------+ */
    /* | Unscale the variable x.                                               | */
    /* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --c2;
    --c1;
    --x;

    /* Function Body */
    /* +-----------------------------------------------------------------------+ */
    /* | Call the function-evaluation subroutine fcn.                          | */
    /* +-----------------------------------------------------------------------+ */
    *flag__ = 0;
    // TODO: PyArray_SimpleNewFromData gives segmentation fault
    // and therefore using list to pass to the user function.
    // Once the above function works, replace with NumPy arrays.
    for (i = 0; i < *n; i++) {
        doublereal x_i_scaled = (x[i + 1] + c2[i + 1]) * c1[i + 1];
        PyList_SetItem(x_seq, i, PyFloat_FromDouble(x_i_scaled));
    }
    PyObject* arg_tuple = NULL;
    if (PyObject_IsTrue(args)) {
        arg_tuple = Py_BuildValue("(OO)", x_seq, args);
    } else {
        arg_tuple = Py_BuildValue("(O)", x_seq);
    }
    PyObject* f_py = PyObject_CallObject(fcn, arg_tuple);
    Py_DECREF(arg_tuple);
    if (!f_py ) {
        return NULL;
    }
    *f = PyFloat_AsDouble(f_py);
    return f_py;
} /* dirinfcn_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRGet_I                                                | */
/* +-----------------------------------------------------------------------+ */


注释：
/* Subroutine */ void direct_dirget_i__(integer *length, integer *pos, integer *
    arrayi, integer *maxi, integer *n, integer *maxfunc)
{
    /* System generated locals */
    integer length_dim1, length_offset, i__1;

    /* Local variables */
    integer i__, j, help;

    (void) maxfunc;  // 忽略未使用的 maxfunc 参数

    /* Parameter adjustments */
    --arrayi;  // 调整数组 arrayi 以从1开始索引
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;  // 调整 length 数组以从1开始索引

    /* Function Body */
    j = 1;  // 初始化索引 j
    help = length[*pos * length_dim1 + 1];  // 获取初始帮助值
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
        if (length[i__ + *pos * length_dim1] < help) {
            help = length[i__ + *pos * length_dim1];  // 更新最小帮助值
        }
        /* L10: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        if (length[i__ + *pos * length_dim1] == help) {
            arrayi[j] = i__;  // 将满足条件的索引 i 存入 arrayi 数组
            ++j;  // 更新 j 索引
        }
        /* L20: */
    }
    *maxi = j - 1;  // 更新 maxi，指示符合条件的索引数目
} /* dirget_i__ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInit                                                 | */
/* |    Initialise all needed variables and do the first run of the        | */
/* |    algorithm.                                                         | */
/* |    Changed 02/24/2000                                                 | */
/* |       Changed fcn Double precision to fcn external!                   | */
/* |    Changed 09/15/2000                                                 | */
/* |       Added distinction between Jones way to characterize rectangles  | */
/* |       and our way. Common variable JONES controls which way we use.   | */
/* |          JONES = 0    Jones way (Distance from midpoint to corner)    | */
/* |          JONES = 1    Our way (Length of longest side)                | */
/* |    Changed 09/24/00                                                   | */
/* |       Added array levels. Levels contain the values to characterize   | */
/* |       the hyperrectangles.                                            | */
/* |    Changed 01/22/01                                                   | */
/* |       Added variable fmax to keep track of maximum value found.       | */
/* |       Added variable Ifeasiblef to keep track if feasibel point has   | */
/* |       been found.                                                     | */
/* |    Changed 01/23/01                                                   | */
/* |       Added variable Ierror to keep track of errors.                  | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ PyObject* direct_dirinit_(doublereal *f, PyObject* fcn, doublereal *c__,
    integer *length, integer *actdeep, integer *point, integer *anchor,
    integer *free, FILE *logfile, integer *arrayi,
    integer *maxi, integer *list2, doublereal *w, doublereal *x, PyObject *x_seq,
    doublereal *l, doublereal *u, doublereal *minf, integer *minpos,
    # 声明一个指向 doublereal 类型数组的指针 thirds
    doublereal *thirds,
    # 声明一个指向 doublereal 类型数组的指针 levels
    doublereal *levels,
    # 声明一个指向 integer 类型的变量 maxfunc
    integer *maxfunc,
    # 声明一个指向 const integer 类型的指针 maxdeep
    const integer *maxdeep,
    # 声明一个指向 integer 类型的变量 n
    integer *n,
    # 声明一个指向 integer 类型的变量 maxor
    integer *maxor,
    # 声明一个指向 doublereal 类型的变量 fmax
    doublereal *fmax,
    # 声明一个指向 integer 类型的变量 ifeasiblef
    integer *ifeasiblef,
    # 声明一个指向 integer 类型的变量 iinfeasible
    integer *iinfeasible,
    # 声明一个指向 integer 类型的变量 ierror
    integer *ierror,
    # 声明一个 PyObject 类型的指针 args
    PyObject* args,
    # 声明一个 integer 类型的变量 jones
    integer jones,
    # 声明一个指向 int 类型的变量 force_stop
    int *force_stop
{
    /* System generated locals */
    integer c_dim1, c_offset, length_dim1, length_offset, list2_dim1,
        list2_offset, i__1, i__2;

    /* Local variables */
    integer i__, j;
    integer new__, help, oops;
    doublereal help2, delta;

/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable Ifeasiblef to keep track if feasibel point | */
/* |             has been found.                                           | */
/* | JG 01/23/01 Added variable Ierror to keep track of errors.            | */
/* | JG 03/09/01 Added IInfeasible to keep track if an infeasible point has| */
/* |             been found.                                               | */
/* +-----------------------------------------------------------------------+ */
/* JG 09/15/00 Added variable JONES (see above) */
/* +-----------------------------------------------------------------------+ */
/* | Variables to pass user defined data to the function to be optimized.  | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --point;
    f -= 3;
    ++anchor;
    --u;
    --l;
    --x;
    --w;
    list2_dim1 = *maxor;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;
    --arrayi;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    c_dim1 = *maxor;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    *minf = HUGE_VAL;
/* JG 09/15/00 If Jones way of characterising rectangles is used, */
/*             initialise thirds to reflect this. */
    if (jones == 0) {
    i__1 = *n - 1;
    for (j = 0; j <= i__1; ++j) {
        w[j + 1] = sqrt(*n - j + j / 9.) * .5;
/* L5: */
    }
    help2 = 1.;
    i__1 = *maxdeep / *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = *n - 1;
        for (j = 0; j <= i__2; ++j) {
        levels[(i__ - 1) * *n + j] = w[j + 1] / help2;
/* L8: */
        }
        help2 *= 3.;
/* L10: */
    }
    } else {
/* JG 09/15/00 Initialiase levels to contain 1/j */
    help2 = 3.;
    i__1 = *maxdeep;
    for (i__ = 1; i__ <= i__1; ++i__) {
        levels[i__] = 1. / help2;
        help2 *= 3.;
/* L11: */
    }
    levels[0] = 1.;
    }
    help2 = 3.;
    i__1 = *maxdeep;
    for (i__ = 1; i__ <= i__1; ++i__) {
    thirds[i__] = 1. / help2;
    help2 *= 3.;
/* L21: */
    }
    thirds[0] = 1.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    c__[i__ + c_dim1] = .5;
    x[i__] = .5;
    length[i__ + length_dim1] = 0;
/* L20: */
    }
    PyObject* ret = direct_dirinfcn_(fcn, &x[1], x_seq, &l[1], &u[1], n, &f[3], &help, args);
    if (!ret) {
        return NULL;
    }
}
    # 检查 force_stop 指针是否为真值，并且其指向的值为真（非零）
    if (force_stop && *force_stop) {
        # 如果满足条件，设置错误码为 -102
        *ierror = -102;
        # 返回 ret 变量的值
        return ret;
    }
    # 将数组 f 的第五个元素（索引为 4）设置为 help 的双精度浮点数值
    f[4] = (doublereal) help;
    # 将 iinfeasible 指针指向的位置设置为 help 的值
    *iinfeasible = help;
    # 将 fmax 指针指向的位置设置为 f 数组的第四个元素（索引为 3）的值
    *fmax = f[3];
/* 09/25/00 Added this */
/*      if (f(1,1) .ge. 1.E+6) then */
    /* 检查 f[4] 是否大于 0 */
    if (f[4] > 0.) {
    /* 设置 f[3] 为无穷大 */
    f[3] = HUGE_VAL;
    /* 更新 fmax 的值为 f[3] */
    *fmax = f[3];
    /* 设置 ifeasiblef 为 1，表示找到可行点 */
    *ifeasiblef = 1;
    } else {
    /* 设置 ifeasiblef 为 0，表示未找到可行点 */
    *ifeasiblef = 0;
    }
/* JG 09/25/00 Remove IF */
    /* 设置 minf 为 f[3] */
    *minf = f[3];
    /* 设置 minpos 为 1 */
    *minpos = 1;
    /* 设置 actdeep 为 2 */
    *actdeep = 2;
    /* 设置 point[1] 为 0 */
    point[1] = 0;
    /* 设置 free 为 2 */
    *free = 2;
    /* 设置 delta 为 thirds[1] */
    delta = thirds[1];
    /* 调用 direct_dirget_i__ 函数 */
    direct_dirget_i__(&length[length_offset], &c__1, &arrayi[1], maxi, n, maxfunc);
    /* 设置 new__ 为 *free 的值 */
    new__ = *free;
    /* 调用 direct_dirsamplepoints_ 函数 */
    direct_dirsamplepoints_(&c__[c_offset], &arrayi[1], &delta, &c__1, &new__, &
        length[length_offset], logfile, &f[3], free, maxi, &
        point[1], &x[1], &l[1], minf, minpos, &u[1], n,
        maxfunc, maxdeep, &oops);
/* +-----------------------------------------------------------------------+ */
/* | JG 01/23/01 Added error checking.                                     | */
/* +-----------------------------------------------------------------------+ */
    /* 检查 oops 是否大于 0 */
    if (oops > 0) {
    /* 设置 ierror 为 -4 */
    *ierror = -4;
    /* 返回 ret */
    return ret;
    }
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* |             Added variable to keep track if feasible point was found. | */
/* +-----------------------------------------------------------------------+ */
    /* 调用 direct_dirsamplef_ 函数 */
    direct_dirsamplef_(&c__[c_offset], &arrayi[1], &delta, &c__1, &new__, &length[
        length_offset], logfile, &f[3], free, maxi, &point[
        1], fcn, &x[1], x_seq, &l[1], minf, minpos, &u[1], n, maxfunc,
        maxdeep, &oops, fmax, ifeasiblef, iinfeasible, args,
        force_stop);
    /* 如果 force_stop 为真且 *force_stop 为真 */
    if (force_stop && *force_stop) {
     /* 设置 ierror 为 -102 */
     *ierror = -102;
     /* 返回 ret */
     return ret;
    }
/* +-----------------------------------------------------------------------+ */
/* | JG 01/23/01 Added error checking.                                     | */
/* +-----------------------------------------------------------------------+ */
    /* 检查 oops 是否大于 0 */
    if (oops > 0) {
    /* 设置 ierror 为 -5 */
    *ierror = -5;
    /* 返回 ret */
    return ret;
    }
    /* 调用 direct_dirdivide_ 函数 */
    direct_dirdivide_(&new__, &c__0, &length[length_offset], &point[1], &arrayi[1], &
        c__1, &list2[list2_offset], &w[1], maxi, &f[3], maxfunc,
        maxdeep, n);
    /* 调用 direct_dirinsertlist_ 函数 */
    direct_dirinsertlist_(&new__, &anchor[-1], &point[1], &f[3], maxi, &
        length[length_offset], maxfunc, maxdeep, n, &c__1, jones);
    /* 返回 ret */
    return ret;
} /* dirinit_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInitList                                             | */
/* |    Initialise the list.                                               | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirinitlist_(integer *anchor, integer *free, integer *
    point, doublereal *f, integer *maxfunc, const integer *maxdeep)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;

/*   f -- values of functions. */
/*   anchor -- anchors of lists with deep i */
/*   point -- lists */
/*   free  -- first free position */
/* 参数调整 */
f -= 3;          /* 将数组 f 的起始位置向左偏移 3 个单位 */
--point;         /* 将指针 point 往左移动一个位置 */
++anchor;        /* 将指针 anchor 往右移动一个位置 */

/* 函数主体 */
i__1 = *maxdeep;
for (i__ = -1; i__ <= i__1; ++i__) {
    anchor[i__] = 0;  /* 初始化 anchor 数组中的每个元素为 0 */
/* L10: */
}
i__1 = *maxfunc;
for (i__ = 1; i__ <= i__1; ++i__) {
    f[(i__ << 1) + 1] = 0.;   /* 初始化 f 数组中偶数索引位置的元素为 0.0 */
    f[(i__ << 1) + 2] = 0.;   /* 初始化 f 数组中奇数索引位置的元素为 0.0 */
    point[i__] = i__ + 1;     /* 初始化 point 数组中每个元素为其索引加一 */
/*       point(i) = 0 */
/* L20: */
}
point[*maxfunc] = 0;  /* 将 point 数组中索引为 maxfunc 的元素设置为 0 */
*free = 1;            /* 设置指针 free 指向的值为 1，表示第一个空闲位置 */
} /* dirinitlist_ */

/* +-----------------------------------------------------------------------+ */
/* |                                                                       | */
/* |                       SUBROUTINE DIRPREPRC                            | */
/* |                                                                       | */
/* | Subroutine DIRpreprc uses an afine mapping to map the hyper-box given | */
/* | by the constraints on the variable x onto the n-dimensional unit cube.| */
/* | This mapping is done using the following equation:                    | */
/* |                                                                       | */
/* |               x(i)=x(i)/(u(i)-l(i))-l(i)/(u(i)-l(i)).                 | */
/* |                                                                       | */
/* | DIRpreprc checks if the bounds l and u are well-defined. That is, if  | */
/* |                                                                       | */
/* |               l(i) < u(i) forevery i.                                 | */
/* |                                                                       | */
/* | On entry                                                              | */
/* |                                                                       | */
/* |          u -- A double-precision vector of length n. The vector       | */
/* |               containing the upper bounds for the n independent       | */
/* |               variables.                                              | */
/* |                                                                       | */
/* |          l -- A double-precision vector of length n. The vector       | */
/* |               containing the lower bounds for the n independent       | */
/* |               variables.                                              | */
/* |                                                                       | */
/* |          n -- An integer. The dimension of the problem.               | */
/* |                                                                       | */
/* | On return                                                             | */
/* |                                                                       | */
/* |        xs1 -- A double-precision vector of length n, used for scaling | */
/* |               and unscaling the vector x.                             | */
/* |                                                                       | */
/* |        xs2 -- A double-precision vector of length n, used for scaling | */
/* |               and unscaling the vector x.                             | */
/* |                                                                       | */
/* |       oops -- An integer. If an upper bound is less than a lower      | */
/* |               bound or if the initial point is not in the             | */
/* |               hyper-box oops is set to 1 and iffco terminates.        | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirpreprc_(doublereal *u, doublereal *l, integer *n,
    doublereal *xs1, doublereal *xs2, integer *oops)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    doublereal help;

    /* Parameter adjustments */
    --xs2;
    --xs1;
    --l;
    --u;

    /* Function Body */
    *oops = 0; // 初始化 oops 为 0

    // 循环检查每个维度的上下界是否合理
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* +-----------------------------------------------------------------------+ */
/* | Check if the hyper-box is well-defined.                               | */
/* +-----------------------------------------------------------------------+ */
        if (u[i__] <= l[i__]) { // 如果上界小于等于下界，则设置 oops 为 1 并返回
            *oops = 1;
            return;
        }
/* L20: */
    }

/* +-----------------------------------------------------------------------+ */
/* | Scale the initial iterate so that it is in the unit cube.             | */
/* +-----------------------------------------------------------------------+ */
    // 对每个维度的初始迭代进行缩放，使其位于单位立方体内
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        help = u[i__] - l[i__];
        xs2[i__] = l[i__] / help; // 计算 xs2[i]，用于缩放和反缩放向量 x
        xs1[i__] = help; // 将 help 存储到 xs1[i] 中
/* L50: */
    }
} /* dirpreprc_ */

/* Subroutine */ void direct_dirheader_(FILE *logfile, integer *version,
    doublereal *x, PyObject *x_seq, integer *n, doublereal *eps, integer *maxf, integer *
    maxt, doublereal *l, doublereal *u, integer *algmethod, integer *
    maxfunc, const integer *maxdeep, doublereal *fglobal, doublereal *fglper,
    integer *ierror, doublereal *epsfix, integer *iepschange, doublereal *
    volper, doublereal *sigmaper)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer imainver, i__, numerrors, isubsubver, ihelp, isubver;

    (void) maxdeep; (void) ierror;

/* +-----------------------------------------------------------------------+ */
/* | Variables to pass user defined data to the function to be optimized.  | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --u;
    --l;
    --x;

    /* Function Body */
    if (logfile)
        fprintf(logfile, "------------------- Log file ------------------\n");

    numerrors = 0;
    *ierror = 0;
    imainver = *version / 100;
    # 计算版本号中的各个部分
    ihelp = *version - imainver * 100;
    # 计算次版本号
    isubver = ihelp / 10;
    # 更新ihelp以计算修订版本号
    ihelp -= isubver * 10;
    # 计算修订版本号
    isubsubver = ihelp;
/* +-----------------------------------------------------------------------+ */
/* | JG 01/13/01 Added check for epsilon. If epsilon is smaller 0, we use  | */
/* |             the update formula from Jones. We then set the flag       | */
/* |             iepschange to 1, and store the absolute value of eps in   | */
/* |             epsfix. epsilon is then changed after each iteration.     | */
/* +-----------------------------------------------------------------------+ */
    if (*eps < 0.) {
        *iepschange = 1;        // 设置标志 iepschange 为 1
        *epsfix = -(*eps);      // 存储 eps 的绝对值到 epsfix
        *eps = -(*eps);         // 更新 eps 的值为其相反数
    } else {
        *iepschange = 0;        // 设置标志 iepschange 为 0
        *epsfix = 1e100;        // 将 epsfix 设置为一个很大的值
    }

/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Removed printout of contents in cdata(1).                 | */
/* +-----------------------------------------------------------------------+ */
/*      write(logfile,*) cdata(1) */

    if (logfile) {
        // 打印 DIRECT 版本信息和相关参数到日志文件
        fprintf(logfile, "DIRECT Version %d.%d.%d\n"
             " Problem dimension n: %d\n"
             " Eps value: %e\n"
             " Maximum number of f-evaluations (maxf): %d\n"
             " Maximum number of iterations (MaxT): %d\n"
             " Value of f_global: %e\n"
             " Global minimum tolerance set: %e\n"
             " Volume tolerance set: %e\n"
             " Length tolerance set: %e\n",
             imainver, isubver, isubsubver, *n, *eps, *maxf, *maxt,
             *fglobal, *fglper, *volper, *sigmaper);
        // 根据 iepschange 的值选择打印不同的信息
        fprintf(logfile, *iepschange == 1
             ? "Epsilon is changed using the Jones formula.\n"
             : "Epsilon is constant.\n");
        // 根据 algmethod 的值选择打印不同的信息
        fprintf(logfile, *algmethod == 0
             ? "Using original DIRECT algorithm .\n"
             : "Using locally biased DIRECT_L algorithm.\n");
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        if (u[i__] <= l[i__]) {
            *ierror = -1;       // 设置错误码为 -1
            if (logfile)
                // 打印变量 x 的边界信息到日志文件
                fprintf(logfile, "WARNING: bounds on variable x%d: "
                     "%g <= xi <= %g\n", i__, l[i__], u[i__]);
            ++numerrors;        // 错误计数加一
        } else {
            if (logfile)
                // 打印变量 x 的边界信息到日志文件
                fprintf(logfile, "Bounds on variable x%d: "
                     "%g <= xi <= %g\n", i__, l[i__], u[i__]);
        }
    /* L1010: */
    }

/* +-----------------------------------------------------------------------+ */
/* | If there are to many function evaluations or to many iteration, note  | */
/* | this and set the error flag accordingly. Note: If more than one error | */
/* | occurred, we give out an extra message.                               | */
/* +-----------------------------------------------------------------------+ */
    if (*maxf + 20 > *maxfunc) {
        if (logfile)
            // 如果超过最大函数评估次数，打印警告信息到日志文件
            fprintf(logfile,
                "WARNING: The maximum number of function evaluations (%d) is higher than\n"
                "         the constant maxfunc (%d).  Increase maxfunc in subroutine DIRECT\n"
                "         or decrease the maximum number of function evaluations.\n",
                *maxf, *maxfunc);
        ++numerrors;            // 错误计数加一
        *ierror = -2;           // 设置错误码为 -2
    }
    if (*ierror < 0) {
    # 如果日志文件存在，则在日志中输出一条分隔线
    if (logfile) fprintf(logfile, "----------------------------------\n");
    # 如果错误数量为1，则输出警告：输入中有一个错误！到日志文件中
    if (numerrors == 1) {
         if (logfile)
          fprintf(logfile, "WARNING: One error in the input!\n");
    } else {
         # 如果错误数量不为1，则输出警告：输入中有%d个错误！到日志文件中，并包括错误数量
         if (logfile)
          fprintf(logfile, "WARNING: %d errors in the input!\n",
              numerrors);
    }
    # 如果日志文件存在，则在日志中输出一条分隔线
    }
    # 如果指针 *ierror 的值大于等于0，则输出迭代号、函数评估次数和最小函数值的表头到日志文件中
    if (*ierror >= 0) {
     if (logfile)
          fprintf(logfile, "Iteration # of f-eval. minf\n");
    }
/* L10005: */
/* 结束 dirheader_ 的定义 */

/* Subroutine */ void direct_dirsummary_(FILE *logfile, doublereal *x, doublereal *
    l, doublereal *u, integer *n, doublereal *minf, doublereal *fglobal,
    integer *numfunc, integer *ierror)
{
    /* Local variables */
    integer i__;

    /* Parameter adjustments */
    --u;        /* 调整参数 u 的下标，使其从1开始 */
    --l;        /* 调整参数 l 的下标，使其从1开始 */
    --x;        /* 调整参数 x 的下标，使其从1开始 */

    (void) ierror;  /* 忽略未使用的变量 ierror */

    /* Function Body */
    /* 如果 logfile 不为空，则输出以下信息 */
    if (logfile) {
     /* 输出最终的函数值和函数评估次数 */
     fprintf(logfile, "-----------------------Summary------------------\n"
         "Final function value: %g\n"
         "Number of function evaluations: %d\n", *minf, *numfunc);
     /* 如果全局最优值 fglobal 足够大，则输出最终函数值与全局最优值之间的百分比差距 */
     if (*fglobal > -1e99)
          fprintf(logfile, "Final function value is within %g%% of global optimum\n", 100*(*minf - *fglobal) / MAX(1.0, fabs(*fglobal)));
     /* 输出索引、最终解、解与下界的差、上界与解的差 */
     fprintf(logfile, "Index, final solution, x(i)-l(i), u(i)-x(i)\n");
     /* 循环输出每个变量的索引及其对应的最终解和边界差值 */
     for (i__ = 1; i__ <= *n; ++i__)
          fprintf(logfile, "%d, %g, %g, %g\n", i__, x[i__],
              x[i__]-l[i__], u[i__] - x[i__]);
     /* 输出分隔线 */
     fprintf(logfile, "-----------------------------------------------\n");
    }
} /* dirsummary_ */
/* 结束 direct_dirsummary_ 的定义 */
```