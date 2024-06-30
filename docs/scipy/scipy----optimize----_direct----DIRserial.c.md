# `D:\src\scipysrc\scipy\scipy\optimize\_direct\DIRserial.c`

```
/* DIRserial-transp.f -- translated by f2c (version 20050501).
   f2c output hand-cleaned by SGJ (August 2007).
*/

#include "direct-internal.h"

/* +-----------------------------------------------------------------------+ */
/* | Program       : Direct.f (subfile DIRserial.f)                        | */
/* | Last modified : 04-12-2001                                            | */
/* | Written by    : Joerg Gablonsky                                       | */
/* | SUBROUTINEs, which differ depENDing on the serial or parallel version.| */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | SUBROUTINE for sampling.                                              | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ PyObject* direct_dirsamplef_(doublereal *c__, integer *arrayi, doublereal
    *delta, integer *sample, integer *new__, integer *length,
    FILE *logfile, doublereal *f, integer *free, integer *maxi,
    integer *point, PyObject* fcn, doublereal *x, PyObject* x_seq, doublereal *l, doublereal *
    minf, integer *minpos, doublereal *u, integer *n, integer *maxfunc,
    const integer *maxdeep, integer *oops, doublereal *fmax, integer *
    ifeasiblef, integer *iinfesiblef, PyObject* args, int *force_stop)
{
    PyObject* ret = NULL;
    /* System generated locals */
    integer length_dim1, length_offset, c_dim1, c_offset, i__1, i__2;
    doublereal d__1;

    /* Local variables */
    integer i__, j, helppoint, pos, kret;

    // Ignore these variables as they are not used in this function context
    (void) logfile; (void) free; (void) maxfunc; (void) maxdeep; (void) oops;
    (void) delta; (void) sample;

/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 fcn must be declared external.                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Removed fcn.                                              | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* |             Added variable to keep track IF feasible point was found. | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Variables to pass user defined data to the function to be optimized.  | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Set the pointer to the first function to be evaluated,                | */
/* | store this position also in helppoint.                                | */
/* +-----------------------------------------------------------------------+ */
/* 参数调整 */
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

/* 函数体 */
pos = *new__;
helppoint = pos;
/* +-----------------------------------------------------------------------+ */
/* | 遍历所有需要评估函数的点。                                              | */
/* +-----------------------------------------------------------------------+ */
i__1 = *maxi + *maxi;
for (j = 1; j <= i__1; ++j) {
/* +-----------------------------------------------------------------------+ */
/* | 将位置复制到辅助数组 x 中。                                             | */
/* +-----------------------------------------------------------------------+ */
i__2 = *n;
for (i__ = 1; i__ <= i__2; ++i__) {
    x[i__] = c__[i__ + pos * c_dim1];
/* L60: */
}
/* +-----------------------------------------------------------------------+ */
/* | 调用函数进行评估。                                                     | */
/* +-----------------------------------------------------------------------+ */
if (force_stop && *force_stop)  /* 强制停止后跳过评估 */
    f[(pos << 1) + 1] = *fmax;
else {
    ret = direct_dirinfcn_(fcn, &x[1], x_seq, &l[1], &u[1], n, &f[(pos << 1) + 1],
                                      &kret, args);
    if (!ret) {
        return NULL;
    }
}
if (force_stop && *force_stop)
    kret = -1; /* 标记为无效点 */
/* +-----------------------------------------------------------------------+ */
/* | 记录是否发现了不可行点。                                               | */
/* +-----------------------------------------------------------------------+ */
*iinfesiblef = MAX(*iinfesiblef,kret);
if (kret == 0) {
/* +-----------------------------------------------------------------------+ */
/* | 如果函数评估正常，设置 f(2,pos) 的标志，并标记找到可行点。               | */
/* +-----------------------------------------------------------------------+ */
    f[(pos << 1) + 2] = 0.;
    *ifeasiblef = 0;
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 添加变量以跟踪找到的最大值。                                  | */
/* +-----------------------------------------------------------------------+ */
/* 计算最大值 */
d__1 = f[(pos << 1) + 1];
*fmax = MAX(d__1,*fmax);
}
if (kret >= 1) {
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | IF the function could not be evaluated at the given point,            | */
/* | set flag to mark this (f(2,pos) and store the maximum                 | */
/* | box-sidelength in f(1,pos).                                           | */
/* +-----------------------------------------------------------------------+ */
    f[(pos << 1) + 2] = 2.;
    f[(pos << 1) + 1] = *fmax;
}
/* +-----------------------------------------------------------------------+ */
/* | IF the function could not be evaluated due to a failure in            | */
/* | the setup, mark this.                                                 | */
/* +-----------------------------------------------------------------------+ */
if (kret == -1) {
    f[(pos << 1) + 2] = -1.;
}
/* +-----------------------------------------------------------------------+ */
/* | Set the position to the next point, at which the function             | */
/* | should be evaluated.                                                  | */
/* +-----------------------------------------------------------------------+ */
pos = point[pos];
/* L40: */
}
pos = helppoint;
/* +-----------------------------------------------------------------------+ */
/* | Iterate over all evaluated points and see, IF the minimal             | */
/* | value of the function has changed.  IF this has happened,             | */
/* | store the minimal value and its position in the array.                | */
/* | Attention: Only valid values are checked!!                            | */
/* +-----------------------------------------------------------------------+ */
i__1 = *maxi + *maxi;
for (j = 1; j <= i__1; ++j) {
if (f[(pos << 1) + 1] < *minf && f[(pos << 1) + 2] == 0.) {
    *minf = f[(pos << 1) + 1];
    *minpos = pos;
}
pos = point[pos];
/* L50: */
}
return ret;
} /* dirsamplef_ */


注释：


/* +-----------------------------------------------------------------------+ */
/* | IF the function could not be evaluated at the given point,            | */
/* | set flag to mark this (f(2,pos) and store the maximum                 | */
/* | box-sidelength in f(1,pos).                                           | */
/* +-----------------------------------------------------------------------+ */
    如果函数在给定点无法求值，将标记此点（f(2,pos)），并存储盒子边长的最大值到 f(1,pos)。
    f[(pos << 1) + 2] = 2.;
    f[(pos << 1) + 1] = *fmax;
}
/* +-----------------------------------------------------------------------+ */
/* | IF the function could not be evaluated due to a failure in            | */
/* | the setup, mark this.                                                 | */
/* +-----------------------------------------------------------------------+ */
如果由于设置失败导致函数无法评估，在 f(2,pos) 中标记 -1。
if (kret == -1) {
    f[(pos << 1) + 2] = -1.;
}
/* +-----------------------------------------------------------------------+ */
/* | Set the position to the next point, at which the function             | */
/* | should be evaluated.                                                  | */
/* +-----------------------------------------------------------------------+ */
将位置设定为下一个应该评估函数的点。
pos = point[pos];
/* L40: */
}
将 pos 重置为 helppoint。
/* +-----------------------------------------------------------------------+ */
/* | Iterate over all evaluated points and see, IF the minimal             | */
/* | value of the function has changed.  IF this has happened,             | */
/* | store the minimal value and its position in the array.                | */
/* | Attention: Only valid values are checked!!                            | */
/* +-----------------------------------------------------------------------+ */
遍历所有已评估的点，检查函数的最小值是否已经改变。如果改变了，将最小值及其位置存储在数组中。
i__1 = *maxi + *maxi;
for (j = 1; j <= i__1; ++j) {
if (f[(pos << 1) + 1] < *minf && f[(pos << 1) + 2] == 0.) {
    *minf = f[(pos << 1) + 1];
    *minpos = pos;
}
pos = point[pos];
/* L50: */
}
返回 ret。
```