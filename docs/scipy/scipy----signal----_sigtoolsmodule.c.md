# `D:\src\scipysrc\scipy\scipy\signal\_sigtoolsmodule.c`

```
/* SIGTOOLS module by Travis Oliphant

Copyright 2005 Travis Oliphant
Permission to use, copy, modify, and distribute this software without fee
is granted under the SciPy License.
*/
// 包含 Python.h 头文件，用于 Python C 扩展模块
#include <Python.h>
// 定义 PY_ARRAY_UNIQUE_SYMBOL 符号，用于 NumPy C 扩展
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
// 包含 NumPy 的 ndarrayobject.h 头文件
#include "numpy/ndarrayobject.h"
// 包含 NumPy 2.x 兼容性头文件
#include "npy_2_compat.h"

// 包含 SIGTOOLS 自定义头文件
#include "_sigtools.h"
// 包含 setjmp.h 头文件，用于实现非局部跳转
#include <setjmp.h>
// 包含标准库的头文件，用于内存分配和释放
#include <stdlib.h>

// 定义宏 PYERR，用于设置 Python 异常并跳转到错误处理标签
#define PYERR(message) {PyErr_SetString(PyExc_ValueError, message); goto fail;}

// 定义全局跳转标签 MALLOC_FAIL
jmp_buf MALLOC_FAIL;

// 函数：检查内存分配，分配指定大小的内存块
char *check_malloc(size_t size)
{
    char *the_block = malloc(size);
    if (the_block == NULL) {
        printf("\nERROR: unable to allocate %zu bytes!\n", size);
        // 如果内存分配失败，使用 longjmp 进行跳转到 MALLOC_FAIL 标签处理错误
        longjmp(MALLOC_FAIL,-1);
    }
    return the_block;
}

/********************************************************
 *
 *  Code taken from remez.c by Erik Kvaleberg which was
 *    converted from an original FORTRAN by
 *
 * AUTHORS: JAMES H. MCCLELLAN
 *
 *         DEPARTMENT OF ELECTRICAL ENGINEERING AND COMPUTER SCIENCE
 *         MASSACHUSETTS INSTITUTE OF TECHNOLOGY
 *         CAMBRIDGE, MASS. 02139
 *
 *         THOMAS W. PARKS
 *         DEPARTMENT OF ELECTRICAL ENGINEERING
 *         RICE UNIVERSITY
 *         HOUSTON, TEXAS 77001
 *
 *         LAWRENCE R. RABINER
 *         BELL LABORATORIES
 *         MURRAY HILL, NEW JERSEY 07974
 *
 *
 *  Adaptation to C by
 *      egil kvaleberg
 *      husebybakken 14a
 *      0379 oslo, norway
 *  Email:
 *      egil@kvaleberg.no
 *  Web:
 *      http://www.kvaleberg.com/
 *
 *
 *********************************************************/

// 定义常量：滤波器类型常量
#define BANDPASS       1
#define DIFFERENTIATOR 2
#define HILBERT        3

// 定义宏：跳转到标签
#define GOBACK goto
// 定义宏：for 循环宏，用于简化循环
#define DOloop(a,from,to) for ( (a) = (from); (a) <= (to); ++(a))
// 定义常量 PI 和 TWOPI
#define PI    3.14159265358979323846
#define TWOPI (PI+PI)

/*
 *-----------------------------------------------------------------------
 * FUNCTION: lagrange_interp (d)
 *  FUNCTION TO CALCULATE THE LAGRANGE INTERPOLATION
 *  COEFFICIENTS FOR USE IN THE FUNCTION gee.
 *-----------------------------------------------------------------------
 */
// 静态函数：计算拉格朗日插值系数
static double lagrange_interp(int k, int n, int m, double *x)
{
    int j, l;
    double q, retval;

    retval = 1.0;
    q = x[k];
    DOloop(l,1,m) {
        for (j = l; j <= n; j += m) {
            if (j != k)
                retval *= 2.0 * (q - x[j]);
        }
    }
    return 1.0 / retval;
}

/*
 *-----------------------------------------------------------------------
 * FUNCTION: freq_eval (gee)
 *  FUNCTION TO EVALUATE THE FREQUENCY RESPONSE USING THE
 *  LAGRANGE INTERPOLATION FORMULA IN THE BARYCENTRIC FORM
 *-----------------------------------------------------------------------
 */
// 静态函数：评估频率响应，使用拉格朗日插值公式的重心形式
static double freq_eval(int k, int n, double *grid, double *x, double *y, double *ad)
{
    int j;
    double p,c,d,xf;

    d = 0.0;
    p = 0.0;
    xf = cos(TWOPI * grid[k]);

    DOloop(j,1,n) {
        c = ad[j] / (xf - x[j]);
        d += c;
        p += c * y[j];
    }

    return p/d;
}
/*
 *-----------------------------------------------------------------------
 * SUBROUTINE: remez
 *  THIS SUBROUTINE IMPLEMENTS THE REMEZ EXCHANGE ALGORITHM
 *  FOR THE WEIGHTED CHEBYSHEV APPROXIMATION OF A CONTINUOUS
 *  FUNCTION WITH A SUM OF COSINES.  INPUTS TO THE SUBROUTINE
 *  ARE A DENSE GRID WHICH REPLACES THE FREQUENCY AXIS, THE
 *  DESIRED FUNCTION ON THIS GRID, THE WEIGHT FUNCTION ON THE
 *  GRID, THE NUMBER OF COSINES, AND AN INITIAL GUESS OF THE
 *  EXTREMAL FREQUENCIES.  THE PROGRAM MINIMIZES THE CHEBYSHEV
 *  ERROR BY DETERMINING THE BSMINEST LOCATION OF THE EXTREMAL
 *  FREQUENCIES (POINTS OF MAXIMUM ERROR) AND THEN CALCULATES
 *  THE COEFFICIENTS OF THE BEST APPROXIMATION.
 *-----------------------------------------------------------------------
 */
static int remez(double *dev, double des[], double grid[], double edge[],
       double wt[], int ngrid, int nbands, int iext[], double alpha[],
       int nfcns, int itrmax, double *work, int dimsize, int *niter_out)
        /* dev, iext, alpha                         are output types */
        /* des, grid, edge, wt, ngrid, nbands, nfcns are input types */
{
    int k, k1, kkk, kn, knz, klow, kup, nz, nzz, nm1;
    int cn;
    int j, jchnge, jet, jm1, jp1;
    int l, luck=0, nu, nut, nut1=0, niter;

    double ynz=0.0, comp=0.0, devl, gtemp, fsh, y1=0.0, err, dtemp, delf, dnum, dden;
    double aa=0.0, bb=0.0, ft, xe, xt;

    static double *a, *p, *q;
    static double *ad, *x, *y;

    // Allocate memory within the work array for different variables
    a = work; p = a + dimsize+1; q = p + dimsize+1;
    ad = q + dimsize+1; x = ad + dimsize+1; y = x + dimsize+1;
    devl = -1.0;
    nz  = nfcns+1;
    nzz = nfcns+2;
    niter = 0;

    do {
    L100:
    iext[nzz] = ngrid + 1;
    ++niter;

    if (niter > itrmax) break;

    /* printf("ITERATION %2d: ",niter); */

    // Initialize x array with cosine values of grid points
    DOloop(j,1,nz) {
        x[j] = cos(grid[iext[j]]*TWOPI);
    }
    jet = (nfcns-1) / 15 + 1;

    // Calculate ad array using Lagrange interpolation
    DOloop(j,1,nz) {
        ad[j] = lagrange_interp(j,nz,jet,x);
    }

    dnum = 0.0;
    dden = 0.0;
    k = 1;

    // Compute dnum and dden for approximation error calculation
    DOloop(j,1,nz) {
        l = iext[j];
        dnum += ad[j] * des[l];
        dden += (double)k * ad[j] / wt[l];
        k = -k;
    }
    *dev = dnum / dden;

    /* printf("DEVIATION = %lg\n",*dev); */

    // Adjust dev and y values based on current deviation
    nu = 1;
    if ( (*dev) > 0.0 ) nu = -1;
    (*dev) = -(double)nu * (*dev);
    k = nu;
    DOloop(j,1,nz) {
        l = iext[j];
        y[j] = des[l] + (double)k * (*dev) / wt[l];
        k = -k;
    }
    if ( (*dev) <= devl ) {
        // Return if deviation is less than or equal to previous minimum
        *niter_out = niter;
        return -1;
    }
    devl = (*dev);
    jchnge = 0;
    k1 = iext[1];
    knz = iext[nz];
    klow = 0;
    nut = -nu;
    j = 1;

    /*
     * SEARCH FOR THE EXTREMAL FREQUENCIES OF THE BEST APPROXIMATION
     */

    L200:
    if (j == nzz) ynz = comp;
    if (j >= nzz) goto L300;
    kup = iext[j+1];
    l = iext[j]+1;
    nut = -nut;
    if (j == 2) y1 = comp;
    comp = (*dev);
    if (l >= kup) goto L220;
    err = (freq_eval(l,nz,grid,x,y,ad)-des[l]) * wt[l];
    // 检查条件，如果 ((nut*err-comp) <= 0.0)，跳转到标签 L220
    if (((double)nut*err-comp) <= 0.0) goto L220;
    // 更新 comp 为 nut * err 的值
    comp = (double)nut * err;
    // 标签 L210，循环条件检查
    L210:
    // 如果 l 自增后大于等于 kup，跳转到标签 L215
    if (++l >= kup) goto L215;
    // 计算 err 的新值
    err = (freq_eval(l,nz,grid,x,y,ad)-des[l]) * wt[l];
    // 如果 ((nut*err-comp) <= 0.0)，跳转到标签 L215
    if (((double)nut*err-comp) <= 0.0) goto L215;
    // 更新 comp 为 nut * err 的值
    comp = (double)nut * err;
    // 回到标签 L210
    GOBACK L210;

    // 标签 L215，条件满足时执行以下操作
    L215:
    // 将 l-1 的值赋给 iext 数组的第 j 个元素
    iext[j++] = l - 1;
    // 更新 klow 的值为 l-1
    klow = l - 1;
    // jchnge 自增
    ++jchnge;
    // 回到标签 L200
    GOBACK L200;

    // 标签 L220，条件满足时执行以下操作
    L220:
    // l 自减
    --l;
    // 标签 L225，循环条件检查
    L225:
    // 如果 l 自减后小于等于 klow，跳转到标签 L250
    if (--l <= klow) goto L250;
    // 计算 err 的新值
    err = (freq_eval(l,nz,grid,x,y,ad)-des[l]) * wt[l];
    // 如果 ((double)nut*err-comp) > 0.0，跳转到标签 L230
    if (((double)nut*err-comp) > 0.0) goto L230;
    // 如果 jchnge <= 0，跳转到标签 L225
    if (jchnge <= 0) goto L225;
    // 跳转到标签 L260
    goto L260;

    // 标签 L230，条件满足时执行以下操作
    L230:
    // 更新 comp 为 nut * err 的值
    comp = (double)nut * err;
    // 标签 L235，循环条件检查
    L235:
    // 如果 l 自减后小于等于 klow，跳转到标签 L240
    if (--l <= klow) goto L240;
    // 计算 err 的新值
    err = (freq_eval(l,nz,grid,x,y,ad)-des[l]) * wt[l];
    // 如果 ((double)nut*err-comp) <= 0.0，跳转到标签 L240
    if (((double)nut*err-comp) <= 0.0) goto L240;
    // 更新 comp 为 nut * err 的值
    comp = (double)nut * err;
    // 回到标签 L235
    GOBACK L235;
    // 标签 L240，条件满足时执行以下操作
    L240:
    // 更新 klow 为 iext 数组的第 j 个元素的值
    klow = iext[j];
    // 将 l+1 的值赋给 iext 数组的第 j 个元素
    iext[j] = l+1;
    // j 自增
    ++j;
    // jchnge 自增
    ++jchnge;
    // 回到标签 L200
    GOBACK L200;

    // 标签 L250，条件满足时执行以下操作
    L250:
    // l 更新为 iext 数组的第 j 个元素加 1
    l = iext[j]+1;
    // 如果 jchnge 大于 0，跳转到标签 L215
    if (jchnge > 0) GOBACK L215;

    // 标签 L255，条件满足时执行以下操作
    L255:
    // 如果 l 自增后大于等于 kup，跳转到标签 L260
    if (++l >= kup) goto L260;
    // 计算 err 的新值
    err = (freq_eval(l,nz,grid,x,y,ad)-des[l]) * wt[l];
    // 如果 ((double)nut*err-comp) <= 0.0，跳转到标签 L255
    if (((double)nut*err-comp) <= 0.0) GOBACK L255;
    // 更新 comp 为 nut * err 的值
    comp = (double)nut * err;

    // 回到标签 L210
    GOBACK L210;
    // 标签 L260，条件满足时执行以下操作
    L260:
    // 更新 klow 为 iext 数组的第 j 个元素的值
    klow = iext[j++];
    // 回到标签 L200
    GOBACK L200;

    // 标签 L300，条件满足时执行以下操作
    L300:
    // 如果 j 大于 nzz，跳转到标签 L320
    if (j > nzz) goto L320;
    // 如果 k1 大于 iext 数组的第 1 个元素的值，更新 k1
    if (k1 > iext[1] ) k1 = iext[1];
    // 如果 knz 小于 iext 数组的第 nz 个元素的值，更新 knz
    if (knz < iext[nz]) knz = iext[nz];
    // 将 nut1 的值赋给 nut
    nut1 = nut;
    // 更新 nut 为 -nu
    nut = -nu;
    // l 设为 0
    l = 0;
    // 更新 kup 为 k1
    kup = k1;
    // 更新 comp 为 ynz*(1.00001)
    comp = ynz*(1.00001);
    // luck 设为 1
    luck = 1;
    // 标签 L310，循环条件检查
    L310:
    // 如果 l 自增后大于等于 kup，跳转到标签 L315
    if (++l >= kup) goto L315;
    // 计算 err 的新值
    err = (freq_eval(l,nz,grid,x,y,ad)-des[l]) * wt[l];
    // 如果 ((double)nut*err-comp) <= 0.0，跳转到标签 L310
    if (((double)nut*err-comp) <= 0.0) GOBACK L310;
    // 更新 comp 为 nut * err 的值
    comp = (double) nut * err;
    // j 设为 nzz
    j = nzz;
    // 回到标签 L210
    GOBACK L210;

    // 标签 L315，条件满足时执行以下操作
    L315:
    // 更新 luck 为 6
    luck = 6;
    // 跳转到标签 L325
    goto L325;

    // 标签 L320，条件满足时执行以下操作
    L320:
    // 如果 luck 大于 9，跳转到标签 L350
    if (luck > 9) goto L350;
    // 如果 comp 大于 y1，更新 y1
    if (comp > y1) y1 = comp;
    // 更新 k1 为 iext 数组的第 nzz 个元素的值
    k1 = iext[nzz];
    // 标签 L325，条件满足时执行以下操作
    L325:
    // l 设为 ngrid+1
    l = ngrid+1;
    // 更新 klow 为 knz
    klow = knz;
    // 更新 nut 为 -nut1
    nut = -nut1;
    // 更新 comp 为 y1*(1.00001)
    comp = y1*(1.00001);
    // 标签 L330，循环条件检查
    L330:
    // 如果 l 自减后小于等于 klow，跳转到标签 L340
    if (--l <= klow) goto L340;
    // 计算 err 的新值
    err = (freq_eval(l,nz,grid,x,y,ad)-des[l]) * wt[l];
    // 如果 ((double)nut*err-comp) <= 0.0，跳转到标签 L330
    if (((double
/*
 *    CALCULATION OF THE COEFFICIENTS OF THE BEST APPROXIMATION
 *    USING THE INVERSE DISCRETE FOURIER TRANSFORM
 */
nm1 = nfcns - 1;    // 计算 nfcns 减 1 的值，用于后续循环控制
fsh = 1.0e-06;      // 设置一个小的浮点数值 fsh 作为比较阈值
gtemp = grid[1];    // 保存 grid 数组的第一个元素的值到 gtemp 变量
x[nzz] = -2.0;      // 将 x 数组的第 nzz 个元素设置为 -2.0
cn  = 2*nfcns - 1;  // 计算一个常数值并存储到 cn 变量中
delf = 1.0/cn;      // 计算步长值并存储到 delf 变量中
l = 1;              // 初始化 l 为 1
kkk = 0;            // 初始化 kkk 为 0

if (edge[1] == 0.0 && edge[2*nbands] == 0.5) kkk = 1;  // 检查条件，设置 kkk 变量的值

if (nfcns <= 3) kkk = 1;  // 根据 nfcns 的大小设置 kkk 变量的值
if (kkk != 1) {
    dtemp = cos(TWOPI*grid[1]);  // 计算余弦值并存储到 dtemp 变量中
    dnum  = cos(TWOPI*grid[ngrid]);  // 计算余弦值并存储到 dnum 变量中
    aa    = 2.0/(dtemp-dnum);   // 计算系数 aa
    bb    = -(dtemp+dnum)/(dtemp-dnum);  // 计算系数 bb
}

DOloop(j,1,nfcns) {   // 开始循环，j 从 1 到 nfcns
    ft = (j - 1) * delf;    // 计算频率值 ft
    xt = cos(TWOPI*ft);     // 计算余弦值并存储到 xt 变量中
    if (kkk != 1) {
        xt = (xt-bb)/aa;    // 应用线性变换到 xt 变量
#if 0
        /*XX* check up !! */
        xt1 = sqrt(1.0-xt*xt);   // 计算平方根值并存储到 xt1 变量中
        ft = atan2(xt1,xt)/TWOPI;    // 计算反正切值并存储到 ft 变量中
#else
        ft = acos(xt)/TWOPI;    // 计算反余弦值并存储到 ft 变量中
#endif
    }
L410:
    xe = x[l];    // 从数组 x 中取出第 l 个元素的值到 xe 变量
    if (xt > xe) goto L420;    // 根据条件跳转到 L420 标签处
    if ((xe-xt) < fsh) goto L415;   // 根据条件跳转到 L415 标签处
    ++l;    // l 自增
    GOBACK L410;    // 返回 L410 处继续执行
L415:
    a[j] = y[l];    // 将数组 y 中第 l 个元素的值存储到数组 a 中第 j 个位置
    goto L425;    // 跳转到 L425 标签处
L420:
    if ((xt-xe) < fsh) GOBACK L415;    // 根据条件跳转到 L415 标签处
    grid[1] = ft;    // 将 ft 的值存储到 grid 数组的第一个元素中
    a[j] = freq_eval(1,nz,grid,x,y,ad);   // 调用 freq_eval 函数，并将结果存储到数组 a 的第 j 个位置
L425:
    if (l > 1) l = l-1;    // 如果 l 大于 1，则 l 减 1
}

grid[1] = gtemp;    // 恢复 grid 数组的第一个元素的原始值
dden = TWOPI / cn;    // 计算常数值并存储到 dden 变量中
DOloop (j,1,nfcns) {    // 开始循环，j 从 1 到 nfcns
    dtemp = 0.0;    // 初始化 dtemp 为 0.0
    dnum = (j-1) * dden;    // 计算一个常数值并存储到 dnum 变量中
    if (nm1 >= 1) {
        DOloop(k,1,nm1) {    // 开始内部循环，k 从 1 到 nm1
        dtemp += a[k+1] * cos(dnum*k);    // 根据公式计算累加值并存储到 dtemp 变量中
        }
    }
    alpha[j] = 2.0 * dtemp + a[1];    // 计算 alpha 数组的第 j 个元素的值
}

DOloop(j,2,nfcns) alpha[j] *= 2.0 / cn;    // 根据公式修改 alpha 数组的元素值
alpha[1] /= cn;    // 根据公式修改 alpha 数组的第一个元素的值

if (kkk != 1) {
p[1] = 2.0*alpha[nfcns]*bb+alpha[nm1];    // 根据公式计算 p 数组的第一个元素的值
p[2] = 2.0*aa*alpha[nfcns];    // 根据公式计算 p 数组的第二个元素的值
q[1] = alpha[nfcns-2]-alpha[nfcns];    // 根据公式计算 q 数组的第一个元素的值
DOloop(j,2,nm1) {    // 开始循环，j 从 2 到 nm1
    if (j >= nm1) {    // 如果 j 大于等于 nm1
    aa *= 0.5;    // aa 自乘 0.5
    bb *= 0.5;    // bb 自乘 0.5
    }
    p[j+1] = 0.0;    // 将 p 数组的第 j+1 个元素的值设置为 0.0
    DOloop(k,1,j) {    // 开始内部循环，k 从 1 到 j
    a[k] = p[k];    // 将 p 数组中第 k 个元素的值存储到数组 a 中第 k 个位置
    p[k] = 2.0 * bb * a[k];    // 根据公式修改 p 数组的第 k 个元素的值
    }
    p[2] += a[1] * 2.0 *aa;    // 根据公式修改 p 数组的第二个元素的值
    jm1 = j - 1;    // 将 j-1 的值存储到 jm1 变量中
    DOloop(k,1,jm1) p[k] += q[k] + aa * a[k+1];    // 根据公式修改 p 数组的前 jm1 个元素的值
    jp1 = j + 1;    // 将 j+1 的值存储到 jp1 变量中
    DOloop(k,3,jp1) p[k] += aa * a[k-1];    // 根据公式修改 p 数组的后 jp1-1 个元素的值

    if (j != nm1) {    // 如果 j 不等于 nm1
    DOloop(k,1,j) q[k] = -a[k];    // 根据公式修改 q 数组的前 j 个元素的值
    q[1] += alpha[nfcns - 1 - j];    // 根据公式修改 q 数组的第一个元素的值
    }
}
DOloop(j,1,nfcns) alpha[j] = p[j];    // 根据公式更新 alpha 数组的元素值
}

if (nfcns <= 3) {
alpha[nfcns+1] = alpha[nfcns+2] = 0.
/*
 *-----------------------------------------------------------------------
 * FUNCTION: wate
 *  FUNCTION TO CALCULATE THE WEIGHT FUNCTION AS A FUNCTION
 *  OF FREQUENCY.  SIMILAR TO THE FUNCTION eff, THIS FUNCTION CAN
 *  BE REPLACED BY A USER-WRITTEN ROUTINE TO CALCULATE ANY
 *  DESIRED WEIGHTING FUNCTION.
 *-----------------------------------------------------------------------
 */
static double wate(double freq, double *fx, double *wtx, int lband, int jtype)
{
    // 如果 jtype 不等于 2，则返回 wtx[lband]
    if (jtype != 2)          return wtx[lband];
    // 如果 fx[lband] 大于等于 0.0001，则返回 wtx[lband] / freq
    if (fx[lband] >= 0.0001) return wtx[lband] / freq;
    // 否则返回 wtx[lband]
    return                          wtx[lband];
}

/*********************************************************/

/*  This routine accepts basic input information and puts it in
 *  the form expected by remez.
 *
 *  Adapted from main() by Travis Oliphant
 */

static int pre_remez(double *h2, int numtaps, int numbands, double *bands,
                     double *response, double *weight, int type, int maxiter,
                     int grid_density, int *niter_out) {

    int jtype, nbands, nfilt, lgrid, nz;
    int neg, nodd, nm1;
    int j, k, l, lband, dimsize;
    double delf, change, fup, temp;
    double *tempstor, *edge, *h, *fx, *wtx;
    double *des, *grid, *wt, *alpha, *work;
    double dev;
    int ngrid;
    int *iext;
    int nfcns, wrksize, total_dsize, total_isize;

    // 设置 lgrid 为 grid_density
    lgrid = grid_density;
    // 计算 dimsize，ceil() 函数向上取整
    dimsize = (int) ceil(numtaps/2.0 + 2);
    // 计算 wrksize
    wrksize = grid_density * dimsize;
    nfilt = numtaps;
    jtype = type; nbands = numbands;
    /* Note:  code assumes these arrays start at 1 */
    // 将 bands、response、weight 数组向前偏移一位
    edge = bands-1;
    h = h2 - 1;
    fx = response - 1;
    wtx = weight - 1;

    total_dsize = (dimsize+1)*7 + 3*(wrksize+1);
    total_isize = (dimsize+1);
    /* Need space for:  (all arrays ignore the first element).

       des  (wrksize+1)
       grid (wrksize+1)
       wt   (wrksize+1)
       iext (dimsize+1)   (integer)
       alpha (dimsize+1)
       work  (dimsize+1)*6

    */
    // 分配临时存储空间
    tempstor = malloc((total_dsize)*sizeof(double)+(total_isize)*sizeof(int));
    if (tempstor == NULL) return -2;

    // 设置数组指针
    des = tempstor; grid = des + wrksize+1;
    wt = grid + wrksize+1; alpha = wt + wrksize+1;
    work = alpha + dimsize+1; iext = (int *)(work + (dimsize+1)*6);

    /* Set up problem on dense_grid */

    // 根据 jtype 和 neg 初始化 neg
    neg = 1;
    if (jtype == 1) neg = 0;
    // 计算 nfilt 是否为奇数
    nodd = nfilt % 2;
    nfcns = nfilt / 2;
    if (nodd == 1 && neg == 0) nfcns = nfcns + 1;

    /*
     * SET UP THE DENSE GRID. THE NUMBER OF POINTS IN THE GRID
     * IS (FILTER LENGTH + 1)*GRID DENSITY/2
     */
    // 设置 grid 的第一个元素为 edge 的第一个元素
    grid[1] = edge[1];
    // 计算 delf
    delf = lgrid * nfcns;
    delf = 0.5 / delf;
    // 如果 neg 不为 0，则根据条件更新 grid 的第一个元素
    if (neg != 0) {
        if (edge[1] < delf) grid[1] = delf;
    }
    j = 1;
    l = 1;
    lband = 1;

    /*
     * CALCULATE THE DESIRED MAGNITUDE RESPONSE AND THE WEIGHT
     * FUNCTION ON THE GRID
     */
    for (;;) {
        fup = edge[l + 1];
        // 这里需要补充代码注释，由于字数限制，请查阅相关资料
    do {
        temp = grid[j];  # 将数组grid中第j个元素的值赋给temp
        des[j] = eff(temp,fx,lband,jtype);  # 计算并存储eff函数在temp处的返回值到数组des的第j个位置
        wt[j] = wate(temp,fx,wtx,lband,jtype);  # 计算并存储wate函数在temp处的返回值到数组wt的第j个位置
        if (++j > wrksize) {  # 如果j增加后超过了wrksize
                /* too many points, or too dense grid */  # 太多点或者网格过于密集
                free(tempstor);  # 释放tempstor指向的内存空间
                return -1;  # 返回-1表示错误
            }
        grid[j] = temp + delf;  # 将temp加上delf后的值赋给数组grid的第j个位置
    } while (grid[j] <= fup);  # 当grid的第j个位置的值小于等于fup时循环

    grid[j-1] = fup;  # 将fup赋给数组grid的第j-1个位置
    des[j-1] = eff(fup,fx,lband,jtype);  # 计算并存储eff函数在fup处的返回值到数组des的第j-1个位置
    wt[j-1] = wate(fup,fx,wtx,lband,jtype);  # 计算并存储wate函数在fup处的返回值到数组wt的第j-1个位置
    ++lband;  # lband增加1
    l += 2;  # l增加2
    if (lband > nbands) break;  # 如果lband大于nbands则跳出循环
    grid[j] = edge[l];  # 将数组edge的第l个位置的值赋给数组grid的第j个位置
    }

    ngrid = j - 1;  # 将j-1赋给ngrid
    if (neg == nodd) {  # 如果neg等于nodd
    if (grid[ngrid] > (0.5-delf)) --ngrid;  # 如果数组grid的第ngrid个位置的值大于0.5-delf，则ngrid减1
    }

    /*
     * SET UP A NEW APPROXIMATION PROBLEM WHICH IS EQUIVALENT
     * TO THE ORIGINAL PROBLEM
     */
    if (neg <= 0) {  # 如果neg小于等于0
    if (nodd != 1) {  # 如果nodd不等于1
        DOloop(j,1,ngrid) {  # 从1到ngrid循环，对每个j执行以下操作
        change = cos(PI*grid[j]);  # 计算cos(PI*grid[j])并赋给change
        des[j] = des[j] / change;  # 将des数组第j个位置的值除以change
        wt[j]  = wt[j] * change;  # 将wt数组第j个位置的值乘以change
        }
    }
    } else {  # 否则
    if (nodd != 1) {  # 如果nodd不等于1
        DOloop(j,1,ngrid) {  # 从1到ngrid循环，对每个j执行以下操作
        change = sin(PI*grid[j]);  # 计算sin(PI*grid[j])并赋给change
        des[j] = des[j] / change;  # 将des数组第j个位置的值除以change
        wt[j]  = wt[j]  * change;  # 将wt数组第j个位置的值乘以change
        }
    } else {  # 否则
        DOloop(j,1,ngrid) {  # 从1到ngrid循环，对每个j执行以下操作
        change = sin(TWOPI * grid[j]);  # 计算sin(TWOPI * grid[j])并赋给change
        des[j] = des[j] / change;  # 将des数组第j个位置的值除以change
        wt[j]  = wt[j]  * change;  # 将wt数组第j个位置的值乘以change
        }
    }
    }

    /*XX*/
    temp = (double)(ngrid-1) / (double)nfcns;  # 计算(ngrid-1)/nfcns并将结果赋给temp
    DOloop(j,1,nfcns) {  # 从1到nfcns循环，对每个j执行以下操作
    iext[j] = (int)((j-1)*temp) + 1; /* round? !! */  # 计算(j-1)*temp的整数部分加1，并赋给iext数组的第j个位置
    }
    iext[nfcns+1] = ngrid;  # 将ngrid赋给iext数组的第nfcns+1个位置
    nm1 = nfcns - 1;  # 将nfcns-1赋给nm1
    nz  = nfcns + 1;  # 将nfcns+1赋给nz

    if (remez(&dev, des, grid, edge, wt, ngrid, numbands, iext, alpha, nfcns,
              maxiter, work, dimsize, niter_out) < 0) {
        free(tempstor);  # 释放tempstor指向的内存空间
        return -1;  # 返回-1表示错误
    }

    /*
     * CALCULATE THE IMPULSE RESPONSE.
     */
    if (neg <= 0) {  # 如果neg小于等于0

    if (nodd != 0) {  # 如果nodd不等于0
        DOloop(j,1,nm1) {  # 从1到nm1循环，对每个j执行以下操作
        h[j] = 0.5 * alpha[nz-j];  # 将0.5乘以alpha数组的第nz-j个位置的值赋给h数组的第j个位置
        }
        h[nfcns] = alpha[1];  # 将alpha数组的第1个位置的值赋给h数组的第nfcns个位置
    } else {  # 否则
        h[1] = 0.25 * alpha[nfcns];  # 将0.25乘以alpha数组的第nfcns个位置的值赋给h数组的第1个位置
        DOloop(j,2,nm1) {  # 从2到nm1循环，对每个j执行以下操作
        h[j] = 0.25 * (alpha[nz-j] + alpha[nfcns+2-j]);  # 将0.25乘以alpha数组的第nz-j个位置的值与alpha数组的第nfcns+2-j个位置的值之和赋给h数组的第j个位置
        }
        h[nfcns] = 0.5*alpha[1] + 0.25*alpha[2];  # 将0.5乘以alpha数组的第1个位置的值与0.25乘以alpha数组的第2个位置的值之和赋给h数组的第nfcns个位置
    }
    } else {  # 否则
    if (nodd != 0) {  # 如果nodd不等于0
        h[1] = 0.25 * alpha[nfcns];  # 将0.25乘以alpha数组的第nfcns个位置的值赋给h数组的第1个位置
        h[2] = 0.25 * alpha[nm1];  # 将0.25乘以alpha数组的第nm1个位置的值赋给h数组的第2个位置
        DOloop(j,3,nm1) {  # 从3到nm1循环，对每个j执行以下操作
        h[j] = 0.25 * (alpha[nz-j] - alpha[nfcns+3-j]);  # 将0.25乘以alpha数组的第nz-j个位置的值与alpha数组的第nfcns+3-j个位置的值之差赋给h数组的第j个位置
        }
        h[nfcns] = 0.5 * alpha[1] - 0.25 * alpha[3];  # 将0.5乘以alpha数组的第1个位置的值与0.25乘以alpha数组的第3个位置的值之差赋给h数组的第
}

/**************************************************************
 * End of remez routines
 **************************************************************/


/****************************************************/
/* End of python-independent routines               */
/****************************************************/

/******************************************/

static char doc_correlateND[] = "out = _correlateND(a,kernel,mode) \n\n   mode = 0 - 'valid', 1 - 'same', \n  2 - 'full' (default)";

/*******************************************************************/

static char doc_convolve2d[] = "out = _convolve2d(in1, in2, flip, mode, boundary, fillvalue)";

extern int pylab_convolve_2d(char*, npy_intp*, char*, npy_intp*, char*,
                             npy_intp*, npy_intp*, npy_intp*, int, char*);

static PyObject *_sigtools_convolve2d(PyObject *NPY_UNUSED(dummy), PyObject *args) {
    // 定义函数参数和局部变量
    PyObject *in1=NULL, *in2=NULL, *fill_value=NULL;
    int mode=2, boundary=0, typenum, flag, flip=1, ret;
    npy_intp *aout_dimens=NULL;
    int i;
    PyArrayObject *ain1=NULL, *ain2=NULL, *aout=NULL;
    PyArrayObject *afill=NULL;

    // 解析 Python 函数参数
    if (!PyArg_ParseTuple(args, "OO|iiiO", &in1, &in2, &flip, &mode, &boundary,
                          &fill_value)) {
        return NULL;
    }

    // 确定输入数组的数据类型
    typenum = PyArray_ObjectType(in1, 0);
    typenum = PyArray_ObjectType(in2, typenum);

    // 根据输入创建 NumPy 数组对象
    ain1 = (PyArrayObject *)PyArray_FromObject(in1, typenum, 2, 2);
    if (ain1 == NULL) goto fail;
    ain2 = (PyArrayObject *)PyArray_FromObject(in2, typenum, 2, 2);
    if (ain2 == NULL) goto fail;

    // 检查边界条件是否正确
    if ((boundary != PAD) && (boundary != REFLECT) && (boundary != CIRCULAR))
      PYERR("Incorrect boundary value.");

    // 处理 PAD 边界条件的填充值
    if ((boundary == PAD) & (fill_value != NULL)) {
        // 尝试从 Python 对象创建填充值的 NumPy 数组对象
        afill = (PyArrayObject *)PyArray_FromObject(fill_value, typenum, 0, 0);
        if (afill == NULL) {
            /* 对于向后兼容性，尝试通过复数路径 */
            PyArrayObject *tmp;
            PyErr_Clear();
            tmp = (PyArrayObject *)PyArray_FromObject(fill_value,
                                                      NPY_CDOUBLE, 0, 0);
            if (tmp == NULL) goto fail;
            // 尝试将复数类型转换为指定类型
            afill = (PyArrayObject *)PyArray_Cast(tmp, typenum);
            Py_DECREF(tmp);
            if (afill == NULL) goto fail;
            PYERR("could not cast `fillvalue` directly to the output "
                  "type (it was first converted to complex).");
        }
        // 检查填充值数组的大小是否为1
        if (PyArray_SIZE(afill) != 1) {
            if (PyArray_SIZE(afill) == 0) {
                PyErr_SetString(PyExc_ValueError,
                                "`fillvalue` cannot be an empty array.");
                goto fail;
            }
            PYERR("`fillvalue` must be scalar or an array with "
                  "one element.");
        }
    }
    else {
        // 创建一个零填充的数组
        afill = (PyArrayObject *)PyArray_ZEROS(0, NULL, typenum, 0);
        if (afill == NULL) goto fail;
    }
    # 分配存储输出数组维度信息的内存空间，大小为输入数组的维度数乘以 npy_intp 的大小
    aout_dimens = malloc(PyArray_NDIM(ain1)*sizeof(npy_intp));
    # 检查内存分配是否成功，如果失败则跳转到失败处理标签 fail
    if (aout_dimens == NULL) goto fail;
    
    # 根据 mode 变量的值进行不同的操作
    switch(mode & OUTSIZE_MASK) {
    case VALID:
        # 当 mode 是 VALID 时，计算输出数组每个维度的大小
        for (i = 0; i < PyArray_NDIM(ain1); i++) {
            aout_dimens[i] = PyArray_DIMS(ain1)[i] - PyArray_DIMS(ain2)[i] + 1;
            # 如果计算出的维度大小小于 0，则输出无效，设置错误信息并跳转到 fail 处理标签
            if (aout_dimens[i] < 0) {
                PyErr_SetString(PyExc_ValueError,
                                "no part of the output is valid, use option 1 (same) or 2 "
                                "(full) for third argument");
                goto fail;
            }
        }
        break;
    case SAME:
        # 当 mode 是 SAME 时，将输出数组的维度设置为输入数组 ain1 的维度
        for (i = 0; i < PyArray_NDIM(ain1); i++) {
            aout_dimens[i] = PyArray_DIMS(ain1)[i];
        }
        break;
    case FULL:
        # 当 mode 是 FULL 时，计算输出数组每个维度的大小，使得满足输出的完全卷积
        for (i = 0; i < PyArray_NDIM(ain1); i++) {
            aout_dimens[i] = PyArray_DIMS(ain1)[i] + PyArray_DIMS(ain2)[i] - 1;
        }
        break;
    default:
        # 如果 mode 的值不在预期范围内，设置错误信息并跳转到 fail 处理标签
        PyErr_SetString(PyExc_ValueError,
                        "mode must be 0 (valid), 1 (same), or 2 (full)");
        goto fail;
    }

    # 根据计算得到的维度信息创建一个新的 PyArrayObject 对象，表示输出数组 aout
    aout = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ain1), aout_dimens,
                                              typenum);
    # 检查创建数组是否成功，如果失败则跳转到 fail 处理标签
    if (aout == NULL) goto fail;

    # 根据不同的参数设置构造 flag 变量
    flag = mode + boundary + (typenum << TYPE_SHIFT) + \
      (flip != 0) * FLIP_MASK;

    # 调用二维卷积函数进行卷积计算
    ret = pylab_convolve_2d (PyArray_DATA(ain1),      /* Input data Ns[0] x Ns[1] */
                     PyArray_STRIDES(ain1),   /* Input strides */
                     PyArray_DATA(aout),      /* Output data */
                     PyArray_STRIDES(aout),   /* Output strides */
                     PyArray_DATA(ain2),      /* coefficients in filter */
                     PyArray_STRIDES(ain2),   /* coefficients strides */
                     PyArray_DIMS(ain2),      /* Size of kernel Nwin[2] */
                     PyArray_DIMS(ain1),      /* Size of image Ns[0] x Ns[1] */
                     flag,                    /* convolution parameters */
                     PyArray_DATA(afill));    /* fill value */

    # 根据卷积函数的返回值进行不同的处理
    switch (ret) {
    case 0:
        # 当卷积成功完成时，释放输出数组维度信息的内存空间，解除对输入数组和填充值对象的引用，并返回输出数组对象
        free(aout_dimens);
        Py_DECREF(ain1);
        Py_DECREF(ain2);
        Py_XDECREF(afill);
        return (PyObject *)aout;
        break;
    case -5:
    case -4:
        # 当卷积函数返回特定的错误码时，设置相应的错误信息并跳转到 fail 处理标签
        PyErr_SetString(PyExc_ValueError,
                "convolve2d not available for this type.");
        goto fail;
    case -3:
        # 当卷积函数返回内存分配失败的错误码时，设置内存分配失败的错误信息并跳转到 fail 处理标签
        PyErr_NoMemory();
        goto fail;
    case -2:
        # 当卷积函数返回无效边界类型的错误码时，设置相应的错误信息并跳转到 fail 处理标签
        PyErr_SetString(PyExc_ValueError,
                "Invalid boundary type.");
        goto fail;
    case -1:
        # 当卷积函数返回无效输出标志的错误码时，设置相应的错误信息并跳转到 fail 处理标签
        PyErr_SetString(PyExc_ValueError,
                "Invalid output flag.");
        goto fail;
    }
fail:
    // 释放内存：释放 aout_dimens 所指向的内存空间
    free(aout_dimens);
    // 减少对 ain1 所引用对象的引用计数
    Py_XDECREF(ain1);
    // 减少对 ain2 所引用对象的引用计数
    Py_XDECREF(ain2);
    // 减少对 aout 所引用对象的引用计数
    Py_XDECREF(aout);
    // 减少对 afill 所引用对象的引用计数
    Py_XDECREF(afill);
    // 返回 NULL，表示函数执行失败
    return NULL;
}

/*******************************************************************/

static char doc_remez[] =
    "h = _remez(numtaps, bands, des, weight, type, fs, maxiter, grid_density)\n"
    "  returns the optimal (in the Chebyshev/minimax sense) FIR filter impulse\n"
    "  response given a set of band edges, the desired response on those bands,\n"
    "  and the weight given to the error in those bands.  Bands is a monotonic\n"
    "  vector with band edges given in frequency domain where fs is the sampling\n"
    "  frequency.";

static PyObject *_sigtools_remez(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *bands, *des, *weight;
    int k, numtaps, numbands, type = BANDPASS, err;
    PyArrayObject *a_bands=NULL, *a_des=NULL, *a_weight=NULL;
    PyArrayObject *h=NULL;
    npy_intp ret_dimens; int maxiter = 25, grid_density = 16;
    double oldvalue, *dptr, fs = 1.0;
    char mystr[255];
    int niter = -1;

    // 解析 Python 函数参数
    if (!PyArg_ParseTuple(args, "iOOO|idii", &numtaps, &bands, &des, &weight,
                          &type, &fs, &maxiter, &grid_density)) {
        return NULL;
    }

    // 检查滤波器类型是否合法
    if (type != BANDPASS && type != DIFFERENTIATOR && type != HILBERT) {
        PyErr_SetString(PyExc_ValueError,
                        "The type must be BANDPASS, DIFFERENTIATOR, or HILBERT.");
        return NULL;
    }

    // 检查 numtaps 是否合法
    if (numtaps < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "The number of taps must be greater than 1.");
        return NULL;
    }

    // 从 Python 对象创建连续的 PyArrayObject
    a_bands = (PyArrayObject *)PyArray_ContiguousFromObject(bands, NPY_DOUBLE, 1, 1);
    if (a_bands == NULL) goto fail;
    a_des = (PyArrayObject *)PyArray_ContiguousFromObject(des, NPY_DOUBLE, 1, 1);
    if (a_des == NULL) goto fail;
    a_weight = (PyArrayObject *)PyArray_ContiguousFromObject(weight, NPY_DOUBLE, 1, 1);
    if (a_weight == NULL) goto fail;

    // 获取数组的维度信息
    numbands = PyArray_DIMS(a_des)[0];
    // 检查输入数组的长度是否匹配
    if ((PyArray_DIMS(a_bands)[0] != 2 * numbands) ||
        (PyArray_DIMS(a_weight)[0] != numbands)) {
        PyErr_SetString(PyExc_ValueError,
                        "The inputs desired and weight must have same length.\n  "
                        "The input bands must have twice this length.");
        goto fail;
    }

    /*
     * Check the bands input to see if it is monotonic, divide by
     * fs to take from range 0 to 0.5 and check to see if in that range
     */
    // 获取数组的数据指针，并初始化变量 oldvalue
    dptr = (double *)PyArray_DATA(a_bands);
    oldvalue = 0;
    for (k=0; k < 2*numbands; k++) {
        // 检查每个频带的端点值是否单调递增，起始必须为零
        if (*dptr < oldvalue) {
            PyErr_SetString(PyExc_ValueError,
                            "Bands must be monotonic starting at zero.");
            // 设置异常并跳转到失败标签
            goto fail;
        }
        // 检查频带边缘值是否小于采样频率的一半
        if (*dptr * 2 > fs) {
            PyErr_SetString(PyExc_ValueError,
                            "Band edges should be less than 1/2 the sampling frequency");
            // 设置异常并跳转到失败标签
            goto fail;
        }
        // 更新旧的频带端点值
        oldvalue = *dptr;
        // 将频带端点值转换为以采样频率为1.0的比例
        *dptr = oldvalue / fs;  /* Change so that sampling frequency is 1.0 */
        // 指针移动到下一个频带端点值
        dptr++;
    }

    // 返回的数组维度为滤波器的长度
    ret_dimens = numtaps;
    // 创建一个双精度数组对象来存储滤波器系数
    h = (PyArrayObject *)PyArray_SimpleNew(1, &ret_dimens, NPY_DOUBLE);
    // 检查数组对象是否创建成功
    if (h == NULL) goto fail;

    // 调用预处理函数 pre_remez 生成滤波器系数
    err = pre_remez((double *)PyArray_DATA(h), numtaps, numbands,
                    (double *)PyArray_DATA(a_bands),
                    (double *)PyArray_DATA(a_des),
                    (double *)PyArray_DATA(a_weight),
                    type, maxiter, grid_density, &niter);
    // 检查生成滤波器系数的过程中是否出现错误
    if (err < 0) {
        // 如果迭代失败
        if (err == -1) {
            // 格式化错误消息
            sprintf(mystr, "Failure to converge at iteration %d, try reducing transition band width.\n", niter);
            // 设置异常并跳转到失败标签
            PyErr_SetString(PyExc_ValueError, mystr);
            goto fail;
        }
        // 如果内存分配失败
        else if (err == -2) {
            // 设置内存错误异常并跳转到失败标签
            PyErr_NoMemory();
            goto fail;
        }
    }

    // 释放使用过的数组对象
    Py_DECREF(a_bands);
    Py_DECREF(a_des);
    Py_DECREF(a_weight);

    // 返回包含滤波器系数的数组对象
    return PyArray_Return(h);
fail:
    // 释放 Python 对象引用，避免内存泄漏
    Py_XDECREF(a_bands);
    // 释放 Python 对象引用，避免内存泄漏
    Py_XDECREF(a_des);
    // 释放 Python 对象引用，避免内存泄漏
    Py_XDECREF(a_weight);
    // 释放 Python 对象引用，避免内存泄漏
    Py_XDECREF(h);
    // 返回 NULL 指示函数执行失败
    return NULL;
}

static char doc_median2d[] = "filt = _median2d(data, size)";

// 声明 _sigtools_median2d 函数，接受一个 PyObject 参数和一个 PyObject* 参数
static PyObject *_sigtools_median2d(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *image=NULL, *size=NULL;
    int typenum;
    PyArrayObject *a_image=NULL, *a_size=NULL;
    PyArrayObject *a_out=NULL;
    npy_intp Nwin[2] = {3,3};

    // 解析 Python 函数参数，返回 NULL 表示解析失败
    if (!PyArg_ParseTuple(args, "O|O", &image, &size)) return NULL;

    // 获取图像数据的类型编号
    typenum = PyArray_ObjectType(image, 0);
    // 将图像数据转换为连续的 PyArrayObject 对象，如果失败则跳转到 fail 标签
    a_image = (PyArrayObject *)PyArray_ContiguousFromObject(image, typenum, 2, 2);
    if (a_image == NULL) goto fail;

    // 如果 size 参数不为空，则处理 size 参数
    if (size != NULL) {
        // 将 size 转换为连续的 PyArrayObject 对象，如果失败则跳转到 fail 标签
        a_size = (PyArrayObject *)PyArray_ContiguousFromObject(size, NPY_INTP, 1, 1);
        if (a_size == NULL) goto fail;
        // 如果 size 参数不符合要求，抛出异常
        if ((PyArray_NDIM(a_size) != 1) || (PyArray_DIMS(a_size)[0] < 2))
            PYERR("Size must be a length two sequence");
        // 从 a_size 中读取窗口大小
        Nwin[0] = ((npy_intp *)PyArray_DATA(a_size))[0];
        Nwin[1] = ((npy_intp *)PyArray_DATA(a_size))[1];
    }

    // 创建一个新的 PyArrayObject 对象用于存储输出结果，如果失败则跳转到 fail 标签
    a_out = (PyArrayObject *)PyArray_SimpleNew(2, PyArray_DIMS(a_image), typenum);
    if (a_out == NULL) goto fail;

    // 使用 setjmp 和 MALLOC_FAIL 宏来捕获内存分配错误
    if (setjmp(MALLOC_FAIL)) {
        // 发生内存分配错误时抛出异常
        PYERR("Memory allocation error.");
    }
    else {
        // 根据 typenum 类型选择不同的函数进行二维中值滤波
        switch (typenum) {
        case NPY_UBYTE:
            b_medfilt2((unsigned char *)PyArray_DATA(a_image),
                           (unsigned char *)PyArray_DATA(a_out),
                           Nwin, PyArray_DIMS(a_image));
            break;
        case NPY_FLOAT:
            f_medfilt2((float *)PyArray_DATA(a_image),
                           (float *)PyArray_DATA(a_out), Nwin,
                           PyArray_DIMS(a_image));
            break;
        case NPY_DOUBLE:
            d_medfilt2((double *)PyArray_DATA(a_image),
                           (double *)PyArray_DATA(a_out), Nwin,
                           PyArray_DIMS(a_image));
            break;
        default:
          // 如果 typenum 类型不支持，则抛出异常
          PYERR("2D median filter only supports uint8, float32, and float64.");
        }
    }

    // 释放 Python 对象引用
    Py_DECREF(a_image);
    // 释放 Python 对象引用
    Py_XDECREF(a_size);

    // 返回 PyArrayObject 对象作为函数执行结果
    return PyArray_Return(a_out);

 fail:
    // 释放 Python 对象引用
    Py_XDECREF(a_image);
    // 释放 Python 对象引用
    Py_XDECREF(a_size);
    // 释放 Python 对象引用
    Py_XDECREF(a_out);
    // 返回 NULL 指示函数执行失败
    return NULL;

}

static char doc_linear_filter[] =
    "(y,Vf) = _linear_filter(b,a,X,Dim=-1,Vi=None)  " \
    "implemented using Direct Form II transposed flow " \
    "diagram. If Vi is not given, Vf is not returned.";

// 定义一个 PyMethodDef 结构体数组，用于描述模块中的方法和文档字符串
static struct PyMethodDef toolbox_module_methods[] = {
    {"_correlateND", scipy_signal__sigtools_correlateND, METH_VARARGS, doc_correlateND},
    {"_convolve2d", _sigtools_convolve2d, METH_VARARGS, doc_convolve2d},
    {"_linear_filter", scipy_signal__sigtools_linear_filter, METH_VARARGS, doc_linear_filter},
    {"_remez", _sigtools_remez, METH_VARARGS, doc_remez},
    {"_medfilt2d", _sigtools_median2d, METH_VARARGS, doc_median2d},
    {NULL, NULL, 0, NULL}        /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,   // 定义 Python 模块的结构体，使用宏进行初始化
    "_sigtools",             // 模块名为 "_sigtools"
    NULL,                    // 模块文档字符串为空
    -1,                      // 模块状态为 -1，表示可选模块
    toolbox_module_methods,  // 模块方法的定义数组，指向 toolbox_module_methods
    NULL,                    // 模块的槽函数无需定义，置为 NULL
    NULL,                    // 模块的全局状态无需定义，置为 NULL
    NULL,                    // 模块的清理函数无需定义，置为 NULL
    NULL                     // 模块的内存分配函数无需定义，置为 NULL
};

PyMODINIT_FUNC
PyInit__sigtools(void)
{
    PyObject *module;   // 定义 Python 模块对象指针

    import_array();     // 导入 NumPy C API

    module = PyModule_Create(&moduledef);   // 创建 Python 模块对象
    if (module == NULL) {
        return NULL;    // 如果创建失败，返回 NULL
    }

    scipy_signal__sigtools_linear_filter_module_init();   // 初始化 scipy.signal._sigtools 模块

    return module;      // 返回创建的 Python 模块对象
}
```