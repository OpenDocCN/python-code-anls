# `.\numpy\numpy\linalg\lapack_lite\f2c.c`

```py
/*
  Functions here are copied from the source code for libf2c.

  Typically each function there is in its own file.

  We don't link against libf2c directly, because we can't guarantee
  it is available, and shipping a static library isn't portable.
*/

// 包含必要的头文件和宏定义
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 包含自定义的 Fortran 2 C 转换工具的头文件
#include "f2c.h"

// 定义外部链接的函数 s_wsfe
extern int s_wsfe(cilist *f) {return 0;}

// 定义外部链接的函数 e_wsfe
extern int e_wsfe(void) {return 0;}

// 定义外部链接的函数 do_fio
extern int do_fio(integer *c, char *s, ftnlen l) {return 0;}

/* You'll want this if you redo the f2c_*.c files with the -C option
 * to f2c for checking array subscripts. (It's not suggested you do that
 * for production use, of course.) */
// 定义数组下标检查函数 s_rnge
extern int
s_rnge(char *var, int index, char *routine, int lineno)
{
    // 输出错误信息至标准错误流
    fprintf(stderr, "array index out-of-bounds for %s[%d] in routine %s:%d\n",
            var, index, routine, lineno);
    // 刷新标准错误流
    fflush(stderr);
    // 中止程序
    abort();
}

#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 sqrtf 和 f__cabsf 函数
extern float sqrtf();
double f__cabsf(real, imag) float real, imag;
#else
// 否则取消 abs 宏定义，定义 f__cabsf 函数
#undef abs

double f__cabsf(float real, float imag)
#endif
{
    float temp;

    // 若实部小于 0，则取其相反数
    if(real < 0.0f)
        real = -real;
    // 若虚部小于 0，则取其相反数
    if(imag < 0.0f)
        imag = -imag;
    // 若虚部大于实部，则交换实部和虚部
    if(imag > real){
        temp = real;
        real = imag;
        imag = temp;
    }
    // 若虚部加实部等于实部，则返回实部的浮点数形式
    if((imag+real) == real)
        return((float)real);

    // 计算复数的模长并返回
    temp = imag/real;
    temp = real*sqrtf(1.0 + temp*temp);  /*overflow!!*/
    return(temp);
}


#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 sqrt 和 f__cabs 函数
extern double sqrt();
double f__cabs(real, imag) double real, imag;
#else
// 否则取消 abs 宏定义，定义 f__cabs 函数
#undef abs

double f__cabs(double real, double imag)
#endif
{
    double temp;

    // 若实部小于 0，则取其相反数
    if(real < 0)
        real = -real;
    // 若虚部小于 0，则取其相反数
    if(imag < 0)
        imag = -imag;
    // 若虚部大于实部，则交换实部和虚部
    if(imag > real){
        temp = real;
        real = imag;
        imag = temp;
    }
    // 若虚部加实部等于实部，则返回实部的双精度浮点数形式
    if((imag+real) == real)
        return((double)real);

    // 计算复数的模长并返回
    temp = imag/real;
    temp = real*sqrt(1.0 + temp*temp);  /*overflow!!*/
    return(temp);
}

// 定义 VOID 类型的函数 r_cnjg
VOID
#ifdef KR_headers
r_cnjg(r, z) singlecomplex *r, *z;
#else
r_cnjg(singlecomplex *r, singlecomplex *z)
#endif
{
    // 复数共轭操作：实部不变，虚部取反
    r->r = z->r;
    r->i = - z->i;
}

// 定义 VOID 类型的函数 d_cnjg
VOID
#ifdef KR_headers
d_cnjg(r, z) doublecomplex *r, *z;
#else
d_cnjg(doublecomplex *r, doublecomplex *z)
#endif
{
    // 复数共轭操作：实部不变，虚部取反
    r->r = z->r;
    r->i = - z->i;
}


#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 r_imag 函数
float r_imag(z) singlecomplex *z;
#else
// 否则取消 abs 宏定义，定义 r_imag 函数
float r_imag(singlecomplex *z)
#endif
{
    // 返回复数的虚部
    return(z->i);
}

#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 d_imag 函数
double d_imag(z) doublecomplex *z;
#else
// 否则取消 abs 宏定义，定义 d_imag 函数
double d_imag(doublecomplex *z)
#endif
{
    // 返回复数的虚部
    return(z->i);
}


#define log10e 0.43429448190325182765

#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 logf 和 r_lg10 函数
float logf();
float r_lg10(x) real *x;
#else
// 否则取消 abs 宏定义，定义 r_lg10 函数
#undef abs

float r_lg10(real *x)
#endif
{
    // 返回以 10 为底的对数值
    return( log10e * logf(*x) );
}

#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 log 和 d_lg10 函数
double log();
double d_lg10(x) doublereal *x;
#else
// 否则取消 abs 宏定义，定义 d_lg10 函数
#undef abs

double d_lg10(doublereal *x)
#endif
{
    // 返回以 10 为底的对数值
    return( log10e * log(*x) );
}

#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 r_sign 函数
double r_sign(a,b) real *a, *b;
#else
double r_sign(real *a, real *b)
#endif
{
    float x;
    // 取 a 的绝对值
    x = (*a >= 0.0f ? *a : - *a);
    // 根据 b 的正负号返回对应的值
    return( *b >= 0.0f ? x : -x);
}

#ifdef KR_headers
// 如果使用 K&R 风格的函数头，则定义 d_sign 函数
double d_sign(a,b) doublereal *a, *b;
#else
double d_sign(doublereal *a, doublereal *b)
#endif
{
    double x;

    // 取 a 的绝对值
    x = (*a >= 0.0 ? *a : - *a);
    // 根据 b 的正负号返回对应的值
    return( *b >= 0.0 ? x : -x);
}
#ifdef KR_headers
// 声明 floor 函数
double floor();
// 定义 i_dnnt 函数，参数为双精度浮点数指针 x
integer i_dnnt(x) doublereal *x;
#else
// 取消 abs 宏定义
#undef abs

// 定义 i_dnnt 函数，参数为双精度浮点数指针 x
integer i_dnnt(doublereal *x)
#endif
{
// 如果 *x 大于等于 0，则执行以下语句块，否则执行 else 语句块
return( (*x)>=0 ?
    // 返回 *x 加 0.5 后向下取整的结果
    floor(*x + .5) : -floor(.5 - *x) );
}


#ifdef KR_headers
// 声明 floor 函数
double floor();
// 定义 i_nint 函数，参数为实数指针 x
integer i_nint(x) real *x;
#else
// 取消 abs 宏定义
#undef abs
// 定义 i_nint 函数，参数为实数指针 x
integer i_nint(real *x)
#endif
{
// 返回 *x 大于等于 0 后，将 *x 加 0.5 后向下取整的结果，否则返回将 0.5 减去 *x 后向下取整的结果
return (integer)(*x >= 0 ? floor(*x + .5) : -floor(.5 - *x));
}

#ifdef KR_headers
// 声明 pow 函数
double pow();
// 定义 pow_dd 函数，参数为双精度浮点数指针 ap 和 bp
double pow_dd(ap, bp) doublereal *ap, *bp;
#else
// 取消 abs 宏定义

// 定义 pow_dd 函数，参数为双精度浮点数指针 ap 和 bp
double pow_dd(doublereal *ap, doublereal *bp)
#endif
{
// 返回 ap 的 bp 次方的结果
return(pow(*ap, *bp) );
}


#ifdef KR_headers
// 声明 pow 函数
double pow();
// 定义 pow_ri 函数，参数为实数指针 ap 和整数指针 bp
double pow_ri(ap, bp) real *ap; integer *bp;
#else
// 取消 abs 宏定义
double pow_ri(real *ap, integer *bp)
#endif
{
// 定义局部变量 pow、x、n、u
float pow, x;
integer n;
unsigned long u;

// 将 x 初始化为 *ap，将 n 初始化为 *bp，将 pow 初始化为 1
pow = 1;
x = *ap;
n = *bp;

// 如果 n 不等于 0，则执行以下语句块
if(n != 0)
    {
    // 如果 n 小于 0，则执行以下语句块
    if(n < 0)
        {
        // 将 n 取反，将 x 设为 1/x
        n = -n;
        x = 1.0f/x;
        }
    // 使用二进制形式遍历 n
    for(u = n; ; )
        {
        // 如果 u 的最低位为 1，则将 pow 乘以 x
        if(u & 01)
            pow *= x;
        // 右移 u 一位，同时将 x 自乘
        if(u >>= 1)
            x *= x;
        else
            break;
        }
    }
// 返回 pow
return(pow);
}

#ifdef KR_headers
// 声明 pow 函数
double pow();
// 定义 pow_di 函数，参数为双精度浮点数指针 ap 和整数指针 bp
double pow_di(ap, bp) doublereal *ap; integer *bp;
#else
// 取消 abs 宏定义
double pow_di(doublereal *ap, integer *bp)
#endif
{
// 定义局部变量 pow、x、n、u
double pow, x;
integer n;
unsigned long u;

// 将 x 初始化为 *ap，将 n 初始化为 *bp，将 pow 初始化为 1
pow = 1;
x = *ap;
n = *bp;

// 如果 n 不等于 0，则执行以下语句块
if(n != 0)
    {
    // 如果 n 小于 0，则执行以下语句块
    if(n < 0)
        {
        // 将 n 取反，将 x 设为 1/x
        n = -n;
        x = 1/x;
        }
    // 使用二进制形式遍历 n
    for(u = n; ; )
        {
        // 如果 u 的最低位为 1，则将 pow 乘以 x
        if(u & 01)
            pow *= x;
        // 右移 u 一位，同时将 x 自乘
        if(u >>= 1)
            x *= x;
        else
            break;
        }
    }
// 返回 pow
return(pow);
}

#ifdef KR_headers
// 声明 pow 函数
VOID pow_zi(p, a, b)     /* p = a**b  */
 doublecomplex *p, *a; integer *b;
#else
// 声明 pow_zi 函数
extern void pow_zi(doublecomplex*, doublecomplex*, integer*);
// 定义 pow_ci 函数，参数为单精度复数指针 p、a 和整数指针 b
void pow_ci(singlecomplex *p, singlecomplex *a, integer *b)     /* p = a**b  */
#endif
{
    // 定义局部变量 n、u、t、q、x
    integer n;
    unsigned long u;
    double t;
    doublecomplex q, x;
    // 初始化 one 为 {1.0, 0.0}
    static doublecomplex one = {1.0, 0.0};

    // 将 n 设为 *b，将 q 初始化为 {1, 0}
    n = *b;
    q.r = 1;
    q.i = 0;

    // 如果 n 等于 0，则跳转到 done 标签
    if(n == 0)
        goto done;
    // 如果 n 小于 0，则执行以下语句块
    if(n < 0)
        {
        // 将 n 取反，调用 z_div 函数，将结果存入 x
        n = -n;
        z_div(&x, &one, a);
        }
    // 否则执行以下语句块
    else
        {
        // 将 x 设为 a 的实部和虚部
        x.r = a->r;
        x.i = a->i;
        }

    // 使用二进制形式遍历 n
    for(u = n; ; )
        {
        // 如果 u 的最低位为 1，则执行以下语句块
        if(u & 01)
            {
            // 计算 t，更新 q 的实部和虚部
            t = q.r * x.r - q.i * x.i;
            q.i = q.r * x.i + q.i * x.r;
            q.r = t;
            }
        // 右移 u 一位，同时更新 x 的实部和虚部
        if(u >>= 1)
            {
            t = x.r * x.r - x.i * x.i;
            x.i = 2 * x.r * x.i;
            x.r = t;
            }
        else
            break;
        }
 done:
    // 将结果写入 p 的实部和虚部
    p->i = q.i;
    p->r = q.r;
    }

#ifdef KR_headers
// 声明 pow 函数
VOID pow_ci(p, a, b)     /* p = a**b  */
 singlecomplex *p, *a; integer *b;
#else
// 声明 pow_zi 函数
extern void pow_zi(doublecomplex*, doublecomplex*, integer*);
// 定义 pow_ci 函数，参数为单精度复数指针 p、a 和整数指针 b
void pow_ci(singlecomplex *p, singlecomplex *a, integer *b)     /* p = a**b  */
#endif
{
// 定义局部变量 p1、a1
doublecomplex p1, a1;

// 将 a1 设为 a 的实部和虚部
a1.r = a->r;
a1.i = a->i;

// 调用 pow_zi 函数，将结果存入 p1
pow_zi(&p1, &a1, b);

// 将 p 的实部和虚部设为 p1 的实部和虚部
p->r = p1.r;
p->i = p1.i;
}
/* Unless compiled with -DNO_OVERWRITE, this variant of s_cat allows the
 * target of a concatenation to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90).
 */
#ifndef NO_OVERWRITE
/* 如果没有定义NO_OVERWRITE，则允许s_cat的这个变体允许将连接的目标出现在右侧（与Fortran 77标准相反，但符合Fortran 90）。 */
#undef abs
#ifdef KR_headers
 extern char *F77_aloc();
 extern void free();
 extern void exit_();
#else
 extern char *F77_aloc(ftnlen, char*);
#endif
#endif /* NO_OVERWRITE */

#ifdef KR_headers
/* 使用K&R风格的函数头定义s_cat函数 */
int
s_cat(lp, rpp, rnp, np, ll) char *lp, *rpp[]; ftnlen rnp[], *np, ll;
#else
/* 使用ANSI风格的函数头定义s_cat函数 */
int
s_cat(char *lp, char *rpp[], ftnlen rnp[], ftnlen *np, ftnlen ll)
#endif
{
    ftnlen i, nc;
    char *rp;
    ftnlen n = *np;
#ifndef NO_OVERWRITE
    ftnlen L, m;
    char *lp0, *lp1;

    lp0 = 0;
    lp1 = lp;
    L = ll;
    i = 0;
    while(i < n) {
        rp = rpp[i];
        m = rnp[i++];
        /* 检查目标区域是否与源字符串区域没有重叠 */
        if (rp >= lp1 || rp + m <= lp) {
            if ((L -= m) <= 0) {
                n = i;
                break;
            }
            lp1 += m;
            continue;
        }
        /* 如果存在重叠，重新分配目标空间 */
        lp0 = lp;
        lp = lp1 = F77_aloc(L = ll, "s_cat");
        break;
    }
    lp1 = lp;
#endif /* NO_OVERWRITE */

    /* 执行字符串的连接操作 */
    for(i = 0 ; i < n ; ++i) {
        nc = ll;
        if(rnp[i] < nc)
            nc = rnp[i];
        ll -= nc;
        rp = rpp[i];
        while(--nc >= 0)
            *lp++ = *rp++;
    }

    /* 填充剩余目标区域为 ' ' */
    while(--ll >= 0)
        *lp++ = ' ';

#ifndef NO_OVERWRITE
    /* 如果曾经重新分配过目标空间，将数据移回原位置 */
    if (lp0) {
        memmove(lp0, lp1, L);
        free(lp1);
    }
#endif

    return 0;
}

/* compare two strings */

#ifdef KR_headers
/* 使用K&R风格的函数头定义s_cmp函数 */
integer s_cmp(a0, b0, la, lb) char *a0, *b0; ftnlen la, lb;
#else
/* 使用ANSI风格的函数头定义s_cmp函数 */
integer s_cmp(char *a0, char *b0, ftnlen la, ftnlen lb)
#endif
{
    register unsigned char *a, *aend, *b, *bend;
    a = (unsigned char *)a0;
    b = (unsigned char *)b0;
    aend = a + la;
    bend = b + lb;

    /* 比较两个字符串 */
    if(la <= lb) {
        while(a < aend)
            if(*a != *b)
                return( *a - *b );
            else
                { ++a; ++b; }

        while(b < bend)
            if(*b != ' ')
                return( ' ' - *b );
            else
                ++b;
    }
    else {
        while(b < bend)
            if(*a == *b)
                { ++a; ++b; }
            else
                return( *a - *b );
        while(a < aend)
            if(*a != ' ')
                return(*a - ' ');
            else
                ++a;
    }

    return(0);
}

/* Unless compiled with -DNO_OVERWRITE, this variant of s_copy allows the
 * target of an assignment to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90),
 * as in  a(2:5) = a(4:7) .
 */

/* assign strings:  a = b */

#ifdef KR_headers
/* 使用K&R风格的函数头定义s_copy函数 */
int s_copy(a, b, la, lb) register char *a, *b; ftnlen la, lb;
#else
/* 使用ANSI风格的函数头定义s_copy函数 */
int s_copy(register char *a, register char *b, ftnlen la, ftnlen lb)
#endif
{
    register char *aend, *bend;

    aend = a + la;

    /* 如果目标字符串长度小于等于源字符串长度，则执行赋值操作 */
    if(la <= lb)
#ifndef NO_OVERWRITE
        /* 如果没有定义NO_OVERWRITE，且目标区域与源字符串区域无重叠 */
        if (a <= b || a >= b + la)
#endif
            while(a < aend)
                *a++ = *b++;
#ifdef NO_OVERWRITE
        // 如果定义了 NO_OVERWRITE 宏，则执行以下代码块
#else
        // 否则执行以下代码块
        for(b += la; a < aend; )
            *--aend = *--b;
        // 将指针 b 的内容复制到指针 aend，直到指针 a 达到 aend
#endif

    else {
        // 如果条件不满足上面的 else 分支，则执行以下代码块
        bend = b + lb;
#ifndef NO_OVERWRITE
        // 如果未定义 NO_OVERWRITE 宏，则执行以下代码块
        if (a <= b || a >= bend)
#endif
            // 在 b 到 bend 之间复制数据到 a，直到 b 达到 bend
            while(b < bend)
                *a++ = *b++;
#ifndef NO_OVERWRITE
        // 如果定义了 NO_OVERWRITE 宏，则执行以下代码块
        else {
            // 将指针 a 后移 lb 个位置
            a += lb;
            // 将指针 b 和指针 bend 之间的数据反向复制到指针 a 和指针 a + lb 之间
            while(b < bend)
                *--a = *--bend;
            // 再次将指针 a 后移 lb 个位置
            a += lb;
            }
#endif
        // 将指针 a 到 aend 之间的数据设置为空格字符 ' '
        while(a < aend)
            *a++ = ' ';
        }
        // 返回值 0
        return 0;
    }


#ifdef KR_headers
double f__cabsf();
double c_abs(z) singlecomplex *z;
#else
double f__cabsf(float, float);
double c_abs(singlecomplex *z)
#endif
{
// 调用 f__cabsf 函数计算 z 的模
return( f__cabsf( z->r, z->i ) );
}

#ifdef KR_headers
double f__cabs();
double z_abs(z) doublecomplex *z;
#else
double f__cabs(double, double);
double z_abs(doublecomplex *z)
#endif
{
// 调用 f__cabs 函数计算 z 的模
return( f__cabs( z->r, z->i ) );
}


#ifdef KR_headers
extern void sig_die();
VOID c_div(c, a, b) singlecomplex *a, *b, *c;
#else
extern void sig_die(char*, int);
void c_div(singlecomplex *c, singlecomplex *a, singlecomplex *b)
#endif
{
float ratio, den;
float abr, abi;

// 如果 b 的实部小于 0，则取其相反数
if( (abr = b->r) < 0.f)
    abr = - abr;
// 如果 b 的虚部小于 0，则取其相反数
if( (abi = b->i) < 0.f)
    abi = - abi;
// 如果 b 的实部小于等于其虚部，执行以下代码块
if( abr <= abi )
    {
      /*Let IEEE Infinities handle this ;( */
      /*if(abi == 0)
        sig_die("complex division by zero", 1);*/
    // 计算 ratio 和 den
    ratio = b->r / b->i ;
    den = b->i * (1 + ratio*ratio);
    // 计算复数 c 的值
    c->r = (a->r*ratio + a->i) / den;
    c->i = (a->i*ratio - a->r) / den;
    }

else
    {
    // 计算 ratio 和 den
    ratio = b->i / b->r ;
    den = b->r * (1.f + ratio*ratio);
    // 计算复数 c 的值
    c->r = (a->r + a->i*ratio) / den;
    c->i = (a->i - a->r*ratio) / den;
    }

}

#ifdef KR_headers
extern void sig_die();
VOID z_div(c, a, b) doublecomplex *a, *b, *c;
#else
extern void sig_die(char*, int);
void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b)
#endif
{
double ratio, den;
double abr, abi;

// 如果 b 的实部小于 0，则取其相反数
if( (abr = b->r) < 0.)
    abr = - abr;
// 如果 b 的虚部小于 0，则取其相反数
if( (abi = b->i) < 0.)
    abi = - abi;
// 如果 b 的实部小于等于其虚部，执行以下代码块
if( abr <= abi )
    {
      /*Let IEEE Infinities handle this ;( */
      /*if(abi == 0)
        sig_die("complex division by zero", 1);*/
    // 计算 ratio 和 den
    ratio = b->r / b->i ;
    den = b->i * (1 + ratio*ratio);
    // 计算复数 c 的值
    c->r = (a->r*ratio + a->i) / den;
    c->i = (a->i*ratio - a->r) / den;
    }

else
    {
    // 计算 ratio 和 den
    ratio = b->i / b->r ;
    den = b->r * (1 + ratio*ratio);
    // 计算复数 c 的值
    c->r = (a->r + a->i*ratio) / den;
    c->i = (a->i - a->r*ratio) / den;
    }

}


#ifdef KR_headers
float sqrtf(), f__cabsf();
VOID c_sqrt(r, z) singlecomplex *r, *z;
#else
#undef abs

extern double f__cabsf(float, float);
void c_sqrt(singlecomplex *r, singlecomplex *z)
#endif
{
float mag;

// 计算复数 z 的模
if( (mag = f__cabsf(z->r, z->i)) == 0.f)
    // 如果 z 的模为 0，则 r 的实部和虚部都为 0
    r->r = r->i = 0.f;
else if(z->r > 0.0f)
    {
    // 如果 z 的实部大于 0，则计算 r 的实部和虚部
    r->r = sqrtf(0.5f * (mag + z->r) );
    r->i = z->i / r->r / 2.0f;
    }
else
    {
    // 如果 z 的实部不大于 0，则计算 r 的实部和虚部
    r->i = sqrtf(0.5f * (mag - z->r) );
    if(z->i < 0.0f)
        r->i = - r->i;
    r->r = z->i / r->i / 2.0f;
    }
}


#ifdef KR_headers
double sqrt(), f__cabs();
VOID z_sqrt(r, z) doublecomplex *r, *z;
#ifdef KR_headers
// 如果是 K&R 风格的头文件声明，则使用特定参数列表
integer pow_ii(ap, bp) integer *ap, *bp;
#else
// 否则，使用标准参数列表声明 pow_ii 函数
integer pow_ii(integer *ap, integer *bp)
#endif
{
    integer pow, x, n;
    unsigned long u;

    // 获取传入指针所指向的整数值
    x = *ap;
    n = *bp;

    // 处理指数为非正数的情况
    if (n <= 0) {
        // 指数为零或底数为 1 时返回 1
        if (n == 0 || x == 1)
            return 1;
        // 底数为 -1 时，根据情况返回结果
        if (x != -1)
            return x == 0 ? 1/x : 0;
        // 当底数为 -1 时，指数为负数的情况下，转为正数处理
        n = -n;
    }

    // 初始化幂结果为 1
    u = n;
    for(pow = 1; ; )
    {
        // 如果指数的当前位为 1，则乘以当前底数
        if(u & 01)
            pow *= x;
        // 右移一位，底数平方
        if(u >>= 1)
            x *= x;
        else
            break; // 当指数为 0 时跳出循环
    }
    // 返回计算结果
    return(pow);
}
#ifdef __cplusplus
// 如果是 C++ 编译环境，使用 extern "C" 包裹函数声明
extern "C" {
#endif

#ifdef KR_headers
// 如果是 K&R 风格的头文件声明，声明退出函数
extern void f_exit();
// 定义 s_stop 函数，接收字符指针和长度参数
VOID s_stop(s, n) char *s; ftnlen n;
#else
// 否则，声明 s_stop 函数，接收字符指针和长度参数
int s_stop(char *s, ftnlen n)
#endif
{
    int i;

    // 如果长度大于 0
    if(n > 0)
    {
        // 输出 "STOP " 到 stderr
        fprintf(stderr, "STOP ");
        // 输出字符串 s 的前 n 个字符到 stderr
        for(i = 0; i < n; ++i)
            putc(*s++, stderr);
        // 输出 " statement executed\n" 到 stderr
        fprintf(stderr, " statement executed\n");
    }
#ifdef NO_ONEXIT
// 如果定义了 NO_ONEXIT，调用 f_exit 函数
f_exit();
#endif
// 终止程序运行
exit(0);

/* We cannot avoid (useless) compiler diagnostics here:        */
/* some compilers complain if there is no return statement,    */
/* and others complain that this one cannot be reached.        */

return 0; /* NOT REACHED */
}
#ifdef __cplusplus
// 如果是 C++ 编译环境，使用 extern "C" 包裹函数实现结束
}
#endif
#ifdef __cplusplus
// 如果是 C++ 编译环境，使用 extern "C" 包裹整体实现结束
}
#endif
```