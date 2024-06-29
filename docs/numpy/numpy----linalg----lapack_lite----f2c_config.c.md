# `.\numpy\numpy\linalg\lapack_lite\f2c_config.c`

```py
/*
 * NOTE: This is generated code. Look in numpy/linalg/lapack_lite for
 *       information on remaking this file.
 */
#include "f2c.h"

#ifdef HAVE_CONFIG
#include "config.h"
#else
extern doublereal dlamch_(char *);
#define EPSILON dlamch_("Epsilon")
#define SAFEMINIMUM dlamch_("Safe minimum")
#define PRECISION dlamch_("Precision")
#define BASE dlamch_("Base")
#endif

extern doublereal dlapy2_(doublereal *x, doublereal *y);

/*
f2c knows the exact rules for precedence, and so omits parentheses where not
strictly necessary. Since this is generated code, we don't really care if
it's readable, and we know what is written is correct. So don't warn about
them.
*/
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

/* Table of constant values */

static integer c__1 = 1;
static doublereal c_b32 = 0.;
static real c_b66 = 0.f;

/*
   doublereal dlamch_(char *cmach)
   DLAMCH determines double precision machine parameters.
   Arguments
   =========
   CMACH   (input) CHARACTER*1
           Specifies the value to be returned by DLAMCH:
           = 'E' or 'e',   DLAMCH := eps
           = 'S' or 's ,   DLAMCH := sfmin
           = 'B' or 'b',   DLAMCH := base
           = 'P' or 'p',   DLAMCH := eps*base
           = 'N' or 'n',   DLAMCH := t
           = 'R' or 'r',   DLAMCH := rnd
           = 'M' or 'm',   DLAMCH := emin
           = 'U' or 'u',   DLAMCH := rmin
           = 'L' or 'l',   DLAMCH := emax
           = 'O' or 'o',   DLAMCH := rmax
           where
           eps   = relative machine precision
           sfmin = safe minimum, such that 1/sfmin does not overflow
           base  = base of the machine
           prec  = eps*base
           t     = number of (base) digits in the mantissa
           rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise
           emin  = minimum exponent before (gradual) underflow
           rmin  = underflow threshold - base**(emin-1)
           emax  = largest exponent before overflow
           rmax  = overflow threshold  - (base**emax)*(1-eps)
*/

doublereal dlamch_(char *cmach)
{
    /* Initialized data */
    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    doublereal ret_val;

    /* Local variables */
    static doublereal t;
    static integer it;
    static doublereal rnd, eps, base;
    static integer beta;
    static doublereal emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static doublereal rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static doublereal small, sfmin;
    extern /* Subroutine */ int dlamc2_(integer *, integer *, logical *,
        doublereal *, integer *, doublereal *, integer *, doublereal *);

    /*
       -- LAPACK auxiliary routine (version 3.2) --
          Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
          November 2006
       
       
       Purpose
       =======
       
       DLAMCH determines double precision machine parameters.
       
       Arguments
       =========
       
       CMACH   (input) CHARACTER*1
               Specifies the value to be returned by DLAMCH:
               = 'E' or 'e',   DLAMCH := eps
               = 'S' or 's ,   DLAMCH := sfmin
               = 'B' or 'b',   DLAMCH := base
               = 'P' or 'p',   DLAMCH := eps*base
               = 'N' or 'n',   DLAMCH := t
               = 'R' or 'r',   DLAMCH := rnd
               = 'M' or 'm',   DLAMCH := emin
               = 'U' or 'u',   DLAMCH := rmin
               = 'L' or 'l',   DLAMCH := emax
               = 'O' or 'o',   DLAMCH := rmax
       
               where
       
               eps   = relative machine precision
               sfmin = safe minimum, such that 1/sfmin does not overflow
               base  = base of the machine
               prec  = eps*base
               t     = number of (base) digits in the mantissa
               rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise
               emin  = minimum exponent before (gradual) underflow
               rmin  = underflow threshold - base**(emin-1)
               emax  = largest exponent before overflow
               rmax  = overflow threshold  - (base**emax)*(1-eps)
    */

    if (first) {
        /* Initialized data */

        /* Set first to FALSE to prevent reinitialization */
        first = FALSE_;
    }

    /* Here would be additional implementation details for DLAMCH function */

    /* Return the appropriate value based on CMACH */
    return ret_val;
}
    调用外部函数 dlamc2 来初始化一些变量，传入指针参数来获取返回值
    dlamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
    将 beta 转换为双精度浮点数赋值给变量 base
    base = (doublereal) beta;
    将 it 转换为双精度浮点数赋值给变量 t
    t = (doublereal) it;
    检查 lrnd 是否为真
    if (lrnd) {
        如果 lrnd 为真，设置 rnd 为 1.0
        rnd = 1.;
        计算 eps 的值，使用 pow_di 函数计算 base 的 (1 - it) 次幂，再除以 2
        i__1 = 1 - it;
        eps = pow_di(&base, &i__1) / 2;
    } else {
        如果 lrnd 不为真，设置 rnd 为 0.0
        rnd = 0.;
        计算 eps 的值，使用 pow_di 函数计算 base 的 (1 - it) 次幂
        i__1 = 1 - it;
        eps = pow_di(&base, &i__1);
    }
    计算 prec 的值，即 eps 乘以 base
    prec = eps * base;
    将 imin 转换为双精度浮点数赋值给变量 emin
    emin = (doublereal) imin;
    将 imax 转换为双精度浮点数赋值给变量 emax
    emax = (doublereal) imax;
    将 rmin 赋值给变量 sfmin
    sfmin = rmin;
    计算 small 的值，即 1.0 除以 rmax
    small = 1. / rmax;
    检查 small 是否大于或等于 sfmin
    if (small >= sfmin) {
/*
             Use SMALL plus a bit, to avoid the possibility of rounding
             causing overflow when computing  1/sfmin.
*/

        sfmin = small * (eps + 1.);
    }
    }

    if (lsame_(cmach, "E")) {
    rmach = eps;
    } else if (lsame_(cmach, "S")) {
    rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
    rmach = base;
    } else if (lsame_(cmach, "P")) {
    rmach = prec;
    } else if (lsame_(cmach, "N")) {
    rmach = t;
    } else if (lsame_(cmach, "R")) {
    rmach = rnd;
    } else if (lsame_(cmach, "M")) {
    rmach = emin;
    } else if (lsame_(cmach, "U")) {
    rmach = rmin;
    } else if (lsame_(cmach, "L")) {
    rmach = emax;
    } else if (lsame_(cmach, "O")) {
    rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of DLAMCH */

} /* dlamch_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc1_(integer *beta, integer *t, logical *rnd, logical
    *ieee1)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    doublereal d__1, d__2;

    /* Local variables */
    static doublereal a, b, c__, f, t1, t2;
    static integer lt;
    static doublereal one, qtr;
    static logical lrnd;
    static integer lbeta;
    static doublereal savec;
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static logical lieee1;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    DLAMC1 determines the machine parameters given by BETA, T, RND, and
    IEEE1.

    Arguments
    =========

    BETA    (output) INTEGER
            The base of the machine.

    T       (output) INTEGER
            The number of ( BETA ) digits in the mantissa.

    RND     (output) LOGICAL
            Specifies whether proper rounding  ( RND = .TRUE. )  or
            chopping  ( RND = .FALSE. )  occurs in addition. This may not
            be a reliable guide to the way in which the machine performs
            its arithmetic.

    IEEE1   (output) LOGICAL
            Specifies whether rounding appears to be done in the IEEE
            'round to nearest' style.

    Further Details
    ===============

    The routine is based on the routine  ENVRON  by Malcolm and
    incorporates suggestions by Gentleman and Marovich. See

       Malcolm M. A. (1972) Algorithms to reveal properties of
          floating-point arithmetic. Comms. of the ACM, 15, 949-951.

       Gentleman W. M. and Marovich S. B. (1974) More on algorithms
          that reveal properties of floating point arithmetic units.
          Comms. of the ACM, 17, 276-277.

   =====================================================================
*/


    if (first) {
    // 初次执行时，设置初始值
    one = 1.;
/*
          LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA,
          IEEE1, T and RND.

          Throughout this routine  we use the function  DLAMC3  to ensure
          that relevant values are  stored and not held in registers,  or
          are not affected by optimizers.

          Compute  a = 2.0**m  with the  smallest positive integer m such
          that

             fl( a + 1.0 ) = a.
*/

    a = 1.;
    c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
    if (c__ == one) {
        a *= 2;
        c__ = dlamc3_(&a, &one);  // Compute c__ = DLAMC3(a, 1.0)
        d__1 = -a;
        c__ = dlamc3_(&c__, &d__1);  // Compute c__ = DLAMC3(c__, -a)
        goto L10;  // Loop until c__ is not equal to 1.0
    }
/*
   +       END WHILE

          Now compute  b = 2.0**m  with the smallest positive integer m
          such that

             fl( a + b ) .gt. a.
*/

    b = 1.;
    c__ = dlamc3_(&a, &b);  // Compute c__ = DLAMC3(a, b)

/* +       WHILE( C.EQ.A )LOOP */
L20:
    if (c__ == a) {
        b *= 2;
        c__ = dlamc3_(&a, &b);  // Compute c__ = DLAMC3(a, b)
        goto L20;  // Loop until c__ is not equal to a
    }
/*
   +       END WHILE

          Now compute the base.  a and c  are neighbouring floating point
          numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so
          their difference is beta. Adding 0.25 to c is to ensure that it
          is truncated to beta and not ( beta - 1 ).
*/

    qtr = one / 4;
    savec = c__;  // Save the current value of c__
    d__1 = -a;
    c__ = dlamc3_(&c__, &d__1);  // Compute c__ = DLAMC3(c__, -a)
    lbeta = (integer) (c__ + qtr);  // Calculate lbeta as integer part of (c__ + 0.25)

/*
          Now determine whether rounding or chopping occurs,  by adding a
          bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a.
*/

    b = (doublereal) lbeta;
    d__1 = b / 2;
    d__2 = -b / 100;
    f = dlamc3_(&d__1, &d__2);  // Compute f = DLAMC3(b/2, -b/100)
    c__ = dlamc3_(&f, &a);  // Compute c__ = DLAMC3(f, a)
    if (c__ == a) {
        lrnd = TRUE_;
    } else {
        lrnd = FALSE_;
    }
    d__1 = b / 2;
    d__2 = b / 100;
    f = dlamc3_(&d__1, &d__2);  // Compute f = DLAMC3(b/2, b/100)
    c__ = dlamc3_(&f, &a);  // Compute c__ = DLAMC3(f, a)
    if (lrnd && c__ == a) {
        lrnd = FALSE_;
    }

/*
          Try and decide whether rounding is done in the  IEEE  'round to
          nearest' style. B/2 is half a unit in the last place of the two
          numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit
          zero, and SAVEC is odd. Thus adding B/2 to A should not  change
          A, but adding B/2 to SAVEC should change SAVEC.
*/

    d__1 = b / 2;
    t1 = dlamc3_(&d__1, &a);  // Compute t1 = DLAMC3(b/2, a)
    d__1 = b / 2;
    t2 = dlamc3_(&d__1, &savec);  // Compute t2 = DLAMC3(b/2, savec)
    lieee1 = t1 == a && t2 > savec && lrnd;

/*
          Now find  the  mantissa, t.  It should  be the  integer part of
          log to the base beta of a,  however it is safer to determine  t
          by powering.  So we find t as the smallest positive integer for
          which

             fl( beta**t + 1.0 ) = 1.0.
*/

    lt = 0;
    a = 1.;
    c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
    if (c__ == one) {
        ++lt;  // Increment lt
        a *= lbeta;  // Update a as a *= lbeta
        c__ = dlamc3_(&a, &one);  // Compute c__ = DLAMC3(a, 1.0)
        d__1 = -a;
        c__ = dlamc3_(&c__, &d__1);  // Compute c__ = DLAMC3(c__, -a)
        goto L30;  // Loop until c__ is not equal to 1.0
    }
/* +       END WHILE */

    }

    *beta = lbeta;  // Assign lbeta to the value pointed to by beta
    *t = lt;
    # 将指针变量 t 指向 lt 指针所指向的对象
    *rnd = lrnd;
    # 将指针变量 rnd 指向 lrnd 指针所指向的对象
    *ieee1 = lieee1;
    # 将指针变量 ieee1 指向 lieee1 指针所指向的对象
    first = FALSE_;
    # 将变量 first 设为 FALSE（假值）
    return 0;
    # 返回整数值 0
/*     End of DLAMC1 */

} /* dlamc1_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc2_(integer *beta, integer *t, logical *rnd,
    doublereal *eps, integer *emin, doublereal *rmin, integer *emax,
    doublereal *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;
    static logical iwarn = FALSE_;

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
        "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
        "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
        " the IF block as marked within the code of routine\002,\002 DLAM"
        "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Local variables */
    static doublereal a, b, c__;
    static integer i__, lt;
    static doublereal one, two;
    static logical ieee;
    static doublereal half;
    static logical lrnd;
    static doublereal leps, zero;
    static integer lbeta;
    static doublereal rbase;
    static integer lemin, lemax, gnmin;
    static doublereal small;
    static integer gpmin;
    static doublereal third, lrmin, lrmax, sixth;
    extern /* Subroutine */ int dlamc1_(integer *, integer *, logical *,
        logical *);
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static logical lieee1;
    extern /* Subroutine */ int dlamc4_(integer *, doublereal *, integer *),
        dlamc5_(integer *, integer *, integer *, logical *, integer *,
        doublereal *);
    static integer ngnmin, ngpmin;

    /* Fortran I/O blocks */
    static cilist io___58 = { 0, 6, 0, fmt_9999, 0 };


    /*
        -- LAPACK auxiliary routine (version 3.2) --
           Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
           November 2006


        Purpose
        =======

        DLAMC2 determines the machine parameters specified in its argument
        list.

        Arguments
        =========

        BETA    (output) INTEGER
                The base of the machine.

        T       (output) INTEGER
                The number of ( BETA ) digits in the mantissa.

        RND     (output) LOGICAL
                Specifies whether proper rounding  ( RND = .TRUE. )  or
                chopping  ( RND = .FALSE. )  occurs in addition. This may not
                be a reliable guide to the way in which the machine performs
                its arithmetic.

        EPS     (output) DOUBLE PRECISION
                The smallest positive number such that

                   fl( 1.0 - EPS ) .LT. 1.0,

                where fl denotes the computed value.

        EMIN    (output) INTEGER
                The minimum exponent before (gradual) underflow occurs.

        RMIN    (output) DOUBLE PRECISION
                The smallest normalized number for the machine, given by
                BASE**( EMIN - 1 ), where  BASE  is the floating point value
                of BETA.
    */
    EMAX    (output) INTEGER
            The maximum exponent before overflow occurs.
    # EMAX是一个输出参数，表示在发生溢出之前的最大指数值。

    RMAX    (output) DOUBLE PRECISION
            The largest positive number for the machine, given by
            BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point
            value of BETA.
    # RMAX是一个输出参数，表示机器能表示的最大正数，计算公式为 BASE**EMAX * ( 1 - EPS )，其中BASE是浮点数BETA的值。

    Further Details
    ===============
    # 进一步细节说明部分

    The computation of  EPS  is based on a routine PARANOIA by
    W. Kahan of the University of California at Berkeley.
    # EPS的计算基于加州大学伯克利分校的W. Kahan开发的PARANOIA例程。
    
   =====================================================================
   # 分隔线，标志着说明的结束或新段落的开始
    // 如果是第一次运行，则初始化一些常量
    if (first) {
        zero = 0.;
        one = 1.;
        two = 2.;
    }

    /*
          LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of
          BETA, T, RND, EPS, EMIN and RMIN.

          Throughout this routine  we use the function  DLAMC3  to ensure
          that relevant values are stored  and not held in registers,  or
          are not affected by optimizers.

          DLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1.
    */

    // 调用 DLAMC1 获取 LBETA, LT, LRND 和 LIEEE1 参数
    dlamc1_(&lbeta, &lt, &lrnd, &lieee1);

    /* Start to find EPS. */

    // 计算 a = BETA^(-LT)，其中 BETA 为基数
    b = (doublereal) lbeta;
    i__1 = -lt;
    a = pow_di(&b, &i__1);
    leps = a;

    /* Try some tricks to see whether or not this is the correct EPS. */

    // 尝试一些技巧来确认 leps 是否为正确的 EPS
    b = two / 3;
    half = one / 2;
    d__1 = -half;
    sixth = dlamc3_(&b, &d__1);
    third = dlamc3_(&sixth, &sixth);
    d__1 = -half;
    b = dlamc3_(&third, &d__1);
    b = dlamc3_(&b, &sixth);
    b = abs(b);
    if (b < leps) {
        b = leps;
    }

    leps = 1.;

    /* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
    // 使用迭代法计算 EPS
L10:
    if (leps > b && b > zero) {
        leps = b;
        d__1 = half * leps;
        /* Computing 5th power */
        d__3 = two, d__4 = d__3, d__3 *= d__3;
        /* Computing 2nd power */
        d__5 = leps;
        d__2 = d__4 * (d__3 * d__3) * (d__5 * d__5);
        c__ = dlamc3_(&d__1, &d__2);
        d__1 = -c__;
        c__ = dlamc3_(&half, &d__1);
        b = dlamc3_(&half, &c__);
        d__1 = -b;
        c__ = dlamc3_(&half, &d__1);
        b = dlamc3_(&half, &c__);
        goto L10;
    }
    /* +       END WHILE */

    if (a < leps) {
        leps = a;
    }

    /*
          Computation of EPS complete.

          Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)).
          Keep dividing  A by BETA until (gradual) underflow occurs. This
          is detected when we cannot recover the previous A.
    */

    // 计算 EMIN
    rbase = one / lbeta;
    small = one;
    for (i__ = 1; i__ <= 3; ++i__) {
        d__1 = small * rbase;
        small = dlamc3_(&d__1, &zero);
    }
    a = dlamc3_(&one, &small);
    dlamc4_(&ngpmin, &one, &lbeta);
    d__1 = -one;
    dlamc4_(&ngnmin, &d__1, &lbeta);
    dlamc4_(&gpmin, &a, &lbeta);
    d__1 = -a;
    dlamc4_(&gnmin, &d__1, &lbeta);
    ieee = FALSE_;

    if (ngpmin == ngnmin && gpmin == gnmin) {
        if (ngpmin == gpmin) {
            lemin = ngpmin;
            /*
              ( Non twos-complement machines, no gradual underflow;
                e.g.,  VAX )
            */
        } else if (gpmin - ngpmin == 3) {
            lemin = ngpmin - 1 + lt;
            ieee = TRUE_;
            /*
              ( Non twos-complement machines, with gradual underflow;
                e.g., IEEE standard followers )
            */
        } else {
            lemin = min(ngpmin, gpmin);
            /* ( A guess; no known machine ) */
            iwarn = TRUE_;
        }

    } else if (ngpmin == gpmin && ngnmin == gnmin) {
        if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
            lemin = max(ngpmin, ngnmin);

            lemin = max(ngpmin, ngnmin);
            /* Determine the minimum exponent value, EMIN. */
        } else {
            lemin = min(ngpmin, gpmin);
            /* A guess; no known machine */
            iwarn = TRUE_;
        }

    } else {
        // Inconsistent values found for exponents
        iwarn = TRUE_;
    }
/*
    ( Twos-complement machines, no gradual underflow;
      e.g., CYBER 205 )
*/
} else {
lemin = min(ngpmin,ngnmin);
/* ( A guess; no known machine ) */
iwarn = TRUE_;
}



} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
{
if (gpmin - min(ngpmin,ngnmin) == 3) {
lemin = max(ngpmin,ngnmin) - 1 + lt;
/*
    ( Twos-complement machines with gradual underflow;
      no known machine )
*/
} else {
lemin = min(ngpmin,ngnmin);
/* ( A guess; no known machine ) */
iwarn = TRUE_;
}



} else {
/* Computing MIN */
i__1 = min(ngpmin,ngnmin), i__1 = min(i__1,gpmin);
lemin = min(i__1,gnmin);
/* ( A guess; no known machine ) */
iwarn = TRUE_;
}
first = FALSE_;
/*
   **
   Comment out this if block if EMIN is ok
*/
if (iwarn) {
first = TRUE_;
s_wsfe(&io___58);
do_fio(&c__1, (char *)&lemin, (ftnlen)sizeof(integer));
e_wsfe();
}
/*
   **

Assume IEEE arithmetic if we found denormalised numbers above,
or if arithmetic seems to round in the IEEE style, determined
in routine DLAMC1. A true IEEE machine should have both things
true; however, faulty machines may have one or the other.
*/

ieee = ieee || lieee1;

/*
Compute RMIN by successive division by BETA. We could compute
RMIN as BASE**( EMIN - 1 ), but some machines underflow during
this computation.
*/

lrmin = 1.;
i__1 = 1 - lemin;
for (i__ = 1; i__ <= i__1; ++i__) {
d__1 = lrmin * rbase;
lrmin = dlamc3_(&d__1, &zero);
/* L30: */
}

/* Finally, call DLAMC5 to compute EMAX and RMAX. */

dlamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
}

*beta = lbeta;
*t = lt;
*rnd = lrnd;
*eps = leps;
*emin = lemin;
*rmin = lrmin;
*emax = lemax;
*rmax = lrmax;

return 0;

/* End of DLAMC2 */
} /* dlamc2_ */

/* *********************************************************************** */

doublereal dlamc3_(doublereal *a, doublereal *b)
{
/* System generated locals */
volatile doublereal ret_val;

/*
-- LAPACK auxiliary routine (version 3.2) --
Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
November 2006

Purpose
=======

DLAMC3 is intended to force A and B to be stored prior to doing
the addition of A and B, for use in situations where optimizers
might hold one of these in a register.

Arguments
=========

A (input) DOUBLE PRECISION
B (input) DOUBLE PRECISION
The values A and B.

=====================================================================
*/

ret_val = *a + *b;

return ret_val;

/* End of DLAMC3 */
} /* dlamc3_ */
/*
   ***********************************************************************
*/

/* Subroutine */ int dlamc4_(integer *emin, doublereal *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    static doublereal a;
    static integer i__;
    static doublereal b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern doublereal dlamc3_(doublereal *, doublereal *);

/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006

    Purpose
    =======

    DLAMC4 is a service routine for DLAMC2.

    Arguments
    =========

    EMIN    (output) INTEGER
            The minimum exponent before (gradual) underflow, computed by
            setting A = START and dividing by BASE until the previous A
            can not be recovered.

    START   (input) DOUBLE PRECISION
            The starting point for determining EMIN.

    BASE    (input) INTEGER
            The base of the machine.

   =====================================================================
*/

    a = *start;                     /* Initialize A to the starting point */
    one = 1.;                       /* Initialize ONE to 1.0 */
    rbase = one / *base;            /* Compute reciprocal of BASE */
    zero = 0.;                      /* Initialize ZERO to 0.0 */
    *emin = 1;                      /* Set initial value of EMIN to 1 */
    d__1 = a * rbase;               /* Compute A divided by BASE */
    b1 = dlamc3_(&d__1, &zero);     /* Call dlamc3 to compute B1 */
    c1 = a;                         /* Initialize C1 to A */
    c2 = a;                         /* Initialize C2 to A */
    d1 = a;                         /* Initialize D1 to A */
    d2 = a;                         /* Initialize D2 to A */

/*
   +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND.
      $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP
*/

L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
        --(*emin);                  /* Decrement EMIN */
        a = b1;                     /* Update A with B1 */
        d__1 = a / *base;           /* Compute A divided by BASE */
        b1 = dlamc3_(&d__1, &zero); /* Update B1 */
        d__1 = b1 * *base;          /* Compute B1 times BASE */
        c1 = dlamc3_(&d__1, &zero); /* Update C1 */
        d1 = zero;                  /* Reset D1 to zero */
        i__1 = *base;
        for (i__ = 1; i__ <= i__1; ++i__) {
            d1 += b1;               /* Accumulate B1 to D1 */
        }
        d__1 = a * rbase;           /* Compute A times reciprocal of BASE */
        b2 = dlamc3_(&d__1, &zero); /* Compute B2 */
        d__1 = b2 / rbase;          /* Compute B2 divided by reciprocal of BASE */
        c2 = dlamc3_(&d__1, &zero); /* Compute C2 */
        d2 = zero;                  /* Reset D2 to zero */
        i__1 = *base;
        for (i__ = 1; i__ <= i__1; ++i__) {
            d2 += b2;               /* Accumulate B2 to D2 */
        }
        goto L10;                   /* Repeat loop */
    }
/*
   +    END WHILE
*/

    return 0;

/*
     End of DLAMC4
*/

} /* dlamc4_ */


/*
   ***********************************************************************
*/

/* Subroutine */ int dlamc5_(integer *beta, integer *p, integer *emin,
    logical *ieee, integer *emax, doublereal *rmax)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    static integer i__;
    static doublereal y, z__;
    static integer try__, lexp;
    static doublereal oldy;
    static integer uexp, nbits;
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static doublereal recbas;
    static integer exbits, expsum;

/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006

    Purpose
    =======

    DLAMC5 attempts to compute RMAX, the largest machine floating-point
    number, without overflow.  It assumes that EMAX + abs(EMIN) sum
    approximately to a power of 2.  It will fail on machines where this
*/
    # 首先计算 LEXP 和 UEXP，它们是两个比 abs(EMIN) 大的 2 的幂次方。
    # 我们假设 EMAX + abs(EMIN) 的值大致等于接近 abs(EMIN) 的边界值。
    # 这里的 EMAX 是所需浮点数 RMAX 的指数部分。
    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
    lexp = try__;
    ++exbits;
    goto L10;
    }
    if (lexp == -(*emin)) {
    uexp = lexp;
    } else {
    uexp = try__;
    ++exbits;
    }

/*
       现在 -LEXP 小于或等于 EMIN，并且 -UEXP 大于或等于 EMIN。
       EXBITS 是存储指数所需的位数。
*/

    if (uexp + *emin > -lexp - *emin) {
    expsum = lexp << 1;
    } else {
    expsum = uexp << 1;
    }

/*
       EXPSUM 是指数范围，大约等于 EMAX - EMIN + 1。
*/

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*
       NBITS 是存储浮点数所需的总位数。
*/

    if (nbits % 2 == 1 && *beta == 2) {

/*
          如果存储浮点数需要奇数个位，这种情况不太可能；
          或者有些位在表示数字时没有使用，这种情况可能发生（如 Cray 机器）；
          或者尾数有一个隐式位（如 IEEE 机器、Dec Vax 机器），这种情况最可能。
          我们必须假设最后一种情况。
          如果是这样，那么我们需要减少 EMAX 一位，因为在隐式位系统中必须有一种表示零的方式。
          在像 Cray 这样的机器上，我们可能会不必要地减少 EMAX 一位。
*/

    --(*emax);
    }

    if (*ieee) {

/*
          假设我们在一个 IEEE 机器上，其中一个指数保留给无穷大和 NaN。
*/

    --(*emax);
    }

/*
       现在创建 RMAX，即最大的机器数，应该等于 (1.0 - BETA**(-P)) * BETA**EMAX。

       首先计算 1.0 - BETA**(-P)，确保结果小于 1.0。
*/

    recbas = 1. / *beta;
    z__ = *beta - 1.;
    y = 0.;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
    z__ *= recbas;
    if (y < 1.) {
        oldy = y;
    }
    y = dlamc3_(&y, &z__);
/* L20: */
    }
    if (y >= 1.) {
    y = oldy;
    }

/*     现在乘以 BETA**EMAX 得到 RMAX。 */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
    d__1 = y * *beta;
    y = dlamc3_(&d__1, &c_b32);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     DLAMC5 结束 */

} /* dlamc5_ */

logical lsame_(char *ca, char *cb)
{
    /* 系统生成的局部变量 */
    logical ret_val;

    /* 局部变量 */
    static integer inta, intb, zcode;

/*
    -- LAPACK 辅助例程（版本 3.2） --
       田纳西大学，加利福尼亚大学伯克利分校和 NAG Ltd.。
       2006 年 11 月

    目的
    =======

    LSAME 如果 CA 和 CB 是相同的字母（不区分大小写），则返回 .TRUE.。

    参数
    =========

    CA      （输入） CHARACTER*1
*/
    CB      (input) CHARACTER*1
            # CB 是一个输入参数，类型为字符型，长度为1，表示要比较的第二个字符。

            CA and CB specify the single characters to be compared.
            # CA 和 CB 指定了要比较的两个单个字符。

   =====================================================================


       Test if the characters are equal
       # 测试这两个字符是否相等
    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    // 检查两个字符的首字节是否相同，如果相同则返回真，否则返回假
    if (ret_val) {
    // 如果首字节相同，则直接返回真
    return ret_val;
    }

/*     Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

/*
       Use 'Z' rather than 'A' so that ASCII can be detected on Prime
       machines, on which ICHAR returns a value with bit 8 set.
       ICHAR('A') on Prime machines returns 193 which is the same as
       ICHAR('A') on an EBCDIC machine.
*/

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*
          ASCII is assumed - ZCODE is the ASCII code of either lower or
          upper case 'Z'.
*/

    if (inta >= 97 && inta <= 122) {
        inta += -32;
    }
    if (intb >= 97 && intb <= 122) {
        intb += -32;
    }

    } else if (zcode == 233 || zcode == 169) {

/*
          EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
          upper case 'Z'.
*/

    if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta
        >= 162 && inta <= 169) {
        inta += 64;
    }
    if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb
        >= 162 && intb <= 169) {
        intb += 64;
    }

    } else if (zcode == 218 || zcode == 250) {

/*
          ASCII is assumed, on Prime machines - ZCODE is the ASCII code
          plus 128 of either lower or upper case 'Z'.
*/

    if (inta >= 225 && inta <= 250) {
        inta += -32;
    }
    if (intb >= 225 && intb <= 250) {
        intb += -32;
    }
    }
    ret_val = inta == intb;

/*
       RETURN

       End of LSAME
*/

    return ret_val;
} /* lsame_ */

doublereal slamch_(char *cmach)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    real ret_val;

    /* Local variables */
    static real t;
    static integer it;
    static real rnd, eps, base;
    static integer beta;
    static real emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static real rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static real small, sfmin;
    extern /* Subroutine */ int slamc2_(integer *, integer *, logical *, real
        *, integer *, real *, integer *, real *);


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMCH determines single precision machine parameters.

    Arguments
    =========
    CMACH   (input) CHARACTER*1
            指定 SLAMCH 函数要返回的值：
            = 'E' or 'e',   SLAMCH := eps
            = 'S' or 's',   SLAMCH := sfmin
            = 'B' or 'b',   SLAMCH := base
            = 'P' or 'p',   SLAMCH := eps*base
            = 'N' or 'n',   SLAMCH := t
            = 'R' or 'r',   SLAMCH := rnd
            = 'M' or 'm',   SLAMCH := emin
            = 'U' or 'u',   SLAMCH := rmin
            = 'L' or 'l',   SLAMCH := emax
            = 'O' or 'o',   SLAMCH := rmax

            其中

            eps   = 相对机器精度
            sfmin = 安全最小值，使得 1/sfmin 不会溢出
            base  = 机器的基数
            prec  = eps*base
            t     = 小数部位的 (base) 数字个数
            rnd   = 在加法中发生舍入时为 1.0，否则为 0.0
            emin  = (渐进) 下溢之前的最小指数
            rmin  = 下溢阈值 - base**(emin-1)
            emax  = 溢出之前的最大指数
            rmax  = 溢出阈值 - (base**emax)*(1-eps)

   =====================================================================
/*

    if (first) {
/*
    ! 调用 SLAMC2 获取机器参数
*/
    slamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
/*
    ! 将 beta 转换为实数类型
*/
    base = (real) beta;
/*
    ! 将 it 转换为实数类型
*/
    t = (real) it;
/*
    ! 如果 lrnd 为真，设置 rnd 为 1.0，否则为 0.0
*/
    if (lrnd) {
        rnd = 1.f;
/*
    ! 计算 eps = base^(1-it)/2
*/
        i__1 = 1 - it;
        eps = pow_ri(&base, &i__1) / 2;
    } else {
        rnd = 0.f;
/*
    ! 计算 eps = base^(1-it)
*/
        i__1 = 1 - it;
        eps = pow_ri(&base, &i__1);
    }
/*
    ! 计算精度 prec = eps * base
*/
    prec = eps * base;
/*
    ! 将 imin 转换为实数类型
*/
    emin = (real) imin;
/*
    ! 将 imax 转换为实数类型
*/
    emax = (real) imax;
/*
    ! 初始化 sfmin 和 small
*/
    sfmin = rmin;
    small = 1.f / rmax;
/*
    ! 如果 small >= sfmin，使用 small 加上一点，以避免计算 1/sfmin 时发生溢出的可能性
*/
    if (small >= sfmin) {
/*
    ! sfmin = small * (eps + 1.0)
*/
        sfmin = small * (eps + 1.f);
    }
    }

/*
    ! 根据 cmach 参数选择相应的机器精度参数 rmach
*/
    if (lsame_(cmach, "E")) {
/*
    ! 如果 cmach 是 "E"，则 rmach = eps
*/
    rmach = eps;
    } else if (lsame_(cmach, "S")) {
/*
    ! 如果 cmach 是 "S"，则 rmach = sfmin
*/
    rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
/*
    ! 如果 cmach 是 "B"，则 rmach = base
*/
    rmach = base;
    } else if (lsame_(cmach, "P")) {
/*
    ! 如果 cmach 是 "P"，则 rmach = prec
*/
    rmach = prec;
    } else if (lsame_(cmach, "N")) {
/*
    ! 如果 cmach 是 "N"，则 rmach = t
*/
    rmach = t;
    } else if (lsame_(cmach, "R")) {
/*
    ! 如果 cmach 是 "R"，则 rmach = rnd
*/
    rmach = rnd;
    } else if (lsame_(cmach, "M")) {
/*
    ! 如果 cmach 是 "M"，则 rmach = emin
*/
    rmach = emin;
    } else if (lsame_(cmach, "U")) {
/*
    ! 如果 cmach 是 "U"，则 rmach = rmin
*/
    rmach = rmin;
    } else if (lsame_(cmach, "L")) {
/*
    ! 如果 cmach 是 "L"，则 rmach = emax
*/
    rmach = emax;
    } else if (lsame_(cmach, "O")) {
/*
    ! 如果 cmach 是 "O"，则 rmach = rmax
*/
    rmach = rmax;
    }

/*
    ! 返回计算得到的 rmach 值
*/
    ret_val = rmach;
/*
    ! 将 first 设置为 FALSE，表示第一次调用已经完成
*/
    first = FALSE_;
/*
    ! 返回 rmach 的值作为函数结果
*/
    return ret_val;

/*     End of SLAMCH */

} /* slamch_ */


/* *********************************************************************** */

/* Subroutine */ int slamc1_(integer *beta, integer *t, logical *rnd, logical
    *ieee1)
{
    /* Initialized data */

/*
    ! 静态变量，用于指示是否第一次调用该函数
*/
    static logical first = TRUE_;

    /* System generated locals */

/*
    ! 系统生成的本地变量
*/
    real r__1, r__2;

    /* Local variables */

/*
    ! 局部变量声明
*/
    static real a, b, c__, f, t1, t2;
    static integer lt;
    static real one, qtr;
    static logical lrnd;
    static integer lbeta;
    static real savec;
    static logical lieee1;

/*
    ! 外部函数声明
*/
    extern doublereal slamc3_(real *, real *);

/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006

/*
    ! 该子程序的目的和用法说明
*/
    Purpose
    =======

/*
    ! 确定机器参数 BETA, T, RND, 和 IEEE1

/*
    Arguments
    =========

    BETA    (output) INTEGER
            The base of the machine.

    T       (output) INTEGER
            The number of ( BETA ) digits in the mantissa.

    RND     (output) LOGICAL
            Specifies whether proper rounding  ( RND = .TRUE. )  or
            chopping  ( RND = .FALSE. )  occurs in addition. This may not
            be a reliable guide to the way in which the machine performs
            its arithmetic.

    IEEE1   (output) LOGICAL
            Specifies whether rounding appears to be done in the IEEE
            'round to nearest' style.

/*
    Further Details
    ===============

    ! 详细信息说明
*/
    The routine is based on the routine  ENVRON  by Malcolm and
    # 这部分注释是引用和参考文献列表，指出了对算法进行改进的建议来源及其文献引用。
    # Malcolm M. A. (1972) 的文章描述了揭示浮点算术属性的算法，发表于 ACM 的通信期刊，卷号为 15，页码为 949-951。
    # Gentleman W. M. 和 Marovich S. B. (1974) 的文章进一步讨论了揭示浮点算术单元属性的算法，发表于 ACM 的通信期刊，卷号为 17，页码为 276-277。
    # 这些引用为该代码或其算法的理论基础提供了支持和参考。
    =====================================================================
    // 如果是第一次执行，则将 one 设置为浮点数 1.0
    if (first) {
        one = 1.f;

        // LBETA, LIEEE1, LT 和 LRND 是 BETA、IEEE1、T 和 RND 的本地值。

        // 在整个例程中，使用 SLAMC3 函数确保相关值被存储，不放在寄存器中，或不受优化器影响。

        // 计算 a = 2.0**m，m 是最小的正整数，满足 fl( a + 1.0 ) = a。
        a = 1.f;
        c__ = 1.f;

        // 循环，直到 c__ 不等于 one
L10:
        if (c__ == one) {
            // 将 a 乘以 2
            a *= 2;
            // 使用 SLAMC3 函数更新 c__
            c__ = slamc3_(&a, &one);
            // 计算 -a，并更新 c__
            r__1 = -a;
            c__ = slamc3_(&c__, &r__1);
            // 继续循环
            goto L10;
        }
        // 结束 WHILE 循环

        // 计算 b = 2.0**m，m 是最小的正整数，满足 fl( a + b ) > a。
        b = 1.f;
        c__ = slamc3_(&a, &b);

        // 循环，直到 c__ 不等于 a
L20:
        if (c__ == a) {
            // 将 b 乘以 2
            b *= 2;
            // 使用 SLAMC3 函数更新 c__
            c__ = slamc3_(&a, &b);
            // 继续循环
            goto L20;
        }
        // 结束 WHILE 循环

        // 计算基数。a 和 c__ 是相邻的浮点数，位于区间 ( beta**t, beta**( t + 1 ) ) 内。
        // 将 0.25 添加到 c__ 是为了确保它被截断为 beta，而不是 ( beta - 1 )。
        qtr = one / 4;
        savec = c__;
        r__1 = -a;
        c__ = slamc3_(&c__, &r__1);
        lbeta = c__ + qtr;

        // 确定舍入还是截断，通过向 a 添加略小于 beta/2 和略大于 beta/2 的比特。
        b = (real) lbeta;
        r__1 = b / 2;
        r__2 = -b / 100;
        f = slamc3_(&r__1, &r__2);
        c__ = slamc3_(&f, &a);
        if (c__ == a) {
            lrnd = TRUE_;
        } else {
            lrnd = FALSE_;
        }
        r__1 = b / 2;
        r__2 = b / 100;
        f = slamc3_(&r__1, &r__2);
        c__ = slamc3_(&f, &a);
        if (lrnd && c__ == a) {
            lrnd = FALSE_;
        }

        // 尝试确定是否以 IEEE '最接近' 样式进行舍入。
        // B/2 是 A 和 SAVEC 中最后一位的一半单位。
        // 此外，A 是偶数（最后一位为零），而 SAVEC 是奇数。因此，向 A 添加 B/2 不应更改 A，
        // 但向 SAVEC 添加 B/2 应更改 SAVEC。
        r__1 = b / 2;
        t1 = slamc3_(&r__1, &a);
        r__1 = b / 2;
        t2 = slamc3_(&r__1, &savec);
        lieee1 = t1 == a && t2 > savec && lrnd;

        // 现在找到尾数 t。它应该是 log 以基数 beta 为底 a 的整数部分，但通过幂运算确定 t 更安全。
        // 因此，我们找到 t 作为最小的正整数，满足 fl( beta**t + 1.0 ) = 1.0。
        lt = 0;
        a = 1.f;
        c__ = 1.f;

        // 循环，直到 c__ 不等于 one
L30:
        if (c__ == one) {
            // 增加 lt 的计数
            ++lt;
            // 将 a 乘以 lbeta
            a *= lbeta;
            // 使用 SLAMC3 函数更新 c__
            c__ = slamc3_(&a, &one);
            // 计算 -a，并更新 c__
            r__1 = -a;
            c__ = slamc3_(&c__, &r__1);
            // 继续循环
            goto L30;
        }
        // 结束 WHILE 循环
    }
    # 将变量 lbeta 的值赋给指针 beta
    *beta = lbeta;
    # 将变量 lt 的值赋给指针 t
    *t = lt;
    # 将变量 lrnd 的值赋给指针 rnd
    *rnd = lrnd;
    # 将变量 lieee1 的值赋给指针 ieee1
    *ieee1 = lieee1;
    # 设置变量 first 的值为 FALSE_
    first = FALSE_;
    # 返回值 0 表示函数执行成功
    return 0;
/*     End of SLAMC1 */

} /* slamc1_ */


/* *********************************************************************** */

/* Subroutine */ int slamc2_(integer *beta, integer *t, logical *rnd, real *
    eps, integer *emin, real *rmin, integer *emax, real *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;  /* 静态变量，首次调用标志 */
    static logical iwarn = FALSE_; /* 警告标志 */

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
        "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
        "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
        " the IF block as marked within the code of routine\002,\002 SLAM"
        "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3, r__4, r__5;

    /* Local variables */
    static real a, b, c__;  /* 局部变量 */
    static integer i__, lt;  /* 局部变量 */
    static real one, two;   /* 局部变量 */
    static logical ieee;   /* 逻辑型变量，IEEE浮点数标志 */
    static real half;     /* 半 */
    static logical lrnd;  /* 逻辑型变量，指示舍入方式 */
    static real leps, zero;  /* 最小正浮点数，零 */
    static integer lbeta;   /* 机器数的基 */
    static real rbase;     /* 浮点数表示的基 */
    static integer lemin, lemax, gnmin;  /* 最小指数，最大指数，全局最小指数 */
    static real small;    /* 最小正数 */
    static integer gpmin;  /* 全局最小指数 */
    static real third, lrmin, lrmax, sixth;  /* 1/3, 最小正浮点数的倒数，最大浮点数 */
    static logical lieee1;  /* 逻辑型变量，IEEE浮点数标志 */
    extern /* Subroutine */ int slamc1_(integer *, integer *, logical *,
        logical *);  /* 外部子程序声明 */
    extern doublereal slamc3_(real *, real *);  /* 外部函数声明 */
    extern /* Subroutine */ int slamc4_(integer *, real *, integer *),
        slamc5_(integer *, integer *, integer *, logical *, integer *,
        real *);  /* 外部子程序声明 */
    static integer ngnmin, ngpmin;  /* 全局最小指数，全局最大指数 */

    /* Fortran I/O blocks */
    static cilist io___144 = { 0, 6, 0, fmt_9999, 0 };  /* Fortran I/O 控制块 */


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC2 determines the machine parameters specified in its argument
    list.

    Arguments
    =========

    BETA    (output) INTEGER
            The base of the machine.

    T       (output) INTEGER
            The number of ( BETA ) digits in the mantissa.

    RND     (output) LOGICAL
            Specifies whether proper rounding  ( RND = .TRUE. )  or
            chopping  ( RND = .FALSE. )  occurs in addition. This may not
            be a reliable guide to the way in which the machine performs
            its arithmetic.

    EPS     (output) REAL
            The smallest positive number such that

               fl( 1.0 - EPS ) .LT. 1.0,

            where fl denotes the computed value.

    EMIN    (output) INTEGER
            The minimum exponent before (gradual) underflow occurs.

    RMIN    (output) REAL
            The smallest normalized number for the machine, given by
            BASE**( EMIN - 1 ), where  BASE  is the floating point value
            of BETA.

    EMAX    (output) INTEGER
            The maximum exponent before overflow occurs.
*/
    RMAX    (output) REAL
            The largest positive number for the machine, given by
            BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point
            value of BETA.


RMAX：（输出）实数
        机器能表示的最大正数，由 BASE**EMAX * ( 1 - EPS ) 给出，其中 BASE 是浮点数 BETA 的值。



Further Details
===============


进一步细节
============



The computation of  EPS  is based on a routine PARANOIA by
W. Kahan of the University of California at Berkeley.

=====================================================================


EPS 的计算基于加州大学伯克利分校的 W. Kahan 编写的 PARANOIA 程序。
=====================================================================
    // 如果是第一次执行以下代码
    if (first) {
        // 初始化一些常量
        zero = 0.f;
        one = 1.f;
        two = 2.f;

        /*
          LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of
          BETA, T, RND, EPS, EMIN and RMIN.

          Throughout this routine  we use the function  SLAMC3  to ensure
          that relevant values are stored  and not held in registers,  or
          are not affected by optimizers.

          SLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1.
        */
        // 调用 SLAMC1 函数获取 LBETA, LT, LRND 和 LIEEE1 参数
        slamc1_(&lbeta, &lt, &lrnd, &lieee1);

        /* Start to find EPS. */
        // 计算 EPS 值
        b = (real) lbeta;
        i__1 = -lt;
        a = pow_ri(&b, &i__1);
        leps = a;

        /* Try some tricks to see whether or not this is the correct  EPS. */
        // 尝试一些技巧来确认 leps 是否是正确的 EPS 值
        b = two / 3;
        half = one / 2;
        r__1 = -half;
        sixth = slamc3_(&b, &r__1);
        third = slamc3_(&sixth, &sixth);
        r__1 = -half;
        b = slamc3_(&third, &r__1);
        b = slamc3_(&b, &sixth);
        b = dabs(b);
        if (b < leps) {
            b = leps;
        }

        leps = 1.f;

        /* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
        // 循环直到找到合适的 EPS 值
    L10:
        if (leps > b && b > zero) {
            leps = b;
            r__1 = half * leps;
            /* Computing 5th power */
            r__3 = two, r__4 = r__3, r__3 *= r__3;
            /* Computing 2nd power */
            r__5 = leps;
            r__2 = r__4 * (r__3 * r__3) * (r__5 * r__5);
            c__ = slamc3_(&r__1, &r__2);
            r__1 = -c__;
            c__ = slamc3_(&half, &r__1);
            b = slamc3_(&half, &c__);
            r__1 = -b;
            c__ = slamc3_(&half, &r__1);
            b = slamc3_(&half, &c__);
            goto L10;
        }
        /* +       END WHILE */

        if (a < leps) {
            leps = a;
        }

        /*
          Computation of EPS complete.

          Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)).
          Keep dividing  A by BETA until (gradual) underflow occurs. This
          is detected when we cannot recover the previous A.
        */
        // 计算 EPS 完成，现在计算 EMIN
        rbase = one / lbeta;
        small = one;
        for (i__ = 1; i__ <= 3; ++i__) {
            r__1 = small * rbase;
            small = slamc3_(&r__1, &zero);
            /* L20: */
        }
        a = slamc3_(&one, &small);
        slamc4_(&ngpmin, &one, &lbeta);
        r__1 = -one;
        slamc4_(&ngnmin, &r__1, &lbeta);
        slamc4_(&gpmin, &a, &lbeta);
        r__1 = -a;
        slamc4_(&gnmin, &r__1, &lbeta);
        ieee = FALSE_;

        if (ngpmin == ngnmin && gpmin == gnmin) {
            if (ngpmin == gpmin) {
                lemin = ngpmin;
                /*
                  ( Non twos-complement machines, no gradual underflow;
                    e.g.,  VAX )
                */
            } else if (gpmin - ngpmin == 3) {
                lemin = ngpmin - 1 + lt;
                ieee = TRUE_;
                /*
                  ( Non twos-complement machines, with gradual underflow;
                    e.g., IEEE standard followers )
                */
            } else {
                lemin = min(ngpmin,gpmin);
                /*
                   ( A guess; no known machine )
                */
                iwarn = TRUE_;
            }

        } else if (ngpmin == gpmin && ngnmin == gnmin) {
            if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
                lemin = max(ngpmin,ngnmin);

                /*
                   ( A guess; no known machine )
                */
                iwarn = TRUE_;
            }
        }
    }
/*
    ( Twos-complement machines, no gradual underflow;
      e.g., CYBER 205 )
*/
} else {
lemin = min(ngpmin,ngnmin);
/* ( A guess; no known machine ) */
iwarn = TRUE_;
}



/*
    ( A guess; no known machine )
*/
i__1 = min(ngpmin,ngnmin), i__1 = min(i__1,gpmin);
lemin = min(i__1,gnmin);
/* ( A guess; no known machine ) */
iwarn = TRUE_;



/*
    **
    Comment out this if block if EMIN is ok
*/
if (iwarn) {
first = TRUE_;
s_wsfe(&io___144);
do_fio(&c__1, (char *)&lemin, (ftnlen)sizeof(integer));
e_wsfe();
}
/*
    **

    Assume IEEE arithmetic if we found denormalised  numbers above,
    or if arithmetic seems to round in the  IEEE style,  determined
    in routine SLAMC1. A true IEEE machine should have both  things
    true; however, faulty machines may have one or the other.
*/

ieee = ieee || lieee1;

/*
    Compute  RMIN by successive division by  BETA. We could compute
    RMIN as BASE**( EMIN - 1 ),  but some machines underflow during
    this computation.
*/
lrmin = 1.f;
i__1 = 1 - lemin;
for (i__ = 1; i__ <= i__1; ++i__) {
r__1 = lrmin * rbase;
lrmin = slamc3_(&r__1, &zero);
/* L30: */
}

/* Finally, call SLAMC5 to compute EMAX and RMAX. */

slamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
}

*beta = lbeta;
*t = lt;
*rnd = lrnd;
*eps = leps;
*emin = lemin;
*rmin = lrmin;
*emax = lemax;
*rmax = lrmax;

return 0;


/* End of SLAMC2 */

} /* slamc2_ */


/* *********************************************************************** */

doublereal slamc3_(real *a, real *b)
{
/* System generated locals */
volatile real ret_val;


/*
-- LAPACK auxiliary routine (version 3.2) --
Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
November 2006


Purpose
=======

SLAMC3  is intended to force  A  and  B  to be stored prior to doing
the addition of  A  and  B ,  for use in situations where optimizers
might hold one of these in a register.

Arguments
=========

A       (input) REAL
B       (input) REAL
The values A and B.

=====================================================================
*/


ret_val = *a + *b;

return ret_val;

/* End of SLAMC3 */

} /* slamc3_ */


/* *********************************************************************** */
/* Subroutine */ int slamc4_(integer *emin, real *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    static real a;
    static integer i__;
    static real b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern doublereal slamc3_(real *, real *);


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC4 is a service routine for SLAMC2.

    Arguments
    =========

    EMIN    (output) INTEGER
            The minimum exponent before (gradual) underflow, computed by
            setting A = START and dividing by BASE until the previous A
            can not be recovered.

    START   (input) REAL
            The starting point for determining EMIN.

    BASE    (input) INTEGER
            The base of the machine.

   =====================================================================
*/


    a = *start;                                 /* Initialize 'a' to the input value of 'start' */
    one = 1.f;                                  /* Set 'one' to 1.0 as a single precision floating point number */
    rbase = one / *base;                        /* Calculate reciprocal of 'base' and assign to 'rbase' */
    zero = 0.f;                                 /* Set 'zero' to 0.0 as a single precision floating point number */
    *emin = 1;                                  /* Initialize 'emin' to 1 */
    r__1 = a * rbase;                           /* Compute a * rbase and store in 'r__1' */
    b1 = slamc3_(&r__1, &zero);                  /* Call external function slamc3 with arguments 'r__1' and 'zero', store result in 'b1' */
    c1 = a;                                      /* Set 'c1' to 'a' */
    c2 = a;                                      /* Set 'c2' to 'a' */
    d1 = a;                                      /* Set 'd1' to 'a' */
    d2 = a;                                      /* Set 'd2' to 'a' */
/*
   +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND.
      $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP
*/
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) { /* Start of loop checking if c1, c2, d1, and d2 are equal to 'a' */
    --(*emin);                                  /* Decrement 'emin' */
    a = b1;                                      /* Set 'a' to 'b1' */
    r__1 = a / *base;                           /* Compute a / base and store in 'r__1' */
    b1 = slamc3_(&r__1, &zero);                  /* Call external function slamc3 with arguments 'r__1' and 'zero', store result in 'b1' */
    r__1 = b1 * *base;                          /* Compute b1 * base and store in 'r__1' */
    c1 = slamc3_(&r__1, &zero);                  /* Call external function slamc3 with arguments 'r__1' and 'zero', store result in 'c1' */
    d1 = zero;                                  /* Set 'd1' to 0.0 */
    i__1 = *base;                               /* Set 'i__1' to 'base' */
    for (i__ = 1; i__ <= i__1; ++i__) {          /* Loop from 1 to 'base' */
        d1 += b1;                               /* Accumulate 'b1' into 'd1' */
/* L20: */
    }
    r__1 = a * rbase;                           /* Compute a * rbase and store in 'r__1' */
    b2 = slamc3_(&r__1, &zero);                  /* Call external function slamc3 with arguments 'r__1' and 'zero', store result in 'b2' */
    r__1 = b2 / rbase;                          /* Compute b2 / rbase and store in 'r__1' */
    c2 = slamc3_(&r__1, &zero);                  /* Call external function slamc3 with arguments 'r__1' and 'zero', store result in 'c2' */
    d2 = zero;                                  /* Set 'd2' to 0.0 */
    i__1 = *base;                               /* Set 'i__1' to 'base' */
    for (i__ = 1; i__ <= i__1; ++i__) {          /* Loop from 1 to 'base' */
        d2 += b2;                               /* Accumulate 'b2' into 'd2' */
/* L30: */
    }
    goto L10;                                   /* Return to start of loop */
    }
/* +    END WHILE */

    return 0;

/*     End of SLAMC4 */

} /* slamc4_ */


/* *********************************************************************** */

/* Subroutine */ int slamc5_(integer *beta, integer *p, integer *emin,
    logical *ieee, integer *emax, real *rmax)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    static integer i__;
    static real y, z__;
    static integer try__, lexp;
    static real oldy;
    static integer uexp, nbits;
    extern doublereal slamc3_(real *, real *);
    static real recbas;
    static integer exbits, expsum;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC5 attempts to compute RMAX, the largest machine floating-point
    number, without overflow.  It assumes that EMAX + abs(EMIN) sum
    approximately to a power of 2.  It will fail on machines where this
    assumption does not hold, for example, the Cyber 205 (EMIN = -28625,
    EMAX = 28718).  It will also fail if the value supplied for EMIN is
    # 计算机科学数学库中的常量和函数，用于确定浮点数的表示和计算特性
    
        INTEGER
                The base of floating-point arithmetic.
        INTEGER
                The number of base BETA digits in the mantissa of a
                floating-point value.
        INTEGER
                The minimum exponent before (gradual) underflow.
        LOGICAL
                A logical flag specifying whether or not the arithmetic
                system is thought to comply with the IEEE standard.
        INTEGER
                The largest exponent before overflow
        REAL
                The largest machine floating-point number.
       =====================================================================
    
           First compute LEXP and UEXP, two powers of 2 that bound
           abs(EMIN). We then assume that EMAX + abs(EMIN)
    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    // 如果尝试乘以 2 的 lexp 大于等于 -(*emin)，则更新 lexp 和 exbits
    if (try__ <= -(*emin)) {
        lexp = try__;
        ++exbits;
        goto L10;
    }
    // 如果 lexp 等于 -(*emin)，则设置 uexp 为 lexp，否则设置为 try__，并增加 exbits
    if (lexp == -(*emin)) {
        uexp = lexp;
    } else {
        uexp = try__;
        ++exbits;
    }

/*
       现在 -LEXP 小于等于 EMIN，-UEXP 大于等于 EMIN。EXBITS 是存储指数所需的位数。
*/

    // 根据 uexp 和 emin 计算 expsum，是指数范围的两倍
    if (uexp + *emin > -lexp - *emin) {
        expsum = lexp << 1;
    } else {
        expsum = uexp << 1;
    }

/*
       EXPSUM 是指数范围，大约等于 EMAX - EMIN + 1。
*/

    // 计算 emax，即指数范围的上限减去 1
    *emax = expsum + *emin - 1;
    // 计算 nbits，是存储浮点数所需的总位数
    nbits = exbits + 1 + *p;

/*
       NBITS 是存储浮点数所需的总位数。
*/

    // 如果 nbits 是奇数且 beta 等于 2，则根据特定情况调整 emax
    if (nbits % 2 == 1 && *beta == 2) {

/*
          可能的情况是存储浮点数位数是奇数，这在常规情况下较不可能，或者某些位在数值表示中没有使用（例如 Cray 机器），
          或者尾数有一个隐含位（例如 IEEE 机器，Dec Vax 机器），这是最有可能的情况。我们假设是最后一种情况。
          如果是这样，我们需要减少 EMAX，因为在隐含位系统中必须有一种表示零的方式。在像 Cray 这样的机器上，
          我们可能会不必要地减少 EMAX。
*/

        --(*emax);
    }

    // 如果 *ieee 为真，假设我们在一个 IEEE 标准的机器上，保留一个指数用于表示无穷大和 NaN
    if (*ieee) {

/*
          假设我们在一个 IEEE 标准的机器上，保留一个指数用于表示无穷大和 NaN。
*/

        --(*emax);
    }

/*
       现在创建 RMAX，即最大的机器数，应该等于 (1.0 - BETA**(-P)) * BETA**EMAX。

       首先计算 1.0 - BETA**(-P)，确保结果小于 1.0。
*/

    // 计算 recbas 为 beta 的倒数，z__ 为 beta - 1，y 初始化为 0
    recbas = 1.f / *beta;
    z__ = *beta - 1.f;
    y = 0.f;
    // 循环计算 z__ 的 *p 次幂，同时更新 y
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
        z__ *= recbas;
        if (y < 1.f) {
            oldy = y;
        }
        y = slamc3_(&y, &z__);
/* L20: */
    }
    // 如果 y 大于等于 1.0，则恢复为旧值 oldy
    if (y >= 1.f) {
        y = oldy;
    }

/*     现在乘以 BETA**EMAX 得到 RMAX。 */

    // 计算 y 乘以 beta 的 emax 次幂，得到最终的 RMAX
    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
        r__1 = y * *beta;
        y = slamc3_(&r__1, &c_b66);
/* L30: */
    }

    // 将计算得到的 RMAX 赋值给 *rmax
    *rmax = y;
    // 返回 0 表示成功完成 SLAMC5 函数
    return 0;

/*     SLAMC5 函数的结束 */

} /* slamc5_ */
```