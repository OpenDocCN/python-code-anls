# `.\numpy\numpy\linalg\lapack_lite\f2c_lapack.c`

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

static integer c__1 = 1;  // 定义整数常量 c__1 为 1
static real c_b172 = 0.f;  // 定义实数常量 c_b172 为 0.0f
static real c_b173 = 1.f;  // 定义实数常量 c_b173 为 1.0f
static integer c__0 = 0;   // 定义整数常量 c__0 为 0

integer ieeeck_(integer *ispec, real *zero, real *one)
{
    /* System generated locals */
    integer ret_val;  // 返回值

    /* Local variables */
    static real nan1, nan2, nan3, nan4, nan5, nan6, neginf, posinf, negzro,
        newzro;

/*
    -- LAPACK auxiliary routine (version 3.2.2) --
    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
       June 2010


    Purpose
    =======

    IEEECK is called from the ILAENV to verify that Infinity and
    possibly NaN arithmetic is safe (i.e. will not trap).

    Arguments
    =========

    ISPEC   (input) INTEGER
            Specifies whether to test just for inifinity arithmetic
            or whether to test for infinity and NaN arithmetic.
            = 0: Verify infinity arithmetic only.
            = 1: Verify infinity and NaN arithmetic.

    ZERO    (input) REAL
            Must contain the value 0.0
            This is passed to prevent the compiler from optimizing
            away this code.

    ONE     (input) REAL
            Must contain the value 1.0
            This is passed to prevent the compiler from optimizing
            away this code.

    RETURN VALUE:  INTEGER
            = 0:  Arithmetic failed to produce the correct answers
            = 1:  Arithmetic produced the correct answers
*/

    ret_val = 1;  // 默认返回值为 1，即算术操作正确

    posinf = *one / *zero;  // 计算正无穷
    if (posinf <= *one) {  // 检查正无穷是否小于等于 1
        ret_val = 0;  // 若不满足条件，设置返回值为 0
        return ret_val;  // 返回结果
    }

    neginf = -(*one) / *zero;  // 计算负无穷
    if (neginf >= *zero) {  // 检查负无穷是否大于等于 0
        ret_val = 0;  // 若不满足条件，设置返回值为 0
        return ret_val;  // 返回结果
    }

    negzro = *one / (neginf + *one);  // 计算负零
    if (negzro != *zero) {  // 检查负零是否等于零
        ret_val = 0;  // 若不满足条件，设置返回值为 0
        return ret_val;  // 返回结果
    }

    neginf = *one / negzro;  // 重新计算负无穷
    if (neginf >= *zero) {  // 检查负无穷是否大于等于零
        ret_val = 0;  // 若不满足条件，设置返回值为 0
        return ret_val;  // 返回结果
    }

    newzro = negzro + *zero;  // 计算新的零
    if (newzro != *zero) {  // 检查新的零是否等于零
        ret_val = 0;  // 若不满足条件，设置返回值为 0
        return ret_val;  // 返回结果
    }

    posinf = *one / newzro;  // 重新计算正无穷
    if (posinf <= *one) {  // 检查正无穷是否小于等于 1
        ret_val = 0;  // 若不满足条件，设置返回值为 0
        return ret_val;  // 返回结果
    }

    neginf *= posinf;  // 计算负无穷乘以正无穷
    if (neginf >= *zero) {  // 检查结果是否大于等于零
        ret_val = 0;  // 若不满足条件，设置返回值为 0
        return ret_val;  // 返回结果
    }

    posinf *= posinf;  // 计算正无穷的平方
    # 如果 posinf 小于等于指针 one 指向的值
    if (posinf <= *one) {
        # 将 ret_val 设为 0
        ret_val = 0;
        # 返回 ret_val
        return ret_val;
    }
/*     Return if we were only asked to check infinity arithmetic */
/* 如果 ispec 的值为 0，直接返回 ret_val */
if (*ispec == 0) {
return ret_val;
}

/* 计算几种 NaN (Not a Number) 值 */
nan1 = posinf + neginf; /* 正无穷加负无穷 */
nan2 = posinf / neginf; /* 正无穷除以负无穷 */
nan3 = posinf / posinf; /* 正无穷除以正无穷 */
nan4 = posinf * *zero; /* 正无穷乘以零 */
nan5 = neginf * negzro; /* 负无穷乘以零 */
nan6 = nan5 * *zero; /* NaN 值乘以零 */

/* 检查每种 NaN 值是否等于自身 */
if (nan1 == nan1) {
ret_val = 0;
return ret_val;
}

if (nan2 == nan2) {
ret_val = 0;
return ret_val;
}

if (nan3 == nan3) {
ret_val = 0;
return ret_val;
}

if (nan4 == nan4) {
ret_val = 0;
return ret_val;
}

if (nan5 == nan5) {
ret_val = 0;
return ret_val;
}

if (nan6 == nan6) {
ret_val = 0;
return ret_val;
}

return ret_val;
} /* ieeeck_ */

integer ilaclc_(integer *m, integer *n, singlecomplex *a, integer *lda)
{
/* System generated locals */
integer a_dim1, a_offset, ret_val, i__1, i__2;

/* Local variables */
static integer i__;

/*
-- LAPACK auxiliary routine (version 3.2.2)                        --

-- June 2010                                                       --

-- LAPACK is a software package provided by Univ. of Tennessee,    --
-- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--

Purpose
=======
ILACLC scans A for its last non-zero column.

Arguments
=========
M       (input) INTEGER
        The number of rows of the matrix A.

N       (input) INTEGER
        The number of columns of the matrix A.

A       (input) COMPLEX array, dimension (LDA,N)
        The m by n matrix A.

LDA     (input) INTEGER
        The leading dimension of the array A. LDA >= max(1,M).

=====================================================================

Quick test for the common case where one corner is non-zero.
*/
/* 参数调整 */
a_dim1 = *lda;
a_offset = 1 + a_dim1;
a -= a_offset;

/* 函数体 */
if (*n == 0) {
ret_val = *n;
} else /* if(complicated condition) */ {
i__1 = *n * a_dim1 + 1;
i__2 = *m + *n * a_dim1;
if (a[i__1].r != 0.f || a[i__1].i != 0.f || (a[i__2].r != 0.f || a[
i__2].i != 0.f)) {
ret_val = *n;
} else {
/* 从后向前扫描每列，返回第一个非零列 */
for (ret_val = *n; ret_val >= 1; --ret_val) {
i__1 = *m;
for (i__ = 1; i__ <= i__1; ++i__) {
i__2 = i__ + ret_val * a_dim1;
if (a[i__2].r != 0.f || a[i__2].i != 0.f) {
return ret_val;
}
}
}
}
}
return ret_val;
} /* ilaclc_ */

integer ilaclr_(integer *m, integer *n, singlecomplex *a, integer *lda)
{
/* System generated locals */
integer a_dim1, a_offset, ret_val, i__1, i__2;

/* Local variables */
static integer i__, j;

/*
-- LAPACK auxiliary routine (version 3.2.2)                        --
    -- June 2010                                                       --

    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--

    -- June 2010                                                       --
    -- 2010年6月                                                       --
    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    -- LAPACK是由田纳西大学、加利福尼亚大学伯克利分校、科罗拉多大学丹佛分校和NAG有限公司提供的软件包。--

    Purpose
    =======
    -- 目的
    -- ILACLR scans A for its last non-zero row.
    -- ILACLR扫描矩阵A找到其最后一个非零行。

    Arguments
    =========
    -- 参数说明

    M       (input) INTEGER
            -- 输入，整数
            The number of rows of the matrix A.
            -- 矩阵A的行数。

    N       (input) INTEGER
            -- 输入，整数
            The number of columns of the matrix A.
            -- 矩阵A的列数。

    A       (input) COMPLEX          array, dimension (LDA,N)
            -- 输入，复数数组，维度为(LDA,N)
            The m by n matrix A.
            -- m行n列的矩阵A。

    LDA     (input) INTEGER
            -- 输入，整数
            The leading dimension of the array A. LDA >= max(1,M).
            -- 数组A的主维度。要求 LDA >= max(1,M)。

    =====================================================================
    -- ====================================================================
    

       Quick test for the common case where one corner is non-zero.
       -- 快速测试常见情况，其中一个角落非零。
/*
    -- LAPACK auxiliary routine (version 3.2.2)                        --

    -- June 2010                                                       --

    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--

    Purpose
    =======

    ILADLR scans A for its last non-zero row.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.

    N       (input) INTEGER
            The number of columns of the matrix A.

    A       (input) DOUBLE PRECISION array, dimension (LDA,N)
            The m by n matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    =====================================================================
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    if (*m == 0) {
        ret_val = *m;
    } else /* if(complicated condition) */ {
        i__1 = *m + a_dim1;
        i__2 = *m + *n * a_dim1;
        if (a[i__1].r != 0.f || a[i__1].i != 0.f || (a[i__2].r != 0.f || a[i__2].i != 0.f)) {
            ret_val = *m;
        } else {
/*     Scan up each column tracking the last zero row seen. */
            ret_val = 0;
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                for (i__ = *m; i__ >= 1; --i__) {
                    i__2 = i__ + j * a_dim1;
                    if (a[i__2].r != 0.f || a[i__2].i != 0.f) {
                        goto L10;
                    }
                }
L10:
                ret_val = max(ret_val,i__);
            }
        }
    }
    return ret_val;
} /* ilaclr_ */

integer iladlc_(integer *m, integer *n, doublereal *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, ret_val, i__1;

    /* Local variables */
    static integer i__;

/*
    -- LAPACK auxiliary routine (version 3.2.2)                        --

    -- June 2010                                                       --

    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--

    Purpose
    =======

    ILADLC scans A for its last non-zero column.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.

    N       (input) INTEGER
            The number of columns of the matrix A.

    A       (input) DOUBLE PRECISION array, dimension (LDA,N)
            The m by n matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    =====================================================================
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    if (*n == 0) {
        ret_val = *n;
    } else if (a[*n * a_dim1 + 1] != 0. || a[*m + *n * a_dim1] != 0.) {
        ret_val = *n;
    } else {
/*     Now scan each column from the end, returning with the first non-zero. */
        for (ret_val = *n; ret_val >= 1; --ret_val) {
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                if (a[i__ + ret_val * a_dim1] != 0.) {
                    return ret_val;
                }
            }
        }
    }
    return ret_val;
} /* iladlc_ */
    # ILADLR 函数扫描矩阵 A 的最后一个非零行。
    #
    # Arguments:
    # M     (input) INTEGER
    #       矩阵 A 的行数。
    #
    # N     (input) INTEGER
    #       矩阵 A 的列数。
    #
    # A     (input) DOUBLE PRECISION 数组，维度为 (LDA,N)
    #       m 行 n 列的矩阵 A。
    #
    # LDA   (input) INTEGER
    #       数组 A 的主维度。LDA >= max(1,M)。
    #
    # =====================================================================
    #
    # Quick test for the common case where one corner is non-zero.
    #
    /* System generated locals */
    integer ret_val;

    /* Local variables */
    static integer i__;
    static char c1[1], c2[2], c3[3], c4[2];
    static integer ic, nb, iz, nx;
    static logical cname;
    static integer nbmin;
    static logical sname;
    extern integer ieeeck_(integer *, real *, real *);
    static char subnam[6];
    extern integer iparmq_(integer *, char *, char *, integer *, integer *,
        integer *, integer *, ftnlen, ftnlen);

/*
    -- LAPACK auxiliary routine (version 3.2.1)                        --

    -- April 2009                                                      --

    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--

    Purpose
    =======

    ILAENV is called from the LAPACK routines to choose problem-dependent
    parameters for the local environment.  See ISPEC for a description of
    the parameters.

    ILAENV returns an INTEGER
    if ILAENV >= 0: ILAENV returns the value of the parameter specified by ISPEC
    if ILAENV < 0:  if ILAENV = -k, the k-th argument had an illegal value.

    This version provides a set of parameters which should give good,
    but not optimal, performance on many of the currently available
    computers.  Users are encouraged to modify this subroutine to set
    the tuning parameters for their particular machine using the option
    and problem size information in the arguments.

    This routine will not function correctly if it is converted to all
    lower case.  Converting it to all upper case is allowed.

    Arguments
    =========

    ISPEC   (input) INTEGER
            Specifies the parameter to be returned as the value of ILAENV.

    NAME    (input) CHAR*1
            The name of the calling subroutine.

    OPTS    (input) CHAR*1
            The character options to the subroutine NAME.

    N1, N2, N3, N4 (input) INTEGER
            Problem dimensions for the subroutine NAME.

    NAME_LEN (input) INTEGER
            The length of the NAME parameter.

    OPTS_LEN (input) INTEGER
            The length of the OPTS parameter.

*/
    ! ISPEC (输入) INTEGER
    ! ILAENV 的返回值参数，指定要返回的值。
    ! = 1: 最优块大小；如果此值为1，则未阻塞算法将提供最佳性能。
    ! = 2: 应使用块例程的最小块大小；如果可用的块大小小于此值，则应使用未阻塞例程。
    ! = 3: 交叉点（在块例程中，对于 N 小于此值，应使用未阻塞例程）。
    ! = 4: 在非对称特征值例程中使用的位移数（已弃用）。
    ! = 5: 应使用块处理的最小列维度；矩形块的维度至少为 k x m，其中 k 由 ILAENV(2,...) 给出，m 由 ILAENV(5,...) 给出。
    ! = 6: SVD 的交叉点（当将 m x n 矩阵减少为二对角形式时，如果 max(m,n)/min(m,n) 超过此值，则首先使用 QR 分解将矩阵减少为三角形式）。
    ! = 7: 处理器的数量。
    ! = 8: 用于非对称特征值问题的多位移 QR 方法的交叉点（已弃用）。
    ! = 9: 在分治算法的计算树底部的子问题的最大大小（由 xGELSD 和 xGESDD 使用）。
    ! =10: 可信任 ieee NaN 算术不会陷阱。
    ! =11: 可信任的无穷大算术不会陷阱。
    ! 12 <= ISPEC <= 16:
    !      xHSEQR 或其子程序之一，详细解释请参见 IPARMQ。

    ! NAME (输入) CHARACTER*(*)
    ! 调用子例程的名称，可以是大写或小写。

    ! OPTS (输入) CHARACTER*(*)
    ! 传递给子例程 NAME 的选项字符，连接成单个字符字符串。
    ! 例如，对于三角例程，UPLO = 'U'，TRANS = 'T'，DIAG = 'N'，可以指定 OPTS = 'UTN'。

    ! N1 (输入) INTEGER
    ! N2 (输入) INTEGER
    ! N3 (输入) INTEGER
    ! N4 (输入) INTEGER
    ! 子例程 NAME 的问题维度；可能不是全部都需要。

    ! 进一步细节
    ! ===============
    ! 在调用 ILAENV 从 LAPACK 例程中使用以下约定：
    ! 1）OPTS 是子例程 NAME 的所有字符选项的连接，按照它们在 NAME 的参数列表中出现的顺序排列，即使它们在确定 ISPEC 指定的参数值时未被使用。
    2)  The problem dimensions N1, N2, N3, N4 are specified in the order
        that they appear in the argument list for NAME.  N1 is used
        first, N2 second, and so on, and unused problem dimensions are
        passed a value of -1.
    3)  The parameter value returned by ILAENV is checked for validity in
        the calling subroutine.  For example, ILAENV is used to retrieve
        the optimal blocksize for STRTRI as follows:

        NB = ILAENV( 1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1 )
        IF( NB.LE.1 ) NB = MAX( 1, N )

    =====================================================================


注释：


# 第2条：指定的问题维度 N1、N2、N3、N4 按照它们在 NAME 的参数列表中的顺序排列。
# N1 最先被使用，接着是 N2，以此类推。未使用的问题维度被赋值为 -1。
# 
# 第3条：ILAENV 返回的参数值在调用子程序中被检查其有效性。
# 例如，ILAENV 被用于获取 STRTRI 的最优块大小，操作如下：
# 
# NB = ILAENV( 1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1 )
# 如果 NB 小于等于 1，则将 NB 设置为 1 和 N 中的最大值。
#
# =====================================================================
    switch (*ispec) {
    case 1:  goto L10;  // 如果ispec等于1，则跳转到标签L10处继续执行
    case 2:  goto L10;  // 如果ispec等于2，则跳转到标签L10处继续执行
    case 3:  goto L10;  // 如果ispec等于3，则跳转到标签L10处继续执行
    case 4:  goto L80;  // 如果ispec等于4，则跳转到标签L80处继续执行
    case 5:  goto L90;  // 如果ispec等于5，则跳转到标签L90处继续执行
    case 6:  goto L100; // 如果ispec等于6，则跳转到标签L100处继续执行
    case 7:  goto L110; // 如果ispec等于7，则跳转到标签L110处继续执行
    case 8:  goto L120; // 如果ispec等于8，则跳转到标签L120处继续执行
    case 9:  goto L130; // 如果ispec等于9，则跳转到标签L130处继续执行
    case 10:  goto L140; // 如果ispec等于10，则跳转到标签L140处继续执行
    case 11:  goto L150; // 如果ispec等于11，则跳转到标签L150处继续执行
    case 12:  goto L160; // 如果ispec等于12，则跳转到标签L160处继续执行
    case 13:  goto L160; // 如果ispec等于13，则跳转到标签L160处继续执行
    case 14:  goto L160; // 如果ispec等于14，则跳转到标签L160处继续执行
    case 15:  goto L160; // 如果ispec等于15，则跳转到标签L160处继续执行
    case 16:  goto L160; // 如果ispec等于16，则跳转到标签L160处继续执行
    }

/*     ISPEC的值无效 */

    ret_val = -1;  // 将返回值设为-1
    return ret_val;  // 返回ret_val作为函数的结果

L10:

/*     如果名称的第一个字符是小写，则将名称转换为大写 */

    ret_val = 1;  // 将返回值设为1
    s_copy(subnam, name__, (ftnlen)6, name_len);  // 将name__复制到subnam中，长度为6
    ic = *(unsigned char *)subnam;  // 获取subnam的第一个字符的ASCII值
    iz = 'Z';  // 设置iz为字符'Z'
    if (iz == 90 || iz == 122) {

/*        ASCII字符集 */

    if (ic >= 97 && ic <= 122) {  // 如果第一个字符是小写字母
        *(unsigned char *)subnam = (char) (ic - 32);  // 将第一个字符转换为大写
        for (i__ = 2; i__ <= 6; ++i__) {  // 循环处理subnam的后续字符
        ic = *(unsigned char *)&subnam[i__ - 1];  // 获取当前字符的ASCII值
        if (ic >= 97 && ic <= 122) {  // 如果当前字符是小写字母
            *(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);  // 将其转换为大写
        }
/* L20: */
        }
    }

    } else if (iz == 233 || iz == 169) {

/*        EBCDIC字符集 */

    if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >= 162 &&
        ic <= 169) {  // 如果是EBCDIC字符集中的特定字符范围
        *(unsigned char *)subnam = (char) (ic + 64);  // 转换字符为大写
        for (i__ = 2; i__ <= 6; ++i__) {  // 循环处理subnam的后续字符
        ic = *(unsigned char *)&subnam[i__ - 1];  // 获取当前字符的ASCII值
        if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >=
            162 && ic <= 169) {  // 如果是EBCDIC字符集中的特定字符范围
            *(unsigned char *)&subnam[i__ - 1] = (char) (ic + 64);  // 转换字符为大写
        }
/* L30: */
        }
    }

    } else if (iz == 218 || iz == 250) {

/*        Prime机器: ASCII+128 */

    if (ic >= 225 && ic <= 250) {  // 如果是Prime机器的ASCII字符范围
        *(unsigned char *)subnam = (char) (ic - 32);  // 转换字符为大写
        for (i__ = 2; i__ <= 6; ++i__) {  // 循环处理subnam的后续字符
        ic = *(unsigned char *)&subnam[i__ - 1];  // 获取当前字符的ASCII值
        if (ic >= 225 && ic <= 250) {  // 如果是Prime机器的ASCII字符范围
            *(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);  // 转换字符为大写
        }
/* L40: */
        }
    }
    }

    *(unsigned char *)c1 = *(unsigned char *)subnam;  // 将subnam的第一个字符赋给c1
    sname = *(unsigned char *)c1 == 'S' || *(unsigned char *)c1 == 'D';  // 判断第一个字符是否为'S'或'D'
    cname = *(unsigned char *)c1 == 'C' || *(unsigned char *)c1 == 'Z';  // 判断第一个字符是否为'C'或'Z'
    if (! (cname || sname)) {  // 如果不是以'C'或'Z'开头，也不是以'S'或'D'开头
    return ret_val;  // 返回ret_val作为函数的结果
    }
    s_copy(c2, subnam + 1, (ftnlen)2, (ftnlen)2);  // 将subnam的第二到第三个字符复制到c2
    s_copy(c3, subnam + 3, (ftnlen)3, (ftnlen)3);  // 将subnam的第四到第六个字符复制到c3
    s_copy(c4, c3 + 1, (ftnlen)2, (ftnlen)2);  // 将c3的第二到第三个字符复制到c4

    switch (*ispec) {
    case 1:  goto L50;  // 如果ispec等于1，则跳转到标签L50处继续执行
    case 2:  goto L60;  // 如果ispec等于2，则跳转到标签L60处继续执行
    case 3:  goto L70;  // 如果ispec等于3，则跳转到标签L70处继续执行
    }

L50:

/*
       ISPEC = 1:  块大小

       在这些例子中，提供了设置实数和复数NB的单独代码。
       我们假设NB在单精度和双精度中取相同的值。
*/

    nb = 1;  // 将nb设置为1

    if (s_cmp(c2, "GE", (ftnlen)2, (ftnlen)2) == 0) {  // 如果c2等于"GE"
    if (s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {  // 如果c3等于"TRF"
        if (sname) {  // 如果名称以'S'或'D'开头
        nb = 64;  // 设置nb为64
        } else {
        nb = 64;  // 设置nb为64
        }
        }
    }
    } else if (s_cmp(c3, "QRF", (ftnlen)3, (ftnlen)3) == 0 || s_cmp(c3,
        "RQF", (ftnlen)3, (ftnlen)3) == 0 || s_cmp(c3, "LQF", (ftnlen)
        3, (ftnlen)3) == 0 || s_cmp(c3, "QLF", (ftnlen)3, (ftnlen)3)
        == 0) {
        if (sname) {
        nb = 32;
        } else {
        nb = 32;
        }
    } else if (s_cmp(c3, "HRD", (ftnlen)3, (ftnlen)3) == 0) {
        if (sname) {
        nb = 32;
        } else {
        nb = 32;
        }
    } else if (s_cmp(c3, "BRD", (ftnlen)3, (ftnlen)3) == 0) {
        if (sname) {
        nb = 32;
        } else {
        nb = 32;
        }
    } else if (s_cmp(c3, "TRI", (ftnlen)3, (ftnlen)3) == 0) {
        if (sname) {
        nb = 64;
        } else {
        nb = 64;
        }
    }
    } else if (s_cmp(c2, "PO", (ftnlen)2, (ftnlen)2) == 0) {
    // 如果前两个字符是 "PO"
    if (s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
        // 如果后三个字符是 "TRF"
        if (sname) {
        nb = 64;
        } else {
        nb = 64;
        }
    }
    } else if (s_cmp(c2, "SY", (ftnlen)2, (ftnlen)2) == 0) {
    // 如果前两个字符是 "SY"
    if (s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
        // 如果后三个字符是 "TRF"
        if (sname) {
        nb = 64;
        } else {
        nb = 64;
        }
    // 如果后三个字符是 "TRD" 且 sname 为真
    } else if (sname && s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
        nb = 32;
    // 如果后三个字符是 "GST" 且 sname 为真
    } else if (sname && s_cmp(c3, "GST", (ftnlen)3, (ftnlen)3) == 0) {
        nb = 64;
    }
    } else if (cname && s_cmp(c2, "HE", (ftnlen)2, (ftnlen)2) == 0) {
    // 如果 cname 为真且前两个字符是 "HE"
    if (s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
        // 如果后三个字符是 "TRF"
        nb = 64;
    // 如果后三个字符是 "TRD"
    } else if (s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
        nb = 32;
    // 如果后三个字符是 "GST"
    } else if (s_cmp(c3, "GST", (ftnlen)3, (ftnlen)3) == 0) {
        nb = 64;
    }
    } else if (sname && s_cmp(c2, "OR", (ftnlen)2, (ftnlen)2) == 0) {
    // 如果 sname 为真且前两个字符是 "OR"
    if (*(unsigned char *)c3 == 'G') {
        // 如果第三个字符是 'G'
        // 如果后四个字符是 "QR"、"RQ"、"LQ"、"QL"、"HR"、"TR"、"BR"
        if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ",
            (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "LQ", (ftnlen)2, (
            ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) ==
             0 || s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(
            c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "BR", (
            ftnlen)2, (ftnlen)2) == 0) {
        nb = 32;
        }
    } else if (*(unsigned char *)c3 == 'M') {
        // 如果第三个字符是 'M'
        // 如果后四个字符是 "QR"、"RQ"、"LQ"、"QL"、"HR"、"TR"、"BR"
        if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ",
            (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "LQ", (ftnlen)2, (
            ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) ==
             0 || s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(
            c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "BR", (
            ftnlen)2, (ftnlen)2) == 0) {
        nb = 32;
        }
    }
    } else if (cname && s_cmp(c2, "UN", (ftnlen)2, (ftnlen)2) == 0) {
    // 检查 c3 指向的字符是否为 'G'
    if (*(unsigned char *)c3 == 'G') {
        // 如果 c3 是 'G'，则检查 c4 是否等于以下任一字符串："QR", "RQ", "LQ", "QL", "HR", "TR", "BR"
        if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ",
            (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "LQ", (ftnlen)2, (
            ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) ==
             0 || s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(
            c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "BR", (
            ftnlen)2, (ftnlen)2) == 0) {
            // 如果匹配成功，则设置 nb 为 32
            nb = 32;
        }
    } else if (*(unsigned char *)c3 == 'M') {
        // 如果 c3 是 'M'，则同样检查 c4 是否等于以上述字符串之一
        if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ",
            (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "LQ", (ftnlen)2, (
            ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) ==
             0 || s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(
            c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "BR", (
            ftnlen)2, (ftnlen)2) == 0) {
            // 如果匹配成功，则设置 nb 为 32
            nb = 32;
        }
    }
    // 如果 c2 等于 "GB"
    } else if (s_cmp(c2, "GB", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果 c3 等于 "TRF"
        if (s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
            // 如果 sname 为真，并且 n4 小于等于 64，则设置 nb 为 1；否则设置 nb 为 32
            if (sname) {
                if (*n4 <= 64) {
                    nb = 1;
                } else {
                    nb = 32;
                }
            } else {
                if (*n4 <= 64) {
                    nb = 1;
                } else {
                    nb = 32;
                }
            }
        }
    // 如果 c2 等于 "PB"
    } else if (s_cmp(c2, "PB", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果 c3 等于 "TRF"
        if (s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
            // 如果 sname 为真，并且 n2 小于等于 64，则设置 nb 为 1；否则设置 nb 为 32
            if (sname) {
                if (*n2 <= 64) {
                    nb = 1;
                } else {
                    nb = 32;
                }
            } else {
                if (*n2 <= 64) {
                    nb = 1;
                } else {
                    nb = 32;
                }
            }
        }
    // 如果 c2 等于 "TR"
    } else if (s_cmp(c2, "TR", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果 c3 等于 "TRI"，则设置 nb 为 64
        if (s_cmp(c3, "TRI", (ftnlen)3, (ftnlen)3) == 0) {
            nb = 64;
        }
    // 如果 c2 等于 "LA"
    } else if (s_cmp(c2, "LA", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果 c3 等于 "UUM"，则设置 nb 为 64
        if (s_cmp(c3, "UUM", (ftnlen)3, (ftnlen)3) == 0) {
            nb = 64;
        }
    // 如果 sname 为真，并且 c2 等于 "ST"，且 c3 等于 "EBZ"，则设置 nb 为 1
    } else if (sname && s_cmp(c2, "ST", (ftnlen)2, (ftnlen)2) == 0) {
        if (s_cmp(c3, "EBZ", (ftnlen)3, (ftnlen)3) == 0) {
            nb = 1;
        }
    }
    // 返回最终确定的 nb 值
    ret_val = nb;
    return ret_val;
L60:
/*     ISPEC = 2:  minimum block size */

    nbmin = 2;  // 设置最小块大小为2

    // 如果c2字符串与"GE"相等
    if (s_cmp(c2, "GE", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果c3字符串与"QRF"、"RQF"、"LQF"、"QLF"中的任意一个相等
        if (s_cmp(c3, "QRF", (ftnlen)3, (ftnlen)3) == 0 || s_cmp(c3, "RQF", (ftnlen)3, (ftnlen)3) == 0 || 
            s_cmp(c3, "LQF", (ftnlen)3, (ftnlen)3) == 0 || s_cmp(c3, "QLF", (ftnlen)3, (ftnlen)3) == 0) {
            // 如果sname为真，则设置nbmin为2，否则保持为2
            if (sname) {
                nbmin = 2;
            } else {
                nbmin = 2;
            }
        } 
        // 如果c3字符串与"HRD"相等
        else if (s_cmp(c3, "HRD", (ftnlen)3, (ftnlen)3) == 0) {
            // 如果sname为真，则设置nbmin为2，否则保持为2
            if (sname) {
                nbmin = 2;
            } else {
                nbmin = 2;
            }
        } 
        // 如果c3字符串与"BRD"相等
        else if (s_cmp(c3, "BRD", (ftnlen)3, (ftnlen)3) == 0) {
            // 如果sname为真，则设置nbmin为2，否则保持为2
            if (sname) {
                nbmin = 2;
            } else {
                nbmin = 2;
            }
        } 
        // 如果c3字符串与"TRI"相等
        else if (s_cmp(c3, "TRI", (ftnlen)3, (ftnlen)3) == 0) {
            // 如果sname为真，则设置nbmin为2，否则保持为2
            if (sname) {
                nbmin = 2;
            } else {
                nbmin = 2;
            }
        }
    } 
    // 如果c2字符串与"SY"相等
    else if (s_cmp(c2, "SY", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果c3字符串与"TRF"相等
        if (s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
            // 如果sname为真，则设置nbmin为8，否则保持为8
            if (sname) {
                nbmin = 8;
            } else {
                nbmin = 8;
            }
        } 
        // 如果sname为真且c3字符串与"TRD"相等
        else if (sname && s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
            nbmin = 2;  // 设置nbmin为2
        }
    } 
    // 如果cname为真且c2字符串与"HE"相等
    else if (cname && s_cmp(c2, "HE", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果c3字符串与"TRD"相等
        if (s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
            nbmin = 2;  // 设置nbmin为2
        }
    } 
    // 如果sname为真且c2字符串与"OR"相等
    else if (sname && s_cmp(c2, "OR", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果c3的第一个字符是'G'
        if (*(unsigned char *)c3 == 'G') {
            // 如果c4字符串与"QR"、"RQ"、"LQ"、"QL"、"HR"、"TR"、"BR"中的任意一个相等
            if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
                nbmin = 2;  // 设置nbmin为2
            }
        } 
        // 如果c3的第一个字符是'M'
        else if (*(unsigned char *)c3 == 'M') {
            // 如果c4字符串与"QR"、"RQ"、"LQ"、"QL"、"HR"、"TR"、"BR"中的任意一个相等
            if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
                nbmin = 2;  // 设置nbmin为2
            }
        }
    } 
    // 如果cname为真且c2字符串与"UN"相等
    else if (cname && s_cmp(c2, "UN", (ftnlen)2, (ftnlen)2) == 0) {
        // 如果c3的第一个字符是'G'
        if (*(unsigned char *)c3 == 'G') {
            // 如果c4字符串与"QR"、"RQ"、"LQ"、"QL"、"HR"、"TR"、"BR"中的任意一个相等
            if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || 
                s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
                nbmin = 2;  // 设置nbmin为2
            }
        }
    }
    // 如果第三个字符是 'M'，进入条件判断
    } else if (*(unsigned char *)c3 == 'M') {
        // 检查 c4 是否等于 "QR", "RQ", "LQ", "QL", "HR", "TR", "BR" 中的任意一个
        if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ",
            (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "LQ", (ftnlen)2, (
            ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) ==
             0 || s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(
            c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "BR", (
            ftnlen)2, (ftnlen)2) == 0) {
            // 如果是上述字符串之一，设置 nbmin 为 2
            nbmin = 2;
        }
    }
    // 返回 nbmin 的值作为函数返回值
    ret_val = nbmin;
    return ret_val;
L70:
/*     ISPEC = 3:  crossover point */

    nx = 0;
    if (s_cmp(c2, "GE", (ftnlen)2, (ftnlen)2) == 0) {
        if (s_cmp(c3, "QRF", (ftnlen)3, (ftnlen)3) == 0 || s_cmp(c3, "RQF", (
            ftnlen)3, (ftnlen)3) == 0 || s_cmp(c3, "LQF", (ftnlen)3, (
            ftnlen)3) == 0 || s_cmp(c3, "QLF", (ftnlen)3, (ftnlen)3) == 0)
             {
            if (sname) {
                nx = 128;
            } else {
                nx = 128;
            }
        } else if (s_cmp(c3, "HRD", (ftnlen)3, (ftnlen)3) == 0) {
            if (sname) {
                nx = 128;
            } else {
                nx = 128;
            }
        } else if (s_cmp(c3, "BRD", (ftnlen)3, (ftnlen)3) == 0) {
            if (sname) {
                nx = 128;
            } else {
                nx = 128;
            }
        }
    } else if (s_cmp(c2, "SY", (ftnlen)2, (ftnlen)2) == 0) {
        if (sname && s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
            nx = 32;
        }
    } else if (cname && s_cmp(c2, "HE", (ftnlen)2, (ftnlen)2) == 0) {
        if (s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
            nx = 32;
        }
    } else if (sname && s_cmp(c2, "OR", (ftnlen)2, (ftnlen)2) == 0) {
        if (*(unsigned char *)c3 == 'G') {
            if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ",
                (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "LQ", (ftnlen)2, (
                ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) ==
                 0 || s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(
                c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "BR", (
                ftnlen)2, (ftnlen)2) == 0) {
                nx = 128;
            }
        }
    } else if (cname && s_cmp(c2, "UN", (ftnlen)2, (ftnlen)2) == 0) {
        if (*(unsigned char *)c3 == 'G') {
            if (s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "RQ",
                (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "LQ", (ftnlen)2, (
                ftnlen)2) == 0 || s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) ==
                 0 || s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(
                c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || s_cmp(c4, "BR", (
                ftnlen)2, (ftnlen)2) == 0) {
                nx = 128;
            }
        }
    }
    ret_val = nx;
    return ret_val;

L80:
/*     ISPEC = 4:  number of shifts (used by xHSEQR) */

    ret_val = 6;
    return ret_val;

L90:
/*     ISPEC = 5:  minimum column dimension (not used) */

    ret_val = 2;
    return ret_val;

L100:
/*     ISPEC = 6:  crossover point for SVD (used by xGELSS and xGESVD) */

    ret_val = (integer) ((real) min(*n1,*n2) * 1.6f);
    return ret_val;

L110:
/*     ISPEC = 7:  number of processors (not used) */

    ret_val = 1;
    return ret_val;

L120:
/*     ISPEC = 8:  crossover point for multishift (used by xHSEQR) */

    ret_val = 50;
    return ret_val;

L130:
/*
       ISPEC = 9:  maximum size of the subproblems at the bottom of the
                   computation tree in the divide-and-conquer algorithm
                   (used by xGELSD and xGESDD)
*/

    ret_val = 25;
    return ret_val;

L140:
/* 
       ISPEC = 10: ieee NaN arithmetic can be trusted not to trap

       ILAENV = 0
*/
    ret_val = 1;
    if (ret_val == 1) {
    ret_val = ieeeck_(&c__1, &c_b172, &c_b173);
    }
    return ret_val;

L150:

/* 
       ISPEC = 11: infinity arithmetic can be trusted not to trap

       ILAENV = 0
*/
    ret_val = 1;
    if (ret_val == 1) {
    ret_val = ieeeck_(&c__0, &c_b172, &c_b173);
    }
    return ret_val;

L160:

/*     12 <= ISPEC <= 16: xHSEQR or one of its subroutines. */

    ret_val = iparmq_(ispec, name__, opts, n1, n2, n3, n4, name_len, opts_len)
        ;
    return ret_val;

/*     End of ILAENV */

} /* ilaenv_ */

integer ilaslc_(integer *m, integer *n, real *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, ret_val, i__1;

    /* Local variables */
    static integer i__;


/*
    -- LAPACK auxiliary routine (version 3.2.2)                        --

    -- June 2010                                                       --

    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--


    Purpose
    =======

    ILASLC scans A for its last non-zero column.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.

    N       (input) INTEGER
            The number of columns of the matrix A.

    A       (input) REAL array, dimension (LDA,N)
            The m by n matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    =====================================================================


       Quick test for the common case where one corner is non-zero.
*/
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    if (*n == 0) {
    ret_val = *n;
    } else if (a[*n * a_dim1 + 1] != 0.f || a[*m + *n * a_dim1] != 0.f) {
    ret_val = *n;
    } else {
/*     Now scan each column from the end, returning with the first non-zero. */
    for (ret_val = *n; ret_val >= 1; --ret_val) {
        i__1 = *m;
        for (i__ = 1; i__ <= i__1; ++i__) {
        if (a[i__ + ret_val * a_dim1] != 0.f) {
            return ret_val;
        }
        }
    }
    }
    return ret_val;
} /* ilaslc_ */

integer ilaslr_(integer *m, integer *n, real *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, ret_val, i__1;

    /* Local variables */
    static integer i__, j;


/*
    -- LAPACK auxiliary routine (version 3.2.2)                        --

    -- June 2010                                                       --

    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--


    Purpose
    =======

    ILASLR scans A for its last non-zero row.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.

    N       (input) INTEGER
            The number of columns of the matrix A.

    A       (input) REAL array, dimension (LDA,N)
            The m by n matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    =====================================================================


       Quick test for the common case where one corner is non-zero.
*/
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    if (*m == 0) {
    ret_val = *m;
    } else if (a[*m * a_dim1 + 1] != 0.f || a[*m + *n * a_dim1] != 0.f) {
    ret_val = *m;
    } else {
/*     Now scan each row from the end, returning with the first non-zero. */
    for (ret_val = *m; ret_val >= 1; --ret_val) {
        i__1 = *n;
        for (i__ = 1; i__ <= i__1; ++i__) {
        if (a[ret_val + i__ * a_dim1] != 0.f) {
            return ret_val;
        }
        }
    }
    }
    return ret_val;
} /* ilaslr_ */
    M       (input) INTEGER
            矩阵 A 的行数。

    N       (input) INTEGER
            矩阵 A 的列数。

    A       (input) REAL array, dimension (LDA,N)
            m 行 n 列的实数数组 A。

    LDA     (input) INTEGER
            数组 A 的前导维度。要求 LDA >= max(1,M)，即 A 的第一个维度大小至少为 M。

    =====================================================================


       Quick test for the common case where one corner is non-zero.
/*     -- LAPACK auxiliary routine (version 3.2.2)                        --
       -- June 2010                                                       --
       -- LAPACK is a software package provided by Univ. of Tennessee,    --
       -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*/

    /* System generated locals */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    if (*n == 0) {
    ret_val = *n;
    } else /* if(complicated condition) */ {
    i__1 = *n * a_dim1 + 1;
    i__2 = *m + *n * a_dim1;
    if (a[i__1].r != 0. || a[i__1].i != 0. || (a[i__2].r != 0. || a[i__2]
        .i != 0.)) {
        ret_val = *n;
    } else {
/*     Now scan each column from the end, returning with the first non-zero. */
        for (ret_val = *n; ret_val >= 1; --ret_val) {
        i__1 = *m;
        for (i__ = 1; i__ <= i__1; ++i__) {
            i__2 = i__ + ret_val * a_dim1;
            if (a[i__2].r != 0. || a[i__2].i != 0.) {
            return ret_val;
            }
        }
        }
    }
    }
    return ret_val;
} /* ilazlc_ */

integer ilazlr_(integer *m, integer *n, doublecomplex *a, integer *lda)
{
    /* System generated locals */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Local variables */
    static integer i__, j;

    /* Function Body */
/*     -- LAPACK auxiliary routine (version 3.2.2)                        --
       -- June 2010                                                       --
       -- LAPACK is a software package provided by Univ. of Tennessee,    --
*/

    /* Scan up each column tracking the last zero row seen. */
    ret_val = 0;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        for (i__ = *m; i__ >= 1; --i__) {
        if (a[i__ + j * a_dim1].r != 0. || a[i__ + j * a_dim1].i != 0.) {
            goto L10;
        }
        }
L10:
        ret_val = max(ret_val,i__);
    }

    return ret_val;
} /* ilazlr_ */
    # 导入必要的库和模块
    
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    
    # 用途说明
    # ==========
    # ILAZLR 函数扫描矩阵 A，找到其最后一个非零行。
    
    # 参数
    # ==========
    # M       (input) INTEGER
    #         矩阵 A 的行数。
    
    # N       (input) INTEGER
    #         矩阵 A 的列数。
    
    # A       (input) COMPLEX*16 array, dimension (LDA,N)
    #         m 行 n 列的矩阵 A。
    
    # LDA     (input) INTEGER
    #         数组 A 的领先维度。LDA >= max(1,M)。
    
    # =====================================================================
    
    
       Quick test for the common case where one corner is non-zero.
    
    # 通常情况下的快速测试，检查一个角落是否非零。
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    if (*m == 0) {
    ret_val = *m;
    } else /* if(complicated condition) */ {
    i__1 = *m + a_dim1;
    i__2 = *m + *n * a_dim1;
    if (a[i__1].r != 0. || a[i__1].i != 0. || (a[i__2].r != 0. || a[i__2]
        .i != 0.)) {
        ret_val = *m;
    } else {
/*     Scan up each column tracking the last zero row seen. */
        ret_val = 0;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        for (i__ = *m; i__ >= 1; --i__) {
            i__2 = i__ + j * a_dim1;
            if (a[i__2].r != 0. || a[i__2].i != 0.) {
            goto L10;
            }
        }
L10:
        ret_val = max(ret_val,i__);
        }
    }
    }
    return ret_val;
} /* ilazlr_ */

integer iparmq_(integer *ispec, char *name__, char *opts, integer *n, integer
    *ilo, integer *ihi, integer *lwork, ftnlen name_len, ftnlen opts_len)
{
    /* System generated locals */
    integer ret_val, i__1, i__2;
    real r__1;

    /* Local variables */
    static integer nh, ns;


/*
    -- LAPACK auxiliary routine (version 3.2) --
    -- LAPACK is a software package provided by Univ. of Tennessee,    --
    -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
       November 2006


    Purpose
    =======

         This program sets problem and machine dependent parameters
         useful for xHSEQR and its subroutines. It is called whenever
         ILAENV is called with 12 <= ISPEC <= 16

    Arguments
    =========
    ISPEC   (input) INTEGER
            Specifies the parameter to be returned as the value of IPARMQ.

    NAME    (input) CHARACTER*(*) Name of the calling subroutine.

    OPTS    (input) CHARACTER*(*) Options to subroutines called by ILAENV.

    N       (input) INTEGER
            (description)

    ILO, IHI    (input) INTEGER
            (description)

    LWORK   (input) INTEGER
            (description)

    NAME_LEN (input) INTEGER
            (description)

    OPTS_LEN (input) INTEGER
            (description)

    Further Details
    ===============
    (additional information)
*/
    # IPARMQ 函数的详细说明和默认参数列表。
    # 这些参数控制 LAPACK 中特定 QR 迭代算法的行为和性能。
    
    # IPARMQ(ISPEC=12) 控制 xLAHQR 和 xLAQR0 之间的切换点。
    # 默认值为 75。该值至少必须为 11。
    
    # IPARMQ(ISPEC=13) 推荐的收缩窗口大小。
    # 这取决于 ILO、IHI 和 NS（由 IPARMQ(ISPEC=15) 返回的同时移位数目）。
    # 当 (IHI-ILO+1) <= 500 时，默认值为 NS；当 (IHI-ILO+1) > 500 时，默认为 3*NS/2。
    
    # IPARMQ(ISPEC=14) Nibble 交叉点，默认为 14。
    
    # IPARMQ(ISPEC=15) 同时移位数目 NS，用于多重移位 QR 迭代。
    # 根据 (IHI-ILO+1) 的大小，选择不同的默认值：
    # - 0 到 30: NS = 2+
    # - 30 到 60: NS = 4+
    # - 60 到 150: NS = 10
    # - 150 到 590: NS = **
    # - 590 到 3000: NS = 64
    # - 3000 到 6000: NS = 128
    # - 大于 6000: NS = 256
    # 其中 (+) 表示默认情况下，这些阶数的矩阵将传递给隐式双移位算法 xLAHQR。参见 IPARMQ(ISPEC=12)。
    # (**) 双星号 (**) 表示一个特定的增函数，从 10 增加到 64。
    
    # IPARMQ(ISPEC=16) 选择结构化矩阵乘法。
    # 默认值为 3。详细信息请参见 ISPEC=16。
/*
    if (*ispec == 15 || *ispec == 13 || *ispec == 16) {
*/

    /* 设置同时进行的移动次数 */
    nh = *ihi - *ilo + 1;
    ns = 2;
    if (nh >= 30) {
        ns = 4;
    }
    if (nh >= 60) {
        ns = 10;
    }
    if (nh >= 150) {
        /* 计算最大值 */
        r__1 = log((real) nh) / log(2.f);
        i__1 = 10, i__2 = nh / i_nint(&r__1);
        ns = max(i__1,i__2);
    }
    if (nh >= 590) {
        ns = 64;
    }
    if (nh >= 3000) {
        ns = 128;
    }
    if (nh >= 6000) {
        ns = 256;
    }
    /* 计算最大值 */
    i__1 = 2, i__2 = ns - ns % 2;
    ns = max(i__1,i__2);

    /* 如果 ispec 等于 12 */
    if (*ispec == 12) {

/*
      ===== Matrices of order smaller than NMIN get sent
      .     to xLAHQR, the classic double shift algorithm.
      .     This must be at least 11. ====
*/

        ret_val = 75;

    } else if (*ispec == 14) {

/*
      ==== INIBL: skip a multi-shift qr iteration and
      .    whenever aggressive early deflation finds
      .    at least (NIBBLE*(window size)/100) deflations. ====
*/

        ret_val = 14;

    } else if (*ispec == 15) {

/*        ==== NSHFTS: The number of simultaneous shifts ===== */

        ret_val = ns;

    } else if (*ispec == 13) {

/*        ==== NW: deflation window size.  ==== */

        if (nh <= 500) {
            ret_val = ns;
        } else {
            ret_val = ns * 3 / 2;
        }

    } else if (*ispec == 16) {

/*
      ==== IACC22: Whether to accumulate reflections
      .     before updating the far-from-diagonal elements
      .     and whether to use 2-by-2 block structure while
      .     doing it.  A small amount of work could be saved
      .     by making this choice dependent also upon the
      .     NH=IHI-ILO+1.
*/

        ret_val = 0;
        if (ns >= 14) {
            ret_val = 1;
        }
        if (ns >= 14) {
            ret_val = 2;
        }

    } else {
/*        ===== invalid value of ispec ===== */
        ret_val = -1;

    }

/*     ==== End of IPARMQ ==== */

    return ret_val;
} /* iparmq_ */
```