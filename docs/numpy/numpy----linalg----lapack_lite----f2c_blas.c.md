# `.\numpy\numpy\linalg\lapack_lite\f2c_blas.c`

```
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

static singlecomplex c_b21 = {1.f,0.f};
static doublecomplex c_b1078 = {1.,0.};

/* Subroutine */ int caxpy_(integer *n, singlecomplex *ca, singlecomplex *cx, integer *
    incx, singlecomplex *cy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    singlecomplex q__1, q__2;

    /* Local variables */
    static integer i__, ix, iy;
    extern doublereal scabs1_(singlecomplex *);


/*
    Purpose
    =======

       CAXPY constant times a vector plus a vector.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    // 如果向量长度小于等于0，直接返回
    if (*n <= 0) {
        return 0;
    }
    // 如果ca的绝对值为0，直接返回
    if (scabs1_(ca) == 0.f) {
        return 0;
    }
    // 如果增量都为1，使用优化过的循环（L20）
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

/*
          code for unequal increments or equal increments
            not equal to 1
*/

    // 初始化ix和iy
    ix = 1;
    iy = 1;
    // 如果incx为负数，重新计算ix
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    // 如果incy为负数，重新计算iy
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    // 执行CAXPY操作
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = iy;
        i__3 = iy;
        i__4 = ix;
        q__2.r = ca->r * cx[i__4].r - ca->i * cx[i__4].i, q__2.i = ca->r * cx[
            i__4].i + ca->i * cx[i__4].r;
        q__1.r = cy[i__3].r + q__2.r, q__1.i = cy[i__3].i + q__2.i;
        cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
        ix += *incx;
        iy += *incy;
        /* L10: */
    }
    return 0;

/*        code for both increments equal to 1 */

L20:
    // 执行CAXPY操作，优化循环
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = i__;
        i__3 = i__;
        i__4 = i__;
        q__2.r = ca->r * cx[i__4].r - ca->i * cx[i__4].i, q__2.i = ca->r * cx[
            i__4].i + ca->i * cx[i__4].r;
        q__1.r = cy[i__3].r + q__2.r, q__1.i = cy[i__3].i + q__2.i;
        cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
        /* L30: */
    }
    return 0;
} /* caxpy_ */

/* Subroutine */ int ccopy_(integer *n, singlecomplex *cx, integer *incx, singlecomplex *
    cy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Local variables */
    // 局部变量
    # 声明静态整型变量 i__, ix, iy
    static integer i__, ix, iy;
/*
    Purpose
    =======

       CCOPY copies a vector x to a vector y.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/



    /* Parameter adjustments */
    --cy;
    --cx;



    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

    /*
          code for unequal increments or equal increments
            not equal to 1
    */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = iy;
        i__3 = ix;
        cy[i__2].r = cx[i__3].r, cy[i__2].i = cx[i__3].i;
        ix += *incx;
        iy += *incy;
    /* L10: */
    }
    return 0;

    /* code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = i__;
        i__3 = i__;
        cy[i__2].r = cx[i__3].r, cy[i__2].i = cx[i__3].i;
    /* L30: */
    }
    return 0;
} /* ccopy_ */

/*
    Complex function CDOTC
*/

VOID cdotc_(singlecomplex * ret_val, integer *n, singlecomplex *cx, integer *incx, singlecomplex *cy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, ix, iy;
    static singlecomplex ctemp;

    /*
        Purpose
        =======

           forms the dot product of two vectors, conjugating the first
           vector.

        Further Details
        ===============

           jack dongarra, linpack,  3/11/78.
           modified 12/3/93, array(1) declarations changed to array(*)

        =====================================================================
    */

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    ctemp.r = 0.f, ctemp.i = 0.f;
    ret_val->r = 0.f,  ret_val->i = 0.f;
    if (*n <= 0) {
        return ;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

    /*
          code for unequal increments or equal increments
            not equal to 1
    */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        r_cnjg(&q__3, &cx[ix]);
        i__2 = iy;
        q__2.r = q__3.r * cy[i__2].r - q__3.i * cy[i__2].i, q__2.i = q__3.r *
            cy[i__2].i + q__3.i * cy[i__2].r;
        q__1.r = ctemp.r + q__2.r, q__1.i = ctemp.i + q__2.i;
        ctemp.r = q__1.r, ctemp.i = q__1.i;
        ix += *incx;
        iy += *incy;
    /* L10: */
    }
    ret_val->r = ctemp.r,  ret_val->i = ctemp.i;
    return ;

    /* code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        r_cnjg(&q__3, &cx[i__]);
        i__2 = i__;



        i__3 = i__;
        cy[i__2].r = cx[i__3].r, cy[i__2].i = cx[i__3].i;
    /* L30: */
    }
    return ;
}
    # 计算复数乘法的结果 q__3 和 cy[i__2] 的乘积，将实部和虚部分别存储在 q__2.r 和 q__2.i 中
    q__2.r = q__3.r * cy[i__2].r - q__3.i * cy[i__2].i, q__2.i = q__3.r *
        cy[i__2].i + q__3.i * cy[i__2].r;
    # 将 ctemp 和 q__2 的复数加法结果存储在 q__1 中
    q__1.r = ctemp.r + q__2.r, q__1.i = ctemp.i + q__2.i;
    # 更新 ctemp 的值为 q__1 的值，即将其设置为当前的累积值
    ctemp.r = q__1.r, ctemp.i = q__1.i;
/* L30: */
    }
     ret_val->r = ctemp.r,  ret_val->i = ctemp.i;
    return ;
} /* cdotc_ */

/* Complex */ VOID cdotu_(singlecomplex * ret_val, integer *n, singlecomplex *cx, integer
    *incx, singlecomplex *cy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    singlecomplex q__1, q__2;

    /* Local variables */
    static integer i__, ix, iy;
    static singlecomplex ctemp;


/*
    Purpose
    =======

       CDOTU forms the dot product of two vectors.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    ctemp.r = 0.f, ctemp.i = 0.f;
     ret_val->r = 0.f,  ret_val->i = 0.f;
    if (*n <= 0) {
    return ;
    }
    if (*incx == 1 && *incy == 1) {
    goto L20;
    }

/*
          code for unequal increments or equal increments
            not equal to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
    ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
    iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = ix;
    i__3 = iy;
    q__2.r = cx[i__2].r * cy[i__3].r - cx[i__2].i * cy[i__3].i, q__2.i =
        cx[i__2].r * cy[i__3].i + cx[i__2].i * cy[i__3].r;
    q__1.r = ctemp.r + q__2.r, q__1.i = ctemp.i + q__2.i;
    ctemp.r = q__1.r, ctemp.i = q__1.i;
    ix += *incx;
    iy += *incy;
/* L10: */
    }
     ret_val->r = ctemp.r,  ret_val->i = ctemp.i;
    return ;

/*        code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = i__;
    i__3 = i__;
    q__2.r = cx[i__2].r * cy[i__3].r - cx[i__2].i * cy[i__3].i, q__2.i =
        cx[i__2].r * cy[i__3].i + cx[i__2].i * cy[i__3].r;
    q__1.r = ctemp.r + q__2.r, q__1.i = ctemp.i + q__2.i;
    ctemp.r = q__1.r, ctemp.i = q__1.i;
/* L30: */
    }
     ret_val->r = ctemp.r,  ret_val->i = ctemp.i;
    return ;

} /* cdotu_ */

/* Subroutine */ int cgemm_(char *transa, char *transb, integer *m, integer *
    n, integer *k, singlecomplex *alpha, singlecomplex *a, integer *lda, singlecomplex *b,
    integer *ldb, singlecomplex *beta, singlecomplex *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3, i__4, i__5, i__6;
    singlecomplex q__1, q__2, q__3, q__4;

    /* Local variables */
    static integer i__, j, l, info;
    static logical nota, notb;
    static singlecomplex temp;
    static logical conja, conjb;
    extern logical lsame_(char *, char *);
    static integer nrowa, nrowb;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    CGEMM  performs one of the matrix-matrix operations

       C := alpha*op( A )*op( B ) + beta*C,

    where  op( X ) is one of

       op( X ) = X   or   op( X ) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Parameters
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:
                TRANSA = 'N' or 'n',  op( A ) = A.
                TRANSA = 'T' or 't',  op( A ) = A'.
                TRANSA = 'C' or 'c',  op( A ) = A'.
             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in
             the matrix multiplication as follows:
                TRANSB = 'N' or 'n',  op( B ) = B.
                TRANSB = 'T' or 't',  op( B ) = B'.
                TRANSB = 'C' or 'c',  op( B ) = B'.
             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( C )  and  of the  matrix  op( A ).  M  must  be at least
             zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( C ) and the number of columns of the matrix op( B ). N
             must be at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16      .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX*16       array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B      - COMPLEX*16       array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.

    BETA   - COMPLEX*16      .
             On entry,  BETA  specifies the scalar  beta.
             Unchanged on exit.

    C      - COMPLEX*16       array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case  C  need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
    where  op( X ) is one of

       op( X ) = X   or   op( X ) = X'   or   op( X ) = conjg( X' ),

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Arguments
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n',  op( A ) = A.

                TRANSA = 'T' or 't',  op( A ) = A'.

                TRANSA = 'C' or 'c',  op( A ) = conjg( A' ).

             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in
             the matrix multiplication as follows:

                TRANSB = 'N' or 'n',  op( B ) = B.

                TRANSB = 'T' or 't',  op( B ) = B'.

                TRANSB = 'C' or 'c',  op( B ) = conjg( B' ).

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( A )  and of the  matrix  C.  M  must  be at least  zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( B ) and the number of columns of the matrix C. N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA  - COMPLEX         .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX          array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B      - COMPLEX          array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.


注释：
    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.

```    
    BETA   - COMPLEX         .
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.

```    
    C      - COMPLEX          array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).

```    
    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.

```    
    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

```    
    =====================================================================


       Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
       conjugated or transposed, set  CONJA and CONJB  as true if  A  and
       B  respectively are to be  transposed but  not conjugated  and set
       NROWA and  NROWB  as the number of rows and  columns  of  A
       and the number of rows of  B  respectively.
    /* Parameter adjustments */
    // 调整参数指针的偏移量
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    // 根据 transa 是否为 "N" 判断是否需要转置 A 矩阵
    nota = lsame_(transa, "N");
    // 根据 transb 是否为 "N" 判断是否需要转置 B 矩阵
    notb = lsame_(transb, "N");
    // 根据 transa 是否为 "C" 判断是否需要共轭转置 A 矩阵
    conja = lsame_(transa, "C");
    // 根据 transb 是否为 "C" 判断是否需要共轭转置 B 矩阵
    conjb = lsame_(transb, "C");
    // 根据 transa 是否为 "N" 确定 A 矩阵的行数
    if (nota) {
        nrowa = *m;
    } else {
        nrowa = *k;
    }
    // 根据 transb 是否为 "N" 确定 B 矩阵的行数
    if (notb) {
        nrowb = *k;
    } else {
        nrowb = *n;
    }

/*     Test the input parameters. */
    // 检查输入参数
    info = 0;
    if (! nota && ! conja && ! lsame_(transa, "T")) {
        info = 1;
    } else if (! notb && ! conjb && ! lsame_(transb, "T")) {
        info = 2;
    } else if (*m < 0) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*k < 0) {
        info = 5;
    } else if (*lda < max(1,nrowa)) {
        info = 8;
    } else if (*ldb < max(1,nrowb)) {
        info = 10;
    } else if (*ldc < max(1,*m)) {
        info = 13;
    }
    // 如果参数有误，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("CGEMM ", &info);
        return 0;
    }

/*     Quick return if possible. */
    // 如果某些情况下可以快速返回结果
    if (*m == 0 || *n == 0 || (alpha->r == 0.f && alpha->i == 0.f || *k == 0)
        && (beta->r == 1.f && beta->i == 0.f)) {
        return 0;
    }

/*     And when alpha.eq.zero. */
    // 当 alpha 等于零时的处理逻辑
    if (alpha->r == 0.f && alpha->i == 0.f) {
        if (beta->r == 0.f && beta->i == 0.f) {
            // C := 0 的情况
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * c_dim1;
                    c__[i__3].r = 0.f, c__[i__3].i = 0.f;
                }
            }
        } else {
            // C := beta*C 的情况
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    q__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                    q__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                }
            }
        }
        return 0;
    }

/*     Start the operations. */
    // 开始执行乘法操作
    if (notb) {
        if (nota) {
            // C := alpha*A*B + beta*C 的情况
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (beta->r == 0.f && beta->i == 0.f) {
                    // C := alpha*A*B 的情况
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0.f, c__[i__3].i = 0.f;
                    }
                } else if (beta->r != 1.f || beta->i != 0.f) {
                    // C := alpha*A*B + beta*C 的情况
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        q__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                        q__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                        c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                    }
                }
            }
        } else {
            // C := alpha*A*B 的情况，A 需要转置
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (beta->r == 0.f && beta->i == 0.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0.f, c__[i__3].i = 0.f;
                    }
                } else if (beta->r != 1.f || beta->i != 0.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        q__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                        q__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                        c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                    }
                }
            }
        }
    } else {
        // C := alpha*A*B 的情况，B 需要转置
        if (nota) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (beta->r == 0.f && beta->i == 0.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0.f, c__[i__3].i = 0.f;
                    }
                } else if (beta->r != 1.f || beta->i != 0.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        q__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                        q__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                        c__[i__3].r = q__1.r, c__[i__3].i = q__1.i
/* L60: */
            }
        }
        i__2 = *k;
        // 对于每个列 l，计算 C 的第 j 列
        for (l = 1; l <= i__2; ++l) {
            // 计算 alpha * B 的第 j 列的第 l 行元素乘积
            i__3 = l + j * b_dim1;
            if (b[i__3].r != 0.f || b[i__3].i != 0.f) {
            i__3 = l + j * b_dim1;
            // 计算 alpha * B 的第 j 列的第 l 行元素乘积
            q__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i,
                q__1.i = alpha->r * b[i__3].i + alpha->i * b[
                i__3].r;
            temp.r = q__1.r, temp.i = q__1.i;
            i__3 = *m;
            // 将 alpha * A 的第 l 列加到 C 的第 j 列上
            for (i__ = 1; i__ <= i__3; ++i__) {
                i__4 = i__ + j * c_dim1;
                i__5 = i__ + j * c_dim1;
                i__6 = i__ + l * a_dim1;
                // 计算 temp * A 的第 l 列的第 i 行元素乘积，并加到 C 的第 j 列的第 i 行上
                q__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                    q__2.i = temp.r * a[i__6].i + temp.i * a[
                    i__6].r;
                q__1.r = c__[i__5].r + q__2.r, q__1.i = c__[i__5]
                    .i + q__2.i;
                c__[i__4].r = q__1.r, c__[i__4].i = q__1.i;
/* L70: */
            }
            }
/* L80: */
        }
/* L90: */
        }
    } else if (conja) {

/*           Form  C := alpha*conjg( A' )*B + beta*C. */

        i__1 = *n;
        // 对于每个列 j，计算 C 的第 j 列
        for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        // 对于每个行 i，计算 C 的第 j 列的第 i 行
        for (i__ = 1; i__ <= i__2; ++i__) {
            temp.r = 0.f, temp.i = 0.f;
            i__3 = *k;
            // 对于每个列 l，计算 alpha * conjg(A)' 的第 l 行与 B 的第 j 列的乘积之和
            for (l = 1; l <= i__3; ++l) {
            r_cnjg(&q__3, &a[l + i__ * a_dim1]);
            i__4 = l + j * b_dim1;
            q__2.r = q__3.r * b[i__4].r - q__3.i * b[i__4].i,
                q__2.i = q__3.r * b[i__4].i + q__3.i * b[i__4]
                .r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L100: */
            }
            // 根据 beta 的值更新 C 的第 j 列的第 i 行
            if (beta->r == 0.f && beta->i == 0.f) {
            i__3 = i__ + j * c_dim1;
            q__1.r = alpha->r * temp.r - alpha->i * temp.i,
                q__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            q__2.r = alpha->r * temp.r - alpha->i * temp.i,
                q__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            q__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, q__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
/* L110: */
        }
/* L120: */
        }
    } else {
/*           Form  C := alpha*A'*B + beta*C */

        /* 循环遍历 C 的列 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            /* 循环遍历 C 的行 */
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                /* 初始化临时复数变量 temp */
                temp.r = 0.f, temp.i = 0.f;
                /* 循环遍历 A 的列或 B 的行 */
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    /* 计算复数乘法 */
                    i__4 = l + i__ * a_dim1;
                    i__5 = l + j * b_dim1;
                    q__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5].i, q__2.i = a[i__4].r * b[i__5].i + a[i__4].i * b[i__5].r;
                    q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
                    temp.r = q__1.r, temp.i = q__1.i;
/* L130: */         /* 循环体结束标记 */
                }
                /* 根据 beta 的值更新 C 的元素 */
                if (beta->r == 0.f && beta->i == 0.f) {
                    i__3 = i__ + j * c_dim1;
                    q__1.r = alpha->r * temp.r - alpha->i * temp.i, q__1.i = alpha->r * temp.i + alpha->i * temp.r;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                } else {
                    i__3 = i__ + j * c_dim1;
                    q__2.r = alpha->r * temp.r - alpha->i * temp.i, q__2.i = alpha->r * temp.i + alpha->i * temp.r;
                    i__4 = i__ + j * c_dim1;
                    q__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i, q__3.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                }
/* L140: */     /* 循环体结束标记 */
            }
/* L150: */     /* 循环体结束标记 */
        }
    }
    } else if (nota) {
        if (conjb) {

/*           Form  C := alpha*A*conjg( B' ) + beta*C. */

            /* 循环遍历 C 的列 */
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                /* 根据 beta 的值初始化或更新 C 的元素 */
                if (beta->r == 0.f && beta->i == 0.f) {
                    /* 如果 beta 为零，初始化 C 的元素为零 */
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0.f, c__[i__3].i = 0.f;
/* L160: */             /* 循环体结束标记 */
                    }
                } else if (beta->r != 1.f || beta->i != 0.f) {
                    /* 如果 beta 不为一，按照 alpha 和 beta 更新 C 的元素 */
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        q__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i, q__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                        c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                    }
                }
/* L170: */     /* 循环体结束标记 */
            }
/* L180: */     /* 循环体结束标记 */
        }
    }
/* L170: */
            }
        }
        // 循环：对每个列 l = 1 到 k
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            // 计算 beta 不等于零时的操作
            i__3 = j + l * b_dim1;
            if (b[i__3].r != 0.f || b[i__3].i != 0.f) {
                // 计算 alpha 和 b 的共轭乘积，存储在 temp 中
                r_cnjg(&q__2, &b[j + l * b_dim1]);
                q__1.r = alpha->r * q__2.r - alpha->i * q__2.i,
                    q__1.i = alpha->r * q__2.i + alpha->i *
                    q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
                // 循环：对每个行 i = 1 到 m
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    // 计算 temp 与 a 的乘积，加到 c 上
                    i__4 = i__ + j * c_dim1;
                    i__5 = i__ + j * c_dim1;
                    i__6 = i__ + l * a_dim1;
                    q__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                        q__2.i = temp.r * a[i__6].i + temp.i * a[
                        i__6].r;
                    q__1.r = c__[i__5].r + q__2.r, q__1.i = c__[i__5]
                        .i + q__2.i;
                    c__[i__4].r = q__1.r, c__[i__4].i = q__1.i;
/* L180: */
                }
            }
/* L190: */
        }
/* L200: */
        }
    } else {

/*           Form  C := alpha*A*B'          + beta*C */

        // 循环：对每个列 j = 1 到 n
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 如果 beta 等于零时的操作
            if (beta->r == 0.f && beta->i == 0.f) {
                // 循环：对每个行 i = 1 到 m
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 将 c 设为零
                    i__3 = i__ + j * c_dim1;
                    c__[i__3].r = 0.f, c__[i__3].i = 0.f;
/* L210: */
                }
            } else if (beta->r != 1.f || beta->i != 0.f) {
                // 如果 beta 不等于 1 时的操作
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 计算 beta 与 c 的乘积，存储在 c 中
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    q__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                        .i, q__1.i = beta->r * c__[i__4].i + beta->i *
                         c__[i__4].r;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L220: */
                }
            }
            // 循环：对每个列 l = 1 到 k
            i__2 = *k;
            for (l = 1; l <= i__2; ++l) {
                // 如果 b 的元素不为零时的操作
                i__3 = j + l * b_dim1;
                if (b[i__3].r != 0.f || b[i__3].i != 0.f) {
                    // 计算 alpha 与 b 的乘积，存储在 temp 中
                    i__3 = j + l * b_dim1;
                    q__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i,
                        q__1.i = alpha->r * b[i__3].i + alpha->i * b[
                        i__3].r;
                    temp.r = q__1.r, temp.i = q__1.i;
                    // 循环：对每个行 i = 1 到 m
                    i__3 = *m;
                    for (i__ = 1; i__ <= i__3; ++i__) {
                        // 计算 temp 与 a 的乘积，加到 c 上
                        i__4 = i__ + j * c_dim1;
                        i__5 = i__ + j * c_dim1;
                        i__6 = i__ + l * a_dim1;
                        q__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                            q__2.i = temp.r * a[i__6].i + temp.i * a[
                            i__6].r;
                        q__1.r = c__[i__5].r + q__2.r, q__1.i = c__[i__5]
                            .i + q__2.i;
                        c__[i__4].r = q__1.r, c__[i__4].i = q__1.i;
/* L230: */
                    }
                }
/* L240: */
            }
/* L250: */
        }
    }
    } else if (conja) {
    if (conjb) {
/*           Form  C := alpha*conjg( A' )*conjg( B' ) + beta*C. */

        // 循环遍历 C 矩阵的列
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 循环遍历 C 矩阵的行
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 初始化临时复数变量 temp 为零
            temp.r = 0.f, temp.i = 0.f;
            // 循环遍历 A 和 B 矩阵的列
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 alpha*conjg( A' )*conjg( B' ) 的每一项
            r_cnjg(&q__3, &a[l + i__ * a_dim1]);
            r_cnjg(&q__4, &b[j + l * b_dim1]);
            q__2.r = q__3.r * q__4.r - q__3.i * q__4.i, q__2.i =
                q__3.r * q__4.i + q__3.i * q__4.r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L260: */   // 循环结束
            }
            // 根据 beta 的值更新 C 矩阵的元素
            if (beta->r == 0.f && beta->i == 0.f) {
            i__3 = i__ + j * c_dim1;
            q__1.r = alpha->r * temp.r - alpha->i * temp.i,
                q__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            q__2.r = alpha->r * temp.r - alpha->i * temp.i,
                q__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            q__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, q__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
/* L270: */   // 循环结束
        }
/* L280: */   // 循环结束
        }
    } else {

/*           Form  C := alpha*conjg( A' )*B' + beta*C */

        // 循环遍历 C 矩阵的列
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 循环遍历 C 矩阵的行
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 初始化临时复数变量 temp 为零
            temp.r = 0.f, temp.i = 0.f;
            // 循环遍历 A 矩阵的列和 B 矩阵的行
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 alpha*conjg( A' )*B' 的每一项
            r_cnjg(&q__3, &a[l + i__ * a_dim1]);
            i__4 = j + l * b_dim1;
            q__2.r = q__3.r * b[i__4].r - q__3.i * b[i__4].i,
                q__2.i = q__3.r * b[i__4].i + q__3.i * b[i__4]
                .r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L290: */   // 循环结束
            }
            // 根据 beta 的值更新 C 矩阵的元素
            if (beta->r == 0.f && beta->i == 0.f) {
            i__3 = i__ + j * c_dim1;
            q__1.r = alpha->r * temp.r - alpha->i * temp.i,
                q__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            q__2.r = alpha->r * temp.r - alpha->i * temp.i,
                q__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            q__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, q__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
/* L300: */   // 循环结束
        }
/* L310: 结束之前的代码块 */

        }
    }
    } else {
    if (conjb) {

/*           计算 C := alpha*A'*conjg( B' ) + beta*C */

        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            temp.r = 0.f, temp.i = 0.f;
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            i__4 = l + i__ * a_dim1;
            r_cnjg(&q__3, &b[j + l * b_dim1]);
            q__2.r = a[i__4].r * q__3.r - a[i__4].i * q__3.i,
                q__2.i = a[i__4].r * q__3.i + a[i__4].i *
                q__3.r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L320: 循环内部的结束标记 */
            }
            if (beta->r == 0.f && beta->i == 0.f) {
            i__3 = i__ + j * c_dim1;
            q__1.r = alpha->r * temp.r - alpha->i * temp.i,
                q__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            q__2.r = alpha->r * temp.r - alpha->i * temp.i,
                q__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            q__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, q__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
/* L330: 内层循环的结束标记 */
        }
/* L340: 外层循环的结束标记 */
        }
    } else {

/*           计算 C := alpha*A'*B' + beta*C */

        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            temp.r = 0.f, temp.i = 0.f;
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            i__4 = l + i__ * a_dim1;
            i__5 = j + l * b_dim1;
            q__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5]
                .i, q__2.i = a[i__4].r * b[i__5].i + a[i__4]
                .i * b[i__5].r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L350: 内层循环的结束标记 */
            }
/* L360: 外层循环的结束标记 */
        }
/* L370: 最外层循环的结束标记 */
        }
    }
/* L350: */
            }
            // 检查 beta 是否为零，若为零则只计算 alpha*A*x 的结果
            if (beta->r == 0.f && beta->i == 0.f) {
            // 计算 y[i] = alpha*A*x 的结果，不考虑 beta 的影响
            i__3 = i__ + j * c_dim1;
            q__1.r = alpha->r * temp.r - alpha->i * temp.i,
                q__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            } else {
            // 计算 y[i] = alpha*A*x + beta*y 的结果
            i__3 = i__ + j * c_dim1;
            q__2.r = alpha->r * temp.r - alpha->i * temp.i,
                q__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            // 加上 beta*y 的部分
            i__4 = i__ + j * c_dim1;
            q__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, q__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
/* L360: */
        }
/* L370: */
        }
    }
    }

    return 0;

/*     End of CGEMM . */

} /* cgemm_ */

/* Subroutine */ int cgemv_(char *trans, integer *m, integer *n, singlecomplex *
    alpha, singlecomplex *a, integer *lda, singlecomplex *x, integer *incx, singlecomplex *
    beta, singlecomplex *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static singlecomplex temp;
    static integer lenx, leny;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj;


/*
    Purpose
    =======

    CGEMV performs one of the matrix-vector operations

       y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or

       y := alpha*conjg( A' )*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ==========

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.

                TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.

                TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX         .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX          array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients.
             Unchanged on exit.


注释：以上是部分代码的注释，遵循了每一行都进行注释的要求，保持了原有的缩进和结构。
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    X      - COMPLEX          array of DIMENSION at least
             ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
             Before entry, the incremented array X must contain the
             vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - COMPLEX         .
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - COMPLEX          array of DIMENSION at least
             ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
             Before entry with BETA non-zero, the incremented array Y
             must contain the vector y. On exit, Y is overwritten by the
             updated vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================


       Test the input parameters.
    /* 参数调整 */
    a_dim1 = *lda;                // 将 lda 赋值给 a_dim1，lda 是矩阵 A 的第一维度
    a_offset = 1 + a_dim1;        // 计算 a 在内存中的偏移量
    a -= a_offset;                // 调整 a 指针的位置，指向矩阵 A 的实际数据起始位置
    --x;                          // 将 x 指针向前移动一位，指向实际数据起始位置
    --y;                          // 将 y 指针向前移动一位，指向实际数据起始位置

    /* 函数主体 */
    info = 0;                     // 初始化 info 为 0，用于存储错误信息码
    if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")
        ) {                       // 检查 trans 参数是否有效
    info = 1;                     // 如果无效，设置错误信息码为 1
    } else if (*m < 0) {          // 检查 m 是否小于 0
    info = 2;                     // 如果是，设置错误信息码为 2
    } else if (*n < 0) {          // 检查 n 是否小于 0
    info = 3;                     // 如果是，设置错误信息码为 3
    } else if (*lda < max(1,*m)) { // 检查 lda 是否小于 max(1, m)
    info = 6;                     // 如果是，设置错误信息码为 6
    } else if (*incx == 0) {      // 检查 incx 是否等于 0
    info = 8;                     // 如果是，设置错误信息码为 8
    } else if (*incy == 0) {      // 检查 incy 是否等于 0
    info = 11;                    // 如果是，设置错误信息码为 11
    }
    if (info != 0) {              // 如果存在错误信息
    xerbla_("CGEMV ", &info);     // 调用错误处理函数 xerbla 输出错误信息并终止程序
    return 0;                     // 返回 0，终止函数运行
    }

/*     如果可能的话，快速返回 */

    if (*m == 0 || *n == 0 || alpha->r == 0.f && alpha->i == 0.f && (beta->r
        == 1.f && beta->i == 0.f)) {  // 检查是否可以直接返回
    return 0;                     // 如果可以，直接返回 0
    }

    noconj = lsame_(trans, "T");   // 检查是否为传输矩阵，设置 noconj 变量

/*
       设置 LENX 和 LENY，向量 x 和 y 的长度，并设置 X 和 Y 的起始点。
*/

    if (lsame_(trans, "N")) {     // 如果不是传输矩阵
    lenx = *n;                    // 设置向量 x 的长度为 n
    leny = *m;                    // 设置向量 y 的长度为 m
    } else {                       // 如果是传输矩阵
    lenx = *m;                    // 设置向量 x 的长度为 m
    leny = *n;                    // 设置向量 y 的长度为 n
    }
    if (*incx > 0) {               // 如果 incx 大于 0
    kx = 1;                       // 设置 x 的起始位置为 1
    } else {                       // 否则
    kx = 1 - (lenx - 1) * *incx;  // 根据 incx 计算 x 的起始位置
    }
    if (*incy > 0) {               // 如果 incy 大于 0
    ky = 1;                       // 设置 y 的起始位置为 1
    } else {                       // 否则
    ky = 1 - (leny - 1) * *incy;  // 根据 incy 计算 y 的起始位置
    }

/*
       开始操作。在此版本中，通过一次对 A 的顺序访问来访问 A 的元素。

       首先形成 y := beta*y。
*/

    if (beta->r != 1.f || beta->i != 0.f) {  // 检查 beta 是否为 1
    if (*incy == 1) {             // 如果 incy 等于 1
        if (beta->r == 0.f && beta->i == 0.f) {  // 如果 beta 为 0
        i__1 = leny;
        for (i__ = 1; i__ <= i__1; ++i__) {  // 循环遍历 y
            i__2 = i__;              // 计算 y 中当前位置的索引
            y[i__2].r = 0.f, y[i__2].i = 0.f;  // 将 y 中的元素置为 0
/* L10: */            // 循环标签 L10
        }
        } else {                   // 如果 beta 不为 0
        i__1 = leny;
        for (i__ = 1; i__ <= i__1; ++i__) {  // 循环遍历 y
            i__2 = i__;              // 计算 y 中当前位置的索引
            i__3 = i__;              // 计算 y 中当前位置的索引
            q__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                q__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
                .r;                    // 计算新的 y 中的值
            y[i__2].r = q__1.r, y[i__2].i = q__1.i;  // 更新 y 中的元素
/* L20: */            // 循环标签 L20
        }
        }
    } else {                       // 如果 incy 不等于 1
        iy = ky;                   // 设置 y 的起始位置
        if (beta->r == 0.f && beta->i == 0.f) {  // 如果 beta 为 0
        i__1 = leny;
        for (i__ = 1; i__ <= i__1; ++i__) {  // 循环遍历 y
            i__2 = iy;               // 计算 y 中当前位置的索引
            y[i__2].r = 0.f, y[i__2].i = 0.f;  // 将 y 中的元素置为 0
            iy += *incy;             // 更新 y 的位置
/* L30: */            // 循环标签 L30
        }
        } else {                   // 如果 beta 不为 0
        i__1 = leny;
        for (i__ = 1; i__ <= i__1; ++i__) {  // 循环遍历 y
            i__2 = iy;               // 计算 y 中当前位置的索引
            i__3 = iy;               // 计算 y 中当前位置的索引
            q__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                q__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
                .r;                    // 计算新的 y 中的值
            y[i__2].r = q__1.r, y[i__2].i = q__1.i;  // 更新 y 中的元素
            iy += *incy;             // 更新 y 的位置
/* L40: */            // 循环标签 L40
        }
        }
    }
    }
    if (alpha->r == 0.f && alpha->i == 0.f) {  // 检查 alpha 是否为 0
    return 0;                     // 如果是，直接返回 0
    }
    if (lsame_(trans, "N")) {

/*        形成 y := alpha*A*x + y. */

    jx = kx;                      // 设置 x 的起始位置为 kx
    # 如果增量值为1，则进入循环，遍历列向量 x 中的每一个元素
    if (*incy == 1) {
        # 循环，遍历矩阵 A 的每一列
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            # 计算 x 向量的索引位置
            i__2 = jx;
            # 如果 x 向量的实部或虚部不为零
            if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
                # 计算 alpha 和 x[jx] 的乘积，结果存储在 temp 变量中
                i__2 = jx;
                q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2].r;
                temp.r = q__1.r, temp.i = q__1.i;
                # 循环，遍历矩阵 A 的每一行
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    # 计算 y[i__] 的索引位置
                    i__3 = i__;
                    # 计算 alpha * x[jx] * A[i__, j] 的乘积，结果存储在 q__2 中
                    i__4 = i__;
                    i__5 = i__ + j * a_dim1;
                    q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                        q__2.i = temp.r * a[i__5].i + temp.i * a[i__5].r;
                    # 计算 y[i__] = y[i__] + alpha * x[jx] * A[i__, j]
                    q__1.r = y[i__4].r + q__2.r, q__1.i = y[i__4].i + q__2.i;
                    y[i__3].r = q__1.r, y[i__3].i = q__1.i;
/* L50: */
            }
        }
        jx += *incx;
/* L60: */
        }
    } else {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = jx;
        if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
            i__2 = jx;
            q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2].r;
            temp.r = q__1.r, temp.i = q__1.i;
            iy = ky;
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = iy;
            i__4 = iy;
            i__5 = i__ + j * a_dim1;
            q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                q__2.i = temp.r * a[i__5].i + temp.i * a[i__5].r;
            q__1.r = y[i__4].r + q__2.r, q__1.i = y[i__4].i + q__2.i;
            y[i__3].r = q__1.r, y[i__3].i = q__1.i;
            iy += *incy;
/* L70: */
            }
        }
        jx += *incx;
/* L80: */
        }
    }
    } else {

/*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y. */

    jy = ky;
    if (*incx == 1) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        temp.r = 0.f, temp.i = 0.f;
        if (noconj) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * a_dim1;
            i__4 = i__;
            q__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[i__4].i, 
                q__2.i = a[i__3].r * x[i__4].i + a[i__3].i * x[i__4].r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L90: */
            }
        } else {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            r_cnjg(&q__3, &a[i__ + j * a_dim1]);
            i__3 = i__;
            q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                q__2.i = q__3.r * x[i__3].i + q__3.i * x[i__3].r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L100: */
            }
        }
        i__2 = jy;
        i__3 = jy;
        q__2.r = alpha->r * temp.r - alpha->i * temp.i, q__2.i =
            alpha->r * temp.i + alpha->i * temp.r;
        q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + q__2.i;
        y[i__2].r = q__1.r, y[i__2].i = q__1.i;
        jy += *incy;
/* L110: */
        }
    } else {
        // 计算矩阵向量乘积的累加和（结果复数），初始化为零
        temp.r = 0.f, temp.i = 0.f;
        // 设置初始向量 x 的索引
        ix = kx;
        // 如果不进行共轭操作
        if (noconj) {
            // 遍历矩阵 a 的行
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵元素与向量元素的乘积，结果加到 temp 中
                i__3 = i__ + j * a_dim1;
                i__4 = ix;
                q__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[i__4].i, q__2.i = a[i__3].r * x[i__4].i + a[i__3].i * x[i__4].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
                // 移动到下一个向量 x 的元素
                ix += *incx;
/* L120: */
            }
        } else {
            // 循环处理矩阵 A 的列 j
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算向量 x 中的复共轭乘积
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__3 = ix;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                    q__2.i = q__3.r * x[i__3].i + q__3.i * x[i__3].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
                ix += *incx;
/* L130: */
            }
        }
        // 更新向量 y 的元素 jy
        i__2 = jy;
        i__3 = jy;
        q__2.r = alpha->r * temp.r - alpha->i * temp.i, q__2.i =
            alpha->r * temp.i + alpha->i * temp.r;
        q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + q__2.i;
        y[i__2].r = q__1.r, y[i__2].i = q__1.i;
        jy += *incy;
/* L140: */
        }
    }
    }

    return 0;

/*     End of CGEMV . */

} /* cgemv_ */

/* Subroutine */ int cgerc_(integer *m, integer *n, singlecomplex *alpha, singlecomplex *
    x, integer *incx, singlecomplex *y, integer *incy, singlecomplex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    singlecomplex q__1, q__2;

    /* Local variables */
    static integer i__, j, ix, jy, kx, info;
    static singlecomplex temp;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    CGERC  performs the rank 1 operation

       A := alpha*x*conjg( y' ) + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.

    Arguments
    ==========

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX         .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - COMPLEX          array of dimension at least
             ( 1 + ( m - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the m
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    Y      - COMPLEX          array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y.
             Unchanged on exit.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

*/
    A      - COMPLEX          array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients. On exit, A is
             overwritten by the updated matrix.


# A 是一个复数类型的二维数组，维度为 (LDA, n)。
# 在函数调用前，A 的前 m 行 n 列部分必须包含系数矩阵。
# 函数执行完毕后，A 将被更新后的矩阵覆盖。



    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.


# LDA 是一个整数。
# 在函数调用时，LDA 指定了矩阵 A 在调用子程序中声明的第一个维度。
# LDA 必须至少为 max(1, m)。
# 函数执行完毕后，LDA 的值保持不变。



    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


# 更多细节
# =========

# 这是一个 Level 2 Blas 程序。

# -- 编写于 1986 年 10 月 22 日。
#    Jack Dongarra，Argonne 国家实验室。
#    Jeremy Du Croz，NAG 中央办公室。
#    Sven Hammarling，NAG 中央办公室。
#    Richard Hanson，Sandia 国家实验室。



    =====================================================================

       Test the input parameters.


# 测试输入参数。
    /* Parameter adjustments */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (*m < 0) {
    info = 1;
    } else if (*n < 0) {
    info = 2;
    } else if (*incx == 0) {
    info = 5;
    } else if (*incy == 0) {
    info = 7;
    } else if (*lda < max(1,*m)) {
    info = 9;
    }
    if (info != 0) {
    xerbla_("CGERC ", &info);
    return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || alpha->r == 0.f && alpha->i == 0.f) {
    return 0;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (*incy > 0) {
    jy = 1;
    } else {
    jy = 1 - (*n - 1) * *incy;
    }
    if (*incx == 1) {
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        i__2 = jy;
        if (y[i__2].r != 0.f || y[i__2].i != 0.f) {
        r_cnjg(&q__2, &y[jy]);
        q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i =
            alpha->r * q__2.i + alpha->i * q__2.r;
        temp.r = q__1.r, temp.i = q__1.i;
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * a_dim1;
            i__4 = i__ + j * a_dim1;
            i__5 = i__;
            q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, q__2.i =
                 x[i__5].r * temp.i + x[i__5].i * temp.r;
            q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + q__2.i;
            a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L10: */
        }
        }
        jy += *incy;
/* L20: */
    }
    } else {
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (*m - 1) * *incx;
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        i__2 = jy;
        if (y[i__2].r != 0.f || y[i__2].i != 0.f) {
        r_cnjg(&q__2, &y[jy]);
        q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i =
            alpha->r * q__2.i + alpha->i * q__2.r;
        temp.r = q__1.r, temp.i = q__1.i;
        ix = kx;
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * a_dim1;
            i__4 = i__ + j * a_dim1;
            i__5 = ix;
            q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, q__2.i =
                 x[i__5].r * temp.i + x[i__5].i * temp.r;
            q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + q__2.i;
            a[i__3].r = q__1.r, a[i__3].i = q__1.i;
            ix += *incx;
/* L30: */
        }
        }
        jy += *incy;
/* L40: */
    }
    }

    return 0;

/*     End of CGERC . */

} /* cgerc_ */

/* Subroutine */ int cgeru_(integer *m, integer *n, singlecomplex *alpha, singlecomplex *
    x, integer *incx, singlecomplex *y, integer *incy, singlecomplex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    singlecomplex q__1, q__2;

    /* Local variables */
    static integer i__, j, ix, jy, kx, info;
    static singlecomplex temp;


注释完成。
    # 声明一个外部的子程序 xerbla_，该子程序接受一个字符指针和一个整数作为参数
    extern /* Subroutine */ int xerbla_(char *, integer *);
/*
    Purpose
    =======

    CGERU  performs the rank 1 operation

       A := alpha*x*y' + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.

    Arguments
    ==========

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX         .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - COMPLEX          array of dimension at least
             ( 1 + ( m - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the m
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    Y      - COMPLEX          array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y.
             Unchanged on exit.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    A      - COMPLEX          array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients. On exit, A is
             overwritten by the updated matrix.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================

    Test the input parameters.
*/
    /* Parameter adjustments */
    --x;                                        /* 将数组 x 调整为从 1 开始索引 */
    --y;                                        /* 将数组 y 调整为从 1 开始索引 */
    a_dim1 = *lda;                              /* 计算 A 的第一维的大小 */
    a_offset = 1 + a_dim1;                       /* 计算 A 的偏移量 */
    a -= a_offset;                              /* 调整数组 A 的起始位置 */

    /* Function Body */
    info = 0;                                   /* 初始化 info 为 0 */
    if (*m < 0) {                               /* 检查参数 m 的有效性 */
        info = 1;
    } else if (*n < 0) {                        /* 检查参数 n 的有效性 */
        info = 2;
    } else if (*incx == 0) {                    /* 检查参数 incx 的有效性 */
        info = 5;
    } else if (*incy == 0) {                    /* 检查参数 incy 的有效性 */
        info = 7;
    } else if (*lda < max(1,*m)) {              /* 检查参数 lda 的有效性 */
        info = 9;
    }
    if (info != 0) {                            /* 如果有无效参数，则调用错误处理函数 xerbla_ */
        xerbla_("CGERU ", &info);
        return 0;                               /* 返回 0 表示异常终止 */
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || alpha->r == 0.f && alpha->i == 0.f) {
        return 0;                               /* 如果 m 或者 n 为 0，或者 alpha 为 0，则直接返回 */
    }
/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

if (*incy > 0) {
    jy = 1;
} else {
    jy = 1 - (*n - 1) * *incy;
}
if (*incx == 1) {
    i__1 = *n;
    // Loop over vector y with unit increment
    for (j = 1; j <= i__1; ++j) {
        i__2 = jy;
        // Check if element y(jy) is non-zero
        if (y[i__2].r != 0.f || y[i__2].i != 0.f) {
            i__2 = jy;
            // Compute alpha * y(jy) and store in temp
            q__1.r = alpha->r * y[i__2].r - alpha->i * y[i__2].i, q__1.i =
                alpha->r * y[i__2].i + alpha->i * y[i__2].r;
            temp.r = q__1.r, temp.i = q__1.i;
            i__2 = *m;
            // Perform matrix-vector multiplication A(:,j) += temp * x
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * a_dim1;
                i__4 = i__ + j * a_dim1;
                i__5 = i__;
                q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, q__2.i =
                    x[i__5].r * temp.i + x[i__5].i * temp.r;
                q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + q__2.i;
                a[i__3].r = q__1.r, a[i__3].i = q__1.i;
                // Move to the next element in vector x
            }
        }
        // Move to the next element in vector y
        jy += *incy;
    }
} else {
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (*m - 1) * *incx;
    }
    i__1 = *n;
    // Loop over vector y with non-unit increment
    for (j = 1; j <= i__1; ++j) {
        i__2 = jy;
        // Check if element y(jy) is non-zero
        if (y[i__2].r != 0.f || y[i__2].i != 0.f) {
            i__2 = jy;
            // Compute alpha * y(jy) and store in temp
            q__1.r = alpha->r * y[i__2].r - alpha->i * y[i__2].i, q__1.i =
                alpha->r * y[i__2].i + alpha->i * y[i__2].r;
            temp.r = q__1.r, temp.i = q__1.i;
            ix = kx;
            i__2 = *m;
            // Perform matrix-vector multiplication A(:,j) += temp * x
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * a_dim1;
                i__4 = i__ + j * a_dim1;
                i__5 = ix;
                q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, q__2.i =
                    x[i__5].r * temp.i + x[i__5].i * temp.r;
                q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + q__2.i;
                a[i__3].r = q__1.r, a[i__3].i = q__1.i;
                // Move to the next element in vector x
                ix += *incx;
            }
        }
        // Move to the next element in vector y
        jy += *incy;
    }
}

return 0;

/*     End of CGERU . */
} /* cgeru_ */

/* Subroutine */ int chemv_(char *uplo, integer *n, singlecomplex *alpha, singlecomplex *
    a, integer *lda, singlecomplex *x, integer *incx, singlecomplex *beta, singlecomplex *y,
     integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1;
    singlecomplex q__1, q__2, q__3, q__4;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static singlecomplex temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*
    Purpose
    =======

    CHEMV  performs the matrix-vector  operation

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.

    Arguments
    ==========
*/


注释：这部分代码定义了一个名为 `chemv_` 的子程序，实现了矩阵-向量乘法操作 `y := alpha*A*x + beta*y`，其中 alpha 和 beta 是标量，x 和 y 是长度为 n 的向量，A 是一个 n×n 的埃尔米特矩阵。
    UPLO   - CHARACTER*1.
             在进入时，UPLO 指定数组 A 的上三角部分或下三角部分的引用方式如下：

                UPLO = 'U' or 'u'   仅引用数组 A 的上三角部分。

                UPLO = 'L' or 'l'   仅引用数组 A 的下三角部分。

             在退出时保持不变。

    N      - INTEGER.
             在进入时，N 指定矩阵 A 的阶数。
             N 至少为零。
             在退出时保持不变。

    ALPHA  - COMPLEX         .
             在进入时，ALPHA 指定标量 alpha。
             在退出时保持不变。

    A      - COMPLEX          array of DIMENSION ( LDA, n ).
             在进入时：
               当 UPLO = 'U' 或 'u' 时，数组 A 的前 n 行 n 列应包含埃尔米特矩阵的上三角部分，
                 且严格下三角部分不被引用。
               当 UPLO = 'L' 或 'l' 时，数组 A 的前 n 行 n 列应包含埃尔米特矩阵的下三角部分，
                 且严格上三角部分不被引用。
               注意：对角线元素的虚部不需要设置，默认为零。
             在退出时保持不变。

    LDA    - INTEGER.
             在进入时，LDA 指定数组 A 在调用（子）程序中声明时的第一个维度。
             LDA 至少为 max(1, n)。
             在退出时保持不变。

    X      - COMPLEX          array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             在进入时，增量数组 X 必须包含 n 元素向量 x。
             在退出时保持不变。

    INCX   - INTEGER.
             在进入时，INCX 指定 X 中元素的增量。
             INCX 不能为零。
             在退出时保持不变。

    BETA   - COMPLEX         .
             在进入时，BETA 指定标量 beta。
             当 BETA 为零时，输入时 Y 不需要设置。
             在退出时保持不变。

    Y      - COMPLEX          array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             在进入时，增量数组 Y 必须包含 n 元素向量 y。
             在退出时，数组 Y 被更新后覆盖原来的向量 y。

    INCY   - INTEGER.
             在进入时，INCY 指定 Y 中元素的增量。
             INCY 不能为零。
             在退出时保持不变。

    Further Details
    ===============

    Level 2 Blas routine.
    # Written on 22-October-1986.
    # 由 Jack Dongarra, Argonne National Lab.
    # Jeremy Du Croz, Nag Central Office.
    # Sven Hammarling, Nag Central Office.
    # Richard Hanson, Sandia National Labs.
    # 1986年10月22日编写，作者包括 Jack Dongarra（阿贡国家实验室）、Jeremy Du Croz（NAG 中心办公室）、
    # Sven Hammarling（NAG 中心办公室）、Richard Hanson（Sandia 国家实验室）。
    
    =====================================================================
    
    # Test the input parameters.
    # 测试输入参数。
/*
    /* Parameter adjustments */
    // 调整参数
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    // 函数体开始
    info = 0;
    // 初始化错误信息为0
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
    // 如果uplo不是"U"也不是"L"
    info = 1;
    // 错误代码设为1
    } else if (*n < 0) {
    // 如果n小于0
    info = 2;
    // 错误代码设为2
    } else if (*lda < max(1,*n)) {
    // 如果lda小于1和n中的较大者
    info = 5;
    // 错误代码设为5
    } else if (*incx == 0) {
    // 如果incx等于0
    info = 7;
    // 错误代码设为7
    } else if (*incy == 0) {
    // 如果incy等于0
    info = 10;
    // 错误代码设为10
    }
    if (info != 0) {
    // 如果有错误信息
    xerbla_("CHEMV ", &info);
    // 调用错误处理程序xerbla
    return 0;
    // 返回0
    }

/*     Quick return if possible. */
    // 如果可能的话，快速返回

    if (*n == 0 || alpha->r == 0.f && alpha->i == 0.f && (beta->r == 1.f &&
        beta->i == 0.f)) {
    // 如果n为0或alpha和beta为零向量
    return 0;
    // 返回0
    }

/*     Set up the start points in  X  and  Y. */
    // 设置X和Y的起始点

    if (*incx > 0) {
    // 如果incx大于0
    kx = 1;
    // kx设为1
    } else {
    // 否则
    kx = 1 - (*n - 1) * *incx;
    // kx设为1减去(*n-1)*incx
    }
    if (*incy > 0) {
    // 如果incy大于0
    ky = 1;
    // ky设为1
    } else {
    // 否则
    ky = 1 - (*n - 1) * *incy;
    // ky设为1减去(*n-1)*incy
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.

       First form  y := beta*y.
*/
    // 开始操作。在这个版本中，通过对A的三角部分进行一次顺序访问。

    if (beta->r != 1.f || beta->i != 0.f) {
    // 如果beta不等于(1,0)
    if (*incy == 1) {
    // 如果incy等于1
        if (beta->r == 0.f && beta->i == 0.f) {
        // 如果beta等于(0,0)
        i__1 = *n;
        for (i__ = 1; i__ <= i__1; ++i__) {
            i__2 = i__;
            y[i__2].r = 0.f, y[i__2].i = 0.f;
        // 将y设为零向量
/* L10: */
        }
        } else {
        // 否则
        i__1 = *n;
        for (i__ = 1; i__ <= i__1; ++i__) {
            i__2 = i__;
            i__3 = i__;
            q__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                q__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
                .r;
            y[i__2].r = q__1.r, y[i__2].i = q__1.i;
        // 计算y := beta*y
/* L20: */
        }
        }
    } else {
    // 否则
        iy = ky;
        if (beta->r == 0.f && beta->i == 0.f) {
        // 如果beta等于(0,0)
        i__1 = *n;
        for (i__ = 1; i__ <= i__1; ++i__) {
            i__2 = iy;
            y[i__2].r = 0.f, y[i__2].i = 0.f;
            iy += *incy;
        // 将y设为零向量
/* L30: */
        }
        } else {
        // 否则
        i__1 = *n;
        for (i__ = 1; i__ <= i__1; ++i__) {
            i__2 = iy;
            i__3 = iy;
            q__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                q__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
                .r;
            y[i__2].r = q__1.r, y[i__2].i = q__1.i;
            iy += *incy;
        // 计算y := beta*y
/* L40: */
        }
        }
    }
    }
    if (alpha->r == 0.f && alpha->i == 0.f) {
    // 如果alpha等于(0,0)
    return 0;
    // 返回0
    }
    if (lsame_(uplo, "U")) {

/*        Form  y  when A is stored in upper triangle. */
    // 检查增量向量是否为1，如果是则进入循环
    if (*incx == 1 && *incy == 1) {
        // 循环遍历列向量
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 计算 alpha 与 x[j] 的乘积，存储在 temp1 中
            i__2 = j;
            q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, q__1.i =
                 alpha->r * x[i__2].i + alpha->i * x[i__2].r;
            temp1.r = q__1.r, temp1.i = q__1.i;
            // 初始化 temp2 为零向量
            temp2.r = 0.f, temp2.i = 0.f;
            // 循环遍历上三角矩阵的元素
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算 temp1 与 a[i,j] 的乘积，更新 y[i]
                i__3 = i__;
                i__4 = i__;
                i__5 = i__ + j * a_dim1;
                q__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                    q__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5]
                    .r;
                q__1.r = y[i__4].r + q__2.r, q__1.i = y[i__4].i + q__2.i;
                y[i__3].r = q__1.r, y[i__3].i = q__1.i;
                // 计算共轭转置的 a[i,j] 与 x[i] 的乘积，更新 temp2
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__3 = i__;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i, q__2.i =
                     q__3.r * x[i__3].i + q__3.i * x[i__3].r;
                q__1.r = temp2.r + q__2.r, q__1.i = temp2.i + q__2.i;
                temp2.r = q__1.r, temp2.i = q__1.i;
/* L50: */
/* 循环结束，该部分代码块为空，可能是一个占位符或者之前的代码段被删除 */
        }
        i__2 = j;
        i__3 = j;
        i__4 = j + j * a_dim1;
        r__1 = a[i__4].r;
        q__3.r = r__1 * temp1.r, q__3.i = r__1 * temp1.i;
        q__2.r = y[i__3].r + q__3.r, q__2.i = y[i__3].i + q__3.i;
        q__4.r = alpha->r * temp2.r - alpha->i * temp2.i, q__4.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
        y[i__2].r = q__1.r, y[i__2].i = q__1.i;
/* L60: */
/* 若条件不满足，执行这个分支 */
        }
    } else {
        jx = kx;
        jy = ky;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = jx;
        q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, q__1.i =
             alpha->r * x[i__2].i + alpha->i * x[i__2].r;
        temp1.r = q__1.r, temp1.i = q__1.i;
        temp2.r = 0.f, temp2.i = 0.f;
        ix = kx;
        iy = ky;
        i__2 = j - 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = iy;
            i__4 = iy;
            i__5 = i__ + j * a_dim1;
            q__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                q__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5]
                .r;
            q__1.r = y[i__4].r + q__2.r, q__1.i = y[i__4].i + q__2.i;
            y[i__3].r = q__1.r, y[i__3].i = q__1.i;
            r_cnjg(&q__3, &a[i__ + j * a_dim1]);
            i__3 = ix;
            q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i, q__2.i =
                 q__3.r * x[i__3].i + q__3.i * x[i__3].r;
            q__1.r = temp2.r + q__2.r, q__1.i = temp2.i + q__2.i;
            temp2.r = q__1.r, temp2.i = q__1.i;
            ix += *incx;
            iy += *incy;
/* L70: */
/* 内层循环的结束标志 */
        }
        i__2 = jy;
        i__3 = jy;
        i__4 = j + j * a_dim1;
        r__1 = a[i__4].r;
        q__3.r = r__1 * temp1.r, q__3.i = r__1 * temp1.i;
        q__2.r = y[i__3].r + q__3.r, q__2.i = y[i__3].i + q__3.i;
        q__4.r = alpha->r * temp2.r - alpha->i * temp2.i, q__4.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
        y[i__2].r = q__1.r, y[i__2].i = q__1.i;
        jx += *incx;
        jy += *incy;
/* L80: */
/* 外层循环的结束标志 */
        }
    }
    } else {

/*        Form  y  when A is stored in lower triangle. */
    // 检查是否增量为1
    if (*incx == 1 && *incy == 1) {
        // 对于每个列 j，执行以下操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 计算 alpha * x[j]
            i__2 = j;
            q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, q__1.i =
                 alpha->r * x[i__2].i + alpha->i * x[i__2].r;
            temp1.r = q__1.r, temp1.i = q__1.i;
            // 初始化 temp2 为复数零
            temp2.r = 0.f, temp2.i = 0.f;
            // 对于每个行 i（从 j+1 到 n），执行以下操作
            i__2 = j;
            i__3 = j;
            i__4 = j + j * a_dim1;
            r__1 = a[i__4].r;
            // 计算 y[j] += alpha * x[j] * A[j,j]
            q__2.r = r__1 * temp1.r, q__2.i = r__1 * temp1.i;
            q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + q__2.i;
            y[i__2].r = q__1.r, y[i__2].i = q__1.i;
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                // 计算 y[i] += alpha * x[j] * A[i,j]
                i__3 = i__;
                i__4 = i__;
                i__5 = i__ + j * a_dim1;
                q__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                    q__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5].r;
                q__1.r = y[i__4].r + q__2.r, q__1.i = y[i__4].i + q__2.i;
                y[i__3].r = q__1.r, y[i__3].i = q__1.i;
                // 计算 temp2 += conjg(A[i,j]) * x[i]
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__3 = i__;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i, q__2.i =
                     q__3.r * x[i__3].i + q__3.i * x[i__3].r;
                q__1.r = temp2.r + q__2.r, q__1.i = temp2.i + q__2.i;
                temp2.r = q__1.r, temp2.i = q__1.i;
/* L90: */
        }
        i__2 = j;
        i__3 = j;
        // 计算 alpha * temp2 的实部和虚部，加到 y[i__3] 上
        q__2.r = alpha->r * temp2.r - alpha->i * temp2.i, q__2.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + q__2.i;
        // 更新 y[i__2] 的值
        y[i__2].r = q__1.r, y[i__2].i = q__1.i;
/* L100: */
        }
    } else {
        jx = kx;
        jy = ky;
        i__1 = *n;
        // 循环遍历每一列 j
        for (j = 1; j <= i__1; ++j) {
        i__2 = jx;
        // 计算 alpha * x[jx]，结果存入 temp1
        q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, q__1.i =
             alpha->r * x[i__2].i + alpha->i * x[i__2].r;
        temp1.r = q__1.r, temp1.i = q__1.i;
        // 初始化 temp2 为复数零
        temp2.r = 0.f, temp2.i = 0.f;
        i__2 = jy;
        i__3 = jy;
        // 计算 y[jy] + alpha * temp1 * a[j + j * a_dim1]，结果存入 y[jy]
        i__4 = j + j * a_dim1;
        r__1 = a[i__4].r;
        q__2.r = r__1 * temp1.r, q__2.i = r__1 * temp1.i;
        q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + q__2.i;
        y[i__2].r = q__1.r, y[i__2].i = q__1.i;
        ix = jx;
        iy = jy;
        i__2 = *n;
        // 循环遍历每一行 i，从 j+1 到 n
        for (i__ = j + 1; i__ <= i__2; ++i__) {
            ix += *incx;
            iy += *incy;
            i__3 = iy;
            i__4 = iy;
            // 计算 y[iy] + temp1 * conjg( a[i + j * a_dim1] )，结果存入 y[iy]
            i__5 = i__ + j * a_dim1;
            q__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                q__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5]
                .r;
            q__1.r = y[i__4].r + q__2.r, q__1.i = y[i__4].i + q__2.i;
            y[i__3].r = q__1.r, y[i__3].i = q__1.i;
            // 计算 temp2 + conjg( a[i + j * a_dim1] ) * x[ix]，结果存入 temp2
            r_cnjg(&q__3, &a[i__ + j * a_dim1]);
            i__3 = ix;
            q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i, q__2.i =
                 q__3.r * x[i__3].i + q__3.i * x[i__3].r;
            q__1.r = temp2.r + q__2.r, q__1.i = temp2.i + q__2.i;
            temp2.r = q__1.r, temp2.i = q__1.i;
/* L110: */
        }
        i__2 = jy;
        i__3 = jy;
        // 计算 alpha * temp2 的实部和虚部，加到 y[jy] 上
        q__2.r = alpha->r * temp2.r - alpha->i * temp2.i, q__2.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + q__2.i;
        // 更新 y[jy] 的值
        y[i__2].r = q__1.r, y[i__2].i = q__1.i;
        jx += *incx;
        jy += *incy;
/* L120: */
        }
    }
    }

    return 0;

/*     End of CHEMV . */

} /* chemv_ */

/* Subroutine */ int cher2_(char *uplo, integer *n, singlecomplex *alpha, singlecomplex *
    x, integer *incx, singlecomplex *y, integer *incy, singlecomplex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    real r__1;
    singlecomplex q__1, q__2, q__3, q__4;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static singlecomplex temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*
    Purpose
    =======

    CHER2  performs the hermitian rank 2 operation

       A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A,

    where alpha is a scalar, x and y are n element vectors and A is an n
    by n hermitian matrix.

    Arguments
    ! UPLO - CHARACTER*1.
    !        On entry, UPLO specifies whether the upper or lower
    !        triangular part of the array A is to be referenced as
    !        follows:
    !
    !           UPLO = 'U' or 'u'   Only the upper triangular part of A
    !                               is to be referenced.
    !
    !           UPLO = 'L' or 'l'   Only the lower triangular part of A
    !                               is to be referenced.
    !
    !        Unchanged on exit.
    
    ! N - INTEGER.
    !     On entry, N specifies the order of the matrix A.
    !     N must be at least zero.
    !     Unchanged on exit.
    
    ! ALPHA - COMPLEX.
    !         On entry, ALPHA specifies the scalar alpha.
    !         Unchanged on exit.
    
    ! X - COMPLEX array of dimension at least
    !     (1 + (n - 1)*abs(INCX)).
    !     Before entry, the incremented array X must contain the n
    !     element vector x.
    !     Unchanged on exit.
    
    ! INCX - INTEGER.
    !        On entry, INCX specifies the increment for the elements of
    !        X. INCX must not be zero.
    !        Unchanged on exit.
    
    ! Y - COMPLEX array of dimension at least
    !     (1 + (n - 1)*abs(INCY)).
    !     Before entry, the incremented array Y must contain the n
    !     element vector y.
    !     Unchanged on exit.
    
    ! INCY - INTEGER.
    !        On entry, INCY specifies the increment for the elements of
    !        Y. INCY must not be zero.
    !        Unchanged on exit.
    
    ! A - COMPLEX array of DIMENSION (LDA, n).
    !     Before entry with UPLO = 'U' or 'u', the leading n by n
    !     upper triangular part of the array A must contain the upper
    !     triangular part of the hermitian matrix and the strictly
    !     lower triangular part of A is not referenced. On exit, the
    !     upper triangular part of the array A is overwritten by the
    !     upper triangular part of the updated matrix.
    !     Before entry with UPLO = 'L' or 'l', the leading n by n
    !     lower triangular part of the array A must contain the lower
    !     triangular part of the hermitian matrix and the strictly
    !     upper triangular part of A is not referenced. On exit, the
    !     lower triangular part of the array A is overwritten by the
    !     lower triangular part of the updated matrix.
    !     Note that the imaginary parts of the diagonal elements need
    !     not be set, they are assumed to be zero, and on exit they
    !     are set to zero.
    
    ! LDA - INTEGER.
    !       On entry, LDA specifies the first dimension of A as declared
    !       in the calling (sub) program. LDA must be at least
    !       max(1, n).
    !       Unchanged on exit.
    
    ! Further Details
    ! ===============
    !
    ! Level 2 Blas routine.
    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.
    /* 调整参数 */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* 函数体 */
    info = 0;
    // 检查上三角或下三角参数是否正确
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (*n < 0) {
        // 检查 n 是否为负数
        info = 2;
    } else if (*incx == 0) {
        // 检查增量 incx 是否为零
        info = 5;
    } else if (*incy == 0) {
        // 检查增量 incy 是否为零
        info = 7;
    } else if (*lda < max(1,*n)) {
        // 检查 lda 是否小于 max(1, n)
        info = 9;
    }
    // 如果有错误信息，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("CHER2 ", &info);
        return 0;
    }

    /* 如果可能的话，快速返回 */
    if (*n == 0 || alpha->r == 0.f && alpha->i == 0.f) {
        return 0;
    }

    /*
       如果增量不是一，设置 X 和 Y 的起始点。
       如果增量为正，kx 从 1 开始；如果增量为负，kx 从 1 - (*n - 1) * *incx 开始。
       同理，ky 的计算类似。
    */
    if (*incx != 1 || *incy != 1) {
        if (*incx > 0) {
            kx = 1;
        } else {
            kx = 1 - (*n - 1) * *incx;
        }
        if (*incy > 0) {
            ky = 1;
        } else {
            ky = 1 - (*n - 1) * *incy;
        }
        jx = kx;
        jy = ky;
    }

    /*
       开始操作。在这个版本中，通过一次遍历 A 的三角部分顺序访问 A 的元素。
    */
    if (lsame_(uplo, "U")) {

    /* 当 A 存储在上三角时，形成 A。*/
    
        /* 如果增量为一，直接遍历 */
        if (*incx == 1 && *incy == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = j;
                i__3 = j;
                // 如果 x[j] 和 y[j] 不同时为零，则进行下面的操作
                if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f
                    || y[i__3].i != 0.f)) {
                    // 计算 temp1 和 temp2
                    r_cnjg(&q__2, &y[j]);
                    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i =
                        alpha->r * q__2.i + alpha->i * q__2.r;
                    temp1.r = q__1.r, temp1.i = q__1.i;
                    i__2 = j;
                    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                        q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                        .r;
                    r_cnjg(&q__1, &q__2);
                    temp2.r = q__1.r, temp2.i = q__1.i;
                    i__2 = j - 1;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * a_dim1;
                        i__4 = i__ + j * a_dim1;
                        i__5 = i__;
                        q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
                            q__3.i = x[i__5].r * temp1.i + x[i__5].i *
                            temp1.r;
                        q__2.r = a[i__4].r + q__3.r, q__2.i = a[i__4].i +
                            q__3.i;
                        i__6 = i__;
                        q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
                            q__4.i = y[i__6].r * temp2.i + y[i__6].i *
                            temp2.r;
                        q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
                        a[i__3].r = q__1.r, a[i__3].i = q__1.i;
                        }
                    }
                }
            }
        }
/* L10: 结束当前的 if-else 分支 */
            }
            // 计算当前元素的索引 i__2
            i__2 = j + j * a_dim1;
            // 复制当前元素的实部和虚部给临时变量 temp1
            i__3 = j + j * a_dim1;
            i__4 = j;
            q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
                q__2.i = x[i__4].r * temp1.i + x[i__4].i *
                temp1.r;
            // 复制当前元素的实部和虚部给临时变量 temp2
            i__5 = j;
            q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
                q__3.i = y[i__5].r * temp2.i + y[i__5].i *
                temp2.r;
            // 计算当前元素的实部和虚部，并赋值给 a[i__2]
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            r__1 = a[i__3].r + q__1.r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
        } else {
            // 计算当前元素的索引 i__2
            i__2 = j + j * a_dim1;
            // 将当前元素的实部赋值给 a[i__2]，虚部置为 0
            i__3 = j + j * a_dim1;
            r__1 = a[i__3].r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
        }
/* L20: 循环的下一个迭代 */
        }
    } else {
        // 循环处理向量 x 和 y 中的元素
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 计算当前元素在向量 x 中的索引 i__2 和在向量 y 中的索引 i__3
        i__2 = jx;
        i__3 = jy;
        // 如果 x[i__2] 或 y[i__3] 的实部或虚部不为零
        if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f
            || y[i__3].i != 0.f)) {
            // 计算 alpha 和 y[jy] 的共轭乘积，并赋值给 temp1
            r_cnjg(&q__2, &y[jy]);
            q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i =
                alpha->r * q__2.i + alpha->i * q__2.r;
            temp1.r = q__1.r, temp1.i = q__1.i;
            // 计算 alpha 和 x[jx] 的共轭乘积，并赋值给 temp2
            i__2 = jx;
            q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                .r;
            r_cnjg(&q__1, &q__2);
            temp2.r = q__1.r, temp2.i = q__1.i;
            // 初始化 ix 和 iy
            ix = kx;
            iy = ky;
            // 处理矩阵 a 的每一列
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
            // 计算当前元素的索引 i__3
            i__3 = i__ + j * a_dim1;
            // 计算 x[ix] 和 temp1 的乘积，并加到 a[i__3] 上
            i__4 = i__ + j * a_dim1;
            i__5 = ix;
            q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
                q__3.i = x[i__5].r * temp1.i + x[i__5].i *
                temp1.r;
            q__2.r = a[i__4].r + q__3.r, q__2.i = a[i__4].i +
                q__3.i;
            // 计算 y[iy] 和 temp2 的乘积，并加到 a[i__3] 上
            i__6 = iy;
            q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
                q__4.i = y[i__6].r * temp2.i + y[i__6].i *
                temp2.r;
            q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
            a[i__3].r = q__1.r, a[i__3].i = q__1.i;
            // 更新 ix 和 iy
            ix += *incx;
            iy += *incy;
/* L30: */
/* 结束当前的 for 循环，进入下一个条件分支或结束循环 */
            }
/* 计算索引位置 */
            i__2 = j + j * a_dim1;
/* 计算索引位置 */
            i__3 = j + j * a_dim1;
/* 计算索引位置 */
            i__4 = jx;
/* 计算乘积并减法运算，得到复数结果 */
            q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
                q__2.i = x[i__4].r * temp1.i + x[i__4].i *
                temp1.r;
/* 计算乘积并减法运算，得到复数结果 */
            i__5 = jy;
            q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
                q__3.i = y[i__5].r * temp2.i + y[i__5].i *
                temp2.r;
/* 复数加法运算 */
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
/* 计算和并赋值给矩阵元素 */
            r__1 = a[i__3].r + q__1.r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
        } else {
/* 计算索引位置 */
            i__2 = j + j * a_dim1;
/* 将实部赋值给矩阵元素，虚部设为 0 */
            r__1 = a[i__3].r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
        }
/* 更新向量 x 和 y 的索引 */
        jx += *incx;
        jy += *incy;
/* L40: */
        }
    }
    } else {

/*        Form  A  when A is stored in the lower triangle. */

    if (*incx == 1 && *incy == 1) {
/* 循环处理每一列 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
/* 计算索引位置 */
        i__2 = j;
        i__3 = j;
/* 检查 x[j] 和 y[j] 是否不为零 */
        if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f
            || y[i__3].i != 0.f)) {
/* 计算 y[j] 的共轭，并与 alpha 的乘积赋给 temp1 */
            r_cnjg(&q__2, &y[j]);
            q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i =
                alpha->r * q__2.i + alpha->i * q__2.r;
            temp1.r = q__1.r, temp1.i = q__1.i;
/* 计算 alpha 与 x[j] 的乘积赋给 temp2 */
            i__2 = j;
            q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                .r;
            r_cnjg(&q__1, &q__2);
            temp2.r = q__1.r, temp2.i = q__1.i;
/* 计算矩阵的对角元素 */
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            i__4 = j;
            q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
                q__2.i = x[i__4].r * temp1.i + x[i__4].i *
                temp1.r;
            i__5 = j;
            q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
                q__3.i = y[i__5].r * temp2.i + y[i__5].i *
                temp2.r;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            r__1 = a[i__3].r + q__1.r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
/* 计算矩阵的下三角元素 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * a_dim1;
            i__4 = i__ + j * a_dim1;
            i__5 = i__;
            q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
                q__3.i = x[i__5].r * temp1.i + x[i__5].i *
                temp1.r;
            q__2.r = a[i__4].r + q__3.r, q__2.i = a[i__4].i +
                q__3.i;
            i__6 = i__;
            q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
                q__4.i = y[i__6].r * temp2.i + y[i__6].i *
                temp2.r;
            q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
/* 赋值给矩阵元素 */
            a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L50: */
            }
        } else {
            // 当不满足上述条件时，将对角元素设为实部为alpha.r，虚部为0的值
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            r__1 = a[i__3].r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
        }
/* L60: */
        }
    } else {
        // 当 uplo 不为 'U' 时，即为 'L' 时的处理
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = jx;
        i__3 = jy;
        // 检查 x[jx] 和 y[jy] 是否为非零，以决定是否执行下面的操作
        if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f
            || y[i__3].i != 0.f)) {
            // 计算临时变量 temp1 和 temp2
            r_cnjg(&q__2, &y[jy]);
            q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i =
                alpha->r * q__2.i + alpha->i * q__2.r;
            temp1.r = q__1.r, temp1.i = q__1.i;
            i__2 = jx;
            q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                .r;
            r_cnjg(&q__1, &q__2);
            temp2.r = q__1.r, temp2.i = q__1.i;
            // 更新 a[j][j] 的值
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            i__4 = jx;
            q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
                q__2.i = x[i__4].r * temp1.i + x[i__4].i *
                temp1.r;
            i__5 = jy;
            q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
                q__3.i = y[i__5].r * temp2.i + y[i__5].i *
                temp2.r;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            r__1 = a[i__3].r + q__1.r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
            // 更新 ix 和 iy 的值
            ix = jx;
            iy = jy;
            // 循环处理 a[j+1:n][j] 的元素
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            ix += *incx;
            iy += *incy;
            // 更新 a[i][j] 的值
            i__3 = i__ + j * a_dim1;
            i__4 = i__ + j * a_dim1;
            i__5 = ix;
            q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
                q__3.i = x[i__5].r * temp1.i + x[i__5].i *
                temp1.r;
            q__2.r = a[i__4].r + q__3.r, q__2.i = a[i__4].i +
                q__3.i;
            i__6 = iy;
            q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
                q__4.i = y[i__6].r * temp2.i + y[i__6].i *
                temp2.r;
            q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
            a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L70: */
            }
        } else {
            // 当 x[jx] 和 y[jy] 均为零时，将对角元素设为实部为alpha.r，虚部为0的值
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            r__1 = a[i__3].r;
            a[i__2].r = r__1, a[i__2].i = 0.f;
        }
        // 更新 jx 和 jy 的值
        jx += *incx;
        jy += *incy;
/* L80: */
        }
    }
    }

    // 返回结果
    return 0;

/*     End of CHER2 . */

} /* cher2_ */

/* Subroutine */ int cher2k_(char *uplo, char *trans, integer *n, integer *k,
    singlecomplex *alpha, singlecomplex *a, integer *lda, singlecomplex *b, integer *ldb,
    real *beta, singlecomplex *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3, i__4, i__5, i__6, i__7;
    real r__1;


注释：
    # 声明六个 singlecomplex 类型的变量 q__1 到 q__6
    singlecomplex q__1, q__2, q__3, q__4, q__5, q__6;

    # 声明四个静态局部变量：i__、j、l、info
    # i__: 循环索引
    # j: 循环索引
    # l: 临时变量
    # info: 存储函数调用的返回信息
    static integer i__, j, l, info;

    # 声明两个静态局部变量：temp1 和 temp2，都是 singlecomplex 类型
    # temp1: 临时存储单精度复数值
    # temp2: 临时存储单精度复数值
    static singlecomplex temp1, temp2;

    # 声明外部函数 lsame_
    extern logical lsame_(char *, char *);

    # 声明静态局部变量 nrowa，存储矩阵的行数
    static integer nrowa;

    # 声明静态局部变量 upper，逻辑类型，指示矩阵是否为上三角形式
    static logical upper;

    # 声明外部函数 xerbla_
    extern /* Subroutine */ int xerbla_(char *, integer *);
/*
    Purpose
    =======

    CHER2K  performs one of the hermitian rank 2k operations

       C := alpha*A*conjg( B' ) + conjg( alpha )*B*conjg( A' ) + beta*C,

    or

       C := alpha*conjg( A' )*B + conjg( alpha )*conjg( B' )*A + beta*C,

    where  alpha and beta  are scalars with  beta  real,  C is an  n by n
    hermitian matrix and  A and B  are  n by k matrices in the first case
    and  k by n  matrices in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of  the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'    C := alpha*A*conjg( B' )          +
                                           conjg( alpha )*B*conjg( A' ) +
                                           beta*C.

                TRANS = 'C' or 'c'    C := alpha*conjg( A' )*B          +
                                           conjg( alpha )*conjg( B' )*A +
                                           beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns  of  the  matrices  A and B,  and on  entry  with
             TRANS = 'C' or 'c',  K  specifies  the number of rows of the
             matrices  A and B.  K must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX         .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX          array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.
*/
    # COMPLEX 类型的数组 B，维度为 (LDB, kb)，其中 kb 是 k 当 TRANS = 'N' 或 'n' 时，否则是 n。
    # 在 TRANS = 'N' 或 'n' 时，数组 B 的前 n 行 k 列部分必须包含矩阵 B；否则，前 k 行 n 列部分必须包含矩阵 B。
    # 函数执行后 B 数组保持不变。
    
    # INTEGER 类型的参数 LDB。
    # 在输入时，LDB 指定了数组 B 的第一个维度，与调用程序中声明的维度相同。
    # 当 TRANS = 'N' 或 'n' 时，LDB 必须至少为 max(1, n)；否则，LDB 必须至少为 max(1, k)。
    # 函数执行后 LDB 参数保持不变。
    
    # REAL 类型的参数 BETA。
    # 在输入时，BETA 指定了标量 beta 的值。
    # 函数执行后 BETA 参数保持不变。
    
    # COMPLEX 类型的数组 C，维度为 (LDC, n)。
    # 在输入时，如果 UPLO = 'U' 或 'u'，数组 C 的前 n 行 n 列必须包含上三角部分的厄米特矩阵，且严格下三角部分不被引用。
    # 函数执行后，数组 C 的上三角部分被更新后的矩阵的上三角部分覆盖。
    # 在输入时，如果 UPLO = 'L' 或 'l'，数组 C 的前 n 行 n 列必须包含下三角部分的厄米特矩阵，且严格上三角部分不被引用。
    # 函数执行后，数组 C 的下三角部分被更新后的矩阵的下三角部分覆盖。
    # 注意：对角线元素的虚部不需要设置，假定为零，在函数执行后被设置为零。
    
    # INTEGER 类型的参数 LDC。
    # 在输入时，LDC 指定了数组 C 的第一个维度，与调用程序中声明的维度相同。
    # LDC 必须至少为 max(1, n)。
    # 函数执行后 LDC 参数保持不变。
/*
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    if (lsame_(trans, "N")) {
    nrowa = *n;  // 如果 trans 参数为 "N"，则 nrowa 等于 n
    } else {
    nrowa = *k;  // 否则，nrowa 等于 k
    }
    upper = lsame_(uplo, "U");  // upper 判断 uplo 是否为 "U"，结果存储在 upper 变量中

    info = 0;  // 初始化 info 为 0
    if (! upper && ! lsame_(uplo, "L")) {  // 如果 uplo 不是 "U" 且也不是 "L"
    info = 1;  // 设置 info 为 1
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
        "C")) {  // 否则，如果 trans 不是 "N" 且不是 "C"
    info = 2;  // 设置 info 为 2
    } else if (*n < 0) {  // 否则，如果 n 小于 0
    info = 3;  // 设置 info 为 3
    } else if (*k < 0) {  // 否则，如果 k 小于 0
    info = 4;  // 设置 info 为 4
    } else if (*lda < max(1,nrowa)) {  // 否则，如果 lda 小于 max(1, nrowa)
    info = 7;  // 设置 info 为 7
    } else if (*ldb < max(1,nrowa)) {  // 否则，如果 ldb 小于 max(1, nrowa)
    info = 9;  // 设置 info 为 9
    } else if (*ldc < max(1,*n)) {  // 否则，如果 ldc 小于 max(1, n)
    info = 12;  // 设置 info 为 12
    }
    if (info != 0) {  // 如果 info 不等于 0
    xerbla_("CHER2K", &info);  // 调用错误处理函数 xerbla_，传递错误码 info
    return 0;  // 返回 0
    }

/*     Quick return if possible. */

    if (*n == 0 || (alpha->r == 0.f && alpha->i == 0.f || *k == 0) && *beta ==
         1.f) {  // 如果 n 等于 0 或者 (alpha 的实部和虚部都为 0 或者 k 等于 0) 且 beta 等于 1
    return 0;  // 返回 0
    }

/*     And when  alpha.eq.zero. */

    if (alpha->r == 0.f && alpha->i == 0.f) {  // 如果 alpha 的实部和虚部都为 0
    if (upper) {  // 如果 upper 为真
        if (*beta == 0.f) {  // 如果 beta 等于 0
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 循环 j 从 1 到 n
            i__2 = j;
            for (i__ = 1; i__ <= i__2; ++i__) {  // 循环 i 从 1 到 j
            i__3 = i__ + j * c_dim1;
            c__[i__3].r = 0.f, c__[i__3].i = 0.f;  // 设置 c[i,j] 的实部和虚部为 0
/* L10: */
            }
/* L20: */
        }
        } else {  // 否则，如果 beta 不等于 0
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 循环 j 从 1 到 n
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {  // 循环 i 从 1 到 j-1
            i__3 = i__ + j * c_dim1;
            i__4 = i__ + j * c_dim1;
            q__1.r = *beta * c__[i__4].r, q__1.i = *beta * c__[
                i__4].i;  // 计算 beta * c[i,j] 的复数值
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;  // 设置 c[i,j] 的值
/* L30: */
            }
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = *beta * c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;  // 设置对角线上 c[j,j] 的值
/* L40: */
        }
        }
    } else {  // 否则，如果 upper 不为真
        if (*beta == 0.f) {  // 如果 beta 等于 0
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 循环 j 从 1 到 n
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {  // 循环 i 从 j 到 n
            i__3 = i__ + j * c_dim1;
            c__[i__3].r = 0.f, c__[i__3].i = 0.f;  // 设置 c[i,j] 的实部和虚部为 0
/* L50: */
            }
/* L60: */
        }
        } else {  // 否则，如果 beta 不等于 0
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 循环 j 从 1 到 n
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = *beta * c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;  // 设置对角线上 c[j,j] 的值
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {  // 循环 i 从 j+1 到 n
            i__3 = i__ + j * c_dim1;
            i__4 = i__ + j * c_dim1;
            q__1.r = *beta * c__[i__4].r, q__1.i = *beta * c__[
                i__4].i;  // 计算 beta * c[i,j] 的复数值
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;  // 设置 c[i,j] 的值
/* L70: */
            }
/* L80: */
        }
        }
    }
    return 0;  // 返回 0
    }

/*     Start the operations. */

    if (lsame_(trans, "N")) {  // 如果 trans 参数为 "N"
/*
          Form  C := alpha*A*conjg( B' ) + conjg( alpha )*B*conjg( A' ) +
                     C.
*/

if (upper) {
    // 针对上三角部分的循环，遍历每列 j
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // 如果 beta 等于 0，将 C 的第 j 列及其以上部分全部置零
        if (*beta == 0.f) {
            i__2 = j;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * c_dim1;
                c__[i__3].r = 0.f, c__[i__3].i = 0.f;
            }
        } else if (*beta != 1.f) {
            // 如果 beta 不等于 1，对 C 的第 j 列进行处理
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算 beta * C(i, j)
                i__3 = i__ + j * c_dim1;
                i__4 = i__ + j * c_dim1;
                q__1.r = *beta * c__[i__4].r, q__1.i = *beta * c__[i__4].i;
                c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
            // 处理 C 的第 j 行 j 列处的元素
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = *beta * c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        } else {
            // 如果 beta 等于 1，处理 C 的第 j 行 j 列处的元素
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        }
        // 处理 A 和 B 的列交换操作
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            // 计算 alpha * conjg(B(j, l))
            i__3 = j + l * a_dim1;
            i__4 = j + l * b_dim1;
            if (a[i__3].r != 0.f || a[i__3].i != 0.f || (b[i__4].r != 0.f || b[i__4].i != 0.f)) {
                r_cnjg(&q__2, &b[j + l * b_dim1]);
                q__1.r = alpha->r * q__2.r - alpha->i * q__2.i,
                    q__1.i = alpha->r * q__2.i + alpha->i * q__2.r;
                temp1.r = q__1.r, temp1.i = q__1.i;
                // 计算 alpha * conjg(A(j, l))
                i__3 = j + l * a_dim1;
                q__2.r = alpha->r * a[i__3].r - alpha->i * a[i__3].i,
                    q__2.i = alpha->r * a[i__3].i + alpha->i * a[i__3].r;
                r_cnjg(&q__1, &q__2);
                temp2.r = q__1.r, temp2.i = q__1.i;
                // 更新 C 的第 i 行 j 列的元素
                i__3 = j - 1;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    i__4 = i__ + j * c_dim1;
                    i__5 = i__ + j * c_dim1;
                    i__6 = i__ + l * a_dim1;
                    q__3.r = a[i__6].r * temp1.r - a[i__6].i * temp1.i,
                        q__3.i = a[i__6].r * temp1.i + a[i__6].i * temp1.r;
                    q__2.r = c__[i__5].r + q__3.r, q__2.i = c__[i__5].i + q__3.i;
                    i__7 = i__ + l * b_dim1;
                    q__4.r = b[i__7].r * temp2.r - b[i__7].i * temp2.i,
                        q__4.i = b[i__7].r * temp2.i + b[i__7].i * temp2.r;
                    q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
                    c__[i__4].r = q__1.r, c__[i__4].i = q__1.i;
                }
            }
        }
    }
}
/* L110: */
            }
            // 计算矩阵乘法的一部分，更新 C 矩阵的某些元素
            i__3 = j + j * c_dim1;
            i__4 = j + j * c_dim1;
            i__5 = j + l * a_dim1;
            // 计算 A 矩阵和临时变量 temp1 的乘积，并将结果存储在 q__2 中
            q__2.r = a[i__5].r * temp1.r - a[i__5].i * temp1.i,
                q__2.i = a[i__5].r * temp1.i + a[i__5].i *
                temp1.r;
            i__6 = j + l * b_dim1;
            // 计算 B 矩阵和临时变量 temp2 的乘积，并将结果存储在 q__3 中
            q__3.r = b[i__6].r * temp2.r - b[i__6].i * temp2.i,
                q__3.i = b[i__6].r * temp2.i + b[i__6].i *
                temp2.r;
            // 将 q__2 和 q__3 相加得到最终的乘积结果，并将结果加到 C 矩阵的相应位置
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            r__1 = c__[i__4].r + q__1.r;
            c__[i__3].r = r__1, c__[i__3].i = 0.f;
            }
/* L120: */
        }
/* L130: */
        }
    } else {
        // 当 beta 不等于 0 时的处理逻辑
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        if (*beta == 0.f) {
            // 当 beta 等于 0 时，对 C 矩阵的特定区域进行清零处理
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            c__[i__3].r = 0.f, c__[i__3].i = 0.f;
/* L140: */
            }
        } else if (*beta != 1.f) {
            // 当 beta 不等于 1 时，对 C 矩阵的特定区域进行乘法操作
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            i__4 = i__ + j * c_dim1;
            // 将 C 矩阵的某一元素乘以 beta，并更新到相同位置
            q__1.r = *beta * c__[i__4].r, q__1.i = *beta * c__[
                i__4].i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L150: */
            }
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = *beta * c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        } else {
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            i__3 = j + l * a_dim1;
            i__4 = j + l * b_dim1;
            if (a[i__3].r != 0.f || a[i__3].i != 0.f || (b[i__4].r !=
                0.f || b[i__4].i != 0.f)) {
                /* Calculate complex conjugate of b[j+l*b_dim1] */
                r_cnjg(&q__2, &b[j + l * b_dim1]);
                /* Perform complex multiplication alpha * conjg(b[j+l*b_dim1]) */
                q__1.r = alpha->r * q__2.r - alpha->i * q__2.i,
                    q__1.i = alpha->r * q__2.i + alpha->i *
                    q__2.r;
                temp1.r = q__1.r, temp1.i = q__1.i;
                /* Perform complex conjugate of a[j+l*a_dim1] */
                i__3 = j + l * a_dim1;
                q__2.r = alpha->r * a[i__3].r - alpha->i * a[i__3].i,
                    q__2.i = alpha->r * a[i__3].i + alpha->i * a[
                    i__3].r;
                r_cnjg(&q__1, &q__2);
                temp2.r = q__1.r, temp2.i = q__1.i;
                i__3 = *n;
                for (i__ = j + 1; i__ <= i__3; ++i__) {
                    /* Perform matrix multiplication for each element of C */
                    i__4 = i__ + j * c_dim1;
                    i__5 = i__ + j * c_dim1;
                    i__6 = i__ + l * a_dim1;
                    q__3.r = a[i__6].r * temp1.r - a[i__6].i *
                        temp1.i, q__3.i = a[i__6].r * temp1.i + a[
                        i__6].i * temp1.r;
                    q__2.r = c__[i__5].r + q__3.r, q__2.i = c__[i__5]
                        .i + q__3.i;
                    i__7 = i__ + l * b_dim1;
                    q__4.r = b[i__7].r * temp2.r - b[i__7].i *
                        temp2.i, q__4.i = b[i__7].r * temp2.i + b[
                        i__7].i * temp2.r;
                    q__1.r = q__2.r + q__4.r, q__1.i = q__2.i +
                        q__4.i;
                    c__[i__4].r = q__1.r, c__[i__4].i = q__1.i;
/* L160: */
                }
                i__3 = j + j * c_dim1;
                i__4 = j + j * c_dim1;
                i__5 = j + l * a_dim1;
                q__2.r = a[i__5].r * temp1.r - a[i__5].i * temp1.i,
                    q__2.i = a[i__5].r * temp1.i + a[i__5].i *
                    temp1.r;
                i__6 = j + l * b_dim1;
                q__3.r = b[i__6].r * temp2.r - b[i__6].i * temp2.i,
                    q__3.i = b[i__6].r * temp2.i + b[i__6].i *
                    temp2.r;
                q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
                r__1 = c__[i__4].r + q__1.r;
                c__[i__3].r = r__1, c__[i__3].i = 0.f;
            }
/* L170: */
        }
/* L180: */
        }
    }
    } else {

/*
          Form  C := alpha*conjg( A' )*B + conjg( alpha )*conjg( B' )*A +
                     C.
*/
    // 如果 upper 为真，则执行以下循环
    if (upper) {
        // 循环遍历 j 从 1 到 *n
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 计算当前 i__2 的值为 j
            i__2 = j;
            // 循环遍历 i__ 从 1 到 i__2
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 初始化 temp1 和 temp2 为复数 0
                temp1.r = 0.f, temp1.i = 0.f;
                temp2.r = 0.f, temp2.i = 0.f;
                // 循环遍历 l 从 1 到 *k
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    // 计算共轭复数 q__3 为 a[l + i__ * a_dim1] 的共轭
                    r_cnjg(&q__3, &a[l + i__ * a_dim1]);
                    // 计算 q__2 为 q__3 与 b[l + j * b_dim1] 的乘积
                    i__4 = l + j * b_dim1;
                    q__2.r = q__3.r * b[i__4].r - q__3.i * b[i__4].i,
                        q__2.i = q__3.r * b[i__4].i + q__3.i * b[i__4].r;
                    // 计算 q__1 为 temp1 与 q__2 的和
                    q__1.r = temp1.r + q__2.r, q__1.i = temp1.i + q__2.i;
                    temp1.r = q__1.r, temp1.i = q__1.i;
                    // 计算共轭复数 q__3 为 b[l + i__ * b_dim1] 的共轭
                    r_cnjg(&q__3, &b[l + i__ * b_dim1]);
                    // 计算 q__2 为 q__3 与 a[l + j * a_dim1] 的乘积
                    i__4 = l + j * a_dim1;
                    q__2.r = q__3.r * a[i__4].r - q__3.i * a[i__4].i,
                        q__2.i = q__3.r * a[i__4].i + q__3.i * a[i__4].r;
                    // 计算 q__1 为 temp2 与 q__2 的和
                    q__1.r = temp2.r + q__2.r, q__1.i = temp2.i + q__2.i;
                    temp2.r = q__1.r, temp2.i = q__1.i;
/* L190: */
            }
            ! 检查 i__ 是否等于 j
            if (i__ == j) {
                ! 检查 beta 是否为 0
                if (*beta == 0.f) {
                    ! 计算 c[j,j] 的更新值，当 beta 为 0 时
                    i__3 = j + j * c_dim1;
                    q__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    r_cnjg(&q__4, alpha);
                    q__3.r = q__4.r * temp2.r - q__4.i * temp2.i,
                        q__3.i = q__4.r * temp2.i + q__4.i *
                        temp2.r;
                    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
                        q__3.i;
                    r__1 = q__1.r;
                    c__[i__3].r = r__1, c__[i__3].i = 0.f;
                } else {
                    ! 计算 c[j,j] 的更新值，当 beta 不为 0 时
                    i__3 = j + j * c_dim1;
                    i__4 = j + j * c_dim1;
                    q__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    r_cnjg(&q__4, alpha);
                    q__3.r = q__4.r * temp2.r - q__4.i * temp2.i,
                        q__3.i = q__4.r * temp2.i + q__4.i *
                        temp2.r;
                    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
                        q__3.i;
                    r__1 = *beta * c__[i__4].r + q__1.r;
                    c__[i__3].r = r__1, c__[i__3].i = 0.f;
                }
            } else {
                ! 计算 c[i__, j] 的更新值，当 i__ 不等于 j 时
                if (*beta == 0.f) {
                    i__3 = i__ + j * c_dim1;
                    q__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    r_cnjg(&q__4, alpha);
                    q__3.r = q__4.r * temp2.r - q__4.i * temp2.i,
                        q__3.i = q__4.r * temp2.i + q__4.i *
                        temp2.r;
                    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
                        q__3.i;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                } else {
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    q__3.r = *beta * c__[i__4].r, q__3.i = *beta *
                        c__[i__4].i;
                    q__4.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__4.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    q__2.r = q__3.r + q__4.r, q__2.i = q__3.i +
                        q__4.i;
                    r_cnjg(&q__6, alpha);
                    q__5.r = q__6.r * temp2.r - q__6.i * temp2.i,
                        q__5.i = q__6.r * temp2.i + q__6.i *
                        temp2.r;
                    q__1.r = q__2.r + q__5.r, q__1.i = q__2.i +
                        q__5.i;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                }
            }
/* L200: */
        }
/* L210: */
        }
    } else {
        // 循环：对每个列进行处理
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 循环：对每个行进行处理
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
                // 初始化临时变量 temp1 和 temp2
                temp1.r = 0.f, temp1.i = 0.f;
                temp2.r = 0.f, temp2.i = 0.f;
                // 循环：对每个元素进行处理
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    // 计算矩阵 a 的共轭转置与矩阵 b 的乘积，并累加到 temp1
                    r_cnjg(&q__3, &a[l + i__ * a_dim1]);
                    i__4 = l + j * b_dim1;
                    q__2.r = q__3.r * b[i__4].r - q__3.i * b[i__4].i,
                        q__2.i = q__3.r * b[i__4].i + q__3.i * b[i__4].r;
                    q__1.r = temp1.r + q__2.r, q__1.i = temp1.i + q__2.i;
                    temp1.r = q__1.r, temp1.i = q__1.i;
                    // 计算矩阵 b 的共轭转置与矩阵 a 的乘积，并累加到 temp2
                    r_cnjg(&q__3, &b[l + i__ * b_dim1]);
                    i__4 = l + j * a_dim1;
                    q__2.r = q__3.r * a[i__4].r - q__3.i * a[i__4].i,
                        q__2.i = q__3.r * a[i__4].i + q__3.i * a[i__4].r;
                    q__1.r = temp2.r + q__2.r, q__1.i = temp2.i + q__2.i;
                    temp2.r = q__1.r, temp2.i = q__1.i;
/* L220: */
            }
            // 如果 i__ 等于 j
            if (i__ == j) {
                // 如果 beta 等于 0
                if (*beta == 0.f) {
                    // 计算 alpha * temp1，并存储在 q__2 中
                    q__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    // 计算 alpha 的共轭乘积，并与 temp2 相乘，存储在 q__3 中
                    r_cnjg(&q__4, alpha);
                    q__3.r = q__4.r * temp2.r - q__4.i * temp2.i,
                        q__3.i = q__4.r * temp2.i + q__4.i *
                        temp2.r;
                    // 将 q__2 和 q__3 相加，存储在 q__1 中
                    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
                        q__3.i;
                    // 将结果存储在 c__[j + j * c_dim1] 中
                    r__1 = q__1.r;
                    c__[i__3].r = r__1, c__[i__3].i = 0.f;
                } else {
                    // 计算 alpha * temp1，并存储在 q__2 中
                    q__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    // 计算 alpha 的共轭乘积，并与 temp2 相乘，存储在 q__3 中
                    r_cnjg(&q__4, alpha);
                    q__3.r = q__4.r * temp2.r - q__4.i * temp2.i,
                        q__3.i = q__4.r * temp2.i + q__4.i *
                        temp2.r;
                    // 将 q__2 和 q__3 相加，再与 beta 乘积后加到 c__[j + j * c_dim1] 中
                    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
                        q__3.i;
                    r__1 = *beta * c__[i__4].r + q__1.r;
                    c__[i__3].r = r__1, c__[i__3].i = 0.f;
                }
            } else {
                // 如果 beta 等于 0
                if (*beta == 0.f) {
                    // 计算 alpha * temp1，并存储在 q__2 中
                    q__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    // 计算 alpha 的共轭乘积，并与 temp2 相乘，存储在 q__3 中
                    r_cnjg(&q__4, alpha);
                    q__3.r = q__4.r * temp2.r - q__4.i * temp2.i,
                        q__3.i = q__4.r * temp2.i + q__4.i *
                        temp2.r;
                    // 将 q__2 和 q__3 相加，存储在 c__[i__ + j * c_dim1] 中
                    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
                        q__3.i;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                } else {
                    // 计算 alpha * temp1，并存储在 q__4 中
                    r_cnjg(&q__4, alpha);
                    q__3.r = *beta * c__[i__4].r, q__3.i = *beta *
                        c__[i__4].i;
                    q__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        q__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    q__1.r = q__3.r + q__2.r, q__1.i = q__3.i +
                        q__2.i;
                    // 计算 alpha 的共轭乘积，并与 temp2 相乘，存储在 q__5 中
                    r_cnjg(&q__6, alpha);
                    q__5.r = q__6.r * temp2.r - q__6.i * temp2.i,
                        q__5.i = q__6.r * temp2.i + q__6.i *
                        temp2.r;
                    // 将 q__1 和 q__5 相加，存储在 c__[i__ + j * c_dim1] 中
                    q__1.r = q__1.r + q__5.r, q__1.i = q__1.i +
                        q__5.i;
                    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
                }
            }
/* L230: */
        }
/* L240: */
        }
    }
    }

    // 返回值为 0，表示成功执行
    return 0;

/*     End of CHER2K. */

} /* cher2k_ */

/* Subroutine */ int cherk_(char *uplo, char *trans, integer *n, integer *k,
    real *alpha, singlecomplex *a, integer *lda, real *beta, singlecomplex *c__,
    integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3, i__4, i__5,
        i__6;
    real r__1;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, j, l, info;  // 声明整数变量 i, j, l, info
    static singlecomplex temp;  // 声明复数变量 temp
    extern logical lsame_(char *, char *);  // 外部函数 lsame_ 的声明，用于比较两个字符是否相同
    static integer nrowa;  // 声明整数变量 nrowa，用于存储矩阵 A 的行数
    static real rtemp;  // 声明实数变量 rtemp
    static logical upper;  // 声明逻辑变量 upper，用于指示矩阵是否为上三角形式
    extern /* Subroutine */ int xerbla_(char *, integer *);  // 外部子程序 xerbla_ 的声明，用于处理错误信息
/*
    Purpose
    =======

    CHERK  performs one of the hermitian rank k operations

       C := alpha*A*conjg( A' ) + beta*C,

    or

       C := alpha*conjg( A' )*A + beta*C,

    where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
    matrix and  A  is an  n by k  matrix in the  first case and a  k by n
    matrix in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of  the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   C := alpha*A*conjg( A' ) + beta*C.

                TRANS = 'C' or 'c'   C := alpha*conjg( A' )*A + beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns   of  the   matrix   A,   and  on   entry   with
             TRANS = 'C' or 'c',  K  specifies  the number of rows of the
             matrix A.  K must be at least zero.
             Unchanged on exit.

    ALPHA  - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX          array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.

    BETA   - REAL            .
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.
*/
    C      - COMPLEX          array of DIMENSION ( LDC, n ).
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  hermitian matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  hermitian matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.
             Note that the imaginary parts of the diagonal elements need
             not be set,  they are assumed to be zero,  and on exit they
             are set to zero.

- COMPLEX类型数组C，维度为(LDC, n)。
- 如果UPLO = 'U'或'u'，则在进入时，数组C的前n行n列必须包含厄米特矩阵的上三角部分，C的严格下三角部分不被引用。退出时，数组C的上三角部分被更新后的矩阵的上三角部分覆盖。
- 如果UPLO = 'L'或'l'，则在进入时，数组C的前n行n列必须包含厄米特矩阵的下三角部分，C的严格上三角部分不被引用。退出时，数组C的下三角部分被更新后的矩阵的下三角部分覆盖。
- 注意对角元素的虚部不需要设置，假定为零，退出时将其设置为零。


    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, n ).
             Unchanged on exit.

- 整数LDC。
- 进入时，LDC指定了调用程序或子程序中声明的C的第一个维度。LDC必须至少为max(1, n)。
- 退出时保持不变。


    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

    -- Modified 8-Nov-93 to set C(J,J) to REAL( C(J,J) ) when BETA = 1.
       Ed Anderson, Cray Research Inc.

    =====================================================================

- 更多细节
- Level 3 Blas例程。
- 1989年2月8日编写。
  - Jack Dongarra，阿贡国家实验室。
  - Iain Duff，AERE Harwell。
  - Jeremy Du Croz，Numerical Algorithms Group Ltd。
  - Sven Hammarling，Numerical Algorithms Group Ltd。
- 1993年11月8日修改，当BETA = 1时将C(J,J)设置为REAL(C(J,J))。
  - Ed Anderson，Cray Research Inc。


       Test the input parameters.

- 测试输入参数。
    /* 参数调整 */
    a_dim1 = *lda;  // 获取数组 a 的第一维度长度
    a_offset = 1 + a_dim1;  // 计算 a 数组的偏移量
    a -= a_offset;  // 调整 a 数组的指针位置
    c_dim1 = *ldc;  // 获取数组 c 的第一维度长度
    c_offset = 1 + c_dim1;  // 计算 c 数组的偏移量
    c__ -= c_offset;  // 调整 c 数组的指针位置

    /* 函数体 */
    if (lsame_(trans, "N")) {  // 如果 trans 是 "N"
        nrowa = *n;  // nrowa 取 n 的值
    } else {
        nrowa = *k;  // 否则 nrowa 取 k 的值
    }
    upper = lsame_(uplo, "U");  // upper 标记是否为上三角矩阵

    info = 0;  // 初始化 info 为 0
    if (! upper && ! lsame_(uplo, "L")) {  // 如果不是上三角也不是下三角
        info = 1;  // 设置 info 为 1
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "C")) {  // 如果 trans 不是 "N" 也不是 "C"
        info = 2;  // 设置 info 为 2
    } else if (*n < 0) {  // 如果 n 小于 0
        info = 3;  // 设置 info 为 3
    } else if (*k < 0) {  // 如果 k 小于 0
        info = 4;  // 设置 info 为 4
    } else if (*lda < max(1,nrowa)) {  // 如果 lda 小于 1 和 nrowa 中的最大值
        info = 7;  // 设置 info 为 7
    } else if (*ldc < max(1,*n)) {  // 如果 ldc 小于 1 和 n 中的最大值
        info = 10;  // 设置 info 为 10
    }
    if (info != 0) {  // 如果 info 不为 0
        xerbla_("CHERK ", &info);  // 调用错误处理函数 xerbla_
        return 0;  // 返回 0
    }

/*     若可能尽早返回. */

    if (*n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {  // 如果 n 为 0 或者 alpha 为 0 且 k 为 0 且 beta 为 1
        return 0;  // 直接返回 0
    }

/*     当 alpha 等于零时. */

    if (*alpha == 0.f) {  // 如果 alpha 等于 0
        if (upper) {  // 如果是上三角矩阵
            if (*beta == 0.f) {  // 如果 beta 等于 0
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0.f, c__[i__3].i = 0.f;
/* L10: */          // 循环结束
                    }
/* L20: */          // 循环结束
                }
            } else {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j - 1;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        q__1.r = *beta * c__[i__4].r, q__1.i = *beta * c__[i__4].i;
                        c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L30: */          // 循环结束
                    }
                    i__2 = j + j * c_dim1;
                    i__3 = j + j * c_dim1;
                    r__1 = *beta * c__[i__3].r;
                    c__[i__2].r = r__1, c__[i__2].i = 0.f;
/* L40: */          // 循环结束
                }
            }
        } else {  // 如果是下三角矩阵
            if (*beta == 0.f) {  // 如果 beta 等于 0
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0.f, c__[i__3].i = 0.f;
/* L50: */          // 循环结束
                    }
/* L60: */          // 循环结束
                }
            } else {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j + j * c_dim1;
                    i__3 = j + j * c_dim1;
                    r__1 = *beta * c__[i__3].r;
                    c__[i__2].r = r__1, c__[i__2].i = 0.f;
                    i__2 = *n;
                    for (i__ = j + 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        q__1.r = *beta * c__[i__4].r, q__1.i = *beta * c__[i__4].i;
                        c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L70: */          // 循环结束
                    }
/* L80: */          // 循环结束
                }
            }
        }
        return 0;  // 返回 0
    }

/*     开始运算. */

    if (lsame_(trans, "N")) {  // 如果 trans 是 "N"

/*        计算 C := alpha*A*conjg( A' ) + beta*C. */
    # 如果 upper 为真，则执行以下循环，遍历矩阵 C 的上三角部分
    if (upper) {
        # 设置循环上限为 n 的值
        i__1 = *n;
        # 开始遍历 j 从 1 到 i__1
        for (j = 1; j <= i__1; ++j) {
            # 如果 beta 的值为 0.0，则执行以下操作
            if (*beta == 0.f) {
                # 设置循环上限为 j 的值
                i__2 = j;
                # 开始遍历 i 从 1 到 i__2
                for (i__ = 1; i__ <= i__2; ++i__) {
                    # 计算矩阵 C 中的元素索引 i + j * c_dim1
                    i__3 = i__ + j * c_dim1;
                    # 将该元素设为复数 0 + 0i
                    c__[i__3].r = 0.f, c__[i__3].i = 0.f;
/* L90: */
            }
        } else if (*beta != 1.f) {
            // 当 beta 不等于 1 时，对 C 的前 j 列进行更新
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算 C 的元素乘以 beta
                i__3 = i__ + j * c_dim1;
                i__4 = i__ + j * c_dim1;
                q__1.r = *beta * c__[i__4].r, q__1.i = *beta * c__[i__4].i;
                c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L100: */
            }
            // 计算 C 的对角线元素乘以 beta
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = *beta * c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        } else {
            // 当 beta 等于 1 时，仅更新 C 的对角线元素
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        }
        // 对 A 的第 j 列进行处理，更新 C 的前 k 行
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            i__3 = j + l * a_dim1;
            // 检查 A 的元素是否为非零
            if (a[i__3].r != 0.f || a[i__3].i != 0.f) {
                // 计算 alpha 乘以 A 的第 j 列的共轭转置
                r_cnjg(&q__2, &a[j + l * a_dim1]);
                q__1.r = *alpha * q__2.r, q__1.i = *alpha * q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
                // 更新 C 的前 j 列的非对角线元素
                i__3 = j - 1;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    i__4 = i__ + j * c_dim1;
                    i__5 = i__ + j * c_dim1;
                    i__6 = i__ + l * a_dim1;
                    q__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                        q__2.i = temp.r * a[i__6].i + temp.i * a[i__6].r;
                    q__1.r = c__[i__5].r + q__2.r, q__1.i = c__[i__5].i + q__2.i;
                    c__[i__4].r = q__1.r, c__[i__4].i = q__1.i;
/* L110: */
                }
                // 更新 C 的对角线元素
                i__3 = j + j * c_dim1;
                i__4 = j + j * c_dim1;
                i__5 = i__ + l * a_dim1;
                q__1.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    q__1.i = temp.r * a[i__5].i + temp.i * a[i__5].r;
                r__1 = c__[i__4].r + q__1.r;
                c__[i__3].r = r__1, c__[i__3].i = 0.f;
            }
/* L120: */
        }
/* L130: */
        }
    } else {
        // beta 为零时，将 C 的所有元素置零
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
                i__3 = i__ + j * c_dim1;
                c__[i__3].r = 0.f, c__[i__3].i = 0.f;
/* L140: */
            }
        }
    }
/* L150: 结束当前的 if-else 块 */
            }
        } else {
            /* 设置对角元素为实部，虚部为0 */
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        }
        /* 对每一列进行循环 */
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            /* 检查当前列的非零元素 */
            i__3 = j + l * a_dim1;
            if (a[i__3].r != 0.f || a[i__3].i != 0.f) {
                /* 计算 alpha 乘以共轭转置的结果 */
                r_cnjg(&q__2, &a[j + l * a_dim1]);
                q__1.r = *alpha * q__2.r, q__1.i = *alpha * q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
                /* 更新对角线元素 */
                i__3 = j + j * c_dim1;
                i__4 = j + j * c_dim1;
                i__5 = j + l * a_dim1;
                q__1.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    q__1.i = temp.r * a[i__5].i + temp.i * a[i__5].r;
                r__1 = c__[i__4].r + q__1.r;
                c__[i__3].r = r__1, c__[i__3].i = 0.f;
                /* 更新非对角线元素 */
                i__3 = *n;
                for (i__ = j + 1; i__ <= i__3; ++i__) {
                    i__4 = i__ + j * c_dim1;
                    i__5 = i__ + j * c_dim1;
                    i__6 = i__ + l * a_dim1;
                    q__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                        q__2.i = temp.r * a[i__6].i + temp.i * a[i__6].r;
                    q__1.r = c__[i__5].r + q__2.r, q__1.i = c__[i__5].i + q__2.i;
                    c__[i__4].r = q__1.r, c__[i__4].i = q__1.i;
/* L160: 更新非对角线元素后的标签 */
                }
            }
/* L170: 更新对角线元素后的标签 */
        }
/* L180: 处理完当前列后的标签 */
        }
    }
    } else {

/*        Form  C := alpha*conjg( A' )*A + beta*C. */

    if (upper) {
        /* 处理上三角矩阵的情况，对每一列进行循环 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        /* 对每一列的上三角部分的每一个元素进行循环 */
        i__2 = j - 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
            /* 初始化临时变量 */
            temp.r = 0.f, temp.i = 0.f;
            /* 对每一个 k 进行循环 */
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            /* 计算 alpha*conjg( A' )*A 的每个元素 */
            r_cnjg(&q__3, &a[l + i__ * a_dim1]);
            i__4 = l + j * a_dim1;
            q__2.r = q__3.r * a[i__4].r - q__3.i * a[i__4].i,
                q__2.i = q__3.r * a[i__4].i + q__3.i * a[i__4].r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            temp.r = q__1.r, temp.i = q__1.i;
/* L190: 更新临时变量后的标签 */
            }
            /* 根据 beta 值更新 C 的元素 */
            if (*beta == 0.f) {
            i__3 = i__ + j * c_dim1;
            q__1.r = *alpha * temp.r, q__1.i = *alpha * temp.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            q__2.r = *alpha * temp.r, q__2.i = *alpha * temp.i;
            i__4 = i__ + j * c_dim1;
            q__3.r = *beta * c__[i__4].r, q__3.i = *beta * c__[i__4].i;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
/* L200: 更新 C 的元素后的标签 */
        }
        }
    }
/* L200: */
        }
        rtemp = 0.f;
        i__2 = *k;
        // 循环计算当前列的乘积和
        for (l = 1; l <= i__2; ++l) {
            // 计算共轭乘积
            r_cnjg(&q__3, &a[l + j * a_dim1]);
            i__3 = l + j * a_dim1;
            q__2.r = q__3.r * a[i__3].r - q__3.i * a[i__3].i, q__2.i =
                 q__3.r * a[i__3].i + q__3.i * a[i__3].r;
            q__1.r = rtemp + q__2.r, q__1.i = q__2.i;
            // 更新当前列的乘积和
            rtemp = q__1.r;
/* L210: */
        }
        // 根据 beta 值进行处理
        if (*beta == 0.f) {
            // 当 beta 为 0 时，直接赋值 alpha 乘以 rtemp 到 C 矩阵对角线元素
            i__2 = j + j * c_dim1;
            r__1 = *alpha * rtemp;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        } else {
            // 当 beta 不为 0 时，将 alpha 乘以 rtemp 加上 beta 乘以原 C 矩阵对角线元素赋值到 C 矩阵对角线元素
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = *alpha * rtemp + *beta * c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        }
/* L220: */
        }
    } else {
        // 处理非对称情况下的矩阵乘法
        i__1 = *n;
        // 遍历所有列
        for (j = 1; j <= i__1; ++j) {
        rtemp = 0.f;
        i__2 = *k;
        // 计算当前列的乘积和
        for (l = 1; l <= i__2; ++l) {
            // 计算共轭乘积
            r_cnjg(&q__3, &a[l + j * a_dim1]);
            i__3 = l + j * a_dim1;
            q__2.r = q__3.r * a[i__3].r - q__3.i * a[i__3].i, q__2.i =
                 q__3.r * a[i__3].i + q__3.i * a[i__3].r;
            q__1.r = rtemp + q__2.r, q__1.i = q__2.i;
            // 更新当前列的乘积和
            rtemp = q__1.r;
/* L230: */
        }
        // 根据 beta 值进行处理
        if (*beta == 0.f) {
            // 当 beta 为 0 时，直接赋值 alpha 乘以 rtemp 到 C 矩阵对角线元素
            i__2 = j + j * c_dim1;
            r__1 = *alpha * rtemp;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        } else {
            // 当 beta 不为 0 时，将 alpha 乘以 rtemp 加上 beta 乘以原 C 矩阵对角线元素赋值到 C 矩阵对角线元素
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            r__1 = *alpha * rtemp + *beta * c__[i__3].r;
            c__[i__2].r = r__1, c__[i__2].i = 0.f;
        }
        // 处理非对角线元素
        i__2 = *n;
        for (i__ = j + 1; i__ <= i__2; ++i__) {
            temp.r = 0.f, temp.i = 0.f;
            i__3 = *k;
            // 计算非对角线乘积和
            for (l = 1; l <= i__3; ++l) {
            // 计算共轭乘积
            r_cnjg(&q__3, &a[l + i__ * a_dim1]);
            i__4 = l + j * a_dim1;
            q__2.r = q__3.r * a[i__4].r - q__3.i * a[i__4].i,
                q__2.i = q__3.r * a[i__4].i + q__3.i * a[i__4]
                .r;
            q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
            // 更新非对角线乘积和
            temp.r = q__1.r, temp.i = q__1.i;
/* L240: */
            }
            // 根据 beta 值进行处理
            if (*beta == 0.f) {
            // 当 beta 为 0 时，直接赋值 alpha 乘以 temp 到 C 矩阵元素
            i__3 = i__ + j * c_dim1;
            q__1.r = *alpha * temp.r, q__1.i = *alpha * temp.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            } else {
            // 当 beta 不为 0 时，将 alpha 乘以 temp 加上 beta 乘以原 C 矩阵元素赋值到 C 矩阵元素
            i__3 = i__ + j * c_dim1;
            q__2.r = *alpha * temp.r, q__2.i = *alpha * temp.i;
            i__4 = i__ + j * c_dim1;
            q__3.r = *beta * c__[i__4].r, q__3.i = *beta * c__[
                i__4].i;
            q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
            c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
            }
/* L250: */
        }
/* L260: */
        }
    }
    }

    return 0;

/*     End of CHERK . */

} /* cherk_ */

/* Subroutine */ int cscal_(integer *n, singlecomplex *ca, singlecomplex *cx, integer *
    incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    声明一个名为 q__1 的单精度复数变量

    /* Local variables */
    声明本地变量部分：
    静态整型变量 i__，用于循环计数
    静态整型变量 nincx，用于表示增量
/*
    Purpose
    =======

       CSCAL scales a vector by a constant.

    Further Details
    ===============

       jack dongarra, linpack,  3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

/* Parameter adjustments */
--cx;  // Adjust the base of array 'cx' to account for 1-based indexing in Fortran

/* Function Body */
if (*n <= 0 || *incx <= 0) {
return 0;  // Return early if vector length or increment is non-positive
}
if (*incx == 1) {
goto L20;  // Jump to label L20 if the increment is 1
}

/*        code for increment not equal to 1 */

nincx = *n * *incx;  // Calculate the total number of elements to process
i__1 = nincx;
i__2 = *incx;
for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
i__3 = i__;
i__4 = i__;
q__1.r = ca->r * cx[i__4].r - ca->i * cx[i__4].i, q__1.i = ca->r * cx[
    i__4].i + ca->i * cx[i__4].r;
cx[i__3].r = q__1.r, cx[i__3].i = q__1.i;
/* L10: */
}
return 0;

/*        code for increment equal to 1 */

L20:
i__2 = *n;
for (i__ = 1; i__ <= i__2; ++i__) {
i__1 = i__;
i__3 = i__;
q__1.r = ca->r * cx[i__3].r - ca->i * cx[i__3].i, q__1.i = ca->r * cx[
    i__3].i + ca->i * cx[i__3].r;
cx[i__1].r = q__1.r, cx[i__1].i = q__1.i;
/* L30: */
}
return 0;
} /* cscal_ */

/* Subroutine */ int csrot_(integer *n, singlecomplex *cx, integer *incx, singlecomplex *
    cy, integer *incy, real *c__, real *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, ix, iy;
    static singlecomplex ctemp;


/*
    Purpose
    =======

    CSROT applies a plane rotation, where the cos and sin (c and s) are real
    and the vectors cx and cy are singlecomplex.
    jack dongarra, linpack, 3/11/78.

    Arguments
    ==========

    N        (input) INTEGER
             On entry, N specifies the order of the vectors cx and cy.
             N must be at least zero.
             Unchanged on exit.

    CX       (input) COMPLEX array, dimension at least
             ( 1 + ( N - 1 )*abs( INCX ) ).
             Before entry, the incremented array CX must contain the n
             element vector cx. On exit, CX is overwritten by the updated
             vector cx.

    INCX     (input) INTEGER
             On entry, INCX specifies the increment for the elements of
             CX. INCX must not be zero.
             Unchanged on exit.

    CY       (input) COMPLEX array, dimension at least
             ( 1 + ( N - 1 )*abs( INCY ) ).
             Before entry, the incremented array CY must contain the n
             element vector cy. On exit, CY is overwritten by the updated
             vector cy.

    INCY     (input) INTEGER
             On entry, INCY specifies the increment for the elements of
             CY. INCY must not be zero.
             Unchanged on exit.

    C        (input) REAL
             On entry, C specifies the cosine, cos.
             Unchanged on exit.
*/
    S        (input) REAL
             On entry, S specifies the sine, sin.
             Unchanged on exit.
/* Parameter adjustments */
/* 参数调整 */

    --cy;
    --cx;
    /* 数组下标偏移，将指针向前移动 */

    /* Function Body */
    /* 函数主体 */
    if (*n <= 0) {
    return 0;
    }
    /* 如果 n 小于等于 0，返回 0 */

    if (*incx == 1 && *incy == 1) {
    goto L20;
    }
    /* 如果 incx 等于 1 且 incy 等于 1，跳转到标签 L20 */

/*
          code for unequal increments or equal increments not equal
            to 1
*/
/*
          不等增量或者等增量不等于 1 的情况下的代码
*/

    ix = 1;
    iy = 1;
    /* 初始化 ix 和 iy */

    if (*incx < 0) {
    ix = (-(*n) + 1) * *incx + 1;
    }
    /* 如果 incx 小于 0，对 ix 进行重新赋值 */

    if (*incy < 0) {
    iy = (-(*n) + 1) * *incy + 1;
    }
    /* 如果 incy 小于 0，对 iy 进行重新赋值 */

    i__1 = *n;
    /* 循环次数为 n */
    for (i__ = 1; i__ <= i__1; ++i__) {
    /* 开始循环 */
    i__2 = ix;
    q__2.r = *c__ * cx[i__2].r, q__2.i = *c__ * cx[i__2].i;
    i__3 = iy;
    q__3.r = *s * cy[i__3].r, q__3.i = *s * cy[i__3].i;
    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
    ctemp.r = q__1.r, ctemp.i = q__1.i;
    i__2 = iy;
    i__3 = iy;
    q__2.r = *c__ * cy[i__3].r, q__2.i = *c__ * cy[i__3].i;
    i__4 = ix;
    q__3.r = *s * cx[i__4].r, q__3.i = *s * cx[i__4].i;
    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
    cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
    i__2 = ix;
    cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
    ix += *incx;
    iy += *incy;
    /* 循环体 */
/* L10: */
    }
    /* 循环结束，返回 0 */

    return 0;

/*        code for both increments equal to 1 */

L20:
    /* 标签 L20 */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    /* 开始循环 */
    i__2 = i__;
    q__2.r = *c__ * cx[i__2].r, q__2.i = *c__ * cx[i__2].i;
    i__3 = i__;
    q__3.r = *s * cy[i__3].r, q__3.i = *s * cy[i__3].i;
    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
    ctemp.r = q__1.r, ctemp.i = q__1.i;
    i__2 = i__;
    i__3 = i__;
    q__2.r = *c__ * cy[i__3].r, q__2.i = *c__ * cy[i__3].i;
    i__4 = i__;
    q__3.r = *s * cx[i__4].r, q__3.i = *s * cx[i__4].i;
    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
    cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
    i__2 = i__;
    cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
    /* 循环体 */
/* L30: */
    }
    /* 循环结束，返回 0 */

    return 0;
} /* csrot_ */
/* csrot_ 函数结束 */

/* Subroutine */ int csscal_(integer *n, real *sa, singlecomplex *cx, integer *incx)
{
    /* System generated locals */
    /* 系统生成的局部变量 */
    integer i__1, i__2, i__3, i__4;
    real r__1, r__2;
    singlecomplex q__1;
    /* 实数，复数变量的声明 */

    /* Local variables */
    static integer i__, nincx;
    /* 静态变量 */

/*
    Purpose
    =======

       CSSCAL scales a complex vector by a real constant.
       CSSCAL 通过一个实常数对复向量进行缩放。

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)
       进一步细节

    =====================================================================
*/

    /* Parameter adjustments */
    /* 参数调整 */
    --cx;
    /* 数组下标偏移，将指针向前移动 */

    /* Function Body */
    /* 函数主体 */
    if (*n <= 0 || *incx <= 0) {
    return 0;
    }
    /* 如果 n 小于等于 0 或者 incx 小于等于 0，返回 0 */

    if (*incx == 1) {
    goto L20;
    }
    /* 如果 incx 等于 1，跳转到标签 L20 */

/*        code for increment not equal to 1 */
/*        不等于 1 的增量代码 */

    nincx = *n * *incx;
    /* nincx 为 n 乘以 incx */
    i__1 = nincx;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
    /* 开始循环 */
    i__3 = i__;
    i__4 = i__;
    r__1 = *sa * cx[i__4].r;
    r__2 = *sa * r_imag(&cx[i__]);
    q__1.r = r__1, q__1.i = r__2;
    cx[i__3].r = q__1.r, cx[i__3].i = q__1.i;
    /* 循环体 */
/* L10: */
    }
    /* 循环结束 */
    返回整数 0，结束函数并将 0 作为返回值传递给调用者
/*       code for increment equal to 1 */
L20:
    /* 设置循环变量的上限为*n，以遍历向量cx中的元素 */
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
        /* i__1为当前迭代的索引，i__3为当前迭代的索引 */
        i__1 = i__;
        i__3 = i__;
        /* 计算复数乘法结果，并将结果赋给向量cx的实部和虚部 */
        r__1 = *sa * cx[i__3].r;
        r__2 = *sa * r_imag(&cx[i__]);
        q__1.r = r__1, q__1.i = r__2;
        cx[i__1].r = q__1.r, cx[i__1].i = q__1.i;
        /* L30: 标签，用于跳转或作为循环结尾的标记 */
    }
    return 0;
} /* csscal_ */

/* Subroutine */ int cswap_(integer *n, singlecomplex *cx, integer *incx, singlecomplex *cy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Local variables */
    static integer i__, ix, iy;
    static singlecomplex ctemp;

    /*
        Purpose
        =======

          CSWAP interchanges two vectors.

        Further Details
        ===============

           jack dongarra, linpack, 3/11/78.
           modified 12/3/93, array(1) declarations changed to array(*)

        =====================================================================
    */

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        /* 若增量incx和incy均为1，则执行标签为L20处的代码块 */
        goto L20;
    }

    /*
         code for unequal increments or equal increments not equal
           to 1
    */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    /* 遍历向量，交换元素 */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = ix;
        /* 将向量cx的元素赋值给临时变量ctemp */
        ctemp.r = cx[i__2].r, ctemp.i = cx[i__2].i;
        i__2 = ix;
        i__3 = iy;
        /* 将向量cy的元素赋值给向量cx的元素 */
        cx[i__2].r = cy[i__3].r, cx[i__2].i = cy[i__3].i;
        i__2 = iy;
        /* 将临时变量ctemp的值赋给向量cy的元素 */
        cy[i__2].r = ctemp.r, cy[i__2].i = ctemp.i;
        ix += *incx;
        iy += *incy;
        /* L10: 标签，用于跳转或作为循环结尾的标记 */
    }
    return 0;

    /*       code for both increments equal to 1 */
L20:
    /* 设置循环变量的上限为*n，以遍历向量cx中的元素 */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = i__;
        /* 将向量cx的元素赋值给临时变量ctemp */
        ctemp.r = cx[i__2].r, ctemp.i = cx[i__2].i;
        i__2 = i__;
        i__3 = i__;
        /* 将向量cy的元素赋值给向量cx的元素 */
        cx[i__2].r = cy[i__3].r, cx[i__2].i = cy[i__3].i;
        i__2 = i__;
        /* 将临时变量ctemp的值赋给向量cy的元素 */
        cy[i__2].r = ctemp.r, cy[i__2].i = ctemp.i;
        /* L30: 标签，用于跳转或作为循环结尾的标记 */
    }
    return 0;
} /* cswap_ */

/* Subroutine */ int ctrmm_(char *side, char *uplo, char *transa, char *diag,
    integer *m, integer *n, singlecomplex *alpha, singlecomplex *a, integer *lda,
    singlecomplex *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5,
        i__6;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, j, k, info;
    static singlecomplex temp;
    extern logical lsame_(char *, char *);
    static logical lside;
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;

    /*
        Purpose
        =======

        CTRMM  performs one of the matrix-matrix operations

           B := alpha*op( A )*B,   or   B := alpha*B*op( A )

        where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    */
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).

    Arguments
    ==========

    SIDE   - CHARACTER*1.
             On entry,  SIDE specifies whether  op( A ) multiplies B from
             the left or right as follows:

                SIDE = 'L' or 'l'   B := alpha*op( A )*B.

                SIDE = 'R' or 'r'   B := alpha*B*op( A ).

             Unchanged on exit.

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix A is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n'   op( A ) = A.

                TRANSA = 'T' or 't'   op( A ) = A'.

                TRANSA = 'C' or 'c'   op( A ) = conjg( A' ).

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit triangular
             as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of B. M must be at
             least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of B.  N must be
             at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX         .
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is
             zero then  A is not referenced and  B need not be set before
             entry.
             Unchanged on exit.

    A      - COMPLEX          array of DIMENSION ( LDA, k ), where k is m
             when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
             Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
             upper triangular part of the array  A must contain the upper
             triangular matrix  and the strictly lower triangular part of
             A is not referenced.
             Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
             lower triangular part of the array  A must contain the lower
             triangular matrix  and the strictly upper triangular part of
             A is not referenced.
             Note that when  DIAG = 'U' or 'u',  the diagonal elements of
             A  are not referenced either,  but are assumed to be  unity.
             Unchanged on exit.
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
             then LDA must be at least max( 1, n ).
             Unchanged on exit.


# LDA是一个整数参数，在进入函数时用于指定数组A的第一个维度大小。当 SIDE = 'L' 或 'l' 时，LDA至少必须为 max( 1, m )；当 SIDE = 'R' 或 'r' 时，LDA至少必须为 max( 1, n )。
# 函数执行完毕后，LDA的值保持不变。



    B      - COMPLEX          array of DIMENSION ( LDB, n ).
             Before entry,  the leading  m by n part of the array  B must
             contain the matrix  B,  and  on exit  is overwritten  by the
             transformed matrix.


# B是一个复数数组，维度为 (LDB, n)。
# 在进入函数之前，数组B的前 m 行 n 列必须包含矩阵B的数据。函数执行完毕后，这部分数组将被转换后的矩阵覆盖。



    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   LDB  must  be  at  least
             max( 1, m ).
             Unchanged on exit.


# LDB是一个整数参数，在进入函数时用于指定数组B的第一个维度大小。
# 函数执行完毕后，LDB的值保持不变。



    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


# 提供更多细节
# Level 3 Blas routine.
# 编写于1989年2月8日。
# Jack Dongarra，Argonne National Laboratory。
# Iain Duff，AERE Harwell。
# Jeremy Du Croz，Numerical Algorithms Group Ltd.。
# Sven Hammarling，Numerical Algorithms Group Ltd.。



    =====================================================================

       Test the input parameters.


# 测试输入参数。
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    // 判断 side 参数是否为 "L"，确定矩阵 A 是左乘还是右乘
    lside = lsame_(side, "L");
    if (lside) {
        nrowa = *m;
    } else {
        nrowa = *n;
    }
    // 判断 transa 参数是否为 "T"，确定是否转置矩阵 A
    noconj = lsame_(transa, "T");
    // 判断 diag 参数是否为 "N"，确定是否为非单位对角矩阵
    nounit = lsame_(diag, "N");
    // 判断 uplo 参数是否为 "U"，确定是否为上三角存储
    upper = lsame_(uplo, "U");

    // 初始化错误信息代码
    info = 0;
    // 检查 side 参数是否合法
    if (! lside && ! lsame_(side, "R")) {
        info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
        info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa, "T") && ! lsame_(transa, "C")) {
        info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
        info = 4;
    } else if (*m < 0) {
        info = 5;
    } else if (*n < 0) {
        info = 6;
    } else if (*lda < max(1,nrowa)) {
        info = 9;
    } else if (*ldb < max(1,*m)) {
        info = 11;
    }
    // 如果存在错误信息，则调用错误处理函数并返回
    if (info != 0) {
        xerbla_("CTRMM ", &info);
        return 0;
    }

/*     Quick return if possible. */

    // 若 m 或 n 为零，则直接返回
    if (*m == 0 || *n == 0) {
        return 0;
    }

/*     And when alpha.eq.zero. */

    // 若 alpha 等于零，则将 B 矩阵所有元素置为零
    if (alpha->r == 0.f && alpha->i == 0.f) {
        // 循环遍历 B 矩阵的所有元素
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 将 B 矩阵中的实部和虚部置为零
                i__3 = i__ + j * b_dim1;
                b[i__3].r = 0.f, b[i__3].i = 0.f;
            }
        }
        return 0;
    }

/*     Start the operations. */

    // 开始进行矩阵乘法操作
    if (lside) {
        // 若 side 参数为 "L"，即 A 在左侧
        if (lsame_(transa, "N")) {

/*           Form  B := alpha*A*B. */

            // 若 transa 参数为 "N"，即不转置 A
            // 若 upper 参数为真，即 A 为上三角矩阵
            if (upper) {
                // 循环遍历 B 矩阵的每一列
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    // 循环遍历 A 矩阵的每一行
                    i__2 = *m;
                    for (k = 1; k <= i__2; ++k) {
                        // 计算 B 矩阵中的当前元素位置
                        i__3 = k + j * b_dim1;
                        // 判断 B 矩阵当前位置的实部和虚部是否为零
                        if (b[i__3].r != 0.f || b[i__3].i != 0.f) {
                            // 计算 alpha*A(k,j)，并赋值给临时变量 temp
                            i__3 = k + j * b_dim1;
                            q__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i, q__1.i = alpha->r * b[i__3].i + alpha->i * b[i__3].r;
                            temp.r = q__1.r, temp.i = q__1.i;
                            // 循环遍历 A 矩阵的前 k-1 行
                            i__3 = k - 1;
                            for (i__ = 1; i__ <= i__3; ++i__) {
                                // 计算 temp*A(i,k)，并更新 B 矩阵中的元素
                                i__4 = i__ + j * b_dim1;
                                i__5 = i__ + j * b_dim1;
                                i__6 = i__ + k * a_dim1;
                                q__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i, q__2.i = temp.r * a[i__6].i + temp.i * a[i__6].r;
                                q__1.r = b[i__5].r + q__2.r, q__1.i = b[i__5].i + q__2.i;
                                b[i__4].r = q__1.r, b[i__4].i = q__1.i;
                            }
                            // 若 A 不是单位对角矩阵，则继续更新 temp
                            if (nounit) {
                                i__3 = k + k * a_dim1;
                                q__1.r = temp.r * a[i__3].r - temp.i * a[i__3].i, q__1.i = temp.r * a[i__3].i + temp.i * a[i__3].r;
                                temp.r = q__1.r, temp.i = q__1.i;
                            }
                            // 更新 B 矩阵中的当前位置元素
                            i__3 = k + j * b_dim1;
                            b[i__3].r = temp.r, b[i__3].i = temp.i;
                        }
                    }
                }
            }
        }
    }
```cpp`
/* L50: 结束 if (trans) 的条件判断，开始 else 分支 */
        } else {
/* L51: 设置循环变量 j 的上界为 *n */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
/* L52: 设置循环变量 k 的下界为 *m，循环递减直到 1 */
            for (k = *m; k >= 1; --k) {
/* L53: 计算数组 b 中索引为 k + j*b_dim1 处的元素是否为零 */
            i__2 = k + j * b_dim1;
/* L54: 如果不为零，则执行以下操作 */
            if (b[i__2].r != 0.f || b[i__2].i != 0.f) {
/* L55: 计算数组 b 中索引为 k + j*b_dim1 处的元素 */
                i__2 = k + j * b_dim1;
/* L56: 使用复数乘法计算 alpha 和 b[k + j*b_dim1] 的乘积，结果保存在 temp 中 */
                q__1.r = alpha->r * b[i__2].r - alpha->i * b[i__2]
                    .i, q__1.i = alpha->r * b[i__2].i +
                    alpha->i * b[i__2].r;
                temp.r = q__1.r, temp.i = q__1.i;
/* L57: 更新数组 b 中索引为 k + j*b_dim1 处的元素为 temp */
                i__2 = k + j * b_dim1;
                b[i__2].r = temp.r, b[i__2].i = temp.i;
/* L58: 如果 nounit 为真，则继续执行下面的操作 */
                if (nounit) {
/* L59: 计算数组 a 和 b 中的元素乘积并更新数组 b */
                i__2 = k + j * b_dim1;
                i__3 = k + j * b_dim1;
                i__4 = k + k * a_dim1;
                q__1.r = b[i__3].r * a[i__4].r - b[i__3].i *
                    a[i__4].i, q__1.i = b[i__3].r * a[
                    i__4].i + b[i__3].i * a[i__4].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
                }
/* L60: 循环变量 i__ 从 k+1 开始，遍历更新数组 b 中的元素 */
                i__2 = *m;
                for (i__ = k + 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + k * a_dim1;
                q__2.r = temp.r * a[i__5].r - temp.i * a[i__5]
                    .i, q__2.i = temp.r * a[i__5].i +
                    temp.i * a[i__5].r;
                q__1.r = b[i__4].r + q__2.r, q__1.i = b[i__4]
                    .i + q__2.i;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L61: 结束 k 的内循环 */
                }
/* L62: 结束 if (b[i__2].r != 0.f || b[i__2].i != 0.f) 的条件判断 */
            }
/* L63: 结束 k 的内循环 */
            }
/* L64: 结束 j 的外循环 */
        }
/* L65: 结束 if (trans) 的 else 分支 */
        }
    } else {

/*           Form  B := alpha*A'*B   or   B := alpha*conjg( A' )*B. */

/* L66: 如果 trans 为假，则执行以下操作 */
        if (upper) {
/* L67: 设置循环变量 j 的上界为 *n */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
/* L68: 设置循环变量 i__ 的下界为 *m，循环递减直到 1 */
            for (i__ = *m; i__ >= 1; --i__) {
/* L69: 计算数组 b 中索引为 i__ + j*b_dim1 处的元素，并将其赋值给 temp */
            i__2 = i__ + j * b_dim1;
            temp.r = b[i__2].r, temp.i = b[i__2].i;
/* L70: 如果 noconj 为真，则执行以下操作 */
            if (noconj) {
/* L71: 如果 nounit 为真，则执行以下操作 */
                if (nounit) {
/* L72: 计算数组 a 中索引为 i__ + i__*a_dim1 处的元素与 temp 的乘积，并将结果保存在 temp 中 */
                i__2 = i__ + i__ * a_dim1;
                q__1.r = temp.r * a[i__2].r - temp.i * a[i__2]
                    .i, q__1.i = temp.r * a[i__2].i +
                    temp.i * a[i__2].r;
                temp.r = q__1.r, temp.i = q__1.i;
                }
/* L73: 设置循环变量 k 的下界为 1，循环递增直到 i__-1 */
                i__2 = i__ - 1;
                for (k = 1; k <= i__2; ++k) {
                i__3 = k + i__ * a_dim1;
                i__4 = k + j * b_dim1;
                q__2.r = a[i__3].r * b[i__4].r - a[i__3].i *
                    b[i__4].i, q__2.i = a[i__3].r * b[
                    i__4].i + a[i__3].i * b[i__4].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L74: 结束 k 的内循环 */
                }
/* L75: 如果 noconj 为假，则执行以下操作 */
            } else {
/* L76: 如果 nounit 为真，则执行以下操作 */
                if (nounit) {
/* L77: 计算数组 a 中索引为 i__ + i__*a_dim1 处的元素的共轭乘积与 temp 的乘积，并将结果保存在 temp 中 */
                i__2 = i__ + i__ * a_dim1;
                r_cnjg(&q__2, &a[i__2]);
                q__1.r = temp.r * q__2.r - temp.i * q__2.i, q__1.i =
                     temp.r * q__2.i + temp.i * q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
                }
/* L78: 设置
/* L90: */
                }
            } else {
                // 如果不是单位矩阵并且不是共轭转置，则执行以下操作
                if (nounit) {
                    // 计算矩阵元素的共轭转置
                    r_cnjg(&q__2, &a[i__ + i__ * a_dim1]);
                    q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                        q__1.i = temp.r * q__2.i + temp.i *
                        q__2.r;
                    temp.r = q__1.r, temp.i = q__1.i;
                }
                i__2 = i__ - 1;
                // 遍历矩阵元素
                for (k = 1; k <= i__2; ++k) {
                    // 计算矩阵乘法的共轭转置
                    r_cnjg(&q__3, &a[k + i__ * a_dim1]);
                    i__3 = k + j * b_dim1;
                    q__2.r = q__3.r * b[i__3].r - q__3.i * b[i__3]
                        .i, q__2.i = q__3.r * b[i__3].i +
                        q__3.i * b[i__3].r;
                    q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                        q__2.i;
                    temp.r = q__1.r, temp.i = q__1.i;
/* L100: */
                }
            }
            i__2 = i__ + j * b_dim1;
            // 计算最终的乘法结果
            q__1.r = alpha->r * temp.r - alpha->i * temp.i,
                q__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L110: */
            }
/* L120: */
        }
        } else {
        // 如果不是上三角阵，则执行以下操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            // 遍历矩阵元素
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                temp.r = b[i__3].r, temp.i = b[i__3].i;
                if (noconj) {
                    // 如果不是共轭转置，则执行以下操作
                    if (nounit) {
                        i__3 = i__ + i__ * a_dim1;
                        q__1.r = temp.r * a[i__3].r - temp.i * a[i__3]
                            .i, q__1.i = temp.r * a[i__3].i +
                            temp.i * a[i__3].r;
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                    i__3 = *m;
                    // 计算矩阵乘法
                    for (k = i__ + 1; k <= i__3; ++k) {
                        i__4 = k + i__ * a_dim1;
                        i__5 = k + j * b_dim1;
                        q__2.r = a[i__4].r * b[i__5].r - a[i__4].i *
                            b[i__5].i, q__2.i = a[i__4].r * b[
                            i__5].i + a[i__4].i * b[i__5].r;
                        q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                            q__2.i;
                        temp.r = q__1.r, temp.i = q__1.i;
/* L130: */
                    }
                } else {
                    // 如果是共轭转置，则执行以下操作
                    if (nounit) {
                        r_cnjg(&q__2, &a[i__ + i__ * a_dim1]);
                        q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                            q__1.i = temp.r * q__2.i + temp.i *
                            q__2.r;
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                    i__3 = *m;
                    // 计算矩阵乘法的共轭转置
                    for (k = i__ + 1; k <= i__3; ++k) {
                        r_cnjg(&q__3, &a[k + i__ * a_dim1]);
                        i__4 = k + j * b_dim1;
                        q__2.r = q__3.r * b[i__4].r - q__3.i * b[i__4]
                            .i, q__2.i = q__3.r * b[i__4].i +
                            q__3.i * b[i__4].r;
                        q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                            q__2.i;
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                }
/* L140: */
                }
            }
            // 计算 alpha*B*A 的结果，将结果存储在 B 中
            i__3 = i__ + j * b_dim1;
            q__1.r = alpha->r * temp.r - alpha->i * temp.i,
                q__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L150: */
            }
/* L160: */
        }
        }
    }
    } else {
    if (lsame_(transa, "N")) {

/*           Form  B := alpha*B*A. */

        // 如果上三角矩阵 A
        if (upper) {
        // 从右下角开始遍历 A 的每一列
        for (j = *n; j >= 1; --j) {
            temp.r = alpha->r, temp.i = alpha->i;
            // 如果 A 不是单位矩阵，计算 temp = alpha * A(j,j)
            if (nounit) {
            i__1 = j + j * a_dim1;
            q__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                q__1.i = temp.r * a[i__1].i + temp.i * a[i__1]
                .r;
            temp.r = q__1.r, temp.i = q__1.i;
            }
            // 计算 B 的每一行 i__ 的值
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
            i__2 = i__ + j * b_dim1;
            i__3 = i__ + j * b_dim1;
            q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                q__1.i = temp.r * b[i__3].i + temp.i * b[i__3]
                .r;
            b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L170: */
            }
            // 更新 B 的每一列 i__ 的值，k 从 1 到 j-1
            i__1 = j - 1;
            for (k = 1; k <= i__1; ++k) {
            i__2 = k + j * a_dim1;
            // 如果 A(k,j) 不为零，计算 temp = alpha * A(k,j)
            if (a[i__2].r != 0.f || a[i__2].i != 0.f) {
                i__2 = k + j * a_dim1;
                q__1.r = alpha->r * a[i__2].r - alpha->i * a[i__2]
                    .i, q__1.i = alpha->r * a[i__2].i +
                    alpha->i * a[i__2].r;
                temp.r = q__1.r, temp.i = q__1.i;
                // 更新 B 的每一行 i__ 的值
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + k * b_dim1;
                q__2.r = temp.r * b[i__5].r - temp.i * b[i__5]
                    .i, q__2.i = temp.r * b[i__5].i +
                    temp.i * b[i__5].r;
                q__1.r = b[i__4].r + q__2.r, q__1.i = b[i__4]
                    .i + q__2.i;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L180: */
                }
            }
/* L190: */
            }
/* L200: */
        }
        } else {
        // 从左上角开始遍历 A 的每一列
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            temp.r = alpha->r, temp.i = alpha->i;
            // 如果 A 不是单位矩阵，计算 temp = alpha * A(j,j)
            if (nounit) {
            i__2 = j + j * a_dim1;
            q__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                q__1.i = temp.r * a[i__2].i + temp.i * a[i__2]
                .r;
            temp.r = q__1.r, temp.i = q__1.i;
            }
            // 计算 B 的每一行 i__ 的值
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * b_dim1;
            i__4 = i__ + j * b_dim1;
            q__1.r = temp.r * b[i__4].r - temp.i * b[i__4].i,
                q__1.i = temp.r * b[i__4].i + temp.i * b[i__4]
                .r;
            b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L210: */
            }
            // 循环遍历列索引大于当前行索引的列
            i__2 = *n;
            for (k = j + 1; k <= i__2; ++k) {
            // 计算当前位置在数组中的索引
            i__3 = k + j * a_dim1;
            // 检查矩阵 a 中当前位置是否为非零值
            if (a[i__3].r != 0.f || a[i__3].i != 0.f) {
                // 计算 alpha 与 a 元素相乘后的结果，并存储在 temp 中
                i__3 = k + j * a_dim1;
                q__1.r = alpha->r * a[i__3].r - alpha->i * a[i__3]
                    .i, q__1.i = alpha->r * a[i__3].i +
                    alpha->i * a[i__3].r;
                temp.r = q__1.r, temp.i = q__1.i;
                // 对矩阵 b 的每一行执行累加操作
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                // 计算当前位置在数组 b 中的索引
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + j * b_dim1;
                i__6 = i__ + k * b_dim1;
                // 计算累加结果并更新 b 数组
                q__2.r = temp.r * b[i__6].r - temp.i * b[i__6]
                    .i, q__2.i = temp.r * b[i__6].i +
                    temp.i * b[i__6].r;
                q__1.r = b[i__5].r + q__2.r, q__1.i = b[i__5]
                    .i + q__2.i;
                b[i__4].r = q__1.r, b[i__4].i = q__1.i;
/* L220: */
                }
            }
/* L230: */
            }
/* L240: */
        }
        }
    } else {

/*           Form  B := alpha*B*A'   or   B := alpha*B*conjg( A' ). */

        // 若 upper 为真，则进行下列操作
        if (upper) {
        // 循环遍历矩阵的每一列
        i__1 = *n;
        for (k = 1; k <= i__1; ++k) {
            // 循环遍历当前列之前的每一行
            i__2 = k - 1;
            for (j = 1; j <= i__2; ++j) {
            // 计算当前位置在矩阵 a 中的索引
            i__3 = j + k * a_dim1;
            // 检查矩阵 a 中当前位置是否为非零值
            if (a[i__3].r != 0.f || a[i__3].i != 0.f) {
                // 根据 noconj 的值选择不同的操作
                if (noconj) {
                // 计算 alpha 与 a 元素相乘后的结果，并存储在 temp 中
                i__3 = j + k * a_dim1;
                q__1.r = alpha->r * a[i__3].r - alpha->i * a[
                    i__3].i, q__1.i = alpha->r * a[i__3]
                    .i + alpha->i * a[i__3].r;
                temp.r = q__1.r, temp.i = q__1.i;
                } else {
                // 计算 alpha 与 a 共轭转置元素相乘后的结果，并存储在 temp 中
                r_cnjg(&q__2, &a[j + k * a_dim1]);
                q__1.r = alpha->r * q__2.r - alpha->i *
                    q__2.i, q__1.i = alpha->r * q__2.i +
                    alpha->i * q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
                }
                // 对矩阵 b 的每一行执行累加操作
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                // 计算当前位置在数组 b 中的索引
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + j * b_dim1;
                i__6 = i__ + k * b_dim1;
                // 计算累加结果并更新 b 数组
                q__2.r = temp.r * b[i__6].r - temp.i * b[i__6]
                    .i, q__2.i = temp.r * b[i__6].i +
                    temp.i * b[i__6].r;
                q__1.r = b[i__5].r + q__2.r, q__1.i = b[i__5]
                    .i + q__2.i;
                b[i__4].r = q__1.r, b[i__4].i = q__1.i;
/* L250: */
                }
            }
/* L260: 结束内部循环，处理当前列 */
            }
            // 将 alpha 的值赋给 temp
            temp.r = alpha->r, temp.i = alpha->i;
            // 如果矩阵 A 非单位矩阵，则更新 temp 的值
            if (nounit) {
                // 如果不需要共轭，计算矩阵 A 的元素和 temp 的乘积
                if (noconj) {
                    // 计算矩阵 A 的元素与 temp 的乘积
                    i__2 = k + k * a_dim1;
                    q__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                        q__1.i = temp.r * a[i__2].i + temp.i * a[i__2].r;
                    temp.r = q__1.r, temp.i = q__1.i;
                } else {
                    // 对矩阵 A 的元素进行共轭并计算与 temp 的乘积
                    r_cnjg(&q__2, &a[k + k * a_dim1]);
                    q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                        q__1.i = temp.r * q__2.i + temp.i * q__2.r;
                    temp.r = q__1.r, temp.i = q__1.i;
                }
            }
            // 如果 temp 不等于 1 或 0，则对矩阵 B 进行更新
            if (temp.r != 1.f || temp.i != 0.f) {
                // 对矩阵 B 的每一行进行更新
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + k * b_dim1;
                    i__4 = i__ + k * b_dim1;
                    q__1.r = temp.r * b[i__4].r - temp.i * b[i__4].i,
                        q__1.i = temp.r * b[i__4].i + temp.i * b[i__4].r;
                    b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L270: 循环结束标签 */
                }
            }
/* L280: 外部循环结束标签 */
        }
        } else {
        // 处理非单位三角矩阵的情况
        for (k = *n; k >= 1; --k) {
            // 遍历矩阵 A 的列
            i__1 = *n;
            for (j = k + 1; j <= i__1; ++j) {
                // 如果矩阵 A 的元素不为零，则更新矩阵 B
                i__2 = j + k * a_dim1;
                if (a[i__2].r != 0.f || a[i__2].i != 0.f) {
                    // 根据不同的共轭选项计算临时值 temp
                    if (noconj) {
                        i__2 = j + k * a_dim1;
                        q__1.r = alpha->r * a[i__2].r - alpha->i * a[i__2].i,
                            q__1.i = alpha->r * a[i__2].i + alpha->i * a[i__2].r;
                        temp.r = q__1.r, temp.i = q__1.i;
                    } else {
                        r_cnjg(&q__2, &a[j + k * a_dim1]);
                        q__1.r = alpha->r * q__2.r - alpha->i * q__2.i,
                            q__1.i = alpha->r * q__2.i + alpha->i * q__2.r;
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                    // 更新矩阵 B 的对应元素
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * b_dim1;
                        i__4 = i__ + j * b_dim1;
                        i__5 = i__ + k * b_dim1;
                        q__2.r = temp.r * b[i__5].r - temp.i * b[i__5].i,
                            q__2.i = temp.r * b[i__5].i + temp.i * b[i__5].r;
                        q__1.r = b[i__4].r + q__2.r, q__1.i = b[i__4].i + q__2.i;
                        b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L290: 内部循环结束标签 */
                    }
                }
/* L280: 外部循环结束标签 */
            }
/* L280: 外部循环结束标签 */
        }
/* L300: */
            }
            temp.r = alpha->r, temp.i = alpha->i;
            if (nounit) {
            if (noconj) {
                i__1 = k + k * a_dim1;
                q__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                    q__1.i = temp.r * a[i__1].i + temp.i * a[
                    i__1].r;
                temp.r = q__1.r, temp.i = q__1.i;
            } else {
                r_cnjg(&q__2, &a[k + k * a_dim1]);
                q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                    q__1.i = temp.r * q__2.i + temp.i *
                    q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
            }
            if (temp.r != 1.f || temp.i != 0.f) {
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    q__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L310: */
            }
            }
/* L320: */
        }
        }
    }
    }

    return 0;

/*     End of CTRMM . */

} /* ctrmm_ */

/* Subroutine */ int ctrmv_(char *uplo, char *trans, char *diag, integer *n,
    singlecomplex *a, integer *lda, singlecomplex *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, j, ix, jx, kx, info;
    static singlecomplex temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;


/*
    Purpose
    =======

    CTRMV  performs one of the matrix-vector operations

       x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   x := A*x.

                TRANS = 'T' or 't'   x := A'*x.

                TRANS = 'C' or 'c'   x := conjg( A' )*x.

             Unchanged on exit.
    # DIAG - CHARACTER*1.
    # 在输入时，DIAG 指定矩阵 A 是否为单位三角阵的标志：
    #
    #    DIAG = 'U' 或 'u'   表示 A 被假设为单位三角阵。
    #
    #    DIAG = 'N' 或 'n'   表示 A 不被假设为单位三角阵。
    #
    # 在退出时保持不变。

    # N - INTEGER.
    # 在输入时，N 指定矩阵 A 的阶数。
    # N 必须至少为零。
    # 在退出时保持不变。

    # A - COMPLEX array of DIMENSION ( LDA, n ).
    # 在输入时，如果 UPLO = 'U' 或 'u'，则数组 A 的前 n 行 n 列上三角部分应包含上三角矩阵，
    # 并且不引用 A 的严格下三角部分。
    # 在输入时，如果 UPLO = 'L' 或 'l'，则数组 A 的前 n 行 n 列下三角部分应包含下三角矩阵，
    # 并且不引用 A 的严格上三角部分。
    # 注意，当 DIAG = 'U' 或 'u' 时，A 的对角元素也不被引用，但假定其为单位元。
    # 在退出时保持不变。

    # LDA - INTEGER.
    # 在输入时，LDA 指定数组 A 的第一个维度，即调用程序中声明的维度。
    # LDA 必须至少为 max(1, n)。
    # 在退出时保持不变。

    # X - COMPLEX array of dimension at least ( 1 + ( n - 1 )*abs( INCX ) ).
    # 在输入时，增量数组 X 必须包含 n 元素的向量 x。
    # 在退出时，X 被覆盖为变换后的向量 x。

    # INCX - INTEGER.
    # 在输入时，INCX 指定 X 的元素的增量。
    # INCX 必须不为零。
    # 在退出时保持不变。

    # Further Details
    # ===============
    # Level 2 Blas routine.
    # Level 2 Blas（基础线性代数子程序库）例程。

    # -- Written on 22-October-1986.
    #    Jack Dongarra, Argonne National Lab.
    #    Jeremy Du Croz, Nag Central Office.
    #    Sven Hammarling, Nag Central Office.
    #    Richard Hanson, Sandia National Labs.
    # 编写日期：1986年10月22日。
    # Jack Dongarra，Argonne National Lab。
    # Jeremy Du Croz，Nag Central Office。
    # Sven Hammarling，Nag Central Office。
    # Richard Hanson，Sandia National Labs。

    # =====================================================================

    # Test the input parameters.
    # 测试输入参数。
    /* Parameter adjustments */
    a_dim1 = *lda;  // 将 *lda 赋值给 a_dim1，*lda 是 A 的第一个维度
    a_offset = 1 + a_dim1;  // 计算偏移量 a_offset
    a -= a_offset;  // 调整 A 的起始地址，使其从 (1,1) 开始
    --x;  // 将 X 的起始地址向前移动一个位置，从 1 开始

    /* Function Body */
    info = 0;  // 初始化 info，用于记录错误信息
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {  // 检查 uplo 是否为 'U' 或 'L'
    info = 1;  // 设置 info 为 1，表示 uplo 参数错误
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
    info = 2;  // 设置 info 为 2，表示 trans 参数错误
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
    info = 3;  // 设置 info 为 3，表示 diag 参数错误
    } else if (*n < 0) {
    info = 4;  // 设置 info 为 4，表示 n 参数错误
    } else if (*lda < max(1,*n)) {
    info = 6;  // 设置 info 为 6，表示 lda 参数错误
    } else if (*incx == 0) {
    info = 8;  // 设置 info 为 8，表示 incx 参数错误
    }
    if (info != 0) {
    xerbla_("CTRMV ", &info);  // 调用错误处理程序 xerbla_
    return 0;  // 返回 0 表示出错
    }

/*     Quick return if possible. */

    if (*n == 0) {  // 如果 n 为 0，直接返回
    return 0;
    }

    noconj = lsame_(trans, "T");  // 检查是否不需要共轭
    nounit = lsame_(diag, "N");  // 检查是否不是单位矩阵

/*
       Set up the start point in X if the increment is not unity. This
       will be  ( N - 1 )*INCX  too small for descending loops.
*/

    if (*incx <= 0) {  // 如果 incx 小于等于 0
    kx = 1 - (*n - 1) * *incx;  // 计算 kx 的起始位置
    } else if (*incx != 1) {  // 如果 incx 不等于 1
    kx = 1;  // 设置 kx 的起始位置
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (lsame_(trans, "N")) {  // 如果 trans 为 'N'

/*        Form  x := A*x. */

    if (lsame_(uplo, "U")) {  // 如果 uplo 为 'U'
        if (*incx == 1) {  // 如果 incx 等于 1
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 遍历 j 从 1 到 n
            i__2 = j;
            if (x[i__2].r != 0.f || x[i__2].i != 0.f) {  // 如果 x[j] 不为零
            i__2 = j;
            temp.r = x[i__2].r, temp.i = x[i__2].i;  // 将 x[j] 复制到 temp
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历 i 从 1 到 j-1
                i__3 = i__;
                i__4 = i__;
                i__5 = i__ + j * a_dim1;
                q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,  // 计算乘积的实部
                    q__2.i = temp.r * a[i__5].i + temp.i * a[i__5].r;  // 计算乘积的虚部
                q__1.r = x[i__4].r + q__2.r, q__1.i = x[i__4].i + q__2.i;  // 更新 x[i]
                x[i__3].r = q__1.r, x[i__3].i = q__1.i;
/* L10: */
            }
            if (nounit) {  // 如果不是单位矩阵
                i__2 = j;
                i__3 = j;
                i__4 = j + j * a_dim1;
                q__1.r = x[i__3].r * a[i__4].r - x[i__3].i * a[i__4].i,  // 计算乘积的实部
                    q__1.i = x[i__3].r * a[i__4].i + x[i__3].i * a[i__4].r;  // 计算乘积的虚部
                x[i__2].r = q__1.r, x[i__2].i = q__1.i;  // 更新 x[j]
            }
            }
        }
        } else {
        jx = kx;
        i__1 = *n;
        // 循环遍历矩阵的列
        for (j = 1; j <= i__1; ++j) {
            i__2 = jx;
            // 检查向量 x 中第 jx 位置的值是否非零
            if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
            i__2 = jx;
            // 将向量 x 中第 jx 位置的值存入临时变量 temp
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            ix = kx;
            i__2 = j - 1;
            // 循环遍历矩阵的行，更新向量 x 中的值
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = ix;
                i__4 = ix;
                i__5 = i__ + j * a_dim1;
                // 计算矩阵乘法后更新向量 x 中的值
                q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    q__2.i = temp.r * a[i__5].i + temp.i * a[
                    i__5].r;
                q__1.r = x[i__4].r + q__2.r, q__1.i = x[i__4].i +
                    q__2.i;
                x[i__3].r = q__1.r, x[i__3].i = q__1.i;
                ix += *incx;
/* L30: */
            }
            // 如果不是单位矩阵，更新向量 x 中第 jx 位置的值
            if (nounit) {
                i__2 = jx;
                i__3 = jx;
                i__4 = j + j * a_dim1;
                q__1.r = x[i__3].r * a[i__4].r - x[i__3].i * a[
                    i__4].i, q__1.i = x[i__3].r * a[i__4].i +
                    x[i__3].i * a[i__4].r;
                x[i__2].r = q__1.r, x[i__2].i = q__1.i;
            }
            }
            // 更新 jx 以便处理下一列
            jx += *incx;
/* L40: */
        }
        }
    } else {
        if (*incx == 1) {
        // 逆序循环遍历矩阵的列
        for (j = *n; j >= 1; --j) {
            i__1 = j;
            // 检查向量 x 中第 j 位置的值是否非零
            if (x[i__1].r != 0.f || x[i__1].i != 0.f) {
            i__1 = j;
            // 将向量 x 中第 j 位置的值存入临时变量 temp
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            i__1 = j + 1;
            // 逆序循环遍历矩阵的行，更新向量 x 中的值
            for (i__ = *n; i__ >= i__1; --i__) {
                i__2 = i__;
                i__3 = i__;
                i__4 = i__ + j * a_dim1;
                // 计算矩阵乘法后更新向量 x 中的值
                q__2.r = temp.r * a[i__4].r - temp.i * a[i__4].i,
                    q__2.i = temp.r * a[i__4].i + temp.i * a[
                    i__4].r;
                q__1.r = x[i__3].r + q__2.r, q__1.i = x[i__3].i +
                    q__2.i;
                x[i__2].r = q__1.r, x[i__2].i = q__1.i;
/* L50: */
            }
            // 如果不是单位矩阵，更新向量 x 中第 j 位置的值
            if (nounit) {
                i__1 = j;
                i__2 = j;
                i__3 = j + j * a_dim1;
                q__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
                    i__3].i, q__1.i = x[i__2].r * a[i__3].i +
                    x[i__2].i * a[i__3].r;
                x[i__1].r = q__1.r, x[i__1].i = q__1.i;
            }
            }
/* L60: */
        }
        } else {
        kx += (*n - 1) * *incx;
        jx = kx;
        for (j = *n; j >= 1; --j) {
            i__1 = jx;
            // 检查 x[jx] 是否为零
            if (x[i__1].r != 0.f || x[i__1].i != 0.f) {
            i__1 = jx;
            // 将 x[jx] 的值赋给 temp
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            ix = kx;
            i__1 = j + 1;
            // 对于每个 i 从 j 到 *n，执行累加
            for (i__ = *n; i__ >= i__1; --i__) {
                i__2 = ix;
                i__3 = ix;
                i__4 = i__ + j * a_dim1;
                // 计算矩阵和向量的乘积并加到 x[ix]
                q__2.r = temp.r * a[i__4].r - temp.i * a[i__4].i,
                    q__2.i = temp.r * a[i__4].i + temp.i * a[
                    i__4].r;
                q__1.r = x[i__3].r + q__2.r, q__1.i = x[i__3].i +
                    q__2.i;
                x[i__2].r = q__1.r, x[i__2].i = q__1.i;
                ix -= *incx;
/* L70: */
            }
            // 如果非单位矩阵，还需处理对角线元素
            if (nounit) {
                i__1 = jx;
                i__2 = jx;
                i__3 = j + j * a_dim1;
                q__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
                    i__3].i, q__1.i = x[i__2].r * a[i__3].i +
                    x[i__2].i * a[i__3].r;
                x[i__1].r = q__1.r, x[i__1].i = q__1.i;
            }
            }
            // 更新 jx 的位置
            jx -= *incx;
/* L80: */
        }
        }
    }
    } else {

/*        Form  x := A'*x  or  x := conjg( A' )*x. */

    if (lsame_(uplo, "U")) {
        if (*incx == 1) {
        // 当 incx 为 1 时的情况
        for (j = *n; j >= 1; --j) {
            i__1 = j;
            // 将 x[j] 的值赋给 temp
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            // 如果不使用共轭转置，则处理非单位矩阵的对角元素
            if (noconj) {
            if (nounit) {
                i__1 = j + j * a_dim1;
                q__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                    q__1.i = temp.r * a[i__1].i + temp.i * a[
                    i__1].r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
            // 对于每个 i 从 j-1 到 1，计算累加
            for (i__ = j - 1; i__ >= 1; --i__) {
                i__1 = i__ + j * a_dim1;
                i__2 = i__;
                q__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
                    i__2].i, q__2.i = a[i__1].r * x[i__2].i +
                    a[i__1].i * x[i__2].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L90: */
            }
            } else {
            if (nounit) {
                // 对于使用共轭转置的情况，处理对角元素
                r_cnjg(&q__2, &a[j + j * a_dim1]);
                q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                    q__1.i = temp.r * q__2.i + temp.i *
                    q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
            // 对于每个 i 从 j-1 到 1，计算共轭转置的累加
            for (i__ = j - 1; i__ >= 1; --i__) {
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__1 = i__;
                q__2.r = q__3.r * x[i__1].r - q__3.i * x[i__1].i,
                    q__2.i = q__3.r * x[i__1].i + q__3.i * x[
                    i__1].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L100: */
            }
            }
            i__1 = j;
            x[i__1].r = temp.r, x[i__1].i = temp.i;
/* L110: */
        }
        } else {
        jx = kx + (*n - 1) * *incx;
        for (j = *n; j >= 1; --j) {
            i__1 = jx;
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            ix = jx;
            if (noconj) {
            if (nounit) {
                i__1 = j + j * a_dim1;
                q__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                    q__1.i = temp.r * a[i__1].i + temp.i * a[
                    i__1].r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
            for (i__ = j - 1; i__ >= 1; --i__) {
                ix -= *incx;
                i__1 = i__ + j * a_dim1;
                i__2 = ix;
                q__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
                    i__2].i, q__2.i = a[i__1].r * x[i__2].i +
                    a[i__1].i * x[i__2].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L120: */
            }
            } else {
            if (nounit) {
                r_cnjg(&q__2, &a[j + j * a_dim1]);
                q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                    q__1.i = temp.r * q__2.i + temp.i *
                    q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
            for (i__ = j - 1; i__ >= 1; --i__) {
                ix -= *incx;
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__1 = ix;
                q__2.r = q__3.r * x[i__1].r - q__3.i * x[i__1].i,
                    q__2.i = q__3.r * x[i__1].i + q__3.i * x[
                    i__1].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L130: */
            }
            }
            i__1 = jx;
            x[i__1].r = temp.r, x[i__1].i = temp.i;
            jx -= *incx;
/* L140: */
        }
        }
    } else {
        if (*incx == 1) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = j;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            if (noconj) {
            if (nounit) {
                i__2 = j + j * a_dim1;
                q__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                    q__1.i = temp.r * a[i__2].i + temp.i * a[
                    i__2].r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * a_dim1;
                i__4 = i__;
                q__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, q__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L150: */
            }
            } else {
            if (nounit) {
                r_cnjg(&q__2, &a[j + j * a_dim1]);
                q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                    q__1.i = temp.r * q__2.i + temp.i *
                    q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__3 = i__;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                    q__2.i = q__3.r * x[i__3].i + q__3.i * x[
                    i__3].r;
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L160: */
            }
            }
            i__2 = j;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L170: */
        }
        }
    }


注释完成。
/* L150: */
/* 开始处理当前列的三角矩阵向量乘法 */
            }
            } else {
/* 如果不是单位三角矩阵 */
            if (nounit) {
/* 如果不是单位三角矩阵，计算向量和当前列的内积 */
                r_cnjg(&q__2, &a[j + j * a_dim1]);
                q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                    q__1.i = temp.r * q__2.i + temp.i *
                    q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
/* 遍历当前列后面的所有行 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
/* 计算矩阵元素和向量元素的乘积 */
                i__3 = i__;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                    q__2.i = q__3.r * x[i__3].i + q__3.i * x[
                    i__3].r;
/* 将乘积结果加到临时变量中 */
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L160: */
            }
            }
/* 将结果存入向量 x 中 */
            i__2 = j;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L170: */
        }
        } else {
/* 如果不是单位三角矩阵 */
        jx = kx;
/* 初始化向量 x 中的索引 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
/* 复制向量 x 中的当前元素到临时变量 */
            i__2 = jx;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
/* 初始化向量 x 的索引 */
            ix = jx;
/* 如果是共轭计算 */
            if (noconj) {
/* 如果不是单位三角矩阵 */
            if (nounit) {
/* 如果不是单位三角矩阵，计算向量和当前列的内积 */
                i__2 = j + j * a_dim1;
                q__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                    q__1.i = temp.r * a[i__2].i + temp.i * a[
                    i__2].r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
/* 遍历当前列后面的所有行 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                ix += *incx;
/* 计算矩阵元素和向量元素的乘积 */
                i__3 = i__ + j * a_dim1;
                i__4 = ix;
                q__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, q__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
/* 将乘积结果加到临时变量中 */
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L180: */
            }
            } else {
/* 如果是共轭计算 */
            if (nounit) {
/* 如果不是单位三角矩阵，计算向量和当前列的内积 */
                r_cnjg(&q__2, &a[j + j * a_dim1]);
                q__1.r = temp.r * q__2.r - temp.i * q__2.i,
                    q__1.i = temp.r * q__2.i + temp.i *
                    q__2.r;
                temp.r = q__1.r, temp.i = q__1.i;
            }
/* 遍历当前列后面的所有行 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                ix += *incx;
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
/* 计算矩阵元素和向量元素的乘积 */
                i__3 = ix;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                    q__2.i = q__3.r * x[i__3].i + q__3.i * x[
                    i__3].r;
/* 将乘积结果加到临时变量中 */
                q__1.r = temp.r + q__2.r, q__1.i = temp.i +
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L190: */
            }
            }
/* 将结果存入向量 x 中 */
            i__2 = jx;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* 更新向量 x 的索引 */
            jx += *incx;
/* L200: */
        }
        }
    }
    }

/* 返回成功状态码 */
    return 0;

/*     End of CTRMV . */

} /* ctrmv_ */
/* Subroutine */ int ctrsm_(char *side, char *uplo, char *transa, char *diag,
    integer *m, integer *n, singlecomplex *alpha, singlecomplex *a, integer *lda,
    singlecomplex *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5,
        i__6, i__7;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, j, k, info;
    static singlecomplex temp;
    extern logical lsame_(char *, char *);
    static logical lside;
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;


/*
    Purpose
    =======

    CTRSM  solves one of the matrix equations

       op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).

    The matrix X is overwritten on B.

    Arguments
    ==========

    SIDE   - CHARACTER*1.
             On entry, SIDE specifies whether op( A ) appears on the left
             or right of X as follows:

                SIDE = 'L' or 'l'   op( A )*X = alpha*B.

                SIDE = 'R' or 'r'   X*op( A ) = alpha*B.

             Unchanged on exit.

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix A is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n'   op( A ) = A.

                TRANSA = 'T' or 't'   op( A ) = A'.

                TRANSA = 'C' or 'c'   op( A ) = conjg( A' ).

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit triangular
             as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of B. M must be at
             least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of B.  N must be
             at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX         .
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is
             zero then  A is not referenced and  B need not be set before
             entry.
             Unchanged on exit.
*/

    /* Initialize info to zero */
    info = 0;

    /* Check for zero alpha */
    if (alpha == 0) {
        /* If alpha is zero, no operations on A or B are necessary */
        return;
    }

    /* Set booleans for side, upper/lower, and whether A is unit triangular */
    lside = lsame_(side, "L");
    upper = lsame_(uplo, "U");
    nounit = lsame_(diag, "N");

    /* Determine the effective length of A */
    if (lside) {
        nrowa = *m;
    } else {
        nrowa = *n;
    }

    /* Check for errors in parameters */
    if (!lsame_(side, "L") && !lsame_(side, "R")) {
        info = 1;
    } else if (!lsame_(uplo, "U") && !lsame_(uplo, "L")) {
        info = 2;
    } else if (!lsame_(transa, "N") && !lsame_(transa, "T") && !lsame_(transa, "C")) {
        info = 3;
    } else if (!lsame_(diag, "U") && !lsame_(diag, "N")) {
        info = 4;
    } else if (*m < 0) {
        info = 5;
    } else if (*n < 0) {
        info = 6;
    } else if (*lda < max(1, nrowa)) {
        info = 9;
    } else if (*ldb < max(1, *m)) {
        info = 11;
    }

    /* Report any errors found */
    if (info != 0) {
        xerbla_("CTRSM ", &info);
        return;
    }

    /* Quick return if no rows or columns */
    if (*m == 0 || *n == 0) {
        return;
    }

    /* Adjust lda and ldb for Fortran indexing */
    --a;
    --b;

    /* Apply left side operation */
    if (lside) {
        if (lsame_(transa, "N")) {
            /* Form  B := alpha*inv(A)*B */
            if (upper) {
                /* Solve A*X = B, where A is upper triangular */
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (!lsame_(alpha, 1)) {
                        /* Scale B if alpha is not 1 */
                        q__1.r = alpha->r, q__1.i = alpha->i;
                        temp.r = q__1.r, temp.i = q__1.i;
                        i__2 = *m;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            q__1.r = temp.r * b[i__ + j * b_dim1].r - temp.i * b[i__ + j * b_dim1].i,
                            q__1.i = temp.r * b[i__ + j * b_dim1].i + temp.i * b[i__ + j * b_dim1].r;
                            b[i__ + j * b_dim1].r = q__1.r, b[i__ + j * b_dim1].i = q__1.i;
                        }
                    }
                    /* Division by A(j,j) if A is not unit triangular */
                    if (nounit) {
                        q__1.r = 1.f / a[j + j * a_dim1].r, q__1.i = -1.f / a[j + j * a_dim1].i;
                        temp.r = q__1.r, temp.i = q__1.i;
                        i__2 = *m;
                        for (k = 1; k <= i__2; ++k) {
                            q__2.r = temp.r * b[k + j * b_dim1].r - temp.i * b[k + j * b_dim1].i,
                            q__2.i = temp.r * b[k + j * b_dim1].i + temp.i * b[k + j * b_dim1].r;
                            q__1.r = b[k + j * b_dim1].r, q__1.i = b[k + j * b_dim1].i;
                            b[k + j * b_dim1].r = q__2.r, b[k + j * b_dim1].i = q__2.i;
                        }
                    }
                    /* Update B(j+1:n,j) with -alpha*A(j+1:n,j) */
                    i__2 = *n - j;
                    q__1.r = -alpha->r, q__1.i = -alpha->i;
                    temp.r = q__1.r, temp.i = q__1.i;
                    i__2 = *m;
                    for (k = 1; k <= i__2; ++k) {
                        i__3 = j + 1;
                        i__4 = *n - j;
                        for (i__ = 1; i__ <= i__4; ++i__) {
                            i__5 = k + i__ + j * b_dim1;
                            i__6 = k + i__ + j * b_dim1;
                            q__3.r = temp.r * a[i__3 + (j + i__) * a_dim1].r - temp.i * a[i__3 + (j + i__) * a_dim1].i,
                            q__3.i = temp.r * a[i__3 + (j + i__) * a_dim1].i + temp.i * a[i__3 + (j + i__) * a_dim1].r;
                            q__2.r = b[i__6].r + q__3.r, q__2.i = b[i__6].
    A      - COMPLEX          array of DIMENSION ( LDA, k ), where k is m
             when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
             Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
             upper triangular part of the array  A must contain the upper
             triangular matrix  and the strictly lower triangular part of
             A is not referenced.
             Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
             lower triangular part of the array  A must contain the lower
             triangular matrix  and the strictly upper triangular part of
             A is not referenced.
             Note that when  DIAG = 'U' or 'u',  the diagonal elements of
             A  are not referenced either,  but are assumed to be  unity.
             Unchanged on exit.


# A - 复数数组，维度为 (LDA, k)，其中 k 是 m（当 SIDE = 'L' 或 'l' 时），或者是 n（当 SIDE = 'R' 或 'r' 时）。
       在进入函数时，若 UPLO = 'U' 或 'u'，数组 A 的前 k 行 k 列必须包含上三角矩阵，且严格下三角部分不被引用。
       在进入函数时，若 UPLO = 'L' 或 'l'，数组 A 的前 k 行 k 列必须包含下三角矩阵，且严格上三角部分不被引用。
       当 DIAG = 'U' 或 'u' 时，A 的对角元素也不被引用，但假定其为单位元素。
       函数结束时，A 的内容保持不变。



    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
             then LDA must be at least max( 1, n ).
             Unchanged on exit.


# LDA - 整数。
       在函数进入时，LDA 指定数组 A 的第一个维度，即在调用程序中声明的大小。
       当 SIDE = 'L' 或 'l' 时，LDA 必须至少为 max(1, m)；当 SIDE = 'R' 或 'r' 时，LDA 必须至少为 max(1, n)。
       函数结束时，LDA 的值保持不变。



    B      - COMPLEX          array of DIMENSION ( LDB, n ).
             Before entry,  the leading  m by n part of the array  B must
             contain  the  right-hand  side  matrix  B,  and  on exit  is
             overwritten by the solution matrix  X.


# B - 复数数组，维度为 (LDB, n)。
     在进入函数时，数组 B 的前 m 行 n 列必须包含右手边矩阵 B 的内容；函数结束时，数组 B 被覆盖为解矩阵 X 的内容。



    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   LDB  must  be  at  least
             max( 1, m ).
             Unchanged on exit.


# LDB - 整数。
       在函数进入时，LDB 指定数组 B 的第一个维度，即在调用程序中声明的大小。
       LDB 必须至少为 max(1, m)。
       函数结束时，LDB 的值保持不变。



    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


# 进一步细节
    Level 3 Blas routine.

    -- 编写于1989年2月8日。
       Jack Dongarra，阿贡国家实验室。
       Iain Duff，AERE Harwell。
       Jeremy Du Croz，Numerical Algorithms Group Ltd。
       Sven Hammarling，Numerical Algorithms Group Ltd。



    =====================================================================


       Test the input parameters.


# 测试输入参数。
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    lside = lsame_(side, "L");
    // 判断是否左侧矩阵操作
    if (lside) {
        nrowa = *m;
    } else {
        nrowa = *n;
    }
    // 根据参数确定矩阵 A 的行数 nrowa

    noconj = lsame_(transa, "T");
    // 判断是否转置操作
    nounit = lsame_(diag, "N");
    // 判断是否为单位对角矩阵
    upper = lsame_(uplo, "U");
    // 判断是否为上三角矩阵

    info = 0;
    // 初始化信息代码，用于检测输入参数是否正确
    if (! lside && ! lsame_(side, "R")) {
        info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
        info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa,
         "T") && ! lsame_(transa, "C")) {
        info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
        "N")) {
        info = 4;
    } else if (*m < 0) {
        info = 5;
    } else if (*n < 0) {
        info = 6;
    } else if (*lda < max(1,nrowa)) {
        info = 9;
    } else if (*ldb < max(1,*m)) {
        info = 11;
    }
    // 检测输入参数是否正确，根据不同错误类型赋值 info

    if (info != 0) {
        xerbla_("CTRSM ", &info);
        // 如果参数有误，调用错误处理函数并返回
        return 0;
    }

    /*     Quick return if possible. */

    if (*m == 0 || *n == 0) {
        // 如果 m 或者 n 为 0，则直接返回
        return 0;
    }

    /*     And when  alpha.eq.zero. */

    if (alpha->r == 0.f && alpha->i == 0.f) {
        // 如果 alpha 为零，则将 B 中所有元素置为零
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                b[i__3].r = 0.f, b[i__3].i = 0.f;
            }
        }
        return 0;
    }

    /*     Start the operations. */

    if (lside) {
        // 左侧矩阵操作
        if (lsame_(transa, "N")) {

            /* Form  B := alpha*inv( A )*B. */

            if (upper) {
                // 上三角矩阵操作
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (alpha->r != 1.f || alpha->i != 0.f) {
                        i__2 = *m;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            i__3 = i__ + j * b_dim1;
                            i__4 = i__ + j * b_dim1;
                            q__1.r = alpha->r * b[i__4].r - alpha->i * b[i__4]
                                .i, q__1.i = alpha->r * b[i__4].i +
                                alpha->i * b[i__4].r;
                            b[i__3].r = q__1.r, b[i__3].i = q__1.i;
                        }
                    }
                    // B := alpha*B，其中 alpha 可能为复数

                    for (k = *m; k >= 1; --k) {
                        // 从最后一行开始逐行处理
                        i__2 = k + j * b_dim1;
                        if (b[i__2].r != 0.f || b[i__2].i != 0.f) {
                            // 如果 B(k,j) 不为零
                            if (nounit) {
                                // 如果不是单位对角矩阵，进行除法操作
                                i__2 = k + j * b_dim1;
                                c_div(&q__1, &b[k + j * b_dim1], &a[k + k *
                                    a_dim1]);
                                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
                            }
                            // B(k,j) := B(k,j) / A(k,k)

                            i__2 = k - 1;
                            for (i__ = 1; i__ <= i__2; ++i__) {
                                // 从第一行到 k-1 行处理
                                i__3 = i__ + j * b_dim1;
                                i__4 = i__ + j * b_dim1;
                                i__5 = k + j * b_dim1;
                                i__6 = i__ + k * a_dim1;
                                q__2.r = b[i__5].r * a[i__6].r - b[i__5].i *
                                    a[i__6].i, q__2.i = b[i__5].r * a[
                                    i__6].i + b[i__5].i * a[i__6].r;
                                q__1.r = b[i__4].r - q__2.r, q__1.i = b[i__4]
                                    .i - q__2.i;
                                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
                            }
                            // B(i,j) := B(i,j) - A(i,k) * B(k,j)
                        }
                    }
                }
            } else {
                // 下三角矩阵操作（此代码段未完全显示）


注释：
这段代码是一个复杂的线性代数操作，主要实现了解线性方程组或矩阵乘法的功能。
/*
   循环遍历矩阵 B 的列 j，针对每一列进行操作：
*/
        if (upper) {
            // 如果 upper 为真，则进行上三角矩阵的操作
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                // 循环遍历矩阵 B 的行 i
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 计算 alpha * B(i,j) 并存储在 temp 中
                    i__3 = i__ + j * b_dim1;
                    q__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i,
                        q__1.i = alpha->r * b[i__3].i + alpha->i * b[i__3].r;
                    temp.r = q__1.r, temp.i = q__1.i;
                    // 如果不需要共轭，处理 A' 的贡献
                    if (noconj) {
                        // 对 i 的每个前驱 k 进行累加
                        i__3 = i__ - 1;
                        for (k = 1; k <= i__3; ++k) {
                            i__4 = k + i__ * a_dim1;
                            i__5 = k + j * b_dim1;
                            // 计算 A'(k,i) * B(k,j) 的贡献并更新 temp
                            q__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5].i,
                                q__2.i = a[i__4].r * b[i__5].i + a[i__4].i * b[i__5].r;
                            q__1.r = temp.r - q__2.r, q__1.i = temp.i - q__2.i;
                            temp.r = q__1.r, temp.i = q__1.i;
                        }
                    }
/* L70: */
                }
/* L80: */
            }
/* L90: */
        }
/* L100: */
/* L110: 结束前一个 if-else 块的大括号 */

                }
                // 如果 nounit 为真，则进行除法操作，将 temp 除以 a[i__ + i__ * a_dim1]
                if (nounit) {
                    c_div(&q__1, &temp, &a[i__ + i__ * a_dim1]);
                    temp.r = q__1.r, temp.i = q__1.i;
                }
            } else {
                // 对 i__ 前的所有 k 进行循环
                i__3 = i__ - 1;
                for (k = 1; k <= i__3; ++k) {
                    // 计算 a[k + i__ * a_dim1] 的共轭乘积与 b[k + j * b_dim1] 的乘积，并更新 temp
                    r_cnjg(&q__3, &a[k + i__ * a_dim1]);
                    i__4 = k + j * b_dim1;
                    q__2.r = q__3.r * b[i__4].r - q__3.i * b[i__4].i, q__2.i = q__3.r * b[i__4].i + q__3.i * b[i__4].r;
                    q__1.r = temp.r - q__2.r, q__1.i = temp.i - q__2.i;
                    temp.r = q__1.r, temp.i = q__1.i;
/* L120: 结束对 k 的循环 */
                }
                // 如果 nounit 为真，则进行除法操作，将 temp 除以 a[i__ + i__ * a_dim1] 的共轭
                if (nounit) {
                    r_cnjg(&q__2, &a[i__ + i__ * a_dim1]);
                    c_div(&q__1, &temp, &q__2);
                    temp.r = q__1.r, temp.i = q__1.i;
                }
            }
            // 更新 b[i__ + j * b_dim1] 的值为 temp
            i__3 = i__ + j * b_dim1;
            b[i__3].r = temp.r, b[i__3].i = temp.i;
/* L130: 结束对 i__ 的循环 */
            }
/* L140: 结束对 j 的循环 */
        }
        // 如果不满足上述条件，则执行如下代码块
        } else {
        // 对 j 的循环
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 对 i__ 的逆序循环
            for (i__ = *m; i__ >= 1; --i__) {
                // 计算 alpha 与 b[i__ + j * b_dim1] 的乘积，并将结果存入 temp
                i__2 = i__ + j * b_dim1;
                q__1.r = alpha->r * b[i__2].r - alpha->i * b[i__2].i, q__1.i = alpha->r * b[i__2].i + alpha->i * b[i__2].r;
                temp.r = q__1.r, temp.i = q__1.i;
                // 如果 noconj 为真，则执行以下代码块
                if (noconj) {
                    // 对 k 的循环
                    i__2 = *m;
                    for (k = i__ + 1; k <= i__2; ++k) {
                        // 计算 a[i__ + k * a_dim1] 与 b[k + j * b_dim1] 的乘积，并更新 temp
                        i__3 = k + i__ * a_dim1;
                        i__4 = k + j * b_dim1;
                        q__2.r = a[i__3].r * b[i__4].r - a[i__3].i * b[i__4].i, q__2.i = a[i__3].r * b[i__4].i + a[i__3].i * b[i__4].r;
                        q__1.r = temp.r - q__2.r, q__1.i = temp.i - q__2.i;
                        temp.r = q__1.r, temp.i = q__1.i;
/* L150: 结束对 k 的循环 */
                    }
                    // 如果 nounit 为真，则进行除法操作，将 temp 除以 a[i__ + i__ * a_dim1]
                    if (nounit) {
                        c_div(&q__1, &temp, &a[i__ + i__ * a_dim1]);
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                } else {
                    // 对 k 的循环
                    i__2 = *m;
                    for (k = i__ + 1; k <= i__2; ++k) {
                        // 计算 a[k + i__ * a_dim1] 的共轭乘积与 b[k + j * b_dim1] 的乘积，并更新 temp
                        r_cnjg(&q__3, &a[k + i__ * a_dim1]);
                        i__3 = k + j * b_dim1;
                        q__2.r = q__3.r * b[i__3].r - q__3.i * b[i__3].i, q__2.i = q__3.r * b[i__3].i + q__3.i * b[i__3].r;
                        q__1.r = temp.r - q__2.r, q__1.i = temp.i - q__2.i;
                        temp.r = q__1.r, temp.i = q__1.i;
/* L160: 结束对 k 的循环 */
                    }
                    // 如果 nounit 为真，则进行除法操作，将 temp 除以 a[i__ + i__ * a_dim1] 的共轭
                    if (nounit) {
                        r_cnjg(&q__2, &a[i__ + i__ * a_dim1]);
                        c_div(&q__1, &temp, &q__2);
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                }
                // 更新 b[i__ + j * b_dim1] 的值为 temp
                i__2 = i__ + j * b_dim1;
                b[i__2].r = temp.r, b[i__2].i = temp.i;
/* L170: 结束对 i__ 的循环 */
            }
/* L180: 结束对 j 的循环 */
        }
        }
    }
    // 如果不满足上述条件，则执行如下代码块
    } else {
    # 如果变量 transa 的值是 "N"，则执行以下代码块
/*           Form  B := alpha*B*inv( A ). */

// 根据上三角矩阵计算 B := alpha * B * inv(A)，其中 alpha 是复数标量
if (upper) {
    // 循环遍历列 j = 1 到 n
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // 如果 alpha 不等于 1 或者不是纯实数，则执行以下操作
        if (alpha->r != 1.f || alpha->i != 0.f) {
            // 循环遍历行 i = 1 到 m
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算新值并存储在 B 的第 (i, j) 元素中
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                q__1.r = alpha->r * b[i__4].r - alpha->i * b[i__4].i, q__1.i = alpha->r * b[i__4].i + alpha->i * b[i__4].r;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L190: */
            }
        }
        // 遍历 A 的列索引小于 j 的部分
        i__2 = j - 1;
        for (k = 1; k <= i__2; ++k) {
            // 如果 A 的元素不是零，则执行以下操作
            i__3 = k + j * a_dim1;
            if (a[i__3].r != 0.f || a[i__3].i != 0.f) {
                // 循环遍历行 i = 1 到 m
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    // 计算新值并存储在 B 的第 (i, j) 元素中
                    i__4 = i__ + j * b_dim1;
                    i__5 = i__ + j * b_dim1;
                    i__6 = k + j * a_dim1;
                    i__7 = i__ + k * b_dim1;
                    q__2.r = a[i__6].r * b[i__7].r - a[i__6].i * b[i__7].i, q__2.i = a[i__6].r * b[i__7].i + a[i__6].i * b[i__7].r;
                    q__1.r = b[i__5].r - q__2.r, q__1.i = b[i__5].i - q__2.i;
                    b[i__4].r = q__1.r, b[i__4].i = q__1.i;
/* L200: */
                }
            }
/* L210: */
        }
        // 如果不是单位矩阵，则执行以下操作
        if (nounit) {
            // 计算 A 的对角元素的倒数，并存储在临时变量 temp 中
            c_div(&q__1, &c_b21, &a[j + j * a_dim1]);
            temp.r = q__1.r, temp.i = q__1.i;
            // 循环遍历行 i = 1 到 m
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算新值并存储在 B 的第 (i, j) 元素中
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                q__1.r = temp.r * b[i__4].r - temp.i * b[i__4].i, q__1.i = temp.r * b[i__4].i + temp.i * b[i__4].r;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L220: */
            }
        }
/* L230: */
    }
} else {
    // 下三角矩阵情况下，从右向左遍历列 j = n 到 1
    for (j = *n; j >= 1; --j) {
        // 如果 alpha 不等于 1 或者不是纯实数，则执行以下操作
        if (alpha->r != 1.f || alpha->i != 0.f) {
            // 循环遍历行 i = 1 到 m
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                // 计算新值并存储在 B 的第 (i, j) 元素中
                i__2 = i__ + j * b_dim1;
                i__3 = i__ + j * b_dim1;
                q__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i, q__1.i = alpha->r * b[i__3].i + alpha->i * b[i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L240: */
            }
/* L250: */
        }
/* L260: */
    }
}
        }
        } else {
/* L290: */
        for (k = 1; k <= *n; ++k) {
            if (nounit) {
            if (noconj) {
/* L300: */
                c_div(&q__1, &c_b21, &a[k + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            } else {
                r_cnjg(&q__2, &a[k + k * a_dim1]);
                c_div(&q__1, &c_b21, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    q__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L310: */
            }
            }
/* L320: */
        }
        }
    }
/* L330: */



注释：

/* L240: */
            }
            }
            i__1 = *n;
            for (k = j + 1; k <= i__1; ++k) {
            i__2 = k + j * a_dim1;
            if (a[i__2].r != 0.f || a[i__2].i != 0.f) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                i__5 = k + j * a_dim1;
                i__6 = i__ + k * b_dim1;
                q__2.r = a[i__5].r * b[i__6].r - a[i__5].i *
                    b[i__6].i, q__2.i = a[i__5].r * b[
                    i__6].i + a[i__5].i * b[i__6].r;
                q__1.r = b[i__4].r - q__2.r, q__1.i = b[i__4]
                    .i - q__2.i;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L250: */
                }
            }
/* L260: */
            }
            if (nounit) {
            c_div(&q__1, &c_b21, &a[j + j * a_dim1]);
            temp.r = q__1.r, temp.i = q__1.i;
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + j * b_dim1;
                i__3 = i__ + j * b_dim1;
                q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    q__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L270: */
            }
            }
/* L280: */
        }
        }
    } else {

/*
             Form  B := alpha*B*inv( A' )
             or    B := alpha*B*inv( conjg( A' ) ).
*/

        if (upper) {
        for (k = *n; k >= 1; --k) {
            if (nounit) {
            if (noconj) {
                c_div(&q__1, &c_b21, &a[k + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            } else {
                r_cnjg(&q__2, &a[k + k * a_dim1]);
                c_div(&q__1, &c_b21, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    q__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;

/* L290: */
            }
            } else {
/* L300: */
            if (noconj) {
                c_div(&q__1, &c_b21, &a[k + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            } else {
                r_cnjg(&q__2, &a[k + k * a_dim1]);
                c_div(&q__1, &c_b21, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    q__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L310: */
            }
            }
/* L320: */
        }
        } else {
/* L330: */
        for (k = 1; k <= *n; ++k) {
            if (nounit) {
            if (noconj) {
/* L340: */
                c_div(&q__1, &c_b21, &a[k + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            } else {
                r_cnjg(&q__2, &a[k + k * a_dim1]);
                c_div(&q__1, &c_b21, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    q__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L350: */
            }
            } else {
/* L360: */
            if (noconj) {
                c_div(&q__1, &c_b21, &a[k + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            } else {
                r_cnjg(&q__2, &a[k + k * a_dim1]);
                c_div(&q__1, &c_b21, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                q__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    q__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L370: */
            }
            }
/* L380: */
        }
        }
    }
/* L390: */
/* L290: */
            }
            }
            i__1 = k - 1;
            for (j = 1; j <= i__1; ++j) {
            i__2 = j + k * a_dim1;
            if (a[i__2].r != 0.f || a[i__2].i != 0.f) {
                if (noconj) {
                i__2 = j + k * a_dim1;
                temp.r = a[i__2].r, temp.i = a[i__2].i;
                } else {
                r_cnjg(&q__1, &a[j + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
                }
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + k * b_dim1;
                q__2.r = temp.r * b[i__5].r - temp.i * b[i__5]
                    .i, q__2.i = temp.r * b[i__5].i +
                    temp.i * b[i__5].r;
                q__1.r = b[i__4].r - q__2.r, q__1.i = b[i__4]
                    .i - q__2.i;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L300: */
                }
            }
/* L310: */
            }
            if (alpha->r != 1.f || alpha->i != 0.f) {
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                q__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3]
                    .i, q__1.i = alpha->r * b[i__3].i +
                    alpha->i * b[i__3].r;
                b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L320: */
            }
            }
/* L330: */
        }
        } else {
        i__1 = *n;
        for (k = 1; k <= i__1; ++k) {
            if (nounit) {
            if (noconj) {
                c_div(&q__1, &c_b21, &a[k + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            } else {
                r_cnjg(&q__2, &a[k + k * a_dim1]);
                c_div(&q__1, &c_b21, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + k * b_dim1;
                i__4 = i__ + k * b_dim1;
                q__1.r = temp.r * b[i__4].r - temp.i * b[i__4].i,
                    q__1.i = temp.r * b[i__4].i + temp.i * b[
                    i__4].r;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L340: */
            }
/* L350: */
            }
/* L360: */
        }
        }
/* L370: */
/* L340: */
            }
            }
            // 循环遍历从 k+1 到 n 的列
            i__2 = *n;
            for (j = k + 1; j <= i__2; ++j) {
            // 计算当前列是否有非零元素
            i__3 = j + k * a_dim1;
            if (a[i__3].r != 0.f || a[i__3].i != 0.f) {
                // 如果不需要共轭，直接复制元素
                if (noconj) {
                i__3 = j + k * a_dim1;
                temp.r = a[i__3].r, temp.i = a[i__3].i;
                } else {
                // 否则，取当前元素的共轭
                r_cnjg(&q__1, &a[j + k * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
                }
                // 对当前列 j 进行更新
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + j * b_dim1;
                i__6 = i__ + k * b_dim1;
                q__2.r = temp.r * b[i__6].r - temp.i * b[i__6]
                    .i, q__2.i = temp.r * b[i__6].i +
                    temp.i * b[i__6].r;
                q__1.r = b[i__5].r - q__2.r, q__1.i = b[i__5]
                    .i - q__2.i;
                b[i__4].r = q__1.r, b[i__4].i = q__1.i;
/* L350: */
                }
            }
/* L360: */
            }
            // 如果 alpha 不等于 1，则对当前列 k 进行更新
            if (alpha->r != 1.f || alpha->i != 0.f) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + k * b_dim1;
                i__4 = i__ + k * b_dim1;
                q__1.r = alpha->r * b[i__4].r - alpha->i * b[i__4]
                    .i, q__1.i = alpha->r * b[i__4].i +
                    alpha->i * b[i__4].r;
                b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L370: */
            }
            }
/* L380: */
        }
        }
    }
    }

    return 0;

/*     End of CTRSM . */

} /* ctrsm_ */

/* Subroutine */ int ctrsv_(char *uplo, char *trans, char *diag, integer *n,
    singlecomplex *a, integer *lda, singlecomplex *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, j, ix, jx, kx, info;
    static singlecomplex temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;


/*
    Purpose
    =======

    CTRSV  solves one of the systems of equations

       A*x = b,   or   A'*x = b,   or   conjg( A' )*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular matrix.

    No test for singularity or near-singularity is included in this
    routine. Such tests must be performed before calling this routine.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.


*/
      TRANS  - CHARACTER*1.
             ! 输入参数，指定要解的方程类型：
             ! 'N' 或 'n' 时，解 A*x = b.
             ! 'T' 或 't' 时，解 A'*x = b.
             ! 'C' 或 'c' 时，解 conjg( A' )*x = b.
             ! 函数结束后保持不变。

      DIAG   - CHARACTER*1.
             ! 输入参数，指定矩阵 A 是否是单位三角阵：
             ! 'U' 或 'u' 时，A 被假定为单位三角阵。
             ! 'N' 或 'n' 时，A 不被假定为单位三角阵。
             ! 函数结束后保持不变。

      N      - INTEGER.
             ! 输入参数，指定矩阵 A 的阶数。
             ! N 必须至少为零。
             ! 函数结束后保持不变。

      A      - COMPLEX          array of DIMENSION ( LDA, n ).
             ! 输入/输出参数，大小为 (LDA, n) 的复数数组。
             ! 当 UPLO = 'U' 或 'u' 时，A 的前 n 行 n 列为上三角部分，
             ! 其他部分未被引用。
             ! 当 UPLO = 'L' 或 'l' 时，A 的前 n 行 n 列为下三角部分，
             ! 其他部分未被引用。
             ! 当 DIAG = 'U' 或 'u' 时，A 的对角元素未被引用，假定为单位矩阵。
             ! 函数结束后保持不变。

      LDA    - INTEGER.
             ! 输入参数，指定数组 A 在调用程序中声明的第一个维度。
             ! LDA 必须至少为 max( 1, n )。
             ! 函数结束后保持不变。

      X      - COMPLEX          array of dimension at least
             ! 输入/输出参数，大小至少为 ( 1 + ( n - 1 )*abs( INCX ) ) 的复数数组。
             ! 在输入时，增量数组 X 必须包含 n 元素的右手边向量 b。
             ! 函数结束后，X 被重写为解向量 x。

      INCX   - INTEGER.
             ! 输入参数，指定数组 X 的元素增量。
             ! INCX 不能为零。
             ! 函数结束后保持不变。

      Further Details
      ===============

      Level 2 Blas routine.
             ! Level 2 BLAS（Basic Linear Algebra Subprograms）例程。

      -- Written on 22-October-1986.
         Jack Dongarra, Argonne National Lab.
         Jeremy Du Croz, Nag Central Office.
         Sven Hammarling, Nag Central Office.
         Richard Hanson, Sandia National Labs.
             ! 作者和相关机构信息。

      =====================================================================

      Test the input parameters.
             ! 测试输入参数。
    /* Parameter adjustments */
    a_dim1 = *lda;                          // 获取矩阵A的第一维度大小
    a_offset = 1 + a_dim1;                  // 计算偏移量
    a -= a_offset;                          // 调整指针a，使其指向正确的位置
    --x;                                    // 将x的指针向前移动一个单位

    /* Function Body */
    info = 0;                               // 初始化info，用于存储错误信息编号
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {  // 检查uplo是否为'U'或'L'
        info = 1;                           // 设置错误信息编号为1
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
        info = 2;                           // 设置错误信息编号为2，检查trans是否为'N'、'T'或'C'
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
        info = 3;                           // 设置错误信息编号为3，检查diag是否为'U'或'N'
    } else if (*n < 0) {
        info = 4;                           // 设置错误信息编号为4，检查n是否为负数
    } else if (*lda < max(1,*n)) {
        info = 6;                           // 设置错误信息编号为6，检查lda是否小于1或n
    } else if (*incx == 0) {
        info = 8;                           // 设置错误信息编号为8，检查incx是否为0
    }
    if (info != 0) {
        xerbla_("CTRSV ", &info);            // 调用错误处理函数xerbla_，并传递错误信息编号
        return 0;                           // 函数提前返回0
    }

    /* Quick return if possible. */
    if (*n == 0) {
        return 0;                           // 如果n为0，直接返回0
    }

    noconj = lsame_(trans, "T");            // 检查trans是否为'T'，设置noconj标志
    nounit = lsame_(diag, "N");             // 检查diag是否为'N'，设置nounit标志

    /*
       Set up the start point in X if the increment is not unity. This
       will be  ( N - 1 )*INCX  too small for descending loops.
    */
    if (*incx <= 0) {
        kx = 1 - (*n - 1) * *incx;          // 计算起始点kx，用于x中非单位增量情况
    } else if (*incx != 1) {
        kx = 1;                             // 如果incx不为1，设置起始点kx为1
    }

    /*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
    */
    if (lsame_(trans, "N")) {

        /* Form  x := inv( A )*x. */
        if (lsame_(uplo, "U")) {            // 如果uplo为'U'
            if (*incx == 1) {               // 如果incx为1
                for (j = *n; j >= 1; --j) {  // 逆序遍历j，计算x := inv(A)*x
                    i__1 = j;
                    if (x[i__1].r != 0.f || x[i__1].i != 0.f) {  // 检查x[j]是否为非零
                        if (nounit) {
                            i__1 = j;
                            c_div(&q__1, &x[j], &a[j + j * a_dim1]);  // 计算x[j] /= A[j][j]
                            x[i__1].r = q__1.r, x[i__1].i = q__1.i;
                        }
                        i__1 = j;
                        temp.r = x[i__1].r, temp.i = x[i__1].i;  // 保存x[j]到temp
                        for (i__ = j - 1; i__ >= 1; --i__) {  // 逆序遍历i，计算x[i] -= A[i][j] * temp
                            i__1 = i__;
                            i__2 = i__;
                            i__3 = i__ + j * a_dim1;
                            q__2.r = temp.r * a[i__3].r - temp.i * a[i__3].i,
                                q__2.i = temp.r * a[i__3].i + temp.i * a[i__3].r;
                            q__1.r = x[i__2].r - q__2.r, q__1.i = x[i__2].i - q__2.i;
                            x[i__1].r = q__1.r, x[i__1].i = q__1.i;
                            /* L10: */
                        }
                    }
                }
            }
        }
    }
/* L20: */
        }
        } else {
        jx = kx + (*n - 1) * *incx;
        for (j = *n; j >= 1; --j) {
            i__1 = jx;
            // 检查 x[jx] 是否为非零，如果是则执行下面的操作
            if (x[i__1].r != 0.f || x[i__1].i != 0.f) {
            // 如果非单位矩阵，计算 x[jx] 除以 a[j + j * a_dim1] 的结果并赋值给 x[jx]
            if (nounit) {
                i__1 = jx;
                c_div(&q__1, &x[jx], &a[j + j * a_dim1]);
                x[i__1].r = q__1.r, x[i__1].i = q__1.i;
            }
            // 保存 x[jx] 的值到临时变量 temp
            i__1 = jx;
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            // 初始化 ix 为 jx，逐步更新 ix 并执行更新 x[ix] 的操作
            ix = jx;
            for (i__ = j - 1; i__ >= 1; --i__) {
                ix -= *incx;
                // 计算更新 x[ix] 的值
                i__1 = ix;
                i__2 = ix;
                i__3 = i__ + j * a_dim1;
                q__2.r = temp.r * a[i__3].r - temp.i * a[i__3].i,
                    q__2.i = temp.r * a[i__3].i + temp.i * a[
                    i__3].r;
                q__1.r = x[i__2].r - q__2.r, q__1.i = x[i__2].i -
                    q__2.i;
                x[i__1].r = q__1.r, x[i__1].i = q__1.i;
/* L30: */
            }
            }
            // 减少 jx 的值，以便处理下一个循环
            jx -= *incx;
/* L40: */
        }
        }
    } else {
        if (*incx == 1) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = j;
            // 检查 x[j] 是否为非零，如果是则执行下面的操作
            if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
            // 如果非单位矩阵，计算 x[j] 除以 a[j + j * a_dim1] 的结果并赋值给 x[j]
            if (nounit) {
                i__2 = j;
                c_div(&q__1, &x[j], &a[j + j * a_dim1]);
                x[i__2].r = q__1.r, x[i__2].i = q__1.i;
            }
            // 保存 x[j] 的值到临时变量 temp
            i__2 = j;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            // 从 j+1 开始更新 x[i] 的值
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                // 计算更新 x[i] 的值
                i__3 = i__;
                i__4 = i__;
                i__5 = i__ + j * a_dim1;
                q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    q__2.i = temp.r * a[i__5].i + temp.i * a[
                    i__5].r;
                q__1.r = x[i__4].r - q__2.r, q__1.i = x[i__4].i -
                    q__2.i;
                x[i__3].r = q__1.r, x[i__3].i = q__1.i;
/* L50: */
            }
            }
/* L60: */
        }
        } else {
        jx = kx;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = jx;
            // 检查 x[jx] 是否为非零，如果是则执行下面的操作
            if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
            // 如果非单位矩阵，计算 x[jx] 除以 a[j + j * a_dim1] 的结果并赋值给 x[jx]
            if (nounit) {
                i__2 = jx;
                c_div(&q__1, &x[jx], &a[j + j * a_dim1]);
                x[i__2].r = q__1.r, x[i__2].i = q__1.i;
            }
            // 保存 x[jx] 的值到临时变量 temp
            i__2 = jx;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            // 初始化 ix 为 jx，逐步更新 ix 并执行更新 x[ix] 的操作
            ix = jx;
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                ix += *incx;
                // 计算更新 x[ix] 的值
                i__3 = ix;
                i__4 = ix;
                i__5 = i__ + j * a_dim1;
                q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    q__2.i = temp.r * a[i__5].i + temp.i * a[
                    i__5].r;
                q__1.r = x[i__4].r - q__2.r, q__1.i = x[i__4].i -
                    q__2.i;
                x[i__3].r = q__1.r, x[i__3].i = q__1.i;
/* L70: */
            }
            }
            jx += *incx;
/* L80: */
        }
        }
    }
    } else {

/*        Form  x := inv( A' )*x  or  x := inv( conjg( A' ) )*x. */

    // 如果上三角部分被使用
    if (lsame_(uplo, "U")) {
        // 如果增量为1
        if (*incx == 1) {
        // 循环遍历列数
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 临时变量 temp 存储 x[j]
            i__2 = j;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            // 如果不进行共轭
            if (noconj) {
            // 遍历行数
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵 A 的元素与 x 的乘积
                i__3 = i__ + j * a_dim1;
                i__4 = i__;
                q__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, q__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
                // 计算更新后的 temp
                q__1.r = temp.r - q__2.r, q__1.i = temp.i -
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L90: */
            }
            // 如果不是单位对角阵，则进行除法操作
            if (nounit) {
                c_div(&q__1, &temp, &a[j + j * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            } else {
            // 遍历行数
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算共轭转置后的矩阵 A 的元素与 x 的乘积
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__3 = i__;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                    q__2.i = q__3.r * x[i__3].i + q__3.i * x[
                    i__3].r;
                // 计算更新后的 temp
                q__1.r = temp.r - q__2.r, q__1.i = temp.i -
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
/* L100: */
            }
            // 如果不是单位对角阵，则进行除法操作
            if (nounit) {
                r_cnjg(&q__2, &a[j + j * a_dim1]);
                c_div(&q__1, &temp, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            }
            // 更新 x[j] 的值为 temp
            i__2 = j;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L110: */
        }
        } else {
        // 设置起始点 jx 为 kx
        jx = kx;
        // 循环遍历列数
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 设置起始点 ix 为 kx
            ix = kx;
            // 临时变量 temp 存储 x[jx]
            i__2 = jx;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            // 如果不进行共轭
            if (noconj) {
            // 遍历行数
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵 A 的元素与 x[ix] 的乘积
                i__3 = i__ + j * a_dim1;
                i__4 = ix;
                q__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, q__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
                // 计算更新后的 temp
                q__1.r = temp.r - q__2.r, q__1.i = temp.i -
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
                ix += *incx;
/* L100: */
            }
            // 如果不是单位对角阵，则进行除法操作
            if (nounit) {
                c_div(&q__1, &temp, &a[j + j * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            } else {
            // 遍历行数
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算共轭转置后的矩阵 A 的元素与 x[ix] 的乘积
                r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                i__3 = ix;
                q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                    q__2.i = q__3.r * x[i__3].i + q__3.i * x[
                    i__3].r;
                // 计算更新后的 temp
                q__1.r = temp.r - q__2.r, q__1.i = temp.i -
                    q__2.i;
                temp.r = q__1.r, temp.i = q__1.i;
                ix += *incx;
/* L100: */
            }
            // 如果不是单位对角阵，则进行除法操作
            if (nounit) {
                r_cnjg(&q__2, &a[j + j * a_dim1]);
                c_div(&q__1, &temp, &q__2);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            }
            // 更新 x[jx] 的值为 temp
            i__2 = jx;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L110: */
        }
        }
    }
    }
/* L120: 结束之前的 if 语句块，准备进入下一个 if-else 语句块 */

            }
            // 如果没有单位对角线，需要进行除法运算
            if (nounit) {
                // 计算除法结果并将结果赋给 temp 变量
                c_div(&q__1, &temp, &a[j + j * a_dim1]);
                temp.r = q__1.r, temp.i = q__1.i;
            }
            // 这里的 else 对应上面的 if 条件为 false 的情况
            } else {
                // 循环处理 j 前的元素 i
                i__2 = j - 1;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 求解复数的共轭
                    r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                    // 计算乘法结果
                    i__3 = ix;
                    q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i,
                        q__2.i = q__3.r * x[i__3].i + q__3.i * x[
                        i__3].r;
                    // 计算减法结果
                    q__1.r = temp.r - q__2.r, q__1.i = temp.i -
                        q__2.i;
                    temp.r = q__1.r, temp.i = q__1.i;
                    // 更新 ix
                    ix += *incx;
/* L130: */         // 标签 L130
                }
                // 如果没有单位对角线，需要进行除法运算
                if (nounit) {
                    // 求解复数的共轭
                    r_cnjg(&q__2, &a[j + j * a_dim1]);
                    // 计算除法结果并将结果赋给 temp 变量
                    c_div(&q__1, &temp, &q__2);
                    temp.r = q__1.r, temp.i = q__1.i;
                }
            }
            // 更新 jx 对应的元素
            i__2 = jx;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
            // 更新 jx
            jx += *incx;
/* L140: */     // 标签 L140
        }
        // 如果上面的 if 语句条件不成立
        }
    } else {
        // 如果 *incx 等于 1
        if (*incx == 1) {
            // 逆序循环 j
            for (j = *n; j >= 1; --j) {
                // 复制 x[j] 的值到 temp
                i__1 = j;
                temp.r = x[i__1].r, temp.i = x[i__1].i;
                // 如果不共轭
                if (noconj) {
                    // 逆序循环处理 i
                    i__1 = j + 1;
                    for (i__ = *n; i__ >= i__1; --i__) {
                        // 计算乘法结果
                        i__2 = i__ + j * a_dim1;
                        i__3 = i__;
                        q__2.r = a[i__2].r * x[i__3].r - a[i__2].i * x[
                            i__3].i, q__2.i = a[i__2].r * x[i__3].i +
                            a[i__2].i * x[i__3].r;
                        // 计算减法结果
                        q__1.r = temp.r - q__2.r, q__1.i = temp.i -
                            q__2.i;
                        temp.r = q__1.r, temp.i = q__1.i;
/* L150: */             // 标签 L150
                    }
                    // 如果没有单位对角线，需要进行除法运算
                    if (nounit) {
                        // 计算除法结果并将结果赋给 temp 变量
                        c_div(&q__1, &temp, &a[j + j * a_dim1]);
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                } else {
                    // 逆序循环处理 i
                    i__1 = j + 1;
                    for (i__ = *n; i__ >= i__1; --i__) {
                        // 求解复数的共轭
                        r_cnjg(&q__3, &a[i__ + j * a_dim1]);
                        // 计算乘法结果
                        i__2 = i__;
                        q__2.r = q__3.r * x[i__2].r - q__3.i * x[i__2].i,
                            q__2.i = q__3.r * x[i__2].i + q__3.i * x[
                            i__2].r;
                        // 计算减法结果
                        q__1.r = temp.r - q__2.r, q__1.i = temp.i -
                            q__2.i;
                        temp.r = q__1.r, temp.i = q__1.i;
/* L160: */             // 标签 L160
                    }
                    // 如果没有单位对角线，需要进行除法运算
                    if (nounit) {
                        // 求解复数的共轭
                        r_cnjg(&q__2, &a[j + j * a_dim1]);
                        // 计算除法结果并将结果赋给 temp 变量
                        c_div(&q__1, &temp, &q__2);
                        temp.r = q__1.r, temp.i = q__1.i;
                    }
                }
                // 更新 x[j] 对应的元素
                i__1 = j;
                x[i__1].r = temp.r, x[i__1].i = temp.i;
/*
   Purpose
   =======

      DAXPY constant times a vector plus a vector.
      uses unrolled loops for increments equal to one.

   Further Details
   ===============

      jack dongarra, linpack, 3/11/78.
      modified 12/3/93, array(1) declarations changed to array(*)

   =====================================================================
*/

/* Subroutine */ int daxpy_(integer *n, doublereal *da, doublereal *dx,
    integer *incx, doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, ix, iy;

    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*da == 0.) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

    /*
       code for unequal increments or equal increments
       not equal to 1
    */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[iy] += *da * dx[ix];
        ix += *incx;
        iy += *incy;
        /* L170: */
    }

    return 0;

/*     End of DAXPY . */

} /* daxpy_ */


注释：


/*
   Purpose
   =======

      DAXPY是一个计算常数乘以一个向量加上另一个向量的子程序。
      对于增量为1的情况使用展开循环。

   Further Details
   ===============

      Jack Dongarra编写，Linpack，3/11/78。
      修改于12/3/93，将数组声明从array(1)更改为array(*)

   =====================================================================
*/

/* Subroutine */ int daxpy_(integer *n, doublereal *da, doublereal *dx,
    integer *incx, doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, ix, iy;

    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    如果向量的长度小于等于0，则直接返回。
    if (*n <= 0) {
        return 0;
    }
    如果常数因子da为0，则直接返回。
    if (*da == 0.) {
        return 0;
    }
    如果增量incx和incy都为1，则跳转到标签L20处进行处理。
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

    /*
       对于不等增量或者增量不等于1的情况的代码处理
    */

    初始化ix和iy为1。
    ix = 1;
    iy = 1;
    如果增量incx小于0，则重新计算ix的初始位置。
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    如果增量incy小于0，则重新计算iy的初始位置。
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    循环遍历向量，进行常数乘法后的加法操作。
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[iy] += *da * dx[ix];
        ix += *incx;
        iy += *incy;
        /* L170: */
    }

    返回0表示成功执行。

/*     End of DAXPY . */

} /* daxpy_ */
    # 使用增量值增加 iy 变量的值
    iy += *incy;
/*
    Purpose
    =======

       DCOPY copies a vector, x, to a vector, y.
       uses unrolled loops for increments equal to one.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

/* Subroutine */ int dcopy_(integer *n, doublereal *dx, integer *incx,
    doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;

    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

    /*
              code for unequal increments or equal increments
                not equal to 1
    */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[iy] = dx[ix];
        ix += *incx;
        iy += *incy;
        /* L10: */
    }
    return 0;

    /*
              code for both increments equal to 1


              clean-up loop
    */

L20:
    m = *n % 7;
    if (m == 0) {
        goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[i__] = dx[i__];
        /* L30: */
    }
    if (*n < 7) {
        return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 7) {
        dy[i__] = dx[i__];
        dy[i__ + 1] = dx[i__ + 1];
        dy[i__ + 2] = dx[i__ + 2];
        dy[i__ + 3] = dx[i__ + 3];
        dy[i__ + 4] = dx[i__ + 4];
        dy[i__ + 5] = dx[i__ + 5];
        dy[i__ + 6] = dx[i__ + 6];
        /* L50: */
    }
    return 0;
} /* dcopy_ */


注释：


/*
    Purpose
    =======

       DCOPY copies a vector, x, to a vector, y.
       uses unrolled loops for increments equal to one.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

/* Subroutine */ int dcopy_(integer *n, doublereal *dx, integer *incx,
    doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;

    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

    /*
              code for unequal increments or equal increments
                not equal to 1
    */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[iy] = dx[ix];
        ix += *incx;
        iy += *incy;
        /* L10: */
    }
    return 0;

    /*
              code for both increments equal to 1


              clean-up loop
    */

L20:
    m = *n % 7;
    if (m == 0) {
        goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[i__] = dx[i__];
        /* L30: */
    }
    if (*n < 7) {
        return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 7) {
        dy[i__] = dx[i__];
        dy[i__ + 1] = dx[i__ + 1];
        dy[i__ + 2] = dx[i__ + 2];
        dy[i__ + 3] = dx[i__ + 3];
        dy[i__ + 4] = dx[i__ + 4];
        dy[i__ + 5] = dx[i__ + 5];
        dy[i__ + 6] = dx[i__ + 6];
        /* L50: */
    }
    return 0;
} /* dcopy_ */
    # 定义静态的双精度浮点数变量 dtemp
    static doublereal dtemp;
/*
    Purpose
    =======

       DDOT forms the dot product of two vectors.
       uses unrolled loops for increments equal to one.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

/* Parameter adjustments */
--dy;   // 减少 dy 指针以访问数组元素
--dx;   // 减少 dx 指针以访问数组元素

/* Function Body */
ret_val = 0.;   // 初始化返回值为 0
dtemp = 0.;      // 初始化临时变量 dtemp 为 0
if (*n <= 0) {   // 如果 n 小于等于 0，则直接返回 ret_val
return ret_val;
}
if (*incx == 1 && *incy == 1) {   // 如果增量 incx 和 incy 都为 1，则跳转到 L20
goto L20;
}

/*
          code for unequal increments or equal increments
            not equal to 1
*/

ix = 1;   // 初始化 ix 为 1
iy = 1;   // 初始化 iy 为 1
if (*incx < 0) {   // 如果 incx 小于 0，则重新计算 ix
ix = (-(*n) + 1) * *incx + 1;
}
if (*incy < 0) {   // 如果 incy 小于 0，则重新计算 iy
iy = (-(*n) + 1) * *incy + 1;
}
i__1 = *n;   // 循环次数设为 n
for (i__ = 1; i__ <= i__1; ++i__) {
dtemp += dx[ix] * dy[iy];   // 计算点积的累加
ix += *incx;   // 根据增量更新 ix
iy += *incy;   // 根据增量更新 iy
/* L10: */
}
ret_val = dtemp;   // 将累加结果赋给返回值
return ret_val;

/*
          code for both increments equal to 1


          clean-up loop
*/

L20:   // 标签 L20，开始处理增量为 1 的情况
m = *n % 5;   // 计算余数 m
if (m == 0) {   // 如果 m 等于 0，则跳转到 L40
goto L40;
}
i__1 = m;   // 循环次数为 m
for (i__ = 1; i__ <= i__1; ++i__) {
dtemp += dx[i__] * dy[i__];   // 计算点积的累加
/* L30: */
}
if (*n < 5) {   // 如果 n 小于 5，则跳转到 L60
goto L60;
}
L40:   // 标签 L40，处理主循环，增量为 1 的情况
mp1 = m + 1;   // 计算 mp1
i__1 = *n;   // 循环次数为 n
for (i__ = mp1; i__ <= i__1; i__ += 5) {
dtemp = dtemp + dx[i__] * dy[i__] + dx[i__ + 1] * dy[i__ + 1] + dx[
    i__ + 2] * dy[i__ + 2] + dx[i__ + 3] * dy[i__ + 3] + dx[i__ +
    4] * dy[i__ + 4];   // 计算点积的累加，使用了展开循环
/* L50: */
}
L60:   // 标签 L60，增量为 1 的情况结束处理
ret_val = dtemp;   // 将累加结果赋给返回值
return ret_val;
} /* ddot_ */

/* Subroutine */ int dgemm_(char *transa, char *transb, integer *m, integer *
    n, integer *k, doublereal *alpha, doublereal *a, integer *lda,
    doublereal *b, integer *ldb, doublereal *beta, doublereal *c__,
    integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3;

    /* Local variables */
    static integer i__, j, l, info;
    static logical nota, notb;
    static doublereal temp;
    extern logical lsame_(char *, char *);
    static integer nrowa, nrowb;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DGEMM  performs one of the matrix-matrix operations

       C := alpha*op( A )*op( B ) + beta*C,

    where  op( X ) is one of

       op( X ) = X   or   op( X ) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Arguments
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n',  op( A ) = A.

                TRANSA = 'T' or 't',  op( A ) = A'.

                TRANSA = 'C' or 'c',  op( A ) = A'.

             Unchanged on exit.


*/
    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in
             the matrix multiplication as follows:

                TRANSB = 'N' or 'n',  op( B ) = B.

                TRANSB = 'T' or 't',  op( B ) = B'.

                TRANSB = 'C' or 'c',  op( B ) = B'.

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( A )  and of the  matrix  C.  M  must  be at least  zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( B ) and the number of columns of the matrix C. N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.
    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).
             
    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.
             
    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

    =====================================================================


       Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
       transposed and set  NROWA and  NROWB  as the number of rows
       and  columns of  A  and the  number of  rows  of  B  respectively.
    /* Parameter adjustments */
    a_dim1 = *lda;  // Adjust dimension of A
    a_offset = 1 + a_dim1;  // Calculate offset for A
    a -= a_offset;  // Adjust A to start from a_offset
    b_dim1 = *ldb;  // Adjust dimension of B
    b_offset = 1 + b_dim1;  // Calculate offset for B
    b -= b_offset;  // Adjust B to start from b_offset
    c_dim1 = *ldc;  // Adjust dimension of C
    c_offset = 1 + c_dim1;  // Calculate offset for C
    c__ -= c_offset;  // Adjust C to start from c_offset

    /* Function Body */
    nota = lsame_(transa, "N");  // Check if transa is 'N'
    notb = lsame_(transb, "N");  // Check if transb is 'N'
    if (nota) {
        nrowa = *m;  // Set number of rows in A
    } else {
        nrowa = *k;  // Set number of rows in A
    }
    if (notb) {
        nrowb = *k;  // Set number of rows in B
    } else {
        nrowb = *n;  // Set number of rows in B
    }

    /* Test the input parameters. */
    info = 0;  // Initialize error flag
    if (! nota && ! lsame_(transa, "C") && ! lsame_(transa, "T")) {
        info = 1;  // Set error flag for incorrect transa
    } else if (! notb && ! lsame_(transb, "C") && ! lsame_(transb, "T")) {
        info = 2;  // Set error flag for incorrect transb
    } else if (*m < 0) {
        info = 3;  // Set error flag for negative m
    } else if (*n < 0) {
        info = 4;  // Set error flag for negative n
    } else if (*k < 0) {
        info = 5;  // Set error flag for negative k
    } else if (*lda < max(1,nrowa)) {
        info = 8;  // Set error flag for lda too small
    } else if (*ldb < max(1,nrowb)) {
        info = 10;  // Set error flag for ldb too small
    } else if (*ldc < max(1,*m)) {
        info = 13;  // Set error flag for ldc too small
    }
    if (info != 0) {
        xerbla_("DGEMM ", &info);  // Report error using xerbla
        return 0;  // Exit function with error
    }

    /* Quick return if possible. */
    if (*m == 0 || *n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
        return 0;  // Exit function early if no computation needed
    }

    /* And if alpha.eq.zero. */
    if (*alpha == 0.) {
        if (*beta == 0.) {
            // Set C to zero if alpha and beta are both zero
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    c__[i__ + j * c_dim1] = 0.;
                }
            }
        } else {
            // Scale C by beta if alpha is zero and beta is not zero
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                }
            }
        }
        return 0;  // Exit function after scaling C
    }

    /* Start the operations. */
    if (notb) {
        if (nota) {
            // Compute C := alpha*A*B + beta*C
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (*beta == 0.) {
                    // Initialize C to zero if beta is zero
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                } else if (*beta != 1.) {
                    // Scale C by beta if beta is not one
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
                // Perform matrix multiplication A*B and update C
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    if (b[l + j * b_dim1] != 0.) {
                        temp = *alpha * b[l + j * b_dim1];
                        i__3 = *m;
                        for (i__ = 1; i__ <= i__3; ++i__) {
                            c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
                        }
                    }
                }
            }
        } else {
            // Compute C := alpha*A'*B + beta*C
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    temp = 0.;
                    i__3 = *k;
                    for (l = 1; l <= i__3; ++l) {
                        temp += a[l + i__ * a_dim1] * b[l + j * b_dim1];
                    }
                    c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[i__ + j * c_dim1];
                }
            }
        }
    } else {
        // Handle the case when transb is not 'N'
        if (nota) {
            // Compute C := alpha*A*B' + beta*C
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (*beta == 0.) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                } else if (*beta != 1.) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    if (b[j + l * b_dim1] != 0.) {
                        temp = *alpha * b[j + l * b_dim1];
                        i__3 = *m;
                        for (i__ = 1; i__ <= i__3; ++i__) {
                            c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
                        }
                    }
                }
            }
        } else {
            // Compute C := alpha*A'*B' + beta*C
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (*beta == 0.) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                } else if (*beta != 1.) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    if (b[j + l * b_dim1] != 0.) {
                        temp = *alpha * b[j + l * b_dim1];
                        i__3 = *m;
                        for (i__ = 1; i__ <= i__3; ++i__) {
                            c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
                        }
                    }
                }
            }
        }
    }
/* L100: */
            }
            if (*beta == 0.) {
            c__[i__ + j * c_dim1] = *alpha * temp;
            } else {
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
            }
/* L110: */
        }
/* L120: */
        }
    }
    } else {
    if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        if (*beta == 0.) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            c__[i__ + j * c_dim1] = 0.;
/* L130: */
            }
        } else if (*beta != 1.) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L140: */
            }
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            if (b[j + l * b_dim1] != 0.) {
            temp = *alpha * b[j + l * b_dim1];
            i__3 = *m;
            for (i__ = 1; i__ <= i__3; ++i__) {
                c__[i__ + j * c_dim1] += temp * a[i__ + l *
                    a_dim1];
/* L150: */
            }
            }
/* L160: */
        }
/* L170: */
        }
    } else {

/*           Form  C := alpha*A'*B' + beta*C */

        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            temp = 0.;
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            temp += a[l + i__ * a_dim1] * b[j + l * b_dim1];
/* L180: */
            }
            if (*beta == 0.) {
            c__[i__ + j * c_dim1] = *alpha * temp;
            } else {
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
            }
/* L190: */
        }
/* L200: */
        }
    }
    }

    return 0;

/*     End of DGEMM . */

} /* dgemm_ */

/* Subroutine */ int dgemv_(char *trans, integer *m, integer *n, doublereal *
    alpha, doublereal *a, integer *lda, doublereal *x, integer *incx,
    doublereal *beta, doublereal *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static doublereal temp;
    static integer lenx, leny;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DGEMV  performs one of the matrix-vector operations

       y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ==========

*/
    TRANS  - CHARACTER*1.
             在进入时，TRANS 指定要执行的操作，具体如下：

                TRANS = 'N' 或 'n'   y := alpha*A*x + beta*y.

                TRANS = 'T' 或 't'   y := alpha*A'*x + beta*y.

                TRANS = 'C' 或 'c'   y := alpha*A'*x + beta*y.

             函数返回时，TRANS 的值不变。

    M      - INTEGER.
             在进入时，M 指定矩阵 A 的行数。
             M 必须至少为零。
             函数返回时，M 的值不变。

    N      - INTEGER.
             在进入时，N 指定矩阵 A 的列数。
             N 必须至少为零。
             函数返回时，N 的值不变。

    ALPHA  - DOUBLE PRECISION.
             在进入时，ALPHA 指定标量 alpha 的值。
             函数返回时，ALPHA 的值不变。

    A      - DOUBLE PRECISION 数组，维度为 ( LDA, n ).
             进入之前，数组 A 的前 m 行 n 列必须包含系数矩阵。
             函数返回时，数组 A 的内容不变。

    LDA    - INTEGER.
             进入时，LDA 指定数组 A 的第一个维度，即在调用程序中声明的维度。
             LDA 必须至少为 max( 1, m )。
             函数返回时，LDA 的值不变。

    X      - DOUBLE PRECISION 数组，至少维度为
             ( 1 + ( n - 1 )*abs( INCX ) ) 当 TRANS = 'N' 或 'n' 时，
             或至少维度为 ( 1 + ( m - 1 )*abs( INCX ) ) 其他情况下。
             进入之前，增量数组 X 必须包含向量 x。
             函数返回时，数组 X 的内容不变。

    INCX   - INTEGER.
             进入时，INCX 指定 X 中元素的增量。INCX 不能为零。
             函数返回时，INCX 的值不变。

    BETA   - DOUBLE PRECISION.
             进入时，BETA 指定标量 beta 的值。当 BETA 被设为零时，输入时 Y 不需要设置。
             函数返回时，BETA 的值不变。

    Y      - DOUBLE PRECISION 数组，至少维度为
             ( 1 + ( m - 1 )*abs( INCY ) ) 当 TRANS = 'N' 或 'n' 时，
             或至少维度为 ( 1 + ( n - 1 )*abs( INCY ) ) 其他情况下。
             在进入时，如果 BETA 非零，则增量数组 Y 必须包含向量 y。返回时，数组 Y 被更新为更新后的向量 y。

    INCY   - INTEGER.
             进入时，INCY 指定 Y 中元素的增量。INCY 不能为零。
             函数返回时，INCY 的值不变。

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================


       Test the input parameters.
    /* Parameter adjustments */
    // 设置矩阵 A 的维度和偏移量
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    // 将指针 a 调整到正确的起始位置
    a -= a_offset;
    // 调整向量 x 和 y 的起始位置
    --x;
    --y;

    /* Function Body */
    // 初始化 info 变量，用于记录错误信息
    info = 0;
    // 检查参数 trans 的合法性
    if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
        info = 1;
    } else if (*m < 0) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*lda < max(1,*m)) {
        info = 6;
    } else if (*incx == 0) {
        info = 8;
    } else if (*incy == 0) {
        info = 11;
    }
    // 如果有错误信息，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("DGEMV ", &info);
        return 0;
    }

/*     Quick return if possible. */
    // 如果某些参数使得没有需要计算的内容，直接返回
    if (*m == 0 || *n == 0 || *alpha == 0. && *beta == 1.) {
        return 0;
    }

/*
       Set  LENX  and  LENY, the lengths of the vectors x and y, and set
       up the start points in  X  and  Y.
*/
    // 根据 trans 参数设置向量 x 和 y 的长度 LENX 和 LENY，并初始化它们的起始点
    if (lsame_(trans, "N")) {
        lenx = *n;
        leny = *m;
    } else {
        lenx = *m;
        leny = *n;
    }
    // 设置向量 x 的起始点 kx 和向量 y 的起始点 ky
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (leny - 1) * *incy;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.

       First form  y := beta*y.
*/
    // 开始计算。在这个版本中，通过一次对 A 的遍历来顺序访问 A 的元素。
    // 首先计算 y := beta*y
    if (*beta != 1.) {
        if (*incy == 1) {
            if (*beta == 0.) {
                // 如果 beta 为 0，则将 y 全部置为 0
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = 0.;
                }
            } else {
                // 否则按照 beta 的值对 y 进行缩放
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = *beta * y[i__];
                }
            }
        } else {
            // incy 不为 1 的情况下，按照指定步长对 y 进行处理
            iy = ky;
            if (*beta == 0.) {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = 0.;
                    iy += *incy;
                }
            } else {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = *beta * y[iy];
                    iy += *incy;
                }
            }
        }
    }
    // 如果 alpha 为 0，则直接返回
    if (*alpha == 0.) {
        return 0;
    }
    // 如果 trans 参数为 "N"，则执行以下计算
    if (lsame_(trans, "N")) {

/*        Form  y := alpha*A*x + y. */

        // 设置 x 的起始点 jx
        jx = kx;
        if (*incy == 1) {
            // 当 incy 为 1 时，按列计算
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.) {
                    temp = *alpha * x[jx];
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        // 计算 y[i__] += alpha * A[i__, j] * x[j]
                        y[i__] += temp * a[i__ + j * a_dim1];
                    }
                }
                // 更新 x 的位置
                jx += *incx;
            }
        } else {
            // 当 incy 不为 1 时，按步长 ky 计算
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.) {
                    temp = *alpha * x[jx];
                    iy = ky;
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        // 计算 y[iy] += alpha * A[i__, j] * x[j]
                        y[iy] += temp * a[i__ + j * a_dim1];
                        // 更新 y 的位置
                        iy += *incy;
                    }
                }
                // 更新 x 的位置
                jx += *incx;
            }
        }
    } else {

/*        Form  y := alpha*A'*x + y. */

        // 当 trans 参数为非 "N" 时，执行相应的计算
        jy = ky;
    # 如果增量步长为1，则进入循环
    if (*incx == 1) {
        # 对于每个列索引 j，执行以下操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            # 初始化临时变量 temp 为 0
            temp = 0.;
            # 对于每个行索引 i，执行以下操作
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                # 累加矩阵 a 的第 i 行第 j 列元素与向量 x 的第 i 个元素的乘积到 temp
                temp += a[i__ + j * a_dim1] * x[i__];
/* L90: */
        }
        y[jy] += *alpha * temp;
        jy += *incy;
/* L100: */
        }
    } else {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        temp = 0.;
        ix = kx;
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            temp += a[i__ + j * a_dim1] * x[ix];
            ix += *incx;
/* L110: */
        }
        y[jy] += *alpha * temp;
        jy += *incy;
/* L120: */
        }
    }
    }

    return 0;

/*     End of DGEMV . */

} /* dgemv_ */

/* Subroutine */ int dger_(integer *m, integer *n, doublereal *alpha,
    doublereal *x, integer *incx, doublereal *y, integer *incy,
    doublereal *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, jy, kx, info;
    static doublereal temp;
    extern /* Subroutine */ int xerbla_(char *, integer *);

/* 
    Purpose
    =======

    DGER   performs the rank 1 operation

       A := alpha*x*y' + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.

    Arguments
    ==========

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( m - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the m
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    Y      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y.
             Unchanged on exit.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients. On exit, A is
             overwritten by the updated matrix.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    Further Details
    ===============

    Level 2 Blas routine.

*/
    # 以下是一段注释和头部信息，记录了代码的编写日期、作者及其所在机构
    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.
    
    # 分隔线，用于标记代码的不同部分
    =====================================================================
    
    # 检测输入参数的有效性，这里是代码执行前的一些前期检查
    Test the input parameters.
    /* Parameter adjustments */
    --x;  // 将指针 x 指向的位置向前移动一位，使其指向正确的数组起始位置
    --y;  // 将指针 y 指向的位置向前移动一位，使其指向正确的数组起始位置
    a_dim1 = *lda;  // 将 a_dim1 设为 lda 指向的值，表示二维数组 a 的第一维大小
    a_offset = 1 + a_dim1;  // 计算 a 数组的偏移量，用于正确访问二维数组 a
    a -= a_offset;  // 调整数组 a 的指针，使其指向正确的起始位置

    /* Function Body */
    info = 0;  // 初始化 info 为 0，用于记录函数执行中的错误信息
    if (*m < 0) {  // 检查 m 的值是否小于 0
        info = 1;  // 如果是，则将 info 设为 1，表示参数 m 非法
    } else if (*n < 0) {  // 检查 n 的值是否小于 0
        info = 2;  // 如果是，则将 info 设为 2，表示参数 n 非法
    } else if (*incx == 0) {  // 检查 incx 的值是否为 0
        info = 5;  // 如果是，则将 info 设为 5，表示参数 incx 非法
    } else if (*incy == 0) {  // 检查 incy 的值是否为 0
        info = 7;  // 如果是，则将 info 设为 7，表示参数 incy 非法
    } else if (*lda < max(1,*m)) {  // 检查 lda 的值是否小于 max(1, m)
        info = 9;  // 如果是，则将 info 设为 9，表示参数 lda 非法
    }
    if (info != 0) {  // 如果 info 不等于 0，表示有参数非法
        xerbla_("DGER  ", &info);  // 调用错误处理函数 xerbla_，并传递错误信息 info
        return 0;  // 返回 0，表示函数执行失败
    }

    /* Quick return if possible. */
    if (*m == 0 || *n == 0 || *alpha == 0.) {  // 检查是否满足快速返回的条件
        return 0;  // 如果满足，则直接返回，不执行后续的矩阵更新操作
    }

    /*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
    */

    if (*incy > 0) {  // 检查 incy 的值是否大于 0
        jy = 1;  // 如果是，则将 jy 设为 1，表示按照正向顺序访问向量 y
    } else {  // 如果 incy 小于等于 0
        jy = 1 - (*n - 1) * *incy;  // 计算 jy 的起始位置，以确保正确访问向量 y
    }
    if (*incx == 1) {  // 检查 incx 的值是否为 1
        i__1 = *n;  // 将 i__1 设为 n，表示循环的上界为向量长度 n
        for (j = 1; j <= i__1; ++j) {  // 循环遍历向量 y 的每个元素
            if (y[jy] != 0.) {  // 检查向量 y 的当前元素是否为非零
                temp = *alpha * y[jy];  // 计算临时变量 temp 的值
                i__2 = *m;  // 将 i__2 设为 m，表示内层循环的上界为矩阵行数 m
                for (i__ = 1; i__ <= i__2; ++i__) {  // 循环遍历矩阵的每一行
                    a[i__ + j * a_dim1] += x[i__] * temp;  // 更新矩阵 a 的对应元素
                    /* L10: */
                }
            }
            jy += *incy;  // 调整 jy 的值，以正确访问向量 y 的下一个元素
            /* L20: */
        }
    } else {  // 如果 incx 不等于 1
        if (*incx > 0) {  // 检查 incx 的值是否大于 0
            kx = 1;  // 如果是，则将 kx 设为 1，表示按照正向顺序访问向量 x
        } else {  // 如果 incx 小于等于 0
            kx = 1 - (*m - 1) * *incx;  // 计算 kx 的起始位置，以确保正确访问向量 x
        }
        i__1 = *n;  // 将 i__1 设为 n，表示外层循环的上界为向量长度 n
        for (j = 1; j <= i__1; ++j) {  // 循环遍历向量 y 的每个元素
            if (y[jy] != 0.) {  // 检查向量 y 的当前元素是否为非零
                temp = *alpha * y[jy];  // 计算临时变量 temp 的值
                ix = kx;  // 将 ix 设为 kx，表示内层循环访问向量 x 的起始位置
                i__2 = *m;  // 将 i__2 设为 m，表示内层循环的上界为矩阵行数 m
                for (i__ = 1; i__ <= i__2; ++i__) {  // 循环遍历矩阵的每一行
                    a[i__ + j * a_dim1] += x[ix] * temp;  // 更新矩阵 a 的对应元素
                    ix += *incx;  // 调整 ix 的值，以正确访问向量 x 的下一个元素
                    /* L30: */
                }
            }
            jy += *incy;  // 调整 jy 的值，以正确访问向量 y 的下一个元素
            /* L40: */
        }
    }

    return 0;  // 返回 0，表示函数执行成功

/*     End of DGER  . */

} /* dger_ */

doublereal dnrm2_(integer *n, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal ret_val, d__1;

    /* Local variables */
    static integer ix;
    static doublereal ssq, norm, scale, absxi;


    /*
        Purpose
        =======

        DNRM2 returns the euclidean norm of a vector via the function
        name, so that

           DNRM2 := sqrt( x'*x )

        Further Details
        ===============

        -- This version written on 25-October-1982.
           Modified on 14-October-1993 to inline the call to DLASSQ.
           Sven Hammarling, Nag Ltd.

        =====================================================================
    */

    /* Parameter adjustments */
    --x;  // 将指针 x 指向的位置向前移动一位，使其指向正确的数组起始位置

    /* Function Body */
    if (*n < 1 || *incx < 1) {  // 检查 n 和 incx 是否合法
        norm = 0.;  // 如果不合法，则将 norm 设为 0
    } else if (*n == 1) {  // 如果 n 等于 1
        norm = abs(x[1]);  // 直接计算向量的范数
    } else {  // 如果 n 大于 1
        scale = 0.;  // 初始化 scale 为 0
        ssq = 1.;  // 初始化 ssq 为 1，用于累加平方和

        /*
              The following loop is equivalent to this call to the LAPACK
              auxiliary routine:
              CALL DLASSQ( N, X, INCX, SCALE, SSQ )
        */

        i__1 = (*n - 1) * *incx + 1;  // 计算
/* Computing 2nd power */
d__1 = absxi / scale;
ssq += d__1 * d__1;

计算变量 `absxi` 除以 `scale` 的平方，并将结果累加到 `ssq` 中。


}
}
/* L10: */

结束两层循环的代码块，并标记为 `L10`。


norm = scale * sqrt(ssq);
}

计算向量的二范数，即将 `scale` 乘以 `ssq` 的平方根，并将结果存储在 `norm` 中。


ret_val = norm;
return ret_val;

将计算得到的向量二范数赋值给 `ret_val`，然后返回 `ret_val`。


/*     End of DNRM2. */

标记 `DNRM2` 函数的结束。


} /* dnrm2_ */

结束 `dnrm2_` 函数的定义。


/* Subroutine */ int drot_(integer *n, doublereal *dx, integer *incx,
    doublereal *dy, integer *incy, doublereal *c__, doublereal *s)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, ix, iy;
    static doublereal dtemp;

开始 `drot_` 子程序的定义，其中包括参数声明和局部变量的定义。


/*
    Purpose
    =======

       DROT applies a plane rotation.
*/

说明 `DROT` 函数的目的是应用一个平面旋转。


Further Details
===============

提供更多细节说明。


jack dongarra, linpack, 3/11/78.
modified 12/3/93, array(1) declarations changed to array(*)

作者信息及修改历史。


=====================================================================
*/

分隔函数说明和实际代码的结束。


/* Parameter adjustments */
--dy;
--dx;

调整参数，使它们从 1 开始索引改为从 0 开始索引。


/* Function Body */
if (*n <= 0) {
return 0;
}
if (*incx == 1 && *incy == 1) {
goto L20;
}

检查参数 `n` 是否小于等于 0，如果是则直接返回；检查 `incx` 和 `incy` 是否为 1，如果是则跳转到 `L20` 标签处执行代码。


/*
     code for unequal increments or equal increments not equal
       to 1
*/

处理 `incx` 和 `incy` 不同时为 1 的情况。


i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
dtemp = *c__ * dx[ix] + *s * dy[iy];
dy[iy] = *c__ * dy[iy] - *s * dx[ix];
dx[ix] = dtemp;
ix += *incx;
iy += *incy;
/* L10: */
}

执行平面旋转操作，根据给定的旋转角度 `c__` 和 `s`，对向量 `dx` 和 `dy` 进行旋转。


return 0;

函数结束，返回 0。


/*       code for both increments equal to 1 */
L20:
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
dtemp = *c__ * dx[i__] + *s * dy[i__];
dy[i__] = *c__ * dy[i__] - *s * dx[i__];
dx[i__] = dtemp;
/* L30: */
}
return 0;

处理 `incx` 和 `incy` 均为 1 的情况，执行相同的平面旋转操作。


} /* drot_ */

结束 `drot_` 函数的定义。


/* Subroutine */ int dscal_(integer *n, doublereal *da, doublereal *dx,
    integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__, m, mp1, nincx;

开始 `dscal_` 子程序的定义，包括参数声明和局部变量的定义。


/*
    Purpose
    =======

       DSCAL scales a vector by a constant.
       uses unrolled loops for increment equal to one.
*/

说明 `DSCAL` 函数的目的是通过一个常数来对向量进行缩放，对于 `incx` 等于 1 的情况使用展开循环。


Further Details
===============

提供更多细节说明。


jack dongarra, linpack, 3/11/78.
modified 3/93 to return if incx .le. 0.
modified 12/3/93, array(1) declarations changed to array(*)

作者信息及修改历史。


=====================================================================
*/

分隔函数说明和实际代码的结束。


/* Parameter adjustments */
--dx;

调整参数，使 `dx` 从 1 开始索引改为从 0 开始索引。


/* Function Body */
if (*n <= 0 || *incx <= 0) {
return 0;
}
if (*incx == 1) {
goto L20;
}

检查参数 `n` 是否小于等于 0 或 `incx` 是否小于等于 0，如果是则直接返回；检查 `incx` 是否为 1，如果是则跳转到 `L20` 标签处执行代码。


/*
      code for increment not equal to 1
*/

处理 `incx` 不为 1 的情况。


i__1 = nincx;
i__2 = *incx;
for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
dx[i__] = *da * dx[i__];
/* L10: */
}

对向量 `dx` 进行缩放操作，步长为 `incx`。


/*
      code for increment equal to 1

      clean-up loop
*/
L20:
m = *n % 5;
if (m == 0) {
goto L40;
}

处理 `incx` 等于 1 的情况，特别处理尾部不足 5 个元素的部分。


i__2 = m;
for (i__ = 1; i__ <= i__2; ++i__) {
dx[i__] = *da * dx[i__];
/* L30: */
}

对尾部不足 5 个元素的部分进行缩放操作。


if (*n < 5) {

如果向量长度小于 5，则直接返回。
    # 返回值为 0，函数结束
    return 0;
    }
L40:
    mp1 = m + 1;  
    // 计算 m+1 的值，m 是 *n 取余 3 的结果，用于后续循环的起始点
    i__2 = *n;    
    // 将 *n 的值赋给 i__2，用于循环的终止条件
    for (i__ = mp1; i__ <= i__2; i__ += 5) {  
        // 以步长 5 遍历 i__ 从 mp1 到 i__2
        dx[i__] = *da * dx[i__];  
        // 将 *da 乘以 dx[i__] 的值，赋给 dx[i__]
        dx[i__ + 1] = *da * dx[i__ + 1];  
        // 将 *da 乘以 dx[i__ + 1] 的值，赋给 dx[i__ + 1]
        dx[i__ + 2] = *da * dx[i__ + 2];  
        // 将 *da 乘以 dx[i__ + 2] 的值，赋给 dx[i__ + 2]
        dx[i__ + 3] = *da * dx[i__ + 3];  
        // 将 *da 乘以 dx[i__ + 3] 的值，赋给 dx[i__ + 3]
        dx[i__ + 4] = *da * dx[i__ + 4];  
        // 将 *da 乘以 dx[i__ + 4] 的值，赋给 dx[i__ + 4]
/* L50: */
    }
    // 返回 0 表示函数执行完毕
    return 0;
} /* dscal_ */

/* Subroutine */ int dswap_(integer *n, doublereal *dx, integer *incx,
    doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;
    static doublereal dtemp;


/*
    Purpose
    =======

       interchanges two vectors.
       uses unrolled loops for increments equal one.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
    return 0;
    }
    if (*incx == 1 && *incy == 1) {
    goto L20;
    }

/*
         code for unequal increments or equal increments not equal
           to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
    ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
    iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    dtemp = dx[ix];
    dx[ix] = dy[iy];
    dy[iy] = dtemp;
    ix += *incx;
    iy += *incy;
/* L10: */
    }
    // 返回 0 表示函数执行完毕
    return 0;

/*
         code for both increments equal to 1


         clean-up loop
*/

L20:
    m = *n % 3;    
    // 计算 *n 对 3 取余的结果，存入 m
    if (m == 0) {
    goto L40;    
    // 若 m 为 0，则跳转到标签 L40 处执行
    }
    i__1 = m;    
    // 将 m 的值赋给 i__1
    for (i__ = 1; i__ <= i__1; ++i__) {  
        // 遍历 i__ 从 1 到 i__1
        dtemp = dx[i__];    
        // 将 dx[i__] 的值赋给 dtemp
        dx[i__] = dy[i__];  
        // 将 dy[i__] 的值赋给 dx[i__]
        dy[i__] = dtemp;    
        // 将 dtemp 的值赋给 dy[i__]
/* L30: */
    }
    // 若 *n 小于 3，则返回 0
    if (*n < 3) {
    return 0;
    }
L40:
    mp1 = m + 1;    
    // 计算 m+1 的值，存入 mp1
    i__1 = *n;    
    // 将 *n 的值赋给 i__1
    for (i__ = mp1; i__ <= i__1; i__ += 3) {    
        // 以步长 3 遍历 i__ 从 mp1 到 i__1
        dtemp = dx[i__];    
        // 将 dx[i__] 的值赋给 dtemp
        dx[i__] = dy[i__];  
        // 将 dy[i__] 的值赋给 dx[i__]
        dy[i__] = dtemp;    
        // 将 dtemp 的值赋给 dy[i__]
        dtemp = dx[i__ + 1];  
        // 将 dx[i__ + 1] 的值赋给 dtemp
        dx[i__ + 1] = dy[i__ + 1];  
        // 将 dy[i__ + 1] 的值赋给 dx[i__ + 1]
        dy[i__ + 1] = dtemp;  
        // 将 dtemp 的值赋给 dy[i__ + 1]
        dtemp = dx[i__ + 2];  
        // 将 dx[i__ + 2] 的值赋给 dtemp
        dx[i__ + 2] = dy[i__ + 2];  
        // 将 dy[i__ + 2] 的值赋给 dx[i__ + 2]
        dy[i__ + 2] = dtemp;  
        // 将 dtemp 的值赋给 dy[i__ + 2]
/* L50: */
    }
    // 返回 0 表示函数执行完毕
    return 0;
} /* dswap_ */

/* Subroutine */ int dsymv_(char *uplo, integer *n, doublereal *alpha,
    doublereal *a, integer *lda, doublereal *x, integer *incx, doublereal
    *beta, doublereal *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static doublereal temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DSYMV  performs the matrix-vector  operation

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric matrix.

    Arguments
    ==========
    # 根据 UPLO 参数指定要引用的矩阵 A 的上三角部分还是下三角部分
    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    # 矩阵 A 的阶数 N
    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    # 标量 alpha
    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    # 双精度数组 A，维度为 (LDA, n)
    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of A is not referenced.
             Unchanged on exit.

    # 矩阵 A 的第一维度大小
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.

    # 双精度数组 X，至少维度为 (1 + (n - 1)*abs(INCX))
    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    # X 中元素的增量
    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    # 标量 beta
    BETA   - DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    # 双精度数组 Y，至少维度为 (1 + (n - 1)*abs(INCY))
    Y      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y. On exit, Y is overwritten by the updated
             vector y.

    # Y 中元素的增量
    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    # 更多细节
    # Level 2 Blas routine.
    # 1986年10月22日编写
    # Jack Dongarra, Argonne National Lab.
    # Jeremy Du Croz, Nag Central Office.
    # Sven Hammarling, Nag Central Office.
    # Richard Hanson, Sandia National Labs.
    Further Details
    ===============

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.
    =====================================================================
    # 进行输入参数的测试
       Test the input parameters.
    /* Parameter adjustments */
    a_dim1 = *lda;
    // 计算一维数组 A 在二维数组表示中的第一维度的偏移量
    a_offset = 1 + a_dim1;
    // 根据偏移量和第一维度大小调整指针指向数组 A
    a -= a_offset;
    // 调整指针指向数组 X 和 Y，从而正确访问它们
    --x;
    --y;

    /* Function Body */
    // 初始化 info 为 0，用于记录函数执行过程中的错误信息
    info = 0;
    // 检查 uplo 参数是否为 "U" 或 "L"，若不是则设置错误码并返回
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (*n < 0) { // 检查 n 是否小于 0，若是则设置错误码
        info = 2;
    } else if (*lda < max(1,*n)) { // 检查 lda 是否小于 max(1,n)，若是则设置错误码
        info = 5;
    } else if (*incx == 0) { // 检查 incx 是否为 0，若是则设置错误码
        info = 7;
    } else if (*incy == 0) { // 检查 incy 是否为 0，若是则设置错误码
        info = 10;
    }
    // 若 info 不为 0，则调用错误处理函数并返回
    if (info != 0) {
        xerbla_("DSYMV ", &info);
        return 0;
    }

/*     Quick return if possible. */

    // 若 n 为 0 或者 alpha 为 0 且 beta 为 1，则直接返回
    if (*n == 0 || *alpha == 0. && *beta == 1.) {
        return 0;
    }

/*     Set up the start points in  X  and  Y. */

    // 根据 incx 的正负设置起始点 kx
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (*n - 1) * *incx;
    }
    // 根据 incy 的正负设置起始点 ky
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (*n - 1) * *incy;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.

       First form  y := beta*y.
*/

    // 若 beta 不为 1，则根据 incy 的值更新向量 y
    if (*beta != 1.) {
        if (*incy == 1) {
            // 当 incy 为 1 时，直接更新 y 向量中的元素
            if (*beta == 0.) {
                // 当 beta 为 0 时，将 y 向量置零
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = 0.;
                }
            } else {
                // 否则，按照 beta 的值更新 y 向量中的元素
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = *beta * y[i__];
                }
            }
        } else {
            // 当 incy 不为 1 时，按照 incy 的步长更新 y 向量中的元素
            iy = ky;
            if (*beta == 0.) {
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = 0.;
                    iy += *incy;
                }
            } else {
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = *beta * y[iy];
                    iy += *incy;
                }
            }
        }
    }

    // 若 alpha 为 0，则直接返回
    if (*alpha == 0.) {
        return 0;
    }

    // 当 A 存储在上三角时的计算过程
    if (lsame_(uplo, "U")) {

/*        Form  y  when A is stored in upper triangle. */

        if (*incx == 1 && *incy == 1) {
            // 特定步长下，直接计算 y 向量的更新
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp1 = *alpha * x[j];
                temp2 = 0.;
                i__2 = j - 1;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    y[i__] += temp1 * a[i__ + j * a_dim1];
                    temp2 += a[i__ + j * a_dim1] * x[i__];
                }
                y[j] = y[j] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
            }
        } else {
            // 一般步长下，按步长更新 y 向量的元素
            jx = kx;
            jy = ky;
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp1 = *alpha * x[jx];
                temp2 = 0.;
                ix = kx;
                iy = ky;
                i__2 = j - 1;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    y[iy] += temp1 * a[i__ + j * a_dim1];
                    temp2 += a[i__ + j * a_dim1] * x[ix];
                    ix += *incx;
                    iy += *incy;
                }
                y[jy] = y[jy] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
                jx += *incx;
                jy += *incy;
            }
        }
    } else {

/*        Form  y  when A is stored in lower triangle. */
    // 如果增量 incx 和 incy 都为 1，则进入循环
    if (*incx == 1 && *incy == 1) {
        // 循环遍历 j 从 1 到 n
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 计算 temp1，即 alpha 乘以 x[j] 的结果
            temp1 = *alpha * x[j];
            // 初始化 temp2 为 0
            temp2 = 0.;
            // 更新向量 y[j] 的值，加上 temp1 乘以 a[j + j * a_dim1]
            y[j] += temp1 * a[j + j * a_dim1];
            // 循环遍历 i 从 j+1 到 n
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                // 更新向量 y[i__] 的值，加上 temp1 乘以 a[i__ + j * a_dim1]
                y[i__] += temp1 * a[i__ + j * a_dim1];
                // 更新 temp2，加上 a[i__ + j * a_dim1] 乘以 x[i__]
                temp2 += a[i__ + j * a_dim1] * x[i__];
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static doublereal temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

    /* 主程序开始 */

/*
    Purpose
    =======

    DSYR2  performs the symmetric rank 2 operation

       A := alpha*x*y' + alpha*y*x' + A,

    where alpha is a scalar, x and y are n element vectors and A is an n
    by n symmetric matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    Y      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y.
             Unchanged on exit.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.
*/

    /* 行为分支 */

    if (*n <= 0) {
        return 0;
    }
    if (*alpha == 0.) {
        return 0;
    }

    /* Set up indices */

    kx = 1;
    ky = 1;

    if (*incx != 1 || *incy != 1) {
        if (*incx < 0) {
            kx = (1 - *n) * *incx + 1;
        }
        if (*incy < 0) {
            ky = (1 - *n) * *incy + 1;
        }
    }

    /* Perform the symmetric rank 2 operation */

    if (lsame_(uplo, "U")) {
        jx = kx;
        jy = ky;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            temp1 = *alpha * x[jx];
            temp2 = 0.;
            y[jy] += temp1 * a[j + j * a_dim1];
            ix = jx;
            iy = jy;
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                ix += *incx;
                iy += *incy;
                y[iy] += temp1 * a[i__ + j * a_dim1];
                temp2 += a[i__ + j * a_dim1] * x[ix];
            }
            y[jy] += *alpha * temp2;
            jx += *incx;
            jy += *incy;
        }
    } else {
        jx = kx;
        jy = ky;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            temp1 = *alpha * x[jx];
            temp2 = 0.;
            y[jy] += temp1 * a[j + j * a_dim1];
            ix = jx;
            iy = jy;
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                ix += *incx;
                iy += *incy;
                y[iy] += temp1 * a[i__ + j * a_dim1];
                temp2 += a[i__ + j * a_dim1] * x[ix];
            }
            y[jy] += *alpha * temp2;
            jx += *incx;
            jy += *incy;
        }
    }

    /* End of DSYR2 */

    return 0;

} /* dsyr2_ */



/* Subroutine */ int dsyr2_(char *uplo, integer *n, doublereal *alpha,
    doublereal *x, integer *incx, doublereal *y, integer *incy,
    doublereal *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static doublereal temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

    /* 行为分支 */
    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of A is not referenced. On exit, the
             upper triangular part of the array A is overwritten by the
             upper triangular part of the updated matrix.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of A is not referenced. On exit, the
             lower triangular part of the array A is overwritten by the
             lower triangular part of the updated matrix.


# A 是一个双精度数组，维度为 (LDA, n)。
# 在输入时，如果 UPLO = 'U' 或 'u'，数组 A 的前 n 行 n 列必须包含对称矩阵的上三角部分，且 A 的严格下三角部分不被引用。退出时，数组 A 的上三角部分被更新后的矩阵的上三角部分覆盖。
# 在输入时，如果 UPLO = 'L' 或 'l'，数组 A 的前 n 行 n 列必须包含对称矩阵的下三角部分，且 A 的严格上三角部分不被引用。退出时，数组 A 的下三角部分被更新后的矩阵的下三角部分覆盖。

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.


# LDA 是一个整数。
# 在输入时，LDA 指定了数组 A 在调用程序中声明的第一个维度。LDA 必须至少为 max(1, n)。
# 退出时，LDA 的值保持不变。

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================


# 进一步细节
# ===============

# Level 2 Blas 子程序。

# -- 1986年10月22日编写。
#    Jack Dongarra，阿贡国家实验室。
#    Jeremy Du Croz，NAG 中央办事处。
#    Sven Hammarling，NAG 中央办事处。
#    Richard Hanson，Sandia 国家实验室。

# =====================================================================

       Test the input parameters.


# 测试输入参数。
    /* 调整参数指针，使其指向正确位置 */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* 函数主体 */
    info = 0;
    /* 检查上三角或下三角参数是否正确 */
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (*n < 0) {
        info = 2;
    } else if (*incx == 0) {
        info = 5;
    } else if (*incy == 0) {
        info = 7;
    } else if (*lda < max(1,*n)) {
        info = 9;
    }
    /* 如果有错误信息，则调用错误处理程序并返回 */
    if (info != 0) {
        xerbla_("DSYR2 ", &info);
        return 0;
    }

    /* 如果 n 为零或 alpha 为零，则直接返回 */
    if (*n == 0 || *alpha == 0.) {
        return 0;
    }

    /* 如果增量不是1，则计算起始点 */
    if (*incx != 1 || *incy != 1) {
        if (*incx > 0) {
            kx = 1;
        } else {
            kx = 1 - (*n - 1) * *incx;
        }
        if (*incy > 0) {
            ky = 1;
        } else {
            ky = 1 - (*n - 1) * *incy;
        }
        jx = kx;
        jy = ky;
    }

    /* 开始操作。依次访问 A 的元素，通过一次遍历来处理 A 的三角部分 */
    if (lsame_(uplo, "U")) {

        /* 当 A 存储在上三角时，形成 A */
        if (*incx == 1 && *incy == 1) {
            /* 当增量为1时的情况 */
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[j] != 0. || y[j] != 0.) {
                    temp1 = *alpha * y[j];
                    temp2 = *alpha * x[j];
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] * temp1 + y[i__] * temp2;
                    }
                }
            }
        } else {
            /* 一般情况下的处理 */
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0. || y[jy] != 0.) {
                    temp1 = *alpha * y[jy];
                    temp2 = *alpha * x[jx];
                    ix = kx;
                    iy = ky;
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] * temp1 + y[iy] * temp2;
                        ix += *incx;
                        iy += *incy;
                    }
                }
                jx += *incx;
                jy += *incy;
            }
        }
    } else {

        /* 当 A 存储在下三角时，形成 A */
        if (*incx == 1 && *incy == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[j] != 0. || y[j] != 0.) {
                    temp1 = *alpha * y[j];
                    temp2 = *alpha * x[j];
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] * temp1 + y[i__] * temp2;
                    }
                }
            }
        }
    }
    } else {
        // 循环遍历矩阵的列
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 检查向量 x 和 y 的当前索引处是否为零
            if (x[jx] != 0. || y[jy] != 0.) {
                // 计算临时变量 temp1 和 temp2
                temp1 = *alpha * y[jy];
                temp2 = *alpha * x[jx];
                // 设置向量 x 和 y 的起始索引
                ix = jx;
                iy = jy;
                // 循环遍历矩阵的行
                i__2 = *n;
                for (i__ = j; i__ <= i__2; ++i__) {
                    // 更新矩阵 A 中的元素
                    a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] * temp1 + y[iy] * temp2;
                    // 更新向量 x 和 y 的索引
                    ix += *incx;
                    iy += *incy;
/* L70: */
            }
        }
        jx += *incx;
        jy += *incy;
/* L80: */
        }
    }
    }

    return 0;

/*     End of DSYR2 . */

} /* dsyr2_ */

/* Subroutine */ int dsyr2k_(char *uplo, char *trans, integer *n, integer *k,
    doublereal *alpha, doublereal *a, integer *lda, doublereal *b,
    integer *ldb, doublereal *beta, doublereal *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3;

    /* Local variables */
    static integer i__, j, l, info;
    static doublereal temp1, temp2;
    static integer nrowa;
    static logical upper;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*
    Purpose
    =======

    DSYR2K  performs one of the symmetric rank 2k operations

       C := alpha*A*B' + alpha*B*A' + beta*C,

    or

       C := alpha*A'*B + alpha*B'*A + beta*C,

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
    and  A and B  are  n by k  matrices  in the  first  case  and  k by n
    matrices in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of  the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   C := alpha*A*B' + alpha*B*A' +
                                          beta*C.

                TRANS = 'T' or 't'   C := alpha*A'*B + alpha*B'*A +
                                          beta*C.

                TRANS = 'C' or 'c'   C := alpha*A'*B + alpha*B'*A +
                                          beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns  of  the  matrices  A and B,  and on  entry  with
             TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
             of rows of the matrices  A and B.  K must be at least  zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.
*/
    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.
    # A 是一个双精度数组，维度为 (LDA, ka)，其中 ka 在 TRANS = 'N' 或 'n' 时为 k，否则为 n。
    # 在 TRANS = 'N' 或 'n' 时，数组 A 的前 n 行 k 列必须包含矩阵 A 的数据；否则前 k 行 n 列必须包含矩阵 A 的数据。
    # 函数执行完成后，数组 A 的内容保持不变。

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.
    # 输入参数 LDA 是整数，指定数组 A 的第一个维度，其声明方式需符合调用程序或子程序的定义。
    # 当 TRANS = 'N' 或 'n' 时，LDA 必须至少为 max(1, n)，否则必须至少为 max(1, k)。
    # 函数执行完成后，LDA 的值保持不变。

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  k by n  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.
    # B 是一个双精度数组，维度为 (LDB, kb)，其中 kb 在 TRANS = 'N' 或 'n' 时为 k，否则为 n。
    # 在 TRANS = 'N' 或 'n' 时，数组 B 的前 n 行 k 列必须包含矩阵 B 的数据；否则前 k 行 n 列必须包含矩阵 B 的数据。
    # 函数执行完成后，数组 B 的内容保持不变。

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDB must be at least  max( 1, n ), otherwise  LDB must
             be at least  max( 1, k ).
             Unchanged on exit.
    # 输入参数 LDB 是整数，指定数组 B 的第一个维度，其声明方式需符合调用程序或子程序的定义。
    # 当 TRANS = 'N' 或 'n' 时，LDB 必须至少为 max(1, n)，否则必须至少为 max(1, k)。
    # 函数执行完成后，LDB 的值保持不变。

    BETA   - DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.
    # 输入参数 BETA 是双精度实数，指定标量 beta 的值。
    # 函数执行完成后，BETA 的值保持不变。

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  symmetric matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  symmetric matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.
    # C 是一个双精度数组，维度为 (LDC, n)。
    # 在 UPLO = 'U' 或 'u' 时，在进入函数之前，数组 C 的前 n 行 n 列必须包含对称矩阵的上三角部分，且不引用严格的下三角部分。
    # 函数执行完成后，数组 C 的上三角部分被更新后的矩阵的上三角部分覆盖。
    # 在 UPLO = 'L' 或 'l' 时，在进入函数之前，数组 C 的前 n 行 n 列必须包含对称矩阵的下三角部分，且不引用严格的上三角部分。
    # 函数执行完成后，数组 C 的下三角部分被更新后的矩阵的下三角部分覆盖。

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, n ).
             Unchanged on exit.
    # 输入参数 LDC 是整数，指定数组 C 的第一个维度，其声明方式需符合调用程序或子程序的定义。
    # LDC 必须至少为 max(1, n)。
    # 函数执行完成后，LDC 的值保持不变.

    Further Details
    ===============

    Level 3 Blas routine.
    # 详细信息：Level 3 Blas 例程。
    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.
    /* Parameter adjustments */
    // 计算数组 `a` 的第一维大小
    a_dim1 = *lda;
    // 计算数组 `a` 的起始偏移量
    a_offset = 1 + a_dim1;
    // 对 `a` 进行偏移，使其指向正确的位置
    a -= a_offset;
    // 计算数组 `b` 的第一维大小
    b_dim1 = *ldb;
    // 计算数组 `b` 的起始偏移量
    b_offset = 1 + b_dim1;
    // 对 `b` 进行偏移，使其指向正确的位置
    b -= b_offset;
    // 计算数组 `c__` 的第一维大小
    c_dim1 = *ldc;
    // 计算数组 `c__` 的起始偏移量
    c_offset = 1 + c_dim1;
    // 对 `c__` 进行偏移，使其指向正确的位置
    c__ -= c_offset;

    /* Function Body */
    // 检查参数 `uplo` 是否为 "U"（上三角）或 "L"（下三角）
    if (lsame_(trans, "N")) {
        // 如果 `trans` 是 "N"，则 `a` 的行数为 `n`
        nrowa = *n;
    } else {
        // 否则 `a` 的行数为 `k`
        nrowa = *k;
    }
    // 判断是否为上三角阵
    upper = lsame_(uplo, "U");

    // 初始化错误信息为 0
    info = 0;
    // 检查是否为上三角阵或下三角阵
    if (! upper && ! lsame_(uplo, "L")) {
        // 如果 `uplo` 不是 "U" 且不是 "L"，则错误信息为 1
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
        // 如果 `trans` 不是 "N"、"T"、"C" 中的一个，错误信息为 2
        info = 2;
    } else if (*n < 0) {
        // 如果 `n` 小于 0，错误信息为 3
        info = 3;
    } else if (*k < 0) {
        // 如果 `k` 小于 0，错误信息为 4
        info = 4;
    } else if (*lda < max(1,nrowa)) {
        // 如果 `lda` 小于 `max(1,nrowa)`，错误信息为 7
        info = 7;
    } else if (*ldb < max(1,nrowa)) {
        // 如果 `ldb` 小于 `max(1,nrowa)`，错误信息为 9
        info = 9;
    } else if (*ldc < max(1,*n)) {
        // 如果 `ldc` 小于 `max(1,*n)`，错误信息为 12
        info = 12;
    }
    // 如果存在错误信息，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("DSYR2K", &info);
        return 0;
    }

    /* Quick return if possible. */
    // 如果 `n` 等于 0 或者 `alpha` 为 0 且 `k` 为 0 并且 `beta` 为 1，则快速返回
    if (*n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
        return 0;
    }

    /* And when alpha.eq.zero. */
    // 当 `alpha` 等于 0 时的处理逻辑
    if (*alpha == 0.) {
        // 如果是上三角阵
        if (upper) {
            // 如果 `beta` 为 0
            if (*beta == 0.) {
                // 将 `c__` 矩阵的上三角部分设置为 0
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                }
            } else {
                // 否则，将 `c__` 矩阵的上三角部分乘以 `beta`
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        } else {
            // 如果是下三角阵
            if (*beta == 0.) {
                // 将 `c__` 矩阵的下三角部分设置为 0
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                }
            } else {
                // 否则，将 `c__` 矩阵的下三角部分乘以 `beta`
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        }
        // 返回
        return 0;
    }

    /* Start the operations. */
    // 开始进行矩阵运算
    if (lsame_(trans, "N")) {

        /* Form  C := alpha*A*B' + alpha*B*A' + C. */

        // 如果是上三角阵
        if (upper) {
            // 计算 C := alpha*A*B' + alpha*B*A' + C
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                // 如果 `beta` 为 0
                if (*beta == 0.) {
                    // 将 `c__` 矩阵的第 j 列上三角部分设置为 0
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                } else if (*beta != 1.) {
                    // 否则，将 `c__` 矩阵的第 j 列上三角部分乘以 `beta`
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        } else {
            // 如果是下三角阵
            // 略
        }
    }
/* L100: */
            }
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            if (a[j + l * a_dim1] != 0. || b[j + l * b_dim1] != 0.) {
                temp1 = *alpha * b[j + l * b_dim1];
                temp2 = *alpha * a[j + l * a_dim1];
                i__3 = j;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    // 计算更新 C 的元素，累加 A 和 B 的乘积
                    c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
                        i__ + l * a_dim1] * temp1 + b[i__ + l *
                        b_dim1] * temp2;
/* L110: */
                }
            }
/* L120: */
        }
/* L130: */
        }
    } else {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            if (*beta == 0.) {
                // 如果 beta 等于 0，则将 C 的元素设置为 0
                i__2 = *n;
                for (i__ = j; i__ <= i__2; ++i__) {
                    c__[i__ + j * c_dim1] = 0.;
/* L140: */
                }
            } else if (*beta != 1.) {
                // 如果 beta 不等于 1，则乘以 beta
                i__2 = *n;
                for (i__ = j; i__ <= i__2; ++i__) {
                    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L150: */
                }
            }
            i__2 = *k;
            for (l = 1; l <= i__2; ++l) {
                if (a[j + l * a_dim1] != 0. || b[j + l * b_dim1] != 0.) {
                    temp1 = *alpha * b[j + l * b_dim1];
                    temp2 = *alpha * a[j + l * a_dim1];
                    i__3 = *n;
                    for (i__ = j; i__ <= i__3; ++i__) {
                        // 计算更新 C 的元素，累加 A 和 B 的乘积
                        c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
                            i__ + l * a_dim1] * temp1 + b[i__ + l *
                            b_dim1] * temp2;
/* L160: */
                    }
                }
/* L170: */
            }
/* L180: */
        }
    }
    } else {

/*        Form  C := alpha*A'*B + alpha*B'*A + C. */

    if (upper) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = j;
            for (i__ = 1; i__ <= i__2; ++i__) {
                temp1 = 0.;
                temp2 = 0.;
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    // 计算 A' * B 和 B' * A 的累加和
                    temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
                    temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
/* L190: */
                }
                if (*beta == 0.) {
                    // 如果 beta 等于 0，则直接计算 alpha*A'*B + alpha*B'*A
                    c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
                        temp2;
                } else {
                    // 否则，计算 beta * C + alpha*A'*B + alpha*B'*A
                    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
                        + *alpha * temp1 + *alpha * temp2;
                }
/* L200: */
            }
/* L210: */
        }
    } else {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
                temp1 = 0.;
                temp2 = 0.;
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    // 计算 A' * B 和 B' * A 的累加和
                    temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
                    temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
/* L220: */
                }
                if (*beta == 0.) {
                    // 如果 beta 等于 0，则直接计算 alpha*A'*B + alpha*B'*A
                    c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
                        temp2;
                } else {
                    // 否则，计算 beta * C + alpha*A'*B + alpha*B'*A
                    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
                        + *alpha * temp1 + *alpha * temp2;
                }
/* L230: */
            }
/* L240: */
        }
    }
/* L220: */
            }
            // 检查 beta 是否为 0
            if (*beta == 0.) {
                // 如果 beta 为 0，则计算 C = alpha * temp1 + alpha * temp2
                c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
                    temp2;
            } else {
                // 如果 beta 不为 0，则计算 C = beta * C + alpha * temp1 + alpha * temp2
                c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
                    + *alpha * temp1 + *alpha * temp2;
            }
/* L230: */
        }
/* L240: */
        }
    }
    }

    // 返回值为 0，表示执行成功
    return 0;

/*     End of DSYR2K. */

} /* dsyr2k_ */

/* Subroutine */ int dsyrk_(char *uplo, char *trans, integer *n, integer *k,
    doublereal *alpha, doublereal *a, integer *lda, doublereal *beta,
    doublereal *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, l, info;
    static doublereal temp;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);

    /*
     * Purpose
     * =======
     * 
     * DSYRK  performs one of the symmetric rank k operations
     * 
     *    C := alpha*A*A' + beta*C,
     * 
     * or
     * 
     *    C := alpha*A'*A + beta*C,
     * 
     * where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
     * and  A  is an  n by k  matrix in the first case and a  k by n  matrix
     * in the second case.
     * 
     * Arguments
     * ==========
     * 
     * UPLO   - CHARACTER*1.
     *          On  entry,   UPLO  specifies  whether  the  upper  or  lower
     *          triangular  part  of  the  array  C  is to be  referenced  as
     *          follows:
     * 
     *             UPLO = 'U' or 'u'   Only the  upper triangular part of  C
     *                                 is to be referenced.
     * 
     *             UPLO = 'L' or 'l'   Only the  lower triangular part of  C
     *                                 is to be referenced.
     * 
     *          Unchanged on exit.
     * 
     * TRANS  - CHARACTER*1.
     *          On entry,  TRANS  specifies the operation to be performed as
     *          follows:
     * 
     *             TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C.
     * 
     *             TRANS = 'T' or 't'   C := alpha*A'*A + beta*C.
     * 
     *             TRANS = 'C' or 'c'   C := alpha*A'*A + beta*C.
     * 
     *          Unchanged on exit.
     * 
     * N      - INTEGER.
     *          On entry,  N specifies the order of the matrix C.  N must be
     *          at least zero.
     *          Unchanged on exit.
     * 
     * K      - INTEGER.
     *          On entry with  TRANS = 'N' or 'n',  K  specifies  the number
     *          of  columns   of  the   matrix   A,   and  on   entry   with
     *          TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
     *          of rows of the matrix  A.  K must be at least zero.
     *          Unchanged on exit.
     * 
     * ALPHA  - DOUBLE PRECISION.
     *          On entry, ALPHA specifies the scalar alpha.
     *          Unchanged on exit.
     */
    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  symmetric matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  symmetric matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, n ).
             Unchanged on exit.

    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

    =====================================================================


       Test the input parameters.
    /* Parameter adjustments */
    // 计算数组 `a` 和 `c__` 的偏移量
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    // 根据 `trans` 的取值设置 `nrowa`
    if (lsame_(trans, "N")) {
        nrowa = *n;
    } else {
        nrowa = *k;
    }
    // 判断是否为上三角矩阵
    upper = lsame_(uplo, "U");

    info = 0;
    // 检查参数合法性
    if (! upper && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
        "T") && ! lsame_(trans, "C")) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*k < 0) {
        info = 4;
    } else if (*lda < max(1,nrowa)) {
        info = 7;
    } else if (*ldc < max(1,*n)) {
        info = 10;
    }
    // 如果参数有误，调用错误处理函数 `xerbla_`
    if (info != 0) {
        xerbla_("DSYRK ", &info);
        return 0;
    }

/*     Quick return if possible. */

    // 如果 `n` 为零或者 `alpha` 为零且 `k` 为零且 `beta` 为一，直接返回
    if (*n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
        return 0;
    }

/*     And when  alpha.eq.zero. */

    // 如果 `alpha` 为零
    if (*alpha == 0.) {
        // 如果是上三角矩阵
        if (upper) {
            // 如果 `beta` 为零，将 `c__` 清零
            if (*beta == 0.) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                }
            } else {
                // 否则，对 `c__` 进行缩放
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        } else {
            // 如果是下三角矩阵
            if (*beta == 0.) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                }
            } else {
                // 否则，对 `c__` 进行缩放
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        }
        return 0;
    }

/*     Start the operations. */

    // 当 `trans` 为 "N" 时
    if (lsame_(trans, "N")) {

/*        Form  C := alpha*A*A' + beta*C. */

        // 如果是上三角矩阵
        if (upper) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                // 如果 `beta` 为零，先清零 `c__` 的对应元素
                if (*beta == 0.) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.;
                    }
                // 如果 `beta` 不为一，对 `c__` 的对应元素进行缩放
                } else if (*beta != 1.) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
                // 计算 `alpha*A*A'` 的贡献
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    if (a[j + l * a_dim1] != 0.) {
                        temp = *alpha * a[j + l * a_dim1];
                        i__3 = j;
                        for (i__ = 1; i__ <= i__3; ++i__) {
                            c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
                        }
                    }
                }
            }
        }
    } else {
        // 如果条件不满足（即 *beta != 0），执行以下循环
        i__1 = *n;
        // 循环从 1 到 *n
        for (j = 1; j <= i__1; ++j) {
            // 如果 beta 等于 0，则执行以下循环
            if (*beta == 0.) {
                i__2 = *n;
                // 循环从 j 到 *n
                for (i__ = j; i__ <= i__2; ++i__) {
                    // 将 c__ 数组中的元素 c__[i__ + j * c_dim1] 设置为 0
                    c__[i__ + j * c_dim1] = 0.;
                }
            }
/* L140: */
            }
        } else if (*beta != 1.) {
            // 如果 beta 不等于 1，则执行以下操作
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
            // 对于每个 i 从 j 到 n
            c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L150: */
            }
        }
        // 对于每个 l 从 1 到 k
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            // 如果 a[j + l * a_dim1] 不为 0
            if (a[j + l * a_dim1] != 0.) {
            // 计算临时变量 temp
            temp = *alpha * a[j + l * a_dim1];
            // 对于每个 i 从 j 到 n
            i__3 = *n;
            for (i__ = j; i__ <= i__3; ++i__) {
                // 更新 c__[i__ + j * c_dim1] 的值
                c__[i__ + j * c_dim1] += temp * a[i__ + l *
                    a_dim1];
/* L160: */
            }
            }
/* L170: */
        }
/* L180: */
        }
    }
    } else {

/*        Form  C := alpha*A'*A + beta*C. */

    if (upper) {
        // 如果 upper 为真
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 对于每个 i 从 1 到 j
        i__2 = j;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 初始化临时变量 temp
            temp = 0.;
            // 对于每个 l 从 1 到 k
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 temp 的累加值
            temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L190: */
            }
            // 根据 beta 的值更新 c__[i__ + j * c_dim1]
            if (*beta == 0.) {
            c__[i__ + j * c_dim1] = *alpha * temp;
            } else {
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
            }
/* L200: */
        }
/* L210: */
        }
    } else {
        // 如果 upper 为假
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 对于每个 i 从 j 到 n
        i__2 = *n;
        for (i__ = j; i__ <= i__2; ++i__) {
            // 初始化临时变量 temp
            temp = 0.;
            // 对于每个 l 从 1 到 k
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 temp 的累加值
            temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L220: */
            }
            // 根据 beta 的值更新 c__[i__ + j * c_dim1]
            if (*beta == 0.) {
            c__[i__ + j * c_dim1] = *alpha * temp;
            } else {
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
            }
/* L230: */
        }
/* L240: */
        }
    }
    }

    // 返回操作成功
    return 0;

/*     End of DSYRK . */

} /* dsyrk_ */

/* Subroutine */ int dtrmm_(char *side, char *uplo, char *transa, char *diag,
    integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
    lda, doublereal *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, k, info;
    static doublereal temp;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    DTRMM  performs one of the matrix-matrix operations

       B := alpha*op( A )*B,   or   B := alpha*B*op( A ),

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'.

    Arguments
    ==========
    SIDE   - CHARACTER*1.
             在进入时，SIDE 指定了操作 A 的转置方式对 B 进行左乘或右乘：

                SIDE = 'L' or 'l'   B := alpha*op( A )*B.
                                      B 左乘 alpha 乘以 op( A )。

                SIDE = 'R' or 'r'   B := alpha*B*op( A ).
                                      B 右乘 alpha 乘以 op( A )。

             函数执行完毕后，SIDE 参数保持不变。

    UPLO   - CHARACTER*1.
             在进入时，UPLO 指定矩阵 A 是上三角还是下三角形式：

                UPLO = 'U' or 'u'   A 是一个上三角矩阵。

                UPLO = 'L' or 'l'   A 是一个下三角矩阵。

             函数执行完毕后，UPLO 参数保持不变。

    TRANSA - CHARACTER*1.
             在进入时，TRANSA 指定用于矩阵乘法中的 op( A ) 形式：

                TRANSA = 'N' or 'n'   op( A ) = A.

                TRANSA = 'T' or 't'   op( A ) = A'.

                TRANSA = 'C' or 'c'   op( A ) = A'.

             函数执行完毕后，TRANSA 参数保持不变。

    DIAG   - CHARACTER*1.
             在进入时，DIAG 指定矩阵 A 是否为单位三角形式：

                DIAG = 'U' or 'u'   假定 A 是单位上三角矩阵。

                DIAG = 'N' or 'n'   A 不假定为单位三角形式。

             函数执行完毕后，DIAG 参数保持不变。

    M      - INTEGER.
             在进入时，M 指定矩阵 B 的行数。M 必须至少为零。
             函数执行完毕后，M 参数保持不变。

    N      - INTEGER.
             在进入时，N 指定矩阵 B 的列数。N 必须至少为零。
             函数执行完毕后，N 参数保持不变。

    ALPHA  - DOUBLE PRECISION.
             在进入时，ALPHA 指定标量 alpha。当 alpha 为零时，A 不被引用，B 在进入前不需要被设置。
             函数执行完毕后，ALPHA 参数保持不变。

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, k ), where k is m
             当 SIDE = 'L' 或 'l' 时，k 是 m；当 SIDE = 'R' 或 'r' 时，k 是 n。
             进入前，如果 UPLO = 'U' 或 'u'，数组 A 的前 k 行 k 列必须包含上三角矩阵，且不引用 A 的严格下三角部分。
             进入前，如果 UPLO = 'L' 或 'l'，数组 A 的前 k 行 k 列必须包含下三角矩阵，且不引用 A 的严格上三角部分。
             注意当 DIAG = 'U' 或 'u' 时，不引用 A 的对角元素，但假定其为单位元素。
             函数执行完毕后，A 数组保持不变。
       ! LDA    - INTEGER.
       !          在进入时，LDA指定了数组A在调用子程序中声明的第一个维度。
       !          当 SIDE = 'L' 或 'l' 时，LDA 必须至少为 max(1, m)，
       !          当 SIDE = 'R' 或 'r' 时，LDA 必须至少为 max(1, n)。
       !          退出时保持不变。

       ! B      - DOUBLE PRECISION 数组，维度为 (LDB, n)。
       !          进入之前，数组 B 的前 m 行 n 列必须包含矩阵 B，
       !          退出时会被转换后的矩阵覆盖。

       ! LDB    - INTEGER.
       !          在进入时，LDB指定了数组B在调用子程序中声明的第一个维度。
       !          LDB 必须至少为 max(1, m)。
       !          退出时保持不变。

       ! Further Details
       ! ===============
       ! Level 3 Blas routine.
       ! Level 3 BLAS例程。

       ! -- Written on 8-February-1989.
       !    Jack Dongarra, Argonne National Laboratory.
       !    Iain Duff, AERE Harwell.
       !    Jeremy Du Croz, Numerical Algorithms Group Ltd.
       !    Sven Hammarling, Numerical Algorithms Group Ltd.
       ! -- 编写于1989年2月8日。
       !    Jack Dongarra，阿贡国家实验室。
       !    Iain Duff，AERE Harwell。
       !    Jeremy Du Croz，Numerical Algorithms Group Ltd.
       !    Sven Hammarling，Numerical Algorithms Group Ltd。

       ! =====================================================================
       ! 测试输入参数。
    /* Parameter adjustments */
    // 根据调用方传入的参数调整数组 a 和 b 的维度和偏移量
    a_dim1 = *lda;
    // 计算 a 数组的偏移量
    a_offset = 1 + a_dim1;
    // 调整 a 数组的指针位置
    a -= a_offset;
    // 计算 b 数组的偏移量
    b_dim1 = *ldb;
    // 计算 b 数组的偏移量
    b_offset = 1 + b_dim1;
    // 调整 b 数组的指针位置
    b -= b_offset;

    /* Function Body */
    // 判断 side 是否为 'L'，确定 nrowa 的值
    lside = lsame_(side, "L");
    if (lside) {
        // 如果 side 是 'L'，则 nrowa 等于 m
        nrowa = *m;
    } else {
        // 否则 nrowa 等于 n
        nrowa = *n;
    }
    // 判断 diag 是否为 'N'，确定 nounit 的值
    nounit = lsame_(diag, "N");
    // 判断 uplo 是否为 'U'，确定 upper 的值
    upper = lsame_(uplo, "U");

    // 初始化 info 为 0
    info = 0;
    // 检查 side 是否既不是 'L' 也不是 'R'
    if (! lside && ! lsame_(side, "R")) {
        info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
        // 检查 uplo 是否既不是 'U' 也不是 'L'
        info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa,
         "T") && ! lsame_(transa, "C")) {
        // 检查 transa 是否既不是 'N' 也不是 'T' 也不是 'C'
        info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
        "N")) {
        // 检查 diag 是否既不是 'U' 也不是 'N'
        info = 4;
    } else if (*m < 0) {
        // 检查 m 是否小于 0
        info = 5;
    } else if (*n < 0) {
        // 检查 n 是否小于 0
        info = 6;
    } else if (*lda < max(1,nrowa)) {
        // 检查 lda 是否小于 max(1, nrowa)
        info = 9;
    } else if (*ldb < max(1,*m)) {
        // 检查 ldb 是否小于 max(1, m)
        info = 11;
    }
    // 如果 info 不为 0，则调用错误处理函数并返回
    if (info != 0) {
        xerbla_("DTRMM ", &info);
        return 0;
    }

/*     Quick return if possible. */

    // 如果 m 或 n 为 0，则直接返回
    if (*m == 0 || *n == 0) {
        return 0;
    }

/*     And when alpha.eq.zero. */

    // 如果 alpha 等于 0，则将 b 数组清零
    if (*alpha == 0.) {
        // 循环遍历 b 数组的所有元素并赋值为 0
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + j * b_dim1] = 0.;
            }
        }
        return 0;
    }

/*     Start the operations. */

    // 开始进行矩阵乘法操作
    if (lside) {
        // 如果 side 是 'L'
        if (lsame_(transa, "N")) {

/*           Form  B := alpha*A*B. */

            // 如果 transa 是 'N'，执行 B := alpha*A*B 的计算
            if (upper) {
                // 如果 uplo 是 'U'，执行上三角部分的乘法
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *m;
                    for (k = 1; k <= i__2; ++k) {
                        if (b[k + j * b_dim1] != 0.) {
                            temp = *alpha * b[k + j * b_dim1];
                            i__3 = k - 1;
                            for (i__ = 1; i__ <= i__3; ++i__) {
                                b[i__ + j * b_dim1] += temp * a[i__ + k *
                                    a_dim1];
                            }
                            if (nounit) {
                                temp *= a[k + k * a_dim1];
                            }
                            b[k + j * b_dim1] = temp;
                        }
                    }
                }
            } else {
                // 如果 uplo 是 'L'，执行下三角部分的乘法
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    for (k = *m; k >= 1; --k) {
                        if (b[k + j * b_dim1] != 0.) {
                            temp = *alpha * b[k + j * b_dim1];
                            b[k + j * b_dim1] = temp;
                            if (nounit) {
                                b[k + j * b_dim1] *= a[k + k * a_dim1];
                            }
                            i__2 = *m;
                            for (i__ = k + 1; i__ <= i__2; ++i__) {
                                b[i__ + j * b_dim1] += temp * a[i__ + k *
                                    a_dim1];
                            }
                        }
                    }
                }
            }
        } else {
/*           Form  B := alpha*A'*B. */

if (upper) {
    // 循环遍历矩阵 B 的列
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // 循环遍历矩阵 B 的行（逆序）
        for (i__ = *m; i__ >= 1; --i__) {
            // 临时变量 temp 存储 B 的元素值
            temp = b[i__ + j * b_dim1];
            // 如果非单位矩阵，将 temp 乘以对应的 A 矩阵的对角元素
            if (nounit) {
                temp *= a[i__ + i__ * a_dim1];
            }
            // 循环遍历当前行上三角部分的非对角元素
            i__2 = i__ - 1;
            for (k = 1; k <= i__2; ++k) {
                // 更新 temp，累加上 A 的非对角元素乘以 B 的对应元素
                temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
                /* L90: */
            }
            // 更新 B 的元素为 alpha 倍的 temp
            b[i__ + j * b_dim1] = *alpha * temp;
            /* L100: */
        }
        /* L110: */
    }
} else {
    // 与上述类似，但此处循环遍历顺序矩阵 B 的列
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // 循环遍历顺序矩阵 B 的行
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 临时变量 temp 存储 B 的元素值
            temp = b[i__ + j * b_dim1];
            // 如果非单位矩阵，将 temp 乘以对应的 A 矩阵的对角元素
            if (nounit) {
                temp *= a[i__ + i__ * a_dim1];
            }
            // 循环遍历当前行下三角部分的非对角元素
            i__3 = *m;
            for (k = i__ + 1; k <= i__3; ++k) {
                // 更新 temp，累加上 A 的非对角元素乘以 B 的对应元素
                temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
                /* L120: */
            }
            // 更新 B 的元素为 alpha 倍的 temp
            b[i__ + j * b_dim1] = *alpha * temp;
            /* L130: */
        }
        /* L140: */
    }
}
} else {
if (lsame_(transa, "N")) {

/*           Form  B := alpha*B*A. */

if (upper) {
    // 循环遍历矩阵 B 的列（逆序）
    for (j = *n; j >= 1; --j) {
        // temp 初始值为 alpha
        temp = *alpha;
        // 如果非单位矩阵，将 temp 乘以对应的 A 矩阵的对角元素
        if (nounit) {
            temp *= a[j + j * a_dim1];
        }
        // 循环遍历矩阵 B 的行
        i__1 = *m;
        for (i__ = 1; i__ <= i__1; ++i__) {
            // 更新 B 的元素为 alpha 倍的 temp 乘以原 B 的对应元素
            b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
            /* L150: */
        }
        // 循环遍历上三角部分的非对角元素
        i__1 = j - 1;
        for (k = 1; k <= i__1; ++k) {
            // 如果 A 的非对角元素不为零，则更新 temp，累加上对应的 B 元素乘以 temp
            if (a[k + j * a_dim1] != 0.) {
                temp = *alpha * a[k + j * a_dim1];
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    b[i__ + j * b_dim1] += temp * b[i__ + k *
                        b_dim1];
                    /* L160: */
                }
            }
            /* L170: */
        }
        /* L180: */
    }
} else {
    // 与上述类似，但此处循环遍历顺序矩阵 B 的列
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // temp 初始值为 alpha
        temp = *alpha;
        // 如果非单位矩阵，将 temp 乘以对应的 A 矩阵的对角元素
        if (nounit) {
            temp *= a[j + j * a_dim1];
        }
        // 循环遍历矩阵 B 的行
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 更新 B 的元素为 alpha 倍的 temp 乘以原 B 的对应元素
            b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
            /* L190: */
        }
        // 循环遍历下三角部分的非对角元素
        i__2 = *n;
        for (k = j + 1; k <= i__2; ++k) {
            // 如果 A 的非对角元素不为零，则更新 temp，累加上对应的 B 元素乘以 temp
            if (a[k + j * a_dim1] != 0.) {
                temp = *alpha * a[k + j * a_dim1];
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    b[i__ + j * b_dim1] += temp * b[i__ + k *
                        b_dim1];
                    /* L200: */
                }
            }
            /* L210: */
        }
        /* L220: */
    }
}
} else {
/*           Form  B := alpha*B*A'. */

// 根据 upper 的值选择矩阵 A 的上三角形式或下三角形式进行计算
if (upper) {
    // 循环遍历 A 的列
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
        // 循环遍历 A 的行
        i__2 = k - 1;
        for (j = 1; j <= i__2; ++j) {
            // 如果 A[j + k * a_dim1] 不为零，则计算临时变量 temp
            if (a[j + k * a_dim1] != 0.) {
                temp = *alpha * a[j + k * a_dim1];
                // 循环遍历 B 的行
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    // 更新 B[i__ + j * b_dim1] 的值
                    b[i__ + j * b_dim1] += temp * b[i__ + k * b_dim1];
                    /* L230: */
                }
            }
            /* L240: */
        }
        // 计算临时变量 temp
        temp = *alpha;
        // 如果不是单位三角形矩阵，则更新 temp
        if (nounit) {
            temp *= a[k + k * a_dim1];
        }
        // 如果 temp 不等于 1，则更新 B[i__ + k * b_dim1] 的值
        if (temp != 1.) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
                /* L250: */
            }
        }
        /* L260: */
    }
} else {
    // 与上面相反的顺序遍历 A 的列
    for (k = *n; k >= 1; --k) {
        // 循环遍历 A 的行
        i__1 = *n;
        for (j = k + 1; j <= i__1; ++j) {
            // 如果 A[j + k * a_dim1] 不为零，则计算临时变量 temp
            if (a[j + k * a_dim1] != 0.) {
                temp = *alpha * a[j + k * a_dim1];
                // 循环遍历 B 的行
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 更新 B[i__ + j * b_dim1] 的值
                    b[i__ + j * b_dim1] += temp * b[i__ + k * b_dim1];
                    /* L270: */
                }
            }
            /* L280: */
        }
        // 计算临时变量 temp
        temp = *alpha;
        // 如果不是单位三角形矩阵，则更新 temp
        if (nounit) {
            temp *= a[k + k * a_dim1];
        }
        // 如果 temp 不等于 1，则更新 B[i__ + k * b_dim1] 的值
        if (temp != 1.) {
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
                /* L290: */
            }
        }
        /* L300: */
    }
}
}

return 0;

/*     End of DTRMM . */

} /* dtrmm_ */

/* Subroutine */ int dtrmv_(char *uplo, char *trans, char *diag, integer *n,
    doublereal *a, integer *lda, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, jx, kx, info;
    static doublereal temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    DTRMV  performs one of the matrix-vector operations

       x := A*x,   or   x := A'*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.
*/


注释中详细解释了每个代码行的作用，确保了代码逻辑和含义的清晰传达。
    ! TRANS - 字符*1.
    !         在进入时，TRANS 指定要执行的操作如下：
    !
    !            TRANS = 'N' 或 'n'   x := A*x.
    !
    !            TRANS = 'T' 或 't'   x := A'*x.
    !
    !            TRANS = 'C' 或 'c'   x := A'*x.
    !
    !         在退出时保持不变。

    ! DIAG - 字符*1.
    !        在进入时，DIAG 指定矩阵 A 是否为单位三角阵，具体如下：
    !
    !           DIAG = 'U' 或 'u'   A 假设为单位三角阵。
    !
    !           DIAG = 'N' 或 'n'   A 不假设为单位三角阵。
    !
    !        在退出时保持不变。

    ! N - 整数.
    !     在进入时，N 指定矩阵 A 的阶数。
    !     N 必须至少为零。
    !     在退出时保持不变。

    ! A - 双精度数组，维度为 (LDA, n).
    !     在进入时，如果 UPLO = 'U' 或 'u'，数组 A 的前 n 行 n 列必须包含上三角矩阵，
    !     且不引用 A 的严格下三角部分。
    !     如果 UPLO = 'L' 或 'l'，则数组 A 的前 n 行 n 列必须包含下三角矩阵，
    !     且不引用 A 的严格上三角部分。
    !     注意当 DIAG = 'U' 或 'u' 时，A 的对角元素也不被引用，但假定为单位。
    !     在退出时保持不变。

    ! LDA - 整数.
    !       在进入时，LDA 指定矩阵 A 在调用程序中声明的第一个维度。
    !       LDA 必须至少为 max(1, n)。
    !       在退出时保持不变。

    ! X - 双精度数组，至少维度为 (1 + (n - 1)*abs(INCX)).
    !     在进入时，增量数组 X 必须包含 n 元素向量 x。
    !     在退出时，X 被转换后的向量 x 覆盖。

    ! INCX - 整数.
    !        在进入时，INCX 指定 X 的元素的增量。INCX 必须不为零。
    !        在退出时保持不变。

    ! 进一步细节
    ! ==========
    !
    ! Level 2 Blas 例程。
    !
    ! -- 写于 1986年10月22日。
    !    Jack Dongarra, Argonne National Lab.
    !    Jeremy Du Croz, Nag Central Office.
    !    Sven Hammarling, Nag Central Office.
    !    Richard Hanson, Sandia National Labs.
    !
    ! =====================================================================
    !
    !
       ! 测试输入参数。
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
        info = 2;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*lda < max(1,*n)) {
        info = 6;
    } else if (*incx == 0) {
        info = 8;
    }
    if (info != 0) {
        xerbla_("DTRMV ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
        return 0;
    }

    nounit = lsame_(diag, "N");

/*
       Set up the start point in X if the increment is not unity. This
       will be  ( N - 1 )*INCX  too small for descending loops.
*/

    if (*incx <= 0) {
        kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
        kx = 1;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (lsame_(trans, "N")) {

/*        Form  x := A*x. */

        if (lsame_(uplo, "U")) {
            if (*incx == 1) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (x[j] != 0.) {
                        temp = x[j];
                        i__2 = j - 1;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            x[i__] += temp * a[i__ + j * a_dim1];
                            /* L10: */
                        }
                        if (nounit) {
                            x[j] *= a[j + j * a_dim1];
                        }
                    }
                    /* L20: */
                }
            } else {
                jx = kx;
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (x[jx] != 0.) {
                        temp = x[jx];
                        ix = kx;
                        i__2 = j - 1;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            x[ix] += temp * a[i__ + j * a_dim1];
                            ix += *incx;
                            /* L30: */
                        }
                        if (nounit) {
                            x[jx] *= a[j + j * a_dim1];
                        }
                    }
                    jx += *incx;
                    /* L40: */
                }
            }
        } else {
            if (*incx == 1) {
                for (j = *n; j >= 1; --j) {
                    if (x[j] != 0.) {
                        temp = x[j];
                        i__1 = j + 1;
                        for (i__ = *n; i__ >= i__1; --i__) {
                            x[i__] += temp * a[i__ + j * a_dim1];
                            /* L50: */
                        }
                        if (nounit) {
                            x[j] *= a[j + j * a_dim1];
                        }
                    }
                    /* L60: */
                }
            } else {
                kx += (*n - 1) * *incx;
                jx = kx;
                for (j = *n; j >= 1; --j) {
                    if (x[jx] != 0.) {
                        temp = x[jx];
                        ix = kx;
                        i__1 = j + 1;
                        for (i__ = *n; i__ >= i__1; --i__) {
                            x[ix] += temp * a[i__ + j * a_dim1];
                            ix -= *incx;
                            /* L70: */
                        }
                        if (nounit) {
                            x[jx] *= a[j + j * a_dim1];
                        }
                    }
                    jx -= *incx;
                    /* L80: */
                }
            }
        }
    } else {

/*        Form  x := A'*x or x := A'*x. */

        if (lsame_(uplo, "U")) {
            if (*incx == 1) {
                for (j = *n; j >= 1; --j) {
                    temp = x[j];
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    i__1 = j - 1;
                    for (i__ = 1; i__ <= i__1; ++i__) {
                        temp += a[i__ + j * a_dim1] * x[i__];
                        /* L90: */
                    }
                    x[j] = temp;
                    /* L100: */
                }
            } else {
                kx += (*n - 1) * *incx;
                jx = kx;
                for (j = *n; j >= 1; --j) {
                    temp = x[jx];
                    ix = kx;
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    i__1 = j - 1;
                    for (i__ = 1; i__ <= i__1; ++i__) {
                        temp += a[i__ + j * a_dim1] * x[ix];
                        ix -= *incx;
                        /* L110: */
                    }
                    x[jx] = temp;
                    jx -= *incx;
                    /* L120: */
                }
            }
        } else {
            if (*incx == 1) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    temp = x[j];
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    i__2 = *n;
                    for (i__ = j + 1; i__ <= i__2; ++i__) {
                        temp += a[i__ + j * a_dim1] * x[i__];
                        /* L130: */
                    }
                    x[j] = temp;
                    /* L140: */
                }
            } else {
                jx = kx;
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    temp = x[jx];
                    ix = jx;
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    i__2 = *n;
                    for (i__ = j + 1; i__ <= i__2; ++i__) {
                        ix += *incx;
                        temp += a[i__ + j * a_dim1] * x[ix];
                        /* L150: */
                    }
                    x[jx] = temp;
                    jx += *incx;
                    /* L160: */
                }
            }
        }
    }

/*     End of DTRMV . */

    return 0;
/* L80: */
        }
        }
    }
    } else {

/*        Form  x := A'*x. */

    // 检查是否为上三角矩阵，处理 x := A'*x 的情况
    if (lsame_(uplo, "U")) {
        // 处理增量为1的情况
        if (*incx == 1) {
        // 从最后一列开始向前遍历
        for (j = *n; j >= 1; --j) {
            temp = x[j];
            // 如果 A 不是单位矩阵，则考虑 A 的对角元素
            if (nounit) {
            temp *= a[j + j * a_dim1];
            }
            // 向上迭代处理每一列的非对角元素
            for (i__ = j - 1; i__ >= 1; --i__) {
            temp += a[i__ + j * a_dim1] * x[i__];
/* L90: */
            }
            // 更新 x 的值
            x[j] = temp;
/* L100: */
        }
        } else {
        // 处理增量不为1的情况
        jx = kx + (*n - 1) * *incx;
        // 从最后一列向前遍历
        for (j = *n; j >= 1; --j) {
            temp = x[jx];
            ix = jx;
            // 如果 A 不是单位矩阵，则考虑 A 的对角元素
            if (nounit) {
            temp *= a[j + j * a_dim1];
            }
            // 向上迭代处理每一列的非对角元素
            for (i__ = j - 1; i__ >= 1; --i__) {
            ix -= *incx;
            temp += a[i__ + j * a_dim1] * x[ix];
/* L110: */
            }
            // 更新 x 的值
            x[jx] = temp;
            jx -= *incx;
/* L120: */
        }
        }
    } else {
        // 处理下三角矩阵的情况
        if (*incx == 1) {
        // 从第一列开始向后遍历
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            temp = x[j];
            // 如果 A 不是单位矩阵，则考虑 A 的对角元素
            if (nounit) {
            temp *= a[j + j * a_dim1];
            }
            // 向下迭代处理每一列的非对角元素
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            temp += a[i__ + j * a_dim1] * x[i__];
/* L130: */
            }
            // 更新 x 的值
            x[j] = temp;
/* L140: */
        }
        } else {
        // 处理增量不为1的情况
        jx = kx;
        // 从第一列开始向后遍历
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            temp = x[jx];
            ix = jx;
            // 如果 A 不是单位矩阵，则考虑 A 的对角元素
            if (nounit) {
            temp *= a[j + j * a_dim1];
            }
            // 向下迭代处理每一列的非对角元素
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            ix += *incx;
            temp += a[i__ + j * a_dim1] * x[ix];
/* L150: */
            }
            // 更新 x 的值
            x[jx] = temp;
            jx += *incx;
/* L160: */
        }
        }
    }
    }

    return 0;

/*     End of DTRMV . */

} /* dtrmv_ */

/* Subroutine */ int dtrsm_(char *side, char *uplo, char *transa, char *diag,
    integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
    lda, doublereal *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, k, info;
    static doublereal temp;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    DTRSM  solves one of the matrix equations

       op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'.

    The matrix X is overwritten on B.

    Arguments
    ==========
    SIDE   - CHARACTER*1.
             ! 在进入时，SIDE 指定 op(A) 在 X 左侧或右侧的位置：
             ! 
             !    SIDE = 'L' 或 'l'   op(A)*X = alpha*B.
             ! 
             !    SIDE = 'R' 或 'r'   X*op(A) = alpha*B.
             ! 
             ! 退出时保持不变。

    UPLO   - CHARACTER*1.
             ! 在进入时，UPLO 指定矩阵 A 是上三角还是下三角：
             ! 
             !    UPLO = 'U' 或 'u'   A 是上三角矩阵。
             ! 
             !    UPLO = 'L' 或 'l'   A 是下三角矩阵。
             ! 
             ! 退出时保持不变。

    TRANSA - CHARACTER*1.
             ! 在进入时，TRANSA 指定在矩阵乘法中使用 op(A) 的形式：
             ! 
             !    TRANSA = 'N' 或 'n'   op(A) = A.
             ! 
             !    TRANSA = 'T' 或 't'   op(A) = A'.
             ! 
             !    TRANSA = 'C' 或 'c'   op(A) = A'.
             ! 
             ! 退出时保持不变。

    DIAG   - CHARACTER*1.
             ! 在进入时，DIAG 指定矩阵 A 是否为单位三角形式：
             ! 
             !    DIAG = 'U' 或 'u'   A 假定为单位三角形式。
             ! 
             !    DIAG = 'N' 或 'n'   A 不假定为单位三角形式。
             ! 
             ! 退出时保持不变。

    M      - INTEGER.
             ! 在进入时，M 指定矩阵 B 的行数。M 至少为零。
             ! 退出时保持不变。

    N      - INTEGER.
             ! 在进入时，N 指定矩阵 B 的列数。N 至少为零。
             ! 退出时保持不变。

    ALPHA  - DOUBLE PRECISION.
             ! 在进入时，ALPHA 指定标量 alpha。当 alpha 为零时，A 不被引用，进入前 B 无需设置。
             ! 退出时保持不变。

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, k ), where k is m
             ! 在进入时，A 是 DOUBLE PRECISION 数组，维度为 (LDA, k)，当 SIDE = 'L' 或 'l' 时 k 是 m，当 SIDE = 'R' 或 'r' 时 k 是 n。
             ! 进入时，若 UPLO = 'U' 或 'u'，数组 A 的前 k 行 k 列应包含上三角部分，且严格下三角部分不被引用。
             ! 进入时，若 UPLO = 'L' 或 'l'，数组 A 的前 k 行 k 列应包含下三角部分，且严格上三角部分不被引用。
             ! 当 DIAG = 'U' 或 'u' 时，A 的对角元素也不被引用，但假定为单位矩阵元素。
             ! 退出时保持不变。
    # LDA    - INTEGER.
    #          On entry, LDA specifies the first dimension of A as declared
    #          in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
    #          LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
    #          then LDA must be at least max( 1, n ).
    #          Unchanged on exit.

    # B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
    #          Before entry,  the leading  m by n part of the array  B must
    #          contain  the  right-hand  side  matrix  B,  and  on exit  is
    #          overwritten by the solution matrix  X.

    # LDB    - INTEGER.
    #          On entry, LDB specifies the first dimension of B as declared
    #          in  the  calling  (sub)  program.   LDB  must  be  at  least
    #          max( 1, m ).
    #          Unchanged on exit.

    # Further Details
    # ===============

    # Level 3 Blas routine.


    # -- Written on 8-February-1989.
    #    Jack Dongarra, Argonne National Laboratory.
    #    Iain Duff, AERE Harwell.
    #    Jeremy Du Croz, Numerical Algorithms Group Ltd.
    #    Sven Hammarling, Numerical Algorithms Group Ltd.

    # =====================================================================


    # Test the input parameters.
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    lside = lsame_(side, "L"); // 检查参数 side 是否为 'L'，确定是否左侧矩阵操作
    if (lside) {
    nrowa = *m; // 如果是左侧操作，则 nrowa 为 m
    } else {
    nrowa = *n; // 否则为 n
    }
    nounit = lsame_(diag, "N"); // 检查参数 diag 是否为 'N'，确定是否不是单位对角矩阵
    upper = lsame_(uplo, "U"); // 检查参数 uplo 是否为 'U'，确定是否为上三角矩阵

    info = 0; // 初始化错误信息码为 0
    if (! lside && ! lsame_(side, "R")) { // 如果 side 既不是 'L' 也不是 'R'
    info = 1; // 错误信息码设为 1
    } else if (! upper && ! lsame_(uplo, "L")) { // 如果 uplo 既不是 'U' 也不是 'L'
    info = 2; // 错误信息码设为 2
    } else if (! lsame_(transa, "N") && ! lsame_(transa,
         "T") && ! lsame_(transa, "C")) { // 如果 transa 既不是 'N' 也不是 'T' 也不是 'C'
    info = 3; // 错误信息码设为 3
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
        "N")) { // 如果 diag 既不是 'U' 也不是 'N'
    info = 4; // 错误信息码设为 4
    } else if (*m < 0) { // 如果 m 小于 0
    info = 5; // 错误信息码设为 5
    } else if (*n < 0) { // 如果 n 小于 0
    info = 6; // 错误信息码设为 6
    } else if (*lda < max(1,nrowa)) { // 如果 lda 小于 max(1, nrowa)
    info = 9; // 错误信息码设为 9
    } else if (*ldb < max(1,*m)) { // 如果 ldb 小于 max(1, m)
    info = 11; // 错误信息码设为 11
    }
    if (info != 0) {
    xerbla_("DTRSM ", &info); // 调用错误处理例程，打印错误信息并退出
    return 0; // 返回 0
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0) { // 如果 m 或 n 为 0
    return 0; // 直接返回 0
    }

/*     And when alpha.eq.zero. */

    if (*alpha == 0.) { // 如果 alpha 等于 0
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
        b[i__ + j * b_dim1] = 0.; // 将 B 矩阵中的元素设为 0
/* L10: */
        }
/* L20: */
    }
    return 0; // 返回 0
    }

/*     Start the operations. */

    if (lside) { // 如果是左侧操作
    if (lsame_(transa, "N")) { // 如果不需要转置 A

/*           Form  B := alpha*inv( A )*B. */

        if (upper) { // 如果是上三角矩阵 A
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            if (*alpha != 1.) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
                    ;
/* L30: */
            }
            }
            for (k = *m; k >= 1; --k) {
            if (b[k + j * b_dim1] != 0.) {
                if (nounit) {
                b[k + j * b_dim1] /= a[k + k * a_dim1]; // B 矩阵除以 A 的对角元素
                }
                i__2 = k - 1;
                for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[
                    i__ + k * a_dim1]; // B 矩阵减去 A 的非对角元素乘以相应系数
/* L40: */
                }
            }
/* L50: */
            }
/* L60: */
        }
        } else { // 如果是下三角矩阵 A
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            if (*alpha != 1.) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
                    ;
/* L70: */
            }
            }
            i__2 = *m;
            for (k = 1; k <= i__2; ++k) {
            if (b[k + j * b_dim1] != 0.) {
                if (nounit) {
                b[k + j * b_dim1] /= a[k + k * a_dim1]; // B 矩阵除以 A 的对角元素
                }
                i__3 = *m;
                for (i__ = k + 1; i__ <= i__3; ++i__) {
                b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[
                    i__ + k * a_dim1]; // B 矩阵减去 A 的非对角元素乘以相应系数
/* L80: */
                }
            }
/* L90: */
            }
/* L100: */
        }
        }
    } else {



# 如果前面的条件不满足，执行以下代码块
/*
   Form  B := alpha*inv( A' )*B.
   根据 alpha 和 A 的逆矩阵乘积，更新 B 矩阵的值。
*/
if (upper) {
    // 处理上三角矩阵
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 计算临时变量 temp
            temp = *alpha * b[i__ + j * b_dim1];
            i__3 = i__ - 1;
            for (k = 1; k <= i__3; ++k) {
                // 更新 temp，减去对应的 A 和 B 元素乘积
                temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];
                /* L110: */
            }
            if (nounit) {
                // 如果不是单位对角矩阵，除以对应的 A 对角元素
                temp /= a[i__ + i__ * a_dim1];
            }
            // 更新 B 矩阵的元素值
            b[i__ + j * b_dim1] = temp;
            /* L120: */
        }
        /* L130: */
    }
} else {
    // 处理下三角矩阵
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        for (i__ = *m; i__ >= 1; --i__) {
            // 计算临时变量 temp
            temp = *alpha * b[i__ + j * b_dim1];
            i__2 = *m;
            for (k = i__ + 1; k <= i__2; ++k) {
                // 更新 temp，减去对应的 A 和 B 元素乘积
                temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];
                /* L140: */
            }
            if (nounit) {
                // 如果不是单位对角矩阵，除以对应的 A 对角元素
                temp /= a[i__ + i__ * a_dim1];
            }
            // 更新 B 矩阵的元素值
            b[i__ + j * b_dim1] = temp;
            /* L150: */
        }
        /* L160: */
    }
}
} else {
if (lsame_(transa, "N")) {

    /*
       Form  B := alpha*B*inv( A ).
       根据 alpha 和 A 的逆矩阵乘积，更新 B 矩阵的值。
    */

    if (upper) {
        // 处理上三角矩阵
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            if (*alpha != 1.) {
                // 如果 alpha 不等于 1，对 B 矩阵每列乘以 alpha
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
                    /* L170: */
                }
            }
            // 处理 A 的列元素对 B 的影响
            i__2 = j - 1;
            for (k = 1; k <= i__2; ++k) {
                if (a[k + j * a_dim1] != 0.) {
                    i__3 = *m;
                    for (i__ = 1; i__ <= i__3; ++i__) {
                        // 更新 B 矩阵的元素值，减去对应的 A 和 B 元素乘积
                        b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[i__ + k * b_dim1];
                        /* L180: */
                    }
                }
                /* L190: */
            }
            if (nounit) {
                // 如果不是单位对角矩阵，除以对应的 A 对角元素
                temp = 1. / a[j + j * a_dim1];
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
                    /* L200: */
                }
            }
            /* L210: */
        }
    } else {
        // 处理下三角矩阵
        for (j = *n; j >= 1; --j) {
            if (*alpha != 1.) {
                // 如果 alpha 不等于 1，对 B 矩阵每列乘以 alpha
                i__1 = *m;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
                    /* L220: */
                }
            }
            // 处理 A 的列元素对 B 的影响
            i__1 = *n;
            for (k = j + 1; k <= i__1; ++k) {
                if (a[k + j * a_dim1] != 0.) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        // 更新 B 矩阵的元素值，减去对应的 A 和 B 元素乘积
                        b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[i__ + k * b_dim1];
                        /* L230: */
                    }
                }
                /* L240: */
            }
            if (nounit) {
                // 如果不是单位对角矩阵，除以对应的 A 对角元素
                temp = 1. / a[j + j * a_dim1];
                i__1 = *m;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
                    /* L250: */
                }
            }
            /* L260: */
        }
    }
}
/* L250: */
            }
            }
/* L260: */
        }
        }
    } else {

/*           Form  B := alpha*B*inv( A' ). */

        if (upper) {
        for (k = *n; k >= 1; --k) {
            if (nounit) {
            temp = 1. / a[k + k * a_dim1];
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L270: */
            }
            }
            i__1 = k - 1;
            for (j = 1; j <= i__1; ++j) {
            if (a[j + k * a_dim1] != 0.) {
                temp = a[j + k * a_dim1];
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + j * b_dim1] -= temp * b[i__ + k *
                    b_dim1];
/* L280: */
                }
            }
/* L290: */
            }
            if (*alpha != 1.) {
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1]
                    ;
/* L300: */
            }
            }
/* L310: */
        }
        } else {
        i__1 = *n;
        for (k = 1; k <= i__1; ++k) {
            if (nounit) {
            temp = 1. / a[k + k * a_dim1];
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L320: */
            }
            }
            i__2 = *n;
            for (j = k + 1; j <= i__2; ++j) {
            if (a[j + k * a_dim1] != 0.) {
                temp = a[j + k * a_dim1];
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                b[i__ + j * b_dim1] -= temp * b[i__ + k *
                    b_dim1];
/* L330: */
                }
            }
/* L340: */
            }
            if (*alpha != 1.) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1]
                    ;
/* L350: */
            }
            }
/* L360: */
        }
        }
    }
    }

    return 0;

/*     End of DTRSM . */

} /* dtrsm_ */

doublereal dzasum_(integer *n, doublecomplex *zx, integer *incx)
{
    /* System generated locals */
    integer i__1;
    doublereal ret_val;

    /* Local variables */
    static integer i__, ix;
    static doublereal stemp;
    extern doublereal dcabs1_(doublecomplex *);


/*
    Purpose
    =======

       DZASUM takes the sum of the absolute values.

    Further Details
    ===============

       jack dongarra, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --zx;

    /* Function Body */
    ret_val = 0.;
    stemp = 0.;
    if (*n <= 0 || *incx <= 0) {
    return ret_val;
    }
    if (*incx == 1) {
    goto L20;
    }

/*        code for increment not equal to 1 */
    ix = 1;
    // 初始化 ix 变量，设置为 1，用于索引数组 zx 的起始位置

    i__1 = *n;
    // 将变量 i__1 设置为指针 n 指向的值，表示循环的上界

    for (i__ = 1; i__ <= i__1; ++i__) {
    // 循环，从 i__ = 1 开始，每次增加 1，直到 i__ <= i__1 为止

    stemp += dcabs1_(&zx[ix]);
    // 计算复数 zx[ix] 的模并累加到变量 stemp 上，dcabs1_ 是一个函数，计算复数的绝对值

    ix += *incx;
    // 增加 ix 的值，以便下次循环时索引下一个元素，incx 是一个步长变量，控制每次增加的索引值
/*
    Purpose
    =======

    DZNRM2 returns the euclidean norm of a vector via the function
    name, so that

       DZNRM2 := sqrt( conjg( x' )*x )

    Further Details
    ===============

    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to ZLASSQ.
       Sven Hammarling, Nag Ltd.

    =====================================================================
*/

doublereal dznrm2_(integer *n, doublecomplex *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal ret_val, d__1;

    /* Local variables */
    static integer ix;
    static doublereal ssq, temp, norm, scale;

    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n < 1 || *incx < 1) {
        norm = 0.;
    } else {
        scale = 0.;
        ssq = 1.;
        /*
              The following loop is equivalent to this call to the LAPACK
              auxiliary routine:
              CALL ZLASSQ( N, X, INCX, SCALE, SSQ )
        */
        i__1 = (*n - 1) * *incx + 1;
        i__2 = *incx;
        for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
            i__3 = ix;
            if (x[i__3].r != 0.) {
                i__3 = ix;
                temp = (d__1 = x[i__3].r, abs(d__1));
                if (scale < temp) {
                    /* Computing 2nd power */
                    d__1 = scale / temp;
                    ssq = ssq * (d__1 * d__1) + 1.;
                    scale = temp;
                } else {
                    /* Computing 2nd power */
                    d__1 = temp / scale;
                    ssq += d__1 * d__1;
                }
            }
            if (d_imag(&x[ix]) != 0.) {
                temp = (d__1 = d_imag(&x[ix]), abs(d__1));
                if (scale < temp) {
                    /* Computing 2nd power */
                    d__1 = scale / temp;
                    ssq = ssq * (d__1 * d__1) + 1.;
                    scale = temp;
                } else {
                    /* Computing 2nd power */
                    d__1 = temp / scale;
                    ssq += d__1 * d__1;
                }
            }
/* L10: */
        }
        norm = scale * sqrt(ssq);
    }

    ret_val = norm;
    return ret_val;

/*     End of DZNRM2. */

} /* dznrm2_ */
    # 如果 n 的值小于 1 或者 incx 的值小于等于 0，则返回 ret_val
    if (*n < 1 || *incx <= 0) {
    return ret_val;
    }
    # 将 ret_val 的值设置为 1
    ret_val = 1;
    # 如果 n 的值等于 1，则返回 ret_val
    if (*n == 1) {
    return ret_val;
    }
    # 如果 incx 的值等于 1，则跳转到标签 L20
    if (*incx == 1) {
    goto L20;
    }
/*        code for increment not equal to 1 */

/* 设置初始索引为1 */
ix = 1;
/* 计算第一个元素的绝对值，作为初始的最大绝对值 */
smax = scabs1_(&cx[1]);
/* 增加索引以访问下一个元素 */
ix += *incx;
/* 循环遍历数组 */
i__1 = *n;
for (i__ = 2; i__ <= i__1; ++i__) {
    /* 如果当前元素的绝对值小于等于当前最大绝对值，则跳转到L5 */
    if (scabs1_(&cx[ix]) <= smax) {
        goto L5;
    }
    /* 更新返回值为当前索引 */
    ret_val = i__;
    /* 更新最大绝对值为当前元素的绝对值 */
    smax = scabs1_(&cx[ix]);
L5:
    /* 增加索引以访问下一个元素 */
    ix += *incx;
/* L10: */
}
/* 返回最大绝对值对应的索引 */
return ret_val;

/*        code for increment equal to 1 */

L20:
/* 设置初始的最大绝对值为第一个元素的绝对值 */
smax = scabs1_(&cx[1]);
/* 循环遍历数组 */
i__1 = *n;
for (i__ = 2; i__ <= i__1; ++i__) {
    /* 如果当前元素的绝对值小于等于当前最大绝对值，则跳转到L30 */
    if (scabs1_(&cx[i__]) <= smax) {
        goto L30;
    }
    /* 更新返回值为当前索引 */
    ret_val = i__;
    /* 更新最大绝对值为当前元素的绝对值 */
    smax = scabs1_(&cx[i__]);
L30:
    /* 空语句 */
    ;
}
/* 返回最大绝对值对应的索引 */
return ret_val;
} /* icamax_ */

integer idamax_(integer *n, doublereal *dx, integer *incx)
{
/*        code for increment not equal to 1 */

/* 声明和初始化局部变量 */
/* 初始返回值为0 */
ret_val = 0;
/* 如果n小于1或者incx小于等于0，则直接返回0 */
if (*n < 1 || *incx <= 0) {
    return ret_val;
}
/* 设置初始返回值为1 */
ret_val = 1;
/* 如果n等于1，则直接返回1 */
if (*n == 1) {
    return ret_val;
}
/* 如果incx等于1，则跳转到标签L20 */
if (*incx == 1) {
    goto L20;
}

/* 设置初始索引为1 */
ix = 1;
/* 计算第一个元素的绝对值，作为初始的最大绝对值 */
dmax__ = abs(dx[1]);
/* 增加索引以访问下一个元素 */
ix += *incx;
/* 循环遍历数组 */
i__1 = *n;
for (i__ = 2; i__ <= i__1; ++i__) {
    /* 如果当前元素的绝对值小于等于当前最大绝对值，则跳转到L5 */
    if ((d__1 = dx[ix], abs(d__1)) <= dmax__) {
        goto L5;
    }
    /* 更新返回值为当前索引 */
    ret_val = i__;
    /* 更新最大绝对值为当前元素的绝对值 */
    dmax__ = (d__1 = dx[ix], abs(d__1));
L5:
    /* 增加索引以访问下一个元素 */
    ix += *incx;
/* L10: */
}
/* 返回最大绝对值对应的索引 */
return ret_val;

/*        code for increment equal to 1 */

L20:
/* 计算第一个元素的绝对值，作为初始的最大绝对值 */
dmax__ = abs(dx[1]);
/* 循环遍历数组 */
i__1 = *n;
for (i__ = 2; i__ <= i__1; ++i__) {
    /* 如果当前元素的绝对值小于等于当前最大绝对值，则跳转到L30 */
    if ((d__1 = dx[i__], abs(d__1)) <= dmax__) {
        goto L30;
    }
    /* 更新返回值为当前索引 */
    ret_val = i__;
    /* 更新最大绝对值为当前元素的绝对值 */
    dmax__ = (d__1 = dx[i__], abs(d__1));
L30:
    /* 空语句 */
    ;
}
/* 返回最大绝对值对应的索引 */
return ret_val;
} /* idamax_ */

integer isamax_(integer *n, real *sx, integer *incx)
{
/*        code for increment not equal to 1 */

/* 声明和初始化局部变量 */
/* 初始返回值为0 */
ret_val = 0;
/* 如果n小于1或者incx小于等于0，则直接返回0 */
if (*n < 1 || *incx <= 0) {
    return ret_val;
}
/* 设置初始返回值为1 */
ret_val = 1;
/* 如果n等于1，则直接返回1 */
if (*n == 1) {
    return ret_val;
}
/* 如果incx等于1，则跳转到标签L20 */
if (*incx == 1) {
    goto L20;
}
/*        code for increment not equal to 1 */

    ix = 1;  // Initialize index ix to 1
    smax = dabs(sx[1]);  // Initialize smax to the absolute value of sx[1]
    ix += *incx;  // Increment ix by the value of incx
    i__1 = *n;  // Assign *n to i__1 (loop limit)
    for (i__ = 2; i__ <= i__1; ++i__) {  // Loop from 2 to *n (inclusive)
    if ((r__1 = sx[ix], dabs(r__1)) <= smax) {  // Check if absolute value of sx[ix] is less than or equal to smax
        goto L5;  // Jump to label L5 if condition is true
    }
    ret_val = i__;  // Assign current loop index i__ to ret_val
    smax = (r__1 = sx[ix], dabs(r__1));  // Update smax to the absolute value of sx[ix]
L5:
    ix += *incx;  // Increment ix by the value of incx
/* L10: */  // Label L10 for future reference
    }
    return ret_val;  // Return the index of the element with maximum absolute value

/*        code for increment equal to 1 */

L20:  // Label L20 indicates the start of this block
    smax = dabs(sx[1]);  // Initialize smax to the absolute value of sx[1]
    i__1 = *n;  // Assign *n to i__1 (loop limit)
    for (i__ = 2; i__ <= i__1; ++i__) {  // Loop from 2 to *n (inclusive)
    if ((r__1 = sx[i__], dabs(r__1)) <= smax) {  // Check if absolute value of sx[i__] is less than or equal to smax
        goto L30;  // Jump to label L30 if condition is true
    }
    ret_val = i__;  // Assign current loop index i__ to ret_val
    smax = (r__1 = sx[i__], dabs(r__1));  // Update smax to the absolute value of sx[i__]
L30:  // Label L30 for future reference
    ;
    }
    return ret_val;  // Return the index of the element with maximum absolute value
} /* isamax_ */

integer izamax_(integer *n, doublecomplex *zx, integer *incx)
{
    /* System generated locals */
    integer ret_val, i__1;

    /* Local variables */
    static integer i__, ix;
    static doublereal smax;
    extern doublereal dcabs1_(doublecomplex *);

/*
    Purpose
    =======

       IZAMAX finds the index of element having max. absolute value.

    Further Details
    ===============

       jack dongarra, 1/15/85.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --zx;

    /* Function Body */
    ret_val = 0;  // Initialize ret_val to 0
    if (*n < 1 || *incx <= 0) {  // Check if *n is less than 1 or *incx is less than or equal to 0
    return ret_val;  // Return 0 if condition is true
    }
    ret_val = 1;  // Initialize ret_val to 1
    if (*n == 1) {  // Check if *n is equal to 1
    return ret_val;  // Return 1 if condition is true
    }
    if (*incx == 1) {  // Check if *incx is equal to 1
    goto L20;  // Jump to label L20
    }

/*        code for increment not equal to 1 */

    ix = 1;  // Initialize index ix to 1
    smax = dcabs1_(&zx[1]);  // Initialize smax to the absolute value of zx[1]
    ix += *incx;  // Increment ix by the value of incx
    i__1 = *n;  // Assign *n to i__1 (loop limit)
    for (i__ = 2; i__ <= i__1; ++i__) {  // Loop from 2 to *n (inclusive)
    if (dcabs1_(&zx[ix]) <= smax) {  // Check if absolute value of zx[ix] is less than or equal to smax
        goto L5;  // Jump to label L5 if condition is true
    }
    ret_val = i__;  // Assign current loop index i__ to ret_val
    smax = dcabs1_(&zx[ix]);  // Update smax to the absolute value of zx[ix]
L5:
    ix += *incx;  // Increment ix by the value of incx
/* L10: */  // Label L10 for future reference
    }
    return ret_val;  // Return the index of the element with maximum absolute value

/*        code for increment equal to 1 */

L20:  // Label L20 indicates the start of this block
    smax = dcabs1_(&zx[1]);  // Initialize smax to the absolute value of zx[1]
    i__1 = *n;  // Assign *n to i__1 (loop limit)
    for (i__ = 2; i__ <= i__1; ++i__) {  // Loop from 2 to *n (inclusive)
    if (dcabs1_(&zx[i__]) <= smax) {  // Check if absolute value of zx[i__] is less than or equal to smax
        goto L30;  // Jump to label L30 if condition is true
    }
    ret_val = i__;  // Assign current loop index i__ to ret_val
    smax = dcabs1_(&zx[i__]);  // Update smax to the absolute value of zx[i__]
L30:  // Label L30 for future reference
    ;
    }
    return ret_val;  // Return the index of the element with maximum absolute value
} /* izamax_ */

/* Subroutine */ int saxpy_(integer *n, real *sa, real *sx, integer *incx,
    real *sy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;

/*
    Purpose
    =======

       SAXPY constant times a vector plus a vector.
       uses unrolled loop for increments equal to one.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --sy;
    --sx;

    /* Function Body */
    if (*n <= 0) {  // Check if *n is less than or equal to 0
    return 0;  // Return 0 if condition is true
    }
    if (*sa == 0.f) {  // Check if *sa is equal to 0
    return 0;  // Return 0 if condition is true
    }
    if (*incx == 1 && *incy == 1) {  // Check if both *incx and *incy are equal to 1
    goto L20;  // Jump to label L20
    }
/*
    设置增量不等于1的情况下的代码
*/
    ix = 1;  // 初始化 ix 为1
    iy = 1;  // 初始化 iy 为1
    if (*incx < 0) {  // 如果增量 incx 小于0
        ix = (-(*n) + 1) * *incx + 1;  // 计算 ix 的值
    }
    if (*incy < 0) {  // 如果增量 incy 小于0
        iy = (-(*n) + 1) * *incy + 1;  // 计算 iy 的值
    }
    i__1 = *n;  // 循环次数为 n
    for (i__ = 1; i__ <= i__1; ++i__) {  // 循环 n 次
        sy[iy] += *sa * sx[ix];  // 计算 sy[iy] 的值
        ix += *incx;  // 更新 ix
        iy += *incy;  // 更新 iy
/* L10: */  // 循环标签
    }
    return 0;  // 返回值为0，表示成功

/*
    增量均等于1时的代码
    清理循环
*/

L20:  // 循环开始标签
    m = *n % 4;  // 计算 m
    if (m == 0) {  // 如果 m 等于0
        goto L40;  // 转到标签 L40
    }
    i__1 = m;  // 循环次数为 m
    for (i__ = 1; i__ <= i__1; ++i__) {  // 循环 m 次
        sy[i__] += *sa * sx[i__];  // 计算 sy[i__] 的值
/* L30: */  // 循环标签
    }
    if (*n < 4) {  // 如果 n 小于4
        return 0;  // 返回值为0，表示成功
    }
L40:  // 循环开始标签
    mp1 = m + 1;  // 计算 mp1
    i__1 = *n;  // 循环次数为 n
    for (i__ = mp1; i__ <= i__1; i__ += 4) {  // 循环，步长为4
        sy[i__] += *sa * sx[i__];  // 计算 sy[i__] 的值
        sy[i__ + 1] += *sa * sx[i__ + 1];  // 计算 sy[i__+1] 的值
        sy[i__ + 2] += *sa * sx[i__ + 2];  // 计算 sy[i__+2] 的值
        sy[i__ + 3] += *sa * sx[i__ + 3];  // 计算 sy[i__+3] 的值
/* L50: */  // 循环标签
    }
    return 0;  // 返回值为0，表示成功
} /* saxpy_ */

doublereal scabs1_(singlecomplex *z__)
{
    /* System generated locals */
    real ret_val, r__1, r__2;

    /*
        Purpose
        =======
        计算复数的绝对值

        =====================================================================
    */

    ret_val = (r__1 = z__->r, dabs(r__1)) + (r__2 = r_imag(z__), dabs(r__2));  // 计算复数 z__ 的绝对值
    return ret_val;  // 返回计算结果
} /* scabs1_ */

doublereal scasum_(integer *n, singlecomplex *cx, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    real ret_val, r__1, r__2;

    /* Local variables */
    static integer i__, nincx;
    static real stemp;

    /*
        Purpose
        =======
        计算复向量的绝对值之和，返回单精度结果。

        Further Details
        ===============
        jack dongarra, linpack, 3/11/78.
        modified 3/93 to return if incx .le. 0.
        modified 12/3/93, array(1) declarations changed to array(*)

        =====================================================================
    */

    /* Parameter adjustments */
    --cx;

    /* Function Body */
    ret_val = 0.f;  // 初始化返回值为0
    stemp = 0.f;  // 初始化临时变量 stemp 为0
    if (*n <= 0 || *incx <= 0) {  // 如果 n 小于等于0或者 incx 小于等于0
        return ret_val;  // 返回0
    }
    if (*incx == 1) {  // 如果 incx 等于1
        goto L20;  // 转到标签 L20
    }

/*        code for increment not equal to 1 */

    nincx = *n * *incx;  // 计算 nincx
    i__1 = nincx;  // 循环次数为 nincx
    i__2 = *incx;  // 步长为 incx
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {  // 循环
        i__3 = i__;  // 计算当前索引
        stemp = stemp + (r__1 = cx[i__3].r, dabs(r__1)) + (r__2 = r_imag(&cx[
            i__]), dabs(r__2));  // 计算 stemp 的值
/* L10: */  // 循环标签
    }
    ret_val = stemp;  // 将 stemp 赋值给返回值
    return ret_val;  // 返回计算结果

/*        code for increment equal to 1 */

L20:  // 循环开始标签
    i__2 = *n;  // 循环次数为 n
    for (i__ = 1; i__ <= i__2; ++i__) {  // 循环
        i__1 = i__;  // 计算当前索引
        stemp = stemp + (r__1 = cx[i__1].r, dabs(r__1)) + (r__2 = r_imag(&cx[
            i__]), dabs(r__2));  // 计算 stemp 的值
/* L30: */  // 循环标签
    }
    ret_val = stemp;  // 将 stemp 赋值给返回值
    return ret_val;  // 返回计算结果
} /* scasum_ */

doublereal scnrm2_(integer *n, singlecomplex *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    real ret_val, r__1;
    /* 定义本地变量 */
    static integer ix;
    /* 定义用于累积平方和的变量 */
    static real ssq, temp;
    /* 定义用于归一化和缩放的变量 */
    static real norm, scale;
/*
    Purpose
    =======

    SCOPY copies a vector, x, to a vector, y.
    Uses unrolled loops for increments equal to 1.

    Further Details
    ===============

    Jack Dongarra, LINPACK, 3/11/78.
    Modified 12/3/93, array(1) declarations changed to array(*)
*/

/* Parameter adjustments */
--sy;
--sx;

/* Function Body */
if (*n <= 0) {
    // 如果向量长度小于等于0，直接返回
    return 0;
}
if (*incx == 1 && *incy == 1) {
    // 如果增量均为1，则使用跳转到L20的方式处理
    goto L20;
}

/*
      code for unequal increments or equal increments
      not equal to 1
*/

ix = 1;
iy = 1;
if (*incx < 0) {
    // 如果增量小于0，计算起始索引
    ix = (-(*n) + 1) * *incx + 1;
}
if (*incy < 0) {
    // 如果增量小于0，计算起始索引
    iy = (-(*n) + 1) * *incy + 1;
}
// 循环遍历向量元素
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
    // 将向量sx中的元素复制到向量sy中
    sy[iy] = sx[ix];
    // 更新索引
    ix += *incx;
    iy += *incy;
    // 继续下一个元素
    /* L10: */
}
// 返回完成的复制操作
return 0;

/*
      code for both increments equal to 1

      clean-up loop
*/

L20:
m = *n % 7;
    # 如果 m 等于 0，则跳转到标签 L40
    if (m == 0) {
    # 循环次数为 m
    goto L40;
    }
    # 设置循环变量 i__1 为 m
    i__1 = m;
    # 循环从 1 到 i__1
    for (i__ = 1; i__ <= i__1; ++i__) {
    # 将 sx 数组中第 i__ 个元素的值赋给 sy 数组中第 i__ 个元素
    sy[i__] = sx[i__];
/*
    Purpose
    =======

    SGEMM performs one of the matrix-matrix operations
    C := alpha*op( A )*op( B ) + beta*C,

    where
    - A, B, and C are matrices,
    - alpha and beta are scalars,
    - op(X) is one of op(X) = X or op(X) = X', representing the matrix A or its transpose,
    - lda, ldb, ldc are leading dimensions of matrices A, B, and C respectively.

    Further Details
    ===============

    This routine follows the standard matrix multiplication formula but with options for transposed matrices and scalar coefficients.
*/

/* Subroutine */ int sgemm_(char *transa, char *transb, integer *m, integer *n, integer *k, real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta, real *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, l, info;
    static logical nota, notb;
    static real temp;
    extern logical lsame_(char *, char *);
    static integer nrowa, nrowb;
    extern /* Subroutine */ int xerbla_(char *, integer *);

    /* Parameter adjustments */
    --c__;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */

    /* Test the input parameters. */
    info = 0;
    nota = lsame_(transa, "N");
    notb = lsame_(transb, "N");
    if (! nota && ! lsame_(transa, "T") && ! lsame_(transa, "C")) {
        info = 1;
    } else if (! notb && ! lsame_(transb, "T") && ! lsame_(transb, "C")) {
        info = 2;
    } else if (*m < 0) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*k < 0) {
        info = 5;
    } else if (*lda < max(1,*m)) {
        info = 8;
    } else if (*ldb < max(1,*k)) {
        info = 10;
    } else if (*ldc < max(1,*m)) {
        info = 13;
    }
    if (info != 0) {
        xerbla_("SGEMM ", &info);
        return 0;
    }

    /* Quick return if possible. */
    if (*m == 0 || *n == 0 || ((*alpha == 0.f || *k == 0) && *beta == 1.f)) {
        return 0;
    }

    /* And when alpha.eq.zero. */
    if (*alpha == 0.f) {
        if (*beta == 0.f) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    c__[i__] = 0.f;
                }
                c__ += *ldc;
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    c__[i__] *= *beta;
                }
                c__ += *ldc;
            }
        }
        return 0;
    }

    /* Start the operations. */
    if (notb) {
        if (nota) {

            /* Form  C := alpha*A*A' + beta*C. */
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (*beta == 0.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__] = 0.f;
                    }
                } else if (*beta != 1.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__] *= *beta;
                    }
                }
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    temp = *alpha * a[i__ + l * a_dim1];
                    i__3 = *m;
                    for (i__ = 1; i__ <= i__3; ++i__) {
                        c__[i__] += temp * a[i__ + l * a_dim1];
                    }
                }
                c__ += *ldc;
            }

        } else {

            /* Form  C := alpha*A'*A + beta*C. */
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    temp = 0.f;
                    i__3 = *k;
                    for (l = 1; l <= i__3; ++l) {
                        temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
                    }
                    if (*beta == 0.f) {
                        c__[i__] = *alpha * temp;
                    } else {
                        c__[i__] = *alpha * temp + *beta * c__[i__];
                    }
                }
                c__ += *ldc;
            }

        }
    } else {
        if (nota) {

            /* Form  C := alpha*A*B + beta*C. */
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (*beta == 0.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__] = 0.f;
                    }
                } else if (*beta != 1.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__] *= *beta;
                    }
                }
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    temp = *alpha * b[l + j * b_dim1];
                    i__3 = *m;
                    for (i__ = 1; i__ <= i__3; ++i__) {
                        c__[i__] += temp * a[i__ + l * a_dim1];
                    }
                }
                c__ += *ldc;
            }

        } else {

            /* Form  C := alpha*A'*B + beta*C. */
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    temp = 0.f;
                    i__3 = *k;
                    for (l = 1; l <= i__3; ++l) {
                        temp += a[l + i__ * a_dim1] * b[l + j * b_dim1];
                    }
                    if (*beta == 0.f) {
                        c__[i__] = *alpha * temp;
                    } else {
                        c__[i__] = *alpha * temp + *beta * c__[i__];
                    }
                }
                c__ += *ldc;
            }

        }
    }

    return 0;
} /* sgemm_ */
    where  op( X ) is one of

       op( X ) = X   or   op( X ) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Arguments
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n',  op( A ) = A.

                TRANSA = 'T' or 't',  op( A ) = A'.

                TRANSA = 'C' or 'c',  op( A ) = A'.

             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in
             the matrix multiplication as follows:

                TRANSB = 'N' or 'n',  op( B ) = B.

                TRANSB = 'T' or 't',  op( B ) = B'.

                TRANSB = 'C' or 'c',  op( B ) = B'.

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( A )  and of the  matrix  C.  M  must  be at least  zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( B ) and the number of columns of the matrix C. N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA  - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.
    ! LDB    - INTEGER.
             ! On entry, LDB specifies the first dimension of B as declared
             ! in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             ! LDB must be at least  max( 1, k ), otherwise  LDB must be at
             ! least  max( 1, n ).
             ! Unchanged on exit.

    ! BETA   - REAL            .
             ! On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             ! supplied as zero then C need not be set on input.
             ! Unchanged on exit.

    ! C      - REAL             array of DIMENSION ( LDC, n ).
             ! Before entry, the leading  m by n  part of the array  C must
             ! contain the matrix  C,  except when  beta  is zero, in which
             ! case C need not be set on entry.
             ! On exit, the array  C  is overwritten by the  m by n  matrix
             ! ( alpha*op( A )*op( B ) + beta*C ).

    ! LDC    - INTEGER.
             ! On entry, LDC specifies the first dimension of C as declared
             ! in  the  calling  (sub)  program.   LDC  must  be  at  least
             ! max( 1, m ).
             ! Unchanged on exit.

    ! Further Details
    ! ===============

    ! Level 3 Blas routine.

    ! -- Written on 8-February-1989.
       ! Jack Dongarra, Argonne National Laboratory.
       ! Iain Duff, AERE Harwell.
       ! Jeremy Du Croz, Numerical Algorithms Group Ltd.
       ! Sven Hammarling, Numerical Algorithms Group Ltd.

    ! =====================================================================


       ! Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
       ! transposed and set  NROWA and  NROWB  as the number of rows
       ! and  columns of  A  and the  number of  rows  of  B  respectively.
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    nota = lsame_(transa, "N");
    notb = lsame_(transb, "N");
    if (nota) {
        nrowa = *m;
    } else {
        nrowa = *k;
    }
    if (notb) {
        nrowb = *k;
    } else {
        nrowb = *n;
    }

/*     Test the input parameters. */

    info = 0;
    if (! nota && ! lsame_(transa, "C") && ! lsame_(
        transa, "T")) {
        info = 1;
    } else if (! notb && ! lsame_(transb, "C") && !
        lsame_(transb, "T")) {
        info = 2;
    } else if (*m < 0) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*k < 0) {
        info = 5;
    } else if (*lda < max(1,nrowa)) {
        info = 8;
    } else if (*ldb < max(1,nrowb)) {
        info = 10;
    } else if (*ldc < max(1,*m)) {
        info = 13;
    }
    if (info != 0) {
        xerbla_("SGEMM ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {
        return 0;
    }

/*     And if  alpha.eq.zero. */

    if (*alpha == 0.f) {
        if (*beta == 0.f) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    c__[i__ + j * c_dim1] = 0.f;
                }
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                }
            }
        }
        return 0;
    }

/*     Start the operations. */

    if (notb) {
        if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (*beta == 0.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                } else if (*beta != 1.f) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    if (b[l + j * b_dim1] != 0.f) {
                        temp = *alpha * b[l + j * b_dim1];
                        i__3 = *m;
                        for (i__ = 1; i__ <= i__3; ++i__) {
                            c__[i__ + j * c_dim1] += temp * a[i__ + l *
                                a_dim1];
                        }
                    }
                }
            }
        } else {
/*
    Purpose:
    SGEMM performs matrix-matrix multiplication with options for transposed and scaled results.

    Arguments:
    - trans: Character flag specifying transpose operation ('N', 'T', 'C').
    - m: Number of rows in matrix C.
    - n: Number of columns in matrix C.
    - alpha: Scalar multiplier for A'*B or A*B.
    - a: Matrix A of dimensions (lda, *).
    - lda: Leading dimension of A.
    - b: Matrix B of dimensions (ldb, *).
    - ldb: Leading dimension of B.
    - beta: Scalar multiplier for matrix C.
    - c: Matrix C of dimensions (ldc, *).
    - ldc: Leading dimension of C.
*/

/*           Form  C := alpha*A'*B + beta*C */

i__1 = *n;
for (j = 1; j <= i__1; ++j) {
    i__2 = *m;
    for (i__ = 1; i__ <= i__2; ++i__) {
        temp = 0.f;
        i__3 = *k;
        for (l = 1; l <= i__3; ++l) {
            temp += a[l + i__ * a_dim1] * b[l + j * b_dim1];
        }
        if (*beta == 0.f) {
            c__[i__ + j * c_dim1] = *alpha * temp;
        } else {
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
        }
    }
}

/* L100: */

/* L110: */

/* L120: */

} else {
if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

i__1 = *n;
for (j = 1; j <= i__1; ++j) {
    if (*beta == 0.f) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            c__[i__ + j * c_dim1] = 0.f;
        }
    } else if (*beta != 1.f) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
        }
    }
    i__2 = *k;
    for (l = 1; l <= i__2; ++l) {
        if (b[j + l * b_dim1] != 0.f) {
            temp = *alpha * b[j + l * b_dim1];
            i__3 = *m;
            for (i__ = 1; i__ <= i__3; ++i__) {
                c__[i__ + j * c_dim1] += temp * a[i__ + l *
                    a_dim1];
            }
        }
    }
}

/* L130: */

/* L140: */

/* L150: */

/* L160: */

/* L170: */

} else {

/*           Form  C := alpha*A'*B' + beta*C */

i__1 = *n;
for (j = 1; j <= i__1; ++j) {
    i__2 = *m;
    for (i__ = 1; i__ <= i__2; ++i__) {
        temp = 0.f;
        i__3 = *k;
        for (l = 1; l <= i__3; ++l) {
            temp += a[l + i__ * a_dim1] * b[j + l * b_dim1];
        }
        if (*beta == 0.f) {
            c__[i__ + j * c_dim1] = *alpha * temp;
        } else {
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
        }
    }
}

/* L180: */

/* L190: */

/* L200: */

}

return 0;

/*     End of SGEMM . */

} /* sgemm_ */

/* Subroutine */ int sgemv_(char *trans, integer *m, integer *n, real *alpha,
    real *a, integer *lda, real *x, integer *incx, real *beta, real *y,
    integer *incy)
{
/*
    Purpose:
    SGEMV performs matrix-vector multiplication with options for transposed and scaled results.

    Arguments:
    - trans: Character flag specifying transpose operation ('N', 'T', 'C').
    - m: Number of rows in matrix A.
    - n: Number of columns in matrix A.
    - alpha: Scalar multiplier for A*x or A'*x.
    - a: Matrix A of dimensions (lda, *).
    - lda: Leading dimension of A.
    - x: Vector x of length at least (1 + (n-1)*abs(incx)) when trans = 'N' or 'n' and at least (1 + (m-1)*abs(incx)) otherwise.
    - incx: Increment for the elements of x.
    - beta: Scalar multiplier for vector y.
    - y: Vector y of length at least (1 + (m-1)*abs(incy)).
    - incy: Increment for the elements of y.
*/

/* System generated locals */
integer a_dim1, a_offset, i__1, i__2;

/* Local variables */
static integer i__, j, ix, iy, jx, jy, kx, ky, info;
static real temp;
static integer lenx, leny;
extern logical lsame_(char *, char *);
extern /* Subroutine */ int xerbla_(char *, integer *);

/*
    Purpose
    =======

    SGEMV  performs one of the matrix-vector operations

       y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ==========

*/
    # 操作类型，指定矩阵乘法的变体
    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.

                TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.

                TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.

             Unchanged on exit.

    # 矩阵 A 的行数
    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    # 矩阵 A 的列数
    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    # 标量 alpha，用于乘法操作
    ALPHA  - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    # 输入矩阵 A，存储在二维数组中
    A      - REAL             array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients.
             Unchanged on exit.

    # 矩阵 A 的第一维长度，通常为行数 M
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    # 输入向量 X，根据不同的 TRANS 值确定长度要求
    X      - REAL             array of DIMENSION at least
             ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
             Before entry, the incremented array X must contain the
             vector x.
             Unchanged on exit.

    # X 向量元素之间的增量
    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    # 标量 beta，用于乘法操作
    BETA   - REAL            .
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    # 输入输出向量 Y，根据不同的 TRANS 值确定长度要求
    Y      - REAL             array of DIMENSION at least
             ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
             Before entry with BETA non-zero, the incremented array Y
             must contain the vector y. On exit, Y is overwritten by the
             updated vector y.

    # Y 向量元素之间的增量
    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================


       Test the input parameters.
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")
        ) {
        info = 1;
    } else if (*m < 0) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*lda < max(1,*m)) {
        info = 6;
    } else if (*incx == 0) {
        info = 8;
    } else if (*incy == 0) {
        info = 11;
    }
    if (info != 0) {
        xerbla_("SGEMV ", &info);
        return 0;
    }

    /* Quick return if possible. */
    if (*m == 0 || *n == 0 || *alpha == 0.f && *beta == 1.f) {
        return 0;
    }

    /*
       Set  LENX  and  LENY, the lengths of the vectors x and y, and set
       up the start points in  X  and  Y.
    */
    if (lsame_(trans, "N")) {
        lenx = *n;
        leny = *m;
    } else {
        lenx = *m;
        leny = *n;
    }
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (leny - 1) * *incy;
    }

    /*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.

       First form  y := beta*y.
    */
    if (*beta != 1.f) {
        if (*incy == 1) {
            if (*beta == 0.f) {
                for (i__ = 1; i__ <= leny; ++i__) {
                    y[i__] = 0.f;
                }
            } else {
                for (i__ = 1; i__ <= leny; ++i__) {
                    y[i__] = *beta * y[i__];
                }
            }
        } else {
            iy = ky;
            if (*beta == 0.f) {
                for (i__ = 1; i__ <= leny; ++i__) {
                    y[iy] = 0.f;
                    iy += *incy;
                }
            } else {
                for (i__ = 1; i__ <= leny; ++i__) {
                    y[iy] = *beta * y[iy];
                    iy += *incy;
                }
            }
        }
    }

    if (*alpha == 0.f) {
        return 0;
    }

    if (lsame_(trans, "N")) {
        /* Form  y := alpha*A*x + y. */
        jx = kx;
        if (*incy == 1) {
            for (j = 1; j <= *n; ++j) {
                if (x[jx] != 0.f) {
                    temp = *alpha * x[jx];
                    for (i__ = 1; i__ <= *m; ++i__) {
                        y[i__] += temp * a[i__ + j * a_dim1];
                    }
                }
                jx += *incx;
            }
        } else {
            jx = kx;
            for (j = 1; j <= *n; ++j) {
                if (x[jx] != 0.f) {
                    temp = *alpha * x[jx];
                    iy = ky;
                    for (i__ = 1; i__ <= *m; ++i__) {
                        y[iy] += temp * a[i__ + j * a_dim1];
                        iy += *incy;
                    }
                }
                jx += *incx;
            }
        }
    } else {
        /* Form  y := alpha*A'*x + y. */
        jy = ky;
        for (j = 1; j <= *n; ++j) {
            temp = 0.f;
            jx = kx;
            for (i__ = 1; i__ <= *m; ++i__) {
                temp += a[i__ + j * a_dim1] * x[jx];
                jx += *incx;
            }
            y[jy] += *alpha * temp;
            jy += *incy;
        }
    }
    # 检查增量因子是否为1，如果是则执行以下循环
    if (*incx == 1) {
        # 循环遍历列向量 x 的元素
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            # 初始化临时变量 temp 为 0
            temp = 0.f;
            # 循环遍历矩阵 a 的行数
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                # 计算矩阵 a 的元素和向量 x 的对应元素的乘积，累加到 temp 中
                temp += a[i__ + j * a_dim1] * x[i__];
    /* Loop through columns of matrix A */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        temp = 0.f;  // Initialize temporary variable for summation
        ix = kx;  // Initialize starting index of vector x
        i__2 = *m;
        /* Loop through rows of matrix A */
        for (i__ = 1; i__ <= i__2; ++i__) {
            temp += a[i__ + j * a_dim1] * x[ix];  // Perform matrix-vector multiplication
            ix += *incx;  // Increment index of vector x
/* L110: */
        }
        y[jy] += *alpha * temp;  // Update vector y with scaled summation
        jy += *incy;  // Increment index of vector y
/* L120: */
    }



/* Specialized case where matrix A is empty (M = 0 or N = 0) */
} /* sgemv_ */

/* Subroutine */ int sger_(integer *m, integer *n, real *alpha, real *x,
    integer *incx, real *y, integer *incy, real *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, jy, kx, info;
    static real temp;
    extern /* Subroutine */ int xerbla_(char *, integer *);

    /*
        Purpose
        =======

        SGER   performs the rank 1 operation

           A := alpha*x*y' + A,

        where alpha is a scalar, x is an m element vector, y is an n element
        vector and A is an m by n matrix.

        Arguments
        ==========

        M      - INTEGER.
                 On entry, M specifies the number of rows of the matrix A.
                 M must be at least zero.
                 Unchanged on exit.

        N      - INTEGER.
                 On entry, N specifies the number of columns of the matrix A.
                 N must be at least zero.
                 Unchanged on exit.

        ALPHA  - REAL            .
                 On entry, ALPHA specifies the scalar alpha.
                 Unchanged on exit.

        X      - REAL             array of dimension at least
                 ( 1 + ( m - 1 )*abs( INCX ) ).
                 Before entry, the incremented array X must contain the m
                 element vector x.
                 Unchanged on exit.

        INCX   - INTEGER.
                 On entry, INCX specifies the increment for the elements of
                 X. INCX must not be zero.
                 Unchanged on exit.

        Y      - REAL             array of dimension at least
                 ( 1 + ( n - 1 )*abs( INCY ) ).
                 Before entry, the incremented array Y must contain the n
                 element vector y.
                 Unchanged on exit.

        INCY   - INTEGER.
                 On entry, INCY specifies the increment for the elements of
                 Y. INCY must not be zero.
                 Unchanged on exit.

        A      - REAL             array of DIMENSION ( LDA, n ).
                 Before entry, the leading m by n part of the array A must
                 contain the matrix of coefficients. On exit, A is
                 overwritten by the updated matrix.

        LDA    - INTEGER.
                 On entry, LDA specifies the first dimension of A as declared
                 in the calling (sub) program. LDA must be at least
                 max( 1, m ).
                 Unchanged on exit.

        Further Details
        ===============

        Level 2 Blas routine.
    */
    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.
    /* Parameter adjustments */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (*m < 0) {
        info = 1;  // 设置 info 为 1，表示 m 参数小于 0
    } else if (*n < 0) {
        info = 2;  // 设置 info 为 2，表示 n 参数小于 0
    } else if (*incx == 0) {
        info = 5;  // 设置 info 为 5，表示 incx 参数为 0
    } else if (*incy == 0) {
        info = 7;  // 设置 info 为 7，表示 incy 参数为 0
    } else if (*lda < max(1,*m)) {
        info = 9;  // 设置 info 为 9，表示 lda 参数小于 max(1, m)
    }
    if (info != 0) {
        xerbla_("SGER  ", &info);  // 调用错误处理函数 xerbla，传递错误信息
        return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || *alpha == 0.f) {
        return 0;  // 如果 m 或 n 为 0，或者 alpha 为 0，则直接返回
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (*incy > 0) {
        jy = 1;  // 如果 incy 大于 0，则从 1 开始
    } else {
        jy = 1 - (*n - 1) * *incy;  // 否则，根据 n 和 incy 计算起始位置 jy
    }
    if (*incx == 1) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 循环遍历列索引 j
            if (y[jy] != 0.f) {  // 如果 y[jy] 不为 0
                temp = *alpha * y[jy];  // 计算临时变量 temp
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {  // 循环遍历行索引 i
                    a[i__ + j * a_dim1] += x[i__] * temp;  // 更新矩阵 A 的元素
/* L10: */          // 循环体结束
                }
            }
            jy += *incy;  // 更新 jy 的值
/* L20: */          // 循环体结束
        }
    } else {
        if (*incx > 0) {
            kx = 1;  // 如果 incx 大于 0，则从 1 开始
        } else {
            kx = 1 - (*m - 1) * *incx;  // 否则，根据 m 和 incx 计算起始位置 kx
        }
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 循环遍历列索引 j
            if (y[jy] != 0.f) {  // 如果 y[jy] 不为 0
                temp = *alpha * y[jy];  // 计算临时变量 temp
                ix = kx;  // 将 ix 初始化为 kx
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {  // 循环遍历行索引 i
                    a[i__ + j * a_dim1] += x[ix] * temp;  // 更新矩阵 A 的元素
                    ix += *incx;  // 更新 ix 的值
/* L30: */          // 循环体结束
                }
            }
            jy += *incy;  // 更新 jy 的值
/* L40: */          // 循环体结束
        }
    }

    return 0;

/*     End of SGER  . */

} /* sger_ */



doublereal snrm2_(integer *n, real *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    real ret_val, r__1;

    /* Local variables */
    static integer ix;
    static real ssq, norm, scale, absxi;


/*
    Purpose
    =======

    SNRM2 returns the euclidean norm of a vector via the function
    name, so that

       SNRM2 := sqrt( x'*x ).

    Further Details
    ===============

    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to SLASSQ.
       Sven Hammarling, Nag Ltd.

    =====================================================================
*/

    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n < 1 || *incx < 1) {
        norm = 0.f;  // 如果 n 小于 1 或者 incx 小于 1，返回 0
    } else if (*n == 1) {
        norm = dabs(x[1]);  // 如果 n 等于 1，返回 x[1] 的绝对值
    } else {
        scale = 0.f;  // 初始化 scale 为 0
        ssq = 1.f;  // 初始化 ssq 为 1
/*
          The following loop is equivalent to this call to the LAPACK
          auxiliary routine:
          CALL SLASSQ( N, X, INCX, SCALE, SSQ )
*/

        i__1 = (*n - 1) * *incx + 1;
        i__2 = *incx;
        for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {  // 循环遍历向量 x
            if (x[ix] != 0.f) {  // 如果 x[ix] 不为 0
                absxi = (r__1 = x[ix], dabs(r__1));  // 计算 x[ix] 的绝对值
                if (scale < absxi) {
                    /* Computing 2nd power */
                    r__1 = scale / absxi;
                    ssq = ssq * (r__1 * r__1) + 1.f;  // 更新 ssq
                    scale = absxi;  // 更新 scale
                } else {
                    /* Computing 2nd power */
                    r__1 = absxi / scale;
                    ssq += r__1 * r__1;  // 更新 ssq
                }
            }
/* L50: */      // 循环体结束
        }
        norm = scale * sqrt(ssq);  // 计算向量的欧几里得范数
    }

    return norm;  // 返回计算得到的范数值

} /* snrm2_ */
/* Computing 2nd power */
            r__1 = absxi / scale;
            ssq += r__1 * r__1;
        }
        }
/* L10: */
    }
    norm = scale * sqrt(ssq);
    }

    ret_val = norm;
    return ret_val;

/*     End of SNRM2. */

} /* snrm2_ */

/* Subroutine */ int srot_(integer *n, real *sx, integer *incx, real *sy,
    integer *incy, real *c__, real *s)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, ix, iy;
    static real stemp;


/*
    Purpose
    =======

       applies a plane rotation.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --sy;
    --sx;

    /* Function Body */
    if (*n <= 0) {
    return 0;
    }
    if (*incx == 1 && *incy == 1) {
    goto L20;
    }

/*
         code for unequal increments or equal increments not equal
           to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
    ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
    iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    stemp = *c__ * sx[ix] + *s * sy[iy];
    sy[iy] = *c__ * sy[iy] - *s * sx[ix];
    sx[ix] = stemp;
    ix += *incx;
    iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    stemp = *c__ * sx[i__] + *s * sy[i__];
    sy[i__] = *c__ * sy[i__] - *s * sx[i__];
    sx[i__] = stemp;
/* L30: */
    }
    return 0;
} /* srot_ */

/* Subroutine */ int sscal_(integer *n, real *sa, real *sx, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__, m, mp1, nincx;


/*
    Purpose
    =======

       scales a vector by a constant.
       uses unrolled loops for increment equal to 1.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --sx;

    /* Function Body */
    if (*n <= 0 || *incx <= 0) {
    return 0;
    }
    if (*incx == 1) {
    goto L20;
    }

/*        code for increment not equal to 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
    sx[i__] = *sa * sx[i__];
/* L10: */
    }
    return 0;

/*
          code for increment equal to 1


          clean-up loop
*/

L20:
    m = *n % 5;
    if (m == 0) {
    goto L40;
    }
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
    sx[i__] = *sa * sx[i__];
/* L30: */
    }
    if (*n < 5) {
    return 0;
    }
L40:
    mp1 = m + 1;
    i__2 = *n;
    // 循环迭代，从索引 mp1 开始，每次增加 5
    for (i__ = mp1; i__ <= i__2; i__ += 5) {
        // 将数组 sx 中索引为 i__ 的元素乘以指针 sa 指向的值，然后赋值给 sx[i__]
        sx[i__] = *sa * sx[i__];
        // 将数组 sx 中索引为 i__+1 的元素乘以指针 sa 指向的值，然后赋值给 sx[i__+1]
        sx[i__ + 1] = *sa * sx[i__ + 1];
        // 将数组 sx 中索引为 i__+2 的元素乘以指针 sa 指向的值，然后赋值给 sx[i__+2]
        sx[i__ + 2] = *sa * sx[i__ + 2];
        // 将数组 sx 中索引为 i__+3 的元素乘以指针 sa 指向的值，然后赋值给 sx[i__+3]
        sx[i__ + 3] = *sa * sx[i__ + 3];
        // 将数组 sx 中索引为 i__+4 的元素乘以指针 sa 指向的值，然后赋值给 sx[i__+4]
        sx[i__ + 4] = *sa * sx[i__ + 4];
/* L50: */
    }
    return 0;
} /* sscal_ */

/* Subroutine */ int sswap_(integer *n, real *sx, integer *incx, real *sy,
    integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;
    static real stemp;


/*
    Purpose
    =======

       interchanges two vectors.
       uses unrolled loops for increments equal to 1.

    Further Details
    ===============

       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --sy;
    --sx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

/*
         code for unequal increments or equal increments not equal
           to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        stemp = sx[ix];
        sx[ix] = sy[iy];
        sy[iy] = stemp;
        ix += *incx;
        iy += *incy;
/* L10: */
    }
    return 0;

/*
         code for both increments equal to 1


         clean-up loop
*/

L20:
    m = *n % 3;
    if (m == 0) {
        goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
        stemp = sx[i__];
        sx[i__] = sy[i__];
        sy[i__] = stemp;
/* L30: */
    }
    if (*n < 3) {
        return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 3) {
        stemp = sx[i__];
        sx[i__] = sy[i__];
        sy[i__] = stemp;
        stemp = sx[i__ + 1];
        sx[i__ + 1] = sy[i__ + 1];
        sy[i__ + 1] = stemp;
        stemp = sx[i__ + 2];
        sx[i__ + 2] = sy[i__ + 2];
        sy[i__ + 2] = stemp;
/* L50: */
    }
    return 0;
} /* sswap_ */

/* Subroutine */ int ssymv_(char *uplo, integer *n, real *alpha, real *a,
    integer *lda, real *x, integer *incx, real *beta, real *y, integer *
    incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static real temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    SSYMV  performs the matrix-vector  operation

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric matrix.

    Arguments
    ==========

    UPLO    - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N       - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA   - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A       - REAL             array of DIMENSION ( LDA, n ).
             Before entry, the leading n by n part of the array A must
             contain the symmetric matrix A. On exit, A is overwritten
             by the updated matrix.

    LDA     - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.

    X       - REAL             array of DIMENSION at least
             ( 1 + ( n - 1 )*abs( INCX ) ) when INCX .GE. 0 and at least
             ( 1 + ( n - 1 )*abs( INCX ) ) when INCX .LT. 0.
             Before entry, the incremented array X must contain the
             n element vector x. On exit, X is unchanged.

    INCX    - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA    - REAL            .
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.

    Y       - REAL             array of DIMENSION at least
             ( 1 + ( n - 1 )*abs( INCY ) ) when INCY .GE. 0 and at least
             ( 1 + ( n - 1 )*abs( INCY ) ) when INCY .LT. 0.
             Before entry, the incremented array Y must contain the
             n element vector y. On exit, Y is overwritten by the updated
             vector y.

    INCY    - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.


    Level 2 Blas routine.

    =====================================================================
*/
    UPLO   - CHARACTER*1.
             ! 入口参数，指定数组 A 的上三角部分或下三角部分的引用方式：
             ! 'U' 或 'u'：只引用数组 A 的上三角部分。
             ! 'L' 或 'l'：只引用数组 A 的下三角部分。
             ! 函数结束后保持不变。

    N      - INTEGER.
             ! 入口参数，指定矩阵 A 的阶数 N。
             ! N 必须至少为零。
             ! 函数结束后保持不变。

    ALPHA  - REAL            .
             ! 入口参数，指定标量 alpha。
             ! 函数结束后保持不变。

    A      - REAL             array of DIMENSION ( LDA, n ).
             ! 在入口时，如果 UPLO = 'U' 或 'u'，数组 A 的前 n 行 n 列应包含对称矩阵的上三角部分，
             ! 并且数组 A 的严格下三角部分不被引用。
             ! 如果 UPLO = 'L' 或 'l'，数组 A 的前 n 行 n 列应包含对称矩阵的下三角部分，
             ! 并且数组 A 的严格上三角部分不被引用。
             ! 函数结束后保持不变。

    LDA    - INTEGER.
             ! 入口参数，指定数组 A 在调用程序中声明时的第一维度。
             ! LDA 必须至少为 max(1, n)。
             ! 函数结束后保持不变。

    X      - REAL             array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             ! 在入口时，数组 X 是一个长度为 n 的向量 x。
             ! 函数结束后保持不变。

    INCX   - INTEGER.
             ! 入口参数，指定数组 X 中元素的增量。
             ! INCX 不能为零。
             ! 函数结束后保持不变。

    BETA   - REAL            .
             ! 入口参数，指定标量 beta。
             ! 当 BETA 为零时，输入时 Y 不需要设置。
             ! 函数结束后保持不变。

    Y      - REAL             array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             ! 在入口时，数组 Y 是一个长度为 n 的向量 y。
             ! 函数结束后，数组 Y 被更新为更新后的向量 y。

    INCY   - INTEGER.
             ! 入口参数，指定数组 Y 中元素的增量。
             ! INCY 不能为零。
             ! 函数结束后保持不变。

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.
    =====================================================================
    # 以下是一个注释行，用来标记一条分隔线或者重要的部分

       Test the input parameters.
    # 测试输入参数
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (*n < 0) {
        info = 2;
    } else if (*lda < max(1,*n)) {
        info = 5;
    } else if (*incx == 0) {
        info = 7;
    } else if (*incy == 0) {
        info = 10;
    }
    if (info != 0) {
        xerbla_("SSYMV ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0.f && *beta == 1.f) {
        return 0;
    }

/*     Set up the start points in  X  and  Y. */

    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (*n - 1) * *incx;
    }
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (*n - 1) * *incy;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.

       First form  y := beta*y.
*/

    if (*beta != 1.f) {
        if (*incy == 1) {
            if (*beta == 0.f) {
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = 0.f;
                }
            } else {
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = *beta * y[i__];
                }
            }
        } else {
            iy = ky;
            if (*beta == 0.f) {
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = 0.f;
                    iy += *incy;
                }
            } else {
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = *beta * y[iy];
                    iy += *incy;
                }
            }
        }
    }
    if (*alpha == 0.f) {
        return 0;
    }
    if (lsame_(uplo, "U")) {

/*        Form  y  when A is stored in upper triangle. */

        if (*incx == 1 && *incy == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp1 = *alpha * x[j];
                temp2 = 0.f;
                i__2 = j - 1;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    y[i__] += temp1 * a[i__ + j * a_dim1];
                    temp2 += a[i__ + j * a_dim1] * x[i__];
                }
                y[j] = y[j] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
            }
        } else {
            jx = kx;
            jy = ky;
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp1 = *alpha * x[jx];
                temp2 = 0.f;
                ix = kx;
                iy = ky;
                i__2 = j - 1;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    y[iy] += temp1 * a[i__ + j * a_dim1];
                    temp2 += a[i__ + j * a_dim1] * x[ix];
                    ix += *incx;
                    iy += *incy;
                }
                y[jy] = y[jy] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
                jx += *incx;
                jy += *incy;
            }
        }
    } else {

/*        Form  y  when A is stored in lower triangle. */
    # 检查增量因子是否为1，以优化内存访问
    if (*incx == 1 && *incy == 1) {
        # 对矩阵的每一列进行操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            # 计算临时变量 temp1
            temp1 = *alpha * x[j];
            # 初始化临时变量 temp2
            temp2 = 0.f;
            # 更新向量 y 中的元素
            y[j] += temp1 * a[j + j * a_dim1];
            # 遍历当前列的剩余元素
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                # 更新向量 y 中的元素
                y[i__] += temp1 * a[i__ + j * a_dim1];
                # 计算矩阵元素与向量元素的乘积并累加到 temp2
                temp2 += a[i__ + j * a_dim1] * x[i__];
/* L90: */
        }
        // 对 y[j] 进行累加：y[j] = y[j] + alpha * temp2
        y[j] += *alpha * temp2;
/* L100: */
        }
    } else {
        // 初始化 jx 和 jy 为起始索引 kx 和 ky
        jx = kx;
        jy = ky;
        // 循环处理每个列 j
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 计算 alpha * x[jx] 并赋值给 temp1
        temp1 = *alpha * x[jx];
        // 初始化 temp2 为 0
        temp2 = 0.f;
        // 对 y[jy] 进行累加：y[jy] = y[jy] + temp1 * a[j + j * a_dim1]
        y[jy] += temp1 * a[j + j * a_dim1];
        // 初始化 ix 和 iy 为 jx 和 jy
        ix = jx;
        iy = jy;
        // 循环处理每个行 i，从 j+1 到 n
        i__2 = *n;
        for (i__ = j + 1; i__ <= i__2; ++i__) {
            // 更新 ix 和 iy
            ix += *incx;
            iy += *incy;
            // 对 y[iy] 进行累加：y[iy] = y[iy] + temp1 * a[i__ + j * a_dim1]
            y[iy] += temp1 * a[i__ + j * a_dim1];
            // 累加 a[i__ + j * a_dim1] * x[ix] 到 temp2
            temp2 += a[i__ + j * a_dim1] * x[ix];
/* L110: */
        }
        // 对 y[jy] 进行累加：y[jy] = y[jy] + alpha * temp2
        y[jy] += *alpha * temp2;
        // 更新 jx 和 jy
        jx += *incx;
        jy += *incy;
/* L120: */
        }
    }
    }

    return 0;

/*     End of SSYMV . */

} /* ssymv_ */

/* Subroutine */ int ssyr2_(char *uplo, integer *n, real *alpha, real *x,
    integer *incx, real *y, integer *incy, real *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static real temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    # REAL类型的数组A，维度为(LDA, n)。
    # 如果UPLO='U'或'U'，则输入时数组A的前n行n列应包含对称矩阵的上三角部分，
    # A的严格下三角部分不被引用。退出时，数组A的上三角部分被更新后的矩阵的上三角部分覆盖。
    # 如果UPLO='L'或'l'，则输入时数组A的前n行n列应包含对称矩阵的下三角部分，
    # A的严格上三角部分不被引用。退出时，数组A的下三角部分被更新后的矩阵的下三角部分覆盖。
    A      - REAL             array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of A is not referenced. On exit, the
             upper triangular part of the array A is overwritten by the
             upper triangular part of the updated matrix.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of A is not referenced. On exit, the
             lower triangular part of the array A is overwritten by the
             lower triangular part of the updated matrix.

    # 整数LDA。
    # 输入时，LDA指定数组A在调用程序中声明的第一个维度。
    # LDA必须至少为max(1, n)。
    # 退出时保持不变。
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.

    # 进一步细节
    # Level 2 Blas例程。
    # 编写日期：1986年10月22日。
    # Jack Dongarra，Argonne National Lab。
    # Jeremy Du Croz，Nag Central Office。
    # Sven Hammarling，Nag Central Office。
    # Richard Hanson，Sandia National Labs。
    Further Details
    ===============

    # 测试输入参数。
    # =====================================================================
    Test the input parameters.
    /* Parameter adjustments */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (*n < 0) {
        info = 2;
    } else if (*incx == 0) {
        info = 5;
    } else if (*incy == 0) {
        info = 7;
    } else if (*lda < max(1,*n)) {
        info = 9;
    }
    if (info != 0) {
        xerbla_("SSYR2 ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0.f) {
        return 0;
    }

/*
       Set up the start points in X and Y if the increments are not both
       unity.
*/

    if (*incx != 1 || *incy != 1) {
        if (*incx > 0) {
            kx = 1;
        } else {
            kx = 1 - (*n - 1) * *incx;
        }
        if (*incy > 0) {
            ky = 1;
        } else {
            ky = 1 - (*n - 1) * *incy;
        }
        jx = kx;
        jy = ky;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.
*/

    if (lsame_(uplo, "U")) {

/*        Form  A  when A is stored in the upper triangle. */

        if (*incx == 1 && *incy == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[j] != 0.f || y[j] != 0.f) {
                    temp1 = *alpha * y[j];
                    temp2 = *alpha * x[j];
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] * temp1 + y[i__] * temp2;
                        /* Perform the rank-2 update of A */
                    }
                }
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.f || y[jy] != 0.f) {
                    temp1 = *alpha * y[jy];
                    temp2 = *alpha * x[jx];
                    ix = kx;
                    iy = ky;
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] * temp1 + y[iy] * temp2;
                        ix += *incx;
                        iy += *incy;
                        /* Perform the rank-2 update of A */
                    }
                }
                jx += *incx;
                jy += *incy;
            }
        }
    } else {

/*        Form  A  when A is stored in the lower triangle. */

        if (*incx == 1 && *incy == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[j] != 0.f || y[j] != 0.f) {
                    temp1 = *alpha * y[j];
                    temp2 = *alpha * x[j];
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] * temp1 + y[i__] * temp2;
                        /* Perform the rank-2 update of A */
                    }
                }
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.f || y[jy] != 0.f) {
                    temp1 = *alpha * y[jy];
                    temp2 = *alpha * x[jx];
                    ix = kx;
                    iy = ky;
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] * temp1 + y[iy] * temp2;
                        ix += *incx;
                        iy += *incy;
                        /* Perform the rank-2 update of A */
                    }
                }
                jx += *incx;
                jy += *incy;
            }
        }
    }
    } else {
        // 如果条件不成立，执行以下循环
        i__1 = *n;  // 将指针指向的值赋给 i__1
        for (j = 1; j <= i__1; ++j) {  // 循环，j 从 1 到 i__1
            if (x[jx] != 0.f || y[jy] != 0.f) {  // 如果 x[jx] 或者 y[jy] 不等于 0
                temp1 = *alpha * y[jy];  // 计算 temp1
                temp2 = *alpha * x[jx];  // 计算 temp2
                ix = jx;  // 设置 ix 初始值为 jx
                iy = jy;  // 设置 iy 初始值为 jy
                i__2 = *n;  // 将指针指向的值赋给 i__2
                for (i__ = j; i__ <= i__2; ++i__) {  // 循环，i__ 从 j 到 i__2
                    a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] *
                        temp1 + y[iy] * temp2;  // 更新 a[i__ + j * a_dim1]
                    ix += *incx;  // 更新 ix
                    iy += *incy;  // 更新 iy
/* L70: */
            }
        }
        jx += *incx;
        jy += *incy;
/* L80: */
        }
    }
    }

    return 0;

/*     End of SSYR2 . */

} /* ssyr2_ */

/* Subroutine */ int ssyr2k_(char *uplo, char *trans, integer *n, integer *k,
    real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta,
     real *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3;

    /* Local variables */
    static integer i__, j, l, info;
    static real temp1, temp2;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*
    Purpose
    =======

    SSYR2K  performs one of the symmetric rank 2k operations

       C := alpha*A*B' + alpha*B*A' + beta*C,

    or

       C := alpha*A'*B + alpha*B'*A + beta*C,

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
    and  A and B  are  n by k  matrices  in the  first  case  and  k by n
    matrices in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of  the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   C := alpha*A*B' + alpha*B*A' +
                                          beta*C.

                TRANS = 'T' or 't'   C := alpha*A'*B + alpha*B'*A +
                                          beta*C.

                TRANS = 'C' or 'c'   C := alpha*A'*B + alpha*B'*A +
                                          beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns  of  the  matrices  A and B,  and on  entry  with
             TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
             of rows of the matrices  A and B.  K must be at least  zero.
             Unchanged on exit.

    ALPHA  - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.
*/
    A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.


    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.


    B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  k by n  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.


    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDB must be at least  max( 1, n ), otherwise  LDB must
             be at least  max( 1, k ).
             Unchanged on exit.


    BETA   - REAL            .
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.


    C      - REAL             array of DIMENSION ( LDC, n ).
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  symmetric matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  symmetric matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.


    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, n ).
             Unchanged on exit.


    Further Details
    ===============

    Level 3 Blas routine.
    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.
    /* Parameter adjustments */
    // 设置矩阵A的维度
    a_dim1 = *lda;
    // 计算数组A的偏移量
    a_offset = 1 + a_dim1;
    // 调整指针a以便于正确访问数组A的元素
    a -= a_offset;
    // 设置矩阵B的维度
    b_dim1 = *ldb;
    // 计算数组B的偏移量
    b_offset = 1 + b_dim1;
    // 调整指针b以便于正确访问数组B的元素
    b -= b_offset;
    // 设置矩阵C的维度
    c_dim1 = *ldc;
    // 计算数组C的偏移量
    c_offset = 1 + c_dim1;
    // 调整指针c以便于正确访问数组C的元素
    c__ -= c_offset;

    /* Function Body */
    // 根据转置参数决定矩阵A的行数
    if (lsame_(trans, "N")) {
        nrowa = *n;
    } else {
        nrowa = *k;
    }
    // 判断上三角矩阵的类型（是上三角还是下三角）
    upper = lsame_(uplo, "U");

    // 初始化错误信息码
    info = 0;
    // 检查上三角参数的合法性
    if (! upper && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
        "T") && ! lsame_(trans, "C")) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*k < 0) {
        info = 4;
    } else if (*lda < max(1,nrowa)) {
        info = 7;
    } else if (*ldb < max(1,nrowa)) {
        info = 9;
    } else if (*ldc < max(1,*n)) {
        info = 12;
    }
    // 如果存在错误信息，调用错误处理程序并返回
    if (info != 0) {
        xerbla_("SSYR2K", &info);
        return 0;
    }

    /* Quick return if possible. */
    // 如果n为0，或者alpha为0且k为0且beta为1，直接返回
    if (*n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {
        return 0;
    }

    /* And when alpha.eq.zero. */
    // 当alpha等于0时的情况
    if (*alpha == 0.f) {
        // 如果是上三角矩阵
        if (upper) {
            // 如果beta为0，将C矩阵的所有元素置为0
            if (*beta == 0.f) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                }
            } else {
                // 否则将C矩阵的每个元素乘以beta
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        } else {
            // 如果是下三角矩阵
            if (*beta == 0.f) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                }
            } else {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        }
        return 0;
    }

    /* Start the operations. */
    // 开始计算

    // 如果不是转置操作
    if (lsame_(trans, "N")) {

        /* Form  C := alpha*A*B' + alpha*B*A' + C. */

        // 如果是上三角矩阵
        if (upper) {
            // 遍历C的每列
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                // 如果beta为0，将C的当前列全部置为0
                if (*beta == 0.f) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                } else if (*beta != 1.f) {
                    // 否则将C的当前列乘以beta
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        } else {
            // 如果是下三角矩阵
            if (*beta == 0.f) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                }
            } else {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        }
    }
    return 0;
}


这段代码是一个用于计算对称矩阵乘积的例程，注释解释了每行代码的作用，包括参数调整、错误检查、矩阵计算的特定情况处理等。
/* L100: */
/* 开始一个循环，遍历矩阵 C 的列 */
            }
        }
        /* 设置内循环的上限为 k */
        i__2 = *k;
        /* 开始一个循环，遍历矩阵 C 的列 */
        for (l = 1; l <= i__2; ++l) {
            /* 检查矩阵 A 和 B 中的元素是否为零 */
            if (a[j + l * a_dim1] != 0.f || b[j + l * b_dim1] != 0.f)
                {
            /* 计算临时变量 temp1 和 temp2 */
            temp1 = *alpha * b[j + l * b_dim1];
            temp2 = *alpha * a[j + l * a_dim1];
            /* 设置内循环的上限为 j */
            i__3 = j;
            /* 开始一个循环，遍历矩阵 C 的行 */
            for (i__ = 1; i__ <= i__3; ++i__) {
                /* 更新矩阵 C 的元素 */
                c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
                    i__ + l * a_dim1] * temp1 + b[i__ + l *
                    b_dim1] * temp2;
/* L110: */
            }
            }
/* L120: */
        }
/* L130: */
        }
    } else {
        /* 设置外循环的上限为 n */
        i__1 = *n;
        /* 开始一个循环，遍历矩阵 C 的列 */
        for (j = 1; j <= i__1; ++j) {
        /* 检查 beta 是否为零 */
        if (*beta == 0.f) {
            /* 设置内循环的上限为 n */
            i__2 = *n;
            /* 开始一个循环，更新矩阵 C 的元素 */
            for (i__ = j; i__ <= i__2; ++i__) {
            c__[i__ + j * c_dim1] = 0.f;
/* L140: */
            }
        } else if (*beta != 1.f) {
            /* 设置内循环的上限为 n */
            i__2 = *n;
            /* 开始一个循环，更新矩阵 C 的元素 */
            for (i__ = j; i__ <= i__2; ++i__) {
            c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L150: */
            }
        }
        /* 设置内循环的上限为 k */
        i__2 = *k;
        /* 开始一个循环，遍历矩阵 A 和 B 的列 */
        for (l = 1; l <= i__2; ++l) {
            /* 检查矩阵 A 和 B 中的元素是否为零 */
            if (a[j + l * a_dim1] != 0.f || b[j + l * b_dim1] != 0.f)
                {
            /* 计算临时变量 temp1 和 temp2 */
            temp1 = *alpha * b[j + l * b_dim1];
            temp2 = *alpha * a[j + l * a_dim1];
            /* 设置内循环的上限为 n */
            i__3 = *n;
            /* 开始一个循环，更新矩阵 C 的元素 */
            for (i__ = j; i__ <= i__3; ++i__) {
                /* 更新矩阵 C 的元素 */
                c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
                    i__ + l * a_dim1] * temp1 + b[i__ + l *
                    b_dim1] * temp2;
/* L160: */
            }
            }
/* L170: */
        }
/* L180: */
        }
    }
    } else {

/*        Form  C := alpha*A'*B + alpha*B'*A + C. */

    /* 设置外循环的上限为 n */
    if (upper) {
        /* 开始一个循环，遍历矩阵 C 的列 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        /* 设置内循环的上限为 j */
        i__2 = j;
        /* 开始一个循环，遍历矩阵 C 的行 */
        for (i__ = 1; i__ <= i__2; ++i__) {
            /* 初始化临时变量 temp1 和 temp2 */
            temp1 = 0.f;
            temp2 = 0.f;
            /* 设置内循环的上限为 k */
            i__3 = *k;
            /* 开始一个循环，计算 alpha*A'*B 和 alpha*B'*A 的和 */
            for (l = 1; l <= i__3; ++l) {
            temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
            temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
/* L190: */
            }
            /* 检查 beta 是否为零 */
            if (*beta == 0.f) {
            /* 更新矩阵 C 的元素 */
            c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
                temp2;
            } else {
            /* 更新矩阵 C 的元素 */
            c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
                + *alpha * temp1 + *alpha * temp2;
            }
/* L200: */
        }
/* L210: */
        }
    } else {
        /* 设置外循环的上限为 n */
        i__1 = *n;
        /* 开始一个循环，遍历矩阵 C 的列 */
        for (j = 1; j <= i__1; ++j) {
        /* 设置内循环的上限为 n */
        i__2 = *n;
        /* 开始一个循环，遍历矩阵 C 的行 */
        for (i__ = j; i__ <= i__2; ++i__) {
            /* 初始化临时变量 temp1 和 temp2 */
            temp1 = 0.f;
            temp2 = 0.f;
            /* 设置内循环的上限为 k */
            i__3 = *k;
            /* 开始一个循环，计算 alpha*A'*B 和 alpha*B'*A 的和 */
            for (l = 1; l <= i__3; ++l) {
            temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
            temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
/* L220: */
            }
            /* 检查 beta 是否为零 */
            if (*beta == 0.f) {
            /* 更新矩阵 C 的元素 */
            c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
                temp2;
            } else {
            /* 更新矩阵 C 的元素 */
            c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
                + *alpha * temp1 + *alpha * temp2;
            }
/* L230: */
        }
/* L240: */
        }
    }
    }
/* L220: */
            }
            // 如果 beta 等于 0，直接用 alpha*temp1 + alpha*temp2 更新 C[i][j] 的值
            if (*beta == 0.f) {
                c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
                    temp2;
            } else {
                // 否则，用 beta*C[i][j] + alpha*temp1 + alpha*temp2 更新 C[i][j] 的值
                c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
                    + *alpha * temp1 + *alpha * temp2;
            }
/* L230: */
        }
/* L240: */
        }
    }
    }

    // 返回 0，表示函数执行完毕
    return 0;

/*     End of SSYR2K. */

} /* ssyr2k_ */

/* Subroutine */ int ssyrk_(char *uplo, char *trans, integer *n, integer *k,
    real *alpha, real *a, integer *lda, real *beta, real *c__, integer *
    ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, l, info;
    static real temp;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    SSYRK  performs one of the symmetric rank k operations

       C := alpha*A*A' + beta*C,

    or

       C := alpha*A'*A + beta*C,

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
    and  A  is an  n by k  matrix in the first case and a  k by n  matrix
    in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of  the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C.

                TRANS = 'T' or 't'   C := alpha*A'*A + beta*C.

                TRANS = 'C' or 'c'   C := alpha*A'*A + beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns   of  the   matrix   A,   and  on   entry   with
             TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
             of rows of the matrix  A.  K must be at least zero.
             Unchanged on exit.

    ALPHA  - REAL            .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.
*/
    A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.

    BETA   - REAL            .
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.

    C      - REAL             array of DIMENSION ( LDC, n ).
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  symmetric matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  symmetric matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, n ).
             Unchanged on exit.

    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

    =====================================================================


       Test the input parameters.
    /* Parameter adjustments */
    // 设置矩阵 A 的维度参数
    a_dim1 = *lda;
    // 计算矩阵 A 在存储数组中的偏移量
    a_offset = 1 + a_dim1;
    // 调整指向矩阵 A 的指针，使其指向正确的位置
    a -= a_offset;
    // 设置矩阵 C 的维度参数
    c_dim1 = *ldc;
    // 计算矩阵 C 在存储数组中的偏移量
    c_offset = 1 + c_dim1;
    // 调整指向矩阵 C 的指针，使其指向正确的位置
    c__ -= c_offset;

    /* Function Body */
    // 根据 trans 参数确定 nrowa 的值
    if (lsame_(trans, "N")) {
        nrowa = *n;
    } else {
        nrowa = *k;
    }
    // 判断上三角矩阵是否为要求
    upper = lsame_(uplo, "U");

    // 检查输入参数是否合法，设置错误信息码
    info = 0;
    if (! upper && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
        "T") && ! lsame_(trans, "C")) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*k < 0) {
        info = 4;
    } else if (*lda < max(1,nrowa)) {
        info = 7;
    } else if (*ldc < max(1,*n)) {
        info = 10;
    }
    // 如果存在错误信息，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("SSYRK ", &info);
        return 0;
    }

/*     Quick return if possible. */

    // 如果 n=0 或 alpha=0 且 beta=1，则直接返回
    if (*n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {
        return 0;
    }

/*     And when  alpha.eq.zero. */

    // 如果 alpha=0，则根据 upper 判断需要执行的操作
    if (*alpha == 0.f) {
        if (upper) {
            // 如果 beta=0，则将矩阵 C 的元素置零
            if (*beta == 0.f) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                }
            } else {
                // 否则，对矩阵 C 的元素进行 beta 倍的缩放
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        } else {
            if (*beta == 0.f) {
                // 如果 beta=0，则将矩阵 C 的元素置零（下三角部分）
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                }
            } else {
                // 否则，对矩阵 C 的元素进行 beta 倍的缩放（下三角部分）
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
            }
        }
        return 0;
    }

/*     Start the operations. */

    // 如果 trans="N"，执行 C := alpha*A*A' + beta*C 的操作
    if (lsame_(trans, "N")) {

/*        Form  C := alpha*A*A' + beta*C. */

        // 如果 upper=true，处理上三角部分
        if (upper) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (*beta == 0.f) {
                    // 如果 beta=0，则将矩阵 C 的元素置零（上三角部分）
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = 0.f;
                    }
                } else if (*beta != 1.f) {
                    // 否则，对矩阵 C 的元素进行 beta 倍的缩放（上三角部分）
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
                    }
                }
                // 计算 C 的更新值：C := C + alpha * A(:,j) * A(:,j)'
                i__2 = *k;
                for (l = 1; l <= i__2; ++l) {
                    if (a[j + l * a_dim1] != 0.f) {
                        temp = *alpha * a[j + l * a_dim1];
                        i__3 = j;
                        for (i__ = 1; i__ <= i__3; ++i__) {
                            c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
                        }
                    }
                }
            }
        }
    }
    } else {
        // 如果beta不等于0，则执行以下操作
        i__1 = *n;
        // 循环遍历j从1到n
        for (j = 1; j <= i__1; ++j) {
            // 如果beta等于0，则执行以下操作
            if (*beta == 0.f) {
                // 循环遍历i从j到n
                i__2 = *n;
                for (i__ = j; i__ <= i__2; ++i__) {
                    // 将数组c__的第(i__, j)个元素设为0
                    c__[i__ + j * c_dim1] = 0.f;
/* L140: */
            }
        } else if (*beta != 1.f) {
            i__2 = *n;
            // 循环：从 j 到 n，对每一列 j 进行操作
            for (i__ = j; i__ <= i__2; ++i__) {
            // 更新 C 矩阵的元素为 beta 值乘以原值
            c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L150: */
            }
        }
        // 循环：从 1 到 k，对每一列 l 进行操作
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            // 如果 A 的元素不为零
            if (a[j + l * a_dim1] != 0.f) {
            // 计算 temp 为 alpha 乘以 A 的元素值
            temp = *alpha * a[j + l * a_dim1];
            // 循环：从 j 到 n，对每一列 i 进行操作
            i__3 = *n;
            for (i__ = j; i__ <= i__3; ++i__) {
                // 更新 C 矩阵的元素为 temp 乘以 A 的对应元素加到原值上
                c__[i__ + j * c_dim1] += temp * a[i__ + l *
                    a_dim1];
/* L160: */
            }
            }
/* L170: */
        }
/* L180: */
        }
    }
    } else {

/*        Form  C := alpha*A'*A + beta*C. */

    if (upper) {
        i__1 = *n;
        // 循环：从 1 到 n，对每一列 j 进行操作
        for (j = 1; j <= i__1; ++j) {
        i__2 = j;
        // 循环：从 1 到 j，对每一行 i 进行操作
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 初始化 temp 为 0
            temp = 0.f;
            // 循环：从 1 到 k，对每一行 l 进行操作
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 temp 为 A 的对应元素乘积之和
            temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L190: */
            }
            // 如果 beta 等于 0，则直接用 alpha 乘以 temp 更新 C 的元素
            if (*beta == 0.f) {
            c__[i__ + j * c_dim1] = *alpha * temp;
            } else {
            // 否则用 alpha 乘以 temp，加上 beta 倍的原 C 元素值来更新 C
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
            }
/* L200: */
        }
/* L210: */
        }
    } else {
        i__1 = *n;
        // 循环：从 1 到 n，对每一列 j 进行操作
        for (j = 1; j <= i__1; ++j) {
        i__2 = *n;
        // 循环：从 j 到 n，对每一行 i 进行操作
        for (i__ = j; i__ <= i__2; ++i__) {
            // 初始化 temp 为 0
            temp = 0.f;
            // 循环：从 1 到 k，对每一行 l 进行操作
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 temp 为 A 的对应元素乘积之和
            temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L220: */
            }
            // 如果 beta 等于 0，则直接用 alpha 乘以 temp 更新 C 的元素
            if (*beta == 0.f) {
            c__[i__ + j * c_dim1] = *alpha * temp;
            } else {
            // 否则用 alpha 乘以 temp，加上 beta 倍的原 C 元素值来更新 C
            c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
                i__ + j * c_dim1];
            }
/* L230: */
        }
/* L240: */
        }
    }
    }

    return 0;

/*     End of SSYRK . */

} /* ssyrk_ */

/* Subroutine */ int strmm_(char *side, char *uplo, char *transa, char *diag,
    integer *m, integer *n, real *alpha, real *a, integer *lda, real *b,
    integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, k, info;
    static real temp;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    STRMM  performs one of the matrix-matrix operations

       B := alpha*op( A )*B,   or   B := alpha*B*op( A ),

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'.

    Arguments
    ==========

*/
    SIDE   - CHARACTER*1.
             在进入时，SIDE指定了op(A)如何与B相乘的位置关系，具体如下：

                SIDE = 'L' or 'l'   B := alpha*op( A )*B.
                                      表示B在左边乘以op(A)。

                SIDE = 'R' or 'r'   B := alpha*B*op( A ).
                                      表示B在右边乘以op(A)。

             函数结束后保持不变。

    UPLO   - CHARACTER*1.
             在进入时，UPLO指定了矩阵A是上三角还是下三角矩阵，具体如下：

                UPLO = 'U' or 'u'   A是一个上三角矩阵。

                UPLO = 'L' or 'l'   A是一个下三角矩阵。

             函数结束后保持不变。

    TRANSA - CHARACTER*1.
             在进入时，TRANSA指定了在矩阵乘法中要使用的op(A)的形式，具体如下：

                TRANSA = 'N' or 'n'   op( A ) = A.
                                        表示op(A)为A本身。

                TRANSA = 'T' or 't'   op( A ) = A'.
                                        表示op(A)为A的转置。

                TRANSA = 'C' or 'c'   op( A ) = A'.
                                        表示op(A)为A的共轭转置。

             函数结束后保持不变。

    DIAG   - CHARACTER*1.
             在进入时，DIAG指定了矩阵A是否为单位三角形矩阵，具体如下：

                DIAG = 'U' or 'u'   A被假定为单位三角形矩阵。

                DIAG = 'N' or 'n'   A不被假定为单位三角形矩阵。

             函数结束后保持不变。

    M      - INTEGER.
             在进入时，M指定了矩阵B的行数。M必须至少为零。
             函数结束后保持不变。

    N      - INTEGER.
             在进入时，N指定了矩阵B的列数。N必须至少为零。
             函数结束后保持不变。

    ALPHA  - REAL            .
             在进入时，ALPHA指定了标量alpha的值。当alpha为零时，A不被引用，B在进入时不需要设置。
             函数结束后保持不变。

    A      - REAL             array of DIMENSION ( LDA, k ), where k is m
             在进入时，A是一个大小为(LDA, k)的REAL数组，其中k是当SIDE = 'L'或 'l'时为m，当SIDE = 'R'或 'r'时为n。
             在UPLO = 'U'或 'u'时，A的前k行k列必须包含上三角矩阵的内容，A的严格下三角部分不被引用。
             在UPLO = 'L'或 'l'时，A的前k行k列必须包含下三角矩阵的内容，A的严格上三角部分不被引用。
             注意当DIAG = 'U'或 'u'时，A的对角线元素也不被引用，但被假定为单位元素。
             函数结束后保持不变。
    # LDA    - INTEGER.
    #         On entry, LDA specifies the first dimension of A as declared
    #         in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
    #         LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
    #         then LDA must be at least max( 1, n ).
    #         Unchanged on exit.

    # B      - REAL             array of DIMENSION ( LDB, n ).
    #         Before entry,  the leading  m by n part of the array  B must
    #         contain the matrix  B,  and  on exit  is overwritten  by the
    #         transformed matrix.

    # LDB    - INTEGER.
    #         On entry, LDB specifies the first dimension of B as declared
    #         in  the  calling  (sub)  program.   LDB  must  be  at  least
    #         max( 1, m ).
    #         Unchanged on exit.

    # Further Details
    # ===============

    # Level 3 Blas routine.

    # -- Written on 8-February-1989.
    #    Jack Dongarra, Argonne National Laboratory.
    #    Iain Duff, AERE Harwell.
    #    Jeremy Du Croz, Numerical Algorithms Group Ltd.
    #    Sven Hammarling, Numerical Algorithms Group Ltd.

    # =====================================================================


    # Test the input parameters.
    /* Parameter adjustments */
    a_dim1 = *lda;  // 设置 a 的第一维度大小为 lda
    a_offset = 1 + a_dim1;  // 计算 a 的偏移量
    a -= a_offset;  // 调整 a 的指针使其指向正确的起始位置
    b_dim1 = *ldb;  // 设置 b 的第一维度大小为 ldb
    b_offset = 1 + b_dim1;  // 计算 b 的偏移量
    b -= b_offset;  // 调整 b 的指针使其指向正确的起始位置

    /* Function Body */
    lside = lsame_(side, "L");  // 检查 side 是否为 'L'，返回逻辑值
    if (lside) {  // 如果 side 是 'L'
        nrowa = *m;  // 设置 nrowa 为 m
    } else {
        nrowa = *n;  // 否则设置 nrowa 为 n
    }
    nounit = lsame_(diag, "N");  // 检查 diag 是否为 'N'，返回逻辑值
    upper = lsame_(uplo, "U");  // 检查 uplo 是否为 'U'，返回逻辑值

    info = 0;  // 初始化 info 为 0
    if (! lside && ! lsame_(side, "R")) {  // 如果 side 不是 'L' 且不是 'R'
        info = 1;  // 设置 info 为 1
    } else if (! upper && ! lsame_(uplo, "L")) {  // 如果 uplo 不是 'U' 且不是 'L'
        info = 2;  // 设置 info 为 2
    } else if (! lsame_(transa, "N") && ! lsame_(transa, "T") && ! lsame_(transa, "C")) {
        // 如果 transa 既不是 'N' 也不是 'T' 也不是 'C'
        info = 3;  // 设置 info 为 3
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {  // 如果 diag 既不是 'U' 也不是 'N'
        info = 4;  // 设置 info 为 4
    } else if (*m < 0) {  // 如果 m 小于 0
        info = 5;  // 设置 info 为 5
    } else if (*n < 0) {  // 如果 n 小于 0
        info = 6;  // 设置 info 为 6
    } else if (*lda < max(1,nrowa)) {  // 如果 lda 小于 1 和 nrowa 中的最大值
        info = 9;  // 设置 info 为 9
    } else if (*ldb < max(1,*m)) {  // 如果 ldb 小于 1 和 m 中的最大值
        info = 11;  // 设置 info 为 11
    }
    if (info != 0) {  // 如果 info 不等于 0
        xerbla_("STRMM ", &info);  // 调用错误处理函数 xerbla_
        return 0;  // 返回 0
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0) {  // 如果 m 或 n 等于 0
        return 0;  // 返回 0
    }

/*     And when alpha.eq.zero. */

    if (*alpha == 0.f) {  // 如果 alpha 等于 0
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 循环遍历 j 从 1 到 n
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {  // 循环遍历 i 从 1 到 m
                b[i__ + j * b_dim1] = 0.f;  // 设置 b[i, j] 为 0
/* L10: */
            }
/* L20: */
        }
        return 0;  // 返回 0
    }

/*     Start the operations. */

    if (lside) {  // 如果 side 是 'L'
        if (lsame_(transa, "N")) {  // 如果 transa 是 'N'

/*           Form  B := alpha*A*B. */

            if (upper) {  // 如果 uplo 是 'U'
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {  // 循环遍历 j 从 1 到 n
                    i__2 = *m;
                    for (k = 1; k <= i__2; ++k) {  // 循环遍历 k 从 1 到 m
                        if (b[k + j * b_dim1] != 0.f) {  // 如果 b[k, j] 不等于 0
                            temp = *alpha * b[k + j * b_dim1];  // 计算临时变量 temp
                            i__3 = k - 1;
                            for (i__ = 1; i__ <= i__3; ++i__) {  // 循环遍历 i 从 1 到 k-1
                                b[i__ + j * b_dim1] += temp * a[i__ + k * a_dim1];  // 更新 b[i, j]
/* L30: */
                            }
                            if (nounit) {  // 如果 nounit 是 true
                                temp *= a[k + k * a_dim1];  // 更新 temp
                            }
                            b[k + j * b_dim1] = temp;  // 更新 b[k, j]
                        }
/* L40: */
                    }
/* L50: */
                }
            } else {  // 如果 uplo 不是 'U'
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {  // 循环遍历 j 从 1 到 n
                    for (k = *m; k >= 1; --k) {  // 循环遍历 k 从 m 到 1
                        if (b[k + j * b_dim1] != 0.f) {  // 如果 b[k, j] 不等于 0
                            temp = *alpha * b[k + j * b_dim1];  // 计算临时变量 temp
                            b[k + j * b_dim1] = temp;  // 更新 b[k, j]
                            if (nounit) {  // 如果 nounit 是 true
                                b[k + j * b_dim1] *= a[k + k * a_dim1];  // 更新 b[k, j]
                            }
                            i__2 = *m;
                            for (i__ = k + 1; i__ <= i__2; ++i__) {  // 循环遍历 i 从 k+1 到 m
                                b[i__ + j * b_dim1] += temp * a[i__ + k * a_dim1];  // 更新 b[i, j]
/* L60: */
                            }
                        }
/* L70: */
                    }
/* L80: */
                }
            }
        } else {
/*           Form  B := alpha*A'*B. */

if (upper) {
    // 循环遍历 B 的列
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // 循环遍历 B 的行（反向）
        for (i__ = *m; i__ >= 1; --i__) {
            // 临时变量 temp 存储当前 B 中的元素
            temp = b[i__ + j * b_dim1];
            // 如果非单位矩阵，将 temp 乘以 A 的对角元素
            if (nounit) {
                temp *= a[i__ + i__ * a_dim1];
            }
            // 累加上三角部分的乘积
            i__2 = i__ - 1;
            for (k = 1; k <= i__2; ++k) {
                temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
                /* L90: */
            }
            // 将计算结果乘以 alpha，并写回到 B 中
            b[i__ + j * b_dim1] = *alpha * temp;
            /* L100: */
        }
        /* L110: */
    }
} else {
    // 如果不是上三角形式的 A

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // 循环遍历 B 的列
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 临时变量 temp 存储当前 B 中的元素
            temp = b[i__ + j * b_dim1];
            // 如果非单位矩阵，将 temp 乘以 A 的对角元素
            if (nounit) {
                temp *= a[i__ + i__ * a_dim1];
            }
            // 累加下三角部分的乘积
            i__3 = *m;
            for (k = i__ + 1; k <= i__3; ++k) {
                temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
                /* L120: */
            }
            // 将计算结果乘以 alpha，并写回到 B 中
            b[i__ + j * b_dim1] = *alpha * temp;
            /* L130: */
        }
        /* L140: */
    }
}
/*           Form  B := alpha*B*A'. */

        // 根据 upper 的值判断是上三角还是下三角操作
        if (upper) {
        // 循环处理每列 k
        i__1 = *n;
        for (k = 1; k <= i__1; ++k) {
            // 循环处理每列 j，j < k 的部分
            i__2 = k - 1;
            for (j = 1; j <= i__2; ++j) {
            // 如果 A[j + k * a_dim1] 不为零，则执行下列操作
            if (a[j + k * a_dim1] != 0.f) {
                // 计算临时变量 temp
                temp = *alpha * a[j + k * a_dim1];
                // 循环处理每行 i
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                // 更新 B[i__ + j * b_dim1]
                b[i__ + j * b_dim1] += temp * b[i__ + k *
                    b_dim1];
/* L230: */
                }
            }
/* L240: */
            }
            // 计算临时变量 temp
            temp = *alpha;
            // 如果非单位矩阵，则更新 temp
            if (nounit) {
            temp *= a[k + k * a_dim1];
            }
            // 如果 temp 不为 1，则执行下列操作
            if (temp != 1.f) {
            // 循环处理每行 i
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 更新 B[i__ + k * b_dim1]
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L250: */
            }
            }
/* L260: */
        }
        } else {
        // 与上面相反的循环顺序，处理每列 k
        for (k = *n; k >= 1; --k) {
            // 处理每列 j，j > k 的部分
            i__1 = *n;
            for (j = k + 1; j <= i__1; ++j) {
            // 如果 A[j + k * a_dim1] 不为零，则执行下列操作
            if (a[j + k * a_dim1] != 0.f) {
                // 计算临时变量 temp
                temp = *alpha * a[j + k * a_dim1];
                // 循环处理每行 i
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                // 更新 B[i__ + j * b_dim1]
                b[i__ + j * b_dim1] += temp * b[i__ + k *
                    b_dim1];
/* L270: */
                }
            }
/* L280: */
            }
            // 计算临时变量 temp
            temp = *alpha;
            // 如果非单位矩阵，则更新 temp
            if (nounit) {
            temp *= a[k + k * a_dim1];
            }
            // 如果 temp 不为 1，则执行下列操作
            if (temp != 1.f) {
            // 循环处理每行 i
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                // 更新 B[i__ + k * b_dim1]
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L290: */
            }
            }
/* L300: */
        }
        }
    }
    }

    return 0;

/*     End of STRMM . */

} /* strmm_ */

/* Subroutine */ int strmv_(char *uplo, char *trans, char *diag, integer *n,
    real *a, integer *lda, real *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, ix, jx, kx, info;
    static real temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    STRMV  performs one of the matrix-vector operations

       x := A*x,   or   x := A'*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.


*/
    # 定义字符变量 TRANS，指定矩阵向量乘法操作类型：
    # - 'N' 或 'n' 表示 x := A*x
    # - 'T' 或 't' 表示 x := A'*x
    # - 'C' 或 'c' 也表示 x := A'*x
    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

    # 定义字符变量 DIAG，指定矩阵 A 是否为单位上（下）三角矩阵：
    # - 'U' 或 'u' 表示 A 被假设为单位上（下）三角矩阵
    # - 'N' 或 'n' 表示 A 不被假设为单位三角矩阵
    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit
             triangular as follows:

    # 定义整数变量 N，指定矩阵 A 的阶数（维度）
    N      - INTEGER.
             On entry, N specifies the order of the matrix A.

    # 定义实数数组 A，存储矩阵 A 的数据
    A      - REAL             array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular matrix and the strictly lower triangular part of
             A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular matrix and the strictly upper triangular part of
             A is not referenced.

    # 定义整数变量 LDA，指定矩阵 A 的第一维度大小
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program.

    # 定义实数数组 X，存储向量 x 的数据
    X      - REAL             array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x. On exit, X is overwritten with the
             transformed vector x.

    # 定义整数变量 INCX，指定向量 X 中元素的增量
    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================


       Test the input parameters.

    # 进行输入参数的测试
    /* Parameter adjustments */
    a_dim1 = *lda;  // 计算二维数组 A 在第一维度的步长
    a_offset = 1 + a_dim1;  // 计算 A 数组偏移量
    a -= a_offset;  // 调整 A 的起始地址，使其对应正确的子数组

    --x;  // 将向量 X 的起始地址前移一位，从而实现从 1 开始索引

    /* Function Body */
    info = 0;  // 初始化 info 变量，用于存储错误信息编号
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {  // 检查参数 uplo 是否合法
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {  // 检查参数 trans 是否合法
        info = 2;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {  // 检查参数 diag 是否合法
        info = 3;
    } else if (*n < 0) {  // 检查参数 n 是否合法
        info = 4;
    } else if (*lda < max(1,*n)) {  // 检查参数 lda 是否合法
        info = 6;
    } else if (*incx == 0) {  // 检查参数 incx 是否合法
        info = 8;
    }
    if (info != 0) {  // 如果有错误信息，则调用错误处理程序并返回
        xerbla_("STRMV ", &info);
        return 0;
    }

    /* Quick return if possible. */
    if (*n == 0) {  // 如果 n 为 0，则无需计算，直接返回
        return 0;
    }

    nounit = lsame_(diag, "N");  // 判断是否是非单位矩阵

    /*
       Set up the start point in X if the increment is not unity. This
       will be  ( N - 1 )*INCX  too small for descending loops.
    */
    if (*incx <= 0) {  // 如果增量 incx 小于等于 0
        kx = 1 - (*n - 1) * *incx;  // 计算起始点 kx，确保对于递减循环不会太小
    } else if (*incx != 1) {  // 如果增量 incx 不等于 1
        kx = 1;  // 设置起始点 kx 为 1
    }

    /*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
    */
    if (lsame_(trans, "N")) {  // 如果 trans 等于 "N"

        /* Form  x := A*x. */
        if (lsame_(uplo, "U")) {  // 如果 uplo 等于 "U"
            if (*incx == 1) {  // 如果增量 incx 等于 1
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (x[j] != 0.f) {  // 如果 x[j] 不等于 0
                        temp = x[j];  // 临时存储 x[j]
                        i__2 = j - 1;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            x[i__] += temp * a[i__ + j * a_dim1];  // 更新 x[i__] += temp * A[i__, j]
                        }
                        if (nounit) {  // 如果不是单位矩阵
                            x[j] *= a[j + j * a_dim1];  // 更新 x[j] *= A[j, j]
                        }
                    }
                }
            } else {  // 如果增量 incx 不等于 1
                jx = kx;  // 设置起始点 jx 为 kx
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (x[jx] != 0.f) {  // 如果 x[jx] 不等于 0
                        temp = x[jx];  // 临时存储 x[jx]
                        ix = kx;  // 设置 ix 为 kx
                        i__2 = j - 1;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            x[ix] += temp * a[i__ + j * a_dim1];  // 更新 x[ix] += temp * A[i__, j]
                            ix += *incx;  // 更新 ix
                        }
                        if (nounit) {  // 如果不是单位矩阵
                            x[jx] *= a[j + j * a_dim1];  // 更新 x[jx] *= A[j, j]
                        }
                    }
                    jx += *incx;  // 更新 jx
                }
            }
        } else {  // 如果 uplo 不等于 "U"
            if (*incx == 1) {  // 如果增量 incx 等于 1
                for (j = *n; j >= 1; --j) {
                    if (x[j] != 0.f) {  // 如果 x[j] 不等于 0
                        temp = x[j];  // 临时存储 x[j]
                        i__1 = j + 1;
                        for (i__ = *n; i__ >= i__1; --i__) {
                            x[i__] += temp * a[i__ + j * a_dim1];  // 更新 x[i__] += temp * A[i__, j]
                        }
                        if (nounit) {  // 如果不是单位矩阵
                            x[j] *= a[j + j * a_dim1];  // 更新 x[j] *= A[j, j]
                        }
                    }
                }
            } else {  // 如果增量 incx 不等于 1
                kx += (*n - 1) * *incx;  // 更新 kx
                jx = kx;  // 设置起始点 jx 为 kx
                for (j = *n; j >= 1; --j) {
                    if (x[jx] != 0.f) {  // 如果 x[jx] 不等于 0
                        temp = x[jx];  // 临时存储 x[jx]
                        ix = kx;  // 设置 ix 为 kx
                        i__1 = j + 1;
                        for (i__ = *n; i__ >= i__1; --i__) {
                            x[ix] += temp * a[i__ + j * a_dim1];  // 更新 x[ix] += temp * A[i__, j]
                            ix -= *incx;  // 更新 ix
                        }
                        if (nounit) {  // 如果不是单位矩阵
                            x[jx] *= a[j + j * a_dim1];  // 更新 x[jx] *= A[j, j]
                        }
                    }
                    jx -= *incx;  // 更新 jx
                }
            }
        }
    } else {  // 如果 trans 不等于 "N"

        /* Form  x := A'*x or x := A'*x. */
        // 暂时省略此部分的详细注释，因为未提供此部分的具体代码内容
    }
/* L80: */
        }
        }
    }
    } else {

/*        Form  x := A'*x. */

    if (lsame_(uplo, "U")) {
        if (*incx == 1) {
        // 逐列处理，从最后一列开始向前计算
        for (j = *n; j >= 1; --j) {
            temp = x[j];
            if (nounit) {
            // 如果不是单位上三角矩阵，乘以对角线元素
            temp *= a[j + j * a_dim1];
            }
            // 累加每一列的对应元素乘积
            for (i__ = j - 1; i__ >= 1; --i__) {
            temp += a[i__ + j * a_dim1] * x[i__];
/* L90: */
            }
            // 更新结果向量的当前列
            x[j] = temp;
/* L100: */
        }
        } else {
        // 非单位上三角矩阵，增量计算
        jx = kx + (*n - 1) * *incx;
        for (j = *n; j >= 1; --j) {
            temp = x[jx];
            ix = jx;
            if (nounit) {
            // 如果不是单位上三角矩阵，乘以对角线元素
            temp *= a[j + j * a_dim1];
            }
            // 累加每一列的对应元素乘积
            for (i__ = j - 1; i__ >= 1; --i__) {
            ix -= *incx;
            temp += a[i__ + j * a_dim1] * x[ix];
/* L110: */
            }
            // 更新结果向量的当前列
            x[jx] = temp;
            jx -= *incx;
/* L120: */
        }
        }
    } else {
        if (*incx == 1) {
        // 逐列处理，从第一列开始向后计算
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            temp = x[j];
            if (nounit) {
            // 如果不是单位下三角矩阵，乘以对角线元素
            temp *= a[j + j * a_dim1];
            }
            // 累加每一列的对应元素乘积
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            temp += a[i__ + j * a_dim1] * x[i__];
/* L130: */
            }
            // 更新结果向量的当前列
            x[j] = temp;
/* L140: */
        }
        } else {
        // 非单位下三角矩阵，增量计算
        jx = kx;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            temp = x[jx];
            ix = jx;
            if (nounit) {
            // 如果不是单位下三角矩阵，乘以对角线元素
            temp *= a[j + j * a_dim1];
            }
            // 累加每一列的对应元素乘积
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            ix += *incx;
            temp += a[i__ + j * a_dim1] * x[ix];
/* L150: */
            }
            // 更新结果向量的当前列
            x[jx] = temp;
            jx += *incx;
/* L160: */
        }
        }
    }
    }

    return 0;

/*     End of STRMV . */

} /* strmv_ */

/* Subroutine */ int strsm_(char *side, char *uplo, char *transa, char *diag,
    integer *m, integer *n, real *alpha, real *a, integer *lda, real *b,
    integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, k, info;
    static real temp;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    STRSM  solves one of the matrix equations

       op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'.

    The matrix X is overwritten on B.

    Arguments
    ==========
    # SIDE   - CHARACTER*1.
    #          在输入时，SIDE指定op(A)出现在X的左侧还是右侧，具体如下：

    #             SIDE = 'L' or 'l'   op( A )*X = alpha*B.
    #                                左侧操作：op(A)与X相乘得到alpha*B.

    #             SIDE = 'R' or 'r'   X*op( A ) = alpha*B.
    #                                右侧操作：X与op(A)相乘得到alpha*B.

    #          函数返回后保持不变。

    # UPLO   - CHARACTER*1.
    #          在输入时，UPLO指定矩阵A是上三角还是下三角矩阵，具体如下：

    #             UPLO = 'U' or 'u'   A是上三角矩阵。
    #             UPLO = 'L' or 'l'   A是下三角矩阵。

    #          函数返回后保持不变。

    # TRANSA - CHARACTER*1.
    #          在输入时，TRANSA指定在矩阵乘法中要使用的op(A)的形式，具体如下：

    #             TRANSA = 'N' or 'n'   op( A ) = A.
    #                                不转置操作：op(A)等于A本身。

    #             TRANSA = 'T' or 't'   op( A ) = A'.
    #                                转置操作：op(A)等于A的转置。

    #             TRANSA = 'C' or 'c'   op( A ) = A'.
    #                                共轭转置操作：op(A)等于A的共轭转置。

    #          函数返回后保持不变。

    # DIAG   - CHARACTER*1.
    #          在输入时，DIAG指定矩阵A是否是单位三角形矩阵，具体如下：

    #             DIAG = 'U' or 'u'   A被假设为单位三角形矩阵。
    #             DIAG = 'N' or 'n'   A不被假设为单位三角形矩阵。

    #          函数返回后保持不变。

    # M      - INTEGER.
    #          在输入时，M指定矩阵B的行数。M必须至少为零。
    #          函数返回后保持不变。

    # N      - INTEGER.
    #          在输入时，N指定矩阵B的列数。N必须至少为零。
    #          函数返回后保持不变。

    # ALPHA  - REAL            .
    #          在输入时，ALPHA指定标量alpha。当alpha为零时，A不被引用，B在输入前不需要设置。
    #          函数返回后保持不变。

    # A      - REAL             array of DIMENSION ( LDA, k ), where k is m
    #          when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
    #          在输入时，A是一个二维实数数组，维度为(LDA, k)，其中当SIDE = 'L'或'l'时，k为m；当SIDE = 'R'或'r'时，k为n。
    #          在输入前，当UPLO = 'U'或'u'时，数组A的前k行k列应包含上三角矩阵的上三角部分，且严格下三角部分不被引用。
    #          在输入前，当UPLO = 'L'或'l'时，数组A的前k行k列应包含下三角矩阵的下三角部分，且严格上三角部分不被引用。
    #          注意，当DIAG = 'U'或'u'时，对角线元素也不被引用，但假定为单位元素。
    #          函数返回后保持不变。
    # LDA    - 整数。
             # 在输入时，LDA指定了矩阵A的第一个维度大小，即调用（子）程序中声明的大小。
             # 当 SIDE = 'L' 或 'l' 时，LDA至少必须是 max(1, m)；当 SIDE = 'R' 或 'r' 时，
             # LDA必须至少是 max(1, n)。
             # 在退出时保持不变。

    # B      - 实数数组，大小为 (LDB, n)。
             # 在输入前，数组B的前 m 行 n 列必须包含右侧矩阵B的内容；在退出时，会被重写为解矩阵X。

    # LDB    - 整数。
             # 在输入时，LDB指定了矩阵B的第一个维度大小，即调用（子）程序中声明的大小。
             # LDB必须至少是 max(1, m)。
             # 在退出时保持不变。

    # Further Details
    # ===============
    
    # Level 3 Blas routine.
    # BLAS第三级例程。

    # -- Written on 8-February-1989.
       # Jack Dongarra, Argonne National Laboratory.
       # Iain Duff, AERE Harwell.
       # Jeremy Du Croz, Numerical Algorithms Group Ltd.
       # Sven Hammarling, Numerical Algorithms Group Ltd.

    # =====================================================================


       # Test the input parameters.
       # 测试输入参数。
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    lside = lsame_(side, "L");
    if (lside) {
        nrowa = *m;
    } else {
        nrowa = *n;
    }
    nounit = lsame_(diag, "N");
    upper = lsame_(uplo, "U");

    info = 0;
    if (! lside && ! lsame_(side, "R")) {
        info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
        info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa,
         "T") && ! lsame_(transa, "C")) {
        info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
        "N")) {
        info = 4;
    } else if (*m < 0) {
        info = 5;
    } else if (*n < 0) {
        info = 6;
    } else if (*lda < max(1,nrowa)) {
        info = 9;
    } else if (*ldb < max(1,*m)) {
        info = 11;
    }
    if (info != 0) {
        xerbla_("STRSM ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0) {
        return 0;
    }

/*     And when alpha.eq.zero. */

    if (*alpha == 0.f) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + j * b_dim1] = 0.f;
/* L10: */
            }
/* L20: */
        }
        return 0;
    }

/*     Start the operations. */

    if (lside) {
        if (lsame_(transa, "N")) {

/*           Form B := alpha*inv(A)*B. */

            if (upper) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (*alpha != 1.f) {
                        i__2 = *m;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
/* L30: */
                        }
                    }
                    for (k = *m; k >= 1; --k) {
                        if (b[k + j * b_dim1] != 0.f) {
                            if (nounit) {
                                b[k + j * b_dim1] /= a[k + k * a_dim1];
                            }
                            i__2 = k - 1;
                            for (i__ = 1; i__ <= i__2; ++i__) {
                                b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[i__ + k * a_dim1];
/* L40: */
                            }
                        }
/* L50: */
                    }
/* L60: */
                }
            } else {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    if (*alpha != 1.f) {
                        i__2 = *m;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1];
/* L70: */
                        }
                    }
                    i__2 = *m;
                    for (k = 1; k <= i__2; ++k) {
                        if (b[k + j * b_dim1] != 0.f) {
                            if (nounit) {
                                b[k + j * b_dim1] /= a[k + k * a_dim1];
                            }
                            i__3 = *m;
                            for (i__ = k + 1; i__ <= i__3; ++i__) {
                                b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[i__ + k * a_dim1];
/* L80: */
                            }
                        }
/* L90: */
                    }
/* L100: */
                }
            }
    } else {
/*           Form  B := alpha*inv( A' )*B. */

        if (upper) {
        i__1 = *n;  // 循环：对每列进行操作，从第一列到第n列
        for (j = 1; j <= i__1; ++j) {  // 遍历B的列
            i__2 = *m;  // 循环：对每行进行操作，从第一行到第m行
            for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历B的行
            temp = *alpha * b[i__ + j * b_dim1];  // 计算临时变量temp
            i__3 = i__ - 1;  // 循环：对每行中的前i-1行元素进行操作
            for (k = 1; k <= i__3; ++k) {  // 遍历A的列
                temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];  // 更新temp
/* L110: */  // 标号：循环结束
            }
            if (nounit) {  // 如果非单位三角矩阵
                temp /= a[i__ + i__ * a_dim1];  // 更新temp
            }
            b[i__ + j * b_dim1] = temp;  // 更新B的元素
/* L120: */  // 标号：循环结束
            }
/* L130: */  // 标号：循环结束
        }
        } else {
        i__1 = *n;  // 循环：对每列进行操作，从第一列到第n列
        for (j = 1; j <= i__1; ++j) {  // 遍历B的列
            for (i__ = *m; i__ >= 1; --i__) {  // 逆序遍历B的行
            temp = *alpha * b[i__ + j * b_dim1];  // 计算临时变量temp
            i__2 = *m;  // 循环：对每行进行操作，从第一行到第m行
            for (k = i__ + 1; k <= i__2; ++k) {  // 遍历A的列
                temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];  // 更新temp
/* L140: */  // 标号：循环结束
            }
            if (nounit) {  // 如果非单位三角矩阵
                temp /= a[i__ + i__ * a_dim1];  // 更新temp
            }
            b[i__ + j * b_dim1] = temp;  // 更新B的元素
/* L150: */  // 标号：循环结束
            }
/* L160: */  // 标号：循环结束
        }
        }
    }
    } else {
    if (lsame_(transa, "N")) {

/*           Form  B := alpha*B*inv( A ). */

        if (upper) {
        i__1 = *n;  // 循环：对每列进行操作，从第一列到第n列
        for (j = 1; j <= i__1; ++j) {  // 遍历B的列
            if (*alpha != 1.f) {  // 如果alpha不等于1
            i__2 = *m;  // 循环：对每行进行操作，从第一行到第m行
            for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历B的行
                b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
                    ;  // 更新B的元素
/* L170: */  // 标号：循环结束
            }
            }
            i__2 = j - 1;  // 循环：对每列中的前j-1列进行操作
            for (k = 1; k <= i__2; ++k) {  // 遍历A的列
            if (a[k + j * a_dim1] != 0.f) {  // 如果A的元素不为零
                i__3 = *m;  // 循环：对每行进行操作，从第一行到第m行
                for (i__ = 1; i__ <= i__3; ++i__) {  // 遍历B的行
                b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[
                    i__ + k * b_dim1];  // 更新B的元素
/* L180: */  // 标号：循环结束
                }
            }
/* L190: */  // 标号：循环结束
            }
            if (nounit) {  // 如果非单位三角矩阵
            temp = 1.f / a[j + j * a_dim1];  // 计算临时变量temp
            i__2 = *m;  // 循环：对每行进行操作，从第一行到第m行
            for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历B的行
                b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];  // 更新B的元素
/* L200: */  // 标号：循环结束
            }
            }
/* L210: */  // 标号：循环结束
        }
        } else {
        for (j = *n; j >= 1; --j) {  // 逆序遍历B的列
            if (*alpha != 1.f) {  // 如果alpha不等于1
            i__1 = *m;  // 循环：对每行进行操作，从第一行到第m行
            for (i__ = 1; i__ <= i__1; ++i__) {  // 遍历B的行
                b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
                    ;  // 更新B的元素
/* L220: */  // 标号：循环结束
            }
            }
            i__1 = *n;  // 循环：对每列进行操作，从第一列到第n列
            for (k = j + 1; k <= i__1; ++k) {  // 遍历A的列
            if (a[k + j * a_dim1] != 0.f) {  // 如果A的元素不为零
                i__2 = *m;  // 循环：对每行进行操作，从第一行到第m行
                for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历B的行
                b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[
                    i__ + k * b_dim1];  // 更新B的元素
/* L230: */  // 标号：循环结束
                }
            }
/* L240: */  // 标号：循环结束
            }
            if (nounit) {  // 如果非单位三角矩阵
            temp = 1.f / a[j + j * a_dim1];  // 计算临时变量temp
            i__1 = *m;  // 循环：对每行进行操作，从第一行到第m行
            for (i__ = 1; i__ <= i__1; ++i__) {  // 遍历B的行
                b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];  // 更新B的元素
/* L250: */  // 标号：循环结束
            }
            }
/* L260: */  // 标号：循环结束
        }
        }
    }
    }
/* L250: */
            }
            }
/* L260: */
        }
        }
    } else {

/*           Form  B := alpha*B*inv( A' ). */

        if (upper) {
        // 从右往左处理每列
        for (k = *n; k >= 1; --k) {
            if (nounit) {
            // 如果不是单位对角矩阵，计算 A' 的逆的乘积
            temp = 1.f / a[k + k * a_dim1];
            // 处理 B 的每一行
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L270: */
            }
            }
            // 处理非对角元素
            i__1 = k - 1;
            for (j = 1; j <= i__1; ++j) {
            if (a[j + k * a_dim1] != 0.f) {
                temp = a[j + k * a_dim1];
                // 更新 B 的每一行
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + j * b_dim1] -= temp * b[i__ + k *
                    b_dim1];
/* L280: */
                }
            }
/* L290: */
            }
            // 如果 alpha 不等于 1，乘以 alpha
            if (*alpha != 1.f) {
            // 处理 B 的每一行
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1]
                    ;
/* L300: */
            }
            }
/* L310: */
        }
        } else {
        // 从左往右处理每列
        i__1 = *n;
        for (k = 1; k <= i__1; ++k) {
            if (nounit) {
            // 如果不是单位对角矩阵，计算 A' 的逆的乘积
            temp = 1.f / a[k + k * a_dim1];
            // 处理 B 的每一行
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L320: */
            }
            }
            // 处理非对角元素
            i__2 = *n;
            for (j = k + 1; j <= i__2; ++j) {
            if (a[j + k * a_dim1] != 0.f) {
                temp = a[j + k * a_dim1];
                // 更新 B 的每一行
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                b[i__ + j * b_dim1] -= temp * b[i__ + k *
                    b_dim1];
/* L330: */
                }
            }
/* L340: */
            }
            // 如果 alpha 不等于 1，乘以 alpha
            if (*alpha != 1.f) {
            // 处理 B 的每一行
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1]
                    ;
/* L350: */
            }
            }
/* L360: */
        }
        }
    }
    }

    return 0;

/*     End of STRSM . */

} /* strsm_ */

/* Subroutine */ int zaxpy_(integer *n, doublecomplex *za, doublecomplex *zx,
    integer *incx, doublecomplex *zy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublecomplex z__1, z__2;

    /* Local variables */
    static integer i__, ix, iy;
    extern doublereal dcabs1_(doublecomplex *);


/*
    Purpose
    =======

       ZAXPY constant times a vector plus a vector.

    Further Details
    ===============

       jack dongarra, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --zy;
    --zx;

    /* Function Body */
    if (*n <= 0) {
    return 0;
    }
    if (dcabs1_(za) == 0.) {
    return 0;
    }
    if (*incx == 1 && *incy == 1) {
    goto L20;
    }
/*
          code for unequal increments or equal increments
            not equal to 1
*/
ix = 1;  
iy = 1;  
if (*incx < 0) {
    ix = (-(*n) + 1) * *incx + 1;
}
if (*incy < 0) {
    iy = (-(*n) + 1) * *incy + 1;
}
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = iy;
    i__3 = iy;
    i__4 = ix;
    // Perform complex multiplication and addition
    z__2.r = za->r * zx[i__4].r - za->i * zx[i__4].i, z__2.i = za->r * zx[
        i__4].i + za->i * zx[i__4].r;
    z__1.r = zy[i__3].r + z__2.r, z__1.i = zy[i__3].i + z__2.i;
    // Assign the result to zy[iy]
    zy[i__2].r = z__1.r, zy[i__2].i = z__1.i;
    // Increment ix and iy according to specified increments
    ix += *incx;
    iy += *incy;
    // Label for loop exit
    /* L10: */
}
return 0;

/*        code for both increments equal to 1 */
L20:
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = i__;
    i__3 = i__;
    // Directly copy zx[i__] to zy[i__]
    zy[i__2].r = zx[i__3].r, zy[i__2].i = zx[i__3].i;
    // Label for loop exit
    /* L30: */
}
return 0;
} /* zaxpy_ */

/* Subroutine */ int zcopy_(integer *n, doublecomplex *zx, integer *incx,
    doublecomplex *zy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Local variables */
    static integer i__, ix, iy;

    /*
    Purpose
    =======

       ZCOPY copies a vector, x, to a vector, y.

    Further Details
    ===============

       jack dongarra, linpack, 4/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
    */

    /* Parameter adjustments */
    --zy;
    --zx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        // Jump to loop for increments equal to 1
        goto L20;
    }

    /*
          code for unequal increments or equal increments
            not equal to 1
    */
    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = iy;
        i__3 = ix;
        // Copy zx[ix] to zy[iy]
        zy[i__2].r = zx[i__3].r, zy[i__2].i = zx[i__3].i;
        // Increment ix and iy according to specified increments
        ix += *incx;
        iy += *incy;
        // Label for loop exit
        /* L10: */
    }
    return 0;

    /*        code for both increments equal to 1 */
L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        i__2 = i__;
        i__3 = i__;
        // Directly copy zx[i__] to zy[i__]
        zy[i__2].r = zx[i__3].r, zy[i__2].i = zx[i__3].i;
        // Label for loop exit
        /* L30: */
    }
    return 0;
} /* zcopy_ */

/* Double Complex */ VOID zdotc_(doublecomplex * ret_val, integer *n,
    doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, ix, iy;
    static doublecomplex ztemp;

    /*
    Purpose
    =======

    ZDOTC forms the dot product of a vector.

    Further Details

*/
    # 以下是一段注释文本，描述了代码的作者和修改历史。
    # jack dongarra, 3/11/78. 表示此代码最初由 jack dongarra 在 1978 年 3 月 11 日编写。
    # modified 12/3/93, array(1) declarations changed to array(*) 表示此代码在 1993 年 12 月 3 日被修改，
    # 将 array(1) 声明更改为 array(*)。
    # 这段文本可能是放在代码开头的注释，用于说明代码的起源和重要修改历史。
    /* Parameter adjustments */
    --zy;
    --zx;

    /* Function Body */
    ztemp.r = 0., ztemp.i = 0.;
    ret_val->r = 0.,  ret_val->i = 0.;
    if (*n <= 0) {
        // 如果 n 小于等于 0，直接返回
        return ;
    }
    if (*incx == 1 && *incy == 1) {
        // 如果 incx 和 incy 都等于 1，则跳转到标签 L20
        goto L20;
    }

    /*
      code for unequal increments or equal increments
        not equal to 1
    */
    ix = 1;
    iy = 1;
    if (*incx < 0) {
        // 如果 incx 小于 0，则计算新的起始位置 ix
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        // 如果 incy 小于 0，则计算新的起始位置 iy
        iy = (-(*n) + 1) * *incy + 1;
    }
    // 循环遍历计算向量的复数点乘
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        d_cnjg(&z__3, &zx[ix]);
        i__2 = iy;
        z__2.r = z__3.r * zy[i__2].r - z__3.i * zy[i__2].i, z__2.i = z__3.r * zy[i__2].i + z__3.i * zy[i__2].r;
        z__1.r = ztemp.r + z__2.r, z__1.i = ztemp.i + z__2.i;
        ztemp.r = z__1.r, ztemp.i = z__1.i;
        ix += *incx;
        iy += *incy;
        /* L10: */
    }
    ret_val->r = ztemp.r,  ret_val->i = ztemp.i;
    return ;

    /*        code for both increments equal to 1 */

L20:
    // 当 incx 和 incy 都等于 1 时的循环计算复数点乘
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        d_cnjg(&z__3, &zx[i__]);
        i__2 = i__;
        z__2.r = z__3.r * zy[i__2].r - z__3.i * zy[i__2].i, z__2.i = z__3.r * zy[i__2].i + z__3.i * zy[i__2].r;
        z__1.r = ztemp.r + z__2.r, z__1.i = ztemp.i + z__2.i;
        ztemp.r = z__1.r, ztemp.i = z__1.i;
        /* L30: */
    }
    ret_val->r = ztemp.r,  ret_val->i = ztemp.i;
    return ;
} /* zdotc_ */


注释：

- Parameter adjustments
    调整参数数组的起始位置
- Function Body
    函数体开始
- 如果 n 小于等于 0，直接返回
- 如果 incx 和 incy 都等于 1，则跳转到标签 L20
- 对于不等于 1 的不同增量，使用不同的计算方式
- 当 incx 和 incy 都等于 1 时，使用简化的计算方式
    // 循环迭代变量 i__ 从 1 开始递增，直到 i__1 的值
    for (i__ = 1; i__ <= i__1; ++i__) {
        // 将当前索引 i__2 和 i__3 分别赋值给 i__
        i__2 = i__;
        i__3 = i__;
        // 计算复数乘法结果 zx[i__2] * zy[i__3] 并存储在 z__2 中
        z__2.r = zx[i__2].r * zy[i__3].r - zx[i__2].i * zy[i__3].i,
        z__2.i = zx[i__2].r * zy[i__3].i + zx[i__2].i * zy[i__3].r;
        // 将 ztemp 和 z__2 的复数加法结果存储在 z__1 中
        z__1.r = ztemp.r + z__2.r, z__1.i = ztemp.i + z__2.i;
        // 更新 ztemp 的值为 z__1 的值
        ztemp.r = z__1.r, ztemp.i = z__1.i;
    }
/* L30: */
    }
     ret_val->r = ztemp.r,  ret_val->i = ztemp.i;
    return ;
} /* zdotu_ */

/* Subroutine */ int zdrot_(integer *n, doublecomplex *cx, integer *incx,
    doublecomplex *cy, integer *incy, doublereal *c__, doublereal *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, ix, iy;
    static doublecomplex ctemp;


/*
    Purpose
    =======

    Applies a plane rotation, where the cos and sin (c and s) are real
    and the vectors cx and cy are complex.
    jack dongarra, linpack, 3/11/78.

    Arguments
    ==========

    N        (input) INTEGER
             On entry, N specifies the order of the vectors cx and cy.
             N must be at least zero.
             Unchanged on exit.

    CX       (input) COMPLEX*16 array, dimension at least
             ( 1 + ( N - 1 )*abs( INCX ) ).
             Before entry, the incremented array CX must contain the n
             element vector cx. On exit, CX is overwritten by the updated
             vector cx.

    INCX     (input) INTEGER
             On entry, INCX specifies the increment for the elements of
             CX. INCX must not be zero.
             Unchanged on exit.

    CY       (input) COMPLEX*16 array, dimension at least
             ( 1 + ( N - 1 )*abs( INCY ) ).
             Before entry, the incremented array CY must contain the n
             element vector cy. On exit, CY is overwritten by the updated
             vector cy.

    INCY     (input) INTEGER
             On entry, INCY specifies the increment for the elements of
             CY. INCY must not be zero.
             Unchanged on exit.

    C        (input) DOUBLE PRECISION
             On entry, C specifies the cosine, cos.
             Unchanged on exit.

    S        (input) DOUBLE PRECISION
             On entry, S specifies the sine, sin.
             Unchanged on exit.

   =====================================================================
*/


    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    // 检查向量长度是否有效，若无效则直接返回
    if (*n <= 0) {
    return 0;
    }
    // 检查增量是否为1，如果是则跳转到 L20 处继续执行
    if (*incx == 1 && *incy == 1) {
    goto L20;
    }

/*
          code for unequal increments or equal increments not equal
            to 1
*/

    // 根据增量大小确定起始位置
    ix = 1;
    iy = 1;
    if (*incx < 0) {
    ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
    iy = (-(*n) + 1) * *incy + 1;
    }
    // 执行循环，应用平面旋转
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = ix;
    z__2.r = *c__ * cx[i__2].r, z__2.i = *c__ * cx[i__2].i;
    i__3 = iy;
    z__3.r = *s * cy[i__3].r, z__3.i = *s * cy[i__3].i;
    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
    ctemp.r = z__1.r, ctemp.i = z__1.i;
    i__2 = iy;
    i__3 = iy;
    z__2.r = *c__ * cy[i__3].r, z__2.i = *c__ * cy[i__3].i;
    i__4 = ix;
    z__3.r = *s * cx[i__4].r, z__3.i = *s * cx[i__4].i;
    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
    # 将复数 z__1 的实部赋值给 cy[i__2].r，将虚部赋值给 cy[i__2].i
    cy[i__2].r = z__1.r, cy[i__2].i = z__1.i;
    # 将复数 ctemp 的实部赋值给 cx[i__2].r，将虚部赋值给 cx[i__2].i
    i__2 = ix;
    cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
    # 增加 ix 的值，以便下一次循环访问下一个元素
    ix += *incx;
    # 增加 iy 的值，以确保正确遍历另一个数组的元素
    iy += *incy;
/* L10: */
    }
    return 0;

/*        code for both increments equal to 1 */

L20:
    i__1 = *n;
    // 循环遍历数组，对每个元素进行操作
    for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = i__;
    // 计算新的复数值，使用乘法运算
    z__2.r = *c__ * cx[i__2].r, z__2.i = *c__ * cx[i__2].i;
    i__3 = i__;
    z__3.r = *s * cy[i__3].r, z__3.i = *s * cy[i__3].i;
    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
    // 临时存储计算结果
    ctemp.r = z__1.r, ctemp.i = z__1.i;
    i__2 = i__;
    i__3 = i__;
    z__2.r = *c__ * cy[i__3].r, z__2.i = *c__ * cy[i__3].i;
    i__4 = i__;
    z__3.r = *s * cx[i__4].r, z__3.i = *s * cx[i__4].i;
    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
    // 更新数组元素值
    cy[i__2].r = z__1.r, cy[i__2].i = z__1.i;
    i__2 = i__;
    cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
/* L30: */
    }
    return 0;
} /* zdrot_ */

/* Subroutine */ int zdscal_(integer *n, doublereal *da, doublecomplex *zx,
    integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublecomplex z__1, z__2;

    /* Local variables */
    static integer i__, ix;


/*
    Purpose
    =======

       ZDSCAL scales a vector by a constant.

    Further Details
    ===============

       jack dongarra, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

    /* Parameter adjustments */
    --zx;

    /* Function Body */
    // 如果 n 或者 incx 小于等于 0，则直接返回
    if (*n <= 0 || *incx <= 0) {
    return 0;
    }
    // 如果 incx 等于 1，则跳转到 L20 标签处继续执行
    if (*incx == 1) {
    goto L20;
    }

/*        code for increment not equal to 1 */

    // 初始化 ix 为 1
    ix = 1;
    // 循环遍历数组，对每个元素进行操作
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = ix;
    // 计算新的复数值，使用乘法运算
    z__2.r = *da, z__2.i = 0.;
    i__3 = ix;
    z__1.r = z__2.r * zx[i__3].r - z__2.i * zx[i__3].i, z__1.i = z__2.r *
        zx[i__3].i + z__2.i * zx[i__3].r;
    // 更新数组元素值
    zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;
    // 更新 ix 的值
    ix += *incx;
/* L10: */
    }
    return 0;

/*        code for increment equal to 1 */

L20:
    // 循环遍历数组，对每个元素进行操作
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
    i__2 = i__;
    // 计算新的复数值，使用乘法运算
    z__2.r = *da, z__2.i = 0.;
    i__3 = i__;
    z__1.r = z__2.r * zx[i__3].r - z__2.i * zx[i__3].i, z__1.i = z__2.r *
        zx[i__3].i + z__2.i * zx[i__3].r;
    // 更新数组元素值
    zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;
/* L30: */
    }
    return 0;
} /* zdscal_ */

/* Subroutine */ int zgemm_(char *transa, char *transb, integer *m, integer *
    n, integer *k, doublecomplex *alpha, doublecomplex *a, integer *lda,
    doublecomplex *b, integer *ldb, doublecomplex *beta, doublecomplex *
    c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3, i__4, i__5, i__6;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    static integer i__, j, l, info;
    static logical nota, notb;
    static doublecomplex temp;
    static logical conja, conjb;
    extern logical lsame_(char *, char *);
    static integer nrowa, nrowb;
    extern /* Subroutine */ int xerbla_(char *, integer *);


    # 定义一个外部子程序 xerbla_，该子程序接受一个字符指针和一个整数指针作为参数
"""
Purpose
=======
ZGEMM performs one of the matrix-matrix operations

   C := alpha*op( A )*op( B ) + beta*C,

where op( X ) is one of

   op( X ) = X   or   op( X ) = X'   or   op( X ) = conjg( X' ),

alpha and beta are scalars, and A, B, and C are matrices, with op( A )
an m by k matrix, op( B ) a k by n matrix, and C an m by n matrix.

Arguments
=========
TRANSA - CHARACTER*1.
         On entry, TRANSA specifies the form of op( A ) to be used in
         the matrix multiplication as follows:

            TRANSA = 'N' or 'n',  op( A ) = A.

            TRANSA = 'T' or 't',  op( A ) = A'.

            TRANSA = 'C' or 'c',  op( A ) = conjg( A' ).

         Unchanged on exit.

TRANSB - CHARACTER*1.
         On entry, TRANSB specifies the form of op( B ) to be used in
         the matrix multiplication as follows:

            TRANSB = 'N' or 'n',  op( B ) = B.

            TRANSB = 'T' or 't',  op( B ) = B'.

            TRANSB = 'C' or 'c',  op( B ) = conjg( B' ).

         Unchanged on exit.

M      - INTEGER.
         On entry, M specifies the number of rows of the matrix
         op( A ) and of the matrix C. M must be at least zero.
         Unchanged on exit.

N      - INTEGER.
         On entry, N specifies the number of columns of the matrix
         op( B ) and the number of columns of the matrix C. N must be
         at least zero.
         Unchanged on exit.

K      - INTEGER.
         On entry, K specifies the number of columns of the matrix
         op( A ) and the number of rows of the matrix op( B ). K must
    B      - COMPLEX*16       array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.
    ! B 是一个复数数组，维度为 (LDB, kb)，其中 kb 的取值为：
    ! - 当 TRANSB = 'N' 或 'n' 时，kb = n；
    ! - 否则，kb = k。
    ! 进入函数时，如果 TRANSB = 'N' 或 'n'，则数组 B 的前 k 行 n 列必须包含矩阵 B；
    ! 否则，数组 B 的前 n 行 k 列必须包含矩阵 B。
    ! 函数执行完毕后，数组 B 保持不变。

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.
    ! LDB 是一个整数，指定数组 B 在调用程序中的第一个维度大小。
    ! 进入函数时，如果 TRANSB = 'N' 或 'n'，则 LDB 必须至少为 max(1, k)；
    ! 否则，LDB 必须至少为 max(1, n)。
    ! 函数执行完毕后，LDB 保持不变。

    BETA   - COMPLEX*16      .
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.
    ! BETA 是一个复数，用于指定标量 beta。
    ! 进入函数时，如果 BETA 被设为零，则不需要设置输入的矩阵 C。
    ! 函数执行完毕后，BETA 保持不变。

    C      - COMPLEX*16       array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).
    ! C 是一个复数数组，维度为 (LDC, n)。
    ! 进入函数时，如果 beta 不为零，则数组 C 的前 m 行 n 列必须包含矩阵 C；
    ! 否则，在输入时不需要设置数组 C。
    ! 函数执行完毕后，数组 C 被 alpha*op(A)*op(B) + beta*C 覆盖。

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.
    ! LDC 是一个整数，指定数组 C 在调用程序中的第一个维度大小。
    ! 进入函数时，LDC 必须至少为 max(1, m)。
    ! 函数执行完毕后，LDC 保持不变。

    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

    =====================================================================


       Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
       conjugated or transposed, set  CONJA and CONJB  as true if  A  and
       B  respectively are to be  transposed but  not conjugated  and set
       NROWA and  NROWB  as the number of rows and  columns  of  A
       and the number of rows of  B  respectively.
    ! 设置 NOTA 和 NOTB 为真，如果 A 和 B 分别未共轭或转置；
    ! 设置 CONJA 和 CONJB 为真，如果 A 和 B 分别要转置但不共轭；
    ! 设置 NROWA 和 NROWB 分别为 A 的行数和列数，以及 B 的行数。
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    nota = lsame_(transa, "N");
    notb = lsame_(transb, "N");
    conja = lsame_(transa, "C");
    conjb = lsame_(transb, "C");
    if (nota) {
        nrowa = *m;
    } else {
        nrowa = *k;
    }
    if (notb) {
        nrowb = *k;
    } else {
        nrowb = *n;
    }

/*     Test the input parameters. */

    info = 0;
    if (! nota && ! conja && ! lsame_(transa, "T")) {
        info = 1;
    } else if (! notb && ! conjb && ! lsame_(transb, "T")) {
        info = 2;
    } else if (*m < 0) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*k < 0) {
        info = 5;
    } else if (*lda < max(1,nrowa)) {
        info = 8;
    } else if (*ldb < max(1,nrowb)) {
        info = 10;
    } else if (*ldc < max(1,*m)) {
        info = 13;
    }
    if (info != 0) {
        xerbla_("ZGEMM ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (alpha->r == 0. && alpha->i == 0. || *k == 0) &&
         (beta->r == 1. && beta->i == 0.)) {
        return 0;
    }

/*     And when  alpha.eq.zero. */

    if (alpha->r == 0. && alpha->i == 0.) {
        if (beta->r == 0. && beta->i == 0.) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * c_dim1;
                    c__[i__3].r = 0., c__[i__3].i = 0.;
                }
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                        z__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                }
            }
        }
        return 0;
    }

/*     Start the operations. */

    if (notb) {
        if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (beta->r == 0. && beta->i == 0.) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0., c__[i__3].i = 0.;
                    }
                } else if (beta->r != 1. || beta->i != 0.) {
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                            z__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                        c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                    }
                }
            }
            return 0;
        }
    }
/* L60: */
/* 循环结束 */
            }
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            i__3 = l + j * b_dim1;
            if (b[i__3].r != 0. || b[i__3].i != 0.) {
            i__3 = l + j * b_dim1;
            z__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i,
                z__1.i = alpha->r * b[i__3].i + alpha->i * b[
                i__3].r;
            temp.r = z__1.r, temp.i = z__1.i;
            i__3 = *m;
            for (i__ = 1; i__ <= i__3; ++i__) {
                i__4 = i__ + j * c_dim1;
                i__5 = i__ + j * c_dim1;
                i__6 = i__ + l * a_dim1;
                z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                    z__2.i = temp.r * a[i__6].i + temp.i * a[
                    i__6].r;
                z__1.r = c__[i__5].r + z__2.r, z__1.i = c__[i__5]
                    .i + z__2.i;
                c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
/* L70: */
            }
            }
/* L80: */
        }
/* L90: */
        }
    } else if (conja) {

/*           Form  C := alpha*conjg( A' )*B + beta*C. */

        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            temp.r = 0., temp.i = 0.;
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);
            i__4 = l + j * b_dim1;
            z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4].i,
                z__2.i = z__3.r * b[i__4].i + z__3.i * b[i__4]
                .r;
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
/* L100: */
            }
            if (beta->r == 0. && beta->i == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                z__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            z__2.r = alpha->r * temp.r - alpha->i * temp.i,
                z__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, z__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            }
/* L110: */
        }
/* L120: */
        }
    } else {
/*           Form  C := alpha*A'*B + beta*C */

        /* 循环遍历 C 的列 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            /* 循环遍历 C 的行 */
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                /* 初始化临时变量 temp */
                temp.r = 0., temp.i = 0.;
                /* 循环遍历 A 的列或者 B 的行 */
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    /* 计算乘积并将结果加到 temp */
                    i__4 = l + i__ * a_dim1;
                    i__5 = l + j * b_dim1;
                    z__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5].i, z__2.i = a[i__4].r * b[i__5].i + a[i__4].i * b[i__5].r;
                    z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
                    temp.r = z__1.r, temp.i = z__1.i;
                    /* L130: 用于结束内层循环 */
                }
                /* 根据 beta 的值更新 C */
                if (beta->r == 0. && beta->i == 0.) {
                    i__3 = i__ + j * c_dim1;
                    z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                        z__1.i = alpha->r * temp.i + alpha->i * temp.r;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                } else {
                    i__3 = i__ + j * c_dim1;
                    z__2.r = alpha->r * temp.r - alpha->i * temp.i,
                        z__2.i = alpha->r * temp.i + alpha->i * temp.r;
                    i__4 = i__ + j * c_dim1;
                    z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                        z__3.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                }
                /* L140: 用于结束外层行循环 */
            }
            /* L150: 用于结束外层列循环 */
        }
    }
    } else if (nota) {
    if (conjb) {

/*           Form  C := alpha*A*conjg( B' ) + beta*C. */

        /* 循环遍历 C 的列 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            /* 根据 beta 的值初始化 C 的值 */
            if (beta->r == 0. && beta->i == 0.) {
                /* 循环遍历 C 的行 */
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * c_dim1;
                    c__[i__3].r = 0., c__[i__3].i = 0.;
                    /* L160: 用于结束内层循环 */
                }
            } else if (beta->r != 1. || beta->i != 0.) {
                /* 循环遍历 C 的行 */
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4].i,
                        z__1.i = beta->r * c__[i__4].i + beta->i * c__[i__4].r;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                    /* L170: 用于结束内层循环 */
                }
            }
            /* L180: 用于结束外层列循环 */
        }
        /* L190: 用于结束 else if 块 */
    }
    /* L200: 用于结束 if 块 */
}
/* L170: */
            }
        }
        // 循环遍历 k，执行下面的操作
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            // 计算数组中的索引
            i__3 = j + l * b_dim1;
            // 检查 b 元素的实部和虚部是否都为零
            if (b[i__3].r != 0. || b[i__3].i != 0.) {
                // 计算乘法的复共轭
                d_cnjg(&z__2, &b[j + l * b_dim1]);
                z__1.r = alpha->r * z__2.r - alpha->i * z__2.i,
                    z__1.i = alpha->r * z__2.i + alpha->i *
                    z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
                // 循环遍历 m，执行下面的操作
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    // 计算数组中的索引
                    i__4 = i__ + j * c_dim1;
                    i__5 = i__ + j * c_dim1;
                    i__6 = i__ + l * a_dim1;
                    // 计算乘法
                    z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                        z__2.i = temp.r * a[i__6].i + temp.i * a[
                        i__6].r;
                    z__1.r = c__[i__5].r + z__2.r, z__1.i = c__[i__5]
                        .i + z__2.i;
                    c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
/* L180: */
                }
            }
/* L190: */
        }
/* L200: */
        }
    } else {

/*           Form  C := alpha*A*B'          + beta*C */

        // 循环遍历 n，执行下面的操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 检查 beta 是否为零
            if (beta->r == 0. && beta->i == 0.) {
                // 循环遍历 m，执行下面的操作
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 计算数组中的索引
                    i__3 = i__ + j * c_dim1;
                    // 设置 c__ 数组元素为零
                    c__[i__3].r = 0., c__[i__3].i = 0.;
/* L210: */
                }
            } else if (beta->r != 1. || beta->i != 0.) {
                // 循环遍历 m，执行下面的操作
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 计算数组中的索引
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    // 计算乘法
                    z__1.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                        .i, z__1.i = beta->r * c__[i__4].i + beta->i *
                         c__[i__4].r;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L220: */
                }
            }
            // 循环遍历 k，执行下面的操作
            i__2 = *k;
            for (l = 1; l <= i__2; ++l) {
                // 计算数组中的索引
                i__3 = j + l * b_dim1;
                // 检查 b 元素的实部和虚部是否都为零
                if (b[i__3].r != 0. || b[i__3].i != 0.) {
                    i__3 = j + l * b_dim1;
                    // 计算乘法
                    z__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i,
                        z__1.i = alpha->r * b[i__3].i + alpha->i * b[
                        i__3].r;
                    temp.r = z__1.r, temp.i = z__1.i;
                    // 循环遍历 m，执行下面的操作
                    i__3 = *m;
                    for (i__ = 1; i__ <= i__3; ++i__) {
                        // 计算数组中的索引
                        i__4 = i__ + j * c_dim1;
                        i__5 = i__ + j * c_dim1;
                        i__6 = i__ + l * a_dim1;
                        // 计算乘法
                        z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,
                            z__2.i = temp.r * a[i__6].i + temp.i * a[
                            i__6].r;
                        z__1.r = c__[i__5].r + z__2.r, z__1.i = c__[i__5]
                            .i + z__2.i;
                        c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
/* L230: */
                    }
                }
/* L240: */
            }
/* L250: */
        }
    }
    } else if (conja) {
    if (conjb) {
/*           Form  C := alpha*conjg( A' )*conjg( B' ) + beta*C. */

        // 循环遍历 C 的列
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 循环遍历 C 的行
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 初始化临时变量 temp 为复数 0
            temp.r = 0., temp.i = 0.;
            // 循环遍历 A' 的列
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 A' 和 B' 的共轭乘积
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);
            d_cnjg(&z__4, &b[j + l * b_dim1]);
            z__2.r = z__3.r * z__4.r - z__3.i * z__4.i, z__2.i =
                z__3.r * z__4.i + z__3.i * z__4.r;
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
/* L260: */
            }
            // 根据 beta 是否为 0，计算 C 的元素
            if (beta->r == 0. && beta->i == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                z__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            z__2.r = alpha->r * temp.r - alpha->i * temp.i,
                z__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, z__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            }
/* L270: */
        }
/* L280: */
        }
    } else {

/*           Form  C := alpha*conjg( A' )*B' + beta*C */

        // 循环遍历 C 的列
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 循环遍历 C 的行
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            // 初始化临时变量 temp 为复数 0
            temp.r = 0., temp.i = 0.;
            // 循环遍历 A' 的列
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            // 计算 A' 和 B' 的共轭乘积
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);
            i__4 = j + l * b_dim1;
            z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4].i,
                z__2.i = z__3.r * b[i__4].i + z__3.i * b[i__4]
                .r;
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
/* L290: */
            }
            // 根据 beta 是否为 0，计算 C 的元素
            if (beta->r == 0. && beta->i == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                z__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            z__2.r = alpha->r * temp.r - alpha->i * temp.i,
                z__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, z__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            }
/* L300: */
        }
/* L310: */
        }
    }
/* L310: */
        }
    }
    } else {
    if (conjb) {

/*           Form  C := alpha*A'*conjg( B' ) + beta*C */

        i__1 = *n;  // 设置循环上限为 n
        for (j = 1; j <= i__1; ++j) {  // 遍历 C 的列索引 j
        i__2 = *m;  // 设置循环上限为 m
        for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历 C 的行索引 i
            temp.r = 0., temp.i = 0.;  // 初始化临时变量 temp 为复数 0
            i__3 = *k;  // 设置循环上限为 k
            for (l = 1; l <= i__3; ++l) {  // 遍历求和项 l
            i__4 = l + i__ * a_dim1;  // 计算 A 的元素索引
            d_cnjg(&z__3, &b[j + l * b_dim1]);  // 计算 B 的共轭转置元素
            z__2.r = a[i__4].r * z__3.r - a[i__4].i * z__3.i,
                z__2.i = a[i__4].r * z__3.i + a[i__4].i *
                z__3.r;  // 计算乘积的实部和虚部
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;  // 更新 temp
            temp.r = z__1.r, temp.i = z__1.i;
/* L320: */
            }
            if (beta->r == 0. && beta->i == 0.) {  // 如果 beta 为零
            i__3 = i__ + j * c_dim1;
            z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                z__1.i = alpha->r * temp.i + alpha->i *
                temp.r;  // 计算 alpha * temp
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;  // 更新 C 的元素
            } else {  // 否则
            i__3 = i__ + j * c_dim1;
            z__2.r = alpha->r * temp.r - alpha->i * temp.i,
                z__2.i = alpha->r * temp.i + alpha->i *
                temp.r;  // 计算 alpha * temp
            i__4 = i__ + j * c_dim1;
            z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, z__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;  // 计算 beta * C 的元素
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;  // 更新 C 的元素
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            }
/* L330: */
        }
/* L340: */
        }
    } else {

/*           Form  C := alpha*A'*B' + beta*C */

        i__1 = *n;  // 设置循环上限为 n
        for (j = 1; j <= i__1; ++j) {  // 遍历 C 的列索引 j
        i__2 = *m;  // 设置循环上限为 m
        for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历 C 的行索引 i
            temp.r = 0., temp.i = 0.;  // 初始化临时变量 temp 为复数 0
            i__3 = *k;  // 设置循环上限为 k
            for (l = 1; l <= i__3; ++l) {  // 遍历求和项 l
            i__4 = l + i__ * a_dim1;  // 计算 A 的元素索引
            i__5 = j + l * b_dim1;  // 计算 B 的元素索引
            z__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5]
                .i, z__2.i = a[i__4].r * b[i__5].i + a[i__4]
                .i * b[i__5].r;  // 计算乘积的实部和虚部
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;  // 更新 temp
            temp.r = z__1.r, temp.i = z__1.i;
/* L350: */
            }
            if (beta->r == 0. && beta->i == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                z__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            z__2.r = alpha->r * temp.r - alpha->i * temp.i,
                z__2.i = alpha->r * temp.i + alpha->i *
                temp.r;
            i__4 = i__ + j * c_dim1;
            z__3.r = beta->r * c__[i__4].r - beta->i * c__[i__4]
                .i, z__3.i = beta->r * c__[i__4].i + beta->i *
                 c__[i__4].r;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            }
/* L360: */
        }
/* L370: */
        }
    }
    }

    return 0;

/*     End of ZGEMM . */

} /* zgemm_ */

/* Subroutine */ int zgemv_(char *trans, integer *m, integer *n,
    doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
    x, integer *incx, doublecomplex *beta, doublecomplex *y, integer *
    incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static doublecomplex temp;
    static integer lenx, leny;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj;


/*
    Purpose
    =======

    ZGEMV  performs one of the matrix-vector operations

       y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or

       y := alpha*conjg( A' )*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ==========

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.

                TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.

                TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16      .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients.
             Unchanged on exit.



注释：
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    X      - COMPLEX*16       array of DIMENSION at least
             ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
             Before entry, the incremented array X must contain the
             vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - COMPLEX*16      .
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - COMPLEX*16       array of DIMENSION at least
             ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
             Before entry with BETA non-zero, the incremented array Y
             must contain the vector y. On exit, Y is overwritten by the
             updated vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================


       Test the input parameters.
    /* 参数调整 */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* 函数主体 */
    info = 0;
    if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")
        ) {
        info = 1;
    } else if (*m < 0) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*lda < max(1,*m)) {
        info = 6;
    } else if (*incx == 0) {
        info = 8;
    } else if (*incy == 0) {
        info = 11;
    }
    if (info != 0) {
        xerbla_("ZGEMV ", &info);
        return 0;
    }

/*     如果可能的话，进行快速返回。 */

    if (*m == 0 || *n == 0 || alpha->r == 0. && alpha->i == 0. && (beta->r ==
        1. && beta->i == 0.)) {
        return 0;
    }

    noconj = lsame_(trans, "T");

/*
       设置 LENX 和 LENY，向量 x 和 y 的长度，并设置 X 和 Y 的起始点。
*/

    if (lsame_(trans, "N")) {
        lenx = *n;
        leny = *m;
    } else {
        lenx = *m;
        leny = *n;
    }
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (leny - 1) * *incy;
    }

/*
       开始操作。在这个版本中，通过一次对 A 的顺序访问其元素。

       首先形成 y := beta*y。
*/

    if (beta->r != 1. || beta->i != 0.) {
        if (*incy == 1) {
            if (beta->r == 0. && beta->i == 0.) {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = i__;
                    y[i__2].r = 0., y[i__2].i = 0.;
                }
            } else {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = i__;
                    i__3 = i__;
                    z__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                        z__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
                        .r;
                    y[i__2].r = z__1.r, y[i__2].i = z__1.i;
                }
            }
        } else {
            iy = ky;
            if (beta->r == 0. && beta->i == 0.) {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = iy;
                    y[i__2].r = 0., y[i__2].i = 0.;
                    iy += *incy;
                }
            } else {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = iy;
                    i__3 = iy;
                    z__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                        z__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
                        .r;
                    y[i__2].r = z__1.r, y[i__2].i = z__1.i;
                    iy += *incy;
                }
            }
        }
    }
    if (alpha->r == 0. && alpha->i == 0.) {
        return 0;
    }
    if (lsame_(trans, "N")) {

/*        形成 y := alpha*A*x + y. */

        jx = kx;
    # 检查增量步长是否为1，如果是则执行以下循环
    if (*incy == 1) {
        # 循环遍历列索引 j，从 1 到 n
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            # 计算向量 x 的索引位置
            i__2 = jx;
            # 检查 x 向量在索引位置处是否非零
            if (x[i__2].r != 0. || x[i__2].i != 0.) {
                # 复数乘法运算，计算 alpha 和 x[jx] 的乘积
                i__2 = jx;
                z__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                    z__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2].r;
                # 将乘积结果存储到 temp 变量中
                temp.r = z__1.r, temp.i = z__1.i;
                # 循环遍历行索引 i，从 1 到 m
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    # 计算向量 y 的索引位置
                    i__3 = i__;
                    # 计算矩阵 a 的索引位置
                    i__4 = i__;
                    i__5 = i__ + j * a_dim1;
                    # 复数乘法运算，计算 temp 和 a[i__ + j * a_dim1] 的乘积
                    z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                        z__2.i = temp.r * a[i__5].i + temp.i * a[i__5].r;
                    # 复数加法运算，将结果加到向量 y[i__] 上
                    z__1.r = y[i__4].r + z__2.r, z__1.i = y[i__4].i + z__2.i;
                    y[i__3].r = z__1.r, y[i__3].i = z__1.i;
/* L50: */
            }
        }
        // 更新向量 x 的索引
        jx += *incx;
/* L60: */
        }
    } else {
        // 对每列 j 进行迭代
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 计算 x 的起始索引 i__2
        i__2 = jx;
        // 如果 x[i__2] 不为零，则执行下面的操作
        if (x[i__2].r != 0. || x[i__2].i != 0.) {
            // 计算 alpha*x[i__2] 的结果并存储在 temp 中
            i__2 = jx;
            z__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                z__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                .r;
            temp.r = z__1.r, temp.i = z__1.i;
            // 初始化向量 y 的起始索引 iy
            iy = ky;
            // 对每行 i 进行迭代
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            // 计算 y[iy] 的位置 i__3 和更新 y[iy]
            i__3 = iy;
            i__4 = iy;
            i__5 = i__ + j * a_dim1;
            z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                z__2.i = temp.r * a[i__5].i + temp.i * a[i__5]
                .r;
            z__1.r = y[i__4].r + z__2.r, z__1.i = y[i__4].i +
                z__2.i;
            y[i__3].r = z__1.r, y[i__3].i = z__1.i;
            // 更新 iy
            iy += *incy;
/* L70: */
            }
        }
        // 更新向量 x 的索引
        jx += *incx;
/* L80: */
        }
    }
    } else {

/*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y. */

    // 初始化向量 y 的起始索引 jy
    jy = ky;
    // 如果 x 的增量为 1
    if (*incx == 1) {
        // 对每列 j 进行迭代
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 初始化 temp
        temp.r = 0., temp.i = 0.;
        // 如果不取共轭
        if (noconj) {
            // 对每行 i 进行迭代
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            // 计算矩阵 A 的索引 i__3 和向量 x 的索引 i__4
            i__3 = i__ + j * a_dim1;
            i__4 = i__;
            z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[i__4]
                .i, z__2.i = a[i__3].r * x[i__4].i + a[i__3]
                .i * x[i__4].r;
            // 更新 temp
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
/* L90: */
            }
        } else {
            // 对每行 i 进行迭代
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            // 计算矩阵 A 的索引 i__3 和向量 x 的索引 i__4 的共轭
            d_cnjg(&z__3, &a[i__ + j * a_dim1]);
            i__3 = i__;
            z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                z__2.i = z__3.r * x[i__3].i + z__3.i * x[i__3]
                .r;
            // 更新 temp
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
/* L100: */
            }
        }
        // 计算 y[jy] 的位置 i__2 和更新 y[jy]
        i__2 = jy;
        i__3 = jy;
        z__2.r = alpha->r * temp.r - alpha->i * temp.i, z__2.i =
            alpha->r * temp.i + alpha->i * temp.r;
        z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
        // 更新 jy
        jy += *incy;
/* L110: */
        }
    } else {
        // 对每列 j 进行迭代
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 初始化 temp
        temp.r = 0., temp.i = 0.;
        // 初始化向量 x 的起始索引 ix
        ix = kx;
        // 如果不取共轭
        if (noconj) {
            // 对每行 i 进行迭代
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            // 计算矩阵 A 的索引 i__3 和向量 x 的索引 ix
            i__3 = i__ + j * a_dim1;
            i__4 = ix;
            z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[i__4]
                .i, z__2.i = a[i__3].r * x[i__4].i + a[i__3]
                .i * x[i__4].r;
            // 更新 temp
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
            // 更新 ix
            ix += *incx;
/* L120: */
            }
        }
        // 如果取共轭
        else {
            // 对每行 i 进行迭代
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            // 计算矩阵 A 的索引 i__3 和向量 x 的索引 ix 的共轭
            d_cnjg(&z__3, &a[i__ + j * a_dim1]);
            i__3 = ix;
            z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                z__2.i = z__3.r * x[i__3].i + z__3.i * x[i__3]
                .r;
            // 更新 temp
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
            // 更新 ix
            ix += *incx;
/* L130: */
            }
        }
        // 计算 y[jy] 的位置 i__2 和更新 y[jy]
        i__2 = jy;
        i__3 = jy;
        z__2.r = alpha->r * temp.r - alpha->i * temp.i, z__2.i =
            alpha->r * temp.i + alpha->i * temp.r;
        z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
        // 更新 jy
        jy += *incy;
/* L140: */
        }
    }
    }
/* L120: */
            }
        } else {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // Conjugate of a complex number a[i__ + j * a_dim1]
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                // Compute product of conjugate and x[ix], accumulate into temp
                i__3 = ix;
                z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                    z__2.i = z__3.r * x[i__3].i + z__3.i * x[i__3].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
                ix += *incx;
/* L130: */
            }
        }
        // Perform alpha * temp and accumulate into y[jy]
        i__2 = jy;
        i__3 = jy;
        z__2.r = alpha->r * temp.r - alpha->i * temp.i, z__2.i =
            alpha->r * temp.i + alpha->i * temp.r;
        z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
        jy += *incy;
/* L140: */
        }
    }
    }

    return 0;

/*     End of ZGEMV . */

} /* zgemv_ */

/* Subroutine */ int zgerc_(integer *m, integer *n, doublecomplex *alpha,
    doublecomplex *x, integer *incx, doublecomplex *y, integer *incy,
    doublecomplex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2;

    /* Local variables */
    static integer i__, j, ix, jy, kx, info;
    static doublecomplex temp;
    extern /* Subroutine */ int xerbla_(char *, integer *);


注释完成。
    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients. On exit, A is
             overwritten by the updated matrix.

A 是一个复数数组，类型为 COMPLEX*16，维度为 (LDA, n)。
在进入该子程序之前，数组 A 的前 m 行 n 列必须包含系数矩阵。退出时，A 将被更新后的矩阵覆盖。


    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

LDA 是一个整数。
进入该子程序时，LDA 指定了数组 A 的第一个维度，即在调用（子）程序中声明的尺寸。LDA 必须至少为 max(1, m)。
退出时，LDA 的值保持不变。


    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================

进一步细节
===============
这是一个 Level 2 Blas（Basic Linear Algebra Subprograms，基本线性代数子程序）例程。

-- 编写于 1986 年 10 月 22 日。
   Jack Dongarra，阿尔贡国家实验室。
   Jeremy Du Croz，NAG 中央办公室。
   Sven Hammarling，NAG 中央办公室。
   Richard Hanson，Sandia 国家实验室。

=====================================================================


       Test the input parameters.

测试输入参数。
    /* Parameter adjustments */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (*m < 0) {
        info = 1;
    } else if (*n < 0) {
        info = 2;
    } else if (*incx == 0) {
        info = 5;
    } else if (*incy == 0) {
        info = 7;
    } else if (*lda < max(1,*m)) {
        info = 9;
    }
    if (info != 0) {
        xerbla_("ZGERC ", &info);
        return 0;
    }

    /* Quick return if possible. */
    if (*m == 0 || *n == 0 || alpha->r == 0. && alpha->i == 0.) {
        return 0;
    }

    /*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
    */

    if (*incy > 0) {
        jy = 1;
    } else {
        jy = 1 - (*n - 1) * *incy;
    }
    if (*incx == 1) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = jy;
            if (y[i__2].r != 0. || y[i__2].i != 0.) {
                d_cnjg(&z__2, &y[jy]);
                z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i =
                    alpha->r * z__2.i + alpha->i * z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * a_dim1;
                    i__4 = i__ + j * a_dim1;
                    i__5 = i__;
                    z__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, z__2.i =
                         x[i__5].r * temp.i + x[i__5].i * temp.r;
                    z__1.r = a[i__4].r + z__2.r, z__1.i = a[i__4].i + z__2.i;
                    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
                    /* L10: */
                }
            }
            jy += *incy;
            /* L20: */
        }
    } else {
        if (*incx > 0) {
            kx = 1;
        } else {
            kx = 1 - (*m - 1) * *incx;
        }
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = jy;
            if (y[i__2].r != 0. || y[i__2].i != 0.) {
                d_cnjg(&z__2, &y[jy]);
                z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i =
                    alpha->r * z__2.i + alpha->i * z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
                ix = kx;
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    i__3 = i__ + j * a_dim1;
                    i__4 = i__ + j * a_dim1;
                    i__5 = ix;
                    z__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, z__2.i =
                         x[i__5].r * temp.i + x[i__5].i * temp.r;
                    z__1.r = a[i__4].r + z__2.r, z__1.i = a[i__4].i + z__2.i;
                    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
                    ix += *incx;
                    /* L30: */
                }
            }
            jy += *incy;
            /* L40: */
        }
    }

    return 0;

/*     End of ZGERC . */

} /* zgerc_ */

/* Subroutine */ int zgeru_(integer *m, integer *n, doublecomplex *alpha,
    doublecomplex *x, integer *incx, doublecomplex *y, integer *incy,
    doublecomplex *a, integer *lda)


注释完成。
    extern /* Subroutine */ int xerbla_(char *, integer *);



    # 声明外部子程序 xerbla_
    extern /* Subroutine */ int xerbla_(char *, integer *);


这行代码是Fortran语言中的声明外部子程序（external subroutine）。具体注释如下：

- `extern`: 声明这是一个外部子程序，即该子程序的实现在其他地方定义。
- `/* Subroutine */`: 表示这是一个子程序（即函数或过程）。
- `int xerbla_(char *, integer *);`: 声明了一个名为 `xerbla_` 的子程序，接受一个字符指针 `char *` 和一个整数指针 `integer *` 作为参数，并且返回一个整数 `int`。

在Fortran中，`extern` 关键字不是必须的，但在这里它是为了表明 `xerbla_` 子程序的实现位于其他地方，通常是在一个外部库或模块中定义的。
/*
    Purpose
    =======

    ZGERU  performs the rank 1 operation

       A := alpha*x*y' + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.

    Arguments
    ==========

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16      .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - COMPLEX*16       array of dimension at least
             ( 1 + ( m - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the m
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    Y      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y.
             Unchanged on exit.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients. On exit, A is
             overwritten by the updated matrix.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    Further Details
    ===============

    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.

    =====================================================================


       Test the input parameters.
*/

/* Parameter adjustments */
--x;                    // 将指针 x 向前移动，使其指向 x[1] 而非 x[0]
--y;                    // 将指针 y 向前移动，使其指向 y[1] 而非 y[0]
a_dim1 = *lda;          // 获取 lda 的值，它指定了 A 的第一维度的大小
a_offset = 1 + a_dim1;  // 计算 A 的偏移量
a -= a_offset;          // 将 A 的起始地址调整为正确的起始位置

/* Function Body */
info = 0;               // 初始化 info 为 0，用于记录错误信息编号
if (*m < 0) {           // 检查 M 的有效性，如果小于 0，设置 info 为 1
info = 1;
} else if (*n < 0) {    // 检查 N 的有效性，如果小于 0，设置 info 为 2
info = 2;
} else if (*incx == 0) {// 检查 INCX 的有效性，如果等于 0，设置 info 为 5
info = 5;
} else if (*incy == 0) {// 检查 INCY 的有效性，如果等于 0，设置 info 为 7
info = 7;
} else if (*lda < max(1,*m)) { // 检查 LDA 的有效性，如果小于 max(1, *m)，设置 info 为 9
info = 9;
}
if (info != 0) {        // 如果 info 不为 0，说明有错误发生，调用错误处理函数 xerbla_
xerbla_("ZGERU ", &info);
return 0;               // 返回，终止函数执行
}

/* Quick return if possible. */

if (*m == 0 || *n == 0 || alpha->r == 0. && alpha->i == 0.) {
// 如果 m 或 n 为 0，或者 alpha 的实部和虚部都为 0，直接返回，无需执行操作
return 0;
}
/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

if (*incy > 0) {
    // Determine the starting index for vector y based on the sign of incy
    jy = 1;
} else {
    jy = 1 - (*n - 1) * *incy;
}
if (*incx == 1) {
    // Case where vector x is accessed with a unit increment
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // Compute the complex scalar multiplication of alpha and y[jy]
        i__2 = jy;
        if (y[i__2].r != 0. || y[i__2].i != 0.) {
            i__2 = jy;
            z__1.r = alpha->r * y[i__2].r - alpha->i * y[i__2].i, z__1.i =
                 alpha->r * y[i__2].i + alpha->i * y[i__2].r;
            temp.r = z__1.r, temp.i = z__1.i;
            // Perform the rank-1 update of the matrix A
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * a_dim1;
                i__4 = i__ + j * a_dim1;
                i__5 = i__;
                z__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, z__2.i =
                     x[i__5].r * temp.i + x[i__5].i * temp.r;
                z__1.r = a[i__4].r + z__2.r, z__1.i = a[i__4].i + z__2.i;
                a[i__3].r = z__1.r, a[i__3].i = z__1.i;
            }
        }
        // Increment jy for the next iteration
        jy += *incy;
    }
} else {
    // Case where vector x is accessed with a non-unit increment
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (*m - 1) * *incx;
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        // Compute the complex scalar multiplication of alpha and y[jy]
        i__2 = jy;
        if (y[i__2].r != 0. || y[i__2].i != 0.) {
            i__2 = jy;
            z__1.r = alpha->r * y[i__2].r - alpha->i * y[i__2].i, z__1.i =
                 alpha->r * y[i__2].i + alpha->i * y[i__2].r;
            temp.r = z__1.r, temp.i = z__1.i;
            ix = kx;
            // Perform the rank-1 update of the matrix A
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * a_dim1;
                i__4 = i__ + j * a_dim1;
                i__5 = ix;
                z__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, z__2.i =
                     x[i__5].r * temp.i + x[i__5].i * temp.r;
                z__1.r = a[i__4].r + z__2.r, z__1.i = a[i__4].i + z__2.i;
                a[i__3].r = z__1.r, a[i__3].i = z__1.i;
                // Increment ix for the next iteration
                ix += *incx;
            }
        }
        // Increment jy for the next iteration
        jy += *incy;
    }
}

return 0;

/*     End of ZGERU . */

} /* zgeru_ */



/* Subroutine */ int zhemv_(char *uplo, integer *n, doublecomplex *alpha,
    doublecomplex *a, integer *lda, doublecomplex *x, integer *incx,
    doublecomplex *beta, doublecomplex *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static doublecomplex temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*
    Purpose
    =======

    ZHEMV  performs the matrix-vector  operation

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.

    Arguments
    ==========
*/

/*
   Detailed description

    1)      If UPLO = 'U', ZHEMV computes y := alpha*A*x + beta*y,
            where alpha and beta are scalars, x and y are n element
            vectors and A is an n by n hermitian matrix, supplied in
            packed form.
    # 字符型变量，指定矩阵 A 的上三角或下三角部分的参考方式
    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:
    
                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.
    
                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.
    
             Unchanged on exit.
    
    # 整数型变量，矩阵 A 的阶数
    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.
    
    # 双精度复数型变量，指定标量 alpha
    ALPHA  - COMPLEX*16      .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.
    
    # 双精度复数型数组，维度为 (LDA, n)，存储 Hermitian 矩阵 A 的数据
    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the hermitian matrix and the strictly
             lower triangular part of A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the hermitian matrix and the strictly
             upper triangular part of A is not referenced.
             Note that the imaginary parts of the diagonal elements need
             not be set and are assumed to be zero.
             Unchanged on exit.
    
    # 整数型变量，指定数组 A 的第一个维度大小
    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.
    
    # 双精度复数型数组，至少维度为 ( 1 + ( n - 1 )*abs( INCX ) )，存储向量 x
    X      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.
    
    # 整数型变量，指定数组 X 中元素的增量
    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.
    
    # 双精度复数型变量，指定标量 beta
    BETA   - COMPLEX*16      .
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.
    
    # 双精度复数型数组，至少维度为 ( 1 + ( n - 1 )*abs( INCY ) )，存储向量 y
    Y      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y. On exit, Y is overwritten by the updated
             vector y.
             Unchanged on exit.
    
    # 整数型变量，指定数组 Y 中元素的增量
    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.
    
    Further Details
    ===============
    Level 2 Blas routine.
    # 以下是给定代码的注释
    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (*n < 0) {
        info = 2;
    } else if (*lda < max(1,*n)) {
        info = 5;
    } else if (*incx == 0) {
        info = 7;
    } else if (*incy == 0) {
        info = 10;
    }
    if (info != 0) {
        xerbla_("ZHEMV ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || alpha->r == 0. && alpha->i == 0. && (beta->r == 1. &&
        beta->i == 0.)) {
        return 0;
    }

/*     Set up the start points in  X  and  Y. */

    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (*n - 1) * *incx;
    }
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (*n - 1) * *incy;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.

       First form  y := beta*y.
*/

    if (beta->r != 1. || beta->i != 0.) {
        if (*incy == 1) {
            /* Handle the case where incy is 1 */
            if (beta->r == 0. && beta->i == 0.) {
                /* Set y to zero */
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = i__;
                    y[i__2].r = 0., y[i__2].i = 0.;
                }
            } else {
                /* Scale y by beta */
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = i__;
                    i__3 = i__;
                    z__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                        z__1.i = beta->r * y[i__3].i + beta->i * y[i__3].r;
                    y[i__2].r = z__1.r, y[i__2].i = z__1.i;
                }
            }
        } else {
            /* Handle the case where incy is not 1 */
            iy = ky;
            if (beta->r == 0. && beta->i == 0.) {
                /* Set y to zero */
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = iy;
                    y[i__2].r = 0., y[i__2].i = 0.;
                    iy += *incy;
                }
            } else {
                /* Scale y by beta */
                i__1 = *n;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    i__2 = iy;
                    i__3 = iy;
                    z__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i,
                        z__1.i = beta->r * y[i__3].i + beta->i * y[i__3].r;
                    y[i__2].r = z__1.r, y[i__2].i = z__1.i;
                    iy += *incy;
                }
            }
        }
    }

    /* Check if alpha is zero for early return */
    if (alpha->r == 0. && alpha->i == 0.) {
        return 0;
    }

    /* Process upper triangular part of A */
    if (lsame_(uplo, "U")) {

/*        Form  y  when A is stored in upper triangle. */
    // 如果增量指针指向的值为1，并且两个增量指针值相等（即增量为1）
    if (*incx == 1 && *incy == 1) {
        // 循环迭代每一个列向量元素
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 计算 alpha 和 x 向量元素之间的复数乘法
            i__2 = j;
            z__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, z__1.i =
                 alpha->r * x[i__2].i + alpha->i * x[i__2].r;
            temp1.r = z__1.r, temp1.i = z__1.i;
            // 初始化 temp2 为零
            temp2.r = 0., temp2.i = 0.;
            // 循环迭代当前列 j 之前的每一个行元素
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__;
                i__4 = i__;
                // 计算 temp1 与当前矩阵元素的乘积，累加到向量 y 的对应元素
                i__5 = i__ + j * a_dim1;
                z__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                    z__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5]
                    .r;
                z__1.r = y[i__4].r + z__2.r, z__1.i = y[i__4].i + z__2.i;
                y[i__3].r = z__1.r, y[i__3].i = z__1.i;
                // 计算当前矩阵元素的共轭乘积与 x 向量元素的乘积，累加到 temp2
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                i__3 = i__;
                z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, z__2.i =
                     z__3.r * x[i__3].i + z__3.i * x[i__3].r;
                z__1.r = temp2.r + z__2.r, z__1.i = temp2.i + z__2.i;
                temp2.r = z__1.r, temp2.i = z__1.i;
/* L50: */
        }
        // 处理上三角矩阵的情况
        i__2 = j;
        i__3 = j;
        i__4 = j + j * a_dim1;
        // 提取矩阵元素 a(j, j)
        d__1 = a[i__4].r;
        // 计算临时变量 z__3
        z__3.r = d__1 * temp1.r, z__3.i = d__1 * temp1.i;
        // 计算临时变量 z__2
        z__2.r = y[i__3].r + z__3.r, z__2.i = y[i__3].i + z__3.i;
        // 计算临时变量 z__4
        z__4.r = alpha->r * temp2.r - alpha->i * temp2.i, z__4.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        // 计算结果并赋值给 y(j)
        z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
/* L60: */
        }
    } else {
        // 处理下三角矩阵的情况
        jx = kx;
        jy = ky;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = jx;
        // 计算临时变量 z__1
        z__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, z__1.i =
             alpha->r * x[i__2].i + alpha->i * x[i__2].r;
        // 将结果赋给 temp1
        temp1.r = z__1.r, temp1.i = z__1.i;
        // 初始化 temp2
        temp2.r = 0., temp2.i = 0.;
        ix = kx;
        iy = ky;
        i__2 = j - 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = iy;
            i__4 = iy;
            i__5 = i__ + j * a_dim1;
            // 计算临时变量 z__2
            z__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                z__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5]
                .r;
            // 计算结果并赋值给 y(i__)
            z__1.r = y[i__4].r + z__2.r, z__1.i = y[i__4].i + z__2.i;
            y[i__3].r = z__1.r, y[i__3].i = z__1.i;
            // 对 a(i, j) 进行共轭并与 x(ix) 的乘积计算，结果赋给 z__2
            d_cnjg(&z__3, &a[i__ + j * a_dim1]);
            i__3 = ix;
            z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, z__2.i =
                 z__3.r * x[i__3].i + z__3.i * x[i__3].r;
            // 计算结果并赋给 temp2
            z__1.r = temp2.r + z__2.r, z__1.i = temp2.i + z__2.i;
            temp2.r = z__1.r, temp2.i = z__1.i;
            // 更新 ix 和 iy
            ix += *incx;
            iy += *incy;
/* L70: */
        }
        i__2 = jy;
        i__3 = jy;
        i__4 = j + j * a_dim1;
        // 计算矩阵元素 a(j, j)
        d__1 = a[i__4].r;
        // 计算临时变量 z__3
        z__3.r = d__1 * temp1.r, z__3.i = d__1 * temp1.i;
        // 计算临时变量 z__2
        z__2.r = y[i__3].r + z__3.r, z__2.i = y[i__3].i + z__3.i;
        // 计算临时变量 z__4
        z__4.r = alpha->r * temp2.r - alpha->i * temp2.i, z__4.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        // 计算结果并赋值给 y(jy)
        z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
        // 更新 jx 和 jy
        jx += *incx;
        jy += *incy;
/* L80: */
        }
    }
    } else {

/*        Form  y  when A is stored in lower triangle. */
    # 检查是否增量为1，即 x 和 y 的步长为1
    if (*incx == 1 && *incy == 1) {
        # 循环遍历从 1 到 n 的所有列 j
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            # 计算 alpha 与 x[j] 的乘积并赋值给 temp1
            i__2 = j;
            z__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, z__1.i =
                 alpha->r * x[i__2].i + alpha->i * x[i__2].r;
            temp1.r = z__1.r, temp1.i = z__1.i;
            # 初始化 temp2 为零向量
            temp2.r = 0., temp2.i = 0.;
            # 计算更新 y[j]
            i__2 = j;
            i__3 = j;
            i__4 = j + j * a_dim1;
            d__1 = a[i__4].r;
            z__2.r = d__1 * temp1.r, z__2.i = d__1 * temp1.i;
            z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
            y[i__2].r = z__1.r, y[i__2].i = z__1.i;
            # 循环遍历从 j+1 到 n 的所有行 i
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                # 计算更新 y[i]
                i__3 = i__;
                i__4 = i__;
                i__5 = i__ + j * a_dim1;
                z__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                    z__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5]
                    .r;
                z__1.r = y[i__4].r + z__2.r, z__1.i = y[i__4].i + z__2.i;
                y[i__3].r = z__1.r, y[i__3].i = z__1.i;
                # 计算更新 temp2
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                i__3 = i__;
                z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, z__2.i =
                     z__3.r * x[i__3].i + z__3.i * x[i__3].r;
                z__1.r = temp2.r + z__2.r, z__1.i = temp2.i + z__2.i;
                temp2.r = z__1.r, temp2.i = z__1.i;
/* L90: */
        }
        i__2 = j;
        i__3 = j;
        // 计算 alpha * conjg(y[j]) 并加到 y[i__3] 上
        z__2.r = alpha->r * temp2.r - alpha->i * temp2.i, z__2.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
/* L100: */
        }
    } else {
        jx = kx;
        jy = ky;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        i__2 = jx;
        // 计算 alpha * x[j] 并存储到 temp1 中
        z__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, z__1.i =
             alpha->r * x[i__2].i + alpha->i * x[i__2].r;
        temp1.r = z__1.r, temp1.i = z__1.i;
        temp2.r = 0., temp2.i = 0.;
        i__2 = jy;
        i__3 = jy;
        // 计算 A[j,j] * temp1 并加到 y[jy] 上
        i__4 = j + j * a_dim1;
        d__1 = a[i__4].r;
        z__2.r = d__1 * temp1.r, z__2.i = d__1 * temp1.i;
        z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
        ix = jx;
        iy = jy;
        i__2 = *n;
        for (i__ = j + 1; i__ <= i__2; ++i__) {
            ix += *incx;
            iy += *incy;
            i__3 = iy;
            i__4 = iy;
            // 计算 A[i,j] * temp1 并加到 y[iy] 上
            i__5 = i__ + j * a_dim1;
            z__2.r = temp1.r * a[i__5].r - temp1.i * a[i__5].i,
                z__2.i = temp1.r * a[i__5].i + temp1.i * a[i__5]
                .r;
            z__1.r = y[i__4].r + z__2.r, z__1.i = y[i__4].i + z__2.i;
            y[i__3].r = z__1.r, y[i__3].i = z__1.i;
            d_cnjg(&z__3, &a[i__ + j * a_dim1]);
            i__3 = ix;
            // 计算 conjg(A[i,j]) * x[ix] 并加到 temp2 上
            z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, z__2.i =
                 z__3.r * x[i__3].i + z__3.i * x[i__3].r;
            z__1.r = temp2.r + z__2.r, z__1.i = temp2.i + z__2.i;
            temp2.r = z__1.r, temp2.i = z__1.i;
/* L110: */
        }
        i__2 = jy;
        i__3 = jy;
        // 计算 alpha * conjg(temp2) 并加到 y[jy] 上
        z__2.r = alpha->r * temp2.r - alpha->i * temp2.i, z__2.i =
            alpha->r * temp2.i + alpha->i * temp2.r;
        z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
        y[i__2].r = z__1.r, y[i__2].i = z__1.i;
        jx += *incx;
        jy += *incy;
/* L120: */
        }
    }
    }

    return 0;

/*     End of ZHEMV . */

} /* zhemv_ */

/* Subroutine */ int zher2_(char *uplo, integer *n, doublecomplex *alpha,
    doublecomplex *x, integer *incx, doublecomplex *y, integer *incy,
    doublecomplex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    static integer i__, j, ix, iy, jx, jy, kx, ky, info;
    static doublecomplex temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    # 制定 UPLO 参数，指定矩阵 A 的上三角部分或下三角部分的访问方式
    UPLO   - CHARACTER*1.
             在输入时，UPLO 指定数组 A 的上三角部分或下三角部分的访问方式，具体如下：

                UPLO = 'U' or 'u'   只访问矩阵 A 的上三角部分。
                
                UPLO = 'L' or 'l'   只访问矩阵 A 的下三角部分。
             
             函数返回时，UPLO 参数保持不变。

    # 输入参数 N，指定矩阵 A 的阶数
    N      - INTEGER.
             在输入时，N 指定矩阵 A 的阶数。N 必须至少为零。
             函数返回时，N 参数保持不变。

    # 输入参数 ALPHA，指定标量 alpha
    ALPHA  - COMPLEX*16      .
             在输入时，ALPHA 指定标量 alpha。
             函数返回时，ALPHA 参数保持不变。

    # 输入参数 X，维度至少为 ( 1 + ( n - 1 )*abs( INCX ) ) 的复数数组
    X      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             在输入时，增量数组 X 必须包含 n 元素向量 x。
             函数返回时，X 参数保持不变。

    # 输入参数 INCX，指定 X 数组中元素的增量
    INCX   - INTEGER.
             在输入时，INCX 指定 X 数组中元素的增量。INCX 不能为零。
             函数返回时，INCX 参数保持不变。

    # 输入参数 Y，维度至少为 ( 1 + ( n - 1 )*abs( INCY ) ) 的复数数组
    Y      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             在输入时，增量数组 Y 必须包含 n 元素向量 y。
             函数返回时，Y 参数保持不变。

    # 输入参数 INCY，指定 Y 数组中元素的增量
    INCY   - INTEGER.
             在输入时，INCY 指定 Y 数组中元素的增量。INCY 不能为零。
             函数返回时，INCY 参数保持不变。

    # 输入参数 A，维度为 ( LDA, n ) 的复数数组
    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
             在输入时，如果 UPLO = 'U' 或 'u'，数组 A 的前 n 行 n 列应包含埃尔米特矩阵的上三角部分，
             并且不引用 A 的严格下三角部分。返回时，数组 A 的上三角部分被更新后的矩阵的上三角部分覆盖。

             在输入时，如果 UPLO = 'L' 或 'l'，数组 A 的前 n 行 n 列应包含埃尔米特矩阵的下三角部分，
             并且不引用 A 的严格上三角部分。返回时，数组 A 的下三角部分被更新后的矩阵的下三角部分覆盖。

             注意，对角元素的虚部可以不设置，默认为零；返回时会被设置为零。

    # 输入参数 LDA，指定数组 A 的第一个维度
    LDA    - INTEGER.
             在输入时，LDA 指定数组 A 在调用程序中声明的第一个维度。LDA 必须至少为 max( 1, n )。
             函数返回时，LDA 参数保持不变。

    Further Details
    ===============

    # 级别 2 Blas 算法例程。
    Level 2 Blas routine.
    # 以下是注释的代码块，这段代码的作用是简单的注释和说明，不需要额外的解释。
    
        -- Written on 22-October-1986.
           Jack Dongarra, Argonne National Lab.
           Jeremy Du Croz, Nag Central Office.
           Sven Hammarling, Nag Central Office.
           Richard Hanson, Sandia National Labs.
    
        =====================================================================
    
    
           Test the input parameters.
    /* Parameter adjustments */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    // 检查上三角或下三角矩阵的参数是否正确
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (*n < 0) {
        // 检查矩阵维度是否为负数
        info = 2;
    } else if (*incx == 0) {
        // 检查 x 的增量是否为零
        info = 5;
    } else if (*incy == 0) {
        // 检查 y 的增量是否为零
        info = 7;
    } else if (*lda < max(1,*n)) {
        // 检查矩阵的第一个维度是否小于等于 n
        info = 9;
    }
    // 如果存在错误信息，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("ZHER2 ", &info);
        return 0;
    }

    /* Quick return if possible. */
    // 如果 n 为零或 alpha 的实部和虚部都为零，直接返回
    if (*n == 0 || alpha->r == 0. && alpha->i == 0.) {
        return 0;
    }

    /*
       Set up the start points in X and Y if the increments are not both
       unity.
    */
    // 如果 x 或 y 的增量不为 1，设置起始点 kx 和 ky
    if (*incx != 1 || *incy != 1) {
        if (*incx > 0) {
            kx = 1;
        } else {
            kx = 1 - (*n - 1) * *incx;
        }
        if (*incy > 0) {
            ky = 1;
        } else {
            ky = 1 - (*n - 1) * *incy;
        }
        jx = kx;
        jy = ky;
    }

    /*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.
    */

    if (lsame_(uplo, "U")) {
        /* Form  A  when A is stored in the upper triangle. */

        // 当 A 存储在上三角时，进行计算
        if (*incx == 1 && *incy == 1) {
            // 当 x 和 y 的增量为 1 时的计算方式
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                i__2 = j;
                i__3 = j;
                if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. ||
                    y[i__3].i != 0.)) {
                    d_cnjg(&z__2, &y[j]);
                    z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i =
                        alpha->r * z__2.i + alpha->i * z__2.r;
                    temp1.r = z__1.r, temp1.i = z__1.i;
                    i__2 = j;
                    z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                        z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                        .r;
                    d_cnjg(&z__1, &z__2);
                    temp2.r = z__1.r, temp2.i = z__1.i;
                    i__2 = j - 1;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * a_dim1;
                        i__4 = i__ + j * a_dim1;
                        i__5 = i__;
                        z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
                            z__3.i = x[i__5].r * temp1.i + x[i__5].i *
                            temp1.r;
                        z__2.r = a[i__4].r + z__3.r, z__2.i = a[i__4].i +
                            z__3.i;
                        i__6 = i__;
                        z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
                            z__4.i = y[i__6].r * temp2.i + y[i__6].i *
                            temp2.r;
                        z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
                        a[i__3].r = z__1.r, a[i__3].i = z__1.i;
                    }
                }
            }
        }
    }


以上是对给定代码的详细注释，按照要求将每行代码的作用解释清楚。
/* L10: 结束了一个if-else语句块的起始大括号 */

i__2 = j + j * a_dim1;
i__3 = j + j * a_dim1;
i__4 = j;
z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
    z__2.i = x[i__4].r * temp1.i + x[i__4].i *
    temp1.r;
i__5 = j;
z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
    z__3.i = y[i__5].r * temp2.i + y[i__5].i *
    temp2.r;
z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
d__1 = a[i__3].r + z__1.r;
a[i__2].r = d__1, a[i__2].i = 0.;
/* 对称阵的处理：更新对称阵A的第j列及其对角元素 */

} else {
/* L20: */
/* 处理对称阵的下三角部分 */
i__2 = j + j * a_dim1;
i__3 = j + j * a_dim1;
d__1 = a[i__3].r;
a[i__2].r = d__1, a[i__2].i = 0.;
}

} /* 结束了第一个for循环的大括号 */

} else {
/* 处理一般情况下的矩阵乘法 */
i__1 = *n;
for (j = 1; j <= i__1; ++j) {
i__2 = jx;
i__3 = jy;
if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. ||
    y[i__3].i != 0.)) {
/* 计算temp1和temp2的值 */
d_cnjg(&z__2, &y[jy]);
z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i =
    alpha->r * z__2.i + alpha->i * z__2.r;
temp1.r = z__1.r, temp1.i = z__1.i;
i__2 = jx;
z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
    z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
    .r;
d_cnjg(&z__1, &z__2);
temp2.r = z__1.r, temp2.i = z__1.i;
ix = kx;
iy = ky;
i__2 = j - 1;
for (i__ = 1; i__ <= i__2; ++i__) {
i__3 = i__ + j * a_dim1;
i__4 = i__ + j * a_dim1;
i__5 = ix;
z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
    z__3.i = x[i__5].r * temp1.i + x[i__5].i *
    temp1.r;
z__2.r = a[i__4].r + z__3.r, z__2.i = a[i__4].i +
    z__3.i;
i__6 = iy;
z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
    z__4.i = y[i__6].r * temp2.i + y[i__6].i *
    temp2.r;
z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
a[i__3].r = z__1.r, a[i__3].i = z__1.i;
ix += *incx;
iy += *incy;


注释结束。
/* L30: */
            }
            // 计算 a[j][j] 的更新值
            i__2 = j + j * a_dim1;
            // 计算临时变量 z__2 和 z__3
            i__3 = j + j * a_dim1;
            i__4 = jx;
            z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
                z__2.i = x[i__4].r * temp1.i + x[i__4].i *
                temp1.r;
            i__5 = jy;
            z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
                z__3.i = y[i__5].r * temp2.i + y[i__5].i *
                temp2.r;
            // 计算 a[j][j] 的新值
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            d__1 = a[i__3].r + z__1.r;
            a[i__2].r = d__1, a[i__2].i = 0.;
        } else {
            // 如果 alpha 或 x、y 的某个部分为零，则将 a[j][j] 的虚部设为零
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            d__1 = a[i__3].r;
            a[i__2].r = d__1, a[i__2].i = 0.;
        }
        // 更新 jx 和 jy
        jx += *incx;
        jy += *incy;
/* L40: */
        }
    }
    } else {

/*        Form  A  when A is stored in the lower triangle. */

    // 当 A 存储在下三角形时，形成 A
    if (*incx == 1 && *incy == 1) {
        // 对于每个 j，执行以下操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        // 计算 alpha 和 x[j]、y[j] 的乘积
        i__2 = j;
        i__3 = j;
        if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. ||
            y[i__3].i != 0.)) {
            // 计算 temp1 和 temp2
            d_cnjg(&z__2, &y[j]);
            z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i =
                alpha->r * z__2.i + alpha->i * z__2.r;
            temp1.r = z__1.r, temp1.i = z__1.i;
            i__2 = j;
            z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                .r;
            d_cnjg(&z__1, &z__2);
            temp2.r = z__1.r, temp2.i = z__1.i;
            // 计算 a[j][j] 的更新值
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            i__4 = j;
            z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
                z__2.i = x[i__4].r * temp1.i + x[i__4].i *
                temp1.r;
            i__5 = j;
            z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
                z__3.i = y[i__5].r * temp2.i + y[i__5].i *
                temp2.r;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            d__1 = a[i__3].r + z__1.r;
            a[i__2].r = d__1, a[i__2].i = 0.;
            // 对于每个 i > j，执行以下操作
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * a_dim1;
            i__4 = i__ + j * a_dim1;
            i__5 = i__;
            z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
                z__3.i = x[i__5].r * temp1.i + x[i__5].i *
                temp1.r;
            z__2.r = a[i__4].r + z__3.r, z__2.i = a[i__4].i +
                z__3.i;
            i__6 = i__;
            z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
                z__4.i = y[i__6].r * temp2.i + y[i__6].i *
                temp2.r;
            z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
            // 更新 a[i][j]
            a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L50: */
            }
        } else {
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            d__1 = a[i__3].r;
            a[i__2].r = d__1, a[i__2].i = 0.;
        }
/* L60: */
        }
    } else {
        i__1 = *n;
        // 对于每一列 j，执行以下操作
        for (j = 1; j <= i__1; ++j) {
        // 计算 x 和 y 的起始索引
        i__2 = jx;
        i__3 = jy;
        // 如果 x[jx] 或者 y[jy] 非零，则执行以下操作
        if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. ||
            y[i__3].i != 0.)) {
            // 计算 temp1 = alpha * conj(y[jy])
            d_cnjg(&z__2, &y[jy]);
            z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i =
                alpha->r * z__2.i + alpha->i * z__2.r;
            temp1.r = z__1.r, temp1.i = z__1.i;
            // 计算 temp2 = alpha * x[jx]
            i__2 = jx;
            z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i,
                z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
                .r;
            d_cnjg(&z__1, &z__2);
            temp2.r = z__1.r, temp2.i = z__1.i;
            // 计算 a[j][j] = a[j][j] + x[jx] * temp1 + y[jy] * temp2
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            i__4 = jx;
            z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i,
                z__2.i = x[i__4].r * temp1.i + x[i__4].i *
                temp1.r;
            i__5 = jy;
            z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i,
                z__3.i = y[i__5].r * temp2.i + y[i__5].i *
                temp2.r;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            d__1 = a[i__3].r + z__1.r;
            a[i__2].r = d__1, a[i__2].i = 0.;
            // 更新 ix 和 iy 为下一轮循环准备
            ix = jx;
            iy = jy;
            // 对于每一行 i > j，执行以下操作
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            ix += *incx;
            iy += *incy;
            // 计算 a[i][j] = a[i][j] + x[ix] * temp1 + y[iy] * temp2
            i__3 = i__ + j * a_dim1;
            i__4 = i__ + j * a_dim1;
            i__5 = ix;
            z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i,
                z__3.i = x[i__5].r * temp1.i + x[i__5].i *
                temp1.r;
            z__2.r = a[i__4].r + z__3.r, z__2.i = a[i__4].i +
                z__3.i;
            i__6 = iy;
            z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i,
                z__4.i = y[i__6].r * temp2.i + y[i__6].i *
                temp2.r;
            z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
            a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L70: */
            }
        } else {
            // 如果 x[jx] 和 y[jy] 都为零，则 a[j][j] = a[j][j]
            i__2 = j + j * a_dim1;
            i__3 = j + j * a_dim1;
            d__1 = a[i__3].r;
            a[i__2].r = d__1, a[i__2].i = 0.;
        }
        // 更新 jx 和 jy 为下一列准备
        jx += *incx;
        jy += *incy;
/* L80: */
        }
    }
    }

    return 0;

/*     End of ZHER2 . */

} /* zher2_ */

/* Subroutine */ int zher2k_(char *uplo, char *trans, integer *n, integer *k,
    doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
    b, integer *ldb, doublereal *beta, doublecomplex *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3, i__4, i__5, i__6, i__7;
    doublereal d__1;


注释：
    声明复数变量 z__1, z__2, z__3, z__4, z__5, z__6，用于存储复数值

    /* Local variables */
    static integer i__, j, l, info;
    声明静态整型变量 i__, j, l, info，用于存储整数值
    static doublecomplex temp1, temp2;
    声明静态复数变量 temp1, temp2，用于存储临时复数值
    extern logical lsame_(char *, char *);
    声明外部函数 lsame_，用于比较两个字符数组是否相同
    static integer nrowa;
    声明静态整型变量 nrowa，用于存储整数值
    static logical upper;
    声明静态逻辑变量 upper，用于表示是否为上三角矩阵
    extern /* Subroutine */ int xerbla_(char *, integer *);
    声明外部子程序 xerbla_，用于处理错误信息
/*
    Purpose
    =======

    ZHER2K  performs one of the hermitian rank 2k operations

       C := alpha*A*conjg( B' ) + conjg( alpha )*B*conjg( A' ) + beta*C,

    or

       C := alpha*conjg( A' )*B + conjg( alpha )*conjg( B' )*A + beta*C,

    where  alpha and beta  are scalars with  beta  real,  C is an  n by n
    hermitian matrix and  A and B  are  n by k matrices in the first case
    and  k by n  matrices in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of  the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'    C := alpha*A*conjg( B' )          +
                                           conjg( alpha )*B*conjg( A' ) +
                                           beta*C.

                TRANS = 'C' or 'c'    C := alpha*conjg( A' )*B          +
                                           conjg( alpha )*conjg( B' )*A +
                                           beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns  of  the  matrices  A and B,  and on  entry  with
             TRANS = 'C' or 'c',  K  specifies  the number of rows of the
             matrices  A and B.  K must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16         .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX*16       array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.
*/
    ! B - 复数双精度数组，维度为 (LDB, kb)，其中 kb 是 k 当 TRANS = 'N' 或 'n' 时，否则是 n。
    !     在进入时，如果 TRANS = 'N' 或 'n'，则数组 B 的前 n 行 k 列必须包含矩阵 B；
    !     否则，数组 B 的前 k 行 n 列必须包含矩阵 B。
    !     函数返回后，数组 B 保持不变。

    ! LDB - 整数。
    !     在进入时，LDB 指定了 B 的第一个维度，如在调用程序中声明的。
    !     当 TRANS = 'N' 或 'n' 时，LDB 必须至少为 max(1, n)；
    !     否则，LDB 必须至少为 max(1, k)。
    !     函数返回后，LDB 保持不变。

    ! BETA - 双精度浮点数。
    !     在进入时，BETA 指定了标量 beta。
    !     函数返回后，BETA 保持不变。

    ! C - 复数双精度数组，维度为 (LDC, n)。
    !     在进入时，如果 UPLO = 'U' 或 'u'，数组 C 的前 n 行 n 列必须包含上三角部分的 Hermitian 矩阵；
    !     函数返回后，数组 C 的上三角部分被更新后的矩阵上三角部分覆盖。
    !     在进入时，如果 UPLO = 'L' 或 'l'，数组 C 的前 n 行 n 列必须包含下三角部分的 Hermitian 矩阵；
    !     函数返回后，数组 C 的下三角部分被更新后的矩阵下三角部分覆盖。
    !     注意：对角元素的虚部不需要设置，假定为零，在函数返回后也被设为零。

    ! LDC - 整数。
    !     在进入时，LDC 指定了 C 的第一个维度，如在调用程序中声明的。
    !     必须至少为 max(1, n)。
    !     函数返回后，LDC 保持不变。

    ! Further Details
    ! ===============
    !
    ! Level 3 Blas routine.
    !
    ! -- Written on 8-February-1989.
    !    Jack Dongarra, Argonne National Laboratory.
    !    Iain Duff, AERE Harwell.
    !    Jeremy Du Croz, Numerical Algorithms Group Ltd.
    !    Sven Hammarling, Numerical Algorithms Group Ltd.
    !
    ! -- Modified 8-Nov-93 to set C(J,J) to DBLE( C(J,J) ) when BETA = 1.
    !    Ed Anderson, Cray Research Inc.
    !
    ! =====================================================================
    !
    !
    !    Test the input parameters.
    /* Parameter adjustments */
    // 对参数进行调整，a、b、c 的维度和偏移
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    // 函数体开始

    // 根据转置参数确定 nrowa 的值
    if (lsame_(trans, "N")) {
        nrowa = *n;
    } else {
        nrowa = *k;
    }

    // 判断是否为上三角阵
    upper = lsame_(uplo, "U");

    // 初始化 info 为 0
    info = 0;
    // 检查参数有效性
    if (! upper && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "C")) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*k < 0) {
        info = 4;
    } else if (*lda < max(1,nrowa)) {
        info = 7;
    } else if (*ldb < max(1,nrowa)) {
        info = 9;
    } else if (*ldc < max(1,*n)) {
        info = 12;
    }

    // 如果存在错误信息，调用错误处理函数 xerbla_，并返回
    if (info != 0) {
        xerbla_("ZHER2K", &info);
        return 0;
    }

    /* Quick return if possible. */
    // 如果 n 为 0 或者 alpha 为 0，beta 为 1，则直接返回
    if (*n == 0 || (alpha->r == 0. && alpha->i == 0. || *k == 0) && *beta == 1.) {
        return 0;
    }

    /* And when alpha.eq.zero. */
    // 当 alpha 为 0 时的处理

    if (alpha->r == 0. && alpha->i == 0.) {
        // 当上三角阵时
        if (upper) {
            // 如果 beta 为 0
            if (*beta == 0.) {
                // 将 C 的上三角部分置为 0
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0., c__[i__3].i = 0.;
                    }
                }
            } else {
                // 对 C 的上三角部分进行 beta 倍的缩放
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j - 1;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[i__4].i;
                        c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                    }
                    i__2 = j + j * c_dim1;
                    i__3 = j + j * c_dim1;
                    d__1 = *beta * c__[i__3].r;
                    c__[i__2].r = d__1, c__[i__2].i = 0.;
                }
            }
        } else {
            // 当下三角阵时
            if (*beta == 0.) {
                // 将 C 的下三角部分置为 0
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0., c__[i__3].i = 0.;
                    }
                }
            } else {
                // 对 C 的下三角部分进行 beta 倍的缩放
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j + j * c_dim1;
                    i__3 = j + j * c_dim1;
                    d__1 = *beta * c__[i__3].r;
                    c__[i__2].r = d__1, c__[i__2].i = 0.;
                    i__2 = *n;
                    for (i__ = j + 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[i__4].i;
                        c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                    }
                }
            }
        }
        return 0;
    }

    /* Start the operations. */
    // 开始执行矩阵运算

    // 当 trans = "N" 时
    if (lsame_(trans, "N")) {
/*
      Form  C := alpha*A*conjg( B' ) + conjg( alpha )*B*conjg( A' ) +
                 C.
*/

if (upper) {
    // 遍历矩阵的每一列
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        if (*beta == 0.) {
            // 如果 beta 等于 0，将 C 的上三角部分设为零
            i__2 = j;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * c_dim1;
                c__[i__3].r = 0., c__[i__3].i = 0.;
                // L90 标签，用于跳出内层循环
/* L90: */       }
        } else if (*beta != 1.) {
            // 如果 beta 不等于 1，对 C 的上三角部分进行倍数缩放
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * c_dim1;
                i__4 = i__ + j * c_dim1;
                z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[i__4].i;
                c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                // L100 标签，用于跳出内层循环
/* L100: */      }
            // 对角线上的元素进行缩放
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            d__1 = *beta * c__[i__3].r;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        } else {
            // 如果 beta 等于 1，对角线上的元素保持不变
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            d__1 = c__[i__3].r;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        }
        // 遍历 A 和 B 的第 j 列
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            i__3 = j + l * a_dim1;
            i__4 = j + l * b_dim1;
            // 如果 A 或者 B 的第 j 列不全为零，则进行下面的操作
            if (a[i__3].r != 0. || a[i__3].i != 0. || (b[i__4].r != 0. || b[i__4].i != 0.)) {
                // 计算临时变量 temp1 和 temp2
                d_cnjg(&z__2, &b[j + l * b_dim1]);
                z__1.r = alpha->r * z__2.r - alpha->i * z__2.i,
                         z__1.i = alpha->r * z__2.i + alpha->i * z__2.r;
                temp1.r = z__1.r, temp1.i = z__1.i;
                i__3 = j + l * a_dim1;
                z__2.r = alpha->r * a[i__3].r - alpha->i * a[i__3].i,
                         z__2.i = alpha->r * a[i__3].i + alpha->i * a[i__3].r;
                d_cnjg(&z__1, &z__2);
                temp2.r = z__1.r, temp2.i = z__1.i;
                // 对 C 的上三角部分进行更新
                i__3 = j - 1;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    i__4 = i__ + j * c_dim1;
                    i__5 = i__ + j * c_dim1;
                    i__6 = i__ + l * a_dim1;
                    z__3.r = a[i__6].r * temp1.r - a[i__6].i * temp1.i,
                             z__3.i = a[i__6].r * temp1.i + a[i__6].i * temp1.r;
                    z__2.r = c__[i__5].r + z__3.r, z__2.i = c__[i__5].i + z__3.i;
                    i__7 = i__ + l * b_dim1;
                    z__4.r = b[i__7].r * temp2.r - b[i__7].i * temp2.i,
                             z__4.i = b[i__7].r * temp2.i + b[i__7].i * temp2.r;
                    z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
                    c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
                }
            }
        }
    }
}
/* L110: */
            }
            // 计算 c[j][j] 的更新值
            i__3 = j + j * c_dim1;
            i__4 = j + j * c_dim1;
            i__5 = j + l * a_dim1;
            // 计算临时变量 temp1 的乘积
            z__2.r = a[i__5].r * temp1.r - a[i__5].i * temp1.i,
                z__2.i = a[i__5].r * temp1.i + a[i__5].i *
                temp1.r;
            i__6 = j + l * b_dim1;
            // 计算临时变量 temp2 的乘积
            z__3.r = b[i__6].r * temp2.r - b[i__6].i * temp2.i,
                z__3.i = b[i__6].r * temp2.i + b[i__6].i *
                temp2.r;
            // 计算 c[j][j] 的新值
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            d__1 = c__[i__4].r + z__1.r;
            c__[i__3].r = d__1, c__[i__3].i = 0.;
            }
/* L120: */
        }
/* L130: */
        }
    } else {
        // beta 不等于 0 的情况下
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        if (*beta == 0.) {
            // beta 等于 0 的情况下
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            // 将 c[i][j] 设置为 0
            c__[i__3].r = 0., c__[i__3].i = 0.;
/* L140: */
            }
        } else if (*beta != 1.) {
            // beta 不等于 1 的情况下
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            i__4 = i__ + j * c_dim1;
            // 更新 c[i][j] 为 beta * c[i][j]
            z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[
                i__4].i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L150: */
            }
            // 如果 beta 不为零，则更新 C 矩阵的对角元素
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            d__1 = *beta * c__[i__3].r;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        } else {
            // 否则，直接将 C 矩阵的对角元素设为 A 矩阵的对角元素
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            d__1 = c__[i__3].r;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        }
        // 对于每个 l，执行以下操作
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            // 如果 A 或 B 的第 (j, l) 元素不为零
            i__3 = j + l * a_dim1;
            i__4 = j + l * b_dim1;
            if (a[i__3].r != 0. || a[i__3].i != 0. || (b[i__4].r !=
                0. || b[i__4].i != 0.)) {
            // 计算临时变量 temp1 和 temp2
            d_cnjg(&z__2, &b[j + l * b_dim1]);
            z__1.r = alpha->r * z__2.r - alpha->i * z__2.i,
                z__1.i = alpha->r * z__2.i + alpha->i *
                z__2.r;
            temp1.r = z__1.r, temp1.i = z__1.i;
            i__3 = j + l * a_dim1;
            z__2.r = alpha->r * a[i__3].r - alpha->i * a[i__3].i,
                z__2.i = alpha->r * a[i__3].i + alpha->i * a[
                i__3].r;
            d_cnjg(&z__1, &z__2);
            temp2.r = z__1.r, temp2.i = z__1.i;
            // 对于每个 i，执行以下操作
            i__3 = *n;
            for (i__ = j + 1; i__ <= i__3; ++i__) {
                // 更新 C 矩阵的 (i, j) 元素
                i__4 = i__ + j * c_dim1;
                i__5 = i__ + j * c_dim1;
                i__6 = i__ + l * a_dim1;
                z__3.r = a[i__6].r * temp1.r - a[i__6].i *
                    temp1.i, z__3.i = a[i__6].r * temp1.i + a[
                    i__6].i * temp1.r;
                z__2.r = c__[i__5].r + z__3.r, z__2.i = c__[i__5]
                    .i + z__3.i;
                i__7 = i__ + l * b_dim1;
                z__4.r = b[i__7].r * temp2.r - b[i__7].i *
                    temp2.i, z__4.i = b[i__7].r * temp2.i + b[
                    i__7].i * temp2.r;
                z__1.r = z__2.r + z__4.r, z__1.i = z__2.i +
                    z__4.i;
                c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
/* L160: */
            }
            // 更新 C 矩阵的对角元素
            i__3 = j + j * c_dim1;
            i__4 = j + j * c_dim1;
            i__5 = j + l * a_dim1;
            z__2.r = a[i__5].r * temp1.r - a[i__5].i * temp1.i,
                z__2.i = a[i__5].r * temp1.i + a[i__5].i *
                temp1.r;
            i__6 = j + l * b_dim1;
            z__3.r = b[i__6].r * temp2.r - b[i__6].i * temp2.i,
                z__3.i = b[i__6].r * temp2.i + b[i__6].i *
                temp2.r;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            d__1 = c__[i__4].r + z__1.r;
            c__[i__3].r = d__1, c__[i__3].i = 0.;
            }
/* L170: */
        }
/* L180: */
        }
    }
    } else {

/*
          Form  C := alpha*conjg( A' )*B + conjg( alpha )*conjg( B' )*A +
                     C.
*/
    // 如果 upper 为真，则执行以下循环
    if (upper) {
        // 循环遍历 j 从 1 到 *n
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 设置 i__2 为 j，然后循环遍历 i__ 从 1 到 i__2
            i__2 = j;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 初始化 temp1 和 temp2 为复数 0
                temp1.r = 0., temp1.i = 0.;
                temp2.r = 0., temp2.i = 0.;
                // 循环遍历 l 从 1 到 *k
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    // 计算 a[l + i__ * a_dim1] 的共轭复数，并与 b[l + j * b_dim1] 的乘积相加到 temp1
                    d_cnjg(&z__3, &a[l + i__ * a_dim1]);
                    i__4 = l + j * b_dim1;
                    z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4].i,
                        z__2.i = z__3.r * b[i__4].i + z__3.i * b[i__4].r;
                    z__1.r = temp1.r + z__2.r, z__1.i = temp1.i + z__2.i;
                    temp1.r = z__1.r, temp1.i = z__1.i;
                    // 计算 b[l + i__ * b_dim1] 的共轭复数，并与 a[l + j * a_dim1] 的乘积相加到 temp2
                    d_cnjg(&z__3, &b[l + i__ * b_dim1]);
                    i__4 = l + j * a_dim1;
                    z__2.r = z__3.r * a[i__4].r - z__3.i * a[i__4].i,
                        z__2.i = z__3.r * a[i__4].i + z__3.i * a[i__4].r;
                    z__1.r = temp2.r + z__2.r, z__1.i = temp2.i + z__2.i;
                    temp2.r = z__1.r, temp2.i = z__1.i;
/* L190: 结束内层循环 */
            }
            // 检查是否 i__ 等于 j
            if (i__ == j) {
                // 检查 beta 是否为 0
                if (*beta == 0.) {
                    // 计算 c__ 中的元素
                    i__3 = j + j * c_dim1;
                    z__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    d_cnjg(&z__4, alpha);
                    z__3.r = z__4.r * temp2.r - z__4.i * temp2.i,
                        z__3.i = z__4.r * temp2.i + z__4.i *
                        temp2.r;
                    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
                        z__3.i;
                    d__1 = z__1.r;
                    c__[i__3].r = d__1, c__[i__3].i = 0.;
                } else {
                    // 计算 c__ 中的元素，考虑 beta 值不为 0 的情况
                    i__3 = j + j * c_dim1;
                    i__4 = j + j * c_dim1;
                    z__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    d_cnjg(&z__4, alpha);
                    z__3.r = z__4.r * temp2.r - z__4.i * temp2.i,
                        z__3.i = z__4.r * temp2.i + z__4.i *
                        temp2.r;
                    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
                        z__3.i;
                    d__1 = *beta * c__[i__4].r + z__1.r;
                    c__[i__3].r = d__1, c__[i__3].i = 0.;
                }
            } else {
                // 处理 i__ 不等于 j 的情况
                if (*beta == 0.) {
                    // 计算 c__ 中的元素
                    i__3 = i__ + j * c_dim1;
                    z__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    d_cnjg(&z__4, alpha);
                    z__3.r = z__4.r * temp2.r - z__4.i * temp2.i,
                        z__3.i = z__4.r * temp2.i + z__4.i *
                        temp2.r;
                    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
                        z__3.i;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                } else {
                    // 计算 c__ 中的元素，考虑 beta 值不为 0 的情况
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    z__3.r = *beta * c__[i__4].r, z__3.i = *beta *
                        c__[i__4].i;
                    z__4.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__4.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    z__2.r = z__3.r + z__4.r, z__2.i = z__3.i +
                        z__4.i;
                    d_cnjg(&z__6, alpha);
                    z__5.r = z__6.r * temp2.r - z__6.i * temp2.i,
                        z__5.i = z__6.r * temp2.i + z__6.i *
                        temp2.r;
                    z__1.r = z__2.r + z__5.r, z__1.i = z__2.i +
                        z__5.i;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                }
            }
/* L200: 结束外层循环 */
        }
/* L210: 结束另一个外层循环 */
        }
    } else {
        // 循环：对每个 j，从 1 到 *n
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 循环：对每个 i，从 j 到 *n
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
                // 初始化临时复数变量 temp1 和 temp2
                temp1.r = 0., temp1.i = 0.;
                temp2.r = 0., temp2.i = 0.;
                // 循环：对每个 l，从 1 到 *k
                i__3 = *k;
                for (l = 1; l <= i__3; ++l) {
                    // 复数取共轭：d_cnjg(&z__3, &a[l + i__ * a_dim1])
                    d_cnjg(&z__3, &a[l + i__ * a_dim1]);
                    // 计算矩阵乘法中的一部分，结果加入到 temp1 中
                    i__4 = l + j * b_dim1;
                    z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4].i,
                        z__2.i = z__3.r * b[i__4].i + z__3.i * b[i__4].r;
                    z__1.r = temp1.r + z__2.r, z__1.i = temp1.i + z__2.i;
                    temp1.r = z__1.r, temp1.i = z__1.i;
                    // 复数取共轭：d_cnjg(&z__3, &b[l + i__ * b_dim1])
                    d_cnjg(&z__3, &b[l + i__ * b_dim1]);
                    // 计算矩阵乘法中的一部分，结果加入到 temp2 中
                    i__4 = l + j * a_dim1;
                    z__2.r = z__3.r * a[i__4].r - z__3.i * a[i__4].i,
                        z__2.i = z__3.r * a[i__4].i + z__3.i * a[i__4].r;
                    z__1.r = temp2.r + z__2.r, z__1.i = temp2.i + z__2.i;
                    temp2.r = z__1.r, temp2.i = z__1.i;
/* L220: */
/* 循环结束的右花括号，结束上一个循环的代码块 */

            }
            /* 如果 i__ 等于 j */
            if (i__ == j) {
                /* 如果 beta 等于 0 */
                if (*beta == 0.) {
                    /* 计算 alpha 和 temp1 的乘积，并赋值给 c__[j + j * c_dim1] */
                    i__3 = j + j * c_dim1;
                    z__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    d_cnjg(&z__4, alpha);
                    z__3.r = z__4.r * temp2.r - z__4.i * temp2.i,
                        z__3.i = z__4.r * temp2.i + z__4.i *
                        temp2.r;
                    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
                        z__3.i;
                    d__1 = z__1.r;
                    c__[i__3].r = d__1, c__[i__3].i = 0.;
                } else {
                    /* 计算 alpha 和 temp1 的乘积加上 beta 乘以 c__[j + j * c_dim1]，并赋值给 c__[j + j * c_dim1] */
                    i__3 = j + j * c_dim1;
                    i__4 = j + j * c_dim1;
                    z__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    d_cnjg(&z__4, alpha);
                    z__3.r = z__4.r * temp2.r - z__4.i * temp2.i,
                        z__3.i = z__4.r * temp2.i + z__4.i *
                        temp2.r;
                    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
                        z__3.i;
                    d__1 = *beta * c__[i__4].r + z__1.r;
                    c__[i__3].r = d__1, c__[i__3].i = 0.;
                }
            } else {
                /* 如果 i__ 不等于 j */
                if (*beta == 0.) {
                    /* 计算 alpha 和 temp1 的乘积加上 temp2 的乘积，并赋值给 c__[i__ + j * c_dim1] */
                    i__3 = i__ + j * c_dim1;
                    z__2.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__2.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    d_cnjg(&z__4, alpha);
                    z__3.r = z__4.r * temp2.r - z__4.i * temp2.i,
                        z__3.i = z__4.r * temp2.i + z__4.i *
                        temp2.r;
                    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
                        z__3.i;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                } else {
                    /* 计算 beta 乘以 c__[i__ + j * c_dim1] 加上 alpha 和 temp1 的乘积加上 temp2 的乘积，并赋值给 c__[i__ + j * c_dim1] */
                    i__3 = i__ + j * c_dim1;
                    i__4 = i__ + j * c_dim1;
                    z__3.r = *beta * c__[i__4].r, z__3.i = *beta *
                        c__[i__4].i;
                    z__4.r = alpha->r * temp1.r - alpha->i * temp1.i,
                        z__4.i = alpha->r * temp1.i + alpha->i *
                        temp1.r;
                    z__2.r = z__3.r + z__4.r, z__2.i = z__3.i +
                        z__4.i;
                    d_cnjg(&z__6, alpha);
                    z__5.r = z__6.r * temp2.r - z__6.i * temp2.i,
                        z__5.i = z__6.r * temp2.i + z__6.i *
                        temp2.r;
                    z__1.r = z__2.r + z__5.r, z__1.i = z__2.i +
                        z__5.i;
                    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                }
            }
/* L230: */
/* 用于标识循环体结束的注释，循环体的下一个元素将从这里开始 */

        }
/* L240: */
/* 用于标识循环体结束的注释，循环体的下一个元素将从这里开始 */
        }
    }
    }

    return 0;
/* 返回整数值 0 */

/*     End of ZHER2K. */

} /* zher2k_ */

/* Subroutine */ int zherk_(char *uplo, char *trans, integer *n, integer *k,
    doublereal *alpha, doublecomplex *a, integer *lda, doublereal *beta,
    doublecomplex *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3, i__4, i__5,
        i__6;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, j, l, info;  // 声明整型局部变量 i, j, l, info
    static doublecomplex temp;  // 声明复数类型的静态局部变量 temp
    extern logical lsame_(char *, char *);  // 外部函数 lsame_ 的声明
    static integer nrowa;  // 声明整型静态局部变量 nrowa
    static doublereal rtemp;  // 声明双精度浮点型静态局部变量 rtemp
    static logical upper;  // 声明逻辑型静态局部变量 upper
    extern /* Subroutine */ int xerbla_(char *, integer *);  // 外部子程序 xerbla_ 的声明
"""
Purpose
=======

ZHERK  performs one of the hermitian rank k operations

   C := alpha*A*conjg( A' ) + beta*C,

or

   C := alpha*conjg( A' )*A + beta*C,

where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
matrix and  A  is an  n by k  matrix in the  first case and a  k by n
matrix in the second case.

Arguments
=========

UPLO   - CHARACTER*1.
         On  entry,   UPLO  specifies  whether  the  upper  or  lower
         triangular  part  of the  array  C  is to be  referenced  as
         follows:

            UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                is to be referenced.

            UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                is to be referenced.

         Unchanged on exit.

TRANS  - CHARACTER*1.
         On entry,  TRANS  specifies the operation to be performed as
         follows:

            TRANS = 'N' or 'n'   C := alpha*A*conjg( A' ) + beta*C.

            TRANS = 'C' or 'c'   C := alpha*conjg( A' )*A + beta*C.

         Unchanged on exit.

N      - INTEGER.
         On entry,  N specifies the order of the matrix C.  N must be
         at least zero.
         Unchanged on exit.

K      - INTEGER.
         On entry with  TRANS = 'N' or 'n',  K  specifies  the number
         of  columns   of  the   matrix   A,   and  on   entry   with
         TRANS = 'C' or 'c',  K  specifies  the number of rows of the
         matrix A.  K must be at least zero.
         Unchanged on exit.

ALPHA  - DOUBLE PRECISION.
         On entry, ALPHA specifies the scalar alpha.
         Unchanged on exit.

A      - COMPLEX*16       array of DIMENSION ( LDA, ka ), where ka is
         k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
         Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
         part of the array  A  must contain the matrix  A,  otherwise
         the leading  k by n  part of the array  A  must contain  the
         matrix A.
         Unchanged on exit.

LDA    - INTEGER.
         On entry, LDA specifies the first dimension of A as declared
         in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
         then  LDA must be at least  max( 1, n ), otherwise  LDA must
         be at least  max( 1, k ).
         Unchanged on exit.

BETA   - DOUBLE PRECISION.
         On entry, BETA specifies the scalar beta.
         Unchanged on exit.
"""
    ! C      - COMPLEX*16          array of DIMENSION ( LDC, n ).
    !          在调用前，如果 UPLO = 'U' 或 'u'，则数组 C 的前 n 行 n 列必须包含厄米特矩阵的上三角部分，
    !          并且 C 的严格下三角部分不会被引用。调用结束后，数组 C 的上三角部分被更新后的矩阵上三角部分覆盖。
    !          如果 UPLO = 'L' 或 'l'，则数组 C 的前 n 行 n 列必须包含厄米特矩阵的下三角部分，
    !          并且 C 的严格上三角部分不会被引用。调用结束后，数组 C 的下三角部分被更新后的矩阵下三角部分覆盖。
    !          注意，对角元素的虚部不需要设置，假定为零，并且在调用结束时它们被设为零。

    ! LDC    - INTEGER.
    !          在调用时，LDC 指定了数组 C 在调用程序中声明的第一个维度。
    !          LDC 必须至少为 max(1, n)。
    !          调用结束时保持不变。

    ! Further Details
    ! ===============
    ! Level 3 Blas routine.
    ! Level 3 Blas 子程序。

    ! -- Written on 8-February-1989.
    !    Jack Dongarra, Argonne National Laboratory.
    !    Iain Duff, AERE Harwell.
    !    Jeremy Du Croz, Numerical Algorithms Group Ltd.
    !    Sven Hammarling, Numerical Algorithms Group Ltd.
    ! 编写日期：1989年2月8日。
    ! Jack Dongarra，阿贡国家实验室。
    ! Iain Duff，AERE Harwell。
    ! Jeremy Du Croz，Numerical Algorithms Group Ltd.
    ! Sven Hammarling，Numerical Algorithms Group Ltd。

    ! -- Modified 8-Nov-93 to set C(J,J) to DBLE( C(J,J) ) when BETA = 1.
    !    Ed Anderson, Cray Research Inc.
    ! 修改日期：1993年11月8日，当 BETA = 1 时，将 C(J,J) 设置为 C(J,J) 的双精度值。
    ! Ed Anderson，Cray Research Inc.

    ! =====================================================================

    ! Test the input parameters.
    ! 测试输入参数。
    /* 参数调整 */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* 函数体 */
    if (lsame_(trans, "N")) {
        nrowa = *n;
    } else {
        nrowa = *k;
    }
    upper = lsame_(uplo, "U");

    info = 0;
    if (! upper && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "C")) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*k < 0) {
        info = 4;
    } else if (*lda < max(1,nrowa)) {
        info = 7;
    } else if (*ldc < max(1,*n)) {
        info = 10;
    }

    /* 检查错误信息，调用错误处理函数并返回 */
    if (info != 0) {
        xerbla_("ZHERK ", &info);
        return 0;
    }

    /* 如果可能，快速返回 */
    if (*n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
        return 0;
    }

    /* 当 alpha 等于零时的情况 */
    if (*alpha == 0.) {
        if (upper) {
            if (*beta == 0.) {
                /* 设置 C 的上三角部分为零 */
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0., c__[i__3].i = 0.;
                    }
                }
            } else {
                /* 在 C 的上三角部分乘以 beta */
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j - 1;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[i__4].i;
                        c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                    }
                    i__2 = j + j * c_dim1;
                    i__3 = j + j * c_dim1;
                    d__1 = *beta * c__[i__3].r;
                    c__[i__2].r = d__1, c__[i__2].i = 0.;
                }
            }
        } else {
            if (*beta == 0.) {
                /* 设置 C 的下三角部分为零 */
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *n;
                    for (i__ = j; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        c__[i__3].r = 0., c__[i__3].i = 0.;
                    }
                }
            } else {
                /* 在 C 的下三角部分乘以 beta */
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j + j * c_dim1;
                    i__3 = j + j * c_dim1;
                    d__1 = *beta * c__[i__3].r;
                    c__[i__2].r = d__1, c__[i__2].i = 0.;
                    i__2 = *n;
                    for (i__ = j + 1; i__ <= i__2; ++i__) {
                        i__3 = i__ + j * c_dim1;
                        i__4 = i__ + j * c_dim1;
                        z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[i__4].i;
                        c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
                    }
                }
            }
        }
        return 0;
    }

    /* 开始操作 */

    if (lsame_(trans, "N")) {

        /* 计算 C := alpha*A*conjg( A' ) + beta*C 的情况 */
    # 如果参数 upper 为真值（非零），执行以下代码块
    if (upper) {
        # 循环遍历 j 从 1 到 *n
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            # 如果参数 beta 等于 0.，执行以下代码块
            if (*beta == 0.) {
                # 循环遍历 i 从 1 到 j
                i__2 = j;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    # 计算 c__ 数组中的复数元素的索引
                    i__3 = i__ + j * c_dim1;
                    # 将该复数元素的实部和虚部设为 0
                    c__[i__3].r = 0., c__[i__3].i = 0.;
/* L90: */
/* 如果 beta 不等于 1 */
        } else if (*beta != 1.) {
/* 对于 j-1 范围内的每个 i，执行以下操作 */
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
/* 计算 c[i__, j] 的索引 */
            i__3 = i__ + j * c_dim1;
/* 计算 beta * c[i__, j] 的复数乘积 */
            i__4 = i__ + j * c_dim1;
/* 将结果赋值给 c[i__, j] */
            z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[
                i__4].i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L100: */
            }
/* 计算 c[j, j] 的索引 */
            i__2 = j + j * c_dim1;
/* 计算 beta * c[j, j] 的实数乘积 */
            i__3 = j + j * c_dim1;
            d__1 = *beta * c__[i__3].r;
/* 将结果赋值给 c[j, j] */
            c__[i__2].r = d__1, c__[i__2].i = 0.;
/* 如果 beta 等于 1 */
        } else {
/* 计算 c[j, j] 的索引 */
            i__2 = j + j * c_dim1;
/* 直接将 c[j, j] 的实数部分复制给自身，虚部置零 */
            i__3 = j + j * c_dim1;
            d__1 = c__[i__3].r;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        }
/* 对于每个 l 在 k 范围内的循环 */
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
/* 计算 a[j, l] 的索引 */
            i__3 = j + l * a_dim1;
/* 如果 a[j, l] 不为零 */
            if (a[i__3].r != 0. || a[i__3].i != 0.) {
/* 计算 alpha * conj(a[j, l]) 的复数乘积 */
            d_cnjg(&z__2, &a[j + l * a_dim1]);
            z__1.r = *alpha * z__2.r, z__1.i = *alpha * z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
/* 对于 j-1 范围内的每个 i，执行以下操作 */
            i__3 = j - 1;
            for (i__ = 1; i__ <= i__3; ++i__) {
/* 计算 c[i__, j] 的索引 */
                i__4 = i__ + j * c_dim1;
/* 计算 temp * a[i__, l] 的复数乘积 */
                i__5 = i__ + l * a_dim1;
                z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    z__2.i = temp.r * a[i__5].i + temp.i * a[
                    i__5].r;
/* 将结果添加到 c[i__, j] */
                z__1.r = c__[i__4].r + z__2.r, z__1.i = c__[i__4]
                    .i + z__2.i;
                c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
/* L110: */
            }
/* 计算 c[j, j] 的索引 */
            i__3 = j + j * c_dim1;
/* 计算 temp * a[j, l] 的复数乘积 */
            i__4 = j + l * a_dim1;
            z__1.r = temp.r * a[i__4].r - temp.i * a[i__4].i,
                z__1.i = temp.r * a[i__4].i + temp.i * a[i__4]
                .r;
/* 将结果赋值给 c[j, j] */
            d__1 = c__[i__3].r + z__1.r;
            c__[i__3].r = d__1, c__[i__3].i = 0.;
            }
/* L120: */
        }
/* L130: */
        }
    } else {
/* 对于每个 j 在 n 范围内的循环 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
/* 如果 beta 等于 0 */
        if (*beta == 0.) {
/* 对于 j 到 n 范围内的每个 i，执行以下操作 */
            i__2 = *n;
            for (i__ = j; i__ <= i__2; ++i__) {
/* 计算 c[i__, j] 的索引 */
            i__3 = i__ + j * c_dim1;
/* 将 c[i__, j] 设置为零 */
            c__[i__3].r = 0., c__[i__3].i = 0.;
/* L140: */
            }
        } else if (*beta != 1.) {
/* 计算 c[j, j] 的索引 */
            i__2 = j + j * c_dim1;
/* 计算 beta * c[j, j] 的实数乘积 */
            i__3 = j + j * c_dim1;
            d__1 = *beta * c__[i__3].r;
/* 将结果赋值给 c[j, j] */
            c__[i__2].r = d__1, c__[i__2].i = 0.;
/* 对于 j+1 到 n 范围内的每个 i，执行以下操作 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
/* 计算 c[i__, j] 的索引 */
            i__3 = i__ + j * c_dim1;
/* 计算 beta * c[i__, j] 的复数乘积 */
            i__4 = i__ + j * c_dim1;
            z__1.r = *beta * c__[i__4].r, z__1.i = *beta * c__[
                i__4].i;
/* 将结果赋值给 c[i__, j] */
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L150: */
            }
        } else {
            i__2 = j + j * c_dim1;  // 计算 C[j][j] 在一维数组中的索引
            i__3 = j + j * c_dim1;
            d__1 = c__[i__3].r;  // 获取 C[j][j] 的实部
            c__[i__2].r = d__1, c__[i__2].i = 0.;  // 设置 C[j][j] 的值为其实部，虚部为0
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            i__3 = j + l * a_dim1;  // 计算 A[j][l] 在一维数组中的索引
            if (a[i__3].r != 0. || a[i__3].i != 0.) {  // 检查 A[j][l] 是否非零
            d_cnjg(&z__2, &a[j + l * a_dim1]);  // 计算 A[j][l] 的共轭，并存储在 z__2 中
            z__1.r = *alpha * z__2.r, z__1.i = *alpha * z__2.i;  // 计算 alpha * conj(A[j][l])，结果存储在 z__1 中
            temp.r = z__1.r, temp.i = z__1.i;  // 将 z__1 的值复制给 temp
            i__3 = j + j * c_dim1;
            i__4 = j + j * c_dim1;
            i__5 = j + l * a_dim1;
            z__1.r = temp.r * a[i__5].r - temp.i * a[i__5].i,  // 计算 temp * A[j][l] 的实部
                z__1.i = temp.r * a[i__5].i + temp.i * a[i__5].r;  // 计算 temp * A[j][l] 的虚部
            d__1 = c__[i__4].r + z__1.r;  // 计算 C[j][j] 的新实部
            c__[i__3].r = d__1, c__[i__3].i = 0.;  // 更新 C[j][j]，虚部保持为0
            i__3 = *n;
            for (i__ = j + 1; i__ <= i__3; ++i__) {  // 遍历 j+1 到 n 的所有 i
                i__4 = i__ + j * c_dim1;
                i__5 = i__ + j * c_dim1;
                i__6 = i__ + l * a_dim1;
                z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i,  // 计算 temp * A[i][l] 的实部
                    z__2.i = temp.r * a[i__6].i + temp.i * a[i__6].r;  // 计算 temp * A[i][l] 的虚部
                z__1.r = c__[i__5].r + z__2.r, z__1.i = c__[i__5].i + z__2.i;  // 更新 C[i][j]
                c__[i__4].r = z__1.r, c__[i__4].i = z__1.i;
/* L160: */
            }
            }
/* L170: */
        }
/* L180: */
        }
    }
    } else {

/*        Form  C := alpha*conjg( A' )*A + beta*C. */

    if (upper) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {  // 遍历 j 从 1 到 n
        i__2 = j - 1;
        for (i__ = 1; i__ <= i__2; ++i__) {  // 遍历 i 从 1 到 j-1
            temp.r = 0., temp.i = 0.;  // 初始化 temp 为 0
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {  // 遍历 l 从 1 到 k
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);  // 计算 A[i][l] 的共轭，并存储在 z__3 中
            i__4 = l + j * a_dim1;
            z__2.r = z__3.r * a[i__4].r - z__3.i * a[i__4].i,  // 计算 conj(A[i][l]) * A[j][l] 的实部
                z__2.i = z__3.r * a[i__4].i + z__3.i * a[i__4].r;  // 计算 conj(A[i][l]) * A[j][l] 的虚部
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;  // 更新 temp
            temp.r = z__1.r, temp.i = z__1.i;
/* L190: */
            }
            if (*beta == 0.) {  // 如果 beta 等于 0
            i__3 = i__ + j * c_dim1;
            z__1.r = *alpha * temp.r, z__1.i = *alpha * temp.i;  // 计算 alpha * temp
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;  // 更新 C[i][j]
            } else {
            i__3 = i__ + j * c_dim1;
            z__2.r = *alpha * temp.r, z__2.i = *alpha * temp.i;  // 计算 alpha * temp
            i__4 = i__ + j * c_dim1;
            z__3.r = *beta * c__[i__4].r, z__3.i = *beta * c__[i__4].i;  // 计算 beta * C[i][j]
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;  // 计算 alpha * temp + beta * C[i][j]
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;  // 更新 C[i][j]
            }
/* L200: */
        }
/* L210: */
        }
    }
/* L200: */
        }
        rtemp = 0.;
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            d_cnjg(&z__3, &a[l + j * a_dim1]);
            i__3 = l + j * a_dim1;
            z__2.r = z__3.r * a[i__3].r - z__3.i * a[i__3].i, z__2.i =
                 z__3.r * a[i__3].i + z__3.i * a[i__3].r;
            z__1.r = rtemp + z__2.r, z__1.i = z__2.i;
            rtemp = z__1.r;
/* L210: */
        }
        if (*beta == 0.) {
            i__2 = j + j * c_dim1;
            d__1 = *alpha * rtemp;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        } else {
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            d__1 = *alpha * rtemp + *beta * c__[i__3].r;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        }
/* L220: */
        }
    } else {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
        rtemp = 0.;
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
            d_cnjg(&z__3, &a[l + j * a_dim1]);
            i__3 = l + j * a_dim1;
            z__2.r = z__3.r * a[i__3].r - z__3.i * a[i__3].i, z__2.i =
                 z__3.r * a[i__3].i + z__3.i * a[i__3].r;
            z__1.r = rtemp + z__2.r, z__1.i = z__2.i;
            rtemp = z__1.r;
/* L230: */
        }
        if (*beta == 0.) {
            i__2 = j + j * c_dim1;
            d__1 = *alpha * rtemp;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        } else {
            i__2 = j + j * c_dim1;
            i__3 = j + j * c_dim1;
            d__1 = *alpha * rtemp + *beta * c__[i__3].r;
            c__[i__2].r = d__1, c__[i__2].i = 0.;
        }
        i__2 = *n;
        for (i__ = j + 1; i__ <= i__2; ++i__) {
            temp.r = 0., temp.i = 0.;
            i__3 = *k;
            for (l = 1; l <= i__3; ++l) {
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);
            i__4 = l + j * a_dim1;
            z__2.r = z__3.r * a[i__4].r - z__3.i * a[i__4].i,
                z__2.i = z__3.r * a[i__4].i + z__3.i * a[i__4]
                .r;
            z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
            temp.r = z__1.r, temp.i = z__1.i;
/* L240: */
            }
            if (*beta == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.r = *alpha * temp.r, z__1.i = *alpha * temp.i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            } else {
            i__3 = i__ + j * c_dim1;
            z__2.r = *alpha * temp.r, z__2.i = *alpha * temp.i;
            i__4 = i__ + j * c_dim1;
            z__3.r = *beta * c__[i__4].r, z__3.i = *beta * c__[
                i__4].i;
            z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
            c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
            }
/* L250: */
        }
/* L260: */
        }
    }
    }

    return 0;

/*     End of ZHERK . */

} /* zherk_ */

/* Subroutine */ int zscal_(integer *n, doublecomplex *za, doublecomplex *zx,
    integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3;


注释：
    doublecomplex z__1;
    # 声明一个复数变量 z__1，类型为 doublecomplex

    /* Local variables */
    static integer i__, ix;
    # 声明两个静态局部变量：i__ 和 ix，它们的类型分别为 integer
/*
    Purpose
    =======

       ZSCAL scales a vector by a constant.

    Further Details
    ===============

       jack dongarra, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

/* Parameter adjustments */
--zx;   // 减小指向向量 zx 的指针

/* Function Body */
if (*n <= 0 || *incx <= 0) {
return 0;   // 如果输入的向量长度或增量小于等于零，直接返回
}
if (*incx == 1) {
goto L20;   // 如果增量为 1，则跳转到标签 L20
}

/*        code for increment not equal to 1 */

ix = 1;   // 初始化索引 ix 为 1
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
i__2 = ix;
i__3 = ix;
z__1.r = za->r * zx[i__3].r - za->i * zx[i__3].i, z__1.i = za->r * zx[
    i__3].i + za->i * zx[i__3].r;
zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;   // 计算并更新 zx[ix] 的值
ix += *incx;   // 根据增量更新 ix
/* L10: */
}
return 0;

/*        code for increment equal to 1 */

L20:
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
i__2 = i__;
i__3 = i__;
z__1.r = za->r * zx[i__3].r - za->i * zx[i__3].i, z__1.i = za->r * zx[
    i__3].i + za->i * zx[i__3].r;
zx[i__2].r = z__1.r, zx[i__2].i = z__1.i;   // 计算并更新 zx[i] 的值
/* L30: */
}
return 0;
} /* zscal_ */

/* Subroutine */ int zswap_(integer *n, doublecomplex *zx, integer *incx,
    doublecomplex *zy, integer *incy)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Local variables */
    static integer i__, ix, iy;
    static doublecomplex ztemp;


/*
    Purpose
    =======

       ZSWAP interchanges two vectors.

    Further Details
    ===============

       jack dongarra, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)

    =====================================================================
*/

/* Parameter adjustments */
--zy;   // 减小指向向量 zy 的指针
--zx;   // 减小指向向量 zx 的指针

/* Function Body */
if (*n <= 0) {
return 0;   // 如果输入的向量长度小于等于零，直接返回
}
if (*incx == 1 && *incy == 1) {
goto L20;   // 如果增量均为 1，则跳转到标签 L20
}

/*
     code for unequal increments or equal increments not equal
       to 1
*/

ix = 1;   // 初始化 ix 为 1
iy = 1;   // 初始化 iy 为 1
if (*incx < 0) {
ix = (-(*n) + 1) * *incx + 1;   // 如果增量 incx 小于零，更新 ix
}
if (*incy < 0) {
iy = (-(*n) + 1) * *incy + 1;   // 如果增量 incy 小于零，更新 iy
}
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
i__2 = ix;
ztemp.r = zx[i__2].r, ztemp.i = zx[i__2].i;   // 临时保存 zx[ix] 的值
i__2 = ix;
i__3 = iy;
zx[i__2].r = zy[i__3].r, zx[i__2].i = zy[i__3].i;   // 交换 zx[ix] 和 zy[iy] 的值
i__2 = iy;
zy[i__2].r = ztemp.r, zy[i__2].i = ztemp.i;   // 将保存的值赋给 zy[iy]
ix += *incx;   // 根据增量更新 ix
iy += *incy;   // 根据增量更新 iy
/* L10: */
}
return 0;

/*       code for both increments equal to 1 */
L20:
i__1 = *n;
for (i__ = 1; i__ <= i__1; ++i__) {
i__2 = i__;
ztemp.r = zx[i__2].r, ztemp.i = zx[i__2].i;   // 临时保存 zx[i] 的值
i__2 = i__;
i__3 = i__;
zx[i__2].r = zy[i__3].r, zx[i__2].i = zy[i__3].i;   // 交换 zx[i] 和 zy[i] 的值
i__2 = i__;
zy[i__2].r = ztemp.r, zy[i__2].i = ztemp.i;   // 将保存的值赋给 zy[i]
/* L30: */
}
return 0;
} /* zswap_ */
    integer *m, integer *n, doublecomplex *alpha, doublecomplex *a,
    integer *lda, doublecomplex *b, integer *ldb)



    # 接收指向整数 m 的指针参数，用于指定矩阵 A 的行数
    integer *m,
    # 接收指向整数 n 的指针参数，用于指定矩阵 B 的列数
    integer *n,
    # 接收指向复数 alpha 的指针参数，用于指定矩阵 A 的缩放因子
    doublecomplex *alpha,
    # 接收指向复数矩阵 A 的指针参数，表示矩阵 A
    doublecomplex *a,
    # 接收指向整数 lda 的指针参数，表示矩阵 A 的列数
    integer *lda,
    # 接收指向复数矩阵 B 的指针参数，表示矩阵 B
    doublecomplex *b,
    # 接收指向整数 ldb 的指针参数，表示矩阵 B 的列数
    integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5,
        i__6;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, j, k, info;
    static doublecomplex temp;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;


/*
    Purpose
    =======

    ZTRMM  performs one of the matrix-matrix operations

       B := alpha*op( A )*B,   or   B := alpha*B*op( A )

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).

    Arguments
    ==========

    SIDE   - CHARACTER*1.
             On entry,  SIDE specifies whether  op( A ) multiplies B from
             the left or right as follows:

                SIDE = 'L' or 'l'   B := alpha*op( A )*B.

                SIDE = 'R' or 'r'   B := alpha*B*op( A ).

             Unchanged on exit.

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix A is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n'   op( A ) = A.

                TRANSA = 'T' or 't'   op( A ) = A'.

                TRANSA = 'C' or 'c'   op( A ) = conjg( A' ).

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit triangular
             as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of B. M must be at
             least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of B.  N must be
             at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16      .
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is
             zero then  A is not referenced and  B need not be set before
             entry.
             Unchanged on exit.
*/

    /* Initialize flags and variables */
    static integer i__, j, k, info;
    static doublecomplex temp;
    static logical lside;
    static integer nrowa;
    static logical upper;
    static logical noconj, nounit;

    /* Check if A and B are compatible */
    if (*m < 0 || *n < 0) {
        /* Handle error: dimensions of B are invalid */
        xerbla_("ZTRMM ", &c__9);
        return 0;
    }

    /* Quick return if possible */
    if (*m == 0 || *n == 0 || alpha->r == 0. && alpha->i == 0.) {
        return 0;
    }

    /* Set flags for type of operation */
    lside = lsame_(side, "L") || lsame_(side, "l");
    upper = lsame_(uplo, "U") || lsame_(uplo, "u");
    noconj = lsame_(transa, "T") || lsame_(transa, "t");
    nounit = lsame_(diag, "N") || lsame_(diag, "n");

    /* Determine the number of rows in A */
    if (lside) {
        nrowa = *m;
    } else {
        nrowa = *n;
    }

    /* Perform matrix multiplication */
    if (lside) {
        /* Case when op(A) is applied from the left */
        if (upper) {
            /* Case when A is upper triangular */
            if (noconj) {
                /* Case when op(A) = A */
                /* Perform B := alpha*A*B */
                for (j = 1; j <= *n; ++j) {
                    for (i__ = *m; i__ >= 1; --i__) {
                        z__1.r = alpha->r * a[i__ + j * a_dim1].r - alpha->i * a[i__ + j * a_dim1].i;
                        z__1.i = alpha->r * a[i__ + j * a_dim1].i + alpha->i * a[i__ + j * a_dim1].r;
                        temp.r = z__1.r, temp.i = z__1.i;
                        for (k = 1; k <= i__ - 1; ++k) {
                            i__1 = k + j * b_dim1;
                            i__2 = k + j * b_dim1;
                            z__2.r = temp.r * a[k + i__ * a_dim1].r - temp.i * a[k + i__ * a_dim1].i;
                            z__2.i = temp.r * a[k + i__ * a_dim1].i + temp.i * a[k + i__ * a_dim1].r;
                            z__1.r = b[i__1].r + z__2.r, z__1.i = b[i__1].i + z__2.i;
                            b[i__2].r = z__1.r, b[i__2].i = z__1.i;
                            /* L10: */
                        }
                        if (alpha->r == 1. && alpha->i == 0.) {
                            /* Case when alpha = 1 (identity operation) */
                            /* B[i__, j] := B[i__, j] + A[i__, j] */
                            i__1 = i__ + j * b_dim1;
                            i__2 = i__ + j * a_dim1;
                            z__1.r = b[i__1].r + a[i__2].r, z__1.i = b[i__1].i + a[i__2].i;
                            b[i__1].r = z__1.r, b[i__1].i = z__1.i;
                        } else if (alpha->r == 0. && alpha->i == 0.) {
                            /* Case when alpha = 0 (A is not referenced) */
                            /* No operation */
                        } else {
                            /* General case */
                            /* B[i__, j] := alpha * A[i__, j] + B[i__, j] */
                            i__1 = i__ + j * a_dim1;
                            z__1.r = alpha->r * a[i__1].r - alpha->i * a[i__1].i;
                            z__1.i = alpha->r * a[i__1].i + alpha->i * a[i__1].r;
                            temp.r = z__1.r, temp.i = z__1.i;
                            i__1 = i__ + j * b_dim1;
                            z__2.r = temp.r * b[i__1].r - temp.i * b[i__1].i;
                            z__2.i = temp.r * b[i__1].i + temp.i * b[i__1].r;
                            z__1.r = b[i__1].r + z__2.r, z__1.i = b[i__1].i + z__2.i;
                            b[i__1].r = z__1.r, b[i__1].i = z__1.i;
                        }
                        /* L20: */
                    }
                    /* L30: */
                }
            } else {
                /* Case when op(A) = conjg(A') */
                /* Perform B := alpha*conjg(A')*B */
                for (j = 1; j <= *n; ++j) {
                    for (i__ = *m; i__ >= 1; --i__) {
                        z__1.r = alpha->r * conjg(a[i__ + j * a_dim1]).r - alpha->i * conjg(a[i__ + j * a_dim1]).i;
                        z__1.i = alpha->r * conjg(a[i__ + j * a_dim1]).i + alpha->i * conjg(a[i__ + j * a_dim1]).r;
                        temp.r = z__1.r, temp.i = z__1.i;
                        for (k = 1; k <= i__ - 1; ++k) {
                            i__1 = k + j * b_dim1;
                            i__2 =
    A      - COMPLEX*16       array of DIMENSION ( LDA, k ), where k is m
             when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
             Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
             upper triangular part of the array  A must contain the upper
             triangular matrix  and the strictly lower triangular part of
             A is not referenced.
             Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
             lower triangular part of the array  A must contain the lower
             triangular matrix  and the strictly upper triangular part of
             A is not referenced.
             Note that when  DIAG = 'U' or 'u',  the diagonal elements of
             A  are not referenced either,  but are assumed to be  unity.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
             then LDA must be at least max( 1, n ).
             Unchanged on exit.

    B      - COMPLEX*16       array of DIMENSION ( LDB, n ).
             Before entry,  the leading  m by n part of the array  B must
             contain the matrix  B,  and  on exit  is overwritten  by the
             transformed matrix.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   LDB  must  be  at  least
             max( 1, m ).
             Unchanged on exit.

    Further Details
    ===============

    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

    =====================================================================


       Test the input parameters.
    /* Parameter adjustments */
    // 调整参数，计算矩阵a和b的维度参数
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    // 判断side参数是否为'L'，确定矩阵a的行数
    lside = lsame_(side, "L");
    if (lside) {
        nrowa = *m;
    } else {
        nrowa = *n;
    }
    // 判断transa参数是否为'T'，确定是否不共轭传递
    noconj = lsame_(transa, "T");
    // 判断diag参数是否为'N'，确定是否不是单位对角矩阵
    nounit = lsame_(diag, "N");
    // 判断uplo参数是否为'U'，确定是否为上三角矩阵
    upper = lsame_(uplo, "U");

    info = 0;
    // 检查输入参数是否正确，若不正确设置错误代码info
    if (!lside && !lsame_(side, "R")) {
        info = 1;
    } else if (!upper && !lsame_(uplo, "L")) {
        info = 2;
    } else if (!lsame_(transa, "N") && !lsame_(transa, "T") && !lsame_(transa, "C")) {
        info = 3;
    } else if (!lsame_(diag, "U") && !lsame_(diag, "N")) {
        info = 4;
    } else if (*m < 0) {
        info = 5;
    } else if (*n < 0) {
        info = 6;
    } else if (*lda < max(1, nrowa)) {
        info = 9;
    } else if (*ldb < max(1, *m)) {
        info = 11;
    }
    // 如果有错误信息，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("ZTRMM ", &info);
        return 0;
    }

/*     Quick return if possible. */

    // 若m或n为0，直接返回
    if (*m == 0 || *n == 0) {
        return 0;
    }

/*     And when  alpha.eq.zero. */

    // 若alpha为零，则将矩阵B中的元素全部置零
    if (alpha->r == 0. && alpha->i == 0.) {
        // 遍历矩阵B，并将每个元素设为零
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                b[i__3].r = 0., b[i__3].i = 0.;
            }
        }
        return 0;
    }

/*     Start the operations. */

    // 开始进行矩阵乘法运算
    if (lside) {
        if (lsame_(transa, "N")) {

/*           Form  B := alpha*A*B. */

            // 若不需要转置矩阵A，则执行 B := alpha*A*B
            if (upper) {
                // 若A为上三角矩阵，进行相应乘法操作
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = *m;
                    for (k = 1; k <= i__2; ++k) {
                        i__3 = k + j * b_dim1;
                        // 计算临时变量temp，并更新矩阵B中的元素
                        if (b[i__3].r != 0. || b[i__3].i != 0.) {
                            i__3 = k + j * b_dim1;
                            z__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i, z__1.i = alpha->r * b[i__3].i + alpha->i * b[i__3].r;
                            temp.r = z__1.r, temp.i = z__1.i;
                            i__3 = k - 1;
                            for (i__ = 1; i__ <= i__3; ++i__) {
                                i__4 = i__ + j * b_dim1;
                                i__5 = i__ + j * b_dim1;
                                i__6 = i__ + k * a_dim1;
                                z__2.r = temp.r * a[i__6].r - temp.i * a[i__6].i, z__2.i = temp.r * a[i__6].i + temp.i * a[i__6].r;
                                z__1.r = b[i__5].r + z__2.r, z__1.i = b[i__5].i + z__2.i;
                                b[i__4].r = z__1.r, b[i__4].i = z__1.i;
                            }
                            // 若A不是单位对角矩阵，更新temp并再次更新矩阵B中的元素
                            if (nounit) {
                                i__3 = k + k * a_dim1;
                                z__1.r = temp.r * a[i__3].r - temp.i * a[i__3].i, z__1.i = temp.r * a[i__3].i + temp.i * a[i__3].r;
                                temp.r = z__1.r, temp.i = z__1.i;
                            }
                            i__3 = k + j * b_dim1;
                            b[i__3].r = temp.r, b[i__3].i = temp.i;
                        }
                    }
                }
            }
        }
    }
/* L50: */
        }
        } else {
        i__1 = *n;
        // 循环遍历列
        for (j = 1; j <= i__1; ++j) {
            // 循环遍历行，从后向前
            for (k = *m; k >= 1; --k) {
            // 计算在矩阵 b 中的索引
            i__2 = k + j * b_dim1;
            // 检查 b 元素是否非零
            if (b[i__2].r != 0. || b[i__2].i != 0.) {
                // 计算乘法 alpha * b[k + j*b_dim1]
                i__2 = k + j * b_dim1;
                z__1.r = alpha->r * b[i__2].r - alpha->i * b[i__2]
                    .i, z__1.i = alpha->r * b[i__2].i +
                    alpha->i * b[i__2].r;
                temp.r = z__1.r, temp.i = z__1.i;
                // 更新 b[k + j*b_dim1] 的值为乘积结果
                i__2 = k + j * b_dim1;
                b[i__2].r = temp.r, b[i__2].i = temp.i;
                // 如果非单位矩阵，执行进一步乘法
                if (nounit) {
                i__2 = k + j * b_dim1;
                i__3 = k + j * b_dim1;
                i__4 = k + k * a_dim1;
                z__1.r = b[i__3].r * a[i__4].r - b[i__3].i *
                    a[i__4].i, z__1.i = b[i__3].r * a[
                    i__4].i + b[i__3].i * a[i__4].r;
                b[i__2].r = z__1.r, b[i__2].i = z__1.i;
                }
                // 更新剩余的列元素
                i__2 = *m;
                for (i__ = k + 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + k * a_dim1;
                z__2.r = temp.r * a[i__5].r - temp.i * a[i__5]
                    .i, z__2.i = temp.r * a[i__5].i +
                    temp.i * a[i__5].r;
                z__1.r = b[i__4].r + z__2.r, z__1.i = b[i__4]
                    .i + z__2.i;
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L60: */
                }
            }
/* L70: */
            }
/* L80: */
        }
        }
    } else {

/*           Form  B := alpha*A'*B   or   B := alpha*conjg( A' )*B. */

        if (upper) {
        i__1 = *n;
        // 循环遍历列
        for (j = 1; j <= i__1; ++j) {
            // 循环遍历行，从后向前
            for (i__ = *m; i__ >= 1; --i__) {
            // 计算在矩阵 b 中的索引
            i__2 = i__ + j * b_dim1;
            temp.r = b[i__2].r, temp.i = b[i__2].i;
            // 如果不进行共轭操作
            if (noconj) {
                // 如果非单位矩阵，执行乘法
                if (nounit) {
                i__2 = i__ + i__ * a_dim1;
                z__1.r = temp.r * a[i__2].r - temp.i * a[i__2]
                    .i, z__1.i = temp.r * a[i__2].i +
                    temp.i * a[i__2].r;
                temp.r = z__1.r, temp.i = z__1.i;
                }
                // 更新剩余的行元素
                i__2 = i__ - 1;
                for (k = 1; k <= i__2; ++k) {
                i__3 = k + i__ * a_dim1;
                i__4 = k + j * b_dim1;
                z__2.r = a[i__3].r * b[i__4].r - a[i__3].i *
                    b[i__4].i, z__2.i = a[i__3].r * b[
                    i__4].i + a[i__3].i * b[i__4].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
                }
/* L90: */
                }
            } else {
                // 如果 nounit 为真，计算 alpha * temp * conj(a[i,i])
                if (nounit) {
                    d_cnjg(&z__2, &a[i__ + i__ * a_dim1]);  // 计算 a[i,i] 的共轭
                    z__1.r = temp.r * z__2.r - temp.i * z__2.i,  // 计算实部
                        z__1.i = temp.r * z__2.i + temp.i * z__2.r;  // 计算虚部
                    temp.r = z__1.r, temp.i = z__1.i;  // 更新 temp
                }
                i__2 = i__ - 1;
                // 计算 alpha * temp * conj(a[1:i-1,i]) 的总和
                for (k = 1; k <= i__2; ++k) {
                    d_cnjg(&z__3, &a[k + i__ * a_dim1]);  // 计算 a[k,i] 的共轭
                    i__3 = k + j * b_dim1;
                    z__2.r = z__3.r * b[i__3].r - z__3.i * b[i__3].i,  // 计算实部
                        z__2.i = z__3.r * b[i__3].i + z__3.i * b[i__3].r;  // 计算虚部
                    z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;  // 更新 temp
                    temp.r = z__1.r, temp.i = z__1.i;
/* L100: */
                }
            }
            i__2 = i__ + j * b_dim1;
            // 计算 alpha * temp，更新 b[i,j]
            z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                z__1.i = alpha->r * temp.i + alpha->i * temp.r;
            b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L110: */
            }
/* L120: */
        }
        } else {
        // 处理下三角矩阵的情况
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * b_dim1;
            temp.r = b[i__3].r, temp.i = b[i__3].i;
            if (noconj) {
                // 如果 noconj 为真，计算 alpha * temp * a[i,i]
                if (nounit) {
                    i__3 = i__ + i__ * a_dim1;
                    z__1.r = temp.r * a[i__3].r - temp.i * a[i__3].i,  // 计算实部
                        z__1.i = temp.r * a[i__3].i + temp.i * a[i__3].r;  // 计算虚部
                    temp.r = z__1.r, temp.i = z__1.i;  // 更新 temp
                }
                i__3 = *m;
                // 计算 alpha * temp * a[i+1:m,i] 的总和
                for (k = i__ + 1; k <= i__3; ++k) {
                    i__4 = k + i__ * a_dim1;
                    i__5 = k + j * b_dim1;
                    z__2.r = a[i__4].r * b[i__5].r - a[i__4].i * b[i__5].i,  // 计算实部
                        z__2.i = a[i__4].r * b[i__5].i + a[i__4].i * b[i__5].r;  // 计算虚部
                    z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;  // 更新 temp
                    temp.r = z__1.r, temp.i = z__1.i;
/* L130: */
                }
            } else {
                // 如果 noconj 为假，计算 alpha * temp * conj(a[i,i])
                if (nounit) {
                    d_cnjg(&z__2, &a[i__ + i__ * a_dim1]);  // 计算 a[i,i] 的共轭
                    z__1.r = temp.r * z__2.r - temp.i * z__2.i,  // 计算实部
                        z__1.i = temp.r * z__2.i + temp.i * z__2.r;  // 计算虚部
                    temp.r = z__1.r, temp.i = z__1.i;  // 更新 temp
                }
                i__3 = *m;
                // 计算 alpha * temp * conj(a[i+1:m,i]) 的总和
                for (k = i__ + 1; k <= i__3; ++k) {
                    d_cnjg(&z__3, &a[k + i__ * a_dim1]);  // 计算 a[k,i] 的共轭
                    i__4 = k + j * b_dim1;
                    z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4].i,  // 计算实部
                        z__2.i = z__3.r * b[i__4].i + z__3.i * b[i__4].r;  // 计算虚部
                    z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;  // 更新 temp
                    temp.r = z__1.r, temp.i = z__1.i;
/* L140: */
                }
            }
/* L150: */
            }
/* L160: */
        }
        }
/* L140: */
                }
            }
            // 计算矩阵乘积 B := alpha*B*A
            i__3 = i__ + j * b_dim1;
            z__1.r = alpha->r * temp.r - alpha->i * temp.i,
                z__1.i = alpha->r * temp.i + alpha->i *
                temp.r;
            b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L150: */
            }
/* L160: */
        }
        }
    }
    } else {
    // 如果 transa != "N"，进行另一种矩阵乘积计算
    if (lsame_(transa, "N")) {

/*           Form  B := alpha*B*A. */

        // 如果矩阵 A 是上三角矩阵
        if (upper) {
        // 从最后一列开始遍历
        for (j = *n; j >= 1; --j) {
            // 设置临时变量 temp 为 alpha
            temp.r = alpha->r, temp.i = alpha->i;
            // 如果 A 不是单位矩阵，更新 temp
            if (nounit) {
            i__1 = j + j * a_dim1;
            z__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                z__1.i = temp.r * a[i__1].i + temp.i * a[i__1]
                .r;
            temp.r = z__1.r, temp.i = z__1.i;
            }
            // 对每一行 i 进行更新
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
            i__2 = i__ + j * b_dim1;
            i__3 = i__ + j * b_dim1;
            z__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                z__1.i = temp.r * b[i__3].i + temp.i * b[i__3]
                .r;
            b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L170: */
            }
            // 对每一列 k 进行更新
            i__1 = j - 1;
            for (k = 1; k <= i__1; ++k) {
            i__2 = k + j * a_dim1;
            // 如果 A[k,j] 不为零，更新 temp
            if (a[i__2].r != 0. || a[i__2].i != 0.) {
                i__2 = k + j * a_dim1;
                z__1.r = alpha->r * a[i__2].r - alpha->i * a[i__2]
                    .i, z__1.i = alpha->r * a[i__2].i +
                    alpha->i * a[i__2].r;
                temp.r = z__1.r, temp.i = z__1.i;
                // 对每一行 i 进行更新
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + k * b_dim1;
                z__2.r = temp.r * b[i__5].r - temp.i * b[i__5]
                    .i, z__2.i = temp.r * b[i__5].i +
                    temp.i * b[i__5].r;
                z__1.r = b[i__4].r + z__2.r, z__1.i = b[i__4]
                    .i + z__2.i;
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L180: */
                }
            }
/* L190: */
            }
/* L200: */
        }
        } else {
        // 如果矩阵 A 是下三角矩阵
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 设置临时变量 temp 为 alpha
            temp.r = alpha->r, temp.i = alpha->i;
            // 如果 A 不是单位矩阵，更新 temp
            if (nounit) {
            i__2 = j + j * a_dim1;
            z__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                z__1.i = temp.r * a[i__2].i + temp.i * a[i__2]
                .r;
            temp.r = z__1.r, temp.i = z__1.i;
            }
            // 对每一行 i 进行更新
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * b_dim1;
            i__4 = i__ + j * b_dim1;
            z__1.r = temp.r * b[i__4].r - temp.i * b[i__4].i,
                z__1.i = temp.r * b[i__4].i + temp.i * b[i__4]
                .r;
            b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L210: */
            }
            // 循环：对每个列 k，执行下面的操作
            i__2 = *n;
            for (k = j + 1; k <= i__2; ++k) {
            // 判断条件：如果矩阵元素 a[k+j*a_dim1] 不为零
            i__3 = k + j * a_dim1;
            if (a[i__3].r != 0. || a[i__3].i != 0.) {
                // 计算 alpha 和 a[k+j*a_dim1] 的乘积
                i__3 = k + j * a_dim1;
                z__1.r = alpha->r * a[i__3].r - alpha->i * a[i__3]
                    .i, z__1.i = alpha->r * a[i__3].i +
                    alpha->i * a[i__3].r;
                temp.r = z__1.r, temp.i = z__1.i;
                // 循环：对每个行 i，执行下面的操作
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                // 计算 temp 和 b[i+j*b_dim1] 以及 temp 与 b[i+k*b_dim1] 的乘积
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + j * b_dim1;
                i__6 = i__ + k * b_dim1;
                z__2.r = temp.r * b[i__6].r - temp.i * b[i__6]
                    .i, z__2.i = temp.r * b[i__6].i +
                    temp.i * b[i__6].r;
                z__1.r = b[i__5].r + z__2.r, z__1.i = b[i__5]
                    .i + z__2.i;
                b[i__4].r = z__1.r, b[i__4].i = z__1.i;
/* L220: */
                }
            }
/* L230: */
            }
/* L240: */
        }
        }
    } else {

/*           Form  B := alpha*B*A'   or   B := alpha*B*conjg( A' ). */

        if (upper) {
        // 如果 upper 为真，则执行下面的操作
        i__1 = *n;
        for (k = 1; k <= i__1; ++k) {
            // 循环：对每个列 k，执行下面的操作
            i__2 = k - 1;
            for (j = 1; j <= i__2; ++j) {
            // 判断条件：如果矩阵元素 a[j+k*a_dim1] 不为零
            i__3 = j + k * a_dim1;
            if (a[i__3].r != 0. || a[i__3].i != 0.) {
                // 判断条件：如果 noconj 为真，则执行下面的操作
                if (noconj) {
                // 计算 alpha 和 a[j+k*a_dim1] 的乘积
                i__3 = j + k * a_dim1;
                z__1.r = alpha->r * a[i__3].r - alpha->i * a[
                    i__3].i, z__1.i = alpha->r * a[i__3]
                    .i + alpha->i * a[i__3].r;
                temp.r = z__1.r, temp.i = z__1.i;
                } else {
                // 否则，计算 alpha 和 a[j+k*a_dim1] 的共轭乘积
                d_cnjg(&z__2, &a[j + k * a_dim1]);
                z__1.r = alpha->r * z__2.r - alpha->i *
                    z__2.i, z__1.i = alpha->r * z__2.i +
                    alpha->i * z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
                }
                // 循环：对每个行 i，执行下面的操作
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                // 计算 temp 和 b[i+j*b_dim1] 以及 temp 与 b[i+k*b_dim1] 的乘积
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + j * b_dim1;
                i__6 = i__ + k * b_dim1;
                z__2.r = temp.r * b[i__6].r - temp.i * b[i__6]
                    .i, z__2.i = temp.r * b[i__6].i +
                    temp.i * b[i__6].r;
                z__1.r = b[i__5].r + z__2.r, z__1.i = b[i__5]
                    .i + z__2.i;
                b[i__4].r = z__1.r, b[i__4].i = z__1.i;
/* L250: */
                }
            }
/* L260: */
            }
            // 将 alpha 的值赋给 temp
            temp.r = alpha->r, temp.i = alpha->i;
            // 如果不是单位矩阵，根据 noconj 的值处理 temp
            if (nounit) {
                if (noconj) {
                    // 计算 a[k + k * a_dim1] 与 temp 的乘积
                    i__2 = k + k * a_dim1;
                    z__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                        z__1.i = temp.r * a[i__2].i + temp.i * a[i__2].r;
                    temp.r = z__1.r, temp.i = z__1.i;
                } else {
                    // 对 a[k + k * a_dim1] 取共轭，然后计算与 temp 的乘积
                    d_cnjg(&z__2, &a[k + k * a_dim1]);
                    z__1.r = temp.r * z__2.r - temp.i * z__2.i,
                        z__1.i = temp.r * z__2.i + temp.i * z__2.r;
                    temp.r = z__1.r, temp.i = z__1.i;
                }
            }
            // 如果 temp 不等于 (1, 0)，则更新矩阵 b 的相关元素
            if (temp.r != 1. || temp.i != 0.) {
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    // 计算更新 b[i__, k] 的值
                    i__3 = i__ + k * b_dim1;
                    i__4 = i__ + k * b_dim1;
                    z__1.r = temp.r * b[i__4].r - temp.i * b[i__4].i,
                        z__1.i = temp.r * b[i__4].i + temp.i * b[i__4].r;
                    b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L270: */
                }
            }
/* L280: */
        }
        // 若 nounit 为假，则执行下面的循环
        } else {
            // 逆序遍历 k 从 *n 到 1
            for (k = *n; k >= 1; --k) {
                // 正序遍历 j 从 k+1 到 *n
                i__1 = *n;
                for (j = k + 1; j <= i__1; ++j) {
                    // 判断 a[j + k * a_dim1] 是否为非零
                    i__2 = j + k * a_dim1;
                    if (a[i__2].r != 0. || a[i__2].i != 0.) {
                        if (noconj) {
                            // 计算 alpha 与 a[j + k * a_dim1] 的乘积
                            i__2 = j + k * a_dim1;
                            z__1.r = alpha->r * a[i__2].r - alpha->i * a[i__2].i,
                                z__1.i = alpha->r * a[i__2].i + alpha->i * a[i__2].r;
                            temp.r = z__1.r, temp.i = z__1.i;
                        } else {
                            // 对 a[j + k * a_dim1] 取共轭，然后计算与 alpha 的乘积
                            d_cnjg(&z__2, &a[j + k * a_dim1]);
                            z__1.r = alpha->r * z__2.r - alpha->i * z__2.i,
                                z__1.i = alpha->r * z__2.i + alpha->i * z__2.r;
                            temp.r = z__1.r, temp.i = z__1.i;
                        }
                        // 更新 b 的相关元素
                        i__2 = *m;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            i__3 = i__ + j * b_dim1;
                            i__4 = i__ + j * b_dim1;
                            i__5 = i__ + k * b_dim1;
                            z__2.r = temp.r * b[i__5].r - temp.i * b[i__5].i,
                                z__2.i = temp.r * b[i__5].i + temp.i * b[i__5].r;
                            z__1.r = b[i__4].r + z__2.r, z__1.i = b[i__4].i + z__2.i;
                            b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L290: */
                        }
                    }
                    /* L270: */
                }
/* L280: */
            }
        }
/* L300: */
            }
            // 临时变量 temp 设置为 alpha 对应的值
            temp.r = alpha->r, temp.i = alpha->i;
            // 如果矩阵 A 非单位矩阵
            if (nounit) {
            // 如果不进行共轭操作
            if (noconj) {
                // 计算 A(k, k) * temp，并更新 temp
                i__1 = k + k * a_dim1;
                z__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                    z__1.i = temp.r * a[i__1].i + temp.i * a[
                    i__1].r;
                temp.r = z__1.r, temp.i = z__1.i;
            } else {
                // 计算共轭转置的 A(k, k) * temp，并更新 temp
                d_cnjg(&z__2, &a[k + k * a_dim1]);
                z__1.r = temp.r * z__2.r - temp.i * z__2.i,
                    z__1.i = temp.r * z__2.i + temp.i *
                    z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
            }
            // 如果 temp 不等于 (1,0)，执行更新矩阵 B 的操作
            if (temp.r != 1. || temp.i != 0.) {
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                // 计算 temp * B(i, k) 并更新 B(i, k)
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                z__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    z__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L310: */
            }
            }
/* L320: */
        }
        }
    }
    }

    // 返回 0 表示函数执行成功

/*     End of ZTRMM . */

} /* ztrmm_ */

/* Subroutine */ int ztrmv_(char *uplo, char *trans, char *diag, integer *n,
    doublecomplex *a, integer *lda, doublecomplex *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, j, ix, jx, kx, info;
    static doublecomplex temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;


/*
    Purpose
    =======

    ZTRMV  performs one of the matrix-vector operations

       x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   x := A*x.

                TRANS = 'T' or 't'   x := A'*x.

                TRANS = 'C' or 'c'   x := conjg( A' )*x.

             Unchanged on exit.

*/
    ! DIAG   - CHARACTER*1.
    !          On entry, DIAG specifies whether or not A is unit
    !          triangular as follows:
    !
    !             DIAG = 'U' or 'u'   A is assumed to be unit triangular.
    !
    !             DIAG = 'N' or 'n'   A is not assumed to be unit
    !                                 triangular.
    !
    !          Unchanged on exit.

    ! N      - INTEGER.
    !          On entry, N specifies the order of the matrix A.
    !          N must be at least zero.
    !          Unchanged on exit.

    ! A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
    !          Before entry with  UPLO = 'U' or 'u', the leading n by n
    !          upper triangular part of the array A must contain the upper
    !          triangular matrix and the strictly lower triangular part of
    !          A is not referenced.
    !          Before entry with UPLO = 'L' or 'l', the leading n by n
    !          lower triangular part of the array A must contain the lower
    !          triangular matrix and the strictly upper triangular part of
    !          A is not referenced.
    !          Note that when  DIAG = 'U' or 'u', the diagonal elements of
    !          A are not referenced either, but are assumed to be unity.
    !          Unchanged on exit.

    ! LDA    - INTEGER.
    !          On entry, LDA specifies the first dimension of A as declared
    !          in the calling (sub) program. LDA must be at least
    !          max( 1, n ).
    !          Unchanged on exit.

    ! X      - COMPLEX*16       array of dimension at least
    !          ( 1 + ( n - 1 )*abs( INCX ) ).
    !          Before entry, the incremented array X must contain the n
    !          element vector x. On exit, X is overwritten with the
    !          transformed vector x.

    ! INCX   - INTEGER.
    !          On entry, INCX specifies the increment for the elements of
    !          X. INCX must not be zero.
    !          Unchanged on exit.

    ! Further Details
    ! ===============

    ! Level 2 Blas routine.

    ! -- Written on 22-October-1986.
    !    Jack Dongarra, Argonne National Lab.
    !    Jeremy Du Croz, Nag Central Office.
    !    Sven Hammarling, Nag Central Office.
    !    Richard Hanson, Sandia National Labs.

    ! =====================================================================

    ! Test the input parameters.
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    // 检查上三角和下三角的有效性
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
        info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
        // 检查转置选项的有效性
        info = 2;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
        // 检查单位或非单位对角线的有效性
        info = 3;
    } else if (*n < 0) {
        // 检查向量长度是否为负数
        info = 4;
    } else if (*lda < max(1,*n)) {
        // 检查矩阵的leading dimension是否小于1或小于向量长度
        info = 6;
    } else if (*incx == 0) {
        // 检查x的增量是否为零
        info = 8;
    }
    if (info != 0) {
        // 如果有错误信息，调用错误处理函数并返回
        xerbla_("ZTRMV ", &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
        // 如果n为零，直接返回
        return 0;
    }

    noconj = lsame_(trans, "T");
    nounit = lsame_(diag, "N");

/*
       Set up the start point in X if the increment is not unity. This
       will be  ( N - 1 )*INCX  too small for descending loops.
*/

    if (*incx <= 0) {
        // 计算递减步长时x的起始点
        kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
        // 如果步长不是1，则将起始点设置为1
        kx = 1;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (lsame_(trans, "N")) {

/*        Form  x := A*x. */

        if (lsame_(uplo, "U")) {
            // 如果上三角，进行乘法运算
            if (*incx == 1) {
                // x的增量为1的情况
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    i__2 = j;
                    // 如果x[j]不为零
                    if (x[i__2].r != 0. || x[i__2].i != 0.) {
                        i__2 = j;
                        temp.r = x[i__2].r, temp.i = x[i__2].i;
                        i__2 = j - 1;
                        // 执行矩阵向量乘法
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            i__3 = i__;
                            i__4 = i__;
                            i__5 = i__ + j * a_dim1;
                            z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                                z__2.i = temp.r * a[i__5].i + temp.i * a[i__5].r;
                            z__1.r = x[i__4].r + z__2.r, z__1.i = x[i__4].i + z__2.i;
                            x[i__3].r = z__1.r, x[i__3].i = z__1.i;
                        }
                        // 如果不是单位对角线，则乘上对应的A[j,j]元素
                        if (nounit) {
                            i__2 = j;
                            i__3 = j;
                            i__4 = j + j * a_dim1;
                            z__1.r = x[i__3].r * a[i__4].r - x[i__3].i * a[i__4].i,
                                z__1.i = x[i__3].r * a[i__4].i + x[i__3].i * a[i__4].r;
                            x[i__2].r = z__1.r, x[i__2].i = z__1.i;
                        }
                    }
                }
            }
        }
    }
/* L20: */
        }
        } else {
        jx = kx;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = jx;
            // 检查 x[jx] 是否为非零复数
            if (x[i__2].r != 0. || x[i__2].i != 0.) {
            i__2 = jx;
            // 将 x[jx] 的值赋给临时变量 temp
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            ix = kx;
            i__2 = j - 1;
            // 计算下标 ix 处的 x[ix] += temp * a[i + j * a_dim1] 的值，循环 i 从 1 到 j-1
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = ix;
                i__4 = ix;
                i__5 = i__ + j * a_dim1;
                z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    z__2.i = temp.r * a[i__5].i + temp.i * a[
                    i__5].r;
                z__1.r = x[i__4].r + z__2.r, z__1.i = x[i__4].i +
                    z__2.i;
                x[i__3].r = z__1.r, x[i__3].i = z__1.i;
                ix += *incx;
/* L30: */
            }
            // 如果 nounit 为真，则 x[jx] *= a[j + j * a_dim1]
            if (nounit) {
                i__2 = jx;
                i__3 = jx;
                i__4 = j + j * a_dim1;
                z__1.r = x[i__3].r * a[i__4].r - x[i__3].i * a[
                    i__4].i, z__1.i = x[i__3].r * a[i__4].i +
                    x[i__3].i * a[i__4].r;
                x[i__2].r = z__1.r, x[i__2].i = z__1.i;
            }
            }
            // jx 增加步长 incx
            jx += *incx;
/* L40: */
        }
        }
    } else {
        if (*incx == 1) {
        // 当 incx 等于 1 时，从 n 到 1 循环
        for (j = *n; j >= 1; --j) {
            i__1 = j;
            // 检查 x[j] 是否为非零复数
            if (x[i__1].r != 0. || x[i__1].i != 0.) {
            i__1 = j;
            // 将 x[j] 的值赋给临时变量 temp
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            i__1 = j + 1;
            // 从 n 到 j+1 循环，计算 x[i] += temp * a[i + j * a_dim1] 的值
            for (i__ = *n; i__ >= i__1; --i__) {
                i__2 = i__;
                i__3 = i__;
                i__4 = i__ + j * a_dim1;
                z__2.r = temp.r * a[i__4].r - temp.i * a[i__4].i,
                    z__2.i = temp.r * a[i__4].i + temp.i * a[
                    i__4].r;
                z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i +
                    z__2.i;
                x[i__2].r = z__1.r, x[i__2].i = z__1.i;
/* L50: */
            }
            // 如果 nounit 为真，则 x[j] *= a[j + j * a_dim1]
            if (nounit) {
                i__1 = j;
                i__2 = j;
                i__3 = j + j * a_dim1;
                z__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
                    i__3].i, z__1.i = x[i__2].r * a[i__3].i +
                    x[i__2].i * a[i__3].r;
                x[i__1].r = z__1.r, x[i__1].i = z__1.i;
            }
            }
/* L60: */
        }
        } else {
        kx += (*n - 1) * *incx;
        jx = kx;
        for (j = *n; j >= 1; --j) {
            i__1 = jx;
            if (x[i__1].r != 0. || x[i__1].i != 0.) {
            i__1 = jx;
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            ix = kx;
            i__1 = j + 1;
            for (i__ = *n; i__ >= i__1; --i__) {
                i__2 = ix;
                i__3 = ix;
                i__4 = i__ + j * a_dim1;
                z__2.r = temp.r * a[i__4].r - temp.i * a[i__4].i,
                    z__2.i = temp.r * a[i__4].i + temp.i * a[
                    i__4].r;
                z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i +
                    z__2.i;
                x[i__2].r = z__1.r, x[i__2].i = z__1.i;
                ix -= *incx;
/* L70: */
            }
            if (nounit) {
                i__1 = jx;
                i__2 = jx;
                i__3 = j + j * a_dim1;
                z__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
                    i__3].i, z__1.i = x[i__2].r * a[i__3].i +
                    x[i__2].i * a[i__3].r;
                x[i__1].r = z__1.r, x[i__1].i = z__1.i;
            }
            }
            jx -= *incx;
/* L80: */
        }
        }
    }
    } else {

/*        Form  x := A'*x  or  x := conjg( A' )*x. */

    if (lsame_(uplo, "U")) {
        if (*incx == 1) {
        for (j = *n; j >= 1; --j) {
            i__1 = j;
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            if (noconj) {
            if (nounit) {
                i__1 = j + j * a_dim1;
                z__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                    z__1.i = temp.r * a[i__1].i + temp.i * a[
                    i__1].r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
            for (i__ = j - 1; i__ >= 1; --i__) {
                i__1 = i__ + j * a_dim1;
                i__2 = i__;
                z__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
                    i__2].i, z__2.i = a[i__1].r * x[i__2].i +
                    a[i__1].i * x[i__2].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L90: */
            }
            } else {
            if (nounit) {
                d_cnjg(&z__2, &a[j + j * a_dim1]);
                z__1.r = temp.r * z__2.r - temp.i * z__2.i,
                    z__1.i = temp.r * z__2.i + temp.i *
                    z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
            for (i__ = j - 1; i__ >= 1; --i__) {
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                i__1 = i__;
                z__2.r = z__3.r * x[i__1].r - z__3.i * x[i__1].i,
                    z__2.i = z__3.r * x[i__1].i + z__3.i * x[
                    i__1].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;

/* L100: */
            }
            }
            } else {
/* L100: */
            }
            }
            // 计算向量 x 的复数共轭，存储在 temp 中
            i__1 = j;
            x[i__1].r = temp.r, x[i__1].i = temp.i;
/* L110: */
        }
        } else {
        // 计算向量 x 的起始位置
        jx = kx + (*n - 1) * *incx;
        // 逆序迭代处理向量 x
        for (j = *n; j >= 1; --j) {
            // 从向量 x 中取出元素到 temp 中
            i__1 = jx;
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            // 计算向量 x 的起始位置
            ix = jx;
            // 如果不是共轭操作
            if (noconj) {
            // 如果矩阵 A 非单位矩阵
            if (nounit) {
                // 计算矩阵 A 的元素与向量 x 的乘积，并加到 temp 中
                i__1 = j + j * a_dim1;
                z__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i,
                    z__1.i = temp.r * a[i__1].i + temp.i * a[
                    i__1].r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
            // 逆序迭代处理向量 x
            for (i__ = j - 1; i__ >= 1; --i__) {
                // 计算向量 x 中的下一个元素位置
                ix -= *incx;
                // 计算矩阵 A 的元素与向量 x 的乘积，并加到 temp 中
                i__1 = i__ + j * a_dim1;
                i__2 = ix;
                z__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
                    i__2].i, z__2.i = a[i__1].r * x[i__2].i +
                    a[i__1].i * x[i__2].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L120: */
            }
            } else {
            // 如果矩阵 A 非单位矩阵
            if (nounit) {
                // 计算矩阵 A 的复共轭元素与向量 x 的乘积，并加到 temp 中
                d_cnjg(&z__2, &a[j + j * a_dim1]);
                z__1.r = temp.r * z__2.r - temp.i * z__2.i,
                    z__1.i = temp.r * z__2.i + temp.i *
                    z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
            // 逆序迭代处理向量 x
            for (i__ = j - 1; i__ >= 1; --i__) {
                // 计算向量 x 中的下一个元素位置
                ix -= *incx;
                // 计算矩阵 A 的复共轭元素与向量 x 的乘积，并加到 temp 中
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                i__1 = ix;
                z__2.r = z__3.r * x[i__1].r - z__3.i * x[i__1].i,
                    z__2.i = z__3.r * x[i__1].i + z__3.i * x[
                    i__1].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L130: */
            }
            }
            // 将处理后的值存回向量 x
            i__1 = jx;
            x[i__1].r = temp.r, x[i__1].i = temp.i;
            // 更新向量 x 的下一个元素位置
            jx -= *incx;
/* L140: */
        }
        }
    } else {
        // 如果增量 incx 为 1
        if (*incx == 1) {
        // 正序迭代处理向量 x
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 从向量 x 中取出元素到 temp 中
            i__2 = j;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            // 如果不是共轭操作
            if (noconj) {
            // 如果矩阵 A 非单位矩阵
            if (nounit) {
                // 计算矩阵 A 的元素与向量 x 的乘积，并加到 temp 中
                i__2 = j + j * a_dim1;
                z__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                    z__1.i = temp.r * a[i__2].i + temp.i * a[
                    i__2].r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
            // 正序迭代处理向量 x
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                // 计算矩阵 A 的元素与向量 x 的乘积，并加到 temp 中
                i__3 = i__ + j * a_dim1;
                i__4 = i__;
                z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, z__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
            }
            }
            // 将处理后的值存回向量 x
            i__2 = j;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L150: */
        }
        }
/* L150: */
/* 这里是循环结构的结束点，标记上一个 if/else 结构的结束 */
            }
            } else {
/* 如果 nounit 为真，执行下面的语句 */
            if (nounit) {
/* 计算复数的共轭 */
                d_cnjg(&z__2, &a[j + j * a_dim1]);
/* 执行复数乘法，并将结果存入 temp 变量 */
                z__1.r = temp.r * z__2.r - temp.i * z__2.i,
                    z__1.i = temp.r * z__2.i + temp.i *
                    z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
/* 设置循环上限为 n，循环对下标 i 从 j+1 到 n 进行迭代 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
/* 计算复数的共轭 */
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
/* 计算复数乘法，并将结果与 temp 累加 */
                i__3 = i__;
                z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
                    i__3].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L160: */
            }
            }
/* 将累加结果存入 x 数组对应的位置 */
            i__2 = j;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L170: */
        }
/* 结束外层的 if/else 结构 */
        } else {
/* 设置 jx 的初始值为 kx */
        jx = kx;
/* 设置循环上限为 n，循环对下标 j 从 1 到 n 进行迭代 */
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
/* 设置 temp 变量等于 x 数组中第 jx 个位置的值 */
            i__2 = jx;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
/* 设置 ix 的初始值等于 jx */
            ix = jx;
/* 如果 noconj 为真，执行下面的语句 */
            if (noconj) {
/* 如果 nounit 为真，执行下面的语句 */
            if (nounit) {
/* 计算矩阵元素与向量元素的乘积，并将结果存入 temp 变量 */
                i__2 = j + j * a_dim1;
                z__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i,
                    z__1.i = temp.r * a[i__2].i + temp.i * a[
                    i__2].r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
/* 设置循环上限为 n，循环对下标 i 从 j+1 到 n 进行迭代 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
/* ix 自增 incx，找到下一个 x 数组中的元素 */
                ix += *incx;
/* 计算矩阵元素与向量元素的乘积，并将结果与 temp 累加 */
                i__3 = i__ + j * a_dim1;
                i__4 = ix;
                z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, z__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L180: */
            }
            } else {
/* 如果 noconj 为假，执行下面的语句 */
            if (nounit) {
/* 计算复数的共轭 */
                d_cnjg(&z__2, &a[j + j * a_dim1]);
/* 执行复数乘法，并将结果存入 temp 变量 */
                z__1.r = temp.r * z__2.r - temp.i * z__2.i,
                    z__1.i = temp.r * z__2.i + temp.i *
                    z__2.r;
                temp.r = z__1.r, temp.i = z__1.i;
            }
/* 设置循环上限为 n，循环对下标 i 从 j+1 到 n 进行迭代 */
            i__2 = *n;
            for (i__ = j + 1; i__ <= i__2; ++i__) {
/* ix 自增 incx，找到下一个 x 数组中的元素 */
                ix += *incx;
/* 计算复数的共轭 */
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
/* 执行复数乘法，并将结果与 temp 累加 */
                i__3 = ix;
                z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
                    i__3].r;
                z__1.r = temp.r + z__2.r, z__1.i = temp.i +
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L190: */
            }
            }
/* 将累加结果存入 x 数组对应的位置 */
            i__2 = jx;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* jx 自增 incx，指向下一个 x 数组中的元素位置 */
            jx += *incx;
/* L200: */
        }
        }
    }
/* 函数结束，返回值为 0 */
    }

    return 0;

/*     End of ZTRMV . */

} /* ztrmv_ */


注释解释了每行代码的作用，包括循环的结构、条件判断以及复数运算的细节。
/* Subroutine */ int ztrsm_(char *side, char *uplo, char *transa, char *diag,
    integer *m, integer *n, doublecomplex *alpha, doublecomplex *a,
    integer *lda, doublecomplex *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5,
        i__6, i__7;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, j, k, info;
    static doublecomplex temp;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;

    /*
     * Purpose
     * =======
     *
     * ZTRSM  solves one of the matrix equations
     *
     *    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
     *
     * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
     * non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
     *
     *    op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).
     *
     * The matrix X is overwritten on B.
     *
     * Arguments
     * ==========
     *
     * SIDE   - CHARACTER*1.
     *          On entry, SIDE specifies whether op( A ) appears on the left
     *          or right of X as follows:
     *
     *             SIDE = 'L' or 'l'   op( A )*X = alpha*B.
     *
     *             SIDE = 'R' or 'r'   X*op( A ) = alpha*B.
     *
     *          Unchanged on exit.
     *
     * UPLO   - CHARACTER*1.
     *          On entry, UPLO specifies whether the matrix A is an upper or
     *          lower triangular matrix as follows:
     *
     *             UPLO = 'U' or 'u'   A is an upper triangular matrix.
     *
     *             UPLO = 'L' or 'l'   A is a lower triangular matrix.
     *
     *          Unchanged on exit.
     *
     * TRANSA - CHARACTER*1.
     *          On entry, TRANSA specifies the form of op( A ) to be used in
     *          the matrix multiplication as follows:
     *
     *             TRANSA = 'N' or 'n'   op( A ) = A.
     *
     *             TRANSA = 'T' or 't'   op( A ) = A'.
     *
     *             TRANSA = 'C' or 'c'   op( A ) = conjg( A' ).
     *
     *          Unchanged on exit.
     *
     * DIAG   - CHARACTER*1.
     *          On entry, DIAG specifies whether or not A is unit triangular
     *          as follows:
     *
     *             DIAG = 'U' or 'u'   A is assumed to be unit triangular.
     *
     *             DIAG = 'N' or 'n'   A is not assumed to be unit triangular.
     *
     *          Unchanged on exit.
     *
     * M      - INTEGER.
     *          On entry, M specifies the number of rows of B. M must be at
     *          least zero.
     *          Unchanged on exit.
     *
     * N      - INTEGER.
     *          On entry, N specifies the number of columns of B.  N must be
     *          at least zero.
     *          Unchanged on exit.
     *
     * ALPHA  - COMPLEX*16      .
     *          On entry,  ALPHA specifies the scalar  alpha. When  alpha is
     *          zero then  A is not referenced and  B need not be set before
     *          entry.
     *          Unchanged on exit.
     */
    ! COMPLEX*16 类型的数组 A，维度为 (LDA, k)，其中 k 是 m（若 SIDE = 'L' 或 'l'）或是 n（若 SIDE = 'R' 或 'r'）。
    ! 进入时，如果 UPLO = 'U' 或 'u'，A 的前 k 行 k 列必须包含上三角矩阵，且不引用 A 的严格下三角部分。
    ! 进入时，如果 UPLO = 'L' 或 'l'，A 的前 k 行 k 列必须包含下三角矩阵，且不引用 A 的严格上三角部分。
    ! 当 DIAG = 'U' 或 'u' 时，对角线元素也不引用，但假定其为单位矩阵。
    ! 函数返回后，A 的内容不变。
    A      - COMPLEX*16       array of DIMENSION ( LDA, k ),

    ! 在调用（子）程序中声明时，LDA 指定 A 的第一个维度。
    ! 若 SIDE = 'L' 或 'l'，则 LDA 至少为 max( 1, m )。
    ! 若 SIDE = 'R' 或 'r'，则 LDA 至少为 max( 1, n )。
    ! 函数返回后，LDA 的值不变。
    LDA    - INTEGER,

    ! COMPLEX*16 类型的数组 B，维度为 (LDB, n)。
    ! 进入时，数组 B 的前 m 行 n 列必须包含右侧矩阵 B，函数返回时，B 被覆盖为解矩阵 X。
    B      - COMPLEX*16       array of DIMENSION ( LDB, n ),

    ! 在调用（子）程序中声明时，LDB 指定 B 的第一个维度，至少为 max( 1, m )。
    ! 函数返回后，LDB 的值不变。
    LDB    - INTEGER,
    /* Parameter adjustments */
    // 设置矩阵 A 和 B 的维度和偏移量
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    // 判断是左操作还是右操作
    lside = lsame_(side, "L");
    if (lside) {
        // 如果是左操作，则 nrowa 是 m 的值
        nrowa = *m;
    } else {
        // 如果是右操作，则 nrowa 是 n 的值
        nrowa = *n;
    }
    // 判断是否不共轭
    noconj = lsame_(transa, "T");
    // 判断是否非单位矩阵
    nounit = lsame_(diag, "N");
    // 判断是否为上三角矩阵
    upper = lsame_(uplo, "U");

    // 错误信息初始化为 0
    info = 0;
    // 检查参数是否正确
    if (! lside && ! lsame_(side, "R")) {
        info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
        info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa, "T") && ! lsame_(transa, "C")) {
        info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
        info = 4;
    } else if (*m < 0) {
        info = 5;
    } else if (*n < 0) {
        info = 6;
    } else if (*lda < max(1,nrowa)) {
        info = 9;
    } else if (*ldb < max(1,*m)) {
        info = 11;
    }
    // 如果有错误信息，调用错误处理函数并返回
    if (info != 0) {
        xerbla_("ZTRSM ", &info);
        return 0;
    }

/*     Quick return if possible. */

    // 如果 m 或者 n 为 0，直接返回
    if (*m == 0 || *n == 0) {
        return 0;
    }

/*     And when  alpha.eq.zero. */

    // 如果 alpha 等于零，将 B 矩阵置零
    if (alpha->r == 0. && alpha->i == 0.) {
        // 循环遍历 B 矩阵中的元素，并将其置零
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + j * b_dim1;
                b[i__3].r = 0., b[i__3].i = 0.;
            }
        }
        return 0;
    }

/*     Start the operations. */

    // 开始进行运算
    if (lside) {
        if (lsame_(transa, "N")) {

/*           Form  B := alpha*inv( A )*B. */

            // 如果 A 是非单位上三角矩阵，按行处理 B 矩阵
            if (upper) {
                i__1 = *n;
                for (j = 1; j <= i__1; ++j) {
                    // 如果 alpha 不等于 1，则对每个元素进行乘法运算
                    if (alpha->r != 1. || alpha->i != 0.) {
                        i__2 = *m;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            i__3 = i__ + j * b_dim1;
                            i__4 = i__ + j * b_dim1;
                            z__1.r = alpha->r * b[i__4].r - alpha->i * b[i__4].i, z__1.i = alpha->r * b[i__4].i + alpha->i * b[i__4].r;
                            b[i__3].r = z__1.r, b[i__3].i = z__1.i;
                        }
                    }
                    // 对每列元素进行处理
                    for (k = *m; k >= 1; --k) {
                        i__2 = k + j * b_dim1;
                        // 如果 B 矩阵中元素不为零，进行除法运算
                        if (b[i__2].r != 0. || b[i__2].i != 0.) {
                            if (nounit) {
                                i__2 = k + j * b_dim1;
                                z_div(&z__1, &b[k + j * b_dim1], &a[k + k * a_dim1]);
                                b[i__2].r = z__1.r, b[i__2].i = z__1.i;
                            }
                            i__2 = k - 1;
                            for (i__ = 1; i__ <= i__2; ++i__) {
                                i__3 = i__ + j * b_dim1;
                                i__4 = i__ + j * b_dim1;
                                i__5 = k + j * b_dim1;
                                i__6 = i__ + k * a_dim1;
                                z__2.r = b[i__5].r * a[i__6].r - b[i__5].i * a[i__6].i, z__2.i = b[i__5].r * a[i__6].i + b[i__5].i * a[i__6].r;
                                z__1.r = b[i__4].r - z__2.r, z__1.i = b[i__4].i - z__2.i;
                                b[i__3].r = z__1.r, b[i__3].i = z__1.i;
                            }
                        }
                    }
                }
            }
        }
    }
/* L40: 结束内层循环 */
                }
            }
/* L50: 结束外层循环 */
            }
/* L60: 处理特定条件下的情况 */
        }
        } else {
        i__1 = *n;  // 获取参数 n 的值
        for (j = 1; j <= i__1; ++j) {  // 循环遍历 j 从 1 到 n
            if (alpha->r != 1. || alpha->i != 0.) {  // 检查 alpha 是否不等于 (1, 0)
            i__2 = *m;  // 获取参数 m 的值
            for (i__ = 1; i__ <= i__2; ++i__) {  // 循环遍历 i 从 1 到 m
                i__3 = i__ + j * b_dim1;  // 计算数组 b 中的索引
                i__4 = i__ + j * b_dim1;  // 计算数组 b 中的索引
                z__1.r = alpha->r * b[i__4].r - alpha->i * b[i__4]
                    .i, z__1.i = alpha->r * b[i__4].i +
                    alpha->i * b[i__4].r;  // 计算复数乘法结果
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;  // 更新数组 b 的值
/* L70: 执行内层循环体 */
            }
            }
            i__2 = *m;  // 获取参数 m 的值
            for (k = 1; k <= i__2; ++k) {  // 循环遍历 k 从 1 到 m
            i__3 = k + j * b_dim1;  // 计算数组 b 中的索引
            if (b[i__3].r != 0. || b[i__3].i != 0.) {  // 检查数组 b 的值是否不等于 (0, 0)
                if (nounit) {  // 检查是否不是单位矩阵
                i__3 = k + j * b_dim1;  // 计算数组 b 中的索引
                z_div(&z__1, &b[k + j * b_dim1], &a[k + k *
                    a_dim1]);  // 执行复数除法
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;  // 更新数组 b 的值
                }
                i__3 = *m;  // 获取参数 m 的值
                for (i__ = k + 1; i__ <= i__3; ++i__) {  // 循环遍历 i 从 k+1 到 m
                i__4 = i__ + j * b_dim1;  // 计算数组 b 中的索引
                i__5 = i__ + j * b_dim1;  // 计算数组 b 中的索引
                i__6 = k + j * b_dim1;  // 计算数组 b 中的索引
                i__7 = i__ + k * a_dim1;  // 计算数组 a 中的索引
                z__2.r = b[i__6].r * a[i__7].r - b[i__6].i *
                    a[i__7].i, z__2.i = b[i__6].r * a[
                    i__7].i + b[i__6].i * a[i__7].r;  // 计算复数乘法结果
                z__1.r = b[i__5].r - z__2.r, z__1.i = b[i__5]
                    .i - z__2.i;  // 计算复数减法结果
                b[i__4].r = z__1.r, b[i__4].i = z__1.i;  // 更新数组 b 的值
/* L80: 执行内层循环体 */
                }
            }
/* L90: 执行外层循环体 */
            }
/* L100: 执行最外层循环体 */
        }
        }
    } else {

/*
             Form  B := alpha*inv( A' )*B
             or    B := alpha*inv( conjg( A' ) )*B.
*/

        if (upper) {  // 检查是否为上三角矩阵
        i__1 = *n;  // 获取参数 n 的值
        for (j = 1; j <= i__1; ++j) {  // 循环遍历 j 从 1 到 n
            i__2 = *m;  // 获取参数 m 的值
            for (i__ = 1; i__ <= i__2; ++i__) {  // 循环遍历 i 从 1 到 m
            i__3 = i__ + j * b_dim1;  // 计算数组 b 中的索引
            z__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i,
                z__1.i = alpha->r * b[i__3].i + alpha->i * b[
                i__3].r;  // 计算复数乘法结果
            temp.r = z__1.r, temp.i = z__1.i;  // 更新临时变量的值
            if (noconj) {  // 检查是否不使用共轭转置
                i__3 = i__ - 1;  // 计算循环上限
                for (k = 1; k <= i__3; ++k) {  // 循环遍历 k 从 1 到 i-1
                i__4 = k + i__ * a_dim1;  // 计算数组 a 中的索引
                i__5 = k + j * b_dim1;  // 计算数组 b 中的索引
                z__2.r = a[i__4].r * b[i__5].r - a[i__4].i *
                    b[i__5].i, z__2.i = a[i__4].r * b[
                    i__5].i + a[i__4].i * b[i__5].r;  // 计算复数乘法结果
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;  // 计算复数减法结果
                temp.r = z__1.r, temp.i = z__1.i;  // 更新临时变量的值

                temp.r = z__1.r, temp.i = z__1.i;  // 更新临时变量的值
                }
            }
/* L70: 执行内层循环体 */
            }
/* L80: 执行外层循环体 */
        }
/* L90: 结束外层循环 */
        }
/* L100: 结束最外层循环 */
    }
/* L110: */
                }
                if (nounit) {
                z_div(&z__1, &temp, &a[i__ + i__ * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
                }
            } else {
                i__3 = i__ - 1;
                for (k = 1; k <= i__3; ++k) {
                d_cnjg(&z__3, &a[k + i__ * a_dim1]);
                i__4 = k + j * b_dim1;
                z__2.r = z__3.r * b[i__4].r - z__3.i * b[i__4]
                    .i, z__2.i = z__3.r * b[i__4].i +
                    z__3.i * b[i__4].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L120: */
                }
                if (nounit) {
                d_cnjg(&z__2, &a[i__ + i__ * a_dim1]);
                z_div(&z__1, &temp, &z__2);
                temp.r = z__1.r, temp.i = z__1.i;
                }
            }
            i__3 = i__ + j * b_dim1;
            b[i__3].r = temp.r, b[i__3].i = temp.i;
/* L130: */
            }
/* L140: */
        }
        } else {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            for (i__ = *m; i__ >= 1; --i__) {
            i__2 = i__ + j * b_dim1;
            z__1.r = alpha->r * b[i__2].r - alpha->i * b[i__2].i,
                z__1.i = alpha->r * b[i__2].i + alpha->i * b[
                i__2].r;
            temp.r = z__1.r, temp.i = z__1.i;
            if (noconj) {
                i__2 = *m;
                for (k = i__ + 1; k <= i__2; ++k) {
                i__3 = k + i__ * a_dim1;
                i__4 = k + j * b_dim1;
                z__2.r = a[i__3].r * b[i__4].r - a[i__3].i *
                    b[i__4].i, z__2.i = a[i__3].r * b[
                    i__4].i + a[i__3].i * b[i__4].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L150: */
                }
                if (nounit) {
                z_div(&z__1, &temp, &a[i__ + i__ * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
                }
            } else {
                i__2 = *m;
                for (k = i__ + 1; k <= i__2; ++k) {
                d_cnjg(&z__3, &a[k + i__ * a_dim1]);
                i__3 = k + j * b_dim1;
                z__2.r = z__3.r * b[i__3].r - z__3.i * b[i__3]
                    .i, z__2.i = z__3.r * b[i__3].i +
                    z__3.i * b[i__3].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L160: */
                }
                if (nounit) {
                d_cnjg(&z__2, &a[i__ + i__ * a_dim1]);
                z_div(&z__1, &temp, &z__2);
                temp.r = z__1.r, temp.i = z__1.i;
                }
            }
            i__2 = i__ + j * b_dim1;
            b[i__2].r = temp.r, b[i__2].i = temp.i;
/* L170: */
            }
/* L180: */
        }
        }
    }
    } else {


注释：
    # 检查条件：如果变量 transa 的值是 "N"（不区分大小写），则执行以下语句块
/*           Form  B := alpha*B*inv( A ). */

/* 如果上三角矩阵 */
if (upper) {
    /* 对于每列 j = 1 到 n */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        /* 如果 alpha 不等于 (1,0) */
        if (alpha->r != 1. || alpha->i != 0.) {
            /* 对于每行 i = 1 到 m */
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                /* 计算 B[i+j*b_dim1] = alpha * B[i+j*b_dim1] */
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                z__1.r = alpha->r * b[i__4].r - alpha->i * b[i__4].i, z__1.i = alpha->r * b[i__4].i + alpha->i * b[i__4].r;
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;
                /* L190: */
            }
        }
        /* 对于每个小于 j 的列 k */
        i__2 = j - 1;
        for (k = 1; k <= i__2; ++k) {
            /* 如果 A[k+j*a_dim1] 不等于零 */
            i__3 = k + j * a_dim1;
            if (a[i__3].r != 0. || a[i__3].i != 0.) {
                /* 对于每行 i = 1 到 m */
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                    /* 计算 B[i+j*b_dim1] = B[i+j*b_dim1] - A[k+j*a_dim1] * B[i+k*b_dim1] */
                    i__4 = i__ + j * b_dim1;
                    i__5 = i__ + j * b_dim1;
                    i__6 = k + j * a_dim1;
                    i__7 = i__ + k * b_dim1;
                    z__2.r = a[i__6].r * b[i__7].r - a[i__6].i * b[i__7].i, z__2.i = a[i__6].r * b[i__7].i + a[i__6].i * b[i__7].r;
                    z__1.r = b[i__5].r - z__2.r, z__1.i = b[i__5].i - z__2.i;
                    b[i__4].r = z__1.r, b[i__4].i = z__1.i;
                    /* L200: */
                }
            }
            /* L210: */
        }
        /* 如果非单位矩阵 */
        if (nounit) {
            /* 计算 temp = 1 / A[j+j*a_dim1] */
            z_div(&z__1, &c_b1078, &a[j + j * a_dim1]);
            temp.r = z__1.r, temp.i = z__1.i;
            /* 对于每行 i = 1 到 m */
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                /* 计算 B[i+j*b_dim1] = temp * B[i+j*b_dim1] */
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                z__1.r = temp.r * b[i__4].r - temp.i * b[i__4].i, z__1.i = temp.r * b[i__4].i + temp.i * b[i__4].r;
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;
                /* L220: */
            }
        }
        /* L230: */
    }
} else {
    /* 对于每列 j = n 到 1 */
    for (j = *n; j >= 1; --j) {
        /* 如果 alpha 不等于 (1,0) */
        if (alpha->r != 1. || alpha->i != 0.) {
            /* 对于每行 i = 1 到 m */
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                /* 计算 B[i+j*b_dim1] = alpha * B[i+j*b_dim1] */
                i__2 = i__ + j * b_dim1;
                i__3 = i__ + j * b_dim1;
                z__1.r = alpha->r * b[i__3].r - alpha->i * b[i__3].i, z__1.i = alpha->r * b[i__3].i + alpha->i * b[i__3].r;
                b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/*
   L240: 循环结束标记
*/
            }
            }
            // 循环遍历 k = j + 1 到 n
            i__1 = *n;
            for (k = j + 1; k <= i__1; ++k) {
            // 如果 a(k, j) 不为零
            i__2 = k + j * a_dim1;
            if (a[i__2].r != 0. || a[i__2].i != 0.) {
                // 对每个 i 循环执行累加操作
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵乘法并更新 b(i, j)
                i__3 = i__ + j * b_dim1;
                i__4 = i__ + j * b_dim1;
                i__5 = k + j * a_dim1;
                i__6 = i__ + k * b_dim1;
                z__2.r = a[i__5].r * b[i__6].r - a[i__5].i *
                    b[i__6].i, z__2.i = a[i__5].r * b[
                    i__6].i + a[i__5].i * b[i__6].r;
                z__1.r = b[i__4].r - z__2.r, z__1.i = b[i__4]
                    .i - z__2.i;
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;
                // L250 标记
/* L250: */
                }
            }
/* L260: */
            }
            // 如果不是单位矩阵
            if (nounit) {
            // 计算除法并存储结果到 temp
            z_div(&z__1, &c_b1078, &a[j + j * a_dim1]);
            temp.r = z__1.r, temp.i = z__1.i;
            // 对每个 i 循环执行乘法操作
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                // 计算矩阵乘法并更新 b(i, j)
                i__2 = i__ + j * b_dim1;
                i__3 = i__ + j * b_dim1;
                z__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    z__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L270: */
            }
            }
/* L280: */
        }
        }
    } else {

/*
             Form  B := alpha*B*inv( A' )
             or    B := alpha*B*inv( conjg( A' ) ).
*/

        // 如果是上三角矩阵
        if (upper) {
        // 从后向前循环 k = n 到 1
        for (k = *n; k >= 1; --k) {
            // 如果不是单位矩阵
            if (nounit) {
            // 如果不进行共轭
            if (noconj) {
                // 计算除法并存储结果到 temp
                z_div(&z__1, &c_b1078, &a[k + k * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
            } else {
                // 进行共轭并计算除法，存储结果到 temp
                d_cnjg(&z__2, &a[k + k * a_dim1]);
                z_div(&z__1, &c_b1078, &z__2);
                temp.r = z__1.r, temp.i = z__1.i;
            }
            // 对每个 i 循环执行乘法操作
            i__1 = *m;
            for (i__ = 1; i__ <= i__1; ++i__) {
                // 计算矩阵乘法并更新 b(i, k)
                i__2 = i__ + k * b_dim1;
                i__3 = i__ + k * b_dim1;
                z__1.r = temp.r * b[i__3].r - temp.i * b[i__3].i,
                    z__1.i = temp.r * b[i__3].i + temp.i * b[
                    i__3].r;
                b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L290: 结束第一个内层循环 */

/* L300: 对矩阵 b 进行更新，根据矩阵 a 和向量 temp 的计算结果 */

/* L310: 结束第二个内层循环 */

/* L320: 对矩阵 b 进行更新，根据 alpha 和矩阵 b 的计算结果 */

/* L330: 结束外层循环 */

/* L340: 对称情况下，根据 a 和 c_b1078 更新 temp，并对矩阵 b 进行更新 */

/* L350: 对矩阵 b 进行更新，根据 temp 和矩阵 b 的计算结果 */

/* L360: 结束内层循环 */
/* L340: */
            }
            }
            i__2 = *n;
            for (j = k + 1; j <= i__2; ++j) {
            i__3 = j + k * a_dim1;
            if (a[i__3].r != 0. || a[i__3].i != 0.) {
                if (noconj) {
                i__3 = j + k * a_dim1;
                temp.r = a[i__3].r, temp.i = a[i__3].i;
                } else {
                d_cnjg(&z__1, &a[j + k * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
                }
                i__3 = *m;
                for (i__ = 1; i__ <= i__3; ++i__) {
                i__4 = i__ + j * b_dim1;
                i__5 = i__ + j * b_dim1;
                i__6 = i__ + k * b_dim1;
                z__2.r = temp.r * b[i__6].r - temp.i * b[i__6]
                    .i, z__2.i = temp.r * b[i__6].i +
                    temp.i * b[i__6].r;
                z__1.r = b[i__5].r - z__2.r, z__1.i = b[i__5]
                    .i - z__2.i;
                b[i__4].r = z__1.r, b[i__4].i = z__1.i;
/* L350: */
                }
            }
/* L360: */
            }
            if (alpha->r != 1. || alpha->i != 0.) {
            i__2 = *m;
            for (i__ = 1; i__ <= i__2; ++i__) {
                i__3 = i__ + k * b_dim1;
                i__4 = i__ + k * b_dim1;
                z__1.r = alpha->r * b[i__4].r - alpha->i * b[i__4]
                    .i, z__1.i = alpha->r * b[i__4].i +
                    alpha->i * b[i__4].r;
                b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L370: */
            }
            }
/* L380: */
        }
        }
    }
    }

    return 0;

/*     End of ZTRSM . */

} /* ztrsm_ */

/* Subroutine */ int ztrsv_(char *uplo, char *trans, char *diag, integer *n,
    doublecomplex *a, integer *lda, doublecomplex *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, j, ix, jx, kx, info;
    static doublecomplex temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical noconj, nounit;


/*
    Purpose
    =======

    ZTRSV  solves one of the systems of equations

       A*x = b,   or   A'*x = b,   or   conjg( A' )*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular matrix.

    No test for singularity or near-singularity is included in this
    routine. Such tests must be performed before calling this routine.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.


注释：
    ! TRANS  - CHARACTER*1.
    !         On entry, TRANS specifies the equations to be solved as
    !         follows:
    !
    !            TRANS = 'N' or 'n'   A*x = b.
    !
    !            TRANS = 'T' or 't'   A'*x = b.
    !
    !            TRANS = 'C' or 'c'   conjg( A' )*x = b.
    !
    !         Unchanged on exit.

    ! DIAG   - CHARACTER*1.
    !         On entry, DIAG specifies whether or not A is unit
    !         triangular as follows:
    !
    !            DIAG = 'U' or 'u'   A is assumed to be unit triangular.
    !
    !            DIAG = 'N' or 'n'   A is not assumed to be unit
    !                                triangular.
    !
    !         Unchanged on exit.

    ! N      - INTEGER.
    !         On entry, N specifies the order of the matrix A.
    !         N must be at least zero.
    !         Unchanged on exit.

    ! A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
    !         Before entry with  UPLO = 'U' or 'u', the leading n by n
    !         upper triangular part of the array A must contain the upper
    !         triangular matrix and the strictly lower triangular part of
    !         A is not referenced.
    !         Before entry with UPLO = 'L' or 'l', the leading n by n
    !         lower triangular part of the array A must contain the lower
    !         triangular matrix and the strictly upper triangular part of
    !         A is not referenced.
    !         Note that when  DIAG = 'U' or 'u', the diagonal elements of
    !         A are not referenced either, but are assumed to be unity.
    !         Unchanged on exit.

    ! LDA    - INTEGER.
    !         On entry, LDA specifies the first dimension of A as declared
    !         in the calling (sub) program. LDA must be at least
    !         max( 1, n ).
    !         Unchanged on exit.

    ! X      - COMPLEX*16       array of dimension at least
    !         ( 1 + ( n - 1 )*abs( INCX ) ).
    !         Before entry, the incremented array X must contain the n
    !         element right-hand side vector b. On exit, X is overwritten
    !         with the solution vector x.

    ! INCX   - INTEGER.
    !         On entry, INCX specifies the increment for the elements of
    !         X. INCX must not be zero.
    !         Unchanged on exit.

    ! Further Details
    ! ===============
    !
    ! Level 2 Blas routine.
    !
    ! -- Written on 22-October-1986.
    !    Jack Dongarra, Argonne National Lab.
    !    Jeremy Du Croz, Nag Central Office.
    !    Sven Hammarling, Nag Central Office.
    !    Richard Hanson, Sandia National Labs.

    ! =====================================================================

    ! Test the input parameters.
    /* 参数调整 */
    a_dim1 = *lda;  // a_dim1 是 lda 的副本，用于在多维数组 a 中正确访问元素
    a_offset = 1 + a_dim1;  // 计算 a 的偏移量，确保正确访问 a 的元素
    a -= a_offset;  // 调整 a 的指针，使其指向正确的起始位置
    --x;  // 将 x 指针向前移动一位，从1-based调整为0-based

    /* 函数体 */
    info = 0;  // 初始化 info 变量，用于记录错误信息
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {  // 检查 uplo 是否为 'U' 或 'L'
    info = 1;  // 如果不是，设置错误码为 1
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
    info = 2;  // 如果 trans 不是 'N', 'T' 或 'C'，设置错误码为 2
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
    info = 3;  // 如果 diag 不是 'U' 或 'N'，设置错误码为 3
    } else if (*n < 0) {
    info = 4;  // 如果 n 小于零，设置错误码为 4
    } else if (*lda < max(1,*n)) {
    info = 6;  // 如果 lda 小于 max(1, n)，设置错误码为 6
    } else if (*incx == 0) {
    info = 8;  // 如果 incx 等于零，设置错误码为 8
    }
    if (info != 0) {
    xerbla_("ZTRSV ", &info);  // 调用错误处理函数，打印错误信息并终止程序
    return 0;  // 返回 0，终止函数执行
    }

/*     如果可能，进行快速返回。 */

    if (*n == 0) {
    return 0;  // 如果 n 等于零，直接返回 0
    }

    noconj = lsame_(trans, "T");  // 检查 trans 是否为 'T'，确定是否进行共轭
    nounit = lsame_(diag, "N");  // 检查 diag 是否为 'N'，确定是否单位矩阵

/*
       如果增量不是单位增量，则设置 X 中的起始点。对于递减循环，这将是
       ( N - 1 )*INCX 以便足够小。
*/

    if (*incx <= 0) {
    kx = 1 - (*n - 1) * *incx;  // 计算 kx，确保足够小以适应递减循环
    } else if (*incx != 1) {
    kx = 1;  // 如果 incx 不等于 1，则设置 kx 为 1
    }

/*
       开始操作。在这个版本中，通过一次对 A 的顺序访问元素。
*/

    if (lsame_(trans, "N")) {

/*        形成 x := inv( A )*x. */

    if (lsame_(uplo, "U")) {  // 如果 uplo 是 'U'
        if (*incx == 1) {  // 如果 incx 等于 1
        for (j = *n; j >= 1; --j) {  // 递减循环遍历 j
            i__1 = j;
            if (x[i__1].r != 0. || x[i__1].i != 0.) {  // 检查 x[j] 是否为非零
            if (nounit) {  // 如果不是单位矩阵
                i__1 = j;
                z_div(&z__1, &x[j], &a[j + j * a_dim1]);  // 计算 x[j] / a[j,j]
                x[i__1].r = z__1.r, x[i__1].i = z__1.i;  // 将结果存入 x[j]
            }
            i__1 = j;
            temp.r = x[i__1].r, temp.i = x[i__1].i;  // 将 x[j] 的值复制到临时变量 temp
            for (i__ = j - 1; i__ >= 1; --i__) {  // 递减循环遍历 i
                i__1 = i__;
                i__2 = i__;
                i__3 = i__ + j * a_dim1;
                z__2.r = temp.r * a[i__3].r - temp.i * a[i__3].i,  // 计算 temp * a[i,j]
                    z__2.i = temp.r * a[i__3].i + temp.i * a[i__3].r;
                z__1.r = x[i__2].r - z__2.r, z__1.i = x[i__2].i - z__2.i;  // 更新 x[i]
                x[i__1].r = z__1.r, x[i__1].i = z__1.i;
/* L10: */  // 循环标签，用于 GOTO 语句的跳转目标
            }
            }

                }
            }
        } else {
            // 当 incx 不等于 1 时的处理逻辑
            for (j = *n; j >= 1; --j) {
                i__1 = j;
                if (x[i__1].r != 0. || x[i__1].i != 0.) {
                    if (nounit) {
                        i__1 = j;
                        z_div(&z__1, &x[j], &a[j + j * a_dim1]);
                        x[i__1].r = z__1.r, x[i__1].i = z__1.i;
                    }
                    i__1 = j;
                    temp.r = x[i__1].r, temp.i = x[i__1].i;
                    for (i__ = j - 1; i__ >= 1; --i__) {
                        i__1 = i__;
                        i__2 = i__;
                        i__3 = i__ + j * a_dim1;
                        z__2.r = temp.r * a[i__3].r - temp.i * a[i__3].i,
                            z__2.i = temp.r * a[i__3].i + temp.i * a[i__3].r;
                        z__1.r = x[i__2].r - z__2.r, z__1.i = x[i__2].i - z__2.i;
                        x[i__1].r = z__1.r, x[i__1].i = z__1.i;
                    }
                }
            }
        }
    }
}


注释完成。
/* L20: */
        }
        } else {
        jx = kx + (*n - 1) * *incx;
        for (j = *n; j >= 1; --j) {
            i__1 = jx;
            if (x[i__1].r != 0. || x[i__1].i != 0.) {
            if (nounit) {
                i__1 = jx;
                // 如果不是单位矩阵，计算 x[jx] 除以 a[j + j * a_dim1] 的复数除法结果
                z_div(&z__1, &x[jx], &a[j + j * a_dim1]);
                x[i__1].r = z__1.r, x[i__1].i = z__1.i;
            }
            // 将 x[jx] 的值赋给 temp
            i__1 = jx;
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            ix = jx;
            // 对当前列的下三角部分进行更新
            for (i__ = j - 1; i__ >= 1; --i__) {
                ix -= *incx;
                // 计算更新公式并应用于 x[ix]
                i__1 = ix;
                i__2 = ix;
                i__3 = i__ + j * a_dim1;
                z__2.r = temp.r * a[i__3].r - temp.i * a[i__3].i,
                    z__2.i = temp.r * a[i__3].i + temp.i * a[
                    i__3].r;
                z__1.r = x[i__2].r - z__2.r, z__1.i = x[i__2].i -
                    z__2.i;
                x[i__1].r = z__1.r, x[i__1].i = z__1.i;
/* L30: */
            }
            }
            // 更新 jx 到下一个要处理的元素位置
            jx -= *incx;
/* L40: */
        }
        }
    } else {
        if (*incx == 1) {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = j;
            if (x[i__2].r != 0. || x[i__2].i != 0.) {
            if (nounit) {
                i__2 = j;
                // 如果不是单位矩阵，计算 x[j] 除以 a[j + j * a_dim1] 的复数除法结果
                z_div(&z__1, &x[j], &a[j + j * a_dim1]);
                x[i__2].r = z__1.r, x[i__2].i = z__1.i;
            }
            // 将 x[j] 的值赋给 temp
            i__2 = j;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            i__2 = *n;
            // 对当前列的上三角部分进行更新
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                i__3 = i__;
                i__4 = i__;
                i__5 = i__ + j * a_dim1;
                z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    z__2.i = temp.r * a[i__5].i + temp.i * a[
                    i__5].r;
                z__1.r = x[i__4].r - z__2.r, z__1.i = x[i__4].i -
                    z__2.i;
                x[i__3].r = z__1.r, x[i__3].i = z__1.i;
/* L50: */
            }
            }
/* L60: */
        }
        } else {
        jx = kx;
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            i__2 = jx;
            if (x[i__2].r != 0. || x[i__2].i != 0.) {
            if (nounit) {
                i__2 = jx;
                // 如果不是单位矩阵，计算 x[jx] 除以 a[j + j * a_dim1] 的复数除法结果
                z_div(&z__1, &x[jx], &a[j + j * a_dim1]);
                x[i__2].r = z__1.r, x[i__2].i = z__1.i;
            }
            // 将 x[jx] 的值赋给 temp
            i__2 = jx;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            ix = jx;
            i__2 = *n;
            // 对当前列的上三角部分进行更新
            for (i__ = j + 1; i__ <= i__2; ++i__) {
                ix += *incx;
                i__3 = ix;
                i__4 = ix;
                i__5 = i__ + j * a_dim1;
                z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i,
                    z__2.i = temp.r * a[i__5].i + temp.i * a[
                    i__5].r;
                z__1.r = x[i__4].r - z__2.r, z__1.i = x[i__4].i -
                    z__2.i;
                x[i__3].r = z__1.r, x[i__3].i = z__1.i;
/* L70: */
            }
            }
            jx += *incx;
/* L80: */
        }
        }
    }
    } else {

/*        Form  x := inv( A' )*x  or  x := inv( conjg( A' ) )*x. */

    // 如果 uplo 不是 'U'，执行下面的代码块
    if (lsame_(uplo, "U")) {
        // 如果 incx 等于 1，执行下面的代码块
        if (*incx == 1) {
        // 对于每个列 j，执行以下操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 复制 x[j] 到 temp 变量
            i__2 = j;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            // 如果不需要共轭，执行以下操作
            if (noconj) {
            // 对于每个行 i，执行以下操作
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵 A 的元素 a[i+j*a_dim1] 与向量 x[i] 的乘积，并更新 temp
                i__3 = i__ + j * a_dim1;
                i__4 = i__;
                z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, z__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L90: */
            }
            // 如果矩阵 A 非单位对角线，将 temp 除以 a[j+j*a_dim1]
            if (nounit) {
                z_div(&z__1, &temp, &a[j + j * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
            }
            } else {
            // 对于每个行 i，执行以下操作
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵 A 共轭转置的元素与向量 x[i] 的乘积，并更新 temp
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                i__3 = i__;
                z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
                    i__3].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
/* L100: */
            }
            // 如果矩阵 A 共轭转置非单位对角线，将 temp 除以 conjg(a[j+j*a_dim1])
            if (nounit) {
                d_cnjg(&z__2, &a[j + j * a_dim1]);
                z_div(&z__1, &temp, &z__2);
                temp.r = z__1.r, temp.i = z__1.i;
            }
            }
            // 将 temp 的值赋给 x[j]
            i__2 = j;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L110: */
        }
        } else {
        // 如果 incx 不等于 1，执行以下代码块
        jx = kx;
        // 对于每个列 j，执行以下操作
        i__1 = *n;
        for (j = 1; j <= i__1; ++j) {
            // 计算 x 的索引位置 ix
            ix = kx;
            // 复制 x[jx] 到 temp 变量
            i__2 = jx;
            temp.r = x[i__2].r, temp.i = x[i__2].i;
            // 如果不需要共轭，执行以下操作
            if (noconj) {
            // 对于每个行 i，执行以下操作
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵 A 的元素 a[i+j*a_dim1] 与向量 x[ix] 的乘积，并更新 temp
                i__3 = i__ + j * a_dim1;
                i__4 = ix;
                z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
                    i__4].i, z__2.i = a[i__3].r * x[i__4].i +
                    a[i__3].i * x[i__4].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
                ix += *incx;
/* L100: */
            }
            // 如果矩阵 A 非单位对角线，将 temp 除以 a[j+j*a_dim1]
            if (nounit) {
                z_div(&z__1, &temp, &a[j + j * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
            }
            } else {
            // 对于每个行 i，执行以下操作
            i__2 = j - 1;
            for (i__ = 1; i__ <= i__2; ++i__) {
                // 计算矩阵 A 共轭转置的元素与向量 x[ix] 的乘积，并更新 temp
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                i__3 = ix;
                z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
                    i__3].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
                ix += *incx;
/* L100: */
            }
            // 如果矩阵 A 共轭转置非单位对角线，将 temp 除以 conjg(a[j+j*a_dim1])
            if (nounit) {
                d_cnjg(&z__2, &a[j + j * a_dim1]);
                z_div(&z__1, &temp, &z__2);
                temp.r = z__1.r, temp.i = z__1.i;
            }
            }
            // 将 temp 的值赋给 x[jx]
            i__2 = jx;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L110: */
        }
        }
    }
    }
/* L120: */
            }
            if (nounit) {
                z_div(&z__1, &temp, &a[j + j * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
            }
/* L130: */
            } else {
                i__2 = j - 1;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                    i__3 = ix;
                    z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i,
                        z__2.i = z__3.r * x[i__3].i + z__3.i * x[
                        i__3].r;
                    z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                        z__2.i;
                    temp.r = z__1.r, temp.i = z__1.i;
                    ix += *incx;
/* L130: */
                }
                if (nounit) {
                    d_cnjg(&z__2, &a[j + j * a_dim1]);
                    z_div(&z__1, &temp, &z__2);
                    temp.r = z__1.r, temp.i = z__1.i;
                }
            }
            i__2 = jx;
            x[i__2].r = temp.r, x[i__2].i = temp.i;
            jx += *incx;
/* L140: */
        }
        }
    } else {
        if (*incx == 1) {
            for (j = *n; j >= 1; --j) {
                i__1 = j;
                temp.r = x[i__1].r, temp.i = x[i__1].i;
                if (noconj) {
                    i__1 = j + 1;
                    for (i__ = *n; i__ >= i__1; --i__) {
                        i__2 = i__ + j * a_dim1;
                        i__3 = i__;
                        z__2.r = a[i__2].r * x[i__3].r - a[i__2].i * x[
                            i__3].i, z__2.i = a[i__2].r * x[i__3].i +
                            a[i__2].i * x[i__3].r;
                        z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                            z__2.i;
                        temp.r = z__1.r, temp.i = z__1.i;
/* L150: */
                    }
                    if (nounit) {
                        z_div(&z__1, &temp, &a[j + j * a_dim1]);
                        temp.r = z__1.r, temp.i = z__1.i;
                    }
                } else {
                    i__1 = j + 1;
                    for (i__ = *n; i__ >= i__1; --i__) {
                        d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                        i__2 = i__;
                        z__2.r = z__3.r * x[i__2].r - z__3.i * x[i__2].i,
                            z__2.i = z__3.r * x[i__2].i + z__3.i * x[
                            i__2].r;
                        z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                            z__2.i;
                        temp.r = z__1.r, temp.i = z__1.i;
/* L160: */
                    }
                    if (nounit) {
                        d_cnjg(&z__2, &a[j + j * a_dim1]);
                        z_div(&z__1, &temp, &z__2);
                        temp.r = z__1.r, temp.i = z__1.i;
                    }
                }
                i__1 = j;
                x[i__1].r = temp.r, x[i__1].i = temp.i;
/* L140: */
            }
        }
    }
/* L170: */
        }
        } else {
        kx += (*n - 1) * *incx;
        jx = kx;
        for (j = *n; j >= 1; --j) {
            ix = kx;
            i__1 = jx;
            temp.r = x[i__1].r, temp.i = x[i__1].i;
            if (noconj) {
            i__1 = j + 1;
            for (i__ = *n; i__ >= i__1; --i__) {
                i__2 = i__ + j * a_dim1;
                i__3 = ix;
                z__2.r = a[i__2].r * x[i__3].r - a[i__2].i * x[
                    i__3].i, z__2.i = a[i__2].r * x[i__3].i +
                    a[i__2].i * x[i__3].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
                ix -= *incx;
/* L180: */
            }
            if (nounit) {
                z_div(&z__1, &temp, &a[j + j * a_dim1]);
                temp.r = z__1.r, temp.i = z__1.i;
            }
            } else {
            i__1 = j + 1;
            for (i__ = *n; i__ >= i__1; --i__) {
                d_cnjg(&z__3, &a[i__ + j * a_dim1]);
                i__2 = ix;
                z__2.r = z__3.r * x[i__2].r - z__3.i * x[i__2].i,
                    z__2.i = z__3.r * x[i__2].i + z__3.i * x[
                    i__2].r;
                z__1.r = temp.r - z__2.r, z__1.i = temp.i -
                    z__2.i;
                temp.r = z__1.r, temp.i = z__1.i;
                ix -= *incx;
/* L190: */
            }
            if (nounit) {
                d_cnjg(&z__2, &a[j + j * a_dim1]);
                z_div(&z__1, &temp, &z__2);
                temp.r = z__1.r, temp.i = z__1.i;
            }
            }
            i__1 = jx;
            x[i__1].r = temp.r, x[i__1].i = temp.i;
            jx -= *incx;
/* L200: */
        }
        }
    }
    }

    return 0;

/*     End of ZTRSV . */

} /* ztrsv_ */
```