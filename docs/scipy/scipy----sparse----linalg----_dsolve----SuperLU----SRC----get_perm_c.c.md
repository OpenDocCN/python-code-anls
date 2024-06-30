# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\get_perm_c.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file get_perm_c.c
 * \brief Matrix permutation operations
 *
 * <pre>
 * -- SuperLU routine (version 3.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * August 1, 2008
 * March 25, 2023  add METIS option
 * </pre>
 */
#include "slu_ddefs.h"
#include "colamd.h"

/*! \brief External function declaration for GENMMD */
extern int genmmd_(int *neqns, int_t *xadj, int_t *adjncy, 
           int *invp, int *perm, int_t *delta, int_t *dhead, 
           int_t *qsize, int_t *llist, int_t *marker, int_t *maxint, 
           int_t *nofsub);

/*! \brief Function to compute column permutation using COLAMD */
void
get_colamd(
       const int m,       /*!< number of rows in matrix A */
       const int n,       /*!< number of columns in matrix A */
       const int_t nnz,   /*!< number of nonzeros in matrix A */
       int_t *colptr,     /*!< column pointer of size n+1 for matrix A */
       int_t *rowind,     /*!< row indices of size nz for matrix A */
       int *perm_c        /*!< out - the column permutation vector */
       )
{
    size_t Alen;
    int_t *A, i, *p;
    int info;
    double knobs[COLAMD_KNOBS];
    int_t stats[COLAMD_STATS];

    Alen = COLAMD_recommended(nnz, m, n); /*! Compute recommended length for COLAMD */

    COLAMD_set_defaults(knobs); /*! Set default parameters for COLAMD */

    if ( !(A = intMalloc(Alen)) ) ABORT("Malloc fails for A[]"); /*! Allocate memory for A */
    if ( !(p = intMalloc(n+1)) )  ABORT("Malloc fails for p[]"); /*! Allocate memory for p */
    for (i = 0; i <= n; ++i) p[i] = colptr[i]; /*! Copy colptr to p */
    for (i = 0; i < nnz; ++i) A[i] = rowind[i]; /*! Copy rowind to A */
    
    info = COLAMD_MAIN(m, n, Alen, A, p, knobs, stats); /*! Perform column ordering with COLAMD */

    //printf("after COLAMD_MAIN info %d\n", info); /*! Print debug information */
    if ( info == FALSE ) ABORT("COLAMD failed"); /*! Abort if COLAMD fails */

    for (i = 0; i < n; ++i) perm_c[p[i]] = i; /*! Store the column permutation vector */

    SUPERLU_FREE(A); /*! Free allocated memory for A */
    SUPERLU_FREE(p); /*! Free allocated memory for p */
} /* end get_colamd */

/*! \brief Function to compute column permutation using METIS */
void
get_metis(
      int n,             /*!< dimension of matrix B */
      int_t bnz,         /*!< number of nonzeros in matrix A */
      int_t *b_colptr,   /*!< column pointer of size n+1 for matrix B */
      int_t *b_rowind,   /*!< row indices of size bnz for matrix B */
      int *perm_c        /*!< out - the column permutation vector */
      )
{
#ifdef HAVE_METIS
    /*#define METISOPTIONS 8*/
#define METISOPTIONS 40
    int_t metis_options[METISOPTIONS];
    int numflag = 0; /* C-Style ordering */
    int_t i, nm;
    int_t *perm, *iperm;

    extern int METIS_NodeND(int_t*, int_t*, int_t*, int_t*, int_t*,
                int_t*, int_t*);

    metis_options[0] = 0; /*! Use Defaults for now */

    perm = intMalloc(2*n); /*! Allocate memory for perm */
    if (!perm) ABORT("intMalloc fails for perm."); /*! Abort if memory allocation fails */
    iperm = perm + n;
    nm = n;

    /*! Call METIS */
#undef USEEND
#ifdef USEEND
    METIS_EdgeND(&nm, b_colptr, b_rowind, &numflag, metis_options,
         perm, iperm);
#else

    /*! Earlier version 3.x.x */
    /* 调用 METIS 库中的 NodeND 函数进行节点重编号
       &nm: 输出参数，存储节点数目
       b_colptr: 输入参数，稀疏矩阵 B 的列指针数组
       b_rowind: 输入参数，稀疏矩阵 B 的行索引数组
       NULL: 无需传递 numflag 参数
       NULL: 无需传递 metis_options 参数
       perm: 输出参数，存储重编号后的节点排列顺序
       iperm: 输出参数，存储逆重编号后的节点排列顺序
    */
    METIS_NodeND(&nm, b_colptr, b_rowind, NULL, NULL, perm, iperm);

    /* 检查节点重编号的结果是否符合预期
       "metis perm": 日志中使用的标签，标识这个检查
       n: 节点数目
       perm: METIS_NodeND 函数返回的节点排列顺序数组
    */
    check_perm_dist("metis perm",  n, perm);
    /* 复制置换向量到 SuperLU 数据结构中 */
    for (i = 0; i < n; ++i) perm_c[i] = iperm[i];

    // 释放内存
    SUPERLU_FREE(b_colptr);
    SUPERLU_FREE(b_rowind);
    SUPERLU_FREE(perm);
#endif /* HAVE_METIS */
}

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * Form the structure of A'*A. A is an m-by-n matrix in column oriented
 * format represented by (colptr, rowind). The output A'*A is in column
 * oriented format (symmetrically, also row oriented), represented by
 * (ata_colptr, ata_rowind).
 *
 * This routine is modified from GETATA routine by Tim Davis.
 * The complexity of this algorithm is: SUM_{i=1,m} r(i)^2,
 * i.e., the sum of the square of the row counts.
 *
 * Questions
 * =========
 *     o  Do I need to withhold the *dense* rows?
 *     o  How do I know the number of nonzeros in A'*A?
 * </pre>
 */
void
getata(
       const int m,      /* A 矩阵的行数。 */
       const int n,      /* A 矩阵的列数。 */
       const int_t nz,     /* A 矩阵中的非零元素数。 */
       int_t *colptr,      /* A 矩阵的列指针，大小为 n+1。 */
       int_t *rowind,      /* A 矩阵的行索引，大小为 nz。 */
       int_t *atanz,       /* 输出 - 返回 A'*A 中实际的非零元素数。 */
       int_t **ata_colptr, /* 输出 - 大小为 n+1 */
       int_t **ata_rowind  /* 输出 - 大小为 *atanz */
       )
{
    register int_t i, j, k, col, num_nz, ti, trow;
    int_t *marker, *b_colptr, *b_rowind;
    int_t *t_colptr, *t_rowind; /* T = A' 的列向量形式 */

    // 分配 marker 数组内存
    if ( !(marker = (int_t*) SUPERLU_MALLOC((SUPERLU_MAX(m,n)+1)*sizeof(int_t))) )
    ABORT("SUPERLU_MALLOC fails for marker[]");
    // 分配 t_colptr 数组内存
    if ( !(t_colptr = (int_t*) SUPERLU_MALLOC((m+1) * sizeof(int_t))) )
    ABORT("SUPERLU_MALLOC t_colptr[]");
    // 分配 t_rowind 数组内存
    if ( !(t_rowind = (int_t*) SUPERLU_MALLOC(nz * sizeof(int_t))) )
    ABORT("SUPERLU_MALLOC fails for t_rowind[]");

    /* 获取 T 的每一列的元素计数，并设置列指针 */
    for (i = 0; i < m; ++i) marker[i] = 0;
    for (j = 0; j < n; ++j) {
        for (i = colptr[j]; i < colptr[j+1]; ++i)
            ++marker[rowind[i]];
    }
    t_colptr[0] = 0;
    for (i = 0; i < m; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    /* 将 A 转置到 T */
    for (j = 0; j < n; ++j) {
        for (i = colptr[j]; i < colptr[j+1]; ++i) {
            col = rowind[i];
            t_rowind[marker[col]] = j;
            ++marker[col];
        }
    }
    /* ----------------------------------------------------------------
       compute B = T * A, where column j of B is:

       Struct (B_*j) =    UNION   ( Struct (T_*k) )
                        A_kj != 0

       do not include the diagonal entry
   
       ( Partition A as: A = (A_*1, ..., A_*n)
         Then B = T * A = (T * A_*1, ..., T * A_*n), where
         T * A_*j = (T_*1, ..., T_*m) * A_*j.  )
       ---------------------------------------------------------------- */

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;
    /* 标记数组初始化，-1 表示未使用 */

    /* First pass determines number of nonzeros in B */
    num_nz = 0;
    /* 初始化非零元素计数器 */

    for (j = 0; j < n; ++j) {
    /* 遍历 B 矩阵的每一列 */

        /* Flag the diagonal so it's not included in the B matrix */
        marker[j] = j;
        /* 标记对角线元素，以便后续在 B 矩阵中排除对角线元素 */

        for (i = colptr[j]; i < colptr[j+1]; ++i) {
        /* 遍历 A 矩阵中第 j 列的非零元素 */

            /* A_kj is nonzero, add pattern of column T_*k to B_*j */
            k = rowind[i];
            /* 获取 A 矩阵中非零元素的行索引 */

            for (ti = t_colptr[k]; ti < t_colptr[k+1]; ++ti) {
            /* 遍历 T 矩阵中与 A_kj 对应的非零元素 */

                trow = t_rowind[ti];
                /* 获取 T 矩阵中非零元素的行索引 */

                if ( marker[trow] != j ) {
                    marker[trow] = j;
                    /* 标记当前行，在 B 矩阵中添加其模式 */
                    num_nz++;
                    /* 非零元素计数加一 */
                }
            }
        }
    }
    *atanz = num_nz;
    /* 存储 B 矩阵中的非零元素总数 */

    /* Allocate storage for A'*A */
    if ( !(*ata_colptr = (int_t*) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for ata_colptr[]");
    /* 为 A' * A 的列指针数组分配存储空间 */

    if ( *atanz ) {
        if ( !(*ata_rowind = (int_t*) SUPERLU_MALLOC( *atanz * sizeof(int_t)) ) )
            ABORT("SUPERLU_MALLOC fails for ata_rowind[]");
        /* 如果 B 矩阵中存在非零元素，则为其行索引数组分配存储空间 */
    }

    b_colptr = *ata_colptr; /* aliasing */
    b_rowind = *ata_rowind;
    /* 设置别名，方便后续操作 */

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;
    /* 重新初始化标记数组 */

    /* Compute each column of B, one at a time */
    num_nz = 0;
    /* 重新初始化非零元素计数器 */

    for (j = 0; j < n; ++j) {
    /* 再次遍历 B 矩阵的每一列 */

        b_colptr[j] = num_nz;
        /* 设置当前列在行索引数组中的起始位置 */

        /* Flag the diagonal so it's not included in the B matrix */
        marker[j] = j;
        /* 标记对角线元素，以便后续在 B 矩阵中排除对角线元素 */

        for (i = colptr[j]; i < colptr[j+1]; ++i) {
        /* 遍历 A 矩阵中第 j 列的非零元素 */

            /* A_kj is nonzero, add pattern of column T_*k to B_*j */
            k = rowind[i];
            /* 获取 A 矩阵中非零元素的行索引 */

            for (ti = t_colptr[k]; ti < t_colptr[k+1]; ++ti) {
            /* 遍历 T 矩阵中与 A_kj 对应的非零元素 */

                trow = t_rowind[ti];
                /* 获取 T 矩阵中非零元素的行索引 */

                if ( marker[trow] != j ) {
                    marker[trow] = j;
                    b_rowind[num_nz++] = trow;
                    /* 将当前行加入到 B 矩阵的行索引数组中 */
                }
            }
        }
    }
    b_colptr[n] = num_nz;
    /* 设置最后一个元素的位置 */

    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_colptr);
    SUPERLU_FREE(t_rowind);
    /* 释放临时使用的内存空间 */
} /* end getata */


/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * Form the structure of A'+A. A is an n-by-n matrix in column oriented
 * format represented by (colptr, rowind). The output A'+A is in column
 * oriented format (symmetrically, also row oriented), represented by
 * (b_colptr, b_rowind).
 * </pre>
 */
void
at_plus_a(
      const int n,      /* number of columns in matrix A. */
      const int_t nz,   /* number of nonzeros in matrix A */
      int_t *colptr,    /* column pointer of size n+1 for matrix A. */
      int_t *rowind,    /* row indices of size nz for matrix A. */
      int_t *bnz,       /* out - on exit, returns the actual number of
                           nonzeros in matrix A'*A. */
      int_t **b_colptr, /* out - size n+1 */
      int_t **b_rowind  /* out - size *bnz */
      )
{
    register int_t i, j, k, col, num_nz;
    int_t *t_colptr, *t_rowind; /* a column oriented form of T = A' */
    int_t *marker;

    /* Allocate memory for marker array */
    if ( !(marker = (int_t*) SUPERLU_MALLOC( n * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for marker[]");
    
    /* Allocate memory for t_colptr array */
    if ( !(t_colptr = (int_t*) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for t_colptr[]");
    
    /* Allocate memory for t_rowind array */
    if ( !(t_rowind = (int_t*) SUPERLU_MALLOC( nz * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails t_rowind[]");

    
    /* Get counts of each column of T, and set up column pointers */
    for (i = 0; i < n; ++i) marker[i] = 0;
    for (j = 0; j < n; ++j) {
        for (i = colptr[j]; i < colptr[j+1]; ++i)
            ++marker[rowind[i]];
    }
    t_colptr[0] = 0;
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    /* Transpose the matrix from A to T */
    for (j = 0; j < n; ++j)
        for (i = colptr[j]; i < colptr[j+1]; ++i) {
            col = rowind[i];
            t_rowind[marker[col]] = j;
            ++marker[col];
        }


    /* ----------------------------------------------------------------
       compute B = A + T, where column j of B is:

       Struct (B_*j) = Struct (A_*k) UNION Struct (T_*k)

       do not include the diagonal entry
       ---------------------------------------------------------------- */

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;

    /* First pass determines number of nonzeros in B */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
        /* Flag the diagonal so it's not included in the B matrix */
        marker[j] = j;

        /* Add pattern of column A_*k to B_*j */
        for (i = colptr[j]; i < colptr[j+1]; ++i) {
            k = rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                ++num_nz;
            }
        }

        /* Add pattern of column T_*k to B_*j */
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            k = t_rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                ++num_nz;
            }
        }
    }
    *bnz = num_nz;
    
    /* Allocate storage for A+A' */
    /* 分配并初始化 b_colptr 数组 */
    if ( !(*b_colptr = (int_t*) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for b_colptr[]");

    /* 如果 bnz 不为零，分配并初始化 b_rowind 数组 */
    if ( *bnz) {
        if ( !(*b_rowind = (int_t*) SUPERLU_MALLOC( *bnz * sizeof(int_t)) ) )
            ABORT("SUPERLU_MALLOC fails for b_rowind[]");
    }
    
    /* 初始化 marker 数组，将每个元素设为 -1 */
    for (i = 0; i < n; ++i) marker[i] = -1;
    
    /* 计算 B 矩阵的每一列 */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
        (*b_colptr)[j] = num_nz;
        
        /* 标记对角线元素，确保不包含在 B 矩阵中 */
        marker[j] = j;

        /* 将 A_*j 列的模式添加到 B_*j 列 */
        for (i = colptr[j]; i < colptr[j+1]; ++i) {
            k = rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                (*b_rowind)[num_nz++] = k;
            }
        }

        /* 将 T_*j 列的模式添加到 B_*j 列 */
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            k = t_rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                (*b_rowind)[num_nz++] = k;
            }
        }
    }
    (*b_colptr)[n] = num_nz;  /* 设置 B 列指针的最后一个元素为 num_nz */
       
    SUPERLU_FREE(marker);     /* 释放 marker 数组的内存 */
    SUPERLU_FREE(t_colptr);   /* 释放 t_colptr 数组的内存 */
    SUPERLU_FREE(t_rowind);   /* 释放 t_rowind 数组的内存 */
} /* end at_plus_a */

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * GET_PERM_C obtains a permutation matrix Pc, by applying the multiple
 * minimum degree ordering code by Joseph Liu to matrix A'*A or A+A'.
 * or using approximate minimum degree column ordering by Davis et. al.
 * The LU factorization of A*Pc tends to have less fill than the LU 
 * factorization of A.
 *
 * Arguments
 * =========
 *
 * ispec   (input) int
 *         Specifies the type of column ordering to reduce fill:
 *         = 1: minimum degree on the structure of A^T * A
 *         = 2: minimum degree on the structure of A^T + A
 *         = 3: approximate minimum degree for unsymmetric matrices
 *         If ispec == 0, the natural ordering (i.e., Pc = I) is returned.
 * 
 * A       (input) SuperMatrix*
 *         Matrix A in A*X=B, of dimension (A->nrow, A->ncol). The number
 *         of the linear equations is A->nrow. Currently, the type of A 
 *         can be: Stype = NC; Dtype = _D; Mtype = GE. In the future,
 *         more general A can be handled.
 *
 * perm_c  (output) int*
 *       Column permutation vector of size A->ncol, which defines the 
 *         permutation matrix Pc; perm_c[i] = j means column i of A is 
 *         in position j in A*Pc.
 * </pre>
 */
void
get_perm_c(int ispec, SuperMatrix *A, int *perm_c)
{
    // 获取输入矩阵 A 的存储格式
    NCformat *Astore = A->Store;
    int m, n;
    int_t i, bnz = 0, *b_colptr, *b_rowind;
    int_t delta, maxint, nofsub;
    int *invp;
    int_t *dhead, *qsize, *llist, *marker;
    double t;
    
    // 获取矩阵 A 的行数和列数
    m = A->nrow;
    n = A->ncol;

    // 计时器启动
    t = SuperLU_timer_();
    // 根据 ispec 进行不同的处理
    switch ( ispec ) {
    case (NATURAL): /* Natural ordering */
        // 对于自然顺序，返回列标识符向量 perm_c 为自然顺序
        for (i = 0; i < n; ++i) perm_c[i] = i;
#if ( PRNTlevel>=1 )
        // 打印信息，使用自然列序
        printf("Use natural column ordering.\n");
#endif
        return;
    case (MMD_ATA): /* Minimum degree ordering on A'*A */
        // 调用 getata 函数进行 A'*A 的最小度排序
        getata(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
                 &bnz, &b_colptr, &b_rowind);
#if ( PRNTlevel>=1 )
        // 打印信息，使用 A'*A 的最小度排序
        printf("Use minimum degree ordering on A'*A.\n");
#endif
        // 计时器停止并记录时间
        t = SuperLU_timer_() - t;
        /*printf("Form A'*A time = %8.3f\n", t);*/
        break;
    case (MMD_AT_PLUS_A): /* Minimum degree ordering on A'+A */
        // 检查 A 是否为方阵
        if ( m != n ) ABORT("Matrix is not square");
        // 调用 at_plus_a 函数进行 A'+A 的最小度排序
        at_plus_a(n, Astore->nnz, Astore->colptr, Astore->rowind,
              &bnz, &b_colptr, &b_rowind);
#if ( PRNTlevel>=1 )
        // 打印信息，使用 A'+A 的最小度排序
        printf("Use minimum degree ordering on A'+A.\n");
#endif
        // 计时器停止并记录时间
        t = SuperLU_timer_() - t;
        /*printf("Form A'+A time = %8.3f\n", t);*/
        break;
    case (COLAMD): /* Approximate minimum degree column ordering. */
        // 调用 get_colamd 函数进行近似最小度列排序
        get_colamd(m, n, Astore->nnz, Astore->colptr, Astore->rowind, perm_c);
#if ( PRNTlevel>=1 )
        // 打印信息，使用近似最小度列排序
        printf(".. Use approximate minimum degree column ordering.\n");
#endif
        return;
    }
}
#ifdef HAVE_METIS
        case METIS_ATA: /* METIS ordering on A'*A */
        // 调用 getata 函数获取 A'*A 的结构信息
        getata(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
             &bnz, &b_colptr, &b_rowind);

        if ( bnz ) { /* non-empty adjacency structure */
          // 如果非空邻接结构，则使用 METIS 库进行重新排序
          get_metis(n, bnz, b_colptr, b_rowind, perm_c);
        } else { /* e.g., diagonal matrix */
        // 如果是空结构（如对角矩阵），直接使用默认顺序
        for (i = 0; i < n; ++i) perm_c[i] = i;
        SUPERLU_FREE(b_colptr);
        /* b_rowind is not allocated in this case */
        }

#if ( PRNTlevel>=1 )
        printf(".. Use METIS ordering on A'*A\n");
#endif
        return;
#endif
    
    default:
    ABORT("Invalid ISPEC");
    }

    if ( bnz != 0 ) {
    t = SuperLU_timer_();

    /* Initialize and allocate storage for GENMMD. */
    delta = 0; /* DELTA is a parameter to allow the choice of nodes
              whose degree <= min-degree + DELTA. */
    maxint = 2147483647; /* 2**31 - 1 */
    invp = (int *) SUPERLU_MALLOC((n+delta)*sizeof(int));
    if ( !invp ) ABORT("SUPERLU_MALLOC fails for invp.");
    dhead = intMalloc(n+delta);
    if ( !dhead ) ABORT("SUPERLU_MALLOC fails for dhead.");
    qsize = intMalloc(n+delta);
    if ( !qsize ) ABORT("SUPERLU_MALLOC fails for qsize.");
    llist = intMalloc(n);
    if ( !llist ) ABORT("SUPERLU_MALLOC fails for llist.");
    marker = intMalloc(n);
    if ( !marker ) ABORT("SUPERLU_MALLOC fails for marker.");

    /* Transform adjacency list into 1-based indexing required by GENMMD.*/
    // 将邻接列表转换为 GENMMD 需要的基于1的索引
    for (i = 0; i <= n; ++i) ++b_colptr[i];
    for (i = 0; i < bnz; ++i) ++b_rowind[i];
    
    genmmd_(&n, b_colptr, b_rowind, perm_c, invp, &delta, dhead, 
        qsize, llist, marker, &maxint, &nofsub);

    /* Transform perm_c into 0-based indexing. */
    // 将 perm_c 转换为基于0的索引
    for (i = 0; i < n; ++i) --perm_c[i];

    SUPERLU_FREE(invp);
    SUPERLU_FREE(dhead);
    SUPERLU_FREE(qsize);
    SUPERLU_FREE(llist);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(b_rowind);

    t = SuperLU_timer_() - t;
    /*  printf("call GENMMD time = %8.3f\n", t);*/

    } else { /* Empty adjacency structure */
    // 如果邻接结构为空，则使用默认顺序
    for (i = 0; i < n; ++i) perm_c[i] = i;
    }

    SUPERLU_FREE(b_colptr);
}
```