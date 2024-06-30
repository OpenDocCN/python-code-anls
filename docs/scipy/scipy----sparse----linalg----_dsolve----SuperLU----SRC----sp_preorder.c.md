# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sp_preorder.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file sp_preorder.c
 * \brief Permute and performs functions on columns of orginal matrix
 */
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * sp_preorder() permutes the columns of the original matrix. It performs
 * the following steps:
 *
 *    1. Apply column permutation perm_c[] to A's column pointers to form AC;
 *
 *    2. If options->Fact = DOFACT, then
 *       (1) Compute column elimination tree etree[] of AC'AC;
 *       (2) Post order etree[] to get a postordered elimination tree etree[],
 *           and a postorder permutation post[];
 *       (3) Apply post[] permutation to columns of AC;
 *       (4) Overwrite perm_c[] with the product perm_c * post.
 *
 * Arguments
 * =========
 *
 * options (input) superlu_options_t*
 *         Specifies whether or not the elimination tree will be re-used.
 *         If options->Fact == DOFACT, this means first time factor A, 
 *         etree is computed, postered, and output.
 *         Otherwise, re-factor A, etree is input, unchanged on exit.
 *
 * A       (input) SuperMatrix*
 *         Matrix A in A*X=B, of dimension (A->nrow, A->ncol). The number
 *         of the linear equations is A->nrow. Currently, the type of A can be:
 *         Stype = NC or SLU_NCP; Mtype = SLU_GE.
 *         In the future, more general A may be handled.
 *
 * perm_c  (input/output) int*
 *       Column permutation vector of size A->ncol, which defines the 
 *         permutation matrix Pc; perm_c[i] = j means column i of A is 
 *         in position j in A*Pc.
 *         If options->Fact == DOFACT, perm_c is both input and output.
 *         On output, it is changed according to a postorder of etree.
 *         Otherwise, perm_c is input.
 *
 * etree   (input/output) int*
 *         Elimination tree of Pc'*A'*A*Pc, dimension A->ncol.
 *         If options->Fact == DOFACT, etree is an output argument,
 *         otherwise it is an input argument.
 *         Note: etree is a vector of parent pointers for a forest whose
 *         vertices are the integers 0 to A->ncol-1; etree[root]==A->ncol.
 *
 * AC      (output) SuperMatrix*
 *         The resulting matrix after applied the column permutation
 *         perm_c[] to matrix A. The type of AC can be:
 *         Stype = SLU_NCP; Dtype = A->Dtype; Mtype = SLU_GE.
 * </pre>
 */
void
sp_preorder(superlu_options_t *options,  SuperMatrix *A, int *perm_c, 
        int *etree, SuperMatrix *AC)
{
    NCformat  *Astore;
    NCPformat *ACstore;
    int       *iwork, *post;
    register  int n, i;
    extern int check_perm(char *what, int n, int *perm);
    
    n = A->ncol;  // 获取矩阵 A 的列数

    /* Check the validity of column permutation perm_c[] */
    check_perm("sp_preorder", n, perm_c);  // 检查列置换 perm_c 的有效性

    Astore = A->Store;  // 获取矩阵 A 的存储格式

    /* Allocate working memory */
    iwork = intMalloc(n);  // 分配大小为 n 的整数工作内存
    post = intMalloc(n);   // 分配大小为 n 的整数数组 post

    /* Permute columns of A to form AC using perm_c[] */
    permute_SuperMatrix(n, perm_c, A, etree, AC);  // 使用 perm_c[] 对 A 的列进行置换，形成 AC

    /* Apply postorder permutation to AC if required */
    if (options->Fact == DOFACT) {
        spsymetree(n, AC, etree, post);  // 计算 AC'AC 的列消除树，并对其进行后序遍历
        perm_postorder(n, AC, post, ACstore);  // 对 AC 应用 post[] 的置换
        /* Update perm_c[] with perm_c * post */
        for (i = 0; i < n; ++i) perm_c[i] = post[perm_c[i]];  // 更新 perm_c[] 为 perm_c * post
    }

    SUPERLU_FREE(iwork);  // 释放工作内存 iwork
    SUPERLU_FREE(post);   // 释放数组 post
}
    /* 将列置换 perm_c 应用`
/* 将列置换 perm_c 应用于 A 的列指针，以便在 AC = A*Pc 中获得 NCP 格式。 */

// 设置 AC 的存储格式为 NCP
AC->Stype       = SLU_NCP;
// 复制 A 的数据类型到 AC
AC->Dtype       = A->Dtype;
// 复制 A 的存储类型到 AC
AC->Mtype       = A->Mtype;
// 复制 A 的行数到 AC
AC->nrow        = A->nrow;
// 复制 A 的列数到 AC
AC->ncol        = A->ncol;
// 复制 A 的存储指针到 Astore
Astore          = A->Store;
// 为 AC 的存储分配 NCPformat 结构体内存空间
ACstore = AC->Store = (void *) SUPERLU_MALLOC( sizeof(NCPformat) );
if ( !ACstore ) ABORT("SUPERLU_MALLOC fails for ACstore");
// 复制 A 的非零元数到 ACstore
ACstore->nnz    = Astore->nnz;
// 复制 A 的非零元值指针到 ACstore
ACstore->nzval  = Astore->nzval;
// 复制 A 的行索引指针到 ACstore
ACstore->rowind = Astore->rowind;
// 为 ACstore 的列起始位置数组分配内存空间
ACstore->colbeg = intMalloc(n);
if ( !(ACstore->colbeg) ) ABORT("SUPERLU_MALLOC fails for ACstore->colbeg");
// 为 ACstore 的列结束位置数组分配内存空间
ACstore->colend = intMalloc(n);
if ( !(ACstore->colend) ) ABORT("SUPERLU_MALLOC fails for ACstore->colend");
#if ( DEBUGlevel>=1 )
    // 如果调试级别大于等于1，则检查初始的 perm_c 数组权限
    check_perm("Initial perm_c", n, perm_c);
#endif

#if ( DEBUGlevel>=2 )
    // 如果调试级别大于等于2，则打印 pre_order 数组内容
    print_int_vec("pre_order:", n, perm_c);
#endif

    // 根据 perm_c 数组重排 ACstore 的列起始和结束位置
    for (i = 0; i < n; i++) {
        ACstore->colbeg[perm_c[i]] = Astore->colptr[i]; 
        ACstore->colend[perm_c[i]] = Astore->colptr[i+1];
    }
    
    // 如果选项中的 Fact 等于 DOFACT
    if ( options->Fact == DOFACT ) {
#undef ETREE_ATplusA
#ifdef ETREE_ATplusA
        /*--------------------------------------------
      计算 Pc*(A'+A)*Pc' 的 etree。
      --------------------------------------------*/
        int *b_colptr, *b_rowind, bnz, j;
        int *c_colbeg, *c_colend;

        /*printf("Use etree(A'+A)\n");*/

        // 形成矩阵 B = A + A'
        at_plus_a(n, Astore->nnz, Astore->colptr, Astore->rowind,
                  &bnz, &b_colptr, &b_rowind);

        // 形成矩阵 C = Pc*B*Pc'
        c_colbeg = (int*) SUPERLU_MALLOC(2*n*sizeof(int));
        c_colend = c_colbeg + n;
        if (!c_colbeg ) ABORT("SUPERLU_MALLOC fails for c_colbeg/c_colend");
        for (i = 0; i < n; i++) {
            c_colbeg[perm_c[i]] = b_colptr[i]; 
            c_colend[perm_c[i]] = b_colptr[i+1];
        }
        for (j = 0; j < n; ++j) {
            for (i = c_colbeg[j]; i < c_colend[j]; ++i) {
                b_rowind[i] = perm_c[b_rowind[i]];
            }
        }

        // 计算矩阵 C 的 etree
        sp_symetree(c_colbeg, c_colend, b_rowind, n, etree);

        SUPERLU_FREE(b_colptr);
        if ( bnz ) SUPERLU_FREE(b_rowind);
        SUPERLU_FREE(c_colbeg);
    
#else
        /*--------------------------------------------
      计算列消除树。
      --------------------------------------------*/
        sp_coletree(ACstore->colbeg, ACstore->colend, ACstore->rowind,
                    A->nrow, A->ncol, etree);
#endif

#if ( DEBUGlevel>=2 )
        // 如果调试级别大于等于2，则打印 etree 数组内容
        print_int_vec("etree:", n, etree);
#endif    
    
        // 在非对称模式下，不进行后序处理
        if ( options->SymmetricMode == NO ) {
            // 后序 etree
            post = (int *) TreePostorder(n, etree);

#if ( DEBUGlevel>=1 )
            // 如果调试级别大于等于1，则检查 post 数组的排列
            check_perm("post", n, post);    
#endif

#if ( DEBUGlevel>=2 )
            // 如果调试级别大于等于2，则打印 post 数组内容
            print_int_vec("post:", n+1, post);
#endif

            // 分配空间给 iwork 数组
            iwork = (int*) SUPERLU_MALLOC((n+1)*sizeof(int)); 
            if ( !iwork ) ABORT("SUPERLU_MALLOC fails for iwork[]");

            // 根据后序排列重新编号 etree
            for (i = 0; i < n; ++i) iwork[post[i]] = post[etree[i]];
            for (i = 0; i < n; ++i) etree[i] = iwork[i];

#if ( DEBUGlevel>=2 )
            // 如果调试级别大于等于2，则打印后序排列的 etree 数组内容
            print_int_vec("postorder etree:", n, etree);
#endif
#endif
    
    /* Postmultiply A*Pc by post[] */
    // 将 A*Pc 用 post[] 后置乘法进行后乘
    for (i = 0; i < n; ++i) iwork[post[i]] = ACstore->colbeg[i];
    // 将 ACstore->colbeg 按照 post[] 的顺序重新排列
    for (i = 0; i < n; ++i) ACstore->colbeg[i] = iwork[i];
    // 更新 ACstore->colbeg 为重新排列后的结果
    for (i = 0; i < n; ++i) iwork[post[i]] = ACstore->colend[i];
    // 将 ACstore->colend 按照 post[] 的顺序重新排列
    for (i = 0; i < n; ++i) ACstore->colend[i] = iwork[i];
    // 更新 ACstore->colend 为重新排列后的结果

    for (i = 0; i < n; ++i)
        iwork[i] = post[perm_c[i]];  /* product of perm_c and post */
    // 计算 perm_c 和 post 的乘积，结果保存在 iwork 中
    for (i = 0; i < n; ++i) perm_c[i] = iwork[i];
    // 更新 perm_c 为乘积结果

#if ( DEBUGlevel>=1 )
    check_perm("final perm_c", n, perm_c);    
#endif
#if ( DEBUGlevel>=2 )
    print_int_vec("Pc*post:", n, perm_c);
#endif
    // 根据 DEBUGlevel 输出调试信息

    SUPERLU_FREE (post);
    // 释放 post 数组所占用的内存空间
    SUPERLU_FREE (iwork);
    // 释放 iwork 数组所占用的内存空间
} /* end postordering */

} /* if options->Fact == DOFACT ... */
// 结束条件判断
```