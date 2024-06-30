# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\util.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file util.c
 * \brief Utility functions
 * 
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November, 2010
 *
 * Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 *
 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 * EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 *
 * Permission is hereby granted to use or copy this program for any
 * purpose, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is
 * granted, provided the above notices are retained, and a notice that
 * the code was modified is included with the above copyright notice.
 * </pre>
 */


#include <math.h>
#include "slu_ddefs.h"

/*! \brief Global statistics variable
 */

/*! \brief Function to print an error message and exit.
 * 
 * \param msg Error message to print.
 */
void superlu_abort_and_exit(char* msg)
{
    fprintf(stderr, "%s", msg);  // 输出错误信息到标准错误流
    exit (-1);  // 终止程序执行，并返回 -1
}

/*! \brief Set the default values for the options argument.
 * 
 * \param options Pointer to the structure holding SuperLU options.
 */
void set_default_options(superlu_options_t *options)
{
    options->Fact = DOFACT;  // 设置因子化选项为 DOFACT
    options->Equil = YES;  // 启用等价选项
    options->ColPerm = COLAMD;  // 使用列置换策略 COLAMD
    options->Trans = NOTRANS;  // 不进行转置
    options->IterRefine = NOREFINE;  // 不进行迭代细化
    options->DiagPivotThresh = 1.0;  // 对角元素主元阈值为 1.0
    options->SymmetricMode = NO;  // 非对称模式
    options->PivotGrowth = NO;  // 不计算主元增长
    options->ConditionNumber = NO;  // 不计算条件数
    options->PrintStat = YES;  // 打印统计信息
}

/*! \brief Set the default values for the options argument for ILU.
 * 
 * \param options Pointer to the structure holding SuperLU options.
 */
void ilu_set_default_options(superlu_options_t *options)
{
    set_default_options(options);  // 调用设置默认选项的函数

    /* further options for incomplete factorization */
    options->DiagPivotThresh = 0.1;  // 对角元素主元阈值为 0.1
    options->RowPerm = LargeDiag_MC64;  // 使用行置换策略 LargeDiag_MC64
    options->ILU_DropRule = DROP_BASIC | DROP_AREA;  // 使用基本和区域丢弃规则
    options->ILU_DropTol = 1e-4;  // ILU 丢弃容限为 1e-4
    options->ILU_FillFactor = 10.0;  // ILU 填充因子为 10.0
    options->ILU_Norm = INF_NORM;  // ILU 使用无穷范数
    options->ILU_MILU = SILU;  // 使用选择不完全 LU 分解
    options->ILU_MILU_Dim = 3.0;  // MILU 维度为 3.0
    options->ILU_FillTol = 1e-2;  // ILU 填充容限为 1e-2
}

/*! \brief Print the options setting.
 * 
 * \param options Pointer to the structure holding SuperLU options.
 */
void print_options(superlu_options_t *options)
{
    printf(".. options:\n");
    printf("\tFact\t %8d\n", options->Fact);  // 打印因子化选项
    printf("\tEquil\t %8d\n", options->Equil);  // 打印等价选项
    printf("\tColPerm\t %8d\n", options->ColPerm);  // 打印列置换策略
    printf("\tDiagPivotThresh %8.4f\n", options->DiagPivotThresh);  // 打印对角元素主元阈值
    printf("\tTrans\t %8d\n", options->Trans);  // 打印转置选项
    printf("\tIterRefine\t%4d\n", options->IterRefine);  // 打印迭代细化选项
    printf("\tSymmetricMode\t%4d\n", options->SymmetricMode);  // 打印对称模式选项
    printf("\tPivotGrowth\t%4d\n", options->PivotGrowth);  // 打印主元增长选项
    printf("\tConditionNumber\t%4d\n", options->ConditionNumber);  // 打印条件数计算选项
    printf("..\n");
}
/*! \brief Print ILU options stored in the options struct.
 *
 *  This function prints various ILU options stored in the `options` structure.
 *  It includes DiagPivotThresh, tau (ILU_DropTol), gamma (ILU_FillFactor),
 *  DropRule, MILU, MILU_ALPHA, and DiagFillTol.
 *
 *  \param options Pointer to a superlu_options_t struct containing ILU options.
 */
void print_ilu_options(superlu_options_t *options)
{
    // Print header for ILU options
    printf(".. ILU options:\n");

    // Print DiagPivotThresh option
    printf("\tDiagPivotThresh\t%6.2e\n", options->DiagPivotThresh);

    // Print tau (ILU_DropTol) option
    printf("\ttau\t%6.2e\n", options->ILU_DropTol);

    // Print gamma (ILU_FillFactor) option
    printf("\tgamma\t%6.2f\n", options->ILU_FillFactor);

    // Print DropRule option
    printf("\tDropRule\t%0x\n", options->ILU_DropRule);

    // Print MILU option
    printf("\tMILU\t%d\n", options->ILU_MILU);

    // Print MILU_ALPHA option
    printf("\tMILU_ALPHA\t%6.2e\n", MILU_ALPHA);

    // Print DiagFillTol option
    printf("\tDiagFillTol\t%6.2e\n", options->ILU_FillTol);

    // Print footer for ILU options
    printf("..\n");
}

/*! \brief Deallocate the storage for a SuperMatrix object.
 *
 *  This function frees the memory allocated for the storage of the matrix
 *  stored in the SuperMatrix object `A`.
 *
 *  \param A Pointer to a SuperMatrix object to be deallocated.
 */
void
Destroy_SuperMatrix_Store(SuperMatrix *A)
{
    SUPERLU_FREE ( A->Store );
}

/*! \brief Deallocate the storage for a SuperMatrix in CompCol format.
 *
 *  This function frees the memory allocated for the storage of the matrix
 *  stored in the SuperMatrix object `A` in CompCol format.
 *
 *  \param A Pointer to a SuperMatrix object in CompCol format to be deallocated.
 */
void
Destroy_CompCol_Matrix(SuperMatrix *A)
{
    SUPERLU_FREE( ((NCformat *)A->Store)->rowind );
    SUPERLU_FREE( ((NCformat *)A->Store)->colptr );
    SUPERLU_FREE( ((NCformat *)A->Store)->nzval );
    SUPERLU_FREE( A->Store );
}

/*! \brief Deallocate the storage for a SuperMatrix in CompRow format.
 *
 *  This function frees the memory allocated for the storage of the matrix
 *  stored in the SuperMatrix object `A` in CompRow format.
 *
 *  \param A Pointer to a SuperMatrix object in CompRow format to be deallocated.
 */
void
Destroy_CompRow_Matrix(SuperMatrix *A)
{
    SUPERLU_FREE( ((NRformat *)A->Store)->colind );
    SUPERLU_FREE( ((NRformat *)A->Store)->rowptr );
    SUPERLU_FREE( ((NRformat *)A->Store)->nzval );
    SUPERLU_FREE( A->Store );
}

/*! \brief Deallocate the storage for a SuperMatrix in SuperNode format.
 *
 *  This function frees the memory allocated for the storage of the matrix
 *  stored in the SuperMatrix object `A` in SuperNode format.
 *
 *  \param A Pointer to a SuperMatrix object in SuperNode format to be deallocated.
 */
void
Destroy_SuperNode_Matrix(SuperMatrix *A)
{
    SUPERLU_FREE ( ((SCformat *)A->Store)->rowind );
    SUPERLU_FREE ( ((SCformat *)A->Store)->rowind_colptr );
    SUPERLU_FREE ( ((SCformat *)A->Store)->nzval );
    SUPERLU_FREE ( ((SCformat *)A->Store)->nzval_colptr );
    SUPERLU_FREE ( ((SCformat *)A->Store)->col_to_sup );
    SUPERLU_FREE ( ((SCformat *)A->Store)->sup_to_col );
    SUPERLU_FREE ( A->Store );
}

/*! \brief Deallocate the storage for a SuperMatrix in NCP format.
 *
 *  This function frees the memory allocated for the storage of the matrix
 *  stored in the SuperMatrix object `A` in NCP format.
 *
 *  \param A Pointer to a SuperMatrix object in NCP format to be deallocated.
 */
void
Destroy_CompCol_Permuted(SuperMatrix *A)
{
    SUPERLU_FREE ( ((NCPformat *)A->Store)->colbeg );
    SUPERLU_FREE ( ((NCPformat *)A->Store)->colend );
    SUPERLU_FREE ( A->Store );
}

/*! \brief Deallocate the storage for a SuperMatrix in DN format.
 *
 *  This function frees the memory allocated for the storage of the matrix
 *  stored in the SuperMatrix object `A` in DN format.
 *
 *  \param A Pointer to a SuperMatrix object in DN format to be deallocated.
 */
void
Destroy_Dense_Matrix(SuperMatrix *A)
{
    DNformat* Astore = A->Store;
    SUPERLU_FREE (Astore->nzval);
    SUPERLU_FREE ( A->Store );
}

/*! \brief Reset repfnz[] for the current column 
 *
 *  This function resets the values in the array `repfnz` at positions specified
 *  by `segrep` to EMPTY.
 *
 *  \param nseg Number of segments in `segrep`.
 *  \param segrep Array containing segment indices.
 *  \param repfnz Array to be reset.
 */
void
resetrep_col (const int nseg, const int *segrep, int *repfnz)
{
    int_t i, irep;
    
    for (i = 0; i < nseg; i++) {
        irep = segrep[i];
        repfnz[irep] = EMPTY;
    }
}

/*! \brief Count the total number of nonzeros in factors L and U, and in the symmetrically reduced L.
 *
 *  This function calculates the total number of nonzeros in factors L and U,
 *  and in the symmetrically reduced L for a given matrix.
 *
 *  \param n Number of columns in the matrix.
 *  \param xprune Array storing pruned rows.
 *  \param nnzL Pointer to store the total number of nonzeros in L.
 *  \param nnzU Pointer to store the total number of nonzeros in U.
 *  \param Glu Pointer to GlobalLU_t structure containing matrix information.
 */
void
countnz(const int n, int_t *xprune, int_t *nnzL, int_t *nnzU, GlobalLU_t *Glu)
{
    int          nsuper, fsupc, i, j;
    int_t        jlen;
#if ( DEBUGlevel>=1 )
    int_t        irep = 0, nnzL0 = 0;
#endif
    int          *xsup;
    int_t        *xlsub;

    xsup   = Glu->xsup;
    xlsub  = Glu->xlsub;
    *nnzL  = 0;
    *nnzU  = (Glu->xusub)[n];
    nsuper = (Glu->supno)[n];

    if ( n <= 0 ) return;

    /* 
     * For each supernode
     */
    for (i = 0; i <= nsuper; i++) {
        fsupc = xsup[i];
        jlen = xlsub[fsupc+1] - xlsub[fsupc];

        for (j = fsupc; j < xsup[i+1]; j++) {
            *nnzL += jlen;
            *nnzU += j - fsupc + 1;
            jlen--;
        }
    }
}
#if ( DEBUGlevel>=1 )
    // 如果调试级别大于等于1，则执行以下代码块
    irep = xsup[i+1] - 1;
    // 计算irep，其值为xsup[i+1]减1
    nnzL0 += xprune[irep] - xlsub[irep];
    // 更新nnzL0，加上xprune[irep]减去xlsub[irep]的值
#endif
}

#if ( DEBUGlevel>=1 )
    // 如果调试级别大于等于1，则执行以下代码块
    printf("\tNo of nonzeros in symm-reduced L = %lld\n", (long long) nnzL0); fflush(stdout);
    // 打印对称减少的L中的非零元素数量到标准输出
#endif
}

/*! \brief Count the total number of nonzeros in factors L and U.
 */
void
ilu_countnz(const int n, int_t *nnzL, int_t *nnzU, GlobalLU_t *Glu)
{
    int          nsuper, fsupc, i, j;
    int          jlen;
    int          *xsup;
    int_t        *xlsub;

    xsup   = Glu->xsup;
    xlsub  = Glu->xlsub;
    *nnzL  = 0;
    *nnzU  = (Glu->xusub)[n];
    nsuper = (Glu->supno)[n];

    if ( n <= 0 ) return;

    /*
     * For each supernode
     */
    for (i = 0; i <= nsuper; i++) {
        // 对每个超节点
        fsupc = xsup[i];
        jlen = xlsub[fsupc+1] - xlsub[fsupc];

        for (j = fsupc; j < xsup[i+1]; j++) {
            // 对于超节点i中的每一列j
            *nnzL += jlen;
            *nnzU += j - fsupc + 1;
            jlen--;
        }
        //irep = xsup[i+1] - 1;
    }
}


/*! \brief Fix up the data storage lsub for L-subscripts. It removes the subscript sets for structural pruning,    and applies permuation to the remaining subscripts.
 */
void
fixupL(const int n, const int *perm_r, GlobalLU_t *Glu)
{
    int nsuper, fsupc, i, k;
    int_t nextl, j, jstrt;
    int   *xsup;
    int_t *lsub, *xlsub;

    if ( n <= 1 ) return;

    xsup   = Glu->xsup;
    lsub   = Glu->lsub;
    xlsub  = Glu->xlsub;
    nextl  = 0;
    nsuper = (Glu->supno)[n];
    
    /* 
     * For each supernode ...
     */
    for (i = 0; i <= nsuper; i++) {
        // 对每个超节点i
        fsupc = xsup[i];
        jstrt = xlsub[fsupc];
        xlsub[fsupc] = nextl;
        for (j = jstrt; j < xlsub[fsupc+1]; j++) {
            // 对于超节点i中的每个非零元素j
            lsub[nextl] = perm_r[lsub[j]]; /* Now indexed into P*A */
            nextl++;
        }
        for (k = fsupc+1; k < xsup[i+1]; k++) 
            xlsub[k] = nextl;    /* Other columns in supernode i */
    }

    xlsub[n] = nextl;
}


/*! \brief Diagnostic print of segment info after panel_dfs().
 */
void print_panel_seg(int n, int w, int jcol, int nseg, 
             int *segrep, int *repfnz)
{
    int j, k;
    
    for (j = jcol; j < jcol+w; j++) {
        // 对于每一列jcol到jcol+w-1
        printf("\tcol %d:\n", j);
        // 打印列号
        for (k = 0; k < nseg; k++)
            // 对于每个段k
            printf("\t\tseg %d, segrep %d, repfnz %d\n", k, 
                segrep[k], repfnz[(j-jcol)*n + segrep[k]]);
                // 打印段号、segrep值、以及repfnz数组中的值
    }

}


void
StatInit(SuperLUStat_t *stat)
{
    register int i, w, panel_size, relax;

    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    w = SUPERLU_MAX(panel_size, relax);
    stat->panel_histo = int32Calloc(w+1);
    stat->utime = (double *) SUPERLU_MALLOC(NPHASES * sizeof(double));
    if (!stat->utime) ABORT("SUPERLU_MALLOC fails for stat->utime");
    stat->ops = (flops_t *) SUPERLU_MALLOC(NPHASES * sizeof(flops_t));
    if (!stat->ops) ABORT("SUPERLU_MALLOC fails for stat->ops");
    for (i = 0; i < NPHASES; ++i) {
        stat->utime[i] = 0.;
        stat->ops[i] = 0.;
    }
    stat->TinyPivots = 0;
    stat->RefineSteps = 0;
    stat->expansions = 0;
#if ( PRNTlevel >= 1 )
    # 打印函数调用和参数描述信息
    printf(".. parameters in sp_ienv():\n");
    # 打印 sp_ienv 函数返回的参数值，每个参数对应一个描述
    printf("\t 1: panel size \t %4d \n"
           "\t 2: relax      \t %4d \n"
           "\t 3: max. super \t %4d \n"
           "\t 4: row-dim 2D \t %4d \n"
           "\t 5: col-dim 2D \t %4d \n"
           "\t 6: fill ratio \t %4d \n",
       sp_ienv(1), sp_ienv(2), sp_ienv(3), 
       sp_ienv(4), sp_ienv(5), sp_ienv(6));
void
StatPrint(SuperLUStat_t *stat)
{
    double         *utime;  // 指向统计信息中计时数据的指针
    flops_t        *ops;    // 指向统计信息中浮点操作数的指针

    utime = stat->utime;    // 初始化 utime 指针，指向统计信息中的计时数据数组
    ops   = stat->ops;      // 初始化 ops 指针，指向统计信息中的浮点操作数数组
    printf("Factor time  = %8.5f\n", utime[FACT]);  // 打印因子化时间
    if ( utime[FACT] != 0.0 )
      printf("Factor flops = %e\tMflops = %8.2f\n", ops[FACT],
         ops[FACT]*1e-6/utime[FACT]);  // 如果因子化时间不为零，打印因子化操作的浮点操作数和MFlops

    printf("Solve time   = %8.4f\n", utime[SOLVE]);  // 打印求解时间
    if ( utime[SOLVE] != 0.0 )
      printf("Solve flops = %e\tMflops = %8.2f\n", ops[SOLVE],
         ops[SOLVE]*1e-6/utime[SOLVE]);  // 如果求解时间不为零，打印求解操作的浮点操作数和MFlops

    printf("Number of memory expansions: %d\n", stat->expansions);  // 打印内存扩展次数
}


void
StatFree(SuperLUStat_t *stat)
{
    SUPERLU_FREE(stat->panel_histo);  // 释放分解面板直方图的内存
    SUPERLU_FREE(stat->utime);        // 释放计时数据数组的内存
    SUPERLU_FREE(stat->ops);          // 释放浮点操作数数组的内存
}


flops_t
LUFactFlops(SuperLUStat_t *stat)
{
    return (stat->ops[FACT]);  // 返回因子化操作的浮点操作数
}

flops_t
LUSolveFlops(SuperLUStat_t *stat)
{
    return (stat->ops[SOLVE]);  // 返回求解操作的浮点操作数
}


/*! \brief Fills an integer array with a given value.
 */
void ifill(int *a, int alen, int ival)
{
    register int i;
    for (i = 0; i < alen; i++) a[i] = ival;  // 将整数数组 a 填充为指定值 ival
}


/*! \brief Get the statistics of the supernodes 
 */
#define NBUCKS 10

void super_stats(int nsuper, int *xsup)
{
    register int nsup1 = 0;
    int    i, isize, whichb, bl, bh;
    int    bucket[NBUCKS];
    int    max_sup_size = 0;

    for (i = 0; i <= nsuper; i++) {
    isize = xsup[i+1] - xsup[i];  // 计算超节点大小
    if ( isize == 1 ) nsup1++;    // 统计大小为1的超节点个数
    if ( max_sup_size < isize ) max_sup_size = isize;    // 记录最大超节点大小
    }

    printf("    Supernode statistics:\n\tno of super = %d\n", nsuper+1);  // 打印超节点统计信息
    printf("\tmax supernode size = %d\n", max_sup_size);  // 打印最大超节点大小
    printf("\tno of size 1 supernodes = %d\n", nsup1);  // 打印大小为1的超节点个数

    /* Histogram of the supernode sizes */
    ifill (bucket, NBUCKS, 0);  // 使用指定值填充直方图数组 bucket

    for (i = 0; i <= nsuper; i++) {
        isize = xsup[i+1] - xsup[i];  // 计算超节点大小
        whichb = (float) isize / max_sup_size * NBUCKS;  // 确定超节点大小所在直方图的位置
        if (whichb >= NBUCKS) whichb = NBUCKS - 1;  // 处理边界情况
        bucket[whichb]++;  // 更新对应直方图位置的计数
    }
    
    printf("\tHistogram of supernode sizes:\n");
    for (i = 0; i < NBUCKS; i++) {
        bl = (float) i * max_sup_size / NBUCKS;  // 计算当前直方图桶的下界
        bh = (float) (i+1) * max_sup_size / NBUCKS;  // 计算当前直方图桶的上界
        printf("\tsnode: %d-%d\t\t%d\n", bl+1, bh, bucket[i]);  // 打印直方图桶的统计信息
    }
}


float SpaSize(int n, int np, float sum_npw)
{
    return (sum_npw*8 + np*8 + n*4)/1024.;  // 计算稀疏矩阵大小的估计值
}

float DenseSize(int n, float sum_nw)
{
    return (sum_nw*8 + n*8)/1024.;;  // 计算稠密矩阵大小的估计值
}


/*! \brief Check whether repfnz[] == EMPTY after reset.
 */
void check_repfnz(int n, int w, int jcol, int *repfnz)
{
    int jj, k;

    for (jj = jcol; jj < jcol+w; jj++) 
    for (k = 0; k < n; k++)
        if ( repfnz[(jj-jcol)*n + k] != EMPTY ) {  // 检查重复非零元素数组是否为空
        fprintf(stderr, "col %d, repfnz_col[%d] = %d\n", jj,
            k, repfnz[(jj-jcol)*n + k]);  // 打印错误信息
        ABORT("check_repfnz");  // 终止程序运行
        }
}


/*! \brief Print a summary of the testing results. */
void
PrintSumm(char *type, int nfail, int nrun, int nerrs)
{
    if ( nfail > 0 )  // 如果失败次数大于零
    # 打印测试结果，显示测试失败的数量和总数，同时打印驱动程序类型
    printf("%3s driver: %d out of %d tests failed to pass the threshold\n",
           type, nfail, nrun);
    # 如果所有测试都通过了，打印所有测试均通过的消息，显示驱动程序类型和运行的测试总数
    else
    # 如果有错误消息记录，打印记录的错误消息数量
    printf("All tests for %3s driver passed the threshold (%6d tests run)\n", type, nrun);

    # 如果记录的错误消息数量大于 0，则打印错误消息数量
    if ( nerrs > 0 )
    printf("%6d error messages recorded\n", nerrs);
}

# 打印整数向量的函数，输出指定标签和向量内容
int print_int_vec(char *what, int n, int *vec)
{
    # 输出标签内容
    printf("%s\n", what);
    # 遍历并打印整数向量的索引和对应值
    for (int i = 0; i < n; ++i)
        printf("%d\t%d\n", i, vec[i]);
    # 返回操作成功的标志
    return 0;
}

# 打印格式化的整数向量，每行显示10个元素
int slu_PrintInt10(char *name, int len, int *x)
{
    register int i;
    
    # 输出名称和格式标签
    printf("%10s:", name);
    # 遍历向量并按照每行10个元素的格式打印
    for (i = 0; i < len; ++i)
    {
        if ( i % 10 == 0 ) printf("\n\t[%2d-%2d]", i, i + 9);
        printf("%6d", x[i]);
    }
    printf("\n");
    # 返回操作成功的标志
    return 0;
}

# 检查排列是否有效的函数，输出检查结果和相关信息
int check_perm(char *what, int n, int *perm)
{
    register int i;
    int *marker;
    # 为标记数组分配内存空间
    marker = int32Malloc(n);
    # 初始化标记数组为0
    for (i = 0; i < n; ++i) marker[i] = 0;

    # 遍历排列数组进行检查
    for (i = 0; i < n; ++i) {
        if ( marker[perm[i]] == 1 || perm[i] >= n ) {
            # 若发现重复元素或超出范围，则输出错误信息并中止程序
            printf("%s: Not a valid PERM[%d] = %d\n", what, i, perm[i]);
            ABORT("check_perm");
        } else {
            marker[perm[i]] = 1;
        }
    }

    # 释放动态分配的内存空间
    SUPERLU_FREE(marker);
    # 输出检查通过的信息和相关统计
    printf("check_perm: %s: n %d\n", what, n);
    # 返回操作成功的标志
    return 0;
}
```