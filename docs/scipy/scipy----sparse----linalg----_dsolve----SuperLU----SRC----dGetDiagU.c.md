# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dGetDiagU.c`

```
/*! @file dGetDiagU.c
 * \brief Extracts main diagonal of matrix
 *
 * <pre> 
 * -- Auxiliary routine in SuperLU (version 2.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * Xiaoye S. Li
 * September 11, 2003
 *
 *  Purpose
 * =======
 *
 * GetDiagU extracts the main diagonal of matrix U of the LU factorization.
 *  
 * Arguments
 * =========
 *
 * L      (input) SuperMatrix*
 *        The factor L from the factorization Pr*A*Pc=L*U as computed by
 *        dgstrf(). Use compressed row subscripts storage for supernodes,
 *        i.e., L has types: Stype = SLU_SC, Dtype = SLU_D, Mtype = SLU_TRLU.
 *
 * diagU  (output) double*, dimension (n)
 *        The main diagonal of matrix U.
 *
 * Note
 * ====
 * The diagonal blocks of the L and U matrices are stored in the L
 * data structures.
 * </pre> 
*/
#include "slu_ddefs.h"

void dGetDiagU(SuperMatrix *L, double *diagU)
{
    int_t i, k, nsupers;
    int_t fsupc, nsupr, nsupc, luptr;
    double *dblock, *Lval;
    SCformat *Lstore;

    // 获取 L 矩阵的存储格式
    Lstore = L->Store;
    // 获取 L 矩阵中非零元素的数组
    Lval = Lstore->nzval;
    // 获取 L 矩阵的超节点数目
    nsupers = Lstore->nsuper + 1;

    // 遍历每个超节点
    for (k = 0; k < nsupers; ++k) {
      // 获取当前超节点的第一个列索引
      fsupc = L_FST_SUPC(k);
      // 计算当前超节点包含的列数
      nsupc = L_FST_SUPC(k+1) - fsupc;
      // 获取当前超节点包含的行数
      nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);
      // 获取当前超节点第一个非零元素在 Lval 中的索引
      luptr = L_NZ_START(fsupc);

      // 指向 diagU 中当前超节点对应的对角块起始位置
      dblock = &diagU[fsupc];
      // 遍历当前超节点的列
      for (i = 0; i < nsupc; ++i) {
        // 将 Lval 中的值复制到 diagU 的对应位置
        dblock[i] = Lval[luptr];
        // 移动到下一个对角块的首元素位置
        luptr += nsupr + 1;
      }
    }
}
```