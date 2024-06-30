# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dlangs.c`

```
    for (i = 0; i < A->nrow; ++i) {
        sum = 0.;
        for (j = Astore->rowind[i]; j < Astore->rowind[i+1]; j++) 
        sum += fabs(Aval[j]);
        value = SUPERLU_MAX(value,sum);
    }
    
    } else if (strncmp(norm, "F", 1)==0 || strncmp(norm, "E", 1)==0) {
    /* Find normF(A). */
    value = 0.;
    for (j = 0; j < A->ncol; ++j) {
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) {
            value += Aval[i] * Aval[i];
        }
    }
    value = sqrt(value);
    }
    
    return (value);
}


注释：


    /* 获取稀疏矩阵 A 的存储格式 */
    Astore = A->Store;
    /* 获取矩阵的非零元素数组 */
    Aval   = Astore->nzval;
    
    /* 如果矩阵行数或列数为 0，则直接返回 0 */
    if ( SUPERLU_MIN(A->nrow, A->ncol) == 0) {
    } else if (strncmp(norm, "M", 1)==0) {
    /* 计算 max(abs(A(i,j)))。 */
    for (j = 0; j < A->ncol; ++j)
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++)
        value = SUPERLU_MAX( value, fabs( Aval[i]) );
    
    } else if (strncmp(norm, "O", 1)==0 || *(unsigned char *)norm == '1') {
    /* 计算 norm1(A)。 */
    for (j = 0; j < A->ncol; ++j) {
        sum = 0.;
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) 
        sum += fabs(Aval[i]);
        value = SUPERLU_MAX(value,sum);
    }
    
    } else if (strncmp(norm, "I", 1)==0) {
    /* 计算 normI(A)。 */
    for (i = 0; i < A->nrow; ++i) {
        sum = 0.;
        for (j = Astore->rowind[i]; j < Astore->rowind[i+1]; j++) 
        sum += fabs(Aval[j]);
        value = SUPERLU_MAX(value,sum);
    }
    
    } else if (strncmp(norm, "F", 1)==0 || strncmp(norm, "E", 1)==0) {
    /* 计算 normF(A)。 */
    value = 0.;
    for (j = 0; j < A->ncol; ++j) {
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) {
            value += Aval[i] * Aval[i];
        }
    }
    value = sqrt(value);
    }
    
    /* 返回计算得到的矩阵范数值 */
    return (value);
}
    // 分配 rwork 数组来存储行和列之间的工作空间
    if ( !(rwork = (double *) SUPERLU_MALLOC(A->nrow * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for rwork.");
    
    // 初始化 rwork 数组中的所有元素为 0
    for (i = 0; i < A->nrow; ++i) rwork[i] = 0.;
    
    // 计算矩阵 A 中每列的绝对值之和，并存储在 rwork 数组中对应的行索引位置
    for (j = 0; j < A->ncol; ++j)
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) {
            irow = Astore->rowind[i];
            rwork[irow] += fabs(Aval[i]);
        }
    
    // 计算 rwork 数组中的最大值，作为矩阵 A 的无穷范数
    for (i = 0; i < A->nrow; ++i)
        value = SUPERLU_MAX(value, rwork[i]);
    
    // 释放 rwork 数组占用的内存空间
    SUPERLU_FREE (rwork);
    
} else if (strncmp(norm, "F", 1)==0 || strncmp(norm, "E", 1)==0) {
    // 如果指定了 F 范数或 E 范数，则输出错误信息并终止程序
    ABORT("Not implemented.");
} else {
    // 如果指定的范数不合法，则输出错误信息并终止程序
    ABORT("Illegal norm specified.");
}

// 返回计算得到的矩阵 A 的范数值
return (value);
# 结束对 "dlangs" 的代码块的注释
} /* dlangs */
```