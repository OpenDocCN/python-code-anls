# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slangs.c`

```
    for (i = 0; i < A->nrow; ++i) {
        sum = 0.;
        for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; j++)
            sum += fabs(Aval[j]);
        value = SUPERLU_MAX(value,sum);
    }
    
    } else if (strncmp(norm, "F", 1)==0 || strncmp(norm, "E", 1)==0) {
    /* Find normF(A). */
    /* This uses LAPACK's SLANGE to compute the Frobenius norm of A. */
    rwork = floatMalloc(A->ncol);
    value = slange_(norm, &A->nrow, &A->ncol, Aval, &Astore->colptr[0], rwork);
    SUPERLU_FREE(rwork);
    
    } else {
    /* Invalid norm type. */
    fprintf(stderr, "Illegal value of NORM: %s\n", norm);
    return (-1.);
    }
    
    return value;
}



注释：


    /* 将 A 的存储格式设为列优先存储 */
    Astore = A->Store;
    /* 获取 A 中非零元素的值 */
    Aval   = Astore->nzval;
    
    /* 如果 A 的行数或列数为 0，则直接返回 0 */
    if ( SUPERLU_MIN(A->nrow, A->ncol) == 0) {
    } else if (strncmp(norm, "M", 1)==0) {
    /* 计算 max(abs(A(i,j))) */
    for (j = 0; j < A->ncol; ++j)
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++)
        value = SUPERLU_MAX( value, fabs( Aval[i]) );
    
    } else if (strncmp(norm, "O", 1)==0 || *(unsigned char *)norm == '1') {
    /* 计算 norm1(A) */
    for (j = 0; j < A->ncol; ++j) {
        sum = 0.;
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) 
        sum += fabs(Aval[i]);
        value = SUPERLU_MAX(value,sum);
    }
    
    } else if (strncmp(norm, "I", 1)==0) {
    /* 计算 normI(A) */
    for (i = 0; i < A->nrow; ++i) {
        sum = 0.;
        for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; j++)
            sum += fabs(Aval[j]);
        value = SUPERLU_MAX(value,sum);
    }
    
    } else if (strncmp(norm, "F", 1)==0 || strncmp(norm, "E", 1)==0) {
    /* 计算 normF(A) */
    /* 使用 LAPACK 的 SLANGE 计算 A 的 Frobenius 范数 */
    rwork = floatMalloc(A->ncol);  // 分配工作空间
    value = slange_(norm, &A->nrow, &A->ncol, Aval, &Astore->colptr[0], rwork);  // 调用 SLANGE 计算范数
    SUPERLU_FREE(rwork);  // 释放工作空间
    
    } else {
    /* 若 norm 参数无效，则打印错误信息并返回 -1 */
    fprintf(stderr, "Illegal value of NORM: %s\n", norm);
    return (-1.);
    }
    
    return value;  // 返回计算得到的范数值
}
    # 分配内存以存储大小为 A->nrow 的浮点数数组，用于计算行的绝对值和
    if (!(rwork = (float *) SUPERLU_MALLOC(A->nrow * sizeof(float))))
        ABORT("SUPERLU_MALLOC fails for rwork.");
    
    # 将 rwork 数组的所有元素初始化为 0
    for (i = 0; i < A->nrow; ++i)
        rwork[i] = 0.;

    # 计算矩阵 A 的列的绝对值和，存储在 rwork 数组中的相应行索引位置
    for (j = 0; j < A->ncol; ++j)
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) {
            irow = Astore->rowind[i];
            rwork[irow] += fabs(Aval[i]);
        }
    
    # 找到 rwork 数组中的最大值，作为矩阵 A 的范数
    for (i = 0; i < A->nrow; ++i)
        value = SUPERLU_MAX(value, rwork[i]);
    
    # 释放 rwork 数组所占用的内存
    SUPERLU_FREE(rwork);

} else if (strncmp(norm, "F", 1) == 0 || strncmp(norm, "E", 1) == 0) {
    # 如果请求计算 Frobenius 范数或者 2-范数，目前未实现该功能
    ABORT("Not implemented.");
} else {
    # 如果请求的范数不合法，报错
    ABORT("Illegal norm specified.");
}

# 返回计算得到的矩阵 A 的范数值
return (value);
} /* slangs */
```