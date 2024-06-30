# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\scopy_to_ucol.c`

```
/*
 * Gather from SPA dense[*] to global ucol[*].
 */
int
scopy_to_ucol(
          int        jcol,      /* in: 待处理的列索引 */
          int        nseg,      /* in: 段数 */
          int        *segrep,   /* in: 段的表示 */
          int        *repfnz,   /* in: 非零重复 */
          int        *perm_r,   /* in: 行置换 */
          float     *dense,     /* modified - reset to zero on return: 密集数组，返回时重置为零 */
          GlobalLU_t *Glu       /* modified: 全局LU分解数据结构 */
          )
{
    int ksub, krep, ksupno;   // 定义整型变量
    int i, k, kfnz, segsze;   // 定义整型变量
    int fsupc, irow, jsupno;  // 定义整型变量
    int_t isub, nextu, new_next, mem_error;  // 定义长整型变量
    int       *xsup, *supno;  // 定义整型指针
    int_t     *lsub, *xlsub;  // 定义长整型指针
    float    *ucol;           // 定义浮点型指针
    int_t     *usub, *xusub;  // 定义长整型指针
    int_t       nzumax;       // 定义长整型变量
    float zero = 0.0;         // 定义浮点数变量，值为零

    xsup    = Glu->xsup;      // 获取全局LU结构中的xsup数组指针
    supno   = Glu->supno;     // 获取全局LU结构中的supno数组指针
    lsub    = Glu->lsub;      // 获取全局LU结构中的lsub数组指针
    xlsub   = Glu->xlsub;     // 获取全局LU结构中的xlsub数组指针
    ucol    = (float *) Glu->ucol;  // 获取全局LU结构中的ucol数组指针并转换为浮点型指针
    usub    = Glu->usub;      // 获取全局LU结构中的usub数组指针
    xusub   = Glu->xusub;     // 获取全局LU结构中的xusub数组指针
    nzumax  = Glu->nzumax;    // 获取全局LU结构中的nzumax变量
    
    jsupno = supno[jcol];     // 获取列索引jcol对应的supno值
    nextu  = xusub[jcol];     // 获取列索引jcol对应的xusub值
    k = nseg - 1;             // 初始化k为段数减一
    for (ksub = 0; ksub < nseg; ksub++) {
        krep = segrep[k--];   // 获取当前段的segrep值，并递减k
        ksupno = supno[krep]; // 获取当前段的supno值
        
        // 这里需要继续注释的代码超出了最大字符数限制，请参照上文的格式继续注释。
    if ( ksupno != jsupno ) { /* 如果 ksupno 不等于 jsupno，需要处理到 ucol[] */

        kfnz = repfnz[krep];
        if ( kfnz != EMPTY ) {    /* 如果 kfnz 不为 EMPTY，表示非零 U 段 */

            fsupc = xsup[ksupno];
            isub = xlsub[fsupc] + kfnz - fsupc;
            segsze = krep - kfnz + 1;

            new_next = nextu + segsze;
            while ( new_next > nzumax ) {   /* 扩展 UCOL 和 USUB 内存直到足够 */
                mem_error = sLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu);
                if (mem_error) return (mem_error);
                ucol = (float *) Glu->ucol;
                mem_error = sLUMemXpand(jcol, nextu, USUB, &nzumax, Glu);
                if (mem_error) return (mem_error);
                usub = Glu->usub;
                lsub = Glu->lsub;
            }
            
            for (i = 0; i < segsze; i++) {   /* 将非零 U 段的数据复制到 UCOL 和 USUB 中 */
                irow = lsub[isub];
                usub[nextu] = perm_r[irow];
                ucol[nextu] = dense[irow];
                dense[irow] = zero;
                nextu++;
                isub++;
            } 

        }

    }

    } /* 结束每个段的处理 */

    xusub[jcol + 1] = nextu;      /* 完成 U[*,jcol] 的收尾处理 */
    return 0;
}


注释：


# 这行代码关闭了一个代码块，通常与某个函数或者控制流语句的开始相对应
```