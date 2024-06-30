# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cpruneL.c`

```
    /*
     * If it has not been pruned & it has a nonz in row L[pivrow,i]
     */
    /* 如果未被修剪并且在行 L[pivrow,i] 中有非零元素 */
    # 如果当前迭代的 supernode 号码不等于 jsupno
    if ( supno[irep] != jsupno ) {
        # 如果 xprune[irep] 大于或等于 xlsub[irep1]
        if ( xprune[irep] >= xlsub[irep1] ) {
            # 设置 kmin 为 xlsub[irep]，设置 kmax 为 xlsub[irep1] - 1
            kmin = xlsub[irep];
            kmax = xlsub[irep1] - 1;
            # 遍历 kmin 到 kmax 之间的每一个 krow
            for (krow = kmin; krow <= kmax; krow++) 
                # 如果 lsub[krow] 等于 pivrow
                if ( lsub[krow] == pivrow ) {
                    # 设置 do_prune 为 TRUE，并且跳出循环
                    do_prune = TRUE;
                    break;
                }
        }
        
        # 如果 do_prune 为真
        if ( do_prune ) {

            /* 进行类似快速排序的分区
             * movnum=TRUE 表示需要交换 num 值
             */
            movnum = FALSE;
            # 如果 irep 等于 xsup[supno[irep]]，即 Snode 大小为 1
            if ( irep == xsup[supno[irep]] ) /* Snode of size 1 */
                movnum = TRUE;

            # 当 kmin 小于等于 kmax 时执行循环
            while ( kmin <= kmax ) {

                # 如果 perm_r[lsub[kmax]] 等于 EMPTY
                if ( perm_r[lsub[kmax]] == EMPTY ) 
                    kmax--;
                # 否则如果 perm_r[lsub[kmin]] 不等于 EMPTY
                else if ( perm_r[lsub[kmin]] != EMPTY )
                    kmin++;
                else { /* kmin 在 pivrow 之下（还未被 pivot），而 kmax 在 pivrow 之上：交换这两个下标 */
                    # 交换 lsub[kmin] 和 lsub[kmax]
                    ktemp = lsub[kmin];
                    lsub[kmin] = lsub[kmax];
                    lsub[kmax] = ktemp;

                    /* 如果 supernode 只有一列，则保留一组下标。对于任何执行的下标交换，
                     * 必须对数值也进行类似的交换。
                     */
                    if ( movnum ) {
                        # 计算 minloc 和 maxloc
                        minloc = xlusup[irep] + (kmin - xlsub[irep]);
                        maxloc = xlusup[irep] + (kmax - xlsub[irep]);
                        # 交换 lusup[minloc] 和 lusup[maxloc]
                        utemp = lusup[minloc];
                        lusup[minloc] = lusup[maxloc];
                        lusup[maxloc] = utemp;
                    }

                    # kmin 和 kmax 各自增加和减少
                    kmin++;
                    kmax--;
                }

            }

        } /* while */

        # 设置 xprune[irep] 为 kmin，表示修剪操作完成
        xprune[irep] = kmin;    /* Pruning */
#ifdef CHK_PRUNE
    // 如果定义了 CHK_PRUNE 宏，则执行下面的代码块
    printf("    After cpruneL(),using col %d:  xprune[%d] = %d\n", 
            jcol, irep, kmin);
    // 打印调试信息，显示在执行 cpruneL() 后，使用列号 jcol，xprune 数组中索引 irep 的值为 kmin
#endif
        } /* if do_prune */
    // 结束 if do_prune 的条件判断块

    } /* if */
    // 结束外层 if 的条件判断块

    } /* for each U-segment... */
    // 结束对每个 U-段落的循环遍历
}
// 结束函数定义
```