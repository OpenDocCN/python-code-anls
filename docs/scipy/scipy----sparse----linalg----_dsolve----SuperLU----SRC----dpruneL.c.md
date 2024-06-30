# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dpruneL.c`

```
    /*
     * 如果该超节点未被修剪并且在行 L[pivrow,i] 中具有非零元素
     */
    # 如果当前的 supno[irep] 不等于 jsupno，则执行以下操作
    if ( supno[irep] != jsupno ) {
        # 如果 xprune[irep] 大于等于 xlsub[irep1]，则执行以下操作
        if ( xprune[irep] >= xlsub[irep1] ) {
            # 设置 kmin 为 xlsub[irep]，设置 kmax 为 xlsub[irep1] - 1
            kmin = xlsub[irep];
            kmax = xlsub[irep1] - 1;
            # 遍历 kmin 到 kmax 的范围
            for (krow = kmin; krow <= kmax; krow++) 
                # 如果 lsub[krow] 等于 pivrow，则设置 do_prune 为 TRUE，并退出循环
                if ( lsub[krow] == pivrow ) {
                    do_prune = TRUE;
                    break;
                }
        }

        # 如果 do_prune 为 TRUE，则执行以下操作
        if ( do_prune ) {

            /* 执行类似快速排序的分区操作
             * movnum=TRUE 表示需要交换数值
             */
            movnum = FALSE;
            # 如果 irep 等于 xsup[supno[irep]]，表示 Snode 的大小为 1
            if ( irep == xsup[supno[irep]] ) /* Snode of size 1 */
                movnum = TRUE;

            # 当 kmin 小于等于 kmax 时执行以下循环
            while ( kmin <= kmax ) {

                # 如果 perm_r[lsub[kmax]] 等于 EMPTY，则减小 kmax
                if ( perm_r[lsub[kmax]] == EMPTY ) 
                    kmax--;
                # 否则如果 perm_r[lsub[kmin]] 不等于 EMPTY，则增加 kmin
                else if ( perm_r[lsub[kmin]] != EMPTY )
                    kmin++;
                else { /* kmin 在 pivrow 下方（尚未枢轴化），kmax 在 pivrow 上方：
                            * 交换这两个下标
                    */
                    ktemp = lsub[kmin];
                    lsub[kmin] = lsub[kmax];
                    lsub[kmax] = ktemp;

                    /* 如果超节点只有一列，则只保留一个子脚本。对于任何执行的子脚本交换，
                     * 数值上的类似交换也必须执行。
                     */
                    if ( movnum ) {
                        minloc = xlusup[irep] + (kmin - xlsub[irep]);
                        maxloc = xlusup[irep] + (kmax - xlsub[irep]);
                        utemp = lusup[minloc];
                        lusup[minloc] = lusup[maxloc];
                        lusup[maxloc] = utemp;
                    }

                    kmin++;
                    kmax--;
                }
            }

        } /* while */

        xprune[irep] = kmin;    /* Pruning */
    }


这段代码是一段复杂的算法逻辑，主要用于进行数组的排序和操作。
#ifdef CHK_PRUNE
    // 如果定义了 CHK_PRUNE 宏，则打印调试信息，显示 dpruneL() 后使用的列号、xprune 数组索引、以及 kmin 的值
    printf("    After dpruneL(), using col %d:  xprune[%d] = %d\n", 
            jcol, irep, kmin);
#endif
        } /* if do_prune */
    } /* if */
    // 结束对每个 U 段的循环
    } /* for each U-segment... */
}
```