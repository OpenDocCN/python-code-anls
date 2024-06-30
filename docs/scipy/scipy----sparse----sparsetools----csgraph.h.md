# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\csgraph.h`

```
#ifndef __CSGRAPH_H__
#define __CSGRAPH_H__

#include <vector>

/*
 * Determine connected components of a compressed sparse graph.
 * Note:
 *   Output array flag must be preallocated
 */
// 压缩稀疏图的连通分量确定函数
template <class I>
// cs_graph_components 函数定义，接受节点数 n_nod、行指针数组 Ap、列索引数组 Aj 和输出标记数组 flag
I cs_graph_components(const I n_nod,
              const I Ap[],
              const I Aj[],
                    I flag[])
{
  // pos 是一个工作数组：存储待处理的节点（行）列表
  std::vector<I> pos(n_nod, 0);
  I n_comp = 0;
  I n_tot, n_pos, n_pos_new, n_pos0, n_new, n_stop;
  I icomp, ii, ir, ic;

  n_stop = n_nod;
  // 初始化标记数组 flag，同时减少 n_stop 记录未连接的节点数
  for (ir = 0; ir < n_nod; ir++) {
    flag[ir] = -1;
    if (Ap[ir+1] == Ap[ir]) {
      n_stop--;
      flag[ir] = -2;
    }
  }

  n_tot = 0;
  // 遍历每个连通分量
  for (icomp = 0; icomp < n_nod; icomp++) {
    // 找到一个种子节点
    ii = 0;
    while ((flag[ii] >= 0) || (flag[ii] == -2)) {
      ii++;
      if (ii >= n_nod) {
        /* Sanity check, if this happens, the graph is corrupted. */
        // 如果出现这种情况，图结构可能已损坏，返回错误代码 -1
        return -1;
      }
    }

    flag[ii] = icomp;
    pos[0] = ii;
    n_pos0 = 0;
    n_pos_new = n_pos = 1;

    // 扩展当前连通分量直到无法继续
    for (ii = 0; ii < n_nod; ii++) {
      n_new = 0;
      for (ir = n_pos0; ir < n_pos; ir++) {
        for (ic = Ap[pos[ir]]; ic < Ap[pos[ir]+1]; ic++) {
          if (flag[Aj[ic]] == -1) {
            flag[Aj[ic]] = icomp;
            pos[n_pos_new] = Aj[ic];
            n_pos_new++;
            n_new++;
          }
        }
      }
      n_pos0 = n_pos;
      n_pos = n_pos_new;
      if (n_new == 0) break;
    }
    n_tot += n_pos;

    if (n_tot == n_stop) {
      n_comp = icomp + 1;
      break;
    }
  }

  return n_comp;
}

#endif
```