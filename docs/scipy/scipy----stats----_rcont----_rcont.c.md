# `D:\src\scipysrc\scipy\scipy\stats\_rcont\_rcont.c`

```
/**
  This file contains two algorithms written in C to generate random two-way
  tables. The algorithms rcont1 and rcont2 originate from Boyett and Patefield,
  respectively. For more information, see the docs for each function.

  If you wonder about the spelling, rcont is short for random contingency
  table. Random contingency table is a bit of a misnomer. The tables generated
  by these algorithms have no contingency/association between the two
  variables.

  Author: Hans Dembinski
*/
#include <math.h>
#include <stdbool.h>

#include "logfactorial.h"
#include "_rcont.h"

// helper function to access a 1D array like a C-style 2D array
/**
  Access a 1D array as if it were a 2D array.

  @param m Pointer to the 1D array (matrix)
  @param nr Number of rows in the virtual 2D array
  @param nc Number of columns in the virtual 2D array
  @param ir Row index
  @param ic Column index
  @return Pointer to the element at the specified row and column
*/
tab_t *ptr(tab_t *m, int nr, int nc, int ir, int ic)
{
  return m + nc * ir + ic;
}

/*
  Call this once to initialize workspace for rcont1.

  The work space must have size N, where N is the total number of entries.
*/
/**
  Initialize workspace for the rcont1 function.

  @param work Pointer to the workspace array
  @param nc Number of columns (categories)
  @param c Array of marginal totals for columns
*/
void rcont1_init(tab_t *work, int nc, const tab_t *c)
{
  for (int i = 0; i < nc; ++i)
  {
    tab_t ci = c[i];
    while (ci--)
      *work++ = i;
  }
}

/*
  Generate random two-way table with given marginal totals.

  Boyett's shuffling algorithm adapted from AS 144 Appl. Statist. (1979)
  329-332. The algorithm has O(N) complexity in space and time for
  a table with N entries in total. The algorithm performs poorly for large N,
  but is insensitive to the number K of table cells.

  This function uses a work space of size N which must be preallocated and
  initialized with rcont1_init.
*/
/**
  Generate a random two-way table with given marginal totals using Boyett's algorithm.

  @param table Pointer to the output table (2D array)
  @param nr Number of rows (categories)
  @param r Array of marginal totals for rows
  @param nc Number of columns (categories)
  @param c Array of marginal totals for columns
  @param ntot Total number of entries in the table
  @param work Preallocated workspace array initialized with rcont1_init
  @param rstate Pointer to the random number generator state
*/
void rcont1(tab_t *table, int nr, const tab_t *r, int nc, const tab_t *c,
            const tab_t ntot, tab_t *work, bitgen_t *rstate)
{
  // nothing to do
  if (ntot == 0)
    return;

  // shuffle work with Knuth's algorithm
  for (tab_t i = ntot - 1; i > 0; --i)
  {
    tab_t j = random_interval(rstate, i);
    tab_t tmp = work[j];
    work[j] = work[i];
    work[i] = tmp;
  }

  // clear table
  for (int i = 0, nrc = (nr * nc); i < nrc; ++i)
    table[i] = 0;

  // fill table
  for (int ir = 0; ir < nr; ++ir)
  {
    tab_t ri = r[ir];
    while (ri--)
      *ptr(table, nr, nc, ir, *work++) += 1;
  }
}

/*
  Generate random two-way table with given marginal totals.

  Patefield's algorithm adapted from AS 159 Appl. Statist. (1981) 91-97. This
  algorithm has O(K log(N)) complexity in time for a table with K cells and N
  entries in total. It requires only a small constant stack space.

  The original FORTRAN code was hand-translated to C. Changes to the original:

  - The computation of a look-up table of log-factorials was replaced with
    logfactorial function from numpy (which does something similar).
  - The original implementation allocated a column vector JWORK, but this is
    not necessary. The vector can be folded into the last column of the output
    table.
  - The function uses Numpy's random number generator and distribution library.
  - The algorithm now handles zero entries in row or column vector. When a
    zero is encountered, the output table is filled with zeros along that row
*/
void rcont2(tab_t *table, int nr, const tab_t *r, int nc, const tab_t *c,
            bitgen_t *rstate)
{
  // Implementation details omitted for brevity in this example
}
    # 如果条件 `n <= 1` 成立，则返回 `n`
    if n <= 1:
        return n
    # 初始化 `n` 为 `result`，`a` 为 `0`，`b` 为 `1`
    result, a, b = 0, 0, 1
    # 对于每个整数 `i` 在范围从 `1` 到 `n` (不包括 `n`) 的迭代中执行以下操作
    for i in range(1, n):
        # 设置 `result` 等于 `a` 和 `b` 的和
        result = a + b
        # 将 `a` 设为 `b`
        a = b
        # 将 `b` 设为 `result`
        b = result
    # 返回 `result`
    return result
/*
void rcont2(tab_t *table, int nr, const tab_t *r, int nc, const tab_t *c,
            const tab_t ntot, bitgen_t *rstate)
{
  // 如果总数为零，无需进行任何操作，直接返回
  if (ntot == 0)
    return;

  // 将jwork指向table的最后一行，用于存储中间计算结果
  tab_t *jwork = ptr(table, nr, nc, nr - 1, 0);

  // 将jwork的前nc-1个元素赋值为数组c的对应元素，最后一个元素不使用
  for (int i = 0; i < nc - 1; ++i)
  {
    jwork[i] = c[i];
  }

  tab_t jc = ntot;
  tab_t ib = 0;

  // 对于table的最后一行不需要随机化，所以遍历到nr-1
  for (int l = 0; l < nr - 1; ++l)
  {
    tab_t ia = r[l]; // 第一个术语

    // 如果第一个术语为0，将table第l行的所有元素置为0，并继续下一次循环
    if (ia == 0)
    {
      for (int i = 0; i < nc; ++i)
        *ptr(table, nr, nc, l, i) = 0;
      continue;
    }

    tab_t ic = jc; // 第二个术语
    jc -= r[l];

    // 对于table的最后一列不需要随机化，所以遍历到nc-1
    for (int m = 0; m < nc - 1; ++m)
    {
      const tab_t id = jwork[m]; // 第三个术语
      const tab_t ie = ic;       // 第四个术语
      ic -= id;
      ib = ie - ia;

      // 如果数组c的第m个元素为0，将table的第m列所有元素置为0，并继续下一次循环
      if (c[m] == 0)
      {
        for (int i = 0; i < nr; ++i)
          *ptr(table, nr, nc, i, m) = 0;
        continue;
      }

      const tab_t ii = ib - id; // 第五个术语

      // 如果ie为0，将table的第l行的所有元素（从m到nc-2）置为0，并跳出内层循环
      if (ie == 0)
      {
        for (int j = m; j < nc - 1; ++j)
          *ptr(table, nr, nc, l, j) = 0;
        ia = 0;
        break;
      }

      // 计算随机数z，并基于概率判断是否跳转到l160
      double z = random_standard_uniform(rstate);
      tab_t nlm;
    l131:
      nlm = (tab_t)floor((double)(ia * id) / ie + 0.5);

      // 计算概率x，用于决定是否跳转到l160
      double x = exp(
          logfactorial(ia) + logfactorial(ib) + logfactorial(ic) + logfactorial(id) - logfactorial(ie) - logfactorial(nlm) - logfactorial(id - nlm) - logfactorial(ia - nlm) - logfactorial(ii + nlm));

      // 如果x大于等于z，则跳转到l160
      if (x >= z)
        goto l160;

      // 更新sumprb和y，用于后续概率计算
      double sumprb = x;
      double y = x;
      tab_t nll = nlm;
      bool lsp = false;
      bool lsm = false;
      tab_t j;

    l140:
      // 更新j值并计算新的概率x
      j = (id - nlm) * (ia - nlm);
      if (j == 0)
        goto l156;
      nlm += 1;
      x *= (double)j / (nlm * (ii + nlm));
      sumprb += x;

      // 如果sumprb大于等于z，则跳转到l160
      if (sumprb >= z)
        goto l160;

    l150:
      if (lsm)
        goto l155;

      // 更新j值并计算新的概率y
      j = nll * (ii + nll);
      if (j == 0)
        goto l154;
      nll -= 1;
      y *= (double)j / ((id - nll) * (ia - nll));
      sumprb += y;

      // 如果sumprb大于等于z，则跳转到l159
      if (sumprb >= z)
        goto l159;

      // 如果lsp为false，则跳转到l140，否则跳转到l150
      if (!lsp)
        goto l140;

      goto l150;

    l154:
      lsm = true;

    l155:
      if (!lsp)
        goto l140;

      // 更新z值，并跳转到l131
      z = random_standard_uniform(rstate) * sumprb;
      goto l131;

    l156:
      lsp = true;
      goto l150;

    l159:
      nlm = nll;

    // 将计算结果nlm存储到table的位置(l, m)，并更新ia和jwork[m]
    l160:
      *ptr(table, nr, nc, l, m) = nlm;
      ia -= nlm;
      jwork[m] -= nlm;
    }

    // 计算table的最后一列的值
    *ptr(table, nr, nc, l, nc - 1) = ia;
  }

  // 计算table的最后一行的值，因为jwork已经是table的最后一行，所以无需额外操作到nc - 2
  *ptr(table, nr, nc, nr - 1, nc - 1) = ib - *ptr(table, nr, nc, nr - 1, nc - 2);
}
*/
```