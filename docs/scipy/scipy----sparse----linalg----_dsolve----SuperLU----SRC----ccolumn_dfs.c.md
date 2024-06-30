# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ccolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ccolumn_dfs.c
 * \brief Performs a symbolic factorization
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
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

#include "slu_cdefs.h"

/*! \brief What type of supernodes we want */
#define T2_SUPER

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   CCOLUMN_DFS performs a symbolic factorization on column jcol, and
 *   decide the supernode boundary.
 *
 *   This routine does not use numeric values, but only use the RHS 
 *   row indices to start the dfs.
 *
 *   A supernode representative is the last column of a supernode.
 *   The nonzeros in U[*,j] are segments that end at supernodal
 *   representatives. The routine returns a list of such supernodal 
 *   representatives in topological order of the dfs that generates them.
 *   The location of the first nonzero in each such supernodal segment
 *   (supernodal entry location) is also returned.
 *
 * Local parameters
 * ================
 *   nseg: no of segments in current U[*,j]
 *   jsuper: jsuper=EMPTY if column j does not belong to the same
 *    supernode as j-1. Otherwise, jsuper=nsuper.
 *
 *   marker2: A-row --> A-row/col (0/1)
 *   repfnz: SuperA-col --> PA-row
 *   parent: SuperA-col --> SuperA-col
 *   xplore: SuperA-col --> index to L-structure
 *
 * Return value
 * ============
 *     0  success;
 *   > 0  number of bytes allocated when run out of space.
 * </pre>
 */
int CCOLUMN_DFS(int jcol)
{
    // 初始化变量 nseg，用于记录当前列 jcol 中的段数
    int nseg;

    // 变量 jsuper，如果列 j 不属于与 j-1 相同的超节点，则为 EMPTY；否则为当前超节点编号 nsuper
    int jsuper;

    // 标记数组 marker2，用于将 A 行索引映射到 A 行/列（0/1）
    int marker2;

    // repfnz 数组，将超级 A 列映射到 PA 行
    int repfnz;

    // parent 数组，将超级 A 列映射到超级 A 列
    int parent;

    // xplore 数组，将超级 A 列映射到 L 结构的索引
    int xplore;

    // 函数返回值，0 表示成功；> 0 表示空间耗尽时分配的字节数
    return 0;
}
    ccolumn_dfs(
       const int  m,         /* in - number of rows in the matrix */
       const int  jcol,      /* in - current column index being processed */
       int        *perm_r,   /* in - permutation vector for rows */
       int        *nseg,     /* modified - number of segments in the matrix */
       int        *lsub_col, /* in - defines the RHS vector to start the dfs */
       int        *segrep,   /* modified - segment representation */
       int        *repfnz,   /* modified - first non-zero in a segment */
       int_t      *xprune,   /* modified - prune array */
       int        *marker,   /* modified - marker array for visited nodes */
       int        *parent,   /* working array - parent nodes */
       int_t      *xplore,   /* working array - nodes to explore */
       GlobalLU_t *Glu       /* modified - data structure for LU decomposition */
       )
{
    int     jcolp1, jcolm1, jsuper, nsuper;
    int     krep, krow, kmark, kperm;
    int     *marker2;           /* Used for small panel LU */
    int        fsupc;        /* First column of a snode */
    int     myfnz;        /* First nonz column of a U-segment */
    int        chperm, chmark, chrep, kchild;
    int_t   xdfs, maxdfs, nextl, k;
    int     kpar, oldrep;
    int_t   jptr, jm1ptr;
    int_t   ito, ifrom, istop;    /* Used to compress row subscripts */
    int     *xsup, *supno;
    int_t   *lsub, *xlsub;
    int_t   nzlmax, mem_error;
    int     maxsuper;
    
    xsup    = Glu->xsup;         /* Pointer to supernode start array */
    supno   = Glu->supno;        /* Supernode number array */
    lsub    = Glu->lsub;         /* Column-wise row indices */
    xlsub   = Glu->xlsub;        /* Starting position of each column in lsub */
    nzlmax  = Glu->nzlmax;       /* Maximum capacity of lsub */

    maxsuper = sp_ienv(3);       /* Maximum number of columns in a supernode */
    jcolp1  = jcol + 1;          /* Next column index */
    jcolm1  = jcol - 1;          /* Previous column index */
    nsuper  = supno[jcol];       /* Supernode number of current column */
    jsuper  = nsuper;            /* Temporary variable for supernode number */
    nextl   = xlsub[jcol];       /* Starting position of current column in lsub */
    marker2 = &marker[2*m];      /* Marker array starting position for small panel LU */

    /* For each nonzero in A[*,jcol] do dfs */
    for (k = 0; lsub_col[k] != EMPTY; k++) {
        krow = lsub_col[k];     /* Current row index */
        lsub_col[k] = EMPTY;    /* Clear current row index in lsub_col */
        kmark = marker2[krow];  /* Marker for current row */

        /* krow was visited before, go to the next nonz */
        if ( kmark == jcol ) continue; 

        /* For each unmarked nbr krow of jcol
         *    krow is in L: place it in structure of L[*,jcol]
         */
        marker2[krow] = jcol;   /* Mark krow as visited by jcol */
    } /* else */

} /* for each nonzero ... */

/* Check to see if j belongs in the same supernode as j-1 */
if ( jcol == 0 ) { /* Do nothing for column 0 */
    nsuper = supno[0] = 0;     /* Set supernode number for column 0 */
} else {
    fsupc = xsup[nsuper];       /* First supernodal column of current supernode */
    jptr = xlsub[jcol];         /* Starting position of current column */
    jm1ptr = xlsub[jcolm1];     /* Starting position of previous column */

#ifdef T2_SUPER
    if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;    /* Condition for supernode merge */
#endif

    /* Make sure the number of columns in a supernode doesn't
       exceed threshold. */
    if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;          /* Check if new supernode should start */

    /* If jcol starts a new supernode, reclaim storage space in
     * lsub from the previous supernode. Note we only store
     * the subscript set of the first and last columns of
     * a supernode. (first for num values, last for pruning)
     */
    if ( jsuper == EMPTY ) {    /* starts a new supernode */
        if ( (fsupc < jcolm1-1) ) {    /* >= 3 columns in nsuper */
#ifdef CHK_COMPRESS
        printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);

        printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#endif
    } /* if jsuper == EMPTY */

} /* if jcol == 0 else */
#endif
            ito = xlsub[fsupc+1];   /* Retrieve xlsub[fsupc+1] and assign to ito */
        xlsub[jcolm1] = ito;         /* Set xlsub[jcolm1] to ito */
        istop = ito + jptr - jm1ptr; /* Calculate istop using ito, jptr, and jm1ptr */
        xprune[jcolm1] = istop;      /* Initialize xprune[jcolm1] with istop */

        /* Copy entries from lsub[ifrom] to lsub[ito] */
        xlsub[jcol] = istop;         /* Set xlsub[jcol] to istop */
        for (ifrom = jm1ptr; ifrom < nextl; ++ifrom, ++ito)
            lsub[ito] = lsub[ifrom]; /* Copy lsub[ifrom] to lsub[ito] */
        nextl = ito;                 /* Update nextl to ito */

        /* Increment supernode count and assign supernode number */
        nsuper++;                    /* Increment nsuper */
        supno[jcol] = nsuper;        /* Assign nsuper to supno[jcol] */

    } /* if a new supernode */

    }    /* else: jcol > 0 */ 
    
    /* Tidy up the pointers before exit */
    xsup[nsuper+1] = jcolp1;         /* Set xsup[nsuper+1] to jcolp1 */
    supno[jcolp1]  = nsuper;         /* Assign nsuper to supno[jcolp1] */
    xprune[jcol]   = nextl;          /* Initialize xprune[jcol] with nextl */
    xlsub[jcolp1]  = nextl;          /* Set xlsub[jcolp1] to nextl */

    return 0;                        /* Return 0 to indicate successful execution */
}
```