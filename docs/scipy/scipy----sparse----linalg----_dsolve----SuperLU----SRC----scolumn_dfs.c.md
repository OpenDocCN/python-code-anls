# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\scolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file scolumn_dfs.c
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

#include "slu_sdefs.h"

/*! \brief What type of supernodes we want */
#define T2_SUPER

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   SCOLUMN_DFS performs a symbolic factorization on column jcol, and
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
int SCOLUMN_DFS(
    /*! \brief The column index to perform symbolic factorization on */
    int jcol,
    /*! \brief Output - number of bytes allocated when run out of space */
    int *nseg,
    /*! \brief Output - indicates if column j belongs to the same supernode as j-1 */
    int *jsuper,
    /*! \brief Input - marker array for A */
    int *marker2,
    /*! \brief Output - map from supernodal column to the first nonzero row */
    int *repfnz,
    /*! \brief Output - parent supernodal column */
    int *parent,
    /*! \brief Output - index to L-structure */
    int *xplore
) {
       const int  m,         /* in - number of rows in the matrix */
       const int  jcol,      /* in - current column index */
       int        *perm_r,   /* in - row permutation array */
       int        *nseg,     /* modified - updated with new segments */
       int        *lsub_col, /* in - defines the RHS vector for DFS */
       int        *segrep,   /* modified - updated with new segments */
       int        *repfnz,   /* modified - updated */
       int_t      *xprune,   /* modified - updated */
       int        *marker,   /* modified - marker array */
       int        *parent,     /* working array */
       int_t      *xplore,   /* working array */
       GlobalLU_t *Glu       /* modified - global LU data structure */
       )
{
    int     jcolp1, jcolm1, jsuper, nsuper;
    int     krep, krow, kmark, kperm;
    int     *marker2;           /* Used for small panel LU */
    int        fsupc;        /* First column of a supernode */
    int     myfnz;        /* First non-zero column of a U-segment */
    int        chperm, chmark, chrep, kchild;
    int_t   xdfs, maxdfs, nextl, k;
    int     kpar, oldrep;
    int_t   jptr, jm1ptr;
    int_t   ito, ifrom, istop;    /* Used to compress row subscripts */
    int     *xsup, *supno;
    int_t   *lsub, *xlsub;
    int_t   nzlmax, mem_error;
    int     maxsuper;
    
    xsup    = Glu->xsup;         /* Pointer to supernode starting indices */
    supno   = Glu->supno;        /* Supernode numbers for each column */
    lsub    = Glu->lsub;         /* Array of non-zero indices in L */
    xlsub   = Glu->xlsub;        /* Column pointers for L */
    nzlmax  = Glu->nzlmax;       /* Maximum non-zero entries in L */

    maxsuper = sp_ienv(3);       /* Maximum number of columns in a supernode */
    jcolp1  = jcol + 1;          /* jcol plus one */
    jcolm1  = jcol - 1;          /* jcol minus one */
    nsuper  = supno[jcol];       /* Supernode number of current column */
    jsuper  = nsuper;            /* Initialize jsuper with nsuper */
    nextl   = xlsub[jcol];       /* Next position in lsub for current column */
    marker2 = &marker[2*m];      /* Pointer to marker array */

    /* For each nonzero in A[*,jcol] do dfs */
    for (k = 0; lsub_col[k] != EMPTY; k++) {
        krow = lsub_col[k];         /* Current row index */
        lsub_col[k] = EMPTY;        /* Mark current lsub_col entry as empty */
        kmark = marker2[krow];      /* Marked position in marker2 for krow */

        /* krow was visited before, go to the next non-zero */
        if ( kmark == jcol ) continue; 

        /* For each unmarked neighbor krow of jcol
         *    krow is in L: place it in structure of L[*,jcol]
         */
        marker2[krow] = jcol;       /* Mark krow as visited by jcol */
    } /* else */

} /* for each nonzero ... */

    /* Check to see if j belongs in the same supernode as j-1 */
    if ( jcol == 0 ) { /* Do nothing for column 0 */
        nsuper = supno[0] = 0;    /* Initialize supernode number for column 0 */
    } else {
        fsupc = xsup[nsuper];      /* First column of current supernode */
        jptr = xlsub[jcol];        /* Column pointer for jcol */
        jm1ptr = xlsub[jcolm1];    /* Column pointer for jcol-1 */

#ifdef T2_SUPER
        if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;
#endif

        /* Make sure the number of columns in a supernode doesn't
           exceed threshold. */
        if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;

        /* If jcol starts a new supernode, reclaim storage space in
         * lsub from the previous supernode. Note we only store
         * the subscript set of the first and last columns of
         * a supernode. (first for num values, last for pruning)
         */
        if ( jsuper == EMPTY ) {    /* starts a new supernode */
            if ( (fsupc < jcolm1-1) ) {    /* >= 3 columns in nsuper */
#ifdef CHK_COMPRESS
                printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#else
            ito = xlsub[fsupc+1];

else 分支的开始，处理条件为 `#else` 的情况。将 `xlsub` 中索引为 `fsupc+1` 的值赋给 `ito`。


        xlsub[jcolm1] = ito;

将 `ito` 的值存入 `xlsub` 数组中索引为 `jcolm1` 的位置。


        istop = ito + jptr - jm1ptr;

计算 `istop` 的值，为 `ito` 加上 `jptr` 减去 `jm1ptr` 的结果。


        xprune[jcolm1] = istop; /* Initialize xprune[jcol-1] */

将 `istop` 的值存入 `xprune` 数组中索引为 `jcolm1` 的位置，初始化 `xprune[jcol-1]`。


        xlsub[jcol] = istop;

将 `istop` 的值存入 `xlsub` 数组中索引为 `jcol` 的位置。


        for (ifrom = jm1ptr; ifrom < nextl; ++ifrom, ++ito)
            lsub[ito] = lsub[ifrom];

循环将 `lsub` 数组中从 `jm1ptr` 到 `nextl-1` 的元素复制到从 `ito` 开始的位置。


        nextl = ito;            /* = istop + length(jcol) */

更新 `nextl` 的值为 `ito`，表示当前位置，用于后续处理。


        }
        nsuper++;
        supno[jcol] = nsuper;
    } /* if a new supernode */

结束对 `jcol` 的处理，增加 `nsuper` 的值，并将其存入 `supno[jcol]` 中。


    }    /* else: jcol > 0 */ 

结束 `else` 分支，处理条件 `jcol > 0` 的情况。


    /* Tidy up the pointers before exit */
    xsup[nsuper+1] = jcolp1;

在退出前整理指针，将 `xsup` 数组中索引为 `nsuper+1` 的位置设为 `jcolp1` 的值。


    supno[jcolp1]  = nsuper;

将 `nsuper` 的值存入 `supno[jcolp1]` 中。


    xprune[jcol]   = nextl;    /* Initialize upper bound for pruning */

将 `nextl` 的值存入 `xprune[jcol]` 中，初始化修剪的上界。


    xlsub[jcolp1]  = nextl;

将 `nextl` 的值存入 `xlsub[jcolp1]` 中。


    return 0;
}

函数返回值为 `0`，表示成功结束函数执行。
```