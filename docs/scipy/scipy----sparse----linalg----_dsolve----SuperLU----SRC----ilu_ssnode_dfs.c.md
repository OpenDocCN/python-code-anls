# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_ssnode_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_ssnode_dfs.c
 * \brief Determines the union of row structures of columns within the relaxed node
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_sdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ilu_ssnode_dfs() - Determine the union of the row structures of those
 *    columns within the relaxed snode.
 *    Note: The relaxed snodes are leaves of the supernodal etree, therefore,
 *    the portion outside the rectangular supernode must be zero.
 *
 * Return value
 * ============
 *     0   success;
 *    >0   number of bytes allocated when run out of memory.
 * </pre>
 */

/*!
 * \brief Determines the union of row structures of columns within the relaxed supernode.
 *
 * This function computes the union of row indices for columns within a given supernode
 * range, updating marker arrays and data structures in Glu.
 *
 * \param jcol     [in] Start index of the supernode.
 * \param kcol     [in] End index of the supernode.
 * \param asub     [in] Column indices of the matrix A in compressed format.
 * \param xa_begin [in] Starting indices of column arrays in asub.
 * \param xa_end   [in] Ending indices of column arrays in asub.
 * \param marker   [in,out] Marker array to track visited rows.
 * \param Glu      [in,out] Global data structure containing supernode information.
 *
 * \return
 * - 0 on success.
 * - >0 indicating the number of bytes allocated in case of memory expansion.
 */
int
ilu_ssnode_dfs(
       const int  jcol,        /* in - start of the supernode */
       const int  kcol,        /* in - end of the supernode */
       const int_t  *asub,        /* in */
       const int_t  *xa_begin,    /* in */
       const int_t  *xa_end,        /* in */
       int          *marker,        /* modified */
       GlobalLU_t *Glu        /* modified */
       )
{
    int_t i, k, nextl, mem_error;
    int   nsuper, krow, kmark;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    int_t nzlmax;

    xsup    = Glu->xsup;    /* Supernode starting positions */
    supno   = Glu->supno;   /* Supernode numbers */
    lsub    = Glu->lsub;    /* Array of row indices */
    xlsub   = Glu->xlsub;   /* Starting positions of each column's row indices */
    nzlmax  = Glu->nzlmax;  /* Maximum allowed size for lsub */

    nsuper = ++supno[jcol];    /* Next available supernode number */
    nextl = xlsub[jcol];        /* Next available index in lsub */

    for (i = jcol; i <= kcol; i++)
    {
        /* For each nonzero in A[*,i] */
        for (k = xa_begin[i]; k < xa_end[i]; k++)
        {
            krow = asub[k];         /* Row index */
            kmark = marker[krow];   /* Marker for the row */
            if ( kmark != kcol )
            { /* First time visit krow */
                marker[krow] = kcol;    /* Mark this row as visited */
                lsub[nextl++] = krow;   /* Add krow to the list of rows in the supernode */
                if ( nextl >= nzlmax )
                {
                    if ( (mem_error = sLUMemXpand(jcol, nextl, LSUB, &nzlmax,
                        Glu)) != 0)
                        return (mem_error); /* Memory expansion error handling */
                    lsub = Glu->lsub;   /* Update lsub after potential reallocation */
                }
            }
        }
        supno[i] = nsuper;  /* Assign supernode number for column i */
    }

    /* Supernode > 1 */
    if ( jcol < kcol )
        for (i = jcol+1; i <= kcol; i++) xlsub[i] = nextl;

    xsup[nsuper+1] = kcol + 1;  /* End of supernode */
    supno[kcol+1]  = nsuper;    /* Assign supernode number for end of column range */
    xlsub[kcol+1]  = nextl;     /* End marker for xlsub */

    return 0;   /* Successful completion */
}
```