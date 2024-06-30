# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_csnode_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_csnode_dfs.c
 * \brief Determines the union of row structures of columns within the relaxed node
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_cdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ilu_csnode_dfs() - Determine the union of the row structures of those
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
 * \brief Determines the union of row structures of columns within a relaxed supernode.
 *
 * \param jcol      [in] Start index of the supernode.
 * \param kcol      [in] End index of the supernode.
 * \param asub      [in] Column indices of the matrix A in compressed column storage (CCS) format.
 * \param xa_begin  [in] Starting pointers to column indices in asub.
 * \param xa_end    [in] Ending pointers to column indices in asub.
 * \param marker    [modified] Marker array to track visited rows.
 * \param Glu       [modified] GlobalLU_t structure containing supernode and column data.
 *
 * \return
 *     - 0   if successful.
 *     - >0  if memory allocation fails (number of bytes allocated).
 */
int
ilu_csnode_dfs(
       const int  jcol,        /* in - start of the supernode */
       const int  kcol,        /* in - end of the supernode */
       const int_t  *asub,        /* in - column indices of A */
       const int_t  *xa_begin,    /* in - starting pointers for columns in asub */
       const int_t  *xa_end,        /* in - ending pointers for columns in asub */
       int          *marker,        /* modified - marker array */
       GlobalLU_t *Glu        /* modified - structure containing LU decomposition data */
       )
{
    int_t i, k, nextl, mem_error;
    int   nsuper, krow, kmark;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    int_t nzlmax;

    xsup    = Glu->xsup;    /* Pointer to supernodes */
    supno   = Glu->supno;   /* Pointer to supernode numbers */
    lsub    = Glu->lsub;    /* Pointer to L matrix in column-oriented format */
    xlsub   = Glu->xlsub;   /* Pointer to starting positions of columns in L */
    nzlmax  = Glu->nzlmax;  /* Maximum size of L matrix storage */

    nsuper = ++supno[jcol];    /* Assign the next available supernode number */
    nextl = xlsub[jcol];    /* Start of the current column in L */

    for (i = jcol; i <= kcol; i++)
    {
    /* For each nonzero in A[*,i] */
    for (k = xa_begin[i]; k < xa_end[i]; k++)
    {
        krow = asub[k];    /* Row index corresponding to the nonzero element */
        kmark = marker[krow];    /* Current marker value for the row */

        if ( kmark != kcol )
        { /* First time visit krow */
        marker[krow] = kcol;    /* Mark krow as visited */
        lsub[nextl++] = krow;    /* Store krow in L matrix */

        if ( nextl >= nzlmax )
        {
            if ( (mem_error = cLUMemXpand(jcol, nextl, LSUB, &nzlmax,
                Glu)) != 0)
            return (mem_error);    /* Expand memory allocation for L if needed */
            lsub = Glu->lsub;    /* Update pointer in case of reallocation */
        }
        }
    }
    supno[i] = nsuper;    /* Assign supernode number for column i */
    }

    /* Supernode > 1 */
    if ( jcol < kcol )
    for (i = jcol+1; i <= kcol; i++) xlsub[i] = nextl;

    xsup[nsuper+1] = kcol + 1;    /* Update supernode boundaries */
    supno[kcol+1]  = nsuper;    /* Mark end of last column supernode */
    xlsub[kcol+1]  = nextl;    /* Store end position of the last column in L */

    return 0;    /* Return success */
}
```