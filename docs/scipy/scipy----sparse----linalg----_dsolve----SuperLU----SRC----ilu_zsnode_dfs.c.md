# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_zsnode_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_zsnode_dfs.c
 * \brief Determines the union of row structures of columns within the relaxed node
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_zdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ilu_zsnode_dfs() - Determine the union of the row structures of those
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
 * This function computes the row structure union of columns within a specified supernode range.
 *
 * \param jcol     [in] Start of the supernode.
 * \param kcol     [in] End of the supernode.
 * \param asub     [in] Row indices of nonzeros in the matrix columns.
 * \param xa_begin [in] Starting positions of column data in 'asub'.
 * \param xa_end   [in] Ending positions of column data in 'asub'.
 * \param marker   [modified] Marker array to track visited rows.
 * \param Glu      [modified] Global LU factorization data structure.
 *
 * \return
 *     - 0 if successful.
 *     - >0 if memory allocation fails.
 */

int
ilu_zsnode_dfs(
       const int  jcol,        /* in - start of the supernode */
       const int  kcol,        /* in - end of the supernode */
       const int_t  *asub,     /* in - row indices of nonzeros */
       const int_t  *xa_begin, /* in - starting positions in 'asub' */
       const int_t  *xa_end,   /* in - ending positions in 'asub' */
       int          *marker,   /* modified - marker array */
       GlobalLU_t *Glu         /* modified - global LU data structure */
       )
{
    int_t i, k, nextl, mem_error;
    int   nsuper, krow, kmark;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    int_t nzlmax;

    xsup    = Glu->xsup;    // Pointer to array of supernode starting points
    supno   = Glu->supno;   // Pointer to array of supernode numbers
    lsub    = Glu->lsub;    // Pointer to array of row indices
    xlsub   = Glu->xlsub;   // Pointer to array of starting positions of columns in lsub
    nzlmax  = Glu->nzlmax;  // Maximum capacity of lsub

    nsuper = ++supno[jcol];  // Increment and assign the next available supernode number
    nextl = xlsub[jcol];     // Start position in lsub for this supernode

    for (i = jcol; i <= kcol; i++)
    {
        // Iterate over each column in the supernode
        for (k = xa_begin[i]; k < xa_end[i]; k++)
        {
            krow = asub[k];         // Row index of current nonzero
            kmark = marker[krow];   // Current marker value for this row

            if (kmark != kcol)
            {
                // First time visiting this row in the current supernode
                marker[krow] = kcol;  // Mark this row as visited
                lsub[nextl++] = krow; // Store this row index in lsub

                // Expand lsub if necessary
                if (nextl >= nzlmax)
                {
                    if ((mem_error = zLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu)) != 0)
                        return (mem_error);  // Memory expansion error handling
                    lsub = Glu->lsub;         // Update lsub pointer after potential reallocation
                }
            }
        }
        supno[i] = nsuper;  // Assign supernode number to each column
    }

    // If supernode size > 1, update xlsub for all columns in the supernode
    if (jcol < kcol)
    {
        for (i = jcol+1; i <= kcol; i++)
            xlsub[i] = nextl;
    }

    // Update supernode boundaries and markers
    xsup[nsuper+1] = kcol + 1;
    supno[kcol+1]  = nsuper;
    xlsub[kcol+1]  = nextl;

    return 0;  // Return success
}
```