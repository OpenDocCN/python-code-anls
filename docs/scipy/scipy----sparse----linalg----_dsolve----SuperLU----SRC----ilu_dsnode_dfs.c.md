# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_dsnode_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_dsnode_dfs.c
 * \brief Determines the union of row structures of columns within the relaxed node
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ilu_dsnode_dfs() - Determine the union of the row structures of those
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

/*! \brief Determines the union of row structures of columns within the relaxed supernode
 *
 * \param jcol     [in] - start of the supernode
 * \param kcol     [in] - end of the supernode
 * \param asub     [in] - column indices of the supernode
 * \param xa_begin [in] - starting positions of columns in asub
 * \param xa_end   [in] - ending positions of columns in asub
 * \param marker   [modified] - marker array for row indices
 * \param Glu      [modified] - global LU data structure
 *
 * \return
 *     0   success;
 *    >0   number of bytes allocated when run out of memory.
 */
int
ilu_dsnode_dfs(
       const int  jcol,        /* in - start of the supernode */
       const int  kcol,        /* in - end of the supernode */
       const int_t  *asub,     /* in - column indices of the supernode */
       const int_t  *xa_begin, /* in - starting positions of columns in asub */
       const int_t  *xa_end,   /* in - ending positions of columns in asub */
       int          *marker,   /* modified - marker array for row indices */
       GlobalLU_t *Glu         /* modified - global LU data structure */
       )
{
    int_t i, k, nextl, mem_error;
    int   nsuper, krow, kmark;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    int_t nzlmax;

    xsup    = Glu->xsup;     // Pointer to supernode boundaries
    supno   = Glu->supno;    // Supernode numbers for columns
    lsub    = Glu->lsub;     // Array of row indices
    xlsub   = Glu->xlsub;    // Pointer to start of each column's row indices
    nzlmax  = Glu->nzlmax;   // Maximum capacity of lsub array

    nsuper = ++supno[jcol];  // Assign the next available supernode number to current supernode
    nextl = xlsub[jcol];     // Starting position in lsub for current supernode

    for (i = jcol; i <= kcol; i++)
    {
        // For each column in the supernode
        for (k = xa_begin[i]; k < xa_end[i]; k++)
        {
            krow = asub[k];     // Row index of the nonzero element
            kmark = marker[krow];   // Current marker for krow
            if ( kmark != kcol )
            {
                // First time visit of krow in the current supernode
                marker[krow] = kcol;    // Mark krow as visited by current supernode
                lsub[nextl++] = krow;   // Add krow to the column's structure in lsub
                if ( nextl >= nzlmax )
                {
                    // Expand lsub if it runs out of space
                    if ( (mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu)) != 0)
                        return (mem_error); // Return if memory expansion fails
                    lsub = Glu->lsub;   // Update lsub pointer after potential reallocation
                }
            }
        }
        supno[i] = nsuper;  // Assign the supernode number to each column
    }

    // Handle the end of the supernode
    if ( jcol < kcol )
        for (i = jcol+1; i <= kcol; i++) xlsub[i] = nextl;

    xsup[nsuper+1] = kcol + 1;  // Update supernode boundaries
    supno[kcol+1]  = nsuper;    // Assign supernode number to the end column
    xlsub[kcol+1]  = nextl;     // Pointer to start of next supernode in xlsub

    return 0;   // Return success
}
```