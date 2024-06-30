# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_relax_snode.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file ilu_relax_snode.c
 * \brief Identify initial relaxed supernodes
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 1, 2009
 * </pre>
 */

#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ilu_relax_snode() - Identify the initial relaxed supernodes, assuming
 *    that the matrix has been reordered according to the postorder of the
 *    etree.
 * </pre>
 */
void
ilu_relax_snode (
          const    int n,
          int    *et,           /* column elimination tree */
          const int relax_columns, /* max no of columns allowed in a
                      relaxed snode */
          int    *descendants,  /* no of descendants of each node
                     in the etree */
          int    *relax_end,    /* last column in a supernode
                    * if j-th column starts a relaxed
                    * supernode, relax_end[j] represents
                    * the last column of this supernode */
          int    *relax_fsupc   /* first column in a supernode
                    * relax_fsupc[j] represents the first
                    * column of j-th supernode */
         )
{
    register int j, f, parent;
    register int snode_start;    /* beginning of a snode */

    // Initialize arrays
    ifill (relax_end, n, EMPTY);   // Set all entries of relax_end to EMPTY
    ifill (relax_fsupc, n, EMPTY); // Set all entries of relax_fsupc to EMPTY
    for (j = 0; j < n; j++) descendants[j] = 0;  // Initialize descendants array to zero

    /* Compute the number of descendants of each node in the etree */
    for (j = 0; j < n; j++) {
        parent = et[j];
        if ( parent != n )  // Check if the node is not the dummy root
            descendants[parent] += descendants[j] + 1;  // Increment descendants count
    }

    /* Identify the relaxed supernodes by postorder traversal of the etree. */
    for (j = f = 0; j < n; ) {
        parent = et[j];
        snode_start = j;
        // Traverse up the tree until a suitable supernode is found
        while ( parent != n && descendants[parent] < relax_columns ) {
            j = parent;
            parent = et[j];
        }
        /* Found a supernode with j being the last column. */
        relax_end[snode_start] = j;  // Record the last column of the current supernode
        j++;  // Move to the next column
        relax_fsupc[f++] = snode_start;  // Record the first column of the current supernode
        // Find the next leaf node in the etree
        while ( descendants[j] != 0 && j < n ) j++;
    }
}
```