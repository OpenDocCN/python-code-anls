# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zcolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zcolumn_dfs.c
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

#include "slu_zdefs.h"

/*! \brief What type of supernodes we want */
#define T2_SUPER

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   ZCOLUMN_DFS performs a symbolic factorization on column jcol, and
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
int
    /* Initialize local variables */
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

    /* Assign pointers from Glu struct to local variables */
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    nzlmax  = Glu->nzlmax;

    /* Determine maximum supernode size */
    maxsuper = sp_ienv(3);

    /* Initialize column-related variables */
    jcolp1  = jcol + 1;
    jcolm1  = jcol - 1;
    nsuper  = supno[jcol];
    jsuper  = nsuper;
    nextl   = xlsub[jcol];
    marker2 = &marker[2*m];

    /* Process each nonzero entry in lsub_col array */
    for (k = 0; lsub_col[k] != EMPTY; k++) {
        krow = lsub_col[k];
        lsub_col[k] = EMPTY;
        kmark = marker2[krow];

        /* Skip krow if already marked */
        if ( kmark == jcol ) continue;

        /* Mark krow as visited by jcol */
        marker2[krow] = jcol;
    }

    /* Determine if jcol starts a new supernode */
    if ( jcol == 0 ) {
        /* Handle column 0 case: no new supernode */
        nsuper = supno[0] = 0;
    } else {
        /* Get first supernode column and pointers for jcol and jcol-1 */
        fsupc = xsup[nsuper];
        jptr = xlsub[jcol];    /* Not compressed yet */
        jm1ptr = xlsub[jcolm1];

        /* Conditionally update jsuper based on the difference in pointers */
#ifdef T2_SUPER
        if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;
#endif

        /* Ensure jcol does not exceed the maximum supernode size */
        if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;

        /* Start a new supernode if necessary */
        if ( jsuper == EMPTY ) {
            /* Determine if previous supernode has sufficient columns */
            if ( (fsupc < jcolm1-1) ) {
#ifdef CHK_COMPRESS
                printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);

                /* Print message about compressing lsub[] for the new supernode */
                printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#endif
            ito = xlsub[fsupc+1];    /* Assign ito as xlsub[fsupc+1] */
        xlsub[jcolm1] = ito;    /* Set xlsub[jcolm1] to ito */
        istop = ito + jptr - jm1ptr;    /* Compute istop based on ito, jptr, and jm1ptr */
        xprune[jcolm1] = istop; /* Initialize xprune[jcol-1] */
        xlsub[jcol] = istop;    /* Set xlsub[jcol] to istop */
        for (ifrom = jm1ptr; ifrom < nextl; ++ifrom, ++ito)
            lsub[ito] = lsub[ifrom];    /* Copy elements from lsub[ifrom] to lsub[ito] */
        nextl = ito;            /* Update nextl to ito, indicating end of copied elements */
        }
        nsuper++;    /* Increment nsuper */
        supno[jcol] = nsuper;    /* Assign nsuper to supno[jcol] */
    } /* if a new supernode */

    }    /* else: jcol > 0 */ 
    
    /* Tidy up the pointers before exit */
    xsup[nsuper+1] = jcolp1;    /* Set xsup[nsuper+1] to jcolp1 */
    supno[jcolp1]  = nsuper;    /* Assign nsuper to supno[jcolp1] */
    xprune[jcol]   = nextl;    /* Initialize upper bound for pruning */
    xlsub[jcolp1]  = nextl;    /* Set xlsub[jcolp1] to nextl */

    return 0;    /* Return 0 to indicate successful completion */
}
```