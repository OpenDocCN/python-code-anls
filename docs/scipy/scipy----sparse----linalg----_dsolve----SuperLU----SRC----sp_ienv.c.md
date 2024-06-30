# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sp_ienv.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file SRC/sp_ienv.c
 * \brief Chooses machine-dependent parameters for the local environment.
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November, 2010
 * </pre>
*/

/*
 * File name:        sp_ienv.c
 * History:             Modified from lapack routine ILAENV
 */
#include "slu_Cnames.h"

/*! \brief

 <pre>
    Purpose   
    =======   

    sp_ienv() is inquired to choose machine-dependent parameters for the
    local environment. See ISPEC for a description of the parameters.   

    This version provides a set of parameters which should give good,   
    but not optimal, performance on many of the currently available   
    computers.  Users are encouraged to modify this subroutine to set   
    the tuning parameters for their particular machine using the option   
    and problem size information in the arguments.   

    Arguments   
    =========   

    ISPEC   (input) int
            Specifies the parameter to be returned as the value of SP_IENV.   
            = 1: the panel size w; a panel consists of w consecutive
             columns of matrix A in the process of Gaussian elimination.
         The best value depends on machine's cache characters.
            = 2: the relaxation parameter relax; if the number of
             nodes (columns) in a subtree of the elimination tree is less
         than relax, this subtree is considered as one supernode,
         regardless of their row structures.
            = 3: the maximum size for a supernode in complete LU;
        = 4: the minimum row dimension for 2-D blocking to be used;
        = 5: the minimum column dimension for 2-D blocking to be used;
        = 6: the estimated fills factor for L and U, compared with A;
        = 7: the maximum size for a supernode in ILU.
        
   (SP_IENV) (output) int
            >= 0: the value of the parameter specified by ISPEC   
            < 0:  if SP_IENV = -k, the k-th argument had an illegal value. 
  
    ===================================================================== 
</pre>
*/
int
sp_ienv(int ispec)
{
    int i;
    extern int input_error(char *, int *);

    // 根据参数 ISPEC 返回对应的机器相关参数值
    switch (ispec) {
    case 1: return (20); // 返回默认的面板大小参数
    case 2: return (10); // 返回默认的松弛参数
    case 3: return (200); // 返回完整 LU 中超节点的最大大小
    case 4: return (200); // 返回2-D阻塞中使用的最小行维度
    case 5: return (100); // 返回2-D阻塞中使用的最小列维度
    case 6: return (30);  // 返回L和U的估计填充因子与A相比
    case 7: return (10);  // 返回 ILU 中超节点的最大大小
    }

    // 如果 ISPEC 的值非法，则调用错误处理函数
    i = 1;
    input_error("sp_ienv", &i);
    return 0;

} /* sp_ienv_ */
```