# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\superlu_timer.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file superlu_timer.c
 * \brief Returns the time used
 *
 * <pre>
 * Purpose
 * ======= 
 * 
 * Returns the time in seconds used by the process.
 *
 * Note: the timer function call is machine dependent. Use conditional
 *       compilation to choose the appropriate function.
 * </pre>
 */

/*! \brief Timer function using high-resolution timer on SUN OS
 */
#ifdef SUN 
/*
 *     It uses the system call gethrtime(3C), which is accurate to 
 *    nanoseconds. 
*/
#include <sys/time.h>

double SuperLU_timer_() {
    // 返回当前时间，单位为秒
    return ( (double)gethrtime() / 1e9 );
}

/*! \brief Timer function using clock() on Windows
 */
#elif _WIN32

#include <time.h>

double SuperLU_timer_()
{
    clock_t t;
    t=clock();

    // 返回当前 CPU 时间，单位为秒
    return ((double)t)/CLOCKS_PER_SEC;
}

/*! \brief Timer function using times() on UNIX systems
 */
#elif defined( USE_TIMES )

#ifndef NO_TIMER
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <unistd.h>
#endif

/*! \brief Timer function using times() on UNIX systems
 */ 
double SuperLU_timer_()
{
#ifdef NO_TIMER
    /* no sys/times.h on WIN32 */
    double tmp;
    tmp = 0.0;
#else
    struct tms use;
    double tmp;
    int clocks_per_sec = sysconf(_SC_CLK_TCK);

    // 获取进程的 CPU 时间
    times ( &use );
    tmp = use.tms_utime;
    tmp += use.tms_stime;
#endif
    // 返回 CPU 时间，单位为秒
    return (double)(tmp) / clocks_per_sec;
}

/*! \brief Timer function using gettimeofday() on UNIX systems
 */
#else

#include <sys/time.h>
#include <stdlib.h>

/*! \brief Timer function using gettimeofday() on UNIX systems
 */ 
double SuperLU_timer_(void)
{
    struct timeval tp;
    double tmp;

    // 获取当前时间
    gettimeofday(&tp, NULL);
    tmp = tp.tv_sec + tp.tv_usec/1000000.0;

    // 返回当前时间，单位为秒
    return (tmp);
}

#endif
```