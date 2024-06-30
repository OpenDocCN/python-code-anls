# `D:\src\scipysrc\scipy\scipy\special\_round.h`

```
/*
 * Functions for adding two double precision numbers with rounding to
 * infinity or rounding to negative infinity without using <fenv.h>.
 */
#ifndef ROUND_H
#define ROUND_H

#include <math.h>


/* Computes fl(a+b) and err(a+b).  */
static inline double two_sum(double a, double b, double *err)
{
    // Use volatile to prevent compiler optimizations affecting floating-point results
    volatile double s = a + b;
    volatile double c = s - a;
    volatile double d = b - c;
    volatile double e = s - c;
    // Calculate the error term err(a+b)
    *err = (a - e) + d;
    return s;  // Return the floating-point sum fl(a+b)
}


double add_round_up(double a, double b)
{
    double s, err;

    if (isnan(a) || isnan(b)) {
        return NAN;  // Return NaN if any operand is NaN
    }

    // Compute the floating-point sum and the error term
    s = two_sum(a, b, &err);
    if (err > 0) {
        /* fl(a + b) rounded down */
        return nextafter(s, INFINITY);  // Return fl(a+b) rounded to the next representable number toward positive infinity
    }
    else {
        /* fl(a + b) rounded up or didn't round */
        return s;  // Return fl(a+b) as computed
    }
}


double add_round_down(double a, double b)
{
    double s, err;

    if (isnan(a) || isnan(b)) {
        return NAN;  // Return NaN if any operand is NaN
    }

    // Compute the floating-point sum and the error term
    s = two_sum(a, b, &err);
    if (err < 0) {
        return nextafter(s, -INFINITY);  // Return fl(a+b) rounded to the next representable number toward negative infinity
    }
    else {
        return s;  // Return fl(a+b) as computed
    }
}


/* Helper code for testing _round.h. */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__cplusplus)
/* We have C99, or C++11 or higher; both have fenv.h */
#include <fenv.h>
#else
// Define fallback implementations for systems without <fenv.h>

// Stub function for setting rounding mode (returns -1 indicating failure)
int fesetround(int round)
{
    return -1;
}

// Stub function for getting rounding mode (returns -1 indicating failure)
int fegetround()
{
    return -1;
}

// Define constants for rounding modes (although not used)
#define FE_UPWARD -1
#define FE_DOWNWARD -1

#endif


#endif /* _round.h */
```