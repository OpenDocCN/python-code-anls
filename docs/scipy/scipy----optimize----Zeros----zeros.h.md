# `D:\src\scipysrc\scipy\scipy\optimize\Zeros\zeros.h`

```
/* Written by Charles Harris charles.harris@sdl.usu.edu */

/* Modified to not depend on Python everywhere by Travis Oliphant.
 */

#ifndef ZEROS_H
#define ZEROS_H

/* Definition of a structure to hold solver statistics */
typedef struct {
    int funcalls;               // Number of function calls made during solving
    int iterations;             // Number of iterations performed during solving
    int error_num;              // Error code indicating the reason for termination
} scipy_zeros_info;


/* Constants representing different solver termination conditions */
#define CONVERGED 0      // Indicates convergence (solution found)
#define SIGNERR -1       // Indicates sign error in the solution
#define CONVERR -2       // Indicates convergence error in the solution
#define EVALUEERR -3     // Indicates error related to function values
#define INPROGRESS 1     // Indicates solver is still in progress

/* Definition of function pointer types for callback and solver functions */
typedef double (*callback_type)(double, void*);
typedef double (*solver_type)(callback_type, double, double, double, double,
                              int, void *, scipy_zeros_info*);

/* Function prototypes for different solvers */
extern double bisect(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data_param,
                     scipy_zeros_info *solver_stats);
extern double ridder(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data_param,
                     scipy_zeros_info *solver_stats);
extern double brenth(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data_param,
                     scipy_zeros_info *solver_stats);
extern double brentq(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data_param,
                     scipy_zeros_info *solver_stats);

#endif
```