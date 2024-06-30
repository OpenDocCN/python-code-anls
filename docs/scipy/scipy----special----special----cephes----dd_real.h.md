# `D:\src\scipysrc\scipy\scipy\special\special\cephes\dd_real.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 *
 * The parts of the qd double-double floating point package used in SciPy
 * have been reworked in a more modern C++ style using operator overloading.
 */

/*
 * include/double2.h
 *
 * This work was supported by the Director, Office of Science, Division
 * of Mathematical, Information, and Computational Sciences of the
 * U.S. Department of Energy under contract numbers DE-AC03-76SF00098 and
 * DE-AC02-05CH11231.
 *
 * Copyright (c) 2003-2009, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from U.S. Dept. of Energy) All rights reserved.
 *
 * By downloading or using this software you are agreeing to the modified
 * BSD license "BSD-LBNL-License.doc" (see LICENSE.txt).
 */

/*
 * Double-double precision (>= 106-bit significand) floating point
 * arithmetic package based on David Bailey's Fortran-90 double-double
 * package, with some changes. See
 *
 *   http://www.nersc.gov/~dhbailey/mpdist/mpdist.html
 *
 * for the original Fortran-90 version.
 *
 * Overall structure is similar to that of Keith Brigg's C++ double-double
 * package.  See
 *
 *   http://www-epidem.plansci.cam.ac.uk/~kbriggs/doubledouble.html
 *
 * for more details.  In particular, the fix for x86 computers is borrowed
 * from his code.
 *
 * Yozo Hida
 */

/*
 * This code taken from v2.3.18 of the qd package.
 */

#pragma once

#include "../config.h"

#include "unity.h"

namespace special {
namespace cephes {

    } // namespace detail

} // namespace cephes
} // namespace special
```