# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\newrand\newrand.h`

```
/*
   Creation, 2020:
   - New random number generator using a Mersenne Twister with a tweaked Lemire
     postprocessor. This was implemented to resolve convergence issues specific
     to Windows targets for libraries like LibSVM and LibLinear.
     Sylvain Marie, Schneider Electric
     See <https://github.com/scikit-learn/scikit-learn/pull/13511#issuecomment-481729756>
 */
#ifndef _NEWRAND_H
#define _NEWRAND_H

#ifdef __cplusplus
#include <random>  // needed for Cython to generate a .cpp file from newrand.h
extern "C" {
#endif

// Scikit-Learn-specific random number generator replacing `rand()` originally
// used in LibSVM / LibLinear, ensuring consistent behavior across Windows and Linux,
// with improved speed.

// - (1) Initialize a `mt_rand` object
std::mt19937 mt_rand(std::mt19937::default_seed);

// - (2) Public `set_seed()` function to set a new seed for the random generator.
void set_seed(unsigned custom_seed) {
    mt_rand.seed(custom_seed);
}

// - (3) New internal `bounded_rand_int` function, used instead of rand() throughout.
inline uint32_t bounded_rand_int(uint32_t range) {
    // Original method used in LibSVM / LibLinear: generate a 31-bit positive
    // random number and use modulo to fit within the range.
    // return abs( (int)mt_rand()) % range;

    // Improved method: tweaked Lemire post-processor
    // Source: http://www.pcg-random.org/posts/bounded-rands.html
    uint32_t x = mt_rand();
    uint64_t m = uint64_t(x) * uint64_t(range);
    uint32_t l = uint32_t(m);
    if (l < range) {
        uint32_t t = -range;
        if (t >= range) {
            t -= range;
            if (t >= range)
                t %= range;
        }
        while (l < t) {
            x = mt_rand();
            m = uint64_t(x) * uint64_t(range);
            l = uint32_t(m);
        }
    }
    return m >> 32;
}

#ifdef __cplusplus
}
#endif

#endif /* _NEWRAND_H */
```