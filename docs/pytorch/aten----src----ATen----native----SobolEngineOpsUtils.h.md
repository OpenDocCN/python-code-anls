# `.\pytorch\aten\src\ATen\native\SobolEngineOpsUtils.h`

```py
/// This file contains some tensor-agnostic operations to be used in the
/// core functions of the `SobolEngine`
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#endif

namespace at::native::sobol_utils {

/// Function to return the minimum number of bits required to represent the integer `n`
inline int64_t bit_length(const int64_t n) {
  int64_t nbits, nloc;
  // Calculate the number of bits by iteratively dividing by 2 until nloc becomes zero
  for (nloc = n, nbits = 0; nloc > 0; nloc /= 2, nbits++);
  return nbits;
}

/// Function to get the position of the rightmost zero bit in the binary representation of an integer `n`
/// This position is returned as a zero-indexed value
inline int64_t rightmost_zero(const int64_t n) {
  int64_t z, i;
  // Find the position of the rightmost zero bit by iterating through the binary representation
  for (z = n, i = 0; z % 2 == 1; z /= 2, i++);
  return i;
}

/// Function to extract a subsequence of bits from the binary representation of an integer `n`
/// starting from position `pos` and of length `length`
inline int64_t bitsubseq(const int64_t n, const int64_t pos, const int64_t length) {
  // Extract and return the specified subsequence of bits
  return (n >> pos) & ((1 << length) - 1);
}

/// Function to compute the inner product between a batched square matrix and a vector
/// where each element is a power of 2
inline at::Tensor cdot_pow2(const at::Tensor& bmat) {
  // Generate a vector of powers of 2 in reverse order up to the size of the last dimension of bmat
  at::Tensor inter = at::arange(bmat.size(-1) - 1, -1, -1, bmat.options());
  // Expand the powers of 2 vector to match the shape of bmat and compute element-wise multiplication
  inter = at::pow(2, inter).expand_as(bmat);
  // Sum the result along the last dimension to compute the inner product
  return at::mul(inter, bmat).sum(-1);
}

/// All definitions below this point are data constants and should not be modified
/// without careful consideration

/// Maximum dimension size for Sobol sequences
constexpr int64_t MAXDIM = 21201;
/// Maximum degree size for Sobol sequences
constexpr int64_t MAXDEG = 18;
/// Maximum bit size for representing numbers in Sobol sequences
constexpr int64_t MAXBIT = 30;
/// Largest number representable in MAXBIT bits
constexpr int64_t LARGEST_NUMBER = 1 << MAXBIT;
/// Reciprocal of LARGEST_NUMBER, used for scaling in Sobol sequences
constexpr float RECIPD = 1.0 / LARGEST_NUMBER;

/// External declaration of the polynomial values used in Sobol sequences
extern const int64_t poly[MAXDIM];
/// External declaration of the initial Sobol state values used in Sobol sequences
extern const int64_t initsobolstate[MAXDIM][MAXDEG];

} // namespace at::native::sobol_utils
```