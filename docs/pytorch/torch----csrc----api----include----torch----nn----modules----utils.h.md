# `.\pytorch\torch\csrc\api\include\torch\nn\modules\utils.h`

```py
#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

#include <vector>

namespace torch {
namespace nn {
namespace modules {
namespace utils {

// Reverse the order of `t` and repeat each element for `n` times.
// This can be used to translate padding arg used by Conv and Pooling modules
// to the ones used by `F::pad`.
//
// This mirrors `_reverse_repeat_tuple` in `torch/nn/modules/utils.py`.
inline std::vector<int64_t> _reverse_repeat_vector(
    at::ArrayRef<int64_t> t,
    int64_t n) {
  // Assert that `n` is non-negative
  TORCH_INTERNAL_ASSERT(n >= 0);
  
  // Initialize an empty vector `ret` to store the result
  std::vector<int64_t> ret;
  // Reserve memory for `ret` to avoid reallocations
  ret.reserve(t.size() * n);
  
  // Iterate over `t` in reverse order
  for (auto rit = t.rbegin(); rit != t.rend(); ++rit) {
    // Repeat each element `n` times
    for (const auto i : c10::irange(n)) {
      (void)i; // Suppress unused variable warning
      // Add the element `*rit` to the result vector `ret`
      ret.emplace_back(*rit);
    }
  }
  
  // Return the reversed and repeated vector
  return ret;
}

// Create a vector `ret` based on `out_size` with default values from `defaults`
inline std::vector<int64_t> _list_with_default(
    torch::ArrayRef<std::optional<int64_t>> out_size,
    torch::IntArrayRef defaults) {
  // Check that `defaults` has enough dimensions to accommodate `out_size`
  TORCH_CHECK(
      defaults.size() > out_size.size(),
      "Input dimension should be at least ",
      out_size.size() + 1);
  
  // Initialize an empty vector `ret` to store the result
  std::vector<int64_t> ret;
  
  // Slice `defaults` to match the size of `out_size`
  torch::IntArrayRef defaults_slice =
      defaults.slice(defaults.size() - out_size.size(), out_size.size());
  
  // Iterate over `out_size` to populate `ret`
  for (const auto i : c10::irange(out_size.size())) {
    // Retrieve the value from `out_size` at index `i`
    auto v = out_size.at(i);
    // Retrieve the default value from `defaults_slice` at index `i`
    auto d = defaults_slice.at(i);
    // If `v` has a value, add it to `ret`; otherwise, add `d`
    ret.emplace_back(v.has_value() ? v.value() : d);
  }
  
  // Return the resulting vector `ret`
  return ret;
}

} // namespace utils
} // namespace modules
} // namespace nn
} // namespace torch
```