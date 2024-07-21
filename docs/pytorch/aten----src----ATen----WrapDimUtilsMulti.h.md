# `.\pytorch\aten\src\ATen\WrapDimUtilsMulti.h`

```
#pragma once

#include <ATen/WrapDimUtils.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/irange.h>
#include <bitset>
#include <sstream>

namespace at {

// This function converts a list of dimensions into a std::bitset representation
// where each bit corresponds to whether a dimension is present or not.
// It ensures that no dimension exceeds the maximum supported size defined by dim_bitset_size.

constexpr size_t dim_bitset_size = 64;

static inline std::bitset<dim_bitset_size> dim_list_to_bitset(
    OptionalIntArrayRef opt_dims,  // Optional reference to an array of dimensions
    size_t ndims) {                // Total number of dimensions for the tensor
  TORCH_CHECK(
      ndims <= dim_bitset_size,    // Ensure number of dimensions is within supported range
      "only tensors with up to ",
      dim_bitset_size,
      " dims are supported");
  std::bitset<dim_bitset_size> seen;  // Initialize a bitset to track seen dimensions
  if (opt_dims.has_value()) {         // Check if optional dimensions are provided
    auto dims = opt_dims.value();     // Retrieve the actual dimensions
    for (const auto i : c10::irange(dims.size())) {  // Iterate over provided dimensions
      size_t dim = maybe_wrap_dim(dims[i], static_cast<int64_t>(ndims));  // Wrap dimension index if necessary
      TORCH_CHECK(
          !seen[dim],                // Check if dimension has already been seen
          "dim ",
          dim,
          " appears multiple times in the list of dims");
      seen[dim] = true;             // Set the corresponding bit for the dimension
    }
  } else {                          // If no specific dimensions are provided
    for (size_t dim = 0; dim < ndims; dim++) {
      seen[dim] = true;             // Set all bits up to ndims to true
    }
  }
  return seen;                      // Return the populated bitset representing dimensions
}

} // namespace at
```