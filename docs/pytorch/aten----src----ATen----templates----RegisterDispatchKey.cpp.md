# `.\pytorch\aten\src\ATen\templates\RegisterDispatchKey.cpp`

```
// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

// an external backend might generate file within its code tree
// and check all the source files within the tree with clang-format.
// so, disable it since the backend might have a different config.
// clang-format off

// NOTE: This condition is true for all PyTorch internal libraries, it
//       just excludes external projects such as torch_xla which
//       re-use some of the PyTorch codegen machinery.
#if defined(CAFFE2_BUILD_MAIN_LIB)        || \
    defined(TORCH_CUDA_BUILD_MAIN_LIB)    || \
    defined(TORCH_HIP_BUILD_MAIN_LIB)     || \
    defined(TORCH_CUDA_CU_BUILD_MAIN_LIB) || \
    defined(TORCH_CUDA_CPP_BUILD_MAIN_LIB)
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#endif

// ${generated_comment}

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>

#include <cstddef>                      // for std::size_t
#include <functional>                   // for std::function
#include <memory>                       // for std::shared_ptr, std::unique_ptr
#include <utility>                      // for std::move, std::pair

#include <ATen/Config.h>                // ATen configuration settings
#include <ATen/core/op_registration/adaption.h>  // for operator registration
#include <torch/library.h>              // for Torch library integration

$extra_cuda_headers                  // Additional CUDA headers
$external_backend_headers            // Headers specific to external backend
$dispatch_headers                    // Dispatch headers for operations
$ops_headers                         // Headers related to operations

// See template file RegisterDispatchDefinitions.ini
$dispatch_definitions                 // Dispatch definitions for template registration
```