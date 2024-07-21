# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\arch\mma.h`

```
/**
 * @file
 * @brief Templates exposing architecture support for multiply-add operations
 */

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

// Tag which triggers MMA which will trigger
struct OpMultiplyAddDequantizeInterleavedBToA;

}  // namespace arch
}  // namespace cutlass
```