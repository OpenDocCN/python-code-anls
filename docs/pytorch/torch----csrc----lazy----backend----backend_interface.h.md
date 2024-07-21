# `.\pytorch\torch\csrc\lazy\backend\backend_interface.h`

```py
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <atomic>

namespace torch {
namespace lazy {

struct IrBuilder;

/**
 * Work in progress- don't treat this as a stable interface yet!
 */
class BackendImplInterface {
 public:
    // 默认虚析构函数
    virtual ~BackendImplInterface() = default;

    // Map a particular c10:: device to a concrete backend device
    // Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
    // virtual devices, meaning they may map to a gpu, tpu, etc. behind the
    // scenes. In the future, non-virtual c10:: devices may also use lazy tensors
    // through a mode, in which case these APIs should still work, but should be
    // identity mappings.
    virtual BackendDevice GetBackendDevice(c10::Device device) const = 0;

    // TODO(whc)
    // Additional APIs expected for supporting distributed training, to be
    // designed

    /**
     * Debug/Metrics
     * */

    //   virtual std::map<std::string, Metric> GetMetrics() const = 0;

    //   virtual MemoryInfo GetMemoryInfo(const std::string& device) = 0;

    // Get a textual representation of the computation backend
    virtual std::string GetComputationBackendText(
        const ComputationPtr computation) const = 0;
};

class TORCH_API BackendRegistrar {
 public:
  // Construct a backend registrar with the given backend implementation interface
  BackendRegistrar(const BackendImplInterface* backend_impl_interface);
};

// Check if a backend is available
TORCH_API bool hasBackend();

// Get the backend implementation interface
TORCH_API const BackendImplInterface* getBackend();

// Get the IR builder instance
TORCH_API const IrBuilder* getIrBuilder();

} // namespace lazy
} // namespace torch
```