# `.\pytorch\test\edge\operator_registry.h`

```py
#pragma once

#include <cstring>  // C-style string manipulation functions
#include <functional>  // std::function for defining callable objects
#include <map>  // STL map container for mapping kernel names to functions

#include "Evalue.h"  // Header file for EValue class
#include "kernel_runtime_context.h"  // Header file for KernelRuntimeContext class

#include <c10/util/ArrayRef.h>  // ArrayRef utility from c10 namespace

namespace torch {
namespace executor {

using KernelFunction = std::function<void(KernelRuntimeContext&, EValue**)>;  // Alias for a function type

template<typename T>
using ArrayRef = at::ArrayRef<T>;  // Alias for ArrayRef from at namespace in c10 library

#define EXECUTORCH_SCOPE_PROF(x)  // Macro definition, likely for profiling purposes

struct Kernel {
  const char* name_;  // Pointer to a constant C-style string (name of the kernel)
  KernelFunction kernel_;  // Function pointer type for the kernel function

  Kernel() = default;  // Default constructor

  /**
   * Constructor initializing a Kernel object with a name and a function pointer.
   * Note: It stores the pointer to the name, expecting the name's lifetime to
   * be at least as long as the KernelRegistry.
   */
  explicit Kernel(const char* name, KernelFunction func)
      : name_(name), kernel_(func) {}
};

/**
 * Checks if a kernel function with the given name is registered in the KernelRegistry.
 */
bool hasKernelFn(const char* name);

/**
 * Retrieves the kernel function associated with the given name from the KernelRegistry.
 */
KernelFunction& getKernelFn(const char* name);

/**
 * Registers a set of kernels defined in the provided ArrayRef into the KernelRegistry.
 * Returns true if successful.
 */
[[nodiscard]] bool register_kernels(const ArrayRef<Kernel>&);

struct KernelRegistry {
 public:
  KernelRegistry() : kernelRegSize_(0) {}  // Constructor initializing kernel registry size to zero

  /**
   * Registers a set of kernels defined in the provided ArrayRef into the KernelRegistry.
   * Returns true if successful.
   */
  bool register_kernels(const ArrayRef<Kernel>&);

  /**
   * Checks whether a kernel with a given name is registered in the KernelRegistry.
   * Returns true if found, false otherwise.
   */
  bool hasKernelFn(const char* name);

  /**
   * Retrieves the kernel function associated with the given name from the KernelRegistry.
   * If the kernel is not found, behavior is undefined.
   * Returns a reference to the kernel function.
   */
  KernelFunction& getKernelFn(const char* name);

 private:
  std::map<const char*, KernelFunction> kernels_map_;  // Map storing kernel names and corresponding functions
  uint32_t kernelRegSize_;  // Size of the kernel registry (number of registered kernels)
};

} // namespace executor
} // namespace torch
```