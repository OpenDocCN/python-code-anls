# `.\pytorch\torch\csrc\jit\frontend\resolver.h`

```
#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/frontend/sugared_value.h>

namespace torch {
namespace jit {

/**
 * class Resolver
 *
 * Represents an "outer environment" in which we can look up names and return
 * a corresponding SugaredValue. This is used during compilation to resolve
 * references to names which are not defined internally to the graph.
 *
 * Example: PythonResolver looks at the enclosing Python scope for `name`.
 *
 * NOTE: When adding methods, keep this an abstract class (i.e. all new methods
 * should be purely virtual). Resist the urge to provide a default
 * implementation; you should explicitly think about how each resolver would
 * handle the method.
 */
struct Resolver {
  virtual ~Resolver() = default;

  // Resolve a given name to a SugaredValue. This takes the method `m` that the
  // caller is currently constructing, since we may need to insert nodes into
  // the graph to create a value.
  virtual std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) {
    return nullptr;
  }

  // Resolve `name` to a TypePtr.
  virtual TypePtr resolveType(const std::string& name, const SourceRange& loc) {
    return nullptr;
  }
};

// A resolver that only understands "torch.foo()" lookups.
struct NativeResolver : public Resolver {
  // Override of resolveValue method to handle specific "torch" namespace lookup.
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {
    // Check if the name is "torch"; if true, return a BuiltinModule instance for "aten".
    if (name == "torch") {
      return std::make_shared<BuiltinModule>("aten");
    }
    // Default: return nullptr if name is not recognized.
    return nullptr;
  }

  // Override of resolveType method to handle type resolution (not implemented).
  TypePtr resolveType(const std::string& name, const SourceRange& loc) override {
    return nullptr;
  }
};

// Function to create and return an instance of NativeResolver.
inline std::shared_ptr<NativeResolver> nativeResolver() {
  return std::make_shared<NativeResolver>();
}

} // namespace jit
} // namespace torch
```