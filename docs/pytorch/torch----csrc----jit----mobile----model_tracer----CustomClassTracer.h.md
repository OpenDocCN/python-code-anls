# `.\pytorch\torch\csrc\jit\mobile\model_tracer\CustomClassTracer.h`

```py
#pragma once

#include <ATen/record_function.h>
#include <c10/util/Synchronized.h>
#include <map>
#include <set>
#include <string>

namespace torch {
namespace jit {
namespace mobile {

/* The CustomClassTracer class handles the attachment and removal of a recording
 * callback that traces the invocation of code that handles loading custom
 * classes on mobile.
 *
 * You can get the set of used custom classes using
 * getLoadedClasses().
 *
 * Note: This class is not thread safe or re-entrant, and should not be used
 * across multiple threads of execution.
 *
 */
struct CustomClassTracer final {
  at::CallbackHandle handle_;  // Callback handle to manage the attachment and removal of the tracing callback

  /* These are the custom class names (constant
   * character string) which shows up in code.
   */
  typedef std::set<std::string> custom_classes_type;  // Define a type for a set of custom class names

  CustomClassTracer();  // Constructor declaration for CustomClassTracer

  // Static method to access the synchronized set of loaded custom classes
  static c10::Synchronized<custom_classes_type>& getLoadedClasses();

  ~CustomClassTracer() {
    at::removeCallback(handle_);  // Destructor removes the callback handle when the object is destroyed
  }
};

} // namespace mobile
} // namespace jit
} // namespace torch
```