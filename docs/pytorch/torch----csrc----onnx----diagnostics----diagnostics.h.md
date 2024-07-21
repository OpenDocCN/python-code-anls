# `.\pytorch\torch\csrc\onnx\diagnostics\diagnostics.h`

```py
#pragma once
#include <torch/csrc/onnx/diagnostics/generated/rules.h>
#include <torch/csrc/utils/pybind.h>
#include <string>

namespace torch::onnx::diagnostics {

/**
 * @brief Level of a diagnostic.
 * @details The levels are defined by the SARIF specification, and are not
 * modifiable. For alternative categories, please use Tag instead.
 * @todo Introduce Tag to C++ api.
 */
enum class Level : uint8_t {
  kNone,                  // Diagnostic level representing no severity
  kNote,                  // Diagnostic level representing a note
  kWarning,               // Diagnostic level representing a warning
  kError,                 // Diagnostic level representing an error
};

static constexpr const char* const kPyLevelNames[] = {
    "NONE",                // Corresponding Python string for Level::kNone
    "NOTE",                // Corresponding Python string for Level::kNote
    "WARNING",             // Corresponding Python string for Level::kWarning
    "ERROR",               // Corresponding Python string for Level::kError
};

// Wrappers around Python diagnostics.
// TODO: Move to .cpp file in following PR.

/**
 * @brief Retrieves the Python diagnostics module.
 * @return Python object representing the 'torch.onnx._internal.diagnostics' module
 */
inline py::object _PyDiagnostics() {
  return py::module::import("torch.onnx._internal.diagnostics");
}

/**
 * @brief Retrieves the Python rule corresponding to the given C++ Rule.
 * @param rule The C++ enum Rule to convert to its Python counterpart.
 * @return Python object representing the rule in the diagnostics module
 */
inline py::object _PyRule(Rule rule) {
  return _PyDiagnostics().attr("rules").attr(
      kPyRuleNames[static_cast<uint32_t>(rule)]);
}

/**
 * @brief Retrieves the Python level corresponding to the given C++ Level.
 * @param level The C++ enum Level to convert to its Python counterpart.
 * @return Python object representing the level in the diagnostics module
 */
inline py::object _PyLevel(Level level) {
  return _PyDiagnostics().attr("levels").attr(
      kPyLevelNames[static_cast<uint32_t>(level)]);
}

/**
 * @brief Issues a diagnostic message based on the provided rule, level, and message arguments.
 * @param rule The rule to associate the diagnostic message with.
 * @param level The severity level of the diagnostic.
 * @param messageArgs Additional arguments for formatting the diagnostic message.
 */
inline void Diagnose(
    Rule rule,
    Level level,
    std::unordered_map<std::string, std::string> messageArgs = {}) {
  py::object py_rule = _PyRule(rule);            // Convert C++ rule to Python object
  py::object py_level = _PyLevel(level);         // Convert C++ level to Python object

  // Format the diagnostic message using Python rule and messageArgs
  py::object py_message =
      py_rule.attr("format_message")(**py::cast(messageArgs));

  // Import pybind11 literals for keyword arguments
  using namespace pybind11::literals;
  // Call the diagnose function in Python with rule, level, message, and cpp_stack=True
  _PyDiagnostics().attr("diagnose")(
      py_rule, py_level, py_message, "cpp_stack"_a = true);
}

} // namespace torch::onnx::diagnostics
```