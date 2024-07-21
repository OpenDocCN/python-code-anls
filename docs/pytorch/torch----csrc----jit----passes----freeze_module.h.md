# `.\pytorch\torch\csrc\jit\passes\freeze_module.h`

```py
/** \brief This file defines freezing Torchscript module API.
 *
 * This API has python-binding and can be invoked directly or as a part of
 * general optimization pipeline.
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

/** \brief Freeze Module, i.e., Assume all attributes are constants.
 *
 * Freezing module is a functionality that allows the JIT to internalize
 * immutable attributes. Combined with inlining, the module is aggressively
 * optimized and significant overhead is optimized away. The freezeModule API
 * produces a cloned frozen module.
 */

namespace torch {
namespace jit {

/**
 * \brief Freeze a Torchscript module.
 *
 * This function takes a Torchscript module and freezes it, assuming all
 * attributes are constants. It optionally preserves specified attributes,
 * freezes interfaces, and can preserve parameters if specified.
 *
 * \param module The input module to be frozen.
 * \param preservedAttrs Vector of attribute names to be preserved as constants.
 * \param freezeInterfaces Flag indicating whether to freeze interfaces.
 * \param preserveParameters Flag indicating whether to preserve parameters.
 * \return A new cloned frozen module after freezing.
 */
TORCH_API Module freeze_module(
    const Module& module,
    std::vector<std::string> preservedAttrs = std::vector<std::string>(),
    bool freezeInterfaces = true,
    bool preserveParameters = false);

/**
 * \brief Inplace version of freeze_module.
 *
 * This function freezes the given Torchscript module inplace, modifying it
 * directly. It assumes all attributes are constants, optionally preserves
 * specified attributes, freezes interfaces, and can preserve parameters if
 * specified.
 *
 * \param module Pointer to the module to be frozen inplace.
 * \param preservedAttrs Vector of attribute names to be preserved as constants.
 * \param freezeInterfaces Flag indicating whether to freeze interfaces.
 * \param preserveParameters Flag indicating whether to preserve parameters.
 */
TORCH_API void freeze_module_inplace(
    Module* module,
    std::vector<std::string> preservedAttrs = std::vector<std::string>(),
    bool freezeInterfaces = true,
    bool preserveParameters = false);

} // namespace jit
} // namespace torch
```