# `.\pytorch\c10\util\FunctionRef.h`

```py
//===- llvm/ADT/STLExtras.h - Useful STL related functions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some templates that are useful if you are working with the
// STL at all.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//

// c10: modified from llvm::function_ref
// c10: added more SFINAE to enable use in overloaded functions

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace c10 {

/// An efficient, type-erasing, non-owning reference to a callable. This is
/// intended for use as the type of a function parameter that is not used
/// after the function in question returns.
///
/// This class does not own the callable, so it is not in general safe to store
/// a function_ref.
template <typename Fn>
class function_ref;

template <typename Ret, typename... Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params... params) = nullptr; // Function pointer to callback
  intptr_t callable{}; // Pointer to the callable object

  // Helper function to invoke the callable stored in `callable`
  template <typename Callable>
  static Ret callback_fn(intptr_t callable, Params... params) {
    return (*reinterpret_cast<Callable*>(callable))(std::forward<Params>(params)...);
  }

 public:
  function_ref() = default; // Default constructor

  // Constructor for initializing with a callable object
  template <typename Callable>
  function_ref(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      Callable&& callable,
      std::enable_if_t<
          !std::is_same_v<std::remove_reference_t<Callable>, function_ref>>* =
          nullptr,
      std::enable_if_t<std::is_convertible_v<
          typename std::invoke_result_t<Callable, Params...>,
          Ret>>* = nullptr)
      : callback(callback_fn<std::remove_reference_t<Callable>>), // Initialize callback function pointer
        callable(reinterpret_cast<intptr_t>(&callable)) {} // Store callable object pointer

  // Operator to invoke the stored callable with given parameters
  Ret operator()(Params... params) const {
    return callback(callable, std::forward<Params>(params)...);
  }

  // Conversion operator to check if the function_ref is valid
  operator bool() const {
    return callback;
  }
};

} // namespace c10
```