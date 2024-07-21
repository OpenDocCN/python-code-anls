# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\test\test_utils.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <gtest/gtest.h>

// 定义 qnnpack 命名空间
namespace qnnpack {
namespace testing {

// 定义测试模式枚举类型
enum class Mode {
  Static,    // 静态模式
  Runtime,   // 运行时模式
};

// 定义宏 _MAKE_TEST，用于生成测试用例
#define _MAKE_TEST(TestClass, test_name, test_body, ...)  \
  TEST(TestClass, test_name) {                            \
    test_body.testQ8(__VA_ARGS__);                        \
  }

// 定义宏 _STATIC_TEST，生成静态模式的测试用例
#define _STATIC_TEST(TestClass, test_name, test_body)                   \
  _MAKE_TEST(TestClass, test_name##_static, test_body, Mode::Static)

// 定义宏 _RUNTIME_TEST，生成运行时模式的测试用例
#define _RUNTIME_TEST(TestClass, test_name, test_body)                  \
  _MAKE_TEST(TestClass, test_name##_runtime, test_body, Mode::Runtime)

// 定义宏 _STATIC_AND_RUNTIME_TEST，生成同时支持静态和运行时模式的测试用例
#define _STATIC_AND_RUNTIME_TEST(TestClass, test_name, test_body) \
  _STATIC_TEST(TestClass, test_name, test_body)                   \
  _RUNTIME_TEST(TestClass, test_name, test_body)

}}  // namespace qnnpack::testing


这段代码定义了一些宏和枚举，用于在 C++ 的测试框架中生成静态和运行时模式的测试用例。
```