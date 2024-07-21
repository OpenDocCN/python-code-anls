# `.\pytorch\android\pytorch_android\src\androidTest\cpp\pytorch_jni_common_test.cpp`

```
// 包含 Google Test 框架的头文件，用于单元测试
#include <gtest/gtest.h>

// 包含 ATen 库的类型工厂头文件
#include <ATen/core/type_factory.h>

// 包含 PyTorch Android JNI 的公共头文件
#include "caffe2/android/pytorch_android/src/main/cpp/pytorch_jni_common.h"

// 使用 testing 命名空间，方便直接引用测试框架的函数和类
using namespace ::testing;

// 定义单元测试例程，测试 pytorch_jni_common 中的 newJIValueFromAtIValue 函数
TEST(pytorch_jni_common_test, newJIValueFromAtIValue) {
  // 创建一个 C++ 的字典对象，键是整数类型，值是字符串类型
  auto dict = c10::impl::GenericDict(
      c10::dynT<c10::IntType>(), c10::dynT<c10::StringType>());

  // 定义一个回调函数，返回一个空的 JNI 引用对象
  auto dictCallback = [](auto&&) {
    return facebook::jni::local_ref<pytorch_jni::JIValue>{};
  };

  // 期望不会抛出任何异常，调用 pytorch_jni::JIValue 的 newJIValueFromAtIValue 函数
  EXPECT_NO_THROW(pytorch_jni::JIValue::newJIValueFromAtIValue(
      dict, dictCallback, dictCallback));
}


这段代码是一个使用 Google Test 框架编写的 C++ 单元测试。它测试了 `pytorch_jni_common` 中的 `newJIValueFromAtIValue` 函数。注释解释了每个变量和函数的作用，以及测试的整体目的。
```