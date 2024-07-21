# `.\pytorch\aten\src\ATen\core\GeneratorForPrivateuseone.h`

```py
#pragma once


// 预处理指令：#pragma once，确保头文件只被编译一次，用于防止多重包含。

#include <ATen/core/Generator.h>
#include <c10/util/intrusive_ptr.h>


namespace at {


// 命名空间 at 开始

using GeneratorFuncType = std::function<at::Generator(c10::DeviceIndex)>;


// GeneratorFuncType 是一个函数类型别名，用于定义生成器函数的类型。

std::optional<GeneratorFuncType>& GetGeneratorPrivate();


// GetGeneratorPrivate 函数声明，返回一个可选的 GeneratorFuncType 引用。

class TORCH_API _GeneratorRegister {
 public:
  explicit _GeneratorRegister(const GeneratorFuncType& func);
};


// _GeneratorRegister 类声明，用于注册生成器函数的类。

TORCH_API at::Generator GetGeneratorForPrivateuse1(
    c10::DeviceIndex device_index);


// GetGeneratorForPrivateuse1 函数声明，根据设备索引返回生成器对象。

/**
 * This is used to register Generator to PyTorch for `privateuse1` key.
 *
 * Usage: REGISTER_GENERATOR_PRIVATEUSE1(MakeGeneratorForPrivateuse1)
 *
 * class CustomGeneratorImpl : public c10::GeneratorImpl {
 *   CustomGeneratorImpl(DeviceIndex device_index = -1);
 *   explicit ~CustomGeneratorImpl() override = default;
 *   ...
 * };
 *
 * at::Generator MakeGeneratorForPrivateuse1(c10::DeviceIndex id) {
 *   return at::make_generator<CustomGeneratorImpl>(id);
 * }
 */


// 注释块，解释了 REGISTER_GENERATOR_PRIVATEUSE1 宏的用法和相关函数实现的示例。

#define REGISTER_GENERATOR_PRIVATEUSE1(GeneratorPrivate) \
  static auto temp##GeneratorPrivate = at::_GeneratorRegister(GeneratorPrivate);


// REGISTER_GENERATOR_PRIVATEUSE1 宏定义，用于静态注册生成器函数。

} // namespace at


// 命名空间 at 结束
```