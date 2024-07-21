# `.\pytorch\torch\csrc\jit\mobile\import_data.h`

```py
#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/mobile/module.h>

#include <istream>
#include <map>
#include <string>

namespace torch {
namespace jit {

/**
 * 从输入流 @p in 中加载序列化数据中的命名参数。
 * 如果数据格式不被识别，则调用 #TORCH_CHECK()。
 */
TORCH_API std::map<std::string, at::Tensor> _load_parameters(
    std::istream& in,
    std::optional<at::Device> device = c10::nullopt);

/**
 * 从文件名 @p filename 中加载序列化数据中的命名参数。
 * 如果数据格式不被识别，则调用 #TORCH_CHECK()。
 */
TORCH_API std::map<std::string, at::Tensor> _load_parameters(
    const std::string& filename,
    std::optional<at::Device> device = c10::nullopt);

// 注意：请优先使用 _load_parameters，而不要使用下面的函数。
/**
 * 将移动模块 @p module 转换为参数映射。
 */
TORCH_API std::map<std::string, at::Tensor> mobile_module_to_parameter_map(
    const mobile::Module& module);

} // namespace jit
} // namespace torch
```