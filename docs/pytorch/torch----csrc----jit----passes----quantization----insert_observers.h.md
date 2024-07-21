# `.\pytorch\torch\csrc\jit\passes\quantization\insert_observers.h`

```py
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

namespace std {

template <>
struct hash<torch::jit::Module> {
  // 自定义哈希函数，用于计算torch::jit::Module对象的哈希值
  inline size_t operator()(const torch::jit::Module& arg) const {
    return std::hash<c10::intrusive_ptr<c10::ivalue::Object>>()(arg._ivalue());
  }
};

} // namespace std

namespace torch {
namespace jit {

using QConfig = std::tuple<Module, Module>;
using QConfigDict = std::unordered_map<std::string, std::optional<QConfig>>;

/** \brief 在指定的方法中为需要观察的张量插入观察模块和观察函数调用。
 *
 * 对于需要在方法中观察的每个张量，向输入模块插入观察模块，并在指定方法中添加观察函数的调用。
 *
 * \param module 输入模块
 * \param method_name 我们要为其插入观察器的方法名
 * \param qconfig_dict 指定每个模块如何量化的qconfig字典
 * \param inplace 是否对输入模块进行原地修改或克隆模块
 * \param is_dynamic 是否正在使用动态量化脚本
 */
TORCH_API Module InsertObservers(
    Module& module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    QuantType quant_type = QuantType::STATIC);

/** \brief 在指定的方法中为需要观察的张量插入观察模块和观察方法。
 *
 * 对于需要在方法中观察的每个张量，向输入模块插入观察模块，并在模块中添加observe_<method-name>方法，此方法是指定方法的克隆，其中添加了观察器的前向调用。
 *
 * \param module 输入模块
 * \param method_name 我们要为其插入观察器的方法名
 * \param qconfig_dict 指定每个模块如何量化的qconfig字典
 * \param inplace 是否对输入模块进行原地修改或克隆模块
 * \param is_dynamic 是否正在使用动态量化脚本
 */
TORCH_API Module InsertObserversForOnDevicePTQ(
    Module& module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    QuantType quant_type = QuantType::STATIC);

} // namespace jit
} // namespace torch
```