# `.\pytorch\torch\csrc\jit\passes\quantization\insert_quant_dequant.h`

```
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

namespace torch {
namespace jit {

/** 
 * \brief 复制量化节点以用于 prim::If 块，以便匹配量化模式
 * 
 * \param graph 要操作的图对象的指针
 */
TORCH_API void ReplicateQuant(std::shared_ptr<Graph>& graph);

/** 
 * \brief 复制反量化节点以便每次使用都能匹配量化模式
 * 
 * \param graph 要操作的图对象的指针
 */
TORCH_API void ReplicateDeQuant(std::shared_ptr<Graph>& graph);

/** 
 * \brief 在插入观察器 pass 后，为观察到的 Tensor 插入量化-反量化调用
 * 
 * 对于每个被观察到的 Tensor，获取观察器模块并调用其 calculate_qparam 方法获取量化参数。
 * 使用这些参数添加量化-整数表示-反量化函数调用。目前也有针对"bias"量化的特殊处理。
 * 
 * \param module 输入的模块
 * \param method_name 要为其插入量化调用的方法名称
 * \param inplace 是否原地修改
 * \param debug 是否启用调试模式
 * \param quant_type 量化类型，默认为静态量化
 * \return 修改后的模块
 */
TORCH_API Module InsertQuantDeQuant(
    Module& module,
    const std::string& method_name,
    bool inplace,
    bool debug,
    QuantType quant_type = QuantType::STATIC);

/** 
 * \brief 在设备上执行 PTQ（量化推理）时，为模块的指定方法插入量化-反量化调用
 * 
 * \param module 输入的模块
 * \param method_name 要为其插入量化调用的方法名称
 * \param inplace 是否原地修改
 * \param debug 是否启用调试模式
 * \param quant_type 量化类型，默认为静态量化
 * \return 修改后的模块
 */
TORCH_API Module InsertQuantDeQuantOnDevicePTQ(
    Module& module,
    const std::string& method_name,
    bool inplace,
    bool debug,
    QuantType quant_type = QuantType::STATIC);

} // namespace jit
} // namespace torch
```