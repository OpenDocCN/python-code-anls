# `.\pytorch\torch\csrc\jit\passes\quantization\finalize.h`

```
#pragma once

#include <torch/csrc/jit/api/module.h>  // 引入 Torch 模块 API 头文件
#include <torch/csrc/jit/ir/ir.h>       // 引入 Torch IR 头文件
#include <torch/csrc/jit/passes/quantization/quantization_type.h>  // 引入量化类型头文件

namespace torch {
namespace jit {

/** \brief 后端特定的 pass，用于融合反量化 - 操作 - 量化调用为量化操作调用。
 *
 * 目前此融合适用于 fbgemm 后端，仅适用于量化卷积操作，未来将扩展到更多操作和更多后端。
 *
 * 当前支持的融合：
 * q(conv2d(dq(a), dq(w), dq(b))) --> to_nchw(fbgemm_conv2d(prepack(to_nhwc(a)),
 *                                                          prepack(to_nhwc(w)),
 *                                                          prepack(to_nhwc(b))))
 *
 * q(linear(dq(a), dq(w), dq(b))) --> to_nchw(fbgemm_linear(prepack(to_nhwc(a)),
 *                                                          prepack(to_nhwc(w)),
 *                                                          prepack(to_nhwc(b))))
 *
 * \param graph 要应用融合的图
 * \param quant_type 量化类型，默认为静态量化
 */
TORCH_API void QuantFusion(
    std::shared_ptr<Graph>& graph,
    QuantType quant_type = QuantType::STATIC);

/** \brief 在图中插入预打包和解包函数。
 *  我们希望为量化权重添加打包/解包函数，因为稍后我们希望将打包的权重作为模块的属性折叠，
 *  以便在量化模型中减少动态打包权重的成本。
 *
 *  每个量化操作都有对应的预打包/解包函数，
 *  目前只需要为 quantized::linear 和 quantized::conv2d 进行预打包/解包。
 *
 * \param graph 要操作的图
 */
TORCH_API void InsertPrepackUnpack(std::shared_ptr<Graph>& graph);

/** \brief 在模块的所有图中插入打包和解包函数。
 *
 *   遍历所有子模块的所有方法的图，并在每个图上调用 InsertPrepackUnpack。
 *
 * \param module 要操作的模块
 */
TORCH_API void InsertPrepackUnpack(Module& module);

/** \brief 在模块上执行最终量化。
 *
 * \param module 要量化的模块
 * \param quant_type 量化类型，默认为静态量化
 * \param preserved_attrs 要保留的属性列表，默认为空
 * \return script::Module 经过最终处理的模块
 */
TORCH_API script::Module Finalize(
    script::Module& module,
    QuantType quant_type = QuantType::STATIC,
    const std::vector<std::string>& preserved_attrs =
        std::vector<std::string>());

/** \brief 折叠量化预打包操作。
 *
 * \param module 要操作的模块
 */
TORCH_API void FoldQuantizedPrepackingOps(Module& module);

/** \brief 在设备上执行最终 PTQ（量化训练）。
 *
 * \param module 要量化的模块
 * \param quant_type 量化类型
 * \param method_name 方法名称
 * \return Module 经过最终处理的模块
 */
TORCH_API Module FinalizeOnDevicePTQ(
    Module& module,
    QuantType quant_type,
    const std::string& method_name);

} // namespace jit
} // namespace torch
```