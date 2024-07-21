# `.\pytorch\torch\csrc\jit\passes\quantization\helper.h`

```py
#pragma once
// 包含 Torch 的 JIT 模块 API 头文件
#include <torch/csrc/jit/api/module.h>
// 包含 Torch 的 JIT IR 头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT IR 子图匹配器头文件
#include <torch/csrc/jit/ir/subgraph_matcher.h>
// 包含 Torch 的 JIT 图重写帮助函数头文件
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
// 包含 Torch 的 JIT 量化类型头文件
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

// 包含标准函数库
#include <functional>
// 包含正则表达式库
#include <regex>

// Torch 的 JIT 命名空间
namespace torch {
namespace jit {

// 使用 graph_rewrite_helper 命名空间中的 getFuncName 函数
using graph_rewrite_helper::getFuncName;

// 定义一个模块和其方法名称的向量
using ModuleMethodVector = std::vector<std::pair<Module, std::string>>;
// 定义量化参数名和值的映射的向量
using QParamVector = std::vector<std::pair<std::string, IValue>>;

// =========== Value 的辅助函数 =========
// 检查值是否为权重，因为在权重观察中需要使用权重
TORCH_API bool isWeight(Value* v);

// 检查值是否为卷积和线性层的偏置，这些值不进行量化
TORCH_API bool isBiasOfConvOrLinear(Value* v);

// 检查值是否为非输入的 EmbeddingBag
TORCH_API bool isEmbeddingBagNonInput(Value* v);

// 获取 clamp 操作的输入值作为标量输入的使用情况
std::optional<Use> getClampScalarInputUse(Value* v);

// 对于给定的值 `v`，获取我们需要检查其是否被观察/量化的值列表，
// 如果是，我们可以推断出给定值 `v` 也被观察/量化了，因为我们可以根据值列表推导出 `v` 的量化参数
TORCH_API std::vector<Value*> getPassThroughInputs(Value* v);

// 通过原始方法名克隆方法到新的方法名
TORCH_API void cloneMethod(
    Module& module,
    const std::string& orig_method_name,
    const std::string& new_method_name);

// 检查图中的值是否为标量值
TORCH_API bool isScalar(Value* v);

// 检查值是否为图的输入
TORCH_API bool hitGraphInput(Value* value);

// 将编码名称（如 __torch__.torch.ao.nn.quantized.modules.conv.___torch_mangle_7.Conv2d）
// 转换为非编码名称（如 __torch__.torch.ao.nn.quantized.modules.conv.Conv2d）
TORCH_API std::string removeTorchMangle(const std::string& orig_name);

// 返回与给定值对应的模块名称
TORCH_API std::optional<std::string> getModuleName(Value* value);

// =========== Node 的辅助函数 =========
// 检查节点是否为单输入的一般形状 Aten 函数
TORCH_API bool isSingleInputGeneralShapeAtenFunction(Node* n);

// 检查节点是否为单输入的一般值 Aten 函数
TORCH_API bool isSingleInputGeneralValueAtenFunction(Node* n);

// 检查节点是否为单输入的一般调用函数
TORCH_API bool isSingleInputGeneralCallFunction(Node* n);

// 检查节点是否为单输入的一般 Aten 函数
TORCH_API bool isSingleInputGeneralAtenFunction(Node* n);

// 检查节点是否为 Clamp 操作
TORCH_API bool isClamp(Node* n);

// 检查节点是否会产生相同的结果，无论输入张量是否量化，例如：aten::size
TORCH_API bool isTensorInfoNode(Node* n);

// 检查这是否是具有单输入的传播操作，例如：aten::cat
TORCH_API bool isPropagateQuantSingleInputOp(Node* n);

// 检查这是否是具有两个输入的传播操作，例如：aten::add
// Check if the node `n` represents a binary operation that propagates quantization for binary inputs
TORCH_API bool isPropagateQuantBinaryOp(Node* n);

// Check if the node `n` represents an operation that propagates quantization based on input quantization
// Example: aten::cat
TORCH_API bool isPropagateQuantOp(Node* n);

// Check if the node `n` represents a binary operation with scalar input, like aten::add or aten::mul,
// which will be quantized to quantized::{op}_scalar if input 1 is a scalar
TORCH_API bool isBinaryOpWithScalarInput(Node* n);

// Retrieve fixed quantization parameters for the node `n`, if available
TORCH_API std::optional<std::tuple<c10::QScheme, QParamVector>> getFixedQParams(Node* n);

// Determine if the node `n` represents a user-defined CallFunction that should not be graph-analyzed,
// preserving operation boundaries, e.g., 'linear'
TORCH_API bool userDefinedCallFunction(Node* n);

// Check if the node `n` has scalar input
TORCH_API bool hasScalarInput(Node* n);

// Check if the node `n` is quantizable under the specified quantization type
TORCH_API bool nodeQuantizable(Node* n, QuantType quant_type = QuantType::STATIC);

// Determine if the node `n` only requires quantization of weight values, e.g., 'embedding_bag'
bool isWeightOnlyStaticQuantOp(Node* n);

// Check if a specific use of a value is quantizable, dependent on both the use node and offset
TORCH_API bool useQuantizable(const Use& use, QuantType quant_type);

// Retrieve the graph of the called function from a CallFunction node `n`
TORCH_API std::shared_ptr<Graph> getCallFunctionGraph(Node* n);

// Check if a specific use matches a CallFunction of name `func_name`, with `nth_arg` as its nth argument
bool matchCallFuncToUse(const Use& use, const std::string& func_name, std::optional<int> nth_arg);

// Check if a specific use matches an AtenFunction of name `func_name`, with `nth_arg` as its nth argument
bool matchAtenFuncToUse(const Use& use, const std::string& func_name, std::optional<int> nth_arg);

// ========== helper functions for Block ==========

// Check if a Block `block` will always raise an exception
TORCH_API bool alwaysRaisesException(Block* block);

// ========== helper functions for Module ==========

// TODO: remove
// Retrieve the access path of a Module `instance` within `self`
TORCH_API std::vector<std::string> getModuleAccessPath(Value* instance, Value* self);

// TODO: remove
// Find a child Module within `module` specified by `path`
TORCH_API Module findChildModule(const Module& module, const std::vector<std::string>& path);

// Given a CallMethod node `n`, retrieve the Module instance corresponding to the instance Value `self`
// TODO: refactor all current uses of this function to the Opt one
TORCH_API Module getInvokedModule(Module& module, Node* n, Value* self);

// Given a CallMethod node `n`, retrieve the Module instance corresponding to the instance Value `self`
// if `self` is a module; otherwise, return c10::nullopt
std::optional<Module> getInvokedModuleOpt(const Module& module, Node* n, Value* self);

// ========== filter functions for matches ==========

// Filter to check if Value `vname` is a constant integer with value `value`
bool is_int_constant(const Match& match, const std::unordered_map<std::string, Value*>& vmap,
    // 声明一个名为 vname 的常量引用，类型为 std::string，表示函数的第一个参数，用于传递字符串数据
    const std::string& vname,
    // 声明一个整型参数 value，表示函数的第二个参数，用于传递整数数据
    int value);
// 定义一个函数，用于检查 aten::add 的 %alpha 参数是否为常数 1
bool aten_add_alpha_is_one(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 CallFunction 中的 functional 是否为 relu
bool is_functional_relu(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为 torch.nn.ReLU
bool is_relu_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为线性模块
bool is_linear_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// TODO: 添加一个宏来声明这些过滤器

// 定义一个函数，用于检查 module 是否为 conv1d 模块
bool is_conv1d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为 conv2d 模块
bool is_conv2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为 conv3d 模块
bool is_conv3d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为 conv_transpose1d 模块
bool is_conv_transpose1d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为 conv_transpose2d 模块
bool is_conv_transpose2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为 batchnorm2d 模块
bool is_batchnorm2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 定义一个函数，用于检查 module 是否为 batchnorm3d 模块
bool is_batchnorm3d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

} // namespace jit
} // namespace torch
```