# `.\pytorch\torch\csrc\jit\passes\hoist_conv_packed_params.cpp`

```
#include <stack>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/hoist_conv_packed_params.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace torch {
namespace jit {

// Hoists packed params from a conv module to the parent module.
// The benefit is that after this hoisting, the conv module
// no longer holds anything and can be deleted, reducing model
// size.
//
// Before (easy case):
//
// %1 = prim::GetAttr[name="conv1"][%self]
// %2 = prim::GetAttr[name="_packed_params][%1]
//
// After (easy case):
//
// %2 = prim::GetAttr[name="{prefix}.conv1._packed_params"][%self]
//
// Before (generic case):
//
// %1 = prim::GetAttr[name="name1"][%self]
// %2 = prim::GetAttr[name="name2"][%1]
// ...
// %n = prim::GetAttr[name="_packed_params][%n-1]
//
// After (generic case):
//
// %n =
// prim::GetAttr[name="{prefix}.name1{...}.name(n-1)._packed_params"][%self]
//
// Static function to hoist packed parameters from a conv module to the parent module.
static void hoistConvPackedParams(
    Module& rootModule,                    // Parent module to hoist packed parameters into
    Node* getConvPackedParamsNode,         // Node representing the operation to get packed params
    const std::string& prefix,             // Prefix to prepend to new attribute name
    int& nameUniqueCounter) {              // Counter to ensure uniqueness of new attribute names

  // Retrieve the 'forward' method and its graph from the root module
  auto method = rootModule.get_method("forward");
  auto graph = method.graph();
  // The root module itself represented as a graph input value
  Value* rootModuleAsValue = graph->inputs()[0];

  // Get the conv module's value by tracing back from the getConvPackedParamsNode
  Value* convModuleAsValue = getConvPackedParamsNode->inputs()[0];
  // Compute the access path from root module to conv module
  std::vector<std::string> rootToConvPath =
      getModuleAccessPath(convModuleAsValue, rootModuleAsValue);

  // Find the actual conv module object within the root module using the access path
  Module convModule = findChildModule(rootModule, rootToConvPath);

  // Retrieve the packed params value from the conv module
  c10::IValue packedParams = convModule.attr("_packed_params");

  // Construct a new unique name for the packed params attribute based on the prefix
  std::string suffix = "";
  for (const auto& attrName : rootToConvPath) {
    suffix += attrName + ".";
  }
  std::string newNameBase = prefix + "." + suffix + "_packed_params";
  nameUniqueCounter++;
  std::string newName = newNameBase + "." + std::to_string(nameUniqueCounter);
  // Ensure the new name is unique within the root module
  while (rootModule.hasattr(newName)) {
    nameUniqueCounter++;
    newName = newNameBase + "." + std::to_string(nameUniqueCounter);
  }

  // Register the packed params attribute with the new name in the root module
  rootModule.register_attribute(newName, packedParams.type(), packedParams);

  // Redirect the getConvPackedParamsNode to use the root module as its target module
  getConvPackedParamsNode->replaceInput(0, rootModuleAsValue);

  // Update the attribute name in the getConvPackedParamsNode to the newly created name
  getConvPackedParamsNode->s_(Symbol::attr("name"), newName);
}

// Function to hoist packed parameters from convolutional modules in a script::Module
void HoistConvPackedParams(script::Module& m) {
  auto method = m.get_method("forward");
  auto graph = method.graph();

  // Initialize a stack to visit blocks within the method's graph
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  // Base name for new attribute names
  std::string attr_name_base = "_jit_pass_hoist_conv_packed_params";
  // Counter to ensure new attribute names are unique
  int nameUniqueCounter = 0;

  // Traverse through all blocks in the graph
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    // 遍历块b中的每个节点n
    for (Node* n : b->nodes()) {
        // 确保此节点正在获取 {foo}.{_packed_params}
        bool isGetPackedParamsNode =
            n->kind() == prim::GetAttr && n->s(attr::name) == "_packed_params";
        
        // 如果是获取 _packed_params 的节点
        if (isGetPackedParamsNode) {
            // 确保 {foo} 是量化的卷积操作
            std::optional<std::string> moduleName = getModuleName(n->inputs()[0]);
            bool moduleNameIsQuantizedConv = moduleName.has_value() &&
                (moduleName.value() ==
                     "__torch__.torch.ao.nn.quantized.modules.conv.Conv1d" ||
                 moduleName.value() ==
                     "__torch__.torch.ao.nn.quantized.modules.conv.Conv2d" ||
                 moduleName.value() ==
                     "__torch__.torch.ao.nn.quantized.modules.conv.Conv3d" ||
                 moduleName.value() ==
                     "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU1d" ||
                 moduleName.value() ==
                     "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d" ||
                 moduleName.value() ==
                     "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU3d" ||
                 // BC Stuff
                 moduleName.value() ==
                     "__torch__.torch.nn.quantized.modules.conv.Conv1d" ||
                 moduleName.value() ==
                     "__torch__.torch.nn.quantized.modules.conv.Conv2d" ||
                 moduleName.value() ==
                     "__torch__.torch.nn.quantized.modules.conv.Conv3d");
            
            // 如果 {foo} 是量化的卷积操作
            if (moduleNameIsQuantizedConv) {
                // 输出日志，表示将节点 *n 提升到根模块
                GRAPH_UPDATE("Hoisting ", *n, " to root module.");
                // 将量化卷积操作的打包参数提升到根模块
                hoistConvPackedParams(m, n, attr_name_base, nameUniqueCounter);
            }
        }
        
        // 将节点n的子块加入待访问的块列表中
        for (Block* subblock : n->blocks()) {
            blocks_to_visit.push(subblock);
        }
        
    } // for

} // while
} // 关闭 jit 命名空间

} // 关闭 torch 命名空间
```