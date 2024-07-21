# `.\pytorch\torch\csrc\jit\passes\device_type_analysis.cpp`

```py
// 包含 ATen 库的相关头文件
#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

// 包含 Torch 的 JIT 编译器相关头文件
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/device_type_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <memory>
#include <utility>

// Torch JIT 命名空间
namespace torch {
namespace jit {

// 匿名命名空间，用于内部函数和类型定义
namespace {

// 使用别名定义 Tensor 和 Device
using Tensor = at::Tensor;
using Device = at::Device;

// 定义传播规则的类型别名 PropRule，是一个接受 Node* 参数的函数对象，返回布尔值
using PropRule = std::function<bool(Node*)>;

/*
   设置设备类型的函数，接受一个 Value* 和一个可选的 Device，返回是否发生了改变

   如果发生了改变，将更新 Value* 对应的 TensorType 类型，返回 true，否则返回 false
*/
bool setDeviceType(Value* value, std::optional<Device> device) {
  auto tensor_type = value->type()->expect<TensorType>();
  bool changed = tensor_type->device() != device;
  if (changed) {
    value->setType(tensor_type->withDevice(device));
  }
  return changed;
}

/*
   设置节点输出的设备类型，接受一个 Node* 和一个可选的 Device，返回是否发生了改变

   遍历节点的所有输出，对每个输出应用 setDeviceType 函数，并根据是否有改变返回布尔值
*/
bool setReturnsToDevice(Node* n, std::optional<Device> device) {
  bool changed = false;
  for (Value* out : n->outputs()) {
    auto tensor_type = out->type()->cast<TensorType>();
    if (!tensor_type) {
      continue;
    }
    changed |= setDeviceType(out, device);
  }
  return changed;
}

/*
   返回一个设备类型传播规则的函数对象，接受一个 DeviceType，返回一个 lambda 函数

   Lambda 函数接受一个 Node* 参数，并应用 setReturnsToDevice 函数，传播设备类型并返回结果
*/
PropRule setReturnstoDeviceRule(DeviceType deviceType) {
  Device device = Device(deviceType);
  return [=](Node* n) { return setReturnsToDevice(n, device); };
}

/*
   自定义规则函数，用于处理多个输入参数可能存在设备类型不匹配的情况

   通过获取第一个输入参数的设备类型，然后应用 setReturnsToDevice 函数来传播设备类型
*/
bool returnFirstArgDeviceRule(Node* n) {
  auto tensor_type = n->inputs()[0]->type()->cast<TensorType>();
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  return setReturnsToDevice(n, tensor_type->device());
}

/*
   自定义规则函数，用于处理多个输入参数可能存在设备类型不匹配的情况

   通过获取第二个输入参数的设备类型，然后应用 setReturnsToDevice 函数来传播设备类型
*/
bool returnSecondArgDeviceRule(Node* n) {
  auto tensor_type = n->inputs()[1]->type()->cast<TensorType>();
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  return setReturnsToDevice(n, tensor_type->device());
}

/*
   判断是否是零维 CPU Tensor 的函数，接受一个 TensorType 智能指针，返回布尔值

   如果符号大小(rank)为零且设备是 CPU，则返回 true，否则返回 false
*/
bool isZerodimCPUTensor(std::shared_ptr<TensorType> tensor_type) {
  bool is_zerodim = tensor_type->symbolic_sizes().rank().value_or(-1) == 0;
  bool is_cpu = tensor_type->device() && tensor_type->device()->is_cpu();
  return is_zerodim && is_cpu;
}

/*
   无设备传播规则函数，接受一个 Node* 参数，返回布尔值

   如果可以验证所有输入设备匹配（排除 CPU 零维张量），则传播设备类型并返回 true
*/
bool propWithNoDevice(Node* n) {
  size_t input_num = 0;

  // 遍历节点的所有输入，检查是否有 TensorType 类型
  for (; input_num < n->inputs().size(); input_num++) {
    if (n->inputs()[input_num]->type()->cast<TensorType>()) {
      break;
    }
  }
  if (input_num == n->inputs().size()) {
    // 如果没有找到 TensorType 类型的输入，返回 false
    // 调用函数 setReturnsToDevice，将节点 n 的返回设备设置为 c10::nullopt
    return setReturnsToDevice(n, c10::nullopt);
  }

  // 获取输入节点的张量类型，并检查是否为零维 CPU 张量
  auto tensor_type = n->inputs()[input_num]->type()->expect<TensorType>();
  bool only_seen_cpu_zerodim = isZerodimCPUTensor(tensor_type);
  // 获取第一个输入节点的设备类型
  std::optional<Device> device = tensor_type->device();

  // 现在检查所有输入节点是否具有一致的设备类型
  for (input_num++; input_num < n->inputs().size(); input_num++) {
    // 获取当前输入节点的张量类型，并将其转换为 TensorType
    auto tensor_type = n->inputs()[input_num]->type()->cast<TensorType>();
    // 如果类型为空或者是零维 CPU 张量，则跳过
    if (!tensor_type || isZerodimCPUTensor(tensor_type)) {
      continue;
    }

    // 如果当前节点的设备类型与之前节点的设备类型不一致
    if (device != tensor_type->device()) {
      // 如果之前只看到过零维 CPU 张量，则更新设备类型，并取消该标记
      if (only_seen_cpu_zerodim) {
        device = tensor_type->device();
        only_seen_cpu_zerodim = false;
      } else {
        // 如果之前已经有不匹配的设备类型，则返回空设备，结束函数
        return setReturnsToDevice(n, c10::nullopt);
      }
    }
  }
  // 如果所有输入节点的设备类型都一致，则设置节点 n 的返回设备为当前设备类型
  return setReturnsToDevice(n, device);
}

bool defaultDeviceProp(Node* n) {
  // 检测操作是否具有设备对象参数
  // 因为设备对象参数会隐式地转换为设备类型
  auto schema = n->maybeSchema();
  if (!schema) {
    return false;
  }
  auto arguments = schema->arguments();
  for (size_t i = 0; i < arguments.size(); i++) {
    Argument& argument = arguments[i];
    if (DeviceObjType::get()->isSubtypeOf(argument.type())) {
      // 可选参数由 TorchScript 填充默认值
      auto input_val = toIValue(n->inputs().at(i));
      if (!input_val.has_value()) {
        // 如果存在动态设备类型，则无法传播
        return false;
      }
      if (input_val->isNone()) {
        continue;
      }
      if (!input_val->isDevice()) {
        // 对于联合类型，放弃传播
        return false;
      }
      TORCH_INTERNAL_ASSERT(input_val->isDevice())
      Device device = input_val->toDevice();
      return setReturnsToDevice(n, device);
    }
  }
  // 如果没有设备对象参数，则使用默认设备属性传播策略
  return propWithNoDevice(n);
}

struct DeviceTypePropagationPass : public PropertyPropBase {
  explicit DeviceTypePropagationPass(std::shared_ptr<Graph> graph)
      : PropertyPropBase(graph) {
    buildRuleRegistry();
  }

  // 如果至少有一个节点在张量节点上设置了其标量类型，则返回 true
  bool run() {
    propagateBlock(graph_->block(), false);
    return changed_;
  }

 private:
  void propagateNode(Node* n, bool _ = false) override {
    GRAPH_DEBUG("processNode");
    switch (n->kind()) {
      case prim::If:
        return processIf(n);
      case prim::Loop:
        return processLoop(n);
      case prim::CallMethod:
      case prim::CallFunction:
        return; // 暂时不处理
      default:
        break;
    }

    bool has_tensor_output =
        std::any_of(n->outputs().begin(), n->outputs().end(), [](Value* v) {
          return (bool)v->type()->cast<TensorType>();
        });

    if (!has_tensor_output) {
      // 如果输出不包含张量，则无需传播
      return;
    }

    switch (n->kind()) {
      case prim::Constant:
        // 已经由其他东西传播了
      case prim::ListConstruct:
      case prim::ListUnpack:
        return; // 暂时不处理
      default:
        if (n->kind().is_aten()) {
          return processAtenOps(n);
        } else {
          return; // 暂时不处理
        }
    }
  }

  void processAtenOps(Node* n) {
    GRAPH_DEBUG("processAtenOps");
    GRAPH_DEBUG("case = ", n->kind(), " ", *n);
    // 自定义规则匹配
    auto op = n->maybeOperator();
    if (!op) {
      return;
    }
    auto prop_fn = device_prop_registry_->find(*op);
    if (prop_fn) {
      PropRule rule = *prop_fn;
      changed_ |= rule(n);
      return;
    }
    changed_ |= defaultDeviceProp(n);
  }

  void buildRuleRegistry() {
    // 建立所有自定义设备类型规则的注册表
    if (device_prop_registry_)
      return;
    // 创建一个静态的映射表temp_registry，用于将运算符映射到属性规则
    static OperatorMap<PropRule> temp_registry{
        {"aten::cpu(Tensor self) -> Tensor",
         setReturnstoDeviceRule(DeviceType::CPU)},  // 将 "aten::cpu" 映射到设备类型为 CPU 的返回规则
        {"aten::cuda(Tensor self) -> Tensor",
         setReturnstoDeviceRule(DeviceType::CUDA)},  // 将 "aten::cuda" 映射到设备类型为 CUDA 的返回规则
        {"aten::to_mkldnn(Tensor self, ScalarType? dtype) -> Tensor",
         setReturnstoDeviceRule(DeviceType::MKLDNN)},  // 将 "aten::to_mkldnn" 映射到设备类型为 MKLDNN 的返回规则
        {"aten::reshape_as(Tensor self, Tensor other) -> Tensor",
         returnFirstArgDeviceRule},  // 将 "aten::reshape_as" 映射到第一个参数的设备类型返回规则
        {"aten::view_as(Tensor self, Tensor other) -> Tensor",
         returnFirstArgDeviceRule},  // 将 "aten::view_as" 映射到第一个参数的设备类型返回规则
        {"aten::expand_as(Tensor self, Tensor other) -> Tensor",
         returnFirstArgDeviceRule},  // 将 "aten::expand_as" 映射到第一个参数的设备类型返回规则
        {"aten::type_as(Tensor self, Tensor other) -> Tensor",
         returnSecondArgDeviceRule},  // 将 "aten::type_as" 映射到第二个参数的设备类型返回规则
    };
    // 将 temp_registry 转移所有权给 device_prop_registry_，创建一个唯一指针对象
    device_prop_registry_ =
        std::make_unique<OperatorMap<PropRule>>(std::move(temp_registry));
  }

  // 创建一个唯一指针对象 device_prop_registry_，用于存储设备属性注册表
  static std::unique_ptr<OperatorMap<PropRule>> device_prop_registry_;
  // 创建一个布尔变量 changed_，表示是否发生了改变，默认为 false
  bool changed_ = false;
};

// 结束了一个匿名命名空间

std::unique_ptr<OperatorMap<PropRule>>
    DeviceTypePropagationPass::device_prop_registry_ = nullptr;
// 初始化静态成员变量 device_prop_registry_

} // anonymous namespace
// 结束匿名命名空间

// 该分析通过图传播输入设备类型（如果有的话）
bool DeviceTypePropagation(std::shared_ptr<Graph>& graph) {
  // 创建设备类型传播通行证对象
  auto tp = std::make_unique<DeviceTypePropagationPass>((graph));
  // 运行设备类型传播通行证，并记录是否有改变
  bool changed = tp->run();
  // 如果图发生了改变
  if (changed) {
    // 打印传播后的张量属性传播通行证的图
    GRAPH_DUMP("After TensorPropertyPropagation pass:", graph);
  }
  // 返回是否有改变
  return changed;
}

// 结束了命名空间 jit
} // namespace torch
```