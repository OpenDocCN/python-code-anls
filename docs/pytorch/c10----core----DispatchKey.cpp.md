# `.\pytorch\c10\core\DispatchKey.cpp`

```
// 包含C10库中的DispatchKey.h和DispatchKeySet.h头文件
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>

// 使用命名空间c10
namespace c10 {

// 将BackendComponent枚举类型转换为对应的字符串表示
const char* toString(BackendComponent t) {
  // 根据不同的BackendComponent枚举值返回相应的字符串表示
  switch (t) {
    case BackendComponent::CPUBit:
      return "CPUBit";
    case BackendComponent::CUDABit:
      return "CUDABit";
    case BackendComponent::HIPBit:
      return "HIPBit";
    case BackendComponent::XLABit:
      return "XLABit";
    case BackendComponent::LazyBit:
      return "LazyBit";
    case BackendComponent::MetaBit:
      return "MetaBit";
    case BackendComponent::XPUBit:
      return "XPUBit";
    case BackendComponent::IPUBit:
      return "IPUBit";
    case BackendComponent::MPSBit:
      return "MPSBit";
    case BackendComponent::HPUBit:
      return "HPUBit";
    case BackendComponent::VEBit:
      return "VEBit";
    case BackendComponent::MTIABit:
      return "MTIA";
    case BackendComponent::PrivateUse1Bit:
      return "PrivateUse1Bit";
    case BackendComponent::PrivateUse2Bit:
      return "PrivateUse2Bit";
    case BackendComponent::PrivateUse3Bit:
      return "PrivateUse3Bit";
    case BackendComponent::InvalidBit:
      return "InvalidBit";
    default:
      return "UNKNOWN_BACKEND_BIT";
  }
}

// 将DeviceType枚举类型转换为对应的BackendComponent枚举类型
BackendComponent toBackendComponent(DeviceType device_type) {
  // 根据不同的DeviceType枚举值返回相应的BackendComponent枚举值
  switch (device_type) {
#define DO_CASE(device, _)                          \
  case DeviceType::device: {                        \
    return toBackendComponent(DispatchKey::device); \
  }
    // 使用宏展开对所有后端设备类型进行处理，将其转换为对应的DispatchKey枚举类型，再调用toBackendComponent函数
    C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
#undef DO_CASE
    // 如果未匹配到任何有效的后端设备类型，则返回InvalidBit作为默认值
    default:
      return BackendComponent::InvalidBit;
  }
}

// 将DispatchKey枚举类型转换为对应的字符串表示
const char* toString(DispatchKey t) {
  // 根据不同的DispatchKey枚举值返回相应的字符串表示
  switch (t) {
    case DispatchKey::Undefined:
      return "Undefined";

    case DispatchKey::Dense:
      return "Dense";
    case DispatchKey::FPGA:
      return "FPGA";
    case DispatchKey::MAIA:
      return "MAIA";
    case DispatchKey::Vulkan:
      return "Vulkan";
    case DispatchKey::Metal:
      return "Metal";

    case DispatchKey::Lazy:
      return "Lazy";
    case DispatchKey::MPS:
      return "MPS";
    case DispatchKey::HPU:
      return "HPU";
    case DispatchKey::MTIA:
      return "MTIA";

    case DispatchKey::Quantized:
      return "Quantized";
    case DispatchKey::CustomRNGKeyId:
      return "CustomRNGKeyId";
    case DispatchKey::MkldnnCPU:
      return "MkldnnCPU";

    case DispatchKey::Sparse:
      return "Sparse";

    case DispatchKey::SparseCsr:
      return "SparseCsr";

    case DispatchKey::NestedTensor:
      return "NestedTensor";

    case DispatchKey::BackendSelect:
      return "BackendSelect";

    case DispatchKey::Python:
      return "Python";

    case DispatchKey::Fake:
      return "Fake";
    case DispatchKey::FuncTorchDynamicLayerBackMode:
      return "FuncTorchDynamicLayerBackMode";

    case DispatchKey::Functionalize:
      return "Functionalize";

    case DispatchKey::Named:
      return "Named";

    case DispatchKey::Conjugate:
      return "Conjugate";
    // 如果 DispatchKey 是 Negative，返回字符串 "Negative"
    case DispatchKey::Negative:
      return "Negative";

    // 如果 DispatchKey 是 ZeroTensor，返回字符串 "ZeroTensor"
    case DispatchKey::ZeroTensor:
      return "ZeroTensor";

    // 如果 DispatchKey 是 ADInplaceOrView，返回字符串 "ADInplaceOrView"
    case DispatchKey::ADInplaceOrView:
      return "ADInplaceOrView";

    // 如果 DispatchKey 是 AutogradOther，返回字符串 "AutogradOther"
    case DispatchKey::AutogradOther:
      return "AutogradOther";

    // 如果 DispatchKey 是 AutogradFunctionality，返回字符串 "AutogradFunctionality"
    case DispatchKey::AutogradFunctionality:
      return "AutogradFunctionality";

    // 如果 DispatchKey 是 AutogradNestedTensor，返回字符串 "AutogradNestedTensor"
    case DispatchKey::AutogradNestedTensor:
      return "AutogradNestedTensor";

    // 如果 DispatchKey 是 Tracer，返回字符串 "Tracer"
    case DispatchKey::Tracer:
      return "Tracer";

    // 如果 DispatchKey 是 AutocastCPU，返回字符串 "AutocastCPU"
    case DispatchKey::AutocastCPU:
      return "AutocastCPU";

    // 如果 DispatchKey 是 AutocastXPU，返回字符串 "AutocastXPU"
    case DispatchKey::AutocastXPU:
      return "AutocastXPU";

    // 如果 DispatchKey 是 AutocastIPU，返回字符串 "AutocastIPU"
    case DispatchKey::AutocastIPU:
      return "AutocastIPU";

    // 如果 DispatchKey 是 AutocastHPU，返回字符串 "AutocastHPU"
    case DispatchKey::AutocastHPU:
      return "AutocastHPU";

    // 如果 DispatchKey 是 AutocastCUDA，返回字符串 "AutocastCUDA"
    case DispatchKey::AutocastCUDA:
      return "AutocastCUDA";

    // 如果 DispatchKey 是 AutocastXLA，返回字符串 "AutocastXLA"
    case DispatchKey::AutocastXLA:
      return "AutocastXLA";

    // 如果 DispatchKey 是 AutocastPrivateUse1，返回字符串 "AutocastPrivateUse1"
    case DispatchKey::AutocastPrivateUse1:
      return "AutocastPrivateUse1";

    // 如果 DispatchKey 是 FuncTorchBatched，返回字符串 "FuncTorchBatched"
    case DispatchKey::FuncTorchBatched:
      return "FuncTorchBatched";

    // 如果 DispatchKey 是 BatchedNestedTensor，返回字符串 "BatchedNestedTensor"
    case DispatchKey::BatchedNestedTensor:
      return "BatchedNestedTensor";

    // 如果 DispatchKey 是 FuncTorchVmapMode，返回字符串 "FuncTorchVmapMode"
    case DispatchKey::FuncTorchVmapMode:
      return "FuncTorchVmapMode";

    // 如果 DispatchKey 是 Batched，返回字符串 "Batched"
    case DispatchKey::Batched:
      return "Batched";

    // 如果 DispatchKey 是 VmapMode，返回字符串 "VmapMode"
    case DispatchKey::VmapMode:
      return "VmapMode";

    // 如果 DispatchKey 是 FuncTorchGradWrapper，返回字符串 "FuncTorchGradWrapper"
    case DispatchKey::FuncTorchGradWrapper:
      return "FuncTorchGradWrapper";

    // 如果 DispatchKey 是 DeferredInit，返回字符串 "DeferredInit"
    case DispatchKey::DeferredInit:
      return "DeferredInit";

    // 如果 DispatchKey 是 PythonTLSSnapshot，返回字符串 "PythonTLSSnapshot"
    case DispatchKey::PythonTLSSnapshot:
      return "PythonTLSSnapshot";

    // 如果 DispatchKey 是 FuncTorchDynamicLayerFrontMode，返回字符串 "FuncTorchDynamicLayerFrontMode"
    // 注意 [Out-of-tree vmap+grad prototype] 说明
    case DispatchKey::FuncTorchDynamicLayerFrontMode:
      return "FuncTorchDynamicLayerFrontMode";

    // 如果 DispatchKey 是 TESTING_ONLY_GenericWrapper，返回字符串 "TESTING_ONLY_GenericWrapper"
    case DispatchKey::TESTING_ONLY_GenericWrapper:
      return "TESTING_ONLY_GenericWrapper";

    // 如果 DispatchKey 是 TESTING_ONLY_GenericMode，返回字符串 "TESTING_ONLY_GenericMode"
    case DispatchKey::TESTING_ONLY_GenericMode:
      return "TESTING_ONLY_GenericMode";

    // 如果 DispatchKey 是 PreDispatch，返回字符串 "PreDispatch"
    case DispatchKey::PreDispatch:
      return "PreDispatch";

    // 如果 DispatchKey 是 PythonDispatcher，返回字符串 "PythonDispatcher"
    case DispatchKey::PythonDispatcher:
      return "PythonDispatcher";

    // 如果 DispatchKey 是 Autograd，返回字符串 "Autograd"
    case DispatchKey::Autograd:
      return "Autograd";

    // 如果 DispatchKey 是 CompositeImplicitAutograd，返回字符串 "CompositeImplicitAutograd"
    case DispatchKey::CompositeImplicitAutograd:
      return "CompositeImplicitAutograd";

    // 如果 DispatchKey 是 CompositeImplicitAutogradNestedTensor，返回字符串 "CompositeImplicitAutogradNestedTensor"
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      return "CompositeImplicitAutogradNestedTensor";

    // 如果 DispatchKey 是 CompositeExplicitAutograd，返回字符串 "CompositeExplicitAutograd"
    case DispatchKey::CompositeExplicitAutograd:
      return "CompositeExplicitAutograd";

    // 如果 DispatchKey 是 CompositeExplicitAutogradNonFunctional，返回字符串 "CompositeExplicitAutogradNonFunctional"
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      return "CompositeExplicitAutogradNonFunctional";

    // 如果 DispatchKey 是 FuncTorchBatchedDecomposition，返回字符串 "FuncTorchBatchedDecomposition"
    case DispatchKey::FuncTorchBatchedDecomposition:
      return "FuncTorchBatchedDecomposition";
    # 默认情况处理：根据类型 t 转换为后端组件 bc 和功能键 fk
    default:
      auto bc = toBackendComponent(t);
      auto fk = toFunctionalityKey(t);

      # 根据功能键 fk 进行进一步处理
      switch (fk) {
// 定义宏，根据后端和功能返回字符串，形如 "功能后端"
#define ENTRY(backend, functionality)  \
  case BackendComponent::backend##Bit: \
    return #functionality #backend;

// 定义宏，遍历所有后端组件，并生成对应的返回字符串
#define FORALL_BC(dkname, prefix)                  \
  case DispatchKey::dkname:                        \
    switch (bc) {                                  \
      // 使用宏ENTRY生成每个后端组件对应的返回字符串
      C10_FORALL_BACKEND_COMPONENTS(ENTRY, prefix) \
      default:                                     \
        return #prefix "Undefined";                \
    }

// 遍历所有功能键，并生成对应的返回字符串
C10_FORALL_FUNCTIONALITY_KEYS(FORALL_BC)

// 处理默认情况，生成后端和功能均未定义的返回字符串
default:
  switch (bc) {
    // 使用宏ENTRY生成每个后端组件对应的返回字符串，后端为Unknown
    C10_FORALL_BACKEND_COMPONENTS(ENTRY, Unknown)
    default:
      return "UnknownUnknown";
  }

// 取消定义宏，结束ENTRY和FORALL_BC的作用域
#undef FORALL_BC
#undef ENTRY
```