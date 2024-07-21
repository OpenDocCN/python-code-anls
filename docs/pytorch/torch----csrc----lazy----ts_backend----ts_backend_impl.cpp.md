# `.\pytorch\torch\csrc\lazy\ts_backend\ts_backend_impl.cpp`

```
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
// 引入 TorchScript 后端的实现头文件

#include <ATen/Functions.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/generated/LazyNativeFunctions.h>
#include <torch/csrc/lazy/ts_backend/config.h>
#include <torch/csrc/lazy/ts_backend/ir_builder.h>
#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
// 引入其他相关的头文件

#include <memory>

namespace at {
// 命名空间 at，包含 TorchScript 相关的声明

// 声明一个在 RegisterDispatchKey.cpp 文件中生成的函数，用于注册分发键
// 对于 TorchScript 后端，有一个特殊情况，即注册不会立即发生（在静态初始化时），这样如果加载了外部后端，它有机会注册自己，
// 而 TorchScript 仅在显式初始化时注册自己
extern TORCH_API void RegisterTorchScriptLazyNativeFunctions();
extern TORCH_API void RegisterTorchScriptAutogradLazyNativeFunctions();
} // namespace at

namespace torch {
namespace lazy {

// TSBackendDeviceType 类继承自 BackendDeviceType，用于 TorchScript 后端设备类型
struct TSBackendDeviceType : public BackendDeviceType {
  TSBackendDeviceType() = delete;
  TSBackendDeviceType(c10::DeviceType deviceType)
      : BackendDeviceType((int8_t)deviceType) {
    TORCH_CHECK(deviceType == at::kCPU || deviceType == at::kCUDA);
  }

  std::string toString() const override {
    return c10::DeviceTypeName((c10::DeviceType)type);
  }

  c10::DeviceType c10Type() const {
    return (c10::DeviceType)type;
  }
};

// TSBackendImpl 类实现了 BackendImplInterface 接口，是 TorchScript 后端的具体实现
class TSBackendImpl : public torch::lazy::BackendImplInterface {
 public:
  TSBackendImpl() {
    // 构造函数，根据环境变量和标志位决定默认的设备类型
    // TODO(whc) unify how all our flags are set and parsed as envs
    static bool env_use_cuda = std::getenv("LTC_TS_CUDA") != nullptr;
    auto type =
        (env_use_cuda || FLAGS_torch_lazy_ts_cuda) ? at::kCUDA : at::kCPU;
    default_device_type_ = std::make_shared<TSBackendDeviceType>(type);
  }

  const IrBuilder* GetIrBuilder() const override {
    // 获取 IR 构建器的实例，这里使用了 TorchScriptIrBuilder
    static const IrBuilder* builder = new TorchScriptIrBuilder();
    return builder;
  }

  std::string CreateMetricReport() const override {
    // 创建一个指标报告字符串，这里返回 "TSBackendImpl: N/A"
    return "TSBackendImpl: N/A";
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status) const override {
    // 创建一个降低上下文的唯一指针，这里使用了 TSLoweringContext 类
    return std::make_unique<torch::lazy::TSLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device) const override {
    // 创建一个降低上下文的唯一指针，这里使用了 TSLoweringContext 类
    return std::make_unique<torch::lazy::TSLoweringContext>(name, device);
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
  // 将设备列表转换为包含设备名称的字符串向量，并返回
  return std::vector<std::string>(devices.begin(), devices.end());
}

at::Tensor MakeTensorFromComputationData(
    const torch::lazy::BackendDataPtr data,
    std::optional<at::ScalarType> logical_scalar_type) const override {
  // 将数据指针转换为 TSData 类型的指针
  const auto ts_data = std::static_pointer_cast<TSData>(data);
  // 返回 TSData 对象中的数据张量
  return ts_data->data();
}

torch::lazy::BackendDataPtr MakeComputationDataFromTensor(
    const at::Tensor& tensor,
    const torch::lazy::Shape& shape,
    const torch::lazy::BackendDevice& device) const override {
  // 根据输入张量的设备类型和形状创建选项
  at::TensorOptions options = tensor.options().device(
      default_device_type_->c10Type(), device.ordinal());
  // 如果张量在默认设备类型上并且是 CUDA 类型
  if (tensor.device().type() == default_device_type_->c10Type() &&
      default_device_type_->c10Type() == at::kCUDA) {
    // 将张量转移到指定选项和设备上，并创建 TSData 对象
    return std::make_shared<TSData>(
        tensor.to(options, /*non_blocking=*/true), shape, device);
  } else if (tensor.device().type() == at::kCPU && tensor.numel() == 1) {
    // 对于单个 CPU 张量，通过 .item() 获取其数值，并创建填充的设备张量
    auto device_tensor = at::full(tensor.sizes(), tensor.item(), options);
    // 创建 TSData 对象
    return std::make_shared<TSData>(device_tensor, shape, device);
  } else {
    // 对于其他情况，将张量移到指定选项和设备上，并创建 TSData 对象
    return std::make_shared<TSData>(
        tensor.to(options, /*non_blocking=*/false), shape, device);
  }
}

torch::lazy::BackendDataPtr MakeComputationDataFromScalar(
    const at::Scalar& scalar,
    const torch::lazy::BackendDevice& device) const override {
  // 使用标量和设备创建 TSData 对象
  return std::make_shared<TSData>(scalar, device);
}

torch::lazy::BackendDataPtr GetComputationDataFromNode(
    const Node* node) const override {
  // 尝试将节点转换为设备数据节点，如果不成功则返回空指针
  auto* device_data_node = DeviceData::Cast(node);
  if (!device_data_node) {
    return nullptr;
  }
  // 返回设备数据节点中的数据
  return device_data_node->data();
}

std::string GetComputationBackendText(
    const torch::lazy::ComputationPtr computation) const override {
  // 将计算对象转换为 TSComputation 类型，并获取其图形的字符串表示
  auto ts_computation =
      static_cast<torch::lazy::TSComputation*>(computation.get());
  return ts_computation->graph()->toString();
}

////////////////// 计算客户端接口 ///////////////////////

public:
torch::lazy::BackendDataPtr CreateDataPlaceholder(
    const torch::lazy::BackendDevice& device,
    const torch::lazy::Shape& shape) const override;

std::vector<torch::lazy::ComputationPtr> Compile(
    std::vector<torch::lazy::ComputationPtr> instances) const override;

std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
    torch::lazy::ComputationPtr computation,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    const torch::lazy::BackendDevice& device) const override;

std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType()
    const override {
  // 返回默认设备类型
  return default_device_type_;
}

at::DeviceType EagerFallbackDeviceType() const override;
  
void SetDefaultDeviceType(int8_t type) override {
    // 设置默认设备类型，使用给定的类型参数创建一个 TSBackendDeviceType 的 shared_ptr 对象
    default_device_type_ = std::make_shared<TSBackendDeviceType>(
        static_cast<c10::DeviceType>(type));
    }
    
    // 返回默认设备的序数
    int64_t GetDefaultDeviceOrdinal() const override {
        return default_device_ordinal_;
    }
    
    // 设置默认设备的序数
    void SetDefaultDeviceOrdinal(int64_t ordinal) override {
        default_device_ordinal_ = ordinal;
    }
    
    // 获取后端设备的列表，由派生类实现
    std::vector<torch::lazy::BackendDevice> GetBackendDevices() const override;
    
    // 获取特定设备的后端设备信息，由派生类实现
    torch::lazy::BackendDevice GetBackendDevice(
        c10::Device device) const override;
    
    // 设置随机数生成器的种子，当前方法未实现，记录致命错误日志
    void SetRngSeed(size_t seed) const override {
        LOG(FATAL) << "Not implemented yet.";
    }
    
    // 获取内存信息的方法，当前方法未实现，记录致命错误日志
    // MemoryInfo GetMemoryInfo(const std::string& device) override {
    //   LOG(FATAL) << "Not implemented yet.";
    // }
    
    // 准备退出应用的方法，由派生类实现
    void PrepareToExit() const override;
    
    // 默认设备类型的 shared_ptr 对象
    std::shared_ptr<TSBackendDeviceType> default_device_type_;
    
    // 默认设备的序数，初始化为 0
    int64_t default_device_ordinal_{0};
};

// 创建数据占位符，根据给定的设备和形状创建一个 TSData 对象，并返回其指针
torch::lazy::BackendDataPtr TSBackendImpl::CreateDataPlaceholder(
    const torch::lazy::BackendDevice& device,
    const torch::lazy::Shape& shape) const {
  return std::make_shared<TSData>(shape, device);
}

// 编译函数，接受一组计算实例并返回已编译的实例集合
std::vector<torch::lazy::ComputationPtr> TSBackendImpl::Compile(
    std::vector<torch::lazy::ComputationPtr> instances) const {
  // 遍历每个实例
  for (const auto& instance : instances) {
    // 将实例转换为 TSComputation 类型
    auto ts_computation =
        static_cast<torch::lazy::TSComputation*>(instance.get());
    // 如果不在标记步骤中，则记录警告信息
    if (!ts_computation->in_mark_step) {
      LOG(WARNING) << "Compile outside of mark step";
    }
  }
  return instances;  // 返回编译后的实例集合
}

// 执行计算函数，根据给定的计算实例、参数和设备执行计算，并返回结果集合
std::vector<torch::lazy::BackendDataPtr> TSBackendImpl::ExecuteComputation(
    torch::lazy::ComputationPtr computation,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    const torch::lazy::BackendDevice& device) const {
  // 将计算实例转换为 TSComputation 类型
  auto ts_computation =
      std::dynamic_pointer_cast<torch::lazy::TSComputation>(computation);
  // 检查计算实例是否为 TSComputation 类型，如果不是则抛出错误
  TORCH_CHECK(ts_computation, "Computation isn't TSComputation");
  // 获取计算图执行器
  torch::jit::GraphExecutor& graph_executor = ts_computation->graph_executor();
  std::vector<torch::jit::IValue> stack;  // 创建 IValue 类型的栈
  // 遍历每个参数
  for (const auto& argument : arguments) {
    // 将参数转换为 TSData 类型
    const auto ts_data = std::static_pointer_cast<TSData>(argument);
    // 如果数据是标量，则直接加入栈中
    if (ts_data->scalar.has_value()) {
      stack.emplace_back(ts_data->scalar.value());
    } else {
      // 否则检查数据的设备类型是否匹配当前设备类型
      TORCH_CHECK(
          static_cast<c10::DeviceType>(default_device_type_->type) !=
              at::kCUDA ||
          ts_data->data().device().type() == at::kCUDA);
      stack.emplace_back(ts_data->data());  // 将数据加入栈中
    }
  }
  graph_executor.run(stack);  // 运行计算图执行器
  std::vector<torch::lazy::BackendDataPtr> results;  // 创建结果集合
  // 遍历栈中的每个组件
  for (torch::jit::IValue component : stack) {
    at::Tensor result = component.toTensor();  // 将组件转换为张量
    at::IntArrayRef result_sizes = result.sizes();  // 获取张量的尺寸
    // 创建形状对象，包含张量的数据类型和尺寸
    torch::lazy::Shape shape(
        result.scalar_type(),
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
    // 将结果添加到结果集合中
    results.push_back(std::make_shared<TSData>(result, shape, device));
  }
  return results;  // 返回执行结果集合
}

// 获取支持的后端设备列表
std::vector<torch::lazy::BackendDevice> TSBackendImpl::GetBackendDevices()
    const {
  std::vector<torch::lazy::BackendDevice> devices;  // 创建设备列表
  // TODO(whc) figure out how to query available devices from pytorch
  // 添加 CPU 和 CUDA 设备到设备列表中
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCPU, 0)));
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCUDA, 0)));
  return devices;  // 返回设备列表
}

// 获取后端设备对象，忽略 c10::Device 中指定的设备类型，返回对应的后端设备对象
torch::lazy::BackendDevice TSBackendImpl::GetBackendDevice(
    c10::Device device) const {
  // 注意，这里忽略了 c10::Device 中指定的设备类型，因为预期它是虚拟设备 (lazy::)
  return torch::lazy::BackendDevice(GetDefaultDeviceType(), device.index());
}

// 退出前的准备工作，目前为空实现
void TSBackendImpl::PrepareToExit() const {}
// 返回当前 TS 后端的急切回退设备类型
c10::DeviceType TSBackendImpl::EagerFallbackDeviceType() const {
  // 对于 TS 后端，硬件设备就是急切设备
  return (c10::DeviceType)GetDefaultDeviceType()->type;
}

// 获取 TS 后端实现的指针
torch::lazy::BackendImplInterface* GetTSBackendImpl() {
  // 创建静态的 TS 后端实例
  static TSBackendImpl* ts_backend_impl = new TSBackendImpl();
  return ts_backend_impl;
}

// 初始化 TorchScript 后端
void InitTorchScriptBackend() {
  // 注册 TorchScript 惰性本地函数
  at::RegisterTorchScriptLazyNativeFunctions();
  // 注册 TorchScript 自动微分惰性本地函数
  at::RegisterTorchScriptAutogradLazyNativeFunctions();
  // 注册 TS LTC（TorchScript Long-Term Compatibility）急切回退函数
  register_ts_ltc_eager_fallback();
  // 创建静态的后端注册器实例，并传入 TS 后端实现指针
  static std::unique_ptr<BackendRegistrar> s_registrar;
  s_registrar = std::make_unique<BackendRegistrar>(GetTSBackendImpl());

  // 创建静态的惰性图执行器实例
  static LazyGraphExecutor* executor = new LazyGraphExecutor();
  // 注册惰性图执行器
  LazyGraphExecutor::Register(executor);
}

// 命名空间 lazy 结束
} // namespace lazy

// 命名空间 torch 结束
} // namespace torch


这段代码主要涉及了初始化 TorchScript 后端的过程，包括注册函数和创建实例等。
```