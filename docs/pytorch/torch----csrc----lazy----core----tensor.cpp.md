# `.\pytorch\torch\csrc\lazy\core\tensor.cpp`

```
// 引入 Torch 的 Lazy 模块的相关头文件
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/tensor.h>

// 引入 C10 库的工具和范围控制
#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/csrc/lazy/core/tensor_util.h>

// 引入 ATen 的 FunctionalTensorWrapper 头文件
#include <ATen/FunctionalTensorWrapper.h>

// 定义命名空间 torch::lazy::
namespace torch {
namespace lazy {

// 匿名命名空间，包含一些局部函数和变量，实现局部封装
namespace {

// GetOrCreateLtcTensor 函数，用于获取或创建 LazyTensor
LazyTensorPtr GetOrCreateLtcTensor(
    const at::Tensor& tensor,  // 输入的 ATen 张量
    const BackendDevice& device) {  // 张量的设备信息

  // 如果输入张量未定义，返回空指针
  if (!tensor.defined()) {
    return torch::lazy::LazyTensorPtr();
  }

  // 尝试获取已有的 LazyTensor
  auto lazy_tensor = TryGetLtcTensor(tensor);

  // 如果存在则返回，否则创建一个新的 LazyTensor 并返回
  return lazy_tensor ? lazy_tensor : LazyTensor::Create(tensor, device);
}

} // namespace

// LazyTensor 类的析构函数实现，用于销毁数据时从图执行器中注销张量
LazyTensor::Data::~Data() {
  LazyGraphExecutor::Get()->UnregisterTensor(this);
}

// 创建 LazyTensor 的静态方法，根据 ATen 张量和设备信息创建 LazyTensor
LazyTensorPtr LazyTensor::Create(
    const at::Tensor& tensor,  // 输入的 ATen 张量
    const BackendDevice& device) {  // 张量的设备信息

  // 检查输入张量的设备类型不能是 Lazy，否则抛出异常
  TORCH_CHECK(tensor.device().type() != at::kLazy);

  // 创建一个新的 LazyTensor 对象
  LazyTensorPtr lazy_tensor =
      c10::make_intrusive<LazyTensor>(LazyTensor(tensor, device));

  // 将创建的数据注册到 Lazy 图执行器中
  LazyGraphExecutor::Get()->RegisterTensor(lazy_tensor->data());

  // 返回创建的 LazyTensor 智能指针
  return lazy_tensor;
}

// 根据 IR 值和设备信息创建 LazyTensor 的静态方法
LazyTensorPtr LazyTensor::Create(Value ir_value, const BackendDevice& device) {
  // 创建一个新的 LazyTensor 对象
  LazyTensorPtr lazy_tensor =
      c10::make_intrusive<LazyTensor>(LazyTensor(std::move(ir_value), device));

  // 将创建的数据注册到 Lazy 图执行器中
  LazyGraphExecutor::Get()->RegisterTensor(lazy_tensor->data());

  // 返回创建的 LazyTensor 智能指针
  return lazy_tensor;
}

// 根据后端数据指针创建 LazyTensor 的静态方法
LazyTensorPtr LazyTensor::Create(BackendDataPtr handle) {
  // 创建一个新的 LazyTensor 对象
  LazyTensorPtr lazy_tensor =
      c10::make_intrusive<LazyTensor>(LazyTensor(std::move(handle)));

  // 将创建的数据注册到 Lazy 图执行器中
  LazyGraphExecutor::Get()->RegisterTensor(lazy_tensor->data());

  // 返回创建的 LazyTensor 智能指针
  return lazy_tensor;
}

// 根据共享的数据指针创建 LazyTensor 的静态方法
LazyTensorPtr LazyTensor::Create(std::shared_ptr<Data> data) {
  // 直接使用给定的数据指针创建 LazyTensor
  return c10::make_intrusive<LazyTensor>(LazyTensor(std::move(data)));
}

// 根据 ATen 张量和设备信息创建 LazyTensor 的构造函数
LazyTensor::LazyTensor(const at::Tensor& tensor, const BackendDevice& device)
    : LazyTensor(std::make_shared<Data>(tensor, device)) {}

// 根据后端数据指针创建 LazyTensor 的构造函数
LazyTensor::LazyTensor(BackendDataPtr handle)
    : LazyTensor(std::make_shared<Data>(handle, handle->device())) {}

// 根据 IR 值和设备信息创建 LazyTensor 的构造函数
LazyTensor::LazyTensor(Value ir_value, const BackendDevice& device)
    : LazyTensor(std::make_shared<Data>(std::move(ir_value), device)) {
  // 尝试限制图的大小
  TryLimitGraphSize();
}

// 使用共享的数据指针创建 LazyTensor 的构造函数
LazyTensor::LazyTensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

// 返回当前 LazyTensor 的数据指针
auto LazyTensor::data() const -> const std::shared_ptr<Data>& {
  // 检查数据指针不为空，否则抛出异常
  TORCH_CHECK(data_ != nullptr, "Trying to access a null cursor");
  return data_;
}

// 获取 LazyTensor 指定维度的大小
int64_t LazyTensor::size(int64_t dim) const {
  // 获取张量的形状
  auto tensor_shape = shape();
  int rank = tensor_shape.Get().dim();
  // 获取规范化后的维度索引
  int dim_index = GetCanonicalDimensionIndex(dim, rank);
  // 返回指定维度的大小
  return tensor_shape.Get().size(dim_index);
}

// 返回 LazyTensor 的数据类型
at::ScalarType LazyTensor::dtype() const {
  return shape().Get().scalar_type();
}

// 获取 LazyTensor 的形状
MaybeRef<Shape> LazyTensor::shape() const {
  // 如果数据句柄不为空
  if (data()->handle != nullptr) {
    return Shape(data()->handle->shape());
  }
  如果 `data()->handle` 存在，返回其形状作为一个 `Shape` 对象

  if (data()->ir_value) {
    // TODO(whc) remove shape from LazyTensor API too!
    如果 `data()->ir_value` 存在，返回其形状
    // TODO: (whc) 从 LazyTensor API 中也移除形状相关的功能！
    // 这段代码可能在未来的更新中被删除或修改
    return data()->ir_value.shape();
  }
  如果 `data()->ir_value` 存在，则返回 `data()->ir_value` 的形状

  TORCH_CHECK(data()->tensor_data);
  使用 Torch 提供的检查函数，检查 `data()->tensor_data` 是否存在，如果不存在会抛出异常

  return Shape(
      data()->tensor_data->scalar_type(),
      ToI64Vector(data()->tensor_data->sizes()));
  返回一个 `Shape` 对象，其构造函数接受 `data()->tensor_data` 的标量类型和尺寸，封装成 `Shape` 对象返回
const BackendDevice& LazyTensor::GetDevice() const {
  // 返回当前 LazyTensor 对象关联的设备信息
  return data()->device;
}

int64_t LazyTensor::GetUniqueId() const {
  // 返回当前 LazyTensor 对象的唯一标识符
  return data()->unique_id;
}

BackendDataPtr LazyTensor::GetDataHandle() {
  // 获取当前数据的句柄
  BackendDataPtr handle = CurrentDataHandle();
  // 如果句柄非空，检查其是否包含有效数据，否则抛出错误
  if (handle != nullptr) {
    TORCH_CHECK(
        handle->HasValue(),
        "Trying to access data while an async operation is in flight: ",
        handle->shape().to_string());
    return handle;
  }

  // 如果存在 IR 值，应用悬而未决的计算图
  if (data()->ir_value) {
    ApplyPendingGraph();
  } else {
    // 否则，确保存在有效的张量数据，将其转换为数据句柄
    TORCH_CHECK(data()->tensor_data);
    data()->handle = TensorToDataHandle(*data()->tensor_data, GetDevice());
  }

  // 返回数据句柄
  return data()->handle;
}

BackendDataPtr LazyTensor::CurrentDataHandle() const {
  // 返回当前数据的句柄
  return data()->handle;
}

void LazyTensor::SetDataHandle(BackendDataPtr handle) {
  // 设置当前数据的句柄，并清空关联的 IR 节点，以允许图的修剪
  SetDataHandle(std::move(handle), /*sync=*/true);
}

void LazyTensor::SetDataHandle(BackendDataPtr handle, bool sync) {
  // 设置当前数据的句柄，并清空关联的 IR 节点
  data()->handle = std::move(handle);
  AssignIrValue(Value());
  // 如果需要同步，清空张量数据
  if (sync) {
    data()->tensor_data = c10::nullopt;
  }
}

void LazyTensor::SetIrValue(Value ir_value) {
  // 清空数据句柄和张量数据，设置新的 IR 值，并尝试限制图的大小
  data()->handle = nullptr;
  data()->tensor_data = c10::nullopt;
  AssignIrValue(std::move(ir_value));
  TryLimitGraphSize();
}

void LazyTensor::SetInPlaceIrValue(Value ir_value) {
  // 设置当前 IR 值，如果类型不匹配，则进行类型转换
  auto tensor_shape = shape();
  if (tensor_shape.Get().scalar_type() != ir_value.shape().scalar_type()) {
    ir_value =
        MakeCast(ir_value, tensor_shape.Get().scalar_type(), c10::nullopt);
  }
  SetIrValue(std::move(ir_value));
}

void LazyTensor::AssignIrValue(Value ir_value) const {
  // 分配新的 IR 值，并增加代数计数
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
}

void LazyTensor::TryLimitGraphSize() {
  // 尝试限制计算图的大小，根据条件对 IR 图进行修剪
  if (data()->ir_value &&
      LazyGraphExecutor::Get()->IncTrimCounter() %
              FLAGS_torch_lazy_trim_graph_check_frequency ==
          0) {
    size_t graph_size = Util::GetGraphSize({data()->ir_value.node.get()});
    if (static_cast<int64_t>(graph_size) > FLAGS_torch_lazy_trim_graph_size) {
      TORCH_LAZY_COUNTER("TrimIrGraph", 1);
      ApplyPendingGraph();
    }
  }
}

Value LazyTensor::GetIrValue() const {
  // 获取当前 IR 值，如果不存在，则获取当前数据句柄，并将其转换为张量节点
  Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  BackendDataPtr handle = CurrentDataHandle();
  // 对于张量节点，不清除数据，以便后续调用 GetIrValue() 可以获取相同的 IR 节点
  if (handle != nullptr) {
    AssignIrValue(CreateTensorNode(handle, /*read_only=*/false));
    // 返回新创建的 IR 值
    return CurrentIrValue();
  }
  // 如果无法获取 IR 值，则返回空
  return Value();
}
    # 返回当前数据的IR值
    return data()->ir_value;
    
    
    
    # 获取当前张量的数据，使用std::optional以处理可能的空值
    std::optional<at::Tensor> tensor_data = CurrentTensorData();
    
    
    
    # 检查张量数据是否存在，如果不存在则抛出错误
    TORCH_CHECK(tensor_data);
    
    
    
    # 获取张量数据关联的IR值，并将其分配给当前数据的IR值
    AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
    
    
    
    # 返回当前数据的IR值
    return data()->ir_value;
}

// 返回当前 LazyTensor 对象的 IR 值
Value LazyTensor::CurrentIrValue() const {
  return data()->ir_value;
}

// 设置 LazyTensor 对象的 tensor_data 属性
void LazyTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

// 返回当前 LazyTensor 对象的 tensor_data 属性
std::optional<at::Tensor> LazyTensor::CurrentTensorData() const {
  return data()->tensor_data;
}

// 根据给定的 tensor 和 device 获取对应的 IR 值
Value LazyTensor::GetIrValueForTensor(
    const at::Tensor& tensor,
    const BackendDevice& device) const {
  BackendDataPtr data;
  bool read_only = false;
  
  // 如果 tensor 是标量且只有一个元素，则直接返回标量值或特殊标量
  if (tensor.dim() == 0 && tensor.numel() == 1) {
    at::Scalar value = tensor.item();
    if (IsSpecialScalar(value)) {
      return MakeScalar(value, tensor.scalar_type());
    }
    // 否则根据 tensor 的 CPU 数据和设备获取对应的 BackendDataPtr
    data = LazyGraphExecutor::Get()->GetDeviceData(tensor.cpu(), device);
    read_only = true;
  } else {
    // 对非标量 tensor 进行 LazyGraphExecutor 的 TensorToDataHandle 转换
    TORCH_LAZY_TIMED("IrValueTensorToDataHandle");
    data = TensorToDataHandle(tensor, device);
  }
  
  // 根据 data 和 read_only 创建一个新的 Tensor 节点
  return CreateTensorNode(std::move(data), read_only);
}

// 将 LazyTensor 对象转换为 Tensor
at::Tensor LazyTensor::ToTensor(bool detached) {
  at::Tensor tensor;
  std::optional<at::Tensor> tensor_data = CurrentTensorData();
  
  // 如果 tensor_data 不存在，根据 GetDataHandle 获取对应的 Tensor
  if (!tensor_data) {
    LazyGraphExecutor::Get()->DeviceBarrier(GetDevice());
    // 调用 GetDataHandle() 将触发 ApplyPendingGraph() 如果 tensor 上有 IR 节点
    std::vector<at::Tensor> tensors =
        DataHandlesToTensors({GetDataHandle()}, dtype());
    tensor = std::move(tensors.front());
    if (!detached) {
      SetTensorData(tensor);
    }
  } else {
    tensor = *tensor_data;
    // 如果 detached 为 true，根据条件复制 tensor 或者置为 nullptr
    if (detached) {
      if (data()->ir_value || data()->handle != nullptr) {
        // 如果有其他权威来源，则丢弃当前引用并转移给调用者
        data()->tensor_data = c10::nullopt;
      } else {
        // 否则需要复制以防止调用者修改我们的版本
        tensor = CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

// 浅复制当前 LazyTensor 对象到目标 LazyTensorPtr
void LazyTensor::ShallowCopyTo(LazyTensorPtr dest) const {
  dest->SetIrValue(GetIrValue());
}

// 设置 LazyTensor 的 tensor_data 和 handle，同时清空 IR 值
void LazyTensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  data()->handle = nullptr;
  AssignIrValue(Value());
}

// 根据 sync 标志更新 LazyTensor 的 tensor_data 和 IR 值
void LazyTensor::UpdateFromTensor(at::Tensor tensor, bool sync) {
  if (sync) {
    // 如果 sync 为 true，则复制 tensor，并根据 typed_tensor 获取 IR 值
    at::Tensor typed_tensor = CopyTensor(tensor, dtype(), /*copy=*/false);
    SetIrValue(GetIrValueForTensor(typed_tensor, GetDevice()));
  } else {
    // 否则直接设置 tensor_data 和 handle，同时清空 IR 值
    SetTensorData(tensor);
    data()->handle = nullptr;
    AssignIrValue(Value());
  }
}

// 使用给定的 tensor 更新当前 LazyTensor 对象，并不进行同步
void LazyTensor::UpdateFromTensorOut(at::Tensor tensor) {
  UpdateFromTensor(std::move(tensor), /*sync=*/false);
}

// 使用另一个 LazyTensor 对象更新当前 LazyTensor 的 IR 值
void LazyTensor::UpdateFromTensorOut(const LazyTensorPtr& tensor) {
  SetIrValue(tensor->GetIrValue());
}

// 创建一个新的 Tensor 节点，并设置相关的设备信息
Value LazyTensor::CreateTensorNode(BackendDataPtr data, bool read_only) const {
  data->SetInfo(std::make_shared<LazyGraphExecutor::DeviceDataInfo>(
      GetUniqueId(), read_only));
  return MakeDeviceData(std::move(data));
}
// 创建输出张量列表的函数，基于给定节点生成 LazyTensorPtr 列表
std::vector<LazyTensorPtr> LazyTensor::MakeOutputTensors(NodePtr node) const {
    std::vector<LazyTensorPtr> tensors;
    tensors.reserve(node->num_outputs());
    // 遍历节点的输出数量，创建相应的 LazyTensorPtr 并添加到 tensors 列表中
    for (const auto i : c10::irange(node->num_outputs())) {
        tensors.push_back(Create(Value(node, i), GetDevice()));
    }
    return tensors;  // 返回生成的 LazyTensorPtr 列表
}

// 将张量复制到指定设备上的函数
LazyTensorPtr LazyTensor::CopyTensorToDevice(const BackendDevice& device) {
    // TODO: This can be optimized. (这里可以进行优化)
    return Create(ToTensor(/*detached=*/true), device);  // 创建一个新的 LazyTensorPtr，将其移动到指定设备
}

// 应用待处理图的函数
void LazyTensor::ApplyPendingGraph() {
    LazyGraphExecutor::Get()->DeviceBarrier(GetDevice());
    // 该方法确保张量数据在设备上可用，以便调用 CurrentDataHandle() 返回有效指针
    if (CurrentDataHandle() == nullptr) {
        // 创建包含当前 LazyTensor 副本的 LazyTensorPtr 列表
        std::vector<LazyTensorPtr> tensors(
            {c10::make_intrusive<LazyTensor>(LazyTensor(*this))});
        // 同步待处理图中的张量
        LazyGraphExecutor::Get()->SyncTensorsGraph(
            &tensors,
            {},  // 不包含额外待处理图
            /*wait=*/true,  // 等待所有同步完成
            /*sync_ltc_data=*/false);  // 不同步 LTC 数据
    }
}

// 获取下一个张量 ID 的函数
int64_t LazyTensor::GetNextTensorId() {
    static std::atomic<int64_t>* id_generator = new std::atomic<int64_t>(1);
    return id_generator->fetch_add(1);  // 原子操作：返回递增后的 ID
}

// 获取张量列表的函数
torch::lazy::Value GetTensorList(at::ITensorListRef tensors) {
    std::vector<Value> values;
    // 遍历输入张量列表，并获取其对应的 IR 值，形成值的列表
    for (const auto& t : tensors) {
        auto* impl = dynamic_cast<LTCTensorImpl*>(t.unsafeGetTensorImpl());
        TORCH_INTERNAL_ASSERT(
            impl,
            "GetTensorList only supports lists of valid tensors, but optional support could be added");
        values.push_back(impl->tensor()->GetIrValue());
    }

    return torch::lazy::Value(torch::lazy::MakeTensorList(std::move(values)));  // 返回包含值列表的 torch::lazy::Value
}

// 尝试获取 LTC 张量的函数
LazyTensorPtr TryGetLtcTensor(const at::Tensor& tensor) {
    auto* impl = dynamic_cast<LTCTensorImpl*>(
        maybe_unwrap_functional(tensor).unsafeGetTensorImpl());
    if (impl == nullptr) {
        // 如果未找到 LTC 张量，返回空的 LazyTensorPtr
        return LazyTensorPtr();
    }
    return impl->tensor();  // 返回找到的 LTC 张量的 LazyTensorPtr
}

// 获取 LTC 张量的函数
LazyTensorPtr GetLtcTensor(const at::Tensor& tensor) {
    auto lazy_tensor = TryGetLtcTensor(tensor);
    TORCH_CHECK(
        lazy_tensor, "Input tensor is not a lazy tensor: ", tensor.toString());
    return lazy_tensor;  // 返回 LTC 张量的 LazyTensorPtr
}

// 获取 LTC 张量的列表的函数
std::vector<LazyTensorPtr> GetLtcTensors(c10::ArrayRef<at::Tensor> tensors) {
    std::vector<LazyTensorPtr> ltc_tensors;
    ltc_tensors.reserve(tensors.size());
    // 遍历输入的张量数组，并尝试获取每个张量的 LTC 表示，将结果添加到 ltc_tensors 列表中
    for (const auto& tensor : tensors) {
        ltc_tensors.emplace_back(TryGetLtcTensor(tensor));
    }
    return ltc_tensors;  // 返回包含 LTC 张量的 LazyTensorPtr 列表
}

// 获取或创建 LTC 张量的函数，如果未提供张量，则创建一个新的 LazyTensorPtr
LazyTensorPtr GetOrCreateLtcTensor(
    const std::optional<at::Tensor>& tensor,
    const BackendDevice& device) {
    return GetOrCreateLtcTensor(tensor.value_or(at::Tensor()), device);
}

// 获取或为包装数值创建 LTC 张量的函数
LazyTensorPtr GetLtcTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor,
    // （此处可能还有未完待续的函数定义）
    // 根据给定的张量和设备对象，返回对应的 LTC 张量对象
    const BackendDevice& device) {
      // TODO: 在核心代码中有些地方对标量进行了包装，但未标记为已包装。
      // 如果张量是一个包装的数值或者是一个零维且元素个数为1的张量，则返回创建或获取的 LTC 张量
      return (tensor.unsafeGetTensorImpl()->is_wrapped_number() ||
              (tensor.dim() == 0 && tensor.numel() == 1))
          ? GetOrCreateLtcTensor(tensor, device)
          // 否则，返回已有的 LTC 张量
          : GetLtcTensor(tensor);
    }
} // namespace lazy
} // namespace torch
```