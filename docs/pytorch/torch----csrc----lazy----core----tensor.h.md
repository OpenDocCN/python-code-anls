# `.\pytorch\torch\csrc\lazy\core\tensor.h`

```
#pragma once
// 使用预处理器指令#pragma once，确保头文件只被编译一次，防止重复包含

#include <c10/core/SymNodeImpl.h>
// 引入SymNodeImpl类的头文件，位于c10/core/SymNodeImpl.h

#include <c10/util/intrusive_ptr.h>
// 引入intrusive_ptr工具类的头文件，位于c10/util/intrusive_ptr.h

#include <torch/csrc/lazy/backend/backend_data.h>
// 引入backend_data.h，定义了与后端数据相关的功能和结构

#include <torch/csrc/lazy/backend/backend_device.h>
// 引入backend_device.h，定义了与后端设备相关的功能和结构

#include <torch/csrc/lazy/core/ir.h>
// 引入ir.h，定义了与中间表示（IR）相关的功能和结构

#include <torch/csrc/lazy/core/util.h>
// 引入util.h，定义了一些核心的工具函数和宏

namespace torch {
namespace lazy {

class TORCH_API SymNodeImpl : public c10::SymNodeImpl {
 public:
  // SymNodeImpl类继承自c10::SymNodeImpl类，是对符号节点的实现

  SymNodeImpl(NodePtr ptr) : node_(std::move(ptr)){};
  // SymNodeImpl类的构造函数，接受一个NodePtr类型的指针，并将其移动到node_成员变量中

  NodePtr node_;
  // 使用NodePtr定义了一个成员变量node_

};

class LazyTensor;
using LazyTensorPtr = c10::intrusive_ptr<LazyTensor>;
// 使用LazyTensorPtr作为LazyTensor的intrusive_ptr类型的别名

class TORCH_API LazyTensor : public c10::intrusive_ptr_target {
 public:
  // LazyTensor类继承自c10::intrusive_ptr_target类

  // This is the core lazy tensor data structure where all the tensor data is
  // held. The lazy tensor is nothing more than a shared pointer to a Data
  // object.
  // 这是核心的惰性张量数据结构，其中包含所有张量数据。惰性张量只是指向Data对象的共享指针。
  struct Data {
    Data(BackendDataPtr handle, BackendDevice device)
        : handle(std::move(handle)),
          device(std::move(device)),
          unique_id(GetNextTensorId()) {}
    // Data结构体的构造函数，接受BackendDataPtr和BackendDevice类型的参数，初始化成员变量

    Data(Value ir_value, BackendDevice device)
        : ir_value(std::move(ir_value)),
          device(std::move(device)),
          unique_id(GetNextTensorId()) {}
    // Data结构体的构造函数，接受Value和BackendDevice类型的参数，初始化成员变量

    Data(at::Tensor tensor_data, BackendDevice device)
        : tensor_data(std::move(tensor_data)),
          device(std::move(device)),
          unique_id(GetNextTensorId()) {}
    // Data结构体的构造函数，接受at::Tensor和BackendDevice类型的参数，初始化成员变量

    // TODO(alanwaketan): Remove this ctor. This is a
    // temporary ctor to ease XLA LTC migration. It depends on
    // XLA's Functionalization integration.
    // TODO注释：删除此构造函数。这是一个临时构造函数，以便于XLA LTC迁移。它依赖于XLA的功能化集成。

    Data(BackendDevice device)
        : device(std::move(device)), unique_id(GetNextTensorId()) {}
    // Data结构体的构造函数，接受BackendDevice类型的参数，初始化成员变量

    virtual ~Data();
    // Data结构体的虚拟析构函数

    BackendDataPtr handle;
    // 指向后端数据的指针

    Value ir_value;
    // 表示IR值的变量

    std::optional<at::Tensor> tensor_data;
    // 可选的at::Tensor类型数据变量

    const BackendDevice device;
    // 表示后端设备的常量变量

    const int64_t unique_id = 0;
    // 表示唯一ID的常量变量，默认为0

    size_t generation = 1;
    // 表示代数的变量，默认为1
  };

  static LazyTensorPtr Create(
      const at::Tensor& tensor,
      const BackendDevice& device);
  // 创建LazyTensorPtr对象，接受at::Tensor和BackendDevice类型的参数

  static LazyTensorPtr Create(Value ir_value, const BackendDevice& device);
  // 创建LazyTensorPtr对象，接受Value和BackendDevice类型的参数

  static LazyTensorPtr Create(BackendDataPtr handle);
  // 创建LazyTensorPtr对象，接受BackendDataPtr类型的参数

  static LazyTensorPtr Create(std::shared_ptr<Data> data);
  // 创建LazyTensorPtr对象，接受shared_ptr<Data>类型的参数

  // The default ctor previously created a null LazyTensor (one with no 'data'
  // obj). Creating a null LazyTensor is no longer possible, since the same can
  // be achieved by creating a null LazyTensorPtr and it is way too confusing to
  // have to check both lazy_tensor_ptr && *lazy_tensor_ptr, so everywhere that
  // used to rely on a LazyTensor obj with a null Data can now rely on a null
  // LazyTensorPtr instead.
  // 默认构造函数之前创建一个空的LazyTensor（没有'data'对象）。现在不再可能创建一个空的LazyTensor，因为可以通过创建一个空的LazyTensorPtr来实现相同的效果，并且同时检查lazy_tensor_ptr && *lazy_tensor_ptr会非常混乱，因此之前依赖于具有空Data的LazyTensor对象的地方现在可以依赖于一个空的LazyTensorPtr。

  LazyTensor() = delete;
  // 删除默认构造函数，不再允许创建空的LazyTensor对象

  LazyTensor(const LazyTensor&) = default;
  // 拷贝构造函数，使用默认实现

  LazyTensor(LazyTensor&&) noexcept = default;
  // 移动构造函数，使用默认实现

  ~LazyTensor() override = default;
  // 虚析构函数，使用默认实现

  size_t generation() const {
  // 返回代数的函数
  return data()->generation;
}

// 覆盖以使用自定义的形状。
virtual int64_t size(int64_t dim) const;

// 覆盖以使用自定义的图执行器。
virtual at::Tensor ToTensor(bool detached);

// 将当前 LazyTensor 的数据浅拷贝到目标 LazyTensor。
void ShallowCopyTo(LazyTensorPtr dest) const;

// 将给定的张量值分配给 LazyTensor。
void SetTensor(at::Tensor tensor);

// 从张量更新 LazyTensor 的值，并选择性地进行同步。
void UpdateFromTensor(at::Tensor tensor, bool sync);
void UpdateFromTensorOut(at::Tensor tensor);
void UpdateFromTensorOut(const LazyTensorPtr& tensor);

// 返回与 LazyTensor 关联的数据对象的共享指针。
const std::shared_ptr<Data>& data() const;

// 覆盖以使用自定义的数据类型转换。
virtual at::ScalarType dtype() const;

// 返回 LazyTensor 的形状（可能是引用）。
MaybeRef<Shape> shape() const;

// 返回 LazyTensor 使用的后端设备。
const BackendDevice& GetDevice() const;
// 返回 LazyTensor 的唯一标识符。
int64_t GetUniqueId() const;

// 获取 LazyTensor 数据的后端数据指针。
BackendDataPtr GetDataHandle();

// 获取 LazyTensor 当前值的后端数据指针。
BackendDataPtr CurrentDataHandle() const;

// 设置 LazyTensor 的后端数据指针。
void SetDataHandle(BackendDataPtr handle);
void SetDataHandle(BackendDataPtr handle, bool sync);

// 返回当前 IR 节点，如果不存在则返回 nullptr。
Value CurrentIrValue() const;

// 获取表示此 LazyTensor 的 IR 节点，如果不存在则创建一个。
// 虽然是 const API，但实际上会修改对象的内部状态。
Value GetIrValue() const;

// 设置 LazyTensor 的 IR 值。
void SetIrValue(Value ir_value);
void SetInPlaceIrValue(Value ir_value);

// 返回当前张量数据的可选副本。
std::optional<at::Tensor> CurrentTensorData() const;

// 根据给定节点创建输出 LazyTensor 的向量。
std::vector<LazyTensorPtr> MakeOutputTensors(NodePtr node) const;

// 将张量复制到指定后端设备上的 LazyTensor。
LazyTensorPtr CopyTensorToDevice(const BackendDevice& device);

// 应用挂起的图操作，准备使用数据。
// 覆盖以使用自定义的图执行器。
virtual void ApplyPendingGraph();

// 分配 IR 值时设置额外信息。
// 覆盖以设置额外信息。
virtual void AssignIrValue(Value ir_value) const;

protected:
explicit LazyTensor(std::shared_ptr<Data> data);

// 设置 LazyTensor 的张量数据。
void SetTensorData(at::Tensor tensor_data);

// 构建累积操作的图，但必须在某个点上进行渲染，否则图会无限增长。
// 比如：
//   for i in range(0, 100000):
//     a = a + b
void TryLimitGraphSize();

// 覆盖以实例化自己的数据。
virtual Value GetIrValueForTensor(
    const at::Tensor& tensor,
    const BackendDevice& device) const;

// 创建张量节点。
Value CreateTensorNode(BackendDataPtr data, bool read_only) const;

private:
LazyTensor(const at::Tensor& tensor, const BackendDevice& device);
LazyTensor(Value ir_value, const BackendDevice& device);
explicit LazyTensor(BackendDataPtr handle);

// 获取下一个张量的唯一标识符。
static int64_t GetNextTensorId();

std::shared_ptr<Data> data_;
};

// Utils to convert at::Tensor to LazyTensor, and vice versa.

// Section 0: c10::Tensorlist ==> lazy::TensorList
// note: GetTensorList is not totally parallel to GetLtcTensor; A TensorList
// skips
//       the LazyTensor wrappers, assuming that the list of underlying IR nodes
//       is actually more useful for downstream computations.  TBD.

// 声明一个函数 GetTensorList，接受一个 at::ITensorListRef 参数，返回一个 torch::lazy::Value 对象
TORCH_API torch::lazy::Value GetTensorList(at::ITensorListRef tensors);

// Section 1: at::Tensor => LazyTensor.
// Extracts the LazyTensor out of an at::Tensor. Returns a null LazyTensor
// if the tensor is not a lazy tensor.

// 声明一个函数 TryGetLtcTensor，接受一个 at::Tensor 引用，返回一个 LazyTensorPtr 对象
TORCH_API LazyTensorPtr TryGetLtcTensor(const at::Tensor& tensor);

// Extracts the LazyTensor out of an at::Tensor. Throws an exception
// if the tensor is not a lazy tensor.

// 声明一个函数 GetLtcTensor，接受一个 at::Tensor 引用，返回一个 LazyTensorPtr 对象
TORCH_API LazyTensorPtr GetLtcTensor(const at::Tensor& tensor);

// Same as above, applied to a list of tensors.

// 声明一个函数 GetLtcTensors，接受一个 at::Tensor 数组引用，返回一个 LazyTensorPtr 的向量
TORCH_API std::vector<LazyTensorPtr> GetLtcTensors(
    c10::ArrayRef<at::Tensor> tensors);

// If tensor is a lazy tensor type, returns the LazyTensor embedded within it,
// otherwise creates a new lazy tensor type with tensor as data.

// 声明一个函数 GetOrCreateLtcTensor，接受一个可选的 at::Tensor 对象和 BackendDevice 对象，返回一个 LazyTensorPtr 对象
TORCH_API LazyTensorPtr GetOrCreateLtcTensor(
    const std::optional<at::Tensor>& tensor,
    const BackendDevice& device);

// 声明一个函数 GetLtcTensorOrCreateForWrappedNumber，接受一个 at::Tensor 对象和 BackendDevice 对象，返回一个 LazyTensorPtr 对象
TORCH_API LazyTensorPtr GetLtcTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor,
    const BackendDevice& device);

// Section 2: LazyTensor => at::Tensor.
// Creates an ATen tensor from an LazyTensor.

// 声明一个函数 CreateAtenFromLtcTensor，接受一个 LazyTensorPtr 对象，返回一个 at::Tensor 对象
TORCH_API at::Tensor CreateAtenFromLtcTensor(const LazyTensorPtr& ltc_tensor);
// 声明一个函数 CreateAtenFromLtcTensor，接受一个 LazyTensor 对象，并返回一个 at::Tensor 对象
TORCH_API at::Tensor CreateAtenFromLtcTensor(LazyTensor&& ltc_tensor);

// Note [Lazy Tensor Functionalization]
// The functionalization pass is implemented by wrapping all TensorImpl
// objects in C++ with an extra FunctionalTensorWrapper object,
// that knows how to perform functionalization
//
// Certain functions in the aten API serve as entry/exit points for
// functionalization, where we need to perform the wrapping/unwrapping:
// - aten::to.device
// - aten::empty

// Given a non-lazy tensor, this function creates a lazy tensor on the specified
// (lazy) device. The functionalize_output determines whether or not we should
// wrap the output in a "functional wrapper".
//
// How do you know whether to pass true/false for functionalize_output?
//
// Case 1: nonlazy -> lazy
//   If you're implementing a function that takes in nonlazy tensors and returns
//   lazy tensors, then you should think of that function as an "entrypoint" to
//   functionalization, and use functionalize_output=true Examples include:
//   - factory functions (the LTC kernel for at::empty)
//   - CPU -> Lazy device converions (the LTC kernel for at::to_device)
//
// Case 2: lazy -> lazy
//   If you're implementing a function that takes in lazy tensors and returns
//   lazy tensors,
//   **but** requires creating lazy tensors internally,
//   then you can assume that the current function is running inside of some
// 定义一个名为 to_lazy_tensor 的 Torch API 函数，用于将输入的 self 张量转换为懒惰张量，
// 使用指定的选项和设备，并可以选择在非阻塞模式下执行转换。函数返回转换后的张量。
TORCH_API at::Tensor to_lazy_tensor(
    const at::Tensor& self,
    const c10::TensorOptions& options,
    at::Device device,
    bool non_blocking,
    bool functionalize_output);

// 模板函数：将 LazyTensorPtr 向量中的元素转换为对应的 Aten 张量，并返回一个元组。
template <size_t... Indices>
auto TupleAtenFromLtcTensorsImpl(
    const std::vector<LazyTensorPtr>& tensors,
    std::index_sequence<Indices...>) {
  // 使用 CreateAtenFromLtcTensor 函数对 LazyTensorPtr 向量中的每个元素进行转换，
  // 并将转换结果用 std::make_tuple 包装成元组返回。
  return std::make_tuple(CreateAtenFromLtcTensor(tensors[Indices])...);
}

// 模板函数：接受一个 LazyTensorPtr 向量，并调用 TupleAtenFromLtcTensorsImpl，
// 使用 std::make_index_sequence<N> 生成序列来创建一个包含 N 个元素的元组。
template <size_t N>
auto TupleAtenFromLtcTensors(const std::vector<LazyTensorPtr>& tensors) {
  return TupleAtenFromLtcTensorsImpl(tensors, std::make_index_sequence<N>{});
}

// 命名空间 lazy 下的命名空间 torch 内部结束声明
} // namespace lazy

// 命名空间 torch 内部结束声明
} // namespace torch
```