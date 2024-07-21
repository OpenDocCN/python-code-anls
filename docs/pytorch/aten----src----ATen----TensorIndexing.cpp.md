# `.\pytorch\aten\src\ATen\TensorIndexing.cpp`

```
// 引入 ATen 库中的 TensorIndexing.h 头文件

#include <ATen/TensorIndexing.h>

// 引入 C10 库中的 Exception.h 和 irange.h 头文件
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 命名空间声明，包含了 at::indexing 命名空间
namespace at {
namespace indexing {

// 定义常量 Ellipsis，并初始化为 EllipsisIndexType 类型的实例
const EllipsisIndexType Ellipsis = EllipsisIndexType();

// 重载操作符 << ，用于将 Slice 对象输出到流中
std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
  // 输出 Slice 对象的 start、stop 和 step 成员变量到流中
  stream << slice.start() << ":" << slice.stop() << ":" << slice.step();
  return stream;
}

// 重载操作符 << ，用于将 TensorIndex 对象输出到流中
std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index) {
  // 根据 TensorIndex 对象的类型输出相应内容到流中
  if (tensor_index.is_none()) {
    stream << "None";
  } else if (tensor_index.is_ellipsis()) {
    stream << "...";
  } else if (tensor_index.is_integer()) {
    stream << tensor_index.integer();
  } else if (tensor_index.is_boolean()) {
    stream << std::boolalpha << tensor_index.boolean();
  } else if (tensor_index.is_slice()) {
    stream << tensor_index.slice();
  } else if (tensor_index.is_tensor()) {
    stream << tensor_index.tensor();
  }
  return stream;
}

// 重载操作符 << ，用于将 TensorIndex 对象的 vector 输出到流中
std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices) {
  stream << "(";
  for (const auto i : c10::irange(tensor_indices.size())) {
    stream << tensor_indices[i];
    if (i < tensor_indices.size() - 1) stream << ", ";
  }
  stream << ")";
  return stream;
}

// 设置 Tensor 对象的元素，当被分配的值是 Scalar 类型时
// 对应于 torch/csrc/autograd/python_variable_indexing.cpp 中的 `THPVariable_setitem` 函数
static inline void set_item(const Tensor& self, ArrayRef<TensorIndex> indices, const Scalar& v) {
  Tensor value;

  {
    // 在 ADInplaceOrView 以下进行自动分发
    at::AutoDispatchBelowADInplaceOrView guard;
    // 获取 self 的设备信息
    at::Device self_device = self.device();

    // 如果 self 是 qint 类型，执行特殊处理
    if (isQIntType(self.scalar_type())) {
      // 将标量 v 转换为 Tensor，并指定设备为 CPU，数据类型为 float
      value = at::indexing::scalarToTensor(v, device(kCPU).dtype(kFloat), at::Device(kCPU));
    } else if (self_device.is_cuda()) {
      // 如果 self 的设备是 CUDA 设备，将标量 v 转换为 Tensor，并使用 self 的选项
      value = at::indexing::scalarToTensor(v, self.options(), at::Device(kCPU));
    } else {
      // 否则，将标量 v 转换为 Tensor，并使用 self 的选项和设备信息
      value = at::indexing::scalarToTensor(v, self.options(), self_device);
    }
  }

  // 调用重载的 set_item 函数，将转换后的 Tensor 值设置给 self 的指定索引位置
  return set_item(self, indices, value);
}

} // namespace indexing
} // namespace at

// Tensor 类的成员函数，用于按给定索引获取 Tensor 对象
Tensor Tensor::index(ArrayRef<at::indexing::TensorIndex> indices) const {
  // 检查索引列表是否为空，若为空则抛出异常
  TORCH_CHECK(!indices.empty(), "Passing an empty index list to Tensor::index() is not valid syntax");
  // 获取当前 Tensor 的设备信息，并创建 OptionalDeviceGuard 对象
  OptionalDeviceGuard device_guard(device_of(*this));
  // 调用 at::indexing 命名空间中的 get_item 函数，获取按索引获取的 Tensor 对象并返回
  return at::indexing::get_item(*this, indices);
}

// Tensor 类的成员函数，用于按初始化列表的索引获取 Tensor 对象
Tensor Tensor::index(std::initializer_list<at::indexing::TensorIndex> indices) const {
  // 调用重载的 index 函数，将初始化列表转换为 ArrayRef 后再进行索引操作
  return index(ArrayRef<at::indexing::TensorIndex>(indices));
}

// Tensor 类的成员函数，用于按给定索引修改 Tensor 对象的元素值
Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  // 检查索引列表是否为空，若为空则抛出异常
  TORCH_CHECK(!indices.empty(), "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  // 获取当前 Tensor 的设备信息，并创建 OptionalDeviceGuard 对象
  OptionalDeviceGuard device_guard(device_of(*this));
  // 调用 at::indexing 命名空间中的 set_item 函数，将 rhs 设置到 self 指定索引位置
  at::indexing::set_item(*this, indices, rhs);
  // 返回当前 Tensor 对象的引用
  return *this;
}
// 在当前 Tensor 对象上执行索引赋值操作，将标量 v 放入给定的 indices 中指定的位置
Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, const Scalar& v) {
  // 检查索引列表是否为空，若为空则抛出异常信息
  TORCH_CHECK(!indices.empty(), "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  // 创建 OptionalDeviceGuard 对象，用于管理当前 Tensor 对象的设备情况
  OptionalDeviceGuard device_guard(device_of(*this));
  // 在当前 Tensor 对象上执行索引赋值操作，将标量 v 放入给定的 indices 中指定的位置
  at::indexing::set_item(*this, indices, v);
  // 返回当前 Tensor 对象的引用
  return *this;
}

// 在当前 Tensor 对象上执行索引赋值操作，将另一个 Tensor 对象 rhs 放入给定的 indices 中指定的位置
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  // 调用 index_put_ 函数，将 initializer_list 转换为 ArrayRef 后继续执行
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), rhs);
}

// 在当前 Tensor 对象上执行索引赋值操作，将标量 v 放入给定的 indices 中指定的位置
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, const Scalar& v) {
  // 调用 index_put_ 函数，将 initializer_list 转换为 ArrayRef 后继续执行
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), v);
}

} // namespace at


这些注释解释了每个函数在类 `Tensor` 中的作用和功能，包括参数的含义以及函数的返回值和异常处理。
```