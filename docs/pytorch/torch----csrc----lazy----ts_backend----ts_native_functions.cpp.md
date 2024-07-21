# `.\pytorch\torch\csrc\lazy\ts_backend\ts_native_functions.cpp`

```py
// 包含 ATen 库中的各种头文件，用于张量操作和函数定义
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Functions.h>
#include <ATen/MetaFunctions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>

// 包含 Torch 的 Lazy 模块核心功能头文件，用于构建 IR、操作和形状推断
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/shape_inference.h>
#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/csrc/lazy/core/tensor_util.h>

// 包含 Torch 的 Lazy 模块生成的 Native 函数头文件，用于操作懒执行张量
#include <torch/csrc/lazy/generated/LazyNativeFunctions.h>

// 包含 Torch 的 Lazy 模块后端配置和操作头文件，处理张量操作和自动求导
#include <torch/csrc/lazy/ts_backend/config.h>
#include <torch/csrc/lazy/ts_backend/ops/to_copy.h>
#include <torch/csrc/lazy/ts_backend/tensor_aten_ops.h>
#include <torch/csrc/lazy/ts_backend/ts_autograd_functions.h>
#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h>

// 包含 Torch 核心库
#include <torch/library.h>

// 使用 ATen 命名空间中的 Tensor 类
using at::Tensor;

// 定义 torch 的 lazy 命名空间
namespace torch {
namespace lazy {
namespace {

// 创建一个 LTC 张量，根据输入的 tensor 和设备信息
at::Tensor CreateLtcTensor(
    const at::Tensor& tensor,
    const std::optional<torch::lazy::BackendDevice>& device) {
  // 如果 tensor 有效且有设备信息
  if (tensor.defined() && device) {
    // 创建 LTC 张量，并将其转换为 ATen 张量返回
    return torch::lazy::CreateAtenFromLtcTensor(
        torch::lazy::LazyTensor::Create(tensor, *device));
  }
  // 否则直接返回原始的 tensor
  return tensor;
}

// 获取 LTC 设备，根据输入的设备信息
std::optional<torch::lazy::BackendDevice> GetLtcDevice(
    const std::optional<c10::Device>& device) {
  // 如果设备信息为空，则返回空值
  if (!device) {
    return c10::nullopt;
  }
  // 如果设备类型不是 Lazy 类型，则返回空值
  if (device->type() != at::kLazy) {
    return c10::nullopt;
  }
  // 否则将 ATen 设备信息转换为 LTC 设备信息并返回
  return torch::lazy::atenDeviceToBackendDevice(*device);
}

} // namespace

// 在 LazyTensor 中，clone 操作是一个空操作（no-op）
// 这是安全的，因为 LazyTensor 中的每个操作都是函数式的
at::Tensor LazyNativeFunctions::clone(
    const at::Tensor& self,
    std::optional<at::MemoryFormat> memory_format) {
  // 尝试获取 self 的 LazyTensor 表示
  auto self_lt = torch::lazy::TryGetLtcTensor(self);
  // 创建一个 ATen 张量，从 LazyTensor 的 IR 值和设备信息中获取
  return torch::lazy::CreateAtenFromLtcTensor(
      self_lt->Create(self_lt->GetIrValue(), self_lt->GetDevice()));
}

// 从 self 复制到 dst 的操作
at::Tensor LazyNativeFunctions::_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  // 计算 Lazy 函数调用次数
  TORCH_LAZY_FN_COUNTER("lazy::");
  // 尝试获取 dst 的 LazyTensor 表示
  auto dst_tensor = torch::lazy::TryGetLtcTensor(dst);
  // 尝试获取 self 的 LazyTensor 表示
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);

  // 如果 self_tensor 为空，为现有的 lazy tensor（dst）提供一个新的 'eager' 值（self）
  static bool sync_update = FLAGS_torch_lazy_ts_tensor_update_sync;
  TORCH_CHECK(dst_tensor);
  dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);

  // 如果 dst_tensor 为空，实现一个 lazy tensor（self），并将其值复制到 eager 张量（dst）
  // detached=false 允许我们在 `ToTensor` 中跳过复制，因为我们只会使用这个张量进行 dst.copy_()
  else if (!dst_tensor) {
    TORCH_CHECK(self_tensor);
    // 将 LazyTensor 转换为 Tensor，并根据 dst 的标量类型复制张量，不进行复制
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/false);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    // 如果目标张量和类型匹配，则调整目标张量大小并复制类型化张量数据
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // 复制一个惰性张量到另一个惰性张量
    if (!dst_tensor->CurrentIrValue()) {
      // 如果目标张量没有被 IR 支持（例如某些惰性操作的结果），
      // 那么它应该有 at::Tensor 数据作为支持
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      TORCH_CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        // src 和 dst 都仅由 at::Tensor 数据支持，没有 IR - 执行直接复制
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        // 在将 src 的结果用于复制到 dst 之前，需要先实现化 src 张量
        // 注意：我们仅仅为了复制而使用 src 张量，因此不需要将其分离
        // 注：如果能够直接将 ToTensor 的值直接材料化到 dst 的缓冲区中将更有效
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/false));
      }
    } else {
      // 直接调用复制操作，将 self_tensor 复制到 dst_tensor
      copy_(dst_tensor, self_tensor);
      // 获取 dst 张量的底层实现，并设置为 dst_tensor
      auto* impl =
          dynamic_cast<torch::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
      impl->set_tensor(dst_tensor);
    }
  }
  // 返回处理后的目标张量 dst
  return dst;
}
// 结束函数体，返回类型为 `at::Tensor`，参数为 `self` 和 `dst`
at::Tensor LazyNativeFunctions::_copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  // 记录懒惰函数调用的计数器，以 "lazy::" 开头命名
  TORCH_LAZY_FN_COUNTER("lazy::");
  // 尝试获取 dst 的懒惰张量表示
  auto dst_tensor = torch::lazy::TryGetLtcTensor(dst);
  // 尝试获取 self 的懒惰张量表示
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  // 如果 self 不是懒惰张量
  if (!self_tensor) {
    // 检查 dst 必须是懒惰张量
    TORCH_CHECK(dst_tensor);
    // 使用 self 更新 dst
    dst_tensor->UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    // 如果 dst 不是懒惰张量
    TORCH_CHECK(self_tensor);
    // 将 self 转换为张量，并指定为分离的张量
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    // 将 tensor 拷贝到与 dst 具有相同数据类型的 typed_tensor 中，但不复制数据
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    // 重新调整 dst 的大小为 typed_tensor，并复制 typed_tensor 的数据到 dst
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // 当 dst 是懒惰张量时
    // 确定 dst 是懒惰张量，然后更新其内容为 self_tensor
    auto* dest_impl =
        dynamic_cast<torch::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
    dest_impl->tensor()->UpdateFromTensorOut(self_tensor);
    // 强制刷新懒惰张量的大小信息
    dest_impl->force_refresh_sizes();
  }
  // 返回更新后的 dst 张量
  return dst;
}

// 将 self 转换为指定类型的副本张量
at::Tensor LazyNativeFunctions::_to_copy(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {
  // 如果需要强制回退到急切模式，则抛出内部断言错误
  if (force_eager_fallback(at::aten::_to_copy)) {
    TORCH_INTERNAL_ASSERT(
        false,
        "Fallback is currently impossible for _to_copy since the fallback helper itself reinvokes _to_copy");
  }

  // 获取 self 的选项信息
  auto options = self.options();
  // 如果指定了 dtype，则设置选项的数据类型
  if (dtype) {
    options = options.dtype(dtype);
  }
  // 如果指定了 layout，则设置选项的布局
  if (layout) {
    options = options.layout(layout);
  }
  // 如果指定了 memory_format，则设置选项的内存格式
  if (memory_format) {
    options = options.memory_format(memory_format);
  }
  // 如果指定了 pin_memory，则设置选项为固定内存
  if (pin_memory) {
    options = options.pinned_memory(pin_memory);
    // 发出警告，指出在懒惰 _to_copy 中使用了固定内存，需要检查其行为是否符合预期
    TORCH_WARN_ONCE(
        "Pinned memory used in lazy _to_copy, check if the behavior is as intended");
  }

  // 记录懒惰函数调用的计数器，以 "lazy::" 开头命名
  TORCH_LAZY_FN_COUNTER("lazy::");
  // 尝试获取 self 的懒惰张量表示
  auto lazy_self = torch::lazy::TryGetLtcTensor(self);
  // 如果 self 不是懒惰张量，并且目标设备为懒惰张量类型
  if (!lazy_self && device && device->type() == c10::kLazy) {
    // 情况1：急切模式到懒惰模式的转换（创建一个新的懒惰张量）
    // 参见注释 [Lazy Tensor Functionalization]
    // 不变量：如果功能化键在排除集合中，则我们应返回一个普通张量，稍后将其"提升"为功能化包装器
    bool functionalize_output =
        !c10::impl::tls_local_dispatch_key_set().excluded_.has(
            c10::DispatchKey::Functionalize);
    // 将 self 转换为懒惰张量，并指定选项、设备、是否非阻塞操作、是否功能化输出
    return torch::lazy::to_lazy_tensor(
        self,
        options,
        *device,
        /*non_blocking=*/non_blocking,
        /*functionalize_output=*/functionalize_output);
  } else if (device && device->type() != c10::kLazy) {
    // 如果设备类型不是懒惰张量类型
    // 此处应包含设备类型不同的处理逻辑，但省略在注释中的具体细节
    // Case 2: lazy->eager (forces a graph break since we are materializing a
    // tensor)

    // 断言确保 lazy_self 不为空
    TORCH_INTERNAL_ASSERT(lazy_self);
    // 将 lazy_self 转换为 eager_tensor，并设置为分离状态
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    // 设置选项为指定设备
    options = options.device(device);
    // 将 eager_tensor 移动到指定设备，同时进行复制操作
    auto moved_eager_tensor =
        eager_tensor.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
    // 返回移动后的 eager_tensor
    return moved_eager_tensor;
  } else if (
      device && device->type() == c10::kLazy && device->has_index() &&
      device->index() != self.device().index()) {
    // Case 3: lazy:0 -> lazy:1

    // TODO(whc) what do we actually want to do here?
    //   option 1: materialize, move eager tensor, create new lazy tensor
    //     - this should be our default, as it is what would happen before we
    //     implemented _to_copy
    //     - actually combines case 1 + case 2
    //   option 2: support multiple devices inside one lazy/TS executor (case 4)
    //     - but: we may have other assumptions that there is just one device
    //     per executor? so don't take this lightly

    // 断言确保 lazy_self 不为空
    TORCH_INTERNAL_ASSERT(lazy_self);
    // 将 lazy_self 转换为 eager_tensor，并设置为分离状态
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    // 将 eager_tensor 移动到与当前 lazy 设备对应的 eager 设备
    // 例如，如果当前设备是 lazy:1，则将其映射到 cuda:1
    auto eager_device = c10::Device(
        torch::lazy::getBackend()->EagerFallbackDeviceType(), device->index());
    options = options.device(eager_device);
    // 将 eager_tensor 移动到指定设备，同时进行复制操作
    auto moved_eager_tensor =
        eager_tensor.to(options, /*non_blocking=*/false, /*copy=*/true);
    // 获取或创建与 moved_eager_tensor 相对应的 lazy tensor
    lazy_self = torch::lazy::GetOrCreateLtcTensor(
        moved_eager_tensor,
        torch::lazy::atenDeviceToBackendDevice(eager_device));
    // 根据 lazy_self 创建 Aten 格式的张量并返回
    return torch::lazy::CreateAtenFromLtcTensor(lazy_self);

  } else {
    // Case 4: lazy->lazy (special case: keep the _to_copy INSIDE the lazy
    // graph)

    // Note: captured _to_copy will be executed with real eager tensors, not
    // lazy tensors. We DO NOT want to burn 'lazy:0' as the device into this
    // captured IR, or we will try to convert an eager tensor back to a lazy one
    // inside the torchscript executor lazy:0 -> lazy:1 is handled in case3, so
    // we can safely drop the device argument
    // 将 device 参数设为无效
    device = c10::nullopt;

    // 重用现有的 _to_copy 节点，或者创建新的节点
    torch::lazy::NodePtr node = torch::lazy::ReuseNode<ToCopy>(
        lazy_self->GetIrValue(),
        dtype,
        layout,
        device,
        pin_memory,
        non_blocking,
        memory_format);
    // 如果没有可重用的节点，则计算 _to_copy 的形状并创建新节点
    if (!node) {
      auto shapes = torch::lazy::compute_shape__to_copy(
          self, dtype, layout, device, pin_memory, non_blocking, memory_format);
      TORCH_INTERNAL_ASSERT(shapes.size() == 1);
      node = torch::lazy::MakeNode<ToCopy>(
          lazy_self->GetIrValue(),
          dtype,
          layout,
          device,
          pin_memory,
          non_blocking,
          memory_format,
          std::move(shapes));
      // 缓存新创建的节点
      CacheNode(node);
    }
    # 调用 torch::lazy 命名空间中的函数 CreateAtenFromLtcTensor，将 lazy_self 所持有的节点（node）转换为 Aten 张量
    auto result =
        torch::lazy::CreateAtenFromLtcTensor(torch::lazy::LazyTensor::Create(
            std::move(node), lazy_self->GetDevice()));
    # 返回转换后的结果
    return result;
};

// 定义 LazyNativeFunctions 类的成员函数 empty_symint，用于创建一个未初始化的张量
at::Tensor LazyNativeFunctions::empty_symint(
    at::SymIntArrayRef sym_size,                          // 接受符号整数数组作为尺寸
    std::optional<at::ScalarType> dtype,                   // 可选的张量数据类型
    std::optional<at::Layout> layout,                     // 可选的张量布局
    std::optional<at::Device> device,                     // 可选的设备类型
    std::optional<bool> pin_memory,                       // 可选的固定内存选项
    std::optional<at::MemoryFormat> memory_format) {      // 可选的内存格式

  // TODO: support this directly
  auto size = C10_AS_INTARRAYREF_SLOW(sym_size);          // 将符号整数数组转换为尺寸数组
  const auto device_type = torch::lazy::getBackend()->EagerFallbackDeviceType();  // 获取后端设备类型
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))  // 设置张量选项的设备
                                  .layout(layout)                    // 设置张量选项的布局
                                  .pinned_memory(pin_memory)         // 设置张量选项的固定内存选项
                                  .dtype(dtype);                     // 设置张量选项的数据类型

  auto x_result = at::empty(size, options, memory_format);     // 创建一个未初始化的张量
  auto tensor = CreateLtcTensor(x_result, GetLtcDevice(device));  // 创建 LTC 张量

  // See Note [Lazy Tensor Functionalization]
  if (c10::impl::tls_local_dispatch_key_set().excluded_.has(
          c10::DispatchKey::Functionalize)) {
    // Invariant: if the functionalization key is in the exclude set, then we're
    // expected to return an ordinary tensor, which will be "lifted" into a
    // functional wrapper later.
    return tensor;  // 如果功能化键在排除集中，则返回普通张量
  } else {
    auto wrapped = at::functionalization::impl::to_functional_tensor(tensor);  // 将张量转换为功能化张量
    return wrapped;  // 返回功能化张量
  }
}

// 定义 LazyNativeFunctions 类的成员函数 empty_strided_symint，创建一个步长张量
at::Tensor LazyNativeFunctions::empty_strided_symint(
    at::SymIntArrayRef sym_size,                // 接受符号整数数组作为尺寸
    at::SymIntArrayRef sym_stride,              // 接受符号整数数组作为步长
    std::optional<at::ScalarType> dtype,        // 可选的张量数据类型
    std::optional<at::Layout> layout,          // 可选的张量布局
    std::optional<at::Device> device,          // 可选的设备类型
    std::optional<bool> pin_memory) {          // 可选的固定内存选项

  TORCH_LAZY_FN_COUNTER("lazy::");             // 计数器记录懒执行函数调用
  at::Tensor t =
      empty_symint(sym_size, dtype, layout, device, pin_memory, c10::nullopt);  // 调用 empty_symint 创建未初始化张量
  auto size = C10_AS_INTARRAYREF_SLOW(sym_size);   // 将符号整数数组转换为尺寸数组
  auto stride = C10_AS_INTARRAYREF_SLOW(sym_stride); // 将符号整数数组转换为步长数组
  return t.as_strided(size, stride, /*storage_offset=*/0);  // 返回按步长重塑后的张量
}

// 定义 LazyNativeFunctions 类的成员函数 fill_，用指定值填充张量
at::Tensor& LazyNativeFunctions::fill_(
    at::Tensor& self,                     // 输入张量的引用
    const at::Scalar& value) {            // 要填充的标量值

  TORCH_LAZY_FN_COUNTER("lazy::");       // 计数器记录懒执行函数调用
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);  // 获取 LTC 张量
  torch::lazy::fill_(self_tensor, value);  // 使用指定值填充张量
  return self;                            // 返回填充后的张量
}

// 定义 LazyNativeFunctions 类的成员函数 max_pool3d，调用自定义的最大池化操作
at::Tensor LazyNativeFunctions::max_pool3d(
    const at::Tensor& self,               // 输入张量
    at::IntArrayRef kernel_size,          // 卷积核大小
    at::IntArrayRef stride,               // 步长
    at::IntArrayRef padding,              // 填充
    at::IntArrayRef dilation,             // 膨胀
    bool ceil_mode) {                     // 是否使用天花板模式

  return torch::lazy::MaxPool3dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);  // 调用自定义的最大池化自动求导函数
}

// 覆盖 max pooling 运算符，只调用回退函数，因为我们已经定制了它们的自动求导函数
std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::max_pool3d_with_indices(
    const at::Tensor& self,               // 输入张量
    at::IntArrayRef kernel_size,          // 卷积核大小
    at::IntArrayRef stride,               // 步长
    at::IntArrayRef padding,              // 填充
    at::IntArrayRef dilation,             // 膨胀
    bool ceil_mode) {


// 定义一个名为 max_pool3d_with_indices 的函数，接受多个参数并返回结果
return at::native::
    // 调用 ATEN_OP 宏定义的 max_pool3d_with_indices 操作的后备函数
    call_fallback_fn<&ltc_eager_fallback, ATEN_OP(max_pool3d_with_indices)>::
        // 调用封装了指定参数的函数 self、kernel_size、stride、padding、dilation、ceil_mode
        call(self, kernel_size, stride, padding, dilation, ceil_mode);
// 定义 LazyNativeFunctions 类的 max_pool3d_with_indices_backward 方法，用于计算 3D 最大池化反向传播
at::Tensor LazyNativeFunctions::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
  // 调用 native 命名空间中的回调函数，执行 max_pool3d_with_indices_backward 操作
  return at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool3d_with_indices_backward)>::
      call(
          grad_output,
          self,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices);
}

// 定义 LazyNativeFunctions 类的 _unsafe_view 方法，用于实现不安全的视图操作
at::Tensor LazyNativeFunctions::_unsafe_view(
    const at::Tensor& self,
    at::IntArrayRef size) {
  // 增加懒惰函数计数器，用于记录调用懒惰函数的次数
  TORCH_LAZY_FN_COUNTER("lazy::");
  // 调用 LazyNativeFunctions 类的 view_copy_symint 方法进行视图复制操作
  return LazyNativeFunctions::view_copy_symint(
      self, c10::fromIntArrayRefSlow(size));
}

// 定义 LazyNativeFunctions 类的 lift 方法，用于将张量包装成 FunctionalTensorWrapper 对象
at::Tensor LazyNativeFunctions::lift(const at::Tensor& tensor) {
  // 断言确保 tensor 不是 FunctionalTensor
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  // 调用 functionalization 命名空间中的 to_functional_tensor 方法进行功能化操作
  return at::functionalization::impl::to_functional_tensor(tensor);
}

// 定义 LazyNativeFunctions 类的 lift_fresh 方法，与 lift 方法类似，用于功能化张量
at::Tensor LazyNativeFunctions::lift_fresh(const at::Tensor& tensor) {
  // 断言确保 tensor 不是 FunctionalTensor
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  // 调用 functionalization 命名空间中的 to_functional_tensor 方法进行功能化操作
  return at::functionalization::impl::to_functional_tensor(tensor);
}

// 定义 LazyNativeFunctions 类的 block_diag 方法，用于创建块对角矩阵
at::Tensor LazyNativeFunctions::block_diag(at::TensorList tensors) {
  // 调用 functionalization 命名空间中的 functionalize_aten_op 方法，实现 block_diag 操作的功能化
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      block_diag)>::call(tensors);
}

// 定义 LazyNativeFunctions 类的 new_empty_strided_symint 方法，用于创建新的空张量
at::Tensor LazyNativeFunctions::new_empty_strided_symint(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  // 调用 functionalization 命名空间中的 functionalize_aten_op_symint 方法，实现 new_empty_strided 操作的功能化
  return at::functionalization::
      functionalize_aten_op_symint<ATEN_OP(new_empty_strided)>::call(
          self, size, stride, dtype, layout, device, pin_memory);
}

// 定义 LazyNativeFunctions 类的 narrow_copy_symint 方法，用于复制并缩窄张量
at::Tensor LazyNativeFunctions::narrow_copy_symint(
    const at::Tensor& self,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt length) {
  // 调用 functionalization 命名空间中的 functionalize_aten_op_symint 方法，实现 narrow_copy 操作的功能化
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      narrow_copy)>::call(self, dim, start, length);
}

// 定义 LazyNativeFunctions 类的 pixel_shuffle 方法，用于像素重排操作
at::Tensor LazyNativeFunctions::pixel_shuffle(
    const at::Tensor& self,
    int64_t upscale_factor) {
  // 调用 functionalization 命名空间中的 functionalize_aten_op 方法，实现 pixel_shuffle 操作的功能化
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      pixel_shuffle)>::call(self, upscale_factor);
}
    // 调用 functionalize_aten_op 模板函数，对 ATEN_OP(pixel_unshuffle) 操作进行功能化处理，并传入 self 和 downscale_factor 参数
    return at::functionalization::functionalize_aten_op<ATEN_OP(pixel_unshuffle)>::call(self, downscale_factor);
}
// 定义 LazyNativeFunctions 类的 select_backward_symint 方法，用于对对称整数操作进行反向选择操作
at::Tensor LazyNativeFunctions::select_backward_symint(
    const at::Tensor& grad_output, // 输入：梯度输出张量
    c10::SymIntArrayRef input_sizes, // 输入：输入尺寸的对称整数数组引用
    int64_t dim, // 输入：维度
    c10::SymInt index) { // 输入：对称整数索引
  // 调用 functionalize_aten_op_symint 模板，执行 ATEN_OP(select_backward) 操作
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      select_backward)>::call(grad_output, input_sizes, dim, index);
}

// 定义 LazyNativeFunctions 类的 _trilinear 方法，用于执行三线性操作
at::Tensor LazyNativeFunctions::_trilinear(
    const at::Tensor& i1, // 输入：张量 i1
    const at::Tensor& i2, // 输入：张量 i2
    const at::Tensor& i3, // 输入：张量 i3
    at::IntArrayRef expand1, // 输入：扩展维度数组引用 expand1
    at::IntArrayRef expand2, // 输入：扩展维度数组引用 expand2
    at::IntArrayRef expand3, // 输入：扩展维度数组引用 expand3
    at::IntArrayRef sumdim, // 输入：求和维度数组引用 sumdim
    int64_t unroll_dim) { // 输入：展开维度
  // 调用 functionalize_aten_op 模板，执行 ATEN_OP(_trilinear) 操作
  return at::functionalization::functionalize_aten_op<ATEN_OP(_trilinear)>::
      call(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

// 定义 LazyNativeFunctions 类的 linalg_pinv 方法，用于计算矩阵的伪逆
at::Tensor LazyNativeFunctions::linalg_pinv(
    const at::Tensor& self, // 输入：自身张量
    const std::optional<at::Tensor>& atol, // 输入：可选参数，绝对容差张量
    const std::optional<at::Tensor>& rtol, // 输入：可选参数，相对容差张量
    bool hermitian) { // 输入：是否为厄米矩阵
  // 调用 functionalize_aten_op 模板，执行 ATEN_OP2(linalg_pinv, atol_rtol_tensor) 操作
  return at::functionalization::functionalize_aten_op<ATEN_OP2(
      linalg_pinv, atol_rtol_tensor)>::call(self, atol, rtol, hermitian);
}

// 定义 LazyNativeFunctions 类的 logsumexp_out 方法，用于计算对数和指数操作
at::Tensor& LazyNativeFunctions::logsumexp_out(
    const at::Tensor& self, // 输入：自身张量
    at::IntArrayRef dim, // 输入：维度数组引用
    bool keepdim, // 输入：是否保持维度
    at::Tensor& out) { // 输入/输出：输出张量
  // 将 self 和 out 转换为 functional_tensor，并调用核心的 logsumexp_out 操作
  auto self_wrapped = at::functionalization::impl::to_functional_tensor(self);
  auto out_wrapped = at::functionalization::impl::to_functional_tensor(out);
  
  // 直接从核心调用复合核心
  // 在调用之前确保重新启用 functionalization
  auto curr_tls = c10::impl::tls_local_dispatch_key_set();
  auto tls_reenable_functionalize = c10::impl::PODLocalDispatchKeySet();
  tls_reenable_functionalize.set_included(curr_tls.included_);
  tls_reenable_functionalize.set_excluded(
      curr_tls.excluded_.remove(c10::DispatchKey::Functionalize));
  c10::impl::ForceDispatchKeyGuard guard_(tls_reenable_functionalize);
  
  // 调用核心的 logsumexp_out 操作
  at::native::logsumexp_out(self_wrapped, dim, keepdim, out_wrapped);
  
  // 将操作结果从 functional_tensor 转换回普通张量，并将结果复制回 out（包括调整大小）
  auto out_unwrapped =
      at::functionalization::impl::from_functional_tensor(out_wrapped);
  out.resize_(out_unwrapped.sizes());
  out.copy_(out_unwrapped);
  
  // 返回输出张量的引用
  return out;
}

// 定义 LazyNativeFunctions 类的 diag_embed 方法，用于创建对角线张量
at::Tensor LazyNativeFunctions::diag_embed(
    const at::Tensor& self, // 输入：自身张量
    int64_t offset, // 输入：偏移量
    int64_t dim1, // 输入：维度1
    int64_t dim2) { // 输入：维度2
  // 调用 functionalize_aten_op 模板，执行 ATEN_OP(diag_embed) 操作
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      diag_embed)>::call(self, offset, dim1, dim2);
}

// 定义 LazyNativeFunctions 类的 diagonal_backward_symint 方法，用于对称整数对角线反向操作
at::Tensor LazyNativeFunctions::diagonal_backward_symint(
    const at::Tensor& grad_output, // 输入：梯度输出张量
    at::SymIntArrayRef input_sizes, // 输入：输入尺寸的对称整数数组引用
    int64_t offset, // 输入：偏移量
    int64_t dim1, // 输入：维度1
    int64_t dim2) { // 输入：维度2
  // 调用 functionalize_aten_op_symint 模板，执行 ATEN_OP(diagonal_backward) 操作
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      diagonal_backward)>::call(grad_output, input_sizes, offset, dim1, dim2);
}

// 定义 LazyNativeFunctions 类的 slice_backward_symint 方法，用于对称整数切片反向操作
at::Tensor LazyNativeFunctions::slice_backward_symint(
    const at::Tensor& grad_output,
    // 调用 ATEN_OP 宏定义的 slice_backward 操作的反向传播函数
    // 使用 functionalize_aten_op_symint 模板函数进行操作符函数化
    // 返回经过反向传播处理后的梯度输出
    return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
        slice_backward)>::call(grad_output, input_sizes, dim, start, end, step);
// 重新使用核心中的复合内核，这样我们就不需要为 native_group_norm 提供反向公式
std::tuple<Tensor, Tensor, Tensor> LazyNativeFunctions::native_group_norm(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  // 调用 math_group_norm 函数进行组归一化操作
  return at::native::math_group_norm(
      input, weight, bias, N, C, HxW, group, eps);
}

} // namespace lazy
} // namespace torch
```