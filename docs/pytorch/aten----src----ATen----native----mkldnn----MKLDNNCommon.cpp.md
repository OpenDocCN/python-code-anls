# `.\pytorch\aten\src\ATen\native\mkldnn\MKLDNNCommon.cpp`

```py
/**
 * `MKLDNNCommon.h` 包含了使用 MKL-DNN 的常用功能和数据结构的头文件。
 * `OpaqueTensorImpl.h` 包含了用于不透明张量实现的头文件，其中定义了 OpaqueTensorImpl 类。
 * `Allocator.h` 包含了内存分配器的头文件，提供了内存分配和释放的功能。
 * `torch/library.h` 包含了 Torch 库的头文件，用于 Torch 库的整合和初始化。
 *
 * `IntrusivePtrTargetWrapper` 结构体封装了一个自定义张量存储的句柄（作为模板参数），
 * 并继承了 `c10::intrusive_ptr_target`，以便可以与 `c10::intrusive_ptr` 一起使用。
 * 目前它仅支持通过以下方式封装自定义句柄：
 * - 通过复制/移动构造函数使用现有的自定义句柄。
 *
 * 详见 `OpaqueTensorImpl::opaque_handle_`。
 *
 * 注意：如果这个结构在整体上很有用，可能需要将其移动到自己的头文件中。
 */
template <typename T>
struct TORCH_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
private:
  T target_;

public:
  /**
   * 构造函数，使用给定的自定义句柄初始化。
   */
  IntrusivePtrTargetWrapper(const T& target): target_(target) {}

  /**
   * 移动构造函数，使用给定的自定义句柄进行移动初始化。
   */
  IntrusivePtrTargetWrapper(T&& target): target_(std::move(target)) {}

  /**
   * 获取当前封装的自定义句柄的引用。
   */
  T& get_target() {
    return target_;
  }
};

/**
 * 使用 ideep::tensor 类型作为模板参数的 IntrusivePtrTargetWrapper 类型的别名。
 */
using IDeepTensorWrapper = IntrusivePtrTargetWrapper<ideep::tensor>;

/**
 * 使用 c10::intrusive_ptr<IDeepTensorWrapper> 类型的别名。
 */
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;

/**
 * 使用 OpaqueTensorImpl<IDeepTensorWrapperPtr> 类型的别名。
 */
using MKLDNNTensorImpl = OpaqueTensorImpl<IDeepTensorWrapperPtr>;

/**
 * 使用 Tensor 类型的别名。
 */
using MKLDNNTensor = Tensor;

/**
 * 根据给定的标量类型返回对应的 MKL-DNN 数据类型。
 */
ideep::tensor::data_type get_mkldnn_dtype(ScalarType type) {
  switch (type) {
    case ScalarType::Float:
      return ideep::tensor::data_type::f32;
    case ScalarType::QInt32:
      return ideep::tensor::data_type::s32;
    case ScalarType::QInt8:
    case ScalarType::Char:
      return ideep::tensor::data_type::s8;
    case ScalarType::QUInt8:
    case ScalarType::Byte:
      return ideep::tensor::data_type::u8;
    case ScalarType::BFloat16:
      return ideep::tensor::data_type::bf16;
    case ScalarType::Half:
      return ideep::tensor::data_type::f16;
    default:
      TORCH_CHECK(false, "get_mkldnn_dtype: unsupported data type");
  }
}

/**
 * 从 MKL-DNN 张量获取数据指针。
 */
int64_t data_ptr_from_mkldnn(const Tensor& mkldnn_tensor) {
  // 获取 MKLDNNTensorImpl 指针
  MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(mkldnn_tensor.unsafeGetTensorImpl());
  // 获取不透明句柄的数据指针
  void* data_ptr = mklimpl->unsafe_opaque_handle()->get_target().get_data_handle();
  // 将数据指针转换为 int64_t 类型并返回
  return reinterpret_cast<int64_t>(data_ptr);
}

/**
 * 根据给定的数据指针、维度、数据类型、设备和不透明元数据创建 MKL-DNN 张量。
 */
at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  // 将不透明元数据转换为 std::vector<uint8_t> 类型
  std::vector<uint8_t> vector_serialized_md{
      opaque_metadata, opaque_metadata + opaque_metadata_size};
  // 定义用于描述 ideep::tensor 的 desc
  ideep::tensor::desc deserialized_ideep_desc;
#if IDEEP_PREREQ(3, 4, 1, 2)
  // 从 vector_serialized_md 构造 ideep::tensor 的描述符
  deserialized_ideep_desc = ideep::tensor::desc(vector_serialized_md);
#else
  // 如果不支持当前的 ideep 版本，则抛出错误
  TORCH_CHECK(false, "Unexpected IDeep version to do weight deserialization.");
#endif

  // 使用给定的数据指针和描述符构造 ideep::tensor 对象
  auto a = ideep::tensor(deserialized_ideep_desc, data_ptr);
  // 调用 Torch 的 new_with_itensor_mkldnn 函数创建 MKL-DNN 张量并返回
  return at::native::new_with_itensor_mkldnn(std::move(a), dtype, device);
}
// 从 ideep::tensor&& 类型的参数创建新的 MKLDNN 张量
// 如果提供了 dtype，使用提供的数据类型；否则使用默认的数据类型
// 如果提供了 device，使用提供的设备；否则使用默认的设备
Tensor new_with_itensor_mkldnn(ideep::tensor&& it, std::optional<ScalarType> dtype, std::optional<Device> device) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // ideep::tensor 提供的维度是 int32_t 类型，但是 sizes 需要 int64_t 类型
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  auto dims = it.get_dims(); // 获取 ideep::tensor 的维度信息
  // 创建一个 IDeepTensorWrapper 的智能指针，包装 ideep::tensor
  IDeepTensorWrapperPtr handle = c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  // 将 dtype 转换为 TypeMeta 类型
  caffe2::TypeMeta dtype_ = scalarTypeToTypeMeta(dtype_or_default(dtype));
  // 获取或设置设备信息
  Device device_ = device_or_default(device);
  // 使用 MKLDNNTensorImpl 的详细信息创建张量
  return detail::make_tensor<MKLDNNTensorImpl>(
    DispatchKeySet(DispatchKey::MkldnnCPU), // 设置分发键为 MKL-DNN CPU
    dtype_, device_, handle,
    std::vector<int64_t>(dims.begin(), dims.end())); // 转换维度为 int64_t 类型并创建张量
}

// 从 MKLDNNTensor 类型的输入中获取 ideep::tensor 的引用
ideep::tensor& itensor_from_mkldnn(const MKLDNNTensor& mkldnn_tensor) {
  TORCH_CHECK(mkldnn_tensor.is_mkldnn(),
             "itensor_from_mkldnn expects MKL-DNN tensor input"); // 断言输入为 MKL-DNN 张量
  // 获取 MKLDNNTensorImpl 指针，并转换为 ideep::tensor 的不安全透明句柄
  MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(mkldnn_tensor.unsafeGetTensorImpl());
  // 返回透明句柄中的目标 ideep::tensor 引用
  return mklimpl->unsafe_opaque_handle()->get_target();
}

// 从 Tensor 类型的 MKL-DNN 张量中获取字节大小
int64_t nbytes_from_mkldnn(const Tensor& mkldnn_tensor) {
  // 将 MKL-DNN 张量转换为 ideep::tensor
  ideep::tensor t = itensor_from_mkldnn(mkldnn_tensor);
  // 返回 ideep::tensor 的描述中的大小（字节数）
  return t.get_desc().get_size();
}

// 从稠密 Tensor 创建 ideep::tensor 的视图
ideep::tensor itensor_view_from_dense(const Tensor& tensor, bool from_const_data_ptr) {
  // 断言输入的 Tensor 是 CPU 张量
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  // 断言输入的 Tensor 是 Strided 布局的稠密张量
  TORCH_CHECK(
      tensor.layout() == Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  // 根据不同的数据类型创建 ideep::tensor 视图
  if (tensor.scalar_type() == ScalarType::Float) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::f32,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<float*>(tensor.template const_data_ptr<float>()) :
              tensor.template data_ptr<float>()};
  }
  else if (tensor.scalar_type() == ScalarType::BFloat16) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::bf16,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<BFloat16*>(tensor.template const_data_ptr<BFloat16>()) :
              tensor.template data_ptr<BFloat16>()};
  }
  else if (tensor.scalar_type() == ScalarType::Half) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::f16,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<Half*>(tensor.template const_data_ptr<Half>()) :
              tensor.template data_ptr<Half>()};
  }
  else if (tensor.scalar_type() == ScalarType::Byte) {
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::u8,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<void*>(tensor.const_data_ptr()) :
              tensor.data_ptr()};
  }
  else if (tensor.scalar_type() == ScalarType::Char) {
    # 如果条件成立，则执行以下语句块：创建一个 ideep::tensor::view 对象
    return {{tensor.sizes().vec(),
            ideep::tensor::data_type::s8,
            tensor.strides().vec()},
            from_const_data_ptr ?
              const_cast<void*>(tensor.const_data_ptr()) :
              tensor.data_ptr()};
    }
    else {
        # 如果条件不成立，则抛出错误，提示调用者输入必须是 float/bfloat16/half/int8 类型的张量
        TORCH_CHECK(false, "itensor_view_from_dense expects float/bfloat16/half/int8 tensor input");
    }
// 结束 ideep 命名空间定义

ideep::tensor itensor_view_from_dense(
    const at::Tensor& tensor,  // 输入参数：一个 ATen 张量
    const ideep::tensor::desc& desc) {  // 输入参数：一个 ideep 张量描述符
  // 检查输入张量是否在 CPU 上
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  // 检查输入张量是否为连续布局
  TORCH_CHECK(
      tensor.layout() == at::Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  // 检查输入张量的数据类型是否为 float、bfloat16 或 half
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Float ||
          tensor.scalar_type() == at::ScalarType::BFloat16 ||
          tensor.scalar_type() == at::ScalarType::Half,
      "itensor_view_from_dense expects float, bfloat16 or half tensor input");
  // 返回一个 ideep 张量视图，使用给定的描述符和 ATen 张量的数据指针
  return {desc, tensor.data_ptr()};
}

// 获取一个 ideep 张量，从一个 ATen 张量中
// 如果 ATen 张量是密集张量，返回的 ideep 张量只是 ATen 密集张量存储的一个视图，
// 因此调用者需要确保 ATen 密集张量的生命周期比 ideep 张量长。
ideep::tensor itensor_from_tensor(const Tensor& tensor, bool from_const_data_ptr) {
  // 如果 ATen 张量是 MKLDNN 张量，调用专门的转换函数
  if (tensor.is_mkldnn()) {
    return itensor_from_mkldnn(tensor);
  } else {
    // 否则，调用密集张量的视图函数
    return itensor_view_from_dense(tensor, from_const_data_ptr);
  }
}

// 设置详细输出级别的辅助函数
int set_verbose(int level) {
    return ideep::utils::set_verbose(level);
}

// Torch 库实现：MKLDNN，MkldnnCPU
TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  // 注册实现：mkldnn::data_ptr
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::data_ptr"),
      TORCH_FN(data_ptr_from_mkldnn));
  // 注册实现：mkldnn::_nbytes
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_nbytes"),
      TORCH_FN(nbytes_from_mkldnn));
}

// 结束 AT_MKLDNN_ENABLED() 条件编译
#endif // AT_MKLDNN_ENABLED()
```