# `.\pytorch\test\cpp_extensions\open_registration_extension.cpp`

```py
// 引入头文件：无序映射、C10核心分配器、分配器、标量类型、数组引用
#include <unordered_map>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

// 引入Torch相关头文件：设备、序列化、设备保护接口、宏、扩展
#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

// 引入ATen相关头文件：CPU循环、量化器、分派存根、调整大小、一元操作、CPU回退、绝对值原生操作
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/abs_native.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/view.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/attention.h>

// 静态变量声明
static uint64_t add_counter = 0;
static uint64_t last_saved_value = 0;
static c10::DeviceIndex custom_device_index = 0;

static uint64_t abs_counter = 0;
static uint64_t last_abs_saved_value = 0;

static uint64_t storageImpl_counter = 0;
static uint64_t last_storageImpl_saved_value = 0;

// 注册保护守卫
namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(
    PrivateUse1,  // 注册名称为PrivateUse1的守卫
    c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);  // 使用NoOpDeviceGuardImpl保护私有设备类型

}} // namespace at::detail

// 匿名命名空间，定义abs_kernel函数
namespace {

// 使用最简单的方法获取连续的Tensor数据并处理它。
// 这是使用操作API的演示，可以根据自定义设备内核为输入和输出Tensor添加更复杂的逻辑。
void abs_kernel(at::TensorIteratorBase& iter) {
  // abs_kernel函数处理的是迭代器的操作数
  auto& output_operand = iter.operand(0);  // 输出操作数
  auto& input_operand = iter.operand(1);   // 输入操作数
  auto& output_tensor_base = output_operand.tensor_base();  // 输出Tensor基础
  auto& input_tensor_base = input_operand.tensor_base();    // 输入Tensor基础

  // 检查输入和输出操作数的原始Tensor是否已定义
  TORCH_CHECK(!input_operand.original_tensor_base().defined(),
    "input original tensor is defined.");
  TORCH_CHECK(!output_operand.original_tensor_base().defined(),
    "output original tensor is defined.");

  // 对于简单的测试，只接受连续的输入Tensor进行计算
  auto memory_format = input_tensor_base.suggest_memory_format();
  TORCH_CHECK(input_tensor_base.is_contiguous(memory_format),
    "Input tensor need be contiguous.");

  // 添加必要的限制条件，以确保演示的安全性
  TORCH_CHECK(input_tensor_base.sizes() == output_tensor_base.sizes(),
    "Intput and output tensor size are not equal.");

  // 在TensorIteratorBase中计算公共的数据类型
  TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float,
    "Only support float type.")

  // 使用for循环计算绝对值
  auto abs_function = [](float* output_ptr, const float* input_ptr,
                         const int64_t NUM) {
    for (int64_t i = 0; i < NUM; ++i) {
      *(output_ptr + i) = std::abs(*(input_ptr + i));
    }
  };  // 绝对值计算的Lambda函数
  }
};
// 简化测试演示代码逻辑，仅使用连续张量在设备端进行计算。
// 使用输入张量的内存格式。
if (iter.is_contiguous()) {
  // 添加 will_resize 标志检查。当 will_resize 为 True 时，可以将张量内存格式转换为不同的形式。
  // 如果 TensorIteratorConfig 的 resize_outputs_ 标志为真，存在两种情况：
  // 1) 输出张量未定义，并且 TensorIterator 设置 will_resize 为真；
  // 2) 输出张量已定义，但张量大小与输入张量大小不等；
  //    TensorIterator 设置 will_resize 为真，并调用 set_output_raw_strided
  //    来调整输出张量大小。
  // 当输出操作数的 will_resize 标志为真时，虚拟设备可以将张量转换为虚拟设备首选的内存格式。
  // 这里我们不转换张量内存格式，因为虚拟设备要保持训练网络相同的内存格式会变得复杂。
  TORCH_CHECK(output_operand.will_resize,
    "输出操作数的 will_resize 标志需要为 True。");
  abs_function((float*)iter.data_ptr(0), (float*)iter.data_ptr(1), iter.numel());
} else {
  // 不支持 foo 设备的跨步复制，改用 CPU 设备代替。
  // 对于 abs 操作，最后一种情况是：输出张量不连续且操作数的 will_resize 为 False。
  TORCH_CHECK(!output_operand.will_resize, "输出操作数的 will_resize 为 True。");
  // 使用输入内存格式获取一个连续张量。
  at::Tensor output = at::empty(output_tensor_base.sizes(),
                                input_tensor_base.options()
                                                 .memory_format(memory_format));
  // 对于从 TensorIteratorBase 继承的结构化操作，可能需要调用 set_output_raw_strided 函数
  // 来更新操作中存储的输出。对于 abs 操作不需要这样做。
  output_operand.exchange_tensor(c10::MaybeOwned<at::TensorBase>::owned(std::in_place, output));
  abs_function((float*)output_operand.tensor_base().mutable_data_ptr(),
               (float*)iter.data_ptr(1), iter.numel());
  // 将张量基础复制到原始张量基础，并保持与 CPU 和 GPU 相同的标量类型和步长。
  if (output_operand.original_tensor_base().defined() &&
      !output_operand.original_tensor_base().is_same(output_operand.tensor_base())) {
    output_operand.original_tensor().copy_(output_operand.tensor());
    output_operand.restore_original_tensor();
  }
}
}

// 定义一个函数，用于按张量级别的仿射量化，但此处什么也不做
void quantize_tensor_per_tensor_affine_privateuse1(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {
    // do nothing
}

// 定义一个函数，用于选择 SDP（自注意力机制）的后端，返回一个整数值
int64_t _fused_sdp_choice_privateuse1(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value,
    const c10::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, c10::optional<double> scale){
  // 获取 SDP 的后端，默认为 sdp::SDPBackend::overrideable
  auto backend = sdp::SDPBackend::overrideable;
  // 将后端类型转换为整数返回
  return static_cast<int64_t>(backend);
}
} // namespace

namespace at::native {

// 注册 abs_kernel 函数为 abs_stub 的私有使用 1 版本的调度器
REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel);
// 注册 quantize_tensor_per_tensor_affine_privateuse1 函数为 quantize_tensor_per_tensor_affine_stub 的调度器
REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_tensor_affine_stub, &quantize_tensor_per_tensor_affine_privateuse1);
// 注册 _fused_sdp_choice_privateuse1 函数为 _fused_sdp_choice_stub 的调度器
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_privateuse1);

} // namespace at::native

// 定义一个自定义后端元数据结构，继承自 c10::BackendMeta
struct CustomBackendMetadata : public c10::BackendMeta {
  // 用于测试的字段，在 clone() 被 shallow_copy_from 调用时将会变化
  int backend_version_format_{-1};
  int format_number_{-1};
  mutable bool cloned_{false};

  // 定义构造函数
  CustomBackendMetadata(int backend_version_format, int format_number) :
      backend_version_format_(backend_version_format), format_number_(format_number) {}

  // 实现 clone 方法，返回一个克隆的 BackendMeta 实例
  c10::intrusive_ptr<c10::BackendMeta> clone(
      const c10::intrusive_ptr<c10::BackendMeta>& ptr) const override {
    cloned_ = true;
    return c10::BackendMeta::clone(ptr);
  }
};

// 用于序列化的函数，将张量 t 的信息存入 unordered_map m
void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  // 如果张量的后端元数据为空，则返回
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr() == nullptr) {
    return;
  }
  // 尝试将后端元数据转换为 CustomBackendMetadata 类型
  auto tmeta = dynamic_cast<CustomBackendMetadata*>(t.unsafeGetTensorImpl()->get_backend_meta());
  // 如果后端版本格式为 1，则设置 m 中的相应字段为 true
  if (tmeta->backend_version_format_ == 1) {
    m["backend_version_format"] = true;
  }
  // 如果格式编号为 29，则设置 m 中的相应字段为 true
  if (tmeta->format_number_ == 29) {
    m["format_number"] = true;
  }
}

// 用于反序列化的函数，根据 unordered_map m 来设置张量 t 的后端元数据
void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  int backend_version_format{-1};
  int format_number{-1};
  // 如果 m 中包含 "backend_version_format" 字段，则设置 backend_version_format 为 1
  if (m.find("backend_version_format") != m.end()) {
    backend_version_format = 1;
  }
  // 如果 m 中包含 "format_number" 字段，则设置 format_number 为 29
  if (m.find("format_number") != m.end()) {
    format_number = 29;
  }
  // 创建一个新的 CustomBackendMetadata 实例，并将其设置为张量 t 的后端元数据
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta{std::unique_ptr<c10::BackendMeta>(
      new CustomBackendMetadata(backend_version_format, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}

// 注册自定义后端序列化的函数到 Torch 的 TensorBackendMetaRegistry 中
void custom_serialization_registry() {
  torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1,
                                        &for_serialization,
                                        &for_deserialization);
}

// 检查张量 t 的后端元数据是否正确序列化
bool check_backend_meta(const at::Tensor& t) {
  // 如果张量的后端元数据存在
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr()) {
    // 尝试将后端元数据转换为 CustomBackendMetadata 类型
    CustomBackendMetadata* tmeta = dynamic_cast<CustomBackendMetadata*>(
        t.unsafeGetTensorImpl()->get_backend_meta());
    # 如果 tmeta 结构体的 backend_version_format_ 字段为 1，并且 format_number_ 字段为 29
    if (tmeta->backend_version_format_ == 1 && tmeta->format_number_ == 29) {
      # 返回 true，表示条件满足
      return true;
    }
  }
  # 如果未满足上述条件，返回 false
  return false;
}

// Python侧暴露的虚假设置函数
void custom_set_backend_meta(const at::Tensor& t) {
  // 设定后端元数据格式和版本号
  int backend_version_format{1};
  int format_number{29};
  // 创建自定义后端元数据对象
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta{std::unique_ptr<c10::BackendMeta>(
      new CustomBackendMetadata(backend_version_format, format_number))};
  // 设置张量的后端元数据
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}

// 为我们的自定义设备创建虚假的StorageImpl，实际上使用CPU
c10::intrusive_ptr<c10::StorageImpl> make_custom_storage_impl(c10::StorageImpl::use_byte_size_t,
                                                              c10::SymInt size_bytes,
                                                              c10::DataPtr data_ptr,
                                                              c10::Allocator* allocator,
                                                              bool resizable) {
  // 初始化自定义StorageImpl指针
  c10::intrusive_ptr<c10::StorageImpl> custom_storage_impl;
  // 如果data_ptr为空
  if (data_ptr == nullptr){
    // 创建带有特定大小和分配器的StorageImpl对象
    custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, resizable);
  } else {
    // 创建带有特定大小、数据指针和分配器的StorageImpl对象
    custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, resizable);
  }
  // 增加StorageImpl计数器
  storageImpl_counter += 1;
  // 返回自定义的StorageImpl对象
  return custom_storage_impl;
}

// 注册我们的虚假StorageImpl创建方法
void custom_storage_registry() {
  // 设置PrivateUse1设备类型的StorageImpl创建函数
  c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &make_custom_storage_impl);
}

// 检查是否调用了自定义StorageImpl
bool custom_storageImpl_called() {
  // 如果StorageImpl计数器大于上次保存的值
  if (storageImpl_counter > last_storageImpl_saved_value) {
    // 更新保存的值为当前计数器的值
    last_storageImpl_saved_value = storageImpl_counter;
    // 返回true，表示调用了自定义StorageImpl
    return true;
  }
  // 否则返回false
  return false;
}

// 基本的虚假加法函数
at::Tensor custom_add_Tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  // 增加加法计数器
  add_counter += 1;
  // 由于此自定义设备仅用于测试，不实现核心函数
  return at::empty(self.sizes(), self.options());
}

// 基本的绝对值函数
at::Tensor& custom_abs_out(const at::Tensor& self, at::Tensor& out) {
  // 调用native库中的abs_out函数
  return at::native::abs_out(self, out);
}

// 为我们的自定义设备创建虚假的分配器，实际上使用CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;
  // 重写分配函数
  at::DataPtr allocate(size_t nbytes) override {
    // 分配CPU内存
    void* data = c10::alloc_cpu(nbytes);
    // 返回数据指针和自定义设备的描述信息
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, custom_device_index)};
  }

  // 静态的释放函数
  static void ReportAndDelete(void* ptr) {
    // 如果指针为空则直接返回
    if (!ptr) {
      return;
    }
    // 释放CPU内存
    c10::free_cpu(ptr);
  }

  // 返回原始释放函数指针
  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  // 复制数据的函数
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    // 使用默认的数据复制函数
    default_copy_data(dest, src, count);
  }
};

// 注册我们的虚假分配器
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);
// 创建一个基本的空函数，以便可以直接在自定义设备上构造张量
// 这个虚拟测试设备将只使用CPU分配器，并忽略固定内存。
at::Tensor custom_empty_memory_format(at::IntArrayRef size,
                                      std::optional<at::ScalarType> dtype,
                                      std::optional<at::Layout> layout,
                                      std::optional<at::Device> device,
                                      std::optional<bool> pin_memory,
                                      std::optional<at::MemoryFormat> memory_format) {
  // 定义一个私有的 DispatchKeySet，这里使用 PrivateUse1
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  // 调用底层的 empty_generic 函数创建张量，使用自定义的全局分配器和私有 DispatchKeySet
  return at::detail::empty_generic(size,
                                   &global_custom_alloc,
                                   private_use_ks,
                                   c10::dtype_or_default(dtype),
                                   memory_format);
}

// 创建一个基于符号整数的空函数，用于在自定义设备上构造张量
at::Tensor custom_empty_symint(c10::IntArrayRef size,
                               std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout,
                               std::optional<at::Device> device,
                               std::optional<bool> pin_memory,
                               std::optional<at::MemoryFormat> memory_format) {
  // 定义一个私有的 DispatchKeySet，这里使用 PrivateUse1
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  // 调用底层的 empty_generic 函数创建张量，使用自定义的全局分配器和私有 DispatchKeySet
  return at::detail::empty_generic(size,
                                   &global_custom_alloc,
                                   private_use_ks,
                                   c10::dtype_or_default(dtype),
                                   memory_format);
}

// 不实现具体的填充逻辑，只返回传入的张量自身
at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  // 不实现具体的填充逻辑
  return self;
}

// 从虚拟张量中不安全地使用数据指针创建一个CPU张量，并共享数据指针
at::Tensor unsafe_create_cpu_tensor_from_dummy_tensor(const at::Tensor& src) {
  // 检查源张量的设备类型是否为 PrivateUse1 类型
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "Only support dummy device.");
  // 获取源张量的大小和步长
  const auto& sizes_ = src.sizes();
  const auto& strides_ = src.strides();
  auto storage_offset_ = src.storage_offset();
  // 检查张量的大小是否为非负数
  at::detail::check_size_nonnegative(sizes_);

  // 计算张量的存储字节数
  size_t size_bytes = at::detail::computeStorageNbytes(sizes_, strides_,
                                                       src.element_size(),
                                                       storage_offset_);

  // 创建一个新的 DataPtr，使用源张量的可变数据指针，并指定空函数以释放其内存，设备类型为 CPU
  at::DataPtr data_ptr =
    c10::InefficientStdFunctionContext::makeDataPtr(src.storage().mutable_data_ptr().get(),
                                                    [](void*){}, at::kCPU);

  // 创建一个 Storage 对象，使用计算得到的存储字节数和新的 DataPtr
  c10::Storage storage{c10::Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr),
                       /*allocator=*/nullptr,
                       /*resizable=*/false};

  // 使用计算得到的大小、步长、存储偏移和新的 Storage 创建一个新的 CPU 张量
  return at::empty(sizes_, src.options().dtype(), /*layout=*/c10::kStrided,
                   /*device=*/c10::Device(c10::DeviceType::CPU),
                   /*pin_memory=*/false, /*memory_format=*/c10::MemoryFormat::Contiguous, std::move(storage));
}
    /*allocator=*/&global_custom_alloc, /*resizeable=*/false};
    /* 使用全局自定义分配器作为分配器，不允许调整大小 */

  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
    /* 创建 CPU DispatchKey 的集合 */

  at::Tensor tensor = at::detail::make_tensor<c10::TensorImpl>(
       std::move(storage), cpu_ks, src.dtype());
    /* 利用给定的存储、DispatchKey 集合和数据类型创建张量 */

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
    /* 获取张量的底层实现指针 */

  tensor_impl->set_sizes_and_strides(sizes_, strides_);
    /* 设置张量的尺寸和步长 */

  tensor_impl->set_storage_offset(storage_offset_);
    /* 设置张量的存储偏移量 */

  return tensor;
    /* 返回创建好的张量 */
// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  // 检查源张量是在 CPU 上，或者在自定义设备 PrivateUse1 上
  TORCH_CHECK(
      self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1,
      "Dummy test only allows copy from cpu -> dummy device.");
  // 检查目标张量是在 CPU 上，或者在自定义设备 PrivateUse1 上
  TORCH_CHECK(
      dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1,
      "Dummy test only allows copy from cpu -> dummy device.");

  // 对于基本用例的一些虚拟断言：输入张量具有相同的大小和数据类型，并且是连续的
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());

  if (self.is_contiguous() && dst.is_contiguous()) {
    // 如果源和目标张量都是连续的，则使用 memcpy 进行内存拷贝
    std::memcpy(dst.storage().data_ptr().get(),
                self.storage().data_ptr().get(),
                self.storage().nbytes());
  } else {
    // 如果不是连续的，则创建 CPU 张量并复制数据
    at::Tensor cpu_self = unsafe_create_cpu_tensor_from_dummy_tensor(self);
    at::Tensor cpu_dst = unsafe_create_cpu_tensor_from_dummy_tensor(dst);
    cpu_dst.copy_(cpu_self);
  }

  // 返回目标张量
  return dst;
}

// 将 custom__copy_from 函数封装，设置 non_blocking 参数为 false
at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  return custom__copy_from(self, dst, false);
}

// 创建一个具有给定大小、步幅和可选属性的新张量
at::Tensor custom_empty_strided(c10::IntArrayRef size,
                                c10::IntArrayRef stride,
                                std::optional<at::ScalarType> dtype_opt,
                                std::optional<at::Layout> layout_opt,
                                std::optional<at::Device> device_opt,
                                std::optional<bool> pin_memory_opt) {
  // 定义一个 DispatchKeySet，用于自定义私有使用
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  // 获取数据类型，如果未提供则使用默认数据类型
  auto dtype = c10::dtype_or_default(dtype_opt);
  // 调用内部函数创建空的步幅张量，并指定私有使用的 DispatchKeySet 和数据类型
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

// 为基本用例提供一些设置操作，设置结果张量的存储
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src) {
  // 计算新的大小，根据存储的字节数和结果张量的数据类型大小计算
  int64_t new_size = static_cast<int64_t>(src.nbytes() / result.dtype().itemsize());
  // 设置步幅为空
  c10::IntArrayRef stride = {};
  // 设置结果张量的存储偏移为0
  result.unsafeGetTensorImpl()->set_storage_offset(0);
  // 如果步幅不为空，则将其转换为 OptionalIntArrayRef 类型
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : c10::nullopt;
  // 调用 CPU 实现的 resize_impl_ 函数来调整张量的大小
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(),
                               new_size, stride_opt,
                               /*resize_storage=*/!result.is_meta());
  // 返回设置后的结果张量
  return result;
}
// 在给定的张量 `result` 上设置存储偏移量为 `storage_offset`
at::Tensor& custom_set_source_Storage_storage_offset(at::Tensor& result,
                                                     c10::Storage storage,
                                                     int64_t storage_offset,
                                                     c10::IntArrayRef size,
                                                     c10::IntArrayRef stride) {
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  // 如果 `stride` 不为 nullptr，则使用其创建可选的整数数组引用；否则设置为 `c10::nullopt`
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : c10::nullopt;
  // 调用 CPU 实现的 resize 函数来调整张量的大小和步长
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(),
                               size, stride_opt,
                               /*resize_storage=*/!result.is_meta());
  return result;
}

// 与固定内存相关的基本虚拟函数。
std::vector<void*> custom_pinned_data_ptr;

// 对给定的张量 `self` 进行内存固定操作，记录固定的数据指针
at::Tensor custom__pin_memory(const at::Tensor& self, std::optional<at::Device> device) {
  TORCH_CHECK(
      self.device().is_cpu(),
      "cannot pin '",
      self.toString(),
      "' only dense CPU tensors can be pinned");

  // 记录固定的数据指针
  at::Tensor dump_pinned_tensor = self * 1.0;
  custom_pinned_data_ptr.push_back(dump_pinned_tensor.storage().data_ptr().get());

  return dump_pinned_tensor;
}

// 检查给定的张量 `self` 是否被固定在内存中
bool custom_is_pinned(const at::Tensor& self, std::optional<at::Device> device) {
  // 只有 CPU 张量可以被固定
  if (!self.is_cpu()) {
    return false;
  }

  void* query_pinned_ptr = self.storage().data_ptr().get();
  // 遍历固定的数据指针列表，检查是否存在 `query_pinned_ptr`
  for (const auto& iter_ptr : custom_pinned_data_ptr) {
    if (iter_ptr == query_pinned_ptr) {
      return true;
    }
  }
  return false;
}

// 调整给定张量 `self` 的大小和存储，可能改变内存格式
const at::Tensor& custom_resize_(const at::Tensor& self, at::IntArrayRef size,
                          std::optional<at::MemoryFormat> optional_memory_format) {
  at::TensorImpl* tensor_impl = self.unsafeGetTensorImpl();
  tensor_impl->set_sizes_contiguous(size);
  const auto itemsize = tensor_impl->dtype().itemsize();
  const auto offset = tensor_impl->storage_offset();
  const auto storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, offset);
  // 调用 CPU 实现的 maybe_resize_storage_cpu 函数来重新分配足够的内存空间
  at::native::maybe_resize_storage_cpu(tensor_impl, storage_size);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    // 检查内存格式是否被支持
    TORCH_CHECK(
        memory_format != at::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    // 将张量重排列以匹配给定的内存格式
    tensor_impl->empty_tensor_restride(memory_format);
  }
  return self;
}

// 自定义的可重写的缩放点积注意力函数，返回多个张量和整数
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
custom_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const c10::optional<at::Tensor> & attn_bias,
   `
    # 定义 dropout 概率，表示在训练过程中随机丢弃的比例
    double dropout_p,
    # 是否为因果模型，决定是否使用因果掩码
    bool is_causal,
    # 是否返回调试掩码，用于调试时查看注意力掩码
    bool return_debug_mask,
    # 可选的缩放因子，用于缩放注意力分数
    std::optional<double> scale) {
  # 获取查询张量的批次大小
  const int64_t batch_size = query.size(0);
  # 获取查询张量的头数
  const int64_t num_heads = query.size(1);
  # 获取查询张量每个头的维度大小
  const int64_t head_dim_qk = query.size(3);
  # 获取值张量每个头的维度大小
  const int64_t head_dim_v = value.size(3);
  # 获取查询张量的最大序列长度
  const int64_t max_seqlen_q = query.size(2);
  # 获取键值张量的最大序列长度
  const int64_t max_seqlen_kv = key.size(2);

  # 获取查询张量的选项（数据类型等信息）
  auto opts = query.options();
  # 创建一个形状为(batch_size, num_heads, max_seqlen_q, head_dim_v)的张量，用于存储输出
  auto output = at::empty({batch_size, num_heads, max_seqlen_q, head_dim_v}, opts);
  # 创建一个形状为(batch_size, num_heads, max_seqlen_q)的张量，用于存储对数和指数的结果
  auto logsumexp = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  # 创建一个形状为(batch_size, num_heads, max_seqlen_q, max_seqlen_kv)的张量，用于存储调试注意力掩码
  auto debug_attn_mask = at::empty({batch_size, num_heads, max_seqlen_q, max_seqlen_kv},
                                   opts.dtype(at::kFloat));
  # 创建一个空的张量，用于存储 Philox 种子
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  # 创建一个空的张量，用于存储 Philox 偏移量
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  # 返回一个元组，包含输出张量、logsumexp张量、两个空张量、最大序列长度信息、Philox 种子、Philox 偏移量和调试注意力掩码
  return std::make_tuple(output, logsumexp, at::Tensor(), at::Tensor(), max_seqlen_q, max_seqlen_kv, philox_seed, philox_offset, debug_attn_mask);
}
// 定义一个函数 custom_scaled_dot_product_fused_attention_overrideable_backward，用于计算自定义的缩放点积注意力机制的反向传播。
// 函数接受多个张量作为输入，并返回一个包含梯度的元组。
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
custom_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor & grad_out,    // 梯度输出张量
    const at::Tensor & query,       // 查询张量
    const at::Tensor & key,         // 键张量
    const at::Tensor & value,       // 值张量
    const at::Tensor & attn_bias,   // 注意力偏置张量
    std::array<bool,4> grad_input_mask,  // 梯度输入掩码数组
    const at::Tensor & out,         // 输出张量
    const at::Tensor & logsumexp,   // 对数总和张量
    const at::Tensor & cum_seq_q,   // 累计查询张量
    const at::Tensor & cum_seq_k,   // 累计键张量
    int64_t max_q,                  // 最大查询值
    int64_t max_k,                  // 最大键值
    double dropout_p,               // 丢弃概率
    bool is_causal,                 // 是否因果
    const at::Tensor & philox_seed, // 随机数种子张量
    const at::Tensor & philox_offset,   // 随机数偏移张量
    std::optional<double> scale) {  // 缩放值（可选）
  // 返回一个元组，包含与输入张量相同形状的空张量
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
          at::empty_like(query),     // 返回空的与查询张量相同形状的张量
          at::empty_like(key),       // 返回空的与键张量相同形状的张量
          at::empty_like(value),     // 返回空的与值张量相同形状的张量
          at::empty_like(attn_bias));   // 返回空的与注意力偏置张量相同形状的张量
}

// 此宏执行重要操作。
// 使用 TORCH_LIBRARY_IMPL，可以为后端注册自定义内核。
// 对于开放注册，我们将所有内核注册到 PrivateUse1 调度键。
// 在本文件的后面，我们将一个自定义设备映射到 PrivateUse1 设备类型，
// 允许将张量放在您的 custom_device 上的用户代码最终被连接到这里注册的内核。
//
// 此宏将您的内核注册到 PyTorch 调度程序。
// 有关调度程序的更多详细信息，请访问 http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/。
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs.out", &custom_abs_out);    // 注册自定义 abs 函数
  m.impl("add.Tensor", &custom_add_Tensor);   // 注册自定义 add 函数
  m.impl("empty.memory_format", &custom_empty_symint);   // 注册自定义 empty 函数
  m.impl("fill_.Scalar", &custom_fill__scalar);   // 注册自定义 fill 函数
  m.impl("_copy_from", &custom__copy_from);   // 注册自定义 _copy_from 函数
  m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);   // 注册自定义 _copy_from_and_resize 函数
  m.impl("empty_strided", &custom_empty_strided);   // 注册自定义 empty_strided 函数
  m.impl("set_.source_Storage", &custom_set_source_Storage);   // 注册自定义 set_ 函数
  m.impl("set_.source_Storage_storage_offset",&custom_set_source_Storage_storage_offset);   // 注册自定义 set_ 函数
  m.impl("_pin_memory", &custom__pin_memory);   // 注册自定义 _pin_memory 函数
  m.impl("is_pinned", &custom_is_pinned);   // 注册自定义 is_pinned 函数
  m.impl("resize_", &custom_resize_);   // 注册自定义 resize 函数
  m.impl("as_strided", at::native::as_strided_tensorimpl);   // 注册标准库中的 as_strided 函数
  m.impl("quantize_per_tensor", at::native::quantize_per_tensor);   // 注册标准库中的 quantize_per_tensor 函数
  m.impl("_fused_sdp_choice", &_fused_sdp_choice_privateuse1);   // 注册自定义 _fused_sdp_choice 函数
  m.impl("_scaled_dot_product_fused_attention_overrideable", &custom_scaled_dot_product_fused_attention_overrideable);   // 注册自定义 _scaled_dot_product_fused_attention_overrideable 函数
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward", &custom_scaled_dot_product_fused_attention_overrideable_backward);   // 注册自定义 _scaled_dot_product_fused_attention_overrideable_backward 函数
}

// 自定义 CPU 回退函数
void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 调用原生的 CPU 回退函数
  at::native::cpu_fallback(op, stack);
}
// 注册 PrivateUse1 实现的 ATen 库函数
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // 注册 sub.Tensor 实现为 custom_cpu_fallback 函数
  m.impl("sub.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  // 注册 _foreach_add.List 实现为 custom_cpu_fallback 函数
  m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  // 注册 index.Tensor 实现为 custom_cpu_fallback 函数
  m.impl("index.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  // 注册 triu_indices 实现为 custom_cpu_fallback 函数
  m.impl("triu_indices", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

// 返回一个自定义的设备类型 PrivateUse1 的设备对象
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

// 检查是否调用了 custom_add_called 函数
bool custom_add_called() {
  bool called = false;
  // 如果 add_counter 大于 last_saved_value，则设置 called 为 true，并更新 last_saved_value
  if (add_counter > last_saved_value) {
    called = true;
    last_saved_value = add_counter;
  }
  return called;
}

// PrivateGeneratorImpl 类继承自 ATen 的 CPUGeneratorImpl 类，用于 PrivateUse1 设备生成器的实现
class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
  // 构造函数，根据给定的设备索引创建 PrivateUse1 设备对象，并设置 DispatchKeySet
  PrivateGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~PrivateGeneratorImpl() override = default;
};

// 创建并返回 PrivateUse1 设备的生成器
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

// 注册 PrivateUse1 生成器的第一个实例
void register_generator_first() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

// 注册 PrivateUse1 生成器的第二个实例
void register_generator_second() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

// 设置自定义设备索引为给定的 device_index
void set_custom_device_index(c10::DeviceIndex device_index) {
  custom_device_index = device_index;
}

// FooHooksInterface 结构体实现了 PrivateUse1HooksInterface 接口
struct FooHooksInterface : public at::PrivateUse1HooksInterface {
  ~FooHooksInterface() override = default;
  // 获取指定设备索引的默认生成器
  const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) override {
    // 创建并返回 PrivateUse1 设备的默认生成器
    static auto device_gen = make_generator_privateuse1(device_index);
    return device_gen;
  }
};

// FooHooksArgs 结构体用于传递给 PrivateUse1HooksRegistry 的参数
struct FooHooksArgs : public at::PrivateUse1HooksArgs {};

// 定义 PrivateUse1HooksRegistry 注册表，用于管理 PrivateUse1HooksInterface 和 FooHooksArgs
TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, FooHooksInterface, FooHooksArgs);
// 定义宏 REGISTER_PRIVATEUSE1_HOOKS(clsname)，用于注册 PrivateUse1 钩子类
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

// 定义 PrivateUse1HooksRegistry 的实现，用于管理 PrivateUse1 钩子类和参数
C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, FooHooksInterface, FooHooksArgs)

// 获取并返回 PrivateUse1 钩子接口的实例
static at::PrivateUse1HooksInterface* get_private_hooks() {
  static at::PrivateUse1HooksInterface* privateuse1_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
    // 尝试从 PrivateUse1HooksRegistry 创建 PrivateUse1HooksInterface 实例
    privateuse1_hooks = PrivateUse1HooksRegistry()->Create("PrivateUse1Hooks", {}).release();
    // 如果创建失败，则默认使用 FooHooksInterface 的实例
    if (!privateuse1_hooks) {
      privateuse1_hooks = new FooHooksInterface();
    }
  });
  return privateuse1_hooks;
}

// 注册 PrivateUse1 钩子接口的实现
void register_hook() {
  at::RegisterPrivateUse1HooksInterface(get_private_hooks());
}
const at::Generator& default_generator(c10::DeviceIndex device_index) {
    // 返回全局上下文的默认生成器，针对指定的设备索引
    return at::globalContext().defaultGenerator(at::Device(c10::DeviceType::PrivateUse1, device_index));;
}

struct CustomAutogradFnReturnsSelf : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {

  static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor self) {
    // 在前向传播中，直接返回输入张量自身
    return self;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
    // 在反向传播中，对输入梯度进行缩放
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing : public torch::autograd::Function<CustomAutogradFnAliasing> {

  static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor self) {
    // 在前向传播中，调用张量的视图操作，并返回结果
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
    // 在反向传播中，对输入梯度进行缩放
    return {grad_output[0] * 0.5};
  }
};

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  // 调用自定义的自动求导函数，返回输入张量自身
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  // 调用自定义的自动求导函数，返回输入张量经过视图操作后的结果
  return CustomAutogradFnAliasing::apply(x);
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 在 Python 绑定中注册自定义设备对象的接口函数
    m.def("custom_device", &get_custom_device, "get custom device object");
    // 检查自定义加法函数是否被调用的接口函数
    m.def("custom_add_called", &custom_add_called, "check if our custom add function was called");
    // 注册自定义设备的生成器函数接口函数
    m.def("register_generator_first", &register_generator_first, "register generator for custom device firstly");
    m.def("register_generator_second", &register_generator_second, "register generator for custom device secondly");
    // 设置自定义设备索引的接口函数
    m.def("set_custom_device_index", &set_custom_device_index, "set custom device index");
    // 设置自定义存储注册表的接口函数
    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
    // 检查自定义存储实现是否被调用的接口函数
    m.def("custom_storageImpl_called", &custom_storageImpl_called, "check if our custom abs function was called");
    // 设置伪造的张量后端元数据函数的接口函数
    m.def("custom_set_backend_meta", &custom_set_backend_meta, "a fake set tensor BackendMeta function");
    // 检查后端元数据序列化是否正确的接口函数
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    // 注册自定义序列化函数的接口函数
    m.def("custom_serialization_registry", &custom_serialization_registry, "register custom serialization function");
    // 注册私有使用1号设备的钩子函数的接口函数
    m.def("register_hook", &register_hook, "register_hook for privateuse1");
    // 注册私有使用1号设备的默认生成器函数的接口函数
    m.def("default_generator", &default_generator, "default_generator for privateuse1");

    // 以下代码用于更轻松地测试在 C++ 中对自定义自动求导函数进行 torch.compile 编译
    m.def("custom_autograd_fn_returns_self", &custom_autograd_fn_returns_self);
}
# 定义了一个 Torch 扩展库 _test_funcs，注册了自定义的自动微分函数别名
TORCH_LIBRARY(_test_funcs, m) {
  # 使用 m.def() 方法将 custom_autograd_fn_aliasing 函数注册到 _test_funcs 扩展库中
  m.def("custom_autograd_fn_aliasing(Tensor(a) input)-> Tensor(a)");
}

# 在 AutogradCPU 模块中实现了 _test_funcs 扩展库中的 custom_autograd_fn_aliasing 函数
TORCH_LIBRARY_IMPL(_test_funcs, AutogradCPU, m) {
  # 使用 m.impl() 方法将 custom_autograd_fn_aliasing 函数的实现与对应的 C++ 函数 custom_autograd_fn_aliasing 绑定
  m.impl("custom_autograd_fn_aliasing", &custom_autograd_fn_aliasing);
}
```