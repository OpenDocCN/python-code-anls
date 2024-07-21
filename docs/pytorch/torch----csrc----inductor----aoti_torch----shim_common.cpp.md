# `.\pytorch\torch\csrc\inductor\aoti_torch\shim_common.cpp`

```
// 包含 C10 库的头文件，这些头文件定义了 C10 核心数据类型和操作
#include <c10/core/DeviceType.h>
#include <c10/core/GradMode.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

// 包含 Torch 的头文件，用于特定模块的支持和功能
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/mkldnn_tensor.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/inductor_ops.h>

// 包含 C++ 标准库的头文件
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>

// 根据预处理宏选择性地包含 ATen 库的函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_mm.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/view_as_real_ops.h>
#include <ATen/ops/view_ops.h>
#endif

using namespace torch::aot_inductor;

namespace {
// 定义静态函数，根据设备类型和设备索引返回 C10 设备对象
static c10::Device c10_device(int32_t device_type, int32_t device_index) {
  // 如果设备类型为 CPU，返回对应的 C10 设备对象
  if (device_type == aoti_torch_device_type_cpu()) {
    return c10::Device(static_cast<c10::DeviceType>(device_type));
  } else {
    // 否则返回带有设备类型和索引的 C10 设备对象
    return c10::Device(
        static_cast<c10::DeviceType>(device_type),
        static_cast<c10::DeviceIndex>(device_index));
  }
}
} // namespace

// 返回表示 CPU 设备类型的整数常量
int32_t aoti_torch_device_type_cpu() {
  return (int32_t)c10::DeviceType::CPU;
}

// 返回表示 CUDA 设备类型的整数常量
int32_t aoti_torch_device_type_cuda() {
  return (int32_t)c10::DeviceType::CUDA;
}

// 使用宏定义实现不同 Torch 数据类型到整数的映射函数
#define AOTI_TORCH_DTYPE_IMPL(dtype, stype) \
  int32_t aoti_torch_dtype_##dtype() {      \
    return (int32_t)c10::ScalarType::stype; \
  }

// 映射不同 Torch 数据类型到整数的具体实现
AOTI_TORCH_DTYPE_IMPL(float8_e5m2, Float8_e5m2)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fn, Float8_e4m3fn)
AOTI_TORCH_DTYPE_IMPL(float8_e5m2fnuz, Float8_e5m2fnuz)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fnuz, Float8_e4m3fnuz)
AOTI_TORCH_DTYPE_IMPL(bfloat16, BFloat16)
AOTI_TORCH_DTYPE_IMPL(float16, Half)
AOTI_TORCH_DTYPE_IMPL(float32, Float)
AOTI_TORCH_DTYPE_IMPL(float64, Double)
AOTI_TORCH_DTYPE_IMPL(uint8, Byte)
AOTI_TORCH_DTYPE_IMPL(uint16, UInt16)
AOTI_TORCH_DTYPE_IMPL(uint32, UInt32)
AOTI_TORCH_DTYPE_IMPL(uint64, UInt64)
AOTI_TORCH_DTYPE_IMPL(int8, Char)
AOTI_TORCH_DTYPE_IMPL(int16, Short)
AOTI_TORCH_DTYPE_IMPL(int32, Int)
AOTI_TORCH_DTYPE_IMPL(int64, Long)
AOTI_TORCH_DTYPE_IMPL(bool, Bool)
AOTI_TORCH_DTYPE_IMPL(complex32, ComplexHalf)
AOTI_TORCH_DTYPE_IMPL(complex64, ComplexFloat)
#define AOTI_TORCH_DTYPE_IMPL(complex128, ComplexDouble)
// 定义宏 AOTI_TORCH_DTYPE_IMPL，将 complex128 映射为 ComplexDouble

#undef AOTI_TORCH_DTYPE_IMPL
// 取消宏 AOTI_TORCH_DTYPE_IMPL 的定义

int32_t aoti_torch_layout_strided() {
  return (int32_t)at::kStrided;
}
// 返回 ATen 张量的布局类型 kStrided 对应的整数值

int32_t aoti_torch_layout__mkldnn() {
  return (int32_t)at::kMkldnn;
}
// 返回 ATen 张量的布局类型 kMkldnn 对应的整数值

#define AOTI_TORCH_ITEM_IMPL(dtype, ctype)                     \
  AOTITorchError aoti_torch_item_##dtype(                      \
      AtenTensorHandle tensor, ctype* ret_value) {             \
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({               \
      at::Tensor* t = tensor_handle_to_tensor_pointer(tensor); \
      *ret_value = t->item().to<ctype>();                      \
    });                                                        \
  }
// 定义宏 AOTI_TORCH_ITEM_IMPL，用于生成获取 ATen 张量指定数据类型值的函数模板

AOTI_TORCH_ITEM_IMPL(float32, float)
AOTI_TORCH_ITEM_IMPL(float64, double)
AOTI_TORCH_ITEM_IMPL(uint8, uint8_t)
AOTI_TORCH_ITEM_IMPL(uint16, uint16_t)
AOTI_TORCH_ITEM_IMPL(uint32, uint32_t)
AOTI_TORCH_ITEM_IMPL(uint64, uint64_t)
AOTI_TORCH_ITEM_IMPL(int8, int8_t)
AOTI_TORCH_ITEM_IMPL(int16, int16_t)
AOTI_TORCH_ITEM_IMPL(int32, int32_t)
AOTI_TORCH_ITEM_IMPL(int64, int64_t)
AOTI_TORCH_ITEM_IMPL(bool, bool)
#undef AOTI_TORCH_ITEM_IMPL
// 使用宏 AOTI_TORCH_ITEM_IMPL 生成各种数据类型获取函数，最后取消宏的定义

#define AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(dtype, ctype, ttype)                  \
  AOTITorchError aoti_torch_scalar_to_tensor_##dtype(                          \
      ctype value, AtenTensorHandle* ret_new_tensor) {                         \
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({                               \
      *ret_new_tensor =                                                        \
          new_tensor_handle(at::scalar_tensor(value, c10::ScalarType::ttype)); \
    });                                                                        \
  }
// 定义宏 AOTI_TORCH_SCALAR_TO_TENSOR_IMPL，用于生成将标量转换为 ATen 张量的函数模板

AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float32, float, Float)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float64, double, Double)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint8, uint8_t, Byte)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint16, uint16_t, UInt16)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint32, uint32_t, UInt32)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint64, uint64_t, UInt64)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int8, int8_t, Char)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int16, int16_t, Short)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int32, int32_t, Int)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int64, int64_t, Long)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(bool, bool, Bool)
#undef AOTI_TORCH_SCALAR_TO_TENSOR_IMPL
// 使用宏 AOTI_TORCH_SCALAR_TO_TENSOR_IMPL 生成各种数据类型标量转换为张量的函数，最后取消宏的定义

bool aoti_torch_grad_mode_is_enabled() {
  return c10::GradMode::is_enabled();
}
// 返回当前是否启用了梯度模式的布尔值

void aoti_torch_grad_mode_set_enabled(bool enabled) {
  return c10::GradMode::set_enabled(enabled);
}
// 设置是否启用梯度模式的函数

AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    delete t;
  });
}
// 定义函数用于删除 ATen 张量对象的函数，同时处理异常并转换为错误代码

AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取 ATen 张量数据指针并将其存储在返回的数据指针中
    *ret_data_ptr = t->data_ptr();
  });
}
// 定义函数用于获取 ATen 张量数据指针的函数，同时处理异常并转换为错误代码
    # 检查指针 t 是否指向的数据类型为 mkldnn
    if (t->is_mkldnn()) {
      # 如果是 mkldnn 数据类型，则使用专门的函数获取数据指针并赋值给 ret_data_ptr
      *ret_data_ptr = data_ptr_from_mkldnn(t);
    } else {
      # 如果不是 mkldnn 数据类型，则直接获取普通数据指针并赋值给 ret_data_ptr
      *ret_data_ptr = t->data_ptr();
    }
  });
# 获取PyTorch张量的存储大小
AOTITorchError aoti_torch_get_storage_size(
    AtenTensorHandle tensor,
    int64_t* ret_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的存储空间大小并赋值给ret_size
    *ret_size = t->storage().nbytes();
  });
}

# 获取PyTorch张量的维度
AOTITorchError aoti_torch_get_dim(AtenTensorHandle tensor, int64_t* ret_dim) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的维度并赋值给ret_dim
    *ret_dim = t->dim();
  });
}

# 获取PyTorch张量的元素数量
AOTITorchError aoti_torch_get_numel(
    AtenTensorHandle tensor,
    int64_t* ret_numel) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的元素总数并赋值给ret_numel
    *ret_numel = t->numel();
  });
}

# 获取PyTorch张量的大小（尺寸）
AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_sizes) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的尺寸数组的指针，并赋给ret_sizes
    *ret_sizes = const_cast<int64_t*>(t->sizes().data());
  });
}

# 获取PyTorch张量某一维度的大小
AOTITorchError aoti_torch_get_size(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量在指定维度d上的大小并赋值给ret_size
    *ret_size = t->size(d);
  });
}

# 获取PyTorch张量的步长数组的指针
AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的步长数组的指针，并赋给ret_strides
    *ret_strides = const_cast<int64_t*>(t->strides().data());
  });
}

# 获取PyTorch张量某一维度的步长
AOTITorchError aoti_torch_get_stride(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_stride) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量在指定维度d上的步长并赋值给ret_stride
    *ret_stride = t->stride(d);
  });
}

# 获取PyTorch张量的数据类型
AOTITorchError aoti_torch_get_dtype(
    AtenTensorHandle tensor,
    int32_t* ret_dtype) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的数据类型并转换为整数，赋值给ret_dtype
    *ret_dtype = static_cast<int32_t>(t->scalar_type());
  });
}

# 获取PyTorch张量的设备类型
AOTITorchError aoti_torch_get_device_type(
    AtenTensorHandle tensor,
    int32_t* ret_device_type) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的设备类型并转换为整数，赋值给ret_device_type
    *ret_device_type = static_cast<int32_t>(t->device().type());
  });
}

# 获取PyTorch张量的设备索引
AOTITorchError aoti_torch_get_device_index(
    AtenTensorHandle tensor,
    int32_t* ret_device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的设备索引并赋值给ret_device_index
    *ret_device_index = t->device().index();
  });
}

# 获取PyTorch张量的存储偏移
AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将AtenTensorHandle转换为PyTorch张量指针
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // 获取张量的存储偏移并赋值给ret_storage_offset
    *ret_storage_offset = t->storage_offset();
  });
}

# 重新解释PyTorch张量
AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    // 定义一个指向 int64_t 类型的常量指针，指向张量的步幅数组
    int64_t offset_increment,
    // 定义一个 int64_t 类型的变量，表示偏移增量
    AtenTensorHandle* ret_new_tensor) {
  // 开始异常处理的代码块
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将指向张量的句柄转换为张量指针
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    // 创建一个 c10::IntArrayRef 对象 sizes，用 sizes_ptr 初始化，长度为 ndim
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    // 创建一个 c10::IntArrayRef 对象 strides，用 strides_ptr 初始化，长度为 ndim
    c10::IntArrayRef strides(strides_ptr, ndim);
    // 调用 _reinterpret_tensor 函数重新解释 self_tensor 的形状和步幅，以及偏移增量
    // 返回一个新的张量句柄，并存储在 ret_new_tensor 指向的位置
    *ret_new_tensor = new_tensor_handle(torch::inductor::_reinterpret_tensor(
        *self_tensor, sizes, strides, offset_increment));
  });
// 实现函数 aoti_torch_create_tensor_from_blob_v2，根据给定参数创建张量，处理可能的异常
AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,                                      // 输入数据指针
    int64_t ndim,                                    // 张量维度
    const int64_t* sizes_ptr,                        // 每个维度的大小数组指针
    const int64_t* strides_ptr,                      // 每个维度的步幅数组指针
    int64_t storage_offset,                          // 存储偏移量
    int32_t dtype,                                   // 数据类型
    int32_t device_type,                             // 设备类型
    int32_t device_index,                            // 设备索引
    AtenTensorHandle* ret_new_tensor,                // 返回的新张量句柄
    int32_t layout,                                  // 张量布局
    const uint8_t* opaque_metadata,                  // 不透明元数据的指针
    int64_t opaque_metadata_size                     // 不透明元数据的大小
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);         // 创建 IntArrayRef 以表示大小数组
    c10::IntArrayRef strides(strides_ptr, ndim);     // 创建 IntArrayRef 以表示步幅数组
    c10::Device device = c10_device(device_type, device_index);  // 创建设备对象
    c10::TensorOptions options = c10::TensorOptions()  // 创建张量选项对象
        .device(device)                              // 设置设备
        .dtype(static_cast<c10::ScalarType>(dtype)); // 设置数据类型

    *ret_new_tensor = new_tensor_handle(
        // 如果输入数据非空，则使用 for_blob 创建张量
        (data != nullptr) ? at::for_blob(data, sizes)
                                .strides(strides)
                                .storage_offset(storage_offset)
                                .options(options)
                                .make_tensor()
                          // 如果输入数据为空，创建一个空的按步幅排列的张量
                          : at::empty_strided(sizes, strides, options));
  });
}
    // 如果布局为 MKLDNN，执行以下代码块
    if (layout == static_cast<int32_t>(at::kMkldnn)) {
      // 创建一个 sizes 的 c10::IntArrayRef，使用 sizes_ptr 指针和 ndim
      c10::IntArrayRef sizes(sizes_ptr, ndim);
      // 创建一个 strides 的 c10::IntArrayRef，使用 strides_ptr 指针和 ndim
      c10::IntArrayRef strides(strides_ptr, ndim);
      // 创建一个 c10::Device 对象，表示设备类型和设备索引
      c10::Device device = c10_device(device_type, device_index);
      // 调用 mkldnn_tensor_from_data_ptr 函数，返回一个 mkldnn_tensor，
      // 并将其封装成一个 Torch Tensor (OpaqueTensorImpl)，用于后续的 MKLDNN 操作。
      *ret_new_tensor = new_tensor_handle(mkldnn_tensor_from_data_ptr(
          data,
          sizes,
          static_cast<c10::ScalarType>(dtype),
          device,
          opaque_metadata,
          opaque_metadata_size));
    } else {
      // 如果布局不是 MKLDNN，则调用 aoti_torch_create_tensor_from_blob 函数来创建一个 Torch Tensor。
      aoti_torch_create_tensor_from_blob(
          data,
          ndim,
          sizes_ptr,
          strides_ptr,
          storage_offset,
          dtype,
          device_type,
          device_index,
          ret_new_tensor);
    }
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__embedding_bag(
    AtenTensorHandle weight,                // 权重张量句柄
    AtenTensorHandle indices,               // 索引张量句柄
    AtenTensorHandle offsets,               // 偏移张量句柄
    int32_t scale_grad_by_freq,             // 是否按频率缩放梯度
    int32_t mode,                           // 模式
    int32_t sparse,                         // 是否稀疏
    AtenTensorHandle per_sample_weights,    // 每样本权重（可选参数）
    int32_t include_last_offset,            // 是否包含最后一个偏移
    int32_t padding_idx,                    // 填充索引
    AtenTensorHandle* ret0,                 // 返回值0的新句柄（返回新引用）
    AtenTensorHandle* ret1,                 // 返回值1的新句柄（返回新引用）
    AtenTensorHandle* ret2,                 // 返回值2的新句柄（返回新引用）
    AtenTensorHandle* ret3                  // 返回值3的新句柄（返回新引用）
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto [r0, r1, r2, r3] = at::_embedding_bag(
        *tensor_handle_to_tensor_pointer(weight),    // 转换权重张量句柄为张量指针并使用
        *tensor_handle_to_tensor_pointer(indices),   // 转换索引张量句柄为张量指针并使用
        *tensor_handle_to_tensor_pointer(offsets),   // 转换偏移张量句柄为张量指针并使用
        scale_grad_by_freq,                         // 缩放梯度选项
        mode,                                       // 模式选项
        sparse,                                     // 稀疏选项
        pointer_to_optional(
            tensor_handle_to_tensor_pointer(per_sample_weights)),  // 转换每样本权重句柄为张量指针并使用（可选参数）
        include_last_offset,                        // 是否包含最后一个偏移选项
        padding_idx);                               // 填充索引

    *ret0 = new_tensor_handle(std::move(r0));       // 将返回值0移动到新的张量句柄
    *ret1 = new_tensor_handle(std::move(r1));       // 将返回值1移动到新的张量句柄
    *ret2 = new_tensor_handle(std::move(r2));       // 将返回值2移动到新的张量句柄
    *ret3 = new_tensor_handle(std::move(r3));       // 将返回值3移动到新的张量句柄
  });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__fft_c2c(
    AtenTensorHandle self,                      // 自身张量句柄
    const int64_t* dim_ptr,                     // 维度指针
    int64_t dim_size,                           // 维度大小
    int64_t normalization,                      // 归一化选项
    int32_t forward,                            // 正向/反向选项
    AtenTensorHandle* ret                       // 返回新引用的张量句柄
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto dim = c10::IntArrayRef(dim_ptr, dim_size);     // 创建维度数组引用
    *ret = new_tensor_handle(at::_fft_c2c(
        *tensor_handle_to_tensor_pointer(self),          // 转换自身张量句柄为张量指针并使用
        dim,                                            // 使用维度数组
        normalization,                                  // 归一化选项
        forward));                                      // 正向/反向选项

  });
}

AOTITorchError aoti_torch__scaled_dot_product_flash_attention_v2(
    AtenTensorHandle query,                             // 查询张量句柄
    AtenTensorHandle key,                               // 键张量句柄
    AtenTensorHandle value,                             // 值张量句柄
    double dropout_p,                                   // 丢弃概率
    int is_causal,                                      // 是否因果
    int return_debug_mask,                              // 返回调试掩码
    double* scale,                                      // 缩放（可选参数）
    AtenTensorHandle* ret0,                             // 返回值0的新句柄（返回新引用）
    AtenTensorHandle* ret1,                             // 返回值1的新句柄（返回新引用）
    AtenTensorHandle* ret2,                             // 返回值2的新句柄（返回新引用）
    AtenTensorHandle* ret3,                             // 返回值3的新句柄（返回新引用）
    int64_t* ret4,                                      // 返回值4
    int64_t* ret5,                                      // 返回值5
    AtenTensorHandle* ret6,                             // 返回值6的新句柄（返回新引用）
    AtenTensorHandle* ret7,                             // 返回值7的新句柄（返回新引用）
    AtenTensorHandle* ret8                              // 返回值8的新句柄（返回新引用）
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* query_tensor = tensor_handle_to_tensor_pointer(query);   // 转换查询张量句柄为张量指针
    at::Tensor* key_tensor = tensor_handle_to_tensor_pointer(key);       // 转换键张量句柄为张量指针
    at::Tensor* value_tensor = tensor_handle_to_tensor_pointer(value);   // 转换值张量句柄为张量指针
    auto optional_scale = pointer_to_optional(scale);                    // 创建缩放的可选参数

    auto [r0, r1, r2, r3, r4, r5, r6, r7, r8] = at::_scaled_dot_product_flash_attention(
        *query_tensor,                          // 使用查询张量
        *key_tensor,                            // 使用键张量
        *value_tensor,                          // 使用值张量
        dropout_p,                              // 使用丢弃概率
        is_causal,                              // 是否因果
        return_debug_mask,                      // 返回调试掩码
        optional_scale);                        // 使用缩放的可选参数

    *ret0 = new_tensor_handle(std::move(r0));   // 将返回值0移动到新的张量句柄
    *ret1 = new_tensor_handle(std::move(r1));   // 将返回值1移动到新的张量句柄
    *ret2 = new_tensor_handle(std::move(r2));   // 将返回值2移动到新的张量句柄
    *ret3 = new_tensor_handle(std::move(r3));   // 将返回值3移动到新的张量句柄
    *ret4 = r4;                                 // 直接赋值返回值4
    *ret5 = r5;                                 // 直接赋值返回值5
    *ret6 = new_tensor_handle(std::move(r6));   // 将返回值6移动到新的张量句柄
    *ret7 = new_tensor_handle(std::move(r7));   // 将返回值7移动到新的张量句柄
    *ret8 = new_tensor_handle(std::move(r8));   // 将返回值8移动到新的张量句柄
  });
}
    // 将 r1 移动构造为新的张量句柄，并赋给 ret1
    *ret1 = new_tensor_handle(std::move(r1));
    // 如果 ret2 不为 null，则将 r2 移动构造为新的张量句柄，并赋给 *ret2
    if (ret2) {
      *ret2 = new_tensor_handle(std::move(r2));
    }
    // 如果 ret3 不为 null，则将 r3 移动构造为新的张量句柄，并赋给 *ret3
    if (ret3) {
      *ret3 = new_tensor_handle(std::move(r3));
    }
    // 将 r4 转换为整数并赋给 *ret4
    *ret4 = r4.expect_int();
    // 将 r5 转换为整数并赋给 *ret5
    *ret5 = r5.expect_int();
    // 将 r6 移动构造为新的张量句柄，并赋给 *ret6
    *ret6 = new_tensor_handle(std::move(r6));
    // 将 r7 移动构造为新的张量句柄，并赋给 *ret7
    *ret7 = new_tensor_handle(std::move(r7));
    // 将 r8 移动构造为新的张量句柄，并赋给 *ret8
    *ret8 = new_tensor_handle(std::move(r8));
  });
}
// 函数：执行通过调用优化的注意力机制实现的缩放点积注意力操作
AOTITorchError aoti_torch__scaled_dot_product_flash_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    double scale,
    AtenTensorHandle* ret0, // 返回新引用
    AtenTensorHandle* ret1, // 返回新引用
    AtenTensorHandle* ret2, // 返回新引用
    AtenTensorHandle* ret3, // 返回新引用
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // 返回新引用
    AtenTensorHandle* ret7, // 返回新引用
    AtenTensorHandle* ret8 // 返回新引用
) {
  return aoti_torch__scaled_dot_product_flash_attention_v2(
      query,
      key,
      value,
      dropout_p,
      is_causal,
      return_debug_mask,
      &scale,
      ret0,
      ret1,
      ret2,
      ret3,
      ret4,
      ret5,
      ret6,
      ret7,
      ret8);
}

// 函数：执行通过调用高效注意力机制实现的缩放点积注意力操作
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch__scaled_dot_product_efficient_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    AtenTensorHandle attn_bias, // 可选参数
    int compute_log_sumexp,
    double dropout_p,
    int is_causal,
    double* scale, // 可选参数
    AtenTensorHandle* ret0, // 返回新引用
    AtenTensorHandle* ret1, // 返回新引用
    AtenTensorHandle* ret2, // 返回新引用
    AtenTensorHandle* ret3 // 返回新引用
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将查询、键、值张量从句柄转换为指针
    at::Tensor* query_tensor = tensor_handle_to_tensor_pointer(query);
    at::Tensor* key_tensor = tensor_handle_to_tensor_pointer(key);
    at::Tensor* value_tensor = tensor_handle_to_tensor_pointer(value);
    // 将注意力偏置从句柄转换为可选的张量指针
    auto optional_attn_bias =
        pointer_to_optional(tensor_handle_to_tensor_pointer(attn_bias));
    // 将缩放因子从指针转换为可选值
    auto optional_scale = pointer_to_optional(scale);
    // 调用内部的高效缩放点积注意力机制实现函数
    auto [r0, r1, r2, r3] = at::_scaled_dot_product_efficient_attention(
        *query_tensor,
        *key_tensor,
        *value_tensor,
        optional_attn_bias,
        compute_log_sumexp,
        dropout_p,
        is_causal,
        optional_scale);
    // 将计算结果转换为新的张量句柄，并分配给返回值参数
    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
    *ret2 = new_tensor_handle(std::move(r2));
    *ret3 = new_tensor_handle(std::move(r3));
  });
}

// 函数：执行通过调用卷积操作实现的卷积计算
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias, // 可选参数
    const int64_t* stride_ptr,
    int64_t stride_size,
    const int64_t* padding_ptr,
    int64_t padding_size,
    const int64_t* dilation_ptr,
    int64_t dilation_size,
    int transposed,
    const int64_t* output_padding_ptr,
    int64_t output_padding_size,
    int64_t groups,
    AtenTensorHandle* out // 返回新引用
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输入张量从句柄转换为指针
    at::Tensor* input_tensor = tensor_handle_to_tensor_pointer(input);
    // 将权重张量的句柄转换为张量指针
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    // 将偏置张量的句柄转换为张量指针
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);
    // 将偏置张量的指针转换为可选类型（optional）
    auto optional_bias = pointer_to_optional(bias_tensor);
    // 创建包含给定步幅的整数数组引用
    c10::IntArrayRef stride(stride_ptr, stride_size);
    // 创建包含给定填充的整数数组引用
    c10::IntArrayRef padding(padding_ptr, padding_size);
    // 创建包含给定扩张的整数数组引用
    c10::IntArrayRef dilation(dilation_ptr, dilation_size);
    // 创建包含给定输出填充的整数数组引用
    c10::IntArrayRef output_padding(output_padding_ptr, output_padding_size);

    // 在输出中存储新的张量句柄，该句柄指向卷积操作的结果张量
    *out = new_tensor_handle(at::convolution(
        *input_tensor,                    // 输入张量的指针
        *weight_tensor,                   // 权重张量的指针
        optional_bias,                    // 可选的偏置张量
        stride,                           // 步幅引用
        padding,                          // 填充引用
        dilation,                         // 扩张引用
        static_cast<bool>(transposed),    // 转置标志
        output_padding,                   // 输出填充引用
        groups));                         // 分组数量
}

// 定义函数 aoti_torch_new_uninitialized_tensor，创建一个新的未初始化的张量
AOTITorchError aoti_torch_new_uninitialized_tensor(AtenTensorHandle* ret) {
  // 转换异常为错误代码的宏，捕获异常并处理
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 分配一个新的空张量对象
    at::Tensor* out_tensor = new at::Tensor();
    // 将新创建的张量转换为张量句柄并返回
    *ret = tensor_pointer_to_tensor_handle(out_tensor);
  });
}

// 定义函数 aoti_torch__scaled_mm，执行缩放矩阵乘法
AOTITorchError aoti_torch__scaled_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle bias,
    int32_t* out_dtype,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle scale_result,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1) {
  // 转换异常为错误代码的宏，捕获异常并处理
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 获取输入张量指针
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);
    at::Tensor* scale_a_tensor = tensor_handle_to_tensor_pointer(scale_a);
    at::Tensor* scale_b_tensor = tensor_handle_to_tensor_pointer(scale_b);
    at::Tensor* scale_result_tensor =
        tensor_handle_to_tensor_pointer(scale_result);
    // 执行缩放矩阵乘法操作
    auto r0 = at::_scaled_mm(
        *self_tensor,
        *mat2_tensor,
        *scale_a_tensor,
        *scale_b_tensor,
        pointer_to_optional(bias_tensor),
        pointer_to_optional(scale_result_tensor),
        pointer_to_optional<c10::ScalarType>(out_dtype),
        use_fast_accum);
    // 将结果转换为张量句柄并返回
    *ret0 = new_tensor_handle(std::move(r0));
  });
}

// TODO: 实现一个更有效的版本，而不是调用 aten 函数
// 定义函数 aoti_torch_tensor_copy_，复制张量
AOTITorchError aoti_torch_tensor_copy_(
    AtenTensorHandle src,
    AtenTensorHandle dst) {
  // 调用函数 aoti_torch_copy_，在不阻塞的情况下执行复制操作
  return aoti_torch_copy_(dst, src, /*non_blocking=*/0);
}

// 定义函数 aoti_torch_assign_tensors，赋值张量
AOTITorchError aoti_torch_assign_tensors(
    AtenTensorHandle src,
    AtenTensorHandle dst) {
  // 转换异常为错误代码的宏，捕获异常并处理
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 获取源张量和目标张量的指针，并执行赋值操作
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::Tensor* dst_tensor = tensor_handle_to_tensor_pointer(dst);
    *dst_tensor = *src_tensor;
  });
}

// 定义函数 aoti_torch_assign_tensors_out，赋值张量到输出
AOTITorchError aoti_torch_assign_tensors_out(
    AtenTensorHandle src,
    AtenTensorHandle* ret_dst) {
  // 转换异常为错误代码的宏，捕获异常并处理
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 获取源张量指针，并将其复制到新的目标张量中
    at::Tensor* src_tensor_ptr = tensor_handle_to_tensor_pointer(src);
    at::Tensor dst_tensor = *src_tensor_ptr;
    // 将目标张量转换为张量句柄并返回
    *ret_dst = new_tensor_handle(std::move(dst_tensor));
  });
}

// 定义函数 aoti_torch_clone，克隆张量
AOTITorchError aoti_torch_clone(AtenTensorHandle self, AtenTensorHandle* ret) {
  // 转换异常为错误代码的宏，捕获异常并处理
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 获取输入张量指针，并执行克隆操作
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *ret = new_tensor_handle(self_tensor->clone());
  });
}

// TODO: 实现一个更有效的版本，而不是调用 aten 函数
// 定义函数 aoti_torch_addmm_out，执行加权矩阵乘法
AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha) {
  // 转换异常为错误代码的宏，捕获异常并处理
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 获取输出张量指针，并执行加权矩阵乘法操作
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    // 将 mat1 的张量句柄转换为张量指针
    at::Tensor* mat1_tensor = tensor_handle_to_tensor_pointer(mat1);
    // 将 mat2 的张量句柄转换为张量指针
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    // 使用 mat1_tensor 和 mat2_tensor 进行矩阵乘法运算，并将结果保存到 out_tensor 中，
    // beta 是乘积结果的缩放因子，alpha 是输入张量的缩放因子
    at::addmm_out(
        *out_tensor, *self_tensor, *mat1_tensor, *mat2_tensor, beta, alpha);
    // Lambda 表达式的结束
    });
    
    
    这段代码看起来是使用 C++ 或类似的语言编写的，使用了 `at::Tensor` 类型和相应的张量操作函数。
// TODO: implement a more efficient version instead of calling into aten
// 定义了一个函数aoti_torch_bmm_out，用于执行两个张量的批量矩阵乘法，并将结果存储在指定的输出张量中
AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,      // 输出张量的句柄
    AtenTensorHandle self,     // 第一个输入张量的句柄
    AtenTensorHandle mat2) {   // 第二个输入张量的句柄
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输出张量、第一个输入张量、第二个输入张量从句柄转换为Tensor指针，并执行批量矩阵乘法操作
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::bmm_out(*out_tensor, *self_tensor, *mat2_tensor);
  });
}

// TODO: implement a more efficient version instead of calling into aten
// 定义了一个函数aoti_torch_copy_，用于将源张量的数据复制到目标张量中
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_copy_(
    AtenTensorHandle self,     // 目标张量的句柄
    AtenTensorHandle src,      // 源张量的句柄
    int32_t non_blocking) {    // 非阻塞复制标志
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将目标张量和源张量从句柄转换为Tensor指针，并调用Tensor的copy_方法执行数据复制操作
    tensor_handle_to_tensor_pointer(self)->copy_(
        *tensor_handle_to_tensor_pointer(src), non_blocking);
  });
}

// TODO: implement a more efficient version instead of calling into aten
// 定义了一个函数aoti_torch_mm_out，用于执行两个张量的矩阵乘法，并将结果存储在指定的输出张量中
AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,      // 输出张量的句柄
    AtenTensorHandle self,     // 第一个输入张量的句柄
    AtenTensorHandle mat2) {   // 第二个输入张量的句柄
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输出张量、第一个输入张量、第二个输入张量从句柄转换为Tensor指针，并执行矩阵乘法操作
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::mm_out(*out_tensor, *self_tensor, *mat2_tensor);
  });
}

// 定义了一个函数aoti_torch_cpu_wrapped_fbgemm_pack_gemm_matrix_fp16，用于对FP16格式的权重张量执行FBGEMM的矩阵打包操作
AOTITorchError aoti_torch_cpu_wrapped_fbgemm_pack_gemm_matrix_fp16(
    AtenTensorHandle weight,   // 输入权重张量的句柄
    AtenTensorHandle* out) {  // 输出打包后张量的句柄指针
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输入权重张量从句柄转换为Tensor指针，并调用fbgemm_pack_gemm_matrix_fp16函数进行打包操作
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    *out = new_tensor_handle(at::fbgemm_pack_gemm_matrix_fp16(*weight_tensor));
  });
}

// 定义了一个函数aoti_torch_cpu_wrapped_fbgemm_linear_fp16_weight，用于执行FP16格式的线性层操作
AOTITorchError aoti_torch_cpu_wrapped_fbgemm_linear_fp16_weight(
    AtenTensorHandle input,    // 输入张量的句柄
    AtenTensorHandle weight,   // 权重张量的句柄
    AtenTensorHandle bias,     // 偏置张量的句柄
    int64_t out_channel,       // 输出通道数
    AtenTensorHandle* out) {   // 输出张量的句柄指针
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输入张量、权重张量、偏置张量从句柄转换为Tensor指针，并调用fbgemm_linear_fp16_weight_fp32_activation函数执行线性层操作
    at::Tensor* input_tensor = tensor_handle_to_tensor_pointer(input);
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);

    *out = new_tensor_handle(at::fbgemm_linear_fp16_weight_fp32_activation(
        *input_tensor, *weight_tensor, *bias_tensor));
  });
}

// 定义了一个函数aoti_torch_nonzero，用于获取输入张量中非零元素的索引
AOTITorchError aoti_torch_nonzero(
    AtenTensorHandle self,     // 输入张量的句柄
    AtenTensorHandle* out) {   // 输出张量的句柄指针
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输入张量从句柄转换为Tensor指针，并调用nonzero函数获取非零元素的索引
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *out = new_tensor_handle(at::nonzero(*self_tensor));
  });
}

// 定义了一个函数aoti_torch_repeat_interleave_Tensor，用于复制和重复输入张量的元素
AOTITorchError aoti_torch_repeat_interleave_Tensor(
    AtenTensorHandle repeats,      // 重复次数的张量句柄
    int64_t* output_size,         // 输出大小的指针
    AtenTensorHandle* out) {      // 输出张量的句柄指针
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将重复次数的张量从句柄转换为Tensor指针，并调用_ops::repeat_interleave_Tensor::call方法执行复制和重复操作
    at::Tensor* repeats_tensor = tensor_handle_to_tensor_pointer(repeats);
    *out = new_tensor_handle(at::_ops::repeat_interleave_Tensor::call(
        *repeats_tensor, pointer_to_optional<c10::SymInt>(output_size)));
  });
}
// 检查给定张量是否包含无穷大或 NaN 值，并在出错时转换为 AOTI_Torch 错误代码
AOTITorchError aoti_torch_check_inf_and_nan(
    const char* tensor_name,
    AtenTensorHandle tensor) {
  // 使用宏 AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE 包装异常处理逻辑
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将张量句柄转换为张量指针
    at::Tensor* check_tensor = tensor_handle_to_tensor_pointer(tensor);

    // 调用 assert_inf_and_nan 函数检查张量是否包含无穷大或 NaN 值
    assert_inf_and_nan(tensor_name, *check_tensor);
  });
}

// 使用 scatter_out 函数在指定维度上进行张量的散射操作，并在出错时转换为 AOTI_Torch 错误代码
AOTITorchError aoti_torch_scatter_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src) {
  // 使用宏 AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE 包装异常处理逻辑
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输出、自身、索引和源张量的句柄转换为对应的张量指针
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* index_tensor = tensor_handle_to_tensor_pointer(index);
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);

    // 调用 scatter_out 函数执行散射操作
    at::scatter_out(*out_tensor, *self_tensor, dim, *index_tensor, *src_tensor);
  });
}

// 使用 scatter_reduce_out 函数在指定维度上进行张量的散射归约操作，并在出错时转换为 AOTI_Torch 错误代码
AOTITorchError aoti_torch_scatter_reduce_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src,
    const char* reduce,
    int32_t include_self) {
  // 使用宏 AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE 包装异常处理逻辑
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将输出、自身、索引和源张量的句柄转换为对应的张量指针
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* index_tensor = tensor_handle_to_tensor_pointer(index);
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);

    // 调用 scatter_reduce_out 函数执行散射归约操作
    at::scatter_reduce_out(
        *out_tensor,
        *self_tensor,
        dim,
        *index_tensor,
        *src_tensor,
        reduce,
        (bool)include_self);
  });
}

// 使用 index_put_out 函数在给定索引处放置值到输出张量，并在出错时转换为 AOTI_Torch 错误代码
AOTITorchError aoti_torch_index_put_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    const AtenTensorHandle* indices,
    const uint32_t num_indices,
    const AtenTensorHandle values,
    bool accumulate) {
  // 使用宏 AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE 包装异常处理逻辑
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 创建一个张量指针的列表，用于存储索引张量的指针
    c10::List<std::optional<at::Tensor>> indices_;
    indices_.reserve(num_indices);
    for (size_t i = 0; i < num_indices; i++) {
      // 将每个索引张量的句柄转换为张量指针并存储在列表中
      indices_.emplace_back(
          pointer_to_optional(tensor_handle_to_tensor_pointer(indices[i])));
    }
    // 将输出、自身和值的句柄转换为对应的张量指针
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* values_tensor = tensor_handle_to_tensor_pointer(values);

    // 调用 index_put_out 函数执行索引放置操作
    at::index_put_out(
        *out_tensor, *self_tensor, indices_, *values_tensor, accumulate);
  });
}

// 使用 AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE 宏封装异常处理逻辑，并调用 at::_ops::view_as_real::call 函数创建视图作为实数
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_as_real(
    AtenTensorHandle self,
    AtenTensorHandle* ret // 返回新引用
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 调用 tensor_handle_to_tensor_pointer 函数获取自身张量的指针，并使用 view_as_real 函数创建相应视图
    *ret = new_tensor_handle(
        at::_ops::view_as_real::call(*tensor_handle_to_tensor_pointer(self)));
  });
}

// 使用 AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE 宏封装异常处理逻辑，并调用 at::view_dtype 函数创建指定数据类型的视图
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_dtype(
    AtenTensorHandle self,
    int32_t dtype,
    AtenTensorHandle* ret // 返回新引用
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 将自身张量的句柄转换为张量指针
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    // 调用 view_dtype 函数创建指定数据类型的视图，并将结果存储在 ret 中
    *ret = new_tensor_handle(at::view_dtype(*self_tensor, dtype));
  });
}
    # 使用 new_tensor_handle 函数创建一个新的张量句柄，其值为调用 at::_ops::view_dtype::call 函数的结果。
    # 这个函数期望的参数是 self_tensor 的解引用值，以及一个 static_cast 强制转换后的 c10::ScalarType 类型的值。
    *ret = new_tensor_handle(at::_ops::view_dtype::call(
        *self_tensor, static_cast<c10::ScalarType>(dtype)));
    });
}

// 关闭函数定义，与示例中的代码块相似，代表了一个函数定义的结束

AOTI_TORCH_EXPORT void aoti_torch_print_tensor_handle(
    AtenTensorHandle self,
    const char* msg) {
  // 将 AtenTensorHandle 转换为 at::Tensor 指针
  at::Tensor* t = tensor_handle_to_tensor_pointer(self);
  // 输出左括号
  std::cout << "[";
  // 如果 msg 不为空，则输出 msg
  if (msg) {
    std::cout << msg;
  }
  // 输出右括号，冒号，以及对应的 Tensor 值
  std::cout << "]:" << *t << "\n";
}

// ProxyExecutor
AOTITorchError aoti_torch_proxy_executor_call_function(
    AOTIProxyExecutorHandle proxy_executor,
    int extern_node_index,
    int num_ints,
    int64_t* flatten_int_args,
    int num_tensors,
    AtenTensorHandle* flatten_tensor_args) {
  // 将 AOTIProxyExecutorHandle 转换为 ProxyExecutor 指针
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ProxyExecutor* executor = reinterpret_cast<ProxyExecutor*>(proxy_executor);
    // 调用 ProxyExecutor 对象的函数，传递整数参数和张量参数
    executor->call_function(
        extern_node_index,
        num_ints,
        flatten_int_args,
        num_tensors,
        flatten_tensor_args);
  });
}

// 检查条件函数
void aoti_torch_check(
    bool cond,
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  // 如果条件不满足，则调用 torchCheckFail 函数
  if (C10_UNLIKELY_OR_CONST(!cond)) {
    ::c10::detail::torchCheckFail(func, file, line, msg);
  }
}

// 从内存池分配新张量
AOTITorchError aoti_torch__alloc_from_pool(
    AtenTensorHandle self,
    int64_t offset_bytes,
    int32_t dtype,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret_new_tensor) {
  // 将 AtenTensorHandle 转换为 at::Tensor 指针，分配新张量并将结果存储在 ret_new_tensor 中
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    *ret_new_tensor = new_tensor_handle(torch::inductor::_alloc_from_pool(
        *self_tensor,
        offset_bytes,
        static_cast<c10::ScalarType>(dtype),
        sizes,
        strides));
  });
}
```