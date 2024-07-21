# `.\pytorch\torch\csrc\inductor\aoti_torch\c\shim.h`

```py
// 如果未定义 AOTI_TORCH_SHIM 宏，则定义它，以防止多次包含
#ifndef AOTI_TORCH_SHIM
#define AOTI_TORCH_SHIM

// 包含标准的 C 库头文件，定义了各种标准库函数和类型
#include <stddef.h>
#include <stdint.h>

// 本头文件定义了 libtorch 中特定 ATen 功能的稳定 C API。AOTInductor 编译的 model.so 只会引用这个头文件，
// 而不会直接使用来自 aten/c10 的其它头文件，这意味着它将无法直接使用 libtorch 中的数据结构或调用函数。
//
// 我们试图解决的问题是什么？直接使用 aten/c10 API 意味着在一个没有 ABI 兼容性保证的库上使用 C++ API。
// 然而，我们希望 model.so 在 PyTorch C++ 库更新后仍然可用，这需要稳定的 ABI。通过引入 C 语言接口层，
// 我们可以最小化可能导致不兼容的接口。相应的软件堆栈可以如下所示：
//
// |--------------------------------|
// |     推理服务代码               |
// |--------------------------------|
// |           model.so             |
// |--------------|-----------------|
// |           <c shim>             |
// |          libtorch.so           |
// |--------------------------------|
//
// C API 的一般准则：
//
//  - 不使用异常，返回显式的错误代码以供调用方检查
//  - 头文件中只包含指针（例如 AtenTensorHandle）、整数和浮点数
//
// 如果你想要修改这个头文件，你必须保持 ABI 兼容性。通常，这意味着你需要为想要添加新函数参数的函数添加一个 _v2 版本，
// 并维护旧版本和新版本的 API，直到所有旧的 model.so 都不再使用为止。

// 根据不同的编译器定义 AOTI_TORCH_EXPORT 宏，用于符号的导出和可见性
#ifdef __GNUC__
#define AOTI_TORCH_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
// PyTorch2 目前不支持 Windows。导出这些 API 可能会导致在链接时与 libtorch 包含在 DLL 和依赖该 DLL 的二进制文件
// 发生符号冲突。作为临时解决方案，我们不导出这些符号。长期来看，当 Windows 支持时，需要解决这个问题。
// #define AOTI_TORCH_EXPORT __declspec(dllexport)
#define AOTI_TORCH_EXPORT
#else // !_WIN32
#define AOTI_TORCH_EXPORT
#endif // _WIN32
#endif // __GNUC__

// 如果是 C++ 环境，将下面的函数声明放在 extern "C" 块内，以确保符号按照 C 的方式进行名称修饰
#ifdef __cplusplus
extern "C" {
#endif

// AtenTensorHandle 表示可以在 model.so 和 libtorch.so 之间传递的 Tensor 的抽象概念。
// 结构体本身的内容是私有的；model.so 不允许直接访问任何字段，必须通过本 ABI 中定义的函数进行访问。
// 在底层，它表示为 at::Tensor*，但我们保留更改它的权利（事实上，我们可能应该至少将其更改为 at::TensorImpl*）。
//
// 一个 AtenTensorHandle 可能是拥有的（请查阅 API 参考了解确切的所有权/借用语义）。如果在 model.so 中有一个拥有的
// AtenTensorHandle，你有责任在不需要时调用 aoti_torch_delete_tensor_object 进行删除。
// 定义了几个结构体和类型别名，用于处理 PyTorch 张量相关的操作

// 不透明类型 AtenTensorOpaque 的别名，用于表示 PyTorch 张量的句柄
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

// 不透明类型 AtenGeneratorOpaque 的别名，用于表示 PyTorch 生成器的句柄
struct AtenGeneratorOpaque;
using AtenGeneratorHandle = AtenGeneratorOpaque*;

// 不透明类型 AOTIProxyExecutorOpaque 的别名，用于表示 AOTI 代理执行器的句柄
struct AOTIProxyExecutorOpaque;
using AOTIProxyExecutorHandle = AOTIProxyExecutorOpaque*;

// 定义了表示 Torch 错误码的别名 AOTITorchError，成功和失败的常量
using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

// 下面是一系列用于获取运行时常量的函数声明，这些常量可以传递给其他 aoti_* 函数使用
// 这些函数将设备类型和数据类型隐藏在函数接口之后，避免它们成为 ABI 协议的一部分
// （实际上，aten/c10 在不重新编号这些常量上做得相当好，因此如果性能原因允许，可以考虑将其纳入 ABI）

AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_cpu();
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_cuda();

AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e5m2();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e4m3fn();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e5m2fnuz();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e4m3fnuz();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_bfloat16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint8();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int8();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_bool();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_complex32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_complex64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_complex128();

// 用于表示张量布局的常量的获取函数声明
AOTI_TORCH_EXPORT int32_t aoti_torch_layout_strided();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout__mkldnn();

// 用于将单元素张量转换为标量值的函数声明
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_float32(AtenTensorHandle tensor, float* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_float64(AtenTensorHandle tensor, double* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint8(AtenTensorHandle tensor, uint8_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint16(AtenTensorHandle tensor, uint16_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint32(AtenTensorHandle tensor, uint32_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint64(AtenTensorHandle tensor, uint64_t* ret_value);
// 定义导出的函数aoti_torch_item_int8，用于从给定的AtenTensorHandle中获取int8_t类型的值，存储在ret_value中，并返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int8(AtenTensorHandle tensor, int8_t* ret_value);

// 定义导出的函数aoti_torch_item_int16，用于从给定的AtenTensorHandle中获取int16_t类型的值，存储在ret_value中，并返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int16(AtenTensorHandle tensor, int16_t* ret_value);

// 定义导出的函数aoti_torch_item_int32，用于从给定的AtenTensorHandle中获取int32_t类型的值，存储在ret_value中，并返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int32(AtenTensorHandle tensor, int32_t* ret_value);

// 定义导出的函数aoti_torch_item_int64，用于从给定的AtenTensorHandle中获取int64_t类型的值，存储在ret_value中，并返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int64(AtenTensorHandle tensor, int64_t* ret_value);

// 定义导出的函数aoti_torch_item_bool，用于从给定的AtenTensorHandle中获取bool类型的值，存储在ret_value中，并返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_bool(AtenTensorHandle tensor, bool* ret_value);

// 定义导出的函数aoti_torch_scalar_to_tensor_float32，将单精度浮点值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_float32(float value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_float64，将双精度浮点值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_float64(double value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_uint8，将uint8_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_uint8(uint8_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_uint16，将uint16_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_uint16(uint16_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_uint32，将uint32_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_uint32(uint32_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_uint64，将uint64_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_uint64(uint64_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_int8，将int8_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_int8(int8_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_int16，将int16_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_int16(int16_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_int32，将int32_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_int32(int32_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_int64，将int64_t值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_int64(int64_t value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_scalar_to_tensor_bool，将bool值转换为单元素张量，并存储在ret_new_tensor中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_bool(bool value, AtenTensorHandle* ret_new_tensor);

// 定义导出的函数aoti_torch_grad_mode_is_enabled，返回当前梯度模式是否启用的布尔值
AOTI_TORCH_EXPORT bool aoti_torch_grad_mode_is_enabled();

// 定义导出的函数aoti_torch_grad_mode_set_enabled，设置当前梯度模式是否启用
AOTI_TORCH_EXPORT void aoti_torch_grad_mode_set_enabled(bool enabled);

// 定义导出的函数aoti_torch_delete_tensor_object，释放给定的AtenTensorHandle对象
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_tensor_object(AtenTensorHandle tensor);

// 定义导出的函数aoti_torch_get_data_ptr，获取给定AtenTensorHandle对象的底层存储数据指针，并存储在ret_data_ptr中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr // 返回借用引用
);

// 定义导出的函数aoti_torch_get_storage_size，获取给定AtenTensorHandle对象底层存储的字节大小，并存储在ret_size中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_storage_size(AtenTensorHandle tensor, int64_t* ret_size);

// 定义导出的函数aoti_torch_get_dim，获取给定AtenTensorHandle对象的维度数量，并存储在ret_dim中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_dim(AtenTensorHandle tensor, int64_t* ret_dim);

// 定义导出的函数aoti_torch_get_numel，获取给定AtenTensorHandle对象的元素总数，并存储在ret_numel中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_numel(AtenTensorHandle tensor, int64_t* ret_numel);

// 定义导出的函数aoti_torch_get_sizes，获取给定AtenTensorHandle对象的维度大小数组，并存储在ret_sizes中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_sizes // 返回借用引用
);

// 定义导出的函数aoti_torch_get_size，获取给定AtenTensorHandle对象在指定维度d上的大小，并存储在ret_size中，返回AOTITorchError错误码
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_size(AtenTensorHandle tensor, int64_t d, int64_t* ret_size
// 获取张量的步幅信息
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides // 返回借用引用
);

// 获取张量在指定维度上的步幅
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_stride(AtenTensorHandle tensor, int64_t d, int64_t* ret_stride);

// 获取张量的数据类型
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_dtype(AtenTensorHandle tensor, int32_t* ret_dtype);

// 获取张量的设备类型
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_device_type(AtenTensorHandle tensor, int32_t* ret_device_type);

// 获取张量的设备索引
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_device_index(AtenTensorHandle tensor, int32_t* ret_device_index);

// 获取张量的存储偏移
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset);

// 从内存池中分配新的张量对象
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__alloc_from_pool(
    AtenTensorHandle self,
    int64_t offset_bytes,
    int32_t dtype,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret_new_tensor);

// 重新解释张量的形状和步幅，创建一个新的张量对象
// 返回新的张量对象的指针通过 *out 返回，调用者需负责使用 RAIIAtenTensorHandle
// 封装张量指针，在作用域外调用 aoti_torch_delete_tensor_object 进行清理。
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AtenTensorHandle* ret_new_tensor // 返回新的引用
);

// 创建一个具有指定形状、步幅、数据类型和设备的新张量对象
// 返回新的张量对象的指针通过 *ret_new_tensor 返回
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor // 返回新的引用
);

// 从 blob 数据创建一个新的张量对象
// 返回新的张量对象的指针通过 *ret 返回
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret // 返回新的引用
);

// 从 blob 数据创建一个新的张量对象（版本 2），支持布局和元数据参数
// 返回新的张量对象的指针通过 *ret 返回
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret, // 返回新的引用
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size
);

// 计算嵌入（embedding）的加权和
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__embedding_bag(
    AtenTensorHandle weight,
    AtenTensorHandle indices,
    AtenTensorHandle offsets,
    int32_t scale_grad_by_freq,
    int32_t mode,
    int32_t sparse,
    ...
);  // 后续未完整展示，但是函数名和参数可以提供一个大致的理解
    AtenTensorHandle per_sample_weights, // 每个样本的权重，可选参数
    int32_t include_last_offset,         // 是否包括最后一个偏移量
    int32_t padding_idx,                 // 填充索引
    AtenTensorHandle* ret0,              // 返回的新引用，第一个返回值
    AtenTensorHandle* ret1,              // 返回的新引用，第二个返回值
    AtenTensorHandle* ret2,              // 返回的新引用，第三个返回值
    AtenTensorHandle* ret3               // 返回的新引用，第四个返回值
// 导出的函数声明，用于执行复杂的 FFT 算法
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__fft_c2c(
    AtenTensorHandle self,             // 输入张量的句柄
    const int64_t* dim_ptr,           // 维度指针，指向维度数组的首地址
    int64_t dim_size,                 // 维度数组的大小
    int64_t normalization,            // 归一化参数
    int32_t forward,                  // 前向/反向标志
    AtenTensorHandle* ret             // 返回结果的新引用
);

// 此版本已弃用，将在以后移除
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_dot_product_flash_attention(
    AtenTensorHandle query,           // 查询张量的句柄
    AtenTensorHandle key,             // 键张量的句柄
    AtenTensorHandle value,           // 值张量的句柄
    double dropout_p,                 // dropout 概率
    bool is_causal,                   // 是否是因果注意力
    bool return_debug_mask,           // 是否返回调试掩码
    double scale,                     // 缩放因子
    AtenTensorHandle* ret0,           // 返回结果的新引用
    AtenTensorHandle* ret1,           // 返回结果的新引用
    AtenTensorHandle* ret2,           // 返回结果的新引用
    AtenTensorHandle* ret3,           // 返回结果的新引用
    int64_t* ret4,                    // 返回结果的新引用
    int64_t* ret5,                    // 返回结果的新引用
    AtenTensorHandle* ret6,           // 返回结果的新引用
    AtenTensorHandle* ret7,           // 返回结果的新引用
    AtenTensorHandle* ret8            // 返回结果的新引用
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_dot_product_flash_attention_v2(
    AtenTensorHandle query,           // 查询张量的句柄
    AtenTensorHandle key,             // 键张量的句柄
    AtenTensorHandle value,           // 值张量的句柄
    double dropout_p,                 // dropout 概率
    int is_causal,                    // 是否是因果注意力
    int return_debug_mask,            // 是否返回调试掩码
    double* scale,                    // 可选参数，缩放因子
    AtenTensorHandle* ret0,           // 返回结果的新引用
    AtenTensorHandle* ret1,           // 返回结果的新引用
    AtenTensorHandle* ret2,           // 返回结果的新引用
    AtenTensorHandle* ret3,           // 返回结果的新引用
    int64_t* ret4,                    // 返回结果的新引用
    int64_t* ret5,                    // 返回结果的新引用
    AtenTensorHandle* ret6,           // 返回结果的新引用
    AtenTensorHandle* ret7,           // 返回结果的新引用
    AtenTensorHandle* ret8            // 返回结果的新引用
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_dot_product_efficient_attention(
    AtenTensorHandle query,           // 查询张量的句柄
    AtenTensorHandle key,             // 键张量的句柄
    AtenTensorHandle value,           // 值张量的句柄
    AtenTensorHandle attn_bias,       // 可选参数，注意力偏置
    int compute_log_sumexp,           // 计算对数求和指数
    double dropout_p,                 // dropout 概率
    int is_causal,                    // 是否是因果注意力
    double* scale,                    // 可选参数，缩放因子
    AtenTensorHandle* ret0,           // 返回结果的新引用
    AtenTensorHandle* ret1,           // 返回结果的新引用
    AtenTensorHandle* ret2,           // 返回结果的新引用
    AtenTensorHandle* ret3            // 返回结果的新引用
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_mm(
    AtenTensorHandle self,            // 输入张量的句柄
    AtenTensorHandle mat2,            // 第二个矩阵的句柄
    AtenTensorHandle bias,            // 可选参数，偏置
    int32_t* out_dtype,               // 输出数据类型指针
    AtenTensorHandle scale_a,         // 缩放因子 A 的张量句柄
    AtenTensorHandle scale_b,         // 缩放因子 B 的张量句柄
    AtenTensorHandle scale_result,    // 缩放结果的张量句柄
    int8_t use_fast_accum,            // 使用快速累加标志
    AtenTensorHandle* ret0,           // 返回结果的新引用
    AtenTensorHandle* ret1            // 返回结果的新引用
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_convolution(
    AtenTensorHandle input,           // 输入张量的句柄
    AtenTensorHandle weight,          // 权重张量的句柄
    AtenTensorHandle bias,            // 可选参数，偏置
    const int64_t* stride_ptr,        // 步幅指针
    int64_t stride_size,              // 步幅数组的大小
    const int64_t* padding_ptr,       // 填充指针
    int64_t padding_size,             // 填充数组的大小
    const int64_t* dilation_ptr,      // 膨胀指针
    int64_t dilation_size,            // 膨胀数组的大小
    int transposed,                   // 是否是转置卷积
    const int64_t* output_padding_ptr,// 输出填充指针
    int64_t output_padding_size,      // 输出填充数组的大小
    int64_t groups,                   // 分组数
    AtenTensorHandle* ret // 定义一个指向 AtenTensorHandle 类型的指针变量 ret
// This function declaration indicates that a new uninitialized tensor object will be created,
// and its pointer will be returned through the *ret parameter.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_new_uninitialized_tensor(AtenTensorHandle* ret);

// WARNING: Deprecated function. Use aoti_torch_copy_ instead.
// This function copies the contents of tensor src to tensor dst.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_tensor_copy_(AtenTensorHandle src, AtenTensorHandle dst);

// Make tensor dst an alias for tensor src. Both tensors need to be managed separately.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_assign_tensors(AtenTensorHandle src, AtenTensorHandle dst);

// Create a shallow copy of tensor src and assign it to *ret_dst.
// This function is similar to aoti_torch_assign_tensors but manages memory allocation.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_assign_tensors_out(AtenTensorHandle src, AtenTensorHandle* ret_dst);

// Create a new tensor object by cloning tensor self. The new tensor pointer is returned through *ret.
// The caller must manage the returned tensor pointer with RAIIAtenTensorHandle.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_clone(AtenTensorHandle self, AtenTensorHandle* ret);

// Perform matrix multiplication of mat1 and mat2, scaled by alpha and beta coefficients,
// and store the result in tensor out.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha);

// Perform batch matrix multiplication of self with mat2,
// and store the result in tensor out.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

// Copy the contents of tensor src to tensor self. Supports non-blocking operations.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_copy_(
    AtenTensorHandle self,
    AtenTensorHandle src,
    int32_t non_blocking);

// Perform matrix multiplication of self with mat2,
// and store the result in tensor out.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

// Deprecated function related to ao_quantization.
// Packs the matrix in tensor weight for efficient GEMM computation in FP16 format.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_wrapped_fbgemm_pack_gemm_matrix_fp16(
    AtenTensorHandle weight,
    AtenTensorHandle* out);

// Deprecated function related to ao_quantization.
// Computes linear transformation using input, weight, and bias tensors in FP16 format.
// Stores the result in tensor out.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_wrapped_fbgemm_linear_fp16_weight(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias,
    int64_t out_channel,
    AtenTensorHandle* out);

// Find nonzero elements in tensor self and store the result in tensor out.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_nonzero(AtenTensorHandle self, AtenTensorHandle* out);

// Repeat tensor repeats along specified dimensions to generate output tensor out.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_repeat_interleave_Tensor(
    AtenTensorHandle repeats,
    int64_t* output_size,
    AtenTensorHandle* out);

// Check for infinite and NaN values in tensor tensor.
// Prints a warning message with tensor_name if any such values are found.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_check_inf_and_nan(const char* tensor_name, AtenTensorHandle tensor);
// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于执行带有输出参数的 scatter 操作
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scatter_out(
    AtenTensorHandle out,          // 输出张量句柄
    AtenTensorHandle self,         // 源张量句柄
    int64_t dim,                   // 维度参数
    AtenTensorHandle index,        // 索引张量句柄
    AtenTensorHandle src);         // 数据源张量句柄

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于执行带有输出参数的 scatter_reduce 操作
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scatter_reduce_out(
    AtenTensorHandle out,          // 输出张量句柄
    AtenTensorHandle self,         // 源张量句柄
    int64_t dim,                   // 维度参数
    AtenTensorHandle index,        // 索引张量句柄
    AtenTensorHandle src,          // 数据源张量句柄
    const char* reduce,            // 减少操作类型字符串
    int32_t include_self);         // 是否包含自身的标志

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于执行带有输出参数的 index_put 操作
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_index_put_out(
    AtenTensorHandle out,                     // 输出张量句柄
    AtenTensorHandle self,                    // 源张量句柄
    const AtenTensorHandle* indices,          // 索引张量数组的指针
    const uint32_t num_indices,               // 索引张量的数量
    const AtenTensorHandle values,            // 值张量句柄
    bool accumulate);                         // 是否累加标志

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于执行视图转换为实部操作
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_as_real(
    AtenTensorHandle self,                    // 张量句柄
    AtenTensorHandle* ret // returns new reference  // 返回的新张量句柄的指针
);

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于执行视图转换为指定数据类型操作
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_dtype(
    AtenTensorHandle self,                    // 张量句柄
    int32_t dtype,                            // 目标数据类型
    AtenTensorHandle* ret // returns new reference  // 返回的新张量句柄的指针
);

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于打印张量句柄的信息
AOTI_TORCH_EXPORT void aoti_torch_print_tensor_handle(
    AtenTensorHandle self,                    // 张量句柄
    const char* msg);                         // 要打印的消息字符串

#ifdef USE_CUDA

// 定义 CUDA 保护结构体和句柄类型
struct CUDAGuardOpaque;
using CUDAGuardHandle = CUDAGuardOpaque*;

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于创建 CUDA 保护句柄
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,                      // 设备索引
    CUDAGuardHandle* ret_guard // returns new reference  // 返回的新 CUDA 保护句柄的指针
);

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于删除 CUDA 保护句柄
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_cuda_guard(CUDAGuardHandle guard);  // 要删除的 CUDA 保护句柄

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于设置 CUDA 保护句柄的设备索引
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cuda_guard_set_index(CUDAGuardHandle guard, int32_t device_index);  // 要设置的设备索引

// 定义 CUDA 流保护结构体和句柄类型
struct CUDAStreamGuardOpaque;
using CUDAStreamGuardHandle = CUDAStreamGuardOpaque*;

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于创建 CUDA 流保护句柄
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,                              // CUDA 流指针
    int32_t device_index,                      // 设备索引
    CUDAStreamGuardHandle* ret_guard // returns new reference  // 返回的新 CUDA 流保护句柄的指针
);

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于删除 CUDA 流保护句柄
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_cuda_stream_guard(CUDAStreamGuardHandle guard);  // 要删除的 CUDA 流保护句柄

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于获取当前 CUDA 流
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);  // 返回的当前 CUDA 流指针的指针

#endif

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于通过代理执行器调用函数
// 查看 ir.py 中的 `ProxyExecutor Design Note` 以获取更多细节
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_proxy_executor_call_function(
    AOTIProxyExecutorHandle proxy_executor,    // 代理执行器句柄
    int extern_node_index,                    // 外部节点索引
    int num_ints,                             // 整数参数数量
    int64_t* flatten_int_args,                // 扁平化整数参数数组
    int num_tensors,                          // 张量参数数量
    AtenTensorHandle* flatten_tensor_args);   // 扁平化张量参数数组

// 声明 AOTI_TORCH_EXPORT 宏修饰的函数，用于检查条件并报错
#ifdef STRIP_ERROR_MESSAGES
#define AOTI_TORCH_CHECK(cond, ...)              \
  if (!(cond)) {                                 \
    aoti_torch_check(                            \
        false,                                   \
        __func__,                                \
        __FILE__,                                \
        static_cast<uint32_t>(__LINE__),         \
        TORCH_CHECK_MSG(cond, "", __VA_ARGS__)); \
  }
#else
#define AOTI_TORCH_CHECK(cond, ...)                \
  if (!(cond)) {                                   \
    aoti_torch_check(                              \
        false,                                     \
        __func__,                                  \
        __FILE__,                                  \
        static_cast<uint32_t>(__LINE__),           \
        TORCH_CHECK_MSG(cond, "", __VA_ARGS__));   \
  }
#endif
    aoti_torch_check(                              \
        false,                                     \  # 调用 aoti_torch_check 函数，传入以下参数：
        __func__,                                  \  # 当前函数名
        __FILE__,                                  \  # 当前文件名
        static_cast<uint32_t>(__LINE__),           \  # 当前代码行号，转换为 uint32_t 类型
        TORCH_CHECK_MSG(cond, "", ##__VA_ARGS__)); \  # 使用 TORCH_CHECK_MSG 宏检查条件，可能包含额外的可变参数
  }
#ifdef __cplusplus
} // extern "C"

// 删除模板函数，禁止使用
template <typename T>
int32_t aoti_torch_dtype() = delete;

// 定义模板特化，返回特定类型的 Torch 数据类型码
#define DEFINE_DTYPE_SPECIALIZATION(ctype, typename) \
  template <>                                        \
  inline int32_t aoti_torch_dtype<ctype>() {         \
    return aoti_torch_dtype_##typename();            \
  }

// Torch 的 C10 命名空间，声明两种数据类型结构体
namespace c10 {
struct BFloat16;
struct Half;
} // namespace c10

// 定义特化模板函数，返回 BFloat16 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(c10::BFloat16, bfloat16)
// 定义特化模板函数，返回 Half 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(c10::Half, float16)
// 定义特化模板函数，返回 float 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(float, float32)
// 定义特化模板函数，返回 double 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(double, float64)
// 定义特化模板函数，返回 uint8_t 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(uint8_t, uint8)
// 定义特化模板函数，返回 int8_t 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(int8_t, int8)
// 定义特化模板函数，返回 int16_t 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(int16_t, int16)
// 定义特化模板函数，返回 int32_t 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(int32_t, int32)
// 定义特化模板函数，返回 int64_t 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(int64_t, int64)
// 定义特化模板函数，返回 bool 的 Torch 数据类型码
DEFINE_DTYPE_SPECIALIZATION(bool, bool)

#endif

#endif // AOTI_TORCH_SHIM
```