# `.\pytorch\aten\src\ATen\native\EmbeddingBag.h`

```py
// 引入 ATen 库中的 Tensor 类和配置信息
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
// 引入标准整数类型
#include <cstdint>

#ifdef USE_FBGEMM
// 如果定义了 USE_FBGEMM，则引入 FbgemmEmbedding.h 头文件
#include <fbgemm/FbgemmEmbedding.h>
#endif

// 定义 at::native 命名空间
namespace at::native {

// 检查参数的有效性
void check_arguments(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset);

// 创建并填充包的大小输出 Tensor
void make_bag_size_out(
    Tensor& bag_size_out,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad);

// 创建并填充最大索引输出 Tensor
void make_max_indices_out(
    Tensor& max_indices_out,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset);

// 创建并填充 offset 到 bag 映射输出 Tensor
void make_offset2bag_out(
    Tensor& offset2bag,
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx = -1);

#ifdef USE_FBGEMM

// 嵌套结构模板，用于存储回调函数和块大小
template<bool has_weight, typename TIndex, typename TData>
struct _CallbackAndBlockSize {
    // 回调函数类型定义，根据 FbgemmEmbedding.h 中的模板参数生成
    using TCallback = typename fbgemm::EmbeddingSpMDMKernelSignature<TData, TIndex, TIndex, TData>::Type;

    // 块大小，默认为 -1
    int64_t blockSize = -1;
    // 回调函数指针，默认为空指针
    TCallback callback = nullptr;

    // 根据给定的块大小生成回调函数
    static TCallback generateCallback(int64_t block_size) {
        return fbgemm::GenerateEmbeddingSpMDM<TData, TIndex, TIndex, TData>(
                block_size,
                has_weight,
                /* normalize_by_lengths */false,
                /* prefetch */16,
                /* is_weight_positional */false,
                /* use_offsets */true);
    }

    // 默认构造函数
    _CallbackAndBlockSize() = default;

    // 显式构造函数，根据可选的块大小生成回调函数
    explicit _CallbackAndBlockSize(std::optional<int64_t> maybe_block_size)
      : blockSize(maybe_block_size.value_or(-1))
      , callback(maybe_block_size.has_value() ? generateCallback(maybe_block_size.value()) : nullptr)
    {}
};

// 嵌套结构模板，用于存储嵌入包内核缓存
template<typename... StorageMixins>
struct _EmbeddingBagKernelCacheImpl : private StorageMixins... {

    // 默认构造函数
    _EmbeddingBagKernelCacheImpl() = default;

    // 显式构造函数，使用每个 StorageMixins 存储对应的内核和块大小
    explicit _EmbeddingBagKernelCacheImpl(std::optional<int64_t> maybe_block_size)
      : StorageMixins(maybe_block_size)...
    {}

    // 此方法是线程安全的（调用点可以从不同线程调用）
    template<bool has_weight, typename TIndex, typename TData>
    typename _CallbackAndBlockSize<has_weight, TIndex, TData>::TCallback
    // 获取回调函数，该函数返回一个类型为 int64_t 的块大小常量
    // 如果缓存中不包含与传入块大小对应的内核
    // （即与相应的 mixin 中存储的内核不同）
    // 则重新生成内核（不将其写入缓存以避免锁定）
    if (block_size != _CallbackAndBlockSize<has_weight, TIndex, TData>::blockSize) {
        // 调用生成回调函数的方法，传入当前的块大小
        return _CallbackAndBlockSize<has_weight, TIndex, TData>::generateCallback(block_size);
    }
    // 否则从相应的 mixin 中检索缓存的内核
    return _CallbackAndBlockSize<has_weight, TIndex, TData>::callback;
}
// 结构体 _EmbeddingBagKernelCache 的定义
struct _EmbeddingBagKernelCache {
    // 构造函数，接受一个可选的 int64_t 类型参数，用于初始化
    explicit _EmbeddingBagKernelCache(std::optional<int64_t> /* maybe_block_size */) {}
};

#ifdef USE_FBGEMM
// 使用 FBGEMM，定义 _EmbeddingBagKernelCache 类型别名为 _EmbeddingBagKernelCacheImpl
// 该类型实例化了多个 _CallbackAndBlockSize 类模板，针对不同的类型组合
using _EmbeddingBagKernelCache = _EmbeddingBagKernelCacheImpl<
    _CallbackAndBlockSize<true, int32_t, float>,
    _CallbackAndBlockSize<false, int32_t, float>,
    _CallbackAndBlockSize<true, int64_t, float>,
    _CallbackAndBlockSize<false, int64_t, float>,
    _CallbackAndBlockSize<true, int32_t, unsigned short>,
    _CallbackAndBlockSize<false, int32_t, unsigned short>,
    _CallbackAndBlockSize<true, int64_t, unsigned short>,
    _CallbackAndBlockSize<false, int64_t, unsigned short>>;
#else
// 如果未使用 FBGEMM，则定义一个空的 _EmbeddingBagKernelCache 结构体
struct _EmbeddingBagKernelCache {
    explicit _EmbeddingBagKernelCache(std::optional<int64_t> /* maybe_block_size */) {}
};
#endif

// 声明 _embedding_bag_cpu_impl_out 函数，用于实现 CPU 上的 EmbeddingBag 操作
void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
    Tensor& bag_size, Tensor* max_indices,
    const Tensor &weight, const Tensor &indices,
    const Tensor &offsets, const int64_t mode = 0,
    const std::optional<Tensor>& per_sample_weights = c10::nullopt,
    bool include_last_offset = false,
    int64_t padding_idx = -1,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache = nullptr);

// 声明 _embedding_bag_cpu_out 函数，用于实现 CPU 上的 EmbeddingBag 操作
void _embedding_bag_cpu_out(
    at::Tensor& output,
    at::Tensor& offset2bag,
    at::Tensor& bag_size,
    at::Tensor* p_max_indices,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    const bool sparse,
    const std::optional<at::Tensor>& per_sample_weights,
    const bool include_last_offset,
    const std::optional<int64_t>& padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache = nullptr);

} // namespace at::native
```