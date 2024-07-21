# `.\pytorch\aten\src\ATen\functorch\BatchRulesUnaryOps.cpp`

```py
// 包含标准库和Functorch的头文件
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>

// 定义Functorch命名空间
namespace at::functorch {

// 匿名命名空间，定义了一些内部实用函数和规则
namespace {

// 克隆操作的批处理规则，返回克隆后的张量和可选的批处理维度
std::tuple<Tensor,optional<int64_t>>
clone_batch_rule(
    const Tensor& self,                              // 输入张量
    optional<int64_t> self_bdim,                     // 可选的自身批处理维度
    optional<MemoryFormat> memory_format) {           // 可选的内存格式

  // 对内存格式的支持有些棘手，因为vmap允许移动批处理维度，而某些内存格式依赖于张量的秩
  // 另一个特殊情况是：
  // - 具有MemoryFormat::ChannelsLast的张量必须有4个维度。我们允许用户将具有3个逻辑维度和1个批处理维度的张量克隆为ChannelsLast张量吗？
  // - 对于具有3个逻辑维度和N>1批处理维度的张量呢？
  TORCH_CHECK(!memory_format.has_value() || memory_format == MemoryFormat::Preserve
      || memory_format == MemoryFormat::Contiguous,
      "NYI: Tensor.clone(memory_format) inside vmap is only supported with ",
      "memory_format torch.preserve_format or torch.contiguous_format (got ",
      *memory_format, ")");

  // 如果内存格式为Contiguous
  if (memory_format == MemoryFormat::Contiguous) {
    // 当批处理维度不在张量的前面时存在歧义
    // >>> x = torch.randn(3, B0, 5)
    // >>> y = vmap(lambda x: x.clone(torch.contiguous_format), in_dims=1, out_dims=0)(x)
    // >>> y[0].is_contiguous()
    // ???
    // 我们应该使整个张量连续，还是应该使非批处理维度连续？我们选择了后者，因为在哲学上vmap隐藏了批处理维度，并在每个样本级别上操作。
    auto self_ = moveBatchDimToFront(self, self_bdim);  // 将批处理维度移动到张量的前面
    auto result = at::clone(self_, memory_format);      // 使用指定的内存格式克隆张量
    return std::make_tuple(result, 0);                  // 返回克隆后的结果和批处理维度为0
  }

  // 对于其他情况，使用默认内存格式进行克隆
  TORCH_INTERNAL_ASSERT(!memory_format.has_value() || memory_format == MemoryFormat::Preserve);
  auto result = at::clone(self, memory_format);         // 使用指定的内存格式克隆张量
  return std::make_tuple(result, self_bdim);            // 返回克隆后的结果和当前批处理维度
}

// 视为复数的批处理规则，返回视为复数后的张量和可选的批处理维度
std::tuple<Tensor,optional<int64_t>>
view_as_complex_batch_rule(const Tensor& self, optional<int64_t> self_bdim) {

  // 防止用户传递批量大小为2的标量张量批次
  TORCH_CHECK(self.sym_sizes().size() > 1, "Input tensor must have one or more dimensions");

  auto self_ = moveBatchDimToFront(self, self_bdim);    // 将批处理维度移动到张量的前面
  auto result = at::view_as_complex(self_);             // 将张量视为复数形式
  return std::make_tuple(result, 0);                    // 返回视为复数后的结果和批处理维度为0
}

}

// 在aten命名空间中注册Functorch批处理函数
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {

// 定义宏以支持一元点对点操作的批处理规则
#define UNARY_POINTWISE_ALL2(op, overload) \
  POINTWISE_BOXED2(op ## _, overload); \
  VMAP_SUPPORT2(op, overload, BASIC_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));

// 取消定义宏以支持一元点对点操作的批处理规则
#undef UNARY_POINTWISE
#undef UNARY_POINTWISE_ALL

}

// 取消定义宏INVOKE
#undef INVOKE

} // namespace at::functorch
```