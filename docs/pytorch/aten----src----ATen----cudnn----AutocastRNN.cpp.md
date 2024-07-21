# `.\pytorch\aten\src\ATen\cudnn\AutocastRNN.cpp`

```
// 引入 ATen 库，包括 ATen 头文件和自动混合精度模式头文件
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
// 引入 Torch 库
#include <torch/library.h>

// 通过 CMake 定义的 AT_CUDNN_ENABLED()，检查是否启用了 cuDNN
#include <ATen/cuda/CUDAConfig.h>

#if AT_CUDNN_ENABLED()
// 引入 cuDNN 相关的 RNN 工具函数
#include <ATen/native/cudnn/RNNUtils.h>
#endif

// AT 自定义命名空间
namespace at {
namespace autocast {

/********************************************************************************
Autocast wrapper for CuDNN RNNs (the weight reflattening needs special attention)
********************************************************************************/

// 为 "_cudnn_rnn(...)" schema 注册的函数
// _cudnn_rnn 在 autograd 中可见（test_autocast_cudnn_rnn 在 test_cuda.py 中有对应的测试）
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>
_cudnn_rnn_cast_reflatten(const Tensor & input,
                          TensorList weight,
                          int64_t weight_stride0,
                          const std::optional<Tensor>& weight_buf_opt,
                          const Tensor& hx,
                          const std::optional<Tensor>& cx,
                          int64_t mode,
                          int64_t hidden_size,
                          int64_t proj_size,
                          int64_t num_layers,
                          bool batch_first,
                          double dropout,
                          bool train,
                          bool bidirectional,
                          IntArrayRef batch_sizes,
                          const std::optional<Tensor>& dropout_state) {
#if AT_CUDNN_ENABLED()
  // 排除 Autocast 的 DispatchKey
  c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);

  // 检查权重列表中的所有张量的标量类型是否匹配
  for (const auto& t : weight) {
    TORCH_CHECK(weight[0].scalar_type() == t.scalar_type(), "Weight scalar types do not match.");
  }

  // weight_stride0 表示每层和方向的权重张量数，由 model.parameters() 观察得出
  // 如果启用了偏置，每层和方向有 4 个张量（ih 和 hh 权重，ih 和 hh 偏置）
  // 如果未启用偏置，每层和方向有 2 个张量（ih 和 hh 权重）
  // 这种组织方式适用于所有的 RNN 类型（RNN、GRU 和 LSTM）。如果是带投影的 LSTM，还会额外添加 hr 权重。
  if (proj_size > 0) {
    TORCH_INTERNAL_ASSERT((weight_stride0 == 3) || (weight_stride0 == 5),
                          "weight_stride0 must be 3 (if no bias) or 5 (if bias) for LSTM with projections.  Received ",
                          weight_stride0);
  } else {


继续下一部分的注释。
    // 确保 weight_stride0 符合预期值，否则抛出断言错误信息
    TORCH_INTERNAL_ASSERT((weight_stride0 == 2) || (weight_stride0 == 4),
                          "weight_stride0 must be 2 (if no bias) or 4 (if bias).  Received ",
                          weight_stride0);
  }

  // 声明 Tensor 对象 weight_buf 和 redispatch_weight_buf，以及 Tensor 向量 redispatch_weight
  Tensor weight_buf, redispatch_weight_buf;
  std::vector<Tensor> redispatch_weight;
  
  // 此处与 native/cudnn/RNN.cpp:_cudnn_impl 之间有隐含的协议，
  // _cudnn_impl 在调用时期望 weight_buf_opt 包含一个定义好的张量，
  // 表示这个张量是以其传入的数据类型存储权重的有效扁平存储。
  if (weight_buf_opt.has_value()) {
    // 如果 weight_buf_opt 有值，则将其解引用赋值给 weight_buf
    weight_buf = *weight_buf_opt;
  }
  
  // 判断是否需要进行类型转换和扁平化操作
  bool needs_cast_and_flatten = (weight_buf.defined() ?
                                 // 如果 weight_buf 有效，则根据条件判断是否需要转换成 FP16 类型
                                 is_eligible(weight_buf) && (weight_buf.scalar_type() != at::kHalf) :
                                 // 如果 weight_buf 无效，则检查是否需要创建新的 weight_buf，并转换成 FP16 类型
                                 is_eligible(weight[0]) && (weight[0].scalar_type() != at::kHalf));
  
  if (needs_cast_and_flatten) {
    // 将权重张量转换成 FP16 类型，并确保所有层的权重都是大平坦缓冲区中预期的位置和布局的视图，
    // 这是 cudnn 期望的操作。此过程对 autograd 可见。
    bool include_bias = true;
    // 根据条件确定是否包含偏置项
    if (weight_stride0 == 2 || (weight_stride0 == 3 && proj_size > 0)) {
      include_bias = false;
    }
    
    // 调用函数将权重复制到扁平缓冲区视图中
    std::tie(redispatch_weight_buf, redispatch_weight) =
        at::native::cudnn_rnn::copy_weights_to_flat_buf_views(
            weight,
            weight_stride0,
            input.size(-1),
            mode,
            hidden_size,
            proj_size,
            num_layers,
            batch_first,
            bidirectional,
            /*flat_buf_datatype=*/at::native::getCudnnDataTypeFromScalarType(at::kHalf), // 可以直接硬编码为 CUDNN_DATA_HALF
            /*flat_buf_options=*/weight[0].options().dtype(at::kHalf),
            /*set_orig_weights_to_flat_buf=*/false,
            /*allow_type_change=*/true,
            /*include_bias=*/include_bias);
  }
  
  // 调用底层的 cudnn_rnn 函数，传递相应参数进行计算
  return at::_cudnn_rnn(
      cached_cast(at::kHalf, input),
      // 根据需要进行类型转换和扁平化的情况，选择传递不同的权重数据
      needs_cast_and_flatten ? TensorList(redispatch_weight) : weight,
      weight_stride0,
      needs_cast_and_flatten ? redispatch_weight_buf : weight_buf,
      cached_cast(at::kHalf, hx),
      cached_cast(at::kHalf, cx),
      mode,
      hidden_size,
      proj_size,
      num_layers,
      batch_first,
      dropout,
      train,
      bidirectional,
      batch_sizes,
      dropout_state);
#else // AT_CUDNN_ENABLED()
  // 如果未启用 cuDNN 支持，则抛出错误信息
  AT_ERROR("autocast::_cudnn_rnn_cast_reflatten: ATen not compiled with cuDNN support");
  // 永远不会执行到这里，用于安抚编译器
  return {Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}}; // never reached, placates the compiler
#endif // AT_CUDNN_ENABLED()
}

namespace {
// 定义匿名命名空间，用于实现 TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  // 将 _cudnn_rnn 的实现注册到 Autocast 下的 aten 模块中
  m.impl("_cudnn_rnn",
         TORCH_FN((&at::autocast::_cudnn_rnn_cast_reflatten)));
}
} // anonymous namespace

// 结束 autocast 命名空间
} // namespace autocast
// 结束 at 命名空间
} // namespace at
```