# `.\pytorch\aten\src\ATen\miopen\AutocastRNN.cpp`

```
/**********************************************************************
Autocast wrapper for MIOpen RNNs
**********************************************************************/
// 定义了一个名为 miopen_rnn 的函数，用于处理 MIOpen RNN 操作。
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
miopen_rnn(const Tensor & input_r,
           TensorList weight,
           int64_t weight_stride0,
           const Tensor & hx,
           const std::optional<Tensor>& cx_opt,
           int64_t fn_mode,
           int64_t fn_hidden_size,
           int64_t fn_num_layers,
           bool batch_first,
           double fn_dropout,
           bool fn_train,
           bool fn_bidirectional,
           IntArrayRef fn_batch_sizes,
           const std::optional<Tensor>& fn_dropout_state_opt) {

#if AT_ROCM_ENABLED()

    // 在 ROCm 平台下，禁用自动混合精度
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);

    // 调用 miopen_rnn 函数，将输入和权重转换为半精度，并传递其他参数
    return at::miopen_rnn(
                cached_cast(at::kHalf, input_r),
                cached_cast(at::kHalf, weight),
                weight_stride0,
                cached_cast(at::kHalf, hx),
                cached_cast(at::kHalf, cx_opt),
                fn_mode,
                fn_hidden_size,
                fn_num_layers,
                batch_first,
                fn_dropout,
                fn_train,
                fn_bidirectional,
                fn_batch_sizes,
                fn_dropout_state_opt);

#else
    // 如果不是 ROCm 平台，则抛出错误信息
    AT_ERROR("autocast::miopen_rnn: ATen not compiled with ROCm enabled");
    // 返回空的 Tensor 元组来满足编译器的要求
    return {Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}}; // placate the compiler
#endif

}

// Register Autocast dispatch
namespace {
// 在匿名命名空间中注册 Autocast 的函数调度
TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  m.impl("miopen_rnn",
         TORCH_FN((&at::autocast::miopen_rnn)));
}
} // anonymous namespace
```