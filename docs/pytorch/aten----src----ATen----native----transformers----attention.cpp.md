# `.\pytorch\aten\src\ATen\native\transformers\attention.cpp`

```
// 引入 ATen 库中的头文件，用于张量操作和计算
#include <ATen/core/TensorBody.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

// 引入 C10 库中的类型和设备相关的头文件
#include <c10/util/typeid.h>
#include <c10/core/DeviceType.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Logging.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>

// 引入标准库中的头文件
#include <type_traits>
#include <limits>
#include <utility>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则引入高层级的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则引入特定操作的头文件，用于混合操作
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#include <ATen/ops/_masked_softmax.h>
#include <ATen/ops/_native_multi_head_attention_native.h>
#include <ATen/ops/_nested_from_padded.h>
#include <ATen/ops/_nested_tensor_softmax_with_shape.h>
#include <ATen/ops/_scaled_dot_product_attention_math.h>
#include <ATen/ops/_scaled_dot_product_attention_math_native.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_native.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward_native.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_native.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_backward.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_backward_native.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/_transform_bias_rescale_qkv.h>
#include <ATen/ops/_transform_bias_rescale_qkv_native.h>
#include <ATen/ops/_triton_multi_head_attention_native.h>
#include <ATen/ops/_triton_scaled_dot_attention.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

// 引入 ATen 库中嵌套张量的转换函数相关的头文件
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

// 命名空间 at，包含了 ATen 库中的大部分功能和类
namespace at {
Tensor transform0213_gemm_nt_bias(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& query) {
  // 如果查询张量是嵌套的
  if (query.is_nested()) {
    // 根据嵌套的查询张量创建对应的嵌套张量
    at::Tensor nested_a = _nested_from_padded(
        a, get_nested_tensor_impl(query)->get_nested_sizes(), true);
    // 执行嵌套张量乘以张量加张量的操作
    return NestedTensor_times_Tensor_plus_Tensor_addmm(
        c, nested_a, b.t(), 1, 1);
  } else {
    // 否则，按照固定的维度顺序转换输入张量a
    const Tensor a_0213 = transform_0213(a);
    // 将转换后的张量a按照新的维度形状重新视图
    auto a_ = a_0213.view({a_0213.size(0) * a_0213.size(1), a_0213.size(2)});
    // 执行线性变换操作
    auto r_ = at::native::linear(a_, b, c);
    // 将结果视图转换回原始形状
    return r_.view({a_0213.size(0), a_0213.size(1), r_.size(1)});
  }
}
// 调试断言，用于验证张量的形状是否符合预期
void debug_assert_shape(int line, const Tensor& t, c10::IntArrayRef shape) {
  // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 进行调试断言，验证张量维度是否匹配预期维度数量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (size_t)t.dim() == shape.size(),
      "(called from line ",
      line,
      ") ",
      "expected ",
      shape.size(),
      "-D tensor but got ",
      t.dim());

  // 如果张量是嵌套的，则直接返回，不进行形状验证
  if (t.is_nested()) {
    return;
  }

  // 遍历预期形状的每个维度，验证每个维度的大小是否与预期相符
  for (auto idx : c10::irange(shape.size())) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        shape[idx] == 0 || t.sizes()[idx] == shape[idx],
        "(called from line ",
        line,
        ") ",
        "expected dim ",
        idx,
        " to be ",
        shape[idx],
        " but got ",
        t.sizes()[idx]);
  }
}

// 对查询、键、值进行投影操作，生成 QKV 张量
Tensor qkv_projection(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const Tensor& qkv_weight) {
  // 初始化 QKV 张量
  Tensor qkv;

  // 如果键和值相同
  if (key.is_same(value)) {
    // 如果查询和键也相同，进行自注意力计算
    if (query.is_same(key)) {
      qkv = gemm_nt(query, qkv_weight);  // 使用 gemm_nt 函数进行矩阵乘法计算
    } else {
      // 如果查询和键不同，进行编码器-解码器注意力计算
      // 拆分 QKV 权重张量为查询和键值的投影权重
      auto q_kv_weight_s =
          at::native::split_with_sizes(qkv_weight, {embed_dim, embed_dim * 2}, 0);
      // 断言拆分后应该得到两个张量
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          q_kv_weight_s.size() == 2,
          "expected split to produce 2 tensors but it produced ",
          q_kv_weight_s.size());
      // 分别进行查询和键值的矩阵乘法计算
      auto q = gemm_nt(query, q_kv_weight_s[0]);
      auto kv = gemm_nt(key, q_kv_weight_s[1]);
      // 将计算结果连接起来形成 QKV 张量
      qkv = at::cat({std::move(q), std::move(kv)}, 2);
    }
  } else {
    // 如果键和值不同，拆分 QKV 权重张量为查询、键、值的投影权重
    auto q_k_v_weight_s = at::native::chunk(qkv_weight, 3, 0);
    // 断言拆分后应该得到三个张量
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        q_k_v_weight_s.size() == 3,
        "expected chunk to produce 3 tensors but it produced ",
        q_k_v_weight_s.size());
    // 分别进行查询、键、值的矩阵乘法计算
    auto q = gemm_nt(query, q_k_v_weight_s[0]);
    auto k = gemm_nt(key, q_k_v_weight_s[1]);
    auto v = gemm_nt(value, q_k_v_weight_s[2]);
    // 将计算结果连接起来形成 QKV 张量
    qkv = at::cat({std::move(q), std::move(k), std::move(v)}, 2);
  }

  // 返回计算得到的 QKV 张量
  return qkv;
}

// 对 QKV 张量进行偏置转换和缩放操作
std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_cpu(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head) {
  // 如果 QKV 张量是嵌套的，将其转换为填充后的张量
  auto qkv_ = qkv.is_nested()
    ? c10::MaybeOwned<Tensor>::owned(qkv.to_padded_tensor(0))
  // 获取 qkv_ 的指针，指向 c10::MaybeOwned<Tensor> 类型
  auto qkv = c10::MaybeOwned<Tensor>::borrowed(qkv);
  // 获取 B 的大小，即 qkv_ 的第一维度大小
  auto B = qkv_->size(0);
  // 获取 T 的大小，即 qkv_ 的第二维度大小
  auto T = qkv_->size(1);
  // 获取 _3D 的大小，即 qkv_ 的第三维度大小
  auto _3D = qkv_->size(2);
  // 计算 D，即 qkv_ 的第三维度大小除以 3
  auto D = _3D / 3;
  // 检查 D 是否可以整除 num_head
  TORCH_CHECK(D % num_head == 0);
  // 检查 _3D 是否可以整除 3
  TORCH_CHECK(_3D % 3 == 0);
  // 计算每个注意力头的维度
  const auto dim_per_head = D / num_head;
  // 创建一个空的 Tensor，形状为 {3, B, num_head, T, dim_per_head}，使用 qkv_ 的选项
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv_->options());

  // 确保 qkv_ 是连续的 Tensor
  const auto qkv_contig = qkv_->expect_contiguous();
  // 确保 qkv_bias 是连续的 Tensor
  const auto qkv_bias_contig = qkv_bias.expect_contiguous();
  // 调用 transform_bias_rescale_qkv_stub 函数进行 QKV 转换和偏置调整
  transform_bias_rescale_qkv_stub(
      kCPU,
      qkv_->scalar_type(),
      q_k_v.data_ptr(),
      qkv_contig->const_data_ptr(),
      qkv_bias_contig->const_data_ptr(),
      B, T, D, num_head);
  // 将 q_k_v 按第一维度分割为 3 个 Tensor，每个 Tensor 的大小为 {B, num_head, T, dim_per_head}
  auto q_k_v_s =
      at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
  // 断言 q_k_v_s 的大小为 3
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(q_k_v_s.size() == 3);
  // 返回包含 q_k_v_s 的元组，分别代表 Q、K、V
  return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
// 定义一个函数，执行多头注意力机制的 CPU 实现，返回查询和键值对的元组
std::tuple<Tensor, Tensor> native_multi_head_attention_cpu(
    const Tensor& query,                   // 查询张量
    const Tensor& key,                     // 键张量
    const Tensor& value,                   // 值张量
    const int64_t embed_dim,               // 嵌入维度
    const int64_t num_head,                // 注意力头的数量
    const Tensor& qkv_weight,              // 查询、键、值权重张量
    const Tensor& qkv_bias,                // 查询、键、值偏置张量
    const Tensor& proj_weight,             // 投影权重张量
    const Tensor& proj_bias,               // 投影偏置张量
    const std::optional<Tensor>& mask,     // 可选的遮罩张量
    bool need_weights,                     // 是否需要注意力权重
    bool average_attn_weights,             // 是否需要平均注意力权重
    const std::optional<int64_t> mask_type // 可选的遮罩类型
) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]

  // 检查是否存在嵌套张量和遮罩，目前不支持带有遮罩的嵌套张量
  TORCH_CHECK(
      !mask || !query.is_nested(),
      "NestedTensor with mask is not supported yet");

  const auto D = embed_dim;

  // 检查查询张量的维度和形状
  TORCH_CHECK(
      query.dim() == 3,
      "expected 3-D `query`, got ",
      query.dim(),
      "-D tensor");
  TORCH_CHECK(
      query.is_nested() || query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);

  // 检查键和值张量的维度和形状，确保与查询张量匹配
  TORCH_CHECK(
      key.dim() == 3,
      "expected 3-D `key`, got ",
      key.dim(),
      "-D tensor");
  TORCH_CHECK(
      value.dim() == 3,
      "expected 3-D `value`, got ",
      value.dim(),
      "-D tensor");
  TORCH_CHECK(
      query.is_nested() || key.is_nested() || value.is_nested() ||
          (query.sizes() == key.sizes() && key.sizes() == value.sizes()),
      "expected `query`/`key`/`value` shapes to match");

  // 检查权重张量的维度和形状
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");

  // 检查偏置张量的维度和形状
  TORCH_CHECK(
      qkv_bias.dim() == 1,
      "expected 1-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");

  // 检查嵌入维度是否能被注意力头的数量整除
  TORCH_CHECK(D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

#ifndef NDEBUG
  // 如果处于调试模式，获取批次大小 B
  const auto B = query.is_nested()
      ? get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];
  const auto dim_per_head = D / num_head;
#endif

  // 对查询、键、值进行投影得到 qkv 张量，形状为 [B, T, 3 * D]
  auto qkv = qkv_projection(query, key, value, embed_dim, qkv_weight);

  // 如果 qkv 是嵌套张量且元素数量为 0，返回空张量的元组
  if (!qkv.is_nested() && qkv.numel() == 0) {
    if (query.is_nested()) {
      return std::make_tuple(Tensor(), Tensor());
    }
    return std::make_tuple(at::empty_like(query), Tensor());
  }

#ifndef NDEBUG
  // 在调试模式下，检查 qkv 的形状是否匹配预期
  if (!query.is_nested() || !qkv.is_nested()) {
    if (query.is_nested()) {
      T = qkv.size(1);
    }
    debug_assert_shape(__LINE__, qkv, {B, T, 3 * D});
  }
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  // 在每个步骤打印 qkv 张量，用于调试
  if (!qkv.is_nested()) {
    std::cerr << "qkv: " << qkv << std::endl;
  }
#endif
#endif
  // shape: 3 x [B, num_head, T, dim_per_head]
  // 调用函数_transform_bias_rescale_qkv对qkv进行偏置调整和重新缩放操作
  auto q_k_v = _transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  // 不再需要qkv，释放其内存空间
  qkv = Tensor(); // 不再使用，允许释放

  // 获取q、k、v三个分量的引用
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
#ifndef NDEBUG
  // 调试模式下检查张量的形状是否正确
  debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  // 调试模式下输出q、k、v的值到标准错误流
  std::cerr << "q: " << q << std::endl;
  std::cerr << "k: " << k << std::endl;
  std::cerr << "v: " << v << std::endl;
#endif

  // shape: [B, num_head, T, T]
  // 计算q和k的乘积，结果存储在qkt中
  auto qkt = bmm_nt(q, k);
#ifndef NDEBUG
  // 调试模式下检查张量的形状是否正确
  debug_assert_shape(__LINE__, qkt, {B, num_head, T, T});
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  // 调试模式下输出qkt的值到标准错误流
  std::cerr << "qkt: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, T]
  // 对qkt进行softmax操作，结果存储在qkt中
  qkt = masked_softmax(qkt, mask, query, mask_type);
#ifdef DEBUG_PRINT_EACH_STEP
  // 调试模式下输出softmax后的qkt值到标准错误流
  std::cerr << "qkt after softmax: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, dim_per_head]
  // 使用bmm_nn函数计算注意力上下文，结果存储在attn_ctx中
  auto attn_ctx = bmm_nn(q, qkt, v);

  // 如果不需要权重信息，则释放qkt的内存空间
  if (!need_weights) {
    qkt = Tensor();
  }
#ifndef NDEBUG
  // 调试模式下检查张量的形状是否正确
  debug_assert_shape(__LINE__, attn_ctx, {B, num_head, T, dim_per_head});
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  // 调试模式下输出attn_ctx的值到标准错误流
  std::cerr << "attn_ctx: " << attn_ctx << std::endl;
#endif

  // shape: [B, T, D]
  // 使用transform0213_gemm_nt_bias函数进行转换和偏置处理，结果存储在proj中
  auto proj = transform0213_gemm_nt_bias(
      attn_ctx, proj_weight, proj_bias, query);
#ifndef NDEBUG
  // 调试模式下检查张量的形状是否正确
  debug_assert_shape(__LINE__, proj, {B, T, D});
#endif

  // 如果需要权重信息并且要求平均注意力权重，则计算qkt在第一维度上的和并平均
  if (need_weights && average_attn_weights) {
    qkt = qkt.sum(1);
    qkt /= num_head;
  }

  // 返回proj和qkt的元组作为结果
  return std::make_tuple(std::move(proj), std::move(qkt));
}
// 下面是另一个函数_fused_sdp_choice_cpp，用于选择并返回scaled_dot_product_attention的后端类型
int64_t _fused_sdp_choice_cpp(const Tensor& query_, const Tensor& key, const Tensor& value,
        const std::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal, std::optional<double> scale){
  // 构造sdp_params结构体，存储参数信息
  sdp::sdp_params kernel_params{query_, key, value, attn_mask_, dropout_p, is_causal};
  // 选择合适的C++后端实现scaled_dot_product_attention
  auto backend = sdp::select_sdp_backend_cpp(kernel_params);
  // 如果未找到可行的后端实现，则抛出错误
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  // 将后端类型转换为int64_t类型并返回
  return static_cast<int64_t>(backend);
}

// 注册_fused_sdp_choice_cpp函数的调度器
REGISTER_ARCH_DISPATCH(_fused_sdp_choice_stub, DEFAULT, &_fused_sdp_choice_cpp);
// 注册 AVX2 指令集的分发函数，将 _fused_sdp_choice_stub 映射到 _fused_sdp_choice_cpp
REGISTER_AVX2_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);

// 注册 AVX512 指令集的分发函数，将 _fused_sdp_choice_stub 映射到 _fused_sdp_choice_cpp
REGISTER_AVX512_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);

// 注册 VSX 指令集的分发函数，将 _fused_sdp_choice_stub 映射到 _fused_sdp_choice_cpp
REGISTER_VSX_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);

// 注册 ZVECTOR 指令集的分发函数，将 _fused_sdp_choice_stub 映射到 _fused_sdp_choice_cpp
REGISTER_ZVECTOR_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);

// 定义 _fused_sdp_choice_meta 函数，用于执行融合的自注意力选择操作
int64_t _fused_sdp_choice_meta(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  // 获取查询张量的分发键集合
  auto query_key_set = query_.key_set();
  
  // 如果使用 ROCm 平台，检查是否具有 HIP 分发键
#if defined(USE_ROCM)
  bool has_rocm = query_key_set.has(c10::DispatchKey::HIP);
  if (has_rocm) {
    // 调用 HIP 平台上的 _fused_sdp_choice_stub 函数进行自注意力操作选择
    auto choice_int = _fused_sdp_choice_stub(at::kHIP, query_, key, value, attn_mask_, dropout_p, is_causal, scale);
    return choice_int;
  }
#else
  // 如果使用 CUDA 平台，检查是否具有 CUDA 分发键
  bool has_cuda = query_key_set.has(c10::DispatchKey::CUDA);
  if (has_cuda) {
    // 调用 CUDA 平台上的 _fused_sdp_choice_stub 函数进行自注意力操作选择
    auto choice_int = _fused_sdp_choice_stub(
        at::kCUDA,
        query_,
        key,
        value,
        attn_mask_,
        dropout_p,
        is_causal,
        scale);
    return choice_int;
  }
#endif

  // 返回默认的 SDPBackend::math 值作为后备选项
  return static_cast<int64_t>(sdp::SDPBackend::math);
}

// 匿名命名空间内部的函数，用于验证自注意力操作的输入张量是否符合预期
inline void validate_sdpa_input(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  // 检查查询、键、值张量的数据类型是否相同
  TORCH_CHECK(
      query_.dtype() == key.dtype() && query_.dtype() == value.dtype(),
      "Expected query, key, and value to have the same dtype, but got query.dtype: ",
      query_.dtype(), " key.dtype: ", key.dtype(), " and value.dtype: ", value.dtype(), " instead.");
  
  // 检查查询、键、值张量的设备类型是否相同
  TORCH_CHECK(
      query_.device() == key.device() && query_.device() == value.device(),
      "Expected query, key, and value to have the same device type, but got query.device: ",
      query_.device(), " key.device: ", key.device(), " and value.device: ", value.device(), " instead.");
  
  // 检查查询、键、值张量的维度是否至少为 2
  TORCH_CHECK(
      query_.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
      "Expected query, key, and value to all be at least 2 dimensional, but got query.dim: ",
      query_.dim(), " key.dim: ", key.dim(), " and value.dim: ", value.dim(), " instead.");
  
  // 如果存在注意力掩码，验证其数据类型是否为布尔型、浮点型或与查询张量的数据类型相同，并且验证查询和键张量不是嵌套的
  if (attn_mask_.has_value()){
    auto mask_dtype = attn_mask_->dtype();
    TORCH_CHECK(mask_dtype == at::kBool || mask_dtype == at::kFloat || mask_dtype == query_.dtype(),
      "Expected attn_mask dtype to be bool or float or to match query dtype, but got attn_mask.dtype: ",
      mask_dtype, " and query.dtype: ", query_.dtype(), " instead.");
    TORCH_CHECK(
      !query_.is_nested() && !key.is_nested(),
      "Scaled_dot_product_attention: Nested tensors for query / key are not supported "
      "when an explicit attn_mask is set");
  }
  return;
}
// 使用可选类型的张量 attn_mask，可以是形状为 (B, L, S) 或 (L, S) 或 (B, N_heads, L, S)
std::optional<Tensor> convert_boolean_attn_mask(const std::optional<Tensor>& attn_mask, caffe2::TypeMeta dtype) {
  // 如果 attn_mask 不存在值，则直接返回空值
  if (!attn_mask.has_value()) {
    return c10::nullopt;
  }
  // 如果 attn_mask 的数据类型是布尔型
  if (attn_mask->dtype() == at::kBool) {
    // 创建一个与 attn_mask 相同形状的全零张量，并使用指定的数据类型 dtype
    auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
    // 将新的注意力掩码张量中非真值位置填充为负无穷大，用于表示需要屏蔽的位置
    new_attn_mask.masked_fill_(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
    return new_attn_mask;
  }
  // 如果 attn_mask 不是布尔型，则假定它表示一个加性注意力张量，直接返回
  return attn_mask;
}

// 内存高效注意力需要一个填充的注意力偏置
// 此函数将 attn_bias 填充为 16 的倍数，并切片回原始大小
// 我们将此函数应用于顶层 SDPA，以便在自动反向传播中跟踪填充操作

template<int alignment>
bool aligned_tensor(const at::Tensor& tensor){
  for(const auto i : c10::irange(tensor.dim() - 1)){
    // 检查张量除最后一个维度外的所有维度是否都是 alignment 的倍数
    if(tensor.sym_stride(i) % alignment != 0){
      return false;
    }
  }
  // 检查张量最后一个维度是否步长为 1
  return tensor.sym_stride(-1) == 1;
}

// 使用指定的对齐大小 alignment，对 attn_bias 进行填充处理
template <int alignment>
at::Tensor pad_bias(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  auto pad_count = alignment - (last_dim_size % alignment);
  // 对 attn_bias 进行符号整数填充
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  // 切片回原始大小，只保留最后一个维度
  return padded_bias.slice_symint(-1, 0, last_dim_size);
}

// 预处理掩码，确保掩码张量符合内存高效对齐要求
at::Tensor preprocess_mask(
    const at::Tensor& mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value) {
  constexpr int mem_eff_alignment = 8;
  at::Tensor result_mask = mask;
  // 如果 mask 张量不符合 mem_eff_alignment 的对齐要求，则进行填充处理
  if (!aligned_tensor<mem_eff_alignment>(mask)) {
    result_mask = pad_bias<mem_eff_alignment>(mask);
  }
  // 将结果掩码张量扩展到指定的形状，以适应 query 和 key 的维度
  return result_mask.expand_symint(
      {query.sym_size(0),
       query.sym_size(1),
       query.sym_size(2),
       key.sym_size(2)});
}

// FlashAttentionV2 要求头维度必须是 8 的倍数
// 在复合区域内，我们对头维度进行填充处理，以确保满足这一要求

template <int alignment_size, bool slice>
at::Tensor pad_last_dim(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  // 如果最后一个维度已经是 alignment_size 的倍数，则直接返回
  if (last_dim_size % alignment_size == 0) {
    return attn_bias;
  }
  auto pad_count = alignment_size - (last_dim_size % alignment_size);
  // 对最后一个维度进行符号整数填充
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  // 如果 slice 为真，则切片回原始大小，否则返回填充后的张量
  if (slice) {
    return padded_bias.slice_symint(-1, 0, last_dim_size);
  }
  return padded_bias;
}

// 后处理 FlashAttention 输出，确保其符合原始大小 og_size
at::Tensor post_process_flash_output(
    at::Tensor out,
    c10::SymInt const& og_size) {
  // 如果输出张量不是嵌套结构且最后一个维度不等于 og_size，则进行处理
  if (!out.is_nested() && out.sym_size(-1) != og_size) {
    out = out.slice_symint(-1, 0, og_size);
  }
  return out;



    // 使用 `slice_symint` 方法对 `out` 进行切片操作，参数为 `-1`、`0` 和 `og_size`
    out = out.slice_symint(-1, 0, og_size);
  }
  // 返回切片后的结果 `out`
  return out;
}

// 检查是否需要计算 logsumexp 的条件函数
bool should_compute_logsumexp(const Tensor& query, const Tensor& key, const Tensor& value) {
  // 检查是否有任何一个输入需要梯度
  const bool any_inputs_require_grad = query.requires_grad() || key.requires_grad() || value.requires_grad();
  // 检查梯度模式是否已启用
  const bool gradmode_enabled = at::GradMode::is_enabled();
  // 返回是否有输入需要梯度且梯度模式已启用的逻辑值
  return any_inputs_require_grad && gradmode_enabled;
}

} // namespace

// 对查询、键和值张量执行缩放点积注意力计算，可选择应用注意力掩码（如果传入），并在指定的概率大于0.0时应用 dropout。
//
// 参数:
//     query (Tensor): 查询张量; 形状为 (N, ..., L, E)
//     key (Tensor): 键张量; 形状为 (N, ..., S, E)
//     value (Tensor): 值张量; 形状为 (N, ..., S, E)
//     attn_mask (optional Tensor): 注意力掩码; 形状必须与注意力权重的形状可广播，即 (N,..., L, S)。支持两种类型的掩码。
//         布尔掩码，其中 True 表示元素应参与注意力。
//         与查询、键、值相同类型的浮点掩码，会加到注意力分数中。
//     dropout_p (float): Dropout 概率; 如果大于 0.0，则应用 dropout
//     need_attn_weights (bool): 如果为 true，则第二个返回值将包含使用的注意力权重；否则，第二个返回值未指定
//     is_causal (bool): 如果为 true，则假定因果注意力掩码；对于这种情况，不应设置 attn_mask。
//         TODO: 考虑在将此函数提升为公共 API 之前删除此标志。可以通过张量子类化获取对因果掩码（以及其他类型的掩码，例如局部注意力/块稀疏掩码）的专门支持，从而实现更精简的 API。
//
// 返回张量:
//     output (Tensor): 注意力输出; 形状为 (N, ..., L, E)
//
// 形状说明:
//     N: 批量大小
//     ...: 其他任意批量维度（可选）
//     S: 源序列长度
//     L: 目标序列长度
//     E: 嵌入维度
Tensor scaled_dot_product_attention(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  // 验证缩放点积注意力计算的输入参数有效性
  validate_sdpa_input(query_, key, value, attn_mask_, dropout_p, is_causal, scale);
  // 将枚举值转换为整数
  int64_t choice_int = static_cast<int64_t>(sdp::SDPBackend::math);
  // 检查设备是否支持融合的缩放点积操作选择
  if (_fused_sdp_choice_stub.is_device_supported(query_.device().type())) {
    // 获取融合的缩放点积操作的选择
    choice_int = _fused_sdp_choice_stub(query_.device().type(),
          query_, key, value, attn_mask_, dropout_p, is_causal, scale);
  }
  // 将整数转换为枚举值
  sdp::SDPBackend backend = static_cast<sdp::SDPBackend>(choice_int);
  // 将布尔类型的注意力掩码转换为相同类型的张量掩码
  std::optional<Tensor> attn_mask = convert_boolean_attn_mask(attn_mask_, query_.dtype());
  // 根据后端选择执行不同的操作
  switch (backend) {
    // 处理 cudnn_attention 情况
    case sdp::SDPBackend::cudnn_attention: {
      // 根据条件判断是否需要计算 logsumexp
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      // 调用 cuDNN 加速的注意力计算函数
      auto out_lse_softmax = at::_scaled_dot_product_cudnn_attention(
          query_, key, value, dropout_p, is_causal, compute_logsumexp, scale);
      // 返回注意力计算结果的第一个元素
      return std::get<0>(out_lse_softmax);
    }
    // 处理 flash_attention 情况
    case sdp::SDPBackend::flash_attention: {
      // 如果查询张量在 CUDA 设备上
      if(query_.device().type() == DeviceType::CUDA){
        // 获取查询张量最后一个维度的符号化整数
        c10::SymInt og_size = query_.sym_size(-1);
        // 对查询、键、值张量进行最后一个维度的填充
        Tensor query_padded = pad_last_dim<8, false>(query_);
        Tensor key_padded = pad_last_dim<8, false>(key);
        Tensor value_padded = pad_last_dim<8, false>(value);
        // 根据原始头部维度大小计算缩放因子
        auto og_scale = sdp::calculate_scale(query_, scale);
        // 调用 Flash 注意力计算函数
        auto out_lse_softmax = at::_scaled_dot_product_flash_attention(
            query_padded, key_padded, value_padded, dropout_p, is_causal, false /*return_debug_mask*/, og_scale.as_float_unchecked());
        // 对 Flash 注意力计算结果进行后处理
        return post_process_flash_output(std::get<0>(out_lse_softmax), og_size);
      }
      // 对于 CPU 情况，不需要对最后一个维度进行填充
      return std::get<0>(at::_scaled_dot_product_flash_attention_for_cpu(
          query_, key, value, dropout_p, is_causal, attn_mask, scale));
    }
    // 处理 efficient_attention 情况
    case sdp::SDPBackend::efficient_attention: {
      // 根据条件判断是否存在注意力掩码，并预处理注意力掩码
      if (attn_mask.has_value()) {
        attn_mask.value() = preprocess_mask(attn_mask.value(), query_, key, value);;
      }
      // 判断是否需要计算 logsumexp，并调用高效注意力计算函数
      auto out_and_lse = at::_scaled_dot_product_efficient_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, scale);
      // 返回高效注意力计算结果的第一个元素
      return std::get<0>(out_and_lse);
    }
    // 处理 overrideable 情况
    case sdp::SDPBackend::overrideable: {
      // 调用可重写的融合注意力计算函数
      auto out_lse_softmax = at::_scaled_dot_product_fused_attention_overrideable(
          query_, key, value, attn_mask, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      // 返回融合注意力计算结果的第一个元素
      return std::get<0>(out_lse_softmax);
    }
    // 处理 math 情况
    case sdp::SDPBackend::math:
      // 调用数学运算版本的缩放点积注意力计算函数
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_,
          key,
          value,
          attn_mask,
          dropout_p,
          is_causal,
          c10::nullopt, /*dropout_mask*/
          scale));
    // 处理默认情况，抛出错误信息并返回空张量
    default:
      TORCH_CHECK(
          false,
          "No viable backend for scaled_dot_product_attention was found.");
      return Tensor();
  }
}

// 定义一个函数 _scaled_dot_product_attention_math，用于执行缩放点积注意力机制的数学计算
std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math(
        const Tensor& query_, const Tensor& key, const Tensor& value,
        const std::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal,
        const std::optional<Tensor>& dropout_mask, std::optional<double> scale) {
  // 记录 API 使用情况，这里标记了一次使用 "torch.sdpa.math_fallback"
  C10_LOG_API_USAGE_ONCE("torch.sdpa.math_fallback");

  // 如果输入中包含嵌套张量，要求它们必须是连续的
  if (query_.is_nested() || key.is_nested() || value.is_nested()) {
    TORCH_CHECK(
        query_.is_contiguous() && key.is_contiguous() &&
            value.is_contiguous(),
        "scaled_dot_product_attention: If inputs are nested tensors they must be contiguous");
  }

  // 复制 attn_mask，如果有的话
  auto attn_mask = attn_mask_;

  // 计算缩放因子，用于缩放查询向量和键向量
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto scaling_factor = sdp::calculate_scale(query_, is_negative_scaling ? std::abs(scale.value()) : scale).sqrt();

  // 缩放查询向量
  const auto query = query_ * (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor: scaling_factor);

  // 如果 is_causal 为真，执行以下操作
  if (is_causal) {
    // 校验条件：is_causal 为真时，不应设置显式的 attn_mask
    TORCH_CHECK(!attn_mask.has_value(),
                "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");

    // 校验条件：is_causal 为真时，不支持查询/键的嵌套张量
    TORCH_CHECK(!query.is_nested() && !key.is_nested(),
                "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

    // 生成因果注意力的掩码，下三角元素参与注意力计算
    const auto L = query.sym_size(-2), S = key.sym_size(-2);
    attn_mask = at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
    attn_mask = convert_boolean_attn_mask(attn_mask, query.dtype());
  }

  // 计算注意力分数矩阵
  auto attn = at::matmul(query, key.transpose(-2, -1) * scaling_factor);

  // 如果存在 attn_mask，则将其加到注意力分数上
  if (attn_mask.has_value()) {
    if (at::areAnyTensorSubclassLike({attn, *attn_mask})) {
      attn = attn.add(*attn_mask);
    } else {
      attn.add_(*attn_mask);
    }
  }

  // 对注意力分数进行 softmax 操作
  attn = at::softmax(attn, -1);

  // 如果 dropout_p 大于 0，执行 dropout 操作
  if (dropout_p > 0.0) {
    if (dropout_mask.has_value()) {
      // 用于测试目的，需要使用相同的 dropout 掩码以验证融合内核的正确性
      TORCH_WARN_ONCE("Dropout mask should only be used for testing purposes.");
      attn = attn.masked_fill(dropout_mask->logical_not(), 0.0);
      auto dropout_scaling = 1.0 / (1 - dropout_p);
      return std::make_tuple(at::matmul(attn, value * dropout_scaling), attn);
    } else {
      attn = at::dropout(attn, dropout_p, true);
    }
  }

  // 返回最终的注意力权重和加权值
  return std::make_tuple(at::matmul(attn, value), attn);
}

// 定义一个函数 _scaled_dot_product_flash_attention_cpu，用于在 CPU 上执行缩放点积注意力机制
std::tuple<at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_cpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    // 定义函数 scaled_dot_product_attention_flash_attention，实现注意事项的注意力机制
    std::tuple<at::Tensor, at::Tensor> scaled_dot_product_attention_flash_attention(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        double dropout_p,
        bool is_causal,
        std::optional<at::Tensor> attn_mask,
        std::optional<double> scale) {
      
      // 获取查询张量的数据类型
      const auto dtype = query.scalar_type();
      // 获取批量大小
      int64_t batchSize = query.size(0);
      // 获取查询的序列长度
      int64_t qSize = query.size(2);
      // 获取头数
      int64_t num_head = query.size(1);
      // 获取每个头的大小
      int64_t headSize = query.size(3);
    
      // 检查数据类型是否为浮点型
      TORCH_CHECK(c10::isFloatingType(dtype),
        "scaled_dot_product_attention_flash_attention: Expected data type in FP32, FP64, BF16, FP16, but got ", dtype, " instead.");
      // 检查输入张量维度是否为4
      TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
        "scaled_dot_product_attention_flash_attention: Accept only 4 dims inputs shape of {B, H, T, K}");
      // 检查是否支持 dropout
      TORCH_CHECK(dropout_p == 0.0,
        "scaled_dot_product_attention_flash_attention: Currently do not support dropout > 0");
      // 检查 Q/K/V 张量的头大小是否相同
      TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
        "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
      // 检查注意力掩码的数据类型与查询张量是否一致
      TORCH_CHECK(!attn_mask.has_value() ||
              attn_mask.value().scalar_type() == at::kFloat ||
              dtype == attn_mask.value().scalar_type(),
        "scaled_dot_product_attention_flash_attention: Attention mask is the same data type as query");
      // 检查注意力掩码的维度是否为2或4
      TORCH_CHECK(!attn_mask.has_value() ||
              (attn_mask.value().dim() == 2 || attn_mask.value().dim() == 4),
        "scaled_dot_product_attention_flash_attention: Attention mask dim in {2, 4}");
    
      // 创建输出张量，形状为 {batchSize, qSize, num_head, headSize}，使用与查询张量相同的选项
      at::Tensor output = at::empty({batchSize, qSize, num_head, headSize}, query.options());
      // 根据累积数据类型创建 logsumexp 张量，形状为 {batchSize, qSize, num_head}
      const auto accumulate_dtype = toOpMathType(dtype);
      at::Tensor logsumexp = at::empty({batchSize, qSize, num_head},
          query.options().dtype(accumulate_dtype));
    
      // 调用 flash_attention_kernel 函数执行注意力计算
      flash_attention_kernel(kCPU, output, logsumexp,
          query, key, value, dropout_p, is_causal, attn_mask, scale);
    
      // 转置输出张量和 logsumexp 张量的维度 1 和 2
      output = output.transpose(1, 2);
      logsumexp = logsumexp.transpose(1, 2);
    
      // 返回结果元组，包含输出张量和 logsumexp 张量
      return std::make_tuple(std::move(output), std::move(logsumexp));
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_cpu_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  // 如果梯度 grad_out 未定义，返回空张量元组
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }
  // 对梯度 grad_out 进行转置操作
  auto grad_out_t = grad_out.transpose(1, 2);
  auto q_t = query.transpose(1, 2);
  auto k_t = key.transpose(1, 2);
  auto v_t = value.transpose(1, 2);
  auto o_t = out.transpose(1, 2);
  auto lse_t = logsumexp.transpose(1, 2);

  // 创建与输入 query, key, value 相同大小的零张量作为梯度张量
  auto grad_q = at::zeros(q_t.sizes(), query.options());
  auto grad_k = at::zeros(k_t.sizes(), key.options());
  auto grad_v = at::zeros(v_t.sizes(), value.options());

  // 调用 CPU 上的 flash_attention_backward_kernel 函数进行反向传播
  flash_attention_backward_kernel(kCPU, grad_q, grad_k, grad_v,
      grad_out_t, q_t, k_t, v_t, o_t, lse_t,
      dropout_p, is_causal, attn_mask, scale);

  // 将梯度张量再次进行转置
  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);

  // 返回梯度张量的元组
  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const c10::optional<at::Tensor> & attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  // 抛出未实现错误信息，提示 _scaled_dot_product_fused_attention_overrideable 函数未实现
  TORCH_CHECK_NOT_IMPLEMENTED(false, "_scaled_dot_product_fused_attention_overrideable not implemented. This is an operator for privateuse1 backends, please use TORCH_LIBRARY_IMPL to override this function ");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor & grad_out,
    const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const at::Tensor & attn_bias,
    std::array<bool,4> grad_input_mask,
    const at::Tensor & out,
    const at::Tensor & logsumexp,
    const at::Tensor & cum_seq_q,
    const at::Tensor & cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor & philox_seed,
    const at::Tensor & philox_offset,
    std::optional<double> scale) {
  // 抛出未实现错误信息，提示 _scaled_dot_product_fused_attention_overrideable_backward 函数未实现
  TORCH_CHECK_NOT_IMPLEMENTED(false, "_scaled_dot_product_fused_attention_overrideable_backward not implemented: This is an operator for privateuse1 backends, please use TORCH_LIBRARY_IMPL to override this function ");
  // 返回与 query, key, value, attn_bias 相同形状的空张量元组
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
          at::empty_like(query),
          at::empty_like(key),
          at::empty_like(value),
          at::empty_like(attn_bias));
}

Tensor triton_multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    // 检查是否存在 mask，只支持 Triton 中的因果 mask
    const int64_t embed_dim,
    // 多头注意力的数量
    const int64_t num_head,
    // 查询、键、值的权重张量
    const Tensor& qkv_weight,
    // 查询、键、值的偏置张量
    const Tensor& qkv_bias,
    // 投影的权重张量
    const Tensor& proj_weight,
    // 投影的偏置张量
    const Tensor& proj_bias,
    // 可选的 mask 张量，用于屏蔽特定位置的信息
    const std::optional<Tensor>& mask) {
    // 查询张量的形状: [B, T, D]
    // qkv_weight 的形状: [3 * D, D]
    TORCH_CHECK(!mask, "Only causal mask is supported for Triton.");
    // 断言不支持 mask，仅支持 Triton 的因果 mask
    
    const auto D = embed_dim;
    // 检查查询张量的维度是否为 3
    TORCH_CHECK(
        query.dim() == 3,
        "expected 3-D `query`, got ",
        query.dim(),
        "-D tensor");
    // 检查查询张量的最后一个维度是否与 embed_dim 一致
    TORCH_CHECK(
        query.sizes()[2] == embed_dim,
        "passed-in embed_dim ",
        embed_dim,
        " didn't match last dim of query ",
        query.sizes()[2]);
    // 检查键张量的维度是否为 3
    TORCH_CHECK(
        key.dim() == 3,
        "expected 3-D `key`, got ",
        key.dim(),
        "-D tensor");
    // 检查值张量的维度是否为 3
    TORCH_CHECK(
        value.dim() == 3,
        "expected 3-D `value`, got ",
        value.dim(),
        "-D tensor");
    // 检查查询、键、值张量的形状是否一致
    TORCH_CHECK(
            query.sizes() == key.sizes() && key.sizes() == value.sizes(),
        "expected `query`/`key`/`value` shapes to match");
    // 检查 qkv_weight 的维度是否为 2
    TORCH_CHECK(
        qkv_weight.dim() == 2,
        "expected 2-D `qkv_weight`, got ",
        qkv_weight.dim(),
        "-D tensor");
    // 检查 qkv_weight 的第一个维度是否为 3 倍的 embed_dim
    TORCH_CHECK(
        D * 3 == qkv_weight.sizes()[0],
        "expected `qkv_weight` first dim to be 3x embed_dim");
    // 检查 qkv_weight 的第二个维度是否为 embed_dim
    TORCH_CHECK(
        D == qkv_weight.sizes()[1],
        "expected `qkv_weight` second dim to be embed_Dim");
#ifndef NDEBUG
  // 如果处于调试模式，根据查询张量是否嵌套来确定 B 的大小，否则使用查询张量的第一个维度大小
  const auto B = query.is_nested()
      ? get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  // 如果查询张量是嵌套的，则 T 设为 0；否则使用查询张量的第二个维度大小
  auto T = query.is_nested() ? 0 : query.sizes()[1];
  // 每个注意力头的维度，D 除以注意力头的数量得到
  const auto dim_per_head = D / num_head;
#endif

  // shape: [B, T, 3 x D]
  // 对查询、键、值进行投影得到 qkv 张量
  auto qkv = qkv_projection(query, key, value, embed_dim, qkv_weight);

  // shape: 3 x [B, num_head, T, dim_per_head]
  // 对投影后的 qkv 张量进行偏置和重缩放，得到 q_k_v
  auto q_k_v = _transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  qkv = Tensor(); // 不再使用 qkv，释放其内存

  // 分别获取 q、k、v
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);

#ifndef NDEBUG
  // 如果处于调试模式，验证 q、k、v 张量的形状是否为 [B, num_head, T, dim_per_head]
  debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  // 如果定义了 DEBUG_PRINT_EACH_STEP，输出 q、k、v 张量的值到标准错误流
  std::cerr << "q: " << q << std::endl;
  std::cerr << "k: " << k << std::endl;
  std::cerr << "v: " << v << std::endl;
#endif

  // 执行 Triton 加权缩放点注意力操作，得到 attn_ctx
  auto attn_ctx = at::_triton_scaled_dot_attention(q, k, v);

#ifndef NDEBUG
  // 如果处于调试模式，验证 attn_ctx 张量的形状是否为 [B, num_head, T, dim_per_head]
  debug_assert_shape(__LINE__, attn_ctx, {B, num_head, T, dim_per_head});
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  // 如果定义了 DEBUG_PRINT_EACH_STEP，输出 attn_ctx 张量的值到标准错误流
  std::cerr << "attn_ctx: " << attn_ctx << std::endl;
#endif

  // shape: [B, T, D]
  // 执行内部的 transform_0213_gemm_nt_bias 操作，将 attn_ctx 进行投影得到 proj 张量
  auto proj = transform0213_gemm_nt_bias(
      attn_ctx, proj_weight, proj_bias, query);

#ifndef NDEBUG
  // 如果处于调试模式，验证 proj 张量的形状是否为 [B, T, D]
  debug_assert_shape(__LINE__, proj, {B, T, D});
#endif

  // 返回投影后的张量 proj
  return proj;
}
} // namespace native
} // namespace at
```