# `.\pytorch\aten\src\ATen\native\cudnn\MHA.cpp`

```py
// 引入 ATen 库的相关头文件
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

// 检查是否需要启用 cuDNN Flash Attention 功能
#if defined(USE_ROCM) || !AT_CUDNN_ENABLED() || \
    (defined(CUDNN_VERSION) && CUDNN_VERSION < 8900)

namespace at {
namespace native {

// 当 PyTorch 未启用 cuDNN Flash Attention 时，定义的前向传播函数
void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  // 抛出错误，指示未启用 cuDNN Flash Attention 功能
  TORCH_CHECK(
      false, "PyTorch was not compiled with cuDNN Flash Attention enabled!");
}

// 当 PyTorch 未启用 cuDNN Flash Attention 时，定义的反向传播函数
void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  // 抛出错误，指示未启用 cuDNN Flash Attention 功能
  TORCH_CHECK(
      false, "PyTorch was not compiled with cuDNN Flash Attention enabled!");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8900

// 当启用 cuDNN Flash Attention 时，引入必要的头文件和命名空间
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/native/cudnn/MHA.h>

#include <ATen/cuda/Exceptions.h>
#include <cudnn_frontend.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <cudnn.h>

#include <iostream>

namespace at {
namespace native {

// 引入 cudnn_frontend 命名空间，简化 cudnn_frontend 的别名为 fe
namespace fe = cudnn_frontend;

// 定义用于前向传播的图和张量描述的元组类型别名
using graph_and_tensors = std::tuple<
    std::shared_ptr<fe::graph::Graph>,                  // 图
    std::shared_ptr<fe::graph::Tensor_attributes>,     // Q,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // K,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // V,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // Attn_scale,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // Seed,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // Offset,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // O
    std::shared_ptr<fe::graph::Tensor_attributes>      // Stats
>;

// 定义用于反向传播的图和张量描述的元组类型别名
using graph_and_tensors_backward = std::tuple<
    std::shared_ptr<fe::graph::Graph>,                  // 图
    std::shared_ptr<fe::graph::Tensor_attributes>,     // Q,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // K,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // V,
    std::shared_ptr<fe::graph::Tensor_attributes>,     // O
    std::shared_ptr<fe::graph::Tensor_attributes>      // Stats
>;

#endif // AT_CUDNN_ENABLED && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8900
    // 定义一个包含多个指向 fe::graph::Tensor_attributes 对象的 shared_ptr 的模板参数列表
    std::shared_ptr<fe::graph::Tensor_attributes>, // Attn_scale
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示注意力缩放因子
    std::shared_ptr<fe::graph::Tensor_attributes>, // Seed,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示种子
    std::shared_ptr<fe::graph::Tensor_attributes>, // Offset,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示偏移量
    std::shared_ptr<fe::graph::Tensor_attributes>, // O,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示输出 O
    std::shared_ptr<fe::graph::Tensor_attributes>, // dO,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示输出的导数 dO
    std::shared_ptr<fe::graph::Tensor_attributes>, // stats,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示统计数据
    std::shared_ptr<fe::graph::Tensor_attributes>, // dQ,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示查询的导数 dQ
    std::shared_ptr<fe::graph::Tensor_attributes>, // dK,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示键的导数 dK,
    std::shared_ptr<fe::graph::Tensor_attributes> // dV,
    // 指向一个 fe::graph::Tensor_attributes 对象的 shared_ptr，表示值的导数 dV
    >;
#define MAX_MHA_DIM 4
// 定义最大的多头注意力机制维度

struct MHAParams {
  c10::DeviceIndex device_id;  // 设备索引
  fe::DataType_t dataType;    // 数据类型
  std::array<int, MAX_MHA_DIM> q_dim;  // 查询张量的维度数组
  std::array<int, MAX_MHA_DIM> k_dim;  // 键张量的维度数组
  std::array<int, MAX_MHA_DIM> v_dim;  // 值张量的维度数组
  std::array<int, MAX_MHA_DIM> q_stride;  // 查询张量的步长数组
  std::array<int, MAX_MHA_DIM> k_stride;  // 键张量的步长数组
  std::array<int, MAX_MHA_DIM> v_stride;  // 值张量的步长数组
  int64_t b;  // 批次大小
  int64_t h;  // 头数
  int64_t s_q;  // 查询张量的步长
  int64_t s_kv;  // 键值张量的步长
  int64_t d;  // 向量维度
  double dropout_probability;  // dropout概率
  bool is_causal;  // 是否因果
  bool return_softmaxstats;  // 是否返回softmax统计信息
};

void setMHAParams(
    MHAParams& params,
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    double dropout_probability,
    bool is_causal,
    bool return_softmaxstats) {
  // 将params结构体清零
  memset(&params, 0, sizeof(MHAParams));
  // 获取当前CUDA设备的ID作为设备索引
  params.device_id = at::cuda::current_device();
  // 设置数据类型为HALF
  params.dataType = fe::DataType_t::HALF;
  // 如果查询张量的标量类型为kBFloat16，则设置数据类型为BFLOAT16
  if (q.scalar_type() == kBFloat16) {
    params.dataType = fe::DataType_t::BFLOAT16;
  }
  // 设置批次大小、头数、向量维度、查询张量步长、键值张量步长、dropout概率、是否因果和是否返回softmax统计信息
  params.b = b;
  params.h = h;
  params.d = d;
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.dropout_probability = dropout_probability;
  params.is_causal = is_causal;
  params.return_softmaxstats = return_softmaxstats;
  // 断言查询张量、键张量和值张量的维度和步长大小为MAX_MHA_DIM，否则报告错误
  TORCH_INTERNAL_ASSERT(
      q.sizes().size() == MAX_MHA_DIM,
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      q.strides().size() == MAX_MHA_DIM,
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.sizes().size() == MAX_MHA_DIM,
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.strides().size() == MAX_MHA_DIM,
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.sizes().size() == MAX_MHA_DIM,
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.strides().size() == MAX_MHA_DIM,
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  // 将查询张量、键张量和值张量的维度和步长拷贝到params结构体中的数组中
  std::copy(q.sizes().begin(), q.sizes().end(), params.q_dim.begin());
  std::copy(q.strides().begin(), q.strides().end(), params.q_stride.begin());
  std::copy(k.sizes().begin(), k.sizes().end(), params.k_dim.begin());
  std::copy(k.strides().begin(), k.strides().end(), params.k_stride.begin());
  std::copy(v.sizes().begin(), v.sizes().end(), params.v_dim.begin());
  std::copy(v.strides().begin(), v.strides().end(), params.v_stride.begin());
}

struct MHACacheKeyWrapper : ParamsWrapper<MHAParams> {
  MHACacheKeyWrapper(
      int64_t b,
      int64_t h,
      int64_t s_q,
      int64_t s_kv,
      int64_t d,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      double dropout_probability,
      bool is_causal,
      bool return_softmaxstats) {
    # 调用函数 `setMHAParams` 并传入以下参数：
    # - `this->pod`: 指向当前对象的指针
    # - `b`: batch 大小
    # - `h`: 头数（多头注意力机制中的头数）
    # - `s_q`: 查询向量的尺寸
    # - `s_kv`: 键和值的尺寸
    # - `d`: 模型的维度
    # - `q`: 查询张量
    # - `k`: 键张量
    # - `v`: 值张量
    # - `dropout_probability`: dropout 概率
    # - `is_causal`: 是否为因果（causal）模型
    # - `return_softmaxstats`: 是否返回 softmax 统计信息
};

// 定义一个模板结构体 MHAGraphCache，用于存储类型为 T 的数据，键类型为 KeyType
template <typename T, typename KeyType>
struct MHAGraphCache {
  // 使用无序映射 engine_cache 存储键值对，键类型为 KeyType，值类型为 T，使用 ParamsWrapperHash<KeyType> 进行哈希
  std::unordered_map<KeyType, T, ParamsWrapperHash<KeyType>> engine_cache;

  // 查找指定键的缓存数据，如果找不到返回 nullptr，无线程互斥锁，因为缓存现在是线程局部的，可以返回执行计划的指针，如果我们知道它不会被其他线程失效
  T* find(const KeyType& key) {
    auto it = engine_cache.find(key);
    if (it == engine_cache.end()) {
      return nullptr;
    }
    return &(it->second);
  }

  // 更新指定键的缓存数据
  void update(const KeyType& key, T& results) {
    engine_cache.erase(key);
    engine_cache.emplace(key, std::move(results));
  }
};

// @eqy: 使用线程局部缓存，因为 cuDNN 执行计划不能保证在所有引擎间线程安全，参见 https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html 中的限制
// 定义线程局部变量 mhagraphcache 和 mhagraphbackwardcache，分别存储 graph_and_tensors 和 graph_and_tensors_backward 类型的数据
thread_local MHAGraphCache<graph_and_tensors, MHACacheKeyWrapper> mhagraphcache;
thread_local MHAGraphCache<graph_and_tensors_backward, MHACacheKeyWrapper> mhagraphbackwardcache;

// 定义 build_graph_and_tensors 函数，接受多个参数用于构建图和张量
auto build_graph_and_tensors(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset,
    cudnnHandle_t& handle,
    MHAParams& params) {
  
  // 设置数据类型为 HALF 如果输入张量 q 的标量类型为 kBFloat16
  auto dtype = fe::DataType_t::HALF;
  if (q.scalar_type() == kBFloat16) {
    // scaled_dot_product_flash_attention_options.set_alibi_mask(true);
  }

  // 创建名为 seq_q 和 seq_kv 的张量对象，其维度和数据类型已设置好
  auto seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Seq_q")
                                     .set_dim({b, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));
  auto seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("Seq_kv")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));

  // 如果 cudnn 版本大于等于 8903，则设置 scaled_dot_product_flash_attention_options 的偏置和填充掩码等参数
  // scaled_dot_product_flash_attention_options.set_bias(bias)
  //     .set_padding_mask(true)
  //     .set_seq_len_q(seq_q)
  //     .set_seq_len_kv(seq_kv);

  // 调用 mha_graph->sdpa 方法，传入 Q、K、V 和 scaled_dot_product_flash_attention_options，返回 O 和 Stats
  auto [O, Stats] =
      mha_graph->sdpa(Q, K, V, scaled_dot_product_flash_attention_options);

  // 设置 O 的输出属性，维度和步长与张量 o 的相同
  O->set_output(true)
      .set_dim(std::vector<int64_t>(
          o.sizes().data(), o.sizes().data() + o.sizes().size()))
      .set_stride(std::vector<int64_t>(
          o.strides().data(), o.strides().data() + o.strides().size()));

  // 如果 Stats 存在，则执行以下代码
  if (Stats) {
    // 设置输出为真，并设置数据类型为浮点数
    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
  }

  // 使用 AT_CUDNN_FRONTEND_CHECK 宏验证注意力机制图的有效性
  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  // 使用 AT_CUDNN_FRONTEND_CHECK 宏构建操作图
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  // 使用 AT_CUDNN_FRONTEND_CHECK 宏创建执行计划，使用启发模式 A
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  // 使用 AT_CUDNN_FRONTEND_CHECK 宏检查硬件支持
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  // 使用 AT_CUDNN_FRONTEND_CHECK 宏构建计划
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));

  // 返回一个包含多个移动语义值的元组
  return std::make_tuple(
      std::move(mha_graph),
      std::move(Q),
      std::move(K),
      std::move(V),
      std::move(attn_scale),
      std::move(seed),
      std::move(offset),
      std::move(O),
      std::move(Stats));
}



auto build_graph_and_tensors_backward(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset,
    cudnnHandle_t& handle,
    MHAParams& params) {
  // 设置数据类型为半精度
  auto dtype = fe::DataType_t::HALF;
  // 如果输入张量 q 的数据类型为 kBFloat16，则设置 SDPA 后向选项的 dropout
  if (q.scalar_type() == kBFloat16) {
    sdpa_backward_options.set_dropout(dropout_probability, Seed, Offset);
  }
  // 调用 SDPA 后向传播方法，计算并返回梯度张量 DQ, DK, DV
  auto [DQ, DK, DV] =
      mha_graph->sdpa_backward(Q, K, V, O, DO, STATS, sdpa_backward_options);
  // 设置 DQ 张量为输出，并复制其尺寸和步幅
  DQ->set_output(true)
      .set_dim(std::vector<int64_t>(dQ.sizes().begin(), dQ.sizes().end()))
      .set_stride(
          std::vector<int64_t>(dQ.strides().begin(), dQ.strides().end()));
  // 设置 DK 张量为输出，并复制其尺寸和步幅
  DK->set_output(true)
      .set_dim(std::vector<int64_t>(dK.sizes().begin(), dK.sizes().end()))
      .set_stride(
          std::vector<int64_t>(dK.strides().begin(), dK.strides().end()));
  // 设置 DV 张量为输出，并复制其尺寸和步幅
  DV->set_output(true)
      .set_dim(std::vector<int64_t>(dV.sizes().begin(), dV.sizes().end()))
      .set_stride(
          std::vector<int64_t>(dV.strides().begin(), dV.strides().end()));
  // 检查 MHA 图的有效性
  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  // 构建 MHA 图的操作图
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  // 创建 MHA 图的执行计划
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  // 检查 MHA 图是否支持当前的硬件环境
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  // 构建 MHA 图的计划
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));
  // 返回构建好的 MHA 图、输入张量和各种输出张量的元组
  return std::make_tuple(
      std::move(mha_graph),
      std::move(Q),
      std::move(K),
      std::move(V),
      std::move(attn_scale),
      std::move(Seed),
      std::move(Offset),
      std::move(O),
      std::move(DO),
      std::move(STATS),
      std::move(DQ),
      std::move(DK),
      std::move(DV));
}

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  // 获取 cuDNN 句柄
  cudnnHandle_t handle = getCudnnHandle();
  // 创建一个空的输出张量 o，设定其形状和步幅
  o = at::empty_strided(
      {b, h, s_q, d}, {s_q * h * d, d, h * d, 1}, q.options());
  // 如果需要返回 softmax 的统计信息，则创建相应形状的空张量 softmaxstats
  if (return_softmaxstats) {
    // TODO(eqy): verify that this is correct
    softmaxstats = at::empty({b, h, s_q}, q.options().dtype(kFloat));
  }

  // 创建 MHACacheKeyWrapper 对象作为缓存的键
  auto key = MHACacheKeyWrapper(
      b,
      h,
      s_q,
      s_kv,
      d,
      q,
      k,
      v,
      dropout_probability,
      is_causal,
      return_softmaxstats);
  // 查找缓存中是否已经存在对应的图和张量组合
  auto graph_and_tensors_ptr = mhagraphcache.find(key);
  // 定义图和张量值的结构
  graph_and_tensors graph_and_tensors_values;
  // 如果找到了对应的图和张量组合，则使用它们
  if (graph_and_tensors_ptr) {
    // 将 graph_and_tensors_ptr 指向的值解引用给 graph_and_tensors_values
    graph_and_tensors_values = *graph_and_tensors_ptr;
  } else {
    // 如果 graph_and_tensors_ptr 为 nullptr，则调用 build_graph_and_tensors 函数构建图和张量
    graph_and_tensors_values = build_graph_and_tensors(
        b,                              // 参数 b
        h,                              // 参数 h
        s_q,                            // 参数 s_q
        s_kv,                           // 参数 s_kv
        d,                              // 参数 d
        scaling_factor,                 // 参数 scaling_factor
        return_softmaxstats,            // 参数 return_softmaxstats
        is_causal,                      // 参数 is_causal
        dropout_probability,            // 参数 dropout_probability
        q,                              // 参数 q
        k,                              // 参数 k
        v,                              // 参数 v
        softmaxstats,                   // 参数 softmaxstats
        o,                              // 参数 o
        dropoutseed,                    // 参数 dropoutseed
        dropoutoffset,                  // 参数 dropoutoffset
        handle,                         // 参数 handle
        key.pod);                       // 参数 key.pod
  }
  // 将 graph_and_tensors_values 解构到多个变量中
  auto [mha_graph, Q, K, V, attn_scale, seed, offset, O, Stats] =
      graph_and_tensors_values;
  // 创建一个无序映射，将 Tensor_attributes 指针映射到对应的数据指针
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack = {
          {Q, q.data_ptr()},                // Q 对应的数据指针
          {K, k.data_ptr()},                // K 对应的数据指针
          {V, v.data_ptr()},                // V 对应的数据指针
          {attn_scale, &scaling_factor},    // attn_scale 对应的 scaling_factor 的地址
          //{bias, bias.data_ptr()},       // 注释掉的一行代码，忽略不处理
          {seed, dropoutseed.data_ptr()},   // seed 对应的 dropoutseed 的数据指针
          {offset, dropoutoffset.data_ptr()}, // offset 对应的 dropoutoffset 的数据指针
          {O, o.data_ptr()}                 // O 对应的 o 的数据指针
      };
  // 如果 return_softmaxstats 为真，则将 Stats 对应的 softmaxstats 数据指针加入映射
  if (return_softmaxstats) {
    variant_pack[Stats] = softmaxstats.data_ptr();
  }
  // 获取 mha_graph 的工作空间大小
  auto workspace_size = mha_graph->get_workspace_size();
  // 分配 CUDA 工作空间内存，并将其指针存储在 workspace_ptr 中
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  // 执行 mha_graph 的计算，使用 handle、variant_pack 和 workspace_ptr
  TORCH_CHECK(
      mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());
  // 使用更新后的 graph_and_tensors_values 更新 mhagraphcache 中的缓存项
  mhagraphcache.update(key, graph_and_tensors_values);
void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  // 获取 cuDNN 句柄
  cudnnHandle_t handle = getCudnnHandle();
  // 创建 MHA 缓存键
  auto key = MHACacheKeyWrapper(
      b, h, s_q, s_kv, d, q, k, v, dropout_probability, is_causal, true);
  // 尝试从 MHA 后向图缓存中查找缓存结果
  auto graph_and_tensors_backward_ptr = mhagraphbackwardcache.find(key);
  graph_and_tensors_backward graph_and_tensors_backward_values;
  // 如果找到缓存，使用缓存结果；否则构建后向图并保存到缓存
  if (graph_and_tensors_backward_ptr) {
    graph_and_tensors_backward_values = *graph_and_tensors_backward_ptr;
  } else {
    graph_and_tensors_backward_values = build_graph_and_tensors_backward(
        b,
        h,
        s_q,
        s_kv,
        d,
        scaling_factor,
        is_causal,
        dropout_probability,
        q,
        k,
        v,
        o,
        dO,
        softmaxstats,
        dQ,
        dK,
        dV,
        dropoutseed,
        dropoutoffset,
        handle,
        key.pod);
  }
  // 解包后向图及相关张量
  auto
      [mha_graph, Q, K, V, attn_scale, Seed, Offset, O, Do, Stats, Dq, Dk, Dv] =
          graph_and_tensors_backward_values;
  // 准备传递给 MHA 图执行的变量包
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack = {// inputs
                      {Q, q.data_ptr()},
                      {K, k.data_ptr()},
                      {V, v.data_ptr()},
                      {O, o.data_ptr()},
                      {Do, dO.data_ptr()},
                      {Stats, softmaxstats.data_ptr()},
                      // outputs
                      {Dq, dQ.data_ptr()},
                      {Dk, dK.data_ptr()},
                      {Dv, dV.data_ptr()},
                      // pass by value
                      {attn_scale, &scaling_factor}};
  // 如果 dropout 概率不为零，添加额外的输入数据
  if (dropout_probability != 0.0f) {
    variant_pack[Seed] = dropoutseed.data_ptr();
    variant_pack[Offset] = dropoutoffset.data_ptr();
  }
  // 获取需要的工作空间大小并分配工作空间
  auto workspace_size = mha_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  // 检查工作空间分配情况
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  // 执行 MHA 图计算，并检查执行结果
  TORCH_CHECK(
      mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());
  // 更新 MHA 后向图缓存
  mhagraphbackwardcache.update(key, graph_and_tensors_backward_values);
}
```