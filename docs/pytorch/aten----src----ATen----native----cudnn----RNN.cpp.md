# `.\pytorch\aten\src\ATen\native\cudnn\RNN.cpp`

```py
#ifndef AT_PER_OPERATOR_HEADERS
// 如果没有定义 AT_PER_OPERATOR_HEADERS，则包含标准的 ATen 和 NativeFunctions 头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 否则，包含特定于每个运算符的头文件，用于 CUDA 和 cuDNN 相关操作
#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/_cudnn_init_dropout_state_native.h>
#include <ATen/ops/_cudnn_rnn.h>
#include <ATen/ops/_cudnn_rnn_backward_native.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_native.h>
#include <ATen/ops/_cudnn_rnn_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#if !AT_CUDNN_ENABLED()
// 如果未启用 cuDNN 支持，则定义在 at::native 命名空间下的相关函数

namespace at {
namespace native {

// 定义 _cudnn_rnn_flatten_weight 函数，在缺少 cuDNN 支持时抛出错误
Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    bool fn_bidirectional) {
  AT_ERROR("_cudnn_rnn_flatten_weight: ATen not compiled with cuDNN support");
}

// 定义 _cudnn_rnn 函数，在缺少 cuDNN 支持时抛出错误
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight,
    int64_t weight_stride0,
    const std::optional<Tensor>& weight_buf_r_opt,
    const Tensor& hx,
    const std::optional<Tensor>& cx_opt,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const std::optional<Tensor>& fn_dropout_state_opt) {
  AT_ERROR("_cudnn_rnn: ATen not compiled with cuDNN support");
}

// 定义 _cudnn_rnn_backward 函数，在缺少 cuDNN 支持时抛出错误
std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input,
    TensorList weight,
    int64_t weight_stride0,
    const Tensor& weight_buf,
    const Tensor& hx,
    const std::optional<Tensor>& cx_opt,
    const Tensor& output,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    IntArrayRef batch_sizes,
    const std::optional<Tensor>& dropout_state_opt,
    const Tensor& reserve,
    std::array<bool, 4> output_mask) {
  AT_ERROR("_cudnn_rnn_backward: ATen not compiled with cuDNN support");
}

// 定义 _cudnn_init_dropout_state 函数，在缺少 cuDNN 支持时抛出错误
Tensor _cudnn_init_dropout_state(
    double dropout,
    bool train,
    int64_t dropout_seed,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  AT_ERROR("_cudnn_init_dropout_state: ATen not compiled with cuDNN support");
}
    // 初始化一个 TensorOptions 对象，并设置其属性：数据类型 dtype、布局 layout、设备 device，
    // 以及是否使用 pinned memory（根据 pin_memory 的值设定）。
    // 这里的调用看起来像是为了构造一个 TensorOptions 对象，但却没有将其赋给任何变量或使用它。
    TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
        pin_memory);
    
    // 抛出一个 AT_ERROR 异常，表示 ATen 没有使用 cuDNN 支持进行编译。
    AT_ERROR("_cudnn_init_dropout_state: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/RNNUtils.h>

namespace at {
namespace native {

namespace {
// DropoutDescriptor

// 定义一个结构体，用于存储dropout的相关参数和状态
struct DropoutDescriptorParams {
  bool train;                   // 是否处于训练状态
  double dropout;               // dropout的比例
  Tensor dropout_state;         // dropout状态的张量
  DropoutDescriptorParams() = default;  // 默认构造函数

  // 设置dropout相关参数和状态
  void set(bool train_, double dropout_, Tensor dropout_state_) {
    train = train_;
    dropout = dropout_;
    dropout_state = dropout_state_;
  }

  // 根据当前参数生成对应的dropout描述符
  DropoutDescriptor descriptor(cudnnHandle_t handle) const {
    auto dropout_p = train ? dropout : 0;  // 计算当前的dropout比例
    DropoutDescriptor dropout_desc;

    if (dropout_p == 0) {
      dropout_desc.set_no_dropout(handle);  // 如果dropout比例为0，则设置为无dropout状态
    } else {
      dropout_desc.set(handle, dropout_p, dropout_state);  // 否则设置具体的dropout状态
    }

    return dropout_desc;  // 返回生成的dropout描述符
  }
};

// RNNDescriptor

// 定义一个结构体，用于存储RNN的相关参数
struct RNNDescriptorParams {
#ifdef USE_CUDNN_RNN_V8_API
  int64_t input_size;           // 输入大小
  bool packed;                  // 是否压缩
#endif
  int64_t hidden_size;          // 隐藏层大小
  int64_t proj_size;            // 投影大小
  int64_t num_layers;           // 层数
  cudnnDirectionMode_t bidirectional;  // 双向模式
  cudnnRNNMode_t mode;          // RNN模式
  cudnnDataType_t datatype;     // 数据类型
  cudnnDataType_t input_datatype;  // 输入数据类型
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;  // RNN算法
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;  // 输入模式为线性输入

  // 返回RNN的方向数（单向或双向）
  int64_t num_directions() const {
    return bidirectional ? 2 : 1;
  }

  // 设置RNN的工作模式
  void set_mode(int64_t fn_mode) {
    switch (fn_mode) {
      case CUDNN_RNN_RELU:
        mode = CUDNN_RNN_RELU;
        break;
      case CUDNN_RNN_TANH:
        mode = CUDNN_RNN_TANH;
        break;
      case CUDNN_LSTM:
        mode = CUDNN_LSTM;
        break;
      case CUDNN_GRU:
        mode = CUDNN_GRU;
        break;
      default: {
        std::ostringstream oss;
        oss << "unrecognized cuDNN RNN mode " << fn_mode;
        AT_ERROR(oss.str());  // 如果模式未被识别，则抛出错误
      }
    }
  }

  // 设置RNN的双向模式
  void set_bidirectional(bool fn_bidirectional) {
    bidirectional =
        fn_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;  // 根据传入的参数设置双向或单向模式
  }

  // 设置RNN的算法
  void set_algo(cudnnRNNAlgo_t algo) {
    this->algo = algo;
  }

#ifndef USE_CUDNN_RNN_V8_API
  // 设置RNN的参数（不使用CUDNN RNN V8 API）
  void set(
      int64_t mode,
      int64_t hidden_size,
      int64_t proj_size,
      int64_t num_layers,
      bool bidirectional,
      cudnnDataType_t datatype,
      cudnnDataType_t input_datatype){
#else
  // 设置RNN的参数（使用CUDNN RNN V8 API）
  void set(
      int64_t mode,
      int64_t input_size,
      bool packed,
      int64_t hidden_size,
      int64_t proj_size,
      int64_t num_layers,
      bool bidirectional,
      cudnnDataType_t datatype,
      cudnnDataType_t input_datatype) {
#endif
      this->set_mode(mode);  // 设置RNN的工作模式

#ifdef USE_CUDNN_RNN_V8_API
  this->input_size = input_size;  // 设置输入大小
  this->packed = packed;          // 设置是否压缩
#endif
  this->hidden_size = hidden_size;    // 设置隐藏层大小
  this->proj_size = proj_size;        // 设置投影大小
  this->num_layers = num_layers;      // 设置层数
  this->set_bidirectional(bidirectional);  // 设置双向模式
  this->datatype = datatype;          // 设置数据类型
  this->input_datatype = input_datatype;  // 设置输入数据类型
}

// 根据当前参数生成对应的RNN描述符
RNNDescriptor
descriptor(cudnnHandle_t handle, DropoutDescriptor&& dropout_desc) const {
  RNNDescriptor rnn_desc;
  // 这里应该有更多的实现代码，但截止到此处没有提供足够的信息来完成注释。
  // 应该继续编写代码以完善生成RNN描述符的过程。
// 如果未定义宏 USE_CUDNN_RNN_V8_API，则执行以下代码块
#ifndef USE_CUDNN_RNN_V8_API
// 设置 RNN 描述符，初始化 RNN 模型参数
std::vector<TensorDescriptor> rnn_descriptor_sequence(
    const Tensor& tensor,
    IntArrayRef batch_sizes) {
  // 创建描述符向量，其大小与批处理大小相同
  std::vector<TensorDescriptor> descriptors(batch_sizes.size());
  // 初始化循环变量 i
  size_t i = 0;
  // 获取张量的批次大小
  auto batch_tensor_size = tensor.sizes().vec();
  // 遍历批次大小向量
  for (auto batch_size : batch_sizes) {
    // 更新张量大小为当前批次大小
    batch_tensor_size[0] = batch_size;
    // cuDNN RNN API 不支持 2D 描述符，因此需要将其填充为 3D
    // 设置张量描述符，包括数据类型、大小、步幅和维度
    descriptors[i].set(
        getCudnnDataType(tensor), batch_tensor_size, tensor.strides(), 3);
    // 更新描述符索引
    i++;
  }
  // 返回描述符向量
  return descriptors;
}

// 设置 RNN 描述符向量，用于每个张量和指定数量的张量
std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
  // 创建描述符向量，其大小为 N
  std::vector<TensorDescriptor> descriptors(N);
  // 遍历描述符向量
  for (const auto i : c10::irange(N)) {
    // 设置张量描述符，包括张量和一个指定的维度
    descriptors[i].set(tensor, 5);
  }
  // 返回描述符向量
  return descriptors;
}

// 如果定义了宏 USE_CUDNN_RNN_V8_API，则执行以下代码块
#else
// 设置 RNN 描述符序列，适用于紧凑情况
auto rnn_descriptor_sequence(
    const Tensor& tensor,
    uint32_t batch_size,
    const IntArrayRef batch_sizes,
    uint32_t seq_len,
    uint32_t vector_size) { // packed case
  // 创建 RNN 数据描述符
  RNNDataDescriptor r;
  // 创建序列长度数组，其大小为 batch_size，每个元素初始化为 1
  std::vector<int> seqLengthArray(batch_size, 1);
  // cuDNN 要求紧凑批次的序列长度看起来像是展开的，例如对于下面的情况：
  // Sequence 1: ABCD
  // Sequence 2: EF
  // Sequence 3: G
  // 应生成数组 [4, 2, 1]，长度等于 mini_batch
  // TODO(eqy): 可能有比 O(SN) 更智能的方法来做这件事
  // 遍历批次大小数组
  for (auto it = batch_sizes.begin(); it != batch_sizes.end(); it++) {
    // 每个批次的序列长度从 1 开始，因此跳过第一次迭代
    if (it == batch_sizes.begin()) {
      continue;
    }
    // 对于迭代器 `it` 指向的元素，使用 `c10::irange` 的结果进行循环遍历
    for (const auto idx : c10::irange(*it)) {
      // 增加 `seqLengthArray` 数组中索引 `idx` 处的计数
      seqLengthArray[idx]++;
    }
  }
  // 设置 RNN 描述符 `r`，使用序列为主要布局的紧凑形式打包数据
  r.set(
      tensor,
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
      seq_len,
      batch_size,
      vector_size,
      // 将 `seqLengthArray` 的数据作为序列长度数组传递给 `r`
      seqLengthArray.data());
  // 返回设置后的 RNN 描述符 `r`
  return r;
// 结构体描述了用于 RNN 输入的张量描述符的参数
struct TensorDescriptorListParams {
  IntArrayRef batch_sizes;  // 批次大小数组，描述每个序列的批次大小
  int64_t seq_length;       // 序列长度，表示每个序列的时间步数
  int64_t mini_batch;       // 小批量大小，表示序列的数量
  // 注意：这里的 input_size 并不是 input_sizes 的大小，而是最内层维度的大小。
  // 在自然语言处理中，通常表示嵌入的大小，也可以理解为“通道”维度的大小（尽管这可能会误导视觉研究人员 :)）
  int64_t input_size;       
  // 仅在非打包输入时有效
  int64_t batch_sizes_sum;  // 批次大小之和，等于 batch_sizes 的总和

  // 检查输入是否已打包的方法
  bool is_input_packed() const {
    return batch_sizes.size() != 0;
  }

  // 设置方法，初始化 TensorDescriptorListParams 的各个字段
  void set(
      IntArrayRef input_sizes,   // 输入尺寸数组，描述每个序列的输入尺寸
      IntArrayRef batch_sizes_,  // 批次大小数组，描述每个序列的批次大小
      bool batch_first) {
    batch_sizes = batch_sizes_;  // 设置批次大小数组
    // 检查输入是否已经打包
    if (is_input_packed()) {
      // 如果是打包的输入，序列长度等于批次大小数组的长度
      seq_length = batch_sizes.size();
      // 小批次大小等于批次大小数组的第一个元素
      mini_batch = batch_sizes[0];
      // 当输入已经打包时，批次大小的总和为输入尺寸的第一个元素
      // 注意：当输入已经打包时，mini_batch 的大小并不是外部维度的大小
      batch_sizes_sum = input_sizes[0];
      // 输入的特征大小为输入尺寸的第二个元素
      input_size = input_sizes[1];
    } else {
      // 如果输入没有打包
      if (batch_first) {
        // 如果首批次是批次优先，则序列长度为输入尺寸的第二个元素
        seq_length = input_sizes[1];
        // 小批次大小为输入尺寸的第一个元素
        mini_batch = input_sizes[0];
      } else {
        // 如果不是批次优先，则序列长度为输入尺寸的第一个元素
        seq_length = input_sizes[0];
        // 小批次大小为输入尺寸的第二个元素
        mini_batch = input_sizes[1];
      }
      // 输入的特征大小为输入尺寸的第三个元素
      input_size = input_sizes[2];
      // TODO: 实际上，这样做会让 ASAN（地址安全性分析器）在捕获未初始化访问时更困难吗？
      // 在我们访问它时，将 batch_sizes_sum 设为一个无效的值
      batch_sizes_sum = -1; // 如果访问它，那么它是一个错误的值
    }
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则定义以下函数
  // 获取描述符的函数，根据输入张量 x 判断是否需要打包
  std::vector<TensorDescriptor> descriptors(Tensor x) const {
    // 检查输入张量 x 是否已经打包
    auto is_input_packed = batch_sizes.size() != 0;
    // 如果已打包，则返回序列化的 RNN 描述符
    if (is_input_packed) {
      return rnn_descriptor_sequence(x, batch_sizes);
    } else {
      // 否则，返回单个时间步的 RNN 描述符
      return rnn_descriptor(x[0], seq_length);
    }
  }
#else
  // 如果定义了使用 CUDNN RNN V8 API，则定义以下函数
  auto descriptors(Tensor x) const {
    // 检查输入张量 x 是否已经打包
    auto is_input_packed = batch_sizes.size() != 0;
    // 如果已打包，则返回序列化的 RNN 描述符，带有额外的参数
    if (is_input_packed) {
      return rnn_descriptor_sequence(
          x, mini_batch, batch_sizes, seq_length, x.size(-1));
    } else {
      // 否则，返回单个时间步的 RNN 描述符，带有额外的参数
      return rnn_descriptor(x, mini_batch, seq_length, x.size(-1));
    }
  }
#endif
};

// Everything together

// RNN 参数的结构体，包括 dropout 参数、RNN 描述符参数和张量描述符列表参数
struct RNNParams {
  DropoutDescriptorParams dropout;
  RNNDescriptorParams rnn;
  TensorDescriptorListParams tensors;
};

// 注意：不包括权重描述符
// RNN 描述符的结构体，包括 RNN 描述符本身和各种张量描述符
struct RNNDescriptors {
  RNNDescriptor rnn_desc;  // RNN 描述符
  // 注意：这里实际上不会正确布置张量描述符指针，因此需要预处理它们

#ifndef USE_CUDNN_RNN_V8_API
  std::vector<TensorDescriptor> x_descs;  // 输入张量 x 的描述符列表
  std::vector<TensorDescriptor> y_descs;  // 输出张量 y 的描述符列表
#else
  RNNDataDescriptor x_descs;  // 输入张量 x 的描述符
  RNNDataDescriptor y_descs;  // 输出张量 y 的描述符
#endif
  TensorDescriptor hx_desc;   // 隐藏状态的描述符
  TensorDescriptor hy_desc;   // 输出隐藏状态的描述符
  TensorDescriptor cx_desc;   // 单元状态的描述符
  TensorDescriptor cy_desc;   // 输出单元状态的描述符

  // 构造函数，初始化 RNN 描述符及其相关描述符
  RNNDescriptors(
      const RNNParams& fn,
      cudnnHandle_t handle,
      Tensor x,
      Tensor y,
      Tensor hx,
      Tensor cx) {
    // 根据给定的 RNN 参数和句柄，初始化 RNN 描述符
    rnn_desc = fn.rnn.descriptor(handle, fn.dropout.descriptor(handle));
    // 初始化输入张量 x 和输出张量 y 的描述符列表
    x_descs = fn.tensors.descriptors(x);
    y_descs = fn.tensors.descriptors(y);
    // 设置隐藏状态的描述符
    hx_desc.set(hx, 5);
    hy_desc.set(hx, 5);
    // 如果存在单元状态张量，则设置其描述符
    if (cx.defined()) {
      cx_desc.set(cx, 5);
      cy_desc.set(cx, 5);
    }
  }

  // TODO: This is annoying, having to put the cudnnTensorDescriptor_t
  // in a contiguous array...
  // 获取描述符的函数，将张量描述符列表转换为 cudnnTensorDescriptor_t 的向量
  std::vector<cudnnTensorDescriptor_t> get_descs(
      const std::vector<TensorDescriptor>& descs) {
    std::vector<cudnnTensorDescriptor_t> r;
    r.reserve(descs.size());
    // 遍历给定的张量描述符列表，将其转换为 cudnnTensorDescriptor_t 类型并存储在向量中
    for (auto& desc : descs) {
      r.emplace_back(desc.desc());
    }
    return r;
  }

#ifndef USE_CUDNN_RNN_V8_API
  // 获取输入张量 x 的描述符的函数
  std::vector<cudnnTensorDescriptor_t> get_x_descs() {
    return get_descs(x_descs);
  }

  // 获取输出张量 y 的描述符的函数
  std::vector<cudnnTensorDescriptor_t> get_y_descs() {
    return get_descs(y_descs);
  }
#endif
};

// 获取权重数量的函数
int64_t get_num_weights(
    cudnnHandle_t handle,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
#endif
    cudnnDataType_t datatype) {
  size_t weight_size;
  // 如果未定义使用 CUDNN RNN V8 API，则获取 RNN 参数大小
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnGetRNNParamsSize(
      handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
#else
  // 否则，获取 RNN 权重空间大小
  AT_CUDNN_CHECK(
      cudnnGetRNNWeightSpaceSize(handle, rnn_desc.desc(), &weight_size));

#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则获取 RNN 参数大小
  AT_CUDNN_CHECK(cudnnGetRNNParamsSize(
      handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
#else
  // 否则，获取 RNN 权重空间大小
  AT_CUDNN_CHECK(
      cudnnGetRNNWeightSpaceSize(handle, rnn_desc.desc(), &weight_size));
#endif
#endif
  // 计算元素大小，用于确定权重大小是否合理
  auto elem_size = dataSize(datatype);
  // 断言权重大小应为元素大小的整数倍，否则输出错误信息
  TORCH_INTERNAL_ASSERT(
      weight_size % elem_size == 0,
      "cudnnGetRNNParamsSize returned nonsensical weight_size");
  // 返回权重大小除以元素大小的结果作为函数返回值
  return weight_size / elem_size;
}

// 根据 cuDNN RNN 模式返回线性层的数量
int64_t _num_linear_layers(cudnnRNNMode_t mode) {
  switch (mode) {
    case CUDNN_LSTM:
      return 8;
    case CUDNN_GRU:
      return 6;
    case CUDNN_RNN_RELU:
      return 2;
    case CUDNN_RNN_TANH:
      return 2;
    default:
      // 如果模式未知，则输出错误信息
      AT_ERROR("unknown cuDNN RNN mode ", mode);
  }
}

// 添加投影权重
void add_projection_weights(
    cudnnHandle_t handle,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
    const FilterDescriptor& w_desc,
#endif
    const Tensor& weight_buf,
    int64_t layer,
    std::vector<Tensor>& params) {
  void* matrix_pointer = nullptr;
  // 假设为 LSTM，其中有 8 个线性层（即 4 个权重和 4 个偏置）
  int64_t linear_id = 8;
#ifndef USE_CUDNN_RNN_V8_API
  // 获取 RNN 线性层矩阵参数描述符
  FilterDescriptor lin_layer_mat_desc;
  AT_CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
      /*handle=*/handle,
      /*rnnDesc=*/rnn_desc.desc(),
      /*layer=*/layer,
      /*xDesc=*/x_desc.desc(),
      /*wDesc=*/w_desc.desc(),
      /*w=*/weight_buf.data_ptr(),
      /*linLayerID=*/linear_id,
      /*linLayerMatDesc=*/lin_layer_mat_desc.mut_desc(),
      /*linLayerMat=*/&matrix_pointer));
#else
  // 获取 RNN 权重参数描述符
  TensorDescriptor lin_layer_mat_desc;
  AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
      /*handle=*/handle,
      /*rnnDesc=*/rnn_desc.desc(),
      /*layer=*/layer,
      /*wDesc=*/weight_buf.numel() * weight_buf.element_size(),
      /*w=*/weight_buf.data_ptr(),
      /*linLayerID=*/linear_id,
      /*linLayerMatDesc=*/lin_layer_mat_desc.mut_desc(),
      /*linLayerMat=*/&matrix_pointer,
      nullptr,
      nullptr));
#endif

  cudnnDataType_t data_type;
#ifndef USE_CUDNN_RNN_V8_API
  cudnnTensorFormat_t format;
#else
  int stride_dim_a[5];
#endif
  int nb_dims;
  constexpr int min_dim = 3;
  int filter_dim_a[min_dim];
#ifndef USE_CUDNN_RNN_V8_API
  // 获取线性层矩阵参数的过滤器描述符
  AT_CUDNN_CHECK(cudnnGetFilterNdDescriptor(
      lin_layer_mat_desc.desc(),
      min_dim,
      &data_type,
      &format,
      &nb_dims,
      filter_dim_a));
#else
  // 获取张量的多维描述符
  AT_CUDNN_CHECK(cudnnGetTensorNdDescriptor(
      lin_layer_mat_desc.desc(),
      min_dim,
      &data_type,
      &nb_dims,
      filter_dim_a,
      stride_dim_a));
#endif

  // 检查维度数是否符合最小维度限制，用于断言
  TORCH_INTERNAL_ASSERT(
      nb_dims <= min_dim, "nb_dims = ", nb_dims, "; min_dim  = ", min_dim);
  // 获取元素的字节大小，根据权重缓冲区的 CuDNN 数据类型
  auto elem_size = dataSize(getCudnnDataType(weight_buf));
  // 计算偏移量的字节数，将矩阵指针和权重缓冲区的数据指针进行偏移量计算
  auto offset_bytes = (char*)matrix_pointer - (char*)weight_buf.data_ptr();
  // 断言偏移量字节数应该是元素大小的整数倍
  TORCH_INTERNAL_ASSERT(
      offset_bytes % elem_size == 0,
      "offset_bytes = ",
      offset_bytes,
      "; elem_size = ",
      elem_size);
  // 计算偏移量，即偏移量字节数除以元素大小
  size_t offset = offset_bytes / elem_size;

  // 计算矩阵元素的总数，即所有维度的乘积
  int mat_numel = c10::multiply_integers(filter_dim_a, filter_dim_a + nb_dims);
  // 生成一个新的参数张量，是对权重缓冲区的视图
  std::initializer_list<int64_t> size = {mat_numel, 1};
  Tensor param = at::empty({0}, weight_buf.options())
                     .set_(weight_buf.storage(), offset, size);
  // 将参数张量添加到参数向量中
  params.emplace_back(std::move(param));
}

/*
  返回每个 RNN 层的权重和偏置张量。这些张量是 CuDNN 分配的权重缓冲区的视图。

  注意：对于 LSTM 和 GRU，它们每种类型有多个参数（分别为 4 个和 3 个），
        这些参数沿着第一个维度连接起来。
        这些参数按照 CuDNN 返回的顺序保持一致：
            LSTM 为 (reset, forget, cell, output)
            GRU 为 (reset, input, new)
  参数：
      fn: 包含 RNN 状态的 RNN 函数对象
      handle: CuDNN 句柄
      weight_buf: 包含 CuDNN 分配的权重（或梯度权重）缓冲区的 1D 张量
      include_bias: 是否包括偏置，默认为 true
  返回：
      parameters: [(weight_ih, weight_hh, bias_ih, bias_hh)*] 的列表，长度为 num_layers
      这表示为一对向量和外部维度步幅（注意：无法返回 MatrixRef 因为我们需要分配底层张量）
*/
std::pair<std::vector<Tensor>, size_t> // stride0
get_parameters(
    cudnnHandle_t handle,
    const RNNDescriptorParams& rnn,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
    const FilterDescriptor& w_desc,
#endif
    const Tensor& weight_buf,
    bool include_bias = true) {
#ifndef USE_CUDNN_RNN_V8_API
  // 定义 CuDNN 方法集合，根据是否使用 V8 API 进行选择
  auto cudnn_methods = {
      cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams};
#else
  // 在 V8 API 下，直接选择 true 或 false，表示不同的 CuDNN 方法
  auto cudnn_methods = {true, false};
#endif
  // 存储参数张量的向量
  std::vector<Tensor> params;
  // 计算线性层的数量，根据 RNN 模式确定
  int64_t num_linear_layers = _num_linear_layers(rnn.mode);
  // 计算 RNN 层数，考虑方向和层数的乘积
  int64_t num_layers = rnn.num_directions() * rnn.num_layers;
  // 当前偏移量初始化为 0
  size_t cur_offset = 0;
  // 全局层参数计数器初始化为 0
  size_t global_layer_params_count = 0;
  // 对每个 RNN 层进行迭代
  for (const auto layer : c10::irange(num_layers)) {
    // 当前层参数计数器初始化为 0
    size_t layer_params_count = 0;
    // 遍历 CuDNN 方法集合
    for (auto cudnn_method : cudnn_methods) {
      // 遍历线性层的每个 ID
      for (const auto linear_id : c10::irange(num_linear_layers)) {
        // 矩阵指针的初始化
        void* matrix_pointer;
#ifndef USE_CUDNN_RNN_V8_API
        // 定义用于存储线性层矩阵描述符的对象
        FilterDescriptor lin_layer_mat_desc;
        // 调用 cudnn_method 函数，获取 RNN 层的权重参数
        AT_CUDNN_CHECK(cudnn_method(
            handle,
            rnn_desc.desc(),
            layer,
            x_desc.desc(),
            w_desc.desc(),
            weight_buf.data_ptr(),
            linear_id,
            // 获取线性层矩阵描述符的可变引用
            lin_layer_mat_desc.mut_desc(),
            &matrix_pointer));
#else
        // 定义用于存储张量描述符的对象
        TensorDescriptor lin_layer_mat_desc;
        // 循环执行以下操作，共执行 100 次
        for (int stateless = 0; stateless < 100; stateless++) {
          if (cudnn_method) { // 如果 cudnn_method 为真，表示处理矩阵
            // 获取 RNN 层的权重参数（矩阵部分）
            AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
                handle,
                rnn_desc.desc(),
                layer,
                weight_buf.numel() * weight_buf.element_size(),
                weight_buf.data_ptr(),
                linear_id,
                // 获取矩阵描述符的可变引用
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer,
                nullptr,
                nullptr));
          } else { // 如果 cudnn_method 为假，表示处理偏置
            // 获取 RNN 层的权重参数（偏置部分）
            AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
                handle,
                rnn_desc.desc(),
                layer,
                weight_buf.numel() * weight_buf.element_size(),
                weight_buf.data_ptr(),
                linear_id,
                nullptr,
                nullptr,
                // 获取矩阵描述符的可变引用
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer));
          }
        }
#endif
        // 定义数据类型变量
        cudnnDataType_t data_type;
#ifndef USE_CUDNN_RNN_V8_API
        // 定义张量格式变量
        cudnnTensorFormat_t format;
#else
        // 定义数组，存储张量步长的维度信息
        int stride_dim_a[5];
#endif
        // 定义整数变量，存储张量的维度数
        int nb_dims;
        // 定义常量，表示最小的张量维度数
        constexpr int min_dim = 3;
        // 定义数组，存储滤波器维度信息
        int filter_dim_a[min_dim];
#ifndef USE_CUDNN_RNN_V8_API
        // 获取线性层矩阵描述符的滤波器多维度描述信息
        AT_CUDNN_CHECK(cudnnGetFilterNdDescriptor(
            lin_layer_mat_desc.desc(),
            min_dim,
            &data_type,
            &format,
            &nb_dims,
            filter_dim_a));
#else
        // 获取线性层矩阵描述符的张量多维度描述信息
        AT_CUDNN_CHECK(cudnnGetTensorNdDescriptor(
            lin_layer_mat_desc.desc(),
            min_dim,
            &data_type,
            &nb_dims,
            filter_dim_a,
            stride_dim_a));
#endif
// 与上一行的 `#endif` 配对，结束之前的条件编译块

        TORCH_INTERNAL_ASSERT(
            nb_dims <= min_dim,
            "nb_dims = ",
            nb_dims,
            "; min_dim  = ",
            min_dim);
        // 断言，确保 `nb_dims` 小于等于 `min_dim`

        auto elem_size = dataSize(getCudnnDataType(weight_buf));
        // 计算元素大小，根据 `weight_buf` 的 CUDA 数据类型确定

        auto offset_bytes =
            (char*)matrix_pointer - (char*)weight_buf.data_ptr();
        // 计算偏移字节数，基于 `matrix_pointer` 与 `weight_buf.data_ptr()` 的地址差

        TORCH_INTERNAL_ASSERT(
            offset_bytes % elem_size == 0,
            "offset_bytes = ",
            offset_bytes,
            "; elem_size = ",
            elem_size);
        // 断言，确保 `offset_bytes` 是 `elem_size` 的整数倍

        size_t offset = offset_bytes / elem_size;
        // 计算偏移量，即偏移字节数除以元素大小

        // for all the RNN types provided by CUDNN, all the ih weights
        // are the same size and are allocated in a contiguous chunk
        // (same for the hh weights, and the ih and hh biases).
        // Since we're storing all the weights in a single tensor anyway,
        // might as well merge the CUDNN ones into a single tensor as well
        // 对于 CUDNN 提供的所有 RNN 类型，所有的 ih 权重都是相同大小，并且分配在一个连续的块中
        // （hh 权重也是如此，ih 和 hh 偏置也是如此）。
        // 由于我们无论如何都将所有权重存储在单个张量中，
        // 因此最好将 CUDNN 的权重合并到一个张量中

        int mat_numel =
            c10::multiply_integers(filter_dim_a, filter_dim_a + nb_dims);
        // 计算矩阵元素个数，通过乘积运算计算维度参数之间的乘积

        if (linear_id == 0 || linear_id == num_linear_layers / 2) {
          // 如果 `linear_id` 是 0 或者是 `num_linear_layers` 的一半

          // We could also exclude bias params by restricting cudnn_methods to
          // just { cudnnGetRNNLinLayerMatrixParams } at the very top.  However,
          // to do so would throw off the cur_offset account, which is currently
          // a strict and informative check that all params are laid out the way
          // we think they are.  If include_bias is false, I'd rather keep full
          // cur_offset checks rather than save some CPU overhead by skipping
          // the cudnn_method = cudnnGetRNNLinLayerBiasParams iteration.
#ifndef USE_CUDNN_RNN_V8_API
          if (include_bias || cudnn_method != cudnnGetRNNLinLayerBiasParams) {
#else
          if (include_bias || cudnn_method) {
#endif
            // 如果包含偏置或者 `cudnn_method` 存在

            // Generate a new parameter tensor which is a view into the
            // weight_buf.
            // 生成一个新的参数张量，它是对 `weight_buf` 的视图
            std::initializer_list<int64_t> size = {
                mat_numel * num_linear_layers / 2, 1};
            // 初始化张量的大小

            Tensor param = at::empty({0}, weight_buf.options())
                               .set_(weight_buf.storage(), offset, size);
            // 创建一个空的张量 `param`，并将 `weight_buf` 的数据作为其存储，设置偏移和大小
            params.emplace_back(std::move(param));
            // 将 `param` 添加到 `params` 向量中
            layer_params_count++;
            // 增加层参数计数器
          }
        } else {
          TORCH_INTERNAL_ASSERT(
              cur_offset == offset,
              "cur_offset = ",
              cur_offset,
              "; offset = ",
              offset);
          // 断言，确保 `cur_offset` 等于 `offset`
        }
        cur_offset = offset + mat_numel;
        // 更新 `cur_offset` 为 `offset` 加上 `mat_numel`
      }
    } // for cudnn_method
    // 结束 `cudnn_method` 的循环

    if (rnn.proj_size != 0) {
      // 如果 RNN 投影大小不为 0

#ifndef USE_CUDNN_RNN_V8_API
      add_projection_weights(
          handle, rnn_desc, x_desc, w_desc, weight_buf, layer, params);
#else
      add_projection_weights(handle, rnn_desc, weight_buf, layer, params);
#endif
      // 添加投影权重，根据是否使用 CUDNN RNN V8 API 调用不同的函数

      layer_params_count++;
      // 增加层参数计数器
    }

    if (layer == 0) {
      // 如果是第一层

      global_layer_params_count = layer_params_count;
      // 设置全局层参数计数器为当前层参数计数器的值
    } else {
      // 使用 TORCH_INTERNAL_ASSERT 断言函数，确保 global_layer_params_count 等于 layer_params_count
      TORCH_INTERNAL_ASSERT(
          global_layer_params_count == layer_params_count,
          "global_layer_params_count = ",
          global_layer_params_count,
          "; layer_params_count = ",
          layer_params_count);
    }
  } // 结束对每一层的循环

  // 返回一个 std::pair 对象，包含 params（可能是一个结构体或容器）和 global_layer_params_count
  return std::make_pair(params, global_layer_params_count);
// 这是一个轻量级版本上述方法，用于快速获取预期参数偏移量。

std::vector<void*> get_expected_data_ptrs(
    const Tensor& weight_buf, // 权重缓冲区张量
    cudnnHandle_t handle, // cuDNN句柄
    const RNNDescriptorParams& rnn, // RNN描述参数
    const RNNDescriptor& rnn_desc, // RNN描述符
    const TensorDescriptor& x_desc, // 输入张量描述符
    cudnnDataType_t datatype) { // 数据类型
#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc; // 过滤器描述符用于权重缓冲区
  w_desc.set(weight_buf, 3); // 设置权重缓冲区描述符
#endif

  int64_t num_linear_layers = _num_linear_layers(rnn.mode); // 线性层数量
  int64_t num_dir_layers = rnn.num_directions() * rnn.num_layers; // 方向数乘以层数

#ifndef USE_CUDNN_RNN_V8_API
  const auto cudnn_methods = { // cuDNN方法集合
      cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams};
#else
  const auto cudnn_methods = {true, false}; // cuDNN V8 API方法标记
#endif

  std::vector<void*> data_ptrs; // 数据指针向量
  if (rnn.proj_size != 0) { // 如果投影大小不为0
    data_ptrs.reserve(num_dir_layers * (2 * 2 + 1)); // 预留足够空间
  } else {
    data_ptrs.reserve(num_dir_layers * 2 * 2); // 否则预留空间
  }

  for (const auto layer : c10::irange(num_dir_layers)) { // 对每一层循环
    for (auto cudnn_method : cudnn_methods) { // 对cuDNN方法循环
      // 此API返回每个门的权重的单独指针，但我们将它们表示为单个张量，
      // 因此我们只对可能值的非常有限子集感兴趣。
      const std::array<int64_t, 2> linear_offsets = {0, num_linear_layers / 2}; // 线性偏移量数组
      for (int64_t linear_id : linear_offsets) { // 对每个线性偏移量循环
        void* matrix_pointer; // 矩阵指针
#ifndef USE_CUDNN_RNN_V8_API
        FilterDescriptor lin_layer_mat_desc; // 线性层矩阵描述符
        AT_CUDNN_CHECK(cudnn_method(
            handle,
            rnn_desc.desc(),
            layer,
            x_desc.desc(),
            w_desc.desc(),
            weight_buf.data_ptr(),
            linear_id,
            lin_layer_mat_desc.mut_desc(),
            &matrix_pointer)); // 调用cuDNN方法获取矩阵参数
#else
        TensorDescriptor lin_layer_mat_desc; // 矩阵描述符
        if (cudnn_method) { // 如果是矩阵
          AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
              handle,
              rnn_desc.desc(),
              layer,
              weight_buf.numel() * weight_buf.element_size(),
              weight_buf.data_ptr(),
              linear_id,
              lin_layer_mat_desc.mut_desc(),
              &matrix_pointer,
              nullptr,
              nullptr));
        } else { // 如果是偏置
          AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
              handle,
              rnn_desc.desc(),
              layer,
              weight_buf.numel() * weight_buf.element_size(),
              weight_buf.data_ptr(),
              linear_id,
              nullptr,
              nullptr,
              lin_layer_mat_desc.mut_desc(),
              &matrix_pointer));
        }
#endif
        data_ptrs.push_back(matrix_pointer); // 将获取的矩阵指针存入数据指针向量
      }
    }
    if (rnn.proj_size != 0) { // 如果有投影大小
      // 假设是LSTM，有8个“线性层”（即4个权重和4个偏置）
      int64_t linear_id = 8; // 线性ID
      void* matrix_pointer; // 矩阵指针
#ifndef USE_CUDNN_RNN_V8_API
      // 定义描述滤波器的描述符对象
      FilterDescriptor lin_layer_mat_desc;
      // 获取 RNN 的线性层权重矩阵参数
      AT_CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
          handle,
          rnn_desc.desc(),
          layer,
          x_desc.desc(),
          w_desc.desc(),
          weight_buf.data_ptr(),
          linear_id,
          lin_layer_mat_desc.mut_desc(),
          &matrix_pointer));
#else
      // 定义张量描述符对象
      TensorDescriptor lin_layer_mat_desc;
      // 获取 RNN 的权重参数
      AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
          handle,
          rnn_desc.desc(),
          layer,
          weight_buf.numel() * weight_buf.element_size(),
          weight_buf.data_ptr(),
          linear_id,
          lin_layer_mat_desc.mut_desc(),
          &matrix_pointer,
          nullptr,
          nullptr));
#endif
      // 将获取到的权重数据指针添加到数据指针列表中
      data_ptrs.push_back(matrix_pointer);
    }
  }
  // 返回包含权重数据指针的列表
  return data_ptrs;
}

void _viewOrCopyOneParam(
    const Tensor& param_from,
    const Tensor& param_to,
    bool copy,
    bool allow_type_change = false) {
  // 如果执行复制操作，允许类型改变为 true 或 false
  // 如果执行查看操作，类型改变必须为 false
  TORCH_INTERNAL_ASSERT(
      copy || !allow_type_change, "if viewing, type change is not allowed.");
  // 确保参数类型匹配，如果允许类型改变，则忽略类型不匹配
  TORCH_INTERNAL_ASSERT(
      allow_type_change || (param_from.scalar_type() == param_to.scalar_type()),
      "parameter types mismatch");
  if (copy) {
    // 如果进行复制操作，则将 param_from 的视图复制到 param_to
    param_to.copy_(param_from.view_as(param_to));
  } else {
    // 如果进行查看操作，则调整 param_from 的大小与 param_to 相同
    param_from.resize_as_(param_to);
  }
}

void _viewOrCopyParams(
    MatrixRef<Tensor> params_from,
    MatrixRef<Tensor> params_to,
    bool copy,
    bool allow_type_change = false) {
  // 确保层数相同，否则抛出异常
  TORCH_INTERNAL_ASSERT(
      params_from.size(0) == params_to.size(0), "number of layers mismatch");
  // 遍历每一层的参数
  for (const auto i : c10::irange(params_from.size(0))) {
    auto layer_params_from = params_from[i];
    auto layer_params_to = params_to[i];
    // 注意：这些列表中的参数在所有偏置之前，所以如果层不使用偏置，循环会在 layer_params_from 结束时终止并忽略它们。

    // 注意：上述声明有一个例外。如果使用带有投影的 LSTM，权重布局将是 w_ih, w_hh, b_ih, b_hh, w_hr。
    // 因此，需要特别处理没有偏置的情况，因为需要复制 0->0, 1->1, 2->4。
    // 可以通过检查每层定义的参数数量是否为 3 来唯一标识此情况。
    if (layer_params_from.size() == 3 && layer_params_to.size() != 3) {
      // 对每个参数执行单个视图或复制操作，处理特殊的无偏置情况
      _viewOrCopyOneParam(
          layer_params_from[0], layer_params_to[0], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[1], layer_params_to[1], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[2], layer_params_to[4], copy, allow_type_change);
      continue;
    }
    # 如果目标参数列表的大小为3，并且源参数列表的大小不为3，则执行以下操作
    if (layer_params_to.size() == 3 && layer_params_from.size() != 3) {
      # 将源参数列表中的第一个参数复制或视图到目标参数列表的第一个位置，根据参数设置进行复制或视图操作
      _viewOrCopyOneParam(
          layer_params_from[0], layer_params_to[0], copy, allow_type_change);
      # 将源参数列表中的第二个参数复制或视图到目标参数列表的第二个位置，根据参数设置进行复制或视图操作
      _viewOrCopyOneParam(
          layer_params_from[1], layer_params_to[1], copy, allow_type_change);
      # 将源参数列表中的第五个参数复制或视图到目标参数列表的第三个位置，根据参数设置进行复制或视图操作
      _viewOrCopyOneParam(
          layer_params_from[4], layer_params_to[2], copy, allow_type_change);
      # 继续下一次循环，处理剩余的参数
      continue;
    }
    # 遍历源参数列表和目标参数列表，逐个执行参数的复制或视图操作，根据参数设置进行相应的操作
    for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
         a != layer_params_from.end() && b != layer_params_to.end();
         ++a, ++b) {
      _viewOrCopyOneParam(*a, *b, copy, allow_type_change);
    }
  }
}

// 复制参数的函数，从一个矩阵到另一个矩阵
void _copyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
  // 调用内部函数，进行参数的复制操作
  _viewOrCopyParams(params_from, params_to, true);
}

// 视图参数的函数，从一个矩阵到另一个矩阵
void _viewParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
  // 调用内部函数，设置参数为视图模式
  _viewOrCopyParams(params_from, params_to, false);
}

// 返回输入大小的向量，根据输入参数的描述列表
std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
  // 如果输入是打包的，返回批次大小总和和输入大小的向量
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, tensors.input_size};
  } else {
    // 否则返回序列长度、迷你批次大小和输入大小的向量
    return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
  }
}

// 返回隐藏大小的向量，根据循环神经网络和张量描述列表的参数
std::vector<int64_t> _hidden_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  // 如果投影大小不为零，返回对应的向量
  if (rnn.proj_size != 0) {
    return {
        rnn.num_layers * rnn.num_directions(),
        tensors.mini_batch,
        rnn.proj_size};
  } else {
    // 否则返回隐藏大小的向量
    return {
        rnn.num_layers * rnn.num_directions(),
        tensors.mini_batch,
        rnn.hidden_size};
  }
}

// 返回细胞大小的向量，根据循环神经网络和张量描述列表的参数
std::vector<int64_t> _cell_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  // 总是返回层数和方向数乘积、迷你批次大小和隐藏大小的向量
  return {
      rnn.num_layers * rnn.num_directions(),
      tensors.mini_batch,
      rnn.hidden_size};
}

// 返回输出大小的向量，根据循环神经网络和张量描述列表的参数
std::vector<int64_t> _output_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  // 初始化输出大小为隐藏大小
  auto out_size = rnn.hidden_size;
  // 如果投影大小不为零，使用投影大小作为输出大小
  if (rnn.proj_size != 0) {
    out_size = rnn.proj_size;
  }
  // 如果输入是打包的，返回批次大小总和和输出大小乘以方向数的向量
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, out_size * rnn.num_directions()};
  } else {
    // 否则返回序列长度、迷你批次大小和输出大小乘以方向数的向量
    return {
        tensors.seq_length,
        tensors.mini_batch,
        out_size * rnn.num_directions()};
  }
}

// 使用常见的持久化启发式方法判断是否使用持久化RNN
inline bool use_persist_common_heuristics(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  // 检查RNN的层数、隐藏大小、方向数和输入大小是否符合条件
  return rnn.num_layers == 1 && rnn.hidden_size <= 1024 &&
      rnn.num_directions() == 1 && rnn.hidden_size % 128 == 0 &&
      tensors.input_size % 128 == 0;
}

// 使用设备相关的启发式方法判断是否使用设备的持久化RNN
inline bool use_persist_device_heuristics(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  // 获取当前CUDA设备的属性
  auto bsize = tensors.mini_batch;
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // 如果主版本号为7
  if (prop->major == 7) {
    // 排除Turing架构的设备使用持久化RNN
    if (prop->minor == 5) {
      return false;
    } else {
      // 对于其它设备，使用设备特定的性能检查条件来判断是否使用持久化RNN
      return ((bsize % 16 == 0 && bsize != 80 && bsize != 112) || bsize == 8) &&
          ((tensors.seq_length >= 40 && bsize <= 128) ||
           (tensors.seq_length >= 20 && bsize <= 96) ||
           (tensors.seq_length >= 10 && bsize <= 32));
    }
  } else if (prop->major >= 8 && prop->multiProcessorCount >= 98) {
    // 对于主版本号大于等于8且SM计数大于等于98的设备，排除A30设备使用持久化RNN
    # 检查 prop 指针所指向的结构体中的 minor 字段是否为 6
    if (prop->minor == 6) {
      // 排除 sm_86 GPU 设备使用持久化 RNN。
      // 原因是在 Nvidia A40 GPU 上，cudnn 8.0.5 存在一些边缘情况会抛出异常。
      return false;
    }
    // 基于 Vasily Volkov 和 xwang233 的测试结果。Vasily 只测试了 bsize <= 128，
    // 因此保守地只允许 bsize <= 128 时启用持久化。
    // TODO: 对于 bsize > 128 进行更多测试。
    if (rnn.mode == CUDNN_GRU) {
      // 持久化 GRU 的性能不如其他 RNN 类型稳定。目前排除它们。
      // TODO: 编写更精细的 GRU 启发式算法。
      return false;
    } else if (rnn.mode == CUDNN_LSTM) {
      // 对于 bsize <= 128，持久化 LSTM 的性能可与非持久化相媲美或更好。
      return (bsize % 8 == 0) && (bsize <= 128);
    } else {
      // 当 bsize >= 96 且 hidden size >= 896 时，持久化 RNN_RELU 和 TANH 显示性能较差。
      return (bsize % 8 == 0) && (bsize <= 128) &&
          (bsize < 96 || rnn.hidden_size < 896);
    }
  } else {
    // 如果不满足以上条件，则不启用持久化。
    return false;
  }
}

// namespace native 结束了 native 命名空间的定义

// Utilities exposed in RNNUtils.h 开始了在 RNNUtils.h 中公开的实用程序

namespace cudnn_rnn {

// copy_weights_to_flat_buf_views 函数开始，将权重复制到平缓缓冲区视图

// 从权重数组中复制权重到平缓缓冲区视图
TORCH_CUDA_CPP_API std::tuple<Tensor, std::vector<Tensor>>
copy_weights_to_flat_buf_views(
    // 权重数组
    TensorList weight_arr,
    // 权重步幅
    int64_t weight_stride0,
    // 输入大小
    int64_t input_size,
    // 模式
    int64_t mode,
    // 隐藏大小
    int64_t hidden_size,
    // 投影大小
    int64_t proj_size,
    // 层数
    int64_t num_layers,
    // 批次优先
    bool batch_first,
    // 双向
    bool bidirectional,
    // 平缓缓冲区数据类型
    const cudnnDataType_t flat_buf_datatype,
    // 平缓缓冲区选项
    const TensorOptions& flat_buf_options,
    // 将原始权重设置为平缓缓冲区
    bool set_orig_weights_to_flat_buf,
    // 允许类型更改，默认为假
    bool allow_type_change /*=false*/) {


这段代码是关于在RNNUtils.h中公开的一个函数，用于将权重复制到平缓缓冲区视图中。
    bool include_bias /*=true*/) {


// 定义一个布尔型参数 include_bias，表示是否包含偏置项，默认为 true
// 在函数参数列表中，注释指出了该参数的默认值为 true
// 该参数控制是否将偏置项包含在处理中

  // flat_buf_datatype is accepted as a separate argument (rather than extracted
  // from flat_buf_options) because to extract flat_buf_datatype from
  // flat_buf_options, we'd need to say auto flat_buf_datatype =
  // getCudnnDataTypeFromScalarType(typeMetaToScalarType(options.dtype()));
  // typeMetaToScalarType is a surprisingly nontrivial function.  We should
  // avoid it if we can.


  // 这段注释解释了为什么 flat_buf_datatype 被作为独立的参数接受，
  // 而不是从 flat_buf_options 中提取出来。提取 flat_buf_datatype 
  // 需要调用 typeMetaToScalarType(options.dtype())，这是一个复杂的函数。
  // 函数功能复杂，应该尽量避免调用。


  TORCH_CHECK(
      weight_arr.size() > 0,
      "copy_weights_to_flat_buf_views: cannot flatten empty weight list");


  // 使用 TORCH_CHECK 宏检查 weight_arr 的大小是否大于 0
  // 如果 weight_arr 为空，则抛出错误信息 "copy_weights_to_flat_buf_views: cannot flatten empty weight list"
  // 用来确保 weight_arr 非空，否则无法继续处理


  RNNDescriptorParams rnn;
  rnn.set(
      mode,


  // 创建 RNNDescriptorParams 对象 rnn，用于设置 RNN 描述符的参数
  // 后续会调用 rnn.set 方法来设置具体参数
#ifdef USE_CUDNN_RNN_V8_API
      input_size,
      false, // eqy: bogus as we do not know if the input is packed here
             // but it should not affect the weights (what are are interested
             // in)
#endif
      hidden_size,
      proj_size,
      num_layers,
      bidirectional,
      promote_rnn_math_type(flat_buf_datatype),
      flat_buf_datatype);


// 如果定义了 USE_CUDNN_RNN_V8_API，则使用以下参数创建 RNN 描述符：
//   input_size: 输入的特征维度大小
//   hidden_size: 隐藏层的大小
//   proj_size: 投影层的大小（如果有的话）
//   num_layers: RNN 的层数
//   bidirectional: 是否使用双向 RNN
//   promote_rnn_math_type(flat_buf_datatype): 推广 RNN 的数学类型，根据 flat_buf_datatype
//   flat_buf_datatype: 数据缓冲区的数据类型


  auto handle = getCudnnHandle();
  RNNDescriptor rnn_desc = rnn.descriptor(handle);


// 获取 cuDNN 的句柄
auto handle = getCudnnHandle();
// 根据句柄获取 RNN 描述符
RNNDescriptor rnn_desc = rnn.descriptor(handle);


  TensorGeometry x_geom({1, input_size});
  TensorDescriptor x_desc;
  // Why do we pad to 5 dims here (and elsewhere)?
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNForwardTraining
  // expects descriptors padded to 3 dimensions.
  x_desc.set(flat_buf_datatype, x_geom.sizes(), x_geom.strides(), 5);


// 创建输入张量的几何描述，维度为 {1, input_size}
TensorGeometry x_geom({1, input_size});
// 创建张量描述符 x_desc，设置数据类型、大小、步长，并将维度填充到 5 维
// 这里维度填充到 5 是为了符合 cuDNN API 的要求，尽管实际数据维度为 3
x_desc.set(flat_buf_datatype, x_geom.sizes(), x_geom.strides(), 5);


  auto num_weights =
#ifndef USE_CUDNN_RNN_V8_API
      get_num_weights(handle, rnn_desc, x_desc, flat_buf_datatype);
#else
      get_num_weights(handle, rnn_desc, flat_buf_datatype);
#endif


// 根据是否定义了 USE_CUDNN_RNN_V8_API，获取权重数量
auto num_weights =
#ifndef USE_CUDNN_RNN_V8_API
    get_num_weights(handle, rnn_desc, x_desc, flat_buf_datatype);
#else
    get_num_weights(handle, rnn_desc, flat_buf_datatype);
#endif


  auto weight_buf = at::zeros(num_weights, flat_buf_options);


// 创建一个全零张量 weight_buf，用于存储权重，大小为 num_weights，选项为 flat_buf_options


#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif


// 如果没有定义 USE_CUDNN_RNN_V8_API，则创建过滤器描述符 w_desc，
// 并将 weight_buf 设置为其数据源，维度为 3


  // Slice off views into weight_buf
  auto [params_arr, params_stride0] = get_parameters(
#ifndef USE_CUDNN_RNN_V8_API
      handle, rnn, rnn_desc, x_desc, w_desc, weight_buf, include_bias);
#else
      handle, rnn, rnn_desc, weight_buf, include_bias);
#endif


// 获取权重参数的视图
auto [params_arr, params_stride0] = get_parameters(
#ifndef USE_CUDNN_RNN_V8_API
    handle, rnn, rnn_desc, x_desc, w_desc, weight_buf, include_bias);
#else
    handle, rnn, rnn_desc, weight_buf, include_bias);
#endif


  MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)},
      params{params_arr, params_stride0};


// 创建权重矩阵 weight 和参数矩阵 params 的引用
MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)},
    params{params_arr, params_stride0};


  // Copy weights
  _viewOrCopyParams(weight, params, /*copy=*/true, allow_type_change);
  if (set_orig_weights_to_flat_buf) {
    // Update the storage
    for (const auto i : c10::irange(weight.size(0))) {
      // There is a special case for LSTM with projections and no bias,
      // where weight copy is done in 0->0, 1->1, 2->4 layout
      if (weight[i].size() == 3 && params[i].size() == 5) {
        weight[i][0].set_(params[i][0].view_as(weight[i][0]));
        weight[i][1].set_(params[i][1].view_as(weight[i][1]));
        weight[i][2].set_(params[i][4].view_as(weight[i][2]));
      } else {
        for (auto orig_param_it = weight[i].begin(),
                  new_param_it = params[i].begin();
             orig_param_it != weight[i].end() &&
             new_param_it != params[i].end();
             orig_param_it++, new_param_it++) {
          auto orig_param = *orig_param_it, new_param = *new_param_it;
          orig_param.set_(new_param.view_as(orig_param));
        }
      }
    }
  }


// 复制权重数据
_viewOrCopyParams(weight, params, /*copy=*/true, allow_type_change);
// 如果需要将原始权重更新到 flat_buf 中
if (set_orig_weights_to_flat_buf) {
  // 更新存储
  for (const auto i : c10::irange(weight.size(0))) {
    // 对于 LSTM 的特殊情况，包含投影且没有偏置，
    // 权重复制按照 0->0, 1->1, 2->4 的顺序进行
    if (weight[i].size() == 3 && params[i].size() == 5) {
      weight[i][0].set_(params[i][0].view_as(weight[i][0]));
      weight[i][1].set_(params[i][1].view_as(weight[i][1]));
      weight[i][2].set_(params[i][4].view_as(weight[i][2]));
    } else {
      for (auto orig_param_it = weight[i].begin(),
                new_param_it = params[i].begin();
           orig_param_it != weight[i].end() &&
           new_param_it != params[i].end();
           orig_param_it++, new_param_it++) {
        auto orig_param = *orig_param_it, new_param = *new_param_it;
        orig_param.set_(new_param.view_as(orig_param));
      }
    }
  }
}


  return std::make_tuple(weight_buf, params_arr);
}


// 返回包含权重缓冲区和参数数组的元组
    // 定义一个名为 `copy_weights_to_flat_buf_views` 的函数，返回值类型为 `std::tuple` 中的第一个元素，即 flat weight_buf
    // 调用 `copy_weights_to_flat_buf_views` 函数，传递以下参数进行调用：
    //   - `weight_arr`: 权重数组
    //   - `weight_stride0`: 第一个维度的步长
    //   - `input_size`: 输入大小
    //   - `fn_mode`: 函数模式
    //   - `fn_hidden_size`: 隐藏层大小
    //   - `fn_proj_size`: 投影层大小
    //   - `fn_num_layers`: 层数
    //   - `batch_first`: 是否批量优先
    //   - `fn_bidirectional`: 是否双向
    //   - `getCudnnDataType(weight_arr[0])`: 获取 `weight_arr` 的数据类型
    //   - `weight_arr[0].options()`: 获取 `weight_arr[0]` 的选项
    //   - `true`: 设置原始权重为 flat_buf
    
    return std::get<0>(copy_weights_to_flat_buf_views(
        weight_arr,
        weight_stride0,
        input_size,
        fn_mode,
        fn_hidden_size,
        fn_proj_size,
        fn_num_layers,
        batch_first,
        fn_bidirectional,
        /*flat_buf_datatype=*/getCudnnDataType(weight_arr[0]),
        /*flat_buf_options=*/weight_arr[0].options(),
        /*set_orig_weights_to_flat_buf=*/true));
}

// 定义一个 C 风格字符串常量，用于警告 RNN 模型权重不连续的内存分布可能导致内存使用急剧增加
const char* WEIGHT_FORMAT_WARN =
    "RNN module weights are not part of single contiguous "
    "chunk of memory. This means they need to be compacted "
    "at every call, possibly greatly increasing memory usage. "
    "To compact weights again call flatten_parameters().";

// NB: 当 fn_batch_sizes 为空时，表示未指定批处理大小
// 定义一个函数 _cudnn_rnn，用于执行 CuDNN RNN 操作，返回多个张量作为结果
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight,
    int64_t weight_stride0,
    const std::optional<Tensor>& weight_buf_r_opt,
    const Tensor& hx,
    const std::optional<Tensor>& cx_opt,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const std::optional<Tensor>& fn_dropout_state_opt) {
  
  // 从可选的权重缓冲区张量中借用一个常规的 Tensor 引用
  c10::MaybeOwned<Tensor> weight_buf_r_maybe_owned =
      at::borrow_from_optional_tensor(weight_buf_r_opt);
  const Tensor& weight_buf_r = *weight_buf_r_maybe_owned;
  
  // 从可选的 cx_opt 中获取或者创建一个空张量 cx
  const Tensor& cx = c10::value_or_else(cx_opt, [] { return Tensor(); });
  
  // 从可选的 fn_dropout_state_opt 中获取或者创建一个空张量 fn_dropout_state
  const Tensor& fn_dropout_state =
      c10::value_or_else(fn_dropout_state_opt, [] { return Tensor(); });

  // 检查输入参数和权重张量的属性是否满足要求，确保数据类型匹配
  check_attributes(input_r, weight, {hx, cx}, /*check_dtype=*/true);

  // 复制输入张量作为 input 的副本
  auto input = input_r;
  
  // 复制权重缓冲区张量作为 weight_buf 的副本
  auto weight_buf = weight_buf_r;
  
  // 如果权重缓冲区未定义，则发出警告
  if (!weight_buf.defined()) {
    TORCH_WARN(WEIGHT_FORMAT_WARN);
  }

  // 如果 fn_dropout_state 定义了，则检查输入和 dropout_state 是否在同一 GPU 上
  if (fn_dropout_state.defined()) {
    auto input_arg = TensorArg(input, "input", 1);
    auto dropout_state_arg = TensorArg(fn_dropout_state, "dropout_states", 15);
    checkSameGPU("cudnn_rnn", input_arg, dropout_state_arg);
  }

  // 初始化 RNNParams 结构体实例 fn，获取当前输入张量的 CuDNN 数据类型
  RNNParams fn;
  auto datatype = getCudnnDataType(input);

  // 根据 CuDNN 版本设置 RNN 模型参数
#ifndef USE_CUDNN_RNN_V8_API
  fn.rnn.set(
      fn_mode,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#else
  auto input_size = input_r.size(-1);
  auto packed = fn_batch_sizes.size() != 0;
  fn.rnn.set(
      fn_mode,
      input_size,
      packed,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#endif

  // 设置 RNNParams 结构体中的 dropout 参数
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);

  // 设置 RNNParams 结构体中的 tensors 参数，包括输入张量的大小和批处理大小信息
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // 如果 RNN 模型不是 LSTM，则检查是否定义了 cx，否则报错
  if (fn.rnn.mode != CUDNN_LSTM) {
    TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  // 检查 batch_first 参数是否为真，并且输入是否已打包，以确定是否需要重排输入数据
  auto is_input_packed = fn.tensors.batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    // 对输入进行转置，交换维度0和1
    input = input.transpose(0, 1);
  }

  // 计算隐藏状态的大小
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  // 计算细胞状态的大小
  auto cell_size = _cell_size(fn.rnn, fn.tensors);
  // 计算输出的大小
  auto output_size = _output_size(fn.rnn, fn.tensors);

  // 检查隐藏状态 hx 是否是连续的存储
  TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  // 检查细胞状态 cx 是否已定义且是连续的存储
  TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  // 将输入 x 变成连续的存储
  auto x = input.contiguous();
  // 创建一个空的输出张量，形状为 output_size，使用与输入相同的选项
  auto output = at::empty(output_size, input.options());
  // 创建一个空的隐藏状态张量，形状为 hidden_size，使用与 hx 相同的选项
  auto hy = at::empty(hidden_size, hx.options());
  // 定义细胞状态 cy
  Tensor cy;
  // 如果细胞状态 cx 已定义，则创建一个形状为 cell_size 的空张量，使用与 cx 相同的选项
  if (cx.defined()) {
    cy = at::empty(cell_size, cx.options());
  } else {
    // 否则，创建一个形状为 {0} 的空张量，使用与 hx 相同的选项
    cy = at::empty(
        {0}, hx.options()); // NB: Not allowed to return undefined tensors
  }
  // 将 y 初始化为 output
  auto y = output;

  // 获取当前的 cuDNN 句柄
  auto handle = getCudnnHandle();
  // 获取当前 cuDNN RNN 操作的算法
  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors, input, true);
  // 设置 RNN 操作的算法
  fn.rnn.set_algo(algo);
  // 创建 RNN 的描述符
  RNNDescriptors descs(fn, handle, x, y, hx, cx);
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则定义滤波器描述符 w_desc
  FilterDescriptor w_desc;
#endif
  // 如果权重缓冲未定义
  if (!weight_buf.defined()) {
#ifndef USE_CUDNN_RNN_V8_API
    // 获取权重的数量
    auto num_weights =
        get_num_weights(handle, descs.rnn_desc, descs.x_descs[0], datatype);
#else
    // 使用更新后的接口获取权重的数量
    auto num_weights = get_num_weights(handle, descs.rnn_desc, datatype);
#endif
    // 使用权重数量创建一个空的 Tensor，用于存储权重
    weight_buf = at::empty(num_weights, x.options());
#ifndef USE_CUDNN_RNN_V8_API
    // 如果未定义使用 CUDNN RNN V8 API，则设置 w_desc 的信息
    w_desc.set(weight_buf, 3);
#endif
    // 将权重 Tensor 初始化为零
    weight_buf.zero_();
#ifndef USE_CUDNN_RNN_V8_API
    // 如果未定义使用 CUDNN RNN V8 API，则获取参数信息
    auto [params, params_stride0] = get_parameters(
        handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
#else
    // 使用更新后的接口获取参数信息
    auto [params, params_stride0] =
        get_parameters(handle, fn.rnn, descs.rnn_desc, weight_buf);
#endif
    // 将参数复制到外部权重 Tensor 中
    _copyParams(
        MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
        MatrixRef<Tensor>{params, params_stride0});
  } else {
#ifndef USE_CUDNN_RNN_V8_API
    // 如果未定义使用 CUDNN RNN V8 API，则设置 w_desc 的信息
    w_desc.set(weight_buf, 3);
#endif
  }

  // 检查是否定义了 cx 或其大小与 cell_size 相符
  TORCH_CHECK(
      !cx.defined() || cx.sizes().equals(cell_size),
      "Expected cell size ",
      IntArrayRef{cell_size},
      ", got ",
      cx.sizes());
  // 定义工作空间大小变量
  size_t workspace_size;
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则获取 RNN 的工作空间大小
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
#else
  // 使用更新后的接口获取输入和输出描述符数组的引用
  auto& x_descs_arr = descs.x_descs;
  auto& y_descs_arr = descs.y_descs;
#endif
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则获取 RNN 的工作空间大小
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      &workspace_size));
#endif
  // 定义工作空间和保留空间 Tensor
  Tensor workspace;
  Tensor reserve;
  // 注意：之前测试的是 fn.requires_grad，但我们没有此信息，用 'train' 作为代理
  if (fn_train) {
    // 如果正在训练，则获取 RNN 训练保留空间的大小
    size_t reserve_size;
#ifndef USE_CUDNN_RNN_V8_API
    AT_CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &reserve_size));
#else
    // 使用更新后的接口获取 RNN 临时空间大小
    AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_TRAINING,
        x_descs_arr.desc(),
        &workspace_size,
        &reserve_size));
#endif
    // 创建工作空间和保留空间的 Tensor
    workspace = at::empty(workspace_size, input.options().dtype(kByte));
    reserve = at::empty(reserve_size, input.options().dtype(kByte));
#ifndef USE_CUDNN_RNN_V8_API

    // 如果未定义使用 CUDNN RNN V8 API，则获取 RNN 训练保留空间的大小
    AT_CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &reserve_size));
#else
    // 使用更新后的接口创建保留空间的 Tensor
    reserve = at::empty(reserve_size, input.options().dtype(kByte));
#endif
  }
#ifndef USE_CUDNN_RNN_V8_API
    // 调用 cuDNN 库中的函数 cudnnRNNForwardTraining 来执行 RNN 的前向训练
    AT_CUDNN_CHECK(cudnnRNNForwardTraining(
        handle,                                     // cuDNN 执行句柄
        descs.rnn_desc.desc(),                      // RNN 描述符，描述 RNN 的结构
        fn.tensors.seq_length,                      // 序列长度
        x_descs_arr.data(),                         // 输入数据描述符数组
        x.data_ptr(),                               // 输入数据张量指针
        descs.hx_desc.desc(),                       // RNN 初始隐藏状态描述符
        hx.data_ptr(),                              // RNN 初始隐藏状态张量指针
        descs.cx_desc.desc(),                       // RNN 初始细胞状态描述符
        cx.defined() ? cx.data_ptr() : nullptr,     // RNN 初始细胞状态张量指针（可选）
        w_desc.desc(),                              // 权重描述符
        weight_buf.data_ptr(),                      // 权重数据张量指针
        y_descs_arr.data(),                         // 输出数据描述符数组
        y.data_ptr(),                               // 输出数据张量指针
        descs.hy_desc.desc(),                       // 输出最终隐藏状态描述符
        hy.data_ptr(),                              // 输出最终隐藏状态张量指针
        descs.cy_desc.desc(),                       // 输出最终细胞状态描述符
        cy.defined() ? cy.data_ptr() : nullptr,     // 输出最终细胞状态张量指针（可选）
        workspace.data_ptr(),                       // 工作空间张量指针
        workspace.size(0),                          // 工作空间大小
        reserve.mutable_data_ptr(),                 // 保留内存的可变数据指针
        reserve.size(0)));                          // 保留内存的大小
#else
    // 如果不是推断模式，执行以下代码段
    AT_CUDNN_CHECK(cudnnRNNForward(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_TRAINING,
        nullptr,
        x_descs_arr.desc(),
        x.data_ptr(),
        y_descs_arr.desc(),
        y.data_ptr(),
        descs.hx_desc.desc(),
        hx.data_ptr(),
        hy.data_ptr(),
        descs.cx_desc.desc(),
        cx.defined() ? cx.data_ptr() : nullptr,
        cy.defined() ? cy.data_ptr() : nullptr,
        weight_buf.numel() * weight_buf.element_size(),
        weight_buf.data_ptr(),
        workspace.size(0),
        workspace.data_ptr(),
        reserve.size(0),
        reserve.mutable_data_ptr()));
#endif
  } else { // 推断模式
#ifdef USE_CUDNN_RNN_V8_API
    // 如果使用 CUDNN RNN V8 API，获取推断模式下的临时空间大小
    AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_INFERENCE,
        x_descs_arr.desc(),
        &workspace_size,
        NULL));
#endif
    // 分配推断模式下的工作空间
    workspace = at::empty(workspace_size, input.options().dtype(kByte));
    // 分配空的 reserve 张量
    reserve = at::empty({0}, input.options().dtype(kByte));
#ifndef USE_CUDNN_RNN_V8_API
    // 如果不使用 CUDNN RNN V8 API，执行推断模式下的前向传播
    AT_CUDNN_CHECK(cudnnRNNForwardInference(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        x.data_ptr(),
        descs.hx_desc.desc(),
        hx.data_ptr(),
        descs.cx_desc.desc(),
        cx.defined() ? cx.data_ptr() : nullptr,
        w_desc.desc(),
        weight_buf.data_ptr(),
        y_descs_arr.data(),
        y.data_ptr(),
        descs.hy_desc.desc(),
        hy.data_ptr(),
        descs.cy_desc.desc(),
        cy.defined() ? cy.data_ptr() : nullptr,
        workspace.data_ptr(),
        workspace.size(0)));
#else
    // 使用 CUDNN RNN V8 API 执行推断模式下的前向传播
    AT_CUDNN_CHECK(cudnnRNNForward(
        handle,
        descs.rnn_desc.desc(),
        CUDNN_FWD_MODE_INFERENCE,
        nullptr,
        x_descs_arr.desc(),
        x.data_ptr(),
        y_descs_arr.desc(),
        y.data_ptr(),
        descs.hx_desc.desc(),
        hx.data_ptr(),
        hy.data_ptr(),
        descs.cx_desc.desc(),
        cx.defined() ? cx.data_ptr() : nullptr,
        cy.defined() ? cy.data_ptr() : nullptr,
        weight_buf.numel() * weight_buf.element_size(),
        weight_buf.data_ptr(),
        workspace.size(0),
        workspace.data_ptr(),
        reserve.size(0),
        reserve.mutable_data_ptr()));
#endif
  }

  // 如果 batch_first 为 true 并且输入未打包，则对输出进行转置
  if (batch_first && !is_input_packed) {
    output.transpose_(0, 1);
  }

  // 返回输出、最终隐藏状态、最终细胞状态、reserve 张量和权重缓冲区的元组
  return std::make_tuple(output, hy, cy, reserve, weight_buf);
}

std::tuple<Tensor, Tensor, Tensor> _cudnn_rnn_backward_input(
    const Tensor& input_r,
    const Tensor& weight_buf,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output_r,
    const Tensor& grad_output_r,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
  // 定义一个新的变量 input，并将 input_r 赋值给它
  auto input = input_r;
  // 定义一个新的变量 grad_output，并将 grad_output_r 赋值给它
  auto grad_output = grad_output_r;
  // 定义一个新的变量 output，并将 output_r 赋值给它
  auto output = output_r;

  // 创建一个名为 fn 的 RNNParams 对象
  RNNParams fn;
  // 获取输入张量 input 的 CUDA 数据类型，并将其赋值给 datatype
  auto datatype = getCudnnDataType(input);
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则设置 RNN 参数
  fn.rnn.set(
      fn_mode,                         // RNN 模式
      fn_hidden_size,                  // 隐藏层大小
      fn_proj_size,                    // 投影层大小
      fn_num_layers,                   // RNN 层数
      fn_bidirectional,                // 是否双向
      promote_rnn_math_type(datatype), // 推广 RNN 数学类型
      datatype);                       // 数据类型
#else
  // 如果定义使用 CUDNN RNN V8 API，则设置对应参数
  auto cudnn_input_size = input_r.size(-1);  // 获取输入张量的最后一个维度大小
  auto packed = fn_batch_sizes.size() != 0;  // 检查是否使用了批量大小
  fn.rnn.set(
      fn_mode,                         // RNN 模式
      cudnn_input_size,                // CUDNN 输入大小
      packed,                          // 是否打包输入
      fn_hidden_size,                  // 隐藏层大小
      fn_proj_size,                    // 投影层大小
      fn_num_layers,                   // RNN 层数
      fn_bidirectional,                // 是否双向
      promote_rnn_math_type(datatype), // 推广 RNN 数学类型
      datatype);                       // 数据类型
#endif

// 设置 dropout 参数
fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);

// 设置张量尺寸和批量大小，优先考虑批量维度
fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

// TODO: 将设备设置为输入设备
auto handle = getCudnnHandle();

// 如果 RNN 模式不是 LSTM，则检查上下文张量是否已定义
if (fn.rnn.mode != CUDNN_LSTM) {
  TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
}

// 检查是否输入已打包（按批量优先且未打包情况下）
auto is_input_packed = fn_batch_sizes.size() != 0;
if (batch_first && !is_input_packed) {
  // 如果是按批量优先且未打包，则转置输入和梯度输出张量
  input = input.transpose(0, 1);
  grad_output = grad_output.transpose(0, 1);
}
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto cell_size = _cell_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  // 检查 hx 张量是否是连续的
  TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  // 检查如果 cx 已定义，则其张量是否是连续的
  TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  // 将输入张量转换为连续的
  auto x = input.contiguous();
  // 将梯度输出张量转换为连续的
  auto dy = grad_output.contiguous();
  auto y = output;
  auto w = weight_buf;
  // 创建一个和输入张量相同大小和类型的空张量 dx
  auto dx = at::empty(
      input.sizes(), input.options()); // TODO: more compact way of saying this
  // 将梯度 grad_hy 转换为连续的，并视图为 hidden_size 的形状
  auto dhy = grad_hy.contiguous().view(hidden_size);
  // 如果 grad_cy 已定义，则将其转换为连续的并视图为 cell_size 的形状；否则创建一个空张量
  auto dcy =
      grad_cy.defined() ? grad_cy.contiguous().view(cell_size) : Tensor();
  // 创建一个和 hx 相同大小和类型的空张量 dhx
  auto dhx = at::empty(hidden_size, hx.options());
  // 内部断言，检查是否定义了 cx 或者 output_mask[2] 为假
  TORCH_INTERNAL_ASSERT(
      cx.defined() || !output_mask[2],
      "illegally required grad of cx for non-LSTM RNN");
  // 如果定义了 cx，则创建一个和 cx 相同大小和类型的空张量 dcx；否则创建一个空张量
  auto dcx = cx.defined() ? at::empty(cell_size, cx.options()) : Tensor();

  // 检查是否处于训练模式，否则报错
  TORCH_CHECK(
      fn_train, "cudnn RNN backward can only be called in training mode");

  // 检查输入张量的大小是否符合预期
  TORCH_CHECK(
      input.sizes().equals(input_size),
      "Expected input size ",
      IntArrayRef{input_size},
      ", got ",
      input.sizes());
  // 检查输出张量的大小是否符合预期
  TORCH_CHECK(
      output.sizes().equals(output_size),
      "Expected output size ",
      IntArrayRef{output_size},
      ", got ",
      output.sizes());

  // 检查 hx 张量的大小是否符合预期
  TORCH_CHECK(
      !hx.defined() || hx.sizes().equals(hidden_size),
      "Expected hidden size ",
      IntArrayRef{hidden_size},
      ", got ",
      hx.sizes());
  // 检查 cx 张量的大小是否符合预期
  TORCH_CHECK(
      !cx.defined() || cx.sizes().equals(cell_size),
      "Expected cell size ",
      IntArrayRef{cell_size},
      ", got ",
      cx.sizes());
  // 检查 dhy 张量的大小是否符合预期
  TORCH_CHECK(
      !dhy.defined() || dhy.sizes().equals(hidden_size),
      "Expected d_hidden size ",
      IntArrayRef{hidden_size},
      ", got ",
      dhy.sizes());
  // 检查 dcy 张量的大小是否符合预期
  TORCH_CHECK(
      !dcy.defined() || dcy.sizes().equals(cell_size),
      "Expected d_cell size ",
      IntArrayRef{cell_size},
      ", got ",
      dcy.sizes());

  // 检查 dhy、dy 和 dcy 是否都是 CUDA 张量
  TORCH_CHECK(
      dhy.is_cuda() && dy.is_cuda() && (!dcy.defined() || dcy.is_cuda()),
      "Gradients aren't CUDA tensors");

  // 获取 cudnn 的 RNN 算法
  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors, input, false);
  // 设置 fn.rnn 的算法
  fn.rnn.set_algo(algo);
  // 创建 RNN 描述符
  RNNDescriptors descs(fn, handle, x, y, hx, cx);
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义 USE_CUDNN_RNN_V8_API 宏，则创建一个名为 w_desc 的 FilterDescriptor 对象，并使用 weight_buf 设置其参数
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  // 声明一个变量用于存储 workspace 的大小
  size_t workspace_size;
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义 USE_CUDNN_RNN_V8_API 宏，则获取描述符对象 descs 的 x_descs 和 y_descs 的数组，并通过 cudnnGetRNNWorkspaceSize 函数获取工作空间的大小
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      &workspace_size));
#else
  // 如果定义了 USE_CUDNN_RNN_V8_API 宏，则获取描述符对象 descs 的 x_descs 和 y_descs 的引用，并通过 cudnnGetRNNTempSpaceSizes 函数获取工作空间的大小
  auto& x_descs_arr = descs.x_descs;
  auto& y_descs_arr = descs.y_descs;
  AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
      handle,
      descs.rnn_desc.desc(),
      CUDNN_FWD_MODE_TRAINING,
      x_descs_arr.desc(),
      &workspace_size,
      NULL));
#endif

  // 创建一个名为 workspace 的 Tensor 对象，用于存储工作空间数据，数据类型为 kByte，大小为 workspace_size
  // TODO: put this in the correct device???
  Tensor workspace = at::empty(workspace_size, input.options().dtype(kByte));

#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义 USE_CUDNN_RNN_V8_API 宏，则调用 cudnnRNNBackwardData 函数进行 RNN 反向数据传播
  AT_CUDNN_CHECK(cudnnRNNBackwardData(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      y_descs_arr.data(),
      y.data_ptr(),
      y_descs_arr.data(),
      dy.data_ptr(),
      descs.hy_desc.desc(),
      dhy.data_ptr(),
      descs.cy_desc.desc(),
      cx.defined() ? dcy.data_ptr() : nullptr,
      w_desc.desc(),
      w.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      descs.cx_desc.desc(),
      cx.defined() ? cx.data_ptr() : nullptr,
      x_descs_arr.data(),
      dx.data_ptr(),
      descs.hx_desc.desc(),
      dhx.data_ptr(),
      descs.cx_desc.desc(),
      cx.defined() ? dcx.data_ptr() : nullptr,
      workspace.data_ptr(),
      workspace.size(0),
      fn_reserve.data_ptr(),
      fn_reserve.size(0)));
#else
  // 如果定义了 USE_CUDNN_RNN_V8_API 宏，则调用 cudnnRNNBackwardData_v8 函数进行 RNN 反向数据传播
  AT_CUDNN_CHECK(cudnnRNNBackwardData_v8(
      handle,
      descs.rnn_desc.desc(),
      nullptr,
      y_descs_arr.desc(),
      y.data_ptr(),
      dy.data_ptr(),
      x_descs_arr.desc(),
      dx.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      dhy.data_ptr(),
      dhx.data_ptr(),
      descs.cx_desc.desc(),
      cx.defined() ? cx.data_ptr() : nullptr,
      cx.defined() ? dcy.data_ptr() : nullptr,
      cx.defined() ? dcx.data_ptr() : nullptr,
      weight_buf.numel() * weight_buf.element_size(),
      weight_buf.data_ptr(),
      workspace.size(0),
      workspace.data_ptr(),
      fn_reserve.size(0),
      fn_reserve.data_ptr()));
#endif

  // 如果 batch_first 为 true，并且输入未打包，则对 dx 进行转置操作
  if (batch_first && !is_input_packed) {
    dx = dx.transpose_(0, 1);
  }

  // 返回一个包含 dx、dhx 和 dcx 的 std::tuple
  return std::make_tuple(dx, dhx, dcx);
}

// 注意：此函数必须在 _cudnn_rnn_backward_input 之后调用
// 我们将提供一个用户友好的组合函数...
std::vector<Tensor> _cudnn_rnn_backward_weight(
    // TODO: 我认为张量的几何属性对于 weight_buf/weight 是足够的
    const Tensor& input_r,
    // 权重数组列表
    TensorList weight_arr,
    // 权重数组的步长
    int64_t weight_stride0,
    // 权重缓冲区
    const Tensor& weight_buf,
    // 隐藏状态
    const Tensor& hx,
    // 细胞状态
    const Tensor& cx,
    // 输出
    const Tensor& output_r,
    // 前向模式
    int64_t fn_mode,
    // 隐藏层大小
    int64_t fn_hidden_size,
    // 投影大小
    int64_t fn_proj_size,
    // 层数
    int64_t fn_num_layers,
    // 是否 batch_first
    bool batch_first,
    // 丢弃率
    double fn_dropout,
    // 是否训练
    bool fn_train,
    // 是否双向
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const Tensor& fn_dropout_state,
    const Tensor& fn_reserve) {


# 定义函数参数：批量大小数组引用，Dropout 状态张量引用，保留张量引用
MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)};
# 使用权重数组和步幅创建矩阵引用对象
auto input = input_r;
# 复制输入参数为 input
auto output = output_r;
# 复制输出参数为 output

RNNParams fn;
# 创建 RNN 参数对象 fn

auto datatype = getCudnnDataType(input);
# 调用函数获取输入数据的 CUDA 数据类型并赋值给 datatype
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用CUDNN RNN v8 API，则设置RNN参数
  fn.rnn.set(
      fn_mode,  // RNN 模式
      fn_hidden_size,  // 隐藏层大小
      fn_proj_size,  // 投影层大小
      fn_num_layers,  // RNN 层数
      fn_bidirectional,  // 是否双向
      promote_rnn_math_type(datatype),  // 推广RNN数学类型
      datatype);  // 数据类型
#else
  // 否则，使用CUDNN RNN v8 API，设置RNN参数
  auto cudnn_input_size = input_r.size(-1);  // 获取CUDNN输入大小
  auto packed = fn_batch_sizes.size() != 0;  // 检查是否打包输入
  fn.rnn.set(
      fn_mode,  // RNN 模式
      cudnn_input_size,  // CUDNN输入大小
      packed,  // 输入是否打包
      fn_hidden_size,  // 隐藏层大小
      fn_proj_size,  // 投影层大小
      fn_num_layers,  // RNN 层数
      fn_bidirectional,  // 是否双向
      promote_rnn_math_type(datatype),  // 推广RNN数学类型
      datatype);  // 数据类型
#endif

  // 设置dropout参数
  fn.dropout.set(fn_train,  // 训练模式
                 fn_dropout,  // dropout比例
                 fn_dropout_state);  // dropout状态

  // 设置张量参数
  fn.tensors.set(input.sizes(),  // 输入大小
                 fn_batch_sizes,  // 批次大小
                 batch_first);  // 是否批次优先

  // 获取CUDNN句柄
  auto handle = getCudnnHandle();

  // 对于非LSTM类型的RNN，确保cx未定义
  if (fn.rnn.mode != CUDNN_LSTM) {
    TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  // 检查输入是否已打包
  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    // 如果是批次优先且未打包，转置输入和输出张量
    input = input.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  // 获取输入大小和隐藏层大小
  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);

  // 检查是否处于训练模式
  TORCH_CHECK(
      fn_train, "cudnn RNN backward can only be called in training mode");

  // 检查输入大小是否符合预期
  TORCH_CHECK(
      input.sizes().equals(input_size),
      "Expected input size ",
      IntArrayRef{input_size},
      ", got ",
      input.sizes());

  // 检查隐藏状态大小是否符合预期
  TORCH_CHECK(
      !hx.defined() || hx.sizes().equals(hidden_size),
      "Expected hidden size ",
      IntArrayRef{hidden_size},
      ", got ",
      hx.sizes());

  // 检查hx张量是否连续
  TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  // 检查cx张量是否连续（如果定义了）
  TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  // 保证输入张量是连续的
  auto x = input.contiguous();
  // 引用输出张量
  const auto& y = output;
  // 创建用于权重的零张量
  auto dw = at::zeros(weight_buf.sizes(), weight_buf.options());

  // 获取RNN算法
  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors, input, false);
  // 设置RNN算法
  fn.rnn.set_algo(algo);
  // 创建RNN描述符
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用CUDNN RNN v8 API，创建滤波器描述符
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  size_t workspace_size;
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用CUDNN RNN v8 API，获取RNN工作空间大小
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      &workspace_size));
#else
  // 否则，使用CUDNN RNN v8 API，获取RNN临时空间大小
  auto& x_descs_arr = descs.x_descs;
  auto& y_descs_arr = descs.y_descs;
  AT_CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(
      handle,
      descs.rnn_desc.desc(),
      CUDNN_FWD_MODE_TRAINING,
      x_descs_arr.desc(),
      &workspace_size,
      NULL));
#endif

  // 创建工作空间张量
  Tensor workspace = at::empty(workspace_size, input.options().dtype(kByte));
#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则调用 cudnnRNNBackwardWeights 函数
  AT_CUDNN_CHECK(cudnnRNNBackwardWeights(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors.seq_length,
      x_descs_arr.data(),
      x.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      y_descs_arr.data(),
      y.data_ptr(),
      workspace.data_ptr(),
      workspace.size(0),
      w_desc.desc(),
      dw.data_ptr(),
      fn_reserve.data_ptr(),
      fn_reserve.size(0)));
#else
  // 如果定义了使用 CUDNN RNN V8 API，则调用 cudnnRNNBackwardWeights_v8 函数
  AT_CUDNN_CHECK(cudnnRNNBackwardWeights_v8(
      handle,
      descs.rnn_desc.desc(),
      CUDNN_WGRAD_MODE_ADD,
      nullptr,
      x_descs_arr.desc(),
      x.data_ptr(),
      descs.hx_desc.desc(),
      hx.data_ptr(),
      y_descs_arr.desc(),
      y.data_ptr(),
      weight_buf.numel() * weight_buf.element_size(),
      dw.data_ptr(),
      workspace.size(0),
      workspace.data_ptr(),
      fn_reserve.size(0),
      fn_reserve.data_ptr()));
#endif

#ifndef USE_CUDNN_RNN_V8_API
  // 如果未定义使用 CUDNN RNN V8 API，则通过 get_parameters 函数获取梯度参数数组和步长
  auto [grad_params_arr, grad_params_stride0] = get_parameters(
      handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, dw);
#else
  // 如果定义了使用 CUDNN RNN V8 API，则通过 get_parameters 函数获取梯度参数数组和步长
  auto [grad_params_arr, grad_params_stride0] =
      get_parameters(handle, fn.rnn, descs.rnn_desc, dw);
#endif

  // 检查梯度参数步长是否与权重步长相匹配
  if (grad_params_stride0 == static_cast<size_t>(weight_stride0)) {
    // 如果匹配，则调用 _viewParams 函数进行参数视图转换，并返回梯度参数数组
    _viewParams(
        MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
        MatrixRef<Tensor>{weight_arr, static_cast<size_t>(weight_stride0)});
    return grad_params_arr;
  } else {
    // 如果不匹配，则创建一个存储梯度权重的向量数组，并调用 _copyParams 函数进行参数复制
    std::vector<Tensor> grad_weight_arr;
    grad_weight_arr.reserve(weight.numel());
    for (const auto& w : weight_arr) {
      grad_weight_arr.emplace_back(at::empty(w.sizes(), w.options()));
    }
    _copyParams(
        MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
        MatrixRef<Tensor>{
            grad_weight_arr, static_cast<size_t>(weight_stride0)});
    return grad_weight_arr;
  }
}

// 由于 _cudnn_rnn_backward_weight 与 _cudnn_rnn_backward_input 有严格的顺序要求，因此需要此调度程序
// 定义 _cudnn_rnn_backward 函数，用于反向传播 RNN 的计算
std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input,
    TensorList weight,
    int64_t weight_stride0,
    const Tensor& weight_buf,
    const Tensor& hx,
    const std::optional<Tensor>& cx_opt,
    const Tensor& output,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    IntArrayRef batch_sizes,
    const std::optional<Tensor>& dropout_state_opt,
    const Tensor& reserve,
    // 使用 std::array<bool, 4> 来作为输出掩码参数，控制函数输出的内容
    std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
        const Tensor& input,
        const std::vector<Tensor>& weight_buf,
        const c10::optional<Tensor>& hx_opt,
        const c10::optional<Tensor>& cx_opt,
        const Tensor& output,
        const std::vector<Tensor>& grad_output_r_opt,
        const std::vector<Tensor>& grad_hy_r_opt,
        const std::vector<Tensor>& grad_cy_r_opt,
        bool train,
        int64_t mode,
        int64_t hidden_size,
        int64_t proj_size,
        int64_t num_layers,
        bool batch_first,
        double dropout,
        bool bidirectional,
        const std::vector<int64_t>& batch_sizes,
        const c10::optional<Tensor>& dropout_state_opt,
        const c10::optional<Tensor>& reserve,
        std::array<bool, 4> output_mask) {
    
      // 从可选的张量中借用数据，确保获取到有效的 cx 张量
      c10::MaybeOwned<Tensor> cx_maybe_owned =
          at::borrow_from_optional_tensor(cx_opt);
      const Tensor& cx = *cx_maybe_owned;
    
      // 获取梯度张量的值或者创建一个空的张量作为默认值
      const Tensor& grad_output_r =
          c10::value_or_else(grad_output_r_opt, [] { return Tensor(); });
      const Tensor& grad_hy_r =
          c10::value_or_else(grad_hy_r_opt, [] { return Tensor(); });
      const Tensor& grad_cy_r =
          c10::value_or_else(grad_cy_r_opt, [] { return Tensor(); });
      const Tensor& dropout_state =
          c10::value_or_else(dropout_state_opt, [] { return Tensor(); });
    
      // 如果梯度张量都未定义，则返回一个空的元组
      if (!grad_output_r.defined() && !grad_hy_r.defined() &&
          !grad_cy_r.defined()) {
        return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>(
            Tensor(), Tensor(), Tensor(), std::vector<Tensor>(weight.size()));
      }
    
      // 根据条件是否定义梯度张量来创建对应的梯度张量
      auto grad_output = grad_output_r.defined()
          ? grad_output_r
          : at::zeros_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto grad_hy = grad_hy_r.defined()
          ? grad_hy_r
          : at::zeros_like(hx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto grad_cy = cx.defined()
          ? (grad_cy_r.defined()
                 ? grad_cy_r
                 : at::zeros_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
          : grad_cy_r;
    
      // 无条件计算此梯度，因为它会改变 reserve
      auto [dx, dhx, dcx] = at::native::_cudnn_rnn_backward_input(
          input,
          weight_buf,
          hx,
          cx,
          output,
          grad_output,
          grad_hy,
          grad_cy,
          mode,
          hidden_size,
          proj_size,
          num_layers,
          batch_first,
          dropout,
          train,
          bidirectional,
          batch_sizes,
          dropout_state,
          reserve,
          {output_mask[0], output_mask[1], output_mask[2]});
    
      std::vector<Tensor> dw;
      // 如果输出掩码的第四位为真，则计算权重的梯度
      if (output_mask[3]) {
        dw = at::native::_cudnn_rnn_backward_weight(
            input,
            weight,
            weight_stride0,
            weight_buf,
            hx,
            cx,
            output,
            mode,
            hidden_size,
            proj_size,
            num_layers,
            batch_first,
            dropout,
            train,
            bidirectional,
            batch_sizes,
            dropout_state,
            reserve);
      }
      
      // 返回包含梯度张量的元组
      return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>{
          dx, dhx, dcx, dw};
    }
}

// TODO: 我不确定是否实际上需要 'dropout' 和 'train' 参数
// 来初始化状态张量
//
// 注意：在这种情况下，你可以有任何颜色，只要它是 CUDA 字节张量。
// 为什么这个函数即使是在这种情况下也需要一个 TensorOptions？
// 这是一个工厂函数：它生成张量但不接受张量作为输入。
// 代码生成器当前假设所有工厂函数都接受 TensorOptions，所以这个函数也应该这样做更容易绑定。
Tensor _cudnn_init_dropout_state(
    double dropout,
    bool train,
    int64_t dropout_seed,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 见 [注释：用于 TensorOptions 的 hacky wrapper 去除]
  // 使用指定的 dtype、layout、device 和 pin_memory 创建 TensorOptions 对象
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  // 获取当前的 cuDNN 句柄
  auto handle = getCudnnHandle();
  // 创建 DropoutDescriptor 对象
  DropoutDescriptor dropout_desc;
  // 根据训练模式选择是否使用 dropout
  auto dropout_p = train ? dropout : 0;
  // 使用指定参数初始化 dropout 描述符
  dropout_desc.initialize_rng(handle, dropout_p, dropout_seed, options);
  // 返回 dropout 描述符中的状态张量
  return dropout_desc.state;
}

////////////////////////////////////////////////////////////////////////////////
// 用于通用 RNN 操作的 CUDA 分发 (at::lstm, at::gru, ...)
////////////////////////////////////////////////////////////////////////////////

namespace {

// 用于处理不同隐藏类型的帮助函数。
std::tuple<Tensor, Tensor> unpack_hidden(const Tensor& hidden) {
  // 返回一个包含原隐藏张量和空张量的元组
  return std::make_tuple(hidden, at::Tensor{});
}

// 解包含有两个张量的元组的隐藏状态
std::tuple<Tensor, Tensor> unpack_hidden(
    const std::tuple<Tensor, Tensor>& hidden) {
  return hidden;
}

// 打包隐藏状态，针对特定的隐藏类型
template <typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  static_assert(
      std::is_same<hidden_type, void>::value,
      "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

// 实现针对 Tensor 类型的隐藏状态打包
template <>
Tensor pack_hidden<Tensor>(const Tensor& hx, const Tensor& cx) {
  // 断言条件：cx 张量的元素数应为 0
  AT_ASSERT(cx.numel() == 0);
  // 返回隐藏状态的 hx 张量
  return hx;
}

// 实现针对包含两个张量的元组类型的隐藏状态打包
template <>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(
    const Tensor& hx,
    const Tensor& cx) {
  // 返回一个元组，包含给定的 hx 和 cx 张量
  return std::make_tuple(hx, cx);
}
/**
 * Note [DropoutState and CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * (1) Telling a capturing stream to wait on an event recorded in a non-capturing stream is an error.
 * (2) Telling a non-capturing stream to wait on an event recorded during capture is also an error.
 *
 * So DropoutState's usage syncs could error if an RNN with dropout is called in an uncaptured region
 * then called in a captured region (triggering 1), or called in a captured region then called
 * in an uncaptured region (triggering 2).
 *
 * To prevent 1 and 2, lock() only syncs on the last usage event if it was recorded in the same
 * capture state as the current state (which also means the same graph, if capture is in progress).
 *
 * The solution should be safe as long as capture obeys the following restrictions:
 *  - Only one capture may be underway at a time in a given process.
 *  - While a capture is underway, no calls to eager ops on noncapturing streams (on any thread)
 *    may interleave with the captured ops.
 *
 * TODO: As people experiment with capture, keep an eye out for use cases that might need to
 * relax those restrictions.
 *
 * See https://github.com/pytorch/pytorch/pull/56433 for more discussion.
 */

struct DropoutState {
  // Both buffer and event are lazily instantiated when a dropout state is
  // needed for the first time. Note that in this case needed != used, as we
  // don't need a buffer to e.g. run RNNs in test mode.
  at::Tensor buffer;  // Buffer for storing dropout state information
  std::optional<cuda::CUDAEvent> event;  // Optional CUDA event for synchronization
  std::mutex mutex;  // Mutex for thread safety

#if !defined(USE_ROCM)
  // cudaStreamGetCaptureInfo will never give back a capture id of 0, so 0 can
  // serve as a sentinel value that capture was not underway.
  cuda::CaptureId_t capture_id_last_lock = 0;  // Last capture ID for locking
  cuda::CaptureId_t capture_id_last_unlock = 0;  // Last capture ID for unlocking
#endif

  // Every time we use a dropout state, we need to synchronize with its event,
  // to make sure all previous uses finish running before this one starts. Once
  // we're done, we record the event to allow others to synchronize with this
  // kernel. Those events are really needed only for inter-stream sync on a
  // single GPU. I doubt anyone will want to run cuDNN RNNs in parallel on a
  // single GPU, so they should end up being complete no-ops.
  void lock() {
    // Acquire the mutex to ensure thread safety
    mutex.lock();
    
    // Check if the event is defined
    if (event) {
#if !defined(USE_ROCM)
      // See Note [DropoutState and CUDA graph capture]
      // Get the capture status and ID of the current CUDA stream
      cudaStreamCaptureStatus status;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
          cuda::getCurrentCUDAStream(), &status, &capture_id_last_lock));
      
      // Reset capture ID if no capture is underway
      if (status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        capture_id_last_lock = 0;
      }
      
      // Synchronize if the current lock ID matches the last unlock ID
      if (capture_id_last_lock == capture_id_last_unlock) {
        event->block(cuda::getCurrentCUDAStream());
      }
#else
      // Block the CUDA stream until the event is recorded
      event->block(cuda::getCurrentCUDAStream());
#endif
    }
  }
#endif
    }
  }

  void unlock() {
    if (event) {
      event->record();
#if !defined(USE_ROCM)
      // See Note [DropoutState and CUDA graph capture]
      // 查询当前 CUDA 流的捕获状态和捕获 ID
      cudaStreamCaptureStatus status;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(
          cuda::getCurrentCUDAStream(), &status, &capture_id_last_unlock));
      // 如果没有捕获状态，将上次解锁的捕获 ID 设为 0
      if (status == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        capture_id_last_unlock = 0;
      }
      // 检查上次解锁时的捕获 ID 是否与上次锁定时一致
      TORCH_INTERNAL_ASSERT(capture_id_last_unlock == capture_id_last_lock);
#endif
    }
    // 解锁互斥锁
    mutex.unlock();
  }
};

DropoutState& get_dropout_state(
    double dropout_p,
    bool train,
    TensorOptions options) {
  // 每个状态略大于 2MB，延迟初始化，因此可以缓存它们
  static std::vector<DropoutState> dropout_state_cache{
      static_cast<size_t>(cuda::getNumGPUs())};
  static std::mutex state_cache_mut;

  AT_ASSERT(options.device().is_cuda());
  auto device = options.device().index();

  std::unique_lock<std::mutex> lock{state_cache_mut};
  auto& state = dropout_state_cache.at(device);
  if (train && dropout_p > 0) {
    const auto& gen =
        at::detail::getCUDAHooks().getDefaultCUDAGenerator(device);
    auto gen_impl = gen.get<at::CUDAGeneratorImpl>();
    // 重置 RNN 状态的生成器状态
    bool reset_rnn_state = gen_impl->reset_rnn_state();
    // 如果缓冲区未定义或需要重置 RNN 状态，则重新初始化
    if (!state.buffer.defined() || reset_rnn_state) {
      std::unique_lock<std::mutex> lock{state.mutex};
      // 生成种子并用于初始化缓冲区
      int64_t seed =
          at::empty({}, options.dtype(at::kLong)).random_(gen).item<int64_t>();
      state.buffer = at::_cudnn_init_dropout_state(
          dropout_p, train, seed, options.dtype(at::kByte));
      // 注意：CUDA 在创建时将事件绑定到设备，所以只能在确定正确设备时初始化
      if (!state.event.has_value()) {
        state.event.emplace();
      }
    }
  }
  return state;
}

Tensor try_get_weight_buf(
    const Tensor& input,
    TensorList parameters,
    bool has_biases,
    cudnnRNNMode_t mode,
    c10::SymInt hidden_size,
    c10::SymInt proj_size,
    int64_t num_layers,
    bool bidirectional) {
  // 准备所有相关描述符
  auto handle = getCudnnHandle();
  auto& any_param = parameters.at(0);
  auto datatype = getCudnnDataType(any_param);

  // Something very naughty is happening here.  try_get_weight_buf
  // is called from _cudnn_impl, which is a *composite*.  In other words,
  // inside the composite function we need to query cudnn to figure out how big
  // the weight buf actually is going to be.  This clearly cannot be done
  // symbolically.  For now, we insert guards here; but once we have the black
  // box handling for dynamic shapes, we could also hypothetically infer out
  // the relationships
  RNNDescriptorParams rnn;


注释：
#ifndef USE_CUDNN_RNN_V8_API
  // 如果没有定义 USE_CUDNN_RNN_V8_API 宏，则使用老版本的设置方法
  rnn.set(
      mode,
      hidden_size.guard_int(__FILE__, __LINE__),  // 设置隐藏状态的大小，使用文件名和行号进行范围检查
      proj_size.guard_int(__FILE__, __LINE__),    // 设置投影大小，使用文件名和行号进行范围检查
      num_layers,
      bidirectional,
      promote_rnn_math_type(datatype),  // 推广 RNN 的数学类型
      datatype);                       // 设置数据类型
#else
  // 否则，使用新版本的设置方法
  auto cudnn_input_size = input.size(-1);  // 获取输入的最后一个维度大小
  auto packed = false;  // 指示输入是否打包，这里仅仅是假设，不影响权重
  rnn.set(
      mode,
      cudnn_input_size,
      packed,
      hidden_size.guard_int(__FILE__, __LINE__),  // 设置隐藏状态大小，使用文件名和行号进行范围检查
      proj_size.guard_int(__FILE__, __LINE__),    // 设置投影大小，使用文件名和行号进行范围检查
      num_layers,
      bidirectional,
      promote_rnn_math_type(datatype),  // 推广 RNN 的数学类型
      datatype);                       // 设置数据类型
#endif

// 获取 RNN 描述符
RNNDescriptor rnn_desc = rnn.descriptor(handle);

// 设置输入张量的几何信息
TensorGeometry x_geom({1, input.sym_size(-1).guard_int(__FILE__, __LINE__)});  // 创建输入张量的几何信息，使用文件名和行号进行范围检查
TensorDescriptor x_desc;
// x_desc 的数据类型来自 any_param，而不是 input。
// try_get_weight_buf 的工作是检查“权重缓冲区是否正确布局，以便使用相同数据类型的输入运行它？”
x_desc.set(datatype, x_geom.sizes(), x_geom.strides(), 5);  // 设置张量描述符的数据类型、大小和步幅

#ifndef USE_CUDNN_RNN_V8_API
  // 如果没有定义 USE_CUDNN_RNN_V8_API 宏，则使用老版本的获取权重数量方法
  auto num_params = get_num_weights(handle, rnn_desc, x_desc, datatype);  // 获取权重的数量，使用 RNN 描述符和张量描述符
#else
  // 否则，使用新版本的获取权重数量方法
  auto num_params = get_num_weights(handle, rnn_desc, datatype);  // 获取权重的数量，使用 RNN 描述符
#endif

// 尝试获取参数存储
auto param_storage = any_param.storage();
auto weight_buf = at::empty({0}, any_param.options()).set_(param_storage);
if (weight_buf.size(0) < num_params) {
  return {};  // 如果权重缓冲区大小小于权重数量，则返回空结果
} else if (weight_buf.size(0) > num_params) {
  weight_buf = weight_buf.narrow(0, 0, num_params);  // 如果权重缓冲区大小大于权重数量，则裁剪到正确的大小
}

// 获取并检查数据指针
auto expected_data_ptrs = get_expected_data_ptrs(
    weight_buf, handle, rnn, rnn_desc, x_desc, datatype);  // 获取期望的数据指针集合

int64_t num_parameters = parameters.size();
int64_t num_ptrs = expected_data_ptrs.size();
if (proj_size != 0) {
  // 如果有投影层
  AT_ASSERT(num_parameters % (has_biases ? 5 : 3) == 0);  // 断言参数数量符合有/无偏置的条件
  AT_ASSERT(num_ptrs % 5 == 0);  // 断言数据指针数量可以被5整除
  if (has_biases) {
    AT_ASSERT(num_ptrs == num_parameters);  // 断言数据指针数量与参数数量相等
    for (const auto i : c10::irange(num_parameters)) {
      if (expected_data_ptrs[i] != parameters[i].data_ptr())
        return {};  // 如果期望的数据指针与参数的数据指针不匹配，则返回空结果
    }
  } else {
    AT_ASSERT(num_parameters % 3 == 0);  // 断言参数数量可以被3整除
    AT_ASSERT(num_ptrs == num_parameters * 5 / 3);  // 断言数据指针数量符合无偏置情况下的计算
    for (int64_t param_i = 0, ptr_i = 0; ptr_i < num_ptrs;
         ptr_i += 5, param_i += 3) {
      if (expected_data_ptrs[ptr_i] != parameters[param_i].data_ptr())
        return {};  // 如果期望的数据指针与参数的数据指针不匹配，则返回空结果
      if (expected_data_ptrs[ptr_i + 1] != parameters[param_i + 1].data_ptr())
        return {};  // 如果期望的数据指针与参数的数据指针不匹配，则返回空结果
      if (expected_data_ptrs[ptr_i + 4] != parameters[param_i + 2].data_ptr())
        return {};  // 如果期望的数据指针与参数的数据指针不匹配，则返回空结果
    }
  }
} else {
  // 如果没有投影层
  AT_ASSERT(num_ptrs == (num_parameters * (has_biases ? 1 : 2)));  // 断言数据指针数量符合有/无偏置的条件
  AT_ASSERT(num_parameters % (has_biases ? 4 : 2) == 0);  // 断言参数数量符合有/无偏置的条件
    // 循环遍历参数指针和预期数据指针数组，每次迭代更新 param_i 和 ptr_i
    for (int64_t param_i = 0, ptr_i = 0; ptr_i < num_ptrs;
         ptr_i += (has_biases ? 2 : 4), param_i += 2) {
      // 检查当前参数的数据指针是否与预期的数据指针相符，如果不符则返回空字典
      if (expected_data_ptrs[ptr_i] != parameters[param_i].data_ptr())
        return {};
      // 如果存在偏置项，再次检查下一个参数的数据指针是否与预期的数据指针相符，如果不符则返回空字典
      if (expected_data_ptrs[ptr_i + 1] != parameters[param_i + 1].data_ptr())
        return {};
    }
  }
  // 检查最后一个参数是否是连续存储的，如果不是则返回空字典
  if (!parameters[num_parameters - 1].is_contiguous())
    return {};
  // 返回权重缓冲区
  return weight_buf;
  // 模板函数定义，使用隐藏类型 hidden_type 作为模板参数，返回 Tensor 和 hidden_type 的 pair 对象
  std::pair<Tensor, hidden_type> _cudnn_impl(
      // 输入张量 input
      const Tensor& input,
      // 批量大小张量 _batch_sizes
      const Tensor& _batch_sizes,
      // 隐藏状态 hidden
      const hidden_type& hidden,
      // 参数列表 params
      TensorList params,
      // 是否有偏置参数 has_biases
      bool has_biases,
      // cuDNN 循环神经网络模式 mode
      cudnnRNNMode_t mode,
      // 层数 num_layers
      int64_t num_layers,
      // 丢弃率 dropout_p
      double dropout_p,
      // 是否训练 train
      bool train,
      // 是否双向 bidirectional
      bool bidirectional) {
    // 解包隐藏状态 hx 和 cx
    auto [hx, cx] = unpack_hidden(hidden);
    // 初始化隐藏层大小为 hx 的第二维度大小
    auto hidden_size = hx.sym_size(2);
    // 初始化投影大小为 0
    c10::SymInt proj_size = 0;
    // 对于具有投影的 LSTM 模型，隐藏大小可能不同
    if (cx.defined() && cx.sym_size(2) != hx.sym_size(2)) {
      // 如果定义了 cx 并且其第二维度大小不等于 hx 的第二维度大小，则更新隐藏大小为 cx 的第二维度大小
      hidden_size = cx.sym_size(2);
      // 更新投影大小为 hx 的第二维度大小
      proj_size = hx.sym_size(2);
    }

    // 获取权重缓冲区，try_get_weight_buf 返回一个 Tensor，但 _cudnn_rnn 需要的是 std::optional<Tensor> 类型
    at::cuda::OptionalCUDAGuard guard(input.get_device());
    auto weight_buf = try_get_weight_buf(
        input,
        params,
        has_biases,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        bidirectional);

    // 检查 batch_sizes 张量是否为 1 维
    TORCH_CHECK(_batch_sizes.dim() == 1, "batch_sizes tensor should be 1D");
    // 将 batch_sizes 转换为 IntArrayRef
    IntArrayRef batch_sizes{
        _batch_sizes.data_ptr<int64_t>(),
        static_cast<size_t>(_batch_sizes.size(0))};

    // 获取丢弃状态对象的引用
    auto& dropout_state = get_dropout_state(dropout_p, train, input.options());
    // 使用互斥锁 lock，锁定丢弃状态对象
    std::unique_lock<DropoutState> lock{dropout_state};
    // 初始化参数数量，包括偏置参数
    int64_t num_params = has_biases ? 4 : 2;
    // 如果有投影，则参数数量加一
    if (proj_size != 0) {
      ++num_params;
    }
    // 使用 c10::SymIntArrayRef 封装符号化的 batch_sizes
    auto sym_batch_sizes = c10::SymIntArrayRef(
        reinterpret_cast<const c10::SymInt*>(batch_sizes.data()),
        batch_sizes.size());
    // 调用 _cudnn_rnn_symint 函数进行 cuDNN RNN 操作，返回 std::tuple<output, hy, cy, reserve, new_weight_buf>
    auto cudnn_output = at::_cudnn_rnn_symint(
        input,
        params,
        num_params,
        weight_buf,
        hx,
        cx,
        static_cast<int>(mode),
        hidden_size,
        proj_size,
        num_layers,
        /*batch_first=*/false,
        dropout_p,
        train,
        bidirectional,
        sym_batch_sizes,
        dropout_state.buffer);

    // 返回结果为 output 和打包后的隐藏状态 <hidden_type>
    return {
        std::get<0>(cudnn_output),
        pack_hidden<hidden_type>(
            std::get<1>(cudnn_output), std::get<2>(cudnn_output))};
  }
  // 计算投影大小为 2 的符号量，并将结果赋给 proj_size
  proj_size = hx.sym_size(2);
}
// 如果 input 在 CUDA 设备上，则创建一个 CUDA 设备守卫对象
at::cuda::OptionalCUDAGuard guard(input.get_device());
// 尝试获取权重缓冲区，该缓冲区用于存储权重和偏置参数
auto weight_buf = try_get_weight_buf(
    input,
    params,
    has_biases,
    mode,
    hidden_size,
    proj_size,
    num_layers,
    bidirectional);
// 获取 dropout 的状态对象并加锁
auto& dropout_state = get_dropout_state(dropout_p, train, input.options());
std::unique_lock<DropoutState> lock{dropout_state};
// 计算参数的数量，如果有偏置，则为 4，否则为 2；如果有投影大小，则再加 1
int64_t num_params = has_biases ? 4 : 2;
if (proj_size != 0) {
  ++num_params;
}
// 调用 cudnn 函数进行 RNN 计算，返回一个包含输出、隐藏状态及新的权重缓冲区的元组
// 这里的 cudnn_output 是一个 std::tuple<output, hy, cy, reserve, new_weight_buf>
auto cudnn_output = at::_cudnn_rnn_symint(
    input,
    params,
    num_params,
    weight_buf,
    hx,
    cx,
    static_cast<int>(mode),
    hidden_size,
    proj_size,
    num_layers,
    batch_first,
    dropout_p,
    train,
    bidirectional,
    /*batch_sizes=*/{},
    dropout_state.buffer);

// 返回一个包含输出和打包后的隐藏状态的元组
return {
    std::get<0>(cudnn_output), // 输出
    pack_hidden<hidden_type>( // 打包后的隐藏状态
        std::get<1>(cudnn_output), std::get<2>(cudnn_output))};
# 定义宏函数，用于生成包含 RNN 的隐藏层实现函数
#define ONE_HIDDEN_RNN(NAME, MODE)                          \
  # 创建一个函数 NAME##_cudnn，接受多个参数，包括输入张量 output、输出张量 hy、输入张量 input、初始隐藏状态张量 hx、参数列表 params、是否有偏置 has_biases、层数 num_layers、dropout 概率 dropout_p、训练标志 train、是否双向标志 bidirectional、是否批处理第一标志 batch_first \
  void NAME##_cudnn(                                        \
      Tensor& output,                                       \
      Tensor& hy,                                           \
      const Tensor& input,                                  \
      const Tensor& hx,                                     \
      TensorList params,                                    \
      bool has_biases,                                      \
      int64_t num_layers,                                   \
      double dropout_p,                                     \
      bool train,                                           \
      bool bidirectional,                                   \
      bool batch_first) {                                   \
    # 调用内部函数 _cudnn_impl，返回 output 和 hy \
    std::tie(output, hy) = _cudnn_impl(                     \
        input,                                              \
        hx,                                                 \
        params,                                             \
        has_biases,                                         \
        MODE,                                               \
        num_layers,                                         \
        dropout_p,                                          \
        train,                                              \
        bidirectional,                                      \
        batch_first);                                       \
  }                                                         \
                                                            \
  # 创建一个函数 NAME##_packed_cudnn，接受多个参数，包括输出张量 output、输出张量 hy、数据张量 data、批次大小张量 batch_sizes、初始隐藏状态张量 hx、参数列表 params、是否有偏置 has_biases、层数 num_layers、dropout 概率 dropout_p、训练标志 train、是否双向标志 bidirectional \
  void NAME##_packed_cudnn(                                 \
      Tensor& output,                                       \
      Tensor& hy,                                           \
      const Tensor& data,                                   \
      const Tensor& batch_sizes,                            \
      const Tensor& hx,                                     \
      TensorList params,                                    \
      bool has_biases,                                      \
      int64_t num_layers,                                   \
      double dropout_p,                                     \
      bool train,                                           \
      bool bidirectional) {                                 \
    # 将 _cudnn_impl 函数的返回值解包为 output 和 hy 变量
    std::tie(output, hy) = _cudnn_impl(                     \
        data,                                               \
        batch_sizes,                                        \
        hx,                                                 \
        params,                                             \
        has_biases,                                         \
        MODE,                                               \
        num_layers,                                         \
        dropout_p,                                          \
        train,                                              \
        bidirectional);                                     \
    # 注册 NAME##_cudnn_stub 函数作为 NAME##_cudnn 的 CUDA 分发函数
    REGISTER_CUDA_DISPATCH(NAME##_cudnn_stub, &NAME##_cudnn); \
    # 注册 NAME##_packed_cudnn_stub 函数作为 NAME##_packed_cudnn 的 CUDA 分发函数
    REGISTER_CUDA_DISPATCH(NAME##_packed_cudnn_stub, &NAME##_packed_cudnn);
// 定义宏ONE_HIDDEN_RNN，用于注册一个隐藏层的循环神经网络（RNN）类型，如GRU
ONE_HIDDEN_RNN(gru, CUDNN_GRU)
// 定义宏ONE_HIDDEN_RNN，用于注册一个隐藏层的RNN类型，如使用tanh作为激活函数的RNN
ONE_HIDDEN_RNN(rnn_tanh, CUDNN_RNN_TANH)
// 定义宏ONE_HIDDEN_RNN，用于注册一个隐藏层的RNN类型，如使用ReLU作为激活函数的RNN
ONE_HIDDEN_RNN(rnn_relu, CUDNN_RNN_RELU)

// 定义函数lstm_cudnn，实现基于CUDNN的LSTM操作
void lstm_cudnn(
    Tensor& output,
    Tensor& hy,
    Tensor& cy,
    const Tensor& input,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  // 调用_cudnn_impl函数执行CUDNN的LSTM操作，并将结果存储在output、hy和cy中
  auto result = _cudnn_impl(
      input,
      std::make_tuple(hx[0], hx[1]),
      params,
      has_biases,
      CUDNN_LSTM,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);
  // 将_cudnn_impl返回的输出结果存入output张量
  output = result.first;
  // 将_cudnn_impl返回的隐藏状态输出存入hy张量
  hy = std::get<0>(result.second);
  // 将_cudnn_impl返回的细胞状态输出存入cy张量
  cy = std::get<1>(result.second);
}

// 定义函数lstm_packed_cudnn，实现基于CUDNN的打包LSTM操作
void lstm_packed_cudnn(
    Tensor& output,
    Tensor& hy,
    Tensor& cy,
    const Tensor& data,
    const Tensor& batch_sizes,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  // 调用_cudnn_impl函数执行打包格式的CUDNN LSTM操作，并将结果存储在output、hy和cy中
  auto result = _cudnn_impl(
      data,
      batch_sizes,
      std::make_tuple(hx[0], hx[1]),
      params,
      has_biases,
      CUDNN_LSTM,
      num_layers,
      dropout_p,
      train,
      bidirectional);
  // 将_cudnn_impl返回的输出结果存入output张量
  output = result.first;
  // 将_cudnn_impl返回的隐藏状态输出存入hy张量
  hy = std::get<0>(result.second);
  // 将_cudnn_impl返回的细胞状态输出存入cy张量
  cy = std::get<1>(result.second);
}

// 注册CUDA分发，将lstm_cudnn_stub函数与lstm_cudnn函数关联
REGISTER_CUDA_DISPATCH(lstm_cudnn_stub, &lstm_cudnn);
// 注册CUDA分发，将lstm_packed_cudnn_stub函数与lstm_packed_cudnn函数关联
REGISTER_CUDA_DISPATCH(lstm_packed_cudnn_stub, &lstm_packed_cudnn);

// 结束at命名空间
} // namespace

// 结束at命名空间
} // namespace at

// 结束条件编译指令，检查是否启用了CUDNN
#endif // AT_CUDNN_ENABLED()
```