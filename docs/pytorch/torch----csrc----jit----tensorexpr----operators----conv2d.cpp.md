# `.\pytorch\torch\csrc\jit\tensorexpr\operators\conv2d.cpp`

```py
// 引入 ATen 库的配置文件
#include <ATen/Config.h>
// 引入 JIT 日志模块
#include <torch/csrc/jit/jit_log.h>
// 引入 TensorExpr 库的循环嵌套模块
#include <torch/csrc/jit/tensorexpr/loopnest.h>
// 引入 TensorExpr 库的卷积运算模块
#include <torch/csrc/jit/tensorexpr/operators/conv2d.h>
// 引入 TensorExpr 库的其它运算模块
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
// 引入 TensorExpr 库的张量表示模块
#include <torch/csrc/jit/tensorexpr/tensor.h>

// Torch 根命名空间
namespace torch {
// JIT 根命名空间
namespace jit {
// TensorExpr 子命名空间
namespace tensorexpr {

// 匿名命名空间，定义了一些局部函数和类
namespace {

// 检查缓冲区维度是否为常量的函数
void assert_dims_constant(const BufHandle& buf) {
  // 遍历缓冲区的维度，确保每个维度都是常量
  for (auto const& dim : buf.node()->dims()) {
    TORCH_INTERNAL_ASSERT(dim->isConstant());
  }
}

// 定义初始化函数类型
using InitFunc = std::function<ExprHandle(const std::vector<VarHandle>&)>;

// 静态深度可分离卷积函数，接受输入缓冲区、权重缓冲区、初始化函数及相关参数
Tensor conv2d_depthwise_static(
    BufHandle input,
    BufHandle weight,
    const InitFunc& init_func,
    int stride,
    int pad,
    int groups) {
  // 断言输入缓冲区的维度为四维
  TORCH_INTERNAL_ASSERT(input.ndim() == 4);
  // 断言权重缓冲区的维度为四维
  TORCH_INTERNAL_ASSERT(weight.ndim() == 4);

  // 调用局部函数检查输入和权重的维度是否为常量
  assert_dims_constant(input);
  assert_dims_constant(weight);

  // 提取输入缓冲区的维度值为常量引用
  auto const& N = immediateAs<int>(input.dim(0));
  auto const& C = immediateAs<int>(input.dim(1));
  auto const& H = immediateAs<int>(input.dim(2));
  auto const& W = immediateAs<int>(input.dim(3));

  // 提取权重缓冲区的维度值为常量引用
  auto const& K = immediateAs<int>(weight.dim(0));
  auto const& CperG = immediateAs<int>(weight.dim(1));
  auto const& R = immediateAs<int>(weight.dim(2));
  auto const& S = immediateAs<int>(weight.dim(3));

  // 断言输入通道数等于输出通道数等于分组数等于权重通道数
  TORCH_INTERNAL_ASSERT(C == K && K == groups && CperG == 1);
  // 断言权重的高度等于宽度
  TORCH_INTERNAL_ASSERT(R == S);

  // 计算输出张量的高度和宽度
  auto OH = (H - R + 2 * pad) / stride + 1;
  auto OW = (W - S + 2 * pad) / stride + 1;

  // 定义卷积操作的张量对象
  Tensor conv = Reduce(
      "conv2d_depthwise", // 减少操作的名称
      {N, K, OH, OW},     // 输出张量的维度
      c10::nullopt,       // TODO: 需要进一步补充
      Sum(),              // 使用求和作为减少操作的运算符
      [&](const std::vector<VarHandle>& v) { return init_func(v); }, // 初始化函数
      [&](const std::vector<VarHandle>& v) {
        // 定义内部函数，计算卷积操作的每个元素
        auto const& n = v[0];
        auto const& k = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        auto const& c = v[4];
        auto const& r = v[5];
        auto const& s = v[6];
        // 创建条件表达式，判断当前卷积元素是否在输入边界内
        auto cond = CompareSelect::make(oh * stride - pad + r, 0, 1, 0, kLT);
        cond = CompareSelect::make(ow * stride - pad + s, 0, 1, cond, kLT);
        cond = CompareSelect::make(oh * stride - pad + r, H, 1, cond, kGE);
        cond = CompareSelect::make(ow * stride - pad + s, W, 1, cond, kGE);
        // 如果在边界内，从输入缓冲区加载数据；否则置零
        auto in = ifThenElse(
            cond,
            0.f,
            input.load(n, k, oh * stride - pad + r, ow * stride - pad + s));
        // 返回卷积操作的结果
        return in * weight.load(k, c, r, s);
      },
      {C / groups, R, S}); // 操作的循环维度

  // 创建循环嵌套对象
  LoopNest nest({conv});

  // 如果卷积核的高度是3，步长是2，填充是1
  constexpr int kLoopH = 2, kLoopW = 3;
  if (R == 3 && stride == 2 && pad == 1) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr head, tail;
    auto loops = nest.getLoopStmtsFor(conv);
    nest.sliceHead(loops[kLoopW], 2, &head, &tail);
    loops = nest.getLoopStmtsFor(conv);
    nest.sliceHead(loops[kLoopH], 2, &head, &tail);
  } else if (R == 3 && stride == 1 && pad == 1) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr main, peeled;
    # 获取所有写入 conv.buf() 的循环嵌套结构
    auto loops = nest.getAllLoopNestsWritingToBuf(conv.buf());
    # 从第二个循环获取主循环
    main = loops[1][kLoopW];
    # 从主循环中剥离第一个维度，更新 peeled 和 main
    nest.sliceHead(main, 1, &peeled, &main);
    # 从主循环中剥离最后一个维度，更新 main 和 peeled
    nest.sliceTail(main, 1, &main, &peeled);
    # 获取主循环的父循环
    main = LoopNest::getParentLoop(main);
    # 再次从主循环中剥离第一个维度，更新 peeled 和 main
    nest.sliceHead(main, 1, &peeled, &main);
    # 再次从主循环中剥离最后一个维度，更新 main 和 peeled
    nest.sliceTail(main, 1, &main, &peeled);
  }

  # 返回一个 Tensor 对象，使用 conv.buf() 和 nest.root_stmt() 构造
  return Tensor(conv.buf(), nest.root_stmt());
}

// 功能：实现深度可分离二维卷积的动态版本
Tensor conv2d_depthwise_dynamic(
    BufHandle input,            // 输入缓冲区句柄
    BufHandle weight,           // 权重缓冲区句柄
    const InitFunc& init_func,  // 初始化函数对象的引用
    ExprHandle N,               // 批次大小
    ExprHandle C,               // 输入通道数
    ExprHandle H,               // 输入高度
    ExprHandle W,               // 输入宽度
    ExprHandle K,               // 输出通道数（卷积核数）
    ExprHandle CperG,           // 每组通道数
    ExprHandle R,               // 卷积核高度
    ExprHandle S,               // 卷积核宽度
    ExprHandle stride,          // 步长
    ExprHandle pad,             // 填充数
    ExprHandle groups) {        // 组数
  TORCH_INTERNAL_ASSERT(input.ndim() == 4);   // 断言输入张量的维度为4
  TORCH_INTERNAL_ASSERT(weight.ndim() == 4);  // 断言权重张量的维度为4

  // 计算输出的高度和宽度
  auto OH = (H - R + pad * 2) / stride + 1;
  auto OW = (W - S + pad * 2) / stride + 1;

  // 返回一个按照指定维度约简的张量
  return Reduce(
      "conv2d_depthwise",      // 约简操作的名称
      {N, K, OH, OW},          // 输出张量的维度
      c10::nullopt,            // TODO：约简操作的附加选项
      Sum(),                   // 约简操作的函数，这里是求和
      [&](const std::vector<VarHandle>& v) { return init_func(v); },  // 约简初始化函数
      [&](const std::vector<VarHandle>& v) {   // 约简的操作函数
        auto const& n = v[0];
        auto const& k = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        auto const& c = v[4];
        auto const& r = v[5];
        auto const& s = v[6];
        // 构造条件表达式，用于选择计算或忽略某些位置的值
        auto cond = CompareSelect::make(oh * stride - pad + r, 0, 1, 0, kLT);
        cond = CompareSelect::make(ow * stride - pad + s, 0, 1, cond, kLT);
        cond = CompareSelect::make(oh * stride - pad + r, H, 1, cond, kGE);
        cond = CompareSelect::make(ow * stride - pad + s, W, 1, cond, kGE);
        // 条件选择计算值或零值
        auto in = ifThenElse(
            cond,
            0.f,
            input.load(n, k, oh * stride - pad + r, ow * stride - pad + s));
        // 返回计算后的乘积
        return in * weight.load(k, c, r, s);
      },
      {C / groups, R, S});      // 约简操作的轴向，这里是按输入通道数除以组数、卷积核大小进行约简
}

// namespace 结尾

// 功能：实现深度可分离二维卷积的静态版本
Tensor conv2d_depthwise(
    BufHandle input,    // 输入缓冲区句柄
    BufHandle weight,   // 权重缓冲区句柄
    BufHandle bias,     // 偏置缓冲区句柄
    int stride,         // 步长
    int pad,            // 填充数
    int groups) {       // 组数
  assert_dims_constant(bias);  // 断言偏置的维度为常数
  auto init_func = [&](const std::vector<VarHandle>& v) {
    return bias.load(v[1]);   // 初始化函数，加载偏置
  };
  // 调用静态版本的深度可分离卷积函数
  return conv2d_depthwise_static(input, weight, init_func, stride, pad, groups);
}

// 功能：实现深度可分离二维卷积的静态版本
Tensor conv2d_depthwise(
    BufHandle input,    // 输入缓冲区句柄
    BufHandle weight,   // 权重缓冲区句柄
    int stride,         // 步长
    int pad,            // 填充数
    int groups) {       // 组数
  auto init_func = [](const std::vector<VarHandle>& v) {
    return ExprHandle(Sum().initializer());  // 初始化函数，返回求和的初始值
  };
  // 调用静态版本的深度可分离卷积函数
  return conv2d_depthwise_static(input, weight, init_func, stride, pad, groups);
}

// 功能：实现深度可分离二维卷积的动态版本
Tensor conv2d_depthwise(
    BufHandle input,            // 输入缓冲区句柄
    BufHandle weight,           // 权重缓冲区句柄
    BufHandle bias,             // 偏置缓冲区句柄
    ExprHandle N,               // 批次大小
    ExprHandle C,               // 输入通道数
    ExprHandle H,               // 输入高度
    ExprHandle W,               // 输入宽度
    ExprHandle K,               // 输出通道数（卷积核数）
    ExprHandle CperG,           // 每组通道数
    ExprHandle R,               // 卷积核高度
    ExprHandle S,               // 卷积核宽度
    ExprHandle stride,          // 步长
    ExprHandle pad,             // 填充数
    ExprHandle groups) {        // 组数
  assert_dims_constant(bias);  // 断言偏置的维度为常数
  auto init_func = [&](const std::vector<VarHandle>& v) {
    return bias.load(v[1]);   // 初始化函数，加载偏置
  };
  // 调用动态版本的深度可分离卷积函数
  return conv2d_depthwise_dynamic(
      input,
      weight,
      init_func,
      N,
      C,
      H,
      W,
      K,
      CperG,
      R,
      S,
      stride,
      pad,
      groups);
}

// 功能：实现深度可分离二维卷积的动态版本
Tensor conv2d_depthwise(
    BufHandle input,            // 输入缓冲区句柄
    BufHandle weight,           // 权重缓冲区句柄
    ExprHandle N,               // 批次大小
    ExprHandle C,               // 输入通道数
    ExprHandle H,               // 输入高度
    ExprHandle W,               // 输入宽度
    ExprHandle K,               // 输出通道数（卷积核数）
    ExprHandle CperG,           // 每组通道数
    ExprHandle R,               // 卷积核高度
    ExprHandle S,               // 卷积核宽度
    ExprHandle stride,          // 步长
    # 定义一个 lambda 函数 init_func，用于生成 Sum().initializer() 的表达式
    auto init_func = [](const std::vector<VarHandle>& v) {
        return ExprHandle(Sum().initializer());
    };
    # 调用 conv2d_depthwise_dynamic 函数进行深度可分离卷积计算
    return conv2d_depthwise_dynamic(
        input,      # 输入张量
        weight,     # 卷积核张量
        init_func,  # 初始化函数，用于生成累加器的初始化表达式
        N,          # 批次大小
        C,          # 输入通道数
        H,          # 输入高度
        W,          # 输入宽度
        K,          # 输出通道数（卷积核个数）
        CperG,      # 每组的通道数
        R,          # 卷积核高度
        S,          # 卷积核宽度
        stride,     # 步长
        pad,        # 填充方式
        groups      # 分组数
    );
}

// 将参数 v 转换为包含两个 int64_t 元素的向量
static std::vector<int64_t> _pair_int(ArgValue v) {
  // 如果 v 是 IntList 类型的变量指针，则返回其前两个元素构成的向量
  if (auto t = std::get_if<IntList>(&v)) {
    return {(*t)[0], (*t)[1]};
  }
  // 否则将 v 转换为 int64_t 类型，并返回包含两个相同元素的向量
  auto i = std::get<int64_t>(v);
  return {i, i};
}

// 将参数 v 转换为包含一个 int64_t 元素的向量
static std::vector<int64_t> _single_int_list(ArgValue v) {
  // 如果 v 是 IntList 类型的变量指针，则返回其第一个元素构成的向量
  if (auto t = std::get_if<IntList>(&v)) {
    return {(*t)[0]};
  }
  // 否则将 v 转换为 int64_t 类型，并返回包含一个元素的向量
  auto i = std::get<int64_t>(v);
  return {i};
}

// 检查是否支持给定的 Conv2d 操作
bool conv2dIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight,
    const TensorInfo& bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
  // 检查输入张量的数据类型是否为 float32
  if (input.dtype != c10::ScalarType::Float ||
      weight.dtype != c10::ScalarType::Float ||
      bias.dtype != c10::ScalarType::Float) {
    GRAPH_DEBUG("conv2dIsSupported: only float32 allowed");
    return false;
  }
  // 检查输入张量的维度是否正确
  if (input.dims.size() != 4 || weight.dims.size() != 4 ||
      bias.dims.size() != 1) {
    GRAPH_DEBUG("conv2dIsSupported: inputs are the wrong size");
    return false;
  }
  // 获取输入张量的通道数、权重张量的输出通道数、每个组的通道数
  auto Cin = input.dims[1];
  auto Cout = weight.dims[0];
  auto CperG = weight.dims[1];
  // 检查是否为深度卷积
  if (Cin != Cout || Cin != groups || CperG != 1) {
    GRAPH_DEBUG("conv2dIsSupported: not depthwise");
    return false;
  }
  // 获取权重张量的卷积核高度和宽度
  auto KH = weight.dims[2];
  auto KW = weight.dims[3];
  // 检查卷积核是否为 3x3
  if (KH != 3 || KW != 3) {
    GRAPH_DEBUG("conv2dIsSupported: not 3x3");
    return false;
  }
  // 检查步长是否支持
  if (stride.size() != 2 || stride[0] != stride[1]) {
    GRAPH_DEBUG("conv2dIsSupported: unsupported stride");
    return false;
  }
  // 检查填充是否支持
  if (pad.size() != 2 || pad[0] != pad[1]) {
    GRAPH_DEBUG("conv2dIsSupported: unsupported pad");
    return false;
  }
  // 检查扩展是否支持
  if (dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1) {
    GRAPH_DEBUG("conv2dIsSupported: unsupported dilation");
    return false;
  }
  // 如果所有条件都符合，则返回支持
  return true;
}

// 检查是否支持给定的 MKL-DNN 预打包卷积操作
bool mkldnnPrepackedConvIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
  // 如果 MKL-DNN 功能可用
#if AT_MKLDNN_ENABLED()
  // 检查输入张量和权重张量的数据类型是否为 float32
  if (input.dtype != c10::ScalarType::Float ||
      weight.dtype != c10::ScalarType::Float) {
    GRAPH_DEBUG("mkldnnPrepackedConvIsSupported: only float32 allowed");
    return false;
  }
  // 检查输入张量和权重张量的维度是否正确
  if (input.dims.size() != 4 || weight.dims.size() != 4) {
    GRAPH_DEBUG("mkldnnPrepackedConvIsSupported: inputs are the wrong size");
    return false;
  }
  // 检查步长是否支持
  if (stride.size() != 2) {
    GRAPH_DEBUG("mkldnnPrepackedConvIsSupported: unsupported stride");
    return false;
  }
  // 检查填充是否支持
  if (pad.size() != 2) {
    GRAPH_DEBUG("mkldnnPrepackedConvIsSupported: unsupported pad");
    return false;
  }
  // 检查扩展是否支持
  if (dilation.size() != 2) {
    GRAPH_DEBUG("mkldnnPrepackedConvIsSupported: unsupported dilation");
    return false;
  }
#endif
    // 如果不符合条件，返回 false
    return false;
  }

  // 不要为 mkldnn 优于原生实现的情况重写代码
  // 条件来源于：aten/src/ATen/native/Convolution.cpp:use_mkldnn
  // 根据一系列条件判断是否使用 mkldnn
  bool use_mkldnn = groups > 1 || (weight.dims[2] > 3 && weight.dims[3] > 3) ||
      input.dims[0] > 1 ||
      input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3] > 20480;
  // 输出调试信息，显示 mkldnn 是否被支持
  GRAPH_DEBUG("mkldnnPrepackedConvIsSupported: ", use_mkldnn);
  // 返回是否使用 mkldnn 的布尔值
  return use_mkldnn;
#endif
  // 如果条件不满足，返回 false
  return false;
}

// 计算二维卷积操作
Tensor computeConv2d(
    const std::vector<ArgValue>& inputs,                // 输入参数列表
    const std::vector<ExprHandle>& outputShape,         // 输出形状表达式列表
    const std::vector<ExprHandle>& outputStrides,       // 输出步长表达式列表
    const std::optional<ScalarType>& outputType,        // 输出数据类型（可选）
    at::Device device) {                                // 设备类型

  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);                        // 根据输出类型确定数据类型
  }

  BufHandle ResultBuf("conv", outputShape, dtype);      // 创建结果缓冲区
  const BufHandle& inp = std::get<BufHandle>(inputs[0]);   // 获取输入缓冲区
  const BufHandle& w = std::get<BufHandle>(inputs[1]);     // 获取权重缓冲区
  const BufHandle& b = std::get<BufHandle>(inputs[2]);     // 获取偏置缓冲区

  auto strides = _pair_int(inputs[3]);                   // 提取步长信息
  auto padding = _pair_int(inputs[4]);                   // 提取填充信息
  auto dilation = _pair_int(inputs[5]);                  // 提取扩展信息

  int groups = std::get<int64_t>(inputs[6]);             // 提取分组信息

  auto inpInfo = getTensorInfo(inp);                     // 获取输入张量信息
  auto wInfo = getTensorInfo(w);                         // 获取权重张量信息
  auto bInfo = getTensorInfo(b);                         // 获取偏置张量信息

  // 为深度卷积生成计算表达式
  if (inpInfo && wInfo && bInfo &&
      conv2dIsSupported(
          *inpInfo, *wInfo, *bInfo, strides, padding, dilation, groups)) {
    return conv2d_depthwise(inp, w, b, strides[0], padding[0], groups);
  }

  // 一旦我们对卷积操作有了高效的表示形式，可以在此处使用它，而不是外部调用！
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_conv2d",
      {inp, w, b},
      {strides[0],
       strides[1],
       padding[0],
       padding[1],
       dilation[0],
       dilation[1],
       groups});
  return Tensor(ResultBuf.node(), s);
}

// 计算一维卷积操作
Tensor computeConv1d(
    const std::vector<ArgValue>& inputs,                // 输入参数列表
    const std::vector<ExprHandle>& outputShape,         // 输出形状表达式列表
    const std::vector<ExprHandle>& outputStrides,       // 输出步长表达式列表
    const std::optional<ScalarType>& outputType,        // 输出数据类型（可选）
    at::Device device) {                                // 设备类型

  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);                        // 根据输出类型确定数据类型
  }

  BufHandle ResultBuf("conv", outputShape, dtype);      // 创建结果缓冲区
  const BufHandle& inp = std::get<BufHandle>(inputs[0]);   // 获取输入缓冲区
  const BufHandle& w = std::get<BufHandle>(inputs[1]);     // 获取权重缓冲区
  const BufHandle& b = std::get<BufHandle>(inputs[2]);     // 获取偏置缓冲区

  auto strides = _single_int_list(inputs[3]);           // 提取步长信息
  auto padding = _single_int_list(inputs[4]);           // 提取填充信息
  auto dilation = _single_int_list(inputs[5]);          // 提取扩展信息

  int groups = std::get<int64_t>(inputs[6]);             // 提取分组信息

  auto inpInfo = getTensorInfo(inp);                     // 获取输入张量信息
  auto wInfo = getTensorInfo(w);                         // 获取权重张量信息
  auto bInfo = getTensorInfo(b);                         // 获取偏置张量信息

  // 使用外部调用生成卷积计算语句
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_conv1d",
      {inp, w, b},
      {strides[0], padding[0], dilation[0], groups});
  return Tensor(ResultBuf.node(), s);
}

// 计算预打包的二维卷积带限制运行
Tensor computePrepackedConv2dClampRun(
    const std::vector<ArgValue>& inputs,                // 输入参数列表
    const std::vector<ExprHandle>& outputShape,         // 输出形状表达式列表
    const std::vector<ExprHandle>& outputStrides,       // 输出步长表达式列表
    const std::optional<ScalarType>& outputType,        // 输出数据类型（可选）
    at::Device device) {                                // 设备类型

  Dtype dtype = kFloat;
  if (outputType) {

    dtype = Dtype(*outputType);                        // 根据输出类型确定数据类型
  }

  BufHandle ResultBuf("conv", outputShape, dtype);      // 创建结果缓冲区
  const BufHandle& inp = std::get<BufHandle>(inputs[0]);   // 获取输入缓冲区
  const BufHandle& w = std::get<BufHandle>(inputs[1]);     // 获取权重缓冲区
  const BufHandle& b = std::get<BufHandle>(inputs[2]);     // 获取偏置缓冲区

  // 提取单个整数列表作为步长、填充和扩展信息
  auto strides = _single_int_list(inputs[3]);
  auto padding = _single_int_list(inputs[4]);
  auto dilation = _single_int_list(inputs[5]);

  int groups = std::get<int64_t>(inputs[6]);             // 提取分组信息

  auto inpInfo = getTensorInfo(inp);                     // 获取输入张量信息
  auto wInfo = getTensorInfo(w);                         // 获取权重张量信息
  auto bInfo = getTensorInfo(b);                         // 获取偏置张量信息

  // 使用外部调用生成卷积计算语句
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_conv2d_clamp_run",
      {inp, w, b},
      {strides[0], padding[0], dilation[0], groups});
  return Tensor(ResultBuf.node(), s);
}
  # 根据 outputType 创建一个 Dtype 对象，dtype 是其实例化后的结果
  dtype = Dtype(*outputType);

BufHandle ResultBuf("prepacked_conv2d_clamp_run", outputShape, dtype);
  # 创建一个名为 ResultBuf 的 BufHandle 对象，用于存储输出的结果
  # 参数 "prepacked_conv2d_clamp_run" 是该 BufHandle 的名称
  # outputShape 是输出张量的形状
  # dtype 是输出张量的数据类型

const BufHandle& inp = std::get<BufHandle>(inputs[0]);
  # 获取 inputs 列表中第一个元素（index 0），它被解释为 BufHandle 类型，并命名为 inp

const BufHandle& prepacked = std::get<BufHandle>(inputs[1]);
  # 获取 inputs 列表中第二个元素（index 1），它被解释为 BufHandle 类型，并命名为 prepacked

StmtPtr s = ExternalCall::make(
    ResultBuf, "nnc_prepacked_conv2d_clamp_run", {inp, prepacked}, {});
  # 创建一个外部函数调用的语句对象 s
  # 使用 ExternalCall::make 方法，调用函数名为 "nnc_prepacked_conv2d_clamp_run"
  # ResultBuf 是结果存储的 BufHandle 对象
  # {inp, prepacked} 是传递给外部函数的参数列表，作为输入
  # {} 是一个空字典，用于传递额外的配置或参数，这里未使用

return Tensor(ResultBuf.node(), s);
  # 返回一个 Tensor 对象，其底层数据使用 ResultBuf.node() 存储
  # s 是与该 Tensor 相关联的语句对象，表示如何计算该 Tensor 的值
} // namespace tensorexpr
} // namespace jit
} // namespace torch
```