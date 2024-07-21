# `.\pytorch\test\cpp\api\transformer.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/torch.h>  // 包含 PyTorch C++ API 的头文件

#include <test/cpp/api/support.h>  // 包含测试支持函数的头文件

using namespace torch::nn;  // 使用 torch::nn 命名空间

struct TransformerTest : torch::test::SeedingFixture {};  // 定义 TransformerTest 结构体，继承自 SeedingFixture，用于测试初始化

// 用于设置模型参数为常数，以保证测试结果的确定性
template <typename Model>
void set_parameter_to_constants(
    Model& model,
    const torch::TensorOptions& tensor_options) {
  torch::NoGradGuard guard;  // 创建无梯度上下文管理器，确保不计算梯度
  for (auto& p : model->parameters()) {
    auto sz = p.view(-1).size(0);  // 获取参数展平后的大小
    p.copy_(torch::cos(torch::arange(0, sz, tensor_options).view(p.sizes())));  // 使用余弦函数填充参数数据
  }
}

// 获取测试用的编码器/解码器层，用于所有的Transformer测试
template <typename T_LAYER, typename T_OPTIONS>
T_LAYER get_a_test_layer(
    const torch::TensorOptions& tensor_options,
    bool use_callable_activation) {
  int64_t d_model = 4;  // 模型维度
  int64_t nhead = 2;  // 头数
  int64_t dim_feedforward = 16;  // 前馈网络维度
  double dropout = 0.0;  // dropout比例

  // 激活函数这里始终是ReLU，根据需要后续可以调整
  T_LAYER layer(T_OPTIONS(d_model, nhead)
                    .dim_feedforward(dim_feedforward)
                    .dropout(dropout));
  if (tensor_options.device() == torch::kCUDA) {
    layer->to(torch::kCUDA);  // 将层移动到CUDA设备上
  }
  if (use_callable_activation) {
    // 如果使用可调用激活函数，则设置ReLU作为激活函数
    layer.get()->options.activation(
        [&](const torch::Tensor& t) { return torch::nn::functional::relu(t); });
  }

  // 设置模型的权重为常数
  set_parameter_to_constants<T_LAYER>(layer, tensor_options);

  return layer;  // 返回设置好的层
}

// 帮助测试Transformer编码器层的函数
void transformer_encoder_layer_test_helper(
    bool is_cuda,
}

// Transformer编码器层的测试用例
TEST_F(TransformerTest, TransformerEncoderLayer) {
  transformer_encoder_layer_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/false);  // 测试在CPU上不使用可调用激活函数的情况
  transformer_encoder_layer_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/true);  // 测试在CPU上使用可调用激活函数的情况
}

// Transformer编码器层CUDA版本的测试用例
TEST_F(TransformerTest, TransformerEncoderLayer_CUDA) {
  transformer_encoder_layer_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/false);  // 测试在CUDA上不使用可调用激活函数的情况
  transformer_encoder_layer_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/true);  // 测试在CUDA上使用可调用激活函数的情况
}

// 帮助测试Transformer解码器层的函数
void transformer_decoder_layer_test_helper(
    bool is_cuda,
}

// Transformer解码器层的测试用例
TEST_F(TransformerTest, TransformerDecoderLayer) {
  transformer_decoder_layer_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/false);  // 测试在CPU上不使用可调用激活函数的情况
  transformer_decoder_layer_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/true);  // 测试在CPU上使用可调用激活函数的情况
}

// Transformer解码器层CUDA版本的测试用例
TEST_F(TransformerTest, TransformerDecoderLayer_CUDA) {
  transformer_decoder_layer_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/false);  // 测试在CUDA上不使用可调用激活函数的情况
  transformer_decoder_layer_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/true);  // 测试在CUDA上使用可调用激活函数的情况
}

// 帮助测试具有GELU激活函数的Transformer解码器层的函数
void transformer_decoder_layer_test_helper_gelu(
    bool is_cuda,
    bool use_callable_activation) {
```  
# 接收一个布尔值参数 `use_callable_activation`，用来决定是否使用可调用的激活函数。


  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
```  
# 根据 `is_cuda` 变量的布尔值选择计算设备，如果为 true，则选择 CUDA 设备，否则选择 CPU 设备。


  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
```  
# 创建一个 `torch::TensorOptions` 对象 `tensor_options`，设置张量数据类型为 `torch::kFloat32`，设备根据之前选择的 `device` 决定。


  TransformerDecoderLayer model =
      get_a_test_layer<TransformerDecoderLayer, TransformerDecoderLayerOptions>(
          tensor_options, use_callable_activation);
```  
# 使用 `get_a_test_layer` 函数创建一个 `TransformerDecoderLayer` 对象 `model`，传入 `tensor_options` 和 `use_callable_activation` 参数。


  if (use_callable_activation) {
```  
# 如果 `use_callable_activation` 为真，则执行以下代码块，否则执行 `else` 分支。


    model.get()->options.activation(
        [&](const torch::Tensor& t) { return torch::nn::functional::gelu(t); });
```  
# 在 `model` 对象上设置激活函数为 GELU（Gaussian Error Linear Unit），使用 Lambda 函数作为激活函数的定义。


  } else {
```  
# 如果 `use_callable_activation` 为假，则执行以下代码块。
}

TEST_F(TransformerTest, TransformerDecoderLayer_gelu) {
  // 调用帮助函数测试 TransformerDecoderLayer 的 gelu 函数，非 CUDA 模式，不使用可调用激活函数
  transformer_decoder_layer_test_helper_gelu(
      /*is_cuda=*/false, /*use_callable_activation=*/false);
  // 调用帮助函数测试 TransformerDecoderLayer 的 gelu 函数，非 CUDA 模式，使用可调用激活函数
  transformer_decoder_layer_test_helper_gelu(
      /*is_cuda=*/false, /*use_callable_activation=*/true);
}

TEST_F(TransformerTest, TransformerDecoderLayer_gelu_CUDA) {
  // 调用帮助函数测试 TransformerDecoderLayer 的 gelu 函数，CUDA 模式，不使用可调用激活函数
  transformer_decoder_layer_test_helper_gelu(
      /*is_cuda=*/true, /*use_callable_activation=*/false);
  // 调用帮助函数测试 TransformerDecoderLayer 的 gelu 函数，CUDA 模式，使用可调用激活函数
  transformer_decoder_layer_test_helper_gelu(
      /*is_cuda=*/true, /*use_callable_activation=*/true);
}

void transformer_encoder_test_helper(
    bool is_cuda,
    bool use_callable_activation) {
  // 这是 TransformerEncoderLayer 的确定性测试
  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // 创建一个 TransformerEncoderLayer 对象，用于测试
  TransformerEncoderLayer encoder_layer =
      get_a_test_layer<TransformerEncoderLayer, TransformerEncoderLayerOptions>(
          tensor_options, use_callable_activation);

  // 创建一个 TransformerEncoder 模型对象，包含一个 encoder_layer 层
  TransformerEncoder model(TransformerEncoderOptions(encoder_layer, 1));
  
  // 如果是 CUDA 模式，则将模型移动到 CUDA 设备
  if (is_cuda) {
  // 将模型移动到 CUDA 设备上
  model->to(torch::kCUDA);
}

torch::Tensor encoder_input = torch::tensor(
    {{{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
     {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
     {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
     {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
     {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}},
    tensor_options);
// 将编码器输入数据转换为张量

torch::Tensor result = model(encoder_input).detach();
// 使用模型对编码器输入进行前向推断，并且将输出张量分离出来

torch::Tensor ref_output = torch::tensor(
    {{{2.428589, 0.020835, -0.602055, -0.085249},
      {2.427987, 0.021213, -0.602496, -0.084103}},
     {{2.424689, 0.019155, -0.604793, -0.085672},
      {2.413863, 0.022211, -0.612486, -0.072490}},
     {{2.433774, 0.021598, -0.598343, -0.087548},
      {2.425104, 0.019748, -0.604515, -0.084839}},
     {{2.436185, 0.022682, -0.596625, -0.087261},
      {2.433556, 0.021891, -0.598509, -0.086832}},
     {{2.416246, 0.017512, -0.610712, -0.082961},
      {2.422901, 0.024187, -0.606178, -0.074929}}},
    tensor_options);
// 设置参考输出张量，用于后续的断言比较

ASSERT_EQ(result.sizes(), ref_output.sizes());
// 断言：确保模型输出张量的尺寸与参考输出张量的尺寸相同

ASSERT_TRUE(
    torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));
// 断言：确保模型输出张量与参考输出张量的所有元素在指定误差范围内接近，并处理 NaN 值

// 所有值为 0 的情况下未掩码
torch::Tensor mask = torch::zeros({2, 5}, tensor_options) == 1;
// 创建一个张量 mask，用于指示哪些位置的输入需要被掩码（被忽略）

result = model(
             encoder_input,
             /*src_mask=*/torch::Tensor{},
             /*src_key_padding_mask=*/mask)
             .detach();
// 使用模型对编码器输入进行前向推断，同时传入掩码张量，分离出输出张量

ASSERT_EQ(result.sizes(), ref_output.sizes());
// 断言：确保使用掩码后模型输出张量的尺寸与参考输出张量的尺寸相同

ASSERT_TRUE(
    torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));
// 断言：确保使用掩码后模型输出张量与参考输出张量的所有元素在指定误差范围内接近，并处理 NaN 值

// 使用 0 和 1 的掩码情况
mask[0][1] = 1;
mask[1][3] = 1;
mask[1][4] = 1;
// 修改掩码张量，指定特定位置需要被掩码

result = model(
             encoder_input,
             /*src_mask=*/torch::Tensor{},
             /*src_key_padding_mask=*/mask)
             .detach();
// 使用修改后的掩码张量对模型进行再次前向推断，分离出输出张量

ref_output = torch::tensor(
    {{{2.429026, 0.020793, -0.601741, -0.085642},
      {2.428811, 0.021445, -0.601912, -0.084252}},
     {{2.425009, 0.019155, -0.604566, -0.085899},
      {2.415408, 0.02249, -0.611415, -0.073}},
     {{2.434199, 0.021682, -0.598039, -0.087699},
      {2.42598, 0.019941, -0.603896, -0.085091}},
     {{2.436457, 0.022736, -0.59643, -0.08736},
      {2.434021, 0.022093, -0.598179, -0.08679}},
     {{2.416531, 0.017498, -0.610513, -0.083181},
      {2.4242, 0.024653, -0.605266, -0.074959}}},
    tensor_options);
// 更新参考输出张量，用于修改掩码后的断言比较

ASSERT_EQ(result.sizes(), ref_output.sizes());
// 断言：确保使用修改后的掩码后模型输出张量的尺寸与更新后的参考输出张量的尺寸相同

ASSERT_TRUE(
    torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));
// 断言：确保使用修改后的掩码后模型输出张量与更新后的参考输出张量的所有元素在指定误差范围内接近，并处理 NaN 值

// 测试案例 2，多层无归一化
model = TransformerEncoder(TransformerEncoderOptions(encoder_layer, 2));
// 创建一个 Transformer 编码器模型，指定编码器层数为 2

if (is_cuda) {
    model->to(torch::kCUDA);

# 将模型移动到 CUDA 设备上执行

  }
  result = model(
               encoder_input,
               /*src_mask=*/torch::Tensor{},
               /*src_key_padding_mask=*/mask)
               .detach();

# 使用模型对编码器输入进行推断，不计算梯度，同时应用空的源掩码和指定的填充掩码

  ref_output = torch::tensor(
      {{{2.419051, 0.017446, -0.608738, -0.085003},
        {2.419102, 0.017452, -0.608703, -0.085026}},
       {{2.419043, 0.017445, -0.608744, -0.084999},
        {2.419052, 0.017446, -0.608738, -0.085004}},
       {{2.419067, 0.017448, -0.608727, -0.085010},
        {2.419098, 0.017452, -0.608706, -0.085024}},
       {{2.419072, 0.017449, -0.608724, -0.085012},
        {2.419119, 0.017455, -0.608691, -0.085034}},
       {{2.419019, 0.017442, -0.608761, -0.084989},
        {2.419075, 0.017449, -0.608722, -0.085014}}},
      tensor_options);

# 设置参考输出，这是一个预期的张量，用于与模型输出进行比较

  ASSERT_EQ(result.sizes(), ref_output.sizes());

# 断言：验证模型输出的尺寸与参考输出的尺寸相同

  ASSERT_TRUE(
      torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

# 断言：验证模型输出与参考输出在指定的误差范围内（包括 NaN 值）

  model = TransformerEncoder(TransformerEncoderOptions(encoder_layer, 6));

# 创建一个 TransformerEncoder 模型，使用 6 个编码器层

  if (is_cuda) {
    model->to(torch::kCUDA);
  }

# 如果可用 CUDA，将模型移动到 CUDA 设备上执行

  result = model(
               encoder_input,
               /*src_mask=*/torch::Tensor{},
               /*src_key_padding_mask=*/mask)
               .detach();

# 使用模型对编码器输入进行推断，不计算梯度，同时应用空的源掩码和指定的填充掩码

  ref_output = torch::tensor(
      {{{2.419101, 0.017453, -0.608703, -0.085025},
        {2.419101, 0.017453, -0.608704, -0.085025}},
       {{2.419101, 0.017453, -0.608703, -0.085025},
        {2.419101, 0.017453, -0.608704, -0.085025}},
       {{2.419101, 0.017453, -0.608703, -0.085025},
        {2.419101, 0.017453, -0.608704, -0.085025}},
       {{2.419101, 0.017453, -0.608703, -0.085025},
        {2.419101, 0.017453, -0.608704, -0.085025}},
       {{2.419101, 0.017453, -0.608703, -0.085025},
        {2.419101, 0.017453, -0.608704, -0.085025}}},
      tensor_options);

# 设置参考输出，这是一个预期的张量，用于与模型输出进行比较

  ASSERT_EQ(result.sizes(), ref_output.sizes());

# 断言：验证模型输出的尺寸与参考输出的尺寸相同

  ASSERT_TRUE(
      torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

# 断言：验证模型输出与参考输出在指定的误差范围内（包括 NaN 值）

  // test case 3, multiple layers with norm
  LayerNorm norm(LayerNormOptions({encoder_layer.get()->options.d_model()}));

# 创建一个 LayerNorm 层，该层使用编码器层的模型维度作为输入参数

  model = TransformerEncoder(
      TransformerEncoderOptions(encoder_layer, 2).norm(AnyModule(norm)));

# 创建一个 TransformerEncoder 模型，使用 2 个编码器层，并附加上述的 LayerNorm 层

  if (is_cuda) {
    model->to(torch::kCUDA);
  }

# 如果可用 CUDA，将模型移动到 CUDA 设备上执行
    model->to(torch::kCUDA);
  }
  result = model(
               encoder_input,
               /*src_mask=*/torch::Tensor{},  // 设置一个空的张量作为源遮罩
               /*src_key_padding_mask=*/mask)  // 使用给定的掩码作为源键填充遮罩
               .detach();  // 分离模型的输出结果

  ref_output = torch::tensor(
      {{{1.695949, -0.357635, -0.893077, -0.445238},  // 参考输出张量的第一个子张量
        {1.695955, -0.357639, -0.893050, -0.445266}},  // 参考输出张量的第二个子张量
       {{1.695948, -0.357634, -0.893082, -0.445233},
        {1.695950, -0.357635, -0.893077, -0.445238}},
       {{1.695951, -0.357636, -0.893069, -0.445246},
        {1.695955, -0.357639, -0.893052, -0.445264}},
       {{1.695952, -0.357636, -0.893066, -0.445249},
        {1.695957, -0.357641, -0.893041, -0.445276}},
       {{1.695946, -0.357632, -0.893095, -0.445220},
        {1.695952, -0.357637, -0.893065, -0.445251}}},
      tensor_options);  // 使用给定的选项创建参考输出张量

  ASSERT_EQ(result.sizes(), ref_output.sizes());  // 断言模型输出的尺寸与参考输出张量的尺寸相等
  ASSERT_TRUE(
      torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));  // 断言模型输出与参考输出张量在误差范围内相似

  model = TransformerEncoder(
      TransformerEncoderOptions(encoder_layer, 6).norm(AnyModule(norm)));  // 使用给定的编码器层和标准化模块创建变压器编码器模型
  if (is_cuda) {
    model->to(torch::kCUDA);  // 如果使用 CUDA，则将模型移动到 CUDA 设备
  }
  result = model(
               encoder_input,
               /*src_mask=*/torch::Tensor{},  // 设置一个空的张量作为源遮罩
               /*src_key_padding_mask=*/mask)  // 使用给定的掩码作为源键填充遮罩
               .detach();  // 分离模型的输出结果

  ref_output = torch::tensor(
      {{{1.695955, -0.357639, -0.893051, -0.445265},  // 参考输出张量的第一个子张量
        {1.695955, -0.357639, -0.893051, -0.445265}},  // 参考输出张量的第二个子张量
       {{1.695955, -0.357639, -0.893051, -0.445265},
        {1.695955, -0.357639, -0.893051, -0.445265}},
       {{1.695955, -0.357639, -0.893051, -0.445265},
        {1.695955, -0.357639, -0.893051, -0.445265}},
       {{1.695955, -0.357639, -0.893051, -0.445265},
        {1.695955, -0.357639, -0.893051, -0.445265}},
       {{1.695955, -0.357639, -0.893051, -0.445265},
        {1.695955, -0.357639, -0.893051, -0.445265}}},
      tensor_options);  // 使用给定的选项创建参考输出张量

  ASSERT_EQ(result.sizes(), ref_output.sizes());  // 断言模型输出的尺寸与参考输出张量的尺寸相等
  ASSERT_TRUE(
      torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));  // 断言模型输出与参考输出张量在误差范围内相似
}

// 在 TransformerTest 测试套件中定义名为 TransformerEncoder 的测试用例
TEST_F(TransformerTest, TransformerEncoder) {
  // 调用辅助函数测试 Transformer 编码器，禁用 CUDA 和可调用激活函数
  transformer_encoder_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/false);
  // 调用辅助函数测试 Transformer 编码器，禁用 CUDA 但启用可调用激活函数
  transformer_encoder_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/true);
}

// 在 TransformerTest 测试套件中定义名为 TransformerEncoder_CUDA 的测试用例
TEST_F(TransformerTest, TransformerEncoder_CUDA) {
  // 调用辅助函数测试 Transformer 编码器，启用 CUDA 但禁用可调用激活函数
  transformer_encoder_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/false);
  // 调用辅助函数测试 Transformer 编码器，启用 CUDA 并启用可调用激活函数
  transformer_encoder_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/true);
}

// 在 TransformerTest 测试套件中定义名为 PrettyPrintTransformerEncoderLayer 的测试用例
TEST_F(TransformerTest, PrettyPrintTransformerEncoderLayer) {
  // 断言 TransformerEncoderLayer(4, 2) 的字符串表示符合预期格式
  ASSERT_EQ(
      c10::str(TransformerEncoderLayer(4, 2)),
      "torch::nn::TransformerEncoderLayerImpl(\n"
      "  (self_attn): torch::nn::MultiheadAttention(\n"
      "    (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "  )\n"
      "  (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "  (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "  (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      ")");
}
# 在 TransformerTest 测试类中的 PrettyPrintTransformerEncoder 测试方法
TEST_F(TransformerTest, PrettyPrintTransformerEncoder) {
  # 创建 LayerNorm 对象，设置大小为 [4]
  LayerNorm norm = LayerNorm(LayerNormOptions({4}));
  # 创建 TransformerEncoderOptions 对象，配置包含两个 TransformerEncoderLayerOptions 层，每层包含 4 个头部和 2 个线性层
  TransformerEncoderOptions options(
      TransformerEncoderOptions(TransformerEncoderLayerOptions(4, 2), 2)
          .norm(AnyModule(norm)));
  # 断言 TransformerEncoder 的字符串表示与预期相等
  ASSERT_EQ(
      c10::str(TransformerEncoder(options)),
      "torch::nn::TransformerEncoderImpl(\n"
      "  (layers): torch::nn::ModuleList(\n"
      "    (0): torch::nn::TransformerEncoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "    (1): torch::nn::TransformerEncoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "  )\n"
      "  (norm): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      ")");
}
TEST_F(TransformerTest, PrettyPrintTransformerDecoderLayer) {
  // 断言检查 TransformerDecoderLayer 的字符串表示是否符合预期
  ASSERT_EQ(
      c10::str(TransformerDecoderLayer(4, 2)),
      "torch::nn::TransformerDecoderLayerImpl(\n"
      "  (self_attn): torch::nn::MultiheadAttention(\n"
      "    (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "  )\n"
      "  (multihead_attn): torch::nn::MultiheadAttention(\n"
      "    (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "  )\n"
      "  (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "  (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "  (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (norm3): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (dropout3): torch::nn::Dropout(p=0.1, inplace=false)\n"
      ")");
}

void transformer_decoder_test_helper(
    bool is_cuda,
    bool use_callable_activation) {
  // TransformerDecoder 的辅助测试函数
  // 确定设备类型为 CUDA 还是 CPU
  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  // 定义张量选项，指定为 Float32 类型并在特定设备上
  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // 获取测试用的 TransformerDecoderLayer
  TransformerDecoderLayer decoder_layer =
      get_a_test_layer<TransformerDecoderLayer, TransformerDecoderLayerOptions>(
          tensor_options, use_callable_activation);

  // 创建 TransformerDecoder 模型实例
  TransformerDecoder model(TransformerDecoderOptions(decoder_layer, 1));
  // 如果使用 CUDA，则将模型转移到 CUDA 设备上
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  // 定义解码器输入和记忆输入
  decoder_input = torch::tensor({{{20, 30, 40, 50}}}, tensor_options);
  memory_input = torch::tensor({{{60, 70, 80, 90}}}, tensor_options);
  // 对模型进行推理，获取结果并将其从计算图中分离
  result = model(decoder_input, memory_input).detach();
  // 定义预期输出结果
  ref_output = torch::tensor(
      {{{2.31316, 0.0950293, -0.671995, 0.102802}}}, tensor_options);
  // 断言检查模型输出与预期输出是否一致
  ASSERT_EQ(result.sizes().size(), ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(
      result,
      ref_output,
      1e-7,
      1e-5,
      /*equal_nan=*/true));

  // 多层无归一化的 TransformerDecoder 模型测试
  model = TransformerDecoder(TransformerDecoderOptions(decoder_layer, 6));
  // 如果使用 CUDA，则将模型转移到 CUDA 设备上
  if (is_cuda) {
      model->to(torch::kCUDA);
  }
    model->to(torch::kCUDA);
  }
  // 确保输入数据是确定性的
  decoder_input = torch::tensor(
      {{{0.4517, 0.6793, 0.5313, 0.0034}, {0.2678, 0.3677, 0.4459, 0.7166}},
       {{0.8100, 0.3716, 0.4096, 0.1976}, {0.6958, 0.8844, 0.6081, 0.8315}},
       {{0.0494, 0.9343, 0.5955, 0.3830}, {0.5404, 0.3464, 0.9378, 0.6200}}},
      tensor_options);
  // 设置记忆输入数据
  memory_input = torch::tensor(
      {{{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
       {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
       {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
       {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
       {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}},
      tensor_options);
  // 使用模型进行推理，将结果分离（detach）
  result = model(decoder_input, memory_input).detach();
  // 设置参考输出
  ref_output = torch::tensor(
      {{{2.42794, 0.026164, -0.60263, -0.0747591},
        {2.43113, 0.0279516, -0.600376, -0.0736896}},
       {{2.42794, 0.026164, -0.60263, -0.0747591},
        {2.43113, 0.0279516, -0.600376, -0.0736896}},
       {{2.42794, 0.026164, -0.60263, -0.0747591},
        {2.43113, 0.0279516, -0.600376, -0.0736896}}},
      tensor_options);
  // 断言结果的尺寸与参考输出的尺寸相等
  ASSERT_EQ(result.sizes().size(), ref_output.sizes().size());
  // 断言结果与参考输出在指定的误差范围内相似（包括 NaN 值相等）
  ASSERT_TRUE(torch::allclose(
      result,
      ref_output,
      1e-7,
      1e-5,
      /*equal_nan=*/true));

  // 多层模型，带有标准化处理
  LayerNorm norm(LayerNormOptions({decoder_layer.get()->options.d_model()}));
  // 使用标准化处理创建 TransformerDecoder 模型
  model = TransformerDecoder(
      TransformerDecoderOptions(decoder_layer, 2).norm(AnyModule(norm)));
  // 如果是在 CUDA 上运行，将模型转移到 CUDA 设备
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  // 设置解码器输入数据
  decoder_input = torch::tensor({{{20, 30, 40, 50}}}, tensor_options);
  // 设置记忆输入数据
  memory_input = torch::tensor({{{60, 70, 80, 90}}}, tensor_options);
  // 使用模型进行推理，将结果分离（detach）
  result = model(decoder_input, memory_input).detach();
  // 设置参考输出
  ref_output = torch::tensor(
      {{{1.66166, -0.326986, -1.01466, -0.320017}}}, tensor_options);
  // 断言结果的尺寸与参考输出的尺寸相等
  ASSERT_EQ(result.sizes().size(), ref_output.sizes().size());
  // 断言结果与参考输出在指定的误差范围内相似（包括 NaN 值相等）
  ASSERT_TRUE(torch::allclose(
      result,
      ref_output,
      1e-7,
      1e-5,
      /*equal_nan=*/true));

  // 多层模型，带有标准化处理
  model = TransformerDecoder(
      TransformerDecoderOptions(decoder_layer, 6).norm(AnyModule(norm)));
  // 如果是在 CUDA 上运行，将模型转移到 CUDA 设备
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
    // 将模型移动到 CUDA 设备上
    model->to(torch::kCUDA);
  }
  // 创建确定性输入张量
  decoder_input = torch::tensor(
      {{{0.4517, 0.6793, 0.5313, 0.0034}, {0.2678, 0.3677, 0.4459, 0.7166}},
       {{0.8100, 0.3716, 0.4096, 0.1976}, {0.6958, 0.8844, 0.6081, 0.8315}},
       {{0.0494, 0.9343, 0.5955, 0.3830}, {0.5404, 0.3464, 0.9378, 0.6200}}},
      tensor_options);
  // 创建记忆输入张量
  memory_input = torch::tensor(
      {{{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
       {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
       {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
       {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
       {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}},
      tensor_options);
  // 使用模型进行推理，得到结果并分离计算图
  result = model(decoder_input, memory_input).detach();
  // 定义参考输出张量
  ref_output = torch::tensor(
      {{{1.69559, -0.357291, -0.894741, -0.443553},
        {1.69571, -0.357363, -0.894154, -0.444196}},
       {{1.69559, -0.357291, -0.894741, -0.443553},
        {1.69571, -0.357363, -0.894154, -0.444196}},
       {{1.69559, -0.357291, -0.894741, -0.443553},
        {1.69571, -0.357363, -0.894154, -0.444196}}},
      tensor_options);
  // 断言结果张量的维度与参考输出张量的维度相等
  ASSERT_EQ(result.sizes().size(), ref_output.sizes().size());
  // 断言结果张量与参考输出张量在给定的误差范围内近似相等，包括 NaN 值
  ASSERT_TRUE(torch::allclose(
      result,
      ref_output,
      1e-7,
      1e-5,
      /*equal_nan=*/true));

  // gelu 激活函数的测试用例
  // 设置解码器层的激活函数为 GELU
  decoder_layer.get()->options.activation(torch::kGELU);
  // 重新构造模型，使用新的解码器层选项和 1 层
  model = TransformerDecoder(TransformerDecoderOptions(decoder_layer, 1));
  // 如果运行在 CUDA 上
  if (is_cuda) {
  model->to(torch::kCUDA);

将模型移到 CUDA 设备上。


  }

代码块结束。


  decoder_input = torch::tensor(
      {{{0.4517, 0.6793, 0.5313, 0.0034}, {0.2678, 0.3677, 0.4459, 0.7166}},
       {{0.8100, 0.3716, 0.4096, 0.1976}, {0.6958, 0.8844, 0.6081, 0.8315}},
       {{0.0494, 0.9343, 0.5955, 0.3830}, {0.5404, 0.3464, 0.9378, 0.6200}}},
      tensor_options);

创建一个张量 `decoder_input`，用于模型解码输入。


  memory_input = torch::tensor(
      {{{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
       {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
       {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
       {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
       {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}},
      tensor_options);

创建一个张量 `memory_input`，用于模型的记忆输入。


  result = model(decoder_input, memory_input).detach();

使用模型 `model` 对 `decoder_input` 和 `memory_input` 进行计算，并将结果保存在 `result` 中，同时分离计算图以防止梯度传播。


  ref_output = torch::tensor(
      {{{2.41859, 0.0328114, -0.609269, -0.0560386},
        {2.42138, 0.034598, -0.607316, -0.0546574}},
       {{2.41859, 0.0328114, -0.609269, -0.0560386},
        {2.42138, 0.034598, -0.607316, -0.0546574}},
       {{2.41859, 0.0328114, -0.609269, -0.0560386},
        {2.42138, 0.034598, -0.607316, -0.0546574}}},
      tensor_options);

创建一个张量 `ref_output`，作为模型计算的参考输出。


  ASSERT_EQ(result.sizes().size(), ref_output.sizes().size());

断言 `result` 和 `ref_output` 的维度数量是否相同。


  ASSERT_TRUE(torch::allclose(
      result,
      ref_output,
      1e-7,
      1e-5,
      /*equal_nan=*/true));

断言 `result` 和 `ref_output` 是否在指定的误差范围内（绝对误差 `1e-7`，相对误差 `1e-5`），允许 NaN 值相等。


  // Multiple layers with norm

注释：多层模型，带有规范化（norm）。


  norm = LayerNorm(LayerNormOptions({decoder_layer.get()->options.d_model()}));

使用 `decoder_layer` 的配置创建一个规范化（norm）层 `norm`。


  model = TransformerDecoder(
      TransformerDecoderOptions(decoder_layer, 6).norm(AnyModule(norm)));

使用 `decoder_layer` 和层数 `6` 创建 `TransformerDecoder` 模型 `model`，并添加规范化层 `norm`。


  if (is_cuda) {
    model->to(torch::kCUDA);
  }

如果 `is_cuda` 为真，则将模型 `model` 移到 CUDA 设备上。


  decoder_input = torch::tensor(
      {{{0.4517, 0.6793, 0.5313, 0.0034}, {0.2678, 0.3677, 0.4459, 0.7166}},
       {{0.8100, 0.3716, 0.4096, 0.1976}, {0.6958, 0.8844, 0.6081, 0.8315}},
       {{0.0494, 0.9343, 0.5955, 0.3830}, {0.5404, 0.3464, 0.9378, 0.6200}}},
      tensor_options);

更新 `decoder_input` 张量，用于模型的解码输入。


  memory_input = torch::tensor(
      {{{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
       {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
       {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
       {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
       {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}},
      tensor_options);

更新 `memory_input` 张量，用于模型的记忆输入。


  result = model(decoder_input, memory_input).detach();

使用更新后的模型 `model` 对 `decoder_input` 和 `memory_input` 进行计算，并将结果保存在 `result` 中，同时分离计算图。


  ref_output = torch::tensor(
      {{{1.69298, -0.355163, -0.906375, -0.431439},
        {1.69305, -0.355195, -0.906062, -0.431791}},
       {{1.69298, -0.355163, -0.906375, -0.431439},
        {1.69305, -0.355195, -0.906062, -0.431791}},
       {{1.69298, -0.355163, -0.906375, -0.431439},
        {1.69305, -0.355195, -0.906062, -0.431791}}},
      tensor_options);

更新 `ref_output` 张量，作为模型计算的新的参考输出。


  ASSERT_EQ(result.sizes().size(), ref_output.sizes().size());

再次断言 `result` 和 `ref_output` 的维度数量是否相同。


  ASSERT_TRUE(torch::allclose(
      result,
      ref_output,
      1e-7,
      1e-5,
      /*equal_nan=*/true));

再次断言 `result` 和 `ref_output` 是否在指定的误差范围内，允许 NaN 值相等。
}

TEST_F(TransformerTest, TransformerDecoder) {
  // 调用辅助函数测试 Transformer 解码器，不使用 CUDA，不使用可调用的激活函数
  transformer_decoder_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/false);
  // 调用辅助函数测试 Transformer 解码器，不使用 CUDA，使用可调用的激活函数
  transformer_decoder_test_helper(
      /*is_cuda=*/false, /*use_callable_activation=*/true);
}

TEST_F(TransformerTest, TransformerDecoder_CUDA) {
  // 调用辅助函数测试 Transformer 解码器，使用 CUDA，不使用可调用的激活函数
  transformer_decoder_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/false);
  // 调用辅助函数测试 Transformer 解码器，使用 CUDA，使用可调用的激活函数
  transformer_decoder_test_helper(
      /*is_cuda=*/true, /*use_callable_activation=*/true);
}
TEST_F(TransformerTest, PrettyPrintTransformerDecoder) {
  // 创建一个大小为4的LayerNorm对象
  LayerNorm norm = LayerNorm(LayerNormOptions({4}));
  // 创建TransformerDecoderOptions对象，指定TransformerDecoderLayerOptions和层数
  TransformerDecoderOptions options(
      TransformerDecoderOptions(TransformerDecoderLayerOptions(4, 2), 2)
          .norm(AnyModule(norm)));
  // 断言TransformerDecoder的字符串表示符合预期输出
  ASSERT_EQ(
      c10::str(TransformerDecoder(options)),
      "torch::nn::TransformerDecoderImpl(\n"
      "  (layers): torch::nn::ModuleList(\n"
      "    (0): torch::nn::TransformerDecoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (multihead_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm3): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout3): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "    (1): torch::nn::TransformerDecoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (multihead_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm3): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout3): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "  )\n"
      "  (norm): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      ")");
}
  // 为 Transformer 测试创建辅助函数，测试是否使用 CUDA，是否使用可调用激活函数
  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  // 设置张量选项，指定数据类型为 float32，设备为指定的 CUDA 或 CPU
  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // 创建 Transformer 的选项对象，指定模型参数
  auto options = TransformerOptions()
                     .d_model(4)  // 设置模型的输入维度
                     .nhead(2)  // 设置注意力头数
                     .num_encoder_layers(2)  // 设置编码器层数
                     .num_decoder_layers(1)  // 设置解码器层数
                     .dim_feedforward(16)  // 设置前馈神经网络的隐藏层大小
                     .dropout(0.0)  // 设置 dropout 率为 0.0
                     .activation(torch::kReLU);  // 设置激活函数为 ReLU
  if (use_callable_activation) {
    // 如果使用可调用的激活函数，则重新指定激活函数为 torch::nn::functional::relu
    options.activation(
        [&](const torch::Tensor& t) { return torch::nn::functional::relu(t); });
  }
  // 创建 Transformer 模型对象
  Transformer model(options);

  // 将 Transformer 模型的参数设置为常量
  set_parameter_to_constants<Transformer>(model, tensor_options);
  if (tensor_options.device() == torch::kCUDA) {
    // 如果张量选项的设备为 CUDA，则将模型移动到 CUDA 设备
    model->to(torch::kCUDA);
  }

  // 创建自定义编码器和解码器的 Transformer 对象
  LayerNorm enorm(LayerNormOptions({4}));
  // 创建 Transformer 编码器对象，设置编码器层参数和规范化层
  TransformerEncoder encoder(
      TransformerEncoderOptions(
          TransformerEncoderLayerOptions(4, 2).dim_feedforward(16).dropout(0.0),
          2)
          .norm(AnyModule(enorm)));

  LayerNorm dnorm(LayerNormOptions({4}));
  // 创建 Transformer 解码器对象，设置解码器层参数和规范化层
  TransformerDecoder decoder(
      TransformerDecoderOptions(
          TransformerDecoderLayerOptions(4, 2).dim_feedforward(16).dropout(0.0),
          1)
          .norm(AnyModule(dnorm)));

  // 创建使用自定义编码器和解码器的 Transformer 模型对象
  Transformer model_cus(TransformerOptions()
                            .d_model(4)  // 设置模型的输入维度
                            .nhead(2)  // 设置注意力头数
                            .custom_encoder(AnyModule(encoder))  // 使用自定义编码器
                            .custom_decoder(AnyModule(decoder)));  // 使用自定义解码器

  // 将 Transformer 模型的参数设置为常量
  set_parameter_to_constants<Transformer>(model_cus, tensor_options);
  if (tensor_options.device() == torch::kCUDA) {
    // 如果张量选项的设备为 CUDA，则将模型移动到 CUDA 设备
    model_cus->to(torch::kCUDA);



// 将 model_cus 模型移动到 CUDA 设备上执行
model_cus->to(torch::kCUDA);



  // test cases
  torch::Tensor src = torch::tensor(
      {{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}},
       {{9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}},
       {{17.0, 18.0, 19.0, 20.0}, {21.0, 22.0, 23.0, 24.0}}},
      tensor_options);



// 创建名为 src 的 torch::Tensor，包含三个 2x4 的张量，表示输入数据
torch::Tensor src = torch::tensor(
    {{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}},
     {{9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}},
     {{17.0, 18.0, 19.0, 20.0}, {21.0, 22.0, 23.0, 24.0}}},
    tensor_options);



  torch::Tensor tgt = torch::tensor(
      {{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}},
       {{9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}}},
      tensor_options);



// 创建名为 tgt 的 torch::Tensor，包含两个 2x4 的张量，表示目标数据
torch::Tensor tgt = torch::tensor(
    {{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}},
     {{9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}}},
    tensor_options);



  torch::Tensor ref_output = torch::tensor(
      {{{2.695875, 0.347114, -0.044355, -0.549541},
        {2.696091, 0.347015, -0.044770, -0.548522}},
       {{2.695875, 0.347114, -0.044355, -0.549541},
        {2.696091, 0.347015, -0.044770, -0.548522}}},
      tensor_options);



// 创建名为 ref_output 的 torch::Tensor，包含两个 2x4 的张量，表示期望的输出数据
torch::Tensor ref_output = torch::tensor(
    {{{2.695875, 0.347114, -0.044355, -0.549541},
      {2.696091, 0.347015, -0.044770, -0.548522}},
     {{2.695875, 0.347114, -0.044355, -0.549541},
      {2.696091, 0.347015, -0.044770, -0.548522}}},
    tensor_options);



  torch::Tensor result = model(src, tgt);
  torch::Tensor result_cus = model_cus(src, tgt);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(result.equal(result_cus));
  ASSERT_TRUE(
      torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));



// 使用模型 model 和 model_cus 分别对输入 src 和 tgt 进行推断，比较结果
torch::Tensor result = model(src, tgt);
torch::Tensor result_cus = model_cus(src, tgt);
// 断言推断结果的尺寸与 ref_output 一致
ASSERT_EQ(result.sizes(), ref_output.sizes());
// 断言模型结果与自定义模型结果相等
ASSERT_TRUE(result.equal(result_cus));
// 断言模型结果与 ref_output 在一定误差范围内相等，包括处理 NaN 值
ASSERT_TRUE(
    torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));



  torch::Tensor src_mask =
      Transformer::Impl::generate_square_subsequent_mask(src.size(0))
          .to(tensor_options);



// 创建名为 src_mask 的 torch::Tensor，使用 Transformer::Impl::generate_square_subsequent_mask 函数生成一个大小为 src.size(0) 的方形掩码，并将其转移到指定的 tensor_options
torch::Tensor src_mask =
    Transformer::Impl::generate_square_subsequent_mask(src.size(0))
        .to(tensor_options);



  ref_output = torch::tensor(
      {{{2.695875, 0.347114, -0.044355, -0.549541},
        {2.696091, 0.347015, -0.044770, -0.548522}},
       {{2.695875, 0.347114, -0.044355, -0.549541},
        {2.696091, 0.347015, -0.044770, -0.548522}}},
      tensor_options);



// 更新名为 ref_output 的 torch::Tensor，包含两个 2x4 的张量，表示更新后的期望输出数据
ref_output = torch::tensor(
    {{{2.695875, 0.347114, -0.044355, -0.549541},
      {2.696091, 0.347015, -0.044770, -0.548522}},
     {{2.695875, 0.347114, -0.044355, -0.549541},
      {2.696091, 0.347015, -0.044770, -0.548522}}},
    tensor_options);



  result = model(src, tgt, src_mask);
  result_cus = model_cus(src, tgt, src_mask);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(result.equal(result_cus));
  ASSERT_TRUE(
      torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));



// 使用模型 model 和 model_cus 分别对输入 src、tgt 和 src_mask 进行推断，比较结果
result = model(src, tgt, src_mask);
result_cus = model_cus(src, tgt, src_mask);
// 断言推断结果的尺寸与 ref_output 一致
ASSERT_EQ(result.sizes(), ref_output.sizes());
// 断言模型结果与自定义模型结果相等
ASSERT_TRUE(result.equal(result_cus));
// 断言模型结果与 ref_output 在一定误差范围内相等，包括处理 NaN 值
ASSERT_TRUE(
    torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));



  torch::Tensor tgt_key_padding_mask =
      torch::zeros({tgt.size(1), tgt.size(0)}, tensor_options) == 1;
  tgt_key_padding_mask[0][0] = 1;
  tgt_key_padding_mask[1][1] = 1;



// 创建名为 tgt_key_padding_mask 的 torch::Tensor，其大小为 tgt.size(1) x tgt.size(0)，使用 tensor_options 初始化为全零张量，然后将第一行第一列和第二行第二列设置为 1
torch::Tensor tgt_key_padding_mask =
    torch::zeros({tgt.size(1), tgt.size(0)}, tensor_options) == 1;
tgt_key_padding_mask[0][0] = 1;
tgt_key_padding_mask[1][1] = 1;



  ref_output = torch::tensor(
      {{{2.696114, 0.347004, -0.044813, -0.548417},
        {2.696091, 0.347015, -0.044770, -0.548522}},
       {{2.696114, 0.347004, -0.044813, -0.548417},
        {2.696091, 0.347015, -0.
}

// 定义测试用例 TransformerTest 下的测试函数 Transformer
TEST_F(TransformerTest, Transformer) {
  // 调用 transformer_test_helper 函数，测试不使用 CUDA 和不使用可调用激活函数的情况
  transformer_test_helper(/*is_cuda=*/false, /*use_callable_activation=*/false);
  // 调用 transformer_test_helper 函数，测试不使用 CUDA 和使用可调用激活函数的情况
  transformer_test_helper(/*is_cuda=*/false, /*use_callable_activation=*/true);
}

// 定义测试用例 TransformerTest 下的测试函数 Transformer_CUDA
TEST_F(TransformerTest, Transformer_CUDA) {
  // 调用 transformer_test_helper 函数，测试使用 CUDA 和不使用可调用激活函数的情况
  transformer_test_helper(/*is_cuda=*/true, /*use_callable_activation=*/false);
  // 调用 transformer_test_helper 函数，测试使用 CUDA 和使用可调用激活函数的情况
  transformer_test_helper(/*is_cuda=*/true, /*use_callable_activation=*/true);
}

// 定义测试用例 TransformerArgsCorrectness 下的测试函数
TEST_F(TransformerTest, TransformerArgsCorrectness) {
  // 创建 Transformer 对象并初始化参数
  Transformer model(TransformerOptions()
                        .d_model(4)
                        .nhead(2)
                        .num_encoder_layers(2)
                        .num_decoder_layers(1)
                        .dim_feedforward(16)
                        .dropout(0.0)
                        .activation(torch::kReLU));

  // 创建随机张量 src 和 tgt，形状分别为 {2, 3, 4} 和 {3, 2, 4}
  torch::Tensor src = torch::randn({2, 3, 4});
  torch::Tensor tgt = torch::randn({3, 2, 4});

  // 断言抛出异常，要求 src 和 tgt 的批量大小相等
  ASSERT_THROWS_WITH(
      model(src, tgt), "src and tgt should have equal batch size");

  // 修改 tgt 的形状为 {2, 3, 3}
  tgt = torch::randn({2, 3, 3});
  // 断言抛出异常，要求 src 和 tgt 的特征大小（第三维度）与 d_model 相同
  ASSERT_THROWS_WITH(
      model(src, tgt), "src and tgt should have same feature size as d_model");

  // 修改 src 的形状为 {2, 3}
  src = torch::randn({2, 3});
  // 断言抛出异常，要求 src 和 tgt 至少有三个维度
  ASSERT_THROWS_WITH(model(src, tgt), "src and tgt should have 3 dimensions");
}
```