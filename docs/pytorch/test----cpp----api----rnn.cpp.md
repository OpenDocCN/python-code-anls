# `.\pytorch\test\cpp\api\rnn.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/torch.h>  // 包含 PyTorch C++ 前端的头文件

#include <test/cpp/api/support.h>  // 包含测试支持函数的头文件

using namespace torch::nn;  // 使用 PyTorch 的神经网络命名空间
using namespace torch::test;  // 使用 PyTorch 测试命名空间

template <typename R, typename Func>
bool test_RNN_xor(Func&& model_maker, bool cuda = false) {
  torch::manual_seed(0);  // 设置随机种子为0

  auto nhid = 32;  // 隐藏层维度设为32
  auto model = std::make_shared<SimpleContainer>();  // 创建一个简单的容器模型
  auto l1 = model->add(Linear(1, nhid), "l1");  // 向模型中添加线性层并命名为'l1'
  auto rnn_model = model_maker(nhid);  // 使用模型制造器创建 RNN 模型
  auto rnn = model->add(rnn_model, "rnn");  // 将 RNN 模型添加到容器中并命名为'rnn'
  auto nout = nhid;  // 输出维度初始设为隐藏层维度
  if (rnn_model.get()->options_base.proj_size() > 0) {
    nout = rnn_model.get()->options_base.proj_size();  // 如果 RNN 模型有投影层，则输出维度更新为投影层维度
  }
  auto lo = model->add(Linear(nout, 1), "lo");  // 向模型中添加线性层并命名为'lo'

  torch::optim::Adam optimizer(model->parameters(), 1e-2);  // 使用 Adam 优化器优化模型参数，学习率设为0.01
  auto forward_op = [&](torch::Tensor x) {
    auto T = x.size(0);  // 获取输入张量的时间步维度
    auto B = x.size(1);  // 获取输入张量的批次维度
    x = x.view({T * B, 1});  // 将输入张量重新视图为(T*B, 1)的形状
    x = l1->forward(x).view({T, B, nhid}).tanh_();  // 使用线性层l1进行前向传播，并对结果应用tanh激活函数
    x = std::get<0>(rnn->forward(x))[T - 1];  // 对RNN模型进行前向传播，并取最后一个时间步的输出
    x = lo->forward(x);  // 使用线性层lo进行前向传播
    return x;  // 返回模型的输出
  };

  if (cuda) {
    model->to(torch::kCUDA);  // 如果需要，将模型转移到CUDA设备上
  }

  float running_loss = 1;  // 初始运行损失设为1
  int epoch = 0;  // 初始化迭代次数为0
  auto max_epoch = 1500;  // 最大迭代次数设为1500
  while (running_loss > 1e-2) {  // 当运行损失大于0.01时循环执行以下操作
    auto bs = 16U;  // 批次大小设为16
    auto nlen = 5U;  // 序列长度设为5

    const auto backend = cuda ? torch::kCUDA : torch::kCPU;  // 根据cuda标志选择后端类型
    auto inputs = torch::rand({nlen, bs, 1}, backend).round().to(torch::kFloat32);  // 生成随机输入张量
    auto labels = inputs.sum(0).detach();  // 计算输入张量沿批次维度的和作为标签，并且将其分离出来
    inputs.set_requires_grad(true);  // 设置输入张量需要梯度计算
    auto outputs = forward_op(inputs);  // 获取模型的输出
    torch::Tensor loss = torch::mse_loss(outputs, labels);  // 计算均方误差损失

    optimizer.zero_grad();  // 梯度清零
    loss.backward();  // 反向传播求梯度
    optimizer.step();  // 更新模型参数

    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,bugprone-narrowing-conversions)
    running_loss = running_loss * 0.99 + loss.item<float>() * 0.01;  // 更新运行损失
    if (epoch > max_epoch) {  // 如果迭代次数超过最大迭代次数，返回false
      return false;
    }
    epoch++;  // 迭代次数加1
  }
  return true;  // 运行损失小于等于0.01，返回true
};

void check_lstm_sizes(
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
        lstm_output) {
  // 期望LSTM有64个输出和3层，输入批次大小为10和16个时间步长（10 x 16 x n）

  torch::Tensor output = std::get<0>(lstm_output);  // 获取LSTM输出张量
  std::tuple<torch::Tensor, torch::Tensor> state = std::get<1>(lstm_output);  // 获取LSTM状态元组
  torch::Tensor hx = std::get<0>(state);  // 获取LSTM的隐藏状态
  torch::Tensor cx = std::get<1>(state);  // 获取LSTM的细胞状态

  ASSERT_EQ(output.ndimension(), 3);  // 断言输出张量的维度为3
  ASSERT_EQ(output.size(0), 10);  // 断言输出张量的第一个维度为10
  ASSERT_EQ(output.size(1), 16);  // 断言输出张量的第二个维度为16
  ASSERT_EQ(output.size(2), 64);  // 断言输出张量的第三个维度为64

  ASSERT_EQ(hx.ndimension(), 3);  // 断言隐藏状态的维度为3
  ASSERT_EQ(hx.size(0), 3);  // 断言隐藏状态的第一个维度为3（层数）
  ASSERT_EQ(hx.size(1), 16);  // 断言隐藏状态的第二个维度为16（批次大小）
  ASSERT_EQ(hx.size(2), 64);  // 断言隐藏状态的第三个维度为64（隐藏维度）

  ASSERT_EQ(cx.ndimension(), 3);  // 断言细胞状态的维度为3
  ASSERT_EQ(cx.size(0), 3);  // 断言细胞状态的第一个维度为3（层数）
  ASSERT_EQ(cx.size(1), 16);  // 断言细胞状态的第二个维度为16（批次大小）
  ASSERT_EQ(cx.size(2), 64);  // 断言细胞状态的第三个维度为64（隐藏维度）

  // 断言隐藏状态和细胞状态的范数大于0
  ASSERT_GT(hx.norm().item<float>(), 0);
  ASSERT_GT(cx.norm().item<float>(), 0);
}

void check_lstm_sizes_proj(
  // 获取 LSTM 输出中的主输出张量
  torch::Tensor output = std::get<0>(lstm_output);
  // 获取 LSTM 状态元组，包含隐藏状态和细胞状态
  std::tuple<torch::Tensor, torch::Tensor> state = std::get<1>(lstm_output);
  // 从状态元组中获取隐藏状态张量
  torch::Tensor hx = std::get<0>(state);
  // 从状态元组中获取细胞状态张量
  torch::Tensor cx = std::get<1>(state);

  // 断言主输出张量的维度为3（batch_size x time_steps x hidden_dims）
  ASSERT_EQ(output.ndimension(), 3);
  // 断言主输出张量的第一个维度为10（batch_size）
  ASSERT_EQ(output.size(0), 10);
  // 断言主输出张量的第二个维度为16（time_steps）
  ASSERT_EQ(output.size(1), 16);
  // 断言主输出张量的第三个维度为32（hidden_dims）
  ASSERT_EQ(output.size(2), 32);

  // 断言隐藏状态张量的维度为3（layers x batch_size x hidden_dims）
  ASSERT_EQ(hx.ndimension(), 3);
  // 断言隐藏状态张量的第一个维度为3（layers）
  ASSERT_EQ(hx.size(0), 3);
  // 断言隐藏状态张量的第二个维度为16（batch_size）
  ASSERT_EQ(hx.size(1), 16);
  // 断言隐藏状态张量的第三个维度为32（hidden_dims）
  ASSERT_EQ(hx.size(2), 32);

  // 断言细胞状态张量的维度为3（layers x batch_size x cell_dims）
  ASSERT_EQ(cx.ndimension(), 3);
  // 断言细胞状态张量的第一个维度为3（layers）
  ASSERT_EQ(cx.size(0), 3);
  // 断言细胞状态张量的第二个维度为16（batch_size）
  ASSERT_EQ(cx.size(1), 16);
  // 断言细胞状态张量的第三个维度为64（cell_dims）
  ASSERT_EQ(cx.size(2), 64);

  // 断言隐藏状态张量的 L2 范数大于0，即确保隐藏状态非零
  ASSERT_GT(hx.norm().item<float>(), 0);
  // 断言细胞状态张量的 L2 范数大于0，即确保细胞状态非零
  ASSERT_GT(cx.norm().item<float>(), 0);
}

// 定义一个结构体 RNNTest，继承自 torch::test::SeedingFixture
struct RNNTest : torch::test::SeedingFixture {};

// 测试用例，检查 LSTM 模型的输出尺寸
TEST_F(RNNTest, CheckOutputSizes) {
  // 创建一个具有指定参数的 LSTM 模型
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2));
  // 创建一个大小为 [10, 16, 128] 的随机张量 x，并要求梯度
  auto x = torch::randn({10, 16, 128}, torch::requires_grad());
  // 对模型进行前向传播，得到输出
  auto output = model->forward(x);
  // 计算张量 x 的均值
  auto y = x.mean();

  // 对 y 进行反向传播
  y.backward();
  // 检查 LSTM 输出的尺寸
  check_lstm_sizes(output);

  // 使用 LSTM 模型对输入 x 进行下一步前向传播
  auto next = model->forward(x, std::get<1>(output));

  // 再次检查 LSTM 输出的尺寸
  check_lstm_sizes(next);

  // 获取 output 的隐藏状态和细胞状态
  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  // 获取 next 的隐藏状态和细胞状态
  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  // 计算隐藏状态和细胞状态之间的差异
  torch::Tensor diff =
      torch::cat({next_hx, next_cx}, 0) - torch::cat({output_hx, output_cx}, 0);

  // 断言隐藏状态发生了变化
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

// 测试用例，检查带有投影层的 LSTM 模型的输出尺寸
TEST_F(RNNTest, CheckOutputSizesProj) {
  // 创建一个具有投影层的 LSTM 模型
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2).proj_size(32));
  // 创建一个大小为 [10, 16, 128] 的随机张量 x，并要求梯度
  auto x = torch::randn({10, 16, 128}, torch::requires_grad());
  // 对模型进行前向传播，得到输出
  auto output = model->forward(x);
  // 计算张量 x 的均值
  auto y = x.mean();

  // 对 y 进行反向传播
  y.backward();
  // 检查带投影层的 LSTM 输出的尺寸
  check_lstm_sizes_proj(output);

  // 使用 LSTM 模型对输入 x 进行下一步前向传播
  auto next = model->forward(x, std::get<1>(output));

  // 再次检查带投影层的 LSTM 输出的尺寸
  check_lstm_sizes_proj(next);

  // 获取 output 的隐藏状态和细胞状态
  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  // 获取 next 的隐藏状态和细胞状态
  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  // 计算隐藏状态之间的差异
  torch::Tensor diff = next_hx - output_hx;
  // 断言隐藏状态发生了变化
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
  // 计算细胞状态之间的差异
  diff = next_cx - output_cx;
  // 断言细胞状态发生了变化
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

// 测试用例，检查 LSTM 模型的输出值与 PyTorch 的匹配性
TEST_F(RNNTest, CheckOutputValuesMatchPyTorch) {
  // 设置随机种子为 0
  torch::manual_seed(0);
  // 确保输出与 PyTorch 输出匹配
  LSTM model(2, 2);
  // 遍历模型的参数，并设置其值
  for (auto& v : model->parameters()) {
    float size = v.numel();
    auto p = static_cast<float*>(v.storage().mutable_data());
    for (size_t i = 0; i < size; i++) {
      p[i] = i / size;
    }
  }

  // 创建一个空的张量 x，大小为 [3, 4, 2]，并要求梯度
  auto x = torch::empty({3, 4, 2}, torch::requires_grad());
  float size = x.numel();
  auto p = static_cast<float*>(x.storage().mutable_data());
  for (size_t i = 0; i < size; i++) {
    p[i] = (size - i) / size;
  }

  // 对模型进行前向传播，得到输出 out
  auto out = model->forward(x);
  // 断言输出的张量维度为 3
  ASSERT_EQ(std::get<0>(out).ndimension(), 3);
  // 断言输出的张量第一维大小为 3
  ASSERT_EQ(std::get<0>(out).size(0), 3);
  // 断言输出的张量第二维大小为 4
  ASSERT_EQ(std::get<0>(out).size(1), 4);
  // 断言输出的张量第三维大小为 2
  ASSERT_EQ(std::get<0>(out).size(2), 2);

  // 将输出展平为一维张量 flat
  auto flat = std::get<0>(out).view(3 * 4 * 2);
  // 预期的输出值 c_out
  float c_out[] = {0.4391, 0.5402, 0.4330, 0.5324, 0.4261, 0.5239,
                   0.4183, 0.5147, 0.6822, 0.8064, 0.6726, 0.7968,
                   0.6620, 0.7860, 0.6501, 0.7741, 0.7889, 0.9003,
                   0.7769, 0.8905, 0.7635, 0.8794, 0.7484, 0.8666};
  // 遍历比较输出值和预期值
  for (size_t i = 0; i < 3 * 4 * 2; i++) {



    // 检查每个输出值是否接近预期值
    ASSERT_NEAR(flat[i].item<float>(), c_out[i], 1e-4);
  }
}
    // 确保 flat 中的每个元素与 h_out 中对应位置的元素的绝对误差小于 1e-3
    for (size_t i = 0; i < 16; i++) {
        ASSERT_LT(std::abs(flat[i].item<float>() - h_out[i]), 1e-3);
      }
    
    // 从 out 中提取 hx 和 cx
    auto hx = std::get<0>(std::get<1>(out));
    auto cx = std::get<1>(std::get<1>(out));
    
    // 确保 hx 的维度是 3，对应 layers x B x 2
    ASSERT_EQ(hx.ndimension(), 3);
    ASSERT_EQ(hx.size(0), 1);
    ASSERT_EQ(hx.size(1), 4);
    ASSERT_EQ(hx.size(2), 2);
    
    // 确保 cx 的维度是 3，对应 layers x B x 2
    ASSERT_EQ(cx.ndimension(), 3);
    ASSERT_EQ(cx.size(0), 1);
    ASSERT_EQ(cx.size(1), 4);
    ASSERT_EQ(cx.size(2), 2);
    
    // 将 hx 和 cx 拼接到 flat 中，并按照指定形状重新视图为大小为 16 的张量
    flat = torch::cat({hx, cx}, 0).view(16);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    // 预设的期望输出数组 h_out，用于后续比较
    float h_out[] = {
        0.7889,
        0.9003,
        0.7769,
        0.8905,
        0.7635,
        0.8794,
        0.7484,
        0.8666,
        1.1647,
        1.6106,
        1.1425,
        1.5726,
        1.1187,
        1.5329,
        1.0931,
        1.4911};
    // 检查 flat 中的每个元素与预设的 h_out 数组中对应位置的元素的绝对误差是否小于 1e-3
    for (size_t i = 0; i < 16; i++) {
        ASSERT_LT(std::abs(flat[i].item<float>() - h_out[i]), 1e-3);
    }
TEST_F(RNNTest, EndToEndLSTM) {
  // 使用 LSTM 模型测试 XOR 任务的端到端功能
  ASSERT_TRUE(test_RNN_xor<LSTM>(
      // 匿名函数，返回一个配置了指定大小和层数的 LSTM 模型实例
      [](int s) { return LSTM(LSTMOptions(s, s).num_layers(2)); }));
}

TEST_F(RNNTest, EndToEndLSTMProj) {
  // 使用 LSTM 模型测试 XOR 任务的端到端功能，带投影层
  ASSERT_TRUE(test_RNN_xor<LSTM>([](int s) {
    // 匿名函数，返回一个配置了指定大小、层数和投影大小的 LSTM 模型实例
    return LSTM(LSTMOptions(s, s).num_layers(2).proj_size(s / 2));
  }));
}

TEST_F(RNNTest, EndToEndGRU) {
  // 使用 GRU 模型测试 XOR 任务的端到端功能
  ASSERT_TRUE(test_RNN_xor<GRU>(
      // 匿名函数，返回一个配置了指定大小和层数的 GRU 模型实例
      [](int s) { return GRU(GRUOptions(s, s).num_layers(2)); }));
}

TEST_F(RNNTest, EndToEndRNNRelu) {
  // 使用具有 ReLU 非线性激活函数的 RNN 模型测试 XOR 任务的端到端功能
  ASSERT_TRUE(test_RNN_xor<RNN>([](int s) {
    // 匿名函数，返回一个配置了指定大小、层数和 ReLU 激活函数的 RNN 模型实例
    return RNN(RNNOptions(s, s).nonlinearity(torch::kReLU).num_layers(2));
  }));
}

TEST_F(RNNTest, EndToEndRNNTanh) {
  // 使用具有 Tanh 非线性激活函数的 RNN 模型测试 XOR 任务的端到端功能
  ASSERT_TRUE(test_RNN_xor<RNN>([](int s) {
    // 匿名函数，返回一个配置了指定大小、层数和 Tanh 激活函数的 RNN 模型实例
    return RNN(RNNOptions(s, s).nonlinearity(torch::kTanh).num_layers(2));
  }));
}

TEST_F(RNNTest, Sizes_CUDA) {
  // 设置随机种子
  torch::manual_seed(0);
  // 创建具有指定配置的 LSTM 模型实例
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2));
  // 将模型移动到 CUDA 设备
  model->to(torch::kCUDA);
  // 创建一个指定大小和设备的张量
  auto x =
      torch::randn({10, 16, 128}, torch::requires_grad().device(torch::kCUDA));
  // 进行模型的前向传播
  auto output = model->forward(x);
  // 计算输入张量的均值
  auto y = x.mean();

  // 计算梯度
  y.backward();

  // 检查 LSTM 模型输出的尺寸
  check_lstm_sizes(output);

  // 使用模型进行下一个输入的前向传播
  auto next = model->forward(x, std::get<1>(output));

  // 检查带投影层的 LSTM 模型输出的尺寸
  check_lstm_sizes(next);

  // 提取 LSTM 模型输出的隐藏状态和细胞状态
  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  // 提取下一个 LSTM 模型输出的隐藏状态和细胞状态
  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  // 计算隐藏状态和细胞状态的差异
  torch::Tensor diff =
      torch::cat({next_hx, next_cx}, 0) - torch::cat({output_hx, output_cx}, 0);

  // 断言隐藏状态发生了变化
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, SizesProj_CUDA) {
  // 设置随机种子
  torch::manual_seed(0);
  // 创建具有投影层的 LSTM 模型实例
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2).proj_size(32));
  // 将模型移动到 CUDA 设备
  model->to(torch::kCUDA);
  // 创建一个指定大小和设备的张量
  auto x =
      torch::randn({10, 16, 128}, torch::requires_grad().device(torch::kCUDA));
  // 进行模型的前向传播
  auto output = model->forward(x);
  // 计算输入张量的均值
  auto y = x.mean();

  // 计算梯度
  y.backward();

  // 检查带投影层的 LSTM 模型输出的尺寸
  check_lstm_sizes_proj(output);

  // 使用模型进行下一个输入的前向传播
  auto next = model->forward(x, std::get<1>(output));

  // 检查带投影层的 LSTM 模型输出的尺寸
  check_lstm_sizes_proj(next);

  // 提取 LSTM 模型输出的隐藏状态和细胞状态
  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  // 提取下一个 LSTM 模型输出的隐藏状态和细胞状态
  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  // 计算隐藏状态和细胞状态的差异
  torch::Tensor diff = next_hx - output_hx;
  // 断言隐藏状态发生了变化
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
  // 计算隐藏状态和细胞状态的差异
  diff = next_cx - output_cx;
  // 断言细胞状态发生了变化
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, EndToEndLSTM_CUDA) {
  // 使用 CUDA 加速，测试具有指定大小和层数的 LSTM 模型的端到端功能
  ASSERT_TRUE(test_RNN_xor<LSTM>(
      [](int s) { return LSTM(LSTMOptions(s, s).num_layers(2)); }, true));
}

TEST_F(RNNTest, EndToEndLSTMProj_CUDA) {
  // 使用 CUDA 加速，测试具有指定大小、层数和投影大小的 LSTM 模型的端到端功能
  ASSERT_TRUE(test_RNN_xor<LSTM>(
      [](int s) {
        return LSTM(LSTMOptions(s, s).num_layers(2).proj_size(s / 2));
      },
      true));
}

TEST_F(RNNTest, EndToEndGRU_CUDA) {
  // 使用 CUDA 加速，测试具有指定大小和层数的 GRU 模型的端到端功能
  ASSERT_TRUE(test_RNN_xor<GRU>(
      [](int s) { return GRU(GRUOptions(s, s).num_layers(2)); }, true));
}
TEST_F(RNNTest, EndToEndRNNRelu_CUDA) {
  // 在 RNN 测试框架中执行端到端的测试，使用 ReLU 激活函数和 CUDA 加速
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) {
        // 使用指定大小和激活函数（ReLU），创建 RNN 模型对象，包括两层网络
        return RNN(RNNOptions(s, s).nonlinearity(torch::kReLU).num_layers(2));
      },
      true));
}

TEST_F(RNNTest, EndToEndRNNTanh_CUDA) {
  // 在 RNN 测试框架中执行端到端的测试，使用 Tanh 激活函数和 CUDA 加速
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) {
        // 使用指定大小和激活函数（Tanh），创建 RNN 模型对象，包括两层网络
        return RNN(RNNOptions(s, s).nonlinearity(torch::kTanh).num_layers(2));
      },
      true));
}

TEST_F(RNNTest, PrettyPrintRNNs) {
  // 验证 LSTM 的漂亮打印输出是否符合预期字符串
  ASSERT_EQ(
      c10::str(LSTM(LSTMOptions(128, 64).num_layers(3).dropout(0.2))),
      "torch::nn::LSTM(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.2, bidirectional=false)");
  // 验证带额外投影层参数的 LSTM 的漂亮打印输出是否符合预期字符串
  ASSERT_EQ(
      c10::str(
          LSTM(LSTMOptions(128, 64).num_layers(3).dropout(0.2).proj_size(32))),
      "torch::nn::LSTM(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.2, bidirectional=false, proj_size=32)");
  // 验证 GRU 的漂亮打印输出是否符合预期字符串
  ASSERT_EQ(
      c10::str(GRU(GRUOptions(128, 64).num_layers(3).dropout(0.5))),
      "torch::nn::GRU(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.5, bidirectional=false)");
  // 验证带 Tanh 激活函数参数的 RNN 的漂亮打印输出是否符合预期字符串
  ASSERT_EQ(
      c10::str(RNN(RNNOptions(128, 64).num_layers(3).dropout(0.2).nonlinearity(
          torch::kTanh))),
      "torch::nn::RNN(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.2, bidirectional=false)");
}

// 此测试确保在 bidirectional 设置为 true 时，flatten_parameters 不会崩溃
// 参考：https://github.com/pytorch/pytorch/issues/19545
TEST_F(RNNTest, BidirectionalFlattenParameters) {
  // 创建一个双向 GRU 模型对象
  GRU gru(GRUOptions(100, 256).num_layers(2).bidirectional(true));
  // 调用 flatten_parameters() 方法，确保不会出现崩溃
  gru->flatten_parameters();
}

template <typename Impl>
void copyParameters(
    torch::nn::ModuleHolder<Impl>& target,
    std::string t_suffix,
    const torch::nn::ModuleHolder<Impl>& source,
    std::string s_suffix) {
  // 禁用梯度计算的上下文保护
  at::NoGradGuard guard;
  // 复制源模型的权重和偏置到目标模型中
  target->named_parameters()["weight_ih_l" + t_suffix].copy_(
      source->named_parameters()["weight_ih_l" + s_suffix]);
  target->named_parameters()["weight_hh_l" + t_suffix].copy_(
      source->named_parameters()["weight_hh_l" + s_suffix]);
  target->named_parameters()["bias_ih_l" + t_suffix].copy_(
      source->named_parameters()["bias_ih_l" + s_suffix]);
  target->named_parameters()["bias_hh_l" + t_suffix].copy_(
      source->named_parameters()["bias_hh_l" + s_suffix]);
}

std::tuple<torch::Tensor, torch::Tensor> gru_output_to_device(
    std::tuple<torch::Tensor, torch::Tensor> gru_output,
    torch::Device device) {
  // 将 GRU 输出的张量移到指定的设备上
  return std::make_tuple(
      std::get<0>(gru_output).to(device), std::get<1>(gru_output).to(device));
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
lstm_output_to_device(
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
        lstm_output,
    torch::Device device) {
  // 将 LSTM 输出的张量和元组中的张量移到指定的设备上
  return std::make_tuple(
      std::get<0>(lstm_output).to(device),
      std::make_tuple(std::get<0>(std::get<1>(lstm_output)).to(device),
                      std::get<1>(std::get<1>(lstm_output)).to(device)));
}
    torch::Device device) {
  // 从 LSTM 输出中获取隐藏状态
  auto hidden_states = std::get<1>(lstm_output);
  // 返回转移到指定设备后的结果元组
  return std::make_tuple(
      // 将 LSTM 输出的第一个元组中的内容移动到指定设备
      std::get<0>(lstm_output).to(device),
      // 构造包含隐藏状态移动到指定设备后的元组
      std::make_tuple(
          std::get<0>(hidden_states).to(device),
          std::get<1>(hidden_states).to(device)));
}
// This function tests the reverse forward behavior of a bidirectional GRU in PyTorch.
// It compares the output of a regular forward pass of a unidirectional GRU with the reverse forward
// output of a bidirectional GRU to ensure consistency.
void BidirectionalGRUReverseForward(bool cuda) {
  // Define tensor options based on whether CUDA is enabled
  auto opt = torch::TensorOptions()
                 .dtype(torch::kFloat32)
                 .requires_grad(false)
                 .device(cuda ? torch::kCUDA : torch::kCPU);
  
  // Create input tensors and reshape them
  auto input = torch::tensor({1, 2, 3, 4, 5}, opt).reshape({5, 1, 1});
  auto input_reversed = torch::tensor({5, 4, 3, 2, 1}, opt).reshape({5, 1, 1});
  
  // Define GRU options for both bidirectional and reverse unidirectional GRU
  auto gru_options = GRUOptions(1, 1).num_layers(1).batch_first(false);
  GRU bi_grus{gru_options.bidirectional(true)};
  GRU reverse_gru{gru_options.bidirectional(false)};
  
  // Move GRU modules to CUDA if cuda is true
  if (cuda) {
    bi_grus->to(torch::kCUDA);
    reverse_gru->to(torch::kCUDA);
  }
  
  // Copy parameters of the reversed GRU layer to match those of the bidirectional's reversed part
  copyParameters(reverse_gru, "0", bi_grus, "0_reverse");
  
  // Perform forward passes on both bidirectional and reversed GRUs
  auto bi_output = bi_grus->forward(input);
  auto reverse_output = reverse_gru->forward(input_reversed);
  
  // Move outputs back to CPU if cuda is true
  if (cuda) {
    bi_output = gru_output_to_device(bi_output, torch::kCPU);
    reverse_output = gru_output_to_device(reverse_output, torch::kCPU);
  }
  
  // Assertion to ensure the sizes of the outputs match
  ASSERT_EQ(
      std::get<0>(bi_output).size(0), std::get<0>(reverse_output).size(0));
  
  // Compare specific elements of the outputs to verify reverse forward correctness
  auto size = std::get<0>(bi_output).size(0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(
        std::get<0>(bi_output)[i][0][1].item<float>(),
        std::get<0>(reverse_output)[size - 1 - i][0][0].item<float>());
  }
  
  // Additional assertion to check hidden states consistency
  ASSERT_EQ(
      std::get<1>(bi_output)[1][0][0].item<float>(),
      std::get<1>(reverse_output)[0][0][0].item<float>());
}

// Unit test using Google Test framework for BidirectionalGRUReverseForward function (CPU)
TEST_F(RNNTest, BidirectionalGRUReverseForward) {
  BidirectionalGRUReverseForward(false);
}

// Unit test using Google Test framework for BidirectionalGRUReverseForward function (CUDA)
TEST_F(RNNTest, BidirectionalGRUReverseForward_CUDA) {
  BidirectionalGRUReverseForward(true);
}

// This function tests the reverse forward behavior of a bidirectional LSTM in PyTorch.
// It compares the output of a regular forward pass of a unidirectional LSTM with the reverse forward
// output of a bidirectional LSTM to ensure consistency.
void BidirectionalLSTMReverseForwardTest(bool cuda) {
  // Define tensor options based on whether CUDA is enabled
  auto opt = torch::TensorOptions()
                 .dtype(torch::kFloat32)
                 .requires_grad(false)
                 .device(cuda ? torch::kCUDA : torch::kCPU);
  
  // Create input tensors and reshape them
  auto input = torch::tensor({1, 2, 3, 4, 5}, opt).reshape({5, 1, 1});
  auto input_reversed = torch::tensor({5, 4, 3, 2, 1}, opt).reshape({5, 1, 1});
  
  // Define LSTM options for both bidirectional and reverse unidirectional LSTM
  auto lstm_opt = LSTMOptions(1, 1).num_layers(1).batch_first(false);
  LSTM bi_lstm{lstm_opt.bidirectional(true)};
  LSTM reverse_lstm{lstm_opt.bidirectional(false)};
  
  // Move LSTM modules to CUDA if cuda is true
  if (cuda) {
    bi_lstm->to(torch::kCUDA);
    reverse_lstm->to(torch::kCUDA);
  }
  
    // 将 reverse_lstm 模型移动到 CUDA 设备上
    reverse_lstm->to(torch::kCUDA);
  }

  // 确保反向 LSTM 层的权重与（反向）双向 LSTM 的权重匹配
  copyParameters(reverse_lstm, "0", bi_lstm, "0_reverse");

  // 对输入数据进行双向 LSTM 的前向计算和反向 LSTM 的前向计算
  auto bi_output = bi_lstm->forward(input);
  auto reverse_output = reverse_lstm->forward(input_reversed);

  // 如果使用 CUDA，将双向 LSTM 和反向 LSTM 的输出数据移到 CPU 上
  if (cuda) {
    bi_output = lstm_output_to_device(bi_output, torch::kCPU);
    reverse_output = lstm_output_to_device(reverse_output, torch::kCPU);
  }

  // 断言双向 LSTM 和反向 LSTM 的输出序列长度相同
  ASSERT_EQ(
      std::get<0>(bi_output).size(0), std::get<0>(reverse_output).size(0));
  auto size = std::get<0>(bi_output).size(0);

  // 对每个时间步进行断言，确保双向 LSTM 和反向 LSTM 的输出数据匹配
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(
        std::get<0>(bi_output)[i][0][1].item<float>(),
        std::get<0>(reverse_output)[size - 1 - i][0][0].item<float>());
  }

  // 断言反向 LSTM 的隐藏状态位于第一个维度的奇数索引位置
  ASSERT_EQ(
      std::get<0>(std::get<1>(bi_output))[1][0][0].item<float>(),
      std::get<0>(std::get<1>(reverse_output))[0][0][0].item<float>());
  ASSERT_EQ(
      std::get<1>(std::get<1>(bi_output))[1][0][0].item<float>(),
      std::get<1>(std::get<1>(reverse_output))[0][0][0].item<float>());
}

TEST_F(RNNTest, BidirectionalLSTMReverseForward) {
  // 调用 BidirectionalLSTMReverseForwardTest 函数，传入 false 参数进行测试
  BidirectionalLSTMReverseForwardTest(false);
}

TEST_F(RNNTest, BidirectionalLSTMReverseForward_CUDA) {
  // 调用 BidirectionalLSTMReverseForwardTest 函数，传入 true 参数进行 CUDA 测试
  BidirectionalLSTMReverseForwardTest(true);
}

TEST_F(RNNTest, BidirectionalMultilayerGRU_CPU_vs_CUDA) {
  // 使用相同的选项创建两个具有 3 层、4 个隐藏单元的双向 GRU 模型
  auto opt =
      GRUOptions(2, 4).num_layers(3).batch_first(false).bidirectional(true);
  GRU gru_cpu{opt};
  GRU gru_cuda{opt};

  // 将 CPU GRU 的权重和偏置复制到 CUDA GRU
  {
    at::NoGradGuard guard;  // 创建一个禁止梯度计算的上下文管理器
    // 遍历 CPU GRU 的命名参数，复制到 CUDA GRU 中
    for (const auto& param : gru_cpu->named_parameters(/*recurse=*/false)) {
      gru_cuda->named_parameters()[param.key()].copy_(
          gru_cpu->named_parameters()[param.key()]);
    }
  }

  // 将 GRU 参数打平
  gru_cpu->flatten_parameters();
  gru_cuda->flatten_parameters();

  // 将 GRU 模型移动到 CUDA 设备上
  gru_cuda->to(torch::kCUDA);

  // 创建相同的输入数据
  auto input_opt =
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  auto input_cpu =
      torch::tensor({1, 2, 3, 4, 5, 6}, input_opt).reshape({3, 1, 2});
  auto input_cuda = torch::tensor({1, 2, 3, 4, 5, 6}, input_opt)
                        .reshape({3, 1, 2})
                        .to(torch::kCUDA);

  // 在两个 GRU 模型上调用前向传播
  auto output_cpu = gru_cpu->forward(input_cpu);
  auto output_cuda = gru_cuda->forward(input_cuda);

  // 将 CUDA 上的输出移回到 CPU
  output_cpu = gru_output_to_device(output_cpu, torch::kCPU);

  // 断言 CPU 和 CUDA 上的输出张量维度相同
  ASSERT_EQ(std::get<0>(output_cpu).dim(), std::get<0>(output_cuda).dim());
  // 检查每个维度的大小是否相同
  for (int i = 0; i < std::get<0>(output_cpu).dim(); i++) {
    ASSERT_EQ(
        std::get<0>(output_cpu).size(i), std::get<0>(output_cuda).size(i));
  }
  // 检查每个元素的数值是否在一定误差范围内相等
  for (int i = 0; i < std::get<0>(output_cpu).size(0); i++) {
    for (int j = 0; j < std::get<0>(output_cpu).size(1); j++) {
      for (int k = 0; k < std::get<0>(output_cpu).size(2); k++) {
        ASSERT_NEAR(
            std::get<0>(output_cpu)[i][j][k].item<float>(),
            std::get<0>(output_cuda)[i][j][k].item<float>(),
            1e-5);
      }
    }
  }
}

TEST_F(RNNTest, BidirectionalMultilayerLSTM_CPU_vs_CUDA) {
  // 使用相同的选项创建两个具有 3 层、4 个隐藏单元的双向 LSTM 模型
  auto opt =
      LSTMOptions(2, 4).num_layers(3).batch_first(false).bidirectional(true);
  LSTM lstm_cpu{opt};
  LSTM lstm_cuda{opt};

  // 将 CPU LSTM 的权重和偏置复制到 CUDA LSTM
  {
    at::NoGradGuard guard;  // 创建一个禁止梯度计算的上下文管理器
    // 遍历 CPU LSTM 的命名参数，复制到 CUDA LSTM 中
    for (const auto& param : lstm_cpu->named_parameters(/*recurse=*/false)) {
      lstm_cuda->named_parameters()[param.key()].copy_(
          lstm_cpu->named_parameters()[param.key()]);
    `
      }
      # 调用 LSTM 的 flatten_parameters 方法，优化 LSTM 的内存布局
      lstm_cpu->flatten_parameters();
      lstm_cuda->flatten_parameters();
    
      # 将 LSTM 模型移动到 CUDA 设备上
      lstm_cuda->to(torch::kCUDA);
    
      # 创建一个 tensor 选项，数据类型为 Float32，且不需要计算梯度
      auto options =
          torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
      # 创建一个 CPU 上的 tensor，数据为 {1, 2, 3, 4, 5, 6}，并调整形状为 (3, 1, 2)
      auto input_cpu =
          torch::tensor({1, 2, 3, 4, 5, 6}, options).reshape({3, 1, 2});
      # 创建一个 CUDA 上的 tensor，数据为 {1, 2, 3, 4, 5, 6}，并调整形状为 (3, 1, 2)
      # 将 tensor 移动到 CUDA 设备
      auto input_cuda = torch::tensor({1, 2, 3, 4, 5, 6}, options)
                            .reshape({3, 1, 2})
                            .to(torch::kCUDA);
    
      # 在 CPU 上调用 LSTM 的 forward 方法，计算输出
      auto output_cpu = lstm_cpu->forward(input_cpu);
      # 在 CUDA 上调用 LSTM 的 forward 方法，计算输出
      auto output_cuda = lstm_cuda->forward(input_cuda);
    
      # 将 CPU 上的输出 tensor 移动回 CPU 设备
      output_cpu = lstm_output_to_device(output_cpu, torch::kCPU);
    
      # 断言输出和状态在 CPU 和 CUDA 设备上的维度相同
      ASSERT_EQ(std::get<0>(output_cpu).dim(), std::get<0>(output_cuda).dim());
      # 遍历输出的每一个维度，断言维度大小相同
      for (int i = 0; i < std::get<0>(output_cpu).dim(); i++) {
        ASSERT_EQ(
            std::get<0>(output_cpu).size(i), std::get<0>(output_cuda).size(i));
      }
      # 遍历输出的每一个元素，断言 CPU 和 CUDA 输出之间的值接近
      for (int i = 0; i < std::get<0>(output_cpu).size(0); i++) {
        for (int j = 0; j < std::get<0>(output_cpu).size(1); j++) {
          for (int k = 0; k < std::get<0>(output_cpu).size(2); k++) {
            ASSERT_NEAR(
                std::get<0>(output_cpu)[i][j][k].item<float>(),
                std::get<0>(output_cuda)[i][j][k].item<float>(),
                1e-5);
          }
        }
      }
TEST_F(RNNTest, UsePackedSequenceAsInput) {
  {
    // 设置随机种子为0，确保结果可重复
    torch::manual_seed(0);
    // 创建一个RNN模型，输入维度为2，隐藏状态维度为3
    auto m = RNN(2, 3);
    // 将张量序列打包成PackedSequence对象
    torch::nn::utils::rnn::PackedSequence packed_input =
        torch::nn::utils::rnn::pack_sequence({torch::ones({3, 2})});
    // 调用模型的forward_with_packed_input方法进行前向传播
    auto rnn_output = m->forward_with_packed_input(packed_input);
    // 预期输出的张量
    auto expected_output = torch::tensor(
        {{-0.0645, -0.7274, 0.4531},
         {-0.3970, -0.6950, 0.6009},
         {-0.3877, -0.7310, 0.6806}});
    // 使用allclose函数检查输出张量是否与预期输出接近
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));

    // 测试将可选参数传递给`RNN::forward_with_packed_input`方法
    rnn_output = m->forward_with_packed_input(packed_input, torch::Tensor());
    // 再次使用allclose函数检查输出张量是否与预期输出接近
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));
  }
  {
    // 设置随机种子为0，确保结果可重复
    torch::manual_seed(0);
    // 创建一个LSTM模型，输入维度为2，隐藏状态维度为3
    auto m = LSTM(2, 3);
    // 将张量序列打包成PackedSequence对象
    torch::nn::utils::rnn::PackedSequence packed_input =
        torch::nn::utils::rnn::pack_sequence({torch::ones({3, 2})});
    // 调用模型的forward_with_packed_input方法进行前向传播
    auto rnn_output = m->forward_with_packed_input(packed_input);
    // 创建预期输出张量，包含预期的数值
    auto expected_output = torch::tensor(
        {{-0.2693, -0.1240, 0.0744},
         {-0.3889, -0.1919, 0.1183},
         {-0.4425, -0.2314, 0.1386}});
    // 使用 `torch::allclose` 函数验证 `rnn_output` 的第一个元素与预期输出的接近程度
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));

    // 测试向 `LSTM::forward_with_packed_input` 传递可选参数
    rnn_output = m->forward_with_packed_input(packed_input, torch::nullopt);
    // 再次使用 `torch::allclose` 函数验证更新后的 `rnn_output` 的第一个元素与预期输出的接近程度
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));
  }
  {
    // 设置随机种子为 0
    torch::manual_seed(0);
    // 创建一个 GRU 模型 `m`，输入维度为 2，输出维度为 3
    auto m = GRU(2, 3);
    // 将张量序列打包为压缩序列 `packed_input`
    torch::nn::utils::rnn::PackedSequence packed_input =
        torch::nn::utils::rnn::pack_sequence({torch::ones({3, 2})});
    // 使用 `m` 对 `packed_input` 进行前向传播，得到 `rnn_output`
    auto rnn_output = m->forward_with_packed_input(packed_input);
    // 创建预期输出张量，包含预期的数值
    auto expected_output = torch::tensor(
        {{-0.1134, 0.0467, 0.2336},
         {-0.1189, 0.0502, 0.2960},
         {-0.1138, 0.0484, 0.3110}});
    // 使用 `torch::allclose` 函数验证 `rnn_output` 的第一个元素与预期输出的接近程度
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));

    // 测试向 `GRU::forward_with_packed_input` 传递可选参数
    rnn_output = m->forward_with_packed_input(packed_input, torch::Tensor());
    // 再次使用 `torch::allclose` 函数验证更新后的 `rnn_output` 的第一个元素与预期输出的接近程度
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));
  }
}

# 在 RNNTest 测试类中定义 CheckErrorInfos 方法
TEST_F(RNNTest, CheckErrorInfos) {
  {
    # 创建 RNN 的选项，设置隐藏层大小为1，层数为1
    auto options = torch::nn::RNNOptions(1, 0).num_layers(1);
    # 断言 RNN 构造函数会抛出异常并且异常信息为 "hidden_size must be greater than zero"
    ASSERT_THROWS_WITH(RNN(options), "hidden_size must be greater than zero");

    # 重新设置选项，隐藏层大小为1，层数为0
    options = torch::nn::RNNOptions(1, 1).num_layers(0);
    # 断言 RNN 构造函数会抛出异常并且异常信息为 "num_layers must be greater than zero"
    ASSERT_THROWS_WITH(RNN(options), "num_layers must be greater than zero");
  }
  {
    # 创建 LSTM 的选项，设置隐藏层大小为1，层数为1
    auto options = torch::nn::LSTMOptions(1, 0).num_layers(1);
    # 断言 LSTM 构造函数会抛出异常并且异常信息为 "hidden_size must be greater than zero"
    ASSERT_THROWS_WITH(LSTM(options), "hidden_size must be greater than zero");

    # 重新设置选项，隐藏层大小为1，层数为0
    options = torch::nn::LSTMOptions(1, 1).num_layers(0);
    # 断言 LSTM 构造函数会抛出异常并且异常信息为 "num_layers must be greater than zero"
    ASSERT_THROWS_WITH(LSTM(options), "num_layers must be greater than zero");
  }
  {
    # 创建 GRU 的选项，设置隐藏层大小为1，层数为1
    auto options = torch::nn::GRUOptions(1, 0).num_layers(1);
    # 断言 GRU 构造函数会抛出异常并且异常信息为 "hidden_size must be greater than zero"
    ASSERT_THROWS_WITH(GRU(options), "hidden_size must be greater than zero");

    # 重新设置选项，隐藏层大小为1，层数为0
    options = torch::nn::GRUOptions(1, 1).num_layers(0);
    # 断言 GRU 构造函数会抛出异常并且异常信息为 "num_layers must be greater than zero"
    ASSERT_THROWS_WITH(GRU(options), "num_layers must be greater than zero");
  }
}

# 此测试用例确保 pad_packed_sequence 在使用 CUDA 张量时不会崩溃，
# 详情参见 https://github.com/pytorch/pytorch/issues/115027
TEST_F(RNNTest, CheckPadPackedSequenceWithCudaTensors_CUDA) {
  # 在 GPU 上创建输入张量，大小为 5x5
  auto input = torch::randn({5, 5}).to(at::ScalarType::Float).cuda();
  # 创建长度张量，每个元素为5
  auto lengths = torch::full({5}, 5);

  # 对输入张量进行打包，不进行排序，不进行填充
  auto packed =
      torch::nn::utils::rnn::pack_padded_sequence(input, lengths, false, false);

  # 调用 pad_packed_sequence 函数进行填充
  auto error = torch::nn::utils::rnn::pad_packed_sequence(packed);
}
```