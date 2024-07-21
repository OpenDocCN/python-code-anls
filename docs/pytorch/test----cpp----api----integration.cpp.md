# `.\pytorch\test\cpp\api\integration.cpp`

```
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <c10/util/irange.h> // 引入 C10 库的工具和范围定义
#include <torch/torch.h> // 引入 PyTorch C++ API 的头文件

#include <test/cpp/api/support.h> // 引入测试支持函数的头文件

#include <cmath> // 数学函数库，例如 cos, sin
#include <cstdlib> // 标准库函数，例如 getenv
#include <random> // 随机数生成器库

using namespace torch::nn; // 使用 PyTorch 的 nn 命名空间
using namespace torch::test; // 使用 PyTorch 测试相关的命名空间

const double kPi = 3.1415926535898; // 定义常量 pi

class CartPole {
  // 从 openai/gym 的 cartpole.py 翻译而来

 public:
  double gravity = 9.8; // 重力加速度
  double masscart = 1.0; // 小车的质量
  double masspole = 0.1; // 杆的质量
  double total_mass = (masspole + masscart); // 总质量
  double length = 0.5; // 杆的实际长度的一半
  double polemass_length = (masspole * length); // 杆的质量乘以长度
  double force_mag = 10.0; // 施加在小车上的力的大小
  double tau = 0.02; // 状态更新之间的时间间隔（秒）

  // 导致 episode 失败的角度阈值
  double theta_threshold_radians = 12 * 2 * kPi / 360; // 弧度制的角度阈值
  double x_threshold = 2.4; // 小车位置的阈值
  int steps_beyond_done = -1; // episode 结束后的步数

  torch::Tensor state; // 状态向量
  double reward; // 当前奖励
  bool done; // 是否结束
  int step_ = 0; // 当前步数

  torch::Tensor getState() { // 获取当前状态的函数
    return state;
  }

  double getReward() { // 获取当前奖励的函数
    return reward;
  }

  double isDone() { // 判断是否结束的函数
    return done;
  }

  void reset() { // 重置环境状态的函数
    state = torch::empty({4}).uniform_(-0.05, 0.05); // 初始化状态向量为均匀分布在 [-0.05, 0.05] 内
    steps_beyond_done = -1; // 重置超出结束步数的计数
    step_ = 0; // 重置步数
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CartPole() { // 构造函数，初始化 CartPole 对象
    reset(); // 调用 reset 函数初始化
  }

  void step(int action) { // 执行环境的一步动作的函数
    auto x = state[0].item<float>(); // 获取小车位置
    auto x_dot = state[1].item<float>(); // 获取小车速度
    auto theta = state[2].item<float>(); // 获取杆角度
    auto theta_dot = state[3].item<float>(); // 获取杆角速度

    auto force = (action == 1) ? force_mag : -force_mag; // 根据动作决定施加的力的方向
    auto costheta = std::cos(theta); // 计算角度的余弦值
    auto sintheta = std::sin(theta); // 计算角度的正弦值
    auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta) /
        total_mass; // 计算临时变量
    auto thetaacc = (gravity * sintheta - costheta * temp) /
        (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)); // 计算角加速度
    auto xacc = temp - polemass_length * thetaacc * costheta / total_mass; // 计算位置加速度

    x = x + tau * x_dot; // 更新小车位置
    x_dot = x_dot + tau * xacc; // 更新小车速度
    theta = theta + tau * theta_dot; // 更新杆角度
    theta_dot = theta_dot + tau * thetaacc; // 更新杆角速度
    state = torch::tensor({x, x_dot, theta, theta_dot}); // 更新状态向量

    // 判断是否结束
    done = x < -x_threshold || x > x_threshold ||
        theta < -theta_threshold_radians || theta > theta_threshold_radians ||
        step_ > 200;

    if (!done) { // 如果未结束，设置奖励为 1.0
      reward = 1.0;
    } else if (steps_beyond_done == -1) { // 如果第一次结束，设置奖励为 0，并记录步数
      // Pole just fell!
      steps_beyond_done = 0;
      reward = 0;
    } else { // 如果已经结束过一次，则报错
      if (steps_beyond_done == 0) {
        AT_ASSERT(false); // 不能执行这个操作
      }
    }
    step_++; // 步数加一
  }
};

template <typename M, typename F, typename O>
bool test_mnist(
    size_t batch_size,
    size_t number_of_epochs,
    bool with_cuda,
    M&& model,
    F&& forward_op,
    O&& optimizer) {
  std::string mnist_path = "mnist"; // 设置 mnist 数据集路径
  if (const char* user_mnist_path = getenv("TORCH_CPP_TEST_MNIST_PATH")) { // 检查环境变量是否设置了 mnist 数据集路径
  // 将用户提供的 MNIST 数据集路径赋给变量 mnist_path
  mnist_path = user_mnist_path;

  // 创建训练数据集，使用 MNIST 数据集，并对数据进行堆叠操作
  auto train_dataset =
      torch::data::datasets::MNIST(
          mnist_path, torch::data::datasets::MNIST::Mode::kTrain)
          .map(torch::data::transforms::Stack<>());

  // 创建数据加载器，用于加载训练数据集，指定批处理大小
  auto data_loader =
      torch::data::make_data_loader(std::move(train_dataset), batch_size);

  // 根据是否有 CUDA 加速选择设备，将模型移到对应设备上
  torch::Device device(with_cuda ? torch::kCUDA : torch::kCPU);
  model->to(device);

  // 循环训练模型指定的轮数
  for (const auto epoch : c10::irange(number_of_epochs)) {
    (void)epoch; // 抑制未使用变量的警告
    // 遍历训练数据加载器中的每个批次数据
    for (torch::data::Example<> batch : *data_loader) {
      auto data = batch.data.to(device); // 将批次数据移到指定设备上
      auto targets = batch.target.to(device); // 将批次目标移到指定设备上
      // 前向传播，生成预测结果
      torch::Tensor prediction = forward_op(std::move(data));
      // 计算负对数似然损失
      torch::Tensor loss = torch::nll_loss(prediction, std::move(targets));
      // 断言损失不包含 NaN 值
      AT_ASSERT(!torch::isnan(loss).any().item<int64_t>());
      // 梯度清零
      optimizer.zero_grad();
      // 反向传播，计算梯度
      loss.backward();
      // 更新模型参数
      optimizer.step();
    }
  }

  // 禁止梯度计算
  torch::NoGradGuard guard;
  // 创建测试数据集，使用 MNIST 数据集中的测试模式
  torch::data::datasets::MNIST test_dataset(
      mnist_path, torch::data::datasets::MNIST::Mode::kTest);
  // 将测试集图像和标签移到指定设备上
  auto images = test_dataset.images().to(device),
       targets = test_dataset.targets().to(device);

  // 前向传播测试集图像，获取预测结果
  auto result = std::get<1>(forward_op(images).max(/*dim=*/1));
  // 计算预测结果与目标标签匹配的情况，并转换为 float32 类型
  torch::Tensor correct = (result == targets).to(torch::kFloat32);
  // 判断分类准确率是否超过测试集大小的 80%
  return correct.sum().item<float>() > (test_dataset.size().value() * 0.8);
auto finishEpisode = [&] {
  auto R = 0.;
  // 计算累积回报
  for (int i = rewards.size() - 1; i >= 0; i--) {
    R = rewards[i] + 0.99 * R;
    rewards[i] = R;
  }
  // 创建张量以存储标准化后的回报
  auto r_t = torch::from_blob(
      rewards.data(), {static_cast<int64_t>(rewards.size())});
  r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-5);

  // 存储每一步的策略损失和值函数损失
  std::vector<torch::Tensor> policy_loss;
  std::vector<torch::Tensor> value_loss;
  for (const auto i : c10::irange(0U, saved_log_probs.size())) {
    auto advantage = r_t[i] - saved_values[i].item<float>();
    // 计算策略损失
    policy_loss.push_back(-advantage * saved_log_probs[i]);
    // 计算值函数损失
    value_loss.push_back(
        torch::smooth_l1_loss(saved_values[i], torch::ones(1) * r_t[i]));
  }

  // 计算总损失
  auto loss =
      torch::stack(policy_loss).sum() + torch::stack(value_loss).sum();

  // 清空梯度，进行反向传播和优化步骤
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();

  // 清空记录的奖励、策略和值函数的日志概率
  rewards.clear();
  saved_log_probs.clear();
  saved_values.clear();
};
    // 如果当前 episode 是 10 的倍数，打印以下信息：
    // "Episode %i\tLast length: %5d\tAverage length: %.2f\n"
    // 其中 %i 为当前 episode 数字，%5d 为 t 的值（右对齐至少占5个字符宽度），%.2f 为 running_reward 的值（保留两位小数）
    if (episode % 10 == 0) {
      printf("Episode %i\tLast length: %5d\tAverage length: %.2f\n",
              episode, t, running_reward);
    }
    // 如果 running_reward 大于 150，则跳出循环
    if (running_reward > 150) {
      break;
    }
    // 断言：episode 应该小于 3000，如果不满足将会触发断言错误
    ASSERT_LT(episode, 3000);
  }
}

# 定义一个名为 `IntegrationTest` 的测试类的测试用例 `MNIST_CUDA`
TEST_F(IntegrationTest, MNIST_CUDA) {
  # 设置随机种子为0，确保结果可重复
  torch::manual_seed(0);
  # 创建一个简单的模型容器
  auto model = std::make_shared<SimpleContainer>();
  # 向模型中添加第一个卷积层，输入通道为1，输出通道为10，卷积核大小为5
  auto conv1 = model->add(Conv2d(1, 10, 5), "conv1");
  # 向模型中添加第二个卷积层，输入通道为10，输出通道为20，卷积核大小为5
  auto conv2 = model->add(Conv2d(10, 20, 5), "conv2");
  # 创建一个Dropout层，丢弃率为0.3
  auto drop = Dropout(0.3);
  # 创建一个2D Dropout层，丢弃率为0.3
  auto drop2d = Dropout2d(0.3);
  # 向模型中添加第一个全连接层，输入大小为320，输出大小为50
  auto linear1 = model->add(Linear(320, 50), "linear1");
  # 向模型中添加第二个全连接层，输入大小为50，输出大小为10
  auto linear2 = model->add(Linear(50, 10), "linear2");

  # 定义前向传播函数，接受一个Tensor x作为输入，返回预测结果
  auto forward = [&](torch::Tensor x) {
    # 对输入进行第一次卷积，然后最大池化操作，接着使用ReLU激活函数
    x = torch::max_pool2d(conv1->forward(x), {2, 2}).relu();
    # 对第二个卷积层进行前向传播
    x = conv2->forward(x);
    # 使用2D Dropout进行前向传播
    x = drop2d->forward(x);
    # 再次进行最大池化操作，接着使用ReLU激活函数
    x = torch::max_pool2d(x, {2, 2}).relu();

    # 将Tensor x展平为一维，第一维度自动计算，第二维度为320
    x = x.view({-1, 320});
    # 对第一个全连接层进行前向传播，然后使用clamp_min将所有负值置为0
    x = linear1->forward(x).clamp_min(0);
    # 使用Dropout进行前向传播
    x = drop->forward(x);
    # 对第二个全连接层进行前向传播
    x = linear2->forward(x);
    # 对输出进行log_softmax处理，用于多类别分类问题
    x = torch::log_softmax(x, 1);
    return x;
  };

  # 创建一个SGD优化器，学习率为0.01，动量为0.5，优化模型参数
  auto optimizer = torch::optim::SGD(
      model->parameters(), torch::optim::SGDOptions(1e-2).momentum(0.5));

  # 调用测试函数test_mnist，验证模型在MNIST数据集上的表现
  ASSERT_TRUE(test_mnist(
      32, // batch_size
      3, // number_of_epochs
      true, // with_cuda
      model,
      forward,
      optimizer));
}

# 定义一个名为 `IntegrationTest` 的测试类的测试用例 `MNISTBatchNorm_CUDA`
TEST_F(IntegrationTest, MNISTBatchNorm_CUDA) {
  # 设置随机种子为0，确保结果可重复
  torch::manual_seed(0);
  # 创建一个简单的模型容器
  auto model = std::make_shared<SimpleContainer>();
  # 向模型中添加第一个卷积层，输入通道为1，输出通道为10，卷积核大小为5
  auto conv1 = model->add(Conv2d(1, 10, 5), "conv1");
  # 向模型中添加BatchNorm2d层，对第一个卷积层的输出进行批归一化
  auto batchnorm2d = model->add(BatchNorm2d(10), "batchnorm2d");
  # 向模型中添加第二个卷积层，输入通道为10，输出通道为20，卷积核大小为5
  auto conv2 = model->add(Conv2d(10, 20, 5), "conv2");
  # 向模型中添加第一个全连接层，输入大小为320，输出大小为50
  auto linear1 = model->add(Linear(320, 50), "linear1");
  # 向模型中添加BatchNorm1d层，对第一个全连接层的输出进行批归一化
  auto batchnorm1 = model->add(BatchNorm1d(50), "batchnorm1");
  # 向模型中添加第二个全连接层，输入大小为50，输出大小为10
  auto linear2 = model->add(Linear(50, 10), "linear2");

  # 定义前向传播函数，接受一个Tensor x作为输入，返回预测结果
  auto forward = [&](torch::Tensor x) {
    # 对输入进行第一次卷积，然后最大池化操作，接着使用ReLU激活函数
    x = torch::max_pool2d(conv1->forward(x), {2, 2}).relu();
    # 对第一个批归一化层进行前向传播
    x = batchnorm2d->forward(x);
    # 对第二个卷积层进行前向传播
    x = conv2->forward(x);
    # 再次进行最大池化操作，接着使用ReLU激活函数
    x = torch::max_pool2d(x, {2, 2}).relu();

    # 将Tensor x展平为一维，第一维度自动计算，第二维度为320
    x = x.view({-1, 320});
    # 对第一个全连接层进行前向传播，然后使用批归一化操作
    x = linear1->forward(x).clamp_min(0);
    # 对第一个批归一化层进行前向传播
    x = batchnorm1->forward(x);
    # 对第二个全连接层进行前向传播
    x = linear2->forward(x);
    # 对输出进行log_softmax处理，用于多类别分类问题
    x = torch::log_softmax(x, 1);
    return x;
  };

  # 创建一个SGD优化器，学习率为0.01，动量为0.5，优化模型参数
  auto optimizer = torch::optim::SGD(
      model->parameters(), torch::optim::SGDOptions(1e-2).momentum(0.5));

  # 调用测试函数test_mnist，验证模型在MNIST数据集上的表现
  ASSERT_TRUE(test_mnist(
      32, // batch_size
      3, // number_of_epochs
      true, // with_cuda
      model,
      forward,
      optimizer));
}
```