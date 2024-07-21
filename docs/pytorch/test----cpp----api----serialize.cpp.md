# `.\pytorch\test\cpp\api\serialize.cpp`

```
#include <gtest/gtest.h>

#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/tempfile.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace torch::test;
using namespace torch::nn;
using namespace torch::optim;

namespace {
// 定义一个简单的序列模型，包括输入层、激活函数、输出层、激活函数
Sequential xor_model() {
  return Sequential(
      Linear(2, 8),  // 输入层，接受两个输入，输出八个特征
      Functional(at::sigmoid),  // 使用 sigmoid 函数进行激活
      Linear(8, 1),   // 输出层，接受八个输入特征，输出一个值
      Functional(at::sigmoid));  // 使用 sigmoid 函数进行激活
}

// 将输入张量保存到流中并重新加载
torch::Tensor save_and_load(torch::Tensor input) {
  std::stringstream stream;  // 创建一个字符串流
  torch::save(input, stream);  // 将输入张量保存到字符串流中
  torch::Tensor tensor;  // 声明一个张量
  torch::load(tensor, stream);  // 从字符串流中加载数据到张量
  return tensor;  // 返回加载的张量
}
} // namespace

// 检查两个优化器参数组是否相等的模板函数
template <typename DerivedOptions>
void is_optimizer_param_group_equal(
    const OptimizerParamGroup& lhs,  // 左侧优化器参数组
    const OptimizerParamGroup& rhs) {  // 右侧优化器参数组
  const auto& lhs_params = lhs.params();  // 获取左侧参数列表
  const auto& rhs_params = rhs.params();  // 获取右侧参数列表

  ASSERT_TRUE(lhs_params.size() == rhs_params.size());  // 断言参数列表长度相等
  for (const auto j : c10::irange(lhs_params.size())) {  // 遍历参数列表
    ASSERT_TRUE(torch::equal(lhs_params[j], rhs_params[j]));  // 断言每个参数相等
  }
  ASSERT_TRUE(
      static_cast<const DerivedOptions&>(lhs.options()) ==  // 断言左侧选项和右侧选项相等
      static_cast<const DerivedOptions&>(rhs.options()));
}

// 检查两个优化器状态是否相等的模板函数
template <typename DerivedOptimizerParamState>
void is_optimizer_state_equal(
    const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
        lhs_state,  // 左侧优化器状态
    const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
        rhs_state) {  // 右侧优化器状态
  ASSERT_TRUE(lhs_state.size() == rhs_state.size());  // 断言状态映射大小相等
  for (const auto& value : lhs_state) {  // 遍历左侧状态映射
    auto found = rhs_state.find(value.first);  // 在右侧状态映射中查找对应键
    ASSERT_TRUE(found != rhs_state.end());  // 断言找到了对应的键
    const DerivedOptimizerParamState& lhs_curr_state =
        static_cast<const DerivedOptimizerParamState&>(*(value.second.get()));  // 获取左侧当前状态
    const DerivedOptimizerParamState& rhs_curr_state =
        static_cast<const DerivedOptimizerParamState&>(*(found->second.get()));  // 获取右侧当前状态
    ASSERT_TRUE(lhs_curr_state == rhs_curr_state);  // 断言当前状态相等
  }
}

// 测试序列化优化器的模板函数
template <
    typename OptimizerClass,
    typename DerivedOptimizerOptions,
    typename DerivedOptimizerParamState>
void test_serialize_optimizer(
    DerivedOptimizerOptions options,  // 优化器选项
    bool only_has_global_state = false) {  // 是否仅有全局状态
  torch::manual_seed(0);  // 设置随机种子
  auto model1 = Linear(5, 2);  // 创建线性模型1
  auto model2 = Linear(5, 2);  // 创建线性模型2
  auto model3 = Linear(5, 2);  // 创建线性模型3

  // 将模型1保存到临时文件中并加载到模型2和模型3中
  auto model_tempfile = c10::make_tempfile();
  torch::save(model1, model_tempfile.name);
  torch::load(model2, model_tempfile.name);
  torch::load(model3, model_tempfile.name);

  auto param1 = model1->named_parameters();  // 获取模型1的命名参数
  auto param2 = model2->named_parameters();  // 获取模型2的命名参数
  auto param3 = model3->named_parameters();  // 获取模型3的命名参数
  for (const auto& p : param1) {  // 遍历模型1的命名参数
    ASSERT_TRUE(p->allclose(param2[p.key()]));  // 断言模型1和模型2的对应参数近似相等
  ASSERT_TRUE(param2[p.key()].allclose(param3[p.key()]));
  // 断言：检查模型参数 param2 和 param3 的对应键的数值是否非常接近

  // 创建一些优化器
  auto optim1 = OptimizerClass(
      {torch::optim::OptimizerParamGroup(model1->parameters())}, options);
  auto optim2 = OptimizerClass(model2->parameters(), options);
  auto optim2_2 = OptimizerClass(model2->parameters(), options);
  auto optim3 = OptimizerClass(model3->parameters(), options);
  auto optim3_2 = OptimizerClass(model3->parameters(), options);

  // 对 optim3_2 的每个参数组进行处理
  for (auto& param_group : optim3_2.param_groups()) {
    const double lr = param_group.options().get_lr();
    // 改变学习率，这将在加载时被覆盖，以确保测试可以检查选项的保存和加载是否正确
    param_group.options().set_lr(lr + 0.01);
  }

  auto x = torch::ones({10, 5});

  // 定义一个优化步骤的闭包函数
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();  // 梯度清零
    auto y = model->forward(x).sum();  // 计算模型前向传播后的结果总和
    y.backward();  // 反向传播
    auto closure = []() { return torch::tensor({10}); };  // 一个简单的闭包函数
    optimizer.step(closure);  // 执行优化步骤
  };

  // 对 model1 执行两步优化
  step(optim1, model1);
  step(optim1, model1);

  // 对 model2 执行两步优化，但不保存优化器状态
  step(optim2, model2);
  step(optim2_2, model2);

  // 对 model3 执行一步优化
  step(optim3, model3);

  // 保存优化器状态到临时文件
  auto optim_tempfile = c10::make_tempfile();
  torch::save(optim3, optim_tempfile.name);
  // 从临时文件加载优化器状态到 optim3_2
  torch::load(optim3_2, optim_tempfile.name);

  auto& optim3_2_param_groups = optim3_2.param_groups();
  auto& optim3_param_groups = optim3.param_groups();
  auto& optim3_2_state = optim3_2.state();
  auto& optim3_state = optim3.state();

  // 断言：optim3_2 和 optim1 的参数组和状态大小应该分别为 1 和 state_size
  ASSERT_TRUE(optim3_2_param_groups.size() == 1);
  unsigned state_size = only_has_global_state ? 1 : 2;
  ASSERT_TRUE(optim3_2_state.size() == state_size);

  // 断言：optim3_2 和 optim1 的参数组和状态应该有相同的大小
  ASSERT_TRUE(optim3_2_param_groups.size() == optim3_param_groups.size());
  ASSERT_TRUE(optim3_2_state.size() == optim3_state.size());

  // 检查 optimizer.param_groups_ 和 optimizer.state_ 的序列化逻辑的正确性
  for (const auto i : c10::irange(optim3_2_param_groups.size())) {
    is_optimizer_param_group_equal<DerivedOptimizerOptions>(
        optim3_2_param_groups[i], optim3_param_groups[i]);
    is_optimizer_state_equal<DerivedOptimizerParamState>(
        optim3_2_state, optim3_state);
  }

  // 对 model3 执行第二步优化
  step(optim3_2, model3);

  // 获取 model1、model2、model3 的命名参数
  param1 = model1->named_parameters();
  param2 = model2->named_parameters();
  param3 = model3->named_parameters();

  for (const auto& p : param1) {
    const auto& name = p.key();
    // 断言：检查模型参数 param1 和 param3 对应键的 L2 范数是否相等
    ASSERT_TRUE(
        param1[name].norm().item<float>() == param3[name].norm().item<float>());
  }
    ASSERT_TRUE(
        param1[name].norm().item<float>() != param2[name].norm().item<float>());
  }



    # 断言语句，用于验证两个张量的指定属性不相等
    ASSERT_TRUE(
        # 检查 param1 和 param2 中名为 name 的张量的范数（norm）值不相等
        param1[name].norm().item<float>() != param2[name].norm().item<float>());
  }


这段代码使用了一个断言语句 `ASSERT_TRUE`，用于在程序运行时检查条件是否为真。它验证了两个张量（`param1[name]` 和 `param2[name]`）的范数值（通过 `.norm()` 方法获取）是否不相等。
// 定义一个函数，用于将 int64_t 类型的值保存到序列化输出存档中
void write_int_value(
    torch::serialize::OutputArchive& archive, // 输出存档对象的引用
    const std::string& key, // 要保存的键值的字符串引用
    const int64_t& value) { // 要保存的 int64_t 值的引用
  archive.write(key, c10::IValue(value)); // 将值写入输出存档
}

// 用于保存一组缓冲区的实用函数模板
template <typename BufferContainer>
void write_tensors_to_archive(
    torch::serialize::OutputArchive& archive, // 输出存档对象的引用
    const std::string& key, // 要保存的键值的字符串引用
    const BufferContainer& buffers) { // 包含缓冲区的容器的引用
  archive.write(
      key + "/size", torch::tensor(static_cast<int64_t>(buffers.size()))); // 保存缓冲区大小信息
  for (const auto index : c10::irange(buffers.size())) { // 遍历缓冲区容器
    archive.write(
        key + "/" + std::to_string(index), buffers[index], /*is_buffer=*/true); // 保存每个缓冲区到输出存档
  }
}

// 保存一组步骤缓冲区的实用函数
void write_step_buffers(
    torch::serialize::OutputArchive& archive, // 输出存档对象的引用
    const std::string& key, // 要保存的键值的字符串引用
    const std::vector<int64_t>& steps) { // 包含步骤的 int64_t 向量的引用
  std::vector<torch::Tensor> tensors; // 创建一个存放张量的向量
  tensors.reserve(steps.size()); // 预留步骤向量大小的空间
  for (const auto& step : steps) { // 遍历步骤向量
    tensors.push_back(torch::tensor(static_cast<int64_t>(step))); // 将每个步骤转换为张量并存入向量
  }
  write_tensors_to_archive(archive, key, tensors); // 调用保存缓冲区的函数，将张量向量保存到输出存档
}

// 定义一个宏，用于检查旧的序列化逻辑警告
#define OLD_SERIALIZATION_LOGIC_WARNING_CHECK(funcname, optimizer, filename) \
  {                                                                          \
    WarningCapture warnings;                                                 \
    funcname(optimizer, filename);                                           \
    ASSERT_EQ(                                                               \
        count_substr_occurrences(warnings.str(), "old serialization"), 1);   \
  }

// SerializeTest 测试套件中的 KeysFunc 测试
TEST(SerializeTest, KeysFunc) {
  auto tempfile = c10::make_tempfile(); // 创建临时文件
  torch::serialize::OutputArchive output_archive; // 创建输出存档对象
  for (const auto i : c10::irange(3)) { // 循环三次
    output_archive.write(
        "element/" + std::to_string(i), c10::IValue(static_cast<int64_t>(i))); // 将索引值写入输出存档
  }
  output_archive.save_to(tempfile.name); // 将输出存档保存到临时文件
  torch::serialize::InputArchive input_archive; // 创建输入存档对象
  input_archive.load_from(tempfile.name); // 从临时文件加载输入存档
  std::vector<std::string> keys = input_archive.keys(); // 获取输入存档中的键列表
  ASSERT_EQ(keys.size(), 3); // 断言键列表长度为3
  for (const auto i : c10::irange(keys.size())) { // 遍历键列表
    ASSERT_EQ(keys[i], "element/" + std::to_string(i)); // 断言键与预期的格式相符
  }
}

// SerializeTest 测试套件中的 TryReadFunc 测试
TEST(SerializeTest, TryReadFunc) {
  auto tempfile = c10::make_tempfile(); // 创建临时文件
  torch::serialize::OutputArchive output_archive; // 创建输出存档对象
  for (const auto i : c10::irange(3)) { // 循环三次
    output_archive.write(
        "element/" + std::to_string(i), c10::IValue(static_cast<int64_t>(i))); // 将索引值写入输出存档
  }
  output_archive.save_to(tempfile.name); // 将输出存档保存到临时文件
  torch::serialize::InputArchive input_archive; // 创建输入存档对象
  input_archive.load_from(tempfile.name); // 从临时文件加载输入存档
  c10::IValue ivalue; // 创建存放 IValue 的变量
  ASSERT_FALSE(input_archive.try_read("1", ivalue)); // 尝试读取不存在的键，断言失败
  ASSERT_TRUE(input_archive.try_read("element/1", ivalue)); // 尝试读取存在的键，断言成功
  ASSERT_EQ(ivalue.toInt(), 1); // 断言读取的值为预期的整数
}
TEST(SerializeTest, Basic) {
  // 设置随机种子为0，以确保结果可复现
  torch::manual_seed(0);

  // 生成一个大小为[5, 5]的随机张量x
  auto x = torch::randn({5, 5});

  // 对x进行序列化和反序列化操作，得到张量y
  auto y = save_and_load(x);

  // 断言y已被定义
  ASSERT_TRUE(y.defined());

  // 断言x和y的大小相同
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());

  // 断言x和y在数值上相近
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, MathBits) {
  // 设置随机种子为0，以确保结果可复现
  torch::manual_seed(0);

  // 创建一个复数浮点数类型的选项
  auto options = torch::TensorOptions{}.dtype(torch::kComplexFloat);

  // 生成一个大小为[5, 5]的随机张量x，使用复数浮点数选项
  auto x = torch::randn({5, 5}, options);

  {
    // 对x进行共轭操作，得到期望的张量
    auto expected = torch::conj(x);

    // 对期望的张量进行序列化和反序列化操作，得到实际的张量
    auto actual = save_and_load(expected);

    // 断言实际的张量已被定义
    ASSERT_TRUE(actual.defined());

    // 断言实际的张量和期望的张量的大小相同
    ASSERT_EQ(actual.sizes().vec(), expected.sizes().vec());

    // 断言实际的张量和期望的张量在数值上相近
    ASSERT_TRUE(actual.allclose(expected));
  }

  {
    // 对x进行负视图操作，得到期望的张量
    auto expected = torch::_neg_view(x);

    // 对期望的张量进行序列化和反序列化操作，得到实际的张量
    auto actual = save_and_load(expected);

    // 断言实际的张量已被定义
    ASSERT_TRUE(actual.defined());

    // 断言实际的张量和期望的张量的大小相同
    ASSERT_EQ(actual.sizes().vec(), expected.sizes().vec());

    // 断言实际的张量和期望的张量在数值上相近
    ASSERT_TRUE(actual.allclose(expected));
  }

  {
    // 对x先进行负视图操作，然后再进行共轭操作，得到期望的张量
    auto expected = torch::conj(torch::_neg_view(x));

    // 对期望的张量进行序列化和反序列化操作，得到实际的张量
    auto actual = save_and_load(expected);

    // 断言实际的张量已被定义
    ASSERT_TRUE(actual.defined());

    // 断言实际的张量和期望的张量的大小相同
    ASSERT_EQ(actual.sizes().vec(), expected.sizes().vec());

    // 断言实际的张量和期望的张量在数值上相近
    ASSERT_TRUE(actual.allclose(expected));
  }

  {
    // 测试对ZeroTensor的序列化，预期会抛出异常
    // 因为ZeroTensor目前不支持序列化，不对外公开
    auto t = torch::_efficientzerotensor({5, 5});
    ASSERT_THROWS_WITH(save_and_load(t), "ZeroTensor is not serializable,");
  }
}

TEST(SerializeTest, BasicToFile) {
  // 设置随机种子为0，以确保结果可复现
  torch::manual_seed(0);

  // 生成一个大小为[5, 5]的随机张量x
  auto x = torch::randn({5, 5});

  // 创建一个临时文件来保存张量x
  auto tempfile = c10::make_tempfile();

  // 将张量x保存到临时文件中
  torch::save(x, tempfile.name);

  // 加载保存在临时文件中的张量到张量y
  torch::Tensor y;
  torch::load(y, tempfile.name);

  // 断言y已被定义
  ASSERT_TRUE(y.defined());

  // 断言x和y的大小相同
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());

  // 断言x和y在数值上相近
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, BasicViaFunc) {
  // 设置随机种子为0，以确保结果可复现
  torch::manual_seed(0);

  // 生成一个大小为[5, 5]的随机张量x
  auto x = torch::randn({5, 5});

  // 创建一个空字符串来保存序列化后的张量数据
  std::string serialized;

  // 使用自定义函数将张量x序列化为字符串
  torch::save(x, [&](const void* buf, size_t n) {
    serialized.append(reinterpret_cast<const char*>(buf), n);
    return n;
  });

  // 从字符串中加载数据到张量y
  torch::Tensor y;
  torch::load(y, serialized.data(), serialized.size());

  // 断言y已被定义
  ASSERT_TRUE(y.defined());

  // 断言x和y的大小相同
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());

  // 断言x和y在数值上相近
  ASSERT_TRUE(x.allclose(y));

  // 创建一个新的张量z
  torch::Tensor z;

  // 使用自定义加载函数加载数据到张量z
  torch::load(
      z,
      [&](uint64_t pos, void* buf, size_t n) -> size_t {
        if (pos >= serialized.size())
          return 0;
        size_t nbytes =
            std::min(static_cast<size_t>(pos) + n, serialized.size()) - pos;
        memcpy(buf, serialized.data() + pos, nbytes);
        return nbytes;
      },
      [&]() -> size_t { return serialized.size(); });

  // 断言z已被定义
  ASSERT_TRUE(z.defined());

  // 断言x和z的大小相同
  ASSERT_EQ(x.sizes().vec(), z.sizes().vec());

  // 断言x和z在数值上相近
  ASSERT_TRUE(x.allclose(z));
}

TEST(SerializeTest, Resized) {
  // 设置随机种子为0，以确保结果可复现
  torch::manual_seed(0);

  // 生成一个大小为[11, 5]的随机张量x
  auto x = torch::randn({11, 5});

  // 将张量x调整大小为[5, 5]
  x.resize_({5, 5});

  // 对调整大小后的张量x进行序列化和反序列化操作，得到张量y
  auto y = save_and_load(x);

  // 断言y已被定义
  ASSERT_TRUE(y.defined());

  // 断言x和y的大小相同
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());

  // 断言x和y在数值上相近
  ASSERT_TRUE(x.allclose(y));
}
TEST(SerializeTest, XOR) {
  // 定义一个损失函数，用于计算模型的二元交叉熵损失
  auto getLoss = [](Sequential model, uint32_t batch_size) {
    // 创建输入和标签张量
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    // 生成随机输入和对应的标签
    for (const auto i : c10::irange(batch_size)) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
    }
    // 计算模型的输出
    auto x = model->forward<torch::Tensor>(inputs);
    // 返回二元交叉熵损失
    return torch::binary_cross_entropy(x, labels);
  };

  // 创建三个 XOR 模型实例
  auto model = xor_model();
  auto model2 = xor_model();
  auto model3 = xor_model();
  // 创建优化器，使用 SGD 方法
  auto optimizer = torch::optim::SGD(
      model->parameters(),
      torch::optim::SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(
          1e-6));

  // 初始化运行损失和当前训练 epoch 数
  float running_loss = 1;
  int epoch = 0;
  // 在损失降到 0.1 以下之前持续训练模型
  while (running_loss > 0.1) {
    // 计算当前批次的损失
    torch::Tensor loss = getLoss(model, 4);
    // 梯度清零
    optimizer.zero_grad();
    // 计算梯度
    loss.backward();
    // 更新模型参数
    optimizer.step();

    // 更新运行损失值
    running_loss = running_loss * 0.99 + loss.sum().item<float>() * 0.01;
    // 断言 epoch 数未超过 3000
    ASSERT_LT(epoch, 3000);
    epoch++;
  }



  // 增加 epoch 计数器的值，表示训练循环的下一个周期
  epoch++;



  auto tempfile = c10::make_tempfile();
  torch::save(model, tempfile.name);
  torch::load(model2, tempfile.name);



  // 创建临时文件并保存模型到该文件中，然后加载该文件中的模型到 model2 中
  auto tempfile = c10::make_tempfile();
  torch::save(model, tempfile.name);
  torch::load(model2, tempfile.name);



  auto loss = getLoss(model2, 100);
  ASSERT_LT(loss.item<float>(), 0.1);



  // 计算使用 model2 模型在数据集上的损失，确保损失小于 0.1
  auto loss = getLoss(model2, 100);
  ASSERT_LT(loss.item<float>(), 0.1);
}

// 定义一个名为 "SerializeTest" 的测试集，包含 "Optim" 子测试
TEST(SerializeTest, Optim) {
  // 创建三个线性模型对象，每个模型有相同的输入输出维度
  auto model1 = Linear(5, 2);
  auto model2 = Linear(5, 2);
  auto model3 = Linear(5, 2);

  // 将模型1保存到临时文件中，并从文件中加载到模型2和模型3
  auto model_tempfile = c10::make_tempfile();
  torch::save(model1, model_tempfile.name);
  torch::load(model2, model_tempfile.name);
  torch::load(model3, model_tempfile.name);

  // 获取模型1、模型2和模型3的参数字典
  auto param1 = model1->named_parameters();
  auto param2 = model2->named_parameters();
  auto param3 = model3->named_parameters();

  // 检查模型1和模型2的每个参数是否接近
  for (const auto& p : param1) {
    ASSERT_TRUE(p->allclose(param2[p.key()]));
    // 检查模型2和模型3的每个参数是否接近
    ASSERT_TRUE(param2[p.key()].allclose(param3[p.key()]));
  }

  // 创建带有动量的优化器，分别为模型1、模型2和模型3
  auto optim1 = torch::optim::SGD(
      model1->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim2 = torch::optim::SGD(
      model2->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim2_2 = torch::optim::SGD(
      model2->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim3 = torch::optim::SGD(
      model3->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim3_2 = torch::optim::SGD(
      model3->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));

  // 创建一个张量 x，全为1，形状为 [10, 5]
  auto x = torch::ones({10, 5});

  // 定义一个函数 step，用于执行优化器的零梯度、前向传播、反向传播和更新步骤
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };

  // 对模型1执行两步优化
  step(optim1, model1);
  step(optim1, model1);

  // 对模型2执行两步优化，但不保存优化器
  step(optim2, model2);
  step(optim2_2, model2);

  // 对模型3执行两步优化，并保存优化器
  step(optim3, model3);

  // 将优化器3保存到临时文件中，并从文件中加载到优化器3_2
  auto optim_tempfile = c10::make_tempfile();
  torch::save(optim3, optim_tempfile.name);
  torch::load(optim3_2, optim_tempfile.name);
  // 对模型3使用加载的优化器执行一步优化
  step(optim3_2, model3);

  // 再次获取模型1、模型2和模型3的参数字典
  param1 = model1->named_parameters();
  param2 = model2->named_parameters();
  param3 = model3->named_parameters();

  // 检查模型1和模型3的每个参数的范数是否相等
  for (const auto& p : param1) {
    const auto& name = p.key();
    // 检查模型1和模型3的每个参数的范数是否相等
    ASSERT_TRUE(
        param1[name].norm().item<float>() == param3[name].norm().item<float>());
    // 检查模型1和模型2的每个参数的范数是否不相等
    ASSERT_TRUE(
        param1[name].norm().item<float>() != param2[name].norm().item<float>());
  }
}

// 定义一个名为 "SerializeTest" 的测试集，包含 "Optim_Adagrad" 子测试
TEST(SerializeTest, Optim_Adagrad) {
  // 测试 Adagrad 优化器的序列化与反序列化
  test_serialize_optimizer<Adagrad, AdagradOptions, AdagradParamState>(
      AdagradOptions(1e-1));

  // 兼容性检查
  // 创建一个线性模型对象
  auto model1 = Linear(5, 2);
  // 使用 Adagrad 优化器优化模型1的参数
  auto optim1 = torch::optim::Adagrad(
      model1->parameters(), torch::optim::AdagradOptions(1e-1));

  // 创建一个张量 x，全为1，形状为 [10, 5]
  auto x = torch::ones({10, 5});

  // 定义一个函数 step，用于执行优化器的零梯度、前向传播、反向传播和更新步骤
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };
  // 执行优化器的单步更新操作

  step(optim1, model1);
  // 调用自定义的step函数，传入优化器optim1和模型model1进行更新

  auto optim1_2 =
      Adagrad(model1->parameters(), torch::optim::AdagradOptions(1e-1));
  // 使用Adagrad优化算法创建新的优化器optim1_2，以学习率1e-1初始化

  // fill up with optim1 sum_buffers
  std::vector<torch::Tensor> sum_buffers;
  // 用来存储optim1的sum缓冲区的向量

  // fill up with optim1 state_buffers
  std::vector<int64_t> step_buffers;
  // 用来存储optim1的step缓冲区的向量

  const auto& params_ = optim1.param_groups()[0].params();
  // 获取optim1的第一个参数组的参数列表

  const auto& optim1_state = optim1.state();
  // 获取optim1的状态信息

  for (const auto& param : params_) {
    auto key_ = param.unsafeGetTensorImpl();
    // 获取参数对应的TensorImpl作为key

    const AdagradParamState& curr_state_ =
        static_cast<const AdagradParamState&>(*(optim1_state.at(key_).get()));
    // 获取当前参数的AdagradParamState状态信息

    sum_buffers.emplace_back(curr_state_.sum());
    // 将当前参数的sum值添加到sum_buffers向量中

    step_buffers.emplace_back(curr_state_.step());
    // 将当前参数的step值添加到step_buffers向量中
  }

  // write sum_buffers and step_buffers to the file
  // 将sum_buffers和step_buffers写入文件

  auto optim_tempfile_old_format = c10::make_tempfile();
  // 创建一个临时文件来存储优化器状态

  torch::serialize::OutputArchive output_archive;
  // 创建一个输出存档对象

  write_tensors_to_archive(output_archive, "sum_buffers", sum_buffers);
  // 将sum_buffers写入output_archive存档中，使用键名"sum_buffers"

  write_step_buffers(output_archive, "step_buffers", step_buffers);
  // 将step_buffers写入output_archive存档中，使用键名"step_buffers"

  output_archive.save_to(optim_tempfile_old_format.name);
  // 将output_archive存档保存到optim_tempfile_old_format文件中

  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  // 使用旧的反序列化逻辑检查，加载optim1_2优化器的状态信息

  is_optimizer_state_equal<AdagradParamState>(optim1.state(), optim1_2.state());
  // 检查优化器optim1和optim1_2的状态是否相等，使用AdagradParamState进行比较
}

// 定义一个名为 `TEST` 的测试用例，用于测试序列化优化器 `SGD` 的行为
TEST(SerializeTest, Optim_SGD) {
  // 调用通用测试函数，测试 `SGD` 优化器的序列化行为，使用学习率 `1e-1` 和动量 `0.9`
  test_serialize_optimizer<SGD, SGDOptions, SGDParamState>(
      SGDOptions(1e-1).momentum(0.9));

  // bc 兼容性检查
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  // 添加一个张量以进行惰性初始化检查 - 当所有参数没有动量缓冲区条目时
  model1_params.emplace_back(torch::randn({2, 3}));
  // 创建 `SGD` 优化器，优化 `model1_params` 中的参数，使用学习率 `0.01` 和动量 `0.9`
  auto optim1 = torch::optim::SGD(
      model1_params, torch::optim::SGDOptions(0.01).momentum(0.9));

  auto x = torch::ones({10, 5});
  // 定义一个步骤函数 `step`，该函数接受一个优化器和线性模型作为参数，并执行一次优化步骤
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    // 将梯度置零
    optimizer.zero_grad();
    // 计算模型前向传播的结果，并对结果求和
    auto y = model->forward(x).sum();
    // 计算反向传播
    y.backward();
    // 执行优化步骤
    optimizer.step();
  };
  // 执行一次优化步骤
  step(optim1, model1);

  // 定义一个存储动量缓冲区的向量
  std::vector<at::Tensor> momentum_buffers;
  // 初始化迭代次数为 `0`
  int64_t iteration_{0};
  // 获取优化器 `optim1` 的第一个参数组的参数列表
  const auto& params_ = optim1.param_groups()[0].params();
  // 获取优化器 `optim1` 的状态
  const auto& optim1_state = optim1.state();
  // 遍历参数列表中的每一个参数
  for (const auto i : c10::irange(params_.size())) {
    // 如果不是最后一个参数
    if (i != (params_.size() - 1)) {
      // 获取当前参数的关键字
      auto key_ = params_[i].unsafeGetTensorImpl();
      // 获取当前参数对应的 `SGDParamState` 状态
      const SGDParamState& curr_state_ =
          static_cast<const SGDParamState&>(*(optim1_state.at(key_).get()));
      // 将当前参数的动量缓冲区添加到动量缓冲区向量中
      momentum_buffers.emplace_back(curr_state_.momentum_buffer());
    }
  }
  // 断言动量缓冲区的大小等于参数列表的大小减去 `1`
  ASSERT_TRUE(momentum_buffers.size() == (params_.size() - 1));
  // 将动量缓冲区写入文件
  auto optim_tempfile_old_format = c10::make_tempfile();
  // 创建输出存档对象 `output_archive`
  torch::serialize::OutputArchive output_archive;
  // 将动量缓冲区写入存档中，命名为 "momentum_buffers"
  write_tensors_to_archive(
      output_archive, "momentum_buffers", momentum_buffers);
  // 将迭代次数写入存档中，命名为 "iteration_"
  write_int_value(output_archive, "iteration_", iteration_);
  // 将存档保存到旧格式的临时文件中
  output_archive.save_to(optim_tempfile_old_format.name);
  // 使用学习率 `1e-1` 和动量 `0.9` 重新创建 `SGD` 优化器 `optim1_2`
  auto optim1_2 =
      SGD(model1_params, torch::optim::SGDOptions(1e-1).momentum(0.9));
  // 旧的序列化逻辑警告检查
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  // 检查优化器状态是否相等
  is_optimizer_state_equal<SGDParamState>(optim1.state(), optim1_2.state());
}

// 定义一个名为 `TEST` 的测试用例，用于测试序列化优化器 `Adam` 的行为
TEST(SerializeTest, Optim_Adam) {
  // 调用通用测试函数，测试 `Adam` 优化器的序列化行为，使用学习率 `0.99999`，启用 `amsgrad`，权重衰减 `0.5`
  test_serialize_optimizer<Adam, AdamOptions, AdamParamState>(
      AdamOptions().lr(0.99999).amsgrad(true).weight_decay(0.5));

  // bc 兼容性检查
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  // 添加一个张量以进行惰性初始化检查 - 当所有参数在缓冲区中没有条目时
  model1_params.emplace_back(torch::randn({2, 3}));
  // 创建 `Adam` 优化器，优化 `model1_params` 中的参数，使用权重衰减 `0.5`
  auto optim1 = torch::optim::Adam(
      model1_params, torch::optim::AdamOptions().weight_decay(0.5));

  auto x = torch::ones({10, 5});
  // 定义一个步骤函数 `step`，该函数接受一个优化器和线性模型作为参数，并执行一次优化步骤
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    // 将梯度置零
    optimizer.zero_grad();
    // 计算模型前向传播的结果，并对结果求和
    auto y = model->forward(x).sum();
    // 计算反向传播
    y.backward();

    // 执行优化步骤
    optimizer.step();
  };
  // 执行一次优化步骤
  step(optim1, model1);

  // 定义一个存储动量缓冲区的向量
  std::vector<at::Tensor> momentum_buffers;
  // 初始化迭代次数为 `0`
  int64_t iteration_{0};
  // 获取优化器 `optim1` 的第一个参数组的参数列表
  const auto& params_ = optim1.param_groups()[0].params();
  // 获取优化器 `optim1` 的状态
  const auto& optim1_state = optim1.state();
  // 遍历参数列表中的每一个参数
  for (const auto i : c10::irange(params_.size())) {
    // 如果不是最后一个参数
    if (i != (params_.size() - 1)) {
      // 获取当前参数的关键字
      auto key_ = params_[i].unsafeGetTensorImpl();
      // 获取当前参数对应的 `AdamParamState` 状态
      const AdamParamState& curr_state_ =
          static_cast<const AdamParamState&>(*(optim1_state.at(key_).get()));
      // 将当前参数的动量缓冲区添加到动量缓冲区向量中
      momentum_buffers.emplace_back(curr_state_.momentum_buffer());
    }
  }
  // 断言动量缓冲区的大小等于参数列表的大小减去 `1`
  ASSERT_TRUE(momentum_buffers.size() == (params_.size() - 1));
  // 将动量缓冲区写入文件
  auto optim_tempfile_old_format = c10::make_tempfile();
  // 创建输出存档对象 `output_archive`
  torch::serialize::OutputArchive output_archive;
  // 将动量缓冲区写入存档中，命名为 "momentum_buffers"
  write_tensors_to_archive(
      output_archive, "momentum_buffers", momentum_buffers);
  // 将迭代次数写入存档中，命名为 "iteration_"
  write_int_value(output_archive, "iteration_", iteration_);
  // 将存档保存到旧格式的临时文件中
  output_archive.save_to(optim_tempfile_old_format.name);
  // 使用默认选项重新创建 `Adam` 优化器 `optim1_2`
  auto optim1_2 =
      Adam(model1_params, torch::optim::AdamOptions().weight_decay(0.5));
  // 旧的序列化逻辑警告检查
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  // 检查优化器状态是否相等
  is_optimizer_state_equal<AdamParamState>(optim1.state(), optim1_2.state());
}
    optimizer.step();
  };

  // 执行优化器的一步优化操作，更新模型参数
  step(optim1, model1);

  // 定义存储优化器状态的缓冲区
  std::vector<int64_t> step_buffers;
  std::vector<at::Tensor> exp_average_buffers;
  std::vector<at::Tensor> exp_average_sq_buffers;
  std::vector<at::Tensor> max_exp_average_sq_buffers;

  // 获取第一个参数组的参数列表
  const auto& params_ = optim1.param_groups()[0].params();
  // 获取优化器的状态信息
  const auto& optim1_state = optim1.state();

  // 遍历参数列表
  for (const auto i : c10::irange(params_.size())) {
    // 排除最后一个参数
    if (i != (params_.size() - 1)) {
      // 获取当前参数的 TensorImpl
      auto key_ = params_[i].unsafeGetTensorImpl();
      // 获取当前参数的 AdamParamState 状态
      const AdamParamState& curr_state_ =
          static_cast<const AdamParamState&>(*(optim1_state.at(key_).get()));
      // 将当前状态的 step 添加到 step_buffers
      step_buffers.emplace_back(curr_state_.step());
      // 将当前状态的 exp_avg 添加到 exp_average_buffers
      exp_average_buffers.emplace_back(curr_state_.exp_avg());
      // 将当前状态的 exp_avg_sq 添加到 exp_average_sq_buffers
      exp_average_sq_buffers.emplace_back(curr_state_.exp_avg_sq());
      // 如果当前状态的 max_exp_avg_sq 已定义，则添加到 max_exp_average_sq_buffers
      if (curr_state_.max_exp_avg_sq().defined()) {
        max_exp_average_sq_buffers.emplace_back(curr_state_.max_exp_avg_sq());
      }
    }
  }

  // 将缓冲区数据写入文件
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  // 将 step_buffers 写入 output_archive
  write_step_buffers(output_archive, "step_buffers", step_buffers);
  // 将 exp_average_buffers 写入 output_archive
  write_tensors_to_archive(
      output_archive, "exp_average_buffers", exp_average_buffers);
  // 将 exp_average_sq_buffers 写入 output_archive
  write_tensors_to_archive(
      output_archive, "exp_average_sq_buffers", exp_average_sq_buffers);
  // 将 max_exp_average_sq_buffers 写入 output_archive
  write_tensors_to_archive(
      output_archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
  // 将 output_archive 的内容保存到文件
  output_archive.save_to(optim_tempfile_old_format.name);

  // 创建新的 Adam 优化器
  auto optim1_2 = Adam(model1_params, torch::optim::AdamOptions());
  // 加载旧格式的优化器状态
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  // 检查两个优化器状态是否相等
  is_optimizer_state_equal<AdamParamState>(optim1.state(), optim1_2.state());
}

// 定义测试用例 SerializeTest.Optim_AdamW
TEST(SerializeTest, Optim_AdamW) {
  // 调用通用函数测试序列化优化器 AdamW 的行为
  test_serialize_optimizer<AdamW, AdamWOptions, AdamWParamState>(
      AdamWOptions().lr(0.99999).amsgrad(true).betas(
          std::make_tuple(0.999, 0.1)));

  // bc 兼容性检查
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  // 添加一个张量以进行延迟初始化检查 - 当所有参数在缓冲区中都没有条目时
  model1_params.emplace_back(torch::randn({2, 3}));
  // 创建 AdamW 优化器对象，使用指定的参数和选项
  auto optim1 = torch::optim::AdamW(
      model1_params, torch::optim::AdamWOptions().weight_decay(0.5));

  auto x = torch::ones({10, 5});
  // 定义一个 lambda 函数 step，用于执行优化器的零梯度清空、前向传播、反向传播和更新步骤
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad(); // 清空梯度
    auto y = model->forward(x).sum(); // 计算模型前向传播后的输出和
    y.backward(); // 执行反向传播
    optimizer.step(); // 根据梯度更新优化器状态
  };
  step(optim1, model1); // 执行一次优化步骤

  // 定义用于存储状态的缓冲区
  std::vector<int64_t> step_buffers;
  std::vector<at::Tensor> exp_average_buffers;
  std::vector<at::Tensor> exp_average_sq_buffers;
  std::vector<at::Tensor> max_exp_average_sq_buffers;
  // 获取第一个参数组的参数列表和优化器状态
  const auto& params_ = optim1.param_groups()[0].params();
  const auto& optim1_state = optim1.state();
  // 遍历参数列表的索引
  for (const auto i : c10::irange(params_.size())) {
    // 排除最后一个参数
    if (i != (params_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      // 获取当前参数的 AdamWParamState 状态
      const AdamWParamState& curr_state_ =
          static_cast<const AdamWParamState&>(*(optim1_state.at(key_).get()));
      // 将状态的步数、指数移动平均和指数移动平方平均添加到对应的缓冲区
      step_buffers.emplace_back(curr_state_.step());
      exp_average_buffers.emplace_back(curr_state_.exp_avg());
      exp_average_sq_buffers.emplace_back(curr_state_.exp_avg_sq());
      // 如果存在最大指数移动平方平均值，则添加到对应缓冲区
      if (curr_state_.max_exp_avg_sq().defined()) {
        max_exp_average_sq_buffers.emplace_back(curr_state_.max_exp_avg_sq());
      }
    }
  }
  // 将缓冲区内容写入文件
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  write_step_buffers(output_archive, "step_buffers", step_buffers);
  write_tensors_to_archive(
      output_archive, "exp_average_buffers", exp_average_buffers);
  write_tensors_to_archive(
      output_archive, "exp_average_sq_buffers", exp_average_sq_buffers);
  write_tensors_to_archive(
      output_archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
  output_archive.save_to(optim_tempfile_old_format.name); // 保存存档到临时文件
  // 使用旧的反序列化逻辑从文件加载优化器状态，并检查日志中的警告
  auto optim1_2 = AdamW(model1_params, torch::optim::AdamWOptions());
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  // 检查序列化前后的优化器状态是否相等
  is_optimizer_state_equal<AdamWParamState>(optim1.state(), optim1_2.state());
}
TEST(SerializeTest, Optim_RMSprop) {
  // 创建 RMSprop 的选项，设置学习率、动量和是否居中
  auto options = RMSpropOptions(0.1).momentum(0.9).centered(true);
  // 测试序列化 RMSprop 优化器的功能
  test_serialize_optimizer<RMSprop, RMSpropOptions, RMSpropParamState>(options);

  // bc compatibility check
  // 创建一个线性模型，设置输入和输出维度
  auto model1 = Linear(5, 2);
  // 获取模型参数列表
  auto model1_params = model1->parameters();

  // 添加一个张量以检查惰性初始化 - 当所有参数都没有动量缓冲项时
  model1_params.emplace_back(torch::randn({2, 3}));
  // 使用 RMSprop 优化器初始化模型参数和选项
  auto optim1 = torch::optim::RMSprop(model1_params, options);

  // 创建一个 10x5 的全 1 张量
  auto x = torch::ones({10, 5});
  // 定义一个优化器步骤函数，更新模型参数
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };
  // 执行一步优化器步骤
  step(optim1, model1);

  // 初始化存储 RMSprop 优化器状态的缓冲区
  std::vector<at::Tensor> square_average_buffers;
  std::vector<at::Tensor> momentum_buffers;
  std::vector<at::Tensor> grad_average_buffers;
  // 获取优化器的参数组和状态
  const auto& params_ = optim1.param_groups()[0].params();
  const auto& optim1_state = optim1.state();
  // 遍历参数并收集相关的状态信息
  for (const auto i : c10::irange(params_.size())) {
    if (i != (params_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      // 强制转换为 RMSpropParamState 类型以获取特定参数的状态信息
      const RMSpropParamState& curr_state_ =
          static_cast<const RMSpropParamState&>(*(optim1_state.at(key_).get()));
      // 收集平方平均值缓冲区
      square_average_buffers.emplace_back(curr_state_.square_avg());
      // 如果存在动量缓冲区，则收集动量缓冲区
      if (curr_state_.momentum_buffer().defined()) {
        momentum_buffers.emplace_back(curr_state_.momentum_buffer());
      }
      // 如果存在梯度平均值缓冲区，则收集梯度平均值缓冲区
      if (curr_state_.grad_avg().defined()) {
        grad_average_buffers.emplace_back(curr_state_.grad_avg());
      }
    }
  }
  // 将缓冲区写入文件
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  // 写入平方平均值缓冲区
  write_tensors_to_archive(
      output_archive, "square_average_buffers", square_average_buffers);
  // 写入动量缓冲区
  write_tensors_to_archive(
      output_archive, "momentum_buffers", momentum_buffers);
  // 写入梯度平均值缓冲区
  write_tensors_to_archive(
      output_archive, "grad_average_buffers", grad_average_buffers);
  // 将输出存档保存到临时文件中
  output_archive.save_to(optim_tempfile_old_format.name);
  // 使用 RMSprop 初始化另一个模型参数
  auto optim1_2 = RMSprop(model1_params, options);
  // 警告：使用旧的序列化逻辑检查
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);
  // 获取第二个优化器的参数组和状态
  const auto& params1_2_ = optim1_2.param_groups()[0].params();
  auto& optim1_2_state = optim1_2.state();
  // 旧的 RMSprop 没有跟踪步骤值
  // 将原始 RMSprop 的步骤值复制到当前 RMSprop 的状态中
  for (const auto i : c10::irange(params1_2_.size())) {
    if (i != (params1_2_.size() - 1)) {
      auto key_ = params_[i].unsafeGetTensorImpl();
      const RMSpropParamState& curr_state_ =
          static_cast<const RMSpropParamState&>(*(optim1_state.at(key_).get()));
      RMSpropParamState& curr_state1_2_ =
          static_cast<RMSpropParamState&>(*(optim1_2_state.at(key_).get()));
      curr_state1_2_.step(curr_state_.step());
    }
  }
  // 检查两个优化器状态是否相等
  is_optimizer_state_equal<RMSpropParamState>(optim1.state(), optim1_2.state());
}
TEST(SerializeTest, Optim_LBFGS) {
  // 调用测试函数，序列化 LBFGS 优化器并检查结果
  test_serialize_optimizer<LBFGS, LBFGSOptions, LBFGSParamState>(
      LBFGSOptions(), true);
  
  // 用于兼容性检查的自动化模型创建
  auto model1 = Linear(5, 2);
  auto model1_params = model1->parameters();
  
  // 添加一个张量以进行延迟初始化检查 - 当所有参数在缓冲区中没有条目时
  model1_params.emplace_back(torch::randn({2, 3}));
  
  // 创建 LBFGS 优化器实例，将模型参数和优化器选项传递给构造函数
  auto optim1 =
      torch::optim::LBFGS(model1_params, torch::optim::LBFGSOptions());

  // 创建一个输入张量 x，全为 1，形状为 (10, 5)
  auto x = torch::ones({10, 5});
  
  // 定义一个步骤函数，接受一个优化器和一个线性模型作为参数
  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    // 将模型梯度清零
    optimizer.zero_grad();
    // 计算模型前向传播结果的和 y
    auto y = model->forward(x).sum();
    // 计算 y 对模型参数的梯度
    y.backward();
    // 定义一个空闭包
    auto closure = []() { return torch::tensor({10}); };
    // 执行优化器的一步更新
    optimizer.step(closure);
  };

  // 调用步骤函数，传递优化器 optim1 和模型 model1
  step(optim1, model1);

  // 初始化用于存储的张量和队列
  at::Tensor d, t, H_diag, prev_flat_grad, prev_loss;
  std::deque<at::Tensor> old_dirs, old_stps;

  // 获取第一个参数组的优化器参数
  const auto& params_ = optim1.param_groups()[0].params();
  auto key_ = params_[0].unsafeGetTensorImpl();
  
  // 从优化器状态中获取 LBFGSParamState 对象的引用
  const auto& optim1_state =
      static_cast<const LBFGSParamState&>(*(optim1.state().at(key_).get()));

  // 从 optim1_state 中获取各种张量和数据
  d = optim1_state.d();
  t = at::tensor(optim1_state.t());
  H_diag = optim1_state.H_diag();
  prev_flat_grad = optim1_state.prev_flat_grad();
  prev_loss = at::tensor(optim1_state.prev_loss());
  old_dirs = optim1_state.old_dirs();

  // 将缓冲区写入文件
  auto optim_tempfile_old_format = c10::make_tempfile();
  torch::serialize::OutputArchive output_archive;
  output_archive.write("d", d, /*is_buffer=*/true);
  output_archive.write("t", t, /*is_buffer=*/true);
  output_archive.write("H_diag", H_diag, /*is_buffer=*/true);
  output_archive.write("prev_flat_grad", prev_flat_grad, /*is_buffer=*/true);
  output_archive.write("prev_loss", prev_loss, /*is_buffer=*/true);
  write_tensors_to_archive(output_archive, "old_dirs", old_dirs);
  write_tensors_to_archive(output_archive, "old_stps", old_stps);
  output_archive.save_to(optim_tempfile_old_format.name);

  // 使用旧的序列化逻辑加载优化器
  auto optim1_2 = LBFGS(model1_params, torch::optim::LBFGSOptions());
  OLD_SERIALIZATION_LOGIC_WARNING_CHECK(
      torch::load, optim1_2, optim_tempfile_old_format.name);

  // 获取 optim1_2 的第一个参数组的参数
  const auto& params1_2_ = optim1_2.param_groups()[0].params();
  auto param_key = params1_2_[0].unsafeGetTensorImpl();
  
  // 获取 optim1_2 的状态 LBFGSParamState 的引用
  auto& optim1_2_state =
      static_cast<LBFGSParamState&>(*(optim1_2.state().at(param_key).get()));

  // 复制 func_evals, n_iter, ro, al 等状态信息到 optim1_2_state 中
  optim1_2_state.func_evals(optim1_state.func_evals());
  optim1_2_state.n_iter(optim1_state.n_iter());
  optim1_2_state.ro(optim1_state.ro());
  optim1_2_state.al(optim1_state.al());

  // 检查两个优化器状态是否相等
  is_optimizer_state_equal<LBFGSParamState>(optim1.state(), optim1_2.state());
}
    // 创建一个张量 labels，用于存储标签数据，大小为 batch_size
    auto labels = torch::empty({batch_size});
    // 如果在 CUDA 上运行，将输入数据 inputs 和 labels 移动到 GPU 上
    if (is_cuda) {
      inputs = inputs.cuda();
      labels = labels.cuda();
    }
    // 遍历 batch_size 范围内的索引 i
    for (const auto i : c10::irange(batch_size)) {
      // 生成一个大小为 {2} 的随机整数张量，赋值给 inputs[i]
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      // 计算标签 labels[i]，使用 inputs[i] 的两个元素进行异或操作
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
    }
    // 使用模型 model 计算输入 inputs 的预测输出 x
    auto x = model->forward<torch::Tensor>(inputs);
    // 返回预测输出 x 和真实标签 labels 之间的二元交叉熵损失
    return torch::binary_cross_entropy(x, labels);
  };

  // 创建三个模型实例 model、model2、model3
  auto model = xor_model();
  auto model2 = xor_model();
  auto model3 = xor_model();
  // 使用 SGD 优化器，配置学习率为 0.1，动量为 0.9，启用 Nesterov 动量，设置权重衰减为 1e-6
  auto optimizer = torch::optim::SGD(
      model->parameters(),
      torch::optim::SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(
          1e-6));

  // 初始化运行损失 running_loss 为 1，迭代轮数 epoch 为 0
  float running_loss = 1;
  int epoch = 0;
  // 当运行损失 running_loss 大于 0.1 时执行循环
  while (running_loss > 0.1) {
    // 调用 getLoss 函数计算模型 model 在 batch_size 为 4 时的损失值
    torch::Tensor loss = getLoss(model, 4);
    // 梯度清零
    optimizer.zero_grad();
    // 反向传播计算梯度
    loss.backward();
    // 更新优化器参数
    optimizer.step();

    // 更新运行损失 running_loss，使用指数移动平均计算
    running_loss = running_loss * 0.99 + loss.sum().item<float>() * 0.01;
    // 断言检查 epoch 小于 3000，确保不会无限循环
    ASSERT_LT(epoch, 3000);
    // 增加迭代轮数 epoch
    epoch++;
  }

  // 创建临时文件 tempfile，保存模型 model 到该文件
  auto tempfile = c10::make_tempfile();
  torch::save(model, tempfile.name);
  // 从临时文件中加载模型数据到 model2
  torch::load(model2, tempfile.name);

  // 计算模型 model2 在 batch_size 为 100 时的损失值
  auto loss = getLoss(model2, 100);
  // 断言检查损失值小于 0.1，确保模型训练效果达到要求
  ASSERT_LT(loss.item<float>(), 0.1);

  // 将模型 model2 移动到 CUDA 设备上
  model2->to(torch::kCUDA);
  // 计算模型 model2 在 batch_size 为 100 时在 CUDA 上的损失值
  loss = getLoss(model2, 100, true);
  // 断言检查 CUDA 上的损失值小于 0.1
  ASSERT_LT(loss.item<float>(), 0.1);

  // 创建临时文件 tempfile2，保存模型 model2 到该文件
  auto tempfile2 = c10::make_tempfile();
  torch::save(model2, tempfile2.name);
  // 从临时文件中加载模型数据到 model3
  torch::load(model3, tempfile2.name);

  // 计算模型 model3 在 batch_size 为 100 时在 CUDA 上的损失值
  loss = getLoss(model3, 100, true);
  // 断言检查 CUDA 上的损失值小于 0.1
  ASSERT_LT(loss.item<float>(), 0.1);
TEST(
    SerializeTest,
    CanSerializeModulesWithIntermediateModulesWithoutParametersOrBuffers) {
  // 定义结构体 C，继承自 torch::nn::Module
  struct C : torch::nn::Module {
    // C 类的构造函数，注册名为 "foo" 的缓冲区，内容为 5 个值为 1 的整数张量
    C() {
      register_buffer("foo", torch::ones(5, torch::kInt32));
    }
  };

  // 定义结构体 B，继承自 torch::nn::Module
  struct B : torch::nn::Module {};

  // 定义结构体 A，继承自 torch::nn::Module
  struct A : torch::nn::Module {
    // A 类的构造函数，注册名为 "b" 的子模块，类型为 B 的共享指针
    // 注册名为 "c" 的子模块，类型为 C 的共享指针
    A() {
      register_module("b", std::make_shared<B>());
      register_module("c", std::make_shared<C>());
    }
  };

  // 定义结构体 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // M 类的构造函数，注册名为 "a" 的子模块，类型为 A 的共享指针
    M() {
      register_module("a", std::make_shared<A>());
    }
  };

  // 创建名为 out 的 M 类共享指针
  auto out = std::make_shared<M>();

  // 创建一个字符串流 ss
  std::stringstream ss;

  // 将 out 序列化并保存到 ss 中
  torch::save(out, ss);

  // 创建一个新的 M 类共享指针 in
  auto in = std::make_shared<M>();

  // 从 ss 中加载数据并反序列化到 in 中
  torch::load(in, ss);

  // 获取名为 "a.c.foo" 的缓冲区，并计算其所有元素之和作为整数输出
  const int output = in->named_buffers()["a.c.foo"].sum().item<int>();

  // 断言输出值为 5
  ASSERT_EQ(output, 5);
}

TEST(SerializeTest, VectorOfTensors) {
  // 设置随机种子为 0
  torch::manual_seed(0);

  // 创建包含两个随机张量的向量 x_vec
  std::vector<torch::Tensor> x_vec = {
      torch::randn({1, 2}), torch::randn({3, 4})};

  // 创建一个字符串流 stream
  std::stringstream stream;

  // 将 x_vec 序列化并保存到 stream 中
  torch::save(x_vec, stream);

  // 创建一个空的张量向量 y_vec
  std::vector<torch::Tensor> y_vec;

  // 从 stream 中加载数据并反序列化到 y_vec 中
  torch::load(y_vec, stream);

  // 对比 x_vec 和 y_vec 中的张量，确保大小和数值近似相等
  for (const auto i : c10::irange(x_vec.size())) {
    auto& x = x_vec[i];
    auto& y = y_vec[i];
    ASSERT_TRUE(y.defined());
    ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
    ASSERT_TRUE(x.allclose(y));
  }
}

TEST(SerializeTest, IValue) {
  // 创建一个包含整数值 1 的 IValue 对象 ivalue
  c10::IValue ivalue(1);

  // 创建一个临时文件 tempfile
  auto tempfile = c10::make_tempfile();

  // 创建一个输出归档 output_archive
  torch::serialize::OutputArchive output_archive;

  // 在 output_archive 中写入名为 "value" 的 ivalue
  output_archive.write("value", ivalue);

  // 将 output_archive 的内容保存到 tempfile 中
  output_archive.save_to(tempfile.name);

  // 创建一个输入归档 input_archive
  torch::serialize::InputArchive input_archive;

  // 从 tempfile 中加载数据到 input_archive 中
  input_archive.load_from(tempfile.name);

  // 创建一个空的 IValue 对象 ivalue_out
  c10::IValue ivalue_out;

  // 从 input_archive 中读取名为 "value" 的数据到 ivalue_out 中
  input_archive.read("value", ivalue_out);

  // 断言 ivalue_out 的整数值为 1
  ASSERT_EQ(ivalue_out.toInt(), 1);

  // 断言读取不存在的键 "bad_key" 时，会抛出异常并包含特定的错误信息
  ASSERT_THROWS_WITH(
      input_archive.read("bad_key", ivalue_out),
      "does not have a field with name");
}

// NOTE: if a `Module` contains unserializable submodules (e.g.
// `nn::Functional`), we expect those submodules to be skipped when the `Module`
// is being serialized.
TEST(SerializeTest, UnserializableSubmoduleIsSkippedWhenSavingModule) {
  // 定义结构体 A，继承自 torch::nn::Module
  struct A : torch::nn::Module {
    // A 类的构造函数，注册名为 "relu" 的子模块，类型为 nn::Functional 的 ReLU 函数
    A() {
      register_module("relu", torch::nn::Functional(torch::relu));
    }
  };

  // 创建一个 A 类的共享指针 out
  auto out = std::make_shared<A>();

  // 创建一个字符串流 ss
  std::stringstream ss;

  // 将 out 序列化并保存到 ss 中
  torch::save(out, ss);

  // 创建一个输入归档 archive
  torch::serialize::InputArchive archive;

  // 从 ss 中加载数据到 archive 中
  archive.load_from(ss);

  // 创建一个空的输入归档 relu_archive
  torch::serialize::InputArchive relu_archive;

  // 断言在 archive 中尝试读取名为 "relu" 的子模块会失败，因为 "relu" 是不可序列化的
  ASSERT_FALSE(archive.try_read("relu", relu_archive));
}

// NOTE: If a `Module` contains unserializable submodules (e.g.
// `nn::Functional`), we don't check the existence of those submodules in the
// `InputArchive` when deserializing.
TEST(SerializeTest, UnserializableSubmoduleIsIgnoredWhenLoadingModule) {
  // 定义结构体 B，继承自 torch::nn::Module
  struct B : torch::nn::Module {
    // B 类的构造函数，空实现
  };
  auto out = std::make_shared<A>();
  // 创建一个类型为 A 的智能指针对象 out，用于测试序列化和反序列化功能

  // 手动修改 "b.foo" 的值，以便在反序列化后检查缓冲区是否包含这些值。
  out->named_buffers()["b.foo"].fill_(1);
  // 使用 named_buffers() 方法获取 "b.foo" 缓冲区，并将其所有元素填充为 1

  auto tempfile = c10::make_tempfile();
  // 创建临时文件 tempfile，用于保存序列化后的数据

  torch::save(out, tempfile.name);
  // 将 out 对象序列化保存到临时文件 tempfile 中

  torch::serialize::InputArchive archive;
  // 创建一个输入存档对象 archive，用于加载序列化的数据
  archive.load_from(tempfile.name);
  // 从临时文件中加载数据到 archive

  torch::serialize::InputArchive archive_b;
  torch::serialize::InputArchive archive_relu;
  torch::Tensor tensor_foo;

  ASSERT_TRUE(archive.try_read("b", archive_b));
  // 尝试从 archive 中读取名为 "b" 的子模块数据，并保存到 archive_b 中
  ASSERT_TRUE(archive_b.try_read("foo", tensor_foo, /*is_buffer=*/true));
  // 尝试从 archive_b 中读取名为 "foo" 的缓冲区数据，并保存到 tensor_foo 中

  // 检查 `archive_b` 中不应存在名为 "relu1" 的子模块，
  // 因为 "relu1" 是一个 `nn::Functional` 类型，不支持序列化。
  ASSERT_FALSE(archive_b.try_read("relu1", archive_relu));

  // 检查 `archive` 中不应存在名为 "relu2" 的子模块，
  // 因为 "relu2" 是一个 `nn::Functional` 类型，不支持序列化。
  ASSERT_FALSE(archive.try_read("relu2", archive_relu));

  auto in = std::make_shared<A>();
  // 创建一个新的类型为 A 的智能指针对象 in，用于加载反序列化后的数据

  // `torch::load(...)` 在没有错误的情况下工作，
  // 即使 `A` 包含 `nn::Functional` 子模块，而序列化文件中不包含它们，
  // 因为 `nn::Functional` 子模块不支持序列化，因此在反序列化时被忽略。
  torch::load(in, tempfile.name);
  // 从临时文件中加载数据到 in 对象中

  // 检查 "b.foo" 缓冲区是否从文件正确反序列化。
  const int output = in->named_buffers()["b.foo"].sum().item<int>();
  // 计算 "b.foo" 缓冲区所有元素的总和，并转换为整数类型赋值给 output

  // `output` 应该等于我们在序列化之前手动分配给 "b.foo" 的值的总和。
  ASSERT_EQ(output, 5);
  // 断言 output 的值应该等于 5，即手动分配给 "b.foo" 的值的总和
}


注释：


# 这行代码关闭了一个代码块的开始，匹配了之前的一个左大括号 '{'。
```