# `.\pytorch\test\cpp\api\nn_utils.cpp`

```
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <c10/util/irange.h> // 引入 C10 库中的范围工具头文件
#include <torch/torch.h> // 引入 PyTorch 的核心头文件

#include <test/cpp/api/support.h> // 引入测试支持函数的头文件

#include <algorithm> // 引入标准库中的算法头文件
#include <random> // 引入随机数生成的头文件
#include <sstream> // 引入字符串流的头文件
#include <string> // 引入字符串处理的头文件

using namespace torch::nn; // 使用 PyTorch 的神经网络命名空间

namespace rnn_utils = torch::nn::utils::rnn; // 定义 rnn_utils 作为 PyTorch 神经网络工具包中 RNN 的命名空间别名

struct NNUtilsTest : torch::test::SeedingFixture {}; // 定义 NNUtilsTest 结构体继承自 SeedingFixture，用于神经网络工具测试
struct PackedSequenceTest : torch::test::SeedingFixture {}; // 定义 PackedSequenceTest 结构体继承自 SeedingFixture，用于打包序列测试

TEST_F(NNUtilsTest, ClipGradNorm) { // 定义测试夹具 NNUtilsTest 中的 ClipGradNorm 测试
  auto l = Linear(10, 10); // 创建一个输入维度为 10，输出维度为 10 的线性层
  float max_norm = 2; // 最大梯度范数设定为 2
  auto compute_norm = [&](float norm_type) -> float { // 定义计算梯度范数的 lambda 函数
    float total_norm = 0.0; // 初始化总梯度范数为 0.0
    if (norm_type != std::numeric_limits<float>::infinity()) { // 如果梯度范数类型不是无穷大
      for (const auto& p : l->parameters()) { // 遍历线性层的参数
        total_norm += // 累加每个参数梯度的绝对值的 norm_type 次方后的和
            p.grad().data().abs().pow(norm_type).sum().item().toFloat();
      }
      return std::pow(total_norm, 1.0 / norm_type); // 返回总梯度范数的 norm_type 次方根
    } else { // 如果梯度范数类型是无穷大
      for (const auto& p : l->parameters()) { // 遍历线性层的参数
        auto param_max = p.grad().data().abs().max().item().toFloat(); // 获取参数梯度的最大值
        if (param_max > total_norm) { // 如果最大值大于当前总梯度范数
          total_norm = param_max; // 更新总梯度范数为最大值
        }
      }
      return total_norm; // 返回总梯度范数
    }
  };
  auto compare_scaling = // 定义比较梯度缩放的 lambda 函数
      [&](const std::vector<torch::Tensor>& grads) -> torch::Tensor {
    std::vector<torch::Tensor> p_scale; // 创建存储参数缩放的向量
    for (const auto i : c10::irange(grads.size())) { // 遍历梯度向量的索引范围
      auto param = l->parameters()[i]; // 获取线性层的第 i 个参数
      auto grad = grads[i]; // 获取第 i 个梯度
      p_scale.push_back(param.grad().data().div(grad).view(-1)); // 将参数梯度除以梯度，并视图调整维度后放入 p_scale 中
    }
    auto scale = torch::cat(p_scale); // 拼接 p_scale 中的张量，得到缩放比例
    return scale; // 返回缩放比例张量
  };

  std::vector<torch::Tensor> grads = { // 定义包含张量的梯度向量
      torch::arange(1.0, 101).view({10, 10}), // 生成从 1 到 100 的张量，并视图调整为 10x10
      torch::ones({10}).div(1000), // 生成全为 1 的张量并除以 1000
  };
  std::vector<float> norm_types = { // 定义梯度范数类型向量
      0.5,
      1.5,
      2.0,
      4.0,
      std::numeric_limits<float>::infinity(),
  };
  for (auto norm_type : norm_types) { // 遍历梯度范数类型
    for (const auto i : c10::irange(grads.size())) { // 遍历梯度向量的索引范围
      l->parameters()[i].mutable_grad() = // 将参数 i 的梯度设为可变的，克隆并视图调整为与参数数据相同
          grads[i].clone().view_as(l->parameters()[i].data());
    }
    auto norm_before = compute_norm(norm_type); // 计算梯度范数之前的值
    auto norm = utils::clip_grad_norm_(l->parameters(), max_norm, norm_type); // 调用工具函数限制梯度范数
    auto norm_after = compute_norm(norm_type); // 计算梯度范数之后的值
    ASSERT_FLOAT_EQ(norm, norm_before); // 断言限制梯度范数前后的值相等
    ASSERT_NEAR(norm_after, max_norm, 1e-6); // 断言梯度范数后的值接近最大范数，精度为 1e-6
    ASSERT_LE(norm_after, max_norm); // 断言梯度范数后的值小于等于最大范数
    auto scaled = compare_scaling(grads); // 比较梯度缩放比例
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7); // 断言缩放比例的标准差接近 0，精度为 1e-7
  }
  // Small gradients should be left unchanged
  grads = { // 重新定义梯度向量为小梯度
      torch::rand({10, 10}).div(10000), // 生成均匀分布的随机张量并除以 10000
      torch::ones(10).div(500), // 生成全为 1 的张量并除以 500
  };
  for (auto norm_type : norm_types) { // 再次遍历梯度范数类型
    for (const auto i : c10::irange(grads.size())) { // 遍历梯度向量的索引范围
      l->parameters()[i].grad().data().copy_(grads[i]); // 将参数 i 的梯度数据拷贝为新的梯度
    }
    auto norm_before = compute_norm(norm_type); // 计算梯度范数之前的值
    auto norm = utils::clip_grad_norm_(l->parameters(), max_norm, norm_type); // 调用工具函数限制梯度范数
    auto norm_after = compute_norm(norm_type); // 计算梯度范数之后的值
    ASSERT_FLOAT_EQ(norm, norm_before); // 断言限制梯度范数前后的值相等
    ASSERT_FLOAT_EQ(norm_before, norm_after); // 断言梯度范数前后的值相等
    ASSERT_LE(norm_after, max_norm); // 断言梯度范数后的值小于等于最大范数
    auto scaled = compare_scaling(grads); // 比较梯度缩放比例
    // 断言检查 scaled 的标准差是否接近于 0，精度为 1e-7
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7);
    // 断言检查 scaled 的第一个元素是否浮点值等于 1
    ASSERT_FLOAT_EQ(scaled[0].item().toFloat(), 1);
  }
  // 应当接受单个张量作为输入
  auto p1 = torch::randn({10, 10});  // 创建一个大小为 10x10 的随机张量 p1
  auto p2 = torch::randn({10, 10});  // 创建一个大小为 10x10 的随机张量 p2
  auto g = torch::arange(1., 101).view({10, 10});  // 创建一个从 1 到 100 的序列并reshape成 10x10 张量 g
  p1.mutable_grad() = g.clone();  // 将 g 的克隆分配给 p1 的梯度
  p2.mutable_grad() = g.clone();  // 将 g 的克隆分配给 p2 的梯度
  // 对于每种规范类型 norm_type 中的每一个
  for (const auto norm_type : norm_types) {
    // 使用 utils::clip_grad_norm_ 对 p1 应用梯度裁剪，最大范数为 max_norm
    utils::clip_grad_norm_(p1, max_norm, norm_type);
    // 使用 utils::clip_grad_norm_ 对 {p2} 应用梯度裁剪，最大范数为 max_norm
    utils::clip_grad_norm_({p2}, max_norm, norm_type);
    // 断言检查 p1 和 p2 的梯度是否全部接近
    ASSERT_TRUE(torch::allclose(p1.grad(), p2.grad()));
  }
}

// 结束上一个 Lambda 表达式的定义

// 检查 clip_grad_norm_ 函数是否在梯度范数为非有限值时引发错误
auto param = torch::ones(
    10,
    torch::TensorOptions()
        .dtype(torch::kDouble)
        .device(device_type)
        .requires_grad(true));
// 如果 grad_only_one_elem 为 true，则只对第二个元素进行标量乘法和求和后反向传播
if (grad_only_one_elem) {
  param[1].mul(scalar).sum().backward();
} else {
  // 否则对所有元素进行标量乘法和求和后反向传播
  param.mul(scalar).sum().backward();
}

std::vector<torch::Tensor> parameters;
// 如果 prefix_finite_grad_param 为 true，则创建一个具有单个元素的张量，并对其进行标量乘法和求和后反向传播
if (prefix_finite_grad_param) {
  auto prefix_param = torch::ones(
      1,
      torch::TensorOptions()
          .dtype(torch::kDouble)
          .device(device_type)
          .requires_grad(true));
  prefix_param.mul(1).sum().backward();
  parameters.push_back(prefix_param);
}
// 将之前定义的 param 张量添加到 parameters 向量中
parameters.push_back(param);

// 返回 parameters 向量
return parameters;
};

// 定义一个 Lambda 表达式 run_test_case，用于测试 clip_grad_norm_ 函数的行为
auto run_test_case = [&gen_parameters](
                         double norm_type,
                         bool error_if_nonfinite,
                         double scalar,
                         bool grad_only_one_elem,
                         bool prefix_finite_grad_param,
                         bool is_norm_nonfinite,
                         torch::DeviceType device_type) {
  // 创建一个 stringstream，用于生成测试用例的描述信息
  std::stringstream ss;
  ss << "device: " << device_type << ", norm_type: " << norm_type
     << ", error_if_nonfinite: " << error_if_nonfinite
     << ", scalar: " << scalar
     << ", grad_only_one_elem: " << grad_only_one_elem
     << ", prefix_finite_grad_param: " << prefix_finite_grad_param
     << ", is_norm_nonfinite: " << is_norm_nonfinite;
  std::string msg = ss.str();

  // 生成参数集合，调用 gen_parameters Lambda 表达式
  auto parameters = gen_parameters(
      scalar, grad_only_one_elem, prefix_finite_grad_param, device_type);

  // 如果 is_norm_nonfinite 为 true 且 error_if_nonfinite 为 true，则执行以下代码块
  if (is_norm_nonfinite && error_if_nonfinite) {
    std::vector<torch::Tensor> grads_before;
    // 遍历 parameters 向量，复制每个张量的梯度并存储在 grads_before 向量中
    // NOLINTNEXTLINE(performance-for-range-copy)
    for (auto p : parameters) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      grads_before.push_back(p.grad().clone());
    }
    // 使用 EXPECT_THROW 宏来验证 utils::clip_grad_norm_ 函数在调用时抛出异常
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    EXPECT_THROW(
        utils::clip_grad_norm_(parameters, 1., norm_type, true),
        std::exception)
        << msg;
    // 如果异常抛出成功，验证梯度未发生变化
    // 检查每个参数的梯度是否与之前保存的 grads_before 中的值相等
    for (const auto p_idx : c10::irange(parameters.size())) {
      ASSERT_TRUE(torch::allclose(
          parameters[p_idx].grad(),
          grads_before[p_idx],
          1.0,
          0.0,
          /*equal_nan*/ true))
          << msg;
    }
  } else {
    // 如果 is_norm_nonfinite 为 false 或 error_if_nonfinite 为 false，则执行以下代码块
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    // 使用 EXPECT_NO_THROW 宏来验证 utils::clip_grad_norm_ 函数在调用时不抛出异常
    EXPECT_NO_THROW(
        utils::clip_grad_norm_(parameters, 1., norm_type, error_if_nonfinite))
        << msg;
  }
};

// 遍历设备类型为 CPU 和 CUDA 的所有测试用例
for (auto device_type : {torch::kCPU, torch::kCUDA}) {
  // 如果当前设备类型为 CUDA 但 CUDA 不可用，则跳过此次循环
  if (device_type == torch::kCUDA && !torch::cuda::is_available()) {
    continue;
  }
    // 遍历所有测试用例
    for (auto test_case : test_cases) {
      // 获取当前测试用例的各个参数
      auto grad_only_one_elem = std::get<0>(test_case);  // 提取梯度仅包含一个元素的情况
      auto prefix_finite_grad_param = std::get<1>(test_case);  // 提取前缀有限梯度参数
      auto scalars = std::get<2>(test_case);  // 提取标量列表
      auto norms_nonfinite = std::get<3>(test_case);  // 提取非有限范数类型列表
      auto norms_finite = std::get<4>(test_case);  // 提取有限范数类型列表
    
      // 遍历是否在非有限时报错的选项（true/false）
      for (auto error_if_nonfinite : {false, true}) {
        // 遍历标量列表
        for (auto scalar : scalars) {
          // 遍历非有限范数类型列表
          for (auto norm_type : norms_nonfinite) {
            // 执行测试用例，测试非有限范数情况
            run_test_case(
                norm_type,
                error_if_nonfinite,
                scalar,
                grad_only_one_elem,
                prefix_finite_grad_param,
                true,  // 指示为非有限范数测试
                device_type);  // 设备类型
          }
    
          // 遍历有限范数类型列表
          for (auto norm_type : norms_finite) {
            // 执行测试用例，测试有限范数情况
            run_test_case(
                norm_type,
                error_if_nonfinite,
                scalar,
                grad_only_one_elem,
                prefix_finite_grad_param,
                false,  // 指示为有限范数测试
                device_type);  // 设备类型
          }
        }
      }
    }
}

// 定义一个测试用例 NNUtilsTest 中的 ClipGradValue 函数
TEST_F(NNUtilsTest, ClipGradValue) {
  // 创建一个线性层，输入和输出都是大小为 10 的向量
  auto l = Linear(10, 10);
  // 设定梯度裁剪的阈值
  float clip_value = 2.5;

  // 创建一个张量 grad_w，包含从 -50 到 49 的连续数，并将其 reshape 成 10x10 的矩阵，然后除以 5
  torch::Tensor grad_w = torch::arange(-50., 50).view({10, 10}).div_(5);
  // 创建一个大小为 10 的全一张量 grad_b
  torch::Tensor grad_b = torch::ones({10}).mul_(2);
  // 创建一个张量列表 grad_lists，包含两个子列表，每个子列表包含两个张量
  std::vector<std::vector<torch::Tensor>> grad_lists = {
      {grad_w, grad_b}, {grad_w, torch::Tensor()}};

  // 遍历 grad_lists 中的每个 grad_list
  for (auto grad_list : grad_lists) {
    // 遍历 grad_list 中的每个张量，使用索引 i
    for (const auto i : c10::irange(grad_list.size())) {
      // 获取线性层 l 的第 i 个参数
      auto p = l->parameters()[i];
      // 获取 grad_list 中的第 i 个张量
      auto g = grad_list[i];
      // 如果 g 已定义，则将其克隆并视图化成和 p.data() 相同形状的张量，否则直接赋值给 p 的梯度
      p.mutable_grad() = g.defined() ? g.clone().view_as(p.data()) : g;
    }

    // 对线性层 l 的参数进行梯度裁剪
    utils::clip_grad_value_(l->parameters(), clip_value);

    // 遍历线性层 l 的每个参数 p
    for (const auto& p : l->parameters()) {
      // 如果 p 的梯度已定义
      if (p.grad().defined()) {
        // 断言 p 的梯度的最大值小于等于 clip_value
        ASSERT_LE(p.grad().data().max().item().toFloat(), clip_value);
        // 断言 p 的梯度的最小值大于等于 -clip_value
        ASSERT_GE(p.grad().data().min().item().toFloat(), -clip_value);
      }
    }
  }

  // 应接受单个张量作为输入
  // 创建两个随机张量 p1 和 p2
  auto p1 = torch::randn({10, 10});
  auto p2 = torch::randn({10, 10});
  // 创建一个张量 g，包含从 -50 到 49 的连续数，并将其 reshape 成 10x10 的矩阵，然后除以 5
  auto g = torch::arange(-50., 50).view({10, 10}).div_(5);
  // 设置 p1 和 p2 的梯度为 g 的克隆
  p1.mutable_grad() = g.clone();
  p2.mutable_grad() = g.clone();
  // 对 p1 和 p2 的梯度进行裁剪
  utils::clip_grad_value_(p1, clip_value);
  utils::clip_grad_value_({p2}, clip_value);
  // 断言 p1 和 p2 的梯度是否近似相等
  ASSERT_TRUE(torch::allclose(p1.grad(), p2.grad()));
}

// 定义一个测试用例 NNUtilsTest 中的 ConvertParameters 函数
TEST_F(NNUtilsTest, ConvertParameters) {
  // 创建一个包含三个张量的参数向量 parameters
  std::vector<torch::Tensor> parameters{
      torch::arange(9, torch::kFloat32),
      torch::arange(9, torch::kFloat32).view({3, 3}),
      torch::arange(8, torch::kFloat32).view({2, 2, 2})};

  // 创建一个期望的张量 expected，将 parameters 中的所有张量连接起来
  auto expected = torch::cat(
      {torch::arange(9, torch::kFloat32),
       torch::arange(9, torch::kFloat32).view(-1),
       torch::arange(8, torch::kFloat32).view(-1)});
  // 将 parameters 转换成一个向量 vector
  auto vector = utils::parameters_to_vector(parameters);
  // 断言 vector 是否近似等于 expected
  ASSERT_TRUE(vector.allclose(expected));

  // 创建一个全零参数向量 zero_parameters
  std::vector<torch::Tensor> zero_parameters{
      torch::zeros({9}, torch::kFloat32),
      torch::zeros({9}, torch::kFloat32).view({3, 3}),
      torch::zeros({8}, torch::kFloat32).view({2, 2, 2})};

  // 将向量 vector 转换回 zero_parameters
  utils::vector_to_parameters(vector, zero_parameters);
  // 遍历 zero_parameters 中的每个张量，断言其是否近似等于 parameters 中对应的张量
  for (const auto i : c10::irange(zero_parameters.size())) {
    ASSERT_TRUE(zero_parameters[i].allclose(parameters[i]));
  }

  {
    // 创建一个卷积层 conv1 和一个全连接层 fc1，然后将它们放入序列模型 model
    auto conv1 = Conv2d(3, 10, 5);
    auto fc1 = Linear(10, 20);
    auto model = Sequential(conv1, fc1);

    // 将模型 model 的所有参数转换成一个向量 vec
    auto vec = utils::parameters_to_vector(model->parameters());
    // 断言 vec 的大小是否为 980
    ASSERT_EQ(vec.size(0), 980);
  }
  {
    // 创建一个卷积层 conv1 和一个全连接层 fc1，然后将它们放入序列模型 model
    auto conv1 = Conv2d(3, 10, 5);
    auto fc1 = Linear(10, 20);
    auto model = Sequential(conv1, fc1);

    // 创建一个从 0 到 979 的连续数张量 vec
    auto vec = torch::arange(0., 980);
    // 将 vec 转换回模型 model 的参数
    utils::vector_to_parameters(vec, model->parameters());

    // 获取模型 model 的第一个参数的第一个元素，并将其和 vec 的前五个元素比较
    auto sample = model->parameters()[0][0][0][0];
    ASSERT_TRUE(torch::equal(sample.data(), vec.data().slice(0, 0, 5)));
  }
}

// 定义一个整数变量 PackedSequenceTest_batch_size，并赋值为 5
// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-non-const-global-variables)
int64_t PackedSequenceTest_batch_size = 5;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-non-const-global-variables)
int64_t PackedSequenceTest_max_length = 6;


// 定义最大序列长度为6的整型变量
int64_t PackedSequenceTest_max_length = 6;



std::vector<torch::Tensor> PackedSequenceTest_ordered_sequence(
    torch::ScalarType tensor_type) {
  std::vector<torch::Tensor> seqs;
  seqs.reserve(PackedSequenceTest_batch_size);
  for (const auto i : c10::irange(PackedSequenceTest_batch_size)) {
    (void)i; // Suppress unused variable warning
    seqs.emplace_back(torch::empty(
        {torch::randint(1, PackedSequenceTest_max_length, {1}).item<int64_t>()},
        tensor_type));
  }
  for (auto& s : seqs) {
    s.random_(-128, 128);
  }
  sort(
      seqs.begin(),
      seqs.end(),
      [&](const torch::Tensor& t1, const torch::Tensor& t2) {
        return t1.size(0) > t2.size(0);
      });
  return seqs;
}


// 返回一个按顺序排列的序列的向量，其中每个序列是由随机长度的张量组成的
std::vector<torch::Tensor> PackedSequenceTest_ordered_sequence(
    torch::ScalarType tensor_type) {
  std::vector<torch::Tensor> seqs;
  seqs.reserve(PackedSequenceTest_batch_size);
  for (const auto i : c10::irange(PackedSequenceTest_batch_size)) {
    (void)i; // 抑制未使用变量警告
    // 创建一个随机长度的张量，并将其添加到序列中
    seqs.emplace_back(torch::empty(
        {torch::randint(1, PackedSequenceTest_max_length, {1}).item<int64_t>()},
        tensor_type));
  }
  // 为每个张量设置随机值
  for (auto& s : seqs) {
    s.random_(-128, 128);
  }
  // 按照张量大小降序排序序列
  sort(
      seqs.begin(),
      seqs.end(),
      [&](const torch::Tensor& t1, const torch::Tensor& t2) {
        return t1.size(0) > t2.size(0);
      });
  return seqs;
}



std::tuple<torch::Tensor, torch::Tensor> PackedSequenceTest_padded_sequence(
    torch::ScalarType tensor_type) {
  // Create Tensor of random padded sequences
  auto ordered = PackedSequenceTest_ordered_sequence(tensor_type);
  auto lengths = torch::empty({(int64_t)ordered.size()}, torch::kInt64);
  for (const auto i : c10::irange(ordered.size())) {
    lengths[i] = ordered[i].size(0);
  }
  auto padded_tensor = rnn_utils::pad_sequence(ordered);
  return std::make_tuple(padded_tensor, lengths);
}


// 创建随机填充序列的张量，并返回填充后的张量及其长度
std::tuple<torch::Tensor, torch::Tensor> PackedSequenceTest_padded_sequence(
    torch::ScalarType tensor_type) {
  // 获取有序的随机序列
  auto ordered = PackedSequenceTest_ordered_sequence(tensor_type);
  // 创建一个张量以保存序列的长度
  auto lengths = torch::empty({(int64_t)ordered.size()}, torch::kInt64);
  // 将每个序列的长度存入长度张量中
  for (const auto i : c10::irange(ordered.size())) {
    lengths[i] = ordered[i].size(0);
  }
  // 对有序序列进行填充，生成填充后的张量
  auto padded_tensor = rnn_utils::pad_sequence(ordered);
  return std::make_tuple(padded_tensor, lengths);
}



void assert_is_equal_packed_sequence(
    const rnn_utils::PackedSequence& a,
    const rnn_utils::PackedSequence& b) {
  ASSERT_TRUE(torch::allclose(a.data(), b.data()));
  ASSERT_TRUE(torch::allclose(a.batch_sizes(), b.batch_sizes()));
  ASSERT_TRUE(
      (!a.sorted_indices().defined() && !b.sorted_indices().defined()) ||
      torch::allclose(a.sorted_indices(), b.sorted_indices()));
  ASSERT_TRUE(
      (!a.unsorted_indices().defined() && !b.unsorted_indices().defined()) ||
      torch::allclose(a.unsorted_indices(), b.unsorted_indices()));
}


// 断言两个打包序列相等
void assert_is_equal_packed_sequence(
    const rnn_utils::PackedSequence& a,
    const rnn_utils::PackedSequence& b) {
  // 断言数据张量相等
  ASSERT_TRUE(torch::allclose(a.data(), b.data()));
  // 断言批量大小张量相等
  ASSERT_TRUE(torch::allclose(a.batch_sizes(), b.batch_sizes()));
  // 断言排序索引相等（如果定义了的话）
  ASSERT_TRUE(
      (!a.sorted_indices().defined() && !b.sorted_indices().defined()) ||
      torch::allclose(a.sorted_indices(), b.sorted_indices()));
  // 断言未排序索引相等（如果定义了的话）
  ASSERT_TRUE(
      (!a.unsorted_indices().defined() && !b.unsorted_indices().defined()) ||
      torch::allclose(a.unsorted_indices(), b.unsorted_indices()));
}



void assert_is_same_packed_sequence(
    const rnn_utils::PackedSequence& a,
    const rnn_utils::PackedSequence& b) {
  ASSERT_TRUE(a.data().is_same(b.data()));
  ASSERT_TRUE(a.batch_sizes().is_same(b.batch_sizes()));
  ASSERT_TRUE(a.sorted_indices().is_same(b.sorted_indices()));
  ASSERT_TRUE(a.unsorted_indices().is_same(b.unsorted_indices()));
}


// 断言两个打包序列是相同的对象
void assert_is_same_packed_sequence(
    const rnn_utils::PackedSequence& a,
    const rnn_utils::PackedSequence& b) {
  // 断言数据张量是同一对象
  ASSERT_TRUE(a.data().is_same(b.data()));
  // 断言批量大小张量是同一对象
  ASSERT_TRUE(a.batch_sizes().is_same(b.batch_sizes()));
  // 断言排序索引是同一对象
  ASSERT_TRUE(a.sorted_indices().is_same(b.sorted_indices()));
  // 断言未排序索引是同一对象
  ASSERT_TRUE(a.unsorted_indices().is_same(b.unsorted_indices()));
}



TEST_F(PackedSequenceTest, WrongOrder) {
  auto a = torch::ones({25, 300});
  auto b = torch::ones({22, 300});
  auto b_a = rnn_utils::pad_sequence({b, a});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      rnn_utils::pack_padded_sequence(
          b_a,
          torch::tensor({22, 25}),
          /*batch_first=*/false,
          /*enforce_sorted=*/true),
      c10::Error);
}


// 测试在错误顺序下的填充序列行为
TEST_F(PackedSequenceTest, WrongOrder) {
  // 创建两个张量 a 和 b，分别为大小为 {25, 300} 和 {22, 300} 的全1张量
  auto a = torch::ones({25, 300});
  auto b = torch::ones({22, 300});
  // 对 b 和 a 进行填充，并将结果存储在 b_a 中
  auto b_a = rnn_utils::pad_sequence({b, a});
  // 断言在非正确排序的情况下，pack_padded_sequence 抛出异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      rnn_utils::pack_padded_sequence(
          b_a,
          torch::tensor({22, 25}),
          /*batch_first=*/false,
          /*enforce_sorted=*/true),
      c10::Error);
}



TEST_F(PackedSequenceTest, TotalLength) {
  torch::Tensor padded, lengths;
  std::tie(padded, lengths) = PackedSequenceTest_padded_sequence(torch::kFloat);
  int64_t max_length = torch::max(lengths).item<int64_t>();
  rnn_utils::PackedSequence packed =
      rnn_utils::pack_padded_sequence(padded, lengths);

  // test ValueError if total_length < max_length
  for (int64_t total_length : std::vector<int64_t>{-1, 0, max_length - 1}) {


// 测试总长度
TEST_F(PackedSequenceTest, TotalLength) {
  // 创建填充后的张量和长度张量
  torch::Tensor padded, lengths;
  std::tie(padded, lengths) = PackedSequenceTest_padded_sequence(torch::kFloat);
  // 获取长度
    // 遍历布尔向量，分别测试 batch_first 为 true 和 false 两种情况
    for (bool batch_first : std::vector<bool>{true, false}) {
      // 定义错误处理函数 err_fn，捕获可能抛出的异常
      auto err_fn = [&]() {
        // 调用 rnn_utils::pad_packed_sequence 进行填充解压缩操作
        rnn_utils::pad_packed_sequence(
            packed,
            /*batch_first=*/batch_first,
            /*padding_value=*/0.0,
            /*total_length=*/total_length);
      };
      // 使用 ASSERT_THROWS_WITH 确保调用 err_fn 会抛出预期异常信息
      ASSERT_THROWS_WITH(
          err_fn(),
          "Expected total_length to be at least the length of the longest sequence in input");
    }
  }

  // 测试 pad_packed_sequence 返回的结果长度是否正确
  for (bool batch_first : std::vector<bool>{true, false}) {
    // 调用 pad_packed_sequence，获取解压缩后的结果和忽略的变量
    torch::Tensor no_extra_pad, ignored;
    std::tie(no_extra_pad, ignored) =
        rnn_utils::pad_packed_sequence(packed, /*batch_first=*/batch_first);
    // 遍历不同的 total_length 偏移量，进行测试
    for (int64_t total_length_delta : std::vector<int64_t>{0, 1, 8}) {
      // 计算当前的 total_length
      int64_t total_length = max_length + total_length_delta;
      // 定义解压缩后的输出和长度输出的变量
      torch::Tensor unpacked, lengths_out;
      std::tie(unpacked, lengths_out) = rnn_utils::pad_packed_sequence(
          packed,
          /*batch_first=*/batch_first,
          /*padding_value=*/0.0,
          /*total_length=*/total_length);
      // 断言 lengths 和 lengths_out 是相等的
      ASSERT_TRUE(torch::allclose(lengths, lengths_out));
      // 断言解压缩后的输出维度是否正确
      ASSERT_EQ(unpacked.size(batch_first ? 1 : 0), total_length);
      // 定义参考输出和额外填充的变量
      torch::Tensor ref_output, extra_pad;
      // 根据 total_length_delta 的值选择合适的 ref_output
      if (total_length_delta == 0) {
        ref_output = no_extra_pad;
      } else if (batch_first) {
        // 在 batch_first 模式下生成额外填充的数据
        extra_pad = torch::zeros(
            {PackedSequenceTest_batch_size, total_length_delta},
            no_extra_pad.options());
        ref_output = torch::cat({no_extra_pad, extra_pad}, 1);
      } else {
        // 在非 batch_first 模式下生成额外填充的数据
        extra_pad = torch::zeros(
            {total_length_delta, PackedSequenceTest_batch_size},
            no_extra_pad.options());
        ref_output = torch::cat({no_extra_pad, extra_pad}, 0);
      }
      // 断言解压缩后的输出和参考输出是否相等
      ASSERT_TRUE(torch::allclose(unpacked, ref_output));
    }
  }
}

TEST_F(PackedSequenceTest, To) {
  // 对于每个 enforce_sorted 取值为 true 和 false 的布尔值进行循环测试
  for (bool enforce_sorted : std::vector<bool>{true, false}) {
    // 定义 torch::Tensor 类型的变量 padded 和 lengths
    torch::Tensor padded, lengths;
    // 调用 PackedSequenceTest_padded_sequence 函数，返回结果分别赋给 padded 和 lengths
    std::tie(padded, lengths) = PackedSequenceTest_padded_sequence(torch::kInt);
    // 调用 rnn_utils::pack_padded_sequence 函数，将 padded 和 lengths 打包成 PackedSequence 对象 a
    rnn_utils::PackedSequence a = rnn_utils::pack_padded_sequence(
                                      padded,
                                      lengths,
                                      /*batch_first=*/false,
                                      /*enforce_sorted=*/enforce_sorted)
                                      .cpu();

    // 断言 a 和 a.to(torch::kCPU) 返回的 PackedSequence 对象相同
    assert_is_same_packed_sequence(a, a.to(torch::kCPU));
    // 断言 a 和 a.cpu() 返回的 PackedSequence 对象相同
    assert_is_same_packed_sequence(a, a.cpu());
    // 断言 a 和 a.to(torch::device(torch::kCPU).dtype(torch::kInt32)) 返回的 PackedSequence 对象相同
    assert_is_same_packed_sequence(
        a, a.to(torch::device(torch::kCPU).dtype(torch::kInt32)));

    // 如果 CUDA 可用
    if (torch::cuda::is_available()) {
      // 将 a 转移到 CUDA 设备，生成 b
      auto b = a.cuda();
      // 断言 b 和 b.to(torch::kCUDA) 返回的 PackedSequence 对象相同
      assert_is_same_packed_sequence(b, b.to(torch::kCUDA));
      // 断言 b 和 b.cuda() 返回的 PackedSequence 对象相同
      assert_is_same_packed_sequence(b, b.cuda());
      // 断言 a 和 b.to(torch::kCPU) 返回的 PackedSequence 对象相同
      assert_is_equal_packed_sequence(a, b.to(torch::kCPU));
      // 断言 b 和 a.to(torch::kCUDA) 返回的 PackedSequence 对象相同
      assert_is_equal_packed_sequence(b, a.to(torch::kCUDA));
      // 断言 a 和 b.to(torch::device(torch::kCPU).dtype(torch::kInt32)) 返回的 PackedSequence 对象相同
      assert_is_equal_packed_sequence(
          a, b.to(torch::device(torch::kCPU).dtype(torch::kInt32)));
      // 断言 b 和 b.to(torch::kInt32) 返回的 PackedSequence 对象相同
      assert_is_same_packed_sequence(b, b.to(torch::kInt32));
    }
  }
}

TEST_F(NNUtilsTest, PackSequence) {
  // 定义 _compatibility_test 匿名函数，接受 sequences、lengths、batch_first 和 enforce_sorted 四个参数
  auto _compatibility_test = [&](torch::ArrayRef<torch::Tensor> sequences,
                                 torch::Tensor lengths,
                                 bool batch_first,
                                 bool enforce_sorted = false) {
    // 调用 rnn_utils::pad_sequence 函数，将 sequences 按照 batch_first 填充成 padded
    torch::Tensor padded = rnn_utils::pad_sequence(sequences, batch_first);
    // 调用 rnn_utils::pack_sequence 函数，将 sequences 打包成 PackedSequence 对象 packed
    rnn_utils::PackedSequence packed =
        rnn_utils::pack_sequence(sequences, enforce_sorted);
    // 调用 rnn_utils::pad_packed_sequence 函数，解包 packed，返回解包后的 Tensor 和 lengths
    std::tuple<torch::Tensor, torch::Tensor> unpacked =
        rnn_utils::pad_packed_sequence(packed, batch_first);
    // 使用 allclose 函数断言 padded 和 unpacked 中的第一个 Tensor 相等
    ASSERT_TRUE(torch::allclose(padded, std::get<0>(unpacked)));
    // 调用 rnn_utils::pack_padded_sequence 函数，将 padded 和 lengths 打包成 PackedSequence 对象 pack_padded
    rnn_utils::PackedSequence pack_padded = rnn_utils::pack_padded_sequence(
        padded, lengths, batch_first, enforce_sorted);
  // 对给定的 packed 和 pack_padded 序列进行相等性断言
  assert_is_equal_packed_sequence(packed, pack_padded);
};

// single dimensional
auto a = torch::tensor({1, 2, 3});  // 创建包含 {1, 2, 3} 的 Torch 张量 a
auto b = torch::tensor({4, 5});  // 创建包含 {4, 5} 的 Torch 张量 b
auto c = torch::tensor({6});  // 创建包含 {6} 的 Torch 张量 c
rnn_utils::PackedSequence packed =
    rnn_utils::pack_sequence({a, b, c}, /*enforce_sorted=*/false);  // 调用 rnn_utils 库的 pack_sequence 函数，将张量 a, b, c 打包成 packed 序列，不要求排序
auto expected = torch::tensor({1, 4, 6, 2, 5, 3});  // 创建包含期望结果 {1, 4, 6, 2, 5, 3} 的 Torch 张量 expected
ASSERT_TRUE(torch::allclose(packed.batch_sizes(), torch::tensor({3, 2, 1})));  // 断言 packed 的 batch_sizes 是否与 {3, 2, 1} 的 Torch 张量相等
ASSERT_TRUE(torch::allclose(packed.data(), expected));  // 断言 packed 的数据是否与 expected 张量相等
ASSERT_TRUE(
    torch::allclose(packed.sorted_indices(), torch::tensor({0, 1, 2})));  // 断言 packed 的 sorted_indices 是否与 {0, 1, 2} 的 Torch 张量相等
ASSERT_TRUE(
    torch::allclose(packed.unsorted_indices(), torch::tensor({0, 1, 2})));  // 断言 packed 的 unsorted_indices 是否与 {0, 1, 2} 的 Torch 张量相等

rnn_utils::PackedSequence packed_unsorted =
    rnn_utils::pack_sequence({b, c, a}, /*enforce_sorted=*/false);  // 调用 rnn_utils 库的 pack_sequence 函数，将张量 b, c, a 打包成 packed_unsorted 序列，不要求排序
ASSERT_TRUE(
    torch::allclose(packed_unsorted.batch_sizes(), torch::tensor({3, 2, 1})));  // 断言 packed_unsorted 的 batch_sizes 是否与 {3, 2, 1} 的 Torch 张量相等
ASSERT_TRUE(torch::allclose(packed_unsorted.data(), expected));  // 断言 packed_unsorted 的数据是否与 expected 张量相等
ASSERT_TRUE(torch::allclose(
    packed_unsorted.sorted_indices(), torch::tensor({2, 0, 1})));  // 断言 packed_unsorted 的 sorted_indices 是否与 {2, 0, 1} 的 Torch 张量相等
ASSERT_TRUE(torch::allclose(
    packed_unsorted.unsorted_indices(), torch::tensor({1, 2, 0})));  // 断言 packed_unsorted 的 unsorted_indices 是否与 {1, 2, 0} 的 Torch 张量相等

// single dimensional, enforce_sorted = True
rnn_utils::PackedSequence packed_enforce_sorted =
    rnn_utils::pack_sequence({a, b, c}, /*enforce_sorted=*/true);  // 调用 rnn_utils 库的 pack_sequence 函数，将张量 a, b, c 打包成 packed_enforce_sorted 序列，并要求排序
ASSERT_TRUE(torch::allclose(
    packed_enforce_sorted.batch_sizes(), torch::tensor({3, 2, 1})));  // 断言 packed_enforce_sorted 的 batch_sizes 是否与 {3, 2, 1} 的 Torch 张量相等
ASSERT_TRUE(torch::allclose(packed_enforce_sorted.data(), expected));  // 断言 packed_enforce_sorted 的数据是否与 expected 张量相等
ASSERT_FALSE(packed_enforce_sorted.sorted_indices().defined());  // 断言 packed_enforce_sorted 的 sorted_indices 是否未定义
ASSERT_FALSE(packed_enforce_sorted.unsorted_indices().defined());  // 断言 packed_enforce_sorted 的 unsorted_indices 是否未定义

ASSERT_THROWS_WITH(
    rnn_utils::pack_sequence({b, c, a}, /*enforce_sorted=*/true),
    "must be sorted in decreasing order");  // 断言调用 pack_sequence 函数时，当 enforce_sorted 为 true 时抛出异常，错误消息为 "must be sorted in decreasing order"

ASSERT_THROWS_WITH(
    rnn_utils::pack_sequence({b, c, a}, /*enforce_sorted=*/true),
    "You can pass `enforce_sorted=False`");  // 断言调用 pack_sequence 函数时，当 enforce_sorted 为 true 时抛出异常，错误消息包含 "You can pass `enforce_sorted=False`"

// more dimensions
int64_t maxlen = 9;  // 定义最大长度 maxlen 为 9
for (int64_t num_dim : std::vector<int64_t>{0, 1, 2, 3}) {  // 遍历维度数为 {0, 1, 2, 3} 的循环
  std::vector<torch::Tensor> sequences;  // 创建空的 Torch 张量向量 sequences
  std::vector<int64_t> lengths_vec;  // 创建空的整数向量 lengths_vec
  std::vector<int64_t> trailing_dims(num_dim, 4);  // 创建长度为 num_dim，每个元素为 4 的整数向量 trailing_dims
  for (int64_t i = maxlen; i > 0; i--) {  // 从 maxlen 到 1 的循环
    int64_t seq_len = i * i;  // 计算序列长度为 i*i
    lengths_vec.emplace_back(seq_len);  // 将 seq_len 添加到 lengths_vec 中
    std::vector<int64_t> tensor_sizes{seq_len, 5};  // 创建包含 {seq_len, 5} 的整数向量 tensor_sizes
    tensor_sizes.insert(
        tensor_sizes.end(), trailing_dims.begin(), trailing_dims.end());  // 将 trailing_dims 的内容插入到 tensor_sizes 的末尾
    sequences.emplace_back(torch::rand(tensor_sizes));  // 将基于 tensor_sizes 创建的随机张量添加到 sequences 中
  }
  std::vector<torch::Tensor> unsorted_sequences;  // 创建空的 Torch 张量向量 unsorted_sequences
  for (const auto& s : sequences) {  // 遍历 sequences 中的每个张量 s
    // NOLINTNEXTLINE(performance-inefficient-vector-operation)
    unsorted_sequences.emplace_back(s.clone());  // 克隆每个张量 s 并将其添加到 unsorted_sequences 中
  }
  std::shuffle(
      std::begin(unsorted_sequences),  // 对 unsorted_sequences 中的张量进行随机打乱
      std::end(unsorted_sequences),
      std::default_random_engine{});  // 使用默认的随机引擎

  std::vector<int64_t> unsorted_sequences_lengths_vec;
    // 对未排序序列中的每个张量进行迭代
    for (const auto& t : unsorted_sequences) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      // 将每个张量的第一个维度大小加入到长度向量中
      unsorted_sequences_lengths_vec.emplace_back(t.size(0));
    }

    // 兼容其他实用工具
    // 针对批处理优先和排序强制执行的所有组合进行循环测试
    for (bool batch_first : std::vector<bool>{true, false}) {
      for (bool enforce_sorted : std::vector<bool>{true, false}) {
        // 执行兼容性测试，传入序列、长度向量、批处理优先标志和排序强制执行标志
        _compatibility_test(
            sequences, torch::tensor(lengths_vec), batch_first, enforce_sorted);
      }
      // 对未排序序列执行兼容性测试，传入未排序序列、其长度向量和批处理优先标志
      _compatibility_test(
          unsorted_sequences,
          torch::tensor(unsorted_sequences_lengths_vec),
          batch_first);
    }
  }
  // 在 NNUtilsTest 类中定义 PackPaddedSequence 测试函数
  TEST_F(NNUtilsTest, PackPaddedSequence) {
    // 定义 lambda 函数 generate_test_case，用于生成测试用例
    auto generate_test_case = [&](torch::ArrayRef<int64_t> sorted_lengths,
                                  bool should_shuffle) {
      // 定义 lambda 函数 pad，用于对张量进行填充
      auto pad = [&](torch::Tensor tensor, int64_t length) {
        // 计算需要填充的尺寸，并用零填充张量
        std::vector<int64_t> tensor_sizes{length - tensor.size(0)};
        tensor_sizes.insert(
            tensor_sizes.end(),
            tensor.sizes().slice(1).begin(),
            tensor.sizes().slice(1).end());
        return torch::cat({tensor, torch::zeros(tensor_sizes, tensor.options())});
      };
      // 获取最大长度
      int64_t max_length = sorted_lengths[0];
      // 创建包含批次大小的张量
      torch::Tensor batch_sizes = torch::empty({max_length}, torch::kInt64);
      // 计算每个批次的大小
      for (int64_t i = 1; i < max_length + 1; i++) {
        int64_t total = 0;
        for (const auto& x : sorted_lengths) {
          if (x >= i) {
            total++;
          }
        }
        batch_sizes[i - 1] = total;
      }
      // 创建用于拼接的张量向量
      std::vector<torch::Tensor> tensors_to_be_cat;
      // 生成填充后的张量
      for (int64_t i = 1; i < static_cast<int64_t>(sorted_lengths.size() + 1);
           i++) {
        int64_t l = sorted_lengths.at(i - 1);
        tensors_to_be_cat.emplace_back(pad(
            i * 100 + torch::arange(1., 5 * l + 1).view({l, 1, 5}), max_length));
      }
      // 拼接张量
      auto padded = torch::cat(tensors_to_be_cat, 1);
      // 创建预期数据的张量向量
      std::vector<torch::Tensor> expected_data_vec;
      // 生成预期数据
      for (const auto n : c10::irange(batch_sizes.size(0))) {
        int64_t batch_size = batch_sizes[n].item<int64_t>();
        for (const auto i : c10::irange(batch_size)) {
          expected_data_vec.emplace_back(
              torch::arange(1., 6) + (i + 1) * 100 + 5 * n);
        }
      }
      // 拼接预期数据的张量
      auto expected_data = torch::stack(expected_data_vec, /*dim=*/0);

      // 定义未排序索引和长度张量
      torch::Tensor unsorted_indices, lengths;
      if (should_shuffle) {
        // 如果需要打乱顺序，则打乱填充的序列
        std::vector<int64_t> permutation;
        for (const auto i : c10::irange(sorted_lengths.size())) {
          permutation.emplace_back(i);
        }
        std::shuffle(
            std::begin(permutation),
            std::end(permutation),
            std::default_random_engine{});

        // 创建未排序索引张量，并根据索引选择填充的张量和长度
        unsorted_indices = torch::tensor(permutation);
        padded = padded.index_select(1, unsorted_indices);
        lengths = torch::tensor(sorted_lengths).index_select(0, unsorted_indices);
      } else {
        // 如果不需要打乱顺序，则未排序索引为空张量，并使用排序长度
        unsorted_indices = torch::Tensor();
        lengths = torch::tensor(sorted_lengths);
      }

      // 返回结果元组，包括填充后的张量、长度、预期数据、批次大小及未排序索引
      return std::make_tuple(
          padded.requires_grad_(),
          lengths,
          expected_data,
          batch_sizes,
          unsorted_indices);
    };

    // 定义测试用例向量，包含不同的排序长度及是否打乱的标志
    std::vector<std::pair<std::vector<int64_t>, bool>> test_cases = {
        // sorted_lengths, should_shuffle
        {{10, 8, 4, 2, 2, 2, 1}, false},
        {{11, 10, 8, 6, 4, 3, 1}, false},
        {{11, 10, 8, 6, 4, 3, 1}, true}};

    // 遍历每个测试用例
    for (const auto& test_case : test_cases) {
    // 遍历布尔向量 `{true, false}`，测试不同的批次优先顺序
    for (bool batch_first : std::vector<bool>{true, false}) {
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      // 从测试用例中获取已排序长度和是否需要洗牌的信息
      std::vector<int64_t> sorted_lengths = std::get<0>(test_case);
      bool should_shuffle = std::get<1>(test_case);

      // 生成测试用例，包括填充后的张量、长度张量、预期数据、批次大小和未排序索引
      torch::Tensor padded, lengths, expected_data, batch_sizes,
          unsorted_indices;
      std::tie(padded, lengths, expected_data, batch_sizes, unsorted_indices) =
          generate_test_case(sorted_lengths, should_shuffle);

      // 将源张量设为填充后的张量，如果 `batch_first` 为真，则进行转置
      auto src = padded;
      if (batch_first) {
        src = src.transpose(0, 1);
      }

      // 检查输出
      rnn_utils::PackedSequence packed = rnn_utils::pack_padded_sequence(
          src,
          lengths,
          /*batch_first=*/batch_first,
          /*enforce_sorted=*/!should_shuffle);
      ASSERT_TRUE(torch::allclose(packed.data(), expected_data));
      ASSERT_TRUE(torch::allclose(packed.batch_sizes(), batch_sizes));
      ASSERT_TRUE(
          (!packed.unsorted_indices().defined() &&
           !unsorted_indices.defined()) ||
          torch::allclose(packed.unsorted_indices(), unsorted_indices));

      // 测试逆过程
      torch::Tensor unpacked, unpacked_len;
      std::tie(unpacked, unpacked_len) =
          rnn_utils::pad_packed_sequence(packed, /*batch_first=*/batch_first);
      ASSERT_TRUE(torch::allclose(unpacked, src));
      ASSERT_TRUE(torch::allclose(unpacked_len, lengths));

      // 检查梯度
      if (padded.grad().defined()) {
        // 禁止梯度计算
        torch::NoGradGuard no_grad;
        padded.grad().zero_();
      }
      torch::Tensor grad_output;
      {
        // 禁止梯度计算
        torch::NoGradGuard no_grad;
        // 生成梯度输出
        grad_output = unpacked.clone().normal_();
      }
      // 反向传播梯度
      unpacked.backward(grad_output);
      if (batch_first) {
        // 如果 `batch_first` 为真，则对梯度输出进行转置
        grad_output.transpose_(0, 1);
      }
      // 对每个批次中的每个元素进行验证
      for (const auto i : c10::irange(lengths.size(0))) {
        int64_t l = lengths[i].item<int64_t>();
        ASSERT_TRUE(torch::allclose(
            padded.grad().narrow(0, 0, l).select(1, i),
            grad_output.narrow(0, 0, l).select(1, i)));
        if (l < 10) {
          ASSERT_EQ(
              // 检查梯度张量中特定切片的绝对值和，确保为零
              padded.grad()
                  .narrow(0, l, padded.grad().size(0) - l)
                  .select(1, i)
                  .abs()
                  .sum()
                  .item<double>(),
              0);
        }
      }
    }
  }

  // 测试错误消息
  ASSERT_THROWS_WITH(
      // 检查 `pack_padded_sequence` 对于给定的不正确参数是否会抛出指定错误消息
      rnn_utils::pack_padded_sequence(
          torch::randn({3, 3}), torch::tensor({1, 3, 2})),
      "You can pass `enforce_sorted=False`");
  ASSERT_THROWS_WITH(
      // 检查 `pack_padded_sequence` 对于空张量是否会抛出指定错误消息
      rnn_utils::pack_padded_sequence(torch::randn({0, 0}), torch::tensor({})),
      "empty tensor");
// 在测试框架中定义测试用例 NNUtilsTest.PadSequence
TEST_F(NNUtilsTest, PadSequence) {
  // 定义匿名函数 pad，用于将输入张量 tensor 填充到指定长度 length
  auto pad = [&](const torch::Tensor& tensor, int64_t length) {
    // 禁用梯度计算
    torch::NoGradGuard no_grad;
    // 计算需要填充的维度大小
    std::vector<int64_t> tensor_sizes{length - tensor.size(0)};
    // 将张量 tensor 的其他维度大小插入到 tensor_sizes 中
    tensor_sizes.insert(
        tensor_sizes.end(),
        tensor.sizes().slice(1).begin(),
        tensor.sizes().slice(1).end());
    // 使用零张量进行填充，与原张量拼接得到填充后的张量
    return torch::cat({tensor, torch::zeros(tensor_sizes, tensor.options())});
  };

  // 定义单维度张量 a, b, c
  auto a = torch::tensor({1, 2, 3});
  auto b = torch::tensor({4, 5});
  auto c = torch::tensor({6});

  torch::Tensor expected, padded;

  // 设定期望的填充结果 expected，以及调用 rnn_utils::pad_sequence 进行填充操作，batch_first = true
  expected = torch::tensor({{4, 5, 0}, {1, 2, 3}, {6, 0, 0}});
  padded = rnn_utils::pad_sequence({b, a, c}, true);
  // 断言填充后的张量 padded 与期望结果 expected 在数值上是否接近
  ASSERT_TRUE(padded.allclose(expected));

  // batch_first = false 的情况
  padded = rnn_utils::pad_sequence({b, a, c});
  // 断言填充后的张量 padded 转置后是否与期望结果 expected 在数值上接近
  ASSERT_TRUE(padded.allclose(expected.transpose(0, 1)));

  // 使用非零值进行填充的情况
  expected = torch::tensor({{4, 5, 1}, {1, 2, 3}, {6, 1, 1}});
  padded = rnn_utils::pad_sequence({b, a, c}, true, 1);
  ASSERT_TRUE(padded.allclose(expected));

  // 测试排序后的序列填充
  expected = torch::tensor({{1, 2, 3}, {4, 5, 0}, {6, 0, 0}});
  padded = rnn_utils::pad_sequence({a, b, c}, true);
  ASSERT_TRUE(padded.allclose(expected));

  // 处理更多维度的情况
  int64_t maxlen = 9;
  for (int64_t num_dim : std::vector<int64_t>{0, 1, 2, 3}) {
    std::vector<torch::Tensor> sequences;
    std::vector<int64_t> trailing_dims(num_dim, 4);
    for (int64_t i = 1; i < maxlen + 1; i++) {
      int64_t seq_len = i * i;
      std::vector<int64_t> tensor_sizes{seq_len, 5};
      tensor_sizes.insert(
          tensor_sizes.end(), trailing_dims.begin(), trailing_dims.end());
      sequences.emplace_back(torch::rand(tensor_sizes));
    }
    // 随机打乱序列
    std::shuffle(
        std::begin(sequences),
        std::end(sequences),
        std::default_random_engine{});
    std::vector<torch::Tensor> expected_tensors;
    for (const torch::Tensor& seq : sequences) {
      // 将每个序列 seq 进行填充操作，并存入 expected_tensors 中
      expected_tensors.emplace_back(pad(seq, maxlen * maxlen));
    }

    // batch first = true 的情况下进行断言
    auto expected = torch::stack(expected_tensors);
    auto padded = rnn_utils::pad_sequence(sequences, true);
    ASSERT_TRUE(padded.allclose(expected));

    // batch first = false 的情况下进行断言
    padded = rnn_utils::pad_sequence(sequences);
    ASSERT_TRUE(padded.allclose(expected.transpose(0, 1)));
  }
}
```