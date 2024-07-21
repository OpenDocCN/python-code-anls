# `.\pytorch\aten\src\ATen\test\vulkan_api_test.cpp`

```py
#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下代码

// @lint-ignore-every CLANGTIDY
// 忽略 CLANGTIDY 的所有警告

#include <gtest/gtest.h>
// 包含 Google Test 框架的头文件

#include <ATen/ATen.h>
// 包含 ATen 张量库的头文件

#include <ATen/core/dispatch/Dispatcher.h>
// 包含 ATen 分发器的头文件

#include <ATen/native/vulkan/api/api.h>
// 包含 ATen Vulkan API 的头文件

#include <c10/util/irange.h>
// 包含 c10 的 irange 工具

#include <c10/util/ArrayRef.h>
// 包含 c10 的 ArrayRef 工具

// TODO: These functions should move to a common place.
// TODO: 这些函数应该移到一个通用的地方。

namespace {
// 匿名命名空间，限定了内部函数和变量的作用域

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float kTolerance = 1e-2;
#else
  constexpr float kTolerance = 1e-5;
#endif
// 根据 USE_VULKAN_FP16_INFERENCE 宏定义不同的容差值 kTolerance

bool checkRtol(const at::Tensor& diff, float maxTolerance) {
  // 检查张量 diff 的相对容差是否在最大容差 maxTolerance 内
  if (diff.numel() == 0) {
    return true;
  }
  return diff.abs().max().item<float>() <= maxTolerance;
}

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  // 检查张量 diff 相对容差是否在计算基于 inputs 中张量的最大值的容差 kTolerance 倍的范围内
  if (diff.numel() == 0) {
    return true;
  }
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

  return checkRtol(diff, kTolerance * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  // 检查两个张量 a 和 b 是否几乎相等，基于它们的相对容差
  return checkRtol(a - b, {a, b});
}

bool checkHardShrink(
    const at::Tensor& ref, const at::Tensor& out, const float clamp_thresh) {
  // 检查硬阈值收缩操作的结果是否满足预期
  float* ref_ptr = ref.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  float ref_max = ref.abs().max().item<float>();
  float out_max = out.abs().max().item<float>();
  float max_val = std::fmax(ref_max, out_max);

  float abs_clamp_thresh = std::abs(clamp_thresh);

  for (int i = 0; i < ref.numel(); ++i) {
    float ref_val = ref_ptr[i];
    float out_val = out_ptr[i];

    float abs_diff = std::abs(ref_val - out_val);

    // 对于接近阈值的值，结果可能模糊不清
    float distance_from_thresh = std::abs(std::abs(ref_val) - abs_clamp_thresh);
    if (distance_from_thresh < kTolerance * abs_clamp_thresh) {
      if (out_val != 0.0f) {
        if (abs_diff >= kTolerance * max_val) {
          return false;
        }
      }
    }
    else if (std::abs(ref_val) < std::abs(abs_clamp_thresh)) {
      if (out_val != 0.0f) {
        return false;
      }
    }
    else if (abs_diff >= kTolerance * max_val) {
      return false;
    }
  }
  return true;
}

bool checkThreshold(
    const at::Tensor& ref,
    const at::Tensor& out,
    const float clamp_thresh,
    const float value) {
  // 检查阈值操作的结果是否符合预期
  float* ref_ptr = ref.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  float ref_max = ref.abs().max().item<float>();
  float out_max = out.abs().max().item<float>();
  float max_val = std::fmax(ref_max, out_max);

  for (int i = 0; i < ref.numel(); ++i) {
    float ref_val = ref_ptr[i];
    float out_val = out_ptr[i];

    float abs_diff = std::abs(ref_val - out_val);
    float val_diff = std::abs(out_val - value);

    // 对于接近阈值的值，结果可能模糊不清
    float distance_from_thresh = std::abs(std::abs(ref_val) - clamp_thresh);
    // 检查阈值操作是否在容差范围内
    if (distance_from_thresh < kTolerance * clamp_thresh) {
        if (val_diff >= kTolerance * max_val) {
            return false;
        }
    } else if (abs_diff >= kTolerance * max_val) {
        return false;
    }
  }
  return true;
}
    # 如果距离阈值的差值小于阈值容差与阈值的乘积
    if (distance_from_thresh < kTolerance * clamp_thresh) {
      # 如果值的差值大于阈值容差与值的乘积
      if (val_diff >= kTolerance * value) {
        # 如果绝对差值大于阈值容差与最大值的乘积，则返回 false
        if (abs_diff >= kTolerance * max_val) {
          return false;
        }
      }
    }
    # 否则，如果参考值的绝对值小于阈值的绝对值
    else if (std::abs(ref_val) < std::abs(clamp_thresh)) {
      # 如果值的差值大于阈值容差与值的乘积，则返回 false
      if (val_diff >= kTolerance * value) {
        return false;
      }
    }
    # 否则，如果绝对差值大于阈值容差与最大值的乘积，则返回 false
    else if (abs_diff >= kTolerance * max_val) {
      return false;
    }
  }
  # 如果以上条件均不满足，则返回 true
  return true;
}

void showRtol(const at::Tensor& a, const at::Tensor& b) {
  // 计算张量 a 和 b 的绝对差
  const auto diff = (a - b).abs();

  // 计算张量 a 和 b 的绝对值的最大值
  float maxValue = a.abs().max().item<float>();
  maxValue = fmax(b.abs().max().item<float>(), maxValue);

  // 计算允许的最大差值
  const float maxDiff = maxValue * kTolerance;
  std::cout << "Max Diff allowed: " << maxDiff << std::endl;

  // 如果差值张量的维度为二维
  if (diff.sizes().size() == 2) {
    // 遍历差值张量的第一维度
    for (const auto y : c10::irange(diff.sizes()[0])) {
      std::cout << y << ":";
      // 遍历差值张量的第二维度
      for (const auto x : c10::irange(diff.sizes()[1])) {
        float diff_xy = diff[y][x].item<float>();
        // 如果差值超过允许的最大差值，则输出 x 的位置
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
        }
        else {
          std::cout << std::setw(5) << " ";
        }
      }
      std::cout << std::endl;
    }
  }
}


static void gen_allpermutations(std::vector<std::vector<int64_t>>& out, std::vector<int64_t> in, unsigned i) {
  // 生成给定维度的所有排列组合
  if (i == in.size()) {
    out.push_back(in);
  }
  else {
    for (const auto j : c10::irange(i, in.size())) {
      std::swap(in[i], in[j]);
      gen_allpermutations(out, in, i + 1);
    }
  }
}

static void gen_all_subsets(
    std::vector<std::vector<int64_t>>& out,
    int64_t n,
    unsigned i,
    std::vector<int64_t> curr) {
  // 通过回溯法生成集合 {0,...,n - 1} 的所有子集
  if (i == n) {
    out.push_back(curr);
  } else {
    curr.push_back(i);
    gen_all_subsets(out, n, i + 1, curr);
    curr.pop_back();
    gen_all_subsets(out, n, i + 1, curr);
  }
}

static void slice_test(
    const std::vector<int64_t>& size,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  // 准备
  const auto in_cpu = at::rand(size, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  // 操作
  const auto out_cpu = at::slice(in_cpu, dim, start, end, step);
  const auto out_vulkan = at::slice(in_vulkan, dim, start, end, step);

  // 断言
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

static void slice_tests(const std::unordered_map<int64_t, std::vector<int64_t>>& dim2sizes) {
  for (const auto& dim2size : dim2sizes) {
    // 进行切片测试
    slice_test(dim2size.second, dim2size.first, 10, 30, 1);         // 即 4 维张量的等效索引 = [:,:,:,10:30:1]
    slice_test(dim2size.second, dim2size.first, 10, 30, 7);         // 即 4 维张量的等效索引 = [:,:,:,10:30:7]
    slice_test(dim2size.second, dim2size.first, 10, 50, 2);         // 即 4 维张量的等效索引 = [:,:,:,10:50:2]，其中 end 超出范围
    slice_test(dim2size.second, dim2size.first, -60, 60, 2);        // 即 4 维张量的等效索引 = [:,:,:,-60:60:2]，其中 start/end 超出范围
    slice_test(dim2size.second, dim2size.first, -30, -10, 1);       // 即 4 维张量的等效索引 = [:,:,:,-30:-10:1]，其中负数 start/end
    // 使用 slice_test 函数进行切片测试，用法类似于四维数组的索引操作
    // 第一个参数是 dim2size.second，第二个参数是 dim2size.first，后面的参数依次是起始索引、终止索引、步长
    // 对应于 4D 数组的索引操作：[:,:,:,:0:9223372036854775807:1]，其中终止索引为 INT64_MAX
    slice_test(dim2size.second, dim2size.first, 0, INT64_MAX, 1);

    // 使用 slice_test 函数进行切片测试，用法类似于四维数组的索引操作，但起始索引为负数，终止索引为 INT64_MAX
    // 对应于 4D 数组的索引操作：[:,:,:,-10:9223372036854775807:1]，其中起始索引为 -10，终止索引为 INT64_MAX
    slice_test(dim2size.second, dim2size.first, -10, INT64_MAX, 1);

    // 此行代码会触发 SymInt 断言，因为[-2^63, -2^62-1] 范围保留用于打包的 SymInt
    // slice_test(dim2size.second, dim2size.first, INT64_MIN, INT64_MAX, 1);
    // 上述代码被注释掉了，其对应的 4D 数组索引操作是 [:,:,:,-9223372036854775808:9223372036854775807:1]
    // 其中起始索引为 INT64_MIN，终止索引为 INT64_MAX

    // 使用 slice_test 函数进行切片测试，用法类似于四维数组的索引操作，起始索引和终止索引为空字典
    // 对应于 4D 数组的索引操作：[:,:,:,:,:1]，其中起始索引和终止索引都为空
    slice_test(dim2size.second, dim2size.first, {}, {}, 1);
}
namespace {

class VulkanAPITest : public ::testing::Test {
 public:
  void SetUp() {
    // 检查 Vulkan 是否可用，如果不可用则跳过测试
    if (!at::is_vulkan_available()) {
      GTEST_SKIP() << "Vulkan is not available";
    }
    // 如果定义了使用 Vulkan GPU 诊断并且在 Android 平台上
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    // 如果 Vulkan 上下文中的操作性能分析已启用，则重置查询池
    if (at::native::vulkan::api::context()->op_profiling_enabled()) {
      at::native::vulkan::api::context()->reset_querypool();
    }
#endif
  }

  void TearDown() {
    // 如果定义了使用 Vulkan GPU 诊断并且在 Android 平台上
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    // 如果 Vulkan 上下文中的操作性能分析已启用
    if (at::native::vulkan::api::context()->op_profiling_enabled()) {
      try {
        // 提取查询池的结果并打印
        at::native::vulkan::api::context()->querypool().extract_results();
        at::native::vulkan::api::context()->querypool().print_results();
      } catch (const std::exception& e) {
        // 捕获异常，打印无法获取查询池结果的原因
        std::cout << "Could not get querypool results!"
                  << " Reason: " << e.what() << std::endl;
      }
    }
#endif
  }
};

TEST_F(VulkanAPITest, zero_size_tensor) {
  // 创建一个 CPU 上的零大小张量
  auto cpu = at::rand({1, 0, 0}, at::device(at::kCPU).dtype(at::kFloat));
  // 将其转换为 Vulkan 张量
  auto vk = cpu.vulkan();
  // 将 Vulkan 张量还原为 CPU 张量
  auto out_vk = vk.cpu();
  // 断言 Vulkan 张量与原始 CPU 张量相等
  ASSERT_TRUE(at::equal(out_vk, cpu));
}

TEST_F(VulkanAPITest, zero_size_tensor_numel) {
  // 创建一个 Vulkan 上的零大小张量
  auto vk = at::rand({18, 0, 5}, at::device(at::kVulkan).dtype(at::kFloat));
  // 断言张量元素个数为 0
  ASSERT_TRUE(vk.numel() == 0);
}

TEST_F(VulkanAPITest, zero_dim_tensor_1) {
  // 创建一个零维度的 CPU 张量
  auto cpu = at::rand({}, at::device(at::kCPU).dtype(at::kFloat));
  // 获取 CPU 张量的单个数值
  auto vv = cpu.item<float>();

  // 将 CPU 张量转换为 Vulkan 张量，再转回 CPU 张量
  auto vk = cpu.vulkan();
  auto out_vk = vk.cpu();
  // 断言 Vulkan 张量与原始 CPU 张量几乎相等
  ASSERT_TRUE(almostEqual(cpu, out_vk));

  // 获取转换后 CPU 张量的单个数值，验证其与原始 CPU 张量的单个数值接近
  auto vk_vv = out_vk.item<float>();
  EXPECT_NEAR(vv, vk_vv, kTolerance);
}

} // namespace
TEST_F(VulkanAPITest, zero_dim_tensor_2) {
  // 定义一个浮点数变量v，赋值为3.14
  float v = 3.14f;
  // 创建一个CPU上的零维张量，数据类型为float，并加上v
  auto cpu = at::zeros({}, at::device(at::kCPU).dtype(at::kFloat)) + v;
  // 创建一个Vulkan上的零维张量，数据类型为float，并加上v
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat)) + v;

  // 断言CPU张量和Vulkan张量的值几乎相等
  ASSERT_TRUE(almostEqual(cpu, vk.cpu()));
}

TEST_F(VulkanAPITest, zero_dim_tensor_3) {
  // 创建一个Vulkan上的零维张量，数据类型为float
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat));

  // 断言Vulkan张量经CPU处理后的单个浮点数值等于0.0
  ASSERT_TRUE(vk.cpu().item<float>() == 0.0f);
}

TEST_F(VulkanAPITest, local_scalar_dense) {
  // 定义一个浮点数变量v，赋值为8.31
  float v = 8.31f;
  // 创建一个Vulkan上的零维张量，数据类型为float，并加上v
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat)) + v;
  // 获取Vulkan张量的本地标量值
  c10::Scalar scalar = at::_local_scalar_dense(vk);
  // 使用容差值进行断言，验证标量值与v几乎相等
  EXPECT_NEAR(v, scalar.toFloat(), kTolerance);
}

TEST_F(VulkanAPITest, copy_to_texture) {
  using namespace at::native::vulkan;
  // 定义测试用的CPU张量数组，包含不同维度的随机浮点数张量
  at::Tensor test_tensors[] = {
    // 4维张量
    at::rand({7, 17, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 3维张量
    at::rand({67, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 2维张量
    at::rand({229, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 1维张量
    at::rand({1902}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
  };

  // 遍历CPU张量数组
  for (auto in_cpu : test_tensors) {
    // 将CPU张量复制到Vulkan张量
    at::Tensor in_vk_copied = in_cpu.vulkan();
    // 将Vulkan张量复制回CPU
    at::Tensor out_copied = in_vk_copied.cpu();

    // 检查复制后的张量是否几乎相等
    const auto check_copy = almostEqual(out_copied, in_cpu);

    // 如果检查失败，输出相应的错误信息
    if(!check_copy) {
      std::cout << "Copy failed on size " << in_cpu.sizes()
                << "with dtype" << in_cpu.dtype() << std::endl;
    }

    // 断言复制结果为真
    ASSERT_TRUE(check_copy);
  }
}

void test_copy_to_texture_bool(const at::IntArrayRef input_shape) {
  using namespace at::native::vulkan;
  // 创建一个指定形状的CPU布尔类型张量
  auto cpu = at::randint(0, 2, input_shape, at::TensorOptions(at::kCPU).dtype(at::kBool));
  // 将CPU张量复制到Vulkan张量
  auto in_vulkan = cpu.vulkan();

  // 将Vulkan张量复制回CPU
  auto out_vulkan = in_vulkan.cpu();
  // 检查CPU张量与复制回来的张量是否相等
  auto check = at::equal(cpu, out_vulkan.cpu());

  // 如果检查失败，输出相应的错误信息
  if (!check) {
    std::cout << "Copy texture to bool failed on input_shape " << input_shape << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, copy_to_texture_bool_mul4_hw) {
  // 使用shader: image_to_nchw_quantized_mul4 ((H * W) % 4 == 0)
  // ch % 4 != 0,  ch < 4
  // 测试不同形状的布尔类型CPU张量复制到Vulkan张量并验证结果
  test_copy_to_texture_bool({5, 1, 2, 2});
  test_copy_to_texture_bool({17, 2, 4, 2});
  test_copy_to_texture_bool({9, 3, 3, 8});

  // ch % 4 != 0, ch > 5
  test_copy_to_texture_bool({7, 17, 4, 8});
  test_copy_to_texture_bool({8, 6, 2, 4});
  test_copy_to_texture_bool({13, 31, 4, 57});

  // 3维、2维、1维张量的测试
  test_copy_to_texture_bool({17, 31, 4});
  test_copy_to_texture_bool({64, 16});
  test_copy_to_texture_bool({8});
}

TEST_F(VulkanAPITest, copy_to_texture_bool_mul4_chw) {
  // 使用shader: image_to_nchw_quantized_mul4 ((H * W) % 4 == 0)
  // ch % 4 == 0
  // 测试特定形状的布尔类型CPU张量复制到Vulkan张量并验证结果
  test_copy_to_texture_bool({5, 16, 2, 16});
  test_copy_to_texture_bool({8, 8, 2, 2});
  test_copy_to_texture_bool({16, 31, 4});
}
TEST_F(VulkanAPITest, copy_to_texture_bool) {
  // 使用 shader: image_to_nchw_uint ((H * W) % 4 != 0)
  test_copy_to_texture_bool({13, 1, 3, 5});  // 测试函数：将数据复制到纹理，输入维度为 {13, 1, 3, 5}
  test_copy_to_texture_bool({13, 7, 1, 5});  // 测试函数：将数据复制到纹理，输入维度为 {13, 7, 1, 5}
  test_copy_to_texture_bool({13, 8, 2, 5});  // 测试函数：将数据复制到纹理，输入维度为 {13, 8, 2, 5}
  test_copy_to_texture_bool({13, 31, 2, 57});  // 测试函数：将数据复制到纹理，输入维度为 {13, 31, 2, 57}

  test_copy_to_texture_bool({67, 19, 7});  // 测试函数：将数据复制到纹理，输入维度为 {67, 19, 7}
  test_copy_to_texture_bool({229, 213});  // 测试函数：将数据复制到纹理，输入维度为 {229, 213}
  test_copy_to_texture_bool({1902});  // 测试函数：将数据复制到纹理，输入维度为 {1902}
}

TEST_F(VulkanAPITest, adaptive_avg_pool2d) {
  c10::InferenceMode mode;  // 进入推断模式

  const auto in_cpu = at::rand({5, 7, 47, 31}, at::TensorOptions(at::kCPU).dtype(at::kFloat));  // 生成随机张量，形状为 {5, 7, 47, 31}，CPU 上的 Float 类型
  const auto out_cpu = at::adaptive_avg_pool2d(in_cpu, {3, 3});  // 在 CPU 上进行自适应平均池化，输出形状为 {3, 3}

  const auto out_vulkan = at::adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});  // 在 Vulkan 上进行自适应平均池化，输出形状为 {3, 3}

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());  // 检查 Vulkan 和 CPU 的输出是否几乎相等
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());  // 如果不相等，展示它们的相对误差
  }

  ASSERT_TRUE(check);  // 断言检查结果为真
}

void test_add(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape, float alpha) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));  // 生成输入张量，在 CPU 上，指定形状和数据类型为 Float
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));  // 生成另一个输入张量，在 CPU 上，指定形状和数据类型为 Float

  const auto in_vulkan = in_cpu.vulkan();  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto other_vulkan = other_cpu.vulkan();  // 将 CPU 上的另一个输入张量转换为 Vulkan 张量

  const auto out_cpu = at::add(in_cpu, other_cpu, alpha);  // 执行张量相加操作，在 CPU 上
  const auto out_vulkan = at::add(in_vulkan, other_vulkan, alpha);  // 执行张量相加操作，在 Vulkan 上

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());  // 检查 Vulkan 和 CPU 的输出是否几乎相等
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());  // 如果不相等，展示它们的相对误差
  }

  ASSERT_TRUE(check);  // 断言检查结果为真
}

TEST_F(VulkanAPITest, add_invalid_inputs) {
  // 对于二进制逐元素操作，广播时维度不兼容
  auto in_cpu = at::rand({2, 3, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));  // 生成随机张量，在 CPU 上，形状为 {2, 3, 4, 5}，数据类型为 Float
  auto other_cpu = at::rand({2, 4, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));  // 生成随机张量，在 CPU 上，形状为 {2, 4, 4, 5}，数据类型为 Float

  EXPECT_THROW(at::add(in_cpu.vulkan(), other_cpu.vulkan(), 1.0f), ::std::exception);  // 预期在 Vulkan 上执行张量相加操作时抛出异常
}

TEST_F(VulkanAPITest, add) {
  test_add({2, 3}, {2, 3}, 1.0f);  // 测试函数：执行张量相加，输入形状分别为 {2, 3} 和 {2, 3}，alpha 值为 1.0
  test_add({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);  // 测试函数：执行张量相加，输入形状分别为 {11, 7, 139, 109} 和 {11, 7, 139, 109}，alpha 值为 2.1
}

TEST_F(VulkanAPITest, add_broadcast0) {
  test_add({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);  // 测试函数：执行张量相加，输入形状分别为 {3, 5, 179, 221} 和 {3, 5, 1, 1}，alpha 值为 1.8
}

TEST_F(VulkanAPITest, add_broadcast1) {
  test_add({3, 5, 179, 221}, {3, 5, 1, 221}, 1.8f);  // 测试函数：执行张量相加，输入形状分别为 {3, 5, 179, 221} 和 {3, 5, 1, 221}，alpha 值为 1.8
}

TEST_F(VulkanAPITest, add_broadcast2) {
  test_add({3, 4, 179, 221}, {4, 1, 1}, 2.5f);  // 测试函数：执行张量相加，输入形状分别为 {3, 4, 179, 221} 和 {4, 1, 1}，alpha 值为 2.5
}

TEST_F(VulkanAPITest, add_broadcast3) {
  test_add({3, 4, 41, 53}, {1, 1, 41, 53}, 2.5f);  // 测试函数：执行张量相加，输入形状分别为 {3, 4, 41, 53} 和 {1, 1, 41, 53}，alpha 值为 2.5
}

TEST_F(VulkanAPITest, add_broadcast4) {
  test_add({3, 4, 41, 1}, {1, 41, 53}, 2.5f);  // 测试函数：执行张量相加，输入形状分别为 {3, 4, 41, 1} 和 {1, 41, 53}，alpha 值为 2.5
}

TEST_F(VulkanAPITest, add_broadcast5) {
  test_add({2, 1, 7, 1}, {1, 5, 1, 4}, 1.2f);  // 测试函数：执行张量相加，输入形状分别为 {2, 1, 7, 1} 和 {1, 5, 1, 4}，alpha 值为 1.2
}

TEST_F(VulkanAPITest, add_broadcast6) {
  test_add({1, 15, 5, 4}, {21, 1, 5, 4}, 1.8f);  // 测试函数：执行张量相加，输入形状分别为 {1,
    // 在 CPU 上生成一个指定形状的随机浮点数张量
    const auto in_cpu =
        at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
    // 在 CPU 上生成另一个指定形状的随机浮点数张量，并将其乘以 100 后转换为整型
    const auto other_cpu =
        (at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) * 100)
            .to(at::kInt);
    
    // 将 CPU 上的输入张量转换为 Vulkan 张量
    const auto in_vulkan = in_cpu.vulkan();
    
    // 使用指定的 alpha 值，在 CPU 上执行张量加法
    const auto out_cpu = at::add(in_cpu, other_cpu, alpha);
    // 使用指定的 alpha 值，在 Vulkan 上执行张量加法
    const auto out_vulkan = at::add(in_vulkan, other_cpu, alpha);
    
    // 检查 CPU 上的输出张量和 Vulkan 上的输出张量是否几乎相等
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    // 如果不相等，显示两者的相对误差
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }
    
    // 使用断言确保 CPU 输出和 Vulkan 输出几乎相等
    ASSERT_TRUE(check);
TEST_F(VulkanAPITest, add_other_cpu_int) {
  // 调用测试函数，测试在两个相同形状的张量上进行加法操作
  test_add_other_cpu_int({2, 3}, {2, 3}, 1.0f);
  // 调用测试函数，测试在两个相同形状的张量上进行加法操作
  test_add_other_cpu_int({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);
}

TEST_F(VulkanAPITest, add_broadcast0_other_cpu_int) {
  // 调用测试函数，测试在两个张量中一个维度为1的情况下进行加法操作
  test_add_other_cpu_int({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);
}

TEST_F(VulkanAPITest, add_other_cpu_unsupported_type_should_fail) {
  // 生成形状为 [2, 2, 2] 的随机张量，数据类型为 kFloat，放在 CPU 上
  const auto in_cpu = at::rand({2,2,2}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 [2, 2, 2] 的零张量，数据类型为 kComplexFloat，放在 CPU 上
  const auto other_cpu = at::zeros({2, 2, 2}, at::device(at::kCPU).dtype(at::kComplexFloat));
  
  // 期望在调用 Vulkan 的 add 函数时抛出异常
  EXPECT_THROW(at::add(in_cpu.vulkan(), other_cpu.vulkan(), 1.0f), ::std::exception);
}

TEST_F(VulkanAPITest, add_) {
  // 生成形状为 [61, 17, 29, 83] 的随机张量，数据类型为 kFloat，放在 CPU 上
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 生成形状为 [61, 17, 29, 83] 的随机张量，数据类型为 kFloat，放在 CPU 上
  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 在 a_cpu 上进行原地加法操作，增量因子为 2.1
  a_cpu.add_(b_cpu, 2.1f);
  // 在 a_vulkan 上进行原地加法操作，增量因子为 2.1
  a_vulkan.add_(b_vulkan, 2.1f);

  // 检查 a_cpu 与 a_vulkan 的近似相等性
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果检查不通过，展示它们之间的相对容差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言检查通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast0_) {
  // 生成形状为 [16, 17, 29, 83] 的随机张量，数据类型为 kFloat，放在 CPU 上
  auto a_cpu = at::rand({16, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 生成形状为 [16, 17, 29, 1] 的随机张量，数据类型为 kFloat，放在 CPU 上
  const auto b_cpu = at::rand({16, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 在 a_cpu 上进行原地加法操作，增量因子为 2.1
  a_cpu.add_(b_cpu, 2.1f);
  // 在 a_vulkan 上进行原地加法操作，增量因子为 2.1
  a_vulkan.add_(b_vulkan, 2.1f);

  // 检查 a_cpu 与 a_vulkan 的近似相等性
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果检查不通过，展示它们之间的相对容差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言检查通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_other_cpu_int_) {
  // 定义形状为 [12, 17, 29, 33] 的输入张量形状
  std::vector<int64_t> input_shape{12, 17, 29, 33};
  // 生成形状为 input_shape 的随机张量，数据类型为 kFloat，放在 CPU 上
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 input_shape 的随机张量，数据类型为 kFloat，乘以 100 后转换为 kInt，放在 CPU 上
  const auto other_cpu = (at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) * 100).to(at::kInt);

  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 定义增量因子 alpha
  float alpha = -8.31f;
  // 在 in_cpu 上进行加法操作，增量因子为 alpha
  in_cpu.add(other_cpu, alpha);
  // 在 in_vulkan 上进行加法操作，增量因子为 alpha
  in_vulkan.add(other_cpu, alpha);

  // 检查 in_cpu 与 in_vulkan 的近似相等性
  const auto check = almostEqual(in_cpu, in_vulkan.cpu());
  // 如果检查不通过，展示它们之间的相对容差
  if (!check) {
    showRtol(in_cpu, in_vulkan.cpu());
  }
}

TEST_F(VulkanAPITest, add_broadcast1_) {
  // 生成形状为 [3, 8, 29, 83] 的随机张量，数据类型为 kFloat，放在 CPU 上
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 生成形状为 [3, 8, 1, 1] 的随机张量，数据类型为 kFloat，放在 CPU 上
  const auto b_cpu = at::rand({3, 8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 在 a_cpu 上进行原地加法操作，增量因子为 2.1
  a_cpu.add_(b_cpu, 2.1f);
  // 在 a_vulkan 上进行原地加法操作，增量因子为 2.1
  a_vulkan.add_(b_vulkan, 2.1f);

  // 检查 a_cpu 与 a_vulkan 的近似相等性
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果检查不通过，展示 b_cpu 与 b_vulkan 之间的相对容差
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  // 断言检查通过
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, add_scalar) {
  // 生成一个形状为 [13, 23, 59, 73] 的随机张量，数据类型为 float，放在 CPU 上
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto a_vulkan = a_cpu.vulkan();

  // 定义一个浮点数标量 b_scalar
  const float b_scalar = 3.1415f;

  // 对 a_cpu 和 b_scalar 进行加法操作，结果存储在 c_cpu 中，使用缩放因子 2.1
  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  // 对 a_vulkan 和 b_scalar 进行加法操作，结果存储在 c_vulkan 中，使用缩放因子 2.1
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  // 检查 c_cpu 和 c_vulkan 是否近似相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果不相等，则显示它们的相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言 c_cpu 和 c_vulkan 近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_) {
  // 生成一个形状为 [47, 2, 23, 97] 的随机张量，数据类型为 float，放在 CPU 上
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 定义一个浮点数标量 b_scalar
  const float b_scalar = 3.1415f;

  // 对 a_cpu 执行原地加法操作，加上 b_scalar，使用缩放因子 2.1
  a_cpu.add_(b_scalar, 2.1f);
  // 对 a_vulkan 执行原地加法操作，加上 b_scalar，使用缩放因子 2.1
  a_vulkan.add_(b_scalar, 2.1f);

  // 检查 a_cpu 和 a_vulkan 是否近似相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果不相等，则显示它们的相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言 a_cpu 和 a_vulkan 近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_wrapped) {
  // 如果 Vulkan 不可用，则直接返回
  if (!at::is_vulkan_available()) {
    return;
  }

  // 生成一个形状为 [13, 23, 59, 73] 的随机张量，数据类型为 float，放在 CPU 上
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto a_vulkan = a_cpu.vulkan();

  // 生成一个形状为 [1] 的随机张量，数据类型为 float，放在 CPU 上
  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 对 a_cpu 和 b_scalar 进行加法操作，结果存储在 c_cpu 中，使用缩放因子 2.1
  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  // 对 a_vulkan 和 b_scalar 进行加法操作，结果存储在 c_vulkan 中，使用缩放因子 2.1
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  // 检查 c_cpu 和 c_vulkan 是否近似相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果不相等，则显示它们的相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言 c_cpu 和 c_vulkan 近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_wrapped_) {
  // 如果 Vulkan 不可用，则直接返回
  if (!at::is_vulkan_available()) {
    return;
  }

  // 生成一个形状为 [47, 2, 23, 97] 的随机张量，数据类型为 float，放在 CPU 上
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 生成一个形状为 [1] 的随机张量，数据类型为 float，放在 CPU 上
  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 对 a_cpu 执行原地加法操作，加上 b_scalar，使用缩放因子 2.1
  a_cpu.add_(b_scalar, 2.1f);
  // 对 a_vulkan 执行原地加法操作，加上 b_scalar，使用缩放因子 2.1
  a_vulkan.add_(b_scalar, 2.1f);

  // 检查 a_cpu 和 a_vulkan 是否近似相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果不相等，则显示它们的相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言 a_cpu 和 a_vulkan 近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_to_scalar_wrapped) {
  // 如果 Vulkan 不可用，则直接返回
  if (!at::is_vulkan_available()) {
    return;
  }

  // 生成一个形状为 [1] 的随机张量，数据类型为 float，放在 CPU 上
  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 生成一个形状为 [11, 7, 139, 109] 的随机张量，数据类型为 float，放在 CPU 上
  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 对 a 和 b_cpu 进行加法操作，结果存储在 c_cpu 中，使用缩放因子 2.1
  const auto c_cpu = at::add(a, b_cpu, 2.1f);
  // 对 a 和 b_vulkan 进行加法操作，结果存储在 c_vulkan 中，使用缩放因子 2.1
  const auto c_vulkan = at::add(a, b_vulkan, 2.1f);

  // 检查 c_cpu 和 c_vulkan 是否近似相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果不相等，则显示它们的相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言 c_cpu 和 c_vulkan 近似相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, addmm) {
  // 定义常量 alpha 和 beta 用于矩阵相乘操作
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // 生成随机偏置、矩阵 m1 和 m2，并在 CPU 上指定数据类型为浮点数
  const auto bias_cpu = at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行 addmm 操作，将结果存储在 out_cpu 中
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();

  // 在 Vulkan 环境中执行 addmm 操作，将结果存储在 out_vulkan 中
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  // 检查在相对容差下 out_cpu 和 out_vulkan 是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());

  // 如果结果不接近，显示相对容差的详细信息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 check 为真，即确认 CPU 和 Vulkan 环境下的结果一致
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_expand) {
  // 定义常量 alpha 和 beta 用于矩阵相乘操作
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // 生成随机偏置、矩阵 m1 和 m2，并在 CPU 上指定数据类型为浮点数
  const auto bias_cpu = at::rand({1000}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({1, 1280}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({1280, 1000}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行 addmm 操作，将结果存储在 out_cpu 中
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();

  // 在 Vulkan 环境中执行 addmm 操作，将结果存储在 out_vulkan 中
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  // 检查在相对容差下 out_cpu 和 out_vulkan 是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());

  // 如果结果不接近，显示相对容差的详细信息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 check 为真，即确认 CPU 和 Vulkan 环境下的结果一致
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_expand2) {
  // 定义常量 alpha 和 beta 用于矩阵相乘操作
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // 生成随机偏置、矩阵 m1 和 m2，并在 CPU 上指定数据类型为浮点数
  const auto bias_cpu = at::rand({9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({17, 6}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({6, 9}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行 addmm 操作，将结果存储在 out_cpu 中
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();

  // 在 Vulkan 环境中执行 addmm 操作，将结果存储在 out_vulkan 中
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  // 检查在相对容差下 out_cpu 和 out_vulkan 是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());

  // 如果结果不接近，显示相对容差的详细信息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 check 为真，即确认 CPU 和 Vulkan 环境下的结果一致
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_error_bias) {
  // 定义常量 alpha 和 beta 用于矩阵相乘操作
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // 生成随机偏置、矩阵 m1 和 m2，并在 CPU 上指定数据类型为浮点数
  // 注意：此处的 bias_cpu 尺寸不匹配，应该是一维数组或者{17, 9}的形状
  const auto bias_cpu = at::rand({5, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({17, 6}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({6, 9}, at::device(at::kCPU).dtype(at::kFloat));

  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();

  // 预期抛出异常，因为 bias_cpu 的尺寸不正确
  EXPECT_THROW(at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha), ::std::exception);
}

TEST_F(VulkanAPITest, avg_pool2d) {
  // 生成随机输入张量 in_cpu，指定数据类型为浮点数
  const auto in_cpu = at::rand({3, 19, 43, 79}, at::TensorOptions(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行 2D 平均池化操作，将结果存储在 out_cpu 中
  const auto out_cpu = at::avg_pool2d(in_cpu, {5, 3}, {1, 2}, {2, 0}, true);

  // 将 in_cpu 转换为 Vulkan 张量，并在 Vulkan 环境中执行 2D 平均池化操作
  const auto out_vulkan = at::avg_pool2d(in_cpu.vulkan(), {5, 3}, {1, 2}, {2, 0}, true);

  // 检查在相对容差下 out_cpu 和 out_vulkan 是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());

  // 如果结果不接近，显示相对容差的详细信息
  if (!check) {
    // showRtol(out_cpu, out_vulkan.cpu());  // 此行为示例中未提供的函数，可以根据需要添加注释
  }

  // 断言 check 为真，即确认 CPU 和 Vulkan 环境下的结果一致
  ASSERT_TRUE(check);
}
    // 调用函数showRtol，将out_cpu和out_vulkan.cpu()作为参数传递
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用ASSERT_TRUE宏来验证check变量为true，如果不为true则输出错误信息
  ASSERT_TRUE(check);
// 测试用例函数结束标志，批量归一化的无效输入测试
TEST_F(VulkanAPITest, DISABLED_batch_norm_invalid_inputs) {
  // 进入推理模式
  c10::InferenceMode mode;

  // 行为: Vulkan 批量归一化仅支持评估模式
  EXPECT_THROW({
    // 执行 Vulkan 批量归一化操作，期望抛出异常
    at::batch_norm(
      // 生成随机张量作为输入，并转换为 Vulkan 张量
      at::rand({3, 8, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 生成随机张量作为参数，并转换为 Vulkan 张量
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 指定其他批量归一化参数
      true,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // 行为: Vulkan 批量归一化期望4维输入
  EXPECT_THROW({
    // 执行 Vulkan 批量归一化操作，期望抛出异常
    at::batch_norm(
      // 生成随机张量作为输入，并转换为 Vulkan 张量
      at::rand({3, 8, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 生成随机张量作为参数，并转换为 Vulkan 张量
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 指定其他批量归一化参数
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // 行为: Vulkan 批量归一化期望4维输入
  EXPECT_THROW({
    // 执行 Vulkan 批量归一化操作，期望抛出异常
    at::batch_norm(
      // 生成随机张量作为输入，并转换为 Vulkan 张量
      at::rand({2, 8, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 生成随机张量作为参数，并转换为 Vulkan 张量
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 指定其他批量归一化参数
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // 行为: Vulkan 批量归一化期望通道维度是4的倍数
  EXPECT_THROW({
    // 执行 Vulkan 批量归一化操作，期望抛出异常
    at::batch_norm(
      // 生成随机张量作为输入，并转换为 Vulkan 张量
      at::rand({4, 7, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 生成随机张量作为参数，并转换为 Vulkan 张量
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 指定其他批量归一化参数
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // 行为: 权重张量包含不正确数量的元素
  EXPECT_THROW({
    // 执行 Vulkan 批量归一化操作，期望抛出异常
    at::batch_norm(
      // 生成随机张量作为输入，并转换为 Vulkan 张量
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 生成随机张量作为参数，并转换为 Vulkan 张量
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 指定其他批量归一化参数
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // 行为: 偏置张量包含不正确数量的元素
  EXPECT_THROW({
    // 调用 ATen 库中的 batch_norm 函数，进行批量归一化操作
    at::batch_norm(
      // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      // 是否在训练模式下执行批量归一化
      false,
      // 动量参数
      0.1,
      // 用于数值稳定性的 epsilon 参数
      1e-05,
      // 是否使用 CuDNN 后端（这里为 false 表示不使用）
      false);
    }, ::std::exception);
    
    // 预期：运行均值张量包含不正确数量的元素时，抛出异常
    EXPECT_THROW({
      at::batch_norm(
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 是否在训练模式下执行批量归一化
        false,
        // 动量参数
        0.1,
        // 用于数值稳定性的 epsilon 参数
        1e-05,
        // 是否使用 CuDNN 后端（这里为 false 表示不使用）
        false);
    }, ::std::exception);
    
    // 预期：运行方差张量包含不正确数量的元素时，抛出异常
    EXPECT_THROW({
      at::batch_norm(
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 使用 CPU 设备生成指定形状的随机张量，数据类型为浮点数，然后转换为 Vulkan 张量
        at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        // 是否在训练模式下执行批量归一化
        false,
        // 动量参数
        0.1,
        // 用于数值稳定性的 epsilon 参数
        1e-05,
        // 是否使用 CuDNN 后端（这里为 false 表示不使用）
        false);
    }, ::std::exception);
TEST_F(VulkanAPITest, batch_norm_small) {
  c10::InferenceMode mode;  // 进入推断模式

  // 生成 CPU 上的随机输入张量
  const auto input_cpu = at::rand({1, 4, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto input_vulkan = input_cpu.vulkan();

  // 生成 CPU 上的随机权重张量
  const auto weight_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的权重张量转换为 Vulkan 张量
  const auto weight_vulkan = weight_cpu.vulkan();

  // 生成 CPU 上的随机偏置张量
  const auto bias_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的偏置张量转换为 Vulkan 张量
  const auto bias_vulkan = bias_cpu.vulkan();

  // 生成 CPU 上的随机 running_mean 张量
  const auto running_mean_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的 running_mean 张量转换为 Vulkan 张量
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  // 生成 CPU 上的随机 running_var 张量
  const auto running_var_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的 running_var 张量转换为 Vulkan 张量
  const auto running_var_vulkan = running_var_cpu.vulkan();

  // 使用输入张量、权重张量、偏置张量、running_mean 张量和 running_var 张量进行批归一化操作，并生成 CPU 上的输出张量
  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  // 使用 Vulkan 张量版本的输入、权重、偏置、running_mean 和 running_var 进行批归一化操作，并生成 Vulkan 张量版本的输出
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  // 检查 Vulkan 张量版本的输出与 CPU 张量版本的输出是否几乎相等
  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  // 断言检查通过，确保 Vulkan 张量版本的输出与 CPU 张量版本的输出几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, batch_norm_medium) {
  c10::InferenceMode mode;  // 进入推断模式

  // 生成 CPU 上的随机输入张量
  const auto input_cpu = at::rand({3, 8, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto input_vulkan = input_cpu.vulkan();

  // 生成 CPU 上的随机权重张量
  const auto weight_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的权重张量转换为 Vulkan 张量
  const auto weight_vulkan = weight_cpu.vulkan();

  // 生成 CPU 上的随机偏置张量
  const auto bias_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的偏置张量转换为 Vulkan 张量
  const auto bias_vulkan = bias_cpu.vulkan();

  // 生成 CPU 上的随机 running_mean 张量
  const auto running_mean_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的 running_mean 张量转换为 Vulkan 张量
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  // 生成 CPU 上的随机 running_var 张量
  const auto running_var_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的 running_var 张量转换为 Vulkan 张量
  const auto running_var_vulkan = running_var_cpu.vulkan();

  // 使用输入张量、权重张量、偏置张量、running_mean 张量和 running_var 张量进行批归一化操作，并生成 CPU 上的输出张量
  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  // 使用 Vulkan 张量版本的输入、权重、偏置、running_mean 和 running_var 进行批归一化操作，并生成 Vulkan 张量版本的输出
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  // 检查 Vulkan 张量版本的输出与 CPU 张量版本的输出是否几乎相等
  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  // 断言检查通过，确保 Vulkan 张量版本的输出与 CPU 张量版本的输出几乎相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, batch_norm_large) {
  c10::InferenceMode mode;

  // 生成一个指定大小的随机张量作为 CPU 上的输入数据
  const auto input_cpu = at::rand({11, 52, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  // 将输入数据转换为 Vulkan 张量
  const auto input_vulkan = input_cpu.vulkan();

  // 生成一个指定大小的随机张量作为 CPU 上的权重
  const auto weight_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  // 将权重转换为 Vulkan 张量
  const auto weight_vulkan = weight_cpu.vulkan();

  // 生成一个指定大小的随机张量作为 CPU 上的偏置
  const auto bias_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  // 将偏置转换为 Vulkan 张量
  const auto bias_vulkan = bias_cpu.vulkan();

  // 生成一个指定大小的随机张量作为 CPU 上的 running_mean
  const auto running_mean_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 running_mean 转换为 Vulkan 张量
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  // 生成一个指定大小的随机张量作为 CPU 上的 running_var
  const auto running_var_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 running_var 转换为 Vulkan 张量
  const auto running_var_vulkan = running_var_cpu.vulkan();

  // 调用 batch_norm 函数计算 CPU 上的输出
  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  // 调用 batch_norm 函数计算 Vulkan 张量上的输出
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  // 检查 CPU 输出和 Vulkan 输出的准确性
  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  // 如果结果不准确，展示相对误差信息
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  // 断言 CPU 输出和 Vulkan 输出的一致性
  ASSERT_TRUE(check);
}

void test_baddbmm(
    at::Tensor bias_cpu,
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    float beta,
    float alpha) {
  // 调用 baddbmm 函数计算结果
  const auto out_cpu = at::baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();
  // 调用 baddbmm 函数计算 Vulkan 张量上的结果
  const auto out_vulkan =
      at::baddbmm(bias_cpu, m1_vulkan, m2_cpu.vulkan(), beta, alpha);

  // 检查 CPU 输出和 Vulkan 输出的准确性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不准确，展示相对误差信息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 CPU 输出和 Vulkan 输出的一致性
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, baddbmm) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  int batch = 9;
  int n = 10;
  int p = 41;
  int m = 13;

  // 生成指定大小的随机张量作为 CPU 上的偏置
  const auto bias_cpu =
      at::rand({batch, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成指定大小的随机张量作为 CPU 上的 m1
  const auto m1_cpu =
      at::rand({batch, n, p}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成指定大小的随机张量作为 CPU 上的 m2
  const auto m2_cpu =
      at::rand({batch, p, m}, at::device(at::kCPU).dtype(at::kFloat));

  // 调用测试函数 test_baddbmm 进行测试
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_small) {
  constexpr float alpha = -1.0f;
  constexpr float beta = 2.0f;
  int batch = 3;
  int n = 3;
  int p = 5;
  int m = 4;

  // 生成指定大小的随机张量作为 CPU 上的偏置的部分子张量
  const auto bias_cpu_0 =
      at::rand({1, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成指定大小的全1张量作为 CPU 上的偏置的部分子张量
  const auto bias_cpu_1 =
      at::ones({1, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成指定大小的随机张量作为 CPU 上的偏置的部分子张量，并乘以-1
  const auto bias_cpu_2 =
      at::rand({1, n, m}, at::device(at::kCPU).dtype(at::kFloat)) * -1;
  // 拼接上述三个部分张量，形成完整的偏置张量
  const auto bias_cpu = at::cat({bias_cpu_0, bias_cpu_1, bias_cpu_2}, 0);

  // 生成指定大小的随机张量作为 CPU 上的 m1
  const auto m1_cpu =
      at::rand({batch, n, p}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成指定大小的随机张量作为 CPU 上的 m2
  const auto m2_cpu =
      at::rand({batch, p, m}, at::device(at::kCPU).dtype(at::kFloat));

  // 调用测试函数 test_baddbmm 进行测试
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}
TEST_F(VulkanAPITest, baddbmm_one) {
  // 定义 alpha 和 beta 值，用于矩阵乘法中的参数
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // 创建随机张量 bias_cpu，表示偏置，形状为 1x1x1
  const auto bias_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m1_cpu，表示第一个乘数，形状为 1x1x1
  const auto m1_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m2_cpu，表示第二个乘数，形状为 1x1x1
  const auto m2_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));

  // 调用测试函数 test_baddbmm，对 baddbmm 操作进行测试
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bais_error) {
  // 定义 alpha 和 beta 值，用于矩阵乘法中的参数
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // 创建随机张量 bias_cpu，表示偏置，形状为 200x179x163
  const auto bias_cpu =
      at::rand({200, 179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m1_cpu，形状为 150x179x67
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m2_cpu，形状为 150x67x163
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();
  // 预期调用 baddbmm 时会抛出异常，因为 batch size 维度不匹配
  EXPECT_THROW(
      at::baddbmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha), ::std::exception);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch) {
  // 定义 alpha 和 beta 值，用于矩阵乘法中的参数
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 创建随机张量 bias_cpu，表示偏置，形状为 1x179x163
  const auto bias_cpu =
      at::rand({1, 179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m1_cpu，形状为 150x179x67
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m2_cpu，形状为 150x67x163
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，对 baddbmm 操作进行测试
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_height) {
  // 定义 alpha 和 beta 值，用于矩阵乘法中的参数
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 创建随机张量 bias_cpu，表示偏置，形状为 150x1x163
  const auto bias_cpu =
      at::rand({150, 1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m1_cpu，形状为 150x179x67
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m2_cpu，形状为 150x67x163
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，对 baddbmm 操作进行测试
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_width) {
  // 定义 alpha 和 beta 值，用于矩阵乘法中的参数
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 创建随机张量 bias_cpu，表示偏置，形状为 150x179x1
  const auto bias_cpu =
      at::rand({150, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m1_cpu，形状为 150x179x67
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m2_cpu，形状为 150x67x163
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，对 baddbmm 操作进行测试
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch_width) {
  // 定义 alpha 和 beta 值，用于矩阵乘法中的参数
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 创建随机张量 bias_cpu，表示偏置，形状为 1x179x1
  const auto bias_cpu =
      at::rand({1, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m1_cpu，形状为 150x179x67
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机张量 m2_cpu，形状为 150x67x163
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，对 baddbmm 操作进行测试
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}
TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch_height) {
  // 定义常量 alpha 和 beta
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 生成形状为 {1, 1, 163} 的随机张量 bias_cpu，存储在 CPU 上，数据类型为 float
  const auto bias_cpu =
      at::rand({1, 1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 179, 67} 的随机张量 m1_cpu，存储在 CPU 上，数据类型为 float
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 67, 163} 的随机张量 m2_cpu，存储在 CPU 上，数据类型为 float
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，传递生成的随机张量和常量 alpha、beta 作为参数
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_one) {
  // 定义常量 alpha 和 beta
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 生成形状为 {1, 1, 1} 的随机张量 bias_cpu，存储在 CPU 上，数据类型为 float
  const auto bias_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 179, 67} 的随机张量 m1_cpu，存储在 CPU 上，数据类型为 float
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 67, 163} 的随机张量 m2_cpu，存储在 CPU 上，数据类型为 float
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，传递生成的随机张量和常量 alpha、beta 作为参数
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch) {
  // 定义常量 alpha 和 beta
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 生成形状为 {179, 163} 的随机张量 bias_cpu，存储在 CPU 上，数据类型为 float
  const auto bias_cpu =
      at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 179, 67} 的随机张量 m1_cpu，存储在 CPU 上，数据类型为 float
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 67, 163} 的随机张量 m2_cpu，存储在 CPU 上，数据类型为 float
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，传递生成的随机张量和常量 alpha、beta 作为参数
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch1) {
  // 定义常量 alpha 和 beta
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 生成形状为 {179, 1} 的随机张量 bias_cpu，存储在 CPU 上，数据类型为 float
  const auto bias_cpu =
      at::rand({179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 179, 67} 的随机张量 m1_cpu，存储在 CPU 上，数据类型为 float
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 67, 163} 的随机张量 m2_cpu，存储在 CPU 上，数据类型为 float
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，传递生成的随机张量和常量 alpha、beta 作为参数
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch2) {
  // 定义常量 alpha 和 beta
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 生成形状为 {1, 163} 的随机张量 bias_cpu，存储在 CPU 上，数据类型为 float
  const auto bias_cpu =
      at::rand({1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 179, 67} 的随机张量 m1_cpu，存储在 CPU 上，数据类型为 float
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 67, 163} 的随机张量 m2_cpu，存储在 CPU 上，数据类型为 float
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，传递生成的随机张量和常量 alpha、beta 作为参数
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch_height) {
  // 定义常量 alpha 和 beta
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 生成形状为 {163} 的随机张量 bias_cpu，存储在 CPU 上，数据类型为 float
  const auto bias_cpu = at::rand({163}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 179, 67} 的随机张量 m1_cpu，存储在 CPU 上，数据类型为 float
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成形状为 {150, 67, 163} 的随机张量 m2_cpu，存储在 CPU 上，数据类型为 float
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用测试函数 test_baddbmm，传递生成的随机张量和常量 alpha、beta 作为参数
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}
TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_all) {
  // 定义常量 alpha 和 beta
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  // 生成一个随机的大小为 {1} 的张量作为偏置，使用 CPU 设备和 float 类型
  const auto bias_cpu = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成一个随机的大小为 {150, 179, 67} 的张量作为 m1_cpu，使用 CPU 设备和 float 类型
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成一个随机的大小为 {150, 67, 163} 的张量作为 m2_cpu，使用 CPU 设备和 float 类型
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用 test_baddbmm 函数，传入偏置、m1_cpu、m2_cpu、beta 和 alpha
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

void test_matmul(
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    bool m2_use_vulkan = false) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 在 CPU 上计算 m1_cpu 和 m2_cpu 的矩阵乘法，结果保存在 out_cpu 中
  const auto out_cpu = at::matmul(m1_cpu, m2_cpu);
  // 将 m1_cpu 和 m2_cpu 转换为 Vulkan 张量后进行矩阵乘法计算，结果保存在 out_vk 中
  auto out_vk =
      at::matmul(m1_cpu.vulkan(), m2_use_vulkan ? m2_cpu.vulkan() : m2_cpu);

  // 检查 out_cpu 和 out_vk 是否近似相等
  const auto check = almostEqual(out_cpu, out_vk.cpu());
  // 如果检查不通过，显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vk.cpu());
  }

  // 断言 out_cpu 和 out_vk 近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_matmul_3d_weight_vulkan) {
  // 这将调用 at::bmm。因未知原因可能导致崩溃
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用 test_matmul 函数，传入 m1_cpu、m2_cpu 和 true（使用 Vulkan 加速）
  test_matmul(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, DISABLED_matmul_3d_weight_cpu) {
  // 这将调用 at::bmm。因未知原因可能导致崩溃
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用 test_matmul 函数，传入 m1_cpu 和 m2_cpu（使用 CPU 计算）
  test_matmul(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, matmul_2d_weight_vulkan) {
  // 这将调用 at::mm
  const auto m1_cpu = at::rand({7, 42}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({42, 9}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用 test_matmul 函数，传入 m1_cpu、m2_cpu 和 true（使用 Vulkan 加速）
  test_matmul(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, matmul_2d_weight_cpu) {
  // 这将调用 at::mm
  const auto m1_cpu =
      at::rand({23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用 test_matmul 函数，传入 m1_cpu 和 m2_cpu（使用 CPU 计算）
  test_matmul(m1_cpu, m2_cpu);
}

void test_bmm(
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    bool m2_use_vulkan = false) {
  // 计算 m1_cpu 和 m2_cpu 的批次矩阵乘法，结果保存在 out_cpu 中
  const auto out_cpu = m1_cpu.bmm(m2_cpu);

  // 将 m1_cpu 转换为 Vulkan 张量后进行批次矩阵乘法计算，结果保存在 out_vulkan 中
  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan =
      m1_vulkan.bmm(m2_use_vulkan ? m2_cpu.vulkan() : m2_cpu);

  // 检查 out_cpu 和 out_vulkan 是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 out_cpu 和 out_vulkan 近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, bmm_vulkan_small) {
  // 生成两个大小为 {5, 2, 3} 和 {5, 3, 4} 的随机张量 m1_cpu 和 m2_cpu，使用 CPU 设备和 float 类型
  const auto m1_cpu =
      at::rand({5, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({5, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用 test_bmm 函数，传入 m1_cpu、m2_cpu 和 true（使用 Vulkan 加速）
  test_bmm(m1_cpu, m2_cpu, true);
}
TEST`
TEST_F(VulkanAPITest, bmm_vulkan_small_width) {
  // Generate random tensor m1_cpu with shape [9, 32, 5] on CPU
  const auto m1_cpu = at::rand({9, 32, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // Generate random tensor m2_cpu with shape [9, 5, 13] on CPU
  const auto m2_cpu = at::rand({9, 5, 13}, at::device(at::kCPU).dtype(at::kFloat));
  // Perform batch matrix multiplication using Vulkan API
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_vulkan_large_width) {
  // Generate random tensor m1_cpu with shape [9, 7, 45] on CPU
  const auto m1_cpu = at::rand({9, 7, 45}, at::device(at::kCPU).dtype(at::kFloat));
  // Generate random tensor m2_cpu with shape [9, 45, 6] on CPU
  const auto m2_cpu = at::rand({9, 45, 6}, at::device(at::kCPU).dtype(at::kFloat));
  // Perform batch matrix multiplication using Vulkan API
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_cpu) {
  // Generate random tensor m1_cpu with shape [13, 23, 45] on CPU
  const auto m1_cpu = at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  // Generate random tensor m2_cpu with shape [13, 45, 26] on CPU
  const auto m2_cpu = at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  // Perform batch matrix multiplication on CPU
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_small) {
  // Generate random tensor m1_cpu with shape [2, 6, 5] on CPU
  const auto m1_cpu = at::rand({2, 6, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // Generate random tensor m2_cpu with shape [2, 5, 3] on CPU
  const auto m2_cpu = at::rand({2, 5, 3}, at::device(at::kCPU).dtype(at::kFloat));
  // Perform batch matrix multiplication on CPU
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_one) {
  // Generate random tensor m1_cpu with shape [1, 1, 1] on CPU
  const auto m1_cpu = at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // Generate random tensor m2_cpu with shape [1, 1, 1] on CPU
  const auto m2_cpu = at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // Perform batch matrix multiplication on CPU
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_error) {
  // Generate large random tensors m1_cpu and m2_cpu with mismatched batch dimensions on CPU
  const auto m1_cpu = at::rand({100, 235, 546}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({200, 546, 267}, at::device(at::kCPU).dtype(at::kFloat));
  // Convert m1_cpu to Vulkan tensor
  const auto m1_vulkan = m1_cpu.vulkan();
  // Expect an exception when performing batch matrix multiplication on Vulkan tensors
  EXPECT_THROW(m1_vulkan.bmm(m2_cpu), ::std::exception);
}

TEST_F(VulkanAPITest, clamp) {
  // Generate random tensor in_cpu with shape [17, 197, 302, 5] on CPU
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // Convert in_cpu to Vulkan tensor
  const auto in_vulkan = in_cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  // Apply element-wise clamping operation on CPU tensors
  const auto out_cpu = at::clamp(in_cpu, min_value, max_value);
  // Apply element-wise clamping operation on Vulkan tensors
  const auto out_vulkan = at::clamp(in_vulkan, min_value, max_value);

  // Check if the results are almost equal
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // Assert that the results are almost equal
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, clamp_) {
  // Generate random tensor cpu with shape [17, 197, 302, 5] on CPU
  const auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // Convert cpu to Vulkan tensor
  const auto vulkan = cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  // In-place element-wise clamping operation on CPU tensor
  cpu.clamp_(min_value, max_value);
  // In-place element-wise clamping operation on Vulkan tensor
  vulkan.clamp_(min_value, max_value);

  // Check if the results are almost equal
  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // Assert that the results are almost equal
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, conv1d_simple) {
  // 定义卷积核大小、通道数和长度
  int64_t kernel_size = 3;
  int64_t channels = 5;
  int64_t lengths = 9;

  // 进入推理模式
  c10::InferenceMode mode;

  // 创建CPU端的输入张量，包括使用arange生成的输入数据、全为1的权重和使用arange生成的偏置
  const auto input_cpu = at::arange(lengths * channels, at::kFloat).reshape({1, channels, lengths});
  const auto weights_cpu = at::ones({channels, 1, kernel_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::arange(channels, at::kFloat);

  // 将CPU端的输入张量转换为Vulkan端的张量
  const auto input_vk = input_cpu.vulkan();
  const auto weights_vk = weights_cpu.vulkan();
  const auto bias_vk = bias_cpu.vulkan();

  // 设置卷积的步长、填充和扩展率
  int64_t stride = 1;
  int64_t padding = 0;
  int64_t dilation = 1;

  // 使用conv1d函数在CPU端进行卷积操作，并得到输出
  const auto output_cpu = at::conv1d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, channels);

  // 使用conv1d函数在Vulkan端进行卷积操作，并得到输出
  const auto output_vk = at::conv1d(
      input_vk, weights_vk, bias_vk, stride, padding, dilation, channels);
  // 将Vulkan端输出转换回CPU端
  const auto output_vk_cpu = output_vk.cpu();

  // 检查CPU端输出与Vulkan端输出的近似相等性
  const bool check = almostEqual(output_cpu, output_vk_cpu);
  // 如果不满足近似相等性，则展示它们之间的相对误差
  if (!check) {
    showRtol(output_cpu, output_vk_cpu);
  }

  // 断言近似相等性，确保测试通过
  ASSERT_TRUE(check);
}

void test_conv1d(
    int64_t kernel_size,
    int64_t groups,
    int64_t lengths,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t in_group_size = 1,
    int64_t out_group_size = 1,
    int64_t batch_size = 1) {
  // 进入推理模式
  c10::InferenceMode mode;

  // 计算输入通道数和输出通道数
  int64_t in_channels = in_group_size * groups;
  int64_t out_channels = out_group_size * groups;

  // 创建随机数填充的CPU端输入张量、权重和偏置
  const auto input_cpu = at::rand({batch_size, in_channels, lengths}, at::kFloat);
  const auto weights_cpu = at::rand({out_channels, in_group_size, kernel_size}, at::kFloat);
  const auto bias_cpu = at::rand({out_channels,}, at::kFloat);

  // 将CPU端的输入张量转换为Vulkan端的张量
  const auto input_vk = input_cpu.vulkan();
  const auto weights_vk = weights_cpu.vulkan();
  const auto bias_vk = bias_cpu.vulkan();

  // 使用conv1d函数在CPU端进行卷积操作，并得到输出
  const auto output_cpu = at::conv1d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  // 使用conv1d函数在Vulkan端进行卷积操作，并得到输出
  const auto output_vk = at::conv1d(
      input_vk, weights_vk, bias_vk, stride, padding, dilation, groups);
  // 将Vulkan端输出转换回CPU端
  const auto output_vk_cpu = output_vk.cpu();

  // 检查CPU端输出与Vulkan端输出的近似相等性
  const bool check = almostEqual(output_cpu, output_vk_cpu);
  // 如果不满足近似相等性，则展示它们之间的相对误差
  if (!check) {
    showRtol(output_cpu, output_vk_cpu);
  }

  // 断言近似相等性，确保测试通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv1d) {
  // 测试不同参数配置下的conv1d函数
  test_conv1d(3, 5, 8);
  test_conv1d(9, 5, 9);
  test_conv1d(1, 12, 3);
  test_conv1d(1, 12, 1);
  test_conv1d(10, 12, 20);
  test_conv1d(3, 5, 9, 2, 0, 1);
  test_conv1d(3, 5, 9, 2, 1, 1);
  test_conv1d(3, 5, 9, 2, 1, 2);
  test_conv1d(3, 5, 9, 1, 4, 2);
  test_conv1d(6, 22, 30, 5, 5, 3);
  test_conv1d(6, 5, 30, 5, 5, 3, 3, 5);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 2);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 5);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 9);
}
  // 进入推断模式
  c10::InferenceMode mode;

  // 创建随机输入张量，形状为 input_shape，数据类型为 float，存储于 CPU 上
  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机权重张量，形状为 weight_shape，数据类型为 float，存储于 CPU 上
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 创建随机偏置张量，形状为 bias_shape，数据类型为 float，存储于 CPU 上
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行卷积操作，输出为 out_cpu
  const auto out_cpu = at::conv2d(
    input, weight, bias, stride, padding, dilation, groups);

  // 在 Vulkan 上预先打包卷积参数和张量
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::create_conv2d_context",
      "",
      weight, bias, stride, padding, dilation, groups, c10::nullopt, c10::nullopt);

  // 在 Vulkan 上执行卷积操作，输入为 input 在 Vulkan 上的表示，输出为 vulkan_output
  const auto vulkan_output = callOpByName(
      "vulkan_prepack::run_conv2d_context",
      "",
      input.vulkan(), prepack_vulkan[0]);

  // 从 Vulkan 输出中获取张量，并将其移回 CPU
  const auto out_vulkan = vulkan_output[0].toTensor();
  const auto out_vk_cpu = out_vulkan.cpu();

  // 检查 Vulkan 输出与 CPU 输出的近似相等性
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  // 如果检查失败，展示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 测试向后兼容的 conv2d 上下文
void test_backwards_compatible_conv2d_context(
    const at::IntArrayRef input_shape,        // 输入张量的形状
    const at::IntArrayRef weight_shape,       // 权重张量的形状
    const at::IntArrayRef bias_shape,         // 偏置张量的形状
    std::vector<int64_t> stride,              // 卷积的步长
    std::vector<int64_t> padding,             // 卷积的填充
    std::vector<int64_t> dilation,            // 卷积的扩展
    int64_t groups) {                         // 分组卷积数

  c10::InferenceMode mode;  // 进入推断模式

  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));  // 创建随机输入张量
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));  // 创建随机权重张量
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));  // 创建随机偏置张量

  // 在 CPU 上进行卷积计算
  const auto out_cpu = at::conv2d(
    input, weight, bias, stride, padding, dilation, groups);

  // 在 Vulkan 上执行预打包和卷积运算
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::conv2d_clamp_prepack",
      "",
      weight, bias, stride, padding, dilation, groups, c10::nullopt, c10::nullopt);

  const auto vulkan_output = callOpByName(
      "vulkan_prepack::conv2d_clamp_run",
      "",
      input.vulkan(), prepack_vulkan[0]);

  const auto out_vulkan = vulkan_output[0].toTensor();
  const auto out_vk_cpu = out_vulkan.cpu();

  // 检查 Vulkan 计算结果与 CPU 计算结果的近似相等性
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);  // 显示两个张量之间的相对误差
  }

  ASSERT_TRUE(check);  // 断言检查结果为真
}

// 测试转置 conv2d 上下文
void test_transposed_conv2d_context(
    const at::IntArrayRef input_shape,        // 输入张量的形状
    const at::IntArrayRef weight_shape,       // 权重张量的形状
    const at::IntArrayRef bias_shape,         // 偏置张量的形状
    std::vector<int64_t> stride,              // 转置卷积的步长
    std::vector<int64_t> padding,             // 转置卷积的填充
    std::vector<int64_t> output_padding,      // 转置卷积的输出填充
    std::vector<int64_t> dilation,            // 转置卷积的扩展
    int64_t groups) {                         // 分组卷积数

  c10::InferenceMode mode;  // 进入推断模式

  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));  // 创建随机输入张量
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));  // 创建随机权重张量
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));  // 创建随机偏置张量

  // 在 CPU 上进行转置卷积计算
  const auto out_cpu = at::conv_transpose2d(
    input, weight, bias, stride, padding, output_padding, groups, dilation);

  // 在 Vulkan 上执行预打包和转置卷积运算
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::create_tconv2d_context",
      "",
      weight, bias, stride, padding, output_padding, dilation, groups, c10::nullopt, c10::nullopt);

  const auto vulkan_output = callOpByName(
      "vulkan_prepack::run_tconv2d_context",
      "",
      input.vulkan(), prepack_vulkan[0]);

  const auto out_vulkan = vulkan_output[0].toTensor();
  const auto out_vk_cpu = out_vulkan.cpu();

  // 检查 Vulkan 计算结果与 CPU 计算结果的近似相等性
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);  // 显示两个张量之间的相对误差
  }

  ASSERT_TRUE(check);  // 断言检查结果为真
}

// Vulkan API 测试类中的 conv2d 测试
TEST_F(VulkanAPITest, conv2d) {
  constexpr int64_t groups = 1;  // 卷积分组数为 1
  constexpr std::array<int64_t, 2u> stride{2, 2};  // 步长为 2x2
  constexpr std::array<int64_t, 2u> padding{1, 1};  // 填充为 1x1
  //TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};  // 考虑扩展卷积，目前设置为 1x1

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;
    // 定义一个结构体，表示输入张量的大小，包含批次数、通道数、宽度和高度
    constexpr struct {
      uint32_t batches;     // 输入张量的批次数
      uint32_t channels;    // 输入张量的通道数
      uint32_t width;       // 输入张量的宽度
      uint32_t height;      // 输入张量的高度
    
      // 返回输入张量的大小作为数组，顺序为批次数、通道数、宽度、高度
      std::array<int64_t, 4u> size() const {
        return {
          batches,
          channels,
          width,
          height,
        };
      }
    } input {1, 3, 8, 8};  // 初始化输入张量的大小为 {1, 3, 8, 8}
    
    // 定义一个结构体，表示卷积层的权重，包含输出通道数、输入通道数、卷积核宽度和高度
    constexpr struct {
      uint32_t output_channels;   // 卷积层输出的通道数
      uint32_t input_channels;    // 卷积层输入的通道数，使用输入张量的通道数
      uint32_t width;             // 卷积核的宽度
      uint32_t height;            // 卷积核的高度
    
      // 返回卷积层权重的大小作为数组，顺序为输出通道数、输入通道数、宽度、高度
      std::array<int64_t, 4u> size() const {
        return {
          output_channels,
          input_channels,
          width,
          height,
        };
      }
    } weights {1, input.channels, 3, 3};  // 初始化卷积层的权重
    
    // 在 CPU 上生成随机输入张量，大小由 input 的 size() 方法确定，数据类型为 float
    const auto input_cpu = at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
    
    // 在 CPU 上生成随机卷积核权重，大小由 weights 的 size() 方法确定，数据类型为 float
    const auto weights_cpu = at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
    
    // 在 CPU 上生成随机偏置项，大小为 weights 的 output_channels，数据类型为 float
    const auto bias_cpu = at::randn({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));
    
    // 使用输入张量、权重、偏置项以及给定的步长、填充、扩展率和组数，在 CPU 上进行二维卷积运算
    const auto output_cpu = at::conv2d(
        input_cpu,
        weights_cpu,
        bias_cpu,
        stride,
        padding,
        dilation,
        groups);
    
    // 将输入张量转换为 Vulkan 格式后，在 CPU 上进行二维卷积运算，并将结果返回 CPU
    const auto output_vulkan = at::conv2d(
        input_cpu.vulkan(),
        weights_cpu,
        bias_cpu,
        stride,
        padding,
        dilation,
        groups).cpu();
    
    // 检查两种卷积运算的输出结果是否几乎相等
    const bool check = almostEqual(output_cpu, output_vulkan);
    if (!check) {
      // 如果结果不几乎相等，展示两种输出的相对误差
      showRtol(output_cpu, output_vulkan);
    }
    
    // 断言两种卷积运算的输出结果几乎相等，如果不是则会触发断言失败
    ASSERT_TRUE(check);
TEST_F(VulkanAPITest, conv2d_dw_5x5) {
  // 定义卷积操作的分组数
  constexpr int64_t groups = 7;
  // 定义步幅数组
  constexpr std::array<int64_t, 2u> stride{2, 3};
  // 定义填充数组
  constexpr std::array<int64_t, 2u> padding{0, 4};
  // 定义扩张数组
  constexpr std::array<int64_t, 2u> dilation{3, 1};

  // 定义输入数据结构体
  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    // 返回数据大小数组
    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{1, groups, 137, 199};

  // 定义权重数据结构体
  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    // 返回数据大小数组
    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{groups, 1, 3, 3};

  // 生成随机输入数据张量，存储在 CPU 上
  const auto input_cpu =
      at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  // 生成随机权重数据张量，存储在 CPU 上
  const auto weights_cpu =
      at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  // 生成随机偏置数据张量，存储在 CPU 上
  const auto bias_cpu = at::rand(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  // 使用 CPU 上的数据进行卷积计算，存储在 output_cpu 中
  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  // 使用 Vulkan API 加速的数据进行卷积计算，存储在 output_vulkan 中
  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  // 检查 output_cpu 和 output_vulkan 的近似相等性
  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  // 如果检查失败，展示两者的相对容差
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}
  }
} weights{groups, 1, 5, 5};

const auto input_cpu =
    at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
const auto weights_cpu =
    at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
const auto bias_cpu = at::rand(
    {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

const auto output_cpu = at::conv2d(
    input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

const auto output_vulkan = at::conv2d(
    input_cpu.vulkan(),
    weights_cpu,
    bias_cpu,
    stride,
    padding,
    dilation,
    groups);

const bool check = almostEqual(output_cpu, output_vulkan.cpu());
if (!check) {
  showRtol(output_cpu, output_vulkan.cpu());
}

ASSERT_TRUE(check);



} weights{groups, 1, 5, 5};



// 定义卷积层的权重，具有指定的维度和分组数
const auto input_cpu =
    at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
// 生成指定大小的随机输入张量在 CPU 上
const auto weights_cpu =
    at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
// 生成指定大小的随机权重张量在 CPU 上
const auto bias_cpu = at::rand(
    {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));
// 生成指定大小的随机偏置张量在 CPU 上

const auto output_cpu = at::conv2d(
    input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);
// 使用 CPU 上的张量进行卷积操作，得到输出张量

const auto output_vulkan = at::conv2d(
    input_cpu.vulkan(),
    weights_cpu,
    bias_cpu,
    stride,
    padding,
    dilation,
    groups);
// 使用 Vulkan 加速的输入张量进行卷积操作，得到 Vulkan 加速后的输出张量

const bool check = almostEqual(output_cpu, output_vulkan.cpu());
// 检查两个输出张量是否近似相等

if (!check) {
  showRtol(output_cpu, output_vulkan.cpu());
  // 如果输出不相等，显示它们的相对误差
}

ASSERT_TRUE(check);
// 断言两个输出张量应当相等
}

TEST_F(VulkanAPITest, conv2d_dw) {
  // 定义深度可分离卷积的参数
  constexpr int64_t groups = 7;  // 卷积组数
  constexpr std::array<int64_t, 2u> stride{2, 3};  // 步幅数组
  constexpr std::array<int64_t, 2u> padding{0, 4};  // 填充数组
  constexpr std::array<int64_t, 2u> dilation{3, 1};  // 膨胀数组

  // 定义输入张量的结构体
  constexpr struct {
    uint32_t batches;  // 批次大小
    uint32_t channels;  // 通道数
    uint32_t width;  // 宽度
    uint32_t height;  // 高度

    std::array<int64_t, 4u> size() const {  // 返回张量尺寸的数组
      return {
        batches,
        channels,
        width,
        height,
      };
    }
  } input {1, groups, 137, 199};  // 输入张量的具体参数

  // 定义权重张量的结构体
  constexpr struct {
    uint32_t output_channels;  // 输出通道数
    uint32_t input_channels;  // 输入通道数
    uint32_t width;  // 宽度
    uint32_t height;  // 高度

    std::array<int64_t, 4u> size() const {  // 返回张量尺寸的数组
      return {
        output_channels,
        input_channels,
        width,
        height,
      };
    }
  } weights {groups, 1, 17, 7};  // 权重张量的具体参数

  // 在 CPU 上生成随机输入张量、权重张量和偏置张量
  const auto input_cpu = at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::rand({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  // 使用 CPU 上的输入张量、权重张量和偏置张量进行深度可分离卷积计算
  const auto output_cpu = at::conv2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  // 使用 Vulkan API 在输入张量上执行深度可分离卷积计算
  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  // 检查 CPU 计算结果与 Vulkan 计算结果是否接近
  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  // 如果结果不接近，则展示相对误差
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  // 断言 CPU 计算结果与 Vulkan 计算结果接近
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_dw_prepack) {
  // 使用预包装功能测试深度可分离卷积的上下文
  test_conv2d_context(
    {1, 7, 137, 199}, // input_shape 输入张量的形状
    {7, 1, 17, 7},    // weight_shape 权重张量的形状
    {7},              // bias_shape 偏置张量的形状
    {2, 3},           // stride 步幅
    {0, 4},           // padding 填充
    {3, 1},           // dilation 膨胀
    7);               // groups 卷积组数
}

TEST_F(VulkanAPITest, conv2d_dw_prepack_bc) {
  // 使用向后兼容的功能测试深度可分离卷积的上下文
  test_backwards_compatible_conv2d_context(
    {1, 7, 137, 199}, // input_shape 输入张量的形状
    {7, 1, 17, 7},    // weight_shape 权重张量的形状
    {7},              // bias_shape 偏置张量的形状
    {2, 3},           // stride 步幅
    {0, 4},           // padding 填充
    {3, 1},           // dilation 膨胀
    7);               // groups 卷积组数
}

TEST_F(VulkanAPITest, conv2d_pw) {
  // 定义普通卷积的参数
  constexpr int64_t groups = 1;  // 卷积组数
  constexpr std::array<int64_t, 2u> stride{1, 1};  // 步幅数组
  constexpr std::array<int64_t, 2u> padding{0, 0};  // 填充数组
  constexpr std::array<int64_t, 2u> dilation{1, 1};  // 膨胀数组

  // 定义输入张量的结构体
  constexpr struct {
    uint32_t batches;  // 批次大小
    uint32_t channels;  // 通道数
    uint32_t width;  // 宽度
    uint32_t height;  // 高度

    std::array<int64_t, 4u> size() const {  // 返回张量尺寸的数组
      return {
        batches,
        channels,
        width,
        height,
      };
    }
  } input {1, 17, 127, 397};  // 输入张量的具体参数

  // 定义权重张量的结构体
  constexpr struct {
    uint32_t output_channels;  // 输出通道数
    uint32_t input_channels;  // 输入通道数
    uint32_t width;  // 宽度
    uint32_t height;  // 高度

    std::array<int64_t, 4u> size() const {  // 返回张量尺寸的数组
      return {
        output_channels,
        input_channels,
        width,
        height,
      };
    }
  } weights {17, 1, 17, 7};  // 权重张量的具体参数
``
  }
} weights {29, input.channels, 1, 1};

// 在 CPU 上生成一个与输入形状相同的随机张量
const auto input_cpu = at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
// 在 CPU 上生成一个与权重形状相同的随机张量
const auto weights_cpu = at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
// 在 CPU 上生成一个与偏置形状相同的随机张量
const auto bias_cpu = at::randn({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

// 使用 CPU 上的数据进行二维卷积操作，得到输出张量
const auto output_cpu = at::conv2d(
    input_cpu,
    weights_cpu,
    bias_cpu,
    stride,
    padding,
    dilation,
    groups);

// 将输入数据转换为 Vulkan 张量，并在 Vulkan 上执行相同的二维卷积操作，得到 Vulkan 输出张量
const auto output_vulkan = at::conv2d(
    input_cpu.vulkan(),
    weights_cpu,
    bias_cpu,
    stride,
    padding,
    dilation,
    groups);

// 检查 CPU 输出张量和 Vulkan 输出张量之间的近似相等性
const bool check = almostEqual(output_cpu, output_vulkan.cpu());
// 如果结果不近似相等，则显示它们的相对误差
if (!check) {
  showRtol(output_cpu, output_vulkan.cpu());
}

// 断言 CPU 输出张量和 Vulkan 输出张量近似相等
ASSERT_TRUE(check);
TEST_F(VulkanAPITest, conv2d_pw_prepack_medium) {
  // 设置输入通道数、输出通道数、高度、宽度
  int in_channels = 17;
  int out_channels = 29;
  int height = 27;
  int width = 39;
  // 调用测试函数，验证卷积操作的上下文
  test_conv2d_context(
    {1, in_channels, height, width},  // input_shape 输入形状
    {out_channels, in_channels, 1, 1},     // weight_shape 权重形状
    {out_channels},               // bias_shape 偏置形状
    {1, 1},             // stride 步幅
    {0, 0},             // padding 填充
    {1, 1},             // dilation 膨胀率
    1);                 // groups 分组数
}

TEST_F(VulkanAPITest, conv2d_pw_prepack_bc_medium) {
  // 设置输入通道数、输出通道数、高度、宽度
  int in_channels = 17;
  int out_channels = 29;
  int height = 27;
  int width = 39;
  // 调用测试函数，验证向后兼容的卷积操作的上下文
  test_backwards_compatible_conv2d_context(
    {1, in_channels, height, width},  // input_shape 输入形状
    {out_channels, in_channels, 1, 1},     // weight_shape 权重形状
    {out_channels},               // bias_shape 偏置形状
    {1, 1},             // stride 步幅
    {0, 0},             // padding 填充
    {1, 1},             // dilation 膨胀率
    1);                 // groups 分组数
}

// 下面的两个测试用例在 Meta 的 CI 中全部测试执行时失败，输出包含大量的 NaN。原因未知。
// 当单独运行这些测试时（使用 gtest_filter），测试通过。
// 在较小的平面上，如 "conv2d_pw_prepack_medium" 中所示，测试也是通过的。
TEST_F(VulkanAPITest, DISABLED_conv2d_pw_prepack) {
  // 调用测试函数，验证卷积操作的上下文
  test_conv2d_context(
    {1, 17, 127, 397},  // input_shape 输入形状
    {29, 17, 1, 1},     // weight_shape 权重形状
    {29},               // bias_shape 偏置形状
    {1, 1},             // stride 步幅
    {0, 0},             // padding 填充
    {1, 1},             // dilation 膨胀率
    1);                 // groups 分组数
}

TEST_F(VulkanAPITest, DISABLED_conv2d_pw_prepack_bc) {
  // 调用测试函数，验证向后兼容的卷积操作的上下文
  test_backwards_compatible_conv2d_context(
    {1, 17, 127, 397},  // input_shape 输入形状
    {29, 17, 1, 1},     // weight_shape 权重形状
    {29},               // bias_shape 偏置形状
    {1, 1},             // stride 步幅
    {0, 0},             // padding 填充
    {1, 1},             // dilation 膨胀率
    1);                 // groups 分组数
}

TEST_F(VulkanAPITest, conv2d_transposed) {
  // 准备阶段
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 2};
  constexpr std::array<int64_t, 2u> padding{1, 0};
  constexpr std::array<int64_t, 2u> output_padding{0, 1};
  // TODO: 支持 dilation != 1 的转置卷积操作
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  // 输入数据结构定义
  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t height;
    uint32_t width;

    // 返回输入数据的大小数组
    std::array<int64_t, 4u> size() const {
      return {
        batches,
        channels,
        height,
        width,
      };
    }
  } input {1, 55, 7, 19};

  // 权重数据结构定义
  constexpr struct {
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t height;
    uint32_t width;

    // 返回权重数据的大小数组
    std::array<int64_t, 4u> size() const {
      return {
        input_channels,
        output_channels,
        height,
        width,
      };
    }
  }
} weights {input.channels, 47, 2, 3};



// 定义神经网络层的权重张量，其形状由输入通道数、输出通道数、卷积核大小和数量决定
const auto input_cpu = at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
const auto weights_cpu = at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
const auto bias_cpu = at::zeros({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));



// 通过随机生成的张量数据，创建 CPU 设备上的输入、权重和偏置张量
// input_cpu: 输入张量
// weights_cpu: 权重张量
// bias_cpu: 偏置张量



// Act
const auto output_cpu = at::conv_transpose2d(
    input_cpu,
    weights_cpu,
    bias_cpu,
    stride,
    padding,
    output_padding,
    groups,
    dilation);



// 执行反卷积操作，并将结果存储在 output_cpu 中
// 使用 input_cpu、weights_cpu 和 bias_cpu 进行反卷积，参数包括步长 (stride)、填充 (padding)、输出填充 (output_padding)、分组数 (groups) 和扩张 (dilation)



const auto output_vk = at::conv_transpose2d(
    input_cpu.vulkan(),
    weights_cpu,
    bias_cpu,
    stride,
    padding,
    output_padding,
    groups,
    dilation).cpu();



// 在 Vulkan 设备上执行反卷积操作，然后将结果传输回 CPU，并存储在 output_vk 中
// 使用 Vulkan 设备上的 input_cpu 执行反卷积操作，其余参数与上一步相同，最后将结果传回 CPU



// Assert
const bool check = almostEqual(output_cpu, output_vk);
if (!check) {
  showRtol(output_cpu, output_vk);
}
ASSERT_TRUE(check);



// 断言检查 output_cpu 和 output_vk 是否几乎相等
// 如果二者不满足几乎相等条件，则显示它们的相对误差，并且断言失败
// 如果二者几乎相等，则断言成功
TEST_F(VulkanAPITest, conv2d_transposed_prepack) {
  // 在 Vulkan API 测试框架中，测试转置卷积的上下文
  test_transposed_conv2d_context(
    {1, 55, 7, 19}, // 输入张量的形状
    {55, 47, 2, 3}, // 权重张量的形状
    {47},           // 偏置张量的形状
    {1, 2},         // 步长
    {1, 0},         // 填充
    {0, 1},         // 输出填充
    {1, 1},         // 膨胀
    1);             // 分组数
}

TEST_F(VulkanAPITest, conv2d_clamp_after_div) {
  // 进入推理模式
  c10::InferenceMode mode;

  // 定义常量数组：步长、填充、膨胀和分组数
  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
  constexpr std::array<int64_t, 2u> dilation{1, 1};
  constexpr int64_t groups = 1;

  // 生成输入张量的分子和分母，并进行除法运算得到 CPU 上的输入张量和 Vulkan 上的输入张量
  const auto input_numerator = at::rand({1, 3, 64, 64}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_denominator = at::rand({3, 1, 1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto input_cpu = at::div(input_numerator, input_denominator);
  const auto input_vk = at::div(input_numerator.vulkan(), input_denominator.vulkan());

  // 生成随机的权重张量和偏置张量
  at::Tensor weight = at::rand({24, 3, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor bias = at::rand({24}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上调用预先打包的卷积操作，得到预打包结果
  const auto prepack_cpu = callOpByName(
      "prepacked::conv2d_clamp_prepack",
      "",
      weight, bias, stride, padding, dilation, groups, 0.0f, c10::nullopt)[0];

  // 在 CPU 上运行预打包的卷积操作，得到输出结果
  const auto out_cpu = callOpByName(
      "prepacked::conv2d_clamp_run",
      "",
      input_cpu, prepack_cpu)[0].toTensor();

  // 在 Vulkan 上调用预先打包的卷积操作，得到预打包结果
  const auto prepack_vk = callOpByName(
      "vulkan_prepack::create_conv2d_context",
      "",
      weight, bias, stride, padding, dilation, groups, 0.0f, c10::nullopt)[0];

  // 在 Vulkan 上运行预打包的卷积操作，得到输出结果
  const auto out_vk = callOpByName(
      "vulkan_prepack::run_conv2d_context",
      "",
      input_vk, prepack_vk)[0].toTensor();

  // 将 Vulkan 输出结果复制到 CPU 上
  const auto out_vk_cpu = out_vk.cpu();

  // 检查 CPU 和 Vulkan 输出结果之间的近似相等性
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, copy) {
  // 生成一个随机的 CPU 张量
  const auto cpu = at::rand({13, 17, 37, 19}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto vulkan = cpu.vulkan();

  // 检查 CPU 张量和其 Vulkan 转换版本之间的近似相等性
  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

void test_cumsum(const at::IntArrayRef input_shape, const int64_t dim) {
  // 生成具有给定形状的随机 CPU 张量
  const auto in_cpu = at::rand(input_shape, at::TensorOptions(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上进行累积和操作
  const auto out_cpu = at::cumsum(in_cpu, dim);
  // 在 Vulkan 上进行累积和操作，并将结果转换为 CPU 张量
  const auto out_vulkan = at::cumsum(in_cpu.vulkan(), dim);

  // 检查 CPU 和 Vulkan 输出结果之间的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cumsum_1d) {
  // 测试一维累积和，维度为0
  test_cumsum({37}, 0);
  // 测试一维累积和，维度为-1（最后一个维度）
  test_cumsum({37}, -1);
}

TEST_F(VulkanAPITest, cumsum_2d) {
  // 遍历不同的维度值进行二维累积和测试
  for (int64_t i = -1; i <= 1; i++) {
    test_cumsum({17, 37}, i);
  }
}

TEST_F(VulkanAPITest, cumsum_3d) {
  // 遍历不同的维度值进行三维累积和测试
  for (int64_t i = -2; i <= 2; i++) {
    test_cumsum({17, 37, 49}, i);
  }
}
TEST_F(VulkanAPITest, cumsum_4d) {
  // 对于给定的范围内的整数 i，调用 test_cumsum 函数，测试累加操作
  for (int64_t i = -3; i <= 3; i++) {
    test_cumsum({12, 17, 37, 49}, i);
  }
}

void test_div(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  // 创建具有指定形状和在 CPU 上随机浮点数值的张量 in_cpu
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 创建具有指定形状和在 CPU 上随机浮点数值的张量 other_cpu，并添加 0.01 偏移量
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;

  // 将 CPU 上的张量转换为 Vulkan 张量 in_vulkan
  const auto in_vulkan = in_cpu.vulkan();
  // 将 CPU 上的张量转换为 Vulkan 张量 other_vulkan
  const auto other_vulkan = other_cpu.vulkan();

  // 计算 CPU 上的张量的除法结果 out_cpu
  const auto out_cpu = at::div(in_cpu, other_cpu);
  // 计算 Vulkan 张量的除法结果 out_vulkan
  const auto out_vulkan = at::div(in_vulkan, other_vulkan);

  // 检查两种计算结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言结果近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div) {
  // 调用 test_div 函数，测试具有相同形状的除法操作
  test_div({11, 7, 139, 109}, {11, 7, 139, 109});
}

TEST_F(VulkanAPITest, div_broadcast0) {
  // 调用 test_div 函数，测试广播形状为 {3, 5, 1, 1} 和 {3, 5, 179, 221} 的除法操作
  test_div({3, 5, 1, 1}, {3, 5, 179, 221});
}

TEST_F(VulkanAPITest, div_broadcast1) {
  // 调用 test_div 函数，测试广播形状为 {3, 5, 179, 221} 和 {3, 5, 1, 221} 的除法操作
  test_div({3, 5, 179, 221}, {3, 5, 1, 221});
}

TEST_F(VulkanAPITest, div_broadcast2) {
  // 调用 test_div 函数，测试广播形状为 {3, 4, 179, 221} 和 {4, 1, 1} 的除法操作
  test_div({3, 4, 179, 221}, {4, 1, 1});
}

TEST_F(VulkanAPITest, div_broadcast3) {
  // 调用 test_div 函数，测试广播形状为 {3, 4, 179, 221} 和 {1, 1, 179, 221} 的除法操作
  test_div({3, 4, 179, 221}, {1, 1, 179, 221});
}

TEST_F(VulkanAPITest, div_broadcast4) {
  // 调用 test_div 函数，测试广播形状为 {3, 4, 41, 1} 和 {1, 41, 53} 的除法操作
  test_div({3, 4, 41, 1}, {1, 41, 53});
}

TEST_F(VulkanAPITest, div_broadcast5) {
  // 调用 test_div 函数，测试广播形状为 {2, 1, 7, 1} 和 {1, 5, 1, 4} 的除法操作
  test_div({2, 1, 7, 1}, {1, 5, 1, 4});
}

TEST_F(VulkanAPITest, div_broadcast6) {
  // 调用 test_div 函数，测试广播形状为 {1, 15, 5, 4} 和 {21, 1, 5, 4} 的除法操作
  test_div({1, 15, 5, 4}, {21, 1, 5, 4});
}

TEST_F(VulkanAPITest, div_zero_dim) {
  // 调用 test_div 函数，测试形状为 {1, 15, 5, 4} 和 {} 的除法操作（其中一个操作数为空张量）
  test_div({1, 15, 5, 4}, {});
}

TEST_F(VulkanAPITest, div_) {
  // 创建具有指定形状和在 CPU 上随机浮点数值的张量 a_cpu
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量 a_vulkan
  auto a_vulkan = a_cpu.vulkan();

  // 创建具有指定形状和在 CPU 上随机浮点数值的张量 b_cpu，并添加 0.01 偏移量
  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  // 将 CPU 上的张量转换为 Vulkan 张量 b_vulkan
  const auto b_vulkan = b_cpu.vulkan();

  // 在原地对 CPU 上的张量进行除法运算
  a_cpu.div_(b_cpu);
  // 在原地对 Vulkan 张量进行除法运算
  a_vulkan.div_(b_vulkan);

  // 检查两种计算结果的近似相等性
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  // 断言结果近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast0_) {
  // 创建具有指定形状和在 CPU 上随机浮点数值的张量 a_cpu
  auto a_cpu = at::rand({12, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量 a_vulkan
  auto a_vulkan = a_cpu.vulkan();

  // 创建具有指定形状和在 CPU 上随机浮点数值的张量 b_cpu，并添加 0.01 偏移量
  const auto b_cpu = at::rand({12, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  // 将 CPU 上的张量转换为 Vulkan 张量 b_vulkan
  const auto b_vulkan = b_cpu.vulkan();

  // 在原地对 CPU 上的张量进行除法运算
  a_cpu.div_(b_cpu);
  // 在原地对 Vulkan 张量进行除法运算
  a_vulkan.div_(b_vulkan);

  // 检查两种计算结果的近似相等性
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  // 断言结果近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast1_) {
  // 创建具有指定形状和在 CPU 上随机浮点数值的张量 a_cpu
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量 a_vulkan
  auto a_vulkan = a_cpu.vulkan();

  // 创建具有指定形状和在 CPU 上随机浮点数
TEST_F(VulkanAPITest, div_scalar) {

  // 在 CPU 上生成一个指定形状的随机张量
  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量移动到 Vulkan 设备上
  const auto a_vulkan = a_cpu.vulkan();

  // 定义一个标量
  const float b_scalar = 3.1415f;

  // 在 CPU 上对张量 a_cpu 和标量 b_scalar 进行除法操作
  const auto c_cpu = at::div(a_cpu, b_scalar);
  // 在 Vulkan 设备上对张量 a_vulkan 和标量 b_scalar 进行除法操作
  const auto c_vulkan = at::div(a_vulkan, b_scalar);

  // 检查两个结果张量是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果结果不几乎相等，则展示它们的相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言结果张量几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_scalar_) {
  // 在 CPU 上生成一个指定形状的随机张量
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量移动到 Vulkan 设备上
  auto a_vulkan = a_cpu.vulkan();

  // 定义一个标量
  const float b_scalar = 3.1415f;

  // 在 CPU 上对张量 a_cpu 进行标量除法操作
  a_cpu.div_(b_scalar);
  // 在 Vulkan 设备上对张量 a_vulkan 进行标量除法操作
  a_vulkan.div_(b_scalar);

  // 检查两个结果张量是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果结果不几乎相等，则展示它们的相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言结果张量几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_scalar_wrapped) {
  // 如果 Vulkan 不可用，则直接返回
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成一个指定形状的随机张量
  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量移动到 Vulkan 设备上
  const auto a_vulkan = a_cpu.vulkan();

  // 在 CPU 上生成一个随机标量，并稍微增加其值
  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;

  // 在 CPU 上对张量 a_cpu 和标量 b_scalar 进行除法操作
  const auto c_cpu = at::div(a_cpu, b_scalar);
  // 在 Vulkan 设备上对张量 a_vulkan 和标量 b_scalar 进行除法操作
  const auto c_vulkan = at::div(a_vulkan, b_scalar);

  // 检查两个结果张量是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果结果不几乎相等，则展示它们的相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言结果张量几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_scalar_wrapped_) {
  // 如果 Vulkan 不可用，则直接返回
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成一个指定形状的随机张量
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量移动到 Vulkan 设备上
  auto a_vulkan = a_cpu.vulkan();

  // 在 CPU 上生成一个随机标量，并稍微增加其值
  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;

  // 在 CPU 上对张量 a_cpu 进行标量除法操作
  a_cpu.div_(b_scalar);
  // 在 Vulkan 设备上对张量 a_vulkan 进行标量除法操作
  a_vulkan.div_(b_scalar);

  // 检查两个结果张量是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果结果不几乎相等，则展示它们的相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言结果张量几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_to_scalar_wrapped) {
  // 如果 Vulkan 不可用，则直接返回
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成一个随机标量
  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上生成一个指定形状的随机张量，并稍微增加其值
  const auto b_cpu = at::rand({2, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  // 将 CPU 上的张量移动到 Vulkan 设备上
  const auto b_vulkan = b_cpu.vulkan();

  // 在 CPU 上对标量 a 和张量 b_cpu 进行除法操作
  const auto c_cpu = at::div(a, b_cpu);
  // 在 Vulkan 设备上对标量 a 和张量 b_vulkan 进行除法操作
  const auto c_vulkan = at::div(a, b_vulkan);

  // 检查两个结果张量是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果结果不几乎相等，则展示它们的相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言结果张量几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, empty) {

  // 断言在 Vulkan 设备上创建一个指定形状和数据类型的空张量不会抛出异常
  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

void test_expand(const at::IntArrayRef input_shape, const at::IntArrayRef output_shape) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 在 CPU 上生成一个指定形状的随机张量
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量移动到 Vulkan 设备上
  const auto vulkan = cpu.vulkan();

  // 在 CPU 上将张量 cpu 按照指定形状扩展
  cpu.expand(output_shape);
  // 在 Vulkan 设备上将张量 vulkan 按照指定形状扩展
  vulkan.expand(output_shape);

  // 检查两个结果张量是否几乎相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果结果不几乎相等，则展示它们的相对误差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // 断言结果张量几乎相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, expand_exceptions) {
  // Vulkan expand supports input dims <= 4
  auto in_cpu = at::rand({1, 2, 3, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // 期望抛出异常，因为输入维度超过了 Vulkan 支持的最大维度
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({1, 2, 3, 4}), ::std::exception);

  // Vulkan expand supports output_size <= 4
  in_cpu = at::rand({1, 2, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  // 期望抛出异常，因为输出维度超过了 Vulkan 支持的最大维度
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({1, 1, 2, 3, 4}), ::std::exception);

  // Vulkan expand expects output size >= input
  in_cpu = at::rand({1, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  // 期望抛出异常，因为 Vulkan 要求输出维度至少要大于等于输入维度
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({2, 3}), ::std::exception);

  // Non-singleton dimensions must match
  in_cpu = at::rand({3, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 期望抛出异常，因为非单例维度必须匹配
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({1, 1}), ::std::exception);

  // -1 not allowed in leading, non-existing dimension
  in_cpu = at::rand({3, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 期望抛出异常，因为在主导的、不存在的维度中不允许使用 -1
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({-1, 3, 1}), ::std::exception);
}

TEST_F(VulkanAPITest, expand_1d) {
  test_expand({1}, {3});

  test_expand({1}, {9, 3});       // 1d->2d
  test_expand({1}, {8, 9, 3});    // 1d->3d
  test_expand({1}, {7, 8, 9, 3}); // 1d->4d
}

TEST_F(VulkanAPITest, expand_2d) {
  test_expand({5, 1}, {-1, 5}); // W
  test_expand({1, 5}, {5, 5});  // H

  test_expand({5, 1}, {2, -1, 5});    // 2d->3d
  test_expand({1, 5}, {2, 5, 3, -1}); // 2d->4d
}

TEST_F(VulkanAPITest, expand_3d) {
  test_expand({3, 4, 1}, {3, 4, -1}); // W
  test_expand({3, 1, 5}, {-1, 4, 5}); // H
  test_expand({1, 4, 5}, {3, -1, 5}); // C

  test_expand({5, 4, 3}, {2, -1, -1, -1}); // 3d->4d
}

TEST_F(VulkanAPITest, expand_4d) {
  test_expand({5, 4, 3, 1}, {5, 4, 3, 9}); // W
  test_expand({5, 4, 1, 2}, {5, 4, 9, 2}); // H
  test_expand({5, 1, 3, 2}, {5, 9, 3, 2}); // C
  test_expand({1, 4, 3, 2}, {9, 4, 3, 2}); // N
}

TEST_F(VulkanAPITest, expand_as) {
  // expand_as calls into expand, without negative sizes, those tests should be sufficient.
  c10::InferenceMode mode;
  const auto cpu = at::rand({1, 1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();
  const auto other = at::rand({9, 11, 33, 22}, at::device(at::kCPU).dtype(at::kFloat));

  // 调用 expand_as 方法，不包含负数维度，这些测试应该足够了
  cpu.expand_as(other);
  vulkan.expand_as(other);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }
  ASSERT_TRUE(check);
}

void test_flip(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list) {
  c10::InferenceMode mode;
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::flip(in_cpu, dim_list);
  const auto out_vulkan = at::flip(in_vulkan, dim_list);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }
    // 输出测试失败信息，包括输入形状和维度列表
    std::cout << "test flip failed with input_shape: " << input_shape
              << " and dim_list: " << dim_list << std::endl;
  }

  // 使用断言来验证 check 变量为真，如果不为真，则测试失败
  ASSERT_TRUE(check);
}

// 在 VulkanAPITest 测试套件中，测试一维数组的翻转操作
TEST_F(VulkanAPITest, flip_1d) {
  // 调用 test_flip 函数测试长度为 5 的数组的翻转，期望结果是数组 [0]
  test_flip({5}, {0});
  // 调用 test_flip 函数测试长度为 5 的数组的翻转，期望结果是数组 [-1]
  test_flip({5}, {-1});
}

// 在 VulkanAPITest 测试套件中，测试二维数组的翻转操作
TEST_F(VulkanAPITest, flip_2d) {
  // 调用 test_flip 函数测试形状为 [5, 5] 的数组的翻转，期望结果是数组 [-1]
  test_flip({5, 5}, {-1});
  // 调用 test_flip 函数测试形状为 [2, 7] 的数组的翻转，期望结果是数组 [-2]
  test_flip({2, 7}, {-2});

  // 调用 test_flip 函数测试形状为 [5, 5] 的数组的翻转，期望结果是数组 [0, 1]
  test_flip({5, 5}, {0, 1});
}

// 在 VulkanAPITest 测试套件中，测试三维数组的翻转操作
TEST_F(VulkanAPITest, flip_3d) {
  // 调用 test_flip 函数测试形状为 [5, 7, 5] 的数组的翻转，期望结果是数组 [-1]
  test_flip({5, 7, 5}, {-1});
  // 调用 test_flip 函数测试形状为 [2, 9, 7] 的数组的翻转，期望结果是数组 [-2]
  test_flip({2, 9, 7}, {-2});
  // 调用 test_flip 函数测试形状为 [9, 7, 5] 的数组的翻转，期望结果是数组 [-3]
  test_flip({9, 7, 5}, {-3});

  // 调用 test_flip 函数测试形状为 [10, 7, 5] 的数组的翻转，期望结果是数组 [0, 1]
  test_flip({10, 7, 5}, {0, 1});
  // 调用 test_flip 函数测试形状为 [10, 7, 5] 的数组的翻转，期望结果是数组 [0, 2]
  test_flip({10, 7, 5}, {0, 2});
  // 调用 test_flip 函数测试形状为 [10, 7, 5] 的数组的翻转，期望结果是数组 [1, 2]
  test_flip({10, 7, 5}, {1, 2});

  // 调用 test_flip 函数测试形状为 [10, 7, 5] 的数组的翻转，期望结果是数组 [2, 1, 0]
  test_flip({10, 7, 5}, {2, 1, 0});
}

// 在 VulkanAPITest 测试套件中，测试四维数组的翻转操作
TEST_F(VulkanAPITest, flip_4d) {
  // 调用 test_flip 函数测试形状为 [2, 9, 1, 1] 的数组的翻转，期望结果是数组 [-1]
  test_flip({2, 9, 1, 1}, {-1});
  // 调用 test_flip 函数测试形状为 [7, 5, 9, 3] 的数组的翻转，期望结果是数组 [-2]
  test_flip({7, 5, 9, 3}, {-2});
  // 调用 test_flip 函数测试形状为 [3, 8, 5, 2] 的数组的翻转，期望结果是数组 [-3]
  test_flip({3, 8, 5, 2}, {-3});
  // 调用 test_flip 函数测试形状为 [7, 9, 5, 3] 的数组的翻转，期望结果是数组 [-4]
  test_flip({7, 9, 5, 3}, {-4});

  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [0, 1]
  test_flip({10, 7, 5, 6}, {0, 1});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [0, 2]
  test_flip({10, 7, 5, 6}, {0, 2});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [0, 3]
  test_flip({10, 7, 5, 6}, {0, 3});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [1, 2]
  test_flip({10, 7, 5, 6}, {1, 2});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [1, 3]
  test_flip({10, 7, 5, 6}, {1, 3});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [2, 3]
  test_flip({10, 7, 5, 6}, {2, 3});

  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [0, 1, 2]
  test_flip({10, 7, 5, 6}, {0, 1, 2});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [0, 1, 3]
  test_flip({10, 7, 5, 6}, {0, 1, 3});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [0, 2, 3]
  test_flip({10, 7, 5, 6}, {0, 2, 3});
  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [3, 2, 1]
  test_flip({10, 7, 5, 6}, {3, 2, 1});

  // 调用 test_flip 函数测试形状为 [10, 7, 5, 6] 的数组的翻转，期望结果是数组 [3, 2, 1, 0]
  test_flip({10, 7, 5, 6}, {3, 2, 1, 0});
}

// 在 VulkanAPITest 测试套件中，测试 GELU 函数的正确性
TEST_F(VulkanAPITest, gelu) {
  // 创建形状为 [17, 197, 302, 5] 的随机浮点数 CPU 张量
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 在 CPU 上调用 GELU 函数，使用 "tanh" 版本
  auto out_cpu = at::gelu(in_cpu, "tanh");
  // 在 Vulkan 上调用 GELU 函数，使用 "tanh" 版本
TEST_F(VulkanAPITest, hardsigmoid) {
  // 生成一个形状为 [17, 197, 302, 5] 的随机张量，元素值在 [-6, 6] 之间，数据类型为 float，设备为 CPU
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 对 CPU 上的输入张量应用 hardsigmoid 激活函数
  const auto out_cpu = at::hardsigmoid(in_cpu);
  // 对 Vulkan 张量应用 hardsigmoid 激活函数
  const auto out_vulkan = at::hardsigmoid(in_vulkan);

  // 检查 CPU 和 Vulkan 张量的输出结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查结果为假，则展示两个张量的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardsigmoid_) {
  // 生成一个形状为 [17, 197, 302, 5] 的随机张量，元素值在 [-6, 6] 之间，数据类型为 float，设备为 CPU
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto vulkan = cpu.vulkan();

  // 对 CPU 上的输入张量应用原位（in-place）的 hardsigmoid 激活函数
  at::hardsigmoid_(cpu);
  // 对 Vulkan 张量应用原位（in-place）的 hardsigmoid 激活函数
  at::hardsigmoid_(vulkan);

  // 检查 CPU 和 Vulkan 张量的输出结果是否几乎相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查结果为假，则展示两个张量的相对误差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardshrink) {
  // 遍历 lambd_value 取值范围内的每个值
  for (const auto lambd_value : {-4.2, -1.0, 0.42, 1.0, 4.2, 13.7}) {
    // 生成一个形状为 [3, 63, 79, 17] 的随机张量，元素值在 [-10, 10] 之间，数据类型为 float，设备为 CPU
    const auto in_cpu = (at::rand({3, 63, 79, 17}, at::device(at::kCPU).dtype(at::kFloat)) - 0.5) * 20;
    // 将 CPU 上的张量转换为 Vulkan 张量
    const auto in_vulkan = in_cpu.vulkan();

    // 在 Vulkan 张量上应用 hardshrink 函数
    const auto out_vulkan = at::hardshrink(in_vulkan, lambd_value);

    // 检查 CPU 和 Vulkan 张量的输出结果是否符合 hardshrink 的定义
    const auto check = checkHardShrink(in_cpu, out_vulkan.cpu(), lambd_value);
    // 断言检查结果为真
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, hardshrink_) {
  // 遍历 lambd_value 取值范围内的每个值
  for (const auto lambd_value : {0.42, 1.0, 4.2, 13.7}) {
    // 生成一个形状为 [3, 63, 79, 17] 的随机张量，元素值在 [-10, 10] 之间，数据类型为 float，设备为 CPU
    const auto in_cpu = (at::rand({3, 63, 79, 17}, at::device(at::kCPU).dtype(at::kFloat)) - 0.5) * 20;
    // 将 CPU 上的张量转换为 Vulkan 张量
    const auto in_vulkan = in_cpu.vulkan();

    // 在 CPU 上应用 hardshrink 函数
    const auto out_cpu = in_cpu.hardshrink(lambd_value);
    // 在 Vulkan 张量上应用 hardshrink 函数，并将结果转换回 CPU
    const auto out_vulkan = in_vulkan.hardshrink(lambd_value).cpu();

    // 检查 CPU 和 Vulkan 张量的输出结果是否符合 hardshrink 的定义
    const auto check = checkHardShrink(out_cpu, out_vulkan, lambd_value);
    // 断言检查结果为真
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, hardtanh) {
  // 生成一个形状为 [17, 197, 302, 5] 的随机张量，元素值在 [0, 10] 之间，数据类型为 float，设备为 CPU
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 10;
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 对 CPU 上的输入张量应用 hardtanh 激活函数，参数为 min_val=3, max_val=7
  const auto out_cpu = at::hardtanh(in_cpu, 3, 7);
  // 对 Vulkan 张量应用 hardtanh 激活函数，参数为 min_val=3, max_val=7
  const auto out_vulkan = at::hardtanh(in_vulkan, 3, 7);

  // 检查 CPU 和 Vulkan 张量的输出结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查结果为假，则展示两个张量的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardtanh_) {
  // 生成一个形状为 [17, 197, 302, 5] 的随机张量，元素值在 [0, 10] 之间，数据类型为 float，设备为 CPU
  auto a_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 10;
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 对 CPU 上的输入张量应用原位（in-place）的 hardtanh 激活函数，参数为 min_val=3, max_val=7
  at::hardtanh_(a_cpu, 3, 7);
  // 对 Vulkan 张量应用原位（in-place）的 hardtanh 激活函数，参数为 min_val=3, max_val=7
  at::hardtanh_(a_vulkan, 3, 7);

  // 检查 CPU 和 Vulkan 张量的输出结果是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果检查结果为假，则展示两个张量的相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

void test_packed_layer_norm(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef normalized_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    const float eps) {
```  
# 定义一个函数，输入参数包括一个浮点数 eps。

  c10::InferenceMode mode;
```py  
# 进入 PyTorch C++ 基础设施推断模式。

  const auto input_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
```  
# 生成一个指定形状和数据类型的随机张量 `input_cpu` 在 CPU 上。

  const auto input_vulkan = input_cpu.vulkan();
```py  
# 将 CPU 上的张量 `input_cpu` 转换为 Vulkan 张量 `input_vulkan`。

  const auto weight_cpu =
      at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
```  
# 生成一个指定形状和数据类型的随机权重张量 `weight_cpu` 在 CPU 上。

  const auto bias_cpu =
      at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));
```py  
# 生成一个指定形状和数据类型的随机偏置张量 `bias_cpu` 在 CPU 上。

  const auto output_cpu = at::layer_norm(
      input_cpu, normalized_shape, weight_cpu, bias_cpu, eps, false);
```  
# 对 `input_cpu` 进行层归一化操作，使用给定的 `weight_cpu`、`bias_cpu` 和 `eps`，并得到输出 `output_cpu`。

  auto prepack = callOpByName(
      "vulkan_prepack::create_layernorm_context",
      "",
      weight_cpu, bias_cpu, eps);
```py  
# 调用 Vulkan 前置打包操作，创建层归一化的上下文 `prepack`，使用给定的 `weight_cpu`、`bias_cpu` 和 `eps`。

  auto vulkan_output = callOpByName(
      "vulkan_prepack::run_layernorm_context",
      "",
      input_cpu.vulkan(), normalized_shape, prepack[0]);
```  
# 调用 Vulkan 运行层归一化的上下文，使用 Vulkan 张量 `input_cpu.vulkan()`、`normalized_shape` 和 `prepack[0]`。

  auto output_vulkan = vulkan_output[0].toTensor();
```py  
# 将 Vulkan 输出转换为 PyTorch 张量 `output_vulkan`。

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
```  
# 检查 CPU 和 Vulkan 输出张量 `output_cpu` 和 `output_vulkan.cpu()` 的近似相等性。

  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }
```py  
# 如果检查不通过，则展示它们的相对误差。

  ASSERT_TRUE(check);
```  
# 断言检查结果为真，即 CPU 和 Vulkan 输出张量近似相等。
}

TEST_F(VulkanAPITest, packed_layer_norm_2d) {
  // 调用测试函数 test_packed_layer_norm 进行二维数据的打包层归一化测试
  test_packed_layer_norm({5, 7}, {7}, {7}, {7}, 1e-05);
  // 调用测试函数 test_packed_layer_norm 进行二维数据的打包层归一化测试，包括输入、权重和偏置形状相同的情况
  test_packed_layer_norm({5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, packed_layer_norm_3d) {
  // 调用测试函数 test_packed_layer_norm 进行三维数据的打包层归一化测试
  test_packed_layer_norm({11, 5, 7}, {7}, {7}, {7}, 1e-05);
  // 调用测试函数 test_packed_layer_norm 进行三维数据的打包层归一化测试，包括输入、权重和偏置形状相同的情况
  test_packed_layer_norm({11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  // 调用测试函数 test_packed_layer_norm 进行三维数据的打包层归一化测试，包括输入、权重和偏置形状均相同且较复杂的情况
  test_packed_layer_norm({11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, packed_layer_norm_4d) {
  // 调用测试函数 test_packed_layer_norm 进行四维数据的打包层归一化测试
  test_packed_layer_norm({3, 11, 5, 7}, {7}, {7}, {7}, 1e-05);
  // 调用测试函数 test_packed_layer_norm 进行四维数据的打包层归一化测试，包括输入、权重和偏置形状相同的情况
  test_packed_layer_norm({3, 11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  // 调用测试函数 test_packed_layer_norm 进行四维数据的打包层归一化测试，包括输入、权重和偏置形状均相同且较复杂的情况
  test_packed_layer_norm({3, 11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
  // 调用测试函数 test_packed_layer_norm 进行四维数据的打包层归一化测试，包括输入、权重和偏置形状均相同且非常复杂的情况
  test_packed_layer_norm(
      {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, layer_norm_invalid_inputs) {
  c10::InferenceMode mode;

  // Act: incorrect normalized shape
  // 期望抛出异常：归一化形状不正确
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {8, 5},
      at::rand({8, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);

  // Act: incorrect weight dimensions
  // 期望抛出异常：权重维度不正确
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);

  // Act: incorrect bias dimensions
  // 期望抛出异常：偏置维度不正确
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);

  // Act: input has too many dimensions
  // 期望抛出异常：输入具有太多维度
  EXPECT_THROW({
    at::layer_norm(
      at::rand({1, 2, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);
}

void test_layer_norm(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef normalized_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    // 进入推断模式，确保在此期间不会修改模型的状态
    c10::InferenceMode mode;
    
    // 生成指定形状的随机张量，存储在 CPU 上，并且数据类型为 float
    const auto input_cpu =
        at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
    // 将 CPU 上的输入张量转换为 Vulkan 张量
    const auto input_vulkan = input_cpu.vulkan();
    
    // 生成指定形状的随机权重张量，存储在 CPU 上，并且数据类型为 float
    const auto weight_cpu =
        at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
    // 将 CPU 上的权重张量转换为 Vulkan 张量
    const auto weight_vulkan = weight_cpu.vulkan();
    
    // 生成指定形状的随机偏置张量，存储在 CPU 上，并且数据类型为 float
    const auto bias_cpu =
        at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));
    // 将 CPU 上的偏置张量转换为 Vulkan 张量
    const auto bias_vulkan = bias_cpu.vulkan();
    
    // 使用 layer_norm 函数在 CPU 上对输入张量进行归一化，计算输出结果
    const auto output_cpu = at::layer_norm(
        input_cpu, normalized_shape, weight_cpu, bias_cpu, eps, false);
    // 使用 layer_norm 函数在 Vulkan 张量上对输入张量进行归一化，计算输出结果
    const auto output_vulkan = at::layer_norm(
        input_vulkan, normalized_shape, weight_vulkan, bias_vulkan, eps, false);
    
    // 检查两种计算方式得到的输出张量在浮点数精度下是否几乎相等
    const auto check = almostEqual(output_cpu, output_vulkan.cpu());
    // 如果检查不通过，显示两者的相对误差
    if (!check) {
      showRtol(output_cpu, output_vulkan.cpu());
    }
    
    // 断言检查通过，确保两种计算方式得到的输出张量几乎相等
    ASSERT_TRUE(check);
TEST_F(VulkanAPITest, layer_norm_2d) {
  // 调用测试函数 test_layer_norm，测试二维情况下的层归一化
  test_layer_norm({5, 7}, {7}, {7}, {7}, 1e-05);
  // 再次调用测试函数 test_layer_norm，测试二维情况下另一组参数的层归一化
  test_layer_norm({5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, layer_norm_3d) {
  // 调用测试函数 test_layer_norm，测试三维情况下的层归一化
  test_layer_norm({11, 5, 7}, {7}, {7}, {7}, 1e-05);
  // 再次调用测试函数 test_layer_norm，测试三维情况下另一组参数的层归一化
  test_layer_norm({11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  // 再次调用测试函数 test_layer_norm，测试三维情况下另一组参数的层归一化
  test_layer_norm({11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, layer_norm_4d) {
  // 调用测试函数 test_layer_norm，测试四维情况下的层归一化
  test_layer_norm({3, 11, 5, 7}, {7}, {7}, {7}, 1e-05);
  // 再次调用测试函数 test_layer_norm，测试四维情况下另一组参数的层归一化
  test_layer_norm({3, 11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  // 再次调用测试函数 test_layer_norm，测试四维情况下另一组参数的层归一化
  test_layer_norm({3, 11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
  // 再次调用测试函数 test_layer_norm，测试四维情况下另一组参数的层归一化
  test_layer_norm(
      {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, 1e-05);
}

void test_native_layer_norm(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef normalized_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    const float eps) {
  // 进入推理模式
  c10::InferenceMode mode;

  // 生成指定形状的随机 CPU 输入数据
  const auto input_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 转换 CPU 输入数据为 Vulkan 张量
  const auto input_vulkan = input_cpu.vulkan();

  // 生成指定形状的随机 CPU 权重数据
  const auto weight_cpu =
      at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 转换 CPU 权重数据为 Vulkan 张量
  const auto weight_vulkan = weight_cpu.vulkan();

  // 生成指定形状的随机 CPU 偏置数据
  const auto bias_cpu =
      at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 转换 CPU 偏置数据为 Vulkan 张量
  const auto bias_vulkan = bias_cpu.vulkan();

  // 使用 CPU 数据进行层归一化操作，得到 CPU 输出结果
  const auto output_cpu = at::native_layer_norm(
      input_cpu, normalized_shape, weight_cpu, bias_cpu, eps);
  // 使用 Vulkan 数据进行层归一化操作，得到 Vulkan 输出结果
  const auto output_vulkan = at::native_layer_norm(
      input_vulkan, normalized_shape, weight_vulkan, bias_vulkan, eps);

  // 检查第一个输出是否几乎相等
  const auto check0 =
      almostEqual(std::get<0>(output_cpu), std::get<0>(output_vulkan).cpu());
  if (!check0) {
    // 若不相等，输出错误信息并展示相对误差
    std::cout
        << "the first output of native_layer_norm: layer_norm is incorrect"
        << std::endl;
    showRtol(std::get<0>(output_cpu), std::get<0>(output_vulkan).cpu());
  }

  // 检查第二个输出是否几乎相等
  const auto check1 =
      almostEqual(std::get<1>(output_cpu), std::get<1>(output_vulkan).cpu());
  if (!check1) {
    // 若不相等，输出错误信息并展示相对误差
    std::cout << "the second output of native_layer_norm: mean is incorrect"
              << std::endl;
    showRtol(std::get<1>(output_cpu), std::get<1>(output_vulkan).cpu());
  }

  // 检查第三个输出是否几乎相等
  const auto check2 =
      almostEqual(std::get<2>(output_cpu), std::get<2>(output_vulkan).cpu());
  if (!check2) {
    // 若不相等，输出错误信息并展示相对误差
    std::cout
        << "the third output of native_layer_norm: 1/sqrt(var+eps) is incorrect"
        << std::endl;
    showRtol(std::get<2>(output_cpu), std::get<2>(output_vulkan).cpu());
  }

  // 断言所有检查均通过
  ASSERT_TRUE(check0 && check1 && check2);
}

TEST_F(VulkanAPITest, native_layer_norm_2d) {
  // 调用测试函数 test_native_layer_norm，测试二维情况下的原生层归一化
  test_native_layer_norm({5, 7}, {7}, {7}, {7}, 1e-05);
  // 再次调用测试函数 test_native_layer_norm，测试二维情况下另一组参数的原生层归一化
  test_native_layer_norm({5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
}
TEST_F(VulkanAPITest, native_layer_norm_3d) {
  // 调用测试函数 test_native_layer_norm 进行三维张量的本地归一化测试，各参数分别为输入形状、normalized_shape、weight、bias 和 epsilon
  test_native_layer_norm({11, 5, 7}, {7}, {7}, {7}, 1e-05);
  // 同上，但参数为输入形状、normalized_shape、weight、bias 和 epsilon
  test_native_layer_norm({11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  // 同上，但参数为输入形状、normalized_shape、weight、bias 和 epsilon
  test_native_layer_norm({11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, native_layer_norm_4d) {
  // 调用测试函数 test_native_layer_norm 进行四维张量的本地归一化测试，各参数分别为输入形状、normalized_shape、weight、bias 和 epsilon
  test_native_layer_norm({3, 11, 5, 7}, {7}, {7}, {7}, 1e-05);
  // 同上，但参数为输入形状、normalized_shape、weight、bias 和 epsilon
  test_native_layer_norm({3, 11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  // 同上，但参数为输入形状、normalized_shape、weight、bias 和 epsilon
  test_native_layer_norm({3, 11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
  // 同上，但参数为输入形状、normalized_shape、weight、bias 和 epsilon
  test_native_layer_norm({3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, leaky_relu) {
  // 对于给定的负斜率值进行 leaky ReLU 激活函数的测试
  for (const auto negative_slope : {0.01, 0.001, 1.0, -0.001}) {
    // 生成指定形状的随机张量在 CPU 上，数据类型为 float32
    const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
    // 将 CPU 上的张量转换到 Vulkan 设备上
    const auto in_vulkan = in_cpu.vulkan();

    // 在 CPU 和 Vulkan 设备上分别应用 leaky ReLU 激活函数
    const auto out_cpu = at::leaky_relu(in_cpu, negative_slope);
    const auto out_vulkan = at::leaky_relu(in_vulkan, negative_slope);

    // 检查 CPU 和 Vulkan 设备上输出张量的近似相等性
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());

    // 如果输出不近似相等，则展示它们的相对容差
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    // 断言 CPU 和 Vulkan 设备上的输出张量近似相等
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, leaky_relu_) {
  // 对于给定的负斜率值进行 in-place leaky ReLU 激活函数的测试
  for (const auto negative_slope : {0.01, 0.001, 1.0, -0.001}) {
    // 生成指定形状的随机张量在 CPU 上，数据类型为 float32
    auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
    // 将 CPU 上的张量转换到 Vulkan 设备上
    auto vulkan = cpu.vulkan();

    // 在 CPU 和 Vulkan 设备上应用 in-place leaky ReLU 激活函数
    at::leaky_relu_(cpu, negative_slope);
    at::leaky_relu_(vulkan, negative_slope);

    // 检查 CPU 和 Vulkan 设备上张量的近似相等性
    const auto check = almostEqual(cpu, vulkan.cpu());

    // 如果输出不近似相等，则展示它们的相对容差
    if (!check) {
      showRtol(cpu, vulkan.cpu());
    }

    // 断言 CPU 和 Vulkan 设备上的张量近似相等
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, lerp) {
  // 对于给定的两个输入张量和权重张量，进行线性插值的测试
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  // 在 CPU 和 Vulkan 设备上进行线性插值操作
  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_cpu);
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_vulkan);

  // 检查 CPU 和 Vulkan 设备上输出张量的近似相等性
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());

  // 如果输出不近似相等，则展示它们的相对容差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 设备上的输出张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_broadcast0) {
  // 对于形状不同的张量进行广播线性插值的测试
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  // 在 CPU 和 Vulkan 设备上进行广播线性插值操作
  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_cpu);
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_vulkan);

  // 检查 CPU 和 Vulkan 设备上输出张量的近似相等性
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());

  // 如果输出不近似相等，则展示它们的相对容差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 设备上的输出张量近似相等
  ASSERT_TRUE(check);
}
// 在 VulkanAPITest 测试套件中的 lerp_broadcast1 测试用例
TEST_F(VulkanAPITest, lerp_broadcast1) {
  // 创建一个形状为 {3, 4, 179, 221} 的随机张量 a_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 a_cpu 转换为 Vulkan 张量
  const auto a_vulkan = a_cpu.vulkan();

  // 创建一个形状为 {4, 179, 221} 的随机张量 b_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto b_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 b_cpu 转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 创建一个形状为 {4, 179, 221} 的随机张量 w_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto w_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 w_cpu 转换为 Vulkan 张量
  const auto w_vulkan = w_cpu.vulkan();

  // 使用 lerp 函数在 CPU 上对张量 a_cpu, b_cpu, w_cpu 进行插值操作，结果存储在 c_cpu 中
  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_cpu);
  // 使用 Vulkan API 在 GPU 上对张量 a_vulkan, b_vulkan, w_vulkan 进行插值操作，结果存储在 c_vulkan 中
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_vulkan);

  // 检查 c_cpu 和 c_vulkan 是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果不相等，则展示它们的相对容差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 在 VulkanAPITest 测试套件中的 lerp_ 测试用例
TEST_F(VulkanAPITest, lerp_) {
  // 创建一个形状为 {61, 17, 29, 83} 的随机张量 a_cpu，存储在 CPU 上，并指定数据类型为 float
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 a_cpu 转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 创建一个形状为 {61, 17, 29, 83} 的随机张量 b_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 b_cpu 转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 创建一个形状为 {61, 17, 29, 83} 的随机张量 w_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto w_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 w_cpu 转换为 Vulkan 张量
  const auto w_vulkan = w_cpu.vulkan();

  // 使用 lerp_ 函数在 CPU 上对张量 a_cpu, b_cpu, w_cpu 进行原地插值操作
  a_cpu.lerp_(b_cpu, w_cpu);
  // 使用 Vulkan API 在 GPU 上对张量 a_vulkan, b_vulkan, w_vulkan 进行原地插值操作
  a_vulkan.lerp_(b_vulkan, w_vulkan);

  // 检查 a_cpu 和 a_vulkan 是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果不相等，则展示它们的相对容差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 在 VulkanAPITest 测试套件中的 lerp_broadcast0_ 测试用例
TEST_F(VulkanAPITest, lerp_broadcast0_) {
  // 创建一个形状为 {3, 5, 179, 221} 的随机张量 a_cpu，存储在 CPU 上，并指定数据类型为 float
  auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 a_cpu 转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 创建一个形状为 {3, 5, 1, 1} 的随机张量 b_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto b_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 b_cpu 转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 创建一个形状为 {3, 5, 1, 221} 的随机张量 w_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto w_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 w_cpu 转换为 Vulkan 张量
  const auto w_vulkan = w_cpu.vulkan();

  // 使用 lerp_ 函数在 CPU 上对张量 a_cpu, b_cpu, w_cpu 进行原地插值操作
  a_cpu.lerp_(b_cpu, w_cpu);
  // 使用 Vulkan API 在 GPU 上对张量 a_vulkan, b_vulkan, w_vulkan 进行原地插值操作
  a_vulkan.lerp_(b_vulkan, w_vulkan);

  // 检查 a_cpu 和 a_vulkan 是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果不相等，则展示它们的相对容差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 在 VulkanAPITest 测试套件中的 lerp_broadcast1_ 测试用例
TEST_F(VulkanAPITest, lerp_broadcast1_) {
  // 创建一个形状为 {3, 4, 179, 221} 的随机张量 a_cpu，存储在 CPU 上，并指定数据类型为 float
  auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 a_cpu 转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 创建一个形状为 {4, 179, 221} 的随机张量 b_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto b_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 b_cpu 转换为 Vulkan 张量
  const auto b_vulkan = b_cpu.vulkan();

  // 创建一个形状为 {4, 179, 221} 的随机张量 w_cpu，存储在 CPU 上，并指定数据类型为 float
  const auto w_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 w_cpu 转换为 Vulkan 张量
  const auto w_vulkan = w_cpu.vulkan();

  // 使用 lerp_ 函数在 CPU 上对张量 a_cpu, b_cpu, w_cpu 进行原地插值操作
  a_cpu.lerp_(b_cpu, w_cpu);
  // 使用 Vulkan API 在 GPU 上对张量 a_vulkan, b_vulkan, w_vulkan 进行原地插值操作
  a_vulkan.lerp_(b
TEST_F(VulkanAPITest, lerp_scalar) {
  // 生成一个指定形状的随机张量 `a_cpu`，使用 CPU 计算设备和浮点数类型
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 `a_cpu` 转换为 Vulkan 张量 `a_vulkan`
  const auto a_vulkan = a_cpu.vulkan();

  // 生成一个指定形状的随机张量 `b_cpu`，使用 CPU 计算设备和浮点数类型
  const auto b_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 `b_cpu` 转换为 Vulkan 张量 `b_vulkan`
  const auto b_vulkan = b_cpu.vulkan();

  // 定义标量权重 `w_scalar`
  const float w_scalar = 3.1415f;

  // 使用 `lerp` 函数计算 `a_cpu` 和 `b_cpu` 的线性插值结果 `c_cpu`
  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_scalar);
  // 使用 Vulkan 下的 `lerp` 函数计算 `a_vulkan` 和 `b_vulkan` 的线性插值结果 `c_vulkan`
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_scalar);

  // 检查 `c_cpu` 和 `c_vulkan.cpu()` 是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言 `check` 为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_scalar_) {
  // 生成一个指定形状的随机张量 `a_cpu`，使用 CPU 计算设备和浮点数类型
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 `a_cpu` 转换为 Vulkan 张量 `a_vulkan`
  auto a_vulkan = a_cpu.vulkan();

  // 生成一个指定形状的随机张量 `b_cpu`，使用 CPU 计算设备和浮点数类型
  const auto b_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 `b_cpu` 转换为 Vulkan 张量 `b_vulkan`
  const auto b_vulkan = b_cpu.vulkan();

  // 定义标量权重 `w_scalar`
  const float w_scalar = 3.1415f;

  // 在 `a_cpu` 上应用 `lerp_` 函数，原地进行线性插值操作
  a_cpu.lerp_(b_cpu, w_scalar);
  // 在 `a_vulkan` 上应用 `lerp_` 函数，原地进行 Vulkan 下的线性插值操作
  a_vulkan.lerp_(b_vulkan, w_scalar);

  // 检查 `a_cpu` 和 `a_vulkan.cpu()` 是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言 `check` 为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardswish) {
  // 生成一个指定形状的随机张量 `in_cpu`，使用 CPU 计算设备和浮点数类型
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  // 将 `in_cpu` 转换为 Vulkan 张量 `in_vulkan`
  const auto in_vulkan = in_cpu.vulkan();

  // 在 `in_cpu` 上应用 `hardswish` 函数，生成输出张量 `out_cpu`
  const auto out_cpu = at::hardswish(in_cpu);
  // 在 `in_vulkan` 上应用 `hardswish` 函数，生成 Vulkan 下的输出张量 `out_vulkan`
  const auto out_vulkan = at::hardswish(in_vulkan);

  // 检查 `out_cpu` 和 `out_vulkan.cpu()` 是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 `check` 为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, threshold) {
  // 生成一个指定形状的随机张量 `in_cpu`，使用 CPU 计算设备和浮点数类型
  const auto in_cpu = at::rand({2, 11, 57, 23}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  // 将 `in_cpu` 转换为 Vulkan 张量 `in_vulkan`
  const auto in_vulkan = in_cpu.vulkan();

  // 定义阈值和值
  const float threshold = 2.0f;
  const float value = 5.0f;

  // 在 `in_cpu` 上应用 `threshold` 函数，生成输出张量 `out_cpu`
  const auto out_cpu = at::threshold(in_cpu, threshold, value);
  // 在 `in_vulkan` 上应用 `threshold` 函数，生成 Vulkan 下的输出张量 `out_vulkan`
  const auto out_vulkan = at::threshold(in_vulkan, threshold, value);

  // 检查 `out_cpu` 和 `out_vulkan.cpu()` 是否满足阈值检查
  const auto check = checkThreshold(out_cpu, out_vulkan.cpu(), threshold, value);
  // 断言 `check` 为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardswish_) {
  // 生成一个指定形状的随机张量 `cpu`，使用 CPU 计算设备和浮点数类型
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  // 将 `cpu` 转换为 Vulkan 张量 `vulkan`
  auto vulkan = cpu.vulkan();

  // 在 `cpu` 上原地应用 `hardswish_` 函数
  at::hardswish_(cpu);
  // 在 `vulkan` 上原地应用 `hardswish_` 函数
  at::hardswish_(vulkan);

  // 检查 `cpu` 和 `vulkan.cpu()` 是否几乎相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // 断言 `check` 为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, masked_fill_invalidinputs_exceptions) {
  // Arrange: Vulkan masked_fill 函数预期输入张量的维度不超过 4
  {
    // 生成一个形状为 {3, 5, 2, 3, 2} 的随机张量 `in_cpu`，使用 CPU 计算设备和浮点数类型
    const auto in_cpu =
        at::rand({3, 5, 2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
    // 生成一个形状为 {2, 3, 2} 的随机布尔张量 `mask_cpu`，使用 CPU 计算设备
    const auto mask_cpu =
        at::randint(0, 2, {2, 3, 2}, at::device(at::kCPU).dtype(at::kBool));

    // Act
    // 期望抛出异常，因为 `in_cpu.vulkan().masked_fill` 的 mask 张量维度超过 4
    EXPECT_THROW(
        {
          const auto out_vulkan =
              in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), -7.0f);
          ;
        },
    // 创建一个形状为 [2, 3, 2] 的随机浮点数张量 `in_cpu`，放置在 CPU 上
    const auto in_cpu =
        at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
    // 创建一个形状为 [3, 5, 2, 3, 2] 的随机布尔数张量 `mask_cpu`，放置在 CPU 上
    const auto mask_cpu = at::randint(
        0, 2, {3, 5, 2, 3, 2}, at::device(at::kCPU).dtype(at::kBool));
    
    // 期望操作：检查以下代码块是否会抛出异常
    EXPECT_THROW(
        {
          // 使用 Vulkan 引擎处理 `in_cpu`，然后使用 `mask_cpu` 对其进行掩码填充，用 -7.0f 填充被掩码的位置
          const auto out_vulkan =
              in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), -7.0f);
          ;
        },
        ::std::exception);
    }
    
    // Arrange: 输入张量和掩码张量的形状应该可以进行广播
    {
      // 创建一个形状为 [2, 3, 2] 的随机浮点数张量 `in_cpu`，放置在 CPU 上
      const auto in_cpu =
          at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
      // 创建一个形状为 [3, 3, 2] 的随机布尔数张量 `mask_cpu`，放置在 CPU 上
      const auto mask_cpu =
          at::randint(0, 2, {3, 3, 2}, at::device(at::kCPU).dtype(at::kBool));
    
      // 期望操作：检查以下代码块是否会抛出异常
      EXPECT_THROW(
          {
            // 使用 Vulkan 引擎处理 `in_cpu`，然后使用 `mask_cpu` 对其进行掩码填充，用 -7.0f 填充被掩码的位置
            const auto out_vulkan =
                in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), -7.0f);
            ;
          },
          ::std::exception);
    }
    
    // Arrange: 值应该是一个零维值张量或标量
    {
      // 创建一个形状为 [2, 3, 2] 的随机浮点数张量 `in_cpu`，放置在 CPU 上
      const auto in_cpu =
          at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
      // 创建一个形状为 [2, 3, 2] 的随机布尔数张量 `mask_cpu`，放置在 CPU 上
      const auto mask_cpu =
          at::randint(0, 2, {2, 3, 2}, at::device(at::kCPU).dtype(at::kBool));
    
      // 期望操作：检查以下代码块是否会抛出异常
      EXPECT_THROW(
          {
            // 使用 Vulkan 引擎处理 `in_cpu`，然后使用 `mask_cpu` 对其进行掩码填充，尝试用形状为 [1, 2] 的随机张量填充被掩码的位置
            const auto out_vulkan =
                in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), at::rand({1, 2}));
            ;
          },
          ::std::exception);
    }
/**
 * 输出形状的函数，打印给定形状向量中的每个元素
 */
void print_shape(const std::vector<int64_t>& shape) {
  for (const auto& num : shape) {
    std::cout << num << " ";
  }
}

/**
 * 测试 masked_fill_scalar 函数，考虑 input_shape 和 mask_shape 的所有广播情况。
 * 给定的 input_shape 和 mask_shape 相同，例如都等于 [3, 5, 2, 3]。
 * 首先分别截取 input_shape 和 mask_shape 的所有可能的前置维度。
 * 将结果分别表示为 curr_input_shape 和 curr_mask_shape，例如
 * curr_input_shape = [5, 2, 3] 和 curr_mask_shape = [2, 3]。
 * 然后为 curr_input_shape 和 curr_mask_shape 生成所有可能的索引子集，
 * 并将每个子集对应的元素设置为 1。例如，对于 curr_input_shape = [5, 2, 3]，
 * 一个可能的 input_idx_subset = [0, 2]。我们将 curr_input_shape 的第0和第2个元素设置为 1，
 * 然后 curr_input_shape = [1, 2, 1]。对于 curr_mask_shape = [2, 3]，一个可能的 mask_idx_subset = [0]，
 * 然后更新 curr_mask_shape = [1, 3]。
 * 最后，使用 curr_input_shape 和 curr_mask_shape 的组合测试 masked_fill 函数。
 * 在上面的示例中，将生成形状为 [1, 2, 3] 的输出张量。
 */
void test_masked_fill_scalar(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef mask_shape) {
  c10::InferenceMode mode;

  const size_t input_dim = input_shape.size();
  const size_t mask_dim = mask_shape.size();
  for (int input_shape_id = input_dim - 1; input_shape_id >= 0;
       --input_shape_id) {
    // 截断 input_shape 的前置维度
    auto curr_input_shape =
        input_shape.slice(input_shape_id, input_dim - input_shape_id);

    // 生成所有可能的子集，包含在 0 到 input_dim - input_shape_id - 1 之间的数
    std::vector<std::vector<int64_t>> input_indices_subsets;
    std::vector<int64_t> curr_input_indices;
    gen_all_subsets(
        input_indices_subsets,
        input_dim - input_shape_id,
        0,
        curr_input_indices);
    // 遍历输入索引子集
    for (auto input_idx_subset : input_indices_subsets) {
      // 将当前输入形状的向量表示复制到临时向量中
      auto tmp_curr_input_shape = curr_input_shape.vec();
      // 针对当前子集中的每个输入索引，将对应位置的元素设置为1
      for (auto input_idx : input_idx_subset) {
        tmp_curr_input_shape[input_idx] = 1;
      }

      // 从最后一个维度向前遍历掩码的形状
      for (int mask_shape_id = mask_dim - 1; mask_shape_id >= 0;
           --mask_shape_id) {
        // 从当前掩码形状中切片得到切片后的形状
        auto curr_mask_shape =
            mask_shape.slice(mask_shape_id, mask_dim - mask_shape_id);

        // 生成0到mask_dim - mask_shape_id - 1（包含）之间所有可能的子集
        std::vector<std::vector<int64_t>> mask_indices_subsets;
        std::vector<int64_t> curr_mask_indices;
        // 调用函数生成所有子集
        gen_all_subsets(
            mask_indices_subsets,
            mask_dim - mask_shape_id,
            0,
            curr_mask_indices);

        // 遍历掩码索引的所有子集
        for (auto mask_idx_subset : mask_indices_subsets) {
          // 将当前掩码形状的向量表示复制到临时向量中
          auto tmp_curr_mask_shape = curr_mask_shape.vec();
          // 针对当前子集中的每个掩码索引，将对应位置的元素设置为1
          for (auto mask_idx : mask_idx_subset) {
            tmp_curr_mask_shape[mask_idx] = 1;
          }

          // 在CPU上生成随机张量in_cpu，形状为tmp_curr_input_shape，数据类型为float
          at::Tensor in_cpu = at::rand(
              tmp_curr_input_shape, at::device(at::kCPU).dtype(at::kFloat));
          // 在CPU上生成随机整数张量mask_cpu，形状为tmp_curr_mask_shape，数据类型为bool
          at::Tensor mask_cpu = at::randint(
              0, 2, tmp_curr_mask_shape, at::device(at::kCPU).dtype(at::kBool));
          // 对in_cpu应用掩码操作，将mask_cpu为True的位置用-7.0f填充，结果存储在out_cpu中
          at::Tensor out_cpu = in_cpu.masked_fill(mask_cpu, -7.0f);

          // 获取in_cpu的Vulkan张量表示
          at::Tensor in_vulkan = in_cpu.vulkan();
          // 获取mask_cpu的Vulkan张量表示
          at::Tensor mask_vulkan = mask_cpu.vulkan();
          // 在Vulkan张量上应用掩码操作，将mask_vulkan为True的位置用-7.0f填充，结果存储在out_vulkan中
          at::Tensor out_vulkan = in_vulkan.masked_fill(mask_vulkan, -7.0f);
          // 检查out_cpu和out_vulkan是否几乎相等
          const bool check = almostEqual(out_cpu, out_vulkan.cpu());

          // 如果检查失败，则显示相对误差和相关信息
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
            std::cout << "Masked_fill test failed when input is of shape [";
            print_shape(tmp_curr_input_shape);
            std::cout << "], and mask of shape [";
            print_shape(tmp_curr_mask_shape);
            std::cout << "]" << std::endl;
          }

          // 使用断言确保检查通过
          ASSERT_TRUE(check);
        }
      }
    }
  }
TEST_F(VulkanAPITest, masked_fill_scalar_mult4ch) {
  // 调用 test_masked_fill_scalar 函数，测试标量填充，输入和输出形状均为 {3, 4, 5, 7}
  test_masked_fill_scalar({3, 4, 5, 7}, {3, 4, 5, 7});
}

TEST_F(VulkanAPITest, masked_fill_scalar_nonmult4ch) {
  // 调用 test_masked_fill_scalar 函数，测试标量填充，输入和输出形状均为 {3, 5, 2, 3}
  test_masked_fill_scalar({3, 5, 2, 3}, {3, 5, 2, 3});
}

void test_masked_fill_tensor(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef mask_shape) {
  // 进入推断模式
  c10::InferenceMode mode;

  // 生成在 CPU 上的随机输入张量
  at::Tensor in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 生成在 CPU 上的随机掩码张量（布尔类型）
  at::Tensor mask_cpu =
      at::randint(0, 2, mask_shape, at::device(at::kCPU).dtype(at::kBool));
  // 使用掩码对输入张量进行标量填充，填充值为 -7.0f
  at::Tensor out_cpu = in_cpu.masked_fill(mask_cpu, at::scalar_tensor(-7.0f));
  
  // 将 CPU 上的输入张量转换为 Vulkan 张量
  at::Tensor in_vulkan = in_cpu.vulkan();
  // 将 CPU 上的掩码张量转换为 Vulkan 张量
  at::Tensor mask_vulkan = mask_cpu.vulkan();
  // 使用 Vulkan 张量进行标量填充，填充值为 -7.0f
  at::Tensor out_vulkan =
      in_vulkan.masked_fill(mask_vulkan, at::scalar_tensor(-7.0f));
  
  // 检查 Vulkan 张量的输出与 CPU 张量的输出是否几乎相等
  const bool check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }
  
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, masked_fill_tensor_mult4ch) {
  // 调用 test_masked_fill_tensor 函数，测试张量填充，输入形状为 {3, 4, 2, 3}，掩码形状为 {1, 4, 1, 1}
  test_masked_fill_tensor({3, 4, 2, 3}, {1, 4, 1, 1});
}

TEST_F(VulkanAPITest, masked_fill_tensor_nonmult4ch) {
  // 调用 test_masked_fill_tensor 函数，测试张量填充，输入形状为 {3, 5, 2, 3}，掩码形状为 {1, 5, 1, 1}
  test_masked_fill_tensor({3, 5, 2, 3}, {1, 5, 1, 1});
}

TEST_F(VulkanAPITest, max_pool2d) {
  // 进入推断模式
  c10::InferenceMode mode;

  // 生成在 CPU 上的随机输入张量
  const auto in_cpu = at::rand({5, 13, 55, 68}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 在 CPU 上执行 max_pool2d 操作
  const auto out_cpu = at::max_pool2d(in_cpu, {3, 4}, {2, 1}, {1, 1}, {1, 1}, false);
  // 将 CPU 上的输入张量转换为 Vulkan 张量，并在 Vulkan 上执行 max_pool2d 操作
  const auto out_vulkan = at::max_pool2d(in_cpu.vulkan(), {3, 4}, {2, 1}, {1, 1}, {1,1}, false);

  // 检查 Vulkan 张量的输出与 CPU 张量的输出是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mean_invalid_inputs) {
  // 进入推断模式
  c10::InferenceMode mode;

  // 测试：输入维度过大，期望抛出异常
  EXPECT_THROW({
    at::mean(at::rand({3, 5, 7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // 测试：维度超出范围，期望抛出异常
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // 测试：维度超出范围，期望抛出异常
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {-4});
  }, ::std::exception);

  // 测试：重复的维度，期望抛出异常
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, 1});
  }, ::std::exception);

  // 测试：重复的维度，期望抛出异常
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, -2});
  }, ::std::exception);
}
// 测试函数，计算指定维度上的均值，并进行 Vulkan 后端的验证
void test_mean_dim(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list, bool keepdim=false) {
  // 生成指定形状的随机张量在 CPU 上
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 使用 Vulkan 后端处理 CPU 上的张量
  const auto in_vulkan = in_cpu.vulkan();

  // 计算在指定维度上的均值并保持维度
  const auto out_cpu = at::mean(in_cpu, dim_list, keepdim);
  // 使用 Vulkan 后端计算在指定维度上的均值并保持维度
  const auto out_vulkan = at::mean(in_vulkan, dim_list, keepdim);

  // 检查 CPU 和 Vulkan 计算结果是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不接近，则输出测试失败的信息，显示输入形状和维度列表
  if (!check) {
    std::cout << "mean_dim test failed with input shape: "
              << input_shape << " and dim_list: " << dim_list << std::endl;
    // 显示实际和期望的相对误差
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用 ASSERT_TRUE 确认 CPU 和 Vulkan 计算结果接近
  ASSERT_TRUE(check);
}

// 测试用例：测试二维张量的均值计算
TEST_F(VulkanAPITest, mean_dim_2d) {
  // 测试在最后一个维度上计算均值
  test_mean_dim({2, 3}, {-1});
  // 测试在倒数第二个维度上计算均值
  test_mean_dim({2, 7}, {-2});
}

// 测试用例：测试三维张量的均值计算
TEST_F(VulkanAPITest, mean_dim_3d) {
  // 测试在最后一个维度上计算均值
  test_mean_dim({9, 7, 5}, {-1});
  // 测试在倒数第二个维度上计算均值
  test_mean_dim({5, 7, 9}, {-2});
  // 测试在倒数第三个维度上计算均值
  test_mean_dim({5, 7, 9}, {-3});

  // 测试在前两个维度上计算均值
  test_mean_dim({10, 7, 5}, {0, 1});
  // 测试在第一个和第三个维度上计算均值
  test_mean_dim({10, 7, 5}, {0, 2});
  // 测试在第二个和第三个维度上计算均值
  test_mean_dim({10, 7, 5}, {1, 2});
  // 测试在倒数第一个和倒数第二个维度上计算均值
  test_mean_dim({10, 7, 5}, {-1, -2});
  // 测试在第一个和倒数第二个维度上计算均值
  test_mean_dim({10, 7, 5}, {0, -2});
}

// 测试用例：测试四维张量的均值计算
TEST_F(VulkanAPITest, mean_dim_4d) {
  // 测试在最后一个维度上计算均值
  test_mean_dim({7, 9, 6, 5}, {-1});
  // 测试在倒数第二个维度上计算均值
  test_mean_dim({6, 5, 7, 9}, {-2});
  // 测试在倒数第三个维度上计算均值
  test_mean_dim({6, 5, 7, 9}, {-3});
  // 测试在倒数第四个维度上计算均值
  test_mean_dim({6, 5, 7, 9}, {-4});

  // 测试在前两个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {0, 1});
  // 测试在第一个和第三个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {0, 2});
  // 测试在第一个和第四个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {0, 3});
  // 测试在第二个和第三个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {1, 2});
  // 测试在第二个和第四个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {1, 3});
  // 测试在第三个和第四个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {2, 3});
  // 测试在倒数第二个和倒数第四个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {-2, -4});

  // 测试在前三个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {0, 1, 2});
  // 测试在前两个和第四个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {0, 1, 3});
  // 测试在第一个、第三个和第四个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {0, 2, 3});
  // 测试在第四个、第三个和第二个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {3, 2, 1});
  // 测试在倒数第三个、倒数第二个和第一个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {3, -2, 1});
  // 测试在倒数第一个、倒数第二个和倒数第三个维度上计算均值
  test_mean_dim({10, 7, 5, 6}, {-3, -2, -1});
}

// 测试用例：测试保持维度的二维张量均值计算
TEST_F(VulkanAPITest, mean_dim_keepdim_2d) {
  // 测试在最后一个维度上计算均值并保持维度
  test_mean_dim({5, 7}, {-1}, true);
  // 测试在倒数第二个维度上计算均值并保持维度
  test_mean_dim({5, 7}, {-2}, true);
}

// 测试用例：测试保持维度的三维张量均值计算
TEST_F(VulkanAPITest, mean_dim_keepdim_3d) {
  // 测试在最后一个维度上计算均值并保持维度
  test_mean_dim({9, 5, 7}, {-1}, true);
  // 测试在倒数第二个维度上计算均值并保持维度
  test_mean_dim({5, 9, 7}, {-2}, true);
  // 测试在倒数第三个维度上计算均值并保持维度
  test_mean_dim({7, 9, 5}, {-3}, true);

  // 测试在前两个维度上计算均值并保持维度
  test_mean_dim({9, 5, 7}, {0, 1}, true);
  // 测试在第一个和第三个维度上计算均值并保持维度
  test_mean_dim({5, 9, 7}, {0, 2}, true);
  // 测试在第二个和第三个维度上计算均值并保持维度
  test_mean_dim({7,
TEST_F(VulkanAPITest, mm) {
  // 创建大小为 [179, 67] 的随机张量，使用 CPU 设备和 float 类型
  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建大小为 [67, 163] 的随机张量，使用 CPU 设备和 float 类型
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 在 CPU 上执行矩阵乘法运算 m1_cpu * m2_cpu，并返回结果
  const auto out_cpu = m1_cpu.mm(m2_cpu);

  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();
  // 在 Vulkan 上执行矩阵乘法运算 m1_vulkan * m2_cpu，并返回结果
  const auto out_vulkan = m1_vulkan.mm(m2_cpu);

  // 检查两个结果张量是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个结果张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mm_m2_is_variable) {
  // 定义矩阵维度变量
  int n = 19;
  int p = 25;
  int m = 21;
  // 创建大小为 [n, p] 的随机张量，使用 CPU 设备和 float 类型
  const auto m1_cpu = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建大小为 [p, m] 的随机张量，使用 CPU 设备和 float 类型
  const auto m2_cpu = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行矩阵乘法运算 m1_cpu * m2_cpu，并返回结果
  const auto out_cpu = m1_cpu.mm(m2_cpu);

  // 将 m1_cpu 和 m2_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();
  const auto m2_vulkan = m2_cpu.vulkan();

  // 在 Vulkan 上执行矩阵乘法运算 m1_vulkan * m2_vulkan，并返回结果
  const auto out_vulkan = m1_vulkan.mm(m2_vulkan);

  // 检查两个结果张量是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个结果张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mm_m1_m2_variable) {
  // 定义矩阵维度变量
  int n = 19;
  int p = 25;
  int m = 21;
  // 创建大小为 [n, p] 的随机张量，使用 CPU 设备和 float 类型
  const auto m1_cpu = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建大小为 [p, m] 的随机张量，使用 CPU 设备和 float 类型
  const auto m2_cpu = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行矩阵乘法运算 m1_cpu * m2_cpu，并返回结果
  const auto out_cpu = at::mm(m1_cpu, m2_cpu);

  // 将 m1_cpu 和 m2_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();
  const auto m2_vulkan = m2_cpu.vulkan();

  // 在 Vulkan 上执行矩阵乘法运算 m1_vulkan * m2_vulkan，并返回结果
  const auto out_vulkan = at::mm(m1_vulkan, m2_vulkan);

  // 检查两个结果张量是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个结果张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mm_error) {
  // 创建大小为 [179, 99] 的随机张量，使用 CPU 设备和 float 类型
  const auto m1_cpu = at::rand({179, 99}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建大小为 [67, 163] 的随机张量，使用 CPU 设备和 float 类型
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 m1_cpu 转换为 Vulkan 张量
  const auto m1_vulkan = m1_cpu.vulkan();

  // 期望抛出异常：m1_vulkan 和 m2_cpu 的维度不匹配
  EXPECT_THROW(m1_vulkan.mm(m2_cpu), ::std::exception);
}

void test_mul(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  // 创建指定形状的随机张量，使用 CPU 设备和 float 类型
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  // 将 in_cpu 和 other_cpu 转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  // 在 CPU 上执行元素级乘法运算 in_cpu * other_cpu，并返回结果
  const auto out_cpu = at::mul(in_cpu, other_cpu);
  // 在 Vulkan 上执行元素级乘法运算 in_vulkan * other_vulkan，并返回结果
  const auto out_vulkan = at::mul(in_vulkan, other_vulkan);

  // 检查两个结果张量是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个结果张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul) {
  // 测试具有相同形状的广播乘法
  test_mul({11, 7, 139, 109}, {11, 7, 139, 109});
}

TEST_F(VulkanAPITest, mul_broadcast0) {
  // 测试广播乘法，其中一个维度为 1
  test_mul({3, 5, 1, 1}, {3, 5, 179, 221});
}

TEST_F(VulkanAPITest, mul_broadcast1) {
  // 测试广播乘法，其中一个维度为 1
  test_mul({3, 5, 179, 221}, {3, 5, 1, 221});
}

TEST_F(VulkanAPITest, mul_broadcast2) {
  // 测试广播乘法，其中一个维度为 1
  test_mul({3, 4, 179, 221}, {4, 1, 1});
}
TEST_F(VulkanAPITest, mul_broadcast3) {
  // 调用测试函数，对形状为 {3, 4, 179, 221} 和 {1, 1, 179, 221} 的张量进行乘法测试
  test_mul({3, 4, 179, 221}, {1, 1, 179, 221});
}

TEST_F(VulkanAPITest, mul_broadcast4) {
  // 调用测试函数，对形状为 {3, 4, 179, 1} 和 {1, 179, 221} 的张量进行乘法测试
  test_mul({3, 4, 179, 1}, {1, 179, 221});
}

TEST_F(VulkanAPITest, mul_broadcast5) {
  // 调用测试函数，对形状为 {2, 1, 7, 1} 和 {1, 5, 1, 4} 的张量进行乘法测试
  test_mul({2, 1, 7, 1}, {1, 5, 1, 4});
}

TEST_F(VulkanAPITest, mul_broadcast6) {
  // 调用测试函数，对形状为 {1, 15, 5, 4} 和 {21, 1, 5, 4} 的张量进行乘法测试
  test_mul({1, 15, 5, 4}, {21, 1, 5, 4});
}

TEST_F(VulkanAPITest, mul_zero_dim) {
  // 调用测试函数，对形状为 {1, 15, 5, 4} 和空张量进行乘法测试
  test_mul({1, 15, 5, 4}, {});
}

TEST_F(VulkanAPITest, mul_) {
  // 生成形状为 {61, 17, 29, 83} 的随机张量，并将其转换为 Vulkan 张量
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  // 生成形状相同的随机张量，并转换为 Vulkan 张量
  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  // 在 CPU 张量上执行原地乘法操作
  a_cpu.mul_(b_cpu);
  // 在 Vulkan 张量上执行原地乘法操作
  a_vulkan.mul_(b_vulkan);

  // 检查两种计算结果的相对误差是否很小
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    // 如果误差较大，则显示两个张量的相对差异
    showRtol(b_cpu, b_vulkan.cpu());
  }

  // 断言两种计算结果应该非常接近
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast0_) {
  // 生成形状为 {12, 17, 29, 83} 的随机张量，并将其转换为 Vulkan 张量
  auto a_cpu = at::rand({12, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  // 生成形状为 {12, 17, 29, 1} 的随机张量，并将其转换为 Vulkan 张量
  const auto b_cpu = at::rand({12, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  // 在 CPU 张量上执行原地乘法操作
  a_cpu.mul_(b_cpu);
  // 在 Vulkan 张量上执行原地乘法操作
  a_vulkan.mul_(b_vulkan);

  // 检查两种计算结果的相对误差是否很小
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    // 如果误差较大，则显示两个张量的相对差异
    showRtol(b_cpu, b_vulkan.cpu());
  }

  // 断言两种计算结果应该非常接近
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast1_) {
  // 生成形状为 {3, 8, 29, 83} 的随机张量，并将其转换为 Vulkan 张量
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  // 生成形状为 {8, 1, 1} 的随机张量，并将其转换为 Vulkan 张量
  const auto b_cpu = at::rand({8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  // 在 CPU 张量上执行原地乘法操作
  a_cpu.mul_(b_cpu);
  // 在 Vulkan 张量上执行原地乘法操作
  a_vulkan.mul_(b_vulkan);

  // 检查两种计算结果的相对误差是否很小
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    // 如果误差较大，则显示两个张量的相对差异
    showRtol(b_cpu, b_vulkan.cpu());
  }

  // 断言两种计算结果应该非常接近
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_scalar) {
  // 生成形状为 {17, 213, 213, 7} 的随机张量，并将其转换为 Vulkan 张量
  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  // 定义标量乘法因子
  const float b_scalar = 3.1415f;

  // 在 CPU 张量上执行标量乘法操作
  const auto c_cpu = at::mul(a_cpu, b_scalar);
  // 在 Vulkan 张量上执行标量乘法操作
  const auto c_vulkan = at::mul(a_vulkan, b_scalar);

  // 检查两种计算结果的相对误差是否很小
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    // 如果误差较大，则显示两个张量的相对差异
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言两种计算结果应该非常接近
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_scalar_) {
  // 生成形状为 {11, 7, 139, 109} 的随机张量，并将其转换为 Vulkan 张量
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  // 定义标量乘法因子
  const float b_scalar = 3.1415f;

  // 在 CPU 张量上执行原地标量乘法操作
  a_cpu.mul_(b_scalar);
  // 在 Vulkan 张量上执行原地标量乘法操作
  a_vulkan.mul_(b_scalar);

  // 检查两种计算结果的相对误差是否很小
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    // 如果误差较大，则显示两个张量的相对差异
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言两种计算结果应该非常接近
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_scalar_wrapped) {
  // 检查 Vulkan 是否可用，如果不可用，则跳过这个测试
  if (!at::is_vulkan_available()) {
  // 程序返回
  return;
}

// 创建一个形状为 [17, 213, 213, 7] 的随机张量 a_cpu，在 CPU 上使用浮点数数据类型
const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
// 将 CPU 上的张量 a_cpu 转换为 Vulkan 张量 a_vulkan
const auto a_vulkan = a_cpu.vulkan();

// 创建一个标量张量 b_scalar，形状为 [1]，在 CPU 上使用浮点数数据类型
const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

// 计算张量 a_cpu 与标量张量 b_scalar 的乘积，结果保存在张量 c_cpu 中
const auto c_cpu = at::mul(a_cpu, b_scalar);
// 计算 Vulkan 张量 a_vulkan 与标量张量 b_scalar 的乘积，结果保存在张量 c_vulkan 中
const auto c_vulkan = at::mul(a_vulkan, b_scalar);

// 检查 c_cpu 和 c_vulkan 是否几乎相等
const auto check = almostEqual(c_cpu, c_vulkan.cpu());
// 如果不相等，则展示它们的相对容差（rtol）
if (!check) {
  showRtol(c_cpu, c_vulkan.cpu());
}

// 断言 c_cpu 和 c_vulkan 几乎相等
ASSERT_TRUE(check);
TEST_F(VulkanAPITest, mul_scalar_wrapped_) {
  // 检查 Vulkan 是否可用，如果不可用则退出测试
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成一个随机张量 a_cpu，大小为 [11, 7, 139, 109]，数据类型为 float
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 a_cpu 转换为 Vulkan 张量 a_vulkan
  auto a_vulkan = a_cpu.vulkan();

  // 在 CPU 上生成一个随机标量张量 b_scalar，大小为 [1]，数据类型为 float
  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上对 a_cpu 执行原地乘法操作，即 a_cpu = a_cpu * b_scalar
  a_cpu.mul_(b_scalar);
  // 在 Vulkan 上对 a_vulkan 执行原地乘法操作，即 a_vulkan = a_vulkan * b_scalar
  a_vulkan.mul_(b_scalar);

  // 检查在 CPU 和 Vulkan 上的结果是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果结果不几乎相等，则展示相对误差并输出
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言结果几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_to_scalar_wrapped) {
  // 检查 Vulkan 是否可用，如果不可用则退出测试
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成一个随机标量张量 a，大小为 [1]，数据类型为 float
  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上生成一个随机张量 b_cpu，大小为 [11, 7, 139, 109]，数据类型为 float
  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 b_cpu 转换为 Vulkan 张量 b_vulkan
  const auto b_vulkan = b_cpu.vulkan();

  // 在 CPU 上计算标量 a 与张量 b_cpu 的乘法结果 c_cpu
  const auto c_cpu = at::mul(a, b_cpu);
  // 在 Vulkan 上计算标量 a 与张量 b_vulkan 的乘法结果 c_vulkan
  const auto c_vulkan = at::mul(a, b_vulkan);

  // 检查在 CPU 和 Vulkan 上的结果是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果结果不几乎相等，则展示相对误差并输出
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言结果几乎相等
  ASSERT_TRUE(check);
}

void test_pow(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  // 在 CPU 上生成一个形状为 input_shape 的随机张量 in_cpu，数据类型为 float
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 在 CPU 上生成一个形状为 other_shape 的随机张量 other_cpu，数据类型为 float
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  // 将 in_cpu 转换为 Vulkan 张量 in_vulkan
  const auto in_vulkan = in_cpu.vulkan();
  // 将 other_cpu 转换为 Vulkan 张量 other_vulkan
  const auto other_vulkan = other_cpu.vulkan();

  // 在 CPU 上计算 in_cpu 的 other_cpu 次幂的结果 out_cpu
  const auto out_cpu = at::pow(in_cpu, other_cpu);
  // 在 Vulkan 上计算 in_vulkan 的 other_vulkan 次幂的结果 out_vulkan
  const auto out_vulkan = at::pow(in_vulkan, other_vulkan);

  // 检查在 CPU 和 Vulkan 上的结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不几乎相等，则展示相对误差并输出，同时输出输入形状信息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "pow test failed with input shape: "
              << input_shape << " and other shape: " << other_shape << std::endl;
  }

  // 断言结果几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, pow) {
  // 测试不同形状的输入和其他张量进行幂运算
  test_pow({4}, {4});
  test_pow({4, 2}, {4, 2});
  test_pow({11, 7, 9}, {11, 7, 9});
  test_pow({3, 11, 9, 7}, {3, 11, 9, 7});
}

TEST_F(VulkanAPITest, pow_broadcast) {
  // 广播输入形状的测试用例
  test_pow({1}, {3});
  test_pow({1, 1}, {3, 2});
  test_pow({2, 1, 3}, {2, 2, 5, 3});
  test_pow({1, 1, 4}, {4, 8, 5, 4}); // mul4ch
  test_pow({3, 7, 1, 4}, {3, 7, 9, 4});

  // 广播其他形状的测试用例
  test_pow({3}, {1});
  test_pow({3, 2}, {1, 2});
  test_pow({2, 2, 5, 3}, {2, 1, 3});
  test_pow({3, 7, 9, 4}, {3, 7, 1, 4});
  test_pow({3, 8, 2, 5}, {1, 1, 2, 5}); // mul4ch

  // 同时广播输入和其他形状的测试用例
  test_pow({2, 1, 2}, {1, 5, 1});
  test_pow({5, 1, 4}, {7, 1, 2, 1});
  test_pow({2, 1, 7, 1}, {1, 5, 1, 4});
  test_pow({1, 15, 5, 4}, {21, 1, 5, 4});
  test_pow({1, 1, 5, 5}, {8, 8, 1, 1}); // mul4ch
}

TEST_F(VulkanAPITest, pow_zero_dim) {
  // 测试零维输入的幂运算
  test_pow({1, 15, 5, 4}, {});
}
void test_pow_(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  // 创建一个指定形状的随机张量，并指定为 CPU 上的 Float 类型
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 创建另一个指定形状的随机张量，并指定为 CPU 上的 Float 类型
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  // 将 CPU 张量转换为 Vulkan 张量
  const auto vulkan = cpu.vulkan();
  // 将另一个 CPU 张量转换为 Vulkan 张量
  const auto other_vulkan = other_cpu.vulkan();

  // 对 CPU 张量进行原地乘方操作，结果保存在原张量中
  cpu.pow_(other_cpu);
  // 对 Vulkan 张量进行原地乘方操作，结果保存在原张量中
  vulkan.pow_(other_vulkan);

  // 检查两个张量是否几乎相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查不通过，则展示相对误差并输出测试失败信息
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "pow_ test failed with input shape: "
              << input_shape << " and other shape: " << other_shape << std::endl;
  }

  // 使用断言确认检查通过
  ASSERT_TRUE(check);
}

// VulkanAPITest 类的 pow_ 测试用例
TEST_F(VulkanAPITest, pow_) {
  // 执行测试函数，传入形状为 {4} 的输入和 {4} 的其他张量形状
  test_pow_({4}, {4});
  // 执行测试函数，传入形状为 {4, 2} 的输入和 {4, 2} 的其他张量形状
  test_pow_({4, 2}, {4, 2});
  // 执行测试函数，传入形状为 {11, 7, 9} 的输入和 {11, 7, 9} 的其他张量形状
  test_pow_({11, 7, 9}, {11, 7, 9});
  // 执行测试函数，传入形状为 {3, 11, 9, 7} 的输入和 {3, 11, 9, 7} 的其他张量形状
  test_pow_({3, 11, 9, 7}, {3, 11, 9, 7});
}

// 对 VulkanAPITest 类的 pow_ 测试用例进行广播的 pow_ 操作测试
TEST_F(VulkanAPITest, pow_broadcast_other_) {
  // 执行测试函数，传入形状为 {3} 的输入和 {1} 的其他张量形状
  test_pow_({3}, {1});
  // 执行测试函数，传入形状为 {3, 2} 的输入和 {1, 2} 的其他张量形状
  test_pow_({3, 2}, {1, 2});
  // 执行测试函数，传入形状为 {2, 2, 5, 3} 的输入和 {2, 1, 3} 的其他张量形状
  test_pow_({2, 2, 5, 3}, {2, 1, 3});
  // 执行测试函数，传入形状为 {3, 7, 9, 4} 的输入和 {3, 7, 1, 4} 的其他张量形状
  test_pow_({3, 7, 9, 4}, {3, 7, 1, 4});
}

void test_pow_tensor_scalar(const at::IntArrayRef input_shape, const float exp) {
  // 创建一个指定形状的随机张量，并指定为 CPU 上的 Float 类型
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 对输入张量和指数进行幂运算，结果保存在输出张量中
  const auto out_cpu = at::pow(in_cpu, exp);
  // 对 Vulkan 张量和指数进行幂运算，结果保存在输出张量中
  const auto out_vulkan = at::pow(in_vulkan, exp);

  // 检查两个输出张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示相对误差并输出测试失败信息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "pow_tensor_scalar test failed with input shape: "
              << input_shape << std::endl;
  }

  // 使用断言确认检查通过
  ASSERT_TRUE(check);
}

// VulkanAPITest 类的 pow_tensor_scalar 测试用例
TEST_F(VulkanAPITest, pow_tensor_scalar) {
  // 执行测试函数，传入形状为 {4} 的输入和指数为 2.5
  test_pow_tensor_scalar({4}, 2.5);             // 1d
  // 执行测试函数，传入形状为 {4, 2} 的输入和指数为 -1
  test_pow_tensor_scalar({4, 2}, -1);           // 2d
  // 执行测试函数，传入形状为 {11, 7, 9} 的输入和指数为 7.7
  test_pow_tensor_scalar({11, 7, 9}, 7.7);      // 3d
  // 执行测试函数，传入形状为 {3, 11, 9, 7} 的输入和指数为 -0.03
  test_pow_tensor_scalar({3, 11, 9, 7}, -0.03); // 4d
}

void test_pow_tensor_scalar_(const at::IntArrayRef input_shape, const float exp) {
  // 确保输入张量的值不为 0，否则无法进行比较
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto vulkan = cpu.vulkan();

  // 对 CPU 张量进行原地幂运算，结果保存在原张量中
  cpu.pow_(exp);
  // 对 Vulkan 张量进行原地幂运算，结果保存在原张量中
  vulkan.pow_(exp);

  // 检查两个张量是否几乎相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查不通过，则展示相对误差并输出测试失败信息
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "pow_scalar_ test failed with input shape: "
              << input_shape << std::endl;
  }

  // 使用断言确认检查通过
  ASSERT_TRUE(check);
}

// VulkanAPITest 类的 pow_tensor_scalar_ 测试用例
TEST_F(VulkanAPITest, pow_tensor_scalar_) {
  // 执行测试函数，传入形状为 {4} 的输入和指数为 2.5
  test_pow_tensor_scalar_({4}, 2.5);             // 1d
  // 执行测试函数，传入形状为 {4, 2} 的输入和指数为 -1
  test_pow_tensor_scalar_({4, 2}, -1);           // 2d
  // 执行测试函数，传入形状为 {11, 7, 9} 的输入和指数为 7.7
  test_pow_tensor_scalar_({11, 7, 9}, 7.7);      // 3d
  // 执行测试函数，传入形状为 {3, 11, 9, 7} 的输入和指数为 -0.03
  test_pow_tensor_scalar_({3, 11, 9, 7}, -0.03); // 4d
}
void test_pow_scalar_tensor(const float base, const at::IntArrayRef other) {
  // 在 CPU 上生成随机张量，数据类型为 float，形状由参数 other 指定
  const auto other_cpu = at::rand(other, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto other_vulkan = other_cpu.vulkan();

  // 计算 base 的 other_cpu 次方
  const auto out_cpu = at::pow(base, other_cpu);
  // 计算 base 的 other_vulkan 次方
  const auto out_vulkan = at::pow(base, other_vulkan);

  // 检查两种计算结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    // 若结果不相等，则显示详细信息，并输出测试失败信息
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "pow_scalar_tensor test failed with other shape: "
              << other << std::endl;
  }

  // 断言两种计算结果应该是相等的
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, pow_scalar_tensor) {
  // 测试不同维度的 pow_scalar_tensor 函数
  test_pow_scalar_tensor(2.5, {4});             // 1d
  test_pow_scalar_tensor(2, {4, 2});            // 2d
  test_pow_scalar_tensor(7.7, {11, 7, 9});      // 3d
  test_pow_scalar_tensor(3, {3, 11, 9, 7});     // 4d
}

void test_floor_divide_scalar(const at::IntArrayRef input_shape, float input_scale, float other) {
  c10::InferenceMode mode;

  // 在 CPU 上生成指定形状的随机张量，数据类型为 float
  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量按照 input_scale 进行缩放
  in_cpu = at::mul(in_cpu, input_scale);

  // 将 CPU 上的张量转换为 Vulkan 张量
  auto in_vulkan = in_cpu.vulkan();
  // 在 Vulkan 上执行 floor_divide 运算
  auto out_vk = at::floor_divide(in_vulkan, other);
  // 在 CPU 上执行 floor_divide 运算
  auto out_cpu = at::floor_divide(in_cpu, other);

  // 最大容忍误差为 1.0，由于使用 floor 操作，误差控制在合理范围内
  const auto check = checkRtol(out_cpu - out_vk.cpu(), 1.0f);
  if (!check) {
    // 若结果不符合容忍误差，则输出测试失败信息
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale
              << " other: " << other
              << std::endl;
  }

  // 断言 floor_divide 运算结果应该符合容忍误差
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, floor_divide_scalar) {
  // 测试不同形状和参数的 floor_divide_scalar 函数
  test_floor_divide_scalar({3, 3, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar({12, 12}, 10.0, 3.4);
  test_floor_divide_scalar({4, 5, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar({3, 3, 12, 12}, 0.3, 0.08);
}

TEST_F(VulkanAPITest, floor_divide_scalar_error) {
  c10::InferenceMode mode;

  // 在 CPU 上生成指定形状的随机张量，数据类型为 float
  auto in_cpu = at::rand({2, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto in_vulkan = in_cpu.vulkan();
  // 预期此操作会抛出异常，因为 other 为零
  EXPECT_THROW(at::floor_divide(in_vulkan, 0.0f), ::std::exception);
}

void test_floor_divide_scalar_inplace(const at::IntArrayRef input_shape, float input_scale, float other) {
  c10::InferenceMode mode;

  // 在 CPU 上生成指定形状的随机张量，数据类型为 float
  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量按照 input_scale 进行缩放
  in_cpu = at::mul(in_cpu, input_scale);
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto in_vk = in_cpu.vulkan();

  // 在 CPU 上执行 inplace floor_divide 运算
  in_cpu.floor_divide_(other);
  // 在 Vulkan 上执行 inplace floor_divide 运算
  in_vk.floor_divide_(other);

  // 最大容忍误差为 1.0，由于使用 floor 操作，误差控制在合理范围内
  const auto check = checkRtol(in_cpu - in_vk.cpu(), 1.0f);
  if (!check) {
    // 若结果不符合容忍误差，则输出测试失败信息
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale
              << " other: " << other
              << std::endl;
  }

  // 断言 inplace floor_divide 运算结果应该符合容忍误差
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, floor_divide_scalar_inplace_error) {
  // 进入推断模式，确保操作不会影响全局状态
  c10::InferenceMode mode;

  // 创建一个在 CPU 上生成的随机张量，形状为 [2, 3, 4]，数据类型为 float
  auto in_cpu = at::rand({2, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto in_vulkan = in_cpu.vulkan();
  // 预期这个操作会抛出异常，因为除以零是不允许的
  EXPECT_THROW(in_vulkan.floor_divide(0.0f), ::std::exception);
}

TEST_F(VulkanAPITest, floor_divide_scalar_inplace) {
  // 调用测试函数来执行就地标量除法的测试，传入不同的参数
  test_floor_divide_scalar_inplace({3, 3, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar_inplace({12, 12}, 10.0, 3.4);
  test_floor_divide_scalar_inplace({4, 5, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar_inplace({3, 3, 12, 12}, 0.3, 0.08);
}

TEST_F(VulkanAPITest, floor_divide_zero_dim_tensor) {
  // 进入推断模式，确保操作不会影响全局状态
  c10::InferenceMode mode;

  // 定义输入张量的形状和缩放比例
  std::vector<int64_t> input_shape{5, 3, 4, 5};
  float input_scale = 100.0;

  // 在 CPU 上生成一个随机张量，形状为 input_shape，数据类型为 float，并乘以 input_scale
  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  in_cpu = at::mul(in_cpu, input_scale);
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto in_vk = in_cpu.vulkan();

  // 创建另一个 CPU 上的零维张量，并设置其值为 10.0f
  auto other_cpu = at::zeros({}, at::device(at::kCPU).dtype(at::kFloat)) + 10.0f;
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto other_vk = other_cpu.vulkan();

  // 对输入张量和其他张量进行按元素的地板除法运算，分别在 CPU 和 Vulkan 上执行
  auto out_cpu = at::floor_divide(in_cpu, other_cpu);
  auto out_vk = at::floor_divide(in_vk, other_vk);

  // 最大容差为 1.0，由于是地板除法
  // 可能考虑额外检查违规数目，但这应该很少见
  const auto check = checkRtol(out_cpu - out_vk.cpu(), 1.0f);
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale
              << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, floor_divide_tensor) {
  // 进入推断模式，确保操作不会影响全局状态
  c10::InferenceMode mode;

  // 定义输入张量的形状和缩放比例
  std::vector<int64_t> input_shape{6, 3, 5, 5};
  float input_scale = 10.0;

  // 在 CPU 上生成一个随机张量，形状为 input_shape，数据类型为 float，并乘以 input_scale
  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  in_cpu = at::mul(in_cpu, input_scale);
  
  // 创建另一个形状相同的随机张量，加上 0.5，以避免由极小值引起的舍入误差
  auto other_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.5;

  // 将 CPU 上的张量转换为 Vulkan 张量
  auto in_vk = in_cpu.vulkan();
  auto other_vk = other_cpu.vulkan();

  // 对输入张量和其他张量进行按元素的地板除法运算，分别在 CPU 和 Vulkan 上执行
  auto out_cpu = at::floor_divide(in_cpu, other_cpu);
  auto out_vk = at::floor_divide(in_vk, other_vk);

  // 最大容差为 1.0，由于是地板除法
  // 可能考虑额外检查违规数目，但这应该很少见
  const auto check = checkRtol(out_cpu - out_vk.cpu(), 1.0f);
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, floor_divide_tensor_inplace) {
  c10::InferenceMode mode;  // 进入推断模式

  std::vector<int64_t> input_shape{5, 3, 5, 5};  // 定义输入张量的形状
  float input_scale = 10.0;  // 定义输入的缩放比例

  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));  // 生成在 CPU 上的随机浮点数张量
  in_cpu = at::mul(in_cpu, input_scale);  // 将输入张量乘以缩放比例

  // "other" 至少为 0.5，以避免由非常小的值引起的舍入误差
  auto other_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.5;  // 生成在 CPU 上的随机浮点数张量，并添加偏移量

  auto in_vk = in_cpu.vulkan();  // 将 CPU 上的张量转换为 Vulkan 张量
  auto other_vk = other_cpu.vulkan();  // 将 CPU 上的张量转换为 Vulkan 张量

  in_cpu.floor_divide_(other_cpu);  // 在 CPU 上原地执行张量的 floor_divide 操作
  in_vk.floor_divide_(other_vk);  // 在 Vulkan 上原地执行张量的 floor_divide 操作

  // 最大容忍度为 1.0，由于执行了 floor 操作
  // 可能需要考虑额外检查违规数目，但这应该是罕见的情况
  const auto check = checkRtol(in_cpu - in_vk.cpu(), 1.0f);  // 检查在 CPU 和 Vulkan 张量之间的相对容差
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale << std::endl;  // 如果检查失败，则输出错误信息
  }

  ASSERT_TRUE(check);  // 断言检查通过
}

TEST_F(VulkanAPITest, relu) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));  // 生成在 CPU 上的随机浮点数张量
  const auto in_vulkan = in_cpu.vulkan();  // 将 CPU 上的张量转换为 Vulkan 张量

  const auto out_cpu = at::relu(in_cpu);  // 在 CPU 上执行 ReLU 操作
  const auto out_vulkan = at::relu(in_vulkan);  // 在 Vulkan 上执行 ReLU 操作

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());  // 检查 CPU 和 Vulkan 张量之间的几乎相等性
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());  // 如果检查失败，则展示相对容差信息
  }

  ASSERT_TRUE(check);  // 断言检查通过
}

TEST_F(VulkanAPITest, relu_) {
  auto a_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));  // 生成在 CPU 上的随机浮点数张量
  auto a_vulkan = a_cpu.vulkan();  // 将 CPU 上的张量转换为 Vulkan 张量

  at::relu_(a_cpu);  // 在 CPU 上原地执行 ReLU 操作
  at::relu_(a_vulkan);  // 在 Vulkan 上原地执行 ReLU 操作

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());  // 检查 CPU 和 Vulkan 张量之间的几乎相等性
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());  // 如果检查失败，则展示相对容差信息
  }

  ASSERT_TRUE(check);  // 断言检查通过
}

TEST_F(VulkanAPITest, reflection_pad2d) {
  const auto a_cpu = at::rand({2, 3, 47, 63}, at::device(at::kCPU).dtype(at::kFloat));  // 生成在 CPU 上的随机浮点数张量
  const auto a_vulkan = a_cpu.vulkan();  // 将 CPU 上的张量转换为 Vulkan 张量

  const auto out_cpu = at::reflection_pad2d(a_cpu, {9,8,5,12});  // 在 CPU 上执行反射填充操作
  const auto out_vulkan = at::reflection_pad2d(a_vulkan, {9,8,5,12}).cpu();  // 在 Vulkan 上执行反射填充操作，并转回到 CPU 张量

  const auto check = almostEqual(out_cpu, out_vulkan);  // 检查 CPU 和 Vulkan 张量之间的几乎相等性
  if (!check) {
    showRtol(out_cpu, out_vulkan);  // 如果检查失败，则展示相对容差信息
  }

  ASSERT_TRUE(check);  // 断言检查通过
}

TEST_F(VulkanAPITest, repeat_invalid_inputs_outputs_exceptions) {
  // Arrange: Vulkan repeat only supports input of dims <= 4
  {
    const auto in_cpu =
        at::rand({3, 9, 11, 7, 3}, at::device(at::kCPU).dtype(at::kFloat));  // 生成在 CPU 上的随机浮点数张量，维度超过 4
    const at::IntArrayRef repeats = {5, 7, 3, 9, 2};  // 定义重复的维度

    // Act
    EXPECT_THROW(
        { const auto out_vulkan = in_cpu.vulkan().repeat(repeats); },  // 在 Vulkan 上尝试执行重复操作，预期会抛出异常
        ::std::exception);
  }

  // Arrange: Number of dimensions of repeat dims can not be smaller than
  // number of dimensions of tensor
  {
    const auto in_cpu =
        at::rand({3, 5, 11, 13}, at::device(at::kCPU).dtype(at::kFloat));  // 生成在 CPU 上的随机浮点数张量
    const at::IntArrayRef repeats = {5, 7};  // 定义重复的维度

    // Act
    EXPECT_THROW(
        { const auto out_vulkan = in_cpu.vulkan().repeat(repeats); },  // 在 Vulkan 上尝试执行重复操作，预期会抛出异常
        ::std::exception);
  }

  // Arrange: Vulkan repeat only supports output of dims <= 4
  {
    // 创建一个包含随机数据的张量，形状为 [3, 9, 11, 7]，数据类型为浮点型，位于 CPU 上
    const auto in_cpu =
        at::rand({3, 9, 11, 7}, at::device(at::kCPU).dtype(at::kFloat));
    
    // 定义一个 IntArrayRef 类型的常量 repeats，包含五个整数元素
    const at::IntArrayRef repeats = {5, 7, 3, 9, 2};

    // 期望抛出异常
    EXPECT_THROW(
        // 尝试在 CPU 上的张量 in_cpu 上调用 vulkan() 方法，并使用 repeats 数组进行重复操作
        { const auto out_vulkan = in_cpu.vulkan().repeat(repeats); },
        // 期望捕获的异常类型为 std::exception
        ::std::exception);
void test_repeat(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef repeats) {
  c10::InferenceMode mode;  // 进入推断模式

  at::Tensor in_cpu;  // 定义 CPU 上的输入张量
  at::Tensor out_cpu;  // 定义 CPU 上的输出张量
  at::Tensor in_vulkan;  // 定义 Vulkan 上的输入张量
  at::Tensor out_vulkan;  // 定义 Vulkan 上的输出张量
  at::IntArrayRef repeat;  // 重复数组引用
  bool check = true;  // 初始化检查标志为真

  // 循环遍历输入形状的每个维度
  for (int idx_input = 1; (unsigned)idx_input < input_shape.size() + 1; ++idx_input) {
    // 循环遍历重复数组的每个元素
    for (int idx_repeat = idx_input; (unsigned)idx_repeat < repeats.size() + 1;
          ++idx_repeat) {
      // 在 CPU 上生成指定形状的随机张量
      in_cpu = at::rand(
          input_shape.slice(0, idx_input),
          at::device(at::kCPU).dtype(at::kFloat));
      // 获取部分重复数组的引用
      repeat = repeats.slice(0, idx_repeat);
      // 在 CPU 上对输入张量进行重复操作
      out_cpu = in_cpu.repeat(repeats);
      // 将 CPU 上的输入张量转换为 Vulkan 张量
      in_vulkan = in_cpu.vulkan();
      // 在 Vulkan 上对输入张量进行重复操作
      out_vulkan = in_vulkan.repeat(repeats);
      // 检查 CPU 和 Vulkan 输出张量的近似相等性
      bool local_check = almostEqual(out_cpu, out_vulkan.cpu());
      // 如果检查失败
      if (!local_check) {
        check = false;  // 设置总体检查标志为假
        // 输出失败信息，显示失败的输入形状和重复数组
        std::cout << "Repeat test failed when input is of shape "
                  << input_shape.slice(0, idx_input) << " and repeat of "
                  << repeat << std::endl;
        // 显示 CPU 和 Vulkan 输出张量的相对误差
        showRtol(out_cpu, out_vulkan.cpu());
      }
    }
  }

  // 断言总体检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, repeat) {
  // 调用测试函数，输入指定的形状和重复数组
  test_repeat({13, 5, 13, 7}, {7, 2, 3, 5});
}

TEST_F(VulkanAPITest, replication_pad2d) {
  // 生成指定形状的随机 CPU 张量
  const auto a_cpu = at::rand({2, 3, 47, 63}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto a_vulkan = a_cpu.vulkan();

  // 定义填充参数数组
  constexpr std::array<int64_t, 4u> padding_params{9, 8, 5, 12};

  // 在 CPU 上应用二维复制填充操作
  const auto out_cpu = at::replication_pad2d(a_cpu, padding_params);
  // 在 Vulkan 上应用二维复制填充操作，并将结果转换为 CPU 张量
  const auto out_vulkan = at::replication_pad2d(a_vulkan, padding_params).cpu();

  // 检查 CPU 和 Vulkan 输出张量的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan);
  // 如果检查失败
  if (!check) {
    // 显示 CPU 和 Vulkan 输出张量的相对误差
    showRtol(out_cpu, out_vulkan);
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, reshape) {
  c10::InferenceMode mode;  // 进入推断模式

  // 生成指定形状的随机 CPU 张量
  const auto in_cpu = at::rand({7, 11, 8, 9}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 定义目标形状数组
  const std::array<int64_t, 2> shape{7 * 8, 11 * 9};

  // 在 CPU 上对输入张量进行形状重塑操作
  const auto out_cpu = at::reshape(in_cpu, shape);
  // 在 Vulkan 上对输入张量进行形状重塑操作
  const auto out_vulkan = at::reshape(in_vulkan, shape);

  // 检查 CPU 和 Vulkan 输出张量的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查失败
  if (!check) {
    // 显示 CPU 和 Vulkan 输出张量的相对误差
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, reshape_) {
  c10::InferenceMode mode;  // 进入推断模式

  // 生成指定形状的随机 CPU 张量
  const auto cpu = at::rand({9, 4, 12, 6}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto vulkan = cpu.vulkan();

  // 定义目标形状数组
  const std::array<int64_t, 3> shape{9, 4 * 6, 12};

  // 在 CPU 上对输入张量进行形状重塑操作
  cpu.reshape(shape);
  // 在 Vulkan 上对输入张量进行形状重塑操作
  vulkan.reshape(shape);

  // 检查 CPU 和 Vulkan 张量的相等性
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查失败
  if (!check) {
    // 显示 CPU 和 Vulkan 输出张量的相对误差
    showRtol(cpu, vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}
void test_select(const at::IntArrayRef input_shape, int64_t dim, int64_t index) {
  // 创建指定形状的随机张量在 CPU 上
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 从输入张量中选择指定维度和索引的切片在 CPU 上
  const auto out_cpu = at::select(in_cpu, dim, index);

  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 从 Vulkan 张量中选择指定维度和索引的切片
  const auto out_vulkan = at::select(in_vulkan, dim, index);

  // 检查两个选择操作的结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不几乎相等，则显示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个选择操作的结果应当几乎相等
  ASSERT_TRUE(check);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（深度小）
TEST_F(VulkanAPITest, select_3d_depth_small) {
  test_select({1, 1, 1}, 0, 0);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（深度中等）
TEST_F(VulkanAPITest, select_3d_depth_medium) {
  test_select({3, 2, 5}, 0, 2);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（深度较大）
TEST_F(VulkanAPITest, select_3d_depth_large) {
  test_select({100, 1, 144}, 0, 50);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（高度小）
TEST_F(VulkanAPITest, select_3d_height_small) {
  test_select({1, 1, 1}, 1, 0);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（高度中等）
TEST_F(VulkanAPITest, select_3d_height_medium) {
  test_select({3, 5, 2}, 1, 2);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（高度中等1）
TEST_F(VulkanAPITest, select_3d_height_medium1) {
  test_select({16, 16, 5}, 1, 6);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（高度中等2）
TEST_F(VulkanAPITest, select_3d_height_medium2) {
  test_select({17, 17, 5}, 1, 6);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（高度较大）
TEST_F(VulkanAPITest, select_3d_height_large) {
  test_select({100, 144, 5}, 1, 50);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（宽度小）
TEST_F(VulkanAPITest, select_3d_width_small) {
  test_select({1, 1, 1}, 2, 0);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（宽度中等）
TEST_F(VulkanAPITest, select_3d_width_medium) {
  test_select({3, 5, 3}, 2, 2);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（宽度中等2）
TEST_F(VulkanAPITest, select_3d_width_medium2) {
  test_select({17, 17, 8}, 2, 6);
}

// 在 VulkanAPI 测试框架中测试三维张量的选择操作（宽度较大）
TEST_F(VulkanAPITest, select_3d_width_large) {
  test_select({100, 3, 144}, 2, 50);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（批次小）
TEST_F(VulkanAPITest, select_4d_batch_small) {
  test_select({1, 1, 1, 1}, 0, 0);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（批次中等）
TEST_F(VulkanAPITest, select_4d_batch_medium) {
  test_select({3, 2, 5, 4}, 0, 1);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（批次较大）
TEST_F(VulkanAPITest, select_4d_batch_large) {
  test_select({30, 8, 12, 17}, 0, 27);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（深度小）
TEST_F(VulkanAPITest, select_4d_depth_small) {
  test_select({1, 1, 1, 1}, 1, 0);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（深度中等）
TEST_F(VulkanAPITest, select_4d_depth_medium) {
  test_select({7, 5, 2, 4}, 1, 4);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（深度较大）
TEST_F(VulkanAPITest, select_4d_depth_large) {
  test_select({5, 30, 12, 30}, 1, 23);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（高度小）
TEST_F(VulkanAPITest, select_4d_height_small) {
  test_select({1, 1, 1, 1}, 2, 0);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（高度中等）
TEST_F(VulkanAPITest, select_4d_height_medium) {
  test_select({3, 5, 4, 2}, 2, 3);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（高度较大）
TEST_F(VulkanAPITest, select_4d_height_large) {
  test_select({5, 8, 50, 50}, 2, 41);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（宽度小）
TEST_F(VulkanAPITest, select_4d_width_small) {
  test_select({1, 1, 1, 1}, 3, 0);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（宽度中等）
TEST_F(VulkanAPITest, select_4d_width_medium) {
  test_select({3, 5, 4, 2}, 3, 1);
}

// 在 VulkanAPI 测试框架中测试四维张量的选择操作（宽度较大）
TEST_F(VulkanAPITest, select_4d_width_large) {
  test_select({5, 8, 50, 50}, 3, 33);
}

// 在 VulkanAPI 测试框架中测试 sigmoid 激活函数的实现
TEST_F(VulkanAPITest, sigmoid) {
  // 创建指定形状的随机张量在 CPU 上
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 计算 CPU 上张量的 sigmoid 激活
  const auto out_cpu = at::sigmoid(in_cpu);
  // 计算 Vulkan 张量的 sigmoid 激活
  const auto out_vulkan = at::sigmoid(in_vulkan);

  // 检查两个 sigmoid 操作的结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不几乎相等，则显示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
TEST_F(VulkanAPITest, sigmoid_) {
  // 创建一个形状为 [17, 197, 302, 5] 的随机浮点数张量，存储在 CPU 上
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto vulkan = cpu.vulkan();

  // 在 CPU 上应用 sigmoid_ 操作
  at::sigmoid_(cpu);
  // 在 Vulkan 上应用 sigmoid_ 操作
  at::sigmoid_(vulkan);

  // 检查在 CPU 和 Vulkan 张量上应用 sigmoid_ 后的近似相等性
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查不通过，显示相对误差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_log_softmax_underflow_exception) {
  // 我们在张量 [20, 0] 上依次应用 softmax 和 log 操作
  // 在 CPU 上的 softmax 输出为 [1.0000e+00, 2.0612e-09]
  // 在 Vulkan 上的 softmax 输出为 [1, 0]，因为 2.0612e-09 小于 5.96e−8
  // 我们期望在应用 log 操作时看到 nan 或 -inf
  float data[] = {20, 0};
  const auto in_cpu = at::from_blob(data, {2}, at::kFloat);
  const auto in_vulkan = in_cpu.vulkan();

  // 在 CPU 和 Vulkan 上应用 softmax 操作
  const auto softmax_out_cpu = at::softmax(in_cpu, 0);
  const auto softmax_out_vulkan = at::softmax(in_vulkan, 0);

  // 在 CPU 和 Vulkan 上应用 log 操作
  const auto log_out_cpu = at::log(softmax_out_cpu);
  const auto log_out_vulkan = at::log(softmax_out_vulkan);

  // 检查 Vulkan 上的 log 输出是否包含 nan 或 inf
  auto has_nan = log_out_vulkan.cpu().isnan().any().item().to<bool>();
  auto has_inf = log_out_vulkan.cpu().isinf().any().item().to<bool>();

  // 我们期望 log 的输出包含 nan 或 inf
  const auto check = has_nan || has_inf;
  // 如果检查不通过，显示 Vulkan 上的 log 输出
  if (!check) {
    std::cout << "expect log_out_vulkan contains nan or inf, but got" << std::endl;
    std::cout << log_out_vulkan.cpu() << std::endl;
  }
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, log_softmax_underflow) {
  // Vulkan 上 float16 的最小严格正常数值为 2−24 ≈ 5.96 × 10^−8
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
  // 最小可表示的 log 值为 -16.64，`log_softmax` 实现在应用 `log` 前会将输出 softmax 结果加上 6e-8
  // 以处理下溢，因此不会像上述 `log_softmax_underflow_exception` 测试中显示 nan 或 -inf
  float smallest_representable_log = -16.64f;
  float data[] = {20, 0};
  const auto in_cpu = at::from_blob(data, {2}, at::kFloat);
  const auto in_vulkan = in_cpu.vulkan();

  // 在 CPU 和 Vulkan 上应用 log_softmax 操作
  const auto log_softmax_cpu = at::log_softmax(in_cpu, 0);
  const auto log_softmax_vulkan = at::log_softmax(in_vulkan, 0);

  // 检查 CPU 和 Vulkan 上 log_softmax 结果之间的相对误差
  const auto check = checkRtol(log_softmax_cpu - log_softmax_vulkan.cpu(), -smallest_representable_log);
  // 如果检查不通过，显示相对误差
  if (!check) {
    showRtol(log_softmax_cpu, log_softmax_vulkan.cpu());
  }
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

void test_softmax(const at::IntArrayRef shape, bool log_softmax = false) {
  // 在 CPU 上创建一个形状为 shape 的随机浮点数张量
  at::Tensor in_cpu =
      at::rand(shape, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const at::Tensor in_vulkan = in_cpu.vulkan();

  // 将 shape 的长度转换为 int64_t 类型，用于测试负索引
  int64_t size = static_cast<int64_t>(shape.size());

  // 在所有维度上测试
  for (auto dim = -size; dim < size; dim++) {
    # 计算输入张量的 softmax 或 log_softmax，根据 log_softmax 变量的布尔值选择不同的操作
    const at::Tensor out_cpu = log_softmax ? at::log_softmax(in_cpu, dim) : at::softmax(in_cpu, dim);
    
    # 计算输入张量的 softmax 或 log_softmax，根据 log_softmax 变量的布尔值选择不同的操作，使用 Vulkan 后端进行计算
    const at::Tensor out_vulkan = log_softmax ? at::log_softmax(in_vulkan, dim)
                                              : at::softmax(in_vulkan, dim);
    
    # 检查两个张量 out_cpu 和 out_vulkan 在 CPU 上是否几乎相等
    const bool check = almostEqual(out_cpu, out_vulkan.cpu());
    
    # 如果 check 不为真，则打印失败消息，指示在特定轴上（axis dim）的 Softmax 测试失败，并列出张量的维度
    if (!check) {
      std::cout << "Softmax test failed on axis " << dim << " for tensor dims {";
      for (uint32_t place = 0; place < shape.size() - 1; place++) {
        std::cout << shape[place] << " ";
      }
      std::cout << shape.back() << "}" << std::endl;
    
      # 调用 showRtol 函数显示两个张量的相对容差，这是自定义的显示函数
      showRtol(out_cpu, out_vulkan.cpu());
    }
    
    # 断言 check 必须为真，如果不是，会触发断言失败
    ASSERT_TRUE(check);
}

// 定义 VulkanAPITest 类中的 softmax 测试方法
TEST_F(VulkanAPITest, softmax) {
  // 进入推断模式
  c10::InferenceMode mode;
  // 定义测试输入维度的向量数组
  std::vector<std::vector<int64_t>> test_in_dims = {
      {1, 3, 4, 2},
      {4, 8, 5, 7},
      {9, 11, 12, 12},
  };
  // 是否进行 log_softmax 计算，初始化为 false
  bool log_softmax = false;
  // 遍历每个测试输入维度向量
  for (const std::vector<int64_t>& dim_vec : test_in_dims) {
    // 遍历截断点
    for (uint32_t trunc = 0; trunc < dim_vec.size(); trunc++) {
      // 截取维度向量的子集
      const std::vector<int64_t> trunc_dim_vec =
          std::vector<int64_t>(dim_vec.begin(), dim_vec.end() - trunc);
      // 调用测试函数 test_softmax 进行 softmax 计算
      test_softmax(trunc_dim_vec, log_softmax);
    }
  }
}

// 定义 VulkanAPITest 类中的 log_softmax 测试方法（禁用状态）
TEST_F(VulkanAPITest, DISABLED_log_softmax) {
  // 进入推断模式
  c10::InferenceMode mode;
  // 定义测试输入维度的向量数组
  std::vector<std::vector<int64_t>> test_in_dims = {
      {1, 3, 4, 2},
      {4, 8, 5, 7},
      {9, 11, 12, 12},
  };
  // 是否进行 log_softmax 计算，初始化为 true
  bool log_softmax = true;
  // 遍历每个测试输入维度向量
  for (const std::vector<int64_t>& dim_vec : test_in_dims) {
    // 遍历截断点
    for (uint32_t trunc = 0; trunc < dim_vec.size(); trunc++) {
      // 截取维度向量的子集
      const std::vector<int64_t> trunc_dim_vec =
          std::vector<int64_t>(dim_vec.begin(), dim_vec.end() - trunc);
      // 调用测试函数 test_softmax 进行 log_softmax 计算
      test_softmax(trunc_dim_vec, log_softmax);
    }
  }
}

// 定义 VulkanAPITest 类中的 abs 测试方法
TEST_F(VulkanAPITest, abs) {
  // 生成随机的 CPU 张量作为输入
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  // 转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 计算 CPU 和 Vulkan 张量的绝对值
  const auto out_cpu = at::abs(in_cpu);
  const auto out_vulkan = at::abs(in_vulkan);

  // 检查 CPU 和 Vulkan 计算结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则显示相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 计算结果近似相等
  ASSERT_TRUE(check);
}

// 定义 VulkanAPITest 类中的 abs_ 测试方法
TEST_F(VulkanAPITest, abs_) {
  // 生成随机的 CPU 张量作为输入
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  // 转换为 Vulkan 张量
  auto vulkan = cpu.vulkan();

  // 对 CPU 和 Vulkan 张量执行原位计算绝对值
  at::abs_(cpu);
  at::abs_(vulkan);

  // 检查 CPU 和 Vulkan 张量是否在相对容差范围内近似相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果结果不近似相等，则显示相对容差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 计算结果近似相等
  ASSERT_TRUE(check);
}

// 定义 VulkanAPITest 类中的 tanh 测试方法
TEST_F(VulkanAPITest, tanh) {
  // 生成随机的 CPU 张量作为输入
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  // 转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 计算 CPU 和 Vulkan 张量的双曲正切
  const auto out_cpu = at::tanh(in_cpu);
  const auto out_vulkan = at::tanh(in_vulkan);

  // 检查 CPU 和 Vulkan 计算结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则显示相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 计算结果近似相等
  ASSERT_TRUE(check);
}

// 定义 VulkanAPITest 类中的 tanh_ 测试方法
TEST_F(VulkanAPITest, tanh_) {
  // 生成随机的 CPU 张量作为输入
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  // 转换为 Vulkan 张量
  auto vulkan = cpu.vulkan();

  // 对 CPU 和 Vulkan 张量执行原位计算双曲正切
  at::tanh_(cpu);
  at::tanh_(vulkan);

  // 检查 CPU 和 Vulkan 张量是否在相对容差范围内近似相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果结果不近似相等，则显示相对容差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 计算结果近似相等
  ASSERT_TRUE(check);
}
void test_sub(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape, float alpha) {
  // 使用随机数生成输入张量，数据类型为浮点数，在 CPU 上进行生成
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 使用随机数生成其他张量，数据类型为浮点数，在 CPU 上进行生成
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 将 CPU 上的其他张量转换为 Vulkan 张量
  const auto other_vulkan = other_cpu.vulkan();

  // 在 CPU 上计算输入张量和其他张量的差，乘以一个标量 alpha
  const auto out_cpu = at::sub(in_cpu, other_cpu, alpha);
  // 在 Vulkan 上计算输入张量和其他张量的差，乘以一个标量 alpha
  const auto out_vulkan = at::sub(in_vulkan, other_vulkan, alpha);

  // 检查在 CPU 和 Vulkan 计算出的结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不相等，则显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用断言确保 CPU 和 Vulkan 计算出的结果几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub) {
  // 测试函数，对具有相同形状的张量进行测试，标量 alpha 为 2.1
  test_sub({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);
}

TEST_F(VulkanAPITest, sub_broadcast0) {
  // 测试函数，对张量进行广播操作，其中一个张量的形状为 {3, 5, 179, 221}，另一个为 {3, 5, 1, 1}，标量 alpha 为 1.8
  test_sub({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);
}

// 以下测试函数类似地对其他广播情况进行测试，直到最后一个测试函数为止
    // 返回空，结束当前函数的执行
    return;
  }

  // 使用 ATen 创建一个形状为 [13, 23, 59, 73] 的随机张量，位于 CPU 上，数据类型为 float
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto a_vulkan = a_cpu.vulkan();

  // 定义一个 float 类型的标量 b_scalar，赋值为 3.1415
  const float b_scalar = 3.1415f;

  // 对 a_cpu 和 b_scalar 进行按元素减法运算，缩放因子分别为 2.1
  const auto c_cpu = at::sub(a_cpu, b_scalar, 2.1f);
  // 对 a_vulkan 和 b_scalar 进行按元素减法运算，缩放因子分别为 2.1
  const auto c_vulkan = at::sub(a_vulkan, b_scalar, 2.1f);

  // 检查 c_cpu 和 c_vulkan 是否几乎相等，返回布尔值
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果检查结果为假（不相等），显示 c_cpu 和 c_vulkan.cpu() 的相对容差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言 check 为真，如果为假则会终止程序并输出错误信息
  ASSERT_TRUE(check);
TEST_F(VulkanAPITest, sub_scalar_) {
  // 如果 Vulkan 不可用，则退出测试
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成大小为 [47, 2, 23, 97] 的随机张量
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 上的张量转换为 Vulkan 张量
  auto a_vulkan = a_cpu.vulkan();

  // 定义标量 b_scalar 为 3.1415
  const float b_scalar = 3.1415f;

  // 在 CPU 上对张量 a_cpu 进行就地减法操作，减去 b_scalar 和 2.1
  a_cpu.sub_(b_scalar, 2.1f);
  // 在 Vulkan 上对张量 a_vulkan 进行就地减法操作，减去 b_scalar 和 2.1
  a_vulkan.sub_(b_scalar, 2.1f);

  // 检查 Vulkan 张量和 CPU 张量的减法结果是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果结果不几乎相等，则显示相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言减法结果几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_scalar_wrapped) {
  // 如果 Vulkan 不可用，则退出测试
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成大小为 [13, 23, 59, 73] 的随机张量，并将其转换为 Vulkan 张量
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  // 在 CPU 上生成大小为 [1] 的随机标量张量 b_scalar
  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上计算张量 a_cpu 减去 b_scalar 和 2.1 的结果，并转换为 Vulkan 张量
  const auto c_cpu = at::sub(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::sub(a_vulkan, b_scalar, 2.1f);

  // 检查 Vulkan 张量和 CPU 张量的减法结果是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果结果不几乎相等，则显示相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言减法结果几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_scalar_wrapped_) {
  // 如果 Vulkan 不可用，则退出测试
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成大小为 [47, 2, 23, 97] 的随机张量，并将其转换为 Vulkan 张量
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  // 在 CPU 上生成大小为 [1] 的随机标量张量 b_scalar
  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上对张量 a_cpu 进行就地减法操作，减去 b_scalar 和 2.1
  a_cpu.sub_(b_scalar, 2.1f);
  // 在 Vulkan 上对张量 a_vulkan 进行就地减法操作，减去 b_scalar 和 2.1
  a_vulkan.sub_(b_scalar, 2.1f);

  // 检查 Vulkan 张量和 CPU 张量的减法结果是否几乎相等
  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  // 如果结果不几乎相等，则显示相对误差
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  // 断言减法结果几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_to_scalar_wrapped) {
  // 如果 Vulkan 不可用，则退出测试
  if (!at::is_vulkan_available()) {
    return;
  }

  // 在 CPU 上生成大小为 [1] 的随机张量 a
  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上生成大小为 [11, 7, 139, 109] 的随机张量 b_cpu，并将其转换为 Vulkan 张量 b_vulkan
  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  // 在 CPU 上计算张量 a 减去 b_cpu 和 2.1 的结果，并转换为 Vulkan 张量
  const auto c_cpu = at::sub(a, b_cpu, 2.1f);
  const auto c_vulkan = at::sub(a, b_vulkan, 2.1f);

  // 检查 Vulkan 张量和 CPU 张量的减法结果是否几乎相等
  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  // 如果结果不几乎相等，则显示相对误差
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  // 断言减法结果几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sum_invalid_inputs) {
  // 进入推断模式
  c10::InferenceMode mode;

  // Act: 输入维度过大，预期抛出异常
  EXPECT_THROW({
    at::sum(at::rand({3, 5, 7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // Act: 维度超出范围，预期抛出异常
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // Act: 维度超出范围，预期抛出异常
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {-4});
  }, ::std::exception);

  // Act: 重复的维度，预期抛出异常
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, 1});
  }, ::std::exception);

  // Act: 重复的维度，预期抛出异常
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, -2});
  }, ::std::exception);
}
// 测试函数，计算给定形状和维度列表的和，并比较 Vulkan 和 CPU 的结果
void test_sum_dim(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list, bool keepdim=false) {
  // 在 CPU 上生成随机张量
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 在指定维度上对 CPU 张量求和
  const auto out_cpu = at::sum(in_cpu, dim_list, keepdim);
  // 在指定维度上对 Vulkan 张量求和
  const auto out_vulkan = at::sum(in_vulkan, dim_list, keepdim);

  // 检查 Vulkan 和 CPU 结果是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不相等，输出错误信息和相对容差
  if (!check) {
    std::cout << "sum_dim test failed with input shape: "
              << input_shape << " and dim_list: " << dim_list << std::endl;
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言结果相等
  ASSERT_TRUE(check);
}

// 测试 Vulkan API 的一维求和
TEST_F(VulkanAPITest, sum_dim_1d) {
  test_sum_dim({7}, {-1});
  test_sum_dim({3}, {0});
}

// 测试 Vulkan API 的二维求和
TEST_F(VulkanAPITest, sum_dim_2d) {
  test_sum_dim({2, 3}, {-1});
  test_sum_dim({2, 7}, {-2});
  test_sum_dim({2, 7}, {-1, -2});
}

// 测试 Vulkan API 的三维求和
TEST_F(VulkanAPITest, sum_dim_3d) {
  test_sum_dim({9, 7, 5}, {-1});
  test_sum_dim({5, 7, 9}, {-2});
  test_sum_dim({5, 7, 9}, {-3});

  test_sum_dim({10, 7, 5}, {0, 1});
  test_sum_dim({10, 7, 5}, {0, 2});
  test_sum_dim({10, 7, 5}, {1, 2});

  test_sum_dim({10, 7, 5}, {-1, -2});
  test_sum_dim({10, 7, 5}, {-1, -3});
  test_sum_dim({10, 7, 5}, {-2, -3});

  test_sum_dim({10, 7, 5}, {0, 1, 2});
  test_sum_dim({10, 7, 5}, {-1, -2, -3});
}

// 测试 Vulkan API 的四维求和
TEST_F(VulkanAPITest, sum_dim_4d) {
  test_sum_dim({7, 9, 6, 5}, {-1});
  test_sum_dim({6, 5, 7, 9}, {-2});
  test_sum_dim({6, 5, 7, 9}, {-3});
  test_sum_dim({6, 5, 7, 9}, {-4});

  test_sum_dim({10, 7, 5, 6}, {0, 1});
  test_sum_dim({10, 7, 5, 6}, {0, 2});
  test_sum_dim({10, 7, 5, 6}, {0, 3});
  test_sum_dim({10, 7, 5, 6}, {1, 2});
  test_sum_dim({10, 7, 5, 6}, {1, 3});
  test_sum_dim({10, 7, 5, 6}, {2, 3});
  test_sum_dim({10, 7, 5, 6}, {-2, -4});

  test_sum_dim({10, 7, 5, 6}, {0, 1, 2});
  test_sum_dim({10, 7, 5, 6}, {0, 1, 3});
  test_sum_dim({10, 7, 5, 6}, {0, 2, 3});
  test_sum_dim({10, 7, 5, 6}, {3, 2, 1});
  test_sum_dim({10, 7, 5, 6}, {3, -2, 1});
  test_sum_dim({10, 7, 5, 6}, {-3, -2, -1});

  test_sum_dim({10, 7, 5, 6}, {-1, -2, -3});
  test_sum_dim({10, 7, 5, 6}, {-1, -2, -4});
  test_sum_dim({10, 7, 5, 6}, {-1, -3, -4});
  test_sum_dim({10, 7, 5, 6}, {-2, -3, -4});

  test_sum_dim({10, 7, 5, 6}, {-1, -2, -3, -4});
}

// 测试 Vulkan API 的一维求和（保留维度）
TEST_F(VulkanAPITest, sum_dim_keepdim_1d) {
  test_sum_dim({5}, {-1}, true);
  test_sum_dim({3}, {-1}, true);
}

// 测试 Vulkan API 的二维求和（保留维度）
TEST_F(VulkanAPITest, sum_dim_keepdim_2d) {
  test_sum_dim({5, 7}, {-1}, true);
  test_sum_dim({5, 7}, {-2}, true);
}

// 测试 Vulkan API 的三维求和（保留维度）
TEST_F(VulkanAPITest, sum_dim_keepdim_3d) {
  test_sum_dim({9, 5, 7}, {-1}, true);
  test_sum_dim({5, 9, 7}, {-2}, true);
  test_sum_dim({7, 9, 5}, {-3}, true);

  test_sum_dim({9, 5, 7}, {0, 1}, true);
  test_sum_dim({5, 9, 7}, {0, 2}, true);
  test_sum_dim({7, 9, 5}, {1, 2}, true);

  test_sum_dim({7, 9, 5}, {0, 1, 2}, true);
}
TEST_F(VulkanAPITest, sum_dim_keepdim_4d) {
  // 测试在4维张量上对指定维度进行求和，保持维度信息
  test_sum_dim({9, 5, 7, 11}, {-1}, true);
  // 测试在4维张量上对指定维度进行求和，保持维度信息
  test_sum_dim({5, 9, 11, 7}, {-2}, true);
  // 测试在4维张量上对指定维度进行求和，保持维度信息
  test_sum_dim({7, 11, 9, 5}, {-3}, true);
  // 测试在4维张量上对指定维度进行求和，保持维度信息
  test_sum_dim({11, 7, 9, 5}, {-4}, true);

  // 测试在4维张量上同时对多个指定维度进行求和，保持维度信息
  test_sum_dim({9, 5, 7, 11}, {0, 1}, true);
  // 测试在4维张量上同时对多个指定维度进行求和，保持维度信息
  test_sum_dim({5, 9, 11, 7}, {0, 2}, true);
  // 测试在4维张量上同时对多个指定维度进行求和，保持维度信息
  test_sum_dim({7, 11, 9, 5}, {0, 3}, true);
  // 测试在4维张量上同时对多个指定维度进行求和，保持维度信息
  test_sum_dim({11, 7, 9, 5}, {1, 2}, true);
  // 测试在4维张量上同时对多个指定维度进行求和，保持维度信息
  test_sum_dim({9, 5, 7, 11}, {1, 3}, true);
  // 测试在4维张量上同时对多个指定维度进行求和，保持维度信息
  test_sum_dim({5, 9, 11, 7}, {2, 3}, true);

  // 测试在4维张量上对负索引指定的多个维度进行求和，保持维度信息
  test_sum_dim({7, 11, 9, 5}, {-1, -2, -3}, true);
  // 测试在4维张量上对负索引指定的多个维度进行求和，保持维度信息
  test_sum_dim({11, 7, 9, 5}, {-1, -2, -4}, true);
  // 测试在4维张量上对负索引指定的多个维度进行求和，保持维度信息
  test_sum_dim({9, 5, 7, 11}, {-2, -3, -4}, true);

  // 测试在4维张量上对负索引指定的所有维度进行求和，保持维度信息
  test_sum_dim({9, 5, 7, 11}, {-1, -2, -3, -4}, true);
}

void test_sum(const at::IntArrayRef input_shape) {
  // 在CPU上生成指定形状的随机张量
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 转换为Vulkan张量
  const auto in_vulkan = in_cpu.vulkan();

  // 计算CPU上张量的总和
  const auto out_cpu = at::sum(in_cpu);
  // 计算Vulkan张量的总和
  const auto out_vulkan = at::sum(in_vulkan);

  // 断言：Vulkan张量的维度应为0
  ASSERT_TRUE(out_vulkan.dim() == 0);
  // 检查CPU和Vulkan张量的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "sum test failed with input shape: "
              << input_shape << std::endl;
    // 显示CPU和Vulkan张量的相对误差
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言：检查通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sum_test) {
  // 测试不同形状的张量的总和计算
  test_sum({6});
  test_sum({5, 6});
  test_sum({0, 3, 1});
  test_sum({5, 0, 1});
  test_sum({5, 3, 0});
  test_sum({3, 3, 1});
  test_sum({7, 6, 6});
  test_sum({7, 8, 5, 6});
}

void test_uniform(at::Tensor a_vulkan, const float a_min, const float a_max) {
  // 将Vulkan张量转换为CPU张量
  auto a_cpu = a_vulkan.cpu();
  // 断言：CPU张量的最大值不超过a_max
  ASSERT_TRUE(a_cpu.max().item<float>() <= a_max);
  // 断言：CPU张量的最小值不低于a_min
  ASSERT_TRUE(a_cpu.min().item<float>() >= a_min);

  // 定义均匀分布的范围并生成CPU张量
  float b_min = 0.0f;
  float b_max = 10.0f;
  auto b_vulkan =
      at::rand({80, 7, 12, 10}, at::device(at::kCPU).dtype(at::kFloat))
          .vulkan();
  // 将生成的CPU张量转换为Vulkan张量并进行均匀分布设置
  b_vulkan.uniform_(b_min, b_max);
  auto b_cpu = b_vulkan.cpu();

  // 定义直方图的bin数量
  int bins = 10;
  // 计算CPU张量的直方图
  auto b_hist_tuple = at::histogram(b_cpu, bins);

  // 计算期望每个bin的元素数量
  int64_t expected_per_bin = b_vulkan.numel() / bins;
  auto b_hist = std::get<0>(b_hist_tuple);

  // 松散定义均匀性检查：如果所有的bin元素数量都在期望值的5%范围内，则通过
  ASSERT_TRUE(
      (b_hist - expected_per_bin).abs().max().item<float>() <=
      (expected_per_bin * 0.05));
}

TEST_F(VulkanAPITest, uniform) {
  // 定义均匀分布的范围
  float a_min = -8.2f;
  float a_max = -1.4f;
  // 在CPU上生成指定形状的随机张量并转换为Vulkan张量
  auto a_vulkan =
      at::rand({8, 7, 12, 10}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  // 对Vulkan张量应用均匀分布
  a_vulkan.uniform_(a_min, a_max);
  // 测试均匀性
  test_uniform(a_vulkan, a_min, a_max);
}
TEST_F(VulkanAPITest, rand_like) {
  // 设置随机数生成的范围
  float a_min = 0.0f;
  float a_max = 1.0f;
  // 创建一个 Vulkan 张量，并初始化为全零
  auto a_vulkan =
      at::zeros({8, 7, 12, 10}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  // 使用 rand_like 在 a_vulkan 上生成随机数
  const auto out_vulkan = at::rand_like(a_vulkan);
  // 验证输入张量仍全为零（非原地操作）
  ASSERT_TRUE(at::mean(a_vulkan.cpu()).item<float>() == 0.0);
  // 测试生成的随机数是否满足均匀分布要求
  test_uniform(out_vulkan, a_min, a_max);
}

void test_normal(at::Tensor out_vulkan, const float mean, const float std) {
  // 验证分布是否为正态分布。生成的均值与给定均值之间的差应该在标准差的5%以内，标准差同理。
  ASSERT_TRUE(std::abs(at::mean(out_vulkan.cpu()).item<float>() - mean) < std::abs(std) * 0.05);
  ASSERT_TRUE(std::abs(at::std(out_vulkan.cpu()).item<float>() - std) < std::abs(std) * 0.05);
}

TEST_F(VulkanAPITest, normal_) {
  // 设置正态分布的均值和标准差
  float a_mean = -10.0;
  float a_std = 2.0;

  // 创建一个 Vulkan 张量，并初始化为全零
  auto a_vulkan =
      at::zeros({3, 4, 5, 6}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  // 在 a_vulkan 上应用 normal_ 方法生成正态分布的随机数
  a_vulkan.normal_(a_mean, a_std);

  // 测试生成的随机数是否满足正态分布要求
  test_normal(a_vulkan, a_mean, a_std);
}

TEST_F(VulkanAPITest, normal_large) {
  // 设置正态分布的均值和标准差
  float a_mean = 1.0;
  float a_std = 0.01;

  // 创建一个 Vulkan 张量，并初始化为全零
  auto a_vulkan =
      at::zeros({30, 40, 50, 60}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  // 在 a_vulkan 上应用 normal_ 方法生成正态分布的随机数
  a_vulkan.normal_(a_mean, a_std);

  // 测试生成的随机数是否满足正态分布要求
  test_normal(a_vulkan, a_mean, a_std);
}

TEST_F(VulkanAPITest, normal_error) {
  // 设置错误的标准差（负值）
  float a_mean = 1.0;
  float a_std = -1;

  // 创建一个 Vulkan 张量，并初始化为全零
  auto a_vulkan =
      at::zeros({30, 40, 50, 60}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  // 预期 normal_ 方法会抛出异常
  EXPECT_THROW(a_vulkan.normal_(a_mean, a_std), ::std::exception);
}

TEST_F(VulkanAPITest, randn_like) {
  // 设置正态分布的均值和标准差
  float a_mean = 0.0;
  float a_std = 1.0;

  // 创建一个 Vulkan 张量，并初始化为全零
  auto a_vulkan =
      at::zeros({8, 7, 6, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  // 使用 randn_like 在 a_vulkan 上生成正态分布的随机数
  const auto out_vulkan = at::randn_like(a_vulkan);
  // 验证输入张量仍全为零（非原地操作）
  ASSERT_TRUE(at::mean(a_vulkan.cpu()).item<float>() == 0.0);
  // 测试生成的随机数是否满足正态分布要求
  test_normal(out_vulkan, a_mean, a_std);
}

TEST_F(VulkanAPITest, randn_like_large) {
  // 设置正态分布的均值和标准差
  float a_mean = 0.0;
  float a_std = 1.0;

  // 创建一个 Vulkan 张量，并初始化为全零
  auto a_vulkan =
      at::zeros({80, 70, 60, 50}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  // 使用 randn_like 在 a_vulkan 上生成正态分布的随机数
  const auto out_vulkan = at::randn_like(a_vulkan);

  // 测试生成的随机数是否满足正态分布要求
  test_normal(out_vulkan, a_mean, a_std);
}

void test_t(const at::IntArrayRef input_shape) {
  // 在 CPU 上生成指定形状的随机数张量
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 对 CPU 上的张量进行转置操作
  const auto out_cpu = at::t(in_cpu);

  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 在 Vulkan 张量上应用转置操作
  const auto out_vulkan = at::t(in_vulkan);

  // 检查 Vulkan 张量和 CPU 张量的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，展示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 Vulkan 张量和 CPU 张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, transpose_t_1d) {
  // 测试一维张量的转置操作
  test_t({7});
}

TEST_F(VulkanAPITest, transpose_t_2d_small) {
  // 测试较小的二维张量的转置操作
  test_t({1, 1});
}

TEST_F(VulkanAPITest, transpose_t_2d_medium) {
  // 测试中等大小的二维张量的转置操作
  test_t({7, 5});
}

TEST_F(VulkanAPITest, transpose_t_2d_large) {
  // 测试较大的二维张量的转置操作
  test_t({53, 117});
}
// 定义函数 test_transpose，用于测试矩阵转置操作
void test_transpose(const at::IntArrayRef input_shape, int64_t index0, int64_t index1) {
  // 创建随机生成的 CPU 上的浮点数张量 in_cpu，形状由 input_shape 指定
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 对 in_cpu 执行转置操作，index0 和 index1 指定了转置的维度
  const auto out_cpu = at::transpose(in_cpu, index0, index1);

  // 获取 in_cpu 在 Vulkan 上的表示
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 上的张量 in_vulkan 执行转置操作，index0 和 index1 指定了转置的维度
  const auto out_vulkan = at::transpose(in_vulkan, index0, index1);

  // 检查 out_cpu 和 out_vulkan 是否接近相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，则显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用 ASSERT_TRUE 确保 out_cpu 和 out_vulkan 相等
  ASSERT_TRUE(check);
}

// 定义测试用例 VulkanAPITest.transpose_2d_height_and_width_small
TEST_F(VulkanAPITest, transpose_2d_height_and_width_small) {
  // 调用 test_transpose 函数，测试 2D 张量高度和宽度的转置，形状为 {1, 1}，转置索引为 0 和 1
  test_transpose({1, 1}, 0, 1);
}

// 定义测试用例 VulkanAPITest.transpose_2d_height_and_width_medium
TEST_F(VulkanAPITest, transpose_2d_height_and_width_medium) {
  // 调用 test_transpose 函数，测试 2D 张量高度和宽度的转置，形状为 {7, 5}，转置索引为 0 和 1
  test_transpose({7, 5}, 0, 1);
}

// 定义测试用例 VulkanAPITest.transpose_2d_height_and_width_large
TEST_F(VulkanAPITest, transpose_2d_height_and_width_large) {
  // 调用 test_transpose 函数，测试 2D 张量高度和宽度的转置，形状为 {53, 117}，转置索引为 0 和 1
  test_transpose({53, 117}, 0, 1);
}

// 定义测试用例 VulkanAPITest.transpose_2d_height_and_height_large
TEST_F(VulkanAPITest, transpose_2d_height_and_height_large) {
  // 调用 test_transpose 函数，测试 2D 张量高度和高度的转置，形状为 {53, 117}，转置索引为 0 和 0
  test_transpose({53, 117}, 0, 0);
}

// 定义测试用例 VulkanAPITest.transpose_2d_width_and_width_large
TEST_F(VulkanAPITest, transpose_2d_width_and_width_large) {
  // 调用 test_transpose 函数，测试 2D 张量宽度和宽度的转置，形状为 {53, 117}，转置索引为 1 和 1
  test_transpose({53, 117}, 1, 1);
}

// 定义测试用例 VulkanAPITest.transpose_3d_height_and_width_small
TEST_F(VulkanAPITest, transpose_3d_height_and_width_small) {
  // 调用 test_transpose 函数，测试 3D 张量高度和宽度的转置，形状为 {1, 1, 1}，转置索引为 1 和 2
  test_transpose({1, 1, 1}, 1, 2);
}

// 定义测试用例 VulkanAPITest.transpose_3d_height_and_width_medium
TEST_F(VulkanAPITest, transpose_3d_height_and_width_medium) {
  // 调用 test_transpose 函数，测试 3D 张量高度和宽度的转置，形状为 {3, 2, 5}，转置索引为 1 和 2
  test_transpose({3, 2, 5}, 1, 2);
}

// 定义测试用例 VulkanAPITest.transpose_3d_height_and_width_large
TEST_F(VulkanAPITest, transpose_3d_height_and_width_large) {
  // 调用 test_transpose 函数，测试 3D 张量高度和宽度的转置，形状为 {100, 1, 144}，转置索引为 1 和 2
  test_transpose({100, 1, 144}, 1, 2);
}

// 定义测试用例 VulkanAPITest.transpose_3d_width_and_width_large
TEST_F(VulkanAPITest, transpose_3d_width_and_width_large) {
  // 调用 test_transpose 函数，测试 3D 张量宽度和宽度的转置，形状为 {100, 1, 144}，转置索引为 2 和 2
  test_transpose({100, 1, 144}, 2, 2);
}

// 定义测试用例 VulkanAPITest.transpose_3d_depth_and_width_small
TEST_F(VulkanAPITest, transpose_3d_depth_and_width_small) {
  // 调用 test_transpose 函数，测试 3D 张量深度和宽度的转置，形状为 {1, 1, 1}，转置索引为 0 和 2
  test_transpose({1, 1, 1}, 0, 2);
}

// 定义测试用例 VulkanAPITest.transpose_3d_depth_and_width_medium
TEST_F(VulkanAPITest, transpose_3d_depth_and_width_medium) {
  // 调用 test_transpose 函数，测试 3D 张量深度和宽度的转置，形状为 {3, 2, 5}，转置索引为 0 和 2
  test_transpose({3, 2, 5}, 0, 2);
}

// 定义测试用例 VulkanAPITest.transpose_3d_depth_and_width_large
TEST_F(VulkanAPITest, transpose_3d_depth_and_width_large) {
  // 调用 test_transpose 函数，测试 3D 张量深度和宽度的转置，形状为 {113, 1, 141}，转置索引为 0 和 2
  test_transpose({113, 1, 141}, 0, 2);
}

// 定义测试用例 VulkanAPITest.transpose_3d_depth_and_depth_large
TEST_F(VulkanAPITest, transpose_3d_depth_and_depth_large) {
  // 调用 test_transpose 函数，测试 3D 张量深度和深度的转置，形状为 {113, 2, 131}，转置索引为 0 和 0
  test_transpose({113, 2, 131}, 0, 0);
}

// 定义测试用例 VulkanAPITest.transpose_3d_depth_and_height_small
TEST_F(VulkanAPITest, transpose_3d_depth_and_height_small) {
  // 调用 test_transpose 函数，测试 3D 张量深度和高度的转置，形状为 {1, 1, 1}，转置索引为 0 和 1
  test_transpose({1, 1, 1}, 0, 1);
}

// 定义测试用例 VulkanAPITest.transpose_3d_depth_and_height_medium
TEST_F(VulkanAPITest, transpose_3d_depth_and_height_medium) {
  // 调用 test_transpose 函数，测试 3D 张量深度和高度的转置，形状为 {3, 7, 5}，转置索引为 0 和 1
  test_transpose({3, 7, 5}, 0, 1);
}

// 定义测试用例 VulkanAPITest.transpose_3d_depth_and_height_large
TEST_F(VulkanAPITest, transpose_3d_depth_and_height_large) {
  // 调用 test_transpose 函数，测试 3D 张量深度和高度的转置，形
// 调用 VulkanAPITest 的 transpose_4d_depth_and_height_large 测试用例，测试矩阵转置函数
TEST_F(VulkanAPITest, transpose_4d_depth_and_height_large) {
  // 调用 test_transpose 函数，传入维度为 {7, 51, 41, 3} 的输入，进行深度和高度的转置测试
  test_transpose({7, 51, 41, 3}, 1, 2);
}

// 调用 VulkanAPITest 的 transpose_4d_depth_and_width_large 测试用例，测试矩阵转置函数
TEST_F(VulkanAPITest, transpose_4d_depth_and_width_large) {
  // 调用 test_transpose 函数，传入维度为 {7, 51, 41, 3} 的输入，进行深度和宽度的转置测试
  test_transpose({7, 51, 41, 3}, 1, 3);
}

// 调用 VulkanAPITest 的 transpose_4d_height_and_width_large 测试用例，测试矩阵转置函数
TEST_F(VulkanAPITest, transpose_4d_height_and_width_large) {
  // 调用 test_transpose 函数，传入维度为 {7, 51, 41, 3} 的输入，进行高度和宽度的转置测试
  test_transpose({7, 51, 41, 3}, 2, 3);
}

// 定义一个名为 test_exp 的测试函数，测试 exp（指数函数）的操作
void test_exp(const at::IntArrayRef input_shape) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 生成在 CPU 上的随机数，形状由 input_shape 决定，数据类型为 float
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 对 CPU 上的输入应用指数函数，生成输出 out_cpu
  const auto out_cpu = at::exp(in_cpu);

  // 获取在 Vulkan 设备上的输入数据
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 设备上的输入应用指数函数，生成输出 out_vulkan
  const auto out_vulkan = at::exp(in_vulkan);

  // 检查 CPU 和 Vulkan 输出的近似性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查失败，则展示相对误差，输出失败信息和输入形状
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "exp test failed with input shape: "
              << input_shape << std::endl;
  }
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 调用 VulkanAPITest 的 unary_op_exp 测试用例，测试指数函数 exp 的单目运算
TEST_F(VulkanAPITest, unary_op_exp) {
  // 分别使用不同形状调用 test_exp 函数进行测试
  test_exp({5});
  test_exp({5, 6});
  test_exp({7, 3, 5});
  test_exp({11, 1, 4, 2});
}

// 定义一个名为 test_exp_ 的测试函数，测试 exp_（原位指数函数）的操作
void test_exp_(const at::IntArrayRef input_shape) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 生成在 CPU 上的随机数，形状由 input_shape 决定，数据类型为 float
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 获取在 Vulkan 设备上的数据
  const auto vulkan = cpu.vulkan();

  // 对 CPU 和 Vulkan 设备上的数据原位应用指数函数
  cpu.exp_();
  vulkan.exp_();

  // 检查 CPU 和 Vulkan 输出的近似性
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查失败，则展示相对误差，输出失败信息和输入形状
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "exp_ test failed with input shape: "
              << input_shape << std::endl;
  }
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 调用 VulkanAPITest 的 unary_op_exp_ 测试用例，测试原位指数函数 exp_ 的单目运算
TEST_F(VulkanAPITest, unary_op_exp_) {
  // 分别使用不同形状调用 test_exp_ 函数进行测试
  test_exp_({5});
  test_exp_({5, 6});
  test_exp_({7, 3, 5});
  test_exp_({11, 1, 4, 2});
}

// 定义一个名为 test_sqrt 的测试函数，测试 sqrt（平方根函数）的操作
void test_sqrt(const at::IntArrayRef input_shape) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 生成在 CPU 上的随机数，形状由 input_shape 决定，数据类型为 float
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 对 CPU 上的输入应用平方根函数，生成输出 out_cpu
  const auto out_cpu = at::sqrt(in_cpu);

  // 获取在 Vulkan 设备上的输入数据
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 设备上的输入应用平方根函数，生成输出 out_vulkan
  const auto out_vulkan = at::sqrt(in_vulkan);

  // 检查 CPU 和 Vulkan 输出的近似性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查失败，则展示相对误差，输出失败信息和输入形状
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "sqrt test failed with input shape: "
              << input_shape << std::endl;
  }
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 调用 VulkanAPITest 的 unary_op_sqrt 测试用例，测试平方根函数 sqrt 的单目运算
TEST_F(VulkanAPITest, unary_op_sqrt) {
  // 分别使用不同形状调用 test_sqrt 函数进行测试
  test_sqrt({5});
  test_sqrt({5, 6});
  test_sqrt({7, 3, 5});
  test_sqrt({11, 1, 4, 2});
}

// 定义一个名为 test_sqrt_ 的测试函数，测试 sqrt_（原位平方根函数）的操作
void test_sqrt_(const at::IntArrayRef input_shape) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 生成在 CPU 上的随机数，形状由 input_shape 决定，数据类型为 float
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 获取在 Vulkan 设备上的数据
  const auto vulkan = cpu.vulkan();

  // 对 CPU 和 Vulkan 设备上的数据原位应用平方根函数
  cpu.sqrt_();
  vulkan.sqrt_();

  // 检查 CPU 和 Vulkan 输出的近似性
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果检查失败，则展示相对误差，输出失败信息和输入形状
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "sqrt_ test failed with input shape: "
              << input_shape << std::endl;
  }
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

// 调用 VulkanAPITest 的 unary_op_sqrt_ 测试用例，测试原位平方根函数 sqrt_ 的单目运算
TEST_F(VulkanAPITest, unary_op_sqrt_) {
  // 分别使用不同形状调用 test_sqrt_ 函数进行测试
  test_sqrt_({5});
  test_sqrt_({5, 6});
  test_sqrt_({7, 3, 5});
  test_sqrt_({11, 1, 4, 2});
}
void test_log(const at::IntArrayRef input_shape) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 为避免输入值为0，需要添加一个非常小的常量
  const auto in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.0001;
  // 计算 CPU 上的对数
  const auto out_cpu = at::log(in_cpu);

  // 将 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 在 Vulkan 上计算对数
  const auto out_vulkan = at::log(in_vulkan);

  // 检查 CPU 和 Vulkan 计算结果的近似性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似，则展示相对误差，打印失败消息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "log test failed with input shape: " << input_shape
              << std::endl;
  }
  // 断言结果近似
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_log) {
  // 测试不同形状的输入张量对 log 函数的操作
  test_log({5});
  test_log({5, 6});
  test_log({7, 3, 5});
  test_log({11, 1, 4, 2});
}

void test_log_(const at::IntArrayRef input_shape) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 为避免输入值为0，需要添加一个非常小的常量
  const auto cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.0001;
  // 将 CPU 张量转换为 Vulkan 张量
  const auto vulkan = cpu.vulkan();

  // 在原地对 CPU 张量执行对数操作
  cpu.log_();
  // 在原地对 Vulkan 张量执行对数操作
  vulkan.log_();

  // 检查 CPU 和 Vulkan 操作结果的近似性
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果结果不近似，则展示相对误差，打印失败消息
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "log_ test failed with input shape: " << input_shape
              << std::endl;
  }
  // 断言结果近似
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_log_) {
  // 测试不同形状的输入张量对 log_ 函数的原地操作
  test_log_({5});
  test_log_({5, 6});
  test_log_({7, 3, 5});
  test_log_({11, 1, 4, 2});
}

void test_unsqueeze(const at::IntArrayRef input_shape, int64_t dim) {
  // 进入推理模式
  c10::InferenceMode mode;
  // 生成指定形状的随机 CPU 张量
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 对 CPU 张量进行在指定维度上的unsqueeze操作
  const auto out_cpu = at::unsqueeze(in_cpu, dim);

  // 将 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 在 Vulkan 张量上进行在指定维度上的unsqueeze操作
  const auto out_vulkan = at::unsqueeze(in_vulkan, dim);

  // 检查 CPU 和 Vulkan 操作结果的近似性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似，则展示相对误差，打印失败消息
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "unsqueeze test failed with input shape: "
              << input_shape << std::endl;
  }
  // 断言结果近似
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unsqueeze_0dto1d_dim0) {
  // 测试将 0 维张量转为 1 维张量的 unsqueeze 操作在维度 0 上的效果
  test_unsqueeze({}, 0);
  test_unsqueeze({}, -1);
}

TEST_F(VulkanAPITest, unsqueeze_1dto2d_dim0) {
  // 测试将 1 维张量转为 2 维张量的 unsqueeze 操作在维度 0 上的效果
  test_unsqueeze({5}, 0);
  test_unsqueeze({6}, -2);
  test_unsqueeze({111}, 0);
  test_unsqueeze({112}, -2);
}

TEST_F(VulkanAPITest, unsqueeze_1dto2d_dim1) {
  // 测试将 1 维张量转为 2 维张量的 unsqueeze 操作在维度 1 上的效果
  test_unsqueeze({5}, 1);
  test_unsqueeze({6}, -1);
  test_unsqueeze({111}, 1);
  test_unsqueeze({112}, -1);
}

TEST_F(VulkanAPITest, unsqueeze_2dto3d_dim0) {
  // 测试将 2 维张量转为 3 维张量的 unsqueeze 操作在维度 0 上的效果
  test_unsqueeze({1, 5}, 2);
  test_unsqueeze({5, 7}, 0);
  test_unsqueeze({7, 5}, -3);
  test_unsqueeze({111, 222}, 0);
  test_unsqueeze({222, 111}, -3);
}

TEST_F(VulkanAPITest, unsqueeze_2dto3d_dim1) {
  // 测试将 2 维张量转为 3 维张量的 unsqueeze 操作在维度 1 上的效果
  test_unsqueeze({5, 7}, 1);
  test_unsqueeze({7, 5}, -2);
  test_unsqueeze({111, 222}, 1);
  test_unsqueeze({222, 111}, -2);
}

TEST_F(VulkanAPITest, unsqueeze_2dto3d_dim2) {
  // 测试将 2 维张量转为 3 维张量的 unsqueeze 操作在维度 2 上的效果
  test_unsqueeze({5, 7}, 2);
  test_unsqueeze({7, 5}, -1);
  test_unsqueeze({111, 222}, 2);
  test_unsqueeze({222, 111}, -1);
}
TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim0) {
  // 调用测试函数 test_unsqueeze，将 {2, 3, 4} 按照维度 0 进行 unsqueeze 操作
  test_unsqueeze({2, 3, 4}, 0);
  // 调用测试函数 test_unsqueeze，将 {4, 3, 2} 按照维度 -4 进行 unsqueeze 操作
  test_unsqueeze({4, 3, 2}, -4);
  // 调用测试函数 test_unsqueeze，将 {22, 33, 11} 按照维度 0 进行 unsqueeze 操作
  test_unsqueeze({22, 33, 11}, 0);
  // 调用测试函数 test_unsqueeze，将 {33, 11, 22} 按照维度 -4 进行 unsqueeze 操作
  test_unsqueeze({33, 11, 22}, -4);
}

TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim1) {
  // 调用测试函数 test_unsqueeze，将 {2, 3, 4} 按照维度 1 进行 unsqueeze 操作
  test_unsqueeze({2, 3, 4}, 1);
  // 调用测试函数 test_unsqueeze，将 {4, 3, 2} 按照维度 -3 进行 unsqueeze 操作
  test_unsqueeze({4, 3, 2}, -3);
  // 调用测试函数 test_unsqueeze，将 {22, 33, 11} 按照维度 1 进行 unsqueeze 操作
  test_unsqueeze({22, 33, 11}, 1);
  // 调用测试函数 test_unsqueeze，将 {33, 11, 22} 按照维度 -3 进行 unsqueeze 操作
  test_unsqueeze({33, 11, 22}, -3);
}

TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim2) {
  // 调用测试函数 test_unsqueeze，将 {2, 3, 4} 按照维度 2 进行 unsqueeze 操作
  test_unsqueeze({2, 3, 4}, 2);
  // 调用测试函数 test_unsqueeze，将 {4, 3, 2} 按照维度 -2 进行 unsqueeze 操作
  test_unsqueeze({4, 3, 2}, -2);
  // 调用测试函数 test_unsqueeze，将 {22, 33, 11} 按照维度 2 进行 unsqueeze 操作
  test_unsqueeze({22, 33, 11}, 2);
  // 调用测试函数 test_unsqueeze，将 {33, 11, 22} 按照维度 -2 进行 unsqueeze 操作
  test_unsqueeze({33, 11, 22}, -2);
}

TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim3) {
  // 调用测试函数 test_unsqueeze，将 {1, 5, 2} 按照维度 3 进行 unsqueeze 操作
  test_unsqueeze({1, 5, 2}, 3);
  // 调用测试函数 test_unsqueeze，将 {2, 3, 4} 按照维度 3 进行 unsqueeze 操作
  test_unsqueeze({2, 3, 4}, 3);
  // 调用测试函数 test_unsqueeze，将 {4, 3, 2} 按照维度 -1 进行 unsqueeze 操作
  test_unsqueeze({4, 3, 2}, -1);
  // 调用测试函数 test_unsqueeze，将 {22, 33, 11} 按照维度 3 进行 unsqueeze 操作
  test_unsqueeze({22, 33, 11}, 3);
  // 调用测试函数 test_unsqueeze，将 {33, 11, 22} 按照维度 -1 进行 unsqueeze 操作
  test_unsqueeze({33, 11, 22}, -1);
}

TEST_F(VulkanAPITest, upsample_nearest2d) {
  // 生成一个在 CPU 上随机生成的形状为 {1, 2, 2, 3} 的张量
  const auto in_cpu = at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 对 CPU 上的输入张量进行最近邻插值上采样，目标大小为 {4, 6}
  const auto out_cpu = at::upsample_nearest2d(in_cpu, {4, 6});

  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 张量进行最近邻插值上采样，目标大小为 {4, 6}
  const auto out_vulkan = at::upsample_nearest2d(in_vulkan, {4, 6});

  // 检查 CPU 和 Vulkan 张量的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则显示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, upsample_bilinear2d_align_false_small) {
  // 生成一个在 CPU 上随机生成的形状为 {1, 2, 2, 3} 的张量
  const auto in_cpu = at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 对 CPU 上的输入张量进行双线性插值上采样，目标大小为 {4, 6}，不进行对齐
  const auto out_cpu = at::upsample_bilinear2d(in_cpu, {4, 6}, false);

  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 张量进行双线性插值上采样，目标大小为 {4, 6}，不进行对齐
  const auto out_vulkan = at::upsample_bilinear2d(in_vulkan, {4, 6}, false);

  // 检查 CPU 和 Vulkan 张量的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则显示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, upsample_bilinear2d_align_false_large) {
  // 生成一个在 CPU 上随机生成的形状为 {1, 7, 25, 25} 的张量
  const auto in_cpu = at::rand({1, 7, 25, 25}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 对 CPU 上的输入张量进行双线性插值上采样，目标大小为 {45, 45}，不进行对齐
  const auto out_cpu = at::upsample_bilinear2d(in_cpu, {45, 45}, false);

  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 张量进行双线性插值上采样，目标大小为 {45, 45}，不进行对齐
  const auto out_vulkan = at::upsample_bilinear2d(in_vulkan, {45, 45}, false);

  // 检查 CPU 和 Vulkan 张量的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则显示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 CPU 和 Vulkan 张量近似相等
  ASSERT_TRUE(check);
}

TEST_F(Vulkan
TEST_F(VulkanAPITest, upsample_bilinear2d_align_true_large) {
  // 生成一个大小为 [1, 7, 25, 25] 的随机张量，存储在 CPU 上，并使用 float 数据类型
  const auto in_cpu = at::rand({1, 7, 25, 25}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 对 in_cpu 进行双线性插值上采样，目标大小为 [45, 45]，align_corners 设置为 true
  const auto out_cpu = at::upsample_bilinear2d(in_cpu, {45, 45}, true);

  // 将 in_cpu 转换为 Vulkan 张量格式
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 格式的张量进行双线性插值上采样，目标大小为 [45, 45]，align_corners 设置为 true
  const auto out_vulkan = at::upsample_bilinear2d(in_vulkan, {45, 45}, true);

  // 检查 out_cpu 和 out_vulkan 是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，则显示其相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 check 为真
  ASSERT_TRUE(check);
}

void test_unbind(const at::IntArrayRef input_shape, int64_t dim) {
  // 生成指定形状和设备 (CPU) 的随机张量
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 对输入张量按照指定维度进行 unbind 操作
  const auto out_cpu = at::unbind(in_cpu, dim);

  // 将 in_cpu 转换为 Vulkan 张量格式
  const auto in_vulkan = in_cpu.vulkan();
  // 对 Vulkan 格式的输入张量按照指定维度进行 unbind 操作
  const auto out_vulkan = at::unbind(in_vulkan, dim);

  // 获取 out_vulkan 的大小
  int64_t size = out_vulkan.size();

  // 遍历 out_vulkan 中的元素
  for (const auto i : c10::irange(size)) {
    // 检查 out_cpu[i] 和 out_vulkan[i] 是否几乎相等
    const auto check = almostEqual(out_cpu[i], out_vulkan[i].cpu());
    // 如果不相等，则显示其相对误差
    if (!check) {
      std::cout << "The " << i << "th vectors aren't equal." << std::endl;
      showRtol(out_cpu[i], out_vulkan[i].cpu());
    }

    // 断言 check 为真
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, unbind_3d_depth_small) {
  // 测试维度为 3 的小尺寸张量的 unbind 操作
  test_unbind({1, 1, 1}, 0);
}

TEST_F(VulkanAPITest, unbind_3d_depth_medium) {
  // 测试维度为 3 的中等尺寸张量的 unbind 操作
  test_unbind({3, 2, 5}, 0);
}

TEST_F(VulkanAPITest, unbind_3d_depth_large) {
  // 测试维度为 3 的大尺寸张量的 unbind 操作
  test_unbind({100, 1, 144}, 0);
}

void test_var(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list, bool unbiased=true, bool keepdim=false) {
  // 进入推断模式
  c10::InferenceMode mode;

  // 生成指定形状和设备 (CPU) 的随机张量
  const auto in_cpu = at::rand(input_shape, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 计算输入张量在指定维度上的方差
  const auto out_cpu = at::var(in_cpu, dim_list, unbiased, keepdim);

  // 将 in_cpu 转换为 Vulkan 张量格式
  const auto in_vulkan = in_cpu.vulkan();
  // 计算 Vulkan 格式的输入张量在指定维度上的方差
  const auto out_vulkan = at::var(in_vulkan, dim_list, unbiased, keepdim);

  // 检查 out_cpu 和 out_vulkan 是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，则显示其相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 check 为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, var_2d_unbiased) {
  // 测试维度为 2 的张量在 unbiased 模式下计算方差
  test_var({3, 5}, {1}, true, true);
  test_var({3, 5}, {1}, true, false);

  // 输入张量的维度与 dim_list 的大小相同，仅支持 keepdim 设置为 true
  test_var({3, 5}, {0, 1}, true, true);
}

TEST_F(VulkanAPITest, var_2d_biased) {
  // 测试维度为 2 的张量在 biased 模式下计算方差
  test_var({3, 5}, {1}, false, true);
  test_var({3, 5}, {1}, false, false);

  // 输入张量的维度与 dim_list 的大小相同，仅支持 keepdim 设置为 true
  test_var({3, 5}, {0, 1}, false, true);
}

TEST_F(VulkanAPITest, var_3d_unbiased) {
  // 测试维度为 3 的张量在 unbiased 模式下计算方差
  test_var({3, 5, 7}, {1}, true, true);
  test_var({3, 5, 7}, {1}, true, false);

  test_var({3, 5, 7}, {0, 1}, true, true);
  test_var({3, 5, 7}, {0, 1}, true, false);

  test_var({3, 5, 7}, {0, 2}, true, true);
  test_var({3, 5, 7}, {0, 2}, true, false);

  test_var({3, 5, 7}, {-1, -2}, true, true);
  test_var({3, 5, 7}, {-1, -2}, true, false);

  test_var({3, 5, 7}, {0, 1, 2}, true, true);
}
TEST_F(VulkanAPITest, var_3d_biased) {
  // 调用 test_var 函数，测试三维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7}, {1}, false, true);
  // 调用 test_var 函数，测试三维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7}, {1}, false, false);

  // 调用 test_var 函数，测试三维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7}, {0, 1}, false, true);
  // 调用 test_var 函数，测试三维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7}, {0, 1}, false, false);

  // 调用 test_var 函数，测试三维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7}, {0, 2}, false, true);
  // 调用 test_var 函数，测试三维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7}, {0, 2}, false, false);

  // 调用 test_var 函数，测试三维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7}, {-1, -2}, false, true);
  // 调用 test_var 函数，测试三维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7}, {-1, -2}, false, false);

  // 调用 test_var 函数，测试三维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7}, {0, 1, 2}, false, true);
}

TEST_F(VulkanAPITest, var_4d_unbiased) {
  // 调用 test_var 函数，测试四维数据，带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0}, true, true);
  // 调用 test_var 函数，测试四维数据，带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {1}, true, false);

  // 调用 test_var 函数，测试四维数据，带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1}, true, true);
  // 调用 test_var 函数，测试四维数据，带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1}, true, false);

  // 调用 test_var 函数，测试四维数据，带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 2}, true, true);
  // 调用 test_var 函数，测试四维数据，带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {0, 2}, true, false);

  // 调用 test_var 函数，测试四维数据，带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {-1, -2}, true, true);
  // 调用 test_var 函数，测试四维数据，带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {-1, -2}, true, false);

  // 调用 test_var 函数，测试四维数据，带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1, 2}, true, true);
  // 调用 test_var 函数，测试四维数据，带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {0, -1, 2}, true, false);

  // 调用 test_var 函数，测试四维数据，带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1, 2, 3}, true, true);
}

TEST_F(VulkanAPITest, var_4d_biased) {
  // 调用 test_var 函数，测试四维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0}, false, true);
  // 调用 test_var 函数，测试四维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {1}, false, false);

  // 调用 test_var 函数，测试四维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1}, false, true);
  // 调用 test_var 函数，测试四维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1}, false, false);

  // 调用 test_var 函数，测试四维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 2}, false, true);
  // 调用 test_var 函数，测试四维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {0, 2}, false, false);

  // 调用 test_var 函数，测试四维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {-1, -2}, false, true);
  // 调用 test_var 函数，测试四维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {-1, -2}, false, false);

  // 调用 test_var 函数，测试四维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1, 2}, false, true);
  // 调用 test_var 函数，测试四维数据，不带偏置，不启用偏置校正
  test_var({3, 5, 7, 11}, {0, -1, 2}, false, false);

  // 调用 test_var 函数，测试四维数据，不带偏置，启用偏置校正
  test_var({3, 5, 7, 11}, {0, 1, 2, 3}, false, true);
}

TEST_F(VulkanAPITest, view_explicit) {
  c10::InferenceMode mode;

  // 生成大小为 7x8x9 的随机张量，数据类型为 float，在 CPU 上
  const auto in_cpu = at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat));
  // 调用 Vulkan 后端的函数，将张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 指定视图的形状为 {7, 8, 9, 1}
  const std::array<int64_t, 4> shape{7, 8, 9, 1};

  // 在 CPU 上对输入张量进行形状变换
  const auto out_cpu = in_cpu.view(shape);
  // 在 Vulkan 张量上进行形状变换
    // 使用 PyTorch 的 ATen 库生成一个形状为 {7, 8, 9} 的张量，数据类型为 float，在 CPU 上进行计算
    at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      // 使用 Vulkan 后端进行张量操作
      .vulkan()
      // 修改张量的视图为 {7, 8, -2}，其中 -2 表示自动计算维度大小
      .view({7, 8, -2});
  }, ::std::exception);

  // 预期：形状不兼容的异常
  EXPECT_THROW({
    // 使用 PyTorch 的 ATen 库生成一个形状为 {7, 8, 9} 的张量，数据类型为 float，在 CPU 上进行计算
    at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      // 使用 Vulkan 后端进行张量操作
      .vulkan()
      // 尝试修改张量的视图为 {7, 70}，这里会抛出运行时异常
      .view({7, 70});
  }, ::std::runtime_error);
TEST_F(VulkanAPITest, cat_4d_dim0_invalidinputs_exceptions) {
  // Arrange: Vulkan cat inputs must have matching sizes except concatenated dimension
  {
    // 定义三个不同形状的CPU张量，用于测试
    const auto in_cpu1 = at::rand({3, 5, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    // 期望此处抛出异常，因为 Vulkan cat 要求输入张量除了连接维度外其余维度需匹配
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);
    }, ::std::exception);
  }

  // Arrange: Vulkan cat expects 4 dimensional inputs
  {
    // 定义三个不同形状的CPU张量，用于测试
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    // 期望此处抛出异常，因为 Vulkan cat 要求输入张量必须是四维的
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);
    }, ::std::exception);
  }
}

TEST_F(VulkanAPITest, cat_4d_dim0_samebatch_success) {
  // Arrange
  // 定义三个相同形状的CPU张量，用于测试
  const auto in_cpu1 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第0维度上进行张量拼接操作，分别使用CPU和Vulkan后端进行拼接
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0); // dim=batch

  // Assert
  // 检查CPU和Vulkan计算结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果应该为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_diffbatch_success) {
  // Arrange
  // 定义三个不同形状的CPU张量，用于测试
  const auto in_cpu1 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({117, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({139, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第0维度上进行张量拼接操作，分别使用CPU和Vulkan后端进行拼接
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0); // dim=batch

  // Assert
  // 检查CPU和Vulkan计算结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果应该为真
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_4d_dim0_singledepth_success) {
  // Arrange: batch x channel (1x1) = single depth texture
  // 创建三个 1x1 大小的张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 0 维度上进行张量拼接，生成一个新的张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 调用 Vulkan 后端的拼接操作，同样在第 0 维度上进行，生成 Vulkan 后端的结果 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0); // dim=batch

  // Assert
  // 检查两个张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则显示相对误差，并调用 showRtol 函数
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用 ASSERT_TRUE 来确保检查通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_singletensor_success) {
  // Arrange: single input tensor
  // 创建一个大小为 3x7x221x193 的张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 0 维度上进行张量拼接，生成一个新的张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1}, 0);
  // 调用 Vulkan 后端的拼接操作，同样在第 0 维度上进行，生成 Vulkan 后端的结果 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1}, 0); // dim=batch

  // Assert
  // 检查两个张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则显示相对误差，并调用 showRtol 函数
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用 ASSERT_TRUE 来确保检查通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_twotensors_success) {
  // Arrange: two input tensors
  // 创建两个大小为 3x7x221x193 的张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 0 维度上进行张量拼接，生成一个新的张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2}, 0);
  // 调用 Vulkan 后端的拼接操作，同样在第 0 维度上进行，生成 Vulkan 后端的结果 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan()}, 0); // dim=batch

  // Assert
  // 检查两个张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则显示相对误差，并调用 showRtol 函数
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用 ASSERT_TRUE 来确保检查通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_negdim_success) {
  // Arrange
  // 创建三个大小为 221x9x193x3 的张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({221, 9, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({113, 9, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({331, 9, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 -4 维度上进行张量拼接，生成一个新的张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -4);
  // 调用 Vulkan 后端的拼接操作，同样在第 -4 维度上进行，生成 Vulkan 后端的结果 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -4);

  // Assert
  // 检查两个张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则显示相对误差，并调用 showRtol 函数
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 使用 ASSERT_TRUE 来确保检查通过
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_4d_dim1_negdim_success) {
  // Arrange
  // 创建三个不同形状的随机张量，存储在CPU上，数据类型为float
  const auto in_cpu1 = at::rand({9, 221, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 113, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 331, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第四维度（从后往前数）上进行张量拼接操作，生成CPU上的输出张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -3);
  // 使用Vulkan API在第四维度上进行张量拼接操作，生成Vulkan上的输出张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -3);

  // Assert
  // 检查CPU和Vulkan生成的输出张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 确保检查通过，即CPU和Vulkan生成的输出张量几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim2_negdim_success) {
  // Arrange
  // 创建三个不同形状的随机张量，存储在CPU上，数据类型为float
  const auto in_cpu1 = at::rand({9, 193, 221, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 113, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 331, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第三维度上进行张量拼接操作，生成CPU上的输出张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -2);
  // 使用Vulkan API在第三维度上进行张量拼接操作，生成Vulkan上的输出张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -2);

  // Assert
  // 检查CPU和Vulkan生成的输出张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 确保检查通过，即CPU和Vulkan生成的输出张量几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim3_negdim_success) {
  // Arrange
  // 创建三个不同形状的随机张量，存储在CPU上，数据类型为float
  const auto in_cpu1 = at::rand({9, 193, 3, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 3, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 3, 331}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第二维度上进行张量拼接操作，生成CPU上的输出张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  // 使用Vulkan API在第二维度上进行张量拼接操作，生成Vulkan上的输出张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  // 检查CPU和Vulkan生成的输出张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 确保检查通过，即CPU和Vulkan生成的输出张量几乎相等
  ASSERT_TRUE(check);
}

#if !defined(__APPLE__)
TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_samefeature_success) {
  // Arrange
  // 创建三个形状相同的随机张量，存储在CPU上，数据类型为float
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第一维度上进行张量拼接操作，生成CPU上的输出张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 使用Vulkan API在第一维度上进行张量拼接操作，生成Vulkan上的输出张量（维度为特征/通道）
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert
  // 检查CPU和Vulkan生成的输出张量是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 确保检查通过，即CPU和Vulkan生成的输出张量几乎相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_difffeature_success) {
  // Arrange
  // 创建三个不同形状的CPU张量作为输入
  const auto in_cpu1 = at::rand({3, 3, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 8, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 11, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第一个维度（dim=1）上拼接三个CPU张量，生成输出张量out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 在Vulkan环境中，在第一个维度（dim=1）上拼接三个CPU张量的Vulkan版本，生成输出张量out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  // 检查out_cpu和out_vulkan是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不接近，展示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查是否通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_texture2d_success) {
  // Arrange: 2D Texture (VK_IMAGE_VIEW_TYPE_2D)
  // 创建三个相同形状的CPU张量作为输入
  const auto in_cpu1 = at::rand({2, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({2, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({2, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第一个维度（dim=1）上拼接三个CPU张量，生成输出张量out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 在Vulkan环境中，在第一个维度（dim=1）上拼接三个CPU张量的Vulkan版本，生成输出张量out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  // 检查out_cpu和out_vulkan是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不接近，展示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查是否通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_singledepth_success) {
  // Arrange: batch x channel (1x1) = single depth texture
  // 创建三个相同形状的CPU张量作为输入
  const auto in_cpu1 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第一个维度（dim=1）上拼接三个CPU张量，生成输出张量out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 在Vulkan环境中，在第一个维度（dim=1）上拼接三个CPU张量的Vulkan版本，生成输出张量out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  // 检查out_cpu和out_vulkan是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不接近，展示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查是否通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_singletensor_success) {
  // Arrange: single input tensor
  // 创建一个CPU张量作为输入
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第一个维度（dim=1）上拼接一个CPU张量，生成输出张量out_cpu
  const auto out_cpu = at::cat({in_cpu1}, 1);
  // 在Vulkan环境中，在第一个维度（dim=1）上拼接一个CPU张量的Vulkan版本，生成输出张量out_vulkan
  const auto out_vulkan = at::cat({in_cpu1}, 1); // dim=feature(channel)

  // Assert
  // 检查out_cpu和out_vulkan是否接近
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不接近，展示两者的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查是否通过
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_twotensors_success) {
  // Arrange: two input tensors of shape [3, 7, 221, 193] on CPU
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: concatenate tensors along dimension 1
  const auto out_cpu = at::cat({in_cpu1, in_cpu2}, 1);
  // Convert tensors to Vulkan tensors and concatenate along dimension 1 (channel dimension)
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan()}, 1);

  // Assert: check if the outputs are almost equal
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    // Display the relative tolerance if the check fails
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // Ensure the test passes by asserting the check
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_bat1_mult4ch_success) {
  // Arrange: batch=1 and channel (multiple of 4, i.e., channel % 4 == 0)
  const auto in_cpu1 = at::rand({1, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: concatenate tensors along dimension 1
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // Convert tensors to Vulkan tensors and concatenate along dimension 1 (channel dimension)
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert: check if the outputs are almost equal
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    // Display the relative tolerance if the check fails
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // Ensure the test passes by asserting the check
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_bat2_mult4ch_success) {
  // Arrange: batch=2 and channel (multiple of 4, i.e., channel % 4 == 0)
  const auto in_cpu1 = at::rand({2, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({2, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({2, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: concatenate tensors along dimension 1
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // Convert tensors to Vulkan tensors and concatenate along dimension 1 (channel dimension)
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert: check if the outputs are almost equal
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    // Display the relative tolerance if the check fails
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // Ensure the test passes by asserting the check
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_mult4ch_mixed_success) {
  // Arrange: batch=1 and channels with different multiples of 4
  const auto in_cpu1 = at::rand({3, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 8, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 12, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: concatenate tensors along dimension 1
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // Convert tensors to Vulkan tensors and concatenate along dimension 1 (channel dimension)
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert: check if the outputs are almost equal
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    // Display the relative tolerance if the check fails
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // Ensure the test passes by asserting the check
  ASSERT_TRUE(check);
}
// 禁用测试例：将多个张量在指定维度上拼接，并验证 Vulkan 实现的正确性
TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_mult4ch_nonmult4ch_success) {
  // Arrange: 设置不同通道数的张量，用于测试拼接功能，要求通道数部分为4的倍数
  const auto in_cpu1 = at::rand({3, 3, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu4 = at::rand({3, 8, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: 在维度1上拼接 CPU 张量，并通过 Vulkan API 在相同维度上拼接
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3, in_cpu4}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan(), in_cpu4.vulkan()}, 1); // dim=feature(channel)

  // Assert: 检查 CPU 和 Vulkan 输出是否几乎相等，若不等则显示相对误差
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言：验证最终结果的正确性
  ASSERT_TRUE(check);
}

// 测试例：在第2维度上拼接相同高度的张量，并验证 Vulkan 实现的正确性
TEST_F(VulkanAPITest, cat_4d_dim2_sameheight_success) {
  // Arrange: 设置相同高度的张量，用于测试拼接功能
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: 在维度2上拼接 CPU 张量，并通过 Vulkan API 在相同维度上拼接
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert: 检查 CPU 和 Vulkan 输出是否几乎相等，若不等则显示相对误差
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言：验证最终结果的正确性
  ASSERT_TRUE(check);
}

// 测试例：在第2维度上拼接不同高度的张量，并验证 Vulkan 实现的正确性
TEST_F(VulkanAPITest, cat_4d_dim2_diffheight_success) {
  // Arrange: 设置不同高度的张量，用于测试拼接功能
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: 在维度2上拼接 CPU 张量，并通过 Vulkan API 在相同维度上拼接
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert: 检查 CPU 和 Vulkan 输出是否几乎相等，若不等则显示相对误差
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言：验证最终结果的正确性
  ASSERT_TRUE(check);
}

// 测试例：在第2维度上拼接单通道的张量，并验证 Vulkan 实现的正确性
TEST_F(VulkanAPITest, cat_4d_dim2_singledepth_success) {
  // Arrange: 设置单通道张量，用于测试拼接功能
  const auto in_cpu1 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: 在维度2上拼接 CPU 张量，并通过 Vulkan API 在相同维度上拼接
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert: 检查 CPU 和 Vulkan 输出是否几乎相等，若不等则显示相对误差
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言：验证最终结果的正确性
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_4d_dim2_invalidinputs_exceptions) {
  // Arrange: Vulkan cat inputs must have matching sizes except concatenated dimension
  {
    // 定义三个不同尺寸的CPU张量
    const auto in_cpu1 = at::rand({3, 5, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    // 使用Vulkan的cat函数尝试合并张量，期望抛出异常
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);
    }, ::std::exception);
  }

  // Arrange: Vulkan cat expects inputs of same dimensions
  {
    // 定义三个CPU张量，其中第二个张量维度与其他不同
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    // 使用Vulkan的cat函数尝试合并张量，期望抛出异常
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);
    }, ::std::exception);
  }
}

TEST_F(VulkanAPITest, cat_4d_dim3_invalidinputs_exceptions) {
  // Arrange: Vulkan cat inputs must have matching sizes except concatenated dimension
  {
    // 定义三个不同尺寸的CPU张量
    const auto in_cpu1 = at::rand({3, 5, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    // 使用Vulkan的cat函数尝试合并张量，期望抛出异常
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);
    }, ::std::exception);
  }

  // Arrange: Vulkan cat expects 4 dimensional inputs
  {
    // 定义三个CPU张量，其中第二个张量维度不是4维
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    // 使用Vulkan的cat函数尝试合并张量，期望抛出异常
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);
    }, ::std::exception);
  }
}

TEST_F(VulkanAPITest, cat_4d_dim3_samewidth_success) {
  // Arrange
  // 定义三个相同尺寸的CPU张量
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 使用CPU的cat函数合并张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 3);
  // 使用Vulkan的cat函数合并张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);

  // Assert
  // 检查两种合并结果是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两种合并结果应该几乎相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_4d_dim3_diffwidth_success) {
  // Arrange
  // 创建具有不同维度的三个张量，并指定它们的设备和数据类型
  const auto in_cpu1 = at::rand({3, 9, 193, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 193, 331}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 对三个张量沿第三维度进行拼接操作，生成新的 CPU 张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 3);
  // 调用 Vulkan API 对三个张量在第三维度上进行拼接，生成 Vulkan 张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);

  // Assert
  // 检查两个张量在指定容差范围内是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示实际和期望值之间的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个张量在指定容差范围内近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim0_mult4ch_success) {
  // Arrange
  // 创建具有相同维度的三个张量，并指定它们的设备和数据类型
  const auto in_cpu1 = at::rand({4, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({4, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({4, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 对三个张量沿第零维度进行拼接操作，生成新的 CPU 张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 调用 Vulkan API 对三个张量在第零维度上进行拼接，生成 Vulkan 张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  // 检查两个张量在指定容差范围内是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示实际和期望值之间的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个张量在指定容差范围内近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim0_diff_channel_success) {
  // Arrange
  // 创建具有不同维度的三个张量，并指定它们的设备和数据类型
  const auto in_cpu1 = at::rand({221, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({113, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({331, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 对三个张量沿第零维度进行拼接操作，生成新的 CPU 张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 调用 Vulkan API 对三个张量在第零维度上进行拼接，生成 Vulkan 张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  // 检查两个张量在指定容差范围内是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示实际和期望值之间的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个张量在指定容差范围内近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim0_same_channel_success) {
  // Arrange
  // 创建具有相同维度的三个张量，并指定它们的设备和数据类型
  const auto in_cpu1 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 对三个张量沿第零维度进行拼接操作，生成新的 CPU 张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 调用 Vulkan API 对三个张量在第零维度上进行拼接，生成 Vulkan 张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  // 检查两个张量在指定容差范围内是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，则展示实际和期望值之间的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个张量在指定容差范围内近似相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_3d_dim1_diffheight_success) {
  // Arrange

  // 生成大小为 [9, 221, 193] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 113, 193] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu2 = at::rand({9, 113, 193}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 331, 193] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu3 = at::rand({9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act

  // 按第 1 维度（列维度）拼接三个输入张量，得到输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 对每个输入张量应用 Vulkan API 并按第 1 维度拼接，得到输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert

  // 检查 out_cpu 和 out_vulkan 的值是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，则显示它们的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 out_cpu 和 out_vulkan 几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim1_same_height_success) {
  // Arrange

  // 生成大小为 [9, 193, 113] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 193, 113] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 193, 113] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu3 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act

  // 按第 1 维度（列维度）拼接三个输入张量，得到输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 对每个输入张量应用 Vulkan API 并按第 1 维度拼接，得到输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert

  // 检查 out_cpu 和 out_vulkan 的值是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，则显示它们的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 out_cpu 和 out_vulkan 几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim2_diffwidth_success) {
  // Arrange

  // 生成大小为 [9, 193, 221] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({9, 193, 221}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 193, 113] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 193, 331] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu3 = at::rand({9, 193, 331}, at::device(at::kCPU).dtype(at::kFloat));

  // Act

  // 按第 2 维度（行维度）拼接三个输入张量，得到输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  // 对每个输入张量应用 Vulkan API 并按第 2 维度拼接，得到输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert

  // 检查 out_cpu 和 out_vulkan 的值是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，则显示它们的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 out_cpu 和 out_vulkan 几乎相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim2_samewidth_success) {
  // Arrange

  // 生成大小为 [9, 193, 113] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu1 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 193, 113] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成大小为 [9, 193, 113] 的随机张量，存储在 CPU 上，数据类型为 float
  const auto in_cpu3 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act

  // 按第 2 维度（行维度）拼接三个输入张量，得到输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  // 对每个输入张量应用 Vulkan API 并按第 2 维度拼接，得到输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert

  // 检查 out_cpu 和 out_vulkan 的值是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不相等，则显示它们的相对容差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言 out_cpu 和 out_vulkan 几乎相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_3d_dim0_negdim_success) {
  // Arrange
  // 创建三个不同大小的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({221, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({113, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({331, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 0 维度上拼接三个输入张量，维度参数为 -3
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -3);
  // 使用 Vulkan API 在第 0 维度上拼接三个输入张量的 Vulkan 版本，维度参数为 -3
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -3);

  // Assert
  // 检查两个拼接结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不相等，展示其相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个拼接结果近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim1_negdim_success) {
  // Arrange
  // 创建三个不同大小的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 113, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 1 维度上拼接三个输入张量，维度参数为 -2
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -2);
  // 使用 Vulkan API 在第 1 维度上拼接三个输入张量的 Vulkan 版本，维度参数为 -2
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -2);

  // Assert
  // 检查两个拼接结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不相等，展示其相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个拼接结果近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim2_negdim_success) {
  // Arrange
  // 创建三个不同大小的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({193, 13, 89}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 13, 59}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 13, 67}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 2 维度上拼接三个输入张量，维度参数为 -1
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  // 使用 Vulkan API 在第 2 维度上拼接三个输入张量的 Vulkan 版本，维度参数为 -1
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  // 检查两个拼接结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不相等，展示其相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个拼接结果近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim0_same_height_success) {
  // Arrange
  // 创建三个相同大小的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在第 0 维度上拼接三个输入张量，维度参数为 0
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 使用 Vulkan API 在第 0 维度上拼接三个输入张量的 Vulkan 版本，维度参数为 0
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  // 检查两个拼接结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不相等，展示其相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言两个拼接结果近似相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_2d_dim0_diff_height_success) {
  // Arrange
  // 创建三个不同高度的随机张量，存储在CPU上
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({191, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({137, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 沿着第0维度拼接三个CPU张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 在Vulkan上执行相同操作，拼接三个Vulkan张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  // 检查CPU和Vulkan操作结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言结果应当是近似相等的
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim1_same_width_success) {
  // Arrange
  // 创建三个具有相同宽度的随机CPU张量
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 沿着第1维度拼接三个CPU张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 在Vulkan上执行相同操作，拼接三个Vulkan张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert
  // 检查CPU和Vulkan操作结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言结果应当是近似相等的
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim1_diff_width_success) {
  // Arrange
  // 创建三个具有不同宽度的随机CPU张量
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 131}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 127}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 沿着第1维度拼接三个CPU张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  // 在Vulkan上执行相同操作，拼接三个Vulkan张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert
  // 检查CPU和Vulkan操作结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言结果应当是近似相等的
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim0_negdim_success) {
  // Arrange
  // 创建三个具有不同尺寸的随机CPU张量
  const auto in_cpu1 = at::rand({113, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({131, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({127, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 沿着负第2维度（倒数第二维度）拼接三个CPU张量
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -2);
  // 在Vulkan上执行相同操作，拼接三个Vulkan张量
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -2);

  // Assert
  // 检查CPU和Vulkan操作结果的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果结果不近似相等，则展示相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言结果应当是近似相等的
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, cat_2d_dim1_negdim_success) {
  // Arrange
  // 创建三个不同尺寸的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 131}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 127}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在维度 1 上对三个输入张量进行拼接，生成输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  // 使用 Vulkan API 在维度 1 上对三个输入张量进行拼接，生成 Vulkan 输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  // 检查 out_cpu 和 out_vulkan 的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不满足近似相等性，则显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_1d_dim0_same_width_success) {
  // Arrange
  // 创建三个长度相同的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在维度 0 上对三个输入张量进行拼接，生成输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 使用 Vulkan API 在维度 0 上对三个输入张量进行拼接，生成 Vulkan 输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  // 检查 out_cpu 和 out_vulkan 的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不满足近似相等性，则显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_1d_dim0_diff_width_success) {
  // Arrange
  // 创建三个长度不同的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({137}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({131}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在维度 0 上对三个输入张量进行拼接，生成输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  // 使用 Vulkan API 在维度 0 上对三个输入张量进行拼接，生成 Vulkan 输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  // 检查 out_cpu 和 out_vulkan 的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不满足近似相等性，则显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_1d_dim0_negdim_success) {
  // Arrange
  // 创建三个长度不同的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu1 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({137}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({131}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 在维度 -1（最后一个维度）上对三个输入张量进行拼接，生成输出张量 out_cpu
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  // 使用 Vulkan API 在维度 -1（最后一个维度）上对三个输入张量进行拼接，生成 Vulkan 输出张量 out_vulkan
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  // 检查 out_cpu 和 out_vulkan 的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不满足近似相等性，则显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, permute_2d_success) {
  // Arrange
  // 创建一个 2x3 的随机张量，设备为 CPU，数据类型为 float
  const auto in_cpu = at::rand({2, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  // 对输入张量进行维度置换，维度顺序为 {1, 0}，生成输出张量 out_cpu
  const auto out_cpu = at::permute(in_cpu, {1, 0});
  // 使用 Vulkan API 对输入张量进行维度置换，维度顺序为 {1, 0}，生成 Vulkan 输出张量 out_vulkan
  const auto out_vulkan = at::permute(in_cpu.vulkan(), {1, 0});

  // Assert
  // 检查 out_cpu 和 out_vulkan 的近似相等性
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果不满足近似相等性，则显示它们的相对误差
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, permute_3d_success) {
  // Arrange: 设置测试所需的初始条件
  const auto in_cpu = at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建存储所有排列组合维度的向量
  std::vector<std::vector<int64_t>> all_dims;
  // 定义输入张量的维度顺序
  std::vector<int64_t> in{0, 1, 2};
  // 生成所有可能的排列组合
  gen_allpermutations(all_dims, in, 0);

  // 遍历所有的维度排列组合
  for (const auto i : c10::irange(1, all_dims.size())) {
    const auto dims = all_dims[i];

    // Act: 执行操作
    const auto out_cpu = at::permute(in_cpu, dims);
    const auto out_vulkan = at::permute(in_cpu.vulkan(), dims);

    // Assert: 断言检查输出是否几乎相等
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    // 如果检查不通过，展示相对误差并输出
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    // 断言检查是否通过
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, permute_4d_success) {
  // Arrange: 设置测试所需的初始条件
  const auto in_cpu = at::rand({2, 3, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建存储所有排列组合维度的向量
  std::vector<std::vector<int64_t>> all_dims;
  // 定义输入张量的维度顺序
  std::vector<int64_t> in{0, 1, 2, 3};
  // 生成所有可能的排列组合
  gen_allpermutations(all_dims, in, 0);

  // 遍历所有的维度排列组合
  for (const auto i : c10::irange(1, all_dims.size())) {
    const auto dims = all_dims[i];

    // Act: 执行操作
    const auto out_cpu = at::permute(in_cpu, dims);
    const auto out_vulkan = at::permute(in_cpu.vulkan(), dims);

    // Assert: 断言检查输出是否几乎相等
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    // 如果检查不通过，展示相对误差并输出
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    // 断言检查是否通过
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, permute_4dmclaren_success) {
  // Arrange: 设置测试所需的初始条件，这里标注了McLaren模型的使用
  const auto in_cpu = at::rand({1, 2, 1, 161}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: 执行操作
  const auto out_cpu = at::permute(in_cpu, {0, 2, 1, 3});
  const auto out_vulkan = at::permute(in_cpu.vulkan(), {0, 2, 1, 3});

  // Assert: 断言检查输出是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，展示相对误差并输出
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查是否通过
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, permute_4dbig_success) {
  // Arrange: 设置测试所需的初始条件
  const auto in_cpu = at::rand({3, 9, 51, 41}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建存储所有排列组合维度的向量
  std::vector<std::vector<int64_t>> all_dims;
  // 定义输入张量的维度顺序
  std::vector<int64_t> in{0, 1, 2, 3};
  // 生成所有可能的排列组合
  gen_allpermutations(all_dims, in, 0);

  // 遍历所有的维度排列组合
  for (const auto i : c10::irange(1, all_dims.size())) {
    const auto dims = all_dims[i];

    // Act: 执行操作
    const auto out_cpu = at::permute(in_cpu, dims);
    const auto out_vulkan = at::permute(in_cpu.vulkan(), dims);

    // Assert: 断言检查输出是否几乎相等
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    // 如果检查不通过，展示相对误差并输出
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    // 断言检查是否通过
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, permute_negativedims_success) {
  // Arrange: 设置测试所需的初始条件
  const auto in_cpu = at::rand({5, 4, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: 执行操作，使用负数维度
  // {-1,-2,-3,0} 等效于 {3,2,1,0}
  const auto out_cpu = at::permute(in_cpu, {-1, -2, -3, 0});
  const auto out_vulkan = at::permute(in_cpu.vulkan(), {-1, -2, -3, 0});

  // Assert: 断言检查输出是否几乎相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  // 如果检查不通过，展示相对误差并输出
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  // 断言检查是否通过
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, permute_invalidinputs_exceptions) {
  // Arrange
  // 创建一个形状为 {1, 2, 1, 161} 的CPU张量，并填充随机数据
  const auto in_cpu = at::rand({1, 2, 1, 161}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: Repeated dim
  // 期望抛出异常：重复的维度索引 {2, 2, 1, 0} 调用 permute
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {2, 2, 1, 0});
  }, ::std::exception);

  // 期望抛出异常：调用 permute 但重复维度 {2, 2, 1, 0}
  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({2, 2, 1, 0});
  }, ::std::exception);

  // Act: Number of dims don't match
  // 期望抛出异常：维度数量不匹配 {4, 3, 2, 1, 0} 调用 permute
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {4, 3, 2, 1, 0});
  }, ::std::exception);

  // 期望抛出异常：维度数量不匹配 {2, 1, 0} 调用 permute
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {2, 1, 0});
  }, ::std::exception);

  // 期望抛出异常：调用 permute 但维度数量不匹配 {4, 3, 2, 1, 0}
  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({4, 3, 2, 1, 0});
  }, ::std::exception);

  // 期望抛出异常：调用 permute 但维度数量不匹配 {2, 1, 0}
  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({2, 1, 0});
  }, ::std::exception);

  // Act: Dim out of range
  // 期望抛出异常：维度索引超出范围 {5, 2, 1, 0} 调用 permute
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {5, 2, 1, 0});
  }, ::std::exception);

  // 期望抛出异常：调用 permute 但维度索引超出范围 {5, 2, 1, 0}
  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({5, 2, 1, 0});
  }, ::std::exception);

  // Act: Input tensor size > 4D
  // 创建一个形状为 {1, 2, 1, 2, 161} 的CPU张量，并填充随机数据
  const auto in_cpu_5d = at::rand({1, 2, 1, 2, 161}, at::device(at::kCPU).dtype(at::kFloat));

  // 期望抛出异常：5维输入张量调用 permute
  EXPECT_THROW({
    const auto out_vulkan_5d = at::permute(in_cpu_5d.vulkan(), {4, 3, 2, 1, 0});
  }, ::std::exception);

  // 期望抛出异常：调用 permute 但5维输入张量
  EXPECT_THROW({
    const auto out_vulkan_5d = in_cpu_5d.vulkan();
    out_vulkan_5d.permute({4, 3, 2, 1, 0});
  }, ::std::exception);
}

TEST_F(VulkanAPITest, slice_width_success) {
  // Arrange
  // 创建一个映射，指定不同维度对应的张量尺寸
  std::unordered_map<int64_t, std::vector<int64_t>> dim2sizes {
    {3, {2, 3, 40, 50}},  // 4维张量，其中维度3对应宽度
    {2, {3, 40, 50}},     // 3维张量，其中维度2对应宽度
    {1, {40, 50}},        // 2维张量，其中维度1对应宽度
    {0, {50}},            // 1维张量，其中维度0对应宽度
  };

  // Act/Assert
  // 调用 slice_tests 函数对所有映射进行切片测试
  slice_tests(dim2sizes);
}

TEST_F(VulkanAPITest, slice_height_success) {
  // Arrange
  // 创建一个映射，指定不同维度对应的张量尺寸
  std::unordered_map<int64_t, std::vector<int64_t>> dim2sizes {
    {2, {2, 3, 40, 50}},  // 4维张量，其中维度2对应高度
    {1, {3, 40, 50}},     // 3维张量，其中维度1对应高度
    {0, {40, 50}},        // 2维张量，其中维度0对应高度
                          // 1维张量不含高度维度，用于测试
  };

  // Act/Assert
  // 调用 slice_tests 函数对所有映射进行切片测试
  slice_tests(dim2sizes);
}

TEST_F(VulkanAPITest, slice_feature_success) {
  // Arrange
  // 创建一个映射，指定不同维度对应的张量尺寸
  std::unordered_map<int64_t, std::vector<int64_t>> dim2sizes {
    {1, {2, 40, 13, 14}}, // 4维张量，其中维度1对应特征（通道）
    {0, {40, 13, 14}},    // 3维张量，其中维度0对应特征（通道）
                          // 1维和2维张量不含特征（通道）维度，用于测试
  };

  // Act/Assert
  // 调用 slice_tests 函数对所有映射进行切片测试
  slice_tests(dim2sizes);
}

TEST_F(VulkanAPITest, slice_batch_success) {
  // Arrange
  // 创建一个映射，指定不同维度对应的张量尺寸
    {
        0, {40, 3, 13, 14}
    }, 
    // 定义一个集合，包含一个索引为0的键值对，值是包含4个元素的集合，表示一个4维张量的尺寸，其中第一个维度是批处理维度
    
    // 以下是注释的示例
    // 1D、2D和3D张量不具备批处理维度，这些测试是为了验证这一点
    };
    
    // Act/Assert
    // 调用slice_tests函数，传递dim2sizes作为参数
    slice_tests(dim2sizes);
TEST_F(VulkanAPITest, slice_zero_sized) {
  // 测试零长度切片情况
  slice_test({2, 3, 4, 5}, 3, 0, 0, 1);
  // 测试起始大于结束的切片情况
  slice_test({2, 3, 4, 5}, 3, 3, 2, 1);
}

TEST_F(VulkanAPITest, slice_invalidinputs_exceptions) {
  // 行为测试：切片步长必须为正数
  EXPECT_THROW({
    slice_test({2, 3, 4, 5}, 3, 0, 3, 0);
  }, ::std::exception);
}

TEST_F(VulkanAPITest, stack_invalid_inputs) {
  // 行为测试：Vulkan堆栈至少需要一个张量
  EXPECT_THROW({
    at::stack({}, 0);
  }, ::std::exception);

  // 行为测试：Vulkan堆栈的输入张量必须具有匹配的尺寸
  EXPECT_THROW({
    at::stack({
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({6, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan()}, 0);
  }, ::std::exception);
}

void test_stack(const at::IntArrayRef input_shape, int64_t dim, int numTensors) {
  // 创建CPU和Vulkan张量的向量
  std::vector<at::Tensor> tensors_cpu = {};
  std::vector<at::Tensor> tensors_vulkan = {};

  // 循环创建指定数量的张量
  for (int i = 0; i < numTensors; i++) {
    // 创建具有指定形状和类型的CPU张量
    at::Tensor in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
    tensors_cpu.emplace_back(in_cpu);
    // 将CPU张量转换为Vulkan张量并添加到向量中
    tensors_vulkan.emplace_back(in_cpu.vulkan());
  }

  // 在指定维度上堆叠CPU和Vulkan张量
  at::Tensor out_cpu = at::stack(tensors_cpu, 0);
  at::Tensor out_vulkan = at::stack(tensors_vulkan, 0);
  // 检查CPU和Vulkan张量堆叠后是否近似相等
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Error when stacking " << numTensors << " tensors" << std::endl;
    showRtol(out_cpu, out_vulkan.cpu());
  }
  // 断言CPU和Vulkan张量堆叠后近似相等
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, stack_0d) {
  // 测试0维堆叠
  test_stack({}, 0, 1);
  test_stack({}, 0, 2);
  test_stack({}, 0, 3);
}

TEST_F(VulkanAPITest, stack_1d) {
  // 测试1维堆叠
  test_stack({221}, 0, 2);
  test_stack({193}, 1, 3);

  test_stack({221}, -1, 2);
  test_stack({193}, -2, 3);
}

TEST_F(VulkanAPITest, stack_2d) {
  // 测试2维堆叠
  test_stack({221, 193}, 0, 2);
  test_stack({221, 193}, 1, 3);
  test_stack({221, 193}, 2, 4);

  test_stack({221, 193}, -1, 2);
  test_stack({221, 193}, -2, 3);
  test_stack({221, 193}, -3, 4);
}

TEST_F(VulkanAPITest, stack_3d) {
  // 测试3维堆叠
  test_stack({221, 193, 11}, 0, 2);
  test_stack({221, 193, 11}, 1, 3);
  test_stack({221, 193, 11}, 2, 4);
  test_stack({221, 193, 11}, 3, 5);

  test_stack({221, 193, 11}, -1, 2);
  test_stack({221, 193, 11}, -2, 3);
  test_stack({221, 193, 11}, -3, 4);
  test_stack({221, 193, 11}, -4, 5);
}

TEST_F(VulkanAPITest, tile_invalid_inputs_exceptions) {
  // 安排：Vulkan瓦片仅支持输入维度 <= 4
  {
    const auto in_cpu =
        at::rand({3, 9, 5, 7, 3}, at::device(at::kCPU).dtype(at::kFloat));
    const at::IntArrayRef repeats = {7, 3, 9, 2};

    // 行为测试
    EXPECT_THROW(
        { const auto out_vulkan = at::tile(in_cpu.vulkan(), repeats); },
        ::std::exception);
  }
}

TEST_F(VulkanAPITest, tile_invalid_outpus_exceptions) {
  // 安排：Vulkan瓦片仅支持输出维度 <= 4
  {
    // 创建一个形状为[3, 9, 5, 13]的随机张量，数据类型为float，存储在CPU上
    const auto in_cpu =
        at::rand({3, 9, 5, 13}, at::device(at::kCPU).dtype(at::kFloat));
    // 定义一个包含重复次数的整数数组 {5, 7, 3, 9, 2}
    const at::IntArrayRef repeats = {5, 7, 3, 9, 2};

    // 断言：尝试执行以下代码块，期望抛出异常 ::std::exception
    EXPECT_THROW(
        // 在Vulkan上执行 in_cpu 张量的重复操作，使用 repeats 数组指定重复次数
        { const auto out_vulkan = at::tile(in_cpu.vulkan(), repeats); },
        ::std::exception);
}

// 定义一个测试函数，用于测试 tile 操作
void test_tile(
    const at::IntArrayRef input_shape,  // 输入张量的形状
    const at::IntArrayRef repeats) {    // 重复次数数组
  c10::InferenceMode mode;  // 进入推断模式

  at::Tensor in_cpu;        // 定义 CPU 上的输入张量
  at::Tensor out_cpu;       // 定义 CPU 上的输出张量
  at::Tensor in_vulkan;     // 定义 Vulkan 上的输入张量
  at::Tensor out_vulkan;    // 定义 Vulkan 上的输出张量
  at::IntArrayRef repeat;   // 重复次数数组的引用
  bool check = true;        // 检查结果标志，初始为 true

  // 循环遍历输入张量的每个维度
  for (int idx_input = 1; (unsigned)idx_input < input_shape.size() + 1; ++idx_input) {
    // 循环遍历重复次数数组的每个维度
    for (int idx_repeat = 1; (unsigned)idx_repeat < repeats.size() + 1; ++idx_repeat) {
      // 在 CPU 上生成随机数据的输入张量，并指定设备和数据类型
      in_cpu = at::rand(
          input_shape.slice(0, idx_input),
          at::device(at::kCPU).dtype(at::kFloat));
      // 从重复次数数组中取出部分切片作为当前重复次数
      repeat = repeats.slice(0, idx_repeat);
      // 在 CPU 上对输入张量进行 tile 操作
      out_cpu = at::tile(in_cpu, repeat);
      // 将 CPU 上的输入张量转换为 Vulkan 张量
      in_vulkan = in_cpu.vulkan();
      // 在 Vulkan 上对输入张量进行 tile 操作
      out_vulkan = at::tile(in_vulkan, repeat);
      // 检查 CPU 和 Vulkan 上的输出张量是否几乎相等
      check = almostEqual(out_cpu, out_vulkan.cpu());
      // 如果不相等，则输出错误信息并显示相对误差
      if (!check) {
        check = false;
        std::cout << "Tile test failed when input is of shape "
                  << input_shape.slice(0, idx_input) << " and repeat of "
                  << repeat << std::endl;
        showRtol(out_cpu, out_vulkan.cpu());
      }
    }
  }

  // 断言检查结果为 true
  ASSERT_TRUE(check);
}

// 定义 Vulkan API 的 tile 测试
TEST_F(VulkanAPITest, tile) {
  // 调用 test_tile 函数进行测试，给定输入形状和重复次数
  test_tile({13, 5, 13, 7}, {7, 2, 3, 5});
}

// 定义测试函数，用于测试 zero_ 操作
void test_zero_(const at::IntArrayRef input_shape) {
  // 在 CPU 上生成指定形状的随机张量
  auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  // 将 CPU 张量转换为 Vulkan 张量
  auto vulkan = cpu.vulkan();

  // 在 CPU 和 Vulkan 上执行 zero_ 操作
  cpu.zero_();
  vulkan.zero_();

  // 检查 CPU 和 Vulkan 张量是否几乎相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果不相等，则输出错误信息并显示相对误差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "zero_ test failed with input shape: "
              << input_shape << std::endl;
  }
  // 断言检查结果为 true
  ASSERT_TRUE(check);
}

// 定义 Vulkan API 的 zero_ 测试
TEST_F(VulkanAPITest, zero_) {
  // 分别对多种输入形状调用 test_zero_ 函数进行测试
  test_zero_({5});
  test_zero_({5, 7});
  test_zero_({9, 7, 5});
  test_zero_({22, 11, 19, 17});
}

// 定义测试函数，用于测试 zeros 函数
void test_zeros(const at::IntArrayRef input_shape) {
  // 在 CPU 和 Vulkan 上分别生成指定形状的全零张量
  auto cpu = at::zeros(input_shape);
  auto vulkan = at::zeros(input_shape, at::device(at::kVulkan));

  // 检查 CPU 和 Vulkan 张量是否几乎相等
  const auto check = almostEqual(cpu, vulkan.cpu());
  // 如果不相等，则输出错误信息并显示相对误差
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "zeros test failed with input shape: "
              << input_shape << std::endl;
  }
  // 断言检查结果为 true
  ASSERT_TRUE(check);
}

// 定义 Vulkan API 的 zeros 测试
TEST_F(VulkanAPITest, zeros) {
  // 分别对多种输入形状调用 test_zeros 函数进行测试
  test_zeros({5});
  test_zeros({5, 7});
  test_zeros({9, 7, 5});
  test_zeros({22, 11, 19, 17});
}

// 定义成功克隆测试函数
TEST_F(VulkanAPITest, clone_success) {
  // 准备测试数据：多映射的内存格式与尺寸组合
  std::multimap<std::optional<c10::MemoryFormat>, std::vector<int64_t>> mem2sizes {
    {c10::MemoryFormat::Preserve, {2, 3, 5, 161}},    // 保留内存格式的四维张量
    {c10::MemoryFormat::Contiguous, {2, 3, 5, 161}},  // 连续内存格式的四维张量
    {{}, {2, 3, 5, 161}},                             // 空内存格式的四维张量
    {c10::MemoryFormat::Preserve, {3, 5, 161}},       // 保留内存格式的三维张量
    {c10::MemoryFormat::Contiguous, {3, 5, 161}},     // 连续内存格式的三维张量
    {{}, {3, 5, 161}},                                // 空内存格式的三维张量
    {
      c10::MemoryFormat::Preserve, {5, 161}},          // 保留内存格式的二维张量，尺寸为 {5, 161}
    {
      c10::MemoryFormat::Contiguous, {5, 161}},        // 连续内存格式的二维张量，尺寸为 {5, 161}
    {
      {}, {5, 161}},                                   // 二维张量，内存格式为空，尺寸为 {5, 161}
    {
      c10::MemoryFormat::Preserve, {161}},             // 保留内存格式的一维张量，尺寸为 {161}
    {
      c10::MemoryFormat::Contiguous, {161}},           // 连续内存格式的一维张量，尺寸为 {161}
    {
      {}, {161}},                                      // 一维张量，内存格式为空，尺寸为 {161}
    };
    
    // Act/Assert
    // 对于每个 mem2size 中的条目，执行 clone_test 函数，参数为条目的第二个元素（尺寸）和第一个元素（内存格式）
    for (const auto& mem2size : mem2sizes) {
      clone_test(mem2size.second, mem2size.first);
    }
}

// 在 VulkanAPITest 类中定义一个测试用例，测试 clone_test 函数对无效输入是否会抛出异常
TEST_F(VulkanAPITest, clone_invalidinputs_exceptions) {
  // Act: 调用 clone_test 函数，期望它抛出 std::exception 异常
  EXPECT_THROW({
    clone_test({2, 3, 5, 161}, c10::MemoryFormat::ChannelsLast);
  }, ::std::exception);

  // Act: 调用 clone_test 函数，期望它抛出 std::exception 异常
  EXPECT_THROW({
    clone_test({2, 3, 5, 161}, c10::MemoryFormat::ChannelsLast3d);
  }, ::std::exception);
}

// 定义一个枚举类型 OpType，表示不同的操作类型
enum class OpType {
  addmm,      // 矩阵乘法加法操作
  conv2d,     // 二维卷积操作
  hardtanh_,  // 硬切线函数操作
  mean,       // 平均值计算操作
};

// BaseOp 类是一个抽象基类，定义了操作的接口
class BaseOp {
 public:
  // 构造函数，初始化操作类型
  explicit BaseOp(const OpType) {}
  // 虚析构函数，用于多态
  virtual ~BaseOp() = default;

  // 纯虚函数，子类需要实现的运行函数接口
  virtual at::Tensor run(at::Tensor&) const = 0;
  // 纯虚函数，子类需要实现的返回描述字符串接口
  virtual std::string toString() const = 0;
};

// Addmm 类继承自 BaseOp 类，实现了矩阵乘法加法操作
class Addmm final : public BaseOp {
 public:
  // 构造函数，初始化操作需要的参数和数据
  Addmm(
      const int64_t m1H,
      const int64_t m1W,
      const int64_t m2W,
      const float beta,
      const float alpha)
    : BaseOp(OpType::addmm),
      // 初始化随机生成的张量 m2_ 和 b_
      m2_(at::rand(c10::IntArrayRef({m1W, m2W}), at::device(at::kCPU).dtype(at::kFloat))),
      b_(at::rand(c10::IntArrayRef({m1H, m2W}), at::device(at::kCPU).dtype(at::kFloat))),
      beta_(beta),
      alpha_(alpha) {
  }

  // 实现基类的虚函数，执行矩阵乘法加法操作
  at::Tensor run(at::Tensor& t) const override {
    // 如果张量 t 使用 Vulkan 运算，则调用 Vulkan 支持的加法乘法操作
    if (t.is_vulkan()) {
      return at::addmm(b_, t, m2_, beta_, alpha_);
    }

    // 否则调用默认的加法乘法操作
    return at::addmm(b_, t, m2_, beta_, alpha_);
  }

  // 返回操作的描述字符串
  std::string toString() const override {
    return "addmm";
  }

 private:
  at::Tensor m2_;  // 第二个矩阵张量
  at::Tensor b_;   // 偏置张量
  float beta_;     // beta 参数
  float alpha_;    // alpha 参数
};

// Conv2d 类继承自 BaseOp 类，实现了二维卷积操作
class Conv2d final : public BaseOp {
 public:
  // 构造函数，初始化操作需要的参数和数据
  Conv2d(
      const c10::IntArrayRef wsizes,
      const int64_t groups,
      const int64_t stride,
      const int64_t padding)
      : BaseOp(OpType::conv2d),
        groups_(groups),
        stride_(stride),
        padding_(padding),
        // 初始化随机生成的权重张量 w_ 和偏置张量 b_
        w_(at::rand(wsizes, at::device(at::kCPU).dtype(at::kFloat))),
        b_(at::rand(wsizes[0], at::device(at::kCPU).dtype(at::kFloat))) {
  }

  // 实现基类的虚函数，执行二维卷积操作
  at::Tensor run(at::Tensor& t) const override {
    return at::conv2d(t, w_, b_, {stride_}, {padding_}, {1}, groups_);
  }

  // 返回操作的描述字符串
  std::string toString() const override {
    return "conv2d";
  }

 private:
  int64_t groups_;    // 分组数量
  int64_t stride_;    // 步幅
  int64_t padding_;   // 填充
  at::Tensor w_;      // 权重张量
  at::Tensor b_;      // 偏置张量
};

// Hardtanh_ 类继承自 BaseOp 类，实现了硬切线函数操作
class Hardtanh_ final : public BaseOp {
 public:
  // 构造函数，初始化操作类型
  Hardtanh_() : BaseOp(OpType::hardtanh_) {}

  // 实现基类的虚函数，执行硬切线函数操作
  at::Tensor run(at::Tensor& input) const override {
    return at::hardtanh_(input, 0, 6);
  }

  // 返回操作的描述字符串
  std::string toString() const override {
    return "hardtanh_";
  }
};

// Mean 类继承自 BaseOp 类，实现了计算均值操作
class Mean final : public BaseOp {
 public:
  // 构造函数，初始化操作类型
  Mean() : BaseOp(OpType::mean) {}

  // 实现基类的虚函数，执行计算均值操作
  at::Tensor run(at::Tensor& input) const override {
    return at::mean(input, {2, 3}, false);
  }

  // 返回操作的描述字符串
  std::string toString() const override {
    return "mean";
  }
};

// OpsList 类封装了多个操作，可以依次对输入数据执行这些操作
class OpsList {
 public:
  OpsList() {}
  explicit OpsList(std::vector<std::unique_ptr<BaseOp>> ops)
    : ops_(std::move(ops)) {
  }

  // 执行所有操作并返回最终的输出张量
  auto run(const at::Tensor& input) {
    at::Tensor output = input;

    // 遍历所有操作并依次执行
    for (const auto& op : ops_) {
      output = op->run(output);
    }

    // 返回最终的输出张量
    return output;
  }
    return output;
  }



    // 返回当前函数的输出
    return output;
  }



  auto run(const at::Tensor& input, const at::Tensor& v_input) {
    // 将输入张量赋给输出张量
    at::Tensor output = input;
    // 将输入的v_input张量赋给v_output张量
    at::Tensor v_output = v_input;

    // 遍历操作列表中的每个操作
    for (const auto& op : ops_) {
      // 对当前输出张量应用操作，并更新输出张量
      output = op->run(output);
      // 对当前v_output张量应用操作，并更新v_output张量
      v_output = op->run(v_output);
    }

    // 返回更新后的输出张量和v_output张量的组合
    return std::make_pair(output, v_output);
  }



  // 保护部分：定义了一个存储BaseOp对象指针的向量
 protected:
  std::vector<std::unique_ptr<BaseOp>> ops_;



  // 保护部分：定义了一个存储BaseOp对象指针的向量
 protected:
  std::vector<std::unique_ptr<BaseOp>> ops_;
};

class MobileNetV2 final : public OpsList {
 public:
  MobileNetV2() {
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {32, 3, 3, 3}，步幅为 1，填充为 2
    ops_.emplace_back(new Conv2d({32, 3, 3, 3}, 1, 2, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {32, 1, 3, 3}，步幅为 1，填充为 1
    ops_.emplace_back(new Conv2d({32, 1, 3, 3}, 32, 1, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {16, 32, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({16, 32, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {96, 16, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({96, 16, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {96, 1, 3, 3}，步幅为 2，填充为 1
    ops_.emplace_back(new Conv2d({96, 1, 3, 3}, 96, 2, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {24, 96, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({24, 96, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {144, 24, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {144, 1, 3, 3}，步幅为 1，填充为 1
    ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 1, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {24, 144, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({24, 144, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {144, 24, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {144, 1, 3, 3}，步幅为 2，填充为 1
    ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 2, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {32, 144, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({32, 144, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {192, 32, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {192, 1, 3, 3}，步幅为 1，填充为 1
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {32, 192, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {192, 32, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {192, 1, 3, 3}，步幅为 1，填充为 1
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {32, 192, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {192, 32, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {192, 1, 3, 3}，步幅为 1，填充为 1
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    // 向 ops_ 中添加一个 Hardtanh_ 操作
    ops_.emplace_back(new Hardtanh_());
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {32, 192, 1, 1}，步幅为 1，填充为 0
    ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    // 向 ops_ 中添加一个新的 Conv2d 操作，参数为 {192, 32, 1, 1}，步幅为 1，填充为 0
    ops_.emplace
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{576, 96, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{576, 1, 3, 3}），输入通道数为 576，步长为 1
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{96, 576, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{576, 96, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{576, 1, 3, 3}），输入通道数为 576，步长为 1
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{96, 576, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{576, 96, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{576, 1, 3, 3}），输入通道数为 576，步长为 2
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 2, 1));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{160, 576, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({160, 576, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{960, 160, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{960, 1, 3, 3}），输入通道数为 960，步长为 1
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{160, 960, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{960, 160, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{960, 1, 3, 3}），输入通道数为 960，步长为 1
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{160, 960, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{960, 160, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{960, 1, 3, 3}），输入通道数为 960，步长为 1
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{320, 960, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({320, 960, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Conv2d 操作到 ops_ 列表，该操作使用指定的参数（{1280, 320, 1, 1}），步长为 1，填充为 0
    ops_.emplace_back(new Conv2d({1280, 320, 1, 1}, 1, 1, 0));
    
    # 添加一个新的 Hardtanh_ 操作到 ops_ 列表
    ops_.emplace_back(new Hardtanh_());
    
    # 添加一个新的 Mean 操作到 ops_ 列表
    ops_.emplace_back(new Mean());
    
    # 添加一个新的 Addmm 操作到 ops_ 列表，使用指定的参数（1, 1280, 1000, 0, 1）
    ops_.emplace_back(new Addmm(1, 1280, 1000, 0, 1));
  // 定义一个名为 weight_hh_l 的列表，存储每一层的隐藏状态到隐藏状态的权重张量
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  // 定义一个名为 bias_ih_l 的列表，存储每一层的输入到隐藏状态的偏置张量
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  // 循环遍历每一层，初始化权重和偏置张量列表
  for (int i = 0; i < num_layers; ++i) {
    // 对于第一层，根据输入大小初始化输入到隐藏状态的权重张量
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      // 对于其他层，根据隐藏状态大小初始化输入到隐藏状态的权重张量
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    // 初始化隐藏状态到隐藏状态的权重张量
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    // 初始化输入到隐藏状态的偏置张量
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));

// 将一个新的偏置张量添加到 `bias_hh_l` 后面，其大小为 `3 * H_out`，数据类型为浮点型，存储在 CPU 上。


  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

// 引入此保护措施以便于运行推断而不是训练，以避免以下错误：
//     在 "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp" 的位置内部断言失败，详细信息请向 PyTorch 报告。aten::gru.input 注册了既属于 CompositeImplicitAutograd 又映射到 AutogradOther 后端的核心。这使得后端核心不可达；调度程序将始终优先选择 CompositeImplicitAutograd 降级（参见注释 [Ambiguity in AutogradOther kernel]）。如果要覆盖 CompositeImplicitAutograd，请提出问题请求专用的 Autograd 调度键给后端。
//     如果只想运行推断而不是训练，请在 model.forward() 前添加 `c10::InferenceMode mode;`。请注意，此保护措施目前仅在 C++ 中可用，而不是在 Python 中。


  // Act
  const auto out_cpu = at::gru(in_cpu, h0_cpu,
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
        weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1],
        weight_ih_l[2], weight_hh_l[2], bias_ih_l[2], bias_hh_l[2] },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

// 在 CPU 上运行 GRU 模型，计算出输出 `out_cpu`，输入参数包括 `in_cpu` 作为输入张量，`h0_cpu` 作为初始隐藏状态，以及多组权重和偏置。


  // weights/biases should be always on CPU.
  const auto out_vulkan = at::gru(in_cpu.vulkan(), h0_cpu.vulkan(),
      { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
        weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1),
        weight_ih_l.get(2), weight_hh_l.get(2), bias_ih_l.get(2), bias_hh_l.get(2) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

// 在 Vulkan 后端上运行 GRU 模型，计算出输出 `out_vulkan`，输入参数与 CPU 上的计算相同，确保权重和偏置始终在 CPU 上。


  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto vulkan_output = std::get<0>(out_vulkan);
  auto vulkan_hidden = std::get<1>(out_vulkan);

// 从输出元组中提取 CPU 和 Vulkan 后端的输出和隐藏状态。


  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);

// 断言：检查 CPU 和 Vulkan 后端的输出和隐藏状态是否几乎相等，如果不相等，则展示它们之间的相对误差，并断言检查通过。
// 在 VulkanAPITest 测试固件中定义 gru_mclareninputs_success 测试用例
TEST_F(VulkanAPITest, gru_mclareninputs_success) {
  // 设置输入大小为 384
  const int H_in = 384;  // input_size
  // 设置隐藏层大小为 384
  const int H_out = 384; // hidden_size
  // 设置 RNN 层的数量为 2
  const int num_layers = 2;
  // 设置序列长度为 1
  const int L = 1;
  // 设置批次大小为 1
  const int N = 1;
  // GRU 的 dropout 率设为 0
  const double gru_dropout = .0;
  // 是否包含偏置
  const bool has_biases = true;
  // 是否训练模型（此处为 false，表示测试阶段）
  const bool train = false;
  // 是否双向 GRU（此处为 false，表示单向）
  const bool bidirectional = false;
  // 是否批次优先
  const bool batch_first = true;
  // 创建输入张量，形状为 (N, L, H_in)，CPU 上随机初始化
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建初始隐藏状态张量，形状为 (num_layers, N, H_out)，CPU 上随机初始化
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  // 初始化存储权重和偏置的列表
  c10::List<at::Tensor> weight_ih_l; // 形状为 (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // 形状为 (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // 形状为 (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // 形状为 (3 * hidden_size)

  // 循环创建每个 RNN 层的权重和偏置张量
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      // 第一层权重形状为 (3 * H_out, H_in)，CPU 上随机初始化
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      // 非第一层权重形状为 (3 * H_out, H_out)，CPU 上随机初始化
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    // 每层的隐藏状态权重形状为 (3 * H_out, H_out)，CPU 上随机初始化
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    // 每层的输入偏置形状为 (3 * H_out)，CPU 上随机初始化
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));


继续循环创建权重和偏置的部分。
  // 将偏置权重 bias_hh_l 添加到列表末尾
  bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
}

// 在此处加入保护措施以运行推断而非训练
// 避免以下错误：
//     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
//     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
c10::InferenceMode mode;

// 执行前向传播
const auto out_cpu = at::gru(in_cpu, h0_cpu,
    { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0], weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
    has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

// 权重和偏置应始终在 CPU 上
const auto out_vulkan = at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

// 提取 CPU 输出和隐藏状态
auto cpu_output = std::get<0>(out_cpu);
auto cpu_hidden = std::get<1>(out_cpu);
// 提取 Vulkan 输出和隐藏状态
auto vulkan_output = std::get<0>(out_vulkan);
auto vulkan_hidden = std::get<1>(out_vulkan);

// 检查输出是否几乎相等
const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
if (!check_output) {
  showRtol(cpu_output, vulkan_output.cpu());
}
// 断言输出几乎相等
ASSERT_TRUE(check_output);

// 检查隐藏状态是否几乎相等
const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
if (!check_hidden) {
  showRtol(cpu_hidden, vulkan_hidden.cpu());
}
// 断言隐藏状态几乎相等
ASSERT_TRUE(check_hidden);
}

TEST_F(VulkanAPITest, gru_invalidinputs_exceptions) {
  // Arrange

  // 定义输入数据和模型参数
  const int H_in = 17;  // input_size
  const int H_out = 50; // hidden_size
  const int num_layers = 2;
  const int L = 5;
  const int N = 4;
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;

  // 创建随机的输入数据张量和初始隐藏状态张量
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  // 定义 GRU 模型的权重和偏置列表
  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)

  // 为每一层 GRU 模型生成权重和偏置张量
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // 配置推断模式以避免训练中的异常
  // 避免以下错误：
  // C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  // If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act: incorrect # of weights/biases
  EXPECT_THROW({
    // 执行：使用不正确数量的权重和偏置参数调用 GRU 模型
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // Act: non-3D input tensor
  EXPECT_THROW({
    // 执行：使用非三维输入张量调用 GRU 模型
    const auto in_cpu_2d = at::rand({1, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  // 使用 ATen 库中的 GRU 函数，执行在 Vulkan 上的计算
  at::gru(in_cpu_2d.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
}, ::std::exception);

// Act: 非3D隐藏层张量时抛出异常
EXPECT_THROW({
  // 创建一个随机的2D隐藏层张量 h0_cpu_2d
  const auto h0_cpu_2d = at::rand({num_layers, H_out}, at::device(at::kCPU).dtype(at::kFloat));
  // 使用 ATen 库中的 GRU 函数，执行在 Vulkan 上的计算
  at::gru(in_cpu.vulkan(), h0_cpu_2d.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
}, ::std::exception);

// Act: has_biases 应为 true 时抛出异常
EXPECT_THROW({
  // 使用 ATen 库中的 GRU 函数，执行在 Vulkan 上的计算，但设置 has_biases 参数为 false
  at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    false, num_layers, gru_dropout, train, bidirectional, batch_first);
}, ::std::exception);

// Act: train 应为 false 时抛出异常
EXPECT_THROW({
  // 使用 ATen 库中的 GRU 函数，执行在 Vulkan 上的计算，但设置 train 参数为 true
  at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    has_biases, num_layers, gru_dropout, true, bidirectional, batch_first);
}, ::std::exception);

// Act: bidirectional 应为 false 时抛出异常
EXPECT_THROW({
  // 使用 ATen 库中的 GRU 函数，执行在 Vulkan 上的计算，但设置 bidirectional 参数为 true
  at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    has_biases, num_layers, gru_dropout, train, true, batch_first);
}, ::std::exception);

// Act: batch_first 应为 true 时抛出异常
EXPECT_THROW({
  // 使用 ATen 库中的 GRU 函数，执行在 Vulkan 上的计算，但设置 batch_first 参数为 false
  at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    has_biases, num_layers, gru_dropout, train, bidirectional, false);
}, ::std::exception);

// Act: dropout 应为 0.0 时抛出异常
EXPECT_THROW({
  // 使用 ATen 库中的 GRU 函数，执行在 Vulkan 上的计算，但设置 gru_dropout 参数为 1.0
  at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
    weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
    has_biases, num_layers, 1.0, train, bidirectional, batch_first);
}, ::std::exception);
}

TEST_F(VulkanAPITest, gru_prepack_success) {
  // Arrange

  // 定义输入向量的维度
  const int H_in = 81;  // input_size
  // 定义隐藏状态向量的维度
  const int H_out = 10; // hidden_size
  // 定义GRU层数
  const int num_layers = 2;
  // 定义序列长度
  const int L = 1;
  // 定义批次大小
  const int N = 1;
  // 定义GRU的dropout率
  const double gru_dropout = .0;
  // 是否存在偏置项
  const bool has_biases = true;
  // 是否处于训练状态
  const bool train = false;
  // 是否为双向GRU
  const bool bidirectional = false;
  // 是否批量优先
  const bool batch_first = true;

  // 创建输入数据张量，随机初始化
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建初始隐藏状态张量，随机初始化
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  // 初始化用于存储权重和偏置的张量列表
  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)

  // 遍历每一层的权重和偏置初始化
  for (int i = 0; i < num_layers; ++i) {
    // 根据层索引选择不同形状的输入到隐藏权重张量初始化
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    // 初始化隐藏到隐藏权重张量
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    // 初始化输入到隐藏的偏置张量
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    // 初始化隐藏到隐藏的偏置张量
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    // 在此处放置保护以运行推断而不是训练
    // 避免以下错误：
    //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
    //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
    c10::InferenceMode mode;

    // 调用 ATen 的 GRU 前向计算
    const auto out_cpu = at::gru(in_cpu, h0_cpu,
        { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0], weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
        has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

    // 调用自定义 Vulkan 运算的预打包函数
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
          weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
    
    // 在 Vulkan 上运行预打包的 GRU 计算上下文
    auto out_vulkan = callOpByName(
        "vulkan_prepack::run_gru_context",
        "",
        in_cpu.vulkan(), h0_cpu.vulkan(), prepack[0]);

    // 提取 CPU 和 Vulkan 计算结果
    auto cpu_output = std::get<0>(out_cpu);
    auto cpu_hidden = std::get<1>(out_cpu);
    auto vulkan_output = out_vulkan[0].toTensor();
    auto vulkan_hidden = out_vulkan[1].toTensor();

    // 断言：检查 CPU 输出与 Vulkan 输出的近似性
    const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
    if (!check_output) {
        showRtol(cpu_output, vulkan_output.cpu());
    }
    ASSERT_TRUE(check_output);

    // 断言：检查 CPU 隐藏状态与 Vulkan 隐藏状态的近似性
    const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
    if (!check_hidden) {
        showRtol(cpu_hidden, vulkan_hidden.cpu());
    }
    ASSERT_TRUE(check_hidden);
}

TEST_F(VulkanAPITest, gru_prepack_invalidinputs_exceptions) {
  // 定义测试用例名称和参数
  // Arrange
  const int H_in = 70;  // input_size
  const int H_out = 2; // hidden_size
  const int num_layers = 2;
  const int L = 3;
  const int N = 5;
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  // 创建随机的输入和初始隐藏状态张量
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  // 定义权重和偏置的列表
  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)
  // 填充权重和偏置的列表
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // 运行推断模式以避免训练时的异常
  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // 执行异常测试：不正确的权重/偏置数量
  // Act: incorrect # of weights/biases
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
            weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // 执行异常测试：非3D输入张量
  // Act: non-3D input tensor
  EXPECT_THROW({
    const auto in_cpu_2d = at::rand({1, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  // 使用 `callOpByName` 函数调用 Vulkan 前向计算函数 `vulkan_prepack::create_gru_context`，传入权重和偏置张量列表
  auto prepack = callOpByName(
      "vulkan_prepack::create_gru_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
          weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

  // 使用 `callOpByName` 函数调用 Vulkan 执行 GRU 计算函数 `vulkan_prepack::run_gru_context`，传入输入张量和初始化隐藏状态
  auto out_vulkan = callOpByName(
      "vulkan_prepack::run_gru_context",
      "",
      in_cpu_2d.vulkan(), h0_cpu.vulkan(), prepack[0]);
}, ::std::exception);

// 期望抛出异常：隐藏状态张量维度不是 3D
EXPECT_THROW({
  // 生成一个随机张量 `h0_cpu_2d`，形状为 (num_layers, H_out)，在 CPU 上
  const auto h0_cpu_2d = at::rand({num_layers, H_out}, at::device(at::kCPU).dtype(at::kFloat));
  // 使用 `callOpByName` 函数调用 Vulkan 前向计算函数 `vulkan_prepack::create_gru_context`，传入权重和偏置张量列表
  auto prepack = callOpByName(
      "vulkan_prepack::create_gru_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
          weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  // 使用 `callOpByName` 函数调用 Vulkan 执行 GRU 计算函数 `vulkan_prepack::run_gru_context`，传入输入张量和初始化隐藏状态
  auto out_vulkan = callOpByName(
      "vulkan_prepack::run_gru_context",
      "",
      in_cpu.vulkan(), h0_cpu_2d.vulkan(), prepack[0]);
}, ::std::exception);

// 期望抛出异常：`has_biases` 参数应为 true
EXPECT_THROW({
  // 使用 `callOpByName` 函数调用 Vulkan 前向计算函数 `vulkan_prepack::create_gru_context`，传入权重和偏置张量列表
  auto prepack = callOpByName(
      "vulkan_prepack::create_gru_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
         weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      false, num_layers, gru_dropout, train, bidirectional, batch_first);
}, ::std::exception);

// 期望抛出异常：`train` 参数应为 false
EXPECT_THROW({
  // 使用 `callOpByName` 函数调用 Vulkan 前向计算函数 `vulkan_prepack::create_gru_context`，传入权重和偏置张量列表
  auto prepack = callOpByName(
      "vulkan_prepack::create_gru_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
         weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, gru_dropout, true, bidirectional, batch_first);
}, ::std::exception);

// 期望抛出异常：`bidirectional` 参数应为 false
EXPECT_THROW({
   // 使用 `callOpByName` 函数调用 Vulkan 前向计算函数 `vulkan_prepack::create_gru_context`，传入权重和偏置张量列表
   auto prepack = callOpByName(
      "vulkan_prepack::create_gru_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
         weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, gru_dropout, train, true, batch_first);
}, ::std::exception);

// 期望抛出异常：`batch_first` 参数应为 true
EXPECT_THROW({
  // 使用 `callOpByName` 函数调用 Vulkan 前向计算函数 `vulkan_prepack::create_gru_context`，传入权重和偏置张量列表
  auto prepack = callOpByName(
      "vulkan_prepack::create_gru_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
         weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, gru_dropout, train, bidirectional, false);
    // 调用指定名称的函数 "vulkan_prepack::run_gru_context"，传入参数 in_cpu.vulkan(), h0_cpu.vulkan(), prepack[0]，并将结果存储在 out_vulkan 中
    auto out_vulkan = callOpByName(
        "vulkan_prepack::run_gru_context",
        "",
        in_cpu.vulkan(), h0_cpu.vulkan(), prepack[0]);
  }, ::std::exception);

  // 执行测试：预期 dropout 应为 0.0
  EXPECT_THROW({
    // 调用指定名称的函数 "vulkan_prepack::create_gru_context"，传入多个参数构成的向量，包括权重、偏置等，以及其他控制参数
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, 1.0, train, bidirectional, batch_first);
  }, ::std::exception);
  // 定义一个 LSTM 层的输入大小
  const int input_size = 5;
  // 定义 LSTM 层的隐藏状态大小
  const int hidden_size = 7;
  // 定义 LSTM 层数量
  const int num_layers = 4;
  // 定义 LSTM 时间步数
  const int L = 1;
  // 定义 LSTM 输入序列数量
  const int N = 1;
  // 定义 LSTM 的 dropout 概率
  const double lstm_dropout = .0;
  // 是否包含偏置项
  const bool has_biases = true;
  // 是否处于训练模式
  const bool train = false;
  // 是否为双向 LSTM
  const bool bidirectional = false;
  // 是否批量优先模式
  const bool batch_first = true;

  // 生成一个指定形状的随机张量作为 CPU 上的输入数据
  const auto in_cpu = at::rand({N, L, input_size}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成一个指定形状的随机张量作为 CPU 上的初始隐藏状态
  const auto h0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));
  // 生成一个指定形状的随机张量作为 CPU 上的初始记忆状态
  const auto c0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));

  // 初始化一个空列表，用于存储每层 LSTM 的输入到隐藏状态的权重张量
  c10::List<at::Tensor> weight_ih_l; // shape (4 * hidden_size, input_size)
  // 初始化一个空列表，用于存储每层 LSTM 的隐藏到隐藏状态的权重张量
  c10::List<at::Tensor> weight_hh_l; // shape (4 * hidden_size, hidden_size)
  // 初始化一个空列表，用于存储每层 LSTM 的输入到隐藏状态的偏置张量
  c10::List<at::Tensor> bias_ih_l;   // shape (4 * hidden_size)
  // 初始化一个空列表，用于存储每层 LSTM 的隐藏到隐藏状态的偏置张量
  c10::List<at::Tensor> bias_hh_l;   // shape (4 * hidden_size)

  // 循环遍历每一层 LSTM
  for (int l = 0; l < num_layers; ++l) {
    // 如果是第一层，随机生成指定形状的权重张量，添加到输入到隐藏状态的权重列表中
    if (l == 0) {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, input_size}, at::device(at::kCPU).dtype(at::kFloat)));
      // 添加到隐藏到隐藏状态的权重列表中
      weight_hh_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
      // 如果包含偏置项，生成指定形状的偏置张量，添加到输入到隐藏状态的偏置列表中
      if (has_biases) {
        bias_ih_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
        // 添加到隐藏到隐藏状态的偏置列表中
        bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
      }
    }
  }
  } else {
    // 如果不是第一层，添加权重矩阵到 weight_ih_l 向量
    weight_ih_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  }
  // 添加权重矩阵到 weight_hh_l 向量
  weight_hh_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  // 添加偏置向量到 bias_ih_l 向量
  bias_ih_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  // 添加偏置向量到 bias_hh_l 向量
  bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
}

// 添加此守卫以便运行推断而非训练
// 避免以下错误：
//     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
//     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
c10::InferenceMode mode;

// 执行 LSTM 操作
const auto out_cpu = at::lstm(in_cpu, {h0_cpu, c0_cpu},
    { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
      weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1],
      weight_ih_l[2], weight_hh_l[2], bias_ih_l[2], bias_hh_l[2],
      weight_ih_l[3], weight_hh_l[3], bias_ih_l[3], bias_hh_l[3] },
    has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

// 将权重和偏置始终保留在 CPU 上
const auto out_vulkan = at::lstm(in_cpu.vulkan(), {h0_cpu.vulkan(), c0_cpu.vulkan()},
    { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1),
      weight_ih_l.get(2), weight_hh_l.get(2), bias_ih_l.get(2), bias_hh_l.get(2),
      weight_ih_l.get(3), weight_hh_l.get(3), bias_ih_l.get(3), bias_hh_l.get(3) },
    has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

// 提取 CPU 输出和隐藏状态
auto cpu_output = std::get<0>(out_cpu);
auto cpu_hidden = std::get<1>(out_cpu);
auto cpu_cell = std::get<2>(out_cpu);

// 提取 Vulkan 输出和隐藏状态
auto vulkan_output = std::get<0>(out_vulkan);
auto vulkan_hidden = std::get<1>(out_vulkan);
auto vulkan_cell = std::get<2>(out_vulkan);

// 断言 CPU 和 Vulkan 输出近似相等
const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
if (!check_output) {
  showRtol(cpu_output, vulkan_output.cpu());
}
ASSERT_TRUE(check_output);

// 断言 CPU 和 Vulkan 隐藏状态近似相等
const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());

显示两个变量 `cpu_hidden` 和 `vulkan_hidden` 的 CPU 数据之间的相对误差。


  }
  ASSERT_TRUE(check_hidden);

结束 `if` 语句块，并断言 `check_hidden` 应为真。


  const auto check_cell = almostEqual(cpu_cell, vulkan_cell.cpu());

计算 `cpu_cell` 和 `vulkan_cell` 的 CPU 数据是否几乎相等，将结果存储在 `check_cell` 中。


  if (!check_cell) {
    showRtol(cpu_cell, vulkan_cell.cpu());
  }
  ASSERT_TRUE(check_cell);

如果 `check_cell` 为假，则显示 `cpu_cell` 和 `vulkan_cell` 的 CPU 数据之间的相对误差，并断言 `check_cell` 应为真。
TEST_F(VulkanAPITest, lstm_mclareninputs_success) {
  // 定义测试用例函数，测试 LSTM 模型处理 McLaren 输入的情况

  // 定义 LSTM 模型的参数
  const int input_size = 384;          // 输入数据的特征维度
  const int hidden_size = 384;         // LSTM 隐状态的维度
  const int num_layers = 2;            // LSTM 的层数
  const int L = 1;                     // 输入序列的长度
  const int N = 1;                     // 批量大小
  const double lstm_dropout = .0;      // LSTM 层的 dropout 概率
  const bool has_biases = true;        // 是否使用偏置
  const bool train = false;            // 是否处于训练模式
  const bool bidirectional = false;    // 是否使用双向 LSTM
  const bool batch_first = true;       // 输入张量是否按照 (batch, seq, feature) 排列

  // 创建 CPU 上的输入张量及初始隐藏状态张量
  const auto in_cpu = at::rand({N, L, input_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto c0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));

  // 初始化 LSTM 模型的权重和偏置列表
  c10::List<at::Tensor> weight_ih_l;   // 输入到隐藏状态的权重列表，形状为 (4 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l;   // 隐藏状态到隐藏状态的权重列表，形状为 (4 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;     // 输入到隐藏状态的偏置列表，形状为 (4 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;     // 隐藏状态到隐藏状态的偏置列表，形状为 (4 * hidden_size)

  // 循环遍历每一层 LSTM，初始化权重和偏置
  for (int l = 0; l < num_layers; ++l) {
    if (l == 0) {
      // 第一层的输入到隐藏状态的权重初始化为随机张量
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, input_size}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      // 后续层的输入到隐藏状态的权重初始化为随机张量
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    // 隐藏状态到隐藏状态的权重初始化为随机张量
    weight_hh_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    // 输入到隐藏状态的偏置初始化为随机张量
    bias_ih_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    // 隐藏状态到隐藏状态的偏置初始化为随机张量
    bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));

// 将一个包含随机数的张量添加到 bias_hh_l 向量的末尾，张量大小为 4 * hidden_size，数据类型为 float，存储在 CPU 上。

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

// 设置推理模式保护，用于在推理过程中而非训练过程中运行，以避免以下错误：
//     异常信息："0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31，请向 PyTorch 报告一个 bug。aten::gru.input 在 CompositeImplicitAutograd 和映射到 AutogradOther 的后端上都注册了内核。这使得后端内核无法访问；调度程序将始终优先选择 CompositeImplicitAutograd 降级（参见注释 [Ambiguity in AutogradOther kernel]）。如果要覆盖 CompositeImplicitAutograd，请提出问题请求后端的专用 Autograd 调度密钥。
//     如果只想运行推理而不是训练，请在 model.forward() 之前添加 `c10::InferenceMode mode;`。请注意，此保护仅在 C++ 中有效，目前不适用于 Python。

  // Act
  const auto out_cpu = at::lstm(in_cpu, {h0_cpu, c0_cpu},
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
        weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

// 在 CPU 上执行 LSTM 操作，使用输入 in_cpu 和初始隐藏状态 h0_cpu、c0_cpu，以及一组权重和偏置参数。这些参数包括：两个输入门权重（weight_ih_l[0]、weight_ih_l[1]）、两个隐藏状态权重（weight_hh_l[0]、weight_hh_l[1]）、两个输入门偏置（bias_ih_l[0]、bias_ih_l[1]）和两个隐藏状态偏置（bias_hh_l[0]、bias_hh_l[1]）。

  // weights/biases should be always on CPU.
  const auto out_vulkan = at::lstm(in_cpu.vulkan(), {h0_cpu.vulkan(), c0_cpu.vulkan()},
      { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
        weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

// 在 Vulkan 后端执行 LSTM 操作，使用 Vulkan 格式的输入 in_cpu.vulkan() 和对应的初始化隐藏状态 h0_cpu.vulkan()、c0_cpu.vulkan()，以及一组权重和偏置参数。这些参数也应当保持在 CPU 上处理，即使在 Vulkan 后端执行。

  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto cpu_cell = std::get<2>(out_cpu);
  auto vulkan_output = std::get<0>(out_vulkan);
  auto vulkan_hidden = std::get<1>(out_vulkan);
  auto vulkan_cell = std::get<2>(out_vulkan);

// 提取 CPU 操作输出的 LSTM 结果，包括输出值、隐藏状态和细胞状态，分别存储在 cpu_output、cpu_hidden 和 cpu_cell 变量中。
// 提取 Vulkan 后端操作输出的 LSTM 结果，包括输出值、隐藏状态和细胞状态，分别存储在 vulkan_output、vulkan_hidden 和 vulkan_cell 变量中。

  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

// 检查 CPU 和 Vulkan 后端的 LSTM 输出值是否几乎相等，使用 almostEqual 函数进行比较。如果不相等，则显示它们的相对误差，并断言检查结果为真。

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);

// 检查 CPU 和 Vulkan 后端的 LSTM 隐藏状态是否几乎相等，使用 almostEqual 函数进行比较。如果不相等，则显示它们的相对误差，并断言检查结果为真。

  const auto check_cell = almostEqual(cpu_cell, vulkan_cell.cpu());
  if (!check_cell) {
    showRtol(cpu_cell, vulkan_cell.cpu());
  }
  ASSERT_TRUE(check_cell);

// 检查 CPU 和 Vulkan 后端的 LSTM 细胞状态是否几乎相等，使用 almostEqual 函数进行比较。如果不相等，则显示它们的相对误差，并断言检查结果为真。
}

TEST_F(VulkanAPITest, lstm_prepack_success) {
  // 在 Vulkan API 测试框架中，定义名为 lstm_prepack_success 的测试用例
  // Arrange 部分，准备测试数据和参数
  const int input_size = 81;  // 输入大小
  const int hidden_size = 10;  // 隐藏层大小
  const int num_layers = 2;  // LSTM 层的数量
  const int L = 1;
  const int N = 1;
  const double lstm_dropout = .0;  // LSTM dropout 设置
  const bool has_biases = true;  // 是否包含偏置
  const bool train = false;  // 是否训练模式
  const bool bidirectional = false;  // 是否双向
  const bool batch_first = true;  // 是否批次优先
  // 创建在 CPU 上的随机输入数据张量
  const auto in_cpu = at::rand({N, L, input_size}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建在 CPU 上的随机初始隐藏状态张量
  const auto h0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));
  // 创建在 CPU 上的随机初始细胞状态张量
  const auto c0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // 用于存储输入到隐藏层权重的列表
  // shape (4 * hidden_size, l == 0 ? input_size : hidden_size)
  c10::List<at::Tensor> weight_hh_l; // 用于存储隐藏层到隐藏层权重的列表
  // shape (4 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // 用于存储输入到隐藏层偏置的列表
  // shape (4 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // 用于存储隐藏层到隐藏层偏置的列表
  // 循环遍历每一层的 LSTM
  for (int l = 0; l < num_layers; ++l) {
    // 根据当前层的索引 l 选择不同的权重张量形状
    if (l == 0) {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, input_size}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    // 随机生成隐藏层到隐藏层的权重张量
    weight_hh_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    // 随机生成输入到隐藏层的偏置张量
    bias_ih_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    // 随机生成隐藏层到隐藏层的偏置张量
    bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // 将此守卫放置在此处以运行推断而不是训练
  // 以避免以下错误：
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // 执行 LSTM 前向传播
  const auto out_cpu = at::lstm(in_cpu, {h0_cpu, c0_cpu},
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
        weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  // 调用 Vulkan 前置打包操作
  auto prepack = callOpByName(
      "vulkan_prepack::create_lstm_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
                                weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  // 调用 Vulkan 执行 LSTM 计算
  auto out_vulkan = callOpByName(
      "vulkan_prepack::run_lstm_context",
      "",
      in_cpu.vulkan(), h0_cpu.vulkan(), c0_cpu.vulkan(), prepack[0]);

  // 提取 CPU 和 Vulkan 的输出
  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto cpu_cell = std::get<2>(out_cpu);
  auto vulkan_output = out_vulkan[0].toTensor();
  auto vulkan_hidden = out_vulkan[1].toTensor();
  auto vulkan_cell = out_vulkan[2].toTensor();

  // 断言 CPU 和 Vulkan 的输出是否几乎相等
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  // 断言 CPU 和 Vulkan 的隐藏状态是否几乎相等
  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);

  // 断言 CPU 和 Vulkan 的单元状态是否几乎相等
  const auto check_cell = almostEqual(cpu_cell, vulkan_cell.cpu());
  if (!check_cell) {
    showRtol(cpu_cell, vulkan_cell.cpu());
  }
  ASSERT_TRUE(check_cell);
}

TEST_F(VulkanAPITest, querypool_flushed_shader_log) {
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  // 获取初始时操作性能分析是否启用，并保存状态
  const bool op_profiling_enabled_initially =
      at::native::vulkan::api::context()->op_profiling_enabled();

  // 启用操作性能分析
  at::native::vulkan::api::context()->enable_op_profiling();

  // 创建大小为 [11, 7, 139, 109] 的随机张量，使用 Vulkan 后端
  const at::Tensor a_add_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor a_add_vulkan = a_add_cpu.vulkan();

  // 创建大小为 [11, 7, 139, 109] 的随机张量，使用 Vulkan 后端
  const at::Tensor b_add_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor b_add_vulkan = b_add_cpu.vulkan();

  // 使用 Vulkan 执行张量加法
  at::add(a_add_vulkan, b_add_vulkan, 2.1f).cpu();

  // 提取 Vulkan 查询池中的结果
  at::native::vulkan::api::context()->querypool().extract_results();
  // 重置 Vulkan 查询池
  at::native::vulkan::api::context()->reset_querypool();

  // 创建大小为 [11, 7, 139, 109] 的随机张量，使用 Vulkan 后端
  const at::Tensor a_sub_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor a_sub_vulkan = a_sub_cpu.vulkan();

  // 创建大小为 [11, 7, 139, 109] 的随机张量，使用 Vulkan 后端
  const at::Tensor b_sub_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor b_sub_vulkan = b_sub_cpu.vulkan();

  // 使用 Vulkan 执行张量减法
  at::sub(a_sub_vulkan, b_sub_vulkan, 2.1f).cpu();

  // 提取 Vulkan 查询池中的结果
  at::native::vulkan::api::context()->querypool().extract_results();
  // 重置 Vulkan 查询池
  at::native::vulkan::api::context()->reset_querypool();

  // 创建大小为 [11, 7, 139, 109] 的随机张量，使用 Vulkan 后端
  const at::Tensor a_mul_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor a_mul_vulkan = a_mul_cpu.vulkan();

  // 创建大小为 [11, 7, 139, 109] 的随机张量，使用 Vulkan 后端
  const at::Tensor b_mul_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor b_mul_vulkan = b_mul_cpu.vulkan();

  // 使用 Vulkan 执行张量乘法
  at::mul(a_mul_vulkan, b_mul_vulkan).cpu();

  /*
    The most recent shaders should be
    (-12) vulkan.nchw_to_image
    (-11) vulkan.nchw_to_image
    (-10) vulkan.add
    (-9)  vulkan.image_to_nchw

    (-8)  vulkan.nchw_to_image
    (-7)  vulkan.nchw_to_image
    (-6)  vulkan.sub
    (-5)  vulkan.image_to_nchw

    (-4)  vulkan.nchw_to_image
    (-3)  vulkan.nchw_to_image
    (-2)  vulkan.mul
    (-1)  vulkan.image_to_nchw
  */

  // 获取查询池中记录的条目数
  const size_t entry_count =
      at::native::vulkan::api::context()->querypool().shader_logs_entry_count();

  // 获取指定索引处的着色器名称和执行时间（减去10）
  std::tuple<std::string, uint64_t> add_shader_details =
      at::native::vulkan::api::context()
          ->querypool()
          .get_shader_name_and_execution_duration_ns(entry_count - 10);
  // 获取指定索引处的着色器名称和执行时间（减去6）
  std::tuple<std::string, uint64_t> sub_shader_details =
      at::native::vulkan::api::context()
          ->querypool()
          .get_shader_name_and_execution_duration_ns(entry_count - 6);
  // 获取指定索引处的着色器名称和执行时间（减去2）
  std::tuple<std::string, uint64_t> mul_shader_details =
      at::native::vulkan::api::context()
          ->querypool()
          .get_shader_name_and_execution_duration_ns(entry_count - 2);

  // 检查获取的着色器名称是否符合预期
  EXPECT_EQ(std::get<0>(add_shader_details), "vulkan.add");
  EXPECT_EQ(std::get<0>(sub_shader_details), "vulkan.sub");
  EXPECT_EQ(std::get<0>(mul_shader_details), "vulkan.mul");

  // 如果初始时操作性能分析未启用，则进行相应操作
  if (!op_profiling_enabled_initially) {
    # 调用 Vulkan API 的上下文对象，并重置查询池
    at::native::vulkan::api::context()->reset_querypool();
    # 调用 Vulkan API 的上下文对象，并禁用操作性能分析
    at::native::vulkan::api::context()->disable_op_profiling();
#else
  // 如果未定义 USE_VULKAN_API，则执行以下代码块
  GTEST_SKIP() << "QueryPool is not available";
#endif
}

} // namespace

#endif /* USE_VULKAN_API */
```