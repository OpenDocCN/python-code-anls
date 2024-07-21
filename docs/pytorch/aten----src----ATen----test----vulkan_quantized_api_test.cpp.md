# `.\pytorch\aten\src\ATen\test\vulkan_quantized_api_test.cpp`

```
#ifdef USE_VULKAN_API
// 引入相关的头文件
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convert.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Factory.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <random>

#include <cstdio>

// 使用 at::native::vulkan::api::utils 命名空间简化代码
using namespace at::native::vulkan::api::utils;

/*
 * TODO: 将此文件重命名为 vulkan_experimental_test 并移到 caffe2/fb/vulkan 目录下。
 * 该文件用于测试 Vulkan 后端的实验性特性。
 * vulkan_api_test 不能胜任此任务，因为它无法链接到 ATen/native/vulkan 文件夹中的符号。
 */
namespace {

// 使用 at::native::vulkan 命名空间简化代码
using namespace at::native::vulkan;

#ifdef USE_VULKAN_FP16_INFERENCE
// 如果使用 Vulkan FP16 推理，设置容差为 1e-2
constexpr float kTolerance = 1e-2;
#else
// 否则设置容差为 1e-5
constexpr float kTolerance = 1e-5;
#endif

// 检查张量 diff 是否在容差范围内
bool checkRtol(
    const at::Tensor& diff,
    const std::vector<at::Tensor>& inputs,
    const float tolerated_error = 0) {
  // 计算输入张量中的最大绝对值
  double maxValue = 0.0;
  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<double>(), maxValue);
  }

#ifdef USE_VULKAN_FP16_INFERENCE
  // 如果使用 Vulkan FP16 推理，设置容差为 1e-2
  constexpr float tolerance = 1e-2;
#else
  // 否则设置容差为 1e-5
  constexpr float tolerance = 1e-5;
#endif

  // 返回是否 diff 的最大绝对值在容差范围内
  return diff.abs().max().item<double>() <= (tolerance * maxValue + tolerated_error);
}

// 检查两个张量 a 和 b 是否几乎相等
bool almostEqual(
    const at::Tensor& a,
    const at::Tensor& b,
    const float tolerated_error = 0) {
  // 调用 checkRtol 检查 a 和 b 的容差
  return checkRtol(a - b, {a, b}, tolerated_error);
}

/* Unused function
bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.0f;
}
*/

// 显示两个张量 a 和 b 之间的容差
void showRtol(
    const at::Tensor& a,
    const at::Tensor& b,
    long* xpos = nullptr,
    long* ypos = nullptr) {
  // 计算 a 和 b 之间的绝对差异
  const auto diff = (a - b).abs();

  // 计算 a 和 b 的绝对值的最大值
  double maxValue = a.abs().max().item<double>();
  maxValue = fmax(b.abs().max().item<double>(), maxValue);

#ifdef USE_VULKAN_FP16_INFERENCE
  // 如果使用 Vulkan FP16 推理，设置容差为 1e-2
  constexpr float tolerance = 1e-2;
#else
  // 否则设置容差为 1e-5
  constexpr float tolerance = 1e-5;
#endif

  // 计算最大容差
  const double maxDiff = maxValue * tolerance;
  std::cout << "Max Diff allowed: " << maxDiff << std::endl;
  std::cout << "Max Diff found is: " << diff.max().item<double>() << std::endl;

  // 如果 diff 是二维张量，则打印 xpos 和 ypos
  if (diff.sizes().size() == 2) {
    // 对于 diff 张量的第一维度进行迭代，y 是索引
    for (const auto y : c10::irange(diff.sizes()[0])) {
      // 输出当前 y 的索引值，用于调试和展示
      std::cout << y << ":";
      // 对于 diff 张量的第二维度进行迭代，x 是索引
      for (const auto x : c10::irange(diff.sizes()[1])) {
        // 获取 diff 张量在 (y, x) 处的值，并转换为 double 类型
        double diff_xy = diff[y][x].item<double>();
        // 如果 diff_xy 大于 maxDiff，则输出 x 的值
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
          // 如果 diff 张量的最大值等于 diff_xy，则输出 diff_xy 的值
          if (diff.max().item<double>() == diff_xy) {
            std::cout << " : " << diff_xy;
            // 如果传入了 xpos 和 ypos 的指针，则将 x 和 y 的值写入并返回
            if (xpos && ypos) {
              *xpos = x;
              *ypos = y;
              return;
            }
          }
        } else {
          // 如果 diff_xy 不大于 maxDiff，则输出空白符
          std::cout << std::setw(5) << " ";
        }
      }
      // 每行输出结束后换行
      std::cout << std::endl;
    }
  }
namespace {

// 生成一个包含输入参数的堆栈的函数模板
template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

// 通过操作符句柄调用操作的函数模板
template <class... Args>
inline std::vector<c10::IValue> callOpByHandle(
    const c10::OperatorHandle& op,
    Args... args) {
  // 使用输入参数构造一个堆栈
  auto stack = makeStack(std::forward<Args>(args)...);
  // 调用分发器的函数来执行具体的操作
  c10::Dispatcher::singleton().callBoxed(op, &stack);
  // 返回堆栈
  return stack;
}

// 通过操作符名称和重载名称调用操作的函数模板
template <class... Args>
inline std::vector<c10::IValue> callOpByName(
    const char* func_name,
    const char* overload_name,
    Args... args) {
  // 查找指定函数名和重载名对应的操作符句柄
  const std::optional<c10::OperatorHandle> op_handle =
      c10::Dispatcher::singleton().findSchema({func_name, overload_name});
  // 断言操作符句柄的可用性
  assert(op_handle.has_value());
  // 通过操作符句柄调用操作函数，并传递参数
  return callOpByHandle(op_handle.value(), std::forward<Args>(args)...);
}

// 使用 Vulkan 命名空间和相关工具类别名
using namespace at::native::vulkan;
using at::native::vulkan::api::utils::ivec3;
using at::native::vulkan::api::utils::ivec4;
using at::native::vulkan::api::utils::vec4;

// 自定义输出操作符重载，用于输出 vec4 类型对象
std::ostream& operator<<(std::ostream& os, const vec4& v) {
  // 格式化输出 vec4 对象的数据内容
  os << "(" << v.data[0u] << ", " << v.data[1u] << ", " << v.data[2u] << ", "
     << v.data[3u] << ")";
  return os;
}

// 自定义输出操作符重载，用于输出 ivec3 类型对象
std::ostream& operator<<(std::ostream& os, const ivec3& v) {
  // 格式化输出 ivec3 对象的数据内容
  os << "(" << v.data[0u] << ", " << v.data[1u] << ", " << v.data[2u] << ")";
  return os;
}

// 自定义输出操作符重载，用于输出 ivec4 类型对象
std::ostream& operator<<(std::ostream& os, const ivec4& v) {
  // 格式化输出 ivec4 对象的数据内容
  os << "(" << v.data[0u] << ", " << v.data[1u] << ", " << v.data[2u] << ", "
     << v.data[3u] << ")";
  return os;
}

} // namespace

// 匿名命名空间内的函数定义

// 生成指定范围内的随机双精度浮点数
double rand_double(const double min, const double max) {
  // 使用随机设备和 Mersenne Twister 生成器创建随机数引擎
  std::random_device rd;
  std::mt19937 gen(rd());
  // 如果最大值和最小值之间的差小于浮点数的最小精度，则直接返回最小值
  if (std::fabs(max - min) < std::numeric_limits<double>::epsilon()) {
    return min;
  }
  // 使用均匀分布生成指定范围内的随机双精度浮点数
  return std::uniform_real_distribution<double>(min, max)(gen);
}

// 生成指定范围内的随机整数
int rand_int(const int min, const int max) {
  // 使用随机设备和 Mersenne Twister 生成器创建随机数引擎
  std::random_device rd;
  std::mt19937 gen(rd());
  // 使用均匀分布生成指定范围内的随机整数
  return std::uniform_int_distribution<int>(min, max)(gen);
}

// 生成指定范围内的随机正整数
int rand_pos_int(const int max_val) {
  // 断言最大值必须大于零
  TORCH_CHECK(max_val > 0, "max value must be positive");
  // 返回在 [1, max_val] 范围内的随机整数
  return 1 + rand_int(0, max_val);
}

// 生成随机张量
at::Tensor produce_random_tensor(
    const at::IntArrayRef tensor_shape,
    const double s_min = 1.0,
    const double s_max = 100.0,
    const double shift = 0.45) {
  // 生成一个随机张量，其值位于 [-shift * s, (1-shift) * s) 的范围内
  // 其中 s 是在 [s_min, s_max] 范围内随机生成的双精度浮点数
  TORCH_CHECK(s_min > 0, "scalar lower bound must be positive");
  TORCH_CHECK(s_min <= s_max, "scalar lower bound must be <= upper bound");
  const auto scalar = rand_double(s_min, s_max);
  return scalar *
      (at::rand(tensor_shape, at::device(at::kCPU).dtype(at::kFloat)) - shift);
}

// 生成随机比例因子
double produce_random_scale(
    const double scale_min = 0.001,
    // 定义一个常量 scale_max，表示缩放的最大值为 2.0
    const double scale_max = 2.0);
    
    // 使用 TORCH_CHECK 进行断言检查，确保 scale_min 小于等于 scale_max，否则输出错误信息
    TORCH_CHECK(scale_min <= scale_max, "scale min must be <= scale max");
    
    // 返回一个随机生成的双精度浮点数，范围为 [scale_min, scale_max)
    return rand_double(scale_min, scale_max);
    ;
} // 结束函数 produce_random_zero_point

// 生成随机的零点值，根据给定的数据类型 dtype
int produce_random_zero_point(const c10::ScalarType dtype) {
  int zero_point = 0;
  switch (dtype) {
    case c10::ScalarType::QUInt8:
      zero_point = rand_int(0, 255);  // 在 [0, 255] 范围内生成随机数作为零点值
      break;
    case c10::ScalarType::QInt8:
      zero_point = rand_int(-128, 127);  // 在 [-128, 127] 范围内生成随机数作为零点值
      break;
    case c10::ScalarType::QInt32:
      zero_point = rand_int(-100000, 100000);  // 在 [-100000, 100000] 范围内生成随机数作为零点值
      break;
    default:
      TORCH_CHECK(
          false,
          "Vulkan quantization currently not supported for dtype ",
          dtype);  // 若未支持的 dtype，抛出错误信息
  }
  return zero_point;  // 返回生成的零点值
}

// 计算量化参数的函数，返回量化的缩放因子和零点值
std::tuple<double, int> compute_quant_params(
    const at::Tensor& tensor,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8) {
  int zero_point_min = 0;
  int zero_point_max = 255;
  if (dtype == c10::ScalarType::QUInt8) {
    zero_point_min = 0;  // 设置量化的最小零点值为 0
    zero_point_max = 255;  // 设置量化的最大零点值为 255
  } else if (dtype == c10::ScalarType::QInt8) {
    zero_point_min = -128;  // 设置量化的最小零点值为 -128
    zero_point_max = 127;  // 设置量化的最大零点值为 127
  } else {
    TORCH_CHECK(
        false,
        "Computation of quant params only available for dtypes",
        "QUInt8 and QInt8");  // 若 dtype 不是支持的类型，抛出错误信息
  }
  const auto tensor_max = tensor.max().item<double>();  // 计算张量的最大值
  const auto tensor_min = tensor.min().item<double>();  // 计算张量的最小值
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/safe_downcast<float>(tensor_min),
      /*max=*/safe_downcast<float>(tensor_max),
      /*qmin=*/zero_point_min,
      /*qmax=*/zero_point_max,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/false);  // 根据张量的最小值、最大值及零点值范围选择量化参数
  return std::tuple<double, int>(q_params.scale, q_params.zero_point);  // 返回计算得到的缩放因子和零点值
}

} // namespace

namespace {

// VulkanAPI 测试类，继承自 Google Test 的测试基类
class VulkanAPITest : public ::testing::Test {
 public:
  void SetUp() override {
    if (!at::is_vulkan_available()) {
      GTEST_SKIP() << "Vulkan is not available";  // 若 Vulkan 不可用，则跳过测试
    }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    at::native::vulkan::api::context()->reset_querypool();  // 重置 Vulkan 的查询池
#endif
  }

  void TearDown() override {
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    try {
      at::native::vulkan::api::context()->querypool().extract_results();  // 提取 Vulkan 查询池结果
      at::native::vulkan::api::context()->querypool().print_results();  // 打印 Vulkan 查询池结果
    } catch (const std::exception& e) {
      std::cout << "Could not get querypool results!"
                << " Reason: " << e.what() << std::endl;  // 捕获异常，输出无法获取查询池结果的原因
    }
#endif
  }
};

// 将 CPU 上的张量转换到 Vulkan 上的张量
at::Tensor cpu_to_vulkan(at::Tensor in_cpu) {
  auto options = in_cpu.options();
  if (options.dtype().toScalarType() == c10::ScalarType::QUInt8 ||
      options.dtype().toScalarType() == c10::ScalarType::QInt8 ||
      options.dtype().toScalarType() == c10::ScalarType::QInt32) {
    auto ret = at::native::vulkan::ops::_empty_affine_quantized(
        in_cpu.sizes(),
        options.dtype().toScalarType(),
        options.layout(),
        options.device(),
        options.pinned_memory(),
        in_cpu.q_scale(),
        in_cpu.q_zero_point(),
        c10::MemoryFormat::Contiguous);  // 使用 Vulkan 的函数创建仿射量化的张量
    at::native::vulkan::ops::copy_(ret, in_cpu);  // 将 CPU 上的张量数据复制到 Vulkan 上的张量中
    return ret;  // 返回 Vulkan 上的张量
  } else {
    // 创建一个新的 Tensor 对象，其形状与 in_cpu 相同，但是未初始化
    auto ret = at::empty(in_cpu.sizes(), options);
    // 使用 Vulkan 后端的原生操作，将 in_cpu 的数据复制到 ret 中
    at::native::vulkan::ops::copy_(ret, in_cpu);
    // 返回复制后的 Tensor 对象 ret
    return ret;
}
}

// 将 Vulkan 张量转换为 CPU 张量
at::Tensor vulkan_to_cpu(at::Tensor vulkan, at::Tensor in_cpu) {
  // 获取输入 CPU 张量的选项信息
  auto q_options = in_cpu.options();
  // 如果数据类型是量化类型（QUInt8、QInt8 或 QInt32）
  if (q_options.dtype().toScalarType() == c10::ScalarType::QUInt8 ||
      q_options.dtype().toScalarType() == c10::ScalarType::QInt8 ||
      q_options.dtype().toScalarType() == c10::ScalarType::QInt32) {
    // 创建一个新的仿射量化空张量，与输入 CPU 张量相同的大小和量化参数
    auto output = at::native::empty_affine_quantized(
        in_cpu.sizes(),
        q_options.dtype().toScalarType(),
        q_options.layout(),
        q_options.device(),
        q_options.pinned_memory(),
        in_cpu.q_scale(),
        in_cpu.q_zero_point());
    // 使用 Vulkan 操作复制数据到新创建的仿射量化张量
    at::native::vulkan::ops::copy_(output, vulkan);
    // 返回复制后的仿射量化张量
    return output;
  } else {
    // 创建一个新的 CPU 张量，与输入 CPU 张量相同的大小和选项
    auto output = at::empty(in_cpu.sizes(), q_options);
    // 使用 Vulkan 操作复制数据到新创建的 CPU 张量
    at::native::vulkan::ops::copy_(output, vulkan);
    // 返回复制后的 CPU 张量
    return output;
  }
}

// VulkanAPI 测试类中的统一缓冲区复制测试
TEST_F(VulkanAPITest, uniform_buffer_copy) {
  using namespace at::native::vulkan;

  // 定义一个测试结构体
  struct TestStruct {
    int a;
    int b;
    int c;
  };

  // 初始化测试结构体实例
  TestStruct test_struct{4, 9, 10};

  // 使用测试结构体初始化 Vulkan 统一参数缓冲区
  api::UniformParamsBuffer params(api::context(), test_struct);
  // 将原始参数缓冲区复制到另一个实例
  api::UniformParamsBuffer params_copy = params;

  // 创建一个映射，用于访问复制后的参数缓冲区
  api::MemoryMap copy_mapping(
      params_copy.buffer(), api::MemoryAccessType::READ);

  // 将映射的数据解释为 TestStruct 类型指针
  TestStruct* test_copy_p = copy_mapping.template data<TestStruct>();

  // 断言复制后的数据与原始测试结构体的数据相等
  ASSERT_TRUE(test_copy_p->a == test_struct.a);
  ASSERT_TRUE(test_copy_p->b == test_struct.b);
  ASSERT_TRUE(test_copy_p->c == test_struct.c);
}

// VulkanAPI 测试类中的缓冲区复制测试
TEST_F(VulkanAPITest, copy_to_buffer) {
  using namespace at::native::vulkan;

  // 创建四个不同维度的随机张量，存储在数组中
  std::array<at::Tensor, 4> test_tensors = {
      // 4维
      at::rand(
          {7, 17, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
      // 3维
      at::rand({67, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
      // 2维
      at::rand({229, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
      // 1维
      at::rand({1902}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
  };

  // 对每个输入 CPU 张量执行操作
  for (auto in_cpu : test_tensors) {
    // 将 CPU 张量转换为 Vulkan 张量，存储在 vTensor 中
    vTensor in_vk_copied = ops::to_vulkan(in_cpu, api::StorageType::BUFFER);
    // 将 Vulkan 张量转换回 CPU 张量
    at::Tensor out_copied = ops::from_vulkan(in_vk_copied);

    // 检查复制后的张量是否与原始张量几乎相等
    const auto check_copy = almostEqual(out_copied, in_cpu);

    // 如果复制失败，则打印相关信息
    if (!check_copy) {
      std::cout << "Copy failed on size " << in_cpu.sizes() << "with dtype"
                << in_cpu.dtype() << std::endl;
    }

    // 断言复制成功
    ASSERT_TRUE(check_copy);
  }
}

// VulkanAPI 测试类中的通道最后缓冲区复制测试
TEST_F(VulkanAPITest, copy_to_buffer_channels_last) {
  using namespace at::native::vulkan;

  // 创建一个带通道最后内存格式的随机 4D CPU 张量
  at::TensorOptions options(at::kCPU);
  options = options.dtype(at::kFloat);
  std::array<at::Tensor, 1> test_tensors = {
      at::rand({7, 17, 134, 213}, options).to(at::MemoryFormat::ChannelsLast),
  };

  // 对每个输入 CPU 张量执行操作
  for (auto in_cpu : test_tensors) {
    // 将 CPU 张量转换为 Vulkan 张量，存储在 vTensor 中
    vTensor in_vk_copied = ops::to_vulkan(in_cpu, api::StorageType::BUFFER);
    // 将 Vulkan 张量转换回 CPU 张量
    at::Tensor out_copied = ops::from_vulkan(in_vk_copied);

    // 检查复制后的张量是否与原始张量几乎相等
    const auto check_copy = almostEqual(out_copied, in_cpu);
    // 如果复制失败，则输出错误消息，包含失败时的大小和数据类型信息
    if (!check_copy) {
      std::cout << "Copy failed on size " << in_cpu.sizes() << "with dtype"
                << in_cpu.dtype() << std::endl;
    }

    // 使用断言确保复制成功
    ASSERT_TRUE(check_copy);
}

// 结束当前的函数定义

// TODO: Fix vulkan to cpu on Android

// 定义一个测试夹具，测试 Vulkan API
TEST_F(VulkanAPITest, DISABLED_support_vulkan) {
  // 定义缩放因子和零点
  const double scale = 0.1;
  const int zero_point = 10;

  // 创建一个在 CPU 上的随机浮点数张量，并量化为 uint8 类型
  auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 12 -
      6;
  auto in_cpu_quantized = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);

  // 将量化后的 CPU 张量复制到 Vulkan 张量
  auto in_vulkan_quantized = cpu_to_vulkan(in_cpu_quantized);

  // 初始化 Vulkan API 的流水线屏障对象
  at::native::vulkan::api::PipelineBarrier pipeline_barrier{};

  // 转换 Vulkan 张量对象
  at::native::vulkan::vTensor& v_self =
      at::native::vulkan::ops::convert(in_vulkan_quantized);

  // 如果输入的 CPU 张量类型为 qint8，执行以下操作
  if (in_cpu.dtype() == c10::kQUInt8) {
    // 执行 Vulkan 图像操作，指定计算阶段和读取内存访问类型
    v_self.image(
        pipeline_barrier,
        at::native::vulkan::api::PipelineStage::COMPUTE,
        at::native::vulkan::api::MemoryAccessType::READ);
    // 执行 Vulkan 图像操作，指定计算阶段和写入内存访问类型
    v_self.image(
        pipeline_barrier,
        at::native::vulkan::api::PipelineStage::COMPUTE,
        at::native::vulkan::api::MemoryAccessType::WRITE);
  }

  // 将 Vulkan 张量转换回 CPU 张量
  auto output = vulkan_to_cpu(in_vulkan_quantized, in_cpu_quantized);

  // 检查输出是否几乎等于输入
  const auto check = almostEqual(
      at::native::int_repr_quantized_cpu(in_cpu_quantized),
      at::native::int_repr_quantized_cpu(output));

  // 如果检查不通过，显示相对误差
  if (!check) {
    showRtol(
        at::native::int_repr_quantized_cpu(in_cpu_quantized),
        at::native::int_repr_quantized_cpu(output));
  }

  // 使用断言确保检查通过
  ASSERT_TRUE(check);
}

// 定义一个测试函数，测试 CPU 到 Vulkan 和 Vulkan 到 CPU 的转换
void test_cpu_to_vulkan_and_vulkan_to_cpu(
    const at::IntArrayRef input_shape,
    const double scale,
    const int zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8) {
  // 生成随机的量化 CPU 张量
  auto in_cpu = produce_random_tensor(input_shape);
  auto in_q_cpu = at::quantize_per_tensor(in_cpu, scale, zero_point, dtype);

  // 将量化 CPU 张量复制到 Vulkan
  auto in_q_cpu_vk = cpu_to_vulkan(in_q_cpu);

  // 将量化 Vulkan 张量复制回 CPU
  auto out_q_cpu = vulkan_to_cpu(in_q_cpu_vk, in_q_cpu);

  // 检查复制的结果是否与原始输入相等
  const auto diff = at::native::int_repr_quantized_cpu(in_q_cpu) -
      at::native::int_repr_quantized_cpu(out_q_cpu);

  const int error = diff.abs().max().item<int>();

  const auto check = (error == 0);

  // 如果检查不通过，输出错误信息
  if (!check) {
    std::cout << "Copy to vulkan and back to cpu failed with input shape: "
              << input_shape << " scale: " << scale
              << " and zero point: " << zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }

  // 使用断言确保检查通过
  ASSERT_TRUE(check);
}

// 定义一个测试函数，测试随机量化 CPU 到 Vulkan 和 Vulkan 到 CPU 的转换
void test_cpu_to_vulkan_and_vulkan_to_cpu_random(const c10::ScalarType dtype) {
  // 生成随机的缩放因子和零点
  const double scale = produce_random_scale();
  const int zero_point = produce_random_zero_point(dtype);
  // 生成随机形状的张量
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  // 执行 CPU 到 Vulkan 和 Vulkan 到 CPU 的转换测试
  test_cpu_to_vulkan_and_vulkan_to_cpu(tensor_shape, scale, zero_point, dtype);
}

// TODO: Fix vulkan to cpu on Android
// 在 VulkanAPI 测试中，禁用 quint8 类型的 CPU 到 Vulkan 和 Vulkan 到 CPU 的转换测试
TEST_F(VulkanAPITest, DISABLED_cpu_to_vulkan_and_vulkan_to_cpu_quint8) {
  // 定义数据类型为 QUInt8
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  
  // 进行多组 CPU 到 Vulkan 和 Vulkan 到 CPU 的转换测试，参数依次是 tensor 形状、浮点数、整数、数据类型
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, 21, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, 120, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, 15, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, 43, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);

  // 对数据类型为 dtype 的随机测试进行 20 次迭代
  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_cpu_to_vulkan_and_vulkan_to_cpu_qint8) {
  // 定义数据类型为 QInt8
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  
  // 调用测试函数，测试 CPU 到 Vulkan 和 Vulkan 到 CPU 的数据传输，使用不同的参数
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, -120, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, -15, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, -43, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);

  // 使用循环调用随机数据测试函数 20 次
  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_cpu_to_vulkan_and_vulkan_to_cpu_qint32) {
  // 定义量化类型为 QInt32
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  // 调用函数测试从 CPU 到 Vulkan 和从 Vulkan 到 CPU 的数据传输和转换
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21123, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 4}, 0.339, 8734, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 4, 1}, 0.228, -12023, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 7, 7}, 0.338, 8723, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 8, 8}, 0.193, -1023, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 8, 8}, 0.0449, 972, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 11, 17}, 0.073, -15, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1572, 102, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 12, 17}, 0.147, -156, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 17, 12}, 0.129, 10448, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({2, 4, 17, 12}, 0.137, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, -43267, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1243, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1889, -19784, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1345, 196, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 25, 29}, 0.129, -19489, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);

  // 循环测试随机输入数据的 CPU 到 Vulkan 和解量化操作
  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_vulkan_to_cpu_random(dtype);
  }
}

void test_cpu_to_vulkan_and_dequantize(
    const at::IntArrayRef input_shape,
    const double scale,
    const int zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8) {
  // 生成随机量化的 CPU 张量
  auto in_cpu = produce_random_tensor(input_shape);
  // 对 CPU 张量进行量化，使用给定的缩放因子和零点，以及数据类型
  auto in_q_cpu = at::quantize_per_tensor(in_cpu, scale, zero_point, dtype);

  // 将量化的 CPU 张量复制到 Vulkan
  auto in_q_cpu_vk = cpu_to_vulkan(in_q_cpu);

  // 对张量进行解量化操作
  const auto out_cpu_deq = at::dequantize(in_q_cpu);
  const auto out_vk_deq = at::dequantize(in_q_cpu_vk);
  const auto out_vk_deq_cpu = out_vk_deq.cpu();

  // 检查解量化后的张量是否相等
  const auto check = almostEqual(out_cpu_deq, out_vk_deq_cpu);

  if (!check) {
    // 如果检查失败，计算错误的最大值并输出
    const auto error =
        at::abs(out_vk_deq_cpu - out_cpu_deq).max().item<float>();
    std::cout << "Copy cpu to vulkan and dequantize failed with input shape: "
              << input_shape << " scale: " << scale
              << " and zero point: " << zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }
  // 使用断言确保解量化后的张量相等
  ASSERT_TRUE(check);
}
// 定义一个测试函数，用于测试将 CPU 张量转换为 Vulkan 张量并反量化，使用随机产生的缩放因子和零点
void test_cpu_to_vulkan_and_dequantize_random(const c10::ScalarType dtype) {
  // 产生一个随机的缩放因子
  const double scale = produce_random_scale();
  // 产生一个随机的零点值，根据数据类型不同选择不同的范围
  const int zero_point = produce_random_zero_point(dtype);
  // 随机生成一个四维张量的形状
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  // 调用测试函数，测试将 CPU 张量转换为 Vulkan 张量并反量化的过程
  test_cpu_to_vulkan_and_dequantize(tensor_shape, scale, zero_point, dtype);
}

// 定义一个测试用例，测试将 CPU QUInt8 类型张量转换为 Vulkan 张量并反量化的过程
TEST_F(VulkanAPITest, cpu_to_vulkan_and_dequantize_quint8) {
  // 设置数据类型为 QUInt8
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  // 运行多组测试，每组指定不同的张量形状、缩放因子和零点值，测试转换过程的正确性
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 1}, 0.13, 21, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 4, 1}, 0.2, 120, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 8, 8}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 11, 17}, 0.07, 15, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({2, 4, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 14}, 0.009, 43, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 9, 17}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  // 随机测试 20 次，每次使用相同的数据类型 dtype
  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_dequantize_random(dtype);
  }
}
// 测试固定点量化整数类型为QInt8的函数 `test_cpu_to_vulkan_and_dequantize` 的 Vulkan API 实现
TEST_F(VulkanAPITest, cpu_to_vulkan_and_dequantize_qint8) {
  // 定义固定点量化整数类型为QInt8
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  // 调用测试函数，传入数据形状{1, 1, 1, 1}，比例因子0.13，偏置-21，数据类型为QInt8
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 1}, 0.13, -21, dtype);
  // 依次调用测试函数，传入不同数据形状、比例因子和偏置，数据类型为QInt8
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 4, 1}, 0.2, -120, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 8, 8}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 11, 17}, 0.07, -15, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 12, 17}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({2, 4, 17, 12}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 14}, 0.009, -43, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 9, 17}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 25, 29}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  // 对随机数据进行固定点量化整数类型为QInt8的测试
  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_dequantize_random(dtype);
  }
}

// 测试固定点量化整数类型为QInt32的函数 `test_cpu_to_vulkan_and_dequantize` 的 Vulkan API 实现
TEST_F(VulkanAPITest, cpu_to_vulkan_and_dequantize_qint32) {
  // 定义固定点量化整数类型为QInt32
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  // 调用测试函数，传入数据形状{1, 1, 1, 1}，比例因子0.13，偏置-21123，数据类型为QInt32
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 1}, 0.13, -21123, dtype);
  // 依次调用测试函数，传入不同数据形状、比例因子和偏置，数据类型为QInt32
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 4}, 0.339, 8734, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 4, 1}, 0.228, -12023, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 7, 7}, 0.338, 8723, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 8, 8}, 0.193, -1023, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 8, 8}, 0.0449, 972, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 11, 17}, 0.073, -15, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 12, 17}, 0.1572, 102, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 12, 17}, 0.147, -156, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 17, 12}, 0.129, 10448, dtype);
  test_cpu_to_vulkan_and_dequantize({2, 4, 17, 12}, 0.137, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 14}, 0.009, -43267, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 15}, 0.1243, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 9, 17}, 0.1889, -19784, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 25, 29}, 0.1345, 196, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 25, 29}, 0.129, -19489, dtype);
  test_cpu_to_vulkan_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  // 对随机数据进行固定点量化整数类型为QInt32的测试
  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_dequantize_random(dtype);
  }
}
// 在 Android 上禁用 Vulkan 后，修复量化操作至 CPU 的测试
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor) {
  // 创建一个在 CPU 上的随机张量，形状为 [2, 13, 32, 27]，数据类型为 float
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 使用 Vulkan API 将 CPU 上的张量转换到 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  // 设置量化参数
  const double scale = 0.1;
  const int zero_point = 10;

  // 在 CPU 上对输入张量进行量化操作，输出为 uint8 类型的张量
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  // 在 Vulkan 上对输入张量进行量化操作，输出为 uint8 类型的张量
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  // 将 Vulkan 张量转换回 CPU 上的张量
  auto output_for_quantized_vulkan = vulkan_to_cpu(out_vulkan, out_cpu);

  // 设置相对误差容忍度
  int rtol = 1;
  // 检查两个量化后的 CPU 张量的整数表示是否接近，使用相对容忍度 rtol
  const auto check = at::allclose(
      at::native::int_repr_quantized_cpu(out_cpu),
      at::native::int_repr_quantized_cpu(output_for_quantized_vulkan),
      rtol);

  // 如果检查不通过，则打印最大容忍误差值
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查通过
  ASSERT_TRUE(check);
}

// 根据给定的输入形状、量化参数和数据类型，在 CPU 和 Vulkan 之间测试量化和转换至 CPU 的操作
void test_quantize_per_tensor_and_vulkan_to_cpu(
    const at::IntArrayRef input_shape,
    const double input_scale,
    const int input_zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8,
    const int tolerance = 1) {
  // tolerance = 1，允许由于除以随机比例导致的精度差异最多 1 个单位的量化结果差异

  // 生成指定形状的随机张量
  at::Tensor input = produce_random_tensor(input_shape);

  // 在 CPU 上对输入张量进行量化操作
  at::Tensor out_q_cpu =
      at::quantize_per_tensor(input, input_scale, input_zero_point, dtype);

  // 在 Vulkan 上对输入张量进行量化操作
  at::Tensor out_q_vk = at::quantize_per_tensor(
      input.vulkan(), input_scale, input_zero_point, dtype);

  // 将 Vulkan 张量转换为 CPU 上的张量
  at::Tensor out_q_vk_cpu = vulkan_to_cpu(out_q_vk, out_q_cpu);

  // 计算两个量化后的 CPU 张量的整数表示之间的差异
  const auto diff = at::native::int_repr_quantized_cpu(out_q_vk_cpu) -
      at::native::int_repr_quantized_cpu(out_q_cpu);

  // 计算差异的最大绝对值，作为误差
  const int error = diff.abs().max().item<int>();

  // 检查误差是否在容忍范围内
  const auto check = (error <= tolerance);

  // 如果检查不通过，则打印相关错误信息
  if (!check) {
    std::cout << "Quantize and copy to cpu failed with input shape: "
              << input_shape << " scale: " << input_scale
              << " and zero point: " << input_zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }

  // 断言检查通过
  ASSERT_TRUE(check);
}

// 在给定数据类型上进行随机量化和从 Vulkan 转换至 CPU 的测试
void test_quantize_per_tensor_and_vulkan_to_cpu_random(
    const c10::ScalarType dtype) {
  // 生成随机的量化比例
  const double scale = produce_random_scale();
  // 生成随机的零点
  const int zero_point = produce_random_zero_point(dtype);
  // 生成随机形状的张量
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  // 执行量化和从 Vulkan 转换至 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu(
      tensor_shape, scale, zero_point, dtype);
}

// TODO: 在 Android 上修复 Vulkan 到 CPU 的问题
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor_and_vulkan_to_cpu_quint8) {
  // 设置数据类型为无符号 8 位整数（quint8）
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  // 调用测试函数，对 {1, 1, 1, 1} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, 21, dtype);
  // 调用测试函数，对 {1, 1, 1, 4} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  // 调用测试函数，对 {1, 1, 4, 1} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, 120, dtype);
  // 调用测试函数，对 {1, 1, 7, 7} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  // 调用测试函数，对 {1, 1, 8, 8} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, 10, dtype);
  // 调用测试函数，对 {3, 5, 8, 8} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  // 调用测试函数，对 {1, 1, 11, 17} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, 15, dtype);
  // 调用测试函数，对 {1, 1, 12, 17} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  // 调用测试函数，对 {3, 5, 12, 17} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, 10, dtype);
  // 调用测试函数，对 {1, 1, 17, 12} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  // 调用测试函数，对 {2, 4, 17, 12} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, 10, dtype);
  // 调用测试函数，对 {1, 1, 10, 14} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  // 调用测试函数，对 {3, 5, 10, 14} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, 43, dtype);
  // 调用测试函数，对 {3, 5, 10, 15} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  // 调用测试函数，对 {4, 4, 9, 17} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, 19, dtype);
  // 调用测试函数，对 {3, 5, 25, 29} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  // 调用测试函数，对 {4, 4, 25, 29} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, 19, dtype);
  // 调用测试函数，对 {11, 17, 25, 29} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);
  // 调用测试函数，对 {3, 16, 77, 54} 大小的张量进行量化和从 Vulkan 到 CPU 的测试
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 16, 77, 54}, 0.204173, 229, dtype);

  // 循环 20 次，调用随机生成数据的测试函数，对随机张量进行量化和从 Vulkan 到 CPU 的测试
  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: 修复在安卓平台上的 Vulkan 到 CPU 的问题
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor_and_vulkan_to_cpu_qint8) {
  // 定义量化后数据类型为QInt8
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  // 调用测试函数，量化单个张量，并在Vulkan和CPU之间进行转换，传入张量形状{1, 1, 1, 1}、比例因子0.13、零点-21和数据类型QInt8
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21, dtype);
  // 同上，传入张量形状{1, 1, 1, 4}、比例因子0.3、零点87
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  // 同上，传入张量形状{1, 1, 4, 1}、比例因子0.2、零点-120
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, -120, dtype);
  // 同上，传入张量形状{1, 1, 7, 7}、比例因子0.3、零点87
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  // 同上，传入张量形状{1, 1, 8, 8}、比例因子0.1、零点-10
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, -10, dtype);
  // 同上，传入张量形状{3, 5, 8, 8}、比例因子0.04、零点97
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  // 同上，传入张量形状{1, 1, 11, 17}、比例因子0.07、零点-15
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, -15, dtype);
  // 同上，传入张量形状{1, 1, 12, 17}、比例因子0.1、零点10
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  // 同上，传入张量形状{3, 5, 12, 17}、比例因子0.1、零点-10
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, -10, dtype);
  // 同上，传入张量形状{1, 1, 17, 12}、比例因子0.1、零点10
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  // 同上，传入张量形状{2, 4, 17, 12}、比例因子0.1、零点-10
  test_quantize_per_tensor_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, -10, dtype);
  // 同上，传入张量形状{1, 1, 10, 14}、比例因子0.0001、零点101
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  // 同上，传入张量形状{3, 5, 10, 14}、比例因子0.009、零点-43
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, -43, dtype);
  // 同上，传入张量形状{3, 5, 10, 15}、比例因子0.1、零点19
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  // 同上，传入张量形状{4, 4, 9, 17}、比例因子0.1、零点-19
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, -19, dtype);
  // 同上，传入张量形状{3, 5, 25, 29}、比例因子0.1、零点19
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  // 同上，传入张量形状{4, 4, 25, 29}、比例因子0.1、零点-19
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, -19, dtype);
  // 同上，传入张量形状{11, 17, 25, 29}、比例因子0.027、零点89
  test_quantize_per_tensor_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);
  // 同上，传入张量形状{3, 16, 77, 54}、比例因子0.204173、零点229
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 16, 77, 54}, 0.204173, 229, dtype);

  // 循环20次，每次调用随机量化函数，用于测试不同的随机张量
  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor_and_vulkan_to_cpu_qint32) {
  // 定义数据类型为 QInt32
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  // 调用测试函数，验证量化过程对 CPU 和 Vulkan 的影响
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21123, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 4}, 0.339, 8734, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 4, 1}, 0.228, -12023, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 7, 7}, 0.338, 8723, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 8, 8}, 0.193, -1023, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 8, 8}, 0.0449, 972, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 11, 17}, 0.073, -15, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 12, 17}, 0.1572, 102, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 5, 12, 17}, 0.147, -156, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 17, 12}, 0.129, 10448, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({2, 4, 17, 12}, 0.137, -10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 10, 14}, 0.0001, 101, dtype, 1);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 5, 10, 14}, 0.009, -43267, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1243, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {4, 4, 9, 17}, 0.1889, -19784, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 5, 25, 29}, 0.1345, 196, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {4, 4, 25, 29}, 0.129, -19489, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {11, 17, 25, 29}, 0.027, 89, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 16, 77, 54}, 0.204173, 229, dtype);

  // 随机测试量化和反量化过程，执行20次
  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_vulkan_to_cpu_random(dtype);
  }
}

TEST_F(VulkanAPITest, quantize_dequantize) {
  // 在 CPU 上生成一个随机张量
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;
  // 对张量进行量化操作
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  // 在 Vulkan 上执行量化操作
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  // 对量化后的张量进行反量化操作
  const auto out_cpu_deq = at::dequantize(out_cpu);
  // 在 Vulkan 上对量化后的张量进行反量化操作
  const auto out_vulkan_deq = at::native::vulkan::ops::dequantize(out_vulkan);
  // 将 Vulkan 张量反量化后，转换回 CPU 张量
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu);

  float rtol = 1;
  float atol = 0.5;
  // 检查反量化后的结果是否与原始 CPU 张量接近
  const auto check =
      at::allclose(in_cpu, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    // 输出允许的最大差异
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查，确保反量化后的结果与原始 CPU 张量接近
  ASSERT_TRUE(check);

  // 再次检查，确保 CPU 和 Vulkan 的反量化结果一致
  const auto check_two =
      at::allclose(out_cpu_deq, output_for_dequantized_vulkan, rtol, atol);

  if (!check_two) {
    // 输出最大允许的差异值 rtol 到标准输出流
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }
  
  // 断言确保 check_two 为真
  ASSERT_TRUE(check_two);
void test_quantize_per_tensor_and_dequantize(
    const at::IntArrayRef input_shape,
    const double input_scale,
    const int input_zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8,
    bool use_qparams = false) {
  // 生成一个随机的输入张量
  at::Tensor input = produce_random_tensor(input_shape);

  // 创建输入 scale 的张量参数
  at::Tensor input_scale_qparam = at::empty({1});
  input_scale_qparam[0] = input_scale;

  // 创建输入 zero_point 的张量参数
  at::Tensor input_zero_point_qparam = at::empty({1});
  input_zero_point_qparam[0] = input_zero_point;

  // 对张量进行量化
  at::Tensor out_q_cpu = use_qparams
      ? at::quantize_per_tensor(
            input, input_scale_qparam, input_zero_point_qparam, dtype)
      : at::quantize_per_tensor(input, input_scale, input_zero_point, dtype);

  // 使用 Vulkan 后端进行量化
  at::Tensor out_q_vk = use_qparams
      ? at::quantize_per_tensor(
            input.vulkan(), input_scale_qparam, input_zero_point_qparam, dtype)
      : at::quantize_per_tensor(
            input.vulkan(), input_scale, input_zero_point, dtype);

  // 对张量进行反量化
  const auto out_cpu_deq = at::dequantize(out_q_cpu);
  const auto out_vk_deq = at::dequantize(out_q_vk);
  const auto out_vk_deq_cpu = out_vk_deq.cpu();

  // 检查反量化后的张量是否相等
  const float tolerance = safe_downcast<float>(input_scale);
  // 容许的误差等于 scale，以允许在除以随机 scale 后的精度差异
  // 这可能导致量化结果差异为 1 单位。
  const auto check = almostEqual(out_cpu_deq, out_vk_deq_cpu, tolerance);

  // 如果不相等，则输出错误信息
  if (!check) {
    const auto error =
        at::abs(out_vk_deq_cpu - out_cpu_deq).max().item<float>();
    std::cout << "Quantize and Dequantize failed with input shape: "
              << input_shape << " scale: " << input_scale
              << " and zero point: " << input_zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }
  // 断言张量相等
  ASSERT_TRUE(check);
}

void test_quantize_per_tensor_and_dequantize_random(
    const c10::ScalarType dtype,
    bool use_qparams = false) {
  // 生成随机的 scale 值
  const double scale = produce_random_scale();
  // 生成随机的 zero_point 值
  const int zero_point = produce_random_zero_point(dtype);
  // 生成随机形状的张量
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  // 调用测试函数来测试量化和反量化过程
  test_quantize_per_tensor_and_dequantize(
      tensor_shape, scale, zero_point, dtype, use_qparams);
}
# 在 VulkanAPITest 测试固件中，测试量化为每张张量和反量化为 8 位无符号整数类型的函数
TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_quint8) {
    # 定义数据类型为 8 位无符号整数
    const c10::ScalarType dtype = c10::ScalarType::QUInt8;
    # 调用测试函数，量化和反量化张量，参数为 {1, 1, 1, 1}，量化因子 0.13，偏移 21，数据类型为 dtype
    test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, 21, dtype);
    # 同上，参数为 {1, 1, 1, 4}，量化因子 0.3，偏移 87
    test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
    # 同上，参数为 {1, 1, 4, 1}，量化因子 0.2，偏移 120
    test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, 120, dtype);
    # 同上，参数为 {1, 1, 7, 7}，量化因子 0.3，偏移 87
    test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
    # 同上，参数为 {1, 1, 8, 8}，量化因子 0.1，偏移 10
    test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, 10, dtype);
    # 同上，参数为 {3, 5, 8, 8}，量化因子 0.04，偏移 97
    test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
    # 同上，参数为 {1, 1, 11, 17}，量化因子 0.07，偏移 15
    test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.07, 15, dtype);
    # 同上，参数为 {1, 1, 12, 17}，量化因子 0.1，偏移 10
    test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
    # 同上，参数为 {3, 5, 12, 17}，量化因子 0.1，偏移 10
    test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.1, 10, dtype);
    # 同上，参数为 {1, 1, 17, 12}，量化因子 0.1，偏移 10
    test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
    # 同上，参数为 {2, 4, 17, 12}，量化因子 0.1，偏移 10
    test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.1, 10, dtype);
    # 同上，参数为 {1, 1, 10, 14}，量化因子 0.001，偏移 101
    test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype);
    # 同上，参数为 {3, 5, 10, 14}，量化因子 0.009，偏移 43
    test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, 43, dtype);
    # 同上，参数为 {3, 5, 10, 15}，量化因子 0.1，偏移 19
    test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
    # 同上，参数为 {4, 4, 9, 17}，量化因子 0.1，偏移 19
    test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, 19, dtype);
    # 同上，参数为 {3, 5, 25, 29}，量化因子 0.1，偏移 19
    test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
    # 同上，参数为 {4, 4, 25, 29}，量化因子 0.1，偏移 19
    test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.1, 19, dtype);
    # 同上，参数为 {11, 17, 25, 29}，量化因子 0.027，偏移 89
    test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

    # 循环调用 20 次，随机量化和反量化函数，数据类型为 dtype
    for (int i = 0; i < 20; i += 1) {
        test_quantize_per_tensor_and_dequantize_random(dtype);
    }
}
# 在 VulkanAPITest 测试框架下，测试量化和反量化函数对于QUInt8数据类型的张量的功能
TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_quint8_qparams) {
  # 定义数据类型为QUInt8
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  
  # 调用测试函数，对{1, 1, 1, 1}的张量进行量化和反量化，指定0.13为量化参数，21为反量化参数，测试QUInt8数据类型，打开详细输出
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, 21, dtype, true);
  
  # 同上，对{1, 1, 1, 4}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype, true);
  
  # 同上，对{1, 1, 4, 1}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, 120, dtype, true);
  
  # 同上，对{1, 1, 7, 7}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype, true);
  
  # 同上，对{1, 1, 8, 8}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, 10, dtype, true);
  
  # 同上，对{3, 5, 8, 8}的张量进行测试
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype, true);
  
  # 同上，对{1, 1, 11, 17}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.07, 15, dtype, true);
  
  # 同上，对{1, 1, 12, 17}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype, true);
  
  # 同上，对{3, 5, 12, 17}的张量进行测试
  test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.1, 10, dtype, true);
  
  # 同上，对{1, 1, 17, 12}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype, true);
  
  # 同上，对{2, 4, 17, 12}的张量进行测试
  test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.1, 10, dtype, true);
  
  # 同上，对{1, 1, 10, 14}的张量进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype, true);
  
  # 同上，对{3, 5, 10, 14}的张量进行测试
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, 43, dtype, true);
  
  # 同上，对{3, 5, 10, 15}的张量进行测试
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype, true);
  
  # 同上，对{4, 4, 9, 17}的张量进行测试
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, 19, dtype, true);
  
  # 同上，对{3, 5, 25, 29}的张量进行测试
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype, true);
  
  # 同上，对{4, 4, 25, 29}的张量进行测试
  test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.1, 19, dtype, true);
  
  # 同上，对{11, 17, 25, 29}的张量进行测试
  test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype, true);

  # 使用随机数据对QUInt8数据类型的张量进行20次随机测试
  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype, true);
  }
}
# 在 VulkanAPITest 测试 fixture 中，测试量化和反量化 QInt8 类型的张量
TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint8) {
  # 定义数据类型为 QInt8
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  
  # 调用测试函数，测试量化和反量化，输入张量形状为 {1, 1, 1, 1}，比例因子为 0.13，零点为 -21
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, -21, dtype);
  # 同上，但输入张量形状为 {1, 1, 1, 4}，比例因子为 0.3，零点为 87
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
  # 同上，但输入张量形状为 {1, 1, 4, 1}，比例因子为 0.2，零点为 -120
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, -120, dtype);
  # 同上，但输入张量形状为 {1, 1, 7, 7}，比例因子为 0.3，零点为 87
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
  # 同上，但输入张量形状为 {1, 1, 8, 8}，比例因子为 0.1，零点为 -10
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, -10, dtype);
  # 同上，但输入张量形状为 {3, 5, 8, 8}，比例因子为 0.04，零点为 97
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
  # 同上，但输入张量形状为 {1, 1, 11, 17}，比例因子为 0.07，零点为 -15
  test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.07, -15, dtype);
  # 同上，但输入张量形状为 {1, 1, 12, 17}，比例因子为 0.1，零点为 10
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
  # 同上，但输入张量形状为 {3, 5, 12, 17}，比例因子为 0.1，零点为 -10
  test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.1, -10, dtype);
  # 同上，但输入张量形状为 {1, 1, 17, 12}，比例因子为 0.1，零点为 10
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
  # 同上，但输入张量形状为 {2, 4, 17, 12}，比例因子为 0.1，零点为 -10
  test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.1, -10, dtype);
  # 同上，但输入张量形状为 {1, 1, 10, 14}，比例因子为 0.001，零点为 101
  test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype);
  # 同上，但输入张量形状为 {3, 5, 10, 14}，比例因子为 0.009，零点为 -43
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, -43, dtype);
  # 同上，但输入张量形状为 {3, 5, 10, 15}，比例因子为 0.1，零点为 19
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
  # 同上，但输入张量形状为 {4, 4, 9, 17}，比例因子为 0.1，零点为 -19
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, -19, dtype);
  # 同上，但输入张量形状为 {3, 5, 25, 29}，比例因子为 0.1，零点为 19
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
  # 同上，但输入张量形状为 {4, 4, 25, 29}，比例因子为 0.1，零点为 -19
  test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.1, -19, dtype);
  # 同上，但输入张量形状为 {11, 17, 25, 29}，比例因子为 0.027，零点为 89
  test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  # 循环测试随机生成的张量量化和反量化，共 20 次
  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype);
  }
}
# 在 VulkanAPITest 测试固件中，测试量化为每个张量和反量化为 QInt8 类型的量化参数
TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint8_qparams) {
  # 定义变量 dtype 为 QInt8 类型
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  
  # 调用测试函数 test_quantize_per_tensor_and_dequantize，分别传入不同的参数进行测试
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, -21, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, -120, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, -10, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.07, -15, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.1, -10, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.1, -10, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, -43, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, -19, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.1, -19, dtype, true);
  test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype, true);

  # 使用循环进行随机测试，循环 20 次
  for (int i = 0; i < 20; i += 1) {
    # 调用随机测试函数 test_quantize_per_tensor_and_dequantize_random，传入 dtype 和 true 作为参数
    test_quantize_per_tensor_and_dequantize_random(dtype, true);
  }
}
# 在 VulkanAPITest 测试套件中，测试量化为每个张量和反量化为 QInt32 类型的函数
TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint32) {
    # 定义变量 dtype，指定为 QInt32 类型
    const c10::ScalarType dtype = c10::ScalarType::QInt32;
    
    # 调用测试函数 test_quantize_per_tensor_and_dequantize，分别测试不同维度和参数的情况
    test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, -21123, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.339, 8734, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.228, -12023, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.338, 8723, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.193, -1023, dtype);
    test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.0449, 972, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.073, -15, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1572, 102, dtype);
    test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.147, -156, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.129, 10448, dtype);
    test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.137, -10, dtype);
    test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype);
    test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, -43267, dtype);
    test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1243, 19, dtype);
    test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1889, -19784, dtype);
    test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1345, 196, dtype);
    test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.129, -19489, dtype);
    test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);
    
    # 使用循环进行随机测试，测试量化和反量化的随机数据
    for (int i = 0; i < 20; i += 1) {
        test_quantize_per_tensor_and_dequantize_random(dtype);
    }
}
# 在 VulkanAPITest 测试套件中定义了一个测试函数，用于测试量化和反量化 QInt32 类型的张量
TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint32_qparams) {
  # 定义变量 dtype 为 QInt32 类型，用于表示张量的数据类型
  const c10::ScalarType dtype = c10::ScalarType::QInt32;

  # 调用 test_quantize_per_tensor_and_dequantize 函数进行测试，传入张量维度 {1, 1, 1, 1}，量化因子 0.13，量化零点 -21123，数据类型 dtype，及 true 作为参数
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 1, 1}, 0.13, -21123, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 1, 4}，量化因子 0.339，量化零点 8734
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 1, 4}, 0.339, 8734, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 4, 1}，量化因子 0.228，量化零点 -12023
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 4, 1}, 0.228, -12023, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 7, 7}，量化因子 0.338，量化零点 8723
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 7, 7}, 0.338, 8723, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 8, 8}，量化因子 0.193，量化零点 -1023
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 8, 8}, 0.193, -1023, dtype, true);
  
  # 同上，测试张量维度 {3, 5, 8, 8}，量化因子 0.0449，量化零点 972
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 8, 8}, 0.0449, 972, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 11, 17}，量化因子 0.073，量化零点 -15
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 11, 17}, 0.073, -15, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 12, 17}，量化因子 0.1572，量化零点 102
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 12, 17}, 0.1572, 102, dtype, true);
  
  # 同上，测试张量维度 {3, 5, 12, 17}，量化因子 0.147，量化零点 -156
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 12, 17}, 0.147, -156, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 17, 12}，量化因子 0.129，量化零点 10448
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 17, 12}, 0.129, 10448, dtype, true);
  
  # 同上，测试张量维度 {2, 4, 17, 12}，量化因子 0.137，量化零点 -10
  test_quantize_per_tensor_and_dequantize(
      {2, 4, 17, 12}, 0.137, -10, dtype, true);
  
  # 同上，测试张量维度 {1, 1, 10, 14}，量化因子 0.001，量化零点 101
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 10, 14}, 0.001, 101, dtype, true);
  
  # 同上，测试张量维度 {3, 5, 10, 14}，量化因子 0.009，量化零点 -43267
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 10, 14}, 0.009, -43267, dtype, true);
  
  # 同上，测试张量维度 {3, 5, 10, 15}，量化因子 0.1243，量化零点 19
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 10, 15}, 0.1243, 19, dtype, true);
  
  # 同上，测试张量维度 {4, 4, 9, 17}，量化因子 0.1889，量化零点 -19784
  test_quantize_per_tensor_and_dequantize(
      {4, 4, 9, 17}, 0.1889, -19784, dtype, true);
  
  # 同上，测试张量维度 {3, 5, 25, 29}，量化因子 0.1345，量化零点 196
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 25, 29}, 0.1345, 196, dtype, true);
  
  # 同上，测试张量维度 {4, 4, 25, 29}，量化因子 0.129，量化零点 -19489
  test_quantize_per_tensor_and_dequantize(
      {4, 4, 25, 29}, 0.129, -19489, dtype, true);
  
  # 同上，测试张量维度 {11, 17, 25, 29}，量化因子 0.027，量化零点 89
  test_quantize_per_tensor_and_dequantize(
      {11, 17, 25, 29}, 0.027, 89, dtype, true);

  # 循环调用 test_quantize_per_tensor_and_dequantize_random 函数 20 次，测试随机量化和反量化 QInt32 类型的张量
  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype, true);
  }
}
TEST_F(VulkanAPITest, quantized_add) {
  // 创建一个大小为 [2, 13, 32, 27] 的随机张量，使用 CPU 设备和浮点数类型，并乘以 6
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  
  // 创建第二个相同大小的随机张量，同样使用 CPU 设备和浮点数类型，并乘以 6
  const auto in_cpu2 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将第二个 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan2 = in_cpu2.vulkan();

  // 定义量化参数：缩放因子为 0.1，零点为 10
  const double scale = 0.1;
  const int zero_point = 10;

  // 对第一个 CPU 张量进行量化，使用给定的缩放因子和零点，量化到无符号 8 位整数类型
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  // 对第二个 CPU 张量进行同样的量化操作
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  // 使用 Vulkan 实现的量化操作，对第一个 Vulkan 张量进行量化
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  // 使用 Vulkan 实现的量化操作，对第二个 Vulkan 张量进行量化
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  // 定义第二组量化参数：缩放因子为 0.15，零点为 15
  const double scale3 = 0.15;
  const int zero_point3 = 15;
  
  // 调用量化加法操作 "quantized::add"，传入量化后的 CPU 张量及其参数
  const auto reg_added_tensors = callOpByName(
      "quantized::add", "", out_cpu, out_cpu2, scale3, zero_point3);
  // 使用 Vulkan 实现的量化加法操作，传入量化后的 Vulkan 张量及其参数
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  // 对 Vulkan 张量执行反量化操作
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  // 将反量化后的 Vulkan 张量转换回 CPU 张量
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  // 设置相对容差和绝对容差
  float rtol = 0;
  float atol = 0.5;
  // 检查两个张量是否在容差范围内相等
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()),
      output_for_dequantized_vulkan,
      rtol,
      atol);

  // 如果检查未通过，输出容许的最大差异
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 使用断言确保检查通过
  ASSERT_TRUE(check);
}
# VulkanAPITest 测试类的第一个测试案例，测试量化加法的广播操作

TEST_F(VulkanAPITest, quantized_add_broadcast) {
  # 创建一个形状为 [2, 13, 1, 27] 的随机张量，元素类型为 float，分布在 CPU 上，值范围为 [0, 6)
  const auto in_cpu = at::rand({2, 13, 1, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  # 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  # 创建一个形状为 [2, 13, 32, 1] 的随机张量，元素类型为 float，分布在 CPU 上，值范围为 [0, 6)
  const auto in_cpu2 = at::rand({2, 13, 32, 1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  # 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan2 = in_cpu2.vulkan();

  # 定义量化参数
  const double scale = 0.1;
  const int zero_point = 10;

  # 对 in_cpu 进行量化操作，得到量化后的输出张量 out_cpu
  const auto out_cpu = at::quantize_per_tensor(in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  # 对 in_cpu2 进行量化操作，得到量化后的输出张量 out_cpu2
  const auto out_cpu2 = at::quantize_per_tensor(in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  # 在 Vulkan 上执行量化操作，得到量化后的输出张量 out_vulkan
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  # 在 Vulkan 上执行量化操作，得到量化后的输出张量 out_vulkan2
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  # 定义另一组量化参数
  const double scale3 = 0.15;
  const int zero_point3 = 15;

  # 调用 "quantized::add" 运算符，对 out_cpu 和 out_cpu2 执行量化加法
  const auto reg_added_tensors = callOpByName("quantized::add", "", out_cpu, out_cpu2, scale3, zero_point3);
  # 在 Vulkan 上执行量化加法操作，对 out_vulkan 和 out_vulkan2 进行量化加法
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(out_vulkan, out_vulkan2, scale3, zero_point3);

  # 创建一个形状为 [2, 13, 32, 27] 的随机张量，元素类型为 float，分布在 CPU 上，值范围为 [0, 6)
  const auto in_cpu3 = at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  # 对 Vulkan 张量进行反量化操作，得到反量化后的输出张量 out_vulkan_deq
  const auto out_vulkan_deq = at::native::vulkan::ops::dequantize(vulk_added_tensors);
  # 将 Vulkan 张量 out_vulkan_deq 转换为 CPU 张量，与 in_cpu3 进行比较
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

  # 设置相对误差和绝对误差的容忍度
  float rtol = 0;
  float atol = 0.5;
  # 检查两个张量是否在容忍误差内相等
  const auto check = at::allclose(at::dequantize(reg_added_tensors[0].toTensor()),
                                  output_for_dequantized_vulkan,
                                  rtol,
                                  atol);

  # 如果检查不通过，则输出最大允许差异
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  # 断言检查结果为真
  ASSERT_TRUE(check);
}
    // 返回语句，结束当前函数的执行并返回调用者
    return;
  }

  // 创建一个形状为 {2, 12, 32, 27} 的随机张量，使用 CPU 设备和 float 数据类型，并乘以 6
  const auto in_cpu =
      at::rand({2, 12, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 创建另一个形状为 {12, 1, 1} 的随机张量，使用 CPU 设备和 float 数据类型，并乘以 6
  const auto in_cpu2 =
      at::rand({12, 1, 1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 CPU 上的第二个张量转换为 Vulkan 张量
  const auto in_vulkan2 = in_cpu2.vulkan();

  // 定义量化的比例和零点值
  const double scale = 0.1;
  const int zero_point = 10;

  // 对第一个 CPU 张量进行张量级别的量化
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  // 对第二个 CPU 张量进行张量级别的量化
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  // 对第一个 Vulkan 张量进行 Vulkan 操作级别的量化
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  // 对第二个 Vulkan 张量进行 Vulkan 操作级别的量化
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  // 定义第三个量化的比例和零点值
  const double scale3 = 0.15;
  const int zero_point3 = 15;
  // 调用自定义函数，执行量化加法操作，并获得结果
  const auto reg_added_tensors = callOpByName(
      "quantized::add", "", out_cpu, out_cpu2, scale3, zero_point3);
  // 调用 Vulkan 操作级别的量化加法，并获得结果
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  // 创建第三个形状为 {2, 12, 32, 27} 的随机张量，使用 CPU 设备和 float 数据类型，并乘以 6
  const auto in_cpu3 =
      at::rand({2, 12, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 对 Vulkan 张量执行反量化操作，并获得结果
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  // 将 Vulkan 张量的结果转换为 CPU 张量的输出
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

  // 设置相对误差和绝对误差的阈值
  float rtol = 0;
  float atol = 0.5;
  // 检查两个张量是否在指定的相对误差和绝对误差下全部相等
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()),
      output_for_dequantized_vulkan,
      rtol,
      atol);

  // 如果检查不通过，输出允许的最大差异
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_add_broadcast2) {
  // 检查 Vulkan 是否可用，如果不可用则退出测试
  if (!at::is_vulkan_available()) {
    return;
  }

  // 创建具有指定形状和数据类型的随机张量 in_cpu
  const auto in_cpu =
      at::rand({32, 1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 in_cpu 转换为 Vulkan 张量 in_vulkan
  const auto in_vulkan = in_cpu.vulkan();
  // 创建具有指定形状和数据类型的随机张量 in_cpu2
  const auto in_cpu2 =
      at::rand({1, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 in_cpu2 转换为 Vulkan 张量 in_vulkan2
  const auto in_vulkan2 = in_cpu2.vulkan();

  // 定义量化参数 scale 和 zero_point
  const double scale = 0.1;
  const int zero_point = 10;

  // 使用量化参数对 in_cpu 进行张量量化，得到 out_cpu
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  // 使用量化参数对 in_cpu2 进行张量量化，得到 out_cpu2
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  // 使用 Vulkan 操作对 in_vulkan 进行张量量化，得到 out_vulkan
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  // 使用 Vulkan 操作对 in_vulkan2 进行张量量化，得到 out_vulkan2
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  // 定义第三组量化参数 scale3 和 zero_point3
  const double scale3 = 0.15;
  const int zero_point3 = 15;
  // 调用 quantized::add 操作对量化后的张量进行加法运算，得到 reg_added_tensors
  const auto reg_added_tensors = callOpByName(
      "quantized::add", "", out_cpu, out_cpu2, scale3, zero_point3);
  // 使用 Vulkan 操作对量化后的张量进行加法运算，得到 vulk_added_tensors
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  // 创建具有指定形状和数据类型的随机张量 in_cpu3
  const auto in_cpu3 =
      at::rand({32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 对 Vulkan 中的结果张量进行反量化，得到 out_vulkan_deq
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  // 将 Vulkan 张量 out_vulkan_deq 转换为 CPU 张量，使用 in_cpu3 作为参考
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

  // 设置相对误差和绝对误差的容忍度
  float rtol = 0;
  float atol = 0.5;
  // 检查两个张量是否在给定的相对误差和绝对误差容忍度内相等
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()),
      output_for_dequantized_vulkan,
      rtol,
      atol);

  // 如果检查不通过，打印允许的最大差异
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_add_broadcast3) {
  // 检查 Vulkan 是否可用，如果不可用则退出测试
  if (!at::is_vulkan_available()) {
  // 空语句，用于退出函数
  return;
}

const auto in_cpu =
    at::rand({32, 24}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
const auto in_vulkan = in_cpu.vulkan();
const auto in_cpu2 =
    at::rand({1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
const auto in_vulkan2 = in_cpu2.vulkan();

const double scale = 0.1;
const int zero_point = 10;

// 使用量化函数 quantize_per_tensor 对输入进行量化处理，返回量化后的张量
const auto out_cpu = at::quantize_per_tensor(
    in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
const auto out_cpu2 = at::quantize_per_tensor(
    in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);

// 使用 Vulkan 操作中的量化函数 quantize_per_tensor 对输入进行量化处理，返回量化后的张量
const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
    in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
    in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

const double scale3 = 0.15;
const int zero_point3 = 15;

// 调用自定义函数 callOpByName 执行量化后的张量加法操作，返回结果张量
const auto reg_added_tensors = callOpByName(
    "quantized::add", "", out_cpu, out_cpu2, scale3, zero_point3);

// 使用 Vulkan 操作中的量化加法函数 quantized_add 执行量化后的张量加法操作，返回结果张量
const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
    out_vulkan, out_vulkan2, scale3, zero_point3);

const auto in_cpu3 =
    at::rand({32, 24}, at::device(at::kCPU).dtype(at::kFloat)) * 6;

// 使用 Vulkan 操作中的反量化函数 dequantize 对张量进行反量化处理，返回反量化后的张量
const auto out_vulkan_deq =
    at::native::vulkan::ops::dequantize(vulk_added_tensors);

// 调用自定义函数 vulkan_to_cpu 将 Vulkan 张量转换为 CPU 张量，返回 CPU 张量
auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

float rtol = 0;
float atol = 0.5;

// 使用 allclose 函数比较两个张量是否接近，根据指定的 rtol 和 atol 判断
const auto check = at::allclose(
    at::dequantize(reg_added_tensors[0].toTensor()),
    output_for_dequantized_vulkan,
    rtol,
    atol);

// 如果比较结果为假，则输出最大允许的差异值 rtol
if (!check) {
  std::cout << "Max Diff allowed: " << rtol << std::endl;
}

// 断言比较结果为真，即确保张量比较没有问题
ASSERT_TRUE(check);
TEST_F(VulkanAPITest, quantized_add_dif_params) {
  // 生成一个大小为 [2, 13, 32, 27] 的随机张量，使用 CPU 设备和 float 类型，并乘以 6
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 CPU 上的张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 生成另一个与 in_cpu 同样大小的随机张量
  const auto in_cpu2 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 CPU 上的第二个张量转换为 Vulkan 张量
  const auto in_vulkan2 = in_cpu2.vulkan();
  // 定义量化的参数
  const double scale = 0.1;
  const int zero_point = 10;
  const double scale2 = 0.2;
  const int zero_point2 = 20;

  // 对第一个输入张量进行量化
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  // 对第二个输入张量进行量化
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale2, zero_point2, c10::ScalarType::QUInt8);
  // 使用 Vulkan 运算对第一个 Vulkan 张量进行量化
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  // 使用 Vulkan 运算对第二个 Vulkan 张量进行量化
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale2, zero_point2, c10::ScalarType::QUInt8);

  // 定义新的量化参数
  const double scale3 = 0.15;
  const int zero_point3 = 15;
  // 调用 quantized::add 操作，对两个量化后的张量进行加法运算
  const auto reg_added_tensors = callOpByName(
      "quantized::add", "", out_cpu, out_cpu2, scale3, zero_point3);
  // 使用 Vulkan 运算对两个量化后的 Vulkan 张量进行加法运算
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  // 对 Vulkan 张量进行反量化操作
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  // 将反量化后的 Vulkan 张量转换为 CPU 张量
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  // 定义相对误差和绝对误差的阈值
  float rtol = 0;
  float atol = 0.5;
  // 检查两个张量的相等性，根据给定的相对误差和绝对误差进行比较
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()),
      output_for_dequantized_vulkan,
      rtol,
      atol);

  // 如果检查不通过，则输出最大允许的差异值
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查通过
  ASSERT_TRUE(check);
}
  }
} weights{1, input.channels, 3, 3};

float r1 = 0.1;  // 设置变量 r1 并赋值为 0.1
float r2 = 0.7;  // 设置变量 r2 并赋值为 0.7
const auto input_cpu = (r1 - r2) *
        at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat)) +
    r2;  // 生成一个指定形状和数据类型的随机张量 input_cpu，使用 CPU 设备，并应用线性变换 (r1 - r2) * rand + r2

const auto weights_cpu = (r1 - r2) *
        at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat)) +
    r2;  // 生成一个指定形状和数据类型的随机张量 weights_cpu，使用 CPU 设备，并应用线性变换 (r1 - r2) * rand + r2

const auto bias_cpu = (r1 - r2) *
        at::rand({weights.output_channels},
                 at::device(at::kCPU).dtype(at::kFloat)) +
    r2;  // 生成一个指定形状和数据类型的随机张量 bias_cpu，使用 CPU 设备，并应用线性变换 (r1 - r2) * rand + r2

const double w_scale = 0.1;  // 设置权重的量化参数 w_scale
const int w_zero_point = 10;  // 设置权重的量化参数 w_zero_point

const double b_scale = 0.1;  // 设置偏置的量化参数 b_scale
const int b_zero_point = 10;  // 设置偏置的量化参数 b_zero_point

const auto weight_q = at::quantize_per_tensor(
    weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);  // 对权重进行张量级别的量化操作

const auto bias_q = at::quantize_per_tensor(
    bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);  // 对偏置进行张量级别的量化操作

const auto output_cpu = at::conv2d(
    input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);  // 执行 CPU 上的二维卷积操作

const double scale = 0.10;  // 设置量化的比例因子 scale
const int zero_point = 10;  // 设置量化的零点 zero_point
const auto shape_match =
    at::rand({1, 1, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)) * 6;  // 生成指定形状和数据类型的随机张量 shape_match

const auto in_vulkan = input_cpu.vulkan();  // 将 CPU 上的输入张量转换为 Vulkan 张量
const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
    in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);  // 对 Vulkan 张量进行张量级别的量化操作

const double scale2 = 0.15;  // 设置第二个量化的比例因子 scale2
const int zero_point2 = 15;  // 设置第二个量化的零点 zero_point2
const auto output_vulkan = at::native::vulkan::ops::quantized_conv2d(
    out_vulkan,
    weight_q,
    bias_quantized ? bias_q : bias_cpu,
    stride,
    padding,
    dilation,
    groups,
    scale2,
    zero_point2);  // 在 Vulkan 环境下执行量化卷积操作

const auto out_vulkan_deq =
    at::native::vulkan::ops::dequantize(output_vulkan);  // 对 Vulkan 张量进行反量化操作

auto output_for_dequantized_vulkan =
    vulkan_to_cpu(out_vulkan_deq, shape_match);  // 将 Vulkan 张量转换为 CPU 张量

float rtol = 0;  // 设置相对容差 rtol
float atol = 1.5;  // 设置绝对容差 atol
const auto check =
    at::allclose(output_cpu, output_for_dequantized_vulkan, rtol, atol);  // 检查两个张量的所有元素是否在容差范围内相等

if (!check) {
  std::cout << "Max Diff allowed: " << rtol << std::endl;  // 如果检查未通过，输出容差的最大允许差异
}

ASSERT_TRUE(check);  // 使用断言确保检查通过
}

// 在 VulkanAPITest 测试套件中执行 conv2d 测试用例
TEST_F(VulkanAPITest, conv2d) {
  // 调用 test_conv2d 函数进行非混合精度的 conv2d 测试
  test_conv2d(false);
  // 调用 test_conv2d 函数进行混合精度的 conv2d 测试
  test_conv2d(true);
}

// 在 VulkanAPITest 测试套件中执行 conv2d_pw 测试用例
TEST_F(VulkanAPITest, conv2d_pw) {
  // 定义卷积的参数：组数、步幅、填充和膨胀
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 1};
  constexpr std::array<int64_t, 2u> padding{0, 0};
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  // 定义输入张量的结构体，包含批次、通道数、宽度和高度
  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    // 返回输入张量的大小数组
    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{1, 17, 127, 397};

  // 定义权重张量的结构体，包含输出通道数、输入通道数、宽度和高度
  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    // 返回权重张量的大小数组
    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{29, input.channels, 1, 1};

  // 设置输入张量和权重张量的随机范围和数据类型
  float r1 = 0.1;
  float r2 = 0.7;
  const auto input_cpu = (r1 - r2) *
          at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto weights_cpu = (r1 - r2) *
          at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto bias_cpu = (r1 - r2) *
          at::rand({weights.output_channels},
                   at::device(at::kCPU).dtype(at::kFloat)) +
      r2;

  // 设置权重和偏置量化的参数：缩放因子和零点
  const double w_scale = 0.1;
  const int w_zero_point = 10;

  const double b_scale = 0.1;
  const int b_zero_point = 10;

  // 对权重和偏置进行张量量化
  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  // 使用 conv2d 函数计算 CPU 上的输出张量
  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  // 设置 Vulkan 加速计算的量化参数：缩放因子和零点
  const double scale = 0.10;
  const int zero_point = 10;

  // 生成与输入张量形状相匹配的随机张量并量化为 Vulkan 张量
  const auto shape_match =
      at::rand({1, 29, 127, 397}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = input_cpu.vulkan();
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  // 设置 Vulkan 加速计算的量化参数：缩放因子和零点
  const double scale2 = 0.15;
  const int zero_point2 = 15;

  // 使用 Vulkan 进行量化卷积计算
  const auto output_vulkan = at::native::vulkan::ops::quantized_conv2d(
      out_vulkan,
      weight_q,
      bias_q,
      stride,
      padding,
      dilation,
      groups,
      scale2,
      zero_point2);

  // 对 Vulkan 输出进行反量化
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(output_vulkan);

  // 将 Vulkan 输出转换为 CPU 张量
  auto output_for_dequantized_vulkan =
      vulkan_to_cpu(out_vulkan_deq, shape_match);

  // 定义误差容忍度参数
  float rtol = 0;
  float atol = 1.5;

  // 检查 CPU 和 Vulkan 输出张量之间的相似性
  const auto check =
      at::allclose(output_cpu, output_for_dequantized_vulkan, rtol, atol);

  // 如果检查未通过，则打印最大允许误差
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}
// 在 VulkanAPITest 测试固件中定义了一个名为 conv2d_dw 的测试函数
TEST_F(VulkanAPITest, conv2d_dw) {
  // 定义了常量 groups，表示卷积操作的分组数
  constexpr int64_t groups = 7;
  // 定义了常量 stride，表示卷积操作的步长
  constexpr std::array<int64_t, 2u> stride{2, 3};
  // 定义了常量 padding，表示卷积操作的填充大小
  constexpr std::array<int64_t, 2u> padding{0, 4};
  // 定义了常量 dilation，表示卷积操作的扩展大小
  constexpr std::array<int64_t, 2u> dilation{3, 1};

  // 定义了一个结构体 input，描述了输入张量的大小和特征
  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    // 返回一个包含输入张量维度信息的数组
    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{1, groups, 137, 199};

  // 定义了一个结构体 weights，描述了卷积核张量的大小和特征
  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    // 返回一个包含卷积核张量维度信息的数组
    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{groups, 1, 17, 7};

  // 初始化输入张量的随机值，用于 CPU 计算
  float r1 = 0;
  float r2 = 0.2;
  const auto input_cpu = (r1 - r2) *
          at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;

  // 初始化卷积核张量的随机值，用于 CPU 计算
  const auto weights_cpu = (r1 - r2) *
          at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;

  // 初始化偏置值的随机值，用于 CPU 计算
  const auto bias_cpu = (r1 - r2) *
          at::rand({weights.output_channels},
                   at::device(at::kCPU).dtype(at::kFloat)) +
      r2;

  // 定义权重的量化参数
  const double w_scale = 0.1;
  const int w_zero_point = 10;

  // 定义偏置的量化参数
  const double b_scale = 0.1;
  const int b_zero_point = 10;

  // 对卷积核张量进行量化
  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);

  // 对偏置值进行量化
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  // 使用 CPU 计算执行卷积操作
  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  // 定义 Vulkan 加速量化的参数
  const double scale = 0.10;
  const int zero_point = 10;

  // 在 Vulkan 上量化输入张量
  const auto shape_match =
      at::rand({1, 7, 45, 67}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = input_cpu.vulkan();
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  // 定义 Vulkan 加速卷积操作的量化参数
  const double scale2 = 0.15;
  const int zero_point2 = 15;

  // 在 Vulkan 上执行量化卷积操作
  const auto output_vulkan = at::native::vulkan::ops::quantized_conv2d(
      out_vulkan,
      weight_q,
      bias_q,
      stride,
      padding,
      dilation,
      groups,
      scale2,
      zero_point2);

  // 在 Vulkan 上反量化输出结果
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(output_vulkan);

  // 将 Vulkan 输出结果转换为 CPU 结果
  auto output_for_dequantized_vulkan =
      vulkan_to_cpu(out_vulkan_deq, shape_match);

  // 定义相对误差和绝对误差的阈值
  float rtol = 0;
  float atol = 1;

  // 检查 CPU 计算结果和 Vulkan 计算结果是否在允许的误差范围内
  const auto check =
      at::allclose(output_cpu, output_for_dequantized_vulkan, rtol, atol);

  // 如果检查不通过，则输出最大允许的误差值
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查通过
  ASSERT_TRUE(check);
}
    // 定义一个函数，该函数接受输入形状、权重形状、偏置形状、步长、填充、输出填充、扩张、分组参数
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
      // 进入推断模式
      c10::InferenceMode mode;
    
      // 创建一个指定形状和数据类型的随机张量作为输入
      const at::Tensor input =
          at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
      // 创建一个指定形状和数据类型的随机张量作为权重
      const at::Tensor weight =
          at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
      // 创建一个指定形状和数据类型的随机张量作为偏置
      const at::Tensor bias =
          at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));
    
      // 计算输入的量化参数，返回输入的量化比例和零点
      const auto input_quant_params =
          compute_quant_params(input, c10::ScalarType::QUInt8);
      double input_scale = std::get<0>(input_quant_params);
      input_scale = safe_downcast<float>(input_scale);
      int32_t input_zero_point = std::get<1>(input_quant_params);
      // 对输入进行量化为8位无符号整数
      auto input_cpu_q = at::quantize_per_tensor(
          input, input_scale, input_zero_point, c10::ScalarType::QUInt8);
    
      // 计算权重的量化参数，返回权重的量化比例和零点
      const auto weight_quant_params = compute_quant_params(weight, w_dtype);
      double weight_scale = std::get<0>(weight_quant_params);
      weight_scale = safe_downcast<float>(weight_scale);
      int32_t weight_zero_point = std::get<1>(weight_quant_params);
      // 对权重进行量化，使用指定的数据类型
      auto weight_cpu_q =
          at::quantize_per_tensor(weight, weight_scale, weight_zero_point, w_dtype);
    
      // 生成一个随机的输出量化比例
      double out_scale = produce_random_scale();
      out_scale = safe_downcast<float>(out_scale);
      // 生成一个随机的输出零点，数据类型为8位无符号整数
      int out_zero_point = produce_random_zero_point(c10::ScalarType::QUInt8);
    
      at::Tensor bias_cpu_q;
      // 如果偏置的数据类型不是浮点数类型
      if (bias_dtype != c10::ScalarType::Float) {
        // 计算偏置的量化参数，返回偏置的量化比例和零点
        const auto bias_quant_params = compute_quant_params(bias, bias_dtype);
        double bias_scale = std::get<0>(weight_quant_params);
        bias_scale = safe_downcast<float>(bias_scale);
        int32_t bias_zero_point = std::get<1>(bias_quant_params);
        // 对偏置进行量化，使用指定的数据类型
        bias_cpu_q =
            at::quantize_per_tensor(bias, bias_scale, bias_zero_point, bias_dtype);
      } else {
        // 如果偏置数据类型是浮点数类型，则不进行量化处理
        // ...
  // 将 bias 赋值给 bias_cpu_q，用于 CPU 端的偏置
  bias_cpu_q = bias;
}

// 调用指定名称的操作，生成量化卷积转置层的预打包对象
auto pack = callOpByName(
    "quantized::conv_transpose2d_prepack",
    "",
    weight_cpu_q,  // 权重数据（CPU 端）
    bias_cpu_q,    // 偏置数据（CPU 端）
    stride,        // 步长
    padding,       // 填充
    output_padding, // 输出填充
    dilation,      // 膨胀
    groups);       // 分组数

// 调用指定名称的操作，执行量化卷积转置操作
auto out_cpu_quant = callOpByName(
    "quantized::conv_transpose2d",
    "",
    input_cpu_q,   // 输入数据（CPU 端，量化）
    pack[0],        // 预打包的卷积转置对象
    out_scale,      // 输出的缩放因子
    out_zero_point  // 输出的零点偏移量
);

// 对输出进行反量化，得到 CPU 端的输出张量
const at::Tensor out_cpu = at::dequantize(out_cpu_quant[0].toTensor());

// vulkan
// 创建 Vulkan 端量化卷积上下文的预打包对象
const auto prepack_vulkan = callOpByName(
    "vulkan_prepack::create_qtconv2d_context",
    "",
    weight_cpu_q,    // 权重数据（CPU 端）
    bias_cpu_q,      // 偏置数据（CPU 端）
    stride,          // 步长
    padding,         // 填充
    output_padding,  // 输出填充
    dilation,        // 膨胀
    groups,          // 分组数
    c10::nullopt,    // 可选参数为空
    c10::nullopt     // 可选参数为空
);

// 对输入进行 Vulkan 端的量化
const auto input_vk_q = at::quantize_per_tensor(
    input.vulkan(),  // Vulkan 端输入数据
    input_scale,     // 输入的缩放因子
    input_zero_point, // 输入的零点偏移量
    c10::ScalarType::QUInt8  // 数据类型为 8 位无符号整数
);

// 在 Vulkan 端执行量化卷积操作
auto vulkan_output = callOpByName(
    "vulkan_prepack::run_qconv2d_context",
    "",
    input_vk_q,        // Vulkan 端量化输入
    out_scale,         // 输出的缩放因子
    out_zero_point,    // 输出的零点偏移量
    prepack_vulkan[0]  // Vulkan 端预打包对象
);

// 对 Vulkan 端的输出进行反量化
const auto out_vk_dequant = at::dequantize(vulkan_output[0].toTensor());

// 将 Vulkan 端的输出数据移到 CPU 端
const auto out_vk_cpu = out_vk_dequant.cpu();

// 检查 CPU 端和 Vulkan 端的输出是否几乎相等
const auto check = almostEqual(out_cpu, out_vk_cpu, out_scale);

// 如果检查不通过，则显示相对误差
if (!check) {
  showRtol(out_cpu, out_vk_cpu);
}

// 断言检查结果为真
ASSERT_TRUE(check);
TEST_F(VulkanAPITest, conv_tranpose2d_quantized_int8_float) {
  // 调用测试函数，验证量化整数8位与浮点数的转置卷积操作
  test_quantized_conv_transpose2d(
      {1, 3, 2, 2}, // 输入张量形状
      {3, 3, 2, 2}, // 权重张量形状
      {3}, // 偏置张量形状
      c10::ScalarType::QInt8, // 权重量化数据类型
      c10::ScalarType::Float, // 偏置量化数据类型
      {1, 2}, // 步幅
      {1, 0}, // 填充
      {0, 1}, // 输出填充
      {1, 1}, // 膨胀
      1); // 分组

  test_quantized_conv_transpose2d(
      {1, 55, 7, 19}, // 输入张量形状
      {55, 47, 2, 3}, // 权重张量形状
      {47}, // 偏置张量形状
      c10::ScalarType::QInt8, // 权重量化数据类型
      c10::ScalarType::Float, // 偏置量化数据类型
      {1, 2}, // 步幅
      {1, 0}, // 填充
      {0, 1}, // 输出填充
      {1, 1}, // 膨胀
      1); // 分组
}

TEST_F(VulkanAPITest, quantized_sub) {
  float r1 = 4.0;
  float r2 = 7.0;

  float r3 = 2.0;
  float r4 = 5.0;
  
  // 在 CPU 上生成随机张量，并进行算术运算
  const auto in_cpu = (r1 - r2) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  // 将 CPU 上的张量转换到 Vulkan 设备上
  const auto in_vulkan = in_cpu.vulkan();
  
  // 在 CPU 上生成另一个随机张量，并进行算术运算
  const auto in_cpu2 = (r3 - r4) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r4;
  // 将 CPU 上的张量转换到 Vulkan 设备上
  const auto in_vulkan2 = in_cpu2.vulkan();

  // 设置量化参数
  const double scale = 0.1;
  const int zero_point = 10;

  // 对 CPU 上的张量进行整体量化为8位无符号整数
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  
  // 对 Vulkan 设备上的张量进行整体量化为8位无符号整数
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  // 在 CPU 上执行张量减法操作
  const auto reg_subtracted_tensors = at::sub(in_cpu, in_cpu2);

  // 设置另一组量化参数
  const double scale3 = 0.15;
  const int zero_point3 = 15;
  
  // 在 Vulkan 设备上执行量化整数8位张量的减法操作
  const auto vulk_subtracted_tensors = at::native::vulkan::ops::quantized_sub(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  // 对 Vulkan 设备上的张量执行反量化操作
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_subtracted_tensors);
  
  // 将 Vulkan 设备上的张量转换回 CPU 上，并进行比较
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  // 设置比较精度
  float rtol = 0;
  float atol = 0.5;
  
  // 检查两个张量是否在指定精度下相似
  const auto check = at::allclose(
      reg_subtracted_tensors, output_for_dequantized_vulkan, rtol, atol);

  // 如果检查不通过，则输出最大允许差异
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}
// 在 VulkanAPI 测试中的量化乘法功能
TEST_F(VulkanAPITest, quantized_mul) {
  // 创建大小为 [2, 13, 32, 27] 的随机浮点数张量在 CPU 上，范围为 [0, 6)
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  // 创建第二个相同大小的随机浮点数张量在 CPU 上，范围为 [0, 6)
  const auto in_cpu2 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  // 将第二个 CPU 张量转换为 Vulkan 张量
  const auto in_vulkan2 = in_cpu2.vulkan();

  // 设置量化参数
  const double scale = 0.1;
  const int zero_point = 10;

  // 在 CPU 上对输入张量进行量化
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  // 在 Vulkan 上对输入张量进行量化
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  // 设置第二组量化参数
  const double scale3 = 0.15;
  const int zero_point3 = 15;
  // 在 CPU 上执行量化乘法操作
  const auto reg_mul_tensors = callOpByName(
      "quantized::mul", "", out_cpu, out_cpu2, scale3, zero_point3);
  // 在 Vulkan 上执行量化乘法操作
  const auto vulk_mul_tensors = at::native::vulkan::ops::quantized_mul(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  // 将 Vulkan 输出张量反量化
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_mul_tensors);
  // 将 Vulkan 输出张量转换回 CPU 张量
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  // 设置误差容限
  float rtol = 0;
  float atol = 1.5;
  // 检查两组张量是否在给定容限内近似相等
  const auto check = at::allclose(
      at::dequantize(reg_mul_tensors[0].toTensor()),
      output_for_dequantized_vulkan,
      rtol,
      atol);

  // 如果检查未通过，则输出容许的最大差异
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 断言检查结果为真
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, quantized_div) {
  // 定义测试用例中的浮点数变量
  float r1 = 2.0;
  float r2 = 3.5;

  float r3 = 4.0;
  float r4 = 5.5;
  
  // 在 CPU 上生成随机数张量，并执行 Vulkan 的转换操作
  const auto in_cpu = (r1 - r2) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto in_vulkan = in_cpu.vulkan();
  
  // 在 CPU 上生成另一组随机数张量，并执行 Vulkan 的转换操作
  const auto in_cpu2 = (r3 - r4) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r4;
  const auto in_vulkan2 = in_cpu2.vulkan();

  // 定义量化所需的参数
  const double scale = 0.1;
  const int zero_point = 10;

  // 使用 CPU 上的量化函数对输入进行量化操作
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);

  // 使用 Vulkan 的量化函数对输入进行量化操作
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  // 使用 CPU 上的除法运算函数对两组输入张量进行除法运算
  const auto reg_div_tensors = at::div(in_cpu, in_cpu2);

  // 定义第二组量化所需的参数
  const double scale3 = 0.15;
  const int zero_point3 = 15;

  // 使用 Vulkan 的量化除法运算函数对量化后的输入进行除法运算
  const auto vulk_div_tensors = at::native::vulkan::ops::quantized_div(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  // 对 Vulkan 输出结果进行反量化操作
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_div_tensors);
  
  // 将反量化的 Vulkan 输出转换回 CPU 张量
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  // 定义用于比较两个张量是否接近的容差参数
  float rtol = 0;
  float atol = 1;

  // 检查两个张量是否在指定的容差范围内接近
  const auto check =
      at::allclose(reg_div_tensors, output_for_dequantized_vulkan, rtol, atol);

  // 如果检查不通过，则输出最大容差
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 使用断言验证检查结果
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_upsample_nearest2d) {
  // 在 CPU 上生成随机输入张量
  const auto in_cpu =
      at::rand({2, 13, 12, 27}, at::TensorOptions(at::kCPU).dtype(at::kFloat));

  // 在 CPU 上执行最近邻上采样操作
  const auto out_cpu = at::upsample_nearest2d(in_cpu, {4, 6}, 1, 1);

  // 定义量化所需的参数
  const double scale = 0.1;
  const int zero_point = 10;

  // 将 CPU 上的输入张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  
  // 使用 Vulkan 的量化函数对输入张量进行量化操作
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  
  // 在 Vulkan 上执行最近邻上采样操作
  const auto upsample_vulkan = at::upsample_nearest2d(out_vulkan, {4, 6}, 1, 1);

  // 在 CPU 上生成另一组随机输入张量
  const auto in_cpu2 =
      at::rand({2, 13, 4, 6}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  
  // 将 Vulkan 输出张量进行反量化操作
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(upsample_vulkan);
  
  // 将反量化的 Vulkan 输出转换回 CPU 张量
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  // 定义用于比较两个张量是否接近的容差参数
  float rtol = 0;
  float atol = 1;

  // 检查 CPU 上的最近邻上采样结果与反量化的 Vulkan 输出是否接近
  const auto check =
      at::allclose(out_cpu, output_for_dequantized_vulkan, rtol, atol);

  // 如果检查不通过，则输出最大容差
  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  // 使用断言验证检查结果
  ASSERT_TRUE(check);
}

std::tuple<double, double, int, int> produce_inputs_for_binary_op(
    const bool compute_quantization_params,
    const bool random_quantization_params,
    const char* op_name,
    const at::IntArrayRef input1_shape,
    const at::IntArrayRef input2_shape,
    double in1_scale,
    double in2_scale,
    double in1_zp,
    double in2_zp,
    const std::tuple<double, double, int, int>& inputs) {
  return {};
}
    int in1_zero_point,               // 第一个输入张量的零点
    int in2_zero_point,               // 第二个输入张量的零点
    at::Tensor& input1_cpu,           // 第一个CPU张量
    at::Tensor& input1_cpu_q,         // 第一个CPU量化张量
    at::Tensor& input1_cpu_deq,       // 第一个CPU反量化张量
    at::Tensor& input1_vk,            // 第一个Vulkan张量
    at::Tensor& input1_vk_q,          // 第一个Vulkan量化张量
    at::Tensor& input1_vk_deq,        // 第一个Vulkan反量化张量
    at::Tensor& input1_vk_deq_cpu,    // 第一个Vulkan反量化后的CPU张量
    at::Tensor& input2_cpu,           // 第二个CPU张量
    at::Tensor& input2_cpu_q,         // 第二个CPU量化张量
    at::Tensor& input2_cpu_deq,       // 第二个CPU反量化张量
    at::Tensor& input2_vk,            // 第二个Vulkan张量
    at::Tensor& input2_vk_q,          // 第二个Vulkan量化张量
    at::Tensor& input2_vk_deq,        // 第二个Vulkan反量化张量
    at::Tensor& input2_vk_deq_cpu) {  // 第二个Vulkan反量化后的CPU张量
  int num_attempts = 5;  // 尝试生成相同输入张量的最大次数
  // 为了确保我们从数值上开始相同的输入张量（CPU vs Vulkan），允许在随机生成输入时进行多次尝试。
  // 如果CPU量化张量和Vulkan量化张量不相同（可能由于四舍五入和精度差异导致差异为1），我们会再试一次。
  for (int i = 0; i < num_attempts; i += 1) {
    // 生成随机输入
    input1_cpu = produce_random_tensor(input1_shape);  // 生成随机CPU张量
    input2_cpu = produce_random_tensor(input2_shape);  // 生成随机CPU张量

    if (compute_quantization_params) {
      // 计算输入的适当缩放因子和零点
      const auto in1_quant_params = compute_quant_params(input1_cpu);
      in1_scale = std::get<0>(in1_quant_params);
      in1_zero_point = std::get<1>(in1_quant_params);

      const auto in2_quant_params = compute_quant_params(input2_cpu);
      in2_scale = std::get<0>(in2_quant_params);
      in2_zero_point = std::get<1>(in2_quant_params);
    } else if (random_quantization_params) {
      // 为输入生成随机缩放因子和零点
      in1_scale = produce_random_scale();  // 生成随机缩放因子
      in1_zero_point = produce_random_zero_point(c10::ScalarType::QUInt8);  // 生成随机零点

      in2_scale = produce_random_scale();  // 生成随机缩放因子
      in2_zero_point = produce_random_zero_point(c10::ScalarType::QUInt8);  // 生成随机零点
    }

    // 为了避免除以零，我们这样做
    if (strcmp(op_name, "quantized::div") == 0) {
      // 如果我们允许除数的随机缩放和零点，可能会导致除以0的情况。
      if (random_quantization_params) {
        const auto in2_quant_params = compute_quant_params(input2_cpu);
        in2_scale = std::get<0>(in2_quant_params);
        in2_zero_point = std::get<1>(in2_quant_params);
      }

      const auto non_zero_sign =
          input2_cpu.sign() - input2_cpu.sign().abs() + 1;
      // 如果值是非负，则non_zero_sign = 1，如果是负数，则non_zero_sign = -1
      input2_cpu = input2_cpu + in2_scale * non_zero_sign;
      // 这将强制 abs(input2_cpu) >= in2_scale，这意味着第二个输入的量化值中没有一个会等于零点。
    }

    // 对CPU输入进行量化
    input1_cpu_q = at::quantize_per_tensor(
        input1_cpu, in1_scale, in1_zero_point, c10::ScalarType::QUInt8);
    input2_cpu_q = at::quantize_per_tensor(
        input2_cpu, in2_scale, in2_zero_point, c10::ScalarType::QUInt8);

    // 反量化量化的CPU输入
    // 对第一个输入进行反量化操作
    input1_cpu_deq = at::dequantize(input1_cpu_q);
    // 对第二个输入进行反量化操作
    input2_cpu_deq = at::dequantize(input2_cpu_q);

    // 将第一个输入转换为 Vulkan 张量
    input1_vk = input1_cpu.vulkan();
    // 对 Vulkan 张量进行量化操作，使用给定的缩放因子和零点，转换为无符号8位整数
    input1_vk_q = at::quantize_per_tensor(
        input1_vk, in1_scale, in1_zero_point, c10::ScalarType::QUInt8);
    // 将第二个输入转换为 Vulkan 张量
    input2_vk = input2_cpu.vulkan();
    // 对 Vulkan 张量进行量化操作，使用给定的缩放因子和零点，转换为无符号8位整数
    input2_vk_q = at::quantize_per_tensor(
        input2_vk, in2_scale, in2_zero_point, c10::ScalarType::QUInt8);

    // 对量化后的 Vulkan 输入进行反量化操作
    input1_vk_deq = at::dequantize(input1_vk_q);
    input2_vk_deq = at::dequantize(input2_vk_q);

    // 将反量化后的 Vulkan 张量转移到 CPU
    input1_vk_deq_cpu = input1_vk_deq.cpu();
    input2_vk_deq_cpu = input2_vk_deq.cpu();

    // 计算输入1和输入2之间的最大差异，并取其绝对值
    const float input1_dif =
        at::abs(input1_cpu_deq - input1_vk_deq_cpu).max().item<float>();
    const float input2_dif =
        at::abs(input2_cpu_deq - input2_vk_deq_cpu).max().item<float>();
    
    // 检查差异是否小于指定的阈值，并且小于各自的缩放因子的一半
    if (input1_dif < 1e-5 && input2_dif < 1e-5 && input1_dif < in1_scale / 2 &&
        input2_dif < in2_scale / 2) {
      // 如果满足条件，则跳出循环
      break;
    }

  }

  // 返回量化所需的参数：缩放因子和零点
  return {in1_scale, in2_scale, in1_zero_point, in2_zero_point};
}

// 应用于 CPU 的量化二元操作函数
at::Tensor apply_cpu_quantized_binary_op(
    const char* op_name,            // 操作名称，用于选择要执行的操作
    at::Tensor input1_cpu_deq,      // 输入张量1，CPU 上的量化表示
    at::Tensor input2_cpu_deq) {    // 输入张量2，CPU 上的量化表示
  // 根据操作名称执行相应的张量操作：加法、减法、乘法、除法
  if (strcmp(op_name, "quantized::add") == 0) {
    return at::add(input1_cpu_deq, input2_cpu_deq);   // 执行加法操作
  } else if (strcmp(op_name, "quantized::sub") == 0) {
    return at::sub(input1_cpu_deq, input2_cpu_deq);   // 执行减法操作
  } else if (strcmp(op_name, "quantized::mul") == 0) {
    return at::mul(input1_cpu_deq, input2_cpu_deq);   // 执行乘法操作
  } else if (strcmp(op_name, "quantized::div") == 0) {
    return at::div(input1_cpu_deq, input2_cpu_deq);   // 执行除法操作
  } else {
    TORCH_CHECK(false, "Invalid op");  // 若操作名称无效，则抛出错误
  }
}

// 应用于 Vulkan 的量化二元操作函数
at::Tensor apply_vulkan_quantized_binary_op(
    const char* op_name,            // 操作名称，用于选择要执行的操作
    at::Tensor input1_vk_q,         // 输入张量1，Vulkan 上的量化表示
    at::Tensor input2_vk_q,         // 输入张量2，Vulkan 上的量化表示
    double out_scale,               // 输出张量的量化比例
    int out_zero_point) {           // 输出张量的零点
  // 根据操作名称执行相应的 Vulkan 张量操作：加法、减法、乘法、除法
  if (strcmp(op_name, "quantized::add") == 0) {
    return at::native::vulkan::ops::quantized_add(
        input1_vk_q, input2_vk_q, out_scale, out_zero_point);   // 执行 Vulkan 加法操作
  } else if (strcmp(op_name, "quantized::sub") == 0) {
    return at::native::vulkan::ops::quantized_sub(
        input1_vk_q, input2_vk_q, out_scale, out_zero_point);   // 执行 Vulkan 减法操作
  } else if (strcmp(op_name, "quantized::mul") == 0) {
    return at::native::vulkan::ops::quantized_mul(
        input1_vk_q, input2_vk_q, out_scale, out_zero_point);   // 执行 Vulkan 乘法操作
  } else if (strcmp(op_name, "quantized::div") == 0) {
    return at::native::vulkan::ops::quantized_div(
        input1_vk_q, input2_vk_q, out_scale, out_zero_point);   // 执行 Vulkan 除法操作
  } else {
    TORCH_CHECK(false, "Invalid op");  // 若操作名称无效，则抛出错误
  }
}

// 测试量化二元操作的函数
void test_quantized_binary_op(
    const bool compute_quantization_params,      // 是否计算量化参数
    const bool random_quantization_params,       // 是否使用随机量化参数
    const char* op_name,                        // 操作名称，用于选择要执行的操作
    const at::IntArrayRef input1_shape,          // 输入张量1 的形状
    const at::IntArrayRef input2_shape,          // 输入张量2 的形状
    double in1_scale_default = 0.103,            // 输入张量1 的默认比例
    double in2_scale_default = 0.171,            // 输入张量2 的默认比例
    double out_scale_default = 0.139,            // 输出张量的默认比例
    int in1_zero_point_default = 11,             // 输入张量1 的默认零点
    int in2_zero_point_default = 9,              // 输入张量2 的默认零点
    // 设置默认的输出零点值为17
    int out_zero_point_default = 17) {
    // 生成输入张量
    at::Tensor input1_cpu, input1_cpu_q, input1_cpu_deq;
    at::Tensor input1_vk, input1_vk_q, input1_vk_deq, input1_vk_deq_cpu;
    at::Tensor input2_cpu, input2_cpu_q, input2_cpu_deq;
    at::Tensor input2_vk, input2_vk_q, input2_vk_deq, input2_vk_deq_cpu;
    
    // 调用函数产生用于二元操作的输入参数
    auto input_params = produce_inputs_for_binary_op(
        compute_quantization_params,
        random_quantization_params,
        op_name,
        input1_shape,
        input2_shape,
        in1_scale_default,
        in2_scale_default,
        in1_zero_point_default,
        in2_zero_point_default,
        input1_cpu,
        input1_cpu_q,
        input1_cpu_deq,
        input1_vk,
        input1_vk_q,
        input1_vk_deq,
        input1_vk_deq_cpu,
        input2_cpu,
        input2_cpu_q,
        input2_cpu_deq,
        input2_vk,
        input2_vk_q,
        input2_vk_deq,
        input2_vk_deq_cpu);
    
    // 从输入参数中获取输入1和输入2的量化参数
    double in1_scale = std::get<0>(input_params);
    double in2_scale = std::get<1>(input_params);
    int in1_zero_point = std::get<2>(input_params);
    int in2_zero_point = std::get<3>(input_params);
    
    // 设置输出的量化比例和零点
    double out_scale = out_scale_default;
    int out_zero_point = out_zero_point_default;
    
    // 在去量化的CPU张量上应用操作
    at::Tensor output_cpu =
        apply_cpu_quantized_binary_op(op_name, input1_cpu_deq, input2_cpu_deq);
    
    // 如果需要计算量化参数或随机量化参数
    if (compute_quantization_params || random_quantization_params) {
      // 计算输出的合适比例和零点
      const auto out_quant_params = compute_quant_params(output_cpu);
      out_scale = std::get<0>(out_quant_params);
      out_zero_point = std::get<1>(out_quant_params);
    }
    
    // 对CPU输出进行量化和去量化
    const auto output_cpu_q = at::quantize_per_tensor(
        output_cpu, out_scale, out_zero_point, c10::ScalarType::QUInt8);
    const auto output_cpu_deq = at::dequantize(output_cpu_q);
    
    // Vulkan量化输出
    at::Tensor output_vk_q = apply_vulkan_quantized_binary_op(
        op_name, input1_vk_q, input2_vk_q, out_scale, out_zero_point);
    
    // 对Vulkan量化输出进行去量化
    const auto output_vk_deq = at::dequantize(output_vk_q);
    const auto output_vk_deq_cpu = output_vk_deq.cpu();
    
    // 检查输出是否接近
    const float tolerance =
        (compute_quantization_params || random_quantization_params)
        ? safe_downcast<float>(out_scale)
        : 0;
    const auto check = almostEqual(output_cpu_deq, output_vk_deq_cpu, tolerance);
    
    // 如果检查失败，输出错误信息
    if (!check) {
      const auto vk_q_error =
          at::abs(output_vk_deq_cpu - output_cpu_deq).max().item<float>();
      std::cout << "Binary op " << op_name
                << " failed with inputs: " << std::endl;
      std::cout << "input1: shape " << input1_shape << " scale " << in1_scale
                << " and zero point " << in1_zero_point << std::endl;
      std::cout << "input2: shape " << input2_shape << " scale " << in2_scale
                << " and zero point " << in2_zero_point << std::endl;
      std::cout << "output scale " << out_scale << " and zero point "
                << out_zero_point << std::endl;
    // 输出错误消息 "error: " 后跟 vk_q_error 的值，然后换行
    std::cout << "error: " << vk_q_error << std::endl;
    // 使用 ASSERT_TRUE 宏检查 check 变量的值是否为真，如果为假，则断言失败
    ASSERT_TRUE(check);
void test_quantized_conv2d(
    // 指示是否预打包权重以提高卷积性能
    const bool prepacking,
    // 指示是否计算量化参数
    const bool compute_quantization_params,
    // 指示是否使用随机量化参数
    const bool random_quantization_params,
    // 输入张量的形状
    const at::IntArrayRef input_shape,
    // 权重张量的形状
    const at::IntArrayRef weight_shape,
    // 偏置张量的形状
    const at::IntArrayRef bias_shape,
    // 权重张量的数据类型
    const c10::ScalarType w_dtype,
    // 偏置张量的数据类型
    const c10::ScalarType b_dtype,
    // 卷积操作的步长
    std::vector<int64_t> stride,
    // 卷积操作的填充
    std::vector<int64_t> padding,
    // 卷积操作的扩张
    std::vector<int64_t> dilation,
    // 卷积操作的分组数
    int64_t groups,
    // 输入张量的量化缩放因子，默认为0.13
    double in_scale = 0.13,
    // 权重张量的量化缩放因子，默认为0.29
    double w_scale = 0.29,
    // 偏置张量的量化缩放因子，默认为0.19
    double b_scale = 0.19,
    // 输出张量的量化缩放因子，默认为0.15
    double out_scale = 0.15,
    // 输入张量的零点值，默认为11
    int in_zero_point = 11,
    // 权重张量的零点值，默认为19
    int w_zero_point = 19,
    // 偏置张量的零点值，默认为27
    int b_zero_point = 27,
    // 设置输出的零点为10
    int out_zero_point = 10;

    // 进入推断模式
    c10::InferenceMode mode;

    // 定义输入数据类型和输出数据类型为无符号8位整数
    const c10::ScalarType in_dtype = c10::ScalarType::QUInt8;
    const c10::ScalarType out_dtype = c10::ScalarType::QUInt8;

    // input cpu
    // 输入的CPU张量
    at::Tensor input_cpu;
    // 输入的CPU张量 -> 量化后
    at::Tensor input_cpu_q;
    // 输入的CPU张量 -> 量化后 -> 反量化
    at::Tensor input_cpu_deq;

    // input vulkan
    // 输入的CPU张量 -> 转换到Vulkan
    at::Tensor input_vk;
    // 输入的CPU张量 -> 转换到Vulkan -> 量化
    at::Tensor input_vk_q;
    // 输入的CPU张量 -> 转换到Vulkan -> 量化 -> 反量化
    at::Tensor input_vk_deq;
    // 输入的CPU张量 -> 转换到Vulkan -> 量化 -> 反量化 -> 转回CPU
    at::Tensor input_vk_deq_cpu;

    // weight cpu
    // 权重的CPU张量
    at::Tensor weight_cpu;
    // 权重的CPU张量 -> 量化
    at::Tensor weight_cpu_q;
    // 权重的CPU张量 -> 量化 -> 反量化
    at::Tensor weight_cpu_deq;

    // bias cpu
    // 偏置的CPU张量
    at::Tensor bias_cpu;
    // 偏置的CPU张量 -> 量化
    at::Tensor bias_cpu_q;
    // 偏置的CPU张量 -> 量化 -> 反量化
    at::Tensor bias_cpu_deq;

    // 当随机生成输入张量时，可能不幸地得到某些条目，使得除以比例时得到像2.50003这样的结果，
    // 它可能被四舍五入为2或3，这取决于精度和舍入方法。
    // 因此，我们生成输入并检查 input_cpu_deq 和 input_vk_deq_cpu 之间的差异。
    // 如果它们不同，我们再次生成它们（最多3次）。
    // 目标是使用在量化后保持相等的输入张量开始。
    int num_attempts = 5;
    for (int i = 0; i < num_attempts; i += 1) {
        // 生成随机的输入、权重和偏置张量
        input_cpu = produce_random_tensor(input_shape, 1.26, 5.97, 0.59);
        weight_cpu = produce_random_tensor(weight_shape, 1.26, 5.97, 0.59);
        bias_cpu = produce_random_tensor(bias_shape, 1.26, 5.97, 0.59);

        if (compute_quantization_params) {
            // 计算输入、权重和偏置的适当比例和零点
            const auto in_quant_params = compute_quant_params(input_cpu, in_dtype);
            in_scale = std::get<0>(in_quant_params);
            in_zero_point = std::get<1>(in_quant_params);

            const auto w_quant_params = compute_quant_params(weight_cpu, w_dtype);
            w_scale = std::get<0>(w_quant_params);
            w_zero_point = std::get<1>(w_quant_params);

            const auto input_max = input_cpu.max().item<float>();
            const auto input_min = input_cpu.min().item<float>();
            const auto input_range = input_max - input_min;

            // 生成偏置张量，范围在输入范围内，用于计算偏置的比例和零点
            bias_cpu = input_range * at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat)) + input_min;
            b_scale = in_scale;
            b_zero_point = in_zero_point;

            // 如果偏置数据类型是QInt32，则使用输入和权重的比例计算偏置的比例和零点
            if (b_dtype == c10::ScalarType::QInt32) {
                b_scale = in_scale * w_scale;
                b_zero_point = 0;
            }
    } else if (random_quantization_params) {
      // 如果需要随机量化参数，则生成随机的输入、权重和偏置的缩放因子和零点
      in_scale = produce_random_scale();  // 生成随机的输入缩放因子
      in_zero_point = produce_random_zero_point(in_dtype);  // 生成随机的输入零点

      w_scale = produce_random_scale();  // 生成随机的权重缩放因子
      w_zero_point = produce_random_zero_point(w_dtype);  // 生成随机的权重零点

      b_scale = produce_random_scale();  // 生成随机的偏置缩放因子
      b_zero_point = produce_random_zero_point(b_dtype);  // 生成随机的偏置零点
    }

    // 对 CPU 输入、权重和偏置进行量化
    input_cpu_q =
        at::quantize_per_tensor(input_cpu, in_scale, in_zero_point, in_dtype);  // 对输入进行量化
    weight_cpu_q =
        at::quantize_per_tensor(weight_cpu, w_scale, w_zero_point, w_dtype);  // 对权重进行量化
    bias_cpu_q =
        at::quantize_per_tensor(bias_cpu, b_scale, b_zero_point, b_dtype);  // 对偏置进行量化

    // 对量化后的 CPU 输入、权重和偏置进行反量化
    input_cpu_deq = at::dequantize(input_cpu_q);  // 反量化输入
    weight_cpu_deq = at::dequantize(weight_cpu_q);  // 反量化权重
    bias_cpu_deq = at::dequantize(bias_cpu_q);  // 反量化偏置

    // 将 Vulkan 输入转换为 Vulkan 张量
    input_vk = input_cpu.vulkan();  // 转换为 Vulkan 张量
    input_vk_q =
        at::quantize_per_tensor(input_vk, in_scale, in_zero_point, in_dtype);  // 对 Vulkan 输入进行量化

    // 对量化后的 Vulkan 输入进行反量化
    input_vk_deq = at::dequantize(input_vk_q);  // 反量化 Vulkan 输入
    input_vk_deq_cpu = input_vk_deq.cpu();  // 将反量化的 Vulkan 输入转回 CPU 张量

    // 计算 CPU 输入和 Vulkan 输入之间的差异
    const float input_dif =
        at::abs(input_cpu_deq - input_vk_deq_cpu).max().item<float>();  // 计算最大差异

    // 如果差异很小，并且小于输入缩放因子的一半，则跳出循环
    if (input_dif < 1e-5 && input_dif < in_scale / 2) {
      break;
    } else {
      std::cout << "input_dif too big: " << input_dif;  // 输出差异过大的信息
      if (i + 1 < num_attempts) {
        std::cout << ". generating input again ..." << std::endl;  // 如果还有尝试次数，输出重新生成输入的信息
      } else {
        std::cout << std::endl;  // 否则，换行
      }
    }
  }

  // 在反量化的 CPU 张量上应用卷积操作
  // 注意：我们在反量化的量化张量上执行卷积操作，确保在相同的数值上进行操作。
  const auto output_cpu = at::conv2d(
      input_cpu_deq,
      weight_cpu_deq,
      bias_cpu_deq,
      stride,
      padding,
      dilation,
      groups);

  if (compute_quantization_params || random_quantization_params) {
    // 计算输出的适当缩放因子和零点
    const auto out_quant_params = compute_quant_params(output_cpu, out_dtype);  // 计算输出的量化参数
    out_scale = std::get<0>(out_quant_params);  // 获取输出的缩放因子
    out_zero_point = std::get<1>(out_quant_params);  // 获取输出的零点
  }

  // 对 CPU 输出进行量化和反量化
  at::Tensor output_cpu_q =
      at::quantize_per_tensor(output_cpu, out_scale, out_zero_point, out_dtype);  // 对输出进行量化
  at::Tensor output_cpu_deq = at::dequantize(output_cpu_q);  // 对量化后的输出进行反量化

  // Vulkan 量化输出
  at::Tensor output_vk_q;

  if (!prepacking) {
    // 使用 Vulkan 进行量化的卷积操作
    output_vk_q = at::native::vulkan::ops::quantized_conv2d(
        input_vk_q,
        weight_cpu_q,
        bias_cpu_q,
        stride,
        padding,
        dilation,
        groups,
        out_scale,
        out_zero_point);
  } else {
    // 通过名称调用 Vulkan 量化的卷积操作
    // 调用函数 `callOpByName` 执行 Vulkan 预打包操作，创建 QConv2d 上下文，返回结果
    const auto prepack_vulkan_call_by_name = callOpByName(
        "vulkan_prepack::create_qconv2d_context",  // 调用的函数名，用于 Vulkan 预打包操作
        "",  // 空字符串作为第二个参数
        weight_cpu_q,  // 权重的量化 Tensor
        bias_cpu_q,  // 偏置的量化 Tensor
        stride,  // 卷积的步长
        padding,  // 卷积的填充
        dilation,  // 卷积的膨胀
        groups,  // 分组卷积的组数
        c10::nullopt,  // 可选参数，这里未指定
        c10::nullopt  // 可选参数，这里未指定
    );
    
    // 调用函数 `callOpByName` 执行 Vulkan 预打包操作，运行 QConv2d 上下文，返回结果
    const auto vulkan_output = callOpByName(
        "vulkan_prepack::run_qconv2d_context",  // 调用的函数名，用于运行 Vulkan 上下文
        "",  // 空字符串作为第二个参数
        input_vk_q,  // Vulkan 输入的量化 Tensor
        out_scale,  // 输出的量化比例
        out_zero_point,  // 输出的零点
        prepack_vulkan_call_by_name[0]  // 前面预打包操作返回的第一个结果作为参数
    );
    
    // 将 Vulkan 输出的量化 Tensor 转换为普通 Tensor，并赋给 output_vk_q
    output_vk_q = vulkan_output[0].toTensor();
    
    // 解量化 Vulkan 输出的量化 Tensor
    const auto output_vk_deq = at::dequantize(output_vk_q);
    
    // 将解量化后的 Tensor 转移到 CPU 上
    const auto output_vk_deq_cpu = output_vk_deq.cpu();
    
    // 计算准确性，检查 Vulkan 输出与 CPU 输出是否几乎相等
    const float tolerance = safe_downcast<float>(out_scale);
    const auto check = almostEqual(output_cpu_deq, output_vk_deq_cpu, tolerance);
    
    // 如果检查不通过，则输出错误信息
    if (!check) {
        const auto vk_q_error = at::abs(output_vk_deq_cpu - output_cpu_deq).max().item<float>();
        std::cout << "Quantized Conv2d failed with: " << std::endl;
        std::cout << "input: shape " << input_shape << " scale " << in_scale
                  << " and zero point " << in_zero_point << std::endl;
        std::cout << "weight: shape " << weight_shape << " scale " << w_scale
                  << " and zero point " << w_zero_point << std::endl;
        std::cout << "bias: shape " << bias_shape << " scale " << b_scale
                  << " and zero point " << b_zero_point << std::endl;
        std::cout << "output scale " << out_scale << " and zero point "
                  << out_zero_point << std::endl;
        std::cout << "error: " << vk_q_error << std::endl;
    }
    
    // 断言检查，确保最终的准确性检查通过
    ASSERT_TRUE(check);
TEST_F(VulkanAPITest, conv2d_quantized_fixed_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ false,             // 是否进行预打包，此处为否
      /* compute params */ false,          // 是否计算参数，此处为否
      /* random params */ false,           // 是否使用随机参数，此处为否
      /* input_shape */ {1, 3, 8, 8},       // 输入数据的形状：1个样本，3个通道，8x8大小
      /* weight_shape */ {1, 3, 3, 3},      // 权重的形状：1个卷积核，3个输入通道，3x3大小
      /* bias_shape */ {1},                // 偏置的形状：1个偏置
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重的数据类型为无符号8位整数
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置的数据类型为无符号8位整数
      /* stride */ {2, 2},                 // 卷积步长为2x2
      /* padding */ {1, 1},                // 填充为1x1
      /* dilation */ {1, 1},               // 空洞卷积系数为1x1
      /* groups */ 1);                     // 卷积组数为1
}

TEST_F(VulkanAPITest, conv2d_quantized_computed_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ false,             // 是否进行预打包，此处为否
      /* compute params */ true,           // 是否计算参数，此处为是
      /* random params */ false,           // 是否使用随机参数，此处为否
      /* input_shape */ {1, 3, 8, 8},       // 输入数据的形状：1个样本，3个通道，8x8大小
      /* weight_shape */ {1, 3, 3, 3},      // 权重的形状：1个卷积核，3个输入通道，3x3大小
      /* bias_shape */ {1},                // 偏置的形状：1个偏置
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重的数据类型为无符号8位整数
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置的数据类型为无符号8位整数
      /* stride */ {2, 2},                 // 卷积步长为2x2
      /* padding */ {1, 1},                // 填充为1x1
      /* dilation */ {1, 1},               // 空洞卷积系数为1x1
      /* groups */ 1);                     // 卷积组数为1
}

TEST_F(VulkanAPITest, conv2d_quantized_random_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ false,             // 是否进行预打包，此处为否
      /* compute params */ false,          // 是否计算参数，此处为否
      /* random params */ true,            // 是否使用随机参数，此处为是
      /* input_shape */ {1, 3, 8, 8},       // 输入数据的形状：1个样本，3个通道，8x8大小
      /* weight_shape */ {1, 3, 3, 3},      // 权重的形状：1个卷积核，3个输入通道，3x3大小
      /* bias_shape */ {1},                // 偏置的形状：1个偏置
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重的数据类型为无符号8位整数
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置的数据类型为无符号8位整数
      /* stride */ {2, 2},                 // 卷积步长为2x2
      /* padding */ {1, 1},                // 填充为1x1
      /* dilation */ {1, 1},               // 空洞卷积系数为1x1
      /* groups */ 1);                     // 卷积组数为1
}

TEST_F(VulkanAPITest, conv2d_quantized_prepack_fixed_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ true,              // 是否进行预打包，此处为是
      /* compute params */ false,          // 是否计算参数，此处为否
      /* random params */ false,           // 是否使用随机参数，此处为否
      /* input_shape */ {1, 3, 8, 8},       // 输入数据的形状：1个样本，3个通道，8x8大小
      /* weight_shape */ {1, 3, 3, 3},      // 权重的形状：1个卷积核，3个输入通道，3x3大小
      /* bias_shape */ {1},                // 偏置的形状：1个偏置
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重的数据类型为无符号8位整数
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置的数据类型为无符号8位整数
      /* stride */ {2, 2},                 // 卷积步长为2x2
      /* padding */ {1, 1},                // 填充为1x1
      /* dilation */ {1, 1},               // 空洞卷积系数为1x1
      /* groups */ 1);                     // 卷积组数为1
}

TEST_F(VulkanAPITest, conv2d_quantized_prepack_computed_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ true,              // 是否进行预打包，此处为是
      /* compute params */ true,           // 是否计算参数，此处为是
      /* random params */ false,           // 是否使用随机参数，此处为否
      /* input_shape */ {1, 3, 8, 8},       // 输入数据的形状：1个样本，3个通道，8x8大小
      /* weight_shape */ {1, 3, 3, 3},      // 权重的形状：1个卷积核，3个输入通道，3x3大小
      /* bias_shape */ {1},                // 偏置的形状：1个偏置
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重的数据类型为无符号8位整数
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置的数据类型为无符号8位整数
      /* stride */ {2, 2},                 // 卷积步长为2x2
      /* padding */ {1, 1},                // 填充为1x1
      /* dilation */ {1, 1},               // 空洞卷积系数为1x1
      /* groups */ 1);                     // 卷积组数为1
}
TEST_F(VulkanAPITest, conv2d_quantized_prepack_random_params_uint8) {
  test_quantized_conv2d(
      /* prepacking? */ true,                    // 是否进行预打包
      /* compute params */ false,                // 是否计算参数
      /* random params */ true,                  // 是否使用随机参数
      /* input_shape */ {1, 3, 8, 8},            // 输入张量形状
      /* weight_shape */ {1, 3, 3, 3},           // 权重张量形状
      /* bias_shape */ {1},                      // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8, // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,  // 偏置数据类型
      /* stride */ {2, 2},                       // 卷积步幅
      /* padding */ {1, 1},                      // 卷积填充
      /* dilation */ {1, 1},                     // 卷积扩展
      /* groups */ 1);                           // 卷积分组
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_fixed_params_uint8) {
  test_quantized_conv2d(
      /* prepacking? */ false,                   // 是否进行预打包
      /* compute params */ false,                // 是否计算参数
      /* random params */ false,                 // 是否使用随机参数
      /* input_shape */ {1, 7, 137, 199},        // 输入张量形状
      /* weight_shape */ {7, 1, 17, 7},          // 权重张量形状
      /* bias_shape */ {7},                      // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8, // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,  // 偏置数据类型
      /* stride */ {2, 3},                       // 卷积步幅
      /* padding */ {0, 4},                      // 卷积填充
      /* dilation */ {3, 1},                     // 卷积扩展
      /* groups */ 7);                           // 卷积分组
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_computed_params_uint8) {
  test_quantized_conv2d(
      /* prepacking? */ false,                   // 是否进行预打包
      /* compute params */ true,                 // 是否计算参数
      /* random params */ false,                 // 是否使用随机参数
      /* input_shape */ {1, 7, 137, 199},        // 输入张量形状
      /* weight_shape */ {7, 1, 17, 7},          // 权重张量形状
      /* bias_shape */ {7},                      // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8, // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,  // 偏置数据类型
      /* stride */ {2, 3},                       // 卷积步幅
      /* padding */ {0, 4},                      // 卷积填充
      /* dilation */ {3, 1},                     // 卷积扩展
      /* groups */ 7);                           // 卷积分组
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_random_params_uint8) {
  test_quantized_conv2d(
      /* prepacking? */ false,                   // 是否进行预打包
      /* compute params */ false,                // 是否计算参数
      /* random params */ true,                  // 是否使用随机参数
      /* input_shape */ {1, 7, 137, 199},        // 输入张量形状
      /* weight_shape */ {7, 1, 17, 7},          // 权重张量形状
      /* bias_shape */ {7},                      // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8, // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,  // 偏置数据类型
      /* stride */ {2, 3},                       // 卷积步幅
      /* padding */ {0, 4},                      // 卷积填充
      /* dilation */ {3, 1},                     // 卷积扩展
      /* groups */ 7);                           // 卷积分组
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_prepack_fixed_params_uint8) {
  test_quantized_conv2d(
      /* prepacking? */ true,                    // 是否进行预打包
      /* compute params */ false,                // 是否计算参数
      /* random params */ false,                 // 是否使用随机参数
      /* input_shape */ {1, 7, 137, 199},        // 输入张量形状
      /* weight_shape */ {7, 1, 17, 7},          // 权重张量形状
      /* bias_shape */ {7},                      // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8, // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,  // 偏置数据类型
      /* stride */ {2, 3},                       // 卷积步幅
      /* padding */ {0, 4},                      // 卷积填充
      /* dilation */ {3, 1},                     // 卷积扩展
      /* groups */ 7);                           // 卷积分组
}
TEST_F(VulkanAPITest, conv2d_dw_quantized_prepack_computed_params_uint8) {
  // 调用函数 test_quantized_conv2d 进行量化深度卷积测试，使用预打包参数和计算参数
  test_quantized_conv2d(
      /* prepacking? */ true,                          // 是否预打包？
      /* compute params */ true,                       // 是否计算参数？
      /* random params */ false,                       // 是否随机参数？
      /* input_shape */ {1, 7, 137, 199},              // 输入张量形状
      /* weight_shape */ {7, 1, 17, 7},                // 权重张量形状
      /* bias_shape */ {7},                           // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8,      // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,        // 偏置数据类型
      /* stride */ {2, 3},                            // 步幅
      /* padding */ {0, 4},                           // 填充
      /* dilation */ {3, 1},                          // 膨胀
      /* groups */ 7);                                // 分组数量
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_prepack_random_params_uint8) {
  // 调用函数 test_quantized_conv2d 进行量化深度卷积测试，使用预打包参数和随机参数
  test_quantized_conv2d(
      /* prepacking? */ true,                          // 是否预打包？
      /* compute params */ false,                      // 是否计算参数？
      /* random params */ true,                        // 是否随机参数？
      /* input_shape */ {1, 7, 137, 199},              // 输入张量形状
      /* weight_shape */ {7, 1, 17, 7},                // 权重张量形状
      /* bias_shape */ {7},                           // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8,      // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,        // 偏置数据类型
      /* stride */ {2, 3},                            // 步幅
      /* padding */ {0, 4},                           // 填充
      /* dilation */ {3, 1},                          // 膨胀
      /* groups */ 7);                                // 分组数量
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_fixed_params_uint8) {
  // 调用函数 test_quantized_conv2d 进行量化点卷积测试，使用固定参数
  test_quantized_conv2d(
      /* prepacking? */ false,                         // 是否预打包？
      /* compute params */ false,                      // 是否计算参数？
      /* random params */ false,                       // 是否随机参数？
      /* input_shape */ {1, 17, 127, 397},             // 输入张量形状
      /* weight_shape */ {29, 17, 1, 1},               // 权重张量形状
      /* bias_shape */ {29},                          // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8,      // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,        // 偏置数据类型
      /* stride */ {1, 1},                            // 步幅
      /* padding */ {0, 0},                           // 填充
      /* dilation */ {1, 1},                          // 膨胀
      /* groups */ 1);                                // 分组数量
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_computed_params_uint8) {
  // 调用函数 test_quantized_conv2d 进行量化点卷积测试，使用计算参数
  test_quantized_conv2d(
      /* prepacking? */ false,                         // 是否预打包？
      /* compute params */ true,                       // 是否计算参数？
      /* random params */ false,                       // 是否随机参数？
      /* input_shape */ {1, 17, 127, 397},             // 输入张量形状
      /* weight_shape */ {29, 17, 1, 1},               // 权重张量形状
      /* bias_shape */ {29},                          // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8,      // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,        // 偏置数据类型
      /* stride */ {1, 1},                            // 步幅
      /* padding */ {0, 0},                           // 填充
      /* dilation */ {1, 1},                          // 膨胀
      /* groups */ 1);                                // 分组数量
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_random_params_uint8) {
  // 调用函数 test_quantized_conv2d 进行量化点卷积测试，使用随机参数
  test_quantized_conv2d(
      /* prepacking? */ false,                         // 是否预打包？
      /* compute params */ false,                      // 是否计算参数？
      /* random params */ true,                        // 是否随机参数？
      /* input_shape */ {1, 17, 127, 397},             // 输入张量形状
      /* weight_shape */ {29, 17, 1, 1},               // 权重张量形状
      /* bias_shape */ {29},                          // 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QUInt8,      // 权重数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,        // 偏置数据类型
      /* stride */ {1, 1},                            // 步幅
      /* padding */ {0, 0},                           // 填充
      /* dilation */ {1, 1},                          // 膨胀
      /* groups */ 1);                                // 分组数量
}
TEST_F(VulkanAPITest, conv2d_pw_quantized_prepack_fixed_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化的深度卷积操作
  test_quantized_conv2d(
      /* prepacking? */ true,                   // 是否进行预打包
      /* compute params */ false,               // 是否计算参数
      /* random params */ false,                // 是否使用随机参数
      /* input_shape */ {1, 17, 127, 397},      // 输入张量的形状
      /* weight_shape */ {29, 17, 1, 1},        // 权重张量的形状
      /* bias_shape */ {29},                    // 偏置张量的形状
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重张量的数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置张量的数据类型
      /* stride */ {1, 1},                      // 卷积操作的步长
      /* padding */ {0, 0},                     // 输入张量的填充
      /* dilation */ {1, 1},                    // 卷积核的扩展率
      /* groups */ 1);                          // 卷积操作的组数
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_prepack_computed_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化的深度卷积操作（计算参数）
  test_quantized_conv2d(
      /* prepacking? */ true,                   // 是否进行预打包
      /* compute params */ true,                // 是否计算参数
      /* random params */ false,                // 是否使用随机参数
      /* input_shape */ {1, 17, 127, 397},      // 输入张量的形状
      /* weight_shape */ {29, 17, 1, 1},        // 权重张量的形状
      /* bias_shape */ {29},                    // 偏置张量的形状
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重张量的数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置张量的数据类型
      /* stride */ {1, 1},                      // 卷积操作的步长
      /* padding */ {0, 0},                     // 输入张量的填充
      /* dilation */ {1, 1},                    // 卷积核的扩展率
      /* groups */ 1);                          // 卷积操作的组数
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_prepack_random_params_uint8) {
  // 调用 test_quantized_conv2d 函数，测试量化的深度卷积操作（随机参数）
  test_quantized_conv2d(
      /* prepacking? */ true,                   // 是否进行预打包
      /* compute params */ false,               // 是否计算参数
      /* random params */ true,                 // 是否使用随机参数
      /* input_shape */ {1, 17, 127, 397},      // 输入张量的形状
      /* weight_shape */ {29, 17, 1, 1},        // 权重张量的形状
      /* bias_shape */ {29},                    // 偏置张量的形状
      /* weight_dtype */ c10::ScalarType::QUInt8,   // 权重张量的数据类型
      /* bias_dtype */ c10::ScalarType::QUInt8,     // 偏置张量的数据类型
      /* stride */ {1, 1},                      // 卷积操作的步长
      /* padding */ {0, 0},                     // 输入张量的填充
      /* dilation */ {1, 1},                    // 卷积核的扩展率
      /* groups */ 1);                          // 卷积操作的组数
}

TEST_F(VulkanAPITest, conv2d_quantized_fixed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数，测试量化的深度卷积操作（固定参数，不进行预打包）
  test_quantized_conv2d(
      /* prepacking? */ false,                  // 是否进行预打包
      /* compute params */ false,               // 是否计算参数
      /* random params */ false,                // 是否使用随机参数
      /* input_shape */ {1, 3, 8, 8},           // 输入张量的形状
      /* weight_shape */ {1, 3, 3, 3},          // 权重张量的形状
      /* bias_shape */ {1},                     // 偏置张量的形状
      /* weight_dtype */ c10::ScalarType::QInt8,    // 权重张量的数据类型
      /* bias_dtype */ c10::ScalarType::QInt32,     // 偏置张量的数据类型
      /* stride */ {2, 2},                     // 卷积操作的步长
      /* padding */ {1, 1},                    // 输入张量的填充
      /* dilation */ {1, 1},                   // 卷积核的扩展率
      /* groups */ 1);                         // 卷积操作的组数
}

TEST_F(VulkanAPITest, conv2d_quantized_computed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数，测试量化的深度卷积操作（计算参数，不进行预打包）
  test_quantized_conv2d(
      /* prepacking? */ false,                  // 是否进行预打包
      /* compute params */ true,                // 是否计算参数
      /* random params */ false,                // 是否使用随机参数
      /* input_shape */ {1, 3, 8, 8},           // 输入张量的形状
      /* weight_shape */ {1, 3, 3, 3},          // 权重张量的形状
      /* bias_shape */ {1},                     // 偏置张量的形状
      /* weight_dtype */ c10::ScalarType::QInt8,    // 权重张量的数据类型
      /* bias_dtype */ c10::ScalarType::QInt32,     // 偏置张量的数据类型
      /* stride */ {2, 2},                     // 卷积操作的步长
      /* padding */ {1, 1},                    // 输入张量的填充
      /* dilation */ {1, 1},                   // 卷积核的扩展率
      /* groups */ 1);                         // 卷积操作的组数
}
TEST_F(VulkanAPITest, conv2d_quantized_random_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数，测试量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ false,  // 不进行预打包
      /* compute params */ false,  // 不计算参数
      /* random params */ true,  // 使用随机参数
      /* input_shape */ {1, 3, 8, 8},  // 输入张量形状为 [1, 3, 8, 8]
      /* weight_shape */ {1, 3, 3, 3},  // 权重张量形状为 [1, 3, 3, 3]
      /* bias_shape */ {1},  // 偏置张量形状为 [1]
      /* weight_dtype */ c10::ScalarType::QInt8,  // 权重数据类型为 QInt8
      /* bias_dtype */ c10::ScalarType::QInt32,  // 偏置数据类型为 QInt32
      /* stride */ {2, 2},  // 步长为 [2, 2]
      /* padding */ {1, 1},  // 填充为 [1, 1]
      /* dilation */ {1, 1},  // 膨胀率为 [1, 1]
      /* groups */ 1);  // 分组数为 1
}

TEST_F(VulkanAPITest, conv2d_quantized_prepack_fixed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数，测试预打包固定参数的量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ true,  // 进行预打包
      /* compute params */ false,  // 不计算参数
      /* random params */ false,  // 不使用随机参数
      /* input_shape */ {1, 3, 8, 8},  // 输入张量形状为 [1, 3, 8, 8]
      /* weight_shape */ {1, 3, 3, 3},  // 权重张量形状为 [1, 3, 3, 3]
      /* bias_shape */ {1},  // 偏置张量形状为 [1]
      /* weight_dtype */ c10::ScalarType::QInt8,  // 权重数据类型为 QInt8
      /* bias_dtype */ c10::ScalarType::QInt32,  // 偏置数据类型为 QInt32
      /* stride */ {2, 2},  // 步长为 [2, 2]
      /* padding */ {1, 1},  // 填充为 [1, 1]
      /* dilation */ {1, 1},  // 膨胀率为 [1, 1]
      /* groups */ 1);  // 分组数为 1
}

TEST_F(VulkanAPITest, conv2d_quantized_prepack_computed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数，测试预打包计算参数的量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ true,  // 进行预打包
      /* compute params */ true,  // 计算参数
      /* random params */ false,  // 不使用随机参数
      /* input_shape */ {1, 3, 8, 8},  // 输入张量形状为 [1, 3, 8, 8]
      /* weight_shape */ {1, 3, 3, 3},  // 权重张量形状为 [1, 3, 3, 3]
      /* bias_shape */ {1},  // 偏置张量形状为 [1]
      /* weight_dtype */ c10::ScalarType::QInt8,  // 权重数据类型为 QInt8
      /* bias_dtype */ c10::ScalarType::QInt32,  // 偏置数据类型为 QInt32
      /* stride */ {2, 2},  // 步长为 [2, 2]
      /* padding */ {1, 1},  // 填充为 [1, 1]
      /* dilation */ {1, 1},  // 膨胀率为 [1, 1]
      /* groups */ 1);  // 分组数为 1
}

TEST_F(VulkanAPITest, conv2d_quantized_prepack_random_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数，测试预打包随机参数的量化卷积操作
  test_quantized_conv2d(
      /* prepacking? */ true,  // 进行预打包
      /* compute params */ false,  // 不计算参数
      /* random params */ true,  // 使用随机参数
      /* input_shape */ {1, 3, 8, 8},  // 输入张量形状为 [1, 3, 8, 8]
      /* weight_shape */ {1, 3, 3, 3},  // 权重张量形状为 [1, 3, 3, 3]
      /* bias_shape */ {1},  // 偏置张量形状为 [1]
      /* weight_dtype */ c10::ScalarType::QInt8,  // 权重数据类型为 QInt8
      /* bias_dtype */ c10::ScalarType::QInt32,  // 偏置数据类型为 QInt32
      /* stride */ {2, 2},  // 步长为 [2, 2]
      /* padding */ {1, 1},  // 填充为 [1, 1]
      /* dilation */ {1, 1},  // 膨胀率为 [1, 1]
      /* groups */ 1);  // 分组数为 1
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_fixed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数，测试深度可分离量化卷积固定参数的操作
  test_quantized_conv2d(
      /* prepacking? */ false,  // 不进行预打包
      /* compute params */ false,  // 不计算参数
      /* random params */ false,  // 不使用随机参数
      /* input_shape */ {1, 7, 137, 199},  // 输入张量形状为 [1, 7, 137, 199]
      /* weight_shape */ {7, 1, 17, 7},  // 权重张量形状为 [7, 1, 17, 7]
      /* bias_shape */ {7},  // 偏置张量形状为 [7]
      /* weight_dtype */ c10::ScalarType::QInt8,  // 权重数据类型为 QInt8
      /* bias_dtype */ c10::ScalarType::QInt32,  // 偏置数据类型为 QInt32
      /* stride */ {2, 3},  // 步长为 [2, 3]
      /* padding */ {0, 4},  // 填充为 [0, 4]
      /* dilation */ {3, 1},  // 膨胀率为 [3, 1]
      /* groups */ 7);  // 分组数为 7
}
TEST_F(VulkanAPITest, conv2d_dw_quantized_computed_params_int8_int32) {
  // 调用测试函数 test_quantized_conv2d 进行量化卷积测试，不进行预打包
  test_quantized_conv2d(
      /* prepacking? */ false,
      // 计算卷积参数
      /* compute params */ true,
      // 使用固定的输入参数而非随机参数
      /* random params */ false,
      // 输入张量形状为 {1, 7, 137, 199}
      /* input_shape */ {1, 7, 137, 199},
      // 卷积核形状为 {7, 1, 17, 7}
      /* weight_shape */ {7, 1, 17, 7},
      // 偏置项形状为 {7}
      /* bias_shape */ {7},
      // 卷积核数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {2, 3}
      /* stride */ {2, 3},
      // 卷积填充为 {0, 4}
      /* padding */ {0, 4},
      // 卷积扩展率为 {3, 1}
      /* dilation */ {3, 1},
      // 分组卷积数为 7
      /* groups */ 7);
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_random_params_int8_int32) {
  // 调用测试函数 test_quantized_conv2d 进行量化卷积测试，不进行预打包
  test_quantized_conv2d(
      /* prepacking? */ false,
      // 不计算卷积参数
      /* compute params */ false,
      // 使用随机的输入参数
      /* random params */ true,
      // 输入张量形状为 {1, 7, 137, 199}
      /* input_shape */ {1, 7, 137, 199},
      // 卷积核形状为 {7, 1, 17, 7}
      /* weight_shape */ {7, 1, 17, 7},
      // 偏置项形状为 {7}
      /* bias_shape */ {7},
      // 卷积核数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {2, 3}
      /* stride */ {2, 3},
      // 卷积填充为 {0, 4}
      /* padding */ {0, 4},
      // 卷积扩展率为 {3, 1}
      /* dilation */ {3, 1},
      // 分组卷积数为 7
      /* groups */ 7);
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_prepack_fixed_params_int8_int32) {
  // 调用测试函数 test_quantized_conv2d 进行量化卷积测试，并进行预打包
  test_quantized_conv2d(
      /* prepacking? */ true,
      // 不计算卷积参数
      /* compute params */ false,
      // 使用固定的输入参数而非随机参数
      /* random params */ false,
      // 输入张量形状为 {1, 7, 137, 199}
      /* input_shape */ {1, 7, 137, 199},
      // 卷积核形状为 {7, 1, 17, 7}
      /* weight_shape */ {7, 1, 17, 7},
      // 偏置项形状为 {7}
      /* bias_shape */ {7},
      // 卷积核数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {2, 3}
      /* stride */ {2, 3},
      // 卷积填充为 {0, 4}
      /* padding */ {0, 4},
      // 卷积扩展率为 {3, 1}
      /* dilation */ {3, 1},
      // 分组卷积数为 7
      /* groups */ 7);
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_prepack_computed_params_int8_int32) {
  // 调用测试函数 test_quantized_conv2d 进行量化卷积测试，并进行预打包
  test_quantized_conv2d(
      /* prepacking? */ true,
      // 计算卷积参数
      /* compute params */ true,
      // 使用固定的输入参数而非随机参数
      /* random params */ false,
      // 输入张量形状为 {1, 7, 137, 199}
      /* input_shape */ {1, 7, 137, 199},
      // 卷积核形状为 {7, 1, 17, 7}
      /* weight_shape */ {7, 1, 17, 7},
      // 偏置项形状为 {7}
      /* bias_shape */ {7},
      // 卷积核数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {2, 3}
      /* stride */ {2, 3},
      // 卷积填充为 {0, 4}
      /* padding */ {0, 4},
      // 卷积扩展率为 {3, 1}
      /* dilation */ {3, 1},
      // 分组卷积数为 7
      /* groups */ 7);
}

TEST_F(VulkanAPITest, conv2d_dw_quantized_prepack_random_params_int8_int32) {
  // 调用测试函数 test_quantized_conv2d 进行量化卷积测试，并进行预打包
  test_quantized_conv2d(
      /* prepacking? */ true,
      // 不计算卷积参数
      /* compute params */ false,
      // 使用随机的输入参数
      /* random params */ true,
      // 输入张量形状为 {1, 7, 137, 199}
      /* input_shape */ {1, 7, 137, 199},
      // 卷积核形状为 {7, 1, 17, 7}
      /* weight_shape */ {7, 1, 17, 7},
      // 偏置项形状为 {7}
      /* bias_shape */ {7},
      // 卷积核数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {2, 3}
      /* stride */ {2, 3},
      // 卷积填充为 {0, 4}
      /* padding */ {0, 4},
      // 卷积扩展率为 {3, 1}
      /* dilation */ {3, 1},
      // 分组卷积数为 7
      /* groups */ 7);
}
TEST_F(VulkanAPITest, conv2d_pw_quantized_fixed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数进行量化卷积测试，不进行预打包
  test_quantized_conv2d(
      /* prepacking? */ false,
      // 不计算参数
      /* compute params */ false,
      // 不使用随机参数
      /* random params */ false,
      // 输入张量形状为 {1, 17, 127, 397}
      /* input_shape */ {1, 17, 127, 397},
      // 权重张量形状为 {29, 17, 1, 1}
      /* weight_shape */ {29, 17, 1, 1},
      // 偏置张量形状为 {29}
      /* bias_shape */ {29},
      // 权重数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {1, 1}
      /* stride */ {1, 1},
      // 卷积填充为 {0, 0}
      /* padding */ {0, 0},
      // 卷积扩展率为 {1, 1}
      /* dilation */ {1, 1},
      // 卷积分组数为 1
      /* groups */ 1);
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_computed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数进行量化卷积测试，不进行预打包
  test_quantized_conv2d(
      /* prepacking? */ false,
      // 计算参数
      /* compute params */ true,
      // 不使用随机参数
      /* random params */ false,
      // 输入张量形状为 {1, 17, 127, 397}
      /* input_shape */ {1, 17, 127, 397},
      // 权重张量形状为 {29, 17, 1, 1}
      /* weight_shape */ {29, 17, 1, 1},
      // 偏置张量形状为 {29}
      /* bias_shape */ {29},
      // 权重数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {1, 1}
      /* stride */ {1, 1},
      // 卷积填充为 {0, 0}
      /* padding */ {0, 0},
      // 卷积扩展率为 {1, 1}
      /* dilation */ {1, 1},
      // 卷积分组数为 1
      /* groups */ 1);
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_random_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数进行量化卷积测试，不进行预打包
  test_quantized_conv2d(
      /* prepacking? */ false,
      // 不计算参数
      /* compute params */ false,
      // 使用随机参数
      /* random params */ true,
      // 输入张量形状为 {1, 17, 127, 397}
      /* input_shape */ {1, 17, 127, 397},
      // 权重张量形状为 {29, 17, 1, 1}
      /* weight_shape */ {29, 17, 1, 1},
      // 偏置张量形状为 {29}
      /* bias_shape */ {29},
      // 权重数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {1, 1}
      /* stride */ {1, 1},
      // 卷积填充为 {0, 0}
      /* padding */ {0, 0},
      // 卷积扩展率为 {1, 1}
      /* dilation */ {1, 1},
      // 卷积分组数为 1
      /* groups */ 1);
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_prepack_fixed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数进行量化卷积测试，预先打包
  test_quantized_conv2d(
      /* prepacking? */ true,
      // 不计算参数
      /* compute params */ false,
      // 不使用随机参数
      /* random params */ false,
      // 输入张量形状为 {1, 17, 127, 397}
      /* input_shape */ {1, 17, 127, 397},
      // 权重张量形状为 {29, 17, 1, 1}
      /* weight_shape */ {29, 17, 1, 1},
      // 偏置张量形状为 {29}
      /* bias_shape */ {29},
      // 权重数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {1, 1}
      /* stride */ {1, 1},
      // 卷积填充为 {0, 0}
      /* padding */ {0, 0},
      // 卷积扩展率为 {1, 1}
      /* dilation */ {1, 1},
      // 卷积分组数为 1
      /* groups */ 1);
}

TEST_F(VulkanAPITest, conv2d_pw_quantized_prepack_computed_params_int8_int32) {
  // 调用 test_quantized_conv2d 函数进行量化卷积测试，预先打包
  test_quantized_conv2d(
      /* prepacking? */ true,
      // 计算参数
      /* compute params */ true,
      // 不使用随机参数
      /* random params */ false,
      // 输入张量形状为 {1, 17, 127, 397}
      /* input_shape */ {1, 17, 127, 397},
      // 权重张量形状为 {29, 17, 1, 1}
      /* weight_shape */ {29, 17, 1, 1},
      // 偏置张量形状为 {29}
      /* bias_shape */ {29},
      // 权重数据类型为 QInt8
      /* weight_dtype */ c10::ScalarType::QInt8,
      // 偏置数据类型为 QInt32
      /* bias_dtype */ c10::ScalarType::QInt32,
      // 卷积步长为 {1, 1}
      /* stride */ {1, 1},
      // 卷积填充为 {0, 0}
      /* padding */ {0, 0},
      // 卷积扩展率为 {1, 1}
      /* dilation */ {1, 1},
      // 卷积分组数为 1
      /* groups */ 1);
}
# 在 Vulkan API 测试框架中定义一个测试，测试量化卷积操作的功能
TEST_F(VulkanAPITest, conv2d_pw_quantized_prepack_random_params_int8_int32) {
  # 调用量化卷积测试函数，指定以下参数:
  test_quantized_conv2d(
      /* prepacking? */ true,                     # 是否预打包？
      /* compute params */ false,                 # 是否计算参数？
      /* random params */ true,                   # 是否使用随机参数？
      /* input_shape */ {1, 17, 127, 397},        # 输入张量形状
      /* weight_shape */ {29, 17, 1, 1},           # 权重张量形状
      /* bias_shape */ {29},                      # 偏置张量形状
      /* weight_dtype */ c10::ScalarType::QInt8,   # 权重数据类型
      /* bias_dtype */ c10::ScalarType::QInt32,    # 偏置数据类型
      /* stride */ {1, 1},                        # 卷积步长
      /* padding */ {0, 0},                       # 卷积填充
      /* dilation */ {1, 1},                      # 卷积膨胀率
      /* groups */ 1);                            # 卷积分组数
}

# 在 Vulkan API 测试框架中定义一个测试，测试量化张量获取比例和零点
TEST_F(VulkanAPITest, quantized_tensor_get_scale_zero_point) {
  # 创建一个 CPU 上的随机浮点数张量，形状为 2x13x12x27
  const auto in_cpu =
      at::rand({2, 13, 12, 27}, at::TensorOptions(at::kCPU).dtype(at::kFloat));

  # 定义量化的比例为 0.1
  const double scale = 0.1;
  # 定义量化的零点为 10
  const int zero_point = 10;

  # 对 CPU 上的浮点数张量进行整数量化
  const auto cpu_quantized = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);

  # 将输入张量转换为 Vulkan 张量
  const auto in_vulkan = in_cpu.vulkan();
  # 在 Vulkan 上对张量进行整数量化
  const auto vulkan_quantized = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  # 获取 CPU 量化后张量的比例和零点
  double cpu_quantized_scale = cpu_quantized.q_scale();
  int64_t cpu_quantized_zero_point = cpu_quantized.q_zero_point();
  # 获取 Vulkan 量化后张量的比例和零点
  double vulkan_quantized_scale = vulkan_quantized.q_scale();
  int64_t vulkan_quantized_zero_point = vulkan_quantized.q_zero_point();

  # 断言 CPU 和 Vulkan 量化后张量的比例和零点相等
  ASSERT_TRUE(
      cpu_quantized_scale == vulkan_quantized_scale &&
      cpu_quantized_zero_point == vulkan_quantized_zero_point);
}

# 私有函数，用于测试量化线性层的功能
bool _test_quantized_linear(
    const at::Tensor& input_cpu,    # 输入 CPU 张量
    const at::Tensor& weight,       # 权重张量
    const at::Tensor& bias,         # 偏置张量
    double out_scale,               # 输出比例
    int out_zero_point,             # 输出零点
    bool input_quant_dtype_int8,    # 输入量化数据类型是否为 int8
    ...
    // 根据输入的量化类型，计算输入数据的量化参数
    const auto input_quant_params = compute_quant_params(
        input_cpu,
        input_quant_dtype_int8 ? c10::ScalarType::QInt8
                               : c10::ScalarType::QUInt8);
    // 获取输入数据的缩放因子
    double scale = std::get<0>(input_quant_params);
    // 将缩放因子转换为 float 类型
    scale = safe_downcast<float>(scale);
    // 获取输入数据的零点
    int zero_point = std::get<1>(input_quant_params);
    // 对输入数据进行量化
    auto input_cpu_quantized = at::quantize_per_tensor(
        input_cpu,
        scale,
        zero_point,
        input_quant_dtype_int8 ? c10::ScalarType::QInt8
                               : c10::ScalarType::QUInt8);

    // 根据权重的量化类型，计算权重数据的量化参数
    const auto weight_quant_params = compute_quant_params(
        weight,
        weight_quant_dtype_int8 ? c10::ScalarType::QInt8
                                : c10::ScalarType::QUInt8);
    // 获取权重数据的缩放因子
    double w_scale = std::get<0>(weight_quant_params);
    // 将缩放因子转换为 float 类型
    w_scale = safe_downcast<float>(w_scale);
    // 权重数据的零点预期始终为 0
    int w_zero_point = 0;
    // 对权重数据进行量化
    const auto weight_cpu_quantized = at::quantize_per_tensor(
        weight,
        w_scale,
        w_zero_point,
        weight_quant_dtype_int8 ? c10::ScalarType::QInt8
                                : c10::ScalarType::QUInt8);

    // 调用预打包操作，准备量化线性运算所需的数据
    auto pack =
        callOpByName("quantized::linear_prepack", "", weight_cpu_quantized, bias);

    // 执行量化线性运算
    auto out_cpu_quant = callOpByName(
        "quantized::linear",
        "",
        input_cpu_quantized,
        pack[0],
        out_scale,
        out_zero_point);

    // 对输出数据进行反量化
    at::Tensor out_cpu_dequant = at::dequantize(out_cpu_quant[0].toTensor());

    // Vulkan
    // 使用 Vulkan 引擎对输入数据进行量化
    auto input_vk_quantized = at::quantize_per_tensor(
        input_cpu.vulkan(),
        scale,
        zero_point,
        input_quant_dtype_int8 ? c10::ScalarType::QInt8
                               : c10::ScalarType::QUInt8);

    // 使用 Vulkan 引擎执行量化线性上下文创建
    c10::intrusive_ptr<at::native::vulkan::ops::LinearPackedContext> vk_pack =
        at::native::vulkan::ops::create_linear_context(
            weight_cpu_quantized.t(), bias);

    // 在 Vulkan 引擎上执行量化线性运算
    out_vk_quant = at::native::vulkan::ops::run_qlinear_context(
        input_vk_quantized, out_scale, out_zero_point, vk_pack);

    // 对 Vulkan 引擎输出的数据进行反量化
    auto out_vk_dequant = at::dequantize(out_vk_quant);

    // 将 Vulkan 引擎输出的数据转换为 CPU 上的数据进行比较
    auto out_vk_to_cpu_dequant = vulkan_to_cpu(out_vk_dequant, out_cpu_dequant);

    // 检查两种计算结果是否几乎相等
    const auto check = almostEqual(
        out_cpu_dequant, out_vk_to_cpu_dequant, safe_downcast<float>(out_scale));
    if (!check) {
        long xpos = -1, ypos = -1;
        // 如果输入数据是二维张量，找到导致失败的行列位置
        if (input_cpu.sizes().size() == 2) {
            showRtol(out_cpu_dequant, out_vk_to_cpu_dequant, &xpos, &ypos);
        } else {
            // 显示几乎相等的结果
            showRtol(out_cpu_dequant, out_vk_to_cpu_dequant);
        }
    // 检查 xpos 和 ypos 是否都不为 -1，表示存在失败的位置
    if (xpos != -1 && ypos != -1) {
      // 输出失败发生的行/列信息
      std::cout << "\nFailure caused on row/col: " << ypos << "/" << xpos
                << "\n";
      // 输出输入张量的缩放和零点信息
      std::cout << "Input tensor scale: " << scale << " zerop: " << zero_point
                << "\n";
      // 输出失败行的输入张量数据
      std::cout << "Input tensor row " << ypos << "\n";
      // 遍历输出输入张量指定行的数据
      for (int i = 0; i < input_cpu.sizes()[1]; i++) {
        std::cout << input_cpu[ypos][i].item<double>() << ", ";
      }
      std::cout << "\n";

      // 输出权重张量的缩放和零点信息
      std::cout << "Weight tensor scale: " << w_scale
                << " zerop: " << w_zero_point << "\n";
      // 输出失败列的权重张量数据
      std::cout << "Weight tensor col " << xpos << "\n";
      // 遍历输出权重张量指定列的数据
      for (int i = 0; i < weight.sizes()[1]; i++) {
        std::cout << weight[xpos][i].item<double>() << ", ";
      }
      std::cout << "\n";

      // 输出量化后输入张量的指定行数据和数据类型
      std::cout << "Input tensor quantized row " << ypos << " with dtype "
                << (input_quant_dtype_int8 ? "QInt8" : "QUInt8") << "\n";
      // 遍历输出量化后输入张量指定行的数据
      for (int i = 0; i < input_cpu.sizes()[1]; i++) {
        std::cout << input_cpu_quantized[ypos][i].item<double>() << ", ";
      }
      std::cout << "\n";

      // 输出量化后权重张量的指定列数据和数据类型
      std::cout << "Weight tensor quantized col " << xpos << " with dtype "
                << (weight_quant_dtype_int8 ? "QInt8" : "QUInt8") << "\n";
      // 遍历输出量化后权重张量指定列的数据
      for (int i = 0; i < weight.sizes()[1]; i++) {
        std::cout << weight_cpu_quantized[xpos][i].item<double>() << ", ";
      }
      std::cout << "\n";

      // 输出偏置张量数据
      std::cout << "bias tensor\n";
      // 遍历输出偏置张量的数据
      for (int i = 0; i < bias.sizes()[0]; i++) {
        std::cout << bias[i].item<double>() << ", ";
      }
      std::cout << "\n";

      // 输出输出张量的缩放和零点信息
      std::cout << "out_scale: " << out_scale
                << " out_zero_point: " << out_zero_point << "\n";

      // 输出 CPU 计算的未匹配输出
      std::cout << "cpu unmatched output: "
                << out_cpu_dequant[ypos][xpos].item<double>() << "\n";
      // 输出 VK 计算的未匹配输出
      std::cout << "vk unmatched output: "
                << out_vk_to_cpu_dequant[ypos][xpos].item<double>() << "\n";
    }
  }
  // 返回检查结果
  return check;
}

// 检测量化线性运算函数，根据输入和权重的不同量化类型进行测试
bool test_quantized_linear_for_dtypes(
    const at::Tensor& input_cpu,
    const at::Tensor& weight,
    const at::Tensor& bias,
    bool input_quant_dtype_int8,
    bool weight_quant_dtype_int8) {
  
  // 生成随机的输出缩放因子
  double out_scale = produce_random_scale();
  // 将输出缩放因子安全地转换为单精度浮点数
  out_scale = safe_downcast<float>(out_scale);
  
  // 生成随机的输出零点，根据输入量化类型的不同选择 QInt8 或 QUInt8
  int out_zero_point = produce_random_zero_point(
      input_quant_dtype_int8 ? c10::ScalarType::QInt8
                             : c10::ScalarType::QUInt8);
  
  // 调用内部函数 _test_quantized_linear 进行量化线性运算的测试
  const auto check = _test_quantized_linear(
      input_cpu,
      weight,
      bias,
      out_scale,
      out_zero_point,
      input_quant_dtype_int8,
      weight_quant_dtype_int8);
  
  // 如果测试失败，希望打印导致失败的确切行/列，以便调试 2D 情况
  if (!check) {
    // 如果输入不是二维的，计算出总的维度数
    if (input_cpu.sizes().size() != 2) {
      const auto d = c10::multiply_integers(
          input_cpu.sizes().cbegin(), input_cpu.sizes().end() - 1);
      // 将输入张量重新视图为二维张量进行再次测试
      auto input_cpu_2d = input_cpu.view({d, input_cpu.size(-1)});
      
      // 再次调用 _test_quantized_linear 进行量化线性运算的测试
      _test_quantized_linear(
          input_cpu_2d,
          weight,
          bias,
          out_scale,
          out_zero_point,
          input_quant_dtype_int8,
          weight_quant_dtype_int8);
    }
  }
  
  // 返回测试结果
  return check;
}

// 对量化线性运算进行测试，使用给定的输入形状、权重形状和偏置形状
void test_quantized_linear(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape) {
  
  // 进入推断模式
  c10::InferenceMode mode;
  
  // 生成随机的输入张量
  const auto input_cpu = produce_random_tensor(input_shape);
  
  // 生成随机的权重张量
  const auto weight = produce_random_tensor(weight_shape);
  
  // 生成随机的偏置张量
  const auto bias = produce_random_tensor(bias_shape);
  
  // 测试不同量化类型组合下的量化线性运算
  bool check =
      test_quantized_linear_for_dtypes(input_cpu, weight, bias, false, true);
  ASSERT_TRUE(check);
  check = test_quantized_linear_for_dtypes(input_cpu, weight, bias, true, true);
  ASSERT_TRUE(check);
}

// 测试线性量化运算在二维扁平输入下的情况
TEST_F(VulkanAPITest, linear_2d_flat) {
  test_quantized_linear({1, 100}, {1, 100}, {1});
}

// 测试线性量化运算在二维小型输入下的情况
TEST_F(VulkanAPITest, linear_2d_small) {
  test_quantized_linear({2, 3}, {4, 3}, {4});
}

// 测试线性量化运算在二维大型输入下的情况
TEST_F(VulkanAPITest, linear_2d_large) {
  test_quantized_linear({1287, 17}, {23, 17}, {23});
}

// 测试线性量化运算在三维扁平输入下的情况
TEST_F(VulkanAPITest, linear_3d_flat) {
  test_quantized_linear({1, 1, 37}, {41, 37}, {41});
}

// 测试线性量化运算在三维小型输入下的情况
TEST_F(VulkanAPITest, linear_3d_small) {
  test_quantized_linear({2, 3, 4}, {5, 4}, {5});
}

// 测试线性量化运算在三维大型输入下的情况
TEST_F(VulkanAPITest, linear_3d_large) {
  test_quantized_linear({23, 17, 41}, {15, 41}, {15});
}

// 测试线性量化运算在四维扁平输入下的情况
TEST_F(VulkanAPITest, linear_4d_flat) {
  test_quantized_linear({1, 1, 1, 37}, {41, 37}, {41});
}

// 测试线性量化运算在四维小型输入下的情况
TEST_F(VulkanAPITest, linear_4d_small) {
  test_quantized_linear({2, 3, 4, 5}, {6, 5}, {6});
}

// 测试线性量化运算在四维大型输入下的情况
TEST_F(VulkanAPITest, linear_4d_large) {
  test_quantized_linear({9, 13, 11, 17}, {23, 17}, {23});
}

// 以下代码与量化无关，因无法在 GitHub CI 上运行测试，移至此以便保留测试功能
// 定义一个函数，用于比较整数和浮点数之间的接近程度，支持预期值为-1的情况
bool texel_almost_equal(int expected, float actual) {
  // 如果预期值为-1，表示不关心实际值，直接返回true
  return (expected == -1) || (fabs(expected - actual) < kTolerance);
}

// 定义一个函数，用于比较ivec4和vec4结构的数据是否接近
bool texel_almost_equal(const ivec4& expected, const vec4& actual) {
  // 分别比较四个分量的接近程度
  return (
      texel_almost_equal(expected.data[0], actual.data[0]) &&
      texel_almost_equal(expected.data[1], actual.data[1]) &&
      texel_almost_equal(expected.data[2], actual.data[2]) &&
      texel_almost_equal(expected.data[3], actual.data[3]));
}

// 定义一个测试案例，使用Google Test框架的测试夹具VulkanAPITest，测试extract_texel函数
TEST_F(VulkanAPITest, extract_texel_test) {
  // 定义测试所需的维度参数和计算一些相关的常量
  int n = 3;
  int c = 5;
  int h = 6;
  int w = 7;
  int hw = h * w;
  int chw = c * h * w;

  // 创建一个CPU端的张量，范围从0到n*c*h*w-1，数据类型为float
  auto cpu =
      at::range(0, n * c * h * w - 1, at::device(at::kCPU).dtype(at::kFloat))
          .reshape({n, c, h, w});
  // 将CPU端的张量转换为Vulkan端的张量
  auto vk = cpu.vulkan();

  // 默认使用通道打包的3D张量
  // x和y通道是典型的平面
  // z通道与批次和通道打包在一起，例如每4个通道打包成一个texel。
  // 访问批次nn和通道cc的张量时，计算z坐标为nn * ceil(c / 4) + cc / 4，其中c是通道数。
  // 新的批次总是从新的z坐标开始。因此，当c不能被4整除时，填充区域会有一些未定义值。我们用-1表示不在这些值上进行比较。
  std::tuple<ivec3, ivec4> test_cases[]{
      {{0, 0, 0}, {0, hw, 2 * hw, 3 * hw}},
      {{1, 0, 0}, {1, hw + 1, 2 * hw + 1, 3 * hw + 1}},
      {{0, 0, 1}, {4 * hw, -1, -1, -1}},
      {{0, 0, 2}, {chw, chw + hw, chw + 2 * hw, chw + 3 * hw}},
      {{0, 1, 2}, {chw + w, chw + hw + w, chw + 2 * hw + w, chw + 3 * hw + w}},
      {{0, 0, 3}, {chw + 4 * hw, -1, -1, -1}},
      {{0, 1, 3}, {chw + 4 * hw + w, -1, -1, -1}},
      {{0, 0, 4}, {2 * chw, 2 * chw + hw, 2 * chw + 2 * hw, 2 * chw + 3 * hw}},
      {{0, 1, 4},
       {2 * chw + w,
        2 * chw + hw + w,
        2 * chw + 2 * hw + w,
        2 * chw + 3 * hw + w}},
  };

  bool has_failure = false;
  // 遍历测试案例数组，执行测试
  for (const auto& test_case : test_cases) {
    const auto [loc, expected] = test_case;

    // 调用ops::utils::extract_texel函数提取vk张量中指定位置loc的texel数据
    vec4 actual = ops::utils::extract_texel(vk, loc);
    // 检查提取的实际值是否与预期值接近
    if (!texel_almost_equal(expected, actual)) {
      // 输出测试失败的信息
      std::cout << "On loc: " << loc << " expected: " << expected
                << " actual: " << actual << std::endl;
      // 标记测试失败
      has_failure = true;
    }
  }
  // 断言所有测试都通过
  ASSERT_TRUE(!has_failure);
}
// 定义一个测试用例，用于测试通道到高度打包的转换
TEST_F(VulkanAPITest, channel_to_height_packing_test) {
  int n = 3;  // 设置批次大小为 3
  int c = 5;  // 设置通道数为 5
  int h = 6;  // 设置高度为 6
  int w = 7;  // 设置宽度为 7
  int hw = h * w;  // 计算单个通道的大小，即高度乘以宽度
  int chw = c * h * w;  // 计算整个数据张量的大小，即通道数乘以高度乘以宽度

  // 创建一个包含从 0 到 n * c * h * w - 1 的整数序列的张量，数据类型为 float，存储于 CPU 上
  auto data =
      at::range(0, n * c * h * w - 1, at::device(at::kCPU).dtype(at::kFloat))
          .reshape({n, c, h, w});

  // 将数据张量转换为 Vulkan 张量格式
  auto v_input = at::native::vulkan::ops::convert(data.vulkan());

  // 执行通道打包到高度打包的转换操作
  auto v_output =
      packing::convert_image_channels_packed_to_height_packed(v_input);

  // 断言转换后的 Vulkan 张量的 GPU 存储布局为 TENSOR_HEIGHT_PACKED
  ASSERT_EQ(
      v_output.gpu_memory_layout(), api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED);

  // 将 Vulkan 张量转换回普通张量格式，用于后续验证
  at::Tensor output = at::native::vulkan::ops::convert(v_output);

  // 准备测试用例，包含一系列测试数据和期望输出
  std::tuple<ivec3, ivec4> test_cases[]{
      {{0, 0, 0}, {0, w, 2 * w, 3 * w}},  // 第一个测试点，验证第一层第一通道的高度打包结果
      {{0, 1, 0}, {4 * w, 5 * w, -1, -1}},  // 第二个测试点，验证第一层第二通道的高度打包结果
      {{1, 0, 0}, {0 * w + 1, 1 * w + 1, 2 * w + 1, 3 * w + 1}},  // 第三个测试点，验证第二层第一通道的高度打包结果
      {{1, 1, 0}, {4 * w + 1, 5 * w + 1, -1, -1}},  // 第四个测试点，验证第二层第二通道的高度打包结果
      {{0, 0, 4}, {4 * hw, 4 * hw + w, 4 * hw + 2 * w, 4 * hw + 3 * w}},  // 第五个测试点，验证第一层第一通道在第四层的高度打包结果
      {{0, 0, 4 + 2 * c},
       {2 * chw + 4 * hw,
        2 * chw + 4 * hw + w,
        2 * chw + 4 * hw + 2 * w,
        2 * chw + 4 * hw + 3 * w}},  // 第六个测试点，验证第一层第一通道在第四层的高度打包结果（扩展通道数）
  };

  bool has_failure = false;  // 用于标记是否存在测试失败的情况
  for (const auto& test_case : test_cases) {
    const auto [loc, expected] = test_case;

    // 提取输出张量中指定位置的像素块
    vec4 actual = ops::utils::extract_texel(output, loc);

    // 检查实际结果与期望结果是否接近，如果不接近则标记测试失败并输出详细信息
    if (!texel_almost_equal(expected, actual)) {
      std::cout << "On loc: " << loc << " expected: " << expected
                << " actual: " << actual << std::endl;
      has_failure = true;
    }
  }

  // 断言最终的测试结果是否全部通过，即没有失败的测试用例
  ASSERT_TRUE(!has_failure);
}
// 在 VulkanAPI 测试框架中，定义了一个名为 channel_to_width_packing_test 的测试函数
TEST_F(VulkanAPITest, channel_to_width_packing_test) {
  // 初始化变量 n、c、h、w，分别表示样本数、通道数、高度、宽度
  int n = 3;
  int c = 5;
  int h = 6;
  int w = 7;
  // 计算高度和宽度的乘积
  int hw = h * w;
  // 计算通道数、高度和宽度的乘积
  int chw = c * h * w;

  // 创建一个 ATen 张量 data，用于存储指定范围内的数据，数据类型为浮点型，存储在 CPU 上
  auto data =
      at::range(0, n * c * h * w - 1, at::device(at::kCPU).dtype(at::kFloat))
          .reshape({n, c, h, w});

  // 将 data 张量转换为 Vulkan 张量 v_input
  auto v_input = at::native::vulkan::ops::convert(data.vulkan());
  // 将通道按宽度打包的 Vulkan 张量 v_input 进行处理，得到 v_output
  auto v_output =
      packing::convert_image_channels_packed_to_width_packed(v_input);
  // 断言 v_output 的 GPU 存储布局为 TENSOR_WIDTH_PACKED
  ASSERT_EQ(
      v_output.gpu_memory_layout(), api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

  // 将 v_output 张量从 Vulkan 转换回 CPU 上的张量 output
  at::Tensor output = at::native::vulkan::ops::convert(v_output);

  // 定义一组测试用例，测试输出张量的 texel 提取是否正确
  std::tuple<ivec3, ivec4> test_cases[]{
      {{0, 0, 0}, {0, 1, 2, 3}},
      {{1, 0, 0}, {4, 5, 6, -1}},
      {{0, 2, 0}, {2 * w + 0, 2 * w + 1, 2 * w + 2, 2 * w + 3}},
      {{1, 2, 0}, {2 * w + 4, 2 * w + 5, 2 * w + 6, -1}},
      {{0, 0, 4}, {4 * hw + 0, 4 * hw + 1, 4 * hw + 2, 4 * hw + 3}},
      {{1, 0, 4}, {4 * hw + 4, 4 * hw + 5, 4 * hw + 6, -1}},
      {{0, 0, 4 + 2 * c},
       {2 * chw + 4 * hw,
        2 * chw + 4 * hw + 1,
        2 * chw + 4 * hw + 2,
        2 * chw + 4 * hw + 3}},
  };

  // 初始化是否存在失败的标志位
  bool has_failure = false;
  // 遍历每个测试用例
  for (const auto& test_case : test_cases) {
    // 从测试用例中解包出位置 loc 和预期值 expected
    const auto [loc, expected] = test_case;

    // 从 output 张量中提取指定位置 loc 的 texel 值 actual
    vec4 actual = ops::utils::extract_texel(output, loc);
    // 检查提取的 texel 值是否与预期值 expected 几乎相等，如果不相等则输出错误信息
    if (!texel_almost_equal(expected, actual)) {
      std::cout << "On loc: " << loc << " expected: " << expected
                << " actual: " << actual << std::endl;
      has_failure = true;
    }
  }
  // 断言所有测试用例均未出现失败
  ASSERT_TRUE(!has_failure);
}

// 执行 GELU 激活函数的测试函数
void test_gelu(
    const at::IntArrayRef input_shape,
    const c10::ScalarType dtype,
    bool self_test) {
  // 生成指定形状的随机张量 in_cpu
  const auto& in_cpu = produce_random_tensor(input_shape);

  // 计算 in_cpu 的量化参数 scale 和 zero_point
  auto [scale, zero_point] = compute_quant_params(in_cpu, dtype);
  // 将 scale 转换为 float 类型
  scale = safe_downcast<float>(scale);

  // 对 in_cpu 进行按张量量化处理，得到 in_cpu_quantized
  auto in_cpu_quantized =
      at::quantize_per_tensor(in_cpu, scale, zero_point, dtype);

  // 将 in_cpu 转换为 Vulkan 张量，并进行量化处理，得到 in_vk_quantized
  auto in_vk_quantized =
      at::quantize_per_tensor(in_cpu.vulkan(), scale, zero_point, dtype);

  // 设置 GELU 的近似类型为 "tanh"
  auto approximate = "tanh";

  // 根据 self_test 决定是否就地执行 GELU 操作，得到 out_cpu_quantized
  const auto& out_cpu_quantized = self_test
      ? at::gelu_(in_cpu_quantized, approximate)
      : at::gelu(in_cpu_quantized, approximate);

  // 根据 self_test 决定是否就地执行 GELU 操作，得到 out_vk_quantized
  const auto& out_vk_quantized = self_test
      ? at::gelu_(in_vk_quantized, approximate)
      : at::gelu(in_vk_quantized, approximate);

  // 对 out_cpu_quantized 进行去量化操作，得到 out_cpu_deq
  const auto& out_cpu_deq = at::dequantize(out_cpu_quantized);
  // 对 out_vk_quantized 进行去量化操作，得到 out_vk_deq
  const auto& out_vk_deq = at::dequantize(out_vk_quantized);
  // 将 out_vk_deq 转换为 CPU 上的张量，得到 out_vk_deq_cpu
  const auto& out_vk_deq_cpu = out_vk_deq.cpu();

  // 检查 out_vk_deq_cpu 和 out_cpu_deq 是否几乎相等，使用 scale 作为容差
  const auto check = almostEqual(out_vk_deq_cpu, out_cpu_deq, scale);

  // 如果检查不通过，则展示相对容差超出阈值的信息
  if (!check) {
    showRtol(out_cpu_deq, out_vk_deq_cpu);
  }
  // 断言检查结果为真，即 out_vk_deq_cpu 和 out_cpu_deq 几乎相等
  ASSERT_TRUE(check);
}
TEST_F(VulkanAPITest, gelu_qint8) {
  // 在 Vulkan API 测试框架中，测试 QInt8 类型的 gelu 函数
  test_gelu({200, 20}, c10::ScalarType::QInt8, false);
  // 同上，测试不同形状的输入
  test_gelu({200, 20, 10}, c10::ScalarType::QInt8, false);
  test_gelu({200, 20, 30, 10}, c10::ScalarType::QInt8, false);
}

TEST_F(VulkanAPITest, gelu_qint8_self) {
  // 在 Vulkan API 测试框架中，测试 QInt8 类型的 gelu 函数，且 inplace 为 true
  test_gelu({4, 1, 4}, c10::ScalarType::QInt8, true);
  // 同上，测试不同形状的输入
  test_gelu({200, 20}, c10::ScalarType::QInt8, true);
  test_gelu({200, 20, 10}, c10::ScalarType::QInt8, true);
  test_gelu({200, 20, 30, 10}, c10::ScalarType::QInt8, true);
}

TEST_F(VulkanAPITest, gelu_quint8) {
  // 在 Vulkan API 测试框架中，测试 QUInt8 类型的 gelu 函数
  test_gelu({200, 20}, c10::ScalarType::QUInt8, false);
  // 同上，测试不同形状的输入
  test_gelu({200, 20, 10}, c10::ScalarType::QUInt8, false);
  test_gelu({200, 20, 30, 10}, c10::ScalarType::QUInt8, false);
}

TEST_F(VulkanAPITest, gelu_quint8_self) {
  // 在 Vulkan API 测试框架中，测试 QUInt8 类型的 gelu 函数，且 inplace 为 true
  test_gelu({4, 1, 4}, c10::ScalarType::QUInt8, true);
  // 同上，测试不同形状的输入
  test_gelu({200, 20}, c10::ScalarType::QUInt8, true);
  test_gelu({200, 20, 10}, c10::ScalarType::QUInt8, true);
  test_gelu({200, 20, 30, 10}, c10::ScalarType::QUInt8, true);
}

void test_relu(
    const at::IntArrayRef input_shape,
    const c10::ScalarType dtype,
    bool inplace) {
  // 创建随机张量作为输入
  const auto in_cpu = produce_random_tensor(input_shape);

  // 计算输入张量的量化参数
  const auto input_quant_params = compute_quant_params(in_cpu, dtype);
  double scale = std::get<0>(input_quant_params);
  scale = safe_downcast<float>(scale);
  int zero_point = std::get<1>(input_quant_params);

  // 使用量化参数对 CPU 上的输入张量进行量化
  auto in_cpu_quantized =
      at::quantize_per_tensor(in_cpu, scale, zero_point, dtype);

  // 使用量化参数对 Vulkan 设备上的输入张量进行量化
  auto in_vk_quantized =
      at::quantize_per_tensor(in_cpu.vulkan(), scale, zero_point, dtype);

  // 对 CPU 上的量化输入张量进行 ReLU 操作（原地或非原地）
  const auto out_cpu_quantized =
      inplace ? at::relu_(in_cpu_quantized) : at::relu(in_cpu_quantized);

  // 对 Vulkan 设备上的量化输入张量进行 ReLU 操作（原地或非原地）
  const auto out_vk_quantized =
      inplace ? at::relu_(in_vk_quantized) : at::relu(in_vk_quantized);

  // 将 CPU 上的量化输出张量反量化
  const auto out_cpu_deq = at::dequantize(out_cpu_quantized);

  // 将 Vulkan 设备上的量化输出张量反量化
  const auto out_vk_deq = at::dequantize(out_vk_quantized);

  // 将 Vulkan 设备上的量化输出张量转移到 CPU 并反量化
  const auto out_vk_deq_cpu = out_vk_deq.cpu();

  // 检查两个反量化后的张量是否几乎相等
  const auto check =
      almostEqual(out_vk_deq_cpu, out_cpu_deq, safe_downcast<float>(scale));

  // 如果检查不通过，则展示相对误差，并断言检查通过
  if (!check) {
    showRtol(out_cpu_deq, out_vk_deq_cpu);
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, relu_qint8) {
  // 在 Vulkan API 测试框架中，测试 QInt8 类型的 relu 函数
  test_relu({200, 20}, c10::ScalarType::QInt8, false);
  // 同上，测试不同形状的输入
  test_relu({200, 20, 10}, c10::ScalarType::QInt8, false);
  test_relu({200, 20, 30, 10}, c10::ScalarType::QInt8, false);
}

TEST_F(VulkanAPITest, relu_qint8_inplace) {
  // 在 Vulkan API 测试框架中，测试 QInt8 类型的 relu 函数，且 inplace 为 true
  test_relu({4, 1, 4}, c10::ScalarType::QInt8, true);
  // 同上，测试不同形状的输入
  test_relu({200, 20}, c10::ScalarType::QInt8, true);
  test_relu({200, 20, 10}, c10::ScalarType::QInt8, true);
  test_relu({200, 20, 30, 10}, c10::ScalarType::QInt8, true);
}

TEST_F(VulkanAPITest, relu_quint8) {
  // 在 Vulkan API 测试框架中，测试 QUInt8 类型的 relu 函数
  test_relu({200, 20}, c10::ScalarType::QUInt8, false);
  // 同上，测试不同形状的输入
  test_relu({200, 20, 10}, c10::ScalarType::QUInt8, false);
  test_relu({200, 20, 30, 10}, c10::ScalarType::QUInt8, false);
}
TEST_F(VulkanAPITest, relu_quint8_inplace) {
  // 调用 test_relu 函数测试 QUInt8 类型的 inplace ReLU 操作，对每个给定的维度参数进行测试
  test_relu({4, 1, 4}, c10::ScalarType::QUInt8, true);
  test_relu({200, 20}, c10::ScalarType::QUInt8, true);
  test_relu({200, 20, 10}, c10::ScalarType::QUInt8, true);
  test_relu({200, 20, 30, 10}, c10::ScalarType::QUInt8, true);
}

} // namespace

#endif /* USE_VULKAN_API */
```