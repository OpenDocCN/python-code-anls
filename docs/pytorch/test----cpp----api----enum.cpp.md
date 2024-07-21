# `.\pytorch\test\cpp\api\enum.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/torch.h>  // 引入 PyTorch C++ API 的头文件
#include <variant>        // 引入标准库中的 variant 头文件，用于支持多态类型

#include <test/cpp/api/support.h>  // 引入测试支持函数的头文件

#define TORCH_ENUM_PRETTY_PRINT_TEST(name)                           \  // 定义宏 TORCH_ENUM_PRETTY_PRINT_TEST，用于测试枚举类型的输出
  {                                                                  \
    v = torch::k##name;                                              \  // 将枚举值 k##name 赋值给变量 v
    std::string pretty_print_name("k");                              \  // 创建一个字符串 pretty_print_name，初始化为 "k"
    pretty_print_name.append(#name);                                 \  // 将宏参数 name 转换为字符串并追加到 pretty_print_name
    ASSERT_EQ(torch::enumtype::get_enum_name(v), pretty_print_name); \  // 使用断言检查枚举值 v 的名称是否与 pretty_print_name 相等
  }
# 定义名为 TEST 的单元测试，用于测试枚举类型的所有取值
TEST(EnumTest, AllEnums) {
  # 声明一个 std::variant 变量 v，包含了多种 torch 的枚举类型
  std::variant<
      torch::enumtype::kLinear,
      torch::enumtype::kConv1D,
      torch::enumtype::kConv2D,
      torch::enumtype::kConv3D,
      torch::enumtype::kConvTranspose1D,
      torch::enumtype::kConvTranspose2D,
      torch::enumtype::kConvTranspose3D,
      torch::enumtype::kSigmoid,
      torch::enumtype::kTanh,
      torch::enumtype::kReLU,
      torch::enumtype::kLeakyReLU,
      torch::enumtype::kFanIn,
      torch::enumtype::kFanOut,
      torch::enumtype::kConstant,
      torch::enumtype::kReflect,
      torch::enumtype::kReplicate,
      torch::enumtype::kCircular,
      torch::enumtype::kNearest,
      torch::enumtype::kBilinear,
      torch::enumtype::kBicubic,
      torch::enumtype::kTrilinear,
      torch::enumtype::kArea,
      torch::enumtype::kSum,
      torch::enumtype::kMean,
      torch::enumtype::kMax,
      torch::enumtype::kNone,
      torch::enumtype::kBatchMean,
      torch::enumtype::kZeros,
      torch::enumtype::kBorder,
      torch::enumtype::kReflection,
      torch::enumtype::kRNN_TANH,
      torch::enumtype::kRNN_RELU,
      torch::enumtype::kLSTM,
      torch::enumtype::kGRU>
      v;

  # 调用 TORCH_ENUM_PRETTY_PRINT_TEST 宏来测试并打印每个枚举类型的输出
  TORCH_ENUM_PRETTY_PRINT_TEST(Linear)
  TORCH_ENUM_PRETTY_PRINT_TEST(Conv1D)
  TORCH_ENUM_PRETTY_PRINT_TEST(Conv2D)
  TORCH_ENUM_PRETTY_PRINT_TEST(Conv3D)
  TORCH_ENUM_PRETTY_PRINT_TEST(ConvTranspose1D)
  TORCH_ENUM_PRETTY_PRINT_TEST(ConvTranspose2D)
  TORCH_ENUM_PRETTY_PRINT_TEST(ConvTranspose3D)
  TORCH_ENUM_PRETTY_PRINT_TEST(Sigmoid)
  TORCH_ENUM_PRETTY_PRINT_TEST(Tanh)
  TORCH_ENUM_PRETTY_PRINT_TEST(ReLU)
  TORCH_ENUM_PRETTY_PRINT_TEST(LeakyReLU)
  TORCH_ENUM_PRETTY_PRINT_TEST(FanIn)
  TORCH_ENUM_PRETTY_PRINT_TEST(FanOut)
  TORCH_ENUM_PRETTY_PRINT_TEST(Constant)
  TORCH_ENUM_PRETTY_PRINT_TEST(Reflect)
  TORCH_ENUM_PRETTY_PRINT_TEST(Replicate)
  TORCH_ENUM_PRETTY_PRINT_TEST(Circular)
  TORCH_ENUM_PRETTY_PRINT_TEST(Nearest)
  TORCH_ENUM_PRETTY_PRINT_TEST(Bilinear)
  TORCH_ENUM_PRETTY_PRINT_TEST(Bicubic)
  TORCH_ENUM_PRETTY_PRINT_TEST(Trilinear)
  TORCH_ENUM_PRETTY_PRINT_TEST(Area)
  TORCH_ENUM_PRETTY_PRINT_TEST(Sum)
  TORCH_ENUM_PRETTY_PRINT_TEST(Mean)
  TORCH_ENUM_PRETTY_PRINT_TEST(Max)
  TORCH_ENUM_PRETTY_PRINT_TEST(None)
  TORCH_ENUM_PRETTY_PRINT_TEST(BatchMean)
  TORCH_ENUM_PRETTY_PRINT_TEST(Zeros)
  TORCH_ENUM_PRETTY_PRINT_TEST(Border)
  TORCH_ENUM_PRETTY_PRINT_TEST(Reflection)
  TORCH_ENUM_PRETTY_PRINT_TEST(RNN_TANH)
  TORCH_ENUM_PRETTY_PRINT_TEST(RNN_RELU)
  TORCH_ENUM_PRETTY_PRINT_TEST(LSTM)
  TORCH_ENUM_PRETTY_PRINT_TEST(GRU)
}
```