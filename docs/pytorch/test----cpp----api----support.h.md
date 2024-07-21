# `.\pytorch\test\cpp\api\support.h`

```
#pragma once
// 防止头文件被多次包含的预处理指令

#include <test/cpp/common/support.h>
// 包含自定义测试支持头文件

#include <gtest/gtest.h>
// 包含 Google 测试框架头文件

#include <ATen/TensorIndexing.h>
// 包含 ATen 张量索引头文件

#include <c10/util/Exception.h>
// 包含 C10 异常处理头文件

#include <torch/nn/cloneable.h>
// 包含 Torch 克隆相关头文件

#include <torch/types.h>
// 包含 Torch 类型定义头文件

#include <torch/utils.h>
// 包含 Torch 实用工具头文件

#include <string>
// 包含字符串处理相关头文件

#include <utility>
// 包含实用程序头文件

namespace torch {
namespace test {

// 允许在不创建新类的情况下使用容器，用于实验性实现
class SimpleContainer : public nn::Cloneable<SimpleContainer> {
 public:
  void reset() override {}
  // 重置函数的覆盖实现

  template <typename ModuleHolder>
  ModuleHolder add(
      ModuleHolder module_holder,
      std::string name = std::string()) {
    // 添加模块到容器，并注册模块名称
    return Module::register_module(std::move(name), module_holder);
  }
};

struct SeedingFixture : public ::testing::Test {
  SeedingFixture() {
    torch::manual_seed(0);
    // 使用种子 0 进行 Torch 手动种子设置
  }
};

struct WarningCapture : public WarningHandler {
  WarningCapture() : prev_(WarningUtils::get_warning_handler()) {
    // 构造函数，捕获警告并设置警告处理程序
    WarningUtils::set_warning_handler(this);
  }

  ~WarningCapture() override {
    // 析构函数，恢复先前的警告处理程序
    WarningUtils::set_warning_handler(prev_);
  }

  const std::vector<std::string>& messages() {
    return messages_;
    // 返回警告消息向量的引用
  }

  std::string str() {
    return c10::Join("\n", messages_);
    // 将警告消息向量连接为单个字符串
  }

  void process(const c10::Warning& warning) override {
    messages_.push_back(warning.msg());
    // 处理捕获的警告信息并存储在消息向量中
  }

 private:
  WarningHandler* prev_;
  // 指向先前警告处理程序的指针
  std::vector<std::string> messages_;
  // 存储警告消息的向量
};

inline bool pointer_equal(at::Tensor first, at::Tensor second) {
  return first.data_ptr() == second.data_ptr();
  // 比较两个张量的数据指针是否相等
}

// 此函数与 torch/testing/_internal/common_utils.py 中的 TestCase.assertEqual 函数中的
// `isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)` 分支相对应
inline void assert_tensor_equal(
    at::Tensor a,
    at::Tensor b,
    bool allow_inf = false) {
  ASSERT_TRUE(a.sizes() == b.sizes());
  // 断言：确保两个张量具有相同的尺寸

  if (a.numel() > 0) {
    if (a.device().type() == torch::kCPU &&
        (a.scalar_type() == torch::kFloat16 ||
         a.scalar_type() == torch::kBFloat16)) {
      // 如果是 CPU 上的半精度或 BF16 张量，转换为 float32
      a = a.to(torch::kFloat32);
    }
    if (a.device().type() == torch::kCUDA &&
        a.scalar_type() == torch::kBFloat16) {
      // 如果是 CUDA 上的 BF16 张量，转换为 float32
      a = a.to(torch::kFloat32);
    }
    b = b.to(a);

    if ((a.scalar_type() == torch::kBool) !=
        (b.scalar_type() == torch::kBool)) {
      TORCH_CHECK(false, "Was expecting both tensors to be bool type.");
      // 如果一个张量是 bool 类型而另一个不是，则报错
      // TORCH_CHECK 是 Torch 中的断言宏
    }
    } else {
      // 如果 a 和 b 均为布尔类型，需要将它们转换为整型以进行准确的减法运算
      if (a.scalar_type() == torch::kBool && b.scalar_type() == torch::kBool) {
        a = a.to(torch::kInt);
        b = b.to(torch::kInt);
      }

      // 计算张量 a 和 b 的差
      auto diff = a - b;
      
      // 如果张量 a 是浮点型
      if (a.is_floating_point()) {
        // 检查 NaN 是否出现在相同的位置
        auto nan_mask = torch::isnan(a);
        ASSERT_TRUE(torch::equal(nan_mask, torch::isnan(b)));
        // 将差值中 NaN 所在位置的元素设为 0
        diff.index_put_({nan_mask}, 0);
        
        // 如果允许处理无穷大数值
        if (allow_inf) {
          // 检查是否存在无穷大数值
          auto inf_mask = torch::isinf(a);
          // 检查无穷大数值的符号是否相同
          auto inf_sign = inf_mask.sign();
          ASSERT_TRUE(torch::equal(inf_sign, torch::isinf(b).sign()));
          // 将差值中无穷大数值所在位置的元素设为 0
          diff.index_put_({inf_mask}, 0);
        }
      }
      
      // 如果差值是有符号的并且不是 int8 类型，取其绝对值
      if (diff.is_signed() && diff.scalar_type() != torch::kInt8) {
        diff = diff.abs();
      }
      
      // 计算差值的最大绝对误差
      auto max_err = diff.max().item<double>();
      // 断言最大绝对误差不超过 1e-5
      ASSERT_LE(max_err, 1e-5);
    }
// This function checks if two tensors are not equal and raises an assertion error if they are.
// It ensures that the tensors have the same sizes and are not equal element-wise.
inline void assert_tensor_not_equal(at::Tensor x, at::Tensor y) {
  if (x.sizes() != y.sizes()) { // Check if sizes of x and y are different
    return; // Return early if sizes are different
  }
  ASSERT_GT(x.numel(), 0); // Assert that the number of elements in x is greater than zero
  y = y.type_as(x); // Convert y to the same type as x
  y = x.is_cuda() ? y.to({torch::kCUDA, x.get_device()}) : y.cpu(); // Move y to CUDA if x is on CUDA, otherwise to CPU
  auto nan_mask = x != x; // Create a mask for NaN values in x
  if (torch::equal(nan_mask, y != y)) { // Check if both tensors have the same NaN pattern
    auto diff = x - y; // Compute the absolute difference between x and y
    if (diff.is_signed()) { // If the difference tensor is signed
      diff = diff.abs(); // Take the absolute value of the difference tensor
    }
    diff.index_put_({nan_mask}, 0); // Set elements in diff corresponding to NaNs in x to 0
    // Use `item()` to extract the maximum absolute error from the difference tensor
    auto max_err = diff.max().item<double>();
    ASSERT_GE(max_err, 1e-5); // Assert that the maximum error is greater than or equal to 1e-5
  }
}

// This function counts the occurrences of a substring `substr` in a string `str`.
inline int count_substr_occurrences(
    const std::string& str,
    const std::string& substr) {
  int count = 0; // Initialize the counter for substring occurrences
  size_t pos = str.find(substr); // Find the first occurrence of substr in str

  while (pos != std::string::npos) { // Loop while substr is found in str
    count++; // Increment the occurrence counter
    pos = str.find(substr, pos + substr.size()); // Find the next occurrence of substr in str
  }

  return count; // Return the total count of substring occurrences
}

// This structure provides a RAII (Resource Acquisition Is Initialization) guard that changes
// the default data type upon construction and restores it upon destruction, ensuring thread safety.
struct AutoDefaultDtypeMode {
  static std::mutex default_dtype_mutex; // Static mutex to synchronize access across threads

  // Constructor that changes the default data type and locks the mutex
  AutoDefaultDtypeMode(c10::ScalarType default_dtype)
      : prev_default_dtype(
            torch::typeMetaToScalarType(torch::get_default_dtype())) {
    default_dtype_mutex.lock(); // Acquire the mutex lock
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(default_dtype)); // Set the default data type
  }

  // Destructor that unlocks the mutex and restores the previous default data type
  ~AutoDefaultDtypeMode() {
    default_dtype_mutex.unlock(); // Release the mutex lock
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(prev_default_dtype)); // Restore the previous default data type
  }

  c10::ScalarType prev_default_dtype; // Previous default data type stored for restoration
};

// This function asserts that a tensor `x` has a specific creation meta `creation_meta`.
inline void assert_tensor_creation_meta(
    torch::Tensor& x,
    torch::autograd::CreationMeta creation_meta) {
  auto autograd_meta = x.unsafeGetTensorImpl()->autograd_meta(); // Get the autograd metadata of tensor x
  TORCH_CHECK(autograd_meta); // Ensure autograd metadata exists
  auto view_meta =
      static_cast<torch::autograd::DifferentiableViewMeta*>(autograd_meta); // Cast autograd metadata to DifferentiableViewMeta
  TORCH_CHECK(view_meta->has_bw_view()); // Ensure the view meta has a backward view
  ASSERT_EQ(view_meta->get_creation_meta(), creation_meta); // Assert that the creation meta matches the expected value
}
```