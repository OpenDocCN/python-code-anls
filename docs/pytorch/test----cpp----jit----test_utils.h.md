# `.\pytorch\test\cpp\jit\test_utils.h`

```
#pragma once

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/testing/file_check.h>

// 命名空间定义
namespace {
// 字符串修剪函数，去除首尾空格和多余的换行符和空格
static inline void trim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
  s.erase(
      std::find_if(
          s.rbegin(),
          s.rend(),
          [](unsigned char ch) { return !std::isspace(ch); })
          .base(),
      s.end());
  // 移除多余的换行符
  for (size_t i = 0; i < s.size(); ++i) {
    while (i < s.size() && s[i] == '\n') {
      s.erase(i, 1);
    }
  }
  // 移除连续的空格
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == ' ') {
      while (i + 1 < s.size() && s[i + 1] == ' ') {
        s.erase(i + 1, 1);
      }
    }
  }
}
} // namespace

// 定义断言宏，验证执行给定语句时是否抛出异常，并检查异常信息是否包含指定的子字符串
#define ASSERT_THROWS_WITH_MESSAGE(statement, substring)             \
  try {                                                              \
    (void)statement;                                                 \
    FAIL();                                                          \
  } catch (const std::exception& e) {                                \
    std::string substring_s(substring);                              \
    trim(substring_s);                                               \
    auto exception_string = std::string(e.what());                   \
    trim(exception_string);                                          \
    // 断言异常信息中是否包含指定的子字符串，如果不包含则输出详细错误信息
    ASSERT_NE(exception_string.find(substring_s), std::string::npos) \
        << " Error was: \n"                                          \
        << exception_string;                                         \
  }

// Torch 命名空间下的 JIT 模块
namespace torch {
namespace jit {

// 引入别名
using tensor_list = std::vector<at::Tensor>;
using namespace torch::autograd;

// 创建一个 Stack 对象，用于在解释器中存储变量列表
Stack createStack(std::vector<at::Tensor>&& list);

// 验证两个 tensor_list 中的张量是否全部接近（值相等），用于测试
void assertAllClose(const tensor_list& a, const tensor_list& b);

// 运行解释器 interp，并返回输出的张量列表
std::vector<at::Tensor> run(
    InterpreterState& interp,
    const std::vector<at::Tensor>& inputs);

// 运行梯度计算，返回输入张量和梯度的对应列表
std::pair<tensor_list, tensor_list> runGradient(
    Gradient& grad_spec,
    tensor_list& tensors_in,
    tensor_list& tensor_grads_in);

// 构建 LSTM 模型的计算图
std::shared_ptr<Graph> build_lstm();
// 构建移动端导出分析图
std::shared_ptr<Graph> build_mobile_export_analysis_graph();
// 构建带输出参数的移动端导出图
std::shared_ptr<Graph> build_mobile_export_with_out();
// 构建带可变参数的移动端导出分析图
std::shared_ptr<Graph> build_mobile_export_analysis_graph_with_vararg();
// 构建嵌套的移动端导出分析图
std::shared_ptr<Graph> build_mobile_export_analysis_graph_nested();
// 构建非常量移动端导出分析图
std::shared_ptr<Graph> build_mobile_export_analysis_graph_non_const();

// 使用张量 x 的函数定义
at::Tensor t_use(at::Tensor x);
// 定义张量 x 的函数使用
at::Tensor t_def(at::Tensor x);

// 给定输出与期望张量之间的差异，检查差异是否在相对公差范围内，用于精度匹配
# 定义函数签名，检查两个张量的相对误差是否符合条件
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs);

# 定义函数签名，检查两个张量是否几乎相等
bool almostEqual(const at::Tensor& a, const at::Tensor& b);

# 定义函数签名，检查两个张量是否完全相等
bool exactlyEqual(const at::Tensor& a, const at::Tensor& b);

# 定义函数签名，检查两个张量向量是否完全相等
bool exactlyEqual(
    const std::vector<at::Tensor>& a,
    const std::vector<at::Tensor>& b);

# 定义函数签名，运行给定图形的计算图，并返回结果张量向量
std::vector<at::Tensor> runGraph(
    std::shared_ptr<Graph> graph,
    const std::vector<at::Tensor>& inputs);

# 定义函数签名，执行 LSTM 操作，返回输出张量和细胞状态张量
std::pair<at::Tensor, at::Tensor> lstm(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor w_ih,
    at::Tensor w_hh);

} // namespace jit
} // namespace torch
```