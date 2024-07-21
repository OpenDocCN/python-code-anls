# `.\pytorch\torch\csrc\jit\runtime\register_ops_utils.h`

```py
#pragma once

// 包含 ATen 库的上下文头文件
#include <ATen/Context.h>
// 包含 C10 核心设备类型头文件
#include <c10/core/DeviceType.h>
// 包含 Torch 自动求导头文件
#include <torch/csrc/autograd/autograd.h>
// 包含 Torch 自动求导边缘头文件
#include <torch/csrc/autograd/edge.h>
// 包含 Torch 自动求导函数头文件
#include <torch/csrc/autograd/function.h>
// 包含 Torch 自动求导生成的变量工厂头文件
#include <torch/csrc/autograd/generated/variable_factories.h>
// 包含 Torch 自动求导变量头文件
#include <torch/csrc/autograd/variable.h>
// 包含 Torch JIT 编译单元头文件
#include <torch/csrc/jit/api/compilation_unit.h>
// 包含 Torch JIT 模块头文件
#include <torch/csrc/jit/api/module.h>
// 包含 Torch JIT 前端错误报告头文件
#include <torch/csrc/jit/frontend/error_report.h>
// 包含 Torch JIT IR 头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 移动设备操作注册工具头文件
#include <torch/csrc/jit/mobile/register_ops_common_utils.h>
// 包含 Torch JIT 运行时自定义运算符头文件
#include <torch/csrc/jit/runtime/custom_operator.h>
// 包含 Torch JIT 运行时图执行器头文件
#include <torch/csrc/jit/runtime/graph_executor.h>
// 包含 Torch JIT 运行时 JIT 异常头文件
#include <torch/csrc/jit/runtime/jit_exception.h>
// 包含 Torch JIT 运行时日志记录头文件
#include <torch/csrc/jit/runtime/logging.h>
// 包含 Torch JIT 运行时操作符头文件
#include <torch/csrc/jit/runtime/operator.h>
// 包含 Torch JIT 运行时打印处理器头文件
#include <torch/csrc/jit/runtime/print_handler.h>
// 包含 Torch JIT 运行时性能记录头文件
#include <torch/csrc/jit/runtime/profiling_record.h>
// 包含 Torch JIT 运行时变长参数函数头文件
#include <torch/csrc/jit/runtime/vararg_functions.h>
// 包含 Torch JIT 序列化 pickle 头文件
#include <torch/csrc/jit/serialization/pickle.h>

// 包含 ATen 扩展工具头文件
#include <ATen/ExpandUtils.h>
// 包含 ATen 并行工具头文件
#include <ATen/Parallel.h>
// 包含 ATen 尺寸包装工具头文件
#include <ATen/WrapDimUtils.h>
// 包含 ATen 核心字典头文件
#include <ATen/core/Dict.h>
// 包含 ATen 核心生成器头文件
#include <ATen/core/Generator.h>
// 包含 ATen 核心 IValue 头文件
#include <ATen/core/ivalue.h>
// 包含 C10 核心设备头文件
#include <c10/core/Device.h>
// 包含 C10 核心线程池头文件
#include <c10/core/thread_pool.h>
// 包含 C10 核心小向量头文件
#include <c10/util/SmallVector.h>
// 包含 C10 核心范围头文件
#include <c10/util/irange.h>

// Torch JIT 命名空间
namespace torch::jit {

// 从模式中生成别名分析类型
constexpr inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// 保守的别名分析类型
constexpr inline c10::AliasAnalysisKind aliasAnalysisConservative() {
  return c10::AliasAnalysisKind::CONSERVATIVE;
}

// 特殊情况的别名分析类型
constexpr inline c10::AliasAnalysisKind aliasAnalysisSpecialCase() {
  return c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}

// 创建结果列表的模板函数，返回一个空的 C10 列表
template <class T>
c10::List<T> make_result_list(const TypePtr& elemType) {
  return c10::List<T>();
}

// 特化的 make_result_list 函数，返回一个空的 IValue 列表
template <>
c10::impl::GenericList make_result_list<IValue>(const TypePtr& elemType);

// 对于一个数，采用 Python 的 round 函数进行偶数舍入
// 特别处理了处于两个整数中间的情况
inline double round_to_even(double a) {
  return a - std::floor(a) == 0.5 ? (std::round(a * 0.5) * 2.0) : std::round(a);
}

// 检查隐式的张量到数值转换
// 张量不能有梯度，必须是零维的，如果目标是整数，则源必须是整数类型
void checkImplicitTensorToNum(const at::Tensor& t, bool toInt);

// 不使用的 floordiv 函数，实现整数的整除
static C10_UNUSED int64_t floordiv(int64_t a, int64_t b) {
  if (b == 0) {
    throw std::runtime_error("division by 0");
  }
  if ((a > 0) == (b > 0)) {
    // 简单情况，两者同号
    return a / b;
  } else {
    // 使用 lldiv 函数计算 a 除以 b 的商和余数
    auto r = lldiv(a, b);
    // 如果余数不为 0，则返回商减去 1；否则返回商本身
    return (r.rem) ? r.quot - 1 : r.quot;
`
}
# 定义一个检查 double 类型值是否在有效范围内的函数，作为 API 的一部分
TORCH_API void checkDoubleInRange(double a);

# 声明 floor 函数，返回 double 类型值向下取整的 int64_t 类型结果，并且检查输入值是否在有效范围内
static C10_UNUSED int64_t floor(double a) {
  # 调用 checkDoubleInRange 函数检查输入值
  checkDoubleInRange(a);
  # 使用标准库的 std::floor 函数计算并返回向下取整后的结果
  return std::floor(a);
}

# 声明 ceil 函数，返回 double 类型值向上取整的 int64_t 类型结果，并且检查输入值是否在有效范围内
static C10_UNUSED int64_t ceil(double a) {
  # 调用 checkDoubleInRange 函数检查输入值
  checkDoubleInRange(a);
  # 使用标准库的 std::ceil 函数计算并返回向上取整后的结果
  return std::ceil(a);
}

# 声明 gcd 函数，计算两个 int64_t 类型整数的最大公约数
static C10_UNUSED int64_t gcd(int64_t a, int64_t b) {
  # 使用 while 循环，直到 b 为 0 时结束循环
  while (b != 0) {
    # 计算 a 除以 b 的余数
    int64_t r = a % b;
    # 将 b 的值赋给 a，将余数赋给 b
    a = b;
    b = r;
  }
  # 返回最大公约数，使用 std::abs 确保返回值为非负
  return std::abs(a);
}

# 声明计算部分乘积的函数
int64_t partProduct(int n, int m);

# 声明循环计算的函数，接受参数 n、p 和 r，p 和 r 为引用参数
void loop(int n, int64_t& p, int64_t& r);

# 声明计算整数 v 的 n-1 个比特数之和的函数
int nminussumofbits(int v);

# 声明计算整数 n 的阶乘的函数
int64_t factorial(int n);

# 定义将度数转换为弧度的常量
static const double degToRad = std::acos(-1.0) / 180.0;
# 定义将弧度转换为度数的常量
static const double radToDeg = 180.0 / std::acos(-1.0);

# 声明将角度转换为弧度的函数
double degrees(double x);
# 声明将弧度转换为角度的函数
double radians(double x);

# 注释说明：将 Python 索引（可能为负数）转换为可用于 C++ 容器的索引

# 函数模板，获取 C++ 容器中的指定索引的元素，支持负数索引
template <typename T>
decltype(auto) getItem(const c10::List<T>& list, int64_t idx) {
  # 获取列表的大小
  const int64_t list_size = list.size();
  # 将 Python 索引转换为 C++ 索引
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  # 检查索引是否越界，抛出异常
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  # 返回指定索引的元素
  return list.get(normalized_idx);
}

# 函数模板，设置 C++ 容器中指定索引的元素，支持负数索引
template <typename T>
void setItem(const c10::List<T>& list, int64_t idx, T&& value) {
  # 获取列表的大小
  const int64_t list_size = list.size();
  # 将 Python 索引转换为 C++ 索引
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  # 检查索引是否越界，抛出异常
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  # 设置指定索引的元素为传入的值
  list.set(normalized_idx, std::forward<T>(value));
}

# 声明向列表中添加元素的函数
void listAppend(Stack& stack);

# 声明反转列表的函数
void listReverse(Stack& stack);

# 函数模板，计算两个列表的最小列表
template <typename T>
void minList(Stack& stack) {
  # 从栈中弹出两个列表
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  # 计算两个列表中较小的大小
  size_t min_size = std::min(a.size(), b.size());
  # 遍历较小的大小，比较两个列表对应元素
  for (const auto i : c10::irange(min_size)) {
    if (a[i] == b[i]) {
      continue;
    }

    # 如果存在不同元素，将较小的列表推入栈中，并返回
    push(stack, a[i] < b[i] ? a : b);
    return;
  }

  # 如果所有对应元素相等，推入较小长度的列表
  push(stack, b.size() < a.size() ? b : a);
}

# 函数模板，计算两个列表的最大列表
template <typename T>
void maxList(Stack& stack) {
  # 从栈中弹出两个列表
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  # 计算两个列表中较小的大小
  size_t min_size = std::min(a.size(), b.size());
  # 遍历较小的大小，比较两个列表对应元素
  for (const auto i : c10::irange(min_size)) {
    if (a[i] == b[i]) {
      continue;
    }

    # 如果存在不同元素，将较大的列表推入栈中，并返回
    push(stack, a[i] > b[i] ? a : b);
    return;
  }

  # 如果所有对应元素相等，推入较长长度的列表
  push(stack, b.size() > a.size() ? b : a);
}

# 声明从列表中弹出元素的实现函数，接受栈和空列表时的提示信息
void listPopImpl(Stack& stack, const char* empty_message);

# 声明从列表中弹出元素的函数
void listPop(Stack& stack);

# 声明清空列表的函数
void listClear(Stack& stack);

# 声明删除列表元素的函数
void listDelete(Stack& stack);

# 声明插入元素到列表中的函数
void listInsert(Stack& stack);

# 函数模板，移除列表中的指定元素
template <typename T>
void listRemove(Stack& stack) {
  # 从栈中弹出元素和列表
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  # 查找元素在列表中的位置
  auto pos = std::find(list.begin(), list.end(), elem);

  # 如果找到该元素，移除该元素
  if (pos != list.end()) {
    list.erase(pos);
  } else {
    # 如果未找到，抛出错误
    AT_ERROR("list.remove(x): x not in list");
  }
}

# 函数模板的结束
template <typename T>
// 计算列表中的最小元素，并将其压入堆栈
void listMin(Stack& stack) {
  // 从堆栈中弹出一个列表对象，并转换为指定类型的列表
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  // 获取列表的长度
  size_t list_size = list.size();
  // 如果列表为空，则抛出异常
  if (list_size == 0) {
    throw std::runtime_error("min() arg is an empty sequence");
  }

  // 初始化最小元素为列表的第一个元素
  T min_elem = list[0];
  // 遍历列表，找到最小的元素
  for (const auto i : c10::irange(1, list_size)) {
    T elem = list[i];
    min_elem = elem < min_elem ? elem : min_elem;
  }

  // 将最小元素压入堆栈
  stack.push_back(min_elem);
}

template <typename T>
// 计算列表中的最大元素，并将其压入堆栈
void listMax(Stack& stack) {
  // 从堆栈中弹出一个列表对象，并转换为指定类型的列表
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  // 获取列表的长度
  size_t list_size = list.size();
  // 如果列表为空，则抛出异常
  if (list_size == 0) {
    throw std::runtime_error("max() arg is an empty sequence");
  }

  // 初始化最大元素为列表的第一个元素
  T max_elem = list[0];
  // 遍历列表，找到最大的元素
  for (const auto i : c10::irange(1, list_size)) {
    T elem = list[i];
    max_elem = elem > max_elem ? elem : max_elem;
  }

  // 将最大元素压入堆栈
  stack.push_back(max_elem);
}

template <>
// 在特化为 at::Tensor 类型的情况下，移除 listRemove 函数的声明
void listRemove<at::Tensor>(Stack& stack);

template <typename T>
// 查找元素在列表中的索引，并将索引值压入堆栈
void listIndex(Stack& stack) {
  // 从堆栈中弹出一个元素，并转换为指定类型
  T elem = pop(stack).to<T>();
  // 从堆栈中弹出一个列表对象，并转换为指定类型的列表
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  // 在列表中查找元素的位置
  auto pos = std::find(list.begin(), list.end(), elem);

  // 如果找到元素，则将其索引值压入堆栈；否则抛出异常
  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }
}

template <>
// 在特化为 at::Tensor 类型的情况下，移除 listIndex 函数的声明
void listIndex<at::Tensor>(Stack& stack);

template <typename T>
// 计算列表中特定元素的数量，并将其压入堆栈
void listCount(Stack& stack) {
  // 从堆栈中弹出一个元素，并转换为指定类型
  T elem = pop(stack).to<T>();
  // 从堆栈中弹出一个列表对象，并转换为指定类型的列表
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  // 计算列表中特定元素的数量
  const int64_t count = std::count(list.begin(), list.end(), elem);
  // 将计数值压入堆栈
  push(stack, count);
}

template <>
// 在特化为 at::Tensor 类型的情况下，移除 listCount 函数的声明
void listCount<at::Tensor>(Stack& stack);

// 扩展列表功能，暂无具体实现
void listExtend(Stack& stack);

// 复制列表，暂无具体实现
void listCopy(Stack& stack);

// 选择列表中的元素，暂无具体实现
void listSelect(Stack& stack);

// 获取列表的长度，并将其压入堆栈
void listLen(Stack& stack);

template <typename T>
// 比较两个列表是否相等，并将结果压入堆栈
void listEq(Stack& stack) {
  // 从堆栈中弹出两个列表对象，并转换为指定类型的列表
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  // 将比较结果压入堆栈
  push(stack, a == b);
}

template <typename T>
// 比较两个列表是否不相等，并将结果压入堆栈
void listNe(Stack& stack) {
  // 从堆栈中弹出两个列表对象，并转换为指定类型的列表
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  // 将比较结果压入堆栈
  push(stack, a != b);
}

// 比较两个 tensor 列表是否相等的辅助函数
inline bool tensor_list_equal(
    const c10::List<at::Tensor>& a,
    const c10::List<at::Tensor>& b) {
  // 如果列表长度不相等，则直接返回 false
  if (a.size() != b.size()) {
    return false;
  }

  // 逐个比较两个列表中的元素是否相等
  for (const auto i : c10::irange(a.size())) {
    const at::Tensor& a_element = a[i];
    const at::Tensor& b_element = b[i];
    // 使用 eq() 方法比较两个元素是否相等，并将结果转换为布尔值
    const auto cmp_result = a_element.eq(b_element);
    // 如果有任何一个元素不相等，则返回 false
    if (!at::native::is_nonzero(cmp_result)) {
      return false;
    }
  }

  // 如果所有元素都相等，则返回 true
  return true;
}

// 在特化为 at::Tensor 类型的情况下，移除 listEq 函数的声明
template <>
void listEq<at::Tensor>(Stack& stack);

// 在特化为 at::Tensor 类型的情况下，移除 listNe 函数的声明
template <>
void listNe<at::Tensor>(Stack& stack);

// 列表转换为列表，暂无具体实现
void listList(Stack& stack);

template <typename T>
void listContains(Stack& stack) {
  auto key = pop(stack).to<T>();  // 从栈中弹出一个元素，并转换为类型 T，作为查找的关键字
  auto list = pop(stack).to<c10::List<T>>();  // 从栈中弹出一个元素，并转换为类型 c10::List<T>，作为待查找的列表
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const T& item : list) {  // 遍历列表中的每一个元素 item
    if (item == key) {  // 如果当前元素等于关键字 key
      push(stack, true);  // 将 true 推入栈中，表示列表包含该关键字
      return;  // 结束函数
    }
  }
  push(stack, false);  // 若未找到关键字，将 false 推入栈中
}

void listAdd(Stack& stack);  // 未提供注释的函数声明

void listInplaceAdd(Stack& stack);  // 未提供注释的函数声明

void listMulIntLeftInPlace(Stack& stack);  // 未提供注释的函数声明

void listMulIntLeft(Stack& stack);  // 未提供注释的函数声明

void listMulIntRight(Stack& stack);  // 未提供注释的函数声明

void listSlice(Stack& stack);  // 未提供注释的函数声明

template <typename T>
void listSort(Stack& stack) {
  bool reverse = pop(stack).toBool();  // 从栈中弹出一个元素，并转换为布尔值，表示排序是否倒序
  c10::List<T> list = pop(stack).to<c10::List<T>>();  // 从栈中弹出一个元素，并转换为类型 c10::List<T>，作为待排序的列表
  std::sort(list.begin(), list.end(), [reverse](const T& a, const T& b) {
    // FBCode errors without this check - "strict weak ordering"
    // TODO: remove when possible, since it just slows down
    // sorting and doesn't do anything useful
    if (a == b) {  // 如果两个元素相等
      return false;  // 返回 false，以避免影响排序
    }
    return (a < b) != reverse;  // 按照给定的排序方式进行比较
  });
}

// Specialization for at::Tensor
template <>
void listSort<at::Tensor>(Stack& stack);  // 未提供注释的模板特化声明

template <typename T>
void listCopyAndSort(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();  // 从栈中弹出一个元素，并转换为类型 c10::List<T>，作为待复制和排序的列表
  auto list_copied = list.copy();  // 复制列表
  std::sort(list_copied.begin(), list_copied.end(), [](const T& a, const T& b) {
    // "strict weak ordering" issue - see other sort
    if (a == b) {  // 如果两个元素相等
      return false;  // 返回 false，以避免影响排序
    }
    return a < b;  // 按照给定的排序方式进行比较
  });
  push(stack, list_copied);  // 将排序后的列表推入栈中
}

// Specialization for at::Tensor
template <>
void listCopyAndSort<at::Tensor>(Stack& stack);  // 未提供注释的模板特化声明

void listSetItem(Stack& stack);  // 未提供注释的函数声明

struct OperatorGeneratorArgs {
  const char* schema_str;  // 操作符生成器的模式字符串
  bool isOperationCreator;  // 标志，指示是否为操作生成器
  union {
    void (*operation)(Stack&);  // 操作函数指针
    OperationCreator operationCreator;  // 操作生成器对象
  };
  AliasAnalysisKind aliasAnalysis;  // 别名分析类型

  explicit constexpr OperatorGeneratorArgs(
      torch::detail::SelectiveStr<true> schema_str,
      void (*op)(Stack&),
      AliasAnalysisKind aa)
      : schema_str(schema_str),
        isOperationCreator(false),
        operation(op),
        aliasAnalysis(aa) {}

  explicit constexpr OperatorGeneratorArgs(
      torch::detail::SelectiveStr<true> schema_str,
      OperationCreator opCreator,
      AliasAnalysisKind aa)
      : schema_str(schema_str),
        isOperationCreator(true),
        operationCreator(opCreator),
        aliasAnalysis(aa) {}

  template <typename... Args>
  explicit constexpr OperatorGeneratorArgs(
      torch::detail::SelectiveStr<false>,
      Args...)
      : schema_str(nullptr),
        isOperationCreator(false),
        operation(nullptr),
        aliasAnalysis(AliasAnalysisKind::INTERNAL_SPECIAL_CASE) {}
};

#define DEFINE_GENERIC_BINARY_OP(  // 未提供注释的宏定义开始
    // 定义宏 TORCH_SELECTIVE_SCHEMA，将 aten_op 转换为选择性的架构字符串
    aten_op, op, int_float_result, complex_result)                            \
  // 使用 OperatorGeneratorArgs 定义操作符生成器参数
      OperatorGeneratorArgs(                                                      \
      // 使用 TORCH_SELECTIVE_SCHEMA 定义整数参数操作符的架构
      TORCH_SELECTIVE_SCHEMA(#aten_op                                         \
                             ".int_int(int a, int b) -> " #int_float_result), \
      // lambda 函数定义整数参数操作符的实现
      [](Stack& stack) {                                                      \
        int64_t a, b;                                                         \
        pop(stack, a, b);                                                     \
        push(stack, op);                                                      \
      },                                                                      \
      // 从架构中获取别名分析信息
      aliasAnalysisFromSchema()),                                             \
      // OperatorGeneratorArgs 定义浮点数参数操作符生成器参数
      OperatorGeneratorArgs(                                                  \
          // 使用 TORCH_SELECTIVE_SCHEMA 定义浮点数参数操作符的架构
          TORCH_SELECTIVE_SCHEMA(                                             \
              #aten_op                                                        \
              ".float_float(float a, float b) -> " #int_float_result),        \
          // lambda 函数定义浮点数参数操作符的实现
          [](Stack& stack) {                                                  \
            double a, b;                                                      \
            pop(stack, a, b);                                                 \
            push(stack, op);                                                  \
          },                                                                  \
          // 从架构中获取别名分析信息
          aliasAnalysisFromSchema()),                                         \
      // OperatorGeneratorArgs 定义复数参数操作符生成器参数
      OperatorGeneratorArgs(                                                  \
          // 使用 TORCH_SELECTIVE_SCHEMA 定义复数参数操作符的架构
          TORCH_SELECTIVE_SCHEMA(                                             \
              #aten_op                                                        \
              ".complex_complex(complex a, complex b) -> " #complex_result),  \
          // lambda 函数定义复数参数操作符的实现
          [](Stack& stack) {                                                  \
            c10::complex<double> a, b;                                        \
            pop(stack, a, b);                                                 \
            push(stack, op);                                                  \
          },                                                                  \
          // 从架构中获取别名分析信息
          aliasAnalysisFromSchema())
// 定义通用操作的宏，用于基本数值操作
#define DEFINE_GENERIC_OP(aten_op, int_op, float_op, int_result, float_result) \
  OperatorGeneratorArgs(                                                       \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a, int b) -> " #int_result),   \
      [](Stack& stack) {                                                       \
        int64_t a, b;                                                          \
        pop(stack, a, b);                                                      \
        push(stack, int_op);                                                   \
      },                                                                       \
      aliasAnalysisFromSchema()),                                              \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA(                                              \
              #aten_op ".float(float a, float b) -> " #float_result),          \
          [](Stack& stack) {                                                   \
            double a, b;                                                       \
            pop(stack, a, b);                                                  \
            push(stack, float_op);                                             \
          },                                                                   \
          aliasAnalysisFromSchema())


这段代码定义了一个宏 `DEFINE_GENERIC_OP`，用于生成多个操作的实现。具体注释如下：

- `# 定义通用操作的宏，用于基本数值操作`
- `# 根据传入的操作符和操作数类型，生成不同类型的操作函数`
- `# 第一个操作：整数操作`
- `# 从堆栈中弹出两个整数，执行整数操作，将结果推回堆栈`
- `# 第二个操作：浮点数操作`
- `# 从堆栈中弹出两个浮点数，执行浮点数操作，将结果推回堆栈`
- `# 这两个操作分别针对整数和浮点数执行不同的操作，并且依赖于操作符的具体定义和别名分析`
# 定义宏 `DEFINE_INT_FLOAT_OP`，用于生成处理整数和浮点数操作数的运算符生成器参数
#define DEFINE_INT_FLOAT_OP(aten_op, op, result)                            \
  OperatorGeneratorArgs(                                                    \
      TORCH_SELECTIVE_SCHEMA(#aten_op                                       \
                             ".int_float(int a, float b) -> " #result),     \
      [](Stack& stack) {                                                    \
        int64_t a;                                                          \
        double b;                                                           \
        pop(stack, a, b);                                                   \
        push(stack, op);                                                    \
      },                                                                    \
      aliasAnalysisFromSchema()),                                           \
      OperatorGeneratorArgs(                                                \
          TORCH_SELECTIVE_SCHEMA(#aten_op                                   \
                                 ".float_int(float a, int b) -> " #result), \
          [](Stack& stack) {                                                \
            double a;                                                       \
            int64_t b;                                                      \
            pop(stack, a, b);                                               \
            push(stack, op);                                                \
          },                                                                \
          aliasAnalysisFromSchema())

# 定义宏 `DEFINE_INT_OP`，用于生成处理整数操作数的运算符生成器参数
#define DEFINE_INT_OP(aten_op, op)                                  \
  OperatorGeneratorArgs(                                            \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a, int b) -> int"), \
      [](Stack& stack) {                                            \
        int64_t a, b;                                               \
        pop(stack, a, b);                                           \
        push(stack, op); /* NOLINT(hicpp-signed-bitwise) */         \
      },                                                            \
      aliasAnalysisFromSchema())

# 定义宏 `DEFINE_STR_CMP_OP`，用于生成处理字符串比较操作的运算符生成器参数
#define DEFINE_STR_CMP_OP(aten_op, op)                               \
  OperatorGeneratorArgs(                                             \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".str(str a, str b) -> bool"), \
      [](Stack& stack) {                                             \
        auto b = pop(stack).toStringRef();                           \
        auto a = pop(stack).toStringRef();                           \
        push(stack, op);                                             \
      },                                                             \
      aliasAnalysisFromSchema())

# 定义一个原始操作，处理标量操作数。
# 必须在整数/浮点数变体之后注册此重载，以避免意外地将标量参数陷入隐式转换
# 定义一个宏，生成通用的避免冲突的标量二元操作函数
#define DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION_GENERIC(          \
    aten_op, int_op, float_op, result, string_val)                \
  OperatorGeneratorArgs(                                          \
      TORCH_SELECTIVE_SCHEMA(#aten_op string_val                  \
                             "(Scalar a, Scalar b) -> " #result), \
      [](Stack& stack) {                                          \
        IValue x, y;                                              \  # 声明变量 x 和 y 用于存储操作数
        pop(stack, x, y);                                         \  # 从堆栈中弹出两个值 x 和 y
        if (x.isDouble()) {                                       \  # 如果 x 是 double 类型
          if (y.isDouble()) {                                     \  # 并且 y 也是 double 类型
            double a = x.toDouble();                              \  # 将 x 转换为 double 类型
            double b = y.toDouble();                              \  # 将 y 转换为 double 类型
            push(stack, float_op);                                \  # 将 float_op 结果推送回堆栈
          } else {                                                \  # 如果 y 不是 double 类型
            double a = x.toDouble();                              \  # 将 x 转换为 double 类型
            int64_t b = y.toInt();                                \  # 将 y 转换为 int64_t 类型
            push(stack, float_op);                                \  # 将 float_op 结果推送回堆栈
          }                                                       \
        } else {                                                  \  # 如果 x 不是 double 类型
          if (y.isDouble()) {                                     \  # 并且 y 是 double 类型
            int64_t a = x.toInt();                                \  # 将 x 转换为 int64_t 类型
            double b = y.toDouble();                              \  # 将 y 转换为 double 类型
            push(stack, float_op);                                \  # 将 float_op 结果推送回堆栈
          } else {                                                \  # 如果 y 不是 double 类型
            int64_t a = x.toInt();                                \  # 将 x 转换为 int64_t 类型
            int64_t b = y.toInt();                                \  # 将 y 转换为 int64_t 类型
            push(stack, int_op);                                  \  # 将 int_op 结果推送回堆栈
          }                                                       \
        }                                                         \
      },                                                          \
      aliasAnalysisFromSchema())

# 定义标量二元操作函数，使用通用的避免冲突的宏
#define DEFINE_SCALAR_BINARY_OP(aten_op, int_op, float_op, result) \
  DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION_GENERIC(                 \
      aten_op, int_op, float_op, result, "")

# 定义避免冲突的标量二元操作函数
#define DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(   \
    aten_op, int_op, float_op, result)             \
  DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION_GENERIC( \
      aten_op, int_op, float_op, result, ".Scalar_Scalar")

# 定义通用的二元操作函数
#define DEFINE_BINARY_OP(aten_op, op)             \
  DEFINE_GENERIC_OP(aten_op, op, op, int, float), \  # 定义通用操作的宏，整数和浮点数类型
      DEFINE_INT_FLOAT_OP(aten_op, op, float),    \  # 定义整数和浮点数操作的宏
      DEFINE_SCALAR_BINARY_OP(aten_op, op, op, Scalar)  # 定义标量二元操作的宏

# 定义浮点数二元操作函数
#define DEFINE_BINARY_FLOAT_OP(aten_op, op)         \
  DEFINE_GENERIC_OP(aten_op, op, op, float, float), \  # 定义通用操作的宏，两个浮点数类型
      DEFINE_INT_FLOAT_OP(aten_op, op, float),      \  # 定义整数和浮点数操作的宏
      DEFINE_SCALAR_BINARY_OP(aten_op, op, op, float)  # 定义标量二元操作的宏
# 定义比较运算符的宏，生成多个运算符的函数定义
#define DEFINE_COMPARISON_OP(aten_op, op)             \
  # 在代码中展开的效果为：DEFINE_GENERIC_OP(aten_op, op, op, bool, bool),     \
      DEFINE_INT_FLOAT_OP(aten_op, op, bool),         \
      DEFINE_SCALAR_BINARY_OP(aten_op, op, op, bool), \
      DEFINE_STR_CMP_OP(aten_op, op)

# 定义针对整数的一元操作宏，生成整数类型操作的函数定义
#define DEFINE_UNARY_INT_OP(aten_op, op, result)                  \
  # 在代码中展开的效果为：OperatorGeneratorArgs(                                          \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a) -> " #result), \
      [](Stack& stack) {                                          \
        int64_t a;                                                \
        pop(stack, a);                                            \
        push(stack, op);                                          \
      },                                                          \
      aliasAnalysisFromSchema())

# 定义针对浮点数的一元操作宏，生成浮点数类型操作的函数定义
#define DEFINE_UNARY_FLOAT_OP(aten_op, op, result)                    \
  # 在代码中展开的效果为：OperatorGeneratorArgs(                                              \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".float(float a) -> " #result), \
      [](Stack& stack) {                                              \
        double a;                                                     \
        pop(stack, a);                                                \
        push(stack, op);                                              \
      },                                                              \
      aliasAnalysisFromSchema())

# 定义通用的一元操作宏，根据参数类型选择生成整数或浮点数操作的函数定义
#define DEFINE_UNARY_OP(aten_op, op, int_result, float_result)            \
  # 在代码中展开的效果为：DEFINE_UNARY_INT_OP(aten_op, op, int_result),                           \
      DEFINE_UNARY_FLOAT_OP(aten_op, op, float_result),                   \
      OperatorGeneratorArgs(                                              \
          TORCH_SELECTIVE_SCHEMA(#aten_op ".Scalar(Scalar a) -> Scalar"), \
          [](Stack& stack) {                                              \
            IValue x;                                                     \
            pop(stack, x);                                                \
            if (x.isDouble()) {                                           \
              double a = x.toDouble();                                    \
              push(stack, static_cast<float_result>(op));                 \
            } else {                                                      \
              int64_t a = x.toInt();                                      \
              push(stack, static_cast<int_result>(op));                   \
            }                                                             \
          },                                                              \
          aliasAnalysisFromSchema())
# 定义一个宏，用于生成布尔类型的操作符，例如 and、or 等
#define DEFINE_BOOL_OP(aten_op, op)                                     \
  OperatorGeneratorArgs(                                                \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".bool(bool a, bool b) -> bool"), \  # 定义操作符的 Torch 脚本格式字符串，指定参数和返回类型
      [](Stack& stack) {                                                \  # Lambda 函数定义开始，接收一个名为 stack 的栈参数
        bool a, b;                                                      \  # 声明布尔型变量 a 和 b
        pop(stack, a, b);                                               \  # 从栈中弹出两个布尔型参数并赋值给 a 和 b
        push(stack, op);                                                \  # 将操作的结果推入栈中
      },                                                                \  # Lambda 函数定义结束
      aliasAnalysisFromSchema())                                         \  # 调用 aliasAnalysisFromSchema 函数，进行别名分析

# 定义一个宏，用于生成字符串类型的操作符，例如字符串连接等
#define DEFINE_STRING_OP(op_name, string_op, result)                    \
  OperatorGeneratorArgs(                                                \
      TORCH_SELECTIVE_SCHEMA(#op_name ".str(str a, str b) ->" #result), \  # 定义操作符的 Torch 脚本格式字符串，指定参数和返回类型
      [](Stack& stack) {                                                \  # Lambda 函数定义开始，接收一个名为 stack 的栈参数
        auto b = pop(stack).toStringRef();                              \  # 从栈中弹出字符串类型参数 b，并转换为引用
        auto a = pop(stack).toStringRef();                              \  # 从栈中弹出字符串类型参数 a，并转换为引用
        push(stack, string_op);                                         \  # 将操作的结果推入栈中
      },                                                                \  # Lambda 函数定义结束
      aliasAnalysisFromSchema())                                         \  # 调用 aliasAnalysisFromSchema 函数，进行别名分析

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

# 定义一个宏，用于生成复数类型的一元操作符，例如复数的绝对值等
#define DEFINE_UNARY_COMPLEX_OP(aten_op, op, result)                      \
  OperatorGeneratorArgs(                                                  \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".complex(complex a) -> " #result), \  # 定义操作符的 Torch 脚本格式字符串，指定参数和返回类型
      [](Stack& stack) {                                                  \  # Lambda 函数定义开始，接收一个名为 stack 的栈参数
        c10::complex<double> a;                                           \  # 声明双精度复数类型变量 a
        pop(stack, a);                                                    \  # 从栈中弹出一个复数类型参数 a
        push(stack, op);                                                  \  # 将操作的结果推入栈中
      },                                                                  \  # Lambda 函数定义结束
      aliasAnalysisFromSchema())                                           \  # 调用 aliasAnalysisFromSchema 函数，进行别名分析

// Some complex unary ops (like abs, angle) return real valued output, but most
// other unary ops return complex valued output. So, this macro is used in the
// former case where we can explicitly pass complex_result_cast argument, which
// is set to c10::complex<float> in the macro `DEFINE_UNARY_OP_WITH_COMPLEX`
// defined below.

# 定义一个宏，用于生成一元操作符，返回结果可能是实数或复数类型
#define DEFINE_UNARY_OP_WITH_COMPLEX_CAST(                                \
    aten_op,                                                              \
    op,                                                                   \
    int_result,                                                           \
    float_result,                                                         \
    complex_result_cast)                                                  \
    complex_result,                                                       \
    complex_result_cast)                                                  \
  DEFINE_UNARY_INT_OP(aten_op, op, int_result),                           \
      DEFINE_UNARY_FLOAT_OP(aten_op, op, float_result),                   \
      DEFINE_UNARY_COMPLEX_OP(aten_op, op, complex_result),               \
      OperatorGeneratorArgs(                                              \
          TORCH_SELECTIVE_SCHEMA(#aten_op ".Scalar(Scalar a) -> Scalar"), \
          [](Stack& stack) {                                              \
            IValue x;                                                     \
            pop(stack, x);                                                \
            if (x.isDouble()) {                                           \
              double a = x.toDouble();                                    \
              // 如果输入值是双精度浮点数，将其转换为 float_result 类型后推入栈
              push(stack, static_cast<float_result>(op));                 \
            } else if (x.isComplexDouble()) {                             \
              c10::complex<double> a = x.toComplexDouble();               \
              // 如果输入值是复数（双精度），将其转换为 complex_result_cast 类型后推入栈
              push(stack, static_cast<complex_result_cast>(op));          \
            } else {                                                      \
              int64_t a = x.toInt();                                      \
              // 如果输入值是整数，将其转换为 int_result 类型后推入栈
              push(stack, static_cast<int_result>(op));                   \
            }                                                             \
          },                                                              \
          aliasAnalysisFromSchema())
// 定义一个宏，用于生成支持复数操作的一元运算符
#define DEFINE_UNARY_OP_WITH_COMPLEX(aten_op, op, int_result, float_result) \
  // 调用宏DEFINE_UNARY_OP_WITH_COMPLEX_CAST，传入aten_op、op、int_result、float_result参数
  DEFINE_UNARY_OP_WITH_COMPLEX_CAST(                                        \
      aten_op, op, int_result, float_result, complex, c10::complex<double>)

// 定义一个宏，用于生成支持复数操作的通用运算符
#define DEFINE_GENERIC_OP_WITH_COMPLEX(                                       \
    aten_op,                                                                  \
    int_op,                                                                   \
    float_op,                                                                 \
    complex_op,                                                               \
    int_result,                                                               \
    float_result,                                                             \
    complex_result)                                                           \
  // 第一个OperatorGeneratorArgs，接收整型参数，调用int_op进行操作
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a, int b) -> " #int_result),  \
      [](Stack& stack) {                                                      \
        int64_t a, b;                                                         \
        pop(stack, a, b);                                                     \
        push(stack, int_op);                                                  \
      },                                                                      \
      aliasAnalysisFromSchema()),                                             \
  // 第二个OperatorGeneratorArgs，接收复数参数，调用complex_op进行操作
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA(                                                 \
          #aten_op ".complex(complex a, complex b) -> " #complex_result),     \
      [](Stack& stack) {                                                      \
        c10::complex<double> a, b;                                            \
        pop(stack, a, b);                                                     \
        push(stack, complex_op);                                              \
      },                                                                      \
      aliasAnalysisFromSchema()),                                             \
  // 第三个OperatorGeneratorArgs，接收浮点数参数，调用float_op进行操作
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA(                                                 \
          #aten_op ".float(float a, float b) -> " #float_result),             \
      [](Stack& stack) {                                                      \
        double a, b;                                                          \
        pop(stack, a, b);                                                     \
        push(stack, float_op);                                                \
      },                                                                      \
      aliasAnalysisFromSchema())
// 定义宏 `DEFINE_INT_COMPLEX_OP`，用于生成两种操作符的函数定义
#define DEFINE_INT_COMPLEX_OP(aten_op, op, result)                          \
  // 第一个操作符：整数和复数相结合的操作
  OperatorGeneratorArgs(                                                    \
      // 根据 Torch selective schema 定义操作的签名和结果类型
      TORCH_SELECTIVE_SCHEMA(#aten_op                                       \
                             ".int_complex(int a, complex b) -> " #result), \
      // Lambda 函数，从栈中弹出整数和复数，执行操作 `op`，并将结果推回栈
      [](Stack& stack) {                                                    \
        int64_t a;                                                          \
        c10::complex<double> b;                                             \
        pop(stack, a, b);                                                   \
        push(stack, op);                                                    \
      },                                                                    \
      // 使用 schema 中的别名分析
      aliasAnalysisFromSchema()),                                           \
  // 第二个操作符：复数和整数相结合的操作
  OperatorGeneratorArgs(                                                    \
      // 根据 Torch selective schema 定义操作的签名和结果类型
      TORCH_SELECTIVE_SCHEMA(                                               \
          #aten_op ".complex_int(complex a, int b) -> " #result),           \
      // Lambda 函数，从栈中弹出复数和整数，执行操作 `op`，并将结果推回栈
      [](Stack& stack) {                                                    \
        c10::complex<double> a;                                             \
        int64_t b;                                                          \
        pop(stack, a, b);                                                   \
        push(stack, op);                                                    \
      },                                                                    \
      // 使用 schema 中的别名分析
      aliasAnalysisFromSchema())
# 定义一个宏 `DEFINE_FLOAT_COMPLEX_OP`，用于生成两个特定的操作符
#define DEFINE_FLOAT_COMPLEX_OP(aten_op, op, result)                      \
  OperatorGeneratorArgs(                                                  \
      TORCH_SELECTIVE_SCHEMA(                                             \
          #aten_op ".float_complex(float a, complex b) -> " #result),     \
      [](Stack& stack) {                                                  \
        double a;                                                         \
        c10::complex<double> b;                                           \
        pop(stack, a, b);                                                 \
        push(stack, op);                                                  \
      },                                                                  \
      aliasAnalysisFromSchema()),                                         \
      OperatorGeneratorArgs(                                              \
          TORCH_SELECTIVE_SCHEMA(                                         \
              #aten_op ".complex_float(complex a, float b) -> " #result), \
          [](Stack& stack) {                                              \
            c10::complex<double> a;                                       \
            double b;                                                     \
            pop(stack, a, b);                                             \
            push(stack, op);                                              \
          },                                                              \
          aliasAnalysisFromSchema())

# 定义一个宏 `DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_AVOID_COLLISION_GENERIC`，用于避免与通用整型复数对的冲突
#define DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_AVOID_COLLISION_GENERIC( \

# 定义一个宏 `DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_WITHOUT_INT_COMPLEX_PAIR`，用于处理不包含整型复数对的情况
#define DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_WITHOUT_INT_COMPLEX_PAIR(     \
    aten_op, int_op, float_op, complex_op, result)                         \
  OperatorGeneratorArgs(                                                   \
      TORCH_SELECTIVE_SCHEMA(#aten_op "(Scalar a, Scalar b) -> " #result), \
      [](Stack& stack) {                                                   \
        IValue x, y;                                                       \  // 声明变量 x 和 y，用于存储从栈中弹出的操作数
        pop(stack, x, y);                                                  \  // 从栈中弹出两个操作数，分别存入变量 x 和 y
        if (x.isComplexDouble()) {                                         \  // 检查变量 x 是否为复数类型
          c10::complex<double> a = x.toComplexDouble();                    \  // 将 x 转换为复数类型并赋值给变量 a
          if (y.isComplexDouble()) {                                       \  // 检查变量 y 是否为复数类型
            c10::complex<double> b = y.toComplexDouble();                  \  // 将 y 转换为复数类型并赋值给变量 b
            push(stack, complex_op);                                       \  // 将 complex_op 推入栈中
          } else if (y.isDouble()) {                                       \  // 如果变量 y 是双精度浮点数
            double b = y.toDouble();                                       \  // 将 y 转换为双精度浮点数并赋值给变量 b
            push(stack, complex_op);                                       \  // 将 complex_op 推入栈中
          }                                                                \
        } else if (x.isDouble()) {                                         \  // 如果变量 x 是双精度浮点数
          double a = x.toDouble();                                         \  // 将 x 转换为双精度浮点数并赋值给变量 a
          if (y.isComplexDouble()) {                                       \  // 如果变量 y 是复数类型
            c10::complex<double> b = y.toComplexDouble();                  \  // 将 y 转换为复数类型并赋值给变量 b
            push(stack, complex_op);                                       \  // 将 complex_op 推入栈中
          } else if (y.isDouble()) {                                       \  // 如果变量 y 是双精度浮点数
            double b = y.toDouble();                                       \  // 将 y 转换为双精度浮点数并赋值给变量 b
            push(stack, float_op);                                         \  // 将 float_op 推入栈中
          } else {                                                         \  // 如果变量 y 是整数
            int64_t b = y.toInt();                                         \  // 将 y 转换为整数并赋值给变量 b
            push(stack, float_op);                                         \  // 将 float_op 推入栈中
          }                                                                \
        } else {                                                           \  // 如果变量 x 是整数
          int64_t a = x.toInt();                                           \  // 将 x 转换为整数并赋值给变量 a
          if (y.isDouble()) {                                              \  // 如果变量 y 是双精度浮点数
            double b = y.toDouble();                                       \  // 将 y 转换为双精度浮点数并赋值给变量 b
            push(stack, float_op);                                         \  // 将 float_op 推入栈中
          } else if (y.isInt()) {                                          \  // 如果变量 y 是整数
            int64_t b = y.toInt();                                         \  // 将 y 转换为整数并赋值给变量 b
            push(stack, int_op);                                           \  // 将 int_op 推入栈中
          }                                                                \
        }                                                                  \
      },                                                                   \  // 结束 lambda 函数定义
      aliasAnalysisFromSchema())                                            \  // 使用 schema 进行别名分析
#define DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX(                   \
    aten_op, int_op, float_op, complex_op, result)              \
  DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_AVOID_COLLISION_GENERIC( \
      aten_op, int_op, float_op, complex_op, result, "")
  // 定义一个宏，用于生成标量与复数之间的二元运算函数，避免命名冲突

#define DEFINE_BINARY_OP_WITH_COMPLEX(aten_op, op)                          \
  DEFINE_GENERIC_OP_WITH_COMPLEX(aten_op, op, op, op, int, float, complex), \
      DEFINE_INT_COMPLEX_OP(aten_op, op, complex),                          \
      DEFINE_FLOAT_COMPLEX_OP(aten_op, op, complex),                        \
      DEFINE_INT_FLOAT_OP(aten_op, op, float),                              \
      DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX(aten_op, op, op, op, Scalar)
  // 定义一个宏，用于生成复数间的二元运算函数，包括通用、整数-复数、浮点数-复数、整数-浮点数和标量-复数的运算

#define DEFINE_COMPARISON_OP_WITH_COMPLEX(aten_op, op)                   \
  DEFINE_GENERIC_OP_WITH_COMPLEX(aten_op, op, op, op, bool, bool, bool), \
      DEFINE_INT_FLOAT_OP(aten_op, op, bool),                            \
      DEFINE_FLOAT_COMPLEX_OP(aten_op, op, bool),                        \
      DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_WITHOUT_INT_COMPLEX_PAIR(     \
          aten_op, op, op, op, bool),                                    \
      DEFINE_STR_CMP_OP(aten_op, op)
  // 定义一个宏，用于生成复数间的比较运算函数，包括通用、整数-浮点数、浮点数-复数、标量-复数和字符串-比较运算的函数

TORCH_API at::Generator make_generator_for_device(
    c10::Device device,
    std::optional<int64_t> seed = c10::nullopt);
  // 在 Torch API 中声明一个函数，用于根据设备生成一个随机数生成器，可选地接受一个种子参数

} // namespace torch::jit
  // 定义了 torch::jit 命名空间的结束
```