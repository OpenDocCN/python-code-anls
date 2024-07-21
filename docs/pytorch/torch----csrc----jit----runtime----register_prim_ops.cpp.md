# `.\pytorch\torch\csrc\jit\runtime\register_prim_ops.cpp`

```
// 包含头文件 <ATen/autocast_mode.h>，提供了自动混合精度计算的支持
// 包含头文件 <ATen/core/Generator.h>，定义了随机数生成器的接口
// 包含头文件 <c10/util/Optional.h>，提供了可选值的封装类模板
// 包含头文件 <c10/util/irange.h>，定义了迭代范围的辅助函数
// 包含头文件 <torch/csrc/jit/mobile/promoted_prim_ops.h>，提供了移动端推广基本操作的支持
// 包含头文件 <torch/csrc/jit/runtime/custom_operator.h>，定义了自定义运算符的运行时支持
// 包含头文件 <torch/csrc/jit/runtime/operator.h>，提供了运算符的运行时实现
// 包含头文件 <torch/csrc/jit/runtime/register_ops_utils.h>，提供了操作注册的辅助函数
// 包含头文件 <torch/csrc/jit/runtime/slice_indices_adjust.h>，提供了切片索引调整的支持
// 包含头文件 <torch/library.h>，引用了 Torch 库的全部功能

// 包含 C++ 标准库的头文件
#include <algorithm>  // 包含了各种算法操作
#include <bitset>     // 提供了位集合操作的类模板
#include <cctype>     // 提供了字符分类函数
#include <cmath>      // 包含了数学函数
#include <exception>  // 定义了异常类和异常处理函数
#include <fstream>    // 提供了文件输入输出操作的类
#include <iostream>   // 提供了标准输入输出流对象
#include <limits>     // 提供了各种数据类型的极限值
#include <memory>     // 提供了动态内存管理的功能
#include <mutex>      // 提供了互斥量和锁的类和函数
#include <ostream>    // 定义了输出流对象
#include <stdexcept>  // 定义了一组标准异常类
#include <string>     // 提供了字符串处理的功能
#include <typeinfo>   // 提供了类型信息的功能
#include <unordered_map>  // 提供了哈希表实现的无序映射容器
#include <unordered_set>  // 提供了哈希表实现的无序集合容器
#include <utility>         // 提供了各种实用工具函数
#include <vector>          // 提供了向量容器类模板

namespace torch::jit {

namespace {

// 定义了字符串切片函数，支持指定起始位置、结束位置和步长
std::string stringSlice(
    std::string string,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  int64_t start_val = start.has_value() ? start.value() : INT64_MAX;  // 获取起始位置，若未指定则取最大值
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;        // 获取结束位置，若未指定则取最大值

  // 调用 slice_indices_adjust 函数调整起始位置和结束位置，并返回有效元素数目
  const int64_t num_vals =
      slice_indices_adjust(string.size(), &start_val, &end_val, step);

  int64_t i = start_val;
  std::string result = "";  // 初始化结果字符串
  // 遍历切片范围内的元素，按步长增加索引，将字符添加到结果字符串中
  for (const auto j : c10::irange(num_vals)) {
    (void)j;  // 抑制未使用变量的警告
    result += string[i];
    i += step;
  }

  return result;  // 返回切片后的字符串
}

// 定义了按空白符分割字符串的函数，类似于 Python 的 split() 方法
c10::List<std::string> splitNoneSeparator(const std::string& string) {
  c10::List<std::string> splits;  // 创建用于存储分割结果的列表

  // 定义包含各种空白符的字符串
  std::string whitespaces =
      " \t\n\r\r\n\v\x0b\f\x0c\x1c\x1d\x1e\x85\u2028\u2029";
  std::string::size_type prev_pos = 0;
  std::string::size_type pos = 0;

  // 在字符串中查找空白符，进行分割
  while ((pos = string.find_first_of(whitespaces, pos)) != std::string::npos) {
    auto substr = string.substr(prev_pos, pos - prev_pos);  // 获取非空白符部分的子串
    if (!substr.empty()) {
      splits.emplace_back(substr);  // 将非空字符串添加到分割结果列表中
    }
    pos++;        // 移动到下一个字符
    prev_pos = pos;  // 更新起始位置
  }
  if (prev_pos != string.size()) {
    splits.emplace_back(string.substr(prev_pos));  // 添加最后一个分割结果
  }

  return splits;  // 返回分割后的字符串列表
}

// 检查给定元组类型是否可排序，并将原因描述写入到提供的字符串流中
bool isSortableTupleType(
    const TupleTypePtr& tuple_type,
    std::stringstream& why_not) {
  // 遍历元组类型中的每个元素类型
  for (const TypePtr& ele_type : tuple_type->containedTypes()) {
    switch (ele_type->kind()) {
      // 根据元素类型的种类进行不同的处理
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
        // 如果是基本类型（整数、布尔、浮点数、字符串、张量），继续下一个元素的处理
        continue;
      case TypeKind::TupleType:
        // 如果是元组类型，检查是否可排序，如果不可排序，则返回 false
        if (!isSortableTupleType(ele_type->expect<TupleType>(), why_not)) {
          return false;
        }
        // 继续下一个元素的处理
        continue;
      case TypeKind::ClassType:
        // 如果是类类型，检查其是否符合排序的对象模式
        if (!c10::checkObjectSortSchema(
                ele_type->expect<ClassType>(), why_not)) {
          return false;
        }
        // 继续下一个元素的处理
        continue;
      default:
        // 如果元素类型无法识别，返回不可排序的原因
        why_not << "Contained elements in " << *tuple_type
                << " are not sortable. Only Int, Bool, Float, String, Tensor, "
                << "a User Defined Class with __lt__ method defined or Tuples "
                << "of aforementioned types can be sorted.";
        return false;
    }
  }

  // 所有元素都可以排序，则返回 true
  return true;
}

# 检查是否可以对对象或元组的列表进行排序
bool isSortableListOfObjectsOrTuples(
    c10::List<IValue>& ivalues,    # 输入的对象或元组列表
    std::stringstream& why_not) {  # 错误信息输出流
  if (ivalues.empty()) {           # 如果列表为空，可排序
    return true;
  }

  auto type = ivalues.get(0).type();  # 获取第一个元素的类型作为基准类型
  // 假设列表具有同质类型，使用第一个元素确定最佳排序方法。
  // 如果将来需要支持列表内的异质类型，则排序需要运行时检查排序功能。
  const size_t n = ivalues.size();    # 列表长度
  for (const auto i : c10::irange(n)) {  # 遍历列表
    const IValue& v = ivalues.get(i);    # 获取当前元素
    auto curr_type = v.type();           # 当前元素的类型
    if (*curr_type != *type) {           # 如果当前元素类型与基准类型不同
      why_not << "Only values of same type can be compared. "  # 输出错误信息
              << "Found " << type->repr_str() << " and "
              << curr_type->repr_str();
      return false;                      # 返回不可排序
    }
  }

  if (auto tuple_type = type->cast<TupleType>()) {  # 如果是元组类型
    return isSortableTupleType(tuple_type, why_not);  # 检查元组是否可排序
  }

  if (auto class_type = type->cast<ClassType>()) {  # 如果是类类型
    return c10::checkObjectSortSchema(class_type, why_not) != nullptr;  # 检查类对象的排序模式
  }

  // 基本类型如张量、整数、浮点数、布尔值、字符串在此方法不进行检查，
  // 因为它们应该已经使用 listSort<T> 匹配到特定的 aten::sort 内核。
  why_not << "Only list of Tensors, ints, floats, bools, strs, "
          << "a User Defined Class that defines the __lt__ compare method "
          << "or Tuples of aforementioned types can be sorted, got list of "
          << type->repr_str() << "\n";   # 输出不支持排序的类型信息
  return false;                        # 返回不可排序
}

# 排序操作模板，处理栈中的元素
template <bool has_reverse_arg, bool copy_return_list>
void sort_op(Stack& stack) {
  bool reverse = has_reverse_arg ? pop(stack).toBool() : false;  # 是否逆序排序
  auto g_list = pop(stack).toList();  # 弹出列表作为要排序的对象

  if (copy_return_list) {           # 如果需要复制返回列表
    g_list = g_list.copy();         # 复制列表
  }

  if (!g_list.empty()) {            # 如果列表不为空
    std::stringstream error_str;    # 错误信息流
    if (!isSortableListOfObjectsOrTuples(g_list, error_str)) {  # 检查列表是否可排序
      throw std::runtime_error(error_str.str());  # 抛出运行时错误，附带错误信息
    }

    c10::IValueComparator comparator;  # 比较器
    if (reverse) {                    # 如果逆序排序
      comparator = c10::getGreaterThanComparator(g_list.get(0));  # 获取大于比较器
    } else {                          # 否则
      comparator = c10::getLessThanComparator(g_list.get(0));     # 获取小于比较器
    }
    std::sort(g_list.begin(), g_list.end(), comparator);  # 使用比较器对列表进行排序
  }

  if (copy_return_list) {           # 如果需要复制返回列表
    push(stack, g_list);            # 推送排序后的列表回栈
  }
}

# 幂运算的包装器模板，检查特殊情况后调用标准 pow 函数
template <typename T, typename U>
auto powWrapper(T a, U b) {
  TORCH_CHECK(
      !(a == 0.0 && b < 0.0), "0.0 cannot be raised to a negative power")  # 检查底数为零且指数为负的特殊情况
  return pow(a, b);  # 调用标准 pow 函数计算幂次方
}

# 静态常量运算符生成参数的向量
static const std::vector<OperatorGeneratorArgs> opGenArgs{
    OperatorGeneratorArgs(  # 操作符生成参数对象
        TORCH_SELECTIVE_SCHEMA("aten::str(t elem) -> str"),  # 操作的概要描述
        [](Stack& stack) {                                 # Lambda 函数，执行字符串转换操作
          std::stringstream ss;                            # 字符串流
          ss << pop(stack);                                # 弹出栈顶元素并转换为字符串
          push(stack, ss.str());                           # 将转换后的字符串推回栈
        },
        aliasAnalysisFromSchema()),  # 根据模式生成别名分析信息
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::list(str t) -> str[]"),
        [](Stack& stack) {
          auto str = pop(stack).toStringRef();
          c10::List<std::string> chars;
          chars.reserve(str.size());
          for (auto c : str) {
            chars.push_back(std::string(1, c));
          }
          push(stack, std::move(chars));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::cpu(Tensor(a) self) -> Tensor(a|b)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.cpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::numpy_T.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.numpy_T());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::matrix_H.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.matrix_H());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mT.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.mT());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mH.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.mH());
        },
        aliasAnalysisFromSchema()),

    // only used internally in range() translation
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__range_length(int lo, int hi, int step) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t lo, hi, step;
          pop(stack, lo, hi, step);
          // 检查步长是否为零，如果是则抛出运行时错误
          if (step == 0) {
            throw std::runtime_error("range() arg 3 must not be zero");
          }
          // 根据步长正负和起始值与结束值的关系计算范围长度
          if (step > 0 && lo < hi) {
            push(stack, 1 + (hi - 1 - lo) / step);
          } else if (step < 0 && lo > hi) {
            push(stack, 1 + (lo - 1 - hi) / (0 - step));
          } else {
            push(stack, 0);
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__derive_index(int index, int start, int step) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t index, start, step;
          pop(stack, index, start, step);
          // 根据索引、起始值和步长计算派生索引值
          push(stack, start + index * step);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::TupleUnpack(Any tup) -> ..."),
        [](Stack& stack) { tupleUnpack(stack); },
        aliasAnalysisSpecialCase()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::unchecked_cast(t x) -> t"),
        noop,
        aliasAnalysisSpecialCase()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::IntImplicit(Tensor a) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 Tensor 对象 a
          at::Tensor a;
          pop(stack, a);
          // 检查将 Tensor 隐式转换为 int 是否安全
          checkImplicitTensorToNum(a, /*to int*/ true);
          // 将 Tensor 中的 int 值压入堆栈
          push(stack, a.item<int64_t>());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ComplexImplicit(Tensor a) -> complex"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 Tensor 对象 a
          at::Tensor a;
          pop(stack, a);
          // 检查将 Tensor 隐式转换为 complex 是否安全
          checkImplicitTensorToNum(a, /*to int*/ false);
          // 将 Tensor 中的 complex 值压入堆栈
          push(stack, a.item<c10::complex<double>>());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::FloatImplicit(Tensor a) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 Tensor 对象 a
          at::Tensor a;
          pop(stack, a);
          // 检查将 Tensor 隐式转换为 float 是否安全
          checkImplicitTensorToNum(a, /*to int*/ false);
          // 将 Tensor 中的 float 值压入堆栈
          push(stack, a.item<double>());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ScalarImplicit(Tensor a) -> Scalar"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 Tensor 对象 a
          at::Tensor a;
          pop(stack, a);
          // 检查将 Tensor 隐式转换为 Scalar 是否安全
          checkImplicitTensorToNum(a, /*to int*/ false);
          // 将 Tensor 中的 Scalar 值压入堆栈
          push(stack, a.item());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.Tensor(Tensor a) -> bool"),
        boolTensor,
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.int(int a) -> bool"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 int64_t 类型的整数
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t i;
          pop(stack, i);
          // 将整数转换为布尔值并压入堆栈
          push(stack, (bool)i);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.float(float a) -> bool"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 double 类型的浮点数
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double d;
          pop(stack, d);
          // 将浮点数转换为布尔值并压入堆栈
          push(stack, (bool)d);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.Tensor(Tensor a) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 Tensor 对象 a
          at::Tensor a;
          pop(stack, a);
          // 将 Tensor 中的 int 值压入堆栈
          push(stack, a.item<int64_t>());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.bool(bool a) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 bool 类型的值
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool b;
          pop(stack, b);
          // 将布尔值转换为 int 并压入堆栈
          push(stack, static_cast<int64_t>(b));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.float(float a) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个浮点数，并将其转换为整数后压入堆栈
          double d;
          pop(stack, d);
          push(stack, static_cast<int64_t>(d));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.Scalar(Scalar a) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个标量，如果它是整数，则将其推送回堆栈；否则转换为整数后推送
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isInt()) {
            push(stack, std::move(scalar));
          } else {
            push(stack, static_cast<int64_t>(scalar.toScalar().toInt()));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.str(str a) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个字符串，尝试将其解析为整数，并将结果推送回堆栈
          auto s = pop(stack).toString();
          std::string::size_type sz;
          int64_t val = static_cast<int64_t>(std::stoll(s->string(), &sz));
          if (sz == s->string().size()) {
            push(stack, val);
          } else {
            // 如果解析失败，抛出运行时错误
            std::stringstream error_str;
            error_str << "invalid literal for int() "
                      << "with base 10: '" << s->string() << "'";
            throw std::runtime_error(error_str.str());
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.Tensor(Tensor a) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个张量，将其转换为双精度浮点数后推送回堆栈
          at::Tensor a;
          pop(stack, a);
          push(stack, a.item<double>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.Scalar(Scalar a) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个标量，如果它是双精度浮点数，则将其推送回堆栈；
          // 如果是复杂双精度数，则取实部后推送；否则将其转换为浮点数后推送
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isDouble()) {
            push(stack, std::move(scalar));
          } else if (scalar.isComplexDouble()) {
            push(stack, scalar.toComplexDouble().real());
          } else {
            push(stack, static_cast<double>(scalar.toInt()));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.int(int a) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个整数，并将其转换为单精度浮点数后推送回堆栈
          int64_t i;
          pop(stack, i);
          push(stack, (float)i);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.bool(bool a) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个布尔值，并将其转换为单精度浮点数后推送回堆栈
          bool b;
          pop(stack, b);
          push(stack, (float)b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.str(str a) -> float"),
        [](Stack& stack) {
          // 从栈中弹出字符串对象，将其转换为 std::string 类型
          auto s = pop(stack).toString();
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          // 尝试将字符串转换为 double 类型，同时获取转换后有效字符的长度 sz
          std::string::size_type sz;
          double b = std::stod(s->string(), &sz);
          // 如果成功转换的长度与字符串长度相等，将结果推入栈中
          if (sz == s->string().size()) {
            push(stack, b);
          } else {
            // 否则抛出运行时错误，说明无法将字符串转换为 float
            std::stringstream error_str;
            error_str << "could not convert string "
                      << "to float: '" << s->string() << "'";
            throw std::runtime_error(error_str.str());
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Complex.Scalar(Scalar a) -> complex"),
        [](Stack& stack) {
          // 从栈中弹出标量对象
          IValue scalar;
          pop(stack, scalar);
          // 如果标量为复数类型，直接推入栈中
          if (scalar.isComplexDouble()) {
            push(stack, std::move(scalar));
          } else if (scalar.isDouble()) {
            // 如果标量为 double 类型，构造一个实部为标量值的复数并推入栈中
            push(stack, c10::complex<double>(scalar.toDouble(), 0));
          } else {
            // 否则，构造一个实部为整数值的复数并推入栈中
            push(stack, c10::complex<double>(scalar.toInt(), 0));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::Complex.Tensor_Tensor(Tensor a, Tensor b) -> complex"),
        [](Stack& stack) {
          // 从栈中弹出两个张量对象，并构造一个复数对象推入栈中
          at::Tensor a, b;
          pop(stack, a, b);
          push(stack, c10::complex<double>(a.item<double>(), b.item<double>()));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::format(str self, ...) -> str"),
        [](Stack& stack) { aten_format(stack); },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::einsum.sublist(Tensor a, ...) -> Tensor"),
        [](Stack& stack) {
          // 从栈中弹出一个整数，表示输入的张量个数，并调用 einsum 函数处理
          size_t num_inputs = pop(stack).toInt();
          einsum(stack, num_inputs);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::NumToTensor.Scalar(Scalar a) -> Tensor"),
        numToTensorScalar,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::RaiseException(str msg, str? cls=None) -> ()"),
        raiseException,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Size(int[] sizes) -> int[]"),
        [](Stack& stack) {},
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::size(Tensor self) -> int[]"),
        size,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sym_size(Tensor self) -> SymInt[]"),
        sym_size,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::stride(Tensor self) -> int[]"),
        [](Stack& stack) {
          // 从堆栈中弹出一个Tensor，获取其步幅信息，然后压入堆栈
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.strides());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sym_stride(Tensor self) -> SymInt[]"),
        sym_stride,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumName(AnyEnumType enum) -> str"),
        [](Stack& stack) {
          // 从堆栈中弹出一个IValue对象，获取其枚举类型的名称，然后压入堆栈
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->name());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumValue.int(AnyEnumType enum) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个IValue对象，获取其枚举类型的整数值，然后压入堆栈
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::EnumValue.float(AnyEnumType enum) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个IValue对象，获取其枚举类型的浮点数值，然后压入堆栈
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumValue.str(AnyEnumType enum) -> str"),
        [](Stack& stack) {
          // 从堆栈中弹出一个IValue对象，获取其枚举类型的字符串值，然后压入堆栈
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 注意编译器能更精确地理解TupleIndex的类型
        TORCH_SELECTIVE_SCHEMA("prim::TupleIndex(Any tup, int i) -> Any"),
        tupleIndex,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.int_list(int[] a, int[] b) -> bool"),
        listNe<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::unchecked_unwrap_optional(t(a)? optional) -> t(a)"),
        noop,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::device(Tensor a) -> Device"),
        device,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::dtype(Tensor a) -> int"),
        dtype,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::layout(Tensor a) -> Layout"),
        layout,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__not__(bool self) -> bool"),
        _not,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__is__(t1 self, t2 obj) -> bool"),
        is,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__isnot__(t1 self, t2 obj) -> bool"),
        isNot,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::element_size(Tensor self) -> int"),
        [](Stack& stack) {
          // 从栈中弹出一个张量，并计算其元素大小，将结果推入栈顶
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.element_size());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::numel(Tensor self) -> int"),
        [](Stack& stack) {
          // 从栈中弹出一个张量，并计算其元素个数，将结果推入栈顶
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.numel());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dim(Tensor self) -> int"),
        dim,  // 使用预定义的函数 dim 处理张量，返回其维度数
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::get_device(Tensor self) -> int"),
        [](Stack& stack) {
          // 记录函数调用，获取张量所在的设备编号并将其压入栈顶
          RECORD_FUNCTION("get_device", c10::ArrayRef<const c10::IValue>{});
          auto result =
              at::get_device((std::move(peek(stack, 0, 1))).toTensor());
          drop(stack, 1);  // 弹出栈顶元素
          pack(stack, result);  // 将结果打包入栈
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::storage_offset(Tensor self) -> int"),
        [](Stack& stack) {
          // 记录函数调用，获取张量的存储偏移量并将其压入栈顶
          RECORD_FUNCTION("storage_offset", c10::ArrayRef<const c10::IValue>{});
          auto result =
              ((std::move(peek(stack, 0, 1))).toTensor()).storage_offset();
          drop(stack, 1);  // 弹出栈顶元素
          pack(stack, result);  // 将结果打包入栈
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_contiguous(Tensor self) -> bool"),
        [](Stack& stack) {
          // 记录函数调用，检查张量是否连续，并将结果压入栈顶
          RECORD_FUNCTION("is_contiguous", c10::ArrayRef<const c10::IValue>{});
          auto result =
              ((std::move(peek(stack, 0, 1))).toTensor()).is_contiguous();
          drop(stack, 1);  // 弹出栈顶元素
          pack(stack, result);  // 将结果打包入栈
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_contiguous.memory_format(Tensor self, MemoryFormat memory_format) -> bool"),
        [](Stack& stack) {
          auto memory_format = pop(stack).toMemoryFormat();  // 从栈中弹出内存格式
          auto t = pop(stack).toTensor();  // 从栈中弹出张量
          push(stack, t.is_contiguous(memory_format));  // 检查张量是否按照指定内存格式连续，并将结果推入栈顶
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 注意：为了避免触发 "_like" 后缀的测试，意图上添加了额外的 "_format" 后缀
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_strides_like_format(Tensor self, MemoryFormat memory_format) -> bool"),
        [](Stack& stack) {
          auto memory_format = pop(stack).toMemoryFormat();  // 从栈中弹出内存格式
          auto t = pop(stack).toTensor();  // 从栈中弹出张量
          push(stack, t.unsafeGetTensorImpl()->is_strides_like(memory_format));  // 检查张量的实现是否符合给定的内存格式，将结果推入栈顶
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_non_overlapping_and_dense(Tensor self) -> bool"),
        [](Stack& stack) {
          auto t = pop(stack).toTensor();
          // 弹出栈顶的张量，并检查其是否非重叠且密集
          push(stack, t.unsafeGetTensorImpl()->is_non_overlapping_and_dense());
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::select.t(t[](a) list, int idx) -> t(*)"),
        listSelect,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__getitem__.t(t[](a) list, int idx) -> t(*)"),
        listSelect,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)"),
        listAppend,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::reverse.t(t[](a!) self) -> ()"),
        listReverse,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::extend.t(t[](a!) self, t[] other) -> ()"),
        listExtend,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::copy.t(t[](a) self) -> t[]"),
        listCopy,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_set_item.t(t [](a!) l, int idx, t(b -> *) el) -> t[](a!)"),
        listSetItem,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::clear.t(t[](a!) self) -> ()"),
        listClear,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Delete.t(t[](a!) self, int idx) -> ()"),
        listDelete,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::insert.t(t[](a!) self, int idx, t(b -> *) el) -> ()"),
        listInsert,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::pop.t(t[](a!) self, int idx=-1) -> t(*)"),
        listPop,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::add.t(t[] a, t[] b) -> t[]"),
        listAdd,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::add_.t(t[](a!) self, t[] b) -> t[]"),
        listInplaceAdd,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> t[]"),
        listSlice,
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::list.t(t[] l) -> t[]"),
        listList,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul.left_t(t[] l, int n) -> t[]"),
        listMulIntLeft,
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::mul.left_t 操作，接受列表和整数，返回列表
    # 使用 listMulIntLeft 函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul.right_(int n, t[] l) -> t[]"),
        listMulIntRight,
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::mul.right_ 操作，接受整数和列表，返回列表
    # 使用 listMulIntRight 函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul_.t(t[](a!) l, int n) -> t[](a!)"),
        listMulIntLeftInPlace,
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::mul_.t 操作，接受可变列表和整数，返回可变列表
    # 使用 listMulIntLeftInPlace 函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.t(t[] a) -> int"),
        listLen,
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::len.t 操作，接受列表，返回整数
    # 使用 listLen 函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.int_list(int[] a, int[] b) -> bool"),
        listEq<int64_t>,
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::eq.int_list 操作，接受两个整数列表，返回布尔值
    # 使用 listEq<int64_t> 函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.device(Device a, Device b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack).toDevice();
          auto b = pop(stack).toDevice();
          push(stack, a == b);
        },
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::eq.device 操作，接受两个设备对象，返回布尔值
    # 使用 lambda 表达式定义匿名函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.device(Device a, Device b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack).toDevice();
          auto b = pop(stack).toDevice();
          push(stack, a != b);
        },
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::ne.device 操作，接受两个设备对象，返回布尔值
    # 使用 lambda 表达式定义匿名函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.bool(bool a, bool b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, a == b);
        },
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::eq.bool 操作，接受两个布尔值，返回布尔值
    # 使用 lambda 表达式定义匿名函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.bool(bool a, bool b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, a != b);
        },
        aliasAnalysisFromSchema()),
    # 定义操作符生成器参数：对应于 torch 中的 aten::ne.bool 操作，接受两个布尔值，返回布尔值
    # 使用 lambda 表达式定义匿名函数处理该操作
    # 使用 aliasAnalysisFromSchema() 进行别名分析

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_enabled() -> bool"),
        [](Stack& stack) {

        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_enabled() -> bool"),
        # 定义操作符生成器参数：对应于 torch 中的 aten::is_autocast_enabled 操作，返回布尔值
        # 使用 lambda 表达式定义匿名函数处理该操作
        [](Stack& stack) {
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_cuda_enabled() -> bool"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          // 如果是轻量级解释器或者移动端，禁用自动混合精度
          bool enabled = false;
#else
          // 否则，在CUDA设备上检查自动混合精度是否启用
          bool enabled = at::autocast::is_autocast_enabled(at::kCUDA);
#endif
          // 将结果推入堆栈
          push(stack, enabled);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_cpu_enabled() -> bool"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          // 如果是轻量级解释器或者移动端，禁用自动混合精度
          bool enabled = false;
#else
          // 否则，在CPU设备上检查自动混合精度是否启用
          bool enabled = at::autocast::is_autocast_enabled(at::kCPU);
#endif
          // 将结果推入堆栈
          push(stack, enabled);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::get_autocast_dtype(str device_type) -> ScalarType"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          // 轻量级解释器或移动端不支持自动混合精度，返回未定义的数据类型
          at::ScalarType dtype = at::ScalarType::Undefined;
#else
          // 否则，获取给定设备类型的设备对象，然后查询自动混合精度的数据类型
          at::DeviceType device_type =
              at::Device(pop(stack).toStringRef()).type();
          at::ScalarType dtype = at::autocast::get_autocast_dtype(device_type);
#endif
          // 将结果推入堆栈
          push(stack, dtype);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::Uninitialized() -> Any"),
        unInitialized,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::Print(...) -> ()"),
        [](Stack& stack) {
          // 获取打印的输入参数个数
          auto num_inputs = pop(stack).toInt();
          // 构建打印内容的字符串流
          std::stringstream ss;
          bool first = true;
          // 遍历最后的堆栈中的输入参数
          for (const IValue& i : last(stack, num_inputs)) {
            if (!first)
              ss << " ";
            first = false;
            ss << i;
          }
          // 从堆栈中丢弃打印的输入参数
          drop(stack, num_inputs);
          ss << std::endl;
          // 获取打印处理程序，并进行内部断言
          auto* handler = getPrintHandler();
          TORCH_INTERNAL_ASSERT(handler);
          // 调用打印处理程序打印字符串流中的内容
          handler(ss.str());
        },
        aliasAnalysisSpecialCase()),
    // 这是一个替代aten::cat操作，接受可变数量的参数作为输入。
    // 格式：
    //    prim::VarConcat(Tensors..., dim) -> Tensor
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::VarConcat(...) -> Tensor"),
        [](Stack& stack) {
          // 获取输入参数的数量
          auto num_inputs = pop(stack).toInt();
          // 获取维度参数
          auto dim = pop(stack).toInt();
          // 创建输入张量的向量
          std::vector<at::Tensor> inputs(num_inputs - 1);
          // 从堆栈中弹出和填充输入张量的向量
          for (int i = 0; i < num_inputs - 1; ++i) {
            inputs[num_inputs - 2 - i] = pop(stack).toTensor();
          }
          // 将连接后的张量推入堆栈
          push(stack, at::cat(inputs, dim));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::VarStack(...) -> Tensor"),
        [](Stack& stack) {
          // 从栈中弹出一个元素，将其转换为整数，表示输入的数量
          auto num_inputs = pop(stack).toInt();
          // 从栈中弹出一个元素，将其转换为整数，表示维度
          auto dim = pop(stack).toInt();
          // 创建一个空的张量数组，长度为 num_inputs - 1
          std::vector<at::Tensor> inputs(num_inputs - 1);
          // 遍历 num_inputs - 1 次，从栈中弹出张量，并放入 inputs 数组中
          for (int i = 0; i < num_inputs - 1; ++i) {
            inputs[num_inputs - 2 - i] = pop(stack).toTensor();
          }
          // 将输入的张量数组按指定维度进行堆叠，并压入栈中
          push(stack, at::stack(inputs, dim));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::IfThenElse(bool cond, Any(a) x, Any(b) y) -> Any(a|b)"),
        [](Stack& stack) {
          // 从栈中获取条件值，转换为布尔类型
          const auto cond = stack[stack.size() - 3].toBool();
          // 将栈中的 x 或者 y 移动到合适的位置
          stack[stack.size() - 3] =
              std::move(stack[stack.size() - (cond ? 2 : 1)]);
          // 弹出不再需要的栈元素
          stack.pop_back();
          stack.pop_back();
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.enum(AnyEnumType a, AnyEnumType b) -> bool"),
        [](Stack& stack) {
          // 从栈中弹出两个元素，比较它们是否相等，并将结果压入栈中
          IValue x = pop(stack);
          IValue y = pop(stack);
          push(stack, x == y);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.enum(AnyEnumType a, AnyEnumType b) -> bool"),
        [](Stack& stack) {
          // 从栈中弹出两个元素，比较它们是否不相等，并将结果压入栈中
          IValue x = pop(stack);
          IValue y = pop(stack);
          push(stack, x != y);
        },
        aliasAnalysisFromSchema()),
    // We define aten::dequantize in both native_functions.yaml and here,
    // however, aten::dequantize.any defined here overrides
    // aten::dequantize.tensors in native_functions.yaml. The variants here
    // are only for graph mode quantization, and they should be removed once
    // we deprecate graph mode quantization, and use the variants in
    // native_functions.yaml.
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::dequantize.tensor(Tensor qtensor) -> Tensor"),
        [](Stack& stack) {
          // 从栈中弹出一个张量，对其进行去量化操作，并将结果压入栈中
          at::Tensor qtensor;
          pop(stack, qtensor);
          push(stack, at::dequantize(qtensor));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::dequantize.list(Tensor[] qtensors) -> Tensor[]"),
        [](Stack& stack) {
          // 从栈中弹出一个张量数组，对其中每个张量进行去量化操作，并将结果压入栈中
          auto qtensors = pop(stack).toTensorVector();
          push(stack, at::dequantize(qtensors));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dequantize.any(Any tensors) -> Any"),
        [](Stack& stack) { 
          // 对栈中的任意张量进行去量化操作，并将结果压入栈中
          dequantize(stack); 
        },
        aliasAnalysisFromSchema()),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::log, std::log(a), float, float),
    DEFINE_STRING_OP(aten::add, a + b, str),
    DEFINE_COMPARISON_OP_WITH_COMPLEX(aten::eq, a == b),
    DEFINE_COMPARISON_OP_WITH_COMPLEX(aten::ne, a != b),
    DEFINE_GENERIC_OP(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        complex,
        complex),
    
    定义一个名为 `aten::polar` 的通用操作，用于计算极坐标形式下的复数。使用 `c10::polar` 函数将输入 `a` 和 `b` 转换为 `double` 类型后创建复数。
    
    
    DEFINE_INT_FLOAT_OP(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        complex),
    
    定义一个名为 `aten::polar` 的整数和浮点数操作，同样使用 `c10::polar` 函数处理输入 `a` 和 `b`，生成复数。
    
    
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        Scalar),
    
    定义一个避免冲突的标量二元操作 `aten::polar`，使用 `c10::polar` 函数将输入 `a` 和 `b` 转换为 `double` 类型后创建复数。
    
    
    DEFINE_COMPARISON_OP(aten::lt, a < b),
    
    定义一个名为 `aten::lt` 的比较操作，用于检查 `a` 是否小于 `b`。
    
    
    DEFINE_COMPARISON_OP(aten::gt, a > b),
    
    定义一个名为 `aten::gt` 的比较操作，用于检查 `a` 是否大于 `b`。
    
    
    DEFINE_COMPARISON_OP(aten::le, a <= b),
    
    定义一个名为 `aten::le` 的比较操作，用于检查 `a` 是否小于等于 `b`。
    
    
    DEFINE_COMPARISON_OP(aten::ge, a >= b),
    
    定义一个名为 `aten::ge` 的比较操作，用于检查 `a` 是否大于等于 `b`。
    
    
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::add, a + b),
    
    定义一个名为 `aten::add` 的复数加法操作，计算 `a` 和 `b` 的和。
    
    
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::sub, a - b),
    
    定义一个名为 `aten::sub` 的复数减法操作，计算 `a` 减去 `b` 的结果。
    
    
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::mul, a * b),
    
    定义一个名为 `aten::mul` 的复数乘法操作，计算 `a` 和 `b` 的乘积。
    
    
    DEFINE_BOOL_OP(aten::__and__, a && b),
    
    定义一个名为 `aten::__and__` 的布尔与操作，判断 `a` 和 `b` 是否同时为真。
    
    
    DEFINE_BOOL_OP(aten::__or__, a || b),
    
    定义一个名为 `aten::__or__` 的布尔或操作，判断 `a` 或 `b` 是否为真。
    
    
    DEFINE_BOOL_OP(aten::__xor__, a != b),
    
    定义一个名为 `aten::__xor__` 的布尔异或操作，判断 `a` 和 `b` 是否不相等。
    
    
    DEFINE_UNARY_OP(aten::round, round_to_even(a), float, float),
    
    定义一个名为 `aten::round` 的一元操作，使用 `round_to_even` 函数对浮点数 `a` 进行四舍五入。
    
    
    DEFINE_UNARY_OP(aten::floor, floor(a), int, int),
    
    定义一个名为 `aten::floor` 的一元操作，使用 `floor` 函数对浮点数 `a` 进行向下取整，结果为整数。
    
    
    DEFINE_UNARY_OP(aten::ceil, ceil(a), int, int),
    
    定义一个名为 `aten::ceil` 的一元操作，使用 `ceil` 函数对浮点数 `a` 进行向上取整，结果为整数。
    
    
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::neg, -a, int, float),
    
    定义一个名为 `aten::neg` 的复数取负操作，计算复数 `a` 的负数。
    
    
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::exp, std::exp(a), float, float),
    
    定义一个名为 `aten::exp` 的指数运算操作，计算指数函数 `exp` 应用于浮点数 `a` 的结果。
    
    
    DEFINE_GENERIC_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        int,
        float),
    
    定义一个名为 `aten::remainder` 的通用操作，计算 `a` 除以 `b` 的余数。使用 `fmod` 函数确保余数计算在负数时保持与 Python 一致。
    
    
    DEFINE_INT_FLOAT_OP(aten::remainder, fmod((b + fmod(a, b)), b), float),
    
    定义一个名为 `aten::remainder` 的整数和浮点数操作，计算 `a` 除以 `b` 的余数。
    
    
    DEFINE_SCALAR_BINARY_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        Scalar),
    
    定义一个名为 `aten::remainder` 的标量二元操作，计算 `a` 除以 `b` 的余数。
    
    
    DEFINE_GENERIC_OP_WITH_COMPLEX(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        a / b,
        float,
        float,
        complex),
    
    定义一个名为 `aten::div` 的通用操作，实现除法操作。对于复数，计算复数 `a` 除以 `b`；对于浮点数，将 `a` 和 `b` 转换为 `double` 后进行除法运算。
    
    
    DEFINE_SCALAR_BINARY_OP(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        float),
    
    定义一个名为 `aten::div` 的标量二元操作，计算浮点数 `a` 除以 `b`。
    
    
    DEFINE_GENERIC_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        int,
        float),
    
    定义一个名为 `aten::floordiv` 的通用操作，计算 `a` 除以 `b` 后向下取整的结果。
    
    
    DEFINE_INT_FLOAT_OP(aten::floordiv, std::floor(a / b), float),
    
    定义一个名为 `aten::floordiv` 的整数和浮点数操作，计算 `a` 除以 `b` 后向下取整的结果。
    
    
    DEFINE_SCALAR_BINARY_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        Scalar),
    
    定义一个名为 `aten::floordiv` 的标量二元操作，计算 `a` 除以 `b` 后向下取整的结果。
    
    
    DEFINE_GENERIC_OP_WITH_COMPLEX(
        aten::pow,
        static_cast<double>(powWrapper(a, b)),
        static_cast<double>(powWrapper(a, b)),
        static_cast<c10::complex<double>>(pow(a, b)),
        float,
        float,
        complex),
    
    定义一个名为 `aten::pow` 的通用操作，计算 `a` 的 `b` 次幂。对于复数，使用 `powWrapper` 函数计算；对于浮点数，直接调用 `pow` 函数计算。
    // 定义整数和浮点数操作符的映射，针对 aten::pow 操作符，使用 static_cast 将结果转为 double 类型，适用于 float 类型
    DEFINE_INT_FLOAT_OP(
        aten::pow,
        static_cast<double>(powWrapper(a, b)),
        float),

    // 定义复数操作符的映射，针对 aten::pow 操作符，直接使用 pow 函数计算结果，适用于 complex 类型
    DEFINE_FLOAT_COMPLEX_OP(aten::pow, pow(a, b), complex),

    // 定义避免冲突的标量二元操作符映射，针对 aten::pow 操作符，分别转换 a 和 b 为 double 类型，避免类型冲突，适用于 float 类型
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::pow,
        static_cast<double>(pow(a, b)),
        static_cast<double>(pow(a, b)),
        float),

    // 生成操作符的参数，用于 aten::pow.int_to_int 操作符，实现对整数 a 和 b 的幂运算
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::pow.int_to_int(int a, int b) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t a, b;
          pop(stack, a, b);
          push(stack, powWrapper(a, b));
        },
        aliasAnalysisFromSchema()),

    // 定义 prim::min 操作符，根据 a 和 b 的大小比较返回最小值
    DEFINE_BINARY_OP(prim::min, a < b ? a : b),

    // 定义 prim::max 操作符，根据 a 和 b 的大小比较返回最大值
    DEFINE_BINARY_OP(prim::max, a > b ? a : b),

    // 生成操作符的参数，用于 prim::type 操作符，获取设备的类型并返回字符串形式
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::type(Device self) -> str"),
        [](Stack& stack) {
          auto d = pop(stack);
          push(
              stack, DeviceTypeName(d.toDevice().type(), /* lower_case=*/true));
        },
        aliasAnalysisFromSchema()),

    // 生成操作符的参数，用于 aten::len.Tensor 操作符，返回张量的第一个维度大小
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.Tensor(Tensor t) -> int"),
        [](Stack& stack) {
          at::Tensor t = pop(stack).toTensor();
          if (t.dim() == 0) {
            AT_ERROR("len() of a 0-d tensor");
          }
          push(stack, t.sizes()[0]);
        },
        aliasAnalysisFromSchema()),

    // 生成操作符的参数，用于 aten::ord 操作符，返回字符串的第一个字符的 ASCII 值
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ord(str string) -> int"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          TORCH_CHECK(
              string.size() == 1,
              "String for ord() must be 1 character, found ",
              string.size());
          uint8_t ord = string.at(0);
          push(stack, int64_t(ord));
        },
        aliasAnalysisFromSchema()),

    // 生成操作符的参数，用于 aten::lower 操作符，将输入字符串转换为小写并返回
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::lower(str self) -> str"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          std::stringstream ss;
          for (char c : string) {
            ss << static_cast<char>(::tolower(c));
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    // 生成操作符的参数，用于 aten::__contains__.int_list 操作符，判断整数列表是否包含特定元素
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.int_list(int[] l, int item) -> bool"),
        listContains<int64_t>,
        aliasAnalysisFromSchema()),

    // 生成操作符的参数，用于 aten::__contains__.str_list 操作符，判断字符串列表是否包含特定字符串
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.str_list(str[] l, str item) -> bool"),
        listContains<std::string>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.str(str s) -> int"),  
        // 定义操作符生成器参数，使用选择性模式对应 aten::len.str(str s) -> int 的操作符
        [](Stack& stack) {  
          // Lambda 函数，从栈中弹出字符串并获取其长度，将长度作为整数推送回栈中
          auto string = pop(stack).toStringRef();  
          // 弹出栈顶元素并转换为字符串引用
          push(stack, static_cast<int64_t>(string.size()));  
          // 将字符串的长度转换为 int64_t 类型并推送回栈中
        },
        aliasAnalysisFromSchema()),  
        // 使用模式的别名分析生成器
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dict() -> Dict(str, Tensor)"),  
        // 定义操作符生成器参数，使用选择性模式对应 aten::dict() -> Dict(str, Tensor) 的操作符
        [](Stack& stack) {  
          // Lambda 函数，创建一个空的通用字典并推送到栈中
          auto dict = c10::impl::GenericDict(StringType::get(), TensorType::get());  
          // 使用字符串类型和张量类型创建通用字典
          push(stack, dict);  
          // 将创建的字典推送到栈中
        },
        aliasAnalysisFromSchema()),  
        // 使用模式的别名分析生成器
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__getitem__.str(str s, int index) -> str"),  
        // 定义操作符生成器参数，使用选择性模式对应 aten::__getitem__.str(str s, int index) -> str 的操作符
        [](Stack& stack) {  
          // Lambda 函数，从栈中弹出整数和字符串，获取指定索引处的字符并推送回栈中
          auto index = pop(stack).toInt();  
          // 弹出栈顶的整数索引值
          auto string = pop(stack).toStringRef();  
          // 弹出栈顶的字符串并获取其引用
          auto norm_index = normalizeIndex(index, string.size());  
          // 对索引进行规范化，确保索引在字符串长度范围内
          char c = string.at(norm_index);  
          // 获取规范化后索引处的字符
          push(stack, std::string(&c, 1));  
          // 将获取的字符构造为字符串并推送回栈中
        },
        aliasAnalysisFromSchema()),  
        // 使用模式的别名分析生成器
#define CREATE_COPY_OP(other_type, c_type)                               \
  OperatorGeneratorArgs(                                                 \
      TORCH_SELECTIVE_SCHEMA("aten::copy_." #other_type                  \
                             "(Tensor(a!) self, " #other_type            \
                             " other) -> Tensor(a!)"),                   \
      [](Stack& stack) {                                                 \
        at::Tensor t;                                                    \
        c_type other;                                                    \
        pop(stack, t, other);                                            \
        std::move(t) = other; /* NOLINT(bugprone-use-after-move) */      \
        push(stack, std::move(t)); /* NOLINT(bugprone-use-after-move) */ \
      },                                                                 \
      aliasAnalysisFromSchema())

    // 创建一个操作生成器参数对象，用于生成针对不同类型的复制操作
    CREATE_COPY_OP(Tensor, at::Tensor),
    CREATE_COPY_OP(int, int64_t),
    CREATE_COPY_OP(float, double),
#undef CREATE_COPY_OP

    // 创建一个操作生成器参数对象，用于生成反向传播操作的定义
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::backward(Tensor self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()"),
        [](Stack& stack) {
          // 解析参数并执行反向传播
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          IValue gradient_ivalue = pop(stack);
          at::Tensor gradient = gradient_ivalue.isNone()
              ? at::Tensor()
              : gradient_ivalue.toTensor();
          at::Tensor self = pop(stack).toTensor();
          bool keep_graph = retain_graph ? retain_graph.value() : create_graph;
          self.backward(gradient, keep_graph, create_graph);
        },
        aliasAnalysisConservative()),

    //
    // 创建一个操作生成器参数对象，用于生成带有_hacked_twin重载名称的索引操作的定义
    // 并且消除了TensorList参数类型的空值性
    // TODO 弄清楚为什么存在这个hack以及如何在没有hack的情况下执行它
    //
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor"),
        [](Stack& stack) {
          // 解析索引参数并执行索引操作
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result = at::index(self, opt_list_indices);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unsafe_index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor"),
        [](Stack& stack) {
          // 从堆栈中弹出索引列表并转换为 Tensor 类型的列表
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          // 创建一个可选的 Tensor 类型的列表，并预留空间
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          // 将每个索引 Tensor 放入可选类型的列表中
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          // 从堆栈中弹出自身 Tensor
          auto self = pop(stack).toTensor();
          // 调用 at::_unsafe_index 函数，执行索引操作
          auto result = at::_unsafe_index(self, opt_list_indices);
          // 将结果推送回堆栈
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_index_put_impl_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)"),
        [](Stack& stack) {
          // 从堆栈中弹出是否 unsafe 的标志并转换为布尔型
          auto unsafe = pop(stack).toBool();
          // 从堆栈中弹出是否 accumulate 的标志并转换为布尔型
          auto accumulate = pop(stack).toBool();
          // 从堆栈中弹出 values Tensor
          auto values = pop(stack).toTensor();
          // 从堆栈中弹出索引列表并转换为 Tensor 类型的列表
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          // 创建一个可选的 Tensor 类型的列表，并预留空间
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          // 将每个索引 Tensor 放入可选类型的列表中
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          // 从堆栈中弹出自身 Tensor
          auto self = pop(stack).toTensor();
          // 调用 at::_index_put_impl_ 函数，执行索引赋值操作
          auto result = at::_index_put_impl_(
              self, opt_list_indices, values, accumulate, unsafe);
          // 将结果推送回堆栈
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index_put_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)"),
        [](Stack& stack) {
          // 从堆栈中弹出是否 accumulate 的标志并转换为布尔型
          auto accumulate = pop(stack).toBool();
          // 从堆栈中弹出 values Tensor
          auto values = pop(stack).toTensor();
          // 从堆栈中弹出索引列表并转换为 Tensor 类型的列表
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          // 创建一个可选的 Tensor 类型的列表，并预留空间
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          // 将每个索引 Tensor 放入可选类型的列表中
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          // 从堆栈中弹出自身 Tensor
          auto self = pop(stack).toTensor();
          // 调用 at::index_put_ 函数，执行索引赋值操作
          auto result =
              at::index_put_(self, opt_list_indices, values, accumulate);
          // 将结果推送回堆栈
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"),
        [](Stack& stack) {
          // 从堆栈中弹出是否累加的布尔值
          auto accumulate = pop(stack).toBool();
          // 从堆栈中弹出要放置的值（Tensor）
          auto values = pop(stack).toTensor();
          // 从堆栈中弹出索引列表（c10::List<at::Tensor>）
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          // 创建一个可选的张量列表，用于存储索引列表的可选值
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          // 将每个索引张量添加到可选张量列表中
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          // 从堆栈中弹出要修改的张量（self）
          auto self = pop(stack).toTensor();
          // 调用 ATen 库的 index_put 函数进行索引操作
          auto result =
              at::index_put(self, opt_list_indices, values, accumulate);
          // 将结果推送回堆栈
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unsafe_index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"),
        [](Stack& stack) {
          // 从堆栈中弹出是否累加的布尔值
          auto accumulate = pop(stack).toBool();
          // 从堆栈中弹出要放置的值（Tensor）
          auto values = pop(stack).toTensor();
          // 从堆栈中弹出索引列表（c10::List<at::Tensor>）
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          // 创建一个可选的张量列表，用于存储索引列表的可选值
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          // 将每个索引张量添加到可选张量列表中
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          // 从堆栈中弹出要修改的张量（self）
          auto self = pop(stack).toTensor();
          // 调用 ATen 库的 _unsafe_index_put 函数进行索引操作
          auto result =
              at::_unsafe_index_put(self, opt_list_indices, values, accumulate);
          // 将结果推送回堆栈
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    // reference function parse_to_conversion in python_arg_parsing.h
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool non_blocking;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool copy;
          // 从堆栈中弹出非阻塞和复制标志
          pop(stack, non_blocking, copy);
          // 从堆栈中弹出标量类型（可选）
          std::optional<at::ScalarType> scalarType =
              pop(stack).toOptional<at::ScalarType>();
          // 从堆栈中弹出设备类型（可选）
          std::optional<c10::Device> device =
              pop(stack).toOptional<c10::Device>();
          // 从堆栈中弹出要转换的张量（self）
          at::Tensor self = pop(stack).toTensor();
          // 调用 to_dispatch 函数进行设备转换
          push(
              stack, to_dispatch(self, device, scalarType, non_blocking, copy));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        toPrimDType,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_cuda(Tensor a) -> bool"),
        isCuda,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_cpu(Tensor a) -> bool"),
        [](Stack& stack) {
          // 从堆栈中弹出一个Tensor对象a
          at::Tensor a;
          pop(stack, a);
          // 将a是否在CPU上存储的结果推送回堆栈
          push(stack, a.is_cpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_xla(Tensor a) -> bool"),
        [](Stack& stack) {
          // 从堆栈中弹出一个Tensor对象a
          at::Tensor a;
          pop(stack, a);
          // 将a是否在XLA（加速线性代数）设备上存储的结果推送回堆栈
          push(stack, a.is_xla());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_mtia(Tensor a) -> bool"),
        [](Stack& stack) {
          // 从堆栈中弹出一个Tensor对象a
          at::Tensor a;
          pop(stack, a);
          // 将a是否在MTIA（多线程间模型）设备上存储的结果推送回堆栈
          push(stack, a.is_mtia());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_xpu(Tensor a) -> bool"),
        [](Stack& stack) {
          // 从堆栈中弹出一个Tensor对象a
          at::Tensor a;
          pop(stack, a);
          // 将a是否在XPU（异构处理单元）设备上存储的结果推送回堆栈
          push(stack, a.is_xpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::data(Tensor(a) a) -> Tensor(a)"),
        [](Stack& stack) {
          // 从堆栈中弹出一个Tensor对象a
          at::Tensor a;
          pop(stack, a);
          // 将a的变量数据（variable_data）推送回堆栈
          push(stack, autograd::Variable(a).variable_data());
        },
        aliasAnalysisFromSchema()),
// 定义一个宏，用于生成特定类型（decl_type）的比较操作符列表的算子生成参数
#define CREATE_COMPARATOR_LIST_OPS_SPECIALIZED(decl_type, value_type)        \
  OperatorGeneratorArgs(                                                     \
      TORCH_SELECTIVE_SCHEMA("prim::min." decl_type "_list(" decl_type       \
                             "[] l, " decl_type "[] r) -> " decl_type "[]"), \
      minList<value_type>,                                                   \
      aliasAnalysisFromSchema()),                                            \
      OperatorGeneratorArgs(                                                 \
          TORCH_SELECTIVE_SCHEMA("prim::max." decl_type "_list(" decl_type   \
                                 "[] l, " decl_type "[] r) -> " decl_type    \
                                 "[]"),                                      \
          maxList<value_type>,                                               \
          aliasAnalysisFromSchema()),                                        \
      OperatorGeneratorArgs(                                                 \
          TORCH_SELECTIVE_SCHEMA("prim::min.self_" decl_type "(" decl_type   \
                                 "[] self) -> " decl_type),                  \
          listMin<value_type>,                                               \
          aliasAnalysisFromSchema()),                                        \
      OperatorGeneratorArgs(                                                 \
          TORCH_SELECTIVE_SCHEMA("prim::max.self_" decl_type "(" decl_type   \
                                 "[] self) -> " decl_type),                  \
          listMax<value_type>,                                               \
          aliasAnalysisFromSchema()),

// 为不同类型（int、float、bool）分别调用宏 CREATE_COMPARATOR_LIST_OPS_SPECIALIZED，生成相应的比较操作符列表的算子生成参数
CREATE_COMPARATOR_LIST_OPS_SPECIALIZED("int", int64_t)
CREATE_COMPARATOR_LIST_OPS_SPECIALIZED("float", double)
CREATE_COMPARATOR_LIST_OPS_SPECIALIZED("bool", bool)

// 取消定义宏 CREATE_COMPARATOR_LIST_OPS_SPECIALIZED，结束宏的定义
#undef CREATE_COMPARATOR_LIST_OPS_SPECIALIZED

// 定义一个宏，用于生成字符串操作符 isdigit 的算子生成参数
#define DEFINE_STRING_IS_OP(op_name, char_op)                          \
  OperatorGeneratorArgs(                                               \
      TORCH_SELECTIVE_SCHEMA(#op_name "(str self) -> bool"),           \
      [](Stack& stack) {                                               \
        auto string = pop(stack).toStringRef();                        \
        push(                                                          \
            stack,                                                     \
            // 检查字符串是否非空且每个字符是否符合指定的字符操作函数（char_op）的条件 \
            string.size() != 0 &&                                      \
                std::all_of(string.begin(), string.end(), [](char c) { \
                  return char_op(c);                                   \
                }));                                                   \
      },                                                               \
      aliasAnalysisFromSchema())

// 生成字符串操作符 isdigit 的算子生成参数
DEFINE_STRING_IS_OP(aten::isdigit, ::isdigit),
    # 定义字符串操作宏，检查字符串中字符是否满足特定条件，例如isspace检查空白字符
    DEFINE_STRING_IS_OP(aten::isspace, ::isspace),
    # 定义字符串操作宏，检查字符串中字符是否满足特定条件，例如isalnum检查是否为字母或数字
    DEFINE_STRING_IS_OP(aten::isalnum, ::isalnum),
    # 定义字符串操作宏，检查字符串中字符是否满足特定条件，例如isalpha检查是否为字母
    DEFINE_STRING_IS_OP(aten::isalpha, ::isalpha),
    # 定义字符串操作宏，检查字符串中字符是否满足特定条件，例如isdecimal检查是否为十进制数字
    DEFINE_STRING_IS_OP(aten::isdecimal, ::isdigit),
    # 定义字符串操作宏，检查字符串中字符是否满足特定条件，例如isnumeric检查是否为数字
    DEFINE_STRING_IS_OP(aten::isnumeric, ::isdigit),
#define DEFINE_STRING_CHAR_MAP_OP(op_name, char_op)         \
  OperatorGeneratorArgs(                                    \
      TORCH_SELECTIVE_SCHEMA(#op_name "(str self) -> str"), \
      [](Stack& stack) {                                    \
        auto string = pop(stack).toStringRef();             \
        std::stringstream ss;                               \
        // 遍历字符串中的每个字符，应用给定的字符操作函数并构建新字符串
        for (char c : string) {                             \
          ss << static_cast<char>(char_op(c));              \
        }                                                   \
        // 将构建的新字符串推送到堆栈上
        push(stack, ss.str());                              \
      },                                                    \
      aliasAnalysisFromSchema())

    // 定义字符串操作函数并生成对应的操作符对象
    DEFINE_STRING_CHAR_MAP_OP(aten::upper, ::toupper),
    DEFINE_STRING_CHAR_MAP_OP(aten::swapcase, ([](char c) {
                                if (c == static_cast<char>(::toupper(c))) {
                                  return static_cast<char>(::tolower(c));
                                } else {
                                  return static_cast<char>(::toupper(c));
                                }
                              }))};

static std::vector<std::optional<Operator>> createOperators(
    const std::vector<OperatorGeneratorArgs>& args) {
  std::vector<std::optional<Operator>> result;
  result.reserve(args.size());
  // 遍历操作符生成器参数列表，创建操作符对象并存储在结果向量中
  for (const auto& arg : args) {
    if (arg.schema_str) {
      if (arg.isOperationCreator) {
        result.push_back(OperatorGenerator(
            arg.schema_str, arg.operationCreator, arg.aliasAnalysis));
      } else {
        result.push_back(OperatorGenerator(
            arg.schema_str, arg.operation, arg.aliasAnalysis));
      }
    }
  }
  return result;
}

// 注册操作符函数
RegisterOperators reg(([]() {
  // 创建操作符向量并初始化
  auto v = createOperators(opGenArgs);
  // 添加特殊操作符：prim::tolist
  v.emplace_back(Operator(
      prim::tolist,
      // 该操作符无法通过模式进行描述，因为返回类型依赖于类型提示和输入
      // 下面的实现尽可能接近 torch/csrc/utils/tensor_list.cpp 中的 Python 实现
      [](const Node* /*node*/) -> Operation { return toList; },
      aliasAnalysisSpecialCase()));
  return v;
})());

// 向字典中插入或更新项的函数
void dictSetItem(Stack& stack) {
  auto value = pop(stack);
  auto idx = pop(stack);
  auto dict = pop(stack).toGenericDict();
  dict.insert_or_assign(std::move(idx), std::move(value));
}

// 获取字典长度的函数
void dictLen(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  // 将字典的大小推送到堆栈上
  push(stack, int64_t(dict.size()));
}

// 获取字典值列表的函数
void dictValues(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto values = c10::impl::GenericList(dict.valueType());
  // 遍历字典，将每个值添加到值列表中
  for (const auto& entry : dict) {
    values.emplace_back(entry.value());
  }
  // 将值列表推送到堆栈上
  push(stack, values);
}

// 获取字典键列表的函数
void dictKeys(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto keys = c10::impl::GenericList(dict.keyType());
  // 遍历字典，将每个键添加到键列表中
  for (const auto& entry : dict) {
    keys.emplace_back(entry.key());

# 将 entry 的键值添加到 keys 向量的末尾


  }
  push(stack, keys);

# 将 keys 向量压入栈 stack 中
}

// 模板函数：从字典中获取指定键的值，如果字典中不存在该键，则返回默认值
template <bool has_default>
void dictGet(Stack& stack) {
  // 定义默认值
  IValue default_value;
  // 如果有默认值，从堆栈中取出
  if (has_default) {
    default_value = pop(stack);
  }
  // 取出键值
  auto key = pop(stack);
  // 取出字典
  auto dict = pop(stack).toGenericDict();
  // 查找键在字典中的位置
  auto value = dict.find(key);
  // 如果找不到，则返回默认值；否则返回找到的值
  if (value == dict.end()) {
    push(stack, std::move(default_value));
  } else {
    push(stack, value->value());
  }
}

// 函数：如果键在字典中，则返回其对应的值；否则设置键为默认值并返回该默认值
void dictSetDefault(Stack& stack) {
  // 取出默认值
  auto default_value = pop(stack);
  // 取出键
  auto key = pop(stack);
  // 取出字典
  auto dict = pop(stack).toGenericDict();
  // 查找键在字典中的位置
  auto value = dict.find(key);
  // 如果找不到，则插入键值对并返回默认值；否则返回找到的值
  if (value == dict.end()) {
    dict.insert(key, default_value);
    push(stack, std::move(default_value));
  } else {
    push(stack, value->value());
  }
}

// 模板函数：从字典中弹出指定键的值，如果字典中不存在该键，则抛出错误
template <bool has_default>
void dictPop(Stack& stack) {
  // 定义默认值
  IValue default_value;
  // 如果有默认值，从堆栈中取出
  if (has_default) {
    default_value = pop(stack);
  }
  // 取出键
  auto key = pop(stack);
  // 取出字典
  auto dict = pop(stack).toGenericDict();
  // 查找键在字典中的位置
  auto iter = dict.find(key);
  // 如果找不到键
  if (iter == dict.end()) {
    // 如果有默认值，则返回默认值；否则抛出错误
    if (has_default) {
      push(stack, default_value);
    } else {
      AT_ERROR("KeyError: ", key);
    }
  } else {
    // 如果找到键，则返回其对应的值
    push(stack, iter->value());
    // 从字典中删除该键值对
    auto erase_count = dict.erase(key);
    TORCH_CHECK(
        erase_count == 1, "Expected to erase 1 item, found ", erase_count);
  }
}

// 函数：从字典中删除指定键值对
void dictDelete(Stack& stack) {
  // 调用模板函数 dictPop，没有默认值
  dictPop<false>(stack);
  // 从堆栈中弹出多余的项
  pop(stack);
}

// 函数：弹出字典中的第一个键值对作为元组返回
void dictPopItem(Stack& stack) {
  // 取出字典
  auto dict = pop(stack).toGenericDict();
  // 如果字典为空，则抛出错误
  if (dict.empty()) {
    AT_ERROR("popitem(): dictionary is empty");
  }
  // 取出字典中的第一个元素
  auto head_item = dict.begin();
  // 创建键值对的元组
  IValue tuple =
      c10::ivalue::Tuple::create({head_item->key(), head_item->value()});
  // 从字典中删除第一个元素
  auto erase_count = dict.erase(head_item->key());
  TORCH_CHECK(
      erase_count == 1, "Expected to erase 1 item, found ", erase_count);
  // 将元组推入堆栈
  push(stack, tuple);
}

// 函数：判断字典中是否包含指定键
void dictContains(Stack& stack) {
  // 取出键
  auto key = pop(stack);
  // 取出字典
  auto dict = pop(stack).toGenericDict();
  // 将判断结果推入堆栈
  push(stack, dict.contains(key));
}

// 函数：清空字典
void dictClear(Stack& stack) {
  // 取出字典
  auto dict = pop(stack).toGenericDict();
  // 清空字典
  dict.clear();
}

// 函数：更新字典，将一个字典中的键值对更新到另一个字典中
void dictUpdate(Stack& stack) {
  // 取出要添加的字典
  auto to_add = pop(stack).toGenericDict();
  // 取出目标字典
  auto dict = pop(stack).toGenericDict();

  // 遍历要添加的字典
  for (const auto& item : to_add) {
    // 插入或更新键值对
    dict.insert_or_assign(item.key(), item.value());
  }
}

// 函数：返回字典的键值对列表
void dictItems(Stack& stack) {
  // 取出字典
  auto dict = pop(stack).toGenericDict();
  // 获取字典中键的类型和值的类型
  auto key_type = dict.keyType();
  auto value_type = dict.valueType();
  // 创建元组列表
  auto items =
      c10::impl::GenericList(TupleType::create({key_type, value_type}));
  // 预留空间
  items.reserve(dict.size());
  // 遍历字典，将键值对转换为元组并加入列表
  for (const auto& item : dict) {
    items.emplace_back(c10::ivalue::Tuple::create({item.key(), item.value()}));
  }
  // 将元组列表推入堆栈
  push(stack, std::move(items));
}

// 函数：复制字典
void dictCopy(Stack& stack) {
  // 将字典复制并推入堆栈
  push(stack, pop(stack).toGenericDict().copy());
}
// 从输入栈中弹出列表，并转换为列表对象
void dictConstructFromList(Stack& stack) {
  auto input_list = pop(stack);
  auto list = input_list.toList();
  // 期望列表元素类型为元组类型
  auto tup_type = list.elementType()->expect<TupleType>();
  // 创建一个通用字典对象，使用元组的第一个和第二个元素类型作为键和值的类型
  auto dict = c10::impl::GenericDict(
      tup_type->elements().at(0), tup_type->elements().at(1));
  // 预留足够的空间以容纳列表的大小
  dict.reserve(list.size());
  // 遍历列表中的每个元素
  for (IValue input : list) {
    // 获取元组的引用，并插入或更新字典中的键值对
    const auto& tup = input.toTupleRef().elements();
    dict.insert_or_assign(tup[0], tup[1]);
  }
  // 将构建的字典压入栈中
  push(stack, dict);
}

// 创建字典操作的静态常量向量
static const std::vector<OperatorGeneratorArgs> dict_ops{
    CREATE_DICT_OPS("str"),
    CREATE_DICT_OPS("int"),
    CREATE_DICT_OPS("bool"),
    CREATE_DICT_OPS("float"),
    CREATE_DICT_OPS("complex"),
    CREATE_DICT_OPS("Tensor"),
};
// 注册字典操作符
RegisterOperators reg_dict_ops(createOperators(dict_ops));

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
// 从模式中返回别名分析类型
constexpr c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// 将Python索引（可能为负）转换为适用于C++容器的索引
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // 处理负索引
    idx = list_size + idx;
  }
  return idx;
}

// 在字符串中查找子字符串的实现
int64_t stringFindImpl(
    std::string string,
    const std::string& substr,
    int64_t start,
    int64_t end,
    bool reverse = false) {
  int64_t size = string.size();
  if (start < 0) {
    // 处理负的起始索引
    start = std::max(int64_t(0), int64_t(size + start));
  }
  if (end < 0) {
    // 处理负的结束索引
    end = std::max(int64_t(0), int64_t(size + end + 1));
  }
  if (end > start) {
    // 截取字符串的子串
    string = string.substr(start, end - start);
  } else {
    string = "";
  }

  int64_t result = -1;
  if (string.size() >= substr.size()) {
    // 在截取的字符串中查找子字符串
    auto pos = string.find(substr, 0);
    if (reverse) {
      // 如果需要反向查找，则执行反向查找
      auto rpos = pos;
      do {
        pos = rpos;
        rpos = string.find(substr, pos + 1);
      } while (rpos != std::string::npos);
    }
    if (pos != std::string::npos) {
      result = pos + start;
    }
  }
  return result;
}

// 字符串操作符
// 实现位于torch/csrc/jit/runtime/register_prim_ops.cpp中
static const std::vector<OperatorGeneratorArgs> stringOpGenArgs{
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::slice.str(str string, int? start=None, int? end=None, int step=1) -> str"),
        [](Stack& stack) {
          int64_t step = pop(stack).toInt();
          std::optional<int64_t> end = pop(stack).toOptional<int64_t>();
          std::optional<int64_t> start = pop(stack).toOptional<int64_t>();
          std::string string = pop(stack).toStringRef();
          // 调用字符串切片函数，并将结果推送回栈中
          push(stack, stringSlice(string, start, end, step));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::strip(str self, str chars=' \\n\\t\\f\\v') -> str"),
        [](Stack& stack) {
          // 从堆栈中弹出字符串 chars，表示要去除的字符集合
          std::string chars = pop(stack).toStringRef();
          // 从堆栈中弹出字符串 self，表示要进行去除空白字符操作的字符串
          std::string string = pop(stack).toStringRef();
          // 查找字符串中最后一个不属于 chars 中字符的索引位置
          auto rindex = string.find_last_not_of(chars);
          if (rindex != std::string::npos) {
            // 如果找到非 chars 中的字符，则截取字符串从开头到该位置的子字符串
            string = string.substr(0, rindex + 1);
          } else {
            // 如果未找到非 chars 中的字符，则将字符串置为空字符串
            string = "";
          }
          // 查找字符串中第一个不属于 chars 中字符的索引位置
          auto lindex = string.find_first_not_of(chars);
          if (lindex != std::string::npos) {
            // 如果找到非 chars 中的字符，则截取从该位置开始到字符串末尾的子字符串
            string = string.substr(lindex, string.size());
          } else {
            // 如果未找到非 chars 中的字符，则将字符串置为空字符串
            string = "";
          }
          // 将处理后的字符串推回堆栈中
          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::split.str(str self, str? separator=None, int max=-1) -> str[]"),
        [](Stack& stack) {
          // 从堆栈中弹出整数 max，表示最大分割次数
          int64_t max = pop(stack).toInt();
          // 从堆栈中弹出 IValue 对象 ivalue，可能是分隔符或空值
          IValue ivalue = pop(stack);
          // 从堆栈中弹出字符串 self，表示要进行分割操作的字符串
          std::string string = pop(stack).toStringRef();
    
          // 初始化分割结果的列表
          std::string::size_type prev_pos = 0;
          std::string::size_type pos = 0;
          c10::List<std::string> splits;
    
          // 如果分隔符 ivalue 为空值，则调用 splitNoneSeparator 函数进行特殊分割
          if (ivalue == c10::nullopt) {
            // 如果分隔符未指定，使用非常规的分割算法（与 Python 相似）
            splits = splitNoneSeparator(string);
            // 将分割结果推回堆栈
            push(stack, std::move(splits));
            return;
          }
    
          // 将 ivalue 转换为字符串作为分隔符
          const std::string& separator = ivalue.toStringRef();
    
          // 如果分隔符为空字符串，则抛出错误
          if (separator.empty()) {
            throw std::runtime_error("ValueError: empty separator");
          }
    
          auto count = 0;
    
          // 在字符串中查找分隔符，并进行分割
          while ((pos = string.find(separator, pos)) != std::string::npos) {
            count++;
            // 如果达到最大分割次数，则停止分割
            if (max >= 0 && count > max) {
              break;
            } else {
              // 将分割出的子字符串加入到结果列表中
              splits.emplace_back(string.substr(prev_pos, pos - prev_pos));
            }
            // 调整 pos 到下一个分隔符处
            pos += separator.size();
            prev_pos = pos;
          }
          // 将剩余的字符串（最后一部分）加入到结果列表中
          splits.emplace_back(
              string.substr(prev_pos, string.size() - prev_pos));
          // 将分割结果推回堆栈
          push(stack, std::move(splits));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::splitlines(str self, bool keepends=False) -> str[]"),
        [](Stack& stack) {
          // 弹出堆栈中的布尔值参数 keepends
          bool keepends = pop(stack).toBool();
          // 弹出堆栈中的字符串参数，并转换为标准库的 string 类型
          std::string string = pop(stack).toStringRef();
          // 定义分隔符列表，包含各种换行符
          std::string delimiters =
              "\n\r\r\n\v\x0b\f\x0c\x1c\x1d\x1e\x85\u2028\u2029";
          // 创建用于存储分割后字符串的列表
          c10::List<std::string> splits;
    
          // 初始化字符串查找的位置指针
          std::string::size_type prev_pos = 0;
          std::string::size_type pos = 0;
          // 循环查找字符串中的换行符或回车符，并进行分割处理
          while ((pos = string.find_first_of(delimiters, pos)) !=
                 std::string::npos) {
            // 将分割出的子字符串加入到列表中
            splits.emplace_back(string.substr(prev_pos, pos - prev_pos));
            // 如果 keepends 为真，则将分割出的换行符也加入到列表中
            if (keepends) {
              splits.emplace_back(string.substr(pos, 1));
            }
            // 移动指针到下一个位置
            pos++;
            prev_pos = pos;
          }
          // 处理最后一个分割位置之后的字符串部分
          if (prev_pos != string.size()) {
            splits.emplace_back(
                string.substr(prev_pos, string.size() - prev_pos));
          }
    
          // 将处理好的分割结果列表压入堆栈中
          push(stack, std::move(splits));
        },
        aliasAnalysisFromSchema()),
    
    // upper 和 lower 需要字符串中至少包含一个字母字符，并且忽略其他字符
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::isupper(str self) -> bool"),
        [](Stack& stack) {
          // 弹出堆栈中的字符串参数，并转换为标准库的 string 类型
          std::string string = pop(stack).toStringRef();
          // 初始化标志变量
          bool found_alpha = false;
          bool is_upper = true;
          // 遍历字符串的每个字符
          for (size_t i = 0; i < string.size() && is_upper; ++i) {
            char c = string[i];
            // 检查字符是否为字母，并设置 found_alpha 标志
            found_alpha |= static_cast<bool>(::isalpha(c));
            // 检查字符是否为大写字母，并更新 is_upper 标志
            is_upper &= (!::isalpha(c) || ::isupper(c));
          }
          // 将最终判断结果压入堆栈中
          push(stack, found_alpha && is_upper);
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::islower(str self) -> bool"),
        [](Stack& stack) {
          // 弹出堆栈中的字符串参数，并转换为标准库的 string 类型
          std::string string = pop(stack).toStringRef();
          // 初始化标志变量
          bool found_alpha = false;
          bool is_lower = true;
          // 遍历字符串的每个字符
          for (size_t i = 0; i < string.size() && is_lower; ++i) {
            char c = string[i];
            // 检查字符是否为字母，并设置 found_alpha 标志
            found_alpha |= static_cast<bool>(::isalpha(c));
            // 检查字符是否为小写字母，并更新 is_lower 标志
            is_lower &= (!::isalpha(c) || ::islower(c));
          }
          // 将最终判断结果压入堆栈中
          push(stack, found_alpha && is_lower);
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::capitalize(str self) -> str"),
        [](Stack& stack) {
          // 弹出堆栈中的字符串参数，并转换为标准库的 string 类型
          std::string string = pop(stack).toStringRef();
          // 创建字符串流对象用于构建首字母大写的字符串
          std::stringstream ss;
          // 标志变量，用于判断当前是否为第一个字符
          auto first_char = true;
          // 遍历字符串的每个字符
          for (char c : string) {
            if (first_char) {
              // 若为第一个字符，则转换为大写并写入字符串流
              ss << static_cast<char>(::toupper(c));
              first_char = false;
            } else {
              // 否则转换为小写并写入字符串流
              ss << static_cast<char>(::tolower(c));
            }
          }
          // 将构建好的首字母大写的字符串压入堆栈中
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 定义运算符生成器参数，指定对应的 ATen 操作函数签名
        TORCH_SELECTIVE_SCHEMA("aten::title(str self) -> str"),
        // Lambda 函数，实现对字符串进行首字母大写，其余字母小写的操作
        [](Stack& stack) {
          // 从堆栈中弹出字符串对象并转换为标准字符串
          std::string string = pop(stack).toStringRef();
          // 使用字符串流进行处理
          std::stringstream ss;
          // 初始前一个字符为非字母
          bool prev_is_nonalpha = true;
          // 遍历输入字符串中的每个字符
          for (char c : string) {
            // 根据前一个字符是否为非字母决定当前字符的大小写转换
            if (prev_is_nonalpha) {
              ss << static_cast<char>(::toupper(c));  // 将字符转换为大写
            } else {
              ss << static_cast<char>(::tolower(c));  // 将字符转换为小写
            }
            // 更新前一个字符是否为非字母的状态
            if (::isalpha(c)) {
              prev_is_nonalpha = false;  // 当前字符是字母
            } else {
              prev_is_nonalpha = true;   // 当前字符不是字母
            }
          }
          // 将处理后的字符串压入堆栈
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        // 定义运算符生成器参数，指定对应的 ATen 操作函数签名
        TORCH_SELECTIVE_SCHEMA(
            "aten::center(str self, int width, str fillchar=' ') -> str"),
        // Lambda 函数，实现将字符串居中显示，并使用指定字符填充空白
        [](Stack& stack) {
          // 从堆栈中弹出填充字符并转换为标准字符串
          std::string fillchar = pop(stack).toStringRef();
          // 从堆栈中弹出宽度并转换为整数
          int64_t width = pop(stack).toInt();
          // 从堆栈中弹出字符串对象并转换为标准字符串
          std::string string = pop(stack).toStringRef();
          // 如果填充字符长度不为1，则抛出运行时错误
          if (fillchar.size() != 1) {
            // TODO: 应该是一个 TypeError
            throw std::runtime_error(
                "TypeError: The fill character must be exactly one character long");
          }
          // 如果字符串长度大于指定宽度，则直接将原字符串压入堆栈
          if (string.size() > static_cast<std::string::size_type>(width)) {
            push(stack, string);
            return;
          }
          // 使用字符串流进行处理
          std::stringstream ss;
          // 计算全填充长度
          std::string::size_type full_padding = width - string.size();
          // 计算左填充长度
          std::string::size_type l_pad = full_padding / 2;
          // 计算右填充长度
          std::string::size_type r_pad = (full_padding + 1) / 2;
          // 如果宽度为奇数，则交换左右填充长度
          if (width % 2) {
            auto tmp = r_pad;
            r_pad = l_pad;
            l_pad = tmp;
          }
          // 左侧填充指定字符
          for (std::string::size_type i = 0; i < l_pad; ++i) {
            ss << fillchar;
          }
          // 添加原始字符串
          ss << string;
          // 右侧填充指定字符
          for (std::string::size_type i = 0; i < r_pad; ++i) {
            ss << fillchar;
          }
          // 将处理后的字符串压入堆栈
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    // 从以下链接中适配
    // https://stackoverflow.com/questions/22489073/counting-the-number-of-occurrences-of-a-string-within-a-string
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::count(str self, str substr, int start=0, int end=-1) -> int"),
        // 匿名函数定义，处理计数操作
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();  // 从堆栈中取出结束位置参数
          int64_t start = pop(stack).toInt();  // 从堆栈中取出开始位置参数
          std::string substr = pop(stack).toStringRef();  // 从堆栈中取出子字符串参数
          std::string string = pop(stack).toStringRef();  // 从堆栈中取出主字符串参数
          int64_t size = string.size();  // 获取主字符串的长度
    
          // 处理开始位置超出字符串长度的情况
          if (start > size) {
            push(stack, 0);
            return;
          }
    
          // 处理负的开始位置参数，转换为非负数
          if (start < 0) {
            start = std::max(int64_t(0), int64_t(size + start));
          }
    
          // 处理负的结束位置参数，转换为非负数
          if (end < 0) {
            end = std::max(int64_t(0), int64_t(size + end + 1));
          }
    
          // 初始化计数器
          int64_t occurrences = 0;
          std::string::size_type pos = start;
    
          // 循环查找子字符串在主字符串中的出现次数
          while ((pos = string.find(substr, pos)) != std::string::npos) {
            if (pos < static_cast<std::string::size_type>(end)) {
              ++occurrences;
            } else {
              break;
            }
            pos += substr.length();
          }
    
          // 将结果推送回堆栈
          push(stack, occurrences);
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::endswith(str self, str substr, int start=0, int end=-1) -> bool"),
        // 匿名函数定义，处理字符串是否以指定子字符串结尾的操作
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();  // 从堆栈中取出结束位置参数
          int64_t start = pop(stack).toInt();  // 从堆栈中取出开始位置参数
          std::string substr = pop(stack).toStringRef();  // 从堆栈中取出子字符串参数
          std::string string = pop(stack).toStringRef();  // 从堆栈中取出主字符串参数
          int64_t size = string.size();  // 获取主字符串的长度
    
          // 处理负的开始位置参数，转换为非负数
          if (start < 0) {
            start = std::max(int64_t(0), int64_t(size + start));
          }
    
          // 处理负的结束位置参数，转换为非负数
          if (end < 0) {
            end = std::max(int64_t(0), int64_t(size + end + 1));
          }
    
          // 截取主字符串中指定范围的子字符串
          string = string.substr(start, end - start);
    
          // 判断截取后的子字符串是否以指定的子字符串结尾
          auto result = false;
          if (string.length() >= substr.length()) {
            result = !string.compare(
                string.length() - substr.length(), substr.length(), substr);
          }
    
          // 将结果推送回堆栈
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::startswith(str self, str substr, int start=0, int end=-1) -> bool"),  # 定义以字符串是否以指定子字符串开头的操作
        [](Stack& stack) {  # Lambda函数开始，处理操作的实际逻辑
          int64_t end = pop(stack).toInt();  # 从栈中弹出并转换为整数，表示结束索引
          int64_t start = pop(stack).toInt();  # 从栈中弹出并转换为整数，表示开始索引
          std::string substr = pop(stack).toStringRef();  # 从栈中弹出并获取作为引用的子字符串
          std::string string = pop(stack).toStringRef();  # 从栈中弹出并获取作为引用的字符串
          int64_t size = string.size();  # 获取字符串的长度
    
          if (start < 0) {  # 处理负的起始索引
            start = std::max(int64_t(0), int64_t(size + start));  # 计算实际的起始索引位置
          }
          if (end < 0) {  # 处理负的结束索引
            end = std::max(int64_t(0), int64_t(size + end + 1));  # 计算实际的结束索引位置
          }
    
          string = string.substr(start, end - start);  # 根据计算得到的索引范围截取字符串
    
          auto result = false;  # 初始化结果为false
          if (string.length() >= substr.length()) {  # 如果字符串长度大于等于子字符串长度
            result = !string.compare(0, substr.length(), substr);  // 比较截取的字符串与子字符串是否相等
          }
          push(stack, result);  // 将结果推入栈中
        },
        aliasAnalysisFromSchema()),  // 根据架构执行别名分析
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::expandtabs(str self, int tabsize=8) -> str"),  # 定义扩展制表符操作的架构
        [](Stack& stack) {  // Lambda函数开始，处理操作的实际逻辑
          int64_t tabsize = pop(stack).toInt();  // 从栈中弹出并转换为整数，表示制表符大小
          std::string string = pop(stack).toStringRef();  // 从栈中弹出并获取作为引用的字符串
          std::stringstream ss;  // 创建一个字符串流
          size_t index = 0;  // 初始化索引为0
    
          for (const auto& c : string) {  // 遍历字符串中的每个字符
            if (c != '\t') {  // 如果字符不是制表符
              ss << c;  // 将字符添加到字符串流中
              index++;  // 索引增加
            } else {  // 如果字符是制表符
              if (tabsize <= 0) {  // 如果制表符大小小于等于0
                continue;  // 继续下一个循环
              }
              do {
                ss << ' ';  // 添加空格到字符串流中
                index++;  // 索引增加
              } while (index % tabsize);  // 直到索引对制表符大小取模为0
            }
          }
          push(stack, ss.str());  // 将字符串流中的内容作为字符串推入栈中
        },
        aliasAnalysisFromSchema()),  // 根据架构执行别名分析
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::find(str self, str substr, int start=0, int end=-1) -> int"),  # 定义查找子字符串操作的架构
        [](Stack& stack) {  // Lambda函数开始，处理操作的实际逻辑
          int64_t end = pop(stack).toInt();  // 从栈中弹出并转换为整数，表示结束索引
          int64_t start = pop(stack).toInt();  // 从栈中弹出并转换为整数，表示开始索引
          std::string substr = pop(stack).toStringRef();  // 从栈中弹出并获取作为引用的子字符串
          std::string string = pop(stack).toStringRef();  // 从栈中弹出并获取作为引用的字符串
    
          push(stack, stringFindImpl(string, substr, start, end));  // 调用字符串查找函数并将结果推入栈中
        },
        aliasAnalysisFromSchema()),  // 根据架构执行别名分析
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rfind(str self, str substr, int start=0, int end=-1) -> int"),  # 定义反向查找子字符串操作的架构
        [](Stack& stack) {  // Lambda函数开始，处理操作的实际逻辑
          int64_t end = pop(stack).toInt();  // 从栈中弹出并转换为整数，表示结束索引
          int64_t start = pop(stack).toInt();  // 从栈中弹出并转换为整数，表示开始索引
          std::string substr = pop(stack).toStringRef();  // 从栈中弹出并获取作为引用的子字符串
          std::string string = pop(stack).toStringRef();  // 从栈中弹出并获取作为引用的字符串
    
          push(stack, stringFindImpl(string, substr, start, end, true));  // 调用反向字符串查找函数并将结果推入栈中
        },
        aliasAnalysisFromSchema()),  // 根据架构执行别名分析
    OperatorGeneratorArgs(
        // 定义一个操作生成器参数，使用特定的 ATen 模式选择方案，描述操作 "aten::index.str(str self, str substr, int start=0, int end=-1) -> int"
        TORCH_SELECTIVE_SCHEMA(
            "aten::index.str(str self, str substr, int start=0, int end=-1) -> int"),
        [](Stack& stack) {
          // 从栈中弹出一个值作为 end 参数
          int64_t end = pop(stack).toInt();
          // 从栈中弹出一个值作为 start 参数
          int64_t start = pop(stack).toInt();
          // 从栈中弹出一个字符串值作为 substr 参数
          std::string substr = pop(stack).toStringRef();
          // 从栈中弹出一个字符串值作为 self 参数
          std::string string = pop(stack).toStringRef();
          // 调用 stringFindImpl 函数执行字符串查找操作，返回结果保存在 result 中
          auto result = stringFindImpl(string, substr, start, end);
          // 如果查找结果小于 0，抛出运行时错误
          if (result < 0) {
            throw std::runtime_error("ValueError: substring not found");
          }
          // 将结果推送回栈中
          push(stack, result);
        },
        // 使用模式选择方案生成的别名分析
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 定义一个操作生成器参数，使用特定的 ATen 模式选择方案，描述操作 "aten::rindex(str self, str substr, int start=0, int end=-1) -> int"
        TORCH_SELECTIVE_SCHEMA(
            "aten::rindex(str self, str substr, int start=0, int end=-1) -> int"),
        [](Stack& stack) {
          // 从栈中弹出一个值作为 end 参数
          int64_t end = pop(stack).toInt();
          // 从栈中弹出一个值作为 start 参数
          int64_t start = pop(stack).toInt();
          // 从栈中弹出一个字符串值作为 substr 参数
          std::string substr = pop(stack).toStringRef();
          // 从栈中弹出一个字符串值作为 self 参数
          std::string string = pop(stack).toStringRef();
          // 调用 stringFindImpl 函数执行字符串反向查找操作，返回结果保存在 result 中
          auto result = stringFindImpl(string, substr, start, end, true);
          // 如果查找结果小于 0，抛出运行时错误
          if (result < 0) {
            throw std::runtime_error("ValueError: substring not found");
          }
          // 将结果推送回栈中
          push(stack, result);
        },
        // 使用模式选择方案生成的别名分析
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 定义一个操作生成器参数，使用特定的 ATen 模式选择方案，描述操作 "aten::isidentifier(str self) -> bool"
        TORCH_SELECTIVE_SCHEMA("aten::isidentifier(str self) -> bool"),
        [](Stack& stack) {
          // 从栈中弹出一个字符串值作为 self 参数
          std::string string = pop(stack).toStringRef();
          // 记录警告日志，指出正在使用的 isidentifier() 实现来自于 Python 2
          LOG(WARNING)
              << "The isidentifier() implementation being used is from Python 2\n";
          // 如果字符串为空，将 false 推送回栈中并返回
          if (string.empty()) {
            push(stack, false);
            return;
          }
          // 如果字符串的第一个字符是数字，将 false 推送回栈中并返回
          if (::isdigit(string[0])) {
            push(stack, false);
            return;
          }
          // 对字符串的每个字符执行检查，确保都是字母或数字，将检查结果推送回栈中
          auto result = std::all_of(string.begin(), string.end(), [](char c) {
            return ::isalnum(c);
          });
          push(stack, result);
        },
        // 使用模式选择方案生成的别名分析
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 定义一个操作生成器参数，使用特定的 ATen 模式选择方案，描述操作 "aten::istitle(str self) -> bool"
        TORCH_SELECTIVE_SCHEMA("aten::istitle(str self) -> bool"),
        [](Stack& stack) {
          // 从栈中弹出一个字符串值作为 self 参数
          std::string string = pop(stack).toStringRef();
          // 初始化结果为 false
          auto result = false;

          // 用于跟踪前一个字符是否为字母
          bool prev_is_alpha = false;
          // 遍历字符串中的每个字符
          for (char c : string) {
            // 如果前一个字符是字母
            if (prev_is_alpha) {
              // 如果当前字符不是小写形式，则设置结果为 false 并退出循环
              if (c != static_cast<char>(::tolower(c))) {
                result = false;
                break;
              }
            } else {
              // 如果当前字符不是大写形式，则设置结果为 false 并退出循环
              if (c != static_cast<char>(::toupper(c))) {
                result = false;
                break;
              }
              // 如果当前字符是字母，则设置结果为 true
              if (::isalpha(c)) {
                result = true;
              }
            }
            // 如果当前字符是字母，更新 prev_is_alpha 为 true；否则更新为 false
            if (::isalpha(c)) {
              prev_is_alpha = true;
            } else {
              prev_is_alpha = false;
            }
          }
          // 将最终结果推送回栈中
          push(stack, result);
        },
        // 使用模式选择方案生成的别名分析
        aliasAnalysisFromSchema()),
    // 生成 isprintable 操作符的定义，用于判断字符串中字符是否可打印
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::isprintable(str self) -> bool"),
        [](Stack& stack) {
          // 从堆栈中取出字符串，并转换为标准字符串类型
          std::string string = pop(stack).toStringRef();
          // 使用 all_of 算法检查字符串中的每个字符是否是字母、数字、标点符号或空格
          auto result = std::all_of(string.begin(), string.end(), [](char c) {
            return ::isalnum(c) || ::ispunct(c) || c == ' ';
          });
          // 将检查结果压入堆栈
          push(stack, result);
        },
        aliasAnalysisFromSchema()),

    // 生成 ljust 操作符的定义，用于将字符串左对齐并填充到指定宽度
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ljust(str self, int width, str fillchar=' ') -> str"),
        [](Stack& stack) {
          // 从堆栈中取出填充字符、宽度和字符串本身
          std::string fillchar = pop(stack).toStringRef();
          int64_t width = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          // 如果填充字符不是一个字符，抛出运行时错误
          if (fillchar.size() != 1) {
            // TODO: 应该抛出 TypeError
            throw std::runtime_error(
                "TypeError: The fill character must be exactly one character long");
          }
          // 计算需要填充的字符数
          auto to_append =
              std::max(int64_t(0), width - static_cast<int64_t>(string.size()));

          // 使用 stringstream 构建新的字符串，左对齐并填充
          std::stringstream ss;
          ss << string;
          for (const auto i : c10::irange(to_append)) {
            (void)i; // 抑制未使用变量的警告
            ss << fillchar;
          }
          // 将生成的字符串压入堆栈
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    // 生成 rjust 操作符的定义，用于将字符串右对齐并填充到指定宽度
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rjust(str self, int width, str fillchar=' ') -> str"),
        [](Stack& stack) {
          // 从堆栈中取出填充字符、宽度和字符串本身
          std::string fillchar = pop(stack).toStringRef();
          int64_t width = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          // 如果填充字符不是一个字符，抛出运行时错误
          if (fillchar.size() != 1) {
            // TODO: 应该抛出 TypeError
            throw std::runtime_error(
                "TypeError: The fill character must be exactly one character long");
          }
          // 计算需要填充的字符数
          auto to_append =
              std::max(int64_t(0), width - static_cast<int64_t>(string.size()));

          // 使用 stringstream 构建新的字符串，右对齐并填充
          std::stringstream ss;
          for (const auto i : c10::irange(to_append)) {
            (void)i; // 抑制未使用变量的警告
            ss << fillchar;
          }
          ss << string;
          // 将生成的字符串压入堆栈
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    // 生成 zfill 操作符的定义，用于在字符串左侧填充零字符到指定宽度
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::zfill(str self, int width) -> str"),
        [](Stack& stack) {
          // 从堆栈中取出宽度和字符串本身
          int64_t width = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          // 计算需要填充的零字符数
          auto to_append =
              std::max(int64_t(0), width - static_cast<int64_t>(string.size()));

          // 使用 stringstream 构建新的字符串，在左侧填充零字符
          std::stringstream ss;
          for (const auto i : c10::irange(to_append)) {
            (void)i; // 抑制未使用变量的警告
            ss << '0';
          }
          ss << string;
          // 将生成的字符串压入堆栈
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::lstrip(str self, str chars=' \\n\\t\\f\\v') -> str"),
        [](Stack& stack) {
          // 从堆栈中弹出字符串 chars，并转换为 std::string 类型
          std::string chars = pop(stack).toStringRef();
          // 从堆栈中弹出字符串 self，并转换为 std::string 类型
          std::string string = pop(stack).toStringRef();
          // 查找字符串 self 中第一个不属于 chars 的字符的位置
          auto index = string.find_first_not_of(chars);
          // 如果找到了不属于 chars 的字符
          if (index != std::string::npos) {
            // 从 index 处开始截取字符串 self，并更新 string
            string = string.substr(index, string.size());
          } else {
            // 如果字符串 self 全部属于 chars，则将 string 设为空字符串
            string = "";
          }
          // 将处理后的字符串压入堆栈
          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rstrip(str self, str chars=' \\n\\t\\f\\v') -> str"),
        [](Stack& stack) {
          // 从堆栈中弹出字符串 chars，并转换为 std::string 类型
          std::string chars = pop(stack).toStringRef();
          // 从堆栈中弹出字符串 self，并转换为 std::string 类型
          std::string string = pop(stack).toStringRef();
          // 查找字符串 self 中最后一个不属于 chars 的字符的位置
          auto index = string.find_last_not_of(chars);
          // 如果找到了不属于 chars 的字符
          if (index != std::string::npos) {
            // 从字符串 self 开始位置到 index+1 处截取，并更新 string
            string = string.substr(0, index + 1);
          } else {
            // 如果字符串 self 全部属于 chars，则将 string 设为空字符串
            string = "";
          }
          // 将处理后的字符串压入堆栈
          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::replace(str self, str old, str new, int max=-1) -> str"),
        [](Stack& stack) {
          // 从堆栈中弹出整数 max，并转换为 int64_t 类型
          int64_t max = pop(stack).toInt();
          // 从堆栈中弹出字符串 new，并转换为 std::string 类型
          std::string new_str = pop(stack).toStringRef();
          // 从堆栈中弹出字符串 old，并转换为 std::string 类型
          std::string old_str = pop(stack).toStringRef();
          // 从堆栈中弹出字符串 self，并转换为 std::string 类型
          std::string string = pop(stack).toStringRef();
          // 记录替换次数
          int64_t occurrences = 0;
          // 初始化查找位置为字符串起始位置
          std::string::size_type pos = 0;
          // 循环查找并替换 old_str
          while ((pos = string.find(old_str, pos)) != std::string::npos) {
            // 如果指定了替换次数，并且当前替换次数超过了 max，则停止替换
            if (max >= 0 && ++occurrences > max) {
              break;
            }
            // 在 pos 处开始替换 old_str 为 new_str，并更新 pos
            string = string.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
          }
          // 将处理后的字符串压入堆栈
          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::partition(str self, str separator) -> (str, str, str)"),
        [](Stack& stack) {
          // 从堆栈中弹出字符串 separator，并转换为 std::string 类型
          std::string separator = pop(stack).toStringRef();
          // 从堆栈中弹出字符串 self，并转换为 std::string 类型
          std::string string = pop(stack).toStringRef();
          // 查找字符串 self 中第一个 separator 的位置
          auto pos = string.find(separator, 0);
          // 如果找不到 separator
          if (pos == std::string::npos) {
            // 将 pos 设为字符串长度，并将 separator 设为空字符串
            pos = string.size();
            separator = "";
          }
          // 截取字符串 self，从起始位置到 pos 为止，作为 pre_partition
          auto pre_partition = string.substr(0, pos);
          // 截取字符串 self，从 pos+separator.size() 到结尾，作为 post_partition
          auto post_partition =
              string.substr(pos + separator.size(), string.size());
          // 将 pre_partition、separator、post_partition 压入堆栈
          push(stack, pre_partition, separator, post_partition);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rpartition(str self, str separator) -> (str, str, str)"),
        [](Stack& stack) {
          // 弹出栈顶的字符串作为分隔符
          std::string separator = pop(stack).toStringRef();
          // 弹出栈顶的字符串作为要操作的字符串
          std::string string = pop(stack).toStringRef();
          // 在字符串中查找最后一个分隔符的位置
          auto pos = string.find(separator, 0);
          auto rpos = pos;
          // 循环查找直到找不到分隔符
          do {
            pos = rpos;
            rpos = string.find(separator, pos + 1);
          } while (rpos != std::string::npos);
    
          // 如果没有找到分隔符，将 pos 设置为 0，并清空分隔符
          if (pos == std::string::npos) {
            pos = 0;
            separator = "";
          }
    
          // 获取分割点之前的部分
          auto pre_partition = string.substr(0, pos);
          // 获取分割点之后的部分
          auto post_partition =
              string.substr(pos + separator.size(), string.size());
          // 将结果压入栈中
          push(stack, pre_partition, separator, post_partition);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rsplit(str self, str separator=' ', int max=-1) -> str[]"),
        [](Stack& stack) {
          // 弹出栈顶的整数作为最大分割次数
          int64_t max = pop(stack).toInt();
          // 弹出栈顶的字符串作为分隔符
          std::string separator = pop(stack).toStringRef();
          // 弹出栈顶的字符串作为要操作的字符串
          std::string string = pop(stack).toStringRef();
          
          // 反转分隔符和要操作的字符串
          std::reverse(separator.begin(), separator.end());
          std::reverse(string.begin(), string.end());
    
          std::string::size_type prev_pos = 0;
          std::string::size_type pos = 0;
          c10::List<std::string> splits;
          auto count = 0;
          // 在反转后的字符串中查找反转后的分隔符，进行分割
          while ((pos = string.find(separator, pos)) != std::string::npos) {
            count++;
            if (max >= 0 && count > max) {
              break;
            } else {
              auto substr = string.substr(prev_pos, pos - prev_pos);
              // 将每个分割后的子串反转回来，然后加入到结果列表的开头
              std::reverse(substr.begin(), substr.end());
              splits.emplace(splits.begin(), substr);
            }
            pos += separator.size();
            prev_pos = pos;
          }
          // 处理剩余的部分，反转回来并加入结果列表的开头
          auto substr = string.substr(prev_pos, string.size() - prev_pos);
          std::reverse(substr.begin(), substr.end());
          splits.emplace(splits.begin(), substr);
          // 将结果列表压入栈中
          push(stack, std::move(splits));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::join(str self, str[] values) -> str"),
        // 定义 lambda 函数，处理 "aten::join" 操作符
        [](Stack& stack) {
          // 从堆栈中弹出一个 IValue 对象
          IValue ivalue = pop(stack);
          // 将 IValue 转换为 c10::ArrayRef，表示一个 IValue 数组的引用
          c10::ArrayRef<IValue> ivalues = ivalue.toListRef();
          // 创建一个 c10::List<std::string>，用于保存字符串数组
          c10::List<std::string> values;
          // 遍历 ivalues 数组，将每个元素转换为字符串并添加到 values 列表中
          for (const auto& v : ivalues) {
            values.emplace_back(v.toStringRef());
          }
          // 从堆栈中弹出一个可选的 std::string
          std::optional<std::string> opt_string =
              pop(stack).toOptional<std::string>();
          // 如果可选的字符串有值，则将其赋给 string，否则使用空字符串
          const std::string& string = opt_string.value_or("");
          // 创建一个 stringstream 对象 ss，用于构建最终的字符串结果
          std::stringstream ss;
          // 遍历 values 列表中的字符串，将它们逐个加入 stringstream
          for (auto it = values.begin(); it != values.end(); ++it) {
            ss << static_cast<std::string>(*it);
            // 如果当前不是最后一个字符串，则加入分隔符 string
            if (it != values.end() - 1) {
              ss << string;
            }
          }
          // 将 stringstream 中构建的字符串结果压入堆栈
          push(stack, ss.str());
        },
        // 根据定义的操作符生成别名分析信息
        aliasAnalysisFromSchema()),
};

// 创建字符串操作符注册对象，并传入操作符生成参数
RegisterOperators regStrOps(createOperators(stringOpGenArgs));

// 定义静态常量，包含操作符生成参数列表
static const std::vector<OperatorGeneratorArgs> opGenArgs1{
    // 操作符：prim::rangelist(int n) -> int[]
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::rangelist(int n) -> int[]"),
        [](Stack& stack) {
          // 从栈中弹出整数 n
          int64_t n;
          pop(stack, n);
          // 创建一个整数列表 elems，并预留 n 个空间
          c10::List<int64_t> elems;
          elems.reserve(n);
          // 遍历从 0 到 n-1 的整数，并将它们加入 elems 列表
          for (const auto i : c10::irange(n)) {
            elems.push_back(i);
          }
          // 将 elems 列表推入栈中
          push(stack, std::move(elems));
        },
        aliasAnalysisFromSchema()),

    // 操作符：prim::NumToTensor.bool(bool a) -> Tensor
    // 注：此操作符需要与 Scalar -> Tensor 的转换共享名称，因为所有 _to_tensor 转换都必须具有相同的操作符名称
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::NumToTensor.bool(bool a) -> Tensor"),
        numToTensorBool,
        aliasAnalysisFromSchema()),

    // 操作符：aten::device(str a) -> Device
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::device(str a) -> Device"),
        [](Stack& stack) {
          // 从栈中弹出字符串 a，并将其转换为 Device 对象推入栈中
          push(stack, c10::Device(pop(stack).toStringRef()));
        },
        aliasAnalysisFromSchema()),

    // 操作符：aten::device.with_index(str type, int index) -> Device
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::device.with_index(str type, int index) -> Device"),
        device_with_index,
        aliasAnalysisFromSchema()),

    // 操作符：aten::percentFormat(str self, ...) -> str
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::percentFormat(str self, ...) -> str"),
        [](Stack& stack) {
          // 从栈中弹出一个整数，表示输入参数的数量，然后调用 percentFormat 函数
          size_t num_inputs = pop(stack).toInt();
          percentFormat(stack, num_inputs);
        },
        aliasAnalysisFromSchema()),

    // 操作符：aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        [](Stack& stack) {
          at::Tensor self;
          bool non_blocking;
          bool copy;
          // 从栈中依次弹出 self, non_blocking, copy 参数，并调用 to_dispatch 函数进行转换
          pop(stack, self, non_blocking, copy);
          std::optional<c10::Device> device = c10::nullopt;
          std::optional<at::ScalarType> scalarType = c10::nullopt;
          push(
              stack, to_dispatch(self, device, scalarType, non_blocking, copy));
        },
        aliasAnalysisFromSchema()),

    // 操作符：prim::requires_grad(Tensor a) -> bool
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::requires_grad(Tensor a) -> bool"),
        [](Stack& stack) {
          // 从栈中弹出 Tensor a，并将其是否需要梯度的布尔值推入栈中
          at::Tensor a;
          pop(stack, a);
          push(stack, a.requires_grad());
        },
        aliasAnalysisFromSchema()),

    // 操作符：prim::grad(Tensor a) -> Tensor(*)
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::grad(Tensor a) -> Tensor(*)"),
        [](Stack& stack) {
          // 从栈中弹出 Tensor a，并将其梯度 Tensor 推入栈中
          at::Tensor a;
          pop(stack, a);
          push(stack, a.grad());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_sparse(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_sparse());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_sparse_csr(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_sparse_csr());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_mkldnn(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_mkldnn());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_mps(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_mps());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_vulkan(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_vulkan());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_ipu(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_ipu());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_quantized(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_quantized());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_meta(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_meta());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_maia(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_maia());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_nested(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_nested());
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::name(Tensor a) -> str?"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          // 检查张量是否有命名，若无则推入空值，否则推入张量的名称字符串
          if (a.name().empty()) {
            push(stack, IValue());
          } else {
            push(stack, a.name());
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::nbytes(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          // 获取张量 `a` 的字节数，并转换为 int64_t 类型
          const auto nbytes = static_cast<int64_t>(a.nbytes());
          // 将计算得到的字节数压入栈中
          push(stack, nbytes);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::itemsize(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          // 获取张量 `a` 的每个元素的字节数，并转换为 int64_t 类型
          const auto itemsize = static_cast<int64_t>(a.itemsize());
          // 将计算得到的每个元素的字节数压入栈中
          push(stack, itemsize);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::index(Device self) -> int?"),
        [](Stack& stack) {
          // 弹出栈顶的 Device 对象，并将其转换为 c10::Device 类型
          auto d = pop(stack).toDevice();
          // 检查 Device 对象是否具有索引，如果有则将索引值压入栈中，否则压入空值
          if (d.has_index()) {
            push(stack, d.index());
          } else {
            push(stack, IValue());
          }
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        // TODO return generator object when torchscript supports RNG
        // first-class
        TORCH_SELECTIVE_SCHEMA("aten::manual_seed(int seed) -> ()"),
        [](Stack& stack) { at::manual_seed(pop(stack).toInt()); },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::Generator(*, Device? device=None, int? seed=None) -> Generator"),
        [](Stack& stack) {
          // 弹出栈顶的 seed 和 device，分别转换为 int64_t 和 c10::Device 类型
          auto seed = pop(stack).toOptional<int64_t>();
          auto device = pop(stack).toOptional<c10::Device>();
          // 根据给定的设备和种子创建一个新的生成器对象，并将其压入栈中
          push(
              stack,
              torch::jit::make_generator_for_device(
                  device.value_or(c10::Device("cpu")), seed));
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::initial_seed(Generator self) -> int"),
        [](Stack& stack) {
          // 弹出栈顶的生成器对象，并获取其当前的种子值
          auto generator = pop(stack);
          auto current_seed = generator.toGenerator().current_seed();
          // 将当前种子值转换为 int64_t 类型并压入栈中
          push(stack, (int64_t)current_seed);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::manual_seed.generator(Generator(a!) self, int seed) -> Generator(a!)"),
        [](Stack& stack) {
          // 弹出栈顶的种子值和生成器对象，并将种子值设置为生成器的当前种子
          auto seed = pop(stack).toInt();
          auto generator = pop(stack);
          generator.toGenerator().set_current_seed(seed);
          // 将更新后的生成器对象压入栈中
          push(stack, generator);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::seed(Generator(a!) self) -> int"),
        [](Stack& stack) {
          // 弹出栈顶的生成器对象，并获取其种子值
          auto generator = pop(stack);
          auto current_seed = generator.toGenerator().seed();
          // 将种子值转换为 int64_t 类型并压入栈中
          push(stack, (int64_t)current_seed);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::cuda(Tensor(a) self) -> Tensor(a|b)"),
        [](Stack& stack) {
          // 从堆栈中弹出一个张量，并将其移到 GPU 上
          at::Tensor a;
          pop(stack, a);
          push(stack, a.cuda());
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradZero() -> Tensor"),
        [](Stack& stack) {
          // 在堆栈上推入一个空的张量
          stack.emplace_back(at::Tensor());
        },
        aliasAnalysisSpecialCase()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::ReductionSizes(int[] size, int[] red_axes, bool keepdim = False) -> int[]"),
        [](Stack& stack) {
          // 从堆栈中弹出参数
          bool keepdim = pop(stack).toBool();
          c10::List<int64_t> axes = pop(stack).toIntList();
          c10::List<int64_t> size = pop(stack).toIntList();
          if (keepdim) {
            // 如果 keepdim 为真，则将指定轴上的尺寸设为 1
            for (const auto& axis : axes) {
              size.set(axis, 1);
            }
          } else {
            // 否则，移除指定轴上的尺寸
            int64_t index = 0;
            auto iter = size.begin();
            std::sort(axes.begin(), axes.end());
            for (const auto& axis : axes) {
              // 将 iter 移动到下一个轴
              iter += axis - index;
              // 将 iter 指向的轴更新为 axis + 1，并移动 iter
              iter = size.erase(iter);
              // 更新 iter 的当前索引
              index = axis + 1;
            }
          }
          // 将处理后的尺寸推回堆栈
          push(stack, IValue(std::move(size)));
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::BroadcastSizes(...) -> int[]"),
        [](Stack& stack) {
          // 从堆栈中弹出输入数量
          auto num_inputs = pop(stack).toInt();
          std::vector<int64_t> size;
          size.reserve(8); // 预先分配内存
          for (const auto i : c10::irange(num_inputs)) {
            // 使用输入的尺寸推断输出尺寸
            size = at::infer_size(size, peek(stack, i, num_inputs).toDimVector());
          }
          // 从堆栈中移除输入数量
          drop(stack, num_inputs);
          // 将推断的尺寸推回堆栈
          push(stack, IValue(size));
        },
        aliasAnalysisSpecialCase()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::warn(str message, int stacklevel=2) -> ()"),
        [](Stack& stack) {
          // 报错，因为 warn 函数直接在解释器中实现
          TORCH_CHECK(false, "warn is implemented directly in the interpreter");
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "onnx::Reshape(Tensor input, Tensor shape) -> Tensor"),
        [](Stack& stack) {
          // 从堆栈中弹出输入张量和形状张量
          at::Tensor input, shape;
          pop(stack, input, shape);
          // 确保形状张量是连续的，并且是一维的
          shape = shape.contiguous();
          AT_ASSERT(shape.ndimension() == 1);
          // 将输入张量按照形状张量重新形状，并推回堆栈
          at::IntArrayRef shape_list(
              shape.const_data_ptr<int64_t>(), shape.size(0));
          push(stack, input.reshape(shape_list));
        },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("onnx::Shape(Tensor t) -> Tensor"),
        [](Stack& stack) {
          auto t = pop(stack).toTensor();  // 弹出栈顶元素，将其转换为 Tensor 类型
          at::IntArrayRef sizes = t.sizes();  // 获取张量 t 的大小信息
          auto sizes_tensor = torch::empty(
              {static_cast<int64_t>(sizes.size())}, at::dtype(at::kLong));  // 创建一个大小为 sizes.size() 的长整型张量
          auto accessor = sizes_tensor.accessor<int64_t, 1>();  // 获取 sizes_tensor 的访问器
          for (const auto i : c10::irange(sizes.size())) {  // 遍历 sizes 中的每一个维度
            accessor[i] = sizes[i];  // 将每个维度的大小存入 sizes_tensor 中
          }
          stack.emplace_back(sizes_tensor);  // 将 sizes_tensor 压入栈中
        },
        aliasAnalysisSpecialCase()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAnyNonZero(...) -> bool"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();  // 弹出栈顶元素，转换为整数，表示输入的数量
          bool result = false;  // 初始化结果为 false
          for (const IValue& v : last(stack, num_inputs)) {  // 遍历栈顶的 num_inputs 个元素
            if (v.isTensor()) {  // 如果元素是 Tensor 类型
              if (v.toTensor().defined()) {  // 如果 Tensor 已定义（非空）
                result = true;  // 设置结果为 true
                break;  // 结束循环
              }
            } else if (v.isTensorList()) {  // 如果元素是 Tensor 列表类型
              for (const at::Tensor& t : v.toTensorVector()) {  // 遍历 Tensor 列表中的每个 Tensor
                if (t.defined()) {  // 如果 Tensor 已定义（非空）
                  result = true;  // 设置结果为 true
                }
              }
              if (result) {  // 如果结果已经为 true，则结束循环
                break;
              }
            } else {
              TORCH_INTERNAL_ASSERT(false);  // 抛出内部断言异常，不应该出现其他类型的输入
            }
          }
          drop(stack, num_inputs);  // 丢弃栈顶的 num_inputs 个元素
          stack.emplace_back(result);  // 将结果压入栈中
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAllZero(...) -> bool"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();  // 弹出栈顶元素，转换为整数，表示输入的数量
          bool result = true;  // 初始化结果为 true
          for (const IValue& v : last(stack, num_inputs)) {  // 遍历栈顶的 num_inputs 个元素
            TORCH_INTERNAL_ASSERT(v.isTensor());  // 断言元素为 Tensor 类型
            if (v.toTensor().defined()) {  // 如果 Tensor 已定义（非空）
              result = false;  // 设置结果为 false
              break;  // 结束循环
            }
          }
          drop(stack, num_inputs);  // 丢弃栈顶的 num_inputs 个元素
          stack.emplace_back(result);  // 将结果压入栈中
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAllNonZero(...) -> bool"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();  // 弹出栈顶元素，转换为整数，表示输入的数量
          bool result = true;  // 初始化结果为 true
          for (const IValue& v : last(stack, num_inputs)) {  // 遍历栈顶的 num_inputs 个元素
            TORCH_INTERNAL_ASSERT(v.isTensor());  // 断言元素为 Tensor 类型
            if (!v.toTensor().defined()) {  // 如果 Tensor 未定义（空）
              result = false;  // 设置结果为 false
              break;  // 结束循环
            }
          }
          drop(stack, num_inputs);  // 丢弃栈顶的 num_inputs 个元素
          stack.emplace_back(result);  // 将结果压入栈中
        },
        aliasAnalysisFromSchema()),
    # 定义 OperatorGeneratorArgs 对象，包括函数原型和对应的实现
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAdd(Any a, Any b) -> Any"),
        # 实现函数，根据输入的值进行加法运算
        [](Stack& stack) {
          # 弹出栈顶的两个值
          IValue i_a = pop(stack);
          IValue i_b = pop(stack);
          # 如果两个值都是 None，则返回一个空的 Tensor
          if (i_a.isNone() && i_b.isNone()) {
            stack.emplace_back(at::Tensor{});
            return;
          }
          # 如果 i_a 是 None，则将 i_b 转换为 Tensor 并放入栈中
          if (i_a.isNone()) {
            stack.emplace_back(i_b.toTensor());
            return;
          }
          # 如果 i_b 是 None，则将 i_a 转换为 Tensor 并放入栈中
          if (i_b.isNone()) {
            stack.emplace_back(i_a.toTensor());
            return;
          }
          # 将 i_a 和 i_b 转换为 Tensor
          at::Tensor a = i_a.toTensor();
          at::Tensor b = i_b.toTensor();
          # 如果 a 和 b 都未定义，则返回 a
          if (!a.defined() && !b.defined()) {
            stack.emplace_back(a);
          } else if (!a.defined()) {
            stack.emplace_back(b);
          } else if (!b.defined()) {
            stack.emplace_back(a);
          } else {
            # 将 a 和 b 相加并放入栈中
            stack.emplace_back(a + b);
          }
        },
        aliasAnalysisSpecialCase()),

    # 定义 OperatorGeneratorArgs 对象，包括函数原型和对应的实现
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_size_if_not_equal(int[] self_size, int[] other_size) -> int[]?"),
        # 实现函数，比较两个尺寸是否相等
        [](Stack& stack) {
          # 弹出栈顶的两个值
          IValue self_size, other_size;
          pop(stack, self_size, other_size);
          # 转换为维度向量
          auto s = self_size.toDimVector();
          auto o = other_size.toDimVector();
          # 如果两个尺寸相等，则放入空值
          if (s == o) {
            stack.emplace_back();
          } else {
            stack.emplace_back(std::move(self_size));
          }
        },
        aliasAnalysisFromSchema()),

    # 定义 OperatorGeneratorArgs 对象，包括函数原型和对应的实现
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unwrap_optional(t(a)? optional) -> t(a)"),
        # 实现函数，解包 Optional 值
        [](Stack& stack) {
          # 弹出栈顶的值
          auto val = pop(stack);
          # 检查值是否为 None，如果是则报错
          TORCH_CHECK(!val.isNone(), "Unwrapping null optional");
          # 将值放回栈中
          push(stack, std::move(val));
        },
        aliasAnalysisFromSchema())};
RegisterOperators reg1(createOperators(opGenArgs1));

void hashValue(Stack& stack) {
  auto value = pop(stack);  // 从栈中弹出一个元素
  push(stack, value.hash());  // 计算该元素的哈希值并推入栈中
}

static const std::vector<OperatorGeneratorArgs> opGenArgs2{
    // 将这些操作注册为 Any[] 类型，以便可以使用 len() 函数处理异构元组
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.any(Any[] a) -> int"),
        listLen,
        aliasAnalysisFromSchema()),

    // 下面的操作针对列表元素类型有专门的实现

    // 定义针对 int 类型的列表的操作
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::remove.int(int[](a!) self, int el) -> ()"),
        listRemove<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index.list_int(int[] self, int el) -> int"),
        listIndex<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::count.int(int[] self, int el) -> int"),
        listCount<int64_t>,
        aliasAnalysisFromSchema()),

    // 定义针对 float 类型的列表的操作
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::remove.float(float[](a!) self, float el) -> ()"),
        listRemove<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index.list_float(float[] self, float el) -> int"),
        listIndex<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::count.float(float[] self, float el) -> int"),
        listCount<double>,
        aliasAnalysisFromSchema()),

    // 各种其他类型的列表操作类似地定义...

    // `listContains<T>` 操作对非原始类型未实现
    // TODO: 在 .to<c10::List<bool>> 不再抛出错误后，添加对 List[bool] 的支持
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.float_list(float[] l, float item) -> bool"),
        listContains<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.int(int[](a!) self, bool reverse=False) -> ()"),
        listSort<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.float(float[](a!) self, bool reverse=False) -> ()"),
        listSort<double>,
        aliasAnalysisFromSchema()),
};
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.Tensor(Tensor[](a!) self, bool reverse=False) -> ()"),
        listSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对 Tensor 列表进行排序的操作，不改变输入 (a!)，可以选择是否反转排序
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.bool(bool[](a!) self, bool reverse=False) -> ()"),
        listSort<bool>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对布尔类型列表进行排序的操作，不改变输入 (a!)，可以选择是否反转排序
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.str(str[](a!) self, bool reverse=False) -> ()"),
        listSort<std::string>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对字符串类型列表进行排序的操作，不改变输入 (a!)，可以选择是否反转排序
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sorted.int(int[](a) input) -> (int[])"),
        listCopyAndSort<int64_t>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对整型列表进行复制并排序的操作，返回排序后的整型列表
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sorted.float(float[](a) input) -> (float[])"),
        listCopyAndSort<double>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对浮点数列表进行复制并排序的操作，返回排序后的浮点数列表
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sorted.Tensor(Tensor[](a) input) -> (Tensor[])"),
        listCopyAndSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对 Tensor 类型列表进行复制并排序的操作，返回排序后的 Tensor 类型列表
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sorted.bool(bool[](a) input) -> (bool[])"),
        listCopyAndSort<bool>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对布尔类型列表进行复制并排序的操作，返回排序后的布尔类型列表
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sorted.str(str[](a) input) -> (str[])"),
        listCopyAndSort<std::string>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对字符串类型列表进行复制并排序的操作，返回排序后的字符串类型列表
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.float_list(float[] a, float[] b) -> bool"),
        listEq<double>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个浮点数列表进行相等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.Tensor_list(Tensor[] a, Tensor[] b) -> bool"),
        listEq<at::Tensor>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个 Tensor 类型列表进行相等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.bool_list(bool[] a, bool[] b) -> bool"),
        listEq<bool>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个布尔类型列表进行相等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.str_list(str[] a, str[] b) -> bool"),
        listEq<std::string>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个字符串类型列表进行相等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.float_list(float[] a, float[] b) -> bool"),
        listNe<double>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个浮点数列表进行不等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool"),
        listNe<at::Tensor>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个 Tensor 类型列表进行不等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.bool_list(bool[] a, bool[] b) -> bool"),
        listNe<bool>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个布尔类型列表进行不等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.str_list(str[] a, str[] b) -> bool"),
        listNe<std::string>,
        aliasAnalysisFromSchema()),
    # 定义操作生成器参数：对于 Torch 的选择性模式，生成对两个字符串类型列表进行不等性比较的操作，返回比较结果
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sorted.any(t[](a) self) -> (t[])"),
        sort_op</*has_reverse_arg*/ false, /*copy_return_list*/ true>,
        aliasAnalysisFromSchema()),

# 创建一个 OperatorGeneratorArgs 对象，用于生成有选择性的 PyTorch 操作的模式。
# 使用 TORCH_SELECTIVE_SCHEMA 定义了特定操作的模式，这里是对不带 reverse 参数的 sorted 操作的定义。
# 使用 sort_op 模板函数生成排序操作对象，不带 reverse 参数，且返回列表的副本。
# 使用 aliasAnalysisFromSchema() 来获取与模式相关联的别名分析信息。


    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.any(t[](a!) self, bool reverse=False) -> ()"),
        sort_op</*has_reverse_arg*/ true, /*copy_return_list*/ false>,
        aliasAnalysisFromSchema()),

# 创建另一个 OperatorGeneratorArgs 对象，用于生成有选择性的 PyTorch 操作的模式。
# 使用 TORCH_SELECTIVE_SCHEMA 定义了特定操作的模式，这里是对带有 reverse 参数的 sort 操作的定义。
# 使用 sort_op 模板函数生成排序操作对象，带有 reverse 参数，且不返回列表的副本。
# 使用 aliasAnalysisFromSchema() 来获取与模式相关联的别名分析信息。
#define DEFINE_CONVERT_BASE_OP(op_name, prefix, char_op) \
  OperatorGeneratorArgs(                                 \  # 定义一个宏，用于生成基本转换操作的运算符
      TORCH_SELECTIVE_SCHEMA(#op_name "(int i) -> str"), \  # 使用 Torch 的选择性模式定义运算符的 schema
      [](Stack& stack) {                                 \  # 定义一个 lambda 函数，接受一个栈参数 stack
        auto i = pop(stack).toInt();                     \  # 从栈中弹出一个整数，并转换为 int 类型
        std::stringstream ss;                            \  # 创建一个字符串流 ss
        if (i < 0) {                                     \  # 如果 i 小于 0
          ss << "-";                                     \  # 在 ss 中添加负号 '-'
          i = -i;                                        \  # 取 i 的绝对值
        }                                                \
        ss << "0" << prefix << char_op << i;             \  # 将 "0"、prefix、char_op 和 i 写入 ss
        push(stack, ss.str());                           \  # 将 ss 转换为字符串并推送到栈中
      },                                                 \
      aliasAnalysisFromSchema())                         \  # 从 schema 中进行别名分析

    DEFINE_CONVERT_BASE_OP(aten::hex, "x", std::hex),     \  # 调用宏生成十六进制转换的运算符
    DEFINE_CONVERT_BASE_OP(aten::oct, "o", std::oct),     \  # 调用宏生成八进制转换的运算符

    OperatorGeneratorArgs(                               \  # 定义一个运算符生成器参数
        TORCH_SELECTIVE_SCHEMA("aten::bin(int i) -> str"),\  # 使用 Torch 的选择性模式定义二进制转换的 schema
        [](Stack& stack) {                               \  # 定义一个 lambda 函数，接受一个栈参数 stack
          auto i = pop(stack).toInt();                   \  # 从栈中弹出一个整数，并转换为 int 类型
          std::stringstream ss;                          \  # 创建一个字符串流 ss
          if (i == 0) {                                  \  # 如果 i 等于 0
            push(stack, "0b0");                          \  # 推送字符串 "0b0" 到栈中
          } else {                                       \
            if (i < 0) {                                 \  # 如果 i 小于 0
              ss << "-";                                 \  # 在 ss 中添加负号 '-'
              i = -i;                                    \  # 取 i 的绝对值
            }                                            \
            std::string str = std::bitset<8 * sizeof(i)>(i).to_string(); \  # 将 i 转换为二进制字符串
            str.erase(0, std::min(str.find_first_not_of('0'), str.size() - 1)); \  # 删除字符串中多余的零
            ss << "0b" << str;                           \  # 将二进制表示添加到 ss 中
            push(stack, ss.str());                       \  # 将 ss 转换为字符串并推送到栈中
          }                                              \
        },                                               \
        aliasAnalysisFromSchema()),                      \  # 从 schema 中进行别名分析

    // TODO: deprecate this in favor of aten::getelem
    OperatorGeneratorArgs(                               \  # 定义一个运算符生成器参数，用于字符串索引
        TORCH_SELECTIVE_SCHEMA(                          \  # 使用 Torch 的选择性模式定义字符串索引操作的 schema
            "prim::StringIndex(str string, int index) -> str"), \  # 操作为从字符串中获取指定索引的字符
        [](Stack& stack) {                               \  # 定义一个 lambda 函数，接受一个栈参数 stack
          auto index = pop(stack).toInt();               \  # 从栈中弹出一个整数作为索引
          auto string = pop(stack).toStringRef();        \  # 从栈中弹出一个字符串引用
          auto norm_index = normalizeIndex(index, string.size()); \  # 根据字符串长度规范化索引值
          char c = string.at(norm_index);                \  # 获取字符串中规范化后索引位置的字符
          push(stack, std::string(&c, 1));               \  # 将字符 c 转换为字符串并推送到栈中
        },                                               \
        aliasAnalysisFromSchema()),                      \  # 从 schema 中进行别名分析

    OperatorGeneratorArgs(                               \  # 定义一个运算符生成器参数，用于将整数转换为字符
        TORCH_SELECTIVE_SCHEMA("aten::chr(int i) -> str"), \  # 使用 Torch 的选择性模式定义整数转字符操作的 schema
        [](Stack& stack) {                               \  # 定义一个 lambda 函数，接受一个栈参数 stack
          auto i = pop(stack).toInt();                   \  # 从栈中弹出一个整数
          std::stringstream ss;                          \  # 创建一个字符串流 ss
          TORCH_CHECK(                                   \  # 检查条件是否满足，否则抛出异常
              i >= 0 && i < 1114111,                     \  # i 的值必须在指定范围内
              "chr() arg not in range(0x110000), found ", \  # 异常消息，显示超出范围的值
              i);                                        \
          char c = i;                                    \  # 将整数 i 转换为字符 c
          ss << c;                                       \  # 将字符 c 添加到 ss 中
          push(stack, ss.str());                         \  # 将 ss 转换为字符串并推送到栈中
        },                                               \
        aliasAnalysisFromSchema()),                      \  # 从 schema 中进行别名分析

    // only used in loop unrolling, not exposed to end users  \  # 仅在循环展开中使用，不向最终用户公开
    DEFINE_INT_OP(aten::__round_to_zero_floordiv, a / b),  \  # 使用宏定义整数除法运算符
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::modf(float a) -> (float, float)"),
        [](Stack& stack) {
          // 从堆栈中弹出一个浮点数 a
          double a;
          pop(stack, a);
          // 定义变量 b 和 c，将 modf 函数应用于 a，将整数部分保存在 c 中，小数部分保存在 b 中
          double b, c;
          b = modf(a, &c);
          // 将 b 和 c 推入堆栈
          push(stack, b, c);
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::frexp(float a) -> (float, int)"),
        [](Stack& stack) {
          // 从堆栈中弹出一个浮点数 a
          double a;
          pop(stack, a);
          // 定义变量 m 和 e，将 frexp 函数应用于 a，将尾数保存在 m 中，指数保存在 e 中
          double m;
          int e;
          std::frexp(a, &e);
          // 将 m 和 e 推入堆栈
          push(stack, m, e);
        },
        aliasAnalysisFromSchema()),
    
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ldexp(float x, int i) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个浮点数 x 和一个整数 i
          double a;
          int64_t b;
          pop(stack, a, b);
          // 将 ldexp 函数应用于 x 和 i，并将结果推入堆栈
          push(stack, std::ldexp(a, b));
        },
        aliasAnalysisFromSchema()),
    
    DEFINE_BINARY_FLOAT_OP(aten::mathremainder, std::remainder(a, b)),
    
    DEFINE_INT_OP(aten::__and__, a & b),
    DEFINE_INT_OP(aten::__or__, a | b),
    DEFINE_INT_OP(aten::__xor__, a ^ b),
    DEFINE_INT_OP(aten::__lshift__, a << b),
    DEFINE_INT_OP(aten::__rshift__, a >> b),
    
    DEFINE_GENERIC_BINARY_OP(
        aten::log,
        std::log(a) / std::log(b),
        float,
        complex),
    
    DEFINE_INT_FLOAT_OP(aten::log, std::log(a) / std::log(b), float),
    DEFINE_INT_COMPLEX_OP(aten::log, std::log(a) / std::log(b), complex),
    DEFINE_FLOAT_COMPLEX_OP(aten::log, std::log(a) / std::log(b), complex),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::log,
        std::log(a) / std::log(b),
        std::log(a) / std::log(b),
        float),
    
    DEFINE_UNARY_OP(aten::log1p, std::log1p(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::log10, std::log10(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::sqrt, std::sqrt(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::acos, std::acos(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::asin, std::asin(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::atan, std::atan(a), float, float),
    
    DEFINE_GENERIC_OP(
        aten::atan2,
        std::atan2(a, b),
        std::atan2(a, b),
        float,
        float),
    DEFINE_INT_FLOAT_OP(aten::atan2, std::atan2(a, b), float),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::atan2,
        std::atan2(a, b),
        std::atan2(a, b),
        float),
    
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::cos, std::cos(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::sin, std::sin(a), float, float),
    // 定义单目操作符 sin，对浮点数执行 std::sin 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::tan, std::tan(a), float, float),
    // 定义单目操作符 tan，对浮点数执行 std::tan 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::asinh, std::asinh(a), float, float),
    // 定义单目操作符 asinh，对浮点数执行 std::asinh 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::atanh, std::atanh(a), float, float),
    // 定义单目操作符 atanh，对浮点数执行 std::atanh 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::acosh, std::acosh(a), float, float),
    // 定义单目操作符 acosh，对浮点数执行 std::acosh 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::sinh, std::sinh(a), float, float),
    // 定义单目操作符 sinh，对浮点数执行 std::sinh 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::cosh, std::cosh(a), float, float),
    // 定义单目操作符 cosh，对浮点数执行 std::cosh 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::tanh, std::tanh(a), float, float),
    // 定义单目操作符 tanh，对浮点数执行 std::tanh 函数，返回浮点数
    DEFINE_UNARY_OP_WITH_COMPLEX_CAST(
        aten::angle,
        std::arg(a),
        float,
        float,
        float,
        float),
    // 定义带复数转换的单目操作符 angle，对复数执行 std::arg 函数，返回浮点数
    DEFINE_UNARY_OP(aten::degrees, degrees(a), float, float),
    // 定义单目操作符 degrees，对浮点数执行 degrees 函数，返回浮点数
    DEFINE_UNARY_OP(aten::radians, radians(a), float, float),
    // 定义单目操作符 radians，对浮点数执行 radians 函数，返回浮点数
    DEFINE_BINARY_FLOAT_OP(aten::fmod, std::fmod(a, b)),
    // 定义二元操作符 fmod，对浮点数执行 std::fmod 函数，返回浮点数
    DEFINE_UNARY_INT_OP(aten::factorial, factorial(a), int),
    // 定义单目操作符 factorial，对整数执行 factorial 函数，返回整数
    DEFINE_UNARY_FLOAT_OP(aten::isnan, std::isnan(a), bool),
    // 定义单目操作符 isnan，检查浮点数是否为 NaN，返回布尔值
    DEFINE_UNARY_FLOAT_OP(aten::isfinite, std::isfinite(a), bool),
    // 定义单目操作符 isfinite，检查浮点数是否有限，返回布尔值
    DEFINE_UNARY_FLOAT_OP(aten::isinf, std::isinf(a), bool),
    // 定义单目操作符 isinf，检查浮点数是否为无穷，返回布尔值
    DEFINE_UNARY_COMPLEX_OP(
        aten::isnan,
        std::isnan(a.real()) || std::isnan(a.imag()),
        bool),
    // 定义复数单目操作符 isnan，检查复数的实部或虚部是否为 NaN，返回布尔值
    DEFINE_UNARY_COMPLEX_OP(
        aten::isfinite,
        std::isfinite(a.real()) && std::isfinite(a.imag()),
        bool),
    // 定义复数单目操作符 isfinite，检查复数的实部和虚部是否有限，返回布尔值
    DEFINE_UNARY_COMPLEX_OP(
        aten::isinf,
        std::isinf(a.real()) || std::isinf(a.imag()),
        bool),
    // 定义复数单目操作符 isinf，检查复数的实部或虚部是否为无穷，返回布尔值
    DEFINE_UNARY_OP(aten::gamma, std::tgamma(a), float, float),
    // 定义单目操作符 gamma，对浮点数执行 std::tgamma 函数，返回浮点数
    DEFINE_UNARY_OP(aten::erf, std::erf(a), float, float),
    // 定义单目操作符 erf，对浮点数执行 std::erf 函数，返回浮点数
    DEFINE_UNARY_OP(aten::erfc, std::erfc(a), float, float),
    // 定义单目操作符 erfc，对浮点数执行 std::erfc 函数，返回浮点数
    DEFINE_UNARY_OP(aten::expm1, std::expm1(a), float, float),
    // 定义单目操作符 expm1，对浮点数执行 std::expm1 函数，返回浮点数
    DEFINE_UNARY_OP(aten::fabs, std::fabs(a), float, float),
    // 定义单目操作符 fabs，对浮点数执行 std::fabs 函数，返回浮点数
    DEFINE_UNARY_OP(aten::lgamma, std::lgamma(a), float, float),
    // 定义单目操作符 lgamma，对浮点数执行 std::lgamma 函数，返回浮点数

    // TODO: move abs to aten namespace because it's schematized!
    DEFINE_UNARY_OP_WITH_COMPLEX_CAST(
        prim::abs,
        std::abs(a),
        int,
        float,
        float,
        float),
    // 定义带复数转换的单目操作符 abs，对整数或浮点数执行 std::abs 函数，返回整数或浮点数
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::abs(Tensor x) -> Tensor"),
        // 定义 prim::abs(Tensor x) -> Tensor 的操作生成器
        [](Stack& stack) {
          at::Tensor x;
          // 弹出堆栈中的 Tensor x
          pop(stack, x);
          // 将 x 的绝对值推送到堆栈中
          push(stack, x.abs());
        },
        aliasAnalysisFromSchema()),
        // 使用模式别名分析生成别名

    DEFINE_INT_OP(aten::gcd, gcd(a, b)),
    // 定义整数二元操作符 gcd，对整数 a 和 b 执行 gcd 函数

    DEFINE_GENERIC_OP(
        aten::copysign,
        std::copysign(a, b),
        std::copysign(a, b),
        float,
        float),
    // 定义通用操作符 copysign，对浮点数 a 和 b 执行 std::copysign 函数，返回浮点数
    DEFINE_INT_FLOAT_OP(aten::copysign, std::copysign(a, b), float),
    // 定义整数和浮点数二元操作符 copysign，对 a 和 b 执行 std::copysign 函数，返回浮点数
    DEFINE_SCALAR_BINARY_OP(
        aten::copysign,
        std::copysign(a, b),
        std::copysign(a, b),
        float),
    // 定义标量二元操作符 copysign，对 a 和 b 执行 std::copysign 函数，返回浮点数
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::_tensor_to_list(Tensor self) -> int[]"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 Tensor 对象
          at::Tensor t;
          pop(stack, t);
          // 创建一个空的 int 类型列表 elems
          c10::List<int64_t> elems;
          elems.reserve(t.size(0)); // 预留空间，大小为 t 的第一维度大小
          // 遍历 t 的第一维度，将每个元素添加到 elems 中
          for (const auto i : c10::irange(t.size(0))) {
            elems.push_back(*t[i].const_data_ptr<int32_t>());
          }
          // 将 elems 推送回堆栈
          push(stack, std::move(elems));
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::_list_to_tensor(int[] self) -> Tensor"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 int 类型的列表，并转换为 c10::List<int64_t> 类型
          c10::List<int64_t> l = pop(stack).toIntList();
          // 创建一个大小为 l.size() 的 int 类型的 Tensor t
          auto t = torch::empty(
              {static_cast<int64_t>(l.size())}, at::dtype(at::kInt));
          // 将列表 l 中的元素逐个复制到 Tensor t 中
          for (const auto i : c10::irange(l.size())) {
            t[i] = l.get(i);
          }
          // 将 Tensor t 推送回堆栈
          push(stack, std::move(t));
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.int(int[] self) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 int 类型的列表，并转换为 c10::List<int64_t> 类型
          c10::List<int64_t> l = pop(stack).toIntList();
          auto sum = 0;
          // 计算列表 l 中所有元素的和
          for (const auto& elem : l) {
            sum += elem;
          }
          // 将计算结果 sum 推送回堆栈
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.float(float[] self) -> float"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 float 类型的列表，并转换为 c10::List<double> 类型
          c10::List<double> l = pop(stack).toDoubleList();
          auto sum = 0.0;
          // 计算列表 l 中所有元素的和
          for (const auto& elem : l) {
            sum += elem;
          }
          // 将计算结果 sum 推送回堆栈
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.complex(complex[] self) -> complex"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 complex 类型的列表，并转换为 c10::List<c10::complex<double>> 类型
          c10::List<c10::complex<double>> l = pop(stack).toComplexDoubleList();
          c10::complex<double> sum = 0.0;
          // 计算列表 l 中所有元素的复数和
          for (const auto i : c10::irange(l.size())) {
            sum = sum + l.extract(i);
          }
          // 将计算结果 sum 推送回堆栈
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.bool(bool[] self) -> int"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 bool 类型的列表，并转换为 c10::List<bool> 类型
          c10::List<bool> l = pop(stack).toBoolList();
          auto sum = 0;
          // 计算列表 l 中所有为 true 的元素个数
          for (const auto& elem : l) {
            if (elem) {
              sum += 1;
            }
          }
          // 将计算结果 sum 推送回堆栈
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.str(str[] self) -> bool"),
        [](Stack& stack) {
          // 从堆栈中弹出一个 str 类型的列表，并转换为 c10::List<std::string> 类型
          auto l = pop(stack).toList();
          // 检查列表 l 中是否存在非空字符串，推送 true 或 false 到堆栈
          for (const auto& elem : l) {
            if (elem != "") {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.int(int[] self) -> bool"),
        // Lambda function for 'any' operation on integer lists
        [](Stack& stack) {
          // Pop the stack and convert to C++ list of int64_t
          c10::List<int64_t> l = pop(stack).toIntList();
          // Iterate over the list elements
          for (const auto& elem : l) {
            // If any element is truthy (non-zero), push true to stack and return
            if (elem) {
              push(stack, true);
              return;
            }
          }
          // If no elements are truthy, push false to stack
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.float(float[] self) -> bool"),
        // Lambda function for 'any' operation on float lists
        [](Stack& stack) {
          // Pop the stack and convert to C++ list of double
          c10::List<double> l = pop(stack).toDoubleList();
          // Iterate over the list elements
          for (const auto& elem : l) {
            // If any element is truthy (non-zero), push true to stack and return
            if (elem) {
              push(stack, true);
              return;
            }
          }
          // If no elements are truthy, push false to stack
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.bool(bool[] self) -> bool"),
        // Lambda function for 'any' operation on boolean lists
        [](Stack& stack) {
          // Pop the stack and convert to C++ list of bool
          c10::List<bool> l = pop(stack).toBoolList();
          // Iterate over the list elements
          for (const auto& elem : l) {
            // If any element is true, push true to stack and return
            if (elem) {
              push(stack, true);
              return;
            }
          }
          // If no elements are true, push false to stack
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::all.int(int[] self) -> bool"),
        // Lambda function for 'all' operation on integer lists
        [](Stack& stack) {
          // Pop the stack and convert to C++ list of int64_t
          c10::List<int64_t> l = pop(stack).toIntList();
          // Iterate over the list elements
          for (const auto& elem : l) {
            // If any element is falsy (zero), push false to stack and return
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          // If all elements are truthy, push true to stack
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::all.float(float[] self) -> bool"),
        // Lambda function for 'all' operation on float lists
        [](Stack& stack) {
          // Pop the stack and convert to C++ list of double
          c10::List<double> l = pop(stack).toDoubleList();
          // Iterate over the list elements
          for (const auto& elem : l) {
            // If any element is falsy (zero), push false to stack and return
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          // If all elements are truthy, push true to stack
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::all.bool(bool[] self) -> bool"),
        // Lambda function for 'all' operation on boolean lists
        [](Stack& stack) {
          // Pop the stack and convert to C++ list of bool
          c10::List<bool> l = pop(stack).toBoolList();
          // Iterate over the list elements
          for (const auto& elem : l) {
            // If any element is falsy (false), push false to stack and return
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          // If all elements are truthy, push true to stack
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 定义 ATen 操作符 divmod.int，接受两个整数参数，返回整数商和余数
        TORCH_SELECTIVE_SCHEMA("aten::divmod.int(int x, int y) -> (int, int)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t a, b;
          // 初始化 divresult 结构体
          lldiv_t divresult = {};
          // 从堆栈中弹出两个整数参数
          pop(stack, a, b);
          // 如果除数为零，抛出异常
          if (b == 0) {
            throw std::runtime_error(
                "ZeroDivisionError: integer division or modulo by zero");
          }
          // 执行整数除法
          divresult = lldiv(a, b);
          // 处理余数，确保符号与除数相同
          if (divresult.rem && (a < 0) != (b < 0)) {
            divresult.quot -= 1;
            divresult.rem += b;
          }
          // 将结果压入堆栈，分别为商和余数
          push(
              stack,
              static_cast<int64_t>(divresult.quot),
              static_cast<int64_t>(divresult.rem));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 定义 ATen 操作符 divmod.float，接受两个浮点数参数，返回浮点数商和余数
        TORCH_SELECTIVE_SCHEMA(
            "aten::divmod.float(float x, float y) -> (float, float)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double a, b;
          // 从堆栈中弹出两个浮点数参数
          pop(stack, a, b);
          // 如果除数为零，抛出异常
          if (b == 0) {
            throw std::runtime_error("ZeroDivisionError: float divmod()");
          }
          // 使用 fmod 计算浮点数的余数
          double rem = fmod(a, b);
          // 如果余数不为零且符号不同，则调整余数
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          if (rem && (a < 0) != (b < 0)) {
            rem += b;
          }
          // 将商和余数压入堆栈
          push(stack, (a - rem) / b, rem);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // 定义 prim 操作符 id，接受任意类型参数，返回其指针值的整数表示
        TORCH_SELECTIVE_SCHEMA("prim::id(AnyClassType? x) -> int"),
        [](Stack& stack) {
          IValue a;
          // 从堆栈中弹出一个值
          pop(stack, a);
          // 如果值为 None，则压入整数 0
          if (a.isNone()) {
            push(stack, 0);
          } else {
            // 否则，将其内部指针转换为整数并压入堆栈
            push(stack, reinterpret_cast<int64_t>(a.internalToPointer()));
          }
        },
        aliasAnalysisFromSchema()),
    // This operator is generated inside the compiler for indexing into
    // ModuleList without a statically determinable key. Accordingly,
    // self must be a ModuleType and the output must be an InterfaceType.
    OperatorGeneratorArgs(
        // 定义 prim 操作符 ModuleContainerIndex.list，用于从 ModuleList 中索引
        TORCH_SELECTIVE_SCHEMA(
            "prim::ModuleContainerIndex.list(Any self, int ind) -> Any"),
        [](Stack& stack) {
          // 从堆栈中弹出索引值和 ModuleList
          IValue ind = pop(stack);
          IValue module_dict = pop(stack);
          // 将索引转换为字符串
          std::stringstream ss;
          ss << ind.toInt();
          // 根据索引获取 ModuleList 中对应的对象，并压入堆栈
          push(
              stack, torch::jit::Object(module_dict.toObject()).attr(ss.str()));
        },
        aliasAnalysisFromSchema()),
#define DEFINE_DIVMOD_MIXED_OP(type_a, type_b)                               \
  OperatorGeneratorArgs(                                                     \
      TORCH_SELECTIVE_SCHEMA("aten::divmod." #type_a "_" #type_b "(" #type_a \
                             " x," #type_b " y) -> (float, float)"),         \
      [](Stack& stack) {                                                     \
        type_a a;                                                            \
        type_b b;                                                            \
        pop(stack, a, b);                                                    \
        if (b == 0) {                                                        \
          throw std::runtime_error("ZeroDivisionError: float divmod()");     \
        }                                                                    \
        double quot = floor(a / b);                                          \
        double rem = a - (quot * b);                                         \
        push(stack, quot, rem);                                              \
      },                                                                     \
      aliasAnalysisFromSchema())

    // 定义混合类型的除法和取模操作的生成器宏
    // 生成的操作包括两种类型组合的除法和取模运算

    DEFINE_DIVMOD_MIXED_OP(int, float),
    // 生成 int 和 float 类型混合的除法和取模操作

    DEFINE_DIVMOD_MIXED_OP(float, int),
    // 生成 float 和 int 类型混合的除法和取模操作

#undef DEFINE_DIVMOD_MIXED_OP
    // 取消定义混合类型的除法和取模操作的生成器宏

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::hash.generic(t value) -> int"),
        hashValue,
        aliasAnalysisFromSchema()),

#define DEFINE_COMPLEX_OP(type_a, type_b, actual_type_a, actual_type_b)       \
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA("aten::Complex." #type_a "_" #type_b "(" #type_a \
                             " x," #type_b " y) -> complex"),                 \
      [](Stack& stack) {                                                      \
        actual_type_a a;                                                      \
        actual_type_b b;                                                      \
        pop(stack, a, b);                                                     \
        auto comp = c10::complex<double>(a, b);                               \
        push(stack, comp);                                                    \
      },                                                                      \
      aliasAnalysisFromSchema())

#define DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(                                    \


注释：
    // 定义宏，生成复杂操作符的实现函数
    DEFINE_COMPLEX_OP(int, bool, int, bool),
    DEFINE_COMPLEX_OP(bool, int, bool, int),
    DEFINE_COMPLEX_OP(float, bool, double, bool),
    DEFINE_COMPLEX_OP(bool, float, bool, double),
    DEFINE_COMPLEX_OP(float, int, double, int),
    DEFINE_COMPLEX_OP(int, float, int, double),
    DEFINE_COMPLEX_OP(int, int, int, int),
    DEFINE_COMPLEX_OP(bool, bool, bool, bool),
    DEFINE_COMPLEX_OP(float, float, double, double),
    // 定义带有张量参数的复杂操作符的实现函数
    DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(Tensor, float, at::Tensor, double),
    DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(Tensor, int, at::Tensor, int),
    DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(Tensor, bool, at::Tensor, bool),
};

// 注册操作符，使用 createOperators 函数生成操作符，并注册到 RegisterOperators 对象中
RegisterOperators reg2(createOperators(opGenArgs2));

} // namespace torch::jit
} // namespace torch::jit
```