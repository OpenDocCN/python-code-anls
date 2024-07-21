# `.\pytorch\torch\csrc\jit\runtime\vararg_functions.cpp`

```py
// 包含 TorchScript 运行时的变长函数头文件
#include <torch/csrc/jit/runtime/vararg_functions.h>

// 包含 ATen 库的函数和张量定义
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/core/class_type.h>
#include <c10/util/irange.h>

// 定义 TorchScript 命名空间
namespace torch::jit {

// 匿名命名空间，用于文件内部的静态变量和函数
namespace {

// 默认的精度值
static constexpr int defaultPrecision = 6;

// 将 IValue 类型格式化为指定格式的参数并添加到 stringstream 中
void addFormattedArg(
    char key,                     // 格式化的字符标识符，如 'd', 'e', 'f', 'c', 's'
    const IValue& ival,           // 要格式化的 IValue 对象
    std::stringstream& ss,        // 输出结果的 stringstream
    int precision = defaultPrecision) {  // 格式化数值时的精度，默认为 defaultPrecision
  std::stringstream tmp;         // 临时 stringstream 用于处理不同的格式化要求

  switch (key) {
    case 'd':
    case 'i':
      // 检查是否是标量类型，如果是整数则输出整数值，否则转换为整数后输出
      TORCH_CHECK(
          ival.isScalar(),
          "%",
          key,
          " requires a number for formatting, but got ",
          ival.tagKind());
      if (ival.isInt()) {
        ss << ival.toInt();
      } else {
        ss << static_cast<int>(ival.toDouble());
      }
      break;
    case 'e':
    case 'E':
      // 对科学计数法的处理，根据 key 是 'e' 还是 'E' 选择小写或大写格式
      TORCH_CHECK(
          ival.isScalar(),
          "%",
          key,
          " requires a number for formatting, but got ",
          ival.tagKind());
      tmp << std::setprecision(precision) << std::scientific;
      if (key == 'E') {
        tmp << std::uppercase;
      }
      if (ival.isInt()) {
        tmp << static_cast<float>(ival.toInt());
      } else {
        tmp << static_cast<float>(ival.toDouble());
      }
      ss << tmp.str();  // 将格式化后的字符串添加到输出 stringstream 中
      break;
    case 'f':
    case 'F':
      // 对浮点数的固定小数位数格式化
      TORCH_CHECK(
          ival.isScalar(),
          "%",
          key,
          " requires a number for formatting, but got ",
          ival.tagKind());
      tmp << std::setprecision(precision) << std::fixed;
      if (ival.isInt()) {
        tmp << static_cast<float>(ival.toInt());
      } else {
        tmp << static_cast<float>(ival.toDouble());
      }
      ss << tmp.str();  // 将格式化后的字符串添加到输出 stringstream 中
      break;
    case 'c':
      // 对字符或整数的格式化，要求输入是整数或长度为 1 的字符串
      TORCH_CHECK(
          ival.isInt() || (ival.isString() && ival.toStringRef().length() == 1),
          "%",
          key,
          " requires an int or char for formatting, but got ",
          ival.tagKind());
      if (ival.isInt()) {
        ss << static_cast<char>(ival.toInt());
      } else {
        ss << ival.toStringRef();
      }
      break;
    case 's':
      // 直接输出字符串或者 IValue 对象的内容
      if (ival.isString()) {
        ss << ival.toStringRef();
      } else {
        ss << ival;
      }
      break;
    default:
      // 不支持的格式化字符会抛出错误
      TORCH_CHECK(
          false,
          "The specifier %",
          key,
          " is not supported in TorchScript format strings");
  }
}

} // namespace

// 从堆栈中弹出一个元组，将其元素解包并加入到堆栈中
void tupleUnpack(Stack& stack) {
  auto tuple = pop(stack).toTuple();  // 从堆栈中弹出一个元组对象
  stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());  // 将元组中的元素插入到堆栈中
}
// 格式化函数，根据堆栈中的输入参数进行格式化操作
void format(Stack& stack, size_t num_inputs) {
  // 检查输入参数数量是否有效
  TORCH_CHECK(
      num_inputs > 0 && num_inputs <= stack.size(),
      "Invalid number of inputs for format string: ",
      num_inputs);

  // 获取格式化字符串，这里peek函数用于获取堆栈顶部指定数量的元素，并将其转换为字符串引用
  auto format = peek(stack, 0, num_inputs).toStringRef();

  // 准备格式化后的字符串
  std::stringstream ss;
  for (size_t begin = 0, used_args = 0; true; ++used_args) {
    // 查找格式字符串中的占位符 "{}"
    size_t loc = format.find("{}", begin);
    if (loc == std::string::npos) {
      // 如果没有找到更多的占位符，则将剩余部分直接添加到结果中
      ss << format.substr(begin);
      break;
    }
    // 添加占位符前的部分到结果中
    ss << format.substr(begin, loc - begin);
    if (used_args >= args.size()) {
      // 如果参数数量不足以填充所有占位符，则报错
      AT_ERROR("Too few arguments for format string: ", format);
    }
    // 添加参数值到结果中
    ss << args[used_args];
    begin = loc + 2;  // 移动到下一个占位符的位置
  }

  // 从堆栈中移除输入参数
  drop(stack, num_inputs);
  // 将格式化后的字符串推入堆栈作为结果
  push(stack, ss.str());
}

// einsum函数，执行张量操作，根据给定的方程字符串和操作数进行计算
void einsum(Stack& stack, size_t num_inputs) {
  // 检查参数数量是否符合要求
  TORCH_CHECK(
      num_inputs >= 2,
      "einsum(): must specify the equation string and at least one operand, ",
      "or at least one operand and its subscripts list");

  // 获取函数的所有参数
  const auto args = last(stack, num_inputs);

  // 准备用于构建方程字符串的stringstream
  std::stringstream ss;

  // 定义一个lambda函数，用于解析子脚本列表并添加到结果stringstream中
  auto parse_sublist = [&ss](const c10::List<int64_t>& l, size_t arg_num) {
    for (const auto i : c10::irange(l.size())) {
      // 检查每个子脚本的有效性，并将其转换为对应的字母
      TORCH_CHECK(
          l[i] >= 0 && l[i] < 52,
          "einsum(): expected subscript ",
          i,
          " in argument ",
          arg_num,
          " to be within the range [0, 52), but got ",
          l[i]);
      if (l[i] < 26) {
        ss << static_cast<char>(l[i] + 'A');  // 大写字母表示0-25
      } else {
        ss << static_cast<char>(l[i] - 26 + 'a');  // 小写字母表示26-51
      }
    }
  };

  // 解析输入操作数的子脚本列表
  for (auto i = decltype(num_inputs){1}; i < num_inputs; i += 2) {
    TORCH_CHECK(
        args[i].isIntList(),
        "einsum(): expected List[int] in argument ",
        i,
        ", but got ",
        args[i].type()->repr_str());
    parse_sublist(args[i].toIntList(), i);
    if (i + 2 < num_inputs) {
      ss << ',';  // 添加逗号分隔符
    }
  }

  // 解析可选的输出子脚本（如果参数数量为奇数）
  if (num_inputs % 2 == 1) {
    TORCH_CHECK(
        args.back().isIntList(),
        "einsum(): expected List[int] in argument ",
        num_inputs - 1,
        ", but got ",
        args.back().type()->repr_str());
    ss << "->";  // 添加输出子脚本标识符
    // 调用parse_sublist函数，解析args的倒数第二个元素为整数列表，并处理子列表
    parse_sublist(args.back().toIntList(), num_inputs - 1);
  }

  // 将stringstream ss的内容转换为equation字符串
  const auto equation = ss.str();
  // 创建一个空的Tensor向量operands，用于存储操作数
  std::vector<at::Tensor> operands;

  // 解析输入操作数
  // 如果num_inputs是奇数，则end为num_inputs-1，否则为num_inputs
  const auto end = num_inputs % 2 == 1 ? num_inputs - 1 : num_inputs;
  for (auto i = decltype(num_inputs){0}; i < end; i += 2) {
    // 检查args[i]是否为Tensor类型，否则抛出错误信息
    TORCH_CHECK(
        args[i].isTensor(),
        "einsum(): expected Tensor in argument ",
        i,
        ", but got ",
        args[i].type()->repr_str());
    // 将args[i]转换为Tensor，并加入operands向量
    operands.emplace_back(args[i].toTensor());
  }

  // 从堆栈中移除num_inputs个元素
  drop(stack, num_inputs);
  // 将equation和operands传递给at::einsum函数，并将结果压入堆栈
  push(stack, at::einsum(equation, operands));
}

void percentFormat(Stack& stack, size_t num_inputs) {
  // 获取格式化字符串
  auto format_str = peek(stack, 0, num_inputs).toStringRef();
  // 获取格式化参数
  auto args = last(stack, num_inputs - 1)[0];
  // 假设参数个数为1
  size_t args_size = 1; // assumed size
  // 如果参数是元组，则获取其元素个数
  if (args.isTuple()) {
    args_size = args.toTupleRef().elements().size();
  }
  // 创建字符串流
  std::stringstream ss;
  // 已使用的参数个数
  size_t used_args = 0;
  // 起始查找位置
  size_t begin = 0;
  // 循环处理格式化字符串中的每个百分号
  while (true) {
    // 查找下一个百分号的位置
    size_t percent_idx = format_str.find('%', begin);
    // 如果找不到百分号，则将剩余部分添加到字符串流中并结束循环
    if (percent_idx == std::string::npos) {
      ss << format_str.substr(begin);
      break;
    }
    // 百分号后面的索引
    size_t format_idx = percent_idx + 1;
    // 检查是否是百分号转义
    TORCH_CHECK(
        percent_idx < format_str.length() - 1, "Incomplete format specifier");
    // 将百分号前面的部分添加到字符串流中
    ss << format_str.substr(begin, percent_idx - begin);
    // 如果百分号后面是另一个百分号，则直接添加 '%' 到字符串流，并跳过处理
    if (format_str.at(format_idx) == '%') {
      ss << '%';
      begin = percent_idx + 2; // 跳过 '%' 和格式说明符
      continue;
    }
    // 检查是否还有足够的参数来处理格式字符串
    TORCH_CHECK(used_args < args_size, "Too few arguments for format string");
    // 获取格式化字符串的键
    char key = format_str.at(format_idx);
    IValue arg;
    // 如果参数是元组，则获取对应位置的参数值
    if (args.isTuple()) {
      arg = args.toTupleRef().elements()[used_args];
    } else {
      arg = args;
    }
    // 添加格式化后的参数到字符串流中
    addFormattedArg(key, arg, ss);
    begin = percent_idx + 2;
    ++used_args;
  }
  // 检查是否所有参数都已使用
  TORCH_CHECK(used_args == args_size, "Too many arguments for format string");
  // 丢弃栈中的输入参数
  drop(stack, num_inputs);
  // 将格式化后的字符串推入栈中
  push(stack, ss.str());
}

void listUnpack(Stack& stack, size_t num_outputs) {
  // 从栈中弹出列表
  auto list = pop(stack).toList();
  // 检查列表长度是否与期望的输出个数相符
  TORCH_CHECK(
      list.size() == num_outputs,
      "Expected ",
      num_outputs,
      " elements in a list but found ",
      list.size());
  // 将列表元素推入栈中
  stack.insert(stack.end(), list.begin(), list.end());
}

void tupleConstruct(Stack& stack, size_t num_inputs) {
  // 检查输入参数个数是否合法
  if (num_inputs > stack.size()) {
    TORCH_CHECK(false, "Invalid number of inputs: ", num_inputs);
  }
  // 根据输入参数个数进行不同的元组构建操作
  switch (num_inputs) {
    case 0:
      // 创建空元组并推入栈中
      stack.emplace_back(c10::ivalue::Tuple::create());
      break;
    case 1:
      // 从栈中移动一个元素创建元组，并替换栈顶元素
      stack.back() = c10::ivalue::Tuple::create(std::move(stack.back()));
      break;
    case 2: {
      // 从栈中移动两个元素创建元组，并替换栈顶元素
      auto tuple = c10::ivalue::Tuple::create(
          std::move(stack[stack.size() - 2]),
          std::move(stack[stack.size() - 1]));
      stack.pop_back();
      stack.back() = std::move(tuple);
      break;
    }
    case 3: {
      // 从栈中移动三个元素创建元组，并替换栈顶元素
      auto tuple = c10::ivalue::Tuple::create(
          std::move(stack[stack.size() - 3]),
          std::move(stack[stack.size() - 2]),
          std::move(stack[stack.size() - 1]));
      stack.pop_back();
      stack.pop_back();
      stack.back() = std::move(tuple);
      break;
    }
    default: {
      // 从栈中移动 num_inputs 个元素创建元组，并替换栈顶元素
      std::vector<IValue> elems{
          std::make_move_iterator(stack.end() - num_inputs),
          std::make_move_iterator(stack.end())};
      drop(stack, num_inputs - 1);
      stack.back() = c10::ivalue::Tuple::create(std::move(elems));
      break;
    }
  }
}

void namedTupleConstruct(
    Stack& stack,
    c10::TypePtr tuple_type,
    ```
    size_t num_inputs) {
  # 创建一个包含最后 num_inputs 个元素的 std::vector<IValue> 对象 elems
  std::vector<IValue> elems{
      std::make_move_iterator(stack.end() - num_inputs),
      std::make_move_iterator(stack.end())};
  # 从堆栈中移除最后 num_inputs 个元素
  drop(stack, num_inputs);
  # 将 elems 中的元素作为命名元组的元素，创建一个新的命名元组，并推入堆栈
  push(
      stack,
      c10::ivalue::Tuple::createNamed(std::move(elems), std::move(tuple_type)));
}

void listConstruct(
    Stack& stack,
    const c10::Type& list_type,
    size_t num_inputs) {
  // 定义一个 lambda 函数 makeList，用于构建一个 c10::List<IValue> 对象
  auto makeList =
      [](Stack& stack, const c10::Type& list_type, size_t num_inputs) {
        // 创建一个空的值列表 vals，其元素类型由 list_type 的第一个子类型确定
        c10::List<IValue> vals(list_type.containedType(0));
        vals.reserve(num_inputs);  // 预留空间以容纳 num_inputs 个元素
        // 将栈中从索引 stack.size() - num_inputs 开始到 stack.size() 结束的元素移动到 vals 中
        for (size_t i = stack.size() - num_inputs; i < stack.size(); ++i) {
          vals.push_back(std::move(stack[i]));
        }
        drop(stack, num_inputs);  // 从栈中移除 num_inputs 个元素
        return vals;  // 返回构建好的值列表
      };
  stack.emplace_back(makeList(stack, list_type, num_inputs));  // 将构建好的值列表推入栈中
}

void dictConstruct(
    Stack& stack,
    const c10::Type& dict_type,
    size_t num_inputs) {
  // 创建一个空的泛型字典 vals，其键类型为 dict_type 的第一个子类型，值类型为第二个子类型
  auto vals = c10::impl::GenericDict(
      dict_type.containedType(0), dict_type.containedType(1));
  vals.reserve(num_inputs / 2);  // 预留空间以容纳 num_inputs / 2 对键值对
  // 从栈底开始遍历栈中的输入，以保持输入的顺序
  auto inputs = last(stack, num_inputs);
  for (size_t i = 0; i < num_inputs; i += 2) {
    auto key = inputs[i];  // 键为 inputs 中的第 i 个元素
    auto val = inputs[i + 1];  // 值为 inputs 中的第 i+1 个元素
    vals.insert_or_assign(std::move(key), std::move(val));  // 插入或更新键值对到字典中
  }
  drop(stack, num_inputs);  // 从栈中移除 num_inputs 个元素
  push(stack, std::move(vals));  // 将构建好的泛型字典推入栈中
}

void createObject(
    Stack& stack,
    const at::ClassTypePtr& type,
    bool as_weak_ref) {
  if (as_weak_ref) {
    // 如果指定使用弱引用，创建一个弱类型指针 weak，并用它创建一个用户对象 userObj
    c10::WeakTypePtr weak(type->compilation_unit(), type);
    auto userObj = c10::ivalue::Object::create(
        c10::WeakOrStrongTypePtr(weak), type->numAttributes());
    push(stack, std::move(userObj));  // 将创建的对象推入栈中
  } else {
    // 否则，使用强类型指针创建用户对象 userObj
    auto userObj = c10::ivalue::Object::create(
        c10::StrongTypePtr(type->compilation_unit(), type),
        type->numAttributes());
    push(stack, std::move(userObj));  // 将创建的对象推入栈中
  }
}

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types) {
  at::TypePtr ty = pop(stack).type();  // 弹出栈顶元素的类型
  // 检查该类型是否是 types 中任一候选类型的子类型
  for (const at::TypePtr& candidate : types) {
    if (ty->isSubtypeOf(*candidate)) {
      push(stack, true);  // 如果是子类型，推入 true 到栈中并返回
      return;
    }
  }
  push(stack, false);  // 如果不是任何候选类型的子类型，推入 false 到栈中并返回
}

void tupleSlice(Stack& stack, size_t begin, size_t end) {
  auto tuple = pop(stack).toTuple();  // 弹出栈顶元素并转换为元组类型
  // 对元组的元素进行切片操作，从 begin 到 end，不包括 end
  push(
      stack,
      c10::ivalue::Tuple::create(
          tuple->elements().asArrayRef().slice(begin, end - begin)));
}

void dequantize(Stack& stack) {
  auto iv = pop(stack);  // 弹出栈顶元素
  if (iv.isTuple()) {
    auto tuple = iv.toTuple();  // 如果是元组类型，进行以下处理
    const auto& elems = tuple->elements();  // 获取元组的元素列表
    std::vector<IValue> output_elems;
    output_elems.reserve(elems.size());  // 预留空间以容纳与元组相同数量的元素
    // 遍历元组的每个元素，如果是张量类型则去量化，否则保持不变
    for (const auto& elem : elems) {
      if (elem.isTensor()) {
        output_elems.emplace_back(at::dequantize(elem.toTensor()));
      } else {
        output_elems.emplace_back(elem);
      }
    }
    // 将处理后的元素重新组成一个新的元组推入栈中
    push(stack, c10::ivalue::Tuple::create(std::move(output_elems)));
  } else if (iv.isTensorList()) {
    auto elems = iv.toTensorList();  // 如果是张量列表类型，进行以下处理
    auto output_list = c10::impl::GenericList(elems.elementType());  // 创建一个泛型列表


**续注释：**
    // 遍历 elems 容器中的每个元素，并进行处理
    for (auto&& elem : elems) {
      // 对当前元素进行反量化操作，并将结果添加到 output_list 尾部
      output_list.emplace_back(at::dequantize(elem));
    }
    // 将 output_list 推入栈中，使用 std::move 转移所有权
    push(stack, std::move(output_list));
  } else {
    // 若输入类型不支持反量化操作，抛出异常
    TORCH_CHECK(
        false,
        "Unsupported type in dequantize, only List[Tensor] and \
 Tuple[Tensor or other types] are supported, got type:",
        toString(iv.type()));
  }
}

} // namespace torch::jit
```