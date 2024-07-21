# `.\pytorch\torch\csrc\utils\invalid_arguments.cpp`

```py
// Torch命名空间开始

namespace torch {

// 匿名命名空间，用于限定内部使用

namespace {

// 返回Python对象的类型名字符串
std::string py_typename(PyObject* object) {
  return Py_TYPE(object)->tp_name;
}

// 表示参数类型的基类，定义了虚函数接口
struct Type {
  Type() = default;
  Type(const Type&) = default;
  Type& operator=(const Type&) = default;
  Type(Type&&) noexcept = default;
  Type& operator=(Type&&) noexcept = default;
  
  // 纯虚函数，需要在派生类中实现，用于检查对象是否匹配
  virtual bool is_matching(PyObject* object) = 0;
  
  // 虚析构函数，用于派生类的多态销毁
  virtual ~Type() = default;
};

// 简单类型，继承自Type，用于检查对象是否与给定类型名匹配
struct SimpleType : public Type {
  // 构造函数，接受一个类型名字符串的引用
  SimpleType(std::string& name) : name(name){};

  // 实现基类的虚函数，检查对象是否与类型名匹配
  bool is_matching(PyObject* object) override {
    return py_typename(object) == name;
  }

  std::string name; // 存储类型名
};

// 多类型，继承自Type，用于检查对象是否属于多个指定类型之一
struct MultiType : public Type {
  // 构造函数，接受一个初始化列表，包含多个类型名字符串
  MultiType(std::initializer_list<std::string> accepted_types)
      : types(accepted_types){};

  // 实现基类的虚函数，检查对象是否属于其中之一的类型
  bool is_matching(PyObject* object) override {
    auto it = std::find(types.begin(), types.end(), py_typename(object));
    return it != types.end();
  }

  std::vector<std::string> types; // 存储多个类型名
};

// 可空类型，继承自Type，用于检查对象是否为None或与特定类型匹配
struct NullableType : public Type {
  // 构造函数，接受一个唯一指针指向Type对象
  NullableType(std::unique_ptr<Type> type) : type(std::move(type)){};

  // 实现基类的虚函数，检查对象是否为None或与指定类型匹配
  bool is_matching(PyObject* object) override {
    return object == Py_None || type->is_matching(object);
  }

  std::unique_ptr<Type> type; // 存储指向Type对象的唯一指针
};

// 元组类型，继承自Type，用于检查对象是否为特定类型的元组
struct TupleType : public Type {
  // 构造函数，接受一个唯一指针数组，每个指针指向一个Type对象
  TupleType(std::vector<std::unique_ptr<Type>> types)
      : types(std::move(types)){};

  // 实现基类的虚函数，检查对象是否为指定类型的元组
  bool is_matching(PyObject* object) override {
    if (!PyTuple_Check(object)) // 检查对象是否为元组
      return false;
    auto num_elements = PyTuple_GET_SIZE(object); // 获取元组中元素数量
    if (num_elements != (long)types.size()) // 检查元素数量是否匹配
      return false;
    for (const auto i : c10::irange(num_elements)) { // 遍历元组中的每个元素
      if (!types[i]->is_matching(PyTuple_GET_ITEM(object, i))) // 检查每个元素是否与对应类型匹配
        return false;
    }
    return true;
  }

  std::vector<std::unique_ptr<Type>> types; // 存储指向Type对象的唯一指针数组
};

// 序列类型，继承自Type，用于检查对象是否为特定类型的序列
struct SequenceType : public Type {
  // 构造函数，接受一个唯一指针指向Type对象
  SequenceType(std::unique_ptr<Type> type) : type(std::move(type)){};

  // 实现基类的虚函数，检查对象是否为指定类型的序列
  bool is_matching(PyObject* object) override {
    if (!PySequence_Check(object)) // 检查对象是否为序列
      return false;
    auto num_elements = PySequence_Length(object); // 获取序列中元素数量
    for (const auto i : c10::irange(num_elements)) { // 遍历序列中的每个元素
      if (!type->is_matching(
              py::reinterpret_steal<py::object>(PySequence_GetItem(object, i))
                  .ptr())) // 检查每个元素是否与指定类型匹配
        return false;
    }
    return true;
  }

  std::unique_ptr<Type> type; // 存储指向Type对象的唯一指针
};

// 表示函数参数的结构体，包含参数名称和参数类型的唯一指针
struct Argument {
  // 构造函数，接受参数名称和指向Type对象的唯一指针
  Argument(std::string name, std::unique_ptr<Type> type)
      : name(std::move(name)), type(std::move(type)){};

  std::string name; // 存储参数名称
  std::unique_ptr<Type> type; // 存储指向Type对象的唯一指针
};
struct Option {
  // 构造函数：接受参数列表、是否可变参数、是否具有输出参数，并初始化成员变量
  Option(std::vector<Argument> arguments, bool is_variadic, bool has_out)
      : arguments(std::move(arguments)),
        is_variadic(is_variadic),
        has_out(has_out){};
  // 构造函数：接受是否可变参数、是否具有输出参数，并初始化成员变量
  Option(bool is_variadic, bool has_out)
      : arguments(), is_variadic(is_variadic), has_out(has_out){};
  // 删除复制构造函数
  Option(const Option&) = delete;
  // 移动构造函数：从另一个 Option 对象获取数据
  Option(Option&& other) noexcept
      : arguments(std::move(other.arguments)),
        is_variadic(other.is_variadic),
        has_out(other.has_out){};

  // 参数列表
  std::vector<Argument> arguments;
  // 是否可变参数
  bool is_variadic;
  // 是否具有输出参数
  bool has_out;
};

// 将字符串按指定分隔符分割为子字符串并返回子字符串向量
std::vector<std::string> _splitString(
    const std::string& s,
    const std::string& delim) {
  std::vector<std::string> tokens; // 存储分割后的子字符串
  size_t start = 0; // 分割的起始位置
  size_t end = 0; // 分割的结束位置
  while ((end = s.find(delim, start)) != std::string::npos) {
    // 将分割的子字符串加入向量
    tokens.push_back(s.substr(start, end - start));
    start = end + delim.length(); // 更新起始位置
  }
  tokens.push_back(s.substr(start)); // 加入最后一个子字符串
  return tokens; // 返回分割结果的向量
}

// 根据类型名称和是否可为空构建一个类型的唯一指针
std::unique_ptr<Type> _buildType(std::string type_name, bool is_nullable) {
  std::unique_ptr<Type> result; // 结果类型的唯一指针
  if (type_name == "float") {
    // 如果类型名称为 "float"，则创建一个 MultiType 对象包含 {"float", "int", "long"}
    result = std::make_unique<MultiType>(MultiType{"float", "int", "long"});
  } else if (type_name == "int") {
    // 如果类型名称为 "int"，则创建一个 MultiType 对象包含 {"int", "long"}
    result = std::make_unique<MultiType>(MultiType{"int", "long"});
  } else if (type_name.find("tuple[") == 0) {
    auto type_list = type_name.substr(6); // 提取类型列表部分
    type_list.pop_back(); // 去除末尾的 "]"
    std::vector<std::unique_ptr<Type>> types; // 存储子类型的唯一指针向量
    for (auto& type : _splitString(type_list, ",")) // 遍历分割后的类型列表
      types.emplace_back(_buildType(type, false)); // 递归构建子类型并加入向量
    result = std::make_unique<TupleType>(std::move(types)); // 创建 TupleType 对象
  } else if (type_name.find("sequence[") == 0) {
    auto subtype = type_name.substr(9); // 提取序列的子类型
    subtype.pop_back(); // 去除末尾的 "]"
    result = std::make_unique<SequenceType>(_buildType(subtype, false)); // 创建 SequenceType 对象
  } else {
    result = std::make_unique<SimpleType>(type_name); // 创建 SimpleType 对象
  }
  if (is_nullable)
    result = std::make_unique<NullableType>(std::move(result)); // 如果可为空，则包装成 NullableType
  return result; // 返回类型的唯一指针
}

// 解析选项字符串，并返回选项对象和可打印的选项字符串
std::pair<Option, std::string> _parseOption(
    const std::string& _option_str,
    const std::unordered_map<std::string, PyObject*>& kwargs) {
  if (_option_str == "no arguments")
    // 如果选项字符串为 "no arguments"，返回不带参数和输出的 Option 对象
    return std::pair<Option, std::string>(Option(false, false), _option_str);
  bool has_out = false; // 是否具有输出参数的标志
  std::vector<Argument> arguments; // 存储参数的向量
  std::string printable_option = _option_str; // 可打印的选项字符串
  std::string option_str = _option_str.substr(1, _option_str.length() - 2); // 去除首尾的括号

  /// XXX: this is a hack only for the out arg in TensorMethods
  // 处理针对 TensorMethods 中输出参数的特殊处理
  auto out_pos = printable_option.find('#'); // 查找特殊标记 '#'
  if (out_pos != std::string::npos) { // 如果找到标记
    if (kwargs.count("out") > 0) { // 如果 kwargs 中包含 "out"
      std::string kwonly_part = printable_option.substr(out_pos + 1); // 提取特殊标记后的部分
      printable_option.erase(out_pos); // 删除特殊标记及之后内容
      printable_option += "*, "; // 添加指定输出参数的格式
      printable_option += kwonly_part; // 恢复原字符串内容
    } else if (out_pos >= 2) {
      printable_option.erase(out_pos - 2); // 删除特殊标记前两个字符
      printable_option += ")"; // 添加括号以匹配原格式
    } else {
      printable_option.erase(out_pos); // 删除特殊标记
      printable_option += ")"; // 添加括号以匹配原格式
    }
    // 初始化一个布尔变量 `has_out`，赋值为 true
    has_out = true;
  }

  // 遍历 `_splitString` 函数返回的字符串向量 `option_str` 中的每个字符串
  for (auto& arg : _splitString(option_str, ", ")) {
    // 初始化一个布尔变量 `is_nullable`，赋值为 false
    bool is_nullable = false;
    // 初始化一个整数变量 `type_start_idx`，赋值为 0
    auto type_start_idx = 0;
    // 如果当前字符串 `arg` 的第一个字符是 `#`
    if (arg[type_start_idx] == '#') {
      // 将 `type_start_idx` 自增，跳过 `#` 符号
      type_start_idx++;
    }
    // 如果当前字符串 `arg` 的 `type_start_idx` 位置字符是 `[`
    if (arg[type_start_idx] == '[') {
      // 将 `is_nullable` 设置为 true
      is_nullable = true;
      // 将 `type_start_idx` 自增，跳过 `[` 符号
      type_start_idx++;
      // 删除字符串 `arg` 末尾的 " or None]" 字符串
      arg.erase(arg.length() - std::string(" or None]").length());
    }

    // 查找字符串 `arg` 中最后一个空格的位置，赋值给 `type_end_idx`
    auto type_end_idx = arg.find_last_of(' ');
    // 将 `name_start_idx` 设置为 `type_end_idx` 的下一个位置
    auto name_start_idx = type_end_idx + 1;

    // 如果在字符串 `arg` 中找到了 "..." 的位置
    // 调整 `type_end_idx`，减去 "..." 的长度
    auto dots_idx = arg.find("...");
    if (dots_idx != std::string::npos)
      type_end_idx -= 4;

    // 提取 `type_name`，从 `type_start_idx` 开始到 `type_end_idx` 之间的子字符串
    std::string type_name =
        arg.substr(type_start_idx, type_end_idx - type_start_idx);
    // 提取 `name`，从 `name_start_idx` 开始到字符串末尾的子字符串
    std::string name = arg.substr(name_start_idx);

    // 向 `arguments` 向量末尾添加一个新的元素，元素为 `(name, _buildType(type_name, is_nullable))`
    arguments.emplace_back(name, _buildType(type_name, is_nullable));
  }

  // 检查 `option_str` 是否包含 "..."，赋值给 `is_variadic`
  bool is_variadic = option_str.find("...") != std::string::npos;
  // 返回一个 `std::pair` 对象，包含 `Option` 对象和 `printable_option` 的拷贝
  return std::pair<Option, std::string>(
      Option(std::move(arguments), is_variadic, has_out),
      std::move(printable_option));
}

// 检查参数数量是否与选项期望的数量匹配
bool _argcountMatch(
    const Option& option,                              // 选项对象的引用
    const std::vector<PyObject*>& arguments,           // 参数列表的引用
    const std::unordered_map<std::string, PyObject*>& kwargs) {  // 关键字参数字典的引用
  auto num_expected = option.arguments.size();         // 期望的参数数量
  auto num_got = arguments.size() + kwargs.size();     // 实际获得的参数数量
  // 注意：可变参数函数不接受关键字参数，因此没有关键字参数是可以的
  if (option.has_out && kwargs.count("out") == 0)      // 如果选项中有输出参数，并且未传入 "out" 参数
    num_expected--;                                   // 则期望的参数数量减一
  return num_got == num_expected ||                    // 返回实际参数数量是否与期望相等，或者
      (option.is_variadic && num_got > num_expected);  // 是否是可变参数且实际参数数量大于期望数量
}

// 格式化参数描述信息
std::string _formattedArgDesc(
    const Option& option,                              // 选项对象的引用
    const std::vector<PyObject*>& arguments,           // 参数列表的引用
    const std::unordered_map<std::string, PyObject*>& kwargs) {  // 关键字参数字典的引用
  std::string red;                                     // 红色显示控制字符
  std::string reset_red;                               // 重置红色显示控制字符
  std::string green;                                   // 绿色显示控制字符
  std::string reset_green;                             // 重置绿色显示控制字符
  if (isatty(1) && isatty(2)) {                        // 如果标准输出和标准错误是终端
    red = "\33[31;1m";                                 // 设置红色显示
    reset_red = "\33[0m";                              // 设置重置红色显示
    green = "\33[32;1m";                               // 设置绿色显示
    reset_green = "\33[0m";                            // 设置重置绿色显示
  } else {
    red = "!";                                         // 否则使用简单的符号表示红色
    reset_red = "!";                                   
    green = "";                                        // 不使用绿色显示
    reset_green = "";                                  
  }

  auto num_args = arguments.size() + kwargs.size();     // 参数和关键字参数的总数量
  std::string result = "(";                            // 结果字符串，开始
  for (const auto i : c10::irange(num_args)) {         // 遍历参数和关键字参数的数量范围
    bool is_kwarg = i >= arguments.size();             // 是否是关键字参数
    PyObject* arg =
        is_kwarg ? kwargs.at(option.arguments[i].name) : arguments[i];  // 获取参数对象

    bool is_matching = false;                          // 是否匹配的标志
    if (i < option.arguments.size()) {                 // 如果当前索引小于选项的参数数量
      is_matching = option.arguments[i].type->is_matching(arg);  // 检查参数类型是否匹配
    } else if (option.is_variadic) {                   // 否则如果是可变参数
      is_matching = option.arguments.back().type->is_matching(arg);  // 检查最后一个参数类型是否匹配
    }

    if (is_matching)
      result += green;                                 // 如果匹配，添加绿色显示
    else
      result += red;                                   // 否则添加红色显示
    if (is_kwarg)
      result += option.arguments[i].name + "=";        // 如果是关键字参数，添加参数名和等号
    bool is_tuple = PyTuple_Check(arg);                // 检查是否是元组
    if (is_tuple || PyList_Check(arg)) {               // 如果是元组或者列表
      result += py_typename(arg) + " of ";             // 添加对象类型和 " of "
      auto num_elements = PySequence_Length(arg);      // 获取序列长度
      if (is_tuple) {
        result += "(";                                 // 如果是元组，添加左括号
      } else {
        result += "[";                                 // 否则添加左方括号
      }
      for (const auto i : c10::irange(num_elements)) { // 遍历元素
        if (i != 0) {
          result += ", ";                              // 如果不是第一个元素，添加逗号和空格
        }
        result += py_typename(
            py::reinterpret_steal<py::object>(PySequence_GetItem(arg, i))
                .ptr());                               // 添加元素类型名
      }
      if (is_tuple) {
        if (num_elements == 1) {
          result += ",";                               // 如果元素数量为1，添加逗号
        }
        result += ")";                                 // 添加右括号
      } else {
        result += "]";                                 // 添加右方括号
      }
    } else {
      result += py_typename(arg);                      // 否则添加参数对象的类型名
    }
    if (is_matching)
      result += reset_green;                           // 如果匹配，添加重置绿色显示
    else
      result += reset_red;                             // 否则添加重置红色显示
    result += ", ";                                    // 添加逗号和空格
  }
  if (!arguments.empty())
    result.erase(result.length() - 2);                 // 如果参数不为空，删除最后的逗号和空格
  result += ")";                                       // 添加右括号
  return result;                                       // 返回结果字符串
}

// 生成参数描述信息
std::string _argDesc(
    const std::vector<PyObject*>& arguments,           // 参数列表的引用
    const std::unordered_map<std::string, PyObject*>& kwargs) {  // 关键字参数字典的引用
  std::string result = "(";                            // 结果字符串，开始
  for (auto& arg : arguments)
    result += std::string(py_typename(arg)) + ", ";    // 添加参数对象的类型名和逗号
  for (auto& kwarg : kwargs)
    result += kwarg.first + "=" + py_typename(kwarg.second) + ", ";  // 添加关键字参数名、等号和类型名
  if (!arguments.empty())
    result.erase(result.length() - 2);                 // 如果参数不为空，删除最后的逗号和空格
  result += ")";                                       // 添加右括号
  return result;                                       // 返回结果字符串
}
// 匹配关键字参数和选项，返回未匹配的关键字列表
std::vector<std::string> _tryMatchKwargs(
    const Option& option,                                    // 选项对象的引用
    const std::unordered_map<std::string, PyObject*>& kwargs) {  // 关键字参数字典

  std::vector<std::string> unmatched;  // 未匹配的关键字参数列表

  // 计算起始索引，确保在选项参数范围内
  int64_t start_idx = option.arguments.size() - kwargs.size();
  if (option.has_out && kwargs.count("out") == 0)  // 如果选项包含 'out' 参数且关键字中没有 'out'
    start_idx--;

  if (start_idx < 0)  // 如果起始索引小于0，则将其设为0
    start_idx = 0;

  // 遍历关键字参数字典
  for (auto& entry : kwargs) {
    bool found = false;
    // 在选项的参数列表中查找关键字参数名
    for (unsigned int i = start_idx; i < option.arguments.size(); i++) {
      if (option.arguments[i].name == entry.first) {  // 如果找到匹配的参数名
        found = true;
        break;
      }
    }
    if (!found)
      unmatched.push_back(entry.first);  // 将未匹配的关键字参数名添加到列表中
  }

  return unmatched;  // 返回未匹配的关键字参数列表
}

} // 匿名命名空间结束

// 格式化无效参数的错误消息
std::string format_invalid_args(
    PyObject* given_args,                                 // 给定的位置参数元组
    PyObject* given_kwargs,                               // 给定的关键字参数字典
    const std::string& function_name,                     // 函数名
    const std::vector<std::string>& options) {            // 可选的选项列表

  std::vector<PyObject*> args;  // 位置参数列表
  std::unordered_map<std::string, PyObject*> kwargs;  // 关键字参数字典
  std::string error_msg;  // 错误消息字符串
  error_msg.reserve(2000);  // 预留空间

  error_msg += function_name;  // 将函数名添加到错误消息中
  error_msg += " received an invalid combination of arguments - ";  // 添加固定格式错误提示

  // 获取给定位置参数元组的长度
  Py_ssize_t num_args = PyTuple_Size(given_args);
  // 将位置参数元组转换为 PyObject* 列表
  for (const auto i : c10::irange(num_args)) {
    PyObject* arg = PyTuple_GET_ITEM(given_args, i);
    args.push_back(arg);
  }

  bool has_kwargs = given_kwargs && PyDict_Size(given_kwargs) > 0;  // 检查是否有关键字参数
  if (has_kwargs) {
    // 遍历给定的关键字参数字典，将其转换为无序映射
    PyObject *key = nullptr, *value = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(given_kwargs, &pos, &key, &value)) {
      kwargs.emplace(THPUtils_unpackString(key), value);
    }
  }

  if (options.size() == 1) {  // 如果只有一个选项
    // 解析选项，并返回选项对象及其字符串表示
    auto pair = _parseOption(options[0], kwargs);
    auto& option = pair.first;
    auto& option_str = pair.second;
    std::vector<std::string> unmatched_kwargs;  // 未匹配的关键字参数列表
    if (has_kwargs)
      unmatched_kwargs = _tryMatchKwargs(option, kwargs);  // 尝试匹配关键字参数

    if (!unmatched_kwargs.empty()) {
      error_msg += "got unrecognized keyword arguments: ";  // 添加未识别关键字参数的错误提示
      for (auto& kwarg : unmatched_kwargs)
        error_msg += kwarg + ", ";  // 列出未识别的关键字参数
      error_msg.erase(error_msg.length() - 2);  // 移除末尾多余的逗号和空格
    } else {
      error_msg += "got ";  // 添加正常接受到的参数描述
      if (_argcountMatch(option, args, kwargs)) {
        error_msg += _formattedArgDesc(option, args, kwargs);  // 格式化位置和关键字参数描述
      } else {
        error_msg += _argDesc(args, kwargs);  // 获取参数描述
      }
      error_msg += ", but expected ";  // 添加预期参数描述
      error_msg += option_str;  // 添加选项描述
    }
  } else {
    error_msg += "got ";  // 添加错误消息的参数描述
    error_msg += _argDesc(args, kwargs);  // 获取参数描述
    error_msg += ", but expected one of:\n";  // 添加预期参数的列表提示
    // 遍历选项列表，每个选项表示一个字符串
    for (auto& option_str : options) {
      // 调用_parseOption函数解析选项字符串并返回选项对象和可打印的选项字符串
      auto pair = _parseOption(option_str, kwargs);
      // 获取解析后的选项对象的引用
      auto& option = pair.first;
      // 获取解析后的可打印选项字符串的引用
      auto& printable_option_str = pair.second;
      // 添加错误信息前缀 "* "
      error_msg += " * ";
      // 添加可打印的选项字符串到错误信息中
      error_msg += printable_option_str;
      // 添加换行符到错误信息中
      error_msg += "\n";
      // 检查选项和传入参数是否匹配
      if (_argcountMatch(option, args, kwargs)) {
        // 如果存在关键字参数，则尝试匹配关键字参数并返回未匹配的关键字列表
        std::vector<std::string> unmatched_kwargs;
        if (has_kwargs)
          unmatched_kwargs = _tryMatchKwargs(option, kwargs);
        // 如果存在未匹配的关键字参数，则添加相关信息到错误信息中
        if (!unmatched_kwargs.empty()) {
          error_msg +=
              "      didn't match because some of the keywords were incorrect: ";
          // 遍历未匹配的关键字参数列表，将每个参数添加到错误信息中
          for (auto& kwarg : unmatched_kwargs)
            error_msg += kwarg + ", ";
          // 删除最后多余的逗号和空格
          error_msg.erase(error_msg.length() - 2);
          // 添加换行符到错误信息中
          error_msg += "\n";
        } else {
          // 如果没有未匹配的关键字参数，则添加参数类型错误信息到错误信息中
          error_msg +=
              "      didn't match because some of the arguments have invalid types: ";
          // 添加格式化后的参数描述到错误信息中
          error_msg += _formattedArgDesc(option, args, kwargs);
          // 添加换行符到错误信息中
          error_msg += "\n";
        }
      }
    }
  }
  // 返回完整的错误信息字符串
  return error_msg;
}

// 结束 torch 命名空间
} // namespace torch
```