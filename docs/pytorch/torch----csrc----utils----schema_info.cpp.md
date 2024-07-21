# `.\pytorch\torch\csrc\utils\schema_info.cpp`

```
// 定义命名空间 torch::utils
namespace torch::utils {
// 将参数名称和值添加到 SchemaInfo 对象中
void SchemaInfo::addArgumentValue(
    const std::string& name,
    const at::IValue& value) {
  // 查找参数名称在 schema_ 中的索引
  std::optional<int> index = schema_.argumentIndexWithName(name);
  // 断言确保参数名称存在于 schema_ 中
  TORCH_INTERNAL_ASSERT(
      index != c10::nullopt, "Schema has no argument named ", name);
  // 将参数名称和值添加到 value_map_
  value_map_[name] = value;
  // 标记 alias_maps_current_ 为不是最新的状态
  alias_maps_current_ = false;
}

// 将参数值列表添加到 SchemaInfo 对象中
void SchemaInfo::addArgumentValues(
    const std::vector<std::optional<at::IValue>>& value_list) {
  // 断言确保 value_list 的大小不超过 schema_ 的参数数量
  TORCH_INTERNAL_ASSERT(
      value_list.size() <= schema_.arguments().size(),
      "Schema does not have enough arguments for value list");

  // 遍历 value_list
  for (size_t i = 0; i < value_list.size(); i++) {
    if (value_list[i].has_value()) {
      // 将 value_list 中的值与对应参数名称关联，并标记 alias_maps_current_ 为不是最新的状态
      value_map_[schema_.arguments()[i].name()] = *value_list[i];
      alias_maps_current_ = false;
    }
  }
}

// 将参数名称和值的映射添加到 SchemaInfo 对象中
void SchemaInfo::addArgumentValues(
    const std::unordered_map<std::string, at::IValue>& values) {
  // 遍历 values 中的键值对，并调用 addArgumentValue 将它们添加到 SchemaInfo 对象中
  for (const auto& key_pair : values) {
    addArgumentValue(key_pair.first, key_pair.second);
  }
}

// 检查是否存在指定名称的输入参数
bool SchemaInfo::hasInputArgumentNamed(const std::string& name) const {
  // 使用 std::any_of 在 schema_ 的参数列表中查找是否存在名称为 name 的参数
  return std::any_of(
      schema_.arguments().begin(),
      schema_.arguments().end(),
      [&name](const c10::Argument& arg) { return arg.name() == name; });
}

// 检查是否有可变参数存在
bool SchemaInfo::is_mutable() {
  // 遍历 schema_ 的参数列表，如果有可变参数则返回 true
  for (size_t i = 0; i < schema_.arguments().size(); i++) {
    if (is_mutable({c10::SchemaArgType::input, i})) {
      return true;
    }
  }
  // 如果没有可变参数则返回 false
  return false;
}

// 检查指定的 schema 参数是否可变
bool SchemaInfo::is_mutable(const c10::SchemaArgument& argument) {
  // 断言确保 argument 的索引在 schema_ 中是有效的
  TORCH_INTERNAL_ASSERT(
      argument.index < schema_.getCorrectList(argument.type).size(),
      "Invalid index for schema.");
  // 如果 alias_maps_current_ 不是最新的，则返回 true，表示参数可变
  if (!alias_maps_current_) {
    generateAliasMaps();
  }
  // 定义静态常量，存储训练操作的特例对
  static const std::vector<SchemaSpecialCasePair> training_ops =
      getTrainingOps();
  // 根据参数类型选择正确的别名映射
  const auto& correct_map = (argument.type == c10::SchemaArgType::input)
      ? input_alias_map_
      : output_alias_map_;
  // 由于某些情况下 running_mean 或 running_var 可能会别名其他输入参数，
  // 所以训练操作的检查依赖于索引
  // 返回值是一个布尔值，指示是否有任何一个条件成立
  return std::any_of(
      // 遍历正确的别名映射中给定索引的起始到终止位置
      correct_map[argument.index].begin(),
      correct_map[argument.index].end(),
      [this](size_t aliasing_index) {
        // 查找符合当前模式的训练操作
        const auto is_training_op = std::find_if(
            training_ops.begin(),
            training_ops.end(),
            [this](const auto& training_op) {
              return this->schema_ == training_op.first;
            });

        // 检查是否存在特殊情况，这取决于训练操作和参数名的映射
        bool special_case = (is_training_op != training_ops.end()) &&
            is_training_op->second.count(
                this->schema_.arguments()[aliasing_index].name());
        if (special_case) {
          // 检查是否具有 "training" 参数但未映射到值，或映射到真值
          bool has_training = (hasInputArgumentNamed("training") &&
                               !value_map_.count("training")) ||
              (value_map_.count("training") &&
               value_map_.at("training").toBool());
          // 检查是否具有 "train" 参数但未映射到值，或映射到真值
          bool has_train =
              (hasInputArgumentNamed("train") && !value_map_.count("train")) ||
              (value_map_.count("train") && value_map_.at("train").toBool());
          // 检查是否具有 "use_input_stats" 参数但未映射到值，或映射到真值
          bool has_use_input_stats =
              (hasInputArgumentNamed("use_input_stats") &&
               !value_map_.count("use_input_stats")) ||
              (value_map_.count("use_input_stats") &&
               value_map_.at("use_input_stats").toBool());
          // 返回三个条件中的任意一个为真
          return has_training || has_train || has_use_input_stats;
        } else {
          // 对于非特殊情况，检查模式是否可变
          return this->schema_.is_mutable(
              {c10::SchemaArgType::input, aliasing_index});
        }
      });
}

// 检查给定名称的参数是否存在于模式中
bool SchemaInfo::has_argument(c10::string_view name) {
  // 调用模式对象的方法，检查是否存在具有指定名称的参数索引
  return schema_.argumentIndexWithName(name) != c10::nullopt;
}

// 检查给定名称的参数是否可变
bool SchemaInfo::is_mutable(c10::string_view name) {
  // 获取参数名称对应的索引
  std::optional<int> index = schema_.argumentIndexWithName(name);
  // 内部断言，确保索引值存在，否则抛出异常并显示参数名称
  TORCH_INTERNAL_ASSERT(
      index.has_value(), "Schema has no argument named ", name);

  // 调用另一个重载的 is_mutable 方法，传递参数类型和索引，检查是否可变
  return is_mutable({c10::SchemaArgType::input, static_cast<size_t>(*index)});
}

// 检查模式是否为不确定性操作
bool SchemaInfo::is_nondeterministic() const {
  // 预定义的dropout操作的函数模式
  static const c10::FunctionSchema dropout_schema = torch::jit::parseSchema(
      "aten::dropout(Tensor input, float p, bool train) -> Tensor");
  // 如果当前模式等于dropout操作模式，并且value_map_中存在"train"参数且其值为false，则返回false
  if (dropout_schema == schema_ && value_map_.count("train") &&
      !value_map_.at("train").toBool()) {
    return false;
  }

  // 如果定义了C10_MOBILE，则获取所有不确定性操作的函数模式，并检查当前模式是否其中之一
#if defined C10_MOBILE
  static const std::vector<c10::FunctionSchema> nondeterministic_ops =
      getNonDeterministicOps();
  return std::any_of(
      nondeterministic_ops.begin(),
      nondeterministic_ops.end(),
      [this](const c10 ::FunctionSchema& nondeterministic_op) {
        return nondeterministic_op == this->schema_;
      });
#else
  // 否则，使用Dispatcher查找当前操作，并检查其是否标记为nondeterministic_seeded
  const auto& op = c10::Dispatcher::singleton().findOp(
      c10::OperatorName(schema_.name(), schema_.overload_name()));
  return op && op->hasTag(at::Tag::nondeterministic_seeded);
#endif
}

// 检查两个参数是否可能别名
bool SchemaInfo::may_alias(
    const c10::SchemaArgument& lhs,
    const c10::SchemaArgument& rhs) {
  // 基本检查，调用模式对象的may_alias方法检查两个参数是否可能别名
  bool basic_check = schema_.may_alias(lhs, rhs);
  if (basic_check) {
    return true;
  }
  // 获取左右参数的类型别名集合
  std::optional<c10::AliasTypeSet> lhsAliasTypeSet =
      schema_.mapTypeToAliasTypeSet(
          schema_.getCorrectList(lhs.type)[lhs.index].type());
  std::optional<c10::AliasTypeSet> rhsAliasTypeSet =
      schema_.mapTypeToAliasTypeSet(
          schema_.getCorrectList(rhs.type)[rhs.index].type());
  // 检查左右参数的类型集合是否可能别名
  bool types_can_alias =
      schema_.canAliasTypeSetsAlias(lhsAliasTypeSet, rhsAliasTypeSet);
  if (!types_can_alias) {
    return false;
  }

  // 如果别名映射不是最新的，则重新生成
  if (!alias_maps_current_) {
    generateAliasMaps();
  }
  // 检查通配符集合中是否存在左右参数
  bool wildcard_alias_check =
      wildcardSet().count(lhs) && wildcardSet().count(rhs);
  if (wildcard_alias_check) {
    return true;
  }

  // 根据参数类型进行进一步的别名检查
  if (lhs.type == c10::SchemaArgType::input &&
      rhs.type == c10::SchemaArgType::input) {
    return input_alias_map_[lhs.index].count(rhs.index);
  } else if (
      lhs.type == c10::SchemaArgType::output &&
      rhs.type == c10::SchemaArgType::output) {
    // 如果左参数和右参数都是输出类型，则检查输出别名映射
    for (size_t lhs_alias_input : output_alias_map_[lhs.index]) {
      if (output_alias_map_[rhs.index].count(lhs_alias_input)) {
        return true;
      }
    }
    return false;
  } else if (lhs.type == c10::SchemaArgType::output) {
    // 如果只有左参数是输出类型，则检查左输出别名映射
    return output_alias_map_[lhs.index].count(rhs.index);
  } else {
    // 否则，检查右输出别名映射
    return output_alias_map_[rhs.index].count(lhs.index);
  }
}

// 检查两个参数是否可能包含别名，支持双向检查
bool SchemaInfo::may_contain_alias(
    const c10::SchemaArgument& lhs,
    const c10::SchemaArgument& rhs,
    bool bidirectional) {
  // 基本检查，调用模式对象的may_contain_alias方法或may_alias方法检查是否可能包含别名
  bool basic_check = schema_.may_contain_alias(lhs, rhs) || may_alias(lhs, rhs);
  if (basic_check) {
    // 如果基本检查为真，则直接返回真
    return true;
  }

  // 如果需要双向检查，则返回假
  if (!bidirectional) {
    return false;
  }
    # 返回 true，结束函数执行
    return true;
  }
  # 如果当前别名映射为空，则生成别名映射
  if (!alias_maps_current_) {
    generateAliasMaps();
  }
  # 如果需要双向检查别名
  if (bidirectional) {
    # 返回左右操作数可能包含别名的结果，或者右左操作数可能包含别名的结果
    return mayContainAliasImpl(lhs, rhs) || mayContainAliasImpl(rhs, lhs);
  } else {
    # 否则只返回左右操作数可能包含别名的结果
    return mayContainAliasImpl(lhs, rhs);
  }
}

bool SchemaInfo::mayContainAliasImpl(
    const c10::SchemaArgument& lhs,
    const c10::SchemaArgument& rhs) {
  // 获取 lhs 的别名类型集合（如果存在）
  std::optional<c10::AliasTypeSet> lhsContainedAliasTypeSet =
      schema_.getAliasTypeSetContainedTypes(schema_.mapTypeToAliasTypeSet(
          schema_.getCorrectList(lhs.type)[lhs.index].type()));
  // 获取 rhs 的别名类型集合
  std::optional<c10::AliasTypeSet> rhsAliasTypeSet =
      schema_.mapTypeToAliasTypeSet(
          schema_.getCorrectList(rhs.type)[rhs.index].type());
  // 检查两个别名类型集合是否可以别名
  bool types_can_alias =
      schema_.canAliasTypeSetsAlias(lhsContainedAliasTypeSet, rhsAliasTypeSet);
  // 返回是否可以别名并且 lhs 在 containerSet 中，并且 rhs 在 wildcardSet 中
  return types_can_alias && containerSet().count(lhs) &&
      wildcardSet().count(rhs);
}

void SchemaInfo::ensureConservativity(
    const std::unordered_set<at::Symbol>& duplicates,
    const std::vector<c10::Argument>& arguments_list,
    c10::SchemaArgType type) {
  // 遍历参数列表
  for (size_t i = 0; i < arguments_list.size(); i++) {
    // 如果当前参数具有别名信息
    if (arguments_list[i].alias_info()) {
      // 遍历该参数的 alias_info 的 afterSets
      for (const auto& set : arguments_list[i].alias_info()->afterSets()) {
        // 如果 duplicates 集合中包含当前 set
        if (duplicates.count(set)) {
          // 将 {type, i} 插入到 wildcard_set_ 中
          wildcard_set_.insert({type, i});
        }
      }
    }
  }
}
// 返回一组不确定性操作的函数签名列表
std::vector<c10::FunctionSchema> SchemaInfo::getNonDeterministicOps() {
  // 从JIT ir.cpp中复制的不确定性操作列表
  static const std::vector<std::string> nondeterministic_op_strings = {
      "aten::dropout(Tensor input, float p, bool train) -> Tensor",
      "aten::_fused_dropout(Tensor self, float p, Generator? generator) -> (Tensor, Tensor)",
      "aten::_standard_gamma(Tensor self, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
      "aten::multinomial(Tensor self, int num_samples, bool replacement, *, Generator? generator) -> Tensor",
      "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)",
      "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
      "aten::poisson(Tensor self, Generator? generator) -> Tensor",
      "aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor",
      "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint_like(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randperm(int n, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor"};

  // 创建一个存放不确定性操作函数签名的向量
  std::vector<c10::FunctionSchema> nondeterministic_ops;
  // 预留足够的空间以容纳所有的不确定性操作函数签名
  nondeterministic_ops.reserve(nondeterministic_op_strings.size());
  // 遍历每个不确定性操作的函数签名字符串，并解析为FunctionSchema对象后存入向量
  for (const std::string& signature : nondeterministic_op_strings) {
    nondeterministic_ops.emplace_back(torch::jit::parseSchema(signature));
  }

  // 返回包含所有不确定性操作函数签名的向量
  return nondeterministic_ops;
}
// 定义函数 `getTrainingOps`，返回类型为 `std::vector<SchemaSpecialCasePair>`
std::vector<SchemaSpecialCasePair> SchemaInfo::getTrainingOps() {
  // 这是一组操作及其对应的字符串集合的列表，其中布尔变量（"training"、"train" 或 "use_input_stats"）影响了字符串集合的可变性。
  static const std::vector<std::pair<std::string, std::unordered_set<std::string>>> training_op_pairs =
      {{"aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
        {"running_mean", "running_var"}},
       {"aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor",
        {"running_mean", "running_var"}},
       {"aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)",
        {"running_mean", "running_var"}},
       {"aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)",
        {"running_mean", "running_var"}},
       {"aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)",
        {"running_mean", "running_var"}},
       {"aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
        {"running_mean", "running_var"}},
       {"aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))",
        {"running_mean", "running_var"}},
       {"aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor",
        {"noise"}},
       {"aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)",
        {"noise"}},
       {"rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)",
        {"noise"}}};

  // 初始化一个空的 `std::vector<SchemaSpecialCasePair>`，用于存储处理后的操作对
  std::vector<SchemaSpecialCasePair> training_ops;
  // 预分配 `training_op_pairs` 大小的空间，以提高性能
  training_ops.reserve(training_op_pairs.size());
  // 遍历 `training_op_pairs` 中的每一个操作对
  for (const auto& signature : training_op_pairs) {
    training_ops.emplace_back(
        torch::jit::parseSchema(signature.first), signature.second);



    // 将新创建的操作（由签名解析而来）添加到训练操作列表中
    training_ops.emplace_back(
        torch::jit::parseSchema(signature.first), signature.second);



  }



  // 循环结束，所有操作已经添加到训练操作列表中，函数即将返回
  }
}

void SchemaInfo::initSchemaInfo() {
  // 如果已经初始化过，直接返回
  if (has_init_) {
    return;
  }
  // 标记已经初始化过
  has_init_ = true;

  // 用于存储重复出现的 alias set
  std::unordered_set<at::Symbol> duplicates;

  // Lambda 函数用于初始化 schema 参数信息
  auto init_schema_arguments = [this, &duplicates](
                                   const std::vector<c10::Argument>& arguments_list,
                                   c10::SchemaArgType type) {
    // 用于跟踪已经见过的 alias set
    std::unordered_set<at::Symbol> seen;
    // 遍历参数列表
    for (size_t i = 0; i < arguments_list.size(); i++) {
      const c10::Argument& argument = arguments_list[i];
      // 如果参数有别名信息
      if (argument.alias_info()) {
        // 如果别名信息表示是一个 wildcard after
        if (argument.alias_info()->isWildcardAfter()) {
          // 将其加入 wildcard set
          wildcard_set_.insert({type, i});
        } else {
          // 检查是否有多个参数共享同一个 alias set，以确保函数模式在调用 may_alias 和 may_contain_alias 时能够准确表示
          for (const auto& set : argument.alias_info()->afterSets()) {
            if (seen.count(set)) {
              // 如果已经见过，发出警告并将其加入重复集合
              TORCH_WARN(
                  set.toQualString(),
                  " appears twice in same argument list which will make aliasing checks more conservative.");
              duplicates.insert(set);
            } else {
              // 否则标记为已见过
              seen.insert(set);
            }
          }
        }
      }
      // 检查参数类型是否包含别名类型集合，并且集合不为空
      std::optional<c10::AliasTypeSet> contained_types =
          schema_.getAliasTypeSetContainedTypes(
              schema_.mapTypeToAliasTypeSet(argument.type()));
      if (contained_types && !contained_types->empty()) {
        // 将其加入 container set
        container_set_.insert({type, i});
      }
    }
  };

  // 初始化输入参数的 schema 信息
  init_schema_arguments(schema_.arguments(), c10::SchemaArgType::input);
  // 初始化输出参数的 schema 信息
  init_schema_arguments(schema_.returns(), c10::SchemaArgType::output);
  
  // 确保保守性，即检查是否有重复出现的 alias set
  ensureConservativity(
      duplicates, schema_.arguments(), c10::SchemaArgType::input);
  ensureConservativity(
      duplicates, schema_.returns(), c10::SchemaArgType::output);
}

// 获取 wildcard set 的方法
const std::unordered_set<c10::SchemaArgument>& SchemaInfo::wildcardSet() {
  // 确保 schema 信息已初始化
  initSchemaInfo();
  // 返回 wildcard set
  return wildcard_set_;
}

// 获取 container set 的方法
const std::unordered_set<c10::SchemaArgument>& SchemaInfo::containerSet() {
  // 确保 schema 信息已初始化
  initSchemaInfo();
  // 返回 container set
  return container_set_;
}

// 生成 alias maps 的方法
void SchemaInfo::generateAliasMaps() {
  // 确保 schema 信息已初始化
  initSchemaInfo();

  // 标记当前已经生成了 alias maps
  alias_maps_current_ = true;
  // 初始化 input_alias_map_
  input_alias_map_ = std::vector<std::unordered_set<size_t>>(
      schema_.arguments().size(), std::unordered_set<size_t>());
  // 初始化 output_alias_map_
  output_alias_map_ = std::vector<std::unordered_set<size_t>>(
      schema_.returns().size(), std::unordered_set<size_t>());

  // 填充 input_alias_map_
  for (size_t i = 0; i < schema_.arguments().size(); i++) {
    // 遍历从 i 到 schema_ 的参数数量之间的索引 j
    for (size_t j = i; j < schema_.arguments().size(); j++) {
      // 如果 i 和 j 相等，将 i 插入到 input_alias_map_ 中 i 对应的集合中
      if (i == j) {
        input_alias_map_[i].insert(i);
      } else if (
          // 如果 value_map_ 中存在 schema_.arguments()[i].name() 和 schema_.arguments()[j].name() 的值
          value_map_.count(schema_.arguments()[i].name()) &&
          value_map_.count(schema_.arguments()[j].name())) {
        // 如果 schema_.arguments()[i].name() 的值是 schema_.arguments()[j].name() 的别名
        if (value_map_[schema_.arguments()[i].name()].isAliasOf(
                value_map_[schema_.arguments()[j].name()])) {
          // 将 j 插入到 input_alias_map_ 中 i 对应的集合中，并将 i 插入到 input_alias_map_ 中 j 对应的集合中
          input_alias_map_[i].insert(j);
          input_alias_map_[j].insert(i);
          // 如果 wildcard_set_ 中存在 {c10::SchemaArgType::input, i} 的条目
          if (wildcard_set_.count({c10::SchemaArgType::input, i})) {
            // 向 wildcard_set_ 中添加一个新的 {c10::SchemaArgType::input, j} 的条目
            wildcard_set_.insert({c10::SchemaArgType::input, j});
          } else if (wildcard_set_.count({c10::SchemaArgType::input, j})) {
            // 如果 wildcard_set_ 中存在 {c10::SchemaArgType::input, j} 的条目，向其中添加一个新的 {c10::SchemaArgType::input, i} 的条目
            wildcard_set_.insert({c10::SchemaArgType::input, i});
          }
        }
      }
    }
    
    
    
    // 使用从 i 到 schema_ 的参数数量的索引 i 和 j 来填充 wildcard_set_，确定容器创建的通配符
    for (size_t i = 0; i < schema_.arguments().size(); i++) {
      for (size_t j = 0; j < schema_.arguments().size(); j++) {
        // 如果 i 和 j 不是别名关系，并且 value_map_ 中存在 schema_.arguments()[i].name() 和 schema_.arguments()[j].name() 的值
        if (!input_alias_map_[i].count(j) &&
            value_map_.count(schema_.arguments()[i].name()) &&
            value_map_.count(schema_.arguments()[j].name())) {
          // 获取 schema_.arguments()[i].name() 的子值集合 subValues
          c10::IValue::HashAliasedIValues subValues;
          value_map_[schema_.arguments()[i].name()].getSubValues(subValues);
          // 如果 subValues 中包含 value_map_[schema_.arguments()[j].name()] 的值
          if (subValues.count(value_map_[schema_.arguments()[j].name()])) {
            // 向 wildcard_set_ 中添加一个新的 {c10::SchemaArgType::input, j} 的条目
            wildcard_set_.insert({c10::SchemaArgType::input, j});
          }
        }
      }
    }
    
    
    
    // 使用索引 i 和 j 填充 output_alias_map_
    for (size_t i = 0; i < schema_.arguments().size(); i++) {
      for (size_t j = 0; j < schema_.returns().size(); j++) {
        // 如果 schema_.may_alias() 返回 true，表明参数和返回值可以是别名
        if (schema_.may_alias(
                {c10::SchemaArgType::input, i},
                {c10::SchemaArgType::output, j})) {
          // 如果 wildcard_set_ 中存在 {c10::SchemaArgType::input, i} 的条目，向其中添加一个新的 {c10::SchemaArgType::output, j} 的条目
          if (wildcard_set_.count({c10::SchemaArgType::input, i})) {
            wildcard_set_.insert({c10::SchemaArgType::output, j});
          }
          // 将 input_alias_map_ 中 i 对应的集合的所有元素插入到 output_alias_map_ 中 j 对应的集合中
          output_alias_map_[j].insert(
              input_alias_map_[i].begin(), input_alias_map_[i].end());
        }
      }
    }
}

} // namespace torch::utils
```