# `.\pytorch\aten\src\ATen\core\union_type.cpp`

```py
// 在 c10 命名空间中定义 OptionalType 的静态创建方法
OptionalTypePtr OptionalType::create(const TypePtr& contained) {
  // 使用传入的类型创建 OptionalTypePtr 对象并返回
  return OptionalTypePtr(new OptionalType(contained));
}

// 定义 OptionalType 类的静态方法，返回 TensorType 的 OptionalTypePtr
TypePtr OptionalType::ofTensor() {
  // 创建静态的 OptionalTypePtr 对象，包含 TensorType
  static auto value = OptionalType::create(TensorType::get());
  // 返回包含 TensorType 的 OptionalTypePtr
  return value;
}

// 定义 ListType 类的静态方法，返回包含 OptionalType 的 ListTypePtr
ListTypePtr ListType::ofOptionalTensors() {
  // 创建静态的 ListTypePtr 对象，包含 OptionalType.ofTensor() 的结果
  static auto value = ListType::create(OptionalType::ofTensor());
  // 返回包含 OptionalType.ofTensor() 的 ListTypePtr
  return value;
}

// 匿名命名空间内定义的函数，从 to_subtract 中减去 from 的类型集合
std::optional<TypePtr> subtractTypeSetFrom(std::vector<TypePtr>& to_subtract, ArrayRef<TypePtr> from) {
  // 存储没有被减去的类型集合
  std::vector<TypePtr> types;

  // 判断类型 lhs 是否需要被减去，即 lhs 或其父类型是否在 to_subtract 中
  auto should_subtract = [&](const TypePtr& lhs) -> bool {
    // 遍历 to_subtract 中的每个类型 rhs，看 lhs 是否是 rhs 的子类型
    return std::any_of(to_subtract.begin(), to_subtract.end(),
                       [&](const TypePtr& rhs) {
                         return lhs->isSubtypeOf(*rhs);
                       });
  };

  // 将不需要被减去的类型复制到 types 中
  std::copy_if(from.begin(), from.end(),
               std::back_inserter(types),
               [&](const TypePtr& t) {
                 return !should_subtract(t);
               });

  // 如果 types 为空，则返回空的 optional
  if (types.empty()) {
    return c10::nullopt;
  } else if (types.size() == 1) {
    // 如果 types 只有一个元素，则返回该元素作为 optional 的值
    return types[0];
  } else {
    // 否则创建并返回一个 UnionType，包含 types 中的所有类型
    return UnionType::create(std::move(types));
  }
}

// 递归函数，用于展开 UnionType 中的嵌套 OptionalType 和 UnionType
void flattenUnion(const TypePtr& type, std::vector<TypePtr>* to_fill) {
  // 如果 type 是 UnionType，则递归展开其中的所有类型
  if (auto* union_type = type->castRaw<UnionType>()) {
    for (const auto& inner : union_type->containedTypes()) {
      flattenUnion(inner, to_fill);
    }
  } else if (auto* opt_type = type->castRaw<OptionalType>()) {
    // 如果 type 是 OptionalType，则递归展开其内部类型，并添加 NoneType
    const auto& inner = opt_type->getElementType();
    flattenUnion(inner, to_fill);
    to_fill->emplace_back(NoneType::get());
  } else if (type->kind() == NumberType::Kind) {
    // 如果 type 是 NumberType，则添加 IntType、FloatType 和 ComplexType
    to_fill->emplace_back(IntType::get());
    to_fill->emplace_back(FloatType::get());
    to_fill->emplace_back(ComplexType::get());
  } else {
    // 其他情况直接添加 type 到 to_fill 中
    to_fill->emplace_back(type);
  }
}
// 过滤掉重复的子类型，同时尝试将相同类型的Union进行合并
void filterDuplicateSubtypes(std::vector<TypePtr>* types) {
  // 如果types为空，则直接返回
  if (types->empty()) {
    return;
  }
  // 定义一个lambda函数get_supertype，用于获取两个类型的最顶层的共同父类型
  auto get_supertype = [](const TypePtr& t1, const TypePtr& t2) -> std::optional<TypePtr> {
    // 避免嵌套的Optional类型，不过早统一到Optional类型可能会阻止其他类型的合并
    if ((t1->isSubtypeOf(*NoneType::get()) && !t2->isSubtypeOf(*NoneType::get()))
        || (!t1->isSubtypeOf(*NoneType::get()) && t2->isSubtypeOf(*NoneType::get()))) {
      return c10::nullopt;
    } else {
      // 调用unifyTypes函数，尝试将t1和t2统一到一个类型
      return unifyTypes(t1, t2, /*default_to_union=*/false);
    }
  };

  // 合并类型并删除所有重复项。从右到左遍历向量，尝试将当前元素（i）与向量中每个元素（j）（直到“新”向量尾部end）合并。
  // 如果能够合并types[i]和types[j]的类型，则将types[j]替换为统一后的类型，并将types[i]移动到end位置，然后减少end的值。
  size_t end_idx = types->size()-1;
  for (size_t i = types->size()-1; i > 0; --i) {
    for (size_t j = std::min(i-1, end_idx); ; --j) {
      std::optional<TypePtr> unified;
      unified = get_supertype((*types)[i], (*types)[j]);
      if (unified) {
        (*types)[j] = *unified;
        (*types)[i] = (*types)[end_idx];
        --end_idx;
        break;
      }
      // 避免得到j = 0时的无限循环，导致MAX_INT
      if (j == 0) {
        break;
      }
    }
  }
  // 截断向量的尾部，使得end是实际的最后一个元素位置
  types->erase(types->begin() + static_cast<std::ptrdiff_t>(end_idx) + 1, types->end());
}

// 对types向量进行排序，以便将来轻松比较两个UnionType对象的相等性
static void sortUnion(std::vector<TypePtr>* types) {
  std::sort(types->begin(), types->end(),
          [](const TypePtr& a, const TypePtr& b) -> bool {
            // 首先按类型种类排序，如果类型种类相同，则按字符串排序
            if (a->kind() != b->kind()) {
              return a->kind() < b->kind();
            }
            return a->str() < b->str();
          });
}

// 将reference向量中的所有类型展开，并将结果填充到to_fill向量中，然后对to_fill向量进行标准化：去除重复子类型，并排序
void standardizeVectorForUnion(std::vector<TypePtr>& reference, std::vector<TypePtr>* to_fill) {
  // 遍历reference中的每个类型，将其展开后加入to_fill向量中
  for (const auto& type : reference) {
    flattenUnion(type, to_fill);
  }
  // 过滤掉to_fill中的重复子类型
  filterDuplicateSubtypes(to_fill);
  // 对to_fill中的Union类型进行排序
  sortUnion(to_fill);
}
// 标准化给定的向量以用于 Union 操作，确保不为空指针
void standardizeVectorForUnion(std::vector<TypePtr>* to_flatten) {
  // 内部断言，确保传入的 to_flatten 不为空指针
  TORCH_INTERNAL_ASSERT(to_flatten, "`standardizeVectorForUnion` was ",
                        "passed a `nullptr`");
  // 创建一个空的向量 to_fill
  std::vector<TypePtr> to_fill;
  // 调用另一个重载的 standardizeVectorForUnion 函数，填充 to_flatten 并将结果移动到 to_flatten 中
  standardizeVectorForUnion(*to_flatten, &to_fill);
  // 将 to_fill 的内容移动到 to_flatten 中
  *to_flatten = std::move(to_fill);
}

// OptionalType 类的构造函数，表示一个包含可选类型的 UnionType
OptionalType::OptionalType(const TypePtr& contained)
                           : UnionType({contained, NoneType::get()}, TypeKind::OptionalType) {
  // 检查是否为 NumberType
  bool is_numbertype = false;
  // 如果 contained 是 UnionType，则检查它是否可以包含 NumberType
  if (auto as_union = contained->cast<UnionType>()) {
    is_numbertype = as_union->containedTypes().size() == 3 &&
                    as_union->canHoldType(*NumberType::get());
  }
  // 如果 UnionType 包含的类型数量为 2
  if (UnionType::containedTypes().size() == 2) {
    // 如果第一个类型不是 NoneType，则将其作为 contained_
    contained_ = UnionType::containedTypes()[0]->kind()!= NoneType::Kind
                 ? UnionType::containedTypes()[0]
                 : UnionType::containedTypes()[1];
  } else if (contained == NumberType::get() || is_numbertype) {
    // 如果 contained 是 NumberType 或者 is_numbertype 为 true，则将 contained_ 设为 NumberType
    contained_ = NumberType::get();
    // 清空 types_ 并添加 NumberType 和 NoneType
    types_.clear();
    types_.emplace_back(NumberType::get());
    types_.emplace_back(NoneType::get());
  } else {
    // 否则，创建一个 to_subtract 包含 NoneType，并从 types_ 中减去这个类型集合
    std::vector<TypePtr> to_subtract{NoneType::get()};
    auto without_none = subtractTypeSetFrom(to_subtract, types_);
    // 创建一个 UnionType 包含 without_none 中的内容，并赋值给 contained_
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    contained_ = UnionType::create({*without_none});
  }
  // 检查 contained_ 是否有自由变量
  has_free_variables_ = contained_->hasFreeVariables();
}

// UnionType 类的构造函数，表示一个 Union 类型
UnionType::UnionType(std::vector<TypePtr> reference, TypeKind kind) : SharedType(kind) {
  // 内部断言，确保 reference 不为空
  TORCH_INTERNAL_ASSERT(!reference.empty(), "Cannot create an empty Union");

  // 调用 standardizeVectorForUnion 函数，将 reference 标准化并存储在 types_ 中
  standardizeVectorForUnion(reference, &types_);

  // 如果 types_ 中只有一个类型，则发出断言错误
  if (types_.size() == 1) {
    std::stringstream msg;
    msg << "After type unification was performed, the Union with the "
        << "original types {";
    // 构建详细的错误消息，列出原始类型和转换后的单一类型
    for (const auto i : c10::irange(reference.size())) {
      msg << reference[i]->repr_str();
      if (i > 0) {
        msg << ",";
      }
      msg << " ";
    }
    msg << "} has the single type " << types_[0]->repr_str()
         << ". Use the common supertype instead of creating a Union"
         << "type";
    // 断言失败，并输出详细错误消息
    TORCH_INTERNAL_ASSERT(false, msg.str());
  }

  // 初始化标志变量
  can_hold_none_ = false;
  has_free_variables_ = false;

  // 遍历 types_，检查是否可以包含 NoneType 和是否有自由变量
  for (const TypePtr& type : types_) {
    if (type->kind() == NoneType::Kind) {
      can_hold_none_ = true;
    }
    if (type->hasFreeVariables()) {
      has_free_variables_ = true;
    }
  }

}

// 创建一个 UnionType 对象，包含给定的类型引用
UnionTypePtr UnionType::create(std::vector<TypePtr> reference) {
  // 创建 UnionTypePtr 智能指针，指向新创建的 UnionType 对象
  UnionTypePtr union_type(new UnionType(std::move(reference)));

  // 一些特殊情况下的逻辑处理，将来会在后续 PR 中删除
  bool int_found = false;
  bool float_found = false;
  bool complex_found = false;
  bool nonetype_found = false;

  // 更新标志位，检查是否包含特定类型
  auto update_is_opt_flags = [&](const TypePtr& t) {
    if (t == IntType::get()) {
      int_found = true;
    }
    // 省略其他类型的检查...
  };

  // 更新标志位，检查引用中是否包含特定类型
  for (const auto& type : reference) {
    update_is_opt_flags(type);
  }
  
  // 返回新创建的 UnionType 对象
  return union_type;
}
  } else if (t == FloatType::get()) {
    // 如果类型 t 是 FloatType，则将 float_found 置为 true
    float_found  = true;
  } else if (t == ComplexType::get()) {
    // 如果类型 t 是 ComplexType，则将 complex_found 置为 true
    complex_found = true;
  } else if (t == NoneType::get()) {
    // 如果类型 t 是 NoneType，则将 nonetype_found 置为 true
    nonetype_found = true;
  }
};

// 遍历 union_type 中的每种类型，并更新相应的标志
for (const auto& t : union_type->containedTypes()) {
  update_is_opt_flags(t);
}

// 检查是否找到了所有数值类型
bool numbertype_found = int_found && float_found && complex_found;

// 如果存在 NoneType，则根据条件返回不同的 OptionalType
if (nonetype_found) {
  if (union_type->containedTypes().size() == 4 && numbertype_found) {
    // 如果 union_type 包含 4 种类型且包含所有数值类型，则返回 OptionalType 包装的 NumberType
    return OptionalType::create(NumberType::get());
  }
  if (union_type->containedTypes().size() == 2) {
    // 如果 union_type 包含 2 种类型，则返回 OptionalType 包装的第一个非 NoneType 类型
    auto not_none = union_type->containedTypes()[0] != NoneType::get()
                    ? union_type->containedTypes()[0]
                    : union_type->containedTypes()[1];
    return OptionalType::create(not_none);
  }
}

// 如果没有找到 NoneType 或条件不满足，则直接返回 union_type
return union_type;
}

std::optional<TypePtr> UnionType::subtractTypeSet(std::vector<TypePtr>& to_subtract) const {
  // 调用另一个函数 `subtractTypeSetFrom`，从当前联合类型中减去给定类型集合，并返回结果
  return subtractTypeSetFrom(to_subtract, containedTypes());
}

std::optional<TypePtr> UnionType::toOptional() const {
  // 如果当前联合类型不能容纳 NoneType，则返回空的 optional
  if (!canHoldType(*NoneType::get())) {
      return c10::nullopt;
  }

  // 复制当前联合类型中的所有类型到 `copied_types`
  std::vector<TypePtr> copied_types = this->containedTypes().vec();

  // 使用 `UnionType::create` 创建一个可能的 optional 类型
  auto maybe_opt = UnionType::create(std::move(copied_types));

  // 如果创建的 optional 类型的类型为 UnionType::Kind，则返回空的 optional
  if (maybe_opt->kind() == UnionType::Kind) {
    return c10::nullopt;
  } else {
    return maybe_opt;
  }
}

bool UnionType::equals(const Type& rhs) const {
  // 如果 rhs 能转换为 UnionType
  if (auto union_rhs = rhs.cast<UnionType>()) {
    // 检查两个 UnionType 是否包含相同数量的类型
    if (union_rhs->containedTypes().size() != this->containedTypes().size()) {
      return false;
    }
    // 检查 `this->containedTypes()` 中的所有类型是否也存在于 `union_rhs->containedTypes()` 中
    return std::all_of(this->containedTypes().begin(), this->containedTypes().end(),
                       [&](TypePtr lhs_type) {
                         return std::any_of(union_rhs->containedTypes().begin(),
                                            union_rhs->containedTypes().end(),
                                            [&](const TypePtr& rhs_type) {
                                              return *lhs_type == *rhs_type;
                                            });
                       });
  } else if (auto optional_rhs = rhs.cast<OptionalType>()) {
    // 如果 rhs 能转换为 OptionalType
    if (optional_rhs->getElementType() == NumberType::get()) {
      // 检查当前联合类型是否包含 4 种类型，并且能容纳 NoneType 和 NumberType
      return this->containedTypes().size() == 4
             && this->can_hold_none_
             && this->canHoldType(*NumberType::get());
    }
    // 将当前联合类型转换为 optional，然后比较与 rhs 是否相等
    auto optional_lhs = this->toOptional();
    return optional_lhs && *optional_rhs == *((optional_lhs.value())->expect<OptionalType>());
  } else if (rhs.kind() == NumberType::Kind) {
    // 如果 rhs 的类型是 NumberType，则检查当前联合类型是否包含 3 种类型，并且能容纳 NumberType
    return this->containedTypes().size() == 3 && canHoldType(*NumberType::get());
  } else {
    // 其他情况下返回 false
    return false;
  }
}

bool UnionType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  // 创建一个 rhs 类型的指针向量
  std::vector<const Type*> rhs_types;
  // 如果 rhs 能转换为 UnionType
  if (const auto union_rhs = rhs.cast<UnionType>()) {
    // 快速路径：如果当前联合类型和 rhs 的联合类型相等，则返回 true
    if (this->containedTypes() == rhs.containedTypes()) {
      return true;
    }
    // 将 rhs 的类型添加到 rhs_types 中
    for (const auto& typePtr: rhs.containedTypes()) {
      rhs_types.push_back(typePtr.get());
    }
  } else if (const auto optional_rhs = rhs.cast<OptionalType>()) {
    // 如果 rhs 能转换为 OptionalType，则添加 NoneType 到 rhs_types 中
    rhs_types.push_back(NoneType::get().get());
    // 如果 optional_rhs 的元素类型是 NumberType
    if (optional_rhs->getElementType() == NumberType::get()) {
      // 添加 IntType、FloatType 和 ComplexType 到 rhs_types 中
      std::array<const Type*, 3> number_types{IntType::get().get(), FloatType::get().get(), ComplexType::get().get()};
      rhs_types.insert(rhs_types.end(), number_types.begin(), number_types.end());
    } else {
      // 否则添加 optional_rhs 的元素类型到 rhs_types 中
      rhs_types.push_back(optional_rhs->getElementType().get());
    }
  } else if (const auto number_rhs = rhs.cast<NumberType>()) {
    // 如果 rhs 能转换为 NumberType，则检查当前联合类型是否包含 3 种类型，并且能容纳 NumberType
    return this->containedTypes().size() == 3 && canHoldType(*NumberType::get());
  } else {
    // 其他情况下返回 false
    return false;
  }
    // 创建一个包含指向 Type 类型常量指针的 std::array，包括 IntType、FloatType 和 ComplexType 的地址
    std::array<const Type*, 3> number_types{IntType::get().get(), FloatType::get().get(), ComplexType::get().get()};
    
    // 将 number_types 数组中的元素添加到 rhs_types 向量的末尾
    rhs_types.insert(rhs_types.end(), number_types.begin(), number_types.end());
  } else {
    // 如果 rhs 不是 std::array，则将 rhs 添加到 rhs_types 向量的末尾
    rhs_types.push_back(&rhs);
  }
  
  // 使用 std::all_of 算法遍历 this->containedTypes() 的每个元素，使用 Lambda 表达式作为谓词
  return std::all_of(this->containedTypes().begin(), this->containedTypes().end(),
                     [&](const TypePtr& lhs_type) -> bool {
                      // 对 rhs_types 向量中的每个元素应用 std::any_of 算法，使用 Lambda 表达式作为谓词
                      return std::any_of(rhs_types.begin(),
                                         rhs_types.end(),
                                         [&](const Type* rhs_type) -> bool {
                                           // 调用 lhs_type->isSubtypeOfExt(*rhs_type, why_not) 检查 lhs_type 是否是 rhs_type 的子类型
                                           return lhs_type->isSubtypeOfExt(*rhs_type, why_not);
                                         });
  });
}

std::string UnionType::unionStr(const TypePrinter& printer, bool is_annotation_str)
    const {
  std::stringstream ss;

  // 检查当前 UnionType 是否能包含 NumberType
  bool can_hold_numbertype = this->canHoldType(*NumberType::get());

  // 定义一组数值类型，包括 IntType、FloatType 和 ComplexType
  std::vector<TypePtr> number_types{IntType::get(), FloatType::get(), ComplexType::get()};

  // Lambda 函数用于检查是否为数值类型
  auto is_numbertype = [&](const TypePtr& lhs) {
    for (const auto& rhs : number_types) {
      if (*lhs == *rhs) {
        return true;
      }
    }
    return false;
  };

  // 根据 is_annotation_str 确定使用的分隔符
  std::string open_delimeter = is_annotation_str ? "[" : "(";
  std::string close_delimeter = is_annotation_str ? "]" : ")";

  // 构建 UnionType 字符串的开头
  ss << "Union" + open_delimeter;
  bool printed = false;
  for (size_t i = 0; i < types_.size(); ++i) {
    // 如果当前 UnionType 不能包含数值类型或当前类型不是数值类型，则添加到字符串中
    if (!can_hold_numbertype || !is_numbertype(types_[i])) {
      if (i > 0) {
        ss << ", ";
        printed = true;
      }
      // 根据 is_annotation_str 决定是调用 annotation_str 还是 str 方法
      if (is_annotation_str) {
        ss << this->containedTypes()[i]->annotation_str(printer);
      } else {
        ss << this->containedTypes()[i]->str();
      }
    }
  }
  // 如果当前 UnionType 可以包含数值类型，则添加数值类型到字符串中
  if (can_hold_numbertype) {
    if (printed) {
      ss << ", ";
    }
    if (is_annotation_str) {
      ss << NumberType::get()->annotation_str(printer);
    } else {
      ss << NumberType::get()->str();
    }
  }
  // 完成 UnionType 字符串的构建，添加闭合分隔符
  ss << close_delimeter;
  return ss.str();
}

std::string UnionType::str() const {
  // 返回 UnionType 的字符串表示，不带注解
  return this->unionStr(nullptr, /*is_annotation_str=*/false);
}

std::string UnionType::annotation_str_impl(const TypePrinter& printer) const {
  // 返回 UnionType 的注解字符串表示
  return this->unionStr(printer, /*is_annotation_str=*/true);
}

bool UnionType::canHoldType(const Type& type) const {
  // 如果类型是 NumberType，则检查 UnionType 是否能同时包含 IntType、FloatType 和 ComplexType
  if (&type == NumberType::get().get()) {
    return canHoldType(*IntType::get())
           && canHoldType(*FloatType::get())
           && canHoldType(*ComplexType::get());
  } else {
    // 否则检查 UnionType 是否能包含给定类型的任何一个子类型
    return std::any_of(this->containedTypes().begin(), this->containedTypes().end(),
                    [&](const TypePtr& inner) {
                      return type.isSubtypeOf(*inner);
                    });
  }
}

bool OptionalType::equals(const Type& rhs) const {
  // 比较 OptionalType 是否等于另一个 OptionalType 或者与 UnionType 转换后的 OptionalType 相等
  if (auto union_rhs = rhs.cast<UnionType>()) {
    auto optional_rhs = union_rhs->toOptional();
    // `**optional_rhs` = `*` 获取 `std::optional<TypePtr>` 的值，然后再 `*` 解引用指针
    return optional_rhs && *this == **optional_rhs;
  } else if (auto optional_rhs = rhs.cast<OptionalType>()) {
    // 比较两个 OptionalType 的元素类型是否相等
    return *this->getElementType() == *optional_rhs->getElementType();
  } else {
    return false;
  }
}

bool OptionalType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  // 检查 OptionalType 是否是给定类型的子类型，考虑细节并输出原因
  if (auto optional_rhs = rhs.castRaw<OptionalType>()) {
    return getElementType()->isSubtypeOfExt(*optional_rhs->getElementType(), why_not);
  } else if (auto union_rhs = rhs.castRaw<UnionType>()) {
    // 如果 UnionType 不能包含 NoneType，则返回 false，并输出原因
    if (!union_rhs->canHoldType(*NoneType::get())) {
      if (why_not) {
        *why_not << rhs.repr_str() << " cannot hold None";
      }
      return false;
      // 否则继续判断 OptionalType 是否是 UnionType 的子类型

  // 如果是 UnionType，则检查是否能包含 NoneType
  if (!union_rhs->canHoldType(*NoneType::get())) {
    // 如果不能包含 NoneType，则输出原因并返回 false
    if (why_not) {
      *why_not << rhs.repr_str() << " cannot hold None";
    }
    return false;
  }
  // 否则继续检查 OptionalType 是否是 UnionType 的子类型
    } else if (!union_rhs->canHoldType(*this->getElementType())) {
      // 检查 union_rhs 是否能容纳 this 对象的元素类型
      if (why_not) {
        // 如果 why_not 参数存在，将错误信息写入其中，说明为什么 union_rhs 不能容纳 this 的元素类型
        *why_not << rhs.repr_str() << " cannot hold " << this->getElementType();
      }
      // 返回 false，表示 union_rhs 不能容纳 this 的元素类型
      return false;
    } else {
      // 如果 union_rhs 能容纳 this 的元素类型，返回 true
      return true;
    }
  } else {
    // 如果不是同一类型的对象，调用 Type 类的 isSubtypeOfExt 方法检查是否是子类型关系
    // NOLINTNEXTLINE(bugprone-parent-virtual-call) 禁止对父类虚函数的调用，防止潜在的 bugprone
    return Type::isSubtypeOfExt(rhs, why_not);
  }
}

} // namespace 10
```