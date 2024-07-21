# `.\pytorch\torch\csrc\jit\serialization\callstack_debug_info_serialization.cpp`

```
namespace torch::jit {

namespace {
const int64_t kInvalidSourceRangeTag = -1;
} // namespace

// 序列化内联调用堆栈
c10::IValue InlinedCallStackSerializer::serialize(
    const InlinedCallStackPtr& cs_ptr,
    const SourceRangeTagMap& source_range_tags) {
  // 如果传入的内联调用堆栈指针为空，则返回空的 c10::IValue
  if (!cs_ptr) {
    return c10::IValue();
  }
  // 检查是否已经对当前内联调用堆栈指针进行了序列化，如果是，则直接返回之前序列化的结果
  auto cs_it = serialized_inlined_callstack_.find(cs_ptr);
  if (cs_it != serialized_inlined_callstack_.end()) {
    return cs_it->second;
  }
  // 准备存储序列化后的内联调用堆栈的元素
  std::vector<c10::IValue> elements;
  elements.reserve(4);
  // 序列化模块实例信息并添加到元素中
  elements.emplace_back(
      serialize_module_instance_info(cs_ptr->module_instance()));
  int64_t source_range_tag{kInvalidSourceRangeTag};
  // 获取内联调用堆栈的源代码范围，并查找对应的源代码范围标签
  const SourceRange& sr = cs_ptr->source_range().findSourceRangeThatGenerated()
      ? cs_ptr->source_range().findSourceRangeThatGenerated().value()
      : cs_ptr->source_range();
  auto sr_it = source_range_tags.find(sr);
  // 如果找到了对应的源代码范围标签，则使用它作为源代码范围标签
  if (sr_it != source_range_tags.end()) {
    source_range_tag = sr_it->second;
  }
  // 将源代码范围标签添加到元素中
  elements.emplace_back(source_range_tag);
  // 如果存在被调用的内联调用堆栈，则递归序列化该内联调用堆栈并添加到元素中
  if (cs_ptr->callee()) {
    elements.emplace_back(
        serialize(cs_ptr->callee().value(), source_range_tags));
  } else {
    elements.emplace_back();
  }
  // 添加函数名到元素中，如果函数名为空，则使用默认字符串 "FunctionName_UNKNOWN"
  auto fn_name = cs_ptr->function_name();
  if (!fn_name.empty()) {
    elements.emplace_back(fn_name);
  } else {
    elements.emplace_back("FunctionName_UNKNOWN");
  }
  // 创建包含所有元素的 c10::IValue 元组
  c10::IValue serialized_cs = c10::ivalue::Tuple::create(elements);
  // 将序列化后的内联调用堆栈存储到映射中，以备将来使用
  serialized_inlined_callstack_[cs_ptr] = serialized_cs;
  // 返回序列化后的内联调用堆栈
  return serialized_cs;
}

// 序列化模块实例信息
c10::IValue InlinedCallStackSerializer::serialize_module_instance_info(
    const std::optional<ModuleInstanceInfo>& m) {
  // 如果模块实例信息为空，则返回空的 c10::IValue
  if (!m) {
    return c10::IValue();
  }
  // 获取模块实例信息的值
  const auto& m_val = m.value();
  std::string module_type_name = m_val.class_type()->name()->qualifiedName();
  auto module_instance_name = m_val.instance_name();
  // 如果类类型存在，则获取其完全限定名
  if (m_val.class_type()) {
    module_type_name = m_val.class_type()->name()->qualifiedName();
  }
  // 构建模块类型名和实例名的组合作为键值
  auto key_val = module_type_name + module_instance_name;
  // 查找是否已经序列化过该模块实例信息，如果是，则直接返回之前序列化的结果
  auto m_inst_it = serialized_module_instance_info_.find(key_val);
  if (m_inst_it != serialized_module_instance_info_.end()) {
    // 返回与指定键关联的模块实例信息
    return m_inst_it->second;
  }
  // 模块实例信息被序列化为元组，包含类型名和实例名
  serialized_module_instance_info_[key_val] =
      c10::ivalue::Tuple::create({module_type_name, module_instance_name});
  // 返回序列化后的模块实例信息
  return serialized_module_instance_info_[key_val];
}

std::vector<char> CallStackDebugInfoPickler::pickle(
    const std::unordered_map<int64_t, DebugInfoTuple>& callstack_ptrs,
    const SourceRangeTagMap& source_range_tags) {
  std::vector<c10::IValue> ivalues;
  // 遍历调用栈指针的无序映射
  for (const auto& it : callstack_ptrs) {
    int64_t debug_handle = it.first;
    std::vector<c10::IValue> elements;
    /*
     * 调试句柄和调试信息（源代码范围 + 内联调用栈）
     * 被序列化为一个包含3个元素的元组
     * {debug_handle, source_range_tag, serialized_callstack}
     */
    elements.reserve(4);
    elements.emplace_back(debug_handle);
    int64_t source_range_tag{kInvalidSourceRangeTag};
    const auto& source_range =
        std::get<kDebugInfoTupleSourceRangeIndex>(it.second);
    const SourceRange& sr = source_range.findSourceRangeThatGenerated()
        ? source_range.findSourceRangeThatGenerated().value()
        : source_range;
    auto sr_it = source_range_tags.find(sr);
    // 查找源代码范围的标签值
    if (sr_it != source_range_tags.end()) {
      source_range_tag = sr_it->second;
    }
    elements.emplace_back(source_range_tag);
    elements.emplace_back(std::get<kDebugInfoTupleNodeNameIndex>(it.second));
    const auto& inlined_cs_ptr =
        std::get<kDebugInfoTupleInlinedCSIndex>(it.second);
    // 序列化内联调用栈指针，使用给定的源代码范围标签映射
    elements.emplace_back(css_.serialize(inlined_cs_ptr, source_range_tags));
    ivalues.emplace_back(c10::ivalue::Tuple::create(elements));
  }
  std::vector<at::Tensor> table;
  c10::IValue ivalue = c10::ivalue::Tuple::create(std::move(ivalues));
  // 使用 Torch JIT 序列化 IValue，返回序列化结果
  auto result = jit::pickle(ivalue, &table);
  TORCH_CHECK(table.empty(), "Expected 0 tensors to be written");
  return result;
}

InlinedCallStackPtr InlinedCallStackDeserializer::deserialize(
    const c10::IValue& iv,
    const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
    const std::shared_ptr<CompilationUnit>& cu) {
  if (iv.isNone()) {
    return c10::intrusive_ptr<InlinedCallStack>();
  }
  auto tup = iv.toTuple();
  auto it = cached_inlined_callstacks_.find(tup);
  // 检查是否已经缓存了该元组的内联调用栈指针
  if (it != cached_inlined_callstacks_.end()) {
    return it->second;
  }

  const auto& tup_elems = tup->elements();
  TORCH_INTERNAL_ASSERT(tup_elems.size() == 4);
  // {IValue(module_instance_info), source_range_tag, IValue(InlinedCallStack),
  // function name}
  auto module_instance_info =
      deserialize_module_instance_info(tup_elems[0], cu);
  int64_t source_range_tag = tup_elems[1].toInt();
  auto source_range_it = source_range_map.find(source_range_tag);
  // 检查源代码范围标签是否存在于反序列化的源代码范围映射中
  TORCH_CHECK(
      source_range_tag == kInvalidSourceRangeTag ||
          source_range_it != source_range_map.end(),
      "Source range tag must exist in deserialized source range map."
      " Not found source range tag:",
      source_range_tag);
  SourceRange source_range;
  if (source_range_tag != kInvalidSourceRangeTag) {
    // 如果源代码范围标签有效，从映射中获取源代码范围
    source_range = source_range_it->second;
  }
    // 从源范围映射中获取与当前迭代器指向的键对应的值（源范围）
    source_range = source_range_it->second;
  }
  // 反序列化元组的第三个元素，生成对应的函数调用对象
  auto callee = deserialize(tup_elems[2], source_range_map, cu);
  // 获取元组的第四个元素作为函数名
  auto function_name = tup_elems[3].toStringRef();
  // 声明一个内联调用栈指针
  InlinedCallStackPtr cs_ptr;
  // 如果成功反序列化了函数调用对象
  if (callee) {
    // 使用调用对象、空指针、源范围、模块实例信息和函数名创建内联调用栈对象
    cs_ptr = c10::make_intrusive<InlinedCallStack>(
        callee, nullptr, source_range, module_instance_info, function_name);
  } else {
    // 否则，使用空指针、源范围、模块实例信息和函数名创建内联调用栈对象
    cs_ptr = c10::make_intrusive<InlinedCallStack>(
        nullptr, source_range, module_instance_info, function_name);
  }
  // 将新创建的内联调用栈对象指针存储到缓存中，使用元组作为键
  cached_inlined_callstacks_[tup] = cs_ptr;
  // 返回移动构造函数的调用结果
  // 这里通过返回对象本身来避免引用计数更新，因为对象被移动而不是复制
  return cs_ptr;
}

// deserialize_module_instance_info 方法实现
std::optional<ModuleInstanceInfo> InlinedCallStackDeserializer::
    deserialize_module_instance_info(
        const c10::IValue& iv,
        const std::shared_ptr<CompilationUnit>& cu) {
  // 如果输入值 iv 是 None，则返回空 optional
  if (iv.isNone()) {
    return c10::nullopt;
  }
  // 将输入值 iv 转换为元组
  auto tup = iv.toTuple();
  // 在缓存中查找是否已经存在对应的模块实例信息
  auto it = cached_module_instance_info_.find(tup);
  // 如果找到，则直接返回缓存中的结果
  if (it != cached_module_instance_info_.end()) {
    return it->second;
  }
  // 获取元组的元素引用
  const auto& tup_elems = iv.toTupleRef().elements();
  // 检查元组元素数量是否为 2
  TORCH_CHECK(tup_elems.size() == 2);
  // 提取类型名和实例名
  std::string type_name = tup_elems[0].toStringRef();
  std::string instance_name = tup_elems[1].toStringRef();
  // type_name 可能为空字符串 ""
  // 在这种情况下，type_ptr 应为 nullptr
  auto type_ptr = cu->get_class(type_name);
  // 如果找不到类型信息，则处理特定情况
  if (!type_ptr) {
    // 可能丢失类型信息，例如在降低的后端中原始类类型不相关
    // 然而，为了将操作与其原始模块相关联，保存了类型名和实例名
    // 在这种情况下，当模块被降低后端吸收时，用类型名扩充实例名而不是丢弃它
    auto last_dot_position = type_name.find_last_of('.');
    size_t substring_pos{0};
    if (last_dot_position != std::string::npos) {
      substring_pos = last_dot_position + 1;
    }
    type_name = type_name.substr(substring_pos);
    instance_name = instance_name + "(" + type_name + ")";
  }
  // 缓存模块实例信息并返回
  cached_module_instance_info_[tup] =
      ModuleInstanceInfo(type_ptr, instance_name);
  return cached_module_instance_info_[tup];
}

// unpickle 方法实现
ska::flat_hash_map<int64_t, DebugInfoTuple> CallStackDebugInfoUnpickler::
    unpickle(
        at::DataPtr&& data,
        size_t size,
        const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
        const std::shared_ptr<CompilationUnit>& cu) {
  // 使用 jit::unpickle 反序列化数据并获取 IValue
  auto ival = jit::unpickle(
      reinterpret_cast<const char*>(data.get()),
      size,
      nullptr,
      {},
      c10::parseType);
  // 创建用于存储调用堆栈调试信息的哈希映射
  ska::flat_hash_map<int64_t, DebugInfoTuple> callstack_ptrs;
  // 获取 IValue 的元组引用
  const auto& ivalues = ival.toTupleRef().elements();
  // 遍历每个元组值
  for (auto& val : ivalues) {
    // 获取元组的元素引用
    const auto& tup_elems = val.toTupleRef().elements();
    // 检查元组的元素数量是否为 4
    TORCH_CHECK(
        tup_elems.size() == 4,
        "Pickled map must have four elements: "
        "debug_handle, source_range_tag, op name, IValue(inlined_call_stack)");
    // 提取调试句柄和源代码范围标签
    int64_t debug_handle = tup_elems[0].toInt();
    int64_t source_range_tag = tup_elems[1].toInt();
    const std::string& node_name = tup_elems[2].toStringRef();
    // 在源代码范围映射中查找对应的源代码范围
    auto source_range_it = source_range_map.find(source_range_tag);
    // 检查是否找到源代码范围
    TORCH_CHECK(
        source_range_it != source_range_map.end(),
        "Source range tag must exist in deserialized source range map.");
    auto source_range = source_range_it->second;
    // 检查调试句柄是否唯一
    TORCH_CHECK(
        callstack_ptrs.count(debug_handle) == 0,
        "Debug handles should be unique.");


这段代码主要实现了两个方法，分别是 `deserialize_module_instance_info` 和 `unpickle`，用于反序列化模块实例信息和调用堆栈调试信息。
    # 将调试处理器（debug_handle）映射到调用栈指针的元组
    callstack_ptrs[debug_handle] = std::make_tuple(
        # 将源范围（source_range）、节点名称（node_name）、以及反序列化后的调用栈数据（csds_.deserialize）组成元组
        source_range,
        node_name,
        csds_.deserialize(tup_elems[3], source_range_map, cu));
  }
  # 返回完整的调用栈指针映射
  return callstack_ptrs;
}

} // namespace torch::jit


注释：


}  // 关闭 torch::jit 命名空间
```