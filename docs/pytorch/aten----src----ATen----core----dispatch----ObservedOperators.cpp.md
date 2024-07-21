# `.\pytorch\aten\src\ATen\core\dispatch\ObservedOperators.cpp`

```
/* static */
// 返回未被观察到的运算符列表的引用
std::unordered_set<std::string>& ObservedOperators::getUnobservedOperatorList() {
  // 不应该被观察的运算符的名称集合
  static std::unordered_set<std::string> not_observed_ops = {
    "aten::size",
    "aten::is_leaf",
    "aten::output_nr",
    "aten::_version",
    "aten::is_complex",
    "profiler::_record_function_enter",
    "profiler::_record_function_enter_new",
    "profiler::_record_function_exit",
  };
  // 返回静态的不应该被观察的运算符集合
  return not_observed_ops;
}

/* static */
// 检查给定运算符名称是否被观察
bool ObservedOperators::isObserved(const OperatorName& name) {
  // 返回是否运算符名称不在未被观察到的运算符列表中
  return !ObservedOperators::getUnobservedOperatorList().count(name.name);
}

} // namespace c10
```