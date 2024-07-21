# `.\pytorch\aten\src\ATen\core\Dict.cpp`

```py
// 定义命名空间 c10::detail 下的函数，用于比较两个 DictImpl 对象是否相等
bool operator==(const DictImpl& lhs, const DictImpl& rhs) {
  // 进行快速的属性检查，包括键和值的类型是否相同，以及字典大小是否相等
  bool isEqualFastChecks =
      *lhs.elementTypes.keyType == *rhs.elementTypes.keyType &&
      *lhs.elementTypes.valueType == *rhs.elementTypes.valueType &&
      lhs.dict.size() == rhs.dict.size();
  if (!isEqualFastChecks) {
    return false;
  }

  // 字典的相等性判断不应关心顺序
  // 遍历左边字典中的每个键值对
  for (const auto& pr : lhs.dict) {
    // 在右边字典中查找当前键对应的迭代器
    auto it = rhs.dict.find(pr.first);
    // 如果右边字典中找不到相同的键，则字典不相等
    if (it == rhs.dict.cend()) {
      return false;
    }
    // 执行容器内部的快速相等性检查，详见 [container equality] 注释
    if (!_fastEqualsForContainer(it->second, pr.second)) {
      return false;
    }
  }

  // 若经过以上所有检查都没有问题，则认为两个 DictImpl 对象相等
  return true;
}
```