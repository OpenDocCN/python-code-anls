# `.\pytorch\aten\src\ATen\core\List.cpp`

```
// 定义 c10::detail 命名空间
namespace c10::detail {
    // 比较两个 ListImpl 对象是否相等的运算符重载函数
    bool operator==(const ListImpl& lhs, const ListImpl& rhs) {
        // 比较两个 ListImpl 对象的 elementType 是否相等，并且比较它们的 list 大小是否相同
        return *lhs.elementType == *rhs.elementType &&
            lhs.list.size() == rhs.list.size() &&
            // 使用 _fastEqualsForContainer 函数比较 lhs 和 rhs 的 list 元素是否一一相等
            std::equal(
                lhs.list.cbegin(),
                lhs.list.cend(),
                rhs.list.cbegin(),
                _fastEqualsForContainer);
    }

    // ListImpl 类的构造函数，接受一个 list 和一个 elementType 参数
    ListImpl::ListImpl(list_type list_, TypePtr elementType_)
      : list(std::move(list_))
      , elementType(std::move(elementType_)) {}
} // namespace c10::detail
```