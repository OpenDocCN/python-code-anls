# `.\pytorch\aten\src\ATen\native\CompositeRandomAccessorCommon.h`

```py
// <utility> 头文件包含了一些通用的实用工具函数和类模板，如 std::move 和 std::tuple。
#pragma once

namespace at::native {

namespace {

// operator_brackets_proxy 类被用作 CompositeRandomAccessor 中的 operator[] 的代理。
// 有些迭代器返回的引用可能会变成无效，operator_brackets_proxy 试图通过使 accessor[n] 等价于 *(accessor + n) 来解决这个问题。
template <typename Accessor>
class operator_brackets_proxy {
  using reference = typename std::iterator_traits<Accessor>::reference;
  using value_type = typename std::iterator_traits<Accessor>::value_type;

public:
  C10_HOST_DEVICE
  operator_brackets_proxy(Accessor const& accessor)
    : accessor(accessor)
  {}

  C10_HOST_DEVICE
  operator reference() {
    return *accessor;
  }

  C10_HOST_DEVICE
  reference operator*() {
    return *accessor;
  }

  C10_HOST_DEVICE
  operator_brackets_proxy& operator=(value_type const& val) {
    *accessor = val;
    return *this;
  }

private:
  Accessor accessor;
};

}

// references_holder 类被用作 CompositeRandomAccessor 中的 references 类型的代理。
// 在 CompositeRandomAccessor 中假定 References = tuple<Types&...>，Values = tuple<Types...>，尽管它们可以是任何类型，
// 只要 References 能够转换为 Values 即可。
// 如果计划在 STL 中使用它，例如，需要定义 'swap' 和 'get' 方法（即 std::get）。
template <typename Values, typename References>
class references_holder {
public:
  using values = Values;
  using references = References;

  C10_HOST_DEVICE
  references_holder(references refs)
    : refs{std::move(refs)}
  {}

  C10_HOST_DEVICE
  operator references() {
    return refs;
  }

  C10_HOST_DEVICE
  operator values() {
    return refs;
  }

  C10_HOST_DEVICE
  references_holder& operator=(values vals) {
    refs = vals;
    return *this;
  }

  C10_HOST_DEVICE
  references& data() {
    return refs;
  }

protected:
  references refs;
};

// CompositeRandomAccessor 类本质上是两个随机访问迭代器的简化版本的随机访问迭代器。
// TupleInfo 应包含一个变参类型 `tuple`，和一个方法 `tie`，它从参数列表中构造一个引用的 tuple。
template <typename KeyAccessor, typename ValueAccessor, typename TupleInfo>
class CompositeRandomAccessor {
  using self_type = CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfo>;

  using key_accessor_value_type =
    typename std::iterator_traits<KeyAccessor>::value_type;
  using value_accessor_value_type =
    typename std::iterator_traits<ValueAccessor>::value_type;
  using key_accessor_reference_type =
    typename std::iterator_traits<KeyAccessor>::reference;
  using value_accessor_reference_type =
    typename std::iterator_traits<ValueAccessor>::reference;

  using composite_value_type = typename TupleInfo::template tuple<
    key_accessor_value_type,
    value_accessor_value_type
  >;

  // 省略部分代码...
  // 定义一个模板别名 `value_type`，用于存储元组中值访问器的类型
  value_accessor_value_type>;
  // 使用模板元信息 `TupleInfo` 定义一个模板别名 `composite_reference`，表示键访问器和值访问器的元组类型
  using composite_reference = typename TupleInfo::template tuple<
    // 元组的第一个类型是键访问器的引用类型
    key_accessor_reference_type,
    // 元组的第二个类型是值访问器的引用类型
    value_accessor_reference_type>;
public:
  // 定义值类型为 composite_value_type
  using value_type = composite_value_type;
  // 定义引用类型为 references_holder<composite_value_type, composite_reference>
  using reference = references_holder<composite_value_type, composite_reference>;
  // 注意，CompositeRandomAccessor 不保存键值对在特定数据结构中，
  // 因此没有指向 (key, value) 的指针类型定义。这里使用 KeyAccessor 的指针类型。
  using pointer = typename std::iterator_traits<KeyAccessor>::pointer;
  // 定义差值类型为 KeyAccessor 的差值类型
  using difference_type = typename std::iterator_traits<KeyAccessor>::difference_type;
  // 定义迭代器类型为 std::random_access_iterator_tag
  using iterator_category = std::random_access_iterator_tag;

  // 默认构造函数，在主机和设备端都可用
  C10_HOST_DEVICE
  CompositeRandomAccessor() = default;

  // 构造函数，初始化 keys 和 values 成员
  C10_HOST_DEVICE
  CompositeRandomAccessor(KeyAccessor keys, ValueAccessor values)
    : keys(keys), values(values)
  {}

  // Pointer-like operations {
  // 解引用操作符，返回 references_holder 对象
  C10_HOST_DEVICE
  reference operator*() const {
    return TupleInfo::tie(*keys, *values);
  }

  // 箭头操作符应返回指针类型。
  // 由于 CompositeRandomAccessor 不持有对键值对的指针，
  // 因此这里只返回 keys 的指针。
  C10_HOST_DEVICE
  auto* operator->() const {
    return keys.operator->();
  }

  // 下标操作符重载，返回 operator_brackets_proxy 对象
  C10_HOST_DEVICE
  reference operator[](difference_type idx) {
    return operator_brackets_proxy<self_type>(
      CompositeRandomAccessor(keys + idx, values + idx)
    );
  }
  // }

  // 前缀/后缀递增/递减操作 {
  // 前缀递增
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator++() {
    ++keys;
    ++values;
    return *this;
  }

  // 后缀递增
  C10_HOST_DEVICE
  CompositeRandomAccessor operator++(int) {
    CompositeRandomAccessor copy(*this);
    ++*this;
    return copy;
  }

  // 前缀递减
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator--() {
    --keys;
    --values;
    return *this;
  }

  // 后缀递减
  C10_HOST_DEVICE
  CompositeRandomAccessor operator--(int) {
    CompositeRandomAccessor copy(*this);
    --*this;
    return copy;
  }
  // }

  // 算术操作 {
  // 加法赋值
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator+=(difference_type offset) {
    keys += offset;
    values += offset;
    return *this;
  }

  // 加法
  C10_HOST_DEVICE
  CompositeRandomAccessor operator+(difference_type offset) const {
    return CompositeRandomAccessor(keys + offset, values + offset);
  }

  // 友元函数，加法
  C10_HOST_DEVICE
  friend CompositeRandomAccessor operator+(
    difference_type offset,
    const CompositeRandomAccessor& accessor
  ) {
    return accessor + offset;
  }

  // 减法赋值
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator-=(difference_type offset) {
    keys -= offset;
    values -= offset;
    return *this;
  }

  // 减法
  C10_HOST_DEVICE
  CompositeRandomAccessor operator-(difference_type offset) const {
    return CompositeRandomAccessor(keys - offset, values - offset);
  }

  // 差值
  C10_HOST_DEVICE
  difference_type operator-(const CompositeRandomAccessor& other) const {
    return keys - other.keys;
  }
  // }

  // 比较操作符 {
  // 等于比较
  C10_HOST_DEVICE
  bool operator==(const CompositeRandomAccessor& other) const {
  // 返回当前对象的键是否与另一个对象的键相等的比较结果
  bool operator==(const CompositeRandomAccessor& other) const {
    return keys == other.keys;
  }

  // 返回当前对象的键是否与另一个对象的键不相等的比较结果
  C10_HOST_DEVICE
  bool operator!=(const CompositeRandomAccessor& other) const {
    return keys != other.keys;
  }

  // 返回当前对象的键是否小于另一个对象的键的比较结果
  C10_HOST_DEVICE
  bool operator<(const CompositeRandomAccessor& other) const {
    return keys < other.keys;
  }

  // 返回当前对象的键是否小于或等于另一个对象的键的比较结果
  C10_HOST_DEVICE
  bool operator<=(const CompositeRandomAccessor& other) const {
    return keys <= other.keys;
  }

  // 返回当前对象的键是否大于另一个对象的键的比较结果
  C10_HOST_DEVICE
  bool operator>(const CompositeRandomAccessor& other) const {
    return keys > other.keys;
  }

  // 返回当前对象的键是否大于或等于另一个对象的键的比较结果
  C10_HOST_DEVICE
  bool operator>=(const CompositeRandomAccessor& other) const {
    return keys >= other.keys;
  }
protected:
  KeyAccessor keys;  // 声明一个 KeyAccessor 类型的变量 keys，用于访问键值对中的键
  ValueAccessor values;  // 声明一个 ValueAccessor 类型的变量 values，用于访问键值对中的值
};  // 结束类定义

} // namespace at::native  // 结束 at::native 命名空间
```