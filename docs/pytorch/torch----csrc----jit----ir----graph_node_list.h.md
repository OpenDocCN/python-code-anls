# `.\pytorch\torch\csrc\jit\ir\graph_node_list.h`

```
    return old;
  }
  generic_graph_node_list_iterator& operator--() {
    AT_ASSERT(cur);
    cur = cur->next_in_graph[!d];
    return *this;
  }
  generic_graph_node_list_iterator operator--(int) {
    generic_graph_node_list_iterator old = *this;
    --(*this);
    return old;
  }
  bool operator==(const generic_graph_node_list_iterator& rhs) const {
    return cur == rhs.cur && d == rhs.d;
  }
  bool operator!=(const generic_graph_node_list_iterator& rhs) const {
    return !(*this == rhs);
  }

private:
  T* cur;        // 当前迭代器所指向的节点
  int d;         // 迭代器方向，0表示正向，1表示反向
};

template <typename T>
struct generic_graph_node_list {
  generic_graph_node_list() {
    head.next_in_graph[kNextDirection] = &tail;
    head.next_in_graph[kPrevDirection] = &tail;
    tail.next_in_graph[kNextDirection] = &head;
    tail.next_in_graph[kPrevDirection] = &head;
  }

  void destroy(T* t) {
    AT_ASSERT(t);
    t->next_in_graph[kNextDirection]->next_in_graph[kPrevDirection] =
        t->next_in_graph[kPrevDirection];
    t->next_in_graph[kPrevDirection]->next_in_graph[kNextDirection] =
        t->next_in_graph[kNextDirection];
    t->next_in_graph[kNextDirection] = nullptr;
    t->next_in_graph[kPrevDirection] = nullptr;
  }

  void push_back(T* t) {
    AT_ASSERT(t);
    t->next_in_graph[kNextDirection] = &tail;
    t->next_in_graph[kPrevDirection] = tail.next_in_graph[kPrevDirection];
    tail.next_in_graph[kPrevDirection]->next_in_graph[kNextDirection] = t;
    tail.next_in_graph[kPrevDirection] = t;
  }

  void insert_after(T* l, T* t) {
    AT_ASSERT(l);
    AT_ASSERT(t);
    AT_ASSERT(l != &tail);
    AT_ASSERT(l != &head);
    t->next_in_graph[kPrevDirection] = l;
    t->next_in_graph[kNextDirection] = l->next_in_graph[kNextDirection];
    l->next_in_graph[kNextDirection]->next_in_graph[kPrevDirection] = t;
    l->next_in_graph[kNextDirection] = t;
  }

  void insert_before(T* r, T* t) {
    AT_ASSERT(r);
    AT_ASSERT(t);
    AT_ASSERT(r != &tail);
    AT_ASSERT(r != &head);
    t->next_in_graph[kNextDirection] = r;
    t->next_in_graph[kPrevDirection] = r->next_in_graph[kPrevDirection];
    r->next_in_graph[kPrevDirection]->next_in_graph[kNextDirection] = t;
    r->next_in_graph[kPrevDirection] = t;
  }

  bool empty() const {
    return head.next_in_graph[kNextDirection] == &tail;
  }

  generic_graph_node_list_iterator<T> begin() {
    return generic_graph_node_list_iterator<T>(head.next_in_graph[kNextDirection], kNextDirection);
  }

  generic_graph_node_list_iterator<T> end() {
    return generic_graph_node_list_iterator<T>(&tail, kNextDirection);
  }

private:
  T head;      // 头哨兵节点
  T tail;      // 尾哨兵节点
};



// Intrusive doubly linked list with iterators for Graph's Node lists.
// The head and tail sentinel nodes ensure a circular link.
// Operations include insertion, destruction, and iteration over nodes.
// Supports both forward and reverse iteration through custom iterators.

// Iterator:
// - Points to the current node in the list.
// - Allows dereferencing (* and -> operators) to access the node.
// - Supports increment (++) and decrement (--) operators for traversal.
// - Equality operators (== and !=) for comparison with another iterator.

// List:
// - Contains head and tail sentinel nodes to manage list boundaries.
// - Methods include insertion at the back, destruction of nodes, and checking for emptiness.
// - Provides begin() and end() methods for iterator access to list elements.

// Note: This structure assumes the type T supports specific operations and fields for list management.
  // 返回迭代器指向的旧对象
  generic_graph_node_list_iterator operator--(int) {
    // 复制当前迭代器
    generic_graph_node_list_iterator old = *this;
    // 调用前置递减操作符
    --(*this);
    // 返回旧的迭代器对象
    return old;
  }

  // 删除当前节点而不使迭代器失效
  // 使用不同的命名以避免在箭头操作符或点操作符中静默调用错误函数
  // 调用后，迭代器将指向前一个条目
  void destroyCurrent() {
    // 将当前节点指针保存在n中
    T* n = cur;
    // 移动当前节点指针到下一个节点
    cur = cur->next_in_graph[reverseDir()];
    // 销毁保存的当前节点n
    n->destroy();
  }

  // 返回反向迭代器
  generic_graph_node_list_iterator reverse() {
    // 使用当前节点和反向方向创建并返回反向迭代器
    return generic_graph_node_list_iterator(cur, reverseDir());
  }

 private:
  // 根据当前方向返回相反的方向
  int reverseDir() {
    // 如果当前方向是正向，则返回反向，反之亦然
    return d == kNextDirection ? kPrevDirection : kNextDirection;
  }
  T* cur;  // 当前节点指针
  int d;   // 方向，0代表正向，1代表反向，参见next_in_graph
};
// 结构体模板 `generic_graph_node_list` 的定义开始
template <typename T>
struct generic_graph_node_list {
  // 定义迭代器类型
  using iterator = generic_graph_node_list_iterator<T>;
  using const_iterator = generic_graph_node_list_iterator<const T>;

  // 返回正向迭代器的起始位置
  generic_graph_node_list_iterator<T> begin() {
    return generic_graph_node_list_iterator<T>(head->next_in_graph[d], d);
  }

  // 返回常量正向迭代器的起始位置
  generic_graph_node_list_iterator<const T> begin() const {
    return generic_graph_node_list_iterator<const T>(head->next_in_graph[d], d);
  }

  // 返回正向迭代器的结束位置
  generic_graph_node_list_iterator<T> end() {
    return generic_graph_node_list_iterator<T>(head->next_in_graph[!d], d);
  }

  // 返回常量正向迭代器的结束位置
  generic_graph_node_list_iterator<const T> end() const {
    return generic_graph_node_list_iterator<const T>(head->next_in_graph[!d], d);
  }

  // 返回反向迭代器的起始位置
  generic_graph_node_list_iterator<T> rbegin() {
    return reverse().begin();
  }

  // 返回常量反向迭代器的起始位置
  generic_graph_node_list_iterator<const T> rbegin() const {
    return reverse().begin();
  }

  // 返回反向迭代器的结束位置
  generic_graph_node_list_iterator<T> rend() {
    return reverse().end();
  }

  // 返回常量反向迭代器的结束位置
  generic_graph_node_list_iterator<const T> rend() const {
    return reverse().end();
  }

  // 返回反转的节点列表
  generic_graph_node_list reverse() {
    return generic_graph_node_list(head->next_in_graph[!d], !d);
  }

  // 返回常量反转的节点列表
  const generic_graph_node_list reverse() const {
    return generic_graph_node_list(head->next_in_graph[!d], !d);
  }

  // 返回头部节点的指针
  T* front() {
    return head->next_in_graph[d];
  }

  // 返回头部节点的常量指针
  const T* front() const {
    return head->next_in_graph[d];
  }

  // 返回尾部节点的指针
  T* back() {
    return head->next_in_graph[!d];
  }

  // 返回尾部节点的常量指针
  const T* back() const {
    return head->next_in_graph[!d];
  }

  // 结构体构造函数，初始化头部节点和维度标识
  generic_graph_node_list(T* head, int d) : head(head), d(d) {}

 private:
  T* head; // 头部节点，同时也是哨兵节点
  // d 标识节点的维度，用于选择正向或反向的迭代器
  int d;
};

// 重载迭代器相等比较运算符
template <typename T>
static inline bool operator==(
    generic_graph_node_list_iterator<T> a,
    generic_graph_node_list_iterator<T> b) {
  return *a == *b;
}

// 重载迭代器不等比较运算符
template <typename T>
static inline bool operator!=(
    generic_graph_node_list_iterator<T> a,
    generic_graph_node_list_iterator<T> b) {
  return *a != *b;
}

} // namespace jit
} // namespace torch

// 定义 std 命名空间下的迭代器特性模板
namespace std {

template <typename T>
struct iterator_traits<torch::jit::generic_graph_node_list_iterator<T>> {
  using difference_type = int64_t;
  using value_type = T*;
  using pointer = T**;
  using reference = T*&;
  using iterator_category = bidirectional_iterator_tag;
};

} // namespace std
```