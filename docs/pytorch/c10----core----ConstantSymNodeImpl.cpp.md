# `.\pytorch\c10\core\ConstantSymNodeImpl.cpp`

```
// 包含头文件 <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/ConstantSymNodeImpl.h>

// 命名空间 c10 中的定义
namespace c10 {

// 用于支持左操作数是常量符号节点，右操作数是嵌套整数符号节点的情况。
// 这种情况发生在我们对嵌套整数和普通整数进行二元操作，并且整数提升为常量符号节点时。
// 如果将来想要支持更多组合，可能需要实现某种形式的多重分发。
#define DEFINE_BINARY_OP(OP, ROP)                                        \
  template <typename T>                                                  \
  // ConstantSymNodeImpl<T> 类模板的成员函数 OP 的定义，接受 SymNode 类型的参数 other
  c10::SymNode ConstantSymNodeImpl<T>::OP(const c10::SymNode& other) {   \
    // 内部断言：确保参数 other 是嵌套整数符号节点
    TORCH_INTERNAL_ASSERT(other->is_nested_int());                       \
    // 调用 other 对应的成员函数 ROP，传入当前对象 ConstantSymNodeImpl<T> 的共享指针的拷贝
    return other->ROP(                                                   \
        c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim_copy(this)); \
  }

// 定义各种二元操作符对应的宏，展开为具体的模板函数定义
DEFINE_BINARY_OP(eq, eq)
DEFINE_BINARY_OP(ne, ne)
DEFINE_BINARY_OP(ge, le)
DEFINE_BINARY_OP(le, ge)
DEFINE_BINARY_OP(lt, gt)
DEFINE_BINARY_OP(gt, lt)
DEFINE_BINARY_OP(mul, mul)

// 取消之前定义的宏 DEFINE_BINARY_OP
#undef DEFINE_BINARY_OP

// 实例化 ConstantSymNodeImpl 类模板的 bool 和 int64_t 类型
template class ConstantSymNodeImpl<bool>;
template class ConstantSymNodeImpl<int64_t>;

} // namespace c10
```