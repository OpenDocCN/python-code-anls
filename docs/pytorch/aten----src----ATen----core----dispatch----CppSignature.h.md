# `.\pytorch\aten\src\ATen\core\dispatch\CppSignature.h`

```py
#pragma once
// 引入头文件 <typeindex>，用于支持类型索引的操作
#include <typeindex>
// 引入 C10 核心的调度键集合头文件
#include <c10/core/DispatchKeySet.h>
// 引入 C10 宏定义头文件
#include <c10/macros/Macros.h>
// 引入 C10 的元编程工具头文件
#include <c10/util/Metaprogramming.h>
// 引入 C10 的类型工具头文件
#include <c10/util/Type.h>

// 进入 C10 命名空间
namespace c10 {
// 进入 impl 命名空间
namespace impl {

// CppSignature 类，用于在运行时保存 C++ 函数签名的 RTTI 信息，并允许比较或获取调试输出名称
class TORCH_API CppSignature final {
public:
    // 拷贝构造函数，默认实现
    CppSignature(const CppSignature&) = default;
    // 移动构造函数，默认实现，不抛出异常
    CppSignature(CppSignature&&) noexcept = default;
    // 拷贝赋值运算符，默认实现
    CppSignature& operator=(const CppSignature&) = default;
    // 移动赋值运算符，默认实现，不抛出异常
    CppSignature& operator=(CppSignature&&) noexcept = default;

    // 静态成员函数模板，用于创建 CppSignature 对象
    template<class FuncType>
    static CppSignature make() {
        // 使用 c10::remove_DispatchKeySet_arg_from_func 模板元函数，将函数类型 FuncType 规范化为 plain function type
        using decayed_function_type = typename c10::remove_DispatchKeySet_arg_from_func<std::decay_t<FuncType>>::func_type;

        // 返回由规范化后的函数类型构造的 CppSignature 对象
        return CppSignature(std::type_index(typeid(decayed_function_type)));
    }

    // 返回当前 CppSignature 对象的名称
    std::string name() const {
        // 使用 c10::demangle 函数解析签名的名称并返回
        return c10::demangle(signature_.name());
    }

    // 友元函数重载运算符==，用于比较两个 CppSignature 对象是否相等
    friend bool operator==(const CppSignature& lhs, const CppSignature& rhs) {
        // 如果两个 CppSignature 对象的 signature_ 相等，则它们相等
        if (lhs.signature_ == rhs.signature_) {
            return true;
        }
        // 如果两个对象的 signature_ 的名称相同（考虑不使用 RTLD_GLOBAL 的情况下可能出现的情况），则它们也相等
        if (0 == strcmp(lhs.signature_.name(), rhs.signature_.name())) {
            return true;
        }

        // 否则，它们不相等
        return false;
    }

private:
    // 显式构造函数，用于构造指定类型索引的 CppSignature 对象
    explicit CppSignature(std::type_index signature): signature_(std::move(signature)) {}

    // 成员变量，存储类型索引信息
    std::type_index signature_;
};

// 友元函数重载运算符!=，用于比较两个 CppSignature 对象是否不相等
inline bool operator!=(const CppSignature& lhs, const CppSignature& rhs) {
    // 利用已重载的==运算符实现!=运算符
    return !(lhs == rhs);
}

} // namespace impl
} // namespace c10
```